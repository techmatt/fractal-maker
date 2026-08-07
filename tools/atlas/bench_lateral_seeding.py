#!/usr/bin/env python
r"""Differential bench: `lateral_to_sibling`'s period SWEEP vs the atom-domain SEEDING.

WHY A REPLAY AND NOT A LIVE RUN. The two paths must be compared **on the same inputs**.
A live A/B run is not a comparison: the walk's frontier, its RNG stream and the regions
it reaches all move with the cost of the operator, so the two arms would not even see the
same parent views. So the parent views are replayed out of a recorded run's
`maneuvers.jsonl`, and both arms are driven from an identically-seeded RNG, which makes
the probe seeds (radius + angle) byte-identical between arms.

WHY IT IS A DIFFERENTIAL TEST AND NOT A FROZEN LITERAL. The sweep is a reference
implementation that still exists — `identify_nucleus` with no `periods=` — so the seeded
path can be checked against the thing it replaces rather than against numbers copied out
of a previous report. Where both arms find a nucleus they must find the SAME one; the
seeded set is a subset of the sweep's, so the one legitimate difference is availability
(the sweep can find a period the atom-domain ranking never nominated).

  uv run python tools/atlas/bench_lateral_seeding.py \
      --log data/discovery/maneuver_shakedown/maneuvers.jsonl --limit 120
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from tools import run_record            # noqa: E402  (segments-aware run-record layer)

import minibrot_maneuvers as mnv    # noqa: E402
import paths                        # noqa: E402

DEFAULT_LOG = ROOT / "data" / "discovery" / "maneuver_shakedown" / "maneuvers.jsonl"


def load_cases(log: Path, limit: int | None) -> list[dict]:
    """Recorded lateral calls, deduped on `(batch, parent_node_id, op, k)`.

    The log is append-only and a kill replays a batch, so it is a SUPERSET of the
    checkpointed counters — quoting it undeduped double-counts."""
    seen, cases = set(), []
    for r in run_record.require_rows(log):  # segments-aware; absence stays LOUD
        if r.get("op") != "lateral_to_sibling":
            continue
        key = (r.get("batch"), r.get("parent_node_id"), r.get("op"), str(r.get("k")))
        if key in seen:
            continue
        seen.add(key)
        cases.append(r)
    return cases[:limit] if limit else cases


def run_case(row: dict, seeded: bool, low: int = 0) -> dict:
    """One replayed lateral call. The RNG is re-seeded per (case, arm) from the recorded
    node id, so every arm draws the identical radii/angles."""
    degree = mnv.degree_of(row.get("partition", "mandelbrot")) or 2
    view = dict(node_id=row.get("parent_node_id"), cx=row["parent_cx"],
                cy=row["parent_cy"], fw=row["parent_fw"],
                depth=row.get("parent_depth") or 0)
    rng = np.random.default_rng(abs(int(row.get("parent_node_id") or 0)) + 1)
    t0 = time.time()
    m = mnv.lateral_to_sibling(view, rng, degree=degree, k=row.get("k"),
                               parent_rec=row["_parent_rec"], seed_periods=seeded,
                               low_sweep=low)
    return dict(s=time.time() - t0, available=bool(m.available), reason=m.reason,
                atom_id=m.atom_id, period=m.period, fw=m.fw,
                newton_solves=m.newton_solves)


def summarize(name: str, rs: list[dict]) -> dict:
    ts = sorted(r["s"] for r in rs)
    return dict(arm=name, n=len(rs), total_s=round(sum(ts), 3),
                median_ms=round(1000 * st.median(ts), 1),
                mean_ms=round(1000 * sum(ts) / len(ts), 1),
                p90_ms=round(1000 * ts[int(0.9 * (len(ts) - 1))], 1),
                max_s=round(ts[-1], 3),
                available=sum(1 for r in rs if r["available"]),
                solves=sum(r["newton_solves"] for r in rs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--low", type=int, nargs="+", default=[0, mnv.LAT_LOW_SWEEP],
                    help="exact-sweep floors to bench the seeded arm at "
                         "(0 = pure atom-domain seeding, no floor)")
    ap.add_argument("--out", type=Path,
                    default=paths.scratch("maneuvers", "bench_lateral_seeding.json"))
    a = ap.parse_args()

    cases = load_cases(a.log, a.limit)
    print(f"[bench] {len(cases)} deduped lateral calls from {a.log}")

    # The parent atom is resolved ONCE per case and handed to both arms: it is the same
    # snap in either arm, and charging it twice would dilute the delta being measured.
    prepared, no_parent = [], 0
    for row in cases:
        degree = mnv.degree_of(row.get("partition", "mandelbrot")) or 2
        view = dict(node_id=row.get("parent_node_id"), cx=row["parent_cx"],
                    cy=row["parent_cy"], fw=row["parent_fw"],
                    depth=row.get("parent_depth") or 0)
        snap = mnv.snap_to_nucleus(view, None, degree=degree)
        if not snap.available:
            no_parent += 1
            continue
        row["_parent_rec"] = dict(id=snap.atom_id, cx=snap.cx, cy=snap.cy,
                                  period=snap.period, window_scale=snap.window_scale,
                                  degree=degree)
        prepared.append(row)
    print(f"[bench] {len(prepared)} replayable ({no_parent} had no parent atom today)")

    ref = [run_case(r, seeded=False) for r in prepared]
    arms = {"sweep": ref}
    for low in a.low:
        arms[f"seeded_low{low}"] = [run_case(r, seeded=True, low=low) for r in prepared]

    ref_s = summarize("sweep", ref)
    res = dict(log=str(a.log), n_cases=len(prepared), no_parent_atom=no_parent,
               arms=[ref_s], comparisons=[])
    for name, got in arms.items():
        if name == "sweep":
            continue
        s = summarize(name, got)
        res["arms"].append(s)
        disagree, both = [], 0
        for row, o, n in zip(prepared, ref, got):
            if o["available"] and n["available"]:
                both += 1
                if o["atom_id"] != n["atom_id"]:
                    disagree.append(dict(
                        parent_node_id=row.get("parent_node_id"),
                        partition=row.get("partition"),
                        sweep=dict(atom_id=o["atom_id"], period=o["period"]),
                        seeded=dict(atom_id=n["atom_id"], period=n["period"])))
        lost = [dict(parent_node_id=row.get("parent_node_id"),
                     partition=row.get("partition"), sweep_period=o["period"],
                     seeded_reason=n["reason"])
                for row, o, n in zip(prepared, ref, got)
                if o["available"] and not n["available"]]
        res["comparisons"].append(dict(
            arm=name, both_available=both, nucleus_disagreements=disagree,
            availability_lost=lost,
            availability_gained=sum(1 for o, n in zip(ref, got)
                                    if n["available"] and not o["available"]),
            speedup=round(ref_s["total_s"] / s["total_s"], 2) if s["total_s"] else None))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    for row in res["arms"]:
        print(f"  {row['arm']:>14}: n={row['n']} total={row['total_s']}s "
              f"median={row['median_ms']}ms mean={row['mean_ms']}ms "
              f"p90={row['p90_ms']}ms max={row['max_s']}s "
              f"available={row['available']} newton_solves={row['solves']}")
    for c in res["comparisons"]:
        print(f"  {c['arm']:>14}: speedup x{c['speedup']} both={c['both_available']} "
              f"disagree={len(c['nucleus_disagreements'])} "
              f"lost={len(c['availability_lost'])} gained={c['availability_gained']}")
    print(f"[bench] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
