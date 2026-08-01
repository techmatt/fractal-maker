#!/usr/bin/env python
r"""Differential bench: is `lateral_to_sibling` SUBSUMED by `neighborhood_expand`?

THE QUESTION. Operator 3 enumerates the same disc operator 2 samples, so if it reproduced
operator 2's picks at small `m` the walk would be paying for two operators and getting one.
This bench answers it on replayed inputs rather than by reading the two functions.

WHY A REPLAY, AND WHY IT IS FAIR HERE. Same reasoning as `bench_lateral_seeding.py`: a live
A/B is not a comparison, because the walk's frontier moves with the cost of the operator.
Parent views come out of a recorded run's `maneuvers.jsonl` and BOTH arms are driven from an
identically-seeded RNG. That is exact here, not approximate — the two operators share
`_draw_probe_seed`, so with one seed they draw byte-identical (radius, angle) pairs and
therefore probe byte-identical seed points. Any difference in the answer is a difference in
the FILTERS, which is precisely what is being measured.

WHAT SUBSUMPTION WOULD MEAN. `neighborhood_expand(m=1, probes=LAT_PROBES)` walks the same
seeds in the same order and returns the first survivor, exactly as lateral does. The two
filters differ in one place: lateral's comparable-scale window is SYMMETRIC (`|log10 ratio|
<= LAT_SCALE_TOL_DECADES`) while operator 3's is one-sided (`<= NBH_SCALE_UP_DECADES`,
unbounded below). So the pre-registered prediction is:

  * identical picks wherever the first surviving candidate is inside lateral's window;
  * neighborhood available where lateral is `scale_mismatch`, on candidates below it.

A disagreement is NOT a defect. Both contracts promise "*a* nearby nucleus", not "*that*
one" — the same identity-drift reading `minibrot_maneuvers.md` §2.6 records for the lateral
head, and the dedup key is the nucleus' canonical key, so identity is not reproduced by
anything downstream either way.

  uv run python tools/atlas/bench_neighborhood_subsumption.py \
      --log data/discovery/<run>/maneuvers.jsonl --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

import minibrot_maneuvers as mnv    # noqa: E402
import paths                        # noqa: E402


def load_cases(log: Path, limit: int | None) -> list[dict]:
    """Recorded lateral calls, deduped on `(batch, parent_node_id)`.

    The log is append-only and a kill replays a batch, so it is a SUPERSET of the
    checkpointed counters — quoting it undeduped double-counts (the same correction
    `bench_lateral_seeding.load_cases` carries)."""
    seen, cases = set(), []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("op") != "lateral_to_sibling":
            continue
        # `maneuvers.jsonl` is not homogeneous: the quota's passed-over records carry an
        # `op` too but are frontier-node rows, with no parent view on them. They are not
        # operator calls and cannot be replayed — filter on the field the replay NEEDS
        # rather than on the absence of a marker, so a future row shape cannot slip in.
        if r.get("parent_cx") is None or r.get("parent_fw") is None:
            continue
        key = (r.get("batch"), r.get("parent_node_id"))
        if key in seen:
            continue
        seen.add(key)
        cases.append(r)
    return cases[:limit] if limit else cases


def prepare(cases: list[dict]) -> tuple[list[dict], int]:
    """Resolve each case's parent atom ONCE and hand it to both arms.

    Charging the parent snap twice would dilute the delta being measured, and a parent
    that no longer solves today is dropped rather than counted as a disagreement."""
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
        prepared.append(dict(row=row, view=view, degree=degree,
                             parent_rec=dict(id=snap.atom_id, cx=snap.cx, cy=snap.cy,
                                             period=snap.period,
                                             window_scale=snap.window_scale,
                                             degree=degree)))
    return prepared, no_parent


def _rng(case) -> np.random.Generator:
    """One seed per CASE, shared by both arms — this is what makes the probe seeds
    byte-identical between the operators rather than merely similarly distributed."""
    return np.random.default_rng(abs(int(case["view"]["node_id"] or 0)) + 1)


def run_arms(prepared: list[dict], m: int, probes: int) -> list[dict]:
    out = []
    for case in prepared:
        t0 = time.time()
        lat = mnv.lateral_to_sibling(case["view"], _rng(case), degree=case["degree"],
                                     parent_rec=case["parent_rec"], n_probes=probes)
        t_lat = time.time() - t0
        t0 = time.time()
        nbh = mnv.neighborhood_expand(case["view"], _rng(case), [None],
                                      degree=case["degree"],
                                      parent_rec=case["parent_rec"],
                                      max_found=m, max_probes=probes)
        t_nbh = time.time() - t0
        avail = [x for x in nbh if x.available]
        out.append(dict(
            node_id=case["view"]["node_id"],
            partition=case["row"].get("partition"),
            lat_available=bool(lat.available), lat_reason=lat.reason,
            lat_atom=lat.atom_id, lat_period=lat.period,
            lat_solves=lat.newton_solves, lat_s=t_lat,
            nbh_available=bool(avail), nbh_reason=(nbh[0].reason if not avail else ""),
            nbh_first=(avail[0].atom_id if avail else None),
            nbh_first_period=(avail[0].period if avail else None),
            nbh_atoms=[x.atom_id for x in avail],
            nbh_ratios=[x.extra.get("scale_ratio_decades") for x in avail],
            nbh_solves=nbh[0].newton_solves, nbh_s=t_nbh,
        ))
    return out


def summarize(rows: list[dict], m: int) -> dict:
    both = [r for r in rows if r["lat_available"] and r["nbh_available"]]
    same_first = [r for r in both if r["lat_atom"] == r["nbh_first"]]
    lat_in_set = [r for r in both if r["lat_atom"] in r["nbh_atoms"]]
    lat_only = [r for r in rows if r["lat_available"] and not r["nbh_available"]]
    nbh_only = [r for r in rows if r["nbh_available"] and not r["lat_available"]]
    # The pre-registered explanation for the one-sided arm: neighborhood is available where
    # lateral said scale_mismatch, on a candidate BELOW lateral's window.
    nbh_only_from_scale = [r for r in nbh_only if r["lat_reason"] == "scale_mismatch"]
    below = sum(1 for r in nbh_only_from_scale
                if r["nbh_ratios"] and min(x for x in r["nbh_ratios"] if x is not None)
                < -mnv.LAT_SCALE_TOL_DECADES)
    return dict(
        m=m, n_cases=len(rows),
        lat_available=sum(1 for r in rows if r["lat_available"]),
        nbh_available=sum(1 for r in rows if r["nbh_available"]),
        both_available=len(both),
        identical_first_pick=len(same_first),
        lateral_pick_inside_neighborhood_set=len(lat_in_set),
        lateral_only=len(lat_only), neighborhood_only=len(nbh_only),
        neighborhood_only_where_lateral_said_scale_mismatch=len(nbh_only_from_scale),
        of_those_below_laterals_window=below,
        lateral_reasons=dict(Counter(r["lat_reason"] for r in rows
                                     if not r["lat_available"]).most_common()),
        neighborhood_reasons=dict(Counter(r["nbh_reason"] for r in rows
                                          if not r["nbh_available"]).most_common()),
        mean_atoms_found=(round(sum(len(r["nbh_atoms"]) for r in rows) / len(rows), 2)
                          if rows else 0.0),
        newton_solves_lateral=sum(r["lat_solves"] for r in rows),
        newton_solves_neighborhood=sum(r["nbh_solves"] for r in rows),
        seconds_lateral=round(sum(r["lat_s"] for r in rows), 2),
        seconds_neighborhood=round(sum(r["nbh_s"] for r in rows), 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--m", type=int, nargs="+", default=[1, mnv.NBH_MAX_FOUND],
                    help="neighborhood find-ceilings to bench (1 == the subsumption arm)")
    ap.add_argument("--probes", type=int, default=mnv.LAT_PROBES,
                    help="probe budget for BOTH arms — equal by construction, or the "
                         "comparison measures budget rather than filters")
    ap.add_argument("--out", type=Path,
                    default=paths.scratch("maneuvers", "bench_neighborhood_subsumption.json"))
    a = ap.parse_args()

    cases = load_cases(a.log, a.limit)
    print(f"[bench] {len(cases)} deduped lateral calls from {a.log}")
    prepared, no_parent = prepare(cases)
    print(f"[bench] {len(prepared)} replayable ({no_parent} had no parent atom today)")
    if not prepared:
        print("[bench] nothing replayable — refusing to report a verdict off zero cases")
        return 1

    res = dict(log=str(a.log), n_replayable=len(prepared), no_parent_atom=no_parent,
               probes_per_arm=a.probes,
               lateral_scale_tol_decades=mnv.LAT_SCALE_TOL_DECADES,
               neighborhood_scale_up_decades=mnv.NBH_SCALE_UP_DECADES, arms=[])
    for m in a.m:
        rows = run_arms(prepared, m, a.probes)
        s = summarize(rows, m)
        res["arms"].append(s)
        print(f"  m={m}: lateral avail {s['lat_available']}/{s['n_cases']}, "
              f"neighborhood avail {s['nbh_available']}/{s['n_cases']}; "
              f"identical first pick {s['identical_first_pick']}/{s['both_available']}, "
              f"lateral's pick inside the set "
              f"{s['lateral_pick_inside_neighborhood_set']}/{s['both_available']}")
        print(f"        lateral-only {s['lateral_only']}, neighborhood-only "
              f"{s['neighborhood_only']} (of which lateral said scale_mismatch: "
              f"{s['neighborhood_only_where_lateral_said_scale_mismatch']}, below its "
              f"window: {s['of_those_below_laterals_window']}); mean atoms found "
              f"{s['mean_atoms_found']}; solves {s['newton_solves_lateral']} vs "
              f"{s['newton_solves_neighborhood']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[bench] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
