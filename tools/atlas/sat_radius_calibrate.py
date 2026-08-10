#!/usr/bin/env python
"""sat_radius_calibrate.py — where `steered_frontier.SAT_RADIUS_K` comes from.

The saturation discount multiplies a breadth candidate's steering weight by
`1/(1 + SAT_STRENGTH * density)`, and `SAT_RADIUS_K` is the only knob that decides how much
territory a past visit shadows. Too small and the term never fires; too large and it fires on
everything, at which point it is a CONSTANT OFFSET rather than a gradient and steers nothing.

That failure is not hypothetical here — it is what the morph-novelty term did. Re-derived from
`data/discovery/steered_run2/prio_terms.jsonl` (38,419 pushed candidates, the run the deleted
`steered_run2_report.py` used to read): 99.58% of candidates carried a nonzero novelty penalty
and **92.88%** sat within 10% of the FULL penalty. A term that fires at full strength on 93% of
the population subtracts a near-constant from every priority and reorders nothing. So the
calibration bar is stated against that number rather than against a preference:

  SATURATED share  = fraction of candidates whose discount is within 10% of full,
                     i.e. density >= KNEE_DENSITY (9 at SAT_STRENGTH = 1.0).
  ADOPT the largest k whose saturated share stays <= 5% POOLED and <= 10% in EVERY PARTITION.

Both bars, because a pooled share hides a partition: phoenix crosses 10% while the pool is
still at 2.6%, and phoenix's own z-plane really is the most-visited surface in the store (954
of its 1,933 ledger rows sit on the one classic Ushiki plane).

POPULATION. Query rows are the committed `q4_candidates` streams — the widest record of
BREADTH candidates that carries coordinates AND the dynamical parameter (`prio_terms`, the
only per-pushed-candidate stream, carries neither, which is why this cannot be measured on the
pushed population directly). Visits are the committed ledger union, LEAVE-ONE-RUN-OUT: a
candidate run's own ledger is excluded, every other run's is in — including runs that came
later. That over-states what any of these runs actually faced and is the right direction: the
quantity being calibrated is what the NEXT run will face.

  uv run python tools/atlas/sat_radius_calibrate.py                    # the sweep + verdict
  uv run python tools/atlas/sat_radius_calibrate.py --json out.json    # + machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_record                        # noqa: E402  (reads segmented + .gz streams)
import visited_density as vd             # noqa: E402  THE index under calibration

# The morph term's own knee, restated in density units: `1/(1+1*9) = 0.10`, i.e. the discount
# is within 10% of full. Chosen to match `steered_frontier.sat_cos` ("within 10% of full
# penalty") so the two saturation numbers are comparable at all.
KNEE_DENSITY = 9
POOLED_BAR = 0.05
PARTITION_BAR = 0.10
SWEEP = (0.05, 0.10, 0.20, 0.25, 0.30, 0.32, 0.35, 0.375, 0.50, 0.75, 1.00)


def run2_saturation(root: Path = ROOT) -> dict:
    """The degeneracy reference, re-derived rather than quoted. Returns the share of run 2's
    pushed candidates at/over its own novelty knee, computed the way `push_children` computes
    it (`sat_cos = lo + 0.9*(hi-lo)` off the run's recorded anchors)."""
    run = root / "data" / "discovery" / "steered_run2"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    lo, hi = float(summary["morph_lo"]), float(summary["morph_hi"])
    knee = lo + 0.9 * (hi - lo)
    n = n_sat = n_pen = 0
    for r in run_record.iter_rows(run / "prio_terms.jsonl"):
        n += 1
        n_sat += float(r["cos_max"]) >= knee
        n_pen += float(r["nov_pen"]) > 0.0
    return dict(run="steered_run2", rows=n, morph_lo=lo, morph_hi=hi, sat_cos=round(knee, 4),
                sat_frac=round(n_sat / n, 4), any_penalty_frac=round(n_pen / n, 4))


def candidate_rows(root: Path = ROOT) -> dict:
    """`run_dir -> [(partition, ident, cx, cy)]` from every committed q4_candidates stream."""
    out: dict = defaultdict(list)
    seen = set()
    for p in sorted((root / "data" / "discovery").rglob("q4_candidates*.jsonl*")):
        logical = p.parent / "q4_candidates.jsonl"
        if logical in seen:
            continue                      # one logical stream, N segments
        seen.add(logical)
        run = str(p.parent.relative_to(root)).replace("\\", "/")
        for r in run_record.iter_rows(logical):
            if r.get("cx") is None or r.get("fw") is None:
                continue
            # A candidate row's identity is read with the SAME rule a ledger row's is
            # (`row_ident` keys phoenix off `family`), so give it the family it renders as.
            row = dict(r, family=vd.P.base_partition(r["partition"]))
            out[run].append((r["partition"], vd.ps.row_ident(row),
                             float(r["cx"]), float(r["cy"])))
    return out


def sweep(root: Path = ROOT, ks=SWEEP) -> dict:
    """Per-k, per-partition density stats over the leave-one-run-out population."""
    cands = candidate_rows(root)
    per_k: dict = {k: defaultdict(list) for k in ks}
    for run, rows in sorted(cands.items()):
        own = root / run / "outcome_ledger.jsonl"
        for k in ks:
            idx = vd.build_from_ledgers(k, root, exclude=own if own.exists() else None)
            for part, ident, cx, cy in rows:
                per_k[k][part].append(idx.density(part, ident, cx, cy))
    res = {}
    for k in ks:
        parts = {}
        pooled = []
        for part, ds in sorted(per_k[k].items()):
            pooled += ds
            parts[part] = _stats(ds)
        res[k] = dict(partitions=parts, pooled=_stats(pooled))
    return dict(runs=sorted(cands), n_candidates=sum(len(v) for v in cands.values()), by_k=res)


def _stats(ds: list) -> dict:
    n = len(ds)
    if n == 0:
        return dict(n=0)
    ds_sorted = sorted(ds)
    return dict(n=n,
                frac_discounted=round(sum(1 for d in ds if d > 0) / n, 4),
                frac_saturated=round(sum(1 for d in ds if d >= KNEE_DENSITY) / n, 4),
                p50=ds_sorted[n // 2], p90=ds_sorted[min(n - 1, int(0.90 * n))],
                max=ds_sorted[-1],
                mean_discount=round(sum(1.0 / (1.0 + d) for d in ds) / n, 4))


def verdict(res: dict) -> dict:
    """The largest swept k inside BOTH bars, and where the grid crosses them."""
    passing = [k for k, r in sorted(res["by_k"].items())
               if r["pooled"]["frac_saturated"] <= POOLED_BAR
               and all(p["frac_saturated"] <= PARTITION_BAR
                       for p in r["partitions"].values())]
    return dict(knee_density=KNEE_DENSITY, pooled_bar=POOLED_BAR,
                partition_bar=PARTITION_BAR, passing=passing,
                largest_passing=(max(passing) if passing else None))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None, help="write the full readout here")
    ap.add_argument("--ks", type=str, default=None, help="comma list overriding the sweep")
    args = ap.parse_args()
    ks = tuple(float(x) for x in args.ks.split(",")) if args.ks else SWEEP

    t0 = time.time()
    ref = run2_saturation()
    print(f"[reference] run 2 morph novelty: {ref['sat_frac']:.4f} of {ref['rows']} pushed "
          f"candidates within 10% of FULL penalty ({ref['any_penalty_frac']:.4f} carried any) "
          f"— the constant-offset regime this knob must stay out of")
    res = sweep(ks=ks)
    print(f"[population] {res['n_candidates']} candidate rows over {len(res['runs'])} runs, "
          f"leave-one-run-out against the committed ledger union")
    print(f"\n{'k':>7} {'discounted':>11} {'saturated':>10} {'p50':>5} {'p90':>5} "
          f"{'mean disc':>10}   worst partition")
    for k in ks:
        r = res["by_k"][k]
        worst = max(r["partitions"], key=lambda p: r["partitions"][p]["frac_saturated"])
        w = r["partitions"][worst]
        print(f"{k:7.3f} {r['pooled']['frac_discounted']:11.4f} "
              f"{r['pooled']['frac_saturated']:10.4f} {r['pooled']['p50']:5d} "
              f"{r['pooled']['p90']:5d} {r['pooled']['mean_discount']:10.4f}   "
              f"{worst} {w['frac_saturated']:.4f}")
    v = verdict(res)
    print(f"\n[verdict] bars: saturated <= {POOLED_BAR} pooled and <= {PARTITION_BAR} per "
          f"partition (saturated == density >= {KNEE_DENSITY})")
    print(f"[verdict] passing k: {v['passing']} — largest {v['largest_passing']}")
    print(f"[verdict] adopted default is steered_frontier.SAT_RADIUS_K "
          f"(a round value inside the bar, not the grid maximum)")
    out = dict(reference=ref, sweep=res, verdict=v, wall_s=round(time.time() - t0, 1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[readout] -> {args.json}")


if __name__ == "__main__":
    main()
