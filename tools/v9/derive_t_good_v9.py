#!/usr/bin/env python
r"""Derive v9 per-partition t_good — STAGED, not adopted.

Every threshold derived before the cap raise sits on a moved score distribution: v9 reads
raised-cap renders, so its `P(>=3)` is not v8's `P(>=3)` and a v8 cut applied to a v9 gate
is a number about nothing. This re-derives the table on the SAME objectives, from the
durable v9 eval slice.

  * SAME OBJECTIVES AS v8, deliberately. F0.5 (precision-weighted) for mandelbrot, F2
    (recall-weighted) for julia:multibrot{3,4,5} — recall where supply is scarce, precision
    where it is abundant. Re-deriving on a different objective would confound "the cap
    moved" with "the objective moved", and the whole point of this pass is to read the cap.
  * BOTH objectives reported for every derived partition, as in v8, so the choice stays
    visible rather than implicit in the adopted number.
  * The partitions with no unbiased eval rows stay stamped **UNCALIBRATED** and are NEVER
    written as a derived 0.50. A baseline 0.50 and a derived 0.50 are indistinguishable as
    a bare number in a config file; the distinction is carried in `status`.

STAGED. This writes `data/v9/t_good_derivation.json` and **does not touch**
`production_seeder.T_GOOD_OVERRIDES`, which stays calibrated to v8's p_good scale for as
long as v8 is the deployed scorer. Build is not flip. The mirror happens with the
ACTIVE_CKPT flip, in its own pass, conditional on the pre-registered bar.

Reads data/v9/eval_scores_v9.jsonl (frozen by tools/v9/eval_v9.py). The derivation itself
is imported from tools/v8/derive_t_good_v8.py rather than copied, so the v8 and v9 tables
cannot diverge by an editing accident — which would make them incomparable, which would
defeat the comparison they exist for.

  uv run python tools/v9/derive_t_good_v9.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "v8"))

# The derivation, imported not copied: grid, F_beta, tie-break, LOO-OOF, plateau, the
# per-partition objective table and the family mapping all come from v8's module.
from derive_t_good_v8 import (ALL_FAMS, BASELINE, DEFAULT_BETA, FT2FAM,  # noqa: E402
                              GRID, MIN_POS, NB_GATE, OBJECTIVE, best_t,
                              beta_name, loo_fbeta)

SCORES = ROOT / "data/v9/eval_scores_v9.jsonl"
OUT = "data/v9/t_good_derivation.json"
VERSION = "v9"


def derive_partition(grp, betas=(0.5, 2.0)) -> dict:
    """v8's derive_partition, reading the v9 columns. (v8's is hardwired to `v8_p_ge*`.)"""
    y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
    p_nb = np.array([g[f"{VERSION}_p_ge2"] for g in grp])
    p_gd = np.array([g[f"{VERSION}_p_ge3"] for g in grp])
    out = {}
    for beta in betas:
        blk = best_t(p_nb, p_gd, y, beta)
        blk["beta"] = beta
        blk["fbeta_oof"] = loo_fbeta(p_nb, p_gd, y, beta)
        out[beta_name(beta)] = blk
    return out


def main():
    if not SCORES.exists():
        sys.exit(f"missing {SCORES} — run tools/v9/eval_v9.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[FT2FAM.get(r["fractal_type"], r["fractal_type"])].append(r)

    derived, uncal, adopted, notes = {}, {}, {}, []
    print("=" * 104)
    print("v9 t_good — SAME objectives as v8 (recall where supply is scarce, precision where "
          "it is abundant)")
    print("=" * 104)
    print(f"  decode: keeper(label>=3) iff P(>=2)>={NB_GATE} AND P(>=3)>=t   "
          f"grid {GRID[0]}..{GRID[-1]}")
    print(f"  UNCALIBRATED -> runs at baseline {BASELINE} where <{MIN_POS} positives or no "
          f"unbiased eval slice; NO class-4 t; NO native tightening")
    print("  STAGED — production_seeder.T_GOOD_OVERRIDES is NOT updated here.\n")
    print(f"  {'partition':20s} {'obj':>5} {'n':>4} {'pos':>4}  {'t_good':>7} {'prec':>6} "
          f"{'rec':>6} {'F':>6} {'F_OOF':>7} {'plateau':>13}  status")

    for fam in sorted(by_fam):
        grp = by_fam[fam]
        y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
        n, pos = len(grp), int(y.sum())
        if pos < MIN_POS or pos == n:
            reason = f"<{MIN_POS} positives" if pos < MIN_POS else "no negatives"
            uncal[fam] = {"n": n, "pos": pos, "t_good": BASELINE,
                          "status": "UNCALIBRATED", "reason": reason}
            print(f"  {fam:20s} {'-':>5} {n:>4} {pos:>4}  {BASELINE:>7.2f} {'-':>6} {'-':>6} "
                  f"{'-':>6} {'-':>7} {'-':>13}  UNCALIBRATED ({reason})")
            continue

        beta = OBJECTIVE.get(fam, DEFAULT_BETA)
        blocks = derive_partition(grp)
        chosen = blocks[beta_name(beta)]
        derived[fam] = {"n": n, "pos": pos, "objective": beta_name(beta), "beta": beta,
                        "objective_rationale": ("precision-weighted (supply abundant)"
                                                if beta < 1 else
                                                "recall-weighted (supply scarce)"),
                        "t_good": chosen["t"], "status": "DERIVED", "by_objective": blocks}
        adopted[fam] = chosen["t"]
        for nm, blk in blocks.items():
            mark = "<= ADOPTED" if nm == beta_name(beta) else ""
            print(f"  {fam:20s} {nm:>5} {n:>4} {pos:>4}  {blk['t']:>7.2f} "
                  f"{blk['precision']:>6.3f} {blk['recall']:>6.3f} {blk['fbeta']:>6.3f} "
                  f"{blk['fbeta_oof']:>7.3f} "
                  f"{'[%.2f,%.2f]' % tuple(blk['plateau']):>13}  {mark}")
        if blocks["F0.5"]["t"] != blocks["F2"]["t"]:
            notes.append(f"{fam}: F0.5 picks t={blocks['F0.5']['t']:.2f} vs F2 "
                         f"t={blocks['F2']['t']:.2f} — objective choice is load-bearing")
        gap = chosen["fbeta"] - chosen["fbeta_oof"]
        if gap > 0.02:
            notes.append(f"{fam}: adopted {beta_name(beta)} in-sample {chosen['fbeta']:.3f} vs "
                         f"OOF {chosen['fbeta_oof']:.3f} (gap {gap:+.3f}) — threshold overfit "
                         f"at n={n}, pos={pos}; the OOF value is the honest one")
        if chosen["plateau_steps"] <= 2:
            notes.append(f"{fam}: adopted t={chosen['t']:.2f} sits on a "
                         f"{chosen['plateau_steps']}-step plateau — knife-edge, re-derive "
                         f"rather than nudge")

    for f in ALL_FAMS:
        if f in derived or f in uncal:
            continue
        uncal[f] = {"n": 0, "pos": 0, "t_good": BASELINE, "status": "UNCALIBRATED",
                    "reason": "no unbiased eval slice"}
        print(f"  {f:20s} {'-':>5} {0:>4} {0:>4}  {BASELINE:>7.2f} {'-':>6} {'-':>6} {'-':>6} "
              f"{'-':>7} {'-':>13}  UNCALIBRATED (no unbiased eval slice)")

    # --- v8 comparison: how far did the cap raise move each threshold? ---
    v8_path = ROOT / "data/v8/t_good_derivation.json"
    moved = {}
    if v8_path.exists():
        v8 = json.loads(v8_path.read_text(encoding="utf-8"))
        for fam, t in adopted.items():
            t8 = v8.get("adopted", {}).get(fam)
            if t8 is not None:
                moved[fam] = {"v8": t8, "v9": t, "delta": round(t - t8, 4)}
        if moved:
            print("\n  vs v8 (the thresholds the moved distribution invalidated):")
            for fam, m in sorted(moved.items()):
                print(f"    {fam:20s} v8 {m['v8']:.2f} -> v9 {m['v9']:.2f}  "
                      f"({m['delta']:+.2f})")

    if notes:
        print("\n  read-carefully:")
        for nt in notes:
            print(f"    * {nt}")

    out = {
        "objective": "per-partition F_beta-argmax, tie-break higher t",
        "objective_by_partition": {k: beta_name(v) for k, v in OBJECTIVE.items()},
        "objective_unchanged_from_v8": True,
        "objective_unchanged_rationale": (
            "Re-deriving on a different objective would confound 'the cap moved' with "
            "'the objective moved'. Same F0.5/F2 assignment as v8, deliberately."),
        "default_objective": beta_name(DEFAULT_BETA),
        "decode": f"keeper(label>=3) iff P(>=2)>={NB_GATE} AND P(>=3)>=t",
        "baseline": BASELINE, "min_pos": MIN_POS,
        "no_class4_threshold": True,
        "class4_decode": "natural cutpoint P(>=4) >= 0.5, no per-family calibration",
        "eval_slice": "data/v9/eval_scores_v9.jsonl",
        "model": VERSION,
        "derived": derived,
        "uncalibrated": uncal,
        "adopted": adopted,
        "vs_v8": moved,
        "notes": notes,
        "status": "STAGED — NOT adopted into production_seeder.T_GOOD_OVERRIDES",
        "note": ("STAGED, not adopted. production_seeder.T_GOOD_OVERRIDES stays calibrated "
                 "to v8's p_good scale while v8 is the deployed scorer — a v9 cut on a v8 "
                 "gate is a number about nothing, in exactly the way a v7 cut on a v8 gate "
                 "was. The mirror happens with the ACTIVE_CKPT flip, in its own pass, "
                 "conditional on the pre-registered bar."),
    }
    import paths  # noqa: E402
    paths.durable(OUT, mkparents=True).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} (durable) — STAGED table; do NOT mirror until the flip")


if __name__ == "__main__":
    main()
