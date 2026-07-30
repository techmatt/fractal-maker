#!/usr/bin/env python
r"""Derive v8 per-partition t_good — the ADOPTED discovery table.

The decode is the CORN keeper rule: predict "keeper" (label>=3) iff P(>=2) >= 0.5 AND
P(>=3) >= t. Threshold search is an F_beta-argmax over a p_good grid, per fractal
partition, on the unbiased eval slice, requiring >=15 positives; tie-break toward higher t.

  * OBJECTIVE IS PER-PARTITION, not uniform. **Weight recall where supply is scarce,
    weight precision where supply is abundant.** A missed mandelbrot costs nothing —
    mandelbrot is effectively unlimited and the next hunt finds more — so mandelbrot is
    derived at F0.5 (precision-weighted). A missed julia:multibrot costs real money
    against a saturating supply, so those are derived at F2 (recall-weighted). A false
    admit costs the same everywhere; it is the cost of a MISS that differs by family, and
    that is the whole justification for a split objective. Uniform-F2 on mandelbrot lands
    at t=0.14 / precision 0.292 — roughly three and a half bad locations admitted per good
    one, on the largest family in the corpus. See OBJECTIVE below and
    docs/design/classifier_retrain_protocol.md §4.
  * BOTH objectives are reported for every derived partition (F0.5, F0.5_OOF, F2, F2_OOF)
    so the choice is visible and auditable rather than implicit in the adopted number.
  * NO class-4 threshold. Class 4 decodes at its natural cutpoint P(>=4) >= 0.5 and gets no
    per-family calibration.
  * NO native-multibrot tightening. Native mb3/4/5 (and any partition without an unbiased
    eval slice) are stamped **UNCALIBRATED** and RUN at the baseline 0.50. Uncalibrated is
    not a derivation: a baseline 0.50 and a derived 0.50 are indistinguishable as a bare
    number in a config file, so the distinction is carried explicitly (`status`) here and
    by `production_seeder.T_GOOD_UNCALIBRATED` there.
  * v8's eval slice covers TWO partitions with unbiased data: julia:multibrot{3,4,5}
    (the census) and mandelbrot (the new loose0_v3 floor). Everything else -> UNCALIBRATED.

Reads data/v8/eval_scores_v8.jsonl (frozen by eval_v8). Writes data/v8/t_good_derivation.json
(durable) and prints the table. `production_seeder.T_GOOD_OVERRIDES` is the ADOPTED copy of
the `adopted` block below; tools/v8/test_derive_t_good_v8.py holds the two in agreement.

  uv run python tools/v8/derive_t_good_v8.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCORES = ROOT / "data/v8/eval_scores_v8.jsonl"
OUT = "data/v8/t_good_derivation.json"

BASELINE = 0.50
MIN_POS = 15
NB_GATE = 0.5                     # the fixed P(>=2) keeper gate (corn_decode)
GRID = np.round(np.arange(0.02, 0.98 + 1e-9, 0.01), 4)

# Per-partition objective. beta > 1 weights RECALL, beta < 1 weights PRECISION. The
# assignment follows supply, not model behaviour: scarce supply -> recall, abundant supply
# -> precision (see the module docstring). A partition absent here defaults to F2, the
# historical discovery objective.
OBJECTIVE = {
    "mandelbrot": 0.5,          # abundant supply -> precision-weighted
    "julia:multibrot3": 2.0,    # saturating supply -> recall-weighted
    "julia:multibrot4": 2.0,
    "julia:multibrot5": 2.0,
}
DEFAULT_BETA = 2.0

# fractal_type -> ledger family partition (production_seeder keying)
FT2FAM = {
    "mandelbrot": "mandelbrot", "julia": "julia:mandelbrot",
    "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
    "julia_multibrot3": "julia:multibrot3", "julia_multibrot4": "julia:multibrot4",
    "julia_multibrot5": "julia:multibrot5", "phoenix": "phoenix",
}
ALL_FAMS = ["mandelbrot", "julia:mandelbrot", "multibrot3", "multibrot4", "multibrot5",
            "julia:multibrot3", "julia:multibrot4", "julia:multibrot5", "phoenix"]


def beta_name(beta: float) -> str:
    return f"F{beta:g}"


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * p + r
    return ((1.0 + b2) * p * r / denom) if denom > 0 else 0.0


def prf_at(t, p_nb, p_gd, y, beta):
    pred = (p_nb >= NB_GATE) & (p_gd >= t)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, fbeta(prec, rec, beta), tp, fp, fn


def best_t(p_nb, p_gd, y, beta):
    """F_beta-argmax over GRID, tie-break toward HIGHER t (protocol).

    Also reports the argmax PLATEAU — the contiguous run of grid steps around the winner
    that score within 1e-12 of the optimum. Tie-breaking high puts the adopted t at the
    plateau's upper edge by construction, so the plateau width is the only honest read on
    how knife-edged the pick is."""
    best = (-1.0, -1.0, None)  # (f, t, block)
    curve = []
    for t in GRID:
        prec, rec, fb, tp, fp, fn = prf_at(t, p_nb, p_gd, y, beta)
        curve.append((float(t), fb))
        if fb > best[0] + 1e-12 or (abs(fb - best[0]) <= 1e-12 and t > best[1]):
            best = (fb, float(t), {"t": float(t), "precision": round(prec, 4),
                                   "recall": round(rec, 4), "fbeta": round(fb, 4),
                                   "tp": tp, "fp": fp, "fn": fn})
    blk = best[2]
    at_opt = [t for t, f in curve if abs(f - best[0]) <= 1e-12]
    blk["plateau"] = [min(at_opt), max(at_opt)]
    blk["plateau_steps"] = len(at_opt)
    return blk


def loo_fbeta(p_nb, p_gd, y, beta):
    """Leave-one-out OOF F_beta: for each held-out location, choose t on the rest, predict the
    held-out, then score F_beta over the pooled OOF predictions. Optimism estimate."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        t = best_t(p_nb[m], p_gd[m], y[m], beta)["t"]
        preds[i] = int((p_nb[i] >= NB_GATE) and (p_gd[i] >= t))
    tp = int(((preds == 1) & (y == 1)).sum()); fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return round(fbeta(prec, rec, beta), 4)


def derive_partition(grp, betas=(0.5, 2.0)) -> dict:
    """Every objective's argmax + OOF for one partition, keyed 'F0.5'/'F2'."""
    y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
    p_nb = np.array([g["v8_p_ge2"] for g in grp])
    p_gd = np.array([g["v8_p_ge3"] for g in grp])
    out = {}
    for beta in betas:
        blk = best_t(p_nb, p_gd, y, beta)
        blk["beta"] = beta
        blk["fbeta_oof"] = loo_fbeta(p_nb, p_gd, y, beta)
        out[beta_name(beta)] = blk
    return out


def main():
    if not SCORES.exists():
        sys.exit(f"missing {SCORES} — run tools/v8/eval_v8.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[FT2FAM.get(r["fractal_type"], r["fractal_type"])].append(r)

    derived, uncal, adopted, notes = {}, {}, {}, []
    print("=" * 104)
    print("v8 t_good — PER-FAMILY objective (recall where supply is scarce, precision where "
          "it is abundant)")
    print("=" * 104)
    print(f"  decode: keeper(label>=3) iff P(>=2)>={NB_GATE} AND P(>=3)>=t   grid {GRID[0]}..{GRID[-1]}")
    print(f"  UNCALIBRATED -> runs at baseline {BASELINE} where <{MIN_POS} positives or no unbiased "
          f"eval slice; NO class-4 t; NO native tightening\n")
    print(f"  {'partition':20s} {'obj':>5} {'n':>4} {'pos':>4}  {'t_good':>7} {'prec':>6} {'rec':>6} "
          f"{'F':>6} {'F_OOF':>7} {'plateau':>13}  status")

    for fam in sorted(by_fam):
        grp = by_fam[fam]
        y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
        n, pos = len(grp), int(y.sum())
        if pos < MIN_POS or pos == n:
            reason = (f"<{MIN_POS} positives" if pos < MIN_POS else "no negatives")
            uncal[fam] = {"n": n, "pos": pos, "t_good": BASELINE,
                          "status": "UNCALIBRATED", "reason": reason}
            print(f"  {fam:20s} {'—':>5} {n:>4} {pos:>4}  {BASELINE:>7.2f} {'—':>6} {'—':>6} "
                  f"{'—':>6} {'—':>7} {'—':>13}  UNCALIBRATED ({reason})")
            continue

        beta = OBJECTIVE.get(fam, DEFAULT_BETA)
        blocks = derive_partition(grp)
        chosen = blocks[beta_name(beta)]
        derived[fam] = {"n": n, "pos": pos, "objective": beta_name(beta), "beta": beta,
                        "objective_rationale": ("precision-weighted (supply abundant)" if beta < 1
                                                else "recall-weighted (supply scarce)"),
                        "t_good": chosen["t"], "status": "DERIVED", "by_objective": blocks}
        adopted[fam] = chosen["t"]
        for nm, blk in blocks.items():
            mark = "<= ADOPTED" if nm == beta_name(beta) else ""
            print(f"  {fam:20s} {nm:>5} {n:>4} {pos:>4}  {blk['t']:>7.2f} {blk['precision']:>6.3f} "
                  f"{blk['recall']:>6.3f} {blk['fbeta']:>6.3f} {blk['fbeta_oof']:>7.3f} "
                  f"{'[%.2f,%.2f]' % tuple(blk['plateau']):>13}  {mark}")
        # honesty flags, both reported not silently absorbed.
        if blocks["F0.5"]["t"] != blocks["F2"]["t"]:
            notes.append(f"{fam}: F0.5 picks t={blocks['F0.5']['t']:.2f} vs F2 t={blocks['F2']['t']:.2f} "
                         f"— objective choice is load-bearing, not cosmetic")
        gap = chosen["fbeta"] - chosen["fbeta_oof"]
        if gap > 0.02:
            notes.append(f"{fam}: adopted {beta_name(beta)} in-sample {chosen['fbeta']:.3f} vs OOF "
                         f"{chosen['fbeta_oof']:.3f} (gap {gap:+.3f}) — threshold overfit at n={n}, "
                         f"pos={pos}; the OOF value is the honest one")
        if chosen["plateau_steps"] <= 2:
            notes.append(f"{fam}: adopted t={chosen['t']:.2f} sits on a {chosen['plateau_steps']}-step "
                         f"plateau — knife-edge, re-derive rather than nudge")

    # partitions with no eval rows at all -> UNCALIBRATED, named individually.
    for f in ALL_FAMS:
        if f in derived or f in uncal:
            continue
        uncal[f] = {"n": 0, "pos": 0, "t_good": BASELINE, "status": "UNCALIBRATED",
                    "reason": "no unbiased eval slice"}
        print(f"  {f:20s} {'—':>5} {0:>4} {0:>4}  {BASELINE:>7.2f} {'—':>6} {'—':>6} {'—':>6} "
              f"{'—':>7} {'—':>13}  UNCALIBRATED (no unbiased eval slice)")

    if notes:
        print("\n  read-carefully:")
        for nt in notes:
            print(f"    * {nt}")

    out = {
        "objective": "per-partition F_beta-argmax, tie-break higher t",
        "objective_principle": ("Weight recall where supply is SCARCE, weight precision where "
                               "supply is ABUNDANT. A false admit costs the same everywhere; the "
                               "cost of a MISS is what differs by family. Mandelbrot is "
                               "effectively unlimited (a miss costs nothing, the next hunt finds "
                               "more) -> F0.5. julia:multibrot supply saturates (a miss costs real "
                               "money) -> F2."),
        "objective_by_partition": {k: beta_name(v) for k, v in OBJECTIVE.items()},
        "default_objective": beta_name(DEFAULT_BETA),
        "decode": f"keeper(label>=3) iff P(>=2)>={NB_GATE} AND P(>=3)>=t",
        "baseline": BASELINE, "min_pos": MIN_POS,
        "no_class4_threshold": True,
        "class4_decode": "natural cutpoint P(>=4) >= 0.5, no per-family calibration",
        "native_multibrot_tightening": "NOT adopted (fork-scheduled proposals left on record)",
        "eval_slice": "data/v8/eval_scores_v8.jsonl",
        "model": "v8",
        "derived": derived,
        "uncalibrated": uncal,
        "adopted": adopted,
        "notes": notes,
        "note": ("ADOPTED — production_seeder.T_GOOD_OVERRIDES mirrors `adopted`, and the "
                 "`uncalibrated` partitions are named in production_seeder.T_GOOD_UNCALIBRATED "
                 "(they RUN at the baseline but were never derived; the two 0.50s must stay "
                 "distinguishable). tools/v8/test_derive_t_good_v8.py holds them in agreement."),
    }
    sys.path.insert(0, str(ROOT / "tools"))
    import paths  # noqa: E402
    paths.durable(OUT, mkparents=True).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} (durable) — ADOPTED table; mirror into production_seeder.T_GOOD_OVERRIDES")


if __name__ == "__main__":
    main()
