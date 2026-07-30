#!/usr/bin/env python
r"""Derive v8 per-partition t_good — REPORT ONLY. Touches no production threshold/config.

Same objective v7 used for the DISCOVERY t_good: F2-argmax (recall-weighted) over a p_good
grid, per fractal partition, on the unbiased eval slice, requiring >=15 positives; tie-break
toward higher t. The decode is the CORN keeper rule: predict "keeper" (label>=3) iff
P(>=2) >= 0.5 AND P(>=3) >= t.

  * NO class-4 threshold. Class 4 decodes at its natural cutpoint P(>=4) >= 0.5 and gets no
    per-family calibration (prompt).
  * NO native-multibrot tightening. Native mb3/4/5 (and any partition without an unbiased
    eval slice) stay at the baseline 0.50 — reported, not adopted. The on-record fork-
    scheduled native-multibrot proposals are deliberately NOT applied here.
  * v8's eval slice now covers TWO partitions with unbiased data: julia:multibrot{3,4,5}
    (the census) and mandelbrot (the new loose0_v3 floor). Everything else -> baseline.

Reads data/v8/eval_scores_v8.jsonl (frozen by eval_v8). Writes data/v8/t_good_derivation.json
(durable) and prints the table. Does NOT edit tools/atlas/production_seeder.py.

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

# fractal_type -> ledger family partition (production_seeder keying)
FT2FAM = {
    "mandelbrot": "mandelbrot", "julia": "julia:mandelbrot",
    "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
    "julia_multibrot3": "julia:multibrot3", "julia_multibrot4": "julia:multibrot4",
    "julia_multibrot5": "julia:multibrot5", "phoenix": "phoenix",
}


def f2(p, r):
    if p == 0 and r == 0:
        return 0.0
    return 5.0 * p * r / (4.0 * p + r) if (4.0 * p + r) > 0 else 0.0


def prf2_at(t, p_nb, p_gd, y):
    pred = (p_nb >= NB_GATE) & (p_gd >= t)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, f2(prec, rec), tp, fp, fn


def best_t(p_nb, p_gd, y):
    """F2-argmax over GRID, tie-break toward HIGHER t (protocol)."""
    best = (-1.0, -1.0, None)  # (f2, t, block)
    for t in GRID:
        prec, rec, fb, tp, fp, fn = prf2_at(t, p_nb, p_gd, y)
        if fb > best[0] + 1e-12 or (abs(fb - best[0]) <= 1e-12 and t > best[1]):
            best = (fb, float(t), {"t": float(t), "precision": round(prec, 4),
                                   "recall": round(rec, 4), "f2": round(fb, 4),
                                   "tp": tp, "fp": fp, "fn": fn})
    return best[2]


def loo_f2(p_nb, p_gd, y):
    """Leave-one-out OOF F2: for each held-out location, choose t on the rest, predict the
    held-out, then score F2 over the pooled OOF predictions. Optimism estimate."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        b = best_t(p_nb[m], p_gd[m], y[m])
        t = b["t"]
        preds[i] = int((p_nb[i] >= NB_GATE) and (p_gd[i] >= t))
    tp = int(((preds == 1) & (y == 1)).sum()); fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return round(f2(prec, rec), 4)


def main():
    if not SCORES.exists():
        sys.exit(f"missing {SCORES} — run tools/v8/eval_v8.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[FT2FAM.get(r["fractal_type"], r["fractal_type"])].append(r)

    results, undecidable = {}, {}
    print("=" * 92)
    print("v8 t_good — F2-argmax (recall-weighted), per partition, REPORT ONLY (no config changed)")
    print("=" * 92)
    print(f"  decode: keeper(label>=3) iff P(>=2)>={NB_GATE} AND P(>=3)>=t   grid {GRID[0]}..{GRID[-1]}")
    print(f"  baseline {BASELINE} where <{MIN_POS} positives or no unbiased eval slice; NO class-4 t; "
          f"NO native tightening\n")
    print(f"  {'partition':20s} {'n':>4} {'pos':>4}  {'t_good':>7} {'prec':>6} {'rec':>6} "
          f"{'F2':>6} {'F2_OOF':>7}  note")

    for fam in sorted(by_fam):
        grp = by_fam[fam]
        y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
        p_nb = np.array([g["v8_p_ge2"] for g in grp])
        p_gd = np.array([g["v8_p_ge3"] for g in grp])
        n, pos = len(grp), int(y.sum())
        if pos < MIN_POS or pos == n:
            undecidable[fam] = {"n": n, "pos": pos, "t_good": BASELINE,
                                "reason": ("<%d positives" % MIN_POS) if pos < MIN_POS else "no negatives"}
            print(f"  {fam:20s} {n:>4} {pos:>4}  {BASELINE:>7.2f} {'—':>6} {'—':>6} {'—':>6} {'—':>7}"
                  f"  baseline ({undecidable[fam]['reason']})")
            continue
        b = best_t(p_nb, p_gd, y)
        oof = loo_f2(p_nb, p_gd, y)
        b.update({"n": n, "pos": pos, "f2_oof": oof, "objective": "F2-argmax"})
        results[fam] = b
        print(f"  {fam:20s} {n:>4} {pos:>4}  {b['t']:>7.2f} {b['precision']:>6.3f} "
              f"{b['recall']:>6.3f} {b['f2']:>6.3f} {oof:>7.3f}  derived (in-sample; OOF for optimism)")

    # partitions with no eval rows at all -> baseline, named
    all_fams = ["mandelbrot", "julia:mandelbrot", "multibrot3", "multibrot4", "multibrot5",
                "julia:multibrot3", "julia:multibrot4", "julia:multibrot5", "phoenix"]
    no_eval = [f for f in all_fams if f not in results and f not in undecidable]
    for f in no_eval:
        undecidable[f] = {"n": 0, "pos": 0, "t_good": BASELINE, "reason": "no unbiased eval slice"}
        print(f"  {f:20s} {0:>4} {0:>4}  {BASELINE:>7.2f} {'—':>6} {'—':>6} {'—':>6} {'—':>7}"
              f"  baseline (no unbiased eval slice)")

    out = {"objective": "F2-argmax (recall-weighted), tie-break higher t",
           "decode": f"keeper(label>=3) iff P(>=2)>={NB_GATE} AND P(>=3)>=t",
           "baseline": BASELINE, "min_pos": MIN_POS,
           "no_class4_threshold": True,
           "native_multibrot_tightening": "NOT adopted (fork-scheduled proposals left on record)",
           "derived": results, "baseline_partitions": undecidable,
           "note": ("REPORT ONLY — production_seeder.T_GOOD_OVERRIDES is unchanged. mandelbrot "
                    "now derives from the loose0_v3 floor (was an F0.5+steered exception in v7); "
                    "julia:multibrot{3,4,5} derive from the census. Everything else is baseline.")}
    from pathlib import Path as _P
    sys.path.insert(0, str(ROOT / "tools"))
    import paths  # noqa: E402
    paths.durable(OUT, mkparents=True).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} (durable) — REPORT ONLY, no production threshold changed")


if __name__ == "__main__":
    main()
