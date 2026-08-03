#!/usr/bin/env python
r"""Derive v8 per-partition t_good — the ADOPTED discovery table.

The decode is the CORN keeper rule AS SERVED: predict "keeper" (label>=3) iff
`score_lib.corn_decode(P(>=2), P(>=3), t, P(>=4)) >= 3`, i.e. at least two of the three
cutpoints met — **counting**, not chaining (see `keeper_pred`; this was an AND until
2026-08-02, which is the same predicate only on a K=3 head). Threshold search is an
F_beta-argmax over a p_good grid, per fractal partition, on the unbiased eval slice,
requiring >=15 positives; tie-break toward higher t.

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
the `adopted` block below; tools/scoring/test_t_good_adoption.py holds the ACTIVE
version's derivation and that table in agreement.

  uv run python tools/v8/derive_t_good_v8.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from score_lib import corn_decode   # noqa: E402  THE served decode; the sweep must match it

SCORES = ROOT / "data/v8/eval_scores_v8.jsonl"
OUT = "data/v8/t_good_derivation.json"

BASELINE = 0.50
MIN_POS = 15
NB_GATE = 0.5                     # the fixed P(>=2) keeper gate (corn_decode)
T_GREAT = 0.5                     # class 4's natural cutpoint — never calibrated per family
KEEPER_CLASS = 3                  # admission is decoded-class >= 3
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


def great_column(rows, version: str):
    """`<version>_p_ge4` as an array, or None when the slice has no third cutpoint.

    A K=3 slice (v5..v7) has no `p_ge4` column and `None` is the K=3 decode — byte-identical
    to the AND rule this sweep used to hardcode, so re-deriving an old slice is unchanged.
    The column is required to be present on ALL rows or none: a slice where only some rows
    carry it would sweep two different predicates over one population."""
    key = f"{version}_p_ge4"
    have = [key in r for r in rows]
    if not any(have):
        return None
    if not all(have):
        raise SystemExit(f"{key} present on {sum(have)}/{len(have)} rows — a partially K=4 "
                         f"slice would sweep two predicates over one population")
    return np.array([r[key] for r in rows])


def keeper_pred(p_nb, p_gd, p_gr, t):
    """Vectorized twin of `corn_decode(p_nb, p_gd, t, p_gr) >= KEEPER_CLASS` — THE SERVED RULE.

    The gate that runs in production is `corn_decode(...) >= 3`, and that rule **counts**
    thresholds met (`class = 1 + #{p_ge2>=0.5, p_ge3>=t, p_ge4>=t_great}`); it does not chain
    them. This sweep used to search `(p_ge2>=0.5) & (p_ge3>=t)` — an AND, which equals the
    served rule only on a **K=3** head. On K=4 (v8 onward) they diverge on exactly the rows
    where the count reaches 2 without the `p_ge3` leg: `p_ge4>=t_great` together with either
    `p_ge2>=0.5, p_ge3<t` or `p_ge3>=t, p_ge2<0.5`. Those rows are possible, not hypothetical
    — CORN's cumulative probabilities are not guaranteed monotone (the monotonicity check in
    `tools/v6/threshold_sweep.py`), and the FIRST row of `data/v10/eval_scores_v10.jsonl` has
    `p_ge4` 0.153 above `p_ge3` 0.033. Sweeping a stricter predicate than the served one picks
    an argmax for a gate that is not the gate.

    Vectorized rather than a `corn_decode` loop because LOO-OOF is O(n^2 * |GRID|) calls;
    `tools/v8/test_t_good_sweep_decode.py` holds the two forms to elementwise agreement on
    the live eval slice and on the divergence cases, so this stays a twin and not a fork."""
    cnt = (p_nb >= NB_GATE).astype(int) + (p_gd >= t).astype(int)
    if p_gr is not None:
        cnt = cnt + (p_gr >= T_GREAT).astype(int)
    return cnt >= (KEEPER_CLASS - 1)


def prf_at(t, p_nb, p_gd, p_gr, y, beta):
    pred = keeper_pred(p_nb, p_gd, p_gr, t)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, fbeta(prec, rec, beta), tp, fp, fn


def best_t(p_nb, p_gd, p_gr, y, beta):
    """F_beta-argmax over GRID, tie-break toward HIGHER t (protocol).

    Also reports the argmax PLATEAU — the contiguous run of grid steps around the winner
    that score within 1e-12 of the optimum. Tie-breaking high puts the adopted t at the
    plateau's upper edge by construction, so the plateau width is the only honest read on
    how knife-edged the pick is."""
    best = (-1.0, -1.0, None)  # (f, t, block)
    curve = []
    for t in GRID:
        prec, rec, fb, tp, fp, fn = prf_at(t, p_nb, p_gd, p_gr, y, beta)
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


def loo_fbeta(p_nb, p_gd, p_gr, y, beta):
    """Leave-one-out OOF F_beta: for each held-out location, choose t on the rest, predict the
    held-out, then score F_beta over the pooled OOF predictions. Optimism estimate."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        t = best_t(p_nb[m], p_gd[m], None if p_gr is None else p_gr[m], y[m], beta)["t"]
        preds[i] = int(corn_decode(p_nb[i], p_gd[i], t,
                                   None if p_gr is None else p_gr[i], T_GREAT) >= KEEPER_CLASS)
    tp = int(((preds == 1) & (y == 1)).sum()); fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return round(fbeta(prec, rec, beta), 4)


def derive_partition(grp, betas=(0.5, 2.0), version: str = "v8") -> dict:
    """Every objective's argmax + OOF for one partition, keyed 'F0.5'/'F2'.

    `version` selects the score-column prefix (`<version>_p_ge2` / `_p_ge3` / `_p_ge4`) and
    defaults to "v8". It exists so a NEW eval slice is re-derived through THIS estimator
    rather than a copy of it — a copied deriver is how two thresholds that are supposed to be
    comparable stop being comparable. `_p_ge4` is read when the slice has it (K=4, v8 onward)
    because the SERVED decode counts it; a K=3 slice has no such column and decodes as it
    always did."""
    y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
    p_nb = np.array([g[f"{version}_p_ge2"] for g in grp])
    p_gd = np.array([g[f"{version}_p_ge3"] for g in grp])
    p_gr = great_column(grp, version)
    out = {}
    for beta in betas:
        blk = best_t(p_nb, p_gd, p_gr, y, beta)
        blk["beta"] = beta
        blk["fbeta_oof"] = loo_fbeta(p_nb, p_gd, p_gr, y, beta)
        out[beta_name(beta)] = blk
    return out


def build_table(rows, version: str = "v8", eval_slice: str = None, objective: dict = None,
                uncal_reason: dict = None) -> dict:
    """The whole per-partition table for ONE eval slice — the derivation, not a report of it.

    Extracted verbatim from `main()` so a later version re-derives through THIS code rather
    than a copy. Callers supply the rows (so a population rule — e.g. one instrument per
    partition — is applied by the caller and stated there, not hidden here), the score-column
    `version` prefix, and optionally a re-chosen `objective` map (protocol §4 requires the
    per-family objective to be re-decided from CURRENT supply, not inherited).

    `uncal_reason` lets a caller state a MORE specific reason than "no unbiased eval slice"
    for a partition it knows something about — the difference between "we have never looked"
    and "we looked and there were no keepers" is real and is lost if both print the same
    string."""
    obj = OBJECTIVE if objective is None else objective
    uncal_reason = uncal_reason or {}
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[FT2FAM.get(r["fractal_type"], r["fractal_type"])].append(r)

    derived, uncal, adopted, notes = {}, {}, {}, []
    print("=" * 104)
    print(f"{version} t_good — PER-FAMILY objective (recall where supply is scarce, precision "
          "where it is abundant)")
    print("=" * 104)
    k4 = great_column(rows, version) is not None
    print(f"  decode: keeper(label>=3) iff corn_decode(P(>=2), P(>=3), t"
          f"{', P(>=4)' if k4 else ''}) >= {KEEPER_CLASS}  [threshold COUNTING, the served rule"
          f"{'' if k4 else '; K=3 slice, == the historical AND'}]   grid {GRID[0]}..{GRID[-1]}")
    print(f"  UNCALIBRATED -> runs at baseline {BASELINE} where <{MIN_POS} positives or no unbiased "
          f"eval slice; NO class-4 t; NO native tightening\n")
    print(f"  {'partition':20s} {'obj':>5} {'n':>4} {'pos':>4}  {'t_good':>7} {'prec':>6} {'rec':>6} "
          f"{'F':>6} {'F_OOF':>7} {'plateau':>13}  status")

    for fam in sorted(by_fam):
        grp = by_fam[fam]
        y = np.array([1 if g["label"] >= 3 else 0 for g in grp])
        n, pos = len(grp), int(y.sum())
        if pos < MIN_POS or pos == n:
            reason = uncal_reason.get(
                fam, f"<{MIN_POS} positives" if pos < MIN_POS else "no negatives")
            uncal[fam] = {"n": n, "pos": pos, "t_good": BASELINE,
                          "status": "UNCALIBRATED", "reason": reason}
            print(f"  {fam:20s} {'—':>5} {n:>4} {pos:>4}  {BASELINE:>7.2f} {'—':>6} {'—':>6} "
                  f"{'—':>6} {'—':>7} {'—':>13}  UNCALIBRATED ({reason})")
            continue

        beta = obj.get(fam, DEFAULT_BETA)
        blocks = derive_partition(grp, version=version)
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
        reason = uncal_reason.get(f, "no unbiased eval slice")
        uncal[f] = {"n": 0, "pos": 0, "t_good": BASELINE, "status": "UNCALIBRATED",
                    "reason": reason}
        print(f"  {f:20s} {'—':>5} {0:>4} {0:>4}  {BASELINE:>7.2f} {'—':>6} {'—':>6} {'—':>6} "
              f"{'—':>7} {'—':>13}  UNCALIBRATED ({reason})")

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
        "objective_by_partition": {k: beta_name(v) for k, v in obj.items()},
        "default_objective": beta_name(DEFAULT_BETA),
        "decode": (f"keeper(label>=3) iff score_lib.corn_decode(P(>=2), P(>=3), t"
                   f"{', P(>=4)' if k4 else ''}, t_great={T_GREAT}) >= {KEEPER_CLASS} — the "
                   f"SERVED rule, which COUNTS thresholds met rather than chaining them"
                   f"{'' if k4 else ' (K=3 slice: identical to the historical AND)'}"),
        "baseline": BASELINE, "min_pos": MIN_POS,
        "no_class4_threshold": True,
        "class4_decode": "natural cutpoint P(>=4) >= 0.5, no per-family calibration",
        "native_multibrot_tightening": "NOT adopted (fork-scheduled proposals left on record)",
        "eval_slice": eval_slice or "data/v8/eval_scores_v8.jsonl",
        "model": version,
        "derived": derived,
        "uncalibrated": uncal,
        "adopted": adopted,
        "notes": notes,
        "note": ("ADOPTED — production_seeder.T_GOOD_OVERRIDES mirrors `adopted`, and the "
                 "`uncalibrated` partitions are named in production_seeder.T_GOOD_UNCALIBRATED "
                 "(they RUN at the baseline but were never derived; the two 0.50s must stay "
                 "distinguishable). tools/scoring/test_t_good_adoption.py holds the ACTIVE "
                 "version's artifact and the adopted table in agreement."),
    }
    return out


def main():
    if not SCORES.exists():
        sys.exit(f"missing {SCORES} — run tools/v8/eval_v8.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = build_table(rows, version="v8", eval_slice="data/v8/eval_scores_v8.jsonl")
    sys.path.insert(0, str(ROOT / "tools"))
    import paths  # noqa: E402
    paths.durable(OUT, mkparents=True).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} (durable) — ADOPTED table; mirror into production_seeder.T_GOOD_OVERRIDES")


if __name__ == "__main__":
    main()
