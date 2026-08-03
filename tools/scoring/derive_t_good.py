#!/usr/bin/env python
r"""THE per-partition `t_good` estimator — version-agnostic, imported by every version.

`build_table(rows, version=…)` IS the derivation: the p_good grid, the F_beta-argmax with
tie-break-toward-higher-t, the LOO-OOF optimism estimate, the plateau width, the
>=15-positive sufficiency floor and the UNCALIBRATED stamping. A per-version deriver
(`tools/<v>/derive_t_good_<v>.py`) supplies only what is genuinely its own: the eval slice,
the population rule and the re-read objective. **A copied deriver is how two thresholds that
are supposed to be comparable stop being comparable**, which is the whole reason this is one
module and not one per version.

WAS `tools/v8/derive_t_good_v8.py` until 2026-08-02. It was written for the v8 table, then
v10 imported `build_table` from it rather than copying it — at which point the module had
outlived its version and its name said otherwise, which cost a hop every time someone asked
"where is the estimator". Moved here, under the pins it serves; `tools/v8/derive_t_good_v8.py`
is now the v8 population/objective wrapper, the same shape as v10's.

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
  * A version's eval slice calibrates only the partitions it carries unbiased data for
    (v8/v10: julia:multibrot{3,4,5} from the census, mandelbrot from the loose0_v3 floor).
    Everything else -> UNCALIBRATED, named individually.

`production_seeder.T_GOOD_OVERRIDES` is the ADOPTED copy of a run's `adopted` block;
tools/scoring/test_t_good_adoption.py holds the ACTIVE version's derivation and that table
in agreement, and re-runs this estimator against the committed artifact as a drift gate.

  uv run python tools/<v>/derive_t_good_<v>.py       # the per-version entry points
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from partitions import ALL_FAMS, FT2FAM  # noqa: E402,F401  re-exported: callers take them here
from score_lib import corn_decode   # noqa: E402  THE served decode; the sweep must match it

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


def derive_partition(grp, version: str, betas=(0.5, 2.0)) -> dict:
    """Every objective's argmax + OOF for one partition, keyed 'F0.5'/'F2'.

    `version` selects the score-column prefix (`<version>_p_ge2` / `_p_ge3` / `_p_ge4`) and
    is REQUIRED — it used to default to "v8", which in a module every version now imports is
    a trap: a caller that forgets it would silently derive a v11 slice against v8's columns
    (or KeyError, if lucky). `_p_ge4` is read when the slice has it (K=4, v8 onward) because
    the SERVED decode counts it; a K=3 slice has no such column and decodes as it always
    did."""
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


def build_table(rows, version: str, eval_slice: str, objective: dict = None,
                uncal_reason: dict = None) -> dict:
    """The whole per-partition table for ONE eval slice — the derivation, not a report of it.

    Callers supply the rows (so a population rule — e.g. one instrument per partition — is
    applied by the caller and stated there, not hidden here), the score-column `version`
    prefix, the `eval_slice` the rows came from, and optionally a re-chosen `objective` map
    (protocol §4 requires the per-family objective to be re-decided from CURRENT supply, not
    inherited). `version` and `eval_slice` are REQUIRED: both used to default to v8's, and a
    shared estimator that silently stamps someone else's version onto a new table is the
    provenance failure the stamp exists to prevent.

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
        blocks = derive_partition(grp, version)
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
        "eval_slice": eval_slice,
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
