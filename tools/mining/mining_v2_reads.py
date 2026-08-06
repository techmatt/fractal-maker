r"""mining_v2_reads.py — v1 vs v2 on one harness, then calibration on the winner.

`prompts/mining_v2_finetune_prompt.md` §§2-3, over the eval side of
`data/render_mode_corpus/batches/2026-08-06_render_mode_fresh_sheet_v1` (422 of 960 rows,
the side `provenance.split_side` stamped for exactly this purpose).

ONE HARNESS means one code path: every checkpoint — v1, v2-staged, and each of v2's five
seeds — is scored through `mining_gate.MiningScorer`, which is head-agnostic by construction
(backbone, K, mean/std, geometry all read from the checkpoint's own config). Nothing here
re-implements a transform or a probability. As a check that the harness IS the production
scorer, v1's re-scored `p_ge3` is compared against the `head_mining_v1.p_ge3` the sheet
builder stamped in-row back in August; a drift there would mean these numbers describe some
other scorer than the one that gates.

WHAT THE EVAL SIDE IS AND IS NOT, because it decides which way every gap leans:

  * It is genuinely held out FOR v2 — location-disjoint (union-find over Julia-parent-linked
    locations, 95 units, seed 0), and v2's trainer never saw a row of it.
  * It is NOT held out for v1. v1 trained on renders at these same 112 gate-passer
    locations (`fresh_sheet_reads.py` caveat 1), so v1 is being read on a population it has
    partly memorised.
  * The LABELS were prefilled with v1's own suggestion — this was a correction sheet, sorted
    good->bad, Enter-confirms (`batch.json` -> `labeling.mode`). Label and v1's score are
    coupled by construction (`fresh_sheet_reads.py` caveat 0).

Both of those lean the SAME way: they inflate v1 and not v2. So "v2 holds within noise of
v1" is a stronger statement than it reads, and a v1 win is partly an artifact this sitting
cannot subtract. The report says so at the top rather than in a footnote.

THE WINNER RULE, from the prompt, applied mechanically and stated in the report (no
per-slice cherry-picking; the rule is evaluated on the OVERALL eval side and the POOLED
three-dropped-mode slice, both fixed before any number was looked at):

    v2 is the calibration candidate IFF
      (a) HOLDS   — on the overall eval side, none of {AUC>=3, AP>=3, AUC>=2, AP>=2} is
                    SIGNIFICANTLY worse than v1 (95% paired-bootstrap CI on the delta does
                    not lie entirely below 0), AND
      (b) IMPROVES — on the pooled slice of the three modes v1's trainer dropped
                    (trap_circle, exp_smoothing, direct_trap_screen), at least one boundary
                    improves with its CI excluding 0, and no boundary is significantly worse.
    Otherwise v1 remains the candidate.

`noise` is a PAIRED bootstrap over eval rows (both heads re-scored on the same resample,
`BOOTSTRAP` draws, seeded) — paired because the two heads see identical rows and an unpaired
interval would be far too wide to decide anything.

Calibration then runs on the winner only: `p_ge3` threshold ladders with Wilson intervals,
the two live cuts marked for reference, and candidate cuts for BOTH sites — the release floor
as a precision question (lowest threshold buying each precision target) and the pool floor as
its mirror, a retention question (highest threshold still keeping each share of the good
rows), because `floors.py` calls the pool cut capacity ordering rather than curation.
DERIVED AND RECORDED — this file moves no pin, floor, gate or threshold, and
`mining_pins.ACTIVE_MINING_CKPT` is read, never written.

Metrics are imported, not restated: `ap`/`auc`/`md_table`/`num`/`pct` from
`tools/wallpaper/sitting_reads.py`, `wilson` from `tools/corpus/q4_combined_readout.py`,
`hanley_mcneil_se`/`min_detectable_auc` from `tools/v10/prereg.py` — the same set
`fresh_sheet_reads.py` uses, so a number here and a number there mean the same thing.

Outputs -> data/render_mode_head/v2/report.md + report.json (the run dir; the report is the
uploadable artifact).

  uv run python tools/mining/mining_v2_reads.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.corpus.q4_combined_readout import wilson                     # noqa: E402
from tools.emission import floors as F                                  # noqa: E402
from tools.mining import mining_pins as MP                              # noqa: E402
from tools.mining.mining_roster import MODE_KIND, MODES, TRAINER_DROPPED_V1  # noqa: E402
from tools.v10.prereg import hanley_mcneil_se, min_detectable_auc       # noqa: E402
from tools.wallpaper.sitting_reads import ap, auc, md_table, num, pct   # noqa: E402

BATCH_DIR = (ROOT / "data" / "render_mode_corpus" / "batches"
             / "2026-08-06_render_mode_fresh_sheet_v1")
V1_CKPT = "data/render_mode_head/v1/model_best.pt"
V2_DIR = ROOT / "data" / "render_mode_head" / "v2"
# v2's weights were DE-TRACKED after this sitting decided against them (2026-08-06), so a
# fresh clone has the report but not the checkpoint: re-running this file means re-running
# `classifier/train_mining_head_v2.py` first. `main()` already raises naming the missing
# path rather than scoring one head and calling it a comparison.
V2_CKPT = "data/render_mode_head/v2/model_best.pt"
OUT = V2_DIR

K_TIERS = 3
GOOD = 3

# Paired-bootstrap draws. 4,000 puts the Monte-Carlo error on a 95% interval endpoint well
# under 0.005 AUC at n=422 — smaller than any difference this sitting could act on.
BOOTSTRAP = 4000
BOOT_SEED = 20260806

# The ladder grid. The two live cuts are UNIONED in so they are exact rows, never
# nearest-bin — the same construction fresh_sheet_reads.py uses.
SWEEP = sorted({round(x, 3) for x in np.arange(0.0, 1.0, 0.05)}
               | {F.MINING_POOL.value, F.MINING_RELEASE.value})
# Two target sets, one per SITE. The release floor is a precision question ("what can ship");
# the pool floor is a retention question ("what must not be discarded before selection").
PRECISION_TARGETS = (0.70, 0.80, 0.90)
RECALL_TARGETS = (0.95, 0.90, 0.80)

# Fixed pass-rate points for the volume-matched comparison (see `volume_matched`).
VOLUME_RATES = (0.05, 0.10, 0.20)

# The four overall metrics the winner rule's clause (a) is evaluated on. Fixed here, above
# the code that computes them, so the set cannot be chosen after the fact.
OVERALL_METRICS = (("auc_ge3", "AUC >=3"), ("ap_ge3", "AP >=3"),
                   ("auc_ge2", "AUC >=2"), ("ap_ge2", "AP >=2"))


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
def load_eval_rows(batch_dir: Path = BATCH_DIR) -> list[dict]:
    """The eval-side rows, labels and split both read in-row. Raises on an unlabeled row."""
    rows, unlabeled = [], []
    for line in (batch_dir / "images.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pv = r["provenance"]
        if pv["split_side"] != "eval":
            continue
        if r["label"]["score"] is None:
            unlabeled.append(r["image_id"])
            continue
        h = r["head_mining_v1"]
        rows.append({
            "id": r["image_id"], "crop": batch_dir / "crops" / f"{r['image_id']}.jpg",
            "label": int(r["label"]["score"]), "suggested": int(r["suggested_tier"]),
            "mode": pv["render_mode"], "kind": pv["mode_kind"], "family": pv["family"],
            "loc": pv["location_key"], "order": int(r["sheet_order"]),
            "stamped_v1_p_ge3": float(h["p_ge3"]), "stamped_v1_p_ge2": float(h["p_ge2"]),
        })
    if unlabeled:
        raise SystemExit(f"[mining-v2-reads] {len(unlabeled)} eval rows are unlabeled "
                         f"(e.g. {unlabeled[:3]}) — this readout describes a COMPLETE slice.")
    return rows


def score_with(ckpt: str, rows: list[dict]) -> dict[str, np.ndarray]:
    """`{p_ge2, p_ge3, rank}` for one checkpoint, through the production scorer."""
    from tools.mining.mining_gate import MiningScorer      # noqa: PLC0415 (torch import)
    sc = MiningScorer(model_path=ckpt)
    res = sc.score_paths([r["crop"] for r in rows])
    return {"p_ge2": np.array([x.p_ge2 for x in res]),
            "p_ge3": np.array([x.p_ge3 for x in res]),
            "rank": np.array([x.score for x in res])}


# --------------------------------------------------------------------------- #
# metric blocks
# --------------------------------------------------------------------------- #
def tier_dist(labels) -> dict:
    c = Counter(int(x) for x in labels)
    n = len(labels)
    return {"n": n, "hist": {str(t): c.get(t, 0) for t in range(1, K_TIERS + 1)},
            "frac_ge2": (n - c.get(1, 0)) / n if n else None,
            "frac_ge3": c.get(3, 0) / n if n else None}


def boundary(labels: np.ndarray, s: np.ndarray, thr: int) -> dict:
    """AUC + AP at one tier boundary, with the power this cell actually has.

    `min_detectable` is the smallest AUC whose 95% Hanley-McNeil interval clears 0.50 at
    this cell's (n_pos, n_neg) — so a 26-row mode and the 422-row eval side are each judged
    against their own power instead of a shared eyeball."""
    y = (labels >= thr)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    out = {"n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
           "base_rate": n_pos / len(y) if len(y) else None,
           "measurable": bool(n_pos > 0 and n_neg > 0)}
    if not out["measurable"]:
        out.update(auc=None, ap=None, auc_se=None, auc_lo=None, auc_hi=None,
                   min_detectable=None, at_chance=None)
        return out
    a = auc(y.tolist(), s.tolist())
    se = hanley_mcneil_se(a, n_pos, n_neg)
    out.update(auc=a, ap=ap(y.tolist(), s.tolist()), auc_se=se,
               auc_lo=a - 1.96 * se, auc_hi=a + 1.96 * se,
               min_detectable=min_detectable_auc(n_pos, n_neg),
               at_chance=bool(a - 1.96 * se <= 0.50))
    return out


def four_metrics(labels: np.ndarray, sc: dict) -> dict:
    """The four overall numbers the winner rule reads, on one head and one slice."""
    b3 = boundary(labels, sc["p_ge3"], 3)
    b2 = boundary(labels, sc["p_ge2"], 2)
    return {"auc_ge3": b3["auc"], "ap_ge3": b3["ap"],
            "auc_ge2": b2["auc"], "ap_ge2": b2["ap"],
            "ge3": b3, "ge2": b2,
            "ge3_on_rank": boundary(labels, sc["rank"], 3)}


# --------------------------------------------------------------------------- #
# paired bootstrap — the definition of "within noise"
# --------------------------------------------------------------------------- #
def _metric(labels, s, thr, kind):
    y = (labels >= thr)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    return (auc(y.tolist(), s.tolist()) if kind == "auc" else ap(y.tolist(), s.tolist()))


def paired_bootstrap(labels: np.ndarray, a: dict, b: dict, *, draws=BOOTSTRAP,
                     seed=BOOT_SEED) -> dict:
    """95% CI on (b - a) for each of the four metrics, resampling ROWS in common.

    Paired: one resample indexes both heads, so the (large) row-difficulty variance cancels
    and what is left is the difference between the heads. An unpaired interval at n=422 is
    wide enough to call everything "within noise", which would make the winner rule vacuous.

    A draw in which a boundary has no positives (or no negatives) yields no value for that
    metric; `n_draws` per metric records how many survived, so a CI computed from a thinned
    set says so."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    acc = {k: [] for k, _ in OVERALL_METRICS}
    spec = {"auc_ge3": ("p_ge3", 3, "auc"), "ap_ge3": ("p_ge3", 3, "ap"),
            "auc_ge2": ("p_ge2", 2, "auc"), "ap_ge2": ("p_ge2", 2, "ap")}
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        for key, (col, thr, kind) in spec.items():
            va = _metric(lb, a[col][idx], thr, kind)
            vb = _metric(lb, b[col][idx], thr, kind)
            if va is not None and vb is not None:
                acc[key].append(vb - va)
    out = {}
    for key, _lab in OVERALL_METRICS:
        d = np.asarray(acc[key], dtype=float)
        if d.size == 0:
            out[key] = {"n_draws": 0, "lo": None, "hi": None, "median": None,
                        "significantly_worse": None, "significantly_better": None}
            continue
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        out[key] = {"n_draws": int(d.size), "lo": lo, "hi": hi,
                    "median": float(np.median(d)),
                    "significantly_worse": bool(hi < 0.0),
                    "significantly_better": bool(lo > 0.0)}
    return out


# --------------------------------------------------------------------------- #
# score-scale check + volume matching
# --------------------------------------------------------------------------- #
def _ks_q(lam: float) -> float:
    """`Q_KS(λ)` — the asymptotic two-sided KS tail, Numerical Recipes `probks`.

    The textbook one-liner `2 Σ (-1)^(j-1) exp(-2 j² λ²)` is an asymptotic form that is only
    usable for λ above ~0.5; at λ = 0 (two IDENTICAL distributions) its alternating terms are
    all 1 and a 100-term truncation sums to 0, reporting p = 0 — "certainly different" for a
    sample compared against itself. So the series is summed with a convergence test and
    SATURATES AT 1.0 when it does not converge, which is the small-λ answer."""
    if lam < 1e-3:
        return 1.0
    a2, fac, total, termbf = -2.0 * lam * lam, 2.0, 0.0, 0.0
    for j in range(1, 101):
        term = fac * np.exp(a2 * j * j)
        total += term
        if abs(term) <= 1e-3 * termbf or abs(term) <= 1e-8 * total:
            return float(min(max(total, 0.0), 1.0))
        fac, termbf = -fac, abs(term)
    return 1.0


def scale_shift(a: dict, b: dict) -> dict:
    """Has v2's marginal `p_ge3` moved off v1's scale? Distribution, not performance.

    A fixed threshold is a point on ONE head's probability scale (`floors.py`'s whole
    stamp argument). If the two marginals differ, `p_ge3 >= 0.50` selects different VOLUMES
    from the two heads and a fixed-threshold table compares two different operating points —
    which is why the volume-matched view below exists alongside it."""
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    x, y = np.sort(a["p_ge3"]), np.sort(b["p_ge3"])
    # two-sample KS statistic, computed directly (no scipy dependency added for one number)
    grid = np.union1d(x, y)
    cdf_a = np.searchsorted(x, grid, side="right") / len(x)
    cdf_b = np.searchsorted(y, grid, side="right") / len(y)
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(len(x) * len(y) / (len(x) + len(y)))
    p = _ks_q((en + 0.12 + 0.11 / en) * ks)
    return {
        "quantiles": {f"q{int(q*100)}": {"v1": float(np.quantile(a["p_ge3"], q)),
                                         "v2": float(np.quantile(b["p_ge3"], q))} for q in qs},
        "mean": {"v1": float(a["p_ge3"].mean()), "v2": float(b["p_ge3"].mean())},
        "pass_rate_at": {f"{t:.2f}": {"v1": float((a["p_ge3"] >= t).mean()),
                                      "v2": float((b["p_ge3"] >= t).mean())}
                         for t in (F.MINING_POOL.value, F.MINING_RELEASE.value, 0.75, 0.90)},
        "ks": ks, "ks_p": float(p), "shifted": bool(p < 0.05),
    }


def top_n_block(labels: np.ndarray, s: np.ndarray, n_take: int, thr: int = GOOD) -> dict:
    """Precision/recall of the top `n_take` rows by score — the volume-matched cell.

    Ties at the cut are broken by taking the highest-scoring `n_take` after a stable sort;
    with continuous marginals a tie is measure-zero, and `n_selected` reports what was
    actually taken so a degenerate head cannot hide behind the requested volume."""
    n_take = int(min(max(n_take, 0), len(s)))
    order = np.argsort(-s, kind="stable")[:n_take]
    good = int((labels >= thr).sum())
    k = int((labels[order] >= thr).sum())
    p, lo, hi = wilson(k, n_take) if n_take else (None, None, None)
    return {"n_selected": n_take, "pass_rate": n_take / len(s) if len(s) else None,
            "tp": k, "precision": p, "precision_lo": lo, "precision_hi": hi,
            "recall": k / good if good else None,
            "cut_at": float(s[order[-1]]) if n_take else None}


def volume_matched(labels: np.ndarray, a: dict, b: dict) -> dict:
    """Both heads at EQUAL selected volume — the comparison a scale shift does not corrupt.

    Volumes are (i) whatever v1 passes at each live cut, so "same number of renders as today"
    is directly comparable, and (ii) three fixed pass rates, so the comparison does not
    depend on v1's own calibration at all."""
    n = len(labels)
    out = {"by_v1_live_cut": {}, "by_fixed_rate": {}}
    for f in (F.MINING_POOL, F.MINING_RELEASE):
        v = int((a["p_ge3"] >= f.value).sum())
        out["by_v1_live_cut"][f.name] = {
            "threshold_on_v1": f.value, "matched_volume": v,
            "v1": top_n_block(labels, a["p_ge3"], v),
            "v2": top_n_block(labels, b["p_ge3"], v)}
    for rate in VOLUME_RATES:
        v = int(round(rate * n))
        out["by_fixed_rate"][f"{rate:.2f}"] = {
            "matched_volume": v,
            "v1": top_n_block(labels, a["p_ge3"], v),
            "v2": top_n_block(labels, b["p_ge3"], v)}
    return out


# --------------------------------------------------------------------------- #
# the winner rule
# --------------------------------------------------------------------------- #
RULE_TEXT = (
    "v2 is the calibration candidate iff (a) no overall eval metric is significantly worse "
    "than v1 (95% paired-bootstrap CI on the delta not entirely below 0) AND (b) on the "
    "pooled three-dropped-mode slice at least one boundary is significantly BETTER and none "
    "is significantly worse. Evaluated on the two pre-declared slices; no per-slice "
    "cherry-picking, and the losing head keeps the candidacy rather than the tie being "
    "resolved by whichever number looks best.")


def apply_winner_rule(overall_ci: dict, dropped_ci: dict) -> dict:
    """The prompt's rule, as a pure function of the two CI blocks. Separate from `build` so
    it can be exercised on constructed verdicts — a rule only ever evaluated on the one real
    outcome is a rule whose other branch has never run.

    Clause (b) reads only the boundaries the pooled slice can MEASURE (`n_draws > 0`); a
    boundary with no positives contributes neither an improvement nor a regression, and
    silently treating it as either is how an unmeasurable cell decides an adoption."""
    holds = {k: not bool(overall_ci[k]["significantly_worse"]) for k, _ in OVERALL_METRICS}
    measurable = [k for k, _ in OVERALL_METRICS if dropped_ci[k]["n_draws"] > 0]
    improves_any = any(dropped_ci[k]["significantly_better"] for k in measurable)
    worse_any = any(dropped_ci[k]["significantly_worse"] for k in measurable)
    a_pass, b_pass = all(holds.values()), (improves_any and not worse_any)
    winner = "v2" if (a_pass and b_pass) else "v1"
    return {
        "rule": RULE_TEXT,
        "clause_a_holds": holds, "clause_a_pass": bool(a_pass),
        "clause_b_measurable": measurable,
        "clause_b_improves_any": bool(improves_any),
        "clause_b_worse_any": bool(worse_any),
        "clause_b_pass": bool(b_pass),
        "winner": winner,
        "calibration_candidate_ckpt": V2_CKPT if winner == "v2" else V1_CKPT,
    }


# --------------------------------------------------------------------------- #
# calibration (winner only)
# --------------------------------------------------------------------------- #
def ladder(labels: np.ndarray, s: np.ndarray, *, thr=GOOD, grid=SWEEP) -> list:
    good = int((labels >= thr).sum())
    out = []
    for t in grid:
        fire = s >= t
        nf = int(fire.sum())
        k = int((labels[fire] >= thr).sum())
        p, lo, hi = wilson(k, nf) if nf else (None, None, None)
        out.append({"threshold": float(t), "fires": nf,
                    "pass_rate": nf / len(s) if len(s) else None, "tp": k,
                    "precision": p, "precision_lo": lo, "precision_hi": hi,
                    "recall": k / good if good else None,
                    "marks": [f.name for f in (F.MINING_POOL, F.MINING_RELEASE)
                              if abs(f.value - t) < 1e-9]})
    return out


def recall_candidates(lad, targets=RECALL_TARGETS) -> dict:
    """HIGHEST swept threshold still retaining each recall target — the POOL floor's question.

    The pool cut is capacity ordering, not curation (`floors.py`: "strange colorizes are
    cheap to make and expensive to carry"), so asking it for a precision target is asking the
    wrong question: what it must not do is discard keepers. This is therefore the mirror of
    `candidates` — highest rather than lowest, recall rather than precision — and the two
    tables are labelled with the site each one answers for."""
    out = {}
    for t in targets:
        hit = next((r for r in reversed(lad)
                    if r["recall"] is not None and r["recall"] >= t), None)
        out[f"{t:.2f}"] = None if hit is None else {
            "threshold": hit["threshold"], "recall": hit["recall"],
            "precision": hit["precision"], "precision_lo": hit["precision_lo"],
            "precision_hi": hit["precision_hi"], "fires": hit["fires"],
            "pass_rate": hit["pass_rate"], "tp": hit["tp"]}
    return out


def candidates(lad, targets=PRECISION_TARGETS) -> dict:
    """Lowest swept threshold whose precision POINT ESTIMATE reaches each target.

    `supported` is whether the Wilson LOWER bound clears it too. A target met only on the
    point estimate, off a handful of passers, is a cut this eval side cannot buy — printing
    the bound is what makes that visible instead of arguable."""
    out = {}
    for t in targets:
        hit = next((r for r in lad if r["precision"] is not None and r["precision"] >= t), None)
        out[f"{t:.2f}"] = None if hit is None else {
            "threshold": hit["threshold"], "precision": hit["precision"],
            "precision_lo": hit["precision_lo"], "precision_hi": hit["precision_hi"],
            "recall": hit["recall"], "fires": hit["fires"], "tp": hit["tp"],
            "supported": bool(hit["precision_lo"] is not None and hit["precision_lo"] >= t)}
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(rows, labels, heads, seed_scores) -> dict:                     # noqa: PLR0915
    v1, v2 = heads["v1"], heads["v2"]
    modes = np.array([r["mode"] for r in rows])

    R = {
        "batch": BATCH_DIR.name,
        "slice": "eval side (provenance.split_side == 'eval'); NOT re-derived",
        "n_eval": len(rows), "n_locations": len({r["loc"] for r in rows}),
        "n_modes": int(len(set(modes))),
        "label_dist": tier_dist(labels),
        "heads": {"v1": {"ckpt": V1_CKPT, "role": "incumbent (LIVE pin)"},
                  "v2": {"ckpt": V2_CKPT, "role": "finetune candidate"}},
        "live_pin": MP.ACTIVE_MINING_CKPT,
        "cuts": {f.name: {"value": f.value, "acts": f.acts, "stamp": f"{f.head}/{f.stamp}"}
                 for f in (F.MINING_POOL, F.MINING_RELEASE)},
        "bootstrap": {"draws": BOOTSTRAP, "seed": BOOT_SEED, "kind": "paired over eval rows"},
        "caveats": {
            "eval_is_held_out_for_v2_only": "location-disjoint and unseen by v2's trainer; "
                                            "v1 trained on renders at these same 112 "
                                            "gate-passer locations, so v1 is read on a "
                                            "population it has partly memorised.",
            "labels_are_anchored_to_v1": "correction sheet — every row was served with v1's "
                                         "suggested tier prefilled, sorted good->bad, Enter "
                                         "confirming. label and v1's score are coupled by "
                                         "construction.",
            "direction": "BOTH caveats inflate v1 and neither touches v2. A v2 win is "
                         "understated; a v1 win is partly an artifact this sitting cannot "
                         "subtract.",
            "staged_is_eval_selected": "v2's staged checkpoint is the best of 5 seeds BY "
                                       "eval AP>=3 on this very slice, so the staged number "
                                       "is optimistic. The 5-seed band is reported beside it "
                                       "and is the honest read.",
        },
    }

    # ---- harness parity: is this the scorer that gates? --------------------- #
    stamped = np.array([r["stamped_v1_p_ge3"] for r in rows])
    d = np.abs(stamped - v1["p_ge3"])
    R["harness_parity"] = {
        "what": "v1 re-scored here vs head_mining_v1.p_ge3 stamped into images.jsonl when "
                "the sheet was built. Same checkpoint, same deploy transform, months apart.",
        "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
        "n": int(len(d)), "tol": 1e-6, "ok": bool(d.max() < 1e-6)}

    # ---- overall, per head --------------------------------------------------- #
    R["overall"] = {"v1": four_metrics(labels, v1), "v2": four_metrics(labels, v2)}
    R["delta_ci"] = paired_bootstrap(labels, v1, v2)

    # v2's 5-seed band — the staged checkpoint is eval-selected, this is not.
    seed_rows = []
    for s, sc in sorted(seed_scores.items()):
        m = four_metrics(labels, sc)
        seed_rows.append({"seed": s, **{k: m[k] for k, _ in OVERALL_METRICS}})
    R["v2_seed_band"] = {
        "per_seed": seed_rows,
        "mean_sd": {k: {"mean": float(np.mean([r[k] for r in seed_rows])),
                        "sd": float(np.std([r[k] for r in seed_rows], ddof=0))}
                    for k, _ in OVERALL_METRICS} if seed_rows else {},
        "note": "each seed's OWN best-epoch checkpoint, selected on this eval side; the "
                "band is over seeds, not over held-out populations."}

    # ---- per mode ------------------------------------------------------------ #
    per_mode = {}
    for m in MODES:
        mask = modes == m
        lb = labels[mask]
        cell = {"kind": MODE_KIND[m], "n": int(mask.sum()),
                "untrained_by_v1": m in TRAINER_DROPPED_V1,
                "tiers": tier_dist(lb)}
        for head, sc in (("v1", v1), ("v2", v2)):
            cell[head] = {"ge3": boundary(lb, sc["p_ge3"][mask], 3),
                          "ge2": boundary(lb, sc["p_ge2"][mask], 2)}
        for b in ("ge3", "ge2"):
            for met in ("auc", "ap"):
                a_, b_ = cell["v1"][b][met], cell["v2"][b][met]
                cell.setdefault("delta", {})[f"{b}_{met}"] = (
                    None if (a_ is None or b_ is None) else float(b_ - a_))
        per_mode[m] = cell
    R["per_mode"] = per_mode

    # ---- the three modes v1's trainer never saw ------------------------------- #
    dm = list(TRAINER_DROPPED_V1)
    pooled = np.isin(modes, dm)
    lbp = labels[pooled]
    dropped = {"modes": dm, "n_pooled": int(pooled.sum()), "tiers": tier_dist(lbp),
               "per_mode": {m: per_mode[m] for m in dm},
               "pooled": {"v1": four_metrics(lbp, {k: v[pooled] for k, v in v1.items()}),
                          "v2": four_metrics(lbp, {k: v[pooled] for k, v in v2.items()})}}
    dropped["pooled_delta_ci"] = paired_bootstrap(
        lbp, {k: v[pooled] for k, v in v1.items()}, {k: v[pooled] for k, v in v2.items()})
    R["dropped_modes"] = dropped

    # ---- score scale + volume matching --------------------------------------- #
    R["scale_shift"] = scale_shift(v1, v2)
    R["volume_matched"] = volume_matched(labels, v1, v2)

    # ---- the winner rule, applied -------------------------------------------- #
    R["winner_rule"] = apply_winner_rule(R["delta_ci"], dropped["pooled_delta_ci"])
    winner = R["winner_rule"]["winner"]

    # ---- calibration, on the winner ONLY -------------------------------------- #
    w = v2 if winner == "v2" else v1
    lad3 = ladder(labels, w["p_ge3"], thr=3)
    lad2 = ladder(labels, w["p_ge2"], thr=2)
    R["calibration"] = {
        "on": winner, "ckpt": R["winner_rule"]["calibration_candidate_ckpt"],
        "n": len(rows),
        "base_rate_ge3": R["label_dist"]["frac_ge3"],
        "base_rate_ge2": R["label_dist"]["frac_ge2"],
        "ladder_ge3": lad3, "ladder_ge2": lad2,
        "at_live_cuts": {f.name: next(r for r in lad3 if abs(r["threshold"] - f.value) < 1e-9)
                         for f in (F.MINING_POOL, F.MINING_RELEASE)},
        "release_candidates_ge3": candidates(lad3),
        "pool_candidates_ge3": recall_candidates(lad3),
        "candidates_ge2": candidates(lad2),
        "adopted": None,
        "note": "DERIVED AND RECORDED, ADOPTED NOTHING. The two live cuts are marked for "
                "reference only. A cut set from this slice inherits both caveats above and "
                "would be an optimistic bound on a fresh location.",
    }
    return R


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def d_ci(c) -> str:
    if c["lo"] is None:
        return "—"
    tag = " **worse**" if c["significantly_worse"] else (
        " **better**" if c["significantly_better"] else "")
    return f"{c['median']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{tag}"


def write_md(R) -> None:                                                 # noqa: PLR0915
    w = [f"# mining head v2 — finetune from v1, eval + calibration\n",
         f"Batch `{R['batch']}` · **eval side, n = {R['n_eval']}** "
         f"({R['n_locations']} locations, all {R['n_modes']} roster modes) · "
         f"labels {R['label_dist']['hist']} "
         f"(>=3 base rate **{pct(R['label_dist']['frac_ge3'])}**, "
         f">=2 **{pct(R['label_dist']['frac_ge2'])}**).\n",
         f"\n`v1` = `{R['heads']['v1']['ckpt']}` (the LIVE pin, "
         f"`mining_pins.ACTIVE_MINING_CKPT`) · `v2` = `{R['heads']['v2']['ckpt']}` "
         f"(finetuned from v1 on this batch's train side, 538 rows).\n",
         "\n**Report only. No pin flip, no floor move, no gate change.** The two live cuts "
         f"(pool **{R['cuts']['mining_pool']['value']:.2f}**, acting; release "
         f"**{R['cuts']['mining_release']['value']:.2f}**, report-only) are marked for "
         "reference "
         "and are not touched.\n",
         "\n## 0 · which way every gap leans\n"]
    for k, v in R["caveats"].items():
        w.append(f"- **{k}** — {v}")
    hp = R["harness_parity"]
    w.append(f"\n**Harness parity.** {hp['what']} Max abs diff over {hp['n']} rows: "
             f"**{hp['max_abs_diff']:.2e}** (mean {hp['mean_abs_diff']:.2e}); tolerance "
             f"{hp['tol']:.0e} — **{'PASS' if hp['ok'] else 'FAIL'}**. Both heads are scored "
             f"through this same path.\n")

    # 1 — overall
    o1, o2, dc = R["overall"]["v1"], R["overall"]["v2"], R["delta_ci"]
    w.append(f"\n## 1 · overall, eval side (n = {R['n_eval']})\n")
    w.append(f"AUC/AP at each tier boundary, each on the marginal probability that "
             f"boundary's gate uses. `Δ` is v2 − v1 with a 95% **paired** bootstrap CI "
             f"({R['bootstrap']['draws']} draws, seed {R['bootstrap']['seed']}) — paired "
             f"because both heads score identical rows.\n")
    w.append(md_table(
        ["boundary", "n pos", "base", "v1", "v2", "Δ (v2 − v1), 95% CI"],
        [[lab, o1[key[-3:]]["n_pos"], pct(o1[key[-3:]]["base_rate"]),
          num(o1[key]), num(o2[key]), d_ci(dc[key])]
         for key, lab in OVERALL_METRICS]))
    w.append(f"\nRank-score (Σσ) at >=3: v1 AUC {num(o1['ge3_on_rank']['auc'])}, "
             f"v2 {num(o2['ge3_on_rank']['auc'])}. Smallest AUC distinguishable from 0.50 "
             f"at this n: {num(o1['ge3']['min_detectable'])} (>=3), "
             f"{num(o1['ge2']['min_detectable'])} (>=2).\n")
    sb = R["v2_seed_band"]
    if sb["per_seed"]:
        w.append(f"\n**v2's five seeds** (the staged checkpoint is the best of these BY "
                 f"eval AP>=3 on this slice, so the staged row above is optimistic; this "
                 f"band is not).\n")
        w.append(md_table(
            ["seed"] + [lab for _, lab in OVERALL_METRICS],
            [[r["seed"]] + [num(r[k]) for k, _ in OVERALL_METRICS] for r in sb["per_seed"]]
            + [["**mean ± SD**"] + [f"{sb['mean_sd'][k]['mean']:.3f} ± "
                                    f"{sb['mean_sd'][k]['sd']:.3f}" for k, _ in OVERALL_METRICS]]))
        w.append("")

    # 2 — per mode
    w.append(f"\n## 2 · per-mode, eval side\n")
    w.append("`v1 saw?` is whether v1's TRAINER included the mode — a low AUC on a mode v1 "
             "never trained on is a gap, not a failure. `—` means the boundary is not "
             "measurable in this cell (no positive or no negative at that boundary), which "
             "is a stronger statement than \"at chance\", not a missing one.\n")
    pm = R["per_mode"]
    w.append(md_table(
        ["mode", "kind", "v1 saw?", "n", "1", "2", "3",
         "v1 AUC>=3", "v2 AUC>=3", "Δ", "v1 AUC>=2", "v2 AUC>=2", "Δ"],
        [[m, c["kind"], "NO" if c["untrained_by_v1"] else "yes", c["n"],
          c["tiers"]["hist"]["1"], c["tiers"]["hist"]["2"], c["tiers"]["hist"]["3"],
          num(c["v1"]["ge3"]["auc"]), num(c["v2"]["ge3"]["auc"]),
          "—" if c["delta"]["ge3_auc"] is None else f"{c['delta']['ge3_auc']:+.3f}",
          num(c["v1"]["ge2"]["auc"]), num(c["v2"]["ge2"]["auc"]),
          "—" if c["delta"]["ge2_auc"] is None else f"{c['delta']['ge2_auc']:+.3f}"]
         for m, c in pm.items()]))
    w.append("")
    w.append(md_table(
        ["mode", "v1 AP>=3", "v2 AP>=3", "Δ", "v1 AP>=2", "v2 AP>=2", "Δ"],
        [[m, num(c["v1"]["ge3"]["ap"]), num(c["v2"]["ge3"]["ap"]),
          "—" if c["delta"]["ge3_ap"] is None else f"{c['delta']['ge3_ap']:+.3f}",
          num(c["v1"]["ge2"]["ap"]), num(c["v2"]["ge2"]["ap"]),
          "—" if c["delta"]["ge2_ap"] is None else f"{c['delta']['ge2_ap']:+.3f}"]
         for m, c in pm.items()]))

    # 3 — the dropped three
    dm = R["dropped_modes"]
    w.append(f"\n## 3 · the three modes v1's trainer dropped\n")
    w.append(f"`{'`, `'.join(dm['modes'])}` — v1 never trained on a row of any of them; v2 "
             f"trained on all three. Pooled eval slice: **n = {dm['n_pooled']}**, labels "
             f"{dm['tiers']['hist']} (>=3 base {pct(dm['tiers']['frac_ge3'])}).\n")
    w.append("\n**Individually.**\n")
    for m in dm["modes"]:
        c = dm["per_mode"][m]
        t = c["tiers"]
        rich = (c["v1"]["ge3"]["measurable"] and t["hist"]["3"] >= 10)
        w.append(f"\n- **`{m}`** (n={c['n']}, labels {t['hist']}) — "
                 + ("**the rich one: this mode is the qualitative pass/fail of the finetune.** "
                    if rich else "")
                 + (f">=3: v1 AUC {num(c['v1']['ge3']['auc'])} / AP {num(c['v1']['ge3']['ap'])} "
                    f"→ v2 AUC {num(c['v2']['ge3']['auc'])} / AP {num(c['v2']['ge3']['ap'])}. "
                    if c["v1"]["ge3"]["measurable"] else
                    ">=3 is **not measurable** here (no labeled tier-3 on the eval side), so "
                    "the finetune's effect on this mode can only be read at >=2. ")
                 + (f">=2: v1 AUC {num(c['v1']['ge2']['auc'])} / AP {num(c['v1']['ge2']['ap'])} "
                    f"→ v2 AUC {num(c['v2']['ge2']['auc'])} / AP {num(c['v2']['ge2']['ap'])}."
                    if c["v1"]["ge2"]["measurable"] else ">=2 is not measurable here either."))
    w.append(f"\n\n**Pooled, with paired CIs** (the slice the winner rule's clause (b) reads):\n")
    p1, p2, pci = dm["pooled"]["v1"], dm["pooled"]["v2"], dm["pooled_delta_ci"]
    w.append(md_table(
        ["metric", "n pos", "v1", "v2", "Δ (v2 − v1), 95% CI"],
        [[lab, p1[key[-3:]]["n_pos"], num(p1[key]), num(p2[key]), d_ci(pci[key])]
         for key, lab in OVERALL_METRICS]))
    w.append("")

    # 4 — score scale
    ss = R["scale_shift"]
    w.append(f"\n## 4 · score-scale check\n")
    w.append(f"A fixed threshold is a point on ONE head's probability scale. If the marginals "
             f"differ, `p_ge3 >= 0.50` selects different VOLUMES from the two heads and a "
             f"fixed-threshold table is comparing two different operating points.\n")
    w.append(f"\nTwo-sample KS on eval `p_ge3`: **D = {ss['ks']:.3f}, p = {ss['ks_p']:.2e}** — "
             f"the marginal distribution **{'HAS' if ss['shifted'] else 'has NOT'}** shifted."
             f"{' The volume-matched view below is therefore the load-bearing comparison; the fixed-threshold view is kept beside it.' if ss['shifted'] else ' The fixed-threshold comparison is directly meaningful; the volume-matched view is included anyway as a cross-check.'}\n")
    w.append(md_table(
        ["quantile of p_ge3", "v1", "v2"],
        [[q, num(v["v1"], 4), num(v["v2"], 4)] for q, v in ss["quantiles"].items()]
        + [["mean", num(ss["mean"]["v1"], 4), num(ss["mean"]["v2"], 4)]]))
    w.append("\n**Fixed thresholds — pass rate of each head**\n")
    w.append(md_table(["p_ge3 >=", "v1 pass rate", "v2 pass rate"],
                      [[t, pct(v["v1"]), pct(v["v2"])] for t, v in ss["pass_rate_at"].items()]))
    vm = R["volume_matched"]
    w.append("\n**Volume-matched — both heads take the SAME number of rows**\n")
    w.append(md_table(
        ["matched at", "volume", "v1 precision", "v1 recall", "v2 precision", "v2 recall",
         "v2 cut on p_ge3"],
        [[f"v1 @ {c['threshold_on_v1']:.2f} ({name})", c["matched_volume"],
          f"{pct(c['v1']['precision'])} [{pct(c['v1']['precision_lo'])}–{pct(c['v1']['precision_hi'])}]",
          pct(c["v1"]["recall"]),
          f"{pct(c['v2']['precision'])} [{pct(c['v2']['precision_lo'])}–{pct(c['v2']['precision_hi'])}]",
          pct(c["v2"]["recall"]), num(c["v2"]["cut_at"], 4)]
         for name, c in vm["by_v1_live_cut"].items()]
        + [[f"fixed {float(rate)*100:.0f}% pass rate", c["matched_volume"],
            f"{pct(c['v1']['precision'])} [{pct(c['v1']['precision_lo'])}–{pct(c['v1']['precision_hi'])}]",
            pct(c["v1"]["recall"]),
            f"{pct(c['v2']['precision'])} [{pct(c['v2']['precision_lo'])}–{pct(c['v2']['precision_hi'])}]",
            pct(c["v2"]["recall"]), num(c["v2"]["cut_at"], 4)]
           for rate, c in vm["by_fixed_rate"].items()]))
    w.append("")

    # 5 — winner
    wr = R["winner_rule"]
    w.append(f"\n## 5 · the winner rule, applied\n")
    w.append(f"> {wr['rule']}\n")
    w.append(f"\n**(a) overall — no metric significantly worse:** "
             + ", ".join(f"{lab} {'OK' if wr['clause_a_holds'][k] else '**FAIL**'}"
                         for k, lab in OVERALL_METRICS)
             + f" → **{'PASS' if wr['clause_a_pass'] else 'FAIL'}**.\n")
    w.append(f"\n**(b) dropped modes improve:** measurable boundaries "
             f"{wr['clause_b_measurable']}; at least one significantly better: "
             f"**{wr['clause_b_improves_any']}**; any significantly worse: "
             f"**{wr['clause_b_worse_any']}** → "
             f"**{'PASS' if wr['clause_b_pass'] else 'FAIL'}**.\n")
    w.append(f"\n### → the calibration candidate is **{wr['winner']}** "
             f"(`{wr['calibration_candidate_ckpt']}`)\n")

    # 6 — calibration
    cal = R["calibration"]
    w.append(f"\n## 6 · calibration on the winner ({cal['on']}), eval side (n = {cal['n']})\n")
    w.append(f">=3 base rate **{pct(cal['base_rate_ge3'])}**. Precision is of PASSERS and "
             f"carries a Wilson interval — the top of any ladder is estimated from few "
             f"passers, and a bare 1.000 over 3 rows and a 0.90 over 90 are the same column "
             f"otherwise.\n")
    w.append(md_table(
        ["p_ge3 >=", "fires", "pass rate", "TP", "precision", "95% CI", "recall", "mark"],
        [[f"{r['threshold']:.2f}", r["fires"], pct(r["pass_rate"]), r["tp"],
          pct(r["precision"]),
          "—" if r["precision"] is None else
          f"{pct(r['precision_lo'])}–{pct(r['precision_hi'])}",
          pct(r["recall"]), " ".join(r["marks"])] for r in cal["ladder_ge3"]]))
    w.append("\n**Today's two cuts, for reference only.**\n")
    for name, r in cal["at_live_cuts"].items():
        w.append(f"- `{name}` = {r['threshold']:.2f} — fires {r['fires']}/{cal['n']} "
                 f"({pct(r['pass_rate'])}), precision {pct(r['precision'])} "
                 f"[{pct(r['precision_lo'])}–{pct(r['precision_hi'])}], "
                 f"recall {pct(r['recall'])}.")
    w.append("\n**RELEASE-floor candidates — DERIVED, NOT ADOPTED.** The release floor is a "
             "precision question — what may ship. Lowest swept threshold reaching each "
             "target.\n")
    w.append(md_table(
        ["target precision", "lowest p_ge3", "achieved", "95% CI", "recall", "fires",
         "supported by the CI?"],
        [[t, "—" if c is None else f"{c['threshold']:.2f}",
          "—" if c is None else pct(c["precision"]),
          "—" if c is None else f"{pct(c['precision_lo'])}–{pct(c['precision_hi'])}",
          "—" if c is None else pct(c["recall"]), "—" if c is None else c["fires"],
          "—" if c is None else ("yes" if c["supported"] else "NO")]
         for t, c in cal["release_candidates_ge3"].items()]))
    w.append("\n**POOL-floor candidates — DERIVED, NOT ADOPTED.** The pool floor is capacity "
             "ordering, not curation (`floors.py`), so its question is the mirror one: the "
             "HIGHEST threshold that still keeps each share of the good rows.\n")
    w.append(md_table(
        ["retain recall >=", "highest p_ge3", "recall kept", "pass rate", "fires",
         "precision there", "95% CI"],
        [[t, "—" if c is None else f"{c['threshold']:.2f}",
          "—" if c is None else pct(c["recall"]),
          "—" if c is None else pct(c["pass_rate"]),
          "—" if c is None else c["fires"],
          "—" if c is None else pct(c["precision"]),
          "—" if c is None else f"{pct(c['precision_lo'])}–{pct(c['precision_hi'])}"]
         for t, c in cal["pool_candidates_ge3"].items()]))
    w.append(f"\n**The `>=2` ladder** (base rate {pct(cal['base_rate_ge2'])}), for the "
             f"pool cut's not-bad question.\n")
    w.append(md_table(
        ["p_ge2 >=", "fires", "pass rate", "precision", "95% CI", "recall"],
        [[f"{r['threshold']:.2f}", r["fires"], pct(r["pass_rate"]), pct(r["precision"]),
          "—" if r["precision"] is None else
          f"{pct(r['precision_lo'])}–{pct(r['precision_hi'])}", pct(r["recall"])]
         for r in cal["ladder_ge2"]]))
    w.append(f"\n{cal['note']}\n")
    (OUT / "report.md").write_text("\n".join(w), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_eval_rows()
    labels = np.array([r["label"] for r in rows])
    log(f"[v2-reads] eval n={len(rows)} · {len({r['mode'] for r in rows})} modes · "
        f"{len({r['loc'] for r in rows})} locations · labels {tier_dist(labels)['hist']}")

    heads = {}
    for name, ck in (("v1", V1_CKPT), ("v2", V2_CKPT)):
        if not (ROOT / ck).exists():
            raise SystemExit(f"[v2-reads] {name} checkpoint missing: {ck}")
        log(f"[v2-reads] scoring {name}: {ck}")
        heads[name] = score_with(ck, rows)

    seed_scores = {}
    for sd in sorted(V2_DIR.glob("seed_*/model_best.pt")):
        s = int(sd.parent.name.split("_")[1])
        log(f"[v2-reads] scoring v2 seed {s}")
        seed_scores[s] = score_with(str(sd.relative_to(ROOT).as_posix()), rows)

    R = build(rows, labels, heads, seed_scores)
    (OUT / "report.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    write_md(R)
    wr = R["winner_rule"]
    log(f"[v2-reads] winner = {wr['winner']}  (a={wr['clause_a_pass']} b={wr['clause_b_pass']})")
    log(f"[v2-reads] -> {OUT}/report.md, {OUT}/report.json")


if __name__ == "__main__":
    main()
