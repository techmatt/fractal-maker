#!/usr/bin/env python
r"""view_fit.py — the FITTED neighborhood-quality score for the view-level screen.

WHY IT EXISTS. `view_screen.composite_v3` is a hand-crafted formulation: a size band, a
coverage geometric mean and a winsorized richness product, each weight fixed by Matt's
verdicts on a Q5 sheet. The v4 iteration is the argument against continuing that way —
41 formulations, 16,440 candidates, none adopted, and the reason was not that any one
formulation was wrong but that **G4 and G7 are 5.7 percentile points apart and every
coverage-side lever moves them together** (`retired.md`, 2026-08-01). Hand-weighting cannot
resolve two gate clauses that a single term controls jointly. Labels can: the supply crawl
is 730 human-labeled candidates drawn FROM the screened population, so the weights can be
fitted against the thing the screen is a proxy for instead of against six anchor tiles.

WHAT IS FITTED, AND WHAT IT IS FOR. One L2 logistic regression on standardized recorded
features, target `label >= 2`. **Matt's semantics for this population**: label 2 means "a
q3/q4 is reachable within ~1-2 descents", so the screen's job is delivering a promising
NEIGHBORHOOD to the descent, not a finished wallpaper. `score()` returns the model's logit
— a linear function of the standardized features, monotone in `p(>= 2)` — and that logit is
the sort key this module offers.

SCOPE, and it is narrow. **Sourcing-side steer / screen ONLY.** Never a ranking of finished
images (that is the classifier's job, `production_pins.ACTIVE_CKPT`) and never a
cross-family allocation. The scope matters because `degree` is a fitted feature and on this
population `degree` IS the family (`mandelbrot`=2, `multibrot3`=3, ...): a pooled queue
sorted on a score with a degree coefficient in it allocates across families by that
coefficient. `FEATURES_NO_FAMILY` is the variant with `degree` and `log10_period` removed,
fitted and reported alongside, so that use has a score that does not encode family.

**NOT LIVE.** `composite_v3` remains the live sort key everywhere — `maneuver_view_screen`,
`steered_frontier` and `view_frame_sweep` are untouched, and
`test_view_screen.test_the_live_sort_key_is_composite_v3` still passes. This module is
staged for a later adoption decision against a pre-registered bar in its own prompt.

THE TWO DERIVED FEATURES, and why they are a numpy pass and not a re-render. Matt's labeling
observations named two axes: **minibrot size relative to the frame** and **falloff rate
around it**. The first is arithmetic on recorded columns (`window_scale / fw`). The second
is not on the row at all, so it is computed from the cached 64x36 fields
(`view_field_cache`) — which is exactly the thing the v4 iteration left behind. Neither
re-renders anything.

`falloff_half` is deliberately NOT `field_metrics.falloff_extent`, which is retired as a
quality measure (`retired.md`; 144 of 200 triage atoms beat the eye on it,
`orbital_field_metrics.md` §4). Two differences: the retirement is about a measure standing
ALONE as a quality axis, and this enters a fit as one covariate among sixteen with its
coefficient reported; and the formulation is a normalized 50% half-scale on binned medians
rather than a 90%->10% extent, so a frame with no descent at all reports the frame radius
instead of `falloff_extent`'s `0.0` (which collides with "descends instantly").

  uv run python tools/atlas/view_fit.py fit        # refit, rewrite the record + readout
  uv run python tools/atlas/view_fit.py sheet      # disagreement contact sheet
  uv run python tools/atlas/view_fit.py verify     # record <-> code agreement
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools" / "corpus", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import field_metrics as fm                      # noqa: E402  DENSITY, _plane_radius_grid

# --------------------------------------------------------------------------- #
# the population this was fitted on
# --------------------------------------------------------------------------- #
STAMP = "2026-08-01"
RUN_DIR_REL = "data/discovery/supply_crawl_v15_20260801"
FEATURES_REL = f"data/supply_crawl/{STAMP}/features.jsonl"
RECORD_REL = "data/atlas/view_fit_v1.json"

# leg -> batch_id. The ROLES ARE NOT INTERCHANGEABLE and the fit driver enforces it:
# `strat_*` is the only leg with the negative class's footing (round-robin over every bin),
# `uniform` is the only score-unconditioned draw and is never fitted on, `exemplar` is a
# top-by-similarity draw held out entirely.
BATCH_LEGS = {
    f"{STAMP}_supply_crawl_strat_a_v1": "strat_a",
    f"{STAMP}_supply_crawl_strat_b_v1": "strat_b",
    f"{STAMP}_supply_crawl_uniform_v1": "uniform",
    f"{STAMP}_supply_crawl_exemplar_v1": "exemplar",
}
FIT_LEGS = ("strat_a", "strat_b")
RULE_LABEL_FILE = "rule_labels_interior_gt30_v1.json"

# The pre-registered target. `>= 3` is 14 positives on 730 and is reported descriptively
# only — no fitted cut, no CI.
TARGET_LABEL = 2

# L2 strength, pre-registered before the fit was run: p = 16 features on n = 580 rows with
# a 25.7% positive rate. `fit --c-sweep` reports the OOF metric across {0.1, 1, 10} so the
# choice is measured rather than asserted.
FIT_C = 1.0
N_FOLDS = 5
FIT_SEED = 20260802


# --------------------------------------------------------------------------- #
# the two derived axes: a numpy pass over the cached field
# --------------------------------------------------------------------------- #
FALLOFF_BINS = 16


def _binned_radial_median(field: np.ndarray, n_bins: int = FALLOFF_BINS):
    """`(bin centers, median smooth value in colour cycles)` over ESCAPING pixels.

    Radius is `field_metrics._plane_radius_grid` — the shared plane-radius grid, so this
    bins the same way `interior_profile` and `falloff_extent` do and a radius here means
    what a radius means there. Bins with fewer than 8 escaping pixels are NaN, not zero:
    an unmeasured bin and a bin that measured zero are different facts.
    """
    h, w = field.shape
    r = fm._plane_radius_grid(h, w)
    edges = np.linspace(0.0, float(r.max()), n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    v = np.asarray(field, dtype=np.float64).ravel() * fm.DENSITY
    med = np.full(n_bins, np.nan)
    fin = np.isfinite(v)
    for b in range(n_bins):
        sel = v[(idx == b) & fin]
        if sel.size >= 8:
            med[b] = np.median(sel)
    return 0.5 * (edges[:-1] + edges[1:]), med


def falloff_features(field: np.ndarray, n_bins: int = FALLOFF_BINS) -> dict:
    """Matt's second axis — how fast the field decays away from the nucleus. Pure; no engine.

    Two formulations, because "rate" admits two readings and neither dominates:

      `falloff_rate` — MINUS the least-squares slope of the binned median (colour cycles)
        against radius (frame half-heights). Positive = the field descends outward, which is
        the minibrot-with-decoration case; negative = it climbs outward, which is a frame
        whose subject is not at the centre. Unbounded, and sensitive to the whole profile.

      `falloff_half` — the radius at which the binned median has fallen HALFWAY from its
        innermost value to its outermost. A scale, not a slope, so it is robust to one wild
        bin; small = a thin skin around the nucleus, large = decoration filling the frame.
        A profile that never descends reports the outer bin centre (the frame radius), which
        is the honest reading of "no falloff inside this frame" — `falloff_extent` returns
        0.0 there and so cannot be told apart from an instant descent.

    Both are recorded and both enter the fit; the fit reports what each is worth.
    """
    c, med = _binned_radial_median(field, n_bins)
    ok = np.isfinite(med)
    if ok.sum() < 4:
        return dict(falloff_rate=0.0, falloff_half=0.0)
    cg, mg = c[ok], med[ok]
    slope = float(np.polyfit(cg, mg, 1)[0])
    inner, outer = float(mg[0]), float(mg[-1])
    rng = inner - outer
    if rng <= 0:
        half = float(cg[-1])
    else:
        below = np.nonzero(mg <= outer + 0.5 * rng)[0]
        half = float(cg[below[0]]) if below.size else float(cg[-1])
    return dict(falloff_rate=round(-slope, 6), falloff_half=round(half, 6))


# --------------------------------------------------------------------------- #
# the feature vector
# --------------------------------------------------------------------------- #
# PRE-REGISTERED, and fitted ONCE. Everything here is either a recorded column of
# `features.jsonl` or one of the two derived axes above. What is deliberately NOT here:
#   * `size_factor` and `vetoed` — deterministic functions of `interior_fraction`, which is
#     in; including them would be the same column three times.
#   * `k` — structurally missing on 253 of 730 rows (a `keep`-framed candidate has no `k`),
#     and an imputed value there would be a fiction, not a missing value.
#   * `log10_abs_A`, `parent_depth`, `operator` — recorded and left out to hold the
#     multiplicity down; the stratified draw balanced `operator` by construction, and
#     `log10_abs_A` is near-collinear with the size axis. Named here so the omission is a
#     decision on the record rather than an oversight.
FEATURE_ORDER = (
    "band_coverage",          # v3's coverage half, as recorded
    "band_coverage_q25",      # ... and its spatially pooled sibling
    "log1p_radial_range",     # richness, log1p'd: raw is heavy-tailed to 1.6e4
    "log1p_radial_rings",
    "interior_fraction",      # the axis v3 bands and vetoes on
    "exemplar_sim_max",       # the confounded hypothesis, read #1
    "exemplar_sim_mean",
    "degree",                 # FAMILY on this population — see FEATURES_NO_FAMILY
    "log10_period",
    "log10_size_rel",         # Matt's axis 1: nucleus size relative to the frame
    "falloff_rate",           # Matt's axis 2 (derived)
    "falloff_half",           # ... second formulation of the same axis (derived)
    "log10_fw",               # depth
    "cap_headroom",           # how far the frame sits below the screen's iteration cap
    "clamped",
    "composite_v3",           # the incumbent, as a baseline column
)

# The variant safe for a POOLED queue: no term that identifies the family. Reported beside
# the primary fit, never instead of it.
FAMILY_FEATURES = ("degree", "log10_period")
FEATURES_NO_FAMILY = tuple(f for f in FEATURE_ORDER if f not in FAMILY_FEATURES)

# --------------------------------------------------------------------------- #
# v1.1 (2026-08-02) — the variant that ORDERS a queue
# --------------------------------------------------------------------------- #
# `no_family` minus the two exemplar-similarity columns. Both removals are decisions with a
# reason, and neither is a tuning choice:
#   * `degree`/`log10_period` — `degree` IS the family on this population, so a POOLED queue
#     sorted on a score carrying a degree coefficient allocates across families by that
#     coefficient. That is the module doc's own scope argument, and the label-seeded harvest
#     is exactly the pooled-queue case it was written for.
#   * `exemplar_sim_max`/`_mean` — RETIRED as an ordering/steering feature on two null
#     pre-registered reads (`retired.md`, 2026-08-02). A retired ordering feature cannot stay
#     a column in the model that does the ordering, and the harvest does not compute it at
#     all, so a model that needed it could not score the queue. The cost of the removal is
#     measured, not assumed: `exemplar_read_a.drop_column_ap` (0.7158) is the same drop on
#     the primary feature set and is ABOVE the full fit's 0.7118.
EXEMPLAR_FEATURES = ("exemplar_sim_max", "exemplar_sim_mean")
FEATURES_V11 = tuple(f for f in FEATURES_NO_FAMILY if f not in EXEMPLAR_FEATURES)

# The C grid the nested selection chooses from, pre-registered. v1 asserted C=1.0 and
# reported a flat `--c-sweep` beside it; v1.1 SELECTS, inside the cross-validation, so the
# selection cannot see the fold it is scored on.
#
# THE GRID WAS WIDENED ONCE, AND WHY IS PART OF THE RESULT. The first grid stopped at 10 and
# every outer fold picked 10 — the edge — with inner AP rising monotonically across it
# (0.671 -> 0.729). A selection that lands on its grid boundary has not selected; it has run
# out of grid. Extended to 1000 so the interior optimum is inside the search rather than
# assumed, and the per-fold picks are recorded so an edge pick stays visible if it recurs.
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
INNER_FOLDS = 4
MODEL_ID_V11 = "view_fit_v1.1"
RECORD_V11_REL = "data/atlas/view_fit_v1_1.json"


def row_features(rec: dict, field_feats: dict) -> dict:
    """The full pre-registered feature dict for one `features.jsonl` row.

    `field_feats` is `falloff_features(field)` for the same candidate — passed in rather
    than resolved here so this stays pure and a caller that already has the field does not
    re-open the cache.
    """
    fw = float(rec["fw"])
    ws = float(rec["window_scale"])
    return {
        "band_coverage": float(rec["band_coverage"]),
        "band_coverage_q25": float(rec["band_coverage_q25"]),
        "log1p_radial_range": math.log1p(max(0.0, float(rec["radial_range"]))),
        "log1p_radial_rings": math.log1p(max(0.0, float(rec["radial_rings"]))),
        "interior_fraction": float(rec["interior_fraction"]),
        "exemplar_sim_max": float(rec["exemplar_sim_max"]),
        "exemplar_sim_mean": float(rec["exemplar_sim_mean"]),
        "degree": float(rec["degree"]),
        "log10_period": math.log10(max(1.0, float(rec["period"]))),
        "log10_size_rel": math.log10(ws / fw) if (ws > 0 and fw > 0) else 0.0,
        "falloff_rate": float(field_feats["falloff_rate"]),
        "falloff_half": float(field_feats["falloff_half"]),
        "log10_fw": math.log10(fw) if fw > 0 else 0.0,
        # `cap_headroom` is null exactly when the frame had no escaping pixel to take a
        # smooth max from. NaN here and imputed at the FIT's median, recorded in the model.
        "cap_headroom": (float(rec["cap_headroom"]) if rec.get("cap_headroom") is not None
                         else float("nan")),
        "clamped": 1.0 if rec.get("clamped") else 0.0,
        "composite_v3": float(rec["composite"]),
    }


# --------------------------------------------------------------------------- #
# the fitted model — the thing a later adoption would read
# --------------------------------------------------------------------------- #
class FittedScore:
    """A frozen standardized linear score: `b0 + sum w_i * (x_i - mu_i) / sd_i`.

    The returned value is the model's LOGIT. It is what a sort key wants (monotone in
    `p(>= 2)`, unbounded, no saturation at the top of the queue where the sort actually
    matters); `p_notbad()` is the same number through a sigmoid for a threshold.

    Loaded from the record rather than hardcoded, for the reason `view_screen.ScreenParams`
    is: re-fitting must move the score, not leave stale literals in source
    (`CLAUDE.md`, "Derive state in code; freeze it in records").
    """

    def __init__(self, spec: dict):
        self.features = tuple(spec["features"])
        self.mean = np.asarray(spec["mean"], dtype=float)
        self.scale = np.asarray(spec["scale"], dtype=float)
        self.coef = np.asarray(spec["coef"], dtype=float)
        self.intercept = float(spec["intercept"])
        self.impute = dict(spec.get("impute") or {})
        n = len(self.features)
        if not (len(self.mean) == len(self.scale) == len(self.coef) == n):
            raise ValueError(f"model arity mismatch: {n} features vs "
                             f"{len(self.mean)}/{len(self.scale)}/{len(self.coef)}")

    def vector(self, feats: dict) -> np.ndarray:
        out = np.empty(len(self.features), dtype=float)
        for i, name in enumerate(self.features):
            v = float(feats[name])
            if not math.isfinite(v):
                v = float(self.impute[name])       # KeyError = an unmodelled missing value
            out[i] = v
        return out

    def score(self, feats: dict) -> float:
        z = (self.vector(feats) - self.mean) / self.scale
        return float(self.intercept + float(self.coef @ z))

    def p_notbad(self, feats: dict) -> float:
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.score(feats)))))

    def weights(self) -> dict:
        """`{feature: standardized coefficient}` — the readable form of the fit."""
        return {f: float(c) for f, c in zip(self.features, self.coef)}


def load_model(variant: str = "primary", path=None) -> FittedScore:
    """The fitted score from its record. `variant` is `primary` or `no_family`."""
    p = Path(path) if path else paths.durable(RECORD_REL)
    rec = json.loads(p.read_text(encoding="utf-8"))
    return FittedScore(rec["models"][variant])


# =========================================================================== #
# the fit driver
# =========================================================================== #
def _jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def load_table(root: Path = ROOT) -> list[dict]:
    """The 730-row fit table: recorded features + derived axes + label + leg + CV group.

    Joins four things, and each join is asserted rather than assumed: the feature table
    (`features.jsonl`), the four batches' merged `label.score`, the rule-label sidecars (so
    a rule row is TAGGED and not merely a score), and `root_id` from the run's maneuver log
    (the CV group). Missing anything raises — a silently short table is the failure mode a
    fit cannot detect.
    """
    feats = _jl(root / FEATURES_REL)
    by_id = {r["image_id"]: r for r in feats}
    if len(by_id) != len(feats):
        raise SystemExit(f"{FEATURES_REL}: duplicate image_id")

    batches = root / "data" / "label_corpus" / "batches"
    labels, rule_rows = {}, set()
    for bid, leg in BATCH_LEGS.items():
        d = batches / bid
        for r in _jl(d / "images.jsonl"):
            labels[r["image_id"]] = (leg, r["label"]["score"], r["label"].get("labeler"))
        rp = d / RULE_LABEL_FILE
        if rp.exists():
            rule_rows |= set(json.loads(rp.read_text(encoding="utf-8"))["labels"])
    missing = set(by_id) - set(labels)
    if missing:
        raise SystemExit(f"{len(missing)} feature rows have no label row, e.g. "
                         f"{sorted(missing)[:3]}")

    # root_id: the walk root the candidate descends from. The CV group, and it is
    # partition-qualified because node ids are per-partition tree indices.
    want = {r["candidate_key"] for r in feats}
    roots = {}
    with (root / RUN_DIR_REL / "maneuvers.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("available") or r.get("op") == "probe":
                continue
            key = f"{r.get('atom_key')}|{r.get('k')}"
            if key in want and key not in roots:
                roots[key] = f"{r.get('partition')}:{r.get('root_id')}"
    if len(roots) != len(want):
        raise SystemExit(f"root_id join is short: {len(roots)} of {len(want)}")

    import view_field_cache as vfc
    cache = vfc.FieldCache(root / RUN_DIR_REL / "view_fields")

    out = []
    for r in feats:
        key = r["candidate_key"]
        field = cache.get(key)
        if field is None:
            raise SystemExit(f"{key}: no cached field — rebuild view_fields first")
        leg, score, labeler = labels[r["image_id"]]
        out.append(dict(
            image_id=r["image_id"], batch_id=r["batch_id"], leg=leg, label=score,
            labeler=labeler, rule=r["image_id"] in rule_rows,
            group=roots[key], atom=key.split("|")[0], candidate_key=key,
            composite_v3=float(r["composite"]), operator=r["operator"],
            degree=int(r["degree"]), raw=r,
            feats=row_features(r, falloff_features(field)),
        ))
    return out


def _design(rows, features, impute=None):
    X = np.array([[row["feats"][f] for f in features] for row in rows], dtype=float)
    if impute is None:
        impute = {f: float(np.nanmedian(X[:, i])) for i, f in enumerate(features)}
    for i, f in enumerate(features):
        bad = ~np.isfinite(X[:, i])
        if bad.any():
            X[bad, i] = impute[f]
    return X, impute


def _fit(rows, features, *, C=FIT_C):
    """Standardize-then-L2-logistic on `rows`. Returns `(spec, pipeline)`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, impute = _design(rows, features)
    y = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in rows], dtype=int)
    # `l1_ratio=0` is pure L2. Spelled that way and not `penalty="l2"` because `penalty` is
    # deprecated in sklearn 1.8 and removed in 1.10 — a FutureWarning here would be noise in
    # every refit, and the replacement is exact, not approximate.
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=C, max_iter=5000, l1_ratio=0.0))])
    pipe.fit(X, y)
    sc, lr = pipe.named_steps["sc"], pipe.named_steps["lr"]
    spec = dict(features=list(features), mean=[round(v, 8) for v in sc.mean_],
                scale=[round(v, 8) for v in sc.scale_],
                coef=[round(v, 6) for v in lr.coef_[0]],
                intercept=round(float(lr.intercept_[0]), 6),
                impute={f: round(impute[f], 8) for f in features},
                C=C, n=len(rows), n_pos=int(y.sum()))
    return spec, pipe


def _oof(rows, features, *, C=FIT_C, n_folds=N_FOLDS):
    """Out-of-fold logits under GroupKFold on the walk root. Groups, not rows, are split."""
    from sklearn.model_selection import GroupKFold
    X, impute = _design(rows, features)
    y = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in rows], dtype=int)
    g = np.array([r["group"] for r in rows])
    oof = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=g):
        sub, _ = _fit([rows[i] for i in tr], features, C=C)
        m = FittedScore(sub)
        oof[te] = [m.score(rows[i]["feats"]) for i in te]
    return oof, y


def _metrics(score, y) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr
    score, y = np.asarray(score, float), np.asarray(y, int)
    return dict(ap=round(float(average_precision_score(y, score)), 4),
                auc=round(float(roc_auc_score(y, score)), 4),
                spearman=round(float(spearmanr(score, y).statistic), 4),
                base_rate=round(float(y.mean()), 4), n=int(y.size), n_pos=int(y.sum()))


def _prec_at(score, y, ks) -> dict:
    order = np.argsort(-np.asarray(score, float))
    y = np.asarray(y, int)
    return {f"top{k}": round(float(y[order[:k]].mean()), 4) for k in ks if k <= y.size}


def _kept_missed(a, b, y, k: int) -> dict:
    """What one ordering keeps that the other misses, at a MATCHED emit budget of `k`.

    A precision pair says the two differ; this says where. The budget is matched because an
    ordering compared at its own preferred cut is not being compared at all.
    """
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y, int)
    ka = set(np.argsort(-a)[:k].tolist())
    kb = set(np.argsort(-b)[:k].tolist())
    pos = {i for i in range(y.size) if y[i]}
    return dict(k=k, positives=len(pos), fit_keeps=len(ka & pos), composite_keeps=len(kb & pos),
                fit_only=len((ka - kb) & pos), composite_only=len((kb - ka) & pos),
                overlap_rows=len(ka & kb))


def _boot_ap_delta(a, b, y, *, n=2000, seed=FIT_SEED) -> dict:
    """Paired bootstrap of `AP(a) - AP(b)` on the same rows. Resamples ROWS, which is the
    right unit for a difference of two scores evaluated on one population."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y, int)
    d = []
    for _ in range(n):
        i = rng.integers(0, y.size, y.size)
        if y[i].sum() == 0 or y[i].sum() == y[i].size:
            continue
        d.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    d = np.array(d)
    return dict(delta=round(float(np.mean(d)), 4),
                lo=round(float(np.percentile(d, 2.5)), 4),
                hi=round(float(np.percentile(d, 97.5)), 4), n_boot=int(d.size))


def _boot_coef(rows, features, names, *, n=400, seed=FIT_SEED) -> dict:
    """Percentile CI for named standardized coefficients, resampling GROUPS not rows.

    The row is not the independent unit — siblings off one walk root share a
    neighborhood — so a row bootstrap would report a CI narrower than the data supports.
    """
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    keys = list(by_group)
    draws = defaultdict(list)
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        samp = [r for i in pick for r in by_group[keys[i]]]
        ys = {r["label"] >= TARGET_LABEL for r in samp}
        if len(ys) < 2:
            continue
        spec, _ = _fit(samp, features)
        c = dict(zip(spec["features"], spec["coef"]))
        for nm in names:
            draws[nm].append(c[nm])
    return {nm: [round(float(np.percentile(v, 2.5)), 4),
                 round(float(np.percentile(v, 97.5)), 4)] for nm, v in draws.items()}


def run_fit(root: Path = ROOT, *, write=True, c_sweep=False) -> dict:
    rows = load_table(root)
    fit_rows = [r for r in rows if r["leg"] in FIT_LEGS]
    uni = [r for r in rows if r["leg"] == "uniform"]
    exe = [r for r in rows if r["leg"] == "exemplar"]
    y_fit = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in fit_rows])

    out = dict(
        schema_version=1, stamp=STAMP, target=f"label >= {TARGET_LABEL}",
        population=dict(
            n=len(rows), fit=len(fit_rows), uniform=len(uni), exemplar=len(exe),
            fit_positives=int(y_fit.sum()),
            rule_rows=sum(1 for r in rows if r["rule"]),
            groups=len({r["group"] for r in fit_rows}),
            source=dict(features=FEATURES_REL, run=RUN_DIR_REL,
                        batches=sorted(BATCH_LEGS)),
        ),
        cv=dict(scheme="GroupKFold", n_folds=N_FOLDS,
                group="walk root (partition:root_id)", C=FIT_C),
        models={}, readout={},
    )

    # ---- the two fits, and their grouped-CV out-of-fold performance ----------
    for name, feats in (("primary", FEATURE_ORDER), ("no_family", FEATURES_NO_FAMILY)):
        spec, _ = _fit(fit_rows, feats)
        oof, y = _oof(fit_rows, feats)
        spec["oof"] = _metrics(oof, y)
        spec["oof_prec_at"] = _prec_at(oof, y, (20, 50, 100, 145))
        out["models"][name] = spec

    comp = np.array([r["composite_v3"] for r in fit_rows])
    oof_primary, _ = _oof(fit_rows, FEATURE_ORDER)
    out["readout"]["composite_v3_on_fit"] = _metrics(comp, y_fit)
    out["readout"]["composite_v3_prec_at"] = _prec_at(comp, y_fit, (20, 50, 100, 145))
    out["readout"]["ap_delta_fit_vs_composite"] = _boot_ap_delta(oof_primary, comp, y_fit)
    out["readout"]["kept_missed_top_quartile"] = _kept_missed(oof_primary, comp, y_fit, 145)

    # ---- sensitivity: the 81 rule-labeled rows in and out -------------------
    # The rule labeled `interior_fraction > 0.30` as score 1, and `interior_fraction` is a
    # fitted feature — so the largest coefficient in the model is partly the rule's own
    # definition read back. The coefficient is reported BOTH ways for that reason.
    no_rule = [r for r in fit_rows if not r["rule"]]
    oof_nr, y_nr = _oof(no_rule, FEATURE_ORDER)
    spec_nr, _ = _fit(no_rule, FEATURE_ORDER)
    out["readout"]["sensitivity_rule_rows_dropped"] = dict(
        n=len(no_rule), dropped=len(fit_rows) - len(no_rule),
        oof=_metrics(oof_nr, y_nr),
        composite_v3=_metrics([r["composite_v3"] for r in no_rule], y_nr),
        coef=dict(zip(spec_nr["features"], spec_nr["coef"])))

    # ---- exemplar read (a): does similarity carry a coefficient? ------------
    drop_sim = tuple(f for f in FEATURE_ORDER if not f.startswith("exemplar_sim"))
    oof_ns, _ = _oof(fit_rows, drop_sim)
    w = out["models"]["primary"]
    coefs = dict(zip(w["features"], w["coef"]))
    out["readout"]["exemplar_read_a"] = dict(
        coef_sim_max=coefs["exemplar_sim_max"], coef_sim_mean=coefs["exemplar_sim_mean"],
        max_abs_coef=round(float(np.max(np.abs(w["coef"]))), 6),
        drop_column_ap=_metrics(oof_ns, y_fit)["ap"],
        full_ap=w["oof"]["ap"],
        delta_ap=_boot_ap_delta(oof_primary, oof_ns, y_fit),
        coef_ci95=_boot_coef(fit_rows, FEATURE_ORDER,
                             ("exemplar_sim_max", "exemplar_sim_mean")),
        univariate_spearman={
            k: round(float(np.corrcoef([r["feats"][k] for r in fit_rows], y_fit)[0, 1]), 4)
            for k in ("exemplar_sim_max", "exemplar_sim_mean")})

    # ---- exemplar read (b): does the strat fit predict the leg's rate? ------
    m = FittedScore(out["models"]["primary"])
    p_exe = np.array([m.p_notbad(r["feats"]) for r in exe])
    y_exe = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in exe])
    out["readout"]["exemplar_read_b"] = dict(
        n=len(exe), predicted_rate=round(float(p_exe.mean()), 4),
        realized_rate=round(float(y_exe.mean()), 4),
        realized_ci95=_wilson(int(y_exe.sum()), len(y_exe)),
        expected_count=round(float(p_exe.sum()), 1), realized_count=int(y_exe.sum()),
        note=("in-sample-family caveat: the model was fitted on strat only, but the "
              "exemplar leg is a top-by-similarity draw, so its FEATURES are out of the "
              "fit leg's range on the similarity axis"))

    # ---- the unbiased check: the uniform leg --------------------------------
    p_uni = np.array([m.score(r["feats"]) for r in uni])
    y_uni = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in uni])
    out["readout"]["uniform_leg"] = dict(
        fitted=_metrics(p_uni, y_uni),
        composite_v3=_metrics([r["composite_v3"] for r in uni], y_uni),
        fitted_prec_at=_prec_at(p_uni, y_uni, (10, 20, 30, 45)),
        composite_prec_at=_prec_at([r["composite_v3"] for r in uni], y_uni,
                                   (10, 20, 30, 45)),
        ap_delta=_boot_ap_delta(p_uni, [r["composite_v3"] for r in uni], y_uni),
        kept_missed_top_23=_kept_missed(p_uni, [r["composite_v3"] for r in uni], y_uni, 23))

    # ---- >= 3, descriptive only ---------------------------------------------
    y3 = np.array([1 if r["label"] >= 3 else 0 for r in rows])
    s_all = np.array([m.score(r["feats"]) for r in rows])
    c_all = np.array([r["composite_v3"] for r in rows])
    order_s, order_c = np.argsort(-s_all), np.argsort(-c_all)
    out["readout"]["ge3_descriptive"] = dict(
        n=len(rows), positives=int(y3.sum()), class4=int(sum(r["label"] == 4 for r in rows)),
        fitted_in_top_73=int(y3[order_s[:73]].sum()),
        composite_in_top_73=int(y3[order_c[:73]].sum()),
        note="14 positives, no fitted cut — a rate on 14 is not a bar")

    if c_sweep:
        out["readout"]["c_sweep"] = {
            str(c): _metrics(*_oof(fit_rows, FEATURE_ORDER, C=c))["ap"]
            for c in (0.1, 1.0, 10.0)}

    if write:
        p = paths.durable(RECORD_REL)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {p}")
    return out


# =========================================================================== #
# v1.1 — nested-CV C selection on the strat legs, for the pooled queue
# =========================================================================== #
def _select_c(rows, features, *, grid=C_GRID, n_folds=INNER_FOLDS) -> tuple:
    """The C maximising grouped-CV AP on `rows`. Returns `(best_C, {C: ap})`.

    Grouped on the walk root like every other split here: a row is not the independent
    unit, so a row-wise inner CV would let a sibling of the held-out row train the model
    that scores it and would pick a C that is too weak a penalty.

    Ties break to the LARGEST penalty (smallest C). A tie means the data does not
    distinguish them, and preferring the more regularised model there is the choice that
    does not silently buy variance for nothing.
    """
    from sklearn.model_selection import GroupKFold
    X, _ = _design(rows, features)
    y = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in rows], dtype=int)
    g = np.array([r["group"] for r in rows])
    n_folds = max(2, min(n_folds, len(set(g))))
    aps = {}
    for c in grid:
        oof = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=g):
            spec, _ = _fit([rows[i] for i in tr], features, C=c)
            m = FittedScore(spec)
            oof[te] = [m.score(rows[i]["feats"]) for i in te]
        aps[c] = _metrics(oof, y)["ap"]
    best = max(grid, key=lambda c: (aps[c], -c))
    return best, {str(c): aps[c] for c in grid}


def _nested_oof(rows, features, *, grid=C_GRID, outer=N_FOLDS, inner=INNER_FOLDS):
    """Out-of-fold logits under a NESTED grouped CV: C is chosen inside each outer fold.

    This is the number the adoption question needs and `_oof` at a fixed C is not. Selecting
    C on the same folds the score is then read off makes the readout optimistic by exactly
    the amount the selection is worth — the selection has seen the test fold. Here the inner
    CV runs on the outer TRAINING rows only, so the reported AP prices the whole procedure
    (select C, fit, score) as it would run on data it has never seen.
    """
    from sklearn.model_selection import GroupKFold
    X, _ = _design(rows, features)
    y = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in rows], dtype=int)
    g = np.array([r["group"] for r in rows])
    oof = np.full(len(rows), np.nan)
    picked = []
    for tr, te in GroupKFold(n_splits=outer).split(X, y, groups=g):
        sub = [rows[i] for i in tr]
        c, _aps = _select_c(sub, features, grid=grid, n_folds=inner)
        picked.append(c)
        spec, _ = _fit(sub, features, C=c)
        m = FittedScore(spec)
        oof[te] = [m.score(rows[i]["feats"]) for i in te]
    return oof, y, picked


def run_fit_v11(root: Path = ROOT, *, write=True) -> dict:
    """Fit, stamp and record `view_fit_v1.1` — the score the label-seeded queue is ordered by.

    Same population, same target and the same grouped CV as v1. Three things move, each
    named in the record so a later reader does not have to diff two files to find them:
    the feature set (`FEATURES_V11`), the C (selected rather than asserted), and the scope
    (this one is offered AS a sort key for a pooled sourcing queue).
    """
    rows = load_table(root)
    fit_rows = [r for r in rows if r["leg"] in FIT_LEGS]
    uni = [r for r in rows if r["leg"] == "uniform"]
    y_fit = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in fit_rows])

    best_c, inner_aps = _select_c(fit_rows, FEATURES_V11)
    spec, _ = _fit(fit_rows, FEATURES_V11, C=best_c)
    nested, y, picked = _nested_oof(fit_rows, FEATURES_V11)
    spec["oof_nested"] = _metrics(nested, y)
    spec["oof_nested_prec_at"] = _prec_at(nested, y, (20, 50, 100, 145))
    spec["c_selected"] = best_c
    spec["c_selected_per_outer_fold"] = picked
    spec["c_grid_inner_ap"] = inner_aps
    spec["model_id"] = MODEL_ID_V11

    comp = np.array([r["composite_v3"] for r in fit_rows])
    # The two incumbents this has to be read against, on the SAME rows: the live sort key,
    # and v1's own no_family fit at its asserted C=1.0.
    oof_nf, _ = _oof(fit_rows, FEATURES_NO_FAMILY)
    out = dict(
        schema_version=1, model_id=MODEL_ID_V11, stamp="2026-08-02",
        supersedes=dict(record=RECORD_REL, variant="no_family",
                        NOTE=("v1 stays as written and is not rewritten; this is a second "
                              "record, so the population that produced it is unchanged.")),
        target=f"label >= {TARGET_LABEL}",
        scope=("sourcing-side ORDERING of a pooled candidate queue. Never a ranking of "
               "finished images (that is production_pins.ACTIVE_CKPT) and never a "
               "cross-family allocation — which is why no family term is in it."),
        population=dict(n=len(rows), fit=len(fit_rows),
                        fit_positives=int(y_fit.sum()),
                        groups=len({r["group"] for r in fit_rows}),
                        source=dict(features=FEATURES_REL, run=RUN_DIR_REL,
                                    batches=sorted(BATCH_LEGS))),
        cv=dict(scheme="GroupKFold", outer_folds=N_FOLDS, inner_folds=INNER_FOLDS,
                group="walk root (partition:root_id)", c_grid=list(C_GRID),
                selection="nested: C chosen on the outer fold's TRAINING rows only"),
        features=dict(used=list(FEATURES_V11),
                      dropped_family=list(FAMILY_FEATURES),
                      dropped_exemplar=list(EXEMPLAR_FEATURES)),
        models={"v11": spec},
        readout=dict(
            v11_nested_oof=spec["oof_nested"],
            v11_fixed_c1_oof=_metrics(*_oof(fit_rows, FEATURES_V11, C=1.0)),
            v1_no_family_oof_c1=_metrics(oof_nf, y_fit),
            composite_v3_on_fit=_metrics(comp, y_fit),
            ap_delta_v11_vs_composite=_boot_ap_delta(nested, comp, y_fit),
            ap_delta_v11_vs_v1_no_family=_boot_ap_delta(nested, oof_nf, y_fit),
            kept_missed_top_quartile=_kept_missed(nested, comp, y_fit, 145),
            weights={f: round(float(c), 4) for f, c in
                     zip(spec["features"], spec["coef"])},
        ),
    )
    # The uniform leg is the one draw no score conditioned, so it is the only leg on which
    # "does this order better than the incumbent" is asked of an unbiased sample. It is NOT
    # a fit leg and never was; this is a read, not a selection.
    m = FittedScore(spec)
    p_uni = np.array([m.score(r["feats"]) for r in uni])
    y_uni = np.array([1 if r["label"] >= TARGET_LABEL else 0 for r in uni])
    out["readout"]["uniform_leg"] = dict(
        fitted=_metrics(p_uni, y_uni),
        composite_v3=_metrics([r["composite_v3"] for r in uni], y_uni),
        ap_delta=_boot_ap_delta(p_uni, [r["composite_v3"] for r in uni], y_uni))
    if write:
        p = paths.durable(RECORD_V11_REL)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {p}")
    return out


def load_model_v11(path=None) -> FittedScore:
    """The v1.1 score from its record — what `build_label_seeded_batches` orders on."""
    p = Path(path) if path else paths.durable(RECORD_V11_REL)
    return FittedScore(json.loads(p.read_text(encoding="utf-8"))["models"]["v11"])


def _wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# =========================================================================== #
# the disagreement contact sheet (reject-autopsy habit)
# =========================================================================== #
def build_sheet(root: Path = ROOT, n: int = 12, out=None) -> Path:
    """Where the fit and `composite_v3` disagree most, both directions, from EXISTING crops.

    Rank both scores over the whole 730 and take the largest percentile gaps each way. The
    crops were rendered for labeling and are reused as-is — a disagreement sheet that
    re-renders is a different picture from the one that was judged.
    """
    from PIL import Image, ImageDraw
    import corpus_common as cc
    rows = load_table(root)
    m = load_model("primary")
    s = np.array([m.score(r["feats"]) for r in rows])
    c = np.array([r["composite_v3"] for r in rows])
    rs, rc = _pct_rank(s), _pct_rank(c)
    gap = rs - rc
    order = np.argsort(-gap)
    picks = [("fit ABOVE composite", i) for i in order[:n]]
    picks += [("composite ABOVE fit", i) for i in order[::-1][:n]]

    tw, th = 320, 180
    cols, rows_n = n, 2
    sheet = Image.new("RGB", (cols * tw, rows_n * (th + 34)), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    for j, (side, i) in enumerate(picks):
        r = rows[i]
        col, rw = j % cols, j // cols
        x, y = col * tw, rw * (th + 34)
        cp = Path(cc.crops_dir(r["batch_id"])) / f"{r['image_id']}.jpg"
        if cp.exists():
            sheet.paste(Image.open(cp).convert("RGB").resize((tw, th)), (x, y))
        d.text((x + 4, y + th + 2),
               f"L{r['label']}  fit p{rs[i]*100:.0f}  v3 p{rc[i]*100:.0f}", fill=(235, 235, 235))
        d.text((x + 4, y + th + 14),
               f"d{r['degree']} {r['operator'][:12]} {'RULE' if r['rule'] else r['leg']}",
               fill=(170, 170, 180))
        d.text((x + 4, y + 2), side.split()[0].upper(), fill=(255, 210, 90))
    p = Path(out) if out else paths.scratch("view_fit") / "disagreement_sheet.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(p)
    labels = [(side, rows[i]["label"], round(float(gap[i]), 4)) for side, i in picks]
    print(f"wrote {p}\n" + "\n".join(f"  {a} L{b} gap {g:+.3f}" for a, b, g in labels))
    return p


def _pct_rank(v):
    from scipy.stats import rankdata
    return (rankdata(v) - 1) / (len(v) - 1)


def verify(root: Path = ROOT) -> None:
    """The record and the code agree: feature order, arity, and a re-scored row."""
    rec = json.loads(paths.durable(RECORD_REL).read_text(encoding="utf-8"))
    assert tuple(rec["models"]["primary"]["features"]) == FEATURE_ORDER, "feature drift"
    assert tuple(rec["models"]["no_family"]["features"]) == FEATURES_NO_FAMILY
    rows = load_table(root)
    m = load_model("primary")
    s = [m.score(r["feats"]) for r in rows]
    print(f"OK: {len(rows)} rows re-score; range [{min(s):.3f}, {max(s):.3f}]; "
          f"{len(FEATURE_ORDER)} features")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--c-sweep", action="store_true")
    f.add_argument("--no-write", action="store_true")
    f11 = sub.add_parser("fit11"); f11.add_argument("--no-write", action="store_true")
    s = sub.add_parser("sheet"); s.add_argument("-n", type=int, default=12)
    s.add_argument("--out")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        out = run_fit(write=not a.no_write, c_sweep=a.c_sweep)
        print(json.dumps(out["readout"], indent=1)[:4000])
    elif a.cmd == "fit11":
        out = run_fit_v11(write=not a.no_write)
        print(json.dumps(out["readout"], indent=1)[:4000])
    elif a.cmd == "sheet":
        build_sheet(n=a.n, out=a.out)
    else:
        verify()


if __name__ == "__main__":
    main()
