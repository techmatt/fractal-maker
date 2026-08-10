r"""suggest_tier_mining.py — the MINING head's pre-label tier suggestion (K=3), TORCH-FREE.

The wallpaper sibling (`tools/wallpaper/suggest_tier.py`) owns the RULE — the readout
`expected_tier(marg) = 1 + Sum_k marg[k]`, the cut application `tier_from_pred`, and the
prior-matching deriver `fit_cuts`. All three are k-agnostic, so they are IMPORTED here, not
restated: a second copy of a cut rule is a second chance to change one and not the other, and
the two heads must agree on what "tier" means even though they disagree on how many there
are. What is genuinely per-head lives here: K=3 (bad/okay/good), and the cutpoints.

WHY THE CUTS CANNOT BE FROZEN THE WAY THE WALLPAPER ONES ARE. `fit_cuts` needs aligned
`(pred, human_tier)` pairs on a labeled slice — it reads the cutpoints off the head's own
readout AT the observed tier prior. The wallpaper head has such a slice. The mining head does
not, and cannot: its 1500 human tiers survive (`labels/render_mode_pilot_v1.json`,
`labels/render_mode_scale_v1.json`, flat `image_id -> 1..3` maps) but every crop those ids
name is gone with `data/render_mode_corpus/`, so there is no image to compute a `pred` for.
The labels are orphaned — that is the whole shape of the loss this corpus exists to repair.

SO: QUANTILE-MATCH TO THE OLD PRIOR, AND SAY SO. The prompt sanctions exactly this fallback
("if none is, quantile-match to the old trainer's tier prior and say so in the record"). The
cuts are the quantiles of THIS BATCH's own `expected_tier` distribution placed at the
surviving corpus's tier shares, so the suggestion histogram reproduces the old label
distribution instead of collapsing onto one class.

WHAT THAT COSTS, STATED PLAINLY. These are PER-BATCH quantiles, and the wallpaper module
argues against exactly that for its own frozen cuts: an absolute cut lets a genuinely worse
population receive genuinely more tier-1 suggestions, and a per-batch quantile erases that.
It is erased here. The suggestion carries NO information about the new corpus's absolute
quality level — only about each row's rank within it. That is acceptable for the one job a
suggestion has (order the human's work and cut his keystrokes) and unacceptable for anything
else; nothing downstream may read a suggested tier as a label, and the merge refuses to.
Re-derive with `fit_cuts` the moment this sheet is labeled — a `(pred, tier)` slice on the
new corpus is the first thing that makes real cuts possible again, and it will exist as soon
as Matt's pass lands.

    from tools.mining.suggest_tier_mining import K_TIERS, tier_prior, cuts_from_prior, derivation
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# THE rule, imported. `expected_tier` and `tier_from_pred` are k-agnostic; `fit_cuts` takes
# k. Only the constants below are mining's own.
from tools.wallpaper.suggest_tier import (  # noqa: E402,F401  (re-exported on purpose)
    expected_tier, fit_cuts, tier_from_pred)

K_TIERS = 3          # 1 bad / 2 okay / 3 good — PINNED by the head (train_mining_head.K)

# --------------------------------------------------------------------------- #
# THE FROZEN CUTS (2026-08-10) — the re-derivation the block above scheduled.
# --------------------------------------------------------------------------- #
# "Re-derive with `fit_cuts` the moment this sheet is labeled — a (pred, tier) slice on the
# new corpus is the first thing that makes real cuts possible again, and it will exist as soon
# as Matt's pass lands." It landed: `labels/render_mode_fresh_sheet_v1.json` carries a human
# tier for all 960 rows of `2026-08-06_render_mode_fresh_sheet_v1`, and every row of that
# batch stamps the v1 `pred` it was scored at. So the mining head now has exactly what the
# wallpaper head had, and the per-batch-quantile fallback below is retired for new sheets.
#
# WHAT THIS BUYS, and it is the property the fallback gave up: these cuts are ABSOLUTE. A
# genuinely better population now receives genuinely more tier-3 suggestions, which is what a
# DELIBERATELY OVER-DRAWN slice (the busy fancy/composite high-score sheet) needs — a
# per-batch quantile would have forced exactly 13.7% of it to tier 3 whatever it contained.
#
# CIRCULARITY, NAMED. The slice's labels were collected on a CORRECTION sheet whose rows were
# served with a suggestion prefilled from `cuts_from_prior`, and Matt agreed with 892 of 960
# (92.9%). So this re-derivation is anchored to its predecessor and mostly reproduces it —
# (1.389, 1.933) -> (1.317, 1.873). It is still strictly better evidence than a prior read off
# a corpus whose images no longer exist, and the honest description of the number is "the
# cutpoints that reproduce Matt's OWN tier distribution on the one labeled slice v1 has".
CUTS = (1.317, 1.8727)

FIT_SLICE_BATCH = "2026-08-06_render_mode_fresh_sheet_v1"
FIT_SLICE_LABELS = "labels/render_mode_fresh_sheet_v1.json"

# The surviving human record. Flat `{image_id: 1..3}` maps; their crops and the manifest that
# gave the ids meaning are gone, which is why these can supply a PRIOR and not a slice.
PRIOR_LABEL_FILES = (
    "labels/render_mode_pilot_v1.json",     # 500, the 15-mode equal-per-mode pilot draw
    "labels/render_mode_scale_v1.json",     # 1000, the 13-mode tilt-to-yield scale draw
)
# The trainer consumed 1399 of these 1500 (774 train + 625 eval) after dropping trap_circle,
# exp_smoothing and direct_trap_screen. WHICH 1399 is not recoverable — the join is gone — so
# the prior here is over all 1500. Recorded rather than glossed: it is a ~101-row difference
# on the exact modes this corpus deliberately keeps, so the full set is the closer reference
# for a 15-mode draw anyway, not merely the only one available.
TRAINER_CONSUMED = 1399


def tier_prior(files=PRIOR_LABEL_FILES, root: Path = ROOT) -> dict:
    """`{tier: count}` over the surviving mining label files, plus per-file breakdown.

    Derived from the files at call time, never restated as literals: a hardcoded prior beside
    the paths it summarizes is the "hardcoded True" failure — it outlives the files the
    moment one changes. Raises on an absent file rather than quietly reporting a smaller
    prior, because a prior computed from half the record is not visibly wrong."""
    counts, per_file = Counter(), {}
    for rel in files:
        p = root / rel
        if not p.exists():
            raise SystemExit(
                f"[suggest-tier-mining] prior label file absent: {rel}\n"
                f"These are TRACKED and are the only surviving mining human record — a "
                f"prior over the remainder would be silently wrong, not smaller. Restore "
                f"with `git checkout -- {rel}`.")
        d = json.loads(p.read_text(encoding="utf-8"))
        c = Counter(int(v) for v in d.values())
        per_file[rel] = {str(t): c.get(t, 0) for t in range(1, K_TIERS + 1)}
        counts.update(c)
    total = sum(counts.values())
    if total == 0:
        raise SystemExit("[suggest-tier-mining] the prior label files hold zero tiers")
    return {
        "counts": {str(t): counts.get(t, 0) for t in range(1, K_TIERS + 1)},
        "shares": {str(t): counts.get(t, 0) / total for t in range(1, K_TIERS + 1)},
        "n": total, "per_file": per_file,
        "trainer_consumed": TRAINER_CONSUMED,
        "note": "the trainer used 1399 of these after three mode drops; WHICH 1399 is not "
                "recoverable (the manifest join is gone), so the prior is over all of them",
    }


def cuts_from_prior(pred, prior: dict, k: int = K_TIERS) -> tuple:
    """The k-1 ascending cutpoints on THIS batch's `pred` that place the given tier shares.

    `pred` is the per-row `expected_tier`; `prior["shares"]` the target distribution. Same
    arithmetic `fit_cuts` performs, driven by a prior instead of by aligned labels — which is
    the entire difference between the two heads' situations, and why this is a separate
    function rather than a flag on `fit_cuts` (a `fit_cuts` that accepted a bare prior would
    make the labeled-slice path and the no-slice path look interchangeable at the call site).
    """
    import numpy as np

    pred = np.asarray(pred, dtype=float)
    if pred.size == 0:
        raise ValueError("cuts_from_prior on an empty batch — a quantile cannot be read off "
                         "nothing")
    shares = [float(prior["shares"][str(t)]) for t in range(1, k + 1)]
    qs, run = [], 0.0
    for s in shares[:-1]:
        run += s
        qs.append(run)
    return tuple(float(q) for q in np.quantile(pred, qs))


def suggest_all(pred, cuts) -> list:
    """`tier_from_pred` over a whole batch — one place so a caller cannot half-apply it."""
    return [tier_from_pred(float(p), cuts) for p in pred]


def fit_slice(root: Path = ROOT) -> tuple:
    """`(pred, tiers)` — the labeled slice `CUTS` was fitted on, derived at call time.

    Raises on an absent source rather than fitting to the remainder: cutpoints read off part
    of a slice are not visibly wrong (`tier_prior`'s own reasoning, applied to the thing that
    replaced it)."""
    images = root / "data" / "render_mode_corpus" / "batches" / FIT_SLICE_BATCH / "images.jsonl"
    labels = root / FIT_SLICE_LABELS
    for p in (images, labels):
        if not p.exists():
            raise SystemExit(
                f"[suggest-tier-mining] fit slice absent: {p}. CUTS was fitted on the WHOLE "
                f"of {FIT_SLICE_BATCH}; a re-derivation over what survives would be a "
                f"different number, not a smaller one.")
    lab = json.loads(labels.read_text(encoding="utf-8"))
    pred, tiers = [], []
    for line in images.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = lab.get(row["image_id"])
        if t is None:
            continue
        pred.append(float(row["pred"]))
        tiers.append(int(t))
    return pred, tiers


def derive_cuts(root: Path = ROOT, ndigits: int = 4) -> tuple:
    """Re-derive `CUTS` from the live artifacts, so the frozen constant is checkable."""
    pred, tiers = fit_slice(root)
    return tuple(round(c, ndigits) for c in fit_cuts(pred, tiers, K_TIERS))


def fit_derivation(cuts, pred, ckpt: str, head_version: str, root: Path = ROOT) -> dict:
    """The record carried into a `batch.json` that uses the FITTED cuts. Everything computed
    from the arguments, the realized suggestion histogram included."""
    slice_pred, slice_tiers = fit_slice(root)
    served = Counter(suggest_all(pred, cuts))
    n = len(list(pred))
    fit_hist = Counter(suggest_all(slice_pred, cuts))
    return {
        "rule": "tier = 1 + #{c in cuts : expected_tier >= c}; "
                "expected_tier = 1 + sum_k marginal_k",
        "rule_owner": "tools/wallpaper/suggest_tier.py (expected_tier / tier_from_pred / "
                      "fit_cuts) — imported, not restated",
        "k_tiers": K_TIERS,
        "cuts": [float(c) for c in cuts],
        "cut_method": "fit_cuts — prior-matched quantiles of the head's own readout on a "
                      "LABELED SLICE, frozen and applied ABSOLUTELY. This replaces the "
                      "per-batch `cuts_from_prior` fallback, which had no labeled slice to "
                      "fit and therefore carried no information about a batch's absolute "
                      "quality level.",
        "derived": "2026-08-10",
        "deriver": "tools/mining/suggest_tier_mining.derive_cuts()",
        "fit_slice": {"batch": FIT_SLICE_BATCH, "labels": FIT_SLICE_LABELS,
                      "n": len(slice_pred),
                      "tier_prior": {str(t): sum(1 for x in slice_tiers if x == t)
                                     for t in range(1, K_TIERS + 1)},
                      "suggested_hist_on_the_fit_slice": {
                          str(t): fit_hist.get(t, 0) for t in range(1, K_TIERS + 1)}},
        "anchoring": "the fit slice's labels were collected on a CORRECTION sheet served with "
                     "the previous (per-batch quantile) suggestion prefilled; agreement with "
                     "what was served was 892/960 = 0.929, so these cuts are anchored to "
                     "their predecessor and largely reproduce it "
                     "((1.3893, 1.9331) -> (1.317, 1.8727)).",
        "head": {"ckpt": ckpt, "version": head_version},
        "realized_suggestion_hist": {str(t): served.get(t, 0) for t in range(1, K_TIERS + 1)},
        "realized_suggestion_shares": {str(t): (served.get(t, 0) / n if n else None)
                                       for t in range(1, K_TIERS + 1)},
        "n_rows": n,
        "scope": "a SUGGESTION, never a label — nothing downstream may read it as one, and "
                 "the merge refuses to.",
    }


def derivation(cuts, prior: dict, pred, ckpt: str, head_version: str) -> dict:
    """The record carried into `batch.json`. Freeze in records, derive in code.

    Everything here is COMPUTED from the arguments — the realized suggestion histogram
    included — so the record cannot claim a distribution the cuts did not produce."""
    hist = Counter(suggest_all(pred, cuts))
    n = len(list(pred))
    return {
        "rule": "tier = 1 + #{c in cuts : expected_tier >= c}; "
                "expected_tier = 1 + sum_k marginal_k",
        "rule_owner": "tools/wallpaper/suggest_tier.py (expected_tier / tier_from_pred / "
                      "fit_cuts) — imported, not restated",
        "k_tiers": K_TIERS,
        "cuts": [float(c) for c in cuts],
        "cut_method": "PER-BATCH quantiles of this batch's own expected_tier, placed at the "
                      "surviving mining corpus's tier prior (prior-matched). NOT fit_cuts: "
                      "fit_cuts needs aligned (pred, human tier) pairs and the mining head "
                      "has no such slice — its 1500 human tiers survive but every crop they "
                      "name is gone with data/render_mode_corpus/.",
        "scope_and_cost": "per-batch quantiles carry NO information about this corpus's "
                          "absolute quality level, only each row's rank within it. An "
                          "absolute cut would let a worse population receive more tier-1 "
                          "suggestions; that property is deliberately given up here because "
                          "there is no labeled slice to set an absolute cut against. A "
                          "suggestion, never a label — nothing downstream may read it as one.",
        "re_derive_when": "this sheet is labeled: a (pred, tier) slice on the new corpus is "
                          "the first thing that makes fit_cuts possible again",
        "derived": __import__("time").strftime("%Y-%m-%d"),
        "head": {"ckpt": ckpt, "version": head_version},
        "prior": prior,
        "realized_suggestion_hist": {str(t): hist.get(t, 0) for t in range(1, K_TIERS + 1)},
        "realized_suggestion_shares": {str(t): (hist.get(t, 0) / n if n else None)
                                       for t in range(1, K_TIERS + 1)},
        "n_rows": n,
    }
