"""suggest_tier.py — the wallpaper head's PRE-LABEL tier suggestion, TORCH-FREE.

A pre-labeled correction sheet shows the human a machine tier and asks him to fix the ones
it got wrong. That needs a rule mapping the head's CORN marginals to a tier in 1..4, and the
rule is a real decision — a different rule changes what Matt is anchored on for ~1000 rows.
So it lives here, once, with its derivation, rather than inline in a builder.

THE READOUT. `expected_tier(marg) = 1 + Σ_k marg[k]` — the head's continuous quality readout
(`prelabel_score_v2` calls it `pred`), monotone across all four tiers. `suggest(marg)` cuts
it at `CUTS`.

WHY NOT THE PLAIN CORN 0.5 RULE. `1 + #{k : marg[k] >= 0.5}` is the textbook ordinal
decision. Measured on the ONLY human-labeled slice v3 has never trained on (below) it is
both less accurate AND biased low: exact 0.376, MAE 0.755, mean(pred-true) -0.283, and it
puts 268/686 rows at tier 1 against a true 116. A pre-label that suggests "bad" for 39% of a
sheet where 17% are bad is an anchoring hazard, not a labor saving.

WHY NOT THE ACCURACY-MAXIMIZING CUT. Free cutpoints on the same readout reach exact 0.497 —
by collapsing: they place 476/686 rows at tier 2. A suggestion that is nearly constant
carries no information for the human to correct and biases every judgment the same way.
Accuracy bought by predicting the majority class is the wrong objective for a sheet whose
purpose is spanning the range.

THE RULE THAT IS USED — PRIOR-MATCHED CUTS. The cutpoints are the quantiles of the readout
at the eval slice's own tier prior, so the suggestion distribution REPRODUCES the label
distribution instead of collapsing onto one class. On v4b: exact 0.427, within-one 0.889,
MAE 0.691, and the suggested histogram lands {116, 295, 185, 90} against a true
{116, 295, 185, 90} — matched by construction.
The cuts are ABSOLUTE, not re-quantiled per batch: a genuinely worse population must receive
genuinely more tier-1 suggestions, which is what an absolute cut does and a per-batch
quantile would erase.

THE OBJECTIVE IS PRIOR REPRODUCTION, NOT EXACT AGREEMENT, and the 2026-08-11 head flip is
what made the distinction bite. Under v3 the chosen rule ALSO beat `corn_0.5` on exact
agreement (0.414 vs 0.376), so the two readings agreed and nothing forced a choice between
them. Under v4b they separate: `corn_0.5` scores 0.452 against the chosen rule's 0.427 and is
still rejected, because it puts 167 rows at tier 1 against a true 116 and 29 at tier 4 against
a true 90 — the anchoring hazard the paragraph above describes, unchanged in kind. An exact-
agreement win bought by mis-shaping the suggestion histogram is the same trade the
accuracy-maximizing cut makes, and it is refused for the same reason.

DERIVATION (frozen; see `DERIVATION`). RE-DERIVED 2026-08-11 at the wallpaper head flip
(v3 -> v4b seed 1, `prompts/flip_29.md`) on the SAME 686-render / 98-location slice, whose
crops now live in the six-batch union loader (`train_wallpaper_v4b.split_v4b`, the
dramatic+humanq3 eval rows) rather than in a batch of their own. The v3 cuts it replaces were
(1.017, 2.615, 2.997), fitted 2026-08-05 on the same slice under
`data/wallpaper_head/v3/model_best.pt`. True tiers {1:116, 2:295, 3:185, 4:90}, unchanged —
the slice is the same rows; only the readout moved.

  cuts = quantile(expected_tier, [116/686, 411/686, 596/686])

WHY IT HAD TO BE RE-DERIVED. `expected_tier` is a sum of CORN marginals, which are
train-prior-calibrated, so a cut on it is exactly as scale-bound as a probability floor
(`classifier_retrain_protocol.md` §5a). Keeping the v3 cuts under v4b would have served a
correction sheet a suggestion histogram nobody chose.

SCOPE, STATED (measurement_practice.md, "Labels are distribution-bound"). That slice is the
dramatic + humanq3 population — curated, top-heavy — and NOT the stage-2 intake a fresh sheet
draws from. These cuts are the best anchor this population has, not a calibration for an
arbitrary one. The honest use is a suggestion the human overrides; nothing downstream may
treat a suggested tier as a label. `INTAKE_CUTS` below is the stage-2 sibling.

  from tools.wallpaper.suggest_tier import suggest, expected_tier, CUTS, DERIVATION
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The cuts are cutpoints on ONE head's readout, so which head is live is part of the
# question, not context. Torch-free both sides (that is what `wallpaper_pins` is for).
from tools.wallpaper import wallpaper_pins as _wp     # noqa: E402

# Cutpoints on `expected_tier` ∈ [1, 4]. Tier = 1 + #{c in CUTS : pred >= c}. Ascending.
# Frozen from the derivation below — do not re-quantile per batch (see the module docstring).
CUTS = (1.2573, 2.4007, 2.91)

# The derivation record, carried into every batch.json that uses the rule so the number is
# never a bare literal in a run record. Freeze in records, derive in code
# (`storage_classes.md`); `fit_cuts` below is the deriver these came out of.
DERIVATION = {
    "rule": "tier = 1 + #{c in cuts : expected_tier >= c}; "
            "expected_tier = 1 + sum_k marginal_k",
    "cuts": list(CUTS),
    "cut_method": "quantiles of expected_tier at the derivation slice's own tier prior "
                  "(prior-matched), NOT accuracy-maximizing — see suggest_tier.py",
    "derived": "2026-08-11",
    "deriver": "tools/wallpaper/suggest_tier.derive_cuts()",
    "head": "data/wallpaper_head/v4b/seed_1/model_best.pt",
    "head_version": "v4b",
    "slice": "the dramatic + humanq3 EVAL rows of the six-batch union "
             "(train_wallpaper_v4b.split_v4b) — the same 686 renders / 98 locations the v3 "
             "cuts were fitted on, re-scored under v4b; readout sidecar "
             "data/wallpaper_head/v4b/july_slice_pred.json",
    "n": 686,
    "n_locations": 98,
    "tier_prior": {"1": 116, "2": 295, "3": 185, "4": 90},
    "accuracy_on_slice": {"exact": 0.427, "within_one": 0.889, "mae": 0.691,
                          "suggested_hist": {"1": 116, "2": 295, "3": 185, "4": 90}},
    "alternatives_rejected": {
        "corn_0.5": {"exact": 0.452, "within_one": 0.901, "mae": 0.655, "bias": -0.200,
                     "suggested_hist": {"1": 167, "2": 269, "3": 221, "4": 29},
                     "why": "REJECTED DESPITE A HIGHER EXACT AGREEMENT (0.452 vs 0.427) — "
                            "exact agreement is not the objective. It mis-shapes the "
                            "histogram in both tails: 167 tier-1 suggestions against a true "
                            "116, and 29 tier-4 against a true 90."},
        "accuracy_max_cuts": {"exact": 0.501, "cuts": [1.0757, 2.7387, 3.8593],
                              "suggested_hist": {"1": 69, "2": 445, "3": 154, "4": 18},
                              "why": "collapses — 445/686 (64.9%) suggested tier 2"},
    },
    "supersedes": {
        "cuts": [1.017, 2.615, 2.997], "head": "data/wallpaper_head/v3/model_best.pt",
        "derived": "2026-08-05", "n": 686,
        "accuracy_on_slice": {"exact": 0.414, "within_one": 0.899, "mae": 0.691,
                              "suggested_hist": {"1": 116, "2": 295, "3": 183, "4": 92}},
        "alternatives_rejected": {
            "corn_0.5": {"exact": 0.376, "within_one": 0.885, "mae": 0.755, "bias": -0.283,
                         "why": "biased low; 268/686 suggested tier 1 against a true 116"},
            "accuracy_max_cuts": {"exact": 0.497, "within_one": 0.915,
                                  "why": "collapses — 476/686 suggested tier 2"}},
        "why_superseded": "the 2026-08-11 wallpaper head flip (v3 -> v4b seed 1). "
                          "`expected_tier` is a sum of CORN marginals and is train-prior "
                          "calibrated, so these cutpoints are points on v3's readout scale.",
    },
    "scope": "derived on the dramatic+humanq3 population, applied to the stage-2 intake; "
             "a suggestion, never a label",
}

K_TIERS = 4

# --------------------------------------------------------------------------- #
# THE STAGE-2 INTAKE CUTS (2026-08-10) — the re-derivation the block above asked for.
# --------------------------------------------------------------------------- #
# "Re-derive with `fit_cuts` when a labeled slice from the new population exists." It does
# now. `CUTS` above was fitted on the dramatic+humanq3 eval slice, which is curated and
# top-heavy; the two 2026-08-05 sheets drew from the STAGE-2 ADMITTED INTAKE and Matt labeled
# all 1,140 of their rows. That is the population a fresh sitting draws from, so a sitting
# over it gets cuts fitted on it.
#
# THE TWO PRIORS ARE NOT CLOSE, which is the whole reason for the second set:
#   dramatic+humanq3 (n=686):  {1: 17%, 2: 43%, 3: 27%, 4: 13%}
#   stage-2 intake  (n=1140):  {1: 43%, 2: 38%, 3: 14%, 4:  5%}
# Prior-matching to the second means ~43% of an intake sitting is suggested tier 1 — which is
# the anchoring hazard the `corn_0.5` rejection describes ONLY when the suggestion overstates
# how bad the population is. Here it matches it.
#
# STATED PLAINLY: the old cuts score HIGHER on exact agreement over this same slice — 0.707
# against 0.687 — and that is not an argument for keeping them. Exact agreement is not the
# objective (see WHY NOT THE ACCURACY-MAXIMIZING CUT above); reproducing the label
# distribution is, and only these cuts do it here — {495, 429, 163, 53} against a true
# {493, 431, 163, 53}, i.e. matched up to the ROUNDING of the frozen cuts (two rows sit on the
# tier-1/2 quantile tie), against the old cuts' {519, 426, 131, 64}.
#
# THEY ARE STILL ABSOLUTE. Fitted once on a labeled slice, then applied unchanged to a new
# batch — never re-quantiled per batch. A stratified sitting that deliberately over-draws the
# good end MUST come out with more tier-3/4 suggestions than a uniform draw would, and that
# is exactly what an absolute cut preserves and a per-batch quantile destroys.
#
# CIRCULARITY, NAMED. Both source batches were CORRECTION sheets: their rows were served with
# a suggestion prefilled from `CUTS`, so the labels are anchored to it. Agreement with what
# was served was 0.733 (fresh sheet) and 0.567 (colorize path) — i.e. Matt corrected 27% and
# 43% of the rows, so the slice is anchored but far from a copy of the rule. A future
# re-derivation off a BLIND slice of the same population would be strictly better evidence and
# there is no reason it cannot exist.
# RE-DERIVED 2026-08-11 at the head flip, on the same 1,140 rows re-scored under v4b. The
# v3 cuts were (1.0119, 2.4663, 3.0012) and the whole of that record is kept under
# `supersedes` — the reason the second cut set exists is unchanged, only the readout moved.
INTAKE_CUTS = (1.2304, 2.4197, 2.9713)

INTAKE_DERIVATION = {
    "rule": "tier = 1 + #{c in cuts : expected_tier >= c}; "
            "expected_tier = 1 + sum_k marginal_k",
    "cuts": list(INTAKE_CUTS),
    "cut_method": "quantiles of expected_tier at the derivation slice's own tier prior "
                  "(prior-matched) via fit_cuts — ABSOLUTE, applied unchanged to new batches",
    "derived": "2026-08-11",
    "head": "data/wallpaper_head/v4b/seed_1/model_best.pt",
    "head_version": "v4b",
    "deriver": "tools/wallpaper/suggest_tier.derive_intake_cuts()",
    "slice": "the two 2026-08-05 stage-2 intake correction sheets, whole: "
             "2026-08-05_wallpaper_fresh_sheet_v1 (960) + "
             "2026-08-05_wallpaper_colorize_path_v1 (180), pred re-scored under the live head "
             "(data/wallpaper_head/v4b/fit_slice_pred.json), tier = the merged human sidecar",
    "n": 1140,
    "tier_prior": {"1": 493, "2": 431, "3": 163, "4": 53},
    "accuracy_on_slice": {"exact": 0.665, "within_one": 0.972, "mae": 0.363,
                          "suggested_hist": {"1": 493, "2": 431, "3": 163, "4": 53},
                          "hist_note": "the REALIZED histogram under the frozen 4-dp cuts; "
                                       "here it lands on the exact prior fit, unlike the v3 "
                                       "cuts, where two rows sat on the tier-1/2 tie"},
    "alternatives_rejected": {
        "corn_0.5": {"exact": 0.625, "within_one": 0.968, "mae": 0.407, "bias": -0.156,
                     "suggested_hist": {"1": 619, "2": 325, "3": 175, "4": 21},
                     "why": "biased low; 619/1140 suggested tier 1 against a true 493"},
        "accuracy_max_cuts": {"exact": 0.685, "cuts": [1.2162, 2.7392, 2.9544],
                              "suggested_hist": {"1": 485, "2": 541, "3": 57, "4": 57},
                              "why": "collapses the middle — 57 tier-3 against a true 163"},
    },
    "vs_the_dramatic_cuts": {
        "cuts": list(CUTS),
        "why_not_kept": "the two populations' priors are not close (see the module comment); "
                        "these cuts reproduce the intake prior exactly and the dramatic ones "
                        "do not. Both sets were re-derived on v4b at the same flip.",
    },
    "anchoring": {
        "both_sources_were_correction_sheets": True,
        "agreement_with_what_was_served": {"fresh_sheet": 0.733, "colorize_path": 0.567},
        "note": "the slice is anchored to the v3 CUTS it was served with, not a copy of them "
                "— 27% and 43% of the rows were corrected. A blind slice of the same "
                "population would be better evidence and nothing prevents one.",
    },
    "supersedes": {
        "cuts": [1.0119, 2.4663, 3.0012], "head": "data/wallpaper_head/v3/model_best.pt",
        "derived": "2026-08-10", "n": 1140,
        "accuracy_on_slice": {"exact": 0.687, "within_one": 0.994, "mae": 0.319,
                              "suggested_hist": {"1": 495, "2": 429, "3": 163, "4": 53}},
        "vs_the_v3_dramatic_cuts": {"cuts": [1.017, 2.615, 2.997], "exact": 0.707,
                                    "suggested_hist": {"1": 519, "2": 426, "3": 131,
                                                       "4": 64}},
        "why_superseded": "the 2026-08-11 wallpaper head flip (v3 -> v4b seed 1)",
    },
    "scope": "derived on the stage-2 admitted intake, applied to sittings drawn from it; "
             "a suggestion, never a label",
}

# (batch dir under data/wallpaper_corpus/batches, labels sidecar under labels/)
INTAKE_SLICE_SOURCES = (
    ("2026-08-05_wallpaper_fresh_sheet_v1", "wallpaper_fresh_sheet_v1"),
    ("2026-08-05_wallpaper_colorize_path_v1", "wallpaper_colorize_path_v1"),
)

# --------------------------------------------------------------------------- #
# WHERE A SLICE'S READOUT COMES FROM, per head version.
# --------------------------------------------------------------------------- #
# A fit slice needs `(pred, human tier)` pairs, and `pred` is a HEAD's number. The batches
# carry the readout of the head that BUILT them (`head_v3.pred`), stamped at sheet-build
# time; every later head's readout is a sibling record written by
# `tools/scoring/rescore_fit_slices.py` — never a rewrite of the batch row, which is the
# sheet's own record of what the human was anchored on.
#
# The version is in the PATH for the same reason `ledger_rescore`'s siblings carry it: the
# next flip's reader looks for its own file, does not find it, and RAISES rather than fitting
# cuts to another head's numbers under its name.
INTAKE_PRED_SOURCES = {
    "v3": ("in_row", "head_v3"),
    "v4b": ("sidecar", "data/wallpaper_head/v4b/fit_slice_pred.json"),
}
# The July (dramatic + humanq3) slice has no in-row source at all: its crops were re-rendered
# by the eval-revival pass and its v3 readout was never stamped anywhere, which is why the v3
# `CUTS` record could only ever be prose. From v4b on it is a sidecar like the other.
JULY_PRED_SOURCES = {
    "v4b": ("sidecar", "data/wallpaper_head/v4b/july_slice_pred.json"),
}


def live_head_version() -> str:
    """The head these cuts are cutpoints ON, read off the pin at CALL time."""
    return _wp.HEAD_VERSION


def _sidecar(rel: str, root=None) -> dict:
    p = (Path(root) if root else ROOT) / rel
    if not p.exists():
        raise FileNotFoundError(
            f"suggest_tier: readout sidecar absent: {rel}. It is a tracked durable record; "
            f"write it with `uv run python tools/scoring/rescore_fit_slices.py`.")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_source(table: dict, head: str | None, what: str):
    head = head or live_head_version()
    src = table.get(head)
    if src is None:
        raise KeyError(
            f"suggest_tier: no {what} readout registered for head {head!r} "
            f"(have {sorted(table)}). A cut on `expected_tier` is a point on ONE head's "
            f"readout scale; re-score the slice under {head!r} "
            f"(tools/scoring/rescore_fit_slices.py) and register it here rather than fitting "
            f"to another head's numbers.")
    return head, src


def intake_slice(root=None, sources=INTAKE_SLICE_SOURCES, head: str | None = None):
    """`(pred, tiers)` — the labeled stage-2 intake slice `INTAKE_CUTS` was fitted on.

    Derived from the batches, sidecars and the live pin at call time, never restated as
    literals: a hardcoded pair beside the paths it summarizes outlives the files the moment
    one changes (`storage_classes.md`, "derive state in code"). Raises on an absent source
    rather than fitting cuts to half the record — a prior read off part of a slice is not
    visibly wrong."""
    root = Path(root) if root else ROOT
    head, (kind, ref) = _resolve_source(INTAKE_PRED_SOURCES, head, "intake")
    readout = _sidecar(ref, root)["pred"] if kind == "sidecar" else None

    pred, tiers = [], []
    for batch, sidecar in sources:
        images = root / "data" / "wallpaper_corpus" / "batches" / batch / "images.jsonl"
        labels = root / "labels" / f"{sidecar}.json"
        for p in (images, labels):
            if not p.exists():
                raise FileNotFoundError(
                    f"suggest_tier: derivation source absent: {p}. INTAKE_CUTS was fitted on "
                    f"the whole of {[b for b, _s in sources]}; a re-derivation over the "
                    f"remainder would be a different number, not a smaller one.")
        lab = json.loads(labels.read_text(encoding="utf-8"))
        for line in images.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = lab.get(row["image_id"])
            if t is None:
                continue
            if readout is None:
                pred.append(float(row[ref]["pred"]))
            else:
                pred.append(float(readout[row["image_id"]]))
            tiers.append(int(t))
    return pred, tiers


def july_slice(root=None, head: str | None = None):
    """`(pred, tiers)` — the 686-row dramatic+humanq3 eval slice `CUTS` was fitted on.

    Read wholly from the readout sidecar, which carries the human tier beside the score: the
    rows are the union loader's eval side and there is no batch dir to walk. Same refusal on
    an unregistered head as `intake_slice`."""
    _head, (kind, ref) = _resolve_source(JULY_PRED_SOURCES, head, "july")
    if kind != "sidecar":
        raise KeyError(f"the july slice has no {kind!r} source")
    d = _sidecar(ref, root)
    ids = list(d["pred"])
    return [float(d["pred"][i]) for i in ids], [int(d["tier"][i]) for i in ids]


def derive_intake_cuts(root=None, ndigits: int = 4, head: str | None = None) -> tuple:
    """Re-derive `INTAKE_CUTS` from the live artifacts. Kept live so the frozen constant is
    checkable rather than remembered (`test_wallpaper_sitting.py` asserts the two agree)."""
    pred, tiers = intake_slice(root, head=head)
    return tuple(round(c, ndigits) for c in fit_cuts(pred, tiers, K_TIERS))


def derive_cuts(root=None, ndigits: int = 4, head: str | None = None) -> tuple:
    """Re-derive `CUTS` from the live artifacts — the sibling of `derive_intake_cuts`.

    New at the 2026-08-11 flip. Through v3 this set had no live deriver at all: its slice's
    readout was never stamped anywhere, so the constant could only be checked against prose.
    """
    pred, tiers = july_slice(root, head=head)
    return tuple(round(c, ndigits) for c in fit_cuts(pred, tiers, K_TIERS))


def expected_tier(marg) -> float:
    """`1 + Σ_k marginal_k` — the head's continuous quality readout in [1, K_TIERS].

    `marg` is the CORN MARGINAL vector (cumprod of the conditional sigmoids):
    `marg[0] = P(tier>=2)`, `marg[1] = P(tier>=3)`, `marg[2] = P(tier>=4)`."""
    return 1.0 + float(sum(float(m) for m in marg))


def suggest(marg, cuts=CUTS) -> int:
    """Suggested tier in 1..K_TIERS for one row's CORN marginals."""
    return tier_from_pred(expected_tier(marg), cuts)


def tier_from_pred(pred: float, cuts=CUTS) -> int:
    """The cut rule alone, on an already-computed `expected_tier`."""
    return 1 + sum(1 for c in cuts if float(pred) >= float(c))


def fit_cuts(pred, tiers, k: int = K_TIERS):
    """THE deriver `CUTS` came out of: prior-matched quantiles of `pred` on a labeled slice.

    `pred` is the per-row `expected_tier`, `tiers` the human labels (1..k), aligned. Returns
    the k-1 ascending cutpoints placing exactly the observed tier prior. Kept live (not
    prose) so the next labeled slice re-derives with one call instead of a re-invention."""
    import numpy as np

    pred = np.asarray(pred, dtype=float)
    tiers = np.asarray(tiers, dtype=int)
    if pred.shape != tiers.shape:
        raise ValueError(f"pred/tiers misaligned: {pred.shape} vs {tiers.shape}")
    n = len(pred)
    if n == 0:
        raise ValueError("fit_cuts on an empty slice — a prior cannot be read off nothing")
    counts = [int((tiers == t).sum()) for t in range(1, k + 1)]
    qs, run = [], 0
    for c in counts[:-1]:
        run += c
        qs.append(run / n)
    return tuple(float(q) for q in np.quantile(pred, qs))
