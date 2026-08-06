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
distribution instead of collapsing onto one class. Exact 0.414, within-one 0.899, MAE 0.691,
and the suggested histogram lands {116, 295, 183, 92} against a true {116, 295, 185, 90} —
matched by construction up to the rounding of the frozen cuts.
The cuts are ABSOLUTE, not re-quantiled per batch: a genuinely worse population must receive
genuinely more tier-1 suggestions, which is what an absolute cut does and a per-batch
quantile would erase.

DERIVATION (frozen; see `DERIVATION`). 2026-08-05, head `data/wallpaper_head/v3/model_best.pt`
(CORN K=4), over the 686-render / 98-location held-out eval slice re-rendered by the eval
revival pass (`prompts/wallpaper_eval_revival_prompt.md`: dramatic rows with stamped
`provenance.split_side == "eval"` + humanq3 rows under `split_v2(seed=0, eval_frac=0.30)`,
crops via `label_crop.render_label_crop`). True tiers {1:116, 2:295, 3:185, 4:90}.

  cuts = quantile(expected_tier, [116/686, 411/686, 596/686])

SCOPE, STATED (measurement_practice.md, "Labels are distribution-bound"). That slice is the
dramatic + humanq3 population — curated, top-heavy — and NOT the stage-2 intake a fresh sheet
draws from. These cuts are the best anchor v3 has, not a calibration for an arbitrary
population. The honest use is a suggestion the human overrides; nothing downstream may treat
a suggested tier as a label. Re-derive with `fit_cuts` when a labeled slice from the new
population exists.

  from tools.wallpaper.suggest_tier import suggest, expected_tier, CUTS, DERIVATION
"""
from __future__ import annotations

# Cutpoints on `expected_tier` ∈ [1, 4]. Tier = 1 + #{c in CUTS : pred >= c}. Ascending.
# Frozen from the derivation below — do not re-quantile per batch (see the module docstring).
CUTS = (1.017, 2.615, 2.997)

# The derivation record, carried into every batch.json that uses the rule so the number is
# never a bare literal in a run record. Freeze in records, derive in code
# (`storage_classes.md`); `fit_cuts` below is the deriver these came out of.
DERIVATION = {
    "rule": "tier = 1 + #{c in cuts : expected_tier >= c}; "
            "expected_tier = 1 + sum_k marginal_k",
    "cuts": list(CUTS),
    "cut_method": "quantiles of expected_tier at the derivation slice's own tier prior "
                  "(prior-matched), NOT accuracy-maximizing — see suggest_tier.py",
    "derived": "2026-08-05",
    "head": "data/wallpaper_head/v3/model_best.pt",
    "slice": "wallpaper head v3 held-out eval: dramatic split_side==eval + humanq3 "
             "split_v2(seed=0, eval_frac=0.30), crops re-rendered via "
             "label_crop.render_label_crop (prompts/wallpaper_eval_revival_prompt.md)",
    "n": 686,
    "n_locations": 98,
    "tier_prior": {"1": 116, "2": 295, "3": 185, "4": 90},
    "accuracy_on_slice": {"exact": 0.414, "within_one": 0.899, "mae": 0.691,
                          "suggested_hist": {"1": 116, "2": 295, "3": 183, "4": 92}},
    "alternatives_rejected": {
        "corn_0.5": {"exact": 0.376, "within_one": 0.885, "mae": 0.755, "bias": -0.283,
                     "why": "biased low; 268/686 suggested tier 1 against a true 116"},
        "accuracy_max_cuts": {"exact": 0.497, "within_one": 0.915,
                              "why": "collapses — 476/686 suggested tier 2"},
    },
    "scope": "derived on the dramatic+humanq3 population, applied to the stage-2 intake; "
             "a suggestion, never a label",
}

K_TIERS = 4


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
