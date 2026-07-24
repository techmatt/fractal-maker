# Aesthetic scoring — how to read the signal (metric cautions + classifier semantics)

Distilled from `fair_rerender_richness` and `steered_run2_keeper_calibration`. Governs
how any pixel-space richness metric and the CORN classifier's `p_good` may and may not
be used. Companion: `classifier_retrain_protocol.md` (how the classifier is *built*);
this doc is about how its output is *interpreted*.

## 1. Occupancy / busy-ness metrics are anti-quality — report-only, never a gate

Occupancy, busy-ness, and decoration-area metrics are **orthogonal to — and partly
inverted from** — the composed-hub-with-calm-surround aesthetic that defines a good
wallpaper:

- They **bury a strong money-shot mid-pack** because its calm surround reads as low
  occupancy.
- They are **inflated by large dead-black interiors** whose non-black remainder is dense.

A richness statistic that tracks the target taste needs the **opposite sign on
flat/calm regions** plus a **concentrated-detail (anti-distributed)** term — and only
human labels can teach that. **Therefore occupancy-style stats are report-only and must
never be used as a quality gate.** (This is the same sign-flip that kills composite
transfer to deep Mandelbrot — see `deep_zoom_sourcing.md` §5.)

**Corollary — judge on a fair render.** A striking "N useless" verdict can be a pure
**palette artifact**: a dark low-contrast ramp crushes mid-tone filigree to
invisibility. Make quality judgments on a fair/vivid render, never a muddy one.

## 2. `p_good` is a badness filter, not a goodness ranker

The CORN classifier's `p_good` behaves as a **badness filter, not a goodness ranker**
on descent / steered output:

- The **low-`p_good` band is reliably weak** (bad-rate high, good-rate ~0).
- Above that band, **higher `p_good` does not mean better** — Spearman with human
  labels is carried almost entirely by the bad end (~+0.4 pooled, collapsing to ~0 on
  deep-only sets).

Consequences:

- **No single `p_good` threshold cleanly isolates human-good.** The keeper / operating
  cut tops out near **~30% precision at 100% recall** regardless of placement. Raising
  the cut is **not** the lever.
- Treat any `p_good` gate as **"not-clearly-bad," never "good."** Keep keeper tiers
  **report-only** — do not promote them to hard quality gates.
- Delivering confident-good requires a **dedicated preference / ranking head** beyond
  the ordinal classifier, or a human pass.

(`P(not-bad) = σ(logit₀)` is the black-box for "not-clearly-bad"; the monotone
`score_from_logits ∈ [0,2]` is a rank score, not a calibrated goodness — see `CLAUDE.md`
§classifier.)
