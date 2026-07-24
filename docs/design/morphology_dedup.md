# Morphology & visual dedup

Distilled from `visual_dup`, `prospect_run1_morph_composition_audit`, the phoenix-viewport
rule in `curate_emission`, and the `julia_dup_metric_audit` keying bug. Governs how
"is this the same *look*?" is measured, and why coordinate dedup is not enough.

> **Load-bearing-code warning.** The calibration / embedding code for this axis lived
> only in `scratchpad/visual_dup/embed.py` and *vanished* once — this is the origin of
> the CLAUDE.md scratchpad rule. **Promote any morph-embedding code to `tools/` before
> relying on it.**

## 1. Coordinate dedup is not visual dedup — a morph gate is *additive*

Coordinate-distinct-but-visually-identical is a **real redundancy mode**. A morph-dedup
gate catches locations a coordinate gate passes through: **16% (strict) to 47%
(perceptual, cos>0.95)** of coord-distinct locations are morphological repeats.
Therefore record / coordinate count **overstates delivered visual variety** — most
acutely for phoenix (~40% of records collapse to ≈one theme). Any selection denominated
in morphological coverage must run a morph gate, not trust record count.

**But do not over-collapse.** At the *wallpaper* level the redundancy is largely masked
because the emission selector's color diversification hands same-skeleton pairs
different palettes. A geometry-only dedup gate would kill legitimate color variants —
gate **conservatively**, or better, **add a morphology axis to the selector's niche**
rather than post-hoc dedup.

## 2. The similarity space is cone-compressed — only relative order is trustworthy

The grayscale morph_clip (CLIP) space is **cone-compressed**: the corpus spans cosine
~[0.56, 0.99] as a smooth right-skewed continuum with **no bimodal valley**. So:

- **Absolute cosine is not a similarity** and there is **no natural dedup cut** — only
  relative ordering is trustworthy (cross-family controls ~0.75–0.84 vs near-dups
  ~0.987).
- Every cluster count is **threshold-conditional**. Use `cos > 0.974` as the near-dup
  cut, but note it splits on **framing, not morphology**, so it *overstates* perceptual
  variety.

## 3. Descriptor choice — CLIP fits, the in-house backbone does not

- Compute on a **canonical grayscale render** (deterministic robust-z-score on the
  smooth field) so the descriptor is **palette-blind**.
- Use **zero-shot CLIP ViT-B/16** (control dups ≥0.92, graded tail).
- The **in-house fractal-trained (v6) backbone is unfit on grayscale** — it saturates
  to cos≈1.0 across families. Do not use it for morphology similarity.

## 4. Intrinsic redundancy sources to expect

- **Phoenix collapses to a single morphology.** Fixed Ushiki `(c,p)` ⇒ every "location"
  is one system's viewport. Good phoenix output is overwhelmingly **one log-spiral
  (double-scroll whorl) theme re-framed** — treat phoenix as a depth/quantity vein,
  never a morphological-breadth source.
- **Good morphology is intrinsically narrow** — recurring log-spiral / radial-starburst
  types recur across families.

## 5. Dedup *keys* must match rendered identity (two burned failure classes)

A dedup key that is semantically wrong but aggregate-plausible silently over-merges
distinct images. Two burned cases, pointing opposite directions:

- **Julia (over-merge on too little):** keying a Julia dup on its **z-viewport only**
  merged genuinely distinct sets that share a viewport but differ in seed `c`. A Julia
  location's identity is **its z-viewport AND its seed `c`** — both must be in the key.
- **Phoenix (safe only by accident):** phoenix dedup keys on the z-plane viewport
  `(cx,cy,fw)`, **not** on `(c,p)`. Because phoenix pins a fixed Ushiki `(c,p)` while
  `fw` spans decades, identical `(c,p)` does *not* imply an identical image — a
  parameter key would over-merge distinct zoom levels. The `(c,p)`-absent key is only
  safe **while `(c,p)` stays constant**; any sampler that varies phoenix `(c,p)` must
  put it back in the key.
