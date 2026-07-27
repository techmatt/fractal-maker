# Label rubric — the 1–4 quality scale

The single human-label scale used across the whole location corpus. One question, one scale, all
families (mandelbrot, the julia twins, phoenix, native multibrot, minibrot). The classifier is
trained on this scale as an **ordinal** target; the tiers are ranked, not nominal.

**The question the labeler answers:** *does this crop work as a wallpaper?*

Judge from the **vivid companion render** where one is shown — a crushing palette makes good
material look dead, and the label is a verdict on the *location*, not on one unlucky colormap.
The canonical (model-facing) crop is what the network will eventually see; the vivid one is what
you judge from.

## Tiers

| Score | Name | Meaning |
|-------|------|---------|
| 1 | bad | Does not work. Dead/black-dominated, structureless, muddy, or broken. Would never be shipped. |
| 2 | okay | Has structure but is unremarkable — a competent fill, not something you'd choose. Below the emission floor. |
| 3 | good | A genuine wallpaper. Clean composition, real structure, ships. This is the emission floor. |
| 4 | exceptional | An exceptional wallpaper emission — the best of the "good" material, the ones worth surfacing first. |

## Class 4 — the design intent (settled)

- Class 4 is a **fourth tier on the same quality scale**, not a separate network or head. It is
  scored by the same labeler answering the same question, and consumed by the same ordinal head.
- Class 4 is **preferred where available but is not a new floor.** Final emissions can be class 3
  or class 4; class 4 is not a gate that rejects class-3 wallpapers. It ranks the top of "good."
- The `>=3` (good/emit) boundary is unchanged by the introduction of class 4. Reconstructing the
  pre-class-4, pre-revision `>=3` boundary must stay a one-liner
  (`label_store.resolve_score(row, sidecar) >= 3`).

## Class 4 — aesthetic criteria

<!-- TODO(matt): fill in during the anchor pass. The ~60-image mixed-family anchor batch fixes
     the class-4 bar ACROSS families before the large minibrot volume is labeled, so the bar is
     not silently defined by minibrot crops. Write the concrete "what makes a 4 not just a 3"
     criteria here once that pass is done. -->

_Stub — to be filled in during the anchor pass (Part B of the class-4 rollout)._

## Revisions

Existing labels may be revised on this scale (a q3 demoted to q2, or promoted to q4). Revisions
never modify the original label file; they go to the amendment stream and `resolve_score` prefers
them. See `data/label_corpus/CORPUS_SCHEMA.md` (§ Revisions) and `tools/corpus/merge_amendments.py`.
