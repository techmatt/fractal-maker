# Classifier retrain & versioning protocol

Distilled from the v7 build line (`v7_retrain_scope`, `v7_t_good`, and the
`mandelbrot_tgood_steered` t_good re-derivation). Reusable **every version** — the
specific v7 numbers live in code (`data/classifier/*`, `production_seeder.T_GOOD_OVERRIDES`);
this is the durable method. Companion: `aesthetic_scoring.md` (how to read `p_good`).

## 1. Append, never rebuild — freeze the prior-version manifest prefix

Retrain by **appending post-freeze labels to a byte-frozen prior-version manifest
prefix**. Freezing every prior location's `split` / `group_id` / row-order preserves
the version-to-version eval-comparability chain and carries the large working classes
(mandelbrot, J0) forward on an **identical eval set**, so "did it regress?" is
answerable by a **paired** test. Enforce a **frozen-prefix byte gate** on both the
manifest and the cache-manifest rows.

## 2. Split assignment — the cardinal sin is biased-in-eval

- Force **unbiased / base-rate draws → eval**.
- Force **model-selected or negative-by-construction batches → train**.
- Partition the group union-find by `(fractal_type, split, c-bucket)` so forced splits
  cannot **transitively straddle** train/eval.
  
  Children inherit their seed's split — any parent-derived location (julia twins, dives off a seed) takes the seed's train/eval assignment, so a descendant can't leak the seed's morphology across the split. (The c-bucket union-find mostly enforces this, but stating it explicitly closes the gap where a child's c drifts into a different bucket.)
Manifest-build gates abort-all — if any build gate fails, abort the whole build rather than emit a partial manifest.

Labels attach to **locations**, and training **re-renders from coordinates** — so a
batch's stored crops never constrain training. But the **deploy presentation point**
(geometry / palette / AA) must be covered by the aug fan-out, or the residual covariate
shift stated explicitly.

## 3. Pre-register the success bar before training

Set the credible-win bar **before** training, from **paired DeLong power** for the
q3-vs-rest AUC on the eval slice (e.g. n≈144 ⇒ ~AUC 0.68 as the bar). Then:

- A **null / ambiguous** result means **"label more," not "model failed."**
- Distinguish **eval *power*** (needs more labels) from **train *signal*** (a new
  positive class can be learnable yet unprovable on a small eval).

## 4. `t_good` decode thresholds are scale-bound — re-derive every version

Per-partition `t_good` thresholds are **calibrated to a specific score scale**. A new
head's `p_good` distribution shifts, so **reusing old cuts silently starves recall**.
Re-derive every version:

- **F2-argmax** (recall-weighted — discarding a good location costs more than admitting
  a bad one), only where the slice has **≥15 positives**.
- Expose in-sample optimism with **leave-one-out / OOF**.
- **Fall back to baseline (no invented value)** for undecidable partitions.

(Example durable outcome now in code: `T_GOOD_OVERRIDES["mandelbrot"] = 0.51`,
re-derived via F0.5 with steered_run2 blind labels; phoenix 0.45 via F2-argmax on the
grid — the *values* live in `production_seeder.py`, the *method* lives here.)
