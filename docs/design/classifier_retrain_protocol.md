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

> **Before pre-registering a bar, verify the instrument's inputs actually change.**
>
> Pre-registration protects against moving the bar after seeing the numbers. It does
> **not** protect against a bar that cannot see the intervention at all — and that
> failure looks exactly like success. A "NON-INFERIOR" verdict computed on inputs
> identical to the baseline's is *true and empty*: it reports retrain-to-retrain
> variance and says nothing about the change under test.
>
> **The cheap check is a pixel/byte delta on the eval slice, computed *before* the
> run.** Hash the eval-slice tiles under both conditions and count how many differ. If
> the answer is zero, the instrument is blind and the bar must be rebuilt on a slice
> whose inputs move — *before* spending the training run, not after.
>
> **Why:** v9 (the cap-raise retrain, `auto_maxiter.md`) passed its pre-registered
> PRIMARY arm — census-144 AUC 0.7509 → 0.7390, p = 0.706, NON-INFERIOR — and the
> verdict was worthless. All 144 census tiles were **byte-identical** between the v8
> and v9 caches: the census is entirely `julia:multibrot3/4/5`, already converged at
> maxiter 8000, so raising the cap changed nothing there. The tell was a diagnostic
> arm returning *exactly* 0.0000. **An exact zero is not a null result — it is a
> measurement of nothing**, and it should be treated as a failed instrument check
> rather than a clean baseline.
>
> **How to apply:** when the intervention is a *render-path* change (cap, coloring,
> AA, resolution) rather than a data or architecture change, the eval slice's
> composition decides whether the experiment is answerable. Diff the slice's rendered
> inputs first; pick the slice to include material the change actually moves; and rank
> the arm whose inputs moved as PRIMARY, not SECONDARY.

## 4. `t_good` decode thresholds are scale-bound — re-derive every version

Per-partition `t_good` thresholds are **calibrated to a specific score scale**. A new
head's `p_good` distribution shifts, so **reusing old cuts silently starves recall**.
Re-derive every version:

- **F_beta-argmax** over a `p_good` grid, tie-break toward higher `t`, only where the
  slice has **≥15 positives**. `beta` is chosen per partition — see below.
- Expose in-sample optimism with **leave-one-out / OOF**, and report the argmax
  **plateau width**: tie-breaking high puts the adopted `t` at the plateau's upper edge by
  construction, so the plateau is the only honest read on how knife-edged the pick is.
- **Fall back to baseline for undecidable partitions, and stamp them UNCALIBRATED.** A
  baseline 0.50 and a derived 0.50 are the same character sequence in a config file; the
  distinction has to be carried explicitly or it is lost. See
  `production_seeder.T_GOOD_UNCALIBRATED`.

### The objective is per-family, and the axis is supply

> **Weight recall where supply is scarce, weight precision where supply is abundant.**

A false admit costs the same everywhere — one bad location wasting one render and one
human glance. What differs by family is the cost of a **miss**. Mandelbrot supply is
effectively unlimited: a missed mandelbrot costs nothing because the next hunt finds more,
so mandelbrot is derived **precision-weighted (F0.5)**. `julia:multibrot` supply saturates,
so a missed one is gone, and those are derived **recall-weighted (F2)**.

This is why a *uniform* objective is wrong rather than merely blunt. Uniform-F2 on
mandelbrot lands at `t=0.14`, precision 0.292 — roughly three and a half bad locations
admitted per good one, on the largest family in the corpus. v7 reached a split objective
by a different route (blind-label evidence that F2 over-admitted on mandelbrot); the
supply argument is the general form of that, and it survives the arrival of the v8
mandelbrot eval floor, which removed the *evidentiary* reason for the v7 exception but not
the *economic* one.

Future derivations inherit the principle, not the numbers. Where in-sample and OOF
disagree, prefer the OOF-honest choice and **say so in the report**.

(Current durable outcome: the v8 table in `production_seeder.T_GOOD_OVERRIDES` —
mandelbrot 0.85 via F0.5, `julia:multibrot{3,4,5}` 0.39/0.14/0.20 via F2, five partitions
UNCALIBRATED. The *values* live in `production_seeder.py`, the *derivation* in
`tools/v8/derive_t_good_v8.py` → `data/v8/t_good_derivation.json`, and
`tools/v8/test_derive_t_good_v8.py` holds the two in agreement. The *method* lives here.)
