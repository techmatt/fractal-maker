# `auto_maxiter` — the depth-aware iteration cap

Governs the **iteration cap policy** every render resolves through, the measurement that
showed it was under-provisioned, the adopted raise, and the residual that raise does not
cover. Named for the function, not the concept: the policy has no config file, it is a
handful of module constants and a five-line closed form, replicated at several sites.

> ### ⚠ Corrections (2026-07-31)
>
> This document was first written from a model of the system that the follow-on work
> then **disproved**. The corrections are folded into the sections below and marked
> **[CORRECTION]** where they land, rather than silently rewritten — an argument whose
> premises changed is worth being able to re-read. In summary:
>
> 1. **The augmentation cache never used `auto_maxiter`.** Not for v4, not for v8. Every
>    training tile was rendered at a flat 8000. The training cache and the deploy path
>    have therefore **never** agreed on the cap — a pre-existing skew, not one this raise
>    introduced. → *The aug-cache flat cap*
> 2. **`tools/orbital/` does not run a 24×/clamp-67000 policy.** It measures at **1×**,
>    through `rc.auto_maxiter`. No such constant exists anywhere in the tree.
>    → *Where it lives*
> 3. **The raise's largest effect on the corpus is one nobody predicted:** it *shrank*
>    v8's own train/deploy skew, from 10.0 % to 3.3 % of pixels. v8 is better aligned
>    now than it was before the change. → *What the raise actually changes, in pixels*
> 4. **The 32-atom ladder was found and is now committed.** It had been sitting in the
>    disposable `scratch/` tree. → *The measurement*
> 5. **v9 is trained, evaluated, staged — and deliberately not deployed.**
>    → *Why v9 is shelved*
> 6. **The label corpus holds two render regimes, and new batches will be a third.**
>    → *…and the label corpus is mixed the same way*

## The form

```
maxiter(fw) = clamp( base * (1 + k * log2(FW_HOME / fw)), min, max )
FW_HOME = 3.0
```

`fw` is the frame width in plane units, so `log2(FW_HOME/fw)` is the zoom depth in
octaves. The cap therefore grows **linearly in octaves**, which is the right shape: escape
times at a given visual scale grow roughly like the log of the magnification, not like the
magnification.

| | base | k | min | max |
|---|---|---|---|---|
| **before** (v4 … v8) | 500 | 0.30 | 200 | 8000 |
| **adopted** (2026-07-31) | **4000** | 0.30 | 200 | **67000** |

`k` is unchanged — the **shape** was never the problem.

### Where it lives

The policy is duplicated, not shared. Two sites are load-bearing and are the ones this
raise edits:

| site | role |
|---|---|
| `tools/scoring/active_ckpt.auto_maxiter` | **production.** The label-crop / corpus-crop / discovery-render cap. `tools/descent/store.py`, `build_native_multibrot_band.py`, `rescore_gather_mb4_v7.py`, and every `tools/wallpaper/build_*.py` import it. |
| `tools/explorer/render_core.auto_maxiter` | the shared explorer + descent **navigation** cap; `tools/orbital/` measures through it, **at 1×**. |

They are two independent copies of the same numbers, so they are pinned to agree by
`tools/scoring/test_maxiter_policy.py`. Three further copies exist and are **not**
production: `tools/explorer/app.py` (superseded by `render_core`, kept for the app's own
import), `tools/emission/descriptor.py`, `tools/julia_ladder/build_j0.py`,
`tools/studies/q4_neighborhood_sweep.py` — see "Sites deliberately left alone" below.

The **augmentation cache does not use this policy at all**; see "The aug-cache flat cap".

**[CORRECTION] `tools/orbital/` measures at 1×, not 24×.** Both
`tools/orbital/screen_pool.py` and `tools/orbital/measure_atoms.py` size every field dump
as `rc.auto_maxiter(fw)` with no multiplier (`measure_atoms`'s `maxiter_mult` defaults to
`1.0`; the ×2/×4 passes exist only inside `stability_check`). **There is no 24× constant
and no 67000 clamp anywhere under `tools/orbital/`.**

Where the belief came from, since it was specific enough to be worth naming: the
convergence study *fitted* a scoring-only envelope of `mult_of_prod = 24.0,
clamp_max = 67000` — "ceil(max conv/prod ratio) × production cap" — and wrote it to
`scratch/rescore/scoring_cap.json`. Nothing outside that scratch directory ever read it.
It was a **proposal that was never adopted**, in a file that was never committed — and it
became a stated property of the system anyway, which is the failure
`docs/design/storage_classes.md` names as "a proposal must never leave `scratch/` as a
fact". The loader that would have read it, `rescore_lib.scoring_maxiter`, had no caller
and in the file's absence returned ×8 of the **raised** production cap (200000 at
`fw = 8e-10` against production's 42165) — a dead function returning a wrong number. It
was deleted on 2026-07-31; `tools/orbital/test_rescore_lib.py` pins the deletion.

This matters beyond bookkeeping: because `tools/orbital/` measures through the *live*
production constants, the whole measurement stack **followed the raise silently**. Every
committed orbital score predates it and is not comparable to anything measured from here
on. That is now mechanical rather than remembered — each score record carries a
`maxiter_policy_token`, the committed ones are stamped legacy, and any comparison or
aggregation across two policies raises (`field_metrics.require_one_policy`).

## The measurement

Measured on **32 atoms spanning fw 3.3e-10 … 0.76**, each walked up a cap ladder until
`radial_rings` (`tools/orbital/field_metrics.py`) stopped moving. **All 32 converged.** The
convergent cap is a near-constant **multiple** of the then-production cap:

| statistic | multiple of production |
|---|---|
| mean | **7.7** |
| median | **8.0** |
| max | **24** |
| decorated material | 1.78 – 2.35 |
| flat triage wall | 1.14 (median) |

Three things follow, and they are the whole argument:

1. **The `fw` shape is fine.** A near-constant multiple across ten decades of `fw` is
   exactly what a correct `k` with a wrong `base` looks like. Had `k` been wrong, the
   multiple would trend with depth.
2. **The 8000 clamp is not the culprit.** No measured atom reached it. Over the v8
   location manifest the old policy's *maximum* value is **5424** — the clamp was never
   binding, so raising the clamp alone would have changed nothing.
3. **Everything was clipped.** The clip is universal (median ×1.14 even on the flat triage
   wall) and worst on decorated material (×1.78–2.35) — i.e. exactly the class-3 / class-4
   boundary the aesthetic classifier exists to resolve. Every crop in the label corpus and
   every production render was made under a clipped cap.

**[CORRECTION] The ladder is a committed artifact after all.** It was previously recorded
here as an uncommitted session measurement, with this table as its only record. It was
found intact in the disposable tree (`scratch/rescore/converge.json`) and promoted on
2026-07-31:

| | |
|---|---|
| artifact | **`data/orbital/maxiter_convergence_ladder.json`** (45 KB, durable, tracked) |
| producer | **`tools/orbital/measure_convergence_ladder.py`** (+ `rescore_lib.py`), promoted from `scratch/rescore/` |
| contents | 32 atoms × the multiplier ladder `[1, 2, 4, 8, 12, 16, 24, 32, 48]`, each point carrying `maxiter`, `rings`, `cycles_spanned`, `escaped_px`, plus per-atom `conv_mult` / `conv_maxiter` / `converged` |

Every figure in the table above reproduces from it exactly: **32 atoms, fw 3.31e-10 …
0.756, 0 not converged, ratio mean 7.688, median 8.0, max 24.0.** The ×8 rests on
committed evidence, not on a memory of a measurement.

⚠ **It is not reproducible by re-running the producer at its defaults**, and the artifact
is stamped `maxiter_policy_token: ""` to say so. Every ratio in it is a multiple of the
*legacy* production cap; the live policy now returns ~8× what it did when the ladder was
walked, so a default run measures convergence against the new policy and would report
ratios near 1. That is a different measurement, not a refutation — and it is why the file
is durable rather than regenerable.

Since 2026-07-31 the producer takes the cap policy as a **parameter** rather than reading
the live constants, so the legacy measurement is repeatable
(`measure_convergence_ladder.py --legacy-policy` → `(500, 0.30, 200, 8000)`) and a
deliberate re-measurement under the raised cap is possible. Output is stamped with its
policy token and defaults to `scratch/orbital/`; writing a non-legacy measurement over the
committed artifact is refused, because the two are different quantities and not versions
of one. The committed artifact is unchanged.
`[code: tools/orbital/measure_convergence_ladder.py; tools/orbital/test_rescore_lib.py]`

**Corroborating independent evidence:** `data/orbital/maxiter_stability.json` (n=24, at ×1
/ ×2 / ×4) shows `radial_rings` still climbing at ×4 — 45.0 → 55.25 → 60.75 — with no sign
of a plateau. Consistent with, and methodologically independent of, the ×8 ladder.

## The adopted policy, and why 4000

`base` 500 → **4000** is ×8: the **median** of the measured convergent multiple. `k`
unchanged. `max` 8000 → **67000** so the raised base is not immediately re-clipped by the
old clamp — at base 4000 the clamp must sit above 8 × 5424 ≈ 43k for the corpus, and 67000
leaves headroom for deeper future material.

Verified non-binding over the v8 manifest's actual `fw` distribution (7,117 locations,
fw 3.93e-10 … 4.24):

| | old policy | adopted policy |
|---|---|---|
| min | 425 | 3,400 |
| max | 5,424 | **43,397** |
| mean | 1,961 | 15,687 |
| rows at clamp | 0 | **0** |

43,397 < 64,000, so **67000 is non-binding** on the corpus as it stands. If a future
manifest pushes past ~64k the clamp starts truncating the deep tail and this decision has
to be revisited rather than silently absorbed.

## ⚠ The residual: median-clean, not clean

The adopted ×8 covers the **median** (mean 7.7 / median 8.0), but the measured tail runs
to **×24**. The most decorated material is therefore **still somewhat clipped** — the raise
moves the typical location onto its converged field and leaves the extreme tail short by up
to ×3.

**The cap is median-clean, not clean.** This is a deliberate cost/coverage trade (cap is
linear in render time, and ×24 across the whole corpus is a 3× render bill for the benefit
of a tail), not an oversight. A future reader chasing residual clipping on heavily
decorated locations should start here and not re-derive it.

## The aug-cache flat cap (a separate, pre-existing axis)

**[CORRECTION]** This section replaces the original document's assumption that the
training cache was rendered through `auto_maxiter` like everything else. It was not, and
never was.

`v4-render-batch` (`src/v4_cache.rs`) renders **every** plan row at a single
`--maxiter`, defaulting to **8000**, and **no v4–v8 plan carries a per-row cap** — not
just v8's. So every v4…v8 augmentation cache was rendered at a **flat 8000 regardless of
`fw`**, never touching `auto_maxiter`. The training cache was therefore roughly **10× over
production on shallow locations and short only on deep ones**. Consequences, which were
never written down:

* Shallow training tiles were rendered at ~10× the cap their deploy-time crop used
  (flat 8000 vs ~800), and deep tiles at ~1.5× (flat 8000 vs ~5400). The training cache and
  the deploy path have **never** agreed on the cap.
* Deep tiles were still clipped: flat 8000 is far below the ~43k the ×8 policy asks for at
  fw ≈ 4e-10.

The v9 re-render closes this by emitting a per-row `maxiter` into the plan from the
production policy, so the cache tile and the deploy crop resolve the **same** cap. The
`--maxiter` argument survives as the fallback for a plan row that omits it, which keeps
every pre-v9 plan byte-reproducible.

### …and the label corpus is mixed the same way

The flat cap is not confined to the aug cache. Measured over the live labeled corpus
(`corpus_reader.iter_labeled()`, 8,467 crops): **4,880 — 57.6% — carry `maxiter == 8000`
exactly**, the flat present/gather crop cap, while the remaining 3,587 carry 1,038 distinct
per-location `auto_maxiter` values. The corpus has two render regimes in it:

| cap regime | who writes it |
|---|---|
| `auto_maxiter(fw)` | `build_native_multibrot_band.py`, `descent/store.py`, `rescore_gather_mb4_v7.py`, every `wallpaper/build_*.py` |
| flat **8000** | `gather_select.py`, `recolor_gather_v6.py`, `build_enrich_batch.py`, `build_rev4_batch.py`, `sourcing/build_minibrot_batch.py`, `eda/scale_2x2_build_batch.py`, `mining/score_lib.run_enrich_score`, `mining/harvest.py` |

This does **not** contaminate training — the classifier trains on aug-cache tiles, and
`render.maxiter` is not one of the fields it sees. It bears on **label quality**: a human
judged some locations through a flat-8000 crop and others through an old-`auto_maxiter`
crop, so the two groups were presented at systematically different clip levels. Out of
scope for the cap raise, recorded here so it is not rediscovered as a surprise.

**[CORRECTION] and there will be a third regime.** The two above are *flat 8000* (4,880
crops) and *old-`auto_maxiter`* (3,587 crops across 1,038 distinct values). Every batch
labeled from 2026-07-31 onward is rendered at **new-`auto_maxiter`**, which is neither.
Stated as one line of fact with **no action implied** — this is not a call to re-render
the corpus, and the classifier still never sees `render.maxiter`. It is here so that the
next person who counts distinct caps in the label store finds three groups and an
explanation, rather than three groups and a mystery.

## Sites deliberately left alone

`tools/emission/descriptor.py`, `tools/julia_ladder/build_j0.py`,
`tools/studies/q4_neighborhood_sweep.py` and `tools/explorer/app.py` each carry a private
copy of the old constants. They are **not** edited by this raise:

* `build_j0.py` and `q4_neighborhood_sweep.py` are **frozen builders** for artifacts that
  already exist (`data/library`, the q4 sweep readout). Raising their cap would make them
  stop reproducing the artifact they built, which is the opposite of what a builder is for.
* `descriptor.py`'s cap feeds the morph-CLIP descriptor, whose embedding space is shared
  with already-embedded library rows; moving it silently re-keys a space that other
  artifacts are joined against.
* `app.py`'s copy is dead relative to `render_core.py` (the extraction that made the
  coordinate math exist once left the constants behind).

Each is a **known** divergence, listed here so it is a decision rather than a miss. If any
of them is revived for new production output, it must be pointed at
`active_ckpt.auto_maxiter` first.

## ⚠ What the raise actually changes, in pixels

The ×8 measurement above was taken on the **smooth field dump**
(`render-one --dump-field` → `tools/orbital/field_metrics.radial_rings`). The corpus render
path is a **different surface**: `v4-render-batch` colours through
`generate::color_params()` — smooth channel at density 0.004, interior black, sqrt trap
curve — into a 512×288 q85 JPEG. A cap that moves a field statistic does not automatically
move the JPEG a classifier reads, and here it partly does not.

Measured by `tools/v9/measure_cap_effect.py` (60 deploy-canonical tiles — twilight_shifted,
identity geometry, ss2 — stratified over fw deciles, so the pooled number is the corpus
number; a q85 channel delta ≤ 2 is counted as encoder noise):

| comparison | what it answers | % pixels changed | tiles that moved at all |
|---|---|---|---|
| `old_auto` → `new_auto` | the **deploy / label-crop** path | **10.2 %** | 30/60 |
| `flat8000` → `new_auto` | what a cache **re-render** buys training | **3.3 %** | 19/60 |
| `old_auto` → `flat8000` | the **pre-existing** train-vs-label gap | **10.0 %** | 30/60 |

Read the third row first. **The training tiles were already ahead of the crops the human
judged.** Flat-8000 already sat above `old_auto` almost everywhere, so ~98 % of the
deploy-path gain (10.0 of 10.2 points) was *already present in the aug cache* before this
raise. The raise's effect on the corpus is therefore lopsided:

* **The deploy path gains a lot** — 10.2 % of pixels on a canonical crop, and the effect is
  strongest in the SHALLOW deciles (7–9: 14–21 % of pixels), where `old_auto`'s base of 500
  bit hardest. This is the real win, and it is the fix for every future label crop and
  every production render.
* **The training inputs gain much less** — 3.3 % pooled, and 41 of 60 tiles do not change
  by a single pixel. A retrain on the re-rendered cache is therefore expected to land
  *inside* a non-inferiority band rather than above it. That is a prediction, not an
  excuse: it was measured before the retrain, not after it.

### [CORRECTION] The consequence the table was not read for: v8's skew *shrank*

Read the three rows as a pair of before/after states and a result falls out that the
original framing missed entirely. v8's training tiles are flat-8000 and did not move; what
moved is the **deploy** cap, `old_auto → new_auto`:

| | training input | deploy render | **train/deploy skew** |
|---|---|---|---|
| before the raise | flat 8000 | `old_auto` | **10.0 %** of pixels |
| after the raise | flat 8000 | `new_auto` | **3.3 %** of pixels |

⇒ **The raise cut v8's own train/deploy skew from 10.0 % to 3.3 %, as a side effect.**
Nothing about v8 was retrained, re-rendered or re-tuned; raising the deploy cap moved the
deploy path *toward* the flat-8000 tiles v8 was trained on. **v8 is better aligned now
than it was before the change** — the opposite of the usual expectation that changing a
production constant degrades the deployed model until it is retrained.

This is the single most decision-relevant number in the document, because it removes the
urgency from the v9 flip (see below). It is also a reminder that "the training gain is
3.3 %" and "the deploy gain is 10.2 %" are not competing measurements of one thing: they
are two different comparisons, and the *difference* between them is the skew.

Two further facts worth keeping:

* **Convergence at 8000 is common.** On a deep mandelbrot tile (fw ≈ 2.0e-7) the render is
  bit-identical at 8000, 32,606 and 67,000, while dropping to 2,000 moves 2.45 % of pixels.
  The frame is cap-sensitive; it is simply already saturated at 8000. Any claim that "the
  corpus is clipped" must name which cap it means.
* **The effect is not monotone in depth on this surface.** Decile 8 (fw ≥ 0.31) shows 0.0 %
  for `flat8000 → new_auto` and 16.8 % for `old_auto → new_auto`. What matters is how many
  pixels have an escape time *between* the two caps, and that is a property of the
  structure in frame, not of the magnification.

## [CORRECTION] Why v9 is shelved

v9 — the v8 corpus re-rendered at the raised cap — is **trained, evaluated, and staged.
It is deliberately not deployed.** As written (2026-07-31) `ACTIVE_CKPT` was
`data/classifier/v8/model_best.pt`, `keeper_cuts_v9.json` was unadopted, and the re-derived
thresholds sat staged beside the live v8 gate rather than replacing it. This is a decision,
not an unfinished step, and it needs recording because everything about the artifact says
"ready".

**[2026-08-02] The shelving became permanent, and the staged cut is gone.** `ACTIVE_CKPT`
flipped to **v10** — v8's recipe on the same re-rendered corpus plus 1,267 appended
maneuver-view locations — so v9 was skipped over rather than merely deferred, and the
rollback ladder in `data/v10/build_metadata.json` names v10 → v8 → v7 → v6 → v5 with v9
explicitly **not a rung**. `data/atlas/keeper_cuts_v9.json` was deleted at that point: it
was a threshold on a head no gate will ever run, and a staged cut that outlives its
version's last chance of deployment is only a thing to mistake for live. The reasoning
below is why it was shelved and stays exactly as measured; `tools/v9/keeper_cut_v9.py`
still exists, so the file is rebuildable if a question ever needs it back.

**Reason 1 — the pre-registered instrument could not see the change.** The eval bar was
committed at `2171bbc`, before the numbers existed, exactly as it should have been. v9
passed its PRIMARY arm: census-144 q3 AUC 0.7509 → 0.7390, delta −0.0119, DeLong p = 0.706,
**NON-INFERIOR**. But the diagnostic arm returned v8-on-new = v8-on-old = **exactly
0.0000**, and an exact zero is not a null result — it is a measurement of nothing. **All
144 census canonical tiles are byte-identical between the v8 and v9 caches.** The census is
entirely `julia:multibrot3/4/5`, whose fields are already converged at 8000 (higher degree
escapes faster), so raising the cap to 3,400–24,811 changes those tiles not at all. The
PRIMARY verdict is therefore *true and empty*: −0.0119 is retrain-to-retrain variance on
identical inputs, and "NON-INFERIOR" on it says **nothing whatsoever about the cap**. (The
same applies to the class-4 read, 0.813 → 0.688: all 22 class-4 locations are census.)

The only arm whose inputs actually moved is the SECONDARY mandelbrot floor — 357 of 526
canonical tiles changed — which went **+0.0168, the right direction, not significant**.
That is the honest read on the cap, and the pre-registered hierarchy ranks it second.

**Reason 2 — there is no urgency, and the skew table above is why.** The raise *shrank*
v8's train/deploy skew from 10.0 % to 3.3 %. The usual argument for flipping quickly —
"production moved, the deployed model is now stale against it" — runs backwards here: v8
got **better** aligned when the cap went up. Shipping a model whose only measured advantage
is +0.0168 on a secondary arm, to replace one that just improved for free, in exchange for
re-deriving every downstream threshold on a moved score distribution, is a bad trade at
this moment.

**When the flip happens.** At the **next retrain — after the supply run, on new labels** —
where the corpus has genuinely changed and a fresh eval slice can carry census tiles whose
inputs actually move under the cap. v9's weights and staged thresholds stay exactly where
they are until then. The lesson generalizes and is recorded as an operating rule: *before
pre-registering a bar, verify the instrument's inputs actually change.*

## Consequences of a cap change (the checklist)

A cap change is not a local edit. It moves:

1. **The field cache.** Field-dump stems hash the location's `maxiter`, and now also an
   explicit **maxiter-policy token** (`location.maxiter_policy_token`, mirroring
   `field_mode_token` / `field_source_token`) so the axis is named rather than incidental.
   Without it, fields cached under the old cap are served silently under the new one — the
   `--dump-field-source` failure again.
2. **Every render oracle.** Any test asserting a byte-identity, a SHA, or a literal
   maxiter is now asserting the old policy. Re-bracket (prove old-wrong / new-right /
   check-non-vacuous); never blind-rebaseline.
3. **The whole training corpus.** A mixed-cap corpus is poison in the same way a
   mixed-decode readout is — the model learns the cap, not the fractal. A cap change
   implies a **full** cache re-render and a **new classifier version id**, never a
   same-version retrain.
4. **Every derived threshold.** `t_good`, keeper cuts, τ_h all sit on the moved score
   distribution and must be re-derived together, not piecemeal.
