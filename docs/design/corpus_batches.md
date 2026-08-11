# Corpus batch ROLES — what each batch may be used for, beyond its split

**This doc holds role annotations only. `tools/scoring/batch_registry.py` is THE authority for
split**, and for the 25 batches it registers its per-entry `why` string already carries the
selection story in more detail than any prose here could — read it there. Nothing below restates
a split assignment, and a role recorded here never overrides the registry.

**Why a doc and not comments beside the registry entries** (the choice item 3 of
`handoff_extraction` left open): the registry is fail-closed and deliberately narrow — an entry
exists to answer *"was a model in the selection, and may this be an instrument"*. Nearly half the
corpus's batches are not in it at all (**17 of the 37**, mostly pre-2026-08 material that predates
the registry and classifies train/biased by the fail-closed default), and hanging role
prose on them would mean adding 16 behaviourally-inert entries to a table whose whole value is
that every row is a decision someone made. Roles like *"never a base rate"*, *"labels are
ceilings"*, *"the join is wiped"* are also not split facts, and mixing them into `why` blurs the
one question that table answers.

**Counts below are `(crops, locations)` read off `images.jsonl`**, not label counts — labels live
in three registered places and only `label_store.resolve_score` may be asked for one
(`data/label_corpus/CORPUS_SCHEMA.md`).
`[measured 2026-08-11 over 41 batch dirs = 37 batches + 4 presentation sheets]`

## The standing rules the roles derive from

- **★ The disqualifying property for an EVAL set is MODEL-DRIVEN SELECTION, not
  non-randomness.** A systematic grid, a ladder or a base-rate draw is eval-eligible; rows kept
  because a model scored them well are not.
- **★ Anchored sheets are TRAIN-side instruments; BLIND sheets are the eval instruments.** A
  correction sheet serves the head's own suggestion prefilled and sorts by its score, so its
  labels are anchored to that head: a rate on it measures *agreement with the incumbent* and is
  per-head unusable (a v3-anchored sheet is equally unusable for v3-vs-v5). Protocol §2b.
- **★★ A BLIND SHEET INFORMS EXACTLY ONE BOUNDARY, AND ITS DRAW'S QUALITY CONDITIONING CHOOSES
  WHICH** — not its labeling. Decide the boundary before the draw, or the instrument answers a
  question nobody asked.
- **★ Nominal splits are not realized splits** (a nominal 70/30 realized 17.2% eval). Check the
  realized share before quoting a CI.

## Registered batches — the role the registry does not carry

| batch | role beyond its registration |
|---|---|
| `2026-06-23_flat_generate_loose0_v3` (1043, 526) | The **unbiased mandelbrot base rate**: 35/1043 crops ≥3 = **3.4%** `[cmd: count label.score>=3 in images.jsonl]`. Also v8's forced mandelbrot **eval floor** — the one non-regression instrument the julia:multibrot census cannot give the 59%-of-corpus mandelbrot slice. |
| `2026-07-17_prospect_run1_baserate{,_R}_v1` | The **PINNED primary eval instrument** (protocol §3). Never train, under any build. |
| `2026-07-12_blindspot_v6reject_v1` (219) | **Never in a q3-vs-rest eval** — negative by construction (it *is* v6's rejects), so it inflates any separation measured on it. |
| `2026-08-01_supply_crawl_uniform_v1` (90) | The only **score-unconditioned maneuver-view** draw. **0/90 ≥3**, which BOUNDS that supply at ≲3.3% rather than estimating it. |
| `2026-08-01_supply_crawl_strat_{a,b}_v1` (290+290) | Train, and the **negative footing** — they span every score bin deliberately, which is what most train-side batches do not. |
| `2026-08-01_supply_crawl_exemplar_v1` (60) | Train. The **exemplar-similarity** sourcing it tested is RETIRED after two null reads (`retired.md`) — the rows stay, the method does not. |
| `2026-08-03_q4_uniform_eval_v1` (290) | Score-unconditioned instrument, but **SHORT everywhere it is needed**: julia:mandelbrot 8/48 ≥3, phoenix 6/98, native mb3/4/5 **0 each**. More ε-shell volume buys ~0 expected positives on the natives — they need a *different* draw, not more rows. Uniform draws are DEMOTED to contingency (Matt, 2026-08-06). |
| `2026-08-03_q4_harvest_ranked_v1` · `_near_minibrot_v1` (290+290) | Train legs of the 2026-08-03 harvest sitting; the sitting's third leg is the uniform eval batch above. Merged by route — the sitting itself is not a registration. |
| `2026-08-03_v2_sitting_v1` (1000) | Train, **NEVER eval** — screened/composite-biased, opaque-id export merged via `--route`. |
| `2026-08-05_steady_state_ranked_v1` (654) · `_dive_v1` (94) | The first steady-state run's residue. ⚠ **The dive arms are PARTITION-CONFOUNDED** — the top arm is all mb3, the control is j:mb4+mandelbrot — so the top-vs-control contrast is **unreadable at any n**. The arm labels survive in `provenance.mix_source`; do not build a read on them. 258 of the ranked rows are registered-unlabeled and labelable later. |
| `2026-08-07_label_run_correction_v1` (437) · `_steady_state_v2_backfill_v1` (63) | The first bucketed **CORRECTION** sitting. **Labels are CEILINGS** (anchoring) — state it wherever a rate off them is quoted; it blocks nothing. Buckets: mandelbrot 67 (supply-drained) / j:mandelbrot 138 / phoenix 146 / native-mb 99 / machine-4 slice 50. |
| `2026-08-10_{wallpaper,render_mode}_correction_v2` | Stage-2 correction sheets, one per head. Anchored ⇒ train-side per head, exactly as above. |
| `2026-08-11_wallpaper_blind_minibrot_v1` (sheet D, 197) | **PERMANENT EVAL-ONLY, BLIND.** Its live boundary is **≥4** (48.7% tier-4). 197 of a 200 target is **supply-bound**, so it is the whole eligible population and no selection rule remains that could bias it — but it was drawn behind the location head's `GOOD_FLOOR` and the production colorize path, so its **97.0% ≥3 is the base rate of the GATED minibrot intake, not of raw minibrot material**. With 6 negatives, ≥3 is barely measurable and ≥2 is undefined. |
| `2026-08-11_render_mode_blind_v1` (sheet E, 150) | **PERMANENT EVAL-ONLY, BLIND**, 150 rows over 110 locations. Its live boundary is **≥2**; at ≥3 it has almost no positives. Kept out of training by absence from `near_dup_groups.BATCHES`, not by a flag. |

Both blind sheets are **never re-drawn and never re-spent**: the eval-only pin
(`tools/corpus/eval_only.py`) outranks both of protocol §2a's fixes, applies at UNIT granularity,
and the wallpaper trainers assert on the **c-inclusive coordinate, never `image_id`** — so a
future batch re-rendering a sheet-D location under a fresh id cannot spend the instrument.

## Unregistered batches — fail-closed to train/biased, roles recorded here

| batch | role |
|---|---|
| `2026-07-05_gather_v6` (640, 639) · `julia_ladder_j0` (1000) | Train. Rank- and band-biased. (The `jm3`/`jm45` band batches ARE registered; their role is the registry's.) |
| `2026-07-21_phoenix_grid` (500) · `2026-07-22_native_multibrot_band_v1` (300) | **Train-side only; NEVER a base rate.** A parameter-plane grid looks unbiased and is not a population sample of anything emission draws from. ⚠ The phoenix grid is also the batch that forced the **phoenix grouping rule**: grouping by seed-`c` alone made all 500 rows ONE spatial group (5 holdout groups); grouping by the exact non-`c` axes gives 113 holdout groups / 55 positives, which is what made phoenix calibratable at all. |
| `2026-07-26_anchor_class4_v1` (60) | **Cross-family class-4 ANCHOR** — it calibrates a bar, it does NOT estimate a population. 52 of the 60 rows are revisions (verdicts in the amendment overlay, null in-row by design). |
| `2026-07-26_minibrot_roster_v2` (487, 482) | **WINDOW corpus, not the location corpus** — 3-way STRING classes, separated from location rows by registration and never pooled. |
| `2026-07-27_interior_band_v1` (80) | Uniform-sampled high-interior: **train-side hard negatives**. |
| `2026-07-28_revisit_class3_c{1..4}` (~289 each) | The class-3 **revision sitting** — unlabeled in-row **BY DESIGN**, verdicts in the amendment overlay. Do not "fix" the nulls. Its 338 class-4s are a **biased** population (`mining_v3guided` + `julia_ladder_j0` heavy, train-side). |
| `2026-06-24_guided_descend_rev4{,occfix_v2filtered}` · `2026-06-25_{mining_v3guided_v1, scale_2x2_labelset, scale_controlled_2x2}` · `2026-07-28_gcf_arm_v1` | Pre-registry train-side material. No instrument role; nothing here has ever been eval. |
| `*_sheet_v1` dirs (4) | **Presentation sheets, not batches** — no `images.jsonl` on purpose, because every corpus consumer globs `*/images.jsonl` and a sheet carrying one would double-count every label. `build_combined_label_sheet` owns the rule. |

## The ranker-blind batches: labels survive, the JOIN does not

`steered_run2` blind (60) + dive blind (21) + `campaign1_blind` (298) were drawn for ranker
train/eval. **The 379 hand labels survive; their tile→location manifests were `scratch/`-only and
are wiped**, so the rows cannot be re-attributed and are **unusable for ranker work without
re-collection** (campaign1's is circularly irreproducible). They remain usable as keeper
calibration. This is the standing instance of *a label row carries its join* — see
`storage_classes.md`.

## What the corpus cannot tell you

The unbiased pool is **3 batches / 692 locations**: ~76% mandelbrot, ~21% julia:mb, ~3%
native-mb, **0% phoenix and 0% julia:mandelbrot**. Those two have NEVER been sampled unbiasedly
and no draw size fixes that retroactively. `phoenix:classic` is *structurally* outside
`uniform_eval_draws` — a single-point parameter space needs a VIEWPORT instrument, which is not
built and is registered `NOT_DRAWN` behind a fail-closed coverage check.

⚠ **A rate quoted across batches is a rate across three render regimes** (flat-8000 / old-auto /
new-auto from 2026-07-31). Matt's direction: **the labels hold; not to be raised as a concern.**
