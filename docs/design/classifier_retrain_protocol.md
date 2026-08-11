# Classifier retrain & versioning protocol

Distilled from the v7 build line (`v7_retrain_scope`, `v7_t_good`, and the
`mandelbrot_tgood_steered` t_good re-derivation). Reusable **every version** — the
specific v7 numbers live in code (`data/classifier/*`, `production_seeder.T_GOOD_OVERRIDES`);
this is the durable method. Companion: `aesthetic_scoring.md` (how to read `p_good`).

## 1. Full crop-batch rebuild — SUPERSEDES "append, never rebuild" (2026-08-08, at the v11 flip)

**Build by rebuilding the whole manifest from the committed label corpus, under a seeded
randomized location-GROUPED split.** `tools/v11/build_manifest.py` is the reference
implementation. `loc_id` is dense over the build and carries no cross-version meaning.

### What it replaced, and why that rule ended

The rule here through v10 was *append, never rebuild*: retrain by appending post-freeze
labels to a **byte-frozen prior-version manifest prefix**, freezing every prior location's
`split` / `group_id` / row-order, enforced by a frozen-prefix byte gate on both the manifest
and the cache-manifest rows. Its purpose was **eval comparability**: an identical eval set
makes "did it regress?" a *paired* test.

It ended because the frozen prefix and a re-randomized split cannot both hold, and **the
split rule is the thing that had to change**. v10's eval side WAS the four score-unconditioned
instruments — 1,050 locations, none of them `julia:mandelbrot`, `phoenix`, `phoenix:classic`
or the native multibrots. So six of ten partitions had **no eval row of any kind** while the
corpus held 809 julia:mandelbrot and 375 phoenix keeper positives, and `derive_t_good`'s
`MIN_POS` gate reported "no data" about a corpus that had plenty. Under the append rule that
was not fixable: an instrument is registered before it is drawn, and the partitions in
question had never had one. Six partitions permanently uncalibratable is a larger cost than a
paired eval, so the frozen prefix went.

### What carries comparability now that the prefix does not

**The instruments, reproduced location-for-location.** The four score-unconditioned draws are
forced 100% eval in every build and are re-registered by identity, not by row position — v11's
build gate pins the census-144 that way. Version-over-version verdicts read off those and only
those, which is what they were always for; what changed is that they are now the *whole* of
the comparability argument rather than a subset of a frozen prefix.

### The two eval roles — and they are not interchangeable

    eval_role = "instrument"   the score-unconditioned draws, forced 100% eval. UNBIASED.
                               Base rates and version-over-version non-regression read off
                               these and ONLY these.
    eval_role = "holdout"      a stratified random draw over the remaining split groups.
                               BIASED exactly as training is — a held-out sample of the
                               population the model is trained on. THE calibration
                               population (§4), and a base rate must NEVER be read from it.

**Calibration = randomized location-grouped splits** (Matt, 2026-08-06). `derive_t_good.
select_population` is the shared implementation: per partition the grouped holdout where it
clears `MIN_POS`, the frozen instrument as the fallback, **never their union**. See §4.

### What a rebuild must still enforce

Grouping is the part that has to be right, because it is what a frozen prefix used to do for
free. Group by neighborhood AND by the relations that make two locations the same thing —
`shared_minibrot_atom`, `shared_seed_c` (the dynamical plane), `shared_parent_oid` — union-find
them into split groups, and draw the holdout over GROUPS, never over locations. Stratify the
draw on `fractal_type` (so a small partition is not underserved by luck) and on
`group_has_label_ge3` (so the eval positive rate is the population's rather than the draw's).
Every location's split is then a fact about its group.

**A re-split renumbers `loc_id` and invalidates the render cache**, which is keyed on it. That
is the standing cost of this rule and it is why a rebuild is a build, not a tweak.

## 2. Split assignment — the cardinal sin is biased-in-eval

- Force **unbiased / base-rate draws → eval**.
- Force **model-selected or negative-by-construction batches → train**.
- Partition the group union-find by `(fractal_type, split, c-bucket)` so forced splits
  cannot **transitively straddle** train/eval.
  
  Children inherit their seed's split — any parent-derived location (julia twins, dives off a seed) takes the seed's train/eval assignment, so a descendant can't leak the seed's morphology across the split. (The c-bucket union-find mostly enforces this, but stating it explicitly closes the gap where a child's c drifts into a different bucket.)
Manifest-build gates abort-all — if any build gate fails, abort the whole build rather than emit a partial manifest.

**One registry (2026-08-04).** Every batch's classification lives in
`tools/scoring/batch_registry.py` and nowhere else. It is read twice — by
`tools/v7/build_manifest.assign_split` (what a batch builder consults BEFORE it draws)
and by `tools/v8|v10/build_manifest.classify_batch` (what a build realizes) — and those
were two independently-edited tables until they disagreed about `loose0_v3` for a week.
Split is DERIVED from eval-eligibility, never stored. `tools/scoring/test_batch_registry.py`
fails on a second literal copy and on any corpus batch id inside a manifest-build module.

**The score-unconditioned exemption (Matt, 2026-08-04).** The property that qualifies an
eval instrument is **score-unconditioned draw**, not neighborhood isolation. A group
holding a forced-eval location used to have its biased members DROPPED (they can be
neither eval, which is biased-in-eval, nor train, which straddles the group). For an
instrument registered `score_unconditioned=True` the third constraint gives instead: the
group straddles, the instrument's own locations go eval, and a biased group-mate stays
TRAIN. Measured on the 2026-08-04 corpus under the unified registry, the drop rule cost
**687 train locations (193 threes, 50 fours)** and the exemption recovers all of them
(`uv run python tools/v10/build_manifest.py --dry-run`).

> **Caveat, and it is why the flag is per-instrument data rather than a global switch:**
> model-performance numbers read off an exempted leg are **mildly optimistic** wherever a
> train group-mate exists — on that corpus, 50 straddling groups and 18 train locations
> sharing a minibrot atom with an instrument location. Fine for **base rates and `t_good`
> calibration**, which is what these instruments are for. **Not** fine for fine-grained AUC
> comparisons between checkpoints. An instrument drawn for a model-quality read should
> register `score_unconditioned=False` and keep the drop rule; GATE 3 and GATE 14 stay at
> full strength for it, and count the exempted cases for every other.

Labels attach to **locations**, and training **re-renders from coordinates** — so a
batch's stored crops never constrain training. But the **deploy presentation point**
(geometry / palette / AA) must be covered by the aug fan-out, or the residual covariate
shift stated explicitly.

### 2a. A batch's STAMPED split side is a function of the draw, not of the location

> **When you pool batches, either re-derive the split over the union or freeze one batch
> set's assignment as authority. Honouring every batch's own stamp is the one option that
> is always wrong.**

A per-batch splitter is seeded and location-grouped and looks deterministic, but it draws
over **the set it was given**. Two batches that overlap in locations ran two different
draws, so the same location gets two answers, and a pooled run that honours both stamps
puts one copy of a location in train and another in eval. Nothing is red; the eval side is
simply no longer held out.

It has now happened three times, each worse than the last:

| pooled set | overlap | stamped onto opposite sides |
|---|---|---|
| wallpaper fresh pair (v4, 2026-08-06) | 107 locations | 19 |
| wallpaper sheet A vs the prior five (v4b, 2026-08-10) | 73 locations | 37 |
| mining v1 sitting vs sheet B (v3, 2026-08-10) | **91 of 91** | **33** |

The mining case is the instructive one: sheet B draws the *unserved* (location, mode) pairs
of the same gate-passer population, so its 91 locations are a strict subset of the sitting's
112 and there is no non-overlapping part to reason about.

**Which of the two fixes to take is decided by the BASELINE, not by taste.**
- If the comparison is against a head that trained on some of these rows, **freeze** the
  prior assignment and place only the new batch against it. A re-randomised split moves the
  baseline's own training rows into the eval side and inflates it — and that leans the
  verdict toward a false baseline win, which is the direction a retrain cannot detect.
  (`classifier/train_wallpaper_v4b.split_v4b`.)
- If no consistent prior assignment exists — because the stamps contradict each other, or
  because a batch is stamped 100% train — **re-derive globally** over the pooled locations
  and record how many rows moved. (`tools/mining/mining_corpus.load_corpus`: 999 of 2,460.)

**A batch stamped `eval_only` is pinned to eval by BOTH fixes, unconditionally.** The choice
above is between two authorities for a *contested* location; an eval-only slice is not
contested, it is **spent** the moment it trains — sheet D's 197 blind minibrot labels are the
only unanchored read of that population that will ever exist. So the pin is a third
constraint that outranks both fixes and lives in one owner rather than in each split pass.
It keys on the **c-inclusive coordinate**, not the `image_id`: a later batch re-rendering a
sheet-D location under a fresh id would otherwise train on it without ever naming it.
`[code: tools/corpus/eval_only.py, wired into tools/mining/split_units.build_split (force_eval),
mining_corpus.load_corpus, classifier/train_wallpaper_v4.split_union and v4b.split_v4b;
test: tools/corpus/test_eval_only.py]`

**A batch registered "never an eval INSTRUMENT" may still land in a HOLDOUT.** The two eval
roles of §1 are not the same object: an instrument is an unbiased draw a base rate may be
read from, a holdout is biased exactly as training is. Sheet C is stamped 100% train because
its locations are conditioned on a human 4 — that disqualifies it from being an instrument
and says nothing about a holdout. The render-mode corpus has never had an instrument at all,
so reading its registration as "train-only forever" would have made the rare-palette slice
permanently unmeasurable.

`[code: classifier/train_wallpaper_v4.reconcile_stamped_sides,
classifier/train_wallpaper_v4b.split_v4b, tools/mining/mining_corpus.load_corpus;
test: tools/mining/test_mining_corpus.py, tools/wallpaper/test_wallpaper_v4b_split.py]`

### 2b. ANCHORED correction-sheet labels are TRAIN-SIDE ONLY

> **An incumbent-vs-challenger eval slice must come from BLIND or PRE-INCUMBENT labels. A
> slice served with the incumbent's suggestions measures agreement with the incumbent, never
> quality — and no from-scratch challenger can win it.**

This is a different disqualification from §2's biased-in-eval, and a slice can pass that one
and still fail this. §2 is about how the POPULATION was *selected*; §2b is about how the
LABELS were *elicited*. A correction sheet serves each row with the incumbent's decoded tier
prefilled and the page ordered by the incumbent's continuous score, so the human's answer is
a **correction of the incumbent** and the incumbent's own score is a predictor of it by
construction. Confirming a suggestion is one keystroke and overriding it is a decision, which
is exactly the asymmetry a correction sheet is built to exploit for label throughput — and it
is fatal on the eval side.

The measured size of the effect, on `2026-08-10_wallpaper_correction_v2` (sheet A):

| | value |
|---|---|
| labels returned equal to the served suggestion | **815 / 960 = 84.9%** |
| within one tier of it | 99.8% |
| on the minibrot/maneuver bucket alone | 85.3% |
| v3 AUC≥3 **there** | **0.965** |
| v3 AUC≥3 on `humanq3` / `dramatic` (labels predate v3) | 0.746 / 0.750 |

The (28) wallpaper retrain put that bucket on the eval side as its MOTIVATING arm. v4b lost
clause (b) — the arm the retrain existed for — and the loss says nothing about v4b: 0.965 is
not a quality number, and a head that never served those suggestions can only score below it.
`batch_registry` had already written the reason down for this exact batch before it was built
("a tier rate measured on it is a statement about agreement with v3 and never a base rate …
which is exactly what makes it unusable on the eval side"), so the registration was right and
the slice choice ignored it.

**The rule.** An anchored batch may train, and should — correcting the incumbent is the
cheapest way to buy the labels it is most wrong about. It may not be an eval arm against the
head that anchored it, and it may not be one against a challenger to that head either. Where
a comparison needs the population an anchored sheet covers, buy the eval slice separately:
**fresh locations, blind serving (no prefilled tier, no score-ordered page, shuffled), and
neither head touching the draw or the substrate** — `tools/wallpaper/build_blind_minibrot_
sheet.py` (sheet D) is the worked instance, stamped `eval_only` at build time so it cannot
drift onto the train side later.

Two corollaries worth stating because both were nearly missed:
- **Anchoring is per-HEAD, not per-batch.** A sheet anchored to v3 is unusable for v3-vs-v4b
  and equally unusable for v3-vs-v5. The disqualification travels with the head that
  suggested, and every descendant is compared against that head.
- **A pre-incumbent batch is fine.** `humanq3` and `dramatic` were labeled before v3 existed,
  which is why they are the two clean arms in the same report. "Blind" and "predates the
  incumbent" are both acceptable; "corrected under the incumbent" is not.

**A whole corpus can be anchored, and the mining corpus is.** All three labeled render-mode
batches are correction sheets — the 2026-08-06 v1 sitting, sheet B and sheet C each served
mining v1's suggested tier prefilled and ordered the page by its score, with 0.929 of the
mining sitting's labels returned equal to what was served. So there is no clean arm anywhere
in it, and the (28)/(28b) clause-(a) verdict — five staged arms, every one of them losing on
`pooled.auc_ge2` plus a handful of per-mode cells — is measured entirely against a baseline
the labels are coupled to. That is the state §2b describes at corpus scale rather than batch
scale: it is not fixable by choosing a different arm, only by buying an unanchored slice.
`tools/mining/build_blind_mining_sheet.py` (sheet E) is that slice, and the shape generalizes
— the unit of freshness follows the head's question, so where the wallpaper head judges a
LOCATION and sheet D excludes prior locations, the mining head judges a (location, mode) pair
and sheet E excludes prior pairs.

**Per-sitting correction rates, the whole record.** Every one is *agreement with the head that
served the page*, never quality — they live here so a new sitting can be read against them
instead of against an impression. Sheet A (wallpaper, v3-served) 0.849 · the 2026-08-06 mining
sitting (v1-served) 0.929 · **sheet F (mining, v3-served, 2026-08-11) 0.880 — 176/200 exact
tiers, with 5 up and 4 down across the ≥2 boundary and 7 up / 8 down across ≥3.** Sheet F is
the first convergence datum on the FLIPPED mining head, and the flat ≥2 boundary (9 flips
either way on 200 rows) is what the crossover in
`data/render_mode_head/v3/baserate_audit_2026-08-11.json` is read off — so that crossover is
where the human agreed with v3, and its own record says so. Two comparisons that make the
number readable: sheet F's ≥2 rate is 53.5% (107/200) against sheet E's BLIND 50.7% (76/150)
on the same draw rule, and at ≥3 it is 9.5% against E's 4.0%.

`[code: tools/scoring/batch_registry (the wallpaper_correction_sitting and mining_blind_eval
registrations), tools/wallpaper/build_blind_minibrot_sheet.py,
tools/wallpaper/sheet_d_reverdict.py, tools/mining/build_blind_mining_sheet.py,
tools/mining/sheet_e_reverdict.py; test: tools/wallpaper/test_blind_minibrot_sheet.py,
tools/mining/test_blind_mining_sheet.py]`

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

- **The population is the randomized location-GROUPED HOLDOUT** (Matt, 2026-08-06), with the
  partition's frozen instrument as the fallback where the holdout is short — one or the
  other, **never their union**. `derive_t_good.select_population` is the shared
  implementation and `MIN_POS` is applied to whichever it picks. The holdout is biased
  exactly as training is, which is what a calibration cut wants; the instrument stays
  reserved for base rates and version-over-version verdicts (§1). Through v10 the rule was
  "one frozen instrument per partition", which starved six of ten partitions — see §1 for
  why that ended.
- **F_beta-argmax** over a `p_good` grid, tie-break toward higher `t`, only where the
  chosen population has **≥15 positives**. `beta` is chosen per partition — see below.
- **Read a threshold move as population + head, not head alone, whenever the population
  rule changed.** An F_beta argmax moves with prevalence: at the v11 flip mandelbrot went
  0.03 → 0.90 across a keeper base rate of 4.9% → 11.3%, and no part of that is attributable
  to v11's scale. Where a partition's population is unchanged the move IS readable as a head
  change — v11 kept `julia:multibrot3` on the same census v10 cut it on, for exactly that.
- **A partition the estimator CAN cut and the pass does not adopt gets `withhold`, not a
  dropped row.** The number is derived, printed and written to the artifact's `withheld`
  block with a reason naming the decision's owner, and the partition runs at the baseline.
  Dropping the rows would print "no data" about data there is; adopting silently makes a
  threshold move nobody asked for a side effect of a flip.
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

**One instrument per partition, never pooled.** Where a partition has more than one unbiased
eval instrument, cut it on ONE — its own — and say which. Pooling two unbiased populations
into a single precision denominator is a different cut, not a bigger one: at the v10 flip,
adding 12 zero-positive `maneuver_uniform_v1` rows to mandelbrot's 526-row floor moved the
argmax five grid steps and collapsed the LOO-OOF F0.5 from 0.357 to 0.100. Cutting on the
instrument the previous version used is also what keeps a version-over-version threshold move
readable as a head change rather than a population change.

**A zero-positive instrument still changes the record.** A partition with unbiased draws and
no keepers is UNCALIBRATED for a different reason than one nobody has ever sampled, and the
artifact must carry the two reasons separately — "we looked and found none" is evidence,
"we have never looked" is not.

(Current durable outcome: the **v10** table in `production_seeder.T_GOOD_OVERRIDES` —
mandelbrot 0.03 via F0.5, `julia:multibrot{3,4,5}` 0.27/0.03/0.06 via F2, five partitions
UNCALIBRATED. The *values* live in `production_seeder.py`, the *derivation* in
`tools/v10/derive_t_good_v10.py` → `data/v10/t_good_derivation.json` (which imports the
estimator from `tools/scoring/derive_t_good.py` rather than copying it — a per-version
deriver supplies only its slice, its population rule and its re-read objective), and
`tools/scoring/test_t_good_adoption.py` holds the ACTIVE version's artifact and that table in
agreement. The *method* lives here. NOTE for the next version: v10's mandelbrot cut is the
first that fell to the grid floor with F0.5 and F2 agreeing — an undecidable partition, whose
protocol answer is *label more*, not nudge.)

## 5. The flip itself — what moves with the pin

Adopting a head is not one edit. `ACTIVE_CKPT` is a pointer; every threshold calibrated
against the previous head's probability scale is, the instant it moves, a number about
nothing. The set that must move together is enumerated in exactly one executable place —
**`production_pins.COUPLED_ARTIFACTS`** — and walked by
`tools/scoring/test_coupled_artifacts.py`, which reads each entry's version stamp and holds
it to `ACTIVE_VERSION`. The prose block beside `ACTIVE_CKPT` and the
`rollback_ladder.must_revert_together` block in `data/<v>/build_metadata.json` are the same
set said twice more; the test asserts the record is a subset of the registry, so the three
cannot drift apart silently.

**Before flipping, list what a flip touches instead of discovering it:**

```bash
uv run pytest -m version_pinned --collect-only -q     # the flip-coupled tests, ~90 across 9 files
uv run pytest -m "version_pinned and not slow" -q     # run them (seconds; no GPU)
```

`version_pinned` is a label, not a lane — the marked tests also run in the default suite. It
exists because the v10 flip's answer to "what breaks?" was nine failures across four files,
three of which had hardcoded the outgoing version and went red *for* the flip rather than for
a fault. A test that would need editing when the pin moves is mismarked as a guard; mark it,
or make it resolve the version from the pin.

### 5-0. THERE ARE THREE PINS, AND THEY MOVE INDEPENDENTLY (2026-08-11)

`ACTIVE_CKPT` is the LOCATION head. The two **stage-2** heads —
`wallpaper_pins.HEAD_CKPT_REL` (smooth) and `mining_pins.ACTIVE_MINING_CKPT` (promoted
strange) — are separate pins on separate corpora, and a flip of either is governed by this
section with **§5b and §5c not applicable**: τ_h and the discovery ledgers are cuts on the
LOCATION head's `p_good` and are untouched by a stage-2 flip. Everything else transfers.

```bash
uv run pytest -m stage2_pinned --collect-only -q     # the stage-2-coupled tests, 56
uv run python tools/scoring/volume_match.py wallpaper --incoming <ckpt>
uv run python tools/scoring/rescore_fit_slices.py wallpaper --ckpt <ckpt>
uv run python tools/scoring/adopt_head.py wallpaper --write
```

**`version_pinned` does not cover them, and listing it before a stage-2 flip lists the wrong
set.** The 2026-08-11 flip (`prompts/flip_29.md`) moved both stage-2 pins while `ACTIVE_CKPT`
stayed on v11, and went red in **ten places across six files** — every one a test that had
hardcoded an outgoing value. That is precisely the failure `version_pinned` exists to
prevent, on the pins it does not cover, so `stage2_pinned` is its sibling.

**What moves with a stage-2 pin**, and all of it is scale-bound in the §5a sense:
| what | owner |
|---|---|
| the gate | `wallpaper_pins.GATE_THRESHOLD` / `mining_pins.MINING_GATE_THRESHOLD` |
| the pool floor | `floors.WALLPAPER_POOL` / `floors.MINING_POOL` |
| the suggestion cuts | `suggest_tier.{CUTS,INTAKE_CUTS}` / `suggest_tier_mining.CUTS` |
| the gate lock | `mining_pins.LOCK_PATH` — a NEW record at the new head's path |
| the sheet's score bins | `build_fresh_sheet.SCORE_BINS` — derived from the gate |

**The suggestion cuts are the non-obvious one.** `expected_tier = 1 + Σ marg` is a sum of
CORN marginals, so a cutpoint on it is exactly as train-prior-calibrated as a probability
floor — and unlike a floor, it is not restated volume-matched but **re-fitted** on its own
labeled slice, because prior reproduction (not volume) is what it was chosen for. That needs
the slice's readout under the new head, which the batch rows cannot supply: they carry the
readout of the head that BUILT the sheet, stamped at sheet-build time and *never* rewritten
(it is the record of what the human was anchored on). So the new head's readout goes to a
sidecar — `data/<family>/<version>/fit_slice_pred.json`, written by
`tools/scoring/rescore_fit_slices.py` — and the deriver resolves it through a per-version
`PRED_SOURCES` table that **raises on an unregistered head**. Without that refusal the
deriver silently returns the OLD head's cutpoints under the new head's name.

`[code: tools/scoring/{volume_match,rescore_fit_slices,adopt_head}.py;
test: tools/scoring/test_volume_match.py]`

### 5a. The two floors — RE-SCORE, then VOLUME-MATCH. This is the whole quality half of a flip.

Since 2026-08-09 (`prompts/selection_restructure_3.md`) there is **one quality definition in
the pipeline**, and it is two constants in `tools/emission/floors.py` read against a row's raw
stored P(≥3):

| constant | value | what it decides |
|---|---|---|
| `GOOD_FLOOR` | 0.50 | run-side admission and every run-side count — what a discovery run keeps |
| `JUNK_FLOOR` | 0.20 | the colorize-pool draw — what stage 2 spends compute on. **PERMANENT shared-scale: never restated** (see below) |

Neither is in `COUPLED_ARTIFACTS`, **deliberately**, and this is the part to understand before
a flip. Both are cuts on a train-prior-calibrated CORN scale, so they are exactly as
scale-bound as the artifacts that *are* registered — but a registered artifact is *re-derived*
and carries a stamp saying which head it came from, whereas a floor is *restated*, and the
correct new value depends on a measurement rather than on a version. A stamp check would pass
by moving a string. So they are held by this procedure plus a human:

1. **Re-score the ledgers** (below) so every stored `p_good` is on the new head's scale.
2. **Volume-match `GOOD_FLOOR`** (and each stamped floor): recompute the score that keeps the
   same FRACTION of a fixed reference pool under the new head, and move the constant there.
   Volume-matching is the only restatement that keeps the thing the floor was chosen for
   invariant — how much supply it keeps. Keeping the float silently moves the volume;
   re-deriving from an eval turns a coarse cut back into an operating point, which is precisely
   the per-partition machinery this replaced. **`JUNK_FLOOR` is exempt and is not touched** —
   see the shared-scale paragraph below.

   **The arithmetic has an owner as of 2026-08-11: `tools/scoring/volume_match.py`.** It
   scores a named reference pool under both heads in ONE pass through the harness that
   actually gates with each — never one head's frozen `eval_scores.jsonl` against the
   other's live pass, which is a comparison across two rendering events — and places the new
   cut at the **midpoint** between the k-th and (k+1)-th largest scores. The midpoint, not
   the k-th score, because a report's `cut_at` IS the k-th and admits k under `>=` but k-1
   under `>`, and the two stage-2 sites disagree (`emit_v1` gates `p_ge3 > gate`,
   `MiningScorer.gate` uses `>=`). The realized volume is then RE-COUNTED under the rounded
   constant that gets written, because rounding a midpoint can cross a tie; read
   `volume_preserved` before trusting a restatement.

   **`JUNK_FLOOR` IS EXEMPT — PERMANENT SHARED-SCALE, never restated at a flip** (Matt,
   2026-08-11; this replaces the residual this section carried through the 2026-08-11 stage-2
   flip, which said the question was open). It was read on TWO heads' scales — `ranked_intake`
   on the stage-1 location head's `p_good`, `deploy_tail` on the mining head's `p_ge3` — so a
   single-head flip had no correct volume-match: matching it to the flipped head moves the cut
   at the other site by an amount nobody measured. **`deploy_tail` was repointed at the mining
   gate later the same day** (below), leaving one live reader and making a volume match
   arithmetically available again — and the exemption stands anyway, because the load-bearing
   half was never the reader count. The decision is that 0.20 is a **coarse
   semantic floor valid on any CORN P(≥3) scale** ("the judging head is confident this is
   junk") rather than an operating point on one, and the alternative — one constant per head —
   is refused because it re-creates the per-head operating point this cut was deliberately
   chosen not to be. **The cost is named, not hidden**: the exact volume it removes drifts a
   little at each flip, which is accepted, because the floor's job is removing obvious waste
   and not holding a rate. So step 2 above applies to `GOOD_FLOOR` and to the four stamped
   floors; `JUNK_FLOOR` is left alone by every flip, and leaving it alone is now the checked
   behaviour rather than an omission.
   **THE MIDPOINT CONVENTION IS REUSABLE; THE VOLUME INVARIANT IS NOT.** A cut can also move
   when the head has *not* — a labeled slice says where the score crosses a label boundary, and
   somebody decides to put the cut there. That is a **crossover**, not a restatement, and it is
   the other question this section's question has: volume-matching holds VOLUME fixed and lets
   meaning move, a crossover holds MEANING fixed and lets volume move. The two must not be
   described as one. What carries over unchanged is the placement arithmetic — midpoint between
   the adjacent scores, realized volume re-counted under the rounded constant — because that is
   about `>` vs `>=` and has nothing to do with why the cut moved.
   Worked instance, 2026-08-11 (`prompts/audit_mining_process.md`): sheet F's 200 human tiers
   put the isotonic crossover of `1[label ≥ 2]` against the mining head's `p_ge3` at **0.0949**,
   the gate went there from its own volume-matched 0.6691, and the volume moved **4.6×**
   (129 → 587 of 827). `tools/mining/baserate_audit_reads.py` is the arithmetic;
   `lock_mining_gate.LockSpec` is what stops the resulting lock from claiming a volume match.
   Two things a crossover has to carry that a restatement does not: whether the slice was
   **anchored** (sheet F was v3-prefilled, so every number is a ceiling), and what it does to
   the cuts it did NOT move — the crossover landed below both `JUNK_FLOOR` and the mining pool
   floor, which forced the pool floor to 0.0 to keep `floors.check_below_gate` satisfied and
   left the enforcing junk floor as the strictest cut in stage 2. **That inversion was resolved
   the same day by moving a READER, not a number** (Matt): `deploy_tail`'s colorize-pool draw
   filters through `mining_gate.MiningScorer.gate` instead of `JUNK_FLOOR`, so the mining side's
   pool draw is the gate (455 → 587 of the 827 reference-pool rows) and `JUNK_FLOOR` keeps its
   value and its stage-1 reader. Record:
   `data/render_mode_head/v3/mining_gate_lock_2026-08-11.md`.

3. Nothing else. There is no threshold sweep, no per-partition table to re-adopt and no
   conformance test to re-run — `tools/scoring/derive_t_good.py`,
   `production_seeder.T_GOOD_OVERRIDES`, `tools/atlas/keeper_cut.py` and their five per-version
   drivers were all deleted on 2026-08-09. Their committed json outputs stay as records of what
   those heads served; nothing reads them.

`tools/scoring/test_coupled_artifacts.py::test_the_two_enforcing_floors_are_deliberately_NOT_in_this_set`
holds that reasoning to this section, so the registry and this page cannot drift.

### 5b. τ_h re-derivation — two costs, and only one of them is the long one

`tools/atlas/tau_h_rederive.py` renders and scores a paired (cheap, canonical) sample over the
walk-outcome ledger — that is the expensive arm. Once `rows.jsonl` exists under the work dir,
**`--score-only` re-derives the base from those cached rows in seconds**, which is what makes
fixing an estimator choice (a `--keep`, a `MIN_N`) cheap. Reach for it before re-rendering:

```bash
uv run python tools/atlas/tau_h_rederive.py                # the render arm, once (~1,150 rows)
uv run python tools/atlas/tau_h_rederive.py --score-only   # every re-derivation after
```

The estimator conditions on `GOOD_FLOOR`, not on a per-partition threshold, and a partition
under `MIN_N` good rows gets **τ_h = 0.0** — fail OPEN, which costs visible GPU-minutes rather
than invisible supply. Read the artifact's `detail` per partition (`n_rows` / `n_good` /
`source`) before trusting a value: three partitions clear `MIN_N` by single digits.

### 5c. Re-score the ledgers — an accuracy job now, not an outage repair

Every discovery ledger's `p_good` is the head-of-the-day's number, so after a flip every floor
applied to it means something slightly different. **This used to be far worse than that.**
Until 2026-08-09 a decode-VERSION predicate refused any row an older head had stamped, per-row
and silently, inside `load_admitted`: the v10 flip took the emission stage-2 intake from ~1.4k
admissible locations to **16** (2026-08-04), and it read as "the intake found almost nothing"
rather than as "the ledgers are stale". That predicate is gone. A stale row now sinks in the
ranking instead of vanishing from it, so a deferred re-score costs accuracy, not the corpus.

It is still the first thing to do, and it is a sibling record, never an in-place edit — a
ledger is its run's record of what it found *and* what the head of the day said about it:

```bash
uv run python tools/emission/ledger_rescore.py status    # per-ledger rows / current / admitted
uv run python tools/emission/ledger_rescore.py           # writes <stem>.rescored_<version>.jsonl
```

`descriptor.resolve_rows` overlays the sibling at read time, and the **version is in the
filename** so the next flip's reader looks for `rescored_v12.jsonl`, does not find it, and
falls through to the v11 numbers rather than to another head's verdicts under v12's name. The
cost is ~1 row/s (render at the deploy presentation + guarded score), so the 1,633-row intake
population is ~25 minutes — not a reason to defer it, and step 2 above cannot be done honestly
before it.
