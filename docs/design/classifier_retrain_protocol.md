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

`[code: tools/scoring/batch_registry (the wallpaper_correction_sitting registration),
tools/wallpaper/build_blind_minibrot_sheet.py, tools/wallpaper/sheet_d_reverdict.py;
test: tools/wallpaper/test_blind_minibrot_sheet.py]`

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

### 5a. The two floors — RE-SCORE, then VOLUME-MATCH. This is the whole quality half of a flip.

Since 2026-08-09 (`prompts/selection_restructure_3.md`) there is **one quality definition in
the pipeline**, and it is two constants in `tools/emission/floors.py` read against a row's raw
stored P(≥3):

| constant | value | what it decides |
|---|---|---|
| `GOOD_FLOOR` | 0.50 | run-side admission and every run-side count — what a discovery run keeps |
| `JUNK_FLOOR` | 0.20 | the colorize-pool draw — what stage 2 spends compute on |

Neither is in `COUPLED_ARTIFACTS`, **deliberately**, and this is the part to understand before
a flip. Both are cuts on a train-prior-calibrated CORN scale, so they are exactly as
scale-bound as the artifacts that *are* registered — but a registered artifact is *re-derived*
and carries a stamp saying which head it came from, whereas a floor is *restated*, and the
correct new value depends on a measurement rather than on a version. A stamp check would pass
by moving a string. So they are held by this procedure plus a human:

1. **Re-score the ledgers** (below) so every stored `p_good` is on the new head's scale.
2. **Volume-match each floor**: recompute the score that keeps the same FRACTION of a fixed
   reference pool under the new head, and move the constant there. Volume-matching is the only
   restatement that keeps the thing each floor was chosen for invariant — how much obvious
   waste `JUNK_FLOOR` removes, how much supply `GOOD_FLOOR` keeps. Keeping the float silently
   moves the volume; re-deriving from an eval turns a coarse cut back into an operating point,
   which is precisely the per-partition machinery this replaced.
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
