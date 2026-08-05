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

**τ_h re-derivation is two costs, and only one of them is the 26-minute one.**
`tools/atlas/tau_h_rederive.py` renders and scores a paired sample per partition — that is
the expensive arm. Once `rows.jsonl` exists under the work dir, **`--score-only` re-derives
the base from those cached rows in seconds**, which is what makes fixing an estimator choice
(a pooling rule, a `--combine` arm, a `--keep`) cheap. Reach for it before re-rendering:

```bash
uv run python tools/atlas/tau_h_rederive.py --per-partition 200        # the render arm, once
uv run python tools/atlas/tau_h_rederive.py --score-only               # every re-derivation after
```

Read the resulting artifact's `harvest_detail` / `walk_detail` per partition, not the summary
line: the summary prints the harvest arm's `source`, so a partition whose WALK arm fell back
to the pooled estimate still reads `[own]`.

**Every discovery ledger goes stale at the flip, and nothing tells you.** `is_current_decoded`
compares a row's `scorer_version` to `ACTIVE_VERSION`, so the instant the pin moves every
ledger row written under the old head becomes unreachable — correctly, because its decode is
that head's verdict. But the rejection is silent and happens per-row inside `load_admitted`,
so the visible symptom is a downstream population that quietly shrinks. The v10 flip took the
emission stage-2 intake from ~1.4k admissible locations to **16** (2026-08-04), and it read as
"the intake found almost nothing", not as "the ledgers are stale".

The recovery is a **re-score**, and it is a sibling record, never an in-place edit — a ledger
is its run's record of what it found *and* what the head of the day said about it:

```bash
uv run python tools/emission/ledger_rescore.py status    # per-ledger rows / current / admitted
uv run python tools/emission/ledger_rescore.py           # writes <stem>.rescored_<version>.jsonl
```

`descriptor.resolve_rows` overlays the sibling at read time. The **version is in the filename**
and that is load-bearing: after the *next* flip the reader looks for `rescored_v11.jsonl`, does
not find it, falls through to the original rows, and the current-decode predicate rejects them.
The intake goes empty and loud rather than serving v10 verdicts under v11's name. Add a ledger
re-score to the flip checklist alongside the coupled-artifact walk above — the cost is ~1 row/s
(render at the deploy presentation + guarded score), so the 1,633-row intake population is ~25
minutes, not a reason to defer it.
