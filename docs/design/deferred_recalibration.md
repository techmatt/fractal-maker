# Deferred recalibration

Recalibration work that is **designed but deliberately not built**, plus the record of the one
item that has since been built. Each parked item says what it is, why it's held, what unparks
it, and where to start.

| item | state |
|---|---|
| location-head retrain | **DONE — no longer parked.** v8 shipped, v9 was staged and skipped, v10 shipped, **v11 is live** (records below). |
| ranker growth (`pref_loc_v1`) | **CLOSED — DELETED 2026-08-08.** Not parked, not blocked: the rebuild path is gone and the code with it (closure below). |
| location blind reads | parked |
| mining-head calibration | parked |

## The gate: what unparks the remaining two

Both are gated on the **release review**, which has exactly two outcomes:

- **Distribution off** → dictate new ratios in `tools/scoring/release_mix.RATIO`, the one
  place the intended mix is expressed; the emission measure is derived from it at intake and
  is not hand-edited (`retired.md`, 2026-08-04). This is the trigger — the items below come
  online as needed.
- **Distribution fine** → the cluster stays parked.

**Interpretation guard:** a skewed release is a reweight signal only if the skew is in the
distribution the *measure produced* — not if it's a selection- or gate-stage artifact. Never
reweight off a broken gauge; fix the gauge, then re-judge. (Precedent: an early run's
phoenix-heavy / all-smooth shares were selection artifacts, not measure output.)

As of the last release review: judged fine → the two remaining items stay parked.

## Record: the location-head recalibration, flipped 2026-08-02

**This doc used to say the head retrain was parked pending a release review. It wasn't — it had
already run three times.** What follows is the record and the artifact pointers, so the entry
stops being a stale plan. The reusable *method* is
[`classifier_retrain_protocol.md`](classifier_retrain_protocol.md) §4, not this file.

**v10 is the live head** (`tools/scoring/production_pins.ACTIVE_CKPT`). It certified
**non-inferior on both pre-registered gating arms** and gained nothing measurable on any arm —
adoption is on the certified bar, not on a win. v9 was built, evaluated, staged and never
deployed, and is explicitly **not** a rollback rung.

The flip re-derived the whole scale-bound threshold cluster **together**, because a cut on one
head's `p_good` is a number about nothing on another's:

| artifact | what it is | where |
|---|---|---|
| discovery `t_good` | per-partition q3 operating point, per-family objective | `data/v10/t_good_derivation.json` ← `tools/v10/derive_t_good_v10.py`; adopted copy in `production_seeder.T_GOOD_OVERRIDES` |
| keeper cut | report-only F0.5 twin of `t_good`; nothing gates on it | `data/atlas/keeper_cuts.json` ← `tools/atlas/keeper_cut.py` |
| τ_h | per-partition cheap-render harvest cut | `data/atlas/tau_h_base_v10.json` ← `tools/atlas/tau_h_rederive.py`; vendored into `steered_frontier.TAU_H_FIDELITY_BASE` |
| the coupling itself | what must revert together, and the ladder | the block beside `ACTIVE_CKPT` in `production_pins.py`; `data/v10/build_metadata.json:rollback_ladder` |

Four things from that pass that outlive it:

- **One instrument per partition, never pooled.** v10's eval slice is the first with a third
  unbiased instrument (`maneuver_uniform_v1`, 90 rows). Folding a second population into a
  partition's precision denominator is a different cut, not a bigger one: on mandelbrot it
  moved the argmax five grid steps and collapsed the LOO-OOF F0.5 from 0.357 to 0.100.
- **The objective is re-read from current supply every version, never inherited.** At v10 both
  reads came out unchanged (native abundant → F0.5; `julia:multibrot` scarce → F2), but the
  julia leg was re-affirmed *because* two 2026-08 supply efforts were 100% native-plane and
  bought it nothing.
- **Mandelbrot's `t_good` is now undecidable at the top of the range.** Under v10 the F0.5
  curve is flat and low and the argmax falls to the grid floor, so F0.5 and F2 pick the same
  `t` where under v8 they picked 0.85 vs 0.14. The protocol's answer to an undecidable
  partition is *label more*, not nudge. Watch the first v10-era run's mandelbrot precision.
- **A WATCH, not a gate:** v10's class-4 descriptive AP moved 0.813 → 0.728 (n=22,
  `julia:multibrot`, no pre-registered bar, inside label noise). It rides on
  `production_seeder.cloud_diagnostic` so it surfaces in the first v10-era run's readout, and
  it is keyed on the scorer version so it retires itself at the next flip.

**Open, carried forward:** why 1,310 new labels bought no measurable gain. The appended data is
100% native-plane while the class-4 signal and half the eval power are `julia:multibrot`. A
corpus-mix read belongs before the next labeling round.

**The watch's qualitative look, taken 2026-08-04** on the first v10-era labeled draw
(`2026-08-03_v2_sitting_v1`, 1,000 rows, merged that day; command:
`uv run python tools/atlas/pop_quota.py` for the census half). **Class 4 is 35/1000 = 3.5%
[2.5, 4.8]**, and one-per-cluster at the leader/radius 0.95 cut it is 3.2% over 295 looks —
so the rate is not a near-dup pile-up. **Where it landed is the part worth keeping: 26 of the
35 are parameter-plane** (julia:mandelbrot 16, julia:multibrot5 8, julia:multibrot4 2) **and 5
more are phoenix**, against 4 in native multibrot and **0 in mandelbrot**, which took 0 of its
39 rows past class 2. Class 4 remains a `julia:multibrot`-and-phoenix phenomenon under v10;
nothing here says the head lost the tier. Read it as descriptive only — the draw is
screened-and-ranked and biased more than once (`batch.json § purpose`), so it is a yield, not
a base rate.

That also closes the mix half of the open item above: this sitting is **69.3%
parameter-plane** (487 julia + 206 phoenix), the first appended labels that are not
native-plane, and its currency lands where the deficits are — `julia:mandelbrot` +23.2,
phoenix +15.1, `julia:multibrot5` +14.5 of +67.8 total, with **+0.0 to `mandelbrot`**, the
partition that sets the target anchor (the uniform level then; the max-ratio anchor since
2026-08-04 — mandelbrot sets both). Every other partition's deficit narrowed and the
target did not move.

## Record: the v11 flip, 2026-08-08 — and it moved the POPULATION, not just the head

**v11 is the live head** (`production_pins.ACTIVE_CKPT`). Certified **non-inferior on all
three pre-registered gating arms** (census-144 q3 0.7422→0.7710 p=0.437; floor-526 q3
0.8715→0.8479 p=0.099; uniform-90 q2 0.8289→0.8369 p=0.855, which also SEPARATES at CI_lo
0.749). Bars: `data/v11/prereg_v11.json`; results: `data/v11/eval_results_v11.json`.

The flip re-derived the whole scale-bound cluster together, as v10's did:

| artifact | what it is | where |
|---|---|---|
| discovery `t_good` | per-partition q3 operating point, per-family objective | `data/v11/t_good_derivation.json` ← `tools/v11/derive_t_good_v11.py`; adopted copy in `production_seeder.T_GOOD_OVERRIDES` |
| keeper cut | report-only F0.5 twin of `t_good`; nothing gates on it | `data/atlas/keeper_cuts.json` ← `tools/atlas/keeper_cut.py` |
| τ_h | per-partition cheap-render harvest cut | `data/atlas/tau_h_base_v11.json` ← `tools/atlas/tau_h_rederive.py`; vendored into `steered_frontier.TAU_H_FIDELITY_BASE` |
| the coupling itself | what must revert together, and the ladder | the block beside `ACTIVE_CKPT` in `production_pins.py`; `data/v11/adoption_record.json:rollback_ladder` |

What outlives this pass:

- **The population rule changed, and every threshold move must be read as both.** v10 cut
  each partition on one frozen INSTRUMENT; v11 cuts on the randomized location-GROUPED
  HOLDOUT, with the instrument as a per-partition fallback and still never their union. That
  is the whole point of the flip — six of ten partitions had no eval row of any kind under
  v10 — but it means `mandelbrot 0.03 → 0.90` says nothing about v11's scale: the keeper base
  rate went 4.9% → 11.3% and an F_beta argmax moves with prevalence. Exactly one partition
  kept its v10 population (`julia:multibrot3`, on the census), and that one number IS a clean
  head read: 0.27 → 0.26.
- **Mandelbrot's `t_good` became decidable again.** Under v10 the F0.5 curve was flat and low
  and the argmax fell to the grid floor. Under v11's holdout it has an interior argmax at
  precision 0.781 / recall 0.633 (OOF F0.5 0.731). The v10 entry's "label more, not nudge"
  prescription was right and it is what happened.
- **The holdout is biased exactly as training is**, so those precisions are not what the gate
  will deliver on a discovery frontier. Accepted cost of calibrating six partitions at all;
  the first v11-era run's per-partition admitted precision is the read that checks it, and
  that is what the flip's WATCH (`production_seeder.Q4_WATCH`) is attached to.
- **The v10 WATCH resolved and retired itself.** Class-4 descriptive AP on the census went
  0.7273 → 0.7664 (by P≥4, 0.6315 → 0.6759); the 0.813 → 0.728 fall v10 flagged did not
  persist.
- **Three partitions are derivable and deliberately NOT adopted.** The native multibrots clear
  `MIN_POS` for the first time (49/32/38 holdout positives) and would take 0.61/0.85/0.61.
  Adoption is fork-scheduled (see § "Related" below), so they went through the estimator's new
  `withhold` path: derived, recorded in the artifact's `withheld` block, running at the
  baseline. **This is the fork decision's input and it no longer needs re-deriving.**
- **The library seed is unreconstructable until the next real run.** It is built from
  run-local counts and embeddings under the head that produced them, so any flip retires it.
  Known cost of a flip, not a defect of this one.
- **`phoenix:classic` is now the ONLY partition the rule cannot reach** — 8 holdout rows, 1
  positive, no instrument rows at all. It was one of six; it is one of one.

**Open, carried forward:** the objective for `julia:multibrot{3,4,5}` stayed F2 on CHANGED
evidence. v10's read rested on a hard zero (both 2026-08 supply efforts were 100%
native-plane); 198/266/281 rows have since been drawn, so the zero is gone. They remain the
smallest parameter-plane families and every row came through a gate rather than an exhaustion
test, so the verdict held — but the next version should re-read this with a supply measurement
that is not label-batch counts.

## Ranker growth (`pref_loc_v1`) — CLOSED, DELETED 2026-08-08

**Matt's decision, taken at the v11 flip: the rebuild path is deleted permanently.** This
section used to carry a parked growth loop and, under it, a BLOCKED rebuild with a
pre-registered certification bar. Both are gone. What is kept below is the *history* — why
the object could not be rebuilt — because that is the part a future reader needs and the part
that cost something to establish. The plan is not kept, because a plan nobody may execute is
indistinguishable from an intention.

**What was deleted with it**, in the same commit:

| removed | was |
|---|---|
| `tools/ranker/` (`scorer.py`, `score_locations.py`, `build_features.py`, `train_eval.py`, `train_eval_v1.py`, `report.py`, `test_ranker.py`) | the whole fit-and-serve path, including `PENULTIMATE_CKPT` — the pin that made `data/classifier/v7/model_best.pt` load-bearing |
| `tools/atlas/campaign1_manifest.py` | un-runnable by its own construction: it scored every admission with `LocationRanker()` before sampling, so reproducing its 298-tile key required the head being rebuilt |
| `build_emission_diversity_v1._score_intake_with_ranker` + `--intake-floor` | the live consumer, which caught its own exception on every run this checkout ever made |
| `phoenix_label_diversity` §4b, `steered_run2_report`'s keeper ordering | report/analysis consumers of the same absent head |

**The unranked path is now the plain behaviour, not a fallback.** Emission's colorize queue is
a seeded round-robin across partitions and a **seeded shuffle within one** — unbiased rather
than alphabetical — and that is the whole rule. Selection is unchanged by this closure: it is
what every run has actually done. **Any future within-partition ordering is a new decision**,
re-registered against its own batches; it is not the restoration of this one.

### The history: why it could not be rebuilt

`data/ranker/pref_loc_v1/model.npz` **never existed on this checkout**
(`git log --all --full-history -- data/ranker*` is empty), so every emission run has been
unranked. The rebuild ordered on 2026-08-05 never ran, and was blocked on inputs rather than
on a failed bar: **the 379 labels survive; the join that says which LOCATION each label
belongs to does not.**

| input | state |
|---|---|
| `labels/{steered_run2,steered_v1_2_dive,campaign1}_blind*_scores.json` | present — 60 + 21 + 298 = **379**, keyed by blind tile (`blind_037.jpg`) |
| `scratch/{steered_run2_manifest,dive_manifest,campaign1_blind}/manifest_key.json` | **GONE** — tile → location id. Lived in `scratch/`, never tracked, wiped |
| `data/ranker/**` (v0/v1 `features.npz`, `model.npz`, `metrics.json`) | **GONE** — gitignored, zero commits in history |
| `data/discovery/steered_run2/morph_admissions.npz` | **GONE** |

Neither wiped manifest was re-derivable: `steered_run2_manifest.py` needed the missing
`morph_admissions.npz`, and `campaign1_manifest.py` scored every admission with the very head
being rebuilt. Searched and empty: git history for `data/ranker*` and `*manifest_key.json`,
`docs/`, `data/atlas/`, the artifacts root, the holding and trash trees. **The certification
record is gone too** — `metrics.json` was never tracked, so any quoted pooled Spearman / CI
for pref_loc_v1 (the surviving header claimed "3-batch LOBO meanSp +0.436, certified") is
unverifiable against this tree and always will be.

This is the canonical instance of the `scratch/`-liveness rule in
[`storage_classes.md`](storage_classes.md): a join file in `scratch/` was the only thing
standing between 379 human labels and a usable dataset, and its loss was silent for months.

### What this closure does NOT license

The **HARD SCOPE** that governed the ranker still governs anything that replaces it: a model
that both selects and ranks degrades on its own selections. Any future ordering model ranks an
already-produced set — keeper ranking, emission-intake ordering, dive-result sorting — and is
**never** wired into frontier priority, dive-start selection, scheduling, or any discovery
decision (`aesthetic_scoring.md` §2).

## Location blind reads

- **Why parked:** the live head is taken at its word on certified families; human judgment has
  been moved to the release rather than spent per-location. Fully reversible — coordinates and
  artifacts persist — and the head's authority is structurally bounded (the scheduler
  allocates, the ranker ranks, the head only floors).
- **Trigger:** an explicit order, or a release review that demands one.
- **Standing exception — uncertified populations always get their own read:** any population
  where the live head is not certified is read independently rather than deferred. (Phoenix
  took its own n=500.) This covers any new family or axis, including **deeper-than-f64 zoom**
  and any **q4-mined population** — there the head's *floor* still applies, but quality is
  judged by the q4 objective and the eye, not by the head.
- **Always kept, deferral or not:** visual eyeball passes (admission/reject, medoid, release and
  strange-candidate sheets). These are glances, not labels.

## Mining-head calibration

- **Why parked:** enough of the strange renders are good that the eye currently beats the head;
  calibration can wait.
- **State:** mining v1 is uncalibrated on the strange population.
- **Trigger:** a release that demands it.
- **Entry point:** calibrate from **labeled precision**, never from the head's raw probability.
  `strange_candidates_sheet.png` (82 candidates ranked ≥ 0.50) is the seed view for labeling.

## Related, but not part of this cluster

**Native-multibrot `t_good` adoption** is fork-scheduled, not release-gated: it's decided at the
next fork launch together with τ_h, and a flip retro-re-decodes the library through the
decode-version predicate. Tracked with the discovery/classifier work, not here. As of v10 the
three native multibrot partitions do at last have unbiased eval rows (24/25/29 from
`maneuver_uniform_v1`) and **zero keeper positives among the 78**, so they stay UNCALIBRATED at
the 0.50 baseline — now because they were looked at rather than because they weren't.
