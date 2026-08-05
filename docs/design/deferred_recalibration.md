# Deferred recalibration

Recalibration work that is **designed but deliberately not built**, plus the record of the one
item that has since been built. Each parked item says what it is, why it's held, what unparks
it, and where to start.

| item | state |
|---|---|
| location-head retrain | **DONE — no longer parked.** v8 shipped, v9 was staged and skipped, v10 is live (record below). |
| ranker growth (`pref_loc_v1`) | parked — and the **rebuild is BLOCKED**: no head exists on this checkout and the labels' location join was wiped (record below) |
| location blind reads | parked |
| mining-head calibration | parked |

## The gate: what unparks the remaining three

All three are gated on the **release review**, which has exactly two outcomes:

- **Distribution off** → dictate new ratios in `tools/scoring/release_mix.RATIO`, the one
  place the intended mix is expressed; the emission measure is derived from it at intake and
  is not hand-edited (`retired.md`, 2026-08-04). This is the trigger — the items below come
  online as needed.
- **Distribution fine** → the cluster stays parked.

**Interpretation guard:** a skewed release is a reweight signal only if the skew is in the
distribution the *measure produced* — not if it's a selection- or gate-stage artifact. Never
reweight off a broken gauge; fix the gauge, then re-judge. (Precedent: an early run's
phoenix-heavy / all-smooth shares were selection artifacts, not measure output.)

As of the last release review: judged fine → the three remaining items stay parked.

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

## Ranker growth (`pref_loc_v1`)

- **Why parked:** the growth loop is paused alongside the blind-read deferral — it feeds on
  reads that aren't currently happening.
- **Trigger:** the next location blind read.
- **Loop:** rank → blind-read the top-K → refit on the new labels.
- **Constraints (hold across any growth):**
  - No corpus prior — labels are distribution-bound and pooling eras degrades the fit.
  - The morph-CLIP feature adds nothing; don't re-add it.
  - Scope is fixed: the ranker orders admitted locations only (keeper ordering, emission-intake
    ordering, dive-result sorting). It is never wired into frontier priority, dive-start
    selection, scheduling, or any discovery decision. Role detail in `aesthetic_scoring.md`.
  - Its feature extractor is pinned to a **frozen v7** (`ranker/scorer.PENULTIMATE_CKPT`), not
    to `ACTIVE_CKPT`. Head flips do not move it, and the v10 flip did not.

### Ranker rebuild — BLOCKED (2026-08-05)

Matt ordered a rebuild + recertification of `pref_loc_v1` on 2026-08-05, after the emission
smoke found that `data/ranker/pref_loc_v1/model.npz` **has never existed on this checkout** and
every emission run has therefore been unranked. The rebuild did not run. It is blocked on inputs,
not on a failed bar, and **no weight was fit and none was committed**.

**The pre-registered bar, recorded here before any training** so a later fit cannot be tuned
toward it. The rebuilt head certifies **iff**, under the SAME 3-batch LOBO protocol on the SAME
batches (`tools/ranker/train_eval_v1.py`: run2 + dive + campaign1, folds = each read held out
once, winner = max mean-LOBO Spearman, tie-break mean AUC):

1. pooled percentile-normalized Spearman is **positive**, and
2. its 95% bootstrap CI **excludes the canonical `p_good` baseline** measured on the same folds
   — not merely excludes 0, which is what `train_eval_v1.main` actually tests today, and
3. the per-family LOBO Spearman is **positive in every family**.

Criteria 2 and 3 are STRICTER than the code: `train_eval_v1` certifies on (beats canon_pgood on
mean-LOBO Spearman) ∧ (CI excludes 0), and pref_loc_v1's own per-family table had a family
reading negative on n=7 at the v0 stage. Anything rebuilt has to clear the stricter form.

**Why it is blocked.** The 379 labels survive; the join that says WHICH LOCATION each label
belongs to does not.

| input | state |
|---|---|
| `labels/{steered_run2,steered_v1_2_dive,campaign1}_blind*_scores.json` | present — 60 + 21 + 298 = **379**, keyed by blind tile (`blind_037.jpg`) |
| `scratch/{steered_run2_manifest,dive_manifest,campaign1_blind}/manifest_key.json` | **GONE** — tile → location id. Lived in `scratch/`, never tracked, wiped |
| `data/ranker/**` (v0/v1 `features.npz`, `model.npz`, `metrics.json`, `campaign1/features.npz`) | **GONE** — gitignored, zero commits in history |
| `data/discovery/steered_run2/morph_admissions.npz` | **GONE** |
| ledgers, `outcome_feats.npz`, `data/classifier/v7/model_best.pt` (the pinned extractor) | present |

Neither wiped manifest is re-derivable. `steered_run2_manifest.py` needs the missing
`morph_admissions.npz` (its stratified pick is round-robin over morph clusters), and
`campaign1_manifest.py` scores every admission with `LocationRanker()` before sampling — i.e.
reproducing campaign1's 298-tile key requires the very head being rebuilt. Searched and empty:
git history for `data/ranker*` and `*manifest_key.json`, `docs/`, `data/atlas/`, the artifacts
root (`C:\Code\fractal-maker-artifacts`), the holding and trash trees.

**The certification record is gone too.** No file in this checkout carries pref_loc_v1's numbers.
`metrics.json` was the record and was never tracked; what survives is the `scorer.py` header
("3-batch LOBO meanSp +0.436, certified") and the criteria in `train_eval_v1.py`. Treat any
quoted pooled Spearman / CI for pref_loc_v1 as unverifiable against this tree.

**What unparks it.** A fresh blind read that writes its manifest key to `data/` — not `scratch/` —
is the only route, and it re-bases the protocol rather than continuing it: the batches would be
new, so the bar above is re-registered against them, not inherited. The cheap half is to make the
next read durable by construction; the expensive half is that 379 labels have to be re-collected
to get back to where the LOBO protocol stood.

**Scope, unchanged and restated:** the ranker orders within admitted/eligible only — keeper
ranking, emission-intake ordering, dive-result sorting. Never frontier priority, dive-start,
scheduling, or any discovery decision (`scorer.py` HARD SCOPE). The emission colorize order was
changed on 2026-08-05 to a seeded partition round-robin with the ranker ordering only WITHIN a
partition, so the ranker branch going live later cannot move budget between partitions — it can
only reorder inside one.

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
