# Deferred recalibration

Recalibration work that is **designed but deliberately not built.** Four model/threshold updates are spec'd and ready but held back because building them now would be premature — each waits on a specific trigger. This doc records what each is, why it's held, what unparks it, and where to start.

The four: v8 location-head retrain · ranker growth · location blind reads · mining-head calibration.

## The gate: what unparks the cluster

All four are gated on the **release review**, which has exactly two outcomes:

- **Distribution off** → dictate reweights to the target measure (applied by a prompt; the measure is not hand-edited). This is the trigger — the items below come online as needed.
- **Distribution fine** → the cluster stays parked.

**Interpretation guard:** a skewed release is a reweight signal only if the skew is in the distribution the *measure produced* — not if it's a selection- or gate-stage artifact. Never reweight off a broken gauge; fix the gauge, then re-judge. (Precedent: an early run's phoenix-heavy / all-smooth shares were selection artifacts, not measure output.)

As of the last release review: judged fine → all four parked.

## v8 location-head retrain

- **Why parked:** v7 is performing; no release has demanded recalibration.
- **Trigger:** a release review demanding it.
- **Entry point:** the retrain mechanics (manifest append, split rules, augmentation recipe) live in `classifier_retrain_protocol.md`. The deltas specific to v8:
  - Phoenix enters training for the first time — append its rows under the extended identity key `(family, cx, cy, fw, c, p, z₋₁)`.
  - Successor certification must extend beyond v7's scope (within-family, unselected draws) to add **cross-family calibration** and **selected-population** checks.
  - If the head scores seed fertility, certification must add **human-adjudicated** fertility — the current head over-separates it (machine ICC 0.90–0.965 vs human 0.72–0.82).
  - The n=144 census cannot resolve an AUC gap around 0.62-vs-0.57; pre-registered rule: **AUC 0.55–0.65 ⇒ label more** before trusting the result.

## Ranker growth (`pref_loc_v1`)

- **Why parked:** the growth loop is paused alongside the blind-read deferral — it feeds on reads that aren't currently happening.
- **Trigger:** the next location blind read.
- **Loop:** rank → blind-read the top-K → refit on the new labels.
- **Constraints (hold across any growth):**
  - No corpus prior — labels are distribution-bound and pooling eras degrades the fit.
  - The morph-CLIP feature adds nothing; don't re-add it.
  - Scope is fixed: the ranker orders admitted locations only (keeper ordering, emission-intake ordering, dive-result sorting). It is never wired into frontier priority, dive-start selection, scheduling, or any discovery decision. Role detail in `aesthetic_scoring.md`.

## Location blind reads

- **Why parked:** v7 is taken at its word on certified families; human judgment has been moved to the release rather than spent per-location. Fully reversible — coordinates and artifacts persist — and v7's authority is already structurally bounded (the scheduler allocates, the ranker ranks, v7 only floors).
- **Trigger:** an explicit order, or a release review that demands one.
- **Standing exception — uncertified populations always get their own read:** any population where v7 is not certified is read independently rather than deferred. (Phoenix took its own n=500.) This covers any new family or axis, including **deeper-than-f64 zoom** and any **q4-mined population** — there v7's *floor* still applies, but quality is judged by the q4 objective and the eye, not v7.
- **Always kept, deferral or not:** visual eyeball passes (admission/reject, medoid, release and strange-candidate sheets). These are glances, not labels.

## Mining-head calibration

- **Why parked:** enough of the strange renders are good that the eye currently beats the head; calibration can wait.
- **State:** mining v1 is uncalibrated on the strange population.
- **Trigger:** a release that demands it.
- **Entry point:** calibrate from **labeled precision**, never from the head's raw probability. `strange_candidates_sheet.png` (82 candidates ranked ≥ 0.50) is the seed view for labeling.

## Related, but not part of this cluster

**Native-multibrot `t_good` adoption** is fork-scheduled, not release-gated: it's decided at the next fork launch together with τ_h, and a flip retro-re-decodes the library through the decode-version predicate. Tracked with the discovery/classifier work, not here.
