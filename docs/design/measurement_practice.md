# Measurement practice — designing a measurement, eval, readout or projection

**Consult this before designing any measurement here.** §1 is general; §2 is the lore this
project paid for. Companion: `verification_practice.md` (tests and guards); `retired.md`
(has this been tried).

**Tags.** `[code: path]` — true because the tree says so. `[verdict: who]` — a judgement.

---

## 1. General practice

### Scope of a claim

**A retirement measured under range restriction is scoped to the range, not to the axis.**
State the population a correlation was measured on, and treat every sibling measured on the
same population with the same suspicion. Period was retired as a quality axis on a roster
spanning period 3–15; over 2–74 it reads +0.87. Degree's +0.55 came off the *same*
restricted roster and inherits the doubt. `[code: orbital_field_metrics.md §6]`

**Establish scope by two methods that fail differently, and name the blind spots they
share.** The union is a candidate list, not a verdict — `tools/README.md`'s A/B liveness
index is the worked example, and its two false-survivor classes are named there rather than
argued away.

**A symbol can be load-bearing while its function is dead.** Separate "nothing calls it"
from "nothing imports it": `tools/scoring/active_ckpt.py`'s CLI is entry-point-dead while
~41 modules import the module for the pins it re-exports.

**Absence reports go stale fast when two agents share a tree — anchor to a commit.**

### Instruments

**A verification tool that cannot reach its authority must report UNKNOWN, not ABSENT.**
Failure-to-ask rendered as failure-to-exist is the dangerous direction; `git lfs prune
--verify-remote` called four objects missing on the remote when SSH auth failed, and
"missing" is the one condition under which you must not prune. `[code: artifacts_resolver.md §5]`

**Never characterize anything from truncated output — failures *or* successes.** A `head`
or a `tail` is a biased sample in both directions: a persisted `errs[:10]` described a 19.5%
failure class that was really ~1.2%, because the fastest-returning failure arrives first.

**Before pre-registering a bar, verify the instrument's inputs actually change.** An exact
`0.0000` is not a null result; it is a measurement of nothing. The cheap check is a
byte/pixel delta on the eval slice, run *before* the eval. `[code: auto_maxiter.md, "Why v9
is shelved"]`

**Separate "genuinely bad" from "our instrument is defective" before deprecating anything**
— and expect the artifact to land a layer lower than you guard for.

**Profile the dominant stage first**; apparent duplication is usually deliberate divergence.

### Contrasts and confounds

**Framing both arms by the same objective does not hold confounders constant** — the
objective does the selecting. For a single-variable contrast, MATCH ON THE CONFOUNDER. When
an intervention decouples two correlated variables on purpose, a later read must match on
BOTH: a maneuver changes `fw` without changing walk depth, so any later read has to
depth-match on both or it measures depth. `[code: minibrot_maneuvers.md §5]`

**A search that chooses where it goes confounds its own axes by construction.** To measure
an axis inside a walk, constrain the POPS, not the draws — balancing root *supply* leaves
the batch popped by global priority, which is how a per-degree probe came back 318/234/210/123
— or report the composition and call the reading confounded. `[code: minibrot_maneuvers.md §8]`

**A control arm must differ from the treatment in one thing.** Where a clean control is
impossible, run two comparisons that fail in *opposite* directions and report whether they
agree.

### Cost, wall clock and projections

**Wall clock is not the reproducible quantity.** Quote a work count — solves, iterations,
rows — as the cost invariant, and date any wall clock you keep. The lateral-probe bench
reports Newton solves for exactly this reason. `[code: minibrot_maneuvers.md §2.6]`

**Projecting a long run.** A sample unbiased for *mean per-unit cost* is not unbiased for a
run whose expensive work is contiguous: sample in **run order**, or say plainly it is a
mean-cost estimate and not an ETA. A flat rate in a short sample is a warning when the run
has not yet reached its expensive regime. Reproject from the observed rate — from *recent*
throughput, not the run-to-date average — and **never restate the original ETA**.
(`CLAUDE.md`, "Projecting a long run's wall clock".)

**A backstop longer than the job's budget is not a backstop.** (`CLAUDE.md`, "Four rules".)

### Records and defaults

**Derive state in code; freeze it in records.** A generator reads the state it reports from
the state itself — a hardcoded `True` is how a metadata file outlives what it records — and
a committed record keeps what was true when written. `[code: storage_classes.md, "Derive in
code, freeze in records"]`

**An invariant in prose is not an invariant** → key it and regression-test it.

**Harnesses READ production derived files; they never copy them.** A copy drifts.

**Config changes are announced at decision time, never discovered in a readout.**

**Population-gate at the READER**, not at each of the writers.

**Fail-closed beats a maintained list.** Invert defaults so that forgetting costs
conservatism rather than contamination.

---

## 2. Project measurement lore

**Winner's curse is structural and climbs levels** — instance, family, selection. Steer and
select with one signal; rank and judge with another. A q3/q4 *count* measures the
CLASSIFIER, not the family, until a human looks.

**Enumeration is ~25× the cost of screening**, so *where* you enumerate matters far more
than how widely you screen. `[code: minibrot_maneuvers.md §3]`

**Fertility × ceiling, never fertility alone.** A rate without a ceiling is not a decision.

**Sourcing artifacts hit TAIL statistics, not bulk.** A median over tens of thousands of
swept positions shrugs off biased source atoms; a count of rare accepts is dominated by
which atoms were drawn. Every abundance number is a tail statistic. And **a small curated
pilot does not project a population.** `[code: minibrot_sourcing.md §7]`

**The proxy was never the target.** Each of these was measured and is not the thing it was
read as: nucleus-ness ≠ field-richness · occupancy ≠ mid-detail · f64-field ≠ deep-field ·
p-value ≠ quality · density ≠ balance · edge-energy ≠ quality · crossings ≠ span.

**A cheap proxy's usefulness is PARTITION-dependent.** Never take one threshold, nor one
objective, from a pooled measurement. **A statistic that reverses sign when pooled is not
disqualified — it is conditional**: `coh_scale_drop` is ρ = −0.183 pooled and eval AUC 0.80
/ 0.87 at d4/d5. `[code: minibrot_sourcing.md §8]`

**A verdict from a rendered sheet is only as good as the render policy** — name the cap
beside any number computed off a field. `[code: orbital_field_metrics.md §7]`

**A measure can pass its validation for the wrong reason** — decompose it against its
references before building on it. `[code: orbital_field_metrics.md §5]`

**A ranker cannot evaluate a population it was never trained on.** Reserve a floor for
novel proposals; do not tune a probability into doing the job of a slot.
`[code: minibrot_maneuvers.md §3]`

**Labels are distribution-bound; pooling eras degrades a ranker.**

**Budget in VARIETY, not counts** — distinct-look denomination.
`[code: discovery_pipeline.md]`
