# `tools/atlas/minibrot_maneuvers.py` — minibrot moves as candidate MOVES in a descent

Named for the code artifact that carries the operators. This doc owns **what the operators
are, why there are two of them, and how they enter the walk**. It does not own the `A`
instrument's mathematics (`atom_instrument.md`), the minibrot sourcing arc's measured
evidence (`minibrot_sourcing.md`), or the walk/scheduler machinery
(`discovery_pipeline.md`) — those are referenced by name and their values are not restated.

**Tags.** `[code: path]` — true because the tree says so. `[measured: population]` — a
number, with the population it is true *of*. `[verdict: who]` — a judgement call.

---

## 1. The premise: a minibrot is an OPERATOR, not a source

Seven source sheets settled the question. **Every source rated viable is downstream of a
descent, and every source that enumerates minibrots from first principles is dead.** The
minibrot-sourcing arc reached the same place from the other side: the 487-crop read
measured *auto-framed yield at one scale per atom*, and its correction of record is that a
minibrot's value is **as a marker of an interesting, high-density region, not as a window
that is itself shippable** — with **scale the axis that was never varied**
(`minibrot_sourcing.md` §9). `[verdict: Matt]`

So minibrots belong in the descent pipeline **as candidate moves**: a reframing applied to
a location the walk already found. That single decision fixes everything below — it is why
they enter as frontier *nodes* rather than as a separate run, why they need a slot rather
than a score, and why "unavailable" has to be a first-class, frequent answer.

## 2. Three operators — two reframings and one enumeration

The maneuver set collapses to **two reframings**. Five named moves (snap-preserving,
reframe-outward, descend-to-child, ascend-to-scale, lateral) alias each other: the first
four differ only in what frame you put on the nucleus you snapped to. A **third** operator,
`neighborhood_expand` (§2.7), was added in v1.4 and is not a reframing at all — it
enumerates, which is why it is the only one with a probe budget rather than a probe count.

**`snap_to_nucleus(view, k)`** — atom-domain probe at the view's centre → Newton → recentre
on the nucleus. `k` is the frame: `None` preserves the view's own `fw`; otherwise
`fw = k × atom size`. `k=None` is snap-preserving, large `k` is reframe-outward, small `k`
is descend-to-child, and any `k` is ascend-to-scale. One operator, one continuous axis, one
attribution column. `[code: minibrot_maneuvers.snap_to_nucleus]`

**`lateral_to_sibling(view)`** — move to a nearby sibling nucleus at comparable scale.
Probe seeds are drawn at radii **measured in units of the parent's own window scale**, so
"nearby" means the same thing for a shallow and a deep parent, and the period ceiling is
**scaled to the parent's period** rather than flat. Both come from the source sheets'
neighbourhood generator, where a flat ceiling was measured to return 15 atoms from 360
probes — the ceiling, not the source, was the limit.
`[code: minibrot_maneuvers.lateral_to_sibling; tools/sources/sources.py::src_neighborhood]`

### 2.1 Atom size is `1/|A|`, and the naive law is forbidden

The frame comes from the `A` instrument: `size ≡ 1/|A|`, exact at every period (it is the
same analytic quantity as `nucleus_size_estimate`, not a second estimate).
**The naive degree-2 `λ²` law must never be used at `d ≥ 3`** — it under-sizes the atom by
4–2497× and frames it all-black. `[code: deep_center_finder.atom_instrument;
docs/design/atom_instrument.md]`

### 2.2 The correction that makes the atom-domain probe work

The classical probe is: along the critical orbit at `c`, `|z_k|` is small at `k = p` for
the period-`p` component whose atom domain contains `c`. Taking the **argmin** of `|z_k|`
as the period **is wrong, and wrong silently**. A nucleus is *superattracting*, so just off
it `|z_{2p}| ~ |z_p|²  ≪ |z_p|`: the global argmin lands on a high multiple of `p`, never
on `p`. Newton at that multiple returns a non-minimal solution, `atom_lib.make_atom`
rejects it, and the operator reports "unavailable" on views that are sitting on a perfectly
good atom.

So the candidate set is the argmins' **divisors**, tried in **increasing period order** —
smallest period = largest containing component, the same "smallest period wins" rule
`identify_nucleus` uses. Period 1 is skipped (its nucleus is `c = 0`, which `ORIGIN_EPS`
rejects anyway). `[code: minibrot_maneuvers.period_candidates;
tools/atlas/test_minibrot_maneuvers.py::test_period_candidates_yields_the_true_period_via_divisors_not_the_raw_argmin]`

This is *ranking only*. The known-imperfect satellite over-calling of the same argmin
(`atom_lib` module docstring) costs nothing here: a bad rank is one wasted Newton solve.

### 2.3 Two refusals that are geometry, not taste

* **A snap must not be a teleport.** The nucleus has to land within `SNAP_MAX_FW_MULT ×
  fw` of the centre, i.e. inside the frame being reframed. Otherwise the operator has not
  reframed *this* view, it has jumped to a different one.
* **The f64 pixel-spacing wall is checked a priori.** A frame at `k × size` whose pixel
  spacing at the descent's node width would fall under `PERTURB_SPACING` cannot be rendered
  by the f64 backend the descent uses, and is refused **with no render attempted** — the
  `A` instrument's a-priori wall predictor, applied at node fidelity rather than at deploy
  fidelity, because it is the node render that would quantize.
  `[code: minibrot_maneuvers._frame_for, _wall_margin_decades]`

This is a *renderability* check, not the roster's `A` feasibility **cut**. That cut is
deliberately not applied: it fired on 3 of 163 atoms and removed d2's best material, and
the source sheets record it as recorded-never-enforced. `[measured: 163 atoms,
minibrot_sourcing.md §6]`

### 2.4 Unavailability is the normal case

Newton convergence ran **~17%** in the 200-atom enumeration (2,160 solves → 200 kept). Every
entry point returns a `Maneuver` with `available=False` and a **named reason** —
`no_converge`, `nucleus_outside_frame`, `degenerate_or_not_minimal`, `f64_spacing_wall`,
`fw_over_root_scale`, `hit_parent`, `scale_mismatch`, `no_sibling_found`. Nothing raises.
Callers must expect it, and a quota over these is a quota **of available** (§3).

### 2.5 Dedup: reuse the read-time key, do not write another

Multiple frontier members snapping to one nucleus is the normal case. The collapsing key is
the **shared read-time canonicalization** — `snap_near_zero` + the sector-canonical rounded
key, i.e. exactly `deep_center_finder.snapped_dedup_key`, the function `collapse_population`
uses. An atom found by an operator therefore carries the same `id` as one found by a source
sheet or already sitting in the triage pool, and the overlap falls out for free. The
run-scoped visited key is `atom_key | k`: **the framing is part of the identity**,
because the same atom at two `k`s is two different views — and **the operator is not**.
That second half is a v1.4 correction: the key used to carry `op` as well, which was
harmless with two operators that rarely collided and is not harmless with three. `lateral`
and `neighborhood_expand` sample the *same disc* and routinely reach the same sibling, so
an op-keyed set pushed one view twice under two provenance labels. The pushed node is
`(cx, cy, fw)`, which `(atom, k)` determines and `op` does not; `op` is provenance.
`[code: minibrot_maneuvers.atom_key_of; steered_frontier._consume_maneuver;
test_minibrot_maneuvers.py::test_the_visited_key_is_the_atom_and_its_framing_not_the_operator]`

### 2.6 The lateral probe is a hybrid, not a sweep

`identify_nucleus` sweeps every period `1..pmax`, and with `pmax` reaching
`LAT_PERIOD_CAP = 120` that sweep **was 84% of all maneuver probe cost for 23% of the
pushed nodes** — median 71 ms but mean 442 ms and tail 6.57 s, i.e. tail-dominated.
`[measured: 1,001 deduped operator calls, maneuver_shakedown]`

The fix is the same correction that made the snap probe cheap (§2.2), applied to the
*tail only*: **sweep the cheap head exactly, rank the expensive tail.** Periods `2..16`
are always tried (`LAT_LOW_SWEEP`; `pmax >= 24` always, so the head is a fixed 15 solves)
and everything above comes from the atom-domain ranking. The split is not arbitrary in
either direction — the head is where the ranking is *weakest*, because a probe seed sits
inside many low-period atom domains at once and "smallest period wins" is what stops the
probe returning the parent itself; the tail is where it is *strongest*, because a deep
atom has one sharp `|z_p|` minimum. Pure ranking with no head lost 17% of availability.

Measured by **replay** — the recorded parent views of the shakedown, both arms driven from
an identically-seeded RNG so the probe seeds are byte-identical, which a live A/B cannot
give (the walk's frontier moves with the cost of the operator). Newton solves and
availability are deterministic and reproduce exactly across runs; the wall clocks are one
box's and move a few percent, so read the **solve count** as the cost invariant.
`[code: tools/atlas/bench_lateral_seeding.py; measured: 239 replayable calls,
re-measured at the head-16 default 2026-07-31:
`uv run python tools/atlas/bench_lateral_seeding.py --low 24 16`]`

| arm | total | mean | max | Newton solves | available | names a different sibling |
|---|---|---|---|---|---|---|
| sweep (reference) | 144.4 s | 604 ms | 5.94 s | 39,261 | 116 | — |
| head 24 | 57.6 s | 241 ms | 1.41 s | 13,896 | 109 | 4 / 109 |
| head 16 (**shipped**) | 34.5 s | 144 ms | 0.68 s | 9,869 | 112 | 11 / 105 |
| no head | 16.6 s | 70 ms | 0.50 s | 4,997 | 96 | — |

**4.2× cheaper than the sweep, and the 6-second tail is gone.** Head 16 ships over head 24
(2.5×): it is another 29% fewer Newton solves at **no availability cost** — 112 against
109, i.e. head 16 is if anything the *better* arm on the only axis that gates throughput
(it loses 11 of the sweep's and gains 7 the sweep never found, net 112).

Head 24 was the original default on one argument only: it names a different sibling than
the reference less often (4/109 vs 11/105). That argument does not survive contact with
what the operator promises. A disagreement is not an error — both arms return a minimal,
in-frame, comparable-scale nucleus, and the contract is "**a** sibling", not "that specific
sibling". Sibling identity is not reproduced across reruns by anything downstream (the walk
re-derives it, and the dedup key is the *nucleus'* canonical key, so two arms agreeing or
disagreeing costs nothing either way), so identity drift is a quantity with no consumer —
and paying 1.7× the probe cost to minimise it buys nothing. The head remains a live knob
(`lateral_to_sibling(low_sweep=...)`, and `bench_lateral_seeding.py --low`), so restoring
24 is one argument.

### 2.7 `neighborhood_expand` — the sheet-3 mechanism as an operator

**How sheet 3 actually enumerates, and why its one prior run produced only 22 atoms.**
`src_neighborhood` takes a list of parent nuclei and fires `per_parent` probe seeds around
each: radius drawn uniformly from `(2, 8, 32)` **× the parent's own window scale**, angle
uniform, then `identify_nucleus(seed, period_min=1, period_max=pmax, near=rad*4)` with
`pmax = min(200, max(24, 3 × parent period))` — smallest period wins. It stops on a target
count or a deadline. The sheet-3 run passed 60 parents × `per_parent = 6` = **360 probes**,
and its recorded stats close exactly: `parents 60, probes 360, hit_parent 317,
no_nucleus_near_seed 1, seconds 146.8`, with no `budget_stopped_at` key. So the answer is
**neither exhausted supply nor the wall-clock budget**: the configured probe budget was
spent in full, in 2.4 minutes, and **88% of the probes returned the parent itself**. Only
42 probes found a non-parent nucleus, and those deduped to 22. The binding constraint is
that a seed 2–32 window scales from a nucleus is still inside that nucleus' *atom domain*,
which is far larger than the atom, so "smallest period wins" hands the parent back.
`[measured: data/minibrot_sources/neighborhood/meta.json, built 2026-07-30]`

That single number sets the operator's design. **`m` is a ceiling on the answer; the probe
count is the ceiling on the bill, and it is the one that binds** — a budget expressed as
"find `m` neighbours" is an unbounded budget at an 88% miss rate. So `neighborhood_expand`
takes `max_found = m` (default 8) *and* `max_probes` (default 12), early-exits at `m`, and
counts `hit_parent` apart from `duplicate_neighbour` so a run can tell "the disc is
exhausted" from "the disc keeps returning the parent".

**What is ported and what is not.** Ported: the disc geometry, the parent-relative radii,
the parent-scaled period ceiling, and the §2.6 hybrid seeding. **Not** ported: the
*symmetry* of lateral's comparable-scale window. Sheet 3 probes "at comparable **and
smaller** scale", so operator 3's window is **one-sided** — unbounded below,
`NBH_SCALE_UP_DECADES = 1.0` above. The upper bound is not decoration: unfiltered, the
first smoke case returned a period-2 giant **+1.76 decades** larger than the parent, and
framing that at `k × size` proposes a near-base-scale view the walk's own root draws
already cover. An atom that much larger is an ancestor, not a neighbour.

**k framing applies as for snap** (§7.1): one enumeration, one `Maneuver` per
(nucleus, `k`), the whole enumeration charged to the first emitted row. The operator
returns candidates and selects none — screening spawns a process and this module is pure
mpmath (§4), so the caller screens and takes the top `n` (default 2) by `radial_range`.
`[code: minibrot_maneuvers.neighborhood_expand; steered_frontier._nbh_top_n]`

**Lateral is NOT subsumed, and is not deleted.** §2.8.

### 2.8 Is `lateral_to_sibling` subsumed?

The two operators sample the same disc, so the question is real: at `m = 1` with an equal
probe budget, `neighborhood_expand` walks byte-identical seed points (both call
`_draw_probe_seed`, so one seeded RNG gives both the same radius/angle pairs) and returns
the first survivor — which is what lateral does. The filters are the only difference, and
they differ in exactly one place: lateral's scale window is **symmetric**
(`|log10 ratio| <= 1`), operator 3's is **one-sided**.

Measured by replay, both arms off one RNG seed per case and an EQUAL probe budget (3), or
the comparison measures budget rather than filters.
`[measured: 178 replayable lateral calls from the v1.4 shakedown, 2026-08-01;
cmd: tools/atlas/bench_neighborhood_subsumption.py --log
data/discovery/maneuver_v14_shakedown/maneuvers.jsonl --limit 200]`

| | lateral available | nbh available | identical first pick | lateral's pick inside nbh's set | lateral-only | nbh-only | Newton solves |
|---|---|---|---|---|---|---|---|
| `m = 1` | 83 / 178 | **92** / 178 | 80 / 83 | 80 / 83 | **0** | 9 | 6,878 (vs 7,107) |
| `m = 8` | 83 / 178 | **92** / 178 | 80 / 83 | **83 / 83** | **0** | 9 | 8,840 (vs 7,107) |

**On this population lateral IS subsumed.** There is no case where lateral finds a sibling
and neighbourhood finds nothing — `lateral-only = 0` at both `m` — and at `m = 8` lateral's
pick is inside neighbourhood's set **every time**. At `m = 1` it is also *cheaper* than
lateral (6,878 solves against 7,107): it early-exits at the first find exactly as lateral
does, and its looser scale filter stops it sooner.

The 9 extra availabilities are the one-sided window doing exactly what §2.7 predicts: **5 of
them are cases where lateral refused with `scale_mismatch`, and all 5 are below its window.**

**Scope, because a subsumption verdict is a retirement claim.** 178 replayed calls off one
15-minute shakedown at `probe_p = 0.5` — the parents the walk had reached by then, not the
deep tail. `measurement_practice.md`'s standing caution applies to this reading as much as
to any other: it is scoped to that population. **Lateral is not deleted**, and the run
config keeps both, so the next run's log is what re-tests it at depth.

**A disagreement is not a defect.** Both contracts promise "*a* nearby nucleus", not "*that*
one" — the same identity-drift reading §2.6 records for the lateral head, and the dedup key
is the nucleus' canonical key, so nothing downstream reproduces sibling identity either way.
The 3 of 83 first-pick disagreements are that, not error.

## 3. Selection is a reserved FLOOR of frontier slots, not a probability

The walker already ranks a candidate slate, so a new proposal source needs a **slot**, not a
coin flip.

**The cold-start trap this exists to defeat.** The active head has never been trained on
maneuver-originated views. On score alone it rejects them by default, so the material needed
to train its successor never gets generated — the queue matures and starves an untouched
population. This is the same mechanism, and the same measured rationale, as the julia
breadth floor. `[verdict: Matt]`

**The quota is of AVAILABLE, not of all slots.** Given ~17% convergence the operator is
often simply not there. Whatever the floor cannot fill falls straight back to the ordinary
priority order in the same pop — an unfillable quota must never stall the frontier. Two
counters keep the two constraints apart, and the difference between them is what says
whether *the floor* or *the operator* is the limit at scale:

| counter | means |
|---|---|
| `quota_bound` | reserved slots that promoted a node the plain priority top-B would **not** have taken — the floor actually binding |
| `quota_unfilled` | reserved slots that went unused **for lack of availability** |

`[code: steered_frontier.SteeredFrontier._split_reserved]`

**"Maneuver node" here means the maneuver-descended SUBTREE, not the origin node.** The
`man` stamp rides every rung below a fired operator (§5), so `_split_reserved`'s reserved
pool, the frontier share (§3.2) and every `man_*` counter are over the subtree. This is the
right scope — the head has not seen a maneuver view's *children* either — but it makes the
counters unreadable if you assume otherwise: measured 14 batches into the v1.4 exploration
run, **52% of the frontier was maneuver-descended while only 55 nodes of 686 were origins**,
which is why `quota_passed_over` (1,044) can exceed `nodes_pushed` (341) without either
being wrong. `[measured: data/discovery/maneuver_v14_exploration, batch 14, 2026-08-01]`

**A probability IS used — as a COST governor, not as selection.** The probe is an
enumeration cost, and enumeration measured ~25× the screening cost. `ProbeGovernor` bounds
it two ways: a Bernoulli(`p`) draw per rung, **and** a region cache keyed on a coarse
`(degree, cx, cy, fw-decade)` cell, because siblings in a hot lineage sit in one cell and
re-probing them re-derives the same nucleus at full Newton cost. The cache beats the coin
(a cached cell is skipped whatever the coin says) and both survive a resume.
`[code: minibrot_maneuvers.ProbeGovernor]`

**`pref_loc_v1` stays out of frontier priority.** The ranks-never-steers seam is untouched:
a slot reservation is not a ranker change, and the preference ranker is absent from
`_split_reserved` exactly as it is absent from `pop_batch_scheduled`. Said so in a comment
at the seam. `[code: steered_frontier._split_reserved docstring]`

### 3.1 The richness screen, and what may select on it (v1.4)

Every available candidate is measured: `radial_range` **and** `radial_rings` at the 64×36
screening geometry on the **4× atom-size frame**. What the measures are and what they were
validated against is `orbital_field_metrics.md`'s and is not restated; three things about
their use here are this doc's.

**One field per NUCLEUS, not per row.** The screen frame is 4× the *atom*, so it cannot
depend on `k` — §7.1's shared solve, extended to the screen. A run-scoped cache keyed on
the shared `atom_key` (§2.5) makes a repeated nucleus free, and the whole batch's distinct
nuclei are screened in one concurrent pass, because the cost is process spawn (~40 ms) and
not compute (~2 ms). `[code: maneuver_screen.ScreenCache]`

**The score describes the ATOM, not the view.** A `k = None` row's own frame may be a
thousand atom-widths wide; its score is still the 4× number, because 4× is the only frame
scale any orbital measure has ever been validated at (`orbital_field_metrics.md` §2) and a
score at another scale would carry none of that validation.

**RECORDING IS UNCONDITIONAL; SELECTING IS NOT.** Scores land on every candidate — pushed,
passed over, or beaten to a quota slot — and ride the frontier node into `state.json`, the
harvest log and the ledger. `--maneuver-range-prior` (default **off**) gates the only two
places anything selects on them:

1. **Quota fill order.** When available candidates exceed the per-batch quota, the reserved
   slots go to the highest `radial_range` instead of to the incoming priority order. It
   changes *which* maneuver fills a slot, never *how many*; unscreened candidates sort last
   and are never excluded, because the screen ranks the quota and does not gate it.
2. **The node's prior.** `NEUTRAL_PRIOR` becomes
   `NEUTRAL_PRIOR + gain × (percentile − 0.5)` — the percentile of this atom's
   `radial_range` against **the run's own accumulating distribution**, since absolute ring
   scores are comparable only within one (geometry, cap policy) pair. Below 8 observations
   the percentile returns exactly 0.5, i.e. the unchanged neutral prior.

**The bound on the prior is the design, not a tuning choice.** The term is symmetric about
`NEUTRAL_PRIOR`, so the flag *reorders* maneuvers without inflating them as a class, and it
is bounded to ±`gain/2` = ±0.25 at the shipped `gain = 0.5`. An ordinary node's
`cheap_eord` runs over `[0, K−1] = [0, 3]` on the K=4 head, so the best-ranked maneuver sits
at 1.25 and still loses to any ordinary node scoring above that. **A maneuver out-competes a
scored node via the quota floor, never via the prior** — §3's whole argument would collapse
otherwise. `gain` is the knob; raising it past ~2.0 would break that property.
`[code: maneuver_screen.range_prior_delta; steered_frontier.MAN_RANGE_GAIN_DEFAULT;
test_maneuver_screen.py::test_the_prior_term_is_bounded_and_cannot_reach_a_well_scored_ordinary_node]`

**This is not a ranker change.** No aesthetic score enters. `radial_range` is a
field/geometry measure, exactly like the `A` instrument and the black/band/occupancy gates —
the ranks-never-steers seam is where it was.

**The cap policy is its own, and stamped.** The screen renders under 24× the legacy
production envelope clamped at 67000 (`mi12000k0.3c4800-67000`), so the numbers do not move
when the production cap moves. That policy is listed in `retired.md` and is **un-retired for
this use only** — see the dated entry there. In practice the 67000 clamp binds below
`fw ≈ 2e-5`, so at maneuver frame depths the screen runs at **1.8–3× production**, itself
already the ×8 convergent cap the 32-atom ladder measured (`auto_maxiter.md`). Whether that
is genuinely non-clipping *on this population* is a measurement, not an assertion: every
score carries `cap_headroom` and `clamped`. `[code: maneuver_screen.SCREEN_MAXITER_POLICY]`

**One consequence worth stating.** `FRONTIER_CAP` pruning is by priority, so it would delete
maneuver nodes *first* — silently undoing the floor. Maneuver-originated nodes are therefore
**protected** from the pooled prune.

### 3.2 Protected is not exempt

**Corrected 2026-08-01.** The protection was total, and total
protection has the mirror-image failure: once the maneuver population passes `FRONTIER_CAP`
the ordinary nodes' room goes to zero and *every one of them* is evicted, leaving a frontier
that is 100% maneuver nodes and a walk with nothing else to expand. That is precisely the
capped-root starvation `pop_batch` already records, reached by a second route. It was
unreachable with two operators and is reachable with three: a 2-minute shakedown pushed
**~40 maneuver nodes per batch against ~21 expanded**, which crosses 6000 inside a 7-hour
run. So maneuvers hold a guaranteed **share** (`MAN_FRONTIER_SHARE = 0.5`) and are pruned
among themselves beyond it; unused share falls to the ordinary nodes and unused ordinary
room falls back to the maneuvers, so below the share nothing changes — which is every run
before this one. `[code: steered_frontier.prune_frontier;
test_minibrot_maneuvers.py::test_a_flood_of_maneuver_nodes_cannot_evict_every_ordinary_node]`

**What the rule actually guarantees**, since the subtree scope above makes the maneuver side
the majority quickly: ordinary nodes always keep `min(len(others), CAP × share)` and
maneuvers take the remainder. So neither side can starve the other — a 90%-maneuver frontier
still keeps every ordinary node, and a 90%-ordinary frontier still keeps 3,000 maneuvers.
The share is a mutual floor, not a ration.

## 4. A maneuver enters as a NODE, not as a scored candidate

A fired operator pushes a new **frontier node** — unscored, neutral prior, carrying the
parent's `root_id` (so the per-root `M_CAP` still binds) — exactly as a root does. The
ordinary `guided-descend --expand` / cheap-score / harvest machinery then takes it from
there, which means:

* the maneuver view goes through the **same** black-cap → band → occupancy gates as every
  other view, with no gate logic duplicated in Python;
* nothing new has to render or score, so the operator module stays pure mpmath with no
  subprocess and no torch;
* the reserved floor is *necessary* rather than decorative, because an unscored node sits at
  a neutral prior and a mature frontier will never reach it.

**Maneuver moves are interleaved in the same walk as ordinary moves** — proposed off the
rungs about to be expanded, landing on the frontier for a later batch. Never a separate run,
which would confound the move with the run. `[code: steered_frontier.propose_maneuvers]`

## 5. Provenance, or the run is unreadable later

Every view stamps the **operator**, **`k`**, its **parent view id**, and the atom it is a
reframing of. The stamp rides the whole subtree — `expand_group` propagates `man` down every
rung the way `mix_source` already is — so a maneuver's descendants stay attributable without
reconstructing lineage from coordinates. It lands in four places:

| sink | carries |
|---|---|
| `maneuvers.jsonl` | **every probe decision**: governor skips (coin / region-cache), each operator call, its availability + named reason, `probe_s`, `newton_solves`, and `used` — including **available-but-unused** with `unused_reason` (the atom was already visited) |
| frontier node / `state.json` | `man = {op, k, origin_node_id, atom_id, atom_key, period, log10_abs_A, window_scale, degree, parent_*}` |
| `harvest_log.jsonl` | `maneuver` on every harvest check, admitted or not |
| `outcome_ledger.jsonl` | `maneuver` on the admitted row, plus `mix_source = "maneuver:<op>:k=<k>"` |

**`fw` and depth are both recorded on every view, and after this feature they decouple.** A
maneuver is a *reframing*, not a rung, so `depth` (the walk-rung count) is unchanged by it
while snap-and-rescale changes `fw` by orders of magnitude. Depth correlates with quality,
so **any later read has to depth-match on both, or it measures depth.** Ledger rows carry
`reached_depth`, `outcome_fw` and now `seed_fw`. `[code: steered_frontier.admit]`

**"Available but unused" is recorded deliberately.** "The operator had nothing" and "the
operator had something we already had" are different constraints, and a log that only
records pushes cannot tell them apart at scale.

## 6. Scope: c-plane only, and why

`degree_of(partition)` returns a multibrot degree for the four c-plane partitions and
`None` for julia/phoenix. A julia viewport is a **z-plane**; it has no nucleus in the
parameter-plane sense, so the operators are not defined there and are skipped rather than
faked. Dives force the operators off — a single-track dive has no frontier to reserve slots
in. `[code: minibrot_maneuvers.PARTITION_DEGREE; steered_frontier.__init__]`

## 7. Defaults, and the off switch

`--maneuvers` is **off by default**, and off means every path short-circuits so the run is
byte-identical to the pre-maneuver frontier — the same acceptance shape the morph-novelty
and scheduler features use.

| knob | default | what it is |
|---|---|---|
| `--maneuver-quota` | 4 | reserved slots per batch, **of available** |
| `--maneuver-probe-p` | 0.25 | cost governor: P(probe fires) per popped rung |
| `--maneuver-k` | `none,4,16` | preserve-fw, the 4×-atom frame, and the 16× wallpaper frame |
| `--no-maneuver-lateral` | (lateral on) | disable the expensive operator; snap only |
| `--maneuver-neighborhood` | **off** | enable operator 3 (§2.7) |
| `--maneuver-nbh-m` | 8 | ceiling on distinct nuclei *enumerated* per call |
| `--maneuver-nbh-n` | 2 | how many of them are *proposed*, by `radial_range` |
| `--maneuver-nbh-probes` | 12 | the probe budget — the bound that actually binds (§2.7) |
| `--maneuver-range-prior` | **off** | let the screen select (§3.1); off is byte-identical selection |
| `--maneuver-range-gain` | 0.5 | prior term magnitude; bounded to ±gain/2 (§3.1) |

The screen itself has no off switch: it runs whenever `--maneuvers` does, because
recording is unconditional (§3.1). With `--maneuver-range-prior` off the walk's
**trajectory** is byte-identical to v1.3 — the screen consumes no RNG and gates nothing —
but the run is not byte-identical in wall clock or in log columns, and saying "byte-identical"
without that qualifier would be the wrong claim.

`k = 4` is the framing the deep-center emitter already suggests for a nucleus-centred
frame — `fw ≈ size` is mostly interior black and `fw < size` on-nucleus is pure black.
`[code: deep_center_finder.make_deep_center]`

**`k = 16` is in the set because it is the framing worth LABELING, not because it is
cheap.** The 4× frame answers *"is this atom good?"*; the 16× frame is often close to a
usable wallpaper frame by itself, which is the material the corpus wants. That it is also
free is a consequence of §7.1, not the reason. **No small `k`** — framing *into* the atom
is interior black, which is the same fact that sets the floor at `k = 4`.

### 7.1 A `k` is a reframing, not a probe

The nucleus does not depend on the framing, so `snap_to_nucleus_multi` runs **one**
atom-domain probe + Newton pass per view and emits one `Maneuver` per `k`. Adding a `k`
therefore costs a division, not a probe — which is what makes the k set a design axis
rather than a cost knob. The naive per-`k` loop it replaces re-solved the identical
nucleus once per framing.

The shared solve is charged to the **first row only** (`newton_solves = 0` and
`extra.reused_solve` on the rest), so summing `probe_s` over the emitted rows is the true
cost of the call and a per-`k` cost read is not N copies of one solve. The framing verdict
stays **per `k`**: one solve, three answers — a shallow atom can take `k = 4` and refuse
`k = 16` as `fw_over_root_scale` off the same nucleus.
`[code: minibrot_maneuvers.snap_to_nucleus_multi]`

## 8. Per-degree availability and cost — measured

`[measured: 1,180 distinct operator decisions + 1,113 governor rows over 44 batches /
19.9 active minutes / 1,408 expansions, scheduler off, `--mem-recency`, defaults elsewhere
(`probe_p = 0.25`, `k = none,4,16`), 2026-07-31]`
`[cmd: tools/atlas/steered_frontier.py --run-dir data/discovery/maneuver_degree_probe
--families mandelbrot,multibrot3,multibrot4,multibrot5 --budget 20 --maneuvers
--mem-recency --below-normal --seed 20260731; then tools/atlas/maneuver_degree_readout.py]`

**Availability rises with degree, on both operators — the expensive degrees are the easier
ones.**

| | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| `snap_to_nucleus` calls | 318 | 234 | 210 | 123 |
| snap available | **51.9%** | **75.2%** | **72.9%** | **80.5%** |
| `lateral_to_sibling` calls | 106 | 78 | 70 | 41 |
| lateral available | **22.6%** | **35.9%** | **32.9%** | **46.3%** |
| f64 node margin, median decades | 7.50 | 7.92 | 8.28 | **8.49** |

**The consequence is confirmed; the mechanism is not measured.** The prediction was that a
degree-`d` atom *of a given period* is intrinsically larger. Within period bands median
`log₁₀|A|` is flat within noise in the low bands and inverted or too thin in the high ones
(d5's 32–63 cell is three atoms). What the degrees differ in is **period mix** — median
pushed period d2 22 / d3 16 / d4 9 / d5 8 — and this run cannot separate the two, because
the walk chooses where it goes. `minibrot_sourcing.md` §5 carries the consequence for any
degree decision.

**The populations are unbalanced despite balanced supply.** Root draws are `B` per family
with the scheduler off, so *supply* was balanced; realized snap calls were **318 / 234 / 210
/ 123**, ~2.6:1 against d2, because the batch is popped by global priority. Governor skips
follow the same shape (404 / 288 / 272 / 149), so this is where the walk went, not a
maneuver-side bias. d5's cells are the thinnest in every table — enough to establish
availability, not enough for the period-controlled `|A|` read.

**d2's low availability is one mechanism: the nucleus keeps landing outside the frame.**
`nucleus_outside_frame` — the teleport guard of §2.3, not Newton failure — is **50 of d2's
51 snap refusals** (15/19 at d3 and d4, 5/8 at d5); `no_converge` is 1–4 per degree.
`f64_spacing_wall` has **never fired**, at any degree or any `k`, which is consistent with
the 7.5–8.5-decade margin table: the a-priori wall check has never been the binding
constraint on this population. `fw_over_root_scale` fired once (d3, `k = 16`) — the first
k-dependent refusal ever recorded, and exactly what §7.1 predicts, since a solve-level
refusal cannot depend on `k` and only the framing verdict can.

**Not an evaluation of move quality.** 14 maneuver-originated admissions are reported for
completeness and are not readable as yield — see §9.

### 8.0 The operators feed themselves, and it inflates every pooled rate

**Read this before any availability number in §8.** A view produced by snapping to a nucleus
is, by construction, centred on a nucleus — so snapping it again nearly always succeeds.
Once the operators push enough nodes, the views they are applied to are increasingly views
*they produced*, and the pooled availability rate becomes a property of the feedback loop
rather than of the operator. This was latent with two operators and is dominant with three,
because `neighborhood_expand` pushes several nodes per fired probe.

`[measured: 3,415 distinct operator decisions, v1.4 shakedown, probe_p = 0.5, 2026-08-01;
cmd: tools/atlas/maneuver_degree_readout.py --run-dir data/discovery/maneuver_v14_shakedown]`

| op | fresh view | self-fed view |
|---|---|---|
| `snap_to_nucleus` | **72.7%** (330 calls) | 97.6% (741) |
| `lateral_to_sibling` | **34.5%** (110) | 44.3% (253) |
| `neighborhood_expand` | **88.2%** (473) | 94.2% (1,508) |

**73% of operator decisions and 75% of governor rolls were on views the operators produced**
— and that is a **lower bound**, since a deeper descendant of a maneuver node is not
detectable from the log alone (children get fresh ids from the expand).

⇒ **Quote the `fresh` column.** Snap's fresh 72.7% against §8's pooled two-operator 67%
is a modest population difference, not an improvement; the apparent jump to ~87% pooled is
the loop. The readout now splits this automatically so the pooled number cannot be quoted by
accident. `[code: maneuver_degree_readout.load_maneuvers, §1b]` `[verdict: Matt]`

This is `measurement_practice.md`'s "a search that chooses where it goes confounds its own
axes by construction", one level in: here the search is choosing its own *inputs*.

### 8.1 The config record

`--maneuver-probe-p` has been **0.25 since the operators landed** (`MAN_PROBE_P_DEFAULT`,
`fad68df`), and §7's table has always said so. What ran at **0.5 was the shakedown**, which
passed `--maneuver-probe-p 0.5` explicitly (its `summary.json` records `"probe_p": 0.5`).
This is worth stating because the shakedown's headline **22.4% probe-cost share** is
measured against a coin firing twice as often as the default and overstates the default
configuration's cost: the same measure on this run, at the default, is **5.4%**. Half of
that drop is the §2.6 cost cut and half is firing half as often — so read the per-call
numbers (lateral mean 442 → 176 ms, max 6.57 → 1.32 s) as the clean comparison, not the
share.

## 9. What this is NOT

* **Not an evaluation.** Per-move yield cannot be read until a head has been trained on the
  population these moves generate; the deployed scorer has never seen a maneuver-originated
  view. Composition numbers from a short run are reportable but not readable.
* **Not a ranker change.** No aesthetic score enters the operators or the floor.
* **Not a minibrot source.** Nothing here enumerates atoms from first principles. Every atom
  it returns is downstream of a location the walk already found.
