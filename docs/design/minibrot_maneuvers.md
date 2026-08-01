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

## 2. Two operators, not five

The maneuver set collapses to two. Five named moves (snap-preserving, reframe-outward,
descend-to-child, ascend-to-scale, lateral) alias each other: the first four differ only in
what frame you put on the nucleus you snapped to.

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
run-scoped visited key is `atom_key | op | k`: **the framing is part of the identity**,
because the same atom at two `k`s is two different views.
`[code: minibrot_maneuvers.atom_key_of; steered_frontier._consume_maneuver]`

### 2.6 The lateral probe is a hybrid, not a sweep

`identify_nucleus` sweeps every period `1..pmax`, and with `pmax` reaching
`LAT_PERIOD_CAP = 120` that sweep **was 84% of all maneuver probe cost for 23% of the
pushed nodes** — median 71 ms but mean 442 ms and tail 6.57 s, i.e. tail-dominated.
`[measured: 1,001 deduped operator calls, maneuver_shakedown]`

The fix is the same correction that made the snap probe cheap (§2.2), applied to the
*tail only*: **sweep the cheap head exactly, rank the expensive tail.** Periods `2..24`
are always tried (`LAT_LOW_SWEEP`; `pmax >= 24` always, so the head is a fixed ~23 solves)
and everything above comes from the atom-domain ranking. The split is not arbitrary in
either direction — the head is where the ranking is *weakest*, because a probe seed sits
inside many low-period atom domains at once and "smallest period wins" is what stops the
probe returning the parent itself; the tail is where it is *strongest*, because a deep
atom has one sharp `|z_p|` minimum. Pure ranking with no head lost 17% of availability.

Measured by **replay** — the recorded parent views of the shakedown, both arms driven from
an identically-seeded RNG so the probe seeds are byte-identical, which a live A/B cannot
give (the walk's frontier moves with the cost of the operator).
`[code: tools/atlas/bench_lateral_seeding.py; measured: 239 replayable calls, 2026-07-31]`

| arm | total | mean | max | Newton solves | available |
|---|---|---|---|---|---|
| sweep (reference) | 141.7 s | 593 ms | 6.23 s | 39,261 | 116 |
| head 24 (**shipped**) | 52.6 s | 220 ms | 1.03 s | 13,896 | 109 |
| head 16 | 33.4 s | 140 ms | 0.65 s | 9,869 | 112 |
| no head | 16.6 s | 70 ms | 0.50 s | 4,997 | 96 |

**2.7× cheaper and the 6-second tail is gone.** A cheaper head is available as a knob
(`low_sweep=16` is 4.2×) and is *not* shipped: it triples how often the probe names a
different sibling than the reference (11/105 vs 4/109). A disagreement is not an error —
both arms return a minimal, in-frame, comparable-scale nucleus, and the operator's
contract is "a sibling", not "that specific sibling" — so it is reported as identity
drift, which is the quantity a conservative default should minimise.

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

**One consequence worth stating.** `FRONTIER_CAP` pruning is by priority, so it would delete
maneuver nodes *first* — silently undoing the floor. Maneuver-originated nodes are exempt
from the cap; the exemption is bounded by the same governor that bounds how many can exist.
`[code: steered_frontier.push_children]`

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

## 8. What this is NOT

* **Not an evaluation.** Per-move yield cannot be read until a head has been trained on the
  population these moves generate; the deployed scorer has never seen a maneuver-originated
  view. Composition numbers from a short run are reportable but not readable.
* **Not a ranker change.** No aesthetic score enters the operators or the floor.
* **Not a minibrot source.** Nothing here enumerates atoms from first principles. Every atom
  it returns is downstream of a location the walk already found.
