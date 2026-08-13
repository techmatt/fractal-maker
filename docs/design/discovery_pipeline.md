# Discovery pipeline — walk, freshness prior, deficit scheduler

Distilled from the rescued as-built notes (`descent_algorithm_current`,
`campaign2/readout` §H, `atlas/scheduler_smoke/readout`). Governs the discovery/descent
orchestration: how a single walk is structured, why the freshness prior and dives are
incompatible, and how cross-family budget is allocated. This is the machinery that
feeds the corpus pipeline described in `CLAUDE.md` (§Corpus & classifier pipeline).

## 1. The descent is a walk + a reward pass

Two layers:

- **Walk** (Rust `guided-descend`) — **blind to the aesthetic classifier**. Steers
  purely on **field statistics and geometry**.
- **Reward** (Python, post-walk) — the **only** place a neural CORN score enters.

A walk is a **single greedy chain** (no beam / branching). Per rung it:

1. draws a small fixed set of policy-proposed centers (foci scale-space blob-finder /
   density / boundary-random mixture),
2. passes them through **black-cap → band → occupancy gates**,
3. at the shipped default picks a **uniform-random survivor**.

So field statistics **only gate; they never rank the winner.** Walk length (terminal
depth) is drawn once up front — **no score-based early exit**. Julia / phoenix flavors
share the identical per-rung loop; they differ only in recurrence kernel, root draw
(one deterministic shared z-plane root vs. c-plane seed-list/8k-field/flat mixture),
and per-degree band constants.

**Standing redesign levers** (the "signal computed but not used to choose" surface):

- Priority has **no depth-seeking or novelty term** — best-first on cheap score buys
  shallow breadth, not depth.
- Winner-selection is a **coin flip among survivors** — the largest unused-signal knob.

**Cheap-render steering is proven viable:** cheap 384×216 ss1 twilight renders reproduce
canonical E[ord] ranking at Spearman ≈0.95 / rung top-1 agreement ≈0.84 — a future
classifier-in-the-loop walk can steer on cheap renders.

## 2. Freshness prior and dives are structurally incompatible — run dives with the prior OFF

The **freshness prior** is an *exploration* tool: it seeds the dedup/steering clouds
with prior-library coords so root draws avoid re-covering known ground. A **dive** is
*exploitation*: it descends the greedy argmax path **from an existing admission**, whose
coord and basin are by construction **already in the prior cloud**.

So with the prior on, a dive's pre-canonical coord-dup filter rejects the descent
against the very point it was told to mine (100% precanon-dup, zero canonical renders —
observed 0 admits vs ~250 with the prior off). **Resolution: run dives with the prior
OFF** (dedup against their own accruing cloud only). Cross-era re-mints are acceptable —
emission intake's own coord+CLIP dedup collapses them downstream at library assembly.

## 3. Deficit scheduler — budget in distinct-looks, priced per look

The family-level deficit scheduler allocates cross-partition discovery budget
denominated in **distinct looks against the release-mix ratio table** (an order book —
`release_mix.shares` over the run's tracked partitions), replacing a single global `p_good`
queue whose un-calibrated cross-family comparison drove the mix.

- Each partition's **price = active-minutes per distinct look** (online EMA, seeded
  neutral).
- The pop decision is a **pure function of per-partition deficits and prices only** — no
  `p_good` / score / node term. The preference ranker **never enters scheduling**
  (consistent with `aesthetic_scoring.md`: `p_good` is not a cross-family goodness).

**Julia routing (the one non-obvious mechanism).** A `julia:X` partition **cannot be
popped into existence** — it is fed only by descending c-plane `X` and firing a hook on
a qualifying parent. So when a julia partition has positive deficit but an empty own
queue, its deficit is **folded into its c-plane parent's effective deficit**; serving
the parent fires the hook and seeds julia roots that later compete directly (no
double-count once the twin has a queue). This deficit-fold was chosen over a dedicated
julia budget or twin-price-proportional spending because it needs **no separate
planner** — the existing price-weighted-deficit pop plus the existing hook do all the
work.

### 3.1 Root weights cannot enforce a cross-partition mix — measured

A lighter alternative to the scheduler was tried and **does not work**, and the reason
generalises past the one flag. `steered_frontier --family-weights` sizes the per-family
**root draw** from a deficit computed before launch (there: each partition's class-4 count
gap against a uniform target). Two things then defeat it, and neither is a draw:

* **The julia hook.** Every admitted c-plane parent fires a `julia:X` root (§3's deficit
  fold), so serving a native partition *manufactures* z-plane supply. Native admissions
  become julia candidates at a rate the weights never see.
* **Injected seed pools.** `--julia-seed-pool` / `--phoenix-seed-pool` entered the frontier
  **wholesale at fresh start** — 534 + 96 roots against the ~128 a replenishment draws — and
  the frontier is popped by GLOBAL PRIORITY with every root at `NEUTRAL_PRIOR + gumbel`. 630
  injected roots simply outnumber the native ones, permanently.

`[measured: data/discovery/q4_long_harvest_20260803, 2026-08-03]` Intended native
multibrot3/4/5 share **70%**; at batch 14 the realized candidate stream was julia:mandelbrot
1,500 / phoenix 314 / mb3 44 / mb4 26 / mb5 19, i.e. **5%**. Metering the pools
(`--seed-pool-rate`, N entries per BATCH from a persisted cursor) moved it to **25.6%**, and
over the full 149-batch run it settled at **19.6%** of 17,669 candidates — still a third of
target, with julia:mandelbrot at 39.9%.

**The lever that would work is a POP quota, not a draw weight**: the mix is decided where the
batch is popped, and anything that only changes what enters the frontier is diluted by
whatever multiplies fastest inside it. That is precisely what the §3 deficit scheduler's
price-weighted pop does, which is the argument for using it rather than the light flag when
the mix actually matters.

### 3.2 The pop QUOTA — steering a mix is not enforcing one (harvest v2)

`tools/atlas/pop_quota.py`, `--pop-quota`. Supersedes §3's scheduler as the live allocator;
the two are mutually exclusive by construction (two owners of the pop is two mixes and no
readable number).

§3.1's conclusion is necessary but not sufficient. The §3 scheduler already pops rather than
draws, and it still only **steers**: `choose_partition` is a per-batch stochastic argmax on
price-weighted deficit, so it never MEASURES the mix it produces. A stale price, a partition
that happens to expand cheaply, or an unrepresentative early stretch all move the realized
share with nothing to pull it back. The quota closes that loop — it computes an intended share
vector, tracks the realized share of active minutes, and serves whichever servable partition
is furthest below its intent. Deterministic, because a quota allowed to be lucky is not
evidence about the allocator.

Three things changed with it, each a decision rather than a port:

- **Currency.** Deficits are denominated in **human labels** — `count(label==4) + 0.1 ×
  count(label==3)`, through the amendment overlay + library. §3's distinct-look denomination
  measured variety; this measures the thing the corpus is short of. No machine score touches
  the deficit: a q3/q4 count measures the classifier, not the family, until a human looks.
- **Target: RATIO-WEIGHTED** (Matt, 2026-08-04; was uniform until then). The intended release
  mix is one table — `tools/scoring/release_mix.RATIO`, keyed off `ALL_FAMS` with an
  import-time completeness assertion in both directions — and `target_p ∝ ratio_p`, anchored so
  the maximum-ratio partitions keep the level the uniform rule used (the richest holding). The
  uniform rule said a pinned single-parameter-point plane and the mandelbrot c-plane are owed
  the same number of labels; at 3 : 1 : 0.2 they are not. Consequences of the flip, measured on
  the 2026-08-04 census: `mandelbrot` and `julia:mandelbrot` (ratio 3) keep their deficits
  exactly; every ratio-1 partition's target falls to a third of the anchor, and
  `phoenix:classic` to a fifteenth — which drops its share from **13.45% to the 5% floor**, and
  leaves `phoenix` at its target with zero deficit for the first time. **Emission reads the same
  table since 2026-08-04**: `cells.TargetMeasure.from_partition_shares` re-solves the shares
  against the live feasible cells (`weight_p = share_p / n_cells_p`), and the deficit
  scheduler's order book (`deficit_scheduler.target_shares`) takes them directly rather than
  projecting a measure file down. The hand-placed `data/emission/target_measure.json` and its
  `weight_overrides` / `target_share` machinery are retired (`retired.md`).
- **Price.** Measured **active-minutes per currency unit mined**, credited only on a DISTINCT
  ADMISSION (a q3_dup adds nothing to the corpus the deficit counts against, and pricing dups
  as production would make the churniest partition look cheapest). The classifier does reach
  the price here, so it is **clamped** to a bounded band around its seed — a head that
  over-calls 4s in one family can move that family's service by at most a factor.
- **A universal floor.** Every partition, including the currency-rich ones, receives ~5% of
  TOTAL time; the remainder is deficit-allocated. Implemented as floor-constrained
  proportional water-filling, which is what makes both halves true at once — a zero-deficit
  partition still gets its 5%, and a partition already allocated above it gets nothing extra.
  (The naive "reserve n×floor, split the rest" form fails the second half, and is
  algebraically identical to the correct one whenever the unfloored deficits are equal.) At
  nine partitions the floor's claim is bounded by 45% and its reachable maximum is 40%
  (8 pinned). Rationale, recorded in every run config: spending 100% of the time on a stubborn
  deficit partition means never learning anything new about the rich ones — the floor keeps
  per-partition prices fresh, keeps rich-type material flowing to emission's diversity
  targets, and keeps the cross-feed alive, since rich-base admissions are what trigger
  maneuvers and julia hooks into the deficit partitions.

Julia routing is inherited unchanged: a `julia:X` with intent but no queue folds its share
into its c-plane parent's EFFECTIVE intent, while the realized-vs-intended report is scored
against the ORIGINAL vector — grading a run on a target it moved is not a grade.

**Realized-vs-intended is the headline metric**, reported per partition in three denominations
(minutes, candidates, admissions) and summarised as a total-variation gap. Candidates is there
because §3.1's 19.6% was quoted in it.
`[code: tools/atlas/pop_quota.py; tools/atlas/test_pop_quota.py]`

**Metering is worth keeping regardless**, for a second reason: a pool consumed as the walk
asks for roots is `julia_c_sourcing.md`'s "run to the knee, then refill" by construction,
where a pool dumped at t=0 is run straight into its tail. `--seed-pool-rate 0` restores
wholesale injection and is byte-identical to every run before 2026-08-03.

### 3.3 The price was never seeded from a measurement — and the supply loop, first read

`[measured: data/discovery/steady_state_v1_20260805, 60.0 active min / 76 batches, 2026-08-05]`
`[cmd: uv run python tools/atlas/harvest_v2_readout.py --run-dir <run>]`
The first steady-state run after the v10 flip and the ratio-target / precanon-K adoptions.
Three things it settles, and one it deliberately does not.

**The stored cost-to-mine table was a flat 3.0.** `--quota-prices` had no default and no file
had ever been written for it, so `CostToMine` seeded every partition at `SEED_PRICE` and the
first pops allocated on deficit alone — while `harvest_v2_readout.cost_to_mine` reported "the
prices this run measured, for the next one" into nothing. (`data/atlas/scheduler_prices.json`
is not that table: it is §3's, denominated per DISTINCT LOOK.) Regenerated as
`data/atlas/quota_prices_v1.json` by `tools/atlas/derive_quota_prices.py`, pooling
`sum(min_spent)/sum(units_mined)` — the aggregate the in-run EMA estimates, not the EMA. All
nine partitions cleared the evidence floor (`units >= CLASS_WEIGHT[4]`); **every one measured
cheaper than the flat seed**, from `julia:multibrot3` 0.078 to `multibrot5` 2.51 min/unit —
a 32× spread the seed asserted was 1×. There is deliberately **no magnitude bound** on a
derived seed: an early `[seed/4, seed*4]` band would have reported 0.75 for the four
partitions measured at 0.078–0.139 off 62–148 units, i.e. thrown away the finding. The guard
is on the denominator instead. `[code: tools/atlas/test_derive_quota_prices.py]`

**That flatness is visible in the run's own miss.** The allocation is recomputed every pop
from live prices, so a flat seed guarantees the intent moves as prices are learned:
`multibrot4` 0.190 → 0.050 and `julia:multibrot3` 0.136 → 0.225 inside 76 batches, purely on
price. L1 mix gap 0.203 vs launch intent, 0.133 vs effective. Crucially the mean
effective-intent mass on EMPTY queues was **0.000** (arm B: 0.457), so unlike arm B this miss
is **tracking error, not starvation** — the pop was choosing badly among partitions it could
all serve, which is the failure a seeded price table addresses and a supply fix does not.

**The seed that production actually runs is the REGULARIZED one, not the measured one**
(Matt, 2026-08-05). The measured prices come from a single 60-minute run in which every
partition was warming up at once, so the levels are likely deflated and there is no reason to
think they are deflated evenly — and allocation share is `deficit / price`, so seeding the raw
table would hand the first pops a 32× spread asserted on one warm-up hour, the mirror of the
flat 3.0 asserting 1× on nothing. `tools/atlas/regularize_quota_prices.py` shrinks each
measured price geometrically toward the measured **median** (0.139 min/unit), `seed_p =
exp(α·ln p + (1−α)·ln median)` at **α = 0.7**, writing `data/atlas/quota_prices_regularized_v1
.json`; the measured table is read-only evidence and is never rewritten. Shrinkage rather than
a magnitude band for the reason the band was rejected above: a clamp reports the bound and
discards the measurement, where shrinkage keeps the order and α of every log-distance, so the
table's spread lands at `S^α` — **32.2× → 11.4×** (`multibrot5` 2.51 → 1.05, `julia:multibrot3`
0.078 → 0.093). It applies to the **seed only**: `CostToMine` reads `prices` once into
`self.seed`, and the in-run batch-aggregated EMA still moves off it within a few windows.
That artifact is now the **default** for `--quota-prices`, and its absence is **fatal** —
`load_quota_prices` refuses to fall back to the flat seed, because a flat seed is a different
allocation policy rather than a degraded one and it left no trace in any run record. The run
config stamps `pop_quota.seed_price_table` so the three policies (measured / regularized /
flat) are distinguishable afterwards, which as nine bare floats they are not.
`[code: tools/atlas/test_regularize_quota_prices.py]`

**The julia twins are open.** Twin-queue non-empty fractions jm3/jm4/jm5 = **44.7 / 67.1 /
39.5%**, against arm B's 8.1 / 2.7 / 2.4% — the hook-spacing reconciliation (0.20 → the 0.032
pool floor) did what its commit claimed.

**What this run does NOT test: the starvation the per-partition refill exists to fix.** The
refill fired **once**, at b0, on all four c-plane families (118 roots, 42 s), with 0 deferrals
and `root_draw_share` **0.0114** against its 0.25 cap; `wall_over_active` 1.02, pre-loop draw
8.39 min. So both bounds are armed and neither came near binding — but arm B's collapse
appeared at **b381**, and this run stopped at b76. The mechanism is exercised; it is not
evidence about the regime it was written for. `[code: tools/atlas/steered_frontier.py::refill_starved]`

### 3.4 The floor was allocated and never popped — and the reseed off run 2

`[measured: data/discovery/steady_state_v2_20260807, 356.7 active min / 361 batches, 2026-08-07]`
`[cmd: uv run python tools/atlas/harvest_v2_readout.py --run-dir <run>]`

**A floored partition can be allocated its 5% in every batch and served in none of them.**
`julia:mandelbrot` held effective intent 0.05 for all 361 batches against a queue pinned full
at 209 nodes and took **zero** pops; `mandelbrot` took 1 pop / 1.54 min and `phoenix` 1 pop /
0.37 min against a 17.8-minute floor each. All three starved partitions were the floored ones
and no deficit-driven partition starved, which is what locates the defect in the floor rather
than in the price or the allocation. Unlike arm B this is **not** empty queues — all three
were servable in every batch.

The mechanism is `choose_partition`'s gap, `intended_p − realized_p/total`. Since
`realized_p ≥ 0` that gap is **bounded above by `intended_p`**, so a floored partition's claim
is 0.05 at batch 1 and 0.05 at batch 361 no matter how long it starves — nothing in the rule
grows with time-unserved, and an unspent entitlement is re-offered rather than accumulated.
Meanwhile the effective vector is re-derived every pop and the julia fold keeps swinging the
twins' large intent onto whichever c-plane parent is momentarily unservable, so some
competitor presented a gap above 0.05 in **359 of 361 batches** (median best-competitor gap
0.126, minimum 0.042). In the only two where none did — batches 30 and 31 — the three floored
partitions tied at exactly 0.05 and the `(gap, p)` tie-break, max by NAME, gave those pops to
`phoenix` and `mandelbrot`; `julia:mandelbrot` sorts first of the three and got neither.

**The fix carries the claim** (`pop_quota.FloorLedger`). Each charged batch accrues
`floor × minutes` to every partition that *could* have been popped for it and spends what the
served one took; the debt is `max(0, entitled − realized)` and preempts the gap rule once it
reaches one mean batch (`total_min / pops`, read from the run's own telemetry). The debt grows
without bound while a batch's cost does not, so a floored partition is served **regardless of
its per-pop cost** — deficit round-robin's fairness bound, in minutes. The bound is exact and
cost-free: `debt = floor·T` and `trigger = T/pops` both scale with the same `T`, so a
partition servable throughout comes due at `pops ≥ 1/floor` — **batch 20 at the 5% floor**,
whatever a batch costs. Anything later means it was unservable, capped, or the floor is not
what it says. Denomination is what
keeps it a floor: a cheap partition triggers more often and repays less each time, so it takes
more POPS and the same ~5% of the CLOCK (measured on the reproduction: 20.7% of pops, 4.97% of
minutes). Entitlement accrues over **servable** minutes only, so a partition nobody could feed
does not bank arrears and spend them in a burst when its queue refills.
`[code: tools/atlas/test_pop_quota.py §4b — the carry-off arm starves forever, as the control]`

**And the failure is now un-missable.** `summary.pop_quota.unspent_floor` names any partition
that spent ≤10% of the floor minutes allocated to it, with `servable_min` beside the spend so
"the rule declined to serve it" and "nothing could feed it" are separable without opening the
trace; the alarm is lifted to a top-level `UNSPENT_FLOOR_PARTITIONS` key and printed loud by
both the run's own readout and `harvest_v2_readout`. The quota trace additionally stamps
`via` (`gap` / `floor_carry`) per pop, so a floor being HELD is distinguishable from one that
is merely never tested.

**The price table is reseeded off run 2, and the clamp was the error source.** Run 2 is 357
active minutes against run 1's 60 and mines 10–60× the currency per partition, so the α = 0.7
shrink sized for "one warm-up hour" now removes more than it should: the reseed is **α = 0.9**
(`data/atlas/quota_prices_20260807.json` → `..._regularized_20260807.json`, the new
`--quota-prices` default; the run-1 pair stays as the record of what run 2 itself ran on).
The sharper change is the **live-EMA clamp, 4× → 16×**: run 2 finished with three of nine
partitions pinned at the band edge — `price_raw/seed` was **15.6× (multibrot4), 18.1×
(multibrot3), 5.0× (julia:multibrot3)**, every one reported at 4.0×. A run whose own EMA is
quoting the bound for a third of its partitions is not measuring them, which is the objection
`derive_quota_prices` already raises against a magnitude band on the seed. The clamp is
therefore written by the regularizer rather than inherited from the measured table: it is a
band around *this* seed, so it belongs to the same decision as α. **Two rows are `defaulted`
at `SEED_PRICE` 3.0** — `julia:mandelbrot` (0 units) and `mandelbrot` (0.3 units, below the
`CLASS_WEIGHT[4]` evidence floor) — because the starvation above is exactly what stopped them
being measured. Neither moves an allocation today (both are floor-bound: deficits 0.0 and 3.8
against price-weighted deficits two orders larger elsewhere), and the carry now guarantees
they are served and priced next run.

**Regenerated 2026-08-12 at `price_ema` 0.15 — a table pins its own rate, and production reads
the table.** `pop_quota.PRICE_EMA` was halved 0.30 → 0.15 when the cost estimator moved to
stepping once per SERVED BATCH (2–3× as often, so the halving preserves the intended memory),
but the deployed seed carried `price_ema: 0.3` and would have kept handing every run the rate
the code had already left. Both tables were re-derived from the same `steady_state_v2_20260807`
telemetry at the same α = 0.9 and 16× band — `..._20260812.json` /
`..._regularized_20260812.json`, now the `--quota-prices` default — and **all nine seeds are
byte-identical to the 20260807 pair**, which is the point: it is a one-variable change, and the
two pairs differ only by the code state each was derived under. The regression gate is
`test_the_deployed_table_pins_the_LIVE_ema_rate` — moving the constant again obliges a
regeneration rather than a silent divergence.

**Reseeded off run 27 (2026-08-12) — the two defaulted rows are cleared, and every other seed
moves with them.** `prod27_20260812` is the first source run to price `mandelbrot` and
`julia:mandelbrot` (84.2 and 54.9 units, against run 2's 0.3 and 0.0), which is precisely what a
`defaulted` row waits for: they land at **0.165 and 0.190 min/unit, i.e. 18.2× and 15.8× under
the flat 3.0 the table had been asserting** for the partition whose lockout run 27 was launched
to test. `..._20260812_run27.json` / `..._regularized_20260812_run27.json` is now the
`--quota-prices` default and has **no defaulted row**; the run-2 (`..._20260812`) pair stays as
the previous rung. Two things that did NOT hold and are worth stating, because a reseed off new
telemetry is not the one-variable regeneration above: the other **seven seeds move as well**
(0.58×–1.78× of the run-2 values; the three cheap julias reproduce to 0.94–1.19×, `multibrot5`
moves most on 17.3 units), and the **shrink target moves with the population**, median 0.308 →
0.190 over 9 measured rows instead of 7. α = 0.9 and the 16× band are unchanged, and the
measured spread contracts 11.7× → 9.2×. The source is run 27 **alone, not pooled with run 2**:
τ_h was enlarged on 2026-08-08 (`mandelbrot` 0.023 → 0.315) and `units_mined` counts admissions
past τ_h, so the two runs' denominators are not the same quantity.
`[measured: data/discovery/prod27_20260812, 179 min budget / 236 batches, 2026-08-12]`
`[code: tools/atlas/test_regularize_quota_prices.py; tools/atlas/test_derive_quota_prices.py]`

## 4. τ_h on record — the real per-partition curve

`τ_h` is the **per-partition cheap-`p_good` harvest cut**: cheap score ≥ `τ_h` → one
canonical confirmation render → score. It is a **fixed offline constant**
(`derive_tau_h`, keep=0.90 = the 10th percentile of cheap `p_good` among frames whose
*canonical* `p_good` clears **`floors.GOOD_FLOOR`** — the family's per-partition `t_good`
until 2026-08-09, when the whole table retired), **not** learned per run —
identical across campaign 1 and 2. The question was whether the campaign harvests let us
replace that guessed constant with an **empirical curve** of the real tradeoff: raise
`τ_h` → canonical renders saved (cost) vs q3 admissions lost (benefit).

**Data.** The needed join `(partition, cheap_pgood, canonical fate)` per harvest check —
including the `canon-not-q3` / `precanon-dup` **rejects** — was logged to
`harvest_log.jsonl` (`steered_frontier._log_harvest`, one row per check). It was gitignored
as "regenerable telemetry" and lost from the live repo, but **recovered from a filesystem
backup of the old working tree** (a backup keeps gitignored files) and is now stored
durably **in-tree via LFS** beside each run's ledger (`.gitattributes`;
`data/discovery/**/harvest_log.jsonl`). Rows carry the raw confirmation scores
(`canon_nb`/`canon_pgood`), so fate is recomputable. Two gates pass in
`tools/atlas/tau_h_retained_readout.py`: **reconciliation** (harvest_log admitted ties to
each summary — 314/254/311/271) and **threshold era** (the confirmation decode recomputed
under current `t_good` equals the recorded `canon_decoded` — campaign 1/2 already ran at
today's `t_good`, so this is a no-op that *proves* the era matches). **That readout was
deleted on 2026-08-09** with the τ_h harvest arm it belonged to: τ_h is now derived from the
untruncated walk-outcome ledger alone, so the harvest logs are no longer an input to it and
the era gate has nothing to gate. The curve read below stands as the measurement it was.
Campaign 2 breadth is
**segmented at the batch-1211 resume** where julia hook spacing changed 0.2 → 0.1
(`julia_hooks.jsonl`: 0.2 for batches ≤1204, 0.1 for ≥1260), which shifts the julia
candidate population; the two segments are never pooled.

**Which logs feed a re-derivation is DISCOVERED, not listed** (`tools/atlas/
harvest_log_registry.py`, 2026-08-05). `tau_h_rederive` used a five-entry hand list; a run
enters now by writing its log under a registered store, which
`discovery_sinks.resolve_discovery_dir` guarantees for every non-throwaway run (a
`--smoke` run is redirected to `scratch/` and refused by class). The five are *pinned* —
their absence is a hard failure, since they are the population `tau_h_base_v10.json` was
derived on — and everything else is found. Discovery on 2026-08-05: **18 run dirs, 5
pinned + 13 discovered**; after the two row-level exclusions (pre-geometry rows, which is
what keeps all of campaign1 out *by construction*, and phoenix, which no arm cuts) the
harvest pool is **56,461 rows against the adopted derivation's 29,011**. So the next
re-derivation is over a different population and will not reproduce the adopted arms —
see `arms_used` / `harvest_registry` in the artifact before comparing versions.

**Curve** (`scratch/tau_h/curve.json`; both axes, per partition, per run/segment). A canonical
render happens iff `cheap ≥ τ` **and** the check is not a pre-canonical coord-dup
(campaign 2's `precanon_dup` filter already skips ~82% of checks before rendering; campaign
1 has no such filter so every check renders). Steps are distribution-adapted (τ at the
p25/p50/p75 of *rendered-check* `cheap_pgood`, so each saves ~25/50/75% of renders).
`admits_retained` is a **first-order lower bound** (raising τ only shrinks the dedup cloud →
some `q3_dup`s would promote to distinct). Denomination is **raw admissions**; per-reject
distinct-look attribution is not recoverable (distinct-looks are tallied only on
admissions) — so it is not reported.

**The exchange rate — admissions lost per canonical render saved — is sharply
partition-dependent** (campaign 2 breadth, seg-B / current 0.1-spacing regime):

| partition | save 25% renders → keep admits (lost/saved) | save 50% renders → keep admits (lost/saved) |
|---|---|---|
| mandelbrot | 100% (**0.00**) | 96% (**0.005**) |
| multibrot3 | 95% (0.023) | 68% (0.071) |
| multibrot5 | 87% (0.063) | 70% (0.071) |
| multibrot4 † | 77% (0.71) | 55% (0.71) |
| julia:mandelbrot † | 75% (0.31) | 62% (0.23) |
| julia:multibrot3 † | 64% (0.63) | 43% (0.53) |
| julia:multibrot4 † | 67% (0.40) | 50% (0.33) |
| julia:multibrot5 † | 75% (0.36) | 50% (0.36) |

> **† Band-thin — read the multibrot4/julia rows as directional, not precise.** A read-only
> re-audit of the harvest logs (no re-render) found their high exchange rate is *not* a broken
> cheap score but the **base admit rate showing through a locally flat cheap→fate relationship**,
> on a tiny sample. Two things stack: (1) after `precanon_dup` these partitions render only a
> sliver (seg-B rendered checks: mb4 **29**, julia:mandelbrot **53**, julia:mb3 **30**, julia:mb4
> **19**, julia:mb5 **44**; the q=0.25 cut band is **5–13 rows**), so each rate is a handful of
> admissions; (2) within that rendered slice the cheap score barely ranks fate — bottom-quartile
> and top-quartile admit rates are ~equal — so **exchange ≈ base admit rate** (mb4 ≈0.7, julia
> ≈0.3–0.48), *not* the "several-times-the-average" inversion the raw number suggests. mb4's
> flatness is **restriction-of-range from its very high τ_h=0.774** and persists in campaign 1
> (which has no precanon filter at all), so it is a real, benign margin effect, not the confound.
> Julia's is compounded by **84–97% precanon depletion**, which for julia:mandelbrot/mb3 skims off
> the *higher*-cheap candidates (killed rows average +0.04 / +0.11 cheap vs survivors), leaving the
> low-cheap survivors admission-dense — the confound the audit was checking for, but modest and
> not uniform in sign (julia:mb4's killed rows skew *lower*-cheap). The §4 verdict is unchanged —
> still **no cuttable reject cluster** in mb4/julia — but do not over-trust the individual cells.
> The unmarked **c-plane mandelbrot/mb3/mb5** rows are the trustworthy signal: rendered 170–415
> (not thin) and the cheap score strongly discriminates (bottom-quartile admit 0.00–0.06 vs
> top-quartile 0.18–0.29), which is *why* their exchange rate is genuinely low.

So the **headroom is concentrated in low-degree c-plane** — above all **mandelbrot**, where
half the confirmation renders can be cut for ~4% admission loss (≈0.005 admits per render),
and secondarily multibrot3/5 (cut ~25% for ≤6% loss). **multibrot4 and every julia
partition have essentially no headroom** — raising `τ_h` there sheds admissions at ~0.3–0.7
per render saved, i.e. near 1:1. This directly answers the campaign-1/2 open question (are
the wasted `canon-not-q3` renders clustered just above `τ_h`, cheaply cuttable?): **yes for
c-plane mandelbrot/mb3/mb5, no for mb4 and julia.** The campaign-1 curve (no precanon
filter, so larger absolute render mass) agrees on the ranking and gives lower c-plane
exchange rates still (mandelbrot 0.004–0.006 at 25–50%). Full arrays incl. campaign 1,
campaign 2 dive, and seg-A in `scratch/tau_h/curve.json`.

**Retention fix.** The logging was already correct and unconditional in production
(`_log_harvest`, pure post-decision append). The only defect was durability:
`harvest_log.jsonl` is now un-gitignored and the recovered campaign 1/2 (and shakeout /
steered) logs are committed via LFS, so the curve is on record and future runs retain it.
*Other* gitignored reject-class telemetry still survives only in the backup —
`prio_terms.jsonl` (per pushed candidate incl. the never-admitted majority, 57 MB),
`julia_hooks.jsonl`, `saturation.jsonl` — inventoried but not recovered here.

### The committed record is SEGMENTED (2026-08-07)

Retaining all of it made the record large: measured on `steady_state_v1_20260805` (70 active
min) the committed run dir is **10.30 MB**, i.e. **8.9 MB/h**, and `harvest_v2_proving_20260803`
ran at **14.2 MB/h** — so an 8 h run lands at 50–70 MB and anything past ~2.5 h crosses
CLAUDE.md's 20 MB commit rule. Every figure in this section is **TREE BYTES** — the
working-tree size of what gets tracked, which is the unit that rule counts (settled
2026-08-07) and the one under which these numbers are a crossing at all. None of it is
regenerable and none of it may be thinned, so the
five per-row streams are **rotated into gzipped segments** instead
(`tools/run_record.py`, `SEGMENTED_STREAMS` = harvest_log / prio_terms / maneuvers /
q4_candidates / quota_trace):

```
<run>/harvest_log.000.jsonl.gz   segment 0, closed and compressed (LFS)
<run>/harvest_log.001.jsonl.gz   ...
<run>/harvest_log.jsonl          the LIVE tail — plain, <= 4 MiB, absent once the run finishes
```

Rows are read segments-then-tail, i.e. write order, by `run_record.iter_rows` /
`read_rows` — **which is how every consumer must read them**: a finished run has no plain
`.jsonl` at all, so `open(run/"harvest_log.jsonl")` sees nothing, and a `glob` keyed on that
exact name stops discovering runs the moment they complete (`harvest_log_registry.LOG_GLOB`).
`tools/test_run_record.py` fails on any tracked module that reads one directly. A run dir
written before this change has no segments and reads unchanged.

**Why compression and not field-dropping.** On TREE bytes — the unit the rule counts — the
7–11× is simply the win, for all five streams, because the tree holds the compressed file.
On *remote/pack* bytes it is not: git zlib-compresses an ordinary blob already, so committing
`maneuvers.jsonl` raw packs to 451,089 B and committing the same rows as `.gz` packs to
451,314 B (measured 2026-08-07, `git gc --aggressive`, both ways), and the only reason the
remote shrinks too is that the four big streams are **LFS-tracked** and LFS ships the object
byte-for-byte. So the tree-byte framing is what the change is *for*; the LFS path is why it
also helps the remote. Field-dropping was
measured on the same file as the alternative and is not competitive: `atom_key` −5.7% of
compressed bytes, `screen.interior_radial` −7.4%, every null −0.9%, each paying real
information. Result: the same five runs go from 12–114 MB to **1.6–17.0 MB projected at 8 h**
(worst observed rate: `steady_state_v1_20260805`, 17.0 MB tree / 12.4 MB LFS+pack).

## 5. Cross-run saturation memory — the breadth leg stops re-mining what earlier runs mined

Every run before 2026-08-09 started with **no memory of where its predecessors went**. The
dup cloud starts empty (the freshness prior is off by default, and §2 is why it must stay off
for dives), so a basin three runs already walked ranked exactly like untouched territory and
the breadth leg re-bought it. The fix is a **soft, scale-aware discount on the breadth
candidate's steering weight**, and it is deliberately not a new store:

```
cheap_eord  *=  1 / (1 + SAT_STRENGTH * density)
density      =  # prior ledger visits v with dist((cx,cy), v) <= SAT_RADIUS_K * v.fw,
                in the same partition AND on the same dynamical plane
```

`tools/atlas/visited_density.py` owns the index; `steered_frontier.SAT_RADIUS_K` (0.30) and
`SAT_STRENGTH` (1.0) own the knobs, both on flags, both stamped into `run_config.json`.

**The memory IS the ledgers.** `data/**/outcome_ledger.jsonl` already records, durably and per
run, every place a run confirmed — 15,156 rows over 33 ledgers as of 2026-08-09. Loading them
at run start costs **0.36 s** and the query costs **≤57 µs per candidate** (measured
2026-08-09 on the full store; a 32-node batch pays ~1.8 ms against a ~30 s batch), so nothing
justifies a second store that could drift from the one that already exists. The index is built
once, EXCLUDES this run's own ledger, and is never mutated — the current run's coverage is
what the dup cloud and the morph memory are for, and a frozen index is what makes a resume
rebuild the identical memory. `visited_density.iter_prior_ledger_rows` is the single owner of
that enumeration, shared with the freshness prior.

**No quality filter and no dedup, unlike `build_cloud`.** A place that was checked and
rejected was still visited, and `is_good` is a cut on a stored probability whose meaning moves
with the active head — a quality-filtered cross-run memory would silently re-shape itself at
every classifier flip. **No decay** either: regions do not un-exhaust, and the answer to "that
was a while ago" is the dive channel and stage-2 recolor, not a half-life.

**Scale-aware on the VISIT's frame, not the candidate's** — the difference from `near_dup`,
whose radius is `DEDUP_K * min(a_fw, b_fw)`. A deep confirmation shadows almost nothing; a
base-scale one shadows a neighbourhood. One run passing through a wide frame does not exhaust
what is inside it, but a hundred deep confirmations in one basin do exhaust that basin.

**Identity-aware, and this is load-bearing.** Inside a julia or phoenix partition the
coordinate is a **z-plane** point, so two views at the same z with different seed parameters
are different fractals — the "over-kill" collapse `build_cloud`'s `row_ident` gate already
exists to prevent, reached by another route. Measured on the calibration population below: a
z-only index reads julia:mandelbrot as **46.3%** shadowed at k=0.05 where the identity-aware
index reads **9.7%**, and julia:multibrot5 as **8.5%** where the true answer is **0.0%** — it
would discount channels that have never been visited twice, which are exactly the channels the
next production run exists to serve. The identity is `production_seeder.row_ident`, bucketed on
a grid of `JULIA_SAME_C_EPS`; two identities within eps but straddling a bucket edge simply do
not shadow each other, which errs toward LESS discount.

### 5.1 Scope: the breadth leg's ORDER inside one partition, and nothing else

`priority_terms` has exactly one caller, `push_children`, and that is the whole surface:

- **Root draws are exempt** — a root carries `NEUTRAL_PRIOR + gumbel` built inline.
  `draw_roots` could consult the same index cheaply and deliberately does not: the native
  seeder's rejection sampler is already a density gate (`count_within(REJECT_RADIUS) >=
  Q3_DENSITY_CAP`, on the RUN cloud) that was tuned for a cloud starting EMPTY, and feeding it
  cross-run mass is the exact shape of the part-0 sterilization finding — prior-ON gave a
  productive-region seed a median ~12 neighbours inside 0.20 and rejected ~98% of seeds on
  arrival. Root draws stay undiscounted until that sampler is re-tuned as one decision.
- **Maneuver-originated nodes are exempt** — they build their own priority in
  `_consume_maneuver` (neutral prior, or the view/range screen percentile) and hold a reserved
  quota. A ranker cannot evaluate a population it was never trained on, and a slot is not a
  probability (`measurement_practice.md` §2). **The exemption is on the PROPOSAL, not on its
  subtree**: a maneuver node's ordinary DESCENDANTS are found by ordinary descent, carry a
  `cheap_eord` and compete in the ordinary queue, so they are discounted like any other breadth
  candidate. Exempting a lineage would need a flag threaded down it, and would make a saturated
  basin permanently cheap to re-enter through one operator.
- **Dive mode is exempt** — `run_dive` never reaches `push_children`, so the index is not built
  at all there rather than built and unread (`run_config` stamps `n_a`).
- **Cross-partition allocation is untouched.** `pop_batch_quota` / `pop_batch_scheduled` pick
  the partition from QUEUE LENGTHS and only then sort that partition's nodes by priority, so a
  per-candidate discount cannot move which partition is served. Pop-quota targets are computed
  exactly as before.

**Multiplicative on `eord`, never a subtracted penalty.** `eord` is the head's E[ord] in
[0, K−1] and the only non-negative term in the priority, so scaling it demotes a candidate
towards — never below — what an unscored root ranks at, and the Gumbel/depth terms still
separate the survivors. A saturated place with a great score loses to a fresh place with a
merely good one and still beats a fresh place with a bad one; a partition whose entire frontier
is saturated keeps picking its best candidate rather than stalling. A subtraction would be
unbounded below, i.e. "saturated" would eventually mean "unreachable".

### 5.2 Where `SAT_RADIUS_K = 0.30` comes from

The failure mode is not "too big" in the abstract — it is the one the **morph-novelty term**
already demonstrated. Re-derived from `steered_run2/prio_terms.jsonl` (38,419 pushed
candidates): **99.58%** carried a nonzero novelty penalty and **92.88%** sat within 10% of the
FULL penalty. A term that fires at full strength on 93% of the population subtracts a
near-constant from every priority and reorders nothing. So the bar is stated against that
number:

> **saturated share** = fraction of candidates whose discount is within 10% of full
> (`density >= 9` at strength 1.0). Adopt the largest k whose saturated share stays
> **≤ 5% pooled and ≤ 10% in every partition**.

Both bars, because a pooled share hides a partition. `tools/atlas/sat_radius_calibrate.py`
measures it over **46,798** committed `q4_candidates` rows from 8 runs — the widest committed
record of breadth candidates carrying coordinates *and* the dynamical parameter, which is why
this cannot be measured on `prio_terms` (it carries neither) — leave-one-run-out against the
ledger union, later runs included: the estimand is what the NEXT run will face.

| k | discounted | saturated (pooled) | worst partition |
|---|---|---|---|
| 0.25 | 26.6% | 0.96% | phoenix 6.1% |
| **0.30** | **31.1%** | **1.39%** | **phoenix 7.8%** |
| 0.32 | 33.0% | 1.61% | phoenix 8.7% |
| 0.35 | 35.5% | 2.61% | phoenix 14.4% ✗ |
| 1.00 | 59.8% | 10.9% | phoenix 36.7% ✗ |

*[measured: 46,798 q4_candidates rows against 15,156 ledger rows, 2026-08-09,*
*`uv run python tools/atlas/sat_radius_calibrate.py`]*

The grid crosses the per-partition bar between 0.32 and 0.35; **0.30 is the round value inside
it with margin**, not the grid maximum. Phoenix is the binding partition and honestly so: 954
of its 1,933 ledger rows sit on the ONE classic Ushiki plane (930 of them pre-date the phoenix
parameter axes and resolve to it through `row_phoenix_key`'s Ushiki defaults), which really is
the most-visited surface in the store. The c-plane natives are the other extreme —
multibrot4 reads 0.1% discounted at k=0.05 and 19.8% at k=0.25, because four runs barely dent
a whole parameter plane.

### 5.3 Telemetry

Per candidate: `prio_terms.jsonl` gains `sat_density` and `sat_disc` beside the other terms.
Per batch and per partition: the existing `saturation.jsonl` row gains a `visited` block
(`n` / `discounted` / `frac` / `by_partition`) — one file, because both halves answer the same
question, "is a soft steering term firing on everything?". **The morph half of that row is now
`null` rather than `0` when `lambda_m == 0`**: a `frac: 0.0` written by a run that never
measured novelty would be averaged in as a real observation by `campaign2_readout` /
`steered_v1_2_dive_report`. Per run: `summary.saturation_memory` carries `status`
(`on` / `off` / `dive_n_a` — a run with the memory off and a run whose memory found nothing
tally identical zeros and are opposite facts), the index census, and the per-partition
discounted share.

### 5.4 First exercise — run 25

The first production run with the memory live. The index built in **0.1 s** over **34 ledgers /
15,189 visits / 466 identity buckets**, and **2,188 of 24,833 scored candidates were discounted =
8.81%**, at a **mean density of 1.3 where it fired** — i.e. the live population sits at the soft
end of the curve, nowhere near §5.2's "saturated" bar of density ≥ 9.

**8.81% against the 31.1% §5.2 projected at k=0.30, and the two are different estimands.** §5.2
is leave-one-run-out over 46,798 *committed* `q4_candidates` rows from 8 runs; this is every
candidate `push_children` saw in one run. Part of the gap is visible as pure mix in the run's own
by-partition block: `multibrot3` alone is 10,243 of the 24,833 scored (13.7% discounted) while
`julia:multibrot3/4/5` contribute **6,999 candidates at 0.0%**, which caps the pooled share on
its own. **Read the calibration table as a bound on a run's discounted share, not a prediction of
it.**

Those three zeros are the **designed negative result**, not a broken index: four runs barely dent
a whole parameter plane, and they are exactly the channels the identity-aware index exists to
protect — a z-only index would have read `julia:multibrot5` as 8.5% shadowed and discounted
material nothing has visited twice. Per partition the memory bites hardest where it should:
`mandelbrot` 18.5%, `julia:mandelbrot` 16.3%, `phoenix` 8.7% at the highest mean density (1.87).

`[measured 2026-08-09; data/discovery/prod25_20260809/summary.json § saturation_memory]`
