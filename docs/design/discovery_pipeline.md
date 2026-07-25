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
denominated in **distinct looks against a target measure** (an order book), replacing a
single global `p_good` queue whose un-calibrated cross-family comparison drove the mix.

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

## 4. τ_h on record — the real per-partition curve

`τ_h` is the **per-partition cheap-`p_good` harvest cut**: cheap score ≥ `τ_h` → one
canonical confirmation render → decode. It is a **fixed offline constant**
(`derive_tau_h`, keep=0.90 = the 10th percentile of cheap `p_good` among fidelity-study
frames whose *canonical* `p_good` clears the family's `t_good`), **not** learned per run —
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
today's `t_good`, so this is a no-op that *proves* the era matches). Campaign 2 breadth is
**segmented at the batch-1211 resume** where julia hook spacing changed 0.2 → 0.1
(`julia_hooks.jsonl`: 0.2 for batches ≤1204, 0.1 for ≥1260), which shifts the julia
candidate population; the two segments are never pooled.

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
