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

## 4. τ_h on record — what the retained store can and cannot say

`τ_h` is the **per-partition cheap-`p_good` harvest cut**: cheap score ≥ `τ_h` → one
canonical render → decode. It is a **fixed offline constant** (`derive_tau_h`, keep=0.90
= the 10th percentile of cheap `p_good` among fidelity-study frames whose *canonical*
`p_good` clears the family's `t_good`), **not** learned per run — identical across
campaign 1 and 2. The retroactive question was whether the campaign harvests let us
replace that guessed constant with an **empirical per-partition curve** of the real
tradeoff: raise `τ_h` → renders saved (cost) vs q3 admissions lost (benefit).

**The join needed is `(partition, cheap_pgood, canonical fate)` per harvest check** —
including the `canon-not-q3` / `precanon-dup` **rejects** (~95% of c-plane checks;
readout §2 fate table). That triple *was* logged: `steered_frontier._log_harvest` →
`harvest_log.jsonl`, one row per check, append-only. **But that file was gitignored as
"regenerable telemetry" and was not retained** — for campaign 1/2 it survives nowhere
(repo, `fractal-maker-artifacts/`, trash). Only `outcome_ledger.jsonl` remains =
**admissions only** (distinct q3, each carrying `cheap_pgood`). So:

- **Cost axis (renders saved = f(τ_h)) is permanently unrecoverable** for campaign 1/2 —
  it needs the reject cheap-scores, which lived only in `harvest_log`. Reconstructing it
  would mean re-running the cheap scorer *and* the canonical render+decode over every
  candidate, i.e. re-deriving the harvest, not reading a retained file.
- **Benefit axis (admissions retained = f(τ_h), for τ_h ≥ current) IS recoverable** from
  the admitted `cheap_pgood` in the ledger. It is a conservative **lower bound**: raising
  `τ_h` only shrinks the greedy dedup cloud, which can only promote later `q3_dup`s to
  distinct → true retention ≥ the naive count.

Retained readout (all 4 ledgers pooled, `tools/atlas/tau_h_retained_readout.py` →
`out/tau_h/retained_readout.json`; per-run admits 311/271/314/254 reconcile with the
summaries): **current `τ_h` sits right at the admitted-cheap floor for every partition.**
The gap between `τ_h` and the *lowest* admitted check's cheap score is +0.0004 to +0.0106
— i.e. there is **essentially no zero-loss headroom**; the offline keep-0.90 constant is
empirically pinned just under the observed admissions. Raising `τ_h` above current starts
shedding admissions immediately (e.g. `julia:multibrot3` retains 0.79 of admissions at
0.35 vs 0.31 current; `mandelbrot` 0.80 at 0.40 vs 0.20). What that trade *buys* in saved
renders is exactly the unrecoverable half — so the interesting question (are the ~95%
canon-not-q3 wasted renders clustered just above `τ_h`, cheaply cuttable?) **cannot be
answered from campaign 1/2 and must wait for a future run.**

**Fix (retention, not instrumentation).** The logging is already correct and unconditional
in the production path (`_log_harvest`, pure post-decision append — zero effect on any
admission). The only defect was durability: `harvest_log.jsonl` is now **un-gitignored and
committed** alongside the ledger (`.gitignore`, `data/discovery/**` block), so future runs
retain the full `τ_h` curve. Campaign 1/2's is lost for good.
