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
