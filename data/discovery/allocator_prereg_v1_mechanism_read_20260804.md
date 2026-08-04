> FROZEN COPY, committed 2026-08-04 beside `allocator_prereg_v1.json` because amendment 1
> rests on it. Body verbatim from `scratch/armB_midrun_read_20260804.md` (written 09:55 PDT,
> mid-run, before arm B's terminal numbers existed); only this header was added. The scratch
> original is disposable — this is the copy the amendment cites.

# Arm B mid-run read — mechanism, 2026-08-04 09:55 PDT, batch 381 / 129.5 active min

**Mechanism deviates: the pop decision is conforming better than the proving run, but it is choosing from a servable set that has collapsed to one partition.** Read-only over `data/discovery/popquota_v2_20260804/`; descriptive and unregistered — no `compare_allocator_runs.py`, no pre-registered window. Every number INTERIM.

**1. Mix conformance.** Batch-weighted L1/2 over `quota_trace.jsonl`; the proving run's `effective` reconstructed with `fold_julia_intent` (its trace predates the stamp — the reconstruction reproduces the docstring's 0.091 as 0.093, so the method matches).

| | vs stated | vs effective (folded) | vs effective, renormalized over SERVABLE | effective mass on empty-queue partitions |
|---|---|---|---|---|
| proving, b137 | 0.491 | **0.093** | 0.093 | **0.000** |
| arm B, b381 | 0.495 | **0.427** | **0.041** | **0.457** |

The pop serves what it *can* serve almost exactly as instructed (0.041, tighter than the proving run). The 0.427 is supply, not steering: 46% of post-fold intent points at partitions with an empty queue at pop time. Mean queue length by window shows the collapse — multibrot4/5 at 0 by b201, multibrot3 by b301, mandelbrot and phoenix in the last window. **At b381, 8 of 9 queues are empty and the 944-node frontier is 97% `julia:mandelbrot`** (862). Realized minutes: mandelbrot .183, julia:mandelbrot .194, phoenix .213 against effective intent .05/.06/.08; multibrot5 .110 against .332.

**Nothing will refill them.** In-loop replenishment fires on `len(frontier) < ROOT_LOW_WATER`, and `ROOT_LOW_WATER = B = 32` — a GLOBAL low-water mark cannot see a per-partition starvation inside a healthy frontier, so `draw_roots` has not fired once (wall/active 1.043 vs arm A's 1.15, which is arm A replenishing repeatedly). julia:mandelbrot expansions beget julia:mandelbrot, so it is self-locking.

**2. Floor vs deficit.** 52.9 floor / 72.3 deficit min = **42.3 / 57.7** (proving 36.0 / 64.0); by batch, 131 floor / 237 deficit. Floor share is rising for the same reason as §1 — with the deficit partitions unservable, the floored ones (mandelbrot, julia:mandelbrot) take the pops.

**3. Twin handling — arm A's shape recurs, harder.** Twin queues non-empty in 8.1 / 2.7 / 2.4% of batches (jm3/jm4/jm5). The hook fired 46 times and **manufactured 8 roots** (5/2/1); `julia:mandelbrot` was rejected **28 of 28 times** — the injected 209-c pool at 0.032 spacing saturates the 0.2 hook spacing, so that channel is closed by construction. Twins are bought through parents, but the parents starved too, so after ~b250 the fold buys nothing. Served directly they are the run's cheapest ground: 4.9 min total → 15 of 86 admissions.

**4. Spend sanity — the price signal is inverted on the biggest spenders.** Aggregate `min_spent/units` vs EMA `price`: multibrot4 **14.3 vs 4.69**, multibrot5 11.4 vs 2.48, mandelbrot 1.80 vs 9.24, julia:mandelbrot 3.0 vs 13.54 (**clamped** to the 12.0 ceiling — the only clamp binding). Mechanism: a price sample is dry-minutes ÷ the units of a SINGLE decode, so an isolated class-3 (0.1 unit) samples 10× the gap while decodes clustered in one batch sample ~0. Since share ∝ deficit/price, the two genuinely most expensive partitions are priced cheapest and drew the largest deficit intent. Caps never fired (0 capped batches, no partition reached the 25-min dry bound), so caps explain none of the misses.

**5. Health.** Alive (PID 11624, BelowNormal), no error/warning/reaper/timeout line in `run.log`, `harvest_log.jsonl` still growing (3.93 MB, 09:54), wall cap still the binding one.

**Non-decisional, for completeness:** 88 cumulative admits at b381, 0.68/active-min run-to-date — cold-phase confounded; the registered comparison happens at closeout only.
