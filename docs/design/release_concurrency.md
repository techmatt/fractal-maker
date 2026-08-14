# Process concurrency on the release render pass

**Owns** `tools/emission/release_pass.py` — the emission driver's last stage, which renders each
selected row once at the wallpaper canon (2560×1440 ss4). It is here rather than in a scratch
readout because two production constants (`DEFAULT_RELEASE_WORKERS`, `ENGINE_THREADS_PER_WORKER`) have no
justification other than the measurement below, and `tools/emission/release_sweep.py` reproduces
it. Companion: `pytest_suite_cost.md` (the other place a worker count is measured rather than
argued).

## 1. Why the pass was worth parallelising at all

After `tail_optimize` (`4b9fe5c`) one full-res release render is roughly half **7-thread Rust
engine** and half **single-thread Python coloring tail**. Neither half can use the box: the
engine's rayon pool is sized for one process with the machine to itself, and the tail is one
thread by construction. So a serial pass leaves most of 12 logical cores idle for most of its
wall clock *regardless of which half is running* — the idleness is structural, not a tuning
miss. Rows are independent, which makes process-level concurrency the cheap mechanism and
in-tail threading the expensive one.

**The tail's own chunk-loop threading, measured in `tail_optimize`, is deliberately NOT also
enabled.** One mechanism per adoption. Whether the hybrid is worth anything is a question about
leftover idle cores at the adopted point, and §4 answers it with a number rather than leaving it
open.

## 2. The structural rule: workers render, the parent writes

A worker returns `(image on disk, info block)` and appends to nothing. Every record — the
auto-level stamp row, the per-unit stage time, the caller's manifest — is written by the parent,
from one `sink` callback, **in plan order**, once per task.

This is what makes the concurrent pass's records *identical* to the serial pass's rather than
merely equivalent. `autolevel_stamps.jsonl` and `stage_times.jsonl` are both read downstream as
ordered streams (`autolevel_read.py`, `run_profile.py`), and an append-only log with N writers
has no order at all. The suppression seam is `stamp_log=False`
(`deploy_tail._level_python` / `render_pure` / `render_rust`), which turns off the stamp **write**
and not the levelling — the stamp still rides back in the info block for the parent to write
through `autolevel.append_stamp`, the one writer both paths go through.

Two smaller consequences of running the same render in N processes, both fixed at the name:

* **`deploy_tail.field_tmp_token()`** — a disposable field dump is written and `finally`-unlinked
  inside one render, and its name is keyed on (location, mode, geometry). Two workers rendering
  the *same* location under two palettes therefore derived one name and would unlink each other's
  file. `render_rust` had already solved this for its own temps by keying on the output stem; the
  field stem cannot take that (identifying the FIELD is its whole job), so the pid rides alongside
  it. Nothing caches at these names, so the token costs nothing.
* **`--release-workers 1` is the untouched serial path** — in-process, no pool, no pickling,
  `stamp_log=True`. It is the fallback, so it is not a special case of the concurrent path; the
  two are checked against each other by a real engine parity test rather than by construction.

## 3. The sweep

Plan: the **12 rows prod27 actually released** — 6 `smooth`, 2 `stripe`, 1 `tia`, 1
`composite_c13_smooth_stripe`, 2 `composite_c17_smooth_curvature`, spanning mandelbrot,
multibrot3/4/5, three julia twins and two phoenix — at 2560×1440 ss4 lanczos3. All three render
paths (Python tail over a dumped field / pure-field / Rust composite) are present because they do
not cost the same and are not RAM-alike: in the serial arm the cheapest row was **21.6 s**
(smooth) and the most expensive **227.1 s** (smooth, phoenix) — a 6.5× spread
within one release.

Every arm renders the same plan into its own directory and is hashed against the serial arm —
PNGs *and* the stamp log.

```
uv run python tools/emission/release_sweep.py --arms 1,2,3,4,3:7 \
    --out scratch/release_concurrency/sweep --check      # 2026-08-13, 32 GB / 12 logical cores
```

Two rounds, because the first one's control arm won. Round 1 varied the worker count with the
box divided among the workers (`RAYON_NUM_THREADS = 12/N`, the "size for the actual N" instinct)
plus one deliberately **oversubscribed** control at 3×7. Round 2 then swept the worker count at
7 threads throughout. Baseline for both is round 1's serial arm, 1045.9 s.

| arm | workers | engine threads | wall s | speedup | peak RSS MB | failures |
|---|---|---|---|---|---|---|
| w1 | 1 | 7 (serial path) | 1045.9 | 1.00× | 3944 | 0 |
| w2 | 2 | 6 | 760.7 | 1.37× | 4635 | 0 |
| w3 | 3 | 4 | 749.4 | 1.40× | 5402 | 0 |
| w4 | 4 | 3 | 787.1 | 1.33× | 6821 | 0 |
| w2t7 | 2 | 7 | 742.1 | 1.41× | 5181 | 0 |
| **w3t7** | **3** | **7** | **698.2 / 709.1** | **1.50× / 1.48×** | **5005 / 5189** | 0 |
| w4t7 | 4 | 7 | 703.8 | 1.49× | 6416 | 0 |

`w3t7` was measured twice (once per round); the 1.6% spread between its two runs is the
run-to-run variance this table is read against. **Every one of the 8 concurrent arms is
byte-identical to the serial arm** — 12 PNGs plus `autolevel_stamps.jsonl`, 13 products each,
hashed.

**Dividing the box among the workers loses at every N** — 760.7 vs 742.1 at N=2, 749.4 vs 709.1
at N=3, 787.1 vs 703.8 at N=4 — so `ENGINE_THREADS_PER_WORKER` is 7, passed explicitly. The
reason is §1's asymmetry read the other way: a worker sitting in its single-threaded Python tail
leaves cores that only an over-provisioned sibling engine can pick up, so nominal
oversubscription (3×7 = 21 threads on 12 logical cores) is what keeps the box busy. The generic
advice to size a fan-out for its actual N is right where the fanned-out work is uniformly
parallel; this work is half serial per row.

**Adopted: 3 workers × 7 threads.** 4 workers is indistinguishable on wall clock (703.8 vs
709.1 s, inside the repeat variance) and costs +24% peak RSS and a fourth concurrent engine at
CLAUDE.md's process cap, so it buys nothing for something.

## 4. The speedup is ~1.5×, and the 2.5–3.5× projection does not hold

That projection came from core counts, and core counts are not what binds. Two measurements say
where the wall actually goes:

* **Scheduling is not the loss.** For each arm, `max(longest row, Σ row seconds / workers)` is
  the best any schedule of that plan could do. Realized/bound is **0.99 at w2 and w2t7, 0.95-0.96 at w3
  and w3t7, 0.86-0.91 at w4/w4t7** — the pass is at or near the achievable bound everywhere
  except 4 workers, where the loss is contention rather than ordering.
* **The loss is that concurrency inflates each row.** Σ row-seconds over the same 12 rows goes
  **1045.8 s serial → 1502.8 (w2) → 1998.0 (w3t7) → 2700.2 (w4)**. Two causes, both structural:
  the engine leg already holds 7 of 12 threads, so N engines contend immediately; and the Python
  tail is **memory-bandwidth-bound** — which is exactly what `tail_optimize` established when it
  measured the horizontal downsample leg at 80 M element-ops/s — so N concurrent tails share
  bandwidth rather than multiplying throughput.

A third floor sits under both: **the plan's longest single row**. One phoenix `smooth` row is
227.1 s of the 1045.9 s serial pass, and it gets *slower* under concurrency (345-354 s at 2 workers,
471-478 s at 3), so it alone is ~67% of the adopted point's wall. Release-render cost is
family-driven and phoenix is the expensive family; a release whose mix is heavier in phoenix has
a *lower* concurrency ceiling, not a higher one.

**Per-row cost, which is the number to carry into any throughput projection:** 87.2 s/row serial
→ **58.6 s/row** at the adopted point, over these 12 rows at 2560×1440 ss4 on this box. Per
10,000 full-res release renders that is **242.1 h → 162.9 h**.

**Idle-core headroom for the tail's own chunk-loop threading (the mechanism deliberately not
also enabled): there is none worth having at the adopted point.** 21 engine threads over 12
logical cores plus three concurrent memory-bound tails is not a box with spare capacity — the
hybrid would be competing for the bandwidth that already caps this. It is worth revisiting only
for a release whose mix is dominated by one long row, where the pass is critical-path-bound
rather than throughput-bound and the tail threading would shorten that row itself.

