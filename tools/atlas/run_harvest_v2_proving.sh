#!/usr/bin/env bash
# Harvest v2 — the ONE-HOUR PROVING RUN. Not the continuous launch.
#
# WHAT THIS PROVES, and it is one thing above all: REALIZED vs INTENDED MIX. Harvest v1 set a
# 70% native-multibrot share with root weights and realized 19.6% over 149 batches
# (`discovery_pipeline.md` §3.1); the v2 allocator moves the decision to the POP and measures
# itself. A miss here is a failed shakedown, not a tuning opportunity.
#
# THE FLAGS, each with the section that asks for it:
#   --pop-quota --quota-floor 0.05   §1 + addendum. Per-partition quota enforced at the
#                                    population level, deficit in Matt's currency
#                                    (n4 + 0.1*n3 through the amendment overlay + library)
#                                    against a UNIFORM target, price-weighted by measured
#                                    cost-to-mine, with a universal 5%-of-total-time floor.
#                                    Mutually exclusive with --scheduler by construction.
#   --julia-seed-pool ...v2.json     §2. The merged julia:mandelbrot supply: ranked q4 mining,
#                                    recall q4 mining, near-minibrot at the SINGLE chosen rung,
#                                    and the seeded loop — thinned to the measured c-spacing
#                                    floor of 1e-2 in yield order.
#   --seed-pool-rate 8               §2. Metered injection (8 entries per BATCH from a
#                                    persisted cursor) rather than a t=0 dump, which is
#                                    `julia_c_sourcing.md`'s "run to the knee, then refill" by
#                                    construction. A wholesale dump is what let 630 injected
#                                    roots permanently out-number the native ones in v1.
#   --maneuvers --maneuver-view-prior  §3. The view screen ACTUALLY RUNS (v1:
#                                    man_view_screened=0 because --maneuvers was off), and
#                                    every screened row now carries BOTH view_fit_v1.1 and
#                                    composite_v3. composite_v3 remains the sort key.
#   --maneuvers-on-admissions        §2. The triggered channel, promoted to a budgeted supply
#                                    source (55.0% vs 25.5% at >=3, partition-matched).
#   --budget 60 --wall-budget 75     §6. A ONE-HOUR ACTIVE cap paired with a wall budget. The
#                                    two are not duplicates: active_s counts only the timed
#                                    batch block and root replenishment sits outside it.
#   --below-normal                   standing run mechanic; every engine child inherits it.
#
# STOP / RESUME: `touch <run-dir>/STOP` halts at the next batch boundary; re-run with
# --resume to continue. Both are standing mechanics, exercised by this script's own re-entry.
#
#   bash tools/atlas/run_harvest_v2_proving.sh
#   bash tools/atlas/run_harvest_v2_proving.sh --resume
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_TS="${RUN_TS:-harvest_v2_proving_$(date +%Y%m%d)}"
RUN_DIR="data/discovery/${RUN_TS}"
LOG="scratch/harvest_v2/${RUN_TS}.log"
mkdir -p "$(dirname "$LOG")"

EXTRA=("$@")

echo "[harvest-v2] run_dir=${RUN_DIR}  log=${LOG}"
# Redirected to a FILE, never piped through tail/head: a pipe buffers, so a job that prints
# progress with flush=True shows nothing for its whole runtime and the header lines that said
# what it skipped are lost (`CLAUDE.md`, "Long background runs").
uv run python tools/atlas/steered_frontier.py \
  --run-dir "${RUN_DIR}" \
  --families mandelbrot,multibrot3,multibrot4,multibrot5 \
  --julia-hook \
  --pop-quota --quota-floor 0.05 \
  --julia-seed-pool data/atlas/julia_supply_pool_v2.json \
  --phoenix-seed-pool data/atlas/phoenix_seed_pool.json \
  --seed-pool-rate 8 \
  --maneuvers --maneuver-view-prior --maneuver-neighborhood \
  --maneuvers-on-admissions \
  --mem-recency --recency-k 8 \
  --budget 60 --wall-budget 75 \
  --below-normal \
  "${EXTRA[@]}" \
  > "${LOG}" 2>&1

echo "[harvest-v2] done — summary: ${RUN_DIR}/summary.json"
