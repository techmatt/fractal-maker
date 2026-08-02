#!/usr/bin/env python
"""Supervisor for the v10 augmentation-cache EXTENSION (30,408 tiles of a 201,168-row plan).

v10 is v8's corpus with 1,267 maneuver-view locations appended. The recipe and the cap
policy are v9's, unchanged, so every prefix tile already on disk in `data/v9/aug_cache` is
bit-for-bit what this plan asks for — `tools/v10/build_plan.py` GATE A proves the prefix
rows are byte-identical to v9's. The plan is handed over WHOLE and `v4-render-batch`'s own
resume does the selection: it skips the 170,760 existing tiles and renders the 30,408 new
ones into `data/v10/aug_cache`. Handing it the whole plan rather than a filtered one is
deliberate — the resume is then verifying the prefix's existence on every cycle instead of
trusting a filter written once.

COST, and it is not v9's. Measured in RUN ORDER (`tools/v10/estimate_extend_cost.py`, 10
contiguous 72-row blocks, per-invocation overhead subtracted): 0.149..1.364 s/tile, a
**9.2x spread**, projecting **~5.1 h** at 6 workers. v9's whole-corpus rate was 0.1218
s/tile, so the appended material is ~4.9x more expensive per tile than linear scaling
would suggest. The driver is NOT the iteration cap — block 1 runs cap 16,267 at 0.149
s/tile and block 5 runs cap 16,207 at 0.842 — it is interior mass: a maneuver view is
framed on a minibrot nucleus, and every non-escaping pixel runs the full cap.

Because the rate moves by an order of magnitude across the plan, **reproject from this
run's own observed throughput** rather than restating the 5.1 h (CLAUDE.md's projection
rule). `--status` prints the current rate and a reprojection from the last cycle.

Otherwise this is v9's supervisor: 6 rayon threads in ONE process at BelowNormal (the
project's 4-way cap is about concurrent PROCESSES), restart on any abnormal exit because
resume is free and idempotent, and a per-cycle progress record under scratch/v10_render/.

  uv run python tools/v10/render_cache.py            # foreground (background it yourself)
  uv run python tools/v10/render_cache.py --status   # count tiles on disk and exit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

PLAN = ROOT / "data" / "v10" / "plan.jsonl"
COLORMAPS = ROOT / "data" / "v10" / "colormaps.json"
# TWO trees: the prefix (read-only, already rendered) and the extension (written here).
CACHES = (paths.bulk("data/v9/aug_cache"), paths.bulk("data/v10/aug_cache"))
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
LOG_DIR = paths.scratch("v10_render")
# Rayon worker THREADS inside the single render process — not concurrent processes. See
# tools/v9/render_cache.py's WORKERS note; the reasoning and the measurement are unchanged.
WORKERS = 6
PROJECTED_WALL_H = 5.06     # tools/v10/estimate_extend_cost.py, run-order blocks

_EXPECTED: set[str] | None = None


def expected_outputs() -> set:
    """The exact set of output paths the plan names, loaded once."""
    global _EXPECTED
    if _EXPECTED is None:
        exp = set()
        with PLAN.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    exp.add(os.path.normcase(json.loads(line)["out"]))
        _EXPECTED = exp
    return _EXPECTED


def plan_total() -> int:
    return len(expected_outputs())


def tiles_on_disk() -> int:
    """Completed tiles THE PLAN ASKED FOR, across both trees. Counts only `.jpg` (an
    interrupted encode is a `.jpg.<pid>.tmp` sibling) and only paths in the plan, so
    neither a partial write nor a stale location dir can inflate it."""
    exp = expected_outputs()
    n = 0
    for cache in CACHES:
        if not cache.exists():
            continue
        for root, _dirs, files in os.walk(cache):
            for fn in files:
                if fn.endswith(".jpg") and \
                        os.path.normcase(Path(root, fn).as_posix()) in exp:
                    n += 1
    return n


def sweep_stale_temps() -> int:
    """Remove `.jpg.<pid>.tmp` leftovers from a killed encode."""
    n = 0
    for cache in CACHES:
        if not cache.exists():
            continue
        for root, _dirs, files in os.walk(cache):
            for fn in files:
                if fn.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(root, fn))
                        n += 1
                    except OSError:
                        pass
    return n


def run_once(log_path: Path) -> int:
    env = dict(os.environ, RAYON_NUM_THREADS=str(WORKERS))
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    with log_path.open("ab") as log:
        p = subprocess.Popen(
            [str(BIN), "v4-render-batch", "--plan", str(PLAN),
             "--colormaps", str(COLORMAPS), "--log-every", "2000"],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--max-restarts", type=int, default=8,
                    help="consecutive no-progress restarts before giving up")
    a = ap.parse_args()

    total = plan_total()
    state_path = LOG_DIR / "progress.jsonl"
    if a.status:
        done = tiles_on_disk()
        print(f"v10 aug cache: {done}/{total} tiles ({100*done/total:.2f}%)")
        # Reproject from THIS RUN's throughput, not from the pre-run estimate: the
        # measured rate spans 9.2x across the plan, so the pre-run number is an opening
        # bid. Use the most recent cycle, not the run-to-date average — the latter is
        # dominated by whatever the run has already finished.
        if state_path.exists():
            recs = [json.loads(l) for l in
                    state_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            recs = [r for r in recs if r.get("cycle_s") and r["after"] > r["before"]]
            if recs:
                last = recs[-1]
                rate = (last["after"] - last["before"]) / last["cycle_s"]
                left = (total - done) / rate / 3600 if rate > 0 else float("inf")
                print(f"  last cycle: {rate:.2f} tiles/s  ->  {left:.2f} h remaining "
                      f"(pre-run estimate was {PROJECTED_WALL_H} h for the whole extension; "
                      f"do not restate it)")
        return 0

    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")
    if not COLORMAPS.exists():
        sys.exit(f"colormap library missing: {COLORMAPS} (tools/v10/build_plan.py)")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "render.log"

    t0 = time.time()
    stale = 0
    cycle = 0
    while True:
        cycle += 1
        before = tiles_on_disk()
        if before >= total:
            print(f"[v10-render] complete: {before}/{total} tiles", flush=True)
            break
        print(f"[v10-render] cycle {cycle}: {before}/{total} on disk "
              f"({100*before/total:.2f}%), {total-before} to render, "
              f"workers={WORKERS} BelowNormal", flush=True)
        c0 = time.time()
        rc = run_once(log_path)
        cycle_s = time.time() - c0
        swept = sweep_stale_temps()
        after = tiles_on_disk()
        rec = {"cycle": cycle, "rc": rc, "before": before, "after": after,
               "total": total, "swept_temps": swept, "cycle_s": round(cycle_s, 1),
               "elapsed_s": round(time.time() - t0, 1)}
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        rate = (after - before) / cycle_s if cycle_s > 0 else 0
        print(f"[v10-render] cycle {cycle} rc={rc} {before}->{after}/{total} "
              f"(+{after - before} at {rate:.2f} tiles/s, swept {swept} temps, "
              f"{(time.time()-t0)/3600:.2f} h elapsed)", flush=True)
        if after >= total:
            print(f"[v10-render] complete: {after}/{total} tiles", flush=True)
            break
        if after <= before:
            stale += 1
            if stale >= a.max_restarts:
                print(f"[v10-render] ABORT: {stale} consecutive restarts made no progress "
                      f"(rc={rc}); see {log_path}", flush=True)
                return 1
        else:
            stale = 0
    print(f"[v10-render] done in {(time.time()-t0)/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
