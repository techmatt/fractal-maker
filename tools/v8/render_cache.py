#!/usr/bin/env python
"""Supervisor for the v8 augmentation-cache render (171,384 tiles, ~13-14 h).

`v4-render-batch` already does the work and is already resumable — it skips any plan row
whose output exists, and `render::save_jpeg` is atomic (temp + rename), so a tile on disk
is a COMPLETE tile and a kill costs at most the handful in flight. This wrapper adds the
three things a long unattended run needs and the subcommand has no opinion about:

  * **3 workers, BelowNormal.** `RAYON_NUM_THREADS=3` caps the rayon pool (the project's
    4-worker ceiling), and the process is started in BELOW_NORMAL_PRIORITY_CLASS so a
    half-day render does not fight the desktop for CPU.
  * **Restart on death.** An external reaper kills long runs at random, with no error and
    no exit code worth reading. Because resume is free and idempotent, the correct response
    to *any* abnormal exit is simply to start again; each restart re-scans the plan, skips
    what is already on disk, and continues. Stops when the plan is fully materialised, or
    after --max-restarts consecutive restarts that made NO progress (a real failure, not a
    reaper).
  * **A progress record.** Per-cycle tile counts + the child's own log land under
    scratch/v8_render/ so the run is auditable after the fact.

  uv run python tools/v8/render_cache.py                 # foreground (background it yourself)
  uv run python tools/v8/render_cache.py --status        # count tiles on disk and exit

Renders into `paths.bulk("data/v8/aug_cache")` -> ARTIFACTS_ROOT, out of the working tree.
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

PLAN = ROOT / "data" / "v8" / "plan.jsonl"
# The merged 77-colormap library the v8b recipe draws from. NOT the subcommand's default
# (`data/palettes/clean_colormaps.json`): that library does not contain `blue_orange`, which
# v8b puts on every location, so the default would fail every second row of the plan.
COLORMAPS = ROOT / "data" / "v8" / "colormaps.json"
CACHE = paths.bulk("data/v8/aug_cache")
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
LOG_DIR = paths.scratch("v8_render")
WORKERS = 3                       # project cap is 4; 3 leaves the machine usable


def plan_total() -> int:
    with PLAN.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def tiles_on_disk() -> int:
    """Completed tiles. Counts only `.jpg` — an interrupted encode is a `.jpg.<pid>.tmp`
    sibling that never had the destination name, so it cannot inflate this."""
    if not CACHE.exists():
        return 0
    n = 0
    for _root, _dirs, files in os.walk(CACHE):
        n += sum(1 for fn in files if fn.endswith(".jpg"))
    return n


def sweep_stale_temps() -> int:
    """Remove `.jpg.<pid>.tmp` leftovers from a killed encode. Cosmetic — they are never
    read and never collide with a real tile name — but a half-day run under a reaper can
    accumulate them, and leaving debris in a cache nobody looks at is how a cache stops
    being trustworthy."""
    n = 0
    if not CACHE.exists():
        return 0
    for root, _dirs, files in os.walk(CACHE):
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
    if a.status:
        done = tiles_on_disk()
        print(f"v8 aug cache: {done}/{total} tiles ({100*done/total:.2f}%)  -> {CACHE}")
        return 0

    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")
    if not COLORMAPS.exists():
        sys.exit(f"colormap library missing: {COLORMAPS} (uv run python tools/v8/build_plan.py)")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "render.log"
    state_path = LOG_DIR / "progress.jsonl"

    t0 = time.time()
    stale = 0
    cycle = 0
    while True:
        cycle += 1
        before = tiles_on_disk()
        if before >= total:
            print(f"[v8-render] complete: {before}/{total} tiles", flush=True)
            break
        print(f"[v8-render] cycle {cycle}: {before}/{total} on disk "
              f"({100*before/total:.2f}%), workers={WORKERS} BelowNormal", flush=True)
        rc = run_once(log_path)
        swept = sweep_stale_temps()
        after = tiles_on_disk()
        rec = {"cycle": cycle, "rc": rc, "before": before, "after": after,
               "total": total, "swept_temps": swept,
               "elapsed_s": round(time.time() - t0, 1)}
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[v8-render] cycle {cycle} rc={rc} {before}->{after}/{total} "
              f"(+{after - before}, swept {swept} temps, "
              f"{(time.time()-t0)/3600:.2f} h elapsed)", flush=True)
        if after >= total:
            print(f"[v8-render] complete: {after}/{total} tiles", flush=True)
            break
        if after <= before:
            stale += 1
            if stale >= a.max_restarts:
                print(f"[v8-render] ABORT: {stale} consecutive restarts made no progress "
                      f"(rc={rc}); see {log_path}", flush=True)
                return 1
        else:
            stale = 0
    print(f"[v8-render] done in {(time.time()-t0)/3600:.2f} h -> {CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
