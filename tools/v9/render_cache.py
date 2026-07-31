#!/usr/bin/env python
"""Supervisor for the v9 augmentation-cache render (170,808 tiles, ~5.8 h measured).

v9 is the v8 corpus re-rendered at the raised iteration cap (docs/design/auto_maxiter.md).
Two things differ from the v8 supervisor and both matter:

  * **The plan carries a per-row `maxiter`.** v4..v8 rendered every tile at
    `v4-render-batch`'s flat `--maxiter` default of 8000, never through `auto_maxiter`;
    the v9 plan emits the production cap per row (3,248..43,469 over this corpus), so a
    cache tile and its deploy-time crop finally resolve the same number. `--maxiter` is
    not passed here at all — every v9 row overrides it.
  * **A SEPARATE cache tree.** Resume works by skipping any row whose output exists, so
    pointing this at v8's tree would skip all 170,808 old-cap tiles and silently render
    nothing. A fresh tree also makes a mixed-cap corpus impossible by construction rather
    than by discipline, and left v8's 12.1 GB intact as the rollback anchor for the
    duration of the build. (v8's tree was deleted on 2026-07-31, once v9 was trained and
    evaluated — regenerable from data/v8/plan.jsonl, ~4.7 h at 6 workers.)

Measured (tools/v9/estimate_cap_cost.py, 60 locations x 24 slots stratified over fw
deciles): 0.1218 s/tile at 6 workers, a 1.22x cost ratio against the old flat cap.

`v4-render-batch` already does the work and is already resumable — it skips any plan row
whose output exists, and `render::save_jpeg` is atomic (temp + rename), so a tile on disk
is a COMPLETE tile and a kill costs at most the handful in flight. This wrapper adds the
three things a long unattended run needs and the subcommand has no opinion about:

  * **6 worker threads, BelowNormal.** `RAYON_NUM_THREADS=6` sizes the rayon pool inside
    the ONE render process (see the `WORKERS` note — the project's 4-way cap is about
    concurrent processes, not threads), and the process is started in
    BELOW_NORMAL_PRIORITY_CLASS so a multi-hour render does not fight the desktop for CPU.
  * **Restart on death.** An external reaper kills long runs at random, with no error and
    no exit code worth reading. Because resume is free and idempotent, the correct response
    to *any* abnormal exit is simply to start again; each restart re-scans the plan, skips
    what is already on disk, and continues. Stops when the plan is fully materialised, or
    after --max-restarts consecutive restarts that made NO progress (a real failure, not a
    reaper).
  * **A progress record.** Per-cycle tile counts + the child's own log land under
    scratch/v9_render/ so the run is auditable after the fact.

  uv run python tools/v9/render_cache.py                 # foreground (background it yourself)
  uv run python tools/v9/render_cache.py --status        # count tiles on disk and exit

Renders into `paths.bulk("data/v9/aug_cache")` -> ARTIFACTS_ROOT, out of the working tree.
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

PLAN = ROOT / "data" / "v9" / "plan.jsonl"
# The merged 77-colormap library the v8b recipe draws from. NOT the subcommand's default
# (`data/palettes/clean_colormaps.json`): that library does not contain `blue_orange`, which
# v8b puts on every location, so the default would fail every second row of the plan.
COLORMAPS = ROOT / "data" / "v9" / "colormaps.json"
CACHE = paths.bulk("data/v9/aug_cache")
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
LOG_DIR = paths.scratch("v9_render")
# Rayon worker THREADS inside the single render process — not concurrent processes.
#
# CLAUDE.md caps worker pools at 4, and this deliberately runs 6. The cap is about PROCESS
# fan-out: what makes this machine unusable is 4+ separate `fractal-generator.exe` instances
# competing, each with its own rayon pool, its own plan scan and its own resident colormap
# LUTs. One process at 6 threads is a different resource shape — 6 of 12 logical cores, one
# plan parse, one palette bake — and Matt confirmed (2026-07-29) that the interactive
# slowdown tracks process count, not thread count. Raised 3 -> 6 mid-run; resume is
# idempotent, so the change cost the 3 tiles in flight.
#
# If you are tempted to "fix" this back to 4 to satisfy the cap: read the paragraph above
# first, and if you still think it should change, change it for a measured reason.
WORKERS = 6


_EXPECTED: set[str] | None = None


def expected_outputs() -> set:
    """The exact set of output paths the plan names, loaded once.

    The v8 supervisor counted every `.jpg` under the cache root instead, and that count
    drifted: 24 location dirs left over from a pre-re-split manifest meant the tree held
    171,384 tiles for a 170,808-row plan, so `on_disk >= total` was true before the run
    began. Counting only what the plan names cannot drift that way."""
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
    """Completed tiles THE PLAN ASKED FOR. Counts only `.jpg` — an interrupted encode is a
    `.jpg.<pid>.tmp` sibling that never had the destination name, so it cannot inflate
    this — and only paths in the plan, so a stray tile cannot either."""
    if not CACHE.exists():
        return 0
    exp = expected_outputs()
    n = 0
    for root, _dirs, files in os.walk(CACHE):
        for fn in files:
            if fn.endswith(".jpg") and \
                    os.path.normcase(Path(root, fn).as_posix()) in exp:
                n += 1
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
        print(f"v9 aug cache: {done}/{total} tiles ({100*done/total:.2f}%)  -> {CACHE}")
        return 0

    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")
    if not COLORMAPS.exists():
        sys.exit(f"colormap library missing: {COLORMAPS} (uv run python tools/v9/build_plan.py)")
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
            print(f"[v9-render] complete: {before}/{total} tiles", flush=True)
            break
        print(f"[v9-render] cycle {cycle}: {before}/{total} on disk "
              f"({100*before/total:.2f}%), workers={WORKERS} BelowNormal", flush=True)
        rc = run_once(log_path)
        swept = sweep_stale_temps()
        after = tiles_on_disk()
        rec = {"cycle": cycle, "rc": rc, "before": before, "after": after,
               "total": total, "swept_temps": swept,
               "elapsed_s": round(time.time() - t0, 1)}
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[v9-render] cycle {cycle} rc={rc} {before}->{after}/{total} "
              f"(+{after - before}, swept {swept} temps, "
              f"{(time.time()-t0)/3600:.2f} h elapsed)", flush=True)
        if after >= total:
            print(f"[v9-render] complete: {after}/{total} tiles", flush=True)
            break
        if after <= before:
            stale += 1
            if stale >= a.max_restarts:
                print(f"[v9-render] ABORT: {stale} consecutive restarts made no progress "
                      f"(rc={rc}); see {log_path}", flush=True)
                return 1
        else:
            stale = 0
    print(f"[v9-render] done in {(time.time()-t0)/3600:.2f} h -> {CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
