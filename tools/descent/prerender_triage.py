#!/usr/bin/env python
"""Render the triage wall's thumbnails: every pool atom (and every reference) at
1x / 4x / 16x its own size, vivid `blue_orange`, navigation fidelity.

Framing is the point. At 1x the island fills the frame and every atom looks alike,
so the ladder exists to give the eye a composition to judge instead of a silhouette;
the wall shows 4x by default and a click cycles the other two.

Images land under `data/descent_harness/thumbs/` — repo-relative in records, bytes
out-of-tree through `artifacts.resolve` (600 files at 200 atoms, 3000+ at 1000).
Already-rendered tiles are skipped, so this is safe to re-run after the pool grows.

Concurrency follows CLAUDE.md: at most `--workers` (default 4) engine PROCESSES,
each given an explicit thread count (4 x 3 = 12 logical cores) at BELOW_NORMAL, so
a long pre-render leaves the desktop usable.

Run:  uv run python tools/descent/prerender_triage.py
      uv run python tools/descent/prerender_triage.py --scales 4        # default view first
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "explorer"))

import render_core as rc      # noqa: E402  (shared coord math + render-one invocation)
import triage_store as ts     # noqa: E402

WORKERS = 4                   # concurrent engine PROCESSES (CLAUDE.md cap)
THREADS_PER_WORKER = 3        # 4 x 3 = the box's 12 logical cores


def tile_jobs(scales=None) -> list[tuple[str, int, dict]]:
    """(tile_id, scale, geometry) for every tile the wall can show, references first."""
    scales = tuple(scales or ts.SCALES)
    jobs = []
    for r in ts.load_references():
        for s in scales:
            jobs.append((r["id"], s, {"cx": r["cx"], "cy": r["cy"],
                                      "base": r["base_scale"], "family": r["family"]}))
    for a in ts.load_pool():
        for s in scales:
            jobs.append((a["id"], s, {"cx": a["cx"], "cy": a["cy"],
                                      "base": a["window_scale"], "family": a["family"]}))
    return jobs


def render_tile(tile_id: str, scale: int, geom: dict, *, threads=THREADS_PER_WORKER,
                force=False) -> Path:
    """Render one wall tile. Idempotent: an existing file is returned untouched."""
    out = ts.thumb_path(tile_id, scale)
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fw = ts.frame_width(geom["base"], scale)
    argv = rc.render_one_argv(geom["cx"], geom["cy"], f"{fw:.17e}",
                              rc.auto_maxiter(fw),
                              ts.THUMB_W, ts.THUMB_H, ts.THUMB_SS,
                              ts.THUMB_PALETTE, ts.THUMB_COLORMAPS, out,
                              family=geom["family"])
    rc.run_render_one(argv, out, low_priority=True, threads=threads)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"concurrent engine processes (default {WORKERS}; CLAUDE.md caps at 4)")
    ap.add_argument("--threads", type=int, default=THREADS_PER_WORKER,
                    help=f"rayon threads per engine process (default {THREADS_PER_WORKER})")
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help=f"only these scales (default {list(ts.SCALES)})")
    ap.add_argument("--force", action="store_true", help="re-render tiles that exist")
    ap.add_argument("--limit", type=int, default=None, help="stop after N renders")
    args = ap.parse_args(argv)

    if args.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2
    if not rc.RENDER_BIN.exists():
        print(f"render binary not found: {rc.RENDER_BIN}", file=sys.stderr)
        return 2

    ts.ensure_dirs()
    jobs = tile_jobs(args.scales)
    todo = [j for j in jobs
            if args.force or not (p := ts.thumb_path(j[0], j[1])).exists() or p.stat().st_size == 0]
    if args.limit:
        todo = todo[:args.limit]
    print(f"tiles: {len(jobs)} total, {len(todo)} to render "
          f"({args.workers} processes x {args.threads} threads, BELOW_NORMAL)")
    if not todo:
        return 0

    t0 = time.time()
    done = [0]
    errs: list[str] = []

    def work(job):
        tid, scale, geom = job
        try:
            render_tile(tid, scale, geom, threads=args.threads, force=args.force)
        except Exception as e:                      # keep going; report at the end
            errs.append(f"{tid} x{scale}: {e}")
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(todo):
            el = time.time() - t0
            print(f"  {done[0]:5d}/{len(todo)}  {done[0] / max(1e-9, el):.2f} tile/s  "
                  f"{el:6.1f}s  eta {(len(todo) - done[0]) / max(1e-9, done[0] / max(1e-9, el)):6.1f}s",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    print(f"rendered {len(todo) - len(errs)}/{len(todo)} in {time.time() - t0:.1f}s "
          f"-> {ts.thumbs_dir()}")
    for e in errs[:10]:
        print(f"  ERROR {e}", file=sys.stderr)
    if len(errs) > 10:
        print(f"  ... and {len(errs) - 10} more", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
