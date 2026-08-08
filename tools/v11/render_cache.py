#!/usr/bin/env python
r"""Supervisor for the v11 augmentation-cache render — 361,696 tiles, 11,303 locations.

ONE `fractal-generator.exe` at a time, 7 rayon threads, BelowNormal (CLAUDE.md: the 4-way
cap is about concurrent PROCESSES; this is one process with threads). The engine's own
resume — skip any tile whose output exists — is what makes every restart free, so the
supervisor's job is only to keep restarting and to bound how long a wedged unit can cost.

CHUNKS, AND WHY THE PLAN IS SPLIT AT ALL. `crop-batch` resumes at TILE granularity, which
is already the finest boundary the render has. The MANIFEST does not: it is rewritten whole
on every invocation, so a kill mid-run leaves a partial one and a single-invocation run has
no durable manifest until the very end. Splitting the plan into `--chunks` pieces makes the
per-tile manifest durable at a chunk boundary (each chunk writes `cache_manifest.partNNN
.jsonl`; `--finish` concatenates them in chunk order). It also gives the per-unit backstop
something to bound: a 30-minute timeout on a ~4-minute chunk is a real backstop, where the
same 30 minutes against a single 3-hour invocation would be a kill switch on the job itself
(CLAUDE.md: "a backstop longer than the job's budget is not a backstop").

THE PLAN IS SHUFFLED (tools/v11/build_plan.plan_order), so every chunk is a fair sample of
the corpus and the rate printed after chunk 3 is a rate for the run. `--status` reprojects
from the LAST FEW chunks, never from the run-to-date average.

RESUMED == UNINTERRUPTED. A tile is a pure function of `(seed_tag, loc_id, tile)`; nothing
in the draw depends on run order, on which chunk a location landed in, or on what else was
rendered. Writes are atomic — `render::save_jpeg` encodes to `<name>.jpg.<pid>.tmp` and
renames — so a kill can leave a stray `.tmp` but never a truncated `.jpg` that resume would
then skip. `--sweep-tmp` removes strays; `--verify` in tools/v11/verify_cache.py is what
proves the claim rather than asserting it.

  uv run python tools/v11/render_cache.py                 # foreground (background it)
  uv run python tools/v11/render_cache.py --status        # counts + reprojection, then exit
  uv run python tools/v11/render_cache.py --limit-chunks 1  # bounded end-to-end
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
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import corpus_common as cc  # noqa: E402  THE engine launch defaults — never restated
import paths                # noqa: E402

PLAN = "data/v11/plan.jsonl"
CACHE_DIR = "data/v11/aug_cache"
CACHE_MANIFEST = "data/v11/cache_manifest.jsonl"
COLORMAPS = ROOT / "data" / "v11" / "colormaps.json"
RECIPE = ROOT / "data" / "v11" / "aug_recipe.json"
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
LOG_DIR = paths.scratch("v11_render")

N_CHUNKS = 40                  # ~283 locations each; ~4 min at the measured rate
CHUNK_TIMEOUT_S = 1800         # 7.5x the expected chunk, 6% of the wall cap
WALL_CAP_S = 8 * 3600          # the run's budget; checked between chunks
MAX_RETRIES = 3                # per chunk, before the supervisor gives up on it


def plan_rows():
    """The plan's location rows, header lines dropped — the same `#` rule the engine uses."""
    return [l for l in paths.bulk(PLAN).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def chunk_path(i: int) -> Path:
    return LOG_DIR / f"chunk{i:03d}.jsonl"


def part_path(i: int) -> Path:
    return paths.bulk(f"data/v11/cache_manifest.part{i:03d}.jsonl")


def write_chunks(rows, n_chunks: int) -> list[Path]:
    """Split the plan into contiguous chunks. Contiguous, not strided: the plan is already
    shuffled, so contiguity costs no homogeneity and keeps a chunk file trivially
    reconstructible from the plan by index."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    per = -(-len(rows) // n_chunks)
    out = []
    for i in range(n_chunks):
        part = rows[i * per:(i + 1) * per]
        if not part:
            break
        p = chunk_path(i)
        p.write_text("\n".join(part) + "\n", encoding="utf-8")
        out.append(p)
    return out


def tiles_on_disk() -> int:
    root = paths.bulk(CACHE_DIR)
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob("*/*.jpg"))


def sweep_tmp() -> int:
    """Remove atomic-write leftovers from a killed run. A `.tmp` is never a valid tile and
    never skipped by resume, so this is hygiene, not correctness."""
    root = paths.bulk(CACHE_DIR)
    n = 0
    for p in root.glob("*/*.tmp") if root.exists() else ():
        p.unlink()
        n += 1
    return n


def run_chunk(i: int, chunk: Path, recipe: dict, timeout: int) -> tuple[bool, float, str]:
    """One chunk through `crop-batch`. Returns (ok, seconds, tail-of-stderr)."""
    cmd = list(recipe["render_command"])
    for flag, val in (("--locations", str(chunk)), ("--manifest", str(part_path(i)))):
        cmd[cmd.index(flag) + 1] = val
    t0 = time.time()
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                            env=cc.default_engine_env(),
                            creationflags=cc.default_creationflags())
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, f"TIMEOUT after {timeout}s"
    tail = "\n".join((pr.stderr or "").strip().splitlines()[-4:])
    return pr.returncode == 0, time.time() - t0, tail


def finish(n_chunks: int) -> int:
    """Concatenate the per-chunk manifests, in chunk order, into the one cache manifest."""
    dest = paths.bulk(CACHE_MANIFEST)
    n = 0
    with dest.open("w", encoding="utf-8") as out:
        for i in range(n_chunks):
            p = part_path(i)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    n += 1
    return n


def status(rows, n_chunks: int, expected: int):
    st = LOG_DIR / "progress.jsonl"
    done = [json.loads(l) for l in st.read_text(encoding="utf-8").splitlines()
            if l.strip()] if st.exists() else []
    on_disk = tiles_on_disk()
    print(f"plan       : {len(rows)} locations x 32 = {expected} tiles")
    print(f"on disk    : {on_disk} ({100*on_disk/max(expected,1):.1f}%)")
    print(f"chunks done: {len(done)}/{n_chunks}")
    if len(done) >= 2:
        # Reproject from RECENT throughput, never the run-to-date average — an average over
        # a shuffled plan is fair, but a decaying rate (thermal, contention) is not visible
        # in it. CLAUDE.md's projection rule.
        recent = done[-min(5, len(done)):]
        rate = sum(r["locations"] for r in recent) / sum(r["seconds"] for r in recent)
        left = len(rows) - sum(r["locations"] for r in done)
        print(f"rate       : {rate:.2f} loc/s over the last {len(recent)} chunk(s) "
              f"({sum(r['seconds'] for r in recent)/len(recent):.0f}s/chunk)")
        print(f"remaining  : {left} locations -> {left/max(rate,1e-9)/3600:.2f} h")
    return on_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--finish", action="store_true", help="concatenate the manifests and exit")
    ap.add_argument("--sweep-tmp", action="store_true")
    ap.add_argument("--chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--limit-chunks", type=int, default=None,
                    help="run at most N chunks — the bounded end-to-end")
    ap.add_argument("--wall-cap-s", type=int, default=WALL_CAP_S)
    ap.add_argument("--chunk-timeout-s", type=int, default=CHUNK_TIMEOUT_S)
    a = ap.parse_args()

    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    rows = plan_rows()
    expected = len(rows) * recipe["fan_out"]["tiles_per_location"]

    if a.status:
        status(rows, a.chunks, expected)
        return
    if a.sweep_tmp:
        print(f"removed {sweep_tmp()} stray .tmp file(s)")
        return
    if a.finish:
        print(f"cache_manifest rows: {finish(a.chunks)}")
        return

    if not BIN.exists():
        raise SystemExit(f"{BIN} missing — cargo build --release")
    prio = cc.set_below_normal_priority()
    chunks = write_chunks(rows, a.chunks)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prog = LOG_DIR / "progress.jsonl"
    done_idx = {json.loads(l)["chunk"] for l in prog.read_text(encoding="utf-8").splitlines()
                if l.strip()} if prog.exists() else set()

    print(f"v11 cache render: {len(rows)} locations / {expected} tiles over {len(chunks)} "
          f"chunks; {cc.DEFAULT_ENGINE_THREADS} threads, priority {prio}; wall cap "
          f"{a.wall_cap_s/3600:.1f} h, chunk timeout {a.chunk_timeout_s}s", flush=True)
    swept = sweep_tmp()
    if swept:
        print(f"  swept {swept} stray .tmp file(s) from a previous kill", flush=True)
    if done_idx:
        print(f"  resuming: {len(done_idx)} chunk(s) already recorded complete", flush=True)

    t_start = time.time()
    ran = 0
    for i, chunk in enumerate(chunks):
        if i in done_idx:
            continue
        if time.time() - t_start > a.wall_cap_s:
            print(f"WALL CAP {a.wall_cap_s}s reached at chunk {i}; stopping cleanly. "
                  f"Re-run to continue — every finished tile is skipped.", flush=True)
            break
        if a.limit_chunks is not None and ran >= a.limit_chunks:
            print(f"--limit-chunks {a.limit_chunks} reached; stopping.", flush=True)
            break
        n_loc = len(chunk.read_text(encoding="utf-8").splitlines())
        for attempt in range(1, MAX_RETRIES + 1):
            ok, secs, tail = run_chunk(i, chunk, recipe, a.chunk_timeout_s)
            if ok:
                break
            print(f"  chunk {i} attempt {attempt}/{MAX_RETRIES} FAILED in {secs:.0f}s: "
                  f"{tail}", flush=True)
            sweep_tmp()
        ran += 1
        rec = {"chunk": i, "locations": n_loc, "seconds": round(secs, 1), "ok": ok,
               "tiles_on_disk": tiles_on_disk(), "t": round(time.time() - t_start, 1)}
        with prog.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        el = time.time() - t_start
        rate = n_loc / max(secs, 1e-9)
        print(f"[chunk {i+1}/{len(chunks)}] {n_loc} loc in {secs:.0f}s ({rate:.2f} loc/s), "
              f"tiles {rec['tiles_on_disk']}/{expected}, elapsed {el/60:.1f} min"
              f"{'' if ok else '  ** UNRECOVERED **'}", flush=True)

    n_rows = finish(a.chunks)
    print(f"\ncache_manifest: {n_rows} rows -> {paths.bulk(CACHE_MANIFEST)}", flush=True)
    print(f"tiles on disk : {tiles_on_disk()} / {expected}", flush=True)
    print(f"wall          : {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
