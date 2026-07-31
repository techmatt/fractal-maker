#!/usr/bin/env python
"""Render source-sheet tiles at the **identical framing every sheet shares**:
1x / 4x / 16x the atom's own size, vivid `blue_orange`, navigation fidelity.

Framing constants are imported from `source_store` (which imports them from the
triage wall) rather than restated — if a sheet rendered at different geometry or a
different palette the whole comparison would be void, so there is exactly one
definition and `test_sources.py` pins that the argv matches the wall's byte for byte.

Per the addendum: no atom is excluded a priori by the `A`-feasibility predictor. The
render is **attempted** for every atom; one that actually fails is dropped and logged
as an empirical render failure (`render_failures` in the source meta).

Concurrency follows CLAUDE.md: at most 4 engine PROCESSES, each with an explicit
thread count (4 x 3 = the box's 12 logical cores) at BELOW_NORMAL. A per-tile
hard timeout keeps one pathological location from eating the night.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "explorer"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))

import render_core as rc          # noqa: E402
import corpus_common as cc        # noqa: E402
import source_store as ss         # noqa: E402

WORKERS = 4                        # concurrent engine PROCESSES (CLAUDE.md cap)
THREADS_PER_WORKER = 3             # 4 x 3 = 12 logical cores
TILE_TIMEOUT_S = 180               # hard backstop per tile


def tile_argv(atom: dict, scale: int, out: Path) -> list[str]:
    fw = ss.frame_width(atom["window_scale"], scale)
    return rc.render_one_argv(atom["cx"], atom["cy"], f"{fw:.17e}",
                              rc.auto_maxiter(fw),
                              ss.TILE_W, ss.TILE_H, ss.TILE_SS,
                              ss.TILE_PALETTE, ss.TILE_COLORMAPS, out,
                              family=atom["family"])


def render_tile(atom: dict, scale: int, *, threads=THREADS_PER_WORKER,
                force=False, timeout=TILE_TIMEOUT_S) -> Path:
    """Render one tile. Idempotent; raises on engine failure or timeout."""
    import subprocess
    out = ss.tile_path(atom["id"], scale)
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = tile_argv(atom, scale, out)
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=cc.default_engine_env(threads=threads),
                          creationflags=cc.default_creationflags())
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "render failed").strip()[:300])
    if not (out.exists() and out.stat().st_size > 0):
        raise RuntimeError("render produced no output")
    return out


def render_atoms(atoms: list[dict], *, scales=None, workers=WORKERS,
                 threads=THREADS_PER_WORKER, force=False, log=print) -> dict:
    """Render every (atom, scale). Returns
    `{"ok": [...ids], "failures": [{id, scale, error}], "seconds": float}`.

    An atom whose **default-scale** tile fails is unusable on the sheet and is
    reported so the caller can drop it; a failure at 1x or 16x only costs that
    alternate view."""
    scales = tuple(scales or ss.SCALES)
    jobs = [(a, s) for a in atoms for s in scales]
    todo = [(a, s) for a, s in jobs
            if force or not (p := ss.tile_path(a["id"], s)).exists() or p.stat().st_size == 0]
    log(f"    tiles: {len(jobs)} total, {len(todo)} to render "
        f"({workers} procs x {threads} threads)")
    t0 = time.time()
    failures: list[dict] = []
    done = [0]

    def work(job):
        a, s = job
        try:
            render_tile(a, s, threads=threads, force=force)
        except Exception as e:
            failures.append({"id": a["id"], "scale": s, "error": str(e)[:200]})
        done[0] += 1
        if done[0] % 50 == 0 or done[0] == len(todo):
            el = time.time() - t0
            log(f"    {done[0]:5d}/{len(todo)}  {done[0]/max(1e-9, el):.2f} tile/s  {el:6.1f}s")

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
    bad_default = {f["id"] for f in failures if f["scale"] == ss.DEFAULT_SCALE}
    return {
        "ok": [a["id"] for a in atoms if a["id"] not in bad_default],
        "unusable": sorted(bad_default),
        "failures": failures,
        "seconds": round(time.time() - t0, 1),
    }


def reference_atoms() -> list[dict]:
    """The two known-good references, shaped as tile-renderable atoms.

    Both kinds are carried onto every sheet (addendum §3): `ref_mb19` is a true
    nucleus, and `ref_eye` is a good *view* with no nucleus within ~1e-4 — kept
    precisely because it is known-good material that is NOT minibrot-anchored, which
    is the premise this whole line of work rests on. Their `base_scale` plays the role
    of `window_scale`, chosen so the 4x tile is exactly the canonical known-good frame."""
    import triage_store as _ts
    return [{"id": r["id"], "cx": r["cx"], "cy": r["cy"],
             "window_scale": r["base_scale"], "family": r["family"],
             "label": r["label"], "is_reference": True}
            for r in _ts.load_references()]


def ensure_reference_tiles(log=print) -> dict:
    """Render the references into the source-sheet tile root (they also exist under the
    descent harness, but a sheet addresses tiles by a relative path beside itself)."""
    refs = reference_atoms()
    if not refs:
        log("    WARNING: no references found — run build_triage_pool.py --refs-only")
        return {"ok": [], "unusable": [], "failures": [], "seconds": 0.0}
    return render_atoms(refs, log=log)
