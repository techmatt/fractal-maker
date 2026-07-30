#!/usr/bin/env python
"""Shared pixel→plane math + render-one invocation for the explorer AND descent apps.

Extracted from the original inline `tools/explorer/app.py` so the coordinate
math exists **once**. A silent divergence in this math would mislocate every
emitted descent solution (the descent harness records human-chosen locations as
training data), so both tools import these functions rather than copying them.

Guarded by `tools/explorer/test_render_core.py`:
  * a *differential* test vs a frozen copy of the old inline implementation over a
    grid of clicks / frame widths, and
  * a *zero-change* proof that the explorer's render-one argv is byte-identical to
    the pre-extraction command line.

All plane coordinates are `Decimal` (never round-tripped through a JS/f64 float);
`getcontext().prec = 60` gives plenty of guard digits for deep-zoom math.
"""
from __future__ import annotations

import subprocess
import sys
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 60

# ---------------------------------------------------------------------------
# Config (shared paths)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_BIN = REPO_ROOT / "target" / "release" / "fractal-generator.exe"
CLEAN_COLORMAPS = REPO_ROOT / "data" / "palettes" / "clean_colormaps.json"

# maxiter escalation policy (mirror of the historical explorer / active_ckpt policy):
#   maxiter = base * (1 + k * log2(fw_home / fw)), clamped to [min, max].
FW_HOME = Decimal("3.0")
MAXITER_BASE = 500
MAXITER_K = 0.30
MAXITER_MIN = 200
MAXITER_MAX = 8000


# ---------------------------------------------------------------------------
# Coordinate math (Decimal) — the load-bearing shared code
# ---------------------------------------------------------------------------
def dec_str(x) -> str:
    """Decimal/str/number → plain decimal string (no scientific notation)."""
    return format(Decimal(x), "f")


def click_to_world(px, py, ctr_x, ctr_y, fw, w, h):
    """Pixel (top-left origin) on a `w×h` panel → complex-plane point (Decimal).

    Identical to the original inline `tools/explorer/app.py::click_to_world`.
    """
    fw = Decimal(fw)
    W, H = Decimal(w), Decimal(h)
    fh = fw * H / W
    fx = Decimal(px) / W - Decimal("0.5")
    fy = Decimal(py) / H - Decimal("0.5")
    world_x = Decimal(ctr_x) + fx * fw
    world_y = Decimal(ctr_y) - fy * fh   # screen-y down, imaginary up
    return world_x, world_y


def box_commit(down_px, down_py, cur_px, ctr_x, ctr_y, fw, w, h):
    """Rubber-band box → `(new_cx, new_cy, new_fw)`, all Decimal.

    Semantics (per the descent-harness spec): the mousedown pixel sets the
    **center**; the horizontal drag distance sets the **horizontal radius** in
    plane units; ``new_fw = 2 × radius``. Height follows a fixed 16:9 aspect at
    render time, so a viewport is fully described by *center + fw* alone — no
    aspect is stored. No snapping / no rounding beyond the Decimal precision.
    """
    cx, cy = click_to_world(down_px, down_py, ctr_x, ctr_y, fw, w, h)
    fw = Decimal(fw)
    W = Decimal(w)
    radius_world = (abs(Decimal(cur_px) - Decimal(down_px)) / W) * fw
    new_fw = radius_world * 2
    return cx, cy, new_fw


def auto_maxiter(fw, override=None) -> int:
    """Depth-aware iteration cap. `override` (when not None) wins verbatim.

    Identical policy to the original inline explorer `auto_maxiter` and to
    `tools/scoring/active_ckpt.auto_maxiter` (the label-crop maxiter source).
    """
    if override is not None:
        return int(override)
    import math
    fw = Decimal(fw)
    ratio = FW_HOME / fw if fw > 0 else Decimal(1)
    lz = math.log2(float(ratio)) if ratio > 0 else 0.0
    val = MAXITER_BASE * (1.0 + MAXITER_K * lz)
    return int(max(MAXITER_MIN, min(MAXITER_MAX, val)))


# ---------------------------------------------------------------------------
# render-one invocation (navigation renders; the quality crop goes through
# corpus_common.render_corpus_crop, the sanctioned byte-reproducible path)
# ---------------------------------------------------------------------------
def render_one_argv(cx, cy, fw, maxiter, w, h, ss, palette, colormaps, out,
                    *, julia_c=None, family=None):
    """Build the `render-one` argv. Argument ORDER is frozen: the explorer's
    zero-change proof asserts this list byte-for-byte for `family=None`.

    `family=None` reproduces the historical explorer command exactly (Mandelbrot
    default, no `--family` flag). The descent app passes `family` for multibrot.
    """
    argv = [
        str(RENDER_BIN), "render-one",
        "--cx", dec_str(cx), "--cy", dec_str(cy), "--fw", dec_str(fw),
        "--width", str(w), "--height", str(h), "--supersample", str(ss),
        "--palette", str(palette), "--colormaps", str(colormaps),
        "--maxiter", str(maxiter), "--out", str(out),
    ]
    if family:
        argv += ["--family", str(family)]
    if julia_c is not None:
        argv += ["--julia", "--c", dec_str(julia_c[0]), dec_str(julia_c[1])]
    return argv


def run_render_one(argv, out, *, low_priority=False, threads=None):
    """Run a `render-one` argv, raising on failure; returns the output `Path`.

    `low_priority=False` (the explorer default) reproduces the historical launch
    exactly: no custom env, no creation flags. `low_priority=True` (the descent
    default) launches through the committed engine defaults (BELOW_NORMAL +
    RAYON_NUM_THREADS) so a browse session yields to interactive work. Neither
    affects the rendered bytes (the render is deterministic in thread count).
    """
    env = None
    creationflags = 0
    if low_priority:
        cc = _corpus_common()
        env = cc.default_engine_env(threads=threads)
        creationflags = cc.default_creationflags()
    proc = subprocess.run(argv, capture_output=True, text=True,
                          env=env, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(f"render failed: {proc.stderr or proc.stdout}")
    return Path(out)


def _corpus_common():
    p = str(REPO_ROOT / "tools" / "corpus")
    if p not in sys.path:
        sys.path.insert(0, p)
    import corpus_common  # noqa: E402
    return corpus_common
