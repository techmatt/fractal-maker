"""SLOW: the release pass renders the same bytes serial and concurrent, on the real engine.

`-m slow` because every row here shells out to `fractal-generator.exe` and the concurrent arm
runs two of them at once — under the suite's `-n 4` that is up to 8 engines, which is exactly
the process fan-out CLAUDE.md caps. The lane is manual, so: **run it before committing any
change to the release render pass** (`release_pass.py`, `render_wallpaper`/`render_smooth`,
`deploy_tail`'s render paths or its `stamp_log`/temp-name plumbing). The default-lane
`test_release_pass.py` covers the orchestration and cannot see a pixel.

WHAT IT PINS, and why each row is in the plan:
  * `smooth` twice on ONE location at ONE geometry with two palettes — the temp-name collision
    case. Two workers rendering these simultaneously used to derive the same field-dump name
    and `finally`-unlink each other's file; `deploy_tail.field_tmp_token()` is what stops it,
    and nothing else in this plan would notice if it were removed.
  * `stripe` — the pure-field path (engine dump + Python coloring tail).
  * `composite_c17_smooth_curvature` — the Rust path, whose auto-level costs a SECOND engine
    render when the curve acts.
Identity is asserted over the PNGs **and** `autolevel_stamps.jsonl`: the stamp log is written
by the render itself on the serial path and by the parent on the concurrent one, so equal
pixels with an unequal log is still a failure — the record is half the product.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import location as loc_mod                                    # noqa: E402
from tools.emission import release_pass as RP                 # noqa: E402
from tools.palettes import autolevel as AL                    # noqa: E402
from tools.scoring.production_pins import auto_maxiter        # noqa: E402

pytestmark = pytest.mark.slow

# A shallow mandelbrot from the prod27 release (fw 8.4e-4), so the plan is a real framing and
# every row iterates in seconds rather than minutes. Geometry is deliberately NOT the wallpaper
# canon: this test is about identity, and identity does not need 2560x1440 ss4.
LOC = loc_mod.Location(family="mandelbrot", cx="-0.7490306292276605",
                       cy="-0.11542261923104817", fw="0.0008417568708862397",
                       maxiter=auto_maxiter(0.0008417568708862397))
GEOM = RP.Geom(640, 360, 2, "lanczos3")
PLAN = [("smooth", "BrBG"), ("smooth", "PRGn"), ("stripe", "BrBG"),
        ("composite_c17_smooth_curvature", "BrBG")]


def _tasks(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    return [RP.ReleaseTask(id=f"{i}_{style}_{pal}", loc=LOC, style=style, palette=pal,
                           out=str(out_dir / f"{i}_{style}_{pal}.png"))
            for i, (style, pal) in enumerate(PLAN)]


def _run(out_dir: Path, workers: int):
    tasks = _tasks(out_dir)
    order, results = [], {}

    def sink(task, res):
        order.append(task.id)
        results[task.id] = res
        if res.stamp is not None:                  # the parent's half of the contract
            AL.append_stamp(out_dir, Path(task.out).name, res.stamp)

    RP.run_pass(tasks, GEOM, workers=workers, sink=sink, log=lambda m: None)
    assert order == [t.id for t in tasks], "sink saw completion order, not plan order"
    bad = {i: r.error for i, r in results.items() if not r.ok}
    assert not bad, f"render failures: {bad}"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out_dir.iterdir()) if p.is_file()}


def test_serial_and_concurrent_release_passes_are_byte_identical(tmp_path):
    serial = _run(tmp_path / "serial", workers=1)
    conc = _run(tmp_path / "conc", workers=2)
    # NON-VACUITY: an empty or PNG-less product set would compare equal and prove nothing.
    assert len([k for k in serial if k.endswith(".png")]) == len(PLAN)
    assert AL.STAMP_LOG in serial, "the switch was off — this ran without the record half"
    assert conc == serial
