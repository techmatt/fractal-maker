"""Differential + zero-change proof for the extracted shared coordinate/render module.

Two guarantees the descent-harness extraction MUST hold:

1. **Coordinate differential** — `render_core.click_to_world` / `box_commit` /
   `auto_maxiter` reproduce a *frozen copy* of the original inline
   `tools/explorer/app.py` implementation, bit-for-bit, over a grid of clicks and
   frame widths. A silent divergence here would mislocate every emitted solution.

2. **Explorer zero-change** — the `render-one` argv the explorer builds after the
   extraction is byte-identical to the historical inline command line (Mandelbrot
   nav render *and* Julia nav render), so `tools/explorer/` behaves identically.

Run:  uv run python -m pytest tools/explorer/test_render_core.py -q
"""
import sys
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_core as rc  # noqa: E402

getcontext().prec = 60


# --------------------------------------------------------------------------- #
# Frozen reference: verbatim copies of the ORIGINAL inline implementations
# (git fa9ca42 tools/explorer/app.py, before the render_core extraction).
# --------------------------------------------------------------------------- #
def _ref_click_to_world(px, py, ctr_x, ctr_y, fw, w=700, h=394):
    fw = Decimal(fw)
    W, H = Decimal(w), Decimal(h)
    fh = fw * H / W
    fx = Decimal(px) / W - Decimal("0.5")
    fy = Decimal(py) / H - Decimal("0.5")
    world_x = Decimal(ctr_x) + fx * fw
    world_y = Decimal(ctr_y) - fy * fh
    return world_x, world_y


def _ref_auto_maxiter(fw, override=None):
    if override is not None:
        return int(override)
    import math
    fw = Decimal(fw)
    fw_home = Decimal("3.0")
    ratio = fw_home / fw if fw > 0 else Decimal(1)
    lz = math.log2(float(ratio)) if ratio > 0 else 0.0
    val = 500 * (1.0 + 0.30 * lz)
    return int(max(200, min(8000, val)))


# --------------------------------------------------------------------------- #
# 1. Coordinate differential
# --------------------------------------------------------------------------- #
def test_click_to_world_matches_reference_grid():
    centers = [
        (Decimal("-0.5"), Decimal("0")),
        (Decimal("-0.12256116687665361997524555182073565"),
         Decimal("-0.74486176661974423659317042860439237")),
        (Decimal("-1.7548776662466927600495088963585287"),
         Decimal("9.2329786177857357807589307624968334e-128")),
    ]
    fws = [Decimal("3.0"), Decimal("0.7556553"), Decimal("1e-6"), Decimal("2.5e-40")]
    for (cx, cy) in centers:
        for fw in fws:
            for px in (0, 1, 173, 349.5, 700):
                for py in (0, 197, 393, 394):
                    got = rc.click_to_world(px, py, cx, cy, fw, 700, 394)
                    exp = _ref_click_to_world(px, py, cx, cy, fw, 700, 394)
                    assert got == exp, (px, py, cx, cy, fw)


def test_auto_maxiter_matches_reference():
    for fw in ("3.0", "1.0", "0.75", "1e-3", "1e-6", "1e-20", "2.5e-40"):
        assert rc.auto_maxiter(Decimal(fw)) == _ref_auto_maxiter(Decimal(fw))
    # override wins verbatim
    assert rc.auto_maxiter(Decimal("1e-6"), 1234) == 1234


def test_box_commit_center_and_fw():
    # Center = mousedown world point; new_fw = 2*horizontal-radius in plane units.
    cx, cy, fw = Decimal("-0.5"), Decimal("0"), Decimal("3.0")
    W, H = 700, 394
    down_px, down_py, cur_px = 350.0, 197.0, 450.0
    ncx, ncy, nfw = rc.box_commit(down_px, down_py, cur_px, cx, cy, fw, W, H)
    # mousedown at the exact panel center → new center == old center
    assert ncx == _ref_click_to_world(down_px, down_py, cx, cy, fw, W, H)[0]
    assert ncy == _ref_click_to_world(down_px, down_py, cx, cy, fw, W, H)[1]
    # radius = |450-350| px = 100 px → 100/700 * 3.0 plane; fw = 2× that
    expected_fw = (Decimal("100") / Decimal("700")) * fw * 2
    assert nfw == expected_fw


def test_box_commit_symmetric_in_drag_direction():
    # Horizontal radius is |Δpx|; dragging left or right the same distance is identical.
    cx, cy, fw = Decimal("-0.5"), Decimal("0"), Decimal("3.0")
    a = rc.box_commit(350, 197, 500, cx, cy, fw, 700, 394)
    b = rc.box_commit(350, 197, 200, cx, cy, fw, 700, 394)
    assert a[2] == b[2]  # same fw


# --------------------------------------------------------------------------- #
# 2. Explorer zero-change: render-one argv byte-identity
# --------------------------------------------------------------------------- #
def test_mandelbrot_nav_argv_byte_identical():
    # Historical inline command (family default = mandelbrot, no --family flag).
    cx, cy, fw = Decimal("-0.5"), Decimal("0"), Decimal("3.0")
    out = "/tmp/x.png"
    expected = [
        str(rc.RENDER_BIN), "render-one",
        "--cx", rc.dec_str(cx), "--cy", rc.dec_str(cy), "--fw", rc.dec_str(fw),
        "--width", "700", "--height", "394", "--supersample", "1",
        "--palette", "twilight_shifted", "--colormaps", str(rc.CLEAN_COLORMAPS),
        "--maxiter", "500", "--out", out,
    ]
    got = rc.render_one_argv(cx, cy, fw, 500, 700, 394, 1,
                             "twilight_shifted", rc.CLEAN_COLORMAPS, out)
    assert got == expected


def test_julia_nav_argv_byte_identical():
    # Historical inline Julia command: --julia --c appended AFTER --out, no --family.
    jx, jy, jfw = Decimal("0"), Decimal("0"), Decimal("3.0")
    c = (Decimal("-0.5"), Decimal("0"))
    out = "/tmp/j.png"
    expected = [
        str(rc.RENDER_BIN), "render-one",
        "--cx", rc.dec_str(jx), "--cy", rc.dec_str(jy), "--fw", rc.dec_str(jfw),
        "--width", "700", "--height", "394", "--supersample", "1",
        "--palette", "twilight_shifted", "--colormaps", str(rc.CLEAN_COLORMAPS),
        "--maxiter", "500", "--out", out,
        "--julia", "--c", rc.dec_str(c[0]), rc.dec_str(c[1]),
    ]
    got = rc.render_one_argv(jx, jy, jfw, 500, 700, 394, 1,
                             "twilight_shifted", rc.CLEAN_COLORMAPS, out, julia_c=c)
    assert got == expected


if __name__ == "__main__":
    test_click_to_world_matches_reference_grid()
    test_auto_maxiter_matches_reference()
    test_box_commit_center_and_fw()
    test_box_commit_symmetric_in_drag_direction()
    test_mandelbrot_nav_argv_byte_identical()
    test_julia_nav_argv_byte_identical()
    print("render_core differential + zero-change: PASS")
