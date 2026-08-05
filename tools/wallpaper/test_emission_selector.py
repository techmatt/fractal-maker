#!/usr/bin/env python
"""Unit tests for the emission selector — the fractal-identity dedup (mirrors the
seeder's per-family near-dup) and the <=1/distinct-fractal selection guard.

  uv run pytest tools/wallpaper/test_emission_selector.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("emission_selector", HERE / "emission_selector.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["emission_selector"] = es
_spec.loader.exec_module(es)


def C(family, cx=float("nan"), cy=float("nan"), fw=float("nan"), c_re=None, c_im=None,
      loc="", fit=1.0, cell=0, pal="p", iid=""):
    return es.Candidate(location_id=loc, palette_id=pal, family=family, fitness=fit,
                        color_cell=cell, image_id=iid or loc, cx=cx, cy=cy, fw=fw,
                        c_re=c_re, c_im=c_im)


# --------------------------------------------------------------------------- #
# same_fractal — per-family identity
# --------------------------------------------------------------------------- #
def test_cplane_same_center_merges_inside_the_finer_frames_radius():
    # co-located to well inside 0.25*min(fw) = 2.5e-4 -> same place under the live rule.
    assert es.same_fractal(C("mandelbrot", 0.0, 0.0, 1e-3), C("mandelbrot", 1e-6, 0.0, 2.0))


def test_cplane_distant_centers_distinct():
    assert not es.same_fractal(C("mandelbrot", 5.0, 5.0, 1e-3), C("mandelbrot", 0.0, 0.0, 1e-3))


def test_cplane_radius_is_the_live_calibrated_pair_not_a_local_k():
    """The alignment, stated as the geometry it changed. Symmetric pair, fw 0.1 both:
    the retired 1.5*max(fw)=0.15 merged d=0.1; the calibrated 0.25*min(fw)=0.025 keeps it."""
    near = C("mandelbrot", 0.02, 0.0, 0.1)      # d=0.02 < 0.025 -> merge under both
    mid = C("mandelbrot", 0.1, 0.0, 0.1)        # d=0.10: retired merged, calibrated keeps
    base = C("mandelbrot", 0.0, 0.0, 0.1)
    assert es.same_fractal(near, base)
    assert not es.same_fractal(mid, base)


def test_cplane_branch_reads_the_owners_constants():
    """One-source, by behaviour: the boundary MOVES when the owner's pair moves, so the
    branch cannot be holding a private copy. Injection — monkeypatching the owner's K is
    exactly the edit a recalibration makes."""
    ps = es._seeder()
    base = C("mandelbrot", 0.0, 0.0, 0.1)
    far = C("mandelbrot", 0.1, 0.0, 0.1)
    assert not es.same_fractal(far, base)               # live 0.25 keeps the pair
    k0, s0 = ps.DEDUP_K, ps.DEDUP_SCALE
    try:
        ps.DEDUP_K, ps.DEDUP_SCALE = ps.RETIRED_DEDUP_K, ps.RETIRED_DEDUP_SCALE
        assert es.same_fractal(far, base)               # ...and the branch follows it
    finally:
        ps.DEDUP_K, ps.DEDUP_SCALE = k0, s0
    assert not es.same_fractal(far, base)
    # ...and the live pair really is the calibrated one, so the boundary above is not a
    # coincidence of two numbers that happen to agree.
    assert (ps.DEDUP_K, ps.DEDUP_SCALE) == (0.25, "min")


def test_cplane_keeps_a_deep_zoom_inside_a_wide_outcome():
    """The pair the adoption is ABOUT: a deep zoom sitting inside a wide outcome. The retired
    max-scaled radius (1.5*2.0 = 3.0) swallowed it; the min-scaled one (0.25*1e-3 = 2.5e-4)
    does not. This is the class of pair that now reaches emission at all."""
    wide = C("mandelbrot", 0.0, 0.0, 2.0)
    deep = C("mandelbrot", 0.4, 0.1, 1e-3)
    assert not es.same_fractal(wide, deep)


def test_julia_requires_c_match():
    # identical base-scale (0,0) viewport, DIFFERENT seed c -> distinct fractals.
    a = C("julia", 0.0, 0.0, 3.0, c_re=-0.77, c_im=-0.13)
    b = C("julia", 0.0, 0.0, 3.0, c_re=-0.62, c_im=-0.40)
    assert not es.same_fractal(a, b)


def test_julia_same_c_base_recolors_merge():
    # same seed c, base (0,0) view, comparable zoom (recolor siblings) -> merge.
    a = C("julia", 0.0, 0.0, 0.14, c_re=-0.779, c_im=-0.134)
    b = C("julia", 0.0, 0.0, 0.18, c_re=-0.779, c_im=-0.134)
    assert es.same_fractal(a, b)


def test_julia_same_c_deep_zoom_distinct():
    # same seed c, same (0,0) center, but a 45x-deeper zoom is a genuinely-distinct view.
    a = C("julia", 0.0, 0.0, 1.61, c_re=-0.744, c_im=0.126)
    b = C("julia", 0.0, 0.0, 0.0356, c_re=-0.744, c_im=0.126)
    assert not es.same_fractal(a, b)


def test_julia_same_c_far_viewport_distinct():
    # same seed c, viewports far apart in the z-plane -> distinct sub-locations.
    a = C("julia_multibrot3", 0.716, 0.629, 0.02, c_re=0.525, c_im=-0.144)
    b = C("julia_multibrot3", 0.569, -0.067, 0.011, c_re=0.525, c_im=-0.144)
    assert not es.same_fractal(a, b)


def test_phoenix_recolor_siblings_merge():
    a = C("phoenix", -0.444237, 0.838584, 0.078)
    b = C("phoenix", -0.443955, 0.844047, 0.080)
    assert es.same_fractal(a, b)


def test_phoenix_decade_zoom_not_over_collapsed():
    # nearby centers but a ~1000x zoom gap -> NOT the same fractal (the phoenix carve-out).
    a = C("phoenix", -0.375556, 0.551152, 0.382)
    b = C("phoenix", -0.411817, 0.540521, 3.7e-4)
    assert not es.same_fractal(a, b)


def test_no_geometry_falls_back_to_exact_key():
    assert es.same_fractal(C("mandelbrot", loc="K"), C("mandelbrot", loc="K"))
    assert not es.same_fractal(C("mandelbrot", loc="K1"), C("mandelbrot", loc="K2"))


def test_different_family_never_same():
    a = C("julia", 0.0, 0.0, 3.0, c_re=0.1, c_im=0.1)
    b = C("julia_multibrot3", 0.0, 0.0, 3.0, c_re=0.1, c_im=0.1)
    assert not es.same_fractal(a, b)


# --------------------------------------------------------------------------- #
# select() — the <=1/distinct-fractal guard
# --------------------------------------------------------------------------- #
_NOCAP = dict(palette_cap_frac=1e9, palette_family_cap=None)


def test_select_drops_recolor_dups_keeps_best():
    # three recolors of ONE julia fractal, each in a different color cell.
    cands = [
        C("julia", 0.0, 0.0, 0.15, c_re=-0.779, c_im=-0.134, cell=2, fit=3.0, iid="hi"),
        C("julia", 0.0, 0.0, 0.14, c_re=-0.779, c_im=-0.134, cell=4, fit=2.5, iid="mid"),
        C("julia", 0.0, 0.0, 0.18, c_re=-0.779, c_im=-0.134, cell=8, fit=2.0, iid="lo"),
    ]
    res = es.select(cands, grid=es.ColorGrid(), **_NOCAP)
    assert [c.image_id for c in res.picks] == ["hi"]          # only the best-fitness recolor
    assert res.report["n_dup_rejected"] == 2


def test_select_keeps_distinct_fractals():
    # two genuinely-distinct julia (different seed c) -> both kept.
    cands = [
        C("julia", 0.0, 0.0, 3.0, c_re=-0.779, c_im=-0.134, cell=2, fit=3.0, iid="a"),
        C("julia", 0.0, 0.0, 3.0, c_re=-0.62, c_im=-0.40, cell=4, fit=2.0, iid="b"),
    ]
    res = es.select(cands, grid=es.ColorGrid(), **_NOCAP)
    assert sorted(c.image_id for c in res.picks) == ["a", "b"]
    assert res.report["n_dup_rejected"] == 0


def test_select_keeps_a_pair_the_retired_c_plane_rule_merged():
    """§A's acceptance test, at SELECTION rather than at the predicate: a c-plane pair the
    1.5*max(fw) rule merged (d=0.4 < 3.0) and the calibrated 0.25*min(fw) rule keeps
    (d=0.4 > 2.5e-4) occupies two cells and emits TWO renders, not one.

    Under the retired rule the lower-fitness row was a dup-reject and its cell emptied out —
    exactly the deep-zoom-inside-a-wide-outcome the coordinate calibration exists to keep."""
    wide = C("mandelbrot", 0.0, 0.0, 2.0, cell=1, fit=3.0, iid="wide")
    deep = C("mandelbrot", 0.4, 0.1, 1e-3, cell=5, fit=2.0, iid="deep")
    res = es.select([wide, deep], grid=es.ColorGrid(), **_NOCAP)
    assert sorted(c.image_id for c in res.picks) == ["deep", "wide"]
    assert res.report["n_dup_rejected"] == 0


def test_select_no_regression_without_geometry():
    # geometry-free candidates must select exactly as the historical exact-key guard did.
    cands = [
        C("mandelbrot", loc="L1", cell=1, fit=3.0, iid="a"),
        C("mandelbrot", loc="L1", cell=2, fit=2.0, iid="b"),   # same exact loc -> dropped
        C("mandelbrot", loc="L2", cell=3, fit=1.0, iid="c"),
    ]
    res = es.select(cands, grid=es.ColorGrid(), **_NOCAP)
    assert sorted(c.image_id for c in res.picks) == ["a", "c"]
    assert res.report["n_dup_rejected"] == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
