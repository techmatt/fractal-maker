"""Regression + sanity for the z^d+c generalization of deep_center_finder.

The generalization threads a `degree` parameter through the orbit/derivative
recurrences, the Newton solvers, and the size estimate. The degree-2 (Mandelbrot)
path is kept TEXTUALLY untouched (a `degree == 2` branch), so every d=2 result must
be byte-identical to the original quadratic code. These tests pin that:

  1. the degree=2 orbit/derivative/size branches reproduce a fresh inline
     reimplementation of the ORIGINAL quadratic expressions, exactly (mpmath ==);
  2. a known d=2 nucleus (seahorse p35) Newton-converges as before;
  3. d=3/4/5 nuclei converge to genuine minibrots (finite size in the f64 band).

Run: uv run pytest tools/sourcing/test_deep_center_finder_degree.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.sourcing.deep_center_finder as f  # noqa: E402


# --- reference: the ORIGINAL quadratic recurrences, reimplemented inline --------- #
def _ref_orbit(c, n):
    z = mp.mpc(0); d = mp.mpc(0)
    for _ in range(n):
        d = 2 * z * d + 1
        z = z * z + c
    return z, d


def _ref_orbit_at(c, k, n):
    z = mp.mpc(0); d = mp.mpc(0)
    zk = dk = None
    for i in range(k + n):
        if i == k:
            zk, dk = z, d
        d = 2 * z * d + 1
        z = z * z + c
    if k == 0:
        zk, dk = mp.mpc(0), mp.mpc(0)
    return zk, dk, z, d


def _ref_size(c, period):
    l = mp.mpc(1); b = mp.mpc(1); z = mp.mpc(0)
    for _ in range(1, period):
        z = z * z + c
        l = 2 * z * l
        if l == 0:
            return mp.mpc(0)
        b = b + 1 / l
    denom = b * l * l
    return mp.mpc(0) if denom == 0 else 1 / denom


CS = [mp.mpc("-0.12", "0.75"), mp.mpc("0.28", "-0.53"), mp.mpc("-0.743", "0.113")]


@pytest.mark.parametrize("c", CS)
def test_degree2_orbit_byte_identical(c):
    mp.mp.dps = 60
    for n in (1, 5, 17, 40):
        assert f._orbit(c, n, 2) == _ref_orbit(c, n)         # mpmath exact ==
    for (k, n) in [(0, 3), (1, 4), (5, 7), (8, 2)]:
        assert f._orbit_at(c, k, n, 2) == _ref_orbit_at(c, k, n)


@pytest.mark.parametrize("c", CS)
def test_degree2_size_byte_identical(c):
    mp.mp.dps = 60
    for p in (3, 5, 12, 20):
        assert f.nucleus_size_estimate(c, p, 2) == _ref_size(c, p)


def test_degree2_default_arg_is_degree2():
    # the degree kwarg DEFAULTS to 2 → the old call sites are unchanged.
    mp.mp.dps = 60
    c = CS[0]
    assert f._orbit(c, 10) == f._orbit(c, 10, 2)
    assert f.nucleus_size_estimate(c, 8) == f.nucleus_size_estimate(c, 8, 2)


def test_degree2_seahorse_nucleus_converges():
    mp.mp.dps = 60
    r = f.newton_nucleus(mp.mpc("-0.7453", "0.1127"), 35)     # default degree=2
    assert r.converged and r.degree == 2 and r.residual < -30


@pytest.mark.parametrize("degree,seed,period", [
    (3, (0.7, 0.3), 4),
    (4, (-0.748, 0.263), 5),
    (5, (-0.786, 0.365), 5),
])
def test_multibrot_nucleus_converges(degree, seed, period):
    mp.mp.dps = 60
    r = f.newton_nucleus(mp.mpc(seed[0], seed[1]), period, degree=degree)
    assert r.converged and r.degree == degree
    assert r.residual < -30
    size = f.nucleus_size_estimate(r.c, period, degree)
    assert size != 0 and 1e-12 < float(abs(size)) < 1e2      # a real, finite minibrot
