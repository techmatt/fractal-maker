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


# --- atom instrument `A`: the |A| ≡ 1/|size| identity (§2 build) ---------------- #
@pytest.mark.parametrize("degree,seed,period", [
    (2, (-0.7453, 0.1127), 35),      # d2 seahorse
    (2, (-0.1226, 0.7449), 3),       # d2 rabbit (shallow: n=3, identity still exact)
    (3, (0.7, 0.3), 4),
    (4, (-0.748, 0.263), 5),
    (5, (-0.786, 0.365), 5),
])
def test_atom_A_equals_inverse_size(degree, seed, period):
    """`A = Λ^(1/(d-1))·P_n'(c0)` is the same analytic quantity the size law
    computes: |A| ≡ 1/|size| and arg A ≡ −arg(size), to full precision at every n
    (n=3 included — the identity is exact, not asymptotic). Locks the §2 finding
    that `A` re-derives (and so confirms) the corrected d/(d-1) size exponent."""
    mp.mp.dps = 60
    r = f.newton_nucleus(mp.mpc(seed[0], seed[1]), period, degree=degree)
    assert r.converged
    inst = f.atom_instrument(r.c, period, degree)
    size = f.nucleus_size_estimate(r.c, period, degree)
    # |A|·|size| == 1 to ~50 digits (relative), branch-free in magnitude.
    assert abs(abs(inst.A) * abs(size) - 1) < mp.mpf(10) ** (-40)
    # arg A ≡ −arg(size) mod 2π (principal branches line up here).
    dphase = mp.arg(inst.A) + mp.arg(size)
    assert abs(mp.expjpi(dphase / mp.pi) - 1) < mp.mpf(10) ** (-30)
    # required precision scales with depth; window scale is the atom's inverse size.
    assert inst.required_dps >= 50
    assert inst.window_scale > 0


# --- symmetry-canonical dedup: rotational copies collapse to one key ------------- #
def test_dedup_key_degree2_is_plain_rounded_key():
    """d=2 is 1-fold symmetric → the fundamental sector is the whole plane, so the
    canonical key is byte-identical to the original (nstr(cx), nstr(cy)) rounded key.
    Pins that the d2 corpus dedup is unchanged."""
    mp.mp.dps = 60
    for c in CS:
        assert f.nucleus_dedup_key(c, 2, 22) == (mp.nstr(c.real, 22), mp.nstr(c.imag, 22))


@pytest.mark.parametrize("degree,seed,period", [
    (3, (0.7, 0.3), 4),
    (4, (-0.748, 0.263), 5),
    (5, (-0.786, 0.365), 5),
])
def test_rotational_copies_collapse_to_one_key(degree, seed, period):
    """z^d+c has (d−1)-fold rotational symmetry: c·ω^k (ω=exp(2πi/(d−1))) is the SAME
    atom under z→ωz. Each copy is independently a converged period-p nucleus of equal
    size — a genuine rotational family — yet rounded-coordinate dedup keeps them apart.
    The symmetry-canonical key collapses the whole family to ONE entry, so a clean
    per-degree distinct count is a guard, not luck (the multibrot-transfer read had
    d4=10/12, d5=8/12 distinct precisely from this leak; d3's 12/12 was luck)."""
    mp.mp.dps = 60
    r = f.newton_nucleus(mp.mpc(seed[0], seed[1]), period, degree=degree)
    assert r.converged
    omega = mp.expjpi(mp.mpf(2) / (degree - 1))          # exp(2πi/(d−1))
    size0 = abs(f.nucleus_size_estimate(r.c, period, degree))
    copies = [r.c * omega ** k for k in range(degree - 1)]   # the full rotational orbit
    canon_keys, raw_keys = set(), set()
    for ck in copies:
        # each copy is a genuine period-p nucleus of the same size (a real family)
        rk = f.newton_nucleus(ck, period, degree=degree, max_steps=40)
        assert rk.converged
        assert abs(abs(f.nucleus_size_estimate(ck, period, degree)) - size0) < 1e-9 * size0
        canon_keys.add(f.nucleus_dedup_key(ck, degree, 22))
        raw_keys.add((mp.nstr(ck.real, 22), mp.nstr(ck.imag, 22)))
    # the guard: the (d−1) copies collapse to one canonical key ...
    assert len(canon_keys) == 1, f"d{degree}: {degree-1} copies -> {len(canon_keys)} keys"
    # ... whereas the OLD rounded-coordinate key kept every copy separate.
    assert len(raw_keys) == degree - 1


def test_distinct_atoms_do_not_collapse():
    """The guard only quotients the rotational symmetry — it must not over-merge two
    genuinely different atoms (different |c|, or same |c| but not a symmetry rotation)."""
    mp.mp.dps = 60
    for degree in (3, 4, 5):
        diff_mag = (f.nucleus_dedup_key(mp.mpc("0.30", "0.10"), degree, 22)
                    != f.nucleus_dedup_key(mp.mpc("0.55", "0.42"), degree, 22))
        assert diff_mag
