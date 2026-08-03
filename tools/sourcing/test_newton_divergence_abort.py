"""The divergence abort in `deep_center_finder.newton_nucleus` — fires, and changes nothing else.

`newton_nucleus` is shared by every nucleus/roster/triage derivation in the tree, so the
abort carries two obligations and this file discharges both:

  RED  — it actually fires, proven by injection at a residual straddling the bound (period 1,
         where |z_1(c)| = |c|, so the straddle is exact and controlled to a factor of two),
         and proven to SCALE with the budget rather than being a fixed magnitude — the same
         seed aborts under a tight `max_steps` and survives under a generous one. A guard
         that never fired is untested; a budget-feasibility guard that never varied with the
         budget is a magnitude threshold wearing its name.
  GREEN— an ordinary converging solve is VALUE-IDENTICAL to the pre-abort implementation,
         proven differentially against an inline reimplementation of the original loop (the
         local pattern from `test_deep_center_finder_degree.py`: the expectation must not be
         computed by the code under test), including the seahorse p35 SLOW converger, which
         sits at |z_35| = 1e66 for 150 iterations and finishes at iteration 163. That solve
         is why the abort is phrased against the remaining budget: it falsified the flat
         magnitude threshold this started as.

The pre-registered bar was ZERO lost convergers. It is met on every population checked here
and on the 8,132-solve neighborhood-operator replay the rule was derived from.

Run: uv run pytest tools/sourcing/test_newton_divergence_abort.py -q
     uv run pytest tools/sourcing/test_newton_divergence_abort.py -m slow   # + the full grid
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mpmath as mp
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "sourcing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import deep_center_finder as dcf                        # noqa: E402
import build_minibrot_roster as brs                     # noqa: E402

ROSTER = ROOT / "data" / "minibrot_roster" / "roster.jsonl"


# --------------------------------------------------------------------------- #
# The reference: `newton_nucleus` as it was BEFORE the abort, reimplemented inline.
# Deliberately not a flag on the live function — a knob that switches the guard off is
# still the code under test supplying its own expectation.
# --------------------------------------------------------------------------- #
def ref_newton_nucleus(c0, period, *, degree=2, max_steps=200, tol_dps_margin=6):
    c = mp.mpc(c0)
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    residual = mp.inf
    it = 0
    for it in range(1, max_steps + 1):
        z, d = dcf._orbit(c, period, degree)
        residual = abs(z)
        if d == 0:
            break
        step = z / d
        c = c - step
        if abs(step) < tol and residual < tol:
            break
    z, _ = dcf._orbit(c, period, degree)
    residual = abs(z)
    conv = residual < tol
    return dcf.NewtonResult(c=c, converged=bool(conv), iters=it,
                            residual=float(mp.log10(residual)) if residual > 0 else -999.0,
                            kind="nucleus", period=period, degree=degree)


def _same(a, b):
    """Value identity of two NewtonResults at working precision."""
    return (a.converged == b.converged and a.iters == b.iters
            and a.c == b.c and a.residual == b.residual)


@pytest.fixture(autouse=True)
def _dps():
    """Production Newton precision (atom_lib and build_minibrot_roster both set 60)."""
    old = mp.mp.dps
    mp.mp.dps = brs.NUCLEUS_DPS
    yield
    mp.mp.dps = old


# --------------------------------------------------------------------------- #
# 1. the bound is a budget, derived — not a fitted constant
# --------------------------------------------------------------------------- #
def test_the_bound_is_the_budget_newton_could_retire():
    """Newton retires one nat of |z_p| per step far from the roots (degree-n Newton moves
    c -> c(1-1/n), so |P| shrinks by (1-1/n)^n -> 1/e). The bound is that rate, in bits,
    times the remaining steps, times the safety factor — no free constant."""
    assert dcf.NEWTON_BITS_PER_STEP == pytest.approx(1.0 / math.log(2), rel=1e-12)
    for rem in (1, 7, 59, 199):
        assert dcf.divergence_bound(rem, safety=1.0) == pytest.approx(rem / math.log(2))
        assert dcf.divergence_bound(rem) == pytest.approx(
            dcf.DIVERGENCE_SAFETY * rem / math.log(2))
    assert dcf.divergence_bound(0) == 0.0                 # no budget left, nothing to retire


def test_the_bound_is_proportional_to_the_remaining_budget():
    """The defining property. If this were flat the seahorse p35 solve would die (see
    `test_a_slow_converger_is_untouched_...` below)."""
    assert dcf.divergence_bound(100) == pytest.approx(10 * dcf.divergence_bound(10))
    assert dcf.divergence_bound(199) > dcf.divergence_bound(59)


# --------------------------------------------------------------------------- #
# 2. RED by injection — the straddle, and the budget scaling
# --------------------------------------------------------------------------- #
def _period1_seed(mag_bits):
    """A period-1 seed with |z_1(c)| = |c| = 2**mag_bits, so the bound can be straddled
    exactly: at period 1 the residual IS the seed magnitude."""
    return mp.mpc(mp.mpf(2) ** mag_bits, 0)


def test_the_abort_fires_on_an_injected_divergent_seed():
    """Above the bound: aborted on the very first orbit pass, where the reference burns
    real iterations on the identical input. The guard proven red."""
    bound = dcf.divergence_bound(60 - 1)
    c0 = _period1_seed(bound + 1)
    r = dcf.newton_nucleus(c0, 1, max_steps=60)
    assert r.converged is False
    assert r.iters == 1, "the abort must stop on the first orbit pass, not run the budget"
    assert ref_newton_nucleus(c0, 1, max_steps=60).iters > r.iters


def test_the_abort_does_not_fire_just_below_the_bound():
    """The other half of the straddle: one bit under the bound the solve proceeds and is
    value-identical to the reference. Without this, the test above would also pass for a
    guard that aborted everything."""
    c0 = _period1_seed(dcf.divergence_bound(60 - 1) - 1)
    r = dcf.newton_nucleus(c0, 1, max_steps=60)
    assert r.converged is True and r.c == 0
    assert _same(r, ref_newton_nucleus(c0, 1, max_steps=60))


def test_the_same_seed_aborts_on_a_tight_budget_and_survives_a_generous_one():
    """The rule is feasibility, not magnitude: a residual that cannot be retired in 10
    steps can be retired in 600, and the abort must agree. One seed, two budgets."""
    c0 = _period1_seed(dcf.divergence_bound(10 - 1) + 1)
    tight = dcf.newton_nucleus(c0, 1, max_steps=10)
    roomy = dcf.newton_nucleus(c0, 1, max_steps=600)
    assert tight.converged is False and tight.iters == 1
    assert roomy.converged is True
    assert _same(roomy, ref_newton_nucleus(c0, 1, max_steps=600))


def test_the_abort_fires_mid_solve_on_a_wild_seed():
    """Not only at the seed: a seed well outside M at a production period aborts a few
    iterations in, where the reference burns the whole budget. Both agree it did not
    converge, so no caller sees a different answer — only a cheaper one."""
    c0 = mp.mpc("3.5", "2.5")
    r = dcf.newton_nucleus(c0, 30, max_steps=brs.NEWTON_STEPS)
    ref = ref_newton_nucleus(c0, 30, max_steps=brs.NEWTON_STEPS)
    assert r.converged is False and ref.converged is False
    assert ref.iters == brs.NEWTON_STEPS, "the reference must be a budget burner here"
    assert r.iters < ref.iters


def test_an_aborted_result_reports_the_residual_that_tripped_it():
    """The abort skips the post-loop re-measure (c did not move), so `residual` is the
    log10 of the escaped |z_p| that fired it, not a stale or re-stepped value."""
    r = dcf.newton_nucleus(mp.mpc("3.5", "2.5"), 30, max_steps=brs.NEWTON_STEPS)
    assert r.residual * math.log(10) / math.log(2) > dcf.divergence_bound(
        brs.NEWTON_STEPS - r.iters)


# --------------------------------------------------------------------------- #
# 3. GREEN — ordinary solves are value-identical
# --------------------------------------------------------------------------- #
CONVERGING = [
    (("-0.1592", "1.0317"), 3, 2),
    (("-1.7548776662", "0.0"), 3, 2),
    (("-1.31", "0.0"), 4, 2),                # the real-axis dedup-noise fixture
    (("0.7", "0.3"), 4, 3),                  # the degree fixture's d3 anchor
    (("-0.748", "0.263"), 5, 4),             # ...d4
    (("-0.786", "0.365"), 5, 5),             # ...d5
    (("0.4", "0.6"), 5, 3),
    (("-0.6", "0.5"), 6, 4),
]


@pytest.mark.parametrize("seed,period,degree", CONVERGING)
def test_ordinary_solves_are_value_identical_to_the_reference(seed, period, degree):
    c0 = mp.mpc(mp.mpf(seed[0]), mp.mpf(seed[1]))
    exp = ref_newton_nucleus(c0, period, degree=degree, max_steps=brs.NEWTON_STEPS)
    assert exp.converged, "fixture must be a CONVERGING solve or it proves nothing"
    got = dcf.newton_nucleus(c0, period, degree=degree, max_steps=brs.NEWTON_STEPS)
    assert _same(got, exp), (got, exp)


def test_a_slow_converger_is_untouched_when_the_budget_can_hold_it():
    """The seahorse p35 fixture — |z_35| = 1e66 for 150 iterations, converging at 163 — is
    the solve that falsified a flat magnitude threshold. At the library default budget of
    200 it must be value-identical to the reference; at the production budget of 60 it
    cannot finish either way, and both arms must agree it did not."""
    c0 = mp.mpc("-0.7453", "0.1127")
    roomy_ref = ref_newton_nucleus(c0, 35, max_steps=200)
    assert roomy_ref.converged and roomy_ref.iters > 100, roomy_ref
    assert _same(dcf.newton_nucleus(c0, 35, max_steps=200), roomy_ref)
    tight_ref = ref_newton_nucleus(c0, 35, max_steps=60)
    assert not tight_ref.converged
    assert dcf.newton_nucleus(c0, 35, max_steps=60).converged is False


def _roster_rows():
    if not ROSTER.exists():
        raise AssertionError(
            f"{ROSTER} is missing — it is a tracked durable artifact, not an optional "
            f"input; rebuild with `uv run python tools/sourcing/build_minibrot_roster.py`")
    return [json.loads(l) for l in open(ROSTER, encoding="utf-8") if l.strip()]


def test_committed_roster_keys_round_trip_through_the_live_solver():
    """Bar (b): re-solve every committed roster atom from its own stored coordinates and
    re-form the sector-canonical dedup key. Both arms must produce the same solve AND the
    same key, and the key must match the stored one under the read-time canonicalization
    (`snapped_dedup_key`) that the consumers already apply — the raw stored key carries
    per-solve axis noise (1e-148 imaginary parts) that predates this change and is exactly
    what that helper exists to collapse.

    Deliberately NOT absence-tolerant: an empty roster would make this pass on nothing, so
    the loader raises and the count is asserted against the tracked file."""
    rows = _roster_rows()
    assert len(rows) >= 100, f"roster collapsed to {len(rows)} rows"
    for row in rows:
        deg, per = int(row["degree"]), int(row["period"])
        c0 = mp.mpc(mp.mpf(row["cx"]), mp.mpf(row["cy"]))
        got = dcf.newton_nucleus(c0, per, degree=deg, max_steps=brs.NEWTON_STEPS)
        exp = ref_newton_nucleus(c0, per, degree=deg, max_steps=brs.NEWTON_STEPS)
        assert _same(got, exp), (row["id"], got, exp)
        assert got.converged, row["id"]
        digits = dcf.emit_digits_for_fw(1e-20)
        stored = dcf.snapped_dedup_key(row["cx"], row["cy"], deg, brs.DEDUP_DPS)
        for arm, res in (("live", got), ("ref", exp)):
            cc = dcf.canonical_nucleus_c(res.c, deg)
            key = dcf.snapped_dedup_key(mp.nstr(cc.real, digits, strip_zeros=False),
                                        mp.nstr(cc.imag, digits, strip_zeros=False),
                                        deg, brs.DEDUP_DPS)
            assert key == stored, (row["id"], arm, key, stored)


# --------------------------------------------------------------------------- #
# 4. GREEN on the derivation that actually contains non-convergers
# --------------------------------------------------------------------------- #
def _grid_parity(n_ang, n_rad, degrees, max_steps=None):
    """Run both arms over the roster's own ring-seed x period Newton grid and return
    (n_solves, n_conv, lost, mismatched, aborted)."""
    max_steps = brs.NEWTON_STEPS if max_steps is None else max_steps
    n = n_conv = lost = mismatch = aborted = 0
    for deg in degrees:
        for sr, si in brs.ring_seeds(deg, n_ang, n_rad):
            seed = mp.mpc(sr, si)
            for p in brs.PERIODS:
                got = dcf.newton_nucleus(seed, p, degree=deg, max_steps=max_steps)
                exp = ref_newton_nucleus(seed, p, degree=deg, max_steps=max_steps)
                n += 1
                if exp.converged:
                    n_conv += 1
                    if not got.converged:
                        lost += 1
                    elif not _same(got, exp):
                        mismatch += 1
                elif got.iters < exp.iters:
                    aborted += 1
    return n, n_conv, lost, mismatch, aborted


def test_roster_ring_seed_grid_loses_no_converger_and_aborts_real_burners():
    """The derivation-shaped population: the roster builder's own ring seeds x periods
    3..15, at a reduced ring count so it stays a default-lane test. Zero convergers lost,
    zero converged solves changed, and the abort demonstrably fires — the last clause is
    what stops this passing because nothing happened."""
    n, n_conv, lost, mismatch, aborted = _grid_parity(6, 2, brs.DEGREES)
    assert n_conv > 0 and n - n_conv > 0, f"need both populations, got {n_conv}/{n}"
    assert lost == 0, f"{lost} converged solves lost of {n_conv}"
    assert mismatch == 0, f"{mismatch} converged solves changed value of {n_conv}"
    assert aborted > 0, "the abort never fired — this grid proves nothing"


def test_the_grid_stays_parity_clean_at_the_library_default_budget():
    """The same grid at max_steps=200, the default `scan`/CLI/emit_deep_pool budget. A
    bigger budget admits SLOW convergers the 60-step arm never sees, which is the exact
    population a budget-blind bound destroys. Kept small here (the reference arm burns the
    whole 200 on every failure); the dense version is in the `slow` lane below."""
    n, n_conv, lost, mismatch, aborted = _grid_parity(3, 2, brs.DEGREES, max_steps=200)
    assert lost == 0, f"{lost} converged solves lost of {n_conv} at max_steps=200"
    assert mismatch == 0, f"{mismatch} converged solves changed value of {n_conv}"
    assert aborted > 0, "the abort never fired at max_steps=200"


@pytest.mark.slow
@pytest.mark.parametrize("n_ang,n_rad,max_steps,expect_n", [
    (64, 8, brs.NEWTON_STEPS, 26624),      # the committed roster grid, production budget
    (24, 4, 200, 4992),                    # the same shape at the library default budget
])
def test_full_roster_ring_seed_grid_is_parity_clean(n_ang, n_rad, max_steps, expect_n):
    """The differential at the roster's COMMITTED seed density (64 ang x 8 rad x 4 degrees
    x 13 periods) — the exact Newton grid `build_minibrot_roster.source_degree` walks — and
    again at the 200-step library default. Opt-in because it is minutes of mpmath."""
    n, n_conv, lost, mismatch, aborted = _grid_parity(n_ang, n_rad, brs.DEGREES,
                                                      max_steps=max_steps)
    assert n == expect_n, n
    assert lost == 0, f"{lost} converged solves lost of {n_conv}"
    assert mismatch == 0, f"{mismatch} converged solves changed value of {n_conv}"
    assert aborted > 0.5 * (n - n_conv), f"abort fired on only {aborted} of {n-n_conv} failures"
