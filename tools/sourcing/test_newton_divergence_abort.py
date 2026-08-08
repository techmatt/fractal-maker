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

TWO WAYS THE REFERENCE ARM IS SUPPLIED. The default-lane grids run BOTH arms live — they are
small, and they are what keeps the pinned table below honest about the reference still being
reproducible at all. The slow-lane grids (26,624 + 4,992 solves) read the reference from
`data/sourcing/newton_parity_ref.json` instead. The reference is a reimplementation of code
that no longer exists, so for a fixed (seed, period, degree, max_steps, dps) its outcome is a
constant; re-deriving 31,616 of them every run was 80.6% of the entire `slow` lane (measured
2026-08-08 — the live arm, which is the code under test, was 13.1%). Pinning them changes no
evidence: `lost == 0` is still checked against the reference's real verdict, because that
verdict is exactly what the table holds. Rationale in full: `tools/sourcing/newton_parity.py`.
Rebuild the table with `tools/sourcing/build_newton_parity_ref.py`.

Run: uv run pytest tools/sourcing/test_newton_divergence_abort.py -q
     uv run pytest tools/sourcing/test_newton_divergence_abort.py -m slow   # + the full grid
"""
from __future__ import annotations

import concurrent.futures as cf
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
import newton_parity as npar                            # noqa: E402

ROSTER = ROOT / "data" / "minibrot_roster" / "roster.jsonl"

# Parallel width for the pinned slow grids. These are ~50 MB pure-mpmath processes, one core
# each — NOT the heavyweight `fractal-generator.exe` that CLAUDE.md's 4-process cap is
# written against (own rayon pool, resident LUTs, corpus scan), so the cap does not bind
# here. Measured 2026-08-08 on the 12-core box, 312-solve grid: 2.92x at 4, 4.01x at 8 (that
# grid is only 24 jobs, so load imbalance dominates; the pinned grids are 2,048 jobs and
# scale better). Matches `tools/v8/render_cache.py`'s WORKERS=6 carve-out in kind.
PARITY_WORKERS = 8


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
def _tally(live_conv, live_iters, live_dig, ref_conv, ref_iters, ref_dig, acc):
    """THE bucketing rule, shared by the live-both-arms and the pinned-reference paths so
    the two cannot drift. `acc` is a 5-list [n_conv, lost, mismatch, aborted, gained].

    `gained` is not decoration. The original bucketing had `elif got.iters < exp.iters` as
    the sole non-converger branch, so a solve where the LIVE arm converged and the reference
    did not fell into it and was silently counted as a successful abort. That is impossible
    if the two arms differ only by an early abort — which is exactly why it should be an
    assertion rather than an unreachable branch nobody wrote."""
    if ref_conv:
        acc[0] += 1
        if not live_conv:
            acc[1] += 1                      # lost: the abort killed a real converger
        elif live_dig != ref_dig:
            acc[2] += 1                      # mismatch: converged, but to a different value
    elif live_conv:
        acc[4] += 1                          # gained: live converged where the reference did not
    elif live_iters < ref_iters:
        acc[3] += 1                          # aborted: the guard cut the budget short
    return acc


def _grid_parity(n_ang, n_rad, degrees, max_steps=None):
    """Run BOTH arms live over the roster's own ring-seed x period Newton grid and return
    (n_solves, n_conv, lost, mismatched, aborted, gained).

    Used by the default-lane grids, which are small. The slow lane runs the same comparison
    against a pinned reference instead — see `_grid_parity_pinned`."""
    max_steps = brs.NEWTON_STEPS if max_steps is None else max_steps
    n = 0
    acc = [0, 0, 0, 0, 0]
    for deg in degrees:
        for sr, si in brs.ring_seeds(deg, n_ang, n_rad):
            seed = mp.mpc(sr, si)
            for p in brs.PERIODS:
                got = dcf.newton_nucleus(seed, p, degree=deg, max_steps=max_steps)
                exp = ref_newton_nucleus(seed, p, degree=deg, max_steps=max_steps)
                n += 1
                _tally(got.converged, got.iters, npar.result_digest(got),
                       exp.converged, exp.iters, npar.result_digest(exp), acc)
    return (n, *acc)


# --------------------------------------------------------------------------- #
# The pinned-reference path (slow lane). The reference arm is a reimplementation of code
# that no longer exists, so its outcome for a fixed (seed, period, degree, max_steps, dps)
# is a constant; `newton_parity.py`'s docstring has the full argument. Only the LIVE arm —
# the code actually under test — runs here, and it runs in parallel.
# --------------------------------------------------------------------------- #
def _live_init():
    """Spawn-safe worker setup. The autouse `_dps` fixture does not reach a subprocess, so
    a worker that skipped this would solve at mpmath's default dps and mismatch everything."""
    import mpmath as _mp
    import build_minibrot_roster as _brs
    _mp.mp.dps = _brs.NUCLEUS_DPS


def _live_job(args):
    """One (degree, seed) column of the LIVE arm -> one row per period, in PERIODS order."""
    deg, _idx, sr, si, max_steps = args
    seed = mp.mpc(sr, si)
    return [npar.row_of(dcf.newton_nucleus(seed, p, degree=deg, max_steps=max_steps))
            for p in brs.PERIODS]


def _grid_parity_pinned(n_ang, n_rad, max_steps, *, workers=PARITY_WORKERS):
    """Same tally as `_grid_parity`, with the reference arm read from the pinned table.
    Returns (tally, offenders) where offenders localizes lost/mismatch/gained solves."""
    doc = npar.load()
    assert not doc.get("incomplete", False), (
        f"{npar.fixture_path()} was written by a bounded `--limit` run and covers only part "
        f"of the grid; rebuild it unbounded before trusting this test")
    key = npar.grid_key(n_ang, n_rad, max_steps)
    assert key in doc["grids"], f"no pinned grid {key!r} in {npar.fixture_path()}"
    g = doc["grids"][key]
    pinned = g["rows"]

    jobs = npar.grid_jobs(n_ang, n_rad, brs.DEGREES)
    payload = [(deg, idx, sr, si, max_steps) for deg, idx, sr, si in jobs]
    live = []
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_live_init) as ex:
        for out in ex.map(_live_job, payload, chunksize=1):
            live.extend(out)

    assert len(live) == len(pinned), (
        f"live produced {len(live)} solves, the pinned table has {len(pinned)} — the grid "
        f"enumeration moved out from under the fixture; rebuild it")

    acc = [0, 0, 0, 0, 0]
    offenders = []
    n_per = len(brs.PERIODS)
    for i, ((lc, li, ld), (pc, pi, pd)) in enumerate(zip(live, pinned)):
        before = tuple(acc)
        _tally(bool(lc), li, ld, bool(pc), pi, pd, acc)
        # lost / mismatch / gained moved -> record which solve, so a flip is localized
        if (acc[1], acc[2], acc[4]) != (before[1], before[2], before[4]):
            deg, idx, _sr, _si = jobs[i // n_per]
            offenders.append(f"deg={deg} seed={idx} period={brs.PERIODS[i % n_per]} "
                             f"live=(conv={bool(lc)},iters={li}) "
                             f"pinned=(conv={bool(pc)},iters={pi})")
    return (len(live), *acc), offenders


# The default-lane grids are 2 ang x 2 rad. Their cost is essentially LINEAR IN THE
# NUMBER OF NON-CONVERGERS — the live arm aborts on the first orbit pass, so what is
# being paid for is the reference arm burning its whole `max_steps` on each one
# (~0.22 s per non-converger at 600). Measured 2026-08-03 on the 12-core box, via
# `_grid_parity(n_ang, 2, DEGREES)` at max_steps=600:
#
#   ang=6  45.7s  624 solves  433 conv  191 nonconv  191 aborted   <- was the default
#   ang=4  29.4s  416         291       125          125
#   ang=3  20.8s  312         216        96           96
#   ang=2  13.0s  208         145        63           63           <- now
#
# `lost` and `mismatch` are 0 and `aborted` == nonconv at EVERY density, i.e. the abort
# fires on 100% of non-convergers and the population is homogeneous with respect to
# everything this test asserts. Denser grids buy more solves of the same kind, not a new
# kind of evidence, and the committed 64 x 8 density is what the `slow` lane below is
# for. n_rad must stay >= 2: at n_rad=1 every seed converges (0 non-convergers), the
# abort never fires and both tests go vacuous.
#
# That cost model is ALSO why the slow lane pins its reference arm rather than shrinking its
# grid: the expensive half is the REFERENCE burning its full budget on non-convergers (80.6%
# of that test, measured 2026-08-08), and that half is a constant. Cutting the density would
# delete the one thing the slow lane adds over these two grids.
GRID_ANG, GRID_RAD = 2, 2


def test_roster_ring_seed_grid_loses_no_converger_and_aborts_real_burners():
    """The derivation-shaped population: the roster builder's own ring seeds x periods
    3..15, at a reduced ring count so it stays a default-lane test. Zero convergers lost,
    zero converged solves changed, and the abort demonstrably fires — the last clause is
    what stops this passing because nothing happened."""
    n, n_conv, lost, mismatch, aborted, gained = _grid_parity(GRID_ANG, GRID_RAD, brs.DEGREES)
    assert n_conv > 0 and n - n_conv > 0, f"need both populations, got {n_conv}/{n}"
    assert lost == 0, f"{lost} converged solves lost of {n_conv}"
    assert mismatch == 0, f"{mismatch} converged solves changed value of {n_conv}"
    assert gained == 0, f"{gained} solves converged live but not in the reference arm"
    assert aborted > 0, "the abort never fired — this grid proves nothing"


def test_the_grid_stays_parity_clean_at_the_library_default_budget():
    """The same grid at max_steps=200, the default `scan`/CLI/emit_deep_pool budget —
    a TIGHTER budget than the roster's own `NEWTON_STEPS` (600) that the sibling above
    runs. That matters because the bound is proportional to the remaining budget: the
    same seed must be judged against 200 steps' worth of retirable residual here and 600
    there, so a bound that had quietly become a magnitude threshold would show up as a
    converger lost at exactly one of the two budgets."""
    n, n_conv, lost, mismatch, aborted, gained = _grid_parity(GRID_ANG, GRID_RAD, brs.DEGREES,
                                                              max_steps=200)
    assert n_conv > 0 and n - n_conv > 0, f"need both populations, got {n_conv}/{n}"
    assert lost == 0, f"{lost} converged solves lost of {n_conv} at max_steps=200"
    assert mismatch == 0, f"{mismatch} converged solves changed value of {n_conv}"
    assert gained == 0, f"{gained} solves converged live but not in the reference arm"
    assert aborted > 0, "the abort never fired at max_steps=200"


SLOW_GRIDS = [
    (64, 8, brs.NEWTON_STEPS, 26624),      # the committed roster grid, production budget
    (24, 4, 200, 4992),                    # the same shape at the library default budget
]


def test_the_pinned_reference_table_covers_both_slow_grids():
    """Cheap integrity gate on the fixture itself (no solving) — the sibling of
    `test_guard_tripwire.test_fixture_is_the_canonical_81_20_set`, and for the same reason:
    a corrupt or truncated table is exactly what would make the pinned pass vacuous, and
    checking it costs nothing. Deliberately NOT `slow`.

    Both populations must be present in every grid: a table whose solves all converged
    would let `lost == 0` pass on a grid where the abort can never fire."""
    doc = npar.load()
    assert doc["incomplete"] is False, "the pinned table is from a bounded --limit run"
    assert doc["env"]["dps"] == brs.NUCLEUS_DPS
    assert doc["env"]["newton_steps"] == brs.NEWTON_STEPS
    for n_ang, n_rad, max_steps, expect_n in SLOW_GRIDS:
        g = doc["grids"][npar.grid_key(n_ang, n_rad, max_steps)]
        assert g["expect_n"] == expect_n
        assert len(g["rows"]) == expect_n, f"{len(g['rows'])} rows for {expect_n} solves"
        assert g["degrees"] == brs.DEGREES and g["periods"] == brs.PERIODS
        n_conv = sum(r[0] for r in g["rows"])
        assert n_conv == g["n_converged"]
        assert n_conv > 0 and expect_n - n_conv > 0, (
            f"grid {n_ang}x{n_rad} has {n_conv}/{expect_n} convergers — one population is "
            f"empty, so the parity assertions cannot fail")
        assert all(len(r) == 3 and len(r[2]) == 16 for r in g["rows"])


@pytest.mark.slow
@pytest.mark.parametrize("n_ang,n_rad,max_steps,expect_n", SLOW_GRIDS)
def test_full_roster_ring_seed_grid_is_parity_clean(n_ang, n_rad, max_steps, expect_n):
    """The differential at the roster's COMMITTED seed density (64 ang x 8 rad x 4 degrees
    x 13 periods) — the exact Newton grid `build_minibrot_roster.source_degree` walks — and
    again at the 200-step library default.

    The reference arm is READ FROM `data/sourcing/newton_parity_ref.json`, not re-derived:
    it is a reimplementation of code that no longer exists, so its outcome per solve is a
    constant, and re-deriving 31,616 constants was 80.6% of the whole `slow` lane (measured
    2026-08-08). The evidence is unchanged — `lost == 0` is still checked against the
    reference's real verdict, which is what the table holds. Only the LIVE arm runs here,
    and it runs across `PARITY_WORKERS` processes."""
    (n, n_conv, lost, mismatch, aborted, gained), offenders = _grid_parity_pinned(
        n_ang, n_rad, max_steps)
    assert n == expect_n, n
    assert lost == 0, (f"{lost} converged solves lost of {n_conv}:\n  "
                       + "\n  ".join(offenders[:8]))
    assert mismatch == 0, (f"{mismatch} converged solves changed value of {n_conv}:\n  "
                           + "\n  ".join(offenders[:8]))
    assert gained == 0, (f"{gained} solves converged live but not in the reference arm:\n  "
                         + "\n  ".join(offenders[:8]))
    assert aborted > 0.5 * (n - n_conv), f"abort fired on only {aborted} of {n-n_conv} failures"
