#!/usr/bin/env python
"""Tests for the maneuver richness screen (`maneuver_screen.py`).

Mostly pure — the cap-policy arithmetic, the reachability predicate, the running
distribution and the bounded prior term are all numpy-free logic. Three tests spawn the
real engine (like `tools/orbital/test_orbital.py::test_measure_keeps_no_field_files`) and
so need `target/release/fractal-generator.exe`; they are NOT skip-guarded, because an
absence-tolerant guard un-guards exactly when its subject is removed
(`docs/design/verification_practice.md` §2).

What each one is defending against is named in its docstring — the point of every one is
that the screen could otherwise be measuring the CAP instead of the field.

  uv run pytest tools/atlas/test_maneuver_screen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import maneuver_screen as msc     # noqa: E402
import field_metrics as fm        # noqa: E402
import render_core as rc          # noqa: E402

# fw values spanning shallow -> past where the screen policy clamps.
FWS = [3.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-13]


# =========================================================================== #
# the cap policy
# =========================================================================== #
def test_the_policy_closed_form_reproduces_production():
    """A SECOND COPY of a production closed form is `verification_practice.md` §1.8's
    defect verbatim, so the copy is pinned to the original rather than trusted. Fed the
    live constants, `maxiter_under` must BE `rc.auto_maxiter`."""
    live = (rc.MAXITER_BASE, rc.MAXITER_K, rc.MAXITER_MIN, rc.MAXITER_MAX)
    got = [msc.maxiter_under(live, fw) for fw in FWS]
    assert got == [rc.auto_maxiter(fw) for fw in FWS]
    # not vacuous: the inputs really do move, so this is not 10 copies of one clamp
    assert len(set(got)) >= 5, got


def _legacy_envelope_longhand(fw: float) -> float:
    """The legacy `(500, 0.30, 200, 8000)` envelope written out by hand, unclamped.

    Deliberately NOT `msc.maxiter_under` — an expectation derived from the helper under
    test asserts `f(x) == f(x)` (`verification_practice.md` §1.10). This test was written
    that way first and stayed GREEN when the exponent was perturbed to 0.31."""
    import math
    return 500.0 * (1.0 + 0.30 * math.log2(3.0 / fw))


def test_the_screen_policy_is_24x_the_legacy_envelope_until_the_clamp_binds():
    """The policy IS the retired 24x-of-legacy envelope (un-retired for this narrow use),
    not "a bigger number". Bracketed on both sides: exactly 24x where the clamp is slack,
    the ceiling where it binds — and BOTH regimes must appear in the sample, or the test is
    only checking one branch."""
    legacy, screen = msc.LEGACY_MAXITER_POLICY, msc.SCREEN_MAXITER_POLICY
    assert (legacy[0], legacy[1], legacy[2], legacy[3]) == (500, 0.30, 200, 8000)
    assert screen[0] == 24 * legacy[0] and screen[1] == legacy[1]
    assert screen[2] == 24 * legacy[2]
    slack = clamped = 0
    for fw in FWS:
        got = msc.screen_maxiter(fw)
        if got >= screen[3]:
            clamped += 1
            assert got == screen[3]
            continue
        slack += 1
        want = max(screen[2], 24.0 * _legacy_envelope_longhand(fw))
        assert abs(got - want) <= 1, (fw, got, want)
    assert slack >= 2 and clamped >= 2, (slack, clamped)


def test_the_screen_cap_is_strictly_above_production_at_every_depth():
    """The whole point of a separate policy is headroom above the cap the field would
    otherwise be clipped by. If it were ever BELOW production the screen would be measuring
    a tighter cap than the render path — silently, and in the wrong direction."""
    for fw in FWS:
        assert msc.screen_maxiter(fw) > rc.auto_maxiter(fw), fw


def test_the_screen_policy_gets_its_own_token_and_is_never_the_legacy_blank():
    """`fm.record_policy` reads a MISSING/empty token as legacy, so a screen score that
    stamped the empty string would be silently poolable with every pre-raise
    `data/orbital/` record. It must key disjointly from both legacy and live."""
    tok = msc.screen_policy_token()
    assert tok and tok != fm.LEGACY_POLICY_TOKEN
    assert tok != fm.policy_token()                       # != the live production policy
    with pytest.raises(fm.MaxiterPolicyMixError):
        fm.require_one_policy(("screen", [{fm.POLICY_KEY: tok}]),
                              ("orbital", [{fm.POLICY_KEY: fm.LEGACY_POLICY_TOKEN}]),
                              what="a screen score against a committed orbital score")


# =========================================================================== #
# reachability
# =========================================================================== #
def test_is_screenable_brackets_the_render_one_spacing_guard():
    """`render-one` refuses `fw/width <= 1e-13`. The predicate exists so the deep tail
    costs a comparison instead of a process spawn and a parsed stderr — so it has to agree
    with the guard at the boundary, from both sides."""
    at_wall = 1e-13 * fm.SCREEN_W / msc.SCREEN_FRAME_MULT
    assert msc.is_screenable(at_wall) is False            # exactly at the guard: refused
    assert msc.is_screenable(at_wall * 1.001) is True
    assert msc.is_screenable(0) is False


def test_screen_atom_below_the_wall_returns_a_named_row_not_an_exception():
    """RECORDING, NEVER A GATE: an unreachable nucleus is data. It must come back as a row
    carrying the policy token and a named reason, with no measures — an exception here
    would make the screen's reach a run-stopping condition."""
    at_wall = 1e-13 * fm.SCREEN_W / msc.SCREEN_FRAME_MULT
    r = msc.screen_atom("-0.5", "0.0", at_wall / 10)
    assert r["screened"] is False
    assert r["screen_reason"] == "f64_spacing_wall_at_screen_geometry"
    assert r[fm.POLICY_KEY] == msc.screen_policy_token()
    assert "radial_range" not in r and "radial_rings" not in r


# =========================================================================== #
# the running distribution and the bounded prior
# =========================================================================== #
def test_percentile_is_exactly_neutral_until_the_run_has_enough_to_rank():
    """Below `n_min` a "percentile" over 3 values is noise wearing a rank. It returns
    exactly 0.5, which `range_prior_delta` maps to exactly 0.0 — so an early batch gets the
    UNCHANGED neutral prior rather than a random shove."""
    d = msc.RangeDistribution(n_min=8)
    for v in (1.0, 5.0, 9.0):
        d.add(v)
    assert d.percentile_of(9.0) == 0.5
    assert msc.range_prior_delta(d.percentile_of(9.0), 0.5) == 0.0
    for v in range(10):
        d.add(float(v))
    assert d.percentile_of(100.0) == pytest.approx(1.0)
    assert d.percentile_of(-1.0) == pytest.approx(0.0)
    mid = d.percentile_of(5.0)
    assert 0.0 < mid < 1.0
    # not vacuous: the instrument distinguishes more than two states
    assert len({d.percentile_of(v) for v in (0.5, 2.5, 5.5, 8.5)}) == 4


def test_percentile_ignores_none_and_non_finite_rather_than_poisoning_the_run():
    d = msc.RangeDistribution(n_min=1)
    d.add(None)
    d.add(float("nan"))
    assert d.values == []
    assert d.percentile_of(None) == 0.5
    assert d.percentile_of(float("inf")) == 0.5


def test_the_prior_term_is_bounded_and_cannot_reach_a_well_scored_ordinary_node():
    """The design constraint from `minibrot_maneuvers.md` §3, as arithmetic: a maneuver
    out-competes a SCORED node via the quota floor, never via the prior. An ordinary node's
    `cheap_eord` spans [0, K-1] = [0, 3] on the K=4 head, and the best possible maneuver
    prior must stay well under that."""
    import steered_frontier as sf
    gain = sf.MAN_RANGE_GAIN_DEFAULT
    best = sf.NEUTRAL_PRIOR + msc.range_prior_delta(1.0, gain)
    worst = sf.NEUTRAL_PRIOR + msc.range_prior_delta(0.0, gain)
    assert best == pytest.approx(sf.NEUTRAL_PRIOR + gain / 2)
    assert worst == pytest.approx(sf.NEUTRAL_PRIOR - gain / 2)
    assert best < 3.0 - 1.0, "the top-ranked maneuver must not approach a top ordinary node"
    # symmetric about the neutral prior: the flag REORDERS maneuvers, it does not inflate
    # them as a class.
    assert (best + worst) / 2 == pytest.approx(sf.NEUTRAL_PRIOR)


def test_range_distribution_state_round_trips():
    d = msc.RangeDistribution(n_min=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        d.add(v)
    e = msc.RangeDistribution()
    e.load_state(d.state_dict())
    assert e.values == d.values and e.n_min == d.n_min
    assert e.percentile_of(3.5) == d.percentile_of(3.5)


# =========================================================================== #
# the screen itself — these spawn the engine
# =========================================================================== #
@pytest.fixture(scope="module")
def atom():
    """A REAL atom record, solved by the operator under test's own probe rather than a
    hand-typed `window_scale`. A fictional window scale is how a screen fixture ends up
    framing pure interior black and scoring a legitimate 0.0 — which reads identically to
    a broken screen (`verification_practice.md` §6, the fixture that cannot fail)."""
    import minibrot_maneuvers as mnv
    m = mnv.snap_to_nucleus(dict(node_id=1, cx=-0.7463, cy=0.1102, fw=1e-3, depth=3),
                            4.0, degree=2)
    assert m.available, m.reason
    return dict(cx=m.cx, cy=m.cy, window_scale=m.window_scale, period=m.period)


def test_the_screen_measures_the_field_and_stamps_its_cap(atom):
    r = msc.screen_atom(atom["cx"], atom["cy"], atom["window_scale"])
    assert r["screened"] is True, r.get("screen_reason")
    assert r["radial_range"] > 0 and r["radial_rings"] > 0
    assert r[fm.POLICY_KEY] == msc.screen_policy_token()
    assert r["screen_fw"] == pytest.approx(atom["window_scale"] * msc.SCREEN_FRAME_MULT)
    assert r["screen_maxiter"] == msc.screen_maxiter(r["screen_fw"])
    # the cap-headroom column is the thing that makes "non-clipping" a MEASUREMENT rather
    # than an assertion; it has to actually be populated.
    assert 0.0 < r["cap_headroom"] <= 1.0


def test_the_screen_measures_THE_4x_FRAME_and_not_some_other_one(atom):
    """DIFFERENTIAL against the reference path, because every other test here checks
    plumbing. `screen_atom`'s numbers must equal `rescore_lib.ring_measures` on a field
    dumped independently at 4x the ATOM — so a screen that quietly framed at `k x
    window_scale`, or at the parent's fw, or at 1x, is caught by its VALUES rather than by
    an `screen_fw` field it also computes itself.

    Non-vacuity is the second half: the 1x frame must give different numbers, or this would
    pass on any frame at all.

    The `4.0` is a LITERAL, not `msc.SCREEN_FRAME_MULT`. Written the derived way first, this
    test stayed green when the constant was moved to 8.0 — it pinned "the screen frames at
    whatever it says it frames at", which is `f(x) == f(x)` (`verification_practice.md`
    §1.10). 4x is the only frame scale ANY orbital score has been computed at and the only
    one `orbital_field_metrics.md` §2's validation covers, so it is the thing to pin."""
    import tempfile
    import rescore_lib as rl
    assert msc.SCREEN_FRAME_MULT == 4.0, "the validated frame scale (orbital §2)"
    want_fw = atom["window_scale"] * 4.0
    got = msc.screen_atom(atom["cx"], atom["cy"], atom["window_scale"])
    assert got["screened"] is True

    def measure_at(fw):
        with tempfile.TemporaryDirectory() as td:
            field, _ = fm.dump_field(atom["cx"], atom["cy"], fw, msc.screen_maxiter(fw),
                                     Path(td) / "f.bin", width=fm.SCREEN_W,
                                     height=fm.SCREEN_H, ss=fm.SCREEN_SS, threads=1)
        return rl.ring_measures(field)

    ref = measure_at(want_fw)
    assert got["radial_range"] == pytest.approx(ref["radial_range"], abs=1e-4)
    assert got["radial_rings"] == pytest.approx(ref["radial_rings"], abs=1e-2)
    # ... and a different frame really would have shown up
    other = measure_at(atom["window_scale"] * 1.0)
    assert (other["radial_range"], other["radial_rings"]) != \
           (ref["radial_range"], ref["radial_rings"])


def test_the_screen_actually_moves_when_the_cap_moves(atom):
    """`measurement_practice.md`: before pre-registering a bar, verify the instrument's
    inputs change. If the 24x policy produced the same field as a tiny cap, the whole
    stamping apparatus would be ceremony around a measurement of nothing."""
    import tempfile
    import rescore_lib as rl
    fw = atom["window_scale"] * msc.SCREEN_FRAME_MULT
    hi_cap = msc.screen_maxiter(fw)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for cap in (60, hi_cap):
            field, _ = fm.dump_field(atom["cx"], atom["cy"], fw, cap,
                                     Path(td) / f"f{cap}.bin", width=fm.SCREEN_W,
                                     height=fm.SCREEN_H, ss=fm.SCREEN_SS, threads=1)
            out[cap] = rl.ring_measures(field)
    lo, hi = out[60], out[hi_cap]
    assert (lo["radial_range"], lo["radial_rings"]) != (hi["radial_range"], hi["radial_rings"])


def test_the_cache_screens_one_nucleus_once_however_many_k_rows_ask(atom):
    """One field per NUCLEUS, shared across k rows — §7.1's shared solve extended to the
    screen. The frame is 4x the ATOM, so it cannot depend on k; a per-row screen would pay
    the spawn once per framing."""
    c = msc.ScreenCache()
    job = dict(atom_key="k1", family="mandelbrot",
               cx=atom["cx"], cy=atom["cy"], window_scale=atom["window_scale"])
    got = c.screen_many([job, dict(job), dict(job)])         # three k rows, one atom
    assert set(got) == {"k1"} and c.n_screened == 1 and c.n_hits == 0
    again = c.screen_many([dict(job)])
    assert c.n_screened == 1 and c.n_hits == 1
    assert again["k1"] == got["k1"]

    d = msc.ScreenCache()
    d.load_state(c.state_dict())
    before = d.n_screened
    d.screen_many([dict(job)])
    assert d.n_screened == before and d.n_hits == c.n_hits + 1, \
        "a resume must not re-spawn the engine for a nucleus the killed run screened"


def test_the_screen_pass_is_bounded_and_says_so_when_it_runs_out(atom):
    """The walk checks its active-time cap BETWEEN batches, so anything unbounded inside a
    batch is outside the cap. With a 60 s field timeout per screen and a fat neighbourhood
    enumeration, an unbudgeted pass could spend half an hour here while the budget logic
    believed it was inside its cap — "a backstop longer than the job's budget is not a
    backstop", one level down.

    A job the pass never reached must come back as a NAMED unscreened row, not as a missing
    key: `_nbh_top_n` and the quota sort both read these records, and a silent absence would
    read as "screened, range 0" at both."""
    c = msc.ScreenCache(workers=1)
    jobs = [dict(atom_key=f"k{i}", family="mandelbrot", cx=atom["cx"], cy=atom["cy"],
                 window_scale=atom["window_scale"]) for i in range(6)]
    got = c.screen_many(jobs, budget_s=1e-6)          # already spent before the first job
    assert len(got) == 6, "every job is accounted for, none silently dropped"
    assert all(r["screened"] is False for r in got.values())
    assert all(r["screen_reason"] == "screen_budget_exhausted" for r in got.values())
    assert all(r[fm.POLICY_KEY] == msc.screen_policy_token() for r in got.values())
    # ... and the same jobs DO screen with a real budget, so the test is not passing merely
    # because this fixture cannot be screened at all.
    d = msc.ScreenCache(workers=1)
    ok = d.screen_many(jobs[:1], budget_s=120.0)
    assert next(iter(ok.values()))["screened"] is True


def test_an_unbudgeted_screen_pass_still_works(atom):
    """`budget_s=None` is the plain path — a caller outside the walk (a bench, a readout)
    must not have to invent a budget to use the screen."""
    c = msc.ScreenCache(workers=1)
    got = c.screen_many([dict(atom_key="k", family="mandelbrot", cx=atom["cx"],
                              cy=atom["cy"], window_scale=atom["window_scale"])])
    assert got["k"]["screened"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
