#!/usr/bin/env python
"""Unit tests for the minibrot maneuver operators and their frontier seam.

Pure / fast — mpmath only, no render, no GPU, no binary. Covers the three things that
would be silently wrong otherwise:

  * the operators themselves: a known nucleus is found and framed by the `A` size law,
    and unavailability is returned CLEANLY (never raised) with a named reason;
  * the dedup key is the SHARED read-time canonicalization, so an atom found here
    collapses onto the same key/id as one found by a source sheet or the triage pool;
  * the reserved frontier floor: it is a quota OF AVAILABLE, it does not stall when the
    operator has nothing, and it is inert when maneuvers are off (byte-identical runs).

  uv run pytest tools/atlas/test_minibrot_maneuvers.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import minibrot_maneuvers as mnv    # noqa: E402
import steered_frontier as sf       # noqa: E402


# =========================================================================== #
# atom-domain probe
# =========================================================================== #
def test_period_candidates_yields_the_true_period_via_divisors_not_the_raw_argmin():
    # The "rabbit" period-3 nucleus at c = -0.12256117 + 0.74486177i. The nucleus is
    # SUPERATTRACTING, so the raw |z_k| argmin lands on a MULTIPLE of 3, never on 3 —
    # the candidate set must be divisor-closed and increasing, or Newton only ever gets
    # offered non-minimal periods.
    # A walk centre is never EXACTLY on a nucleus, so use the realistic rounded coordinate.
    c = complex(-0.1226, 0.7449)
    cands = mnv.period_candidates(c, 2, max_period=24, n=4)
    assert 3 in cands
    assert cands == sorted(cands) and cands[0] >= 2          # increasing, period 1 skipped
    # the raw argmin really is a multiple, i.e. this test is not vacuous
    z, best = 0j, None
    for k in range(1, 25):
        z = z * z + c
        if best is None or abs(z) < best[0]:
            best = (abs(z), k)
    assert best[1] % 3 == 0 and best[1] != 3


def test_period_candidates_truncates_at_escape():
    # A parameter well outside the set escapes immediately; the candidate list must be
    # short rather than a full max_period sweep of meaningless values.
    assert len(mnv.period_candidates(complex(3.0, 3.0), 2, max_period=64, n=8)) <= 2


# =========================================================================== #
# snap_to_nucleus
# =========================================================================== #
def test_snap_finds_the_period_3_nucleus_and_preserves_fw_at_k_none():
    view = dict(node_id=7, cx=-0.1226, cy=0.7449, fw=0.01, depth=5)
    m = mnv.snap_to_nucleus(view, None, degree=2)
    assert m.available, m.reason
    assert m.op == "snap_to_nucleus" and m.period == 3
    assert m.fw == view["fw"]                    # k=None preserves the parent frame
    assert m.depth == view["depth"]              # a reframe is not a rung
    assert m.parent_node_id == 7
    assert abs(float(m.cx) + 0.12256117) < 1e-6 and abs(float(m.cy) - 0.74486177) < 1e-6


def test_snap_k_frames_at_k_times_atom_size_not_the_lambda_squared_law():
    view = dict(node_id=1, cx=-0.1226, cy=0.7449, fw=0.01, depth=5)
    m4 = mnv.snap_to_nucleus(view, 4.0, degree=2)
    assert m4.available
    # fw == k / |A| exactly; window_scale IS 1/|A| (the corrected size law), and the
    # forbidden naive degree-2 lambda^2 law would put this somewhere else entirely.
    assert m4.fw == pytest.approx(4.0 * m4.window_scale, rel=1e-12)
    # log10_abs_A is stored rounded to 6 dp, so agreement is only to ~1e-5 relative.
    assert m4.window_scale == pytest.approx(10.0 ** (-m4.log10_abs_A), rel=1e-5)


def test_snap_unavailable_is_clean_and_named_never_raised():
    # Deep exterior: the orbit escapes at once, so there is no atom domain to snap to.
    m = mnv.snap_to_nucleus(dict(node_id=1, cx=3.0, cy=3.0, fw=0.01, depth=2), None)
    assert m.available is False and m.reason and m.cx is None
    assert m.probe_s >= 0.0


def test_snap_refuses_a_teleport_the_nucleus_must_be_inside_the_frame():
    # Same centre, but a frame so tight the (real) nearby nucleus is outside it: a snap that
    # jumped there would be a teleport, not a reframing of THIS view.
    view = dict(node_id=1, cx=-0.1226, cy=0.7449, fw=1e-9, depth=9)
    m = mnv.snap_to_nucleus(view, None, degree=2)
    assert m.available is False and m.reason == "nucleus_outside_frame"


def test_snap_refuses_a_frame_past_the_f64_pixel_spacing_wall():
    # A very deep atom framed at k x its size would put pixel spacing under PERTURB_SPACING
    # at the node width, where the f64 backend the descent uses quantizes. Predicted a
    # priori from |A|, with no render attempted.
    fw_at_wall = mnv.NODE_WIDTH * mnv.PERTURB_SPACING          # exactly at the wall
    assert mnv._wall_margin_decades(fw_at_wall, mnv.NODE_WIDTH) == pytest.approx(0.0)
    rec = dict(window_scale=1e-18, degree=2)
    fw, why = mnv._frame_for(rec, 4.0, 1e-3, mnv.MAX_FW)
    assert fw is None and why == "f64_spacing_wall"


def test_snap_refuses_a_frame_wider_than_a_root_view():
    fw, why = mnv._frame_for(dict(window_scale=100.0), 4.0, 1e-3, mnv.MAX_FW)
    assert fw is None and why == "fw_over_root_scale"


# =========================================================================== #
# lateral_to_sibling
# =========================================================================== #
def test_lateral_returns_a_different_atom_at_comparable_scale():
    view = dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4)
    m = mnv.lateral_to_sibling(view, np.random.default_rng(0), degree=2)
    assert m.available, m.reason
    assert m.op == "lateral_to_sibling"
    assert m.atom_id != m.extra["parent_atom_id"]              # a sibling, not the parent
    assert abs(m.extra["scale_ratio_decades"]) <= mnv.LAT_SCALE_TOL_DECADES
    assert m.fw == view["fw"]                                  # lateral preserves the frame


def test_lateral_without_a_parent_atom_is_cleanly_unavailable():
    m = mnv.lateral_to_sibling(dict(node_id=1, cx=3.0, cy=3.0, fw=0.01, depth=2),
                               np.random.default_rng(0))
    assert m.available is False and m.reason.startswith("no_parent_atom:")


def test_lateral_is_deterministic_given_the_rng_seed():
    view = dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4)
    a = mnv.lateral_to_sibling(view, np.random.default_rng(11), degree=2)
    b = mnv.lateral_to_sibling(view, np.random.default_rng(11), degree=2)
    assert (a.available, a.atom_id, a.fw) == (b.available, b.atom_id, b.fw)


# =========================================================================== #
# dedup key — the SHARED read-time canonicalization
# =========================================================================== #
def test_atom_key_is_the_shared_read_time_snapped_key():
    import deep_center_finder as dcf
    import atom_lib as al
    view = dict(node_id=1, cx=-0.1226, cy=0.7449, fw=0.01, depth=5)
    m = mnv.snap_to_nucleus(view, None, degree=2)
    assert m.available
    # byte-identical to what collapse_population would form for the same stored coords
    assert m.atom_key == dcf.snapped_dedup_key(m.cx, m.cy, 2, al.DEDUP_DPS)


def test_two_views_snapping_to_one_nucleus_share_the_key():
    # Multiple frontier members snapping to the same nucleus is the NORMAL case — the key
    # is what collapses them, and it must not depend on which view got there.
    a = mnv.snap_to_nucleus(dict(node_id=1, cx=-0.1226, cy=0.7449, fw=0.02, depth=4), None)
    b = mnv.snap_to_nucleus(dict(node_id=2, cx=-0.1224, cy=0.7451, fw=0.02, depth=6), None)
    assert a.available and b.available
    assert a.atom_key == b.atom_key and a.atom_id == b.atom_id


# =========================================================================== #
# cost governor
# =========================================================================== #
def test_governor_p_zero_never_fires_and_p_one_always_does():
    g0 = mnv.ProbeGovernor(0.0, np.random.default_rng(0))
    assert g0.should_probe(2, 0.1, 0.1, 0.01) == (False, "cost_governor")
    g1 = mnv.ProbeGovernor(1.0, np.random.default_rng(0))
    assert g1.should_probe(2, 0.1, 0.1, 0.01)[0] is True


def test_governor_region_cache_beats_the_coin():
    g = mnv.ProbeGovernor(1.0, np.random.default_rng(0))
    assert g.should_probe(2, 0.1, 0.1, 0.01)[0] is True
    # same cell (a sub-cell nudge at the same fw decade) -> skipped whatever the coin says
    assert g.should_probe(2, 0.1001, 0.1001, 0.01) == (False, "region_cached")
    assert g.n_cache_skip == 1


def test_governor_state_round_trips_so_a_resume_does_not_re_probe():
    g = mnv.ProbeGovernor(1.0, np.random.default_rng(0))
    g.should_probe(2, 0.1, 0.1, 0.01)
    h = mnv.ProbeGovernor(1.0, np.random.default_rng(0))
    h.load_state(g.state_dict())
    assert h.should_probe(2, 0.1, 0.1, 0.01) == (False, "region_cached")


def test_parse_k_spec():
    assert mnv.parse_k_spec("none,4,16") == [None, 4.0, 16.0]
    assert mnv.parse_k_spec("") == [None]


def test_degree_of_is_c_plane_only():
    assert mnv.degree_of("mandelbrot") == 2 and mnv.degree_of("multibrot5") == 5
    # a julia viewport is a z-plane: the parameter-plane operators are undefined there
    assert mnv.degree_of("julia:mandelbrot") is None


# =========================================================================== #
# the reserved frontier floor
# =========================================================================== #
def _obj(B, quota, maneuvers=True):
    return types.SimpleNamespace(B=B, man_quota=quota, maneuvers=maneuvers,
                                 totals={"man_quota_bound": 0, "man_quota_unfilled": 0})


def _pool(spec):
    """spec: list of (node_id, priority, is_maneuver) already in priority order."""
    return [dict(node_id=i, priority=p, **({"man": {"op": "snap_to_nucleus"}} if m else {}))
            for i, p, m in spec]


def test_floor_promotes_a_low_priority_maneuver_node_over_the_plain_top_b():
    pool = _pool([(1, 9.0, False), (2, 8.0, False), (3, 7.0, False), (4, 0.1, True)])
    batch, rest = sf.SteeredFrontier._split_reserved(_obj(3, 1), pool)
    assert [n["node_id"] for n in batch] == [4, 1, 2]      # reserved first, then priority
    assert [n["node_id"] for n in rest] == [3]


def test_floor_is_a_quota_of_available_and_never_stalls_the_frontier():
    # No maneuver nodes at all -> the batch is the plain top-B, and the whole quota is
    # recorded as unfilled FOR LACK OF AVAILABILITY (not as a stall).
    pool = _pool([(1, 9.0, False), (2, 8.0, False), (3, 7.0, False)])
    o = _obj(2, 4)
    batch, rest = sf.SteeredFrontier._split_reserved(o, pool)
    assert [n["node_id"] for n in batch] == [1, 2] and len(batch) == 2
    assert o.totals["man_quota_unfilled"] == 4
    assert o.totals["man_quota_bound"] == 0

    # partial availability: take what is there, fill the rest normally.
    o2 = _obj(3, 4)
    batch2, _ = sf.SteeredFrontier._split_reserved(
        o2, _pool([(1, 9.0, False), (2, 8.0, False), (3, 0.5, True)]))
    assert set(n["node_id"] for n in batch2) == {1, 2, 3}
    assert o2.totals["man_quota_unfilled"] == 3


def test_floor_does_not_double_count_when_the_maneuver_node_would_have_won_anyway():
    # A maneuver node already inside the plain top-B is not the floor BINDING — that
    # distinction is what says whether the floor or the operator is the constraint.
    pool = _pool([(1, 9.0, True), (2, 8.0, False), (3, 7.0, False)])
    o = _obj(3, 1)
    batch, _ = sf.SteeredFrontier._split_reserved(o, pool)
    assert [n["node_id"] for n in batch] == [1, 2, 3]
    assert o.totals["man_quota_bound"] == 0


def test_floor_is_inert_when_maneuvers_are_off():
    pool = _pool([(1, 9.0, False), (2, 0.1, True)])
    o = _obj(1, 4, maneuvers=False)
    batch, rest = sf.SteeredFrontier._split_reserved(o, pool)
    assert [n["node_id"] for n in batch] == [1]           # plain top-B, floor ignored
    assert [n["node_id"] for n in rest] == [2]
    assert o.totals == {"man_quota_bound": 0, "man_quota_unfilled": 0}


def test_quota_never_exceeds_the_batch_size():
    pool = _pool([(i, 1.0 / (i + 1), True) for i in range(10)])
    o = _obj(3, 99)
    batch, _ = sf.SteeredFrontier._split_reserved(o, pool)
    assert len(batch) == 3


def test_k_is_stamped_on_the_unavailable_path_too():
    # Found by the shakedown: every refusal used to stamp k=None regardless of the k asked
    # for, so a log could not split availability by framing — and two refusal reasons
    # (f64_spacing_wall / fw_over_root_scale) are k-dependent by construction.
    bad = dict(node_id=1, cx=3.0, cy=3.0, fw=0.01, depth=2)
    for k in (None, 4.0, 32.0):
        m = mnv.snap_to_nucleus(bad, k, degree=2)
        assert m.available is False and m.k == k
    # and on the k-dependent refusal itself: a huge k puts the frame past a root view
    deep = dict(node_id=1, cx=-0.1226, cy=0.7449, fw=0.01, depth=5)
    m = mnv.snap_to_nucleus(deep, 1e9, degree=2)
    assert m.available is False and m.reason == "fw_over_root_scale" and m.k == 1e9
