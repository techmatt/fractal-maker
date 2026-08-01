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


def test_snap_multi_pays_one_solve_for_every_framing():
    # k only chooses fw AFTER the nucleus is known, so N framings must cost ONE Newton
    # pass. This is what makes adding k=16 to the default set free; the naive loop of
    # per-k snap_to_nucleus calls re-solved the same nucleus once per k.
    view = dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4)
    ms = mnv.snap_to_nucleus_multi(view, [None, 4.0, 16.0], degree=2)
    assert [m.k for m in ms] == [None, 4.0, 16.0]
    assert all(m.available for m in ms), [m.reason for m in ms]
    assert len({m.atom_id for m in ms}) == 1 and len({m.period for m in ms}) == 1
    # cost is charged to the first row only -> summing probe_s over rows is the true cost
    assert ms[0].newton_solves > 0
    assert [m.newton_solves for m in ms[1:]] == [0, 0]
    assert all(m.extra.get("reused_solve") for m in ms[1:])
    assert ms[0].fw == view["fw"]                       # k=None preserves the frame
    assert ms[1].fw == pytest.approx(4.0 * ms[1].window_scale, rel=1e-12)
    assert ms[2].fw == pytest.approx(16.0 * ms[2].window_scale, rel=1e-12)


def test_snap_multi_refuses_per_k_not_per_probe():
    # One solve, but the framing verdict is still PER k: the shallow period-3 rabbit atom
    # is big enough that 16x its size is wider than a base-scale root view, while k=None
    # and k=4 are fine off the same solve. A shared solve must not share a verdict.
    view = dict(node_id=7, cx=-0.1226, cy=0.7449, fw=0.01, depth=5)
    ms = mnv.snap_to_nucleus_multi(view, [None, 4.0, 16.0], degree=2)
    assert [m.available for m in ms] == [True, True, False]
    assert ms[2].reason == "fw_over_root_scale" and ms[2].k == 16.0


def test_snap_multi_agrees_with_single_k_calls_including_when_unavailable():
    for view in (dict(node_id=1, cx=-0.1226, cy=0.7449, fw=0.01, depth=5),
                 dict(node_id=1, cx=3.0, cy=3.0, fw=0.01, depth=2)):
        ms = mnv.snap_to_nucleus_multi(view, [None, 4.0], degree=2)
        for m, k in zip(ms, [None, 4.0]):
            one = mnv.snap_to_nucleus(view, k, degree=2)
            assert (m.available, m.reason, m.k, m.cx, m.fw) == \
                   (one.available, one.reason, one.k, one.cx, one.fw)


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


def test_lateral_seeded_periods_agree_with_the_sweep_and_cost_far_less():
    # DIFFERENTIAL, not a frozen literal: the period SWEEP is the reference implementation
    # and it still exists (`seed_periods=False`), so the seeded path is checked against the
    # thing it replaces. Where both find a nucleus they must find the SAME one — the
    # hybrid's exact low head is what buys that, since the atom-domain ranking alone can
    # miss a small period the "smallest period wins" rule would have taken.
    # Population-scale version: tools/atlas/bench_lateral_seeding.py.
    view = dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4)
    sweep = mnv.lateral_to_sibling(view, np.random.default_rng(5), degree=2,
                                   seed_periods=False)
    seeded = mnv.lateral_to_sibling(view, np.random.default_rng(5), degree=2)
    assert sweep.available and seeded.available
    assert seeded.atom_id == sweep.atom_id and seeded.period == sweep.period
    assert seeded.newton_solves < sweep.newton_solves


def test_lateral_low_sweep_head_is_swept_exactly_even_when_the_orbit_escapes():
    # A seed whose orbit escapes at once yields NO atom-domain candidates; with the hybrid
    # head the probe still tries the low periods, which is where three of the shakedown
    # replay's lost siblings came back.
    assert mnv.period_candidates(complex(3.0, 3.0), 2, max_period=64, n=4) == []
    assert mnv.LAT_LOW_SWEEP >= 2


def test_lateral_is_deterministic_given_the_rng_seed():
    view = dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4)
    a = mnv.lateral_to_sibling(view, np.random.default_rng(11), degree=2)
    b = mnv.lateral_to_sibling(view, np.random.default_rng(11), degree=2)
    assert (a.available, a.atom_id, a.fw) == (b.available, b.atom_id, b.fw)


# =========================================================================== #
# neighborhood_expand
# =========================================================================== #
_NBH_VIEW = dict(node_id=9, cx=-0.1226, cy=0.7449, fw=0.01, depth=5)


def test_neighborhood_returns_several_distinct_non_parent_nuclei():
    ms = mnv.neighborhood_expand(_NBH_VIEW, np.random.default_rng(1), [None], degree=2)
    ok = [m for m in ms if m.available]
    assert len(ok) >= 2, [m.reason for m in ms]
    assert len({m.atom_id for m in ok}) == len(ok)              # distinct, deduped
    parent = ok[0].extra["parent_atom_id"]
    assert all(m.atom_id != parent for m in ok)                 # never the parent itself
    assert [m.extra["found_rank"] for m in ok] == sorted(m.extra["found_rank"] for m in ok)


def test_neighborhood_frames_per_k_off_one_enumeration():
    """§7.1 for operator 3: the nuclei do not depend on the framing, so N framings cost ONE
    enumeration. The whole bill is charged to the FIRST emitted row, so summing `probe_s`
    over the rows is the true cost rather than N copies of one enumeration."""
    ms = mnv.neighborhood_expand(_NBH_VIEW, np.random.default_rng(1), [None, 4.0, 16.0],
                                 degree=2)
    assert ms[0].newton_solves > 0
    assert all(m.newton_solves == 0 for m in ms[1:])
    assert all(m.extra.get("reused_solve") for m in ms[1:] if m.available)
    # one row per (nucleus, k), in nucleus-major order
    n_atoms = len({m.atom_id for m in ms if m.available})
    assert n_atoms >= 1 and len(ms) == 3 * n_atoms
    for m in ms:
        if m.available and m.k is not None:
            assert m.fw == pytest.approx(m.k * m.window_scale, rel=1e-12)


def test_neighborhood_refuses_an_ancestor_but_not_a_child():
    """The one-sided scale window IS the operator. Unbounded below (sheet 3 probes "at
    comparable AND SMALLER scale"), bounded above — an unfiltered probe returns period-2
    giants, and framing one at k x size proposes a near-base-scale view the walk's own root
    draws already cover."""
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    loose = mnv.neighborhood_expand(dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4),
                                    rng_a, [None], degree=2, scale_up_tol=99.0)
    tight = mnv.neighborhood_expand(dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4),
                                    rng_b, [None], degree=2, scale_up_tol=1.0)
    up_loose = [m.extra["scale_ratio_decades"] for m in loose if m.available]
    up_tight = [m.extra["scale_ratio_decades"] for m in tight if m.available]
    assert max(up_loose) > 1.0, up_loose      # non-vacuous: the loose arm DOES find one
    assert up_tight and max(up_tight) <= 1.0
    assert len(up_tight) < len(up_loose)
    # and the refusal is counted under its OWN name rather than folded into "no neighbour",
    # so a run can tell "the disc had nothing" from "the disc had ancestors".
    seen = set()
    for m in mnv.neighborhood_expand(dict(node_id=3, cx=-1.7686, cy=0.0038, fw=0.01, depth=4),
                                     np.random.default_rng(0), [None], degree=2,
                                     scale_up_tol=1.0, max_found=99):
        seen |= set(m.extra.get("probe_reasons", {}))
    assert "scale_too_large" in seen, seen


def test_neighborhood_keeps_children_lateral_would_refuse_as_scale_mismatch():
    """The reason operator 3 is not operator 2 with a bigger m: lateral enforces a
    SYMMETRIC +/-1 decade window, so a genuine child several decades down is
    `scale_mismatch` there and a legitimate neighbour here. Differential, on ONE view where
    both are exercised — not two separate claims about two fixtures.

    Note the fixture had to be a DEEP parent. At the shallow rabbit every neighbour the
    disc returns is inside lateral's window anyway (measured: min ratio -0.60 decades over
    five seeds), so a shallow fixture would have made this test pass without the lower
    bound being unenforced at all — vacuous from the easy-fixture end
    (`verification_practice.md` §6)."""
    deep = dict(node_id=5, cx=-0.7463, cy=0.1102, fw=1e-3, depth=8)
    ms = mnv.neighborhood_expand(deep, np.random.default_rng(0), [None], degree=2,
                                 max_probes=24, max_found=99)
    dec = [m.extra["scale_ratio_decades"] for m in ms if m.available]
    assert dec, "fixture found nothing — every assertion below would be vacuous"
    assert min(dec) < -mnv.LAT_SCALE_TOL_DECADES, dec     # OUTSIDE lateral's window, kept
    assert all(d <= mnv.NBH_SCALE_UP_DECADES for d in dec)
    # and lateral really does refuse on this view for exactly that reason
    reasons = {mnv.lateral_to_sibling(deep, np.random.default_rng(s), degree=2).reason
               for s in range(8)}
    assert "scale_mismatch" in reasons, reasons


def test_neighborhood_unavailable_is_one_clean_named_row_never_an_empty_list():
    ms = mnv.neighborhood_expand(dict(node_id=1, cx=3.0, cy=3.0, fw=0.01, depth=2),
                                 np.random.default_rng(0), [None, 4.0], degree=2)
    assert len(ms) == 1 and ms[0].available is False
    assert ms[0].reason.startswith("no_parent_atom:")
    assert ms[0].op == "neighborhood_expand"


def test_neighborhood_probe_budget_is_the_bound_that_binds_not_the_find_ceiling():
    """88% of sheet-3's probes returned the parent, so a budget expressed as "find m" is an
    unbounded budget. `max_probes` is the bill; `max_found` only truncates the answer."""
    ms = mnv.neighborhood_expand(_NBH_VIEW, np.random.default_rng(2), [None], degree=2,
                                 max_found=99, max_probes=3)
    assert ms[0].extra.get("probes_tried", 0) <= 3
    lots = mnv.neighborhood_expand(_NBH_VIEW, np.random.default_rng(2), [None], degree=2,
                                   max_found=1, max_probes=12)
    assert len([m for m in lots if m.available]) == 1        # early-exit at max_found


def test_neighborhood_is_deterministic_given_the_rng_seed():
    a = mnv.neighborhood_expand(_NBH_VIEW, np.random.default_rng(3), [None], degree=2)
    b = mnv.neighborhood_expand(_NBH_VIEW, np.random.default_rng(3), [None], degree=2)
    assert [(m.available, m.atom_id, m.fw) for m in a] == \
           [(m.available, m.atom_id, m.fw) for m in b]


def test_the_two_neighbourhood_operators_draw_the_same_seed_stream():
    """Operators 2 and 3 share `_draw_probe_seed`, so an identically-seeded RNG gives them
    byte-identical probe seeds. That is what makes the subsumption replay a comparison of
    PICKS rather than a comparison of two different random walks."""
    w, pcx, pcy = 1e-3, mnv.mp.mpf("-0.5"), mnv.mp.mpf("0.1")
    a = [mnv._draw_probe_seed(np.random.default_rng(4), mnv.LAT_RADII, w, pcx, pcy)[1]
         for _ in range(3)]
    r = np.random.default_rng(4)
    b = [mnv._draw_probe_seed(r, mnv.LAT_RADII, w, pcx, pcy)[1] for _ in range(3)]
    assert a[0] == b[0]
    assert len(set(b)) >= 1 and all(x > 0 for x in b)


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


def test_default_k_set_carries_the_16x_wallpaper_frame_and_no_small_k():
    # k=16 is often a usable wallpaper frame by itself, which is the material worth
    # labeling. A k < 1 frames INTO the atom — interior black.
    #
    # THE PUSH SET AND THE MEASURING FRAME ARE DIFFERENT THINGS, and this test is the one
    # place that could have hidden them being conflated. 4x is the frame every orbital
    # score is MEASURED on and `minibrot_maneuvers.DEFAULT_K` still carries it; the walk's
    # PUSH set dropped it for k=8 on 2026-08-01, because a 4x frame is what the view
    # screen's size band exists to demote (`steered_frontier.MAN_K_DEFAULT`). So the two
    # constants are pinned SEPARATELY and are deliberately no longer equal.
    assert mnv.parse_k_spec(sf.MAN_K_DEFAULT) == [None, 8.0, 16.0]
    assert list(mnv.DEFAULT_K) == [None, 4.0, 16.0]
    assert 4.0 not in mnv.parse_k_spec(sf.MAN_K_DEFAULT)
    assert all(k is None or k >= 1.0 for k in mnv.DEFAULT_K)
    assert all(k is None or k >= 1.0 for k in mnv.parse_k_spec(sf.MAN_K_DEFAULT))


def test_degree_of_is_c_plane_only():
    assert mnv.degree_of("mandelbrot") == 2 and mnv.degree_of("multibrot5") == 5
    # a julia viewport is a z-plane: the parameter-plane operators are undefined there
    assert mnv.degree_of("julia:mandelbrot") is None


# =========================================================================== #
# the reserved frontier floor
# =========================================================================== #
_FLOOR_TOTALS = ("man_quota_bound", "man_quota_unfilled", "man_quota_passed_over")


def _obj(B, quota, maneuvers=True, range_prior=False, logged=None):
    return types.SimpleNamespace(B=B, man_quota=quota, maneuvers=maneuvers,
                                 man_range_prior=range_prior, batch_i=1,
                                 man_passed_logged=set(),
                                 _log_maneuver=(logged.append if logged is not None
                                                else (lambda row: None)),
                                 totals={k: 0 for k in _FLOOR_TOTALS})


def _pool(spec):
    """spec: list of (node_id, priority, is_maneuver) already in priority order."""
    return [dict(node_id=i, priority=p, **({"man": {"op": "snap_to_nucleus"}} if m else {}))
            for i, p, m in spec]


def _man_pool(spec):
    """spec: list of (node_id, priority, radial_range|None). `None` == unscreened."""
    return [dict(node_id=i, priority=p, partition="mandelbrot",
                 man={"op": "snap_to_nucleus", "k": 4.0, "atom_key": f"a{i}",
                      "screened": rr is not None, "radial_range": rr,
                      "radial_rings": (None if rr is None else 2 * rr)})
            for i, p, rr in spec]


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
    assert o.totals == {k: 0 for k in _FLOOR_TOTALS}


def test_quota_never_exceeds_the_batch_size():
    pool = _pool([(i, 1.0 / (i + 1), True) for i in range(10)])
    o = _obj(3, 99)
    batch, _ = sf.SteeredFrontier._split_reserved(o, pool)
    assert len(batch) == 3


# =========================================================================== #
# v1.4 — the richness screen selects (only behind --maneuver-range-prior)
# =========================================================================== #
def test_the_quota_fills_by_incoming_priority_when_the_range_prior_is_off():
    """Flag OFF is the v1.3 order, verbatim: the reserved slots go to the maneuver nodes
    highest in the PRIORITY-sorted pool, whatever their richness says."""
    pool = _man_pool([(1, 9.0, 0.1), (2, 8.0, 99.0), (3, 7.0, 50.0)])
    o = _obj(2, 1, range_prior=False)
    batch, _ = sf.SteeredFrontier._split_reserved(o, pool)
    assert batch[0]["node_id"] == 1


def test_the_quota_fills_by_descending_range_when_the_prior_is_on():
    logged = []
    pool = _man_pool([(1, 9.0, 0.1), (2, 8.0, 99.0), (3, 7.0, 50.0)])
    o = _obj(2, 1, range_prior=True, logged=logged)
    batch, _ = sf.SteeredFrontier._split_reserved(o, pool)
    assert batch[0]["node_id"] == 2                       # richest, not highest priority
    # the two it passed over are RECORDED with their scores and a passed_over marker —
    # "every candidate keeps its scores, including candidates not selected"
    assert o.totals["man_quota_passed_over"] == 2
    assert {r["node_id"] for r in logged} == {1, 3}
    assert all(r["passed_over"] is True and r["unused_reason"] == "quota_passed_over"
               for r in logged)
    assert {r["radial_range"] for r in logged} == {0.1, 50.0}


def test_a_passed_over_node_is_recorded_ONCE_not_once_per_batch():
    """Found by the shakedown, which is what a shakedown is for. A maneuver node that loses
    a quota slot stays on the frontier and loses again next batch, so re-logging it per
    batch writes O(nodes x batches) rows and turns the counter into a backlog reading
    wearing a count's name: 10,176 rows over 24 batches, 7.6 MB, with the maneuver frontier
    still climbing. A 7-hour unattended run would have written hundreds of MB of the same
    nodes restated."""
    logged = []
    o = _obj(2, 1, range_prior=True, logged=logged)
    pool = _man_pool([(1, 9.0, 0.1), (2, 8.0, 99.0), (3, 7.0, 50.0)])
    for batch in range(5):                       # the same pool loses the same slots
        o.batch_i = batch
        sf.SteeredFrontier._split_reserved(o, _man_pool([(1, 9.0, 0.1), (2, 8.0, 99.0),
                                                         (3, 7.0, 50.0)]))
    assert len(logged) == 2, [r["node_id"] for r in logged]
    assert {r["node_id"] for r in logged} == {1, 3}
    assert o.totals["man_quota_passed_over"] == 2, "a COUNT of distinct candidates"
    # a genuinely new candidate still gets recorded
    o.batch_i = 99
    sf.SteeredFrontier._split_reserved(o, _man_pool([(4, 6.0, 5.0), (2, 8.0, 99.0)]))
    assert {r["node_id"] for r in logged} == {1, 3, 4}
    assert o.totals["man_quota_passed_over"] == 3
    assert pool  # (fixture kept explicit; the loop rebuilds it so nodes are not mutated)


def test_the_passed_over_count_is_the_same_with_the_prior_off_only_the_LOG_is_gated():
    """The counter is a diagnostic and costs nothing, so it runs either way; the per-row
    log is what the flag gates. If the two diverged, a flag-off run's backlog would read as
    zero rather than as unrecorded."""
    on, off = _obj(2, 1, range_prior=True), _obj(2, 1, range_prior=False)
    for o in (on, off):
        sf.SteeredFrontier._split_reserved(o, _man_pool([(1, 9.0, 0.1), (2, 8.0, 99.0),
                                                         (3, 7.0, 50.0)]))
    assert on.totals["man_quota_passed_over"] == off.totals["man_quota_passed_over"] == 2


def test_an_unscreened_candidate_sorts_last_but_is_never_excluded():
    """The screen RANKS the quota, it does not gate it — the deep tail is unreachable at
    64 px and must still be able to take a slot nobody else wants."""
    pool = _man_pool([(1, 9.0, None), (2, 8.0, 3.0)])
    o = _obj(2, 1, range_prior=True)
    batch, _ = sf.SteeredFrontier._split_reserved(o, pool)
    assert batch[0]["node_id"] == 2                       # screened beats unscreened
    # with room for both, the unscreened one is still taken
    o2 = _obj(2, 2, range_prior=True)
    batch2, _ = sf.SteeredFrontier._split_reserved(o2, _man_pool([(1, 9.0, None),
                                                                 (2, 8.0, 3.0)]))
    assert {n["node_id"] for n in batch2} == {1, 2}


def test_range_ordering_never_changes_how_MANY_slots_the_floor_takes():
    """The v1.4 change is WHICH maneuver fills a slot, never how many. If reordering could
    move the slot COUNT, the flag would be a quota change wearing a ranker's clothes.

    `quota_bound` is deliberately NOT asserted equal: it counts slots that promoted a node
    the plain top-B would not have taken, so a reordering that reaches deeper into the pool
    legitimately raises it. That is the counter doing its job, not the invariant breaking."""
    spec = [(1, 9.0, 0.1), (2, 8.0, 99.0), (3, 7.0, None), (4, 6.0, 50.0)]
    off = _obj(3, 2, range_prior=False)
    on = _obj(3, 2, range_prior=True)
    b_off, r_off = sf.SteeredFrontier._split_reserved(off, _man_pool(spec))
    b_on, r_on = sf.SteeredFrontier._split_reserved(on, _man_pool(spec))
    assert len(b_off) == len(b_on) == 3 and len(r_off) == len(r_on) == 1
    assert off.totals["man_quota_unfilled"] == on.totals["man_quota_unfilled"] == 0
    assert {n["node_id"] for n in b_on} != {n["node_id"] for n in b_off}   # not vacuous
    assert b_on[0]["node_id"] == 2 and b_on[1]["node_id"] == 4             # the two richest


def test_the_visited_key_is_the_atom_and_its_framing_not_the_operator():
    """With three operators live, lateral and neighborhood routinely reach the SAME
    sibling. Keying the visited set on the operator as well would push that one view twice
    under two provenance labels — identity is (atom, k), and the op is provenance."""
    import types as _t
    pushed = []
    o = _t.SimpleNamespace(
        maneuvers=True, man_visited=set(), man_range_prior=False, man_range_gain=0.5,
        man_range_dist=None, batch_i=1, beta=0.0, rng=np.random.default_rng(0),
        frontier=pushed, totals={k: 0 for k in sf.MAN_TOTALS},
        new_node_id=lambda: len(pushed) + 1, _log_maneuver=lambda row: None)
    parent = dict(root_id=1, partition="mandelbrot", c=None)
    made = [mnv.Maneuver(op=op, available=True, k=4.0, cx="0.1", cy="0.2", fw=1e-3,
                         depth=3, atom_id="x", atom_key="KEY", period=5,
                         log10_abs_A=2.0, window_scale=1e-4, extra={"degree": 2})
            for op in ("lateral_to_sibling", "neighborhood_expand")]
    assert sf.SteeredFrontier._consume_maneuver(o, made[0], parent) == 1
    assert sf.SteeredFrontier._consume_maneuver(o, made[1], parent) == 0
    assert len(pushed) == 1 and o.totals["man_avail_unused"] == 1
    # ... but the SAME atom at a different framing is a different view and still pushes
    made[1].k = 16.0
    assert sf.SteeredFrontier._consume_maneuver(o, made[1], parent) == 1
    assert len(pushed) == 2


def _prune(frontier, maneuvers=True):
    """Drive THE prune, not a copy of it. `prune_frontier` was split out of `push_children`
    for exactly this: a fixture that reimplements its subject stays green while the subject
    rots (measured — the first version of these three tests did)."""
    o = types.SimpleNamespace(frontier=list(frontier), maneuvers=maneuvers,
                              node_embs={}, totals={"man_frontier_pruned": 0})
    sf.SteeredFrontier.prune_frontier(o)
    return o.frontier


def test_a_flood_of_maneuver_nodes_cannot_evict_every_ordinary_node():
    """The v1.4 fix, and the failure it stops. The exemption used to be TOTAL, so once the
    maneuver population passed FRONTIER_CAP the ordinary nodes' room went to zero and the
    frontier became 100% maneuver dead weight — `pop_batch`'s capped-root starvation via a
    second route. Operator 3 is what makes it reachable (~2x the pushes per fired probe)."""
    cap = sf.FRONTIER_CAP
    flood = [dict(node_id=i, priority=1.0, man={"op": "neighborhood_expand"})
             for i in range(cap * 2)]
    ordinary = [dict(node_id=cap * 2 + i, priority=2.5) for i in range(500)]
    kept = _prune(flood + ordinary)
    assert len(kept) == cap
    n_ord = sum(1 for n in kept if not n.get("man"))
    assert n_ord == 500, "every ordinary node survives — they fit inside the non-maneuver half"
    assert sum(1 for n in kept if n.get("man")) == cap - 500


def test_the_maneuver_share_is_a_floor_not_a_ceiling_when_the_walk_is_quiet():
    """Unused ordinary room falls BACK to the maneuvers, so a run with few ordinary nodes
    keeps every maneuver it can hold. The share bounds starvation, it does not ration."""
    cap = sf.FRONTIER_CAP
    kept = _prune([dict(node_id=i, priority=1.0, man={"op": "snap_to_nucleus"})
                   for i in range(cap + 1000)] +
                  [dict(node_id=99000 + i, priority=2.5) for i in range(10)])
    assert sum(1 for n in kept if n.get("man")) == cap - 10
    assert len(kept) == cap


def test_maneuvers_are_still_protected_from_the_pooled_priority_prune():
    """The original property must survive the fix: a low-priority maneuver node inside the
    share is NOT deleted ahead of a high-priority ordinary node. Pruning the pooled frontier
    by priority would delete maneuvers first, which is what silently undoes the floor."""
    cap = sf.FRONTIER_CAP
    kept = _prune([dict(node_id=i, priority=0.01, man={"op": "snap_to_nucleus"})
                   for i in range(100)] +
                  [dict(node_id=1000 + i, priority=9.0) for i in range(cap)])
    ids = {n["node_id"] for n in kept}
    assert all(i in ids for i in range(100)), "the 100 worst-priority maneuvers all survive"
    assert len(kept) == cap


def test_push_children_still_routes_through_the_prune_the_tests_drive():
    """The three tests above call `prune_frontier` directly. That is only worth anything
    while the production path still calls it — an inlined copy in `push_children` would
    leave them green and the run unprotected."""
    import inspect
    assert "self.prune_frontier()" in inspect.getsource(sf.SteeredFrontier.push_children)
    assert 0.0 < sf.MAN_FRONTIER_SHARE < 1.0 and sf.FRONTIER_CAP > 0
    assert "man_frontier_pruned" in sf.MAN_TOTALS


# =========================================================================== #
# v1.4 — the WALL-CLOCK cap (the unattended-overnight bound)
# =========================================================================== #
def _clock(base_s, session_ago_s, est_batch_s, wall_budget_min):
    """A minimal object both wall-clock methods really run against — `wall_elapsed_s` is the
    LIVE implementation bound to it, not a stub, so neither test can pass off a fake clock."""
    import time as _t
    o = types.SimpleNamespace(
        wall_s_base=base_s, _session_t0=_t.time() - session_ago_s,
        est_batch_s=est_batch_s, wall_budget_s=wall_budget_min * 60.0)
    o.wall_elapsed_s = lambda: sf.SteeredFrontier.wall_elapsed_s(o)
    return o


def test_the_wall_cap_is_off_by_default_and_never_fires_when_disabled():
    """Zero is the historical behaviour — every run before v1.4 — so a run that does not ask
    for the cap must not be able to be stopped by it however long it runs."""
    o = _clock(base_s=10 ** 7, session_ago_s=10 ** 6, est_batch_s=999, wall_budget_min=0)
    assert sf.SteeredFrontier.wall_exhausted(o) is False


def test_the_wall_cap_refuses_to_START_a_batch_it_cannot_finish():
    """`+ est_batch_s`, not "already over": the rule is never start a unit that cannot finish
    inside the remaining budget. Bracketed on both sides of the boundary so it is not merely
    "stops eventually"."""
    fits = _clock(base_s=0, session_ago_s=3540, est_batch_s=30, wall_budget_min=60)
    assert sf.SteeredFrontier.wall_exhausted(fits) is False        # 3540 + 30 < 3600
    over = _clock(base_s=0, session_ago_s=3540, est_batch_s=120, wall_budget_min=60)
    assert sf.SteeredFrontier.wall_exhausted(over) is True         # 3540 + 120 > 3600


def test_wall_time_accumulates_across_resumes():
    """A kill/resume loop must not reset the night's bound — that is the whole failure mode
    a wall cap exists to prevent on an unattended run. The base comes off the checkpoint and
    only the CURRENT session is measured live."""
    o = _clock(base_s=3000, session_ago_s=500, est_batch_s=10, wall_budget_min=60)
    assert 3495 <= sf.SteeredFrontier.wall_elapsed_s(o) <= 3505
    assert sf.SteeredFrontier.wall_exhausted(o) is False
    o2 = _clock(base_s=3500, session_ago_s=500, est_batch_s=10, wall_budget_min=60)
    assert sf.SteeredFrontier.wall_exhausted(o2) is True   # the resumed 3500s still counts


def test_wall_elapsed_before_the_loop_starts_is_the_checkpointed_base_alone():
    """`_session_t0` is None until the loop is entered; reading the clock then must not
    charge the run for a session that has not begun."""
    o = types.SimpleNamespace(wall_s_base=1234.0, _session_t0=None)
    assert sf.SteeredFrontier.wall_elapsed_s(o) == 1234.0


def test_the_wall_cap_is_a_SECOND_cap_not_a_restatement_of_the_active_one():
    """It exists because `draw_roots` sits outside the timed block, so active time cannot
    see a root replenishment. If the run loop ever stopped consulting it, an overnight run
    would be bounded only by a budget that does not count the minutes it actually spends."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.run)
    assert "self.wall_exhausted()" in src
    # ... and the active-time check is still there too: this is an addition, not a swap.
    assert "self.active_s + self.est_batch_s > self.budget_s" in src
    # ... and draw_roots really is outside the timed block, which is the premise
    i_draw, i_tb = src.index("self.draw_roots()"), src.index("tb = time.time()")
    assert i_draw < i_tb, "draw_roots moved inside the timed block — the premise changed"
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
