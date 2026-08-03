"""Tests for the label-seeded harvest: the seed pool, the discard rule, the draw, the fit.

WHAT IS AND IS NOT COVERED HERE, stated up front because the gaps matter more than the
coverage. Nothing here drives the render engine: `screen_view` spawns a process per field
and every test below injects a fake for it, so the engine-touching path is exercised by the
shakedown and by the run, not by the suite. Neither the batch writer nor `verify` is covered
end to end — `cc.batch_dir` writes to a fixed repo path, so a unit test would need the real
corpus tree; `verify` is the standing check and it is a driver, not a test.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import label_seeded_harvest as lsh                     # noqa: E402
import minibrot_maneuvers as mnv                       # noqa: E402
import view_fit as vf                                  # noqa: E402
from tools.v7 import build_manifest as bm              # noqa: E402


# =========================================================================== #
# registration — before any batch is built
# =========================================================================== #
def test_both_chunks_are_registered_explicitly_and_biased():
    """Fail-closed would land them train-side anyway. "Nobody registered this" and "this is
    a biased train draw" are different facts, and only the second one is true here."""
    import build_label_seeded_batches as b
    for bid in b.BATCHES:
        split, biased, source = bm.assign_split({"batch": bid, "ft": "mandelbrot"})
        assert (split, biased, source) == ("train", True, "label_seeded_v2"), bid
    assert bm.assign_split({"batch": "never_registered", "ft": "mandelbrot"}) == \
        ("train", True, "unregistered")


def test_the_harvest_is_never_registered_unbiased():
    """The seeds ARE Matt's past verdicts and the queue order IS a model of the label. A
    rate measured on this population is not a base rate, and an unbiased registration would
    make it eval-eligible — which is how an instrument gets moved by the thing it measures."""
    import build_label_seeded_batches as b
    for bid in b.BATCHES:
        assert bm.assign_split({"batch": bid, "ft": "mandelbrot"})[1] is True


# =========================================================================== #
# the seed pool
# =========================================================================== #
@pytest.mark.parametrize("render,expected", [
    ({"cx": "0", "cy": "0", "fw": "1"}, "mandelbrot"),                  # pre-family batch
    ({"cx": "0", "cy": "0", "fw": "1", "c_re": "0.1", "c_im": "0"}, None),  # julia hiding
    ({"fractal_type": "multibrot4"}, "multibrot4"),
    ({"fractal_type": "julia_multibrot3"}, None),
    ({"fractal_type": "phoenix"}, None),
])
def test_only_c_plane_rows_become_seeds(render, expected):
    """A pre-family row is mandelbrot UNLESS it carries a `c_re`, which is how a Julia row
    hides in one. The operators probe the atom domain of the PARAMETER plane, so a z-plane
    seed is a category error rather than a weaker seed."""
    assert lsh._family_of(render) == expected


def test_every_c_plane_family_has_a_degree():
    """`_family_of` returning a family and `PARTITION_DEGREE` knowing its degree must be the
    same predicate, or a seed reaches the operators with no degree to probe at."""
    for fam in lsh.C_PLANE_FAMILIES:
        assert lsh._family_of({"fractal_type": fam}) == fam
        assert mnv.degree_of(fam) == lsh.C_PLANE_FAMILIES[fam]


def test_the_seed_id_is_a_function_of_the_location_only():
    """The id keys the per-seed RNG, so it must not move when the batch a seed was found in
    moves — otherwise re-deriving the pool after a merge re-probes different discs."""
    a = dict(family="mandelbrot", cx="-0.5", cy="0.1", fw="1e-3")
    assert lsh._seed_id(dict(a, batch="X", image_id="1", score=3)) == \
        lsh._seed_id(dict(a, batch="Y", image_id="2", score=4))
    assert lsh._seed_id(dict(a, cx="-0.6")) != lsh._seed_id(a)


# =========================================================================== #
# the interior pre-filter
# =========================================================================== #
def test_the_interior_discard_is_strict_and_at_the_stated_threshold():
    """Matt's rule, and the boundary side is invisible in a count: a frame at exactly 0.30
    is KEPT. Pinned by value so moving the threshold has to move this line too."""
    assert lsh.INTERIOR_DISCARD == 0.30
    assert lsh._verdict_for(dict(screened=True, interior_fraction=0.2999))[0] is True
    assert lsh._verdict_for(dict(screened=True, interior_fraction=0.30))[0] is True
    kept, why = lsh._verdict_for(dict(screened=True, interior_fraction=0.3001))
    assert kept is False and why == "interior_gt_30"


def test_an_unscreenable_candidate_is_dropped_with_its_reason_named():
    """A view the 64x36 screen could not reach has no renderable 1280 px crop — the crop's
    pixel spacing is 20x finer than the screen that already failed the f64 guard. It is not
    a candidate, and "could not be measured" is recorded as itself rather than folded into
    the interior rule."""
    kept, why = lsh._verdict_for(dict(screened=False,
                                      screen_reason="f64_spacing_wall_at_screen_geometry"))
    assert kept is False and why.startswith("unscreenable:")
    assert "f64_spacing_wall" in why
    # A screened row that somehow carries no interior fraction is unscreenable, NOT kept:
    # the filter cannot be applied, and defaulting it to "keep" would let exactly the frames
    # the rule exists to remove through the one door the rule cannot see.
    assert lsh._verdict_for(dict(screened=True, interior_fraction=None))[0] is False


# =========================================================================== #
# the push ladder
# =========================================================================== #
def test_the_push_ladder_is_the_one_the_walk_pushes():
    """A run whose ladder differs from what `steered_frontier` pushes is measuring a
    different population from the one the view screen was tuned on."""
    import steered_frontier as sf
    assert lsh.K_LADDER_SPEC == sf.MAN_K_DEFAULT
    assert mnv.parse_k_spec(lsh.K_LADDER_SPEC) == [None, 8.0, 16.0]


def test_the_seed_snap_keeps_sheet_2s_nearness_rule():
    """`near = 0.75 * fw` is what makes this label-SEEDED: a nucleus counts only if it lies
    inside the judged view. Widening it turns the source into a global scan wearing a
    label, which is the distinction the source race was run to make."""
    assert lsh.SEED_SNAP_FW_MULT == 0.75


# =========================================================================== #
# the enumeration
# =========================================================================== #
def _fake_atom(cx="0.25", cy="0.0", period=3, ws=1e-3):
    return dict(id="a1", cx=cx, cy=cy, period=period, window_scale=ws,
                log10_abs_A=3.0, size=ws, degree=2,
                f64_margin_deploy_decades=7.0,
                provenance={"seed_distance": 1e-6})


def test_a_seed_that_solves_no_nucleus_emits_only_refusals(monkeypatch):
    """A miss is a named refusal per requested framing, not an exception and not silence —
    the availability split by framing is what the cost read needs."""
    monkeypatch.setattr(lsh, "snap_at_seed",
                        lambda seed, ks, **kw: ([mnv._unavailable(
                            "snap_at_seed", "no_nucleus_near_seed",
                            dict(cx=0.0, cy=0.0, fw=1.0), 0.0, 64, k=k) for k in ks], 64))
    seed = dict(seed_id="s0", family="mandelbrot", degree=2, cx="0", cy="0", fw="1.0",
                batch="b", image_id="i", score=3)
    rows, st = lsh.enumerate_seed(seed, [None, 8.0, 16.0],
                                  rng=np.random.default_rng(0))
    assert rows == []
    assert st["seed_no_nucleus"] == 1
    assert st["snap_unavail:no_nucleus_near_seed"] == 3


def test_the_neighbourhood_only_runs_when_the_seed_solved(monkeypatch):
    """Sheet 3 probes discs around the SHEET-2 NUCLEI. A seed with no parent has nothing to
    expand around, and running the operator anyway would pay a second full snap probe for
    the nucleus that just failed to solve."""
    called = []
    monkeypatch.setattr(lsh.mnv, "neighborhood_expand",
                        lambda *a, **k: called.append(1) or [])
    monkeypatch.setattr(lsh, "snap_at_seed",
                        lambda seed, ks, **kw: ([mnv._unavailable(
                            "snap_at_seed", "no_converge",
                            dict(cx=0.0, cy=0.0, fw=1.0), 0.0, 1, k=k) for k in ks], 1))
    seed = dict(seed_id="s0", family="mandelbrot", degree=2, cx="0", cy="0", fw="1.0",
                batch="b", image_id="i", score=3)
    lsh.enumerate_seed(seed, [None], rng=np.random.default_rng(0))
    assert called == []


def test_the_candidate_key_matches_the_screen_and_the_field_cache():
    """The screen, the field cache and the write-path dedup must agree on what ONE candidate
    is, or a row is screened twice and appears twice in the queue."""
    import view_field_cache as vfc
    import maneuver_view_screen as mvs
    assert mvs.view_key("ak", 16.0) == vfc.row_key({"atom_key": "ak", "k": 16.0})


# =========================================================================== #
# the probe deadline (the clamped backstop)
# =========================================================================== #
def test_the_neighbourhood_probe_deadline_stops_between_probes(monkeypatch):
    """`max_probes` prices the call in probes and a probe's cost is not fixed — `pmax`
    scales with the parent period. A caller inside a wall budget needs a bound in the units
    its budget is denominated in."""
    calls = []

    def slow_identify(*a, **k):
        calls.append(1)
        return None, "no_nucleus_near_seed"
    monkeypatch.setattr(mnv.al, "identify_nucleus", slow_identify)
    view = dict(cx=0.0, cy=0.0, fw=1.0, depth=0, node_id=None)
    parent = _fake_atom()
    out = mnv.neighborhood_expand(view, np.random.default_rng(0), [None],
                                  parent_rec=parent, max_probes=40,
                                  deadline=0.0)     # already expired
    assert calls == []                              # not one probe started
    assert len(out) == 1 and not out[0].available
    assert out[0].reason == "probe_deadline"


def test_without_a_deadline_the_probe_loop_is_unchanged(monkeypatch):
    """The default must be byte-identical to the pre-2026-08-02 behaviour: this operator is
    on the live walk's path and a new bound firing there by default would silently change
    what every recorded run means."""
    calls = []
    monkeypatch.setattr(mnv.al, "identify_nucleus",
                        lambda *a, **k: calls.append(1) or (None, "no_nucleus_near_seed"))
    mnv.neighborhood_expand(dict(cx=0.0, cy=0.0, fw=1.0, depth=0, node_id=None),
                            np.random.default_rng(0), [None],
                            parent_rec=_fake_atom(), max_probes=7)
    assert len(calls) == 7


# =========================================================================== #
# the draw
# =========================================================================== #
def _queue(n_per_cell: dict) -> list[dict]:
    rows, rank = [], 0
    for (method, degree), n in n_per_cell.items():
        for i in range(n):
            rank += 1
            rows.append(dict(method=method, degree=degree, queue_rank=rank,
                             candidate_key=f"{method}|{degree}|{i}"))
    rows.sort(key=lambda r: r["queue_rank"])
    return rows


def test_the_draw_is_balanced_to_plus_or_minus_one_over_method_x_degree():
    import build_label_seeded_batches as b
    q = _queue({("snap_at_seed", 2): 200, ("snap_at_seed", 3): 200,
                ("neighborhood_expand", 2): 200, ("neighborhood_expand", 3): 200})
    picked, per_cell = b.draw_ranked_stratified(q, 100)
    from collections import Counter
    c = Counter((r["method"], r["degree"]) for r in picked)
    assert len(picked) == 100
    assert len(c) == 4, "a cell with no rows is invisible to a count over the rows present"
    assert max(c.values()) - min(c.values()) <= 1, dict(c)


def test_within_a_cell_the_draw_takes_the_BEST_ranked_rows():
    """This is the whole difference from the supply crawl's stratified draw, which shuffles
    within a cell. That chunk existed to give the negative class footing across composite
    bins; this one is taking the top of a queue."""
    import build_label_seeded_batches as b
    q = _queue({("snap_at_seed", 2): 50, ("neighborhood_expand", 2): 50})
    picked, _ = b.draw_ranked_stratified(q, 10)
    for method in ("snap_at_seed", "neighborhood_expand"):
        got = [r["queue_rank"] for r in picked if r["method"] == method]
        best = sorted(r["queue_rank"] for r in q if r["method"] == method)[:len(got)]
        assert sorted(got) == best


def test_a_thin_cell_does_not_stall_the_draw_and_the_remainder_goes_where_supply_is():
    """A cell with two rows in it must not cap the chunk at 2 x n_cells."""
    import build_label_seeded_batches as b
    q = _queue({("snap_at_seed", 2): 100, ("neighborhood_expand", 5): 2})
    picked, _ = b.draw_ranked_stratified(q, 50)
    assert len(picked) == 50
    assert sum(1 for r in picked if r["degree"] == 5) == 2


def test_the_two_chunks_are_disjoint_and_both_balanced():
    """A location may appear in only ONE batch (`build_manifest.load_post_freeze` asserts
    it), and the stride split must not hand one chunk all the good rows."""
    import build_label_seeded_batches as b
    from collections import Counter
    q = _queue({("snap_at_seed", 2): 400, ("snap_at_seed", 3): 400,
                ("neighborhood_expand", 2): 400, ("neighborhood_expand", 3): 400})
    chunks, rep = b.draw_all(q, n_chunk=100)
    assert rep["overlap"] == 0
    a = {r["candidate_key"] for r in chunks[b.CHUNK_A]}
    bb = {r["candidate_key"] for r in chunks[b.CHUNK_B]}
    assert not (a & bb)
    assert len(a) == len(bb) == 100
    for rows in chunks.values():
        c = Counter((r["method"], r["degree"]) for r in rows)
        # EVERY cell must be PRESENT, not merely balanced among those that are. A count over
        # the rows a chunk has cannot see a cell it has none of, and that blind spot is what
        # let the stride split ship two chunks each holding half the cells at a perfect
        # 145/145 (`draw_all`).
        assert len(c) == 4, dict(c)
        assert max(c.values()) - min(c.values()) <= 1, dict(c)
    # ... and neither chunk is systematically the better half.
    ma = np.median([r["queue_rank"] for r in chunks[b.CHUNK_A]])
    mb = np.median([r["queue_rank"] for r in chunks[b.CHUNK_B]])
    assert abs(ma - mb) <= 0.05 * max(ma, mb)


def test_odd_cells_do_not_hand_one_chunk_the_extra_row_every_time():
    """A cell with an odd number of picks gives one chunk the extra row and the better-ranked
    member of every pair. With a fixed parity that lands on the SAME chunk in every cell and
    compounds: eight cells of fifteen shipped 64/56."""
    import build_label_seeded_batches as b
    q = _queue({("snap_at_seed", d): 500 for d in (2, 3, 4, 5)}
               | {("neighborhood_expand", d): 500 for d in (2, 3, 4, 5)})
    chunks, rep = b.draw_all(q, n_chunk=60)          # 120 over 8 cells = 15 per cell, odd
    na, nb = len(chunks[b.CHUNK_A]), len(chunks[b.CHUNK_B])
    assert abs(na - nb) <= 1, (na, nb)
    assert na + nb == 120


def test_a_short_queue_is_reported_short_rather_than_padded():
    import build_label_seeded_batches as b
    q = _queue({("snap_at_seed", 2): 30})
    _chunks, rep = b.draw_all(q, n_chunk=100)
    assert rep["drawn"] == 30 and rep["short_by"] == 170


# =========================================================================== #
# the fitted ordering score
# =========================================================================== #
def test_v11_drops_the_family_and_the_exemplar_columns():
    """Both removals are decisions with a reason. `degree` IS the family on this population,
    so a pooled queue sorted on a score carrying it allocates across families by that
    coefficient; the exemplar columns are RETIRED as an ordering feature and the harvest
    does not compute them at all."""
    assert set(vf.FEATURES_V11) == set(vf.FEATURE_ORDER) - set(vf.FAMILY_FEATURES) \
        - set(vf.EXEMPLAR_FEATURES)
    assert not (set(vf.FEATURES_V11) & set(vf.EXEMPLAR_FEATURES))
    assert "degree" not in vf.FEATURES_V11 and "log10_period" not in vf.FEATURES_V11
    # `composite_v3` stays IN the model as a baseline column — recorded beside every row is
    # a different claim from removed from the fit.
    assert "composite_v3" in vf.FEATURES_V11


def test_the_c_selection_does_not_land_on_its_grid_edge():
    """A selection that picks its grid boundary has not selected, it has run out of grid.
    The first v1.1 grid stopped at 10 and every outer fold picked 10; the record must show
    an interior optimum for the reported C to mean anything."""
    rec = json.loads((ROOT / "data" / "atlas" / "view_fit_v1_1.json")
                     .read_text(encoding="utf-8"))
    grid = rec["cv"]["c_grid"]
    assert rec["models"]["v11"]["c_selected"] not in (min(grid), max(grid))


def test_the_v11_record_and_the_code_agree_on_the_feature_set():
    """`FittedScore` reads its features from the record, so a code-side edit that is not
    refitted would silently score a queue with the wrong columns in the wrong order."""
    rec = json.loads((ROOT / "data" / "atlas" / "view_fit_v1_1.json")
                     .read_text(encoding="utf-8"))
    assert tuple(rec["models"]["v11"]["features"]) == vf.FEATURES_V11


def test_the_queue_features_are_the_fits_own_arithmetic():
    """`build_label_seeded_batches._features_for` restates `view_fit.row_features` on a
    differently-shaped row. Every shared column must be the same arithmetic, or the queue is
    ordered by a model evaluated on features that are not the ones it was fitted on."""
    import build_label_seeded_batches as b
    field = np.linspace(1.0, 40.0, 64 * 36).reshape(36, 64).astype("<f4")
    row = dict(fw=1e-3, window_scale=1e-4, band_coverage=0.5, band_coverage_q25=0.25,
               radial_range=12.0, radial_rings=3.0, interior_fraction=0.05,
               cap_headroom=0.4, clamped=False, composite=1.25)
    sc = dict(row, period=7, degree=2, exemplar_sim_max=0.9, exemplar_sim_mean=0.8)
    mine = b._features_for(row, field)
    theirs = vf.row_features(sc, vf.falloff_features(field))
    for k in vf.FEATURES_V11:
        assert mine[k] == pytest.approx(theirs[k], rel=1e-12, abs=1e-12), k


def test_p_notbad_is_monotone_in_the_score_but_saturates():
    """`score()` returns the LOGIT and the queue sorts on THAT; `p_notbad` is the same number
    through a sigmoid, for a threshold.

    The relation is monotone NON-DECREASING, not order-isomorphic, and the difference is the
    reason the queue sorts on the logit: `p_notbad` clamps the logit to +/-60 before the
    sigmoid, so two rows far out in the tail come back at exactly 1.0 and their order is
    lost. That is invisible in the middle of the distribution and it is precisely the TOP of
    the queue, which is where the sort actually matters.
    """
    m = vf.load_model_v11()
    rng = np.random.default_rng(0)
    feats = [{f: float(rng.normal()) for f in vf.FEATURES_V11} for _ in range(200)]
    pairs = sorted((m.score(f), m.p_notbad(f)) for f in feats)
    ps = [p for _s, p in pairs]
    assert all(b >= a for a, b in zip(ps, ps[1:])), "p_notbad decreased as the logit rose"
    # ...and the saturation is real on this model, so the docstring above is not theoretical.
    assert max(abs(s) for s, _p in pairs) > 60.0


def test_the_live_sort_key_elsewhere_is_still_composite_v3():
    """v1.1 orders THIS queue and nothing else. Nothing in the discovery path may import it
    — the same staged-not-adopted contract `test_view_fit` holds for v1."""
    import steered_frontier as sf
    import maneuver_view_screen as mvs
    src = (Path(sf.__file__).read_text(encoding="utf-8")
           + Path(mvs.__file__).read_text(encoding="utf-8"))
    assert "view_fit" not in src


# =========================================================================== #
# the budget logic — every halt branch, and the resume that has to survive it
# =========================================================================== #
# These three mechanisms gated several real multi-hour runs without ever firing, so until
# 2026-08-02 they were presumed rather than tested. `tools/atlas/livefire_harvest_budget.py`
# is the end-to-end half — three tiny sink-isolated runs, each deliberately triggered and
# resumed, all three firing on 2026-08-02 (`active_budget (est 8s > 6s left)` at 13/60 seeds,
# `wall_budget (est 8s > 7s left)` at 13/60, `stop_sentinel` at 4/60, each resuming to 60/60).
# It takes ~13 minutes and drives the real engine, so it is a hand-run driver; the BRANCH
# logic is pinned here instead so it stays tested rather than re-presumed. No engine here: a
# `Harvest` is built against a tmp run dir and its counters are set directly.
#
# One behaviour the live fire turned up and this block does not: with no unit history
# `unit_estimate()` returns a stated 30 s, so a run given a budget of 30 s or less refuses to
# start its first unit and halts at zero seeds. Correct — it is "never start a unit you
# cannot afford" applied to the run's own cold-start guess — but it means a sub-minute budget
# cannot be used to exercise a mid-run halt.
def _harvest(tmp_path, *, budget_min, wall_budget_min):
    return lsh.Harvest(tmp_path / "run", budget_min=budget_min,
                       wall_budget_min=wall_budget_min, workers=1)


def test_the_stop_sentinel_halts_and_outranks_a_wide_open_budget(tmp_path):
    """The sentinel is the only halt a human can reach mid-run, so it must not be
    conditional on the budgets — it is checked first, and a run with hours left stops."""
    h = _harvest(tmp_path, budget_min=600, wall_budget_min=600)
    assert h.stopping(time.time()) == ""
    (h.dir / lsh.STOP_SENTINEL).write_text("", encoding="utf-8")
    assert h.stopping(time.time()) == "stop_sentinel"


def test_the_active_cap_halts_before_starting_a_unit_it_cannot_afford(tmp_path):
    """`active_s` is spent time, not elapsed time. The cap must bind on what the run has
    SPENT, and it must refuse to start a unit whose estimate exceeds what is left rather
    than starting it and overrunning."""
    h = _harvest(tmp_path, budget_min=10, wall_budget_min=600)
    h.unit_s = [20.0] * 10                       # est/unit = 20 s
    h.active_s = 9 * 60.0                        # 60 s of active budget left
    assert h.stopping(time.time()) == ""         # 60 > 20: afford one more
    h.active_s = 10 * 60.0 - 15.0                # 15 s left, under the estimate
    assert h.stopping(time.time()).startswith("active_budget")


def test_the_wall_cap_halts_on_elapsed_time_even_with_active_budget_to_spare(tmp_path):
    """The two caps are different quantities and only separate when one binds and the
    other does not: here the active budget is untouched and the run must still stop."""
    h = _harvest(tmp_path, budget_min=600, wall_budget_min=10)
    h.unit_s = [20.0] * 10
    h.active_s = 0.0                             # nothing spent — the active cap is idle
    assert h.stopping(time.time() - 9 * 60.0) == ""
    assert h.stopping(time.time() - (10 * 60.0 - 15.0)).startswith("wall_budget")


def test_the_unit_estimate_is_recent_p90_not_the_run_to_date_mean(tmp_path):
    """A run whose units get more expensive is the case the cap exists for. The run-to-date
    mean is dominated by the cheap early work and would let exactly that run overrun —
    CLAUDE.md's "refit from RECENT throughput". The window is 40, so 60 cheap units must
    not be able to drag the estimate down."""
    h = _harvest(tmp_path, budget_min=10, wall_budget_min=10)
    h.unit_s = [1.0] * 60 + [50.0] * 40
    est, mean = h.unit_estimate(), sum(h.unit_s) / len(h.unit_s)
    assert est >= 49.0, est
    assert est > 2 * mean, (est, mean)
    # ...and with no history it is a stated constant, not a crash or a zero
    assert lsh.Harvest(tmp_path / "r2", budget_min=1, wall_budget_min=1,
                       workers=1).unit_estimate() == 30.0


def test_a_resume_restores_the_SPENT_budget_and_not_a_fresh_one(tmp_path):
    """The resume bug this pins: reload the done set but not `active_s`, and every resume
    hands the run a full budget again, so a 150-minute cap becomes unbounded across enough
    restarts. The reload must carry the spend, and the cap must bind immediately on it."""
    h = _harvest(tmp_path, budget_min=10, wall_budget_min=600)
    h.done = {"sAAA", "sBBB"}
    h.active_s = 10 * 60.0 - 5.0
    h.unit_s = [20.0] * 5
    h.per_seed = [dict(seed_id="sAAA", seconds=1.0), dict(seed_id="sBBB", seconds=2.0)]
    h.seen_keys = {"k1", "k2"}
    h.save()

    h2 = _harvest(tmp_path, budget_min=10, wall_budget_min=600)
    h2.load()
    assert h2.done == {"sAAA", "sBBB"}
    assert h2.seen_keys == {"k1", "k2"}
    assert h2.active_s == pytest.approx(10 * 60.0 - 5.0)
    assert h2.stopping(time.time()).startswith("active_budget")


def test_the_state_write_is_atomic_and_leaves_no_tmp_behind(tmp_path):
    """`save` writes a sibling `.tmp` and `os.replace`s it, so a kill mid-write cannot
    leave a truncated state.json — the file a resume refuses to start without."""
    h = _harvest(tmp_path, budget_min=10, wall_budget_min=10)
    h.save()
    assert h.state_path.exists()
    assert not h.state_path.with_suffix(".json.tmp").exists()
    json.loads(h.state_path.read_text(encoding="utf-8"))       # parses


def test_resume_without_state_refuses_instead_of_starting_over(tmp_path):
    """A `--resume` that silently started from zero would re-spend the whole budget and
    duplicate every seed's work into an append-only log."""
    h = _harvest(tmp_path, budget_min=10, wall_budget_min=10)
    with pytest.raises(SystemExit, match="cannot --resume"):
        h.load()
