"""Tests for the 2026-08-03 long harvest: registration, the record-and-rank tail, the
interior gate at sourcing, maneuvers-on-admissions, and the channel allocation.

WHAT IS AND IS NOT COVERED, stated first because the gaps matter more than the coverage.
Nothing here drives the render engine or loads a classifier: the harvest tail's rules are
extracted as PURE functions precisely so they can be tested without either, and the
engine-touching path is exercised by the shakedown and by the run. The allocation is tested
on an injected census, not on the live corpus — a test that read `data/label_corpus` would
re-derive its expectation through the same walk the subject uses and assert `f(x) == f(x)`
(`verification_practice.md` §1.10), and it would move every time a label lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.v7 import build_manifest as bm              # noqa: E402


# =========================================================================== #
# registration — before any batch is built (fail-closed)
# =========================================================================== #
ALL_NEW = (bm.Q4_HARVEST_RANKED_BATCHES | bm.Q4_NEAR_MINIBROT_BATCHES
           | bm.Q4_UNIFORM_EVAL_BATCHES)


def test_the_three_legs_are_registered_and_the_sets_are_not_empty():
    """Derive + prove non-empty (`verification_practice.md` §5): a registration test over an
    empty id set passes by evaluating nothing, which is exactly when it is needed."""
    assert bm.Q4_HARVEST_RANKED_BATCHES and bm.Q4_NEAR_MINIBROT_BATCHES \
        and bm.Q4_UNIFORM_EVAL_BATCHES
    assert len(ALL_NEW) == 3, "the three legs must not share an id"
    for bid in ALL_NEW:
        assert bm.assign_split({"batch": bid, "ft": "mandelbrot"})[2] != "unregistered", bid


@pytest.mark.parametrize("bids,expected", [
    (bm.Q4_HARVEST_RANKED_BATCHES, ("train", True, "q4_harvest_ranked")),
    (bm.Q4_NEAR_MINIBROT_BATCHES, ("train", True, "q4_near_minibrot")),
    (bm.Q4_UNIFORM_EVAL_BATCHES, ("eval", False, "q4_uniform_eval")),
])
def test_each_leg_classifies_as_its_registration_says(bids, expected):
    for bid in bids:
        assert bm.assign_split({"batch": bid, "ft": "mandelbrot"}) == expected, bid


def test_the_fail_closed_default_still_holds():
    """The registrations are additions, not a replacement of the safety net."""
    assert bm.assign_split({"batch": "never_registered_2026", "ft": "mandelbrot"}) == \
        ("train", True, "unregistered")


def test_only_the_uniform_leg_is_eval_eligible():
    """The one classification that can move a threshold, pinned to the one leg that earned
    it. The ranked leg is selected on a score twice over and the near-minibrot leg's rows
    survived the run's screens; neither is an instrument."""
    evals = {b for b in ALL_NEW
             if bm.assign_split({"batch": b, "ft": "mandelbrot"})[0] == "eval"}
    assert evals == set(bm.Q4_UNIFORM_EVAL_BATCHES)


def test_the_uniform_leg_does_not_contradict_the_biased_registry():
    """`registration_contradictions` is a hard abort in build_manifest's main(). A leg
    registered unbiased here and train-side-only in `label_store` would fail the build LATER,
    after the batch is drawn and rendered — so it is checked at registration time instead."""
    import label_store as ls
    for bid in bm.Q4_UNIFORM_EVAL_BATCHES:
        assert bid not in ls.TRAIN_SIDE_ONLY_BATCHES
    rows = [{"batch": b, "ft": "mandelbrot", "biased":
             bm.assign_split({"batch": b, "ft": "mandelbrot"})[1]} for b in ALL_NEW]
    assert bm.registration_contradictions(rows) == []


def test_the_uniform_leg_targets_the_partitions_with_no_unbiased_eval():
    """The leg exists for `T_GOOD_UNCALIBRATED`, so that set is what its priority order is
    drawn from. If a partition ever leaves that set, this goes red and the leg's target list
    is re-decided rather than silently covering a partition that no longer needs it."""
    import production_seeder as ps
    assert {"phoenix", "multibrot3", "multibrot4", "multibrot5",
            "julia:mandelbrot"} <= set(ps.T_GOOD_UNCALIBRATED)


# =========================================================================== #
# the recording floor
# =========================================================================== #
def _sf():
    import steered_frontier as sf
    return sf


def test_the_recording_floor_never_raises_a_cut():
    """A recording floor above `tau_h` would silence exactly the rows it exists to keep.
    mandelbrot is the live instance: its tau_h (0.023) is already below the absolute floor,
    so `min` has to bind or its entire sub-cut population would vanish from the store."""
    sf = _sf()
    tau = {"mandelbrot": 0.0229, "multibrot3": 0.3691, "multibrot4": 0.4087,
           "multibrot5": 0.3514, "julia:mandelbrot": 0.4126}
    rec = sf.derive_tau_rec(tau)
    assert set(rec) == set(tau)
    for p in tau:
        assert rec[p] <= tau[p] + 1e-12, f"{p}: floor {rec[p]} ABOVE the cut {tau[p]}"
    # ... and it is not vacuously equal everywhere: the partitions with a large cut must
    # actually record below it, or the feature is a no-op wearing a constant.
    assert sum(1 for p in tau if rec[p] < tau[p] - 1e-9) >= 4


def test_the_recording_floor_respects_the_absolute_floor():
    sf = _sf()
    rec = sf.derive_tau_rec({"x": 1.0, "y": 0.06, "z": 0.001})
    assert rec["x"] == pytest.approx(0.5)                       # frac binds
    assert rec["y"] == pytest.approx(sf.Q4_REC_FLOOR_ABS)        # absolute floor binds
    assert rec["z"] == pytest.approx(0.001)                      # the cut itself binds


# =========================================================================== #
# the interior gate at sourcing
# =========================================================================== #
class _GateStub:
    """The smallest object `interior_gate` needs. A real `SteeredFrontier` loads a torch
    checkpoint in __init__, and the gate is pure list arithmetic — building the driver to
    test it would make the test a model-load smoke."""

    def __init__(self, on=True, thresh=0.30):
        self.interior_gate_on = on
        self.interior_discard = thresh
        self.totals = {"interior_gated": 0, "interior_unmeasured": 0, "q4_recorded": 0}
        self.recorded = []

    def _q4_record(self, c, *, fate, reframe_decoded=None):
        self.recorded.append((c["node_id"], fate))


def _gate(stub, cands):
    return _sf().SteeredFrontier.interior_gate(stub, cands)


def test_the_interior_gate_is_strict_greater_than():
    """A frame at exactly 0.30 is KEPT, mirroring present.rs's strict `<` on the other side
    of the same boundary. Off-by-one-side is invisible in a count, so it is asserted."""
    stub = _GateStub()
    cands = [{"node_id": 1, "int_frac": 0.2999}, {"node_id": 2, "int_frac": 0.30},
             {"node_id": 3, "int_frac": 0.3001}]
    kept = _gate(stub, cands)
    assert [c["node_id"] for c in kept] == [1, 2]
    assert stub.totals["interior_gated"] == 1
    assert stub.recorded == [(3, "interior_gt_30")]


def test_an_absent_interior_measure_is_kept_and_counted_apart():
    """An absent measure is not a high one — the same rule `apply_interior_rule.fires` uses.
    Counted apart so "the gate never fired" and "nothing was measurable" stay distinguishable."""
    stub = _GateStub()
    kept = _gate(stub, [{"node_id": 1, "int_frac": None}, {"node_id": 2}])
    assert len(kept) == 2
    assert stub.totals["interior_unmeasured"] == 2
    assert stub.totals["interior_gated"] == 0


def test_the_gate_off_is_a_pass_through_and_records_nothing():
    """Off must reproduce the pre-v1.6 candidate population exactly, including writing no
    rows — an 'off' switch that still logged would change the store a comparison reads."""
    stub = _GateStub(on=False)
    cands = [{"node_id": 1, "int_frac": 0.99}, {"node_id": 2, "int_frac": 0.01}]
    assert _gate(stub, cands) is cands
    assert stub.recorded == [] and stub.totals["interior_gated"] == 0


def test_every_gated_candidate_is_recorded_not_merely_counted():
    """The discard population IS the answer to "what did the gate cost?". A counter alone
    makes that a re-run; a row makes it a read."""
    stub = _GateStub()
    cands = [{"node_id": i, "int_frac": 0.9} for i in range(7)]
    assert _gate(stub, cands) == []
    assert stub.totals["interior_gated"] == 7
    assert len(stub.recorded) == 7


def test_the_interior_threshold_has_one_value_across_its_three_users():
    """Two independent copies of a constant need a test, because nothing structural keeps
    them equal (`verification_practice.md` §1.8). The three users are the label-store rule,
    the label-seeded harvest's sourcing filter, and this walk's sourcing gate."""
    import apply_interior_rule as air
    import label_seeded_harvest as lsh
    assert air.THRESHOLD == lsh.INTERIOR_DISCARD == _sf()._INTERIOR_DISCARD == 0.30
    assert air.RULE_SCORE == 1, "the rule asserts class 1; the gate's premise is that score"


def test_the_walk_already_enforces_the_interior_rule_at_the_engine():
    """THE INVARIANT THE SOURCING TRIPWIRE ACTUALLY PROTECTS, and the reason its expected
    count is zero.

    `EXPAND_FLAGS` passes `--descent-black-cap ps.BLACK_CAP` to the engine, and `BLACK_CAP`
    is 0.30 — the same bound on the same quantity as Matt's rule. So the Python-side gate is
    redundant BY COINCIDENCE OF TWO INDEPENDENTLY-OWNED CONSTANTS: the engine's is a descent
    parameter somebody could retune for an unrelated reason, and the moment it moves above
    the rule the walk starts emitting candidates the rule calls class 1. This test is what
    makes that divergence loud instead of silent, and it is the reason the tripwire stays.
    """
    import production_seeder as ps
    sf = _sf()
    assert ps.BLACK_CAP == sf._INTERIOR_DISCARD, (
        f"the engine's descent black cap ({ps.BLACK_CAP}) and Matt's interior rule "
        f"({sf._INTERIOR_DISCARD}) have diverged — the sourcing tripwire is now live and "
        f"its count is no longer expected to be zero. Decide which is right.")
    # ... and the cap really is on the wire, not merely defined.
    assert "--descent-black-cap" in sf.EXPAND_FLAGS
    assert sf.EXPAND_FLAGS[sf.EXPAND_FLAGS.index("--descent-black-cap") + 1] == \
        str(ps.BLACK_CAP)


# =========================================================================== #
# the round-robin draw
# =========================================================================== #
def _b():
    sys.path.insert(0, str(ROOT / "tools" / "sourcing"))
    import build_q4_harvest_batches as b
    return b


def test_the_draw_balances_cells_that_have_supply():
    b = _b()
    rows = [dict(cell="a", k=i) for i in range(50)] + \
           [dict(cell="b", k=i) for i in range(50)]
    out, rep = b.draw_round_robin(rows, lambda r: r["cell"], 20,
                                  order_key=lambda r: r["k"])
    assert len(out) == 20
    assert rep["a"]["taken"] == rep["b"]["taken"] == 10
    assert b.cells_balanced(rep)[0]


def test_a_drained_cell_does_not_fail_the_balance_check():
    """The defect a flat-spread assertion has: real cells differ in supply by two orders of
    magnitude, so a correct draw shows a large spread and a flat check goes red on it. On
    this run's own queue the flat spread was 78 with the draw behaving perfectly."""
    b = _b()
    rows = [dict(cell="big", k=i) for i in range(200)] + [dict(cell="tiny", k=0)]
    out, rep = b.draw_round_robin(rows, lambda r: r["cell"], 100,
                                  order_key=lambda r: r["k"])
    assert rep["tiny"] == dict(taken=1, available=1, drained=True)
    assert rep["big"]["taken"] == 99
    flat_spread = max(v["taken"] for v in rep.values()) - \
        min(v["taken"] for v in rep.values())
    assert flat_spread == 98, "the flat spread a naive check would have asserted on"
    ok, detail = b.cells_balanced(rep)
    assert ok, detail          # ... and the correct invariant passes


def test_the_balance_check_is_red_for_a_real_imbalance():
    """Prove it red: a cell that is under-taken while it still HAD rows is the defect."""
    b = _b()
    rep = {"a": dict(taken=10, available=50, drained=False),
           "b": dict(taken=2, available=50, drained=False)}
    ok, detail = b.cells_balanced(rep)
    assert not ok and "b" in detail


def test_the_draw_takes_best_first_inside_a_cell():
    """Round-robin over CELLS, ranked order WITHIN one — the chunk is the top of the queue
    conditioned on not letting one cell own the page."""
    b = _b()
    rows = [dict(cell="a", k=k) for k in (5, 1, 3)] + \
           [dict(cell="b", k=k) for k in (2, 9)]
    out, _ = b.draw_round_robin(rows, lambda r: r["cell"], 4,
                                order_key=lambda r: r["k"])
    assert [r["k"] for r in out if r["cell"] == "a"] == [1, 3]
    assert [r["k"] for r in out if r["cell"] == "b"] == [2, 9]


# =========================================================================== #
# the reconcile identity
# =========================================================================== #
def test_the_gate_is_inside_the_batch_reconcile_identity():
    """A gate that removes candidates without entering the identity is a gate that can eat
    them and still balance. Proved by INJECTION rather than by reading the source: a batch
    where the gate removed rows must balance, and the same batch with the gated count
    zeroed must NOT."""
    sf = _sf()

    class _R:
        RECONCILE_KEYS = sf.SteeredFrontier.RECONCILE_KEYS
        _reconcile_batch = sf.SteeredFrontier._reconcile_batch
        batch_i = 1

    before = {k: 0 for k in _R.RECONCILE_KEYS}
    r = _R()
    r.totals = dict(before, candidates=10, frontier_pushed=7, interior_gated=3)
    r._reconcile_batch(before, 10)                      # balances

    r.totals = dict(before, candidates=10, frontier_pushed=7, interior_gated=0)
    with pytest.raises(SystemExit, match="interior_gated"):
        r._reconcile_batch(before, 10)                  # the defect the term exists to catch


# =========================================================================== #
# phoenix as a partition in the one walk
# =========================================================================== #
PX6 = ("0.5667", "0", "-0.5", "0", "0.03", "-0.02")


def test_the_phoenix_expand_and_render_grammars_are_the_same_flags():
    """Two INDEPENDENT flag builders reach the engine for one phoenix candidate: this
    module's `descend_flags` (the --expand call) and `location.render_one_flags` (every
    render). They are written separately, so nothing structural keeps them equal — and a
    disagreement would descend one phoenix plane and render a different one, which shows up
    only as an inexplicably bad score. Pinned, for the reason
    `verification_practice.md` §1.8 gives."""
    sf = _sf()
    import location as lm
    expand = sf.descend_flags("phoenix", PX6)
    render = lm.render_one_flags(sf.loc_of("phoenix", PX6, "0.33", "0.50", "1.25"))
    # `--expand` names the mode `--phoenix`; `render-one` names it `--family phoenix`.
    # Everything downstream of that must match token for token.
    assert expand[0] == "--phoenix" and render[:2] == ["--family", "phoenix"]
    assert expand[1:] == render[2:], f"{expand[1:]} != {render[2:]}"


def test_a_phoenix_dup_identity_is_the_whole_six_vector():
    """`production_seeder.as_c` truncates to a PAIR, which for phoenix would declare two
    points sharing `c` but differing in `p` or `z_-1` to be the same location. `near_dup`
    already accepts the 6-D identity, so the fix is to hand it the whole vector."""
    sf = _sf()
    assert sf.ident_c("phoenix", PX6) == (0.5667, 0.0, -0.5, 0.0, 0.03, -0.02)
    same_c_other_p = ("0.5667", "0", "-0.25", "0", "0.03", "-0.02")
    assert sf.ident_c("phoenix", PX6) != sf.ident_c("phoenix", same_c_other_p)
    # ... and the julia path is untouched.
    assert sf.ident_c("julia:mandelbrot", ("0.1", "0.2")) == (0.1, 0.2)


def test_phoenix_render_args_never_hand_a_six_vector_to_a_pair_unpacker():
    """`prescreen._render` unpacks `c` as a pair. Passing the 6-vector raises; passing its
    first two entries silently renders the DEFAULT phoenix plane at the right coordinates,
    which is the worse failure because it looks like it worked."""
    sf = _sf()
    a = sf.render_args_for("phoenix", PX6)
    assert len(a["c"]) == 2 and a["family"] == "phoenix"
    assert a["family_params"] == {"p_re": "-0.5", "p_im": "0",
                                  "zm1_re": "0.03", "zm1_im": "-0.02"}
    assert sf.render_args_for("mandelbrot", None)["family_params"] is None


def test_the_shared_render_helpers_are_byte_identical_without_the_new_kwarg():
    """The `family_params` kwarg is additive. An existing caller that does not pass it must
    build the same Location it always did — the whole basis for touching two shared
    production helpers mid-run."""
    import production_seeder as ps
    import location as lm
    a = ps.make_loc_of("mandelbrot", None)(1, 2, 3)
    b = ps.make_loc_of("mandelbrot", None, None)(1, 2, 3)
    assert a == b
    assert not a.family_params            # empty, whatever container Location normalizes to
    assert lm.render_one_flags(a) == lm.render_one_flags(b)
    # ... and the julia path, the other historical caller shape.
    j = ps.make_loc_of("julia", ("0.1", "0.2"))(1, 2, 3)
    assert j == ps.make_loc_of("julia", ("0.1", "0.2"), None)(1, 2, 3)


def test_phoenix_is_never_a_families_entry():
    """The engine refuses `--phoenix` a parameter plane to prospect, so there is no c-plane
    seeder for it and `--families phoenix` must fail loudly rather than half-work."""
    sf = _sf()
    assert "phoenix" not in sf.C_PLANE


def test_phoenix_has_no_derived_tau_h_and_is_excluded_from_the_derivation():
    """Deriving a phoenix cut from the v7 phoenix ledgers would be a v7 number gating a v10
    head. `derive_tau_h` must refuse the partition rather than fall back to a pooled value —
    that refusal is what makes the explicit, stamped override honest."""
    sf = _sf()
    assert "phoenix" not in sf.TAU_H_FIDELITY_BASE
    with pytest.raises(SystemExit):
        sf.derive_tau_h(["phoenix"])


# =========================================================================== #
# the phoenix seed pool
# =========================================================================== #
def test_the_phoenix_pool_drops_the_root_branch_and_holds_the_mid_p_band():
    """Both are settled label verdicts (`phoenix_seed_sampler_spec.md` §8): the root branch
    produced 0 good to humans, and mid-|p| is the sweet band. A band that drifts is a
    different population under the same name."""
    sys.path.insert(0, str(ROOT / "tools" / "phoenix"))
    import phoenix_q4_seeds as pq
    pool, rep = pq.draw_pool(48, seed=7)
    assert len(pool) == 48 and not rep.get("short_by")
    assert {r["branch"] for r in pool} <= set(pq.BRANCHES)
    assert "root" not in {r["branch"] for r in pool}
    assert all(pq.P_LO <= r["abs_p"] <= pq.P_HI for r in pool)
    # |p| < 1 is NECESSARY for an attracting fixed point to exist (spec §2.1); the band's
    # upper bound has to keep that true or the skeleton stops describing the dynamics.
    assert pq.P_HI < 1.0
    # non-vacuity: both live branches actually appear, so the filter is not silently
    # collapsing the draw to one of them.
    assert len({r["branch"] for r in pool}) == 2


def test_the_phoenix_pool_is_reproducible_and_distinct():
    sys.path.insert(0, str(ROOT / "tools" / "phoenix"))
    import phoenix_q4_seeds as pq
    a, _ = pq.draw_pool(24, seed=11)
    b, _ = pq.draw_pool(24, seed=11)
    assert a == b, "the pool is a pure function of its seed"
    keys = {(r["c_re"], r["c_im"], r["p_re"], r["p_im"], r["zm1_re"], r["zm1_im"])
            for r in a}
    assert len(keys) == len(a), "a pool with duplicate parameter points wastes root slots"


def test_the_pool_record_shape_is_what_the_walk_reads():
    """The pool's key names are the WALK's, not `phoenix_sampler.seed_to_record`'s ledger
    identity shape — reusing the latter would make a pool file and a ledger row
    indistinguishable to a reader."""
    sys.path.insert(0, str(ROOT / "tools" / "phoenix"))
    import phoenix_q4_seeds as pq
    r = pq.draw_pool(1, seed=3)[0][0]
    assert {"c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"} <= set(r)
    assert not any(k.startswith("phoenix_") for k in r)


# =========================================================================== #
# channel allocation
# =========================================================================== #
def test_family_weights_fail_loud_on_a_typo_or_an_omission():
    """A typo'd family that silently weighted zero would mute a channel for a 4-hour run and
    read afterwards as "that channel found nothing"."""
    sf = _sf()
    fams = ["mandelbrot", "multibrot3"]
    assert sf._parse_family_weights(None, fams) == {}
    w = sf._parse_family_weights("mandelbrot=1,multibrot3=3", fams)
    assert w == {"mandelbrot": 0.25, "multibrot3": 0.75}
    with pytest.raises(SystemExit, match="not in --families"):
        sf._parse_family_weights("mandelbrot=1,nope=1", fams)
    with pytest.raises(SystemExit, match="missing"):
        sf._parse_family_weights("mandelbrot=1", fams)


def test_the_weighted_draw_preserves_the_total_root_budget():
    """Weights move the MIX, not the budget: sum(round(B*F*w)) ~= B*F. Asserted because a
    weighting that also shrank the budget would look like a yield drop."""
    sf = _sf()
    fams = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5"]
    w = sf._parse_family_weights(
        "mandelbrot=0.176,multibrot3=0.281,multibrot4=0.284,multibrot5=0.259", fams)
    B = 32
    drawn = sum(max(1, int(round(B * len(fams) * w[f]))) for f in fams)
    assert abs(drawn - B * len(fams)) <= len(fams)


# =========================================================================== #
# maneuvers-on-admissions
# =========================================================================== #
def test_triggered_and_fresh_counters_are_disjoint_names():
    """The operators feed themselves, so a pooled rate measures the loop rather than the
    operator (`minibrot_maneuvers.md` §8.0). The two counter families must not overlap, or
    a readout could sum them without noticing."""
    sf = _sf()
    trig = {"trig_fired", "trig_atoms", "trig_nodes_pushed", "trig_admitted",
            "trig_unavailable", "trig_budget_skip", "trig_expanded", "trig_candidates"}
    assert trig.isdisjoint(set(sf.MAN_TOTALS))


def test_the_triggered_k_set_is_not_the_walks():
    """A trigger fires on a location that already decoded >=3, so the 4x atom frame's
    question ("is this atom good?") is answered and paying a third field to re-ask it is
    waste. `none` and `16` are the framings that produce labelable material."""
    sf = _sf()
    assert sf.mnv.parse_k_spec(sf.TRIG_K_DEFAULT) == [None, 16.0]
    assert sf.TRIG_K_DEFAULT != sf.MAN_K_DEFAULT


def test_triggered_maneuvers_are_c_plane_only():
    """A julia/phoenix viewport is a z-plane with no nucleus in the parameter-plane sense,
    so the operators are undefined there and must be SKIPPED rather than faked."""
    sf = _sf()
    for p in ("julia:mandelbrot", "julia:multibrot3", "phoenix"):
        assert sf.mnv.PARTITION_DEGREE.get(p) is None
    for p in ("mandelbrot", "multibrot3", "multibrot4", "multibrot5"):
        assert sf.mnv.PARTITION_DEGREE.get(p) is not None


def test_every_q4_fate_is_named():
    """An unhandled branch must surface as the literal 'unknown' in a count rather than as a
    silently absent row, so 'unknown' is a declared fate and not an accident."""
    sf = _sf()
    assert "unknown" in sf.SteeredFrontier.Q4_FATES
    assert len(set(sf.SteeredFrontier.Q4_FATES)) == len(sf.SteeredFrontier.Q4_FATES)
