"""The v2 supply routing: the c-spacing floor, the single rung, and the priced-at-zero list.

  uv run pytest tools/atlas/test_supply_routing.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import supply_routing as sr   # noqa: E402


# =========================================================================== #
# 1. every tracked partition is routed, and only to channels that exist
# =========================================================================== #
def test_every_partition_has_a_route():
    """Fail-closed coverage: a partition the quota can serve but the routing never named
    would be allocated time and given nothing to spend it on."""
    from partitions import ALL_FAMS
    assert set(sr.ROUTES) == set(ALL_FAMS)
    for p, r in sr.ROUTES.items():
        assert r["channels"], p
        assert r["evidence"], f"{p} has a route with no measurement behind it"


def test_every_partition_has_a_machine_1_discard_decision():
    from partitions import ALL_FAMS
    assert set(sr.MACHINE_1_DISCARD) == set(ALL_FAMS)


def test_the_machine_1_discard_matches_the_measured_partition_split():
    """The q4 readout's headline: the pooled 68.9% is NOT a decision, the per-partition rates
    are. Native multibrot and phoenix discard; julia:mandelbrot must not — 16.5% of its
    machine-1s are >=3, so an auto-discard throws away one good picture in six."""
    assert sr.MACHINE_1_DISCARD["julia:mandelbrot"] is False
    assert sr.MACHINE_1_DISCARD["phoenix"] is True
    for p in ("multibrot3", "multibrot4", "multibrot5"):
        assert sr.MACHINE_1_DISCARD[p] is True
    # and the unmeasured partitions fail CLOSED (keep), which costs labels, not pictures
    for p in ("mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"):
        assert sr.MACHINE_1_DISCARD[p] is False


def test_the_unscreened_shell_draw_is_retired_rather_than_absent():
    """A channel that is missing and a channel that measured zero read identically in a
    config. This one measured zero and says so, with its population and its scope."""
    r = sr.RETIRED_CHANNELS["unscreened_dM_shell"]
    assert set(r["partitions"]) == {"multibrot3", "multibrot4", "multibrot5"}
    assert "0 of 144" in r["measured"] and r["population"] and "NOT a ceiling" in r["scope"]
    for p in r["partitions"]:
        assert "unscreened_dM_shell" not in sr.ROUTES[p]["channels"]


# =========================================================================== #
# 2. the c-spacing floor
# =========================================================================== #
def test_the_floor_is_where_the_near_dup_rate_reaches_the_different_atom_baseline():
    """The floor is a claim about a measurement, so the measurement rides beside it: at the
    floor the near-dup rate is the different-atom baseline, one bucket finer it is ten times
    that. A floor whose basis says neither of those is a number somebody picked."""
    b = sr.CSPACING_BASIS
    assert sr.CSPACING_FLOOR == 1e-2
    assert b["near_dup_rate_at_or_above_floor"] == pytest.approx(b["different_atom_baseline"],
                                                                 abs=0.005)
    assert b["near_dup_rate_one_bucket_below"] > 8 * b["different_atom_baseline"]
    assert b["measured_on"] and b["recipe"] and str(sr.NEAR_DUP_COS) in b["recipe"]


def test_one_c_per_atom_would_not_have_been_enough():
    """The correction to the q4 readout's atom-level framing, kept as an assertion because it
    is the reason the floor is an absolute distance and not a per-atom cap: the roster's
    atoms sit a median 9.1e-4 apart, which is two buckets INSIDE the floor."""
    assert sr.CSPACING_BASIS["atom_nn_median_dc"] < sr.CSPACING_FLOOR


def test_cspacing_ok_is_a_plain_c_plane_distance():
    acc = [("0.0", "0.0")]
    assert sr.cspacing_ok(("0.02", "0.0"), acc) is True
    assert sr.cspacing_ok(("0.005", "0.0"), acc) is False
    assert sr.cspacing_ok(("0.007", "0.007"), acc) is False        # ~0.0099, just inside
    assert sr.cspacing_ok(("0.008", "0.008"), acc) is True         # ~0.0113, just outside
    assert sr.cspacing_ok(("0.0", "0.0"), []) is True              # nothing to clash with


def test_thinning_is_first_wins_so_the_callers_order_is_the_policy():
    rows = [dict(c_re=0.0, c_im=0.0, tag="best"),
            dict(c_re=0.001, c_im=0.0, tag="dup_of_best"),
            dict(c_re=0.5, c_im=0.0, tag="far")]
    kept, dropped = sr.thin_by_cspacing(rows)
    assert [r["tag"] for r in kept] == ["best", "far"]
    assert [r["tag"] for r in dropped] == ["dup_of_best"]


def test_thinning_returns_what_it_dropped_rather_than_only_what_it_kept():
    """"How much did the floor cost?" has to be a read. The ladder leg would have lost
    roughly two rows in three to this."""
    rows = [dict(c_re=i * 1e-4, c_im=0.0) for i in range(30)]
    kept, dropped = sr.thin_by_cspacing(rows)
    assert len(kept) == 1 and len(dropped) == 29
    assert len(kept) + len(dropped) == len(rows)


def test_the_floor_is_not_the_julia_hook_spacing():
    """They are two different spacings measured on two different populations, and reusing
    one for the other is the mistake this assertion exists to make visible."""
    import steered_frontier as sf
    assert sr.CSPACING_FLOOR != sf.JULIA_HOOK_SPACING
    assert sr.CSPACING_FLOOR < sf.JULIA_HOOK_SPACING


# =========================================================================== #
# 3. the single rung
# =========================================================================== #
def test_the_rung_choice_refuses_without_a_cost_measurement(tmp_path):
    """NOT absence-tolerant. Without the cost record there is no basis for the choice, and
    answering anyway is exactly the decided-in-a-launch-script failure the module prevents."""
    with pytest.raises(FileNotFoundError, match="cost measurement"):
        sr.rung_choice(tmp_path / "nope.json")


def test_a_real_cost_difference_picks_the_cheapest(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"1.0": {"mean_s": 1.0}, "4.0": {"mean_s": 0.5},
                             "16.0": {"mean_s": 2.0}}), encoding="utf-8")
    got = sr.rung_choice(p)
    assert got["rung"] == 4.0 and "cheapest" in got["why"]


def test_a_cost_TIE_falls_back_to_the_one_separating_yield_column(tmp_path):
    """The case the real measurement landed in. Yields are flat on >=3 and cost is flat too
    (3.9% spread), so the tie-break is the one column where the rungs actually separate —
    one-per-cluster class-4, where rung 1 is 8.6% against 4.0% and 3.7%."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"1.0": {"mean_s": 1.00}, "4.0": {"mean_s": 1.02},
                             "16.0": {"mean_s": 0.99}}), encoding="utf-8")
    got = sr.rung_choice(p)
    assert got["rung"] == 1.0 and "tie" in got["why"]
    assert sr.LADDER_YIELD[1.0]["eq4_1pc"] > sr.LADDER_YIELD[4.0]["eq4_1pc"]


def test_a_record_missing_a_rung_is_a_loud_failure_not_a_partial_answer(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"1.0": {"mean_s": 1.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no cost for rung"):
        sr.rung_choice(p)


def test_the_committed_cost_record_exists_and_resolves_to_a_rung():
    """The production assertion. Not absence-tolerant: the record is the durable output of
    `near_minibrot_rung.py measure` and the routing cannot be stated without it."""
    assert sr.RUNG_CHOICE_RECORD.exists(), (
        f"{sr.RUNG_CHOICE_RECORD} missing — rebuild with "
        f"`uv run python tools/atlas/near_minibrot_rung.py measure`")
    got = sr.rung_choice()
    assert got["rung"] in sr.LADDER_RUNGS_MEASURED
    rec = json.loads(sr.RUNG_CHOICE_RECORD.read_text(encoding="utf-8"))
    assert rec["work_count"]["renders"] == rec["work_count"]["atoms"] * 3
    assert rec["population"] == "2026-08-03_q4_near_minibrot_v1"


def test_the_ladder_yields_recorded_here_are_flat_on_ge3():
    """The premise of the whole single-rung argument, kept as an assertion so a later edit
    that made one rung look better on >=3 would have to confront it."""
    ge3 = [v["ge3"] for v in sr.LADDER_YIELD.values()]
    assert max(ge3) - min(ge3) < 0.05
    assert sr.SAME_ATOM_SATURATION["frac_at_or_above_cut"] > 0.7


# =========================================================================== #
# 4. the summary the run config stamps
# =========================================================================== #
def test_the_summary_is_json_able_and_carries_every_decision():
    s = json.loads(json.dumps(sr.summary()))
    assert set(s) >= {"routes", "machine_1_discard", "retired_channels", "cspacing_floor",
                      "cspacing_basis", "rung", "same_atom_saturation"}
    assert s["rung"]["rung"] is not None


# =========================================================================== #
# 5. the v2 julia supply pool the routing produces
# =========================================================================== #
def test_the_v2_pool_honours_the_floor_it_was_built_with():
    """The pool is the routing's output, so the floor has to hold ON DISK, not only in the
    builder. Checked pairwise over the whole pool rather than trusting the build report."""
    import build_julia_supply_pool_v2 as b
    p = ROOT / b.POOL_REL
    assert p.exists(), (f"{p} missing — rebuild with "
                        f"`uv run python tools/atlas/build_julia_supply_pool_v2.py`")
    pool = json.loads(p.read_text(encoding="utf-8"))
    assert len(pool) > 100, len(pool)
    acc = []
    for r in pool:
        assert sr.cspacing_ok((r["c_re"], r["c_im"]), acc), r
        acc.append((r["c_re"], r["c_im"]))


def test_every_pool_row_names_the_channel_that_earned_it():
    """And every channel the routing table lists for this partition actually contributed a
    row. A channel that is routed but empty is a routing decision that did not happen."""
    import build_julia_supply_pool_v2 as b
    pool = json.loads((ROOT / b.POOL_REL).read_text(encoding="utf-8"))
    chans = {r["channel"] for r in pool}
    assert chans == set(b.CHANNEL_ORDER) == set(sr.ROUTES["julia:mandelbrot"]["channels"])


def test_the_merge_order_is_the_measured_yield_order():
    """First-wins thinning means this list IS the policy: a cluster collapses to its
    highest-priced member. The unscreened loop must be last."""
    import build_julia_supply_pool_v2 as b
    assert b.CHANNEL_ORDER[0] == "q4_mining_ranked"     # 79.1% >=3, 16.3% class-4
    assert b.CHANNEL_ORDER[-1] == "seeded_loop"         # 16.7% >=3, 0% class-4
    assert b.CHANNEL_ORDER.index("near_minibrot") < b.CHANNEL_ORDER.index("seeded_loop")


def test_the_ladder_contributes_ONE_c_per_nucleus_not_three():
    """The single-rung decision, asserted where it takes effect."""
    import build_julia_supply_pool_v2 as b
    rows = b.channel_near_minibrot(sr.rung_choice()["rung"], n_nuclei=40)
    assert len(rows) == len({r["atom_id"] for r in rows})
    assert {r["ladder_rung"] for r in rows} == {sr.rung_choice()["rung"]}


def test_the_two_mining_tiers_are_never_pooled():
    """Two geometries, two scores. `rank_tier` 2 is a 640x360 ss2 canonical decode and
    tier 1 a 384x216 ss1 cheap score; one ordering over both is the cap/geometry error."""
    import build_julia_supply_pool_v2 as b
    ranked, recall = b.channel_q4_mining()
    assert {r["channel"] for r in ranked} == {"q4_mining_ranked"}
    assert {r["channel"] for r in recall} == {"q4_mining_recall"}
    for lst in (ranked, recall):
        s = [x["score"] for x in lst if x["score"] is not None]
        assert s == sorted(s, reverse=True), "each tier must be best-first WITHIN itself"


def test_the_build_report_accounts_for_every_proposed_c():
    """found == kept + dropped, per channel. A thinning that can lose rows without either
    bucket noticing is a thinning nobody can price."""
    import build_julia_supply_pool_v2 as b
    p = ROOT / b.POOL_REL
    rep = json.loads(p.with_name(p.stem + "_report.json").read_text(encoding="utf-8"))
    assert rep["n_proposed"] == rep["n_kept"] + rep["n_dropped"]
    for ch, n in rep["proposed"].items():
        assert n == rep["kept"].get(ch, 0) + rep["dropped"].get(ch, 0), ch
