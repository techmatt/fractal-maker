"""Guards for the rare-family palette draw.

The failure this draw exists to prevent is a sheet that serves 30 palettes out of 987 with
46% of them one family, so the tests are about VARIETY REALIZED, not about the pick being
"good": distinct-palette coverage, the supply cap, and the prefix bound asserted on the
ORDER THAT WAS BUILT (apportion's own rule — the +/-1 claim is a caller's check, never a
theorem about the module).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                                          # noqa: E402
from tools.mining import rare_palette_draw as RPD         # noqa: E402
from tools.palettes import hue_families as HF             # noqa: E402


def test_target_is_a_share_vector_over_exactly_the_named_families():
    assert set(RPD.RARE_TARGET) == set(HF.FAMILIES)
    assert pytest.approx(sum(RPD.RARE_TARGET.values()), abs=1e-12) == 1.0


def test_the_three_named_heavy_families_are_cut_below_what_the_sheets_served():
    # NOT below the POOL share: purple is 4.7% of the pool and 45.8% of sheet B, so the
    # quantity being cut is the SERVED share. Asserting against the pool column would pass
    # for fire/ice and fail for purple while the target is doing exactly the right thing.
    for f in RPD.DOWN_WEIGHTED:
        assert RPD.RARE_TARGET[f] < RPD.MEASURED_2026_08_10["served_b"][f], f
    heavy_target = sum(RPD.RARE_TARGET[f] for f in RPD.DOWN_WEIGHTED)
    for col in ("pool", "served_a", "served_b"):
        assert heavy_target < sum(RPD.MEASURED_2026_08_10[col][f] for f in RPD.DOWN_WEIGHTED)


def test_every_under_served_family_is_lifted_above_both_sheets():
    for f in RPD.OVER_DRAWN:
        for col in ("served_a", "served_b"):
            assert RPD.RARE_TARGET[f] > RPD.MEASURED_2026_08_10[col][f], (f, col)


def test_green_is_over_drawn_by_more_than_3x_its_pool_share():
    pool = HF.share_table(list(HF.families_over_pool()), HF.families_over_pool())
    assert RPD.RARE_TARGET["green"] > 3 * pool["green"]["share"]


def test_the_frozen_measurement_still_matches_the_live_pool():
    pool = HF.share_table(list(HF.families_over_pool()), HF.families_over_pool())
    for f, want in RPD.MEASURED_2026_08_10["pool"].items():
        assert abs(pool[f]["share"] - want) < 0.002, (f, pool[f]["share"], want)


def test_family_counts_never_exceed_distinct_supply():
    supply = {"green": 5, "fire": 100, "gold": 2, "ice": 50}
    got = RPD.family_counts(60, supply)
    for f, n in got.items():
        assert n <= supply[f], (f, n)


def test_family_counts_meet_n_when_supply_allows_and_report_the_shortfall_otherwise():
    supply = {"green": 5, "fire": 100, "gold": 2, "ice": 50}
    assert sum(RPD.family_counts(60, supply).values()) == 60
    tiny = {"green": 2, "gold": 1}
    assert sum(RPD.family_counts(60, tiny).values()) == 3     # drained, not padded


def test_draw_is_all_distinct_while_supply_lasts():
    d = RPD.PaletteDrawer(250, seed=1)
    names = [d.take()[0] for _ in range(250)]
    assert len(set(names)) == 250, "the draw repeated a palette with supply to spare"


def test_small_families_are_covered_nearly_exhaustively():
    d = RPD.PaletteDrawer(250, seed=1)
    for _ in range(250):
        d.take()
    r = d.report()
    # gold (9) and neutral (15) are small enough that a target-share draw should take all.
    assert r["family_distinct_used"]["gold"] == r["distinct_supply"]["gold"]
    assert r["family_distinct_used"]["neutral"] == r["distinct_supply"]["neutral"]


def test_prefix_bound_holds_on_the_order_actually_built():
    # apportion's contract: the +/-1 claim is a CHECK the caller runs, not a theorem.
    d = RPD.PaletteDrawer(250, seed=1)
    assert apportion.prefix_deviation(d.sequence, d.counts) < 1.0
    assert d.prefix_deviation == apportion.prefix_deviation(d.sequence, d.counts)


def test_every_prefix_is_already_rare_weighted_which_is_the_truncation_property():
    d = RPD.PaletteDrawer(250, seed=1)
    fams = [d.take()[1] for _ in range(60)]      # a render budget that stopped at 60
    c = Counter(fams)
    heavy = sum(c[f] for f in RPD.DOWN_WEIGHTED) / len(fams)
    assert heavy < 0.45, f"a truncated draw is already fire/ice/purple heavy: {c}"


def test_draw_is_deterministic_in_its_seed():
    a = [RPD.PaletteDrawer(80, seed=5).take() for _ in range(1)]
    d1, d2 = RPD.PaletteDrawer(80, seed=5), RPD.PaletteDrawer(80, seed=5)
    assert [d1.take() for _ in range(80)] == [d2.take() for _ in range(80)]
    assert a  # the first construction did not perturb the second


def test_exhaustion_raises_rather_than_wrapping():
    d = RPD.PaletteDrawer(3, seed=0)
    for _ in range(3):
        d.take()
    with pytest.raises(IndexError):
        d.take()
