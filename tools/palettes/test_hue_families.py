"""Guards for the hue/flavor family naming layer.

The point of every test here is that this module ADDS A NAME and no arithmetic: the hue
convention has one owner (`palette_deficit._hsv_signature`) and the special prepull has
another (`palette_categories.json`). A test that re-derived either would be asserting the
copy it was trying to prevent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import palette_deficit as PD          # noqa: E402
from tools.palettes import hue_families as HF             # noqa: E402


def test_hue_groups_partition_every_bin():
    seen = [b for bins in HF.HUE_GROUPS.values() for b in bins]
    assert sorted(seen) == list(range(PD.HUE_BINS))


def test_green_is_the_deficit_modules_green_not_a_second_definition():
    assert HF.HUE_GROUPS["green"] == tuple(PD.GREEN_BINS)


def test_families_constant_covers_every_producible_family():
    producible = set(HF.HUE_GROUPS) | set(HF.SPECIAL_FAMILIES.values())
    assert producible == set(HF.FAMILIES)


def test_specials_come_from_the_committed_artifact():
    specials = HF.load_specials()
    cats = json.loads(HF.CATEGORIES.read_text(encoding="utf-8"))["palettes"]
    assert len(specials) == len(cats)
    for name in ("bone",):                    # a known neutral prepull
        if name in specials:
            assert HF.family_of(name, _stops(name), specials)["family"] == "neutral"


def _stops(name):
    return {p["name"]: p["stops"] for p in HF.load_pool()}[name]


def test_spectral_prepull_wins_over_hue_argmax():
    specials = HF.load_specials()
    spectral = [n for n, s in specials.items() if s == "spectral"]
    assert spectral, "the artifact has no spectral prepull — the test is vacuous"
    stops = {p["name"]: p["stops"] for p in HF.load_pool()}
    for n in spectral[:5]:
        assert HF.family_of(n, stops[n], specials)["family"] == "spectral"


def test_outlier_is_not_a_family_it_classifies_on_hue():
    specials = HF.load_specials()
    stops = {p["name"]: p["stops"] for p in HF.load_pool()}
    out = [n for n, s in specials.items() if s == "outlier"]
    assert out
    fams = {HF.family_of(n, stops[n], specials)["family"] for n in out}
    assert "outlier" not in fams
    assert fams <= set(HF.HUE_GROUPS)          # hue families only


def test_unknown_palette_reports_that_the_prepull_was_not_consulted():
    stops = [(0.0, (255, 0, 0)), (1.0, (255, 128, 0))]
    v = HF.family_of("not-in-the-pool", stops, HF.load_specials())
    assert v["known_special"] is False
    assert v["family"] == "fire"


def test_share_table_shares_sum_to_one_and_count_unknowns():
    verdicts = {"a": {"family": "green"}, "b": {"family": "fire"}}
    tab = HF.share_table(["a", "a", "b", "zzz"], verdicts)
    assert tab["unknown"]["n"] == 1
    assert pytest.approx(sum(v["share"] for v in tab.values()), rel=0, abs=1e-12) == 1.0


def test_pool_classification_is_total():
    v = HF.families_over_pool()
    assert len(v) == 987
    assert all(x["family"] in HF.FAMILIES for x in v.values())
