"""Guards for the smooth-equivalence instrument.

Two things are load-bearing and neither is the arithmetic: the CUT has one owner, and the
three bands are DISJOINT even though the strict cut sits inside the interleave zone (0.974
is inside [0.934, 0.986]). A band function that checked the zone first would call every
near-dup an interleave and the exclusion would silently stop excluding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import descriptor as D                # noqa: E402
from tools.mining import smooth_equivalence as SE         # noqa: E402


def test_cut_is_the_emission_owner_not_a_second_literal():
    assert SE.STRICT_CUT == D.NEAR_DUP_THRESHOLD == 0.974


def test_strict_cut_sits_inside_the_interleave_zone():
    lo, hi = SE.IDENTITY_INTERLEAVE
    assert lo < SE.STRICT_CUT < hi


def test_bands_are_disjoint_and_ordered():
    assert SE.band_of(0.99) == "near_dup"
    assert SE.band_of(SE.STRICT_CUT) == "near_dup"          # inclusive at the cut
    assert SE.band_of(SE.STRICT_CUT - 1e-9) == "interleave"
    assert SE.band_of(SE.IDENTITY_INTERLEAVE[0]) == "interleave"
    assert SE.band_of(SE.IDENTITY_INTERLEAVE[0] - 1e-9) == "distinct"
    assert SE.band_of(0.2) == "distinct"


def test_cos_to_smooth_is_rowwise_not_a_matrix():
    a = np.eye(3, 4, dtype=np.float32)
    b = np.eye(3, 4, dtype=np.float32)
    assert SE.cos_to_smooth(a, b).tolist() == [1.0, 1.0, 1.0]
    c = np.roll(b, 1, axis=1)
    assert SE.cos_to_smooth(a, c).tolist() == [0.0, 0.0, 0.0]


def test_per_group_shares_sum_to_one_within_every_group():
    groups = ["a", "a", "a", "b", "b"]
    cos = [0.99, 0.95, 0.5, 0.2, 0.98]
    t = SE.per_group_table(groups, cos)
    assert t["a"]["n"] == 3 and t["b"]["n"] == 2
    for v in t.values():
        s = v["share_near_dup"] + v["share_interleave"] + v["share_distinct"]
        assert pytest.approx(s, abs=1e-12) == 1.0
    assert t["a"]["share_near_dup"] == pytest.approx(1 / 3)


def test_unrelated_reference_never_pairs_a_row_with_itself():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ref = SE.unrelated_reference(v, seed=3, n_pairs=5000)
    # identical rows would put a spike at exactly 1.0; random unit rows in 8-D never do.
    assert ref["max"] < 0.999999
    assert ref["n_pairs"] == 5000


def test_unrelated_reference_is_defined_for_a_degenerate_population():
    assert SE.unrelated_reference(np.zeros((1, 4), dtype=np.float32))["n_pairs"] == 0


def test_yardstick_block_states_the_borrowed_calibration():
    y = SE.yardstick_block()
    assert y["strict_cut"] == SE.STRICT_CUT
    assert "grayscale" in y["caveat"].lower()
    assert "descriptor" in y["strict_cut_owner"]
