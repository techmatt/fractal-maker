#!/usr/bin/env python
"""Tests for the interior/scroll feature bake-off.

Two jobs: (a) pin the candidate features' semantics on synthetic fields where the right
answer is known by construction — a disc IS blobbier than a filament of equal area, a
radial field DOES lose orientation coherence as the tensor window grows; (b) assert the
durable feature table still joins to the draw manifest and still reproduces the deployed
G exactly, so a silent drift in either upstream artifact fails here rather than in a
finding.

  uv run pytest tools/studies/test_interior_bakeoff.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for _p in (_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.studies import interior_bakeoff as IB      # noqa: E402

ROOT = Path(_ROOT)


# --------------------------------------------------------------------------- #
# synthetic field helpers: NaN marks in-set, finite values are escape times.
# --------------------------------------------------------------------------- #
def _blank(h=180, w=320):
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx + yy).astype(np.float64)          # a smooth, escaping background


def _disc(field, cy, cx, r):
    yy, xx = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    field[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = np.nan
    return field


# --------------------------------------------------------------------------- #
def test_disc_is_blobbier_than_a_filament_of_equal_area():
    """The dendrite-vs-body discriminators must order a disc above a thin bar."""
    r = 20
    area = int(np.pi * r * r)                              # ~1256 px
    H, W = 240, 700                                        # wide enough that the bar fits
    disc = _disc(_blank(H, W), 120, 350, r)
    fil = _blank(H, W)
    bar_h = 4
    fil[118:118 + bar_h, 20:20 + area // bar_h] = np.nan   # same area, 4 px thick

    fd, ff = IB.crop_features(disc), IB.crop_features(fil)
    assert fd["int_frac"] == pytest.approx(ff["int_frac"], rel=0.02)   # equal ink
    assert fd["int_perim_area"] < ff["int_perim_area"]                 # body: less edge/area
    assert fd["int_compactness"] > ff["int_compactness"]               # body: rounder
    assert fd["int_max_inradius"] > 3 * ff["int_max_inradius"]         # body: thicker


def test_no_interior_yields_nan_not_zero():
    """A crop with no in-set pixels has no perimeter-to-area ratio; coding it 0 would
    fake a maximally-blobby reading and silently bias the board."""
    f = IB.crop_features(_blank())
    assert f["int_frac"] == 0.0
    assert f["int_n_comp_a4"] == 0
    assert np.isnan(f["int_perim_area"])
    assert np.isnan(f["int_compactness"])
    assert f["int_max_inradius"] == 0.0


def test_component_counts_separate_the_three_area_scales():
    fld = _blank(360, 640)
    A = 360 * 640                                          # thresholds: 23 / 230 / 2304 px
    for r, cx in ((2, 30), (3, 90), (12, 250), (48, 480)):  # ~12, ~28, ~452, ~7238 px
        _disc(fld, 180, cx, r)
    f = IB.crop_features(fld)
    assert f["int_n_comp_a4"] == 3                          # the 12 px speck is below 1e-4*A
    assert f["int_n_comp_a3"] == 2
    assert f["int_n_comp_a2"] == 1
    assert f["int_largest_frac"] == pytest.approx(np.pi * 48 * 48 / A, rel=0.05)


def test_coherence_drops_across_scale_only_when_orientation_turns():
    """A straight ramp is perfectly oriented at every scale (no drop); a radial field
    turns, so its coherence falls as the structure-tensor window grows. That difference
    is the whole content of coh_scale_drop as a scroll measure."""
    h = 64                                       # tight rings: sigma=8 spans a real arc
    yy, xx = np.mgrid[0:h, 0:h]
    ramp = xx * 1.0
    radial = np.hypot(yy - h / 2, xx - h / 2) * 1.0
    fr, fd = IB.crop_features(ramp), IB.crop_features(radial)
    assert fr["coh_s3"] == pytest.approx(1.0, abs=1e-3)
    assert fr["coh_scale_drop"] == pytest.approx(0.0, abs=1e-3)
    assert fd["coh_s8"] < fd["coh_s3"]
    assert fd["coh_scale_drop"] > fr["coh_scale_drop"] + 0.05
    # and the drop is scale-relative: the same rings spread over a 4x larger frame turn
    # more slowly per pixel, so they read as straighter.
    wide = np.hypot(*np.mgrid[0:4 * h, 0:4 * h] - 2.0 * h) * 1.0
    assert IB.crop_features(wide)["coh_scale_drop"] < fd["coh_scale_drop"]


def test_iou_of_normalized_boxes():
    b = [0.5, 0.5, 0.2, 0.2]
    assert IB._iou(b, b) == pytest.approx(1.0)
    assert IB._iou(b, [0.9, 0.9, 0.2, 0.2]) == 0.0
    # half-overlap in u, full in v -> inter = 0.1*0.2, union = 2*0.04 - 0.02
    assert IB._iou(b, [0.6, 0.5, 0.2, 0.2]) == pytest.approx(0.02 / 0.06)


def test_partial_spearman_removes_a_common_cause():
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    x = z + 0.3 * rng.normal(size=400)
    y = z + 0.3 * rng.normal(size=400)
    raw, _ = IB.spearman(x, y)
    par, _ = IB.partial_spearman(x, y, z)
    assert raw > 0.8
    assert abs(par) < 0.25


def test_atom_bootstrap_partial_spearman_brackets_the_point_estimate():
    """The clustered CI must contain the point estimate, and must be WIDER when the same
    rows are clustered into few atoms than when every row is its own atom — that widening
    IS the correction (3 windows off one atom are not 3 independent observations)."""
    rng = np.random.default_rng(1)
    n, k = 300, 20
    atoms = np.repeat(np.arange(k), n // k)
    a = np.repeat(rng.normal(size=k), n // k)          # atom-level effect the control MISSES
    z = rng.normal(size=n)                             # an unrelated control
    x = a + 0.5 * rng.normal(size=n)
    y = a + 0.5 * rng.normal(size=n)
    point, _ = IB.partial_spearman(x, y, z)
    lo, hi, reps = IB.atom_bootstrap_partial_spearman(x, y, z, atoms, reps=400)
    assert reps > 300 and lo <= point <= hi
    lo2, hi2, _ = IB.atom_bootstrap_partial_spearman(
        x, y, z, np.arange(n), reps=400)               # every row its own cluster
    # The residual correlation here is entirely atom-level, so pretending the 300 rows are
    # independent understates the interval by a wide margin. That gap is the correction.
    assert (hi - lo) > 2 * (hi2 - lo2)


def test_auc_matches_a_hand_computed_case():
    a, _, _ = IB.auc([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    assert a == pytest.approx(1.0)
    a, _, _ = IB.auc([4.0, 3.0, 2.0, 1.0], [0, 0, 1, 1])
    assert a == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The durable table: joins + G parity against the deployed screen.
# --------------------------------------------------------------------------- #
def _table():
    p = ROOT / IB.TABLE_REL
    if not p.exists():
        pytest.skip("feature table not built (run `interior_bakeoff features`)")
    return IB._read_jsonl(p)


def test_feature_table_joins_the_draw_manifest_and_carries_every_field():
    rows = _table()
    draw = {d["image_id"]: d for d in IB._read_jsonl(IB.DRAW)}
    assert len(rows) == len(draw) == 487
    for r in rows:
        d = draw[r["image_id"]]                     # KeyError here = a broken join
        assert r["label"] in (1, 2, 3, 4)
        assert r["split"] in ("train", "eval")
        assert (r["degree"], r["period"], r["fate"]) == (d["degree"], d["period"], d["fate"])
        assert r["crop"] is not None and set(IB.CROP_KEYS) <= set(r["crop"])
        assert r["screen"] is not None and "g_interior" in r["screen"]


def test_recomputed_G_reproduces_the_deployed_G():
    """The screen-resolution re-featurization must land on the SAME window the deployed
    screen scored — otherwise every Part-C statement about G's components is about a
    different window than the one that was accepted or rejected."""
    rows = [r for r in _table() if r["G"] is not None and r["G_recomputed"] is not None]
    assert len(rows) == 462                          # the 25 OOD-masked rows carry no G
    d = np.array([r["G_recomputed"] - r["G"] for r in rows])
    assert np.abs(d).max() < 1e-4


def test_the_two_class_fours_are_one_atom_but_not_one_window():
    """Guards the anecdote caveat in both directions: they are NOT independent examples
    (same atom), and they are NOT the same picture (disjoint windows)."""
    c4 = [r for r in _table() if r["label"] == 4]
    assert len(c4) == 2
    assert c4[0]["atom"] == c4[1]["atom"]
    assert IB._iou(c4[0]["box"], c4[1]["box"]) == 0.0
