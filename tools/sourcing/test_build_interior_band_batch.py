#!/usr/bin/env python
"""Tests for the interior-band batch draw.

The batch's whole value rests on two properties that are easy to break silently and
impossible to see in the finished crops: (a) the ONLY thing separating the two arms is
interior fraction, and (b) the band edges are exactly the deployed mask's ceiling. So the
band predicate, the arm-comparability machinery (scale mix, eval interleave, same-atom
separation) and the manifest's shape are pinned here rather than eyeballed in a report.

  uv run pytest tools/sourcing/test_build_interior_band_batch.py -q
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(os.path.normpath(os.path.join(HERE, "..", "..")))
for _p in (str(ROOT), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_interior_band_batch as B      # noqa: E402
from tools.studies import q4_stage1_linear_fit as LF   # noqa: E402


# --------------------------------------------------------------------------- #
# the band predicate == the deployed mask's ceiling
# --------------------------------------------------------------------------- #
def test_the_control_band_edge_is_exactly_the_deployed_mask_ceiling():
    """The control arm is "everything the mask lets through on interior grounds" and the
    interior arm is "everything it does not". If these drift apart the batch stops being
    about the mask."""
    assert B.BAND_HI["control"] == LF.V2_INTERIOR
    assert B.BAND_LO["i10_20"] == LF.V2_INTERIOR
    assert B.band_of(LF.V2_INTERIOR - 1e-9) == "control"
    assert B.band_of(LF.V2_INTERIOR) == "i10_20"


def test_bands_tile_zero_to_half_and_stop_there():
    assert B.band_of(0.0) == "control"
    assert B.band_of(0.19999) == "i10_20"
    assert B.band_of(0.20) == "i20_35"
    assert B.band_of(0.349) == "i20_35"
    assert B.band_of(0.35) == "i35_50"
    assert B.band_of(0.4999) == "i35_50"
    assert B.band_of(0.50) is None            # >= 0.50 is out of scope, not backfilled
    assert B.band_of(0.99) is None


def test_clause_attribution_matches_the_deployed_predicate():
    """The recorded clauses must agree with `_v2_drop` row for row — they are the only
    record of WHY each drawn window is currently unscoreable."""
    for gi, gf, gs in ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.95, 0.0),
                       (0.0, 0.0, 0.4), (0.3, 0.9, 0.4), (0.09, 0.87, 0.29)):
        f = {"g_interior": gi, "g_flat": gf, "g_speckle": gs}
        assert bool(B.clauses_of(gi, gf, gs)) == LF._v2_drop(f)


# --------------------------------------------------------------------------- #
# arm comparability
# --------------------------------------------------------------------------- #
def test_scale_mix_matches_the_487_batch():
    """"Same scale and geometry as the 487 crops" has to be true of the DISTRIBUTION, not
    just the scale set, or scale becomes a second thing varying between old and new."""
    draw = ROOT / "data" / "minibrot_roster" / "batch_v1" / "draw.jsonl"
    if not draw.exists():
        pytest.skip("487 draw manifest absent")
    rows = [json.loads(l) for l in draw.read_text(encoding="utf-8").splitlines() if l.strip()]
    realized = Counter(round(r["box"][2], 2) for r in rows)
    n = sum(realized.values())
    assert set(B.SCALE_MIX) == set(LF.FIELD_SCALES)
    for s, p in B.SCALE_MIX.items():
        assert realized[round(s, 2)] / n == pytest.approx(p, abs=0.002)


def test_pick_scale_reproduces_the_mix_when_every_scale_is_available():
    rng = np.random.default_rng(0)
    got = Counter(B._pick_scale(rng, list(LF.FIELD_SCALES)) for _ in range(20000))
    for s, p in B.SCALE_MIX.items():
        assert got[s] / 20000 == pytest.approx(p, abs=0.02)


def test_pick_scale_falls_back_when_the_mix_has_no_mass_left():
    rng = np.random.default_rng(0)
    assert B._pick_scale(rng, [0.99]) == 0.99          # unknown scale -> uniform fallback


def test_eval_atoms_are_offered_every_fourth_slot_and_no_split_is_reassigned():
    """The split is inherited; only the ORDER changes. A 5-pick cell must see an eval atom,
    so both arms end up with the same eval share instead of drawing it by luck."""
    A = {f"t{i}": {"split": "train"} for i in range(30)}
    A.update({f"e{i}": {"split": "eval"} for i in range(10)})
    order = B._cell_atom_order(sorted(A), A, np.random.default_rng(0))
    assert sorted(order) == sorted(A)                          # nothing dropped/duplicated
    assert all(A[a]["split"] in ("train", "eval") for a in order)
    # every window of EVAL_EVERY consecutive slots, while eval atoms remain, holds one
    for start in range(0, 4 * B.EVAL_EVERY, B.EVAL_EVERY):
        w = order[start:start + B.EVAL_EVERY]
        assert sum(A[a]["split"] == "eval" for a in w) == 1, (start, w)
    assert sum(A[a]["split"] == "eval" for a in order[:5]) == 1


def test_same_atom_windows_must_clear_the_screens_own_separation():
    c = dict(box=[0.5, 0.5, 0.06, 0.034])
    assert B._clash(c, [c], B.MT.HT.SEP)                       # identical window clashes
    far = dict(box=[0.9, 0.9, 0.06, 0.034])
    assert not B._clash(far, [c], B.MT.HT.SEP)


# --------------------------------------------------------------------------- #
# the built manifest (skipped until `draw` has run)
# --------------------------------------------------------------------------- #
def _manifest():
    p = ROOT / B.DRAW_REL
    if not p.exists():
        pytest.skip("draw manifest not built (run `build_interior_band_batch draw`)")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_every_drawn_row_sits_in_the_band_it_was_drawn_for():
    for r in _manifest():
        assert B.band_of(r["g_interior"]) == r["band"], r["image_id"]
        assert B.BAND_ARM[r["band"]] == r["arm"]


def test_the_per_atom_cap_holds_and_no_two_windows_are_the_same_picture():
    rows = _manifest()
    per_atom = Counter(r["atom_id"] for r in rows)
    assert max(per_atom.values()) <= B.PER_ATOM_CAP
    from tools.studies.interior_bakeoff import _iou
    by_atom = {}
    for r in rows:
        by_atom.setdefault(r["atom_id"], []).append(r)
    for aid, rs in by_atom.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                assert _iou(rs[i]["box"], rs[j]["box"]) < 0.25, (aid, rs[i], rs[j])


def test_the_image_id_leaks_no_answer():
    """The labeler judges blind, but the crop's FILENAME still reaches the browser (it is
    the crops/<image_id>.jpg URL). The 487's ids encoded the screen's fate; these must
    encode nothing: a shuffle-assigned slot plus an opaque content hash."""
    import re
    rows = _manifest()
    for r in rows:
        iid = r["image_id"]
        assert re.fullmatch(r"ib\d{4}_[0-9a-f]{8}", iid), iid
        for leak in (r["arm"], r["band"], r["atom_id"], r["family"]):
            assert leak not in iid, (iid, leak)
    assert len({r["image_id"] for r in rows}) == len(rows)


def test_the_id_ordering_carries_no_band_structure():
    """Sorting by image_id is the natural order anything downstream will fall into, so the
    slot index must be shuffle-assigned, not draw-ordered — otherwise the ids block up by
    band and the sequence itself becomes a hint."""
    rows = sorted(_manifest(), key=lambda r: r["image_id"])
    bands = [r["band"] for r in rows]
    runs = 1 + sum(1 for a, b in zip(bands, bands[1:]) if a != b)
    k = len(set(bands))
    # draw order would give exactly k runs; a shuffle gives ~n*(1-1/k).
    assert runs > 0.5 * len(bands) * (1 - 1 / k), (runs, len(bands), k)


def test_no_batch_row_carries_a_label_or_a_selection_that_used_G():
    bdir = ROOT / "data" / "label_corpus" / "batches" / B.BATCH_ID
    if not (bdir / "batch.json").exists():
        pytest.skip("batch not built")
    bj = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))
    assert bj["sampling_metaparameters"]["g_used_for_selection"] is False
    assert bj["sampling_metaparameters"]["selection_predicate"].startswith(
        "screen-resolution g_interior band")
    rows = [json.loads(l) for l in (bdir / "images.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert rows and all(r["label"]["score"] is None for r in rows)
