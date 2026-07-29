#!/usr/bin/env python
"""Tests for the `G_cf` arm draw.

The batch's value rests on properties that are invisible in the finished crops: the arms
must differ in interior mass and NOTHING else about the objective, the pairing must be exact
within-atom, and the arm must not be reconstructable from anything the browser fetches. The
`verify` stage checks the built manifest; this pins the predicates and the batch's
classification so they cannot drift underneath it.

  uv run pytest tools/sourcing/test_build_gcf_arm_batch.py -q
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(os.path.normpath(os.path.join(HERE, "..", "..")))
for _p in (str(ROOT), HERE, str(ROOT / "tools" / "corpus"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_gcf_arm_batch as B                          # noqa: E402
from tools.studies import q4_stage1_linear_fit as LF     # noqa: E402


# --------------------------------------------------------------------------- #
# the arm predicate IS the deployed ceiling
# --------------------------------------------------------------------------- #
def test_the_arm_boundary_is_exactly_the_deployed_mask_ceiling():
    """HIGH is "what the mask rejects on interior grounds", LOW is "what it lets through".
    If the boundary drifts off the deployed clause the batch stops being about the mask."""
    assert B.INTERIOR_CUT == LF.V2_INTERIOR
    assert B.arm_of(LF.V2_INTERIOR) == B.ARM_HIGH
    assert B.arm_of(LF.V2_INTERIOR - 1e-12) == B.ARM_LOW
    assert B.arm_of(0.0) == B.ARM_LOW
    assert B.arm_of(1.0) == B.ARM_HIGH


def test_the_two_arms_partition_every_interior_value():
    for gi in (0.0, 0.05, 0.0999, 0.1, 0.2, 0.49, 0.5, 0.99, 1.0):
        assert B.arm_of(gi) in (B.ARM_HIGH, B.ARM_LOW)


# --------------------------------------------------------------------------- #
# pairing: argmax G_cf per arm, with separation
# --------------------------------------------------------------------------- #
def _rec(by_scale):
    """by_scale: {scale: [candidate dicts]} -> a candidate-cache record."""
    return {"atom_id": "a", "cands": {f"x|{s}": c for s, c in by_scale.items()}}


def test_each_arm_takes_its_own_argmax_G_cf():
    hi, lo, how = B._pair_one_atom(_rec({0.06: [
        dict(box=[0.2, 0.2, 0.06, 0.034], gi=0.30, gflat=0.0, gspeck=0.0, G_cf=-9.0),
        dict(box=[0.8, 0.2, 0.06, 0.034], gi=0.30, gflat=0.0, gspeck=0.0, G_cf=-3.0),
        dict(box=[0.2, 0.8, 0.06, 0.034], gi=0.01, gflat=0.0, gspeck=0.0, G_cf=-1.0),
        dict(box=[0.8, 0.8, 0.06, 0.034], gi=0.01, gflat=0.0, gspeck=0.0, G_cf=-5.0),
    ]}))
    assert how == "scale0.06"
    assert hi["G_cf"] == -3.0 and hi["gi"] == 0.30      # best G_cf among interior >= 0.10
    assert lo["G_cf"] == -1.0 and lo["gi"] == 0.01      # best G_cf among interior <  0.10


def test_a_pair_always_shares_one_window_scale():
    """The confound this rule exists to remove: an unconstrained argmax would take the
    high-interior window at 0.14 and the control at 0.06, so the arms would differ in window
    SIZE as well as interior mass — and size is plainly visible to the labeler."""
    rec = _rec({
        0.06: [dict(box=[0.2, 0.2, 0.06, 0.034], gi=0.30, gflat=0, gspeck=0, G_cf=-9.0),
               dict(box=[0.8, 0.8, 0.06, 0.034], gi=0.01, gflat=0, gspeck=0, G_cf=+5.0)],
        0.14: [dict(box=[0.3, 0.3, 0.14, 0.079], gi=0.30, gflat=0, gspeck=0, G_cf=-1.0),
               dict(box=[0.7, 0.7, 0.14, 0.079], gi=0.01, gflat=0, gspeck=0, G_cf=-8.0)],
    })
    hi, lo, how = B._pair_one_atom(rec)
    assert hi["scale"] == lo["scale"] == 0.14           # HIGH's best G_cf picks the scale
    assert how == "scale0.14"
    assert lo["G_cf"] == -8.0                           # NOT the global-argmax +5.0 at 0.06


def test_a_scale_without_both_arms_is_skipped():
    rec = _rec({
        0.06: [dict(box=[0.2, 0.2, 0.06, 0.034], gi=0.30, gflat=0, gspeck=0, G_cf=+9.0)],
        0.14: [dict(box=[0.3, 0.3, 0.14, 0.079], gi=0.30, gflat=0, gspeck=0, G_cf=-1.0),
               dict(box=[0.7, 0.7, 0.14, 0.079], gi=0.01, gflat=0, gspeck=0, G_cf=-8.0)],
    })
    hi, lo, how = B._pair_one_atom(rec)
    assert hi["scale"] == 0.14 and lo["scale"] == 0.14  # 0.06 has no LOW window at all


def test_the_low_pick_steps_down_to_clear_separation_and_says_so():
    """A LOW window on top of the HIGH one would make the pair two views of one picture.
    The next-best separated window is taken instead, and the fallback is RECORDED."""
    same = [0.5, 0.5, 0.06, 0.034]
    hi, lo, how = B._pair_one_atom(_rec({0.06: [
        dict(box=same, gi=0.30, gflat=0.0, gspeck=0.0, G_cf=-2.0),
        dict(box=same, gi=0.01, gflat=0.0, gspeck=0.0, G_cf=-1.0),          # top, clashes
        dict(box=[0.9, 0.1, 0.06, 0.034], gi=0.01, gflat=0.0, gspeck=0.0, G_cf=-4.0),
    ]}))
    assert how == "scale0.06+sep(rank1)"
    assert lo["G_cf"] == -4.0


def test_an_atom_with_an_empty_arm_is_dropped_not_backfilled():
    """A pair is 1+1 or nothing — an atom with no high-interior window must NOT contribute a
    lone LOW crop, or the contrast stops being within-atom."""
    hi, lo, how = B._pair_one_atom(_rec({0.06: [
        dict(box=[0.5, 0.5, 0.06, 0.034], gi=0.01, gflat=0.0, gspeck=0.0, G_cf=-1.0)]}))
    assert (hi, lo) == (None, None) and how == "no_scale_with_both_arms"


def test_an_atom_whose_low_arm_cannot_separate_is_dropped():
    same = [0.5, 0.5, 0.06, 0.034]
    hi, lo, how = B._pair_one_atom(_rec({0.06: [
        dict(box=same, gi=0.30, gflat=0.0, gspeck=0.0, G_cf=-2.0),
        dict(box=same, gi=0.01, gflat=0.0, gspeck=0.0, G_cf=-1.0)]}))
    assert (hi, lo) == (None, None) and how == "no_scale_with_both_arms"


# --------------------------------------------------------------------------- #
# the built manifest (skipped until `draw` has run)
# --------------------------------------------------------------------------- #
def _manifest():
    p = ROOT / B.DRAW_REL
    if not p.exists():
        pytest.skip("draw manifest not built (run `build_gcf_arm_batch draw`)")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_every_row_sits_in_the_arm_it_was_drawn_for():
    for r in _manifest():
        assert B.arm_of(r["interior_frac"]) == r["arm"], r["image_id"]


def test_every_atom_is_an_exact_one_plus_one_pair():
    rows = _manifest()
    per = {}
    for r in rows:
        per.setdefault(r["atom_id"], Counter())[r["arm"]] += 1
    for aid, c in per.items():
        assert c[B.ARM_HIGH] == 1 and c[B.ARM_LOW] == 1 and sum(c.values()) == 2, (aid, c)
    assert len(per) * 2 == len(rows)


def test_every_pair_shares_one_window_scale_in_the_built_manifest():
    per = {}
    for r in _manifest():
        per.setdefault(r["atom_id"], set()).add(r["scale"])
    assert all(len(s) == 1 for s in per.values()), \
        [(a, sorted(s)) for a, s in per.items() if len(s) > 1][:3]


def test_the_arms_realized_scale_mixes_are_identical():
    rows = _manifest()
    hi = Counter(r["scale"] for r in rows if r["arm"] == B.ARM_HIGH)
    lo = Counter(r["scale"] for r in rows if r["arm"] == B.ARM_LOW)
    assert hi == lo, (hi, lo)


def test_the_batch_is_train_side_with_the_split_inherited():
    assert all(r["split"] == "train" for r in _manifest())


def test_the_image_id_leaks_no_answer():
    """The labeler judges blind, but crops/<image_id>.jpg is a URL the browser fetches."""
    import re
    rows = _manifest()
    for r in rows:
        iid = r["image_id"]
        assert re.fullmatch(r"gc\d{4}_[0-9a-f]{8}", iid), iid
        for leak in (r["arm"], r["atom_id"], r["family"]):
            assert leak not in iid, (iid, leak)
    assert len({r["image_id"] for r in rows}) == len(rows)


def test_the_served_manifest_carries_no_arm_at_all():
    """Blinding as in the revisit chunks: the browser is served blind.jsonl, and the leak
    keys are ABSENT — not present-and-null — so the arm is not in the fetched bytes under
    any reveal state."""
    import corpus_common as cc
    bdir = Path(cc.batch_dir(B.BATCH_ID))
    if not (bdir / "blind.jsonl").exists():
        pytest.skip("batch not built")
    served = (bdir / "blind.jsonl").read_text(encoding="utf-8")
    for k in ("selection_role", "stratum", "interior_frac", "focus_score", "decoded_class",
              "descend_mode", "family"):
        assert f'"{k}"' not in served, k
    assert B.ARM_HIGH not in served and B.ARM_LOW not in served
    bj = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))
    assert bj["served_manifest"] == "blind.jsonl"
    assert bj["queued_for_labeling"] is False


def test_the_batch_classifies_biased_train_through_the_fail_closed_default():
    """A designed contrast is train-side and biased. It must reach that classification with
    NO registration-list edit — that is what the fail-closed default is for."""
    from tools.v7 import build_manifest as bm
    split, biased, source = bm.assign_split({"batch": B.BATCH_ID, "ft": "mandelbrot"})
    assert (split, biased, source) == ("train", True, "unregistered")
    assert B.BATCH_ID not in bm.CENSUS_BATCHES
    assert B.BATCH_ID not in bm.BAND_BATCHES
    assert B.BATCH_ID not in bm.UNBIASED_TRAIN_BATCHES
    assert B.BATCH_ID != bm.BLINDSPOT_BATCH


def test_nothing_in_the_batch_is_labeled():
    import corpus_common as cc
    bdir = Path(cc.batch_dir(B.BATCH_ID))
    if not (bdir / "images.jsonl").exists():
        pytest.skip("batch not built")
    rows = [json.loads(l) for l in (bdir / "images.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert rows and all(r["label"]["score"] is None for r in rows)
