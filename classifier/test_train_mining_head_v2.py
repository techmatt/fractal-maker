"""Guards for the mining head v2 finetune trainer.

v2's whole claim is "v1's recipe, v1's weights, the fresh sheet, five named deviations". Each
half of that is a way to be silently wrong, and these are the halves:

  * the RECIPE must not drift. Every knob v2 says it inherited is held against v1's own
    checkpoint config — a differential against the surviving reference implementation
    (`verification_practice.md` §7) rather than a frozen literal, because v1's config ships
    inside the weight file this run initialises from.
  * the SPLIT must be READ, never re-derived. A trainer that re-splits would put eval rows in
    training and every number in the report would describe a head that had seen its own eval.
  * all 15 modes must be present. The July mode drops are the thing this finetune exists to
    undo, so a drop that quietly survived would make the run's stated purpose unmeasurable.
  * `load_rows` must fail closed — on an unlabeled row, on an out-of-range tier, and on a
    split side that is neither train nor eval (§2: a silently smaller n reads exactly like a
    complete corpus).
  * `mode_tiers` must derive rich/directional from the data, with a control in each direction
    — v1 hardcoded those lists against a corpus that no longer exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier import train_mining_head_v2 as T                # noqa: E402
from classifier.train_mining_head import BACKBONE, K, MRow      # noqa: E402
from tools.mining.mining_roster import MODES, TRAINER_DROPPED_V1  # noqa: E402


# =========================================================================== #
# The recipe, against v1's own checkpoint config.
# =========================================================================== #
@pytest.fixture(scope="module")
def v1_config():
    """v1's config, read out of the weight file v2 initialises from. Hard-fails on absence:
    without v1 there is no finetune, so a skip here would green-light a run that cannot be
    what it says it is."""
    ck = torch.load(T.INIT_FROM, map_location="cpu", weights_only=False)
    return ck["config"]


def test_every_inherited_knob_still_matches_v1(v1_config):
    defaults = vars(T.build_parser().parse_args([]))
    assert T.RECIPE_KNOBS, "the knob map is empty — this comparison would be vacuous"
    drift = {k: (defaults[k], v1_config[c]) for k, c in T.RECIPE_KNOBS.items()
             if defaults[k] != v1_config[c]}
    assert not drift, f"v2 default != v1 config for {drift} — an undeclared recipe deviation"


def test_the_backbone_and_tier_count_are_v1s_or_it_is_not_a_finetune(v1_config):
    """A different backbone or K makes `load_state_dict` a shape error rather than a
    finetune; the trainer raises on it and this pins the precondition it raises about."""
    assert v1_config["backbone"] == BACKBONE
    assert int(v1_config["num_classes"]) == K == 3


def test_the_declared_deviations_are_exactly_the_knobs_left_out_of_the_map():
    """The map and the deviation list must partition the recipe: a knob in neither is a
    change nobody declared, and a knob in both is a contradiction."""
    assert set(T.RECIPE_KNOBS) & {"selection", "init", "data", "modes"} == set()


# =========================================================================== #
# load_rows — the split is read, and it fails closed.
# =========================================================================== #
def _row(image_id, side, score, mode="tia", loc=None):
    return {"image_id": image_id,
            "render": {"fractal_type": "mandelbrot"},
            "provenance": {"split_side": side, "render_mode": mode,
                           "family": "mandelbrot", "location_key": loc or f"L{image_id}"},
            "label": {"score": score}}


def _batch(tmp_path, rows):
    (tmp_path / "crops").mkdir(exist_ok=True)
    for r in rows:
        (tmp_path / "crops" / f"{r['image_id']}.jpg").write_bytes(b"x")
    (tmp_path / "images.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def _full_roster(side_of):
    """One row per roster mode so `load_rows`'s completeness check is satisfied."""
    return [_row(f"m{i}", side_of(i), 1 + i % 3, mode=m) for i, m in enumerate(MODES)]


def test_rows_land_on_the_side_the_batch_stamped(tmp_path):
    rows = _full_roster(lambda i: "eval" if i % 2 else "train")
    tr, ev = T.load_rows(_batch(tmp_path, rows))
    assert [r.image_id for r in tr] == [r["image_id"] for r in rows
                                        if r["provenance"]["split_side"] == "train"]
    assert [r.image_id for r in ev] == [r["image_id"] for r in rows
                                        if r["provenance"]["split_side"] == "eval"]
    assert tr and ev                                   # non-vacuity: both sides populated


def test_an_unlabeled_row_raises_rather_than_shrinking_the_corpus(tmp_path):
    rows = _full_roster(lambda i: "train")
    rows[3]["label"]["score"] = None
    with pytest.raises(SystemExit, match="null"):
        T.load_rows(_batch(tmp_path, rows))


def test_a_location_on_both_sides_raises(tmp_path):
    """The split is READ, so the trainer cannot fix a broken one — it can only refuse. A
    location spanning both sides is eval leakage, and every read in the report would be
    computed on a head that had seen its own eval rows."""
    rows = _full_roster(lambda i: "eval" if i % 2 else "train")
    rows[0]["provenance"]["location_key"] = "SHARED"
    rows[1]["provenance"]["location_key"] = "SHARED"
    with pytest.raises(AssertionError, match="span train\\+eval"):
        T.load_rows(_batch(tmp_path, rows))


def test_an_unknown_split_side_raises_instead_of_being_dropped(tmp_path):
    rows = _full_roster(lambda i: "train")
    rows[2]["provenance"]["split_side"] = "holdout"
    with pytest.raises(ValueError, match="neither train nor eval"):
        T.load_rows(_batch(tmp_path, rows))


def test_a_tier_outside_1_to_K_raises(tmp_path):
    rows = _full_roster(lambda i: "train")
    rows[1]["label"]["score"] = 4
    with pytest.raises(ValueError, match="out of 1"):
        T.load_rows(_batch(tmp_path, rows))


def test_a_missing_roster_mode_raises(tmp_path):
    """The corpus must carry every mode or the finetune's stated purpose — the three modes
    v1 never saw — is unmeasurable for whichever one went missing."""
    rows = [r for r in _full_roster(lambda i: "train")
            if r["provenance"]["render_mode"] != "trap_circle"]
    with pytest.raises(AssertionError, match="roster modes absent"):
        T.load_rows(_batch(tmp_path, rows))


# =========================================================================== #
# mode_tiers — derived, with a control each way.
# =========================================================================== #
def test_rich_is_decided_by_this_corpus_eval_q3_count_not_a_hardcoded_list():
    ev = ([MRow(f"a{i}", 3, Path("."), f"L{i}", "tia", "mandelbrot", "mandelbrot")
           for i in range(T.RICH_MIN_EVAL_GOOD)]
          + [MRow(f"b{i}", 3, Path("."), f"M{i}", "stripe", "mandelbrot", "mandelbrot")
             for i in range(T.RICH_MIN_EVAL_GOOD - 1)])
    tiers = T.mode_tiers(ev)
    assert tiers["tia"] == "rich"                       # exactly at the floor
    assert tiers["stripe"] == "directional"             # one short of it
    assert tiers["exp_smoothing"] == "directional"      # absent entirely
    assert set(tiers) == set(MODES)                     # every mode gets a tier


# =========================================================================== #
# The committed batch — non-vacuity for the whole run.
# =========================================================================== #
def test_the_committed_batch_splits_538_train_422_eval_over_all_15_modes():
    manifest = json.loads((T.BATCH_DIR / "batch.json").read_text(encoding="utf-8"))
    tr, ev = T.load_rows()
    assert len(tr) == manifest["realized"]["rows_by_split"]["train"] == 538
    assert len(ev) == manifest["realized"]["rows_by_split"]["eval"] == 422
    assert {r.mode for r in tr} == {r.mode for r in ev} == set(MODES)
    assert len(MODES) == manifest["roster"]["n_modes"] == 15


def test_the_three_modes_v1s_trainer_dropped_are_in_v2s_TRAINING_rows():
    """DEVIATION 4, as a fact about the rows rather than a sentence in a docstring. If a
    future edit re-applies a trainer-side drop list, the finetune stops being about the
    three modes it was run for and this goes red."""
    tr, _ = T.load_rows()
    per_mode = {m: sum(1 for r in tr if r.mode == m) for m in TRAINER_DROPPED_V1}
    assert all(n > 0 for n in per_mode.values()), per_mode
    assert sum(per_mode.values()) > 100                 # ~38 train rows each, not a token few
