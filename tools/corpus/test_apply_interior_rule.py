"""The interior auto-reject rule writes RULE labels and never touches a human one.

The property that matters is not "it labels things" — it is the two directions it must
refuse: an already-labeled row (the store's one-mutation invariant) and a row with no
recorded measure (absent is not low). Both are asserted with a fixture that would make a
careless implementation go green: the untouched rows are ALSO over the threshold, so a
rule that ignored `label.score` would relabel them.

Run: uv run pytest tools/corpus/test_apply_interior_rule.py -q
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import apply_interior_rule as air  # noqa: E402


def _row(iid, ifrac, score=None, labeler=None):
    return {"image_id": iid, "render": {"fw": "1e-3"},
            "provenance": ({"interior_fraction": ifrac} if ifrac is not None else {}),
            "label": {"score": score, "labeler": labeler, "labeled_at": None}}


def _batch(tmp_path, rows, with_blind=True):
    bdir = tmp_path / "b1"
    bdir.mkdir()
    for name in (["images.jsonl", "blind.jsonl"] if with_blind else ["images.jsonl"]):
        (bdir / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(bdir)


FIXTURE = [
    _row("a", 0.42),                      # fires
    _row("b", 0.31),                      # fires (just over)
    _row("c", 0.30),                      # NOT: strict >
    _row("d", 0.29),                      # NOT: under
    _row("e", None),                      # NOT: no measure recorded
    _row("f", 0.55, score=3, labeler="matt"),   # NOT: human label stands, though over
    _row("g", 0.99, score=1, labeler="matt"),   # NOT: agreeing human label is still human
]


def test_plan_is_over_threshold_and_unlabeled_only():
    assert [i for i, _ in air.plan(FIXTURE)] == ["a", "b"]


def test_threshold_is_strict():
    assert not air.fires(_row("x", air.THRESHOLD))
    assert air.fires(_row("x", air.THRESHOLD + 1e-9))


def test_absent_measure_never_fires():
    assert not air.fires(_row("x", None))


def test_apply_writes_rule_provenance_and_leaves_human_labels_byte_identical(tmp_path):
    bdir = _batch(tmp_path, FIXTURE)
    before = {r["image_id"]: r["label"] for r in
              [json.loads(l) for l in open(os.path.join(bdir, "images.jsonl"))]}
    rep = air.apply_to_batch(bdir, date="2026-08-01", write=True)
    assert (rep["auto_labeled"], rep["human_labeled"], rep["remaining"]) == (2, 2, 3)

    rows = {r["image_id"]: r for r in
            [json.loads(l) for l in open(os.path.join(bdir, "images.jsonl"))]}
    for iid in ("a", "b"):
        assert rows[iid]["label"] == {"score": 1, "labeler": air.LABELER,
                                      "labeled_at": "2026-08-01"}
        assert rows[iid]["label"]["labeler"] != "matt"
    for iid in ("c", "d", "e", "f", "g"):
        assert rows[iid]["label"] == before[iid], f"{iid} must be untouched"


def test_served_manifest_is_seeded_so_the_rig_skips_them(tmp_path):
    bdir = _batch(tmp_path, FIXTURE)
    air.apply_to_batch(bdir, date="2026-08-01", write=True)
    blind = {r["image_id"]: r["label"]
             for r in [json.loads(l) for l in open(os.path.join(bdir, "blind.jsonl"))]}
    # keyed on the rule tag, not on `score == 1`: a human 1 is also a 1 (row `g`).
    assert [i for i, v in blind.items() if v["labeler"] == air.LABELER] == ["a", "b"]
    assert all(blind[i]["score"] == 1 for i in ("a", "b"))


def test_record_names_the_rule_and_exactly_the_rows_it_fired_on(tmp_path):
    bdir = _batch(tmp_path, FIXTURE)
    air.apply_to_batch(bdir, date="2026-08-01", write=True)
    rec = json.load(open(os.path.join(bdir, air.RECORD_NAME), encoding="utf-8"))
    assert set(rec["labels"]) == {"a", "b"}
    assert rec["threshold"] == air.THRESHOLD and rec["comparison"] == "strict >"
    assert rec["measure"] == "provenance.interior_fraction" and rec["labeler"] == air.LABELER


def test_dry_run_writes_nothing(tmp_path):
    bdir = _batch(tmp_path, FIXTURE)
    raw = open(os.path.join(bdir, "images.jsonl"), encoding="utf-8").read()
    rep = air.apply_to_batch(bdir, date="2026-08-01", write=False)
    assert rep["auto_labeled"] == 2
    assert open(os.path.join(bdir, "images.jsonl"), encoding="utf-8").read() == raw
    assert not os.path.exists(os.path.join(bdir, air.RECORD_NAME))


def test_rerun_is_idempotent(tmp_path):
    bdir = _batch(tmp_path, FIXTURE)
    air.apply_to_batch(bdir, date="2026-08-01", write=True)
    after = open(os.path.join(bdir, "images.jsonl"), encoding="utf-8").read()
    rep = air.apply_to_batch(bdir, date="2026-08-02", write=True)
    assert rep["auto_labeled"] == 0
    assert open(os.path.join(bdir, "images.jsonl"), encoding="utf-8").read() == after
