"""The label-store join is on COORDINATES, not image_id — so a sidecar shared across scale
batches cannot cross-contaminate.

image_id (`A_<idx>_<comp>_<palette>`) does not encode render scale, so the same id can name
different-scale crops in two batches that share one sidecar (`scale_2x2_labelset.json`).
These tests construct exactly that collision and assert the label reaches only the crop
whose render identity matches.

Run: uv run pytest tools/corpus/test_label_store_join.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import label_store as ls  # noqa: E402


def _render(cx, cy, fw, palette="viridis", composition="center"):
    return {"cx": cx, "cy": cy, "fw": fw, "palette": palette, "composition": composition,
            "maxiter": 8000, "width": 1280, "height": 720, "ss": 2}


def _write_batch(batches_dir, batch_id, rows):
    d = os.path.join(batches_dir, batch_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "images.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two batches sharing one sidecar. OWNER labels image_id X (fw=0.1) and Y. CONSUMER
    reuses image_id X at a DIFFERENT scale (fw=0.001) and adds Z at OWNER-Y's coordinates."""
    labels_dir = tmp_path / "labels"
    batches_dir = tmp_path / "batches"
    labels_dir.mkdir()
    batches_dir.mkdir()

    (labels_dir / "s.json").write_text(json.dumps({"X": 3, "Y": 1}), encoding="utf-8")

    owner_rows = [
        {"image_id": "X", "render": _render("0.5", "0.6", "0.1")},
        {"image_id": "Y", "render": _render("0.7", "0.8", "0.1")},
    ]
    consumer_rows = [
        {"image_id": "X", "render": _render("0.5", "0.6", "0.001")},  # SAME id, different scale
        {"image_id": "Z", "render": _render("0.7", "0.8", "0.1")},    # OWNER-Y's coords, new id
    ]
    _write_batch(str(batches_dir), "owner", owner_rows)
    _write_batch(str(batches_dir), "consumer", consumer_rows)

    monkeypatch.setattr(ls, "LABELS_DIR", str(labels_dir))
    monkeypatch.setattr(ls, "SIDECAR_LABELS", {"owner": "s.json", "consumer": "s.json"})
    monkeypatch.setattr(ls, "SIDECAR_OWNER", {"s.json": "owner"})
    return {"batches_dir": str(batches_dir), "owner": owner_rows, "consumer": consumer_rows}


def test_colliding_image_id_at_different_scale_does_not_contaminate(corpus):
    labels = ls.sidecar_for("consumer", corpus["batches_dir"])
    consumer = {r["image_id"]: r for r in corpus["consumer"]}
    # image_id X exists in the sidecar AND in the consumer batch, but at a different fw:
    # the coordinate join must refuse it. An image_id join would have wrongly returned 3.
    assert ls.resolve_score(consumer["X"], labels) is None


def test_matching_coordinates_transfer_across_a_different_image_id(corpus):
    labels = ls.sidecar_for("consumer", corpus["batches_dir"])
    consumer = {r["image_id"]: r for r in corpus["consumer"]}
    # Z has a different image_id but OWNER-Y's exact render identity → it inherits Y's label.
    assert ls.resolve_score(consumer["Z"], labels) == 1


def test_owner_batch_resolves_its_own_rows(corpus):
    labels = ls.sidecar_for("owner", corpus["batches_dir"])
    owner = {r["image_id"]: r for r in corpus["owner"]}
    assert ls.resolve_score(owner["X"], labels) == 3
    assert ls.resolve_score(owner["Y"], labels) == 1


def test_shared_sidecar_without_owner_is_a_loud_error(corpus, monkeypatch):
    monkeypatch.setattr(ls, "SIDECAR_OWNER", {})   # drop the explicit owner
    with pytest.raises(RuntimeError):
        ls.sidecar_for("consumer", corpus["batches_dir"])


def test_merged_label_takes_precedence_over_join(corpus):
    labels = ls.sidecar_for("consumer", corpus["batches_dir"])
    row = {"image_id": "X", "render": _render("0.5", "0.6", "0.001"),
           "label": {"score": 2}}
    assert ls.resolve_score(row, labels) == 2   # in-row score wins, never the join
