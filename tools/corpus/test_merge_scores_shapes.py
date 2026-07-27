"""merge_scores.load_scores accepts BOTH export shapes — the legacy {id:int} and the new
combined {id:{score,revealed}} — with no migration. Both must yield the same {id:score} map.

Run: uv run pytest tools/corpus/test_merge_scores_shapes.py -q
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import merge_scores as ms  # noqa: E402


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_legacy_int_shape(tmp_path):
    p = _write(tmp_path, "scores.json", {"a": 1, "b": 3, "c": None})
    assert ms.load_scores(p) == {"a": 1, "b": 3, "c": None}


def test_combined_shape_extracts_score(tmp_path):
    p = _write(tmp_path, "labels.json",
               {"a": {"score": 1, "revealed": 0}, "b": {"score": 4, "revealed": 1}})
    # only the score is merged; the reveal flag is an audit sidecar, not a store field.
    assert ms.load_scores(p) == {"a": 1, "b": 4}


def test_combined_shape_null_score(tmp_path):
    p = _write(tmp_path, "labels.json", {"a": {"score": None, "revealed": 0}})
    assert ms.load_scores(p) == {"a": None}


def test_both_shapes_agree_on_scores(tmp_path):
    legacy = _write(tmp_path, "scores.json", {"a": 2, "b": 3})
    combined = _write(tmp_path, "labels.json",
                      {"a": {"score": 2, "revealed": 1}, "b": {"score": 3, "revealed": 0}})
    assert ms.load_scores(legacy) == ms.load_scores(combined) == {"a": 2, "b": 3}
