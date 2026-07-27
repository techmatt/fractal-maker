"""The amendment overlay: a revision resolves to the NEW value, the ORIGINAL stays readable.

Revisions to already-labeled rows go to a separate registered amendment stream, never modifying
the original label. `resolve_score` prefers the amendment when one exists (via `amendments_for`)
and falls back to the original otherwise — so the pre-revision label is always recoverable, and
reconstructing the original >=3 boundary is the one-liner `resolve_score(row, sidecar) >= 3`
(no amendments argument). These tests exercise a promotion (3 -> 4), a demotion (3 -> 2 that
crosses the >=3 boundary), amendment-wins-over-sidecar, and coordinate re-keying.

Run: uv run pytest tools/corpus/test_label_amendment.py -q
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
            "maxiter": 8000, "width": 1280, "height": 720, "ss": 4}


def _row(image_id, render, score=None):
    return {"image_id": image_id, "render": render, "label": {"score": score}}


def _write_batch(batches_dir, batch_id, rows):
    d = os.path.join(batches_dir, batch_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "images.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """One in-row-labeled batch `merged` (A=3, B=3) with an amendment promoting A 3->4 and
    demoting B 3->2, plus a sidecar-only batch `sc` (P labeled 3 via sidecar) whose amendment
    demotes P 3->1."""
    labels_dir = tmp_path / "labels"
    batches_dir = tmp_path / "batches"
    labels_dir.mkdir()
    batches_dir.mkdir()

    merged_rows = [
        _row("A", _render("0.5", "0.6", "0.1"), score=3),
        _row("B", _render("0.7", "0.8", "0.1"), score=3),
        _row("C", _render("0.9", "0.1", "0.1"), score=3),   # never amended -> stays 3
    ]
    _write_batch(str(batches_dir), "merged", merged_rows)
    # amendment file keyed by the SOURCE image_id, re-keyed onto coordinates via `merged`.
    (labels_dir / "amend_merged.json").write_text(
        json.dumps({"A": 4, "B": 2}), encoding="utf-8")

    sc_rows = [_row("P", _render("0.2", "0.3", "0.1"))]     # label lives ONLY in the sidecar
    _write_batch(str(batches_dir), "sc", sc_rows)
    (labels_dir / "sc.json").write_text(json.dumps({"P": 3}), encoding="utf-8")
    (labels_dir / "amend_sc.json").write_text(json.dumps({"P": 1}), encoding="utf-8")

    monkeypatch.setattr(ls, "LABELS_DIR", str(labels_dir))
    monkeypatch.setattr(ls, "SIDECAR_LABELS", {"sc": "sc.json"})
    monkeypatch.setattr(ls, "SIDECAR_OWNER", {})
    monkeypatch.setattr(ls, "AMENDMENT_LABELS",
                        {"merged": "amend_merged.json", "sc": "amend_sc.json"})
    monkeypatch.setattr(ls, "AMENDMENT_OWNER", {})
    return {"batches_dir": str(batches_dir), "merged": merged_rows, "sc": sc_rows}


def test_amended_row_resolves_to_the_new_value(corpus):
    bd = corpus["batches_dir"]
    amd = ls.amendments_for("merged", bd)
    rows = {r["image_id"]: r for r in corpus["merged"]}
    assert ls.resolve_score(rows["A"], None, amd) == 4   # promotion wins over in-row 3
    assert ls.resolve_score(rows["B"], None, amd) == 2   # demotion wins over in-row 3


def test_original_is_still_readable_without_amendments(corpus):
    """The pre-revision label is recoverable: resolve_score with NO amendments = original."""
    rows = {r["image_id"]: r for r in corpus["merged"]}
    assert ls.resolve_score(rows["A"], None) == 3
    assert ls.resolve_score(rows["B"], None) == 3
    # ...and the original file on disk is untouched (still {A:3, B:3, C:3} in-row).
    assert rows["A"]["label"]["score"] == 3
    assert rows["B"]["label"]["score"] == 3


def test_original_ge3_boundary_is_a_one_liner(corpus):
    """Reconstructing the ORIGINAL >=3 boundary ignores amendments; the REVISED boundary
    applies them. The demotion of B moves it out of the >=3 set."""
    bd = corpus["batches_dir"]
    amd = ls.amendments_for("merged", bd)
    rows = {r["image_id"]: r for r in corpus["merged"]}
    original_ge3 = {iid for iid, r in rows.items() if (ls.resolve_score(r, None) or 0) >= 3}
    revised_ge3 = {iid for iid, r in rows.items() if (ls.resolve_score(r, None, amd) or 0) >= 3}
    assert original_ge3 == {"A", "B", "C"}
    assert revised_ge3 == {"A", "C"}          # B (3->2) fell below; A (3->4) stayed


def test_amendment_wins_over_sidecar(corpus):
    """A revision overrides a label that lives only in a sidecar, too."""
    bd = corpus["batches_dir"]
    sidecar = ls.sidecar_for("sc", bd)
    amd = ls.amendments_for("sc", bd)
    row = corpus["sc"][0]
    assert ls.resolve_score(row, sidecar) == 3           # original (sidecar) still readable
    assert ls.resolve_score(row, sidecar, amd) == 1      # revision wins


def test_unregistered_batch_has_no_amendments(corpus):
    assert ls.amendments_for("merged_nope", corpus["batches_dir"]) is None


def test_unamended_row_falls_through_to_original(corpus):
    bd = corpus["batches_dir"]
    amd = ls.amendments_for("merged", bd)
    rows = {r["image_id"]: r for r in corpus["merged"]}
    assert ls.resolve_score(rows["C"], None, amd) == 3   # C is not in the amendment file
