"""The build path must see the revision overlay: a promoted row surfaces as class 4.

`load_post_freeze` reduces labeled crops to appended (post-v6-freeze) manifest locations. It
resolves each location's label through `label_store` — and it MUST apply the amendment stream
(`amendments_for`), exactly as `corpus_reader.iter_labeled` and `query_sampler` do, so a
re-judged row reaches the manifest as its revised value. Before that fix `load_post_freeze`
called `resolve_score(row, sidecar)` with NO amendments, so:

  * a q3->q4 PROMOTION surfaced as 3 (the whole class-4 tier was invisible to the build), and
  * a q3->q2 DEMOTION surfaced as 3 (a row that left the >=3 set stayed "good").

`test_promotion_surfaces_as_class_4` is RED against the pre-fix build path (it resolved 3) and
green now. The frozen v6 prefix is NOT exercised here — these are appended (post-freeze) rows,
the only place a build resolves live labels; the prefix freeze is a separate byte-gate.

Run:  uv run python -m pytest tools/v7/test_build_manifest_amendments.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v7"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import build_manifest as bm  # noqa: E402
import label_store as ls  # noqa: E402

BATCH = "2026-07-26_amend_probe"


def _render(cx, cy, fw, palette="viridis", composition="center"):
    return {"cx": cx, "cy": cy, "fw": fw, "palette": palette, "composition": composition,
            "maxiter": 8000, "width": 1280, "height": 720, "ss": 4}


def _row(image_id, render, score):
    return {"image_id": image_id, "render": render, "label": {"score": score}}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """One appended batch: P labeled 3 (promoted 3->4), D labeled 3 (demoted 3->2),
    R labeled 3 (reaffirmed / never amended). Amendment file {P:4, D:2}."""
    labels_dir = tmp_path / "labels"
    batches_dir = tmp_path / "batches"
    labels_dir.mkdir()
    (batches_dir / BATCH).mkdir(parents=True)

    rows = [
        _row("P", _render("0.5", "0.6", "0.1"), 3),   # promotion 3 -> 4
        _row("D", _render("0.7", "0.8", "0.1"), 3),   # demotion  3 -> 2
        _row("R", _render("0.9", "0.1", "0.1"), 3),   # reaffirmed, not in the amendment file
    ]
    with (batches_dir / BATCH / "images.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (labels_dir / f"amend_{BATCH}.json").write_text(json.dumps({"P": 4, "D": 2}), encoding="utf-8")

    # Point both authorities at the tmp corpus; register only this batch's amendment.
    monkeypatch.setattr(ls, "LABELS_DIR", str(labels_dir))
    monkeypatch.setattr(ls, "BATCHES_DIR", str(batches_dir))
    monkeypatch.setattr(ls, "SIDECAR_LABELS", {})
    monkeypatch.setattr(ls, "SIDECAR_OWNER", {})
    monkeypatch.setattr(ls, "AMENDMENT_LABELS", {BATCH: f"amend_{BATCH}.json"})
    monkeypatch.setattr(ls, "AMENDMENT_OWNER", {})
    monkeypatch.setattr(bm, "BATCHES_GLOB", str(batches_dir / "*" / "images.jsonl"))
    return {"batches_dir": str(batches_dir)}


def _by_ident(locs):
    """cx -> loc, keyed on the distinct cx per row above."""
    return {l["cx"]: l for l in locs}


def test_promotion_surfaces_as_class_4(corpus):
    """RED before the fix: the pre-fix build path resolved no amendments, so P came back 3
    and the class-4 tier never entered the manifest."""
    post = bm.load_post_freeze(v6_ids=set())
    by = _by_ident(post)
    assert by["0.5"]["label"] == 4, "a q3->q4 promotion must surface as class 4 through the build"


def test_demotion_crosses_the_ge3_boundary(corpus):
    """A q3->q2 demotion leaves the >=3 set through the build path, not just in resolve_score."""
    post = bm.load_post_freeze(v6_ids=set())
    by = _by_ident(post)
    assert by["0.7"]["label"] == 2
    assert by["0.7"]["label"] < 3


def test_unamended_row_keeps_its_original(corpus):
    post = bm.load_post_freeze(v6_ids=set())
    by = _by_ident(post)
    assert by["0.9"]["label"] == 3   # R is not in the amendment file -> original stands


def test_original_is_recoverable_without_amendments(corpus):
    """The pre-revision label stays readable: resolving WITHOUT amendments reconstructs 3 for
    every row, so the fix adds the revision view without destroying the original one."""
    rows = [json.loads(l) for l in
            (Path(corpus["batches_dir"]) / BATCH / "images.jsonl").read_text().splitlines()]
    for r in rows:
        assert ls.resolve_score(r, None) == 3          # no amendments -> original
    amd = ls.amendments_for(BATCH)
    scores = {r["image_id"]: ls.resolve_score(r, None, amd) for r in rows}
    assert scores == {"P": 4, "D": 2, "R": 3}          # with amendments -> revised
