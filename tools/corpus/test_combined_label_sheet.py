"""The combined label sheet: it must stay a PRESENTATION alias, and its export must route.

Two families of test here, and they fail for different reasons:

  * the TRIPWIRES over the built sheet in `data/label_corpus/batches/` — a sheet directory that
    grew an `images.jsonl` would be unioned by every corpus consumer and double-count 870 labels
    that already live in three registered batches; a served manifest that regained a leak key
    would un-blind a sitting that is about to be labeled. These read the real bytes.
  * the UNIT tests over `ordered_union` and the routed merge, on synthetic corpora — the ±1
    prefix bound and the null->value refusal are properties of the code, not of this one sheet.

Run: uv run pytest tools/corpus/test_combined_label_sheet.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.normpath(os.path.join(HERE, "..", "..")),
           os.path.normpath(os.path.join(HERE, "..", "..", "tools"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_combined_label_sheet as B  # noqa: E402
import corpus_common as cc  # noqa: E402
import merge_scores as ms  # noqa: E402

SD = B.sheet_dir()
built = pytest.mark.skipif(not (SD / B.SHEET_MANIFEST).exists(),
                           reason="combined sheet not built in this checkout")


# --------------------------------------------------------------------------- #
# tripwires over the built sheet
# --------------------------------------------------------------------------- #
@built
def test_sheet_dir_is_not_a_batch():
    """The consequence, not the proxy: every consumer discovers batches by globbing
    `*/images.jsonl`, so the sheet must not appear in that glob."""
    import glob
    found = glob.glob(os.path.join(cc.BATCHES_DIR, "*", "images.jsonl"))
    assert not (SD / "images.jsonl").exists()
    assert str(SD / "images.jsonl") not in found
    assert B.SHEET_ID not in {os.path.basename(os.path.dirname(p)) for p in found}


@built
def test_served_manifest_carries_no_leak_key_and_no_source_identity():
    text = (SD / B.SHEET_MANIFEST).read_text(encoding="utf-8")
    for k in B.SHEET_LEAK_KEYS:
        assert f'"{k}"' not in text, f"leak key {k} reached the served manifest"
    for b in B.SOURCES:
        assert b not in text, f"source batch id {b} reached the served manifest"


@built
def test_served_rows_are_id_render_label_only():
    rows = cc.read_jsonl(str(SD / B.SHEET_MANIFEST))
    assert rows and all(set(r) == {"image_id", "render", "label"} for r in rows)
    assert all(r["label"]["score"] is None for r in rows)
    assert all(set(cc.RENDER_KEYS) <= set(r["render"]) for r in rows)


@built
def test_route_is_a_bijection_onto_the_source_rows():
    rows = cc.read_jsonl(str(SD / B.SHEET_MANIFEST))
    route = ms.load_route(str(SD / B.ROUTE_FILE))
    assert set(route) == {r["image_id"] for r in rows}
    targets = list(route.values())
    assert len(set(targets)) == len(targets), "two opaque ids point at one source row"
    for b in B.SOURCES:
        src = {r["image_id"] for r in
               cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl"))}
        assert {i for bb, i in targets if bb == b} == src


@built
def test_batch_json_pins_the_file_order():
    """The sheet's file order IS the designed stratification; the page must not reshuffle it."""
    bj = json.loads((SD / "batch.json").read_text(encoding="utf-8"))
    assert bj["presentation_order"] == "file"
    assert bj["served_manifest"] == B.SHEET_MANIFEST and bj["route_map"] == B.ROUTE_FILE
    assert bj.get("presentation_only") is True


def test_label_page_honours_the_file_order():
    """The pin above is inert unless corpus_label.html actually implements order=file."""
    page = open(os.path.join(HERE, "..", "viz", "corpus_label.html"), encoding="utf-8").read()
    assert "ORDER_MODE==='file'" in page
    assert "presentation_order" in page


# --------------------------------------------------------------------------- #
# ordered_union — the ±1 property, on synthetic cell mixes
# --------------------------------------------------------------------------- #
def _rows(batch, family, n):
    return [(batch, {"render": {"fractal_type": family}, "image_id": f"{batch}_{family}_{i}"})
            for i in range(n)]


@pytest.mark.parametrize("mix", [
    [("a", "x", 290), ("b", "y", 290), ("c", "z", 290)],          # three equal blocks
    [("a", "x", 500), ("b", "y", 5), ("c", "z", 1)],              # one dominant cell
    [("a", "x", 43), ("a", "y", 36), ("b", "x", 290),             # the real shape
     ("c", "y", 48), ("c", "z", 98)],
    [("a", "x", 1), ("b", "y", 1)],                               # degenerate
])
def test_every_cell_stays_within_one_of_its_share_in_every_prefix(mix):
    pairs = [p for b, f, n in mix for p in _rows(b, f, n)]
    order = B.ordered_union(pairs)
    assert len(order) == len(pairs)
    cells = [c for c, _, _ in order]
    from collections import Counter
    n_c, N, seen = Counter(cells), len(cells), Counter()
    for L, c in enumerate(cells, start=1):
        seen[c] += 1
        for cell, n in n_c.items():
            assert abs(seen[cell] - L * n / N) <= 1.0, f"cell {cell} off share at prefix {L}"


def test_ordered_union_is_a_permutation_and_is_reproducible():
    pairs = [p for b, f, n in [("a", "x", 40), ("b", "y", 25), ("b", "x", 7)]
             for p in _rows(b, f, n)]
    one = [r["image_id"] for _, _, r in B.ordered_union(pairs)]
    two = [r["image_id"] for _, _, r in B.ordered_union(pairs)]
    assert one == two, "the seeded order must be reproducible"
    assert sorted(one) == sorted(r["image_id"] for _, r in pairs)


# --------------------------------------------------------------------------- #
# the routed merge
# --------------------------------------------------------------------------- #
def _sandbox(tmp_path, batches):
    render = cc.render_block(cx=0, cy=0, fw=1, maxiter=100, palette="default",
                             composition="center", width=8, height=8, ss=1,
                             filter="lanczos3", interior_mode="black")
    root = tmp_path / "batches"
    for b, ids in batches.items():
        (root / b).mkdir(parents=True)
        cc.write_jsonl([cc.make_row(i, dict(render), cc.provenance_block("t", b),
                                    cc.label_block()) for i in ids],
                       str(root / b / "images.jsonl"))
    return str(root)


def _read(root, b):
    return {r["image_id"]: r["label"]["score"]
            for r in cc.read_jsonl(os.path.join(root, b, "images.jsonl"))}


def test_routed_merge_places_each_row_in_its_own_batch(tmp_path):
    root = _sandbox(tmp_path, {"b1": ["p", "q"], "b2": ["r"]})
    route = {"o1": ("b1", "p"), "o2": ("b1", "q"), "o3": ("b2", "r")}
    scores = {"o1": 1, "o2": 4, "o3": 3}
    per = {}
    for oid, s in scores.items():
        per.setdefault(route[oid][0], {})[route[oid][1]] = s
    stats = {b: ms.merge_batch(b, root, per[b], labeler="t", labeled_at="d",
                               max_score=4, apply=True) for b in per}
    assert stats["b1"]["filled"] == 2 and stats["b2"]["filled"] == 1
    assert _read(root, "b1") == {"p": 1, "q": 4}
    assert _read(root, "b2") == {"r": 3}


def test_routed_merge_refuses_to_change_a_non_null_label(tmp_path):
    root = _sandbox(tmp_path, {"b1": ["p"]})
    ms.merge_batch("b1", root, {"p": 2}, labeler="t", labeled_at="d", max_score=4, apply=True)
    st = ms.merge_batch("b1", root, {"p": 3}, labeler="t", labeled_at="d",
                        max_score=4, apply=True)
    assert st["conflicts"] == [("p", 2, 3)] and not st["wrote"]
    assert _read(root, "b1") == {"p": 2}, "the ONE allowed mutation is null -> value"


def test_max_score_4_refuses_a_fifth_tier(tmp_path):
    root = _sandbox(tmp_path, {"b1": ["p"]})
    st = ms.merge_batch("b1", root, {"p": 5}, labeler="t", labeled_at="d",
                        max_score=4, apply=True)
    assert st["out_of_range"] == [("p", 5)] and _read(root, "b1") == {"p": None}


def test_unknown_image_id_is_reported_not_guessed(tmp_path):
    root = _sandbox(tmp_path, {"b1": ["p"]})
    st = ms.merge_batch("b1", root, {"nope": 2}, labeler="t", labeled_at="d",
                        max_score=4, apply=True)
    assert st["unknown"] == ["nope"] and st["filled"] == 0


def test_load_route_rejects_an_entry_missing_either_half(tmp_path):
    for bad in ({"o": {"batch": "b1"}}, {"o": {"image_id": "p"}}, {"o": "b1/p"}):
        p = tmp_path / "route.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit):
            ms.load_route(str(p))


def test_batch_and_route_are_mutually_exclusive(monkeypatch, tmp_path, capsys):
    """--batch places rows by batch, --route places them by map; passing both (or neither)
    leaves it ambiguous which authority placed a label, so the CLI refuses."""
    for argv in (["merge_scores.py"],
                 ["merge_scores.py", "--batch", "b1", "--route", str(tmp_path / "r.json")]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            ms.main()
