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

# EVERY built sheet, not just the first one. The tripwires below were written against
# `q4_combined` and stayed pinned to it while three more sittings were built from the same
# code — so a blindness regression in a NEW sheet would have been invisible to the suite until
# somebody re-ran `check` by hand. They are properties of the CODE, so they are parametrized
# over `SPECS` and each one skips only if ITS sheet is unbuilt in this checkout.
ALL_SPECS = sorted(B.SPECS)


def _spec_or_skip(name):
    spec = B.SPECS[name]
    if not (B.sheet_dir(spec) / B.SHEET_MANIFEST).exists():
        pytest.skip(f"{name} sheet not built in this checkout")
    return spec, B.sheet_dir(spec)


# --------------------------------------------------------------------------- #
# tripwires over the built sheets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_SPECS)
def test_sheet_dir_is_not_a_batch(name):
    """The consequence, not the proxy: every consumer discovers batches by globbing
    `*/images.jsonl`, so the sheet must not appear in that glob."""
    import glob
    spec, sd = _spec_or_skip(name)
    found = glob.glob(os.path.join(cc.BATCHES_DIR, "*", "images.jsonl"))
    assert not (sd / "images.jsonl").exists()
    assert str(sd / "images.jsonl") not in found
    assert spec.sheet_id not in {os.path.basename(os.path.dirname(p)) for p in found}


@pytest.mark.parametrize("name", ALL_SPECS)
def test_served_manifest_carries_no_leak_key_and_no_source_identity(name):
    spec, sd = _spec_or_skip(name)
    text = (sd / B.SHEET_MANIFEST).read_text(encoding="utf-8")
    for k in B.SHEET_LEAK_KEYS:
        assert f'"{k}"' not in text, f"{name}: leak key {k} reached the served manifest"
    for b in spec.sources:
        assert b not in text, f"{name}: source batch id {b} reached the served manifest"


@pytest.mark.parametrize("name", ALL_SPECS)
def test_served_rows_are_id_render_label_only(name):
    _spec, sd = _spec_or_skip(name)
    rows = cc.read_jsonl(str(sd / B.SHEET_MANIFEST))
    assert rows and all(set(r) == {"image_id", "render", "label"} for r in rows)
    assert all(r["label"]["score"] is None for r in rows)
    assert all(set(cc.RENDER_KEYS) <= set(r["render"]) for r in rows)


@pytest.mark.parametrize("name", ALL_SPECS)
def test_route_is_a_bijection_onto_the_selected_source_rows(name):
    """Onto the SELECTION, which for an unfiltered sheet is the whole batch. Asserting the
    whole batch unconditionally would have made a subset sheet unrepresentable — and the
    interesting failure (a subset sheet that quietly served rows outside its own filter)
    is exactly what this catches."""
    spec, sd = _spec_or_skip(name)
    rows = cc.read_jsonl(str(sd / B.SHEET_MANIFEST))
    route = ms.load_route(str(sd / B.ROUTE_FILE))
    assert set(route) == {r["image_id"] for r in rows}
    targets = list(route.values())
    assert len(set(targets)) == len(targets), "two opaque ids point at one source row"
    sel = B.load_sources(spec)
    for b in spec.sources:
        assert {i for bb, i in targets if bb == b} == {r["image_id"] for bb, r in sel if bb == b}


@pytest.mark.parametrize("name", ALL_SPECS)
def test_batch_json_pins_the_file_order(name):
    """The sheet's file order IS the designed stratification; the page must not reshuffle it."""
    _spec, sd = _spec_or_skip(name)
    bj = json.loads((sd / "batch.json").read_text(encoding="utf-8"))
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


# --------------------------------------------------------------------------- #
# the SPECS registry — a second sheet must not collide with the first
# --------------------------------------------------------------------------- #
def test_every_spec_is_distinct_where_a_collision_would_be_silent():
    """Three fields decide whether two sheets can coexist. A shared `sheet_id` writes one
    sitting over another; a shared `salt`+`id_prefix` mints the same opaque id for two
    different rows, and `route.json` is keyed on that id — so an export would route a label to
    the wrong batch, which no downstream check can detect."""
    specs = list(B.SPECS.values())
    assert len({s.name for s in specs}) == len(specs)
    assert len({s.sheet_id for s in specs}) == len(specs)
    assert len({(s.salt, s.id_prefix) for s in specs}) == len(specs)


def test_every_spec_names_registered_source_batches():
    """A sheet over an UNREGISTERED batch serves rows nobody classified. `assign_split` falls
    closed to train/biased, so the sitting would build and look correct."""
    from tools.v7 import build_manifest as bm
    for s in B.SPECS.values():
        assert s.sources, s.name
        for b in s.sources:
            _split, _biased, source = bm.assign_split({"batch": b, "ft": "mandelbrot"})
            assert source != "unregistered", f"{s.name} -> {b}"


def test_a_sheet_id_is_never_also_a_source_batch_id():
    """The sheet dir must not be one of the batches it serves — it would gain an images.jsonl
    and be unioned into training as a duplicate of every row it presents."""
    ids = {s.sheet_id for s in B.SPECS.values()}
    srcs = {b for s in B.SPECS.values() for b in s.sources}
    assert not (ids & srcs)


# --------------------------------------------------------------------------- #
# SUBSET sheets — the filter, and what it must not disturb
# --------------------------------------------------------------------------- #
def test_a_filter_without_a_stated_rule_is_refused():
    """A served subset whose rule is not written into batch.json is indistinguishable from a
    build that silently lost rows, so the two fields are set together or not at all."""
    base = dict(name="t", sheet_id="t", sources=("b",), seed=1, salt="s", id_prefix="tt",
                purpose="p", max_run_source=1, max_run_family=1)
    for bad in ({"row_filter": (lambda b, r: True)}, {"filter_rule": "stated but not applied"}):
        with pytest.raises(ValueError):
            B.SheetSpec(**base, **bad)
    B.SheetSpec(**base)                                        # neither: the whole union
    B.SheetSpec(**base, row_filter=(lambda b, r: True), filter_rule="both")


def test_the_status_reader_and_the_adopted_table_agree():
    """Two independent authorities for the same fact: the DERIVATION artifact's per-partition
    `status` stamp, and `production_seeder`'s adopted mirror of it. The filter reads the first;
    the discovery path runs on the second. Equal, or the sitting is cut against a table the
    engine does not use."""
    import sys as _s
    _s.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "scoring")))
    _s.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "mining")))
    _s.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "atlas")))
    import derive_t_good as est
    import production_seeder as ps
    from partitions import ALL_FAMS
    stamped = est.adopted_statuses()
    assert set(stamped) == set(ALL_FAMS)
    assert stamped == {f: ps.t_good_status(f) for f in ALL_FAMS}


def test_the_filtered_sitting_is_exactly_uncalibrated_ranked_plus_the_whole_dive():
    """Re-derived from the artifact, never from a literal count: flip a partition's stamp and
    this moves with it. A frozen 396 would pass while serving the wrong population."""
    spec = B.SPECS["steady_state_uncal"]
    ranked, dive = spec.sources
    sel = B.load_sources(spec)
    full = B.load_sources(spec, filtered=False)

    from partitions import partition_of_row
    want = {(b, r["image_id"]) for b, r in full
            if b != ranked or B.t_good_status_of(r) == "UNCALIBRATED"}
    assert {(b, r["image_id"]) for b, r in sel} == want
    assert {b for b, _ in sel if b == dive} == {dive}
    assert sum(1 for b, _ in sel if b == dive) == sum(1 for b, _ in full if b == dive), \
        "the dive leg is served WHOLE — it is not scoped by the filter"
    # and nothing DERIVED survives from the scoped leg.
    assert not [r for b, r in sel
                if b == ranked and B.t_good_status_of(r) != "UNCALIBRATED"]
    served = {partition_of_row(r["render"]) for b, r in sel if b == ranked}
    assert served and all(B._t_good_statuses()[p] == "UNCALIBRATED" for p in served)


def test_the_excluded_rows_stay_registered_labelable_and_unlabeled():
    """"Nothing deleted, nothing unregistered" as a property of the store, not a claim in a
    report: the rows the filter dropped are still in their batch, still null, and still
    reachable by a later sheet."""
    spec = B.SPECS["steady_state_uncal"]
    sel = {(b, r["image_id"]) for b, r in B.load_sources(spec)}
    full = B.load_sources(spec, filtered=False)
    excluded = [(b, r) for b, r in full if (b, r["image_id"]) not in sel]
    assert excluded, "this spec is supposed to exclude rows"
    for b, r in excluded:
        assert r["label"]["score"] is None
    from tools.v7 import build_manifest as bm
    for b in {b for b, _ in excluded}:
        assert bm.assign_split({"batch": b, "ft": "mandelbrot"})[2] != "unregistered"


def test_an_unscoped_batch_is_served_whole_by_the_scoped_filter():
    """The scoping is the editorial rule and it is easy to invert by accident. A row from a
    batch the filter does not name passes even when its partition is DERIVED."""
    keep = B.uncalibrated_t_good_in("scoped")
    derived_row = {"image_id": "x", "render": {"fractal_type": "mandelbrot"}}   # DERIVED in v10
    assert keep("unscoped", derived_row) is True
    assert keep("scoped", derived_row) is False


def test_an_empty_selection_is_refused_rather_than_built():
    """A filter that matches nothing produces a clean-looking build of an empty sitting —
    the failure this refuses is the one that reads as a success."""
    spec = B.SPECS["steady_state_uncal"]
    never = B.SheetSpec(name="never", sheet_id="never", sources=spec.sources, seed=1,
                        salt="never", id_prefix="nv", purpose="p", max_run_source=1,
                        max_run_family=1, row_filter=(lambda b, r: False),
                        filter_rule="matches nothing")
    with pytest.raises(SystemExit):
        B.load_sources(never)


def test_a_row_whose_partition_is_unregistered_raises_rather_than_being_dropped():
    """An unrecognised family must not fall silently to "not uncalibrated" — that would drop
    a whole new family out of a sitting and read as the filter working."""
    with pytest.raises(KeyError):
        B.t_good_status_of({"image_id": "x", "render": {"fractal_type": "nonesuch"}})
