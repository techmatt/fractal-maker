"""The pooled render-mode corpus: the split properties a retrain depends on.

Every assertion here is about something that would fail SILENTLY — a leak across the split, a
near-dup group counted twice, a representative chosen by the alphabet. The corpus is small
enough that the real thing is loaded once (no crops required: the split is a function of the
rows alone), so these are properties of the actual training set rather than of a fixture.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mining.mining_corpus import (BATCH_TAG, K, POOL_BATCHES,  # noqa: E402
                                        _representatives, group_label_disagreement,
                                        load_corpus)
from tools.mining.near_dup_groups import ARTIFACT as NEAR_DUP_ARTIFACT  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return load_corpus(require_crops=False)


def test_every_row_of_every_batch_is_present_and_labeled(pool):
    counts = Counter(r.batch for r in pool.rows)
    on_disk = {}
    for b in POOL_BATCHES:
        p = ROOT / "data" / "render_mode_corpus" / "batches" / b / "images.jsonl"
        on_disk[BATCH_TAG[b]] = sum(1 for line in p.read_text().splitlines() if line.strip())
    assert dict(counts) == on_disk
    assert all(1 <= r.label <= K for r in pool.rows)


def test_no_location_spans_the_split(pool):
    sides = defaultdict(set)
    for r in pool.rows:
        sides[r.loc].add(r.side)
    assert not [k for k, v in sides.items() if len(v) > 1]


def test_julia_children_share_a_side_with_their_parent_plane_point(pool):
    """The property `split_units` exists for: a Julia seed c IS a point in the parent
    family's plane, so the two are one piece of the fractal seen twice."""
    from tools.mining.split_units import JULIA_PARENT, _fkey
    base = {}
    for r in pool.rows:
        if r.family not in JULIA_PARENT:
            cx, cy = r.loc.split("|")[1:3]
            base[(r.family, _fkey(cx), _fkey(cy))] = r.side
    checked = 0
    for r in pool.rows:
        if r.family in JULIA_PARENT:
            parts = r.loc.split("|")
            c_re, c_im = parts[4], parts[5]
            if not c_re:
                continue
            key = (JULIA_PARENT[r.family], _fkey(c_re), _fkey(c_im))
            if key in base:
                checked += 1
                assert base[key] == r.side, f"{r.image_id} straddles its parent point"
    assert checked > 0, "fixture must actually contain a linked parent"


def test_near_dup_groups_never_straddle_and_are_within_one_location(pool):
    by_group = defaultdict(set)
    locs = defaultdict(set)
    for r in pool.rows:
        by_group[r.group].add(r.side)
        locs[r.group].add(r.loc)
    assert not [g for g, s in by_group.items() if len(s) > 1]
    assert not [g for g, s in locs.items() if len(s) > 1], \
        "a group spanning two locations would make the straddle guarantee accidental"


def test_train_weights_sum_to_one_per_group(pool):
    per_group = defaultdict(float)
    for r in pool.rows:
        per_group[r.group] += r.weight
    assert all(abs(v - 1.0) < 1e-9 for v in per_group.values())


def test_exactly_one_representative_per_group_and_it_is_the_median_label(pool):
    by_group = defaultdict(list)
    for r in pool.rows:
        by_group[r.group].append(r)
    for g, members in by_group.items():
        reps = [m for m in members if m.is_rep]
        assert len(reps) == 1, f"group {g} has {len(reps)} representatives"
        labels = sorted(m.label for m in members)
        import statistics
        assert reps[0].label == statistics.median_low(labels)


def test_eval_side_is_deduplicated_and_train_side_is_not(pool):
    assert len(pool.eval_rows) == len({r.group for r in pool.eval_all})
    assert len(pool.eval_rows) < len(pool.eval_all), "the corpus does contain eval near-dups"
    assert len(pool.train) == sum(1 for r in pool.rows if r.side == "train")


def test_the_stamped_sides_are_INCONSISTENT_which_is_why_the_split_is_re_derived(pool):
    """The load-bearing fact behind the global re-split. If a future rebuild made the
    stamps agree this test should be DELETED with a note, not relaxed — the re-split would
    then be a choice rather than a necessity."""
    by_loc = defaultdict(set)
    for r in pool.rows:
        by_loc[r.loc].add((r.batch, r.stamped_side))
    conflicted = [k for k, v in by_loc.items()
                  if len({s for _b, s in v}) > 1 and len({b for b, _s in v}) > 1]
    assert conflicted, "expected locations stamped onto opposite sides by two batches"
    assert pool.split_meta["rows_moved_off_stamped_side"] > 0


def test_sheet_c_reaches_the_eval_side(pool):
    """Sheet C is stamped 100% train (no location of it may be an eval INSTRUMENT). The
    pooled split is a HOLDOUT, not an instrument, and the rare-palette no-worse slice is
    unmeasurable without it."""
    assert all(r.stamped_side == "train" for r in pool.rows if r.batch == "sheetC")
    assert sum(1 for r in pool.eval_rows if r.batch == "sheetC") > 50


def test_near_dup_artifact_is_complete_and_covers_the_pool():
    doc = json.loads(NEAR_DUP_ARTIFACT.read_text(encoding="utf-8"))
    assert not doc.get("incomplete")
    assert list(doc["batches"]) == list(POOL_BATCHES)
    assert doc["cut"] == pytest.approx(0.974)


def test_group_label_disagreement_is_reported_not_assumed(pool):
    d = pool.group_meta["disagreement"]
    assert d["n_multi_groups"] > 0 and d["share"] is not None
    # the number itself is a finding, not a bar — but it must be BELOW 1.0, else "same
    # picture" would be a claim the labels flatly contradict.
    assert d["share"] < 1.0


def test_a_synthetic_disagreeing_group_keeps_the_middle_judgement():
    recs = [{"row": {"image_id": f"x{i}"}, "label": lab}
            for i, lab in enumerate([1, 3, 2])]
    group_of = {"x0": "g", "x1": "g", "x2": "g"}
    assert _representatives(recs, group_of) == {"x2"}
    d = group_label_disagreement(recs, group_of)
    assert d["n_groups_with_disagreement"] == 1 and d["label_span_hist"] == {"2": 1}


def test_no_second_copy_of_the_pooled_split():
    """`build_split` has one caller for this corpus. A module that re-derives the pooled
    split itself is a second split design wearing the same name."""
    hits = []
    pat = re.compile(r"build_split\s*\(")
    allowed = {"tools/mining/mining_corpus.py", "tools/mining/split_units.py",
               "tools/mining/build_mining_sheet.py", "tools/mining/build_mining_correction.py",
               "tools/mining/build_rare_palette_sheet.py",
               "tools/render_mode_pilot/build_scale_sample.py"}
    for p in list((ROOT / "tools").rglob("*.py")) + list((ROOT / "classifier").rglob("*.py")):
        rel = Path(p).relative_to(ROOT).as_posix()
        if rel in allowed or "__pycache__" in rel or Path(rel).name.startswith("test_"):
            continue
        if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
            hits.append(rel)
    assert not hits, f"a second pooled-split derivation: {hits}"
