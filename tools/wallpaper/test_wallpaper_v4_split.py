"""The v4 training split's invariants, checked without touching a crop.

The split is a function of the ROWS — `images.jsonl` plus the label sidecars — so it is
checkable at suite speed and, more to the point, it stays checkable when the crops are
gone. That is not a hypothetical: the July batches' crops have been deleted once already
and `rerender_batch_crops.py` exists to rebuild them. A guard that needs ~900 MB of
regenerable JPEG to answer "does any location train and evaluate at once?" is off exactly
when the tree is in the state that makes the question urgent.

The invariant that earns the test: the two 2026-08-05 batches stamp 19 of their 107
SHARED locations onto opposite sides, because `build_fresh_sheet.assign_split` shuffles
within a score bin over the SELECTED SET and the sibling selected a different population.
Nothing detected it until the trainer joined the batches. So this file pins both halves —
that the reconciliation removes every span, and that it does so without touching the
old-era slice the v2/v3/v4 comparison is anchored on.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.version_pinned


@pytest.fixture(scope="module")
def split():
    v4 = pytest.importorskip("classifier.train_wallpaper_v4")
    rows = v4.load_rows(require_crops=False)
    tr, ev, _, _, _, old_ids, conflicts = v4.split_union(rows)
    return {"mod": v4, "rows": rows, "train": tr, "eval": ev,
            "old_ids": old_ids, "conflicts": conflicts}


def test_every_row_is_labeled_and_placed(split):
    rows, tr, ev = split["rows"], split["train"], split["eval"]
    assert len(tr) + len(ev) == len(rows) == 3638
    assert {r.image_id for r in tr}.isdisjoint({r.image_id for r in ev})
    assert all(1 <= r.label <= 4 for r in rows)


def test_no_location_spans_both_sides(split):
    """The leakage check, on the c-inclusive coordinate key: distinct Julia `c` at a
    shared base viewport are different locations and must not be conflated."""
    sides = defaultdict(set)
    for side, rows in (("train", split["train"]), ("eval", split["eval"])):
        for r in rows:
            sides[r.full_coord].add(side)
    spanning = {c for c, s in sides.items() if len(s) > 1}
    assert not spanning, f"{len(spanning)} locations train AND evaluate: {list(spanning)[:3]}"


def test_old_era_eval_slice_is_the_686_row_anchor(split):
    """The July eval side is what v2, v3 and v4 are compared on. Its size and tier
    histogram are pinned because a silent change there invalidates every cross-version
    number in the report, and it would not otherwise announce itself."""
    old = [r for r in split["eval"] if r.era == "july"]
    assert len(old) == 686
    assert dict(Counter(r.label for r in old)) == {1: 116, 2: 295, 3: 185, 4: 90}
    assert len(split["old_ids"]) == 287, "humanq3 half of the anchor (byte-identical to v2's)"


def test_reconciliation_moves_only_the_sibling_and_only_fresh_rows(split):
    """19 conflicts, all inside the fresh pair, all resolved onto the sheet's side.

    Asserted as a bound rather than a fixed 19 for the count of MOVED ROWS only in the
    sense that each conflicting location contributes exactly its colorize row; the
    location count is pinned, because a change in it means the sibling batches were
    rebuilt and the report's fresh-era n moved with them."""
    conflicts = split["conflicts"]
    assert len(conflicts) == 19
    moved = [i for c in conflicts for i in c["moved_image_ids"]]
    assert len(moved) == 19, "one render per colorize location — the sheet never moves"
    assert all(i.startswith("wcp_") for i in moved), f"non-sibling rows moved: {moved[:5]}"
    assert all(set(c["stamped"]) == {"fresh_sheet", "colorize_path"} for c in conflicts)
    assert split["mod"].SPLIT_AUTHORITY == "fresh_sheet"


def test_fresh_eval_side_is_357_not_the_stamped_352(split):
    """The reconciliation is why the fresh eval side is 357 and not the 352 the two
    batch.json `split_summary` blocks add up to (296 + 56). Pinned so the discrepancy
    is a documented number rather than a surprise the next reader re-derives."""
    fresh_ev = [r for r in split["eval"] if r.era == "fresh"]
    assert len(fresh_ev) == 357
    assert dict(Counter(r.batch for r in fresh_ev)) == {"fresh_sheet": 296, "colorize_path": 61}


def test_coloring_and_source_axes_are_populated(split):
    """The report cuts the fresh side by coloring regime and intake vein; both axes must
    actually partition it, or a breakout silently reports one bucket."""
    fresh = [r for r in split["rows"] if r.era == "fresh"]
    assert {r.coloring_source for r in fresh} == {"pool_draw", "colorize_path"}
    assert {r.source_group for r in fresh} == {"human_q3plus", "q4_harvest", "machine_admitted"}
    assert all(r.coloring_source == "july_pool" and r.source_group == "july"
               for r in split["rows"] if r.era == "july")
