"""v4b's split: the prior five FROZEN, sheet A placed against them.

The whole comparison rests on one property — v3 has trained on NO eval row — and that
property is invisible when it breaks. A globally re-randomised split would look exactly like
this one and would silently inflate the baseline.

Crops are not required: the split is a function of the rows alone (v4's own `load_rows`
argues this at length), so these run without ~1 GB of regenerable JPGs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.train_wallpaper_v4 import load_rows, split_union   # noqa: E402
from classifier.train_wallpaper_v4b import (PRE_DECLARED, SHEET_A,  # noqa: E402
                                            UNLABELED_BATCH, load_union,
                                            slices_of, split_v4b)


@pytest.fixture(scope="module")
def built():
    prior, sheet_a = load_union(require_crops=False)
    tr, ev, meta = split_v4b(prior, sheet_a)
    return prior, sheet_a, tr, ev, meta


def test_the_prior_five_keep_v4s_assignment_byte_for_byte(built):
    """THE property the baseline rests on. If this fails, v3 has trained on eval rows."""
    prior, _sheet_a, tr, ev, _meta = built
    tr0, ev0, *_ = split_union(prior)
    assert {r.image_id for r in tr0} <= {r.image_id for r in tr}
    assert {r.image_id for r in ev0} <= {r.image_id for r in ev}
    assert {r.image_id for r in ev if r.batch != SHEET_A.name} == {r.image_id for r in ev0}


def test_no_location_spans_the_split_across_all_six_batches(built):
    _p, _a, tr, ev, _m = built
    sides = defaultdict(set)
    for r in tr:
        sides[r.full_coord].add("train")
    for r in ev:
        sides[r.full_coord].add("eval")
    assert not [c for c, s in sides.items() if len(s) > 1]


def test_sheet_a_collisions_are_resolved_to_the_prior_side(built):
    """The reconciliation is REAL — 73 of sheet A's locations already exist in a prior
    batch and 37 were stamped on the opposite side. A future rebuild that made the two
    agree should delete this test with a note, not weaken it."""
    _p, _a, _tr, _ev, meta = built
    rec = meta["sheet_a_reconciliation"]
    assert rec["n_colliding_with_a_prior_location"] > 0
    assert rec["n_rows_moved_to_the_prior_side"] > 0
    assert rec["authority"].startswith("the prior five")
    for m in rec["moved"]:
        assert m["stamped"] != m["resolved_to"]


def test_the_motivating_slice_has_eval_representation(built):
    """The prompt requires it explicitly: a motivating arm with no eval rows makes the
    winner rule's clause (b) unanswerable rather than false."""
    _p, _a, _tr, ev, _m = built
    sl = slices_of(ev)
    for name in PRE_DECLARED["motivating"]:
        mask = sl[name]
        assert mask.sum() >= 30, f"{name} has only {int(mask.sum())} eval rows"
        labs = [r.label for i, r in enumerate(ev) if mask[i]]
        assert sum(1 for x in labs if x >= 3) >= 10, f"{name} has too few good rows"


def test_every_pre_declared_arm_exists_and_is_populated(built):
    _p, _a, _tr, ev, _m = built
    sl = slices_of(ev)
    for role, names in PRE_DECLARED.items():
        for name in names:
            assert name in sl, f"{role} arm {name} is not a slice"
            assert sl[name].sum() > 0, f"{role} arm {name} is empty"


def test_sheet_a_is_its_own_coloring_regime(built):
    """`pool_draw_argmax` is neither `pool_draw` nor `colorize_path`; collapsing it into
    either would silently change what the two no-worse slices mean."""
    _p, sheet_a, _tr, ev, _m = built
    assert {r.coloring_source for r in sheet_a} == {"pool_draw_argmax"}
    sl = slices_of(ev)
    a = sl["sheet_a"]
    assert not (a & sl["fresh_colorize_path"]).any()
    assert not (a & sl["fresh_pool_draw"]).any()


def test_the_unlabeled_batch_is_named_and_really_is_unlabeled():
    """`train_wallpaper_v4b` claims a seventh batch directory exists and contributes
    nothing. Verified rather than asserted in prose — the claim is the reason the prompt's
    "six prior batches" is answered with five."""
    p = ROOT / "data" / "wallpaper_corpus" / "batches" / UNLABELED_BATCH / "images.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert rows and all(r["label"]["score"] is None for r in rows)
    assert not (ROOT / "labels" / "wallpaper_fresh_discovery_v1.json").exists()


def test_union_row_count_and_batch_mix(built):
    prior, sheet_a, tr, ev, _m = built
    assert len(prior) == 3638 and len(sheet_a) == 960
    assert len(tr) + len(ev) == 4598
    assert Counter(r.batch for r in tr + ev)[SHEET_A.name] == 960
