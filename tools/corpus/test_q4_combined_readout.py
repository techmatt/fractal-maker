"""Guards on the three durable records `q4_combined_readout.py` writes.

What these DO check: that each record still covers the population it claims, and that its
claim about the source batch still matches what the authorities say. What they deliberately
do NOT check: that the recorded rates equal a fresh recomputation. A record freezes what was
true when written, and a later REVISION through the amendment stream legitimately moves
`resolve_score` without invalidating the record (`storage_classes.md`, "Derive in code,
freeze in records") — a test asserting equality would go red on a correct amendment.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "tools"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc      # noqa: E402
import label_store as ls        # noqa: E402

INSTRUMENT = os.path.join(ROOT, "data/label_corpus/eval_instruments/q4_uniform_eval_v1.json")
CLUSTERS = os.path.join(ROOT, "data/label_corpus/motif/q4_combined_2026-08-03_clusters.jsonl")
QUEUE = os.path.join(ROOT, "data/discovery/q4_long_harvest_20260803/human_q3plus_queue.jsonl")
SOURCES = ("2026-08-03_q4_harvest_ranked_v1", "2026-08-03_q4_near_minibrot_v1",
           "2026-08-03_q4_uniform_eval_v1")
EVAL_BATCH = "2026-08-03_q4_uniform_eval_v1"


def _rows(batch):
    return cc.read_jsonl(os.path.join(cc.batch_dir(batch), "images.jsonl"))


def test_eval_instrument_covers_every_row_of_its_batch():
    """Sum of the per-family denominators == the batch it names. A record whose cells no
    longer add up to its population is describing a different draw than the one it cites."""
    doc = json.load(open(INSTRUMENT, encoding="utf-8"))
    assert doc["batch"] == EVAL_BATCH
    n = sum(f["ge2"]["n"] for f in doc["rates"].values())
    assert n == len(_rows(EVAL_BATCH)) == 290
    for fam, cells in doc["rates"].items():
        for tier in ("ge2", "ge3", "eq4"):
            assert 0 <= cells[tier]["k"] <= cells[tier]["n"], (fam, tier)
        # monotone by construction: {>=2} superset {>=3} superset {==4}
        assert cells["ge2"]["k"] >= cells["ge3"]["k"] >= cells["eq4"]["k"], fam


def test_eval_instrument_batch_is_still_registered_unbiased():
    """The record's whole claim is that it is an UNBIASED base rate. If the batch's
    registration ever moved train-side, the record would be quoting a biased draw."""
    bm = pytest.importorskip("tools.v7.build_manifest")
    split, biased, source = bm.assign_split({"batch": EVAL_BATCH, "ft": "mandelbrot"})
    assert (split, biased) == ("eval", False), (split, biased, source)
    assert EVAL_BATCH not in ls.TRAIN_SIDE_ONLY_BATCHES


def test_cluster_assignment_is_exactly_the_sitting():
    """One cluster row per labeled row of the three source batches, no more, no fewer —
    a partial assignment would silently under-weight whatever it dropped."""
    cl = [json.loads(l) for l in open(CLUSTERS, encoding="utf-8")]
    assert len(cl) == 870
    got = {(r["batch"], r["image_id"]) for r in cl}
    want = {(b, r["image_id"]) for b in SOURCES for r in _rows(b)}
    assert got == want
    for r in cl:
        assert r["cluster_linkage_size"] >= 1 and r["cluster_ball_size"] >= 1
    # sizes must be consistent with the assignment they came from
    from collections import Counter
    for key, size in (("cluster_linkage", "cluster_linkage_size"),
                      ("cluster_ball", "cluster_ball_size")):
        seen = Counter(r[key] for r in cl)
        assert all(seen[r[key]] == r[size] for r in cl), key


def test_queue_is_ge3_and_excludes_the_admitted_set():
    """The residue's two defining predicates, held as data rather than as prose."""
    q = [json.loads(l) for l in open(QUEUE, encoding="utf-8")]
    assert q, "queue is empty"
    assert all(r["human"] >= 3 for r in q)
    assert all(r["fate"] != "admitted" for r in q)
    ids = {(r["batch"], r["image_id"]) for r in q}
    assert len(ids) == len(q), "duplicate rows in the queue"
    # every queued row still resolves to the score the queue recorded
    for b in SOURCES:
        side, amend = ls.sidecar_for(b), ls.amendments_for(b)
        by_id = {r["image_id"]: r for r in _rows(b)}
        for r in q:
            if r["batch"] == b:
                assert ls.resolve_score(by_id[r["image_id"]], side, amend) == r["human"]
