"""Tripwires for the steady-state sitting readout.

Three things in that module are load-bearing and all three are silent when wrong: the
population is the ROUTE MAP (not "has a label"), the morph join is a spelling contract with
`build_q4_harvest_batches._render_block`, and one-per-cluster is taken at the LOOSE cut
because the strict cut is degenerate on an already-deduped sitting.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
           str(ROOT / "tools" / "scoring"), str(ROOT / "tools" / "atlas")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc                      # noqa: E402
import steady_state_sitting_readout as ro       # noqa: E402


def test_the_sheet_served_a_strict_subset_of_the_batches():
    """490 of 748 stored rows. The 258 the sheet excluded are still registered and still
    labelable by a later sheet, which is exactly when a readout scoped by "has a label"
    would start reporting a wider population than the one it names."""
    served = ro.routed_ids()
    per = {}
    for (b, _iid) in served:
        per[b] = per.get(b, 0) + 1
    assert len(served) == 490 and per == {ro.RANKED: 396, ro.DIVE: 94}
    stored = sum(len(cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")))
                 for b in ro.BATCHES)
    assert stored == 748 > len(served)


def test_load_labeled_is_scoped_by_the_route_map(monkeypatch):
    """PROVED BY SHRINKING THE MAP, not by counting rows: the route map and the labeled set
    coincide today (the merge filled exactly the served rows), so an equality check passes
    whether or not the scoping is there at all. Handing `load_labeled` a 5-entry map must
    yield 5 rows — a readout that scoped on `label.score is not None` would return 490."""
    full = ro.routed_ids()
    few = dict(list(full.items())[:5])
    monkeypatch.setattr(ro, "routed_ids", lambda: few)
    rows = ro.load_labeled()
    assert len(rows) == 5
    assert {(r["batch"], r["image_id"]) for r in rows} == set(few)
    assert all(r["human"] is not None for r in rows)


def test_coord_key_reproduces_the_render_blocks_own_spellings():
    """The morph join is `_render_block`'s three spellings, reproduced. If that module ever
    changes how it stringifies cx/cy/fw, this join silently yields zero hits and the readout
    reports one-per-cluster over an empty clustering."""
    import build_q4_harvest_batches as bq
    q = dict(cx="-1.125", cy="0.25", fw=1.0 / 3.0, family="mandelbrot",
             _palette="default", c_re=None, c_im=None)
    bq._PHOENIX_POOL_CACHE.update({})
    blk = bq._render_block(q)
    assert ro.coord_key(blk["cx"], blk["cy"], blk["fw"]) == \
        ro.coord_key(str(q["cx"]), str(q["cy"]), cc.hp_str(q["fw"]))


def test_one_per_cluster_cut_is_looser_than_the_cut_the_sitting_was_deduped_at():
    """`sitting_cutter.stage_morph_dedup` runs a leader/radius pass at NEAR_DUP_COS over this
    very population, so every served row is below that cosine of every other by construction.
    Reporting one-per-cluster at the same cut would be arithmetically identical to raw."""
    import sitting_cutter as sc
    assert ro.STRICT == sc.NEAR_DUP_COS
    assert ro.LOOSE < ro.STRICT


def test_leader_clusters_never_chains():
    """Leader/radius, not single linkage: every member is within the cut of its own leader, so
    a chain of near-neighbours cannot merge into one component the way linkage does."""
    import numpy as np
    C = np.array([[1.0, 0.96, 0.90],
                  [0.96, 1.0, 0.96],
                  [0.90, 0.96, 1.0]])
    a = ro.leader_clusters(C, 3, 0.95)
    assert a[0] == a[1], "1 is within the cut of leader 0"
    assert a[2] != a[0], "2 is NOT within the cut of leader 0 — linkage would chain it in"
    assert len(set(a.values())) == 2


def test_sheet_partitions_are_read_off_the_sheet_not_listed():
    """The five scoped partitions come from the sheet's own batch.json. A literal list here
    would be a second copy of a selection that was made once, at cut time."""
    parts = ro._sheet_partitions()
    assert parts == {"julia:mandelbrot", "multibrot3", "multibrot4", "multibrot5", "phoenix"}
    sel = json.load(open(os.path.join(cc.batch_dir(ro.SHEET), "batch.json"),
                         encoding="utf-8"))["selection"]
    assert all(sel["read"]["statuses"][p] == "UNCALIBRATED" for p in parts)


def test_the_staged_table_cannot_be_read_as_an_adopted_one():
    """If the staged file exists it must NOT carry an `adopted` block. Every mirror check and
    every per-version deriver reads that key; a staged table that answers to it is a set of
    train-side numbers that looks exactly like the live cut."""
    if not ro.STAGED.exists():
        pytest.skip(f"{ro.STAGED} not built in this checkout (readout stage not run)")
    doc = json.loads(ro.STAGED.read_text(encoding="utf-8"))
    assert "adopted" not in doc
    assert doc["would_be_cut"] and doc["STAGED_NOT_ADOPTED"]


@pytest.mark.parametrize("stage", ["score", "morph", "readout"])
def test_every_stage_is_reachable_by_name(stage):
    with pytest.raises(SystemExit):
        ro.main([stage, "--nonexistent-flag"])
