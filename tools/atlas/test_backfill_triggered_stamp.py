"""The `triggered`-stamp backfill: repairs a lost stamp, and refuses everything else.

  uv run pytest tools/atlas/test_backfill_triggered_stamp.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import backfill_triggered_stamp as bts   # noqa: E402


def _row(**kw):
    base = dict(triggered=False, mix_source="sampler", maneuver=None)
    base.update(kw)
    return base


def _trig(stamped: bool):
    return _row(triggered=stamped, mix_source="triggered:snap:k=16",
                maneuver=dict(op="snap", k=16.0, triggered=True, trigger_oid="oid1"))


def _write(tmp_path, rows) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    (d / bts.STORE).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def test_backfill_sets_only_the_rows_whose_lineage_says_triggered(tmp_path):
    d = _write(tmp_path, [_trig(True), _trig(False), _trig(False), _row(), _row()])
    rec = bts.apply(d)
    assert rec["backfilled"] == 2 and rec["stamped_before"] == 1 and rec["stamped_after"] == 3
    out = bts.read_jsonl(d / bts.STORE)
    assert [bool(r["triggered"]) for r in out] == [True, True, True, False, False]


def test_a_fresh_row_is_never_stamped(tmp_path):
    """The vacuity guard for the test above: a backfill that stamped everything would also
    turn 2 rows here. The fresh rows must come back untouched."""
    d = _write(tmp_path, [_row(), _row(mix_source="maneuver:snap:k=8",
                                maneuver=dict(op="snap", k=8.0))])
    rec = bts.apply(d)
    assert rec["backfilled"] == 0
    assert [bool(r["triggered"]) for r in bts.read_jsonl(d / bts.STORE)] == [False, False]


def test_a_carrier_disagreement_refuses_and_writes_nothing(tmp_path):
    """`mix_source` says triggered, the maneuver dict does not. That is not a lost stamp, it
    is a broken lineage — the repair must not pick a winner."""
    bad = _row(mix_source="triggered:snap:k=16", maneuver=dict(op="snap", k=16.0))
    d = _write(tmp_path, [_trig(False), bad])
    before = (d / bts.STORE).read_bytes()
    with pytest.raises(SystemExit, match="disagree"):
        bts.apply(d)
    assert (d / bts.STORE).read_bytes() == before
    assert not (d / "triggered_backfill.json").exists()


def test_a_row_that_would_be_CLEARED_refuses_and_writes_nothing(tmp_path):
    """One-way by design. A stamped row with no lineage means the RULE is wrong, and a tool
    that silently cleared it would rewrite a correct record from a bad premise."""
    d = _write(tmp_path, [_row(triggered=True)])
    before = (d / bts.STORE).read_bytes()
    with pytest.raises(SystemExit, match="only ever ADDS"):
        bts.apply(d)
    assert (d / bts.STORE).read_bytes() == before


def test_apply_is_idempotent(tmp_path):
    d = _write(tmp_path, [_trig(False), _trig(False)])
    assert bts.apply(d)["backfilled"] == 2
    assert bts.apply(d)["backfilled"] == 0


def test_report_never_writes_and_names_the_columnless_artifacts(tmp_path):
    d = _write(tmp_path, [_trig(False)])
    (d / "outcome_ledger.jsonl").write_text(
        json.dumps(dict(id="a", mix_source="triggered:snap:k=16")) + "\n"
        + json.dumps(dict(id="b", mix_source="sampler")) + "\n", encoding="utf-8")
    out = bts.report([d])["run"]
    assert out["q4_candidates"]["would_set"] == [0]
    assert bts.read_jsonl(d / bts.STORE)[0]["triggered"] is False     # report wrote nothing
    led = out["outcome_ledger.jsonl"]
    assert led["has_triggered_column"] is False and led["lineage"] == 1


def test_the_live_run_store_agrees_on_every_row_after_the_backfill():
    """The production assertion, and the reason it is safe to run the repair at all: on
    `q4_long_harvest_20260803` all three carriers agree on all 7,423 rows. Not
    absence-tolerant — the store is a tracked durable artifact, so a missing file is a
    failure that names the run, never a skip."""
    d = ROOT / "data" / "discovery" / "q4_long_harvest_20260803"
    store = d / bts.STORE
    assert store.exists(), f"{store} is a tracked durable artifact and is missing"
    a = bts.audit_rows(bts.read_jsonl(store))
    assert a["n"] == 7423, a["n"]
    assert a["carrier_disagree"] == []
    assert a["would_clear"] == []
    assert a["lineage"] == a["maneuver_rec"] == 794
