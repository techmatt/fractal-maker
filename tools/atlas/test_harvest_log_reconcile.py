#!/usr/bin/env python
"""Tests for `harvest_log_reconcile.py`.

The load-bearing one is `test_a_resumed_batch_is_deduped_before_the_rate_is_taken`: the
whole tool exists because a re-appended batch inflates a denominator, and a test that only
exercised a clean log would pass on a tool that never deduped at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import harvest_log_reconcile as hlr        # noqa: E402


def _row(node_id, batch=1, partition="mandelbrot", precanon_dup=None, admitted=False):
    return dict(batch=batch, node_id=node_id, partition=partition,
                precanon_dup=precanon_dup, admitted=admitted, cheap_pgood=0.5)


def _run(tmp_path, rows, totals, name="r"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "harvest_log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(dict(totals=totals)), encoding="utf-8")
    return d


def test_a_resumed_batch_is_deduped_before_the_rate_is_taken(tmp_path):
    """Batch 7 is re-run after a kill and re-appends its three rows. The counters already
    hold them, so the deduped log must be 4 rows and the skip rate 1/4, not 2/7."""
    rows = [_row(1), _row(2, batch=7, precanon_dup="n1"), _row(3, batch=7),
            _row(4, batch=8, admitted=True)]
    rows += [_row(2, batch=7, precanon_dup="n1"), _row(3, batch=7)]     # the redone batch
    d = _run(tmp_path, rows, dict(harvest_checks=4, precanon_dup=1, admitted=1))
    rep = hlr.readout(d)
    assert rep["dedup"]["rows_read"] == 6
    assert rep["dedup"]["rows_deduped"] == 4
    assert rep["dedup"]["duplicates_dropped"] == 2
    assert rep["dedup"]["duplicate_batches"] == {"7": 2} or \
           rep["dedup"]["duplicate_batches"] == {7: 2}
    assert rep["reconcile"]["verdict"] == "RECONCILED"
    assert rep["rates"]["pooled"]["precanon_skip_rate"] == pytest.approx(0.25)


def test_a_log_that_does_not_tie_to_the_counters_raises_naming_every_identity(tmp_path):
    """The guard, and it must fail LOUD rather than report a rate over a population the
    run's own counters disagree with."""
    d = _run(tmp_path, [_row(1), _row(2, precanon_dup="n1")],
             dict(harvest_checks=5, precanon_dup=3, admitted=0))
    with pytest.raises(hlr.ReconcileError) as e:
        hlr.readout(d)
    msg = str(e.value)
    assert "harvest_checks: log 2 != summary totals 5" in msg
    assert "precanon_dup: log 1 != summary totals 3" in msg


def test_dedup_keeps_the_last_row_per_node(tmp_path):
    """The surviving row must be the one the counters were last advanced by — a re-run
    batch can decode differently (a re-render is not bit-identical), and keeping the FIRST
    would tie the rate to a fate the summary is not a count of."""
    rows = [_row(1, admitted=False), _row(1, admitted=True)]
    d = _run(tmp_path, rows, dict(harvest_checks=1, precanon_dup=0, admitted=1))
    rep = hlr.readout(d)
    assert rep["rates"]["pooled"]["admitted"] == 1


def test_a_partition_with_no_checks_reports_null_not_zero(tmp_path):
    """Population-gated at the reader: a rate over an empty denominator is not 0.0."""
    assert hlr.rates([])["pooled"]["precanon_skip_rate"] is None
    d = _run(tmp_path, [_row(1, partition="phoenix")],
             dict(harvest_checks=1, precanon_dup=0, admitted=0))
    rep = hlr.readout(d)
    assert set(rep["rates"]["per_partition"]) == {"phoenix"}
    assert rep["rates"]["per_partition"]["phoenix"]["precanon_skip_rate"] == 0.0


def test_a_torn_tail_line_is_counted_not_silently_skipped(tmp_path):
    """A killed process can leave a partial final line. Dropping it quietly is an
    absence-tolerant read of a damaged subject."""
    d = _run(tmp_path, [_row(1)], dict(harvest_checks=1, precanon_dup=0, admitted=0))
    with open(d / "harvest_log.jsonl", "a", encoding="utf-8") as f:
        f.write('{"batch": 2, "node_i')
    rep = hlr.readout(d)
    assert rep["dedup"]["torn_lines"] == 1
    assert rep["reconcile"]["verdict"] == "RECONCILED"


def test_tier2_volume_is_the_checks_that_paid_a_confirmation_render(tmp_path):
    """The bill the precanon adoption moves: checks minus precanon skips, per partition."""
    rows = [_row(1, precanon_dup="a"), _row(2), _row(3, admitted=True),
            _row(4, partition="multibrot3", precanon_dup="b")]
    d = _run(tmp_path, rows, dict(harvest_checks=4, precanon_dup=2, admitted=1))
    rep = hlr.readout(d)
    assert rep["rates"]["pooled"]["tier2_renders"] == 2
    assert rep["rates"]["per_partition"]["mandelbrot"]["tier2_renders"] == 2
    assert rep["rates"]["per_partition"]["multibrot3"]["tier2_renders"] == 0
    assert rep["rates"]["per_partition"]["multibrot3"]["precanon_skip_rate"] == 1.0


def test_an_unfinished_run_is_refused_rather_than_read_off_state_json(tmp_path):
    d = tmp_path / "unfinished"
    d.mkdir()
    (d / "harvest_log.jsonl").write_text(json.dumps(_row(1)) + "\n", encoding="utf-8")
    (d / "state.json").write_text(json.dumps(dict(totals=dict(harvest_checks=1))),
                                  encoding="utf-8")
    with pytest.raises(hlr.ReconcileError, match="summary.json"):
        hlr.readout(d)


def test_a_missing_log_raises_instead_of_reporting_an_empty_population(tmp_path):
    d = tmp_path / "nolog"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(dict(totals={})), encoding="utf-8")
    with pytest.raises(hlr.ReconcileError, match="harvest_log.jsonl"):
        hlr.readout(d)
