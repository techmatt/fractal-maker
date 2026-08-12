#!/usr/bin/env python
"""Guards for `tools/stage_times.py` — the per-unit stage timing stream.

The defects worth guarding here are all about what happens when a run DIES, because that is
the state this stream is most often read in (`verification_practice.md` §11): a torn final
line, a resumed run's replayed units, and an absent stream must each be a distinguishable,
reported outcome rather than a crash or a silent zero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import stage_times as stimes            # noqa: E402


def test_record_round_trips_contract_fields_and_meta(tmp_path):
    w = stimes.StageTimes(tmp_path)
    w.record("frontier_batch", "batch:7", 12.25, partition="multibrot3", n_cands=99)
    rows, diag = stimes.read(tmp_path)
    assert diag == {"present": True, "n_raw": 1, "n_torn": 0, "n_dup": 0}
    (r,) = rows
    assert (r["stage"], r["unit"], r["dur_s"], r["seq"]) == ("frontier_batch", "batch:7", 12.25, 0)
    assert r["meta"] == {"partition": "multibrot3", "n_cands": 99}
    assert r["t_end"] > 0
    assert "source" not in r          # in-run rows are unstamped; only a backfill stamps


def test_unit_is_stringified_so_an_int_and_its_string_are_one_unit(tmp_path):
    w = stimes.StageTimes(tmp_path)
    w.record("dive", 3, 1.0)
    rows, _ = stimes.read(tmp_path)
    assert rows[0]["unit"] == "3"


def test_timed_records_the_block_and_merges_the_yielded_meta(tmp_path):
    w = stimes.StageTimes(tmp_path)
    with w.timed("intake", "all", n_ledgers=12) as m:
        m["n_admitted"] = 4775
    (r,), _ = stimes.read(tmp_path)
    assert r["stage"] == "intake"
    assert r["meta"] == {"n_ledgers": 12, "n_admitted": 4775}
    assert r["dur_s"] >= 0.0


def test_timed_records_a_RAISING_block_and_stamps_it_failed(tmp_path):
    """A stage that died after eight minutes still spent them. Dropping the row would make a
    crashing run read as a fast one — §2's absence-tolerance, in the timing stream."""
    w = stimes.StageTimes(tmp_path)
    with pytest.raises(RuntimeError):
        with w.timed("select", "all"):
            raise RuntimeError("boom")
    (r,), _ = stimes.read(tmp_path)
    assert r["stage"] == "select" and r["meta"]["failed"] is True


def test_t_end_None_OMITS_the_field_rather_than_stamping_now(tmp_path):
    """The backfill case. `None` becoming `time.time()` would stamp a reconstruction of an
    August run with the moment the reconstruction ran, and every span computed downstream
    would be a measurement of that."""
    w = stimes.StageTimes(tmp_path, source="log_backfill")
    w.record("frontier_batch", "batch:1", 20.0, t_end=None)
    (r,), _ = stimes.read(tmp_path)
    assert "t_end" not in r
    assert r["source"] == "log_backfill"


def test_read_counts_a_TORN_final_line_instead_of_raising(tmp_path):
    """A kill mid-append leaves at most one unparseable line. The rows before it are intact
    and must still be readable, and the tear must be REPORTED, not absorbed."""
    w = stimes.StageTimes(tmp_path)
    w.record("colorize", "em_0", 3.0)
    with open(tmp_path / stimes.STREAM, "a", encoding="utf-8") as f:
        f.write('{"stage": "colorize", "un')
    rows, diag = stimes.read(tmp_path)
    assert diag["n_torn"] == 1 and diag["n_raw"] == 2
    assert [r["unit"] for r in rows] == ["em_0"]


def test_read_DEDUPS_a_replayed_unit_keeping_the_last_and_counting_it(tmp_path):
    """§11: an append-only log is a superset of the checkpointed counters after a kill. The
    resumed run re-ran `batch:5` and appended it again; averaging both would bias the stage's
    distribution by however often the run died."""
    w = stimes.StageTimes(tmp_path)
    w.record("frontier_batch", "batch:5", 100.0)
    w.record("frontier_batch", "batch:6", 10.0)
    w2 = stimes.StageTimes(tmp_path)                      # the resumed session
    w2.record("frontier_batch", "batch:5", 40.0)
    rows, diag = stimes.read(tmp_path)
    assert diag["n_raw"] == 3 and diag["n_dup"] == 1
    assert [(r["unit"], r["dur_s"]) for r in rows] == [("batch:5", 40.0), ("batch:6", 10.0)]
    assert stimes.totals_of(rows) == {"frontier_batch": {"n": 2, "total_s": 50.0}}


def test_the_same_unit_in_two_DIFFERENT_stages_is_not_a_duplicate(tmp_path):
    w = stimes.StageTimes(tmp_path)
    w.record("colorize", "all", 1.0)
    w.record("select", "all", 2.0)
    rows, diag = stimes.read(tmp_path)
    assert diag["n_dup"] == 0 and len(rows) == 2


def test_absent_stream_reports_present_False_and_does_not_raise(tmp_path):
    rows, diag = stimes.read(tmp_path)
    assert rows == [] and diag["present"] is False


def test_totals_are_THIS_INSTANCE_not_the_file(tmp_path):
    """A resumed run's file holds the previous session's units. `summary.json` reports what
    THIS session spent, so folding the file's rows in would report a wall it never spent."""
    stimes.StageTimes(tmp_path).record("frontier_batch", "batch:1", 500.0)
    w2 = stimes.StageTimes(tmp_path)
    w2.record("frontier_batch", "batch:2", 7.0)
    assert w2.totals() == {"frontier_batch": {"n": 1, "total_s": 7.0}}
    rows, _ = stimes.read(tmp_path)                        # the FILE has both
    assert stimes.totals_of(rows) == {"frontier_batch": {"n": 2, "total_s": 507.0}}


def test_constructing_a_writer_creates_nothing(tmp_path):
    """A tool builds its writers before it has decided to run; construction must not create a
    run dir (and must not leave an empty stream a reader would call `present`)."""
    d = tmp_path / "never"
    stimes.StageTimes(d)
    assert not d.exists()


def test_every_stage_a_driver_records_is_declared_in_STAGES():
    """Source scan, with the paired control below. `STAGES` is what a reader of a whole run
    expects to see and what `run_profile` orders its table by; a stage recorded by a driver but
    missing from it sorts into the unknown bucket and nobody notices it was never listed."""
    import re
    pat = re.compile(r"stage_times\.(?:record|timed)\(\s*[\"']([a-z_]+)[\"']"
                     r"|\.stage_times\.(?:record|timed)\(\s*[\"']([a-z_]+)[\"']")
    found = set()
    for py in (ROOT / "tools").rglob("*.py"):
        if py.name.startswith("test_") or py.name == "stage_times.py":
            continue
        for m in pat.finditer(py.read_text(encoding="utf-8", errors="replace")):
            found.add(m.group(1) or m.group(2))
    assert found, "the scan matched nothing — the regex has drifted from the call sites"
    assert found <= set(stimes.STAGES), f"undeclared stage(s): {sorted(found - set(stimes.STAGES))}"


def test_the_stage_scan_would_CATCH_an_undeclared_stage(tmp_path):
    """§9's paired control: the scan above passes trivially if its regex matches nothing, so
    this asserts the regex actually recognises a call site."""
    import re
    pat = re.compile(r"stage_times\.(?:record|timed)\(\s*[\"']([a-z_]+)[\"']"
                     r"|\.stage_times\.(?:record|timed)\(\s*[\"']([a-z_]+)[\"']")
    src = 'self.stage_times.record("not_a_declared_stage", "u", 1.0)\n'
    m = pat.search(src)
    assert m and (m.group(1) or m.group(2)) == "not_a_declared_stage"
    assert "not_a_declared_stage" not in stimes.STAGES


def test_rows_are_one_json_object_per_line(tmp_path):
    """The stream is read by `run_record.iter_lines`, which is line-oriented; a pretty-printed
    row would be a torn line to every reader."""
    w = stimes.StageTimes(tmp_path)
    w.record("dive", "dive_000", 359.0, end_cause="target_depth")
    text = (tmp_path / stimes.STREAM).read_text(encoding="utf-8")
    assert text.count("\n") == 1 and text.endswith("\n")
    json.loads(text)
