#!/usr/bin/env python
"""Guards for `tools/run_profile.py` — the post-run stage-time reader.

Two families. The STATISTICS must be right about a population they are quoting (an outlier
call over three units is not a finding), and the ABSENCES must be loud: a run dir with no
stream, a summary key the mode never publishes, and a backfilled stream are three different
states and the reader must not collapse them into "fine".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import run_profile as RP                # noqa: E402
from tools import stage_times as stimes            # noqa: E402


def _write(d: Path, units, summary=None, source=None):
    w = stimes.StageTimes(d, source=source)
    for stage, unit, dur in units:
        w.record(stage, unit, dur, t_end=(None if source else stimes._NOW))
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_stage_table_quotes_n_total_and_the_quantiles(tmp_path):
    _write(tmp_path, [("frontier_batch", f"batch:{i}", float(i)) for i in range(1, 11)])
    prof = RP.profile_dir(tmp_path)
    s = prof["stages"]["frontier_batch"]
    assert s["n"] == 10 and s["total_s"] == 55.0
    assert (s["min_s"], s["max_s"]) == (1.0, 10.0)
    assert s["p50_s"] in (5.0, 6.0)                # nearest-rank on an even population
    assert s["p90_s"] == 9.0                       # index round(0.9*9)=8 -> the 9th of 10


def test_quantiles_are_NEAREST_RANK_so_every_number_is_a_real_unit(tmp_path):
    """Interpolated quantiles print durations no unit took, which makes a p90 uncheckable
    against the top-N list printed under it."""
    xs = [1.0, 2.0, 100.0]
    assert RP._quantile(xs, 0.5) == 2.0
    assert RP._quantile(xs, 0.9) == 100.0
    assert RP._quantile([], 0.5) == 0.0


def test_an_outlier_over_3x_the_stage_median_is_FLAGGED(tmp_path):
    """Prove it red (§3): the flag has to fire on the case it exists for."""
    units = [("frontier_batch", f"batch:{i}", 50.0) for i in range(9)]
    units.append(("frontier_batch", "batch:slow", 600.0))
    _write(tmp_path, units)
    prof = RP.profile_dir(tmp_path, top=5)
    top = prof["top"][0]
    assert top["unit"] == "batch:slow" and top["flagged"] is True
    assert top["ratio"] == 12.0 and top["stage_p50_s"] == 50.0
    assert sum(1 for t in prof["top"] if t["flagged"]) == 1


def test_a_unit_just_UNDER_3x_is_not_flagged(tmp_path):
    units = [("frontier_batch", f"batch:{i}", 50.0) for i in range(9)]
    units.append(("frontier_batch", "batch:meh", 149.0))
    _write(tmp_path, units)
    prof = RP.profile_dir(tmp_path, top=3)
    assert prof["top"][0]["unit"] == "batch:meh" and prof["top"][0]["flagged"] is False


def test_a_stage_with_too_few_units_gets_NO_flags_and_says_so(tmp_path):
    """A median over 3 units is not a population to be an outlier from. The threshold column
    reads `n<5` rather than a number, so the table states the abstention instead of looking
    like a stage that happened to have no outliers."""
    _write(tmp_path, [("dive", "d0", 10.0), ("dive", "d1", 10.0), ("dive", "d2", 900.0)])
    prof = RP.profile_dir(tmp_path)
    assert prof["stages"]["dive"]["outlier_threshold_s"] is None
    assert all(not t["flagged"] for t in prof["top"])
    assert "n<5" in RP._fmt(prof, 8)


def test_top_N_ranks_by_RATIO_so_a_cheap_stage_can_outrank_an_expensive_one(tmp_path):
    """Ranking by raw seconds returns the release renders every time and buries the colorize
    attempt that took 30x its peers."""
    units = [("release_render", f"r{i}", 300.0) for i in range(5)]
    units += [("colorize", f"c{i}", 4.0) for i in range(5)]
    units.append(("colorize", "c_slow", 160.0))       # 40x its median, but only 160 s
    _write(tmp_path, units)
    prof = RP.profile_dir(tmp_path, top=3)
    assert prof["top"][0]["unit"] == "c_slow"
    assert prof["top"][0]["dur_s"] < prof["stages"]["release_render"]["p50_s"]


def test_span_starts_at_the_first_units_IMPLIED_START_not_its_end(tmp_path):
    """Using the first `t_end` drops that unit's own duration out of the span and inflates the
    unaccounted share by exactly it."""
    w = stimes.StageTimes(tmp_path)
    w.record("colorize", "a", 10.0, t_end=1000.0)
    w.record("colorize", "b", 10.0, t_end=1030.0)
    prof = RP.profile_dir(tmp_path)
    assert prof["span"]["span_s"] == 40.0             # 990 -> 1030, not 1000 -> 1030
    assert prof["span"]["accounted_s"] == 20.0
    assert prof["span"]["unaccounted_s"] == 20.0


def test_rows_without_t_end_leave_the_span_blank_rather_than_guessing(tmp_path):
    _write(tmp_path, [("frontier_batch", f"b{i}", 10.0) for i in range(6)], source="log_backfill")
    prof = RP.profile_dir(tmp_path)
    assert prof["span"]["span_s"] is None
    assert prof["stages"]["frontier_batch"]["n"] == 6      # the distribution still works


# --------------------------------------------------------------------------- #
# absences and cross-checks
# --------------------------------------------------------------------------- #
def test_a_run_dir_with_no_stream_says_UNAVAILABLE_loudly(tmp_path):
    """Every run before run 27 is in this state; a reader that printed an empty table would
    describe the whole history of this pipeline as instantaneous (§2)."""
    (tmp_path / "summary.json").write_text(json.dumps({"mode": "steered", "batches": 118}),
                                           encoding="utf-8")
    prof = RP.profile_dir(tmp_path)
    assert prof["present"] is False and prof["stages"] == {}
    assert "PER-UNIT TIMING UNAVAILABLE" in RP._fmt(prof, 8)


def test_a_backfilled_stream_is_announced_as_not_measured_in_run(tmp_path):
    _write(tmp_path, [("frontier_batch", f"b{i}", 10.0) for i in range(6)], source="log_backfill")
    prof = RP.profile_dir(tmp_path)
    assert prof["sources"] == ["log_backfill"]
    assert "NOT measured" in RP._fmt(prof, 8)


def test_stream_total_matching_the_summary_aggregate_is_ok(tmp_path):
    _write(tmp_path, [("frontier_batch", f"b{i}", 60.0) for i in range(10)],
           summary={"mode": "steered", "active_min": 10.0, "batches": 10})
    prof = RP.profile_dir(tmp_path)
    by = {c["check"]: c for c in prof["checks"]}
    assert by["frontier_batch vs summary.active_min"]["verdict"] == "ok"
    assert by["frontier_batch rows vs summary.batches"]["verdict"] == "ok"


def test_an_UNCHARGED_batch_shows_up_as_OFF_on_both_checks(tmp_path):
    """The check's reason for existing: a batch that ran but was never charged leaves the
    stream and the quota's `active_min` disagreeing, and nothing else in the record would
    say so."""
    _write(tmp_path, [("frontier_batch", f"b{i}", 60.0) for i in range(10)],
           summary={"mode": "steered", "active_min": 8.0, "batches": 8})
    by = {c["check"]: c for c in RP.profile_dir(tmp_path)["checks"]}
    assert by["frontier_batch vs summary.active_min"]["verdict"] == "off"
    assert by["frontier_batch rows vs summary.batches"]["verdict"] == "off"


def test_rounding_slack_does_not_trip_the_check_on_a_backfilled_stage(tmp_path):
    """Measured on run 26: a single 30 s refill against a 29.4 s aggregate is 2.04% and tripped
    a flat 2% tolerance on a stream that was exact. A tolerance that fires on the arithmetic of
    its own inputs gets trained out (§4)."""
    _write(tmp_path, [("root_refill", "refill:57", 30.0)], source="log_backfill",
           summary={"mode": "steered", "root_draw_min": 0.49})
    by = {c["check"]: c for c in RP.profile_dir(tmp_path)["checks"]}
    assert by["root_refill vs summary.root_draw_min"]["verdict"] == "ok"


def test_slack_is_NOT_wide_enough_to_hide_a_real_disagreement(tmp_path):
    """The control on the test above: the slack forgives rounding, not a missing unit."""
    _write(tmp_path, [("root_refill", "refill:57", 30.0)], source="log_backfill",
           summary={"mode": "steered", "root_draw_min": 1.5})
    by = {c["check"]: c for c in RP.profile_dir(tmp_path)["checks"]}
    assert by["root_refill vs summary.root_draw_min"]["verdict"] == "off"


def test_a_summary_key_the_mode_never_publishes_is_UNKNOWN_not_ok(tmp_path):
    """An aggregate this run never published cannot confirm anything, and reporting that as a
    pass is §2 exactly."""
    _write(tmp_path, [("frontier_batch", f"b{i}", 60.0) for i in range(10)],
           summary={"mode": "steered", "batches": 10})
    by = {c["check"]: c for c in RP.profile_dir(tmp_path)["checks"]}
    assert by["frontier_batch vs summary.active_min"]["verdict"] == "unknown"


def test_a_stage_neither_side_has_is_not_reported_as_a_finding(tmp_path):
    """A dive dir has no root draw and no `pre_loop_draw_min`. Nothing is being checked and
    nothing is being hidden, so it must not appear as an unknown the reader has to triage."""
    _write(tmp_path, [("dive", f"d{i}", 60.0) for i in range(6)],
           summary={"mode": "dive", "active_min": 6.0})
    checks = {c["check"] for c in RP.profile_dir(tmp_path)["checks"]}
    assert not any("pre_loop_draw_min" in c for c in checks)


def test_active_min_is_checked_against_the_stage_THIS_MODE_spends_it_in(tmp_path):
    """`active_min` names the batch loop in `steered` and the dive loop in `dive`. A flat
    mapping both failed to check the dive total and reported a permanent unknown on every
    dive dir — a red lane that protects nothing (§4)."""
    _write(tmp_path, [("dive", f"d{i}", 60.0) for i in range(6)],
           summary={"mode": "dive", "active_min": 6.0, "n_dives_done": 6})
    by = {c["check"]: c for c in RP.profile_dir(tmp_path)["checks"]}
    assert by["dive vs summary.active_min"]["verdict"] == "ok"
    assert by["dive rows vs summary.n_dives_done"]["verdict"] == "ok"
    assert not any("frontier_batch" in c for c in by)


def test_a_dive_row_count_short_of_the_summary_is_OFF(tmp_path):
    _write(tmp_path, [("dive", f"d{i}", 60.0) for i in range(5)],
           summary={"mode": "dive", "active_min": 6.0, "n_dives_done": 6})
    by = {c["check"]: c for c in RP.profile_dir(tmp_path)["checks"]}
    assert by["dive rows vs summary.n_dives_done"]["verdict"] == "off"


def test_an_emission_dir_has_no_mode_and_gets_no_aggregate_checks(tmp_path):
    """An emission out dir publishes no stage aggregate; its table stands on the rows alone.
    Inventing a check there would compare the colorize total against something that is not it."""
    _write(tmp_path, [("colorize", f"em_{i}", 4.0) for i in range(6)],
           summary={"attempts": 6, "release_rendered": 2})
    prof = RP.profile_dir(tmp_path)
    assert prof["checks"] == []
    assert prof["stages"]["colorize"]["n"] == 6


def test_an_unknown_stage_is_shown_rather_than_dropped(tmp_path):
    """A stage this module has not been taught about is a caller it does not know; sorting it
    last is fine, hiding it is not."""
    _write(tmp_path, [("frontier_batch", "b0", 1.0), ("zzz_new_stage", "u", 2.0)])
    prof = RP.profile_dir(tmp_path)
    assert list(prof["stages"]) == ["frontier_batch", "zzz_new_stage"]


def test_torn_and_duplicate_counts_reach_the_rendered_output(tmp_path):
    w = stimes.StageTimes(tmp_path)
    w.record("colorize", "em_0", 3.0)
    w.record("colorize", "em_0", 5.0)
    with open(tmp_path / stimes.STREAM, "a", encoding="utf-8") as f:
        f.write("{oops")
    txt = RP._fmt(RP.profile_dir(tmp_path), 8)
    assert "1 torn line(s)" in txt and "1 duplicate unit(s)" in txt


# --------------------------------------------------------------------------- #
# where the rows are (the emission leg's durable home, 2026-08-13)
# --------------------------------------------------------------------------- #
def test_an_emission_out_dir_FOLLOWS_its_run_id_to_the_durable_telemetry_home(tmp_path,
                                                                             monkeypatch):
    """The emission leg's two halves now live in different trees — `summary.json` in the scratch
    out dir, the per-unit rows in `data/emission/run_telemetry/<run>/` — so a reader handed the
    out dir has to find the rows and SAY where it read them. Reporting PER-UNIT TIMING
    UNAVAILABLE for a run whose telemetry is on disk is §2 with the absence invented."""
    out = tmp_path / "scratch_out" / "prod99"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"attempts": 6, "release_rendered": 2}),
                                      encoding="utf-8")
    rec = tmp_path / "rec"
    monkeypatch.setattr(RP.esinks, "default_record_root", lambda root: rec)
    home = RP.esinks.run_telemetry_dir(rec, "prod99")
    _write(home, [("colorize", f"em_{i}", 4.0) for i in range(6)])
    prof = RP.profile_dir(out)
    assert prof["present"] and prof["stages"]["colorize"]["n"] == 6
    assert Path(prof["stream_dir"]) == home
    # the summary is still read from the dir that was asked for, not from the telemetry home
    assert prof["summary_aggregates"]["attempts"] == 6
    assert "the run-keyed telemetry home" in RP._fmt(prof, 8)


def test_a_dir_that_HOLDS_the_stream_is_never_redirected(tmp_path, monkeypatch):
    """The control: the fallback is one hop for a dir with no stream, not a rewrite of every
    lookup. A discovery run dir must keep reading itself even when a same-named telemetry dir
    exists — otherwise the reader silently profiles a different run."""
    rec = tmp_path / "rec"
    monkeypatch.setattr(RP.esinks, "default_record_root", lambda root: rec)
    d = tmp_path / "prod99"
    _write(d, [("frontier_batch", "b0", 7.0)])
    _write(RP.esinks.run_telemetry_dir(rec, "prod99"), [("colorize", "em_0", 4.0)])
    prof = RP.profile_dir(d)
    assert prof["stream_dir"] is None
    assert list(prof["stages"]) == ["frontier_batch"]


def test_a_run_with_telemetry_NOWHERE_still_reports_the_absence_loudly(tmp_path, monkeypatch):
    """The fallback must not turn a missing stream into a quiet empty table."""
    monkeypatch.setattr(RP.esinks, "default_record_root", lambda root: tmp_path / "rec")
    (tmp_path / "empty").mkdir()
    txt = RP._fmt(RP.profile_dir(tmp_path / "empty"), 8)
    assert "PER-UNIT TIMING UNAVAILABLE" in txt


def test_the_path_the_run_RECORDED_wins_over_the_production_guess(tmp_path, monkeypatch):
    """An ephemeral run's telemetry is under whatever `--record-root` it was given, which no
    convention can reconstruct — so the summary carries the resolved path and the reader uses
    it. Derived at the write site, read back here (`storage_classes.md` § derive in code)."""
    rec = tmp_path / "prod_home"
    monkeypatch.setattr(RP.esinks, "default_record_root", lambda root: rec)
    _write(RP.esinks.run_telemetry_dir(rec, "prod99"), [("colorize", "wrong_run", 99.0)])
    elsewhere = tmp_path / "some_ephemeral_root" / "run_telemetry" / "prod99"
    _write(elsewhere, [("colorize", f"em_{i}", 4.0) for i in range(3)])
    out = tmp_path / "out" / "prod99"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(
        json.dumps({"stage_times_path": str(elsewhere / stimes.STREAM)}), encoding="utf-8")
    prof = RP.profile_dir(out)
    assert Path(prof["stream_dir"]) == elsewhere
    assert [t["unit"] for t in prof["top"]] == ["em_0", "em_1", "em_2"]


def test_a_recorded_path_that_is_GONE_falls_through_instead_of_deciding(tmp_path, monkeypatch):
    """The summary records where the run wrote, which is not a promise the file still exists —
    a wiped ephemeral root must not shadow a stream that IS on disk."""
    rec = tmp_path / "prod_home"
    monkeypatch.setattr(RP.esinks, "default_record_root", lambda root: rec)
    home = RP.esinks.run_telemetry_dir(rec, "prod99")
    _write(home, [("colorize", "em_0", 4.0)])
    out = tmp_path / "out" / "prod99"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(
        json.dumps({"stage_times_path": str(tmp_path / "wiped" / stimes.STREAM)}),
        encoding="utf-8")
    assert Path(RP.profile_dir(out)["stream_dir"]) == home
