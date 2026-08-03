"""The proving-run readout: each verdict, and the fixture that would make it wrong.

Every verdict here is a claim the run's report will quote, so each is tested in BOTH
directions — a readout that can only say PASS is a readout that says nothing.

  uv run pytest tools/atlas/test_harvest_v2_readout.py -q
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

import harvest_v2_readout as hr   # noqa: E402


def _summary(**over):
    base = dict(
        active_min=60.0, wall_min=72.0, batches=100, totals={},
        pop_quota=dict(
            mix=dict(minutes={"a": dict(intended=0.5, realized=0.5, delta=0.0),
                              "b": dict(intended=0.5, realized=0.5, delta=0.0)},
                     candidates={"a": dict(intended=0.5, realized=0.4, delta=-0.1),
                                 "b": dict(intended=0.5, realized=0.6, delta=0.1)},
                     admitted={"a": dict(intended=0.5, realized=0.5, delta=0.0),
                               "b": dict(intended=0.5, realized=0.5, delta=0.0)},
                     l1_gap_minutes=0.0),
            floor_vs_deficit=dict(floor_min=10.0, deficit_min=50.0, floor_share=0.167,
                                  deficit_share=0.833, per_partition={}),
            allocation=dict(floored=["a"], floor_share_total=0.05),
            cost=dict(price={}, price_raw={}, seed={}, clamped=[], clamp_factor=4.0,
                      units_mined={}, min_spent={}, capped=[])),
        maneuvers=dict(view_prior=True, view_screened=100, view_unscreenable=3, view_vetoed=7,
                       view_fit_model="view_fit_v1.1", view_fit_scored=100,
                       view_fit_coverage=1.0, view_fit_is_sort_key=False),
    )
    base.update(over)
    return base


def _run(tmp_path, summary=None, rows=None) -> Path:
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary or _summary()), encoding="utf-8")
    if rows is not None:
        (d / "q4_candidates.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


# =========================================================================== #
# 1. realized vs intended
# =========================================================================== #
def test_a_perfect_mix_passes_and_a_v1_shaped_miss_does_not():
    assert hr.mix(_summary())["verdict"] == "PASS"
    # v1's own numbers: an intended 70% native share that realized 19.6%. Every denomination
    # is replaced together — `mix` reads all three off the SAME partition list, so a fixture
    # that moved only one would be testing a malformed summary rather than a bad mix.
    s = _summary()
    v1 = {"native": dict(intended=0.70, realized=0.196, delta=-0.504),
          "julia": dict(intended=0.30, realized=0.804, delta=0.504)}
    for denom in ("minutes", "candidates", "admitted"):
        s["pop_quota"]["mix"][denom] = {k: dict(v) for k, v in v1.items()}
    s["pop_quota"]["mix"]["l1_gap_minutes"] = 0.504
    got = hr.mix(s)
    assert got["verdict"] == "MISS" and got["worst"]["partition"] == "native"


def test_the_mix_carries_all_three_denominations_because_v1_was_quoted_in_the_second():
    r = hr.mix(_summary())["per_partition"][0]
    assert set(r) == {"partition", "launch", "intended", "min", "cand", "admit",
                      "delta_min", "delta_launch"}


def test_both_intents_are_reported_because_prices_move_the_intent_mid_run():
    """The allocation is recomputed every pop from live prices, so a partition that prices
    expensive has its intended share fall while the run is still serving it. Quoting only the
    FINAL intent would grade the run against a target the run itself moved; quoting only the
    launch intent would blame the pop for the price model. Both, or neither means anything."""
    launch = {"a": 0.9, "b": 0.1}                  # what was pre-registered
    got = hr.mix(_summary(), launch)               # final intent is 0.5/0.5, realized 0.5/0.5
    assert got["l1_gap"] == 0.0 and got["verdict"] == "PASS"
    assert got["l1_gap_vs_launch"] == pytest.approx(0.4)
    assert got["verdict_vs_launch"] == "MISS"
    by = {r["partition"]: r for r in got["per_partition"]}
    assert by["a"]["launch"] == 0.9 and by["a"]["delta_launch"] == pytest.approx(-0.4)


def test_no_launch_intent_reads_None_rather_than_zero():
    """A run config that never recorded one and a run that matched it perfectly are
    different facts."""
    got = hr.mix(_summary())
    assert got["l1_gap_vs_launch"] is None and got["verdict_vs_launch"] is None


def test_an_allocator_off_run_reads_ABSENT_not_zero():
    """A missing block and a zero mix are different facts."""
    assert hr.mix(dict(active_min=1))["verdict"] == "ABSENT"


# =========================================================================== #
# 3. view screen
# =========================================================================== #
def test_full_coverage_passes_partial_does_not_and_v1s_zero_reads_NOT_RUN():
    assert hr.view_screen(_summary())["verdict"] == "PASS"
    s = _summary()
    s["maneuvers"]["view_fit_scored"] = 60
    assert hr.view_screen(s)["verdict"] == "PARTIAL"
    s2 = _summary()
    s2["maneuvers"].update(view_screened=0, view_fit_scored=0)
    assert hr.view_screen(s2)["verdict"] == "NOT RUN"


# =========================================================================== #
# 4. the triggered stamp, on live output
# =========================================================================== #
def _trow(depth, triggered=True, src="triggered:snap:k=16"):
    return dict(partition="multibrot3", depth=depth, triggered=triggered,
                mix_source=(src if triggered else "sampler"), fate="admitted")


def test_multiple_generations_pass_and_a_single_generation_is_named(tmp_path):
    """THE discriminating fact. A run where the stamp still died after one generation shows
    triggered rows at exactly one depth — which is what v1's record looks like."""
    d = _run(tmp_path, rows=[_trow(5), _trow(6), _trow(7)])
    assert hr.triggered_lineage(d)["verdict"] == "PASS"
    d2 = _run(tmp_path / "x", rows=[_trow(5), _trow(5), _trow(5)])
    assert hr.triggered_lineage(d2)["verdict"] == "SINGLE GENERATION"


def test_a_carrier_mismatch_is_named_rather_than_averaged(tmp_path):
    rows = [_trow(5), _trow(6),
            dict(partition="multibrot3", depth=7, triggered=True, mix_source="sampler",
                 fate="admitted")]
    d = _run(tmp_path, rows=rows)
    got = hr.triggered_lineage(d)
    assert got["verdict"] == "CARRIER MISMATCH" and got["carrier_disagreements"] == 1


def test_a_run_with_no_triggers_is_NO_TRIGGERS_not_a_failure(tmp_path):
    d = _run(tmp_path, rows=[_trow(3, triggered=False), _trow(4, triggered=False)])
    assert hr.triggered_lineage(d)["verdict"] == "NO TRIGGERS"


# =========================================================================== #
# 5. per-channel
# =========================================================================== #
def test_channels_are_split_on_the_mix_source_head_token(tmp_path):
    rows = [dict(mix_source="sampler", fate="admitted", depth=1, partition="p",
                 triggered=False),
            dict(mix_source="triggered:snap:k=16", fate="q3_dup", depth=2, partition="p",
                 triggered=True),
            dict(mix_source="julia_hook<st_multibrot3_x_000001", fate="admitted", depth=3,
                 partition="p", triggered=False),
            dict(mix_source="phoenix_sampler:period2", fate="below_tau_h", depth=1,
                 partition="p", triggered=False)]
    d = _run(tmp_path, rows=rows)
    got = hr.per_stage_per_channel(d)
    assert set(got) == {"sampler", "triggered", "julia_hook", "phoenix_sampler"}
    assert got["sampler"]["_total"] == 1 and got["triggered"]["q3_dup"] == 1


# =========================================================================== #
# the driver
# =========================================================================== #
def test_a_run_with_no_summary_is_a_loud_failure(tmp_path):
    """A killed run leaves state.json but no summary. Reading state.json instead would
    report a partial run's numbers as a finished run's."""
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit, match="has not finished"):
        hr.readout(tmp_path / "run")


def test_the_readout_answers_every_question_the_prompt_asks(tmp_path):
    d = _run(tmp_path, rows=[_trow(5), _trow(6)])
    rep = hr.readout(d)
    assert set(rep) >= {"realized_vs_intended", "floor_vs_deficit", "view_screen",
                        "triggered_lineage", "per_stage_per_channel", "cost_to_mine"}
