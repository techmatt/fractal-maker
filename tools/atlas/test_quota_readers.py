"""The two run-readout tools promoted out of `scratch/` for run 27.

`quota_read.py` (floor/deficit minutes and admissions, early vs late) and
`verify_quota_trace.py` (price health) were per-run throwaways under
`scratch/production_run26/` and `scratch/shakedown27/`. They are on their second run and
they answer a standing question, so they are committed — and a committed reader gets the
same treatment as anything else: the arithmetic that a report will quote is asserted here
rather than eyeballed once against the run it was written for.

What is covered is the part that is EASY TO GET SILENTLY WRONG on real data:

  * `early_late`'s COLLAPSE PREDICATE. Run 26's `mandelbrot` was served 11 times before the
    batch midpoint and never after, while its run TOTAL (6.2% of pop time, 96 admissions)
    read healthy. The predicate is the whole reason the block exists, so it is asserted
    against a trace built to have exactly one collapsed partition, one that starts late, and
    one served throughout — the three cases that must not be confused.
  * `batch_seconds`'s UNIT PARSE. The join from `quota_trace` (per allocation) to
    `stage_times` (per batch) is by `batch:<n>`, and a unit that does not parse must be
    COUNTED, not dropped — an unreported drop turns into missing minutes in a floor/deficit
    table that still sums to something plausible.

Light lane: pure functions over dicts, no run dir, no engine, no GPU.

  uv run pytest tools/atlas/test_quota_readers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "atlas")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quota_read  # noqa: E402
import verify_quota_trace as vqt  # noqa: E402


def _row(batch, chosen, prices):
    return {"batch": batch, "chosen": chosen, "price": prices, "bucket": "deficit",
            "queue_lens": {p: 1 for p in prices}}


@pytest.fixture
def trace():
    """12 batches over three partitions, one of each case the predicate must separate.

    `early` is served 5x before the split and never after (the run-26 shape, and its price
    freezes with it); `late` is served only after; `both` straddles. The batch midpoint of
    12 rows is `batches[6]` = 7, so batches 1-7 are early and 8-12 late.
    """
    rows = []
    price = {"early": 0.5, "late": 0.2, "both": 0.3}
    plan = {1: "early", 2: "early", 3: "both", 4: "early", 5: "early", 6: "early", 7: "both",
            8: "late", 9: "both", 10: "late", 11: "both", 12: "late"}
    for b in range(1, 13):
        rows.append(_row(b, plan[b], dict(price)))
        # the served partition's price steps — which is what "every served batch prices" means
        price[plan[b]] = round(price[plan[b]] * 1.1, 6)
    return rows


def test_collapse_predicate_separates_the_three_cases(trace):
    el = vqt.early_late(trace, split=7)
    assert el["pops_early"] == 7 and el["pops_late"] == 5
    assert el["service_collapsed"] == ["early"]
    assert el["service_started_late"] == ["late"]
    both = el["per_partition"]["both"]
    assert both["pops_early"] == 2 and both["pops_late"] == 2
    assert not both["service_collapsed"] and not both["never_served"]


def test_collapsed_partition_reports_a_frozen_price(trace):
    """The mechanism, not just the symptom: a partition that stops being served stops being
    priced, so `price_at_split == price_final` is the fingerprint of the lockout and must
    survive into the readout."""
    p = vqt.early_late(trace, split=7)["per_partition"]["early"]
    assert p["price_at_split"] == p["price_final"]
    served = vqt.early_late(trace, split=7)["per_partition"]["both"]
    assert served["price_at_split"] != served["price_final"]


def test_shares_are_over_the_half_not_the_run(trace):
    """A share computed against the RUN total would make every late share look small; each
    half is normalized by its own pop count."""
    el = vqt.early_late(trace, split=7)["per_partition"]
    assert el["early"]["service_share_early"] == pytest.approx(5 / 7, abs=1e-4)
    assert el["early"]["service_share_late"] == 0.0
    assert el["late"]["service_share_late"] == pytest.approx(3 / 5, abs=1e-4)


def test_never_served_is_not_a_collapse():
    rows = [_row(1, "a", {"a": 1.0, "z": 1.0}), _row(2, "a", {"a": 1.1, "z": 1.0})]
    el = vqt.early_late(rows, split=1)
    # `z` is never chosen, so it is not in the per-partition table at all and cannot be
    # counted as collapsed — a partition with no supply is a different finding (ask 4).
    assert "z" not in el["per_partition"]
    assert el["service_collapsed"] == []


def test_batch_seconds_joins_on_the_unit_and_counts_what_it_cannot_parse(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    lines = [
        {"stage": "root_draw", "unit": "preloop", "dur_s": 60.0},
        {"stage": "frontier_batch", "unit": "batch:1", "dur_s": 12.5},
        {"stage": "frontier_batch", "unit": "batch:2", "dur_s": 30.0},
        {"stage": "frontier_batch", "unit": "batch:oops", "dur_s": 99.0},
        {"stage": "dive", "unit": "dive_000", "dur_s": 5.0},
    ]
    (run / "stage_times.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    secs, diag = quota_read.batch_seconds(run)
    assert secs == {1: 12.5, 2: 30.0}          # other stages ignored, not merged
    assert diag["unit_unparsed"] == 1          # counted, never silently dropped
    assert diag["n_frontier_batch_rows"] == 2


def test_batch_seconds_refuses_a_run_with_no_stream(tmp_path):
    """A run that predates the stream must RAISE naming the backfill, not return {} — an
    empty join would print a floor/deficit table of all zeros that still cross-checks
    against nothing (`verification_practice.md` §2)."""
    run = tmp_path / "empty"
    run.mkdir()
    with pytest.raises(SystemExit, match="backfill_stage_times"):
        quota_read.batch_seconds(run)


def test_replayed_units_are_deduped_before_they_are_charged(tmp_path):
    """§11: an append-only stream is a SUPERSET of the counters after a kill. The reader
    keeps the LAST occurrence of `(stage, unit)`, so a resumed batch is charged once."""
    run = tmp_path / "run"
    run.mkdir()
    lines = [
        {"stage": "frontier_batch", "unit": "batch:1", "dur_s": 10.0},
        {"stage": "frontier_batch", "unit": "batch:1", "dur_s": 40.0},
    ]
    (run / "stage_times.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    secs, diag = quota_read.batch_seconds(run)
    assert secs == {1: 40.0}
    assert diag["stream"]["n_dup"] == 1
