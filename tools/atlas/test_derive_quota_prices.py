#!/usr/bin/env python
"""Tests for `derive_quota_prices.py` — the pop-quota cost-to-mine seed regenerator.

Each test names the defect it would catch. The round-trip test is the load-bearing one:
it asserts the written file is readable BY THE REAL CONSUMER (`pop_quota.CostToMine`)
rather than by a second parser written here, which would only prove `f(x) == f(x)`
(`verification_practice.md` §1.10).
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

import derive_quota_prices as dqp        # noqa: E402
import pop_quota as pquota               # noqa: E402


def _run(tmp_path, name, *, min_spent, units, samples=None, block=True):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    cost = dict(min_spent=min_spent, units_mined=units,
                price_samples=samples or {p: 1 for p in units if units[p]})
    summary = dict(pop_quota=dict(cost=cost)) if block else dict()
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


def test_price_is_the_pooled_aggregate_not_a_mean_of_per_run_prices(tmp_path):
    """Pooling must sum minutes and units and divide ONCE. A mean of the two runs' own
    ratios (10/1=10 and 2/4=0.5 -> 5.25) weights a run by how many windows it flushed
    rather than by how much work it did; the aggregate is 12/5 = 2.4."""
    a = _run(tmp_path, "a", min_spent={"mandelbrot": 10.0}, units={"mandelbrot": 1.0})
    b = _run(tmp_path, "b", min_spent={"mandelbrot": 2.0}, units={"mandelbrot": 4.0})
    t = dqp.derive([a, b])
    assert t["prices"]["mandelbrot"] == pytest.approx(2.4)
    assert t["_provenance"]["minutes"]["mandelbrot"] == pytest.approx(12.0)
    assert t["_provenance"]["units"]["mandelbrot"] == pytest.approx(5.0)


def test_a_zero_unit_partition_is_defaulted_and_stamped_not_dropped(tmp_path):
    """`min_spent/0` is no measurement. The row must survive at SEED_PRICE and be listed
    in `defaulted` — dropping it makes "never mined" indistinguishable from "never
    tracked" to the next reader."""
    d = _run(tmp_path, "a", min_spent={"mandelbrot": 6.0, "phoenix": 9.0},
             units={"mandelbrot": 2.0, "phoenix": 0.0})
    t = dqp.derive([d])
    assert t["prices"]["phoenix"] == pquota.SEED_PRICE
    assert t["_provenance"]["defaulted"] == ["phoenix"]
    assert t["_provenance"]["measured"] == ["mandelbrot"]
    # and the minutes it DID burn are still reported, or the zero reads as "never served"
    assert t["_provenance"]["minutes"]["phoenix"] == pytest.approx(9.0)
    # zero units is not "thin" — thin means served and productive but under the floor
    assert t["_provenance"]["thin"] == []


def test_a_table_with_no_measured_row_raises_instead_of_writing_the_seed(tmp_path):
    """Non-vacuity. A regenerated table that is byte-identical to the flat seed reports
    itself as a measurement and is not one."""
    d = _run(tmp_path, "a", min_spent={"mandelbrot": 6.0}, units={"mandelbrot": 0.0})
    with pytest.raises(dqp.NoTelemetryError, match="priceable"):
        dqp.derive([d])


def test_a_run_without_a_pop_quota_block_raises_naming_the_file(tmp_path):
    """Fail-closed on the allocator being off, rather than pooling an empty block into a
    table that then looks thinner than it is."""
    d = _run(tmp_path, "a", min_spent={}, units={}, block=False)
    with pytest.raises(dqp.NoTelemetryError, match="pop_quota"):
        dqp.derive([d])


def test_a_missing_summary_raises_and_refuses_state_json_as_a_substitute(tmp_path):
    d = tmp_path / "unfinished"
    d.mkdir()
    (d / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(dqp.NoTelemetryError, match="summary.json"):
        dqp.derive([d])


def test_a_thin_row_is_defaulted_on_ITS_DENOMINATOR_not_on_its_magnitude(tmp_path):
    """One class-3 (0.1 units) after twelve minutes aggregates to 120 min/unit. That is not
    a rate, and the reason is the denominator: allocation share is deficit/price, so seeding
    120 would allocate the partition out of existence on a single class-3. The floor is
    CLASS_WEIGHT[4] — one whole unit — and the discarded estimate stays visible in
    `price_raw`."""
    d = _run(tmp_path, "a", min_spent={"multibrot4": 12.0, "mandelbrot": 6.0},
             units={"multibrot4": 0.1, "mandelbrot": 2.0})
    t = dqp.derive([d])
    assert dqp.MIN_UNITS == pytest.approx(pquota.CLASS_WEIGHT[4])
    assert t["prices"]["multibrot4"] == pquota.SEED_PRICE
    assert t["_provenance"]["defaulted"] == ["multibrot4"]
    assert t["_provenance"]["thin"] == ["multibrot4"]
    assert t["_provenance"]["price_raw"]["multibrot4"] == pytest.approx(120.0)
    assert t["prices"]["mandelbrot"] == pytest.approx(3.0)


def test_a_well_evidenced_extreme_price_is_NOT_bounded(tmp_path):
    """The defect this replaced: a [seed/4, seed*4] magnitude band reported 0.75 for four
    partitions the first steady-state run measured at 0.078-0.139 min/unit off 62-148 units.
    A rate resting on a real denominator is the answer, not an outlier — in BOTH directions."""
    d = _run(tmp_path, "a",
             min_spent={"julia:multibrot3": 11.63, "multibrot5": 40.0},
             units={"julia:multibrot3": 148.2, "multibrot5": 1.0})
    t = dqp.derive([d])
    assert t["prices"]["julia:multibrot3"] == pytest.approx(0.078, abs=1e-3)   # far below 0.75
    assert t["prices"]["multibrot5"] == pytest.approx(40.0)                    # far above 12.0
    assert t["_provenance"]["defaulted"] == []


def test_the_written_table_is_read_by_the_real_consumer(tmp_path):
    """The round trip that matters: `CostToMine` — the thing `--quota-prices` feeds — must
    reproduce every derived price as its seed. A second parser here would assert nothing."""
    d = _run(tmp_path, "a", min_spent={"mandelbrot": 6.0, "multibrot3": 8.0},
             units={"mandelbrot": 2.0, "multibrot3": 1.0})
    t = dqp.derive([d])
    cfg = json.loads(json.dumps(t))          # prove it survives a JSON round trip
    cost = pquota.CostToMine(["mandelbrot", "multibrot3", "julia:mandelbrot"], cfg)
    assert cost.seed["mandelbrot"] == pytest.approx(3.0)
    assert cost.seed["multibrot3"] == pytest.approx(8.0)
    # a partition absent from the table falls to the config's declared seed_price
    assert cost.seed["julia:mandelbrot"] == pytest.approx(pquota.SEED_PRICE)
    assert cost.ema == pquota.PRICE_EMA and cost.clamp == pquota.PRICE_CLAMP
    assert cost.cap_minutes == pquota.CAP_MINUTES


def test_a_disposable_source_run_is_refused_by_class(tmp_path, monkeypatch):
    """Same rule as the harvest-log registry: a durable table must not be derived from a
    population a `rm -r scratch/*` can delete."""
    d = _run(tmp_path / "scratch", "a", min_spent={"mandelbrot": 6.0},
             units={"mandelbrot": 2.0})
    monkeypatch.setattr(dqp, "ROOT", tmp_path)
    with pytest.raises(dqp.SourceClassError, match="disposable"):
        dqp.derive([d])


def test_forced_partitions_appear_defaulted_rather_than_missing(tmp_path):
    """`--partitions` exists so a table can state a row for a partition this run never
    served. It must land defaulted, not absent — an absent row is a partition nobody
    classified."""
    d = _run(tmp_path, "a", min_spent={"mandelbrot": 6.0}, units={"mandelbrot": 2.0})
    t = dqp.derive([d], partitions=["mandelbrot", "phoenix:classic"])
    assert t["prices"]["phoenix:classic"] == pquota.SEED_PRICE
    assert "phoenix:classic" in t["_provenance"]["defaulted"]
