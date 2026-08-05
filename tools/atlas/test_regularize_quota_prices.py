#!/usr/bin/env python
"""Tests for `regularize_quota_prices.py` — the median-shrunk cost-to-mine SEED — and for the
`--quota-prices` default/absence contract in `steered_frontier`.

Each test names the defect it would catch. Two are load-bearing: the round trip goes through
`pop_quota.CostToMine` (the real consumer, not a second parser written here —
`verification_practice.md` §1.10), and the absence test asserts the loud failure, which is the
behaviour that replaced a silent fall-back to the flat seed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pop_quota as pquota                 # noqa: E402
import regularize_quota_prices as rq       # noqa: E402

MEASURED = ROOT / "data" / "atlas" / "quota_prices_v1.json"
REGULARIZED = ROOT / "data" / "atlas" / rq.DEFAULT_OUT.split("/")[-1]


def _table(prices, *, defaulted=(), **kw):
    """A minimal source table in the shape `derive_quota_prices` writes."""
    return dict(_schema=rq.SCHEMA, prices=dict(prices),
                seed_price=pquota.SEED_PRICE, price_ema=pquota.PRICE_EMA,
                price_clamp=pquota.PRICE_CLAMP, cap_minutes=pquota.CAP_MINUTES,
                _provenance=dict(estimand="test", defaulted=list(defaulted),
                                 measured=[p for p in prices if p not in defaulted], **kw))


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def test_a_price_at_the_median_is_unchanged():
    """The fixed point of the shrink. If this moves, the target is not the median and every
    other row's regularized price is shrunk toward something nobody chose."""
    t = rq.regularize(_table({"a": 0.1, "b": 1.0, "c": 10.0}))
    assert t["_provenance"]["shrink_target_value"] == pytest.approx(1.0)
    assert t["prices"]["b"] == pytest.approx(1.0)


def test_every_log_ratio_is_multiplied_by_alpha():
    """The property that makes this shrinkage rather than a bound: ORDER is preserved and each
    partition keeps exactly alpha of its log-distance from the median. A clamp would report
    the bound for both extremes and destroy the ratio entirely."""
    t = rq.regularize(_table({"a": 0.1, "b": 1.0, "c": 10.0}), alpha=0.7)
    lo, mid, hi = t["prices"]["a"], t["prices"]["b"], t["prices"]["c"]
    assert lo < mid < hi
    # rel 1e-3 and not tighter: the written prices are rounded to 4 dp for readability, which
    # is a ~1e-4 relative perturbation on the small end. The property under test is the
    # exponent, not the fourth decimal.
    assert math.log(hi / mid) == pytest.approx(0.7 * math.log(10.0), rel=1e-3)
    assert math.log(mid / lo) == pytest.approx(0.7 * math.log(10.0), rel=1e-3)


def test_the_spread_contracts_as_S_to_the_alpha():
    """The number the alpha choice was made against: 32.2x measured -> 11.4x at alpha=0.7."""
    t = rq.regularize(_table({"a": 0.078, "b": 0.139, "c": 2.51}), alpha=0.7)
    prov = t["_provenance"]
    assert prov["spread_regularized"] == pytest.approx(prov["spread_measured"] ** 0.7, rel=1e-3)


def test_alpha_one_is_the_measured_table_and_alpha_zero_is_flat():
    """The two endpoints, so the knob's meaning is pinned rather than implied by a docstring."""
    src = _table({"a": 0.1, "b": 1.0, "c": 10.0})
    assert rq.regularize(src, alpha=1.0)["prices"] == pytest.approx({"a": 0.1, "b": 1.0,
                                                                    "c": 10.0})
    flat = rq.regularize(src, alpha=0.0)["prices"]
    assert flat == pytest.approx({"a": 1.0, "b": 1.0, "c": 1.0})


def test_a_defaulted_row_is_neither_shrunk_nor_allowed_into_the_median():
    """A defaulted row carries SEED_PRICE because nobody priced it. Shrinking it would
    manufacture a price for an unmeasured partition; letting it set the target would drag the
    median toward the flat seed by however many partitions went unmined — here the measured
    median is 1.0, not 3.0."""
    t = rq.regularize(_table({"a": 0.1, "b": 1.0, "c": 10.0, "d": pquota.SEED_PRICE},
                             defaulted=["d"]))
    assert t["_provenance"]["shrink_target_value"] == pytest.approx(1.0)
    assert t["prices"]["d"] == pytest.approx(pquota.SEED_PRICE)
    assert t["_provenance"]["columns"]["d"]["status"] == "defaulted"


def test_an_all_defaulted_source_is_refused():
    """A table with no measured row has nothing to shrink and nothing to shrink toward. The
    output would be the flat seed wearing a derived name — the same failure
    `derive_quota_prices.NoTelemetryError` exists to prevent, one layer down."""
    with pytest.raises(rq.SourceTableError, match="defaulted"):
        rq.regularize(_table({"a": pquota.SEED_PRICE}, defaulted=["a"]))


def test_a_foreign_price_table_is_refused_by_schema(tmp_path):
    """`data/atlas/scheduler_prices.json` is the DEFICIT SCHEDULER's table, denominated per
    DISTINCT LOOK. It parses as JSON with a `prices` block and would regularize silently into
    a cost-to-mine seed of a different quantity."""
    p = tmp_path / "scheduler_prices.json"
    p.write_text(json.dumps(dict(_schema="deficit_scheduler_prices/1",
                                 prices={"a": 1.0})), encoding="utf-8")
    with pytest.raises(rq.SourceTableError, match="_schema"):
        rq.load_measured(p)


def test_the_ema_clamp_and_cap_are_carried_over_untouched():
    """SEED ONLY. Regularizing the live EMA or the clamp band would be a run that cannot learn
    its own costs; only `prices` may differ from the source table."""
    src = _table({"a": 0.1, "b": 1.0}, )
    src.update(price_ema=0.42, price_clamp=7.0, cap_minutes=13.0, seed_price=5.0)
    t = rq.regularize(src)
    assert (t["price_ema"], t["price_clamp"], t["cap_minutes"], t["seed_price"]) == \
        (0.42, 7.0, 13.0, 5.0)


# --------------------------------------------------------------------------- #
# provenance + the committed artifact
# --------------------------------------------------------------------------- #
def test_alpha_and_the_formula_are_recorded_in_the_artifact():
    """A file of nine floats cannot say which policy produced it. Without the constants and
    the source table in the record, the artifact cannot be re-derived at another alpha or told
    apart from the measured table it came from."""
    t = rq.regularize(_table({"a": 0.1, "b": 1.0}), source="data/atlas/quota_prices_v1.json")
    prov = t["_provenance"]
    assert prov["alpha"] == rq.ALPHA and prov["shrink_target"] == rq.SHRINK_TARGET
    assert prov["formula"] == rq.FORMULA
    assert prov["source_table"] == "data/atlas/quota_prices_v1.json"
    assert "SEED ONLY" in prov["applies_to"]


def test_regularizing_never_writes_the_measured_table(tmp_path, monkeypatch):
    """The measured table is the EVIDENCE. A policy that edits its own evidence cannot be
    re-derived at a different alpha, and the next `derive_quota_prices` run would pool a
    regularized number back into an aggregate."""
    before = MEASURED.read_bytes()
    out = tmp_path / "reg.json"
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--write"])
    monkeypatch.setattr(rq._paths, "durable", lambda p, mkparents=False: Path(p))
    rq.main()
    assert MEASURED.read_bytes() == before
    assert json.loads(out.read_text(encoding="utf-8"))["prices"]


def test_the_committed_artifact_matches_a_rederivation_from_the_measured_table():
    """The drift gate. The committed seed must be exactly what the tool produces from the
    committed measured table at the committed ALPHA — otherwise a hand-edited row, or a stale
    artifact left behind by a re-derived measurement, seeds production."""
    live = json.loads(REGULARIZED.read_text(encoding="utf-8"))
    fresh = rq.regularize(rq.load_measured(MEASURED))
    assert live["prices"] == fresh["prices"]
    assert live["_provenance"]["alpha"] == rq.ALPHA


def test_the_committed_artifact_loads_into_the_real_consumer():
    """The round trip that matters: the file `--quota-prices` defaults to must seed
    `CostToMine` for every registered partition, with the EMA/clamp/cap it declares."""
    import steered_frontier as sf
    cfg = sf.load_quota_prices(None)
    parts = list(cfg["prices"]) + ["phoenix:classic"]
    cost = pquota.CostToMine(parts, cfg)
    for p, v in cfg["prices"].items():
        assert cost.seed[p] == pytest.approx(v)
    # a partition with no row falls to the declared seed_price, not to a shrunk guess
    assert cost.seed["phoenix:classic"] == pytest.approx(cfg["seed_price"])
    assert cost.ema == cfg["price_ema"] and cost.clamp == cfg["price_clamp"]


# --------------------------------------------------------------------------- #
# the `--quota-prices` default + absence contract in steered_frontier
# --------------------------------------------------------------------------- #
def test_the_quota_prices_default_is_the_regularized_artifact():
    import steered_frontier as sf
    assert sf.QUOTA_PRICES_DEFAULT == REGULARIZED
    assert sf.load_quota_prices(None)["prices"] == \
        json.loads(REGULARIZED.read_text(encoding="utf-8"))["prices"]


def test_a_missing_price_table_is_fatal_and_never_the_flat_seed(tmp_path):
    """THE defect this replaced: `if path and path.exists()` sent both "flag not passed" and
    "file moved" to a silent flat SEED_PRICE — a different allocation policy, asserting a 1x
    price spread over prices measured at 32x, with nothing in the run record to say so."""
    import steered_frontier as sf
    with pytest.raises(SystemExit) as e:
        sf.load_quota_prices(tmp_path / "nope.json")
    msg = str(e.value)
    assert "nope.json" in msg and "regularize_quota_prices" in msg
    assert str(pquota.SEED_PRICE) in msg          # names what it is REFUSING to fall back to


def test_an_absent_default_artifact_is_also_fatal(tmp_path, monkeypatch):
    """The default is not exempt from the absence rule — a deleted or unfetched LFS artifact
    must stop the run, not quietly re-flatten the seed."""
    import steered_frontier as sf
    monkeypatch.setattr(sf, "QUOTA_PRICES_DEFAULT", tmp_path / "gone.json")
    with pytest.raises(SystemExit, match="gone.json"):
        sf.load_quota_prices(None)


def test_the_constructor_loads_prices_through_the_loud_loader():
    """The tests above rehearse the loader; this asserts the pop-quota branch actually goes
    through it, so the rehearsal cannot pass against a constructor that kept the old
    exists()-guarded read."""
    import inspect
    import steered_frontier as sf
    src = inspect.getsource(sf.SteeredFrontier.__init__)
    assert "load_quota_prices(getattr(args, \"quota_prices\", None))" in src
    assert "Path(pp).exists()" not in src
