#!/usr/bin/env python
r"""derive_quota_prices.py — regenerate the pop-quota COST-TO-MINE seed table from run
telemetry, so the next run starts from what the last one measured instead of a flat guess.

WHY THIS EXISTS. `pop_quota.CostToMine` seeds every partition at `SEED_PRICE` (3.0
active-minutes per currency unit) unless a `--quota-prices` file says otherwise, and no such
file has ever existed — `data/atlas/scheduler_prices.json` is the DEFICIT SCHEDULER's table
and is denominated per DISTINCT LOOK, a different quantity. So the stored cost-to-mine table
is the flat seed, and a flat seed makes the first pop of every run allocate on deficit alone.
`harvest_v2_readout.cost_to_mine` has been reporting "the prices this run measured, for the
next one" into a file nothing reads.

THE ESTIMAND IS THE AGGREGATE, NOT THE EMA. `price_raw` is a recency-weighted estimate built
per served batch (a ratio of two EMAs, minutes over units — `pop_quota.CostToMine`);
`min_spent / units_mined` is the thing it estimates, over the whole run. Pooling several runs
therefore sums MINUTES and UNITS and divides once — averaging their EMAs would weight a run
by how many windows it happened to flush rather than by how much work it did. This is the
same read that made the v1 sampler's inversion visible
(`allocator_prereg_v1_mechanism_read_20260804.md` §4), and it is why the aggregate is the
quantity carried forward.

A PARTITION WITH ZERO UNITS IS NOT PRICED. It keeps `SEED_PRICE` and is stamped
`defaulted`. `min_spent/0` is not a large price, it is no measurement; and writing a
partition's row out of the file entirely would make "we never mined it" indistinguishable
from "we never tracked it" to the next reader.

THE GUARD IS ON THE DENOMINATOR, NOT ON THE MAGNITUDE. Allocation share is deficit/price, so
a partition that mined 0.1 units in twelve minutes would carry a seed of 120 into the next
run and allocate itself out of existence on one class-3. The first version of this module
bounded the derived seed into `[SEED_PRICE/PRICE_CLAMP, SEED_PRICE*PRICE_CLAMP]` = [0.75,
12.0] — and that was the wrong instrument, as the first steady-state run showed immediately:
four partitions measured 0.078-0.139 min/unit off 62-148 units and 8-17 windows, and the
clamp would have thrown a well-evidenced 10x signal away to report 0.75. A magnitude bound
cannot tell "implausible" from "the thing we ran the run to find out".

So the gate is `units >= CLASS_WEIGHT[4]` — one full class-4's worth of currency, the
existing owner of "one unit". At or above it, the aggregate stands as measured, however
large or small; below it, the row rests on fractions of class-3s, is not a measurement, and
is defaulted to `SEED_PRICE` and stamped. `CostToMine.price` still bounds the LIVE EMA to a
factor around whatever seed it is handed, which is the bound that belongs to the run.

    uv run python tools/atlas/derive_quota_prices.py \
        --run-dir data/discovery/steady_state_v2_20260807 \
        --out data/atlas/quota_prices_20260807.json

EACH REGENERATION TAKES A NEW DATED `--out`. A measured table is the EVIDENCE a seed was
derived from, and every earlier one stays as a record — `quota_prices_v1.json`
(steady_state_v1_20260805) is not superseded by `quota_prices_20260807.json`, it is the other
run. `DEFAULT_OUT` therefore tracks the newest table rather than a stable name, and pointing a
regeneration at an existing file overwrites a record.

Consumed by `steered_frontier.py --quota-prices <path>`; every key in the file is one
`CostToMine.__init__` reads (`prices`, `seed_price`, `price_ema`, `price_clamp`,
`cap_minutes`), and `test_derive_quota_prices.py` round-trips it through that constructor
rather than re-deriving the expectation here. The constructor also accepts
`price_seed_units` / `price_min_units`; those shape the estimator's own accumulators rather
than the seed table, so they are left at their module defaults and are not written here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pop_quota as pquota                              # noqa: E402
from tools import paths as _paths                       # noqa: E402
from tools.corpus import artifacts as _artifacts        # noqa: E402

DEFAULT_OUT = "data/atlas/quota_prices_20260807.json"     # the NEWEST measured table; a new
#: derivation takes a new dated name (see the module docstring) — this is not a stable path.
SCHEMA = "pop_quota_cost_to_mine/1"

# The evidence floor a row must clear to be PRICED rather than defaulted: one full class-4's
# worth of currency. Not a new number — `CLASS_WEIGHT[4]` is already the tree's definition of
# one unit, and "at least one whole unit in the denominator" is the weakest statement that
# distinguishes a rate from a fraction of a single class-3.
MIN_UNITS = pquota.CLASS_WEIGHT[4]


class NoTelemetryError(RuntimeError):
    """A source run carries no `pop_quota` cost block, or no partition in the pooled set
    mined any currency at all.

    Fail-closed rather than fall back to the seed table: a "regenerated" table that is
    byte-identical to the flat seed reports itself as a measurement and is not one, and the
    caller cannot tell the two apart afterwards (`verification_practice.md` §2)."""


class SourceClassError(RuntimeError):
    """A source run dir resolves under a disposable tree. Same rule and same owner as the
    harvest-log registry: a durable table derived from a population a `rm -r scratch/*` can
    delete is a number that outlives its own evidence."""


def _refuse_disposable(run_dir: Path) -> Path:
    hit = _paths.disposable_component(run_dir, (ROOT, _artifacts.artifacts_root()))
    if hit is None:
        return run_dir
    raise SourceClassError(
        f"price source run resolves under the disposable `{hit}/` class:\n    {run_dir}\n"
        f"The derived table is committed under `data/` and would outlive the telemetry it "
        f"was derived from. Use a run under a registered discovery store.")


def cost_block(run_dir: Path) -> dict:
    """The `pop_quota.cost` block of a finished run, or raise naming the file.

    `summary.json` and not `state.json`: the summary is written by `finish()`, so its
    presence is what says the run reached an end. A killed run's checkpoint holds the same
    counters mid-flight and would price a partial population as a whole one."""
    run_dir = _refuse_disposable(Path(run_dir).resolve())
    sp = run_dir / "summary.json"
    if not sp.exists():
        raise NoTelemetryError(
            f"{sp} missing — the run has not finished (or was killed before `finish()`). "
            f"state.json is not a substitute: it holds the same counters mid-flight and "
            f"would price a partial population as a whole one.")
    s = json.loads(sp.read_text(encoding="utf-8"))
    q = s.get("pop_quota")
    if not q or not q.get("cost"):
        raise NoTelemetryError(
            f"{sp} has no `pop_quota.cost` block — the allocator was off for this run, so it "
            f"measured no cost-to-mine. Nothing to regenerate a price table from.")
    return q["cost"]


def pool(blocks: list[dict]) -> tuple[dict, dict]:
    """`(minutes, units)` summed per partition across every source run.

    Sums the two accumulators and divides ONCE downstream. Averaging the runs' `price_raw`
    EMAs instead would weight a run by how many windows it flushed rather than by how much
    work it did."""
    minutes: dict = {}
    units: dict = {}
    for c in blocks:
        for p, v in (c.get("min_spent") or {}).items():
            minutes[p] = minutes.get(p, 0.0) + float(v)
        for p, v in (c.get("units_mined") or {}).items():
            units[p] = units.get(p, 0.0) + float(v)
    for p in set(minutes) | set(units):
        minutes.setdefault(p, 0.0)
        units.setdefault(p, 0.0)
    return minutes, units


def derive(run_dirs: list[Path], *, partitions: list[str] | None = None) -> dict:
    """The regenerated table + the provenance that lets a later reader tell a measured row
    from a defaulted one without re-running this."""
    runs = [Path(r) for r in run_dirs]
    blocks = [cost_block(r) for r in runs]
    minutes, units = pool(blocks)
    parts = sorted(set(partitions) | set(minutes)) if partitions else sorted(minutes)

    prices, raw, defaulted, thin = {}, {}, [], []
    for p in parts:
        u, m = units.get(p, 0.0), minutes.get(p, 0.0)
        if u > 0:
            raw[p] = round(m / u, 3)
        if u < MIN_UNITS:
            prices[p] = pquota.SEED_PRICE
            defaulted.append(p)
            if u > 0:
                thin.append(p)
            continue
        prices[p] = round(m / u, 3)

    measured = [p for p in parts if p not in defaulted]
    if not measured:
        raise NoTelemetryError(
            f"no partition in {[r.name for r in runs]} is priceable — none reached "
            f"MIN_UNITS={MIN_UNITS} currency units, so every row would be the flat seed, and "
            f"a table that is byte-identical to the seed reports itself as a measurement it "
            f"is not. Run longer, or do not regenerate.")

    return dict(
        _schema=SCHEMA,
        _doc=("pop-quota cost-to-mine SEED prices: active-minutes per unit of Matt's "
              "currency (a decoded 4 = 1.0, a decoded 3 = 0.1) mined as a DISTINCT "
              "ADMISSION. Consumed by `steered_frontier.py --quota-prices`; the online EMA "
              "moves from here within a few windows, so this only orders the early batches. "
              "Regenerate with tools/atlas/derive_quota_prices.py — never hand-edit a row."),
        prices=prices,
        seed_price=pquota.SEED_PRICE,
        price_ema=pquota.PRICE_EMA,
        price_clamp=pquota.PRICE_CLAMP,
        cap_minutes=pquota.CAP_MINUTES,
        _provenance=dict(
            estimand="sum(min_spent) / sum(units_mined), pooled across the source runs",
            source_runs=[dict(name=r.name, path=str(r)) for r in runs],
            minutes={p: round(minutes.get(p, 0.0), 2) for p in parts},
            units={p: round(units.get(p, 0.0), 3) for p in parts},
            samples={p: sum(int((c.get("price_samples") or {}).get(p, 0)) for c in blocks)
                     for p in parts},
            price_raw=raw,
            measured=measured,
            # A defaulted row is NOT a measurement, and the next reader must be able to see
            # which rows are which without re-deriving the table.
            defaulted=defaulted,
            defaulted_note=(f"fewer than MIN_UNITS={MIN_UNITS} currency units mined in the "
                            f"source runs; the row carries SEED_PRICE and states nothing "
                            f"about the partition"),
            # A row with SOME currency but under the floor is a different fact from a row
            # with none: it was served and produced, just not enough to denominate a rate.
            thin=thin,
            min_units=MIN_UNITS,
            min_units_note=("one full class-4's worth of currency (pop_quota.CLASS_WEIGHT[4]) "
                            "is the evidence floor to be PRICED. There is deliberately no "
                            "magnitude bound: the first steady-state run measured four "
                            "partitions at 0.078-0.139 min/unit off 62-148 units, and a "
                            "[seed/4, seed*4] band would have reported 0.75 for all of them"),
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True,
                    help="a FINISHED run dir with a pop_quota cost block (repeatable; "
                         "minutes and units pool across all of them)")
    ap.add_argument("--partitions", default=None,
                    help="comma list to force into the table even if a source run never "
                         "tracked them (they land defaulted); default: whatever the runs saw")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    parts = [p for p in (a.partitions or "").split(",") if p] or None
    table = derive([Path(r) for r in a.run_dir], partitions=parts)
    out = _paths.durable(a.out, mkparents=True)
    out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))
    prov = table["_provenance"]
    print(f"\n-> {out}\n   measured {len(prov['measured'])} / defaulted "
          f"{len(prov['defaulted'])} (thin {len(prov['thin'])})", file=sys.stderr)


if __name__ == "__main__":
    main()
