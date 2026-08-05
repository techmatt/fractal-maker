#!/usr/bin/env python
r"""regularize_quota_prices.py — the REGULARIZED cost-to-mine seed, derived from the measured
one by geometric shrinkage toward the table's median price.

WHY THE MEASURED TABLE IS NOT THE SEED (Matt's decision, 2026-08-05). The measured prices in
`data/atlas/quota_prices_v1.json` come from ONE 60-minute run in which every partition was
warming up at once (`discovery_pipeline.md` §3.3): every level is likely deflated, and there
is no reason to think it is deflated evenly. Allocation share is `deficit / price`, so seeding
the raw table hands the run's first pops a 32x spread (0.078 min/unit to 2.51) asserted on one
warm-up hour — the mirror of the flat 3.0 it replaces, which asserted a spread of 1x on
nothing. The decision is that allocation should be BIASED toward the measured prices without
being GOVERNED by them: cheap partitions keep earning flat-ish volume rather than being handed
the whole batch stream.

THE ESTIMATOR IS SHRINKAGE, NOT A BOUND, AND THAT IS THE POINT. `derive_quota_prices` refused
a magnitude band (`[seed/4, seed*4]`) because a clamp cannot tell "implausible" from "the
thing we ran the run to find out" — at the band edge it reports the BAND, discarding the
measurement entirely. Shrinkage never discards: every partition keeps its measured ORDER and
`ALPHA` of its log-distance from the median, so a 10x signal survives as a 5x one instead of
being flattened to the bound. The knob is the confidence in the population (one warm-up run),
not a plausibility opinion about any partition.

    seed_p = exp( ALPHA * ln(price_p) + (1 - ALPHA) * ln(median(measured prices)) )

Three properties worth stating because they are what makes this readable later: a partition AT
the median is unchanged; every log-ratio between two partitions is multiplied by exactly
`ALPHA`; and so the whole table's spread goes from `S` to `S**ALPHA` (32.2x -> 11.4x at
ALPHA=0.7). Geometric and not arithmetic because a price is a RATE — the meaningful distance
between 0.1 and 1.0 min/unit is the same as between 1.0 and 10.0.

SEED ONLY. This touches `prices`, which `pop_quota.CostToMine` reads exactly once, into
`self.seed`. The in-run batch-aggregated pricing (`end_window` -> the EMA in `self.raw`) is
untouched and converges to whatever the run actually measures within a few windows; the seed
orders the early batches and sets the clamp band around itself. Regularizing a live EMA would
be a different and much worse thing — a run that cannot learn its own costs.

DEFAULTED ROWS ARE NOT SHRUNK AND DO NOT SET THE MEDIAN. A defaulted row carries `SEED_PRICE`
because it has no measurement (`derive_quota_prices`, MIN_UNITS); shrinking it toward the
median would manufacture a price for a partition nobody priced, and letting it into the median
would drag the shrink target toward the flat seed by however many partitions went unmined. It
passes through at `SEED_PRICE` and stays stamped.

The measured table is READ-ONLY here and is never rewritten: it is the evidence, this is a
policy applied on top of it, and a policy that edits its own evidence cannot be re-derived at
a different ALPHA.

    uv run python tools/atlas/regularize_quota_prices.py            # print the two columns
    uv run python tools/atlas/regularize_quota_prices.py --write    # + write the artifact

Consumed by `steered_frontier.py --quota-prices`, which DEFAULTS to the artifact this writes
(`steered_frontier.QUOTA_PRICES_DEFAULT`) and fails loudly when it is absent rather than
falling back to the flat seed. `test_regularize_quota_prices.py` round-trips the output
through `pop_quota.CostToMine` — the real consumer — rather than through a second parser.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pop_quota as pquota                              # noqa: E402
from tools import paths as _paths                       # noqa: E402

SCHEMA = "pop_quota_cost_to_mine/1"                     # same schema as the measured table:
#: the CONSUMER contract is identical, only the derivation of `prices` differs.

DEFAULT_SOURCE = "data/atlas/quota_prices_v1.json"
DEFAULT_OUT = "data/atlas/quota_prices_regularized_v1.json"

# The shrinkage weight on the MEASURED price, in log space. Matt, 2026-08-05. 1.0 would be the
# raw measured table (fully governed by one warm-up hour); 0.0 would be a flat table at the
# median (the measurement discarded). 0.7 is the stated intent — "biased toward the prices but
# not fully governed by a 32x spread" — and it lands the table's spread at 32.2**0.7 = 11.4x.
# A NAMED CONSTANT and recorded in the artifact, so a later reader can re-derive at a different
# alpha instead of guessing which one produced the file in front of them.
ALPHA = 0.7

# The shrink target. `median` and not `mean`: the target is a location for the log-price
# distribution and the measured table is 9 rows with a 32x range, where one partition priced
# off a single sample (multibrot5, 1.0 units) would move a mean and cannot move a median.
SHRINK_TARGET = "median"

FORMULA = ("seed_p = exp(ALPHA * ln(price_p) + (1 - ALPHA) * ln(median(measured prices)))")


class SourceTableError(RuntimeError):
    """The measured table is missing, is not a cost-to-mine table, or prices nothing.

    Fail-closed in every case: a regularized table built from an empty or foreign source
    would be a policy applied to nothing while reading as a derived seed
    (`verification_practice.md` §2)."""


def load_measured(path: Path) -> dict:
    """The measured table, proved to BE one before anything is derived from it."""
    p = Path(path)
    if not p.exists():
        raise SourceTableError(
            f"{p} missing — there is no measured cost-to-mine table to regularize. "
            f"Regenerate it from a finished run first:\n"
            f"    uv run python tools/atlas/derive_quota_prices.py --run-dir <run>")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("_schema") != SCHEMA:
        raise SourceTableError(
            f"{p} carries _schema={doc.get('_schema')!r}, not {SCHEMA!r}. "
            f"`data/atlas/scheduler_prices.json` is the DEFICIT SCHEDULER's table and is "
            f"denominated per DISTINCT LOOK — a different quantity, not a source for this.")
    if not (doc.get("prices") or {}):
        raise SourceTableError(f"{p} has an empty `prices` block; nothing to regularize.")
    return doc


def shrink(price: float, target: float, alpha: float = ALPHA) -> float:
    """One price, shrunk geometrically toward `target`. THE formula, in one place."""
    return math.exp(alpha * math.log(price) + (1.0 - alpha) * math.log(target))


def spread(prices) -> float:
    """max/min over a set of prices — the quantity ALPHA is chosen against."""
    vals = [float(v) for v in prices]
    lo = min(vals)
    return (max(vals) / lo) if lo > 0 else float("inf")


def regularize(doc: dict, *, alpha: float = ALPHA, source: str | None = None) -> dict:
    """The regularized table + the provenance that lets a reader re-derive it or read it
    beside the measurement it came from.

    The MEASURED rows set the shrink target and are the only rows shrunk; defaulted rows pass
    through at `SEED_PRICE` (see the module docstring). Everything `CostToMine` reads other
    than `prices` is carried over from the source table verbatim — this is a re-derivation of
    the seed, not a re-decision of the EMA, the clamp or the cap."""
    prices = {p: float(v) for p, v in (doc.get("prices") or {}).items()}
    prov = doc.get("_provenance") or {}
    defaulted = set(prov.get("defaulted") or [])
    measured = [p for p in prices if p not in defaulted]
    if not measured:
        raise SourceTableError(
            "every row in the source table is `defaulted` (no partition cleared the evidence "
            "floor), so there is no measured population to shrink toward and no measurement "
            "to shrink. Regularizing it would produce a flat table wearing a derived name.")

    target = float(statistics.median(prices[p] for p in measured))
    out_prices, cols = {}, {}
    for p in sorted(prices):
        if p in defaulted:
            out_prices[p] = prices[p]      # SEED_PRICE, untouched — no measurement to shrink
        else:
            out_prices[p] = round(shrink(prices[p], target, alpha), 4)
        cols[p] = dict(measured=prices[p], regularized=out_prices[p],
                       status="defaulted" if p in defaulted else "measured")

    return dict(
        _schema=SCHEMA,
        _doc=("pop-quota cost-to-mine SEED prices, REGULARIZED: the measured table shrunk "
              "geometrically toward its own median price at alpha=%.2f. Consumed by "
              "`steered_frontier.py --quota-prices`, which DEFAULTS to this file. Regenerate "
              "with tools/atlas/regularize_quota_prices.py — never hand-edit a row, and "
              "never edit the measured table this is derived from." % alpha),
        prices=out_prices,
        # Carried over verbatim: this re-derives the SEED only. The EMA rate, the clamp band
        # and the dry-time cap are the run's own mechanism and are not a shrinkage decision.
        seed_price=float(doc.get("seed_price", pquota.SEED_PRICE)),
        price_ema=float(doc.get("price_ema", pquota.PRICE_EMA)),
        price_clamp=float(doc.get("price_clamp", pquota.PRICE_CLAMP)),
        cap_minutes=float(doc.get("cap_minutes", pquota.CAP_MINUTES)),
        _provenance=dict(
            estimand="the measured seed, shrunk toward the measured median in LOG space",
            formula=FORMULA,
            alpha=alpha,
            alpha_basis=("Matt, 2026-08-05: allocation biased toward the measured prices but "
                         "not governed by a 32x spread read off a single 60-min all-warm-up "
                         "run. Every log-ratio is multiplied by alpha, so the table's spread "
                         "goes from S to S**alpha."),
            shrink_target=SHRINK_TARGET,
            shrink_target_value=round(target, 6),
            shrink_target_population=("MEASURED rows only; a defaulted row carries SEED_PRICE "
                                      "because it has no measurement, and letting it into the "
                                      "median would drag the target toward the flat seed"),
            source_table=str(source) if source else DEFAULT_SOURCE,
            source_schema=doc.get("_schema"),
            source_estimand=prov.get("estimand"),
            source_runs=prov.get("source_runs"),
            source_measured=prov.get("measured"),
            source_defaulted=prov.get("defaulted"),
            source_is_pristine=("the measured table is read-only to this tool and is never "
                                "rewritten: it is the evidence, this is a policy on top of "
                                "it, and a policy that edits its evidence cannot be "
                                "re-derived at another alpha"),
            applies_to="SEED ONLY — pop_quota.CostToMine reads `prices` once into self.seed; "
                       "the in-run batch-aggregated EMA (end_window -> self.raw) is untouched",
            columns=cols,
            spread_measured=round(spread([prices[p] for p in measured]), 3),
            spread_regularized=round(spread([out_prices[p] for p in measured]), 3),
        ),
    )


def format_table(table: dict) -> str:
    """The two columns side by side — the report the decision is read off."""
    prov = table["_provenance"]
    cols = prov["columns"]
    w = max(len(p) for p in cols)
    lines = [
        f"cost-to-mine SEED, measured -> regularized   "
        f"(alpha={prov['alpha']}, {prov['shrink_target']}={prov['shrink_target_value']:.4g})",
        f"  {'partition':{w}s} {'measured':>10} {'regularized':>12} {'x':>7}  status",
    ]
    for p in sorted(cols, key=lambda k: cols[k]["measured"]):
        c = cols[p]
        ratio = c["regularized"] / c["measured"] if c["measured"] else float("nan")
        lines.append(f"  {p:{w}s} {c['measured']:>10.3f} {c['regularized']:>12.4f} "
                     f"{ratio:>7.2f}  {c['status']}")
    lines.append(f"  {'spread (max/min)':{w}s} {prov['spread_measured']:>10.1f}x "
                 f"{prov['spread_regularized']:>11.1f}x")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="the MEASURED table to regularize (default %(default)s); read-only")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="shrinkage weight on the measured price in log space "
                         "(default %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true",
                    help="write the artifact; without it this prints the two columns only")
    a = ap.parse_args()

    src = Path(a.source)
    if not src.is_absolute():
        src = ROOT / src
    table = regularize(load_measured(src), alpha=a.alpha, source=a.source)
    print(format_table(table))
    if not a.write:
        print("\n(dry run — pass --write to write the artifact)", file=sys.stderr)
        return
    out = _paths.durable(a.out, mkparents=True)
    out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
