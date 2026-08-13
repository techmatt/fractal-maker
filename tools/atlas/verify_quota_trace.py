#!/usr/bin/env python
"""verify_quota_trace.py — read a run's quota trace against the REBASED CostToMine (efdb5dd).

Written for the run-27 shakedown's leg A2; promoted here because it is the standing price-
health readout every pop-quota run now asks for, and because a reader that lives only in a
`scratch/` tree is one `rm -r scratch/*` from not existing (`CLAUDE.md`, "neither scratch tree
is a dependency tier").

Five explicit asks, each with its population printed, verdict to stdout + a JSON appendix:

  1. EVERY SERVED BATCH PRICES, ZERO-YIELD INCLUDED. The trace's `price` map is written at
     PICK time, so batch N's row shows the price table AFTER batch N-1's window closed. The
     served partition of batch N-1 must therefore be exactly the partition whose price moved
     between rows N-1 and N. That is the fix's whole claim, and it is checkable from the trace
     alone with no access to the estimator.
  2. EVERY PRICE FINITE AND INSIDE ITS CLAMP BAND [seed/clamp, seed*clamp].
  3. sample_weight ON THIN BATCHES — the pure function is asserted directly (monotone in
     units, in [0,1), exactly `ema` when the window carries as much currency as the
     accumulator holds), and the run's own thin/thick windows are counted.
  4. FLOOR AND STARVATION, reported SEPARATELY and with the small-sample caveat attached.
  5. EARLY vs LATE SERVICE, AND THE PRICE PATH THAT EXPLAINS IT. Added for run 27. Run 26
     ended with healthy-looking totals for `mandelbrot` (6.2% of pop time, 96 admissions) and
     had in fact STOPPED SERVING IT at the midpoint — 11.9% of early admissions, 0.0% of late.
     A run total cannot show that, so service is split at the batch midpoint here and joined
     to the partition's own price path, which is the mechanism that would cause it: a price
     that runs up early and never comes back is a partition the gap rule puts last forever.
     `service_collapsed` is the run-26 condition stated as a predicate, so the counterfactual
     is checked rather than eyeballed.

Reads a run dir, writes one JSON, changes nothing.

  uv run python tools/atlas/verify_quota_trace.py data/discovery/prod27_20260812 \
      scratch/production_run27/quota_verdict.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "atlas")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import run_record                        # noqa: E402
import pop_quota as pq                              # noqa: E402

EPS = 5e-4          # the trace rounds price to 3 dp; a move must clear the rounding


def early_late(rows, split):
    """Ask 5. Service counts and the price path, per partition, split at `split`.

    Service is counted in POPS (batches served), not admissions: "does the allocator still
    CHOOSE this partition" is the question, and admissions confound it with yield. The
    admissions view is `tools/atlas/quota_read.py`'s, and the two are meant to be read
    together — run 26's mandelbrot collapsed on both."""
    parts = sorted({r["chosen"] for r in rows if r.get("chosen")})
    early = [r for r in rows if r["batch"] <= split]
    late = [r for r in rows if r["batch"] > split]
    n_e = sum(1 for r in early if r.get("chosen"))
    n_l = sum(1 for r in late if r.get("chosen"))
    at_split = late[0] if late else (rows[-1] if rows else {})
    first, last = (rows[0] if rows else {}), (rows[-1] if rows else {})
    out = {}
    for p in parts:
        pe = sum(1 for r in early if r.get("chosen") == p)
        pl = sum(1 for r in late if r.get("chosen") == p)
        pr0 = float((first.get("price") or {}).get(p, float("nan")))
        prs = float((at_split.get("price") or {}).get(p, float("nan")))
        prz = float((last.get("price") or {}).get(p, float("nan")))
        out[p] = {
            "pops_early": pe, "pops_late": pl,
            "service_share_early": round(pe / n_e, 4) if n_e else None,
            "service_share_late": round(pl / n_l, 4) if n_l else None,
            "price_first": round(pr0, 4), "price_at_split": round(prs, 4),
            "price_final": round(prz, 4),
            "price_ratio_final_over_first": round(prz / pr0, 3) if pr0 else None,
            # The run-26 condition, as a predicate: served before the midpoint, never after.
            "service_collapsed": pe > 0 and pl == 0,
            "never_served": pe == 0 and pl == 0,
        }
    return {
        "split_batch": split, "pops_early": n_e, "pops_late": n_l,
        "per_partition": out,
        "service_collapsed": sorted(p for p, v in out.items() if v["service_collapsed"]),
        "service_started_late": sorted(p for p, v in out.items()
                                       if v["pops_early"] == 0 and v["pops_late"] > 0),
    }


def main():
    run_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir.parent / "quota_verdict.json"
    rows = run_record.require_rows(run_dir / "quota_trace.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    quota = summary.get("pop_quota", {})
    cost = quota.get("cost", {})
    parts = sorted(rows[0]["price"]) if rows else []
    # The estimator's own parameters, READ off the run rather than restated: the seed table is
    # what `regularize_quota_prices` deployed, `clamp_factor` is what CostToMine reported, and
    # `price_ema` is the one the DEPLOYED TABLE pins (which is the thing 007b2c7 changed —
    # the code constant and the table's pin are two different numbers and must be read apart).
    rc_path = run_dir / "run_config.json"
    rc = json.loads(rc_path.read_text(encoding="utf-8")) if rc_path.exists() else {}
    seed_table = ((rc.get("pop_quota") or {}).get("seed_price_table"))
    table_ema = None
    if seed_table and Path(seed_table).exists():
        table_ema = json.loads(Path(seed_table).read_text(encoding="utf-8")).get("price_ema")
    v = {"run_dir": str(run_dir), "n_batches": len(rows), "partitions": parts,
         "seed_price_table": seed_table, "table_price_ema": table_ema,
         "code_PRICE_EMA": pq.PRICE_EMA, "clamp_factor": cost.get("clamp_factor")}

    # ---- 1. one price step per SERVED batch, zero-yield included --------------- #
    served_moved, served_static, other_moved = [], [], []
    admitted = quota.get("realized_vs_intended", {})
    seeds0 = cost.get("seed") or {}
    clamp0 = float(cost.get("clamp_factor") or pq.PRICE_CLAMP)

    def at_clamp_edge(p, x):
        """A price pinned at a band edge CANNOT move even though the raw estimate did — the
        one legitimate way a served batch shows a static price, and it must be distinguished
        from the run-26 defect rather than counted as it."""
        seed = float(seeds0.get(p, pq.SEED_PRICE))
        return (abs(float(x) - seed / clamp0) < 1e-3 or abs(float(x) - seed * clamp0) < 1e-3)

    for a, b in zip(rows, rows[1:]):
        served = a["chosen"]
        moved = {p for p in b["price"]
                 if abs(float(b["price"][p]) - float(a["price"].get(p, 0.0))) > EPS}
        if served is None:
            continue
        if served in moved:
            served_moved.append(a["batch"])
        else:
            # A served batch whose price did NOT move: legitimate only where the clamp is
            # already binding (the raw estimate moved, the reported price cannot).
            px = a["price"].get(served)
            served_static.append({"batch": a["batch"], "partition": served, "price": px,
                                  "at_clamp_edge": at_clamp_edge(served, px),
                                  "cost_capped": served in (a.get("capped") or [])})
        extra = moved - {served}
        if extra:
            other_moved.append({"batch": a["batch"], "served": served, "also_moved": sorted(extra)})
    unexplained = [s for s in served_static if not s["at_clamp_edge"]]
    v["priced"] = {
        "served_batches_compared": len(rows) - 1,
        "served_price_moved": len(served_moved),
        "served_price_static_AT_CLAMP_EDGE": len(served_static) - len(unexplained),
        "served_price_static_UNEXPLAINED": unexplained,
        "batches_where_a_NON_served_partition_moved": other_moved,
        "verdict": ("EVERY served batch priced (moves, or pinned at a clamp edge)"
                    if not unexplained else
                    f"{len(unexplained)}/{len(rows)-1} served batches did not price"),
    }

    # ---- 1b. the same claim from the OTHER record ------------------------------ #
    # `cost.price_samples[p]` is the estimator's own count of windows it closed for p; the
    # trace's `chosen` column is the driver's count of batches it served p. Under "every served
    # batch prices" they are the same number, and they come from two independent writers — so
    # this is the relational form of ask 1 (`verification_practice.md` §5) rather than a second
    # reading of the price series. Every trace row counts, the last one included: the budget
    # check that ends the loop runs BEFORE the pick, so a written row always got its batch and
    # therefore its window. Rows with `chosen: null` are skipped on both sides (nothing served,
    # nothing charged, no window).
    samples = cost.get("price_samples") or {}
    pops = {}
    for r in rows:
        if r["chosen"]:
            pops[r["chosen"]] = pops.get(r["chosen"], 0) + 1
    v["samples_vs_pops"] = {
        "per_partition": {p: {"pops": pops.get(p, 0), "price_samples": samples.get(p, 0),
                              "equal": pops.get(p, 0) == samples.get(p, 0)}
                          for p in sorted(set(pops) | set(samples))},
        "n_equal": sum(1 for p in set(pops) | set(samples)
                       if pops.get(p, 0) == samples.get(p, 0)),
        "n_partitions": len(set(pops) | set(samples)),
    }

    # ---- 2. finite and inside the clamp band ---------------------------------- #
    seeds, clamp = seeds0, clamp0
    bad = []
    n_prices = 0
    for r in rows:
        for p, x in r["price"].items():
            n_prices += 1
            x = float(x)
            if not (x == x and abs(x) != float("inf")):
                bad.append({"batch": r["batch"], "partition": p, "price": x, "why": "not finite"})
                continue
            seed = float(seeds.get(p, pq.SEED_PRICE))
            lo, hi = seed / clamp, seed * clamp
            if x < lo - 1e-6 or x > hi + 1e-6:
                bad.append({"batch": r["batch"], "partition": p, "price": x,
                            "band": [round(lo, 4), round(hi, 4)], "why": "outside clamp band"})
    v["clamp"] = {"prices_checked": n_prices, "clamp": clamp, "violations": bad,
                  "seed_table": seed_table,
                  "at_band_edge": sorted(cost.get("clamped", []))}

    # ---- 3. sample_weight, as a pure function and as this run's windows -------- #
    a = float(table_ema if table_ema is not None else pq.PRICE_EMA)
    U = 4.0
    w = pq.CostToMine.sample_weight
    props = {
        "ema_in_use": a,
        "zero_units_is_zero": w(a, 0.0, U) == 0.0,
        "monotone_in_units": all(w(a, u, U) < w(a, u + 0.5, U) for u in [x / 2 for x in range(20)]),
        "in_unit_interval": all(0.0 <= w(a, u, U) < 1.0 for u in [0, 0.1, 1, 10, 1e6]),
        "equals_ema_when_window_matches_accumulator": abs(w(a, U, U) - a) < 1e-12,
        "thin_pulls_less_than_thick": w(a, 0.1, U) < w(a, 10.0, U),
        "weight_at_one_class3_window": round(w(a, 0.1, U), 6),
        "weight_at_one_class4_window": round(w(a, 1.0, U), 6),
    }
    v["sample_weight"] = props

    # ---- 4. floor vs starvation, reported separately --------------------------- #
    unspent = quota.get("unspent_floor") or {}
    fvd = summary.get("floor_vs_deficit") or quota.get("floor_vs_deficit") or {}
    never_served = [p for p in parts if not any(r["chosen"] == p for r in rows)]
    zero_queue_all_run = [p for p in parts
                          if all(int(r["queue_lens"].get(p, 0)) == 0 for r in rows)]
    v["floor"] = {
        "unspent_floor_alarms": unspent.get("alarms", unspent),
        "floor_vs_deficit": fvd,
        "never_served_partitions": never_served,
        "of_those_UNSERVABLE_all_run_queue0": zero_queue_all_run,
        "of_those_STARVED_despite_supply": sorted(set(never_served) - set(zero_queue_all_run)),
        "caveat": (f"n={len(rows)} batches. A floor that never needed to fire and a partition "
                   f"starved of service are DIFFERENT readings and are listed separately; at "
                   f"this n neither is a rate."),
    }
    v["realized_vs_intended"] = admitted

    # ---- 5. early vs late service, and the price path ------------------------- #
    batches = sorted(r["batch"] for r in rows)
    split = batches[len(batches) // 2] if batches else 0
    v["early_late"] = early_late(rows, split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(v, indent=2), encoding="utf-8")
    brief = {
        "n_batches": v["n_batches"], "seed_price_table": v["seed_price_table"],
        "table_price_ema": v["table_price_ema"], "code_PRICE_EMA": v["code_PRICE_EMA"],
        "priced": {k: (x if not isinstance(x, list) else len(x))
                   for k, x in v["priced"].items()},
        "samples_vs_pops": {k: x for k, x in v["samples_vs_pops"].items()
                            if k != "per_partition"},
        "clamp": {k: x for k, x in v["clamp"].items() if k != "violations"},
        "n_clamp_violations": len(v["clamp"]["violations"]),
        "sample_weight": v["sample_weight"],
        "floor": {k: x for k, x in v["floor"].items() if k != "floor_vs_deficit"},
        "early_late": {k: x for k, x in v["early_late"].items() if k != "per_partition"},
    }
    print(json.dumps(brief, indent=2))
    el = v["early_late"]["per_partition"]
    w2 = max((len(p) for p in el), default=10)
    print(f"\n{'partition':<{w2}}  pops_e  pops_l  share_e  share_l   px_0   px_split   px_end  x")
    for p, r in sorted(el.items()):
        print(f"{p:<{w2}}  {r['pops_early']:6d}  {r['pops_late']:6d}  "
              f"{100 * (r['service_share_early'] or 0):6.1f}%  "
              f"{100 * (r['service_share_late'] or 0):6.1f}%  "
              f"{r['price_first']:6.3f}  {r['price_at_split']:8.3f}  {r['price_final']:7.3f}  "
              f"{(r['price_ratio_final_over_first'] or 0):4.2f}"
              + ("   <-- COLLAPSED" if r["service_collapsed"] else ""))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
