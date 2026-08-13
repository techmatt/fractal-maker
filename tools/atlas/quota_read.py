#!/usr/bin/env python
"""quota_read.py — a frontier run's floor/quota readout, EARLY vs LATE.

THE QUESTION IT ANSWERS. A pop-quota run allocates each batch to a partition out of one of
two buckets — `floor` (the 5%-of-time guarantee) or `deficit` (the census gap). Whether the
allocator worked is not answerable from run totals: run 26's `mandelbrot` spent its whole
floor allotment BEFORE the midpoint and took 0.0% of late admissions, which reads as a
healthy 6.2% pop share and 96 admissions in the totals. So every population here is split.

Three populations, kept separate because they answer different halves of it:
  * POP TIME  — active minutes per partition, split by the BUCKET that bought them
                (`floor` vs `deficit`), joined from `quota_trace`'s per-batch
                `chosen`/`bucket` against `stage_times.jsonl`'s per-batch `frontier_batch`
                durations. This is floor SPEND.
  * ADMISSIONS — `harvest_log.jsonl` rows with `admitted`, which is exactly the outcome
                ledger's `distinct=True` population (verified per partition, both totals
                reported). A raw ledger LINE count is not that number: the ledger also keeps
                its non-distinct near-dups as a record. This is NOT the same number as pop
                time, and the two are reported side by side rather than blended.
  * SHARES    — the trace's own cumulative `realized` vector, read at the early/late split so
                the allocator's own accounting is quotable beside the joined one.

EARLY/LATE is the batch midpoint by default (`--split`), so both halves carry the same number
of pop decisions. A time midpoint would put ~2/3 of the batches in "early" — batches are not
equal-cost, and the expensive ones cluster late (run 26: 27 of 118 batches took 61.8% of the
loop).

AN UNSPENT FLOOR AND A STARVED PARTITION ARE DIFFERENT READINGS and are reported separately.
A partition served above its floor through the DEFICIT bucket has a floor that never had to
fire, which is the allocator working; a partition with neither is starved. Conflating them
makes a working allocator read as broken (run 26's report, "Floor/quota read").

PROVENANCE OF THE PER-BATCH COST. `stage_times.jsonl` (`tools/stage_times.py`), which is the
durable in-run record — not the console log, which is what this readout scraped with a regex
when it was written for run 26 and which does not survive a `scratch/` wipe. Run 26's stream
is a backfill from its own log (`source=log_backfill`, 1 s quantization); that stamp rides
through into this readout's output so a comparison can never mistake it for an in-run
measurement.

  uv run python tools/atlas/quota_read.py --run-dir data/discovery/prod27_20260812 \
      --out scratch/production_run27/quota_read.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import paths, run_record, stage_times  # noqa: E402


def batch_seconds(run: Path) -> tuple[dict[int, float], dict]:
    """batch index -> seconds that batch cost, from the run's own stage-time stream.

    `unit` is `batch:<n>`; anything else under `frontier_batch` is counted as unparsed and
    reported rather than dropped silently. The stream is already deduped on `(stage, unit)`
    by `stage_times.read`, which is what makes this safe on a killed-and-resumed run."""
    rows, diag = stage_times.read(run)
    if not diag["present"]:
        raise SystemExit(
            f"no {stage_times.STREAM} in {run}. A run that predates the stream (run 26 and "
            f"earlier) can be backfilled from its console log with "
            f"tools/atlas/backfill_stage_times.py; without it there is no per-batch cost and "
            f"the floor/deficit MINUTES cannot be split early/late.")
    out: dict[int, float] = {}
    unparsed = 0
    sources = Counter()
    for r in rows:
        if r.get("stage") != "frontier_batch":
            continue
        unit = str(r.get("unit") or "")
        sources[r.get("source") or "in_run"] += 1
        if not unit.startswith("batch:") or not unit[6:].isdigit():
            unparsed += 1
            continue
        out[int(unit[6:])] = float(r.get("dur_s") or 0.0)
    return out, {"n_frontier_batch_rows": len(out), "unit_unparsed": unparsed,
                 "dur_sources": dict(sources), "stream": diag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="default: the run's own scratch family, quota_read/<run>.json")
    ap.add_argument("--split", type=int, default=0,
                    help="batch index of the early/late cut (0 = the batch midpoint)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out_path = Path(args.out) if args.out else paths.scratch("quota_read", f"{run.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace = run_record.require_rows(run / "quota_trace.jsonl")
    secs, cost_diag = batch_seconds(run)
    mins = {b: s / 60.0 for b, s in secs.items()}

    batches = sorted(r["batch"] for r in trace)
    split = args.split or (batches[len(batches) // 2] if batches else 0)

    # --- pop time by bucket, early/late -------------------------------------------------
    spend: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {"early": {"floor": 0.0, "deficit": 0.0}, "late": {"floor": 0.0, "deficit": 0.0}})
    pops: dict[str, Counter] = defaultdict(Counter)
    unjoined = 0
    for r in trace:
        p, bucket, b = r.get("chosen"), (r.get("bucket") or "deficit"), r["batch"]
        if not p:
            continue
        if b not in mins:
            unjoined += 1
        half = "early" if b <= split else "late"
        spend[p][half][bucket] = spend[p][half].get(bucket, 0.0) + mins.get(b, 0.0)
        pops[p][f"{half}:{bucket}"] += 1

    # --- admissions, early/late ---------------------------------------------------------
    # From harvest_log.jsonl, which carries BOTH `batch` and `partition` per candidate. The
    # outcome ledger carries neither a batch stamp nor a partition column (its `family` is
    # the partition, but nothing dates a row to a batch), so the join has to happen here.
    # The ledger's row count is still read, as a cross-check that the two agree.
    adm_early, adm_late, adm_nobatch = Counter(), Counter(), Counter()
    for row in run_record.require_rows(run / "harvest_log.jsonl"):
        if not row.get("admitted"):
            continue
        part, b = row.get("partition"), row.get("batch")
        if b is None:
            adm_nobatch[part] += 1
        elif b <= split:
            adm_early[part] += 1
        else:
            adm_late[part] += 1
    # RECONCILIATION, measured on run 26 at full scale: harvest_log `admitted` == the ledger's
    # `distinct=True` rows, exactly, per partition. The ledger ALSO carries its non-distinct
    # near-dups as a record (1,775 lines against 1,418 admitted), so a raw ledger line count
    # overstates admissions by that much. Counting both is the check; `distinct` is the
    # population.
    led = run / "outcome_ledger.jsonl"
    ledger_rows, ledger_distinct = Counter(), Counter()
    if led.exists():
        for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ledger_rows[row.get("family")] += 1
            if row.get("distinct"):
                ledger_distinct[row.get("family")] += 1

    # CROSS-CHECK. summary.json carries the allocator's OWN `realized_by_bucket` totals
    # (`floor_vs_deficit`), which is authority; the join above is an independent path to the
    # same number, built from the trace and the stage-time stream. Agreement is what makes the
    # early/late SPLIT — which the summary cannot give — trustworthy, so it is computed, not
    # assumed.
    summary_fvd = {}
    ssum = run / "summary.json"
    if ssum.exists():
        summary_fvd = json.loads(ssum.read_text(encoding="utf-8")).get("floor_vs_deficit") or {}

    last = trace[-1] if trace else {}
    at_split = next((r for r in trace if r["batch"] > split), last)

    total_floor = sum(h["floor"] for v in spend.values() for h in v.values())
    total_def = sum(h["deficit"] for v in spend.values() for h in v.values())

    rep = {
        "run_dir": str(run),
        "n_batches": len(trace),
        "split_batch": split,
        "batch_cost_provenance": cost_diag,
        "batches_in_trace_with_no_stage_time_row": unjoined,
        "active_min_joined": round(total_floor + total_def, 2),
        "floor_min_total": round(total_floor, 2),
        "deficit_min_total": round(total_def, 2),
        "intended_mix": last.get("intended"),
        "realized_share_at_split": at_split.get("realized"),
        "realized_share_final": last.get("realized"),
        "final_deficit": last.get("deficit"),
        "final_price": last.get("price"),
        "final_floor_debt": last.get("floor_debt"),
        "floor_trigger_min": last.get("floor_trigger_min"),
        "summary_floor_vs_deficit": summary_fvd,
        "per_partition": {},
        "admissions_without_batch_stamp": dict(adm_nobatch),
        "ledger_rows_by_family": dict(ledger_rows),
        "harvest_log_vs_ledger": {
            "harvest_admitted": sum(adm_early.values()) + sum(adm_late.values()) + sum(adm_nobatch.values()),
            "ledger_rows": sum(ledger_rows.values()),
            "ledger_distinct": sum(ledger_distinct.values()),
            "agrees": (sum(adm_early.values()) + sum(adm_late.values()) + sum(adm_nobatch.values()))
                      == sum(ledger_distinct.values()),
        },
    }
    parts = sorted(set(spend) | set(adm_early) | set(adm_late) | set(adm_nobatch))
    tot_adm = sum(adm_early.values()) + sum(adm_late.values()) + sum(adm_nobatch.values())
    for p in parts:
        s = spend.get(p, {"early": {}, "late": {}})
        fl = s["early"].get("floor", 0.0) + s["late"].get("floor", 0.0)
        de = s["early"].get("deficit", 0.0) + s["late"].get("deficit", 0.0)
        a = adm_early[p] + adm_late[p] + adm_nobatch[p]
        rep["per_partition"][p] = {
            "floor_min": round(fl, 2), "deficit_min": round(de, 2),
            "floor_min_early": round(s["early"].get("floor", 0.0), 2),
            "floor_min_late": round(s["late"].get("floor", 0.0), 2),
            "pop_time_share": round((fl + de) / (total_floor + total_def), 4) if (total_floor + total_def) else None,
            "pops": dict(pops.get(p, Counter())),
            "admissions": a,
            "admissions_early": adm_early[p], "admissions_late": adm_late[p],
            "admission_share": round(a / tot_adm, 4) if tot_adm else None,
            "admission_share_early": round(adm_early[p] / sum(adm_early.values()), 4) if sum(adm_early.values()) else None,
            "admission_share_late": round(adm_late[p] / sum(adm_late.values()), 4) if sum(adm_late.values()) else None,
        }
    rep["unspent_floor_partitions"] = sorted(
        p for p, v in rep["per_partition"].items() if v["floor_min"] == 0.0)
    rep["starved_partitions"] = sorted(
        p for p, v in rep["per_partition"].items()
        if v["floor_min"] == 0.0 and v["deficit_min"] == 0.0)
    rep["floor_never_needed_partitions"] = sorted(
        p for p, v in rep["per_partition"].items()
        if v["floor_min"] == 0.0 and v["deficit_min"] > 0.0)

    out_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    w = max(len(p) for p in parts) if parts else 10
    print(f"{len(trace)} pop decisions, split at batch {split}; "
          f"{rep['active_min_joined']:.1f} active min joined "
          f"({rep['floor_min_total']:.1f} floor / {rep['deficit_min_total']:.1f} deficit)")
    print(f"{'partition':<{w}}  floor_m  (early/late)   def_m   pop%   adm  adm%  early%  late%")
    for p in parts:
        v = rep["per_partition"][p]
        print(f"{p:<{w}}  {v['floor_min']:7.1f}  ({v['floor_min_early']:5.1f}/{v['floor_min_late']:5.1f})  "
              f"{v['deficit_min']:6.1f}  {100 * (v['pop_time_share'] or 0):5.1f}  "
              f"{v['admissions']:4d}  {100 * (v['admission_share'] or 0):4.1f}  "
              f"{100 * (v['admission_share_early'] or 0):5.1f}  {100 * (v['admission_share_late'] or 0):5.1f}")
    if rep["floor_never_needed_partitions"]:
        print("   floor never needed (served through deficit):",
              ", ".join(rep["floor_never_needed_partitions"]))
    if rep["starved_partitions"]:
        print("!! STARVED (no floor AND no deficit time):", ", ".join(rep["starved_partitions"]))
    sfd = rep.get("summary_floor_vs_deficit") or {}
    if sfd:
        d_fl = abs(sfd.get("floor_min", 0.0) - rep["floor_min_total"])
        d_de = abs(sfd.get("deficit_min", 0.0) - rep["deficit_min_total"])
        print(f"   cross-check vs summary.json floor_vs_deficit: floor {sfd.get('floor_min')} "
              f"(delta {d_fl:.2f}) / deficit {sfd.get('deficit_min')} (delta {d_de:.2f})")
    if unjoined:
        print(f"   !! {unjoined} trace rows had no stage_times row — their minutes are 0 here")
    print("->", out_path)


if __name__ == "__main__":
    main()
