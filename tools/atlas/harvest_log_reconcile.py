#!/usr/bin/env python
r"""harvest_log_reconcile.py — dedup a run's append-only harvest log and tie it to the
run's own checkpointed counters, BEFORE any rate is quoted off it.

WHY. `harvest_log.jsonl` is opened `"a"` and written per check, while `totals` is
checkpointed to `state.json` per batch and frozen into `summary.json` at `finish()`. A kill
between the two — which is the normal way a long run ends here — re-runs the tail batch on
`--resume` and re-appends rows the counters already hold, so a rate taken straight off the
file's line count is quoted over a denominator with duplicates in it. The duplicates are not
hypothetical noise: the whole point of the per-batch checkpoint is that the run resumes at a
batch boundary and redoes the work inside it.

THE DEDUP KEY IS `node_id`, AND KEEPING THE **LAST** ROW IS THE LOAD-BEARING HALF. `node_ctr`
and `batch_i` are both restored from the checkpoint, so a redone batch re-issues the same
node ids from the same counter position; a repeated `node_id` therefore means "this counter
position was written twice", and the later write is the one the run's counters were advanced
by. (`(batch, node_id)` is NOT a weaker key here — `batch_i` is restored too, so the two
agree; the choice is `node_id` because "a node is checked once" is the tighter invariant, not
because the pair misses anything. Claiming otherwise would be a distinction no test can
show.) What actually proves the dedup was sufficient is the reconcile below: a killed attempt
whose redone batch was SHORTER leaves stale rows the key cannot see, and the identity does.

THE RECONCILE IS THE GUARD, and it fails loud. Three identities have to close against the
summary the run wrote about itself:

    deduped rows                    == totals.harvest_checks
    rows with precanon_dup set      == totals.precanon_dup
    rows with admitted              == totals.admitted

(`render_failed` checks are logged to the q4 store but never to the harvest log, and are
subtracted from `harvest_checks` at the point of failure, so they are outside the identity by
construction.) A mismatch means the log and the counters disagree about the population, and
NO rate may be quoted until that is understood — reporting one anyway is how a denominator
nobody checked becomes a number in a design doc.

WHAT IT THEN REPORTS, per partition and pooled:

  * PRECANON SKIP RATE — `precanon_dup / harvest_checks`. The adoption record
    (`data/atlas/precanon_calibration/adoption.json`) predicts this collapses from the ~91.5%
    the retired K=1.5 x max rule achieved to 1.6-28% at the adopted K=0.25 x min, and says
    outright that the first production run's own telemetry is the read.
  * TIER-2 CONFIRMATION VOLUME — checks that actually paid a 640x360 ss2 canonical
    confirmation render (`harvest_checks - precanon_dup`), which is the bill that moves.

    uv run python tools/atlas/harvest_log_reconcile.py --run-dir data/discovery/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LOG_NAME = "harvest_log.jsonl"


class ReconcileError(RuntimeError):
    """The deduped log and the run's checkpointed counters disagree about the population.

    Not recoverable by a flag. The two are independent records of the same events, which is
    the only reason either can check the other; a caller allowed to proceed past a mismatch
    is quoting a rate over a denominator nobody verified."""


def read_rows(run_dir: Path) -> list[dict]:
    """Every line of the log, torn tail tolerated and COUNTED, never silently dropped.

    A killed process can leave a partial final line. Skipping it quietly would be an
    absence-tolerant read of exactly the kind that un-guards when its subject is damaged, so
    it is returned as a count for the caller to report."""
    p = Path(run_dir) / LOG_NAME
    if not p.exists():
        raise ReconcileError(f"{p} missing — the run wrote no harvest log. A run that "
                             f"harvested nothing still creates it on its first check, so "
                             f"absence means the run never reached a harvest.")
    rows, torn = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    return rows, torn


def dedup(rows: list[dict]) -> tuple[list[dict], dict]:
    """`(deduped, report)` keeping the LAST row per `node_id`.

    Last and not first: a re-run batch writes the row the run's own counters were advanced
    by most recently, so the surviving row is the one the summary is a count of."""
    by_node: dict = {}
    order: list = []
    dup_batches: Counter = Counter()
    for r in rows:
        k = r.get("node_id")
        if k is None:                       # pre-schema row: cannot be keyed, kept as-is
            order.append(object())
            by_node[order[-1]] = r
            continue
        if k in by_node:
            dup_batches[r.get("batch")] += 1
        else:
            order.append(k)
        by_node[k] = r
    deduped = [by_node[k] for k in order]
    return deduped, dict(
        rows_read=len(rows), rows_deduped=len(deduped),
        duplicates_dropped=len(rows) - len(deduped),
        duplicate_batches=dict(sorted(dup_batches.items(), key=lambda kv: -kv[1])),
        keyless_rows=sum(1 for r in rows if r.get("node_id") is None),
    )


def reconcile(deduped: list[dict], totals: dict) -> dict:
    """The three identities, or raise naming every one that failed."""
    got = dict(
        harvest_checks=len(deduped),
        precanon_dup=sum(1 for r in deduped if r.get("precanon_dup") is not None),
        admitted=sum(1 for r in deduped if r.get("admitted")),
    )
    want = {k: int(totals.get(k, 0)) for k in got}
    bad = [f"{k}: log {got[k]} != summary totals {want[k]}" for k in got if got[k] != want[k]]
    if bad:
        raise ReconcileError(
            "the deduped harvest log does not tie to the run's checkpointed counters:\n  "
            + "\n  ".join(bad)
            + "\nNo rate may be quoted off this log until the difference is understood. "
              "The two are independent records of the same events; that is the only reason "
              "either can check the other.")
    return dict(verdict="RECONCILED", log=got, summary=want)


def rates(deduped: list[dict]) -> dict:
    """Precanon skip rate + tier-2 confirmation volume, pooled and per partition.

    Population-gated at the READER: a partition with zero checks reports `null`, not 0.0 —
    a rate over an empty denominator is not a small rate."""
    per: dict = defaultdict(lambda: dict(checks=0, precanon_dup=0, admitted=0))
    for r in deduped:
        d = per[r.get("partition", "?")]
        d["checks"] += 1
        d["precanon_dup"] += int(r.get("precanon_dup") is not None)
        d["admitted"] += int(bool(r.get("admitted")))

    def block(d):
        n, s = d["checks"], d["precanon_dup"]
        return dict(checks=n, precanon_dup=s, tier2_renders=n - s, admitted=d["admitted"],
                    precanon_skip_rate=(round(s / n, 4) if n else None),
                    tier2_render_rate=(round((n - s) / n, 4) if n else None))

    total = dict(checks=len(deduped),
                 precanon_dup=sum(v["precanon_dup"] for v in per.values()),
                 admitted=sum(v["admitted"] for v in per.values()))
    return dict(pooled=block(total),
                per_partition={p: block(v) for p, v in sorted(per.items())})


def readout(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    sp = run_dir / "summary.json"
    if not sp.exists():
        raise ReconcileError(f"{sp} missing — the run has not finished. The checkpointed "
                             f"counters this log is checked against are frozen by `finish()`; "
                             f"state.json holds them mid-flight and is not a substitute.")
    summary = json.loads(sp.read_text(encoding="utf-8"))
    rows, torn = read_rows(run_dir)
    deduped, dd = dedup(rows)
    dd["torn_lines"] = torn
    rec = reconcile(deduped, summary.get("totals") or {})
    return dict(run=run_dir.name, dedup=dd, reconcile=rec, rates=rates(deduped))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reps = [readout(Path(r)) for r in a.run_dir]
    txt = json.dumps(reps if len(reps) > 1 else reps[0], indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
