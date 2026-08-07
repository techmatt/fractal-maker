#!/usr/bin/env python
r"""backfill_triggered_stamp.py — repair the `triggered` column truncated by push_children.

WHAT WENT WRONG. `steered_frontier.push_children` rebuilt each frontier node without
carrying `triggered`, and `expand_group` stamps a child from the parent NODE — so the stamp
survived exactly one generation. Every deeper descendant of a triggered maneuver was written
`triggered=False`, i.e. counted as FRESH supply, which inflates precisely the arm the
triggered/fresh split exists to protect (`minibrot_maneuvers.md` §8.0). The producing defect
is fixed in `push_children`; this repairs the rows already on disk.

THE REPAIR IS A JOIN, NOT AN INFERENCE. Three independent carriers of the same fact ride
every recorded row, and only ONE of them was truncated:

  * `triggered`             — the boolean, rebuilt per generation. THE BROKEN ONE.
  * `mix_source`            — the string `triggered:<method>:k=<k>`, propagated by
                              `push_children` and `expand_group` (both carried it).
  * `maneuver.triggered` +  — the provenance dict, likewise carried the whole way down.
    `maneuver.trigger_oid`

So the backfill does not guess: it asserts the two surviving carriers AGREE ON EVERY ROW and
then writes what they say. A disagreement is a hard failure — two carriers that disagree mean
the lineage itself is broken and no repair is defensible.

DIRECTION IS ONE-WAY BY DESIGN. Only `False/absent -> True` is written. A row already stamped
`True` is never cleared, because the failure mode being repaired can only ever have LOST a
stamp; a repair that could also remove one could silently rewrite a correct record if the
lineage rule were ever wrong. `--apply` refuses if any row would need clearing.

SCOPE. `q4_candidates.jsonl` (the record-and-rank store) is the only artifact in the tree that
carries a `triggered` column at all — `outcome_ledger.jsonl` and `harvest_log.jsonl` never had
one, so for them "backfill" would mean INVENTING a column that was not part of the record when
it was written, which `storage_classes.md` ("a committed record keeps what was true when
written") forbids. Earlier runs are therefore REPORT-ONLY: this tool prints how much
triggered-lineage material each holds, and writes nothing.

  uv run python tools/atlas/backfill_triggered_stamp.py report
  uv run python tools/atlas/backfill_triggered_stamp.py apply --run q4_long_harvest_20260803
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_record            # noqa: E402  (segments-aware run-record layer)

DISCOVERY = ROOT / "data" / "discovery"
STORE = "q4_candidates.jsonl"
LINEAGE_PREFIX = "triggered:"


# --------------------------------------------------------------------------- #
# the lineage rule + the agreement check
# --------------------------------------------------------------------------- #
def lineage_triggered(row: dict) -> bool:
    """`mix_source` carrier: a triggered maneuver stamps `triggered:<method>:k=<k>` and every
    descendant inherits the string verbatim."""
    return str(row.get("mix_source") or "").startswith(LINEAGE_PREFIX)


def maneuver_triggered(row: dict) -> bool:
    """Provenance-dict carrier. `trigger_oid` (the admission that fired the trigger) is
    checked as well as the boolean: a fresh maneuver's `man` dict has neither key, so the two
    together cannot be confused with a fresh operator row that merely lacks the flag."""
    m = row.get("maneuver") or row.get("man") or {}
    return bool(m.get("triggered")) and m.get("trigger_oid") is not None


def audit_rows(rows: list[dict]) -> dict:
    """Compare all three carriers over `rows`. Returns the counts and the disagreement lists;
    decides nothing (the caller does), so a report and an apply read the identical numbers."""
    stamped = lineage = manrec = 0
    carrier_disagree, would_clear, would_set = [], [], []
    for i, r in enumerate(rows):
        s, l, m = bool(r.get("triggered")), lineage_triggered(r), maneuver_triggered(r)
        stamped += s
        lineage += l
        manrec += m
        if l != m:
            carrier_disagree.append(i)
        elif l and not s:
            would_set.append(i)
        elif s and not l:
            would_clear.append(i)
    return dict(n=len(rows), stamped=stamped, lineage=lineage, maneuver_rec=manrec,
                carrier_disagree=carrier_disagree, would_set=would_set,
                would_clear=would_clear)


def read_jsonl(p: Path) -> list[dict]:
    return run_record.read_rows(p)     # segments-aware (q4_candidates.jsonl rotates)


def write_jsonl_atomic(p: Path, rows: list[dict]):
    """Replace the store. Routed through `run_record.replace_stream` for a segmented stream:
    writing a plain `q4_candidates.jsonl` back beside rotated `.jsonl.gz` segments would
    DOUBLE every rotated row, and every reader here concatenates without complaint."""
    if run_record.is_segmented(p):
        run_record.replace_stream(p, rows)
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


# --------------------------------------------------------------------------- #
# report / apply
# --------------------------------------------------------------------------- #
def report(run_dirs: list[Path]) -> dict:
    """Report-only across every run: which hold a `triggered` column, and how much
    triggered-lineage material each carries by `mix_source` regardless."""
    out = {}
    for d in sorted(run_dirs):
        rec: dict = {}
        store = d / STORE
        if run_record.exists(store):
            rows = read_jsonl(store)
            rec["q4_candidates"] = audit_rows(rows)
        for other in ("outcome_ledger.jsonl", "harvest_log.jsonl"):
            p = d / other
            if not run_record.exists(p):
                continue
            rows = read_jsonl(p)
            rec[other] = dict(
                n=len(rows),
                has_triggered_column=any("triggered" in r for r in rows),
                lineage=sum(lineage_triggered(r) for r in rows),
                note="report-only: no `triggered` column was ever written to this artifact",
            )
        if rec:
            out[d.name] = rec
    return out


def apply(run_dir: Path) -> dict:
    """Backfill `q4_candidates.jsonl` in `run_dir`. Fails loud on any carrier disagreement or
    any row that would have to be CLEARED; writes an audit sidecar beside the store."""
    store = run_dir / STORE
    if not run_record.exists(store):
        raise SystemExit(f"{store} does not exist — nothing to backfill "
                         f"(only the record-and-rank store carries a `triggered` column)")
    rows = read_jsonl(store)
    a = audit_rows(rows)
    if a["carrier_disagree"]:
        raise SystemExit(
            f"REFUSING: {len(a['carrier_disagree'])} of {a['n']} rows have `mix_source` and "
            f"`maneuver` disagreeing about triggered-ness (first at index "
            f"{a['carrier_disagree'][0]}). The lineage itself is broken; no repair is "
            f"defensible from here.")
    if a["would_clear"]:
        raise SystemExit(
            f"REFUSING: {len(a['would_clear'])} rows are stamped triggered but have no "
            f"triggered lineage. This tool only ever ADDS a lost stamp; clearing one would "
            f"mean the lineage rule is wrong, which is a question, not a repair.")
    for i in a["would_set"]:
        rows[i]["triggered"] = True
    if a["would_set"]:
        write_jsonl_atomic(store, rows)
    after = audit_rows(rows)
    rec = dict(store=str(store), rows=a["n"], stamped_before=a["stamped"],
               stamped_after=after["stamped"], backfilled=len(a["would_set"]),
               lineage=a["lineage"], maneuver_rec=a["maneuver_rec"],
               carriers_agreed_on_all_rows=True,
               rule="triggered := mix_source.startswith('triggered:') "
                    "== (maneuver.triggered and maneuver.trigger_oid is not None)",
               direction="False/absent -> True only")
    (run_dir / "triggered_backfill.json").write_text(json.dumps(rec, indent=2),
                                                     encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="report-only across every discovery run")
    r.add_argument("--discovery-root", default=str(DISCOVERY))
    a = sub.add_parser("apply", help="backfill ONE run's q4_candidates.jsonl")
    a.add_argument("--run", required=True, help="run directory name under data/discovery/")
    a.add_argument("--discovery-root", default=str(DISCOVERY))
    args = ap.parse_args()
    root = Path(args.discovery_root)

    if args.cmd == "report":
        out = report([d for d in root.iterdir() if d.is_dir()])
        print(json.dumps(out, indent=2))
        return
    rec = apply(root / args.run)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
