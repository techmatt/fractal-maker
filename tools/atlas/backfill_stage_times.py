#!/usr/bin/env python
r"""backfill_stage_times.py — reconstruct a PRE-INSTRUMENTATION run's `stage_times.jsonl` from
its own console log.

    uv run python tools/atlas/backfill_stage_times.py \
        --run-dir data/discovery/prod26_20260812 --log scratch/production_run26/breadth.log
    ... --run-dir data/discovery/prod26_20260812_dive --log scratch/production_run26/chain.log
    ... --dry-run          # parse and print, write nothing

WHY. `stage_times.jsonl` starts with run 27, so run 26 — the only production run available as a
BASELINE for run 27 — has no per-unit timing in its record. It does have per-unit timing in its
console log: `crawl` and `run_dive` both print each unit's duration. This lifts those durations
into the stream so one reader (`tools/run_profile.py`) profiles run 26 and run 27 the same way,
instead of run 26 needing a second, log-shaped reader that would then have to be maintained.

THE ROWS ARE STAMPED `source="log_backfill"` AND THAT IS LOAD-BEARING. They are not what the
in-run writer measures and must never be read as if they were:

  * **1 s quantization.** The crawl prints `{dt:.0f}s`, so a 12.4 s batch is on record as 12 s.
    Ratios against a stage median are unaffected at the scale that matters here (run 26's batch
    median is ~90 s); a stage whose units are seconds long would be destroyed by it, which is
    why this tool refuses any stage it cannot read at full width.
  * **No wall clock.** The log lines carry no timestamp, so `t_end` is OMITTED rather than
    invented (`stage_times.record(t_end=None)`). `run_profile`'s span/unaccounted arithmetic is
    therefore blank for a backfilled run — which is correct: nothing in the log says when the
    run's first batch started, and a span computed from the backfill's own wall clock would be
    a measurement of when someone ran this script.
  * **Only what the log printed.** The emission builder printed no per-attempt duration at all,
    so there is nothing to lift for `intake`/`colorize`/`select`/`release_render` and this tool
    does not pretend otherwise — an emission run dir gets no rows and says so.

A CONSOLE LOG IS NOT A RECORD, which is the whole reason the stream exists. This is a one-time
lift for the runs that predate it, not a supported ingestion path: a run from 27 on writes the
stream directly, and running this against one would double its rows (the tool refuses when the
target already has a stream, rather than appending into it).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import stage_times as stimes           # noqa: E402

SOURCE = "log_backfill"

# `  batch 118: exp=32 cand=126 admitted(cum)=1418 julia_roots=228 frontier=3637 sat=... | 128s active=208.2m`
RE_BATCH = re.compile(
    r"^\s*batch (?P<batch>\d+): exp=(?P<exp>\d+) cand=(?P<cand>\d+) "
    r"admitted\(cum\)=(?P<adm>\d+).*?\|\s*(?P<dt>\d+)s active=(?P<active>[\d.]+)m")
# `  dive_003 [top] start d4 -> d6 (2 rungs, gate_dead_or_floor) admitted=5 | 45s active=0.8m (1/28)`
RE_DIVE = re.compile(
    r"^\s*(?P<id>dive_\d+) \[(?P<group>\w+)\] start d(?P<d0>\d+) -> d(?P<d1>\d+) "
    r"\((?P<rungs>\d+) rungs, (?P<cause>[\w_]+)\) admitted=(?P<adm>\d+) \|\s*(?P<dt>\d+)s")
# `[root-draw] pre-loop draw took 8.1m (outside the active and wall caps by design; ...)`
RE_PRELOOP = re.compile(r"^\[root-draw\] pre-loop draw took (?P<m>[\d.]+)m")
# `[root-refill] b57: ['mandelbrot'] below ... — drew 40 roots in 29s (root-draw 0.5m of ...)`
RE_REFILL = re.compile(
    r"^\[root-refill\] b(?P<batch>\d+):.*?drew (?P<added>\d+) roots in (?P<dt>\d+)s")


def parse(log_text: str) -> list[tuple]:
    """`(stage, unit, dur_s, meta)` for every timed unit the log printed, in log order."""
    out = []
    for line in log_text.splitlines():
        m = RE_PRELOOP.match(line)
        if m:
            # Printed in MINUTES to one decimal, so this one is quantized at 6 s, not 1 s.
            out.append(("root_draw", "preloop", float(m["m"]) * 60.0, {"quant_s": 6}))
            continue
        m = RE_REFILL.match(line)
        if m:
            out.append(("root_refill", f"refill:{m['batch']}", float(m["dt"]),
                        {"roots_added": int(m["added"])}))
            continue
        m = RE_BATCH.match(line)
        if m:
            out.append(("frontier_batch", f"batch:{m['batch']}", float(m["dt"]),
                        {"n_expanded": int(m["exp"]), "n_cands": int(m["cand"]),
                         "admitted_cum": int(m["adm"]),
                         "active_min_cum": float(m["active"])}))
            continue
        m = RE_DIVE.match(line)
        if m:
            out.append(("dive", m["id"], float(m["dt"]),
                        {"start_group": m["group"], "rungs": int(m["rungs"]),
                         "end_cause": m["cause"], "n_admitted": int(m["adm"]),
                         "start_depth": int(m["d0"]), "end_depth": int(m["d1"])}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="run dir the stream is written into")
    ap.add_argument("--log", required=True, nargs="+",
                    help="console log(s) of that run, in order; lines that match nothing are "
                         "ignored, so a chain log holding several legs is safe to pass")
    ap.add_argument("--dry-run", action="store_true", help="parse and print; write nothing")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dst = run_dir / stimes.STREAM
    if dst.exists() and not args.dry_run:
        raise SystemExit(
            f"{dst} already exists — refusing to append. A run from 2026-08-12 on writes this "
            f"stream itself, and backfilling into it would double every unit. Delete it "
            f"deliberately if this really is a re-backfill.")

    units = []
    for lg in args.log:
        p = Path(lg)
        if not p.exists():
            raise SystemExit(f"log not found: {p}")
        units.extend(parse(p.read_text(encoding="utf-8", errors="replace")))
    if not units:
        raise SystemExit(
            f"no timed units parsed from {args.log} — this log holds no per-unit durations "
            f"(the emission builder printed none before 2026-08-12), so there is nothing to "
            f"lift. That is a real absence, not a parse failure to work around.")

    by_stage: dict[str, int] = {}
    for stage, _u, _d, _m in units:
        by_stage[stage] = by_stage.get(stage, 0) + 1
    tot = sum(d for _s, _u, d, _m in units)
    print(f"[backfill] {len(units)} unit(s) from {len(args.log)} log(s): "
          f"{by_stage}; {tot/60:.1f} total minutes")
    if args.dry_run:
        for stage, unit, dur, meta in units[:10]:
            print(f"    {stage:<16}{unit:<16}{dur:>8.1f}s  {meta}")
        print("    (--dry-run: nothing written)")
        return

    w = stimes.StageTimes(run_dir, source=SOURCE)
    for stage, unit, dur, meta in units:
        w.record(stage, unit, dur, t_end=None, **meta)
    print(f"[backfill] -> {dst}  ({w.totals()})")


if __name__ == "__main__":
    main()
