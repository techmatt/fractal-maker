#!/usr/bin/env python
r"""stage_times.py — THE per-unit stage timing stream for a run, and the only writer of it.

WHY THIS EXISTS. Run 26 could not answer "how long did each stage take, and which units were
outliers" from its committed record. Every duration it had was an AGGREGATE in `summary.json`
(`active_min`, `wall_min`, `pre_loop_draw_min`, `root_draw_min`, `man_probe_s`) or a
CONSOLE-ONLY print: `crawl` computes each batch's `dt`, prints it, charges the quota with it,
and drops it. The per-row streams that do survive are per CANDIDATE (`harvest_log`,
`q4_candidates`, `prio_terms`) or per ALLOCATION (`quota_trace`, written at pick time, i.e.
BEFORE the batch it chose has run), so none of them has anywhere to put a duration. The
emission builder had no timing at all — not per attempt, not per stage.

So one stream, one schema, written by the frontier crawl, the dive, and the emission builder
alike, and read by `tools/run_profile.py`. A stage-total roll-up also lands in each tool's
`summary.json` (`StageTimes.totals`), so a reader holding only the summary still gets stage
totals and learns the per-unit rows exist.

THE ROW.

    {"stage": "frontier_batch", "unit": "batch:57", "dur_s": 128.4,
     "t_end": 1755040123.5, "seq": 56, "meta": {"partition": "multibrot3", "n_cands": 126}}

`stage`/`unit`/`dur_s` are the contract; `t_end` is the epoch second the unit finished (so a
reader can order units and see the gaps BETWEEN stages without a second stream); `seq` is this
writer instance's monotone counter, which is also the §11 liveness signal — a tail whose `seq`
has stopped advancing is a run that has stopped, and that is not visible from a duration alone.
Everything tool-specific goes under `meta` so a new caller can never collide with the contract.

NOT SEGMENTED, AND NOT BY OVERSIGHT. `run_record.SEGMENTED_STREAMS` exists because the per-row
streams grow at 8.9-14.2 MB/h. This one is ~150 B per unit — an 8 h run at run 26's rate is 118
batches, i.e. **~18 KB**, three orders of magnitude under `ROTATE_BYTES`. Registering it would
buy nothing and would put its `.gz` segments under the `data/discovery/**/*.jsonl.gz` LFS rule,
which is a real cost for a file this size. It is read through `run_record.iter_rows` anyway
(which short-circuits to the plain path for an unregistered stream), so registering it later
costs nothing at the read sites.

IT IS COMMITTED, ON BOTH LEGS. `data/discovery/` is un-ignored with a deny-list; this file is
not on it, so a discovery run's stage times are tracked exactly as `quota_trace` is. That is the
point — "from durable telemetry alone" is the requirement, and a stream that lives only in a
scratch tree answers it for as long as nobody runs `rm -r scratch/*`. The EMISSION leg wrote
its rows under `--out` (scratch) until 2026-08-13, so the same telemetry had two storage classes
depending on which leg produced it; it now writes to `data/emission/run_telemetry/<run>/`,
resolved through `emission_sinks.stage_times_home` so an ephemeral run still lands in scratch.
This class is the CALLER's to choose — this module only appends to the directory it is handed.

APPEND-ONLY, AND WHAT THAT COSTS. Per `verification_practice.md` §11 an append-only log is a
SUPERSET of the checkpointed counters after a kill: a resumed run re-runs the units between the
last checkpoint and the kill, and appends them again. So the reader DEDUPS on `(stage, unit)`
keeping the LAST occurrence — the attempt that actually completed — and reports how many
duplicates it dropped rather than averaging a unit twice. A kill can also leave one torn final
line; `read` counts those instead of raising. Both are the reader's job (`run_profile.py`), and
both are reported rather than silently absorbed.

Writes are not swallowed. A telemetry writer that catches its own exceptions is
`verification_practice.md` §2 in miniature — it un-guards exactly when the run dir goes away,
which is when you most want to know. This appends to a directory the caller is already writing
its record into; if that fails the run has already lost more than its timings.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

__all__ = ["STREAM", "StageTimes", "read", "totals_of", "STAGES"]

STREAM = "stage_times.jsonl"

# `record(t_end=...)`'s "stamp it now" default, distinct from an explicit `None` (= this
# source has no wall clock; omit the field). A plain `None` default cannot tell those apart.
_NOW = object()

# The stage vocabulary, in RUN ORDER. Not enforced — a caller may record any stage, and
# `run_profile` orders unknown stages after these — but it is the list a reader of a whole run
# expects to see, and an absent one is a visible omission rather than a silent gap.
STAGES = (
    "root_draw",        # frontier: the pre-loop root draw (outside both frontier caps)
    "root_refill",      # frontier: one in-loop refill draw
    "frontier_batch",   # frontier: one served batch, expand -> score -> harvest -> push
    "dive",             # frontier --dive: one dive, root to its end cause
    "intake",           # emission: ledger union -> morph cluster -> axes
    "colorize",         # emission: one colorize attempt (render + score)
    "select",           # emission: gate/pool/select, incl. the two record writes
    "release_render",   # emission: one release render at wallpaper canon
)


class StageTimes:
    """Append-only writer for one run dir's `stage_times.jsonl`.

    Opened lazily: constructing one never creates a directory, so a tool can build it beside
    its other writers before it has decided to run.
    """

    def __init__(self, run_dir, *, source: str | None = None):
        self.path = Path(run_dir) / STREAM
        # Stamped on every row when set. In-run telemetry leaves it absent; a BACKFILL
        # (tools/atlas/backfill_stage_times.py) sets it, so no reader can mistake a
        # log-derived 1 s-quantized duration for one this class measured.
        self.source = source
        self.seq = 0
        self._totals: dict[str, dict] = {}

    # ---- writing --------------------------------------------------------- #
    def record(self, stage: str, unit, dur_s: float, *, t_end=_NOW, **meta):
        """Append one finished unit. `unit` is stringified — callers pass ints freely.

        `t_end` defaults to now. Pass an epoch float to stamp a known end, or **`None` to omit
        the field entirely** — the honest state for a BACKFILL from a source that recorded
        durations but no wall clock. `None` must not silently become `time.time()`: that would
        stamp a 2026-08-12 backfill with the moment the backfill ran, and the span/unaccounted
        arithmetic downstream would then be computed over a fabricated ordering."""
        row = {"stage": str(stage), "unit": str(unit), "dur_s": round(float(dur_s), 3),
               "seq": self.seq}
        if t_end is not None:
            row["t_end"] = round(float(time.time() if t_end is _NOW else t_end), 3)
        if self.source:
            row["source"] = self.source
        if meta:
            row["meta"] = meta
        self.seq += 1
        agg = self._totals.setdefault(row["stage"], {"n": 0, "total_s": 0.0})
        agg["n"] += 1
        agg["total_s"] += row["dur_s"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    @contextmanager
    def timed(self, stage: str, unit, **meta):
        """Time a block and record it. The yielded dict is merged into `meta` at close, so a
        caller can fill in counts it only knows once the work is done.

        An EXCEPTION still records, stamped `meta.failed`: a stage that died after eight
        minutes spent those minutes, and dropping the row would make a crashing run read as a
        fast one — the same reason §2 refuses an absence-tolerant guard."""
        t0 = time.time()
        extra: dict = {}
        try:
            yield extra
        except BaseException:
            meta.update(extra)
            meta["failed"] = True
            self.record(stage, unit, time.time() - t0, **meta)
            raise
        meta.update(extra)
        self.record(stage, unit, time.time() - t0, **meta)

    # ---- reporting ------------------------------------------------------- #
    def totals(self) -> dict:
        """`{stage: {"n": k, "total_s": x}}` for what THIS instance wrote — the roll-up that
        goes into `summary.json`. Deliberately not read back off disk: a resumed run's file
        holds the previous session's units too, and a summary that silently folded them in
        would report a wall this session never spent."""
        return {s: {"n": v["n"], "total_s": round(v["total_s"], 2)}
                for s, v in sorted(self._totals.items())}


# ------------------------------------------------------------------------- #
# Reading
# ------------------------------------------------------------------------- #
def read(run_dir) -> tuple[list[dict], dict]:
    """Every row of a run dir's stream, plus a diagnostic dict.

    Returns `(rows, diag)` where `diag` carries `n_raw`, `n_torn` (unparseable lines, which a
    kill mid-append leaves at most one of) and `n_dup` (rows dropped by the `(stage, unit)`
    dedup described in the module docstring). Rows are the DEDUPED set in first-seen order,
    each holding the LAST occurrence's fields. An absent stream returns `([], {...,
    "present": False})` — whether that is a run that predates this stream or a run whose record
    is gone is the caller's to say, and `run_profile` says it."""
    from tools import run_record

    p = Path(run_dir) / STREAM
    diag = {"present": bool(run_record.segment_paths(p)), "n_raw": 0, "n_torn": 0, "n_dup": 0}
    if not diag["present"]:
        return [], diag
    order: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], dict] = {}
    for line in run_record.iter_lines(p):
        diag["n_raw"] += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            diag["n_torn"] += 1
            continue
        k = (str(r.get("stage")), str(r.get("unit")))
        if k in by_key:
            diag["n_dup"] += 1
        else:
            order.append(k)
        by_key[k] = r
    return [by_key[k] for k in order], diag


def totals_of(rows) -> dict:
    """`{stage: {"n": k, "total_s": x}}` over already-read rows."""
    out: dict[str, dict] = {}
    for r in rows:
        agg = out.setdefault(str(r.get("stage")), {"n": 0, "total_s": 0.0})
        agg["n"] += 1
        agg["total_s"] += float(r.get("dur_s") or 0.0)
    return {s: {"n": v["n"], "total_s": round(v["total_s"], 2)} for s, v in sorted(out.items())}
