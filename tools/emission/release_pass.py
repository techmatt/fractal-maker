#!/usr/bin/env python
r"""release_pass.py — THE release render pass: N rows at wallpaper canon, serial or concurrent.

WHAT THIS IS. The emission driver's last stage renders each selected row once at
2560x1440 ss4 (`build_emission_diversity_v1.render_wallpaper`). After `tail_optimize`
(4b9fe5c) one such render is roughly half 7-thread Rust engine and half single-thread Python
coloring tail, so a serial pass leaves most of a 12-core box idle for most of its wall clock
regardless of which half is running. This module runs the pass as concurrent WORKER PROCESSES
and is the only thing that knows how.

THE ONE STRUCTURAL RULE: WORKERS RENDER, THE PARENT WRITES. A worker returns
`(image on disk, info block)` and appends to nothing. Every record — the auto-level stamp row,
the per-unit stage time, the caller's own manifest — is written by the parent, from `sink`,
**in plan order**, one call per task, exactly once. That is what makes the pass's records
IDENTICAL to the serial pass's rather than merely equivalent: an append-only log with N
writers has no order, and `autolevel_stamps.jsonl` / `stage_times.jsonl` are both read as
ordered streams downstream. The suppression seam is `stamp_log=False`, which turns off the
stamp WRITE and not the levelling (`deploy_tail._level_python`), so the stamp still rides back
in the info block for the parent to write.

`workers <= 1` IS THE UNTOUCHED SERIAL PATH — in-process, no pool, no pickling, and
`stamp_log=True` so the render writes its own stamp exactly where it always did. It is the
fallback, so it must not be a special case of the concurrent path; the two are checked against
each other by a real 2-row engine parity test (`test_release_pass_parity.py`, `slow`) rather
than by construction.

FAILURE AND INTERRUPTION.
  * A row that raises comes back as a failed `ReleaseResult` and is handed to `sink` like any
    other. It never reaches the pool boundary, so it cannot take the other rows down.
  * A worker that DIES (an OOM kill is the realistic cause — see the RAM note below) breaks
    the pool for every future still outstanding. That is caught once and the remaining tasks
    are rendered SERIALLY in the parent, because half a release is a worse outcome than a slow
    one. The fallback announces itself; a silent degrade to serial would read as "concurrency
    bought nothing".
  * Each worker puts ITSELF in a Windows job object with `KILL_ON_JOB_CLOSE`
    (`kill_children_on_exit`). Engine children inherit the job, so a worker killed by anything
    at all — reaper, OOM, `TerminateProcess`, pool teardown — takes its `fractal-generator.exe`
    with it. `subprocess.run` already kills its child on an EXCEPTION; this covers the case
    where there is no exception because the worker simply stopped existing.

SIZING (workers x per-engine-threads). Measured, not doctrined — see the sweep table in
`docs/design/release_concurrency.md`. `DEFAULT_RELEASE_WORKERS` and `ENGINE_THREADS_PER_WORKER`
below are that measurement's verdict and nothing else; both are within CLAUDE.md's process cap
(workers ARE concurrent `fractal-generator.exe`) and both are passed explicitly rather than
inherited.

WHAT THE SPEEDUP IS, so nobody re-derives a bigger one from core counts. **~1.5x, not the
2.5-3.5x a core-count argument gives.** The orchestration is not what limits it — measured
scheduling efficiency is 0.95-0.99 of the achievable bound for the plan. What limits it is that
running rows concurrently inflates each row's own wall by 1.4-1.9x: the engine leg already holds
7 of 12 threads, and the Python tail is MEMORY-BANDWIDTH-bound (which is exactly what
`tail_optimize` established when it measured that leg at 80 M element-ops/s), so N concurrent
tails share bandwidth instead of multiplying throughput. The remaining floor is the plan's
longest single row — one phoenix smooth row is 227 s of the serial 1045.9 s and gets SLOWER
under concurrency, so no worker count can take the pass below it.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------- #
# The adopted point on the (workers x threads) surface — MEASURED, 2026-08-13.
# --------------------------------------------------------------------------- #
# 3 workers x 7 engine threads. Chosen on wall clock over the 12 rows prod27 actually released
# (6 smooth, 2 stripe, 1 tia, 3 composite; phoenix and julia present) at 2560x1440 ss4:
# 1045.9 s serial -> 698.2 / 709.1 s (1.48-1.50x) on two runs of this point, peak RSS ~5.2 GB.
# 4 workers measured indistinguishable (703.8 s, inside the 1.6% repeat variance) for +24% peak
# RSS and a fourth concurrent engine, so 3 is the pick. Full table: docs/design/release_concurrency.md.
DEFAULT_RELEASE_WORKERS = 3

# Per-engine `RAYON_NUM_THREADS`, passed EXPLICITLY at every fan-out — the value happens to
# equal the committed single-engine default, and that is a measurement, not an inheritance.
# The obvious alternative — divide the box among the workers, 12/N, per CLAUDE.md's "size for
# the actual N" — was measured as its own arm at every N and LOST at every N (760.7 vs 742.1 at
# N=2, 749.4 vs 709.1 at N=3, 787.1 vs 703.8 at N=4). It loses because a release render is
# ~half single-threaded Python tail: a worker in its tail leaves cores idle that only an
# over-provisioned sibling engine can take, so nominal oversubscription (3x7 = 21 threads on 12
# logical cores) is what keeps the box busy. `_worker_init` sets it in the environment, which
# `corpus_common.default_engine_env` honours and never overwrites.
LOGICAL_CORES = os.cpu_count() or 12
ENGINE_THREADS_PER_WORKER = 7


def engine_threads_for(workers: int):
    """Per-engine thread count for `workers` concurrent engines; None at `workers <= 1`, where
    there is no fan-out and the render inherits the committed single-engine default."""
    return None if workers <= 1 else ENGINE_THREADS_PER_WORKER


# --------------------------------------------------------------------------- #
# The unit of work and its result.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geom:
    """The release geometry, one object so a worker call cannot drift a field."""
    w: int
    h: int
    ss: int
    filt: str


@dataclass(frozen=True)
class ReleaseTask:
    """One release row. `loc` is a `location.Location` (frozen dataclass, picklable); `out` is
    a str so the task survives a spawn pickle on any platform."""
    id: str
    loc: object
    style: str
    palette: str
    out: str


@dataclass(frozen=True)
class ReleaseResult:
    """What one row came out as. `stamp_pending` is the parent's instruction, not a status:
    True means this render was told not to write its own auto-level stamp and the caller must
    write `info["autolevel"]` itself. False means the row wrote its own (the serial path) or
    has none (a failure, a direct-trap kind, the switch off)."""
    id: str
    ok: bool
    info: dict = field(default_factory=dict)
    dur_s: float = 0.0
    error: str | None = None
    stamp_pending: bool = False

    @property
    def stamp(self):
        return (self.info or {}).get("autolevel") if self.stamp_pending else None


# --------------------------------------------------------------------------- #
# Kill-on-exit: a worker's engine child must never outlive it.
# --------------------------------------------------------------------------- #
_JOB_HANDLE = None          # module-global ONLY to keep the handle alive for process lifetime


def kill_children_on_exit() -> str:
    """Assign THIS process to a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so every
    child it spawns dies with it. Returns what actually happened, never a bool — a cleanup
    guarantee that could not be installed must not be reported as installed
    (`corpus_common.set_below_normal_priority` is the same shape for the same reason).

    The handle is held in a module global on purpose: closing it terminates the job, i.e. this
    process. It is released by process exit, which is the event we are arming for.
    """
    global _JOB_HANDLE
    if sys.platform != "win32":
        return "unavailable (posix; children are reaped by the process group)"
    if _JOB_HANDLE is not None:
        return "job:kill-on-close (already)"
    import ctypes
    from ctypes import wintypes as wt

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    # argtypes/restype are NOT optional here: `GetCurrentProcess` returns the pseudo-handle
    # (HANDLE)-1, and a default-typed ctypes call truncates it to a C int and raises.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_uint32]
    k32.SetInformationJobObject.restype = wt.BOOL
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.AssignProcessToJobObject.restype = wt.BOOL
    h = k32.CreateJobObjectW(None, None)
    if not h:
        return f"FAILED (CreateJobObject err {ctypes.get_last_error()})"
    info = _EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(h, JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        return f"FAILED (SetInformationJobObject err {ctypes.get_last_error()})"
    if not k32.AssignProcessToJobObject(h, k32.GetCurrentProcess()):
        return f"FAILED (AssignProcessToJobObject err {ctypes.get_last_error()})"
    _JOB_HANDLE = h
    return "job:kill-on-close"


# --------------------------------------------------------------------------- #
# The render, in whichever process is asking.
# --------------------------------------------------------------------------- #
def render_task(task: ReleaseTask, geom: Geom, *, stamp_log: bool) -> ReleaseResult:
    """Render one row. Never raises: a failed row is a recorded row, so one bad location
    cannot end a release pass (serial or pooled)."""
    t0 = time.time()
    try:
        from tools import colormap as cm                     # noqa: PLC0415
        from tools.emission import build_emission_diversity_v1 as BED   # noqa: PLC0415
        from tools.mining import deploy_tail as dt           # noqa: PLC0415
        info = BED.render_wallpaper(dt, cm, task.loc, task.style, task.palette,
                                    Path(task.out), geom.w, geom.h, geom.ss, geom.filt,
                                    stamp_log=stamp_log)
        return ReleaseResult(task.id, True, dict(info or {}), time.time() - t0,
                             None, not stamp_log)
    except Exception as ex:                                  # noqa: BLE001
        return ReleaseResult(task.id, False, {}, time.time() - t0, repr(ex), False)


def _pool_entry(task: ReleaseTask, geom: Geom) -> ReleaseResult:
    """The pooled unit. Module-level and picklable by name; `stamp_log` is not a parameter
    here so a worker can never be asked to write the parent's log."""
    return render_task(task, geom, stamp_log=False)


def _serial_entry(task: ReleaseTask, geom: Geom) -> ReleaseResult:
    """The in-parent unit — the pre-concurrency path exactly, stamp write included."""
    return render_task(task, geom, stamp_log=True)


def _worker_init(threads, quiet: bool) -> None:
    # `default_engine_env` honours an explicit RAYON_NUM_THREADS already in the environment and
    # never overwrites it, so setting it here IS how this pass passes `threads=` to every engine
    # launch under it without threading a parameter through `deploy_tail._run`.
    if threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(int(threads))
    import corpus_common as cc                               # noqa: PLC0415
    prio = cc.set_below_normal_priority()
    job = kill_children_on_exit()
    if not quiet:
        print(f"[release-worker {os.getpid()}] threads={os.environ.get('RAYON_NUM_THREADS')} "
              f"priority={prio} cleanup={job}", flush=True)


# --------------------------------------------------------------------------- #
# The pass.
# --------------------------------------------------------------------------- #
def run_pass(tasks, geom: Geom, *, workers: int, sink, log=print,
             pool_entry=_pool_entry, serial_entry=_serial_entry) -> dict:
    """Render `tasks` and deliver each result to `sink(task, result)` IN PLAN ORDER.

    `sink` runs in the PARENT, exactly once per task, in the order `tasks` was given —
    never concurrently with itself — so it is the only place records are written.
    `pool_entry` / `serial_entry` are the two render callables (injectable so the orchestration
    can be tested without an engine; the pooled one must be a module-level function, since a
    spawned worker resolves it by name).

    Returns a small dict of what the pass DID — worker count, engine threads, whether the
    serial fallback fired — for the caller to stamp into its own summary rather than restate.
    """
    tasks = list(tasks)
    stat = {"n": len(tasks), "workers": max(1, int(workers)),
            "engine_threads": engine_threads_for(max(1, int(workers))),
            "fell_back_serial": 0}
    if not tasks:
        return stat
    if stat["workers"] <= 1:
        for i, t in enumerate(tasks):
            res = serial_entry(t, geom)
            sink(t, res)
            log(f"[release] {i + 1}/{len(tasks)} {t.id} "
                f"{'ok' if res.ok else 'FAILED'} {res.dur_s:.1f}s (serial)")
        return stat

    def _serial_rest(rest, why):
        stat["fell_back_serial"] = len(rest)
        log(f"[release] POOL BROKEN ({why}) — rendering the remaining {len(rest)} row(s) "
            f"serially in the parent rather than dropping them")
        for t in rest:
            sink(t, serial_entry(t, geom))

    ex = ProcessPoolExecutor(max_workers=stat["workers"], mp_context=mp.get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(stat["engine_threads"], False))
    clean = False
    try:
        log(f"[release] {len(tasks)} row(s) over {stat['workers']} worker process(es) at "
            f"RAYON_NUM_THREADS={stat['engine_threads']} (serial fallback: --release-workers 1)")
        futs = [ex.submit(pool_entry, t, geom) for t in tasks]
        for i, (t, f) in enumerate(zip(tasks, futs)):
            try:
                res = f.result()
            except BrokenProcessPool as ex_:
                _serial_rest(tasks[i:], repr(ex_))
                clean = True
                return stat
            except Exception as ex_:                          # noqa: BLE001
                res = ReleaseResult(t.id, False, {}, 0.0, repr(ex_), False)
            sink(t, res)
            log(f"[release] {i + 1}/{len(tasks)} {t.id} "
                f"{'ok' if res.ok else 'FAILED'} {res.dur_s:.1f}s")
        clean = True
    finally:
        # A clean end waits for the workers; an interrupted one cancels what has not started
        # and does not block on what has — the workers' own job objects take their engines.
        ex.shutdown(wait=clean, cancel_futures=not clean)
    return stat
