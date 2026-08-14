"""The release render pass's ORCHESTRATION — order, isolation, cleanup, and the thread hand-off.

These tests drive a real spawn-based process pool and inject the render callable, because the
thing under test is the pass, not the renderer: `run_pass` takes `pool_entry`/`serial_entry` and
every fake below is module-level so a spawned worker can resolve it by name. The real renderer's
byte-identity is `test_release_pass_parity.py` (`slow`, drives the engine); a green file here
says nothing about pixels and is not meant to.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc                                    # noqa: E402
from tools.emission import release_pass as RP                 # noqa: E402
from tools.palettes import autolevel as AL                    # noqa: E402

GEOM = RP.Geom(64, 36, 1, "lanczos3")


def _tasks(n, prefix="row"):
    return [RP.ReleaseTask(id=f"{prefix}{i}", loc=None, style="smooth", palette="viridis",
                           out=f"{prefix}{i}.png") for i in range(n)]


def _collect():
    seen = []
    return seen, (lambda t, r: seen.append((t.id, r)))


# --------------------------------------------------------------------------- #
# Module-level fakes — a spawned worker unpickles these BY NAME, so they cannot be
# closures or locals. Each returns a ReleaseResult exactly as the real entry does.
# --------------------------------------------------------------------------- #
def fake_reversed_cost(task, geom):
    """Later rows finish FIRST, so completion order is the reverse of plan order and an
    in-order sink cannot be passing by accident on a pool that happens to stay ordered."""
    idx = int(task.id.replace("row", ""))
    time.sleep(0.30 * (1.0 / (idx + 1)))
    return RP.ReleaseResult(task.id, True, {}, 0.1, None, True)


def fake_row2_raises(task, geom):
    if task.id == "row2":
        return RP.ReleaseResult(task.id, False, {}, 0.0, "RuntimeError('bad location')", False)
    return RP.ReleaseResult(task.id, True, {}, 0.0, None, True)


def fake_row2_kills_the_worker(task, geom):
    if task.id == "row2":
        os._exit(1)                       # the OOM-kill shape: no exception, no result, no pool
    return RP.ReleaseResult(task.id, True, {}, 0.0, None, True)


def fake_reports_thread_env(task, geom):
    return RP.ReleaseResult(task.id, True, {"rayon": os.environ.get("RAYON_NUM_THREADS")},
                            0.0, None, True)


def fake_serial_marker(task, geom):
    return RP.ReleaseResult(task.id, True, {"where": "serial"}, 0.0, None, False)


# --------------------------------------------------------------------------- #
# 1. Plan order, exactly once.
# --------------------------------------------------------------------------- #
def test_the_sink_sees_plan_order_even_though_the_pool_completes_in_reverse():
    tasks = _tasks(6)
    seen, sink = _collect()
    stat = RP.run_pass(tasks, GEOM, workers=3, sink=sink, log=lambda m: None,
                       pool_entry=fake_reversed_cost)
    assert [i for i, _ in seen] == [t.id for t in tasks]
    assert stat["n"] == 6 and stat["workers"] == 3 and stat["fell_back_serial"] == 0


def test_an_empty_plan_starts_no_pool_and_still_reports_the_config():
    seen, sink = _collect()
    stat = RP.run_pass([], GEOM, workers=4, sink=sink, log=lambda m: None,
                       pool_entry=fake_reversed_cost)
    assert seen == []
    assert stat == {"n": 0, "workers": 4, "engine_threads": RP.engine_threads_for(4),
                    "fell_back_serial": 0}


# --------------------------------------------------------------------------- #
# 2. Failure isolation, both shapes.
# --------------------------------------------------------------------------- #
def test_a_failed_row_is_recorded_and_the_rest_still_render():
    tasks = _tasks(5)
    seen, sink = _collect()
    RP.run_pass(tasks, GEOM, workers=2, sink=sink, log=lambda m: None,
                pool_entry=fake_row2_raises)
    by_id = dict(seen)
    assert len(seen) == 5
    assert not by_id["row2"].ok and "bad location" in by_id["row2"].error
    assert all(by_id[f"row{i}"].ok for i in (0, 1, 3, 4))


def test_a_dead_worker_falls_back_to_serial_instead_of_dropping_the_rest():
    """`os._exit` in a worker breaks the pool for every OUTSTANDING future, which is most of
    the plan — the run must still produce them."""
    tasks = _tasks(6)
    seen, sink = _collect()
    stat = RP.run_pass(tasks, GEOM, workers=2, sink=sink, log=lambda m: None,
                       pool_entry=fake_row2_kills_the_worker,
                       serial_entry=fake_serial_marker)
    assert [i for i, _ in seen] == [t.id for t in tasks]      # every row, still in plan order
    assert stat["fell_back_serial"] > 0
    # and the fallback rows really came from the SERIAL entry, not a silently retried pool one
    fell = [r for _, r in seen if r.info.get("where") == "serial"]
    assert len(fell) == stat["fell_back_serial"]


# --------------------------------------------------------------------------- #
# 3. workers=1 is the serial path — no pool, no pickling, the pooled entry unreachable.
# --------------------------------------------------------------------------- #
def test_workers_1_never_touches_the_pooled_entry():
    tasks = _tasks(3)
    seen, sink = _collect()

    def _boom(task, geom):
        raise AssertionError("the pooled entry ran at --release-workers 1")

    stat = RP.run_pass(tasks, GEOM, workers=1, sink=sink, log=lambda m: None,
                       pool_entry=_boom, serial_entry=fake_serial_marker)
    assert [i for i, _ in seen] == [t.id for t in tasks]
    assert stat["engine_threads"] is None          # inherit the committed single-engine default
    assert all(r.info["where"] == "serial" for _, r in seen)


# --------------------------------------------------------------------------- #
# 4. The thread hand-off actually reaches the workers (and is NOT the single-engine default).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [2, 3, 4])
def test_each_worker_runs_its_engine_at_the_fan_out_thread_count(workers, monkeypatch):
    seen, sink = _collect()
    RP.run_pass(_tasks(workers * 2), GEOM, workers=workers, sink=sink, log=lambda m: None,
                pool_entry=fake_reports_thread_env)
    assert {r.info["rayon"] for _, r in seen} == {str(RP.engine_threads_for(workers))}
    # NON-VACUITY. The adopted count happens to EQUAL the committed single-engine default, so
    # "the worker reports 7" is also what an unset environment would produce by inheritance —
    # the assertion above cannot tell the hand-off from its own absence. Re-run against a value
    # the default could not supply: what is being pinned is that the fan-out SETS the number.
    monkeypatch.setattr(RP, "ENGINE_THREADS_PER_WORKER", cc.DEFAULT_ENGINE_THREADS + 1)
    seen2, sink2 = _collect()
    RP.run_pass(_tasks(2), GEOM, workers=workers, sink=sink2, log=lambda m: None,
                pool_entry=fake_reports_thread_env)
    assert {r.info["rayon"] for _, r in seen2} == {str(cc.DEFAULT_ENGINE_THREADS + 1)}


def test_the_fan_out_thread_count_is_a_measured_constant_not_a_division_of_the_box():
    """The 12/N instinct was measured as its own arm at N in {2,3,4} and lost at every N
    (release_concurrency.md). Pinned here so a later reader restoring the "obvious" rule has to
    delete a test that names the measurement rather than edit a bare number."""
    assert RP.engine_threads_for(1) is None          # no fan-out -> nothing to state
    assert RP.ENGINE_THREADS_PER_WORKER == 7
    for n in (2, 3, 4, 6):
        assert RP.engine_threads_for(n) == RP.ENGINE_THREADS_PER_WORKER
        assert RP.engine_threads_for(n) != max(1, RP.LOGICAL_CORES // n)


# --------------------------------------------------------------------------- #
# 5. Stamp routing: a worker NEVER writes the parent's log; the serial path always does.
# --------------------------------------------------------------------------- #
def test_the_pooled_render_is_told_not_to_write_its_stamp_and_the_serial_one_is(monkeypatch):
    from tools.emission import build_emission_diversity_v1 as BED
    calls = []

    def _fake_render_wallpaper(dt, cm, loc, style, palette, out, w, h, ss, filt, *,
                               stamp_log=True):
        calls.append(stamp_log)
        return {"autolevel": {"acted": True, "marker": style}}

    monkeypatch.setattr(BED, "render_wallpaper", _fake_render_wallpaper)
    t = _tasks(1)[0]
    pooled = RP._pool_entry(t, GEOM)
    serial = RP._serial_entry(t, GEOM)
    assert calls == [False, True]
    # `.stamp` is what the parent writes — present exactly for the render that suppressed its own
    assert pooled.stamp_pending and pooled.stamp == {"acted": True, "marker": "smooth"}
    assert not serial.stamp_pending and serial.stamp is None


def test_a_render_that_raises_is_a_failed_result_not_an_exception(monkeypatch):
    from tools.emission import build_emission_diversity_v1 as BED
    monkeypatch.setattr(BED, "render_wallpaper",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine died")))
    res = RP._pool_entry(_tasks(1)[0], GEOM)
    assert not res.ok and "engine died" in res.error and res.stamp is None


def test_append_stamp_is_the_one_writer_both_paths_go_through(tmp_path):
    AL.append_stamp(tmp_path, "a.png", {"acted": False})
    AL._log_stamp(tmp_path, "b.png", {"acted": True})
    AL.append_stamp(None, "never.png", {"acted": True})       # no log_dir -> no row, no crash
    rows = [json.loads(l) for l in
            (tmp_path / AL.STAMP_LOG).read_text(encoding="utf-8").splitlines()]
    assert rows == [{"key": "a.png", "autolevel": {"acted": False}},
                    {"key": "b.png", "autolevel": {"acted": True}}]


# --------------------------------------------------------------------------- #
# 6. Two workers must not derive one temp name.
# --------------------------------------------------------------------------- #
def test_two_processes_derive_different_field_dump_names():
    """A disposable field dump is written and `finally`-unlinked inside one render, so two
    workers rendering the SAME (location, mode, geometry) — which is what a location released
    under two palettes is — must not land on one file. Whether they collide in any given run is
    a race, so this is a MECHANISM test rather than a provoked one: a test that tried to force
    the corruption would be green most of the time and would be evidence of nothing
    (`verification_practice.md` §1.11). `test_release_pass_parity.py` plans exactly that pair
    of rows and would eventually catch it; this catches it deterministically.
    """
    from tools.mining import deploy_tail as dt
    mine = dt.field_tmp_token()
    other = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, r'{ROOT}');"
         "from tools.mining import deploy_tail as dt; print(dt.field_tmp_token())"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert mine and other and mine != other, (mine, other)


# --------------------------------------------------------------------------- #
# 7. The driver's glue: resume, plan order, and the parent's three writes.
# --------------------------------------------------------------------------- #
def test_render_release_resumes_complete_pngs_and_returns_plan_order(tmp_path, monkeypatch):
    """`EmissionDiversity.render_release` is what the sweep does NOT exercise, and it is where
    an ordering bug would land: the report and the release sheet lay tiles out in the order this
    list arrives, and a concurrent pass finishes out of order. Driven with a fake render so the
    glue is tested without an engine.
    """
    import numpy as np
    from PIL import Image
    from tools.emission import build_emission_diversity_v1 as BED
    from tools import stage_times as stimes

    def _png(p, ok=True):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(p)
        if not ok:                                  # truncate -> must be re-rendered, not reused
            p.write_bytes(p.read_bytes()[:20])

    eng = BED.EmissionDiversity.__new__(BED.EmissionDiversity)
    eng.release_dir = tmp_path / "release"
    eng.release_dir.mkdir()
    eng.rel_w, eng.rel_h, eng.rel_ss, eng.rel_filt = 64, 36, 1, "lanczos3"
    eng.release_workers = 1
    eng.stage_times = stimes.StageTimes(tmp_path / "telemetry")
    ids = ["a", "b", "c", "d"]
    eng.by_id = {f"loc_{i}": {"id": f"loc_{i}"} for i in ids}
    selected = [{"_rec": {"id": i, "location_id": f"loc_{i}", "render_style": "smooth",
                          "palette": "viridis"}} for i in ids]
    _png(eng.release_dir / "b.png")                 # complete -> resumed
    _png(eng.release_dir / "c.png", ok=False)       # truncated -> re-rendered

    rendered = []

    def _fake(dt, cm, loc, style, palette, out, w, h, ss, filt, *, stamp_log=True):
        rendered.append(Path(out).stem)
        assert not stamp_log, "the pass must not let a render write the parent's stamp log"
        _png(Path(out))
        return {"autolevel": {"acted": False, "row": Path(out).stem}}

    monkeypatch.setattr(BED, "render_wallpaper", _fake)
    monkeypatch.setattr(BED.D, "location_of", lambda row: row["id"])
    # workers=1 would take the serial entry, which writes its own stamp; the parent-writes
    # contract is what this test is for, so drive the pooled entry in-process.
    def _pooled_inline(tasks, geom, *, workers, sink, log):
        for t in tasks:
            sink(t, BED.RP._pool_entry(t, geom))
        return {"n": len(tasks), "workers": workers, "fell_back_serial": 0}

    monkeypatch.setattr(BED.RP, "run_pass", _pooled_inline)

    out = eng.render_release(selected)
    assert [i for i, _ in out] == ids                        # PLAN order, resumed rows included
    assert sorted(rendered) == ["a", "c", "d"]               # 'b' resumed, 'c' was truncated
    stamps = [json.loads(l) for l in
              (eng.release_dir / AL.STAMP_LOG).read_text(encoding="utf-8").splitlines()]
    assert [s["key"] for s in stamps] == ["a.png", "c.png", "d.png"]
    rows = [json.loads(l) for l in
            (tmp_path / "telemetry" / stimes.STREAM).read_text(encoding="utf-8").splitlines()]
    assert [r["unit"] for r in rows] == ["a", "c", "d"]
    assert eng.release_stat["n_resumed"] == 1


# --------------------------------------------------------------------------- #
# 8. Cleanup — a worker's engine child cannot outlive it (verification_practice §11).
# --------------------------------------------------------------------------- #
_CHILD_SRC = r"""
import subprocess, sys, time
sys.path.insert(0, r"{root}")
from tools.emission import release_pass as RP
print("JOB " + RP.kill_children_on_exit(), flush=True)
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print("GRANDCHILD %d" % p.pid, flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def test_a_hard_killed_worker_takes_its_engine_child_with_it(tmp_path):
    """The reaper shape: the worker is TerminateProcess'd, so it gets no exception and runs no
    cleanup of its own. Without the job object the grandchild survives as an orphan — which for
    the real pass is a `fractal-generator.exe` holding gigabytes and a CPU."""
    src = tmp_path / "worker.py"
    src.write_text(_CHILD_SRC.format(root=str(ROOT)), encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(src)], stdout=subprocess.PIPE, text=True)
    try:
        job = p.stdout.readline().strip()
        gpid = int(p.stdout.readline().strip().split()[1])
        assert job == "JOB job:kill-on-close", job
        assert _alive(gpid)                       # prove-it-red: the grandchild IS running
        p.kill()                                  # TerminateProcess — no handler, no unwind
        p.wait(timeout=30)
        for _ in range(100):                      # the job kill is asynchronous
            if not _alive(gpid):
                break
            time.sleep(0.1)
        assert not _alive(gpid), f"grandchild {gpid} outlived the worker"
    finally:
        if p.poll() is None:
            p.kill()
