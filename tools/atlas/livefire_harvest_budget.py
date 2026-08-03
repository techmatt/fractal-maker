#!/usr/bin/env python
"""Live-fire the three halt mechanisms of `label_seeded_harvest`'s budget logic.

A HAND-RUN DRIVER, not a test — it spawns the real engine and takes ~13 minutes, which is
past even the `slow` lane. The unit-level half lives in the suite
(`test_label_seeded_harvest.py`, the budget-logic block); this is the end-to-end half the
suite cannot be: the whole run loop, with real units, a real halt and a real resume.

Each of `active_budget`, `wall_budget` and `stop_sentinel` had survived every real run
without ever firing, so each was presumed rather than tested. This drives one tiny
sink-isolated run per mechanism, deliberately triggers it, asserts the run stopped at a
STATE-CONSISTENT boundary, then resumes it with a wide budget and asserts the resume is
clean. Sequential, never concurrent: each run spawns 3 engine processes and the box's cap
is 4 (`CLAUDE.md`).

**Result, 2026-08-02** — all three fired, halted consistent, resumed clean:

    active_cap     `active_budget (est 8s > 6s left)`   13/60 seeds, 0.9m -> resume 13->60
    wall_cap       `wall_budget (est 8s > 7s left)`     13/60 seeds, 0.9m -> resume 13->60
    stop_sentinel  `stop_sentinel`                       4/60 seeds, 0.4m -> resume  4->60

Sink isolation: every write is under `scratch/budget_livefire/<case>/` (run.json,
state.json, candidates.jsonl, summary.json, view_fields/). The only reads outside it are
the committed seed pool and the screen refs.

  uv run python -u tools/atlas/livefire_harvest_budget.py
"""
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "scratch" / "budget_livefire"
HARVEST = ROOT / "tools" / "atlas" / "label_seeded_harvest.py"
WORKERS = 3          # engine PROCESSES per run; runs never overlap
LIMIT_SEEDS = 60     # a bounded slice of the 511-seed pool


def run_harvest(run_dir: Path, *, budget, wall, resume=False, limit=LIMIT_SEEDS):
    cmd = [sys.executable, "-u", str(HARVEST), "run", "--run-dir", str(run_dir),
           "--budget", str(budget), "--wall-budget", str(wall),
           "--limit-seeds", str(limit), "--workers", str(WORKERS),
           "--checkpoint-every", "1"]
    if resume:
        cmd.append("--resume")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p, time.time() - t0


def state_of(run_dir: Path):
    return json.loads((run_dir / "state.json").read_text(encoding="utf-8"))


def summary_of(run_dir: Path):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def candidate_seed_ids(run_dir: Path):
    p = run_dir / "candidates.jsonl"
    if not p.exists():
        return []
    return [json.loads(l)["seed_id"]
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_consistent(run_dir: Path, label: str) -> list:
    """The invariants that make a halt boundary a STATE-CONSISTENT one."""
    fails = []
    st, su = state_of(run_dir), summary_of(run_dir)
    done, per = set(st["done"]), st["per_seed"]
    ids = [r["seed_id"] for r in per]

    if len(ids) != len(set(ids)):
        fails.append(f"{label}: a seed appears twice in per_seed")
    if set(ids) != done:
        fails.append(f"{label}: per_seed {len(set(ids))} != done {len(done)}")
    if su["seeds_done"] != len(done):
        fails.append(f"{label}: summary seeds_done {su['seeds_done']} != done {len(done)}")
    # Every candidate row belongs to a seed the state calls done — the append-only log may
    # be a superset across a HARD kill, but a graceful halt must not leave an orphan.
    orphans = {s for s in candidate_seed_ids(run_dir) if s not in done}
    if orphans:
        fails.append(f"{label}: {len(orphans)} candidate rows from seeds not marked done")
    # active_s must be the sum of the recorded unit times, not a free-running clock.
    unit_sum = sum(r["seconds"] for r in per)
    if abs(unit_sum - st["active_s"]) > 1.0 + 0.02 * max(1.0, unit_sum):
        fails.append(f"{label}: active_s {st['active_s']:.1f} != sum(unit) {unit_sum:.1f}")
    return fails


def case(name: str, *, budget, wall, sentinel_after=None, expect: str):
    run_dir = HERE / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    print(f"\n=== {name}: expect halted_by ~ {expect!r} ===", flush=True)

    stop_thread = None
    if sentinel_after is not None:
        def _touch():
            time.sleep(sentinel_after)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "STOP").write_text("", encoding="utf-8")
            print(f"  [{sentinel_after}s] wrote STOP sentinel", flush=True)
        stop_thread = threading.Thread(target=_touch, daemon=True)
        stop_thread.start()

    p, dt = run_harvest(run_dir, budget=budget, wall=wall)
    if stop_thread:
        stop_thread.join()
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        return [f"{name}: harvest exited {p.returncode}"]

    su = summary_of(run_dir)
    fails = []
    print(f"  halted_by={su['halted_by']!r} seeds {su['seeds_done']}/{su['seeds_total']} "
          f"active {su['active_min']}m wall {su['wall_min']}m ({dt:.0f}s real)")
    if not su["halted_by"].startswith(expect):
        fails.append(f"{name}: halted_by {su['halted_by']!r}, expected {expect!r}")
    # NON-VACUITY, both ends: it must have done real work AND stopped short of the pool.
    if su["seeds_done"] == 0:
        fails.append(f"{name}: halted before any unit — the cap fired, but at a boundary "
                     f"with no state, which proves nothing about consistency")
    if su["seeds_left"] == 0:
        fails.append(f"{name}: pool exhausted — the cap did not bind")
    fails += check_consistent(run_dir, f"{name}/halt")

    # --- resume, wide open ------------------------------------------------- #
    before = state_of(run_dir)
    (run_dir / "STOP").unlink(missing_ok=True)
    p2, dt2 = run_harvest(run_dir, budget=60, wall=60, resume=True)
    if p2.returncode != 0:
        print(p2.stdout[-2000:], p2.stderr[-2000:])
        return fails + [f"{name}: resume exited {p2.returncode}"]
    after, su2 = state_of(run_dir), summary_of(run_dir)
    print(f"  resume: {len(before['done'])} -> {len(after['done'])} seeds, "
          f"halted_by={su2['halted_by']!r} ({dt2:.0f}s real)")
    if len(after["done"]) <= len(before["done"]):
        fails.append(f"{name}: resume made no progress")
    if not set(before["done"]) <= set(after["done"]):
        fails.append(f"{name}: resume LOST a completed seed")
    # the resumed run must not redo a seed the first leg finished
    redone = [r["seed_id"] for r in after["per_seed"][len(before["per_seed"]):]
              if r["seed_id"] in set(before["done"])]
    if redone:
        fails.append(f"{name}: resume re-ran {len(redone)} already-done seeds")
    if after["active_s"] < before["active_s"]:
        fails.append(f"{name}: active_s went backwards across the resume")
    fails += check_consistent(run_dir, f"{name}/resume")
    return fails


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    all_fails = []
    # Budgets are 1 minute, not less, and that lower bound is a REAL constraint rather than
    # a convenience: with no unit history `unit_estimate()` returns a stated 30 s, so a run
    # given 30 s or less refuses to start its first unit and halts at seeds_done == 0. That
    # is the "never start a unit you cannot afford" rule applied to the run's own cold-start
    # guess — correct, but it halts at a boundary with no state, which proves nothing about
    # consistency. Measured: budget=0.5 gives `active_budget (est 30s > 30s left)`, 0 seeds.
    # A minute is ~14 units at the observed ~4 s each, so the cap binds mid-run with state.
    all_fails += case("active_cap", budget=1.0, wall=20.0, expect="active_budget")
    # wall cap binds first: active budget is 20x it. Units are sequential so active ~ wall,
    # which is exactly why the two need separate runs to be distinguished at all.
    all_fails += case("wall_cap", budget=20.0, wall=1.0, expect="wall_budget")
    # neither budget can bind inside 20 minutes; the sentinel is the only way out.
    all_fails += case("stop_sentinel", budget=20.0, wall=20.0, sentinel_after=20,
                      expect="stop_sentinel")

    print("\n" + "=" * 70)
    if all_fails:
        for f in all_fails:
            print(f"FAIL  {f}")
        return 1
    print("ALL THREE MECHANISMS FIRED, HALTED CONSISTENT, AND RESUMED CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
