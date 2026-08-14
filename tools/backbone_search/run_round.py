#!/usr/bin/env python
"""Run a round of backbone arms SEQUENTIALLY (the GPU is exclusive) and resumably.

One process at a time by construction — two arms sharing an 8 GB card would either OOM or
make both wall clocks meaningless, and wall clock is a reported column. Each arm is a
subprocess at BELOW_NORMAL priority so the desktop stays usable across a ~10 h round.

Resumable at arm granularity: an arm whose record metrics.json already exists is SKIPPED,
so a killed round is relaunched with the same command. (Within an arm, `train_resumable`'s
per-epoch snapshot already costs at most one epoch.)

Order is cheapest-first with the CONTROL first: the control validates the harness before
10 GPU-hours are spent, and an early cheap arm surfaces a crash that would otherwise
appear four hours in.

  uv run python tools/backbone_search/run_round.py --round 1
  uv run python tools/backbone_search/run_round.py --round 2 --arms convnextv2_tiny --seeds 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

import paths  # noqa: E402
from backbone_search.arms import ARMS, ARMS_BY_NAME, CONTROL  # noqa: E402
from corpus_common import default_creationflags  # noqa: E402

# Cheapest-first by the cost smoke, control first. A name here that is not in ARMS is a
# stale order, so the list is checked rather than trusted.
ORDER = ["mnv4_conv_medium", "mnv4_hybrid_medium", "mnv4_conv_large", "fastvit_sa12",
         "vit_small_p16", "effnetv2_s"]
assert set(ORDER) == set(ARMS_BY_NAME) and ORDER[0] == CONTROL.name, \
    "ORDER is stale against arms.ARMS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    a = ap.parse_args()
    names = a.arms or ORDER
    plan = [(ARMS_BY_NAME[n], s) for n in names for s in a.seeds]
    log_dir = paths.scratch("backbone_search", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"round {a.round}: {len(plan)} run(s)")
    t_round = time.time()
    for arm, seed in plan:
        rec = arm.record_dir(seed) / "metrics.json"
        if rec.exists():
            print(f"[skip] {arm.name} s{seed} — record exists ({rec})", flush=True)
            continue
        log = log_dir / f"{arm.name}_s{seed}.log"
        print(f"[run ] {arm.name} s{seed} -> {log}", flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as f:
            rc = subprocess.run(
                [sys.executable, str(ROOT / "tools/backbone_search/train_arm.py"),
                 "--arm", arm.name, "--seed", str(seed), "--round", str(a.round)],
                cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                creationflags=default_creationflags()).returncode
        print(f"[{'done' if rc == 0 else 'FAIL'}] {arm.name} s{seed} rc={rc} "
              f"{(time.time()-t0)/3600:.2f}h  (round {(time.time()-t_round)/3600:.2f}h)",
              flush=True)
    print(f"round {a.round} complete in {(time.time()-t_round)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
