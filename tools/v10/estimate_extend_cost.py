#!/usr/bin/env python
r"""What will the v10 cache extension cost in wall clock, and in what ORDER?

The 30,408 appended tiles are not the same price as the 170,760 already on disk: the
appended locations are maneuver views and run a mean cap of 23,308 iterations against the
prefix's 15,690 — ~1.49x — so v9's measured 0.1218 s/tile is a floor, not an estimate.

CLAUDE.md's projection rule is the whole design of this script. **A sample unbiased for
mean per-tile cost is NOT unbiased for a run whose expensive work is contiguous.** The
plan is emitted in manifest order — family, then coordinates — so the deep bulk is
CLUSTERED, and a `fw`-decile sample would produce a mean-cost number that is not an ETA.
(That is exactly how the v9 cache render missed by 1.65x.) So the sample here is
**contiguous blocks in run order**: `--blocks` evenly spaced windows of consecutive plan
rows over the appended region, each timed as the executor will actually meet it. The
per-block rates are reported individually, because their SPREAD is the thing a single mean
hides.

Renders into a temp dir, never the cache — an estimate that populated the real tree would
make the run it is estimating shorter, which is the one thing an estimate must not do.

  uv run python tools/v10/estimate_extend_cost.py [--blocks 8] [--block-size 48]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

PLAN = ROOT / "data/v10/plan.jsonl"
COLORMAPS = ROOT / "data/v10/colormaps.json"
BIN = ROOT / "target/release/fractal-generator.exe"
OUT = "v10_extend_estimate.json"
WORKERS = 6          # the render supervisor's setting — estimate under the real shape
PREFIX_MAX_LOC_ID = 7140


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=48,
                    help="consecutive plan rows per block (48 = two locations)")
    a = ap.parse_args()
    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN}")

    rows = [json.loads(l) for l in PLAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    new = [r for r in rows if int(Path(r["out"]).parent.name) > PREFIX_MAX_LOC_ID]
    n = len(new)
    print(f"appended plan rows: {n}   sampling {a.blocks} contiguous blocks of "
          f"{a.block_size} in RUN ORDER")

    starts = [min(int(i * n / a.blocks), n - a.block_size) for i in range(a.blocks)]
    results = []
    with tempfile.TemporaryDirectory(prefix="v10_est_") as td:
        troot = Path(td)

        # Fixed per-invocation cost (binary load, colormap bake, plan parse). The REAL run
        # is one process over the whole plan and pays this once; a block of 48 pays it in
        # full, so leaving it in would inflate every block's rate by the same constant and
        # the projection with it. Measured, not assumed: three 1-tile runs, take the min.
        cheap = min(new, key=lambda r: int(r["maxiter"]))
        ov = []
        for k in range(3):
            q = dict(cheap)
            q["out"] = (troot / f"ov{k}.jpg").as_posix()
            pf = troot / f"ov{k}.jsonl"
            pf.write_text(json.dumps(q) + "\n", encoding="utf-8")
            t0 = time.time()
            subprocess.run([str(BIN), "v4-render-batch", "--plan", str(pf),
                            "--colormaps", str(COLORMAPS), "--log-every", "100000"],
                           cwd=str(ROOT), capture_output=True, text=True)
            ov.append(time.time() - t0)
        overhead = min(ov)
        print(f"per-invocation overhead: {overhead:.2f}s (min of 3 single-tile runs) — "
              f"subtracted from every block; the real run pays it once")

        for bi, s in enumerate(starts):
            block = new[s:s + a.block_size]
            plan = []
            for j, r in enumerate(block):
                q = dict(r)
                q["out"] = (troot / f"b{bi}_{j}.jpg").as_posix()
                plan.append(q)
            pf = troot / f"b{bi}.jsonl"
            pf.write_text("\n".join(json.dumps(x) for x in plan) + "\n", encoding="utf-8")
            env = dict(**__import__("os").environ, RAYON_NUM_THREADS=str(WORKERS))
            t0 = time.time()
            proc = subprocess.run(
                [str(BIN), "v4-render-batch", "--plan", str(pf),
                 "--colormaps", str(COLORMAPS), "--log-every", "100000"],
                cwd=str(ROOT), env=env, capture_output=True, text=True,
                creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
            el = time.time() - t0
            if proc.returncode != 0:
                sys.exit(f"block {bi} render failed:\n{proc.stderr[-2000:]}")
            caps = [int(x["maxiter"]) for x in block]
            net = max(el - overhead, 1e-6)
            rec = {"block": bi, "plan_offset": s, "tiles": len(block),
                   "wall_s": round(el, 2), "net_wall_s": round(net, 2),
                   "s_per_tile": round(net / len(block), 4),
                   "mean_maxiter": round(sum(caps) / len(caps)),
                   "fractal_type": block[0]["fractal_type"]}
            results.append(rec)
            print(f"  block {bi} @{s:>6}  {rec['fractal_type']:<12} cap~{rec['mean_maxiter']:>6}  "
                  f"{rec['wall_s']:>6.2f}s  {rec['s_per_tile']:.4f} s/tile", flush=True)

    # Prefix-weighted: each block stands for the stretch of the plan around it, which is
    # what makes this a run projection rather than a mean of per-tile costs.
    seg = n / len(results)
    total_s = sum(r["s_per_tile"] * seg for r in results)
    rates = [r["s_per_tile"] for r in results]
    out = {
        "appended_tiles": n, "workers": WORKERS,
        "per_invocation_overhead_s": round(overhead, 2),
        "blocks": results,
        "s_per_tile_min": min(rates), "s_per_tile_max": max(rates),
        "s_per_tile_spread_ratio": round(max(rates) / min(rates), 2),
        "projected_wall_s": round(total_s, 1),
        "projected_wall_h": round(total_s / 3600, 2),
        "basis": ("prefix-weighted over contiguous run-order blocks: each block's rate is "
                  "applied to the stretch of the plan it sits in, not averaged. This is an "
                  "ETA, not a mean per-tile cost."),
        "v9_reference_s_per_tile": 0.1218,
        "caveat": ("REPROJECT from the run's own observed throughput once it starts; do "
                   "not restate this number while the rate visibly moves."),
    }
    p = paths.scratch("v10_render", OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  s/tile over blocks : {min(rates):.4f}..{max(rates):.4f} "
          f"(spread {out['s_per_tile_spread_ratio']}x; v9 whole-corpus 0.1218)")
    print(f"  PROJECTED WALL     : {out['projected_wall_h']:.2f} h for {n} tiles "
          f"at {WORKERS} workers")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
