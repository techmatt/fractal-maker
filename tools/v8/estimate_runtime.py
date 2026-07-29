#!/usr/bin/env python
r"""Runtime estimate for the v8 aug-cache render, from a real stride sample of the real plan.

Not a model of the cost — a measurement of it. Takes every Nth row of `data/v8/plan.jsonl`,
rewrites only the `out` path into a temp dir, and renders that subset through the SAME
binary, subcommand, worker count and process priority the full run will use
(`tools/v8/render_cache.py`: `v4-render-batch`, `RAYON_NUM_THREADS=3`, BELOW_NORMAL). The
per-tile wall time that comes back is therefore directly extrapolable.

STRIDE, NOT HEAD, AND STRIDE COPRIME TO 24. The plan is emitted 24 rows per location in
(geometry, ss, palette) order, so a stride sharing a factor with 24 lands on a fixed subset
of slot positions — a stride of 500 (gcd 4) would sample only 6 of the 24 and could miss the
ss2 rows that carry 4x the iteration cost. The default stride is prime for that reason, and
the realised ss1/ss2 mix is reported against the plan's so an aliased sample is visible
rather than silent.

The sample renders to a temp dir, NOT into the cache: a cache tile must come from the run
that is auditable through `scratch/v8_render/progress.jsonl`, and a scattered pre-seed would
make the first cycle's "+N tiles" meaningless.

  uv run python tools/v8/estimate_runtime.py                  # default stride 499 (~343 tiles)
  uv run python tools/v8/estimate_runtime.py --stride 251     # denser, ~683 tiles
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

PLAN = ROOT / "data" / "v8" / "plan.jsonl"
COLORMAPS = ROOT / "data" / "v8" / "colormaps.json"
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
WORKERS = 3            # must match render_cache.WORKERS
SLOTS = 24             # rows per location in the plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=499,
                    help="sample every Nth plan row; MUST be coprime to 24 (asserted)")
    a = ap.parse_args()
    if math.gcd(a.stride, SLOTS) != 1:
        sys.exit(f"stride {a.stride} shares a factor with the {SLOTS}-row slot cycle "
                 f"(gcd {math.gcd(a.stride, SLOTS)}) — it would sample only "
                 f"{SLOTS // math.gcd(a.stride, SLOTS)} of the {SLOTS} slot positions")
    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")

    rows = [json.loads(l) for l in PLAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)
    sample = rows[::a.stride]
    plan_mix = Counter(r["ss"] for r in rows)
    samp_mix = Counter(r["ss"] for r in sample)

    print(f"plan   : {total} rows   ss1 {plan_mix[1]} ({plan_mix[1]/total:.1%}) / "
          f"ss2 {plan_mix[2]} ({plan_mix[2]/total:.1%})")
    print(f"sample : {len(sample)} rows (stride {a.stride})   "
          f"ss1 {samp_mix[1]} ({samp_mix[1]/len(sample):.1%}) / "
          f"ss2 {samp_mix[2]} ({samp_mix[2]/len(sample):.1%})")
    print(f"families in sample: {dict(sorted(Counter(r['fractal_type'] for r in sample).items()))}")

    with tempfile.TemporaryDirectory(prefix="v8_estimate_") as td:
        troot = Path(td)
        redirected = []
        for i, r in enumerate(sample):
            r = dict(r)
            r["out"] = str(troot / "tiles" / f"{i}.jpg")
            redirected.append(r)
        pf = troot / "sample.jsonl"
        pf.write_text("\n".join(json.dumps(r) for r in redirected) + "\n", encoding="utf-8")

        env = dict(os.environ, RAYON_NUM_THREADS=str(WORKERS))
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        print(f"\nrendering {len(sample)} tiles: {WORKERS} workers, BelowNormal, "
              f"the real binary and subcommand ...")
        t0 = time.time()
        proc = subprocess.run(
            [str(BIN), "v4-render-batch", "--plan", str(pf),
             "--colormaps", str(COLORMAPS), "--log-every", "100000"],
            cwd=str(ROOT), env=env, creationflags=flags, capture_output=True, text=True)
        el = time.time() - t0
        if proc.returncode != 0:
            sys.exit(f"sample render failed:\n{proc.stderr[-3000:]}")
        made = sum(1 for _ in (troot / "tiles").glob("*.jpg"))

    per_tile = el / len(sample)
    full_s = per_tile * total
    print(f"\n  rendered      : {made}/{len(sample)} tiles in {el:.1f}s")
    print(f"  per tile      : {per_tile:.4f} s  ({1/per_tile:.1f} tiles/s at {WORKERS} workers)")
    print(f"  FULL RUN      : {total} tiles -> {full_s/3600:.2f} h  ({full_s/60:.0f} min)")
    print(f"  cache size    : ~{total * 26 / 1024 / 1024:.1f} GB at ~26 KB/tile (512x288 q85)")
    print(f"  caveats       : the sample pays one binary start + one 77-colormap parse over "
          f"{len(sample)} tiles instead of {total}, so this OVER-estimates slightly; a "
          f"desktop under load or the external reaper's restarts push the other way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
