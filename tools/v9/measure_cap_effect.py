#!/usr/bin/env python
r"""What does the cap raise actually CHANGE, in pixels, on the corpus render path?

The cap measurement that motivated the raise (32 atoms, `radial_rings` walked up a cap
ladder — docs/design/auto_maxiter.md) was taken on the **smooth field dump**
(`render-one --dump-field`, `tools/orbital/field_metrics.py`). The corpus render path is a
different surface: `v4-render-batch` colours through `generate::color_params()` — smooth
channel at density 0.004, interior black, sqrt trap curve, q85 JPEG at 512x288. A cap that
moves a field statistic does not automatically move the JPEG a classifier reads.

So this measures the thing that actually matters for a retrain: **how many pixels of the
cached tile change**, per fw decile, across the three caps in play:

    OLD-AUTO   auto_maxiter(fw) at base 500  / clamp 8000   — the LABEL-CROP + deploy cap
    FLAT-8000  v4-render-batch's --maxiter default          — what every v4..v8 TILE used
    NEW-AUTO   auto_maxiter(fw) at base 4000 / clamp 67000  — the adopted policy

Three deltas, three different questions:

  FLAT8000 -> NEW-AUTO   what the v9 cache re-render buys the TRAINING inputs.
  OLD-AUTO -> NEW-AUTO   what the raise buys the DEPLOY / label-crop path.
  OLD-AUTO -> FLAT8000   how far the training tiles were already ahead of the crops the
                         human actually judged — the pre-existing train/label gap.

Stratified over fw deciles of the real manifest, which for equal-count bins is also
unbiased, so the pooled number is the corpus number and the per-decile column shows where
the effect lives.

  uv run python tools/v9/measure_cap_effect.py                 # 40 locations
  uv run python tools/v9/measure_cap_effect.py --locations 80
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
import active_ckpt as ac   # noqa: E402
import paths               # noqa: E402

PLAN = ROOT / "data/v9/plan.jsonl"
MANIFEST = ROOT / "data/v8/manifest.jsonl"
COLORMAPS = ROOT / "data/v9/colormaps.json"
BIN = ROOT / "target/release/fractal-generator.exe"
OUT_JSON = paths.scratch("v9_estimate", "cap_effect.json")
N_DECILES = 10
FLAT_OLD = 8000
# The superseded policy, for the OLD-AUTO arm.
OLD_BASE, OLD_K, OLD_MIN, OLD_MAX = 500, 0.30, 200, 8000
# A JPEG q85 channel difference at or below this is encoder noise, not a render change.
NOISE_TOL = 2


def old_auto_maxiter(fw: float) -> int:
    lz = math.log2(3.0 / fw) if fw > 0 else 0.0
    return int(max(OLD_MIN, min(OLD_MAX, OLD_BASE * (1.0 + OLD_K * lz))))


def render(rows, troot, tag):
    pf = troot / f"{tag}.jsonl"
    pf.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    p = subprocess.run(
        [str(BIN), "v4-render-batch", "--plan", str(pf), "--colormaps", str(COLORMAPS),
         "--log-every", "1000000"],
        cwd=str(ROOT), capture_output=True, text=True,
        env=dict(os.environ, RAYON_NUM_THREADS="3"),
        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    if p.returncode != 0:
        sys.exit(f"{tag} render failed:\n{p.stderr[-2000:]}")


def diff(a_path, b_path):
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.int16)
    d = np.abs(a - b)
    return float((d.max(axis=2) > NOISE_TOL).mean()), float(d.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", type=int, default=40)
    a = ap.parse_args()
    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN}")

    plan = [json.loads(l) for l in PLAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    # the deploy-canonical slot of each location: twilight_shifted, identity geometry, ss2
    canon = {}
    for r in plan:
        name = Path(r["out"]).name
        if name.startswith("twilight_shifted__id__") and name.endswith("ss2.jpg"):
            canon[int(Path(r["out"]).parent.name)] = r
    locs = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    order = sorted(locs, key=lambda r: float(r["fw"]))
    per = max(1, a.locations // N_DECILES)
    picked = []
    for d in range(N_DECILES):
        lo, hi = d * len(order) // N_DECILES, (d + 1) * len(order) // N_DECILES
        band = order[lo:hi]
        step = max(1, len(band) // per)
        for r in band[::step][:per]:
            if r["loc_id"] in canon:
                picked.append((d, canon[r["loc_id"]]))
    print(f"sample: {len(picked)} deploy-canonical tiles "
          f"(twilight_shifted, identity geometry, ss2), {per}/decile")
    print(f"noise tolerance: a channel delta <= {NOISE_TOL} at q85 is encoder noise\n")

    arms = {"old_auto": lambda fw: old_auto_maxiter(fw),
            "flat8000": lambda fw: FLAT_OLD,
            "new_auto": lambda fw: int(ac.auto_maxiter(fw))}
    with tempfile.TemporaryDirectory(prefix="v9_capeffect_") as td:
        troot = Path(td)
        paths_by_arm = {}
        for tag, cap_of in arms.items():
            rows = []
            for i, (_d, r) in enumerate(picked):
                rr = dict(r)
                rr["maxiter"] = cap_of(float(r["fw"]))
                rr["out"] = str(troot / tag / f"{i}.jpg")
                rows.append(rr)
            (troot / tag).mkdir(parents=True, exist_ok=True)
            render(rows, troot, tag)
            paths_by_arm[tag] = [r["out"] for r in rows]
            print(f"  rendered {tag}: caps "
                  f"{min(r['maxiter'] for r in rows)}..{max(r['maxiter'] for r in rows)}")

        comparisons = [("flat8000", "new_auto", "TRAINING inputs (what the v9 cache buys)"),
                       ("old_auto", "new_auto", "DEPLOY / label-crop path"),
                       ("old_auto", "flat8000", "pre-existing train-vs-label gap")]
        rec = {"n": len(picked), "noise_tol": NOISE_TOL, "comparisons": {}}
        for lhs, rhs, what in comparisons:
            per_dec = {}
            fr_all, md_all = [], []
            for i, (d, r) in enumerate(picked):
                fr, md = diff(paths_by_arm[lhs][i], paths_by_arm[rhs][i])
                per_dec.setdefault(d, []).append((fr, md, float(r["fw"])))
                fr_all.append(fr)
                md_all.append(md)
            print(f"\n--- {lhs} -> {rhs}   [{what}] ---")
            print(f"  {'dec':>3} {'fw>=':>11} {'%px changed':>12} {'mean|d|':>8} "
                  f"{'n tiles moved':>14}")
            dec_out = {}
            for d in sorted(per_dec):
                v = per_dec[d]
                frs = [x[0] for x in v]
                moved = sum(1 for x in frs if x > 0.0005)
                print(f"  {d:>3} {min(x[2] for x in v):>11.2e} "
                      f"{100*float(np.mean(frs)):>11.3f}% "
                      f"{float(np.mean([x[1] for x in v])):>8.4f} {moved:>10}/{len(v)}")
                dec_out[d] = {"fw_lo": min(x[2] for x in v),
                              "mean_frac_changed": round(float(np.mean(frs)), 6),
                              "tiles_moved": moved, "n": len(v)}
            moved_all = sum(1 for x in fr_all if x > 0.0005)
            print(f"  POOLED: {100*float(np.mean(fr_all)):.3f}% of pixels changed, "
                  f"{moved_all}/{len(fr_all)} tiles moved at all")
            rec["comparisons"][f"{lhs}->{rhs}"] = {
                "what": what,
                "pooled_mean_frac_changed": round(float(np.mean(fr_all)), 6),
                "pooled_max_frac_changed": round(float(np.max(fr_all)), 6),
                "tiles_moved": moved_all, "n": len(fr_all),
                "per_decile": dec_out}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nWROTE {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
