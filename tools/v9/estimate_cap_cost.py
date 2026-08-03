#!/usr/bin/env python
r"""What does the raised iteration cap cost, and how long is the full v9 re-render?

Two arms over the SAME sample of plan rows, rendered through the same binary, subcommand,
worker count and process priority the real run will use:

  OLD   `--maxiter 8000` with the per-row `maxiter` stripped — i.e. exactly what v4..v8
        rendered every tile at (a flat cap, never through `auto_maxiter`).
  NEW   the per-row `maxiter` the v9 plan carries (`auto_maxiter(fw_slot)`).

STRATIFIED BY fw DECILE, WHICH IS ALSO UNBIASED. Deciles of the corpus's own `fw`
distribution are equal-count bins by construction, so drawing the same number of locations
from each is a stratified sample AND a representative one — the projection needs no
reweighting, and the per-decile breakdown shows where the cost actually lands (it is not
uniform: the cap grows linearly in octaves, and so does the escape-time work).

ALL 24 SLOTS OF EACH SAMPLED LOCATION. The ss1/ss2 mix and the palette/geometry mix are
then exactly the plan's, so extrapolation is a single multiplication rather than a model.
A head or a stride could alias onto a subset of slot positions; taking whole locations
cannot.

This is a MEASUREMENT for the runtime estimate, not an input to a decision — the cap raise
is settled (docs/design/auto_maxiter.md).

  uv run python tools/v9/estimate_cap_cost.py                 # 60 locations, both arms
  uv run python tools/v9/estimate_cap_cost.py --locations 20  # quicker, noisier
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import paths          # noqa: E402
from tools.v9 import render_cache   # noqa: E402

PLAN = ROOT / "data" / "v9" / "plan.jsonl"
CACHE_MANIFEST = ROOT / "data" / "v9" / "cache_manifest.jsonl"
MANIFEST = ROOT / "data" / "v8" / "manifest.jsonl"
COLORMAPS = ROOT / "data" / "v9" / "colormaps.json"
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
# IMPORTED, not copied — an estimate produced at a different worker count than the run
# uses is worse than no estimate (the v8 copy had already gone stale against a 3 -> 6 bump).
WORKERS = render_cache.WORKERS
SLOTS = 24
FLAT_OLD_MAXITER = 8000        # the v4..v8 cap: v4-render-batch's `--maxiter` default
N_DECILES = 10
OUT_JSON = paths.scratch("v9_estimate", "cap_cost.json")


def render_arm(rows, tag, troot, per_row_maxiter: bool, flat: int):
    """Render `rows` and return wall seconds. `per_row_maxiter=False` strips the row's
    `maxiter`, so `--maxiter <flat>` applies to every tile — the v4..v8 behaviour."""
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["out"] = str(troot / tag / f"{i}.jpg")
        if not per_row_maxiter:
            r.pop("maxiter", None)
        out.append(r)
    pf = troot / f"{tag}.jsonl"
    pf.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    env = dict(os.environ, RAYON_NUM_THREADS=str(WORKERS))
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    t0 = time.time()
    proc = subprocess.run(
        [str(BIN), "v4-render-batch", "--plan", str(pf), "--colormaps", str(COLORMAPS),
         "--maxiter", str(flat), "--log-every", "1000000"],
        cwd=str(ROOT), env=env, creationflags=flags, capture_output=True, text=True)
    el = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f"{tag} arm failed:\n{proc.stderr[-3000:]}")
    made = sum(1 for _ in (troot / tag).glob("*.jpg"))
    if made != len(out):
        sys.exit(f"{tag} arm rendered {made}/{len(out)} tiles")
    return el


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", type=int, default=60,
                    help="sampled locations (spread evenly over fw deciles)")
    a = ap.parse_args()
    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")

    # counts read off the COMMITTED artifacts, not assumed
    plan = [json.loads(l) for l in PLAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_cache_rows = sum(1 for l in CACHE_MANIFEST.open(encoding="utf-8") if l.strip())
    locs = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_loc = defaultdict(list)
    for r in plan:
        by_loc[int(Path(r["out"]).parent.name)].append(r)
    print(f"plan            : {len(plan)} rows   cache_manifest: {n_cache_rows} rows   "
          f"manifest: {len(locs)} locations x {SLOTS} slots")
    assert len(plan) == n_cache_rows == len(locs) * SLOTS, "artifact counts disagree"

    # --- stratified (== representative) sample: equal count per fw decile ---
    order = sorted(locs, key=lambda r: float(r["fw"]))
    per = max(1, a.locations // N_DECILES)
    sample_ids, decile_of = [], {}
    for d in range(N_DECILES):
        lo, hi = d * len(order) // N_DECILES, (d + 1) * len(order) // N_DECILES
        band = order[lo:hi]
        step = max(1, len(band) // per)
        for r in band[::step][:per]:
            sample_ids.append(r["loc_id"])
            decile_of[r["loc_id"]] = d
    rows = [r for lid in sample_ids for r in by_loc[lid]]
    fam = Counter(r["fractal_type"] for r in rows)
    print(f"sample          : {len(sample_ids)} locations x {SLOTS} = {len(rows)} tiles "
          f"({per}/decile over {N_DECILES} fw deciles)")
    print(f"  families      : {dict(sorted(fam.items()))}")
    print(f"  ss mix        : sample {dict(Counter(r['ss'] for r in rows))}  "
          f"plan {dict(Counter(r['ss'] for r in plan))}")
    mits = [r["maxiter"] for r in rows]
    print(f"  new cap       : {min(mits)}..{max(mits)} (mean {sum(mits)/len(mits):.0f})   "
          f"old cap: flat {FLAT_OLD_MAXITER}")

    with tempfile.TemporaryDirectory(prefix="v9_capcost_") as td:
        troot = Path(td)
        print(f"\nrendering both arms: {WORKERS} workers, BelowNormal, real binary ...")
        # OLD first, then NEW: any thermal/desktop drift over the run then works AGAINST
        # the new arm, so the reported ratio is if anything pessimistic.
        t_old = render_arm(rows, "old", troot, per_row_maxiter=False, flat=FLAT_OLD_MAXITER)
        t_new = render_arm(rows, "new", troot, per_row_maxiter=True, flat=FLAT_OLD_MAXITER)
        # per-decile cost of the new arm, one decile at a time (cheap: same tiles again
        # would double the run, so instead time each decile's rows inside the new arm)
        per_decile = {}
        for d in range(N_DECILES):
            ids = [i for i in sample_ids if decile_of[i] == d]
            drows = [r for lid in ids for r in by_loc[lid]]
            if not drows:
                continue
            t = render_arm(drows, f"d{d}", troot, per_row_maxiter=True, flat=FLAT_OLD_MAXITER)
            per_decile[d] = {"tiles": len(drows), "wall_s": round(t, 2),
                             "s_per_tile": round(t / len(drows), 4),
                             "fw_lo": float(order[d * len(order) // N_DECILES]["fw"]),
                             "mean_maxiter": round(sum(r["maxiter"] for r in drows) / len(drows)),
                             }

    n = len(rows)
    old_pt, new_pt = t_old / n, t_new / n
    full_old, full_new = old_pt * len(plan), new_pt * len(plan)
    print(f"\n  OLD (flat {FLAT_OLD_MAXITER})  {t_old:8.1f}s   {old_pt:.4f} s/tile   "
          f"{1/old_pt:6.1f} tiles/s")
    print(f"  NEW (per-row)     {t_new:8.1f}s   {new_pt:.4f} s/tile   "
          f"{1/new_pt:6.1f} tiles/s")
    print(f"  cost ratio        {t_new/t_old:.2f}x")
    print(f"\n  FULL RE-RENDER    {len(plan)} tiles at {WORKERS} workers BelowNormal:")
    print(f"     new cap        {full_new/3600:6.2f} h   ({full_new/60:.0f} min)")
    print(f"     (old cap ref)  {full_old/3600:6.2f} h")
    print("\n  per fw decile (new cap):")
    print(f"    {'dec':>3} {'fw>=':>12} {'mean cap':>9} {'s/tile':>8}  x vs decile 0")
    base = per_decile.get(0, {}).get("s_per_tile") or new_pt
    for d, v in sorted(per_decile.items()):
        print(f"    {d:>3} {v['fw_lo']:>12.3e} {v['mean_maxiter']:>9} "
              f"{v['s_per_tile']:>8.4f}  {v['s_per_tile']/base:>5.2f}x")

    rec = {"plan_rows": len(plan), "cache_manifest_rows": n_cache_rows,
           "locations": len(locs), "slots": SLOTS, "workers": WORKERS,
           "sample_locations": len(sample_ids), "sample_tiles": n,
           "old_flat_maxiter": FLAT_OLD_MAXITER,
           "old_wall_s": round(t_old, 2), "new_wall_s": round(t_new, 2),
           "old_s_per_tile": round(old_pt, 5), "new_s_per_tile": round(new_pt, 5),
           "cost_ratio": round(t_new / t_old, 3),
           "projected_full_hours_new": round(full_new / 3600, 2),
           "projected_full_hours_old": round(full_old / 3600, 2),
           "per_fw_decile": per_decile,
           "caveat": ("each arm pays one binary start + one 77-colormap parse over "
                      f"{n} tiles instead of {len(plan)}, so both OVER-estimate slightly; "
                      "a desktop under load or a reaper restart pushes the other way.")}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nWROTE {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
