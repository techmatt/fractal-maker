#!/usr/bin/env python
r"""v11 render plan + recipe record — the DICTATED 32-tile independent-draw fan-out.

The recipe is Matt's, given whole (v11_cache_render.md §3); this module realizes it and
records what it realized. Everything it does not decide, it asserts against the committed
v9/v10 artifacts so a "same pool, same held-out set" claim is checked rather than intended.

    32 tiles/location, each an INDEPENDENT seeded draw:
      palette   uniform over the curated 76 MINUS the 8 held-out invariance palettes (68)
      geometry  shift <= 5% of the canonical fw, uniform direction; scale ~ U[0.90, 1.10]
      AA        50/50 {aliased (ss1-equivalent point sample), antialiased (lanczos3)}
      quality   uniform integer 60..95
    floor per location, drawn FROM the 32 and not added to it:
      >=2 twilight_shifted, >=2 blue_orange, >=1 exact-identity geometry

WHAT MOVED FROM v8b/v9/v10, and it is the whole fan-out. v8b is a PRODUCT — 4 palettes x 3
geometries x 2 AA levels — so a location's 24 tiles rest on 6 distinct fields and every
(palette, AA) cell sees the same three framings. v11 draws each tile independently, so the
axes are decorrelated across the corpus instead of crossed within a location, and the
palette axis widens from 4 names/location to 68 in the draw. It costs the same: under the
`crop-batch` executor every tile is a full resample+colourize either way, and the ONE
iteration pass per location is what both are paying for.

THE HELD-OUT 8 ARE v9's, BY ASSERTION. `choose_held_out` is a seeded sample and the seed
(`v9p.HELDOUT_SEED`) has not moved, but a re-derived set that merely happens to match is not
the same claim as a set held equal to the committed roster — the held-out-palette invariance
read only spans versions if the SAME eight names were never trainable in any of them. So the
set is rebuilt and then held against `data/v9/aug_roster.json` and `data/v10/aug_roster.json`.
The difference from v9's DRAWABLE pool is deliberate: v9 drew from 67 (pool minus the deploy
palette minus the held-out 8) because `twilight_shifted` was pinned onto every location
anyway; v11 draws from 68 because it is one of the 68 as well as the floor.

  uv run python tools/v11/build_plan.py [--dry-run]

Writes:
  bulk    data/v11/plan.jsonl        one row per LOCATION — `crop-batch --locations`
  durable data/v11/colormaps.json    the merged render library (asserted == v9's/v10's)
  durable data/v11/aug_recipe.json   the SMALL committed record: seed tag + every flag
The tiles and the per-tile cache manifest are written by `crop-batch` itself, both bulk.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402
import active_ckpt as ac     # noqa: E402  THE production iteration-cap policy

# v9's palette machinery is IMPORTED, never restated: the curated-pool merge and the
# held-out draw are the identity of the invariance instrument, and a second copy of either
# is a second thing that can drift.
from tools.v9 import build_plan as v9p   # noqa: E402

MANIFEST = "data/v11/manifest.jsonl"          # bulk
PLAN_OUT = "data/v11/plan.jsonl"              # bulk
CACHE_MANIFEST_OUT = "data/v11/cache_manifest.jsonl"   # bulk, written by crop-batch
CACHE_DIR = "data/v11/aug_cache"              # bulk, written by crop-batch
COLORMAPS_OUT = "data/v11/colormaps.json"     # durable
RECIPE_OUT = "data/v11/aug_recipe.json"       # durable
V9_ROSTER = "data/v9/aug_roster.json"
V10_ROSTER = "data/v10/aug_roster.json"

# --------------------------------------------------------------------------- #
# THE RECIPE, as dictated. One fixed seed tag for the whole build.
# --------------------------------------------------------------------------- #
SEED_TAG = "v11-aug-20260808"
TILES_PER_LOCATION = 32
FLOOR = (("twilight_shifted", 2), ("blue_orange", 2))
FLOOR_IDENTITY = 1
AA_LEVELS = ("aliased:point", "antialiased:lanczos3")   # the 50/50 coin
SCALE_LO, SCALE_HI = 0.90, 1.10
SHIFT_FRAC_MAX = 0.05
JPG_QUALITY_LO, JPG_QUALITY_HI = 60, 95
FIELD_SS = 2                 # adopted: the deploy-matched antialiased arm is at parity here
EXTEND = 1.2                 # == 1 + 2*shift_max + (H/W)*(scale_hi-1), exactly
TILE_W, TILE_H = 512, 288    # the cache geometry, v4..v10 unchanged
PLAN_ORDER_SEED = 20260808   # see `plan_order` — a seeded shuffle, not family order


def plan_order(rows):
    """The order `crop-batch` walks the plan in: a SEEDED SHUFFLE, not family order.

    CLAUDE.md's projection rule was earned on exactly this file: v9/v10's plans are emitted
    family-ordered with the expensive deep material contiguous and late, so the run's
    own early throughput is not a rate for the run and the v9 cache render missed its ETA by
    1.65x. A shuffled plan makes any prefix a fair sample of the whole, so the supervisor's
    reprojection from observed throughput is honest from the first minutes instead of after
    the cheap head is exhausted.

    Nothing downstream depends on plan order: a tile's identity is its `loc_id` and its slot,
    and `loc_id` is fixed by the manifest, not by this shuffle."""
    order = list(rows)
    random.Random(PLAN_ORDER_SEED).shuffle(order)
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report; write nothing")
    a = ap.parse_args()

    # ---- the palette library + the two pools ----
    library, pool_names = v9p.build_library()
    held_out = v9p.choose_held_out(pool_names)
    for roster_path in (V9_ROSTER, V10_ROSTER):
        r = json.loads((ROOT / roster_path).read_text(encoding="utf-8"))
        if r["palettes"]["held_out"] != held_out:
            raise SystemExit(
                f"held-out palette set differs from {roster_path}: the invariance read only "
                f"spans versions if the same eight names were never trainable in any of them")
    # v11's draw pool is the curated pool MINUS the held-out 8 — `twilight_shifted` included,
    # unlike v9's `drawable`, because v11 draws it rather than pinning it.
    pool = sorted(n for n in pool_names if n not in set(held_out))
    if len(pool) != len(pool_names) - len(held_out):
        raise SystemExit("the held-out set is not a subset of the curated pool")
    floor_names = [n for n, _c in FLOOR]
    lib_names = {c["name"] for c in library}
    missing = [n for n in floor_names + pool if n not in lib_names]
    if missing:
        raise SystemExit(f"palette(s) not in the merged library: {missing}")
    if v9p.LABELER_PALETTE in pool:
        raise SystemExit(f"{v9p.LABELER_PALETTE!r} is in the draw pool; it is floor-only "
                         f"(it is not one of the curated 76)")

    # ---- the population ----
    rows = [json.loads(l) for l in
            paths.bulk(MANIFEST).read_text(encoding="utf-8").splitlines() if l.strip()]

    plan, caps = [], []
    for r in rows:
        fw = float(r["fw"])
        mit = ac.auto_maxiter(fw)          # the CANONICAL frame's cap, never the extended fw
        caps.append(mit)
        row = {"loc_id": r["loc_id"], "cx": r["cx"], "cy": r["cy"], "fw": r["fw"],
               "fractal_type": r["fractal_type"], "maxiter": mit,
               "maxiter_policy": loc_mod.maxiter_policy_token()}
        for k in ("c_re", "c_im"):
            if r.get(k) is not None:
                row[k] = r[k]
        for k in loc_mod.family_param_keys(r["fractal_type"]):
            if r.get(k) is not None:
                row[k] = r[k]
        plan.append(row)

    if max(caps) >= ac.MAXITER_MAX:
        raise SystemExit(f"cap clamp {ac.MAXITER_MAX} is BINDING at {max(caps)} — the deep "
                         f"tail is being truncated; re-read docs/design/auto_maxiter.md")
    ids = [p["loc_id"] for p in plan]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate loc_id in the plan")
    if set(ids) != {r["loc_id"] for r in rows}:
        raise SystemExit("the plan's loc_id set differs from the manifest's")

    out_root = paths.bulk(CACHE_DIR)
    cmd = [
        str(ROOT / "target" / "release" / "fractal-generator.exe"), "crop-batch",
        "--locations", str(paths.bulk(PLAN_OUT)),
        "--colormaps", str(ROOT / COLORMAPS_OUT),
        "--out-root", str(out_root),
        "--manifest", str(paths.bulk(CACHE_MANIFEST_OUT)),
        "--width", str(TILE_W), "--height", str(TILE_H),
        "--field-ss", str(FIELD_SS), "--extend", str(EXTEND),
        "--tiles", str(TILES_PER_LOCATION),
        "--palette-pool", " ".join(pool),
        "--floor-palette", " ".join(f"{n}:{c}" for n, c in FLOOR),
        "--floor-identity", str(FLOOR_IDENTITY),
        "--aa", " ".join(AA_LEVELS),
        "--scale-lo", str(SCALE_LO), "--scale-hi", str(SCALE_HI),
        "--shift-frac-max", str(SHIFT_FRAC_MAX),
        "--jpg-quality-lo", str(JPG_QUALITY_LO), "--jpg-quality-hi", str(JPG_QUALITY_HI),
        "--seed-tag", SEED_TAG,
    ]

    n_tiles = len(plan) * TILES_PER_LOCATION
    print("=" * 84)
    print(f"v11 PLAN — {TILES_PER_LOCATION} independent tiles/location, seed tag {SEED_TAG!r}")
    print("=" * 84)
    print(f"  locations    : {len(plan)}  ->  {n_tiles} tiles")
    print(f"  palettes     : draw pool {len(pool)} = curated {len(pool_names)} - held out "
          f"{len(held_out)}   (held-out set == v9's and v10's)")
    print(f"  floor/loc    : {dict(FLOOR)} + {FLOOR_IDENTITY} identity geometry, drawn FROM "
          f"the {TILES_PER_LOCATION}")
    print(f"  geometry     : scale U[{SCALE_LO},{SCALE_HI}], shift U[0,{SHIFT_FRAC_MAX}]*fw, "
          f"uniform direction")
    print(f"  AA           : {list(AA_LEVELS)} (uniform per tile)")
    print(f"  quality      : uniform integer {JPG_QUALITY_LO}..{JPG_QUALITY_HI}")
    print(f"  field        : ss{FIELD_SS}, extend {EXTEND}, tile {TILE_W}x{TILE_H}")
    print(f"  cap policy   : auto_maxiter(canonical fw)  base {ac.MAXITER_BASE} k {ac.MAXITER_K} "
          f"clamp [{ac.MAXITER_MIN},{ac.MAXITER_MAX}], token "
          f"{loc_mod.maxiter_policy_token()!r}")
    print(f"  cap range    : {min(caps)}..{max(caps)}  (mean {sum(caps)/len(caps):.0f}; "
          f"clamp not binding)")
    print(f"  family mix   : {dict(sorted(Counter(p['fractal_type'] for p in plan).items()))}")
    print(f"  cache root   : {out_root}")
    print(f"\n  RENDER:\n    {' '.join(cmd)}")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    paths.durable(COLORMAPS_OUT, mkparents=True).write_text(
        json.dumps(library, indent=1), encoding="utf-8")

    plan_path = paths.bulk(PLAN_OUT)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        # THE PLAN HEADER. This file is bulk and resolves out-of-tree, away from the module
        # that wrote it, so it names its own rebuild. `crop-batch` skips `#` lines.
        f.write(f"# v11 render plan — {len(plan)} locations x {TILES_PER_LOCATION} tiles "
                f"= {n_tiles}. seed_tag={SEED_TAG}\n")
        f.write("# REBUILD: uv run python tools/v11/build_manifest.py "
                "&& uv run python tools/v11/build_plan.py\n")
        f.write("# RENDER : uv run python tools/v11/render_cache.py\n")
        f.write("# recipe + every realized flag: data/v11/aug_recipe.json (committed)\n")
        f.write(f"# row order is a seeded shuffle (seed {PLAN_ORDER_SEED}), NOT family "
                f"order — see build_plan.plan_order\n")
        for row in plan_order(plan):
            f.write(json.dumps(row) + "\n")

    recipe = {
        "corpus_version": "v11",
        "recipe": "v11-independent-32",
        "rebuild": ("uv run python tools/v11/build_manifest.py && "
                    "uv run python tools/v11/build_plan.py"),
        "render": "uv run python tools/v11/render_cache.py",
        "seed_tag": SEED_TAG,
        "supersedes_recipe": ("v8b — 4 palettes x 3 geometries x 2 AA = 24 as a PRODUCT. "
                              "v11 draws each of 32 tiles independently, so the axes are "
                              "decorrelated across the corpus rather than crossed within a "
                              "location, and the palette axis widens from 4 names per "
                              "location to a 68-name pool."),
        "fan_out": {
            "tiles_per_location": TILES_PER_LOCATION,
            "draw": "independent per tile (palette, geometry, AA level, jpg quality)",
            "floor_per_location": {n: c for n, c in FLOOR},
            "floor_identity_geometries": FLOOR_IDENTITY,
            "floor_note": "drawn FROM the 32, never added to them",
        },
        "palettes": {
            "library": COLORMAPS_OUT,
            "curated_pool": len(pool_names),
            "held_out": held_out,
            "held_out_seed": v9p.HELDOUT_SEED,
            "held_out_asserted_equal_to": [V9_ROSTER, V10_ROSTER],
            "draw_pool": pool,
            "draw_pool_size": len(pool),
            "floor_only": [n for n in floor_names if n not in pool],
            "why_68_not_v9s_67": ("v9 pinned twilight_shifted onto every location and drew "
                                  "the free slots from the pool MINUS it; v11 draws it like "
                                  "any other name, and additionally floors it at 2."),
        },
        "geometry": {"scale": [SCALE_LO, SCALE_HI], "shift_frac_max": SHIFT_FRAC_MAX,
                     "shift_direction": "uniform", "shift_units": "canonical frame width"},
        "aa": {"levels": list(AA_LEVELS), "draw": "uniform per tile (2 levels => 50/50)",
               "note": ("`aliased` is a nearest-neighbour point sample of the ss2 field — an "
                        "ss2 grid holds no ss1 sample point, so this is the closest honest "
                        "ss1 stand-in (src/crop_batch.rs module docs). `antialiased` is "
                        "lanczos3 at ratio scale*field_ss, measured at parity with the "
                        "legacy ss2+lanczos3 tile: 0/30 decision flips.")},
        "jpg_quality": {"draw": "uniform integer", "lo": JPG_QUALITY_LO, "hi": JPG_QUALITY_HI,
                        "supersedes": "flat q85 in v4..v10"},
        "field": {"field_ss": FIELD_SS, "extend": EXTEND, "tile_w": TILE_W, "tile_h": TILE_H,
                  "containment": ("extend >= 1 + 2*shift_frac_max + max(1,H/W)*(scale_hi-1) "
                                  f"= {1 + 2*SHIFT_FRAC_MAX + (TILE_H/TILE_W if TILE_H>TILE_W else 1)*(SCALE_HI-1)}"
                                  " — the flags sit exactly on the bound")},
        "maxiter": {
            "policy": "auto_maxiter(CANONICAL fw), per location",
            "token": loc_mod.maxiter_policy_token(),
            "constants": {"base": ac.MAXITER_BASE, "k": ac.MAXITER_K,
                          "min": ac.MAXITER_MIN, "max": ac.MAXITER_MAX,
                          "fw_home": float(ac.FW_HOME)},
            "range_over_plan": {"min": min(caps), "max": max(caps),
                                "mean": round(sum(caps) / len(caps), 1)},
            "policy_change_vs_v9_v10": (
                "v9/v10 paid auto_maxiter(fw_SLOT) — the scaled per-tile frame width. v11 "
                "pays the CANONICAL frame's cap once, because one field serves all 32 crops "
                "and the extended field must not re-derive a cap from its own wider fw. The "
                "difference is ~3% of the cap at the [0.90,1.10] scale draw, and it is "
                "ACCEPTED rather than corrected (v11_cache_render.md §3)."),
        },
        "plan": {"path": PLAN_OUT, "class": "bulk", "locations": len(plan),
                 "tiles": n_tiles, "row_order": f"seeded shuffle (seed {PLAN_ORDER_SEED})",
                 "why_shuffled": ("a family-ordered plan puts the expensive deep material "
                                  "contiguous and late, which is how the v9 cache render "
                                  "missed its ETA by 1.65x. A shuffled plan makes any "
                                  "prefix a fair sample, so the run reprojects honestly.")},
        "cache": {"root": CACHE_DIR, "class": "bulk",
                  "tile_manifest": CACHE_MANIFEST_OUT,
                  "layout": "<root>/<loc_id>/t<NN>__<palette>__s<scale>__sh<shift>__<aa>__q<q>.jpg"},
        "render_command": cmd,
        "family_mix": dict(sorted(Counter(p["fractal_type"] for p in plan).items())),
    }
    paths.durable(RECIPE_OUT, mkparents=True).write_text(
        json.dumps(recipe, indent=2), encoding="utf-8")

    print(f"\nWROTE {plan_path}   ({len(plan)} locations, {n_tiles} tiles planned)")
    print(f"WROTE {COLORMAPS_OUT}   ({len(library)} colormaps)")
    print(f"WROTE {RECIPE_OUT}   (committed record)")


if __name__ == "__main__":
    main()
