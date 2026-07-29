#!/usr/bin/env python
r"""v8 render plan + cache manifest — the REVISED 24-slot augmentation recipe.

    4 palettes  x  3 geometric samples  x  2 AA levels {ss1 box, ss2 lanczos3}  = 24

WHY THE RECIPE CHANGED (recipe "v8b"; the first v8 recipe rendered 10,538 of 171,384 tiles
and was abandoned). The 24-crop fan-out dump (`dump_fanout.py`) showed the framing axis was
augmenting far too hard: the old recipe shifted by a FIXED 0.4 of the slot's frame width at
scales {0.7, 1.0, 1.3}, so the `s1.3 shifted` tile sat 0.52 base-frame-widths off centre at
1.3x the frame — far enough that the window walks clean off the structure. That tile still
carried the location's class-3 label. The corruption is ASYMMETRIC: an empty crop labeled 1
is roughly true, an empty crop labeled 3 or 4 is a lie, and 3/4 are exactly the rare classes
v8 exists to learn. It also fights deploy, which reframes to CENTRE content before scoring.

So geometry tightens and the freed budget goes to palette:

  * **shift 0.4 (fixed) -> uniform in [0, 0.10] of frame width**, uniform direction. 4x
    smaller at the cap, and now a magnitude the deploy composition actually spans.
  * **scale {0.7, 1.0, 1.3} -> uniform in [0.90, 1.10]**, and the IDENTITY framing (dead
    centre, scale exactly 1.0) is present in EVERY (palette, AA) cell, so the real deploy
    composition is always in front of the network rather than being 1 of 6 framings.
  * **2 palettes -> 4.** `twilight_shifted` (deploy-matched, the pinned scoring instrument)
    and `blue_orange` (the map Matt's labels were actually formed on — closing the
    judge/model presentation gap) on every location, plus 2 drawn per location from the
    curated location-corpus pool.

The OLD magnitudes, for the record (they were never written down, only looked at):
    scales      {0.7, 1.0, 1.3}                       (+-30%, three fixed values)
    shift_frac  0.4 of the SLOT frame width, fixed    (so 0.28 / 0.40 / 0.52 base widths)
    direction   deterministic schedule, angle = 2*pi*(pal_idx*3 + scale_idx)/(n_pal*3)
    AA          {ss1 box, ss2 lanczos3}               (unchanged)

PALETTES ARE DRAWN PER LOCATION, NOT PER CROP. All 4 palettes of a location share that
location's 3 geometries, so a location has only 3 x 2 = 6 DISTINCT (viewport, ss) fields and
each field is used by 4 tiles. Independent per-crop draws would make all 24 tiles distinct
fields. Diversity lives at the corpus level instead: 7,141 locations x 2 free slots over a
67-name drawable pool puts every pool palette in front of the network ~213 times.

  NOTE, and it is measured rather than assumed (see `--measure-marginal`): the executor
  `v4-render-batch` does NOT currently exploit that sharing — it treats every plan row as an
  independent iterate+shade, so a palette slot costs the same as a geometry slot today. The
  sharing is a property of the PLAN, available to a recolor-batching executor; the recipe is
  cost-neutral against the old one either way (both are 12 ss1 + 12 ss2 per location).

HELD-OUT PALETTES. 8 names are removed from the drawable pool entirely — no location may
draw them — and recorded in the roster + build metadata. They are the held-out-palette
invariance read at eval time: a colormap the network provably never saw in training.

REUSE. There is none, and not for a recipe reason: the v4..v7 aug caches were deleted on
2026-07-29 (commit 7068839) after the durability audit showed no surviving
`cache_manifest.jsonl` could attribute any of those 243,477 JPGs back to a location.
`audit_reuse` re-checks the disk rather than trusting this comment.

  uv run python tools/v8/build_plan.py [--dry-run]
  uv run python tools/v8/build_plan.py --measure-marginal   # palette vs geometry slot cost

Writes (all `paths.durable()`):
  data/v8/colormaps.json       the merged render library: the 76-name pool + blue_orange
  data/v8/aug_roster.json      the recipe, the pool, the held-out names, the draw rule
  data/v8/plan.jsonl           one row per render, for `v4-render-batch`
  data/v8/cache_manifest.jsonl one row per cached tile, for the trainer's loader
and AMENDS data/v8/build_metadata.json with an `aug_recipe` block (see `amend_metadata`).
The JPGs themselves are `paths.bulk()` -> ARTIFACTS_ROOT/data/v8/aug_cache/, out of tree.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402

MANIFEST = "data/v8/manifest.jsonl"
META_PATH = "data/v8/build_metadata.json"
POOL_SRC = "data/palettes/score3_colormaps.json"        # the committed curated 76
BLUE_ORANGE_SRC = "data/palettes/vivid_blue_orange.json"  # the labeler's map, its own file
COLORMAPS_OUT = "data/v8/colormaps.json"
ROSTER_OUT = "data/v8/aug_roster.json"
PLAN_OUT = "data/v8/plan.jsonl"
CACHE_MANIFEST_OUT = "data/v8/cache_manifest.jsonl"
V8_CACHE_DIR = "data/v8/aug_cache"          # repo-relative; bulk() resolves it out-of-tree

# --------------------------------------------------------------------------- #
# The v8b recipe
# --------------------------------------------------------------------------- #
RECIPE = "v8b"
DEPLOY_PALETTE = "twilight_shifted"    # data_v4.NEUTRAL_PALETTE; the canonical eval view
LABELER_PALETTE = "blue_orange"        # the map the human labels were formed through
N_DRAWN = 2                            # free palette slots per location
N_HELD_OUT = 8                         # pool names no location may draw
HELDOUT_SEED = 20260729                # fixes the held-out set for the whole build
SEED_TAG = "v8b-aug"                   # per-location seed namespace: f"{SEED_TAG}|{loc_id}"

N_GEOM = 3                             # 1 identity + 2 jittered, SHARED across all cells
SHIFT_FRAC_MAX = 0.10                  # shift magnitude ~ U[0, 0.10] of the slot frame width
SCALE_LO, SCALE_HI = 0.90, 1.10        # scale ~ U[0.90, 1.10]
IDENTITY_SCALE = 1.0                   # data_v4.CANON_SCALE
IDENTITY_SHIFT_ID = "center"           # data_v4.CANON_SHIFT — canonical()/aa_twin() key on it
JITTER_SHIFT_IDS = ("jit0", "jit1")

# (ss, downsample filter). ss1's "box" is a no-op average at ss=1; ss2's lanczos3 is what the
# deploy path actually uses (render-one's default), so the cached ss2 tile carries deploy's
# AA signature rather than a cheaper approximation. Unchanged from the first v8 recipe.
AA_LEVELS = ((1, "box"), (2, "lanczos3"))

N_PALETTES = 2 + N_DRAWN
SLOTS = N_PALETTES * N_GEOM * len(AA_LEVELS)      # 4 x 3 x 2 = 24
assert SLOTS == 24, SLOTS

# Prior cache trees, in the order a reuse search would consult them.
PRIOR_CACHES = [("v4", "data/v4/aug_cache", "data/v4/cache_manifest.jsonl"),
                ("v5", "data/v5/aug_cache_julia", "data/v5/cache_manifest.jsonl"),
                ("v6", "data/v6/aug_cache_gather", "data/v6/cache_manifest.jsonl"),
                ("v7", "data/v7/aug_cache", "data/v7/cache_manifest.jsonl")]

# Render constants — `v4-render-batch` defaults, unchanged from v4..v7.
CACHE_RENDER = {"width": 512, "height": 288, "maxiter": 8000, "jpg_quality": 85}


def fmt_f64(x: float) -> str:
    return repr(float(x))


def _palette_family(name: str) -> str:
    """Grouping token for a palette name, DERIVED from its namespace prefix (`cet_*`,
    `cmr.*`, else the bare name). Nothing in training reads it — `classifier/data_v4.py`
    stores `Render.palette_family` and never consults it, and the sampler weights are
    (class x group x source) only — so this is a faithful label, not a reconstruction."""
    if name.startswith("cet_"):
        return "cet"
    if "." in name:
        return name.split(".", 1)[0]
    return name


# --------------------------------------------------------------------------- #
# The palette library + the drawable pool
# --------------------------------------------------------------------------- #
def build_library() -> tuple[list, list]:
    """Merge the committed curated pool with the labeler's map into ONE library file.

    `v4-render-batch` takes a single `--colormaps` path and looks every plan row's palette
    up in it, but the two sources are separate committed files: `score3_colormaps.json`
    (the curated 76, the location-corpus pool) and `vivid_blue_orange.json` (blue_orange
    alone, which is in NEITHER colormap library). Entries are copied VERBATIM — same stops,
    same `mirror_needed` — so the merge cannot perturb a colormap; it only concatenates.
    Returns (merged_entries, pool_names)."""
    pool = json.loads((ROOT / POOL_SRC).read_text(encoding="utf-8"))
    bo = json.loads((ROOT / BLUE_ORANGE_SRC).read_text(encoding="utf-8"))
    pool_names = [c["name"] for c in pool]
    if DEPLOY_PALETTE not in pool_names:
        raise SystemExit(f"{DEPLOY_PALETTE!r} missing from {POOL_SRC}")
    if len(set(pool_names)) != len(pool_names):
        raise SystemExit(f"duplicate names in {POOL_SRC}")
    bo_entries = [c for c in bo if c["name"] == LABELER_PALETTE]
    if len(bo_entries) != 1:
        raise SystemExit(f"{LABELER_PALETTE!r} not found exactly once in {BLUE_ORANGE_SRC}")
    if LABELER_PALETTE in pool_names:
        raise SystemExit(f"{LABELER_PALETTE!r} unexpectedly already in the pool")
    return pool + bo_entries, pool_names


def choose_held_out(pool_names: list) -> list:
    """The 8 pool names no location may draw. Seeded, and drawn from the pool MINUS the
    deploy palette (which is on every location and so cannot be held out)."""
    candidates = sorted(n for n in pool_names if n != DEPLOY_PALETTE)
    return sorted(random.Random(HELDOUT_SEED).sample(candidates, N_HELD_OUT))


# --------------------------------------------------------------------------- #
# Per-location draw
# --------------------------------------------------------------------------- #
class Geom:
    """One geometric sample of a location: a scale and a (magnitude, direction) shift.

    Shared across all 4 palettes and both AA levels of that location, which is what makes
    the location's 24 tiles rest on only 6 distinct escape-time fields."""

    __slots__ = ("gid", "shift_id", "scale", "mag_frac", "angle")

    def __init__(self, gid, shift_id, scale, mag_frac, angle):
        self.gid, self.shift_id = gid, shift_id
        self.scale, self.mag_frac, self.angle = scale, mag_frac, angle

    def viewport(self, cx0: float, cy0: float, fw0: float):
        """(cx, cy, fw, dx, dy). The shift magnitude is a fraction of THIS slot's frame
        width (scale * base), not the base width — so a shift means the same visual
        displacement whichever side of 1.0 the scale landed on."""
        fw = self.scale * fw0
        mag = self.mag_frac * fw
        dx, dy = mag * math.cos(self.angle), mag * math.sin(self.angle)
        return cx0 + dx, cy0 + dy, fw, dx, dy


def draw_location(loc_id: int, drawable: list) -> tuple[list, list]:
    """This location's (palettes, geometries), from its own seeded RNG.

    The seed is the STRING f"{SEED_TAG}|{loc_id}" — `random.Random` hashes it to the
    Mersenne seed reproducibly across platforms and Python versions, so the whole fan-out
    is a pure function of loc_id and this file. The palette draw is consumed BEFORE the
    geometry draw; changing that order changes every location's fan-out."""
    rng = random.Random(f"{SEED_TAG}|{loc_id}")
    drawn = rng.sample(drawable, N_DRAWN)
    palettes = [DEPLOY_PALETTE, LABELER_PALETTE] + drawn
    geoms = [Geom("id", IDENTITY_SHIFT_ID, IDENTITY_SCALE, 0.0, 0.0)]
    for gid, shift_id in zip(("j0", "j1"), JITTER_SHIFT_IDS):
        geoms.append(Geom(gid, shift_id,
                          rng.uniform(SCALE_LO, SCALE_HI),
                          rng.uniform(0.0, SHIFT_FRAC_MAX),
                          rng.uniform(0.0, 2.0 * math.pi)))
    return palettes, geoms


def slot_filename(pal: str, g: Geom, ss: int) -> str:
    """Cache tile filename. Self-describing (a stray tile still names its own augmentation
    coordinates) and unique within a location by (palette, geom, ss) — the scale/shift
    digits are documentation, not the uniqueness key, now that both are continuous."""
    return f"{pal}__{g.gid}__s{g.scale:.4f}__sh{g.mag_frac:.4f}__ss{ss}.jpg"


# --------------------------------------------------------------------------- #
# Reuse audit
# --------------------------------------------------------------------------- #
def audit_reuse() -> dict:
    """How many of the 24 slots per location already exist in a v4..v7 cache?

    Checks the disk rather than trusting the commit message. A cached tile lives at
    `<cache>/<loc_id>/<palette>__...jpg`: the filename carries palette/geometry/ss but NOT
    the location, `<loc_id>` is a dense index into THAT version's manifest.jsonl, and only
    that version's cache_manifest.jsonl maps it back to (family, cx, cy, fw, c)."""
    dirs, manifests, trees = 0, [], []
    for ver, cache_rel, cm_rel in PRIOR_CACHES:
        cache = paths.bulk(cache_rel)
        if cache.exists():
            trees.append((ver, str(cache)))
            dirs += sum(1 for c in cache.iterdir() if c.is_dir())
        for cand in (ROOT / cm_rel, paths.bulk(cm_rel)):
            if cand.exists():
                manifests.append((ver, str(cand)))
    return {
        "prior_cache_trees_on_disk": trees,
        "prior_cache_location_dirs": dirs,
        "prior_cache_manifests_found": manifests,
        "reusable_slots_per_location": 0,
        "reuse_fraction": 0.0,
        "why": (
            "The v4..v7 aug caches were DELETED on 2026-07-29 (commit 7068839) — 243,477 "
            "JPGs that no surviving cache_manifest.jsonl could attribute back to a "
            "location. Nothing to reuse, and nothing that could have been: a tile's "
            "filename encodes palette/geometry/ss but not its location. Even had the trees "
            "survived, the v8b geometry is per-location randomised, so no jittered tile "
            "from a fixed-schedule recipe could match a v8b slot key."),
    }


# --------------------------------------------------------------------------- #
# Marginal-cost measurement
# --------------------------------------------------------------------------- #
def measure_marginal(rows, drawable, n_locs=64) -> dict:
    """What does one more PALETTE slot cost, versus one more GEOMETRY slot?

    The recipe's premise is that palettes are near-free because several share one
    escape-time field. That is true of the PLAN and false of the current EXECUTOR, and the
    difference is worth a number rather than an assumption. Three plans over the same
    locations, each timed through the real `v4-render-batch` binary into a temp dir:

        base : 2 palettes x 2 geometries      (4 tiles/loc)
        +pal : 3 palettes x 2 geometries      (6 tiles/loc)  -> +2 tiles, 0 new fields
        +geo : 2 palettes x 3 geometries      (6 tiles/loc)  -> +2 tiles, +2 new fields

    All at ss1 so the measurement is not dominated by supersampling. If palettes were free,
    (+pal - base) would be ~0 and (+geo - base) would be the full field cost."""
    binary = ROOT / "target" / "release" / "fractal-generator.exe"
    if not binary.exists():
        raise SystemExit(f"release binary missing: {binary}")
    # Stride, not head: the manifest is family-ordered, so rows[:n] would be all julia and
    # the cost of a field would be measured on the cheapest family in the corpus.
    stride = max(1, len(rows) // n_locs)
    sample = rows[::stride][:n_locs]

    def plan_for(n_pal, n_geo, tag, out_root):
        out = []
        for r in sample:
            palettes, geoms = draw_location(r["loc_id"], drawable)
            cx0, cy0, fw0 = float(r["cx"]), float(r["cy"]), float(r["fw"])
            ft = r.get("fractal_type", "mandelbrot")
            extra = {k: r[k] for k in loc_mod.family_param_keys(ft) if r.get(k) is not None}
            for pal in palettes[:n_pal]:
                for g in geoms[:n_geo]:
                    cx, cy, fw, _dx, _dy = g.viewport(cx0, cy0, fw0)
                    row = {"cx": fmt_f64(cx), "cy": fmt_f64(cy), "fw": fmt_f64(fw),
                           "palette": pal, "ss": 1, "filter": "box",
                           "out": str(out_root / tag / f"{r['loc_id']}_{pal}_{g.gid}.jpg"),
                           "fractal_type": ft}
                    if r.get("c_re") is not None:
                        row["c_re"], row["c_im"] = r["c_re"], r["c_im"]
                    row.update(extra)
                    out.append(row)
        return out

    results = {}
    with tempfile.TemporaryDirectory(prefix="v8_marginal_") as td:
        troot = Path(td)
        for tag, (n_pal, n_geo) in (("base", (2, 2)), ("pal", (3, 2)), ("geo", (2, 3))):
            plan_rows = plan_for(n_pal, n_geo, tag, troot)
            pf = troot / f"{tag}.jsonl"
            pf.write_text("\n".join(json.dumps(r) for r in plan_rows) + "\n", encoding="utf-8")
            t0 = time.time()
            proc = subprocess.run(
                [str(binary), "v4-render-batch", "--plan", str(pf),
                 "--colormaps", str(ROOT / COLORMAPS_OUT),
                 "--log-every", "100000"],
                cwd=str(ROOT), capture_output=True, text=True)
            el = time.time() - t0
            if proc.returncode != 0:
                raise SystemExit(f"marginal measurement failed ({tag}):\n{proc.stderr[-2000:]}")
            results[tag] = {"tiles": len(plan_rows), "wall_s": round(el, 2),
                            "s_per_tile": round(el / len(plan_rows), 4)}
    base, pal, geo = results["base"], results["pal"], results["geo"]
    d_pal = pal["wall_s"] - base["wall_s"]
    d_geo = geo["wall_s"] - base["wall_s"]
    n_extra = pal["tiles"] - base["tiles"]
    return {
        "locations": n_locs, "ss": 1, "note": "same locations, same seeds, all three plans",
        "arms": results,
        "marginal_s_per_palette_tile": round(d_pal / n_extra, 4),
        "marginal_s_per_geometry_tile": round(d_geo / n_extra, 4),
        "palette_over_geometry_ratio": round(d_pal / d_geo, 3) if d_geo else None,
        "interpretation": (
            "ratio ~1.0 => the executor re-iterates per palette, so a palette slot costs "
            "the same as a geometry slot and the plan's field-sharing is unexploited. "
            "ratio ~0 => palettes are near-free (recolor batching is live)."),
    }


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #
def emit_location(r, drawable, plan_rows, cm_rows):
    """The 24 plan rows (and 24 cache rows) for one location.

    Ordered (geometry, ss) OUTER and palette INNER, so the 4 rows resting on one
    escape-time field are ADJACENT in the plan. `v4-render-batch` par_iters, so today this
    only buys locality — but it is the order a recolor-batching executor would need, and
    costs nothing to emit.

    Family extra-constants (`p_re/p_im/zm1_re/zm1_im` for phoenix) are copied onto every
    plan row: without them `v4-render-batch` falls back to PHOENIX_{C,P,ZM1}_DEFAULT and
    renders all 573 phoenix locations at the one default Ushiki spot, silently."""
    loc_id = r["loc_id"]
    palettes, geoms = draw_location(loc_id, drawable)
    cx0, cy0 = float(r["cx"]), float(r["cy"])
    fw0 = float(r["fw"])
    ft = r.get("fractal_type", "mandelbrot")
    base = dict(label=r["label"], split=r["split"], group_id=r["group_id"],
                source=r["source"], biased=r["biased"])
    extra = {k: r[k] for k in loc_mod.family_param_keys(ft) if r.get(k) is not None}
    c_re, c_im = r.get("c_re"), r.get("c_im")

    for g in geoms:
        cx, cy, fw_slot, dx, dy = g.viewport(cx0, cy0, fw0)
        for ss, filt in AA_LEVELS:
            for pal in palettes:
                rel = f"{V8_CACHE_DIR}/{loc_id}/{slot_filename(pal, g, ss)}"
                row = {
                    "cx": fmt_f64(cx), "cy": fmt_f64(cy), "fw": fmt_f64(fw_slot),
                    "palette": pal, "ss": ss, "filter": filt,
                    "out": paths.bulk(rel).as_posix(),
                    "fractal_type": ft,
                }
                if c_re is not None:
                    row["c_re"] = c_re
                    row["c_im"] = c_im
                row.update(extra)
                plan_rows.append(row)
                cm_rows.append({
                    "location_id": loc_id, **base,
                    "palette": pal, "palette_family": _palette_family(pal),
                    # `scale` / `shift_id` keep the v4..v7 field names the trainer's loader
                    # keys on (data_v4.canonical() wants scale==1.0 and shift_id=="center";
                    # the identity slot supplies exactly one such row per palette per AA).
                    "scale": g.scale, "shift_id": g.shift_id,
                    "geom_id": g.gid, "shift_frac": g.mag_frac, "shift_angle": g.angle,
                    "shift_dx": dx, "shift_dy": dy,
                    # Two-value AA vocabulary, as classifier/data_v4.py expects — but
                    # "antialiased" means ss2, the DEPLOY level, not ss4. `ss` is emitted
                    # explicitly so no consumer has to infer it from the label.
                    "aa_level": "aliased" if ss == 1 else "antialiased",
                    "ss": ss, "filter": filt,
                    "fractal_type": ft, "path": rel,
                })


def amend_metadata(recipe_block: dict) -> None:
    """Write the recipe into `build_metadata.json` under `aug_recipe`.

    That file is OWNED by build_manifest.py (population + split decisions); this adds one
    key and rewrites, idempotently. The ordering invariant is build_manifest -> build_plan,
    which already holds because the plan is derived from the manifest — so a manifest
    rebuild that drops this key is always followed by the plan build that restores it."""
    p = paths.durable(META_PATH)
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["aug_recipe"] = recipe_block
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the recipe, reuse audit and counts; write nothing")
    ap.add_argument("--measure-marginal", action="store_true",
                    help="time a palette slot against a geometry slot at the real executor")
    a = ap.parse_args()

    library, pool_names = build_library()
    held_out = choose_held_out(pool_names)
    drawable = sorted(n for n in pool_names
                      if n != DEPLOY_PALETTE and n not in set(held_out))

    rows = [json.loads(l) for l in (ROOT / MANIFEST).read_text(encoding="utf-8").splitlines()
            if l.strip()]

    print("=" * 84)
    print(f"v8 PLAN — recipe {RECIPE} — {len(rows)} locations x {SLOTS} slots")
    print("=" * 84)
    print(f"  palettes/loc : {N_PALETTES}  = [{DEPLOY_PALETTE}, {LABELER_PALETTE}] "
          f"+ {N_DRAWN} drawn per location")
    print(f"  pool         : {len(pool_names)} curated ({POOL_SRC})")
    print(f"  held out     : {N_HELD_OUT} -> {held_out}")
    print(f"  drawable     : {len(drawable)}  (pool - deploy - held-out)")
    print(f"  geometry/loc : {N_GEOM} = identity(scale 1.0, centre) + 2 jittered "
          f"[scale U({SCALE_LO},{SCALE_HI}), shift U(0,{SHIFT_FRAC_MAX}) frame widths, "
          f"uniform direction]")
    print(f"  AA           : {[f'ss{s} {f}' for s, f in AA_LEVELS]}")
    print(f"  distinct fields/loc : {N_GEOM * len(AA_LEVELS)}  "
          f"({N_GEOM} geometries x {len(AA_LEVELS)} AA), each shared by {N_PALETTES} palettes")
    n_ss1_per = N_PALETTES * N_GEOM
    print(f"  cost/loc     : {n_ss1_per}x ss1 + {n_ss1_per}x ss2 = "
          f"{n_ss1_per * (1 + 4)} ss1-equivalents unbatched  "
          f"(= the first v8 recipe; v4..v7 was 132)")

    print("\n  OLD magnitudes, for the record:")
    print("    scales     {0.7, 1.0, 1.3}                    (three fixed values, +-30%)")
    print("    shift      0.4 x the SLOT frame width, FIXED  (0.28 / 0.40 / 0.52 base widths)")
    print("    direction  deterministic, 2*pi*(pal_idx*3 + scale_idx)/(n_pal*3)")
    print("    -> the s1.3 shifted tile sat 0.52 base frame widths off centre. New cap: 0.11.")

    # --- palette exposure ---
    draw_counts = Counter()
    for r in rows:
        for p in draw_location(r["loc_id"], drawable)[0]:
            draw_counts[p] += 1
    free = {n: draw_counts[n] for n in drawable}
    lo, hi = min(free.values()), max(free.values())
    print(f"\n  palette exposure across the corpus:")
    print(f"    {DEPLOY_PALETTE:<22} {draw_counts[DEPLOY_PALETTE]:>6}  (every location)")
    print(f"    {LABELER_PALETTE:<22} {draw_counts[LABELER_PALETTE]:>6}  (every location)")
    print(f"    {len(drawable)} drawable          {lo:>6}..{hi}  "
          f"(mean {sum(free.values())/len(free):.1f})")
    print(f"    {N_HELD_OUT} held out            {sum(draw_counts[n] for n in held_out):>6}  "
          f"(must be 0)")
    assert all(draw_counts[n] == 0 for n in held_out), "a held-out palette was drawn"

    reuse = audit_reuse()
    print("\n--- REUSE AUDIT (before rendering anything) ---")
    print(f"  prior cache trees on disk : {reuse['prior_cache_trees_on_disk'] or 'NONE'}")
    print(f"  prior cache_manifests     : {reuse['prior_cache_manifests_found'] or 'NONE FOUND'}")
    print(f"  reusable slots/location   : {reuse['reusable_slots_per_location']} of {SLOTS}")
    print(f"  why: {reuse['why']}")

    plan_rows, cm_rows = [], []
    for r in rows:
        emit_location(r, drawable, plan_rows, cm_rows)
    assert len(plan_rows) == len(rows) * SLOTS, (len(plan_rows), len(rows) * SLOTS)
    assert len(cm_rows) == len(plan_rows)
    # Every location must expose exactly one deploy-canonical view, or data_v4.canonical()
    # asserts at load: (twilight_shifted, antialiased, scale 1.0, "center").
    n_canon = sum(1 for c in cm_rows if c["palette"] == DEPLOY_PALETTE
                  and c["aa_level"] == "antialiased" and c["scale"] == IDENTITY_SCALE
                  and c["shift_id"] == IDENTITY_SHIFT_ID)
    assert n_canon == len(rows), f"{n_canon} canonical views for {len(rows)} locations"
    assert len({c["path"] for c in cm_rows}) == len(cm_rows), "duplicate tile paths"

    n_ss1 = sum(1 for p in plan_rows if p["ss"] == 1)
    n_ss2 = sum(1 for p in plan_rows if p["ss"] == 2)
    print(f"\n  plan rows      : {len(plan_rows)}  (ss1 box {n_ss1} + ss2 lanczos3 {n_ss2})")
    print(f"  canonical views: {n_canon} (one per location)")
    fam = Counter(p["fractal_type"] for p in plan_rows)
    print(f"  per-family rows: {dict(sorted(fam.items()))}")
    n_c = sum(1 for p in plan_rows if "c_re" in p)
    n_p = sum(1 for p in plan_rows if "p_re" in p)
    print(f"  rows carrying c: {n_c}   rows carrying phoenix p/z-1: {n_p}")
    print(f"  cache root     : {paths.bulk(V8_CACHE_DIR)}  (bulk, out-of-tree)")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # The merged library goes out FIRST — the marginal measurement renders through it.
    paths.durable(COLORMAPS_OUT, mkparents=True).write_text(
        json.dumps(library, indent=1), encoding="utf-8")
    print(f"\nWROTE {COLORMAPS_OUT}  ({len(library)} colormaps: "
          f"{len(pool_names)} pool + {LABELER_PALETTE})")

    marginal = None
    if a.measure_marginal:
        print("\n--- MARGINAL COST: a palette slot vs a geometry slot ---")
        marginal = measure_marginal(rows, drawable)
        for tag, v in marginal["arms"].items():
            print(f"  {tag:<5} {v['tiles']:>4} tiles  {v['wall_s']:>7.2f}s  "
                  f"{v['s_per_tile']:.4f} s/tile")
        print(f"  marginal s/tile — palette {marginal['marginal_s_per_palette_tile']:.4f}"
              f"   geometry {marginal['marginal_s_per_geometry_tile']:.4f}"
              f"   ratio {marginal['palette_over_geometry_ratio']}")
        print(f"  {marginal['interpretation']}")

    recipe_block = {
        "recipe": RECIPE,
        "supersedes": ("the first v8 recipe (2 palettes x 3 fixed scales {0.7,1.0,1.3} x 2 "
                       "fixed shifts at 0.4 frame widths x 2 AA), abandoned at 10,538 of "
                       "171,384 tiles"),
        "why": ("the fixed 0.4-frame-width shift at scale 1.3 put the window 0.52 base frame "
                "widths off centre — off the structure entirely — while the tile kept the "
                "location's label. That corrupts only the positive classes (an empty crop "
                "labeled 1 is roughly true; labeled 3 or 4 it is a lie) and fights deploy, "
                "which reframes to centre content before scoring."),
        "slots_per_location": SLOTS,
        "axes": f"{N_PALETTES} palettes x {N_GEOM} geometries x {len(AA_LEVELS)} AA levels",
        "palettes": {
            "always": [DEPLOY_PALETTE, LABELER_PALETTE],
            "always_roles": {
                DEPLOY_PALETTE: "deploy-matched; data_v4.NEUTRAL_PALETTE, the pinned "
                                "scoring instrument and the canonical eval view",
                LABELER_PALETTE: "the vivid companion the human labels were formed "
                                 "through (data/palettes/vivid_blue_orange.json); closes "
                                 "the judge/model presentation gap",
            },
            "drawn_per_location": N_DRAWN,
            "pool_source": POOL_SRC,
            "pool_size": len(pool_names),
            "drawable": drawable,
            "drawable_size": len(drawable),
            "held_out": held_out,
            "held_out_seed": HELDOUT_SEED,
            "held_out_role": ("no location may draw these; they are the held-out-palette "
                              "invariance read at eval time"),
            "draw_scope": ("PER LOCATION, not per crop — all 4 palettes share the "
                           "location's 3 geometries, so 24 tiles rest on 6 distinct "
                           "escape-time fields"),
            "exposure_per_drawable_palette": {"min": lo, "max": hi,
                                              "mean": round(sum(free.values())/len(free), 1)},
        },
        "geometry": {
            "samples_per_location": N_GEOM,
            "identity": {"scale": IDENTITY_SCALE, "shift": 0.0, "shift_id": IDENTITY_SHIFT_ID,
                         "role": "the deploy composition; present in every (palette, AA) cell"},
            "jittered": {
                "count": 2,
                "shift_frac": f"U(0, {SHIFT_FRAC_MAX}) of the slot frame width",
                "shift_direction": "U(0, 2*pi)",
                "scale": f"U({SCALE_LO}, {SCALE_HI})",
                "shift_ids": list(JITTER_SHIFT_IDS),
            },
            "seed": f'random.Random("{SEED_TAG}|<loc_id>"); palette draw consumed first',
        },
        "aa_levels": [{"ss": s, "filter": f,
                       "aa_level": "aliased" if s == 1 else "antialiased"}
                      for s, f in AA_LEVELS],
        "previous_recipe_magnitudes": {
            "note": "recorded for the first time — these were never written down, only seen",
            "scales": [0.7, 1.0, 1.3],
            "shift_frac": 0.4,
            "shift_frac_basis": "the SLOT frame width (scale x base), so 0.28/0.40/0.52 base",
            "shift_direction": "deterministic: 2*pi*(palette_index*3 + scale_index)/(n_pal*3)",
            "aa_levels": "ss1 box + ss2 lanczos3 (unchanged)",
        },
        "colormap_library": {
            "path": COLORMAPS_OUT,
            "n": len(library),
            "built_from": [POOL_SRC, BLUE_ORANGE_SRC],
            "note": ("v4-render-batch takes ONE --colormaps path and blue_orange is in "
                     "neither committed library, so the two committed sources are "
                     "concatenated verbatim (stops and mirror_needed untouched)"),
        },
        "cache_render": {**CACHE_RENDER,
                         "note": "v4-render-batch defaults, unchanged from v4..v7"},
        "executor_note": ("the plan shares 4 palettes per escape-time field, but "
                          "v4-render-batch renders each row independently, so that sharing "
                          "is NOT exploited today — see marginal_cost below"),
        "marginal_cost": marginal,
        "reuse_audit": reuse,
        "trainer_followup": (
            "classifier/data_v4.py Loc.palette_renders() asserts exactly 6 palette renders "
            "per location and must become 4 (and per-location variable) before a v8 train. "
            "canonical() and aa_twin() are unaffected: the identity slot supplies exactly "
            "one (twilight_shifted, ss2, scale 1.0, center) row per location."),
    }

    paths.durable(ROSTER_OUT, mkparents=True).write_text(
        json.dumps(recipe_block, indent=2), encoding="utf-8")
    with paths.durable(PLAN_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row) + "\n")
    with paths.durable(CACHE_MANIFEST_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for row in cm_rows:
            f.write(json.dumps(row) + "\n")
    amend_metadata(recipe_block)

    print(f"WROTE {ROSTER_OUT}")
    print(f"WROTE {PLAN_OUT}            ({len(plan_rows)} rows)")
    print(f"WROTE {CACHE_MANIFEST_OUT}  ({len(cm_rows)} rows)")
    print(f"AMENDED {META_PATH}         (aug_recipe block)")


if __name__ == "__main__":
    main()
