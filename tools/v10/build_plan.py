#!/usr/bin/env python
r"""v10 render plan + cache manifest — the v9 cache EXTENDED, not rebuilt.

**The recipe does not change and the cap does not change.** Same v8b fan-out, same
per-location seeds, same held-out palettes, same colormap library, same
`auto_maxiter(fw_slot)` policy. The only thing that moved is the POPULATION: 1,267
maneuver-view locations appended onto v8's frozen prefix (tools/v10/build_manifest.py).
The labels are this retrain's single variable, which is what makes the v10-vs-v8
comparison a read on the data rather than on a recipe drift.

  4 palettes  x  3 geometric samples  x  2 AA levels {ss1 box, ss2 lanczos3}  = 24

WHY THE v9 TREE IS EXTENDED RATHER THAN RE-RENDERED. v9's build wrote a fresh tree because
its intervention was the CAP — every v8 tile was at a superseded flat 8000, so no tile
could be reused and a mixed-cap corpus had to be impossible by construction. None of that
applies here: v10 renders under the same cap policy v9 did, so a v9 tile of a surviving
location is bit-for-bit the tile v10 would produce. Re-rendering 170,760 of them would cost
~5.8 h to reproduce files that already exist.

So the plan spans TWO trees, and each stays self-describing:

  data/v9/aug_cache/<loc_id>/   the 7,115 prefix locations — rows BYTE-IDENTICAL to v9's
  data/v10/aug_cache/<loc_id>/  the 1,267 appended locations — the only rows to render

`v4-render-batch` skips any row whose output exists, so the full 201,168-row plan is
itself the resume: the first pass skips the 170,760 tiles already on disk and renders the
30,408 new ones. GATE A below proves the prefix rows are byte-identical to v9's, which is
what makes "extend" a checked claim rather than an intention.

TWO PREFIX LOCATIONS ARE GONE. v10's manifest displaces two v8 TRAIN rows (loc_id 2849 and
4354 — an appended location bridged each into a forced-eval group; see
build_manifest.py GATE 11). Their 48 v9 tiles stay on disk and are simply not named by the
v10 plan. `verify_cache_alignment.py` names them as EXPECTED orphans rather than counting
the tree complete, because a location dir the manifest does not name is exactly how a tile
count can exceed a plan and still read as done.

  uv run python tools/v10/build_plan.py [--dry-run]

READS `data/v10/manifest.jsonl` (the population) and `data/v9/{plan,aug_roster,
colormaps}.json*` (the recipe, asserted unchanged).

Writes (all `paths.durable()`):
  data/v10/colormaps.json       the merged render library (asserted == v9's, byte for byte)
  data/v10/aug_roster.json      the recipe + the extension block
  data/v10/plan.jsonl           one row per render, for `v4-render-batch`
  data/v10/cache_manifest.jsonl one row per cached tile, for the trainer's loader
  data/v10/build_metadata.json  the manifest's build block + `aug_recipe`
The JPGs themselves are `paths.bulk()` -> ARTIFACTS_ROOT/data/v{9,10}/aug_cache/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "v9"))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402
import active_ckpt as ac     # noqa: E402  (THE production iteration-cap policy)

# The recipe is IMPORTED from v9's builder, never restated: the draw order, the seed
# namespace, the geometry magnitudes and the slot filename are the identity of a tile, and
# a second copy of them is a second thing that can drift.
import build_plan as v9p     # noqa: E402

MANIFEST = "data/v10/manifest.jsonl"
V10_META_SRC = "data/v10/build_metadata.json"
V9_ROSTER = "data/v9/aug_roster.json"
V9_COLORMAPS = "data/v9/colormaps.json"
V9_PLAN = "data/v9/plan.jsonl"
META_PATH = "data/v10/build_metadata.json"
COLORMAPS_OUT = "data/v10/colormaps.json"
ROSTER_OUT = "data/v10/aug_roster.json"
PLAN_OUT = "data/v10/plan.jsonl"
CACHE_MANIFEST_OUT = "data/v10/cache_manifest.jsonl"

V9_CACHE_DIR = "data/v9/aug_cache"      # the prefix tree — read, never written
V10_CACHE_DIR = "data/v10/aug_cache"    # the extension tree — the only thing rendered
PREFIX_MAX_LOC_ID = 7140                # every v8 loc_id is <= this; appended ids start above


def emit_location(r, drawable, cache_dir, plan_rows, cm_rows):
    """v9's `emit_location`, with the cache tree as a parameter.

    v9 hardcoded its single tree. The extension needs prefix locations to keep naming the
    v9 tree (so their rows stay byte-identical and their tiles stay found) while appended
    locations name the v10 tree. Everything else — the seeded draw, the per-slot cap, the
    field ordering — is v9's, reached through the imported module."""
    saved = v9p.V9_CACHE_DIR
    try:
        v9p.V9_CACHE_DIR = cache_dir
        v9p.emit_location(r, drawable, plan_rows, cm_rows)
    finally:
        v9p.V9_CACHE_DIR = saved


def assert_recipe_parity(plan_rows, prefix_ids, library) -> dict:
    """Prove against the committed v9 artifacts that the RECIPE did not move.

    Two claims, both load-bearing:
      A. every plan row of a PREFIX location is byte-identical to its v9 row — same cap,
         same palette, same geometry, same `out` path. This is what makes reusing the v9
         tiles legitimate; if it fails, the tiles on disk are not the tiles the plan asks
         for and the whole extension is unsound.
      B. the colormap library rebuilds byte-identically from the same committed sources.
    Any failure aborts before a byte is written."""
    out = {}
    v9_cm = ROOT / V9_COLORMAPS
    built = json.dumps(library, indent=1)
    if built != v9_cm.read_text(encoding="utf-8"):
        raise SystemExit(f"colormap library differs from {V9_COLORMAPS} — the palette "
                         f"sources moved under the recipe")
    out["colormaps_identical_to_v9"] = True

    v9_rows = [json.loads(l) for l in
               (ROOT / V9_PLAN).read_text(encoding="utf-8").splitlines() if l.strip()]
    v9_by_out = {r["out"]: r for r in v9_rows}
    prefix = [r for r in plan_rows if int(Path(r["out"]).parent.name) in prefix_ids]
    missing = [r for r in prefix if r["out"] not in v9_by_out]
    if missing:
        raise SystemExit(
            f"{len(missing)} prefix plan rows name a tile v9's plan never had — the seeded "
            f"draw moved. e.g. {[m['out'] for m in missing[:3]]}")
    drift = []
    for r in prefix:
        o = v9_by_out[r["out"]]
        if r != o:
            drift.append((r["out"], {k: (o.get(k), r.get(k))
                                     for k in set(r) | set(o) if r.get(k) != o.get(k)}))
    if drift:
        raise SystemExit(f"GATE A FAIL: {len(drift)} prefix plan rows differ from v9's — "
                         f"the tiles on disk are not the tiles this plan asks for. "
                         f"e.g. {drift[:3]}")
    out["prefix_plan_rows_byte_identical_to_v9"] = len(prefix)
    out["v9_plan_rows_not_reused"] = len(v9_rows) - len(prefix)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the recipe and counts; write nothing")
    a = ap.parse_args()

    library, pool_names = v9p.build_library()
    held_out = v9p.choose_held_out(pool_names)
    drawable = sorted(n for n in pool_names
                      if n != v9p.DEPLOY_PALETTE and n not in set(held_out))

    # The recipe's palette decisions must be v9's, not merely "computed the same way".
    v9_roster = json.loads((ROOT / V9_ROSTER).read_text(encoding="utf-8"))
    if v9_roster["palettes"]["held_out"] != held_out:
        raise SystemExit(f"held-out palette set moved from v9's: {v9_roster['palettes']['held_out']}"
                         f" -> {held_out}")
    if v9_roster["palettes"]["drawable"] != drawable:
        raise SystemExit("drawable palette pool moved from v9's")

    rows = [json.loads(l) for l in (ROOT / MANIFEST).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    prefix_rows = [r for r in rows if r["loc_id"] <= PREFIX_MAX_LOC_ID]
    new_rows = [r for r in rows if r["loc_id"] > PREFIX_MAX_LOC_ID]
    prefix_ids = {r["loc_id"] for r in prefix_rows}

    print("=" * 84)
    print(f"v10 PLAN — recipe {v9p.RECIPE} at the v9 cap policy — EXTEND, do not rebuild")
    print("=" * 84)
    print(f"  manifest     : {MANIFEST}  ({len(rows)} locations x {v9p.SLOTS} slots)")
    print(f"    prefix     : {len(prefix_rows)} locations -> {V9_CACHE_DIR} (already rendered)")
    print(f"    appended   : {len(new_rows)} locations -> {V10_CACHE_DIR} (to render)")
    print(f"  cap policy   : auto_maxiter(fw_slot)  base {ac.MAXITER_BASE} k {ac.MAXITER_K} "
          f"clamp [{ac.MAXITER_MIN},{ac.MAXITER_MAX}]   (v9's, unchanged)")
    print(f"  palettes/loc : {v9p.N_PALETTES}  held out {len(held_out)}  "
          f"drawable {len(drawable)}   (all asserted == v9's)")

    plan_rows, cm_rows = [], []
    for r in prefix_rows:
        emit_location(r, drawable, V9_CACHE_DIR, plan_rows, cm_rows)
    n_prefix_slots = len(plan_rows)
    for r in new_rows:
        emit_location(r, drawable, V10_CACHE_DIR, plan_rows, cm_rows)
    n_new_slots = len(plan_rows) - n_prefix_slots

    assert len(plan_rows) == len(rows) * v9p.SLOTS, (len(plan_rows), len(rows) * v9p.SLOTS)
    assert len(cm_rows) == len(plan_rows)
    n_canon = sum(1 for c in cm_rows if c["palette"] == v9p.DEPLOY_PALETTE
                  and c["aa_level"] == "antialiased" and c["scale"] == v9p.IDENTITY_SCALE
                  and c["shift_id"] == v9p.IDENTITY_SHIFT_ID)
    assert n_canon == len(rows), f"{n_canon} canonical views for {len(rows)} locations"
    assert len({c["path"] for c in cm_rows}) == len(cm_rows), "duplicate tile paths"

    mit_lo = min(p["maxiter"] for p in plan_rows)
    mit_hi = max(p["maxiter"] for p in plan_rows)
    if mit_hi >= ac.MAXITER_MAX:
        raise SystemExit(f"cap clamp {ac.MAXITER_MAX} is BINDING at {mit_hi} — the deep "
                         f"tail is being truncated; re-read docs/design/auto_maxiter.md")
    parity = assert_recipe_parity(plan_rows, prefix_ids, library)

    print("\n--- RECIPE PARITY vs the committed v9 artifacts ---")
    for k, v in parity.items():
        print(f"  {k:<44} {v}")
    print(f"  maxiter range  : {mit_lo}..{mit_hi}  (clamp {ac.MAXITER_MAX}, not binding)")

    new_mits = [p["maxiter"] for p in plan_rows[n_prefix_slots:]]
    print(f"\n  plan rows      : {len(plan_rows)}  = {n_prefix_slots} prefix (on disk) "
          f"+ {n_new_slots} to render")
    print(f"  appended cap   : {min(new_mits)}..{max(new_mits)}  "
          f"(mean {sum(new_mits)/len(new_mits):.0f}; prefix mean "
          f"{sum(p['maxiter'] for p in plan_rows[:n_prefix_slots])/n_prefix_slots:.0f})")
    fam = Counter(p["fractal_type"] for p in plan_rows[n_prefix_slots:])
    print(f"  appended family: {dict(sorted(fam.items()))}")
    print(f"  cache roots    : {paths.bulk(V9_CACHE_DIR)}")
    print(f"                   {paths.bulk(V10_CACHE_DIR)}")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    paths.durable(COLORMAPS_OUT, mkparents=True).write_text(
        json.dumps(library, indent=1), encoding="utf-8")

    recipe_block = dict(v9_roster)          # v9's recipe block verbatim...
    recipe_block["corpus_version"] = "v10"  # ...with the extension recorded on top
    recipe_block["supersedes_corpus"] = "v9 (same recipe, same cap; v10 adds locations)"
    recipe_block["extension"] = {
        "mode": "extend_in_place",
        "prefix_locations": len(prefix_rows),
        "prefix_tiles": n_prefix_slots,
        "prefix_tree": V9_CACHE_DIR,
        "appended_locations": len(new_rows),
        "appended_tiles": n_new_slots,
        "appended_tree": V10_CACHE_DIR,
        "why_two_trees": (
            "v9's tiles are bit-for-bit what v10 would render — same recipe, same cap "
            "policy, same per-location seed — so re-rendering them would spend ~5.8 h "
            "reproducing files that exist. Splitting the appended tiles into their own "
            "tree keeps BOTH self-describing: v9's tree still matches v9's plan exactly, "
            "and v10's tree holds exactly the extension."),
        "displaced_prefix_locations": (
            "2 v8 loc_ids (2849, 4354) are no longer in the manifest; their 48 v9 tiles "
            "stay on disk unnamed by this plan and are declared EXPECTED orphans by "
            "verify_cache_alignment.py."),
        "single_variable": (
            "the LABELS are this retrain's only variable. The wider palette set stays "
            "PARKED for a second consecutive build and is still wanted for the rebuild "
            "after this one."),
    }
    recipe_block["recipe_parity"] = parity
    recipe_block["cache_render"] = dict(v9_roster["cache_render"])
    recipe_block["cache_render"]["maxiter_range_over_plan"] = {"min": mit_lo, "max": mit_hi}

    paths.durable(ROSTER_OUT, mkparents=True).write_text(
        json.dumps(recipe_block, indent=2), encoding="utf-8")
    with paths.durable(PLAN_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row) + "\n")
    with paths.durable(CACHE_MANIFEST_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for row in cm_rows:
            f.write(json.dumps(row) + "\n")

    meta_path = paths.durable(META_PATH, mkparents=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["aug_recipe"] = recipe_block
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWROTE {COLORMAPS_OUT}  ({len(library)} colormaps)")
    print(f"WROTE {ROSTER_OUT}")
    print(f"WROTE {PLAN_OUT}            ({len(plan_rows)} rows)")
    print(f"WROTE {CACHE_MANIFEST_OUT}  ({len(cm_rows)} rows)")
    print(f"AMENDED {META_PATH}         (aug_recipe block)")


if __name__ == "__main__":
    main()
