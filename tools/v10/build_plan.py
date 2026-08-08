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
30,408 new ones.

GATE A IS RETIRED — 2026-08-08, and this is the record of what it was for. It proved every
prefix plan row byte-identical to its v9 row, which is what made reusing 170,760 v9 tiles
legitimate. Both trees are now GONE (`data/v{9,10}/aug_cache`, 201,216 tiles / 14.30 GiB,
deleted 2026-08-08 under the ACTIVE+PREVIOUS weights-retention policy: a rollback to v10
uses its WEIGHTS, and v11 is a fresh crop-batch build with no chain to either). There is
nothing on disk left for a row to be identical *to*, so the gate can no longer make its
claim — and holding `data/v9/plan.jsonl` tracked to arm a gate over deleted tiles is the
"keep the machinery for reproducing something nobody will run again" failure the durability
contract names. GATE B (the colormap library) is untouched: it compares two committed files
and is unaffected by the tiles. `[docs/design/storage_classes.md]`

TWO PREFIX LOCATIONS ARE GONE. v10's manifest displaces two v8 TRAIN rows (loc_id 2849 and
4354 — an appended location bridged each into a forced-eval group; see
build_manifest.py GATE 11). Their 48 v9 tiles were simply not named by the v10 plan.

  uv run python tools/v10/build_plan.py [--dry-run]

READS `data/v10/manifest.jsonl` (the population) and `data/v9/{aug_roster,colormaps}.json`
(the recipe, asserted unchanged). It no longer reads `data/v9/plan.jsonl` — see GATE A above.

Writes:
  data/v10/colormaps.json       durable() — the merged render library (asserted == v9's)
  data/v10/aug_roster.json      durable() — the recipe + the extension block
  data/v10/build_metadata.json  durable() — the manifest's build block + `aug_recipe`
  data/v10/plan.jsonl           bulk()    — one row per render, for `v4-render-batch`
  data/v10/cache_manifest.jsonl bulk()    — one row per cached tile, for the loader
The JPGs are gone: `data/v{9,10}/aug_cache` (201,216 tiles / 14.30 GiB) was deleted
2026-08-08 under the ACTIVE+PREVIOUS weights-retention policy.

WHY THE PAIR IS bulk() AND THE OTHER THREE ARE durable(), 2026-08-08. The pair was tracked
(LFS, 179 MB) because it is the only thing mapping a cached tile back to a location — an
argument that died with the tiles. It is a byte-identical function of
`data/v10/manifest.jsonl` + v9's committed roster and colormaps: measured, not argued —
rebuilt over the originals and sha256-equal on both (65,741,076 B and 113,162,392 B). The
other three are NOT byte-reproducible, and it is not a disk probe this time but a RETIRED
GATE: `recipe_parity` inside the roster and build_metadata still holds GATE A's counts
(`prefix_plan_rows_byte_identical_to_v9: 170760`, `v9_plan_rows_not_reused: 48`) and a
re-run replaces them with the retirement note, because the gate no longer runs. Those stay
tracked and frozen — the roster keeps the counts it passed on. RESTORE THEM after any
rebuild (`git checkout -- data/v10/aug_roster.json data/v10/build_metadata.json`).
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
sys.path.insert(0, str(ROOT))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402
import active_ckpt as ac     # noqa: E402  (THE production iteration-cap policy)

# The recipe is IMPORTED from v9's builder, never restated: the draw order, the seed
# namespace, the geometry magnitudes and the slot filename are the identity of a tile, and
# a second copy of them is a second thing that can drift.
from tools.v9 import build_plan as v9p   # noqa: E402

MANIFEST = "data/v10/manifest.jsonl"
V10_META_SRC = "data/v10/build_metadata.json"
V9_ROSTER = "data/v9/aug_roster.json"
V9_COLORMAPS = "data/v9/colormaps.json"
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


def assert_recipe_parity(library) -> dict:
    """Prove against the committed v9 artifacts that the RECIPE did not move.

    ONE claim survives:
      B. the colormap library rebuilds byte-identically from the same committed sources.

    GATE A — every PREFIX plan row byte-identical to its v9 row — was RETIRED 2026-08-08
    with the tiles it protected. It existed so that reusing 170,760 v9 tiles was a checked
    claim; `data/v{9,10}/aug_cache` are deleted, so there is no tile for a row to be
    identical to and the gate could only ever compare a plan against another plan. Keeping
    it would mean keeping `data/v9/plan.jsonl` tracked (146 MB across the pair) to arm a
    check with no subject. See the module docstring.

    Failure aborts before a byte is written."""
    out = {}
    v9_cm = ROOT / V9_COLORMAPS
    built = json.dumps(library, indent=1)
    if built != v9_cm.read_text(encoding="utf-8"):
        raise SystemExit(f"colormap library differs from {V9_COLORMAPS} — the palette "
                         f"sources moved under the recipe")
    out["colormaps_identical_to_v9"] = True
    out["gate_a_prefix_plan_parity"] = (
        "RETIRED 2026-08-08 with data/v{9,10}/aug_cache; the committed "
        "data/v10/aug_roster.json keeps the counts it passed on")
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
    parity = assert_recipe_parity(library)

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
            "were unnamed by this plan (EXPECTED orphans while the tree existed)."),
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
    # PLAN_OUT and CACHE_MANIFEST_OUT are bulk(), not durable(), as of 2026-08-08 — the same
    # move v8's pair made on 2026-08-03 and v9's earlier the same day, for the same reason and
    # on the same kind of proof: this very function reproduces both BYTE-IDENTICALLY from
    # data/v10/manifest.jsonl plus v9's committed roster and colormaps (rebuilt over the
    # originals and sha256-equal on both, 2026-08-08). Their .gitignore negations went with the
    # files, so durable() would now REFUSE them — the trap v8's deletion sprang, where the
    # rebuild the deletion's argument rests on could not itself complete. Neither is a
    # RELOCATED_PREFIXES entry, so bulk() resolves them IN-TREE at the same path every reader
    # already opens (tools/v10/{eval_v10,prereg,diagnose_selection}.py), merely untracked.
    #
    # The other three outputs stay durable() and stay tracked: `aug_roster.json` and
    # `build_metadata.json` are NOT byte-reproducible — they carry GATE A's frozen counts
    # (`prefix_plan_rows_byte_identical_to_v9: 170760`), which this builder deliberately no
    # longer computes now that the gate is retired, so a re-run rewrites the record with the
    # retirement note instead. Restore them after any rebuild; the roster is the record.
    for rel, rws in ((PLAN_OUT, plan_rows), (CACHE_MANIFEST_OUT, cm_rows)):
        p = paths.bulk(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in rws:
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
