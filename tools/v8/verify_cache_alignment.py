#!/usr/bin/env python
"""Assert tile<->location agreement in BOTH directions before a v8 train.

The re-split (loose0_v3 -> eval, 24 biased mandelbrot neighbours dropped) rewrote the
manifest, plan and cache_manifest but must NOT have re-rendered or renumbered anything: the
171,384-tile aug cache is keyed on loc_id, and a silent renumber would train on tiles that
belong to a different location — plausible numbers, wrong model.

Checks (all ABORT on failure):

  FORWARD  (location -> tile). Every location in the NEW manifest keeps its identity at its
           loc_id, and the tiles on disk were rendered at exactly the coordinates the NEW
           plan expects. Proven by comparing the NEW plan to the PRIOR plan keyed on the
           output path: for every new plan row, the prior plan row at the same path is
           byte-identical (same cx/cy/fw/palette/ss/... => the tile already on disk from the
           prior render is the tile the new plan wants). Then every new plan tile exists.
  BACKWARD (tile -> location). Every row of the NEW cache_manifest resolves to a file on
           disk, and its loc_id is present in the NEW manifest (no orphan reference).
  CENSUS   the 144 census locations are reproduced location-for-location vs the prior
           eval_slice (identity set equality), and each keeps its prior loc_id.
  COUNTS   every manifest loc_id has exactly SLOTS(=24) plan rows and 24 tiles on disk.

PRECONDITION (as of 2026-07-31): the v8 aug cache was DELETED — 12.13 GB / 171,384 tiles
reclaimed once v9 was trained and evaluated. Both the FORWARD tiles-on-disk check and the
BACKWARD cache->disk check therefore fail against an empty tree, and that is not a bug in
this script: it is a gate that runs immediately before a v8 train, and a v8 train now
begins by REGENERATING the cache (`uv run python tools/v8/render_cache.py`, ~4.7 h at 6
workers, from the committed data/v8/{plan,cache_manifest,aug_roster,colormaps}). Run this
after that rebuild, not before. This script is not collected by pytest, so the deletion
left the suite green.

Usage:
  uv run python tools/v8/verify_cache_alignment.py --prior-plan <backup/plan.jsonl> \
       --prior-eval <backup/eval_slice.jsonl>
The prior plan/eval_slice are the pre-re-split copies (backed up before the rebuild); the
prior plan is the record of what was actually rendered to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402

MANIFEST = ROOT / "data/v8/manifest.jsonl"
EVAL_SLICE = ROOT / "data/v8/eval_slice.jsonl"
PLAN = ROOT / "data/v8/plan.jsonl"
CACHE = ROOT / "data/v8/cache_manifest.jsonl"
SLOTS = 24


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def ident(r):
    return (r["fractal_type"], r["cx"], r["cy"], r["fw"], r.get("c_re"), r.get("c_im"),
            tuple(r.get(k) for k in loc_mod.family_param_keys(r["fractal_type"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-plan", required=True)
    ap.add_argument("--prior-eval", required=True)
    a = ap.parse_args()

    fails = []

    manifest = load_jsonl(MANIFEST)
    man_by_id = {r["loc_id"]: r for r in manifest}
    assert len(man_by_id) == len(manifest), "duplicate loc_id in manifest"
    print(f"manifest: {len(manifest)} locations")

    new_plan = load_jsonl(PLAN)
    prior_plan = load_jsonl(a.prior_plan)
    new_by_out = {r["out"]: r for r in new_plan}
    prior_by_out = {r["out"]: r for r in prior_plan}
    assert len(new_by_out) == len(new_plan), "duplicate out path in new plan"
    print(f"plan: new {len(new_plan)} rows, prior {len(prior_plan)} rows")

    # -------- FORWARD: new plan row == prior plan row at the same path --------
    missing_in_prior = 0
    mismatched = 0
    for out, nr in new_by_out.items():
        pr = prior_by_out.get(out)
        if pr is None:
            missing_in_prior += 1
            if missing_in_prior <= 3:
                fails.append(f"FORWARD: new plan path not in prior render: {out}")
            continue
        if nr != pr:
            mismatched += 1
            if mismatched <= 3:
                diff = {k: (pr.get(k), nr.get(k)) for k in set(nr) | set(pr) if nr.get(k) != pr.get(k)}
                fails.append(f"FORWARD: plan row DIFFERS at {out}: {diff}")
    if missing_in_prior:
        fails.append(f"FORWARD: {missing_in_prior} new plan paths were never rendered (not in prior plan)")
    if mismatched:
        fails.append(f"FORWARD: {mismatched} new plan rows differ from the prior render at the same path")
    print(f"  FORWARD plan-vs-prior: {len(new_by_out)-missing_in_prior-mismatched}/{len(new_by_out)} "
          f"rows byte-identical to the prior render  ({missing_in_prior} missing, {mismatched} differ)")

    # every new plan tile exists on disk (resolved out-of-tree)
    n_disk_missing = 0
    for out in new_by_out:
        if not Path(out).exists():
            n_disk_missing += 1
            if n_disk_missing <= 3:
                fails.append(f"FORWARD: plan tile absent on disk: {out}")
    if n_disk_missing:
        fails.append(f"FORWARD: {n_disk_missing} plan tiles missing on disk")
    print(f"  FORWARD tiles-on-disk: {len(new_by_out)-n_disk_missing}/{len(new_by_out)} exist")

    # -------- BACKWARD: every cache_manifest row -> file on disk, loc_id in manifest --------
    cache = load_jsonl(CACHE)
    assert len(cache) == len(new_plan), f"cache rows {len(cache)} != plan rows {len(new_plan)}"
    cm_missing_loc = 0
    cm_disk_missing = 0
    for c in cache:
        if c["location_id"] not in man_by_id:
            cm_missing_loc += 1
            if cm_missing_loc <= 3:
                fails.append(f"BACKWARD: cache loc_id {c['location_id']} not in manifest")
        f = paths.bulk(c["path"])
        if not f.exists():
            cm_disk_missing += 1
            if cm_disk_missing <= 3:
                fails.append(f"BACKWARD: cache tile absent on disk: {c['path']}")
    if cm_missing_loc:
        fails.append(f"BACKWARD: {cm_missing_loc} cache rows reference a loc_id not in the manifest")
    if cm_disk_missing:
        fails.append(f"BACKWARD: {cm_disk_missing} cache tiles missing on disk")
    print(f"  BACKWARD cache->manifest: {len(cache)-cm_missing_loc}/{len(cache)} loc_ids present")
    print(f"  BACKWARD cache->disk:     {len(cache)-cm_disk_missing}/{len(cache)} tiles exist")

    # cache split/group/label must equal the manifest's for that loc_id (trainer reads cache)
    field_contra = 0
    for c in cache:
        m = man_by_id.get(c["location_id"])
        if m and (c["split"] != m["split"] or c["group_id"] != m["group_id"]
                  or c["label"] != m["label"] or bool(c["biased"]) != bool(m["biased"])):
            field_contra += 1
            if field_contra <= 3:
                fails.append(f"CACHE/MANIFEST split-group-label mismatch at loc {c['location_id']}: "
                             f"cache(split={c['split']},grp={c['group_id']},lbl={c['label']}) "
                             f"vs manifest(split={m['split']},grp={m['group_id']},lbl={m['label']})")
    if field_contra:
        fails.append(f"CACHE/MANIFEST: {field_contra} cache rows disagree with the manifest on split/group/label/biased")
    print(f"  CACHE fields vs manifest: {len(cache)-field_contra}/{len(cache)} agree on split/group/label/biased")

    # -------- COUNTS: exactly 24 plan rows + 24 tiles per manifest loc_id --------
    plan_per_loc = Counter()
    for out in new_by_out:
        # loc_id is the dir under aug_cache/
        lid = int(Path(out).parent.name)
        plan_per_loc[lid] += 1
    bad_counts = [lid for lid in man_by_id if plan_per_loc.get(lid, 0) != SLOTS]
    if bad_counts:
        fails.append(f"COUNTS: {len(bad_counts)} loc_ids do not have exactly {SLOTS} plan rows, "
                     f"e.g. {bad_counts[:5]}")
    print(f"  COUNTS: {len(man_by_id)-len(bad_counts)}/{len(man_by_id)} locations have exactly {SLOTS} tiles")

    # -------- CENSUS: reproduced location-for-location vs the prior eval_slice --------
    new_ev = load_jsonl(EVAL_SLICE)
    prior_ev = load_jsonl(a.prior_eval)
    new_census = {ident(r): r for r in new_ev if r["source"] == "prospect_census"}
    prior_census = {ident(r): r for r in prior_ev if r["source"] == "prospect_census"}
    only_new = set(new_census) - set(prior_census)
    only_old = set(prior_census) - set(new_census)
    if only_new or only_old:
        fails.append(f"CENSUS: identity set changed — {len(only_old)} lost, {len(only_new)} gained")
    # each census loc_id preserved
    id_moved = 0
    for k, r in new_census.items():
        if k in prior_census and r["loc_id"] != prior_census[k]["loc_id"]:
            id_moved += 1
    if id_moved:
        fails.append(f"CENSUS: {id_moved} census locations changed loc_id (cache would misalign)")
    print(f"  CENSUS: {len(new_census)} locations reproduced (expect 144), "
          f"{len(new_census)-id_moved} keep prior loc_id")

    # -------- report --------
    print()
    if fails:
        print("!!! CACHE ALIGNMENT FAILED — DO NOT TRAIN !!!")
        for f in fails[:40]:
            print("   -", f)
        return 1
    orphan_dirs = 0  # informational: prior tiles for dropped locations remain on disk, unreferenced
    print("OK — tile<->location agreement holds in BOTH directions. Safe to train.")
    print(f"  ({len(prior_by_out) - len(new_by_out)} prior tiles belong to dropped locations "
          f"and are now unreferenced on disk — expected, not re-rendered.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
