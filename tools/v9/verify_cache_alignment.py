#!/usr/bin/env python
"""Assert the v9 aug cache is complete, intact, uniform in cap, and aligned BOTH ways.

v9 is the v8 corpus re-rendered at the raised iteration cap (docs/design/auto_maxiter.md).
Nothing about the population moved — same manifest, same split, same `loc_id`s, same
24-slot recipe — so the checks are v8's, plus the three the cap raise makes necessary.

  FORWARD  (location -> tile). Every v9 plan row is its v8 twin except on `maxiter` and
           `out`; the slot-identity set (loc_id, slot filename) is identical; and every
           v9 plan tile exists on disk. The v8 plan is the record of what the recipe WAS,
           so comparing against it is what makes "only the cap changed" a checked claim
           rather than a sentence in a commit message.
  BACKWARD (tile -> location). Every cache_manifest row resolves to a file on disk, its
           loc_id is in the manifest, and its split/group/label/biased agree with the
           manifest's — the trainer reads the cache, not the manifest, so a disagreement
           here is a silently mislabeled corpus.
  COUNTS   every manifest loc_id has exactly SLOTS(=24) plan rows and 24 tiles on disk,
           and the cache tree holds NO location dir the manifest does not name. (The v8
           tree carries 24 orphan dirs from a pre-re-split manifest; that is how a tile
           count can exceed a plan and still read as "complete".)
  CENSUS   the 144 census locations are present with their loc_ids intact — the eval
           instrument must be the same instrument, or the v9-vs-v8 comparison measures
           the instrument instead of the model.
  CAP      every tile's cap is the LIVE policy's `auto_maxiter(fw_slot)`, RECOMPUTED here
           rather than read back from the plan, and no tile survives at the superseded
           flat 8000. A mixed-cap corpus is poison in the same way a mixed-decode readout
           is, and it is completely invisible in a JPEG.
  INTACT   no `.tmp` debris, no zero-byte or truncated tiles. Truncation is checked by the
           JPEG EOI marker (FFD9), not by a size threshold: a half-written tile of a dense
           location can be larger than a complete tile of a sparse one, so size alone
           cannot tell them apart.

All checks ABORT on failure; exit 0 means safe to train.

  uv run python tools/v9/verify_cache_alignment.py
  uv run python tools/v9/verify_cache_alignment.py --skip-intact   # faster, much weaker
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
import active_ckpt as ac      # noqa: E402
import location as loc_mod    # noqa: E402
import paths                  # noqa: E402

MANIFEST = ROOT / "data/v8/manifest.jsonl"          # v9 reads v8's, on purpose
EVAL_SLICE = ROOT / "data/v8/eval_slice.jsonl"
V8_PLAN = ROOT / "data/v8/plan.jsonl"
PLAN = ROOT / "data/v9/plan.jsonl"
CACHE = ROOT / "data/v9/cache_manifest.jsonl"
CACHE_ROOT = paths.bulk("data/v9/aug_cache")
SLOTS = 24
FLAT_OLD_MAXITER = 8000
# Fields a v9 plan row may differ from its v8 twin on. Anything else is a recipe change.
PLAN_DELTA_ALLOWED = {"maxiter", "out"}
JPEG_EOI = b"\xff\xd9"


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def ident(r):
    return (r["fractal_type"], r["cx"], r["cy"], r["fw"], r.get("c_re"), r.get("c_im"),
            tuple(r.get(k) for k in loc_mod.family_param_keys(r["fractal_type"])))


def slot_key(r):
    """(loc_id, slot filename) — a plan row's recipe identity, independent of the tree."""
    p = Path(r["out"])
    return (p.parent.name, p.name)


def rel_key(p: str) -> str:
    q = Path(p)
    return f"{q.parent.name}/{q.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-intact", action="store_true",
                    help="skip the per-tile size/EOI read (much faster, much weaker)")
    a = ap.parse_args()
    fails = []

    manifest = load_jsonl(MANIFEST)
    man_by_id = {r["loc_id"]: r for r in manifest}
    assert len(man_by_id) == len(manifest), "duplicate loc_id in manifest"
    print(f"manifest: {len(manifest)} locations  ({MANIFEST.relative_to(ROOT).as_posix()})")

    plan = load_jsonl(PLAN)
    new_by_out = {r["out"]: r for r in plan}
    assert len(new_by_out) == len(plan), "duplicate out path in v9 plan"
    print(f"plan    : {len(plan)} rows")

    # -------- FORWARD: recipe parity vs v8, then existence on disk --------
    v8_plan = load_jsonl(V8_PLAN)
    if len(v8_plan) != len(plan):
        fails.append(f"FORWARD: v9 plan has {len(plan)} rows, v8 had {len(v8_plan)} — "
                     f"the corpus moved, not just the cap")
    v8_by_slot = {slot_key(r): r for r in v8_plan}
    new_by_slot = {slot_key(r): r for r in plan}
    lost, gained = set(v8_by_slot) - set(new_by_slot), set(new_by_slot) - set(v8_by_slot)
    if lost or gained:
        fails.append(f"FORWARD: slot-identity set changed — {len(lost)} lost, "
                     f"{len(gained)} gained (e.g. {sorted(gained)[:2]} / {sorted(lost)[:2]})")
    off_recipe = cap_moved = 0
    for k, nr in new_by_slot.items():
        pr = v8_by_slot.get(k)
        if pr is None:
            continue
        diff = {f for f in set(nr) | set(pr) if nr.get(f) != pr.get(f)}
        if diff - PLAN_DELTA_ALLOWED:
            off_recipe += 1
            if off_recipe <= 3:
                fails.append(f"FORWARD: plan row differs from v8 outside "
                             f"{sorted(PLAN_DELTA_ALLOWED)} at {k}: "
                             f"{sorted(diff - PLAN_DELTA_ALLOWED)}")
        if nr.get("maxiter") != pr.get("maxiter"):
            cap_moved += 1
    if off_recipe:
        fails.append(f"FORWARD: {off_recipe} plan rows differ from v8 outside the cap")
    print(f"  FORWARD recipe-vs-v8 : {len(new_by_slot)-off_recipe}/{len(new_by_slot)} rows "
          f"identical except {sorted(PLAN_DELTA_ALLOWED)}  ({cap_moved} carry a moved cap)")

    n_disk_missing = 0
    for out in new_by_out:
        if not Path(out).exists():
            n_disk_missing += 1
            if n_disk_missing <= 3:
                fails.append(f"FORWARD: plan tile absent on disk: {out}")
    if n_disk_missing:
        fails.append(f"FORWARD: {n_disk_missing} plan tiles missing on disk")
    print(f"  FORWARD tiles-on-disk: {len(new_by_out)-n_disk_missing}/{len(new_by_out)} exist")

    # -------- BACKWARD: cache row -> file on disk, loc_id in manifest --------
    cache = load_jsonl(CACHE)
    if len(cache) != len(plan):
        fails.append(f"BACKWARD: cache rows {len(cache)} != plan rows {len(plan)}")
    cm_missing_loc = cm_disk_missing = 0
    for c in cache:
        if c["location_id"] not in man_by_id:
            cm_missing_loc += 1
            if cm_missing_loc <= 3:
                fails.append(f"BACKWARD: cache loc_id {c['location_id']} not in manifest")
        if not paths.bulk(c["path"]).exists():
            cm_disk_missing += 1
            if cm_disk_missing <= 3:
                fails.append(f"BACKWARD: cache tile absent on disk: {c['path']}")
    if cm_missing_loc:
        fails.append(f"BACKWARD: {cm_missing_loc} cache rows reference an unknown loc_id")
    if cm_disk_missing:
        fails.append(f"BACKWARD: {cm_disk_missing} cache tiles missing on disk")
    print(f"  BACKWARD cache->manifest: {len(cache)-cm_missing_loc}/{len(cache)} loc_ids present")
    print(f"  BACKWARD cache->disk    : {len(cache)-cm_disk_missing}/{len(cache)} tiles exist")

    field_contra = 0
    for c in cache:
        m = man_by_id.get(c["location_id"])
        if m and (c["split"] != m["split"] or c["group_id"] != m["group_id"]
                  or c["label"] != m["label"] or bool(c["biased"]) != bool(m["biased"])):
            field_contra += 1
            if field_contra <= 3:
                fails.append(f"CACHE/MANIFEST mismatch at loc {c['location_id']}: "
                             f"cache(split={c['split']},grp={c['group_id']},lbl={c['label']}) "
                             f"vs manifest(split={m['split']},grp={m['group_id']},lbl={m['label']})")
    if field_contra:
        fails.append(f"CACHE/MANIFEST: {field_contra} rows disagree on split/group/label/biased")
    print(f"  CACHE fields vs manifest: {len(cache)-field_contra}/{len(cache)} agree")

    # -------- CAP: uniform, live-policy, no survivor at the superseded flat cap --------
    wrong_cap = at_flat = 0
    for r in plan:
        want = int(ac.auto_maxiter(float(r["fw"])))
        if int(r["maxiter"]) != want:
            wrong_cap += 1
            if wrong_cap <= 3:
                fails.append(f"CAP: {r['out']} has maxiter {r['maxiter']}, live policy "
                             f"resolves {want} at fw={r['fw']}")
        if int(r["maxiter"]) == FLAT_OLD_MAXITER and want != FLAT_OLD_MAXITER:
            at_flat += 1
    if wrong_cap:
        fails.append(f"CAP: {wrong_cap} plan rows are not at the live policy's cap — the "
                     f"plan and the cap policy have drifted apart")
    if at_flat:
        fails.append(f"CAP: {at_flat} rows still sit at the superseded flat {FLAT_OLD_MAXITER}")
    caps = [int(r["maxiter"]) for r in plan]
    print(f"  CAP: {len(plan)-wrong_cap}/{len(plan)} rows at the live policy  "
          f"(range {min(caps)}..{max(caps)}; v8 was flat {FLAT_OLD_MAXITER})")
    plan_cap_by_rel = {rel_key(r["out"]): r["maxiter"] for r in plan}
    cm_cap_bad = sum(1 for c in cache
                     if c.get("maxiter") != plan_cap_by_rel.get(rel_key(c["path"])))
    if cm_cap_bad:
        fails.append(f"CAP: {cm_cap_bad} cache_manifest rows record a cap the plan disagrees with")

    # -------- COUNTS: 24 plan rows + 24 tiles per loc_id, and no orphan dirs --------
    plan_per_loc = Counter(int(Path(o).parent.name) for o in new_by_out)
    bad_counts = [lid for lid in man_by_id if plan_per_loc.get(lid, 0) != SLOTS]
    if bad_counts:
        fails.append(f"COUNTS: {len(bad_counts)} loc_ids lack exactly {SLOTS} plan rows, "
                     f"e.g. {bad_counts[:5]}")
    on_disk_dirs, orphan_dirs, short_dirs = 0, [], []
    if CACHE_ROOT.exists():
        for d in CACHE_ROOT.iterdir():
            if not d.is_dir():
                continue
            on_disk_dirs += 1
            try:
                lid = int(d.name)
            except ValueError:
                orphan_dirs.append(d.name)
                continue
            if lid not in man_by_id:
                orphan_dirs.append(d.name)
                continue
            n = sum(1 for f in d.iterdir() if f.name.endswith(".jpg"))
            if n != SLOTS:
                short_dirs.append((lid, n))
    if orphan_dirs:
        fails.append(f"COUNTS: {len(orphan_dirs)} cache dirs the manifest does not name "
                     f"(e.g. {orphan_dirs[:5]}) — a stale dir is how a tile count can "
                     f"exceed the plan and still read as complete")
    if short_dirs:
        fails.append(f"COUNTS: {len(short_dirs)} location dirs do not hold exactly {SLOTS} "
                     f"tiles, e.g. {short_dirs[:5]}")
    print(f"  COUNTS: {len(man_by_id)-len(bad_counts)}/{len(man_by_id)} locations have "
          f"{SLOTS} plan rows; {on_disk_dirs} dirs on disk, {len(orphan_dirs)} orphan, "
          f"{len(short_dirs)} short")

    # -------- CENSUS: the eval instrument is the same instrument --------
    ev = load_jsonl(EVAL_SLICE)
    census = {ident(r): r for r in ev if r["source"] == "prospect_census"}
    census_ids = {r["loc_id"] for r in census.values()}
    missing = census_ids - set(man_by_id)
    if missing:
        fails.append(f"CENSUS: {len(missing)} census loc_ids absent from the manifest")
    census_tiles = sum(1 for c in cache if c["location_id"] in census_ids)
    print(f"  CENSUS: {len(census)} locations (expect 144), {census_tiles} tiles "
          f"(expect {len(census)*SLOTS})")
    if census_tiles != len(census) * SLOTS:
        fails.append(f"CENSUS: {census_tiles} tiles for {len(census)} locations, "
                     f"expected {len(census)*SLOTS}")

    # -------- INTACT: no .tmp debris, no zero-byte or truncated tiles --------
    if not a.skip_intact and CACHE_ROOT.exists():
        tmp_debris, zero, truncated, checked = [], [], [], 0
        for root, _dirs, files in os.walk(CACHE_ROOT):
            for fn in files:
                fp = Path(root, fn)
                if fn.endswith(".tmp"):
                    tmp_debris.append(str(fp))
                    continue
                if not fn.endswith(".jpg"):
                    continue
                checked += 1
                try:
                    if fp.stat().st_size == 0:
                        zero.append(str(fp))
                        continue
                    with fp.open("rb") as fh:
                        fh.seek(-2, os.SEEK_END)
                        if fh.read(2) != JPEG_EOI:
                            truncated.append(str(fp))
                except OSError as e:
                    truncated.append(f"{fp} ({e})")
        if tmp_debris:
            fails.append(f"INTACT: {len(tmp_debris)} .tmp files left behind, "
                         f"e.g. {tmp_debris[:3]}")
        if zero:
            fails.append(f"INTACT: {len(zero)} zero-byte tiles, e.g. {zero[:3]}")
        if truncated:
            fails.append(f"INTACT: {len(truncated)} tiles without a JPEG EOI marker "
                         f"(truncated encode), e.g. {truncated[:3]}")
        print(f"  INTACT: {checked} tiles read; {len(tmp_debris)} .tmp, {len(zero)} zero-byte, "
              f"{len(truncated)} truncated")
    else:
        print("  INTACT: SKIPPED (--skip-intact) — completeness proven, integrity not")

    print()
    if fails:
        print("!!! V9 CACHE VERIFICATION FAILED — DO NOT TRAIN !!!")
        for f in fails[:40]:
            print("   -", f)
        return 1
    print("OK — v9 cache complete, intact, uniform in cap, aligned in BOTH directions.")
    print(f"     {len(plan)} tiles over {len(man_by_id)} locations at "
          f"{min(caps)}..{max(caps)} iterations.  Safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
