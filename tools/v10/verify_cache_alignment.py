#!/usr/bin/env python
"""Assert the v10 aug cache is complete, intact, correctly capped, and aligned BOTH ways.

v10 spans TWO trees — `data/v9/aug_cache` (the 7,115 prefix locations, reused unrendered)
and `data/v10/aug_cache` (the 1,267 appended ones) — so the checks that matter are the
ones that would catch a tile pointing at the wrong location. The `loc_id` renumbering seam
is the silent failure this file exists for: nothing about a JPEG says which location it
belongs to, and a manifest that renumbered while a cache did not would train on a corpus
whose labels are shuffled with respect to its images, with no symptom but a worse model.

  FORWARD  (location -> tile) every prefix plan row is BYTE-IDENTICAL to its v9 row, and
           every plan tile of either tree exists on disk.
  BACKWARD (tile -> location) every cache_manifest row resolves to a file, its loc_id is
           in the manifest, and its split/group/label/biased agree with the manifest's.
  BIJECTION the two directions are asserted as a bijection on (loc_id, slot filename), not
           just as two containments: |plan| == |cache| == |manifest| x 24 AND the slot sets
           are equal AND each is duplicate-free. Two containments with a duplicate on one
           side pass while the correspondence is broken; this is the check the prompt asks
           for in BOTH directions.
  IDENTITY  a location's tile directory is named by its loc_id, and its manifest row's
           COORDINATES are re-derived and compared against the plan row's cx/cy/fw for the
           identity slot. This is the actual anti-renumbering check: if the manifest were
           renumbered under the cache, the tiles in dir <n> would carry another location's
           geometry and this comparison fails.
  ORPHANS   the v9 tree legitimately holds tiles the v10 manifest does not name — the 48
            belonging to the two displaced v8 locations (build_manifest.py GATE 11). They
            are named as EXPECTED here. Any OTHER unnamed dir is an abort, because a stale
            dir is how a tile count exceeds a plan and still reads as complete.
  CAP       every tile's cap is the LIVE policy's `auto_maxiter(fw_slot)`, RECOMPUTED here
            rather than read back from the plan.
  CENSUS    the three eval instruments are present at their registered sizes with their
            loc_ids intact — census 144, mandelbrot floor 526, maneuver-uniform 90.
  INTACT    no `.tmp` debris, no zero-byte or truncated tiles (JPEG EOI marker, not size:
            a half-written dense tile can exceed a complete sparse one).

All checks ABORT on failure; exit 0 means safe to train.

  uv run python tools/v10/verify_cache_alignment.py
  uv run python tools/v10/verify_cache_alignment.py --skip-intact   # faster, much weaker
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

MANIFEST = ROOT / "data/v10/manifest.jsonl"
EVAL_SLICE = ROOT / "data/v10/eval_slice.jsonl"
BUILD_META = ROOT / "data/v10/build_metadata.json"
V9_PLAN = ROOT / "data/v9/plan.jsonl"
PLAN = ROOT / "data/v10/plan.jsonl"
CACHE = ROOT / "data/v10/cache_manifest.jsonl"
CACHE_ROOTS = (paths.bulk("data/v9/aug_cache"), paths.bulk("data/v10/aug_cache"))
SLOTS = 24
PREFIX_MAX_LOC_ID = 7140
JPEG_EOI = b"\xff\xd9"
INSTRUMENTS = {"prospect_census": 144, "loose0_v3_floor": 526, "maneuver_uniform_v1": 90}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def slot_key(out: str):
    p = Path(out)
    return (p.parent.name, p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-intact", action="store_true")
    a = ap.parse_args()
    fails = []

    manifest = load_jsonl(MANIFEST)
    man_by_id = {r["loc_id"]: r for r in manifest}
    assert len(man_by_id) == len(manifest), "duplicate loc_id in manifest"
    print(f"manifest: {len(manifest)} locations")

    plan = load_jsonl(PLAN)
    cache = load_jsonl(CACHE)
    print(f"plan    : {len(plan)} rows   cache_manifest: {len(cache)} rows")

    # -------- FORWARD: prefix parity vs v9, then existence on disk --------
    v9_by_out = {r["out"]: r for r in load_jsonl(V9_PLAN)}
    prefix = [r for r in plan if int(Path(r["out"]).parent.name) <= PREFIX_MAX_LOC_ID]
    off, absent = 0, 0
    for r in prefix:
        o = v9_by_out.get(r["out"])
        if o is None:
            absent += 1
        elif r != o:
            off += 1
            if off <= 3:
                fails.append(f"FORWARD: prefix row differs from v9 at {r['out']}")
    if absent:
        fails.append(f"FORWARD: {absent} prefix rows name a tile v9's plan never had")
    if off:
        fails.append(f"FORWARD: {off} prefix rows differ from v9 — the tiles on disk are "
                     f"not the tiles this plan asks for")
    print(f"  FORWARD prefix-vs-v9 : {len(prefix)-off-absent}/{len(prefix)} byte-identical")

    n_missing = sum(1 for r in plan if not Path(r["out"]).exists())
    if n_missing:
        ex = [r["out"] for r in plan if not Path(r["out"]).exists()][:3]
        fails.append(f"FORWARD: {n_missing} plan tiles missing on disk, e.g. {ex}")
    print(f"  FORWARD tiles-on-disk: {len(plan)-n_missing}/{len(plan)} exist")

    # -------- BACKWARD: cache row -> file, loc_id -> manifest --------
    cm_missing_loc = sum(1 for c in cache if c["location_id"] not in man_by_id)
    cm_disk_missing = sum(1 for c in cache if not paths.bulk(c["path"]).exists())
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
                fails.append(
                    f"CACHE/MANIFEST mismatch at loc {c['location_id']}: "
                    f"cache(split={c['split']},grp={c['group_id']},lbl={c['label']}) vs "
                    f"manifest(split={m['split']},grp={m['group_id']},lbl={m['label']})")
    if field_contra:
        fails.append(f"CACHE/MANIFEST: {field_contra} rows disagree on split/group/label/biased")
    print(f"  CACHE fields vs manifest: {len(cache)-field_contra}/{len(cache)} agree")

    # -------- BIJECTION: equality of slot sets, not two containments --------
    plan_slots = [slot_key(r["out"]) for r in plan]
    cache_slots = [slot_key(c["path"]) for c in cache]
    ps, cs = set(plan_slots), set(cache_slots)
    if len(ps) != len(plan_slots):
        fails.append(f"BIJECTION: {len(plan_slots)-len(ps)} duplicate slot keys in the plan")
    if len(cs) != len(cache_slots):
        fails.append(f"BIJECTION: {len(cache_slots)-len(cs)} duplicate slot keys in the cache")
    if ps != cs:
        fails.append(f"BIJECTION: plan and cache slot sets differ — {len(ps-cs)} plan-only, "
                     f"{len(cs-ps)} cache-only (e.g. {sorted(ps-cs)[:2]} / {sorted(cs-ps)[:2]})")
    want = len(manifest) * SLOTS
    if not (len(plan) == len(cache) == want):
        fails.append(f"BIJECTION: |plan|={len(plan)} |cache|={len(cache)} != "
                     f"{len(manifest)}x{SLOTS}={want}")
    print(f"  BIJECTION: |plan|=|cache|={len(plan)} == {len(manifest)}x{SLOTS}, "
          f"slot sets {'EQUAL' if ps == cs else 'DIFFER'}, 0 dupes")

    # -------- IDENTITY: the tile dir's loc_id really carries that location's geometry --
    # THE anti-renumbering check. For each location, find its identity plan row (scale
    # exactly 1.0, centred) and compare that row's cx/cy/fw against the manifest row's.
    ident_rows = {}
    for r in plan:
        lid = int(Path(r["out"]).parent.name)
        if "__id__s1.0000__sh0.0000__" in Path(r["out"]).name:
            ident_rows.setdefault(lid, r)
    geom_bad, no_ident = 0, 0
    for lid, m in man_by_id.items():
        r = ident_rows.get(lid)
        if r is None:
            no_ident += 1
            continue
        if (r["cx"] != repr(float(m["cx"])) or r["cy"] != repr(float(m["cy"]))
                or r["fw"] != repr(float(m["fw"])) or r["fractal_type"] != m["fractal_type"]):
            geom_bad += 1
            if geom_bad <= 3:
                fails.append(
                    f"IDENTITY: loc_id {lid} tile dir carries geometry "
                    f"({r['fractal_type']} {r['cx']},{r['cy']} fw {r['fw']}) but the "
                    f"manifest says ({m['fractal_type']} {m['cx']},{m['cy']} fw {m['fw']}) "
                    f"— the manifest was renumbered under the cache")
    if no_ident:
        fails.append(f"IDENTITY: {no_ident} locations have no identity-framing plan row")
    if geom_bad:
        fails.append(f"IDENTITY: {geom_bad} locations' tiles carry another location's geometry")
    print(f"  IDENTITY: {len(man_by_id)-geom_bad-no_ident}/{len(man_by_id)} locations' "
          f"identity tile matches the manifest's own coordinates")

    # -------- COUNTS + ORPHANS across both trees --------
    plan_per_loc = Counter(int(Path(r["out"]).parent.name) for r in plan)
    bad_counts = [lid for lid in man_by_id if plan_per_loc.get(lid, 0) != SLOTS]
    if bad_counts:
        fails.append(f"COUNTS: {len(bad_counts)} loc_ids lack exactly {SLOTS} plan rows, "
                     f"e.g. {bad_counts[:5]}")
    meta = json.loads(BUILD_META.read_text(encoding="utf-8"))
    expected_orphans = {r["loc_id"] for r in
                        meta.get("displaced_prefix_rows", {}).get("rows", [])}
    dirs, orphans, short = 0, [], []
    for cache_root in CACHE_ROOTS:
        if not cache_root.exists():
            continue
        for d in cache_root.iterdir():
            if not d.is_dir():
                continue
            dirs += 1
            try:
                lid = int(d.name)
            except ValueError:
                orphans.append(d.name)
                continue
            if lid not in man_by_id:
                if lid not in expected_orphans:
                    orphans.append(d.name)
                continue
            n = sum(1 for f in d.iterdir() if f.name.endswith(".jpg"))
            if n != SLOTS:
                short.append((lid, n))
    if orphans:
        fails.append(f"COUNTS: {len(orphans)} cache dirs neither named by the manifest nor "
                     f"declared displaced (e.g. {orphans[:5]})")
    if short:
        fails.append(f"COUNTS: {len(short)} location dirs lack exactly {SLOTS} tiles, "
                     f"e.g. {short[:5]}")
    print(f"  COUNTS: {len(man_by_id)-len(bad_counts)}/{len(man_by_id)} locations have "
          f"{SLOTS} plan rows; {dirs} dirs across both trees, {len(orphans)} unexplained "
          f"orphan, {len(expected_orphans)} declared displaced, {len(short)} short")

    # -------- CAP: recomputed from the live policy, never read back --------
    wrong_cap = 0
    for r in plan:
        want_mit = int(ac.auto_maxiter(float(r["fw"])))
        if int(r["maxiter"]) != want_mit:
            wrong_cap += 1
            if wrong_cap <= 3:
                fails.append(f"CAP: {r['out']} has maxiter {r['maxiter']}, live policy "
                             f"resolves {want_mit} at fw={r['fw']}")
    if wrong_cap:
        fails.append(f"CAP: {wrong_cap} plan rows are not at the live policy's cap")
    caps = [int(r["maxiter"]) for r in plan]
    print(f"  CAP: {len(plan)-wrong_cap}/{len(plan)} rows at the live policy "
          f"(range {min(caps)}..{max(caps)})")

    # -------- INSTRUMENTS --------
    ev = load_jsonl(EVAL_SLICE)
    by_src = Counter(r["source"] for r in ev)
    for src, want_n in INSTRUMENTS.items():
        got = by_src.get(src, 0)
        ids = {r["loc_id"] for r in ev if r["source"] == src}
        n_tiles = sum(1 for c in cache if c["location_id"] in ids)
        ok = got == want_n and n_tiles == want_n * SLOTS and ids <= set(man_by_id)
        if not ok:
            fails.append(f"INSTRUMENT {src}: {got} locations (expect {want_n}), "
                         f"{n_tiles} tiles (expect {want_n*SLOTS})")
        print(f"  INSTRUMENT {src:<22} {got:>4} loc / {n_tiles:>5} tiles  "
              f"{'OK' if ok else 'FAIL'}")

    # -------- INTACT --------
    if not a.skip_intact:
        tmp, zero, trunc, checked = [], [], [], 0
        for cache_root in CACHE_ROOTS:
            if not cache_root.exists():
                continue
            for root, _dirs, files in os.walk(cache_root):
                for fn in files:
                    fp = Path(root, fn)
                    if fn.endswith(".tmp"):
                        tmp.append(str(fp))
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
                                trunc.append(str(fp))
                    except OSError as e:
                        trunc.append(f"{fp} ({e})")
        for label, lst in (("`.tmp` files left behind", tmp), ("zero-byte tiles", zero),
                           ("tiles without a JPEG EOI marker", trunc)):
            if lst:
                fails.append(f"INTACT: {len(lst)} {label}, e.g. {lst[:3]}")
        print(f"  INTACT: {checked} tiles read; {len(tmp)} .tmp, {len(zero)} zero-byte, "
              f"{len(trunc)} truncated")
    else:
        print("  INTACT: SKIPPED (--skip-intact) — completeness proven, integrity not")

    print()
    if fails:
        print("!!! V10 CACHE VERIFICATION FAILED — DO NOT TRAIN !!!")
        for f in fails[:40]:
            print("   -", f)
        return 1
    print("OK — v10 cache complete, intact, correctly capped, aligned in BOTH directions.")
    print(f"     {len(plan)} tiles over {len(man_by_id)} locations at "
          f"{min(caps)}..{max(caps)} iterations.  Safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
