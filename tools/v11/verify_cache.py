#!/usr/bin/env python
r"""v11 cache verification — the checks `verification_practice.md` asks for, on the tree.

Five reads, each stated so a failure names what is wrong rather than that something is:

  1. REPLAY. ~50 tiles sampled across families, re-rendered from their own manifest rows
     through `crop-batch --replay`, compared BYTE for byte. This is the only check that
     proves the recorded geometry is sufficient to reproduce a tile — i.e. that the manifest
     is a recipe and not a description. Sampled across families because the two colour paths
     (location profile / beautiful smooth) are different code, and a replay that only ever
     hits mandelbrot proves one of them.
  2. TILE <-> LOCATION, both directions. Every manifest row's tile exists, and every tile on
     disk is named by a manifest row. One direction alone is how a tree can be complete and
     wrong: missing tiles hide in a count that a stale orphan directory has already inflated.
  3. LOC_ID <-> COORDINATES, both directions, re-asserted ON THE FINISHED CACHE rather than
     trusted from the manifest build — the cache is keyed on `loc_id` alone, so this is the
     property that makes a tile's directory identify its contents.
  4. COVERAGE. Tiles per location (must be exactly 32 everywhere), per-family locations and
     tiles, and the realized palette / AA / quality distributions against the recipe.
  5. SIZE. Bytes on disk and mean KiB/tile, measured from the tree.

`--quick` skips (1), which is the only expensive read (it renders).

  uv run python tools/v11/verify_cache.py [--quick] [--replay-n 50]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import corpus_common as cc  # noqa: E402
import paths                # noqa: E402

MANIFEST = "data/v11/manifest.jsonl"
CACHE_MANIFEST = "data/v11/cache_manifest.jsonl"
CACHE_DIR = "data/v11/aug_cache"
RECIPE = ROOT / "data" / "v11" / "aug_recipe.json"
COLORMAPS = ROOT / "data" / "v11" / "colormaps.json"
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
REPLAY_SEED = 11
OUT = paths.scratch("v11_verify")


def rows(rel):
    return [json.loads(l) for l in paths.bulk(rel).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def check_replay(cm, recipe, n) -> dict:
    """Re-render `n` sampled tiles from their manifest rows and compare bytes."""
    OUT.mkdir(parents=True, exist_ok=True)
    by_fam = defaultdict(list)
    for r in cm:
        by_fam[r["render"]["fractal_type"]].append(r)
    rng = random.Random(REPLAY_SEED)
    # Spread the sample over families rather than over the tree: an even draw would be 40%
    # mandelbrot and might never touch the beautiful-smooth colour path at all.
    per = max(1, n // len(by_fam))
    sample = []
    for fam in sorted(by_fam):
        sample.extend(rng.sample(by_fam[fam], min(per, len(by_fam[fam]))))
    sample.sort(key=lambda r: (r["loc_id"], r["tile"]))   # location-major, as replay expects

    src = OUT / "replay_rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in sample), encoding="utf-8")
    dest = OUT / "replay_tiles"
    dest.mkdir(parents=True, exist_ok=True)
    for p in dest.rglob("*.jpg"):
        p.unlink()
    pr = subprocess.run(
        [str(BIN), "crop-batch", "--replay", str(src), "--replay-out-root", str(dest),
         "--colormaps", str(COLORMAPS)],
        capture_output=True, text=True,
        env=cc.default_engine_env(), creationflags=cc.default_creationflags())
    if pr.returncode != 0:
        return {"ok": False, "error": (pr.stderr or "")[-800:], "n": len(sample)}

    ident, differ, missing = 0, [], []
    for r in sample:
        orig = Path(r["out"])
        # `--replay-out-root` MIRRORS `<loc_id>/<slot>.jpg` rather than writing flat —
        # a tile's basename carries no loc_id, so a flat root would collide on the
        # reserved slot 0 of two locations that drew the same quality.
        repl = dest / str(r["loc_id"]) / orig.name
        if not orig.exists() or not repl.exists():
            missing.append(str(orig))
            continue
        if orig.read_bytes() == repl.read_bytes():
            ident += 1
        else:
            differ.append(str(orig))
    return {"ok": not differ and not missing, "n": len(sample), "byte_identical": ident,
            "differ": differ[:5], "n_differ": len(differ), "missing": missing[:5],
            "n_missing": len(missing),
            "families": {k: sum(1 for r in sample if r["render"]["fractal_type"] == k)
                         for k in sorted(by_fam)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the replay (the render)")
    ap.add_argument("--replay-n", type=int, default=50)
    ap.add_argument("--json-out", default=str(OUT / "verify.json"))
    a = ap.parse_args()

    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    man = rows(MANIFEST)
    cm = rows(CACHE_MANIFEST)
    root = paths.bulk(CACHE_DIR)
    tiles_per_loc = recipe["fan_out"]["tiles_per_location"]
    report, fails = {}, []

    print("=" * 84)
    print(f"v11 CACHE VERIFICATION — {root}")
    print("=" * 84)

    # ---- 2/3. agreement, both directions ----
    on_disk = {}
    total_bytes = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for p in d.glob("*.jpg"):
            on_disk[str(p)] = p.stat().st_size
            total_bytes += on_disk[str(p)]
    planned = {r["out"] for r in cm}
    missing = sorted(planned - set(on_disk))
    orphan = sorted(set(on_disk) - planned)
    report["agreement"] = {"manifest_rows": len(cm), "tiles_on_disk": len(on_disk),
                           "missing": len(missing), "orphan": len(orphan),
                           "missing_examples": missing[:5], "orphan_examples": orphan[:5]}
    ok = not missing and not orphan
    fails += [] if ok else ["tile<->manifest agreement"]
    print(f"  [2] tile <-> manifest        {'OK' if ok else 'FAIL'}  "
          f"({len(cm)} rows, {len(on_disk)} tiles, {len(missing)} missing, "
          f"{len(orphan)} orphan)")

    # loc_id <-> coordinates, on the CACHE's own rows
    fwd, rev, clash = {}, {}, []
    for r in cm:
        rd = r["render"]
        coord = (rd["fractal_type"], rd["cx"], rd["cy"], rd["fw"],
                 rd.get("c_re"), rd.get("c_im"), rd.get("p_re"), rd.get("p_im"),
                 rd.get("zm1_re"), rd.get("zm1_im"))
        i = r["loc_id"]
        if fwd.setdefault(i, coord) != coord:
            clash.append(("loc_id->coords", i))
        if rev.setdefault(coord, i) != i:
            clash.append(("coords->loc_id", coord))
    # and the cache's loc_id set must be the manifest's
    man_ids = {r["loc_id"] for r in man}
    id_diff = man_ids ^ set(fwd)
    ok = not clash and not id_diff
    fails += [] if ok else ["loc_id<->coordinates"]
    report["loc_id_bijection"] = {"loc_ids": len(fwd), "coordinate_sets": len(rev),
                                  "clashes": len(clash), "symmetric_difference_vs_manifest":
                                  len(id_diff), "examples": clash[:5]}
    print(f"  [3] loc_id <-> coordinates   {'OK' if ok else 'FAIL'}  "
          f"(bijective over {len(fwd)} loc_ids on the finished cache; "
          f"{len(id_diff)} differ from the manifest)")

    # ---- 4. coverage ----
    per_loc = Counter(r["loc_id"] for r in cm)
    disk_per_loc = Counter(Path(p).parent.name for p in on_disk)
    wrong = {i: n for i, n in per_loc.items() if n != tiles_per_loc}
    wrong_disk = {i: n for i, n in disk_per_loc.items() if n != tiles_per_loc}
    ok = not wrong and not wrong_disk and len(per_loc) == len(man)
    fails += [] if ok else ["tiles/location == 32"]
    report["coverage"] = {"locations": len(per_loc), "expected_locations": len(man),
                          "tiles_per_location": tiles_per_loc,
                          "locations_off_in_manifest": len(wrong),
                          "locations_off_on_disk": len(wrong_disk),
                          "examples": dict(list(wrong.items())[:5]
                                           or list(wrong_disk.items())[:5])}
    print(f"  [4] tiles/location == {tiles_per_loc}      {'OK' if ok else 'FAIL'}  "
          f"({len(per_loc)}/{len(man)} locations; {len(wrong)} off in the manifest, "
          f"{len(wrong_disk)} off on disk)")

    fam_loc = Counter()
    fam_tile = Counter()
    seen = set()
    for r in cm:
        ft = r["render"]["fractal_type"]
        fam_tile[ft] += 1
        if r["loc_id"] not in seen:
            seen.add(r["loc_id"])
            fam_loc[ft] += 1
    report["per_family"] = {ft: {"locations": fam_loc[ft], "tiles": fam_tile[ft]}
                            for ft in sorted(fam_tile)}
    print("\n      per family:")
    for ft in sorted(fam_tile):
        print(f"        {ft:20s} {fam_loc[ft]:6d} loc  {fam_tile[ft]:8d} tiles")

    # realized axes vs the recipe
    pal = Counter(r["palette"] for r in cm)
    aa = Counter(r["aa"]["level"] for r in cm)
    q = Counter(r["jpg_quality"] for r in cm)
    pool = set(recipe["palettes"]["draw_pool"])
    floor = recipe["fan_out"]["floor_per_location"]
    held = set(recipe["palettes"]["held_out"])
    n_loc = len(per_loc)

    leaked = sorted(held & set(pal))
    fails += [] if not leaked else ["held-out palette in the cache"]
    # the floor is a MINIMUM per location, so check it per location, not in aggregate
    by_loc_pal = defaultdict(Counter)
    by_loc_ident = Counter()
    for r in cm:
        by_loc_pal[r["loc_id"]][r["palette"]] += 1
        if r["crop"]["scale"] == 1.0 and r["crop"]["shift_frac"] == 0.0:
            by_loc_ident[r["loc_id"]] += 1
    short_floor = {p: sum(1 for c in by_loc_pal.values() if c[p] < need)
                   for p, need in floor.items()}
    short_ident = sum(1 for i in per_loc
                      if by_loc_ident[i] < recipe["fan_out"]["floor_identity_geometries"])
    floor_ok = not any(short_floor.values()) and not short_ident
    fails += [] if floor_ok else ["per-location floor"]

    outside = sorted(set(pal) - pool - set(floor))
    q_lo, q_hi = recipe["jpg_quality"]["lo"], recipe["jpg_quality"]["hi"]
    q_out = sorted(x for x in q if not (q_lo <= x <= q_hi))
    aa_ok = set(aa) == {lvl.split(":")[0] for lvl in recipe["aa"]["levels"]}
    fails += [] if (not outside and not q_out and aa_ok) else ["realized axes vs recipe"]

    # Expected free-slot palette share: the 32 - floor free slots drawn uniformly over the
    # pool. Reported as a ratio so "uniform" is a number, not an adjective.
    free = tiles_per_loc - sum(floor.values())
    exp_free = free * n_loc / len(pool)
    free_counts = {p: pal[p] - (floor.get(p, 0) * n_loc) for p in pool}
    dev = max(abs(v / exp_free - 1.0) for v in free_counts.values())
    # 3.5 sigma of a binomial on the free slots — a uniform draw that misses this is not one.
    sigma = math.sqrt(exp_free * (1 - 1 / len(pool))) / exp_free
    unif_ok = dev <= 3.5 * sigma
    fails += [] if unif_ok else ["palette draw uniformity"]

    report["realized_axes"] = {
        "palettes_used": len(pal),
        "held_out_leaked": leaked,
        "outside_pool_or_floor": outside,
        "free_slot_share_max_deviation": round(dev, 4),
        "free_slot_share_3p5_sigma": round(3.5 * sigma, 4),
        "aa": dict(aa),
        "aa_share": {k: round(v / len(cm), 4) for k, v in sorted(aa.items())},
        "quality_min": min(q), "quality_max": max(q),
        "quality_distinct": len(q),
        "quality_mean": round(sum(k * v for k, v in q.items()) / len(cm), 2),
        "quality_outside_range": q_out,
        "floor_locations_short": short_floor,
        "identity_locations_short": short_ident,
    }
    print(f"\n  [4] realized axes vs recipe:")
    print(f"        palettes     {len(pal)} distinct; held-out leaked {leaked or 'none'}; "
          f"outside pool {outside or 'none'}")
    print(f"        uniformity   max free-slot deviation {dev:.4f} "
          f"(3.5 sigma = {3.5*sigma:.4f})  {'OK' if unif_ok else 'FAIL'}")
    print(f"        AA           {dict(aa)}  -> "
          f"{ {k: round(v/len(cm),4) for k,v in sorted(aa.items())} }")
    print(f"        quality      {min(q)}..{max(q)} over {len(q)} values, mean "
          f"{sum(k*v for k,v in q.items())/len(cm):.2f}  (recipe {q_lo}..{q_hi})")
    print(f"        floor        short: {short_floor}, identity short: {short_ident}  "
          f"{'OK' if floor_ok else 'FAIL'}")

    # ---- 5. size ----
    gib = total_bytes / 1024 ** 3
    kib = total_bytes / 1024 / max(len(on_disk), 1)
    report["size"] = {"tiles": len(on_disk), "bytes": total_bytes,
                      "gib": round(gib, 2), "mean_kib_per_tile": round(kib, 1)}
    print(f"\n  [5] size on disk             {gib:.2f} GiB over {len(on_disk)} tiles "
          f"({kib:.1f} KiB/tile)")

    # ---- 1. replay ----
    if a.quick:
        print("\n  [1] replay                   SKIPPED (--quick)")
    else:
        rep = check_replay(cm, recipe, a.replay_n)
        report["replay"] = rep
        fails += [] if rep["ok"] else ["replay byte-identity"]
        print(f"\n  [1] replay byte-identity     {'OK' if rep['ok'] else 'FAIL'}  "
              f"({rep.get('byte_identical', 0)}/{rep['n']} byte-identical, "
              f"{rep.get('n_differ', 0)} differ, {rep.get('n_missing', 0)} missing)")
        if rep.get("families"):
            print(f"        sampled: {rep['families']}")
        if not rep["ok"]:
            print(f"        {rep.get('error') or rep.get('differ') or rep.get('missing')}")

    report["verdict"] = "PASS" if not fails else "FAIL"
    report["failed_checks"] = fails
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 84)
    print(f"VERDICT: {report['verdict']}" + (f"  — {fails}" if fails else ""))
    print(f"wrote {a.json_out}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
