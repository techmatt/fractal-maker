#!/usr/bin/env python
r"""v8 render plan + cache manifest — a NEW 24-slot augmentation recipe.

RECIPE CHANGE: 42 slots -> 24 slots per location.

    2 palettes  x  3 scales {0.7, 1.0, 1.3}  x  2 shifts {center, shifted}
                x  2 AA levels {ss1 box, ss2 lanczos3}                       = 24

Three things move relative to v4..v7's frozen 42-slot multiset (6 palettes x 3 scales x
2 shifts x {ss1 box, ss4 lanczos3}, the last axis unbalanced at 36+6):

  * **ss2 is IN, on every location, uniformly.** This is the deliberate one. Deploy scores
    at 640x360 **ss2 + Lanczos-3** (`tools/atlas/guard.py` GUARD_STAT_RES, via `render-one`'s
    lanczos3 default), an AA signature the v4..v7 fan-out never contained — it had ss1 and
    ss4 and nothing between. v7 declined to add it (build_metadata.amendment_1_ss2_gap):
    under a frozen prefix the slot could only be added to the 536 appended locations, so
    "has an ss2 tile" would have correlated with both family and label and handed the model
    a shortcut. A from-scratch build removes that objection entirely — every location gets
    the same 24 slots — so the accepted covariate shift is closed rather than re-accepted.
  * **ss4 is OUT.** Nothing deploys at ss4; it was the antialiased anchor only because it
    was the corpus crop's fidelity. ss2 now plays that role, at 1/4 the render cost.
  * **6 palettes -> 2.** Palette was the widest axis and the most expensive; two is enough
    to keep the head from binding to one colormap while the slot budget goes to the AA axis
    that actually matches deploy.

Per-location render cost, in ss1-equivalents (iteration work scales with ss^2):
    v4..v7:  36*1 + 6*16 = 132        v8:  12*1 + 12*4 = 60      (~45%)

THE ROSTER IS RECOVERED, NOT INHERITED. `data/v4/aug_roster.json` is gone with the rest of
the v4..v7 build artifacts, so the six palette names are recovered from the one surviving
witness — the filenames in the relocated v4 aug_cache tree — and re-committed as
`data/v8/aug_roster.json` (durable this time). Their ORDER is not recoverable from a
directory listing, so the roster is canonicalised to sorted order and the second palette is
the cyclic successor of `twilight_shifted` in that order. See `recover_roster`.

REUSE. `audit_reuse` reports how many of the 24 slots per location already exist in a v4..v7
cache under a matching key, before anything is rendered. Spoiler, and the reason it is a
function and not a comment: it needs the per-version `cache_manifest.jsonl` to map a cache
`<loc_id>/` directory back to a location identity, and those are gone too. Run it; it prints
what it found.

  uv run python tools/v8/build_plan.py [--dry-run]

Writes (all `paths.durable()`):
  data/v8/aug_roster.json      the recovered 6-palette roster + the v8 2-palette selection
  data/v8/plan.jsonl           one row per render, for `v4-render-batch`
  data/v8/cache_manifest.jsonl one row per cached tile, for the trainer's loader
The JPGs themselves are `paths.bulk()` -> ARTIFACTS_ROOT/data/v8/aug_cache/, out of tree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402

MANIFEST = "data/v8/manifest.jsonl"
ROSTER_OUT = "data/v8/aug_roster.json"
PLAN_OUT = "data/v8/plan.jsonl"
CACHE_MANIFEST_OUT = "data/v8/cache_manifest.jsonl"
V8_CACHE_DIR = "data/v8/aug_cache"          # repo-relative; bulk() resolves it out-of-tree

# --- the v8 recipe ---
SCALES = [0.7, 1.0, 1.3]
SHIFTS = ("center", "shifted")
SHIFT_FRAC = 0.4                             # unchanged from v4..v7
# (ss, downsample filter). ss1's "box" is a no-op average at ss=1 and matches v4..v7;
# ss2's lanczos3 is what the deploy path actually uses (render-one's default), so the
# cached ss2 tile carries deploy's AA signature rather than a cheaper approximation.
AA_LEVELS = ((1, "box"), (2, "lanczos3"))
SLOTS = len(SCALES) * len(SHIFTS) * len(AA_LEVELS) * 2   # x2 palettes = 24
DEPLOY_PALETTE = "twilight_shifted"

# Prior cache trees, in the order a reuse search would consult them.
PRIOR_CACHES = [("v4", "data/v4/aug_cache", "data/v4/cache_manifest.jsonl"),
                ("v5", "data/v5/aug_cache_julia", "data/v5/cache_manifest.jsonl"),
                ("v6", "data/v6/aug_cache_gather", "data/v6/cache_manifest.jsonl"),
                ("v7", "data/v7/aug_cache", "data/v7/cache_manifest.jsonl")]


def scale_tok(s: float) -> str:
    return f"{s:.1f}"


def fmt_f64(x: float) -> str:
    return repr(float(x))


def slot_filename(pal: str, sc: float, shift_id: str, ss: int) -> str:
    """Cache tile filename. Byte-identical scheme to v4..v7 so a tile is self-describing
    and a future reuse pass has something to key on."""
    return f"{pal}__s{scale_tok(sc)}__sh{shift_id}__ss{ss}.jpg"


# --------------------------------------------------------------------------- #
# Roster recovery
# --------------------------------------------------------------------------- #
def _palette_family(name: str) -> str:
    """Grouping token for a palette name. The authoritative `palette_family` values lived
    in `data/v4/aug_roster.json`, which is gone, so this DERIVES one from the name's own
    namespace prefix (`cet_*`, `cmr.*`, else the bare name). Nothing in training reads it —
    `classifier/data_v4.py` stores `Render.palette_family` and never consults it, and the
    sampler weights are (class x group x source) only — so the derivation is a faithful
    label, not a load-bearing reconstruction. Recorded as derived in the roster file."""
    if name.startswith("cet_"):
        return "cet"
    if "." in name:
        return name.split(".", 1)[0]
    return name


def recover_roster() -> dict:
    """Recover the 6-palette roster from the surviving v4 aug_cache tree and pick the v8 pair.

    The roster FILE is gone; the only witness is the tile filenames, which encode the palette
    name as everything before the first `__`. A directory listing gives the SET but not the
    original order, so the roster is canonicalised to sorted order — the one order that is
    reproducible from the witness — and that canonical order is what "next entry" means."""
    v4 = paths.bulk(PRIOR_CACHES[0][1])
    sample_dir = None
    if v4.exists():
        for child in sorted(v4.iterdir()):
            if child.is_dir():
                sample_dir = child
                break
    if sample_dir is None:
        raise SystemExit(f"cannot recover the palette roster: no v4 cache tree at {v4}")
    names = sorted({f.name.split("__", 1)[0] for f in sample_dir.iterdir()
                    if f.suffix == ".jpg"})
    if DEPLOY_PALETTE not in names:
        raise SystemExit(f"deploy palette {DEPLOY_PALETTE!r} not in recovered roster {names}")
    i = names.index(DEPLOY_PALETTE)
    second = names[(i + 1) % len(names)]      # cyclic successor in canonical (sorted) order
    return {
        "recovered_from": f"{PRIOR_CACHES[0][1]}/{sample_dir.name}/*.jpg (tile filenames)",
        "recovery_note": (
            "data/v4/aug_roster.json is gone (never committed; see "
            "tools/audit/durability_map.py). The six names are recovered from the surviving "
            "relocated v4 cache tree. The original roster ORDER is not recoverable from a "
            "directory listing, so the canonical order below is sorted(names) — the only "
            "order reproducible from the witness."),
        "palette_family_note": (
            "palette_family is DERIVED from the name's namespace prefix, not recovered. "
            "The authoritative values died with aug_roster.json. Nothing in training reads "
            "the field (classifier/data_v4.py stores it and never uses it)."),
        "canonical_order": names,
        "v8_selection_rule": (
            f"[{DEPLOY_PALETTE} (the deploy palette), then its cyclic successor in "
            f"canonical_order]"),
        "v8_palettes": [
            {"name": DEPLOY_PALETTE, "palette_family": _palette_family(DEPLOY_PALETTE),
             "role": "deploy palette (data_v4.NEUTRAL_PALETTE; the canonical eval view)"},
            {"name": second, "palette_family": _palette_family(second),
             "role": f"deterministic second: successor of {DEPLOY_PALETTE} "
                     f"(index {i} -> {(i+1) % len(names)}) in canonical_order"},
        ],
    }


# --------------------------------------------------------------------------- #
# Reuse audit
# --------------------------------------------------------------------------- #
def audit_reuse(n_locs: int, palettes: list) -> dict:
    """How many of the 24 slots per location already exist under a matching key?

    A cached tile lives at `<cache>/<loc_id>/<palette>__s<scale>__sh<shift>__ss<N>.jpg`. The
    filename carries palette/scale/shift/ss — but NOT the location. `<loc_id>` is a dense
    index into that version's `manifest.jsonl`, and the ONLY thing that maps it back to
    (family, cx, cy, fw, c) is that version's `cache_manifest.jsonl`. So reuse is possible
    iff at least one prior cache_manifest survives. This checks, rather than assuming."""
    tiles = 0
    have_manifest = []
    for ver, cache_rel, cm_rel in PRIOR_CACHES:
        cache = paths.bulk(cache_rel)
        n_dirs = sum(1 for c in cache.iterdir() if c.is_dir()) if cache.exists() else 0
        # cache_manifest was never relocated, so it resolves in-tree; check bulk() too in
        # case someone parked a copy alongside the tiles.
        cm_in_tree = ROOT / cm_rel
        cm_bulk = paths.bulk(cm_rel)
        found = next((p for p in (cm_in_tree, cm_bulk) if p.exists()), None)
        if found is not None:
            have_manifest.append((ver, str(found)))
        tiles += n_dirs
    # Even WITH a mapping, the reachable ceiling is the ss1 slots of the two v8 palettes
    # whose shift geometry is unchanged. The shifted frames are not: the angle schedule is
    # 2*pi*(pal_index*len(SCALES) + scale_index)/n_combo, and n_combo went 6*3=18 -> 2*3=6,
    # so every `shshifted` tile sits at a different offset than the v8 recipe asks for.
    # ss2 did not exist in any prior cache at all. Ceiling = 2 palettes x 3 scales x
    # {center} x {ss1} = 6 of 24 slots, and only for locations present in a prior manifest.
    ceiling_slots = len(palettes) * len(SCALES) * 1 * 1
    reusable = 0 if not have_manifest else None   # None == "would need a real join"
    return {
        "prior_cache_location_dirs": tiles,
        "prior_cache_manifests_found": have_manifest,
        "reusable_slots": 0 if reusable == 0 else reusable,
        "reuse_fraction": 0.0 if reusable == 0 else None,
        "why": (
            "A cached tile's filename encodes palette/scale/shift/ss but NOT its location; "
            "`<loc_id>` is an index into that version's manifest.jsonl, and only that "
            "version's cache_manifest.jsonl maps it back to (family, cx, cy, fw, c). None "
            "of data/v{4,5,6,7}/cache_manifest.jsonl survive (nor manifest.jsonl, nor "
            "aug_roster.json) — see tools/audit/durability_map.py — so no tile in the "
            "243,477-file prior cache can be attributed to a v8 location. Reuse is 0/24 "
            "per location, for a mapping reason, not a recipe reason."
            if reusable == 0 else "a prior cache_manifest survives; join on identity"),
        "ceiling_had_the_mapping_survived": {
            "slots_per_location": ceiling_slots,
            "of": SLOTS,
            "reason": (
                "ss2 (12 of 24 slots) exists in no prior cache. Of the 12 ss1 slots, the 6 "
                "`shshifted` ones sit at a different offset: the shift angle is "
                "2*pi*(pal_idx*3 + scale_idx)/n_combo and n_combo went 18 -> 6. Only the 6 "
                "(2 palettes x 3 scales x center x ss1) tiles would have matched — and only "
                "for the 5,797 locations a prior manifest covered, out of 7,141."),
        },
        "not_deleting": ("The prior caches are NOT deleted by this build (prompt §4): they "
                         "are the reuse source if a manifest is ever recovered, and the "
                         "deletion decision belongs after v8 is trained and evaluated."),
    }


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #
def emit_location(loc_id, r, palettes, fam_of, angle_of, plan_rows, cm_rows):
    """The 24 cache rows (and 24 plan rows) for one location.

    Family extra-constants (`p_re/p_im/zm1_re/zm1_im` for phoenix) are copied onto every
    plan row. v4..v7 dropped them, which would have made `v4-render-batch` fall back to
    PHOENIX_{C,P,ZM1}_DEFAULT and render all 573 phoenix locations at the one default Ushiki
    spot regardless of their actual dynamics — silently, since the tile still looks like a
    fractal. v8 is the first manifest with phoenix in it, so this is the first build where
    it matters."""
    cx0, cy0 = float(r["cx"]), float(r["cy"])
    fw0 = float(r["fw"])
    ft = r.get("fractal_type", "mandelbrot")
    base = dict(label=r["label"], split=r["split"], group_id=r["group_id"],
                source=r["source"], biased=r["biased"])
    extra = {k: r[k] for k in loc_mod.family_param_keys(ft) if r.get(k) is not None}
    c_re, c_im = r.get("c_re"), r.get("c_im")

    for pal in palettes:
        for sc in SCALES:
            for shift_id in SHIFTS:
                for ss, filt in AA_LEVELS:
                    fw_slot = sc * fw0
                    if shift_id == "shifted":
                        ang = angle_of[(pal, sc)]
                        mag = SHIFT_FRAC * fw_slot
                        dx, dy = mag * math.cos(ang), mag * math.sin(ang)
                        cx, cy = cx0 + dx, cy0 + dy
                    else:
                        dx = dy = 0.0
                        cx, cy = cx0, cy0
                    rel = f"{V8_CACHE_DIR}/{loc_id}/{slot_filename(pal, sc, shift_id, ss)}"
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
                        "palette": pal, "palette_family": fam_of[pal],
                        "scale": sc, "shift_id": shift_id,
                        "shift_dx": dx, "shift_dy": dy,
                        # Two-value AA vocabulary, as classifier/data_v4.py expects — but
                        # "antialiased" now means ss2, the DEPLOY level, not ss4. `ss` is
                        # emitted explicitly so no consumer has to infer it from the label.
                        "aa_level": "aliased" if ss == 1 else "antialiased",
                        "ss": ss, "filter": filt,
                        "fractal_type": ft, "path": rel,
                    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the recipe, reuse audit and counts; write nothing")
    a = ap.parse_args()

    roster = recover_roster()
    palettes = [p["name"] for p in roster["v8_palettes"]]
    fam_of = {p["name"]: p["palette_family"] for p in roster["v8_palettes"]}
    assert len(palettes) == 2, palettes

    n_combo = len(palettes) * len(SCALES)
    angle_of = {}
    for pi, pal in enumerate(palettes):
        for si, sc in enumerate(SCALES):
            angle_of[(pal, sc)] = 2.0 * math.pi * (pi * len(SCALES) + si) / n_combo

    rows = [json.loads(l) for l in (ROOT / MANIFEST).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print("=" * 82)
    print(f"v8 PLAN — {len(rows)} locations x {SLOTS} slots")
    print("=" * 82)
    print(f"  palettes : {palettes}")
    print(f"             (roster recovered from {roster['recovered_from']})")
    print(f"  scales   : {SCALES}")
    print(f"  shifts   : {list(SHIFTS)}  (frac {SHIFT_FRAC}, angle schedule over {n_combo} combos)")
    print(f"  AA       : {[f'ss{s} {f}' for s, f in AA_LEVELS]}   (ss4 DROPPED; ss2 is the deploy level)")
    print(f"  cost     : {sum(1 for _ in range(len(SCALES)*len(SHIFTS)*len(palettes)))*1 + 0}"
          f" ... {len(palettes)*len(SCALES)*len(SHIFTS)}x ss1 + "
          f"{len(palettes)*len(SCALES)*len(SHIFTS)}x ss2 = "
          f"{len(palettes)*len(SCALES)*len(SHIFTS)*(1+4)} ss1-equivalents/location "
          f"(v4..v7 was 132)")

    reuse = audit_reuse(len(rows), palettes)
    print("\n--- REUSE AUDIT (before rendering anything) ---")
    print(f"  prior cache location dirs : {reuse['prior_cache_location_dirs']}")
    print(f"  prior cache_manifests     : {reuse['prior_cache_manifests_found'] or 'NONE FOUND'}")
    print(f"  reusable slots/location   : {reuse['reusable_slots']} of {SLOTS}"
          f"   (reuse fraction {reuse['reuse_fraction']:.1%})"
          if reuse["reuse_fraction"] is not None else
          f"  reusable slots/location   : needs a real identity join")
    print(f"  why: {reuse['why']}")
    ceil = reuse["ceiling_had_the_mapping_survived"]
    print(f"  ceiling had the mapping survived: {ceil['slots_per_location']}/{ceil['of']} slots")

    plan_rows, cm_rows = [], []
    for r in rows:
        emit_location(r["loc_id"], r, palettes, fam_of, angle_of, plan_rows, cm_rows)
    assert len(plan_rows) == len(rows) * SLOTS, (len(plan_rows), len(rows) * SLOTS)
    assert len(cm_rows) == len(plan_rows)

    n_ss1 = sum(1 for p in plan_rows if p["ss"] == 1)
    n_ss2 = sum(1 for p in plan_rows if p["ss"] == 2)
    print(f"\n  plan rows      : {len(plan_rows)}  (ss1 box {n_ss1} + ss2 lanczos3 {n_ss2})")
    fam = Counter(p["fractal_type"] for p in plan_rows)
    print(f"  per-family rows: {dict(sorted(fam.items()))}")
    n_c = sum(1 for p in plan_rows if "c_re" in p)
    n_p = sum(1 for p in plan_rows if "p_re" in p)
    print(f"  rows carrying c: {n_c}   rows carrying phoenix p/z-1: {n_p}")
    print(f"  cache root     : {paths.bulk(V8_CACHE_DIR)}  (bulk, out-of-tree)")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    paths.durable(ROSTER_OUT, mkparents=True).write_text(
        json.dumps({**roster, "recipe": {
            "slots_per_location": SLOTS, "scales": SCALES, "shifts": list(SHIFTS),
            "shift_frac": SHIFT_FRAC,
            "aa_levels": [{"ss": s, "filter": f,
                           "aa_level": "aliased" if s == 1 else "antialiased"}
                          for s, f in AA_LEVELS],
            "shift_angle_schedule": "2*pi*(palette_index*len(SCALES) + scale_index)/"
                                    f"{n_combo}",
            "dropped_from_v4_v7": "ss4; 4 of the 6 palettes",
            "added_vs_v4_v7": "ss2 + lanczos3 (the deploy AA level), on EVERY location",
            "cache_render": {"width": 512, "height": 288, "maxiter": 8000,
                             "jpg_quality": 85,
                             "note": "v4-render-batch defaults, unchanged from v4..v7"},
        }, "reuse_audit": reuse}, indent=2), encoding="utf-8")

    with paths.durable(PLAN_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row) + "\n")
    with paths.durable(CACHE_MANIFEST_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for row in cm_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWROTE {ROSTER_OUT}")
    print(f"WROTE {PLAN_OUT}            ({len(plan_rows)} rows)")
    print(f"WROTE {CACHE_MANIFEST_OUT}  ({len(cm_rows)} rows)")


if __name__ == "__main__":
    main()
