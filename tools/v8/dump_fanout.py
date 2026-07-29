#!/usr/bin/env python
"""Dump the full 24-crop augmentation fan-out for 12 v8 training locations, as contact sheets.

The point is to see EXACTLY what the network sees, in the palette it sees it in. So:

  * the tiles are the LITERAL cached JPGs the trainer loads — read straight off the paths in
    `data/v8/cache_manifest.jsonl`, at their native 512x288, never re-rendered and never
    resized. If a tile is missing this ABORTS naming it, rather than quietly rendering a
    stand-in that would differ in JPEG quantisation alone;
  * no vivid `blue_orange` companion. That colormap exists for the LABELER, whose job is to
    judge structure through a legible map; the network never sees it. Substituting it here
    would answer a different question;
  * the only text is the parameter captions. A tile's caption is its own augmentation
    coordinates; a sheet's header is the location's identity. Nothing else.

LAYOUT. 4 rows x 6 columns. Row = (palette, AA level); column = (scale, shift). So the
palette axis is read by comparing row 0 vs row 2 (ss1) or row 1 vs row 3 (ss2) — the same
geometry, a different colormap, side by side — and the AA axis by comparing adjacent rows
within a palette. Every tile still carries its own caption, so the grouping is a
convenience, not the source of truth.

SELECTION. 3 locations per quality class {1,2,3,4}, TRAIN split only, drawn under a fixed
seed and spread across families: distinct families are taken first (in a seeded order), and
only once every family carrying that class is used does a second location come from a family
already drawn. All four classes have >=3 families here, so all 12 are family-distinct.

  uv run python tools/v8/dump_fanout.py --select-only   # print the draw, touch nothing
  uv run python tools/v8/dump_fanout.py --emit-plan P   # write just these 288 plan rows
  uv run python tools/v8/dump_fanout.py                 # build the sheets

Writes: scratch/v8_fanout/<class>_<family>_loc<id>.png  (disposable — the sheets are a pure
function of the cache + manifest, both durable).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402

MANIFEST = ROOT / "data" / "v8" / "manifest.jsonl"
CACHE_MANIFEST = ROOT / "data" / "v8" / "cache_manifest.jsonl"
PLAN = ROOT / "data" / "v8" / "plan.jsonl"
OUT_DIR = paths.scratch("v8_fanout")

SEED = 20260729
PER_CLASS = 3
CLASSES = (1, 2, 3, 4)

# Grid: 4 rows x 6 cols = 24. Row key = (palette_index, ss); col key = (scale, shift).
COLS = 6
ROWS = 4
PAD = 10
CAP_H = 20            # per-tile caption strip
HEAD_H = 74           # sheet header
BG = (18, 18, 20)
FG = (232, 232, 236)
DIM = (150, 150, 158)


def _font(size: int):
    """A readable TrueType font if the machine has one, else PIL's bitmap default. The
    captions are the whole deliverable, so this tries a few known-good paths rather than
    silently falling back to a 6px bitmap."""
    for cand in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def select(manifest) -> list:
    """3 train locations per class, family-spread, under a fixed seed.

    Distinct families first: shuffle the families that carry this class, take one location
    from each in turn (each location drawn with the same seeded RNG), and only wrap back to
    an already-used family if the class does not span PER_CLASS families."""
    rng = random.Random(SEED)
    train = [r for r in manifest if r["split"] == "train"]
    picked = []
    for cls in CLASSES:
        by_fam = defaultdict(list)
        for r in train:
            if r["label"] == cls:
                by_fam[r["fractal_type"]].append(r)
        fams = sorted(by_fam)
        rng.shuffle(fams)
        chosen, i = [], 0
        while len(chosen) < PER_CLASS and fams:
            fam = fams[i % len(fams)]
            pool = [r for r in by_fam[fam] if r not in chosen]
            if pool:
                chosen.append(rng.choice(pool))
            i += 1
            if i > len(fams) * PER_CLASS:      # class cannot supply PER_CLASS at all
                break
        picked.extend(chosen)
    return picked


def identity_line(r) -> str:
    """The location's identity, verbatim from the manifest — decimal strings, not floats."""
    parts = [f"cx={r['cx']}", f"cy={r['cy']}", f"fw={r['fw']}"]
    if r.get("c_re") is not None:
        parts.append(f"c=({r['c_re']}, {r['c_im']})")
    fp = [f"{k}={r[k]}" for k in loc_mod.family_param_keys(r["fractal_type"]) if r.get(k)]
    if fp:
        parts.append("  ".join(fp))
    return "   ".join(parts)


def tile_caption(cm) -> str:
    aa = f"ss{cm['ss']} {cm['filter']}"
    return f"{cm['palette']}  ·  s{cm['scale']:.1f}  ·  {cm['shift_id']}  ·  {aa}"


def build_sheet(loc_row, cm_rows, palettes, out_path: Path, font, font_hd, font_sm):
    """One location's 24 cached tiles on a 4x6 grid. Tiles are pasted at native size."""
    scale_order = sorted({c["scale"] for c in cm_rows})
    shift_order = ["center", "shifted"]
    ss_order = sorted({c["ss"] for c in cm_rows})
    cell = {}
    for c in cm_rows:
        row = palettes.index(c["palette"]) * len(ss_order) + ss_order.index(c["ss"])
        col = scale_order.index(c["scale"]) * len(shift_order) + shift_order.index(c["shift_id"])
        cell[(row, col)] = c
    assert len(cell) == ROWS * COLS, f"{out_path.name}: {len(cell)} distinct grid cells, want 24"

    first = Image.open(paths.bulk(cm_rows[0]["path"]))
    tw, th = first.size
    first.close()
    W = PAD + COLS * (tw + PAD)
    H = HEAD_H + ROWS * (th + CAP_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)

    d.text((PAD, 8), f"{loc_row['fractal_type']}   class {loc_row['label']}", font=font_hd, fill=FG)
    d.text((PAD, 34), identity_line(loc_row), font=font_sm, fill=DIM)
    d.text((PAD, 52),
           f"loc_id={loc_row['loc_id']}  split={loc_row['split']}  "
           f"group_id={loc_row['group_id']}  source={loc_row['source']}",
           font=font_sm, fill=DIM)

    for (row, col), c in sorted(cell.items()):
        x = PAD + col * (tw + PAD)
        y = HEAD_H + row * (th + CAP_H + PAD)
        with Image.open(paths.bulk(c["path"])) as im:
            im.load()
            sheet.paste(im.convert("RGB"), (x, y))     # native size; no resample
        d.text((x, y + th + 4), tile_caption(c), font=font, fill=FG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return sheet.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-only", action="store_true", help="print the draw and exit")
    ap.add_argument("--emit-plan", default=None,
                    help="write the plan rows for the selected locations to this path")
    a = ap.parse_args()

    manifest = read_jsonl(MANIFEST)
    picked = select(manifest)
    ids = [r["loc_id"] for r in picked]

    print(f"selection: seed={SEED}, {PER_CLASS} per class over {CLASSES}, train split only")
    print(f"{'class':>5}  {'family':<18} {'loc_id':>7}  {'group':>8}  identity")
    for r in picked:
        print(f"{r['label']:>5}  {r['fractal_type']:<18} {r['loc_id']:>7}  "
              f"{r['group_id']:>8}  {identity_line(r)[:96]}")
    fams_per_class = defaultdict(list)
    for r in picked:
        fams_per_class[r["label"]].append(r["fractal_type"])
    print("\nfamily spread per class: "
          + "; ".join(f"{c}: {fams_per_class[c]}" for c in CLASSES))

    if a.emit_plan:
        # The selected locations' own plan rows, VERBATIM — same `out` paths, same
        # parameters. Rendering this subset produces the canonical cache entries early; the
        # main run then skips them (`v4-render-batch` skips any existing output). It is the
        # cache being filled out of order, not a separate render.
        want_dirs = {f"/aug_cache/{i}/" for i in ids}
        rows = [p for p in read_jsonl(PLAN) if any(w in p["out"] for w in want_dirs)]
        assert len(rows) == len(ids) * 24, f"{len(rows)} plan rows for {len(ids)} locations"
        Path(a.emit_plan).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                     encoding="utf-8")
        print(f"\nwrote {len(rows)} plan rows -> {a.emit_plan}")
        return 0

    if a.select_only:
        return 0

    cm = read_jsonl(CACHE_MANIFEST)
    by_loc = defaultdict(list)
    for c in cm:
        if c["location_id"] in set(ids):
            by_loc[c["location_id"]].append(c)

    # Refuse rather than substitute: every one of the 24 tiles must already be in the cache.
    missing = [c["path"] for i in ids for c in by_loc[i] if not paths.bulk(c["path"]).exists()]
    if missing:
        print(f"\nABORT: {len(missing)} cached tile(s) not on disk — the cache build has not "
              f"reached these locations yet. These sheets must be the LITERAL cached crops, "
              f"so nothing is rendered or substituted here.\n  e.g. {missing[:3]}",
              file=sys.stderr)
        return 1

    font, font_hd, font_sm = _font(13), _font(20), _font(12)
    palettes = [p["name"] for p in json.loads(
        (ROOT / "data" / "v8" / "aug_roster.json").read_text(encoding="utf-8"))["v8_palettes"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for r in picked:
        rows_i = by_loc[r["loc_id"]]
        assert len(rows_i) == 24, f"loc {r['loc_id']}: {len(rows_i)} cache rows"
        out = OUT_DIR / f"class{r['label']}_{r['fractal_type']}_loc{r['loc_id']}.png"
        size = build_sheet(r, rows_i, palettes, out, font, font_hd, font_sm)
        print(f"  {out.name}  {size[0]}x{size[1]}")
    print(f"\n{len(picked)} sheets -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
