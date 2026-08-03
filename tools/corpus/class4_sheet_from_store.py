#!/usr/bin/env python
r"""class4_sheet_from_store.py — contact sheet of every class-4 crop, DERIVED from the store.

Sibling of `class4_combined_sheet.py`, which carries a hand-written `TILES` list frozen at
the 2026-07-28 state (9 crops). That list is a record and stays one; this builds the same
kind of sheet by asking `label_store.resolve_score` which rows are 4 right now, so a sheet
is never stale by one merge (`CLAUDE.md`, "Derive state in code; freeze it in records").

Each row of the sheet is one location shown TWICE: the canonical crop (the tile that was
labeled) beside its blue_orange vivid companion, so the palette-independence of the verdict
is visible at a glance. Both are read at deploy fidelity from the batch's own crop trees —
nothing is re-rendered, so the sheet cannot disagree with what was judged.

  uv run python tools/corpus/class4_sheet_from_store.py                    # whole corpus
  uv run python tools/corpus/class4_sheet_from_store.py --batches A B C    # a subset
  uv run python tools/corpus/class4_sheet_from_store.py --out <path.png>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT / "tools"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc     # noqa: E402
import label_store as ls       # noqa: E402
import paths                   # noqa: E402

TW, TH, PAD, LAB, TITLE_H = 384, 216, 6, 34, 40
BG, STRIP, INK, INK2 = (18, 18, 20), (30, 30, 34), (232, 232, 150), (150, 190, 232)


def class4_rows(batches):
    """[(batch, image_id, render)] for every row resolving to 4, in batch/id order."""
    out = []
    for b in batches:
        p = os.path.join(cc.batch_dir(b), "images.jsonl")
        if not os.path.exists(p):
            continue
        side, amend = ls.sidecar_for(b), ls.amendments_for(b)
        for r in cc.read_jsonl(p):
            if ls.resolve_score(r, side, amend) == 4:
                out.append((b, r["image_id"], r["render"]))
    return out


def build(rows, out_path: Path, title: str) -> Path:
    """Two columns per row (canonical | vivid); a row is dropped only if BOTH are missing."""
    n = len(rows)
    cols, cw, ch = 2, TW + PAD, TH + LAB + PAD
    w = cols * cw + PAD
    h = TITLE_H + n * ch + PAD
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, TITLE_H], fill=STRIP)
    d.text((PAD + 2, 12), title, fill=INK)

    drawn = 0
    for i, (b, iid, render) in enumerate(rows):
        y = TITLE_H + i * ch
        for j, srcdir in enumerate((cc.crops_dir(b), cc.vivid_dir(b))):
            x = PAD + j * cw
            fp = Path(srcdir) / f"{iid}.jpg"
            if fp.exists():
                with Image.open(fp) as t:
                    im.paste(t.convert("RGB").resize((TW, TH), Image.LANCZOS), (x, y))
                drawn += 1
            else:
                d.rectangle([x, y, x + TW, y + TH], outline=(90, 60, 60))
                d.text((x + 8, y + TH // 2), "MISSING", fill=(200, 120, 120))
        fam = render.get("fractal_type", "?")
        d.text((PAD + 2, y + TH + 6),
               f"{b.replace('2026-08-03_q4_', '')} / {iid}  [{fam}]  {render.get('palette', '?')}",
               fill=INK2)
        d.text((PAD + cw + 2, y + TH + 6),
               f"fw={render.get('fw', '?')[:18]}  cx={str(render.get('cx'))[:16]}", fill=INK2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=92)
    print(f"{n} class-4 rows, {drawn}/{2*n} tiles found -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="*", default=None,
                    help="batch_ids (default: every batch in the corpus)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    batches = a.batches or sorted(
        d for d in os.listdir(cc.BATCHES_DIR)
        if os.path.exists(os.path.join(cc.BATCHES_DIR, d, "images.jsonl")))
    rows = class4_rows(batches)
    out = Path(a.out) if a.out else paths.scratch("class4_sheet", "class4_from_store.jpg")
    # ASCII only: the sheet is drawn with PIL's default bitmap font, which has no glyph for
    # an em dash and renders it as a tofu box.
    build(rows, out, f"class-4 ({len(rows)} rows) - canonical | vivid companion "
                     f"- {len(batches)} batches, resolved via label_store")


if __name__ == "__main__":
    main()
