#!/usr/bin/env python
r"""class4_combined_sheet.py — ONE contact sheet of every class-4 crop in the corpus.

The class-4 aesthetic criteria in docs/design/label_rubric.md were to be written "from the
full set rather than from two tiles". This gathers every crop that resolves to score 4 across
ALL batches (via label_store.resolve_score, amendments included) into a single sheet, so the
bar can be written from the whole class at once instead of one batch's view.

The full class-4 set (2026-07-28) is 9 crops = the 7 anchor-pass promotions + the 2 fresh
minibrot rows. The 7 promotions live, by coordinate re-key, in their SOURCE batches
(gather_v6 x4, guided_descend x1, mining x1, phoenix_grid x1); the 2026-07-26 anchor batch
re-rendered those same 7 locations under its own blue_orange vivid map as
`anchor_*` rows scored 4 in its scores.json — those anchor re-renders are the tiles used here
(palette-consistent across the 7). The 2 minibrot tiles carry the minibrot batch's own vivid
map. The interior-band batch contributed nothing (no crop scored >= 3).

Regenerable view -> scratch/, never docs/ (generated-output convention; test_docs_tree.py
forbids untracked content under docs/).

  uv run python tools/corpus/class4_combined_sheet.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402
sys.path.insert(0, str(HERE))
import corpus_common as cc  # noqa: E402

BATCHES = ROOT / "data" / "label_corpus" / "batches"
OUT_DIR = paths.scratch("class4_combined_sheet")
TW, TH, PAD, LAB, TITLE_H = 384, 216, 6, 40, 34
BG, STRIP, INK, INK2 = (18, 18, 20), (30, 30, 34), (232, 232, 150), (150, 190, 232)

# Tiles: the anchor batch's 7 score-4 re-renders (palette-consistent) + the 2 minibrot 4s.
# (source_batch, source_iid) is the ORIGINAL provenance the promotion re-keys onto; tile_*
# is where the rendered JPG actually lives.
TILES = [
    # anchor re-renders of the 7 promotions
    ("2026-07-26_anchor_class4_v1", "anchor_mandelbrot_03", "mandelbrot",       "gather_v6 promo"),
    ("2026-07-26_anchor_class4_v1", "anchor_mandelbrot_05", "mandelbrot",       "src promo"),
    ("2026-07-26_anchor_class4_v1", "anchor_mandelbrot_06", "mandelbrot",       "src promo"),
    ("2026-07-26_anchor_class4_v1", "anchor_julia_13",      "julia:mandelbrot", "gather_v6 promo"),
    ("2026-07-26_anchor_class4_v1", "anchor_julia_21",      "julia:mandelbrot", "gather_v6 promo"),
    ("2026-07-26_anchor_class4_v1", "anchor_phoenix_26",    "phoenix",          "phoenix_grid promo"),
    ("2026-07-26_anchor_class4_v1", "anchor_multibrot5_51", "multibrot5",       "gather_v6 promo"),
    # the 2 fresh minibrot rows (one atom, two disjoint windows — see bakeoff §4)
    ("2026-07-26_minibrot_roster_v2", "mb0095_d5_p11_029_accepted", "multibrot5", "minibrot fresh-4"),
    ("2026-07-26_minibrot_roster_v2", "mb0159_d5_p11_029_accepted", "multibrot5", "minibrot fresh-4"),
]


def tile_img(batch, iid):
    for sub in ("vivid", "crops"):
        p = Path(cc.crops_dir(batch) if sub == "crops" else cc.vivid_dir(batch)) / f"{iid}.jpg"
        if p.exists():
            return Image.open(p).convert("RGB").resize((TW, TH)), sub
    return Image.new("RGB", (TW, TH), (60, 20, 20)), "MISSING"


def main():
    cols = 3
    n = len(TILES)
    rows_n = (n + cols - 1) // cols
    W = cols * (TW + PAD) + PAD
    H = TITLE_H + rows_n * (TH + LAB + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 2, 9),
            f"Class-4 combined — every crop that resolves to score 4 in the corpus  "
            f"(n={n}: 7 anchor promotions + 2 minibrot)", fill=INK)
    for k, (batch, iid, fam, note) in enumerate(TILES):
        im, src = tile_img(batch, iid)
        r, c = divmod(k, cols)
        x = PAD + c * (TW + PAD)
        y = TITLE_H + PAD + r * (TH + LAB + PAD)
        sheet.paste(im, (x, y))
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=STRIP)
        dr.text((x + 4, y + TH + 4), f"{fam}   ({note})", fill=INK)
        dr.text((x + 4, y + TH + 21), f"{iid}  [{src}]", fill=INK2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "label_rubric_class4_combined.png"
    sheet.save(out)
    print(f"wrote {out}  ({n} tiles, {W}x{H})")


if __name__ == "__main__":
    main()
