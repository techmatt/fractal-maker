#!/usr/bin/env python
r"""anchor_rubric_sheets.py — the two sheets the class-4 rubric is written from.

The 2026-07-26 anchor pass re-judged 52 previously-class-3 rows blind on the 1..4 scale and
freshly scored 8 minibrot rows. Two contact sheets distill that pass into the material the
class-4 aesthetic criteria were to be written from. Those criteria were never written: the
section stayed a stub and its doc (`docs/design/label_rubric.md`) has since been retired into
`data/label_corpus/CORPUS_SCHEMA.md` § label, which carries the 1..4 scale but not a
"what makes a 4 not just a 3" bar. **The sheets are the bar** — read them.

  * everything he called 4   (the promotions — what a 4 looks like)
  * everything he demoted    (3 -> 1|2 — what fell OUT of "good")

Tiles are the VIVID companion renders (blue_orange, `<batch>/vivid/<image_id>.jpg`), the same
map across every row so the eye compares structure, not palette. Each tile is captioned with
family + the score move (orig -> new).

The PNGs are a REGENERABLE VIEW, so they go to `scratch/anchor_rubric_sheets/` per the
generated-output convention — not into `docs/`. (They were briefly written into
`docs/design/` and gitignored there, which left `tests/test_repo_size_guard.py`
permanently red on 4 uncovered multi-MB files; `tests/test_docs_tree.py` now forbids
untracked content under `docs/` outright.) Read them beside `data/label_corpus/
CORPUS_SCHEMA.md` § label; rebuild with the one-liner below.

  uv run python tools/corpus/anchor_rubric_sheets.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import corpus_common as cc  # noqa: E402

BATCH_ID = "2026-07-26_anchor_class4_v1"
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

OUT_DIR = paths.scratch("anchor_rubric_sheets")   # regenerable view -> scratch/, never docs/
TW, TH, PAD, LAB = 384, 216, 6, 22          # tile 16:9, caption strip below
BG, STRIP, INK = (18, 18, 20), (30, 30, 34), (232, 232, 150)


def load_rows():
    bdir = Path(cc.batch_dir(BATCH_ID))
    rows = [json.loads(l) for l in (bdir / "images.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = json.loads((bdir / "scores.json").read_text(encoding="utf-8"))
    vivid = Path(cc.vivid_dir(BATCH_ID))
    out = []
    for r in rows:
        iid = r["image_id"]
        p = r.get("provenance") or {}
        out.append({
            "iid": iid,
            "family": p.get("family"),
            "orig": p.get("original_score"),          # None for the fresh minibrot rows
            "new": scores.get(iid),
            "vivid": vivid / f"{iid}.jpg",
        })
    return out


def build_sheet(items, title, out_png, cols):
    n = len(items)
    rows_n = (n + cols - 1) // cols
    W = cols * (TW + PAD) + PAD
    TITLE_H = 34
    H = TITLE_H + rows_n * (TH + LAB + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 2, 9), title, fill=INK)
    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        x = PAD + c * (TW + PAD)
        y = TITLE_H + PAD + r * (TH + LAB + PAD)
        vp = it["vivid"]
        im = (Image.open(vp).convert("RGB").resize((TW, TH)) if vp.exists()
              else Image.new("RGB", (TW, TH), (60, 20, 20)))
        sheet.paste(im, (x, y))
        move = f"{it['orig']}->{it['new']}" if it["orig"] is not None else f"new={it['new']}"
        cap = f"{it['family']}  {move}  [{it['iid']}]"
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=STRIP)
        dr.text((x + 4, y + TH + 5), cap, fill=INK)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"  wrote {out_png}  ({n} tiles, {W}x{H})")


def main():
    rows = load_rows()
    called4 = [r for r in rows if r["new"] == 4]
    # "demoted" = original label moved DOWN (only the revision rows have an original)
    demoted = [r for r in rows if r["orig"] is not None and r["new"] < r["orig"]]
    # stable, readable ordering: family then image_id
    called4.sort(key=lambda r: (r["family"] or "", r["iid"]))
    demoted.sort(key=lambda r: (r["family"] or "", r["iid"]))
    print(f"class-4 rows: {len(called4)}   demoted rows: {len(demoted)}")
    build_sheet(called4, f"Anchor class-4 rollout — everything called 4  (n={len(called4)}, vivid companion)",
                OUT_DIR / "label_rubric_class4_examples.png", cols=3 if len(called4) <= 9 else 4)
    build_sheet(demoted, f"Anchor class-4 rollout — everything demoted  (n={len(demoted)}, vivid companion)",
                OUT_DIR / "label_rubric_demoted_examples.png", cols=4)


if __name__ == "__main__":
    main()
