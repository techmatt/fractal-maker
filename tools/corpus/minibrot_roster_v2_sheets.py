#!/usr/bin/env python
r"""minibrot_roster_v2_sheets.py — the three companion sheets for the G-transfer readout.

Part C of prompt-process-minibrot-labels.md. Three vivid-companion contact sheets, read
beside docs/design/label_rubric.md, that put G's failure modes on minibrot fields in front
of the eye:

  1. class4     — every crop I called 4 (what a minibrot 4 looks like; n is tiny).
  2. hi_g_lo    — high G, low label: the screen said ACCEPT, I said 1 or 2. Top-N by G
                  (the most confident accepts I rejected). If these share a visible failure
                  mode, that mode is what G is measuring and Matt does not want it.
  3. sub_hi     — sub-cutoff, high label: the screen would have DISCARDED these (screen-reject
                  below the G cutoff, plus OOD-masked), and I called them 3 or 4. The direct
                  test of whether hunting near minibrots outside the screen's accepts pays.

Tiles are the VIVID companion renders (blue_orange, `<batch>/vivid/<image_id>.jpg`) — one map
across every row so the eye compares structure, not palette.

The PNGs are a REGENERABLE VIEW, so they go to `scratch/minibrot_roster_v2_sheets/` per the
generated-output convention — not into `docs/`. (They were briefly written into `docs/design/`
and gitignored there, which left `tests/test_repo_size_guard.py` permanently red on 4
uncovered multi-MB files; `tests/test_docs_tree.py` now forbids untracked content under
`docs/` outright.) Read them beside `docs/design/label_rubric.md`; rebuild with the one-liner
below.

  uv run python tools/corpus/minibrot_roster_v2_sheets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import corpus_common as cc  # noqa: E402

BATCH_ID = "2026-07-26_minibrot_roster_v2"
DRAW = ROOT / "data" / "minibrot_roster" / "batch_v1" / "draw.jsonl"
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

OUT_DIR = paths.scratch("minibrot_roster_v2_sheets")  # regenerable view -> scratch/, never docs/
HI_G_LO_MAX = 24                              # top-N by G of the 189 accepts I scored <=2
TW, TH, PAD, LAB = 384, 216, 6, 22            # tile 16:9, caption strip below
BG, STRIP, INK = (18, 18, 20), (30, 30, 34), (232, 232, 150)


def load_rows():
    bdir = Path(cc.batch_dir(BATCH_ID))
    scores = {r["image_id"]: r["label"]["score"]
              for r in (json.loads(l) for l in (bdir / "images.jsonl").read_text().splitlines() if l.strip())}
    draw = {d["image_id"]: d for d in (json.loads(l) for l in DRAW.read_text().splitlines() if l.strip())}
    vivid = Path(cc.vivid_dir(BATCH_ID))
    out = []
    for iid, lab in scores.items():
        d = draw[iid]
        out.append({
            "iid": iid, "label": lab, "G": d["G"], "fate": d["fate"],
            "degree": d["degree"], "period": d["period"], "band": d["band"],
            "vivid": vivid / f"{iid}.jpg",
        })
    return out


def gtxt(r):
    return f"G={r['G']:.2f}" if r["G"] is not None else "G=--"


def build_sheet(items, title, out_png, cols):
    n = len(items)
    rows_n = max(1, (n + cols - 1) // cols)
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
        cap = f"L{it['label']} {gtxt(it)} {it['fate'][:4]} d{it['degree']}p{it['period']} [{it['iid'][:8]}]"
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=STRIP)
        dr.text((x + 4, y + TH + 5), cap, fill=INK)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"  wrote {out_png.name}  ({n} tiles, {W}x{H})")


def main():
    rows = load_rows()

    # 1. class-4
    c4 = sorted([r for r in rows if r["label"] == 4], key=lambda r: r["iid"])

    # 2. high G, low label — accepts (screen said accept) scored 1|2, most-confident-first
    hi_lo_all = sorted([r for r in rows if r["fate"] == "accepted" and r["label"] <= 2],
                       key=lambda r: -r["G"])
    hi_lo = hi_lo_all[:HI_G_LO_MAX]

    # 3. sub-cutoff, high label — the screen would have DISCARDED these and I said 3|4.
    #    screen-rejects (below G cutoff) sorted by G desc, then the OOD-masked (no G) at the end.
    sub_rej = sorted([r for r in rows if r["fate"] == "rejected" and r["label"] >= 3],
                     key=lambda r: -r["G"])
    sub_ood = sorted([r for r in rows if r["fate"] == "ood_masked" and r["label"] >= 3],
                     key=lambda r: r["iid"])
    sub_hi = sub_rej + sub_ood

    print(f"class-4: {len(c4)}   high-G/low-label: {len(hi_lo_all)} (showing top {len(hi_lo)} by G)   "
          f"sub-cutoff/high-label: {len(sub_hi)} ({len(sub_rej)} screen-reject + {len(sub_ood)} OOD)")
    if len(hi_lo_all) > HI_G_LO_MAX:
        print(f"  NOTE: high-G/low-label truncated to the {HI_G_LO_MAX} highest-G of "
              f"{len(hi_lo_all)} accepts scored <=2 (full set in the readout).")

    build_sheet(c4,
                f"minibrot roster v2 - everything I called 4  (n={len(c4)}, vivid companion)",
                OUT_DIR / "minibrot_roster_v2_class4.png", cols=2)
    build_sheet(hi_lo,
                f"minibrot roster v2 - high G, low label: screen ACCEPTED, I said 1-2  "
                f"(top {len(hi_lo)} by G of {len(hi_lo_all)}, vivid companion)",
                OUT_DIR / "minibrot_roster_v2_hi_g_lo.png", cols=4)
    build_sheet(sub_hi,
                f"minibrot roster v2 - sub-cutoff, high label: screen would DISCARD, I said 3-4  "
                f"(n={len(sub_hi)}, vivid companion)",
                OUT_DIR / "minibrot_roster_v2_sub_hi.png", cols=4)


if __name__ == "__main__":
    main()
