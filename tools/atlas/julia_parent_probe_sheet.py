#!/usr/bin/env python
r"""julia parent-sourcing probe — vivid-palette sheets for the eye (§2).

Two montages under scratch/julia_parent_probe/sheets/:
  admitted.png  — every distinct-q3 julia:mandelbrot admission, vivid `default` palette.
  rejects.png   — a sample of REJECTED harvest checks (admitted=False): precanon dups and
                  canon-not-q3, so the eye can confirm the dup rate isn't buying barren c.

A falling precanon_dup rate is blind to admissions collapsing or wandering into dead c-regions;
these sheets are the human check the primary metric can't make.

Run: uv run python tools/atlas/julia_parent_probe_sheet.py
"""
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
UNIT = ROOT / "data/discovery/julia_parent_probe/breadth"
OUT = ROOT / "scratch/julia_parent_probe/sheets"
TILES = OUT / "tiles"
PALETTE = "default"          # vivid UF palette (§2 "vivid-palette sheet")
TW, TH, SS, MAXITER = 480, 270, 2, 1000
WORKERS = 4                  # CLAUDE.md cap
REJECT_SAMPLE = 40


def _src(mix):
    if mix == "sampler":
        return "SMP"
    if isinstance(mix, str) and mix.startswith("julia_hook"):
        return "HOOK"
    return str(mix)[:4]


def render(rec):
    out = TILES / f"{rec['uid']}.jpg"
    if out.exists():
        return True
    cmd = [str(BIN), "render-one", "--julia", "--c", str(rec["c_re"]), str(rec["c_im"]),
           "--cx", str(rec["cx"]), "--cy", str(rec["cy"]), "--fw", str(rec["fw"]),
           "--width", str(TW), "--height", str(TH), "--supersample", str(SS),
           "--maxiter", str(MAXITER), "--palette", PALETTE, "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        sys.stderr.write(f"[render {rec['uid']}] FAILED: {r.stderr[-160:]}\n")
        return False
    return True


def montage(recs, path, title):
    from PIL import Image, ImageDraw
    if not recs:
        print(f"  (no tiles for {title})")
        return
    cols = 8
    rows_n = math.ceil(len(recs) / cols)
    lab_h = 26
    sheet = Image.new("RGB", (cols * TW, rows_n * (TH + lab_h) + 24), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill=(240, 240, 240))
    for i, rec in enumerate(recs):
        cx, cy = i % cols, i // cols
        x, y = cx * TW, 24 + cy * (TH + lab_h)
        p = TILES / f"{rec['uid']}.jpg"
        if p.exists():
            sheet.paste(Image.open(p).convert("RGB").resize((TW, TH)), (x, y))
        else:
            draw.rectangle([x, y, x + TW, y + TH], fill=(40, 0, 0))
        draw.text((x + 3, y + TH + 5), rec["label"], fill=(225, 225, 225))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"  wrote {path}  ({len(recs)} tiles)")


def main():
    TILES.mkdir(parents=True, exist_ok=True)
    led = [json.loads(l) for l in open(UNIT / "outcome_ledger.jsonl", encoding="utf-8") if l.strip()]
    harv = [json.loads(l) for l in open(UNIT / "harvest_log.jsonl", encoding="utf-8") if l.strip()]

    # admitted: distinct-q3 julia:mandelbrot ledger rows. Join to harvest_log admitted rows by
    # coord to recover the ROOT supply (ledger mix_source is the generic "steered" tag).
    src_by_xy = {}
    for h in harv:
        if h.get("admitted") and h["partition"] == "julia:mandelbrot":
            src_by_xy[(round(h["cx"], 10), round(h["cy"], 10))] = h.get("mix_source")

    adm = []
    for r in led:
        if r.get("family") == "julia:mandelbrot" and r.get("distinct") and r.get("decoded_class") == 3:
            mix = src_by_xy.get((round(float(r["seed_cx"]), 10), round(float(r["seed_cy"]), 10)))
            adm.append(dict(
                uid=f"adm_{r['id']}", c_re=r["julia_c_re"], c_im=r["julia_c_im"],
                cx=r["outcome_cx"], cy=r["outcome_cy"], fw=r["outcome_fw"],
                label=f"{_src(mix)} pg={float(r.get('p_good',0)):.2f} c=({float(r['julia_c_re']):.3f},{float(r['julia_c_im']):.3f})",
            ))

    # rejects: sample admitted=False julia:mandelbrot harvest checks (precanon dups + canon-not-q3).
    rej_rows = [h for h in harv if h["partition"] == "julia:mandelbrot" and not h.get("admitted")]
    # deterministic spread: every kth so the sample spans batches/c's, not one cluster.
    step = max(1, len(rej_rows) // REJECT_SAMPLE)
    rej_s = rej_rows[::step][:REJECT_SAMPLE]
    rej = []
    for i, h in enumerate(rej_s):
        fate = "predup" if h.get("precanon_dup") is not None else f"c{h.get('canon_decoded')}"
        rej.append(dict(
            uid=f"rej_{i:04d}", c_re=h["julia_c_re"], c_im=h["julia_c_im"],
            cx=h["cx"], cy=h["cy"], fw=h["fw"],
            label=f"{_src(h.get('mix_source'))} {fate} cg={float(h.get('cheap_pgood',0)):.2f}",
        ))

    allrec = adm + rej
    print(f"rendering {len(allrec)} tiles ({len(adm)} admitted + {len(rej)} rejects) ss{SS} vivid…")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(render, allrec))

    montage(adm, OUT / "admitted.png",
            f"julia_parent_probe — {len(adm)} ADMITTED julia:mandelbrot (vivid `default`)  SMP=sampler HOOK=parent-hook")
    montage(rej, OUT / "rejects.png",
            f"julia_parent_probe — {len(rej)}/{len(rej_rows)} REJECT sample (predup / cN=canon-decode)")


if __name__ == "__main__":
    main()
