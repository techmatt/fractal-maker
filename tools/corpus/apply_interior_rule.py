#!/usr/bin/env python
r"""apply_interior_rule.py — the interior auto-reject rule, applied as a RULE label.

MATT'S RULE (dictated 2026-08-01, firm): a crop whose frame is more than 30% black is
guaranteed class 1 for wallpaper emission. Unambiguous, no gray zone — so it is a rule the
labeler should not spend a keystroke on.

WHAT "BLACK" IS OPERATIONALIZED AS, and why it is the palette-independent one: the
**interior fraction** already recorded on every crawl candidate — `(~isfinite(field)).mean()`
over the VIEW-frame screen, which is the same quantity as the Rust `render::black_fraction`
the production emission gate uses (fraction of non-escaped samples, interior renders black
under `interior_mode=black`), and the same 0.30 threshold as `present.rs::BLACK_THRESH`.
Nothing is re-rendered: the number is read off `provenance.interior_fraction`.

THE FRAME MUST BE THE LABEL-CROP FRAME, and it is: the view screen measures at the frame
that was pushed (`view_fw`), which equals `render.fw` for every batched row — same center,
same 16:9 aspect, 64x36 vs 1280x720. The screen's maxiter is 4.8-13.9x the crop's, so a
pixel counted as interior by the screen is interior in the crop a fortiori: the measure
UNDER-fires relative to what the crop shows and cannot auto-reject a frame that is not
black-dominated. `[measured 2026-08-01 over all 730 supply-crawl rows;
scratch/supply_crawl/interior_auto_reject_rule.md]`

A RULE LABEL IS NOT A HUMAN LABEL. It is written with `label.labeler = "rule:<rule_id>"`
(never `matt`) and a per-batch record naming the rule, threshold, measure and date, so any
downstream consumer can separate the two. It obeys the store's one mutation: `null -> value`
only. An already-labeled row is never touched, and a row the rule fires on that Matt later
scores differently comes back as a LOUD `merge_scores` conflict, not a silent overwrite.

SCOPE: the label store, and nothing else. This is not a discovery-time gate and is not
applied to any historical batch — both are separate decisions (`retired.md` 2026-08-01 keeps
"interior mass as a quality AXIS" retired; this is a floor statement about one frame, the
same shape as the size band and the deployed black gate, not a monotone axis).

  uv run python tools/corpus/apply_interior_rule.py                 # dry run, the 4 batches
  uv run python tools/corpus/apply_interior_rule.py --apply
  uv run python tools/corpus/apply_interior_rule.py --batch <id> --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus_common as cc  # noqa: E402

RULE_ID = "interior_gt30_v1"
THRESHOLD = 0.30                      # strict >, mirroring present.rs BLACK_THRESH's strict <
RULE_SCORE = 1
LABELER = f"rule:{RULE_ID}"
MEASURE = "provenance.interior_fraction"
MEASURE_FRAME = "view@64x36; view_fw == render.fw, 16:9, interior = non-escaped fraction"
RECORD_NAME = f"rule_labels_{RULE_ID}.json"

BATCHES = (
    "2026-08-01_supply_crawl_strat_a_v1",
    "2026-08-01_supply_crawl_strat_b_v1",
    "2026-08-01_supply_crawl_uniform_v1",
    "2026-08-01_supply_crawl_exemplar_v1",
)


def fires(row, threshold: float = THRESHOLD) -> bool:
    """True iff the rule applies to this row: UNLABELED and measurably over the threshold.

    A row with no recorded measure never fires — an absent measure is not a low one."""
    if (row.get("label") or {}).get("score") is not None:
        return False
    v = (row.get("provenance") or {}).get(MEASURE.split(".", 1)[1])
    return v is not None and float(v) > threshold


def plan(rows, threshold: float = THRESHOLD) -> list[tuple[str, float]]:
    return [(r["image_id"], float(r["provenance"][MEASURE.split(".", 1)[1]]))
            for r in rows if fires(r, threshold)]


def apply_to_batch(bdir: str, *, date: str, threshold: float = THRESHOLD,
                   write: bool = False) -> dict:
    """Label every firing row in ONE batch. Returns the per-batch report."""
    rows = cc.read_jsonl(os.path.join(bdir, "images.jsonl"))
    hit = plan(rows, threshold)
    ids = {i for i, _ in hit}
    labeled_before = sum(1 for r in rows if (r["label"] or {}).get("score") is not None)
    rep = dict(batch=os.path.basename(bdir), rows=len(rows), auto_labeled=len(hit),
               human_labeled=labeled_before,
               remaining=len(rows) - labeled_before - len(hit))
    if not write or not hit:
        return rep

    for r in rows:
        if r["image_id"] in ids:
            r["label"] = dict(score=RULE_SCORE, labeler=LABELER, labeled_at=date)
    cc.write_jsonl(rows, os.path.join(bdir, "images.jsonl"))

    # The SERVED manifest gets the score seeded so the rig treats these rows as labeled and
    # "next unlabeled" skips them (the page seeds local scores from `label.score` on load and
    # never overwrites a score the labeler already set). Score + labeler only: the served row
    # stays free of every selection axis, measure included.
    bp = os.path.join(bdir, "blind.jsonl")
    if os.path.exists(bp):
        blind = cc.read_jsonl(bp)
        for r in blind:
            if r["image_id"] in ids:
                r["label"] = dict(score=RULE_SCORE, labeler=LABELER, labeled_at=date)
        cc.write_jsonl(blind, bp)

    record = dict(rule_id=RULE_ID, applied=date, score=RULE_SCORE, labeler=LABELER,
                  threshold=threshold, comparison="strict >", measure=MEASURE,
                  measure_frame=MEASURE_FRAME, n=len(hit),
                  labels={i: v for i, v in sorted(hit)})
    with open(os.path.join(bdir, RECORD_NAME), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=False)
    return rep


def sheet(batches, out_png: str) -> str:
    """ONE contact sheet of everything the rule labeled — vivid companion, interior
    captioned. Acceptance by eye: the rule is Matt's, so the sheet is what he confirms it
    against, and a rule label that does not LOOK black-dominated is the failure it catches."""
    from PIL import Image, ImageDraw
    TW, TH, PAD, LAB, TITLE_H = 384, 216, 6, 22, 34
    items = []
    for b in batches:
        p = os.path.join(cc.batch_dir(b), RECORD_NAME)
        if not os.path.exists(p):
            continue
        rec = json.load(open(p, encoding="utf-8"))
        vd = cc.vivid_dir(b)
        items += [(b, i, v, os.path.join(vd, f"{i}.jpg"))
                  for i, v in sorted(rec["labels"].items(), key=lambda kv: -kv[1])]
    cols = 6
    rows_n = (len(items) + cols - 1) // cols
    W = cols * (TW + PAD) + PAD
    im = Image.new("RGB", (W, TITLE_H + rows_n * (TH + LAB + PAD) + PAD), (18, 18, 20))
    dr = ImageDraw.Draw(im)
    dr.text((PAD + 2, 9), f"{RULE_ID}: auto-labeled class {RULE_SCORE} "
                          f"({MEASURE} > {THRESHOLD}) — n={len(items)}, vivid companion",
            fill=(232, 232, 150))
    for k, (b, iid, v, vp) in enumerate(items):
        r, c = divmod(k, cols)
        x, y = PAD + c * (TW + PAD), TITLE_H + PAD + r * (TH + LAB + PAD)
        tile = (Image.open(vp).convert("RGB").resize((TW, TH)) if os.path.exists(vp)
                else Image.new("RGB", (TW, TH), (60, 20, 20)))
        im.paste(tile, (x, y))
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=(30, 30, 34))
        dr.text((x + 4, y + TH + 5), f"interior {v:.3f}  {b.split('supply_crawl_')[-1]}  "
                                     f"[{iid}]", fill=(232, 232, 150))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    im.save(out_png)
    print(f"  wrote {out_png}  ({len(items)} tiles, {im.width}x{im.height})")
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", default=None,
                    help="batch_id (repeatable; default: the four supply-crawl batches)")
    ap.add_argument("--sheet", metavar="PNG", nargs="?", const="", default=None,
                    help="build the acceptance contact sheet of everything already "
                         "rule-labeled and exit (default path under scratch/)")
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--date", default="2026-08-01")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    a = ap.parse_args()
    if a.corpus_root:
        cc.BATCHES_DIR = a.corpus_root
    if a.sheet is not None:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
        import paths  # noqa: E402  (scratch/ is where a regenerable view goes)
        sheet(a.batch or BATCHES,
              a.sheet or str(paths.scratch("interior_rule", f"{RULE_ID}_auto_labeled.png")))
        return

    print(f"rule {RULE_ID}: {MEASURE} > {a.threshold} -> score {RULE_SCORE} "
          f"as {LABELER} ({'APPLY' if a.apply else 'dry run'})")
    tot = 0
    for b in (a.batch or BATCHES):
        rep = apply_to_batch(cc.batch_dir(b), date=a.date, threshold=a.threshold,
                             write=a.apply)
        tot += rep["auto_labeled"]
        print(f"  {rep['batch']:38s} rows {rep['rows']:3d}  human {rep['human_labeled']:3d}"
              f"  auto {rep['auto_labeled']:3d}  remaining {rep['remaining']:3d}")
    print(f"  total auto-labeled: {tot}")
    if not a.apply:
        print("  DRY RUN — pass --apply to write")


if __name__ == "__main__":
    main()
