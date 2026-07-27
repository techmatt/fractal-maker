#!/usr/bin/env python
"""Build the mixed-family ANCHOR batch (~60 crops) for the class-4 rollout (Part B).

Purpose: fix the class-4 bar ACROSS families BEFORE the large minibrot volume is labeled,
so the bar is not silently defined by minibrot crops. Labeled on the 1..4 scale as REVISIONS
(exercising the amendment path end to end) plus a handful of fresh minibrot crops.

Composition (~60):
  * ~52 already-labeled CLASS-3 locations spanning families (mandelbrot, the julia twins,
    phoenix, native multibrot, and the julia-multibrot twins), drawn spread across existing
    batches, favouring variety of look over any score ordering. Each is rendered at its
    ORIGINAL render identity (same location + palette + composition Matt first judged) so the
    revision is a like-for-like re-judgement. Provenance points back to the source row
    (revises_batch_id / revises_image_id / original_score); when labeled, merge_amendments.py
    routes the new score to that source batch's amendment stream — never mutating the original.
  * 8 minibrot crops (accepted framings) from the roster draw (build_minibrot_batch draw),
    so the class-4 bar sees minibrot material too. These are FRESH rows (no revises_*).

Crops are regenerable (out-of-tree); the durable artifacts are images.jsonl + batch.json.

Run:  uv run python tools/corpus/build_anchor_batch.py [--no-render]
Reads:   the label corpus (via corpus_reader), data/minibrot_roster/batch_v1/anchor_minibrot_picks.jsonl
Writes:  data/label_corpus/batches/2026-07-26_anchor_class4_v1/{images.jsonl,batch.json,crops/}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus_common as cc     # noqa: E402
import corpus_reader as cr     # noqa: E402
import label_store as ls       # noqa: E402

BATCH_ID = "2026-07-26_anchor_class4_v1"
GEN = "anchor_class4_v1"
SEED = 20260726

# class-3 revision quota per family (spans mandelbrot / julia twins / phoenix / native
# multibrot / julia-multibrot twins). Sums to ~52; + 8 minibrot fresh = ~60.
FAMILY_QUOTA = {
    "mandelbrot": 12,
    "julia": 10,
    "phoenix": 8,
    "julia_multibrot3": 4, "julia_multibrot4": 4, "julia_multibrot5": 4,
    "multibrot3": 4, "multibrot4": 3, "multibrot5": 3,
}
REVISION_PALETTES = os.path.join(ROOT, "data", "palettes", "pool_colormaps.json")   # covers all
MINIBROT_PALETTES = os.path.join(ROOT, "data", "palettes", "score3_colormaps.json")
ANCHOR_PICKS = os.path.join(ROOT, "data", "minibrot_roster", "batch_v1",
                            "anchor_minibrot_picks.jsonl")
CROP_W, CROP_H, CROP_SS = 1280, 720, 4
CROP_FILTER, INTERIOR_MODE, COMPOSITION = "lanczos3", "black", "center"


def _fam_of(render):
    return render.get("fractal_type") or render.get("family") or "mandelbrot"


def collect_class3():
    """{family: [ (batch_id, image_id, render) ]} for all ORIGINAL class-3 rows, deduped by
    coordinate join_key (never the same location twice)."""
    byfam = defaultdict(list)
    seen = set()
    for lc in cr.iter_labeled():
        if lc.score != 3:
            continue
        key = ls.join_key(lc.render)
        if key in seen:
            continue
        seen.add(key)
        byfam[_fam_of(lc.render)].append((lc.batch_id, lc.image_id, lc.render))
    return byfam


def draw_revisions(byfam):
    """Per family, spread the quota across source batches (round-robin) with a seeded shuffle
    within each batch, favouring variety of look."""
    rng = random.Random(SEED)
    picks = []
    for fam, quota in FAMILY_QUOTA.items():
        pool = byfam.get(fam, [])
        by_batch = defaultdict(list)
        for row in pool:
            by_batch[row[0]].append(row)
        for b in by_batch:
            rng.shuffle(by_batch[b])
        batches = sorted(by_batch)
        rng.shuffle(batches)
        chosen, bi = [], 0
        while len(chosen) < quota and any(by_batch[b] for b in batches):
            b = batches[bi % len(batches)]
            if by_batch[b]:
                chosen.append(by_batch[b].pop())
            bi += 1
        if len(chosen) < quota:
            print(f"  WARN family {fam}: only {len(chosen)}/{quota} available")
        picks.extend((fam, *c) for c in chosen)
    return picks


def build_rows(revisions, minibrot_picks):
    rng = random.Random(SEED + 1)
    rows = []                      # (image_id, row, palette_source)
    for i, (fam, src_batch, src_iid, render) in enumerate(revisions):
        iid = f"anchor_{fam}_{i:02d}"
        prov = cc.provenance_block(
            GEN, BATCH_ID, family=fam,
            revises_batch_id=src_batch, revises_image_id=src_iid, original_score=3)
        rows.append((iid, cc.make_row(iid, dict(render), prov, cc.label_block()),
                     REVISION_PALETTES))

    mb_names = [e["name"] for e in json.load(open(MINIBROT_PALETTES, encoding="utf-8"))
                if isinstance(e, dict) and e.get("name")]
    for j, pk in enumerate(minibrot_picks):
        # atom_id in the id so a re-draw (different picks) yields new ids -> clean re-render.
        iid = f"anchor_mb_{j:02d}_{pk['atom_id']}"
        pal = mb_names[rng.randrange(len(mb_names))]
        render = cc.render_block(cx=pk["cx"], cy=pk["cy"], fw=pk["fw"], maxiter=pk["maxiter"],
                                 palette=pal, composition=COMPOSITION, width=CROP_W,
                                 height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                                 interior_mode=INTERIOR_MODE)
        render["fractal_type"] = pk["family"]
        render["c_re"] = None
        render["c_im"] = None
        prov = cc.provenance_block(
            GEN, BATCH_ID, family=pk["family"], focus_score=pk.get("G"),
            stratum=pk["band"], decoded_class=pk["fate"],
            descend_mode=f"minibrot_d{pk['degree']}_p{pk['period']}")
        rows.append((iid, cc.make_row(iid, render, prov, cc.label_block()), MINIBROT_PALETTES))
    return rows


def render_all(rows, workers=4):
    bdir = Path(cc.batch_dir(BATCH_ID))
    crops = bdir / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    def one(item):
        iid, row, palsrc = item
        out = crops / f"{iid}.jpg"
        if out.exists():
            return iid, False
        cc.render_corpus_crop(row["render"], str(out), palette_source=palsrc, timeout=180)
        return iid, True
    todo = [it for it in rows if not (crops / f"{it[0]}.jpg").exists()]
    print(f"rendering {len(todo)}/{len(rows)} anchor crops (workers={workers}) ...", flush=True)
    made = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for iid, did in ex.map(one, todo):
            made += did
            if made and made % 10 == 0:
                print(f"  ...{made} rendered", flush=True)
    print(f"  crops -> {crops}  ({made} rendered this run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.workers > 4:
        sys.exit("workers capped at 4 (project rule)")

    print("anchor batch: collecting class-3 rows by family ...")
    byfam = collect_class3()
    for fam in FAMILY_QUOTA:
        print(f"  {fam:20s} available {len(byfam.get(fam, [])):4d}  quota {FAMILY_QUOTA[fam]}")
    revisions = draw_revisions(byfam)

    minibrot_picks = []
    if os.path.exists(ANCHOR_PICKS):
        minibrot_picks = [json.loads(l) for l in open(ANCHOR_PICKS, encoding="utf-8") if l.strip()]
    else:
        print(f"  NOTE: {ANCHOR_PICKS} absent — run build_minibrot_batch draw first for the "
              f"8 minibrot picks. Building the revision rows only for now.")

    rows = build_rows(revisions, minibrot_picks)
    bdir = Path(cc.batch_dir(BATCH_ID))
    bdir.mkdir(parents=True, exist_ok=True)
    cc.write_jsonl([r for _, r, _ in rows], str(bdir / "images.jsonl"))
    batch_json = dict(
        schema_version=1, batch_id=BATCH_ID, generator_version=GEN, created=None, labeler=None,
        purpose="mixed-family class-4 anchor; class-3 revisions (amendment path) + 8 minibrot",
        counts=dict(total=len(rows), revisions=len(revisions), minibrot_fresh=len(minibrot_picks)),
        family_quota=FAMILY_QUOTA,
        render_defaults=dict(width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE, composition=COMPOSITION,
                             note="revisions render at their ORIGINAL palette/identity"),
        render_recipe=cc.render_recipe_stamp(REVISION_PALETTES),
    )
    (bdir / "batch.json").write_text(json.dumps(batch_json, indent=2), encoding="utf-8")
    (bdir / "scores.json").write_text("{}", encoding="utf-8")
    print(f"  batch -> {bdir}  ({len(rows)} rows = {len(revisions)} revisions + "
          f"{len(minibrot_picks)} minibrot)")
    if not a.no_render:
        render_all(rows, a.workers)
    print("done. Label in tools/viz/corpus_label.html "
          f"(?batch={BATCH_ID}); revisions -> merge_amendments.py, minibrot -> merge_scores.py.")


if __name__ == "__main__":
    main()
