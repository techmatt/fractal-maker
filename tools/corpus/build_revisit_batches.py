#!/usr/bin/env python
"""Build the class-3 REVISIT batches — re-judge every current class-3 row on the 1..4 scale.

Class 3 was the top bucket before class 4 existed, so it absorbed everything "good or better".
Re-judging the whole class-3 population on the full scale is the dominant acquisition channel
for class-4 examples. This is a REVISION pass: each row points back at its source row and, when
labeled, `merge_amendments.py` routes the new score into the source batch's amendment overlay
(`labels/amend_<source>.json`) — the source's original label is never mutated.

Selection: every row whose CURRENT resolved score == 3 (label_store.resolve_score WITH the
amendment overlay applied — so rows already demoted/promoted by an earlier revision are excluded
and rows currently sitting at 3 via an amendment are included). One revision row is emitted per
source reference (no join-key dedup), so every current-3 source row receives its own amendment;
the handful of locations that appear in two source batches simply appear twice (a free
intra-rater consistency probe).

Stratification: split into N chunks (default 4) round-robin across every (source_batch, family)
cell, so each chunk mirrors the global composition rather than being blocked by family/batch —
the same drift the cross-family anchor pass exists to prevent. Composition tables are written to
scratch/ for checking.

BLINDNESS (load-bearing — every row carries a prior label of 3):
  * Opaque image_ids `rev_c<chunk>_<NNNN>` — encode nothing (no family, no fate, no prior score).
    The crop filename / img.src / img.alt therefore leak nothing.
  * The browser is served a BLINDED manifest `blind.jsonl` (provenance emptied, label nulled),
    NOT `images.jsonl`. So the prior label (and the revises_* back-pointer) is not present in the
    fetched bytes at all — not in the DOM, not in JS memory, not recoverable by toggling reveal.
    The full-provenance `images.jsonl` is the MERGE-side file `merge_amendments.py` reads; the
    browser never fetches it (serve with `?manifest=blind.jsonl`).
  * Presentation order is a seeded shuffle (per-chunk `presentation_seed` in batch.json).

Presentation mirrors the anchor pass: canonical label crop (original identity + palette, via
pool_colormaps.json) beside the VIVID blue_orange companion (data/palettes/vivid_blue_orange.json,
read straight off the committed library — NOT re-derived).

Run:
  uv run python tools/corpus/build_revisit_batches.py --no-render   # manifests + tables only (fast)
  uv run python tools/corpus/build_revisit_batches.py               # + render all crops (background this)
Writes: data/label_corpus/batches/2026-07-28_revisit_class3_c{1..N}/{images.jsonl,blind.jsonl,
        batch.json,scores.json,crops/,vivid/} and scratch/revisit_class3/composition_*.md
Label:  http://localhost:8000/tools/viz/corpus_label.html?batch=<chunk>&manifest=blind.jsonl
Merge (after labeling each chunk): save the exported labels as <chunk>/scores.json, then
        uv run python tools/corpus/merge_amendments.py --batch <chunk> --apply
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

DATE = "2026-07-28"
GEN = "revisit_class3"
SEED = 20260728
N_CHUNKS = 4

REVISION_PALETTES = os.path.join(ROOT, "data", "palettes", "pool_colormaps.json")  # covers all 74 names
VIVID_PALETTE = "blue_orange"
VIVID_SOURCE = os.path.join(ROOT, "data", "palettes", "vivid_blue_orange.json")
SCRATCH = os.path.join(ROOT, "scratch", "revisit_class3")


def fam_of(render):
    return render.get("fractal_type") or render.get("family") or "mandelbrot"


def batch_id_for(chunk):
    return f"{DATE}_{GEN}_c{chunk}"


def collect_class3():
    """All current class-3 rows (resolve_score WITH amendments applied), one per source ref."""
    rows = []
    for lc in cr.iter_labeled():
        if lc.score == 3:
            rows.append(lc)
    return rows


def stratify(rows, n_chunks):
    """Round-robin every (source_batch, family) cell across chunks (continuing one global
    counter so remainders spread evenly) → each chunk mirrors the global composition."""
    cells = defaultdict(list)
    for lc in rows:
        cells[(lc.batch_id, fam_of(lc.render))].append(lc)
    rng = random.Random(SEED)
    chunks = [[] for _ in range(n_chunks)]
    counter = 0
    for cell in sorted(cells):
        members = cells[cell][:]
        rng.shuffle(members)
        for lc in members:
            chunks[counter % n_chunks].append(lc)
            counter += 1
    return chunks


def build_chunk_rows(chunk_idx, chunk_lcs):
    """Assign opaque ids (after a seeded shuffle so id order does not track the strata) and
    build the full (merge-side) row + the blinded (browser-side) row for each source ref."""
    rng = random.Random(SEED + 100 + chunk_idx)
    lcs = chunk_lcs[:]
    rng.shuffle(lcs)
    bid = batch_id_for(chunk_idx + 1)
    full_rows, blind_rows, render_items = [], [], []
    for i, lc in enumerate(lcs):
        iid = f"rev_c{chunk_idx + 1}_{i:04d}"
        render = dict(lc.render)   # source identity, verbatim (like-for-like re-judgement)
        # merge-side row: carries the back-pointer merge_amendments routes on.
        prov = cc.provenance_block(
            GEN, bid, family=fam_of(render),
            revises_batch_id=lc.batch_id, revises_image_id=lc.image_id, original_score=3)
        full_rows.append(cc.make_row(iid, render, prov, cc.label_block()))
        # browser-side row: only batch identity in provenance — NONE of the leak keys
        # (revises_*, original_score, family) appear at all, not even as null-valued keys, so the
        # prior label is not present in the served bytes / DOM / JS memory under any reveal state.
        blind_prov = {"generator_version": GEN, "batch_id": bid}
        blind_rows.append(cc.make_row(iid, dict(render), blind_prov, cc.label_block()))
        render_items.append((iid, render))
    return bid, full_rows, blind_rows, render_items


def write_chunk(bid, full_rows, blind_rows, chunk_idx):
    bdir = Path(cc.batch_dir(bid))
    bdir.mkdir(parents=True, exist_ok=True)
    cc.write_jsonl(full_rows, str(bdir / "images.jsonl"))
    cc.write_jsonl(blind_rows, str(bdir / "blind.jsonl"))
    comp = defaultdict(int)
    for r in full_rows:
        comp[(r["provenance"]["revises_batch_id"], r["provenance"]["family"])] += 1
    batch_json = dict(
        schema_version=1, batch_id=bid, generator_version=GEN, created=None, labeler=None,
        # distinct seed per chunk so the blind order differs across chunks.
        presentation_seed=(SEED + 1000 + chunk_idx) & 0xFFFFFFFF,
        vivid_companion=VIVID_PALETTE,
        served_manifest="blind.jsonl",
        purpose=("class-3 REVISIT: re-judge current class-3 rows on the 1..4 scale via the "
                 "amendment overlay. Blind (opaque ids + blind.jsonl); serve with "
                 "?manifest=blind.jsonl. Revisions -> merge_amendments.py."),
        counts=dict(total=len(full_rows)),
        composition={f"{b}|{f}": n for (b, f), n in sorted(comp.items())},
        render_defaults=dict(width=None, height=None, note="render fields taken per-row from the "
                             "source render block (varied geometry/maxiter/palette)"),
        render_recipe=cc.render_recipe_stamp(REVISION_PALETTES),
    )
    (bdir / "batch.json").write_text(json.dumps(batch_json, indent=2), encoding="utf-8")
    if not (bdir / "scores.json").exists():
        (bdir / "scores.json").write_text("{}", encoding="utf-8")
    return bdir, comp


def render_chunk(bid, render_items, workers=4):
    bdir = Path(cc.batch_dir(bid))
    crops = bdir / "crops"; vivid = bdir / "vivid"
    crops.mkdir(parents=True, exist_ok=True); vivid.mkdir(parents=True, exist_ok=True)

    def one(item):
        iid, render = item
        canon = crops / f"{iid}.jpg"
        if not canon.exists():
            cc.render_corpus_crop(render, str(canon), palette_source=REVISION_PALETTES, timeout=300)
        vout = vivid / f"{iid}.jpg"
        if not vout.exists():
            vr = dict(render); vr["palette"] = VIVID_PALETTE
            cc.render_corpus_crop(vr, str(vout), palette_source=VIVID_SOURCE, timeout=300)
        return iid

    todo = [it for it in render_items
            if not (crops / f"{it[0]}.jpg").exists() or not (vivid / f"{it[0]}.jpg").exists()]
    print(f"[{bid}] rendering {len(todo)}/{len(render_items)} rows (canonical + vivid) "
          f"workers={workers} ...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(one, todo):
            done += 1
            if done % 50 == 0:
                print(f"  [{bid}] ...{done}/{len(todo)} rows", flush=True)
    print(f"  [{bid}] crops -> {crops} + {vivid}", flush=True)


def write_composition_report(all_comp, chunk_sizes):
    os.makedirs(SCRATCH, exist_ok=True)
    # global (batch x family) table with per-chunk split
    fams = sorted({f for comp in all_comp for (_b, f) in comp})
    batches = sorted({b for comp in all_comp for (b, _f) in comp})
    lines = ["# class-3 revisit — chunk composition (stratification check)", "",
             f"chunks: {len(all_comp)}   sizes: {chunk_sizes}   total: {sum(chunk_sizes)}", ""]
    # per-family across chunks
    lines.append("## per family — count in each chunk (should be ~even)")
    lines.append("")
    lines.append("| family | " + " | ".join(f"c{i+1}" for i in range(len(all_comp))) + " | total |")
    lines.append("|" + "---|" * (len(all_comp) + 2))
    for f in fams:
        per = [sum(n for (b, ff), n in comp.items() if ff == f) for comp in all_comp]
        lines.append(f"| {f} | " + " | ".join(str(x) for x in per) + f" | {sum(per)} |")
    lines.append("")
    # per-source-batch across chunks
    lines.append("## per source batch — count in each chunk (should be ~even)")
    lines.append("")
    lines.append("| source_batch | " + " | ".join(f"c{i+1}" for i in range(len(all_comp))) + " | total |")
    lines.append("|" + "---|" * (len(all_comp) + 2))
    for b in batches:
        per = [sum(n for (bb, f), n in comp.items() if bb == b) for comp in all_comp]
        lines.append(f"| {b} | " + " | ".join(str(x) for x in per) + f" | {sum(per)} |")
    lines.append("")
    # full per-chunk (batch x family) detail
    for i, comp in enumerate(all_comp):
        lines.append(f"## chunk c{i+1} — (source_batch × family), {sum(comp.values())} rows")
        lines.append("")
        for (b, f), n in sorted(comp.items()):
            lines.append(f"- {b} | {f}: {n}")
        lines.append("")
    path = os.path.join(SCRATCH, "composition.md")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\ncomposition table -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true", help="write manifests + tables only")
    ap.add_argument("--chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-chunk", type=int, default=0, help="render only this chunk (1-based); 0 = all")
    a = ap.parse_args()
    if a.workers > 4:
        sys.exit("workers capped at 4 (project rule)")

    print("collecting current class-3 rows (amendments applied) ...", flush=True)
    rows = collect_class3()
    print(f"  {len(rows)} class-3 source refs")
    chunks = stratify(rows, a.chunks)

    built = []
    all_comp = []
    for ci, chunk_lcs in enumerate(chunks):
        bid, full_rows, blind_rows, render_items = build_chunk_rows(ci, chunk_lcs)
        bdir, comp = write_chunk(bid, full_rows, blind_rows, ci)
        all_comp.append(comp)
        built.append((bid, render_items))
        print(f"  {bid}: {len(full_rows)} rows -> {bdir}")
    write_composition_report(all_comp, [len(c) for c in chunks])

    if a.no_render:
        print("\n--no-render: manifests + tables written; skipping crops.")
        return
    for ci, (bid, render_items) in enumerate(built):
        if a.only_chunk and (ci + 1) != a.only_chunk:
            continue
        render_chunk(bid, render_items, a.workers)
    print("\ndone. Serve: uv run python -m http.server 8000 (from repo root), then open")
    for bid, _ in built:
        print(f"  http://localhost:8000/tools/viz/corpus_label.html?batch={bid}&manifest=blind.jsonl")


if __name__ == "__main__":
    main()
