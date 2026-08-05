#!/usr/bin/env python
r"""build_combined_label_sheet.py — serve one or more registered batches as ONE blind sitting.

A PRESENTATION MERGE, AND NOTHING ELSE. The source batches stay exactly as registered —
their `assign_split` classification, their train/eval side, their images.jsonl, their own
`blind.jsonl` — none of it moves. What this builds is a *sheet*: one blinded manifest over
the union, one opaque id per row, one crop tree named by those ids, and the routing map that
sends the sitting's single `labels.json` back to each row's own registered batch
(`merge_scores.py --route`). Nothing downstream can tell the sheet existed.

WHY THE SHEET DIRECTORY HAS NO images.jsonl. It lives under `data/label_corpus/batches/` so
its crop trees relocate out of the working tree by the same class rule as every other batch
(`artifacts._is_label_corpus_crop`). But every corpus consumer — the trainer's
`corpus_reader`, the reachability guard, the label store — discovers batches by globbing
`*/images.jsonl`. A sheet that carried one would union 870 rows that already exist in three
other batches and double-count every label in training. So the sheet carries `sheet.jsonl`
(served) and `route.json` (merge-side, never fetched), and `test_combined_label_sheet.py`
holds that absence as a tripwire.

THE ORDER IS THE POINT. A seeded shuffle over the union, then a stratified round-robin over
every (source_batch x family) cell: each cell is dealt evenly across the whole sitting, so
the count of any cell in any prefix is within 1 of its proportional share. Without it the
union is three blocks — and the first two hours of a sitting would be one source's material,
which is a bar that drifts against provenance. The file order IS that order, so the page must
not reshuffle it: `batch.json` records `presentation_order:"file"`.

BLINDING IS BY ABSENCE. The served row is `image_id` (opaque) + `render` + a null `label`.
The whole `provenance` block is DROPPED, not emptied — it carries `batch_id`, which is the
source giveaway, on top of every selection key the source builders already treat as a leak.
The opaque id is assigned POST-shuffle, so id order is presentation order and encodes nothing
about which batch or which cell a row came from.

ONE SHEET PER `SheetSpec`, AND THE RULES ARE THE MODULE'S, NOT THE INSTANCE'S. Every property
above — the ±1 apportionment sequencing, the post-shuffle opaque id, blinding by absence, the
no-`images.jsonl` refusal, the routed merge — is a property of the CODE. A sheet is a handful
of instance constants (which batches, which seed, which id prefix) in `SPECS`, so a second
sitting inherits the guards instead of growing a second copy of them that can drift.

  uv run python tools/corpus/build_combined_label_sheet.py verify        # sources, pre-build
  uv run python tools/corpus/build_combined_label_sheet.py build [--apply]
  uv run python tools/corpus/build_combined_label_sheet.py check         # the built bytes
  uv run python tools/corpus/build_combined_label_sheet.py merge-dryrun  # export -> the sources
  ... each takes `--spec <name>` (default `q4_combined`; see SPECS).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "viz")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import artifacts as _artifacts        # noqa: E402
import corpus_common as cc            # noqa: E402
import label_store as ls              # noqa: E402
import merge_scores as ms             # noqa: E402
import paths                          # noqa: E402
from tools.v7 import build_manifest as bm   # noqa: E402  (assign_split — the authority)

STAMP = "2026-08-03"
SHEET_ID = f"{STAMP}_q4_combined_label_v1"
SOURCES = ("2026-08-03_q4_harvest_ranked_v1",
           "2026-08-03_q4_near_minibrot_v1",
           "2026-08-03_q4_uniform_eval_v1")
SHEET_MANIFEST = "sheet.jsonl"
ROUTE_FILE = "route.json"

# One seed for the union shuffle, one salt for the opaque id. Both recorded in batch.json so
# the sheet is reproducible byte for byte from the three source batches.
UNION_SEED = 0x4C0B31
ID_SALT = "q4-combined-2026-08-03"

# Fields that must never reach the served manifest. The source builders' own leak list plus
# the two the UNION adds: `batch_id` (which source) and `generator_version` (same, one hop
# removed). Asserted against the served BYTES, not the row objects.
SHEET_LEAK_KEYS = ("batch_id", "generator_version", "provenance", "selection_role", "stratum",
                   "fate", "rank_tier", "rank_score", "queue_rank", "cheap_eord", "cheap_pgood",
                   "canon_eord", "canon_pgood", "canon_decoded", "reframe_decoded",
                   "decoded_class", "tau_h", "tau_rec", "t_good", "eord", "p_good", "p_notbad",
                   "ladder_rung", "ladder_radius", "atom_size", "atom_period", "atom_id",
                   "atom_source", "scorer_version", "draw_rule", "family", "degree", "period",
                   "band_coverage", "guard_verdict", "original_score", "revises_batch_id",
                   "revises_image_id")


# =========================================================================== #
# the instance: which batches, which seed, which id prefix. Everything else is code.
# =========================================================================== #
@dataclass(frozen=True)
class SheetSpec:
    """One sheet. `max_run_source`/`max_run_family` are the "no stretch is single-X" caps,
    and they are PER-SPEC because they depend on the cell mix: a sheet with one source batch
    has a same-source run equal to its length by construction, which is not a defect. Set each
    one step above the MEASURED spread for its own sheet (see `stage_check`'s note)."""
    name: str
    sheet_id: str
    sources: tuple
    seed: int
    salt: str
    id_prefix: str
    purpose: str
    max_run_source: int
    max_run_family: int


Q4_COMBINED = SheetSpec(
    name="q4_combined",
    sheet_id=SHEET_ID,
    sources=SOURCES,
    seed=UNION_SEED,
    salt=ID_SALT,
    id_prefix="qc",
    purpose=("PRESENTATION MERGE of three registered q4 batches into ONE blind sitting. "
             "This directory is NOT a batch: it holds no images.jsonl, no labels, and no "
             "registration. Every label exported from it routes back to its own source "
             "batch via route.json (merge_scores.py --route); the three registrations "
             "are untouched."),
    max_run_source=4,
    max_run_family=5,
)

# --- the harvest-v2 sitting (2026-08-03) ------------------------------------------------- #
# ONE source batch, so the (source x family) cell grid collapses to FAMILY and the
# same-source run cap is vacuous (every row is that source) — it is set to the sheet length
# rather than deleted, because a cap that is absent and a cap that is trivially satisfied read
# differently in a check log, and the second is the true statement here.
V2_SITTING_SOURCES = ("2026-08-03_v2_sitting_v1",)
V2_SITTING = SheetSpec(
    name="v2_sitting",
    sheet_id="2026-08-03_v2_sitting_sheet_v1",
    sources=V2_SITTING_SOURCES,
    seed=0x5177,
    salt="v2-sitting-2026-08-03",
    id_prefix="vs",
    purpose=("The harvest-v2 proving run's ONE labelling sitting, served blind. This "
             "directory is NOT a batch: no images.jsonl, no labels, no registration. The "
             "rows live in the registered batch named in source_batches; the single export "
             "routes back through route.json (merge_scores.py --route)."),
    max_run_source=10_000,      # one source: every run is single-source by construction
    # One step above the MEASURED spread, same rule as q4's 4/5. The built sheet's longest
    # same-family run is 1 — nine cells of comparable size interleave perfectly — and a
    # 5-seed sweep over plausible re-mixes of this population gave 2. 3 leaves room for a
    # reseed without going slack; a run of 4+ here would mean the mix collapsed onto one
    # family, which is worth failing on.
    max_run_family=3,
)

# --- the steady-state telemetry run's sitting (2026-08-05) ------------------------------- #
# TWO source batches — the crawl leg's ranked residue and the same run's dive — cut as ONE
# sitting (`sitting_cutter.STEADY_STATE_SITTING`) and served as ONE page, so the (source x
# family) cell grid is real here and the same-source run cap binds for the first time since
# q4_combined.
STEADY_STATE_SITTING_SOURCES = ("2026-08-05_steady_state_ranked_v1",
                                "2026-08-05_steady_state_dive_v1")
STEADY_STATE_SITTING = SheetSpec(
    name="steady_state",
    sheet_id="2026-08-05_steady_state_sitting_sheet_v1",
    sources=STEADY_STATE_SITTING_SOURCES,
    seed=0x57ED0805,
    salt="steady-state-sitting-2026-08-05",
    id_prefix="ss",
    purpose=("The 2026-08-05 steady-state run's ONE labelling sitting: its crawl-leg "
             "record-and-rank residue and its own dive leg, cut once against a single "
             "1000-row cap and served blind. This directory is NOT a batch: no images.jsonl, "
             "no labels, no registration. The rows live in the two registered batches named "
             "in source_batches; the single export routes back through route.json "
             "(merge_scores.py --route)."),
    # One step above the MEASURED spread for THIS sheet, same rule as q4's 4/5 — see
    # `stage_check`'s note. MEASURED: same-source run 15, same-family run 2, and the source
    # figure is seed-INVARIANT across a 6-seed sweep (the shuffle permutes rows within a cell,
    # not the cell sequence).
    #
    # 16 is not a slack 4. The two sources are 654/94 — 87.4% / 12.6% — so a deal that holds
    # every cell within +/-1 of its share puts the minority source ~8 rows apart on average and
    # 15 apart at its widest BY CONSTRUCTION. The cap is the "no stretch is single-X" statement
    # scaled to the mix it is made about; setting it at q4's 4 would fail a correct deal, which
    # is `verification_practice.md` §4's getting-trained-out failure. The load-bearing balance
    # invariant is the +/-1 prefix bound, which passes at 0.791.
    max_run_source=16,
    max_run_family=3,
)

SPECS = {s.name: s for s in (Q4_COMBINED, V2_SITTING, STEADY_STATE_SITTING)}


# =========================================================================== #
# shared helpers
# =========================================================================== #
def sheet_dir(spec: SheetSpec = Q4_COMBINED) -> Path:
    return Path(cc.batch_dir(spec.sheet_id))


def _loc():
    """tools/corpus/location.py — the per-family render-constant registry."""
    import location
    return location


def family_of(row) -> str:
    """The row's render family — the cell axis crossed with the source batch. Falls back to
    the render block's fractal_type, which is version-invariant and always present."""
    return (row.get("provenance") or {}).get("family") or row["render"]["fractal_type"]


def load_sources(spec: SheetSpec = Q4_COMBINED):
    """[(batch_id, row)] over every source batch, in registered order."""
    out = []
    for b in spec.sources:
        p = os.path.join(cc.batch_dir(b), "images.jsonl")
        if not os.path.exists(p):
            raise SystemExit(f"source batch missing images.jsonl: {p}")
        out += [(b, r) for r in cc.read_jsonl(p)]
    return out


def opaque_id(slot: int, batch: str, image_id: str, spec: SheetSpec = Q4_COMBINED) -> str:
    """`<prefix><slot>_<hash>` — slot is presentation position (assigned POST-shuffle, so it
    orders the sitting and nothing else); the hash makes the id a stable function of the row it
    stands for, so a rebuild of the sheet reproduces the same ids."""
    h = hashlib.blake2b(f"{spec.salt}|{batch}|{image_id}".encode(), digest_size=4).hexdigest()
    return f"{spec.id_prefix}{slot:04d}_{h}"


def ordered_union(pairs, seed: int = UNION_SEED):
    """Seeded shuffle over the union, then a stratified round-robin over (source, family).

    The deal is GREEDY LARGEST-DEFICIT (Webster/Sainte-Laguë sequencing): at position L the
    next row comes from the cell whose running count is furthest below its proportional share
    L*n_c/N. That is the apportionment-sequencing rule, and it is what actually holds every
    cell within 1 of its share in EVERY prefix — the check asserts the bound on the built
    order rather than trusting it.

    The obvious alternative — lay each cell's rows at (i+0.5)/n_c and sort by that key — does
    NOT hold ±1 here and was measured failing it: with 15 cells its per-cell rounding errors
    accumulate through the shared threshold, and the 290-row near-minibrot cell reached a
    deviation of 1.67. Cheaper to compute, wrong at this cell count.

    Ties break on a seeded per-cell jitter, so equal-deficit cells do not resolve in the same
    direction at every collision (which would group one cell ahead of another all sitting).
    """
    rnd = random.Random(seed)
    shuffled = list(pairs)
    rnd.shuffle(shuffled)

    per_cell = defaultdict(list)
    for batch, row in shuffled:
        per_cell[(batch, family_of(row))].append((batch, row))

    cells = sorted(per_cell)
    jitter = {c: random.Random(f"{seed}|{c}").random() for c in cells}
    n_c = {c: len(per_cell[c]) for c in cells}
    N = sum(n_c.values())
    taken = {c: 0 for c in cells}
    cursor = {c: 0 for c in cells}

    out = []
    for L in range(1, N + 1):
        c = max((c for c in cells if cursor[c] < n_c[c]),
                key=lambda c: (L * n_c[c] / N - taken[c], jitter[c], c))
        batch, row = per_cell[c][cursor[c]]
        cursor[c] += 1
        taken[c] += 1
        out.append((c, batch, row))
    return out


# =========================================================================== #
# verify — the SOURCES, before anything is built
# =========================================================================== #
def stage_verify(args) -> int:
    spec = _spec(args)
    L, ok = [], True

    def emit(s=""):
        L.append(s)
        print(s)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    emit(f"=== {spec.name} label sheet — SOURCE verification ({STAMP}) ===")
    pairs = load_sources(spec)
    emit(f"union: {len(pairs)} rows over {len(spec.sources)} batches")

    # --- 1. registration: assign_split vs label_store, per batch ---------------
    emit("\n[registration]")
    for b in spec.sources:
        split, biased, source = bm.assign_split({"batch": b, "ft": "mandelbrot"})
        bj = json.loads((Path(cc.batch_dir(b)) / "batch.json").read_text(encoding="utf-8"))
        check(f"{b}: registered explicitly", source != "unregistered",
              f"-> {(split, biased, source)}")
        check(f"{b}: batch.json records the same classification",
              bj["registration"]["assign_split"] == [split, biased, source])
        # THE contradiction the two authorities exist to catch: classified unbiased/eval
        # while label_store registers the batch train-side-only. Hard abort, never papered.
        contra = bm.registration_contradictions([{"batch": b, "biased": biased}])
        check(f"{b}: no label_store registration contradiction", not contra, str(contra))
        check(f"{b}: eval-eligible iff unbiased", (split == "eval") == (biased is False),
              f"{(split, biased)}")

    # --- 2. no repeats anywhere in the union ----------------------------------
    emit("\n[no repeats / no calibration duplicates / no drift probes]")
    jk = Counter(ls.join_key(r["render"]) for _, r in pairs)
    dup = [k for k, v in jk.items() if v > 1]
    check("every union row is a distinct render identity (join_key)", not dup,
          f"{len(dup)} repeated")
    ids = Counter(r["image_id"] for _, r in pairs)
    check("source image_ids unique across the union",
          all(v == 1 for v in ids.values()),
          f"{sum(1 for v in ids.values() if v > 1)} repeated")
    # A calibration duplicate / drift probe is a row whose render identity ALREADY carries a
    # label elsewhere in the corpus. Scan every other batch, not just the three.
    elsewhere = defaultdict(list)
    for d in sorted(os.listdir(cc.BATCHES_DIR)):
        p = os.path.join(cc.BATCHES_DIR, d, "images.jsonl")
        if d in spec.sources or not os.path.exists(p):
            continue
        for r in cc.read_jsonl(p):
            elsewhere[ls.join_key(r["render"])].append(d)
    reused = {k: elsewhere[k] for k in jk if k in elsewhere}
    check("no union row repeats a render already in the corpus", not reused,
          f"{len(reused)} reused, e.g. {list(reused.items())[:2]}")
    check("every union label is still null",
          all(r["label"]["score"] is None for _, r in pairs))

    # --- 3. join_key round-trip ------------------------------------------------
    emit("\n[join_key integrity]")
    store = defaultdict(list)
    for b in spec.sources:
        for r in cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")):
            store[ls.join_key(r["render"])].append((b, r["image_id"]))
    bad = [k for _, r in pairs for k in [ls.join_key(r["render"])] if len(store[k]) != 1]
    check("every row round-trips to EXACTLY one store row", not bad, f"{len(bad)} ambiguous")

    # --- 4. crops on disk AND through the serve.py resolver --------------------
    emit("\n[crops — on disk and through the serve.py resolver]")
    import serve as _serve  # the live label-UI server: its translate_path IS the resolver
    handler = _serve.CorpusHandler.__new__(_serve.CorpusHandler)
    miss_disk, miss_serve = [], []
    for b, r in pairs:
        iid = r["image_id"]
        for kind, d in (("crops", cc.crops_dir(b)), ("vivid", cc.vivid_dir(b))):
            if not os.path.exists(os.path.join(d, f"{iid}.jpg")):
                miss_disk.append((b, kind, iid))
            # the exact URL path corpus_label.html builds, client-side
            url = f"/data/label_corpus/batches/{b}/{kind}/{iid}.jpg"
            if not os.path.exists(handler.translate_path(url)):
                miss_serve.append((b, kind, iid))
    check("every row has canonical crop AND vivid companion on disk", not miss_disk,
          f"{len(miss_disk)} missing")
    check("every crop resolves through serve.py translate_path", not miss_serve,
          f"{len(miss_serve)} unresolvable")

    # Phoenix explicitly: it needed a recovery join elsewhere, so its copies are named.
    ph = [(b, r) for b, r in pairs if family_of(r) == "phoenix"]
    ph_bad = sum(1 for b, r in ph
                 if not os.path.exists(os.path.join(cc.crops_dir(b), f"{r['image_id']}.jpg"))
                 or not os.path.exists(os.path.join(cc.vivid_dir(b), f"{r['image_id']}.jpg")))
    check("phoenix rows are whole (crop+vivid)", ph_bad == 0,
          f"{len(ph)} phoenix rows, {ph_bad} incomplete")
    by_src = Counter(b for b, _ in ph)
    emit(f"       phoenix by source: {dict(by_src)}")

    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    _write_log(L, "verify_sources.txt", spec)
    return 0 if ok else 1


def _write_log(lines, name, spec: SheetSpec = Q4_COMBINED):
    p = paths.scratch(f"{spec.name}_sheet", name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {p}")


# =========================================================================== #
# build
# =========================================================================== #
def _link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink the crop into the sheet's tree; copy only if the link is refused.

    A hardlink is the right primitive here: the sheet's crop IS the source batch's crop under
    a different name, and 1740 copies would be ~800 MB of duplicate bytes for a presentation
    alias. Falls back to a copy across a filesystem boundary (or a non-NTFS volume), where a
    link is impossible.
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def stage_build(args) -> int:
    spec = _spec(args)
    pairs = load_sources(spec)
    order = ordered_union(pairs, spec.seed)
    sd = sheet_dir(spec)

    sheet_rows, route = [], {}
    for slot, (cell, batch, row) in enumerate(order):
        oid = opaque_id(slot, batch, row["image_id"], spec)
        # BLINDING BY ABSENCE: id + render + null label. No provenance block at all.
        sheet_rows.append({"image_id": oid,
                           "render": row["render"],
                           "label": cc.label_block()})
        route[oid] = {"batch": batch, "image_id": row["image_id"]}

    cells = Counter(cell for cell, _, _ in order)
    bj = {
        "schema_version": 1,
        "batch_id": spec.sheet_id,
        "presentation_only": True,
        "purpose": spec.purpose,
        "source_batches": list(spec.sources),
        "served_manifest": SHEET_MANIFEST,
        "route_map": ROUTE_FILE,
        "presentation_seed": spec.seed,
        "presentation_order": "file",
        "order_rule": ("seeded shuffle over the union, then stratified round-robin over every "
                       "(source_batch x family) cell to +/-1 of proportional share in every "
                       "prefix"),
        "id_salt": spec.salt,
        "id_rule": (f"{spec.id_prefix}<slot>_<blake2b(salt|batch|image_id)>; slot assigned "
                    f"POST-shuffle"),
        "counts": {"total": len(sheet_rows),
                   "by_source": dict(Counter(b for _, b, _ in order)),
                   "cells": {f"{b}|{f}": n for (b, f), n in sorted(cells.items())}},
        "vivid_companion": "blue_orange",
        "calibration_aids": "NONE — no exemplars, no reference strip, no score shown",
    }

    if not args.apply:
        head = [f"{b[-20:]}|{f}" for (b, f), _, _ in order[:12]]
        print(f"DRY RUN — would write {len(sheet_rows)} rows to {sd}")
        print(f"  cells: {len(cells)}; by source: {bj['counts']['by_source']}")
        print(f"  first 12 cells in order: {head}")
        print("  pass --apply to write the sheet, the route map and the crop tree")
        return 0

    # A sheet dir must never be, or become, a real batch: an images.jsonl here is unioned by
    # every corpus consumer and would double-count 870 labels that already live in three
    # batches. Refuse rather than write beside it (and never delete it — if one is there, the
    # right question is which of the two is wrong).
    if (sd / "images.jsonl").exists():
        raise SystemExit(f"{sd / 'images.jsonl'} exists — this is a BATCH, not a sheet dir. "
                         f"Refusing to write a sheet into it.")

    sd.mkdir(parents=True, exist_ok=True)
    cc.write_jsonl(sheet_rows, str(sd / SHEET_MANIFEST))
    (sd / ROUTE_FILE).write_text(json.dumps(route, indent=0, sort_keys=True) + "\n",
                                 encoding="utf-8")
    (sd / "batch.json").write_text(json.dumps(bj, indent=2) + "\n", encoding="utf-8")

    crops_out = Path(cc.crops_dir(spec.sheet_id))
    vivid_out = Path(cc.vivid_dir(spec.sheet_id))
    crops_out.mkdir(parents=True, exist_ok=True)
    vivid_out.mkdir(parents=True, exist_ok=True)
    modes = Counter()
    for oid, ent in route.items():
        b, iid = ent["batch"], ent["image_id"]
        modes[_link_or_copy(Path(cc.crops_dir(b)) / f"{iid}.jpg", crops_out / f"{oid}.jpg")] += 1
        modes[_link_or_copy(Path(cc.vivid_dir(b)) / f"{iid}.jpg", vivid_out / f"{oid}.jpg")] += 1

    print(f"wrote {sd}")
    print(f"  {SHEET_MANIFEST}: {len(sheet_rows)} rows   {ROUTE_FILE}: {len(route)} ids")
    print(f"  crops/vivid: {dict(modes)} -> {crops_out}")
    return 0


# =========================================================================== #
# check — the BUILT bytes
# =========================================================================== #
def stage_check(args) -> int:
    spec = _spec(args)
    L, ok = [], True

    def emit(s=""):
        L.append(s)
        print(s)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    sd = sheet_dir(spec)
    emit(f"=== {spec.name} label sheet — BUILT-BYTES acceptance ({STAMP}) ===")
    emit(f"[{spec.sheet_id}]")
    if not (sd / SHEET_MANIFEST).exists():
        check("sheet built", False, str(sd))
        _write_log(L, "check_sheet.txt", spec)
        return 1

    served_text = (sd / SHEET_MANIFEST).read_text(encoding="utf-8")
    rows = cc.read_jsonl(str(sd / SHEET_MANIFEST))
    route = ms.load_route(str(sd / ROUTE_FILE))
    bj = json.loads((sd / "batch.json").read_text(encoding="utf-8"))
    src = {b: cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl"))
           for b in spec.sources}
    src_by_id = {b: {r["image_id"]: r for r in rs} for b, rs in src.items()}

    check("every source row is served exactly once",
          len(rows) == sum(len(v) for v in src.values()),
          f"{len(rows)} served vs {sum(len(v) for v in src.values())} in the sources")
    check("this directory is NOT a batch (no images.jsonl)",
          not (sd / "images.jsonl").exists())
    check("batch.json names the served manifest and the route map",
          bj.get("served_manifest") == SHEET_MANIFEST and bj.get("route_map") == ROUTE_FILE)
    check("batch.json pins presentation_order=file (the designed order is the file order)",
          bj.get("presentation_order") == "file")
    check("no calibration aids", str(bj.get("calibration_aids", "")).startswith("NONE"))

    # --- blindness, on the served BYTES ---------------------------------------
    emit("\n[blindness]")
    leaked = [k for k in SHEET_LEAK_KEYS if f'"{k}"' in served_text]
    check("no leak key appears anywhere in the served bytes", not leaked, str(leaked))
    check("no source batch_id appears in the served bytes",
          not any(b in served_text for b in spec.sources))
    check("served row is exactly {image_id, render, label}",
          all(set(r) == {"image_id", "render", "label"} for r in rows))
    # The render block is the ONE thing that must survive verbatim (it is what the crop is a
    # pure function of). So: the version-invariant core is present on every row, and the only
    # extras are the family constants the family registry declares — nothing else can ride in
    # the block that blinding did not inspect.
    allowed = set(cc.RENDER_KEYS) | {"fractal_type", "c_re", "c_im"}
    for extra in _loc().FAMILY_PARAM_KEYS.values():
        allowed |= set(extra)
    check("every render block carries the version-invariant key set",
          all(set(cc.RENDER_KEYS) <= set(r["render"]) for r in rows))
    check("render extras are only declared family constants",
          all(set(r["render"]) <= allowed for r in rows),
          str(sorted({k for r in rows for k in r["render"]} - allowed)))
    check("every served label is null", all(r["label"]["score"] is None for r in rows))
    p = len(spec.id_prefix)
    check(f"ids are opaque `{spec.id_prefix}<slot>_<hash>` and unique",
          len({r["image_id"] for r in rows}) == len(rows)
          and all(r["image_id"].startswith(spec.id_prefix)
                  and len(r["image_id"]) == p + 13 for r in rows))
    check("no served id contains a source image_id",
          not any(r["image_id"][p + 5:] in {i for m in src_by_id.values() for i in m}
                  for r in rows))
    # the id is assigned POST-shuffle: slot order == file order, and nothing else.
    check("slot numbering follows the served order",
          all(int(r["image_id"][p:p + 4]) == i for i, r in enumerate(rows)))

    # --- the union is faithful -------------------------------------------------
    emit("\n[union fidelity]")
    check("route covers every served row exactly once",
          set(route) == {r["image_id"] for r in rows} and len(route) == len(rows))
    per_src = Counter(b for b, _ in route.values())
    check("every source row appears exactly once",
          all(per_src[b] == len(src[b]) for b in spec.sources), str(dict(per_src)))
    mismatch = [r["image_id"] for r in rows
                if src_by_id[route[r["image_id"]][0]][route[r["image_id"]][1]]["render"]
                != r["render"]]
    check("served render block is byte-identical to its store row", not mismatch,
          f"{len(mismatch)} differ")
    jk = Counter(ls.join_key(r["render"]) for r in rows)
    check("no repeat row in the sheet", all(v == 1 for v in jk.values()))

    # --- the ORDER -------------------------------------------------------------
    emit("\n[stratified order]")
    cells = [(route[r["image_id"]][0],
              family_of(src_by_id[route[r["image_id"]][0]][route[r["image_id"]][1]]))
             for r in rows]
    n_c = Counter(cells)
    N = len(cells)
    worst, worst_at = 0.0, None
    seen = Counter()
    for L_i, c in enumerate(cells, start=1):
        seen[c] += 1
        for cell, n in n_c.items():
            dev = abs(seen[cell] - L_i * n / N)
            if dev > worst:
                worst, worst_at = dev, (cell, L_i)
    check("every cell is within +/-1 of proportional share in EVERY prefix", worst <= 1.0,
          f"max deviation {worst:.3f} at {worst_at}")
    runs_src = _max_run([c[0] for c in cells])
    runs_fam = _max_run([c[1] for c in cells])
    # Run-length caps, one step above the MEASURED spread FOR THIS SPEC. Over six union seeds
    # the q4 deal gave same-source run 3 every time and same-family run 3-4 (the cell SEQUENCE
    # is near seed-independent — the shuffle permutes rows within a cell, not the cells
    # themselves), so 4/5 leave headroom for a reseed without going slack. The load-bearing
    # balance invariant is the +/-1 prefix bound above; these two are the "no stretch is
    # single-X" statement itself, and a one-source sheet satisfies the first vacuously.
    check("no stretch of the sitting is single-source", runs_src <= spec.max_run_source,
          f"longest run {runs_src} (cap {spec.max_run_source})")
    check("no stretch of the sitting is single-family", runs_fam <= spec.max_run_family,
          f"longest run {runs_fam} (cap {spec.max_run_family})")
    emit(f"       {len(n_c)} cells; longest same-source run {runs_src}, "
         f"same-family run {runs_fam}")

    # --- crops, through the resolver the browser actually hits -----------------
    emit("\n[crops through the serve.py resolver]")
    import serve as _serve
    handler = _serve.CorpusHandler.__new__(_serve.CorpusHandler)
    miss = []
    for r in rows:
        for kind in ("crops", "vivid"):
            url = f"/data/label_corpus/batches/{spec.sheet_id}/{kind}/{r['image_id']}.jpg"
            if not os.path.exists(handler.translate_path(url)):
                miss.append((kind, r["image_id"]))
    check("every sheet row serves a canonical crop AND a vivid companion", not miss,
          f"{len(miss)} missing")
    # the alias must be the same bytes as the source crop, not a stale or wrong-row copy.
    sample = rows[::97]
    bad = []
    for r in sample:
        b, iid = route[r["image_id"]]
        for kind, d_s, d_b in (("crops", cc.crops_dir(spec.sheet_id), cc.crops_dir(b)),
                               ("vivid", cc.vivid_dir(spec.sheet_id), cc.vivid_dir(b))):
            a = Path(d_s) / f"{r['image_id']}.jpg"
            c = Path(d_b) / f"{iid}.jpg"
            if a.read_bytes() != c.read_bytes():
                bad.append((kind, r["image_id"]))
    check(f"sheet crop bytes == source crop bytes ({len(sample)} sampled rows, both trees)",
          not bad, str(bad[:3]))

    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    _write_log(L, "check_sheet.txt", spec)
    return 0 if ok else 1


def _max_run(seq) -> int:
    """Longest run of equal consecutive values — the "no stretch is single-X" measure."""
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


# =========================================================================== #
# merge-dryrun — a synthesized FULL export, merged in a sandbox
# =========================================================================== #
def stage_merge_dryrun(args) -> int:
    """Prove the one-file export routes: synthesize a FULL labels.json for the sheet, merge it
    into a COPY of the corpus, and assert per-batch counts and an untouched amendment
    overlay."""
    import tempfile
    spec = _spec(args)
    L, ok = [], True

    def emit(s=""):
        L.append(s)
        print(s)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    sd = sheet_dir(spec)
    route = ms.load_route(str(sd / ROUTE_FILE))
    rows = cc.read_jsonl(str(sd / SHEET_MANIFEST))
    emit(f"=== {spec.name} export -> {len(spec.sources)} batch(es) — SANDBOX dry run "
         f"({STAMP}) ===")

    # A full sitting: every row scored, cycling 1..4 so all four tiers exercise the path.
    export = {r["image_id"]: {"score": 1 + (i % 4), "revealed": 0}
              for i, r in enumerate(rows)}
    emit(f"synthesized export: {len(export)} scores, tiers "
         f"{dict(Counter(v['score'] for v in export.values()))}")

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td) / "batches"
        sandbox.mkdir(parents=True)
        for b in spec.sources:
            (sandbox / b).mkdir()
            shutil.copy2(os.path.join(cc.batch_dir(b), "images.jsonl"), sandbox / b / "images.jsonl")
        amend_before = _amendment_digest()

        exp_path = Path(td) / "labels.json"
        exp_path.write_text(json.dumps(export), encoding="utf-8")

        per_batch = defaultdict(dict)
        for oid, score in ms.load_scores(str(exp_path)).items():
            b, iid = route[oid]
            per_batch[b][iid] = score
        check("every exported id routes to a registered source batch",
              set(per_batch) == set(spec.sources), str(sorted(per_batch)))

        stats = {b: ms.merge_batch(b, str(sandbox), per_batch[b], labeler="dryrun",
                                   labeled_at=STAMP, max_score=4, apply=True)
                 for b in spec.sources}
        for b in spec.sources:
            n_src = len(cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")))
            st = stats[b]
            check(f"{b}: all {n_src} rows filled",
                  st["filled"] == n_src and st["labeled"] == n_src,
                  f"filled={st['filled']} labeled={st['labeled']}/{st['n_rows']}")
            check(f"{b}: no conflict, no unknown id, nothing out of range",
                  not st["conflicts"] and not st["unknown"] and not st["out_of_range"])
        check("routed row counts sum to the export",
              sum(s["filled"] for s in stats.values()) == len(export),
              f"{sum(s['filled'] for s in stats.values())} vs {len(export)}")

        # Per-row placement, not just counts: a sampled row must land in ITS batch with ITS score.
        placed = []
        merged = {b: {r["image_id"]: r for r in cc.read_jsonl(str(sandbox / b / "images.jsonl"))}
                  for b in spec.sources}
        for oid in list(route)[::53]:
            b, iid = route[oid]
            if merged[b][iid]["label"]["score"] != export[oid]["score"]:
                placed.append(oid)
        check(f"sampled rows carry their own score in their own batch ({len(list(route)[::53])})",
              not placed, str(placed[:3]))

        # The real store must be untouched by a sandbox merge, and so must the amendments.
        live_labeled = sum(1 for b in spec.sources
                           for r in cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl"))
                           if r["label"]["score"] is not None)
        check("the LIVE store is untouched (still 0 labels)", live_labeled == 0,
              f"{live_labeled} labeled")
        check("the amendment overlay is untouched", _amendment_digest() == amend_before)

        # max-score: a 5 must be refused as out of range, never written.
        bad_b, bad_i = route[rows[0]["image_id"]]
        st5 = ms.merge_batch(bad_b, str(sandbox), {bad_i: 5}, labeler="dryrun",
                             labeled_at=STAMP, max_score=4, apply=True)
        check("--max-score 4 refuses a tier-5 score", st5["out_of_range"] and not st5["wrote"])
        # and a CHANGE to a now-non-null label is refused, not clobbered.
        other = 1 + (export[rows[0]["image_id"]]["score"] % 4)
        st_c = ms.merge_batch(bad_b, str(sandbox), {bad_i: other}, labeler="dryrun",
                              labeled_at=STAMP, max_score=4, apply=True)
        check("a re-merge that would CHANGE a non-null label is refused",
              len(st_c["conflicts"]) == 1 and not st_c["wrote"])

    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    _write_log(L, "merge_dryrun.txt", spec)
    return 0 if ok else 1


def _amendment_digest() -> str:
    """Digest of every registered amendment file — the overlay a merge must never touch."""
    h = hashlib.blake2b(digest_size=16)
    for b, fn in sorted(ls.AMENDMENT_LABELS.items()):
        p = Path(ls.LABELS_DIR) / fn
        h.update(f"{b}|{fn}|".encode())
        h.update(p.read_bytes() if p.exists() else b"<absent>")
    return h.hexdigest()


# =========================================================================== #
def _spec(args) -> SheetSpec:
    """The sheet this invocation is about. Named, never positional — a sheet built against the
    wrong spec would write a correct-looking sitting over somebody else's directory."""
    return SPECS[getattr(args, "spec", None) or Q4_COMBINED.name]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add(name, fn):
        p = sub.add_parser(name)
        p.add_argument("--spec", choices=sorted(SPECS), default=Q4_COMBINED.name)
        p.set_defaults(fn=fn)
        return p

    _add("verify", stage_verify)
    _add("build", stage_build).add_argument("--apply", action="store_true",
                                            help="write (default: dry run)")
    _add("check", stage_check)
    _add("merge-dryrun", stage_merge_dryrun)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
