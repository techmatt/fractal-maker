r"""build_blind_minibrot_sheet.py — SHEET D: the blind minibrot/maneuver EVAL slice.

THE INSTRUMENT SHEET A CANNOT BE (`prompts/settlement_28b.md` §2, protocol §2b). Sheet A's
960 rows were served with wallpaper v3's tier PREFILLED and ordered by its continuous score;
84.9% of its labels came back equal to the served suggestion, so v3 reads AUC>=3 0.965 on its
minibrot bucket against 0.746/0.750 on the two batches whose labels predate v3. The (28)
retrain put that bucket on the eval side as v4b's MOTIVATING arm, which no from-scratch head
can win. This sheet buys the same population again with the anchoring removed.

FOUR PROPERTIES, and each is a rule the builder enforces rather than a hope:

  1. FRESH LOCATIONS ONLY. Every location in EVERY prior wallpaper batch is excluded — by
     exact location key AND by the head-corpus proximity guard (`build_fresh_discovery.
     _spatially_in`, DEDUP_FRAC 0.5 of min(fw), c-identity aware). The batch list is GLOBBED
     off `data/wallpaper_corpus/batches/`, never a constant: a hardcoded list is how batch
     eight silently stops being excluded.

  2. NEITHER WALLPAPER HEAD TOUCHES THE DRAW OR THE SUBSTRATE. Location quality is
     conditioned through the LOCATION head only — `floors.passes_good_floor` on the intake's
     own `p_good`, which `ledger_rescore` keeps current under `production_pins.ACTIVE_CKPT`
     (v11). Palette comes from the production colorize path's pref proposal (pref-v3-gvo, a
     PALETTE-preference head, not a wallpaper-quality head). No v3 or v4b score is read,
     computed, stamped or sorted on anywhere in this file — asserted by
     `test_blind_minibrot_sheet.py`, which fails on the symbol.

  3. BLIND SERVING. No `suggested_tier`, no `head_v3` block, no flat `p_ge3`/`pred` — so
     `wallpaper_label.html` cannot enter correction mode and has nothing to sort by. Order is
     a SEEDED SHUFFLE stamped into `sheet_order` at build time and served with `&order=file`,
     so the order is reproducible from the artifact instead of re-derived in a browser.

  4. EVAL-ONLY, PERMANENTLY. Every row stamps `split_side="eval"` and the batch stamps
     `eval_only: true` with the reason. A blind slice bought to referee two heads is spent
     the moment it enters a train split.

THE COLORING IS THE LIVE EMISSION PATH, imported from the batch that first ran it
(`build_colorize_sheet`, 2026-08-05) rather than restated: morph cluster (production intake
descriptor, library-seeded, cos 0.974) -> deficit-assigned palette flavor
(`cells.choose_option` against the `release_mix` target) -> pref-v3-gvo argmax within that
flavor -> `deploy_tail._color_params({})` canonical params, smooth. The CANVAS is the
corpus's shared label-crop pins (1280x720 ss2 lanczos3 q90), not the emission pool's 960x540,
so these rows can be read beside every other wallpaper batch.

    uv run python -u tools/wallpaper/build_blind_minibrot_sheet.py select
    uv run python -u tools/wallpaper/build_blind_minibrot_sheet.py embed  --limit 4   # smoke
    uv run python -u tools/wallpaper/build_blind_minibrot_sheet.py embed  > scratch/blind_minibrot/embed.log 2>&1
    uv run python -u tools/wallpaper/build_blind_minibrot_sheet.py render --limit 4   # bounded E2E
    uv run python -u tools/wallpaper/build_blind_minibrot_sheet.py render > scratch/blind_minibrot/render.log 2>&1
    uv run python -u tools/wallpaper/build_blind_minibrot_sheet.py write
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "queries", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "atlas", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                              # noqa: E402  THE apportionment rules
import colormap as cm                         # noqa: E402
import location as loc_mod                    # noqa: E402
import production_seeder as ps                # noqa: E402  THE calibrated near-dup rule
import release_mix as RM                      # noqa: E402  THE release-mix ratio table
from label_crop import (                      # noqa: E402  THE shared label-crop pins
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q, ensure_label_field)
from label_crop import render_label_crop      # noqa: E402
from tools.emission import cells as C         # noqa: E402
from tools.emission import descriptor as D    # noqa: E402
from tools.emission import floors as F        # noqa: E402  THE location-head cut owner
from tools.emission import library_seed_v2 as LSEED   # noqa: E402
# The (27) sitting owns the intake population and the minibrot/maneuver VEIN definition.
# Imported, never restated: a second copy of `vein_of` is a second answer to "is this row
# minibrot-centred", and the whole point of sheet D is that it covers sheet A's population.
from tools.wallpaper import build_wallpaper_sitting as BWS   # noqa: E402
from tools.wallpaper.build_fresh_discovery import _spatially_in   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

WALLPAPER_CORPUS = ROOT / "data" / "wallpaper_corpus"
RENDER_STYLES = ("smooth",)      # the wallpaper head judges smooth; strange modes route to mining

# THE ROW SHAPE, declared rather than emergent, and asserted at write time. This is what makes
# "the sheet is blind" a checkable property instead of a claim about a builder's code: the
# labeling rig enters correction mode iff a row carries a numeric `suggested_tier`, and shows
# a machine readout iff a row carries `head_v2_pred` / `pred` / `p_ge3`. None of those is in
# this tuple, and `test_blind_minibrot_sheet.py` fails if one is added.
ROW_KEYS = ("image_id", "sheet_order", "render", "provenance", "label")


# =========================================================================== #
# The sheet spec — a frozen dataclass from the start (CLAUDE.md, "Writing a builder for one
# instance"), so a sheet E is an entry rather than a refactor.
# =========================================================================== #
@dataclass(frozen=True)
class SheetSpec:
    key: str
    batch_id: str
    generator_version: str
    img_prefix: str
    target_rows: int
    draw_seed: int
    shuffle_seed: int
    cell_seed: int

    @property
    def batch_dir(self) -> Path:
        return WALLPAPER_CORPUS / "batches" / self.batch_id

    @property
    def work(self) -> Path:
        return ROOT / "scratch" / "blind_minibrot" / self.key

    @property
    def labels_export(self) -> str:
        return f"labels/{self.generator_version}.json"

    @property
    def ui_url(self) -> str:
        # `order=file` honours the builder's stamped shuffle; `tiers=4` is the wallpaper
        # scale. No `&correction` knob exists — correction mode is entered by the rows
        # carrying `suggested_tier`, which these do not.
        return (f"tools/viz/wallpaper_label.html?corpus=wallpaper_corpus&tiers=4"
                f"&order=file&batch={self.batch_id}")


SHEETS = {
    "d": SheetSpec(
        key="d",
        batch_id="2026-08-11_wallpaper_blind_minibrot_v1",
        generator_version="wallpaper_blind_minibrot_v1",
        img_prefix="bmb",
        target_rows=200,
        draw_seed=28,
        shuffle_seed=20260811,
        cell_seed=28,
    ),
}


def log(msg: str):
    print(msg, flush=True)


def _safe(unit_key: str) -> str:
    return unit_key.replace(":", "__").replace("/", "_")


# =========================================================================== #
# 1. Population — intake ∩ minibrot/maneuver ∩ fresh ∩ location-head-good, near-dup filtered.
# =========================================================================== #
def prior_wallpaper_locations(exclude_batch: str | None = None) -> tuple[set, dict, dict]:
    """`(location_keys, per_family_coords, per_batch_counts)` over every OTHER wallpaper batch.

    Globbed, not listed. `build_fresh_discovery._head_corpus_exclusion` does the same job off
    a hardcoded `HEAD_CORPUS_BATCHES`, which predates three of the seven batches on disk — the
    exact failure mode a constant has here, since a sheet that silently stops excluding batch
    eight is a sheet whose "fresh locations only" claim is false and invisible.

    `exclude_batch` IS LOAD-BEARING AND THE GLOB IS WHY. Once this sheet has written its own
    `images.jsonl` it is one of the batches under `batches/`, so a scan that does not skip it
    excludes all 197 of its own locations and the population goes to zero. The observed
    failure was ALTERNATING, which is the worst shape it could have taken: the empty run
    still rewrote `images.jsonl` (empty), which made the NEXT run's scan find nothing to
    exclude and succeed, which made the one after that fail again. Callers pass the spec's
    own batch id; `None` means "no batch is being built", which is what an audit wants."""
    keys, coords, per_batch = set(), defaultdict(list), {}
    root = WALLPAPER_CORPUS / "batches"
    for bdir in sorted(root.iterdir()) if root.exists() else []:
        if exclude_batch is not None and bdir.name == exclude_batch:
            continue
        p = bdir / "images.jsonl"
        if not p.exists():
            continue
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loc = loc_mod.from_render_block(json.loads(line)["render"])
            keys.add(loc.key())
            n += 1
            try:
                coords[loc.family].append((float(loc.cx), float(loc.cy), float(loc.fw),
                                           loc.c_re, loc.c_im))
            except (TypeError, ValueError):
                pass
        per_batch[bdir.name] = n
    return keys, coords, per_batch


def _c_vector(loc):
    """The row's parameter-space identity for `near_dup` (None on a c-plane family)."""
    if loc.c_re is None:
        return None
    try:
        return (float(loc.c_re), float(loc.c_im))
    except (TypeError, ValueError):
        return None


def near_dup_filter(srcs: list) -> tuple[list, list, dict]:
    """`(kept, groups, report)` — one row per near-dup group, first-in-order wins.

    THE rule is the seeder's, read from its owner at call time (`production_seeder.near_dup`
    under the live calibrated `DEDUP_K`/`DEDUP_SCALE`), so a recalibration moves this filter
    too. Order is by `unit_key`, so the representative is a pure function of the population
    and never of a score.

    This is a DIFFERENT rule from the prior-batch exclusion above, deliberately: exclusion
    asks "is this already in the corpus" (the head-corpus proximity guard, DEDUP_FRAC 0.5),
    and this asks "are these two rows the same place" (cloud hygiene, DEDUP_K 0.25). Each is
    read from the module that owns it rather than one being made to do both jobs."""
    ordered = sorted(srcs, key=lambda s: s["unit_key"])
    kept, groups = [], []
    for s in ordered:
        l = s["loc"]
        hit = None
        for i, k in enumerate(kept):
            lk = k["loc"]
            if lk.family != l.family:
                continue
            try:
                same = ps.near_dup(float(l.cx), float(l.cy), float(l.fw),
                                   float(lk.cx), float(lk.cy), float(lk.fw),
                                   ps.DEDUP_K, a_c=_c_vector(l), b_c=_c_vector(lk),
                                   scale=ps.DEDUP_SCALE)
            except (TypeError, ValueError):
                same = False
            if same:
                hit = i
                break
        if hit is None:
            kept.append(s)
            groups.append([s["unit_key"]])
        else:
            groups[hit].append(s["unit_key"])
    multi = [g for g in groups if len(g) > 1]
    return kept, groups, {
        "rule": f"production_seeder.near_dup(k={ps.DEDUP_K}, scale={ps.DEDUP_SCALE!r}, "
                f"c_eps={ps.JULIA_SAME_C_EPS}) — read from the owner at call time",
        "n_in": len(ordered), "n_kept": len(kept), "n_dropped": len(ordered) - len(kept),
        "n_multi_row_groups": len(multi),
        "representative": "first by unit_key — a pure function of the population, never a score",
    }


def population(spec: SheetSpec) -> tuple[list, dict]:
    """`(eligible sources, report)` — the four filters, in order, each counted."""
    srcs, pop_report = BWS.population()
    minibrot = [s for s in srcs if s["vein"] in BWS.MINIBROT_VEINS]

    keys, coords, per_batch = prior_wallpaper_locations(exclude_batch=spec.batch_id)
    n_key = n_spatial = 0
    fresh = []
    for s in minibrot:
        if s["key"] in keys:
            n_key += 1
            continue
        if _spatially_in(s["loc"], coords):
            n_spatial += 1
            continue
        fresh.append(s)

    good = [s for s in fresh if F.passes_good_floor(s["source_p_good"])]
    kept, groups, dedup_rep = near_dup_filter(good)

    report = {
        "intake": {
            "source": pop_report["source"], "ledgers": pop_report["ledgers"],
            "n_population": pop_report["n_population"],
            "by_vein": pop_report["by_vein"],
        },
        "filters": [
            {"filter": "vein in MINIBROT_VEINS",
             "rule": "build_wallpaper_sitting.MINIBROT_VEINS = "
                     + "/".join(sorted(BWS.MINIBROT_VEINS))
                     + " — the SAME definition sheet A's motivating bucket was drawn under",
             "in": pop_report["n_population"], "out": len(minibrot)},
            {"filter": "fresh locations only",
             "rule": "excluded if the location key appears in ANY wallpaper batch, or if it "
                     "is within DEDUP_FRAC*min(fw) of one at a matching c "
                     "(build_fresh_discovery._spatially_in)",
             "in": len(minibrot), "out": len(fresh),
             "excluded_by_key": n_key, "excluded_by_proximity": n_spatial,
             "prior_batches": per_batch, "n_prior_location_keys": len(keys)},
            {"filter": "LOCATION-head quality",
             "rule": f"floors.passes_good_floor(p_good) — p_good is the intake ledger's "
                     f"LOCATION-head P(>=3), kept current by ledger_rescore under the live "
                     f"ACTIVE_CKPT. Floor {F.GOOD_FLOOR:g}. NEITHER wallpaper head is read.",
             "in": len(fresh), "out": len(good)},
            {"filter": "near-dup", "rule": dedup_rep["rule"],
             "in": len(good), "out": len(kept),
             "n_multi_row_groups": dedup_rep["n_multi_row_groups"]},
        ],
        "near_dup": dedup_rep,
        "n_eligible": len(kept),
        "eligible_by_partition": dict(sorted(Counter(s["partition"] for s in kept).items())),
        "eligible_by_vein": dict(sorted(Counter(s["vein"] for s in kept).items())),
        "eligible_by_source_tag": dict(Counter(s["source_tag"] for s in kept).most_common()),
    }
    return kept, report


def draw(spec: SheetSpec, eligible: list, target_rows=None, seed=None) -> tuple[list, dict]:
    """`(selected, report)` — the whole eligible set when it fits, else a balanced draw.

    NO SCORE ORDERS THIS. When supply exceeds the target the cut is `deal_round_robin` over
    partitions (balanced-or-drained) with a seeded shuffle inside each cell, which is a
    function of the population and the seed and of nothing else. Taking a "top slice" of any
    score would reintroduce exactly the conditioning this sheet exists without."""
    target_rows = int(target_rows or spec.target_rows)
    seed = int(spec.draw_seed if seed is None else seed)
    if len(eligible) <= target_rows:
        selected = sorted(eligible, key=lambda s: s["unit_key"])
        rep = {"rule": "TAKE ALL — supply is at or under the target, so there is no draw and "
                       "no rule to bias",
               "target_rows": target_rows, "eligible": len(eligible),
               "drawn_rows": len(selected), "supply_bound": True, "seed": seed,
               "partition_alloc": dict(sorted(Counter(s["partition"] for s in selected).items()))}
        return selected, rep
    rng = np.random.default_rng([seed, 2])
    cells = defaultdict(list)
    for s in eligible:
        cells[s["partition"]].append(s)
    sizes = {k: len(v) for k, v in sorted(cells.items())}
    take = apportion.deal_round_robin(sizes, target_rows)
    selected = []
    for k in sorted(cells):
        members = sorted(cells[k], key=lambda s: s["unit_key"])
        rng.shuffle(members)
        selected.extend(members[:take[k]])
    selected.sort(key=lambda s: s["unit_key"])
    rep = {"rule": "apportion.deal_round_robin over partitions (balanced-or-drained), "
                   "seeded shuffle inside each cell. NO score orders this draw.",
           "target_rows": target_rows, "eligible": len(eligible),
           "drawn_rows": len(selected), "supply_bound": False, "seed": seed,
           "partition_available": sizes, "partition_alloc": take}
    return selected, rep


def selection(spec: SheetSpec, target_rows=None, seed=None):
    """`(selected, pop_report, draw_report)` — the whole CPU-only front half, deterministic."""
    eligible, pop_report = population(spec)
    selected, draw_report = draw(spec, eligible, target_rows, seed)
    return selected, pop_report, draw_report


# =========================================================================== #
# 2. Embed — the production intake descriptor (retained 640x360 ss2 field + morph CLIP).
# =========================================================================== #
def _emb_path(spec: SheetSpec, unit_key: str) -> Path:
    return spec.work / "embs" / f"{_safe(unit_key)}.npy"


def _field_cache(spec: SheetSpec) -> Path:
    return spec.work / "intake_fields"


def run_embed(spec: SheetSpec, args):
    """One retained 640x360 ss2 smooth field + one CLIP morph embedding per drawn location.

    Checkpointed per location (atomic `.npy`), so a kill loses at most the in-flight row."""
    from tools.curation.colored_clip import embed_clip, load_clip
    from tools.wallpaper import library_annotate as la

    selected, _pop, _dr = selection(spec, args.target_rows, args.seed)
    cache = _field_cache(spec)
    cache.mkdir(parents=True, exist_ok=True)
    (spec.work / "embs").mkdir(parents=True, exist_ok=True)
    todo = [s for s in selected if not _emb_path(spec, s["unit_key"]).exists()]
    n_done = len(selected) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    log(f"[embed] {len(selected)} drawn · {n_done} embedded · {len(todo)} to run")
    if not todo:
        return
    model, tf = load_clip()
    t0, times = time.time(), []
    for i, s in enumerate(todo):
        t = time.time()
        loc = s["loc"]
        field = la.ensure_field(loc, retain=True, tmp_dir=cache, cache_root=cache)
        emb = embed_clip(model, tf, [la.morph_gray_image(field)])[0].astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        p = _emb_path(spec, s["unit_key"])
        tmp = p.with_name(p.name + ".tmp")
        # `np.save(path, ...)` appends ".npy" to a path that lacks it; an open file object
        # writes exactly here (the 2026-08-05 sibling's bug, not re-earned).
        with open(tmp, "wb") as fh:
            np.save(fh, emb)
        os.replace(tmp, p)
        times.append(time.time() - t)
        if (i + 1) % 20 == 0 or i < 2:
            recent = float(np.mean(times[-20:]))
            log(f"[embed] {i+1}/{len(todo)} {loc.family:16} [{times[-1]:.1f}s] recent "
                f"{recent:.1f}s/loc -> eta {recent*(len(todo)-i-1)/60:.0f} min "
                f"(elapsed {(time.time()-t0)/60:.0f} min)")
    log(f"[embed] done in {(time.time()-t0)/60:.1f} min")


def cluster(spec: SheetSpec, selected: list) -> tuple[dict, dict]:
    """`unit_key -> "<partition>#<k>"`, WITHIN the base partition, seeded from the library.

    `descriptor.cluster_incremental` verbatim (same 0.974 knee, same frozen-medoid rule).
    A missing embedding is a hard stop, not a skipped location: an unclustered row would be
    assigned a cell nothing has a target for."""
    from partitions import base_partition
    embs, missing = {}, []
    for s in selected:
        p = _emb_path(spec, s["unit_key"])
        if not p.exists():
            missing.append(s["unit_key"])
            continue
        embs[s["unit_key"]] = np.load(p).astype(np.float32).reshape(-1)
    if missing:
        raise SystemExit(f"[cluster] {len(missing)} drawn locations have no embedding "
                         f"(e.g. {missing[:3]}) — run `embed` to completion first.")
    seed = D.library_medoids(LSEED.INTAKE_JSON, LSEED.EMB_DIR)
    by_group = defaultdict(list)
    for s in selected:
        by_group[base_partition(s["partition"])].append(s)
    tags, n_seeded_joins = {}, 0
    for group, rows in sorted(by_group.items()):
        items = [(s["unit_key"], embs[s["unit_key"]])
                 for s in sorted(rows, key=lambda r: r["unit_key"])]
        seed_keys = {k for k, _e in (seed.get(group) or [])}
        assign = D.cluster_incremental(items, D.NEAR_DUP_THRESHOLD,
                                       seed_medoids=seed.get(group))
        by_key = {s["unit_key"]: s for s in rows}
        for uk, k in assign.items():
            tags[uk] = f"{by_key[uk]['partition']}#{k}"
            if k in seed_keys:
                n_seeded_joins += 1
    return tags, {"library_seed_groups": {g: len(v) for g, v in sorted(seed.items())},
                  "seeded_joins": n_seeded_joins,
                  "distinct_clusters": len(set(tags.values())),
                  "threshold": D.NEAR_DUP_THRESHOLD}


# =========================================================================== #
# 3. Render — deficit cell -> pref palette -> canonical params -> label crop.
# =========================================================================== #
def _ledger_path(spec: SheetSpec) -> Path:
    return spec.batch_dir / "_progress_ledger.jsonl"


def load_ledger(spec: SheetSpec) -> dict:
    done, p = {}, _ledger_path(spec)
    crops = spec.batch_dir / "crops"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if (crops / f"{rec['crop_stem']}.jpg").exists():
                    done[rec["unit_key"]] = rec
    return done


def build_deficit(selected, tags, cell_to_names, lib):
    """The live emission driver's grid, built the way it builds it."""
    flavors = sorted(f for f, names in cell_to_names.items()
                     if any(p in lib.colormaps for p in names))
    observed = sorted({(s["partition"], tags[s["unit_key"]]) for s in selected})
    feasible = C.build_feasible_cells(observed, flavors, RENDER_STYLES)
    shares = RM.shares(sorted({p for (p, _c) in observed}))
    target = C.TargetMeasure.from_partition_shares(shares, feasible)
    model = C.DeficitModel(feasible, target)
    return model, flavors, {
        "flavors": len(flavors), "styles": list(RENDER_STYLES),
        "observed_type_cluster": len(observed), "feasible_cells": len(feasible),
        "release_shares": {k: round(v, 5) for k, v in sorted(target.shares.items())},
        "attempt_cap": target.attempt_cap, "softmax_temp": target.softmax_temp,
        "target_source": "tools/scoring/release_mix.RATIO re-solved against the live feasible "
                         "cells (cells.TargetMeasure.from_partition_shares)",
    }


def render_block(loc, palette) -> dict:
    blk = {
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
        "fractal_type": loc.family, "c_re": loc.c_re, "c_im": loc.c_im,
        "palette": palette, "composition": "center",
        "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
        "filter": LABEL_FILTER, "interior_mode": "black",
    }
    for k, v in loc.params.items():
        blk[k] = v
    return blk


def run_render(spec: SheetSpec, args):
    """Colorize + render every drawn location. NO head scores the result.

    The one deliberate difference from `build_colorize_sheet`'s render stage, and it is
    forced: that stage read the wallpaper head's `p_ge3` off each finished crop to decide
    whether a deficit cell FILLED. Reading it here would put a wallpaper head into this
    sheet's substrate. So a cell fills on ATTEMPT — recorded as a named deviation in
    `batch.json.colorize_path.deficit_fill_rule`, not silently."""
    import importlib.util

    from tools.mining import deploy_tail as dt
    from tools.studies import conditioned_colorize as cond
    from tools.wallpaper import library_annotate as la
    from tools.wallpaper import library_store as store

    # The live colorize driver, by file path: `tools/emission/` is a package but the driver is
    # a script inside it, so a plain package import executes its body under an unexpected name.
    _spec = importlib.util.spec_from_file_location(
        "build_emission_diversity_v1",
        ROOT / "tools" / "emission" / "build_emission_diversity_v1.py")
    EM = importlib.util.module_from_spec(_spec)
    sys.modules["build_emission_diversity_v1"] = EM
    _spec.loader.exec_module(EM)

    selected, pop_report, draw_report = selection(spec, args.target_rows, args.seed)
    print_composition(pop_report, draw_report)
    if args.dry_run:
        return

    tags, cluster_rep = cluster(spec, selected)
    import query_sampler as qs
    lib = qs.load_pool_library()
    _name_to_cell, cell_to_names = cond.load_cell_map()
    model, flavors, grid_rep = build_deficit(selected, tags, cell_to_names, lib)
    ranker = EM.PaletteRanker(dt, cell_to_names, lib, pick_mode="pref")
    cparams = dt._color_params({})            # the canonical inherited coloring

    log(f"[render] {len(selected)} locations · {grid_rep['flavors']} flavors × "
        f"{list(RENDER_STYLES)} · {grid_rep['feasible_cells']} feasible cells · "
        f"{cluster_rep['distinct_clusters']} morph clusters "
        f"({cluster_rep['seeded_joins']} joined a library cluster)")
    log(f"[render] palette ranker = {ranker.mode} (pref-v3-gvo — a PALETTE head; no "
        f"wallpaper-quality head is loaded anywhere in this run)")

    # Resume: replay the ledger into the deficit model so the flavor sequence continues.
    done = load_ledger(spec)
    for rec in done.values():
        cell = tuple(rec["cell"])
        model.record_attempt(cell)
        model.record_fill(cell)

    crops = spec.batch_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    fields_dir = spec.work / "render_fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    cache = _field_cache(spec)

    todo = [s for s in selected if s["unit_key"] not in done]
    if args.limit:
        # SPREAD across the plan, not a prefix: the draw is unit_key-major, so a prefix
        # exercises one family's render path and calls it an end-to-end.
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
        log(f"[render] --limit {args.limit}: SPREAD -> partitions "
            f"{sorted({s['partition'] for s in todo})}")
    log(f"[render] {len(done)} in the ledger, {len(todo)} to run")

    failures, times = [], []
    t_wall = time.time()
    for i, s in enumerate(todo):
        t0 = time.time()
        uk, loc = s["unit_key"], s["loc"]
        ftype, clus = s["partition"], tags[uk]
        # PER-LOCATION rng, not a shared stream. `choose_option`'s softmax tie-break consumes
        # a draw whose width depends on how many options that cell had, so a shared stream
        # cannot be re-aligned on resume by counting completed rows — the 2026-08-05 sibling
        # advances it with `rng.random()` per done row, which is an approximation nobody can
        # check. Seeding off the unit key makes a resumed run byte-identical to an
        # uninterrupted one, and the flavor sequence still varies across locations.
        rng = np.random.default_rng([spec.cell_seed, _unit_seed(uk)])
        choice = C.choose_option(model, ftype, clus, flavors, RENDER_STYLES, rng)
        if choice is None:
            failures.append({"unit_key": uk, "stage": "cell", "error": "all cells capped"})
            continue
        flavor, style, deficit, n_opts, _p = choice

        stem = store.field_stem(loc, "smooth", la.W, la.H, la.SS)
        fbin, fjson = str(cache / f"{stem}.bin"), str(cache / f"{stem}.json")
        try:
            palette, pref_fit = ranker.best(uk, flavor, fbin, fjson)
        except Exception as e:                                   # noqa: BLE001
            failures.append({"unit_key": uk, "stage": "palette",
                             "error": f"{type(e).__name__}: {e}"})
            log(f"[render] {i+1}/{len(todo)} PALETTE FAILED {uk}: {e}")
            continue
        if palette is None:
            failures.append({"unit_key": uk, "stage": "palette",
                             "error": f"no pool member in flavor {flavor}"})
            continue

        # The crop stem is a salted digest of the unit key, NOT the served id: the served
        # `image_id` is assigned at WRITE time from the shuffled position, so naming a crop
        # after it would rename every file on a re-write.
        stem_id = crop_stem(spec, uk)
        out = crops / f"{stem_id}.jpg"
        try:
            field = ensure_label_field(loc, fields_dir=fields_dir)
            ptype = lib.palette_type(palette)
            cfg = cm.CandidateConfig(
                palette=palette, location=field.location,
                eval_width=LABEL_W, eval_height=LABEL_H,
                reverse=cparams["reverse"], log_premap=cparams["log_premap"],
                gamma=cparams["gamma"],
                phase=cparams["phase"] if ptype == "cyclic" else 0.0,
                n_cycles=cparams["n_cycles"] if ptype == "cyclic" else 1,
                transfer=cparams["transfer"], transfer_gamma=cparams["transfer_gamma"])
            w, h = render_label_crop(field, cfg, lib, out, prep=cm.stretch_field(field))
            assert (w, h) == (LABEL_W, LABEL_H), (uk, w, h)
        except Exception as e:                                   # noqa: BLE001
            failures.append({"unit_key": uk, "stage": "crop",
                             "error": f"{type(e).__name__}: {e}"})
            log(f"[render] {i+1}/{len(todo)} CROP FAILED {uk}: {e}")
            continue
        finally:
            _wipe_label_field(loc, fields_dir)

        cell = (ftype, clus, flavor, style)
        model.record_attempt(cell)
        model.record_fill(cell)
        with _ledger_path(spec).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "unit_key": uk, "crop_stem": stem_id, "cell": list(cell),
                "palette": palette, "palette_type": ptype,
                "morph_cluster": clus, "palette_flavor": flavor, "render_style": style,
                "cell_deficit": round(float(deficit), 6), "n_cell_options": n_opts,
                "pref_fit": pref_fit, "ranker": ranker.mode,
                "cfg": {"reverse": cfg.reverse, "log_premap": cfg.log_premap,
                        "gamma": cfg.gamma, "phase": cfg.phase, "n_cycles": cfg.n_cycles,
                        "transfer": cfg.transfer, "transfer_gamma": cfg.transfer_gamma,
                        "interior_color": list(cfg.interior_color), "filter": cfg.filter},
                "secs": round(time.time() - t0, 2)}) + "\n")
        times.append(time.time() - t0)
        if (i + 1) % 10 == 0 or i < 3:
            recent = float(np.mean(times[-20:]))
            log(f"[render] {i+1}/{len(todo)} {loc.family:14} {flavor:22} "
                f"{palette[:22]:22} [{times[-1]:.0f}s] recent {recent:.0f}s/loc -> eta "
                f"{recent*(len(todo)-i-1)/60:.0f} min "
                f"(elapsed {(time.time()-t_wall)/60:.0f} min)")

    log(f"[render] done: {len(times)} crops in {(time.time()-t_wall)/60:.1f} min, "
        f"{len(failures)} failures")
    ep = spec.batch_dir / "_render_errors.json"
    prior = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else []
    merged = {e["unit_key"]: e for e in prior}
    merged.update({e["unit_key"]: e for e in failures})
    now_done = set(load_ledger(spec))
    merged = {k: v for k, v in merged.items() if k not in now_done}
    ep.write_text(json.dumps(sorted(merged.values(), key=lambda e: e["unit_key"]), indent=1),
                  encoding="utf-8")
    # The run's own record of what it colorized, beside the ledger.
    (spec.batch_dir / "_colorize_meta.json").write_text(
        json.dumps({"cluster": cluster_rep, "grid": grid_rep, "ranker": ranker.mode},
                   indent=1), encoding="utf-8")


def crop_stem(spec: SheetSpec, unit_key: str) -> str:
    import hashlib
    return hashlib.blake2b(f"{spec.batch_id}|{unit_key}".encode(),
                           digest_size=8).hexdigest()


def _unit_seed(unit_key: str) -> int:
    """A stable 32-bit seed from a unit key. `hash()` is salted per process on Windows and
    would make a resumed run draw different cells than the run it resumes."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(unit_key.encode(), digest_size=4).digest(), "big")


def _wipe_label_field(loc, fields_dir: Path):
    """15 MB per label field x ~200 locations is 3 GB of scratch for fields read once."""
    import hashlib
    ptok = loc_mod.maxiter_policy_token()
    suffix = f"|{ptok}" if ptok else ""
    h = hashlib.sha1(
        f"{loc.key()}|{LABEL_W}x{LABEL_H}ss{LABEL_SS}|{loc.maxiter}{suffix}".encode()
    ).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{LABEL_W}x{LABEL_H}ss{LABEL_SS}"
    for ext in (".bin", ".json"):
        try:
            (fields_dir / f"{stem}{ext}").unlink(missing_ok=True)
        except OSError:
            pass


# =========================================================================== #
# 4. Write — seeded shuffle, blind rows, eval-only stamp.
# =========================================================================== #
def provenance_block(spec, src, rec, loc, split_side) -> dict:
    cfg = rec["cfg"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "blind_minibrot_eval_slice",
        "family": loc.family,
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
        "palette": rec["palette"],
        # THE COLORMAP RECIPE — the crop is a pure function of `render` + this block.
        "params": {
            "palette": rec["palette"], "palette_type": rec["palette_type"],
            "palette_source": "colorize_path:pref_argmax_in_flavor",
            "reverse": cfg["reverse"], "log_premap": cfg["log_premap"],
            "gamma": cfg["gamma"], "phase": cfg["phase"], "n_cycles": cfg["n_cycles"],
            "transfer": cfg["transfer"], "transfer_gamma": cfg["transfer_gamma"],
            "interior_color": cfg["interior_color"], "eval_filter": cfg["filter"],
        },
        "render_mode": "smooth",
        # The regime axis. Identical to the 2026-08-05 colorize batch's, so the two pool
        # against each other on `coloring_source` — sheet A's `pool_draw_argmax` is a third
        # regime and stays separate.
        "coloring_source": "colorize_path",
        "colorize": {
            "morph_cluster": rec["morph_cluster"], "palette_flavor": rec["palette_flavor"],
            "render_style": rec["render_style"], "cell": rec["cell"],
            "cell_deficit": rec["cell_deficit"], "n_cell_options": rec["n_cell_options"],
            "pref_fit": rec["pref_fit"], "ranker": rec["ranker"], "palette_pick": "pref",
            "color_params": "deploy_tail._color_params({}) — canonical inherited",
        },
        # the draw axes
        "vein": src["vein"],
        "partition": src["partition"],
        "intake_source": src["intake_source"],
        "source_tag": src["source_tag"],
        "floor_admit": src["floor_admit"],
        "source_ledger": src["source_ledger"],
        "source_oid": src["source_oid"],
        # THE ONLY MODEL SCORE ON THE ROW, and it is the LOCATION head's.
        "source_p_good": src["source_p_good"],
        "source_p_good_head": "LOCATION head (tools/scoring/production_pins.ACTIVE_CKPT via "
                              "ledger_rescore) — P(>=3) on the intake frame. The wallpaper "
                              "heads are absent from this row by construction.",
        "source_decoded_class": src.get("source_decoded_class"),
        "split_side": split_side,
        "split_origin": "blind_eval_only",
    }


def run_write(spec: SheetSpec, args):
    selected, pop_report, draw_report = selection(spec, args.target_rows, args.seed)
    srcs = {s["unit_key"]: s for s in selected}
    done = load_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")
    live = [s for s in selected if s["unit_key"] in done]
    # FAIL BEFORE TRUNCATING. `images.jsonl` is opened "w" below, so a run that reaches that
    # line with nothing to write REPLACES a good sheet with an empty one — and an empty
    # images.jsonl then feeds back into the next run's own prior-batch scan. That is exactly
    # how the self-exclusion bug turned into an ALTERNATING failure instead of a loud one.
    if not live:
        raise SystemExit(
            f"[write] {len(selected)} selected, {len(done)} in the render ledger, 0 in both — "
            f"refusing to overwrite {spec.batch_dir / 'images.jsonl'} with an empty sheet. "
            f"The draw and the ledger disagree; re-run `select` and compare.")

    # PRESENTATION ORDER — a SEEDED SHUFFLE, stamped. Not sorted, not grouped, not scored.
    rng = np.random.default_rng([spec.shuffle_seed, 1])
    order = sorted(live, key=lambda s: s["unit_key"])
    perm = rng.permutation(len(order))
    order = [order[int(i)] for i in perm]

    rows = []
    for i, s in enumerate(order):
        rec = done[s["unit_key"]]
        loc = s["loc"]
        image_id = f"{spec.img_prefix}{i:04d}_{rec['crop_stem'][:8]}"
        rows.append({
            "image_id": image_id,
            "sheet_order": i,
            "render": render_block(loc, rec["palette"]),
            "provenance": provenance_block(spec, s, rec, loc, "eval"),
            # THE HUMAN SLOT, and the ONLY tier field on the row. There is no
            # `suggested_tier`, no `head_v3`, no flat `p_ge3`/`pred`: the labeling rig enters
            # correction mode iff a row carries `suggested_tier`, so their absence is what
            # makes the sheet blind, and it is enforced by test_blind_minibrot_sheet.py.
            "label": {"score": None, "labeler": None, "labeled_at": None},
            "_unit_key": s["unit_key"], "_crop_stem": rec["crop_stem"],
        })
    assert len({r["image_id"] for r in rows}) == len(rows), "opaque ids collided"
    for r in rows:
        extra = set(r) - set(ROW_KEYS) - {"_unit_key", "_crop_stem"}
        assert not extra, f"{r['image_id']}: undeclared row field(s) {sorted(extra)} — a row " \
                          f"field this sheet did not declare is how a blind sheet stops " \
                          f"being blind (ROW_KEYS)"

    # Link crop stems to served ids (the UI builds crops/<image_id>.jpg client-side).
    import shutil
    crops = spec.batch_dir / "crops"
    for r in rows:
        dst = crops / f"{r['image_id']}.jpg"
        if not dst.exists():
            shutil.copyfile(crops / f"{r['_crop_stem']}.jpg", dst)
    route = {r["image_id"]: {"unit_key": r.pop("_unit_key"), "crop_stem": r.pop("_crop_stem")}
             for r in rows}

    spec.batch_dir.mkdir(parents=True, exist_ok=True)
    with (spec.batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (spec.batch_dir / "route.json").write_text(json.dumps(route, indent=1), encoding="utf-8")

    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
    meta = {}
    mp = spec.batch_dir / "_colorize_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    # INCOMPLETE is DERIVED from the counts, never a flag: a bounded `--limit` run and a
    # killed run both produce a short batch and only one of them would have set one.
    incomplete = len(rows) < draw_report["drawn_rows"]

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_id": spec.batch_id,
        "generator_version": spec.generator_version,
        "labeler": None,
        "n_rows": len(rows),
        "schema_note":
            "SHEET D — the BLIND minibrot/maneuver eval slice for the wallpaper heads. Fresh "
            "locations only (excluded against every prior wallpaper batch by key and by "
            "proximity), drawn from the stage-2 admitted intake's minibrot/maneuver veins, "
            "conditioned on quality through the LOCATION head alone. Coloured by the live "
            "emission colorize path (morph cluster -> deficit flavor -> pref-v3-gvo argmax -> "
            "canonical params) at the shared label-crop pins. NO wallpaper head score appears "
            "anywhere: no suggested_tier, no head_v3 block, no p_ge3, no score-ordered page. "
            "Presentation is a seeded shuffle stamped in sheet_order. EVAL-ONLY, PERMANENTLY.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {draw_report['drawn_rows']} drawn rows are present — this batch "
            f"is a BOUNDED or INTERRUPTED run and must not be treated as the full sheet. "
            f"Re-run `render` then `write`.") if incomplete else None,
        # --- the two properties this sheet exists for -------------------------------
        "blind": {
            # The row shape, DERIVED from the declared tuple rather than asserted in prose:
            # a reader checks that no machine-opinion field is in it, which is the whole
            # property. (The names of the fields that would break it are deliberately NOT
            # spelled here — the source scan in test_blind_minibrot_sheet.py owns that list,
            # and duplicating it as string data would defeat the scan.)
            "row_keys": list(ROW_KEYS),
            "machine_prelabel": None,
            "presentation": "seeded shuffle, stamped in sheet_order and served with "
                            "&order=file",
            "shuffle_seed": spec.shuffle_seed,
            "why": "an incumbent-vs-challenger eval slice must come from blind or "
                   "pre-incumbent labels; a slice served with the incumbent's suggestions "
                   "measures agreement with the incumbent, never quality "
                   "(classifier_retrain_protocol.md §2b). Sheet A returned 84.9% of its "
                   "labels equal to the served v3 suggestion.",
        },
        "eval_only": True,
        "eval_only_note":
            "PERMANENT. Every row is stamped split_side=eval and this slice must never enter "
            "any train split, for this head generation or any later one. It was bought to "
            "referee wallpaper v3 vs v4b on unanchored labels; training on it spends the "
            "only unanchored read of this population that exists.",
        "heads_read": {
            "wallpaper": "NONE — neither v3 nor v4b is loaded, scored, stamped or sorted on "
                         "by this builder.",
            "location": "tools/scoring/production_pins.ACTIVE_CKPT, indirectly: the intake "
                        "ledger's p_good (kept current by ledger_rescore) is compared against "
                        f"floors.GOOD_FLOOR = {F.GOOD_FLOOR:g}. Selection only.",
            "palette": "pref-v3-gvo (tools/studies/conditioned_colorize.Scorer) — the "
                       "production colorize path's palette proposal, a PALETTE-preference "
                       "head with no quality opinion about a location.",
        },
        "population_report": pop_report,
        "draw_report": draw_report,
        "colorize_path": {**meta,
                          "deficit_fill_rule":
                              "a cell fills on ATTEMPT. The 2026-08-05 colorize batch filled a "
                              "cell only when the render cleared the wallpaper pool floor; "
                              "reading that floor here would put a wallpaper head into this "
                              "sheet's substrate, so the bookkeeping is relaxed and the "
                              "deviation is named."},
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "jpg_quality": JPG_Q,
            "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(tools/wallpaper/label_crop.py — the locked label-crop pins)",
        },
        "sampling_metaparameters": {
            "target_rows": spec.target_rows, "draw_seed": spec.draw_seed,
            "cell_seed": spec.cell_seed, "shuffle_seed": spec.shuffle_seed,
            "maxiter_policy": loc_mod.maxiter_policy_token(),
            "renders_per_location": 1,
        },
        "split_summary": {
            "eval_rows": len(rows), "train_rows": 0,
            "rule": "EVERY row is eval. This is not a stratified assignment and there is "
                    "nothing to re-derive: the batch is an instrument, not a training set.",
        },
        "realized": {
            "rows_by_partition": dict(sorted(Counter(
                r["provenance"]["partition"] for r in rows).items())),
            "rows_by_vein": dict(sorted(Counter(
                r["provenance"]["vein"] for r in rows).items())),
            "rows_by_flavor": dict(Counter(
                r["provenance"]["colorize"]["palette_flavor"] for r in rows).most_common()),
            "distinct_palettes": len({r["render"]["palette"] for r in rows}),
            "palette_top10": dict(Counter(
                r["render"]["palette"] for r in rows).most_common(10)),
            "distinct_morph_clusters": len({
                r["provenance"]["colorize"]["morph_cluster"] for r in rows}),
            "source_p_good": {
                "min": round(min(r["provenance"]["source_p_good"] for r in rows), 4),
                "median": round(float(np.median([r["provenance"]["source_p_good"]
                                                 for r in rows])), 4),
                "max": round(max(r["provenance"]["source_p_good"] for r in rows), 4)},
        },
        "presentation": {
            "order": "sheet_order — a SEEDED SHUFFLE of the drawn set, contiguous",
            "sorted_on": "NOTHING. No score, machine or human, orders this sheet.",
            "contiguous": True,
            "image_id": "OPAQUE `<prefix><slot>_<hash8>` — slot is shuffled position, the "
                        "hash a salted digest of the unit key. route.json maps it back.",
        },
        "labels_export": spec.labels_export,
        "labeling": {
            "ui": spec.ui_url,
            "mode": "BLIND — no suggestion is prefilled and no row carries a machine tier. "
                    "1/2/3/4 label and advance; there is no confirm key and no bulk accept "
                    "because there is nothing to accept.",
            "blind_rows": len(rows),
            "calibration_duplicates": 0,
            # THREE DIFFERENT FILES, named explicitly because two of them are easy to
            # confuse and the confusion lands at the END of a labeling sitting: the page
            # downloads `scores.json`, `--scores` READS whatever you save that as (beside the
            # sidecar, NOT under scratch/ — a label export is the one artifact in the
            # pipeline with no rebuild path, and scratch/ is wiped wholesale), and the
            # merge WRITES the sidecar (`labels/<generator_version>.json`, derived by
            # `merge_sitting.sidecar_for` from this batch's own manifest). The re-verdict
            # then reads the sidecar, never the export.
            "export_download": "scores.json (the page's export button)",
            "save_export_as": f"labels/scores_{spec.batch_id}.json",
            "sidecar_written": spec.labels_export,
            "merge": f"uv run python tools/wallpaper/merge_sitting.py "
                     f"--corpus wallpaper_corpus --batch {spec.batch_id} "
                     f"--scores labels/scores_{spec.batch_id}.json --apply",
            "then": f"uv run python tools/wallpaper/sheet_d_reverdict.py   "
                    f"# the one-command re-verdict, after labeling",
        },
        "render_failures": errors,
        "run_status": {
            "drawn_rows": draw_report["drawn_rows"],
            "rendered_rows": len(rows),
            "n_failures": len(errors),
        },
    }
    (spec.batch_dir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print_summary(spec, batch)
    return batch


# =========================================================================== #
# Reporting.
# =========================================================================== #
def print_composition(pop, dr):
    log("-" * 96)
    log(f"{'filter':<28}{'in':>8}{'out':>8}   rule")
    for f in pop["filters"]:
        log(f"{f['filter']:<28}{f['in']:>8}{f['out']:>8}   {f['rule'][:56]}")
    log("-" * 96)
    log(f"eligible {pop['n_eligible']}  ·  drawn {dr['drawn_rows']} / target "
        f"{dr['target_rows']}  ·  {'SUPPLY-BOUND' if dr['supply_bound'] else 'apportioned'}")
    log(f"by partition: {pop['eligible_by_partition']}")
    log(f"by vein:      {pop['eligible_by_vein']}")
    log("-" * 96)


def print_summary(spec, batch):
    r = batch["realized"]
    log("\n" + "=" * 96)
    log(f"SHEET D — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 96)
    log(f"rows {batch['n_rows']} / drawn {batch['run_status']['drawn_rows']}  ·  failures "
        f"{batch['run_status']['n_failures']}")
    log(f"by partition: {r['rows_by_partition']}")
    log(f"by vein:      {r['rows_by_vein']}")
    log(f"palettes:     {r['distinct_palettes']} distinct over "
        f"{len(r['rows_by_flavor'])} flavors  ·  {r['distinct_morph_clusters']} morph clusters")
    log(f"location-head p_good: {r['source_p_good']}")
    log(f"BLIND: {batch['blind']['presentation']}  ·  EVAL-ONLY: {batch['eval_only']}")
    log(f"-> {spec.batch_dir}")
    log(f"-> serve: uv run python tools/viz/serve.py   then")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Blind minibrot/maneuver eval sheet builder.")
    ap.add_argument("stage", choices=("select", "embed", "render", "write"))
    ap.add_argument("--sheet", default="d", choices=sorted(SHEETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--target-rows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap units this run. A short batch STAMPS itself sheet_incomplete "
                         "at write time.")
    ap.add_argument("--dry-run", action="store_true", help="render: composition only")
    a = ap.parse_args(argv)
    spec = SHEETS[a.sheet]
    spec.work.mkdir(parents=True, exist_ok=True)

    if a.stage == "select":
        selected, pop_report, draw_report = selection(spec, a.target_rows, a.seed)
        print_composition(pop_report, draw_report)
        out = spec.work / "draw.json"
        out.write_text(json.dumps({
            "population": pop_report, "draw": draw_report,
            "unit_keys": [s["unit_key"] for s in selected]}, indent=1), encoding="utf-8")
        log(f"-> {out}")
    elif a.stage == "embed":
        run_embed(spec, a)
    elif a.stage == "render":
        run_render(spec, a)
    else:
        run_write(spec, a)


if __name__ == "__main__":
    main()
