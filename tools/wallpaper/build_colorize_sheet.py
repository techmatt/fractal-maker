r"""build_colorize_sheet.py — the fresh-era sheet's SIBLING: colorings from the LIVE
production colorize path instead of pool draws.

Same intake population, same seeded stratification, same label-crop pins, same v3 pre-label,
same correction-sheet serving. ONE thing differs, and it is the whole point
(prompts/wallpaper_sheet_addendum_colorize_path.md): where `build_fresh_sheet.py` draws a
palette and a parameter set out of the query-sampler pool, this batch colours each location
exactly the way an emission run colours it —

  1. MORPH CLUSTER   the production intake descriptor: a retained 640x360 ss2 smooth field ->
                     `library_annotate.morph_gray_image` -> CLIP -> incremental medoid at
                     cos 0.974, SEEDED from the library's own medoids (`library_seed_v2`), so
                     a location that near-duplicates a library look reports the LIBRARY's
                     cluster index instead of founding a parallel one.
  2. DEFICIT CELL    `cells.choose_option` over the live feasible grid (partition x cluster x
                     palette flavor x render style) against the target measure derived from
                     `release_mix.RATIO`. Style is pinned to `smooth` per the addendum, so the
                     deficit is choosing the palette FLAVOR.
  3. PALETTE         `build_emission_diversity_v1.PaletteRanker` in its deployed `pref` mode:
                     the pref-v3-gvo argmax over that flavor's pool members, scored on the
                     location's cached field through the coarse recolor — the exact object the
                     emission driver uses, imported, not reimplemented.
  4. PARAMS          `deploy_tail._color_params({})` — the canonical inherited coloring
                     (transfer=pct, gamma 1, no reverse/phase/cycles). This is the sharpest
                     difference from the pool-draw regime, which randomizes all of it.

WHAT IS DELIBERATELY *NOT* THE EMISSION PATH. The crop is rendered at the shared label-crop
pins (1280x720 ss2 lanczos3 q90), not the emission pool's 960x540: a label batch that does not
share the other four batches' geometry cannot be unioned with them at train time, and ss-level
correlating with regime would be a batch effect on exactly the axis this batch exists to
isolate. The coloring recipe is the emission path's; the canvas is the corpus's.

NO GATE IS APPLIED. `floors.WALLPAPER_POOL` is READ, for one purpose only: the deficit model
counts a cell as FILLED when a render clears the pool floor, and reproducing that bookkeeping
is what makes the flavor sequence match a real run's. Every render lands in the sheet whatever
it scores. No head, gate or floor is changed anywhere here.

  uv run python -u tools/wallpaper/build_colorize_sheet.py estimate
  uv run python -u tools/wallpaper/build_colorize_sheet.py embed    # stage 1 (resumable)
  uv run python -u tools/wallpaper/build_colorize_sheet.py render   # stages 2-4 (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "queries", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import colormap as cm                    # noqa: E402
import corpus_common as cc               # noqa: E402
import release_mix as RM                 # noqa: E402  THE release-mix ratio table
from label_crop import (                 # noqa: E402  THE shared label-crop pins
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q, render_label_crop)
from tools.emission import cells as C            # noqa: E402
from tools.emission import descriptor as D       # noqa: E402
from tools.emission import floors as F           # noqa: E402  THE stage-2 cut owner
from tools.emission import library_seed_v2 as LSEED   # noqa: E402
from tools.wallpaper import wallpaper_pins as WP      # noqa: E402
from tools.wallpaper.suggest_tier import (            # noqa: E402
    DERIVATION as SUGGEST_DERIVATION, expected_tier, tier_from_pred)

# The sibling builder owns the population, the screen, the bins, the stratified draw and the
# split. Imported wholesale — a second copy of the draw is a second sheet composition nobody
# decided (`test_fresh_sheet.py` pins these, and it must keep covering both batches).
import build_fresh_sheet as FS           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

BATCH_ID = "2026-08-05_wallpaper_colorize_path_v1"
GENERATOR_VERSION = "wallpaper_colorize_path_v1"
IMG_PREFIX = "wcp"
COLORING_SOURCE = "colorize_path"        # the regime axis; the sibling stamps "pool_draw"
LABELS_EXPORT = ROOT / "labels" / "wallpaper_colorize_path_v1.json"

WALLPAPER_CORPUS = ROOT / "data" / "wallpaper_corpus"
WORK = ROOT / "scratch" / "wallpaper_colorize_sheet"
FIELD_CACHE = WORK / "intake_fields"     # retained 640x360 ss2 fields (embed AND pref ranker)
EMB_DIR = WORK / "embs"                  # one <unit_key>.npy per location (resumable)

# --- composition ------------------------------------------------------------
TARGET_LOCS = 180        # 1 colorize per location -> 180 renders, inside the addendum's 150-200
SEED = 11                # a DIFFERENT draw from the sibling's 7; overlap is allowed and reported
RENDER_STYLES = ("smooth",)   # the addendum pins smooth; the live axis also carries the
                              # promoted strange modes, which route to the mining head and
                              # would not be judged by v3 at all.
SHUFFLE_SEED = 20260806


def log(msg: str):
    print(msg, flush=True)


def batch_dir() -> Path:
    return WALLPAPER_CORPUS / "batches" / BATCH_ID


def _safe(unit_key: str) -> str:
    return unit_key.replace(":", "__").replace("/", "_")


# =========================================================================== #
# The draw — the sibling's population, screen, bins and stratification.
# =========================================================================== #

def drawn_locations(target=TARGET_LOCS, seed=SEED):
    """(selected screen records, sources by unit_key, selection report, split sides)."""
    screen = FS.load_screen()
    if not screen:
        raise SystemExit("[colorize] no screen records — run build_fresh_sheet.py screen first")
    srcs = {s["unit_key"]: s for s in FS.population()[0]}
    selected, rep = FS.select(list(screen.values()), target, seed)
    sides, n_eval = FS.assign_split(selected)
    return selected, srcs, rep, sides, n_eval


# =========================================================================== #
# Stage 1 — the production intake descriptor: retained field + morph-CLIP embedding.
# =========================================================================== #

def _emb_path(unit_key: str) -> Path:
    return EMB_DIR / f"{_safe(unit_key)}.npy"


def _field_paths(la, store, loc):
    stem = store.field_stem(loc, "smooth", la.W, la.H, la.SS)
    return str(FIELD_CACHE / f"{stem}.bin"), str(FIELD_CACHE / f"{stem}.json")


def run_embed(args):
    """One retained 640x360 ss2 smooth field + one CLIP morph embedding per drawn location.

    Checkpointed per location (atomic `.npy`), so a kill loses at most the in-flight row —
    the same discipline `campaign1_intake` runs under, for the same reason."""
    from tools.wallpaper import library_annotate as la
    from tools.wallpaper import library_store as store
    from tools.curation.colored_clip import load_clip, embed_clip

    FIELD_CACHE.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    selected, srcs, _rep, _sides, _ne = drawn_locations(args.target_locs, args.seed)
    todo = [r for r in selected if not _emb_path(r["unit_key"]).exists()]
    n_done = len(selected) - len(todo)          # count BEFORE --limit truncates the queue,
    if args.limit:                              # else a smoke reports 177 already embedded
        todo = todo[:args.limit]
    log(f"[embed] {len(selected)} drawn · {n_done} embedded · {len(todo)} to run")
    if not todo:
        return
    model, tf = load_clip()
    t0 = time.time()
    times = []
    for i, rec in enumerate(todo):
        t = time.time()
        loc = srcs[rec["unit_key"]]["loc"]
        field = la.ensure_field(loc, retain=True, tmp_dir=FIELD_CACHE, cache_root=FIELD_CACHE)
        emb = embed_clip(model, tf, [la.morph_gray_image(field)])[0].astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        p = _emb_path(rec["unit_key"])
        tmp = p.with_name(p.name + ".tmp")
        # `np.save(path, ...)` APPENDS ".npy" to a path that lacks it, so a "<id>.npy.tmp"
        # target silently becomes "<id>.npy.tmp.npy" and the atomic replace then fails on a
        # file that was never written. Handing it an open file object writes exactly here.
        with open(tmp, "wb") as fh:
            np.save(fh, emb)
        os.replace(tmp, p)
        times.append(time.time() - t)
        if (i + 1) % 20 == 0 or i < 2:
            recent = float(np.mean(times[-20:]))
            log(f"[embed] {i+1}/{len(todo)} {loc.family:18} [{times[-1]:.1f}s] recent "
                f"{recent:.1f}s/loc -> eta {recent*(len(todo)-i-1)/60:.0f} min "
                f"(elapsed {(time.time()-t0)/60:.0f} min)")
    log(f"[embed] done in {(time.time()-t0)/60:.1f} min")


def cluster(selected, srcs):
    """unit_key -> `<partition>#<k>`, WITHIN the base partition, seeded from the library.

    `descriptor.cluster_incremental` verbatim (same 0.974 knee, same frozen-medoid rule); the
    seed is `library_medoids` over the `human_q3plus` snapshot, which is the only library
    medoid set on disk. A missing embedding is a hard stop, not a skipped location — a
    silently-unclustered row would be assigned a cell nothing has a target for."""
    from partitions import base_partition
    embs, missing = {}, []
    for rec in selected:
        p = _emb_path(rec["unit_key"])
        if not p.exists():
            missing.append(rec["unit_key"])
            continue
        embs[rec["unit_key"]] = np.load(p).astype(np.float32).reshape(-1)
    if missing:
        raise SystemExit(f"[cluster] {len(missing)} drawn locations have no embedding "
                         f"(e.g. {missing[:3]}) — run `embed` to completion first.")
    seed = D.library_medoids(LSEED.INTAKE_JSON, LSEED.EMB_DIR)
    by_group: dict = {}
    for rec in selected:
        by_group.setdefault(base_partition(srcs[rec["unit_key"]]["partition"]), []).append(rec)
    tags, n_seeded_joins = {}, 0
    for group, recs in sorted(by_group.items()):
        items = [(r["unit_key"], embs[r["unit_key"]]) for r in sorted(recs, key=lambda r: r["unit_key"])]
        seed_keys = {k for k, _e in (seed.get(group) or [])}
        assign = D.cluster_incremental(items, D.NEAR_DUP_THRESHOLD,
                                       seed_medoids=seed.get(group))
        for uk, k in assign.items():
            tags[uk] = f"{srcs[uk]['partition']}#{k}"
            if k in seed_keys:
                n_seeded_joins += 1
    return tags, {"library_seed_groups": {g: len(v) for g, v in sorted(seed.items())},
                  "seeded_joins": n_seeded_joins,
                  "distinct_clusters": len(set(tags.values())),
                  "threshold": D.NEAR_DUP_THRESHOLD}


# =========================================================================== #
# Stage 2 — the deficit grid, built the way the emission driver builds it.
# =========================================================================== #

def build_deficit(selected, srcs, tags, cell_to_names, lib):
    flavors = sorted(f for f, names in cell_to_names.items()
                     if any(p in lib.colormaps for p in names))
    observed = sorted({(srcs[r["unit_key"]]["partition"], tags[r["unit_key"]]) for r in selected})
    feasible = C.build_feasible_cells(observed, flavors, RENDER_STYLES)
    shares = RM.shares(sorted({p for (p, _c) in observed}))
    target = C.TargetMeasure.from_partition_shares(shares, feasible)
    model = C.DeficitModel(feasible, target)
    return model, flavors, {
        "flavors": len(flavors), "styles": list(RENDER_STYLES),
        "observed_type_cluster": len(observed), "feasible_cells": len(feasible),
        "release_shares": {k: round(v, 5) for k, v in sorted(target.shares.items())},
        "attempt_cap": target.attempt_cap, "softmax_temp": target.softmax_temp,
        "target_source": "tools/scoring/release_mix.RATIO re-solved against the live "
                         "feasible cells (cells.TargetMeasure.from_partition_shares)",
    }


# =========================================================================== #
# Stage 3 — colorize + render + pre-label.
# =========================================================================== #

def _ledger_path() -> Path:
    return batch_dir() / "_progress_ledger.jsonl"


def load_ledger() -> dict:
    done = {}
    p = _ledger_path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit_key"]] = rec
    return done


def render_block(loc, palette):
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


def run_render(args):
    from classifier.inference import load_scorer
    from tools.studies import conditioned_colorize as cond
    from tools.mining import deploy_tail as dt
    from tools.wallpaper import library_annotate as la
    from tools.wallpaper import library_store as store
    # The live colorize driver, by file path: `tools/emission/` is a package but the driver
    # is a script inside it, so a plain `from tools.emission import ...` executes its module
    # body under a name the module itself does not expect.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "build_emission_diversity_v1", ROOT / "tools" / "emission" / "build_emission_diversity_v1.py")
    EM = importlib.util.module_from_spec(_spec)
    sys.modules["build_emission_diversity_v1"] = EM
    _spec.loader.exec_module(EM)

    selected, srcs, sel_rep, sides, n_eval = drawn_locations(args.target_locs, args.seed)
    tags, cluster_rep = cluster(selected, srcs)
    lib = cond.qs.load_pool_library() if hasattr(cond, "qs") else None
    if lib is None:
        import query_sampler as qs
        lib = qs.load_pool_library()
    _name_to_cell, cell_to_names = cond.load_cell_map()
    model, flavors, grid_rep = build_deficit(selected, srcs, tags, cell_to_names, lib)
    ranker = EM.PaletteRanker(dt, cell_to_names, lib, pick_mode="pref")
    pool_floor = F.WALLPAPER_POOL.for_style("smooth") if hasattr(F.WALLPAPER_POOL, "for_style") \
        else F.WALLPAPER_POOL.value
    cparams = dt._color_params({})               # the canonical inherited coloring

    log(f"[colorize] {len(selected)} locations · {grid_rep['flavors']} flavors × "
        f"{list(RENDER_STYLES)} · {grid_rep['feasible_cells']} feasible cells · "
        f"{cluster_rep['distinct_clusters']} morph clusters "
        f"({cluster_rep['seeded_joins']} joined a library cluster)")
    log(f"[colorize] ranker={ranker.mode} · pool floor {pool_floor} (deficit bookkeeping only)")
    FS._print_composition(sel_rep)
    if args.estimate:
        return

    # Resume: replay the ledger into the deficit model so the flavor sequence continues
    # rather than restarting from an empty pool.
    done = load_ledger()
    for rec in done.values():
        cell = tuple(rec["cell"])
        model.record_attempt(cell)
        if rec["passed"]:
            model.record_fill(cell)
    rng = np.random.default_rng(args.seed)
    for _ in range(sum(len(r["rows"]) for r in done.values())):
        rng.random()                              # keep the choice stream aligned on resume

    scorer = load_scorer(str(WP.HEAD_CKPT))
    crops = batch_dir() / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    todo = [r for r in selected if r["unit_key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"[colorize] {len(done)} in the ledger, {len(todo)} to run")

    order = {r["unit_key"]: i for i, r in enumerate(selected)}
    failures, times = [], []
    t_wall = time.time()
    for i, rec in enumerate(todo):
        t0 = time.time()
        uk = rec["unit_key"]
        src = srcs[uk]
        loc = src["loc"]
        ui = order[uk]
        ftype, clus = src["partition"], tags[uk]

        choice = C.choose_option(model, ftype, clus, flavors, RENDER_STYLES, rng)
        if choice is None:
            failures.append({"unit_key": uk, "stage": "cell", "error": "all cells capped"})
            continue
        flavor, style, deficit, n_opts, _p = choice
        fbin, fjson = _field_paths(la, store, loc)
        try:
            palette, pref_fit = ranker.best(uk, flavor, fbin, fjson)
        except Exception as e:                                   # noqa: BLE001
            failures.append({"unit_key": uk, "stage": "palette", "error": f"{type(e).__name__}: {e}"})
            log(f"[colorize] {i+1}/{len(todo)} PALETTE FAILED {uk}: {e}")
            continue
        if palette is None:
            failures.append({"unit_key": uk, "stage": "palette", "error": f"no pool member in flavor {flavor}"})
            continue

        image_id = f"{IMG_PREFIX}_{ui:03d}"
        out = crops / f"{image_id}.jpg"
        try:
            # The label-crop canvas, coloured by the emission path's recipe: canonical params
            # + the flavor-constrained pref pick. `ensure_label_field` is the SAME dump the
            # sibling batch renders off, so the two regimes differ only in the colouring.
            field = FS.ensure_label_field(loc, fields_dir=FS.WORK / "render_fields")
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
            assert (w, h) == (LABEL_W, LABEL_H), (image_id, w, h)
        except Exception as e:                                   # noqa: BLE001
            failures.append({"unit_key": uk, "stage": "crop", "error": f"{type(e).__name__}: {e}"})
            log(f"[colorize] {i+1}/{len(todo)} CROP FAILED {uk}: {e}")
            continue

        marg = FS._marginals_from_paths(scorer, [out])[0]
        pred = expected_tier(marg)
        cell = (ftype, clus, flavor, style)
        # The live path's bookkeeping: a cell FILLS when the render clears the pool floor.
        # Read off the label crop rather than the emission pool's 960x540 render — the floor
        # was set on a head trained from 1280-sourced crops, so this is the closer read of the
        # two, and it is the only render this batch makes.
        passed = bool(marg[1] >= pool_floor)
        capped = model.record_attempt(cell)
        if passed:
            model.record_fill(cell)

        row = {
            "image_id": image_id,
            "render": render_block(loc, palette),
            "provenance": {
                "generator_version": GENERATOR_VERSION, "batch_id": BATCH_ID,
                "lineage": "emission_colorize_path",
                "family": loc.family,
                "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
                "c_re": loc.c_re, "c_im": loc.c_im,
                "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
                "palette": palette,
                "params": {
                    "palette": palette, "palette_type": ptype,
                    "palette_source": "colorize_path:pref_argmax_in_flavor",
                    "reverse": cfg.reverse, "log_premap": cfg.log_premap, "gamma": cfg.gamma,
                    "phase": cfg.phase, "n_cycles": cfg.n_cycles,
                    "transfer": cfg.transfer, "transfer_gamma": cfg.transfer_gamma,
                    "interior_color": list(cfg.interior_color), "eval_filter": cfg.filter,
                },
                "render_mode": "smooth",
                "coloring_source": COLORING_SOURCE,
                # the colorize path's own record
                "colorize": {
                    "morph_cluster": clus, "palette_flavor": flavor, "render_style": style,
                    "cell": list(cell), "cell_deficit": round(float(deficit), 6),
                    "n_cell_options": n_opts, "capped_cell": bool(capped),
                    "pref_fit": pref_fit, "ranker": ranker.mode, "palette_pick": "pref",
                    "color_params": "deploy_tail._color_params({}) — canonical inherited",
                    "pool_floor_read": pool_floor, "passed_pool_floor": passed,
                },
                # intake provenance (identical fields to the sibling batch)
                "intake_source": src["intake_source"], "source_tag": src["source_tag"],
                "floor_admit": src["floor_admit"], "partition": src["partition"],
                "source_ledger": src["source_ledger"], "source_oid": src["source_oid"],
                "source_p_good": src["source_p_good"],
                "source_decoded_class": src["source_decoded_class"],
                "human_q3plus_label": src["human_label"],
                "screen_bin": FS.BIN_LABELS[rec["bin"]],
                "loc_screen_p_ge3": rec["loc_p_ge3"],
                "split_side": sides[uk], "split_origin": "colorize_sheet_binstratified",
            },
            "label": {"score": None, "labeler": None, "labeled_at": None},
            "head_v3": {
                "pred": pred, "p_ge2": float(marg[0]), "p_ge3": float(marg[1]),
                "p_ge4": float(marg[2]) if len(marg) > 2 else None,
                "ckpt": WP.HEAD_CKPT_REL, "head_version": WP.HEAD_VERSION,
            },
            "p_ge3": float(marg[1]),
            "suggested_tier": tier_from_pred(pred),
        }
        with _ledger_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"unit_key": uk, "unit_index": ui, "cell": list(cell),
                                 "passed": passed, "rows": [row]}) + "\n")
        times.append(time.time() - t0)
        if (i + 1) % 10 == 0 or i < 3:
            recent = float(np.mean(times[-20:]))
            log(f"[colorize] {i+1}/{len(todo)} {loc.family:16} {flavor:22} {palette[:24]:24} "
                f"p_ge3={marg[1]:.2f} [{times[-1]:.0f}s] recent {recent:.0f}s/loc -> eta "
                f"{recent*(len(todo)-i-1)/60:.0f} min (elapsed {(time.time()-t_wall)/60:.0f} min)")

    write_batch(sel_rep, cluster_rep, grid_rep, sides, n_eval, failures,
                time.time() - t_wall, args)


# =========================================================================== #
# Stage 4 — batch assembly.
# =========================================================================== #

def write_batch(sel_rep, cluster_rep, grid_rep, sides, n_eval, failures, wall_s, args):
    bd = batch_dir()
    bd.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in load_ledger().values():
        rows.extend(rec["rows"])
    rng = np.random.default_rng(SHUFFLE_SEED)
    rows = [rows[int(i)] for i in rng.permutation(len(rows))]
    for i, r in enumerate(rows):
        r["sheet_order"] = i
    with (bd / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    sib = FS.batch_dir() / "images.jsonl"
    sib_keys = set()
    if sib.exists():
        for line in sib.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)["render"]
                sib_keys.add((d["cx"], d["cy"], d["fw"], d["fractal_type"], d["c_re"], d["c_im"]))
    mine = {(r["render"]["cx"], r["render"]["cy"], r["render"]["fw"],
             r["render"]["fractal_type"], r["render"]["c_re"], r["render"]["c_im"]) for r in rows}

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "schema_note": "Colorize-path sibling of the fresh-era correction sheet. Same intake "
                       "population, same seeded bin stratification, same label-crop pins, same "
                       "v3 pre-label — but each location is COLOURED the way a live emission "
                       "run colours it: morph cluster (production intake descriptor, library-"
                       "seeded) -> deficit-assigned palette flavor (cells.choose_option against "
                       "the release_mix target) -> pref-v3-gvo argmax palette within that "
                       "flavor -> canonical inherited coloring params, smooth. "
                       "provenance.coloring_source == 'colorize_path' separates it from the "
                       "sibling's 'pool_draw'. label.score is null on every row.",
        "coloring_source": COLORING_SOURCE,
        "sibling_batch": FS.BATCH_ID,
        "head": {"ckpt": WP.HEAD_CKPT_REL, "version": WP.HEAD_VERSION,
                 "role": "pre-label only; the pool floor is read for deficit bookkeeping and "
                         "gates nothing in this batch"},
        "suggested_tier_rule": SUGGEST_DERIVATION,
        "population": args._population_report,
        "colorize_path": {
            "cluster": cluster_rep,
            "grid": grid_rep,
            "palette_pick": "build_emission_diversity_v1.PaletteRanker(pick_mode='pref') — "
                            "pref-v3-gvo argmax over the flavor's pool members, scored on the "
                            "cached 640x360 field via the coarse recolor",
            "color_params": "tools/mining/deploy_tail._color_params({}) — canonical inherited "
                            "(transfer=pct, gamma=1, no reverse/phase/cycles)",
            "geometry_note": "rendered at the shared label-crop pins (1280x720 ss2 lanczos3 "
                             "q90), NOT the emission pool's 960x540: the batch has to union "
                             "with the other four at train time.",
            "flavor_histogram": dict(Counter(
                r["provenance"]["colorize"]["palette_flavor"] for r in rows).most_common()),
            "palette_histogram": dict(Counter(
                r["render"]["palette"] for r in rows).most_common(15)),
            "n_distinct_palettes": len({r["render"]["palette"] for r in rows}),
            "passed_pool_floor": sum(1 for r in rows
                                     if r["provenance"]["colorize"]["passed_pool_floor"]),
        },
        "sampling_metaparameters": {
            "target_locations": args.target_locs, "renders_per_location": 1,
            "seed": args.seed, "sibling_seed": FS.SEED,
            "split_seed": FS.SPLIT_SEED, "eval_frac": FS.EVAL_FRAC,
            "shuffle_seed": SHUFFLE_SEED,
            "score_bins": list(FS.SCORE_BINS), "bin_labels": list(FS.BIN_LABELS),
            "floor_admit_frac_target": FS.FLOOR_SOURCE_DRAW_FRAC,
        },
        "selection_report": sel_rep,
        "sibling_location_overlap": {
            "n_locations_here": len(mine),
            "n_shared_with_sibling": len(mine & sib_keys),
            "note": "shared locations are a FEATURE: at those the two batches differ only in "
                    "the coloring regime, which is the cleanest contrast the pair supports.",
        },
        "split_summary": {
            "eval_locations": n_eval,
            "train_locations": sum(1 for v in sides.values() if v == "train"),
            "eval_rows": sum(1 for r in rows if r["provenance"]["split_side"] == "eval"),
            "train_rows": sum(1 for r in rows if r["provenance"]["split_side"] == "train"),
            "rule": "location-grouped, seeded, stratified by screen score bin (the sibling's "
                    "assign_split, same seed) — one location -> one side",
        },
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS, "filter": LABEL_FILTER,
            "jpg_quality": JPG_Q, "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(tools/wallpaper/label_crop.py)",
        },
        "labels_export": str(LABELS_EXPORT.relative_to(ROOT)).replace("\\", "/"),
        "labeling": {
            "ui": f"tools/viz/wallpaper_label.html?batch={FS.BATCH_ID},{BATCH_ID}",
            "mode": "correction — served as the SECOND section after the sibling sheet; "
                    "each batch exports its own scores file.",
        },
        "render_failures": failures,
        "run_status": {"planned_locations": len(sides), "completed_locations": len(load_ledger()),
                       "n_failures": len(failures), "wall_seconds": wall_s},
        "n_rows": len(rows),
    }
    (bd / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")

    log("\n" + "=" * 78)
    log(f"COLORIZE-PATH SHEET — {BATCH_ID}")
    log("=" * 78)
    log(f"rows {len(rows)} · locations {len(load_ledger())} · failures {len(failures)}")
    log(f"suggested tiers: {dict(sorted(Counter(r['suggested_tier'] for r in rows).items()))}")
    log(f"flavors used: {len(batch['colorize_path']['flavor_histogram'])} · distinct palettes "
        f"{batch['colorize_path']['n_distinct_palettes']}")
    log(f"split rows: {dict(Counter(r['provenance']['split_side'] for r in rows))}")
    log(f"shared locations with sibling: {len(mine & sib_keys)}/{len(mine)}")
    log(f"-> {bd}")
    return bd


# =========================================================================== #
# Driver.
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description="Colorize-path wallpaper correction sheet.")
    ap.add_argument("stage", choices=("estimate", "embed", "render", "write"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target-locs", type=int, default=TARGET_LOCS)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()

    prio = cc.set_below_normal_priority()
    os.environ.setdefault("RAYON_NUM_THREADS", str(cc.DEFAULT_ENGINE_THREADS))
    log(f"[colorize-sheet] priority {prio} · RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']}")
    WORK.mkdir(parents=True, exist_ok=True)
    args._population_report = FS.population()[1]

    if args.stage == "estimate":
        selected, _srcs, rep, _sides, n_eval = drawn_locations(args.target_locs, args.seed)
        FS._print_composition(rep)
        log(f"\nembed  : {len(selected)} locations x (640x360 ss2 field + CLIP)")
        log(f"render : {len(selected)} locations x 1 colorize-path label crop")
        log(f"split  : {n_eval} eval-side locations")
        return
    if args.stage == "embed":
        run_embed(args)
        return
    if args.stage == "render":
        run_render(args)
        return
    if args.stage == "write":
        selected, srcs, rep, sides, n_eval = drawn_locations(args.target_locs, args.seed)
        tags, cluster_rep = cluster(selected, srcs)
        from tools.studies import conditioned_colorize as cond
        import query_sampler as qs
        lib = qs.load_pool_library()
        _n2c, cell_to_names = cond.load_cell_map()
        _model, _flavors, grid_rep = build_deficit(selected, srcs, tags, cell_to_names, lib)
        write_batch(rep, cluster_rep, grid_rep, sides, n_eval, [], 0.0, args)
        return


if __name__ == "__main__":
    main()
