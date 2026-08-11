r"""Train the wallpaper-quality head — v4b: sheet A folded in, FROM SCRATCH.

Forks `train_wallpaper_v4` and imports everything it can rather than restating it: the model
is still byte-for-byte v2/v3/v4 (MobileNetV4 `conv_medium`, CORN K=4, geometric-only aug,
384x224 stretch, same loss/optim/schedule/LRs/epochs), the ImageNet backbone is the init
(`pretrained=True`, no warm start off v3 or v4), and the five prior batches keep the exact
rows, labels and SIDES v4 gave them.

**"From scratch" here means what it meant for v4** — a fresh ImageNet backbone, not a
finetune of a previous head. It does NOT mean a fresh split, and the difference is the whole
design of this file (see below).

WHAT CHANGES — one thing, and its consequences:

  1. **DATA = the SIX labeled batches**: v4's five plus
     `2026-08-10_wallpaper_correction_v2` (sheet A, 960 renders / 960 locations, one row per
     location at the head's argmax palette). 3,638 -> 4,598 renders.

     The prompt said "the six prior + sheet A". There are seven batch directories under
     `data/wallpaper_corpus/batches/`, but `2026-07-07_wallpaper_fresh_discovery_v1` is
     **entirely unlabeled** (364 rows, every `label.score` null, no `labels/` sidecar, no
     crops) — so the labeled corpus is five prior batches, not six, and this trains on
     5 + 1. Recorded here because it is the difference between "a batch was dropped" and
     "a batch has nothing to contribute".

  2. **SPLIT: the prior five are FROZEN at v4's assignment; only sheet A is placed.**
     A globally re-randomised split would put rows v3 TRAINED ON into v4b's eval side, and
     the prompt's baseline is "wallpaper v3 re-scored on the identical eval tiles". A
     baseline that has memorised part of the eval side is not a baseline — it is an
     inflated one, and it leans the comparison the wrong way (toward a false v3 win). So
     v4's `split_union` is called verbatim on the five prior batches and its answer is
     authority.

     Sheet A honours its own stamped `provenance.split_side` EXCEPT where its coordinate
     already exists in a prior batch: **73 of its 960 locations collide with a prior
     location and 37 of those are stamped on the opposite side**, which is the same
     "assign_split is a function of the SELECTED SET" failure `train_wallpaper_v4.
     reconcile_stamped_sides` found inside the fresh pair. Those 37 rows take the PRIOR
     side. Every move is listed in `metrics.sheet_a_reconciliation`.

  3. **SELECTION = pooled-eval AP>=3 (marginal P>=3)** — DECLARED HERE, BEFORE THE RUN, and
     identical to v4's. It is a controlled variable, not a tuning knob: v2/v3 selected on
     AP>=2, v4 moved to AP>=3, and v4b does not move it again. No post-hoc switching.

STAGED, NOT ADOPTED. Writes `data/wallpaper_head/v4b/` and moves nothing — `wallpaper_pins`
still points at v3. The v3-vs-v4b winner-rule verdict is
`tools/wallpaper/wallpaper_v4b_reads.py`, which re-scores BOTH heads on the same crops
through one harness (a trainer cannot: it only ever holds one checkpoint).

    uv run python -m classifier.train_wallpaper_v4b --seeds "0 1 2 3 4"
    uv run python -m classifier.train_wallpaper_v4b --dry-run          # split only
    uv run python -m classifier.train_wallpaper_v4b --seeds "0 1 2" --epochs 2   # bounded

Outputs -> data/wallpaper_head/v4b/ (per-seed under v4b/seed_<s>/).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Transform
from .model import data_config
from .train_v2 import detect_device
from .train_wallpaper_v2 import build_wallpaper_model, eval_block, label_hist
from .train_wallpaper_v4 import (
    BATCHES, SOURCES, WRow, agg, assert_eval_only_pinned, fmt, load_rows, split_union,
    train_one_seed)
from .train_wallpaper_v4 import BatchSource, K, SPLIT_SEED

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "wallpaper_head" / "v4b"

# Sheet A. One render per location, so the location key IS the image_id; `stamped_split` is
# true but the reconciliation below can still override it (prior batches are authority).
SHEET_A = BatchSource("correction_v2", r"(wc\d+_[0-9a-f]+)$",
                      "2026-08-10_wallpaper_correction_v2",
                      "wallpaper_correction_v2.json", True, "correction")

# The unlabeled batch, named so "why is it not here" is answerable from this file.
UNLABELED_BATCH = "2026-07-07_wallpaper_fresh_discovery_v1"

SELECTION_TEXT = ("max POOLED-eval AP>=3 (marginal P>=3); full schedule, no early stop. "
                  "IDENTICAL to v4's objective and declared before the run — the "
                  "v3->v4->v4b comparison basis is a controlled variable, and v2/v3's AP>=2 "
                  "objective is the difference v4 already paid for, not one v4b re-opens.")

log = logging.getLogger("train_wallpaper_v4b")


@dataclass
class ARow(WRow):
    """A `WRow` plus sheet A's own draw axes. Prior-batch rows carry None for all three,
    which is what makes `bucket == "minibrot_maneuver"` a slice over the whole union
    instead of a lookup that only works for one batch."""
    bucket: str | None = None
    vein: str | None = None
    partition: str | None = None


def _widen(rows: list[WRow]) -> list[ARow]:
    return [ARow(**{f: getattr(r, f) for f in WRow.__dataclass_fields__}) for r in rows]


def load_sheet_a(require_crops: bool = True) -> list[ARow]:
    """Sheet A as `ARow`s. `coloring_source` is its own third regime, `pool_draw_argmax`:
    every row is the head's argmax palette at that location, which is neither the July pool
    draw nor the live colorize path, and collapsing it into either would make the two
    pre-declared no-worse slices mean something they don't."""
    src = SHEET_A
    labels = json.loads(src.labels_path.read_text())
    loc_re = re.compile(src.id_re)
    rows, seen = [], set()
    for line in (BATCHES / src.batch_id / "images.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["image_id"]
        if iid not in labels:
            raise ValueError(f"[{src.name}] row {iid} has no label — batch must be fully labeled")
        m = loc_re.match(iid)
        if m is None:
            raise ValueError(f"[{src.name}] unexpected image_id shape: {iid}")
        jpg = BATCHES / src.batch_id / "crops" / f"{iid}.jpg"
        if require_crops and not jpg.exists():
            raise FileNotFoundError(f"crop missing: {jpg}")
        rd, prov = r["render"], r["provenance"]
        if prov["split_side"] not in ("train", "eval"):
            raise ValueError(f"{iid}: bad split_side {prov['split_side']!r}")
        rows.append(ARow(
            image_id=iid, label=int(labels[iid]), jpg=jpg, loc=m.group(1),
            fractal_type=rd["fractal_type"], batch=src.name, era=src.era,
            family=prov["family"],
            coord=(rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"]),
            full_coord=(rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"],
                        rd.get("c_re"), rd.get("c_im")),
            palette_source="fresh",
            coloring_source=prov.get("coloring_source", "pool_draw_argmax"),
            source_group=("q4_harvest" if prov.get("source_tag") == "q4_harvest"
                          else "machine_admitted"),
            floor_admit=prov.get("floor_admit"),
            split_side=prov["split_side"], split_origin=prov.get("split_origin"),
            v3_p_ge3=(r.get("head_v3") or {}).get("p_ge3"),
            bucket=prov.get("bucket"), vein=prov.get("vein"),
            partition=prov.get("partition")))
        seen.add(iid)
    extra = set(labels) - seen
    if extra:
        raise ValueError(f"[{src.name}] {len(extra)} labels have no row: {sorted(extra)[:5]}")
    return rows


def load_union(require_crops: bool = True) -> tuple[list[ARow], list[ARow]]:
    """`(prior_rows, sheet_a_rows)` — the five v4 batches and sheet A, both widened."""
    return _widen(load_rows(require_crops=require_crops)), load_sheet_a(require_crops)


def split_v4b(prior: list[ARow], sheet_a: list[ARow]):
    """v4's split for the prior five, FROZEN; sheet A placed against it.

    Returns `(train, eval, meta)`. The prior side assignment is v4's answer verbatim —
    asserted afterwards by rebuilding the id sets, because "frozen" is a claim and this is
    the one place it can be checked."""
    tr0, ev0, hq3_eval_locs, strata, forced, old_slice_ids, conflicts = split_union(prior)
    prior_side = {r.image_id: "train" for r in tr0}
    prior_side.update({r.image_id: "eval" for r in ev0})
    if set(prior_side) != {r.image_id for r in prior}:
        raise AssertionError("v4's split did not cover every prior row")

    by_coord = {}
    for r in prior:
        by_coord.setdefault(r.full_coord, set()).add(prior_side[r.image_id])
    for coord, sides in by_coord.items():
        if len(sides) > 1:
            raise AssertionError(f"prior coord {coord} already straddles — v4's own "
                                 f"disjointness assert should have caught this")
    coord_side = {c: next(iter(s)) for c, s in by_coord.items()}

    moved, collisions = [], 0
    a_side = {}
    for r in sheet_a:
        prior_s = coord_side.get(r.full_coord)
        if prior_s is None:
            a_side[r.image_id] = r.split_side
            continue
        collisions += 1
        a_side[r.image_id] = prior_s
        if prior_s != r.split_side:
            moved.append({"image_id": r.image_id, "stamped": r.split_side,
                          "resolved_to": prior_s,
                          "coord": [str(x) for x in r.full_coord]})

    side = dict(prior_side)
    side.update(a_side)
    rows = list(prior) + list(sheet_a)
    train = [r for r in rows if side[r.image_id] == "train"]
    ev = [r for r in rows if side[r.image_id] == "eval"]

    # Global c-inclusive disjointness across all SIX batches.
    seen = defaultdict(set)
    for r in rows:
        seen[r.full_coord].add(side[r.image_id])
    spanning = [c for c, s in seen.items() if len(s) > 1]
    if spanning:
        raise AssertionError(f"{len(spanning)} locations span both sides after "
                             f"reconciliation (e.g. {spanning[:3]})")
    # The prior five must be untouched — this is what makes v3/v4 comparable on this slice.
    if any(side[r.image_id] != prior_side[r.image_id] for r in prior):
        raise AssertionError("a PRIOR row moved side — v4's split must be inert here")
    # The eval-only pin, re-checked on the SIX-batch answer: `split_union` above asserted it
    # for the prior five, and sheet A is placed after that call. A frozen-authority split is
    # exactly as capable of training on a blind slice as a globally re-derived one.
    eval_only_pin = assert_eval_only_pinned(rows, lambda r: side[r.image_id],
                                            where="train_wallpaper_v4b.split_v4b")

    meta = {
        "prior_train": len(tr0), "prior_eval": len(ev0),
        "old_slice_n": len(old_slice_ids), "split_strata": strata,
        "forced_train_side": forced, "v4_split_conflicts": conflicts,
        "sheet_a_reconciliation": {
            "n_locations": len(sheet_a),
            "n_colliding_with_a_prior_location": collisions,
            "n_rows_moved_to_the_prior_side": len(moved),
            "authority": "the prior five batches (v4's frozen split)",
            "why": ("sheet A ran its own bucket-stratified split over its own 960-location "
                    "draw; a location it shares with a prior batch was assigned twice, by "
                    "two draws over two different sets. Honouring sheet A's stamp there "
                    "would put a v3/v4 TRAINING location into the eval side and inflate "
                    "the baseline."),
            "moved": moved,
        },
        "eval_only_pin": eval_only_pin,
        "train_by_batch": dict(Counter(r.batch for r in train)),
        "eval_by_batch": dict(Counter(r.batch for r in ev)),
    }
    return train, ev, meta


def slices_of(ev: list[ARow]) -> dict:
    """The pre-declared eval slices, defined ABOVE the numbers.

    MOTIVATING — `sheet_a_minibrot_maneuver`: sheet A's 300-row minibrot-centered /
    maneuver-view stratum, the one the retrain is FOR.
    NO-WORSE — `fresh_colorize_path` (the v4 regression slice, and the regime production
    actually runs), `fresh_pool_draw`, and `overall`.
    Everything else is a diagnostic and is labelled as one."""
    b = np.asarray([r.batch for r in ev])
    cs = np.asarray([r.coloring_source for r in ev])
    era = np.asarray([r.era for r in ev])
    bucket = np.asarray([r.bucket or "" for r in ev])
    vein = np.asarray([r.vein or "" for r in ev])
    return {
        "overall": np.ones(len(ev), bool),
        "sheet_a_minibrot_maneuver": (b == "correction_v2") & (bucket == "minibrot_maneuver"),
        "fresh_colorize_path": cs == "colorize_path",
        "fresh_pool_draw": cs == "pool_draw",
        # diagnostics
        "sheet_a": b == "correction_v2",
        "sheet_a_maneuver_vein": (b == "correction_v2") & (vein == "maneuver"),
        "old_era": era == "july",
        "fresh_era": era == "fresh",
        "old_humanq3": b == "humanq3",
        "old_dramatic": b == "dramatic",
        "fresh_sheet": b == "fresh_sheet",
    }


PRE_DECLARED = {
    "motivating": ["sheet_a_minibrot_maneuver"],
    "no_worse": ["fresh_colorize_path", "fresh_pool_draw", "overall"],
}


def main():
    import argparse
    import shutil
    import time

    import torch

    ap = argparse.ArgumentParser(description="Train wallpaper head v4b (six-batch union).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--seeds", default="0 1 2 3 4")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--border-crop", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="load + split + report the union, then exit (no training)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    if len(seeds) < 3 and not args.dry_run:
        raise SystemExit("need >=3 seeds for a measured band")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    log.info(f"device={device}  torch={torch.__version__}  seeds={seeds}  "
             f"init=imagenet_backbone_fresh (NOT a finetune of v3 or v4)")

    prior, sheet_a = load_union(require_crops=not args.dry_run)
    rows = list(prior) + list(sheet_a)
    log.info(f"loaded {len(rows)} renders  by_batch={dict(Counter(r.batch for r in rows))}  "
             f"union_tier_hist={label_hist(rows)}")
    log.info(f"  NOT loaded: {UNLABELED_BATCH} — 364 rows, all labels null, no sidecar")
    for name in [s.name for s in SOURCES] + [SHEET_A.name]:
        sub = [r for r in rows if r.batch == name]
        log.info(f"  {name:14s}: {len(sub):4d} renders / {len({r.loc for r in sub}):3d} loc  "
                 f"{label_hist(sub)}")

    tr, ev, meta = split_v4b(prior, sheet_a)
    rec = meta["sheet_a_reconciliation"]
    log.info(f"=== split: prior five FROZEN at v4's assignment; sheet A placed against it ===")
    log.info(f"  prior {meta['prior_train']} train / {meta['prior_eval']} eval (v4, verbatim)")
    log.info(f"  sheet A: {rec['n_colliding_with_a_prior_location']} of "
             f"{rec['n_locations']} locations collide with a prior location; "
             f"{rec['n_rows_moved_to_the_prior_side']} rows moved to the prior side")
    log.info(f"  train {len(tr)} renders / {len({r.loc for r in tr})} loc  {label_hist(tr)}")
    log.info(f"  eval  {len(ev)} renders / {len({r.loc for r in ev})} loc  {label_hist(ev)}")

    sl = slices_of(ev)
    log.info("=== pre-declared eval slices (declared in PRE_DECLARED, above any number) ===")
    for role, names in PRE_DECLARED.items():
        for name in names:
            sub = [r for i, r in enumerate(ev) if sl[name][i]]
            log.info(f"  {role.upper():11s} {name:28s} n={len(sub):4d} {label_hist(sub)}")
    log.info("  diagnostics: " + ", ".join(
        f"{k}={int(v.sum())}" for k, v in sl.items()
        if k not in PRE_DECLARED["motivating"] + PRE_DECLARED["no_worse"]))
    log.info(f"SELECTION (declared): {SELECTION_TEXT}")

    probe = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    del probe
    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)
    cfg = {
        "model": "wallpaper_head_v4b", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=3, K pinned=4)", "geometry": "stretch",
        "label_unit": "render (image_id) — NO max-over-crops",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "class_weighting": "none", "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
        "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": SELECTION_TEXT,
        "selection_declared": "before training, in classifier/train_wallpaper_v4b.py",
        "split": ("the five prior batches keep v4's assignment VERBATIM (so v3 and v4 have "
                  "not trained on any eval row); sheet A honours its stamped split_side "
                  "except at the 73 locations it shares with a prior batch, where the prior "
                  "side wins"),
        "batch_ids": [s.batch_id for s in SOURCES] + [SHEET_A.batch_id],
        "unlabeled_batch_excluded": UNLABELED_BATCH,
        "init": "imagenet_backbone_fresh (NOT warm-started from v3 or v4)",
        "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "split_seed": SPLIT_SEED, "seeds": seeds,
        "pre_declared_slices": PRE_DECLARED,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if args.dry_run:
        log.info("=== DRY RUN — split only, no training ===")
        for name, mask in sl.items():
            sub = [r for i, r in enumerate(ev) if mask[i]]
            log.info(f"  {name:28s} n={int(mask.sum()):4d}  {label_hist(sub)}")
        (out_dir / "split_dryrun.json").write_text(json.dumps(meta, indent=1, default=str))
        return

    eval_labels = np.asarray([r.label for r in ev])
    per_seed, seed_blocks = [], defaultdict(list)
    best_for_stage = None
    for seed in seeds:
        log.info(f"================= SEED {seed} =================")
        info, cond, marg, ssum = train_one_seed(
            seed, tr, ev, args, device, train_tf, deploy_tf, cfg, out_dir / f"seed_{seed}")
        per_seed.append(info)
        for name, mask in sl.items():
            seed_blocks[name].append(eval_block(eval_labels, cond, marg, ssum, mask))
        sel = info["val_best_ap_good"] or -1.0
        if best_for_stage is None or sel > best_for_stage[0]:
            best_for_stage = (sel, seed, out_dir / f"seed_{seed}")
        (out_dir / "per_seed.json").write_text(json.dumps(per_seed, indent=2))

    log.info("================= CROSS-SEED AGGREGATION =================")
    headline = ["ap_not_bad", "ap_good", "ap_exceptional", "ap_4_vs_3",
                "auc_good_vs_rest", "auc_4_vs_3", "spearman_score_vs_tier"]
    agg_metrics = {}
    for name, blocks in seed_blocks.items():
        b0 = next((b for b in blocks if b is not None), None)
        agg_metrics[name] = {
            "role": ("MOTIVATING" if name in PRE_DECLARED["motivating"] else
                     "NO_WORSE" if name in PRE_DECLARED["no_worse"] else "diagnostic"),
            "n": (b0["n"] if b0 else 0), "n_good": (b0["n_good"] if b0 else 0),
            "n_exceptional": (b0["n_exceptional"] if b0 else 0),
            **{k: agg(blocks, k) for k in headline}}
        m = agg_metrics[name]
        log.info(f"  [{m['role']:11s} {name:28s}] n={m['n']:4d} good={m['n_good']:3d}  "
                 f"AP_nb {fmt(m['ap_not_bad'])}  AP_good {fmt(m['ap_good'])}  "
                 f"AUC>=3 {fmt(m['auc_good_vs_rest'])}")

    stage_sel, stage_seed, s_dir = best_for_stage
    shutil.copy(s_dir / "model_best.pt", out_dir / "model_best.pt")
    shutil.copy(s_dir / "eval_scores.jsonl", out_dir / "eval_scores.jsonl")

    eh = label_hist(ev)
    metrics = {
        "seeds": seeds, "split_seed": SPLIT_SEED,
        "selection": {"metric": "pooled-eval AP>=3 (marginal P>=3)",
                      "text": SELECTION_TEXT, "same_as": "v4",
                      "staged_seed": stage_seed, "staged_value": float(stage_sel)},
        "train_n": len(tr), "eval_n": len(ev), "eval_tier_hist": eh,
        "eval_by_batch": dict(Counter(r.batch for r in ev)),
        "eval_by_coloring_source": dict(Counter(r.coloring_source for r in ev)),
        "split": meta, "pre_declared_slices": PRE_DECLARED,
        "aggregate": agg_metrics, "per_seed": per_seed,
        "staged": {"seed": stage_seed, "ap_good": float(stage_sel),
                   "checkpoint": str(out_dir / "model_best.pt"),
                   "rule": "best per-seed pooled-eval AP>=3",
                   "adopted": False,
                   "ACTIVE_STATUS": "STAGED — NOT flipped; wallpaper_pins still points at v3",
                   "note": "the winner-rule verdict is tools/wallpaper/wallpaper_v4b_reads.py"},
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    log.info("================= WALLPAPER HEAD v4b SUMMARY =================")
    ov = agg_metrics["overall"]
    mot = agg_metrics["sheet_a_minibrot_maneuver"]
    log.info(f"  train n={len(tr)}  eval n={len(ev)} (good {eh[3]+eh[4]})")
    log.info(f"  OVERALL     n={ov['n']:4d} AP>=3 {fmt(ov['ap_good'])}  "
             f"AUC>=3 {fmt(ov['auc_good_vs_rest'])}")
    log.info(f"  MOTIVATING  n={mot['n']:4d} AP>=3 {fmt(mot['ap_good'])}  "
             f"AUC>=3 {fmt(mot['auc_good_vs_rest'])}")
    log.info(f"  STAGED -> {out_dir / 'model_best.pt'} (seed {stage_seed}) — HELD; ACTIVE is v3")
    log.info("  VERDICT is NOT decided here — run tools/wallpaper/wallpaper_v4b_reads.py")
    log.info("DONE")


if __name__ == "__main__":
    main()
