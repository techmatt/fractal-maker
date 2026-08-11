r"""Train the render-mode quality (mining) head — v3: FROM SCRATCH on the whole corpus.

NOT A FINETUNE, and that is the one decision this file exists to make loudly. v2 was a
finetune of v1 on a single 960-row sheet and it LOST its winner rule the same day; the
standing precedent (`prompts/retrains_28.md`, `train_mining_head_v2`'s outcome note) is that
a small-data finetune of this head damages it. v3 initialises from the ImageNet backbone
exactly as v1 did — `pretrained=True`, no `--init-from`, no state dict off a prior head.

WHAT IS THE SAME AS v1 (the controlled variables, imported from `train_mining_head` rather
than restated so a drift is a merge conflict): backbone `mobilenetv4_conv_small`, K=3 PINNED,
CORN ordinal loss, 384x224 stretch, the checkpoint's own mean/std, geometric-only
augmentation (border crop + h/v flip — palette and mode ARE the label), AdamW at the same
LRs, cosine schedule, 40 epochs, batch 32, dropout/weight-decay/grad-clip, fp32, and the
marginal `p_ge = cumprod(sigma)` gate.

WHAT CHANGES, all four consequences of the corpus finally existing:

  1. DATA = the POOLED corpus, all three labeled batches, 2,460 rows over 339 locations
     (`tools.mining.mining_corpus`). v1's own `dataset_v1/` did not survive the corpus loss,
     so v1 is not re-runnable and v3 is not a re-run of it on more rows — it is the first
     mining head trained on a corpus that still exists.
  2. SPLIT = re-derived GLOBALLY over the pooled locations (`split_units.build_split`, seed
     0, eval_frac 0.40). Honoring the per-batch stamps is not an option: 33 of the 91
     locations sheet B shares with the v1 sitting are stamped train by one and eval by the
     other, and sheet C is stamped 100% train. Full argument in `mining_corpus`'s docstring.
  3. NEAR-DUP WEIGHTING. 505 of the 2,460 rows sit in a multi-row near-dup group (mostly
     `direct_*` modes duplicating themselves across two cells of sheet B's 3x3
     opacity x threshold sweep). Train rows carry `weight = 1/group_size` through
     `model.corn_loss_weighted`; the eval side keeps ONE row per group so its statistics
     stay ordinary and unweighted. Groups never straddle the split — asserted, not assumed.
  4. SELECTION = max eval **AP at the >=3 boundary** (marginal `p_ge3`) on the deduplicated
     pooled eval side. DECLARED BEFORE THE RUN and identical to v2's objective; >=3 is the
     boundary `mining_gate` actually cuts at. v1 selected on >=2. Recorded in
     `config.selection`.

STAGED, NOT ADOPTED. This writes `data/render_mode_head/v3/` and moves NOTHING: no pin, no
gate, no floor, no lock. The winner-rule verdict against v1 is a separate two-checkpoint
harness (`tools/mining/mining_v3_reads.py`) because a head must be compared on the same
crops through one scorer, which a trainer cannot do.

    uv run python -m classifier.train_mining_head_v3 --seeds "0 1 2 3 4"
    uv run python -m classifier.train_mining_head_v3 --seeds "0 1 2" --epochs 2   # bounded

Outputs -> data/render_mode_head/v3/ (per-seed under v3/seed_<s>/).
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import CropDataset, Transform
from .eval import _ap
from .model import corn_loss_weighted, data_config
from .train_v2 import detect_device, set_seed

# v1's harness, imported wholesale — every metric block and the model builder are v1's.
from .train_mining_head import (
    BACKBONE, K, agg, build_mining_model, eval_block, fmt, per_mode_good_ap, predict_all)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mining.mining_corpus import (label_hist, load_corpus,       # noqa: E402
                                        summary as corpus_summary)

OUT_DIR = ROOT / "data" / "render_mode_head" / "v3"

# The selection objective, fixed ABOVE the code that computes it so it cannot be chosen
# after a number is seen. See §4 of the docstring.
SELECTION_METRIC = "ap_ge3"
SELECTION_TEXT = ("max eval AP>=3 (marginal P>=3) on the deduplicated pooled eval side; "
                  "full schedule, no early stop")

# A mode needs this many eval good rows before its per-mode AP is a CLAIM rather than a
# direction. v1 hardcoded its rich/directional lists against a corpus that no longer exists;
# here the tier is DERIVED from each mode's own eval-q3 count.
RICH_MIN_GOOD = 10

log = logging.getLogger("train_mining_head_v3")


def train_one_seed(seed, tr, ev, weights, args, device, train_tf, deploy_tf, cfg, seed_dir):
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = build_mining_model(args.drop_rate, args.drop_path_rate, pretrained=True).to(device)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # cache=False mirrors v1/v2 (the Windows commitment-limit fix for sequential seeds).
    train_loader = torch.utils.data.DataLoader(
        CropDataset(tr, train_tf, seed=seed, cache=False), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0), drop_last=False)
    eval_loader = torch.utils.data.DataLoader(
        CropDataset(ev, deploy_tf, seed=0, cache=False), batch_size=args.batch_size,
        shuffle=False, num_workers=min(4, args.num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=(min(4, args.num_workers) > 0), drop_last=False)

    w_all = torch.tensor(weights, dtype=torch.float32, device=device)
    eval_labels = np.asarray([r.label for r in ev])

    best_sel, best_epoch = -1.0, -1
    best_state = best_cond = best_marg = best_sum = None
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            w = w_all[idx.to(device)]
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            loss = corn_loss_weighted(logits, (y - 1).long(), w, num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(tr)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"[seed {seed}] NaN/Inf at epoch {epoch} — aborting seed"); break

        cond, marg, ssum = predict_all(model, eval_loader, len(ev), device)
        ap_nb = _ap((eval_labels >= 2).astype(int), marg[:, 0])
        ap_gd = _ap((eval_labels >= 3).astype(int), marg[:, 1])
        sel = -1.0 if (ap_gd is None or not np.isfinite(ap_gd)) else float(ap_gd)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_not_bad": float(ap_nb), "ap_good": float(ap_gd),
                        "selection_metric": sel})
        log.info(f"[seed {seed}] epoch {epoch:2d}  loss {train_loss:.4f}  "
                 f"AP>=3 {ap_gd:.4f}*  AP>=2 {ap_nb:.4f}  ({time.time()-t0:.1f}s)")
        if sel > best_sel:
            best_sel, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
    log.info(f"[seed {seed}] best epoch {best_epoch}: AP>=3 {best_sel:.4f} "
             f"(wall {time.time()-t_start:.0f}s)")

    seed_cfg = dict(cfg, seed=seed, best_epoch=best_epoch)
    torch.save({"state_dict": best_state, "config": seed_cfg}, seed_dir / "model_best.pt")
    with open(seed_dir / "eval_scores.jsonl", "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "mode": r.mode, "kind": r.kind,
                "batch": r.batch, "bucket": r.bucket, "family": r.family,
                "fractal_type": r.fractal_type, "label": r.label,
                "p_ge2": float(best_marg[i, 0]), "p_ge3": float(best_marg[i, 1]),
                "p_not_bad": float(best_cond[i, 0]), "p_good_cond": float(best_cond[i, 1]),
                "score": float(best_sum[i]),
            }) + "\n")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return ({"seed": seed, "best_epoch": best_epoch, "val_best_ap_good": float(best_sel),
             "history": history, "checkpoint": str(seed_dir / "model_best.pt")},
            best_cond, best_marg, best_sum)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Train render-mode (mining) head v3 — FROM SCRATCH, pooled corpus.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--seeds", default="0 1 2 3 4", help="space-separated train seeds (>=3)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--border-crop", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    if len(seeds) < 3:
        raise SystemExit("need >=3 seeds for a measured band")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
             f"backbone={BACKBONE}  seeds={seeds}  init=imagenet_backbone_fresh (NOT a finetune)")

    # --- data: pooled corpus, global split, near-dup weights ---
    pool = load_corpus()
    tr, ev = pool.train, pool.eval_rows
    weights = [r.weight for r in tr]
    summ = corpus_summary(pool)
    log.info(f"pooled corpus {len(pool.rows)} rows / {summ['n_locations']} loc / "
             f"{summ['n_units']} split units")
    log.info(f"  split re-derived globally; {pool.split_meta['rows_moved_off_stamped_side']} "
             f"rows sit on a different side than their batch stamped "
             f"({pool.split_meta['stamped_vs_pooled']})")
    log.info(f"  near-dup: {pool.group_meta['n_groups']} groups, sizes "
             f"{pool.group_meta['group_size_hist']}, "
             f"{pool.group_meta['n_rows_in_a_multi_group']} rows in a multi-row group; "
             f"label disagreement within groups "
             f"{pool.group_meta['disagreement']['n_groups_with_disagreement']}/"
             f"{pool.group_meta['disagreement']['n_multi_groups']}")
    log.info(f"train {len(tr)} rows (weight sum {sum(weights):.1f})  {label_hist(tr)}")
    log.info(f"eval  {len(ev)} rows (deduped from {len(pool.eval_all)})  {label_hist(ev)}")
    log.info(f"eval by batch {summ['eval_by_batch']}  by kind {summ['eval_by_kind']}")
    log.info(f"eval by mode {summ['eval_by_mode']}")
    log.info(f"SELECTION (declared): {SELECTION_TEXT}")

    eval_labels = np.asarray([r.label for r in ev])
    eval_modes = np.asarray([r.mode for r in ev])
    good_by_mode = {m: int(((eval_modes == m) & (eval_labels >= 3)).sum())
                    for m in sorted(set(eval_modes.tolist()))}
    rich = [m for m, g in good_by_mode.items() if g >= RICH_MIN_GOOD]
    directional = [m for m in sorted(good_by_mode) if m not in rich]
    log.info(f"per-mode tiers (>= {RICH_MIN_GOOD} eval-q3 = rich): rich={rich} "
             f"directional={directional}")

    # --- config / transforms (v1's, verbatim) ---
    probe = build_mining_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    del probe
    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)
    cfg = {
        "model": "render_mode_head_v3", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=2, K pinned=3), PER-SAMPLE WEIGHTED "
                "(model.corn_loss_weighted)",
        "geometry": "stretch", "label_unit": "render (image_id)",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "aug_rationale": "palette/mode/color IS the label (same as v1/head-v3)",
        "class_weighting": "none", "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
        "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": SELECTION_TEXT, "selection_metric": SELECTION_METRIC,
        "selection_declared": "before training, in classifier/train_mining_head_v3.py",
        "init": "imagenet_backbone_fresh — NOT a finetune of v1 or v2",
        "split": "re-derived GLOBALLY over the pooled corpus "
                 "(split_units.build_split, union-find over Julia-seed == parent point, "
                 "family-stratified over UNITS); the per-batch stamps are INCONSISTENT "
                 "and are recorded, not honored",
        "split_seed": pool.split_meta["seed"], "eval_frac": pool.split_meta["eval_frac"],
        "row_weighting": "1/near_dup_group_size on the TRAIN side; the eval side keeps one "
                         "row per group (data/render_mode_corpus/near_dup_groups_v1.json)",
        "batches": list(summ["by_batch_side"]),
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "gate": "marginal p_ge = cumprod(sigma); NEVER the CORN conditional",
        "seeds": seeds,
        "deviations_from_v1": [
            "DATA: pooled 3-batch corpus, not the lost dataset_v1/",
            "SPLIT: re-derived globally over the pooled locations",
            "WEIGHTS: 1/near_dup_group_size (v1 had no grouping)",
            "SELECTION: AP>=3, where v1 selected AP>=2",
            "MODES: every mode in the corpus; v1's trainer dropped three",
        ],
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # --- multi-seed train ---
    per_seed, overall_blocks = [], []
    mode_blocks = defaultdict(list)
    best_for_stage = None
    for seed in seeds:
        log.info(f"================= SEED {seed} =================")
        info, cond, marg, ssum = train_one_seed(
            seed, tr, ev, weights, args, device, train_tf, deploy_tf, cfg,
            out_dir / f"seed_{seed}")
        per_seed.append(info)
        overall_blocks.append(eval_block(eval_labels, cond, marg, ssum, np.ones(len(ev), bool)))
        for mode in sorted(good_by_mode):
            mode_blocks[mode].append(per_mode_good_ap(eval_labels, marg, eval_modes, mode))
        sel = info["val_best_ap_good"]
        if best_for_stage is None or sel > best_for_stage[0]:
            best_for_stage = (sel, seed, out_dir / f"seed_{seed}")
        (out_dir / "per_seed.json").write_text(json.dumps(per_seed, indent=2))

    # --- cross-seed aggregation ---
    log.info("================= CROSS-SEED AGGREGATION =================")
    ov_keys = ["ap_not_bad", "ap_good", "auc_good_vs_rest", "auc_not_bad_vs_bad",
               "spearman_pge3_vs_tier", "spearman_score_vs_tier"]
    b0 = overall_blocks[0]
    overall_agg = {"n": b0["n"], "n_not_bad": b0["n_not_bad"], "n_good": b0["n_good"],
                   **{k: agg(overall_blocks, k) for k in ov_keys}}
    log.info(f"  [OVERALL] n={overall_agg['n']}  not_bad={overall_agg['n_not_bad']}  "
             f"good={overall_agg['n_good']}")
    log.info(f"     good-vs-rest AUC(p_ge3) {fmt(overall_agg['auc_good_vs_rest'])}   "
             f"AP>=3 {fmt(overall_agg['ap_good'])}   AP>=2 {fmt(overall_agg['ap_not_bad'])}")

    conf_stack = np.array([b["confusion_true_x_pred"] for b in overall_blocks], dtype=float)
    mean_conf = conf_stack.mean(axis=0).round(1).tolist()
    mean_by_tier = {t: float(np.mean([b["mean_score_by_tier"][t] for b in overall_blocks
                                      if b["mean_score_by_tier"][t] is not None]))
                    for t in range(1, K + 1)}
    log.info(f"     mean confusion(true x pred, 1..3) {mean_conf}")
    log.info(f"     mean score by tier {mean_by_tier}")

    log.info("=== PER-MODE q3 AP (marginal p_ge3) ===")
    mode_agg = {}
    for mode in sorted(good_by_mode):
        blks = mode_blocks[mode]
        b0m = next((b for b in blks if b is not None), None)
        mode_agg[mode] = {
            "tier": ("rich" if mode in rich else "directional"),
            "n": (b0m["n"] if b0m else 0), "n_good": (b0m["n_good"] if b0m else 0),
            "ap_good": agg(blks, "ap_good"), "auc_good_vs_rest": agg(blks, "auc_good_vs_rest"),
        }
        m = mode_agg[mode]
        log.info(f"  {'RICH' if mode in rich else 'DIR '} [{mode:30s}] n={m['n']:3d} "
                 f"good={m['n_good']:3d}  AP {fmt(m['ap_good'])}  "
                 f"AUC {fmt(m['auc_good_vs_rest'])}")

    stage_sel, stage_seed, s_dir = best_for_stage
    shutil.copy(s_dir / "model_best.pt", out_dir / "model_best.pt")
    shutil.copy(s_dir / "eval_scores.jsonl", out_dir / "eval_scores.jsonl")

    good_base = overall_agg["n_good"] / max(overall_agg["n"], 1)
    metrics = {
        "seeds": seeds, "backbone": BACKBONE, "num_classes": K,
        "selection": {"metric": SELECTION_METRIC, "text": SELECTION_TEXT,
                      "staged_seed": stage_seed, "staged_value": float(stage_sel)},
        "corpus": summ, "split": pool.split_meta, "near_dup": pool.group_meta,
        "eval_n": len(ev), "eval_tier_hist": label_hist(ev),
        "eval_good_base_rate": float(good_base),
        "overall": overall_agg,
        "mean_confusion_true_x_pred": mean_conf,
        "mean_score_by_tier": mean_by_tier,
        "per_mode_good_ap": mode_agg,
        "rich_modes": rich, "directional_modes": directional,
        "per_seed": per_seed,
        "staged": {"seed": stage_seed, "ap_good": float(stage_sel),
                   "checkpoint": str(out_dir / "model_best.pt"),
                   "rule": f"best per-seed eval {SELECTION_METRIC}",
                   "adopted": False,
                   "note": "STAGED ONLY — mining_pins.ACTIVE_MINING_CKPT still points at v1. "
                           "The winner-rule verdict is tools/mining/mining_v3_reads.py."},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info("================= RENDER-MODE (MINING) HEAD v3 SUMMARY =================")
    log.info(f"  seeds={seeds}  eval n={len(ev)}  eval-good={overall_agg['n_good']} "
             f"(base {good_base:.3f})")
    log.info(f"  OVERALL  AUC>=3 {fmt(overall_agg['auc_good_vs_rest'])}  "
             f"AP>=3 {fmt(overall_agg['ap_good'])}  AP>=2 {fmt(overall_agg['ap_not_bad'])}")
    log.info(f"  STAGED -> {out_dir / 'model_best.pt'} (seed {stage_seed}, "
             f"AP>=3 {stage_sel:.3f}) — HELD; ACTIVE is still mining v1")
    log.info(f"  VERDICT is NOT decided here — run tools/mining/mining_v3_reads.py")
    log.info("DONE")


if __name__ == "__main__":
    main()
