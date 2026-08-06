"""Train the render-mode quality (mining) head — v2: a FINETUNE of v1 on the fresh sheet.

v1's dataset (`data/render_mode_corpus/dataset_v1/{train,eval}.jsonl`, 774/625) did not
survive the corpus loss, so v1 cannot be reproduced and v2 cannot be a re-run of it. What
v2 IS: the same recipe and the same backbone, **initialised from v1's weights**, trained on
the 960-row rebuilt correction sheet
(`data/render_mode_corpus/batches/2026-08-06_render_mode_fresh_sheet_v1`) respecting the
split side that batch stamped **in-row** (`provenance.split_side`, 538 train / 422 eval,
union-find over Julia-parent-linked locations).

FIVE DEVIATIONS FROM THE COMMITTED v1 RECIPE, each deliberate and each stamped into
`config.json` under `deviations_from_v1`. Everything else — backbone, K, CORN loss, LRs,
epochs, batch size, dropout, weight decay, grad clip, fp32, the geometric-only augmentation
and the marginal-`p_ge` gate — is v1's, imported from `train_mining_head` rather than
restated so a drift between the two is a merge conflict and not a silent difference:

  1. DATA. The fresh batch's `images.jsonl`, not `dataset_v1/`. The split is still read,
     never re-derived — it moved from a pair of files to a per-row field.
  2. INIT. `data/render_mode_head/v1/model_best.pt` (`--init-from`), not a fresh ImageNet
     backbone. Strict `load_state_dict`: same backbone, same K, so a shape mismatch here
     means the pin moved under us and must fail loudly.
  3. SELECTION. Max eval **AP at the >=3 boundary** (marginal `p_ge3`), where v1 selected on
     the >=2 boundary. `>=3` is the boundary the mining gate actually cuts at, and the
     prompt asks for it explicitly.
  4. MODES. All 15 roster modes. v1's trainer dropped `trap_circle`, `exp_smoothing` and
     `direct_trap_screen`; that was a v1-era decision about a corpus that no longer exists
     and does not carry forward (`mining_roster.TRAINER_DROPPED_V1` keeps the record).
  5. PER-MODE REPORTING. v1 hardcoded a rich/directional mode split sized to the lost
     corpus's eval mass. Here every mode carries 26-30 eval rows by construction
     (`batch.json` -> `allocation.per_mode[*].eval_rows`), so the tier is DERIVED from each
     mode's own eval-q3 count instead of declared.

    uv run python -m classifier.train_mining_head_v2 --seeds "0 1 2 3 4"

Outputs -> data/render_mode_head/v2/ (per-seed under v2/seed_<s>/). The eval/calibration
reads that decide whether any of this is adopted are a SEPARATE harness that scores v1 and
v2 through one code path: `tools/mining/mining_v2_reads.py`. Nothing here moves a pin, a
floor or a gate.
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

from .data import Transform
from .eval import _ap
from .model import corn_loss, data_config
from .train_v2 import detect_device, set_seed

# v1's harness, imported wholesale. K, BACKBONE and every metric block are v1's.
from .train_mining_head import (
    BACKBONE, K, MRow, agg, build_mining_model, eval_block, fmt, label_hist, make_loader,
    per_mode_good_ap, predict_all)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mining.mining_roster import MODES, TRAINER_DROPPED_V1  # noqa: E402

BATCH_DIR = (ROOT / "data" / "render_mode_corpus" / "batches"
             / "2026-08-06_render_mode_fresh_sheet_v1")
INIT_FROM = ROOT / "data" / "render_mode_head" / "v1" / "model_best.pt"
OUT_DIR = ROOT / "data" / "render_mode_head" / "v2"

# A mode needs this many eval positives before its per-mode AP is reported as a CLAIM
# rather than a direction. 10 is where a single flipped pair stops moving AP by >0.1.
RICH_MIN_EVAL_GOOD = 10

log = logging.getLogger("train_mining_head_v2")


# --------------------------------------------------------------------------- #
# Data — one batch, labels and split side both IN-ROW.
# --------------------------------------------------------------------------- #
def load_rows(batch_dir: Path = BATCH_DIR) -> tuple[list[MRow], list[MRow]]:
    """(train, eval) as `MRow`s. Raises on an unlabeled row rather than dropping it.

    A silently smaller n reads exactly like a complete corpus, and this trainer's whole
    claim is that it saw the fresh sheet — all of it."""
    tr: list[MRow] = []
    ev: list[MRow] = []
    unlabeled: list[str] = []
    for line in (batch_dir / "images.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label"]["score"] is None:
            unlabeled.append(r["image_id"])
            continue
        lab = int(r["label"]["score"])
        if lab not in tuple(range(1, K + 1)):
            raise ValueError(f"{r['image_id']}: label {lab} out of 1..{K}")
        jpg = batch_dir / "crops" / f"{r['image_id']}.jpg"
        if not jpg.exists():
            raise FileNotFoundError(f"crop missing: {jpg}")
        pv = r["provenance"]
        row = MRow(r["image_id"], lab, jpg, pv["location_key"], pv["render_mode"],
                   pv["family"], r["render"]["fractal_type"])
        side = pv["split_side"]
        if side == "train":
            tr.append(row)
        elif side == "eval":
            ev.append(row)
        else:
            raise ValueError(f"{r['image_id']}: split_side {side!r} is neither train nor eval")
    if unlabeled:
        raise SystemExit(
            f"[mining-v2] {len(unlabeled)} rows still carry label.score = null "
            f"(e.g. {unlabeled[:3]}). This trainer needs the MERGED sheet.")
    span = {r.loc for r in tr} & {r.loc for r in ev}
    if span:
        raise AssertionError(f"{len(span)} locations span train+eval (e.g. {sorted(span)[:3]})")
    missing = sorted(set(MODES) - {r.mode for r in tr + ev})
    if missing:
        raise AssertionError(f"roster modes absent from the batch: {missing}")
    return tr, ev


def mode_tiers(ev: list[MRow]) -> dict[str, str]:
    """`mode -> "rich" | "directional"`, DERIVED from each mode's own eval-q3 count.

    v1 declared this list; declaring it again would carry the lost corpus's mode mass into a
    corpus with a different one. A mode below the floor is still reported — the tier only
    decides whether its number is quotable as a claim."""
    good = Counter(r.mode for r in ev if r.label >= K)
    return {m: ("rich" if good.get(m, 0) >= RICH_MIN_EVAL_GOOD else "directional")
            for m in MODES}


# --------------------------------------------------------------------------- #
# One training run (single seed). v1's loop, with init-from and >=3 selection.
# --------------------------------------------------------------------------- #
def train_one_seed(seed, tr, ev, args, device, train_tf, deploy_tf, cfg, seed_dir, init_state):
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # pretrained=False: the weights come from v1, and downloading an ImageNet checkpoint
    # only to overwrite every tensor of it is a network round-trip that can also fail.
    model = build_mining_model(args.drop_rate, args.drop_path_rate, pretrained=False)
    model.load_state_dict(init_state)          # strict — a shape mismatch is a moved pin
    model = model.to(device)

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = make_loader(tr, train_tf, args.batch_size, device, train=True,
                               num_workers=args.num_workers, seed=seed)
    eval_loader = make_loader(ev, deploy_tf, args.batch_size, device, train=False,
                              num_workers=min(4, args.num_workers))
    eval_labels = np.asarray([r.label for r in ev])

    # Epoch -1: v1's own read on THIS eval side, through this loop's scorer. Free (one
    # forward pass) and it is what makes "the finetune moved it" a measurement rather than
    # a comparison against a number computed somewhere else.
    cond0, marg0, sum0 = predict_all(model, eval_loader, len(ev), device)
    init_ap3 = _ap((eval_labels >= 3).astype(int), marg0[:, 1])
    init_ap2 = _ap((eval_labels >= 2).astype(int), marg0[:, 0])
    log.info(f"[seed {seed}] epoch -1 (v1 init, no training): "
             f"AP>=3 {init_ap3:.4f}  AP>=2 {init_ap2:.4f}")

    best_ap, best_epoch = -1.0, -1
    best_state = best_cond = best_marg = best_sum = None
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            loss = corn_loss(logits, (y - 1).long(), num_classes=K)
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
        # DEVIATION 3: selection is the >=3 boundary — the one the gate cuts at.
        sel = -1.0 if (ap_gd is None or not np.isfinite(ap_gd)) else ap_gd
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_good": sel, "ap_not_bad": float(ap_nb)})
        log.info(f"[seed {seed}] epoch {epoch:2d}  loss {train_loss:.4f}  "
                 f"AP_good {sel:.4f}  AP_nb {ap_nb:.4f}  ({time.time()-t0:.1f}s)")
        if sel > best_ap:
            best_ap, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
    log.info(f"[seed {seed}] best epoch {best_epoch}: good AP {best_ap:.4f} "
             f"(wall {time.time()-t_start:.0f}s)")

    seed_cfg = dict(cfg, seed=seed, best_epoch=best_epoch)
    torch.save({"state_dict": best_state, "config": seed_cfg}, seed_dir / "model_best.pt")
    with open(seed_dir / "eval_scores.jsonl", "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "mode": r.mode, "family": r.family,
                "fractal_type": r.fractal_type, "label": r.label,
                "p_ge2": float(best_marg[i, 0]), "p_ge3": float(best_marg[i, 1]),
                "p_not_bad": float(best_cond[i, 0]), "p_good_cond": float(best_cond[i, 1]),
                "score": float(best_sum[i]),
            }) + "\n")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"seed": seed, "best_epoch": best_epoch, "val_best_good_ap": float(best_ap),
            "init_ap_good": float(init_ap3), "init_ap_not_bad": float(init_ap2),
            "history": history, "checkpoint": str(seed_dir / "model_best.pt")}, \
        best_cond, best_marg, best_sum


# --------------------------------------------------------------------------- #
def build_parser():
    """The CLI, as a function, so a test can read the DEFAULTS and hold them against v1's
    checkpoint config. "Keep the committed trainer recipe otherwise" is only a claim until
    something compares the two sets of numbers."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Finetune the render-mode (mining) head from v1 on the fresh sheet.")
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
    ap.add_argument("--init-from", default=str(INIT_FROM))
    ap.add_argument("--batch-dir", default=str(BATCH_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    return ap


# The recipe knobs that must NOT drift from v1: the trainer's CLI default on the left, the
# key it is stored under in v1's checkpoint config on the right. Held equal by
# `classifier/test_train_mining_head_v2.py`; deviations from v1 are declared in
# `cfg["deviations_from_v1"]` and are deliberately absent from this map.
RECIPE_KNOBS = {"epochs": "epochs", "batch_size": "batch_size",
                "backbone_lr": "backbone_lr", "head_lr": "head_lr",
                "weight_decay": "weight_decay", "drop_rate": "drop_rate",
                "drop_path_rate": "drop_path_rate", "border_crop": "border_crop"}


def main():                                                             # noqa: PLR0915
    args = build_parser().parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    if len(seeds) < 3:
        raise SystemExit("need >=3 seeds for a measured band")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
             f"backbone={BACKBONE}  seeds={seeds}")

    # --- init weights (DEVIATION 2) ---------------------------------------- #
    init_path = Path(args.init_from)
    if not init_path.exists():
        raise SystemExit(f"--init-from does not exist: {init_path}. v2 is defined as a "
                         f"finetune OF v1; without v1's weights this run is a different "
                         f"experiment and must not be written to {out_dir}.")
    init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
    init_cfg = init_ck["config"]
    if init_cfg["backbone"] != BACKBONE or int(init_cfg["num_classes"]) != K:
        raise SystemExit(f"init checkpoint is {init_cfg['backbone']} K={init_cfg['num_classes']}, "
                         f"this trainer is {BACKBONE} K={K} — not a finetune.")
    init_state = init_ck["state_dict"]
    log.info(f"init from {init_path} (v1 seed {init_cfg.get('seed')}, "
             f"epoch {init_cfg.get('best_epoch')})")

    # --- data (DEVIATION 1: in-row split, never re-derived) ------------------ #
    batch_dir = Path(args.batch_dir)
    tr, ev = load_rows(batch_dir)
    log.info(f"train {len(tr)} renders / {len({r.loc for r in tr})} loc  {label_hist(tr)}")
    log.info(f"eval  {len(ev)} renders / {len({r.loc for r in ev})} loc  {label_hist(ev)}  "
             f"(location-disjoint OK)")
    eh = label_hist(ev)
    log.info(f"eval q3 (good) = {eh[K]}  [reads reported as mean +/- SD over seeds]")
    log.info(f"eval by-mode: {dict(sorted(Counter(r.mode for r in ev).items()))}")
    tiers = mode_tiers(ev)
    rich = [m for m in MODES if tiers[m] == "rich"]
    directional = [m for m in MODES if tiers[m] == "directional"]
    log.info(f"per-mode tiers (>= {RICH_MIN_EVAL_GOOD} eval-q3 = rich): "
             f"rich={rich} directional={directional}")
    log.info(f"modes v1's trainer never saw, included here (DEVIATION 4): "
             f"{list(TRAINER_DROPPED_V1)}")

    # --- config / transforms (v1's, unchanged) ------------------------------- #
    probe = build_mining_model(args.drop_rate, args.drop_path_rate, pretrained=False)
    data_cfg = data_config(probe)
    del probe
    log.info(f"data_config: {data_cfg}")
    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)
    cfg = {
        "model": "render_mode_head_v2", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=2, K pinned=3)", "geometry": "stretch",
        "label_unit": "render (image_id)",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "aug_rationale": "palette/mode/color IS the label (same as head-v3 and v1)",
        "class_weighting": "none", "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
        "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": "max eval good AP (marginal P>=3); full schedule (no early stop)",
        "split": "PRE-SPLIT in-row provenance.split_side (union-find over Julia-parent-"
                 "linked locations, seed 0); NOT re-derived",
        "dataset": str(batch_dir), "batch_id": batch_dir.name,
        "init": "finetune_from_render_mode_head_v1",
        "init_from": str(init_path),
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "gate": "marginal p_ge = cumprod(sigma); NEVER the CORN conditional",
        "seeds": seeds,
        "n_train": len(tr), "n_eval": len(ev),
        "modes": list(MODES), "n_modes": len(MODES),
        "deviations_from_v1": {
            "data": f"fresh sheet {batch_dir.name} (960 rows, in-row split_side); v1's "
                    f"dataset_v1/{{train,eval}}.jsonl did not survive the corpus loss",
            "init": "v1 model_best.pt, strict load_state_dict; v1 was imagenet_backbone_fresh",
            "selection": "eval AP at the >=3 boundary; v1 selected on >=2",
            "modes": f"all {len(MODES)} roster modes; v1's trainer dropped "
                     f"{list(TRAINER_DROPPED_V1)}",
            "per_mode_tiering": f"rich/directional DERIVED from each mode's eval-q3 count "
                                f"(>= {RICH_MIN_EVAL_GOOD}); v1 hardcoded the two lists",
        },
        "unchanged_from_v1": ["backbone", "num_classes", "loss", "geometry", "epochs",
                              "batch_size", "backbone_lr", "head_lr", "weight_decay",
                              "drop_rate", "drop_path_rate", "border_crop", "grad_clip",
                              "amp", "augmentation", "gate", "class_weighting"],
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    eval_labels = np.asarray([r.label for r in ev])
    eval_modes = np.asarray([r.mode for r in ev])

    # --- multi-seed finetune ------------------------------------------------- #
    per_seed = []
    overall_blocks = []
    mode_blocks = defaultdict(list)
    best_for_stage = None
    for seed in seeds:
        log.info(f"================= SEED {seed} =================")
        info, cond, marg, ssum = train_one_seed(
            seed, tr, ev, args, device, train_tf, deploy_tf, cfg, out_dir / f"seed_{seed}",
            init_state)
        per_seed.append(info)
        overall_blocks.append(eval_block(eval_labels, cond, marg, ssum, np.ones(len(ev), bool)))
        for mode in MODES:
            mode_blocks[mode].append(per_mode_good_ap(eval_labels, marg, eval_modes, mode))
        gd = info["val_best_good_ap"]
        if best_for_stage is None or gd > best_for_stage[0]:
            best_for_stage = (gd, seed, ssum, out_dir / f"seed_{seed}")
        (out_dir / "per_seed.json").write_text(json.dumps(per_seed, indent=2))

    # --- cross-seed aggregation ---------------------------------------------- #
    log.info("================= CROSS-SEED AGGREGATION =================")
    ov_keys = ["ap_not_bad", "ap_good", "auc_good_vs_rest", "auc_not_bad_vs_bad",
               "spearman_pge3_vs_tier", "spearman_score_vs_tier"]
    b0 = overall_blocks[0]
    overall_agg = {"n": b0["n"], "n_not_bad": b0["n_not_bad"], "n_good": b0["n_good"],
                   **{k: agg(overall_blocks, k) for k in ov_keys}}
    log.info(f"  [OVERALL] n={overall_agg['n']}  not_bad={overall_agg['n_not_bad']}  "
             f"good={overall_agg['n_good']}")
    log.info(f"     good-vs-rest AUC(p_ge3) {fmt(overall_agg['auc_good_vs_rest'])}   "
             f"not-bad AP {fmt(overall_agg['ap_not_bad'])}   good AP {fmt(overall_agg['ap_good'])}")
    log.info(f"     Spearman(p_ge3, tier) {fmt(overall_agg['spearman_pge3_vs_tier'])}   "
             f"Spearman(score, tier) {fmt(overall_agg['spearman_score_vs_tier'])}")

    conf_stack = np.array([b["confusion_true_x_pred"] for b in overall_blocks], dtype=float)
    mean_conf = conf_stack.mean(axis=0).round(1).tolist()
    log.info(f"     mean confusion(true x pred, 1..{K}) {mean_conf}")
    mean_by_tier = {t: float(np.mean([b["mean_score_by_tier"][t] for b in overall_blocks
                                      if b["mean_score_by_tier"][t] is not None]))
                    for t in range(1, K + 1)}
    log.info(f"     mean score by tier {mean_by_tier}")

    # --- per-mode q3 AP, all 15 ----------------------------------------------- #
    log.info("=== PER-MODE q3 AP (marginal p_ge3), all 15 roster modes ===")
    mode_agg = {}
    for mode in MODES:
        blks = mode_blocks[mode]
        b0m = next((b for b in blks if b is not None), None)
        mode_agg[mode] = {
            "tier": tiers[mode],
            "untrained_by_v1": mode in TRAINER_DROPPED_V1,
            "n": (b0m["n"] if b0m else 0), "n_good": (b0m["n_good"] if b0m else 0),
            "ap_good": agg(blks, "ap_good"), "auc_good_vs_rest": agg(blks, "auc_good_vs_rest"),
        }
        m = mode_agg[mode]
        log.info(f"  {'RICH' if m['tier'] == 'rich' else 'DIR '}"
                 f"{'*' if m['untrained_by_v1'] else ' '}[{mode:32s}] "
                 f"n={m['n']:3d} good={m['n_good']:2d}  AP_good {fmt(m['ap_good'])}  "
                 f"AUC {fmt(m['auc_good_vs_rest'])}"
                 f"{'   (directional only)' if m['tier'] != 'rich' else ''}")

    # --- stage the best-good-AP seed as v2/model_best.pt ---------------------- #
    stage_ap, stage_seed, s_sum, s_dir = best_for_stage
    shutil.copy(s_dir / "model_best.pt", out_dir / "model_best.pt")
    shutil.copy(s_dir / "eval_scores.jsonl", out_dir / "eval_scores.jsonl")

    auc = overall_agg["auc_good_vs_rest"]
    ap_g = overall_agg["ap_good"]
    good_base = overall_agg["n_good"] / max(overall_agg["n"], 1)
    init_ap = [p["init_ap_good"] for p in per_seed]
    separates = (auc["mean"] is not None and auc["mean"] > 0.60
                 and ap_g["mean"] is not None and ap_g["mean"] > 1.3 * good_base)
    verdict = (
        f"{'SEPARATES' if separates else 'WEAK'}: overall good-vs-rest AUC {fmt(auc)} and "
        f"good AP {fmt(ap_g)} vs base {good_base:.3f}. v1's own AP>=3 on this eval side "
        f"(epoch -1, before any finetune step) was {np.mean(init_ap):.3f}. "
        f"Adoption is NOT decided here — see tools/mining/mining_v2_reads.py.")

    metrics = {
        "seeds": seeds, "backbone": BACKBONE, "num_classes": K,
        "batch_id": batch_dir.name,
        "train_n": len(tr), "eval_n": len(ev), "eval_tier_hist": eh,
        "eval_good_base_rate": float(good_base),
        "init_from": str(init_path),
        "v1_init_eval_ap_good": {"per_seed": init_ap, "mean": float(np.mean(init_ap))},
        "overall": overall_agg,
        "mean_confusion_true_x_pred": mean_conf,
        "mean_score_by_tier": mean_by_tier,
        "per_mode_good_ap": mode_agg,
        "per_mode_tier_rule": f"rich iff eval-q3 count >= {RICH_MIN_EVAL_GOOD}",
        "modes_untrained_by_v1": list(TRAINER_DROPPED_V1),
        "per_seed": per_seed,
        "staged": {"seed": stage_seed, "good_ap": float(stage_ap),
                   "checkpoint": str(out_dir / "model_best.pt"),
                   "rule": "best per-seed eval good AP (>=3 boundary)"},
        "deviations_from_v1": cfg["deviations_from_v1"],
        "separates": bool(separates), "verdict": verdict,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info("================= RENDER-MODE (MINING) HEAD v2 SUMMARY =================")
    log.info(f"  seeds={seeds}  train n={len(tr)}  eval n={len(ev)}  "
             f"eval-good={overall_agg['n_good']} (base {good_base:.3f})")
    log.info(f"  v1 init on this eval side: AP>=3 {np.mean(init_ap):.4f}")
    log.info(f"  OVERALL  good-vs-rest AUC {fmt(auc)}  not-bad AP {fmt(overall_agg['ap_not_bad'])}  "
             f"good AP {fmt(ap_g)}  Spearman(p_ge3) {fmt(overall_agg['spearman_pge3_vs_tier'])}")
    for m in [x for x in MODES if mode_agg[x]["tier"] == "rich"]:
        log.info(f"  RICH {m:32s} good AP {fmt(mode_agg[m]['ap_good'])} "
                 f"(n_good={mode_agg[m]['n_good']})")
    log.info(f"  STAGED -> {out_dir / 'model_best.pt'}  (seed {stage_seed}, good AP {stage_ap:.3f})")
    log.info(f"  VERDICT: {verdict}")
    log.info("DONE")


if __name__ == "__main__":
    main()
