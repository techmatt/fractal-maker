#!/usr/bin/env python
"""Train ONE backbone-comparison arm — v11's recipe verbatim, one variable moved.

The design law of `prompts/backbone_search_v1.md` is that the backbone is the ONLY thing
that differs between arms, so this trainer restates no hyperparameter: it reads v11's
config out of `data/classifier/v11/model_best.pt["config"]` — the same read
`train_v11` does of v10's — and asserts every behavioural key survives into the arm config
unchanged. The three keys an arm may legitimately move are declared in `_ARM_KEYS`, and
`seed` is one of them because round 2 scores a BAND across seeds rather than a point.

WHAT IS FROZEN AND WHY IT IS NOT OBVIOUS. The deploy transform is bit-pinned, so
`geometry` (stretch to 384x224) AND `interpolation` (bicubic) are taken from v11's config
rather than from the arm's own `timm` data config — an arm whose pretrain_cfg says
`bilinear` would otherwise resize differently and the comparison would be reading two
image pipelines. `mean`/`std` DO follow the arm: normalization is a property of the
pretrained weights (feeding in1k statistics to a 0.5/0.5-normalized checkpoint is a
handicap, not a control), and it is downstream of the pixels the deploy path produces.
Both halves are stamped into the arm's config.json.

The SELECTION objective is also a controlled variable: the checkpoint pick is
max not-bad AP over v8/v9/v10/v11's frozen census+floor 670 and nothing else. The
1,810-row holdout, the uniform-90 and the q4-uniform-290 touch neither training nor the
pick, and are read only at eval time (`eval_arms.py`).

  uv run python tools/backbone_search/train_arm.py --arm convnextv2_tiny --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import paths  # noqa: E402
from backbone_search.arms import ARMS_BY_NAME, V11_CKPT  # noqa: E402

from classifier.data_v11 import load_locations_v11  # noqa: E402
from classifier.data_v4 import hist, make_weighted_sampler  # noqa: E402
from classifier.model import build_model, data_config  # noqa: E402
from classifier.train_v2 import detect_device, set_seed  # noqa: E402
from classifier.train_v8 import (CENSUS_SOURCE, FLOOR_SOURCE, cutpoint_positive_counts,  # noqa: E402
                                 train_resumable)

SELECTION_SOURCES = (CENSUS_SOURCE, FLOOR_SOURCE)
SELECTION_N = 670                     # v8/v9/v10/v11's selection population, to the row
log = logging.getLogger("train_arm")

# Outputs of the v11 run, not inputs to an arm.
_RUN_OUTPUT_KEYS = ("best_epoch",)
# Provenance v11 stamped about ITSELF — free text, carries no behaviour.
_PROVENANCE_KEYS = ("cache_manifest", "corpus_version", "init", "recipe_vs_v7",
                    "recipe_vs_v8", "recipe_vs_v9", "recipe_vs_v10", "maxiter_policy",
                    "cap_doc", "corpus_note", "selection_population", "split_rule",
                    "aug_recipe", "loss", "selection")
# The ONLY keys an arm may move, and each is the experiment or a consequence of it.
_ARM_KEYS = ("backbone",          # THE variable
             "backbone_kwargs",   # create-time kwargs the variable forces (ViT img_size)
             "seed",              # round 2 scores a band across seeds
             "mean", "std",       # normalization belongs to the pretrained weights
             "input_size",        # what timm reports for the arm; deploy geometry is frozen
             "grad_checkpointing",   # memory-time trade; same gradients, same optimization
             "arm", "arm_pretrain", "arm_why", "frozen_from", "round")


def assert_recipe_untouched(v11: dict, arm_cfg: dict) -> list:
    drift = [(k, v, arm_cfg.get(k)) for k, v in v11.items()
             if k not in _RUN_OUTPUT_KEYS and k not in _PROVENANCE_KEYS
             and k not in _ARM_KEYS and arm_cfg.get(k) != v]
    if drift:
        raise SystemExit(f"arm config DRIFTED from v11 on behavioural keys — the comparison "
                         f"would measure the drift, not the backbone:\n  {drift}")
    return sorted(k for k in v11 if k not in _RUN_OUTPUT_KEYS and k not in _PROVENANCE_KEYS
                  and k not in _ARM_KEYS)


def build_arm_config(v11: dict, arm, seed: int, arm_data_cfg: dict, rnd: int) -> dict:
    cfg = {k: v for k, v in v11.items() if k not in _RUN_OUTPUT_KEYS}
    cfg["backbone"] = arm.timm_model
    cfg["backbone_kwargs"] = dict(arm.create_kwargs) or None
    cfg["seed"] = int(seed)
    cfg["grad_checkpointing"] = bool(arm.grad_checkpointing)
    cfg["mean"], cfg["std"] = arm_data_cfg["mean"], arm_data_cfg["std"]
    cfg["input_size"] = arm_data_cfg["input_size"]
    # geometry + interpolation are v11's, NOT the arm's — the deploy transform is pinned.
    cfg["interpolation"] = v11["interpolation"]
    cfg["arm"], cfg["arm_pretrain"], cfg["arm_why"] = arm.name, arm.pretrain, arm.why
    cfg["round"] = rnd
    cfg["frozen_from"] = ("data/classifier/v11/model_best.pt[config] — every behavioural key "
                          "read verbatim and asserted unchanged (train_arm.assert_recipe_"
                          "untouched). Moved: backbone, its create kwargs, the seed, and the "
                          "normalization/input_size that belong to the pretrained weights.")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Train one backbone-comparison arm.")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS_BY_NAME))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="SMOKE ONLY — a run with a "
                    "non-inherited epoch count stamps itself bounded_run and is not "
                    "comparable to any other arm")
    ap.add_argument("--limit-locations", type=int, default=None,
                    help="SMOKE ONLY — train on the first N locations; stamps bounded_run")
    a = ap.parse_args()

    arm = ARMS_BY_NAME[a.arm]
    bounded = bool(a.epochs or a.limit_locations)
    if bounded:
        # A bounded rehearsal writes REAL files, so it writes them where nothing reads a
        # comparison from: it is stamped bounded_run AND parked in scratch, rather than
        # sitting in the tracked record tree looking like an arm.
        out_dir = rec_dir = paths.scratch("backbone_search", "bounded", arm.name, f"s{a.seed}")
    else:
        out_dir = arm.weights_dir(a.seed)      # bulk: weights + resume, out of tree
        rec_dir = arm.record_dir(a.seed)       # durable: the record that survives them
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    # The log goes beside the WEIGHTS, not into the tracked record: `metrics.json` already
    # carries every epoch's loss and per-cutpoint AP as `history`, so a tracked train.log
    # would be a second copy of the same rows in a form nothing reads — and
    # `!/data/backbone_search/` would commit it (tests/test_large_tracked_blobs.py refuses
    # an undeclared `.log`, correctly).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(a.device)
    v11 = torch.load(V11_CKPT, map_location="cpu", weights_only=False)["config"]
    K = int(v11["num_classes"])

    # The probe answers what the ARM's pretrained weights want for normalization; it is
    # also where a backbone that cannot take the frozen geometry fails, before any data.
    probe = build_model(target="ordinal", pretrained=True, num_classes=K,
                        backbone=arm.timm_model, backbone_kwargs=arm.create_kwargs or None)
    arm_data_cfg = data_config(probe)
    n_params = sum(p.numel() for p in probe.parameters())
    del probe

    cfg = build_arm_config(v11, arm, a.seed, arm_data_cfg, a.round)
    # The drift guard runs on the UNBOUNDED config, then the rehearsal override is applied
    # on top — so a bounded run still proves the arm's real recipe passes the guard, and
    # `epochs` never becomes an exempt key for a run that claims to be comparable.
    inherited = assert_recipe_untouched(v11, cfg)
    if a.epochs:
        cfg["epochs"] = int(a.epochs)
    set_seed(int(cfg["seed"]))

    log.info(f"=== ARM {arm.name} seed {a.seed} round {a.round} ===")
    log.info(f"  backbone {arm.timm_model}  ({n_params/1e6:.2f}M params)  device {device}")
    log.info(f"  {len(inherited)} behavioural keys inherited unchanged from v11: {inherited}")
    log.info(f"  deploy geometry FROZEN: {cfg['geometry']} / {cfg['interpolation']} -> "
             f"{cfg['target_dims']};  arm normalization mean={arm_data_cfg['mean']}")
    if bounded:
        log.warning("  BOUNDED RUN (--epochs/--limit-locations) — stamped bounded_run, NOT "
                    "comparable to a full arm")

    # data_cfg drives the Transform inside train_resumable: interpolation is v11's, the
    # normalization is the arm's.
    data_cfg = dict(arm_data_cfg)
    data_cfg["interpolation"] = cfg["interpolation"]

    t_load = time.time()
    locs = load_locations_v11(verify_paths=False)
    log.info(f"  loaded {len(locs)} locations x {len(locs[0].renders)} tiles "
             f"in {time.time()-t_load:.0f}s")
    train_locs = [l for l in locs if l.split == "train"]
    eval_locs = [l for l in locs if l.split == "eval"]
    if a.limit_locations:
        train_locs = train_locs[:a.limit_locations]
    sel_locs = [l for l in eval_locs if l.source in SELECTION_SOURCES]
    if not bounded and len(sel_locs) != SELECTION_N:
        raise SystemExit(f"selection population is {len(sel_locs)}, expected {SELECTION_N} — "
                         f"the objective is not v11's and the arms are not comparable")
    assert all(l.eval_role == "instrument" and not l.biased for l in sel_locs), \
        "a selection location is not an unbiased instrument"
    log.info(f"  train {len(train_locs)} {hist(train_locs)}   selection {len(sel_locs)} "
             f"(census+floor, frozen)   other eval {len(eval_locs)-len(sel_locs)} (unseen)")

    sampler, mass_table = make_weighted_sampler(train_locs, beta=cfg["beta_biased"],
                                                class_balance=cfg["class_balance"])
    eval_canon = [l.canonical() for l in sel_locs]
    eval_labels = np.asarray([l.label for l in sel_locs])

    log.info(f"=== TRAIN {len(train_locs)} loc/epoch, batch {cfg['batch_size']}, "
             f"{cfg['epochs']} epochs (patience {cfg['patience']}) ===")
    t0 = time.time()
    _best_state, best_epoch, best_val_ap, history, ckpt_cfg = train_resumable(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    wall = time.time() - t0
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {wall:.0f}s = {wall/3600:.2f}h) ===")

    cfg["best_epoch"] = best_epoch
    cfg["bounded_run"] = bounded
    metrics = {
        "arm": arm.name, "seed": int(a.seed), "round": int(a.round),
        "backbone": arm.timm_model, "params_m": round(n_params / 1e6, 3),
        "bounded_run": bounded, "best_epoch": best_epoch,
        "val_best_not_bad_ap": best_val_ap, "train_wall_s": round(wall, 1),
        "epoch_wall_s_median": round(float(np.median([h.get("epoch_s", np.nan)
                                                      for h in history])), 1)
        if history and "epoch_s" in history[0] else None,
        "n_epochs_run": len(history), "train_n": len(train_locs),
        "selection_n": len(sel_locs), "class_counts_train": hist(train_locs),
        "cutpoint_positive_counts_train": {
            name: npos for name, _t, npos in
            cutpoint_positive_counts([l.label for l in train_locs], K)},
        "peak_train_vram_mb": (round(torch.cuda.max_memory_allocated() / 2**20)
                               if device == "cuda" else None),
        "mass_table": mass_table, "history": history,
        "weights": {"best": str(out_dir / "model_best.pt"),
                    "last": str(out_dir / "model_last.pt")},
        "fractal_type_train": dict(Counter(l.fractal_type for l in train_locs)),
    }
    (rec_dir / "config.json").write_text(json.dumps(ckpt_cfg, indent=2))
    (rec_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"DONE — record {rec_dir}, weights {out_dir}. Nothing is pinned; "
             f"ACTIVE_CKPT untouched.")


if __name__ == "__main__":
    main()
