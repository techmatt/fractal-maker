#!/usr/bin/env python
"""Cost smoke for the backbone comparison — run BEFORE committing to a schedule.

Answers three questions per arm, cheaply (no data, no cache, ~20 s each):
  1. Does it BUILD at the frozen deploy geometry (224x384) with pretrained weights, and
     what does timm resolve as its normalization? (A ViT also has to resample its
     pos-embed here, and this is where that fails if it is going to.)
  2. Params, and PEAK TRAIN VRAM at the inherited batch size. 8 GB is the constraint the
     prompt says to trim against — an arm that OOMs is DROPPED, not shrunk, because the
     batch size is part of the frozen recipe.
  3. Step time, forward+backward, fp32 (the recipe has amp="off"), on synthetic tensors —
     the GPU-only cost, which is the half that varies by arm.

The projection is `max(gpu, cpu_floor)`, NOT `gpu`. v11's real epoch was 76.0 s for 8,443
locations = 111 img/s, and the train transform decodes a JPG, crops, resizes, RE-ENCODES a
JPG and normalizes on 4 workers — so the control is CPU-bound and its measured epoch is a
FLOOR every arm inherits. Projecting from GPU time alone would over-state a heavy arm's
cost and under-state the total schedule's sensitivity to it.

  uv run python tools/backbone_search/cost_smoke.py            # all arms
  uv run python tools/backbone_search/cost_smoke.py --arms vit_small_p16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import paths  # noqa: E402
from backbone_search.arms import ARMS, ARMS_BY_NAME, V11_CKPT  # noqa: E402

from classifier.model import build_model, data_config  # noqa: E402

# v11's measured epoch (data/classifier/v11/train.log, 40 epochs, 76.0 +- 0.2 s) over its
# 8,443 train locations. The CPU-side floor of any arm on this box, at this recipe.
CONTROL_EPOCH_S = 76.0
TRAIN_LOCS = 8443
OUT = "scratch/backbone_search/cost_smoke.json"


def measure(arm, cfg, device, steps=12, warmup=4):
    K = int(cfg["num_classes"])
    bs = int(cfg["batch_size"])
    h, w = 224, 384                                  # the frozen deploy geometry
    t0 = time.time()
    model = build_model(target="ordinal", drop_rate=cfg["drop_rate"],
                        drop_path_rate=cfg["drop_path_rate"], pretrained=True,
                        num_classes=K, backbone=arm.timm_model,
                        backbone_kwargs=arm.create_kwargs or None)
    build_s = time.time() - t0
    if arm.grad_checkpointing:
        model.set_grad_checkpointing(True)
    dcfg = data_config(model)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, 3, h, w, device=device)
    y = torch.randint(0, K, (bs,), device=device)
    from classifier.model import compute_loss

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    model.train()
    for i in range(warmup + steps):
        if i == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = compute_loss(model(x).float(), y + 1, "ordinal", num_classes=K)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    train_step_s = (time.time() - t0) / steps
    peak_train_mb = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None

    # inference-side (score) rate at the same geometry, batch 64 like the eval loader
    model.eval()
    xi = torch.randn(64, 3, h, w, device=device)
    with torch.no_grad():
        for i in range(warmup + steps):
            if i == warmup:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            model(xi)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_img_s = 64 * steps / (time.time() - t0)

    del model, opt, x, xi
    if device == "cuda":
        torch.cuda.empty_cache()

    steps_per_epoch = -(-TRAIN_LOCS // bs)
    gpu_epoch_s = train_step_s * steps_per_epoch
    return {
        "arm": arm.name, "timm_model": arm.timm_model, "pretrain": arm.pretrain,
        "grad_checkpointing": arm.grad_checkpointing,
        "params_m": round(n_params / 1e6, 2), "build_s": round(build_s, 1),
        "data_config": dcfg, "batch_size": bs,
        "train_step_s": round(train_step_s, 4),
        "peak_train_vram_mb": None if peak_train_mb is None else round(peak_train_mb),
        "gpu_epoch_s": round(gpu_epoch_s, 1),
        "cpu_floor_epoch_s": CONTROL_EPOCH_S,
        "proj_epoch_s": round(max(gpu_epoch_s, CONTROL_EPOCH_S), 1),
        "proj_train_h": round(max(gpu_epoch_s, CONTROL_EPOCH_S) * int(cfg["epochs"]) / 3600, 2),
        "gpu_infer_img_s": round(infer_img_s, 1),
        "gpu_score_s_per_1k": round(1000 / infer_img_s, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    arms = [ARMS_BY_NAME[n] for n in a.arms] if a.arms else list(ARMS)

    cfg = torch.load(V11_CKPT, map_location="cpu", weights_only=False)["config"]
    print(f"recipe: batch {cfg['batch_size']}  epochs {cfg['epochs']}  K {cfg['num_classes']}  "
          f"amp {cfg['amp']}  device {a.device}")
    rows, failed = [], []
    for arm in arms:
        try:
            r = measure(arm, cfg, a.device)
        except Exception as e:                                   # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:300]
            print(f"  {arm.name:20s} FAILED  {msg}")
            failed.append({"arm": arm.name, "timm_model": arm.timm_model, "error": msg})
            if a.device == "cuda":
                torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"  {arm.name:20s} {r['params_m']:6.2f}M  step {r['train_step_s']*1000:6.1f}ms  "
              f"vram {r['peak_train_vram_mb']:5d}MB  epoch~{r['proj_epoch_s']:6.1f}s  "
              f"train~{r['proj_train_h']:4.2f}h  score {r['gpu_score_s_per_1k']:5.2f}s/1k  "
              f"interp={r['data_config']['interpolation']} mean={r['data_config']['mean'][0]:.3f}")

    total = sum(r["proj_train_h"] for r in rows)
    print(f"\nround-1 projected total (1 seed x {len(rows)} arms): {total:.2f} h")
    out = paths.scratch(*OUT.split("/")[1:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"recipe_epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
                               "train_locations": TRAIN_LOCS,
                               "control_epoch_s_measured": CONTROL_EPOCH_S,
                               "arms": rows, "failed": failed,
                               "round1_total_h": round(total, 2)}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
