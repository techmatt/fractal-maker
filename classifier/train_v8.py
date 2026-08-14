"""Train v8 — the 1..4 ordinal extension of the location-quality classifier.

v8 = the v7 recipe VERBATIM, with ONE change forced by the label scale: the CORN ordinal
head grows from 2 cutpoints (1..3) to **3 cutpoints (1..4)**. Class 4 = "exceptional
wallpaper emission". Everything else — backbone, two-group AdamW, cosine schedule,
grad-clip, fp32 (no AMP/EMA), the deploy/train Transform, the biased WeightedRandomSampler,
and the per-epoch resumable checkpoint machinery — is inherited unchanged.

THE RECIPE IS READ, NOT RESTATED. The hyperparameters come from the v7 config embedded in
`data/classifier/v7/model_best.pt["config"]` (the only surviving copy of the v7 recipe;
`data/v7/` was cleared). We override exactly three things: num_classes 3->4, the loss/label
strings, and the data source (data/v8 cache). If a knob is not in the v7 config we do not
invent one.

WHAT CHANGES FOR K=4 (all mechanical):
  * head emits K-1 = 3 logits (`build_model(num_classes=4)`);
  * `compute_loss(..., num_classes=4)` -> `corn_loss` runs 3 conditional-subset tasks;
  * scoring/derive are written K-agnostic here (train_v4's are hardwired to 2 logits);
  * selection is UNCHANGED: max eval not-bad AP = AP of (label>=2) ranked by sigma(logit0).

Cutpoint semantics (0-based k): logit_k = P(rank > k) = P(label >= k+2). So
  cut0 = P(label>=2)  (not-bad),  cut1 = P(label>=3)  (q3/"good"),  cut2 = P(label>=4)  (q4).
score = sum_k sigma(logit_k) in [0,3] is the monotone rank score used for AP/selection.

RESUMABLE. An external reaper kills long runs silently (observed on v7). Every epoch snapshots
{model, opt, sched, best-tracking, history, RNG} atomically to out_dir/resume.pt; a relaunch
continues from the next epoch. A kill costs at most one epoch. A clean run deletes resume.pt.

**ACTIVE_CKPT is NOT switched and t_good is NOT set here** — v7 stays the deployed scorer
until v8 is measured (evaluate step). This trainer only writes under data/classifier/v8/.

  uv run python -m classifier.train_v8
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Transform
from .data_v4 import LocationDataset, hist, load_locations, make_weighted_sampler
from .eval import _ap
from .model import BACKBONE, build_model, compute_loss, data_config
from .train_v2 import detect_device, set_seed
from .train_v4 import _RenderSet  # K-agnostic render scoring reuses the dataset only

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v8"
V7_CKPT = ROOT / "data" / "classifier" / "v7" / "model_best.pt"
V8_CACHE = ROOT / "data" / "v8" / "cache_manifest.jsonl"
NUM_CLASSES = 4                       # 1..4 ordinal; K-1 = 3 cutpoints
CENSUS_SOURCE = "prospect_census"     # the pinned primary eval instrument
FLOOR_SOURCE = "loose0_v3_floor"      # the mandelbrot eval floor
log = logging.getLogger("train_v8")


# --------------------------------------------------------------------------- #
# K-agnostic scoring (train_v4.score_renders/derive are hardwired to 2 logits).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_renders_k(model, renders, deploy_tf, device, n_logits,
                    batch_size=64, num_workers=4):
    """Logits (N, n_logits) aligned to `renders` order, through the deterministic deploy xform."""
    model.eval()
    loader = DataLoader(_RenderSet(renders, deploy_tf), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    out = np.zeros((len(renders), n_logits), dtype=np.float64)
    for x, idx in loader:
        out[idx.numpy()] = model(x.to(device, non_blocking=True)).float().cpu().numpy()
    del loader
    return out


def derive_k(logits: np.ndarray):
    """logits (N, K-1) -> (probs (N, K-1), score_sum (N,)). probs[:,k] = P(label >= k+2).
    Stable sigmoid (no exp-overflow warning on large-magnitude logits)."""
    from scipy.special import expit
    p = expit(np.asarray(logits, dtype=np.float64))
    return p, p.sum(axis=1)


def cutpoint_positive_counts(labels, K):
    """Effective positive count at each CORN cutpoint: cut k (0-based) has positives
    (label >= k+2). Returns list of (name, threshold_label, n_pos)."""
    labels = np.asarray(labels)
    out = []
    for k in range(K - 1):
        thr = k + 2
        out.append((f"cut{k} (P>={thr})", thr, int((labels >= thr).sum())))
    return out


def load_v7_recipe():
    """Read the v7 recipe out of its config (embedded in model_best.pt). Raises if absent —
    v8 does not invent hyperparameters."""
    if not V7_CKPT.exists():
        raise SystemExit(f"v7 config source missing: {V7_CKPT} (the v8 recipe is read from it)")
    ck = torch.load(V7_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    if not cfg:
        raise SystemExit(f"{V7_CKPT} has no embedded 'config' — cannot read the v7 recipe")
    return cfg


def train_resumable(train_locs, eval_renders, eval_labels, cfg, data_cfg, device,
                    sampler, out_dir: Path):
    """v7's train_resumable, K-generalized. Selection = max eval not-bad AP (sigma(logit0));
    per-epoch atomic resume snapshot."""
    K = cfg["num_classes"]
    n_logits = K - 1
    train_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                         data_cfg["std"], train=True,
                         jpeg_q=(85, 95) if not cfg["no_jpeg_aug"] else None)
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)
    train_ds = LocationDataset(train_locs, train_tf, seed=cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                              num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
                              persistent_workers=False, drop_last=False)

    # `backbone`/`backbone_kwargs` are absent from every v8..v11 cfg dict except as the
    # default name, so this builds what it always built; the backbone comparison
    # (tools/backbone_search/) is what puts a different name in cfg.
    model = build_model(target="ordinal", drop_rate=cfg["drop_rate"],
                        drop_path_rate=cfg["drop_path_rate"], pretrained=True,
                        num_classes=K, backbone=cfg.get("backbone"),
                        backbone_kwargs=cfg.get("backbone_kwargs")).to(device)
    if cfg.get("grad_checkpointing"):
        # Absent from every v8..v11 cfg. A memory-time trade only: the recomputed graph
        # yields the same gradients, so the recipe is untouched and only wall clock moves.
        model.set_grad_checkpointing(True)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["backbone_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    eval_labels = np.asarray(eval_labels)
    best_metric, best_state, best_epoch, history = -1.0, None, -1, []
    since_improve = 0
    start_epoch = 0

    resume_path = out_dir / "resume.pt"
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best_metric, best_epoch = ck["best_metric"], ck["best_epoch"]
        best_state = ck["best_state"]; since_improve = ck["since_improve"]
        history = ck["history"]; start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        log.info(f"  RESUMED from {resume_path}: continuing at epoch {start_epoch} "
                 f"(best so far {best_metric:.4f} @ epoch {best_epoch})")

    for epoch in range(start_epoch, cfg["epochs"]):
        train_ds.set_epoch(epoch)
        model.train(); t0 = time.time(); running = 0.0; nseen = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(logits.float(), y, "ordinal", num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0); nseen += x.size(0)
        sched.step()
        train_loss = running / max(nseen, 1)

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"  NaN/Inf at epoch {epoch} — aborting"); break

        logits = score_renders_k(model, eval_renders, deploy_tf, device, n_logits,
                                 batch_size=cfg["batch_size"], num_workers=0)
        probs, _ = derive_k(logits)
        ap = [_ap((eval_labels >= (k + 2)).astype(int), probs[:, k]) for k in range(n_logits)]
        sel = -1.0 if (ap[0] is None or not np.isfinite(ap[0])) else ap[0]
        rec = {"epoch": epoch, "train_loss": train_loss, "val_ap_not_bad": sel}
        for k in range(n_logits):
            rec[f"val_ap_cut{k}_ge{k+2}"] = ap[k]
        history.append(rec)
        apstr = "  ".join(f"AP>={k+2} {('n/a' if ap[k] is None else f'{ap[k]:.4f}')}"
                          for k in range(n_logits))
        log.info(f"  epoch {epoch:2d}  loss {train_loss:.4f}  {apstr}  ({time.time()-t0:.1f}s)")

        if sel > best_metric:
            best_metric, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg["patience"]:
                log.info(f"  early stop at {epoch} (best {best_epoch}, "
                         f"val_AP_notbad {best_metric:.4f})")
                break

        tmp = out_dir / "resume.pt.tmp"
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "best_metric": best_metric,
            "best_epoch": best_epoch, "best_state": best_state,
            "since_improve": since_improve, "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "numpy_rng": np.random.get_state(),
        }, tmp)
        tmp.replace(resume_path)

    last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:  # never improved (degenerate); fall back to last
        best_state = last_state
    ckpt_cfg = dict(cfg)
    ckpt_cfg.update({"backbone": cfg.get("backbone") or BACKBONE,
                     "mean": data_cfg["mean"], "std": data_cfg["std"],
                     "interpolation": data_cfg["interpolation"],
                     "input_size": data_cfg["input_size"], "best_epoch": best_epoch})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": ckpt_cfg}, out_dir / "model_best.pt")
    torch.save({"state_dict": last_state, "config": ckpt_cfg}, out_dir / "model_last.pt")
    if resume_path.exists():
        resume_path.unlink()

    del train_loader, opt, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_state, best_epoch, best_metric, history, ckpt_cfg


def main():
    ap = argparse.ArgumentParser(description="Train v8 (1..4 ordinal, 3 cutpoints).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the v7 epoch count (default: inherit from v7 config)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(a.device)
    v7 = load_v7_recipe()
    set_seed(int(v7["seed"]))
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    log.info(f"recipe read from v7 config: {V7_CKPT}")

    # v8 cfg = v7 recipe, verbatim, with only the label-scale deltas.
    cfg = {
        "target": "ordinal", "num_classes": NUM_CLASSES,
        "geometry": v7["geometry"], "epochs": int(a.epochs or v7["epochs"]),
        "batch_size": int(v7["batch_size"]),
        "backbone_lr": float(v7["backbone_lr"]), "head_lr": float(v7["head_lr"]),
        "weight_decay": float(v7["weight_decay"]),
        "drop_rate": float(v7["drop_rate"]), "drop_path_rate": float(v7["drop_path_rate"]),
        "patience": int(v7["patience"]), "seed": int(v7["seed"]),
        "num_workers": int(v7["num_workers"]),
        "amp": v7.get("amp", "off"), "grad_clip": float(v7["grad_clip"]),
        "no_jpeg_aug": bool(v7["no_jpeg_aug"]),
        "beta_biased": float(v7["beta_biased"]), "class_balance": v7["class_balance"],
        "black_thresh": float(v7["black_thresh"]),
        "src_dims": v7.get("src_dims", [512, 288]), "target_dims": v7.get("target_dims", [384, 224]),
        "init": "imagenet_backbone_fresh (NOT warm-started); recipe inherited from v7 config",
        "loss": f"CORN ordinal (K-1={NUM_CLASSES-1})",
        "sampler": v7.get("sampler"),
        "selection": "max eval not-bad AP (rank by sigma(logit0)); patience==epochs (full schedule)",
        "eval_split_is_val": True,
        "cache_manifest": "data/v8/cache_manifest.jsonl",
        "recipe_vs_v7": "IDENTICAL recipe read from v7 config; only delta = 1..4 head (3 cutpoints)",
    }

    locs = load_locations(cache_path=V8_CACHE)
    train_locs = [l for l in locs if l.split == "train"]
    eval_locs = [l for l in locs if l.split == "eval"]
    assert all(not l.biased for l in eval_locs), "eval split must be unbiased-only"
    ftypes = Counter(l.fractal_type for l in locs)
    log.info(f"locations: {len(locs)} (train {len(train_locs)} {hist(train_locs)}, "
             f"eval {len(eval_locs)} {hist(eval_locs)})  fractal_type={dict(ftypes)}")
    for ft in sorted(ftypes):
        tr = [l for l in train_locs if l.fractal_type == ft]
        ev = [l for l in eval_locs if l.fractal_type == ft]
        log.info(f"  {ft:18s}: train {len(tr):4d} {hist(tr)}  eval {len(ev):3d} {hist(ev)}")
    n_census = sum(1 for l in eval_locs if l.source == CENSUS_SOURCE)
    n_floor = sum(1 for l in eval_locs if l.source == FLOOR_SOURCE)
    log.info(f"  eval instruments: census(julia:mb)={n_census}  mandelbrot-floor={n_floor}")

    # --- effective positive count at each cutpoint (the report line) ---
    tr_labels = [l.label for l in train_locs]
    log.info("=== effective positive count at each cutpoint (TRAIN) ===")
    for name, thr, npos in cutpoint_positive_counts(tr_labels, NUM_CLASSES):
        log.info(f"  {name:14s}: {npos} train locations  ({100*npos/len(tr_labels):.1f}%)")

    sampler, mass_table = make_weighted_sampler(train_locs, beta=cfg["beta_biased"],
                                                class_balance=cfg["class_balance"])
    log.info(f"=== sampled mass (beta={cfg['beta_biased']}, class_balance={cfg['class_balance']}) ===")
    log.info(f"  class_count={mass_table['class_count']}  "
             f"w_class={ {k: round(v,4) for k,v in mass_table['w_class'].items()} }")

    probe = build_model(target="ordinal", pretrained=True, num_classes=NUM_CLASSES)
    data_cfg = data_config(probe); del probe
    log.info(f"data_config: {data_cfg}")

    eval_canon = [l.canonical() for l in eval_locs]
    eval_labels = np.asarray([l.label for l in eval_locs])

    log.info(f"=== TRAIN: {len(train_locs)} loc/epoch, batch {cfg['batch_size']}, "
             f"{cfg['epochs']} epochs (patience {cfg['patience']}) ===")
    t_start = time.time()
    best_state, best_epoch, best_val_ap, history, ckpt_cfg = train_resumable(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    cfg["best_epoch"] = best_epoch
    metrics = {"best_epoch": best_epoch, "val_best_not_bad_ap": best_val_ap,
               "eval_split_n": len(eval_locs), "census_n": n_census, "floor_n": n_floor,
               "cutpoint_positive_counts_train": {
                   name: npos for name, _thr, npos in
                   cutpoint_positive_counts(tr_labels, NUM_CLASSES)},
               "mass_table": mass_table, "history": history,
               "checkpoints": {"best": str(out_dir / "model_best.pt"),
                               "last": str(out_dir / "model_last.pt")}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    log.info("================= V8 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info("DONE — ACTIVE_CKPT NOT switched; t_good NOT set. Run the evaluate step next.")


if __name__ == "__main__":
    main()
