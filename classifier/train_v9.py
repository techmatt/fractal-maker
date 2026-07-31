"""Train v9 — the v8 recipe, VERBATIM, on the re-rendered corpus.

**Nothing changes but the data underneath.** The v8 augmentation cache was iterated to a
flat `maxiter=8000` regardless of frame width (`v4-render-batch`'s `--maxiter` default;
no v4..v8 plan carried a per-row cap), and production's own `auto_maxiter` was itself ~8x
too low — measured on 32 atoms spanning fw 3.3e-10..0.76, the convergent cap is a
near-constant multiple of the old policy, mean 7.7 / median 8.0 / max 24. So every crop the
classifier has ever trained on was a clipped field, worst on decorated material
(x1.78-2.35) — exactly the class-3/class-4 boundary this head exists to resolve. v9 is the
same 7,117 locations, the same split, the same `loc_id`s and the same 24-slot v8b fan-out,
re-rendered with the cap raised (base 500 -> 4000, clamp 8000 -> 67000) and applied PER
ROW. See docs/design/auto_maxiter.md.

THE RECIPE IS READ, NOT RESTATED — and this time not even partially. train_v8 read the v7
config and overrode three things (the K=3 -> K=4 head); v9 overrides NOTHING about the
recipe. The config comes out of `data/classifier/v8/model_best.pt["config"]` verbatim; the
only fields that move are provenance (`cache_manifest`, `corpus_version`, and the strings
that say so). If a knob is not in the v8 config, v9 does not invent one — that is what
makes the v9-vs-v8 comparison a read on the DATA rather than on a recipe drift.

WHY A NEW VERSION ID RATHER THAN A v8 RETRAIN. If the retrained head shipped as "v8", the
decode-version predicate (`corpus_common.is_current_decoded`, keyed off
`active_ckpt.ACTIVE_VERSION`) could not separate old-v8 rows from new-v8 rows, and
mixed-decode readouts come back. Same argument that put the re-rendered tiles in
`data/v9/aug_cache` instead of overwriting v8's.

The machinery — K-generalized scoring/derive, the two-group AdamW, cosine schedule,
grad-clip, fp32, the biased WeightedRandomSampler, and the per-epoch atomic resume — is
imported from `train_v8`, not copied, so a v9 run cannot drift from the v8 procedure by an
editing accident.

RESUMABLE. Every epoch snapshots to out_dir/resume.pt; a relaunch continues from the next
epoch. A kill costs at most one epoch.

**ACTIVE_CKPT is NOT switched and t_good is NOT set here.** v8 stays the deployed scorer
until v9 is measured against the pre-registered bar. This trainer only writes under
data/classifier/v9/.

  uv run python -m classifier.train_v9
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

from .data_v4 import hist, load_locations, make_weighted_sampler
from .model import build_model, data_config
from .train_v2 import detect_device, set_seed
from .train_v8 import (CENSUS_SOURCE, FLOOR_SOURCE, NUM_CLASSES,
                       cutpoint_positive_counts, train_resumable)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v9"
V8_CKPT = ROOT / "data" / "classifier" / "v8" / "model_best.pt"
V9_CACHE = ROOT / "data" / "v9" / "cache_manifest.jsonl"
# Provenance only — never read as a hyperparameter. Recorded so a checkpoint carries the
# cap it was trained under, which is the one fact no v4..v8 checkpoint records.
CAP_DOC = "docs/design/auto_maxiter.md"
log = logging.getLogger("train_v9")


def load_v8_recipe() -> dict:
    """The v8 recipe, read out of its own checkpoint config. Raises if absent — v9 does
    not invent hyperparameters, and a v9 trained on a guessed recipe would make the
    pre-registered v9-vs-v8 bar meaningless."""
    if not V8_CKPT.exists():
        raise SystemExit(f"v8 config source missing: {V8_CKPT} (the v9 recipe is read from it)")
    ck = torch.load(V8_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    if not cfg:
        raise SystemExit(f"{V8_CKPT} has no embedded 'config' — cannot read the v8 recipe")
    return cfg


# Keys that are OUTPUTS of a v8 run rather than inputs to one; they must not be carried
# into v9's config as if they described v9.
_RUN_OUTPUT_KEYS = ("best_epoch",)
# Keys v9 legitimately overrides — all provenance, none behavioural.
_PROVENANCE_KEYS = ("cache_manifest", "corpus_version", "init", "recipe_vs_v7",
                    "recipe_vs_v8", "maxiter_policy", "cap_doc")


def build_v9_config(v8: dict, epochs_override=None) -> dict:
    """v8's config verbatim, minus its run outputs, plus v9's provenance."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "active_ckpt", ROOT / "tools" / "scoring" / "active_ckpt.py")
    ac = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("active_ckpt", ac)
    spec.loader.exec_module(ac)

    cfg = {k: v for k, v in v8.items() if k not in _RUN_OUTPUT_KEYS}
    cfg["cache_manifest"] = "data/v9/cache_manifest.jsonl"
    cfg["corpus_version"] = "v9"
    cfg["init"] = ("imagenet_backbone_fresh (NOT warm-started from v8); recipe read "
                   "verbatim from the v8 checkpoint config")
    cfg["recipe_vs_v8"] = ("IDENTICAL — every hyperparameter is v8's own config value. "
                           "The ONLY change is the corpus underneath: the same 7,117 "
                           "locations re-rendered at the raised iteration cap.")
    cfg["maxiter_policy"] = {"base": ac.MAXITER_BASE, "k": ac.MAXITER_K,
                             "min": ac.MAXITER_MIN, "max": ac.MAXITER_MAX,
                             "fw_home": float(ac.FW_HOME),
                             "applied": "per plan row, auto_maxiter(fw_slot)",
                             "supersedes": "flat maxiter=8000 on every v4..v8 tile"}
    cfg["cap_doc"] = CAP_DOC
    if epochs_override:
        cfg["epochs"] = int(epochs_override)
    return cfg


def assert_recipe_untouched(v8: dict, v9: dict) -> list:
    """Every key that is neither a v8 run output nor v9 provenance must be IDENTICAL.

    This is the load-bearing guard of the whole comparison: 'nothing changes but the data'
    is the claim the pre-registered bar rests on, and a claim that is only in a docstring
    is a claim nobody checks."""
    drift = []
    for k, val in v8.items():
        if k in _RUN_OUTPUT_KEYS or k in _PROVENANCE_KEYS:
            continue
        if v9.get(k) != val:
            drift.append((k, val, v9.get(k)))
    if drift:
        raise SystemExit("v9 config DRIFTED from v8 on behavioural keys — the v9-vs-v8 "
                         f"comparison would measure the drift, not the data:\n  {drift}")
    return sorted(k for k in v8 if k not in _RUN_OUTPUT_KEYS and k not in _PROVENANCE_KEYS)


def main():
    ap = argparse.ArgumentParser(description="Train v9 (v8 recipe, re-rendered corpus).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the inherited epoch count (default: v8's)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(a.device)
    v8 = load_v8_recipe()
    cfg = build_v9_config(v8, a.epochs)
    inherited = assert_recipe_untouched(v8, cfg)
    set_seed(int(cfg["seed"]))
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    log.info(f"recipe read VERBATIM from v8 config: {V8_CKPT}")
    log.info(f"  {len(inherited)} behavioural keys inherited unchanged: {inherited}")
    log.info(f"  cap policy: base {cfg['maxiter_policy']['base']} k {cfg['maxiter_policy']['k']} "
             f"clamp [{cfg['maxiter_policy']['min']},{cfg['maxiter_policy']['max']}], "
             f"per plan row  (v8: flat 8000)")

    if not V9_CACHE.exists():
        raise SystemExit(f"v9 cache manifest missing: {V9_CACHE} "
                         f"(uv run python tools/v9/build_plan.py)")
    locs = load_locations(cache_path=V9_CACHE)
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

    tr_labels = [l.label for l in train_locs]
    log.info("=== effective positive count at each cutpoint (TRAIN) ===")
    for name, _thr, npos in cutpoint_positive_counts(tr_labels, NUM_CLASSES):
        log.info(f"  {name:14s}: {npos} train locations  ({100*npos/len(tr_labels):.1f}%)")

    sampler, mass_table = make_weighted_sampler(train_locs, beta=cfg["beta_biased"],
                                                class_balance=cfg["class_balance"])
    log.info(f"=== sampled mass (beta={cfg['beta_biased']}, "
             f"class_balance={cfg['class_balance']}) ===")
    log.info(f"  class_count={mass_table['class_count']}  "
             f"w_class={ {k: round(v, 4) for k, v in mass_table['w_class'].items()} }")

    probe = build_model(target="ordinal", pretrained=True, num_classes=NUM_CLASSES)
    data_cfg = data_config(probe)
    del probe
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
               "recipe_inherited_from": str(V8_CKPT),
               "recipe_keys_inherited_unchanged": inherited,
               "checkpoints": {"best": str(out_dir / "model_best.pt"),
                               "last": str(out_dir / "model_last.pt")}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    log.info("================= V9 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info("DONE — ACTIVE_CKPT NOT switched; t_good NOT set. Run tools/v9/eval_v9.py next.")


if __name__ == "__main__":
    main()
