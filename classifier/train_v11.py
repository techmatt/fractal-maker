"""Train v11 — the v8/v9/v10 recipe, VERBATIM, on the v11 corpus and the v11 split.

WHAT MOVES. Three things, all upstream of this file: (1) the SPLIT RULE — v11 randomizes a
stratified holdout over leakage-closure groups instead of freezing v8's prefix, which is why
`julia:mandelbrot` and `phoenix` have an eval population at all for the first time; (2) the
CORPUS — 11,303 labeled locations against v10's 8,382; (3) the AUG RECIPE — 32 independently
drawn tiles per location (palette from a 68-name pool, geometry, AA level and JPEG quality
each drawn per tile) against v10's 24-slot product. All three are recorded in
`data/v11/build_record.json` and `data/v11/aug_recipe.json`.

WHAT DOES NOT MOVE, and is asserted rather than asserted-in-a-docstring: every behavioural
hyperparameter. The config is read out of `data/classifier/v10/model_best.pt["config"]` and
cross-checked against v9's, exactly as `train_v10` read v9's and cross-checked v8's, so
"identical recipe all the way back to the deployed head" stays a checked claim. K comes off
that config (`num_classes`), never a literal, and is asserted to cover the corpus.

THE MODEL-SELECTION OBJECTIVE IS A CONTROLLED VARIABLE. `train_resumable` selects on not-bad
AP over whatever eval renders it is handed, so the selection POPULATION is part of the
recipe. v10 attempt 1 moved it by 90 locations and lost 0.10 AUC on the census
(`data/v10/prereg_v10.json` amendment 1). v11 hands it the same 670 — `prospect_census` +
`loose0_v3_floor` — that v8, v9 and v10 selected over. v11's other 2,190 eval locations
(the 90 + 290 newer instruments and the 1,810-row stratified holdout) touch neither training
nor the pick, and are scored only at certification.

THE EVAL SPLIT IS NOT UNBIASED-ONLY ANY MORE, and that is a v11 design change rather than a
leak. v8..v10 asserted `all(not biased)` over the eval split because every eval row was a
score-unconditioned instrument. v11 adds a second eval ROLE: `holdout`, a stratified random
draw over the remaining split groups, biased exactly as training is — held out so a
calibration cut has an out-of-sample population, and explicitly NOT a base-rate instrument
(`build_record.json:eval_roles`). So the assertion moves to where it still holds: every
`instrument` row must be unbiased, and the holdout must not reach the selection objective.

RESUMABLE (per-epoch atomic snapshot, inherited from `train_v8.train_resumable`).

**ACTIVE_CKPT is NOT switched and t_good is NOT set here.** v10 stays the deployed scorer;
this trainer only writes under `data/classifier/v11/`.

  uv run python -m classifier.train_v11
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

from .data_v4 import hist, make_weighted_sampler
from .data_v11 import load_locations_v11
from .model import build_model, data_config
from .train_v2 import detect_device, set_seed
from .train_v8 import CENSUS_SOURCE, FLOOR_SOURCE, cutpoint_positive_counts, train_resumable

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v11"
V10_CKPT = ROOT / "data" / "classifier" / "v10" / "model_best.pt"
V9_CKPT = ROOT / "data" / "classifier" / "v9" / "model_best.pt"

UNIFORM_SOURCE = "maneuver_uniform_v1"     # v10's third instrument
Q4_UNIFORM_SOURCE = "q4_uniform_eval"      # registered 2026-08-03, first eval use here
SELECTION_SOURCES = (CENSUS_SOURCE, FLOOR_SOURCE)
SELECTION_N = 670                          # v8/v9/v10's selection population, to the row
log = logging.getLogger("train_v11")

# Keys that are OUTPUTS of a prior run rather than inputs to one.
_RUN_OUTPUT_KEYS = ("best_epoch",)
# Keys v11 legitimately overrides — all provenance, none behavioural.
_PROVENANCE_KEYS = ("cache_manifest", "corpus_version", "init", "recipe_vs_v7",
                    "recipe_vs_v8", "recipe_vs_v9", "recipe_vs_v10", "maxiter_policy",
                    "cap_doc", "corpus_note", "selection_population", "split_rule",
                    "aug_recipe")


def load_recipe() -> tuple[dict, dict]:
    """(v10 config, v9 config). v10's is the recipe v11 inherits; v9's is the cross-check."""
    out = []
    for p in (V10_CKPT, V9_CKPT):
        if not p.exists():
            raise SystemExit(f"recipe source missing: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck.get("config")
        if not cfg:
            raise SystemExit(f"{p} has no embedded 'config' — cannot read the recipe")
        out.append(cfg)
    return out[0], out[1]


def build_v11_config(v10: dict, epochs_override=None) -> dict:
    """v10's config verbatim, minus its run outputs, plus v11's provenance."""
    cfg = {k: v for k, v in v10.items() if k not in _RUN_OUTPUT_KEYS}
    cfg["cache_manifest"] = "data/v11/cache_manifest.jsonl"
    cfg["corpus_version"] = "v11"
    cfg["init"] = ("imagenet_backbone_fresh (NOT warm-started from v8/v9/v10); recipe read "
                   "verbatim from the v10 checkpoint config")
    cfg["recipe_vs_v10"] = ("IDENTICAL — every hyperparameter is v10's own config value, "
                            "itself asserted identical to v9's and v8's. What changes is "
                            "the corpus, the split rule and the augmentation recipe.")
    cfg["corpus_note"] = ("11,303 labeled locations (v10: 8,382) — a FRESH build, not an "
                          "append: protocol §1's frozen prefix and a re-randomized split "
                          "cannot both hold, and the split rule is what v11 changes. "
                          "Comparability is carried by the instruments, reproduced "
                          "location-for-location (data/v11/build_record.json).")
    cfg["split_rule"] = ("forced-eval score-unconditioned instruments (1,050) + a "
                         "stratified random holdout (1,810) drawn over leakage-closure "
                         "split groups; 8,443 train. seed 20260808.")
    cfg["aug_recipe"] = ("v11-independent-32: 32 tiles per location, each drawing its own "
                         "palette (68-name pool), geometry, AA level and JPEG quality "
                         "(q60..95). Supersedes v8b's 4x3x2 product.")
    cfg["maxiter_policy"] = {"base": 4000, "k": 0.3, "min": 200, "max": 67000,
                             "fw_home": 3.0,
                             "applied": "per location, auto_maxiter(CANONICAL fw)",
                             "supersedes": "v9/v10's auto_maxiter(fw_SLOT)"}
    cfg["selection_population"] = (
        f"census + floor ({SELECTION_N}) — v8/v9/v10-COMPARABLE, frozen on purpose. The "
        f"uniform-90, the q4-uniform-290 and the 1,810-row holdout are held out of training "
        f"AND of the checkpoint pick; the objective is a controlled variable "
        f"(prereg_v10.json amendment 1).")
    if epochs_override:
        cfg["epochs"] = int(epochs_override)
    return cfg


def assert_recipe_untouched(prior: dict, new: dict, label: str) -> list:
    """Every key that is neither a run output nor v11 provenance must be IDENTICAL.

    The load-bearing guard of the whole comparison: v11 moves the corpus, the split and the
    augmentation, and the pre-registered bars are a read on those. A hyperparameter that
    drifted alongside them would be measured as if it were one of them."""
    drift = []
    for k, val in prior.items():
        if k in _RUN_OUTPUT_KEYS or k in _PROVENANCE_KEYS:
            continue
        if new.get(k) != val:
            drift.append((k, val, new.get(k)))
    if drift:
        raise SystemExit(f"v11 config DRIFTED from {label} on behavioural keys — the "
                         f"pre-registered bars would measure the drift, not the corpus:"
                         f"\n  {drift}")
    return sorted(k for k in prior if k not in _RUN_OUTPUT_KEYS and k not in _PROVENANCE_KEYS)


def main():
    ap = argparse.ArgumentParser(description="Train v11 (v10 recipe, v11 corpus + split).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the inherited epoch count (default: v10's)")
    ap.add_argument("--no-verify-paths", action="store_true",
                    help="skip the 361,696-file existence sweep (tools/v11/verify_cache.py "
                         "checks it both ways; this only skips re-checking at load)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(a.device)
    v10, v9 = load_recipe()
    cfg = build_v11_config(v10, a.epochs)
    inherited = assert_recipe_untouched(v10, cfg, "v10")
    assert_recipe_untouched({k: v for k, v in v9.items() if k not in _PROVENANCE_KEYS},
                            cfg, "v9 (via v10)")

    K = int(cfg["num_classes"])
    set_seed(int(cfg["seed"]))
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    log.info(f"recipe read VERBATIM from the v10 config: {V10_CKPT}")
    log.info(f"  cross-checked against v9's config: {V9_CKPT}")
    log.info(f"  {len(inherited)} behavioural keys inherited unchanged: {inherited}")
    log.info(f"  K = {K} (read from config['num_classes'], not hardcoded)")

    t_load = time.time()
    locs = load_locations_v11(verify_paths=not a.no_verify_paths)
    log.info(f"loaded {len(locs)} locations x {len(locs[0].renders)} tiles "
             f"in {time.time()-t_load:.0f}s")
    max_label = max(l.label for l in locs)
    assert max_label <= K, (
        f"corpus holds label {max_label} but the inherited head has K={K} — training would "
        f"silently truncate the top class")

    train_locs = [l for l in locs if l.split == "train"]
    eval_locs = [l for l in locs if l.split == "eval"]
    instrument = [l for l in eval_locs if l.eval_role == "instrument"]
    holdout = [l for l in eval_locs if l.eval_role == "holdout"]
    assert len(instrument) + len(holdout) == len(eval_locs), \
        "an eval row carries neither eval_role — the build's roles are not exhaustive"
    # v8..v10 asserted the whole eval split unbiased. v11's holdout is biased BY DESIGN
    # (see the module docstring), so the assertion holds where it still means something.
    biased_instruments = [l.location_id for l in instrument if l.biased]
    assert not biased_instruments, (
        f"{len(biased_instruments)} instrument rows are biased — the cardinal sin of "
        f"protocol §2, e.g. {biased_instruments[:5]}")

    ftypes = Counter(l.fractal_type for l in locs)
    log.info(f"locations: {len(locs)} (train {len(train_locs)} {hist(train_locs)}, "
             f"eval {len(eval_locs)} {hist(eval_locs)})  fractal_type={dict(ftypes)}")
    log.info(f"  eval roles: instrument {len(instrument)} {hist(instrument)}  "
             f"holdout {len(holdout)} {hist(holdout)}")
    for ft in sorted(ftypes):
        tr = [l for l in train_locs if l.fractal_type == ft]
        ev = [l for l in eval_locs if l.fractal_type == ft]
        log.info(f"  {ft:18s}: train {len(tr):4d} {hist(tr)}  eval {len(ev):3d} {hist(ev)}")
    n_by_src = Counter(l.source for l in instrument)
    log.info(f"  instruments: census {n_by_src[CENSUS_SOURCE]}  floor {n_by_src[FLOOR_SOURCE]}"
             f"  uniform {n_by_src[UNIFORM_SOURCE]}  q4-uniform {n_by_src[Q4_UNIFORM_SOURCE]}")

    tr_labels = [l.label for l in train_locs]
    log.info("=== effective positive count at each cutpoint (TRAIN) ===")
    for name, _thr, npos in cutpoint_positive_counts(tr_labels, K):
        log.info(f"  {name:14s}: {npos} train locations  ({100*npos/len(tr_labels):.1f}%)")

    sampler, mass_table = make_weighted_sampler(train_locs, beta=cfg["beta_biased"],
                                                class_balance=cfg["class_balance"])
    log.info(f"=== sampled mass (beta={cfg['beta_biased']}, "
             f"class_balance={cfg['class_balance']}) ===")
    log.info(f"  class_count={mass_table['class_count']}  "
             f"w_class={ {k: round(v, 4) for k, v in mass_table['w_class'].items()} }")

    probe = build_model(target="ordinal", pretrained=True, num_classes=K)
    data_cfg = data_config(probe)
    del probe
    log.info(f"data_config: {data_cfg}")

    # ---- the SELECTION population: frozen to v8/v9/v10's 670 (see module docstring) ----
    sel_locs = [l for l in eval_locs if l.source in SELECTION_SOURCES]
    assert len(sel_locs) == SELECTION_N, (
        f"selection population is {len(sel_locs)}, expected {SELECTION_N} (census "
        f"{n_by_src[CENSUS_SOURCE]} + floor {n_by_src[FLOOR_SOURCE]}) — the "
        f"baseline-comparable subset moved, so the objective is not v10's")
    assert all(l.eval_role == "instrument" and not l.biased for l in sel_locs), \
        "a selection location is not an unbiased instrument"
    log.info(f"  SELECTION population: {len(sel_locs)} (census + floor — v8/v9/v10 "
             f"comparable). The other {len(eval_locs)-len(sel_locs)} eval locations touch "
             f"neither training nor the pick.")
    eval_canon = [l.canonical() for l in sel_locs]      # raises if the cell is missing
    eval_labels = np.asarray([l.label for l in sel_locs])

    log.info(f"=== TRAIN: {len(train_locs)} loc/epoch, batch {cfg['batch_size']}, "
             f"{cfg['epochs']} epochs (patience {cfg['patience']}) ===")
    t_start = time.time()
    best_state, best_epoch, best_val_ap, history, ckpt_cfg = train_resumable(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    cfg["best_epoch"] = best_epoch
    metrics = {"best_epoch": best_epoch, "val_best_not_bad_ap": best_val_ap,
               "eval_split_n": len(eval_locs), "eval_instrument_n": len(instrument),
               "eval_holdout_n": len(holdout), "selection_n": len(sel_locs),
               "instrument_n_by_source": dict(n_by_src), "num_classes": K,
               "train_n": len(train_locs),
               "class_counts": {"train": hist(train_locs), "instrument": hist(instrument),
                                "holdout": hist(holdout)},
               "cutpoint_positive_counts_train": {
                   name: npos for name, _thr, npos in
                   cutpoint_positive_counts(tr_labels, K)},
               "mass_table": mass_table, "history": history,
               "recipe_inherited_from": str(V10_CKPT),
               "recipe_cross_checked_against": str(V9_CKPT),
               "recipe_keys_inherited_unchanged": inherited,
               "checkpoints": {"best": str(out_dir / "model_best.pt"),
                               "last": str(out_dir / "model_last.pt")}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    log.info("================= V11 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info("DONE — ACTIVE_CKPT NOT switched; t_good NOT set. "
             "Run tools/v11/eval_v11.py next.")


if __name__ == "__main__":
    main()
