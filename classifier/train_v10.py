"""Train v10 — the v8/v9 recipe, VERBATIM, on the EXTENDED corpus.

**Nothing changes but the labels.** v9 proved the machinery on a re-rendered corpus at an
identical recipe; v10 keeps that recipe bit-for-bit and moves only the population: 1,267
maneuver-view locations appended onto v8's frozen 7,115-row prefix (tools/v10/
build_manifest.py), rendered under v9's own cap policy into an extension of v9's tree
(tools/v10/build_plan.py). Same 24-slot v8b fan-out, same per-location seeds, same
held-out palettes. That single-variable discipline is what makes the pre-registered
v10-vs-v8 bars a read on the DATA rather than on a recipe drift.

THE RECIPE IS READ, NOT RESTATED. The config comes out of
`data/classifier/v9/model_best.pt["config"]` verbatim and is ALSO cross-checked against
v8's, so "identical recipe all the way back to the deployed head" is a checked claim and
not a sentence in this docstring. The only fields that move are provenance. K is read off
that config (`num_classes`), never hardcoded, and asserted against what the corpus actually
contains — a K smaller than the label range would silently truncate the top class.

WHY A NEW VERSION ID. Same argument as v9's: if the retrained head shipped as "v8", the
decode-version predicate (`corpus_common.is_current_decoded`, keyed off
`active_ckpt.ACTIVE_VERSION`) could not separate old-v8 rows from new-v8 rows, and
mixed-decode readouts come back.

The machinery — K-generalized scoring/derive, the two-group AdamW, cosine schedule,
grad-clip, fp32, the biased WeightedRandomSampler, and the per-epoch atomic resume — is
imported from `train_v8`, not copied, so a v10 run cannot drift from the v8 procedure by an
editing accident.

RESUMABLE. Every epoch snapshots to out_dir/resume.pt; a relaunch continues from the next
epoch. A kill costs at most one epoch.

**ACTIVE_CKPT is NOT switched and t_good is NOT set here.** v8 stays the deployed scorer
until v10 is measured against the bars in `data/v10/prereg_v10.json`. This trainer only
writes under data/classifier/v10/.

  uv run python -m classifier.train_v10
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
from .train_v8 import CENSUS_SOURCE, FLOOR_SOURCE, cutpoint_positive_counts, train_resumable

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v10"
V9_CKPT = ROOT / "data" / "classifier" / "v9" / "model_best.pt"
V8_CKPT = ROOT / "data" / "classifier" / "v8" / "model_best.pt"
V10_CACHE = ROOT / "data" / "v10" / "cache_manifest.jsonl"
UNIFORM_SOURCE = "maneuver_uniform_v1"     # the third eval instrument, new in v10
log = logging.getLogger("train_v10")


def load_recipe() -> tuple[dict, dict]:
    """(v9 config, v8 config). v9's is the recipe v10 inherits; v8's is the cross-check."""
    out = []
    for p in (V9_CKPT, V8_CKPT):
        if not p.exists():
            raise SystemExit(f"recipe source missing: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck.get("config")
        if not cfg:
            raise SystemExit(f"{p} has no embedded 'config' — cannot read the recipe")
        out.append(cfg)
    return out[0], out[1]


# Keys that are OUTPUTS of a prior run rather than inputs to one.
_RUN_OUTPUT_KEYS = ("best_epoch",)
# Keys v10 legitimately overrides — all provenance, none behavioural.
_PROVENANCE_KEYS = ("cache_manifest", "corpus_version", "init", "recipe_vs_v7",
                    "recipe_vs_v8", "recipe_vs_v9", "maxiter_policy", "cap_doc",
                    "corpus_note", "selection_population")


def build_v10_config(v9: dict, epochs_override=None) -> dict:
    """v9's config verbatim, minus its run outputs, plus v10's provenance."""
    cfg = {k: v for k, v in v9.items() if k not in _RUN_OUTPUT_KEYS}
    cfg["cache_manifest"] = "data/v10/cache_manifest.jsonl"
    cfg["corpus_version"] = "v10"
    cfg["init"] = ("imagenet_backbone_fresh (NOT warm-started from v8 or v9); recipe read "
                   "verbatim from the v9 checkpoint config")
    cfg["recipe_vs_v9"] = ("IDENTICAL — every hyperparameter is v9's own config value, "
                           "which was itself asserted identical to v8's. The ONLY change "
                           "is the population: 1,267 maneuver-view locations appended.")
    cfg["corpus_note"] = ("v8's 7,115-row frozen prefix + 1,267 appended locations from "
                          "the 2026-08 supply crawl and label-seeded harvest. Third eval "
                          "instrument: maneuver_uniform_v1 (90 loc, forced eval).")
    cfg["selection_population"] = (
        "census + floor (670) — v8/v9-COMPARABLE. The 90 maneuver_uniform locations are "
        "held out of training AND of checkpoint selection (prereg_v10.json amendment 1); "
        "attempt 1 selected over all 760 and that moved the objective off v8's.")
    if epochs_override:
        cfg["epochs"] = int(epochs_override)
    return cfg


def assert_recipe_untouched(prior: dict, new: dict, label: str) -> list:
    """Every key that is neither a run output nor v10 provenance must be IDENTICAL.

    The load-bearing guard of the whole comparison: 'nothing changes but the labels' is the
    claim the pre-registered bars rest on, and a claim that is only in a docstring is a
    claim nobody checks."""
    drift = []
    for k, val in prior.items():
        if k in _RUN_OUTPUT_KEYS or k in _PROVENANCE_KEYS:
            continue
        if new.get(k) != val:
            drift.append((k, val, new.get(k)))
    if drift:
        raise SystemExit(f"v10 config DRIFTED from {label} on behavioural keys — the "
                         f"pre-registered bars would measure the drift, not the labels:"
                         f"\n  {drift}")
    return sorted(k for k in prior if k not in _RUN_OUTPUT_KEYS and k not in _PROVENANCE_KEYS)


def main():
    ap = argparse.ArgumentParser(description="Train v10 (v9 recipe, extended corpus).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the inherited epoch count (default: v9's)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(a.device)
    v9, v8 = load_recipe()
    cfg = build_v10_config(v9, a.epochs)
    inherited = assert_recipe_untouched(v9, cfg, "v9")
    # ...and v9's recipe really is v8's, so "identical to the deployed head" is checked too.
    assert_recipe_untouched({k: v for k, v in v8.items()
                             if k not in _PROVENANCE_KEYS}, cfg, "v8 (via v9)")

    # K comes off the checkpoint config, never a literal — and must cover the corpus.
    K = int(cfg["num_classes"])
    set_seed(int(cfg["seed"]))
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    log.info(f"recipe read VERBATIM from the v9 config: {V9_CKPT}")
    log.info(f"  cross-checked against v8's config: {V8_CKPT}")
    log.info(f"  {len(inherited)} behavioural keys inherited unchanged: {inherited}")
    log.info(f"  K = {K} (read from config['num_classes'], not hardcoded)")

    if not V10_CACHE.exists():
        raise SystemExit(f"v10 cache manifest missing: {V10_CACHE} "
                         f"(uv run python tools/v10/build_plan.py)")
    locs = load_locations(cache_path=V10_CACHE)
    max_label = max(l.label for l in locs)
    assert max_label <= K, (
        f"corpus holds label {max_label} but the inherited head has K={K} — training would "
        f"silently truncate the top class")
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
    n_uniform = sum(1 for l in eval_locs if l.source == UNIFORM_SOURCE)
    log.info(f"  eval instruments: census(julia:mb)={n_census}  mandelbrot-floor={n_floor} "
             f" maneuver-uniform={n_uniform}")
    # The class-4 discipline the pre-registration turns into a verdict: every appended
    # class-4 location is train-side, so the eval slice's fours must still be the census's.
    n_eval_q4 = sum(1 for l in eval_locs if l.label == 4)
    n_census_q4 = sum(1 for l in eval_locs if l.label == 4 and l.source == CENSUS_SOURCE)
    assert n_eval_q4 == n_census_q4, (
        f"{n_eval_q4 - n_census_q4} class-4 eval locations are NOT census — the appended "
        f"fours were supposed to be train-side only")
    log.info(f"  class-4: {n_eval_q4} in eval (all census), "
             f"{sum(1 for l in train_locs if l.label == 4)} in train")

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

    # ---- the SELECTION population (prereg_v10.json amendment 1) ----------------------
    # `train_resumable` selects on not-bad AP over whatever eval set it is handed, and
    # `cfg["eval_split_is_val"]` is True — so the eval split IS the model-selection
    # objective. Attempt 1 handed it all 760 eval locations, which silently moved that
    # objective: v8 and v9 selected over 670 (census + floor), and 12% of attempt 1's
    # criterion was a population v8's selection never saw. It cost the census arm 0.10 AUC
    # — model_last, chosen by nothing, beat the selected checkpoint by +0.1036 there
    # (tools/v10/diagnose_selection.py).
    #
    # So selection is pinned to the v8-COMPARABLE subset, and the uniform-90 becomes a
    # fully held-out instrument: it touches neither training nor the pick, and is scored
    # only by tools/v10/eval_v10.py. That is strictly stronger for that arm than attempt 1,
    # where it influenced the checkpoint it was later used to judge.
    sel_locs = [l for l in eval_locs if l.source in (CENSUS_SOURCE, FLOOR_SOURCE)]
    assert len(sel_locs) == n_census + n_floor == 670, (
        f"selection population is {len(sel_locs)}, expected v8's 670 (census {n_census} + "
        f"floor {n_floor}) — the v8-comparable subset moved")
    assert not any(l.source == UNIFORM_SOURCE for l in sel_locs), \
        "the uniform leg leaked into the selection objective"
    log.info(f"  SELECTION population: {len(sel_locs)} (census + floor — v8-comparable). "
             f"The {n_uniform} uniform locations are held out of training AND of the "
             f"checkpoint pick; see prereg_v10.json amendment 1.")
    eval_canon = [l.canonical() for l in sel_locs]
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
               "eval_split_n": len(eval_locs), "census_n": n_census, "floor_n": n_floor,
               "uniform_n": n_uniform, "num_classes": K,
               "cutpoint_positive_counts_train": {
                   name: npos for name, _thr, npos in
                   cutpoint_positive_counts(tr_labels, K)},
               "mass_table": mass_table, "history": history,
               "recipe_inherited_from": str(V9_CKPT),
               "recipe_cross_checked_against": str(V8_CKPT),
               "recipe_keys_inherited_unchanged": inherited,
               "checkpoints": {"best": str(out_dir / "model_best.pt"),
                               "last": str(out_dir / "model_last.pt")}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    log.info("================= V10 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info("DONE — ACTIVE_CKPT NOT switched; t_good NOT set. "
             "Run tools/v10/eval_v10.py next.")


if __name__ == "__main__":
    main()
