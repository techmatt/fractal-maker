#!/usr/bin/env python
r"""Why did v10 lose the census arm? One question, one cheap read — POST-HOC, NON-GATING.

The pre-registered verdict stands and this file cannot move it: v10 as trained is INFERIOR
on census-144 (0.6598 vs v8's 0.7509, paired DeLong p=0.0126). What this asks is a
different question, and only because the answer changes what the NEXT build should do:

    is the loss a worse MODEL, or a worse CHECKPOINT PICK?

The suspicion is specific. `train_resumable` selects on "max eval not-bad AP" over the
WHOLE eval split, and v10's eval split is not v8's: 90 maneuver-uniform locations joined
the census and the floor, so 12% of the selection objective is now a population v8's
selection never saw. v10 also stopped at best epoch 20 where v9 stopped at 36. If the
selection objective moved, then "the labels are the only variable" is FALSE for this
build — the model-selection criterion moved with them, and that is a flaw in the build, not
a property of the data.

The read: score `model_last.pt` (epoch 40, selected by nothing) on the same census tiles.

  * last ~= best  -> selection is exonerated; the census loss is in the trained model, and
                     the appended native-plane data genuinely cost julia:multibrot accuracy.
  * last >> best  -> the pick is implicated: the run passed through checkpoints that would
                     have certified, and the objective that rejected them was the one the
                     new instrument changed.

Neither outcome adopts anything, and `model_last.pt` is not a candidate — it is untracked
by policy and was selected by no criterion at all. This is diagnosis for the next build.

  uv run python tools/v10/diagnose_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "v7"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

import paths  # noqa: E402
from eval_delong import boot_ci, delong_paired  # noqa: E402
from classifier.data_v4 import load_locations  # noqa: E402
from classifier.train_v2 import detect_device  # noqa: E402
from classifier.train_v8 import derive_k, score_renders_k  # noqa: E402
from eval_model import load_model  # THE shared eval loader (tools/scoring)  # noqa: E402

V10_CACHE = ROOT / "data/v10/cache_manifest.jsonl"
OUT = "v10_selection_diagnosis.json"
CKPTS = {"v8": "data/classifier/v8/model_best.pt",
         "v10_best": "data/classifier/v10/model_best.pt",
         "v10_last": "data/classifier/v10/model_last.pt"}
INSTRUMENTS = {"prospect_census": 3, "loose0_v3_floor": 3, "maneuver_uniform_v1": 2}


def main() -> int:
    device = detect_device("auto")
    locs = [l for l in load_locations(cache_path=V10_CACHE) if l.split == "eval"]
    labels = np.array([l.label for l in locs])
    src = np.array([l.source for l in locs])
    canon = [l.canonical() for l in locs]

    scores = {}
    for name, rel in CKPTS.items():
        m, tf, K, _ = load_model(ROOT / rel, device)
        _, s = derive_k(score_renders_k(m, canon, tf, device, K - 1, num_workers=0))
        scores[name] = s
        del m

    out = {"question": "worse model, or worse checkpoint pick?",
           "status": "POST-HOC, NON-GATING — the pre-registered verdict is unchanged",
           "selection_objective": {
               "rule": "max eval not-bad AP over the WHOLE eval split",
               "v8_eval_split": "census 144 + floor 526 = 670",
               "v10_eval_split": "census 144 + floor 526 + uniform 90 = 760",
               "consequence": ("12% of v10's selection objective is a population v8's "
                               "selection never saw, so the checkpoint-pick criterion "
                               "moved with the labels — 'labels are the only variable' "
                               "does not hold for this build"),
               "v10_best_epoch": 20, "v9_best_epoch": 36},
           "per_instrument": {}}

    for inst, thr in INSTRUMENTS.items():
        m = src == inst
        y = (labels[m] >= thr).astype(int)
        blk = {"n": int(m.sum()), "thr": thr, "n_pos": int(y.sum())}
        for name in CKPTS:
            a, _b, _z, _p = delong_paired(y, scores[name][m], scores[name][m])
            ci = boot_ci(y, scores[name][m])
            blk[name] = {"auc": round(a, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}
        # last vs best, paired — the read this file exists for
        _ab, _al, z, p = delong_paired(y, scores["v10_best"][m], scores["v10_last"][m])
        blk["last_minus_best"] = round(blk["v10_last"]["auc"] - blk["v10_best"]["auc"], 4)
        blk["last_vs_best_delong_p"] = round(p, 4)
        out["per_instrument"][inst] = blk

    c = out["per_instrument"]["prospect_census"]
    d = c["last_minus_best"]
    out["read"] = (
        "SELECTION IMPLICATED: model_last (chosen by nothing) beats model_best on the "
        "census by %+.4f, so the run passed through checkpoints that would have certified "
        "and the objective that rejected them is the one the new instrument changed. The "
        "next build must freeze the selection objective to the v8-comparable subset "
        "(census + floor) and keep the uniform leg as a reported instrument only."
        % d if d > 0.03 else
        "SELECTION EXONERATED: model_last is %+.4f on the census, i.e. no better than the "
        "selected checkpoint, so the loss is in the trained model rather than the pick. "
        "The appended native-plane data cost julia:multibrot accuracy, and that is a real "
        "finding about the corpus mix, not a build flaw." % d)
    out["not_a_candidate"] = ("model_last.pt is untracked by policy and was selected by no "
                              "criterion; it is a probe, not an adoption option.")

    p = paths.scratch("v10_train", OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print("v10 SELECTION DIAGNOSIS — post-hoc, non-gating")
    print("=" * 78)
    for inst, b in out["per_instrument"].items():
        print(f"\n  {inst}  n={b['n']} pos={b['n_pos']} (label>={b['thr']})")
        for name in CKPTS:
            print(f"    {name:<10} AUC {b[name]['auc']:.4f} CI{b[name]['ci95']}")
        print(f"    last-best {b['last_minus_best']:+.4f}  (paired p={b['last_vs_best_delong_p']})")
    print(f"\n  READ: {out['read']}")
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
