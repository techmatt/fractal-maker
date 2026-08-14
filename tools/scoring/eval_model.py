"""THE eval-side checkpoint loader + binary AUC helper, shared by every version's battery.

Both functions lived in `tools/v8/eval_v8.py` and were imported from there by v9, v10, v11 and
`v10/diagnose_selection.py` — so the LIVE version's battery took a hard dependency on a
version-scoped module whose own inputs (`data/v8/cache_manifest.jsonl`) had already been
deleted. `eval_v8.py` went on 2026-08-10 (docs/design/retired.md); these two came here, beside
the other shared owners the version dirs used to re-declare (`production_pins`, `partitions`,
`eval_slice`, `batch_registry`, `release_mix`).

Neither function is version-aware: `load_model` reads K and the deploy geometry off the
checkpoint's own `config` block, which is why it has survived K=3 -> K=4 unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_model(ckpt, device):
    """Build the CORN ordinal head described by `ckpt`'s own config and load its weights.

    Returns `(model, transform, K, cfg)`. K comes from the checkpoint (`num_classes`,
    defaulting to the pre-v8 K=3), and the transform is the DEPLOY one — `train=False`,
    i.e. the deterministic stretch + normalize mirror of the Rust JPG path, no jitter.
    """
    import torch                                        # noqa: PLC0415  (heavy)
    from classifier.data import Transform               # noqa: PLC0415
    from classifier.model import build_model            # noqa: PLC0415

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    K = int(cfg.get("num_classes", 3))
    # `backbone`/`backbone_kwargs` come off the checkpoint's own config, so a staged
    # backbone-comparison arm loads through this same function. Every v5..v11 config
    # stamps `backbone` as the default name, so this is a no-op for them.
    m = build_model(target="ordinal", drop_rate=cfg.get("drop_rate", 0.2),
                    drop_path_rate=cfg.get("drop_path_rate", 0.1), pretrained=False,
                    num_classes=K, backbone=cfg.get("backbone"),
                    backbone_kwargs=cfg.get("backbone_kwargs")).to(device)
    m.load_state_dict(ck["state_dict"])
    tf = Transform(cfg["geometry"], cfg["interpolation"], tuple(cfg["mean"]), tuple(cfg["std"]),
                   train=False)
    return m, tf, K, cfg


def q_auc(y_bin, s):
    """Binary AUC of score `s` against indicator `y_bin`, or None when one class is empty.

    None rather than a raise: a per-source slice that happens to be all-positive is a normal
    state of an eval table, and the caller reports the hole rather than aborting the battery.
    """
    from sklearn.metrics import roc_auc_score           # noqa: PLC0415

    y_bin = np.asarray(y_bin)
    if y_bin.min() == y_bin.max():
        return None
    return float(roc_auc_score(y_bin, s))
