#!/usr/bin/env python
r"""Extract a checkpoint's `config` block to `data/classifier/<v>/config.json`.

WHY: the weights-retention policy (docs/design/storage_classes.md § weights retention)
de-tracks every head older than ACTIVE + PREVIOUS. v5/v6/v7 shipped the `.pt` and NOTHING
else — no `config.json`, no `metrics.json` — so de-tracking the weight would take the recipe
with it, and `classifier/train_v8.py` names `data/classifier/v7/model_best.pt["config"]` as
"the only surviving copy of the v7 recipe". A recipe is a RECORD; a record is small; the
policy de-tracks weights, not records. So the record is lifted out before the weight goes.

This is the same split the mining-v2 precedent makes and the same one the policy states:
the weight leaves, the run record and the report stay.

It writes only what the checkpoint itself carries — no defaults, no reconstruction. A version
that already has a `config.json` (v8 onward) is skipped rather than overwritten.

  uv run python tools/scoring/extract_retired_config.py v5 v6 v7
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    versions = (argv if argv is not None else sys.argv[1:])
    if not versions:
        sys.exit("usage: extract_retired_config.py <version> [<version> ...]")
    import torch
    for v in versions:
        d = ROOT / "data" / "classifier" / v
        ckpt, out = d / "model_best.pt", d / "config.json"
        if not ckpt.exists():
            print(f"{v}: SKIP — {ckpt} absent (already de-tracked and pruned?)")
            continue
        if out.exists():
            print(f"{v}: SKIP — {out.name} already present")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = blob.get("config") if isinstance(blob, dict) else None
        if not isinstance(cfg, dict):
            print(f"{v}: SKIP — checkpoint carries no `config` dict")
            continue
        cfg = dict(cfg)
        cfg["_extracted"] = (
            f"lifted verbatim out of data/classifier/{v}/model_best.pt['config'] on "
            f"2026-08-08, before that weight de-tracked under the ACTIVE+PREVIOUS weights "
            f"retention policy. The recipe is the record; the weight is not.")
        out.write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")
        print(f"{v}: wrote {out.relative_to(ROOT)} ({len(cfg)} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
