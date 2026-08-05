"""wallpaper_pins.py — the wallpaper (smooth) quality-head pin, TORCH-FREE.

Extracted from `emit_v1.py` for the same reason `tools/mining/mining_pins.py` was extracted
from `mining_gate.py`: `emit_v1` imports torch, numpy, PIL, the colormap tail and the
emission selector at module scope, so "which wallpaper head is live, and at what gate?"
could not be asked without paying all of it. `emit_v1` re-exports these names, so every
existing reader (`emit_v1.HEAD_CKPT`, `emit_v1.GATE_THRESHOLD`) is unchanged.

The reader that forced the split is `tools/emission/floors.py`: every stage-2 floor is
stamped with the head version it was set against and refuses to gate when the active head
disagrees. The pure readouts (`first_release_readout`, `reselect_readout`,
`q4_harvest_readout`) run that check, and they are pure on purpose.

`HEAD_VERSION` is DERIVED from the pin path, never declared beside it.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HEAD_NAME = "wallpaper_head"
# rollback: revert to v2/model_best.pt (see classifier/train_wallpaper_v3.py's rollback note).
HEAD_CKPT_REL = "data/wallpaper_head/v3/model_best.pt"
HEAD_CKPT = ROOT / HEAD_CKPT_REL

# Marginal p_ge3 > threshold (good-only). QUALITY-FLOOR / volume-policy dial, overridable
# via emit_v1's --gate. Retuned 0.5 -> 0.90 for head v3 (see prompts/prompt_gate_retune_v3.md):
# v3 gained a real precision GRADIENT the flat-on-v2 head lacked (eval precision of passers
# 0.58@0.5 -> 0.68@0.90 -> 0.78@0.99). On the current dramatic beam the gate is NOT a volume
# dial — the emission SELECTOR saturates first (winners flat ~21/52-loc-batch across
# thr in [0.5,0.90], all winners already p_ge3>0.94), so 0.90 buys a higher-quality floor
# feeding the selector at ZERO volume cost and holds the line on weaker/future pools. Raise
# toward 0.95+ only to trade ~1 winner for a bit more precision; lower for more volume.
GATE_THRESHOLD = 0.90


def head_version(ckpt: str | None = None) -> str:
    """`"v3"` — the head version READ OFF the pin path (`data/<HEAD_NAME>/<version>/...`).
    Derived, not declared: see `mining_pins.head_version` for why."""
    parts = Path(ckpt or HEAD_CKPT_REL).as_posix().split("/")
    try:
        return parts[parts.index(HEAD_NAME) + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"cannot read a head version out of wallpaper pin {ckpt or HEAD_CKPT_REL!r}: "
            f"expected a `data/{HEAD_NAME}/<version>/...` path. Every stage-2 floor stamped "
            f"against this head needs a version to compare with.") from None


HEAD_VERSION = head_version()                       # "v3"
