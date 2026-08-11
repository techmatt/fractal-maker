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
# ADOPTED 2026-08-11 (prompts/flip_29.md). v4b is the (28) from-scratch retrain; the SEED is
# seed 1, picked on best blind sheet-D AUC>=4 (0.609 of {0.572, 0.609, 0.557, 0.585, 0.512},
# all five above v3's 0.510) — the job boundary, because production sees post-floor 3/4
# material. That pick SPENDS a 197-row selection: sheet D was the only unanchored read of the
# minibrot population and the seed choice consumed it, so the band is no longer held out.
# rollback: revert to data/wallpaper_head/v3/model_best.pt — the PREVIOUS rung under the
# ACTIVE+PREVIOUS weights-retention policy (docs/design/storage_classes.md), and the reversion
# must take GATE_THRESHOLD, floors.WALLPAPER_POOL, suggest_tier.{CUTS,INTAKE_CUTS} with it.
HEAD_CKPT_REL = "data/wallpaper_head/v4b/seed_1/model_best.pt"
HEAD_CKPT = ROOT / HEAD_CKPT_REL
V3_CKPT_ROLLBACK = "data/wallpaper_head/v3/model_best.pt"    # one-flip rollback anchor

# Marginal p_ge3 > threshold (good-only). QUALITY-FLOOR / volume-policy dial, overridable
# via emit_v1's --gate.
#
# 0.90 -> 0.6052 at the 2026-08-11 v4b flip, and this is a VOLUME-MATCHED RESTATEMENT, not a
# retune: a CORN marginal is calibrated to its training prior, so 0.90 is a point on v3's
# scale and says nothing on v4b's. 0.6052 is the score that passes the SAME 416 of the 1,337
# reference-pool rows (31.1%) that 0.90 passed on v3 — same volume, same job. Precision>=3 of
# the passers moves 0.798 -> 0.748 on that pool, which is what the HEAD changed, not what the
# cut bought. Record: data/wallpaper_head/v4b/volume_match_wallpaper.json
# (tools/scoring/volume_match.py), procedure: classifier_retrain_protocol.md §5a.
#
# The v3 rationale, kept because it is the reason this dial sits where it does and it did not
# change: v3 gained a real precision GRADIENT the flat-on-v2 head lacked (eval precision of
# passers 0.58@0.5 -> 0.68@0.90 -> 0.78@0.99), and on the dramatic beam the gate is NOT a
# volume dial — the emission SELECTOR saturates first — so the high floor bought quality at
# ZERO volume cost. Raise for a bit more precision at ~1 winner; lower for more volume.
GATE_THRESHOLD = 0.6052


def head_version(ckpt: str | None = None) -> str:
    """`"v4b"` — the head version READ OFF the pin path (`data/<HEAD_NAME>/<version>/...`).
    Derived, not declared: see `mining_pins.head_version` for why. Insensitive to a per-seed
    subdirectory: the version is the segment AFTER `wallpaper_head`, so
    `.../v4b/seed_1/model_best.pt` reads `v4b`, which is what every stamp is against."""
    parts = Path(ckpt or HEAD_CKPT_REL).as_posix().split("/")
    try:
        return parts[parts.index(HEAD_NAME) + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"cannot read a head version out of wallpaper pin {ckpt or HEAD_CKPT_REL!r}: "
            f"expected a `data/{HEAD_NAME}/<version>/...` path. Every stage-2 floor stamped "
            f"against this head needs a version to compare with.") from None


HEAD_VERSION = head_version()                       # "v4b"
