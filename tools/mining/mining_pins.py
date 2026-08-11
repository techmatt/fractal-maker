"""mining_pins.py — the render-mode ("mining") head pin, TORCH-FREE.

Extracted from `mining_gate.py` for exactly one reason: something has to be able to ask
"which mining head is live?" without paying a torch + timm import. `mining_gate` cannot be
that thing — it imports torch at module scope because its job is to run the model — so the
pin block moved down here and `mining_gate` re-exports it. This is the mining analogue of
`tools/scoring/production_pins.py`, which is the same split for the location head, and it is
imported the same way: BARE, so every reader holds one module object rather than two that
merely agree.

The reader that forced the split is `tools/emission/floors.py`, the stage-2 cut owner. Every
floor there is stamped with the head version it was set against and REFUSES to gate when the
active head disagrees; a stamp check that costs a torch import is a stamp check that gets
skipped on the pure-readout paths, which are precisely the paths that annotate a number with
"clears the production floor".

`MINING_GATE_VERSION` is DERIVED from the pin path, not restated. It used to be the literal
`"mining_v1"` in two files (`mining_gate.py` and `gate_report.py`, which stamps it into a
durable log), so a pin flip to v2 would have moved the checkpoint and left both version tags
reading v1 — a durable record of the wrong gate.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# The pin. Flip ACTIVE_MINING_CKPT to move the gate; everything else derives.
# --------------------------------------------------------------------------- #
HEAD_NAME = "render_mode_head"
# ADOPTED 2026-08-11 (prompts/flip_29.md). v3 is the `dedup_weighted` arm of the (28)
# render-mode retrain — a from-scratch build on the fully-tracked mining corpus, against a v1
# whose own training data cannot be regenerated. It lost the (28) winner rule on
# `pooled.auc_ge2`, and sheet E's 150 BLIND rows say that loss does not reproduce: clause (a)
# passes 0/7 there, 38 of 40 anchored failures across the five arms are non-reproducing, and
# v1's anchoring price at that boundary is -0.278 (0.953 anchored -> 0.676 blind).
ACTIVE_MINING_CKPT = "data/render_mode_head/v3/model_best.pt"   # staged seed-0 (LIVE)
MINING_V1_ROLLBACK = "data/render_mode_head/v1/model_best.pt"   # the PREVIOUS rung
MINING_GATE_THRESHOLD = 0.6691  # marginal p_ge3 boundary; conservative / high-precision
# 0.50 -> 0.6691 at the v3 flip: a VOLUME-MATCHED RESTATEMENT, not a retune. 0.6691 passes the
# SAME 129 of the 827 reference-pool rows (15.6%) that 0.50 passed on v1. Precision>=3 of the
# passers moves 0.636 -> 0.760 there — the head's gain, read at fixed volume. Record:
# data/render_mode_head/v3/volume_match_mining.json; procedure protocol §5a.
# The frozen operating point the release floor is set against: written by
# `lock_mining_gate.py --write` from the committed volume-match record, read back through its
# `read_lock()`, which REFUSES when HEAD_VERSION no longer matches the head it was measured
# on. The path is the PINNED head's because the lock describes it; a pin flip needs a new
# record at the new head's path, not an edit to the old one — v1's stays as the record of what
# v1's cuts bought.
LOCK_PATH = "data/render_mode_head/v3/mining_gate_lock.json"   # frozen ladder + operating point


def head_version(ckpt: str | None = None) -> str:
    """`"v3"` — the head version READ OFF the pin path (`data/<HEAD_NAME>/<version>/...`).

    Derived rather than declared: a hand-kept version constant beside a pin path is the
    "hardcoded True" failure — it outlives the thing it reports the moment the pin moves.
    Raises rather than guessing, because a pin whose version cannot be read is a pin no
    stamp check can be run against."""
    parts = Path(ckpt or ACTIVE_MINING_CKPT).as_posix().split("/")
    try:
        return parts[parts.index(HEAD_NAME) + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"cannot read a head version out of mining pin {ckpt or ACTIVE_MINING_CKPT!r}: "
            f"expected a `data/{HEAD_NAME}/<version>/...` path. Every stage-2 floor stamped "
            f"against this head needs a version to compare with.") from None


HEAD_VERSION = head_version()                       # "v3"
MINING_GATE_VERSION = f"mining_{HEAD_VERSION}"      # "mining_v3"
