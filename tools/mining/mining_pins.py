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
ACTIVE_MINING_CKPT = "data/render_mode_head/v1/model_best.pt"   # staged seed-0 (LIVE)
MINING_V1_ROLLBACK = None       # first version -> no prior gate to fall back to
MINING_GATE_THRESHOLD = 0.50    # marginal p_ge3 boundary; conservative / high-precision
# The frozen operating point the release floor is set against: written by
# `lock_mining_gate.py --write` from the committed 2026-08-06 sitting, read back through its
# `read_lock()`, which REFUSES when HEAD_VERSION no longer matches the head it was measured
# on. The path is v1's because the lock describes the pinned head; a pin flip needs a new
# record at the new head's path, not an edit to this one.
LOCK_PATH = "data/render_mode_head/v1/mining_gate_lock.json"   # frozen ladder + operating point


def head_version(ckpt: str | None = None) -> str:
    """`"v1"` — the head version READ OFF the pin path (`data/<HEAD_NAME>/<version>/...`).

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


HEAD_VERSION = head_version()                       # "v1"
MINING_GATE_VERSION = f"mining_{HEAD_VERSION}"      # "mining_v1"
