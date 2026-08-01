"""The production pins — live classifier checkpoint + canonical crop/iteration policy.

THE single source of truth for four things every discovery/corpus/wallpaper path
resolves rather than hardcodes:

  * `ACTIVE_CKPT` / `ACTIVE_VERSION` — which classifier checkpoint is live, and what
    "current" means for decode stamps (`corpus_common.is_current_decoded`,
    `production_seeder.SCORER_VERSION`).
  * the canonical crop constants — `PALETTE`, `JPG_Q`, `DEFAULT_SS`, `BIN` — the
    geometry-independent half of "rebuild a location the way the classifier expects".
  * `auto_maxiter` — the fw-dependent iteration cap (see docs/design/auto_maxiter.md).
  * `make_scorer` — build a `score_lib.Scorer` on an explicit checkpoint path.

These lived inside `active_ckpt.py` — a retired one-off reframing probe — until the
2026-07-31 split. `active_ckpt` now re-exports every name here, so its ~41 importers
are unchanged; new code should import from this module by name.

`tools/scoring/test_production_pins.py` pins the resolved values.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# score_lib (make_scorer) lives under tools/mining; corpus helpers under tools/corpus.
for _p in (ROOT, ROOT / "tools" / "corpus", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
PALETTE = "twilight_shifted"           # v4/v5 deploy-canonical palette
# --- Active discovery/guard/reframe classifier checkpoint (SINGLE SOURCE OF TRUTH) ---
# Every discovery-path scorer (production_seeder, guard, reframe) resolves the live
# checkpoint from here. Flip ACTIVE_CKPT and the whole gate moves; nothing else hardcodes
# a version. The load path is version-agnostic (score_lib.Scorer reads mean/std/head from
# the checkpoint's own config), so only this string changes between versions.
ACTIVE_CKPT = "data/classifier/v8/model_best.pt"    # v8 unified location classifier (LIVE)
# v8 is the first K=4 head: it emits THREE cutpoint logits (P>=2, P>=3, P>=4) where v5..v7
# emitted two, so it can decode class 4. The load path stays version-agnostic — score_lib.Scorer
# reads num_classes/mean/std/geometry off the checkpoint's own config — but a K=3-shaped consumer
# that hardcoded two logits will now fail loudly on the state-dict shape rather than mis-score.
# Rollback: point ACTIVE_CKPT back at v7 (the one-flip rollback anchor, the role v6 held) to
# restore the prior gate; v6/v5 remain the deeper rollbacks. A rollback must also revert the
# per-partition t_good table (production_seeder.T_GOOD_OVERRIDES) and the keeper cut
# (data/atlas/keeper_cuts.json) — those thresholds are calibrated to v8's p_good scale and are
# meaningless on v7's. tools/atlas/test_steered_frontier.py holds keeper_cuts to the active
# version, so a rollback that forgets goes red rather than silent.
V7_CKPT_ROLLBACK = "data/classifier/v7/model_best.pt"
V6_CKPT_ROLLBACK = "data/classifier/v6/model_best.pt"
V5_CKPT_ROLLBACK = "data/classifier/v5/model_best.pt"
DEFAULT_MODEL = ACTIVE_CKPT             # unified location-quality model (== ACTIVE_CKPT)
# Version token of the live checkpoint ("v7"/"v8"...), parsed off the checkpoint dir. This
# is the SINGLE SOURCE OF TRUTH for what "current" means: corpus_common.is_current_decoded
# and production_seeder.SCORER_VERSION both resolve the decode-stamp version from here, so
# flipping ACTIVE_CKPT moves the whole notion of "current-decoded" with it.
ACTIVE_VERSION = Path(ACTIVE_CKPT).parent.name   # "v8"
JPG_Q = 90                              # match corpus crop quality
DEFAULT_SS = 4                          # ss4 = v4/v5 deploy-canonical antialiased view

# --- auto_maxiter: native fw-dependent policy (PRODUCTION; see docs/design/auto_maxiter.md) ---
# base 500 -> 4000 and clamp 8000 -> 67000 on 2026-07-31. Measured on 32 atoms spanning
# fw 3.3e-10..0.76, each walked up a cap ladder until radial_rings stopped moving (all 32
# converged): the convergent cap is a near-constant MULTIPLE of the old policy — mean 7.7,
# median 8.0, max 24 — so the fw SHAPE (k) was right and the BASE was 8x too low. The old
# 8000 clamp was never binding (old-policy max over the v8 manifest was 5424), so the clamp
# rises only to stop re-clipping the raised base; 67000 is non-binding over that manifest
# (new max 43,397). RESIDUAL: x8 covers the median, the measured tail runs to x24 — the most
# decorated material stays somewhat clipped. Median-clean, not clean.
#
# tools/explorer/render_core.py carries an independent copy of these four constants; the two
# are pinned to agree by tools/scoring/test_maxiter_policy.py.
FW_HOME = 3.0
MAXITER_BASE, MAXITER_K, MAXITER_MIN, MAXITER_MAX = 4000, 0.30, 200, 67000


def auto_maxiter(fw: float) -> int:
    ratio = FW_HOME / fw if fw > 0 else 1.0
    lz = math.log2(ratio) if ratio > 0 else 0.0
    val = MAXITER_BASE * (1.0 + MAXITER_K * lz)
    return int(max(MAXITER_MIN, min(MAXITER_MAX, val)))


def make_scorer(model_path: str):
    """Build a `score_lib.Scorer` on an EXPLICIT checkpoint path (no silent default)."""
    from score_lib import Scorer
    return Scorer(model_path=model_path)
