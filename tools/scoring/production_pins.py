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
ACTIVE_CKPT = "data/classifier/v10/model_best.pt"   # v10 unified location classifier (LIVE)
# ADOPTED 2026-08-02 (the v10 flip). v10 is v8's recipe on a re-rendered corpus at the raised
# iteration cap plus 1,267 appended maneuver-view locations; K=4, same three cutpoint logits.
# It CERTIFIED non-inferior on both pre-registered gating arms (census-144 q3 Δ−0.0090 p=0.79;
# floor-526 q3 Δ+0.0042 p=0.84) and gained nothing measurable on any arm — adoption is on the
# certified bar, not on a win. The load path stays version-agnostic (score_lib.Scorer reads
# num_classes/mean/std/geometry off the checkpoint's own config).
#
# ============================ THE REVERT-TOGETHER SET ============================
# Rolling back the pin ALONE leaves live thresholds calibrated to a head that is no longer
# serving — cuts on a specific head's p_good are numbers about nothing on another's. The
# ladder is `data/v10/build_metadata.json:rollback_ladder` — v10 -> v8 -> v7 -> v6 -> v5 (v9
# is NOT a rung: built, staged, never deployed). Reverting one rung means reverting ALL of:
#
#   1. ACTIVE_CKPT (here)                          — ACTIVE_VERSION and every decode stamp
#                                                    (corpus_common.is_current_decoded,
#                                                    production_seeder.SCORER_VERSION) follow it
#   2. production_seeder.T_GOOD_OVERRIDES          — v10: mandelbrot 0.03 / j:mb{3,4,5}
#                                                    0.27/0.03/0.06 (was 0.85/0.39/0.14/0.20)
#   3. data/atlas/keeper_cuts.json                 — stamped model="v10"
#   4. steered_frontier.TAU_H_FIDELITY_BASE
#      + TAU_H_FIDELITY_BASE_MODEL                 — vendored base, stamped "v10"
#   5. data/atlas/tau_h_base_v10.json              — provenance for (4)
#   6. tools/v10/derive_t_good_v10.py's output
#      data/v10/t_good_derivation.json             — derivation half of (2)
#
# Items 2-6 are the v10-stamped threshold files this flip wrote; 1-5 are the four the build
# metadata already named as coupled. The list above is PROSE — `COUPLED_ARTIFACTS` at the
# bottom of this module is the same enumeration as DATA, walked by
# tools/scoring/test_coupled_artifacts.py, which is what makes each entry's stamp actually
# checked rather than remembered. Keep the two in step; the test asserts the count matches.
#
# ROLLING BACK TO v8 MUST RE-DERIVE ITS TABLE, NOT COPY IT. data/v8/t_good_derivation.json
# was cut when the sweep's admission predicate was an AND, which is not the rule corn_decode
# serves on a K=4 head; re-derived through the aligned estimator (2026-08-02) v8's mandelbrot
# t_good is 0.14, not the committed 0.85. The artifact is left as the record of what v8
# actually served, so a rollback that copies it reinstates a threshold chosen against the
# wrong predicate — and test_t_good_adoption's drift gate will go red on mandelbrot the
# moment ACTIVE_VERSION is v8. v10's table is unaffected (byte-identical under both rules);
# the divergence and its 8 causal rows are pinned by tools/v8/test_t_good_sweep_decode.py.
# ================================================================================
V8_CKPT_ROLLBACK = "data/classifier/v8/model_best.pt"   # one-flip rollback anchor
V7_CKPT_ROLLBACK = "data/classifier/v7/model_best.pt"   # also the frozen pref_loc_v1 ranker pin
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


# ============================ THE REVERT-TOGETHER SET, AS DATA ============================
# The block beside ACTIVE_CKPT above says this in prose; `data/<v>/build_metadata.json`
# says four of them in JSON; four test files each asserted their own slice of it. Three
# representations, none executable against the others — so "what must move with the pin?"
# could only be answered by flipping and seeing what went red, which is what the v10 flip
# actually cost. This is the executable one:
# `tools/scoring/test_coupled_artifacts.py` walks it, reads every stamp it declares, and
# holds each to ACTIVE_VERSION.
#
# `stamp` says HOW TO READ this artifact's version stamp, as data rather than as an import,
# so this module stays dependency-free for its ~41 importers:
#   ("ckpt",)                          -> the pin itself (the dir name of ACTIVE_CKPT)
#   ("json", relpath, *keys)           -> nested key lookup in a committed JSON artifact.
#                                         `{v}` in relpath interpolates ACTIVE_VERSION.
#   ("attr", module, attribute)        -> a module constant, imported by the test
#   None                               -> carries no stamp of its own; `guard` names the
#                                         test that holds it to one that does
# `guard` is the test file that would go RED if this entry alone were left behind.
COUPLED_ARTIFACTS = (
    {
        "what": "tools/scoring/production_pins.ACTIVE_CKPT",
        "why": "ACTIVE_VERSION derives from it, and with it every decode stamp",
        "stamp": ("ckpt",),
        "guard": "tools/scoring/test_production_pins.py",
    },
    {
        "what": "tools/atlas/production_seeder.T_GOOD_OVERRIDES",
        "why": "per-partition t_good is calibrated to ONE head's p_good scale (protocol §4)",
        "stamp": None,          # a bare table of floats; its stamp is the derivation below
        "guard": "tools/scoring/test_t_good_adoption.py",
    },
    {
        "what": "data/{v}/t_good_derivation.json",
        "why": "the derivation half of T_GOOD_OVERRIDES — the numbers' provenance",
        "stamp": ("json", "data/{v}/t_good_derivation.json", "model"),
        "guard": "tools/scoring/test_t_good_adoption.py",
    },
    {
        "what": "data/atlas/keeper_cuts.json",
        "why": "same scale-bound argument, on the report-time keeper bar",
        "stamp": ("json", "data/atlas/keeper_cuts.json", "provenance", "model"),
        "guard": "tools/atlas/test_steered_frontier.py",
    },
    {
        "what": "tools/atlas/steered_frontier.TAU_H_FIDELITY_BASE + TAU_H_FIDELITY_BASE_MODEL",
        "why": "tau_h is a cut on a specific head's cheap p_good",
        "stamp": ("attr", "steered_frontier", "TAU_H_FIDELITY_BASE_MODEL"),
        "guard": "tools/atlas/test_steered_frontier.py",
    },
    {
        "what": "tools/atlas/steered_frontier.TAU_H_CAMPAIGN_FLOOR",
        "why": "the campaign floor is the same cut, at the campaign's own bar",
        "stamp": ("attr", "steered_frontier", "TAU_H_CAMPAIGN_FLOOR_MODEL"),
        "guard": "tools/v10/test_v10_flip.py",
    },
    {
        "what": "data/atlas/tau_h_base_{v}.json",
        "why": "the provenance record behind the vendored TAU_H_FIDELITY_BASE",
        "stamp": ("json", "data/atlas/tau_h_base_{v}.json", "model"),
        "guard": "tools/v10/test_v10_flip.py",
    },
)


def coupled_stamp(entry: dict, root: Path = ROOT):
    """Read one `COUPLED_ARTIFACTS` entry's version stamp, or None if it declares none.

    Raises FileNotFoundError / KeyError / ImportError rather than returning a sentinel: an
    artifact this list names and cannot be read is a broken revert-together set, and the
    guard that reports "absent" where it means "could not look" is the failure mode this
    repo has already paid for once."""
    spec = entry.get("stamp")
    if spec is None:
        return None
    kind = spec[0]
    if kind == "ckpt":
        return ACTIVE_VERSION
    if kind == "json":
        import json as _json
        rel = spec[1].format(v=ACTIVE_VERSION)
        doc = _json.loads((root / rel).read_text(encoding="utf-8"))
        for key in spec[2:]:
            doc = doc[key]
        return doc
    if kind == "attr":
        import importlib
        return getattr(importlib.import_module(spec[1]), spec[2])
    raise ValueError(f"unknown stamp kind {kind!r} in {entry['what']}")
