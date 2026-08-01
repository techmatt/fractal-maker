#!/usr/bin/env python
"""steered_frontier.py — classifier-steered frontier descent (a new mode beside the walk).

The current production descent (`production_seeder.py` -> `guided-descend` walk -> reward)
picks a **uniform-random survivor** per rung and only scores FINISHED frames; the aesthetic
classifier never touches the trajectory (see `scratch/descent_algorithm_current.md`). Here the
classifier STEERS: a best-first frontier where each pop expands one rung
(`guided-descend --expand`, all gate survivors), scores every survivor's cheap 384-wide
twilight_shifted field with the active checkpoint, and re-prioritises by
`E[ord] + Gumbel - dup-penalty`. The fidelity study (`scratch/descent_score_fidelity.md`)
proved v7 on that cheap presentation ranks frames at Spearman 0.95 vs canonical, so the
steering signal is nearly free.

Everything downstream of "which node to expand" is REUSED verbatim from the production
seeder — the gates (black-cap 0.30 -> band -> occ-floor 0.321, node 384 / sigma-band), the
root pipeline (native depth-1 seeds + q3-density rejection + depth-2 probe), the julia hook,
the harvest (reframe + CORN decode at the per-partition t_good), the near-dup cloud, the
guard, and the ledger schema. Only the trajectory POLICY is new; the current walk path is
byte-untouched.

v1.1 priority (both coefficients default-on; set BOTH to 0 to reproduce the pilot exactly):
  priority = cheap_eord + Gumbel(T) - dup_penalty - novelty_penalty + beta*depth
`novelty_penalty` (`--lambda-m`, default 0.5) damps morph-space near-repeats: every scored
candidate's cheap twilight image is CLIP-embedded (library recipe) alongside the v7 forward
and compared (cos_max) against a run-scoped morph memory of all admitted + already-expanded
looks; the penalty ramps 0->lambda_m across cos [lo, hi], where the knee is re-anchored
EMPIRICALLY on this cheap substrate (morph_anchor_calibrate.py -> data/atlas/morph_anchors.json;
the library morph_gray anchors 0.851/0.974 are grayscale-scale and do not transfer). Siblings look alike,
so a hot lineage self-suppresses and perceptual re-buys sink before expansion. `beta*depth`
(`--beta`, default 0.02) is a small depth tie-breaker. Per-term contributions are logged to
`prio_terms.jsonl` per pushed candidate.

Crash safety is load-bearing (long processes here get killed at random): the frontier +
budget + RNG + per-root cap counters checkpoint to state.json every batch; `--resume`
continues; a STOP sentinel halts at a batch boundary; the admitted-outcome cloud is rebuilt
from the run ledger (the durable source of truth) so a kill/resume can never lose or
duplicate an admission.

  # one arm (steered), fresh run-scoped dir, 45 min:
  uv run python tools/atlas/steered_frontier.py --run-dir data/discovery/steered_runs/A \
      --families mandelbrot,multibrot3,multibrot4,multibrot5 --julia-hook --budget 45
  uv run python tools/atlas/steered_frontier.py --run-dir <dir> --resume        # after a kill
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools"))

import paths as _paths                   # noqa: E402  (storage-class helper: bulk() -> out-of-tree)

# production_seeder wires its own sub-imports (prescreen / reframe / guard / score_lib /
# active_ckpt) and owns the constants, root pipeline, near-dup machinery, guard, and the
# per-partition t_good table. Reuse it wholesale.
import production_seeder as ps          # noqa: E402
import prescreen                        # noqa: E402
import reframe                          # noqa: E402
import guard                            # noqa: E402
import location as loc_mod              # noqa: E402
import julia_ledger_schema as jls       # noqa: E402  (campaign/walk julia schema tag)
from score_lib import corn_decode       # noqa: E402
from active_ckpt import ACTIVE_CKPT, auto_maxiter  # noqa: E402
import deficit_scheduler as dsched       # noqa: E402  (pure; torch-free scheduling logic)
import minibrot_maneuvers as mnv         # noqa: E402  (pure mpmath; no subprocess, no torch)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BIN = ps.prescreen.BIN

# --- steering knobs ---
JULIA_HOOK_SPACING = 0.20   # item 3: hard 1-neighbour spacing (in the c/parameter plane) for the
                            # julia hook — don't hook a parent whose seed c is within this radius of
                            # an already-hooked parent THIS run. Replaces the old Q3_DENSITY_CAP
                            # density gate on hooked_c. Radius from the audit's chain-neighbour
                            # collision scale: genuine near-c dups sat <0.20, distinct-c over-kills
                            # sat >=1.0, so 0.20 spaces out redundant near-c hooks while leaving
                            # every genuinely distinct-c hook free to fire. Config knob
                            # (--julia-hook-spacing). See docs/design/morphology_dedup.md §5.
B_DEFAULT = 32            # nodes popped + expanded per batch
T_GUMBEL = 0.08          # priority exploration temperature (Gumbel scale)
M_CAP = 40               # hard cap on expansions per root_id
DIVE_NOISE_T = 0.02      # small Gumbel tie-break on the dive argmax-child selection
DUP_P0 = 1.0             # dup-penalty magnitude at zero distance to the q3 cloud (E[ord] units)
DUP_SCALE = ps.REJECT_RADIUS   # Gaussian decay scale of the dup penalty (plane coords)
NEUTRAL_PRIOR = 1.0      # root prior priority (mid E[ord] in [0,2])

# --- morph-novelty + depth knobs (v1.1; both zero => byte-identical pilot behaviour) ---
LAMBDA_M_DEFAULT = 0.5   # morph-novelty penalty magnitude (E[ord] units); CLI --lambda-m
BETA_DEFAULT = 0.02      # depth bonus per rung (E[ord] units); CLI --beta
CLIP_MODEL = "vit_base_patch16_clip_224.openai"  # matches the library morph_clip recipe
# The penalty knee is on the CHEAP-JPG substrate (not grayscale morph_gray), so the library
# morph_gray anchors do NOT transfer. Re-anchored empirically by morph_anchor_calibrate.py ->
# data/atlas/morph_anchors.json; these are only the last-resort fallback if that file is absent.
ANCHORS_PATH = ROOT / "data" / "atlas" / "morph_anchors.json"
MORPH_LO_FALLBACK = 0.85
MORPH_HI_FALLBACK = 0.974


def load_morph_anchors(cli_lo=None, cli_hi=None):
    """Resolve (lo, hi, source) for the novelty knee: CLI override > calibrated anchors file >
    fallback. Either CLI value alone overrides just that knee."""
    lo, hi, src = MORPH_LO_FALLBACK, MORPH_HI_FALLBACK, "fallback"
    if ANCHORS_PATH.exists():
        a = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
        lo, hi, src = float(a["lo"]), float(a["hi"]), "morph_anchors.json"
    if cli_lo is not None:
        lo, src = float(cli_lo), src + "+cli_lo"
    if cli_hi is not None:
        hi, src = float(cli_hi), src + "+cli_hi"
    if hi <= lo:
        hi = lo + 0.05
    return lo, hi, src
# --- minibrot maneuvers (v1.3; --maneuvers, default OFF => byte-identical) --------------
# A minibrot is a REFRAMING OPERATOR applied to a location already found, not a source of
# locations, so the two operators enter here as candidate MOVES: a fired probe pushes a new
# frontier NODE (like a root does — unscored, neutral prior), and the ordinary --expand /
# score / harvest machinery takes it from there. See docs/design/minibrot_maneuvers.md.
MAN_QUOTA_DEFAULT = 4        # reserved frontier SLOTS per batch (a floor, not a probability)
MAN_PROBE_P_DEFAULT = 0.25   # cost governor: P(probe fires) per popped rung
# Framing set. k=4 answers "is this atom good?"; k=16 is often close to a usable wallpaper
# frame by ITSELF, which is the material worth labeling — and it is free, because
# snap_to_nucleus_multi solves the nucleus once and reframes per k (a k is not a probe).
# No small k: framing INTO the atom is interior black (docs/design/minibrot_maneuvers.md §7).
MAN_K_DEFAULT = "none,4,16"
# Counters, named once so __init__/load_state/the summary can never drift apart.
#   probes_*        — cost-governor accounting (did the probe even get to run)
#   op_avail/unavail— the operator's own availability (the ~17% expectation)
#   avail_unused    — AVAILABLE BUT NOT PUSHED: the atom was already visited this run.
#                     Recorded because "the operator had nothing" and "the operator had
#                     something we already had" are different constraints at scale.
#   quota_bound     — reserved slots that promoted a node the plain priority top-B would
#                     NOT have taken (the floor actually binding, not merely present)
#   quota_unfilled  — reserved slots that went unused for lack of AVAILABILITY
MAN_TOTALS = ("man_probes_rolled", "man_probes_fired", "man_probes_coin_skip",
              "man_probes_cache_skip", "man_op_available", "man_op_unavailable",
              "man_avail_unused", "man_nodes_pushed", "man_quota_bound",
              "man_quota_unfilled", "man_nodes_expanded", "man_admitted")
FRONTIER_CAP = 6000      # prune the frontier to the top-N by priority (memory bound)
JULIA_ROOT_FW = 3.0      # fixed z-plane base-scale root view (matches --julia-root-fw)
EXPAND_TIMEOUT_S = 900   # hard-kill backstop on a hung --expand call
MIN_UNIT_TIMEOUT_S = 60  # floor for the budget-clamped per-unit backstop (unit_timeout_s)
ROOT_LOW_WATER = None    # replenish roots when frontier < this (set to B at runtime)

# Steered production walk config (mirror of production_seeder; keeps the gates identical).
EXPAND_FLAGS = [
    "--node-width", str(ps.NODE_WIDTH), "--sigma-band", ps.SIGMA_BAND,
    "--descent-occ-floor", str(ps.OCC_FLOOR), "--descent-black-cap", str(ps.BLACK_CAP),
]

FIDELITY_RECORDS = ROOT / "scratch" / "descent_score_fidelity_records.json"
C_PLANE = ("mandelbrot", "multibrot3", "multibrot4", "multibrot5")


# --------------------------------------------------------------------------- #
# Family <-> partition helpers (mirror production_seeder.resolve_family grammar).
# --------------------------------------------------------------------------- #
def render_family_of(partition: str) -> str:
    if partition == "mandelbrot" or partition in ("multibrot3", "multibrot4", "multibrot5"):
        return partition
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    raise ValueError(f"unknown partition {partition!r}")


def descend_flags(partition: str, c) -> list:
    """guided-descend --expand kernel flags for a homogeneous group (mirrors the walk grammar)."""
    if partition == "mandelbrot":
        return []
    if partition in ("multibrot3", "multibrot4", "multibrot5"):
        return ["--family", partition]
    if partition == "julia:mandelbrot":
        return ["--julia", "--c", str(c[0]), str(c[1])]
    if partition.startswith("julia:multibrot"):
        base = partition.split(":", 1)[1]
        return ["--family", base, "--julia", "--c", str(c[0]), str(c[1])]
    raise ValueError(f"unknown partition {partition!r}")


def loc_of(partition: str, c, cx, cy, fw):
    return ps.make_loc_of(render_family_of(partition), c)(cx, cy, fw)


# --------------------------------------------------------------------------- #
# tau_h — per-partition cheap p_good harvest cut from the fidelity study's paired scores.
# The cheap p_good cut that RETAINS ~90% of frames whose canonical p_good clears the
# family's t_good (= the 10th percentile of cheap p_good among those frames).
# --------------------------------------------------------------------------- #
# Prospective per-family floors raised from the campaign harvest logs (2026-07-25
# closeout). For these three c-plane families the fidelity-derived cut (~0.201) sits
# well below where the *campaign* cheap/canonical relationship starts costing admits:
# pooling every confirmation render across campaign1+2 (breadth+dive) and raising the
# cut to the point that loses <=5% of that family's admits gives the values below.
# mandelbrot has real headroom (drops ~30% of its confirmation renders for 4.8% admit
# loss); mb3/mb5 curves are band-thin near the current cut (~9-10% renders for ~5%
# loss). mb4 and every julia partition are left at the fidelity-derived value — their
# curves are band-thin and the cheap score is flat within the retained slice.
# Applied as a floor (max with the derived value) so it only ever raises, never lowers.
#
# RETIRED AT THE v8 FLIP, NOT CARRIED. The three values above were cuts on **v7's** cheap
# p_good, raised from v7-era campaign harvest logs, exactly like the `julia:mandelbrot`
# / `phoenix` t_good overrides that `production_seeder.T_GOOD_UNCALIBRATED` retires by
# name. A v7 floor on a v8 base is the same category error the version stamp below exists
# to stop, so it is not applied — and it cannot be re-derived yet: the floor's definition
# ("raise the cut to where it starts costing admits") needs ADMISSIONS under the active
# head, and no v8-era discovery run has happened. The mechanism stays live and tested; the
# table is empty on purpose, with its own stamp so an unstamped re-add is visible.
#
# It would be a no-op even if applied — every re-derived v8 base (0.199..0.704) is already
# far above every v7 floor (0.216..0.269) — but "harmless today" is not why it is empty.
TAU_H_CAMPAIGN_FLOOR_MODEL = "v8"
TAU_H_CAMPAIGN_FLOOR: dict = {}
# The v7 table, kept for the record only. NEVER read by the code path.
TAU_H_CAMPAIGN_FLOOR_V7_RETIRED = {
    "mandelbrot": 0.2690,
    "multibrot3": 0.2193,
    "multibrot5": 0.2162,
}

# Vendored fidelity-derived base tau_h — the decoupling from a DISPOSABLE artifact.
# derive_tau_h once loaded the cheap-p_good cuts straight from
# `scratch/descent_score_fidelity_records.json`, a study output living in the disposable `scratch/`
# tree. When `scratch/` was wiped, this launch-critical derivation lost its only input and
# `SystemExit`'d during frontier setup — a config that "looks applied" but strands the next
# campaign launch. The records file is gone and unrecoverable short of re-running the (renders +
# dual-scorer) study, BUT the derived VALUES survive in committed config: the campaign1/2 and
# shakeout_* run summaries (`data/discovery/*/summary.json`) all recorded this identical
# per-partition tau_h — the PRE-floor base derive_tau_h computed from the records. We vendor it
# here so the derivation no longer depends on a file that can vanish with `rm -r scratch/*`. The live
# study path still overrides these whenever the records are regenerated; the campaign floors in
# TAU_H_CAMPAIGN_FLOOR apply on top either way (they only ever raise).
#
# VERSION-STAMPED, AND THE MISMATCH IS FATAL. tau_h is a cut on the CHEAP-render p_good of a
# specific head: the fidelity study that produced these numbers ran under v7
# (tools/studies/descent_score_fidelity.py, which resolves its scorer through ACTIVE_CKPT),
# and TAU_H_CAMPAIGN_FLOOR was raised from v7-era campaign harvest logs. On a different head
# the cheap p_good distribution is a different distribution and these are numbers about
# nothing. Vendored constants are exactly the kind of thing that keeps returning a confident
# stale answer after a head change — nothing about a float says which model it describes — so
# `derive_tau_h` FAILS LOUDLY when the stamp does not match the active version rather than
# quietly gating a campaign at a threshold from the previous scorer.
#
# Re-derivation is deliberately NOT done here: it happens from the harvest logs at campaign
# launch (tools/atlas/tau_h_retained_readout.py builds both axes of the curve from
# harvest_log.jsonl). Until then the loud failure is the correct state — emission is dark
# after a flip anyway, so nothing is blocked that was not already waiting on a discovery run.
#
# RE-DERIVED UNDER v8 on 2026-07-31 by `tools/atlas/tau_h_rederive.py`, which is the
# regeneration path this comment's failure message points at. Provenance artifact:
# `data/atlas/tau_h_base_v8.json` (per-partition n, t_good, both population estimates).
# Method — the fidelity study's estimator verbatim, on a population the harvest logs make
# re-renderable: each sampled harvest-check geometry is re-rendered at BOTH presentations
# (384x216 ss1 cheap / 640x360 ss2 canonical) and re-scored under the ACTIVE head, then
# tau_h = the 10th percentile of cheap p_good among frames whose canonical p_good clears
# the family's t_good. All 8 partitions cut on their OWN population (n_pass 39..285); no
# partition fell back to the pooled cut.
#
# The values move a LOT versus v7 (mandelbrot 0.201 -> 0.704) and that is the point: v8's
# per-partition t_good is a different, much stricter bar (mandelbrot 0.85), so the frames
# that clear it sit far higher on the cheap axis. Serving the v7 numbers to a v8 gate would
# have rendered confirmations for a population v8 does not consider q3 at all.
#
# ONE BIAS, STATED. The harvest log only holds checks that already cleared the PREVIOUS
# head's tau_h, so it is left-truncated and its quantile is an UPPER bound. The untruncated
# walk-outcome ledger (prospect_run1: uniform-random gate survivors, never tau-selected) is
# re-derived alongside it as a cross-check, and the committed value is the per-partition
# MINIMUM of the two — the conservative side, since a too-high cut sheds admissions.
TAU_H_FIDELITY_BASE_MODEL = "v8"
TAU_H_FIDELITY_BASE = {
    "mandelbrot": 0.704061222076416,
    "multibrot3": 0.41670822501182553,
    "multibrot4": 0.550365686416626,
    "multibrot5": 0.4374629855155945,
    "julia:mandelbrot": 0.3485920131206512,
    "julia:multibrot3": 0.38111798763275145,
    "julia:multibrot4": 0.19956488609313963,
    "julia:multibrot5": 0.19899649918079373,
}


def _apply_campaign_floor(part: str, val: float) -> float:
    """Raise a base tau_h to `part`'s campaign floor (max — only ever raises, never lowers)."""
    floor = TAU_H_CAMPAIGN_FLOOR.get(part)
    return max(val, floor) if floor is not None else val


def _derive_tau_h_base_from_records(partitions: list[str], keep: float) -> dict:
    """Per-partition cheap-p_good cut from the fidelity study records (PRE campaign floor).

    The cut RETAINS ~`keep` of frames whose canonical p_good clears the family's t_good (= the
    (1-keep) quantile of cheap p_good among those frames), with a pooled cross-family fallback
    for partitions too thin to cut on their own."""
    rec = json.loads(FIDELITY_RECORDS.read_text(encoding="utf-8"))
    can, cheap = rec["scores"]["canonical"], rec["scores"]["cheap"]
    fam_of = {s["id"]: s["family"] for s in rec["samples"]}
    q = 1.0 - keep

    def cut(ids):
        vals = [cheap[i][2] for i in ids if i in cheap and i in can]  # cheap p_good
        return float(np.quantile(vals, q)) if len(vals) >= 5 else None

    # pooled fallback over every frame clearing its own family's t_good.
    pooled_pass = [i for i in can
                   if can[i][2] >= ps.t_good_for(fam_of.get(i, "mandelbrot"))]
    pooled = cut(pooled_pass)
    if pooled is None:
        pooled = 0.5

    base = {}
    for part in partitions:
        tg = ps.t_good_for(part)
        ids = [i for i in can if fam_of.get(i) == part and can[i][2] >= tg]
        base[part] = cut(ids)
        if base[part] is None:
            base[part] = pooled
    return base


def _active_scorer_version() -> str:
    """The live scorer version, from the single source of truth (tools/scoring/active_ckpt)."""
    import active_ckpt
    return active_ckpt.ACTIVE_VERSION


def derive_tau_h(partitions: list[str], keep=0.90) -> dict:
    """Per-partition harvest cut, campaign floors applied.

    Base is derived live from the fidelity study records when they are present (the study was
    re-run); otherwise it falls back to the vendored `TAU_H_FIDELITY_BASE` — the launch-critical
    path must NOT depend on the disposable `scratch/descent_score_fidelity_records.json`. A partition
    with neither a record-derived nor a vendored base fails loudly and immediately (naming the
    regenerator) rather than aborting deep in a frontier run.

    The vendored fallback is additionally GATED ON THE MODEL VERSION it was derived under
    (`TAU_H_FIDELITY_BASE_MODEL`): tau_h is a cut on a specific head's cheap p_good, so serving
    a v7-derived constant to a v8 gate is serving a number about nothing. That path raises
    instead. The LIVE record-derived path is not gated — if the study has been re-run, it was
    run under the active checkpoint by construction."""
    if FIDELITY_RECORDS.exists():
        base = _derive_tau_h_base_from_records(partitions, keep)
    else:
        active = _active_scorer_version()
        if active != TAU_H_FIDELITY_BASE_MODEL:
            raise SystemExit(
                f"tau_h derivation: the vendored TAU_H_FIDELITY_BASE was derived under "
                f"{TAU_H_FIDELITY_BASE_MODEL} but the active scorer is {active} "
                f"(tools/scoring/active_ckpt.ACTIVE_CKPT). tau_h is a cut on the CHEAP-render "
                f"p_good of a SPECIFIC head — a {TAU_H_FIDELITY_BASE_MODEL} cut on a {active} "
                f"gate is a number about nothing, and TAU_H_CAMPAIGN_FLOOR is "
                f"{TAU_H_FIDELITY_BASE_MODEL}-era too.\n"
                f"  Re-derive at campaign launch from the harvest logs "
                f"(tools/atlas/tau_h_retained_readout.py builds both axes of the curve), or "
                f"re-run tools/studies/descent_score_fidelity.py under {active} to regenerate "
                f"{FIDELITY_RECORDS.name}. Then update TAU_H_FIDELITY_BASE + "
                f"TAU_H_FIDELITY_BASE_MODEL together — never one without the other.")
        base = {p: TAU_H_FIDELITY_BASE.get(p) for p in partitions}
        missing = sorted(p for p, v in base.items() if v is None)
        if missing:
            raise SystemExit(
                f"tau_h derivation: {FIDELITY_RECORDS} absent and no vendored base for "
                f"{missing} — regenerate via tools/studies/descent_score_fidelity.py, or add "
                f"the partition to TAU_H_FIDELITY_BASE")
    return {p: _apply_campaign_floor(p, base[p]) for p in partitions}


# --------------------------------------------------------------------------- #
# Priority.
# --------------------------------------------------------------------------- #
def gumbel(rng: np.random.Generator, T: float) -> float:
    u = float(rng.random())
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return -T * math.log(-math.log(u))


def dup_penalty(cx, cy, cloud) -> float:
    """Large near an admitted q3, decaying (Gaussian, scale DUP_SCALE) with plane distance.

    Coords `float()`-coerced like `near_dup`: a cloud carrying string coords (q4_harvest /
    classic_phoenix serialize outcome_cx/cy as decimal STRINGS) would otherwise raise
    `float - str` here (a latent crash of the steering penalty). `load_prior_library_rows`
    already coerces the prior cloud at ingestion; this coercion makes the reader itself safe."""
    if not cloud:
        return 0.0
    cx, cy = float(cx), float(cy)
    d = min(math.hypot(cx - float(m["outcome_cx"]), cy - float(m["outcome_cy"])) for m in cloud)
    return DUP_P0 * math.exp(-(d / DUP_SCALE) ** 2)


def priority_terms(eord, g, dup_pen, cos_max, lambda_m, beta, depth, lo, hi):
    """Pure priority decomposition. Returns (priority, {terms}). At lambda_m==0 AND beta==0 this
    is byte-identical to the pilot's `eord + gumbel - dup_pen` (novelty/depth terms vanish)."""
    nov_pen = novelty_penalty(cos_max, lambda_m, lo, hi)
    depth_bonus = beta * depth
    prio = eord + g - dup_pen - nov_pen + depth_bonus
    return prio, dict(eord=eord, gumbel=g, dup_pen=dup_pen, cos_max=cos_max,
                      nov_pen=nov_pen, depth_bonus=depth_bonus, priority=prio)


def novelty_penalty(cos_max: float, lambda_m: float, lo: float, hi: float) -> float:
    """Morph-space near-repeat penalty: zero at substrate-typical similarity (cos<=lo), ramping
    linearly to full lambda_m at the near-repeat knee (cos>=hi). Anchors are empirical on the
    cheap-JPG substrate (morph_anchor_calibrate.py). A near-perceptual-dup of an admitted/
    expanded look sinks by ~lambda_m E[ord] units BEFORE it is popped. lambda_m=0 -> zero."""
    if lambda_m <= 0.0:
        return 0.0
    frac = (cos_max - lo) / (hi - lo)
    return lambda_m * min(max(frac, 0.0), 1.0)


# --------------------------------------------------------------------------- #
# Run-scoped morph memory — CLIP embeddings (library recipe) of the looks a candidate's
# novelty is measured against. cos_max vs this set is the novelty signal. Embeddings are
# L2-normalized; the max-cosine reduction runs on the CLIP device.
#
# Two semantics (the v1.2 fix). run2 grew ONE undifferentiated set of admitted+expanded
# looks to 10,420 rows; at that density the cheap-substrate cos_max is past the knee for
# ~90% of candidates, so the penalty acted as a near-constant down-shift, not a gradient
# (see steered_run2_report.md "Saturation caveat").
#   - LEGACY (recency_k == 0, the v1.1/pilot default): every admitted AND expanded look is
#     permanent; the set grows without bound. Kept as the default so v1.1 runs reproduce.
#   - RECENCY (recency_k > 0, the fix): the memory the novelty term buys against is
#     ADMITTED looks only (permanent — "don't re-buy a banked look") PLUS a rolling window
#     of the last `recency_k` COMPLETED batches' EXPANDED-node looks (cross-batch sibling
#     suppression on hot lineages, evicted once the lineage cools). The current batch's own
#     parents are excluded from its candidates' cos_max (see _all_rows) — comparing a child
#     to its own parent trivially saturates. The window keeps |memory| O(admitted +
#     recency_k*batch), so cos_max stays a live gradient instead of saturating.
# Persisted as `perm` (admitted / all-legacy) + `recency` (concatenated window blocks) +
# `block_sizes` (per-batch block lengths, so a resume evicts on the same boundaries); a
# legacy `mem`-keyed file still loads (folded into `perm`).
# --------------------------------------------------------------------------- #
class MorphMemory:
    def __init__(self, device: str, path: Path, recency_k: int = 0):
        self.device = device
        self.path = path
        self.recency_k = int(recency_k)             # 0 => legacy (all looks permanent)
        self._perm: list = []                        # admitted looks (all looks, in legacy)
        self._cur: list = []                         # current-batch expanded looks (recency mode)
        self._blocks: list = []                      # last <=recency_k finalized batch blocks
        self.mem = None                              # torch (M,768) on device (lazy)
        self._dirty = True
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if "perm" in z.files:
                if len(z["perm"]):
                    self._perm = [z["perm"].astype(np.float32)]
                if "recency" in z.files and "block_sizes" in z.files:
                    rec = z["recency"].astype(np.float32)
                    off = 0
                    for s in z["block_sizes"].astype(int):
                        if s:
                            self._blocks.append(rec[off:off + s])
                        off += int(s)
            elif len(z["mem"]):                      # legacy single-matrix file
                self._perm = [z["mem"].astype(np.float32)]

    # -- writes --
    def add_admitted(self, emb):
        """An admitted look joins memory PERMANENTLY (never re-buy a banked look)."""
        if emb is not None:
            self._perm.append(np.asarray(emb, np.float32).reshape(1, 768))
            self._dirty = True

    def add_expanded(self, emb):
        """An expanded node's look: recency window in recency mode, permanent in legacy."""
        if emb is None:
            return
        e = np.asarray(emb, np.float32).reshape(1, 768)
        (self._cur if self.recency_k > 0 else self._perm).append(e)
        self._dirty = True

    def end_batch(self):
        """Finalize the current batch's expanded-look block and evict blocks older than K."""
        if self.recency_k <= 0:
            return
        if self._cur:
            self._blocks.append(np.concatenate([a.reshape(-1, 768) for a in self._cur], axis=0))
            self._cur = []
        if len(self._blocks) > self.recency_k:
            self._blocks = self._blocks[-self.recency_k:]
        self._dirty = True

    # -- reduce --
    @staticmethod
    def _stack(lst):
        if not lst:
            return np.zeros((0, 768), np.float32)
        return np.concatenate([a.reshape(-1, 768) for a in lst], axis=0).astype(np.float32)

    def _all_rows(self):
        # RECENCY: the window is the last K COMPLETED batches (_blocks). The current batch's
        # expanded parents sit in _cur and are DELIBERATELY excluded from cos_max until
        # end_batch folds them into a block — otherwise every candidate is compared against
        # its own just-expanded parent (a child looks near-identical to its parent on the cheap
        # substrate), which pins cos_max past the knee and re-creates the run-2 saturation
        # regardless of memory size. _cur is only ever populated in recency mode; in legacy
        # add_expanded goes to _perm, so this exclusion is a no-op there.
        parts = [self._stack(self._perm)] + [b.reshape(-1, 768) for b in self._blocks]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts, axis=0).astype(np.float32) if parts \
            else np.zeros((0, 768), np.float32)

    def _rebuild(self):
        rows = self._all_rows()
        self.mem = torch.from_numpy(rows).to(self.device) if len(rows) else None
        self._dirty = False

    def cos_max(self, embs) -> np.ndarray:
        """Max cosine of each row of `embs` (normalized, N x 768) vs memory; 0 if empty."""
        if self._dirty:
            self._rebuild()
        n = len(embs)
        if self.mem is None or n == 0:
            return np.zeros(n, np.float32)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(embs, np.float32)).to(self.device)
            c = (x @ self.mem.T).max(dim=1).values
        return c.float().cpu().numpy()

    def save(self):
        perm = self._stack(self._perm)
        blocks = self._blocks + ([np.concatenate([a.reshape(-1, 768) for a in self._cur], axis=0)]
                                 if self._cur else [])
        rec = self._stack(blocks) if blocks else np.zeros((0, 768), np.float32)
        sizes = np.asarray([len(b) for b in blocks], np.int64)
        if not len(perm) and not len(rec):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / (self.path.stem + "_tmp.npz")
        np.savez_compressed(tmp, perm=perm, recency=rec, block_sizes=sizes)
        os.replace(tmp, self.path)

    @property
    def n_perm(self) -> int:
        return sum(len(a.reshape(-1, 768)) for a in self._perm)

    @property
    def n_recency(self) -> int:
        return sum(len(b) for b in self._blocks) + sum(len(a.reshape(-1, 768)) for a in self._cur)

    def __len__(self):
        return self.n_perm + self.n_recency


# --------------------------------------------------------------------------- #
# Run-scoped ledger (append-only jsonl + atomic npz feature store). Schema parity with
# production's outcome_ledger.jsonl; the q3 cloud is rebuilt from these rows.
# --------------------------------------------------------------------------- #
class RunLedger:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.path = run_dir / "outcome_ledger.jsonl"
        self.feats_path = run_dir / "outcome_feats.npz"
        self.rows: list[dict] = []
        self.feats: dict = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if self.feats_path.exists():
            z = np.load(self.feats_path, allow_pickle=False)
            self.feats = {k: z[k] for k in z.files}

    def append(self, row: dict, feat):
        row.setdefault("scorer_version", ps.SCORER_VERSION)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)
        if feat is not None:
            self.feats[row["id"]] = np.asarray(feat, np.float32)

    def save_feats(self):
        if not self.feats:
            return
        tmp = self.feats_path.parent / (self.feats_path.stem + "_tmp.npz")
        np.savez_compressed(tmp, **self.feats)
        os.replace(tmp, self.feats_path)

    def clouds(self, partitions: list[str]) -> dict:
        return {p: ps.build_cloud(self.rows, p) for p in partitions}


# --------------------------------------------------------------------------- #
# The driver.
# --------------------------------------------------------------------------- #
class SteeredFrontier:
    def _bulk_scratch(self) -> Path:
        """``<run_dir>/scratch`` as a bulk() path: routed out-of-tree via the resolver
        when run_dir is inside the repo, so a discovery run's scratch is born under
        ARTIFACTS_ROOT. A run_dir that is already outside the repo is left untouched
        (``<run_dir>/scratch``) — nothing to relocate."""
        try:
            rel = self.run_dir.relative_to(ROOT)
        except ValueError:
            return self.run_dir / "scratch"     # run_dir already out-of-tree
        return _paths.bulk(rel.as_posix() + "/scratch")

    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        # Per-run render/field scratch is a file-count bomb (campaign2 breadth/dive were
        # 317k files / ~45 GB). Declare it bulk() HERE, at the write site, so it is BORN
        # out-of-tree — not written in-tree and moved by hand afterward as campaign2's was.
        # paths.bulk() routes any data/discovery/**/scratch under ARTIFACTS_ROOT via the
        # single resolver, so a new campaign needs no registry line at all; a run whose
        # run_dir is already out-of-tree (or under scratch/) keeps <run_dir>/scratch.
        self.scratch = self._bulk_scratch()
        self.state_path = self.run_dir / "state.json"
        self.stop_path = self.run_dir / "STOP"
        self.harvest_log = self.run_dir / "harvest_log.jsonl"
        self.families = [f.strip() for f in args.families.split(",") if f.strip()]
        for f in self.families:
            if f not in C_PLANE:
                raise SystemExit(f"--families must be c-plane ({C_PLANE}); got {f!r}")
        self.B = args.batch or B_DEFAULT
        self.budget_s = args.budget * 60.0
        self.seed = args.seed
        # --- dive mode (single-track descent off a completed run's admissions) ---
        self.dive = bool(getattr(args, "dive", False))
        if self.dive and not getattr(args, "dive_source", None):
            raise SystemExit("--dive requires --dive-source <completed run dir>")
        # single-track dives don't spawn julia roots (no frontier); force the hook off there.
        self.julia_hook = bool(args.julia_hook) and not self.dive
        self.julia_hook_spacing = float(getattr(args, "julia_hook_spacing", JULIA_HOOK_SPACING))
        # PRIMARY julia supply under test (julia_parent_sourcing_probe): a file of c-diverse
        # near-∂M sampler c's, injected as julia:mandelbrot roots at fresh start (see
        # seed_julia_pool). None => current path (julia roots only via the parent-fired hook).
        jp = getattr(args, "julia_seed_pool", None)
        self.julia_seed_pool_path = Path(jp).resolve() if jp else None
        # item 5: cross-run coordinate freshness prior — seed this run's dup/rejection clouds
        # from prior-library admitted coords at start (ON by default; --no-freshness-prior off).
        self.freshness_prior = bool(getattr(args, "freshness_prior", False))
        self.julia_hooks_path = self.run_dir / "julia_hooks.jsonl"   # item 2: durable hook log
        self.dive_source = Path(args.dive_source).resolve() if getattr(args, "dive_source", None) else None
        self.dive_target_depth = int(getattr(args, "dive_target_depth", 23))
        self.dive_min_fw = float(getattr(args, "dive_min_fw", 2e-9))
        self.expand_min_fw = self.dive_min_fw if self.dive else None
        self.dive_state_path = self.run_dir / "dive_state.json"
        self.dive_log = self.run_dir / "dive_log.jsonl"
        self.cur_dive = None                             # (dive_id, start_group, source_id) live
        # v1.1 steering coefficients (both 0 -> byte-identical pilot behaviour). Dive forces the
        # morph term OFF (single-track has no frontier to steer; novelty is measured OFFLINE).
        self.lambda_m = 0.0 if self.dive else float(args.lambda_m)
        self.beta = float(args.beta)
        self.morph_lo, self.morph_hi, self.anchor_src = load_morph_anchors(
            args.morph_lo, args.morph_hi)
        self.prio_log = self.run_dir / "prio_terms.jsonl"
        self.sat_log = self.run_dir / "saturation.jsonl"
        # v1.2 morph-memory semantics: recency_k>0 => admitted-only + last-K-batch expanded
        # window (the saturation fix); 0 => legacy all-permanent (v1.1 default, reproduces).
        self.recency_k = int(args.recency_k) if getattr(args, "mem_recency", False) else 0
        # saturation = candidates whose novelty penalty is within 10% of full (cos_max past
        # 90% of the [lo,hi] ramp): a near-constant offset, not a gradient. Report shows this
        # dropping under the recency fix.
        self.sat_cos = self.morph_lo + 0.9 * (self.morph_hi - self.morph_lo)

        # partitions this run tracks a cloud for (c-plane + julia twins if hooked; dive covers
        # all twins so a start from any source partition has a cloud + tau_h).
        self.partitions = list(self.families)
        if self.julia_hook or self.dive or self.julia_seed_pool_path:
            self.partitions += [ps.julia_partition(f) for f in self.families]

        # --- minibrot maneuvers (v1.3). OFF => every path short-circuits on
        # `self.maneuvers is False` and the run is byte-identical to the pre-maneuver
        # frontier. Dives are single-track with no frontier to reserve slots in, so the
        # operators are forced off there. ---
        self.maneuvers = bool(getattr(args, "maneuvers", False)) and not self.dive
        self.man_quota = int(getattr(args, "maneuver_quota", MAN_QUOTA_DEFAULT))
        self.man_ks = mnv.parse_k_spec(getattr(args, "maneuver_k", MAN_K_DEFAULT))
        self.man_lateral = bool(getattr(args, "maneuver_lateral", True))
        self.man_log = self.run_dir / "maneuvers.jsonl"
        self.man_visited: set = set()      # canonical atom keys already turned into a node
        self.man_gov = mnv.ProbeGovernor(
            float(getattr(args, "maneuver_probe_p", MAN_PROBE_P_DEFAULT)),
            np.random.default_rng(self.seed + 9901))
        self.man_probe_s = 0.0             # cumulative probe+solve wall time (cost sizing)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)

        # Guarded scorer: cheap images (no field sidecar) pass through unguarded == raw
        # scoring; reframe tiles (DUMP_GUARD_FIELD) get the model-free field guard.
        assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
        reframe.DUMP_GUARD_FIELD = True
        self.scorer = guard.make_guarded_scorer(ps.SCORER_PATH)

        self.tau_h = derive_tau_h(self.partitions)

        # mutable run state
        self.frontier: list[dict] = []
        self.expansions_per_root: dict[str, int] = {}
        self.node_ctr = 0
        self.seq = 0
        self.batch_i = 0
        self.active_s = 0.0            # accumulated active wall time (survives resume)
        self.est_batch_s = 0.0
        self.totals = dict(expanded=0, candidates=0, harvest_checks=0,
                           canonical_q3=0, admitted=0, q3_dup=0, guarded=0,
                           julia_roots=0, julia_hooks_skipped=0, precanon_dup=0,
                           cap_hits=0, dead_nodes=0, novelty_hits=0,
                           nov_scored=0, sat_hits=0, distinct_looks=0,
                           # reconcile terms: every harvest check must land in exactly one
                           # of these buckets, checked per batch (see _reconcile_batch).
                           canon_not_q3=0, reframe_not_q3=0, render_failed=0,
                           frontier_pushed=0,
                           **{k: 0 for k in MAN_TOTALS})
        self.rng = np.random.default_rng(self.seed)
        # per-family native seeders (root source) — re-created fresh on resume.
        self.seeders = {f: ps.NativeSeeder(self.seed, self.scratch / f"native_{f}",
                                           np.random.default_rng(self.seed + i + 1),
                                           self._flags(f))
                        for i, f in enumerate(self.families)}

        self.ledger = RunLedger(self.run_dir)
        # cloud = (prior-library admitted coords, if the freshness prior is on) + this run's own
        # ledger rows. Prior rows live ONLY in the cloud (dedup/rejection/steering) — they never
        # enter self.ledger.rows and are never counted as this-run admissions. build_cloud dedups
        # the union, so a resume is idempotent.
        self.prior_rows = self.load_prior_library_rows() if self.freshness_prior else []
        self.clouds = self.build_clouds()
        self.run_clouds = self.build_run_clouds()
        self.hooked_c = defaultdict(list)                   # jpart -> [(c_re,c_im)] already hooked
        self.rebuild_hooked_c()

        # --- morph-novelty machinery (only when lambda_m > 0; off == pilot). ---
        self.clip_model = self.clip_tf = None
        self.node_embs: dict = {}                           # node_id -> normalized emb (frontier)
        clip_dev = "cpu"
        if self.lambda_m > 0.0:
            from tools.curation.colored_clip import load_clip   # noqa: E402  (heavy; lazy)
            self.clip_model, self.clip_tf = load_clip()
            clip_dev = str(next(self.clip_model.parameters()).device)
            self.node_embs = self.load_node_embs()
        self.morph = MorphMemory(clip_dev, self.run_dir / "morph_mem.npz", self.recency_k)

        # --- deficit scheduler (item: cross-partition allocation; default OFF). When off,
        # every path below short-circuits on `self.scheduler is None` and the run is
        # byte-identical to the pre-scheduler frontier. When on, the scheduler names which
        # partition's sub-queue to pop each batch (deficits/prices only — NEVER p_good, NEVER
        # the preference ranker) and deficit-weights the root family mix. ---
        self.scheduler = None
        self._served_partition = None
        self._sched_mt = None                # lazily-loaded (model, tf) for the canonical embed
        if getattr(args, "scheduler", False):
            self.scheduler = dsched.DeficitScheduler(
                self.partitions, self.run_dir,
                target_path=getattr(args, "scheduler_target", None),
                prices_path=getattr(args, "scheduler_prices", None))
            # Seed the distinct-look baseline from the campaign-1 library (per-partition medoid
            # embeddings) so deficits measure LIBRARY-WIDE scarcity, not run-local scarcity.
            # No-op on resume (tally reloaded from its npz; total > 0). Resume-safe: seeds and
            # persists the npz once, before the first batch.
            #
            # FAIL-CLOSED: no seed => UnseededRunError unless --allow-unseeded. main() runs the
            # same guard as a preflight so the CLI aborts before this object is even built (no
            # run dir, no ledger); repeating it here means a programmatic constructor cannot
            # slip past it either. `_library_seed` carries main()'s already-loaded record so
            # the embeddings are read exactly once.
            self._sched_seeded = self.scheduler.seed_from_library(
                record=getattr(args, "_library_seed", None),
                allow_unseeded=bool(getattr(args, "allow_unseeded", False)))

    def build_clouds(self) -> dict:
        """Per-partition q3 cloud from (prior-library rows ⊕ this run's ledger rows). Prior rows
        first so an earlier prior place wins a dedup cluster (item 5: the freshness prior).
        Consumers: the DEDUP path (pre-canonical filter, admission near-dup) and the soft steering
        penalty — everywhere the prior's "don't re-cover a library coord" intent belongs."""
        combined = self.prior_rows + self.ledger.rows
        return {p: ps.build_cloud(combined, p) for p in self.partitions}

    def build_run_clouds(self) -> dict:
        """Per-partition q3 cloud from THIS RUN's ledger rows ONLY — the freshness prior is
        deliberately excluded. Sole consumer: the native-seed rejection sampler in draw_roots
        (count_within(REJECT_RADIUS) >= Q3_DENSITY_CAP). That gate's fixed 0.20 radius + cap-5 was
        tuned for a cloud that STARTS EMPTY and accrues a handful of places; feeding it the 100s of
        prior places sterilizes the run — a productive-region seed has a median ~12 prior neighbours
        within 0.20, so ~98% of seeds reject on arrival and the sampler saturates at 0 roots (part-0
        finding). Scoping the sampler to the run cloud restores empty-start behaviour while the DEDUP
        clouds still carry the prior. Prior OFF => identical to self.clouds (prior_rows empty)."""
        return {p: ps.build_cloud(self.ledger.rows, p) for p in self.partitions}

    def load_prior_library_rows(self) -> list:
        """Item 5: every admitted-coord row in the prior library — all
        data/**/outcome_ledger.jsonl EXCEPT this run's own ledger. Read for the coordinate
        freshness prior only (their coords seed the dup/rejection clouds); nothing here enters
        this run's ledger or records. Matches campaign1_readout's prior-ledger enumeration."""
        rows = []
        own = self.ledger.path.resolve()
        for led in sorted((ROOT / "data").rglob("outcome_ledger.jsonl")):
            if led.resolve() == own:
                continue
            for line in open(led, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # Some ledgers (deep / q4_harvest / phoenix) serialize outcome coords as
                # high-precision decimal STRINGS. The prior rows feed float arithmetic in every
                # dedup/steering site (near_dup, dup_penalty, count_within), so coerce once here
                # at ingestion — float64 is ample for these O(1) dedup coords and lossless for the
                # purpose (prior rows are never re-rendered from this run).
                for k in ("outcome_cx", "outcome_cy", "outcome_fw"):
                    v = r.get(k)
                    if isinstance(v, str):
                        r[k] = float(v)
                rows.append(r)
        return rows

    def rebuild_hooked_c(self):
        """Reconstruct the set of already-hooked julia parameters so hook spacing survives a
        resume. Prefers the durable hook log (item 2 — records EVERY accepted hook, incl.
        zero-admit roots); falls back to the admitted julia ledger rows for pre-hook-log runs."""
        self.hooked_c = defaultdict(list)
        if self.julia_hooks_path.exists():
            for line in open(self.julia_hooks_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("hooked"):
                    self.hooked_c[r["jpart"]].append((float(r["c_re"]), float(r["c_im"])))
            return
        for r in self.ledger.rows:
            fam = r.get("family", "")
            if fam.startswith("julia:") and r.get("julia_c_re") is not None:
                self.hooked_c[fam].append((float(r["julia_c_re"]), float(r["julia_c_im"])))

    # --- c-plane family_flags for the native seeder / probe ---
    @staticmethod
    def _flags(family: str) -> list:
        return [] if family == "mandelbrot" else ["--family", family]

    # ---------------------------------------------------------------- morph
    @property
    def node_embs_path(self) -> Path:
        return self.run_dir / "node_embs.npz"

    def load_node_embs(self) -> dict:
        """Reload frontier-node embeddings (node_id -> normalized emb) so a resume can fold a
        popped node into morph memory. Keyed by str(node_id)."""
        p = self.node_embs_path
        if not p.exists():
            return {}
        z = np.load(p, allow_pickle=False)
        return {int(k): z[k].astype(np.float32) for k in z.files}

    def save_node_embs(self):
        """Persist embeddings only for node_ids still on the frontier (drop popped/pruned)."""
        if self.lambda_m <= 0.0:
            return
        live = {n["node_id"] for n in self.frontier}
        keep = {str(k): v for k, v in self.node_embs.items() if k in live}
        p = self.node_embs_path
        tmp = p.parent / (p.stem + "_tmp.npz")
        np.savez_compressed(tmp, **keep)
        os.replace(tmp, p)

    @torch.no_grad()
    def clip_embed(self, imgs: list, bs: int = 64) -> np.ndarray:
        """L2-normalized CLIP embeddings (library recipe) of PIL RGB images (N x 768)."""
        outs = []
        for i in range(0, len(imgs), bs):
            xb = torch.stack([self.clip_tf(im) for im in imgs[i:i + bs]])
            xb = xb.to(next(self.clip_model.parameters()).device)
            outs.append(self.clip_model(xb).float().cpu().numpy())
        E = np.concatenate(outs, axis=0).astype(np.float32)
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return E

    def fold_expanded_into_memory(self, batch):
        """A node that is about to be expanded joins morph memory (its cheap emb). Roots carry
        no emb and contribute nothing (they are whole-view seeds, not the near-repeats we damp)."""
        if self.lambda_m <= 0.0:
            return
        for n in batch:
            e = self.node_embs.pop(n["node_id"], None)
            if e is not None:
                self.morph.add_expanded(e)

    def score_morph(self, cands):
        """Embed each candidate's cheap twilight image, stash the normalized emb on the cand,
        and set cand['cos_max'] = max cosine vs the current morph memory (admitted + expanded).
        No-op (cos_max=0) when the novelty term is disabled."""
        for c in cands:
            c["cos_max"] = 0.0
            c["emb"] = None
        if self.lambda_m <= 0.0 or not cands:
            return
        imgs = []
        for c in cands:
            with Image.open(c["img"]) as im:
                im.load()
                imgs.append(im.convert("RGB"))
        E = self.clip_embed(imgs)
        cm = self.morph.cos_max(E)
        for c, e, v in zip(cands, E, cm):
            c["emb"] = e
            c["cos_max"] = float(v)

    def new_node_id(self) -> int:
        self.node_ctr += 1
        return self.node_ctr

    # ---------------------------------------------------------------- roots
    def draw_roots(self):
        """Draw a batch of native depth-1 seeds per family (q3-density rejection +
        depth-2 descendability probe) and enter the survivors as depth-1 frontier nodes
        with a neutral prior priority — exactly the current path's root pipeline."""
        added = 0
        # item 7: deficit-aware root mix. Scheduler ON => split the B draws across families by
        # their price-weighted, julia-twin-inclusive deficit; OFF => B per family (unchanged).
        alloc = (self.scheduler.root_allocation(self.families, self.B, self.rng)
                 if self.scheduler is not None else None)
        for fam in self.families:
            nb = alloc[fam] if alloc is not None else self.B
            if nb <= 0:
                continue
            # run-only cloud: the freshness prior must NOT feed this hard rejection gate (part-0
            # sterilization finding) — only this run's own accruing q3 places spread new seeds.
            cloud = self.run_clouds[fam]
            props = self.seeders[fam].draw_batch(cloud, nb)
            if not props:
                continue
            pw = self.scratch / f"roots_b{self.batch_i:04d}_{fam}"
            survivors, rejects, _ = ps.depth2_probe(props, pw, self.seed, self._flags(fam))
            for sv in survivors:
                nid = self.new_node_id()
                self.frontier.append(dict(
                    node_id=nid, root_id=nid, partition=fam, c=None,
                    cx=float(sv["seed_cx"]), cy=float(sv["seed_cy"]), fw=float(sv["fw"]),
                    depth=1, priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
                    cheap_eord=None, cheap_pgood=None, branch="root",
                    mix_source=sv.get("mix_source", "native"),
                ))
                added += 1
        return added

    def add_julia_root(self, partition: str, c, parent_oid: str):
        """Julia hook: a fixed z-plane base-scale root at the parent's outcome `c` — the
        current path's julia hook, fired per qualifying (admitted-q3) c-plane parent.

        Adaptation vs production: the steered frontier explores the z-plane, so a julia
        partition's OUTCOME cloud is keyed on the z-viewport (correct image-distinctness +
        steering penalty). Root spawning is instead gated by the PARAMETER c against a
        separate `hooked_c` set (so the same c is not re-hooked) — production keys its julia
        cloud on c directly; here the two roles are split."""
        jpart = ps.julia_partition(partition)
        cr, ci = float(c[0]), float(c[1])
        hooked = self.hooked_c[jpart]
        # item 3: hard 1-neighbour spacing on the seed c (replaces the Q3_DENSITY_CAP density
        # gate) — a parent whose c is within JULIA_HOOK_SPACING of an already-hooked c is
        # redundant (near-identical Julia set) and is skipped.
        nearest = min((math.hypot(hr - cr, hi - ci) for (hr, hi) in hooked), default=float("inf"))
        skipped = nearest < self.julia_hook_spacing
        # item 2: durably log EVERY hook decision (accepted + skipped) at hook time, so a
        # suppressed / zero-admit hook's seed c is recoverable without re-discovery.
        self._log_julia_hook(jpart, cr, ci, parent_oid, hooked=not skipped, nearest=nearest)
        if skipped:
            self.totals["julia_hooks_skipped"] += 1
            return False
        hooked.append((cr, ci))
        nid = self.new_node_id()
        self.frontier.append(dict(
            node_id=nid, root_id=nid, partition=jpart, c=[str(c[0]), str(c[1])],
            cx=0.0, cy=0.0, fw=JULIA_ROOT_FW, depth=1,
            priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
            cheap_eord=None, cheap_pgood=None, branch="julia_root",
            mix_source=f"julia_hook<{parent_oid}", parent_oid=parent_oid,
        ))
        self.totals["julia_roots"] += 1
        return True

    def _log_julia_hook(self, jpart, cr, ci, parent_oid, *, hooked: bool, nearest: float):
        """Item 2: append one hook decision to the durable per-run julia_hooks.jsonl. Records
        the seed c for EVERY hooked root (incl. zero-admit) and every spacing-skipped one — the
        gap the audit found unrecoverable (seed c was absent from harvest_log for zero-admit
        roots). `nearest` = c-distance to the closest already-hooked parent (inf if first)."""
        with open(self.julia_hooks_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(
                batch=self.batch_i, jpart=jpart, c_re=cr, c_im=ci, parent_oid=parent_oid,
                hooked=bool(hooked), nearest_c_dist=(None if nearest == float("inf") else nearest),
                spacing=self.julia_hook_spacing,
            )) + "\n")

    def seed_julia_pool(self) -> int:
        """PRIMARY julia supply under test (julia_parent_sourcing_probe). Inject the c-diverse
        near-∂M sampler pool as julia:mandelbrot base-scale z-plane roots at fresh start.

        Deliberately BYPASSES add_julia_root's hook-spacing gate: the sampler already dedups its
        c's at MIN_SEP (0.006) « the 0.2 hook spacing, so routing them through the gate would
        collapse the pool to a handful. Each injected c is registered in `hooked_c` so a later
        parent-fired hook whose seed c lands within spacing of a sampler c is suppressed — the
        hook stays available (§1 secondary path) but does not re-cover the sampler's ground.
        The pool is degree-2 near-∂M (z²+c), so every root is the julia:mandelbrot twin."""
        if self.julia_seed_pool_path is None:
            return 0
        jpart = ps.julia_partition("mandelbrot")
        if jpart not in self.partitions:
            raise SystemExit(f"--julia-seed-pool needs 'mandelbrot' in --families (for {jpart})")
        pool = json.loads(self.julia_seed_pool_path.read_text(encoding="utf-8"))
        added = 0
        for e in pool:
            cr, ci = float(e["c_re"]), float(e["c_im"])
            nid = self.new_node_id()
            self.frontier.append(dict(
                node_id=nid, root_id=nid, partition=jpart, c=[str(cr), str(ci)],
                cx=0.0, cy=0.0, fw=JULIA_ROOT_FW, depth=1,
                priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
                cheap_eord=None, cheap_pgood=None, branch="julia_root",
                mix_source="sampler",
            ))
            self.hooked_c[jpart].append((cr, ci))
            self.totals["julia_roots"] += 1
            added += 1
        print(f"[julia-seed-pool] injected {added} sampler-sourced {jpart} roots "
              f"(fw={JULIA_ROOT_FW}) from {self.julia_seed_pool_path.name}", flush=True)
        return added

    # ------------------------------------------------------------- maneuvers
    def propose_maneuvers(self, batch) -> int:
        """Fire the minibrot operators on the rungs about to be expanded, and push every
        available, not-yet-visited result as a new frontier NODE.

        A maneuver is a MOVE, so it enters as a node (unscored, neutral prior) exactly as a
        root does — not as a scored candidate. That is deliberate and it is the whole reason
        the reserved floor in `pop_batch` is needed: the active head has never seen a
        maneuver-originated view, so on score alone these would sink and the material needed
        to train its successor would never be generated.

        `lateral_to_sibling` reuses the snap's parent atom record, so a fired probe costs one
        atom-domain pass + the snap solves + the lateral's neighbourhood sweep, never two
        parent solves. Every probe decision is logged to `maneuvers.jsonl`."""
        if not self.maneuvers:
            return 0
        pushed = 0
        for n in batch:
            degree = mnv.degree_of(n["partition"])
            if degree is None:            # julia/phoenix viewport — operators undefined
                continue
            go, why = self.man_gov.should_probe(degree, n["cx"], n["cy"], n["fw"])
            if not go:
                self._log_maneuver(dict(batch=self.batch_i, op="probe", fired=False,
                                        skip=why, partition=n["partition"],
                                        node_id=n["node_id"], depth=n["depth"],
                                        cx=n["cx"], cy=n["cy"], fw=n["fw"]))
                continue
            view = dict(node_id=n["node_id"], cx=n["cx"], cy=n["cy"], fw=n["fw"],
                        depth=n["depth"])
            t0 = time.time()
            parent_rec = None
            # ONE solve, one row per framing: the nucleus does not depend on k, so adding
            # a k to the set costs a reframing, not another probe.
            for m in mnv.snap_to_nucleus_multi(view, self.man_ks, degree=degree):
                pushed += self._consume_maneuver(m, n)
                if m.available and parent_rec is None:
                    parent_rec = dict(id=m.atom_id, cx=m.cx, cy=m.cy, period=m.period,
                                      window_scale=m.window_scale, degree=degree)
            if self.man_lateral:
                m = mnv.lateral_to_sibling(view, self.rng, degree=degree,
                                           parent_rec=parent_rec)
                pushed += self._consume_maneuver(m, n)
            self.man_probe_s += time.time() - t0
        # mirror the governor's counters into totals (the checkpointed, resumable copy)
        g = self.man_gov
        self.totals["man_probes_rolled"] = g.n_rolled
        self.totals["man_probes_fired"] = g.n_fired
        self.totals["man_probes_coin_skip"] = g.n_coin_skip
        self.totals["man_probes_cache_skip"] = g.n_cache_skip
        return pushed

    def _consume_maneuver(self, m, parent) -> int:
        """Record a maneuver outcome and, when it is available AND new, push its node."""
        row = m.as_row()
        row.update(batch=self.batch_i, partition=parent["partition"],
                   root_id=parent["root_id"])
        if not m.available:
            self.totals["man_op_unavailable"] += 1
            row["used"] = False
            self._log_maneuver(row)
            return 0
        self.totals["man_op_available"] += 1
        # Multiple frontier members snapping to ONE nucleus is the normal case; the
        # read-time canonical key (snap_near_zero + sector-canonical rounding) is what
        # collapses them. Framing (k) is part of the identity — the same atom at two k's
        # is two distinct views — so the visited key carries k.
        vkey = f"{m.atom_key}|{m.op}|{m.k}"
        if vkey in self.man_visited:
            self.totals["man_avail_unused"] += 1
            row["used"] = False
            row["unused_reason"] = "atom_already_visited"
            self._log_maneuver(row)
            return 0
        self.man_visited.add(vkey)
        nid = self.new_node_id()
        man = dict(op=m.op, k=m.k, origin_node_id=nid, atom_id=m.atom_id,
                   atom_key=m.atom_key, period=m.period, log10_abs_A=m.log10_abs_A,
                   window_scale=m.window_scale, degree=m.extra.get("degree"),
                   parent_node_id=m.parent_node_id, parent_cx=m.parent_cx,
                   parent_cy=m.parent_cy, parent_fw=m.parent_fw,
                   parent_depth=m.parent_depth)
        self.frontier.append(dict(
            node_id=nid, root_id=parent["root_id"], partition=parent["partition"],
            c=parent["c"], cx=float(m.cx), cy=float(m.cy), fw=float(m.fw),
            depth=int(m.depth), branch="maneuver",
            priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL) + self.beta * int(m.depth),
            cheap_eord=None, cheap_pgood=None,
            mix_source=f"maneuver:{m.op}:k={m.k}", man=man,
        ))
        self.totals["man_nodes_pushed"] += 1
        row["used"] = True
        row["node_id_pushed"] = nid
        self._log_maneuver(row)
        return 1

    def _log_maneuver(self, row: dict):
        with open(self.man_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _split_reserved(self, pool: list[dict]) -> tuple[list[dict], list[dict]]:
        """Take the batch out of a PRIORITY-SORTED `pool`, honouring the maneuver floor.

        The floor is a reserved count of SLOTS, not a probability and not a priority bonus:
        the walker already ranks a slate, so a new proposal source needs a slot. It is a
        quota **of available** — with ~17% Newton convergence the operator is often simply
        not there, and an unfillable quota must never stall the frontier, so whatever is not
        filled falls straight back to the ordinary priority order.

        `pref_loc_v1` (the preference ranker) is ABSENT from this seam, as it is from
        `pop_batch_scheduled`: reserving a slot is not a ranker change, and the
        ranks-never-steers boundary is untouched by it."""
        if not self.maneuvers or self.man_quota <= 0:
            return pool[:self.B], pool[self.B:]
        plain = pool[:self.B]
        plain_ids = {n["node_id"] for n in plain}
        man = [n for n in pool if n.get("man")]
        take = min(self.man_quota, len(man), self.B)
        self.totals["man_quota_unfilled"] += self.man_quota - take
        if take <= 0:
            return plain, pool[self.B:]
        reserved = man[:take]
        self.totals["man_quota_bound"] += sum(1 for n in reserved
                                              if n["node_id"] not in plain_ids)
        taken = {n["node_id"] for n in reserved}
        rest = [n for n in pool if n["node_id"] not in taken]
        batch = reserved + rest[:self.B - take]
        return batch, rest[self.B - take:]

    # ---------------------------------------------------------------- expand
    def pop_batch(self) -> list[dict]:
        """Top-B expandable nodes by priority. A node whose root has hit the M cap can NEVER be
        expanded, so it is EVICTED from the frontier (not merely skipped): a capped root spawns
        ~M*b children before capping, so if capped nodes are retained they accumulate ~b faster
        than they drain and eventually saturate FRONTIER_CAP, starving pop_batch and forcing
        perpetual root replenishment (observed live at batch ~110: 100% of a 6000-node frontier
        was capped-root dead weight, throughput ~0). Eviction is a no-op below the cap regime the
        pilot ran in (few caps, frontier << cap), so it does not change short-run behaviour."""
        self.frontier.sort(key=lambda n: -n["priority"])
        live = []
        for n in self.frontier:
            if self.expansions_per_root.get(str(n["root_id"]), 0) >= M_CAP:
                self.node_embs.pop(n["node_id"], None)   # evict: capped root -> dead weight
                continue
            live.append(n)
        batch, rest = self._split_reserved(live)
        self.frontier = rest
        # cap_hits = distinct roots that have reached the M cap (derived, not per-batch).
        self.totals["cap_hits"] = sum(1 for v in self.expansions_per_root.values() if v >= M_CAP)
        for n in batch:
            self.expansions_per_root[str(n["root_id"])] = \
                self.expansions_per_root.get(str(n["root_id"]), 0) + 1
        return batch

    def pop_batch_scheduled(self) -> list[dict]:
        """Scheduler pop: the cross-partition CHOICE is the scheduler's (deficits/prices only),
        the within-partition ORDER is the unchanged priority sort over that ONE partition's
        nodes. Capped-root dead weight is evicted from the whole frontier first (same rationale
        as pop_batch). Sets self._served_partition (the batch's active-time charge target)."""
        live = []
        for n in self.frontier:
            if self.expansions_per_root.get(str(n["root_id"]), 0) >= M_CAP:
                self.node_embs.pop(n["node_id"], None)      # evict: capped root -> dead weight
            else:
                live.append(n)
        self.frontier = live
        self.totals["cap_hits"] = sum(1 for v in self.expansions_per_root.values() if v >= M_CAP)
        queue_lens = {}
        for n in self.frontier:
            queue_lens[n["partition"]] = queue_lens.get(n["partition"], 0) + 1
        # RANKER SCOPE (item 9): the partition choice uses ONLY per-partition deficits/prices
        # (dsched.choose_partition). The preference ranker (pref_loc_*) ranks ADMITTED output
        # for keeper/emission ordering ONLY and MUST NOT enter scheduling — it is absent here.
        part = self.scheduler.pick_partition(queue_lens, self.rng)
        self.scheduler.log_choice(self.batch_i, part, queue_lens)
        self._served_partition = part
        if part is None:
            return []
        pool = [n for n in self.frontier if n["partition"] == part]
        pool.sort(key=lambda n: -n["priority"])
        batch, _rest = self._split_reserved(pool)   # maneuver floor is WITHIN the served partition
        taken = {n["node_id"] for n in batch}
        self.frontier = [n for n in self.frontier if n["node_id"] not in taken]
        for n in batch:
            self.expansions_per_root[str(n["root_id"])] = \
                self.expansions_per_root.get(str(n["root_id"]), 0) + 1
        return batch

    # ------------------------------------------------------------ scheduler embed
    def _sched_clip(self):
        """(model, tf) for the canonical-render embed. Reuses the novelty CLIP if already
        loaded (same vit_base_patch16_clip_224.openai), else loads once and caches."""
        if self.clip_model is not None:
            return self.clip_model, self.clip_tf
        if self._sched_mt is None:
            from tools.curation.colored_clip import load_clip   # noqa: E402 (heavy; lazy)
            self._sched_mt = load_clip()
        return self._sched_mt

    def scheduler_embed_admitted(self, row) -> np.ndarray:
        """L2-normalized CLIP embedding of an admitted location's CANONICAL render via the
        LIBRARY morph recipe (640x360 ss2 smooth field -> robust-z tanh gray -> CLIP) — the
        exact recipe emission clusters at cos 0.974, so the scheduler's distinct-look tally is
        consistent with the library's type x morph occupancy. One embed per admission."""
        from tools.emission import descriptor as D            # noqa: E402
        from tools.wallpaper import library_annotate as la    # noqa: E402
        from tools.curation.colored_clip import embed_clip     # noqa: E402
        loc = D.location_of(row)
        fcache = self.scratch / "sched_fields"
        field = la.ensure_field(loc, retain=False, tmp_dir=fcache, cache_root=fcache)
        gray = la.morph_gray_image(field)
        model, tf = self._sched_clip()
        emb = embed_clip(model, tf, [gray])[0].astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    # ---------------------------------------------------------------- reconcile
    RECONCILE_KEYS = ("candidates", "frontier_pushed", "harvest_checks", "precanon_dup",
                      "canonical_q3", "canon_not_q3", "render_failed", "admitted",
                      "q3_dup", "guarded", "reframe_not_q3")

    def _reconcile_snapshot(self) -> dict:
        return {k: self.totals[k] for k in self.RECONCILE_KEYS}

    def _reconcile_batch(self, before: dict, n_cands: int):
        """`found == written + dropped_*` per work unit, or EXIT LOUD.

        Two identities have to close on every batch, and a long unattended run that silently
        loses candidates is exactly the failure a summary cannot show you afterwards:

          1. FRONTIER   every scored candidate is pushed:
                        candidates == frontier_pushed
          2. HARVEST    every check lands in exactly one fate:
                        harvest_checks == precanon_dup + canonical_q3 + canon_not_q3
                        canonical_q3   == admitted + q3_dup + guarded + reframe_not_q3

        (`render_failed` checks are subtracted from `harvest_checks` at the point of failure,
        so they are outside both identities by construction rather than by omission.)"""
        d = {k: self.totals[k] - before[k] for k in self.RECONCILE_KEYS}
        problems = []
        if d["candidates"] != d["frontier_pushed"]:
            problems.append(f"frontier: found {d['candidates']} candidates but pushed "
                            f"{d['frontier_pushed']}")
        if d["harvest_checks"] != d["precanon_dup"] + d["canonical_q3"] + d["canon_not_q3"]:
            problems.append(
                f"harvest: {d['harvest_checks']} checks != precanon_dup {d['precanon_dup']} "
                f"+ canonical_q3 {d['canonical_q3']} + canon_not_q3 {d['canon_not_q3']}")
        if d["canonical_q3"] != d["admitted"] + d["q3_dup"] + d["guarded"] + d["reframe_not_q3"]:
            problems.append(
                f"q3 fates: {d['canonical_q3']} canonical_q3 != admitted {d['admitted']} "
                f"+ q3_dup {d['q3_dup']} + guarded {d['guarded']} + reframe_not_q3 "
                f"{d['reframe_not_q3']}")
        if problems:
            raise SystemExit(f"[reconcile] batch {self.batch_i} DOES NOT BALANCE "
                             f"(n_cands={n_cands}):\n  " + "\n  ".join(problems))

    def unit_timeout_s(self) -> float:
        """Hard-kill backstop for ONE subprocess work unit (an --expand call, a confirmation
        render), bounded by what is left of the wall-clock budget.

        The standing `EXPAND_TIMEOUT_S` is 900s. On a run whose whole budget is 15 minutes
        that backstop is LONGER THAN THE RUN: a single hung unit doubles the run's wall
        clock while the cap logic sits there believing it is inside budget. So when a budget
        is set the timeout is additionally clamped to the REMAINING budget, floored at
        `MIN_UNIT_TIMEOUT_S` so a legitimately slow unit near the end is not shot merely for
        being slow. With no budget the historical 900s stands unchanged."""
        if not self.budget_s:
            return float(EXPAND_TIMEOUT_S)
        remaining = max(0.0, self.budget_s - self.active_s)
        return float(min(EXPAND_TIMEOUT_S, max(MIN_UNIT_TIMEOUT_S, remaining)))

    def expand_group(self, key, nodes) -> list[dict]:
        partition, c = key
        gdir = self.scratch / f"expand_b{self.batch_i:04d}" / f"{partition.replace(':','_')}"
        gdir.mkdir(parents=True, exist_ok=True)
        nodes_in = gdir / "nodes.jsonl"
        with open(nodes_in, "w", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps(dict(node_id=n["node_id"], root_id=n["root_id"],
                                        cx=n["cx"], cy=n["cy"], fw=n["fw"], depth=n["depth"])) + "\n")
        cmd = [str(BIN), "guided-descend", "--expand", str(nodes_in),
               "--seed", str(self.seed), "--out-dir", str(gdir)] + EXPAND_FLAGS + \
              descend_flags(partition, c)
        if self.expand_min_fw is not None:               # dive: stop before the fw floor w/ margin
            cmd += ["--min-fw", repr(self.expand_min_fw)]
        to = self.unit_timeout_s()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=to)
        except subprocess.TimeoutExpired:
            print(f"  WARN expand group {partition} timed out ({to:.0f}s) — skipped", flush=True)
            return []
        if r.returncode != 0:
            print(f"  WARN expand group {partition} failed: {r.stderr[-400:]}", flush=True)
            return []
        by_node = {n["node_id"]: n for n in nodes}
        cands = []
        ep = gdir / "expand.jsonl"
        if not ep.exists():
            return []
        for line in open(ep, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parent = by_node[row["node_id"]]
            if row["kind"] == "dead":
                self.totals["dead_nodes"] += 1
                continue
            cands.append(dict(
                node_id=self.new_node_id(), root_id=parent["root_id"],
                partition=partition, c=c,
                cx=float(row["cx"]), cy=float(row["cy"]), fw=float(row["fw"]),
                depth=int(row["depth"]), branch=row["branch"],
                mix_source=parent.get("mix_source"),   # propagate root supply for harvest attribution
                # A maneuver's whole subtree stays attributable to the operator that made
                # it: op / k / origin node / parent view ride down every rung, so a later
                # read never has to reconstruct lineage from coordinates.
                man=parent.get("man"),
                img=str((gdir / row["img"]).resolve()),
                int_frac=row["int_frac"], occ=row["occ"],
            ))
        return cands

    def expand_batch(self, batch) -> list[dict]:
        # group by (partition, tuple(c)) so each --expand call is homogeneous in kernel.
        groups: dict = {}
        for n in batch:
            key = (n["partition"], tuple(n["c"]) if n["c"] else None)
            groups.setdefault(key, []).append(n)
        cands = []
        for key, nodes in groups.items():
            cands += self.expand_group(key, nodes)
        return cands

    # ---------------------------------------------------------------- score
    def score_cheap(self, cands):
        if not cands:
            return
        triples = self.scorer.score_paths([c["img"] for c in cands])
        for c, (eord, nb, pg) in zip(cands, triples):
            c["cheap_eord"] = float(eord)
            c["cheap_nb"] = float(nb)
            c["cheap_pgood"] = float(pg)

    # ---------------------------------------------------------------- harvest
    def harvest(self, cands):
        """cheap p_good >= tau_h -> single canonical render + decode -> if q3, reframe +
        near-dup + admission. Logs every harvest check's (cheap, canonical, decode) triple."""
        checks = [c for c in cands if c["cheap_pgood"] >= self.tau_h[c["partition"]]]
        if not checks:
            return
        self.totals["harvest_checks"] += len(checks)
        # item 4: PRE-CANONICAL coord-dup filter. A candidate already inside an admitted q3's
        # dedup radius (under the FIXED seed-c-aware metric) cannot escape it via reframe (center
        # nudge <=0.25*fw << the dedup radius), so it can never become a distinct admission —
        # skip its canonical confirmation render entirely. This pushes the existing pre-reframe
        # skip one render earlier; under the fixed metric it is correctly aimed at the genuine
        # churn (multibrot4 + residual same-c julia descent), not distinct-c julias. The cloud is
        # read as of batch start; the in-loop pre-reframe check below still catches intra-batch
        # admits. Saved renders are counted (precanon_dup) for the readout.
        kept = []
        for c in checks:
            distinct, dup_of = ps.is_distinct(c["cx"], c["cy"], c["fw"],
                                              self.clouds.get(c["partition"], []), ps.DEDUP_K,
                                              c=ps.as_c(c["c"]))
            if distinct:
                kept.append(c)
            else:
                self.totals["precanon_dup"] += 1
                self._log_harvest(c, admitted=False, reframe_decoded=None, precanon_dup=dup_of)
        checks = kept
        if not checks:
            return
        # 1. batch the single canonical confirmation renders (640x360 ss2, the reward fidelity).
        cdir = self.scratch / f"harvest_b{self.batch_i:04d}"
        cdir.mkdir(parents=True, exist_ok=True)
        import concurrent.futures as cf
        tiles = []
        for i, c in enumerate(checks):
            tiles.append(cdir / f"confirm_{i:04d}.jpg")
        with cf.ThreadPoolExecutor(max_workers=ps.WORKERS) as ex:
            futs = {ex.submit(prescreen._render, c["cx"], c["cy"], c["fw"], tiles[i],
                              family=render_family_of(c["partition"]), c=c["c"],
                              timeout=self.unit_timeout_s()): i
                    for i, c in enumerate(checks)}
            for fut in cf.as_completed(futs):
                fut.result()
        # A timed-out / failed confirmation render leaves no tile; scoring a missing path
        # would raise deep in the scorer, so drop those checks here and count them.
        missing = [i for i, t in enumerate(tiles) if not t.exists()]
        if missing:
            self.totals["render_failed"] += len(missing)
            keep = [i for i in range(len(checks)) if i not in set(missing)]
            checks = [checks[i] for i in keep]
            tiles = [tiles[i] for i in keep]
            self.totals["harvest_checks"] -= len(missing)
            if not checks:
                return
        # K-aware (`score_paths_k`): on a K=4 head the third cutpoint comes back too, so the
        # confirmation decode can reach class 4 instead of being capped at 3 by the reader.
        rows_k = self.scorer.score_paths_k([str(t) for t in tiles])
        for c, row in zip(checks, rows_k):
            eord, nb, pg = row[0], row[1], row[2]
            pg4 = row[3] if len(row) > 3 else None
            c["canon_nb"], c["canon_pg"], c["canon_eord"] = float(nb), float(pg), float(eord)
            c["canon_pge4"] = None if pg4 is None else float(pg4)
            c["canon_decoded"] = corn_decode(nb, pg, ps.t_good_for(c["partition"]), pg4)

        # 2. reframe + admit the canonical-q3 confirmations. Cheap pre-reframe dedup:
        # reframe only nudges the center by <=0.25*fw and fw by <=1.41x, so a candidate
        # already inside an admitted q3's dedup radius cannot escape it — skip the 12-render
        # reframe and log it as a dup (this is where most compute is saved in a hot region).
        for c in checks:
            admitted = False
            reframe_decoded = None
            if c["canon_decoded"] >= 3:          # q3+ — a canonical class-4 confirms too
                self.totals["canonical_q3"] += 1
                pre_distinct, _ = ps.is_distinct(c["cx"], c["cy"], c["fw"],
                                                 self.clouds.get(c["partition"], []), ps.DEDUP_K,
                                                 c=ps.as_c(c["c"]))
                if not pre_distinct:
                    self.totals["q3_dup"] += 1
                else:
                    admitted, reframe_decoded = self.admit(c, cdir)
            else:
                self.totals["canon_not_q3"] += 1
            self._log_harvest(c, admitted, reframe_decoded)

    def admit(self, c, cdir):
        """Existing reframe + near-dup + admission path (guarded scorer, per-partition t_good)."""
        loc = loc_of(c["partition"], c["c"], c["cx"], c["cy"], c["fw"])
        wd = cdir / f"reframe_n{c['node_id']}"
        res = reframe.reframe_location(loc, scorer=self.scorer, seed=0, workdir=wd, workers=ps.WORKERS)
        guard_pass = res.score > guard.GUARD_SENTINEL + 1e-6
        nb, pg, pg4 = ps._chosen_probs(res)
        t_good = ps.t_good_for(c["partition"])
        decoded = corn_decode(nb, pg, t_good, pg4) if guard_pass else None
        is_q3 = guard_pass and (decoded or 0) >= 3   # q3+ — class 4 admits too
        ocx, ocy, ofw = float(res.cx), float(res.cy), float(res.fw)
        distinct, dup_of = (False, None)
        if is_q3:
            distinct, dup_of = ps.is_distinct(ocx, ocy, ofw, self.clouds[c["partition"]],
                                              ps.DEDUP_K, c=ps.as_c(c["c"]))

        run_ts = self.run_dir.name
        id_tag = {"mandelbrot": "m"}.get(c["partition"], c["partition"].replace(":", "_"))
        oid = f"st_{id_tag}_{run_ts}_{self.seq:06d}"
        self.seq += 1
        feat = None
        if is_q3 and distinct:
            tile = cdir / f"{oid}.jpg"
            feat = ps.outcome_feature(self.scorer, ocx, ocy, ofw, tile,
                                      family=render_family_of(c["partition"]), c=c["c"])
        row = dict(
            id=oid, ts=run_ts, family=c["partition"], mix_source="steered",
            node_id=c["node_id"], root_id=c["root_id"],
            seed_cx=c["cx"], seed_cy=c["cy"],
            outcome_cx=ocx, outcome_cy=ocy, outcome_fw=ofw,
            k3=float(res.score), raw_top3=[float(c["cheap_eord"])],
            reached_depth=int(c["depth"]),
            decoded_class=decoded, p_notbad=nb, p_good=pg, p_ge4=pg4, t_good=t_good,
            distinct=distinct, dup_of=dup_of,
            guard_pass=guard_pass, guard_fail=None if guard_pass else "sentinel",
            cheap_pgood=c["cheap_pgood"], canon_pgood=c["canon_pg"], branch=c["branch"],
            # fw + depth on EVERY row: a maneuver's snap-and-rescale changes fw without
            # changing the walk-rung count, so after this feature depth is no longer a
            # monotone stand-in for scale. Any later read has to depth-match on BOTH or it
            # measures depth (outcome_fw/reached_depth above; seed_fw here for the pre-
            # reframe view).
            seed_fw=c["fw"],
        )
        if c.get("man"):                             # maneuver-originated lineage
            row["maneuver"] = c["man"]
            row["mix_source"] = c.get("mix_source") or row["mix_source"]
        if c["c"] is not None:                       # julia twin outcome carries the parameter c
            row["julia_c_re"], row["julia_c_im"] = c["c"][0], c["c"][1]
            # CAMPAIGN schema (outcome_* = viewport, c = julia_c_*): stamp it so the row is
            # born tagged and no reader has to infer the era from field presence.
            row[jls.SCHEMA_KEY] = jls.CAMPAIGN
        if self.cur_dive is not None:                # dive: stamp provenance for the read
            row["mix_source"] = "dive"
            row["dive_id"], row["dive_start_group"], row["dive_source_id"] = self.cur_dive
        self.ledger.append(row, feat)
        if is_q3 and distinct:
            self.clouds[c["partition"]].append(row)
            self.run_clouds[c["partition"]].append(row)   # keep the rejection-sampler cloud current
            self.totals["admitted"] += 1
            if c.get("man"):
                self.totals["man_admitted"] += 1
            # fold the admitted location's look into morph memory (its cheap emb; reframe only
            # nudges the frame <=0.25*fw, so the candidate's cheap look stands in for it).
            if self.lambda_m > 0.0 and c.get("emb") is not None:
                self.morph.add_admitted(c["emb"])
            # scheduler: DISTINCT-LOOK tally (item 3). Embed the CANONICAL render via the
            # library morph recipe and count it iff cos_max < 0.974 vs this partition's set.
            if self.scheduler is not None:
                emb = self.scheduler_embed_admitted(row)
                if self.scheduler.on_admission(c["partition"], emb):
                    self.totals["distinct_looks"] = self.totals.get("distinct_looks", 0) + 1
            # julia hook: fire per qualifying (admitted-q3) c-plane parent.
            if self.julia_hook and c["partition"] in self.families:
                self.add_julia_root(c["partition"], (ocx, ocy), oid)
            return True, decoded
        elif is_q3:
            self.totals["q3_dup"] += 1
        elif not guard_pass:
            self.totals["guarded"] += 1
        else:
            # guard passed but the REFRAMED frame decoded below q3 — previously the one
            # uncounted fate, which is why a per-unit reconcile could not close.
            self.totals["reframe_not_q3"] += 1
        return False, decoded

    def _log_harvest(self, c, admitted, reframe_decoded, precanon_dup=None):
        # precanon_dup (item 4): the dup_of id when this check was skipped BEFORE its canonical
        # render by the pre-canonical filter (no canon_* fields exist for it); None otherwise.
        # cx/cy/fw (+ julia seed c) make EVERY reject fate renderable from the log alone — the
        # gap the julia audit hit (precanon_dup / canonical-not-q3 rejects had no coords, so no
        # human could ever eyeball them). Cheap: 3 floats + an optional c pair per harvest check.
        jc = c.get("c")
        with open(self.harvest_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(
                batch=self.batch_i, partition=c["partition"], depth=c["depth"],
                node_id=c["node_id"], root_id=c["root_id"],
                cx=c["cx"], cy=c["cy"], fw=c["fw"],
                julia_c_re=(None if jc is None else str(jc[0])),
                julia_c_im=(None if jc is None else str(jc[1])),
                cheap_pgood=c["cheap_pgood"], cheap_eord=c["cheap_eord"],
                canon_nb=c.get("canon_nb"), canon_pgood=c.get("canon_pg"),
                canon_pge4=c.get("canon_pge4"),   # third cutpoint (None on a K=3 head)
                canon_decoded=c.get("canon_decoded"), reframe_decoded=reframe_decoded,
                admitted=bool(admitted), tau_h=self.tau_h[c["partition"]],
                precanon_dup=precanon_dup, mix_source=c.get("mix_source"),
                maneuver=c.get("man"),   # operator/k/origin ride every check, admitted or not
            )) + "\n")

    # ---------------------------------------------------------------- push
    def push_children(self, cands):
        prio_rows = []
        batch_sat = 0                                    # saturated candidates this batch
        for c in cands:
            dup_pen = dup_penalty(c["cx"], c["cy"], self.clouds.get(c["partition"], []))
            cos_max = float(c.get("cos_max", 0.0))
            g = gumbel(self.rng, T_GUMBEL)               # RNG draw order unchanged from pilot
            prio, terms = priority_terms(
                c["cheap_eord"], g, dup_pen, cos_max,
                self.lambda_m, self.beta, c["depth"], self.morph_lo, self.morph_hi)
            self.frontier.append(dict(
                node_id=c["node_id"], root_id=c["root_id"], partition=c["partition"], c=c["c"],
                cx=c["cx"], cy=c["cy"], fw=c["fw"], depth=c["depth"], priority=prio,
                cheap_eord=c["cheap_eord"], cheap_pgood=c["cheap_pgood"], branch=c["branch"],
                mix_source=c.get("mix_source"),   # carry root supply down the tree (probe attribution)
                man=c.get("man"),                 # maneuver provenance, if this lineage has one
            ))
            self.totals["frontier_pushed"] += 1
            if c.get("emb") is not None:
                self.node_embs[c["node_id"]] = c["emb"]
            if terms["nov_pen"] > 0.0:
                self.totals["novelty_hits"] += 1
            if cos_max >= self.sat_cos:                  # within 10% of full penalty
                batch_sat += 1
            prio_rows.append(dict(
                batch=self.batch_i, node_id=c["node_id"], root_id=c["root_id"],
                partition=c["partition"], depth=c["depth"],
                **{k: round(v, 5) for k, v in terms.items()},
            ))
        # per-batch saturation fraction (the v1.2 novelty-fix telemetry): fraction of pushed
        # candidates whose novelty penalty is within 10% of full. A high fraction => the term
        # is a constant offset, not a gradient. Logged so the report shows it drop under the fix.
        if self.lambda_m > 0.0 and cands:
            self.totals["nov_scored"] += len(cands)
            self.totals["sat_hits"] += batch_sat
            with open(self.sat_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(
                    batch=self.batch_i, n=len(cands), sat=batch_sat,
                    frac=round(batch_sat / len(cands), 4),
                    mem_perm=self.morph.n_perm, mem_recency=self.morph.n_recency,
                    mem_total=len(self.morph),
                )) + "\n")
        if prio_rows:
            with open(self.prio_log, "a", encoding="utf-8") as f:
                for r in prio_rows:
                    f.write(json.dumps(r) + "\n")
        # prune to the memory bound (keep the best); drop pruned nodes' cached embeddings.
        # Maneuver-originated nodes are exempt: they are the population the reserved floor
        # exists to protect, and pruning by priority would delete them first (they carry a
        # neutral prior, or a score from a head that has never seen their kind) — the cap
        # would silently undo the floor. The exemption is bounded by the same probe
        # governor that bounds how many can ever be created.
        if len(self.frontier) > FRONTIER_CAP:
            self.frontier.sort(key=lambda n: -n["priority"])
            keep_man = [n for n in self.frontier if n.get("man")] if self.maneuvers else []
            room = max(0, FRONTIER_CAP - len(keep_man))
            others = [n for n in self.frontier if not n.get("man")] if self.maneuvers \
                else self.frontier
            kept_ids = {n["node_id"] for n in keep_man} | {n["node_id"] for n in others[:room]}
            dropped = [n for n in self.frontier if n["node_id"] not in kept_ids]
            self.frontier = [n for n in self.frontier if n["node_id"] in kept_ids]
            for n in dropped:
                self.node_embs.pop(n["node_id"], None)

    # ---------------------------------------------------------------- state
    def save_state(self):
        state = dict(
            run_ts=self.run_dir.name, families=self.families, julia_hook=self.julia_hook,
            seed=self.seed, B=self.B, budget_s=self.budget_s, tau_h=self.tau_h,
            lambda_m=self.lambda_m, beta=self.beta, recency_k=self.recency_k,
            morph_lo=self.morph_lo, morph_hi=self.morph_hi, anchor_src=self.anchor_src,
            node_ctr=self.node_ctr, seq=self.seq, batch_i=self.batch_i,
            active_s=self.active_s, est_batch_s=self.est_batch_s,
            expansions_per_root=self.expansions_per_root, totals=self.totals,
            frontier=self.frontier, rng=self.rng.bit_generator.state,
        )
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.maneuvers:
            # The visited-atom set and the governor's region cache are the two pieces of
            # maneuver state a resume must not lose: without them a restart re-pays the
            # Newton cost for regions the killed run already probed and re-pushes nodes the
            # ledger already carries.
            state["maneuvers"] = dict(
                quota=self.man_quota, ks=[("none" if k is None else k) for k in self.man_ks],
                lateral=self.man_lateral, probe_s=self.man_probe_s,
                visited=sorted(self.man_visited), governor=self.man_gov.state_dict())
        # morph memory + frontier-node embeddings first (state.json references them), then the
        # checkpoint. Both are heuristic (priority only) — a stale copy never loses an admission.
        self.morph.save()
        self.save_node_embs()
        if self.scheduler is not None:      # distinct-look embedding sets (per-partition npz)
            self.scheduler.save()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.state_path)
        self.ledger.save_feats()

    def load_state(self):
        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.node_ctr = st["node_ctr"]; self.seq = st["seq"]; self.batch_i = st["batch_i"]
        self.active_s = st["active_s"]; self.est_batch_s = st["est_batch_s"]
        self.expansions_per_root = st["expansions_per_root"]; self.totals = st["totals"]
        self.frontier = st["frontier"]; self.tau_h = st["tau_h"]
        self.totals.setdefault("novelty_hits", 0)
        self.totals.setdefault("nov_scored", 0); self.totals.setdefault("sat_hits", 0)
        self.totals.setdefault("precanon_dup", 0); self.totals.setdefault("julia_hooks_skipped", 0)
        self.totals.setdefault("distinct_looks", 0)
        for k in MAN_TOTALS:
            self.totals.setdefault(k, 0)
        if self.maneuvers and "maneuvers" in st:
            m = st["maneuvers"]
            self.man_visited = set(m.get("visited", []))
            self.man_probe_s = float(m.get("probe_s", 0.0))
            self.man_gov.load_state(m.get("governor", {}))
        self.rng.bit_generator.state = st["rng"]
        # scheduler prices/caps reload from the checkpoint; caps re-open on resume (item 4). The
        # distinct-look tally reloaded from its npz in the scheduler's __init__.
        if self.scheduler is not None and "scheduler" in st:
            self.scheduler.load_state(st["scheduler"], reopen_caps=True)
        # cloud is rebuilt from the DURABLE ledger (source of truth) ⊕ the freshness prior, not
        # the checkpoint, so a kill between ledger-append and checkpoint cannot lose/duplicate an
        # admission. build_clouds folds self.prior_rows (set in __init__ before load_state).
        self.clouds = self.build_clouds()
        self.run_clouds = self.build_run_clouds()
        self.rebuild_hooked_c()
        # morph memory (+ frontier-node embeddings) reload from their npz sidecars.
        if self.lambda_m > 0.0:
            self.node_embs = self.load_node_embs()
        print(f"[resume] batch {self.batch_i}, frontier {len(self.frontier)}, "
              f"active {self.active_s/60:.1f}m, admitted {self.totals['admitted']} "
              f"(cloud rebuilt from ledger: "
              f"{sum(len(v) for v in self.clouds.values())} places)", flush=True)

    # ================================================================= dive
    # Single-track descent off a completed run's admissions. Each rung reuses the frontier
    # expand machinery (up to --descent-candidates survivors, existing gates), harvests every
    # survivor at the per-partition tau_h exactly as normal mode, then continues down the
    # cheap-p_good argmax child (small Gumbel tie-break, no breadth). One path per dive.
    # Terminates on: target depth reached, all candidates gate-dead, or the fw floor (the
    # Rust expand emits a min_fw_floor `dead` before crossing --min-fw = dive_min_fw). The
    # run-scoped ledger is the durable admission record; a per-dive checkpoint makes resume
    # skip finished dives without re-descending.
    # ---------------------------------------------------------------------- #
    def _load_source_admissions(self):
        led = self.dive_source / "outcome_ledger.jsonl"
        if not led.exists():
            raise SystemExit(f"--dive-source has no outcome_ledger.jsonl: {led}")
        rows = [json.loads(l) for l in open(led, encoding="utf-8") if l.strip()]
        adm = [r for r in rows if r.get("distinct") and (r.get("decoded_class") or 0) >= 3]
        if not adm:
            raise SystemExit(f"no distinct-q3 admissions in {led}")
        return adm

    @staticmethod
    def _canon_pgood(r):
        v = r.get("canon_pgood")
        return float(v) if v is not None else float(r.get("p_good", 0.0))

    def _build_dive_plan(self):
        """Deterministic plan: top-N admissions by canonical p_good + M random controls
        (disjoint from top, drawn regardless of score). Each entry starts a dive at the
        admission's outcome viewport, continuing downward."""
        adm = self._load_source_admissions()
        ranked = sorted(adm, key=lambda r: (-self._canon_pgood(r), r["id"]))
        n_top = int(self.args.n_top)
        n_ctrl = int(self.args.n_control)
        top = ranked[:n_top]
        top_ids = {r["id"] for r in top}
        pool = [r for r in ranked if r["id"] not in top_ids]
        rng = np.random.default_rng(self.seed)
        k = min(n_ctrl, len(pool))
        ctrl = [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))] if k else []

        def entry(r, group, i):
            c = None
            if r.get("julia_c_re") is not None:
                c = [str(r["julia_c_re"]), str(r["julia_c_im"])]
            return dict(
                dive_id=f"dive_{i:03d}", start_group=group, source_id=r["id"],
                partition=r["family"], c=c,
                cx=float(r["outcome_cx"]), cy=float(r["outcome_cy"]), fw=float(r["outcome_fw"]),
                depth=int(r.get("reached_depth", 2)), source_pgood=self._canon_pgood(r),
            )
        plan = [entry(r, "top", i) for i, r in enumerate(top)]
        plan += [entry(r, "control", n_top + i) for i, r in enumerate(ctrl)]
        # item 8: scheduler ON => order dive SOURCES by partition deficit (stable sort keeps the
        # within-partition p_good rank), so a budget-truncated dive covers deficit families first.
        # Minimal v1 — ordering only, nothing fancier.
        if self.scheduler is not None:
            dfs = self.scheduler.deficits()
            plan.sort(key=lambda e: -dfs.get(e["partition"], 0.0))
        return plan

    def one_dive(self, e) -> dict:
        """Descend a single track from plan entry `e`. Returns the dive record."""
        self.cur_dive = (e["dive_id"], e["start_group"], e["source_id"])
        partition, c = e["partition"], e["c"]
        key = (partition, tuple(c) if c else None)
        rid = self.new_node_id()
        node = dict(node_id=rid, root_id=rid, partition=partition, c=c,
                    cx=e["cx"], cy=e["cy"], fw=e["fw"], depth=e["depth"])
        admissions, rungs = [], 0
        cause = "target_depth"
        n_adm_before = self.totals["admitted"]
        while node["depth"] < self.dive_target_depth:
            self.batch_i += 1
            cands = self.expand_group(key, [node])
            if not cands:
                cause = "gate_dead_or_floor"
                break
            self.score_cheap(cands)
            adm_before = self.totals["admitted"]
            self.harvest(cands)                          # standard admission to the run ledger
            n_new = self.totals["admitted"] - adm_before
            rungs += 1
            # argmax child by cheap p_good (small Gumbel tie-break; no breadth).
            best = max(cands, key=lambda cc: cc["cheap_pgood"] + gumbel(self.rng, DIVE_NOISE_T))
            node = dict(node_id=best["node_id"], root_id=rid, partition=partition, c=c,
                        cx=best["cx"], cy=best["cy"], fw=best["fw"], depth=best["depth"])
            if n_new:                                    # collect this dive's admitted oids
                for r in self.ledger.rows[-n_new:]:
                    admissions.append(dict(id=r["id"], depth=r["reached_depth"],
                                           p_good=r["p_good"], canon_pgood=r.get("canon_pgood"),
                                           cx=r["outcome_cx"], cy=r["outcome_cy"], fw=r["outcome_fw"]))
        rec = dict(dive_id=e["dive_id"], start_group=e["start_group"], source_id=e["source_id"],
                   partition=partition, start_depth=e["depth"], start_pgood=e["source_pgood"],
                   end_depth=node["depth"], rungs=rungs, end_cause=cause,
                   n_admitted=self.totals["admitted"] - n_adm_before, admissions=admissions)
        self.cur_dive = None
        with open(self.dive_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def save_dive_state(self, plan, done_idx):
        self.morph.save()
        state = dict(
            run_ts=self.run_dir.name, mode="dive", seed=self.seed,
            dive_source=str(self.dive_source), target_depth=self.dive_target_depth,
            min_fw=self.dive_min_fw, plan=plan, done_idx=done_idx,
            node_ctr=self.node_ctr, seq=self.seq, batch_i=self.batch_i,
            active_s=self.active_s, est_dive_s=self.est_batch_s,
            totals=self.totals, rng=self.rng.bit_generator.state,
        )
        tmp = self.dive_state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.dive_state_path)
        self.ledger.save_feats()

    def load_dive_state(self):
        st = json.loads(self.dive_state_path.read_text(encoding="utf-8"))
        self.node_ctr = st["node_ctr"]; self.seq = st["seq"]; self.batch_i = st["batch_i"]
        self.active_s = st["active_s"]; self.est_batch_s = st["est_dive_s"]
        self.totals = st["totals"]
        self.totals.setdefault("nov_scored", 0); self.totals.setdefault("sat_hits", 0)
        self.rng.bit_generator.state = st["rng"]
        self.clouds = self.ledger.clouds(self.partitions)
        # the ledger is the durable source of truth; re-sync the admitted counter to it so a
        # resume across a mid-dive boundary can't leave the stat lagging the real admissions.
        self.totals["admitted"] = sum(
            1 for r in self.ledger.rows if r.get("distinct") and (r.get("decoded_class") or 0) >= 3)
        print(f"[dive-resume] {st['done_idx']}/{len(st['plan'])} dives done, "
              f"active {self.active_s/60:.1f}m, admitted {self.totals['admitted']}", flush=True)
        return st["plan"], st["done_idx"]

    def run_dive(self):
        if self.args.resume and self.dive_state_path.exists():
            plan, done_idx = self.load_dive_state()
        else:
            plan = self._build_dive_plan()
            done_idx = 0
            ng = sum(1 for e in plan if e["start_group"] == "control")
            print(f"[dive-fresh] {len(plan)} dives ({len(plan)-ng} top + {ng} control) off "
                  f"{self.dive_source.name}; target_depth={self.dive_target_depth} "
                  f"min_fw={self.dive_min_fw:g}", flush=True)
            print(f"[tau_h] {self.tau_h}", flush=True)
            self.save_dive_state(plan, done_idx)

        while done_idx < len(plan):
            if self.stop_path.exists():
                print("[STOP] sentinel present — halting at dive boundary.", flush=True)
                break
            # don't start a dive that can't finish in the remaining budget (est from history).
            if self.budget_s and self.est_batch_s > 0 and \
                    self.active_s + self.est_batch_s > self.budget_s:
                print(f"[budget] active {self.active_s/60:.1f}m + est dive "
                      f"{self.est_batch_s:.0f}s would exceed {self.budget_s/60:.0f}m — stopping "
                      f"at {done_idx}/{len(plan)}.", flush=True)
                break
            e = plan[done_idx]
            tb = time.time()
            rec = self.one_dive(e)
            dt = time.time() - tb
            self.active_s += dt
            self.est_batch_s = dt if self.est_batch_s == 0 else 0.6 * self.est_batch_s + 0.4 * dt
            done_idx += 1
            self.save_dive_state(plan, done_idx)
            print(f"  {rec['dive_id']} [{rec['start_group']}] start d{rec['start_depth']} "
                  f"-> d{rec['end_depth']} ({rec['rungs']} rungs, {rec['end_cause']}) "
                  f"admitted={rec['n_admitted']} | {dt:.0f}s active={self.active_s/60:.1f}m "
                  f"({done_idx}/{len(plan)})", flush=True)

        self.finish_dive(plan, done_idx)

    def finish_dive(self, plan, done_idx):
        summary = dict(
            run_ts=self.run_dir.name, mode="dive", dive_source=str(self.dive_source),
            target_depth=self.dive_target_depth, min_fw=self.dive_min_fw,
            n_dives_planned=len(plan), n_dives_done=done_idx,
            active_min=round(self.active_s / 60.0, 2), totals=self.totals,
            cloud_sizes={p: len(v) for p, v in self.clouds.items()},
        )
        if self.scheduler is not None:      # same stamp as finish(): a dive under --scheduler
            summary["scheduler"] = self.scheduler.summary()   # must be as readable afterwards
            summary["library_seed"] = summary["scheduler"]["library_seed"]
            if summary["library_seed"].get("status") != "seeded":
                summary["UNSEEDED_RUN"] = (
                    "library seed absent (status=%s): deficits/look_frac in this run measure "
                    "RUN-LOCAL scarcity, NOT library-wide. Do not compare them to a seeded run."
                    % summary["library_seed"].get("status"))
        else:                                        # never_attempted, not absent — see finish()
            summary["library_seed"] = dict(
                status="never_attempted",
                reason="run has --scheduler OFF: nothing to seed",
                source=str(dsched.library_seed_paths()[0]),
                source_exists=dsched.library_seed_paths()[0].exists())
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n=== DIVE SUMMARY ===")
        print(f"  {done_idx}/{len(plan)} dives, active {self.active_s/60:.1f}m")
        print(f"  ADMITTED distinct q3={self.totals['admitted']} q3_dup={self.totals['q3_dup']} "
              f"guarded={self.totals['guarded']} canonical_q3={self.totals['canonical_q3']}")
        print(f"  cloud: {summary['cloud_sizes']}")
        print(f"  ledger -> {self.ledger.path}\n  dive_log -> {self.dive_log}")

    # ---------------------------------------------------------------- run
    def run(self):
        if self.dive:
            return self.run_dive()
        global ROOT_LOW_WATER
        ROOT_LOW_WATER = self.B
        if self.args.resume and self.state_path.exists():
            self.load_state()
        else:
            print(f"[fresh] run {self.run_dir.name}: families={self.families} "
                  f"julia_hook={self.julia_hook} budget={self.budget_s/60:.0f}m B={self.B} "
                  f"lambda_m={self.lambda_m} beta={self.beta} recency_k={self.recency_k}", flush=True)
            print(f"[dup-fix] julia_hook_spacing={self.julia_hook_spacing} "
                  f"freshness_prior={self.freshness_prior} "
                  f"(prior_rows={len(self.prior_rows)}) "
                  f"seeded_cloud_sizes={ {p: len(v) for p, v in self.clouds.items()} }", flush=True)
            if self.lambda_m > 0.0:
                mode = f"recency (admitted + last {self.recency_k} batches)" if self.recency_k \
                    else "legacy (all-permanent)"
                print(f"[morph-anchors] lo={self.morph_lo:.4f} hi={self.morph_hi:.4f} "
                      f"({self.anchor_src}); memory={mode}, sat knee cos>={self.sat_cos:.4f}",
                      flush=True)
            print(f"[tau_h] {self.tau_h}", flush=True)
            if self.maneuvers:
                print(f"[maneuvers] ON — quota={self.man_quota} slots/batch (of AVAILABLE), "
                      f"probe_p={self.man_gov.p} k={self.man_ks} lateral={self.man_lateral}",
                      flush=True)
            if self.scheduler is not None:
                tf = {p: round(v, 3) for p, v in self.scheduler.target_frac.items()}
                print(f"[scheduler] ON — target_frac={tf} "
                      f"explore_floor={self.scheduler.explore_floor} "
                      f"julia_route_gain={self.scheduler.julia_route_gain} "
                      f"(observed_cells={len(self.scheduler.observed)})", flush=True)
                seeded = getattr(self, "_sched_seeded", {}) or {}
                lf = {p: round(v, 3) for p, v in self.scheduler.look_frac().items()}
                df = {p: round(v, 3) for p, v in self.scheduler.deficits().items()}
                rec = self.scheduler.seed_record or {}
                print(f"[scheduler] library seed: status={rec.get('status')} "
                      f"source={rec.get('source')} seeded_looks={rec.get('seeded_looks')} "
                      f"tallies={self.scheduler.tally.counts()} "
                      f"(total {self.scheduler.tally.total()}, newly seeded {sum(seeded.values())})", flush=True)
                print(f"[scheduler] launch look_frac={lf}\n[scheduler] launch deficits={df}", flush=True)
            self.draw_roots()
            self.seed_julia_pool()          # PRIMARY julia supply: sampler-sourced roots (probe)
            self.save_state()

        while True:
            if self.stop_path.exists():
                print("[STOP] sentinel present — halting at batch boundary.", flush=True)
                break
            if self.budget_s and self.active_s + self.est_batch_s > self.budget_s:
                print(f"[budget] active {self.active_s/60:.1f}m + est batch "
                      f"{self.est_batch_s:.0f}s would exceed {self.budget_s/60:.0f}m — stopping.", flush=True)
                break
            if len(self.frontier) < ROOT_LOW_WATER:
                self.draw_roots()
            if not self.frontier:
                print("[frontier] empty and no fresh roots — stopping.", flush=True)
                break

            tb = time.time()
            self.batch_i += 1
            batch = self.pop_batch_scheduled() if self.scheduler is not None else self.pop_batch()
            if not batch:
                # scheduler: an empty pop with a non-empty frontier means every servable
                # partition is PRICE-capped -> reopen (redistribute demand) and retry.
                if self.scheduler is not None and self.frontier:
                    self.scheduler.prices.reopen_caps()
                    self.batch_i -= 1
                    continue
                # everything capped; try fresh roots, else stop.
                if self.draw_roots() == 0:
                    print("[frontier] all roots capped (M) and no fresh seeds — stopping.", flush=True)
                    break
                self.batch_i -= 1
                continue
            self.fold_expanded_into_memory(batch)   # parents join morph memory before scoring
            self.totals["expanded"] += len(batch)
            self.totals["man_nodes_expanded"] += sum(1 for n in batch if n.get("man"))
            # Maneuvers are proposed off the rungs ABOUT TO BE EXPANDED and are INTERLEAVED
            # into this same walk — never a separate run, which would confound the move with
            # the run. They land on the frontier for a later batch, competing (with a
            # reserved floor) against ordinary nodes.
            self.propose_maneuvers(batch)
            rec0 = self._reconcile_snapshot()
            cands = self.expand_batch(batch)
            self.totals["candidates"] += len(cands)
            self.score_cheap(cands)
            self.score_morph(cands)                  # embed + cos_max vs memory (parents incl.)
            self.harvest(cands)                      # admissions fold into memory
            self.push_children(cands)                # novelty penalty applied from cos_max
            self.morph.end_batch()                   # finalize recency block, evict > K (no-op legacy)
            self._reconcile_batch(rec0, len(cands))  # found == written + dropped_*, or exit loud

            dt = time.time() - tb
            self.active_s += dt
            self.est_batch_s = dt if self.est_batch_s == 0 else 0.5 * self.est_batch_s + 0.5 * dt
            # scheduler: charge this batch's active time to the served partition (price EMA +
            # attempt-cap accounting). Cross-partition arithmetic only; no p_good.
            if self.scheduler is not None and self._served_partition is not None:
                self.scheduler.charge(self._served_partition, dt / 60.0)
            self.save_state()
            if self.batch_i % 1 == 0:
                sat = ""
                if self.lambda_m > 0.0 and cands:
                    bs = sum(1 for c in cands if float(c.get("cos_max", 0.0)) >= self.sat_cos)
                    sat = (f" sat={bs}/{len(cands)}={bs/len(cands):.2f} "
                           f"mem={self.morph.n_perm}+{self.morph.n_recency}")
                print(f"  batch {self.batch_i}: exp={len(batch)} cand={len(cands)} "
                      f"admitted(cum)={self.totals['admitted']} julia_roots={self.totals['julia_roots']} "
                      f"frontier={len(self.frontier)}{sat} | {dt:.0f}s active={self.active_s/60:.1f}m", flush=True)

        self.finish()

    def finish(self):
        self.save_state()
        summary = dict(
            run_ts=self.run_dir.name, mode="steered", families=self.families,
            julia_hook=self.julia_hook, julia_hook_spacing=self.julia_hook_spacing,
            freshness_prior=self.freshness_prior, prior_rows=len(self.prior_rows),
            budget_min=self.budget_s / 60.0,
            lambda_m=self.lambda_m, beta=self.beta, recency_k=self.recency_k,
            morph_mem=len(self.morph), morph_perm=self.morph.n_perm,
            morph_recency=self.morph.n_recency,
            morph_lo=self.morph_lo, morph_hi=self.morph_hi, anchor_src=self.anchor_src,
            sat_cos=round(self.sat_cos, 4),
            sat_frac=(round(self.totals["sat_hits"] / self.totals["nov_scored"], 4)
                      if self.totals.get("nov_scored") else None),
            active_min=round(self.active_s / 60.0, 2), batches=self.batch_i,
            tau_h=self.tau_h, totals=self.totals,
            cloud_sizes={p: len(v) for p, v in self.clouds.items()},
        )
        summary["maneuvers"] = self.maneuver_summary()
        if self.scheduler is not None:
            summary["scheduler"] = self.scheduler.summary()
            # Stamp the seed provenance at the TOP level too, not only nested under
            # "scheduler" — campaign-2's summary is indistinguishable from a seeded run's
            # precisely because this fact was nowhere in it. An unseeded run additionally
            # gets a screaming key so no reader can quote its deficits as library-wide.
            summary["library_seed"] = summary["scheduler"]["library_seed"]
            if summary["library_seed"].get("status") != "seeded":
                summary["UNSEEDED_RUN"] = (
                    "library seed absent (status=%s): deficits/look_frac in this run measure "
                    "RUN-LOCAL scarcity, NOT library-wide. Do not compare them to a seeded run."
                    % summary["library_seed"].get("status"))
        else:
            # A SEEDER THAT WAS NEVER CALLED RECORDS never_attempted, NOT ABSENCE. The
            # library seed is a scheduler-only concept, so a scheduler-off run has no seed
            # by construction — but "the key is missing" and "the seed was missing" read
            # identically in a summary six months later, which is exactly how campaign-2's
            # seeded/unseeded status became unrecoverable. Say which it is, in the file.
            summary["library_seed"] = dict(
                status="never_attempted", reason="run has --scheduler OFF: no deficit "
                "scheduler, therefore no distinct-look tally and nothing to seed",
                source=str(dsched.library_seed_paths()[0]),
                emb_dir=str(dsched.library_seed_paths()[1]),
                source_exists=dsched.library_seed_paths()[0].exists())
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n=== STEERED FRONTIER SUMMARY ===")
        print(f"  active {self.active_s/60:.1f}m over {self.batch_i} batches")
        print(f"  expanded={self.totals['expanded']} candidates={self.totals['candidates']} "
              f"harvest_checks={self.totals['harvest_checks']} canonical_q3={self.totals['canonical_q3']}")
        print(f"  ADMITTED distinct q3={self.totals['admitted']}  q3_dup={self.totals['q3_dup']} "
              f"guarded={self.totals['guarded']} julia_roots={self.totals['julia_roots']} "
              f"cap_hits={self.totals['cap_hits']}")
        print(f"  precanon_dup(renders saved)={self.totals['precanon_dup']} "
              f"julia_hooks_skipped(spacing)={self.totals['julia_hooks_skipped']} "
              f"freshness_prior={self.freshness_prior} (prior_rows={len(self.prior_rows)})")
        sf = (f"{self.totals['sat_hits']}/{self.totals['nov_scored']}="
              f"{self.totals['sat_hits']/self.totals['nov_scored']:.3f}"
              if self.totals.get("nov_scored") else "n/a")
        print(f"  lambda_m={self.lambda_m} beta={self.beta} recency_k={self.recency_k} "
              f"novelty_hits={self.totals['novelty_hits']} sat_frac={sf} "
              f"morph_mem={len(self.morph)} (perm {self.morph.n_perm} + recency {self.morph.n_recency})")
        print(f"  cloud: {summary['cloud_sizes']}")
        if self.scheduler is not None:
            s = summary["scheduler"]
            print(f"  SCHEDULER: distinct_looks={self.totals.get('distinct_looks',0)} "
                  f"total_looks={s['total_looks']} looks={s['looks']}")
            print(f"    target_frac={s['target_frac']}\n    look_frac={s['look_frac']}")
            print(f"    prices={s['prices']} capped={s['capped']}")
            print(f"    trace -> {self.scheduler.trace_path}")
        if self.maneuvers:
            m = summary["maneuvers"]
            print(f"  MANEUVERS: probes {m['probes_fired']}/{m['probes_rolled']} fired "
                  f"(coin-skip {m['probes_coin_skip']}, region-cache-skip "
                  f"{m['probes_cache_skip']}); operator available "
                  f"{m['op_available']}/{m['op_calls']} = "
                  f"{(m['op_available']/m['op_calls']) if m['op_calls'] else float('nan'):.3f}")
            print(f"    pushed={m['nodes_pushed']} (available-but-unused {m['avail_unused']}) "
                  f"expanded={m['nodes_expanded']} admitted={m['admitted']}")
            print(f"    quota={m['quota']} bound={m['quota_bound']} "
                  f"unfilled={m['quota_unfilled']}  probe+solve "
                  f"{m['probe_s']:.1f}s = {m['probe_share_of_active']:.3%} of active")
        print(f"  ledger -> {self.ledger.path}\n  summary -> {self.run_dir/'summary.json'}")

    def maneuver_summary(self) -> dict:
        """The §7 read: did each operator fire, what did availability actually run at, did
        the floor bind, and what share of the run was probe+solve."""
        t = self.totals
        calls = t["man_op_available"] + t["man_op_unavailable"]
        return dict(
            enabled=self.maneuvers, quota=self.man_quota,
            probe_p=self.man_gov.p, ks=[("none" if k is None else k) for k in self.man_ks],
            lateral=self.man_lateral,
            probes_rolled=t["man_probes_rolled"], probes_fired=t["man_probes_fired"],
            probes_coin_skip=t["man_probes_coin_skip"],
            probes_cache_skip=t["man_probes_cache_skip"],
            op_calls=calls, op_available=t["man_op_available"],
            op_unavailable=t["man_op_unavailable"],
            avail_unused=t["man_avail_unused"], nodes_pushed=t["man_nodes_pushed"],
            nodes_expanded=t["man_nodes_expanded"], admitted=t["man_admitted"],
            quota_bound=t["man_quota_bound"], quota_unfilled=t["man_quota_unfilled"],
            visited_atoms=len(self.man_visited), probe_s=round(self.man_probe_s, 2),
            probe_share_of_active=(self.man_probe_s / self.active_s
                                   if self.active_s > 0 else 0.0),
            log=str(self.man_log),
        )


def set_below_normal_priority() -> str:
    """Drop THIS process to BELOW_NORMAL, which every child inherits.

    A long unattended discovery run must yield to interactive work. This driver fans out
    `fractal-generator.exe` through several call sites (--expand, the confirmation renders,
    reframe's own renders), and on win32 a child inherits the parent's priority class — so
    lowering the driver once covers all of them, without threading `creationflags` through
    every launcher. The thread-count half of the pairing stays with the engine defaults
    (`corpus_common.DEFAULT_ENGINE_THREADS`); this is only the priority half."""
    if sys.platform != "win32":
        try:
            os.nice(10)
            return "nice+10"
        except Exception as e:
            return f"unavailable ({e})"
    try:
        import ctypes
        from corpus_common import BELOW_NORMAL_PRIORITY_CLASS
        # A PRIVATE kernel32 handle, with `use_last_error=True`. Two reasons, both about
        # the failure path rather than the success path:
        #   * `ctypes.windll.kernel32` is a process-global CACHED library object, so the
        #     restype/argtypes below would mutate it for every other user in the process.
        #   * that cached object is created WITHOUT use_last_error, so
        #     `ctypes.get_last_error()` always reads 0 — the old error branch could only
        #     ever print "FAILED (err 0)", which is a silent failure wearing a report.
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # restypes/argtypes are load-bearing: GetCurrentProcess returns the pseudo-handle
        # (HANDLE)-1, and without c_void_p ctypes truncates it to a 32-bit int that
        # SetPriorityClass rejects — the call then "fails" for a reason nothing reports.
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.SetPriorityClass.restype = ctypes.c_int
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        return "BELOW_NORMAL" if ok else f"FAILED (err {ctypes.get_last_error()})"
    except Exception as e:                                   # never fatal
        return f"unavailable ({e})"


def preflight_library_seed(args):
    """Scheduler-only fail-closed preflight: refuse to start unseeded, BEFORE a run dir,
    ledger, state file or render exists. Called from main() ahead of `SteeredFrontier(args)`,
    which is what makes "aborts before doing any work" true rather than merely early — the
    driver's __init__ mkdirs the run dir and opens the ledger.

    Skipped on a resume whose distinct-look tally is already on disk: that tally IS the seed,
    and re-checking a since-moved artifact would abort legitimate resumes of seeded runs.

    The loaded record rides on `args._library_seed` so the driver reuses the one load.
    Raises SystemExit (message names both paths) when the seed is absent and not overridden."""
    if not getattr(args, "scheduler", False):
        return None
    if getattr(args, "resume", False) and (Path(args.run_dir) / "distinct_looks.npz").exists():
        return None
    try:
        args._library_seed = dsched.require_library_seed(
            allow_unseeded=bool(getattr(args, "allow_unseeded", False)))
    except dsched.UnseededRunError as e:
        raise SystemExit(f"[scheduler] REFUSING TO START UNSEEDED\n{e}")
    if args._library_seed["status"] != "seeded":
        print("[scheduler] WARNING --allow-unseeded: no library seed; deficits are RUN-LOCAL. "
              "summary.json will be stamped library_seed.status=unseeded.", flush=True)
    return args._library_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="fresh run-scoped dir (ledger + state.json)")
    ap.add_argument("--families", default="mandelbrot,multibrot3,multibrot4,multibrot5")
    ap.add_argument("--julia-hook", action="store_true")
    ap.add_argument("--julia-hook-spacing", type=float, default=JULIA_HOOK_SPACING,
                    help="c-plane spacing radius for the julia hook: skip a parent whose seed c is "
                         f"within this of an already-hooked one (default {JULIA_HOOK_SPACING})")
    ap.add_argument("--julia-seed-pool", type=str, default=None,
                    help="PRIMARY julia supply under test: a JSON list of {c_re,c_im} c's from the "
                         "c-diverse near-∂M sampler, injected as julia:mandelbrot roots at fresh "
                         "start (bypasses the hook-spacing gate; requires 'mandelbrot' in --families). "
                         "DEFAULT None => julia roots only via the parent-fired hook.")
    ap.add_argument("--freshness-prior", action="store_true",
                    help="ENABLE the cross-run coordinate freshness prior: seed this run's DEDUP "
                         "clouds (pre-canonical + admission near-dup + steering) from prior-library "
                         "admitted coords. DEFAULT OFF — prior-ON sterilized the native-seed "
                         "rejection sampler.")
    ap.add_argument("--budget", type=float, default=45.0, help="active-time budget (minutes)")
    ap.add_argument("--batch", type=int, default=0, help="nodes per batch (0 = default 32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-m", type=float, default=LAMBDA_M_DEFAULT,
                    help="morph-novelty penalty magnitude (0 disables; == pilot)")
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT,
                    help="depth bonus per rung (0 disables; == pilot)")
    ap.add_argument("--morph-lo", type=float, default=None,
                    help="override the zero-penalty cos knee (default: calibrated anchors file)")
    ap.add_argument("--morph-hi", type=float, default=None,
                    help="override the full-penalty cos knee (default: calibrated anchors file)")
    ap.add_argument("--mem-recency", action="store_true",
                    help="v1.2 morph-memory fix: novelty measured vs ADMITTED looks + a rolling "
                         "window of the last --recency-k batches' expanded looks (default off => "
                         "legacy all-permanent, reproduces v1.1)")
    ap.add_argument("--recency-k", type=int, default=8,
                    help="recency window size in batches for --mem-recency (default 8)")
    # --- minibrot maneuvers (default OFF; off is byte-identical to pre-change) ---
    ap.add_argument("--maneuvers", action="store_true",
                    help="ENABLE the minibrot reframing operators (snap_to_nucleus / "
                         "lateral_to_sibling) as candidate moves interleaved in this walk. "
                         "DEFAULT OFF (byte-identical).")
    ap.add_argument("--maneuver-quota", type=int, default=MAN_QUOTA_DEFAULT,
                    help=f"reserved frontier SLOTS per batch for maneuver-originated nodes, "
                         f"regardless of score — a floor OF AVAILABLE, never a probability "
                         f"(default {MAN_QUOTA_DEFAULT})")
    ap.add_argument("--maneuver-probe-p", type=float, default=MAN_PROBE_P_DEFAULT,
                    help=f"COST GOVERNOR: probability the atom probe fires per popped rung "
                         f"(default {MAN_PROBE_P_DEFAULT}). Not a selection probability — "
                         f"selection is the reserved quota.")
    ap.add_argument("--maneuver-k", type=str, default=MAN_K_DEFAULT,
                    help=f"framing set for snap_to_nucleus: comma list where 'none' preserves "
                         f"the parent fw and a number k frames at k x atom size "
                         f"(default {MAN_K_DEFAULT!r})")
    ap.add_argument("--no-maneuver-lateral", dest="maneuver_lateral", action="store_false",
                    help="disable lateral_to_sibling (the expensive operator); snap only")
    ap.set_defaults(maneuver_lateral=True)
    # --- deficit scheduler (default OFF; scheduler-off is byte-identical to pre-change) ---
    ap.add_argument("--scheduler", action="store_true",
                    help="ENABLE the family-level deficit scheduler: cross-partition allocation "
                         "by price-weighted DISTINCT-LOOK deficit vs the target measure, instead "
                         "of a single global p_good queue. DEFAULT OFF (byte-identical).")
    ap.add_argument("--scheduler-target", type=str, default=None,
                    help="target measure to project into the per-partition order book "
                         "(default data/emission/target_measure.json)")
    ap.add_argument("--scheduler-prices", type=str, default=None,
                    help="seed price / cap / routing config (default data/atlas/scheduler_prices.json)")
    ap.add_argument("--allow-unseeded", action="store_true",
                    help="proceed with --scheduler even though the library look seed is absent "
                         "or empty. Deficits then measure RUN-LOCAL scarcity, not library-wide; "
                         "the run summary is permanently stamped library_seed.status=unseeded.")
    # --- dive mode ---
    ap.add_argument("--dive", action="store_true",
                    help="single-track descent off a completed run's admissions (uses dive_state.json)")
    ap.add_argument("--dive-source", type=str, default=None,
                    help="completed run dir whose admissions seed the dives (required with --dive)")
    ap.add_argument("--dive-target-depth", type=int, default=23,
                    help="stop a dive at this reached depth (default 23)")
    ap.add_argument("--dive-min-fw", type=float, default=2e-9,
                    help="dive fw floor: stop before a zoom would cross it (default 2e-9)")
    ap.add_argument("--n-top", type=int, default=20,
                    help="dives from the top source admissions by canonical p_good (default 20)")
    ap.add_argument("--n-control", type=int, default=8,
                    help="control dives from random source admissions regardless of score (default 8)")
    ap.add_argument("--below-normal", action="store_true",
                    help="run this process (and every engine child it spawns) at "
                         "BELOW_NORMAL priority so a long run yields to interactive work")
    ap.add_argument("--resume", action="store_true", help="continue from state.json / dive_state.json")
    args = ap.parse_args()
    if args.below_normal:
        print(f"[priority] {set_below_normal_priority()}", flush=True)
    preflight_library_seed(args)
    SteeredFrontier(args).run()


if __name__ == "__main__":
    main()
