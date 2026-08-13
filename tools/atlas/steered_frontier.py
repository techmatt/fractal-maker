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
the harvest (reframe + the raw P(>=3) good floor), the near-dup cloud, the
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
import shutil
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
import apportion                         # noqa: E402  (THE two apportionment rules; stdlib-only)
import run_record                        # noqa: E402  (THE segmented run-record layer; stdlib-only)
from tools import stage_times as stimes  # noqa: E402  (THE per-unit stage timing stream; stdlib-only)
import discovery_sinks as dsinks         # noqa: E402  (THE sink paths + the feats bulk() class)

# production_seeder wires its own sub-imports (prescreen / reframe / guard / score_lib /
# active_ckpt) and owns the constants, root pipeline, near-dup machinery and guard. Reuse it
# wholesale.
import production_seeder as ps          # noqa: E402
import prescreen                        # noqa: E402
import reframe                          # noqa: E402
import guard                            # noqa: E402
import location as loc_mod              # noqa: E402
import julia_ledger_schema as jls       # noqa: E402  (campaign/walk julia schema tag)
from active_ckpt import ACTIVE_CKPT, auto_maxiter  # noqa: E402
from tools.emission import floors as F  # noqa: E402  THE cut owner (GOOD_FLOOR)
import tau_h_rederive as _thr           # noqa: E402  THE tau_h deriver (MIN_N, the fail-open n)
import partitions as P                   # noqa: E402  (THE partition registry + phoenix split)
import deficit_scheduler as dsched       # noqa: E402  (pure; torch-free scheduling logic)
import pop_quota as pquota               # noqa: E402  (harvest v2 allocator; pure, torch-free)
import regularize_quota_prices as rqp    # noqa: E402  (THE seed-table paths; pure, no torch)
import supply_routing as srt             # noqa: E402  (harvest v2 channel table; pure data)
import minibrot_maneuvers as mnv         # noqa: E402  (pure mpmath; no subprocess, no torch)
import maneuver_screen as msc            # noqa: E402  (the field half: spawns the engine)
import maneuver_view_screen as mvs       # noqa: E402  (the same, on the VIEW's own frame)
import view_field_cache as vfc           # noqa: E402  (the run-local f32 field store)
import view_screen as vscr               # noqa: E402  (composite_v3 + the reference params)
import view_fit as vfit                  # noqa: E402  (the staged view_fit v1.1 fitted score)
# The label-seeded harvest's two primitives (`snap_at_seed`, `enumerate_seed`) are what
# maneuvers-on-admissions fires, and its `INTERIOR_DISCARD` is the sourcing copy of Matt's
# rule. Imported rather than restated: a third literal 0.30 in this tree is how the three
# drift (`verification_practice.md` §1.8).
import label_seeded_harvest as lsh       # noqa: E402  (pure mpmath + the view screen)
import visited_density as vd             # noqa: E402  (cross-run saturation memory; pure)

_INTERIOR_DISCARD = lsh.INTERIOR_DISCARD

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BIN = ps.prescreen.BIN

# The julia:mandelbrot c-supply pool, and the floor it must clear.
#
# THE FLOOR WAS NEVER ENFORCED AT RUN TIME. `seed_julia_pool` deliberately bypasses the
# hook-spacing gate (which was then 0.20, far coarser than the pool), and nothing replaced
# it — so the c-spacing a run actually applied was whatever the passed FILE happened to
# carry, invisibly. Measured on the three pools in the tree:
#
#     julia_seed_pool.json     534 c   min |dc| = 6.09e-3   (q4_decisive MIN_SEP, 5.3x under)
#     julia_supply_pool_v2     539 c   min |dc| = 1.00e-2   (the SUPERSEDED floor)
#     julia_supply_pool_v3     209 c   min |dc| = 3.20e-2   (== CSPACING_FLOOR)
#
# The last live run (data/discovery/harvest_v2_proving_20260803) passed **v2**, i.e. it ran
# at the 1e-2 floor that `supply_routing.CSPACING_BASIS.supersedes` records as an artifact of
# pairs rendered at their own viewports. v3 was built and committed and nothing loaded it.
# So: v3 is the default, and the file is VERIFIED against the floor at load rather than
# trusted — a pool is a config input, and this one silently decides how much of the c-plane
# a whole campaign covers.
JULIA_SUPPLY_POOL = ROOT / "data" / "atlas" / "julia_supply_pool_v3.json"

# --- steering knobs ---
# THE HOOK SPACING IS THE POOL FLOOR. One c-spacing floor, one constant, referenced not
# restated — and the reconciliation is a fix for a channel that was closed by construction.
#
# The two gates ask the identical question ("may this julia parameter be accepted, given the
# ones already accepted?") over the identical set: `seed_julia_pool` registers every injected
# pool c into `hooked_c`, which is exactly what `add_julia_root` measures a parent's c
# against. So the pool and the hook were two floors on ONE population, and they disagreed by
# 6.25x. With 209 pool c thinned at 3.2e-2 spanning the near-dM shell, a 0.20 hook radius
# covers the shell outright: arm B of allocator_prereg_v1 hooked julia:mandelbrot 0 times in
# 28 attempts, twin queues were empty in >80% of batches, and the twin demand silently folded
# to the c-plane parent. `julia_supply_pool_v3.json` cannot serve a gate coarser than the
# floor it was thinned at.
#
# Which constant moves is settled by which one was measured. 3.2e-2 is the adopted floor from
# the fixed-viewport re-measurement (`supply_routing.CSPACING_BASIS`, 4,263 embeddings, Matt's
# decision 2026-08-03) — 7.4% near-dup among the closest pairs it admits. 0.20 came off the
# audit's chain-neighbour collision scale on a different population and has no near-dup rate
# attached to it. Adopting the floor cannot reintroduce what the floor was chosen to prevent:
# a hooked c now clears exactly the separation every pool c already clears.
#
# The INVARIANT, not the value, is what `test_supply_routing.py` pins: the hook spacing may
# never EXCEED the pool floor, or the pool saturates the hook again the moment either moves.
JULIA_HOOK_SPACING = srt.CSPACING_FLOOR
                            # hard 1-neighbour spacing (in the c/parameter plane) for the julia
                            # hook — don't hook a parent whose seed c is within this radius of
                            # an already-hooked c THIS run (injected pool c included). Replaces
                            # the old Q3_DENSITY_CAP density gate on hooked_c. Config knob
                            # (--julia-hook-spacing). See docs/design/morphology_dedup.md §5
                            # and julia_c_sourcing.md § the c-spacing floor.
B_DEFAULT = 32            # nodes popped + expanded per batch
T_GUMBEL = 0.08          # priority exploration temperature (Gumbel scale)
M_CAP = 40               # hard cap on expansions per root_id
DIVE_NOISE_T = 0.02      # small Gumbel tie-break on the dive argmax-child selection
DUP_P0 = 1.0             # dup-penalty magnitude at zero distance to the q3 cloud (E[ord] units)
DUP_SCALE = ps.REJECT_RADIUS   # Gaussian decay scale of the dup penalty (plane coords)
NEUTRAL_PRIOR = 1.0      # root prior priority (mid E[ord] in [0,2])

# --- v1.7 CROSS-RUN SATURATION MEMORY (2026-08-09). The breadth leg's steering weight is
# discounted by how much of the committed ledger already shadows a candidate's place:
#     eord *= 1 / (1 + SAT_STRENGTH * density)
# Mechanism + why the memory is the ledgers rather than a new store: `visited_density.py`.
# SCOPE, and it is the whole scope: `push_children` and nothing else. Root draws
# (NEUTRAL_PRIOR), maneuver-originated nodes (their own screen prior + the reserved quota) and
# the dive path are all EXEMPT by construction — none of them routes through `priority_terms`.
# The exemption is on the PROPOSAL, not on its whole subtree: a maneuver node's ordinary
# descendants are found by ordinary descent, carry a `cheap_eord`, and compete in the ordinary
# queue, so they are discounted like any other breadth candidate. Exempting a lineage forever
# would need a flag threaded down it, and would make a saturated basin permanently cheap to
# re-enter through one operator.
# Cross-partition allocation is untouched: the discount moves an ORDER inside one partition's
# queue, and `pop_batch_quota`/`pop_batch_scheduled` pick the partition before they ever look
# at a priority.
SAT_STRENGTH = 1.0       # discount magnitude; 0 disables the mechanism ENTIRELY (no ledger
                         # load, no index, priorities byte-identical to v1.6 — and the RNG
                         # draw order is unchanged either way, nothing here draws).
SAT_RADIUS_K = 0.30      # a visit at framewidth fw shadows a disc of radius k*fw AROUND
                         # ITSELF (the visit's fw, never the candidate's — see the module
                         # docstring). ADOPTED 2026-08-09 from
                         #   uv run python tools/atlas/sat_radius_calibrate.py
                         # over 46,798 committed q4_candidates rows (8 runs) leave-one-run-out
                         # against the 15,156-row ledger union: 31.1% of candidates carry a
                         # discount and 1.4% land within 10% of full (7.8% in phoenix, the
                         # worst partition), against the 92.9% at which run 2's morph-novelty
                         # term degenerated into a constant offset. The grid crosses the
                         # per-partition bar between 0.32 and 0.35; 0.30 is the round value
                         # inside it. NOT the dedup radius (ps.DEDUP_K, also 0.25-ish) — that
                         # one scales on min(a_fw, b_fw) and answers a boolean.

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
# Framing set. k=16 is often close to a usable wallpaper frame by ITSELF, which is the
# material worth labeling — and every extra k is free of NEWTON cost, because
# snap_to_nucleus_multi solves the nucleus once and reframes per k (a k is not a probe).
# No small k: framing INTO the atom is interior black (docs/design/minibrot_maneuvers.md §7).
#
# WHY k=4 LEFT THE PUSH SET AND k=8 REPLACED IT (2026-08-01). 4x is the frame every orbital
# score has ever been MEASURED on and it stays exactly that (`maneuver_screen.py`) — but as a
# pushed picture it is the frame the view screen's size band exists to demote: the v3 gate's
# calibration set is a series of k4 tiles Matt read as "minibrot too big" / dominated, at
# interior 0.17-0.25 (`view_screen.py`, SIZE_BAND_EDGE). Pushing a framing the sort key is
# built to rank down spends probe budget on material the same run then declines. k=8 sits
# between it and 16 and is Newton-free for the same reason 16 is: one solve, three reframings.
# A measuring frame and a pushed frame are different things, and this is the line between them.
MAN_K_DEFAULT = "none,8,16"
# --- v1.4: the richness screen and what may select on it -------------------------------
# Every available candidate is SCREENED (both ring measures at 64x36 on the atom's 4x
# frame, `maneuver_screen.py`) and the scores are RECORDED on the maneuver row and on the
# pushed node. Recording is unconditional; SELECTING on it is not. `--maneuver-range-prior`
# gates the two selection changes, and with it off the walk's trajectory is byte-identical
# to v1.3 — the screen consumes no RNG, gates nothing, and only adds columns.
MAN_RANGE_GAIN_DEFAULT = 0.5   # bounded prior term: +/- gain/2 around NEUTRAL_PRIOR.
# The bound is the whole design. An ordinary node's `cheap_eord` runs over [0, K-1] = [0, 3]
# (v8 is a K=4 CORN head), so the best-ranked maneuver sits at 1.25 and still loses to any
# ordinary node scoring above that: a maneuver out-competes a SCORED node only via the
# quota floor, never via the prior (docs/design/minibrot_maneuvers.md §3). Symmetric about
# NEUTRAL_PRIOR, so the flag REORDERS maneuvers without inflating them as a class.
MAN_NBH_M_DEFAULT = mnv.NBH_MAX_FOUND      # enumerate up to m nearby nuclei
MAN_NBH_N_DEFAULT = mnv.NBH_TOP_N          # propose the top n by radial_range
MAN_NBH_PROBES_DEFAULT = mnv.NBH_MAX_PROBES
# Counters, named once so __init__/load_state/the summary can never drift apart.
#   probes_*        — cost-governor accounting (did the probe even get to run)
#   op_avail/unavail— the operator's own availability (the ~17% expectation)
#   avail_unused    — AVAILABLE BUT NOT PUSHED: the atom was already visited this run.
#                     Recorded because "the operator had nothing" and "the operator had
#                     something we already had" are different constraints at scale.
#   quota_bound     — reserved slots that promoted a node the plain priority top-B would
#                     NOT have taken (the floor actually binding, not merely present)
#   quota_unfilled  — reserved slots that went unused for lack of AVAILABILITY
#   screened/unscreenable — the richness screen's own reach (the deep tail is genuinely
#                     below the f64 spacing guard at 64 px; that is data, not an error)
#   nbh_passed_over — neighbourhood candidates enumerated, screened and NOT in the top n
#   quota_passed_over — available maneuver nodes the quota could not take THIS batch
#   view_*          — the v1.5 VIEW screen's own reach, counted apart from the atom
#                     screen's `man_screened`/`man_unscreenable` because they are
#                     measurements on different frames and must never be summed.
MAN_TOTALS = ("man_probes_rolled", "man_probes_fired", "man_probes_coin_skip",
              "man_probes_cache_skip", "man_op_available", "man_op_unavailable",
              "man_avail_unused", "man_nodes_pushed", "man_quota_bound",
              "man_quota_unfilled", "man_nodes_expanded", "man_admitted",
              "man_screened", "man_unscreenable", "man_screen_cache_hits",
              "man_nbh_passed_over", "man_quota_passed_over", "man_frontier_pruned",
              "man_view_screened", "man_view_unscreenable", "man_view_vetoed",
              "man_view_fields_cached")

# --- v1.5: the VIEW-level screen selects (2026-08-01) -----------------------------------
# The successor to `--maneuver-range-prior`. That flag selects on `radial_range` measured on
# the ATOM's 4x frame; this one selects on `view_screen.composite_v3` measured on the frame
# the candidate is actually PUSHED at. The retrospective study behind the change is
# `orbital_field_metrics.md` §11: the atom score cannot see the two failures that dominated
# the dry run's own top quintile (a nucleus-centred black blob, and one deep pocket setting a
# large radial span across an otherwise flat frame), because both are failures of spatial
# PARTICIPATION and the ring measures describe dynamic RANGE.
#
# The two flags are mutually exclusive by construction and the view screen REPLACES the atom
# screen when it is on, rather than running beside it. Cost is the reason and it was priced
# first: the view screen is ~3x the fields (one per k, against one shared across k), ~7.6% of
# active on the exploration run's population; adding the atom screen's 2.6% on top crosses the
# ~10% the screen is budgeted at, to produce a second number nothing would select on.
MAN_VIEW_GAIN_DEFAULT = MAN_RANGE_GAIN_DEFAULT   # same bounded +/- gain/2 term, same reason
# --- v1.6: RECORD-AND-RANK, and the interior gate at sourcing (2026-08-03) --------------
# The tail this replaces gates and ADMITS: a candidate below `tau_h` left no trace at all,
# and a `canon-not-q3` or pre-canonically-duplicated one left a `harvest_log` line and was
# dropped. That is correct for building a ledger and wrong for a run whose deliverable is a
# RANKED LIST a human picks a cutoff from — the material just under a cut is exactly the
# material the cut is being chosen against, and it cannot be reviewed if it was never
# written down.
#
# So: admission is UNCHANGED (the ledger, the clouds, the julia hook and every reconcile
# identity below behave exactly as before), and a second, wider store is added beside it.
# Every candidate scoring at or above a low per-partition floor is appended to
# `q4_candidates.jsonl` with its cheap scores, its canonical scores WHEN IT GOT A CANONICAL
# RENDER, and its per-stage fate. Nothing above the floor is discarded unrecorded.
#
# THE FLOOR IS A RECORDING FLOOR, NOT A RENDERING FLOOR, and the distinction is the whole
# cost story: a record is a JSONL line, a canonical confirmation is a 640x360 ss2 render.
# `tau_h` still decides who gets rendered. Rows below it are recorded on their cheap score
# alone and carry `rank_tier=1`; rows with a canonical decode carry `rank_tier=2`. The two
# tiers are NOT commensurable and are stamped so a later ranking cannot silently pool them.
#
# WHY THIS FLOOR. Sized read-only against `data/atlas/canon_waste_v10.json` (1,092 v10
# cheap/canonical pairs): at `tau_h` the cut already retains essentially every canonically-q3
# row it could (multibrot3 53/54, multibrot4 42/42, multibrot5 43/43), so lowering it buys
# almost no q3 RECALL. What it buys is the sub-cut population being on record at all. Half of
# `tau_h` roughly doubles the recorded set at zero render cost; the absolute floor stops a
# partition whose `tau_h` is already tiny (mandelbrot's is 0.023) from recording its entire
# candidate stream. `min(tau_h, ...)` because the floor must never RAISE a cut.
# `[measured: 2026-08-03, scratchpad floor_calib.py over canon_waste_v10.json]`
Q4_REC_FRAC = 0.5          # ... of tau_h
Q4_REC_FLOOR_ABS = 0.05    # ... but never record below this absolute cheap p_good


def _parse_family_weights(spec, families):
    """`"mandelbrot=0.176,multibrot3=0.281,..."` -> normalized dict, or `{}` when absent.

    FAIL LOUD ON AN UNKNOWN OR MISSING FAMILY rather than defaulting it to zero. A typo'd
    family name that silently weighted nothing would mute a whole channel for a 4-hour run
    and show up only as an empty partition in the readout — which is indistinguishable from
    "that channel found nothing"."""
    if not spec:
        return {}
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in families:
            raise SystemExit(f"--family-weights names {k!r}, which is not in --families "
                             f"({', '.join(families)})")
        out[k] = float(v)
    missing = [f for f in families if f not in out]
    if missing:
        raise SystemExit(f"--family-weights is missing {missing} — weight every family in "
                         f"--families explicitly, or omit the flag entirely (B per family)")
    tot = sum(out.values())
    if tot <= 0:
        raise SystemExit("--family-weights must sum to something positive")
    return {k: v / tot for k, v in out.items()}


def derive_tau_rec(tau_h: dict, *, frac: float = Q4_REC_FRAC,
                   floor_abs: float = Q4_REC_FLOOR_ABS) -> dict:
    """Per-partition RECORDING floor from the harvest cut. Pure, so the rule is testable.

    Never above `tau_h` (a recording floor that raised a cut would silence the rows it
    exists to keep) and never below `floor_abs`."""
    return {p: min(float(t), max(float(floor_abs), float(frac) * float(t)))
            for p, t in tau_h.items()}


# Matt's interior rule at SOURCING — AND IT IS A PARITY TRIPWIRE HERE, NOT A NEW GATE.
# That correction is the shakedown's, measured rather than reasoned: the rule is
# `interior_fraction > 0.30 => class 1` (`tools/corpus/apply_interior_rule.py`), and the walk
# has been enforcing exactly that bound on exactly that quantity all along —
# `EXPAND_FLAGS` passes `--descent-black-cap ps.BLACK_CAP` and `BLACK_CAP` IS 0.30. So every
# candidate that reaches Python is already under the threshold by construction, and on the
# shakedown's 154 recorded candidates the observed maximum was 0.262 with a median of 0.000.
# `interior_gated` is expected to be ZERO on this path, and that is the point of keeping it:
#
#   * it costs one float comparison per candidate and one term in the reconcile identity;
#   * it fires the moment `BLACK_CAP` and Matt's rule diverge — which is a real hazard,
#     because they are two independently-owned constants that happen to be equal, and the
#     engine's is a DESCENT parameter somebody could tune for a different reason entirely;
#   * a guard whose input is deliberately empty keeps its MECHANISM tested via an injected
#     value (`verification_practice.md` §2), which is what the unit tests do.
#
# So do NOT read `interior_gated: 0` in a readout as "the rule found nothing to reject". It
# means the engine rejected it one stage earlier. Where the rule is genuinely load-bearing is
# the label store (crops already rendered) and `label_seeded_harvest`, which screens at the
# VIEW frame — a different frame from the walk's node frame, and one no black cap covers.
#
# The measure itself is free: `--expand` emits `int_frac` per candidate in its own sidecar,
# the same non-escaped fraction the Rust `render::black_fraction` and `present.rs::BLACK_THRESH`
# use, on the candidate's OWN frame.
INTERIOR_GATE_DEFAULT = True

# --- maneuvers-on-admissions (v1.6). The prompt's k set, and it is NOT the walk's.
# `MAN_K_DEFAULT` is `none,4,16`: the 4x frame answers "is this atom good?", which the walk
# needs because it is deciding whether to expand. A trigger fires on a location that has
# ALREADY decoded >= 3 — the atom question is answered — so the 4x frame buys a third field
# per nucleus to re-ask it. `none` (preserve the admitted frame's scale at the nucleus) and
# `16` (the framing worth LABELING, `minibrot_maneuvers.md` §7) are the two that produce
# material this run wants.
TRIG_K_DEFAULT = "none,16"
TRIG_MAX_PER_BATCH_DEFAULT = 4   # bounds the bill; the admission rate bounds the opportunity
TRIG_DEADLINE_S = 45.0           # per-trigger enumeration deadline (mnv honours it internally)

FRONTIER_CAP = 6000      # prune the frontier to the top-N by priority (memory bound)
MAN_FRONTIER_SHARE = 0.5  # ... of which maneuver nodes may hold at most this fraction. They
                          # are PROTECTED from the pooled priority prune, not exempt from the
                          # bound — see push_children for the starvation this stops.
JULIA_ROOT_FW = 3.0      # fixed z-plane base-scale root view (matches --julia-root-fw)
EXPAND_TIMEOUT_S = 900   # hard-kill backstop on a hung --expand call
MIN_UNIT_TIMEOUT_S = 60  # floor for the budget-clamped per-unit backstop (unit_timeout_s)
# Standing bound on ONE `draw_roots` call, clamped to what is left of the wall budget by
# `root_draw_budget_s`. The PRE-LOOP draw is the one that needed this: it runs before
# `_session_t0` is set and before `active_s` starts accruing, so it is outside BOTH caps, and
# a hung engine there hangs the night with every budget check still believing the run has not
# started. A backstop longer than the job's budget is not a backstop, so the clamp is the
# load-bearing half, not the constant.
#
# SIZED FROM A MEASUREMENT, and CONTENTION DOMINATES IT — three numbers, 2026-08-03:
#   * 11.4 min for TWO families, box running two other smokes (`scratch/shakedown/e_sched.log`)
#     -> ~5.7 min/family, which projected ~23 min for the four-family mix;
#   * 9.2 min for FOUR families on a QUIET box (`data/discovery/continuous_v1_20260803/run.log`,
#     the launch this bound was raised for) -> ~2.3 min/family. The contended projection
#     over-estimated by 2.5x.
#   * `wall_elapsed_s` records ~12 min for four families, consistent with the quiet number.
# So the honest reading is that this cost is set by what else is on the box, not by the family
# count alone, and a pre-loop draw cannot know what else is on the box. The bound stays at the
# PESSIMISTIC end deliberately: a backstop sized off the quiet case starts truncating real work
# the first busy night, and a pre-loop truncation costs whichever family is LAST its fresh-start
# roots entirely. 40 min is ~1.7x the contended projection, ~4.3x the quiet measurement, and
# still ~8% of one night — cheap insurance against a hang, which is what it is actually for.
ROOT_DRAW_BUDGET_S = 40 * 60
MIN_ROOT_DRAW_S = 120    # floor: never shoot a draw merely for being the last thing running
ROOT_LOW_WATER = None    # replenish roots when frontier < this (set to B at runtime)

# --- per-partition root low-water (the fix for a starved partition inside a healthy frontier).
#
# `ROOT_LOW_WATER` is GLOBAL, and a global low-water mark cannot see per-partition starvation:
# arm B of allocator_prereg_v1 ran 427 batches with `draw_roots` firing exactly zero times
# in-loop, because the frontier held 944 nodes against a low-water of 32 the whole time. Eight
# of nine partition queues were empty at b381 and the survivor held 97% of the frontier
# (julia:mandelbrot expansions beget julia:mandelbrot, so the collapse is self-locking); 46% of
# post-fold pop intent pointed at a partition with an empty queue. The frontier was healthy and
# the run was starving.
#
# So the check is per-partition, and it is the SAME lever — a family below its own floor is
# drawn even when nothing global is low, and when everything is low every family is starved,
# which is the old behaviour exactly. Two bounds keep a targeted refill from eating the run,
# because it is not free: arm B's 4-family pre-loop draw cost 9.2 min against a ~0.34 min
# batch, i.e. ~7 batches of mining per family drawn.
PARTITION_LOW_WATER = None   # a partition below this many frontier nodes is STARVED (=B at runtime)
ROOT_REFILL_COOLDOWN = 10    # batches a family must wait between targeted refills
ROOT_REFILL_SHARE = 0.25     # in-loop root-draw seconds may not exceed this share of loop wall

# --- the pop-quota cost-to-mine SEED table (`--quota-prices`), and why it has a DEFAULT.
#
# `--quota-prices` used to default to None and the loader used to be `if path and
# path.exists()`, so BOTH ways of getting no table — not passing the flag, and passing a path
# that is not there — landed on the same silent flat 3.0. A flat seed asserts every partition
# costs the same, which the first steady-state run measured to be wrong by 32x, and it does so
# invisibly: nothing in the run record distinguishes "seeded flat on purpose" from "the file
# moved". So the default is the artifact, and its ABSENCE IS FATAL (`load_quota_prices`).
#
# The REGULARIZED table and not the measured one: allocation is biased toward the measured
# prices without being governed by them (`regularize_quota_prices.py`). The measured table
# stays on disk as the evidence and is what the regularized one is re-derived from.
#
# RESEEDED OFF RUN 2 (2026-08-07): `..._regularized_20260807.json`, derived from
# `steady_state_v2_20260807` (357 active min) at alpha=0.9 with a 16x live-EMA band. The
# run-1 pair (`quota_prices_v1.json` / `..._regularized_v1.json`, 60 warm-up minutes,
# alpha=0.7, 4x band) stays on disk as the record of what run 2 itself was seeded with.
#
# REGENERATED AT price_ema=0.15 (2026-08-12): `..._regularized_20260812.json`. The SAME run-2
# telemetry re-derived after `pop_quota.PRICE_EMA` was halved for the per-served-batch
# estimator — a one-variable change, every seed byte-identical to the 20260807 pair. The table
# pins its own `price_ema`, so until this default moved, production ran the 0.30 rate the code
# had already left.
#
# RESEEDED OFF RUN 27 (2026-08-12): `..._regularized_20260812_run27.json`, derived from
# `prod27_20260812` alone at the same alpha=0.9 and 16x band. Run 2 never priced `mandelbrot`
# or `julia:mandelbrot` (0.3 and 0.0 units), so every table down to the 20260812 pair carried
# them at the flat 3.0 — ~18x and ~16x above what run 27 measured, on the partition whose
# lockout the run was launched to test. Run 27 prices all NINE rows above the evidence floor,
# so the reseeded pair has NO defaulted row. Single-source and not pooled with run 2: `tau_h`
# was enlarged on 2026-08-08, and `units_mined` is counted past `tau_h`, so run 2's units are
# a different quantity. The 20260812 (run-2) pair stays on disk as the previous rung.
QUOTA_PRICES_DEFAULT_REL = "data/atlas/quota_prices_regularized_20260812_run27.json"
QUOTA_PRICES_DEFAULT = ROOT / QUOTA_PRICES_DEFAULT_REL


def load_quota_prices(path=None) -> dict:
    """The `--quota-prices` config for `pop_quota.CostToMine`, or RAISE naming the file.

    Absence-INTOLERANT, in both directions and deliberately (`verification_practice.md` §2 —
    an absence-tolerant load un-guards exactly when its subject goes missing). A missing
    artifact silently becomes `CostToMine`'s flat `SEED_PRICE`, and a flat seed is not a
    degraded version of a priced one: it is a different allocation policy, asserting a 1x
    spread over prices measured at 32x, and it leaves no trace in the run record. `None`
    resolves to the default artifact rather than to the flat seed for the same reason."""
    p = Path(path) if path else QUOTA_PRICES_DEFAULT
    if not p.exists():
        raise SystemExit(
            f"--quota-prices table missing: {p}\n"
            f"The pop quota will NOT fall back to the flat seed price "
            f"({pquota.SEED_PRICE} min/unit for every partition) — that is a different "
            f"allocation policy, not a degraded one, and it would run unrecorded.\n"
            f"Rebuild it:\n"
            f"    uv run python tools/atlas/regularize_quota_prices.py --write\n"
            f"(which reads the measured table `{rqp.DEFAULT_SOURCE}`; regenerate THAT from a "
            f"finished run with tools/atlas/derive_quota_prices.py)")
    return json.loads(p.read_text(encoding="utf-8"))

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
    # A DERIVED partition renders exactly like its base — `phoenix:classic` is the same Rust
    # family at a pinned parameter point, so it is normalized away here rather than given a
    # second arm in every branch below (and in descend_flags / render_args_for / ident_c /
    # loc_of, which all route through the same normalization).
    partition = P.base_partition(partition)
    if partition == "mandelbrot" or partition in ("multibrot3", "multibrot4", "multibrot5"):
        return partition
    if partition == "phoenix":
        return "phoenix"
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    raise ValueError(f"unknown partition {partition!r}")


# A phoenix candidate's dynamical parameter is a SIX-vector `(c, p, z_{-1})`, not the julia
# two-vector, and it rides the same `c` slot on the frontier node so `expand_batch`'s
# group-by-(partition, tuple(c)) keeps working unchanged. These two helpers are the only
# places that have to know the difference.
def _phoenix_cpz(c):
    """`(c_re, c_im, p_re, p_im, zm1_re, zm1_im)` as strings from a phoenix node's `c`."""
    if c is None or len(c) != 6:
        raise ValueError(f"phoenix needs a 6-vector (c, p, z_-1); got {c!r}")
    return [str(v) for v in c]


def phoenix_family_params(c) -> dict:
    """The Location `family_params` for a phoenix node — `p` and `z_{-1}`.

    `location.FAMILY_PARAM_KEYS['phoenix']` is `(p_re, p_im, zm1_re, zm1_im)`; the primary
    constant `c` goes in the Location's own `c_re`/`c_im`, exactly as julia's does."""
    v = _phoenix_cpz(c)
    return {"p_re": v[2], "p_im": v[3], "zm1_re": v[4], "zm1_im": v[5]}


def partition_for_phoenix_c(c) -> str:
    """`phoenix:classic` or `phoenix` for a phoenix node's 6-vector.

    DERIVED, never carried: a phoenix root's partition is a function of its parameter point,
    so the seed pool, the ledger and the corpus cannot disagree about which half a point
    belongs to. Reads the same registry the label census reads."""
    v = _phoenix_cpz(c)
    row = dict(zip(("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"), v))
    row["fractal_type"] = "phoenix"
    row["cx"] = row["fw"] = "0"          # schema marker: this IS a render-shaped point
    return P.partition_of_row(row)


def descend_flags(partition: str, c) -> list:
    """guided-descend --expand kernel flags for a homogeneous group (mirrors the walk grammar)."""
    partition = P.base_partition(partition)
    if partition == "mandelbrot":
        return []
    if partition in ("multibrot3", "multibrot4", "multibrot5"):
        return ["--family", partition]
    if partition == "phoenix":
        v = _phoenix_cpz(c)
        # `--phoenix` is the degree-2 two-state plane and is mutually exclusive with
        # `--julia` and with `--family multibrot*` (src/guided_descend.rs).
        return ["--phoenix", "--c", v[0], v[1], "--p", v[2], v[3],
                "--phoenix-z1", v[4], v[5]]
    if partition == "julia:mandelbrot":
        return ["--julia", "--c", str(c[0]), str(c[1])]
    if partition.startswith("julia:multibrot"):
        base = partition.split(":", 1)[1]
        return ["--family", base, "--julia", "--c", str(c[0]), str(c[1])]
    raise ValueError(f"unknown partition {partition!r}")


def render_args_for(partition: str, c) -> dict:
    """`family` / `c` / `family_params` for `prescreen._render` and `outcome_feature`.

    Exists because a phoenix node's `c` is a 6-vector and those helpers unpack `c` as a
    PAIR — passing the 6-vector straight through would raise, and passing its first two
    entries would silently render the DEFAULT phoenix plane at the right coordinates."""
    partition = P.base_partition(partition)
    if partition == "phoenix":
        v = _phoenix_cpz(c)
        return dict(family="phoenix", c=(v[0], v[1]),
                    family_params=phoenix_family_params(c))
    return dict(family=render_family_of(partition), c=c, family_params=None)


def ident_c(partition: str, c):
    """The candidate's DUP-IDENTITY vector for `production_seeder.is_distinct`.

    `ps.as_c` coerces to a `(float, float)` pair, which is right for julia and WRONG for
    phoenix: it would keep only `c` and silently declare two phoenixes with the same `c` but
    different `p` or `z_{-1}` to be the same point. `ps.near_dup` already accepts the 6-D
    phoenix identity (`row_phoenix_key` / `row_ident`), so the fix is to hand it the whole
    vector rather than to widen the coercion."""
    if P.base_partition(partition) == "phoenix":
        return tuple(float(v) for v in _phoenix_cpz(c))
    return ps.as_c(c)


def loc_of(partition: str, c, cx, cy, fw):
    partition = P.base_partition(partition)
    if partition == "phoenix":
        v = _phoenix_cpz(c)
        return ps.make_loc_of("phoenix", (v[0], v[1]),
                              phoenix_family_params(c))(cx, cy, fw)
    return ps.make_loc_of(render_family_of(partition), c)(cx, cy, fw)


# --------------------------------------------------------------------------- #
# tau_h — per-partition CHEAP p_good harvest cut: the cut that RETAINS ~90% of the frames a
# canonical render would have kept (= the 10th percentile of cheap p_good among the frames
# whose CANONICAL p_good clears `floors.GOOD_FLOOR`).
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
# / `phoenix` per-partition t_good overrides that retired with the whole t_good table on
# 2026-08-09. A v7 floor on a v8 base is the same category error the version stamp below
# exists to stop, so it is not applied — and it cannot be re-derived: the floor's definition
# ("raise the cut to where it starts costing admits") needs ADMISSIONS under the active
# head, and no post-v7 discovery run has produced the curve. The mechanism stays live and
# tested; the table is empty on purpose, with its own stamp so an unstamped re-add is visible.
#
# It would be a no-op even if applied — every re-derived v8 base (0.199..0.704) is already
# far above every v7 floor (0.216..0.269) — but "harmless today" is not why it is empty.
TAU_H_CAMPAIGN_FLOOR_MODEL = "v11"
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
# RE-DERIVED UNDER v11 on 2026-08-09 by `tools/atlas/tau_h_rederive.py`, which is the
# regeneration path this comment's failure message points at. Provenance artifact:
# `data/atlas/tau_h_base_v11.json` (per-partition n_rows / n_good / value / source).
#
# THIS IS THE THIRD v11 TABLE AND THE FIRST UNDER THE FIXED CUT. All three are kept, because
# each is the record of what production served for a stretch of 2026-08-08/09:
#   `tau_h_base_v11_adoption.json`  — the flip's derivation, 3,492 rows, two arms.
#   `tau_h_base_v11_two_arm.json`   — the same-day enlargement, 64,365 rows, two arms.
#   `tau_h_base_v11.json`           — THIS one (prompts/selection_restructure_3.md).
#
# WHAT CHANGED, AND BOTH CHANGES REMOVE A CONFOUND RATHER THAN ADD PRECISION.
#
#   1. THE BAR IS `floors.GOOD_FLOOR`, NOT `t_good_for(partition)`. The run side admits on
#      the fixed floor now, so conditioning this estimator on a per-partition table would
#      derive a harvest cut for a gate that no longer exists. It also ends the reading
#      problem the two-arm era never solved: the v10 -> v11 move was described as "the head"
#      and was really t_good (mandelbrot's bar went 0.03 -> 0.90, ~30x stricter, and the cut
#      followed it 0.0229 -> 0.6257). With one bar for every partition and every version, a
#      future move is a move in the head or in the population and nothing else.
#
#   2. THE HARVEST ARM IS GONE — with it the two-arm minimum, `harvest_log_registry` and the
#      per-run truncation record. The harvest log holds only checks that already cleared a
#      PREVIOUS head's tau_h, at a level that differed per run: a left-truncated sample at a
#      MIXTURE of levels (mandelbrot's rows alone spanned 0.0229 to 0.7041 across four tau
#      eras), whose quantile is an upper bound of unknown tightness. Every derivation had to
#      carry a paragraph naming its own least trustworthy number — v11's multibrot4 0.8245
#      rested on that arm alone with nothing bounding it from below. The walk-outcome ledger
#      is a uniform-random gate survivor per rung, never tau-selected, so it is untruncated;
#      a smaller unbiased sample beats a larger one with an unquantifiable bias.
#
# METHOD, unchanged in shape: each walk-outcome geometry is re-rendered at BOTH presentations
# (384x216 ss1 cheap / 640x360 ss2 canonical) and re-scored under the ACTIVE head; tau_h is
# the 10th percentile of cheap p_good among the frames whose canonical p_good clears
# GOOD_FLOOR. The whole ledger is used (`--per-partition 0`), 1,148 rows over 8 partitions.
#
# THE VALUES ARE FLATTER THAN THE TWO-ARM TABLE and that is the point: the spread was
# 0.20..0.82, it is now 0.20..0.55. Both effects push the same way — the bar fell for the
# partitions that had been tightened (multibrot4 0.85 -> 0.50), and the arm that used to
# supply the high numbers was the truncated one.
#
# THE THIN END IS NAMED, because the fail-open rule means thinness costs render time rather
# than supply and is therefore easy to stop looking at. Good-row counts: julia:mandelbrot 59,
# julia:multibrot3 34, julia:multibrot4 32, julia:multibrot5 23, mandelbrot 23, multibrot4 15,
# multibrot3 12, **multibrot5 8** — three above min_n=5 by single digits. A 10th percentile on
# 8 rows is the second-smallest value; read multibrot5, multibrot3 and multibrot4 as "roughly
# half" rather than as four significant figures.
#
# THE ONE BIAS THAT REMAINS, STATED. The walk population is off-distribution in the other
# direction: walk outcomes are not frontier candidates, so the cheap/canonical relationship
# is measured on a rung survivor rather than on a maneuver push. That is not corrected for and
# there is nothing in the tree to correct it with — the only alternative population is the
# truncated one this derivation just dropped.
TAU_H_FIDELITY_BASE_MODEL = "v11"
TAU_H_FIDELITY_BASE = {
    "mandelbrot": 0.3145107567310333,
    "multibrot3": 0.5449502170085907,
    "multibrot4": 0.4743058800697326,
    "multibrot5": 0.5378899842500686,
    "julia:mandelbrot": 0.200508177280426,
    "julia:multibrot3": 0.4735631048679352,
    "julia:multibrot4": 0.5339927613735199,
    "julia:multibrot5": 0.46945214867591856,
}


# Below this many GOOD frames a partition is not cut at all and harvests everything (0.0).
# IMPORTED from the deriver rather than restated: the live record-derived path here and the
# regeneration path there must not disagree about which partitions are cuttable.
TAU_H_MIN_N = _thr.MIN_N


def _apply_campaign_floor(part: str, val: float) -> float:
    """Raise a base tau_h to `part`'s campaign floor (max — only ever raises, never lowers)."""
    floor = TAU_H_CAMPAIGN_FLOOR.get(part)
    return max(val, floor) if floor is not None else val


def _derive_tau_h_base_from_records(partitions: list[str], keep: float) -> dict:
    """Per-partition cheap-p_good cut from the fidelity study records (PRE campaign floor).

    The cut RETAINS ~`keep` of the frames whose CANONICAL p_good clears `floors.GOOD_FLOOR`
    (= the (1-keep) quantile of cheap p_good among those frames). Same estimator, same bar and
    same fail-open rule as `tools/atlas/tau_h_rederive.derive`, which is the path that actually
    produces the vendored table — two estimators that disagreed about the bar or about what a
    thin partition gets would be the second quality definition this restructure removed.

    A partition with fewer than `TAU_H_MIN_N` good rows gets 0.0 and harvests everything: a
    too-high cut sheds admissions invisibly, a too-low one shows up as GPU-minutes."""
    rec = json.loads(FIDELITY_RECORDS.read_text(encoding="utf-8"))
    can, cheap = rec["scores"]["canonical"], rec["scores"]["cheap"]
    fam_of = {s["id"]: s["family"] for s in rec["samples"]}
    q = 1.0 - keep

    base = {}
    for part in partitions:
        vals = [cheap[i][2] for i in can                      # cheap p_good
                if fam_of.get(i) == part and i in cheap and can[i][2] >= F.GOOD_FLOOR]
        base[part] = float(np.quantile(vals, q)) if len(vals) >= TAU_H_MIN_N else 0.0
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
                f"  Re-derive with tools/atlas/tau_h_rederive.py (walk-outcome ledger, "
                f"canonical p_good >= floors.GOOD_FLOOR), then update TAU_H_FIDELITY_BASE + "
                f"TAU_H_FIDELITY_BASE_MODEL together — never one without the other. Or re-run "
                f"tools/studies/descent_score_fidelity.py under {active} to regenerate "
                f"{FIDELITY_RECORDS.name}, which overrides the vendored table.")
        base = {p: TAU_H_FIDELITY_BASE.get(p) for p in partitions}
        missing = sorted(p for p, v in base.items() if v is None)
        if missing:
            raise SystemExit(
                f"tau_h derivation: {FIDELITY_RECORDS} absent and no vendored base for "
                f"{missing} — regenerate via tools/atlas/tau_h_rederive.py (or "
                f"tools/studies/descent_score_fidelity.py), or add the partition to "
                f"TAU_H_FIDELITY_BASE")
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


def priority_terms(eord, g, dup_pen, cos_max, lambda_m, beta, depth, lo, hi,
                   sat_density=0, sat_strength=0.0):
    """Pure priority decomposition. Returns (priority, {terms}). At lambda_m==0 AND beta==0 AND
    sat_strength==0 this is byte-identical to the pilot's `eord + gumbel - dup_pen` (the
    novelty, depth and saturation terms all vanish).

    THE SATURATION DISCOUNT IS MULTIPLICATIVE ON `eord` AND ADDITIVE ON NOTHING, which is what
    keeps it soft. `eord` is the head's E[ord] in [0, K-1] and is the only non-negative term
    here, so scaling it demotes a candidate towards — never below — what an unscored root
    would rank at, and the Gumbel/depth terms still separate the survivors. A saturated place
    with a great score therefore loses to a fresh place with a merely good one, and still
    beats a fresh place with a bad one. Subtracting a penalty instead would be unbounded below
    and would make "saturated" eventually mean "unreachable"."""
    nov_pen = novelty_penalty(cos_max, lambda_m, lo, hi)
    depth_bonus = beta * depth
    sat_disc = vd.discount(sat_density, sat_strength)
    prio = eord * sat_disc + g - dup_pen - nov_pen + depth_bonus
    return prio, dict(eord=eord, gumbel=g, dup_pen=dup_pen, cos_max=cos_max,
                      nov_pen=nov_pen, depth_bonus=depth_bonus,
                      sat_density=sat_density, sat_disc=sat_disc, priority=prio)


def novelty_penalty(cos_max: float, lambda_m: float, lo: float, hi: float) -> float:
    """Morph-space near-repeat penalty: zero at substrate-typical similarity (cos<=lo), ramping
    linearly to full lambda_m at the near-repeat knee (cos>=hi). Anchors are empirical on the
    cheap-JPG substrate (morph_anchor_calibrate.py). A near-perceptual-dup of an admitted/
    expanded look sinks by ~lambda_m E[ord] units BEFORE it is popped. lambda_m=0 -> zero."""
    if lambda_m <= 0.0:
        return 0.0
    frac = (cos_max - lo) / (hi - lo)
    return lambda_m * min(max(frac, 0.0), 1.0)


def check_wall_budget_supported(dive: bool, wall_budget_min) -> None:
    """`--wall-budget` DOES NOT REACH DIVE MODE, and this refuses rather than no-ops.

    `run_dive` has its own loop and checks only the active budget and the STOP sentinel —
    `wall_elapsed_s`, `over_wall_budget` and the batch-boundary check all live on the crawl
    path. So `--wall-budget --dive` is a flag that silently does nothing, and the 2026-08-05
    dive's launch record had to carry a hand-written note saying so; a launch record is not a
    place a guarantee can live.

    REFUSED rather than warned, because the reason anyone passes it is to bound a run they are
    not watching, and a warning scrolled past in a background log bounds nothing (`CLAUDE.md`,
    "a backstop longer than the job's budget is not a backstop"). The feature is deliberately
    NOT implemented here — this records the limitation at the flag."""
    if dive and float(wall_budget_min or 0.0) > 0.0:
        raise SystemExit(
            "--wall-budget has NO EFFECT in --dive mode: run_dive() checks only the active "
            "budget (--budget) and the STOP sentinel, so the cap would silently never fire. "
            "Drop the flag and bound the dive with --budget, or stop it with "
            "`touch <run_dir>/STOP`.")


# --------------------------------------------------------------------------- #
# THE TWO-ENTRY-POINT CONTRACT. `run()` and `run_dive()` share one constructor, and 41 of
# its attributes are read on the crawl path and never on the dive path. That is not a bug by
# itself — most of them are genuinely inapplicable to a single-track descent — but until this
# table existed the code could not tell "deliberately N/A in dive mode" from "silently dropped
# in dive mode", and nothing detected the difference. `--wall-budget` was the second kind: it
# parsed, converted, stored, and then no-oped, and the fact had nowhere to live but a
# hand-written note in a launch record.
#
# So the inapplicable set is DECLARED, one line of reason each, and
# `test_steered_frontier.py::test_every_crawl_only_constructor_attribute_is_declared`
# recomputes the reachability from the AST and asserts set equality. A new flag that does not
# reach the dive path now fails there — cheap, at the time it is written — instead of in a
# launch record. Three ways to satisfy the test, in preference order:
#
#   1. make the dive path read it (it was meant to apply);
#   2. REFUSE at the flag, like `check_wall_budget_supported` — the right answer whenever the
#      attribute is a BOUND rather than a tuning knob, because a bound that silently does not
#      apply is worse than no bound;
#   3. add it here with the reason it cannot apply.
#
# NOT a list of dead attributes: every one of these is live on the crawl path.
# --------------------------------------------------------------------------- #
_MANEUVER_NA = ("maneuver machinery — `self.maneuvers` is forced False in dive mode, so "
                "`propose_maneuvers` and everything under it is unreachable")
DIVE_IGNORES: dict[str, str] = {
    # --- explicitly neutralized in the constructor -------------------------------
    "maneuvers": "forced False in dive mode (`... and not self.dive`): a dive is one track "
                 "down from an admission, and a maneuver is a lateral re-aim",
    # --- the maneuver internals, unreachable THROUGH that flag --------------------
    **{a: _MANEUVER_NA for a in (
        "man_ks", "man_lateral", "man_quota", "man_gov", "man_probe_s", "man_screens",
        "man_screen_s", "man_range_dist", "man_range_prior", "man_range_gain", "man_nbh",
        "man_nbh_m", "man_nbh_n", "man_nbh_probes", "man_passed_logged", "man_fields",
        "man_view_prior", "man_view_gain", "man_view_params")},
    # --- the frontier queue: a dive has no queue, no roots and no breadth ---------
    "expansions_per_root": "per-root expansion budget, spent by `pop_batch*` — a dive expands "
                           "one node per rung and pops nothing",
    "last_refill_batch": "refill bookkeeping (`starved_families`/`refill_starved`); the dive "
                         "never refills a queue because it never draws from one",
    "seeders": "native root seeders, read only by `draw_roots` — a dive's starting points are "
               "the source run's admissions, so no roots are drawn",
    "pool_cursor": "julia/phoenix supply-pool cursor (`_take_from_pool`), seeding only",
    "seed_pool_rate": "how often the supply pools are topped up — nothing seeds in dive mode",
    "freshness_prior": "library-freshness prior on the ROOT DRAW; no roots are drawn",
    "prior_rows": "the library rows that prior reads (`build_clouds`), same reason",
    "root_draw_s": "cumulative in-loop root-draw wall, for the crawl's cost model",
    "_served_partition": "round-robin partition service inside `pop_batch_scheduled/_quota`",
    "node_embs": "frontier node embeddings, used to prune and re-rank a QUEUE of pending "
                 "nodes; a dive holds one node at a time",
    # --- priority / novelty: `lambda_m` is forced 0, so `push_children` never runs ---
    "beta": "depth term of the frontier priority — the dive's child is argmax cheap p_good "
            "with a Gumbel tie-break, never a priority score",
    "morph_lo": "morph-novelty anchor (lo), read by `push_children` and the crawl's records",
    "morph_hi": "morph-novelty anchor (hi), same",
    "anchor_src": "provenance of those anchors, recorded by `save_state`/`finish`",
    "sat_cos": "the saturation knee the crawl reports novelty against",
    "recency_k": "morph-memory window size; consumed at construction into `MorphMemory`, "
                 "which the dive only saves — it folds no expanded looks in",
    "prio_log": "`push_children`'s per-candidate priority log",
    "sat_log": "`push_children`'s saturation log",
    "sat_by_partition": "per-partition tally of the cross-run saturation discount, filled by "
                        "`push_children`. The knobs themselves (`sat_k`/`sat_strength`/"
                        "`sat_on`/`sat_index`) are NOT here — `write_run_config` stamps them "
                        "on both paths, and stamps `n_a` for the dive — but a dive orders no "
                        "frontier, so there is nothing for this to count",
    # --- accounting the dive replaces wholesale ------------------------------------
    "state_path": "the crawl checkpoint; the dive checkpoints to `dive_state_path`",
    "_session_t0": "wall-clock origin for `wall_elapsed_s` — THE `--wall-budget` CLASS. The "
                   "dive loop checks `--budget` and the STOP sentinel only, and the flag is "
                   "refused at parse time by `check_wall_budget_supported` rather than "
                   "silently ignored here",
    "wall_s_base": "wall seconds carried across resumes, same class as `_session_t0`",
}


def interleave_dive_arms(plan: list) -> list:
    """Re-sequence a dive plan so EVERY PREFIX carries both arms in proportion.

    A dive plan is two arms — `top` (the source admissions with the highest canonical p_good)
    and `control` (drawn from the same admissions regardless of score) — and the ONLY thing it
    measures is the contrast between them. Every dive plan so far ordered the arms in blocks:
    built top-then-control, and then (scheduler ON) sorted by partition deficit, which is an
    arm-blind key that happened to concentrate one arm at the front. The 2026-08-05 dive stopped
    at 7 of 28 on its active budget, and its first FOUR entries were controls; a plan built
    without the scheduler stops even worse, because 20 top followed by 8 control truncated at 7
    yields zero controls. A truncating budget is the normal case, not the exception — the run
    that produced this plan hit it — so the plan has to be readable at whatever length it
    reaches, not only at N.

    GREEDY LARGEST-DEFICIT (Webster/Sainte-Laguë) over the ARM axis — literally the same code
    the label sheet deals (source x family) with, `apportion.sequence_by_deficit`, which is
    where the rule now lives: at position L the next entry comes from the arm furthest below
    its proportional share L*n_a/N, so each arm is within 1 of its share in EVERY prefix.
    TWO ARMS is the case where that bound is a property of the rule rather than of the
    population (worst deviation 0.5, exhaustive to 60x60 in `test_apportion.py`); it is still
    asserted on the built order here, because the ARM COUNT is the reason it holds. WITHIN an
    arm the incoming order is preserved untouched, so the deficit sort above still does exactly
    what item 8 asked of it — deficit families first, within each arm.

    Unconditional, and that is the point: the scheduler-OFF plan is the worse of the two and had
    no re-ordering at all. Ties break toward the LARGER arm and then by name, so the sequence is
    a pure function of the plan."""
    arms: dict = {}
    for e in plan:
        arms.setdefault(e["start_group"], []).append(e)
    keys = sorted(arms)
    sizes = {k: len(arms[k]) for k in keys}
    if len(keys) < 2 or sum(sizes.values()) == 0:
        return list(plan)
    cursor = {k: 0 for k in keys}
    out = []
    for k in apportion.sequence_by_deficit(sizes):      # default tie: larger arm, then name
        out.append(arms[k][cursor[k]])
        cursor[k] += 1
    return out


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
        # bulk() since 2026-08-08 — the feature store resolves OUT of the tree for a
        # run under data/discovery/, and stays beside the ledger for a tmp_path or
        # scratch store (discovery_sinks.feats_path).
        self.feats_path = dsinks.feats_path(run_dir)
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
        # The parent is NOT self.dir any more (see __init__): make the FILE's parent.
        self.feats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.feats_path.parent / (self.feats_path.stem + "_tmp.npz")
        np.savez_compressed(tmp, **self.feats)
        os.replace(tmp, self.feats_path)

    def clouds(self, partitions: list[str]) -> dict:
        return {p: ps.build_cloud(self.rows, p) for p in partitions}


# --------------------------------------------------------------------------- #
# Scratch teardown.
#
# A run's own scratch (`<run_dir>/scratch`, routed under ARTIFACTS_ROOT by `_bulk_scratch`)
# is THE byte and file-count bomb: one 6 h steady-state run left 118.07 GB / 138,567 files,
# ~18 GB per engine-hour, and on 2026-08-07 two runs from that single day were 86% of a
# 178 GB bulk store (`scratch/artifacts_audit_report.md`). Nothing guards that store's size
# — `tools/audit/size_guard.py` walks REPO_ROOT only, and its rule (c) exists to push bulk
# out to exactly here, where no registry, threshold or test looks.
#
# The tree is residue the moment the run closes: every verdict it carries is already in the
# tracked `outcome_ledger.jsonl` (`guard_pass`/`guard_fail` beside `outcome_cx/cy/fw`, so any
# field is re-renderable), and the two post-run readers of any run scratch both default to a
# run dir that no longer exists. So it was deleted by hand — which is why the 154 GB above
# was 8 h late.
#
# CLEAN CLOSE ONLY, and that is the whole safety argument. Teardown hangs off the summary
# write and nothing else: no `finally`, no `atexit`, no signal handler (pinned by
# `test_steered_frontier.py`'s teardown-reachability gate). An interrupted, killed or crashed
# run keeps its scratch, because that is precisely the run whose intermediate state you may
# still need to read — a closed run can be asked its summary, a dead one cannot.
SCRATCH_TEARDOWN_KEY = "scratch_teardown"

# --------------------------------------------------------------------------- #
# ...and the number the teardown above is judged against: the WHOLE bulk store's size.
#
# `tools/audit/size_guard.py` walks REPO_ROOT, so the one tree it can never see is the one
# its own rule (c) pushes everything into. This is the only number in the system that looks
# at ARTIFACTS_ROOT, and it is stamped at the close of every run — the moment when the run
# that just grew it is still identifiable.
#
# MEASURED AFTER TEARDOWN, deliberately: the run's own scratch lives INSIDE this store, so a
# pre-teardown walk would count ~100 GB that is about to be deleted and trip the watch on
# every long run. The number reported is the store as this run LEAVES it.
BULK_STORE_KEY = "bulk_store"
# Loud key above this. Set 2026-08-10 at ~2x the store's then-current 33.47 GiB
# (389,325 files / 35,938,614,727 B, measured at C:\Code\fractal-maker-artifacts). It is a
# WATCH, not a cap: nothing refuses, the run has already finished, and the point is that the
# next reader of `summary.json` sees the growth instead of discovering it at 178 GB. Adjust
# freely — a threshold that trips every run is trained out (`verification_practice.md` §4).
BULK_STORE_WATCH_BYTES = 67 * 2**30
BULK_STORE_OVER_WATCH_KEY = "BULK_STORE_OVER_WATCH"


def measure_bulk_store(root=None) -> dict:
    """Total bytes + file count of the bulk store, plus the loud key when over the watch.

    NEVER RAISES, same contract as `teardown_scratch`: this runs after a closed run's summary
    is already on disk, so an unreadable directory is a recorded `error`, not a traceback that
    turns a finished 6 h run into a failure. A store that does not exist is `files=0`, which is
    the honest answer for a fresh checkout that has never run one.
    """
    # Resolved through the ONE ARTIFACTS_ROOT resolver `paths.bulk` delegates to, never a
    # second copy of the env-var-or-sibling rule — a store measured at a different address
    # from the one the run wrote into is a number about nothing.
    root = Path(root) if root is not None else _paths._artifacts.artifacts_root()
    rec = {"root": str(root), "watch_bytes": BULK_STORE_WATCH_BYTES}
    if not root.exists():
        return {**rec, "files": 0, "bytes": 0, "gib": 0.0, "present": False}
    n_files, n_bytes, n_err = 0, 0, 0
    for dirpath, _dirs, names in os.walk(root, onerror=lambda _e: None):
        for nm in names:
            n_files += 1
            try:
                n_bytes += os.path.getsize(os.path.join(dirpath, nm))
            except OSError:
                n_err += 1
    rec.update(present=True, files=n_files, bytes=n_bytes,
               gib=round(n_bytes / 2**30, 3), unstatable=n_err)
    if n_bytes >= BULK_STORE_WATCH_BYTES:
        rec[BULK_STORE_OVER_WATCH_KEY] = (
            f"bulk store is {n_bytes / 2**30:.1f} GiB across {n_files} files at {root}, over "
            f"the {BULK_STORE_WATCH_BYTES / 2**30:.0f} GiB watch. Nothing guards this tree — "
            f"size_guard walks the repo only. Prune finished runs' retained scratch, or raise "
            f"steered_frontier.BULK_STORE_WATCH_BYTES if this is the new normal.")
    return rec


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

    def _writer(self, path: Path):
        """THE append path for a committed per-row stream. One `run_record.SegmentWriter` per
        stream, opened lazily (so constructing the crawl creates no run dir) and cached (so the
        live-tail size is tracked in memory rather than stat'ed per row). Every one of these
        streams is in `run_record.SEGMENTED_STREAMS`; a bare `open(path, "a")` beside one is
        what `test_run_record.py`'s source scan fails on."""
        w = self._stream_writers.get(path)
        if w is None:
            w = self._stream_writers[path] = run_record.SegmentWriter(path)
        return w

    def finalize_streams(self):
        """Compress every live tail at the end of a run so the committed record is entirely
        `.jsonl.gz` segments. A killed run simply never calls this and keeps a plain tail of at
        most `run_record.ROTATE_BYTES` — readable by the same reader, just bigger."""
        for w in self._stream_writers.values():
            w.finalize()
        q = getattr(self, "quota", None)
        if q is not None and getattr(q, "_trace_writer", None) is not None:
            q._trace_writer.finalize()

    def teardown_scratch(self) -> dict:
        """Delete this run's own scratch subtree; return the record stamped into
        `summary.json`. See the SCRATCH_TEARDOWN_KEY block above for why, and for why this is
        reachable ONLY from the clean-close path.

        NEVER RAISES. The summary is already on disk when this runs, so a Windows file lock
        (the label servers hold handles under this tree) must degrade to a recorded
        `scratch_delete_failed`, not turn a closed 6 h run into a traceback."""
        target = self.scratch
        rec = dict(path=str(target), retain_flag=bool(self.retain_scratch))
        if self.retain_scratch:
            rec.update(outcome="scratch_retained", reason="--retain-scratch")
            return rec
        # The target is `<run_dir>/scratch` or its bulk() image. Anything else means
        # `_bulk_scratch` moved under us, and a recursive delete is the wrong response to
        # that: refuse and say so rather than guess. Derived here, not asserted at
        # construction, because this is the line that does the deleting.
        if (target.name != "scratch" or target == self.run_dir
                or target in self.run_dir.parents):
            rec.update(outcome="scratch_retained",
                       reason=f"REFUSED: {target} is not a run scratch subtree")
            return rec
        if not target.exists():
            rec.update(outcome="scratch_absent", files=0, bytes=0)
            return rec
        # Measured before the delete, because "how much did this free" is the number the next
        # run's disk budget is sized from and it is unrecoverable afterwards. One metadata
        # pass over ~140k files; a stat that fails is skipped, never fatal.
        n_files, n_bytes = 0, 0
        for dirpath, _dirs, names in os.walk(target):
            for nm in names:
                n_files += 1
                try:
                    n_bytes += os.path.getsize(os.path.join(dirpath, nm))
                except OSError:
                    pass
        try:
            shutil.rmtree(target)
        except OSError as e:
            rec.update(outcome="scratch_delete_failed", files=n_files, bytes=n_bytes,
                       error=f"{type(e).__name__}: {e}", still_present=target.exists())
            return rec
        rec.update(outcome="scratch_deleted", files=n_files, bytes=n_bytes,
                   gb=round(n_bytes / 2**30, 3))
        return rec

    def _close_summary(self, summary: dict) -> dict:
        """THE close path, shared by `finish` (crawl) and `finish_dive`. Compress the live
        tails, land `summary.json`, tear the scratch down, restamp the outcome.

        TWO WRITES, deliberately. The summary lands FIRST carrying `outcome="not_reached"`,
        so (a) the run is durably closed before a 140k-file delete begins, and (b) the
        third outcome is an OBSERVABLE STATE rather than a missing key: a summary still
        saying `not_reached` is a run interrupted *during* teardown, whose scratch may be
        half-gone — which reads identically to a pre-teardown summary if absence is the only
        signal. Both writes derive the value from what teardown actually did."""
        self.finalize_streams()
        path = self.run_dir / "summary.json"
        summary[SCRATCH_TEARDOWN_KEY] = dict(
            outcome="not_reached", path=str(self.scratch),
            note="teardown had not returned when this summary was written; a summary that "
                 "STILL says this was interrupted mid-teardown and its scratch may be "
                 "partially deleted")
        summary[BULK_STORE_KEY] = {"measured": False,
                                   "note": "measured after teardown; see BULK_STORE_KEY"}
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary[SCRATCH_TEARDOWN_KEY] = rec = self.teardown_scratch()
        # After teardown, so the store is measured as this run LEAVES it (this run's own
        # ~100 GB of scratch lives inside it and has just gone).
        summary[BULK_STORE_KEY] = bulk_rec = measure_bulk_store(self.bulk_store_root)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if rec["outcome"] == "scratch_deleted":
            print(f"[teardown] scratch deleted: {rec['files']} files / {rec['gb']:.2f} GB "
                  f"-> {rec['path']}", flush=True)
        else:
            print(f"[teardown] scratch {rec['outcome']}: {rec['path']}"
                  + (f" — {rec['reason']}" if rec.get("reason") else "")
                  + (f" — {rec['error']}" if rec.get("error") else ""), flush=True)
        print(f"[bulk-store] {bulk_rec.get('gib', 0.0):.2f} GiB / {bulk_rec.get('files', 0)} "
              f"files at {bulk_rec['root']} (watch "
              f"{BULK_STORE_WATCH_BYTES / 2**30:.0f} GiB)", flush=True)
        if BULK_STORE_OVER_WATCH_KEY in bulk_rec:
            print(f"[bulk-store] {BULK_STORE_OVER_WATCH_KEY}: "
                  f"{bulk_rec[BULK_STORE_OVER_WATCH_KEY]}", flush=True)
        return summary

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
        # The store that scratch is BORN into, resolved once here so the close-time size line
        # measures the same address the run wrote to (and so a test can point it elsewhere the
        # same way it points `scratch`).
        self.bulk_store_root = _paths._artifacts.artifacts_root()
        # ...and it is torn down again on a clean close unless this is set (see
        # SCRATCH_TEARDOWN_KEY). Read on BOTH entry points (finish / finish_dive both close
        # through `_close_summary`), so it needs no DIVE_IGNORES exemption.
        self.retain_scratch = bool(getattr(args, "retain_scratch", False))
        self._stream_writers: dict = {}          # path -> run_record.SegmentWriter (see _writer)
        # Per-unit stage timing. Every duration this crawl used to compute and drop —
        # the pre-loop root draw, each in-loop refill, each served batch, each dive — lands
        # here as one row, and its stage totals fold into `summary.json`. Purely additive:
        # nothing reads it back in-run, so a stage timer cannot change what the run does.
        self.stage_times = stimes.StageTimes(self.run_dir)
        self.state_path = self.run_dir / "state.json"
        self.stop_path = self.run_dir / "STOP"
        self.harvest_log = self.run_dir / "harvest_log.jsonl"
        # v1.6 record-and-rank: beside the ledger and beside the harvest log, on purpose.
        # The harvest log is the tau_h curve's input and is keyed on CHECKS; this is keyed on
        # CANDIDATES and is wider by construction (it holds the below-tau_h and gated
        # populations the harvest log has no row for). Append-only, so a kill loses at most
        # the batch in flight.
        self.q4_log = self.run_dir / "q4_candidates.jsonl"
        self.record_canon_dups = bool(getattr(args, "record_canon_dups", False))
        self.interior_gate_on = bool(getattr(args, "interior_gate", INTERIOR_GATE_DEFAULT))
        self.interior_discard = float(getattr(args, "interior_discard", _INTERIOR_DISCARD))
        # --- v1.6 maneuvers-on-admissions. A PER-BATCH cap rather than a probability:
        # the trigger fires on an event that is already rare (an admission), so a coin on
        # top would make the mechanism's rate a property of two rarities multiplied and
        # unreadable. The cap bounds the bill; the admission rate bounds the opportunity.
        self.trig_on = bool(getattr(args, "maneuvers_on_admissions", False))
        self.trig_ks = mnv.parse_k_spec(getattr(args, "trig_k", TRIG_K_DEFAULT))
        self.trig_nbh_m = int(getattr(args, "trig_nbh_m", mnv.NBH_MAX_FOUND))
        self.trig_nbh_probes = int(getattr(args, "trig_nbh_probes", mnv.NBH_MAX_PROBES))
        self.trig_period_max = int(getattr(args, "trig_period_max", lsh.SEED_PERIOD_MAX))
        self.trig_max_per_batch = int(getattr(args, "trig_max_per_batch",
                                              TRIG_MAX_PER_BATCH_DEFAULT))
        self.trig_deadline_s = float(getattr(args, "trig_deadline_s", TRIG_DEADLINE_S))
        self.trig_fired_this_batch = 0
        self.trig_probe_s = 0.0
        self.seed_pool_rate = int(getattr(args, "seed_pool_rate", 0) or 0)
        self.pool_cursor: dict = {}
        self.families = [f.strip() for f in args.families.split(",") if f.strip()]
        # --- v1.6 channel allocation: deficit-by-q4-gap root weights. A lighter instrument
        # than `--scheduler` on purpose — the scheduler allocates by price-weighted
        # DISTINCT-LOOK deficit against a target measure and needs a library look seed to
        # mean anything, and this run's deficit is a plain per-partition class-4 COUNT gap
        # computed off the label corpus before launch. Normalized here so the run config
        # records the same numbers the draw uses.
        self.family_weights = _parse_family_weights(
            getattr(args, "family_weights", None), self.families)
        for f in self.families:
            if f not in C_PLANE:
                raise SystemExit(f"--families must be c-plane ({C_PLANE}); got {f!r}")
        self.B = args.batch or B_DEFAULT
        # Per-partition root low-water. Default is B — one batch's worth — because that is
        # the quantity servability is denominated in: a partition that cannot fill a pop is a
        # partition the allocator's intent for it cannot be spent on.
        plw = getattr(args, "partition_low_water", None)
        self.partition_low_water = int(plw) if plw is not None else int(self.B)
        self.root_refill_cooldown = int(getattr(args, "root_refill_cooldown",
                                                ROOT_REFILL_COOLDOWN))
        self.root_refill_share = float(getattr(args, "root_refill_share",
                                               ROOT_REFILL_SHARE))
        self.root_draw_s = 0.0        # cumulative IN-LOOP root-draw wall (excl. the pre-loop draw)
        self.last_refill_batch: dict = {}
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
        # Whether the operator NAMED a pool or inherited the default. It decides one thing:
        # what a run without 'mandelbrot' in --families does. Naming a pool and getting no
        # julia:mandelbrot partition to inject it into is a contradiction and stays fatal;
        # inheriting the default on a phoenix-only run is not, and must not be — the default
        # exists so the common case gets the right pool, not so every other case fails.
        self.julia_pool_explicit = bool(jp) and str(jp) != str(JULIA_SUPPLY_POOL)
        pp = getattr(args, "phoenix_seed_pool", None)
        self.phoenix_seed_pool_path = Path(pp).resolve() if pp else None
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
        # --- v1.7 cross-run saturation memory. Strength 0 => the index is never built and
        # priorities are byte-identical to v1.6. Dive mode never reaches `push_children`, so
        # the memory is switched off there rather than built and unread — a loaded index the
        # dive path cannot consult is 0.4 s and a misleading `run_config` stamp.
        self.sat_k = float(getattr(args, "sat_radius_k", SAT_RADIUS_K))
        self.sat_strength = float(getattr(args, "sat_strength", SAT_STRENGTH))
        self.sat_on = self.sat_strength > 0.0 and not self.dive
        self.sat_index = None
        # Per-partition run tally, CHECKPOINTED: the headline this mechanism is judged on is
        # "did the discount fire, and where", and a resumed run that restarted the tally would
        # report the last session's share as the run's.
        self.sat_by_partition: dict = {}

        # partitions this run tracks a cloud for (c-plane + julia twins if hooked; dive covers
        # all twins so a start from any source partition has a cloud + tau_h).
        self.partitions = list(self.families)
        if self.julia_hook or self.dive or self.julia_seed_pool_path:
            self.partitions += [ps.julia_partition(f) for f in self.families]
        # PHOENIX IS A PARTITION, NEVER A `--families` ENTRY, and the asymmetry is the
        # engine's, not a convention: `production_seeder.resolve_family` refuses `--phoenix`
        # a parameter plane to prospect ("a single fixed dynamical plane"), so there is no
        # c-plane seeder to draw roots from and no deficit to fold. What phoenix has instead
        # is a SEED POOL — points in (c, p, z_{-1}) parameter space from
        # `phoenix_sampler` — injected as base-scale z-plane roots exactly as
        # `--julia-seed-pool` injects julia c's. Everything downstream (expand, cheap score,
        # tau_h, canonical confirm, decode, reframe, admit, ledger) is the shared path.
        #
        # THE PHOENIX SPLIT (2026-08-04). A pool entry's partition is a function of its
        # parameter point (`partition_for_phoenix_c`), so a pool carrying the pinned Ushiki
        # point produces `phoenix:classic` roots and one carrying swept points produces
        # `phoenix` roots. The tracked set is DERIVED from the pool rather than declared:
        # tracking a partition the pool cannot feed would hand a permanently dry partition a
        # 5% quota floor, and not tracking one the pool DOES feed would leave its nodes
        # keyed on a partition with no cloud, no tau_h and no quota row.
        if self.phoenix_seed_pool_path:
            for part in self.phoenix_pool_partitions():
                if part not in self.partitions:
                    self.partitions.append(part)

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
        self.man_passed_logged: set = set()   # nodes already recorded as quota-passed-over
        self.man_gov = mnv.ProbeGovernor(
            float(getattr(args, "maneuver_probe_p", MAN_PROBE_P_DEFAULT)),
            np.random.default_rng(self.seed + 9901))
        self.man_probe_s = 0.0             # cumulative probe+solve wall time (cost sizing)
        # --- v1.4 richness screen. The cache is keyed on the SHARED atom key and is
        # checkpointed: a resume must not re-spawn the engine for nuclei the killed run
        # already screened. The distribution is the run's OWN accumulating radial_range
        # population — absolute ring scores mean nothing across geometries or cap policies,
        # only orderings within one pair (orbital_field_metrics.md §5, §7). ---
        self.man_screens = msc.ScreenCache()
        self.man_range_dist = msc.RangeDistribution()
        self.man_range_prior = bool(getattr(args, "maneuver_range_prior", False))
        self.man_range_gain = float(getattr(args, "maneuver_range_gain",
                                            MAN_RANGE_GAIN_DEFAULT))
        self.man_nbh = bool(getattr(args, "maneuver_neighborhood", False))
        self.man_nbh_m = int(getattr(args, "maneuver_nbh_m", MAN_NBH_M_DEFAULT))
        self.man_nbh_n = int(getattr(args, "maneuver_nbh_n", MAN_NBH_N_DEFAULT))
        self.man_nbh_probes = int(getattr(args, "maneuver_nbh_probes",
                                          MAN_NBH_PROBES_DEFAULT))
        self.man_screen_s = 0.0            # cumulative screen wall time, priced separately
        # --- v1.5 VIEW screen. Built ONLY when the flag is on: the reference record has to
        # be read and the field store has to be created, and doing either unconditionally
        # would make an OFF run depend on `data/atlas/view_screen_refs.json` existing. The
        # atom screen is skipped entirely while this is on (see MAN_VIEW_GAIN_DEFAULT). ---
        self.man_view_prior = bool(getattr(args, "maneuver_view_prior", False))
        self.man_view_gain = float(getattr(args, "maneuver_view_gain",
                                           MAN_VIEW_GAIN_DEFAULT))
        self.man_views = None
        self.man_fields = None
        self.man_comp_dist = msc.RangeDistribution()   # generic running-percentile tracker
        if self.maneuvers and self.man_view_prior:
            if self.man_range_prior:
                raise SystemExit(
                    "--maneuver-range-prior and --maneuver-view-prior both set. They are "
                    "two sort keys for one seam (the 4x atom radial_range and the view's "
                    "composite_v3) measured on different frames; running both would screen "
                    "twice and let which one selected depend on argument order.")
            self.man_view_params = vscr.screen_params(vscr.load_refs())
            self.man_fields = vfc.RunFieldCache(self.run_dir / "view_fields",
                                                policy=msc.screen_policy_token())
            # harvest v2 §3: the screen records BOTH sourcing scores on every screened row.
            # `composite_v3` remains the live sort key; `view_fit_v1.1` rides beside it as a
            # column only, because its adoption bar (delta-AP >= +0.1181) is pre-registered
            # against a SITTING's labels and has never been read — the q4 sitting recorded
            # neither score, so its NOT-ADOPT was the absence of evidence rather than a
            # measured loss. Load failure is loud: a run that silently dropped the column
            # would reproduce exactly that unreadable state.
            try:
                fit_model = vfit.load_model_v11()
            except Exception as e:                                       # noqa: BLE001
                raise SystemExit(
                    f"--maneuver-view-prior needs the staged view_fit v1.1 record so both "
                    f"sourcing scores can be recorded ({vfit.RECORD_V11_REL}): "
                    f"{type(e).__name__}: {e}")
            self.man_views = mvs.ViewScreenCache(self.man_view_params,
                                                 fields=self.man_fields,
                                                 fit_model=fit_model)
        # Wall-clock cap (0 = off). Distinct from --budget: see the check in run().
        #
        # AND IT DOES NOT REACH DIVE MODE. `run_dive` has its own loop and checks only the
        # active budget and the STOP sentinel — `wall_elapsed_s`, `over_wall_budget` and the
        # batch-boundary check all live on the crawl path. So `--wall-budget --dive` is a flag
        # that silently does nothing, which is worse than an absent feature: the 2026-08-05
        # dive's launch record had to carry a hand-written note saying the flag is a no-op,
        # and a launch record is not a place a guarantee can live. REFUSED rather than warned,
        # because the reason anyone passes it is to bound a run they are not watching, and a
        # warning scrolled past in a background log bounds nothing (`CLAUDE.md`, "a backstop
        # longer than the job's budget is not a backstop"). The feature is deliberately NOT
        # added here — this records the limitation at the flag rather than implementing it.
        check_wall_budget_supported(self.dive, getattr(args, "wall_budget", 0.0))
        self.wall_budget_s = float(getattr(args, "wall_budget", 0.0) or 0.0) * 60.0
        # None => ROOT_DRAW_BUDGET_S (see `root_draw_budget_s`). Minutes on the flag, seconds here.
        _rdb = getattr(args, "root_draw_budget", None)
        self.root_draw_budget_override = float(_rdb) * 60.0 if _rdb else None
        self.wall_s_base = 0.0             # wall seconds spent by PREVIOUS sessions
        self._session_t0 = None

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)

        # Guarded scorer: cheap images (no field sidecar) pass through unguarded == raw
        # scoring; reframe tiles (DUMP_GUARD_FIELD) get the model-free field guard.
        assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
        reframe.DUMP_GUARD_FIELD = True
        self.scorer = guard.make_guarded_scorer(ps.SCORER_PATH)

        # PHOENIX HAS NO DERIVABLE tau_h AND IS NOT GIVEN AN INVENTED ONE. `derive_tau_h`
        # cuts on a specific head's CHEAP p_good, and no such population exists for phoenix
        # under v10: the fidelity records are absent, the vendored base has no phoenix row,
        # and the only phoenix ledgers in the tree (`phoenix_grid`, `classic_phoenix`) are
        # v7-scored and carry no cheap column at all. Deriving from them would be precisely
        # the "a v7 cut on a v10 gate is a number about nothing" failure that function
        # raises to prevent, so phoenix is cut OUT of the derivation and given an explicit,
        # stamped, deliberately CONSERVATIVE value instead.
        #
        # Conservative in a direction that costs recall and not correctness: the default is
        # `floors.GOOD_FLOOR`, i.e. "only pay for a canonical confirmation
        # when the CHEAP score already clears the canonical bar". Fewer confirmations, never
        # wrong ones — and no phoenix material is lost from the deliverable, because the
        # recording floor still keeps everything from 0.25 up in the record-and-rank store.
        #
        # BOTH phoenix partitions are cut out of the derivation, for the same reason and
        # separately: `classic_phoenix` is the v7-scored ledger named above, so it is exactly
        # as underivable as `phoenix_grid` is, and pooling the two to manufacture one number
        # would be a cut on a population that is now two partitions.
        phoenix_parts = [p for p in self.partitions if P.base_partition(p) == "phoenix"]
        derived = [p for p in self.partitions if p not in phoenix_parts]
        self.tau_h = derive_tau_h(derived)
        self.tau_h_uncalibrated = {}
        for part in phoenix_parts:
            v = getattr(args, "tau_h_phoenix", None)
            v = F.GOOD_FLOOR if v is None else float(v)
            self.tau_h[part] = v
            self.tau_h_uncalibrated[part] = (
                f"UNCALIBRATED: no v10 cheap-p_good population exists for {part} "
                f"(fidelity records absent; vendored base has no phoenix row; the phoenix "
                f"ledgers are v7 and carry no cheap column). Serving GOOD_FLOOR={v} as a "
                f"conservative cheap cut.")
        # The RECORDING floor, derived from the harvest cut rather than configured beside it:
        # a second independent constant would drift off tau_h the first time tau_h moved.
        self.tau_rec = derive_tau_rec(self.tau_h)

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
                           # v1.7 cross-run saturation memory. Initialised to 0 rather than
                           # created on first fire, for the reason `root_draw_truncated` is:
                           # "the memory never discounted anything" has to be a positive
                           # statement in the summary, not an absent key.
                           sat_mem_scored=0, sat_mem_discounted=0, sat_mem_density_sum=0,
                           # reconcile terms: every harvest check must land in exactly one
                           # of these buckets, checked per batch (see _reconcile_batch).
                           canon_not_q3=0, reframe_not_q3=0, render_failed=0,
                           frontier_pushed=0,
                           # v1.6 record-and-rank + the sourcing interior gate.
                           # `interior_gated` is a RECONCILE term (see _reconcile_batch):
                           # a gate that removes candidates without entering the identity
                           # is a gate that can silently eat them.
                           interior_gated=0, interior_unmeasured=0,
                           q4_recorded=0, q4_recorded_below_tau_h=0,
                           # Root-draw backstop utilisation. Initialised to 0 rather than
                           # created on first fire, so "the bound never bound" is a positive
                           # statement in every summary instead of an absent key — the same
                           # reason `never_attempted` is reported and not omitted. NOT
                           # reconcile terms (RECONCILE_KEYS is an explicit list): a skipped
                           # family draws no candidates, so no identity has a hole in it.
                           root_draw_truncated=0, root_draw_timeouts=0,
                           # Per-partition refill, same rule: zero is a statement. A run that
                           # reports root_refills=0 with queues at zero is a run whose fix did
                           # not fire, and that has to be visible in the summary rather than
                           # inferred from a missing key.
                           root_refills=0, root_refill_families=0, root_refill_deferred=0,
                           # TRIGGERED yield, disjoint from every fresh counter by
                           # construction. Never summed with the `man_*` block: the
                           # operators feed themselves, so a pooled rate measures the loop.
                           trig_fired=0, trig_atoms=0, trig_nodes_pushed=0,
                           trig_admitted=0, trig_unavailable=0, trig_budget_skip=0,
                           trig_expanded=0, trig_candidates=0,
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
        # CROSS-RUN SATURATION MEMORY, built once and never mutated (see `visited_density`):
        # the current run's coverage is the dup cloud's and the morph memory's job, and a
        # frozen index is what makes a resume rebuild the identical memory. Reuses
        # `prior_rows` when the freshness prior already paid for the read — same files, same
        # exclusion rule, one owner (`vd.iter_prior_ledger_rows`).
        self.sat_build_s = 0.0
        if self.sat_on:
            _t0 = time.time()
            self.sat_index = (vd.VisitedIndex.from_rows(self.prior_rows, self.sat_k)
                              if self.freshness_prior
                              else vd.build_from_ledgers(self.sat_k, ROOT,
                                                         exclude=self.ledger.path))
            if self.freshness_prior:
                self.sat_index.sources = [str(p.relative_to(ROOT)) for p in
                                          vd.ledger_paths(ROOT, self.ledger.path)]
            self.sat_build_s = time.time() - _t0
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

        # --- POP QUOTA (harvest v2; default OFF). The v1 replacement for the cross-partition
        # allocation, and it supersedes `--scheduler` rather than tuning it: the scheduler
        # STEERS the mix (per-batch stochastic argmax on price-weighted deficit) where this
        # ENFORCES it (deterministic pop of whichever servable partition is furthest below its
        # intended share of realized time). `discovery_pipeline.md` §3.1 is the measurement
        # that made the difference matter. The two are mutually exclusive — running both would
        # give two owners of the pop and no readable mix number.
        self.quota = None
        if getattr(args, "pop_quota", False):
            if self.scheduler is not None:
                raise SystemExit("--pop-quota and --scheduler both name the pop; pick one "
                                 "(--pop-quota is the harvest-v2 allocator)")
            # Loud on absence, and `None` means THE DEFAULT ARTIFACT, never the flat seed —
            # see `load_quota_prices`.
            pcfg = load_quota_prices(getattr(args, "quota_prices", None))
            self.quota_prices_path = str(
                Path(getattr(args, "quota_prices", None) or QUOTA_PRICES_DEFAULT))
            # RUN-SCOPED TARGET OVERRIDE. Absent (the default) => the derived release_mix path,
            # byte-identical to every run before this flag existed.
            self.currency_targets_path = getattr(args, "currency_targets", None)
            _tgt = _tsrc = None
            if self.currency_targets_path:
                _tgt, _tsrc = pquota.load_currency_targets(self.currency_targets_path,
                                                           self.partitions)
                print(f"[quota] EXPLICIT currency targets <- {self.currency_targets_path} "
                      f"(release_mix.RATIO not read)", flush=True)
            self.quota = pquota.PopQuota(
                self.partitions, self.run_dir,
                floor=float(getattr(args, "quota_floor", pquota.FLOOR_FRAC)),
                prices_config=pcfg, targets=_tgt, targets_source=_tsrc)

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
        this run's ledger or records. Matches campaign1_readout's prior-ledger enumeration.

        The enumeration itself moved to `visited_density.iter_prior_ledger_rows` when the
        saturation memory became a second consumer of exactly this population under exactly
        this exclusion rule — the same rglob written twice is how the two would eventually
        disagree about which files are "prior"."""
        return list(vd.iter_prior_ledger_rows(ROOT, self.ledger.path))

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
    def root_draw_budget_s(self) -> float:
        """Wall bound for ONE `draw_roots` call, clamped to the remaining wall budget.

        Same shape as `unit_timeout_s`, and here for a sharper reason. The PRE-LOOP draw runs
        before `_session_t0` is set and before `active_s` accrues, so it is outside BOTH
        caps — a slow or hung draw there eats the night while every budget check reports the
        run has not started. Clamped so the backstop can never be longer than what is left of
        the run (the failure `unit_timeout_s` was written for), floored at
        `MIN_ROOT_DRAW_S` so a legitimately slow draw near the end is not shot for being slow.
        No wall budget => the standing constant, which is still a bound where there was none.

        `--root-draw-budget` overrides the constant. It exists because 40 min is sized for an
        overnight run and is NOT a bound at all for a run whose total commitment is a few
        hours: a contended pre-loop draw can spend 40 min outside both caps and there is no
        flag-reachable way to say otherwise. Default `None` => `ROOT_DRAW_BUDGET_S`, i.e. every
        run that does not pass it is byte-identical."""
        cap = float(getattr(self, "root_draw_budget_override", None) or ROOT_DRAW_BUDGET_S)
        if not self.wall_budget_s:
            return cap
        remaining = max(0.0, self.wall_budget_s - self.wall_elapsed_s())
        return float(min(cap, max(MIN_ROOT_DRAW_S, remaining)))

    def draw_roots(self, only=None):
        """Draw a batch of native depth-1 seeds per family (q3-density rejection +
        depth-2 descendability probe) and enter the survivors as depth-1 frontier nodes
        with a neutral prior priority — exactly the current path's root pipeline.

        `only` restricts the draw to a subset of families (the per-partition refill); None
        draws every family, which is the unchanged global path. The scheduler's root
        allocation is recomputed OVER THE SUBSET rather than sliced out of the full one — a
        targeted refill that handed a starved family its proportional share of B would draw
        the starved family a rounding error, which is the failure it exists to fix.

        BOUNDED, at two granularities, because one alone does not cover the failure. The
        per-probe `timeout` bounds a HUNG engine subprocess (that call had no backstop at
        all); the per-call deadline bounds a merely SLOW draw, which no single timeout
        catches — nine families each finishing just inside their own timeout is still hours.
        A truncated draw is REPORTED and counted, never silent: a short draw and a fast draw
        produce the same frontier length, and only one of them is a problem."""
        fams = list(self.families) if only is None else \
            [f for f in self.families if f in set(only)]
        if not fams:
            return 0
        added = 0
        t0 = time.time()
        budget = self.root_draw_budget_s()
        skipped = []
        # item 7: deficit-aware root mix. Scheduler ON => split the B draws across families by
        # their price-weighted, julia-twin-inclusive deficit; OFF => B per family (unchanged).
        alloc = (self.scheduler.root_allocation(fams, self.B, self.rng)
                 if self.scheduler is not None else None)
        for fam in fams:
            nb = alloc[fam] if alloc is not None else self.B
            if alloc is None and self.family_weights:
                # v1.6 DEFICIT-BY-q4-GAP allocation. Total draws are preserved (B per family
                # summed = B*F), only their DISTRIBUTION moves, so this changes the mix
                # without changing the run's root budget. Floored at 1 so a channel with a
                # small weight still gets touched every replenishment — a weight is a
                # preference, not a mute switch, and a partition that never draws a root
                # cannot report a zero yield distinguishable from "never tried".
                nb = max(1, int(round(self.B * len(self.families)
                                      * self.family_weights.get(fam, 0.0))))
            if nb <= 0:
                continue
            spent = time.time() - t0
            if spent >= budget:
                skipped.append(fam)
                continue
            # run-only cloud: the freshness prior must NOT feed this hard rejection gate (part-0
            # sterilization finding) — only this run's own accruing q3 places spread new seeds.
            cloud = self.run_clouds[fam]
            props = self.seeders[fam].draw_batch(cloud, nb)
            if not props:
                continue
            pw = self.scratch / f"roots_b{self.batch_i:04d}_{fam}"
            try:
                # The per-family probe gets what is left of the draw's own budget, so one
                # family cannot spend the whole allowance and leave the rest unbounded.
                survivors, rejects, _ = ps.depth2_probe(
                    props, pw, self.seed, self._flags(fam),
                    timeout=max(MIN_ROOT_DRAW_S, budget - spent))
            except TimeoutError as e:
                # A dead probe costs this family's roots, not the run. Counted and named:
                # a family that silently contributes no roots is indistinguishable from one
                # that was never tried, which is the reading `draw_roots` must not allow.
                self.totals["root_draw_timeouts"] = \
                    self.totals.get("root_draw_timeouts", 0) + 1
                skipped.append(f"{fam}(timeout)")
                print(f"[root-draw] {fam} depth-2 probe TIMED OUT — skipping this family's "
                      f"roots this draw: {e}", flush=True)
                continue
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
        if skipped:
            self.totals["root_draw_truncated"] = \
                self.totals.get("root_draw_truncated", 0) + 1
            print(f"[root-draw] BOUND HIT after {time.time()-t0:.0f}s of a {budget:.0f}s "
                  f"budget — {added} roots added, families not drawn: {skipped}. "
                  f"(wall {self.wall_elapsed_s()/3600:.2f}h of "
                  f"{self.wall_budget_s/3600:.2f}h)", flush=True)
        return added

    # ------------------------------------------------- per-partition refill
    def queue_lens(self) -> dict:
        """Frontier node count per partition — the servability the pop reads, and the
        quantity a low-water mark has to be measured in."""
        q: dict = {}
        for n in self.frontier:
            q[n["partition"]] = q.get(n["partition"], 0) + 1
        return q

    def starved_families(self, queue_lens: dict | None = None) -> list[str]:
        """c-plane families below the per-partition low-water and off cooldown.

        c-plane ONLY, and that is a scope statement rather than an omission: `draw_roots` is
        the native seeder, and a julia twin or phoenix CANNOT be drawn into existence — those
        partitions are fed by the metered pools (`seed_julia_pool` / `seed_phoenix_pool`, per
        batch) and by the hook. A julia queue at zero is a hook/pool problem, and listing it
        here would produce a refill request no draw can serve.

        `phoenix:classic` is the strongest instance and is deferred for a reason of its own,
        recorded so it reads as a decision rather than an oversight: its plane is PINNED, so
        the parameter space holds exactly one point and there is no second root to draw. New
        classic supply comes from descending that one plane (`supply_routing.ROUTES`'s
        `classic_plane_descent`, i.e. `production_seeder --run-phoenix`), which is a
        different job from a root draw. A refill request here could never be served, and
        `starved_families` returning it would spend the cooldown and the wall budget proving
        that every ROOT_REFILL_COOLDOWN batches. The starvation is REPORTED instead — see
        `deferred_partitions`."""
        q = self.queue_lens() if queue_lens is None else queue_lens
        out = []
        for fam in self.families:
            if q.get(fam, 0) >= self.partition_low_water:
                continue
            last = self.last_refill_batch.get(fam)
            if last is not None and (self.batch_i - last) < self.root_refill_cooldown:
                continue
            out.append(fam)
        return out

    # Why a non-c-plane partition below its low-water gets no refill. Keyed by the partition's
    # BASE so a julia twin and its parent share one entry, plus the derived partitions that
    # need a reason of their own.
    REFILL_DEFERRAL = {
        "julia": "fed by --julia-hook / --julia-seed-pool, not by a root draw",
        "phoenix": "fed by --phoenix-seed-pool (sampled parameter points), not by a root draw",
        "phoenix:classic": ("the plane is PINNED: one parameter point, so no second root "
                            "exists to draw. New supply is classic_plane_descent "
                            "(production_seeder --run-phoenix), not a root draw"),
    }

    def deferred_partitions(self, queue_lens: dict | None = None) -> dict:
        """Partitions below the low-water that `starved_families` deliberately will not
        refill, each with the reason. A starved partition that is silently absent from both
        the refill list and the run record is indistinguishable from a healthy one — which is
        exactly how arm B ran 427 batches with eight empty queues and `root_refills=0`.

        SKIP SITE 1 of 3 (the crawl CENSUS). An EXTERNALLY-SUPPLIED partition
        (`supply_routing.is_externally_supplied`) is not reported here at all. It is not
        starved and it is not deferred: no crawl channel feeds it, so an empty queue is its
        NORMAL state, and calling it starved every batch is a permanent false alarm that
        trains the reader to ignore this dict. That is a real cost — this dict exists to make
        eight silently empty queues loud, and a row that is always red does the opposite.

        WHERE THE VISIBILITY WENT, because "skipped" must not mean "invisible": the run
        summary stamps `externally_supplied` (via `pop_quota.PopQuota.summary`), and the count
        that can actually be acted on is printed at EMISSION INTAKE, where the servable classic
        population is known and the manual job to run can be named
        (`ledger_rescore.classic_supply_note`)."""
        q = self.queue_lens() if queue_lens is None else queue_lens
        out = {}
        for part in self.partitions:
            if part in self.families or q.get(part, 0) >= self.partition_low_water:
                continue
            if srt.is_externally_supplied(part):
                continue
            base = P.base_partition(part)
            key = (part if part in self.REFILL_DEFERRAL else
                   ("julia" if base.startswith("julia:") or base == "julia" else base))
            out[part] = dict(queue=q.get(part, 0), low_water=self.partition_low_water,
                             reason=self.REFILL_DEFERRAL.get(
                                 key, "no root-draw channel for this partition"))
        return out

    def refill_affordable(self) -> bool:
        """Have in-loop root draws stayed inside their share of the loop's wall clock?

        Expressed against total loop wall (`active + root-draw`) rather than against
        `active_s` alone so it is well-defined at zero: the first refill of a run always
        clears it, and a run that has spent nothing but drawing roots always fails it. This
        is the bound that keeps a family whose depth-2 probe rejects everything from being
        re-drawn forever at ~2 min a time — the cooldown spaces the retries, this caps their
        total. Root-draw seconds sit OUTSIDE `active_s` by construction (the active timer
        wraps the batch block only), so the two do not double-count."""
        total = self.active_s + self.root_draw_s
        return self.root_draw_s <= self.root_refill_share * total

    def refill_starved(self) -> int:
        """Draw fresh roots for the starved c-plane families only. Returns roots added.

        Cost is charged to `root_draw_s` whether or not the draw produced anything: a draw
        that survives nothing still spent the wall clock, and an affordability bound fed only
        by successful draws is a bound that loosens exactly when the draws stop working."""
        starved = self.starved_families()
        if not starved:
            return 0
        if not self.refill_affordable():
            self.totals["root_refill_deferred"] = \
                self.totals.get("root_refill_deferred", 0) + 1
            return 0
        t0 = time.time()
        added = self.draw_roots(only=starved)
        self.root_draw_s += time.time() - t0
        self.stage_times.record("root_refill", f"refill:{self.batch_i}", time.time() - t0,
                                families=sorted(starved), roots_added=added)
        for fam in starved:
            self.last_refill_batch[fam] = self.batch_i
        self.totals["root_refills"] = self.totals.get("root_refills", 0) + 1
        self.totals["root_refill_families"] = \
            self.totals.get("root_refill_families", 0) + len(starved)
        print(f"[root-refill] b{self.batch_i}: {starved} below the per-partition low-water "
              f"({self.partition_low_water}) — drew {added} roots in "
              f"{time.time()-t0:.0f}s (root-draw {self.root_draw_s/60:.1f}m of "
              f"{(self.active_s + self.root_draw_s)/60:.1f}m loop wall)", flush=True)
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

    # ------------------------------------------------------------ seed pools
    # METERED INJECTION, and it is the fix for a measured failure rather than a refinement.
    # Both z-plane pools used to be injected WHOLESALE at fresh start: 534 julia + 96 phoenix
    # roots landing on the frontier at once. The frontier is popped by GLOBAL PRIORITY and
    # every root enters at NEUTRAL_PRIOR + gumbel, so 630 injected roots simply outnumber the
    # ~128 native roots a replenishment draws, and they keep outnumbering them for the rest of
    # the run. Measured 14 batches into the first launch: 1,500 julia and 314 phoenix
    # candidates against 44 / 26 / 19 for multibrot3/4/5 — 5% of the candidate stream going to
    # the partitions holding 70% of the allocation and the largest q4 gaps.
    #
    # `--family-weights` cannot reach this: it sizes c-plane ROOT DRAWS, and the imbalance is
    # in the injected pools, which are not draws. So the pools are metered — `rate` entries
    # per replenishment, from a persisted cursor — and injection moves from "once at start"
    # to "whenever roots are replenished". That also gives the operating rule
    # `julia_c_sourcing.md` states outright: run the sampler TO THE KNEE, then refill. A pool
    # consumed as the walk asks for roots is a pool run to its knee by construction, where a
    # pool dumped at t=0 is a pool run straight into its tail.
    #
    # `rate = 0` restores wholesale injection and is byte-identical to every run before this.
    def _take_from_pool(self, pool: list, cursor_key: str) -> list:
        i = int(self.pool_cursor.get(cursor_key, 0))
        if i >= len(pool):
            return []
        n = len(pool) - i if self.seed_pool_rate <= 0 else self.seed_pool_rate
        chunk = pool[i:i + n]
        self.pool_cursor[cursor_key] = i + len(chunk)
        return chunk

    def load_julia_supply_pool(self) -> list:
        """The julia c-supply pool, VERIFIED against `supply_routing.CSPACING_FLOOR`.

        Read once and cached: this is called per batch, and re-reading is not the cost — a
        per-batch verification that could disagree with the first one is.

        REFUSES rather than thins. Thinning would be the friendlier failure and it is the
        wrong one twice over: `pool_cursor` is persisted in state.json and indexes into this
        list, so silently returning a shorter list makes a resume land somewhere else in the
        supply; and a pool is a config input whose length is quoted in the run record, so
        quietly replacing 539 c with 210 would put a number in that record nobody chose. The
        pool that clears the floor already exists — this names it."""
        if getattr(self, "_julia_pool_cache", None) is not None:
            return self._julia_pool_cache
        pool = json.loads(self.julia_seed_pool_path.read_text(encoding="utf-8"))
        floor = srt.CSPACING_FLOOR
        pts = [(float(r["c_re"]), float(r["c_im"])) for r in pool]
        worst, wi, wj = float("inf"), -1, -1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if d < worst:
                    worst, wi, wj = d, i, j
        # A hair of tolerance: a pool thinned AT the floor stores rounded decimals, so its
        # closest surviving pair can sit a few ulp under the float it was thinned against.
        if len(pts) > 1 and worst < floor * (1 - 1e-6):
            raise SystemExit(
                f"--julia-seed-pool {self.julia_seed_pool_path} does NOT clear the adopted "
                f"c-spacing floor.\n"
                f"    closest pair : |dc| = {worst:.6g}  (rows {wi} and {wj} of {len(pts)})\n"
                f"    floor        : {floor:.6g}  (supply_routing.CSPACING_FLOOR)\n"
                f"This pool was thinned against a superseded floor, and NOTHING downstream "
                f"would have caught it: seed_julia_pool bypasses the hook-spacing gate, so "
                f"the c-spacing a run applies is whatever this file carries. The 1e-2 floor "
                f"it probably came from was an artifact of pairs rendered at their own "
                f"viewports (supply_routing.CSPACING_BASIS['supersedes']).\n"
                f"Use {JULIA_SUPPLY_POOL.name}, or rebuild with "
                f"`uv run python tools/atlas/build_julia_supply_pool_v2.py`.")
        self._julia_pool_cache = pool
        self._julia_pool_min_dc = worst if len(pts) > 1 else None
        print(f"[julia-seed-pool] {self.julia_seed_pool_path.name}: {len(pool)} c, "
              f"closest pair |dc|={worst:.4g} >= floor {floor:.4g}", flush=True)
        return pool

    def seed_julia_pool(self) -> int:
        """PRIMARY julia supply under test (julia_parent_sourcing_probe). Inject the c-diverse
        near-∂M sampler pool as julia:mandelbrot base-scale z-plane roots at fresh start.

        Deliberately BYPASSES add_julia_root's hook-spacing gate. That predates the spacing
        reconciliation (the gate was 6.25x coarser than the pool and would have collapsed it
        to a handful) and it stays for a second reason the reconciliation does not touch:
        `pool_cursor` is persisted and indexes into this list, so a gate that dropped entries
        would make the cursor mean something different on resume. It is now a no-op in
        substance — the pool is thinned AT `CSPACING_FLOOR` and the gate is `CSPACING_FLOOR`,
        so every entry clears it up to float equality. `load_julia_supply_pool` verifies the
        file against the floor regardless: with the gate skipped, the file IS the floor in the
        live path. Each injected c is registered in `hooked_c` so a later
        parent-fired hook whose seed c lands within spacing of a sampler c is suppressed — the
        hook stays available (§1 secondary path) but does not re-cover the sampler's ground.
        The pool is degree-2 near-∂M (z²+c), so every root is the julia:mandelbrot twin."""
        if self.julia_seed_pool_path is None:
            return 0
        jpart = ps.julia_partition("mandelbrot")
        if jpart not in self.partitions:
            if self.julia_pool_explicit:
                raise SystemExit(
                    f"--julia-seed-pool needs 'mandelbrot' in --families (for {jpart})")
            return 0        # inherited default on a run with no julia:mandelbrot — not an error
        pool = self.load_julia_supply_pool()
        chunk = self._take_from_pool(pool, "julia")
        if not chunk:
            return 0
        added = 0
        for e in chunk:
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
        chatty = (self.seed_pool_rate <= 0
                  or self.pool_cursor["julia"] >= len(pool)
                  or self.batch_i % 25 == 0)
        if chatty:
            print(f"[julia-seed-pool] injected {added} sampler-sourced {jpart} roots "
                  f"(fw={JULIA_ROOT_FW}) from {self.julia_seed_pool_path.name}; "
                  f"{self.pool_cursor['julia']}/{len(pool)} of the pool consumed", flush=True)
        return added

    def phoenix_pool_partitions(self) -> list[str]:
        """The phoenix partitions this run's seed pool can actually feed, in `ALL_FAMS` order.

        Read from the pool ONCE at construction (the pool is a static file), and derived from
        each entry's parameter point by the same function `seed_phoenix_pool` labels nodes
        with — so the tracked set and the produced nodes cannot disagree. A missing or
        unreadable pool is not silently an empty set: the caller already gated on the path
        being set, so a read failure here is a config error and must raise."""
        pool = json.loads(self.phoenix_seed_pool_path.read_text(encoding="utf-8"))
        seen = set()
        for e in pool:
            seen.add(partition_for_phoenix_c(
                (str(e["c_re"]), str(e["c_im"]), str(e["p_re"]), str(e["p_im"]),
                 str(e.get("zm1_re", 0.0)), str(e.get("zm1_im", 0.0)))))
        return [p for p in P.ALL_FAMS if p in seen]

    def seed_phoenix_pool(self) -> int:
        """Inject `phoenix_sampler` seeds as base-scale phoenix z-plane roots at fresh start.

        The phoenix analogue of `seed_julia_pool`, and it exists for the same reason: a
        z-plane partition cannot be popped into existence, it has to be given roots. Each
        pool entry is a point in phoenix PARAMETER space `(c, p, z_{-1})` drawn near the
        closed-form neutral-stability skeleton (`phoenix_seed_sampler_spec.md` §2) — which is
        the phoenix replacement for "sample near ∂M", since an invertible Hénon map has no
        critical point and so no honest connectedness locus to sit on the boundary of.

        The 6-vector goes in the node's `c` slot so `expand_batch`'s group-by
        `(partition, tuple(c))` keeps each `--expand` call homogeneous in kernel, which it
        must be: p and z_{-1} are engine flags, not per-node data.
        """
        if self.phoenix_seed_pool_path is None:
            return 0
        pool = json.loads(self.phoenix_seed_pool_path.read_text(encoding="utf-8"))
        chunk = self._take_from_pool(pool, "phoenix")
        if not chunk:
            return 0
        added = 0
        for e in chunk:
            c6 = (str(e["c_re"]), str(e["c_im"]), str(e["p_re"]), str(e["p_im"]),
                  str(e.get("zm1_re", 0.0)), str(e.get("zm1_im", 0.0)))
            nid = self.new_node_id()
            self.frontier.append(dict(
                node_id=nid, root_id=nid, partition=partition_for_phoenix_c(c6), c=list(c6),
                cx=0.0, cy=0.0, fw=JULIA_ROOT_FW, depth=1,
                priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
                cheap_eord=None, cheap_pgood=None, branch="phoenix_root",
                mix_source=f"phoenix_sampler:{e.get('branch', '?')}",
                phoenix=dict(branch=e.get("branch"), theta=e.get("theta"),
                             offset=e.get("offset"), abs_p=e.get("abs_p")),
            ))
            self.totals["phoenix_roots"] = self.totals.get("phoenix_roots", 0) + 1
            added += 1
        chatty = (self.seed_pool_rate <= 0
                  or self.pool_cursor["phoenix"] >= len(pool)
                  or self.batch_i % 25 == 0)
        if chatty:
            print(f"[phoenix-seed-pool] injected {added} sampler-sourced phoenix roots "
                  f"(fw={JULIA_ROOT_FW}) from {self.phoenix_seed_pool_path.name}; "
                  f"{self.pool_cursor['phoenix']}/{len(pool)} of the pool consumed", flush=True)
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
        parent solves. Every probe decision is logged to `maneuvers.jsonl`.

        THREE PHASES, and the split is what makes the screen affordable. (1) ENUMERATE:
        every operator runs, pure mpmath, no process spawned. (2) SCREEN: every DISTINCT
        available nucleus in the whole batch is measured in one concurrent pass — the
        screen's cost is process spawn, not compute, so batching hides it, and a per-row
        screen would have paid it once per k. (3) CONSUME: rows are recorded and pushed in
        enumeration order, so the RNG stream and the push order are unchanged from v1.3.
        """
        if not self.maneuvers:
            return 0
        produced: list[dict] = []          # {m, parent, nbh_group} in enumeration order
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
                produced.append(dict(m=m, parent=n, nbh_group=None))
                if m.available and parent_rec is None:
                    parent_rec = dict(id=m.atom_id, cx=m.cx, cy=m.cy, period=m.period,
                                      window_scale=m.window_scale, degree=degree)
            if self.man_lateral:
                m = mnv.lateral_to_sibling(view, self.rng, degree=degree,
                                           parent_rec=parent_rec)
                produced.append(dict(m=m, parent=n, nbh_group=None))
            if self.man_nbh:
                for m in mnv.neighborhood_expand(
                        view, self.rng, self.man_ks, degree=degree, parent_rec=parent_rec,
                        max_found=self.man_nbh_m, max_probes=self.man_nbh_probes):
                    produced.append(dict(m=m, parent=n, nbh_group=n["node_id"]))
            self.man_probe_s += time.time() - t0

        scores = self._screen_produced(produced)
        keep_nbh = self._nbh_top_n(produced, scores)

        pushed = 0
        for item in produced:
            m = item["m"]
            drop = None
            if (item["nbh_group"] is not None and m.available
                    and m.atom_key not in keep_nbh.get(item["nbh_group"], ())):
                drop = "nbh_not_top_n"
            pushed += self._consume_maneuver(m, item["parent"],
                                             scores.get(self._screen_key(m)),
                                             passed_over=drop)
        # mirror the governor's counters into totals (the checkpointed, resumable copy)
        g = self.man_gov
        self.totals["man_probes_rolled"] = g.n_rolled
        self.totals["man_probes_fired"] = g.n_fired
        self.totals["man_probes_coin_skip"] = g.n_coin_skip
        self.totals["man_probes_cache_skip"] = g.n_cache_skip
        self.totals["man_screen_cache_hits"] = self.man_screens.n_hits
        return pushed

    def _screen_key(self, m) -> str:
        """The key one screen record is filed under: the VIEW under v1.5, else the ATOM.

        One function so the enumeration, the lookup and the cache can never disagree about
        what a screened thing is — the bug that shape invites is a screen filed per view and
        read back per atom, which silently hands one framing's numbers to another's."""
        return (mvs.view_key(m.atom_key, m.k)
                if getattr(self, "man_view_prior", False) else m.atom_key)

    def _screen_produced(self, produced: list[dict]) -> dict:
        """Screen every DISTINCT available candidate in this batch's enumeration, once.

        RECORDING, NEVER A GATE (the prompt's phrasing, and the reason this returns a dict
        rather than filtering `produced`): a candidate whose screen fails is still a
        candidate. The run's own score distribution is fed here — from the screened
        candidates only, which is the population the percentile is later taken against.

        WHICH screen is the v1.5 branch. Under `--maneuver-view-prior` the unit is the VIEW
        (`atom|k`) at the frame that is actually pushed, and the atom screen does not run at
        all; otherwise it is the ATOM at its 4x frame, exactly as v1.4 did."""
        if not produced:
            return {}
        if getattr(self, "man_view_prior", False):
            return self._screen_views(produced)
        jobs = []
        for item in produced:
            m = item["m"]
            if m.available and m.atom_key and m.window_scale:
                jobs.append(dict(atom_key=m.atom_key, cx=m.cx, cy=m.cy,
                                 window_scale=m.window_scale,
                                 family=item["parent"]["partition"]))
        if not jobs:
            return {}
        t0 = time.time()
        before = set(self.man_screens.by_key)
        # Bounded by the SAME budget-clamped backstop the expand call uses, so the screen
        # pass can never be the unbounded thing inside a batch the between-batch cap cannot
        # see. `unit_timeout_s` is already `min(900s, remaining budget)` floored at 60s.
        scores = self.man_screens.screen_many(jobs, budget_s=self.unit_timeout_s())
        self.man_screen_s += time.time() - t0
        for key, rec in scores.items():
            if key in before:                   # cache hit: already counted and already
                continue                        # in the distribution
            if rec.get("screened"):
                self.totals["man_screened"] += 1
                self.man_range_dist.add(rec.get("radial_range"))
            else:
                self.totals["man_unscreenable"] += 1
        return scores

    def _screen_views(self, produced: list[dict]) -> dict:
        """The v1.5 half of `_screen_produced`: one field per DISTINCT VIEW, at its own frame.

        A `k` is no longer free here and that is the point — under the atom screen one field
        served every framing of a nucleus, because the 4x frame does not depend on `k`; a
        view screen's frame IS `k`, so three framings are three fields. The cost was priced
        against the exploration run before the k-set was fixed at three
        (`maneuver_view_screen.py`, "THE COST").
        """
        jobs, seen = [], set()
        for item in produced:
            m = item["m"]
            if not (m.available and m.atom_key and m.fw):
                continue
            key = mvs.view_key(m.atom_key, m.k)
            if key in seen:
                continue
            seen.add(key)
            # `window_scale` rides the job because `view_fit_v1.1` needs `log10_size_rel`
            # (the nucleus size relative to the frame) and the screen record does not carry
            # it — the screen measures the FIELD, and the atom's size is the operator's fact.
            jobs.append(dict(view_key=key, cx=m.cx, cy=m.cy, fw=m.fw, atom_key=m.atom_key,
                             k=m.k, family=item["parent"]["partition"],
                             window_scale=m.window_scale))
        if not jobs:
            return {}
        t0 = time.time()
        before = set(self.man_views.by_key)
        scores = self.man_views.screen_many(jobs, budget_s=self.unit_timeout_s())
        self.man_screen_s += time.time() - t0
        for key, rec in scores.items():
            if key in before:                   # cache hit: already counted, already ranked
                continue
            if rec.get("screened"):
                self.totals["man_view_screened"] += 1
                self.man_comp_dist.add(rec.get("composite"))
                if rec.get("vetoed"):
                    self.totals["man_view_vetoed"] += 1
            else:
                self.totals["man_view_unscreenable"] += 1
        self.totals["man_screen_cache_hits"] = self.man_views.n_hits
        self.totals["man_view_fields_cached"] = self.man_views.n_fields_cached
        return scores

    def _nbh_top_n(self, produced: list[dict], scores: dict) -> dict:
        """Per parent node, the `n` neighbourhood atom keys to propose, by the live screen.

        Selection is by DISTINCT ATOM, not by row: a nucleus emits one row per `k`, and
        ranking rows would spend the whole budget on the k-set of a single nucleus. Under
        v1.5 the sort key is the atom's BEST view — `max composite_v3` over its framings —
        which is the right reduction for the question this selection asks ("is there a good
        picture here?") and not the same as the mean, which would let two weak framings
        outvote one strong one. Under v1.4 it is the atom's `radial_range`, one number for
        the nucleus, which is why the reduction did not arise there.

        Unscreenable atoms sort LAST but are not excluded — the screen is a ranker here,
        not a gate, so an atom the 64 px geometry cannot reach still gets a slot if one is
        left over. Enumeration order breaks ties, which is what keeps a run with a totally
        unreachable neighbourhood behaving like the plain operator."""
        groups: dict = {}
        for i, item in enumerate(produced):
            m = item["m"]
            if item["nbh_group"] is None or not m.available:
                continue
            g = groups.setdefault(item["nbh_group"], {})
            rec = scores.get(self._screen_key(m)) or {}
            if getattr(self, "man_view_prior", False):
                cand = (1 if rec.get("screened") else 0,
                        float(rec.get("composite") if rec.get("composite") is not None
                              else -1e18), -i)
                prev = g.get(m.atom_key)
                # max over the atom's framings, and the tie-break stays the FIRST row's
                # enumeration index so the ordering does not depend on which k won.
                if prev is None or cand[:2] > prev[:2]:
                    g[m.atom_key] = cand[:2] + (prev[2] if prev is not None else cand[2],)
                continue
            if m.atom_key not in g:
                g[m.atom_key] = (1 if rec.get("screened") else 0,
                                 float(rec.get("radial_range") or 0.0), -i)
        keep = {}
        for gid, cands in groups.items():
            order = sorted(cands, key=lambda k: cands[k], reverse=True)
            keep[gid] = set(order[:max(0, self.man_nbh_n)])
            self.totals["man_nbh_passed_over"] += max(0, len(order) - self.man_nbh_n)
        return keep

    def _consume_maneuver(self, m, parent, screen=None, passed_over=None) -> int:
        """Record a maneuver outcome and, when it is available AND new, push its node.

        `screen` is the richness record for this nucleus (may be None / unscreened). It is
        written onto the row and onto `man` in EVERY case, including the rows that are not
        pushed — the prompt's "every candidate keeps its scores, including candidates not
        selected". `passed_over` names a candidate that lost a bounded selection (today:
        the neighbourhood top-n) rather than being unavailable or already visited."""
        row = m.as_row()
        row.update(batch=self.batch_i, partition=parent["partition"],
                   root_id=parent["root_id"])
        if screen:
            row["screen"] = screen
        if not m.available:
            self.totals["man_op_unavailable"] += 1
            row["used"] = False
            self._log_maneuver(row)
            return 0
        self.totals["man_op_available"] += 1
        if passed_over:
            row["used"] = False
            row["unused_reason"] = passed_over
            row["passed_over"] = True
            self._log_maneuver(row)
            return 0
        # Multiple frontier members snapping to ONE nucleus is the normal case; the
        # read-time canonical key (snap_near_zero + sector-canonical rounding) is what
        # collapses them. Framing (k) is part of the identity — the same atom at two k's
        # is two distinct views. The OPERATOR is not: with three operators live, lateral
        # and neighborhood routinely reach the same sibling, and keying on op as well would
        # push that one view twice under two provenance labels. Identity is (atom, k).
        vkey = f"{m.atom_key}|{m.k}"
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
        # The richness scores ride the NODE (and therefore state.json, the harvest log and
        # the ledger), not only the probe log — a readout that has to re-derive a score
        # from coordinates is a readout that will re-derive it under a different cap.
        if screen:
            man["radial_range"] = screen.get("radial_range")
            man["radial_rings"] = screen.get("radial_rings")
            man["screened"] = bool(screen.get("screened"))
            man["screen_policy"] = screen.get("maxiter_policy_token")
            # WHICH FRAME the two ring measures above were taken on. It is on the node and
            # not only in the run config because a readout that joins two runs' rows would
            # otherwise pool a 4x-atom `radial_range` with a view-frame one and report the
            # mixture as one distribution. Same measure, different frames, never summed.
            view_prior = getattr(self, "man_view_prior", False)
            man["screen_frame"] = "view" if view_prior else "atom4x"
            if view_prior:
                for key in ("composite", "vetoed", "size_factor", "band_coverage",
                            "band_coverage_q25", "interior_fraction", "view_fw",
                            # harvest v2 §3: BOTH sourcing scores ride the NODE, so they
                            # reach the harvest record and the ledger and are on a labelled
                            # row when the pre-registered bar is finally read. A score that
                            # lives only in `maneuvers.jsonl` is a score no sitting can see.
                            "view_fit", "view_fit_p_notbad", "view_fit_model",
                            "view_fit_reason"):
                    man[key] = screen.get(key)
        prior = NEUTRAL_PRIOR
        if self.man_range_prior:
            pct = self.man_range_dist.percentile_of(
                (screen or {}).get("radial_range") if (screen or {}).get("screened") else None)
            prior += msc.range_prior_delta(pct, self.man_range_gain)
            man["range_pct"] = round(pct, 4)
        elif getattr(self, "man_view_prior", False):
            # Against the run's OWN accumulating composite distribution, for the reason the
            # range prior was: an absolute field number means nothing across geometries and
            # cap policies, only an ordering within one pair does. An unscreened row draws
            # the neutral 0.5, i.e. exactly no delta — not a penalty. A VETOED row does not:
            # it has a composite in [-1, 0) and takes the percentile that score earns it,
            # which is near the bottom of the run's population and is the intended demotion.
            comp = (screen or {}).get("composite") if (screen or {}).get("screened") else None
            pct = self.man_comp_dist.percentile_of(comp)
            prior += msc.range_prior_delta(pct, self.man_view_gain)
            man["composite_pct"] = round(pct, 4)
        self.frontier.append(dict(
            node_id=nid, root_id=parent["root_id"], partition=parent["partition"],
            c=parent["c"], cx=float(m.cx), cy=float(m.cy), fw=float(m.fw),
            depth=int(m.depth), branch="maneuver",
            priority=prior + gumbel(self.rng, T_GUMBEL) + self.beta * int(m.depth),
            cheap_eord=None, cheap_pgood=None,
            mix_source=f"maneuver:{m.op}:k={m.k}", man=man,
        ))
        self.totals["man_nodes_pushed"] += 1
        row["used"] = True
        row["node_id_pushed"] = nid
        self._log_maneuver(row)
        return 1

    def _log_maneuver(self, row: dict):
        self._writer(self.man_log).write_row(row, default=str)

    def _split_reserved(self, pool: list[dict]) -> tuple[list[dict], list[dict]]:
        """Take the batch out of a PRIORITY-SORTED `pool`, honouring the maneuver floor.

        The floor is a reserved count of SLOTS, not a probability and not a priority bonus:
        the walker already ranks a slate, so a new proposal source needs a slot. It is a
        quota **of available** — with ~17% Newton convergence the operator is often simply
        not there, and an unfillable quota must never stall the frontier, so whatever is not
        filled falls straight back to the ordinary priority order.

        `pref_loc_v1` (the preference ranker) is ABSENT from this seam, as it is from
        `pop_batch_scheduled`: reserving a slot is not a ranker change, and the
        ranks-never-steers boundary is untouched by it.

        WHICH available maneuver fills a slot is the v1.4 change, and only behind
        `--maneuver-range-prior`: with the flag on the reserved slots are filled by
        descending `radial_range` instead of by the incoming priority order. That is still
        not a ranker — no aesthetic score enters — it is a field/geometry measure choosing
        among candidates that already hold the slots. Flag off, the order is v1.3's."""
        if not self.maneuvers or self.man_quota <= 0:
            return pool[:self.B], pool[self.B:]
        plain = pool[:self.B]
        plain_ids = {n["node_id"] for n in plain}
        man = [n for n in pool if n.get("man")]
        screen_sorted = getattr(self, "man_range_prior", False) or \
            getattr(self, "man_view_prior", False)
        if screen_sorted and man:
            # Unscreened sorts last (0 in the first key), never excluded — the screen ranks
            # the quota, it does not gate it. Ties keep the incoming priority order.
            key = (mvs.composite_sort_key if getattr(self, "man_view_prior", False)
                   else (lambda d: (1 if d.get("screened") else 0,
                                    float(d.get("radial_range") or 0.0))))
            man = sorted(man, key=lambda n: key(n["man"]), reverse=True)
        take = min(self.man_quota, len(man), self.B)
        self.totals["man_quota_unfilled"] += self.man_quota - take
        # ONCE PER NODE, not once per node per batch. A maneuver node that loses a quota
        # slot stays on the frontier and loses again next batch, so a per-batch record is a
        # BACKLOG-PRESSURE reading wearing a count's name — and it writes O(nodes x batches)
        # rows. Measured: 10,176 rows over 24 shakedown batches, 7.6 MB, and the maneuver
        # frontier was still climbing toward its share cap; a 7-hour run would have written
        # hundreds of MB of the same nodes restated. The count is of DISTINCT candidates
        # passed over at least once; live backlog is `len(man)`, which is derivable.
        newly = [n for n in man[take:] if n["node_id"] not in self.man_passed_logged]
        self.totals["man_quota_passed_over"] += len(newly)
        if screen_sorted:
            for n in newly:
                self.man_passed_logged.add(n["node_id"])
                self._log_maneuver(dict(batch=self.batch_i, op=n["man"].get("op"),
                                        k=n["man"].get("k"), node_id=n["node_id"],
                                        atom_key=n["man"].get("atom_key"),
                                        partition=n["partition"], used=False,
                                        passed_over=True, unused_reason="quota_passed_over",
                                        priority=n["priority"],
                                        radial_range=n["man"].get("radial_range"),
                                        radial_rings=n["man"].get("radial_rings"),
                                        composite=n["man"].get("composite"),
                                        screen_frame=n["man"].get("screen_frame"),
                                        screened=n["man"].get("screened")))
        else:
            self.man_passed_logged.update(n["node_id"] for n in newly)
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

    def pop_batch_quota(self) -> list[dict]:
        """Harvest-v2 pop: the cross-partition CHOICE is the quota's (intended vs realized
        time share only), the within-partition ORDER is the unchanged priority sort over that
        ONE partition's nodes.

        Structurally identical to `pop_batch_scheduled` — same capped-root eviction, same
        maneuver floor applied WITHIN the served partition — and deliberately so: the two
        allocators must differ in the partition CHOICE and in nothing else, or a
        realized-mix comparison between them measures the other differences.

        The ranker boundary is inherited verbatim: `pref_loc_*` ranks admitted output for
        keeper/emission ordering and never enters scheduling. It is absent here, as it is
        from `_split_reserved` and `pop_batch_scheduled`."""
        live = []
        for n in self.frontier:
            if self.expansions_per_root.get(str(n["root_id"]), 0) >= M_CAP:
                self.node_embs.pop(n["node_id"], None)
            else:
                live.append(n)
        self.frontier = live
        self.totals["cap_hits"] = sum(1 for v in self.expansions_per_root.values() if v >= M_CAP)
        queue_lens: dict = {}
        for n in self.frontier:
            queue_lens[n["partition"]] = queue_lens.get(n["partition"], 0) + 1
        part = self.quota.pick(queue_lens)
        self.quota.log_choice(self.batch_i, part, queue_lens)
        self._served_partition = part
        if part is None:
            return []
        pool = [n for n in self.frontier if n["partition"] == part]
        pool.sort(key=lambda n: -n["priority"])
        batch, _rest = self._split_reserved(pool)
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
                      "q3_dup", "guarded", "reframe_not_q3", "interior_gated")

    def _reconcile_snapshot(self) -> dict:
        return {k: self.totals[k] for k in self.RECONCILE_KEYS}

    def _reconcile_batch(self, before: dict, n_cands: int):
        """`found == written + dropped_*` per work unit, or EXIT LOUD.

        Two identities have to close on every batch, and a long unattended run that silently
        loses candidates is exactly the failure a summary cannot show you afterwards:

          1. FRONTIER   every scored candidate is pushed OR named as gated:
                        candidates == frontier_pushed + interior_gated
                        (v1.6: the sourcing interior gate is the only thing allowed to
                        remove a candidate before the frontier, and it enters the identity
                        rather than sitting outside it — a gate outside the identity is a
                        gate that can eat candidates and still balance.)
          2. HARVEST    every check lands in exactly one fate:
                        harvest_checks == precanon_dup + canonical_q3 + canon_not_q3
                        canonical_q3   == admitted + q3_dup + guarded + reframe_not_q3

        (`render_failed` checks are subtracted from `harvest_checks` at the point of failure,
        so they are outside both identities by construction rather than by omission.)"""
        d = {k: self.totals[k] - before[k] for k in self.RECONCILE_KEYS}
        problems = []
        if d["candidates"] != d["frontier_pushed"] + d["interior_gated"]:
            problems.append(f"frontier: found {d['candidates']} candidates but pushed "
                            f"{d['frontier_pushed']} + interior_gated "
                            f"{d['interior_gated']}")
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

    def wall_elapsed_s(self) -> float:
        """Wall seconds this run has burned in its BATCH LOOP, across resumes.

        `wall_s_base` is what previous sessions spent (checkpointed); `_session_t0` is when
        this one entered the loop. Derived rather than stored-and-incremented so a kill
        between the increment and the checkpoint cannot lose or double-count a session —
        the same reason the admitted count is re-derived from the ledger.

        SCOPE, because "wall clock" invites the wrong reading: the clock starts when the
        loop is entered, so the one-time pre-loop cost — model load plus the first
        `draw_roots` across every family, measured at ~12 min for four families — is outside
        it, exactly as it is outside `active_s`. What the cap DOES cover is every
        replenishment `draw_roots` inside the loop, which is the recurring cost the active
        cap cannot see and the reason this exists. Add the startup by hand when comparing to
        a process wall clock. Measured on the v1.4 exploration run: 57.6 wall min against
        57.2 active min at batch 101, i.e. in-loop replenishment was ~1% there — the
        maneuver push rate kept the frontier above the low-water mark."""
        if getattr(self, "_session_t0", None) is None:
            return float(self.wall_s_base)
        return float(self.wall_s_base) + (time.time() - self._session_t0)

    def wall_exhausted(self) -> bool:
        """Would the NEXT batch cross the wall cap? Its own method so it is testable without
        driving the whole loop — the same reason `prune_frontier` is one.

        Note the `+ est_batch_s`: the rule is never START a unit that cannot finish inside
        the remaining budget, not stop once the budget is already blown. `wall_budget_s = 0`
        disables it, which is the historical behaviour and every run before this one."""
        if not self.wall_budget_s:
            return False
        return self.wall_elapsed_s() + self.est_batch_s > self.wall_budget_s

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
                # ... and the TRIGGERED half of that lineage rides the same way, for the
                # same reason plus one more: triggered and fresh yield are reported as
                # separate populations, so a descendant that lost the stamp would be
                # counted as fresh supply and inflate exactly the number the split exists
                # to protect (`minibrot_maneuvers.md` §8.0).
                triggered=parent.get("triggered"),
                # the phoenix seed's own provenance (branch / theta / offset), so a phoenix
                # candidate stays attributable to the skeleton point it came from
                phoenix=parent.get("phoenix"),
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
        near-dup + admission. Logs every harvest check's (cheap, canonical, decode) triple.

        v1.6 RECORD-AND-RANK. Admission below is byte-for-byte the path it always was; what
        is added is that every candidate at or above `tau_rec` is APPENDED to
        `q4_candidates.jsonl` with its scores and its per-stage fate, whatever that fate
        turns out to be. Three populations that used to leave no reviewable trace now do:
        rows below `tau_h` (recorded on their cheap score, `rank_tier=1`), pre-canonical
        coord-dups, and `canon-not-q3` rows.

        THE FATE COUNTERS ARE STILL INCREMENTED EXACTLY ONCE PER CHECK, and deliberately
        AFTER the render step rather than before it. A pre-canonical dup that is rendered and
        whose render then fails must leave the population without touching `precanon_dup`, or
        the batch reconcile stops balancing — so the dup verdict is computed up front, stashed
        on the row, and only *counted* at the fate branch alongside every other fate.
        """
        # -- the recording floor. A record is a line; `tau_h` still decides who is RENDERED.
        for c in cands:
            c["q4_checked"] = c["cheap_pgood"] >= self.tau_h[c["partition"]]
            c["q4_recorded"] = c["cheap_pgood"] >= self.tau_rec[c["partition"]]
        for c in cands:
            if c["q4_recorded"] and not c["q4_checked"]:
                self.totals["q4_recorded_below_tau_h"] += 1
                self._q4_record(c, fate="below_tau_h")
        checks = [c for c in cands if c["q4_checked"]]
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
        #
        # v1.6: the SKIP is now optional (`--record-canon-dups`). Dup-ness is a property of the
        # ledger's coordinate cloud, not of the picture — a candidate inside an admitted q3's
        # radius can still be the better image, and this run's deliverable is a ranked list a
        # human cuts, not a set of distinct ledger rows. With the flag on the dup is rendered
        # and decoded so it can be RANKED, and is still refused admission. The flag exists
        # because the saving it gives up is large (campaign 2 skipped ~82% of checks this way),
        # so it is measured in the shakedown rather than assumed affordable.
        for c in checks:
            distinct, dup_of = ps.is_distinct(c["cx"], c["cy"], c["fw"],
                                              self.clouds.get(c["partition"], []), ps.DEDUP_K,
                                              c=ident_c(c["partition"], c["c"]))
            c["precanon_dup_of"] = None if distinct else dup_of
        if not self.record_canon_dups:
            for c in [c for c in checks if c["precanon_dup_of"] is not None]:
                self.totals["precanon_dup"] += 1
                self._log_harvest(c, admitted=False, reframe_decoded=None,
                                  precanon_dup=c["precanon_dup_of"])
                self._q4_record(c, fate="precanon_dup")
            checks = [c for c in checks if c["precanon_dup_of"] is None]
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
                              timeout=self.unit_timeout_s(),
                              **render_args_for(c["partition"], c["c"])): i
                    for i, c in enumerate(checks)}
            for fut in cf.as_completed(futs):
                fut.result()
        # A timed-out / failed confirmation render leaves no tile; scoring a missing path
        # would raise deep in the scorer, so drop those checks here and count them.
        missing = [i for i, t in enumerate(tiles) if not t.exists()]
        if missing:
            self.totals["render_failed"] += len(missing)
            # v1.6: a check whose confirmation render FAILED still gets a record. It cleared
            # the floor, so "nothing scoring above the floor is discarded unrecorded" covers
            # it — and the row that vanishes on a render failure is exactly the row a later
            # question about render failures would need. It carries its cheap score and
            # `rank_tier=1`, because the canonical scores it would have had do not exist.
            for i in missing:
                self._q4_record(checks[i], fate="render_failed")
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
            # The frame's CLASS under the fixed cut: None below `floors.GOOD_FLOOR`, else 3,
            # or 4 when P(>=4) also clears `floors.GREAT_CUT`. Same column, same meaning as
            # the retired per-partition `corn_decode` produced ("the head's class for this
            # frame, None if it is not good") — what changed is that the bar is one number
            # for every partition and is read from the floor owner, not frozen per row.
            c["canon_decoded"] = F.good_class(pg, pg4)

        # 2. reframe + admit the canonical-q3 confirmations. Cheap pre-reframe dedup:
        # reframe only nudges the center by <=0.25*fw and fw by <=1.41x, so a candidate
        # already inside an admitted q3's dedup radius cannot escape it — skip the 12-render
        # reframe and log it as a dup (this is where most compute is saved in a hot region).
        for c in checks:
            admitted = False
            reframe_decoded = None
            # A pre-canonical dup that was RENDERED anyway (--record-canon-dups) is counted
            # as the dup it is and never offered to admit(): its canonical scores exist for
            # the RANK, and admission is untouched by this feature. Counted here rather than
            # up front so a failed render leaves the population before any fate fires.
            if c.get("precanon_dup_of") is not None:
                self.totals["precanon_dup"] += 1
                self._log_harvest(c, admitted=False, reframe_decoded=None,
                                  precanon_dup=c["precanon_dup_of"])
                self._q4_record(c, fate="precanon_dup")
                continue
            if c["canon_decoded"] is not None:      # good — a class 4 confirms too
                self.totals["canonical_q3"] += 1
                pre_distinct, _ = ps.is_distinct(c["cx"], c["cy"], c["fw"],
                                                 self.clouds.get(c["partition"], []), ps.DEDUP_K,
                                                 c=ident_c(c["partition"], c["c"]))
                if not pre_distinct:
                    self.totals["q3_dup"] += 1
                    c["admit_fate"] = "q3_dup"
                else:
                    admitted, reframe_decoded = self.admit(c, cdir)
            else:
                self.totals["canon_not_q3"] += 1
                c["admit_fate"] = "canon_not_q3"
            self._log_harvest(c, admitted, reframe_decoded)
            self._q4_record(c, fate=c.get("admit_fate") or "unknown",
                            reframe_decoded=reframe_decoded)

    def admit(self, c, cdir):
        """Existing reframe + near-dup + admission path (guarded scorer, `floors.GOOD_FLOOR`)."""
        loc = loc_of(c["partition"], c["c"], c["cx"], c["cy"], c["fw"])
        wd = cdir / f"reframe_n{c['node_id']}"
        res = reframe.reframe_location(loc, scorer=self.scorer, seed=0, workdir=wd, workers=ps.WORKERS)
        guard_pass = res.score > guard.GUARD_SENTINEL + 1e-6
        nb, pg, pg4 = ps._chosen_probs(res)
        decoded = F.good_class(pg, pg4) if guard_pass else None
        is_q3 = decoded is not None
        ocx, ocy, ofw = float(res.cx), float(res.cy), float(res.fw)
        distinct, dup_of = (False, None)
        if is_q3:
            distinct, dup_of = ps.is_distinct(ocx, ocy, ofw, self.clouds[c["partition"]],
                                              ps.DEDUP_K, c=ident_c(c["partition"], c["c"]))

        run_ts = self.run_dir.name
        id_tag = {"mandelbrot": "m"}.get(c["partition"], c["partition"].replace(":", "_"))
        oid = f"st_{id_tag}_{run_ts}_{self.seq:06d}"
        self.seq += 1
        feat = None
        if is_q3 and distinct:
            tile = cdir / f"{oid}.jpg"
            feat = ps.outcome_feature(self.scorer, ocx, ocy, ofw, tile,
                                      **render_args_for(c["partition"], c["c"]))
        row = dict(
            id=oid, ts=run_ts, family=c["partition"], mix_source="steered",
            node_id=c["node_id"], root_id=c["root_id"],
            seed_cx=c["cx"], seed_cy=c["cy"],
            outcome_cx=ocx, outcome_cy=ocy, outcome_fw=ofw,
            k3=float(res.score), raw_top3=[float(c["cheap_eord"])],
            reached_depth=int(c["depth"]),
            # RAW probabilities only. The ledger stores no derived class and no threshold:
            # `production_seeder.is_good` re-applies the live `floors.GOOD_FLOOR` to `p_good`
            # at every read, so a floor move reaches every row ever written. The harvest log
            # and the q4 store DO carry a class column, because the reframe probabilities are
            # not otherwise recoverable there — see `_log_harvest`.
            p_notbad=nb, p_good=pg, p_ge4=pg4,
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
        if c["partition"] == "phoenix":
            # The phoenix analogue of the julia `c` stamp: the FULL parameter point, because
            # a phoenix row's dup identity is the (c, p, z_{-1}) 6-vector and a row carrying
            # only `c` would collide with every other p and z_{-1} at the same c
            # (`production_seeder.row_phoenix_key`).
            v = _phoenix_cpz(c["c"])
            row.update(ps.phoenix_ident_fields(c=(float(v[0]), float(v[1])),
                                               p=(float(v[2]), float(v[3])),
                                               z_m1=(float(v[4]), float(v[5]))))
            if c.get("phoenix"):
                row["phoenix_seed"] = c["phoenix"]    # branch/theta/offset, provenance only
        elif c["c"] is not None:                     # julia twin outcome carries the parameter c
            row["julia_c_re"], row["julia_c_im"] = c["c"][0], c["c"][1]
            # CAMPAIGN schema (outcome_* = viewport, c = julia_c_*): stamp it so the row is
            # born tagged and no reader has to infer the era from field presence.
            row[jls.SCHEMA_KEY] = jls.CAMPAIGN
        if self.cur_dive is not None:                # dive: stamp provenance for the read
            row["mix_source"] = "dive"
            row["dive_id"], row["dive_start_group"], row["dive_source_id"] = self.cur_dive
        self.ledger.append(row, feat)
        # v1.6: the reframed frame + the outcome id ride back on the candidate so the
        # record-and-rank store points at the frame that was ADMITTED, not the pre-reframe
        # one. A sheet built off the pre-reframe coords would show a different picture from
        # the one the ledger holds.
        c["outcome"] = (ocx, ocy, ofw)
        c["outcome_oid"] = oid
        c["outcome_guard_pass"] = guard_pass
        if is_q3 and distinct:
            c["admit_fate"] = "admitted"
            self.clouds[c["partition"]].append(row)
            self.run_clouds[c["partition"]].append(row)   # keep the rejection-sampler cloud current
            self.totals["admitted"] += 1
            if c.get("man"):
                self.totals["man_admitted"] += 1
            if c.get("triggered"):
                self.totals["trig_admitted"] += 1
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
            # POP QUOTA: this partition just MINED currency, so its measured cost-to-mine
            # updates here. The credit is taken on a DISTINCT ADMISSION only — not on every
            # canonical decode >= 3 — because a q3_dup and a pre-canonical dup add nothing to
            # the corpus the deficit is counted against, and pricing them as production would
            # make the churniest partition look the cheapest. The weight is the reframed
            # decode's own class through `CLASS_WEIGHT`, so a class-4 admission is worth 10
            # distinct class-3s, exactly as the deficit denominates them.
            if self.quota is not None:
                self.quota.note_admission(c["partition"])
                self.quota.credit_decode(c["partition"], decoded)
            # julia hook: fire per qualifying (admitted-q3) c-plane parent.
            if self.julia_hook and c["partition"] in self.families:
                self.add_julia_root(c["partition"], (ocx, ocy), oid)
            # v1.6 MANEUVERS-ON-ADMISSIONS: the admitted location is itself a judged-good
            # seed, so it gets the label-seeded harvest's own pair of operators, live.
            self.fire_triggered_maneuvers(c, ocx, ocy, ofw, oid, decoded)
            return True, decoded
        elif is_q3:
            self.totals["q3_dup"] += 1
            c["admit_fate"] = "q3_dup"
        elif not guard_pass:
            self.totals["guarded"] += 1
            c["admit_fate"] = "guarded"
        else:
            c["admit_fate"] = "reframe_not_q3"
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
        #
        # `canon_decoded` / `reframe_decoded` are ANNOTATION columns and are now
        # `floors.good_class(...)` — None below the good floor, else 3, or 4 when P(>=4) also
        # clears `floors.GREAT_CUT`. Same column, same meaning, one bar for every partition
        # instead of nine. Nothing DECIDES on them: admission is `is_good` on the raw `p_good`
        # in the outcome ledger, and these exist so a sheet can label a tile without joining.
        jc = c.get("c")
        self._writer(self.harvest_log).write_row(dict(
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
        ))

    # ------------------------------------------- maneuvers-on-admissions (v1.6)
    def fire_triggered_maneuvers(self, c, ocx, ocy, ofw, oid, decoded) -> int:
        """Run `snap_at_seed` -> `neighborhood_expand` on a freshly ADMITTED location.

        WHY THIS IS THE SAME MECHANISM AS THE LABEL-SEEDED HARVEST AND NOT A NEW ONE. That
        harvest seeds from the corpus's own class-3/4 locations — the two generation methods
        Matt rated top of seven. An admission IS a location this run just decoded at class 3
        or better, i.e. the same kind of seed, produced hours earlier than a label could
        arrive. So the primitives are imported from `label_seeded_harvest` verbatim rather
        than reimplemented; a second copy would let the run's live loop and the offline
        harvest silently diverge on what "the nucleus at a judged view" means.

        THE YIELD OF THIS IS TALLIED SEPARATELY FROM FRESH-SEED YIELD, ALWAYS. The operators
        feed themselves (`minibrot_maneuvers.md` §8.0: a view produced by snapping to a
        nucleus is centred on a nucleus, so snapping it again nearly always succeeds), and a
        pooled rate becomes a property of the feedback loop rather than of the operator. The
        `triggered` stamp rides the whole subtree exactly as `man` does, so a descendant three
        rungs down is still attributable, and every `trig_*` counter is disjoint from the
        fresh ones.

        c-PLANE ONLY, for the reason `minibrot_maneuvers.md` §6 gives: a julia/phoenix
        viewport is a z-plane with no nucleus in the parameter-plane sense, so the operators
        are not defined there and are skipped rather than faked.

        Never raises. A trigger is an enrichment, and losing one must never cost the
        admission that produced it — the admission is already in the ledger by this point.
        """
        if not self.trig_on:
            return 0
        deg = mnv.PARTITION_DEGREE.get(c["partition"])
        if deg is None:                       # julia / phoenix z-plane: undefined, not skipped-as-failed
            self.totals["trig_unavailable"] += 1
            return 0
        if self.trig_fired_this_batch >= self.trig_max_per_batch:
            self.totals["trig_budget_skip"] += 1
            return 0
        seed = dict(seed_id=oid, family=c["partition"], degree=deg,
                    cx=str(ocx), cy=str(ocy), fw=str(ofw),
                    score=int(decoded or 3), batch=self.run_dir.name, image_id=oid)
        t0 = time.time()
        try:
            rng = np.random.default_rng(self.seed ^ (hash(oid) & 0xFFFFFFFF))
            rows, st = lsh.enumerate_seed(
                seed, self.trig_ks, rng=rng,
                deadline=t0 + self.trig_deadline_s,
                max_found=self.trig_nbh_m, max_probes=self.trig_nbh_probes,
                period_max=self.trig_period_max)
        except Exception as e:                                        # noqa: BLE001
            self.totals["trig_unavailable"] += 1
            print(f"  WARN triggered maneuver on {oid} failed: {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)
            return 0
        self.trig_fired_this_batch += 1
        self.totals["trig_fired"] += 1
        self.totals["trig_atoms"] += len({r["atom_key"] for r in rows})
        self.trig_probe_s += time.time() - t0
        # harvest v2 §3: the TRIGGERED channel is screened too. In v1 it was not — triggered
        # views were pushed with a bare neutral prior — so the one channel §2 promotes to a
        # budgeted supply source was also the one channel whose rows carried neither sourcing
        # score. One batched pass over this trigger's distinct views, on the same cache and
        # the same budget clamp as the fresh operators, so a view already screened as a fresh
        # maneuver is a cache hit rather than a second field.
        screens: dict = {}
        if self.man_views is not None and rows:
            jobs, seen = [], set()
            for r in rows:
                key = mvs.view_key(r["atom_key"], r["k"])
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(dict(view_key=key, cx=r["cx"], cy=r["cy"], fw=r["fw"],
                                 atom_key=r["atom_key"], k=r["k"],
                                 family=c["partition"],
                                 window_scale=r.get("window_scale")))
            ts = time.time()
            before = set(self.man_views.by_key)
            screens = self.man_views.screen_many(jobs, budget_s=self.unit_timeout_s())
            self.man_screen_s += time.time() - ts
            for key, rec in screens.items():
                if key in before:
                    continue
                if rec.get("screened"):
                    self.totals["man_view_screened"] += 1
                    self.man_comp_dist.add(rec.get("composite"))
                    if rec.get("vetoed"):
                        self.totals["man_view_vetoed"] += 1
                else:
                    self.totals["man_view_unscreenable"] += 1
            self.totals["man_screen_cache_hits"] = self.man_views.n_hits
            self.totals["man_view_fields_cached"] = self.man_views.n_fields_cached
        pushed = 0
        for r in rows:
            # The SHARED visited key (`atom|k`), so an atom a fresh maneuver already pushed
            # is not pushed again under a triggered label — the two populations must be
            # disjoint or the separate tallies are double-counting one node.
            key = mvs.view_key(r["atom_key"], r["k"])
            if key in self.man_visited:
                self.totals["man_avail_unused"] += 1
                continue
            self.man_visited.add(key)
            nid = self.new_node_id()
            man = dict(op=r["op"], k=r["k"], origin_node_id=c["node_id"],
                       atom_id=r["atom_id"], atom_key=r["atom_key"],
                       period=r["period"], log10_abs_A=r["log10_abs_A"],
                       window_scale=r["window_scale"], degree=deg,
                       trigger_oid=oid, triggered=True)
            sc = screens.get(key)
            if sc is not None:
                man["screened"] = bool(sc.get("screened"))
                man["screen_policy"] = sc.get(mvs.fm.POLICY_KEY)
                man["screen_frame"] = "view"
                for fld in ("composite", "vetoed", "size_factor", "band_coverage",
                            "band_coverage_q25", "interior_fraction", "view_fw",
                            "radial_range", "radial_rings",
                            "view_fit", "view_fit_p_notbad", "view_fit_model",
                            "view_fit_reason"):
                    man[fld] = sc.get(fld)
            self.frontier.append(dict(
                node_id=nid, root_id=c["root_id"], partition=c["partition"], c=None,
                cx=float(r["cx"]), cy=float(r["cy"]), fw=float(r["fw"]),
                depth=int(c["depth"]),
                priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
                cheap_eord=None, cheap_pgood=None, branch="triggered",
                mix_source=f"triggered:{r['method']}:k={r['k']}",
                triggered=True, man=man,
            ))
            pushed += 1
        self.totals["trig_nodes_pushed"] += pushed
        self._log_maneuver(dict(
            kind="triggered", batch=self.batch_i, oid=oid, partition=c["partition"],
            cx=str(ocx), cy=str(ocy), fw=str(ofw), decoded=decoded,
            rows=len(rows), pushed=pushed, atoms=len({r["atom_key"] for r in rows}),
            probe_s=round(time.time() - t0, 3), stats=st))
        return pushed

    # ------------------------------------------------------- record-and-rank
    # The fates a recorded row can carry. Named once, so the writer, the reconcile and the
    # readout cannot drift, and so an unhandled branch shows up as the literal string
    # "unknown" in a count rather than as a silently absent row.
    Q4_FATES = ("below_tau_h", "precanon_dup", "canon_not_q3", "q3_dup", "guarded",
                "reframe_not_q3", "admitted", "interior_gt_30", "render_failed", "unknown")

    def _q4_record(self, c, *, fate: str, reframe_decoded=None):
        """Append one candidate to the run's record-and-rank store.

        EVERY ROW IS RENDERABLE FROM THIS FILE ALONE — geometry, family, and the dynamical
        parameter (julia seed c, or the phoenix (c, p, z_{-1}) point) — because the whole
        point of recording a reject is that a human can look at it later, and the julia audit
        already paid for the version of this that stored fates without coordinates.

        `rank_tier` is stamped, never inferred. A row with a canonical decode (tier 2) and a
        row carrying only a cheap score (tier 1) are scores from two different geometries
        (640x360 ss2 against 384x216 ss1) and pooling them into one ordering would be the
        cap/geometry error `orbital_field_metrics.md` §5 forbids. The ranking sorts WITHIN a
        tier; the tier is the row's own property and is written down at the moment the
        distinction is still known.
        """
        jc = c.get("c")
        canon = c.get("canon_eord")
        out = c.get("outcome")
        # `cheap_eord` is present for every row the gate sees, because the gate runs AFTER
        # score_cheap (see run()) — deliberately, so a gated row still records what the
        # classifier thought of it and "does the head agree with the >0.30 rule?" stays a
        # read. `.get` rather than `[]` so relocating the gate degrades to an unranked row
        # instead of killing the batch.
        cheap = c.get("cheap_eord")
        rec = dict(
            batch=self.batch_i, partition=c["partition"], fate=fate,
            rank_tier=(2 if canon is not None else (1 if cheap is not None else 0)),
            rank_score=(float(canon) if canon is not None
                        else (float(cheap) if cheap is not None else None)),
            node_id=c["node_id"], root_id=c["root_id"], depth=c["depth"],
            cx=c["cx"], cy=c["cy"], fw=c["fw"],
            # the dynamical parameter, whichever plane this partition lives on
            julia_c_re=(None if jc is None else str(jc[0])),
            julia_c_im=(None if jc is None else str(jc[1])),
            # PHOENIX NEEDS ALL SIX NUMBERS, and writing two of them is the bug this line
            # fixes. A phoenix row's identity is the whole (c, p, z_{-1}) point; `julia_c_*`
            # above captures only `c`, and `phoenix` below is the seed's PROVENANCE
            # (branch/theta/offset), not its parameters. A store missing p and z_{-1} cannot
            # rebuild the candidate — it rebuilds the DEFAULT phoenix plane at the right
            # coordinates, which renders a different fractal that looks like it worked. That
            # is the same failure `prescreen._render`'s `family_params` kwarg exists to
            # prevent, reintroduced one layer up.
            **(dict(phoenix_c_re=str(jc[0]), phoenix_c_im=str(jc[1]),
                    phoenix_p_re=str(jc[2]), phoenix_p_im=str(jc[3]),
                    phoenix_zm1_re=str(jc[4]), phoenix_zm1_im=str(jc[5]))
               if (c["partition"] == "phoenix" and jc is not None and len(jc) == 6) else {}),
            phoenix=c.get("phoenix"),
            # the admitted frame, when there is one — a sheet must show what the ledger holds
            outcome_cx=(None if out is None else out[0]),
            outcome_cy=(None if out is None else out[1]),
            outcome_fw=(None if out is None else out[2]),
            outcome_id=c.get("outcome_oid"),
            cheap_eord=cheap, cheap_pgood=c.get("cheap_pgood"),
            cheap_nb=c.get("cheap_nb"),
            canon_eord=canon, canon_pgood=c.get("canon_pg"), canon_nb=c.get("canon_nb"),
            canon_pge4=c.get("canon_pge4"), canon_decoded=c.get("canon_decoded"),
            reframe_decoded=reframe_decoded,
            tau_h=self.tau_h[c["partition"]], tau_rec=self.tau_rec[c["partition"]],
            good_floor=F.GOOD_FLOOR,
            int_frac=c.get("int_frac"), occ=c.get("occ"),
            mix_source=c.get("mix_source"), maneuver=c.get("man"),
            # TRIGGERED vs FRESH is stamped on the ROW, never derived at readout time from
            # `mix_source` string-matching: the two yields must never be pooled, and a
            # readout that has to infer the split will eventually infer it wrong.
            triggered=bool(c.get("triggered")),
            branch=c.get("branch"), scorer_version=ps.SCORER_VERSION,
        )
        self.totals["q4_recorded"] += 1
        self._writer(self.q4_log).write_row(rec)

    # ------------------------------------------------- interior gate (sourcing)
    def interior_gate(self, cands):
        """Matt's >0.30-interior rule, applied at SOURCING. Returns the surviving candidates.

        FIRST PRODUCTION CONSUMER of the rule inside the walk. `apply_interior_rule.py` applies
        it to the LABEL STORE after the fact (a crop already rendered and about to be judged);
        `label_seeded_harvest` applies it at sourcing but only on that harvest's own path. Here
        it removes the candidate before it can consume a canonical confirmation render, a
        reframe, a ledger row or a frontier slot.

        THE DISCARD IS RECORDED, NOT SILENT. Every gated candidate lands in the
        record-and-rank store with `fate="interior_gt_30"`, so "how much did the gate cost?"
        is a read rather than a re-run — and the count enters the batch reconcile identity
        below, which is what makes it impossible for the gate to quietly eat candidates.

        Strict `>`: a frame at exactly 0.30 is KEPT, mirroring `present.rs`'s strict `<` on
        the other side of the same boundary. An off-by-one-side threshold is invisible in a
        count, which is why the comparison is stated here and asserted by a test.

        A candidate with NO measure is KEPT and counted apart — an absent measure is not a
        high one, the same rule `apply_interior_rule.fires` uses.
        """
        if not self.interior_gate_on:
            return cands
        kept, gated = [], []
        for c in cands:
            v = c.get("int_frac")
            if v is None:
                self.totals["interior_unmeasured"] += 1
                kept.append(c)
            elif float(v) > self.interior_discard:
                gated.append(c)
            else:
                kept.append(c)
        for c in gated:
            self.totals["interior_gated"] += 1
            # Recorded unconditionally, not only above the floor: the gate's own population
            # is the thing a later question about the gate needs, and it is exactly the
            # population no score was ever computed for.
            self._q4_record(c, fate="interior_gt_30")
        return kept

    # ---------------------------------------------------------------- push
    def visited_density_of(self, c) -> int:
        """Cross-run visited density for one BREADTH candidate, or 0 with the memory off.

        The identity comes from `ident_c` — the same parameter vector the dedup path hands
        `is_distinct` — so a julia/phoenix candidate is compared only against prior visits on
        ITS OWN dynamical plane. A c-plane candidate has no identity (None) and matches the
        c-plane bucket, which is what makes the mandelbrot/multibrot memory work at all."""
        if not self.sat_on:
            return 0
        return self.sat_index.density(c["partition"], ident_c(c["partition"], c.get("c")),
                                      c["cx"], c["cy"])

    def push_children(self, cands):
        prio_rows = []
        batch_sat = 0                                    # saturated candidates this batch
        sat_seen: dict = defaultdict(int)                # partition -> candidates scored
        sat_disc_n: dict = defaultdict(int)              # partition -> density > 0
        for c in cands:
            dup_pen = dup_penalty(c["cx"], c["cy"], self.clouds.get(c["partition"], []))
            cos_max = float(c.get("cos_max", 0.0))
            sat_d = self.visited_density_of(c)
            g = gumbel(self.rng, T_GUMBEL)               # RNG draw order unchanged from pilot
            prio, terms = priority_terms(
                c["cheap_eord"], g, dup_pen, cos_max,
                self.lambda_m, self.beta, c["depth"], self.morph_lo, self.morph_hi,
                sat_density=sat_d,
                sat_strength=(self.sat_strength if self.sat_on else 0.0))
            if self.sat_on:
                sat_seen[c["partition"]] += 1
                self.totals["sat_mem_scored"] += 1
                agg = self.sat_by_partition.setdefault(
                    c["partition"], dict(n=0, discounted=0, density_sum=0))
                agg["n"] += 1
                if sat_d > 0:
                    sat_disc_n[c["partition"]] += 1
                    agg["discounted"] += 1
                    agg["density_sum"] += sat_d
                    self.totals["sat_mem_discounted"] += 1
                    self.totals["sat_mem_density_sum"] += sat_d
            self.frontier.append(dict(
                node_id=c["node_id"], root_id=c["root_id"], partition=c["partition"], c=c["c"],
                cx=c["cx"], cy=c["cy"], fw=c["fw"], depth=c["depth"], priority=prio,
                cheap_eord=c["cheap_eord"], cheap_pgood=c["cheap_pgood"], branch=c["branch"],
                mix_source=c.get("mix_source"),   # carry root supply down the tree (probe attribution)
                man=c.get("man"),                 # maneuver provenance, if this lineage has one
                # THE LINEAGE STAMPS RIDE THE REBUILT NODE, NOT ONLY THE CANDIDATE. This
                # dict is a FRESH node, so anything not named here is dropped — and
                # `expand_group` reads these back off the parent NODE to stamp the next
                # generation's candidates. Omitting one does not lose it once; it truncates
                # it at generation 1 and every descendant is then stamped with the DEFAULT,
                # which reads as a positive fact about a different population:
                #   `triggered`  — a triggered descendant counted as FRESH supply inflates
                #                  exactly the arm the split exists to protect
                #                  (`minibrot_maneuvers.md` §8.0). Measured on
                #                  q4_long_harvest_20260803: 178 of 794 triggered-lineage
                #                  rows kept the stamp, 616 were written as fresh.
                #   `phoenix`    — the (branch, theta, offset) of the skeleton point the
                #                  seed came from; without it a phoenix row is unattributable
                #                  to its sampler branch. Same run: 17 of 1,238 kept it.
                triggered=c.get("triggered"),
                phoenix=c.get("phoenix"),
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
        #
        # v1.7 shares this row rather than opening a second stream: both halves answer the same
        # question ("is a soft steering term firing on everything?") and the visited-density
        # half is what says, per partition and per batch, whether the cross-run memory is doing
        # anything at all. THE MORPH HALF IS `null`, NOT ZERO, WHEN `lambda_m == 0` — a
        # `frac: 0.0` written by a run that never measured novelty is a claim about a
        # population that does not exist, and the pre-v1.7 readers (`campaign2_readout`,
        # `steered_v1_2_dive_report`) would average it in as a real observation.
        if cands and (self.lambda_m > 0.0 or self.sat_on):
            morph_on = self.lambda_m > 0.0
            if morph_on:
                self.totals["nov_scored"] += len(cands)
                self.totals["sat_hits"] += batch_sat
            with open(self.sat_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(
                    batch=self.batch_i, n=len(cands),
                    sat=(batch_sat if morph_on else None),
                    frac=(round(batch_sat / len(cands), 4) if morph_on else None),
                    mem_perm=self.morph.n_perm, mem_recency=self.morph.n_recency,
                    mem_total=len(self.morph),
                    # cross-run visited-density half (null when the memory is off)
                    visited=(dict(
                        n=sum(sat_seen.values()),
                        discounted=sum(sat_disc_n.values()),
                        frac=round(sum(sat_disc_n.values()) / max(1, sum(sat_seen.values())), 4),
                        by_partition={p: dict(n=sat_seen[p], discounted=sat_disc_n.get(p, 0),
                                              frac=round(sat_disc_n.get(p, 0) / sat_seen[p], 4))
                                      for p in sorted(sat_seen)},
                    ) if self.sat_on else None),
                )) + "\n")
        if prio_rows:
            self._writer(self.prio_log).write_rows(prio_rows)
        # prune to the memory bound (keep the best); drop pruned nodes' cached embeddings.
        # Maneuver-originated nodes are PROTECTED, not exempt: they are the population the
        # reserved floor exists to protect, and pruning the pooled frontier by priority
        # would delete them first (they carry a neutral prior, or a score from a head that
        # has never seen their kind) — the cap would silently undo the floor.
        #
        # v1.4: PROTECTED IS NOT UNLIMITED. The exemption used to be total, so once the
        # maneuver population passed FRONTIER_CAP the ordinary nodes' room went to zero and
        # every one of them was evicted — the frontier becomes 100% maneuver nodes and the
        # walk starves. That is the same failure `pop_batch` records for capped-root dead
        # weight, reached by another route, and the third operator is what makes it
        # reachable: it pushes ~2x as many nodes per fired probe, and a 2-minute shakedown
        # already pushed 40/batch against ~21 expanded, i.e. it crosses 6000 inside a
        # 7-hour run. So maneuvers get a guaranteed SHARE and are pruned among themselves
        # beyond it. Below the share nothing changes — which is every run before this one.
        self.prune_frontier()

    def prune_frontier(self):
        """Prune the frontier to `FRONTIER_CAP`, protecting maneuver nodes up to their share.

        Its own method so the tests can drive THE code rather than a hand-mirrored copy of
        it — a fixture that reimplements its subject asserts `f(x) == f(x)`
        (`verification_practice.md` §1.10) and, worse, stays green while the subject rots."""
        if len(self.frontier) <= FRONTIER_CAP:
            return
        self.frontier.sort(key=lambda n: -n["priority"])
        man = [n for n in self.frontier if n.get("man")] if self.maneuvers else []
        others = [n for n in self.frontier if not n.get("man")] if self.maneuvers \
            else self.frontier
        # maneuvers keep up to their share; unused share falls to the ordinary nodes, and
        # unused ordinary room falls back to the maneuvers.
        man_room = min(len(man), max(int(FRONTIER_CAP * MAN_FRONTIER_SHARE),
                                     FRONTIER_CAP - len(others)))
        keep_man = man[:man_room]
        room = max(0, FRONTIER_CAP - len(keep_man))
        kept_ids = {n["node_id"] for n in keep_man} | {n["node_id"] for n in others[:room]}
        dropped = [n for n in self.frontier if n["node_id"] not in kept_ids]
        self.frontier = [n for n in self.frontier if n["node_id"] in kept_ids]
        self.totals["man_frontier_pruned"] += sum(1 for n in dropped if n.get("man"))
        for n in dropped:
            self.node_embs.pop(n["node_id"], None)

    # ---------------------------------------------------------------- state
    # The pre-registered bars, written to disk BEFORE the first batch. A bar stated after a
    # readout is not a bar, and `measurement_practice.md`'s "config changes are announced at
    # decision time, never discovered in a readout" cuts the same way: the file exists before
    # there is anything to be tempted by.
    PREREG = {
        "view_fit_v1_1_vs_composite_v3": {
            "claim": ("view_fit v1.1 supersedes composite_v3 as the SOURCING order "
                      "(including --maneuver-view-prior), pipeline-wide."),
            "adopted_only_if": ("view_fit's ordered top-k beats composite_v3's on THIS "
                                "run's labeled outcome by delta-AP >= +0.1181"),
            "margin": 0.1181,
            "margin_basis": ("the lower bound of the fit-era CI, "
                             "data/atlas/view_fit_v1_1.json readout."
                             "ap_delta_v11_vs_composite = 0.1819 [0.1181, 0.2466], "
                             "n=580, 149 positives, 2000 bootstrap"),
            "scope": "SOURCING-side only — never ranking, never cross-family.",
            "read_status": ("DEFERRED. The labels this bar reads do not exist inside the "
                            "run; both scores are recorded on every row so the read is "
                            "taken when the batches come back."),
        },
        "maneuvers_on_admissions": {
            "read": "QUALITATIVE this run.",
            "why": ("the >=3 trigger boundary has no eval instrument, so there is nothing "
                    "to certify against"),
            "deliverable": ("the triggered-vs-fresh yield table + sheets — NOT a gate "
                            "decision, and the two yields are never pooled"),
        },
    }

    def write_run_config(self):
        """`run_config.json`: what this run is, and what it pre-registered, before batch 1."""
        # Load+verify the julia pool HERE, before the config is built. The stamp below records
        # the pool's own measured closest pair, and `seed_julia_pool` does not run until after
        # this — so reading the measurement lazily wrote `pool_min_dc: null` into every run
        # config, which is precisely the invisibility the stamp exists to end. Cached, so the
        # pre-loop draw pays nothing; and it moves the floor REFUSAL ahead of batch 1 rather
        # than into the middle of the first root draw.
        # ...but NOT on the dive path. A dive spawns no roots and never injects the pool, so
        # loading it there buys a stamp of something the run does not use and adds a refusal
        # (the floor check raises) to a run the pool cannot affect.
        if self.julia_seed_pool_path is not None and not self.dive:
            try:
                self.load_julia_supply_pool()
            except SystemExit:
                raise
            except Exception as exc:                       # noqa: BLE001
                # A malformed pool must not take the run config down with it — the refusal
                # path above is the one that matters, and it re-raises.
                print(f"[julia-seed-pool] could not measure pool for the run config: {exc}",
                      flush=True)
        cfg = dict(
            run_ts=self.run_dir.name, started=time.strftime("%FT%T"),
            mode="dive" if self.dive else "steered",
            scorer_version=ps.SCORER_VERSION, ckpt=ACTIVE_CKPT,
            families=self.families, partitions=self.partitions,
            family_weights=self.family_weights,
            budget_min=self.budget_s / 60.0, wall_budget_min=self.wall_budget_s / 60.0,
            # The CONFIGURED cap, not `root_draw_budget_s()`: run_config is the
            # pre-registration, and that method returns the value already clamped to whatever
            # wall happened to be left when it was called. (Calling it here also drags
            # `wall_elapsed_s` into the dive path's static reachability, which
            # `test_every_crawl_only_constructor_attribute_is_declared` correctly refuses.)
            root_draw_budget_min=round(
                float(getattr(self, "root_draw_budget_override", None)
                      or ROOT_DRAW_BUDGET_S) / 60.0, 2),
            tau_h=self.tau_h, tau_rec=self.tau_rec,
            tau_h_uncalibrated=self.tau_h_uncalibrated,
            good_floor=F.GOOD_FLOOR,
            record_and_rank=dict(
                frac=Q4_REC_FRAC, floor_abs=Q4_REC_FLOOR_ABS,
                record_canon_dups=self.record_canon_dups,
                store="q4_candidates.jsonl", fates=list(self.Q4_FATES),
                rank_tiers="2 = has a canonical decode; 1 = cheap score only; never pooled"),
            # SUPPLY LOOP: the two low-water marks and the julia c-spacing, stamped together
            # because the three of them are what decides whether the allocator's intent is
            # servable at all. Arm B of allocator_prereg_v1 was supply-bound on all three.
            root_supply=dict(
                global_low_water=self.B, partition_low_water=self.partition_low_water,
                refill_cooldown_batches=self.root_refill_cooldown,
                refill_share_cap=self.root_refill_share,
                scope="c-plane families only; julia twins and phoenix are pool/hook-fed",
                julia_hook_spacing=self.julia_hook_spacing,
                cspacing_floor=srt.CSPACING_FLOOR,
                spacing_reconciled=(self.julia_hook_spacing <= srt.CSPACING_FLOOR)),
            # CROSS-RUN SATURATION MEMORY (v1.7), stamped before batch 1 like every other
            # allocation-adjacent decision. `memory` is the index's own census, so a run that
            # started with an empty or truncated ledger union says so in its own config
            # instead of leaving it to be inferred from behaviour six weeks later.
            saturation_memory=dict(
                on=self.sat_on, radius_k=self.sat_k, strength=self.sat_strength,
                form="cheap_eord *= 1/(1 + strength * density)",
                scope=("breadth-leg candidate ordering inside one partition (push_children). "
                       "Root draws, maneuver-originated nodes and dive mode are EXEMPT; "
                       "cross-partition allocation is untouched."),
                radius_rule="a visit at fw shadows a disc of radius k*fw around ITSELF",
                decay="none — a visit from any past run counts the same as yesterday's",
                source="data/**/outcome_ledger.jsonl minus this run's own; frozen at start",
                build_s=round(self.sat_build_s, 2),
                memory=(self.sat_index.summary() if self.sat_index is not None else None),
                **({} if not self.dive else
                   {"n_a": "dive mode has no frontier to order; the index is not built"})),
            interior_gate=dict(on=self.interior_gate_on, threshold=self.interior_discard,
                               comparison="strict >",
                               measure="expand sidecar int_frac (= render::black_fraction)"),
            maneuvers_on_admissions=dict(
                on=self.trig_on, k=[str(k) for k in self.trig_ks],
                max_per_batch=self.trig_max_per_batch, nbh_m=self.trig_nbh_m,
                nbh_probes=self.trig_nbh_probes, period_max=self.trig_period_max),
            julia_seed_pool=(None if self.dive else
                             (str(self.julia_seed_pool_path)
                              if self.julia_seed_pool_path else None)),
            # The c-spacing this run's julia supply ACTUALLY carries — derived from the pool
            # file, not restated from the constant it is checked against. The floor was
            # invisible in every prior run config (only the path was stamped, and the path
            # does not say which floor thinned it), which is how harvest_v2_proving ran the
            # superseded 1e-2 without that being legible anywhere afterwards.
            julia_seed_pool_cspacing=dict(
                floor=srt.CSPACING_FLOOR,
                pool_min_dc=getattr(self, "_julia_pool_min_dc", None),
                verified_at_load=(self.julia_seed_pool_path is not None and not self.dive),
                explicit=self.julia_pool_explicit,
                **({"n_a": "dive mode spawns no roots and injects no pool"}
                   if self.dive else {})),
            phoenix_seed_pool=(str(self.phoenix_seed_pool_path)
                               if self.phoenix_seed_pool_path else None),
            prereg=self.PREREG,
            # WHICH SUPPLY CHANNEL FEEDS WHICH PARTITION, and the measurement that priced it.
            # Stamped from `supply_routing` rather than restated, so the config a reader
            # diffs against the labels is the same table the code routes on.
            supply_routing=srt.summary(),
        )
        if self.dive:
            # PRE-REGISTRATION PARITY WITH THE CRAWL PATH. `run_dive` used to write no
            # run_config.json at all, so a dive's tau_h, scorer version, checkpoint, interior
            # gate and the shape of its own plan existed only in `dive_state.json` (an
            # append-as-you-go checkpoint) and in a hand-written launch.txt. What a run
            # PRE-REGISTERED and what it later checkpointed are two different records, and only
            # the first is evidence about the decision.
            cfg["dive"] = dict(
                dive_source=str(self.dive_source),
                target_depth=self.dive_target_depth, min_fw=self.dive_min_fw,
                n_top=int(self.args.n_top), n_control=int(self.args.n_control),
                child_rule="cheap p_good argmax with a Gumbel tie-break at T=%g" % DIVE_NOISE_T,
                source_rule=("top-N source admissions by canonical p_good + M controls drawn "
                             "at random from the REST of the same admissions"),
                order_rule=("deficit sort (scheduler ON) WITHIN each arm, then "
                            "apportionment-sequenced across arms so every prefix carries both "
                            "to +/-1 — see interleave_dive_arms"),
                wall_budget="NOT SUPPORTED in dive mode; --wall-budget is refused, --budget "
                            "and the STOP sentinel are the only bounds",
                budget_min=self.budget_s / 60.0,
            )
        if self.quota is not None:
            # THE ALLOCATION IS ANNOUNCED AT DECISION TIME, NEVER DISCOVERED IN A READOUT
            # (`measurement_practice.md`). The intended mix, the currency it was computed
            # from, and the floor's recorded rationale all land here before batch 1, so the
            # realized-vs-intended read at the end is scored against a target that was
            # written down first.
            q = self.quota
            cfg["pop_quota"] = dict(
                floor=q.floor, julia_route_gain=q.julia_route_gain,
                currency="count(label==4) + 0.1*count(label==3), through the amendment "
                         "overlay + library",
                # The rule string is `pop_quota`'s own constant, not a second copy: this stamp
                # and the module's `summary()` described the target in two independent literals
                # until 2026-08-04, which is how a run_config can announce a target rule the
                # allocator stopped using.
                target_rule=q.target_rule,
                ratio=q.ratios, target={p: round(v, 3) for p, v in q.target.items()},
                anchor=(None if q.anchor is None else round(q.anchor, 3)),
                # The override file VERBATIM (None on the derived path). Recorded as passed,
                # not re-serialized from the parsed vector: a per-run instrument that moves the
                # whole allocation has to be readable back exactly as it was handed in.
                currency_targets_file=str(self.currency_targets_path)
                if self.currency_targets_path else None,
                currency_targets=q.targets_source,
                census=q.census.summary(), deficit={p: round(v, 3)
                                                    for p, v in q.deficit.items()},
                intended=q.allocation().summary(),
                seed_prices=q.cost.summary()["seed"],
                # WHICH table the seed came from. The prices alone cannot say whether they
                # were measured, regularized or flat, and those are three different
                # allocation policies that read identically as nine floats.
                seed_price_table=self.quota_prices_path,
                price="measured active-minutes per currency unit mined, EMA %.2f, clamped to "
                      "[seed/%.0f, seed*%.0f]" % (q.cost.ema, q.cost.clamp, q.cost.clamp),
                floor_rationale=(
                    "Every partition — including the q4-rich ones — receives a floor of ~5% "
                    "of total time. Spending 100% of the time on a stubborn deficit partition "
                    "means never learning anything new about the rich ones; the floor keeps "
                    "per-partition cost-to-mine prices fresh for the scheduler, keeps "
                    "rich-type material flowing to emission's diversity targets, and keeps "
                    "the cross-feed alive (rich-base admissions trigger maneuvers/hooks into "
                    "deficit partitions). It is a FLOOR, not a quota: a partition whose "
                    "deficit allocation already exceeds 5% gets nothing extra."),
                headline_metric="realized vs intended mix, per partition, in minutes / "
                                "candidates / admissions",
            )
        p = self.run_dir / "run_config.json"
        p.write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[prereg] run_config.json written BEFORE batch 1 -> {p}", flush=True)
        for k, v in self.tau_h_uncalibrated.items():
            print(f"[tau_h] {k}: {v}", flush=True)
        if self.family_weights:
            print(f"[alloc] family weights {self.family_weights} "
                  f"(root draws = B*F*w, B={self.B}, F={len(self.families)})", flush=True)
        return p

    def save_state(self):
        state = dict(
            run_ts=self.run_dir.name, families=self.families, julia_hook=self.julia_hook,
            seed=self.seed, B=self.B, budget_s=self.budget_s, tau_h=self.tau_h,
            lambda_m=self.lambda_m, beta=self.beta, recency_k=self.recency_k,
            morph_lo=self.morph_lo, morph_hi=self.morph_hi, anchor_src=self.anchor_src,
            node_ctr=self.node_ctr, seq=self.seq, batch_i=self.batch_i,
            active_s=self.active_s, est_batch_s=self.est_batch_s,
            wall_s=self.wall_elapsed_s(), wall_budget_s=self.wall_budget_s,
            expansions_per_root=self.expansions_per_root, totals=self.totals,
            frontier=self.frontier, rng=self.rng.bit_generator.state,
        )
        state["pool_cursor"] = self.pool_cursor
        # Refill accounting is CHECKPOINTED for the same reason `wall_s` is: a kill/resume
        # loop that reset the root-draw spend would reset the bound that caps it, and a
        # resumed run could then spend its whole share again every session.
        state["root_draw_s"] = self.root_draw_s
        state["last_refill_batch"] = self.last_refill_batch
        state["tau_rec"] = self.tau_rec
        state["sat_by_partition"] = self.sat_by_partition
        state["tau_h_uncalibrated"] = self.tau_h_uncalibrated
        state["family_weights"] = self.family_weights
        # The visited set is maneuver state under `--maneuvers`, but a TRIGGERED run writes
        # to it with maneuvers off, and losing it on a resume would re-push atoms the ledger
        # already carries under a second node id.
        if self.trig_on and not self.maneuvers:
            state["man_visited"] = sorted(self.man_visited)
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.quota is not None:
            state["pop_quota"] = self.quota.state_dict()
        if self.maneuvers:
            # The visited-atom set and the governor's region cache are the two pieces of
            # maneuver state a resume must not lose: without them a restart re-pays the
            # Newton cost for regions the killed run already probed and re-pushes nodes the
            # ledger already carries.
            # The screen cache and the range distribution join them for the same reason:
            # a resume without the cache re-spawns the engine for every nucleus already
            # measured, and a resume without the distribution restarts the percentile from
            # n=0 — which silently reverts the range prior to neutral for the next 8 rows.
            state["maneuvers"] = dict(
                quota=self.man_quota, ks=[("none" if k is None else k) for k in self.man_ks],
                lateral=self.man_lateral, probe_s=self.man_probe_s,
                visited=sorted(self.man_visited), governor=self.man_gov.state_dict(),
                passed_logged=sorted(self.man_passed_logged),
                neighborhood=self.man_nbh, nbh_m=self.man_nbh_m, nbh_n=self.man_nbh_n,
                nbh_probes=self.man_nbh_probes, range_prior=self.man_range_prior,
                range_gain=self.man_range_gain, screen_s=round(self.man_screen_s, 3),
                screens=self.man_screens.state_dict(),
                range_dist=self.man_range_dist.state_dict(),
                view_prior=self.man_view_prior, view_gain=self.man_view_gain,
                # The view cache checkpoints COMPACTLY (`maneuver_view_screen.STATE_KEYS`):
                # what resumed selection reads back, not every measure. The full record is
                # already durable twice — on the maneuver row and as the raw field — and a
                # per-batch rewrite of ~25 MB to restate it is a cost, not a safeguard.
                views=(self.man_views.state_dict() if self.man_views else None),
                comp_dist=self.man_comp_dist.state_dict(),
                view_fields=(dict(root=str(self.man_fields.root), n=self.man_fields.n)
                             if self.man_fields else None))
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
        # Absent on a pre-v1.4 checkpoint: 0.0 is correct there (no wall cap was in force),
        # and it is a DEFAULT for a key that did not exist, not tolerance of a missing one.
        self.wall_s_base = float(st.get("wall_s", 0.0))
        self.expansions_per_root = st["expansions_per_root"]; self.totals = st["totals"]
        self.frontier = st["frontier"]; self.tau_h = st["tau_h"]
        self.totals.setdefault("novelty_hits", 0)
        self.totals.setdefault("nov_scored", 0); self.totals.setdefault("sat_hits", 0)
        self.totals.setdefault("precanon_dup", 0); self.totals.setdefault("julia_hooks_skipped", 0)
        self.totals.setdefault("distinct_looks", 0)
        for k in MAN_TOTALS:
            self.totals.setdefault(k, 0)
        # The metered seed-pool cursor. Restored, not re-derived: it is the record of how
        # much of each pool this run has already consumed, and a resume that reset it would
        # re-inject roots the ledger already carries.
        self.pool_cursor = dict(st.get("pool_cursor") or {})
        self.root_draw_s = float(st.get("root_draw_s") or 0.0)
        self.last_refill_batch = {k: int(v)
                                  for k, v in (st.get("last_refill_batch") or {}).items()}
        # v1.6 keys, absent on any earlier checkpoint. `tau_rec` is RE-DERIVED from the
        # restored `tau_h` rather than restored, so the two can never come back out of step;
        # the counters default to 0 because they did not exist, which is a default for a
        # missing key and not tolerance of a missing one.
        self.tau_rec = derive_tau_rec(self.tau_h)
        for k in ("interior_gated", "interior_unmeasured", "q4_recorded",
                  "q4_recorded_below_tau_h", "trig_fired", "trig_atoms",
                  "trig_nodes_pushed", "trig_admitted", "trig_unavailable",
                  "trig_budget_skip", "trig_expanded", "trig_candidates",
                  "root_draw_truncated", "root_draw_timeouts",
                  "root_refills", "root_refill_families", "root_refill_deferred",
                  "sat_mem_scored", "sat_mem_discounted", "sat_mem_density_sum"):
            self.totals.setdefault(k, 0)
        self.sat_by_partition = dict(st.get("sat_by_partition") or {})
        if self.trig_on and not self.maneuvers:
            self.man_visited = set(st.get("man_visited") or [])
        if self.maneuvers and "maneuvers" in st:
            m = st["maneuvers"]
            self.man_visited = set(m.get("visited", []))
            self.man_passed_logged = set(m.get("passed_logged", []))
            self.man_probe_s = float(m.get("probe_s", 0.0))
            self.man_gov.load_state(m.get("governor", {}))
            self.man_screen_s = float(m.get("screen_s", 0.0))
            self.man_screens.load_state(m.get("screens") or {})
            self.man_range_dist.load_state(m.get("range_dist") or {})
            if self.man_views is not None:
                self.man_views.load_state(m.get("views") or {})
            self.man_comp_dist.load_state(m.get("comp_dist") or {})
        self.rng.bit_generator.state = st["rng"]
        # scheduler prices/caps reload from the checkpoint; caps re-open on resume (item 4). The
        # distinct-look tally reloaded from its npz in the scheduler's __init__.
        if self.scheduler is not None and "scheduler" in st:
            self.scheduler.load_state(st["scheduler"], reopen_caps=True)
        # The quota restores its REALIZED tally (a resume that reset it would re-serve the
        # partitions the previous session already paid for) but RE-CENSUSES its deficit from
        # the live corpus, because a sitting may have landed between sessions — see
        # `PopQuota.load_state`.
        if self.quota is not None and "pop_quota" in st:
            self.quota.load_state(st["pop_quota"], reopen_caps=True)
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
    #
    # WHAT THE CONSTRUCTOR HANDS THIS PATH AND IT CANNOT USE is declared at module scope in
    # `DIVE_IGNORES` (41 attributes, one reason each) and asserted by reachability in
    # `test_steered_frontier.py`. Add a knob that misses this loop and that test fails.
    # ---------------------------------------------------------------------- #
    def _load_source_admissions(self):
        led = self.dive_source / "outcome_ledger.jsonl"
        if not led.exists():
            raise SystemExit(f"--dive-source has no outcome_ledger.jsonl: {led}")
        rows = [json.loads(l) for l in open(led, encoding="utf-8") if l.strip()]
        adm = [r for r in rows if r.get("distinct") and ps.is_good(r)]
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
        return interleave_dive_arms(plan)

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
            1 for r in self.ledger.rows if r.get("distinct") and ps.is_good(r))
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
            print(f"[dive-order] arms apportionment-sequenced: "
                  f"{[e['start_group'][0] for e in plan]}", flush=True)
            print(f"[tau_h] {self.tau_h}", flush=True)
            # BEFORE dive 1, exactly as the crawl writes it before batch 1.
            self.write_run_config()
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
            # Same line as the crawl's, for the same reason: `dive_log` is keyed on the dive
            # and has no duration, so a dive that ran 9x the median was previously visible
            # only in the console. Recorded before the checkpoint, so a kill loses the resume
            # position rather than the timing of the dive that was in flight.
            self.stage_times.record(
                "dive", rec["dive_id"], dt, partition=rec.get("partition"),
                start_group=rec.get("start_group"), rungs=rec.get("rungs"),
                end_cause=rec.get("end_cause"), n_admitted=rec.get("n_admitted"))
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
            # THIS SESSION's stage totals (tools/stage_times.py). The per-unit rows are in
            # `stage_times.jsonl` beside this file; these are the roll-up, so a reader holding
            # only the summary gets stage totals and learns the per-unit stream exists.
            stage_times=self.stage_times.totals(),
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
        # Compress every live tail BEFORE the summary lands, so a finished run's committed
        # record is entirely `.jsonl.gz` segments (run_record.SegmentWriter.finalize) — then
        # land the summary and tear the run's scratch down. Both halves live in
        # `_close_summary`, shared with the crawl path.
        self._close_summary(summary)
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
        global ROOT_LOW_WATER, PARTITION_LOW_WATER
        ROOT_LOW_WATER = self.B
        PARTITION_LOW_WATER = self.partition_low_water
        if self.args.resume and self.state_path.exists():
            self.load_state()
        else:
            print(f"[fresh] run {self.run_dir.name}: families={self.families} "
                  f"julia_hook={self.julia_hook} budget={self.budget_s/60:.0f}m B={self.B} "
                  f"lambda_m={self.lambda_m} beta={self.beta} recency_k={self.recency_k}", flush=True)
            print(f"[root-refill] per-partition low-water={self.partition_low_water} "
                  f"(global={ROOT_LOW_WATER}), cooldown={self.root_refill_cooldown} batches, "
                  f"share cap={self.root_refill_share:.0%} of loop wall", flush=True)
            print(f"[dup-fix] julia_hook_spacing={self.julia_hook_spacing} "
                  f"(== supply_routing.CSPACING_FLOOR={srt.CSPACING_FLOOR}) "
                  f"freshness_prior={self.freshness_prior} "
                  f"(prior_rows={len(self.prior_rows)}) "
                  f"seeded_cloud_sizes={ {p: len(v) for p, v in self.clouds.items()} }", flush=True)
            if self.sat_on:
                m = self.sat_index.summary()
                print(f"[sat-memory] ON — k={self.sat_k} strength={self.sat_strength}; "
                      f"{m['visits']} prior visits over {m['ledgers']} ledgers / "
                      f"{m['identity_buckets']} identity buckets, built in "
                      f"{self.sat_build_s:.2f}s ({m['unusable_rows']} unusable rows); "
                      f"per-partition {m['partitions']}", flush=True)
            else:
                print(f"[sat-memory] OFF (--sat-strength {self.sat_strength}) — breadth "
                      f"priorities are v1.6-identical", flush=True)
            if self.lambda_m > 0.0:
                mode = f"recency (admitted + last {self.recency_k} batches)" if self.recency_k \
                    else "legacy (all-permanent)"
                print(f"[morph-anchors] lo={self.morph_lo:.4f} hi={self.morph_hi:.4f} "
                      f"({self.anchor_src}); memory={mode}, sat knee cos>={self.sat_cos:.4f}",
                      flush=True)
            print(f"[tau_h] {self.tau_h}", flush=True)
            if self.maneuvers:
                print(f"[maneuvers] ON — quota={self.man_quota} slots/batch (of AVAILABLE), "
                      f"probe_p={self.man_gov.p} k={self.man_ks} lateral={self.man_lateral} "
                      f"neighborhood={self.man_nbh}"
                      + (f" (m={self.man_nbh_m} n={self.man_nbh_n} "
                         f"probes={self.man_nbh_probes})" if self.man_nbh else ""),
                      flush=True)
                if self.man_view_prior:
                    p = self.man_view_params
                    print(f"[maneuvers] VIEW screen at {msc.fm.SCREEN_W}x{msc.fm.SCREEN_H} "
                          f"on each candidate's OWN frame, cap policy "
                          f"{msc.screen_policy_token()!r}; sort key composite_v3 "
                          f"(veto {p.veto}, caps {p.cap_range}/{p.cap_rings}, band "
                          f"{p.band_edge}^{p.band_exp:g}); gain {self.man_view_gain}; "
                          f"fields -> {self.man_fields.root} ({self.man_fields.n} present)",
                          flush=True)
                else:
                    print(f"[maneuvers] screen at {msc.fm.SCREEN_W}x{msc.fm.SCREEN_H} on the "
                          f"{msc.SCREEN_FRAME_MULT:g}x atom frame, cap policy "
                          f"{msc.screen_policy_token()!r}; range_prior="
                          f"{self.man_range_prior} (gain {self.man_range_gain})", flush=True)
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
            if self.quota is not None:
                a = self.quota.allocation()
                print(f"[pop-quota] ON — floor={self.quota.floor:.0%} of total time per "
                      f"partition (up to {self.quota.floor*len(self.partitions):.0%} floored); "
                      f"currency = n4 + 0.1*n3 through the amendment overlay + library",
                      flush=True)
                print(f"[pop-quota] census={ {p: round(v, 1) for p, v in self.quota.census.currency.items()} } "
                      f"(defaulted_rows={self.quota.census.defaulted_rows}, "
                      f"sources={self.quota.census.sources})", flush=True)
                print(f"[pop-quota] deficit={ {p: round(v, 1) for p, v in self.quota.deficit.items()} }",
                      flush=True)
                print(f"[pop-quota] INTENDED mix={ {p: round(v, 3) for p, v in sorted(a.share.items())} } "
                      f"floored={sorted(a.floored)}", flush=True)
            self.write_run_config()
            # THE PRE-LOOP DRAW, the one outside both caps. Bounded by `root_draw_budget_s`
            # like every other draw, and additionally TIMED and recorded — a cost that no cap
            # counts must at least be a number in the summary, or the next run's wall-clock
            # projection is built from a clock that never saw it.
            _pre_t0 = time.time()
            self.draw_roots()
            self.pre_loop_draw_s = time.time() - _pre_t0
            self.stage_times.record("root_draw", "preloop", self.pre_loop_draw_s,
                                    budget_s=round(self.root_draw_budget_s(), 1),
                                    frontier=len(self.frontier))
            print(f"[root-draw] pre-loop draw took {self.pre_loop_draw_s/60:.1f}m "
                  f"(outside the active and wall caps by design; bounded at "
                  f"{self.root_draw_budget_s()/60:.0f}m)", flush=True)
            # At fresh start these inject the WHOLE pool when `--seed-pool-rate` is 0 (the
            # historical behaviour) and only the first metered chunk when it is set.
            self.seed_julia_pool()          # PRIMARY julia supply: sampler-sourced roots (probe)
            self.seed_phoenix_pool()        # phoenix channel: skeleton-sampled parameter points
            self.save_state()

        session_t0 = time.time()
        while True:
            if self.stop_path.exists():
                print("[STOP] sentinel present — halting at batch boundary.", flush=True)
                break
            if self.budget_s and self.active_s + self.est_batch_s > self.budget_s:
                print(f"[budget] active {self.active_s/60:.1f}m + est batch "
                      f"{self.est_batch_s:.0f}s would exceed {self.budget_s/60:.0f}m — stopping.", flush=True)
                break
            # WALL-CLOCK CAP, and it is not a duplicate of the active cap. `active_s` counts
            # only the timed batch block, and `draw_roots` sits OUTSIDE it — a root
            # replenishment is minutes of real time that the active cap cannot see, so on a
            # replenishment-heavy run the wall clock outruns the active budget without limit.
            # For an unattended overnight run the wall clock is the constraint that actually
            # matters, so it gets its own cap, checked at the same boundary and against the
            # same "never start a unit that cannot finish" rule. Accumulated ACROSS resumes
            # (`wall_s` is checkpointed), or a kill/resume loop would reset the night's bound.
            self._session_t0 = session_t0
            if self.wall_exhausted():
                print(f"[wall] elapsed {self.wall_elapsed_s()/3600:.2f}h + est batch "
                      f"{self.est_batch_s:.0f}s would exceed the "
                      f"{self.wall_budget_s/3600:.2f}h wall cap — stopping "
                      f"(active {self.active_s/60:.1f}m).", flush=True)
                break
            # ROOT REPLENISHMENT, at two granularities. The global mark is the emergency one
            # and is unchanged (unbounded by cooldown or share — a frontier under B is a run
            # about to stop). The per-partition mark is the fix: it fires on a starved family
            # inside a frontier the global mark calls healthy, which is the state arm B spent
            # all 427 of its batches in.
            if len(self.frontier) < ROOT_LOW_WATER:
                _rt0 = time.time()
                self.draw_roots()
                self.root_draw_s += time.time() - _rt0
            else:
                self.refill_starved()
            # The metered z-plane pools top up PER BATCH, not per replenishment, and the
            # difference is the whole fix. `ROOT_LOW_WATER` is B (32) while the frontier runs
            # in the thousands, so replenishment almost never fires — hanging the meter on it
            # would starve both z-plane channels to nothing instead of front-loading them,
            # which is the same bug with the sign flipped. Per batch gives a steady trickle
            # the native descendants can actually compete with.
            if self.seed_pool_rate > 0:
                self.seed_julia_pool()
                self.seed_phoenix_pool()
            if not self.frontier:
                print("[frontier] empty and no fresh roots — stopping.", flush=True)
                break

            tb = time.time()
            self.batch_i += 1
            if self.quota is not None:
                batch = self.pop_batch_quota()
            elif self.scheduler is not None:
                batch = self.pop_batch_scheduled()
            else:
                batch = self.pop_batch()
            if not batch:
                # allocator: an empty pop with a non-empty frontier means every servable
                # partition is PRICE-capped -> reopen (redistribute demand) and retry.
                if self.quota is not None and self.frontier:
                    self.quota.cost.reopen_caps()
                    self.batch_i -= 1
                    continue
                if self.scheduler is not None and self.frontier:
                    self.scheduler.prices.reopen_caps()
                    self.batch_i -= 1
                    continue
                # everything capped; try fresh roots, else stop. Timed like every other draw
                # — an emergency draw is still wall clock the refill share has to see, or the
                # share reports less root-drawing than the run did.
                _ct0 = time.time()
                _cadd = self.draw_roots()
                self.root_draw_s += time.time() - _ct0
                if _cadd == 0:
                    print("[frontier] all roots capped (M) and no fresh seeds — stopping.", flush=True)
                    break
                self.batch_i -= 1
                continue
            self.fold_expanded_into_memory(batch)   # parents join morph memory before scoring
            self.totals["expanded"] += len(batch)
            self.totals["man_nodes_expanded"] += sum(1 for n in batch if n.get("man"))
            self.totals["trig_expanded"] += sum(1 for n in batch if n.get("triggered"))
            self.trig_fired_this_batch = 0        # the per-batch trigger budget resets here
            # Maneuvers are proposed off the rungs ABOUT TO BE EXPANDED and are INTERLEAVED
            # into this same walk — never a separate run, which would confound the move with
            # the run. They land on the frontier for a later batch, competing (with a
            # reserved floor) against ordinary nodes.
            self.propose_maneuvers(batch)
            rec0 = self._reconcile_snapshot()
            cands = self.expand_batch(batch)
            self.totals["candidates"] += len(cands)
            self.score_cheap(cands)
            # v1.6: the interior gate sits AFTER the cheap score and BEFORE the harvest, and
            # both halves of that placement are deliberate. After, so a gated candidate still
            # records what the head thought of it (the rule asserts class 1 — whether the
            # head agrees is a question worth being able to ask, and it is free here because
            # the cheap score is one batched GPU pass over the whole batch). Before, so a
            # gated candidate cannot consume a canonical confirmation render, a reframe, a
            # ledger row or a frontier slot — which is the entire saving.
            cands = self.interior_gate(cands)
            self.score_morph(cands)                  # embed + cos_max vs memory (parents incl.)
            self.harvest(cands)                      # admissions fold into memory
            self.push_children(cands)                # novelty penalty applied from cos_max
            self.morph.end_batch()                   # finalize recency block, evict > K (no-op legacy)
            self._reconcile_batch(rec0, len(cands))  # found == written + dropped_*, or exit loud

            dt = time.time() - tb
            self.active_s += dt
            self.est_batch_s = dt if self.est_batch_s == 0 else 0.5 * self.est_batch_s + 0.5 * dt
            # THE per-batch duration, on record. `quota_trace` cannot carry it — that row is
            # written at PICK time, before the batch it chose has run — and every other
            # per-row stream here is keyed on a candidate, so until this line `dt` existed
            # only in a console print and in the quota's aggregate minutes. Recorded BEFORE
            # the charges below so a kill between the two loses the accounting, not the
            # evidence of what the run was doing when it died.
            self.stage_times.record(
                "frontier_batch", f"batch:{self.batch_i}", dt,
                partition=self._served_partition, n_expanded=len(batch),
                n_cands=len(cands), admitted_cum=self.totals["admitted"])
            # scheduler: charge this batch's active time to the served partition (price EMA +
            # attempt-cap accounting). Cross-partition arithmetic only; no p_good.
            if self.scheduler is not None and self._served_partition is not None:
                self.scheduler.charge(self._served_partition, dt / 60.0)
            # POP QUOTA: the realized-time tally is what the quota corrects against, so the
            # charge is the single most load-bearing line in the loop — an uncharged batch is
            # a batch the quota believes never happened, and the realized mix would drift
            # exactly as far as the charges are missing. Candidates are attributed to the
            # partition they were EXPANDED under (not each candidate's own field), so the
            # per-denomination shares stay comparable with `discovery_pipeline.md` §3.1's
            # candidate-stream number.
            if self.quota is not None and self._served_partition is not None:
                self.quota.charge(self._served_partition, dt / 60.0)
                self.quota.note_candidates(self._served_partition, len(cands))
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
            wall_budget_min=self.wall_budget_s / 60.0,
            wall_min=round(self.wall_elapsed_s() / 60.0, 2),
            # The ratio the next run's ETA has to be projected from: active time is what the
            # budget bounds, wall clock is what the night bounds, and root replenishment is
            # the gap between them.
            wall_over_active=(round(self.wall_elapsed_s() / self.active_s, 2)
                              if self.active_s > 0 else None),
            # The pre-loop draw, which NO cap counts (`wall_elapsed_s` starts at the loop).
            # Recorded because an ETA projected from `wall_min` alone is short by exactly
            # this, and because a bound nobody can see the utilisation of is a bound nobody
            # can size. `None` on a resume — the pre-loop draw is a fresh-start cost.
            pre_loop_draw_min=(round(self.pre_loop_draw_s / 60.0, 2)
                               if getattr(self, "pre_loop_draw_s", None) is not None
                               else None),
            root_draw_budget_min=round(
                float(getattr(self, "root_draw_budget_override", None)
                      or ROOT_DRAW_BUDGET_S) / 60.0, 2),
            # IN-LOOP root draws (global replenishment + per-partition refill), and the share
            # of loop wall they took. The share is the affordability bound's own utilisation:
            # a run that reports it pinned at the cap spent its whole allowance on refills and
            # the next run's low-water/cooldown should be sized from that number.
            root_draw_min=round(self.root_draw_s / 60.0, 2),
            root_draw_share=(round(self.root_draw_s / (self.active_s + self.root_draw_s), 4)
                             if (self.active_s + self.root_draw_s) > 0 else None),
            # THIS SESSION's stage totals; per-unit rows in `stage_times.jsonl` beside this
            # file. `frontier_batch.total_s` is the same quantity `active_min` aggregates,
            # quoted per stage — a disagreement between them is a batch that ran uncharged.
            stage_times=self.stage_times.totals(),
            partition_low_water=self.partition_low_water,
            root_refill_cooldown=self.root_refill_cooldown,
            root_refill_share_cap=self.root_refill_share,
            final_queue_lens=self.queue_lens(),
            # Starved partitions the refill deliberately did NOT serve, each with its reason.
            # Reported beside `final_queue_lens` because the two together are the only way to
            # tell a partition nobody could feed from one nobody noticed.
            refill_deferred_partitions=self.deferred_partitions(),
            lambda_m=self.lambda_m, beta=self.beta, recency_k=self.recency_k,
            morph_mem=len(self.morph), morph_perm=self.morph.n_perm,
            morph_recency=self.morph.n_recency,
            morph_lo=self.morph_lo, morph_hi=self.morph_hi, anchor_src=self.anchor_src,
            sat_cos=round(self.sat_cos, 4),
            sat_frac=(round(self.totals["sat_hits"] / self.totals["nov_scored"], 4)
                      if self.totals.get("nov_scored") else None),
            saturation_memory=self.saturation_memory_summary(),
            active_min=round(self.active_s / 60.0, 2), batches=self.batch_i,
            tau_h=self.tau_h, totals=self.totals,
            cloud_sizes={p: len(v) for p, v in self.clouds.items()},
        )
        if self.quota is not None:
            summary["pop_quota"] = self.quota.summary()
            # The headline is lifted to the TOP level too. v1's realized 19.6% against an
            # intended 70% had to be reconstructed from a candidate stream after the fact;
            # this run's equivalent number is the first thing in its own summary.
            summary["realized_vs_intended"] = summary["pop_quota"]["mix"]
            summary["floor_vs_deficit"] = summary["pop_quota"]["floor_vs_deficit"]
            # THE UNSPENT-FLOOR ALARM, lifted to a screaming top-level key on the same idiom
            # as UNSEEDED_RUN. run 2 allocated julia:mandelbrot 17.8 floor minutes against a
            # full queue and spent 0.0 of them; that was recoverable only by reading a 361-row
            # trace, and a run whose floor never bought anything must say so in its own summary.
            uf = summary["pop_quota"]["unspent_floor"]
            if uf.get("alarms"):
                summary["UNSPENT_FLOOR_PARTITIONS"] = dict(
                    partitions=uf["alarms"], threshold=uf["threshold"],
                    allocated_min_per_partition=uf["allocated_min_per_partition"],
                    detail={p: uf["per_partition"][p] for p in uf["alarms"]},
                    why=("these partitions were ALLOCATED floor minutes and spent <= "
                         f"{(1 - uf['threshold']):.0%} of them. `servable_min` says which "
                         "kind of failure it is: near the run's total_min means the pop rule "
                         "declined to serve a partition it could have, near zero means "
                         "nothing could feed it (see refill_deferred_partitions)."))
        vf = self.finalize_view_fields()
        summary["maneuvers"] = self.maneuver_summary()
        if vf:
            summary["maneuvers"]["view_fields_finalized"] = vf
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
            # A --pop-quota run is ALSO scheduler-off, and its reason is different: the quota
            # denominates demand in HUMAN LABELS (n4 + 0.1*n3 through the amendment overlay
            # + library), not in distinct looks, so it has no look tally to seed and its
            # deficits are library-wide WITHOUT one. Saying "no deficit scheduler" for it
            # would read as the same missing-seed hazard the scheduler has, which it is not.
            reason = ("run has --pop-quota: demand is denominated in HUMAN LABELS, not in "
                      "distinct looks, so there is no look tally to seed and the deficits "
                      "are library-wide without one (see summary.pop_quota.currency)"
                      if self.quota is not None else
                      "run has --scheduler OFF: no deficit scheduler, therefore no "
                      "distinct-look tally and nothing to seed")
            summary["library_seed"] = dict(
                status="never_attempted", reason=reason,
                source=str(dsched.library_seed_paths()[0]),
                emb_dir=str(dsched.library_seed_paths()[1]),
                source_exists=dsched.library_seed_paths()[0].exists(),
                resolved_from=dsched.resolve_seed_source()[0])
        # Compress every live tail BEFORE the summary lands, so a finished run's committed
        # record is entirely `.jsonl.gz` segments (run_record.SegmentWriter.finalize) — then
        # land the summary and tear the run's scratch down. Both halves live in
        # `_close_summary`, shared with the dive path.
        self._close_summary(summary)
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
        sm = summary["saturation_memory"]
        if sm["status"] == "on":
            mem = sm["memory"] or {}
            print(f"  SATURATION MEMORY (k={sm['radius_k']} strength={sm['strength']}): "
                  f"{mem.get('visits')} prior visits over {mem.get('ledgers')} ledgers, "
                  f"loaded+indexed in {sm['build_s']:.2f}s")
            print(f"    discounted {sm['discounted']}/{sm['scored']} = "
                  f"{sm['discounted_frac']} of breadth candidates; mean density when "
                  f"discounted {sm['mean_density_when_discounted']}")
            for p, v in sm["by_partition"].items():
                print(f"    {p:20s} {v['discounted']:6d}/{v['n']:<6d} = {v['frac']} "
                      f"(mean density {v['mean_density_when_discounted']})")
        else:
            print(f"  saturation memory: {sm['status']}")
        if self.quota is not None:
            q = summary["pop_quota"]
            print(f"  POP QUOTA (floor {q['allocation']['floor']:.0%}): "
                  f"L1 mix gap = {q['mix']['l1_gap_minutes']:.1%} of minutes")
            print(f"    {'partition':20s} {'intend':>7} {'min':>7} {'cand':>7} {'admit':>7}")
            for p in self.partitions:
                m = q["mix"]
                print(f"    {p:20s} {m['minutes'][p]['intended']:7.3f} "
                      f"{m['minutes'][p]['realized']:7.3f} "
                      f"{m['candidates'][p]['realized']:7.3f} "
                      f"{m['admitted'][p]['realized']:7.3f}")
            f = q["floor_vs_deficit"]
            print(f"    floor {f['floor_min']:.1f}m ({f['floor_share']}) vs deficit "
                  f"{f['deficit_min']:.1f}m ({f['deficit_share']})")
            uf = q["unspent_floor"]
            if uf.get("alarms"):
                print(f"    !! UNSPENT FLOOR: {len(uf['alarms'])} partition(s) spent <= "
                      f"{(1 - uf['threshold']):.0%} of the "
                      f"{uf['allocated_min_per_partition']:.1f} floor minutes allocated "
                      f"to them")
                for p in uf["alarms"]:
                    d = uf["per_partition"][p]
                    print(f"       {p:20s} spent {d['spent_min']:7.2f}m of "
                          f"{d['allocated_min']:.1f}m  (servable {d['servable_min']:.1f}m = "
                          f"{d['servable_frac']:.0%} of the run)")
            else:
                print(f"    unspent-floor alarm: none "
                      f"(floor {uf['allocated_min_per_partition']:.1f}m/partition, "
                      f"carry trigger {uf['trigger_min']:.2f}m)")
            print(f"    price={q['cost']['price']} clamped={q['cost']['clamped']} "
                  f"capped={q['cost']['capped']}\n    trace -> {self.quota.trace_path}")
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
                  f"unfilled={m['quota_unfilled']} passed_over={m['quota_passed_over']}  "
                  f"probe+solve {m['probe_s']:.1f}s = "
                  f"{m['probe_share_of_active']:.3%} of active")
            print(f"    screen: {m['screened']} scored / {m['unscreenable']} unscreenable "
                  f"({m['screen_cache_hits']} cache hits) at "
                  f"{m['screen_geometry'][0]}x{m['screen_geometry'][1]} "
                  f"x{m['screen_frame_mult']:g} frame, policy {m['screen_policy']!r} "
                  f"— {m['screen_s']:.1f}s = {m['screen_share_of_active']:.3%} of active")
            print(f"    range_prior={m['range_prior']} (gain {m['range_gain']}, "
                  f"dist n={m['range_dist_n']})  neighborhood={m['neighborhood']} "
                  f"(m={m['nbh_m']} n={m['nbh_n']} probes={m['nbh_probes']}, "
                  f"passed_over={m['nbh_passed_over']})")
            if m["view_prior"]:
                vshare = (f"{m['view_vetoed'] / m['view_screened']:.1%}"
                          if m["view_screened"] else "n/a")
                print(f"    VIEW screen: {m['view_screened']} scored / "
                      f"{m['view_unscreenable']} unscreenable, {m['view_vetoed']} vetoed "
                      f"({vshare} of scored)")
                print(f"      composite dist n={m['view_dist_n']} gain={m['view_gain']}; "
                      f"fields cached {m['view_fields_cached']} -> {m['view_fields']}")
                print(f"      view_fit ({m['view_fit_model']}) scored "
                      f"{m['view_fit_scored']}/{m['view_screened']} = "
                      f"{m['view_fit_coverage']} — RECORDED ONLY, composite_v3 still sorts")
        print(f"  ledger -> {self.ledger.path}\n  summary -> {self.run_dir/'summary.json'}")

    def saturation_memory_summary(self) -> dict:
        """Did the cross-run discount do anything, and where — the §11 read.

        `status` is a WORD, not an absence: a run with the memory off and a run whose memory
        found nothing both write zeros, and those are opposite facts. Off says so; on with
        `discounted_frac == 0` means the ledger union genuinely does not overlap anything this
        run walked, which is the interesting negative result."""
        t = self.totals
        n, d = t.get("sat_mem_scored", 0), t.get("sat_mem_discounted", 0)
        return dict(
            status=("off" if not self.sat_on else
                    ("dive_n_a" if self.dive else "on")),
            radius_k=self.sat_k, strength=self.sat_strength,
            build_s=round(self.sat_build_s, 2),
            memory=(self.sat_index.summary() if self.sat_index is not None else None),
            scored=n, discounted=d,
            discounted_frac=(round(d / n, 4) if n else None),
            # Mean density AMONG THE DISCOUNTED, not over all scored: the zeros are already
            # reported by `discounted_frac`, and pooling them turns "how saturated is the
            # saturated territory" into a restatement of how much of it there is.
            mean_density_when_discounted=(round(t.get("sat_mem_density_sum", 0) / d, 2)
                                          if d else None),
            by_partition={p: dict(
                n=v["n"], discounted=v["discounted"],
                frac=round(v["discounted"] / v["n"], 4) if v["n"] else None,
                mean_density_when_discounted=(round(v["density_sum"] / v["discounted"], 2)
                                              if v["discounted"] else None))
                for p, v in sorted(self.sat_by_partition.items())},
        )

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
            quota_passed_over=t["man_quota_passed_over"],
            visited_atoms=len(self.man_visited), probe_s=round(self.man_probe_s, 2),
            probe_share_of_active=(self.man_probe_s / self.active_s
                                   if self.active_s > 0 else 0.0),
            # --- v1.4 ---
            neighborhood=self.man_nbh, nbh_m=self.man_nbh_m, nbh_n=self.man_nbh_n,
            nbh_probes=self.man_nbh_probes, nbh_passed_over=t["man_nbh_passed_over"],
            range_prior=self.man_range_prior, range_gain=self.man_range_gain,
            screened=t["man_screened"], unscreenable=t["man_unscreenable"],
            screen_cache_hits=t["man_screen_cache_hits"],
            screen_policy=msc.screen_policy_token(),
            screen_geometry=[msc.fm.SCREEN_W, msc.fm.SCREEN_H, msc.fm.SCREEN_SS],
            screen_frame_mult=msc.SCREEN_FRAME_MULT,
            screen_s=round(self.man_screen_s, 2),
            screen_share_of_active=(self.man_screen_s / self.active_s
                                    if self.active_s > 0 else 0.0),
            range_dist_n=len(self.man_range_dist.values),
            # --- v1.5: the view screen, counted APART from the atom screen above ---
            view_prior=self.man_view_prior, view_gain=self.man_view_gain,
            view_screened=t["man_view_screened"],
            view_unscreenable=t["man_view_unscreenable"],
            view_vetoed=t["man_view_vetoed"],
            view_fields_cached=t["man_view_fields_cached"],
            view_dist_n=len(self.man_comp_dist.values),
            view_params=(None if not self.man_view_prior
                         else dict(self.man_view_params._asdict())),
            view_fields=(None if self.man_fields is None else str(self.man_fields.root)),
            # --- harvest v2 §3: BOTH sourcing scores, and the COVERAGE of the second one.
            # `view_fit_scored` vs `view_screened` is the verification the prompt asks for —
            # "screened rows carry both scores" is a ratio, and a ratio nobody prints is a
            # claim nobody checked. `view_fit_model` is stamped so a later reader knows which
            # fit produced the column without re-deriving it.
            view_fit_model=(None if self.man_views is None or self.man_views.fit_model is None
                            else vfit.MODEL_ID_V11),
            view_fit_scored=(0 if self.man_views is None else self.man_views.n_view_fit),
            view_fit_coverage=((self.man_views.n_view_fit / t["man_view_screened"])
                               if (self.man_views is not None
                                   and t["man_view_screened"]) else None),
            view_fit_is_sort_key=False,     # RECORDED ONLY — composite_v3 still orders
            log=str(self.man_log),
        )

    def finalize_view_fields(self) -> dict | None:
        """Write the retrospective index beside the run's append-only field store.

        Called at the END of the run rather than per batch: `finalize` states what the store
        holds, and a store that is still growing has no such statement to make. It is
        idempotent and derived from the append index, so a resumed run re-finalizes over the
        larger population and a killed run simply leaves the store un-finalized (still
        readable through `RunFieldCache`, which is the format that tolerates a torn tail).
        """
        if self.man_fields is None:
            return None
        rep = self.man_fields.finalize()
        self.man_fields.close()
        return rep


def set_below_normal_priority() -> str:
    """Drop THIS process to BELOW_NORMAL, which every child inherits.

    A long unattended discovery run must yield to interactive work. This driver fans out
    `fractal-generator.exe` through several call sites (--expand, the confirmation renders,
    reframe's own renders), and on win32 a child inherits the parent's priority class — so
    lowering the driver once covers all of them, without threading `creationflags` through
    every launcher. The thread-count half of the pairing stays with the engine defaults
    (`corpus_common.DEFAULT_ENGINE_THREADS`); this is only the priority half.

    DELEGATES to `corpus_common.set_below_normal_priority`, which now holds the one
    definition — it lives beside `BELOW_NORMAL_PRIORITY_CLASS` and `default_creationflags`,
    the rest of the same pairing, and a second driver needed it. Kept as a name here because
    `main()` and the tests call it, and because a `from ... import` would read as a
    re-definition to the no-duplication scan. The ctypes subtleties (private WinDLL,
    `use_last_error`, the c_void_p pseudo-handle) moved WITH the code and are documented
    there; they are exactly what a copy of this silently loses.
    """
    from corpus_common import set_below_normal_priority as _impl
    return _impl()


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
    ap.add_argument("--partition-low-water", type=int, default=None,
                    help="a c-plane family with fewer than N frontier nodes is STARVED and "
                         "gets a targeted root draw, even when the global frontier is healthy "
                         "(default: B, one batch's worth). The global --batch low-water is "
                         "blind to this and arm B of allocator_prereg_v1 ran 427 batches with "
                         "8 of 9 queues empty and zero in-loop draws because of it.")
    ap.add_argument("--root-refill-cooldown", type=int, default=ROOT_REFILL_COOLDOWN,
                    help=f"batches a family waits between targeted refills (default "
                         f"{ROOT_REFILL_COOLDOWN}); stops a family whose depth-2 probe "
                         f"rejects everything from being re-drawn every batch.")
    ap.add_argument("--root-refill-share", type=float, default=ROOT_REFILL_SHARE,
                    help=f"cap on in-loop root-draw seconds as a share of loop wall "
                         f"(default {ROOT_REFILL_SHARE}); refills above it are deferred and "
                         f"counted (totals.root_refill_deferred). The global low-water "
                         f"emergency draw is NOT subject to it.")
    ap.add_argument("--phoenix-seed-pool", type=str, default=None,
                    help="the PHOENIX channel: a JSON list of {c_re,c_im,p_re,p_im,"
                         "zm1_re,zm1_im} parameter points from tools/phoenix/"
                         "phoenix_q4_seeds.py, injected as base-scale phoenix z-plane roots "
                         "at fresh start. Phoenix cannot be a --families entry (it has no "
                         "parameter plane to prospect); a seed pool is how it gets roots. "
                         "DEFAULT None => no phoenix channel.")
    ap.add_argument("--seed-pool-rate", type=int, default=0,
                    help="inject at most N entries per z-plane seed pool per ROOT "
                         "REPLENISHMENT instead of the whole pool at fresh start. 0 (the "
                         "default) is wholesale injection, byte-identical to every earlier "
                         "run. Wholesale is what let 534 julia + 96 phoenix roots take 95%% "
                         "of the candidate stream from the native partitions holding 70%% of "
                         "the allocation; metering also gives julia_c_sourcing.md's "
                         "run-to-the-knee-then-refill operating rule for free.")
    ap.add_argument("--tau-h-phoenix", type=float, default=None,
                    help="phoenix's harvest cut. There is no derivable value under the "
                         "active head (see __init__), so this is explicit and the run "
                         "config stamps it UNCALIBRATED. Default: floors.GOOD_FLOOR.")
    ap.add_argument("--julia-seed-pool", type=str, default=str(JULIA_SUPPLY_POOL),
                    help="PRIMARY julia supply: a JSON list of {c_re,c_im} c's from the "
                         "c-diverse near-∂M sampler, injected as julia:mandelbrot roots at fresh "
                         "start (bypasses the hook-spacing gate; requires 'mandelbrot' in --families). "
                         f"DEFAULT {JULIA_SUPPLY_POOL.name} — the pool thinned at the ADOPTED "
                         f"c-spacing floor. Whatever file is passed is verified against "
                         f"supply_routing.CSPACING_FLOOR at load; pass '' for hook-only julia.")
    ap.add_argument("--freshness-prior", action="store_true",
                    help="ENABLE the cross-run coordinate freshness prior: seed this run's DEDUP "
                         "clouds (pre-canonical + admission near-dup + steering) from prior-library "
                         "admitted coords. DEFAULT OFF — prior-ON sterilized the native-seed "
                         "rejection sampler.")
    ap.add_argument("--budget", type=float, default=45.0, help="active-time budget (minutes)")
    ap.add_argument("--wall-budget", type=float, default=0.0,
                    help="WALL-CLOCK cap in minutes (0 = off, the historical behaviour). "
                         "Not a duplicate of --budget: --budget counts only the timed batch "
                         "block, and root replenishment sits outside it, so a "
                         "replenishment-heavy run outruns its active budget in real time "
                         "without limit. Accumulates across resumes. Checked at the same "
                         "batch boundary, under the same never-start-a-unit-that-cannot-"
                         "finish rule. CRAWL MODE ONLY: run_dive() has its own loop and "
                         "checks only --budget and the STOP sentinel, so passing this with "
                         "--dive is REFUSED rather than silently ignored.")
    ap.add_argument("--root-draw-budget", type=float, default=None,
                    help="wall bound in MINUTES for one draw_roots call, overriding the "
                         f"{ROOT_DRAW_BUDGET_S // 60:.0f}-minute standing constant. The bound "
                         "that matters is the PRE-LOOP draw's: it runs before either cap's "
                         "clock starts, so it is spent whatever --budget/--wall-budget say, "
                         "and the constant is sized for an overnight run. A run whose TOTAL "
                         "commitment is a few hours should pass this. Default: the constant "
                         "(byte-identical to not passing it).")
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
    # --- v1.7 cross-run saturation memory (ON by default; --sat-strength 0 reproduces v1.6) ---
    ap.add_argument("--sat-radius-k", type=float, default=SAT_RADIUS_K,
                    help=f"CROSS-RUN SATURATION MEMORY radius multiple: a prior ledger visit "
                         f"at framewidth fw shadows a disc of radius k*fw around ITSELF "
                         f"(default {SAT_RADIUS_K}, calibrated by "
                         f"tools/atlas/sat_radius_calibrate.py). Larger => more of the plane "
                         f"counts as already-mined; past ~0.35 the discount fires on so much "
                         f"of phoenix that it stops being a gradient.")
    ap.add_argument("--sat-strength", type=float, default=SAT_STRENGTH,
                    help=f"magnitude of the visited-density discount on a BREADTH candidate's "
                         f"steering weight: cheap_eord *= 1/(1 + strength*density) "
                         f"(default {SAT_STRENGTH}). 0 disables the mechanism entirely — no "
                         f"ledger load, no index, priorities byte-identical to v1.6. Root "
                         f"draws, maneuver nodes and dive mode are exempt at any strength.")
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
    # --- v1.4: the richness screen selects (recording is unconditional) ---
    ap.add_argument("--maneuver-range-prior", action="store_true",
                    help="LET THE RICHNESS SCREEN SELECT: fill reserved quota slots by "
                         "descending radial_range, and replace the maneuver node's neutral "
                         "prior with a bounded range-percentile term. DEFAULT OFF — the "
                         "scores are recorded either way, and off is byte-identical "
                         "selection.")
    ap.add_argument("--maneuver-range-gain", type=float, default=MAN_RANGE_GAIN_DEFAULT,
                    help=f"magnitude of the range prior term; the term is "
                         f"gain*(percentile-0.5), i.e. bounded to +/-gain/2 around the "
                         f"neutral prior (default {MAN_RANGE_GAIN_DEFAULT})")
    # --- v1.5: the VIEW screen selects (recording is still unconditional) ---
    ap.add_argument("--maneuver-view-prior", action="store_true",
                    help="SUCCESSOR TO --maneuver-range-prior. Screen every candidate at the "
                         "frame it is actually PUSHED at (64x36, one field per atom x k) and "
                         "select on view_screen.composite_v3 instead of the 4x atom's "
                         "radial_range: fill reserved quota slots by descending composite, "
                         "rank the neighbourhood top-n by each atom's BEST framing, and "
                         "replace the neutral prior with a bounded composite-percentile "
                         "term. Replaces the atom screen rather than running beside it "
                         "(~3x the fields). Mutually exclusive with --maneuver-range-prior. "
                         "DEFAULT OFF — off is byte-identical selection.")
    ap.add_argument("--maneuver-view-gain", type=float, default=MAN_VIEW_GAIN_DEFAULT,
                    help=f"magnitude of the composite prior term; gain*(percentile-0.5), "
                         f"i.e. bounded to +/-gain/2 around the neutral prior "
                         f"(default {MAN_VIEW_GAIN_DEFAULT})")
    ap.add_argument("--maneuver-neighborhood", action="store_true",
                    help="ENABLE the third operator, neighborhood_expand: enumerate up to "
                         "m nearby nuclei around the solved nucleus, screen each, propose "
                         "the top n. DEFAULT OFF.")
    ap.add_argument("--maneuver-nbh-m", type=int, default=MAN_NBH_M_DEFAULT,
                    help=f"neighborhood_expand: ceiling on DISTINCT nuclei enumerated per "
                         f"call (default {MAN_NBH_M_DEFAULT})")
    ap.add_argument("--maneuver-nbh-n", type=int, default=MAN_NBH_N_DEFAULT,
                    help=f"neighborhood_expand: how many of them are proposed, by "
                         f"radial_range (default {MAN_NBH_N_DEFAULT})")
    ap.add_argument("--maneuver-nbh-probes", type=int, default=MAN_NBH_PROBES_DEFAULT,
                    help=f"neighborhood_expand: PROBE budget per call — the bound that "
                         f"actually binds, since 88%% of sheet-3's probes returned the "
                         f"parent (default {MAN_NBH_PROBES_DEFAULT})")
    ap.add_argument("--family-weights", type=str, default=None,
                    help="deficit-by-q4-gap root allocation, e.g. "
                         "'mandelbrot=0.176,multibrot3=0.281,multibrot4=0.284,"
                         "multibrot5=0.259'. Normalized; every family in --families must be "
                         "named (a missing one is an error, not a zero). Preserves the total "
                         "root budget and moves only its distribution. Lighter than "
                         "--scheduler, which allocates by distinct-look deficit against a "
                         "target measure and needs a library seed.")
    # --- v1.6: maneuvers-on-admissions (2026-08-03) ---
    ap.add_argument("--maneuvers-on-admissions", action="store_true",
                    help="fire the label-seeded harvest's operator pair (snap_at_seed -> "
                         "neighborhood_expand) on every ADMITTED c-plane location, live. An "
                         "admission is a location this run just decoded >=3, i.e. the same "
                         "kind of judged-good seed the offline harvest uses, available hours "
                         "before a label could arrive. Triggered yield is tallied SEPARATELY "
                         "from fresh-seed yield everywhere. DEFAULT OFF.")
    ap.add_argument("--trig-k", type=str, default=TRIG_K_DEFAULT,
                    help=f"framing set for a triggered maneuver (default "
                         f"{TRIG_K_DEFAULT!r}; NOT the walk's none,4,16 — the 4x atom "
                         f"question is already answered by the admission)")
    ap.add_argument("--trig-max-per-batch", type=int, default=TRIG_MAX_PER_BATCH_DEFAULT,
                    help=f"cap on triggers per batch — bounds the bill, where the admission "
                         f"rate bounds the opportunity (default {TRIG_MAX_PER_BATCH_DEFAULT})")
    ap.add_argument("--trig-nbh-m", type=int, default=mnv.NBH_MAX_FOUND)
    ap.add_argument("--trig-nbh-probes", type=int, default=mnv.NBH_MAX_PROBES,
                    help="the probe budget, which is the bound that actually binds "
                         "(88%% of sheet-3's probes returned the parent)")
    ap.add_argument("--trig-period-max", type=int, default=lsh.SEED_PERIOD_MAX)
    ap.add_argument("--trig-deadline-s", type=float, default=TRIG_DEADLINE_S)
    # --- v1.6: record-and-rank + the sourcing interior gate (2026-08-03) ---
    ap.add_argument("--record-canon-dups", action="store_true",
                    help="RENDER a pre-canonically-duplicated check anyway, so it carries "
                         "canonical scores into the record-and-rank store and can be "
                         "RANKED. It is still refused admission — dup-ness is a property of "
                         "the ledger's coordinate cloud, not of the picture. DEFAULT OFF, "
                         "because the render it gives up is large (campaign 2 skipped ~82%% "
                         "of checks this way); measure it in a shakedown before a long run.")
    ap.add_argument("--no-interior-gate", dest="interior_gate", action="store_false",
                    help="disable the sourcing interior gate (>0.30 interior => class 1, "
                         "Matt's rule). ON by default; off reproduces the pre-v1.6 "
                         "candidate population exactly.")
    ap.set_defaults(interior_gate=INTERIOR_GATE_DEFAULT)
    ap.add_argument("--interior-discard", type=float, default=_INTERIOR_DISCARD,
                    help=f"interior-fraction discard threshold, strict > "
                         f"(default {_INTERIOR_DISCARD}, the one value shared with "
                         f"label_seeded_harvest and apply_interior_rule)")
    # --- deficit scheduler (default OFF; scheduler-off is byte-identical to pre-change) ---
    ap.add_argument("--scheduler", action="store_true",
                    help="ENABLE the family-level deficit scheduler: cross-partition allocation "
                         "by price-weighted DISTINCT-LOOK deficit vs the target measure, instead "
                         "of a single global p_good queue. DEFAULT OFF (byte-identical).")
    ap.add_argument("--scheduler-prices", type=str, default=None,
                    help="seed price / cap / routing config (default data/atlas/scheduler_prices.json)")
    ap.add_argument("--allow-unseeded", action="store_true",
                    help="proceed with --scheduler even though the library look seed is absent "
                         "or empty. Deficits then measure RUN-LOCAL scarcity, not library-wide; "
                         "the run summary is permanently stamped library_seed.status=unseeded.")
    # --- pop quota (harvest v2; default OFF, and mutually exclusive with --scheduler) ---
    ap.add_argument("--pop-quota", action="store_true",
                    help="ENABLE the harvest-v2 POP QUOTA: per-partition allocation ENFORCED "
                         "at the population level (serve whichever servable partition is "
                         "furthest below its intended share of realized active time). "
                         "Deficit = shortfall of n4 + 0.1*n3 through the amendment overlay + "
                         "library against a UNIFORM target, price-weighted by measured "
                         "cost-to-mine. Supersedes --scheduler; DEFAULT OFF.")
    ap.add_argument("--quota-floor", type=float, default=pquota.FLOOR_FRAC,
                    help="universal per-partition floor as a fraction of TOTAL time "
                         "(default %(default)s). A floor, not a quota: a partition already "
                         "allocated above it gets nothing extra.")
    ap.add_argument("--quota-prices", type=str, default=None,
                    help="seed cost-to-mine prices + EMA/clamp/cap config for --pop-quota "
                         "(JSON; keys: prices, seed_price, price_ema, price_clamp, "
                         f"cap_minutes). DEFAULT: {QUOTA_PRICES_DEFAULT_REL} (the "
                         "median-shrunk seed). A MISSING table is fatal — the flat seed "
                         "price is never a silent fallback.")
    ap.add_argument("--currency-targets", type=str, default=None,
                    help="RUN-SCOPED override of the per-partition currency TARGET vector "
                         "(JSON; required key `targets` = {partition: currency}, everything "
                         "else recorded as provenance). MUTUALLY EXCLUSIVE with the derived "
                         "path: release_mix.RATIO and the richest-holding anchor are not read "
                         "at all, and the canonical ratio table is untouched. For a run whose "
                         "purpose is not the standing release mix — a label-deficient "
                         "partition that is also the census maximum has zero deficit under the "
                         "derived rule, and no reweighting of the ratios can move it. "
                         "DEFAULT: the derived path.")
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
    ap.add_argument("--retain-scratch", action="store_true",
                    help="keep <run_dir>/scratch after a clean close. Default is to delete "
                         "it: a 6h run leaves ~118 GB / 138k render+field files whose "
                         "verdicts are already in the ledger. Pass this only when you intend "
                         "to re-read the run's own tiles/fields. An interrupted or crashed "
                         "run keeps its scratch either way (see SCRATCH_TEARDOWN_KEY).")
    args = ap.parse_args()
    if args.below_normal:
        print(f"[priority] {set_below_normal_priority()}", flush=True)
    preflight_library_seed(args)
    SteeredFrontier(args).run()


if __name__ == "__main__":
    main()
