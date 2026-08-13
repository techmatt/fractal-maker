#!/usr/bin/env python
"""build_emission_diversity_v1.py — diversity-aware emission v1.

Deficit-driven colorize + resume-safe persistent pool + greedy release selection, built
against a steered-frontier run's run-scoped ledger (the first ledger whose rows decode as
current). The flow:

  1. INTAKE   — READ-TIME RANKED (tools/emission/ranked_intake.py, 2026-08-09). Admitted
                locations are `guard ∧ distinct ∧ (raw P(>=3) >= the 0.20 junk floor, or a
                floor-admit source)`, ranked best-first per partition on the stage-1 head's
                stored raw probability. NOT `current-decode ∧ q3` any more: a stored
                `decoded_class` is a per-partition derived threshold frozen at harvest time,
                and the stamp that guards it deleted the whole intake at the last head flip.
                Each admitted location gets a canonical morph-CLIP embedding and a within-type
                morph-cluster id, as before.
  2. BUDGET   — the colorize attempt budget is split HEAD-FIRST against release need
                (tools/emission/attempt_budget.py, 2026-08-09): `4 × that head's release
                slots`, both heads scaled down proportionally if the total budget cannot cover
                the pair, then split per partition by the same release-mix apportionment the
                release SLOTS use, and filled in rank order from the ranked intake. It used to
                fall out of the deficit model below, which spreads over a style axis carrying
                one smooth style against N strange ones — so smooth drew ~1/(N+1) of the
                attempts whatever the release asked for (3 of 30 in the selrestruct_1 smoke,
                against 6 smooth slots).
  3. COLORIZE — for each planned attempt the HEAD fixes the style set; within it the (palette
                flavor, render style) that maximizes the joint deficit is picked as before
                (softmax tie-break) over the joint counts of (partition × morph_cluster ×
                palette_flavor × render_style), whose target measure is DERIVED at intake from
                the canonical release-mix ratio table (tools/scoring/release_mix.py) re-solved
                against the live feasible cells (cells.py). Then the best palette in that
                flavor (pref ranker), render, and score with that head.
  4. POOL     — every SCORED candidate enters the append-only, resume-safe pool with full
                descriptor, head scores, realized palette statistics and provenance (pool.py).
                The permissive per-head POOL floor no longer admits or rejects; it rides along
                as the `above_pool_floor` annotation.
  5. SELECT   — RANK selection (selection.rank_select): top-N by the head's own p_ge3, per
                partition, under the partition's release_mix slot allocation crossed with the
                thin-supply emit cap `floor(passing_supply / 4)`, and a run-wide cap of 2 picks
                per morph cluster. Two disjoint per-head passes, sharing one cluster counter.

The four stamped cuts in `tools/emission/floors.py` are ANNOTATION-ONLY as of 2026-08-09
(prompts/selection_restructure_1.md): nothing in stages 4-5 removes a row on a per-head floor.
The one enforcing cut is the 0.20 junk floor at stage 1, where the colorize pool is drawn. The
retired floors' verdicts are recorded on every pool row, every release-record row and every
sheet tile (`would_pass_release_floor`), so what the old cut would have removed stays a number
in the record rather than a memory.

`--target-gated` therefore counts SCORED rows now, not rows above 0.90/0.50 — a weaker surplus
in exactly the way those floors were strong, and the counterfactual count is reported beside it.

See prompts/build_emission_diversity_v1.md and prompts/selection_restructure_1.md.

  # smoke: build to ≥3×N gated, select N=12, write report + sheets:
  uv run python tools/emission/build_emission_diversity_v1.py \
      --ledger data/discovery/steered_run2/outcome_ledger.jsonl --release-n 12
  uv run python tools/emission/build_emission_diversity_v1.py --resume ...   # after a kill
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "mining",
          ROOT / "tools" / "wallpaper", ROOT / "tools" / "scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import release_mix as RM                          # noqa: E402  THE release-mix ratio table
from tools import stage_times as stimes           # noqa: E402  THE per-unit stage timing stream
from tools.emission import attempt_budget as AB   # noqa: E402  THE colorize attempt budget
from tools.emission import emission_sinks as ESINKS  # noqa: E402  central sink-isolation
from tools.emission import floors as F           # noqa: E402  THE stage-2 cut owner
from tools.emission import descriptor as D       # noqa: E402
from tools.emission import cells as C            # noqa: E402
from tools.emission import selection as SEL       # noqa: E402
from tools.emission import ranked_intake as RI    # noqa: E402  read-time ranked intake
from tools.emission.pool import Pool             # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- geometry -------------------------------------------------------------- #
# Pool render res = the SMALLEST res that faithfully feeds every render consumer. Both quality
# heads deploy Transform(train=False) = 384×224 bicubic stretch, the pref pick scores on the
# cached 640×360 field, and realized-stats is a resolution-robust histogram — so the pool render
# only has to survive the 384×224 downsample. The scores-match guard (docs/design/
# pool-render-res-960.md) pinned 960×540 ss2: the mining head is res-robust everywhere (median
# |Δ|=0.007 vs the old 1280×720 ss2) and the wallpaper head — the 0.90 gate (retired to
# annotation-only 2026-08-09, but still the res-sensitive one), trained on
# 1280-sourced crops, hence res-sensitive — matches at 960 ss2 (median |Δ|=0.027, spearman 0.933,
# 0 release-floor flips) but NOT at 640 ss2 (max |Δ|=0.30, 5 flips). 960 ss2 is 0.56× the pixels
# of the old 1280×720 ss2 leftover corpus-crop default → ~1.8× faster, floors unchanged.
POOL_W, POOL_H, POOL_SS, POOL_FILT = 960, 540, 2, "lanczos3"      # head-scoring / pool render
REL_W, REL_H, REL_SS, REL_FILT = 2560, 1440, 4, "lanczos3"        # release full-res (wallpaper canon)
JPG_Q = 95

# Stage-2 cuts: IMPORTED from `tools/emission/floors.py`, the one owner. Each carries the
# head it reads and the head version it was set against, and refuses to gate when the live
# pin disagrees. This module used to declare all four as its own literals while three
# readouts declared them again — six copies of four numbers, none checked against each other.
# NONE OF THE FOUR CUT since 2026-08-09 — they annotate (floors.py). They are still resolved
# here, still written onto every row, and still stamped, because the annotation is only
# readable if the scale it was computed on is pinned. The one enforcing cut is `F.JUNK_FLOOR`,
# applied where the colorize pool is drawn (`ranked_intake`).
DEFAULT_FLOOR = F.WALLPAPER_POOL.value                    # 0.75  wallpaper-head POOL floor
DEFAULT_MINING_FLOOR = F.MINING_POOL.value                # 0.25  mining-head POOL floor
DEFAULT_RELEASE_FLOOR = F.WALLPAPER_RELEASE.value         # 0.90  smooth  → wallpaper gate
DEFAULT_MINING_RELEASE_FLOOR = F.MINING_RELEASE.value     # 0.50  strange → mining gate
DEFAULT_STRANGE_FRAC = 0.5    # target strange share of the release (render-mode split)
# `STRANGE_STYLE_WEIGHT` (0.5) was here until 2026-08-09: within the strange pass it floored
# `greedy_select`'s coverage kernel for same-render-style pairs, so the greedy spread across
# the promoted modes before doubling up on one. It went with `greedy_select`
# (prompts/selection_restructure_3.md). Nothing replaced it and nothing needs to: the live
# rule spreads the strange budget across modes UPSTREAM of selection, in
# `attempt_budget.plan` + `styles_for_head`, so the pass is already offered a mode-spread
# population instead of being asked to de-duplicate one afterwards.


# --------------------------------------------------------------------------- #
# Render-style axis + wallpaper render (reuse deploy_tail's roster + render dispatch).
# --------------------------------------------------------------------------- #
def _deploy_tail():
    from tools.mining import deploy_tail as dt
    return dt


def render_styles(dt) -> list:
    """smooth (base carrier) + the registry-promoted strange modes (deploy_tail ROSTER)."""
    return ["smooth"] + [m for (m, _kind) in dt.ROSTER]


def _roster_kind(dt, style: str):
    for (m, kind) in dt.ROSTER:
        if m == style:
            return kind
    raise KeyError(style)


def render_smooth(dt, cm, loc, palette, cp, out_path, w, h, ss, filt):
    """Smooth base carrier: dump the plain smooth field, apply the palette via the colormap
    tail (no --coloring spec). Mirrors deploy_tail.render_pure minus the field spec."""
    dt.FIELD_TMP.mkdir(parents=True, exist_ok=True)
    binp = dt.FIELD_TMP / f"{dt._field_stem(loc, 'smooth', w, h, ss)}.bin"
    lev = None
    try:
        dt._run([str(dt.EXE), "render-one"] + dt._locflags(loc) + [
            "--width", str(w), "--height", str(h), "--supersample", str(ss),
            "--dump-field", str(binp)])
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        ptype = dt.lib().palette_type(palette)
        phase = cp["phase"] if ptype == "cyclic" else 0.0
        ncyc = cp["n_cycles"] if ptype == "cyclic" else 1
        cfg = cm.CandidateConfig(palette=palette, location=fld.location, eval_width=ow,
                                 eval_height=oh, reverse=cp["reverse"], log_premap=cp["log_premap"],
                                 gamma=cp["gamma"], phase=phase, n_cycles=ncyc,
                                 transfer=cp["transfer"], transfer_gamma=cp["transfer_gamma"],
                                 filter=filt)
        prep = cm.stretch_field(fld)
        img = cm.render_candidate(fld, cfg, dt.lib(), prep=prep)
        # BAND AUTO-LEVEL (switch default ON since 2026-08-11) — smooth is a palette-mapped
        # mode like any
        # other, so the base carrier is in scope. Same seam as deploy_tail's pure path: the
        # re-render is another LUT over the SAME cached field.
        lev = dt._level_python(img, palette, out_path,
                               lambda ovr: cm.render_candidate(fld, cfg, ovr, prep=prep))
        dt._save(lev.img, out_path)
    finally:
        binp.unlink(missing_ok=True)
        binp.with_suffix(".json").unlink(missing_ok=True)
    return dt._info({}, lev)


def render_wallpaper(dt, cm, loc, style, palette, out_path, w, h, ss, filt) -> dict:
    """One production wallpaper render. Returns the render's info block — `{"autolevel":
    <stamp>}` under the shipped switch (ON since 2026-08-11), empty only with the switch
    forced off (`FRACTAL_AUTOLEVEL=0`)."""
    cp = dt._color_params({})       # canonical inherited coloring (transfer=pct, γ1, no reverse)
    if style == "smooth":
        return render_smooth(dt, cm, loc, palette, cp, out_path, w, h, ss, filt)
    return dt.render_candidate(loc, style, _roster_kind(dt, style), palette, cp,
                               out_path, w, h, ss, filt)


# --------------------------------------------------------------------------- #
# Realized palette statistics (hue/chroma histogram of the ACTUAL render).
# --------------------------------------------------------------------------- #
HUE_BINS, CHROMA_BINS = 12, 8
BLACK_V = 0.06


def realized_palette_stats(jpg_path: Path) -> dict:
    im = np.asarray(Image.open(jpg_path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    chroma = mx - mn
    v = mx
    # hue in [0,1)
    hue = np.zeros_like(mx)
    nz = chroma > 1e-6
    with np.errstate(invalid="ignore"):
        rc = np.where(mx == r, (g - b) / np.where(chroma == 0, 1, chroma), 0)
        gc = np.where(mx == g, 2.0 + (b - r) / np.where(chroma == 0, 1, chroma), 0)
        bc = np.where(mx == b, 4.0 + (r - g) / np.where(chroma == 0, 1, chroma), 0)
    h6 = np.where(mx == r, rc, np.where(mx == g, gc, bc))
    hue = (h6 / 6.0) % 1.0
    hue = np.where(nz, hue, 0.0)
    nonblack = v >= BLACK_V
    black_fraction = float(1.0 - nonblack.mean())
    mask = nonblack & nz
    if mask.sum() > 0:
        hue_hist, _ = np.histogram(hue[mask], bins=HUE_BINS, range=(0, 1),
                                   weights=chroma[mask])
        hh = hue_hist / (hue_hist.sum() + 1e-9)
        chroma_hist, _ = np.histogram(chroma[nonblack], bins=CHROMA_BINS, range=(0, 1))
        ch = chroma_hist / (chroma_hist.sum() + 1e-9)
        mean_chroma = float(chroma[nonblack].mean())
    else:
        hh = np.zeros(HUE_BINS)
        ch = np.zeros(CHROMA_BINS)
        mean_chroma = 0.0
    return {
        "hue_hist": [round(float(x), 5) for x in hh],
        "chroma_hist": [round(float(x), 5) for x in ch],
        "mean_chroma": round(mean_chroma, 5),
        "black_fraction": round(black_fraction, 5),
    }


# --------------------------------------------------------------------------- #
# Palette ranker (pref-v3-gvo best-in-flavor; deterministic fallback).
# --------------------------------------------------------------------------- #
class PaletteRanker:
    """Best concrete palette IN a flavor for a location, by the deployed pref-v3-gvo head
    (conditioned_colorize.Scorer) scored on the location's cached 640×360 smooth field. If
    the pref stack cannot load, falls back to a deterministic representative (first pool
    palette in the flavor) so the pipeline still runs — the head floor still gates quality."""

    def __init__(self, dt, cell_to_names: dict, lib, pick_mode: str = "pref",
                 deficit_lambda: float = 1.5):
        self.dt = dt
        self.lib = lib
        self.cell_to_names = cell_to_names
        self.cache: dict = {}                        # (loc,flavor) -> (members, scores|None)
        self.pref = None
        self.canonical_config = None
        self.pick_mode = pick_mode                   # "pref" (argmax) | "deficit" (spread)
        self.lam = float(deficit_lambda)
        self.sigs: dict | None = None                # intrinsic palette signatures (deficit)
        if pick_mode == "deficit":
            from tools.emission import palette_deficit as pdf
            self.sigs = pdf.signatures_from_lib(lib)
        try:
            from tools.studies import conditioned_colorize as cond
            self.pref = cond.Scorer()
            self.canonical_config = cond.canonical_config
            self.mode = f"pref:{self.pref.name}"
        except Exception as e:                       # noqa: BLE001
            print(f"[ranker] pref scorer unavailable ({e!r}); deterministic fallback", flush=True)
            self.mode = "deterministic"
        if pick_mode == "deficit":
            self.mode += f"+deficit(λ={self.lam})"

    # cap the per-flavor candidate set so one colorize's pref pass stays cheap (a flavor
    # holds up to ~90 pool palettes; 32 is ample to pick a good one and bounds cost).
    MAX_PALETTES = 32

    def _members(self, flavor: str) -> list:
        members = [p for p in self.cell_to_names.get(flavor, []) if p in self.lib.colormaps]
        return members[:self.MAX_PALETTES]

    def _members_scores(self, loc_id, flavor, field_bin, field_json):
        """(members, v3-gvo scores) for a (loc,flavor), scoring cached (the expensive part).
        scores is None when the pref scorer is unavailable (deterministic fallback)."""
        key = (loc_id, flavor)
        if key in self.cache:
            return self.cache[key]
        members = self._members(flavor)
        if not members:
            self.cache[key] = (None, None)
            return None, None
        if self.pref is None:
            self.cache[key] = (members, None)
            return members, None
        from tools import colormap as cm
        field = cm.load_field(field_bin, field_json)
        prep = cm.stretch_field(field)
        cfield = cm.coarse_field(prep)
        cfgs = [self.canonical_config(field, pn) for pn in members]
        imgs = cm.render_candidates_coarse(cfield, cfgs, self.lib)   # batched (K,h,w,3)
        scores = [float(s) for s in self.pref.score(imgs)]
        self.cache[key] = (members, scores)
        return members, scores

    def best(self, loc_id: str, flavor: str, field_bin: str, field_json: str, tracker=None):
        """Pick one palette in `flavor` for the location. `pref` mode = v3-gvo argmax (the
        baseline). `deficit` mode = the running realized chroma×hue deficit fills the pick
        with v3-gvo as the within-deficit tiebreaker (palette_deficit.pick)."""
        members, scores = self._members_scores(loc_id, flavor, field_bin, field_json)
        if members is None:
            return None, None
        if self.pick_mode == "deficit" and tracker is not None and self.sigs is not None:
            from tools.emission import palette_deficit as pdf
            i = pdf.pick(members, scores, self.sigs, tracker, lam=self.lam)
        else:
            i = 0 if scores is None else int(np.argmax(scores))
        return members[i], (None if scores is None else scores[i])


# --------------------------------------------------------------------------- #
# Head scoring — per render style.
#
# The prompt specifies "the wallpaper head" (v3, 0.90 production gate). That head was
# trained on SMOOTH wallpapers only and scores strange fields (tia/stripe/composite)
# ~0, which would collapse the render-style descriptor axis to smooth in the gated pool.
# The repo already gates the two render classes with two heads: the wallpaper head for
# smooth, the MINING head (render_mode_head/v1, 0.50 gate) for the promoted strange
# modes (deploy_tail). We therefore route each render style to its own head and apply a
# permissive floor below THAT head's production gate. Quality is only ever compared
# within a niche (which pins the style, hence the head), so the two heads never mix in a
# single comparison. This is the one place §4 is extended beyond the literal "wallpaper
# head"; it is flagged in the report.
#
# The style→head routing itself lives in `floors.head_for_style` (the cut owner has to know
# it to answer `for_style`); the short names below are this module's own vocabulary for the
# same split, kept because they are written into every pool row's `head` column.
# --------------------------------------------------------------------------- #
WALLPAPER_STYLES = F.WALLPAPER_STYLES


def head_for_style(style: str) -> str:
    return "wallpaper" if F.head_for_style(style) == F.WALLPAPER_HEAD else "mining"


class Heads:
    def __init__(self):
        import torch
        from tools.wallpaper import emit_v1
        from tools.mining.mining_gate import MiningScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wp_score, _cfg = emit_v1.load_v2_scorer(device)
        self.wp_gate = emit_v1.GATE_THRESHOLD                 # 0.90
        self.mining = MiningScorer()
        self.mining_gate = self.mining.threshold             # 0.50

    def score(self, style: str, jpg_path: Path) -> dict:
        head = head_for_style(style)
        if head == "wallpaper":
            _cond, marg, ssum = self.wp_score([str(jpg_path)])
            return {"head": "wallpaper", "gate": self.wp_gate, "p_ge2": float(marg[0, 0]),
                    "p_ge3": float(marg[0, 1]), "ssum": float(ssum[0])}
        ms = self.mining.score_paths([str(jpg_path)])[0]
        return {"head": "mining", "gate": self.mining_gate, "p_ge2": float(ms.p_ge2),
                "p_ge3": float(ms.p_ge3), "ssum": float(ms.score)}


# --------------------------------------------------------------------------- #
# The driver.
# --------------------------------------------------------------------------- #
class EmissionDiversity:
    def __init__(self, args):
        self.args = args
        # Multi-ledger intake: a release can draw on more than one run-scoped ledger
        # (e.g. steered_run2 + the dive) — every row is still admitted independently and
        # carries its own source ledger in provenance. `self.ledger` stays the first for
        # legacy single-ledger call sites / short report strings.
        ledger_args = args.ledger if isinstance(args.ledger, (list, tuple)) else [args.ledger]
        self.ledgers = [Path(x).resolve() for x in ledger_args]
        self.ledger = self.ledgers[0]
        self.out = Path(args.out).resolve()
        self.report_path = Path(args.report).resolve() if args.report else \
            (ROOT / "scratch" / "emission_v1_report.md")
        self.renders = self.out / "renders"
        self.release_dir = self.out / "release"
        self.field_cache = self.out / "fields"
        self.embs_path = self.out / "morph_embs.npz"
        self.intake_path = self.out / "intake.json"
        # The released library this run's intake is deduplicated AGAINST (its per-type medoids
        # seed the clustering). None -> `descriptor.DEFAULT_LIBRARY_DIR`. There is NO opt-out:
        # `--library ""` used to advertise one and never had it (the `or DEFAULT_LIBRARY_DIR`
        # below put the default straight back), and the seed load is fail-closed now.
        self.library_dir = (Path(args.library).resolve()
                            if getattr(args, "library", None) else None)
        self.colorize_log = self.out / "colorize_log.jsonl"
        self.floor = float(args.floor)                 # wallpaper-head POOL floor (smooth)
        self.mining_floor = float(args.mining_floor)   # mining-head POOL floor (strange styles)
        self.release_floor = float(args.release_floor)              # wallpaper-head RELEASE floor
        self.mining_release_floor = float(args.mining_release_floor)  # mining-head RELEASE floor
        # The stamp check, run once per run rather than per gated row: every cut this run
        # applies is a point on a head's probability scale, and it refuses if the live pin
        # has moved off the version the value was set against. A CLI override is UNSTAMPED by
        # construction (nobody derived it against anything), so it is named in the log rather
        # than silently inheriting the owner's stamp.
        F.check_stamps()
        self.floor_overrides = {
            name: v for name, v, own in (
                ("pool_wallpaper", self.floor, F.WALLPAPER_POOL.value),
                ("pool_mining", self.mining_floor, F.MINING_POOL.value),
                ("release_wallpaper", self.release_floor, F.WALLPAPER_RELEASE.value),
                ("release_mining", self.mining_release_floor, F.MINING_RELEASE.value))
            if v != own}
        self.release_n = int(args.release_n)
        self.strange_frac = float(args.strange_frac)   # target strange share of the release
        # release render geometry — default = wallpaper canon (REL_W/H/SS/FILT); overridable
        # for a fast judge pass (e.g. 1024×576 ss2). Defaults keep batch reproducibility.
        self.rel_w = int(getattr(args, "release_w", None) or REL_W)
        self.rel_h = int(getattr(args, "release_h", None) or REL_H)
        self.rel_ss = int(getattr(args, "release_ss", None) or REL_SS)
        self.rel_filt = getattr(args, "release_filt", None) or REL_FILT
        # target counts POST-FLOOR rows (release-eligible AND above their head's release
        # floor — `post_floor()`), so the colorize builds a 3×N surplus of genuinely
        # release-grade candidates, not merely pool-admitted ones (§4 "3× post-floor
        # surplus"). With every floor enforcing the two sets coincide; the accounting still
        # computes both, because a divergence is what a report-only floor looks like.
        self.target_gated = args.target_gated or (3 * self.release_n)
        self.max_attempts = int(args.max_attempts)
        self.time_budget_s = float(args.time_budget_min) * 60.0
        # cover-all: colorize every admitted location exactly once, then stop. Explicit
        # one-pass semantics — bypasses the surplus-building target/attempt/time cutoffs
        # (which invite the round-robin double-dip when hand-encoded as huge --target-gated).
        self.cover_all = bool(getattr(args, "cover_all", False))
        # within-flavor palette pick: "pref" = v3-gvo argmax (baseline, batch-stable default);
        # "deficit" = serve the running realized chroma×hue deficit, v3-gvo tiebreaker.
        self.palette_pick = getattr(args, "palette_pick", "pref")
        self.deficit_lambda = float(getattr(args, "deficit_lambda", 1.5))
        self.deficit_green_boost = float(getattr(args, "deficit_green_boost", 1.6))
        self.seed = int(args.seed)
        # SINK ISOLATION, decided and asserted here — before intake, before the first write.
        # Everything under `--out` is scratch by construction; three sinks are not
        # (`data/emission/{release_records,mining_gate_reports,run_telemetry}`). The first two
        # UPSERT-AND-ACCUMULATE, so a throwaway run adds rows a later calibration pass cannot
        # tell from a real release's; the third is this run's own timings and accumulates
        # nothing, but a throwaway run still must not write `data/`. `--ephemeral` redirects
        # all three under scratch/; `--record-root` names the root explicitly. See
        # tools/emission/emission_sinks.py.
        self.ephemeral = bool(getattr(args, "ephemeral", False))
        self.record_root = ESINKS.resolve_record_root(
            ROOT, smoke=self.ephemeral, explicit=getattr(args, "record_root", None),
            run_id=self.out.name)
        if self.ephemeral or getattr(args, "record_root", None):
            ESINKS.use(self.record_root)
            sinks = ESINKS.assert_isolated(ROOT, self.record_root, self.RECORD_SITE,
                                           run_id=self._run_id())
            print(f"[sinks] EPHEMERAL — record sinks isolated under "
                  f"{self.record_root}; nothing writes data/emission/. "
                  # parent/name, not name: the run-telemetry sink is a DIRECTORY named after
                  # the run, and printed bare it reads as a stray file next to two .jsonl.
                  + " ".join(f"{p.parent.name}/{p.name}" for p in sinks), flush=True)
        else:
            ESINKS.use(None)
            print(f"[sinks] PRODUCTION — records accumulate under "
                  f"{ESINKS.default_record_root(ROOT)}", flush=True)
        # Per-unit stage timing (tools/stage_times.py). This builder had NO timing at all
        # before 2026-08-12 — not per attempt, not per stage — so a run that spent 40 of its
        # 75 wall minutes in intake looked identical to one that spent them colorizing.
        # Additive: nothing reads it back, and no cutoff consults it.
        #
        # DURABLE SINCE 2026-08-13 (Matt), and built HERE rather than beside the other `--out`
        # paths above because it is resolved through the sink binding, which is decided
        # immediately above: production lands in `data/emission/run_telemetry/<run>/`, an
        # ephemeral run under its bound scratch root. Before this the stream lived under
        # `--out` and wiped with `rm -r scratch/*` while the discovery half was committed —
        # the same telemetry, two storage classes, decided by which leg wrote it.
        self.stage_times = stimes.StageTimes(ESINKS.stage_times_home(ROOT, self._run_id()))
        for d in (self.out, self.renders):
            d.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(self.seed)
        self.pool = Pool(self.out)
        # THE LOCATION RANKER IS GONE, PERMANENTLY (2026-08-08, Matt). `pref_loc_v1` never
        # existed on this checkout and its rebuild was DELETED rather than left blocked: the
        # blind-read manifest keys that joined its 379 labels to locations were wiped from
        # `scratch/` and are not re-derivable (deferred_recalibration.md § "Ranker rebuild —
        # DELETED"). These stay as empty dicts because they are WRITTEN into every emission
        # row's provenance block (`ranker_score` / `ranker_pct`), and a row schema that
        # changes shape depending on which era wrote it is worse than a null.
        #
        # Not to be confused with the PALETTE ranker (pref-v3-gvo, `colorize(..., ranker, ...)`)
        # which is live, deployed and untouched by this.
        self.ranker_score: dict = {}     # permanently empty — no location ranker is deployed
        self.ranker_pct: dict = {}       # permanently empty
        self.ranker_mode = "none"        # was "unavailable" when a head was still expected
        # within-round colorize order (pick_location): the round currently being served and
        # the id queue for it. Derived from the DURABLE pool counts, so a resume rebuilds
        # rather than restores.
        self._round_idx = None
        self._round_queue = None
        # read-time rank + supply census, filled by `intake()` -> `_index_ranks`. Declared
        # here so the attributes exist for the paths that never intake (a report rebuild, a
        # pool-only test) rather than each reader guarding with getattr.
        self.ranked: dict = {}
        self.rank_of: dict = {}
        self.mined_rows: list = []
        self.intake_scope = None      # None = the whole union; a set = the snapshot's ids
        self.mined_supply: dict = {}
        self.passing_supply: dict = {}
        self.good_supply: dict = {}      # above `floors.GOOD_FLOOR` — the guarantee's trigger
        self.emit_caps: dict = {}
        # per-selected-id slot provenance ("guarantee" | "mix"), filled by `select_release`
        # and read by the release record. Declared here for the paths that never select.
        self.slot_source: dict = {}
        # the colorize attempt budget (`attempt_budget.plan`), filled by `plan_attempts()`.
        # Declared here for the same reason as the census above: `--select-only` and a report
        # rebuild never plan, and every reader takes the empty dict rather than a getattr.
        self.attempt_plan: list = []
        self.attempt_budget: dict = {}

    # ---- intake ---------------------------------------------------------- #
    @staticmethod
    def _source_tag_of(row) -> str | None:
        """Durable intake source tag carried on the ledger row. The classic-phoenix supply
        stamps `mix_source`; older intake tooling used `_source_tag`. None when untagged (a
        row that no source-tag override can name)."""
        return row.get("mix_source") or row.get("_source_tag")

    def _load_all_admitted(self) -> list:
        """Admitted rows across every ledger, READ-TIME RANKED (`ranked_intake`, 2026-08-09).

        Still `descriptor.load_union_admitted` underneath — THE union reader, so row identity
        is namespaced by ledger (the 11 run-scoped campaign1/campaign2 id collisions do not
        alias) and deduplication is by LOCATION identity (`descriptor.loc_key`). What changed
        is the PREDICATE it is given: guard ∧ distinct ∧ (junk floor ∨ floor-admit), with no
        decode-version predicate and no stored-`decoded_class` q3 gate. See `ranked_intake`.

        Returns the rows in UNION ORDER, not rank order, and that is deliberate: ledger order
        is the stable incremental order `assign_morph_clusters` clusters in, so ranking here
        would silently re-index every in-batch morph cluster. The rank lives beside the rows
        (`self.ranked`, `self.rank_of`) and is consumed by `pick_location`."""
        # ONE union walk. `mined_rows` is the PRE-FLOOR population and is kept, because it is
        # the denominator of every supply line and cannot be recovered from the passing subset.
        self.mined_rows, mdiag = RI.load_mined(self.ledgers)
        ranked, diag = RI.ranked_from_mined(self.mined_rows, mdiag)
        rows = [r for r in self.mined_rows if RI.passes(r)]     # union order, not rank order
        self.intake_diag = diag
        self.ranked = ranked
        print(f"[intake] union over {len(self.ledgers)} ledger(s): {diag['n_mined']} mined "
              f"(guard ∧ distinct), {diag['n_passing']} above the {diag['junk_floor']} junk "
              f"floor across {len(ranked)} partition(s) "
              f"({diag['n_location_overlaps']} cross-ledger same-location overlaps dropped; "
              f"{diag['n_id_collisions']} run-scoped id collisions namespaced apart)", flush=True)
        for line in RI.supply_lines(diag):
            print(f"  [supply] {line}", flush=True)
        return rows

    def intake(self):
        """Admit locations, then either REUSE a pre-staged snapshot or embed+cluster fresh.

        Snapshot-reuse contract (the `if self.intake_path.exists() and self.embs_path.exists()`
        branch): a run resumes/starts against a snapshot pinned at first intake, laid down as
        two sibling files under `self.out`:

          * `intake.json` — `{cluster_tags: {id: "<family>#<k>"}, fields: {id: [bin, json]},
                              n_admitted: int}`. `cluster_tags` is the AUTHORITATIVE membership
                              set: only its ids are colorized (a still-growing frontier ledger's
                              newer admits are deferred, counted, and logged — rerun fresh to
                              fold them in). `fields` maps each id to its cached 640×360 ss2
                              smooth field (bin + json) so no re-render is needed.
          * `morph_embs.npz` — `descriptor._save_embs` format ({ids, emb}); the morph-CLIP
                               embeddings keyed by the same ids, loaded via `D.load_embs`.

        `stage_first_release.py` writes exactly these two files to pre-union committed intake
        passes without re-embedding; the fresh branch below writes the identical shapes."""
        rows = self._load_all_admitted()
        if not rows:
            srcs = ", ".join(str(l) for l in self.ledgers)
            raise SystemExit(f"no admitted (current-decode ∧ q3 ∧ guard ∧ distinct) rows in {srcs}")
        if self.intake_path.exists() and self.embs_path.exists():
            meta = json.loads(self.intake_path.read_text(encoding="utf-8"))
            embs = D.load_embs(self.embs_path)
            fields = {k: tuple(v) for k, v in meta["fields"].items()}
            tags = meta["cluster_tags"]
            # Snapshot semantics: a resume works against the locations embedded at first
            # intake. The run-scoped ledger may keep growing (a live frontier appends), but
            # those newly-admitted locations are NOT in the cached embeddings/tags/fields —
            # restrict to the snapshot and log how many fresh admits are being deferred.
            snap = set(tags)
            n_new = sum(1 for r in rows if r["id"] not in snap)
            rows = [r for r in rows if r["id"] in snap]
            # ...and the snapshot is restricted BACK to the admitted rows. The intersection
            # used to be one-sided — every snapshot id was admitted, because the snapshot was
            # written by a run under the same predicate — and the read-time intake broke that:
            # a snapshot row now below the junk floor stays in `cluster_tags` while dropping
            # out of `rows`. Every reader that walks the tags and joins to `by_id` then dies on
            # the last statement of the run (`report.write_report` did, after 30 colorizes and
            # 9 full-res release renders). The tags/fields ARE the population's, so they are
            # narrowed here rather than defended against at each reader.
            self.intake_scope = set(snap)     # the census population is the SNAPSHOT's
            kept = {r["id"] for r in rows}
            n_dropped = len(snap - kept)
            tags = {i: t for i, t in tags.items() if i in kept}
            fields = {i: f for i, f in fields.items() if i in kept}
            print(f"[intake] reused {len(rows)} admitted locations (snapshot), "
                  f"{len(set(tags.values()))} morph clusters"
                  + (f"; {n_new} newer admits deferred (rerun fresh to include)" if n_new else "")
                  + (f"; {n_dropped} snapshot location(s) no longer admitted, dropped"
                     if n_dropped else ""),
                  flush=True)
        else:
            print(f"[intake] {len(rows)} admitted locations — embedding morph + clustering ...", flush=True)
            embs, fields = D.embed_locations(rows, self.field_cache, self.embs_path)
            # Seed the clustering with the LIBRARY's per-type medoids, so this batch is
            # deduplicated against the atlas and not only against itself. Without it every
            # intake adds an un-deduped seam and the cluster count (and every deficit computed
            # over those cells) drifts upward by one seam's worth of near-duplicates.
            lib, prior, note = D.load_library_seed(self.library_dir)
            print(f"[intake] {note}", flush=True)
            tags = D.assign_morph_clusters(rows, embs, library=lib)
            D.verify_library_unmoved(prior, tags)   # nothing already in the library moves
            self.intake_path.write_text(json.dumps(
                {"cluster_tags": tags, "fields": {k: list(v) for k, v in fields.items()},
                 "n_admitted": len(rows),
                 # durable per-location source tag so source-tag overrides resolve from the
                 # snapshot alone (the first_release readout reads artifacts, not ledgers).
                 "source_tags": {r["id"]: self._source_tag_of(r) for r in rows}}),
                encoding="utf-8")
            print(f"[intake] {len(set(tags.values()))} morph clusters "
                  f"across {len(set(r['family'] for r in rows))} types", flush=True)
        self.rows = rows
        self.by_id = {r["id"]: r for r in rows}
        # The cell axis's first coordinate. `row["family"]` is the ledger's partition for the
        # nine base partitions and WRONG for exactly one — a classic-phoenix row says
        # `phoenix` — so it goes through the row resolver, which splits on the parameter point.
        self.partition_of = {r["id"]: D.cell_partition(r) for r in rows}
        self.embs = embs
        self.fields = fields
        self.cluster_tags = tags
        self._index_ranks()

    def _index_ranks(self):
        """Rank position per location id, and the per-partition supply census — all three
        numbers (mined / passing / emit cap) over the rows this run will ACTUALLY serve.

        `self.intake_scope` is that restriction: `None` for a fresh intake (the whole union),
        and the SNAPSHOT's id set when a run resumes against pre-staged embeddings. Both halves
        of the census come from `RI.supply_census` in one walk over the same scoped population,
        because taking the mined count from the ledgers and the passing count from the served
        subset is how the sheet printed "81 mined, 18 above floor" for a partition whose real
        served numbers were 33 and 18."""
        by_part: dict = {}
        for r in self.rows:
            by_part.setdefault(self.partition_of[r["id"]], []).append(r)
        self.ranked = {p: RI.rank_rows(v) for p, v in by_part.items()}
        self.rank_of = {r["id"]: k for v in self.ranked.values()
                        for k, r in enumerate(v)}
        mined, passing, good = RI.supply_census(self.mined_rows, self.intake_scope)
        self.mined_supply = mined
        self.passing_supply = passing
        # The slot guarantee's trigger population: candidates above `floors.GOOD_FLOOR`, from
        # the SAME scoped walk as the other two (see `RI.supply_census`).
        self.good_supply = good
        self.emit_caps = {p: RI.emit_cap(n) for p, n in passing.items()}

    # ---- axes + deficit model ------------------------------------------- #
    def build_axes(self, dt, cell_to_names: dict, lib):
        # palette flavors: only cells with at least one pool-loadable palette are feasible.
        self.flavors = sorted(f for f, names in cell_to_names.items()
                              if any(p in lib.colormaps for p in names))
        self.styles = render_styles(dt)
        observed = sorted({(self.partition_of[i], self.cluster_tags[i]) for i in self.by_id})
        feasible = C.build_feasible_cells(observed, self.flavors, self.styles)
        # THE target: the canonical release-mix ratio table, re-solved against THIS intake's
        # feasible cells (weight_p = share_p / n_cells_p). There is no measure file and no
        # hand-placed override — `release_mix.RATIO` is the single source, and the same derived
        # shares are what `deficit_scheduler` puts in its order book.
        self.release_shares = RM.shares(sorted({p for (p, _c) in observed}))
        self.target = C.TargetMeasure.from_partition_shares(self.release_shares, feasible)
        print(f"[axes] target from release_mix: "
              + ", ".join(f"{p} {s:.2%}" for p, s in sorted(self.target.shares.items())),
              flush=True)
        self.model = C.DeficitModel(feasible, self.target)
        # rebuild deficit counts from the DURABLE pool log (resume safety).
        for r in self.pool.rows:
            cell = tuple(r["cell"])
            if cell in self.model.support or cell in self.model.capped:
                self.model.record_attempt(cell)
                if r.get("passed"):
                    self.model.record_fill(cell)
        print(f"[axes] {len(observed)} (type,cluster) × {len(self.flavors)} flavors × "
              f"{len(self.styles)} styles = {len(feasible)} feasible cells "
              f"| resumed attempts={self.pool.n_attempts()} gated={len(self.pool.gated())}", flush=True)

    # ---- the colorize attempt budget (§1, prompts/selection_restructure_2.md) --------- #
    def styles_for_head(self, head: str | None) -> list:
        """The render styles a head's budget may spend an attempt on. `None` = every style.

        This is what makes the head budget REAL rather than an accounting fiction: the deficit
        model still picks the (flavor, style) pair, but it picks it inside the head that paid
        for the attempt. `None` is the `--cover-all` path, which is explicitly not budgeted."""
        if head is None:
            return list(self.styles)
        return [s for s in self.styles if head_for_style(s) == head]

    def plan_attempts(self) -> list:
        """Build (and record) THIS run's colorize plan — `attempt_budget.plan`.

        The plan is over the RANKED INTAKE (`self.ranked`, floor-passing, best-first per
        partition), so a location is only ever planned for if it is in this run's supply. Its
        length is the run's colorize volume: `--max-attempts` is the total attempt budget the
        4x-per-slot want is scaled against, and the loop no longer stops on `--target-gated`
        (see `run_colorize`).

        A RESUME re-derives the same plan (it is a pure function of the intake, the release
        split and the budget) and then subtracts what the DURABLE pool already did, matched on
        (location, head). Nothing is restored from a checkpoint and nothing is re-colorized."""
        ranked_ids = {p: [r["id"] for r in rows] for p, rows in self.ranked.items()}
        # THE GUARANTEE'S TRIGGER POPULATION, handed to the budget so the colorize can fund
        # what the release is about to be obliged to seat. It is the same `good_supply` census
        # `_plan_with_guarantees` triggers on, and it is available here because it comes from
        # the INTAKE walk — no colorize has happened yet, which is the whole reason the attempt
        # budget could be blind to it.
        guaranteed = [p for p, n in sorted((self.good_supply or {}).items()) if n >= 1]
        plan, budget = AB.plan(release_n=self.release_n, strange_frac=self.strange_frac,
                               total_budget=self.max_attempts, ranked_ids=ranked_ids,
                               guaranteed=guaranteed)
        done = Counter((r["location_id"], head_for_style(r["render_style"]))
                       for r in self.pool.rows)
        todo = []
        for att in plan:
            key = (att.location_id, att.head)
            if done.get(key, 0) > 0:
                done[key] -= 1
                continue
            todo.append(att)
        budget["resumed_attempts"] = len(plan) - len(todo)
        self.attempt_plan, self.attempt_budget = todo, budget
        return todo

    def realized_fills(self) -> dict:
        """`{head: {partition: attempts}}` actually made, DERIVED from the durable pool."""
        return AB.realized_fills(self.pool.rows, head_for_style)

    # ---- location pick (coverage round-robin, ranker-ordered within a round) --------- #
    #
    # RETIRED FROM THE BUDGETED PATH on 2026-08-09 (prompts/selection_restructure_2.md), and
    # KEPT after the dead-code pass that followed it (selection_restructure_3.md).
    # `_round_order`/`pick_location` are the coverage round-robin that decided colorize order
    # while volume was a deficit-model side effect; `--cover-all` (an explicit one-pass over
    # every admitted location, which has no release budget to be sized against) still runs
    # them, and they are otherwise superseded by `plan_attempts`. `--cover-all` is a live
    # flag and this is its only ordering rule, so this is a LIVE path with one caller — not
    # something the deletion pass overlooked.
    def _round_order(self, rows, round_idx: int) -> list:
        """The order ONE round is served in: seeded round-robin ACROSS partitions and RANK
        ORDER WITHIN one. Returns a list of location ids.

        The within-partition order was a seeded shuffle until 2026-08-09 and is now the
        read-time rank — raw P(>=3) descending (`ranked_intake.rank_key`, tie-broken on id).
        The shuffle was the honest thing to do while every admitted row carried the same
        binary verdict (`decoded_class >= 3`) and nothing distinguished them; ranking on the
        stored probability is only available because the intake stopped throwing that
        probability away in favour of the verdict. Under a budget smaller than the union the
        within-round order IS the selection, so this is where the ranking actually pays.

        The ACROSS-partition round-robin is UNCHANGED and is still seeded, including the
        partition shuffle: a bounded budget must still reach every partition (a rank-ordered
        global queue would spend the whole budget on whichever partition scores highest), and
        no partition may be systematically first.

        The partition set is DERIVED from `rows` (`self.partition_of`), never a literal — the
        ten base partitions are already a moving set (`descriptor.cell_partition` splits
        classic-phoenix off `family`), and a hardcoded list would silently drop a new one to
        the tail. Pure and deterministic in (seed, round_idx, the round's membership), so a
        `--resume` that rebuilds the queue mid-round rebuilds the same one."""
        by_part: dict = {}
        for r in rows:
            by_part.setdefault(self.partition_of[r["id"]], []).append(r["id"])
        parts = sorted(by_part)
        rng = np.random.default_rng([self.seed, int(round_idx)])
        rank_of = getattr(self, "rank_of", {})
        for p in parts:
            # rank ascending (best first); an id the rank index does not know goes to the tail
            # rather than to the front, so a missing rank can never outrank a scored row.
            by_part[p] = sorted(by_part[p], key=lambda i: (rank_of.get(i, 1 << 30), i))
        rng.shuffle(parts)                           # no partition is systematically first
        out = []
        for k in range(max(len(v) for v in by_part.values())):
            for p in parts:
                if k < len(by_part[p]):
                    out.append(by_part[p][k])
        return out

    def pick_location(self, exhausted: set):
        """Next location to colorize: fewest-attempts-first ACROSS rounds, seeded partition
        round-robin WITHIN a round.

        Fewest-attempts-first is what keeps diversity supply intact — every location gets a
        colorize before any gets a second. But it only preserves COVERAGE when the budget
        covers the whole union: under any smaller budget the run never leaves round 0, so the
        within-round order IS the selection, and coverage is whatever that order happens to
        give. It used to be `id`, which made a bounded run colorize an alphabetical prefix —
        the 200-attempt smoke took `ids[0:200]`, campaign1 only, and four of the seven source
        ledgers got exactly zero. So the within-round order is a round-robin across the
        partitions present in the round: a budget smaller than the union now spends
        proportionately across every partition that has admitted rows instead of exhausting the
        alphabetically-first one.

        Inside a partition it is RANK ORDER — the read-time ranked intake's best-first list
        (`_round_order`). It was a seeded shuffle until 2026-08-09; there is still no location
        ranker (deferred_recalibration.md § "Ranker rebuild — DELETED") and this is not one:
        the key is the stage-1 head's own stored P(>=3), read at intake rather than frozen
        into a class."""
        counts = self.pool.attempts_per_location()
        cand = {r["id"]: r for r in self.rows if r["id"] not in exhausted}
        if not cand:
            return None
        while True:
            lo = min(counts.get(i, 0) for i in cand)
            if self._round_idx != lo:
                self._round_idx, self._round_queue = lo, None
            if self._round_queue is None:
                self._round_queue = self._round_order(
                    [r for i, r in cand.items() if counts.get(i, 0) == lo], lo)
            while self._round_queue:
                row = cand.get(self._round_queue.pop(0))
                if row is not None and counts.get(row["id"], 0) == lo:
                    return row
            if any(counts.get(i, 0) == lo for i in cand):
                self._round_queue = None      # queue went stale (resume / exhaustion): rebuild
                continue
            self._round_idx = None            # round complete: `lo` advances on the next pass

    # ---- one colorize ---------------------------------------------------- #
    def floor_for(self, style: str) -> float:
        """The head's RETIRED pool floor — an ANNOTATION since 2026-08-09. Still resolved and
        still written onto every pool row (`floor`), so "would this row have been pooled under
        the old bar" stays answerable off the durable log; it admits nothing and rejects
        nothing."""
        return self.floor if head_for_style(style) == "wallpaper" else self.mining_floor

    def release_floor_for(self, style: str) -> float:
        """The head's RETIRED release floor — an ANNOTATION since 2026-08-09 (same as
        `floor_for`). Read by `would_pass_release_floor` and by the sheets; it gates nothing."""
        return self.release_floor if head_for_style(style) == "wallpaper" \
            else self.mining_release_floor

    def above_pool_floor(self, r) -> bool:
        """Annotation: would this row have cleared its head's retired POOL floor?"""
        return (r.get("p_ge3") or 0.0) >= self.floor_for(r["render_style"])

    def would_pass_release_floor(self, r) -> bool:
        """Annotation: would this row have cleared its head's retired RELEASE floor (0.90
        wallpaper / 0.50 mining)? Recorded as a column on every release record row and marked
        on every sheet tile — the old cut's value stays inspectable, which is the whole reason
        the four `Floor` objects were retired rather than deleted."""
        return (r.get("p_ge3") or 0.0) >= self.release_floor_for(r["render_style"])

    def release_eligible(self) -> list:
        """Pool rows eligible for release selection: every row that got a head SCORE.

        No floor is applied here any more. Until 2026-08-09 this was "pool-admitted ∧ clears
        its head's RELEASE floor" — two enforcing per-head cuts — and the whole point of the
        restructure is that a threshold is a read-time choice, not frozen enforcing state. The
        cut that does remain is upstream and coarse: a location only reached colorize at all if
        it cleared the junk floor at the colorize-pool draw (`ranked_intake`). What decides the
        release now is RANK plus the two caps (`select_release`), and the retired floors ride
        along as annotation (`would_pass_release_floor`).

        A row with no score is still not eligible — a render or scoring error is an absence of
        a verdict, not a bad one, and ranking on a missing number is not a choice anybody made.

        `write_gate_report` still logs every scored strange candidate at BOTH sites. With the
        mining floor annotating again, its `would_cut ∧ selected` join is no longer zero by
        construction — the free false-cut signal the 2026-08-06 flip cost comes back."""
        return [r for r in self.pool.rows
                if r.get("passed") and r.get("p_ge3") is not None]

    def post_floor(self, rows=None) -> list:
        """What `--target-gated` counts: the release-eligible rows, i.e. every SCORED row.

        This used to be "eligible ∧ clears its head's release floor", and while both floors
        enforced it was an identity on `release_eligible()` maintained by a separately-computed
        predicate so the two would visibly diverge if a floor ever went report-only. Both
        floors are now ANNOTATION-ONLY, so the identity is real rather than coincidental and
        the divergence it was watching for is reported by name instead:
        `would_pass_release_floor_*` in the accounting below.

        WHAT THAT COSTS, SAID PLAINLY. The default surplus target is 3xN, and it now counts
        3xN *scored* candidates rather than 3xN candidates above 0.90/0.50. That is a weaker
        surplus in exactly the way the floors were strong: a run reaching its target no longer
        implies it has 3xN release-grade rows by the retired bar. The counterfactual counts are
        in the accounting so the difference is a number in the report, not a surprise."""
        return list(self.release_eligible() if rows is None else rows)

    def target_accounting(self) -> dict:
        """What the surplus target sees, and what the RETIRED release floors would have cut.

        `post_floor` counts toward `--target-gated` and is now every scored row (see above).
        `would_pass_release_floor` / `_smooth` / `_strange` are the counterfactual: how much of
        that population the 0.90 / 0.50 cuts would have left. `below_retired_release_floor` is
        its complement — the rows this restructure made shippable, which is the number the
        change actually turns on, and `cut_by_release_floor_strange` is the strange half of it
        (kept under its old name because the readouts and the run banner join on it)."""
        elig = self.release_eligible()
        would = [r for r in elig if self.would_pass_release_floor(r)]
        would_ids = {r["id"] for r in would}
        below = [r for r in elig if r["id"] not in would_ids]
        cut_strange = [r for r in below if head_for_style(r["render_style"]) == "mining"]
        return {
            "target_gated": self.target_gated,
            "post_floor": len(elig),
            "post_floor_smooth": sum(1 for r in elig
                                     if head_for_style(r["render_style"]) == "wallpaper"),
            "post_floor_strange": sum(1 for r in elig
                                      if head_for_style(r["render_style"]) == "mining"),
            "would_pass_release_floor": len(would),
            "would_pass_release_floor_smooth": sum(
                1 for r in would if head_for_style(r["render_style"]) == "wallpaper"),
            "would_pass_release_floor_strange": sum(
                1 for r in would if head_for_style(r["render_style"]) == "mining"),
            "below_retired_release_floor": len(below),
            # retained name: the strange rows the retired 0.50 would have removed. Non-zero
            # now means "shippable that would not have been", the opposite sign of what it
            # meant while the floor enforced — the key is the same population either way.
            "cut_by_release_floor_strange": len(cut_strange),
            "ungated_eligible": 0,
            "release_eligible": len(elig),
            "above_retired_pool_floor": sum(1 for r in elig if self.above_pool_floor(r)),
        }

    def target_met(self) -> bool:
        """THE `--target-gated` break condition, as one testable predicate rather than an
        expression buried in the colorize loop."""
        return self.target_accounting()["post_floor"] >= self.target_gated

    # ---- durable gate/release record ------------------------------------- #
    #
    # The record of WHICH locations were gated, WHICH were released, and OUT OF WHAT
    # population — written to data/ through paths.durable(), not under --out. Everything
    # under --out is a scratch/ path in practice, which is why campaign-2's emission stage
    # has no record for any run. See tools/emission/release_record.py.
    #
    # Distinct from `write_gate_report` below: that logs only the strange head's REPORT-ONLY
    # counterfactual. This logs the decisions that actually cut, for both heads, plus the
    # denominators.

    RECORD_SITE = "emission_diversity_v1"

    def _run_id(self) -> str:
        """The run this record belongs to = the output dir it was driven into. A --resume of
        the same run re-derives the same id and upserts in place; a different run accumulates
        alongside."""
        return self.out.name

    def _record_location(self, location_id) -> dict:
        loc = self.by_id.get(location_id, {})
        return {"location_id": location_id, "cx": loc.get("outcome_cx"),
                "cy": loc.get("outcome_cy"), "fw": loc.get("outcome_fw"),
                "julia_c_re": loc.get("julia_c_re"), "julia_c_im": loc.get("julia_c_im")}

    @staticmethod
    def _record_join_key(r) -> str:
        return "|".join(str(x) for x in (r["location_id"], r["render_style"], r["palette"]))

    def _gate_decision_rows(self):
        """One row per colorized candidate. The only rejection left at this stage is the
        absence of a verdict; the retired pool floor is recorded as `would_pass_floor`."""
        from tools.emission import release_record as RR
        run_id = self._run_id()
        rows = []
        for r in self.pool.rows:
            if r.get("error"):
                decision, reason = "rejected", f"render_error: {r['error'][:120]}"
            elif r.get("p_ge3") is None:
                decision, reason = "rejected", "unscored (no head result)"
            else:
                decision, reason = "admitted", None
            rows.append(RR.decision_row(
                run_id=run_id, stage=RR.STAGE_GATE,
                join_key=self._record_join_key(r), location_id=r["location_id"],
                location=self._record_location(r["location_id"]),
                partition=r.get("type"), morph_cluster=r.get("morph_cluster"),
                decision=decision, score=r.get("p_ge3"), reason=reason,
                head=r.get("head"), floor=r.get("floor"),
                would_pass_floor=(None if r.get("p_ge3") is None
                                  else self.above_pool_floor(r)),
                # The band auto-level's stamp for the SCORED render — the image this verdict
                # was actually taken on. Read off the pool row rather than re-derived: the
                # pool row is what the operator stamped, and a second derivation here could
                # describe a render nobody made.
                autolevel=r.get("autolevel"),
                style=r.get("render_style"), palette=r.get("palette")))
        return rows

    def _release_decision_rows(self, selected):
        """One row per RELEASE-ELIGIBLE candidate: selected, or eligible-and-passed-over. The
        not_selected rows are the point — a released row alone cannot say what it beat — and
        they are recorded exactly as before. What is new is `would_pass_floor`: the retired
        0.90 / 0.50 verdict on every row, selected or not, so the cut this restructure stopped
        applying stays inspectable on the record it stopped applying to.

        `reason` names WHICH cap passed a row over, not just that one did — the rank, the
        cluster cap and the partition budget fail differently and a bare "not picked" cannot
        tell a thin partition from a saturated cluster.

        `slot_source` (2026-08-10) is the per-slot provenance of a SELECTED row: `guarantee` if
        that pick took the partition's guaranteed slot, `mix` if it took a `release_mix` one,
        None on every not_selected row. It is on the record and not only in the run's log
        because the question it answers — "which of these tiles only shipped because of the
        guarantee" — is exactly the one the for-now policy will be judged on later."""
        from tools.emission import release_record as RR
        run_id = self._run_id()
        sel_ids = {e["_rec"]["id"] for e in selected}
        skips = {l["id"]: l for l in (getattr(self, "release_log", None) or [])
                 if not l.get("picked")}
        rows = []
        for r in self.release_eligible():
            chosen = r["id"] in sel_ids
            if chosen:
                reason = None
            elif r["id"] in skips:
                reason = (f"passed over: {skips[r['id']]['skip']} "
                          f"(rank {skips[r['id']]['rank_in_partition']} in "
                          f"{skips[r['id']]['partition']})")
            else:
                reason = "eligible; outranked within its partition's slot/supply budget"
            rows.append(RR.decision_row(
                run_id=run_id, stage=RR.STAGE_RELEASE,
                join_key=self._record_join_key(r), location_id=r["location_id"],
                location=self._record_location(r["location_id"]),
                partition=r.get("type"), morph_cluster=r.get("morph_cluster"),
                decision=("selected" if chosen else "not_selected"),
                score=r.get("p_ge3"), reason=reason,
                head=r.get("head"), floor=self.release_floor_for(r["render_style"]),
                would_pass_floor=self.would_pass_release_floor(r),
                slot_source=(self.slot_source or {}).get(r["id"]),
                autolevel=r.get("autolevel"),      # the scored render's stamp; see decision_row
                style=r.get("render_style"), palette=r.get("palette")))
        return rows

    def _record_counts(self, gate_rows, release_rows, selected) -> dict:
        """The funnel, with per-partition breakdowns at every stage. The breakdowns are what
        make a per-family rate recoverable later; a bare total cannot answer 'out of how many
        julia'."""
        def by_part(rows, pred=lambda _r: True):
            out: dict = {}
            for r in rows:
                if pred(r):
                    out[r["partition"]] = out.get(r["partition"], 0) + 1
            return out
        intake_parts: dict = {}
        for r in self.rows:
            part = self.partition_of[r["id"]]
            intake_parts[part] = intake_parts.get(part, 0) + 1
        return {
            "intake_admitted": len(self.rows),
            "intake_by_partition": intake_parts,
            "colorized": len(gate_rows),
            "colorized_by_partition": by_part(gate_rows),
            "gate_admitted": sum(1 for r in gate_rows if r["decision"] == "admitted"),
            "gate_admitted_by_partition": by_part(gate_rows, lambda r: r["decision"] == "admitted"),
            "release_eligible": len(release_rows),
            "release_eligible_by_partition": by_part(release_rows),
            "released": sum(1 for r in release_rows if r["decision"] == "selected"),
            "released_by_partition": by_part(release_rows, lambda r: r["decision"] == "selected"),
            "release_n_requested": self.release_n,
            # THE COLORIZE BUDGET, planned beside realized (2026-08-09). It belongs in `counts`
            # for the same reason the per-partition breakdowns do: it is the colorize stage's
            # DENOMINATOR. Without it a later reader can see that a head shipped 3 of 6 slots
            # and cannot tell whether it was given 3 attempts or 15 — which is exactly the
            # question the selrestruct_1 smoke could not answer about itself.
            "colorize_budget": dict(self.attempt_budget or {}),
            "colorize_realized_by_head": self.realized_fills(),
        }

    def write_release_record(self, selected):
        """Write the durable gate + release record for this run. Returns
        (path, n_total, n_new, runs_path)."""
        from tools.emission import release_record as RR
        gate_rows = self._gate_decision_rows()
        rel_rows = self._release_decision_rows(selected)
        path, n_total, n_new = RR.write_decisions(self.RECORD_SITE, gate_rows + rel_rows)
        rpath, _, _ = RR.write_run(self.RECORD_SITE, RR.run_row(
            run_id=self._run_id(), site=self.RECORD_SITE,
            out_dir=str(self.out.relative_to(ROOT)) if str(self.out).startswith(str(ROOT))
            else str(self.out),
            ledgers=[str(p) for p in self.ledgers],
            counts=self._record_counts(gate_rows, rel_rows, selected),
            floors={"pool_wallpaper": self.floor, "pool_mining": self.mining_floor,
                    "release_wallpaper": self.release_floor,
                    "release_mining": self.mining_release_floor},
            ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
        return path, n_total, n_new, rpath

    def write_gate_report(self, selected):
        """Mining-head would-cut log, PAIRED with the actual outcome AT BOTH CUT SITES.

        One row per scored strange candidate: what the release floor cut and whether the row
        was actually selected, AND what the mining POOL floor cut and whether the row was
        actually pooled. Both sites keep accruing after the 2026-08-06 flip — this reads
        `self.pool.rows`, not the eligible set, so the denominator is still every scored
        strange candidate.

        WHAT THE FLIP TOOK. Both outcome joins are now zero by construction: selection implies
        clearing 0.50, which implies clearing 0.25, so neither `would_cut ∧ selected` nor
        `would_cut_pool ∧ selected` can ever be non-zero again. That pairing was the free
        false-cut signal report-only bought, and it is the price of enforcing (floors.py).
        What the log still is: the durable population record of every strange candidate, its
        p_ge3, and what each floor did to it — the denominator a labeled calibration pass
        needs, which now has to bring its own labels. Committed under data/emission/."""
        from tools.mining import gate_report as GR
        sel_ids = {e["_rec"]["id"] for e in selected}
        rows = []
        for r in self.pool.rows:
            if head_for_style(r["render_style"]) != "mining" or r.get("p_ge3") is None:
                continue
            loc = self.by_id.get(r["location_id"], {})
            location = {"cx": loc.get("outcome_cx"), "cy": loc.get("outcome_cy"),
                        "fw": loc.get("outcome_fw"), "julia_c_re": loc.get("julia_c_re"),
                        "julia_c_im": loc.get("julia_c_im"), "location_id": r["location_id"]}
            key = "|".join(str(x) for x in (r["location_id"], r["render_style"], r["palette"]))
            rows.append(GR.gate_report_row(
                site="emission_diversity_v1", key=key, location=location,
                style=r["render_style"], palette=r["palette"], p_ge3=r.get("p_ge3"),
                release_threshold=self.mining_release_floor, pool_floor=self.mining_floor,
                pooled=bool(r.get("passed")),
                selected=(r["id"] in sel_ids), selection_stage="release"))
        return GR.write_gate_report("emission_diversity_v1", rows)

    def colorize(self, dt, cm, ranker, heads, row, tracker=None,
                 budget_head=None) -> dict | None:
        """One colorize attempt. `budget_head` is the head whose attempt budget paid for it and
        fixes the STYLE SET the deficit model chooses within (`styles_for_head`); `None` (the
        `--cover-all` path) offers every style, which is the pre-2026-08-09 behaviour.

        Spelled `budget_head`, not `head`: `head` is this function's local for the SCORE dict
        the head returned, and a parameter of that name would be silently overwritten by it
        half way down — the log line would then report the score dict as the paying head."""
        _t_attempt = time.time()
        loc_id = row["id"]
        ftype = self.partition_of[loc_id]
        cluster = self.cluster_tags[loc_id]
        choice = C.choose_option(self.model, ftype, cluster, self.flavors,
                                 self.styles_for_head(budget_head), self.rng)
        if choice is None:
            return None                              # all cells for this (type,cluster) capped
        flavor, style, deficit, n_opts, _probs = choice
        fbin, fjson = self.fields[loc_id]
        palette, pref_fit = ranker.best(loc_id, flavor, fbin, fjson, tracker=tracker)
        if palette is None:
            return None
        emid = self.pool.next_id()
        jpg = self.renders / f"{emid}.jpg"
        loc = D.location_of(row)
        cell = (ftype, cluster, flavor, style)
        floor = self.floor_for(style)
        err = None
        head = None
        stats = None
        rinfo = {}
        try:
            rinfo = render_wallpaper(dt, cm, loc, style, palette, jpg,
                                     POOL_W, POOL_H, POOL_SS, POOL_FILT) or {}
            head = heads.score(style, jpg)
            stats = realized_palette_stats(jpg)
            if tracker is not None:
                tracker.ingest(stats)                # running deficit reflects TRUE output
        except Exception as e:                       # noqa: BLE001
            err = repr(e)[:300]
        # POOL ADMISSION IS NO LONGER A FLOOR (2026-08-09). `passed` means "this candidate got
        # a head score" — a render/scoring error is the only thing that keeps a row out of the
        # pool. The retired pool floor rides along as `above_pool_floor`, computed at the write
        # site off the same score, so the durable log can still answer what the old bar did.
        passed = bool(head and head.get("p_ge3") is not None)
        above_pool = bool(passed and head["p_ge3"] >= floor)
        capped = self.model.record_attempt(cell)
        if passed:
            self.model.record_fill(cell)
        rec = {
            "id": emid, "location_id": loc_id,
            "type": ftype, "morph_cluster": cluster,
            "palette_flavor": flavor, "render_style": style, "palette": palette,
            "cell": list(cell),
            "head": (head or {}).get("head"), "head_gate": (head or {}).get("gate"),
            "p_ge2": (head or {}).get("p_ge2"), "p_ge3": (head or {}).get("p_ge3"),
            "score": (head or {}).get("ssum"),
            "floor": floor, "passed": passed, "error": err,
            # the two retired cuts, as annotation on the durable row.
            "above_pool_floor": above_pool,
            "release_floor": self.release_floor_for(style),
            "would_pass_release_floor": bool(
                passed and head["p_ge3"] >= self.release_floor_for(style)),
            "realized_palette": stats,
            "render": {"w": POOL_W, "h": POOL_H, "ss": POOL_SS},
            "jpg": str(jpg.relative_to(ROOT)) if jpg.exists() else None,
            "pref_fit": pref_fit, "ranker": ranker.mode, "palette_pick": ranker.pick_mode,
            "provenance": {
                "source_ledger": row.get("_source_ledger", str(self.ledger.relative_to(ROOT))),
                "ranker_score": self.ranker_score.get(loc_id),
                "ranker_pct": self.ranker_pct.get(loc_id),
                "source_run": row.get("ts"), "node_id": row.get("node_id"),
                "root_id": row.get("root_id"), "branch": row.get("branch"),
                "reached_depth": row.get("reached_depth"), "p_good": row.get("p_good"),
                "scorer_version": row.get("scorer_version"),
            },
        }
        # The band auto-level's stamp, present ONLY on rows the operator actually produced.
        # Under the shipped switch (ON) that is every colored row, `acted` saying whether the
        # band moved it; with the switch forced off `maybe_level` returns no stamp and the row
        # is byte-identical to a pre-operator one.
        if rinfo.get("autolevel"):
            rec["autolevel"] = rinfo["autolevel"]
        self.pool.append(rec)
        with open(self.colorize_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": emid, "location_id": loc_id, "type": ftype, "cluster": cluster,
                # which head's attempt budget paid for this render. `None` on the --cover-all
                # path (unbudgeted); the style→head routing recovers it either way, and this is
                # the direct read.
                "budget_head": budget_head,
                "chosen_flavor": flavor, "chosen_style": style, "palette": palette,
                "deficit": round(deficit, 6), "n_options": n_opts,
                "p_ge3": (head or {}).get("p_ge3"), "passed": passed,
                "above_pool_floor": above_pool,
                "capped_cell": bool(capped), "error": err,
            }) + "\n")
        # The attempt's wall cost. Timed from the top of the call (the palette rank is part of
        # what an attempt costs, not overhead beside it) and recorded here rather than in the
        # two loops, so the `--cover-all` path and the budgeted path are timed identically.
        # An attempt that RAISED is a row with `error` set and its seconds still spent —
        # dropping it would make a failing run read as a fast one.
        self.stage_times.record("colorize", emid, time.time() - _t_attempt,
                                type=ftype, style=style, budget_head=budget_head,
                                passed=passed, error=bool(err))
        return rec

    # ---- main colorize loop --------------------------------------------- #
    def run_colorize(self):
        dt = _deploy_tail()
        from tools import colormap as cm
        from tools.studies import conditioned_colorize as cond
        _, cell_to_names = cond.load_cell_map()
        lib = dt.lib()
        self.build_axes(dt, cell_to_names, lib)
        ranker = PaletteRanker(dt, cell_to_names, lib, pick_mode=self.palette_pick,
                               deficit_lambda=self.deficit_lambda)
        # deficit tracker: rebuilt by replaying the durable pool log's realized stats, so a
        # resume continues the exact running deficit (order-independent — it is a sum).
        tracker = None
        if self.palette_pick == "deficit":
            from tools.emission.palette_deficit import DeficitTracker
            tracker = DeficitTracker(green_boost=self.deficit_green_boost)
            for r in self.pool.rows:
                tracker.ingest(r.get("realized_palette"))
            print(f"[colorize] palette-pick=deficit (λ={self.deficit_lambda} "
                  f"green_boost={self.deficit_green_boost}); tracker resumed from "
                  f"{tracker.n} logged renders", flush=True)
        heads = Heads()
        print(f"[colorize] cuts: {F.summary()}", flush=True)
        if self.floor_overrides:
            print(f"[colorize] CLI floor OVERRIDE (unstamped — no head version vouches for "
                  f"these): {self.floor_overrides}", flush=True)
        print(f"[colorize] ANNOTATION-ONLY floors: pool wallpaper={self.floor} "
              f"mining={self.mining_floor} · release wallpaper={self.release_floor} "
              f"(gate {heads.wp_gate}) mining={self.mining_release_floor} "
              f"(gate {heads.mining_gate}) · the one enforcing cut is the "
              f"{F.JUNK_FLOOR} junk floor, already applied at intake · "
              f"target={self.target_gated} scored rows · palette-ranker={ranker.mode} · "
              f"loc-ranker={self.ranker_mode}", flush=True)
        # The retired mining release floor still carries its measured operating point, and the
        # run still states it — that is what the annotation MEANS. Read from the frozen record,
        # which refuses if the pin has moved off the head it was measured on.
        from tools.mining.lock_mining_gate import read_lock       # noqa: PLC0415 (torch-free)
        _rel = read_lock()["cuts"]["mining_release"]
        print(f"[colorize] retired mining release floor {_rel['value']} was measured: fires "
              f"{_rel['fires']}/{_rel['n']} at precision {_rel['precision']:.3f} "
              f"[{_rel['precision_ci95'][0]:.3f}-{_rel['precision_ci95'][1]:.3f}], recall "
              f"{_rel['recall']:.3f} — OPTIMISTIC (mining_gate_lock.json caveats). It now "
              f"annotates; rows below it ship if they rank.", flush=True)
        if self.cover_all:
            print(f"[colorize] --cover-all: one colorize per location over "
                  f"{len(self.rows)} admitted locations (target/attempt/time cutoffs bypassed)",
                  flush=True)
            return self._run_cover_all(dt, cm, ranker, heads, tracker)
        return self._run_budgeted(dt, cm, ranker, heads, tracker)

    def _log_attempt(self, rec) -> None:
        """Persist the resume state and print the one-line progress for a completed attempt."""
        self.pool.save_state({"seed": self.seed, "rng": self.rng.bit_generator.state,
                              "n_attempts": self.pool.n_attempts()})
        acct = self.target_accounting()
        print(f"  [{self.pool.n_attempts()}] {rec['id']} {rec['type']}/{rec['morph_cluster']} "
              f"{rec['palette_flavor']}/{rec['render_style']} p_ge3="
              f"{rec['p_ge3'] if rec['p_ge3'] is not None else 'ERR'} "
              f"{'SCORED' if rec['passed'] else 'ERR'}"
              f"{'' if rec.get('would_pass_release_floor') else ' (below retired floor)'}"
              f" | pooled={len(self.pool.gated())} "
              f"scored={acct['post_floor']}/{self.target_gated} "
              f"({acct['would_pass_release_floor']} above the retired release floors)",
              flush=True)

    def _run_budgeted(self, dt, cm, ranker, heads, tracker):
        """THE colorize loop since 2026-08-09: run the budgeted plan, in plan order.

        The plan (`plan_attempts`) IS the volume decision — `--max-attempts` is the total
        attempt budget it is sized against, and it is spent in a near-proportional interleave
        across (head, partition) so a run the time backstop cuts short has still spent its
        prefix in the planned mix.

        `--target-gated` NO LONGER STOPS THIS LOOP. It was the old volume rule (build a 3xN
        surplus of scored rows, whatever the mix), and leaving it in as a break would silently
        truncate the head budgets it knows nothing about — a run that stops at 3N scored has
        spent ~3N/2 per head regardless of what the release asked for, which is the failure
        this restructure removes one level down. The target still computes and is still
        reported (`target_accounting`); it just does not decide when to stop.

        The time budget stays a HARD-KILL BACKSTOP and is reported as one when it fires."""
        t0 = time.time()
        plan = self.plan_attempts()
        print(f"[budget] colorize attempt budget from release need "
              f"(x{self.attempt_budget['attempt_multiplier']} per slot, total budget "
              f"{self.attempt_budget['total_budget']}"
              + (", SCALED DOWN proportionally — both heads"
                 if self.attempt_budget["scaled_to_budget"] else "") + "):", flush=True)
        for line in AB.fill_lines(self.attempt_budget, self.realized_fills()):
            print(f"  [budget] {line}", flush=True)
        for h in AB.HEADS:
            print(f"  [budget] {h} per partition: "
                  f"{ {p: k for p, k in self.attempt_budget['planned_by_partition'][h].items() if k} }",
                  flush=True)
        if self.attempt_budget.get("resumed_attempts"):
            print(f"  [budget] {self.attempt_budget['resumed_attempts']} planned attempt(s) "
                  f"already in the durable pool (resume) — {len(plan)} left to run", flush=True)
        n_capped = 0
        for k, att in enumerate(plan):
            if time.time() - t0 > self.time_budget_s:
                acct = self.target_accounting()
                print(f"[colorize] hit the time-budget backstop with {len(plan) - k} of "
                      f"{len(plan)} planned attempt(s) unspent (scored={acct['post_floor']}, "
                      f"below-retired-release-floor={acct['below_retired_release_floor']})",
                      flush=True)
                break
            row = self.by_id.get(att.location_id)
            if row is None:
                continue           # planned off the ranked intake; cannot happen for a served row
            rec = self.colorize(dt, cm, ranker, heads, row, tracker=tracker,
                                budget_head=att.head)
            if rec is None:
                # every cell for this (partition, cluster) x this head's styles is attempt-capped.
                n_capped += 1
                continue
            self._log_attempt(rec)
        acct = self.target_accounting()
        realized = self.realized_fills()
        self.attempt_budget["realized_by_partition"] = realized
        self.attempt_budget["realized_total"] = sum(sum(v.values()) for v in realized.values())
        self.attempt_budget["capped_cell_skips"] = n_capped
        print(f"[budget] realized: "
              + " · ".join(AB.fill_lines(self.attempt_budget, realized))
              + (f" · {n_capped} attempt(s) had no uncapped cell" if n_capped else ""),
              flush=True)
        return acct["post_floor"]

    def _run_cover_all(self, dt, cm, ranker, heads, tracker):
        """`--cover-all`: one colorize per admitted location, then stop. UNBUDGETED by design —
        it is a full sweep of the intake, so there is no release need to size it against, and it
        is the one live caller of the coverage round-robin (`pick_location`) the attempt budget
        replaced on the normal path."""
        exhausted: set = set()
        while True:
            # no target / attempt / TIME cutoff here — `--cover-all` is one pass over the whole
            # intake by definition, and it stops when `pick_location` wraps. Unchanged.
            acct = self.target_accounting()
            n_post, n_below = acct["post_floor"], acct["below_retired_release_floor"]
            row = self.pick_location(exhausted)
            if row is None:
                print(f"[colorize] all locations exhausted (scored={n_post}, "
                      f"below-retired-release-floor={n_below})", flush=True)
                break
            # cover-all stops the instant pick_location wraps to a 2nd pass: it returns
            # fewest-attempts-first, so an already-attempted row means every location has one.
            if self.pool.attempts_per_location().get(row["id"], 0) >= 1:
                # `n_rel` was a bare name that has never existed in this scope: the one
                # branch that reads it is `--cover-all`'s stop, so the flag documented as
                # "explicit one-pass semantics" raised NameError the moment it did its job.
                print(f"[colorize] --cover-all: every location colorized once — stopping "
                      f"before a 2nd pass (release-eligible={acct['release_eligible']})",
                      flush=True)
                break
            rec = self.colorize(dt, cm, ranker, heads, row, tracker=tracker)
            if rec is None:
                exhausted.add(row["id"])
                continue
            self._log_attempt(rec)
        return self.target_accounting()["post_floor"]

    def ranker_reach(self) -> dict:
        """How far down the ranker ordering the colorize had to reach. Ordering = admitted
        locations by ranker score desc; 'rank' is the 0-based position in that order. Reports
        the deepest rank among locations that (a) got any colorize attempt and (b) contributed
        a RELEASE-ELIGIBLE pool row — the practical measure of whether ranked intake
        concentrated budget on good locations."""
        if not self.ranker_score:
            return {}
        order = [r["id"] for r in sorted(self.rows,
                                         key=lambda r: -self.ranker_score.get(r["id"], float("-inf")))]
        rank_of = {i: k for k, i in enumerate(order)}
        attempted = {r["location_id"] for r in self.pool.rows}
        rel_locs = {r["location_id"] for r in self.release_eligible()}
        att_ranks = [rank_of[i] for i in attempted if i in rank_of]
        rel_ranks = [rank_of[i] for i in rel_locs if i in rank_of]
        n = len(order)
        return {
            "n_locations": n,
            "n_attempted": len(att_ranks),
            "deepest_attempted_rank": max(att_ranks) + 1 if att_ranks else 0,
            "deepest_attempted_pct": (max(att_ranks) + 1) / n if att_ranks else 0.0,
            "n_release_locs": len(rel_ranks),
            "deepest_release_rank": max(rel_ranks) + 1 if rel_ranks else 0,
            "deepest_release_pct": (max(rel_ranks) + 1) / n if rel_ranks else 0.0,
        }

    # ---- release selection ---------------------------------------------- #
    def _release_entries(self, rows: list) -> list:
        """Pool rows → selector entries. `emb` (the location's morph-CLIP embedding as a plain
        list) is the RETIRED `greedy_select` coverage kernel's input and is not read by
        `rank_select`; it is still attached so a side-by-side against the old rule needs no
        second builder."""
        entries = [{
            "id": r["id"], "type": r["type"], "cluster": r["morph_cluster"],
            "flavor": r["palette_flavor"], "style": r["render_style"],
            "score": r["p_ge3"], "emb": self.embs.get(r["location_id"], None),
            "_rec": r,
        } for r in rows]
        for e in entries:
            emb = e["emb"]
            e["emb"] = emb.tolist() if emb is not None else None
        return entries

    def _slot_plan(self, entries: list, n_slots: int, guaranteed=()) -> tuple:
        """`(slots, caps)` for ONE head pass: the partition slot allocation and the
        thin-supply emit cap it is crossed with.

        `guaranteed` is the set of partitions THIS pass owes a guaranteed slot to; it reaches
        `ranked_intake.partition_slots`, which pins each of them at 1 and apportions the
        remainder by the mix. Which head owes what is `_plan_with_guarantees`'s decision, not
        this one's.

        Slots come from `release_mix` — the canonical ratio table — re-solved over the
        partitions THIS PASS actually has candidates for, then apportioned to `n_slots` through
        `apportion.sequence_by_deficit` (the truncating-consumer rule; `ranked_intake
        .partition_slots`). Solving over the pass's own partitions rather than over the whole
        registry is deliberate: a partition with nothing to offer this head must not hold a
        slot hostage, and the honest place to see that it offered nothing is the supply line,
        not a silently-unfilled slot.

        Caps come from the INTAKE supply census (`self.emit_caps`), which is per partition and
        head-agnostic — the mined population is the mined population. The cap is therefore
        applied to each head pass independently and is a CEILING, not a budget being split."""
        parts = sorted({e["type"] for e in entries})
        shares = RM.shares(parts) if parts else {}
        slots = RI.partition_slots(shares, n_slots, guaranteed)
        # A partition with no supply census entry is UNCAPPED, not capped to zero. Zero would
        # be a silent kill on the one path where the census can be short of the pool — a
        # `--select-only` resume against a pool built from a larger intake — and a cap nobody
        # measured must not remove rows. It is printed, because "uncapped" is a decision.
        caps = {p: self.emit_caps[p] for p in parts if p in self.emit_caps}
        missing = [p for p in parts if p not in self.emit_caps]
        if missing:
            print(f"[select] no intake supply census for {missing} — UNCAPPED by the "
                  f"thin-supply rule (their slot budget still applies)", flush=True)
        return slots, caps

    # ---- the slot guarantee (2026-08-10, prompts/slot_guarantee_26.md) --------------- #
    def _guarantee_head(self, part: str, entries_by_head: dict, slot_n: dict,
                        assigned: dict) -> str:
        """WHICH head owes `part` its guaranteed slot. Deterministic, three keys in order:

          1. the head has room — a head cannot owe more guarantees than it has slots;
          2. fewer guarantees placed so far, so the two heads' mixes are eroded evenly rather
             than one head paying for every guarantee;
          3. more candidates for `part` in that head, then the head name, so the choice is a
             pure function of the pass and two runs over one pool agree.

        Returns None when NO head has a candidate for `part` — the caller reports that rather
        than raising, because a partition nobody colorized is a supply fact, not a policy
        failure. When a head has candidates but every such head is already full of guarantees,
        it RAISES (`SlotGuaranteeOverflow`): that is the "guaranteed partitions exceed N" case,
        and pro-rating it silently is what the raise exists to prevent."""
        placed = {h: sum(1 for hh in assigned.values() if hh == h) for h in entries_by_head}
        n_cand = {h: sum(1 for e in entries_by_head[h] if e["type"] == part)
                  for h in entries_by_head}
        able = [h for h in sorted(entries_by_head) if n_cand[h] > 0]
        if not able:
            return None
        cand = [h for h in able if placed[h] < int(slot_n.get(h, 0))]
        if not cand:
            raise RI.SlotGuaranteeOverflow(
                f"{part} has floor-passing supply and candidates in {able}, but every head "
                f"that could seat it is already full of guarantees (placed {placed} against "
                f"slots {dict(slot_n)}, release N={self.release_n}). Raise --release-n or "
                f"re-decide the guarantee; pro-rating it silently would make it not a "
                f"guarantee.")
        return min(cand, key=lambda h: (placed[h], -n_cand[h], h))

    def _plan_with_guarantees(self, entries_by_head: dict, slot_n: dict) -> tuple:
        """`(plans, caps, owed, unseatable)` — the slot allocation with the guarantee applied
        ACROSS the whole release.

        THE GUARANTEE (Matt, 2026-08-10, for-now): every partition with floor-passing supply
        gets one release slot; the remainder is apportioned by `release_mix` exactly as before.
        Floor-passing = at least one intake candidate above `floors.GOOD_FLOOR`
        (`self.good_supply`) — the higher of the two floors, deliberately (`RI.passes_good`).

        WHY IT IS A FIXED POINT AND NOT A ONE-PASS PATCH. The guarantee is defined on the WHOLE
        release ("a partition served by either head counts as served"), and the two heads are
        planned separately, so seating a guarantee in one head removes a mix slot that may have
        been the ONLY thing serving some other supplied partition. Each round re-plans both
        heads under the guarantees placed so far and re-asks which supplied partitions still
        emit nothing; a round that finds none is the answer. It terminates because `owed` only
        grows and is bounded by the partition count.

        "EMITS NOTHING" IS `min(slots, cap)`, NOT `slots == 0` (`RI.would_emit`). A partition
        the mix seats but the thin-supply cap zeroes ships no tile, and a guarantee that
        counted it as served would be a guarantee of an allocation rather than of a wallpaper.

        `unseatable` names supplied partitions with NO scored candidate in either head pass:
        reported and short-filled, never raised, because that is a colorize/supply outcome and
        no slot rule can fix it. The other way to fail — candidates exist but every head that
        could seat them is full of guarantees — DOES raise (`_guarantee_head`), because that
        one is the policy exceeding N."""
        supplied = [p for p, n in sorted((self.good_supply or {}).items()) if n >= 1]
        seatable = [p for p in supplied
                    if any(any(e["type"] == p for e in entries_by_head[h]) for h in entries_by_head)]
        unseatable = [p for p in supplied if p not in seatable]
        owed: dict = {}
        while True:
            plans, caps = {}, {}
            for h in entries_by_head:
                plans[h], caps[h] = self._slot_plan(
                    entries_by_head[h], slot_n[h],
                    {p for p, hh in owed.items() if hh == h})
            emits = {p: sum(RI.would_emit(plans[h].get(p, 0), caps[h].get(p))
                            for h in plans) for p in seatable}
            short = [p for p in seatable if p not in owed and emits[p] == 0]
            if not short:
                return plans, caps, owed, sorted(unseatable)
            for p in short:
                h = self._guarantee_head(p, entries_by_head, slot_n, owed)
                if h is None:
                    seatable.remove(p)
                    unseatable.append(p)
                    continue
                owed[p] = h

    def select_release(self):
        """Head-split RANK release selection — the two heads are NEVER compared in one step.

        Per head: `top-N by that head's own p_ge3`, taken per partition under two caps —
        the partition's slot allocation (release_mix, apportioned) crossed with its
        thin-supply emit cap `floor(passing_supply / 4)` — and a run-wide cap of
        `floors.CLUSTER_CAP` picks per MORPH CLUSTER. `selection.rank_select` is the rule; the
        cluster counter is threaded through BOTH passes, so the cap is per run and two disjoint
        head passes cannot each take two tiles of the same look.

        NO FLOOR GATES THIS. Until 2026-08-09 each pass drew only above its head's release
        floor (0.90 / 0.50); those floors annotate now and every scored row is in the draw.
        What replaced them is the pair of caps above plus the junk floor already applied
        upstream at the colorize-pool draw.

        The head split is UNCHANGED and is not optional: `score` is compared directly inside a
        pass and the two heads' `p_ge3` are on incommensurable train-prior-calibrated scales.
        One pass over both shuts the smaller-scaled head out entirely — that is how 82
        release-eligible strange tiles lost every slot to smooth in the v1 release. Slot
        budget: strange_slots = round(N·strange_frac), smooth = N − strange.

        THE SLOT GUARANTEE (2026-08-10) is applied across BOTH passes before either runs:
        every partition with floor-passing supply gets one slot somewhere in the release, the
        remainder is the mix as before. `_plan_with_guarantees` decides which head owes what;
        the two passes below are otherwise unchanged.

        Honest short-fill: a head that cannot fill its quota under the caps ships fewer. Never
        pad, never backfill from the other head, and never redistribute a slot a thin partition
        could not use — a redistributed slot is the thin-supply rule undone one level up."""
        eligible = self.release_eligible()
        smooth = [r for r in eligible if head_for_style(r["render_style"]) == "wallpaper"]
        strange = [r for r in eligible if head_for_style(r["render_style"]) == "mining"]
        strange_slots = int(round(self.release_n * self.strange_frac))
        smooth_slots = self.release_n - strange_slots

        # ONE cluster counter across both passes — the cap is per RUN (§4).
        cluster_used: dict = {}
        sm_entries, st_entries = self._release_entries(smooth), self._release_entries(strange)
        entries_by_head = {"smooth": sm_entries, "strange": st_entries}
        slot_n = {"smooth": smooth_slots, "strange": strange_slots}
        plans, caps, owed, unseatable = self._plan_with_guarantees(entries_by_head, slot_n)
        sm_slots, sm_caps = plans["smooth"], caps["smooth"]
        st_slots, st_caps = plans["strange"], caps["strange"]
        guar = {h: {p for p, hh in owed.items() if hh == h} for h in entries_by_head}
        sm_sel, sm_log = SEL.rank_select(sm_entries, sm_slots, sm_caps, cluster_used,
                                         guaranteed=guar["smooth"])
        st_sel, st_log = SEL.rank_select(st_entries, st_slots, st_caps, cluster_used,
                                         guaranteed=guar["strange"])
        selected = sm_sel + st_sel
        log = sm_log + st_log
        self.release_log = log
        # Per-slot provenance, keyed by candidate id, for the durable release record.
        self.slot_source = {l["id"]: l["slot_source"] for l in log
                            if l.get("picked") and l.get("slot_source")}

        self.release_split = {
            "strange_frac_target": self.strange_frac,
            "smooth_slots": smooth_slots, "strange_slots": strange_slots,
            "smooth_eligible": len(smooth), "strange_eligible": len(strange),
            "smooth_selected": len(sm_sel), "strange_selected": len(st_sel),
            "strange_frac_realized": (len(st_sel) / len(selected)) if selected else 0.0,
            "strange_modes": dict(Counter(e["_rec"]["render_style"] for e in st_sel)),
            "cluster_cap": SEL.CLUSTER_CAP,
            "n_cluster_cap_skips": sum(1 for l in log if l.get("skip") == "cluster_cap"),
            "partition_slots": {"smooth": sm_slots, "strange": st_slots},
            "emit_caps": dict(self.emit_caps),
            "passing_supply": dict(self.passing_supply),
            # THE SLOT GUARANTEE's whole state, so a sheet or a later readout can say which
            # slots were guarantees without re-deriving the allocation.
            "slot_guarantee": {
                "good_floor": F.GOOD_FLOOR,
                "good_supply": dict(self.good_supply),
                "owed_by_head": {h: sorted(v) for h, v in sorted(guar.items())},
                "unseatable": list(unseatable),
                "n_guarantee_slots": sum(1 for v in self.slot_source.values()
                                         if v == "guarantee"),
            },
        }
        self.release_short_fill = {
            "requested": self.release_n, "eligible": len(eligible), "selected": len(selected),
            "short_by": max(0, self.release_n - len(selected)),
            "smooth_short_by": max(0, smooth_slots - len(sm_sel)),
            "strange_short_by": max(0, strange_slots - len(st_sel)),
        }
        thin = [p for p, n in sorted(self.passing_supply.items()) if not self.emit_caps.get(p)]
        if thin:
            print(f"[select] thin supply — emit 0 (floor(passing/"
                  f"{F.THIN_SUPPLY_DIVISOR}) == 0) beyond any guaranteed slot: "
                  + ", ".join(f"{p} {self.passing_supply[p]} above floor" for p in thin),
                  flush=True)
        if owed:
            print(f"[select] slot guarantee (>=1 candidate above the {F.GOOD_FLOOR:g} good "
                  f"floor): " + ", ".join(f"{p}→{h}" for p, h in sorted(owed.items()))
                  + f" — {self.release_split['slot_guarantee']['n_guarantee_slots']} of "
                    f"{len(selected)} pick(s) are guarantee slots", flush=True)
        if unseatable:
            print(f"[select] slot guarantee UNSEATABLE for {unseatable} — floor-passing supply "
                  f"but no scored candidate in either head pass (a colorize/supply fact, not a "
                  f"slot one)", flush=True)
        if len(selected) < self.release_n:
            print(f"[select] SHORT-FILL {len(selected)}/{self.release_n}: smooth "
                  f"{len(sm_sel)}/{smooth_slots} (elig {len(smooth)}) + strange "
                  f"{len(st_sel)}/{strange_slots} (elig {len(strange)}). Shipping fewer rather "
                  f"than filling past a partition's slot/supply cap or a cluster cap "
                  f"({self.release_split['n_cluster_cap_skips']} cluster-cap skips).",
                  flush=True)
        print(f"[select] head-split rank: smooth {len(sm_sel)} (wallpaper head) + strange "
              f"{len(st_sel)} (mining head) = {len(selected)}; realized strange frac "
              f"{self.release_split['strange_frac_realized']:.2f} (target {self.strange_frac}); "
              f"strange modes {self.release_split['strange_modes']}", flush=True)
        return selected, log

    def render_release(self, selected, skip_render=False):
        # skip_render: reuse the full-res PNGs already on disk (report/sheet regen without
        # re-paying the ~30-min wallpaper-canon render pass).
        if skip_render:
            return [(e["_rec"]["id"], self.release_dir / f"{e['_rec']['id']}.png")
                    for e in selected if (self.release_dir / f"{e['_rec']['id']}.png").exists()]
        dt = _deploy_tail()
        from tools import colormap as cm
        self.release_dir.mkdir(parents=True, exist_ok=True)
        out_paths = []
        for e in selected:
            r = e["_rec"]
            loc = D.location_of(self.by_id[r["location_id"]])
            png = self.release_dir / f"{r['id']}.png"
            # Per-file resume: selection is deterministic from the durable pool, so a relaunch
            # picks the same N — reuse any complete PNG already on disk (a reaper kill mid-pass
            # then only re-renders the missing tiles, never restarts all N). Validate the file
            # is a whole PNG so a truncated mid-write victim is re-rendered, not reused.
            if png.exists():
                try:
                    with Image.open(png) as _im:
                        _im.verify()
                    out_paths.append((r["id"], png))
                    continue
                except Exception:              # noqa: BLE001  truncated/corrupt → re-render
                    png.unlink(missing_ok=True)
            # Timed per render: these are the run's most expensive individual units (ss4 at
            # wallpaper canon) and their spread is wide, so a stage total alone cannot say
            # whether a long release pass was N ordinary renders or one pathological one.
            _t_rel = time.time()
            try:
                # The full-res release render is its OWN render, so the operator measures and
                # stamps it at release geometry rather than inheriting the 960x540 pool row's
                # curve. The stamp lands in `<release_dir>/autolevel_stamps.jsonl`, written by
                # the operator itself (one row per leveled render, keyed by file name).
                render_wallpaper(dt, cm, loc, r["render_style"], r["palette"], png,
                                 self.rel_w, self.rel_h, self.rel_ss, self.rel_filt)
                out_paths.append((r["id"], png))
                _rel_err = False
            except Exception as ex:                  # noqa: BLE001
                print(f"[release] {r['id']} full-res render failed: {ex!r}", flush=True)
                _rel_err = True
            self.stage_times.record("release_render", r["id"], time.time() - _t_rel,
                                    style=r["render_style"], w=self.rel_w, h=self.rel_h,
                                    ss=self.rel_ss, error=_rel_err)
        return out_paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", nargs="+",
                    default=["data/discovery/steered_run2/outcome_ledger.jsonl"],
                    help="one or more run-scoped ledgers; admitted rows are unioned (dedup by id)")
    ap.add_argument("--out", default="scratch/emission_v1")
    ap.add_argument("--library", default=str(D.DEFAULT_LIBRARY_DIR),
                    help="library seed snapshot dir (intake.json + either morph_embs.npz or "
                         "the registry's per-look emb dir) whose per-type medoids SEED this "
                         "intake's clustering, so the batch is deduplicated against the "
                         "library and not only against itself. FAIL-CLOSED: an absent or "
                         "empty seed aborts the run (there is no unseeded mode).")
    ap.add_argument("--report", default=None,
                    help="report .md path (default scratch/emission_v1_report.md)")
    ap.add_argument("--release-n", type=int, default=12)
    ap.add_argument("--strange-frac", type=float, default=DEFAULT_STRANGE_FRAC,
                    help="target strange share of the release; strange_slots = round(N·frac), "
                         "smooth = N − strange. Heads are selected by DISJOINT within-head "
                         "greedy passes (never compared in one step).")
    ap.add_argument("--target-gated", type=int, default=0,
                    help="0 → 3×release-n SCORED rows. REPORTING ONLY since 2026-08-09: the "
                         "colorize volume is the attempt budget (--max-attempts, split 4× per "
                         "release slot per head), and this no longer stops the loop. It is "
                         "still computed and reported — how many scored rows a run built, and "
                         "how many would also have cleared the retired 0.90/0.50.")
    ap.add_argument("--cover-all", action="store_true",
                    help="colorize every admitted location exactly once, then stop (explicit "
                         "one-pass; bypasses --target-gated/--max-attempts/--time-budget-min)")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help=f"wallpaper-head POOL floor for smooth, ANNOTATION-ONLY since "
                         f"2026-08-09 (default {DEFAULT_FLOOR}). It admits and rejects "
                         f"nothing; it is the threshold the `above_pool_floor` column "
                         f"compares against. An override is UNSTAMPED.")
    ap.add_argument("--mining-floor", type=float, default=DEFAULT_MINING_FLOOR,
                    help=f"mining-head POOL floor for strange styles, ANNOTATION-ONLY since "
                         f"2026-08-09 (default {DEFAULT_MINING_FLOOR})")
    ap.add_argument("--release-floor", type=float, default=DEFAULT_RELEASE_FLOOR,
                    help=f"wallpaper-head RELEASE floor, ANNOTATION-ONLY since 2026-08-09 "
                         f"(default = the {DEFAULT_RELEASE_FLOOR} production gate). Recorded "
                         f"as `would_pass_release_floor`; it gates nothing.")
    ap.add_argument("--mining-release-floor", type=float, default=DEFAULT_MINING_RELEASE_FLOOR,
                    help=f"mining-head RELEASE floor, ANNOTATION-ONLY since 2026-08-09 "
                         f"(default = the {DEFAULT_MINING_RELEASE_FLOOR} production gate; it "
                         f"enforced from 2026-08-06 to 2026-08-09). Strange below it now "
                         f"ships if it ranks.")
    ap.add_argument("--release-w", type=int, default=None,
                    help=f"release render width (default wallpaper canon {REL_W})")
    ap.add_argument("--release-h", type=int, default=None,
                    help=f"release render height (default wallpaper canon {REL_H})")
    ap.add_argument("--release-ss", type=int, default=None,
                    help=f"release supersample (default wallpaper canon {REL_SS})")
    ap.add_argument("--release-filt", default=None,
                    help=f"release downsample filter (default {REL_FILT})")
    ap.add_argument("--max-attempts", type=int, default=240,
                    help="THE TOTAL COLORIZE ATTEMPT BUDGET (2026-08-09). Each head asks for "
                         f"{F.ATTEMPT_MULTIPLIER}× its release slots; if the pair exceeds this "
                         "budget BOTH scale down proportionally. The default is far above "
                         f"{F.ATTEMPT_MULTIPLIER}×N for any usual N, so it binds only when set "
                         "low (a smoke) — and then it binds on both heads, never one.")
    ap.add_argument("--time-budget-min", type=float, default=45.0,
                    help="hard-kill backstop; unspent planned attempts are reported when it fires")
    ap.add_argument("--palette-pick", choices=["pref", "deficit"], default="pref",
                    help="within-flavor palette pick: 'pref' = v3-gvo argmax (batch-stable "
                         "default); 'deficit' = serve the running realized chroma×hue deficit "
                         "(restores green/high-chroma/spectral), v3-gvo as within-deficit tiebreaker")
    ap.add_argument("--deficit-lambda", type=float, default=1.5,
                    help="deficit-pick: weight of z(deficit-gain) vs z(v3-gvo) (higher = more spread)")
    ap.add_argument("--deficit-green-boost", type=float, default=1.6,
                    help="deficit-pick: standing over-weight on the green hue bins in the target")
    ap.add_argument("--ephemeral", action="store_true",
                    help="THROWAWAY run: redirect the two durable record stores "
                         "(release_records, mining_gate_reports) under scratch/ so nothing "
                         "reaches data/emission/. The sinks are asserted before the first "
                         "write. Use for every smoke — those stores accumulate by run id, so "
                         "a smoke ADDS rows rather than overwriting them.")
    ap.add_argument("--record-root", default=None,
                    help="explicit record-store root (wins over --ephemeral). A path outside "
                         "data/ is written as scratch; the production default is data/emission.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue (pool log is durable)")
    ap.add_argument("--select-only", action="store_true", help="skip colorize; select from pool")
    ap.add_argument("--no-release-render", action="store_true",
                    help="with --select-only: reuse existing release PNGs (regen report/sheets only)")
    args = ap.parse_args()

    from tools.emission import report as R
    # Priority, the half `creationflags` cannot reach: intake's field renders go through
    # `library_annotate.ensure_field`, which takes no creationflags. On win32 a child inherits
    # its parent's priority class, so lowering this driver once covers every launch site.
    # THE one definition (corpus_common) — never re-derived; a copy silently no-ops.
    import corpus_common as cc
    print(f"[priority] {cc.set_below_normal_priority()}", flush=True)
    eng = EmissionDiversity(args)
    # The three coarse stages are timed AT THEIR CALL SITES, which is the only place the
    # boundaries are unambiguous: `intake` and `select_release` are each one call, and
    # colorize is per-attempt inside `colorize()`. `select` deliberately spans the release
    # record write — it is part of what "gate/pool/select" costs, and billing it elsewhere
    # would invite the reader to think the selection was cheap because the write was not
    # counted.
    with eng.stage_times.timed("intake", "all") as m:
        eng.intake()
        m["n_admitted"] = len(getattr(eng, "rows", ()) or ())
        m["n_ledgers"] = len(eng.ledgers)
    if not args.select_only:
        eng.run_colorize()
    else:
        # build axes so the report has the deficit model populated from the durable log.
        dt = _deploy_tail()
        from tools.studies import conditioned_colorize as cond
        _, cell_to_names = cond.load_cell_map()
        eng.build_axes(dt, cell_to_names, dt.lib())
    with eng.stage_times.timed("select", "all") as m:
        selected, sel_log = eng.select_release()
        rpath, r_tot, r_new, runs_path = eng.write_release_record(selected)
        m["n_selected"] = len(selected)
    print(f"[release-record] durable gate+release record: {r_new} new row(s), {r_tot} "
          f"accumulated → {rpath.relative_to(ROOT)} (population → {runs_path.relative_to(ROOT)})",
          flush=True)
    gpath, n_tot, n_cut, n_cut_sel, pool_c = eng.write_gate_report(selected)
    acting = "ANNOTATION-ONLY"      # a Floor cannot act since 2026-08-09 (floors.py)
    # The `would_cut ∧ selected` join is NON-ZERO again. It was zero by construction from
    # 2026-08-06 (enforcing implies selection implies passing); with the floor annotating, a
    # row below 0.50 can be selected on rank, so the log recovers the free labelled false-cut
    # signal enforcing cost — named here because that recovery is the readable half of the
    # retirement, and a reader who remembers "always 0" would otherwise mistrust it.
    print(f"[gate-report] RELEASE site ({eng.mining_release_floor} floor, {acting}): {n_tot} "
          f"strange candidate(s) logged, {n_cut} below it ({n_cut_sel} of those selected "
          f"anyway — no longer 0 by construction) → {gpath.relative_to(ROOT)}", flush=True)
    print(f"[gate-report] POOL site ({eng.mining_floor} floor, ANNOTATION-ONLY): "
          f"{pool_c['n_would_cut_pool']}/{pool_c['n_with_pool_site']} below it, of those "
          f"{pool_c['n_would_cut_pool_pooled']} pooled and "
          f"{pool_c['n_would_cut_pool_selected']} selected", flush=True)
    rel_paths = eng.render_release(selected, skip_render=args.no_release_render)
    R.write_report(eng, selected, sel_log, rel_paths)


if __name__ == "__main__":
    main()
