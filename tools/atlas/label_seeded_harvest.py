#!/usr/bin/env python
r"""label_seeded_harvest.py — q3/q4 candidate supply seeded from JUDGED-GOOD locations.

WHY THIS EXISTS, AND WHAT IT IS NOT. The supply crawl
(`build_supply_crawl_batches.py`, 2026-08-01) seeded from the walker's own frontier and
produced **0 class-4 in 730** labelled rows. The source race had already answered the
question that failure asks: of seven generation algorithms, the two Matt rated top were
**label-seeded** (nuclei solved at/near locations he had already judged good) and
**neighbourhood** (discs probed around those nuclei) — `tools/sources/run_sheets.py`
sheets 2 and 3. This module is that pair, run at scale, as a candidate QUEUE for Matt to
label. It is a **train-side biased harvest**: the seeds are conditioned on his past
verdicts and the queue is ordered by a fitted score, so nothing measured here is a base
rate and no eval claim is made anywhere.

THE PIPELINE, one seed at a time (a seed is the unit of budget, resume and instrumentation):

  1. **Seed** — a distinct c-plane location in the label corpus resolving to score >= 3
     through `label_store.resolve_score` WITH the amendment overlay. Location corpus only;
     Julia/phoenix are excluded because the operators are undefined on a z-plane
     (`minibrot_maneuvers.PARTITION_DEGREE`), not because they are uninteresting.
  2. **Nucleus at the seed** (`method="snap_at_seed"`) — `snap_to_nucleus_multi` at
     `snap_max_fw_mult=0.75`, i.e. the nucleus counts only if it lies INSIDE the judged
     view. That 0.75 is `sources.src_label_seeded`'s `near = fw * 0.75` and it is what
     makes this label-seeded rather than a global scan wearing a label.
  3. **Neighbourhood expansion** (`method="neighborhood_expand"`) — operator 3 around the
     solved nucleus, which is the sheet-3 mechanism ported (`minibrot_maneuvers` §op3).
  4. **The push ladder** — every atom is framed at `k in {None, 8, 16}`, the set
     `steered_frontier.MAN_K_DEFAULT` pushes. One Newton solve, three framings.
  5. **Screen** — `maneuver_view_screen.screen_view` at the frame the candidate would be
     pushed at, so `composite_v3` and every term under it are measured on the picture
     rather than on the atom. The raw f32 field is kept (`RunFieldCache`) because the
     queue's fitted score needs two derived axes off it.
  6. **Interior pre-filter, UPSTREAM** — `interior_fraction > 0.30` is discarded here, at
     sourcing, not at draw time. Matt's rule, and the fit agrees with it: `interior_fraction`
     carries the largest negative coefficient in `view_fit_v1` (-3.58 standardized).

A-FEASIBILITY IS A RECORDED MARGIN, NOT AN EXCLUSION (the mb19 lesson,
`minibrot_sourcing.md`). `f64_margin_deploy_decades` rides every row and nothing filters on
it; the render is attempted and a row is dropped only on EMPIRICAL failure. The one
geometric refusal that does exclude is the operators' own `f64_spacing_wall` at the descent
node width, which is a statement about a frame that cannot be rendered at all.

WHAT IS RECORDED VS WHAT SURVIVES. Every enumerated candidate is appended to
`candidates.jsonl` with `kept` and, when false, `drop_reason` — so the per-stage discard
counts are auditable rather than asserted, and a later question about the discarded
population is a read rather than a re-run. The QUEUE is the `kept` subset.

  uv run python tools/atlas/label_seeded_harvest.py seeds
  uv run python tools/atlas/label_seeded_harvest.py run --run-dir data/discovery/<run> \
      --budget 150 --wall-budget 170
  uv run python tools/atlas/label_seeded_harvest.py run --run-dir <dir> --resume
  uv run python tools/atlas/label_seeded_harvest.py readout --run-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus",
           ROOT / "tools" / "sources", ROOT / "tools" / "sourcing",
           ROOT / "tools" / "orbital", ROOT / "tools" / "descent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                # noqa: E402
import minibrot_maneuvers as mnv            # noqa: E402  (pure mpmath operators)
import maneuver_view_screen as mvs          # noqa: E402  (the view screen + its cache)
import view_screen as vscr                  # noqa: E402  (composite_v3 + ScreenParams)
import view_field_cache as vfc              # noqa: E402  (the run-local f32 field store)

STAMP = "2026-08-02"
GEN_VERSION = "label_seeded_v2"
SEEDS_REL = f"data/label_seeded_harvest/{STAMP}/seeds.jsonl"

# The push ladder. Pinned to `steered_frontier.MAN_K_DEFAULT` by value here and by a test
# that reads that constant, so the two cannot drift: a run whose ladder differs from what
# the walk pushes is measuring a different population from the one the screen was tuned on.
K_LADDER_SPEC = "none,8,16"

# `sources.src_label_seeded`'s own `near`, and the line between this and a global scan.
SEED_SNAP_FW_MULT = 0.75

# Matt's rule, applied at SOURCING. Strict `>`, stated once and asserted by a test — the
# black-gate parity convention in `CLAUDE.md` is a strict `<` on the other side of the same
# kind of boundary, and an off-by-one-side threshold is invisible in a count.
INTERIOR_DISCARD = 0.30

# Enumeration is ~25x the screening cost (`minibrot_maneuvers` §COST), so the prompt's
# "spend budget there" is these two numbers. `NBH_MAX_PROBES` is the ceiling on the bill and
# is what binds; `NBH_MAX_FOUND` is a ceiling on the answer.
NBH_MAX_FOUND = mnv.NBH_MAX_FOUND        # 8
NBH_MAX_PROBES = mnv.NBH_MAX_PROBES      # 12

SCREEN_WORKERS = mvs.SCREEN_WORKERS      # concurrent engine PROCESSES (the CLAUDE.md cap)
RNG_SEED = 20260802

# Per-unit hard-kill backstop, and it is CLAMPED to what is left of the budget rather than
# being a flat constant. A 900 s backstop inside a 15-minute budget is not a backstop
# (`CLAUDE.md`); this one can never exceed the remaining budget, so one pathological seed
# cannot outlive the run it is inside of.
UNIT_TIMEOUT_S = 180.0
MIN_UNIT_TIMEOUT_S = 20.0

STOP_SENTINEL = "STOP"


# =========================================================================== #
# 1. the seed pool
# =========================================================================== #
# Julia and phoenix seeds are DROPPED, and the reason is structural rather than a
# judgement about the material: `snap_to_nucleus` probes the ATOM DOMAIN of the parameter
# plane, and a julia/phoenix viewport is a z-plane with no nucleus in that sense
# (`minibrot_maneuvers.PARTITION_DEGREE` simply does not define a degree for them). Running
# the operators there would not be a weaker harvest, it would be a category error. The count
# dropped is reported, because "the recipe does not reach these" is a fact about coverage.
C_PLANE_FAMILIES = dict(mnv.PARTITION_DEGREE)


def _family_of(render: dict):
    """The c-plane family of a corpus render block, or None if it is not one.

    A pre-family batch carries no `fractal_type` at all; those rows are mandelbrot UNLESS
    they carry a `c_re`, which is how a Julia row hides in one (`sources._deg2_good_labels`
    found the same thing). Both cases are handled here so the seed pool and the source
    sheets agree on what a degree-2 mandelbrot row is."""
    fam = render.get("fractal_type") or render.get("family")
    if fam is None:
        return None if render.get("c_re") is not None else "mandelbrot"
    return fam if fam in C_PLANE_FAMILIES else None


def build_seed_pool(root: Path = ROOT, *, min_score: int = 3) -> tuple[list[dict], dict]:
    """Every distinct c-plane location resolving to `>= min_score`, with the amendments on.

    Scores resolve through `label_store.resolve_score(row, sidecar, amendments)` — the
    revision overlay included — and NOT off `row["label"]["score"]`, which is null in nine
    batches and would undercount badly. Locations, not crops: several palettes of one view
    are one seed, keyed on the canonical location identity, and the seed carries the MAX
    resolved score over its crops.
    """
    sys.path.insert(0, str(root / "tools" / "corpus"))
    import label_store as ls
    import location as loc_mod

    batches = root / "data" / "label_corpus" / "batches"
    by_key: dict = {}
    rep = Counter()
    for bdir in sorted(batches.iterdir()):
        f = bdir / "images.jsonl"
        if not f.exists():
            continue
        sidecar = ls.sidecar_for(bdir.name)
        amend = ls.amendments_for(bdir.name)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rep["rows"] += 1
            score = ls.resolve_score(row, sidecar, amend)
            if score is None or int(score) < min_score:
                continue
            rep["labeled_ge_min"] += 1
            rd = row["render"]
            fam = _family_of(rd)
            if fam is None:
                rep["dropped_not_c_plane"] += 1
                continue
            key = loc_mod.from_render_block(rd).key()
            cur = by_key.get(key)
            if cur is None:
                by_key[key] = dict(
                    seed_id=None, family=fam, degree=C_PLANE_FAMILIES[fam],
                    cx=str(rd["cx"]), cy=str(rd["cy"]), fw=str(rd["fw"]),
                    score=int(score), batch=bdir.name, image_id=row["image_id"],
                    n_crops=1)
            else:
                cur["n_crops"] += 1
                if int(score) > cur["score"]:
                    cur.update(score=int(score), batch=bdir.name,
                               image_id=row["image_id"])
    seeds = list(by_key.values())
    # Deterministic identity and deterministic ORDER. The id is a hash of the location so it
    # survives a re-derivation; the order is sorted on it rather than on corpus order, which
    # is batch order, which correlates with both family and depth — a budget-truncated run
    # in corpus order would harvest one era of the corpus and call it the corpus.
    for s in seeds:
        s["seed_id"] = _seed_id(s)
    seeds.sort(key=lambda s: s["seed_id"])
    rep["seeds"] = len(seeds)
    return seeds, dict(rep)


def _seed_id(s: dict) -> str:
    import hashlib
    h = hashlib.sha1(f'{s["family"]}|{s["cx"]}|{s["cy"]}|{s["fw"]}'.encode()).hexdigest()
    return f"s{h[:12]}"


def stage_seeds(args) -> int:
    seeds, rep = build_seed_pool(min_score=args.min_score)
    p = paths.durable(SEEDS_REL, mkparents=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s) + "\n")
    print(f"seeds: {rep['seeds']} distinct c-plane locations >= {args.min_score} "
          f"from {rep['labeled_ge_min']} labeled crops "
          f"({rep['dropped_not_c_plane']} crops dropped: not a c-plane family)")
    print("  by (family, score): " + json.dumps(
        {f"{k[0]}|{k[1]}": v for k, v in sorted(
            Counter((s["family"], s["score"]) for s in seeds).items())}))
    print(f"  -> {p}")
    return 0


def load_seeds(root: Path = ROOT) -> list[dict]:
    p = paths.durable(SEEDS_REL)
    if not p.exists():
        raise SystemExit(f"{p} missing — run `seeds` first.")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# =========================================================================== #
# 2. one seed -> candidate views
# =========================================================================== #
def _parent_rec_from(snap: mnv.Maneuver, degree: int) -> dict:
    return dict(id=snap.atom_id, cx=snap.cx, cy=snap.cy, period=snap.period,
                window_scale=snap.window_scale, degree=degree)


# THE SEED SNAP IS SHEET 2's PRIMITIVE, NOT THE MANEUVER OPERATOR'S — measured, not assumed.
# `snap_to_nucleus_multi` ranks periods by an atom-domain orbit pass (the argmins' divisors)
# because inside a walk the probe fires on every rung and the sweep is 84% of maneuver cost.
# Here the probe fires ONCE PER SEED over a fixed pool of 511, so the trade is the other way
# round, and the yields are far apart. Measured on the first 60 seeds, hits / ~Newton solves:
#   full sweep 1..64        34/60    3,840    59.7 s
#   atom-domain n=4         21/60      492     3.3 s   <- what the maneuver operator does
#   hybrid low16 + n=4      21/60    1,022     6.2 s
#   hybrid low32 + n=6      28/60    1,963    19.2 s
# The ranked sets miss 6-13 of the 34 nuclei the sweep finds, and a missed seed costs the
# neighbourhood expansion too, because sheet 3 probes discs around the SHEET-2 nuclei — a
# seed with no parent contributes nothing at all. And the ceiling itself binds: re-trying the
# 26 misses at periods 65..160 recovered 8 more (periods 66-135), i.e. 42/60 against 34/60.
# `[measured: 2026-08-02, first 60 seeds of data/label_seeded_harvest/2026-08-02/seeds.jsonl]`
SEED_PERIOD_MAX = 128


def snap_at_seed(seed: dict, ks, *, period_max: int = SEED_PERIOD_MAX):
    """`(maneuvers, solves)` — the nucleus at/near the judged view, framed at every `k`.

    `al.identify_nucleus` with `near = 0.75 * fw` IS `sources.src_label_seeded`, called with
    the same arguments; the only thing added here is the `k` ladder, which is free (one
    solve, three framings — `snap_to_nucleus_multi` §COST ATTRIBUTION). Emitting `Maneuver`s
    rather than a bespoke record keeps one row shape across both methods, so the framing
    refusals (`f64_spacing_wall`, `fw_over_root_scale`) are counted the same way for both.
    """
    import mpmath as mp
    import atom_lib as al
    al.set_precision()
    deg = int(seed["degree"])
    fw = float(seed["fw"])
    view = dict(cx=seed["cx"], cy=seed["cy"], fw=fw, depth=0, node_id=None)
    c0 = mp.mpc(mp.mpf(str(seed["cx"])), mp.mpf(str(seed["cy"])))
    t0 = time.time()
    rec, why = al.identify_nucleus(
        c0, period_min=1, period_max=int(period_max), degree=deg,
        near=SEED_SNAP_FW_MULT * fw, source="label_seeded_v2_snap",
        provenance={"seed_batch": seed["batch"], "seed_image_id": seed["image_id"],
                    "label_score": seed["score"], "label_fw": fw})
    # Upper bound: the sweep stops at the first success, so a hit costs `period` solves.
    solves = int(period_max) if rec is None else int(rec["period"])
    if rec is None:
        return [mnv._unavailable("snap_at_seed", why, view, t0, solves, k=k) for k in ks], solves
    out = []
    for i, k in enumerate(ks):
        t_row = t0 if i == 0 else time.time()
        n_solves = solves if i == 0 else 0
        shared = {} if i == 0 else {"reused_solve": True}
        newfw, whyf = mnv._frame_for(rec, k, fw, mnv.MAX_FW)
        if newfw is None:
            out.append(mnv._unavailable("snap_at_seed", whyf, view, t_row, n_solves, k=k,
                                        period=rec["period"],
                                        window_scale=rec["window_scale"], **shared))
            continue
        out.append(mnv.Maneuver(
            op="snap_at_seed", available=True, k=(None if k is None else float(k)),
            cx=rec["cx"], cy=rec["cy"], fw=newfw, depth=0,
            atom_id=rec["id"], atom_key=mnv.atom_key_of(rec), period=rec["period"],
            log10_abs_A=rec["log10_abs_A"], window_scale=float(rec["window_scale"]),
            f64_margin_node_decades=round(
                mnv._wall_margin_decades(newfw, mnv.NODE_WIDTH), 4),
            f64_margin_deploy_decades=rec["f64_margin_deploy_decades"],
            parent_node_id=None, parent_cx=float(seed["cx"]), parent_cy=float(seed["cy"]),
            parent_fw=fw, parent_depth=0,
            probe_s=time.time() - t_row, newton_solves=n_solves,
            extra=dict(seed_distance=rec["provenance"].get("seed_distance"),
                       degree=deg, atom_size=rec["size"], **shared)))
    return out, solves


def enumerate_seed(seed: dict, ks, *, rng, deadline: float | None = None,
                   max_found: int = NBH_MAX_FOUND, max_probes: int = NBH_MAX_PROBES,
                   period_max: int = SEED_PERIOD_MAX):
    """`(rows, stats)` — every candidate view this seed produces, both methods.

    `rows` are plain dicts (not `Maneuver`s) because the row is what gets appended to the
    append-only log and screened; a dataclass here would be re-serialized twice for nothing.
    Only AVAILABLE maneuvers become candidate views — an unavailable one is a refusal with a
    named reason and is counted, never queued.

    THE SECOND OPERATOR ONLY RUNS IF THE FIRST SOLVED. That is the recipe, not an
    optimisation: sheet 3 probes discs around the SHEET-2 NUCLEI, so a seed whose own
    nucleus did not solve has no parent to expand around. `parent_rec` is handed in from the
    snap so the expansion does not pay a second solve for the nucleus it already has.
    """
    view = dict(cx=seed["cx"], cy=seed["cy"], fw=float(seed["fw"]), depth=0, node_id=None)
    deg = int(seed["degree"])
    st = Counter()
    rows: list[dict] = []

    t0 = time.time()
    snaps, snap_solves = snap_at_seed(seed, ks, period_max=period_max)
    st["snap_rows"] = len(snaps)
    st["snap_newton_solves"] = snap_solves
    parent = None
    for m in snaps:
        if m.available:
            rows.append(_row(m, seed, "snap_at_seed"))
            parent = parent or m
        else:
            st[f"snap_unavail:{m.reason}"] += 1
    st["snap_available"] = sum(1 for m in snaps if m.available)
    st["enum_snap_s"] = round(time.time() - t0, 3)

    if parent is None:
        st["seed_no_nucleus"] = 1
        return rows, dict(st)

    t1 = time.time()
    nbh = mnv.neighborhood_expand(view, rng, ks, degree=deg,
                                  parent_rec=_parent_rec_from(parent, deg),
                                  max_found=max_found, max_probes=max_probes,
                                  deadline=deadline, source="label_seeded_v2_nbh")
    st["nbh_rows"] = len(nbh)
    st["nbh_newton_solves"] = sum(m.newton_solves for m in nbh)
    for m in nbh:
        if m.available:
            rows.append(_row(m, seed, "neighborhood_expand"))
        else:
            st[f"nbh_unavail:{m.reason}"] += 1
    st["nbh_available"] = sum(1 for m in nbh if m.available)
    st["nbh_distinct_atoms"] = len({m.atom_key for m in nbh if m.available})
    st["enum_nbh_s"] = round(time.time() - t1, 3)
    st["enum_s"] = round(time.time() - t0, 3)
    return rows, dict(st)


def _row(m: mnv.Maneuver, seed: dict, method: str) -> dict:
    """One candidate view: the geometry, its atom, and where the seed came from.

    `candidate_key` is `atom_key|k` — byte-identical to `maneuver_view_screen.view_key`, to
    `view_field_cache.row_key` and to the walk's own visited key, on purpose: the screen,
    the field cache and the dedup must agree on what one candidate is.
    """
    return dict(
        candidate_key=mvs.view_key(m.atom_key, m.k), atom_key=m.atom_key,
        atom_id=m.atom_id, method=method, op=m.op, k=m.k,
        family=seed["family"], degree=int(seed["degree"]),
        cx=m.cx, cy=m.cy, fw=float(m.fw), period=int(m.period),
        window_scale=float(m.window_scale), log10_abs_A=m.log10_abs_A,
        # RECORDED, never filtered on (the mb19 lesson).
        f64_margin_deploy_decades=m.f64_margin_deploy_decades,
        f64_margin_node_decades=m.f64_margin_node_decades,
        scale_ratio_decades=m.extra.get("scale_ratio_decades"),
        found_rank=m.extra.get("found_rank"),
        parent_atom_id=m.extra.get("parent_atom_id"),
        seed_id=seed["seed_id"], seed_batch_id=seed["batch"],
        seed_image_id=seed["image_id"], seed_score=int(seed["score"]),
        seed_fw=float(seed["fw"]),
    )


# =========================================================================== #
# 3. the run
# =========================================================================== #
class Harvest:
    """One resumable run. State is the processed-seed set + counters; the candidates and
    the fields are append-only files, so a kill loses at most the seed in flight.

    THE RNG IS PER-SEED AND DERIVED FROM THE SEED ID, not a run-global stream. A global
    stream would make the neighbourhood probes depend on the ORDER seeds were processed in,
    which a resume changes — so a resumed run would probe different discs from the run it
    resumed. Deriving it per seed makes a unit's enumeration a pure function of the seed.
    """

    def __init__(self, run_dir: Path, *, budget_min: float, wall_budget_min: float,
                 max_found: int = NBH_MAX_FOUND, max_probes: int = NBH_MAX_PROBES,
                 workers: int = SCREEN_WORKERS, period_max: int = SEED_PERIOD_MAX):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget_s = float(budget_min) * 60.0
        self.wall_budget_s = float(wall_budget_min) * 60.0
        self.max_found, self.max_probes = int(max_found), int(max_probes)
        self.period_max = int(period_max)
        self.ks = mnv.parse_k_spec(K_LADDER_SPEC)
        self.done: set = set()
        self.active_s = 0.0
        self.unit_s: list = []
        self.totals = Counter()
        self.reasons = Counter()
        self.per_seed: list = []
        self.params = vscr.screen_params(vscr.load_refs())
        self.fields = vfc.RunFieldCache(self.dir / "view_fields",
                                        policy=mvs.msc.screen_policy_token())
        self.screen = mvs.ViewScreenCache(self.params, workers=int(workers))
        self.seen_keys: set = set()

    # -- persistence -------------------------------------------------------- #
    @property
    def state_path(self):
        return self.dir / "state.json"

    @property
    def log_path(self):
        return self.dir / "candidates.jsonl"

    def save(self):
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(
            schema_version=1, stamp=STAMP, gen_version=GEN_VERSION,
            k_ladder=K_LADDER_SPEC, interior_discard=INTERIOR_DISCARD,
            seed_snap_fw_mult=SEED_SNAP_FW_MULT, seed_period_max=self.period_max,
            max_found=self.max_found, max_probes=self.max_probes,
            done=sorted(self.done), active_s=round(self.active_s, 3),
            unit_s=[round(u, 3) for u in self.unit_s[-400:]],
            totals=dict(self.totals), reasons=dict(self.reasons),
            per_seed=self.per_seed, seen_keys=sorted(self.seen_keys),
            screen=self.screen.state_dict(),
        ), separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def load(self):
        if not self.state_path.exists():
            raise SystemExit(f"{self.state_path} missing — cannot --resume a run that has "
                             f"no state. Drop --resume to start it.")
        d = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.done = set(d.get("done") or [])
        self.active_s = float(d.get("active_s", 0.0))
        self.unit_s = list(d.get("unit_s") or [])
        self.totals = Counter(d.get("totals") or {})
        self.reasons = Counter(d.get("reasons") or {})
        self.per_seed = list(d.get("per_seed") or [])
        self.seen_keys = set(d.get("seen_keys") or [])
        self.screen.load_state(d.get("screen") or {})
        return d

    # -- budget ------------------------------------------------------------- #
    def unit_estimate(self) -> float:
        """What the NEXT unit is expected to cost, from this run's OWN recent throughput.

        The p90 of the last 40 units, not the run-to-date mean: a mean over the whole run is
        dominated by whatever the early cheap units cost, and this number's only job is to
        decide whether starting one more unit can overrun the budget (`CLAUDE.md`, projecting
        a long run's wall clock). p90 rather than the median because the decision is
        one-sided — underestimating overruns the cap, overestimating stops one seed early.
        """
        if not self.unit_s:
            return 30.0
        recent = self.unit_s[-40:]
        return float(np.percentile(recent, 90))

    def remaining(self, t_start: float) -> tuple:
        return (self.budget_s - self.active_s,
                self.wall_budget_s - (time.time() - t_start))

    def stopping(self, t_start: float) -> str:
        if (self.dir / STOP_SENTINEL).exists():
            return "stop_sentinel"
        act_left, wall_left = self.remaining(t_start)
        est = self.unit_estimate()
        if act_left <= est:
            return f"active_budget (est {est:.0f}s > {act_left:.0f}s left)"
        if wall_left <= est:
            return f"wall_budget (est {est:.0f}s > {wall_left:.0f}s left)"
        return ""

    # -- the unit ----------------------------------------------------------- #
    def run_seed(self, seed: dict, t_start: float) -> dict:
        act_left, wall_left = self.remaining(t_start)
        # The backstop, clamped: never longer than what is actually left to spend.
        cap = max(MIN_UNIT_TIMEOUT_S, min(UNIT_TIMEOUT_S, act_left, wall_left))
        t0 = time.time()
        rng = np.random.default_rng(RNG_SEED ^ (int(seed["seed_id"][1:], 16) & 0xFFFFFFFF))
        rows, st = enumerate_seed(seed, self.ks, rng=rng, deadline=t0 + cap * 0.75,
                                  max_found=self.max_found, max_probes=self.max_probes,
                                  period_max=self.period_max)

        # -- write-path dedup, already snapped. `atom_key` is `snapped_dedup_key`, so an
        # atom reached from two seeds collapses here rather than being screened twice and
        # then appearing twice in the queue.
        fresh, dup = [], 0
        for r in rows:
            if r["candidate_key"] in self.seen_keys:
                dup += 1
                continue
            self.seen_keys.add(r["candidate_key"])
            fresh.append(r)
        st["dup_candidates"] = dup

        screened = self.screen.screen_many(
            [dict(view_key=r["candidate_key"], cx=r["cx"], cy=r["cy"], fw=r["fw"],
                  family=r["family"], atom_key=r["atom_key"], k=r["k"]) for r in fresh],
            budget_s=max(1.0, cap - (time.time() - t0)))
        for r in fresh:
            # A key the pass never reached comes back as an unscreened row with a named
            # reason, never as a silent absence — the two are different facts.
            rec = screened.get(r["candidate_key"]) or dict(screened=False,
                                                           screen_reason="not_reached")
            r.update({k: rec.get(k) for k in _SCREEN_COLS})
            r["kept"], r["drop_reason"] = _verdict_for(rec)
        st["screened"] = sum(1 for r in fresh if r.get("screened"))
        st["kept"] = sum(1 for r in fresh if r["kept"])
        for r in fresh:
            if not r["kept"]:
                self.reasons[r["drop_reason"]] += 1

        with open(self.log_path, "a", encoding="utf-8") as f:
            for r in fresh:
                f.write(json.dumps(r, default=str) + "\n")

        dt = time.time() - t0
        self.active_s += dt
        self.unit_s.append(dt)
        for k, v in st.items():
            if isinstance(v, (int, float)) and not k.startswith("enum_"):
                self.totals[k] += v
        self.totals["seeds_done"] += 1
        self.totals["candidates_enumerated"] += len(rows)
        self.totals["candidates_kept"] += st["kept"]
        rec = dict(seed_id=seed["seed_id"], family=seed["family"],
                   degree=seed["degree"], score=seed["score"],
                   enumerated=len(rows), fresh=len(fresh), dup=dup,
                   screened=st["screened"], kept=st["kept"],
                   atoms=len({r["atom_key"] for r in rows}),
                   nbh_atoms=st.get("nbh_distinct_atoms", 0),
                   solves=st.get("snap_newton_solves", 0) + st.get("nbh_newton_solves", 0),
                   seconds=round(dt, 2))
        self.per_seed.append(rec)
        self.done.add(seed["seed_id"])
        return rec


# The screen columns copied onto a candidate row. Named once so the row schema, the queue
# and the batch's provenance block cannot drift apart.
_SCREEN_COLS = ("screened", "screen_reason", "composite", "vetoed", "size_factor",
                "radial_range", "radial_rings", "interior_fraction", "band_coverage",
                "band_coverage_q25", "cap_headroom", "clamped", "view_maxiter",
                mvs.fm.POLICY_KEY)


def _verdict_for(rec: dict) -> tuple:
    """`(kept, drop_reason)` for one screened record. Pure, so the rule is testable.

    Two discards, and they are different kinds of fact:
      * `unscreenable` — the 64x36 screen could not reach this frame, which means a 1280 px
        label crop (20x finer pixel spacing) certainly cannot. There is no crop to label, so
        the row is not a candidate. This is the supply crawl's correction 8, applied at
        sourcing instead of at draw time.
      * `interior_gt_30` — Matt's rule. Strict `>`: a frame at exactly 0.30 is kept.
    """
    if not rec.get("screened"):
        return False, f"unscreenable:{(rec.get('screen_reason') or 'unknown')[:40]}"
    itf = rec.get("interior_fraction")
    if itf is None:
        return False, "unscreenable:no_interior_fraction"
    if float(itf) > INTERIOR_DISCARD:
        return False, "interior_gt_30"
    return True, ""


def stage_run(args) -> int:
    run_dir = Path(args.run_dir)
    h = Harvest(run_dir, budget_min=args.budget, wall_budget_min=args.wall_budget,
                max_found=args.max_found, max_probes=args.max_probes,
                workers=args.workers, period_max=args.seed_period_max)
    h.screen.fields = h.fields
    seeds = load_seeds()
    if args.limit_seeds:
        seeds = seeds[:args.limit_seeds]
    if args.resume:
        h.load()
        print(f"resume: {len(h.done)} seeds done, {h.active_s/60:.1f} min active spent, "
              f"{len(h.seen_keys)} candidate keys seen")
    else:
        if h.state_path.exists():
            raise SystemExit(f"{h.state_path} exists — pass --resume, or point --run-dir "
                             f"at a fresh directory. Refusing to overwrite a run's state.")
        (run_dir / "run.json").write_text(json.dumps(dict(
            stamp=STAMP, gen_version=GEN_VERSION, started=time.strftime("%FT%T"),
            k_ladder=K_LADDER_SPEC, interior_discard=INTERIOR_DISCARD,
            seed_snap_fw_mult=SEED_SNAP_FW_MULT, budget_min=args.budget,
            wall_budget_min=args.wall_budget, n_seeds=len(seeds),
            seed_period_max=args.seed_period_max,
            max_found=args.max_found, max_probes=args.max_probes,
            screen_policy=mvs.msc.screen_policy_token(),
            screen_params=h.params._asdict(),
        ), indent=2) + "\n", encoding="utf-8")

    todo = [s for s in seeds if s["seed_id"] not in h.done]
    print(f"harvest: {len(todo)} seeds to do of {len(seeds)}; budget {args.budget:g} min "
          f"active / {args.wall_budget:g} min wall; ladder {K_LADDER_SPEC}; "
          f"{args.workers} screen processes", flush=True)

    t_start = time.time()
    halted = ""
    for i, seed in enumerate(todo):
        halted = h.stopping(t_start)
        if halted:
            break
        rec = h.run_seed(seed, t_start)
        if (i + 1) % max(1, args.checkpoint_every) == 0 or i == len(todo) - 1:
            h.save()
        if (i + 1) % 10 == 0 or i == 0:
            act_left, wall_left = h.remaining(t_start)
            rate = h.totals["seeds_done"] / max(1e-9, h.active_s / 60.0)
            print(f"  [{h.totals['seeds_done']}/{len(seeds)}] "
                  f"kept {h.totals['candidates_kept']} of "
                  f"{h.totals['candidates_enumerated']}  "
                  f"{rate:.1f} seed/min  est/unit {h.unit_estimate():.0f}s  "
                  f"left {act_left/60:.0f}m active / {wall_left/60:.0f}m wall",
                  flush=True)
    h.save()
    h.fields.close()
    try:
        fin = h.fields.finalize()
    except Exception as e:                                   # noqa: BLE001
        fin = {"error": str(e)[:200]}
    summary = dict(
        halted_by=halted or "seed_pool_exhausted",
        seeds_total=len(seeds), seeds_done=len(h.done),
        seeds_left=len(seeds) - len(h.done),
        active_min=round(h.active_s / 60.0, 1),
        wall_min=round((time.time() - t_start) / 60.0, 1),
        candidates_enumerated=int(h.totals["candidates_enumerated"]),
        candidates_kept=int(h.totals["candidates_kept"]),
        screened=int(h.totals.get("screened", 0)),
        drop_reasons=dict(h.reasons), totals=dict(h.totals),
        fields=fin,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    # NO SILENT CAP: what was not started is named, not left to be inferred from a count.
    print(f"\nhalted by {summary['halted_by']}; {summary['seeds_done']}/{len(seeds)} seeds, "
          f"{summary['seeds_left']} NOT STARTED")
    print(f"  {summary['candidates_enumerated']} candidates enumerated, "
          f"{summary['candidates_kept']} kept; drops {json.dumps(dict(h.reasons))}")
    print(f"  {summary['active_min']} min active / {summary['wall_min']} min wall")
    return 0


# =========================================================================== #
# 4. readout
# =========================================================================== #
def load_candidates(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "candidates.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    # The log is append-only and a kill can replay a seed, so it is a SUPERSET of the
    # population; first occurrence wins, exactly as the maneuver loader does.
    out, seen = [], set()
    for r in rows:
        if r["candidate_key"] in seen:
            continue
        seen.add(r["candidate_key"])
        out.append(r)
    return out


def stage_readout(args) -> int:
    run_dir = Path(args.run_dir)
    rows = load_candidates(run_dir)
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    per_seed = st.get("per_seed") or []
    kept = [r for r in rows if r["kept"]]
    out = dict(
        n_candidates=len(rows), n_kept=len(kept),
        drop_reasons=dict(Counter(r["drop_reason"] for r in rows if not r["kept"])),
        by_method=dict(Counter(r["method"] for r in kept)),
        by_degree=dict(Counter(str(r["degree"]) for r in kept)),
        by_k=dict(Counter(str(r["k"]) for r in kept)),
        by_method_degree=dict(Counter(f'{r["method"]}|d{r["degree"]}' for r in kept)),
        vetoed=sum(1 for r in kept if r.get("vetoed")),
    )
    # THE PROMPT'S OPEN QUESTION: was sheet-3's 22 atoms a SUPPLY limit or a budget cap?
    # A supply limit shows up as `nbh_atoms` sitting BELOW the `max_found` ceiling on most
    # seeds; a budget cap shows up as it sitting AT the ceiling. The two are distinguished
    # by the distribution, not by the mean, so both are reported.
    if per_seed:
        na = np.array([s.get("nbh_atoms", 0) for s in per_seed], dtype=float)
        solved = np.array([1.0 if s.get("atoms", 0) else 0.0 for s in per_seed])
        out["per_seed_yield"] = dict(
            seeds=len(per_seed), seeds_with_a_nucleus=int(solved.sum()),
            nbh_atoms_mean=round(float(na.mean()), 2),
            nbh_atoms_median=float(np.median(na)),
            nbh_atoms_at_ceiling=int((na >= st.get("max_found", NBH_MAX_FOUND)).sum()),
            nbh_atoms_zero=int((na == 0).sum()),
            ceiling=st.get("max_found", NBH_MAX_FOUND),
            hist={str(int(v)): int(c) for v, c in
                  zip(*np.unique(na, return_counts=True))},
            kept_per_seed_mean=round(float(np.mean([s["kept"] for s in per_seed])), 2),
            seconds_per_seed_median=round(float(np.median(
                [s["seconds"] for s in per_seed])), 2))
    p = paths.scratch("label_seeded_harvest", "readout.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"  -> {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds")
    s.add_argument("--min-score", type=int, default=3)
    s.set_defaults(fn=stage_seeds)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--budget", type=float, default=150.0, help="ACTIVE minutes")
    r.add_argument("--wall-budget", type=float, default=170.0, help="WALL minutes")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--limit-seeds", type=int, default=0)
    r.add_argument("--max-found", type=int, default=NBH_MAX_FOUND)
    r.add_argument("--max-probes", type=int, default=NBH_MAX_PROBES)
    r.add_argument("--seed-period-max", type=int, default=SEED_PERIOD_MAX)
    r.add_argument("--workers", type=int, default=SCREEN_WORKERS)
    r.add_argument("--checkpoint-every", type=int, default=5)
    r.set_defaults(fn=stage_run)
    o = sub.add_parser("readout")
    o.add_argument("--run-dir", type=Path, required=True)
    o.set_defaults(fn=stage_readout)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 0) and a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
