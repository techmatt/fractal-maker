r"""build_blind_mining_sheet.py — SHEET E: the BLIND render-mode EVAL slice.

THE MINING ANALOGUE OF SHEET D (`tools/wallpaper/build_blind_minibrot_sheet.py`), and it
exists for the same reason. The v1-vs-v3 clause-(a) comparison is scored on the whole
labeled render-mode corpus, and every batch in it is a CORRECTION sheet: the v1 sitting,
sheet B and sheet C each served mining v1's own suggested tier PREFILLED and ordered the page
by its continuous score. 0.929 of the mining sheet's labels came back equal to what was
served. So the four contested per-mode cells — `curv_linear`, `direct_trap_lines`,
`direct_trap_ring`, `direct_trap_screen`, which fail clause (a) in four or five of the five
staged arms — are measured against a baseline the labels are coupled to by construction.
This sheet buys that same population again with the anchoring removed.

FIVE PROPERTIES, each a rule this builder ENFORCES rather than a claim it makes:

  1. FRESH (location, mode) PAIRS ONLY. Every prior render-mode batch is excluded — by
     location key AND by the proximity guard (`build_fresh_discovery._spatially_in`,
     DEDUP_FRAC 0.5 of min(fw), c-identity aware). The batch list is GLOBBED off
     `data/render_mode_corpus/batches/`, never a constant, and the glob SKIPS THIS SHEET'S
     OWN batch id: once sheet E has written its `images.jsonl` it is one of the batches under
     `batches/`, and a scan that does not skip it excludes its own population (sheet D's
     alternating self-exclusion bug, not re-earned).
     The exclusion is applied at the LOCATION level, which is strictly stronger than the pair
     rule and is affordable here — the human label corpus is 11k locations against the ~350
     the three prior batches spent. `pair_freshness()` reports the pair-level count anyway.

  2. NEITHER MINING CHECKPOINT TOUCHES THE DRAW OR THE SUBSTRATE. No mining head is loaded,
     scored, stamped or sorted on anywhere in this file — asserted by a tokenizing source scan
     in `test_blind_mining_sheet.py`, which fails on the symbol. Location quality is
     conditioned through the HUMAN label corpus (sheet C's standing rule, imported from it);
     the palette is a SCREENED POOL DRAW (`rare_palette_draw.PaletteDrawer` against the
     declared rare-family target, then `palette_deficit.pick` within the family); the mode
     draw is a seeded apportionment; and the near-dup filter breaks ties by DRAW ORDER, which
     is a pure function of the population and the seed rather than of a score.

  3. BLIND SERVING. No `suggested_tier`, no head block, no flat `pred`/score field — so
     `wallpaper_label.html` cannot enter correction mode and has nothing to sort by. Order is
     a SEEDED SHUFFLE stamped into `sheet_order` and served with `&order=file`.

  4. EVAL-ONLY, PERMANENTLY. Every row stamps `split_side="eval"` and the batch stamps
     `eval_only: true`. The slice is also deliberately absent from `near_dup_groups.BATCHES`,
     so `mining_corpus.load_corpus` — which is what the trainer and the reads harness both
     read — cannot pool it into a training set by default.

  5. WEIGHTED TOWARD THE CONTESTED CELLS. The four contested modes get `contested_per_mode`
     rows each; the remainder is dealt round-robin over the other active modes so a pooled
     read exists. `exp_smoothing` is EXCLUDED: it is measured 118/118 and 249/250
     smooth-equivalent (cos p50 0.997) on the two batches that have a smooth-equivalence
     table, and a blind label spent on a smooth twin buys nothing this sheet needs.

THE COLORING is sheet C's, imported rather than restated: one palette per LOCATION shared by
that location's rows (so the smooth twin is one render per location and two rows at a
location differ only in mode), coloured with `deploy_tail._color_params({})` — the canonical
inherited recipe the LIVE emission path uses. The CANVAS is the frozen render-mode corpus
pins (1280x720 ss2 lanczos3 q95), so these rows can be read beside every other mining batch.

    uv run python -u tools/mining/build_blind_mining_sheet.py pool
    uv run python -u tools/mining/build_blind_mining_sheet.py estimate      # + the render bill
    uv run python -u tools/mining/build_blind_mining_sheet.py screen --limit 8    # smoke
    uv run python -u tools/mining/build_blind_mining_sheet.py screen > scratch/blind_mining/screen.log 2>&1
    uv run python -u tools/mining/build_blind_mining_sheet.py select
    uv run python -u tools/mining/build_blind_mining_sheet.py render --limit 6    # bounded E2E
    uv run python -u tools/mining/build_blind_mining_sheet.py render > scratch/blind_mining/render.log 2>&1
    uv run python -u tools/mining/build_blind_mining_sheet.py write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                                            # noqa: E402  THE two draw rules
import corpus_common as cc                                  # noqa: E402  engine launch defaults
import location as loc_mod                                  # noqa: E402
import partitions as PART                                   # noqa: E402  THE partition resolver
# Sheet C owns the human-good location pool and its partition-balanced draw; both are
# IMPORTED, never restated. A second copy of "which locations may a strange sheet stand on"
# is a second answer to the standing rule that a sheet conditions on location quality first.
from tools.mining import build_mining_sheet as BMS          # noqa: E402  THE render paths
from tools.mining import build_rare_palette_sheet as RPS    # noqa: E402  THE location pool
from tools.mining import deploy_tail as DT                  # noqa: E402  THE canonical params
from tools.mining import mining_roster as MR                # noqa: E402  THE class vocabulary
from tools.mining import rare_palette_draw as RPD           # noqa: E402  THE pool palette draw
from tools.mining import smooth_equivalence as SE           # noqa: E402
from tools.palettes import hue_families as HF               # noqa: E402
from tools.scoring import batch_registry as BR              # noqa: E402
from tools.wallpaper.build_fresh_discovery import _spatially_in   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

CORPUS = ROOT / "data" / "render_mode_corpus"

# The SCORING-ONLY screen geometry — sheets B and C's, imported, so all three sheets measure
# smooth-equivalence and near-dup on the same instrument.
SCREEN_GEOM = RPS.SCREEN_GEOM

WORKERS = 4
ENGINE_THREADS = BMS.ENGINE_THREADS

# --------------------------------------------------------------------------- #
# The mode axis, declared above the code that deals it.
# --------------------------------------------------------------------------- #
# The four cells the (28)/(28b) arms contest. DERIVED nowhere: these are the modes whose
# clause-(a) cells fail across the staged arms, and naming them here is what makes the draw
# weighting checkable. `sheet_e_reverdict.py` re-reads the actual failing (arm, cell) pairs
# out of the committed arm reports rather than trusting this tuple.
CONTESTED_MODES = ("curv_linear", "direct_trap_lines", "direct_trap_ring",
                   "direct_trap_screen")
# Measured ~100% smooth-equivalent on both batches with a table
# (scratch/smooth_equivalence/*/smooth_equivalence.json): 118/118 on sheet B at cos p50
# 0.9972, 249/250 on sheet C's own screen. Blind labels are too expensive to spend on a
# mode that is a smooth twin under another name.
EXCLUDED_MODES = ("exp_smoothing",)
ACTIVE_MODES = tuple(m for m in MR.MODES if m not in EXCLUDED_MODES)

# THE ROW SHAPE, declared rather than emergent, and asserted at write time — the same
# mechanism sheet D uses. The labeling rig enters CORRECTION mode iff a row carries a numeric
# `suggested_tier` and shows a machine readout iff a row carries `head_v2_pred` / `pred` /
# `p_ge3`. None of those is in this tuple, and `test_blind_mining_sheet.py` fails if one is
# added.
ROW_KEYS = ("image_id", "sheet_order", "render", "provenance", "label")


# =========================================================================== #
# The sheet spec — a frozen dataclass from the start (CLAUDE.md, "Writing a builder for one
# instance"), so a sheet F is an entry rather than a refactor.
# =========================================================================== #
@dataclass(frozen=True)
class SheetSpec:
    key: str
    batch_id: str
    generator_version: str
    img_prefix: str
    id_salt: str
    target_rows: int
    contested_per_mode: int
    n_locations: int
    max_rows_per_location: int
    oversample: float               # screen candidates per served row
    draw_seed: int
    shuffle_seed: int
    # Per-partition LOCATION caps, in the render bill by name. `phoenix:classic` costs ~54 s
    # per keeper render against ~8 s for mandelbrot; an uncapped slice would own the bill.
    location_caps: dict = field(default_factory=dict)
    classic_partition: str = PART.CLASSIC_PHOENIX

    @property
    def batch_dir(self) -> Path:
        return CORPUS / "batches" / self.batch_id

    @property
    def work(self) -> Path:
        return ROOT / "scratch" / "blind_mining" / self.key

    @property
    def screen_log(self) -> Path:
        return self.work / "screen.jsonl"

    @property
    def embed_store(self) -> Path:
        return self.work / "screen_embeddings.npz"

    @property
    def labels_sidecar(self) -> str:
        """What the MERGE writes and the re-verdict reads. Not the page's export."""
        return f"labels/{self.generator_version}.json"

    @property
    def labels_export(self) -> str:
        """What the page downloads and `merge_sitting --scores` READS. A different file from
        the sidecar above — sheet D's batch record pointed both at the sidecar, which would
        have merged the destination into itself."""
        return f"labels/scores_{self.batch_id}.json"

    @property
    def ui_url(self) -> str:
        # `order=file` honours the builder's stamped shuffle; `tiers=3` is the render-mode
        # scale. There is no `&correction` knob — correction mode is entered by the rows
        # carrying `suggested_tier`, which these do not.
        return (f"tools/viz/wallpaper_label.html?corpus=render_mode_corpus&tiers=3"
                f"&order=file&batch={self.batch_id}")


SHEETS = {
    "e": SheetSpec(
        key="e",
        batch_id="2026-08-11_render_mode_blind_v1",
        generator_version="render_mode_blind_v1",
        img_prefix="bmn",
        id_salt="render_mode_blind_v1/2026-08-11",
        target_rows=150,
        contested_per_mode=25,
        n_locations=150,
        max_rows_per_location=2,
        oversample=1.6,
        draw_seed=29,
        shuffle_seed=20260811,
        location_caps={"phoenix": 16, PART.CLASSIC_PHOENIX: 6},
    ),
}


def log(msg):
    print(msg, flush=True)


# =========================================================================== #
# 1. Population — human-good locations MINUS every location any prior mining batch served.
# =========================================================================== #
def prior_mining_rows(exclude_batch: str | None = None) -> tuple[set, dict, set, dict]:
    """`(location_keys, per_family_coords, served_pairs, per_batch_counts)` over every OTHER
    render-mode batch.

    GLOBBED, not listed: a hardcoded batch list is how batch four silently stops being
    excluded, and "fresh pairs only" then becomes a claim nothing checks.

    `exclude_batch` IS LOAD-BEARING AND THE GLOB IS WHY. Once this sheet has written its own
    `images.jsonl` it is one of the batches under `batches/`, so a scan that does not skip it
    excludes its own locations and the population goes to zero — and the zero-row run would
    still rewrite `images.jsonl`, making the NEXT run's scan find nothing to exclude and
    succeed. Sheet D observed exactly that alternating failure; callers here pass the spec's
    own batch id and `run_write` additionally refuses to truncate a good sheet."""
    keys, coords, pairs, per_batch = set(), defaultdict(list), set(), {}
    root = CORPUS / "batches"
    for bdir in sorted(root.iterdir()) if root.exists() else []:
        if exclude_batch is not None and bdir.name == exclude_batch:
            continue
        p = bdir / "images.jsonl"
        if not p.exists():
            continue
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pv, rd = row["provenance"], row["render"]
            k = pv["location_key"]
            keys.add(k)
            pairs.add((k, rd["render_mode"]))
            n += 1
            loc = loc_mod.from_render_block(rd)
            try:
                coords[loc.family].append((float(loc.cx), float(loc.cy), float(loc.fw),
                                           loc.c_re, loc.c_im))
            except (TypeError, ValueError):
                pass
        per_batch[bdir.name] = n
    return keys, coords, pairs, per_batch


def fresh_locations(spec: SheetSpec) -> tuple[dict, dict]:
    """`(pool, report)` — human >=3 label-corpus locations that NO prior mining batch stands
    on, by key or by proximity.

    Two filters, each read from the module that owns it:
      * the human-quality condition is sheet C's `human_good_locations` (label corpus,
        `label_store.resolve_score` with amendments, location label = MAX over its crops);
      * the freshness condition is the head-corpus proximity guard `_spatially_in`.
    """
    pool = RPS.human_good_locations()
    keys, coords, pairs, per_batch = prior_mining_rows(exclude_batch=spec.batch_id)
    n_key = n_spatial = 0
    fresh = {}
    for k, v in sorted(pool.items()):
        if k in keys:
            n_key += 1
            continue
        if _spatially_in(v["loc"], coords):
            n_spatial += 1
            continue
        fresh[k] = v
    rep = {
        "population": RPS.pool_report(pool),
        "exclusion": {
            "rule": "a location is excluded if its key appears in ANY prior render-mode "
                    "batch, or if it is within DEDUP_FRAC*min(fw) of one at a matching c "
                    "(build_fresh_discovery._spatially_in). LOCATION level, which implies "
                    "the prompt's (location, mode) pair rule and is stronger than it — "
                    "affordable because the label corpus is ~30x the spent population.",
            "prior_batches": per_batch,
            "n_prior_location_keys": len(keys),
            "n_prior_location_mode_pairs": len(pairs),
            "excluded_by_key": n_key,
            "excluded_by_proximity": n_spatial,
            "n_fresh": len(fresh),
            "self_excluded_batch": spec.batch_id,
        },
        "fresh_by_partition": dict(sorted(Counter(v["partition"] for v in fresh.values()).items())),
        "fresh_by_human_score": dict(sorted(Counter(v["score"] for v in fresh.values()).items())),
    }
    return fresh, rep


def pair_freshness(entries, exclude_batch: str | None = None) -> dict:
    """The prompt's own predicate, checked on the built universe rather than assumed from the
    stronger location rule above. Zero here is the property; a non-zero is a regression.

    `exclude_batch` is threaded here for the same reason it is threaded into the filter: once
    this sheet is on disk, a scan that does not skip it reports every one of its own pairs as
    stale — a REPORTING copy of the alternating self-exclusion bug, which is worse than the
    original because it looks like a finding."""
    _keys, _coords, pairs, _pb = prior_mining_rows(exclude_batch=exclude_batch)
    stale = sorted(e["unit_key"] for e in entries
                   if (e["location_key"], e["mode"]) in pairs)
    return {"rule": "(location_key, render_mode) served by any prior render-mode batch",
            "n_universe": len(entries), "n_stale_pairs": len(stale), "stale": stale[:20]}


# =========================================================================== #
# 2. The universe: fresh locations x the active modes, one direct cell per (location, mode).
# =========================================================================== #
def _mode_seed(mode: str) -> int:
    """A stable 32-bit seed from a mode name. `hash()` is salted per process on Windows and
    would make a resumed run draw a different location order than the run it resumes."""
    return int.from_bytes(hashlib.blake2b(mode.encode(), digest_size=4).digest(), "big")


def _screen_stem(spec: SheetSpec, unit_key: str) -> str:
    return hashlib.blake2b(f"{spec.id_salt}|{unit_key}".encode(), digest_size=8).hexdigest()


def mode_targets(spec: SheetSpec, supply: dict) -> tuple[dict, dict]:
    """`(rows_per_mode, report)` — the contested cells first, the remainder dealt.

    The contested four take `contested_per_mode` each; whatever is left of `target_rows` goes
    round-robin over the remaining active modes through `apportion.deal_round_robin`, so a
    mode short on supply drains and its slack lands on the others instead of being lost."""
    take = {m: min(spec.contested_per_mode, supply.get(m, 0)) for m in CONTESTED_MODES}
    rest = max(0, spec.target_rows - sum(take.values()))
    others = {m: supply.get(m, 0) for m in ACTIVE_MODES if m not in CONTESTED_MODES}
    take.update(apportion.deal_round_robin(dict(sorted(others.items())), rest))
    rep = {
        "rule": f"the {len(CONTESTED_MODES)} contested modes take "
                f"{spec.contested_per_mode} rows each; the remaining "
                f"{rest} rows are apportion.deal_round_robin over the other "
                f"{len(others)} active modes (balanced-or-drained). NO score weights this.",
        "contested_modes": list(CONTESTED_MODES),
        "excluded_modes": list(EXCLUDED_MODES),
        "excluded_why": "measured ~100% smooth-equivalent (118/118 on sheet B at cos p50 "
                        "0.9972; 249/250 on sheet C) — a blind label spent on a smooth twin "
                        "buys nothing",
        "active_modes": len(ACTIVE_MODES),
        "target_rows": spec.target_rows,
        "rows_by_mode": dict(sorted(take.items())),
        "supply_by_mode": dict(sorted(supply.items())),
    }
    return take, rep


def draw_pairs(spec: SheetSpec, drawn_locs, targets, oversample=None):
    """`(pairs, report)` — the SCREEN candidate list: `(location, mode)` units, in draw order.

    NO SCORE ORDERS THIS AND NOTHING RANKS IT. Each mode walks its OWN seeded permutation of
    the drawn locations; the modes take turns in a fixed order (contested first) so an early
    mode cannot exhaust the location supply of a late one; and a location is capped at
    `max_rows_per_location` units so the sheet is not a handful of locations seen many ways.
    The result is a pure function of (population, seed, targets).

    `oversample` is the reserve the near-dup and smooth-equivalence filters spend: those run
    AFTER the screen, and a candidate list sized exactly to the target would come out short of
    every cell they touch."""
    over = float(spec.oversample if oversample is None else oversample)
    quota = {m: int(np.ceil(targets.get(m, 0) * over)) for m in ACTIVE_MODES}
    order = [k for k, _v in drawn_locs]
    entry_of = dict(drawn_locs)

    seq = {}
    for m in ACTIVE_MODES:
        rng = np.random.default_rng([spec.draw_seed, _mode_seed(m)])
        idx = rng.permutation(len(order))
        seq[m] = [order[int(i)] for i in idx]

    mode_order = list(CONTESTED_MODES) + [m for m in ACTIVE_MODES if m not in CONTESTED_MODES]
    cursor = {m: 0 for m in ACTIVE_MODES}
    have = Counter()
    used = Counter()
    picked = []
    progress = True
    while progress:
        progress = False
        for m in mode_order:
            if have[m] >= quota[m]:
                continue
            i = cursor[m]
            while i < len(seq[m]) and used[seq[m][i]] >= spec.max_rows_per_location:
                i += 1
            cursor[m] = i + 1
            if i >= len(seq[m]):
                continue
            k = seq[m][i]
            used[k] += 1
            have[m] += 1
            picked.append((k, m))
            progress = True

    entries = [_entry(spec, entry_of[k], k, m, j) for j, (k, m) in enumerate(picked)]
    short = {m: quota[m] - have[m] for m in ACTIVE_MODES if have[m] < quota[m]}
    rep = {
        "rule": "round-robin over modes (contested first), each mode walking its own seeded "
                "permutation of the drawn locations, a location capped at "
                f"{spec.max_rows_per_location} units. Deterministic in (population, seed, "
                "targets); NO score orders it.",
        "oversample": over,
        "quota_by_mode": dict(sorted(quota.items())),
        "drawn_by_mode": dict(sorted(have.items())),
        "short_of_quota": short,
        "n_candidates": len(entries),
        "n_locations_used": len(used),
        "rows_per_location_hist": dict(sorted(Counter(used.values()).items())),
        "seed": spec.draw_seed,
    }
    return entries, rep


def _entry(spec: SheetSpec, v, key: str, mode: str, order: int) -> dict:
    """One candidate unit. The `direct_*` grid cell is ONE per (location, mode) — never two
    sweep cells of the same direct mode at one location, which is the self-duplication class
    sheet B produced 631 times (631 of its 688 same-location near-dup pairs).

    Distinct direct modes at a location take successive cells of a per-location permutation,
    which SPREAD them while the grid was 9 cells and can no longer make them all distinct: the
    grid was coarsened to 3 cells on 2026-08-11 (the opacity axis measured null,
    `mining_roster.DIRECT_GRID`) against 4 direct modes, so by pigeonhole one cell is reused at
    every location. That is a weakening and it is a small one — two DIFFERENT direct modes are
    two different trap shapes, so the cell is not what makes them different pictures, and the
    duplication class that mattered was always the SAME mode twice."""
    loc = v["loc"]
    kind = MR.kind_of(mode)
    mode_params = {}
    if kind == "direct":
        rng = np.random.default_rng([spec.draw_seed, _mode_seed(key)])
        perm = [MR.DIRECT_GRID[int(j)] for j in rng.permutation(len(MR.DIRECT_GRID))]
        # WHICH mode takes which cell is permuted per location too. With 4 direct modes over a
        # 3-cell grid some pair must share, and a fixed `DIRECT_MODES.index(mode) % 3` makes it
        # ALWAYS the same pair (ring and lines), everywhere — a systematic confound rather than
        # a pigeonhole. Permuting the modes as well moves which pair collides from location to
        # location, so no two direct modes are cell-identical across the sheet.
        slot = {m: int(j) for m, j in
                zip(MR.DIRECT_MODES, rng.permutation(len(MR.DIRECT_MODES)))}
        op, th = perm[slot[mode] % len(perm)]
        mode_params = {"direct_opacity": op, "direct_threshold": th}
    cp = dict(v["color_params"])
    e = {
        "unit_key": f"{key}|{mode}",
        "location_key": key, "mode": mode, "kind": kind,
        "family": loc.family, "partition": v["partition"],
        "human_score": v["score"], "hue_family": v["hue_family"],
        "palette": v["palette"], "color_params": cp,
        "render": RPS.render_block_of(loc, v["palette"]),
        "mode_params": mode_params, "draw_order": order,
    }
    e["image_id"] = _screen_stem(spec, e["unit_key"])
    return e


def _smooth_entry(spec: SheetSpec, v, key: str) -> dict:
    """The location's SMOOTH twin — rendered at screen geometry for the equivalence measure
    and NEVER served. Sheet E is 150 STRANGE rows; a smooth comparison slice would spend blind
    labels on the one mode this instrument has no question about."""
    e = _entry(spec, v, key, MR.SMOOTH_MODE, -1)
    e["unit_key"] = f"{key}|{MR.SMOOTH_MODE}"
    e["image_id"] = _screen_stem(spec, e["unit_key"])
    return e


def universe(spec: SheetSpec):
    """`(candidates, twins, loc_meta, report)` — deterministic and pure, recomputed by every
    stage so no scratch file a `rm -r scratch/*` could take out is load-bearing."""
    pool, pop_rep = fresh_locations(spec)
    drawn, draw_rep = RPS.draw_locations(spec, pool)
    drawer = RPD.PaletteDrawer(len(drawn), seed=spec.draw_seed)
    cparams = DT._color_params({})

    loc_meta = {}
    enriched = []
    for key, v in drawn:
        palette, family = drawer.take()
        cp = dict(cparams)
        cp["palette"] = palette
        cp["palette_type"] = None        # filled at render time by the worker's library
        cp["palette_source"] = None
        cp["interior_color"] = [0.0, 0.0, 0.0]
        w = dict(v)
        w["palette"], w["hue_family"], w["color_params"] = palette, family, cp
        enriched.append((key, w))
        loc_meta[key] = {"palette": palette, "hue_family": family,
                         "partition": v["partition"], "human_score": v["score"],
                         "label_batches": sorted(v["batch_ids"]), "family": v["loc"].family}

    supply = {m: len(enriched) for m in ACTIVE_MODES}
    targets, target_rep = mode_targets(spec, supply)
    candidates, pair_rep = draw_pairs(spec, enriched, targets)
    used = {e["location_key"] for e in candidates}
    twins = [_smooth_entry(spec, w, k) for k, w in enriched if k in used]

    candidates.sort(key=lambda e: (e["location_key"], e["mode"]))
    twins.sort(key=lambda e: e["location_key"])

    rep = {
        "population": pop_rep,
        "location_draw": draw_rep,
        "palette_draw": drawer.report(),
        "color_params": {**cparams,
                         "owner": "tools/mining/deploy_tail._color_params({}) — the canonical "
                                  "inherited coloring the LIVE emission path uses",
                         "one_palette_per": "LOCATION, shared by that location's rows, so the "
                                            "smooth twin is one render per location and two "
                                            "rows at a location differ ONLY in mode"},
        "roster": {"n_modes": len(MR.MODES), "n_active": len(ACTIVE_MODES),
                   "excluded": list(EXCLUDED_MODES),
                   "kinds": dict(Counter(MR.MODE_KIND[m] for m in ACTIVE_MODES))},
        "mode_targets": target_rep,
        "candidate_draw": pair_rep,
        "n_candidates": len(candidates),
        "n_smooth_twins": len(twins),
        "candidates_by_mode": dict(sorted(Counter(e["mode"] for e in candidates).items())),
        "candidates_by_partition": dict(sorted(Counter(e["partition"] for e in candidates).items())),
        "direct_rule": f"ONE cell per (location, direct mode), off a per-location permutation "
                       f"of the {len(MR.DIRECT_GRID)}-cell DIRECT_GRID — so no direct mode "
                       f"appears twice at a location (sheet B's self-dup class). Distinct "
                       f"direct modes take successive cells; with "
                       f"{len(MR.DIRECT_MODES)} direct modes over {len(MR.DIRECT_GRID)} cells "
                       f"they cannot all differ",
        "pair_freshness": pair_freshness(candidates, exclude_batch=spec.batch_id),
    }
    return candidates, twins, loc_meta, rep


# =========================================================================== #
# 3. The render bill — measured per-partition costs, projected onto THIS draw.
# =========================================================================== #
BILL_PROBE = ROOT / "scratch" / "sittings27c" / "bill_probe.json"
BILL_SPEEDUP = 3.0      # 4 workers, measured on the 160-unit bounded screen (sheet C's bill)


def render_bill(spec: SheetSpec, candidates, twins) -> dict:
    """The bill, stated UP FRONT and per partition, off the 2026-08-10 measured probe.

    Reported as UNKNOWN rather than silently dropped when the probe is absent: `scratch/` is
    wiped wholesale, and a bill that quietly disappears reads as a free run."""
    out = {"basis": BILL_PROBE.relative_to(ROOT).as_posix(),
           "speedup_assumed": BILL_SPEEDUP,
           "speedup_basis": "4 workers x 3 rayon threads at BELOW_NORMAL — sheet C's measured "
                            "screen speedup, not a core count",
           "n_screen_units": len(candidates) + len(twins),
           "n_keeper_units": spec.target_rows}
    if not BILL_PROBE.exists():
        out["status"] = f"UNKNOWN — {out['basis']} is not on disk (scratch/ is disposable)"
        return out
    probe = json.loads(BILL_PROBE.read_text(encoding="utf-8"))
    screen_n = Counter(e["partition"] for e in candidates + twins)
    # The keeper draw is not known until `select`, so the bill projects the CANDIDATE mix
    # scaled to the served row count — stated, because a projection is not a measurement.
    scale = spec.target_rows / max(1, len(candidates))
    keep_n = {p: n * scale for p, n in Counter(e["partition"] for e in candidates).items()}
    stages = {}
    for name, counts, key in (("screen", screen_n, "screen"),
                              ("keeper", keep_n, "keeper")):
        per = {}
        for p, n in sorted(counts.items()):
            mean = (probe.get(key, {}).get(p) or {}).get("mean_s")
            per[p] = {"units": round(float(n), 1),
                      "mean_s": mean,
                      "single_process_min": round(n * mean / 60.0, 2) if mean else None}
        total = sum(v["single_process_min"] or 0.0 for v in per.values())
        stages[name] = {"per_partition": per,
                        "single_process_min": round(total, 1),
                        "wall_min_at_4x": round(total / BILL_SPEEDUP, 1)}
    out["stages"] = stages
    out["total_wall_min_at_4x"] = round(sum(s["wall_min_at_4x"] for s in stages.values()), 1)
    ph = {p: v for p, v in stages["keeper"]["per_partition"].items() if p.startswith("phoenix")}
    out["phoenix"] = {
        "share_of_keeper_units": round(sum(v["units"] for v in ph.values())
                                       / max(1.0, spec.target_rows), 4),
        "per_partition": ph,
        "note": "phoenix:classic is the expensive slice (~54 s per keeper render against ~8 s "
                "for mandelbrot), which is why it carries its own location cap. It is empty "
                "whenever sheet C already spent its whole 7-location supply.",
    }
    out["projection_caveat"] = (
        "the keeper stage projects the CANDIDATE partition mix scaled to the served row "
        "count; the served mix is not known until `select` runs its filters. Reproject from "
        "the run's own observed rate rather than restating this "
        "(CLAUDE.md, 'projecting a long run's wall clock').")
    return out


# =========================================================================== #
# 4. Screen — colored-CLIP embeddings off ONE render. NO head scores anything.
# =========================================================================== #
def load_screen(spec: SheetSpec) -> dict:
    done = {}
    if spec.screen_log.exists():
        for line in spec.screen_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit_key"]] = rec
    return done


def run_screen(spec: SheetSpec, args):
    """Embed every candidate and every smooth twin at the screen geometry.

    THE ONE DELIBERATE DIFFERENCE FROM SHEET C'S SCREEN, and it is the point of this sheet:
    sheet C also ran the mining head over each screen crop and used its score to rank the
    per-location take. Nothing here loads a head. The screen buys exactly two things — the
    smooth-equivalence exclusion and the near-dup filter — and both are questions about
    pictures, not about quality."""
    candidates, twins, _lm, rep = universe(spec)
    print_universe(rep)
    units = candidates + twins
    done, emb = load_screen(spec), RPS.load_embeddings(spec)
    todo = [e for e in units if e["unit_key"] not in done or e["unit_key"] not in emb]
    if args.limit:
        todo = RPS.spread_over(todo, args.limit, keys=("mode",))
        log(f"[screen] --limit {args.limit}: SPREAD over "
            f"{len({e['mode'] for e in todo})} modes / "
            f"{len({e['partition'] for e in todo})} partitions")
    log(f"[screen] units {len(units)} ({len(candidates)} candidates + {len(twins)} twins) · "
        f"done {len(done)} · todo {len(todo)} · {args.workers} workers x "
        f"{ENGINE_THREADS} threads · geom {SCREEN_GEOM}")
    if not todo:
        return

    crops = spec.work / "screen_crops"
    fields = spec.work / "screen_fields"
    crops.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    embedder = SE.Embedder()
    log(f"[screen] colored-CLIP {embedder.model_name} on {embedder.device} · "
        f"NO quality head is loaded anywhere in this run")

    timeout_s = max(90.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    t0, n, errs, pending = time.time(), 0, [], []

    def flush(batch):
        if not batch:
            return
        paths = [crops / f"{e['image_id']}.jpg" for e in batch]
        vecs = embedder.embed_paths(paths)
        with spec.screen_log.open("a", encoding="utf-8") as fh:
            for e in batch:
                fh.write(json.dumps({
                    "unit_key": e["unit_key"], "mode": e["mode"], "kind": e["kind"],
                    "location_key": e["location_key"], "partition": e["partition"],
                    "family": e["family"], "palette": e["palette"],
                    "hue_family": e["hue_family"], "human_score": e["human_score"],
                    "mode_params": e["mode_params"], "draw_order": e["draw_order"],
                }) + "\n")
        for e, v in zip(batch, vecs):
            emb[e["unit_key"]] = np.asarray(v, dtype=np.float32)
        RPS.save_embeddings(spec, emb)
        for p in paths:
            p.unlink(missing_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=BMS._init_worker) as ex:
        futs = {ex.submit(BMS.render_one,
                          (e, str(crops), str(fields), timeout_s, SCREEN_GEOM)): e
                for e in todo}
        by_id = {e["image_id"]: e for e in todo}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:                         # noqa: BLE001
                errs.append({"unit_key": e["unit_key"], "mode": e["mode"],
                             "partition": e["partition"],
                             "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                log(f"[screen] ERR {e['unit_key']}: {str(exc)[:160]}")
                continue
            pending.append(by_id[res["image_id"]])
            n += 1
            if len(pending) >= 64:
                flush(pending)
                pending = []
            if n % 100 == 0:
                el = time.time() - t0
                log(f"[screen] {n}/{len(todo)}  {len(errs)} failed  {n/el:.2f} row/s -> eta "
                    f"{(len(todo)-n)/(n/el)/60:.0f} min (elapsed {el/60:.0f} min)")
    flush(pending)
    (spec.work / "screen_errors.json").write_text(json.dumps(errs, indent=1), encoding="utf-8")
    log(f"[screen] done: {n} screened, {len(errs)} failed, {(time.time()-t0)/60:.1f} min")


# =========================================================================== #
# 5. Select — smooth-equivalence exclusion, near-dup filter, then the per-mode take.
# =========================================================================== #
def select(spec: SheetSpec, candidates, twins, emb: dict, targets: dict):
    """`(selected, report)`. Three rules, in this order, each recorded:

      1. EXCLUDE smooth-equivalent rows — cos to their own location's smooth twin >=
         `SE.STRICT_CUT`. An UNMEASURED row is dropped too: "we could not measure this" must
         never read as "this is distinct".
      2. Near-dup filter over the survivors at `SE.STRICT_CUT`, greedy in DRAW ORDER. Sheet C
         broke these ties best-first by the mining score; there is no score here, so the
         survivor of a collision is the earlier-drawn row — a pure function of the population
         and the seed.
      3. The per-mode take, in draw order, up to each mode's target.
    """
    by_key = {e["unit_key"]: e for e in candidates}
    twin_of = {e["location_key"]: e["unit_key"] for e in twins}

    rows = []
    n_no_twin = 0
    for e in sorted(candidates, key=lambda e: e["draw_order"]):
        r = dict(e)
        tw = twin_of.get(e["location_key"])
        if tw is None or tw not in emb or e["unit_key"] not in emb:
            r["cos_smooth"], r["band"] = None, "unmeasured"
            n_no_twin += 1
        else:
            c = float(np.dot(emb[e["unit_key"]], emb[tw]))
            r["cos_smooth"], r["band"] = c, SE.band_of(c)
        rows.append(r)

    # THE TWO EXCLUSIONS ARE COUNTED APART. Sheet C reported one number over both bands, so a
    # slice where the twin render failed read as a slice full of smooth-equivalent rows. They
    # are different facts: one is "this mode looks like smooth here", the other is "we could
    # not measure it" — and only the first is a statement about a mode.
    keep = [r for r in rows if r["band"] in ("distinct", "interleave")]
    excluded = [r for r in rows if r["band"] == "near_dup"]
    unmeasured = [r for r in rows if r["band"] == "unmeasured"]

    dist = {}
    for m in ACTIVE_MODES:
        ms = [r for r in rows if r["mode"] == m and r["cos_smooth"] is not None]
        dist[m] = {"n": len(ms),
                   "share_distinct": (sum(1 for r in ms if r["band"] == "distinct") / len(ms))
                   if ms else 0.0,
                   "median_cos": float(np.median([r["cos_smooth"] for r in ms])) if ms else None}

    kept, kept_vecs, dropped = [], [], []
    for r in keep:
        v = emb.get(r["unit_key"])
        if v is None:
            dropped.append({**_thin(r), "why": "no embedding"})
            continue
        if kept_vecs:
            cs = np.stack(kept_vecs) @ v
            j = int(np.argmax(cs))
            if float(cs[j]) >= SE.STRICT_CUT:
                dropped.append({**_thin(r), "why": "near-dup of a kept row",
                                "dup_cos": float(cs[j]), "dup_of": kept[j]["unit_key"]})
                continue
        kept.append(r)
        kept_vecs.append(v)

    have, selected = Counter(), []
    for r in kept:
        m = r["mode"]
        if have[m] >= targets.get(m, 0):
            continue
        have[m] += 1
        r["bucket"] = "contested" if m in CONTESTED_MODES else "pooled_read"
        selected.append(r)
    selected.sort(key=lambda r: r["unit_key"])

    short = {m: targets[m] - have[m] for m in sorted(targets) if have[m] < targets[m]}
    rep = {
        "target_rows": spec.target_rows, "drawn_rows": len(selected),
        "screened": len(rows),
        "targets_by_mode": dict(sorted(targets.items())),
        "smooth_equivalence": {
            **SE.yardstick_block(),
            "measured_at_geometry": list(SCREEN_GEOM),
            "excluded_smooth_equivalent": len(excluded),
            "unmeasured_dropped": len(unmeasured),
            "unmeasured_note": "counted APART from the smooth-equivalent exclusion: 'we could "
                               "not measure this' must not read as 'this is a smooth twin'. "
                               "Both are dropped — an unmeasured row must not read as "
                               "distinct either.",
            "n_rows_without_a_twin_embedding": n_no_twin,
            "bands_over_candidates": dict(Counter(r["band"] for r in rows)),
            "by_mode": dist,
        },
        "near_dup_filter": {
            "cut": SE.STRICT_CUT, "substrate": "colored_clip",
            "n_dropped": len(dropped), "dropped": dropped[:60],
            "rule": "greedy in DRAW ORDER over the survivors — no two served rows within cos "
                    f"{SE.STRICT_CUT}, INCLUDING two modes of one location. The survivor of a "
                    "collision is the earlier-drawn row; NO score breaks the tie.",
        },
        "buckets": dict(Counter(r["bucket"] for r in selected)),
        "short_of_target": short,
        "drawn_by_mode": dict(sorted(Counter(r["mode"] for r in selected).items())),
        "drawn_by_kind": dict(sorted(Counter(r["kind"] for r in selected).items())),
        "drawn_by_partition": dict(sorted(Counter(r["partition"] for r in selected).items())),
        "drawn_by_hue_family": {f: sum(1 for r in selected if r["hue_family"] == f)
                                for f in HF.FAMILIES},
        "drawn_by_human_score": dict(sorted(Counter(r["human_score"] for r in selected).items())),
        "distinct_locations": len({r["location_key"] for r in selected}),
        "distinct_palettes": len({r["palette"] for r in selected}),
        "rows_per_location_hist": dict(sorted(Counter(
            Counter(r["location_key"] for r in selected).values()).items())),
        "seed": spec.draw_seed,
        "unused_candidates": len(by_key) - len(selected),
    }
    return selected, rep


def _thin(r) -> dict:
    return {k: r[k] for k in ("unit_key", "mode", "partition", "palette", "cos_smooth")
            if k in r}


def _selected(spec: SheetSpec, args):
    candidates, twins, loc_meta, uni = universe(spec)
    screen = load_screen(spec)
    if not screen:
        raise SystemExit("[select] no screen records — run `screen` first")
    emb = RPS.load_embeddings(spec)
    targets = {m: n for m, n in uni["mode_targets"]["rows_by_mode"].items()}
    sel, rep = select(spec, candidates, twins, emb, targets)
    by_key = {e["unit_key"]: e for e in candidates}
    return by_key, loc_meta, uni, sel, rep, render_bill(spec, candidates, twins)


# =========================================================================== #
# 6. Render — the frozen corpus pins. NO head scores the result.
# =========================================================================== #
def ledger_path(spec: SheetSpec) -> Path:
    return spec.batch_dir / "_progress_ledger.jsonl"


def load_ledger(spec: SheetSpec) -> dict:
    done, p, crops = {}, ledger_path(spec), spec.batch_dir / "crops"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if (crops / f"{rec['image_id']}.jpg").exists():
                    done[rec["unit_key"]] = rec
    return done


def run_render(spec: SheetSpec, args):
    by_key, _lm, _uni, selected, rep, _bill = _selected(spec, args)
    print_composition(rep)
    if args.dry_run:
        return
    crops = spec.batch_dir / "crops"
    fields = spec.batch_dir / "_fields"
    crops.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    done = load_ledger(spec)
    todo = [r for r in selected if r["unit_key"] not in done]
    if args.limit:
        # SPREAD over MODE cells, not a linspace over a location-major list: a stride sharing
        # a factor with the mode count walks one render path for the whole run and calls it an
        # end-to-end. Keyed on mode alone rather than (mode, partition) because the three
        # render PATHS are per-kind and the kinds are per-mode — a (mode, partition) deal at
        # n=8 spends every slot inside the first mode and exercises one path.
        todo = RPS.spread_over(todo, args.limit, keys=("mode",))
        log(f"[render] --limit {args.limit}: SPREAD -> modes "
            f"{sorted({r['mode'] for r in todo})}")
    log(f"[render] drawn {len(selected)} · done {len(done)} · todo {len(todo)} · "
        f"{args.workers} workers x {ENGINE_THREADS} threads")
    if not todo:
        return
    timeout_s = max(90.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    t0, n, errors = time.time(), 0, []
    with ledger_path(spec).open("a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=BMS._init_worker) as ex:
            futs = {ex.submit(BMS.render_one,
                              (by_key[r["unit_key"]], str(crops), str(fields), timeout_s)):
                    r["unit_key"] for r in todo}
            for fut in as_completed(futs):
                uk = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:                     # noqa: BLE001
                    errors.append({"unit_key": uk,
                                   "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                    log(f"[render] ERR {uk}: {str(exc)[:160]}")
                    continue
                res["unit_key"] = uk
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                n += 1
                if n % 25 == 0 or n <= 3:
                    el = time.time() - t0
                    log(f"[render] {n}/{len(todo)}  {n/el:.2f} row/s -> eta "
                        f"{(len(todo)-n)/(n/el)/60:.0f} min (elapsed {el/60:.0f} min)")
    log(f"[render] done: {n} crops in {(time.time()-t0)/60:.1f} min, {len(errors)} errors")
    ep = spec.batch_dir / "_render_errors.json"
    prior = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else []
    merged = {e["unit_key"]: e for e in prior}
    merged.update({e["unit_key"]: e for e in errors})
    now_done = set(load_ledger(spec))
    merged = {k: v for k, v in merged.items() if k not in now_done}
    ep.write_text(json.dumps(sorted(merged.values(), key=lambda e: e["unit_key"]), indent=1),
                  encoding="utf-8")


# =========================================================================== #
# 7. Write — seeded shuffle, blind rows, eval-only stamp.
# =========================================================================== #
def provenance_block(spec, entry, rec, loc_meta, transfer_dropped, split_side) -> dict:
    p = entry["color_params"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "mining_blind_eval_slice",
        "family": entry["family"],
        "partition": entry["partition"],
        "location_key": entry["location_key"],
        "render_mode": entry["mode"],
        "mode_kind": entry["kind"],
        "mode_params": dict(entry.get("mode_params", {})),
        "rolloff": MR.rolloff_token(entry["mode"]),
        # THE COLORMAP RECIPE — the crop is a pure function of `render` + this block.
        "color_params": {
            "palette": entry["palette"],
            "palette_type": p.get("palette_type"), "palette_source": p.get("palette_source"),
            "reverse": p["reverse"], "log_premap": p["log_premap"], "gamma": p["gamma"],
            "phase": p["phase"], "n_cycles": p["n_cycles"],
            "transfer": p["transfer"], "transfer_gamma": p["transfer_gamma"],
            "interior_color": list(p.get("interior_color", [0.0, 0.0, 0.0])),
        },
        "transfer_dropped": transfer_dropped,
        "hue_family": entry["hue_family"],
        "palette_source_rule": "screened POOL draw — rare_palette_draw.PaletteDrawer against "
                               "the declared rare hue-family target, then palette_deficit.pick "
                               "within the family. No head proposes it and no head ranks it.",
        "bucket": rec["bucket"],
        "draw_order": entry["draw_order"],
        "cos_smooth": rec.get("cos_smooth"),
        "smooth_band": rec.get("band"),
        "screen_path": f"the keeper render paths at a SCORING-ONLY geometry "
                       f"{SCREEN_GEOM[0]}x{SCREEN_GEOM[1]}ss{SCREEN_GEOM[2]} "
                       f"(build_mining_sheet.render_one with `geom`) — embedded, never scored",
        "split_side": split_side,
        "split_origin": "blind_eval_only",
        "source": {
            "corpus": "data/label_corpus",
            "human_score": entry["human_score"],
            "label_batches": loc_meta.get("label_batches"),
            "rule": "location label = MAX over its crops, resolved through "
                    "label_store.resolve_score with amendments applied. THE ONLY quality "
                    "condition on this row, and it is a HUMAN one.",
        },
    }


def run_write(spec: SheetSpec, args):
    by_key, loc_meta, uni, selected, sel_rep, bill = _selected(spec, args)
    done = load_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")
    live = [r for r in selected if r["unit_key"] in done]
    # FAIL BEFORE TRUNCATING. `images.jsonl` is opened "w" below, so a run that reaches that
    # line with nothing to write REPLACES a good sheet with an empty one — and an empty
    # images.jsonl then feeds back into the next run's own prior-batch scan. That is how
    # sheet D's self-exclusion turned into an ALTERNATING failure instead of a loud one.
    if not live:
        raise SystemExit(
            f"[write] {len(selected)} selected, {len(done)} in the render ledger, 0 in both — "
            f"refusing to overwrite {spec.batch_dir / 'images.jsonl'} with an empty sheet. "
            f"The draw and the ledger disagree; re-run `select` and compare.")

    # PRESENTATION ORDER — a SEEDED SHUFFLE, stamped. Not sorted, not grouped, not scored.
    rng = np.random.default_rng([spec.shuffle_seed, 1])
    order = sorted(live, key=lambda r: r["unit_key"])
    order = [order[int(i)] for i in rng.permutation(len(order))]

    rows = []
    for i, rec in enumerate(order):
        e = by_key[rec["unit_key"]]
        loc = loc_mod.from_render_block(e["render"])
        image_id = f"{spec.img_prefix}{i:04d}_{e['image_id'][:8]}"
        rows.append({
            "image_id": image_id,
            "sheet_order": i,
            "render": BMS.render_block(e, loc),
            "provenance": provenance_block(
                spec, e, rec, loc_meta[e["location_key"]],
                bool(done[rec["unit_key"]]["transfer_dropped"]), "eval"),
            # THE HUMAN SLOT, and the ONLY tier field on the row. There is no
            # `suggested_tier`, no head block, no flat score: the labeling rig enters
            # correction mode iff a row carries `suggested_tier`, so their absence is what
            # makes the sheet blind, and it is enforced by test_blind_mining_sheet.py.
            "label": {"score": None, "labeler": None, "labeled_at": None},
            "_unit_key": rec["unit_key"], "_crop_stem": e["image_id"],
        })
    assert len({r["image_id"] for r in rows}) == len(rows), "opaque ids collided"
    for r in rows:
        extra = set(r) - set(ROW_KEYS) - {"_unit_key", "_crop_stem"}
        assert not extra, f"{r['image_id']}: undeclared row field(s) {sorted(extra)} — a row " \
                          f"field this sheet did not declare is how a blind sheet stops " \
                          f"being blind (ROW_KEYS)"

    crops = spec.batch_dir / "crops"
    for r in rows:
        dst = crops / f"{r['image_id']}.jpg"
        if not dst.exists():
            shutil.copyfile(crops / f"{r['_crop_stem']}.jpg", dst)
    route = {r["image_id"]: {"unit_key": r.pop("_unit_key"), "crop_stem": r.pop("_crop_stem")}
             for r in rows}

    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
    # INCOMPLETE is DERIVED from the counts, never a flag: a bounded `--limit` run and a
    # killed run both produce a short batch and only one of them would have set one.
    incomplete = len(rows) < sel_rep["drawn_rows"]
    accounted = set(done) | {e["unit_key"] for e in errors}
    unaccounted = sorted(r["unit_key"] for r in selected if r["unit_key"] not in accounted)
    reg = BR.lookup(spec.batch_id, "mandelbrot")

    spec.batch_dir.mkdir(parents=True, exist_ok=True)
    with (spec.batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (spec.batch_dir / "route.json").write_text(json.dumps(route, indent=1), encoding="utf-8")

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_id": spec.batch_id,
        "generator_version": spec.generator_version,
        "labeler": None,
        "n_rows": len(rows),
        "schema_note":
            "SHEET E — the BLIND render-mode eval slice for the mining heads. Fresh "
            "(location, mode) pairs only (excluded against every prior render-mode batch by "
            "key AND by proximity, globbed, self-excluding), locations conditioned on HUMAN "
            "label quality alone, palettes drawn from the POOL against the declared rare "
            "hue-family target, coloured with the canonical emission recipe at the frozen "
            "corpus pins. Weighted toward the four contested per-mode cells; exp_smoothing "
            "excluded as measured smooth-equivalent. NO mining head score appears anywhere: "
            "no suggested_tier, no head block, no flat score, no score-ordered page. "
            "Presentation is a seeded shuffle stamped in sheet_order. EVAL-ONLY, PERMANENTLY.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {sel_rep['drawn_rows']} drawn rows are present — this batch is a "
            f"BOUNDED or INTERRUPTED run and must not be treated as the full sheet. Re-run "
            f"`render` then `write`.") if incomplete else None,
        "registration": {
            "source": reg.source, "biased": reg.biased,
            "eval_eligible": reg.eval_eligible,
            "split": BR.split_of(reg),
            "registered_before_build": True,
            "owner": "tools/scoring/batch_registry.py",
        },
        # --- the properties this sheet exists for -----------------------------------
        "blind": {
            # The row shape, DERIVED from the declared tuple rather than asserted in prose.
            # The names of the fields that would break it are deliberately NOT spelled here —
            # the source scan in test_blind_mining_sheet.py owns that list, and duplicating it
            # as string data would defeat the scan.
            "row_keys": list(ROW_KEYS),
            "machine_prelabel": None,
            "presentation": "seeded shuffle, stamped in sheet_order and served with "
                            "&order=file",
            "shuffle_seed": spec.shuffle_seed,
            "why": "an incumbent-vs-challenger eval slice must come from blind or "
                   "pre-incumbent labels; a slice served with the incumbent's suggestions "
                   "measures agreement with the incumbent, never quality "
                   "(classifier_retrain_protocol.md 2b). Every prior render-mode batch is a "
                   "correction sheet, and 0.929 of the mining sitting's labels came back "
                   "equal to the served suggestion.",
        },
        "eval_only": True,
        "eval_only_note":
            "PERMANENT. Every row is stamped split_side=eval and this slice must never enter "
            "any train split, for this head generation or any later one. It was bought to "
            "referee mining v1 against the five staged arms on unanchored labels; training on "
            "it spends the only unanchored read of this population that exists. It is also "
            "deliberately absent from near_dup_groups.BATCHES, so mining_corpus.load_corpus "
            "cannot pool it in by default.",
        "heads_read": {
            "mining": "NONE — no mining checkpoint is loaded, scored, stamped or sorted on by "
                      "this builder. A tokenizing source scan fails the build on the symbol.",
            "location": "NONE — the location head is not read either; quality is conditioned "
                        "on the HUMAN label corpus.",
            "palette": "NONE — the palette is a screened POOL draw (rare_palette_draw against "
                       "the declared rare-family target, palette_deficit.pick within family), "
                       "not a head proposal.",
        },
        "universe": uni,
        "selection_report": sel_rep,
        "render_bill": bill,
        "split": {"rule": "EVERY row is eval. This is not a stratified assignment and there "
                          "is nothing to re-derive: the batch is an instrument, not a "
                          "training set.",
                  "eval_rows": len(rows), "train_rows": 0},
        "seeds": {"draw_seed": spec.draw_seed, "shuffle_seed": spec.shuffle_seed,
                  "id_salt": spec.id_salt},
        "render_defaults": {
            "width": BMS.W, "height": BMS.H, "ss": BMS.SS, "filter": BMS.FILT,
            "jpg_quality": BMS.JPG_Q, "interior_mode": "black", "composition": "center",
            "why_these_pins": "the July render-mode batches' own pins, which every other "
                              "batch in this corpus carries — a corpus whose parts differ in "
                              "geometry cannot be unioned, and the heads being refereed were "
                              "trained on crops at these settings",
            "screen_geometry": list(SCREEN_GEOM),
        },
        "realized": {
            "rows_by_mode": dict(sorted(Counter(
                r["render"]["render_mode"] for r in rows).items())),
            "rows_by_kind": dict(sorted(Counter(
                r["provenance"]["mode_kind"] for r in rows).items())),
            "rows_by_bucket": dict(Counter(r["provenance"]["bucket"] for r in rows)),
            "rows_by_partition": dict(sorted(Counter(
                r["provenance"]["partition"] for r in rows).items())),
            "rows_by_hue_family": {f: sum(1 for r in rows
                                          if r["provenance"]["hue_family"] == f)
                                   for f in HF.FAMILIES},
            "rows_by_human_score": dict(sorted(Counter(
                r["provenance"]["source"]["human_score"] for r in rows).items())),
            "distinct_locations": len({r["provenance"]["location_key"] for r in rows}),
            "distinct_palettes": len({r["render"]["palette"] for r in rows}),
            "transfer_dropped_rows": sum(1 for r in rows
                                         if r["provenance"]["transfer_dropped"]),
            "cos_smooth": _cos_summary(rows),
            "contested_rows": sum(1 for r in rows
                                  if r["render"]["render_mode"] in CONTESTED_MODES),
        },
        "presentation": {
            "order": "sheet_order — a SEEDED SHUFFLE of the drawn set, contiguous",
            "sorted_on": "NOTHING. No score, machine or human, orders this sheet.",
            "contiguous": True,
            "image_id": "OPAQUE `<prefix><slot>_<hash8>` — slot is shuffled position, the "
                        "hash a salted digest of the unit key, so the id encodes no mode, "
                        "palette, family or band. route.json maps it back.",
        },
        "labels_export": spec.labels_sidecar,
        "labeling": {
            "ui": spec.ui_url,
            "mode": "BLIND — no suggestion is prefilled and no row carries a machine tier. "
                    "1/2/3 label and advance; there is no confirm key and no bulk accept "
                    "because there is nothing to accept.",
            "blind_rows": len(rows),
            "calibration_duplicates": 0,
            # THREE DIFFERENT FILES, named apart because two of them are easy to confuse and
            # the confusion lands at the END of a labeling sitting: the page downloads
            # `scores.json`, `--scores` READS whatever you save that as (beside the sidecar,
            # NOT under scratch/ — a label export is the one artifact in the pipeline with no
            # rebuild path), and the merge WRITES the sidecar. The re-verdict then reads the
            # sidecar, never the export.
            "export_download": "scores.json (the page's export button)",
            "save_export_as": spec.labels_export,
            "sidecar_written": spec.labels_sidecar,
            "merge": f"uv run python tools/wallpaper/merge_sitting.py "
                     f"--corpus render_mode_corpus --batch {spec.batch_id} "
                     f"--scores {spec.labels_export} --apply",
            "then": "uv run python tools/mining/sheet_e_reverdict.py   "
                    "# the one-command re-verdict, after labeling",
        },
        "render_failures": errors,
        "run_status": {
            "drawn_rows": sel_rep["drawn_rows"], "rendered_rows": len(rows),
            "n_failures": len(errors), "n_unaccounted": len(unaccounted),
            "unaccounted_rows": unaccounted[:50],
            "unaccounted_note": "drawn but neither rendered nor failed — a bounded (--limit) "
                                "or interrupted run; re-run `render` then `write`",
        },
    }
    (spec.batch_dir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print_summary(spec, batch)
    return batch


def _cos_summary(rows) -> dict:
    cs = [r["provenance"]["cos_smooth"] for r in rows
          if r["provenance"]["cos_smooth"] is not None]
    if not cs:
        return {"n": 0}
    return {**SE.quantiles(cs),
            "bands": dict(Counter(r["provenance"]["smooth_band"] for r in rows))}


# =========================================================================== #
# Reporting.
# =========================================================================== #
def print_universe(rep):
    p = rep["population"]["population"]
    x = rep["population"]["exclusion"]
    log("-" * 96)
    log(f"POPULATION  {p['n_locations']} human >=3 label-corpus locations "
        f"({p['totals']['score4']} fours / {p['totals']['score3']} threes)")
    log(f"FRESH       {x['n_fresh']} after exclusion  ({x['excluded_by_key']} by key, "
        f"{x['excluded_by_proximity']} by proximity, over {x['n_prior_location_keys']} prior "
        f"locations / {x['n_prior_location_mode_pairs']} prior pairs)")
    log(f"            prior batches {x['prior_batches']}")
    d = rep["location_draw"]
    log(f"LOCATIONS   {d['n_locations']} drawn (target {d['target']})  caps {d['location_caps']}")
    log(f"            by partition {d['drawn_by_partition']}")
    log(f"            by human score {d['drawn_by_score']}  ·  fallback-3 "
        f"{d['score3_by_partition']}")
    q = rep["palette_draw"]
    log(f"PALETTES    {q['distinct_palettes_used']} distinct, max repeats {q['max_repeats']}, "
        f"prefix dev {q['sequence_prefix_deviation']:.3f}")
    t = rep["mode_targets"]
    log(f"MODES       {t['active_modes']} active (excluded {t['excluded_modes']})")
    log(f"            targets {t['rows_by_mode']}")
    c = rep["candidate_draw"]
    log(f"CANDIDATES  {rep['n_candidates']} at oversample {c['oversample']} + "
        f"{rep['n_smooth_twins']} smooth twins (screen only, never served)")
    if c["short_of_quota"]:
        log(f"            SHORT of quota: {c['short_of_quota']}")
    pf = rep["pair_freshness"]
    log(f"FRESH PAIRS {pf['n_universe'] - pf['n_stale_pairs']}/{pf['n_universe']} "
        f"({pf['n_stale_pairs']} stale)")
    log("-" * 96)


def print_bill(bill):
    log("-" * 96)
    if bill.get("status"):
        log(f"RENDER BILL {bill['status']}")
        log("-" * 96)
        return
    log(f"RENDER BILL (basis {bill['basis']}, {bill['speedup_assumed']}x at 4 workers)")
    for name, s in bill["stages"].items():
        log(f"  {name:8} {s['single_process_min']:7.1f} min single-process -> "
            f"{s['wall_min_at_4x']:6.1f} min wall")
    log(f"  TOTAL    {bill['total_wall_min_at_4x']:.1f} min wall")
    ph = bill["phoenix"]
    log(f"  phoenix  {ph['share_of_keeper_units']*100:.1f}% of keeper units  "
        f"{ {k: v['units'] for k, v in ph['per_partition'].items()} }")
    log("-" * 96)


def print_composition(rep):
    log("-" * 96)
    s = rep["smooth_equivalence"]
    log(f"smooth-equivalence over the candidates: {s['bands_over_candidates']}   "
        f"excluded {s['excluded_smooth_equivalent']}   unmeasured {s['unmeasured_dropped']}")
    log(f"near-dup filter: dropped {rep['near_dup_filter']['n_dropped']} at cos "
        f">= {rep['near_dup_filter']['cut']}")
    log(f"drawn {rep['drawn_rows']} / target {rep['target_rows']}  ·  buckets {rep['buckets']}")
    log(f"by mode: {rep['drawn_by_mode']}")
    if rep["short_of_target"]:
        log(f"SHORT of the per-mode target (supply-bound): {rep['short_of_target']}")
    log(f"by kind: {rep['drawn_by_kind']}")
    log(f"by partition: {rep['drawn_by_partition']}")
    log(f"by hue family: {rep['drawn_by_hue_family']}")
    log(f"by human score: {rep['drawn_by_human_score']}  ·  locations "
        f"{rep['distinct_locations']}  ·  palettes {rep['distinct_palettes']}")
    log("-" * 96)


def print_summary(spec, batch):
    r = batch["realized"]
    log("\n" + "=" * 96)
    log(f"SHEET E — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 96)
    log(f"rows {batch['n_rows']} / drawn {batch['run_status']['drawn_rows']}  ·  failures "
        f"{batch['run_status']['n_failures']}")
    log(f"by mode:      {r['rows_by_mode']}")
    log(f"contested:    {r['contested_rows']} rows over {list(CONTESTED_MODES)}")
    log(f"by kind:      {r['rows_by_kind']}   buckets {r['rows_by_bucket']}")
    log(f"by partition: {r['rows_by_partition']}")
    log(f"human score:  {r['rows_by_human_score']}  ·  locations {r['distinct_locations']}  ·  "
        f"palettes {r['distinct_palettes']}")
    log(f"cos-to-smooth bands: {r['cos_smooth'].get('bands')}  median "
        f"{r['cos_smooth'].get('q', {}).get('p50')}")
    log(f"BLIND: {batch['blind']['presentation']}  ·  EVAL-ONLY: {batch['eval_only']}")
    log(f"-> {spec.batch_dir}")
    log(f"-> serve: uv run python tools/viz/serve.py   then")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Blind render-mode eval sheet builder (sheet E).")
    ap.add_argument("stage", choices=("pool", "estimate", "screen", "select", "render", "write"))
    ap.add_argument("--sheet", default="e", choices=sorted(SHEETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap units this run. A short batch STAMPS itself sheet_incomplete "
                         "at write time.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="render: composition only")
    ap.add_argument("--unit-timeout", type=float, default=900.0)
    ap.add_argument("--wall-budget-s", type=float, default=4 * 3600.0)
    args = ap.parse_args(argv)

    if args.workers > WORKERS:
        raise SystemExit(f"[blind-mining] --workers {args.workers} exceeds the project "
                         f"process cap of {WORKERS} (CLAUDE.md).")
    missing = MR.missing_recipes()
    if missing:
        raise SystemExit(f"[blind-mining] roster/recipe mismatch: {missing}")

    spec = SHEETS[args.sheet]
    if not BR.is_registered(spec.batch_id):
        raise SystemExit(f"[blind-mining] {spec.batch_id} is NOT in the batch registry. "
                         f"Register it BEFORE building — an unregistered batch classifies "
                         f"fail-closed and its split story is lost.")
    spec.work.mkdir(parents=True, exist_ok=True)
    prio = cc.set_below_normal_priority()
    log(f"[blind-mining] {spec.batch_id} · priority {prio} · {args.workers} workers x "
        f"{ENGINE_THREADS} rayon threads")

    if args.stage == "pool":
        _pool, rep = fresh_locations(spec)
        log(json.dumps(rep, indent=2))
        return 0
    if args.stage == "estimate":
        cands, twins, _lm, rep = universe(spec)
        print_universe(rep)
        bill = render_bill(spec, cands, twins)
        print_bill(bill)
        (spec.work / "universe.json").write_text(
            json.dumps({"universe": rep, "render_bill": bill}, indent=2), encoding="utf-8")
        log(f"-> {spec.work / 'universe.json'}")
        return 0
    if args.stage == "screen":
        run_screen(spec, args)
        return 0
    if args.stage == "select":
        _bk, _lm, _uni, _sel, rep, bill = _selected(spec, args)
        print_composition(rep)
        print_bill(bill)
        (spec.work / "selection_report.json").write_text(
            json.dumps({"selection": rep, "render_bill": bill}, indent=2), encoding="utf-8")
        return 0
    if args.stage == "render":
        run_render(spec, args)
        return 0
    if args.stage == "write":
        run_write(spec, args)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
