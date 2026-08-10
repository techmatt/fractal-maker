r"""build_rare_palette_sheet.py — SHEET C of the (27) sittings: strange renders of
HUMAN-GOOD locations, under RARE palettes, with the smooth-equivalent modes filtered out.

WHAT IS NEW HERE, AND IT IS THE STANDING RULE FROM NOW ON: **the location pool is human
label quality, not a head's gate.** Every location on this page carries a HUMAN label of 4
(3 only where a partition has no fours to give), read from the location label corpus through
`corpus_reader.iter_labeled` -> `label_store.resolve_score`. A strange-wallpaper sheet
conditions on location quality FIRST, so what Matt is left judging is the RENDERING and never
the location. Sheets A and B could not do that: their population was
`gate_passers_v3.json` — 112 locations, 38 palettes, frozen out of one July batch.

That frozen population is also why this sheet had to change population at all. Measured
2026-08-10 (appendix `sheetB_universe.json`): sheet B v1 served every one of the 15 modes at
every location of SIX of the eight partitions, so v2's "nothing served twice" rule left those
partitions with **0 remaining (location, mode) pairs** and v2 came out mandelbrot + julia
only. The universe was exhausted, not unlucky. 11,303 human-labeled locations — 634 fours —
is a population that does not run out.

THE FOUR DRAWS, each an imported rule rather than a local one:

  LOCATION   human score 4 (fallback 3 per partition), `apportion.deal_round_robin` over
             partitions with `phoenix:classic` PRESEEDED take-all (its whole supply is 7
             positives; a bucketed cut never reserves, so it needs its own floor), and
             phoenix capped for cost — phoenix fields run 6-20x, and the cap is in the
             render bill by name rather than the slice being dropped silently.
  PALETTE    `rare_palette_draw.PaletteDrawer` — the declared rare-family target realized
             through `apportion.sequence_by_deficit`, then `palette_deficit.pick` within the
             family. ONE palette per location, shared by that location's rows, so the smooth
             twin is rendered once per location and the two strange rows differ ONLY in mode.
  PARAMS     `deploy_tail._color_params({})` — the canonical inherited coloring
             (transfer=pct, gamma 1, no reverse/phase/cycles). This is what the LIVE emission
             path colours with, and it is the only recipe that applies to a palette nothing
             has ever fitted a head to.
  MODE       screened, then biased toward what the SMOOTH-EQUIVALENCE measure calls DISTINCT
             (`smooth_equivalence.py`). A mode render at cos >= 0.974 to its own location's
             smooth twin is the "duplicate" Matt named mid-labeling: it is EXCLUDED, not
             down-weighted. A small smooth-for-comparison slice is served deliberately.

THE SCREEN IS NOT THE STAMP (the pattern `build_fresh_sheet` established, reused): every
(location, mode) is rendered once at a SCORING-ONLY geometry through the SAME two render
paths the keeper uses (`build_mining_sheet.render_one` with a `geom`), scored by mining v1
AND colored-CLIP embedded, and only the selected rows are re-rendered at the frozen corpus
pins. The embedding is what pays for two guarantees at once — the smooth-equivalence
exclusion and the presentation near-dup filter — off ONE render.

NEAR-DUP FILTER AT BUILD. No two served rows are within cos 0.974 of each other on the
colored-CLIP substrate, INCLUDING two modes of one location. Greedy best-first by the mining
score, so the row that survives a collision is the better one.

    uv run python -u tools/mining/build_rare_palette_sheet.py pool
    uv run python -u tools/mining/build_rare_palette_sheet.py estimate
    uv run python -u tools/mining/build_rare_palette_sheet.py screen --limit 8   # smoke
    uv run python -u tools/mining/build_rare_palette_sheet.py screen > scratch/rare_palette/screen.log 2>&1
    uv run python -u tools/mining/build_rare_palette_sheet.py select
    uv run python -u tools/mining/build_rare_palette_sheet.py render > scratch/rare_palette/render.log 2>&1
    uv run python -u tools/mining/build_rare_palette_sheet.py write
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
import corpus_reader as cr                                  # noqa: E402  THE label-corpus reader
import location as loc_mod                                  # noqa: E402
import partitions as PART                                   # noqa: E402  THE partition resolver
from tools.mining import build_mining_sheet as BMS          # noqa: E402  THE render paths
from tools.mining import deploy_tail as DT                  # noqa: E402  THE canonical params
from tools.mining import mining_pins as MP                  # noqa: E402
from tools.mining import mining_roster as MR                # noqa: E402
from tools.mining import rare_palette_draw as RPD           # noqa: E402
from tools.mining import smooth_equivalence as SE           # noqa: E402
from tools.mining import suggest_tier_mining as ST          # noqa: E402
from tools.palettes import hue_families as HF               # noqa: E402
from tools.scoring import batch_registry as BR              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

CORPUS = ROOT / "data" / "render_mode_corpus"
LABEL_CORPUS = ROOT / "data" / "label_corpus"

# The SCORING-ONLY screen geometry — sheet B's, unchanged, so the two sheets' screens are
# the same instrument.
SCREEN_GEOM = (640, 360, 1)

WORKERS = 4
ENGINE_THREADS = BMS.ENGINE_THREADS


@dataclass(frozen=True)
class SheetSpec:
    key: str
    batch_id: str
    generator_version: str
    img_prefix: str
    id_salt: str
    target_rows: int
    n_locations: int
    strange_per_location: int
    smooth_slice: int
    mode_floor: int
    draw_seed: int
    # Per-partition LOCATION caps. `phoenix:classic` is take-all (its supply is 7); varied
    # phoenix is capped because a phoenix field costs 6-20x a mandelbrot one and an uncapped
    # 109-location supply would own the render bill.
    location_caps: dict = field(default_factory=dict)
    classic_partition: str = PART.CLASSIC_PHOENIX
    fallback_score: int = 3

    @property
    def batch_dir(self) -> Path:
        return CORPUS / "batches" / self.batch_id

    @property
    def work(self) -> Path:
        return ROOT / "scratch" / "rare_palette" / self.key

    @property
    def screen_log(self) -> Path:
        return self.work / "screen.jsonl"

    @property
    def embed_store(self) -> Path:
        return self.work / "screen_embeddings.npz"

    @property
    def labels_export(self) -> str:
        return f"labels/scores_{self.batch_id}.json"

    @property
    def ui_url(self) -> str:
        return (f"tools/viz/wallpaper_label.html?corpus=render_mode_corpus&tiers=3"
                f"&order=file&batch={self.batch_id}")


SHEETS = {
    "c1": SheetSpec(
        key="c1",
        batch_id="2026-08-10_render_mode_rare_palette_v1",
        generator_version="render_mode_rare_palette_v1",
        img_prefix="rp1",
        id_salt="render_mode_rare_palette_v1/2026-08-10",
        target_rows=500,
        n_locations=250,
        strange_per_location=2,
        smooth_slice=40,
        mode_floor=12,
        draw_seed=20260810,
        location_caps={"phoenix": 24, PART.CLASSIC_PHOENIX: 7},
    ),
}


def log(msg):
    print(msg, flush=True)


# =========================================================================== #
# 1. The human-good location pool.
# =========================================================================== #
def human_good_locations(min_score: int = 3) -> dict:
    """`{location_key: {loc, score, partition, batch_ids}}` — every label-corpus location
    whose HUMAN label is >= `min_score`, where a location's label is the MAX over its crops
    (the corpus contract).

    Reads `corpus_reader.iter_labeled` directly rather than
    `query_sampler.LocationPool.from_corpus`: that constructor runs
    `assert_sidecars_joined` over the SCORE-FILTERED census, so asking it for (3, 4) would
    raise on any registered sidecar batch that happens to hold no 3s or 4s — a guard about
    join integrity firing on a legitimate score filter. `iter_labeled` runs the same guard
    over the UNFILTERED pass, which is the one that means something."""
    best: dict = {}
    for lc in cr.iter_labeled():
        r = lc.render
        if not r or not r.get("cx"):
            continue
        loc = loc_mod.from_render_block(r)
        key = loc_mod.location_key(loc)
        cur = best.get(key)
        if cur is None:
            row = dict(r)
            row["fractal_type"] = loc.family        # absent fractal_type IS mandelbrot
            best[key] = {"loc": loc, "score": lc.score,
                         "partition": PART.partition_of_row(row),
                         "batch_ids": {lc.batch_id}, "image_ids": [lc.image_id]}
        else:
            cur["score"] = max(cur["score"], lc.score)
            cur["batch_ids"].add(lc.batch_id)
            cur["image_ids"].append(lc.image_id)
    return {k: v for k, v in best.items() if v["score"] >= min_score}


def pool_report(pool: dict) -> dict:
    by = defaultdict(Counter)
    for v in pool.values():
        by[v["partition"]][v["score"]] += 1
    return {
        "source": "data/label_corpus — corpus_reader.iter_labeled -> label_store.resolve_score "
                  "(amendments applied); location label = MAX over its crops",
        "n_locations": len(pool),
        "by_partition": {p: {"score4": c[4], "score3": c[3], "total": c[4] + c[3]}
                         for p, c in sorted(by.items())},
        "totals": {"score4": sum(c[4] for c in by.values()),
                   "score3": sum(c[3] for c in by.values())},
    }


# =========================================================================== #
# 2. The location draw.
# =========================================================================== #
def draw_locations(spec: SheetSpec, pool: dict):
    """`(drawn, report)` — `drawn` is an ordered list of `(location_key, entry)`.

    Fours before threes WITHIN a partition, seeded-shuffled inside each tier, so a partition
    only reaches its threes when its fours are exhausted (the prompt's "fall back to human 3
    where a cell is short")."""
    rng = np.random.default_rng([spec.draw_seed, 2])
    by_part: dict = defaultdict(lambda: {4: [], 3: []})
    for k, v in sorted(pool.items()):
        by_part[v["partition"]][v["score"]].append(k)

    ordered, sizes = {}, {}
    for p in sorted(by_part):
        fours, threes = by_part[p][4], by_part[p][3]
        rng.shuffle(fours)
        rng.shuffle(threes)
        seq = fours + threes
        cap = spec.location_caps.get(p)
        ordered[p] = seq
        sizes[p] = min(len(seq), cap) if cap is not None else len(seq)

    # `phoenix:classic` is TAKE-ALL, spent as a RESERVATION: its whole (capped) supply is
    # handed out first, `preseed` credits the round-robin with those rows so the cell is not
    # served twice, and its REMAINING supply drops to 0 so the balanced draw cannot add to
    # it. `n` drops by the reservation for the same reason. A cell absent from the pool
    # entirely is not reserved at all — `deal_round_robin` refuses a preseed key that is not
    # in `sizes`, which is the right refusal and must not be reached by construction.
    classic = spec.classic_partition
    reserved = {classic: sizes[classic]} if sizes.get(classic, 0) > 0 else {}
    rest_sizes = {p: sizes[p] - reserved.get(p, 0) for p in sizes}
    alloc = apportion.deal_round_robin(
        dict(sorted(rest_sizes.items())),
        max(0, spec.n_locations - sum(reserved.values())),
        preseed=reserved or None)
    take = {p: alloc.get(p, 0) + reserved.get(p, 0) for p in sizes}
    balanced, why = apportion.cells_balanced(
        {p: n for p, n in take.items() if p != classic},
        {p: n for p, n in sizes.items() if p != classic})

    drawn = []
    for p in sorted(take):
        for k in ordered[p][:take[p]]:
            drawn.append((k, pool[k]))
    rep = {
        "rule": "apportion.deal_round_robin over partitions (balanced-or-drained), with "
                f"{classic} PRESEEDED take-all — a bucketed cut never reserves, so the "
                "classic slice needs its own floor or it is absent by construction",
        "n_locations": len(drawn),
        "target": spec.n_locations,
        "location_caps": dict(spec.location_caps),
        "reserved_take_all": dict(reserved),
        "available_after_caps": dict(sorted(sizes.items())),
        "drawn_by_partition": dict(sorted(take.items())),
        "drawn_by_score": dict(Counter(v["score"] for _k, v in drawn)),
        "score4_by_partition": dict(sorted(Counter(
            v["partition"] for _k, v in drawn if v["score"] == 4).items())),
        "score3_by_partition": dict(sorted(Counter(
            v["partition"] for _k, v in drawn if v["score"] == 3).items())),
        "balanced_excl_classic": balanced, "balance": why,
        "fallback_note": "a partition reaches its human-3s only after its human-4s are "
                         "exhausted; score3_by_partition IS the shortfall, per partition",
    }
    return drawn, rep


# =========================================================================== #
# 3. The universe: (location, palette) x the 15 modes + the smooth twin.
# =========================================================================== #
def universe(spec: SheetSpec):
    """`(entries, report)` — deterministic and pure, recomputed by every stage."""
    pool = human_good_locations()
    drawn, draw_rep = draw_locations(spec, pool)
    drawer = RPD.PaletteDrawer(len(drawn), seed=spec.draw_seed)
    cparams = DT._color_params({})

    rng = np.random.default_rng([spec.draw_seed, 3])
    cell_perm = [MR.DIRECT_GRID[int(j)] for j in rng.permutation(len(MR.DIRECT_GRID))]

    entries, loc_meta = [], {}
    for i, (key, v) in enumerate(drawn):
        palette, family = drawer.take()
        loc = v["loc"]
        render = render_block_of(loc, palette)
        cp = dict(cparams)
        cp["palette"] = palette
        cp["palette_type"] = None            # filled at render time by the worker's library
        cp["palette_source"] = None
        cp["interior_color"] = [0.0, 0.0, 0.0]
        loc_meta[key] = {"palette": palette, "hue_family": family,
                         "partition": v["partition"], "human_score": v["score"],
                         "label_batches": sorted(v["batch_ids"]),
                         "family": loc.family}
        for mode in (MR.SMOOTH_MODE,) + MR.MODES:
            kind = MR.kind_of(mode)
            mode_params = {}
            if kind == "direct":
                op, th = cell_perm[i % len(cell_perm)]
                mode_params = {"direct_opacity": op, "direct_threshold": th}
            entries.append({
                "mode": mode, "kind": kind, "location_key": key,
                "family": loc.family, "partition": v["partition"],
                "human_score": v["score"], "hue_family": family,
                "palette": palette, "color_params": cp, "render": render,
                "mode_params": mode_params, "variant": 0,
            })
    entries.sort(key=lambda e: (e["location_key"], _mode_order(e["mode"])))
    for e in entries:
        e["unit_key"] = f"{e['location_key']}|{e['mode']}"
        e["image_id"] = _screen_stem(spec, e["unit_key"])

    rep = {
        "population": pool_report(pool),
        "location_draw": draw_rep,
        "palette_draw": drawer.report(),
        "color_params": {**cparams,
                         "owner": "tools/mining/deploy_tail._color_params({}) — the canonical "
                                  "inherited coloring the LIVE emission path uses"},
        "roster": {"n_modes": len(MR.MODES), "smooth_baseline": MR.SMOOTH_MODE,
                   "kinds": dict(Counter(MR.MODE_KIND[m] for m in MR.MODES))},
        "n_universe": len(entries),
        "universe_by_mode": dict(sorted(Counter(e["mode"] for e in entries).items())),
        "universe_by_partition": dict(sorted(Counter(e["partition"] for e in entries).items())),
        "direct_rule": "one DIRECT_GRID cell per (location, direct mode), rotating a permuted "
                       "cycle across locations — sheet v1's rule with the per-location sweep "
                       "collapsed, because this sheet's axis is the palette, not the grid",
    }
    return entries, loc_meta, rep


def spread_over(units, n: int, keys=("mode", "partition")) -> list:
    """`n` units dealt round-robin over every (mode, partition) cell present.

    A `linspace` over a location-major list is NOT a spread here: with 16 modes per location
    any stride that shares a factor with 16 walks the same few modes for the whole run, so a
    "bounded end-to-end" would exercise one render path and the first execution of the other
    two would be the production run (CLAUDE.md, "Give a long path a bounded end-to-end").
    Dealing over the cells is stride-independent."""
    cells: dict = defaultdict(list)
    for u in units:
        cells[tuple(u[k] for k in keys)].append(u)
    sizes = {c: len(v) for c, v in sorted(cells.items())}
    take = apportion.deal_round_robin(sizes, min(int(n), len(units)))
    out = []
    for c in sorted(cells):
        out.extend(cells[c][:take.get(c, 0)])
    return out


def _mode_order(mode: str) -> int:
    return -1 if mode == MR.SMOOTH_MODE else MR.MODES.index(mode)


def render_block_of(loc, palette: str) -> dict:
    blk = {"cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
           "fractal_type": loc.family, "c_re": loc.c_re, "c_im": loc.c_im,
           "palette": palette}
    for k, v in loc.params.items():
        blk[k] = v
    return blk


def _screen_stem(spec: SheetSpec, unit_key: str) -> str:
    return hashlib.blake2b(f"{spec.id_salt}|{unit_key}".encode(), digest_size=8).hexdigest()


# =========================================================================== #
# 4. Screen — mining score AND colored-CLIP embedding off ONE render.
# =========================================================================== #
def load_screen(spec: SheetSpec) -> dict:
    done = {}
    if spec.screen_log.exists():
        for line in spec.screen_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit_key"]] = rec
    return done


def load_embeddings(spec: SheetSpec) -> dict:
    if not spec.embed_store.exists():
        return {}
    z = np.load(spec.embed_store, allow_pickle=False)
    return {k: v for k, v in zip(z["keys"].tolist(), z["vecs"])}


def save_embeddings(spec: SheetSpec, emb: dict) -> None:
    """Whole-store rewrite through a `.tmp` sibling + rename. The store is small (a few MB)
    and rewritten once per screen batch, so the atomic-rename form applies (CLAUDE.md: an
    interruptible write must be atomic or resume-by-skip poisons the output)."""
    keys = sorted(emb)
    tmp = spec.embed_store.with_name(spec.embed_store.name + ".tmp")
    # Through an OPEN HANDLE, not a path: `np.savez` appends `.npz` to any path that does not
    # already end in it, so `…npz.tmp` would be written as `…npz.tmp.npz` and the rename would
    # fail on a file that does not exist — which is exactly the interrupted-write shape the
    # atomic rename is here to prevent.
    with open(tmp, "wb") as fh:
        np.savez(fh, keys=np.asarray(keys), vecs=np.stack([emb[k] for k in keys]))
    tmp.replace(spec.embed_store)


def run_screen(spec: SheetSpec, args):
    from tools.mining.mining_gate import MiningScorer

    entries, _lm, rep = universe(spec)
    print_universe(rep)
    done, emb = load_screen(spec), load_embeddings(spec)
    todo = [e for e in entries if e["unit_key"] not in done or e["unit_key"] not in emb]
    if args.limit:
        todo = spread_over(todo, args.limit)
        log(f"[screen] --limit {args.limit}: SPREAD over "
            f"{len(set(e['mode'] for e in todo))} modes / "
            f"{len(set(e['partition'] for e in todo))} partitions")
    log(f"[screen] universe {len(entries)} · done {len(done)} · todo {len(todo)} · "
        f"{args.workers} workers x {ENGINE_THREADS} threads · geom {SCREEN_GEOM}")
    if not todo:
        return

    crops = spec.work / "screen_crops"
    fields = spec.work / "screen_fields"
    crops.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    scorer = MiningScorer(model_path=MP.ACTIVE_MINING_CKPT)
    if scorer.k != ST.K_TIERS:
        raise SystemExit(f"[screen] head K={scorer.k} but the suggestion rule is written for "
                         f"K={ST.K_TIERS} — fix the rule, do not coerce.")
    embedder = SE.Embedder()
    log(f"[screen] mining head {MP.HEAD_VERSION} (K={scorer.k}) on {scorer.device} · "
        f"colored-CLIP {embedder.model_name} on {embedder.device}")

    timeout_s = max(90.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    t0, n, errs, pending = time.time(), 0, [], []

    def flush(batch):
        if not batch:
            return
        paths = [crops / f"{e['image_id']}.jpg" for e in batch]
        scores = scorer.score_paths(paths)
        vecs = embedder.embed_paths(paths)
        with spec.screen_log.open("a", encoding="utf-8") as fh:
            for e, s in zip(batch, scores):
                fh.write(json.dumps({
                    "unit_key": e["unit_key"], "mode": e["mode"], "kind": e["kind"],
                    "location_key": e["location_key"], "partition": e["partition"],
                    "family": e["family"], "palette": e["palette"],
                    "hue_family": e["hue_family"], "human_score": e["human_score"],
                    "mode_params": e["mode_params"],
                    "screen_pred": ST.expected_tier([s.p_ge2, s.p_ge3]),
                    "screen_p_ge2": s.p_ge2, "screen_p_ge3": s.p_ge3,
                    "screen_would_pass_gate": bool(s.passed),
                }) + "\n")
        for e, v in zip(batch, vecs):
            emb[e["unit_key"]] = np.asarray(v, dtype=np.float32)
        save_embeddings(spec, emb)
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
            if n % 200 == 0:
                el = time.time() - t0
                log(f"[screen] {n}/{len(todo)}  {len(errs)} failed  {n/el:.2f} row/s -> eta "
                    f"{(len(todo)-n)/(n/el)/60:.0f} min (elapsed {el/60:.0f} min)")
    flush(pending)
    (spec.work / "screen_errors.json").write_text(json.dumps(errs, indent=1), encoding="utf-8")
    log(f"[screen] done: {n} screened, {len(errs)} failed, {(time.time()-t0)/60:.1f} min")


# =========================================================================== #
# 5. Select.
# =========================================================================== #
def select(spec: SheetSpec, screen_recs, emb: dict, max_rows=None, seed=None):
    """`(selected, report)`. Five rules, applied in this order and each recorded:

      1. EXCLUDE smooth-equivalent strange rows — cos to their own location's smooth twin
         >= `SE.STRICT_CUT`. These are the duplicates the mid-labeling verdict named.
      2. Rank modes by measured DISTINCTNESS on this sheet's own screen (share below the
         interleave zone), not by sheet B's table: sheet B is two partitions wide.
      3. Per-location take `strange_per_location` rows, best-first by a score that is the
         mining pred tilted toward distinct modes.
      4. Mode floor — every mode reaches `mode_floor` rows if it has supply.
      5. Near-dup filter over the SELECTED set at `SE.STRICT_CUT`, greedy best-first.
    """
    max_rows = int(max_rows or spec.target_rows)
    rng = np.random.default_rng([int(seed if seed is not None else spec.draw_seed), 4])
    recs = {r["unit_key"]: dict(r) for r in screen_recs}

    # -- smooth twins -------------------------------------------------------- #
    smooth_by_loc = {r["location_key"]: k for k, r in recs.items()
                     if r["mode"] == MR.SMOOTH_MODE}
    n_no_twin = 0
    for k, r in recs.items():
        tw = smooth_by_loc.get(r["location_key"])
        if r["mode"] == MR.SMOOTH_MODE:
            r["cos_smooth"], r["band"] = 1.0, "self"
            continue
        if tw is None or tw not in emb or k not in emb:
            r["cos_smooth"], r["band"] = None, "unmeasured"
            n_no_twin += 1
            continue
        c = float(np.dot(emb[k], emb[tw]))
        r["cos_smooth"], r["band"] = c, SE.band_of(c)

    strange = [r for r in recs.values() if r["mode"] != MR.SMOOTH_MODE]
    # (1) exclusion. `unmeasured` rows are dropped too: "we could not measure this" must not
    # read as "this is distinct" (the wrap()-does-not-swallow rule, applied at the draw).
    keep = [r for r in strange if r["band"] in ("distinct", "interleave")]
    excluded = [r for r in strange if r["band"] not in ("distinct", "interleave")]

    # (2) per-mode distinctness, measured on THIS sheet.
    dist = {}
    for m in MR.MODES:
        rows = [r for r in strange if r["mode"] == m and r["cos_smooth"] is not None]
        dist[m] = {"n": len(rows),
                   "share_distinct": (sum(1 for r in rows if r["band"] == "distinct")
                                      / len(rows)) if rows else 0.0,
                   "median_cos": float(np.median([r["cos_smooth"] for r in rows]))
                   if rows else None}

    # (3) per-location take, best-first on pred tilted by mode distinctness. The tilt is a
    # RANK nudge inside a location, never a cut: the mining score still orders the page.
    for r in keep:
        r["sel_score"] = float(r["screen_pred"]) + 0.5 * dist[r["mode"]]["share_distinct"] \
            + (0.15 if r["band"] == "distinct" else 0.0)
    by_loc = defaultdict(list)
    for r in keep:
        by_loc[r["location_key"]].append(r)
    picked = []
    for k in sorted(by_loc):
        rows = sorted(by_loc[k], key=lambda r: (-r["sel_score"], r["unit_key"]))
        seen_modes = set()
        for r in rows:
            if len(seen_modes) >= spec.strange_per_location:
                break
            if r["mode"] in seen_modes:
                continue
            seen_modes.add(r["mode"])
            r["bucket"] = "per_location"
            picked.append(r)

    # (4) mode floor over what is still unpicked.
    chosen = {r["unit_key"] for r in picked}
    have = Counter(r["mode"] for r in picked)
    floor_added = Counter()
    for m in MR.MODES:
        short = spec.mode_floor - have.get(m, 0)
        if short <= 0:
            continue
        pool = sorted((r for r in keep
                       if r["mode"] == m and r["unit_key"] not in chosen),
                      key=lambda r: (-r["sel_score"], r["unit_key"]))
        for r in pool[:short]:
            r["bucket"] = "mode_floor"
            chosen.add(r["unit_key"])
            picked.append(r)
            floor_added[m] += 1

    # (5) near-dup filter over the selected set, greedy best-first.
    picked.sort(key=lambda r: (-r["sel_score"], r["unit_key"]))
    kept, kept_vecs, dropped = [], [], []
    for r in picked:
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

    # the smooth comparison slice — a seeded sample of locations that kept a strange row, so
    # every smooth row on the page has a strange sibling to be compared against.
    strange_locs = sorted({r["location_key"] for r in kept})
    n_smooth = min(spec.smooth_slice, max(0, max_rows - len(kept)), len(strange_locs))
    pick_locs = set(rng.permutation(len(strange_locs))[:n_smooth].tolist())
    smooth_rows = []
    for i, k in enumerate(strange_locs):
        if i in pick_locs and smooth_by_loc.get(k) in recs:
            r = dict(recs[smooth_by_loc[k]])
            r["bucket"] = "smooth_comparison"
            r["sel_score"] = float(r["screen_pred"])
            smooth_rows.append(r)

    selected = kept[:max(0, max_rows - len(smooth_rows))] + smooth_rows
    selected.sort(key=lambda r: r["unit_key"])

    rep = {
        "max_rows": max_rows, "drawn_rows": len(selected),
        "screened": len(recs),
        "smooth_equivalence": {
            **SE.yardstick_block(),
            "measured_at_geometry": list(SCREEN_GEOM),
            "excluded_smooth_equivalent": len(excluded),
            "unmeasured_dropped": n_no_twin,
            "bands_over_universe": dict(Counter(r["band"] for r in strange)),
            "by_mode": dist,
        },
        "near_dup_filter": {
            "cut": SE.STRICT_CUT, "substrate": "colored_clip",
            "n_dropped": len(dropped),
            "dropped": dropped[:60],
            "rule": "greedy best-first over the selected set — no two served rows within "
                    "cos 0.974, INCLUDING two modes of one location",
        },
        "buckets": dict(Counter(r["bucket"] for r in selected)),
        "mode_floor": {"floor": spec.mode_floor, "added": dict(sorted(floor_added.items()))},
        "drawn_by_mode": dict(sorted(Counter(r["mode"] for r in selected).items())),
        "drawn_by_kind": dict(sorted(Counter(r["kind"] for r in selected).items())),
        "drawn_by_partition": dict(sorted(Counter(r["partition"] for r in selected).items())),
        "drawn_by_hue_family": {f: sum(1 for r in selected if r["hue_family"] == f)
                                for f in HF.FAMILIES},
        "drawn_by_human_score": dict(sorted(Counter(r["human_score"] for r in selected).items())),
        "modes_below_floor": {m: c for m, c in sorted(
            Counter(r["mode"] for r in selected).items()) if c < spec.mode_floor},
        "distinct_locations": len({r["location_key"] for r in selected}),
        "distinct_palettes": len({r["palette"] for r in selected}),
        "seed": int(seed if seed is not None else spec.draw_seed),
    }
    return selected, rep


def _thin(r) -> dict:
    return {k: r[k] for k in ("unit_key", "mode", "partition", "palette", "cos_smooth")
            if k in r}


# =========================================================================== #
# 6. Render + write.
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


def _selected(spec, args):
    entries, loc_meta, uni = universe(spec)
    by_key = {e["unit_key"]: e for e in entries}
    screen = load_screen(spec)
    if not screen:
        raise SystemExit("[select] no screen records — run `screen` first")
    emb = load_embeddings(spec)
    sel, rep = select(spec, list(screen.values()), emb, args.max_rows, args.seed)
    return by_key, loc_meta, uni, sel, rep


def run_render(spec: SheetSpec, args):
    by_key, _lm, _uni, selected, rep = _selected(spec, args)
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
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
        log(f"[render] --limit {args.limit}: SPREAD -> modes {sorted({r['mode'] for r in todo})}")
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


def provenance_block(spec, entry, rec, transfer_dropped) -> dict:
    p = entry["color_params"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "mining_rare_palette_human_good_locations",
        "family": entry["family"],
        "partition": entry["partition"],
        "location_key": entry["location_key"],
        "render_mode": entry["mode"],
        "mode_kind": entry["kind"],
        "mode_params": dict(entry.get("mode_params", {})),
        "rolloff": MR.rolloff_token(entry["mode"]),
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
        "bucket": rec["bucket"],
        "screen_pred": rec["screen_pred"],
        "screen_p_ge3": rec["screen_p_ge3"],
        "cos_smooth": rec.get("cos_smooth"),
        "smooth_band": rec.get("band"),
        "screen_path": f"the keeper render paths at a SCORING-ONLY geometry "
                       f"{SCREEN_GEOM[0]}x{SCREEN_GEOM[1]}ss{SCREEN_GEOM[2]} "
                       f"(build_mining_sheet.render_one with `geom`)",
        "split_side": "train",
        "split_origin": "batch_registry — the whole batch is biased (human-4 locations, "
                        "rare-palette draw, distinctness-biased modes, head-prefilled page), "
                        "so every row is train-side and no location of it is an instrument",
        "source": {
            "corpus": "data/label_corpus",
            "human_score": entry["human_score"],
            "label_batches": rec.get("label_batches"),
            "rule": "location label = MAX over its crops, resolved through "
                    "label_store.resolve_score with amendments applied",
        },
    }


def run_write(spec: SheetSpec, args):
    from tools.mining.mining_gate import MiningScorer

    by_key, loc_meta, uni, selected, sel_rep = _selected(spec, args)
    done = load_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")
    live = [r for r in selected if r["unit_key"] in done]

    crops = spec.batch_dir / "crops"
    scorer = MiningScorer(model_path=MP.ACTIVE_MINING_CKPT)
    log(f"[write] mining head {MP.HEAD_VERSION} (K={scorer.k}) on {scorer.device} · "
        f"scoring {len(live)} crops")
    if scorer.k != ST.K_TIERS:
        raise SystemExit(f"[write] head K={scorer.k} but the suggestion rule is written for "
                         f"K={ST.K_TIERS} — fix the rule, do not coerce.")
    scores = scorer.score_paths([crops / f"{by_key[r['unit_key']]['image_id']}.jpg"
                                 for r in live])
    pred = [ST.expected_tier([s.p_ge2, s.p_ge3]) for s in scores]
    tiers = ST.suggest_all(pred, ST.CUTS)

    rows = []
    for j, rec in enumerate(live):
        e = by_key[rec["unit_key"]]
        loc = loc_mod.from_render_block(e["render"])
        s = scores[j]
        r2 = dict(rec)
        r2["label_batches"] = loc_meta[e["location_key"]]["label_batches"]
        rows.append({
            "_unit_key": rec["unit_key"],
            "_crop_stem": e["image_id"],
            "render": BMS.render_block(e, loc),
            "provenance": provenance_block(
                spec, e, r2, bool(done[rec["unit_key"]]["transfer_dropped"])),
            "label": {"score": None, "labeler": None, "labeled_at": None},
            "head_mining_v1": {
                "pred": pred[j], "p_ge2": s.p_ge2, "p_ge3": s.p_ge3, "score": s.score,
                "ckpt": MP.ACTIVE_MINING_CKPT, "head_version": MP.HEAD_VERSION,
                "gate_version": MP.MINING_GATE_VERSION,
                "would_pass_gate": bool(s.passed), "gate_threshold": scorer.threshold,
            },
            "p_ge3": s.p_ge3,
            "pred": pred[j],
            "suggested_tier": int(tiers[j]),
        })

    rows.sort(key=lambda r: (-r["pred"], r["_crop_stem"]))
    for i, r in enumerate(rows):
        r["sheet_order"] = i
        r["image_id"] = f"{spec.img_prefix}{i:04d}_{r['_crop_stem'][:8]}"
    assert len({r["image_id"] for r in rows}) == len(rows), "opaque ids collided"
    for r in rows:
        dst = crops / f"{r['image_id']}.jpg"
        if not dst.exists():
            shutil.copyfile(crops / f"{r['_crop_stem']}.jpg", dst)
    route = {r["image_id"]: {"unit_key": r.pop("_unit_key"), "crop_stem": r.pop("_crop_stem")}
             for r in rows}

    sp = np.array([r["provenance"]["screen_p_ge3"] for r in rows])
    fp = np.array([r["p_ge3"] for r in rows])
    from tools.wallpaper import build_fresh_sheet as FS
    agree = {"n": len(rows), "spearman": FS._spearman(sp, fp),
             "mean_abs_delta": float(np.abs(sp - fp).mean()) if len(rows) else None,
             "note": f"screen = the same render path at {SCREEN_GEOM}, SCORING-ONLY; keeper "
                     f"= the stored {BMS.W}x{BMS.H} ss{BMS.SS} crop. head_mining_v1 IS the "
                     f"keeper score."}

    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
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
            "RARE-PALETTE STRANGE SHEET (sheet C of the (27) sittings). Its locations are "
            "conditioned on HUMAN label quality FIRST (score 4, falling back to 3 only where "
            "a partition is short), so what is being judged is the RENDERING and never the "
            "location — the new standing rule for strange-wallpaper sheets. Palettes are "
            "drawn against a declared RARE hue-family target (green/spectral/rose/gold/"
            "neutral over-drawn; purple/fire/ice cut), the coloring is the canonical "
            "emission recipe, and every mode row is measured against a SMOOTH render of its "
            "own location on the colored-CLIP substrate: smooth-equivalent rows are excluded "
            "and no two served rows are within cos 0.974 of each other. label.score is null "
            "on every row; the suggestion is NOT a label. Presentation is sorted good->bad "
            "by pred with opaque image_ids.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {sel_rep['drawn_rows']} drawn rows are present — a BOUNDED or "
            f"INTERRUPTED run; re-run `render` then `write`.") if incomplete else None,
        "registration": {
            "source": reg.source, "biased": reg.biased,
            "eval_eligible": reg.eval_eligible,
            "split": BR.split_of(reg),
            "registered_before_build": True,
            "owner": "tools/scoring/batch_registry.py",
        },
        "head": {
            "ckpt": MP.ACTIVE_MINING_CKPT, "version": MP.HEAD_VERSION,
            "gate_version": MP.MINING_GATE_VERSION,
            "role": "PRE-LABEL + SELECTION SCREEN — no gate, floor, threshold or pin is "
                    "applied or moved here; would_pass_gate is stamped and gates nothing",
            "out_of_distribution_note":
                "mining v1 was trained on crops built from `gate_passers_v3.json` — 112 "
                "locations, 38 palettes, one July batch. This sheet's palettes are drawn "
                "from the whole 987 pool and its locations from the human label corpus, so "
                "the pre-label is an EXTRAPOLATION and its correction rate is not comparable "
                "to sheet B's.",
            "scorer": "tools/mining/mining_gate.MiningScorer (fp32, no autocast; marginal "
                      "p_ge = cumprod(sigmoid), NEVER the CORN conditional)",
            "deploy_transform": "classifier.data.Transform(train=False) — 384x224 bicubic "
                                "stretch + the checkpoint's own mean/std",
        },
        "suggested_tier_rule": ST.fit_derivation(ST.CUTS, pred, MP.ACTIVE_MINING_CKPT,
                                                 MP.HEAD_VERSION),
        "universe": uni,
        "selection_report": sel_rep,
        "screen_vs_keeper": agree,
        "split": {"rule": "the whole batch is train-side (see registration)",
                  "train_rows": len(rows), "eval_rows": 0},
        "seeds": {"draw_seed": sel_rep["seed"], "id_salt": spec.id_salt},
        "render_defaults": {
            "width": BMS.W, "height": BMS.H, "ss": BMS.SS, "filter": BMS.FILT,
            "jpg_quality": BMS.JPG_Q, "interior_mode": "black", "composition": "center",
            "why_these_pins": "sheet v1/v2's pins, which are the July render-mode batches' "
                              "own — the mining head was trained on crops at these settings, "
                              "and a corpus whose parts differ in geometry cannot be unioned",
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
            "suggested_tier_hist": dict(sorted(Counter(
                r["suggested_tier"] for r in rows).items())),
            "would_pass_mining_gate": sum(1 for r in rows
                                          if r["head_mining_v1"]["would_pass_gate"]),
            "transfer_dropped_rows": sum(1 for r in rows
                                         if r["provenance"]["transfer_dropped"]),
            "cos_smooth": _cos_summary(rows),
            "pred": {"min": round(min(r["pred"] for r in rows), 4),
                     "max": round(max(r["pred"] for r in rows), 4)},
        },
        "presentation": {
            "order": "sheet_order — DESCENDING pred (good -> bad), ties on the crop stem",
            "sorted_on": "the continuous head score (pred), NOT the suggested tier",
            "contiguous": True,
            "image_id": "OPAQUE `<prefix><slot>_<hash8>` — slot is presentation position "
                        "(published anyway: the sheet is sorted), hash is a salted digest of "
                        "the unit key, so the id encodes no mode, palette, family or band. "
                        "route.json maps it back.",
        },
        "labels_export": spec.labels_export,
        "labeling": {
            "ui": spec.ui_url,
            "mode": "correction — every row shows its suggested tier PREFILLED; Enter "
                    "confirms, 1-3 override. Only rows Matt acts on are exported; an "
                    "unreviewed suggestion never leaves the page as a label.",
            "bulk": "accept all remaining, and accept all BELOW THIS ROW — both behind a "
                    "confirm",
            "blind_rows": 0,
            "calibration_duplicates": 0,
            "merge": f"uv run python tools/wallpaper/merge_sitting.py "
                     f"--corpus render_mode_corpus --batch {spec.batch_id} "
                     f"--scores {spec.labels_export} --apply",
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
          if r["provenance"]["cos_smooth"] is not None
          and r["provenance"]["render_mode"] != MR.SMOOTH_MODE]
    if not cs:
        return {"n": 0}
    return {**SE.quantiles(cs),
            "bands": dict(Counter(r["provenance"]["smooth_band"] for r in rows
                                  if r["provenance"]["render_mode"] != MR.SMOOTH_MODE))}


# =========================================================================== #
# Reporting.
# =========================================================================== #
def print_universe(rep):
    p = rep["population"]
    log("-" * 96)
    log(f"POPULATION  {p['n_locations']} human >=3 locations "
        f"({p['totals']['score4']} fours / {p['totals']['score3']} threes)")
    d = rep["location_draw"]
    log(f"LOCATIONS   {d['n_locations']} drawn (target {d['target']})  caps {d['location_caps']}")
    log(f"            by partition {d['drawn_by_partition']}")
    log(f"            by human score {d['drawn_by_score']}  ·  fallback-3 "
        f"{d['score3_by_partition']}")
    q = rep["palette_draw"]
    log(f"PALETTES    {q['distinct_palettes_used']} distinct, max repeats {q['max_repeats']}, "
        f"prefix dev {q['sequence_prefix_deviation']:.3f}")
    log(f"            {q['drawn_counts']}")
    log(f"UNIVERSE    {rep['n_universe']} candidates "
        f"({len(MR.MODES)} modes + the smooth twin per location)")
    log("-" * 96)


def print_composition(rep):
    log("-" * 96)
    s = rep["smooth_equivalence"]
    log(f"smooth-equivalence over the universe: {s['bands_over_universe']}   "
        f"excluded {s['excluded_smooth_equivalent']}   unmeasured {s['unmeasured_dropped']}")
    log(f"near-dup filter: dropped {rep['near_dup_filter']['n_dropped']} at cos "
        f">= {rep['near_dup_filter']['cut']}")
    log(f"drawn {rep['drawn_rows']} / cap {rep['max_rows']}  ·  buckets {rep['buckets']}")
    log(f"by kind: {rep['drawn_by_kind']}")
    log(f"by partition: {rep['drawn_by_partition']}")
    log(f"by hue family: {rep['drawn_by_hue_family']}")
    log(f"by human score: {rep['drawn_by_human_score']}  ·  locations "
        f"{rep['distinct_locations']}  ·  palettes {rep['distinct_palettes']}")
    log(f"by mode: {rep['drawn_by_mode']}")
    if rep["modes_below_floor"]:
        log(f"BELOW the {rep['mode_floor']['floor']}-row mode floor (supply-bound): "
            f"{rep['modes_below_floor']}")
    log("-" * 96)


def print_summary(spec, batch):
    r = batch["realized"]
    log("\n" + "=" * 96)
    log(f"RARE-PALETTE STRANGE SHEET — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 96)
    log(f"rows {batch['n_rows']} / drawn {batch['run_status']['drawn_rows']}  ·  failures "
        f"{batch['run_status']['n_failures']}")
    log(f"by mode:      {r['rows_by_mode']}")
    log(f"by kind:      {r['rows_by_kind']}   buckets {r['rows_by_bucket']}")
    log(f"by partition: {r['rows_by_partition']}")
    log(f"by hue fam:   {r['rows_by_hue_family']}")
    log(f"human score:  {r['rows_by_human_score']}  ·  locations {r['distinct_locations']}  ·  "
        f"palettes {r['distinct_palettes']}")
    log(f"suggested:    {r['suggested_tier_hist']}  (cuts {batch['suggested_tier_rule']['cuts']})")
    log(f"cos-to-smooth bands: {r['cos_smooth'].get('bands')}  median "
        f"{r['cos_smooth'].get('q', {}).get('p50')}")
    log(f"-> {spec.batch_dir}")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Rare-palette strange sheet builder (sheet C).")
    ap.add_argument("stage", choices=("pool", "estimate", "screen", "select", "render", "write"))
    ap.add_argument("--sheet", default="c1", choices=sorted(SHEETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unit-timeout", type=float, default=900.0)
    ap.add_argument("--wall-budget-s", type=float, default=4 * 3600.0)
    args = ap.parse_args(argv)

    if args.workers > WORKERS:
        raise SystemExit(f"[rare-palette] --workers {args.workers} exceeds the project "
                         f"process cap of {WORKERS} (CLAUDE.md).")
    missing = MR.missing_recipes()
    if missing:
        raise SystemExit(f"[rare-palette] roster/recipe mismatch: {missing}")

    spec = SHEETS[args.sheet]
    if not BR.is_registered(spec.batch_id):
        raise SystemExit(f"[rare-palette] {spec.batch_id} is NOT in the batch registry. "
                         f"Register it BEFORE building — an unregistered batch classifies "
                         f"fail-closed and its split story is lost.")
    spec.work.mkdir(parents=True, exist_ok=True)
    prio = cc.set_below_normal_priority()
    log(f"[rare-palette] {spec.batch_id} · priority {prio} · {args.workers} workers x "
        f"{ENGINE_THREADS} rayon threads")

    if args.stage == "pool":
        rep = pool_report(human_good_locations())
        log(json.dumps(rep, indent=2))
        return 0
    if args.stage == "estimate":
        _e, _lm, rep = universe(spec)
        print_universe(rep)
        (spec.work / "universe.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return 0
    if args.stage == "screen":
        run_screen(spec, args)
        return 0
    if args.stage == "select":
        _bk, _lm, _uni, _sel, rep = _selected(spec, args)
        print_composition(rep)
        (spec.work / "selection_report.json").write_text(json.dumps(rep, indent=2),
                                                         encoding="utf-8")
        return 0
    if args.stage == "render":
        run_render(spec, args)
        return 0
    if args.stage == "write":
        run_write(spec, args)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
