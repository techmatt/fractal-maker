r"""build_mining_correction.py — the MODE x SCORE mining correction sheet.

The second of the (27) sittings (prompts/sittings_27.md) and the motivating slice for the
from-scratch mining retrain. Run 25's era gate found ONE aimed problem in the whole emission:
**busy strange renders from the fancy modes** — good locations, over-busy presentation, scored
high by mining v1. Busyness is not machine-measurable and nothing in the pipeline measures
composition, so the only instrument is Matt, and the only way to point him at the failure is
to over-draw the cell it lives in.

THE CELL IS (mode kind) x (mining score), AND THE OVERSAMPLE IS AT ITS TOP-RIGHT.
Measured on the 960 labeled rows of `2026-08-06_render_mode_fresh_sheet_v1`: of the 148 rows
v1 put in its top tier, 15 carry a human 2 — and 12 of those 15 are composite/direct modes
(`composite_c7` 2, `composite_c13` 1, `direct_trap_*` 6, `smooth_angle_min` 2, `trap_circle`
1). That is the population. It is thin BY CONSTRUCTION — a false positive at the top of a
scale is rare — so a proportional draw would put ~15 of them in front of Matt again.

WHAT THIS SHEET DOES INSTEAD. It takes EVERY high-scoring row it can find, fancy modes first,
then spends the rest of the page on the mid band with a declared 60% share to the fancy modes.
"Size to the passer supply" is literal: the sheet is as big as the high band plus what the
budget can carry, capped at MAX_ROWS, and a bucket that cannot fill records its shortfall
rather than borrowing.

THE SCREEN IS NOT THE STAMP (the pattern `build_fresh_sheet` established, applied to strange
renders for the first time). Stratifying by score needs a mining score for every candidate
BEFORE any keeper crop exists, and the unserved universe is 3,486 combos against a <=1000-row
sheet. So every candidate is rendered ONCE at a SCORING-ONLY geometry through the SAME two
render paths the keeper uses (`build_mining_sheet._render_pure` / `_render_rust`, called with
a `geom` — not a second copy), scored, and only the selected rows are re-rendered at the
frozen corpus pins. The stamped `head_mining_v1` is always the keeper score; both are kept per
row and the run prints their rank agreement.

THE UNIVERSE is the same one sheet v1 drew from — `gate_passers_v3.json` (401 (location,
palette) rows over 112 locations, the wallpaper head's own gate passers) x the 15-mode roster
— MINUS every (location, mode) pair v1 already served. Nothing Matt has judged is served
twice. `direct_*` is palette-INDIFFERENT by construction, so it spreads the 9-cell
opacity x threshold grid on one deterministic palette per location instead of spreading
palettes; that is sheet v1's rule, unchanged.

  uv run python -u tools/mining/build_mining_correction.py estimate
  uv run python -u tools/mining/build_mining_correction.py screen --limit 8      # smoke
  uv run python -u tools/mining/build_mining_correction.py screen > scratch/mining_correction/screen.log 2>&1
  uv run python -u tools/mining/build_mining_correction.py select                # dry composition
  uv run python -u tools/mining/build_mining_correction.py render > scratch/mining_correction/render.log 2>&1
  uv run python -u tools/mining/build_mining_correction.py write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                                        # noqa: E402  THE two draw rules
import corpus_common as cc                              # noqa: E402  engine launch defaults
import location as loc_mod                              # noqa: E402
from tools.mining import build_mining_sheet as BMS      # noqa: E402  THE render paths
from tools.mining import mining_pins as MP              # noqa: E402
from tools.mining import mining_roster as MR            # noqa: E402
from tools.mining import suggest_tier_mining as ST      # noqa: E402
from tools.mining.split_units import build_split, units_are_disjoint   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                        # noqa: BLE001
    pass

CORPUS = ROOT / "data" / "render_mode_corpus"
PRIOR_SHEET = "2026-08-06_render_mode_fresh_sheet_v1"

# The SCORING-ONLY screen geometry. The mining head sees a 1280x720 crop stretched to 384x224,
# so a 640x360 ss1 render reaching the same transform is a proxy of the same shape as the
# wallpaper coarse screen — 8x fewer samples than the keeper pins.
SCREEN_GEOM = (640, 360, 1)

MAX_ROWS = 1000                 # the standing sitting cap
MODE_FLOOR = 20                 # every mode with supply is represented by at least this many
FANCY_MID_SHARE = 0.60          # of the post-floor remainder, to composite + direct
FANCY_KINDS = frozenset({"composite", "direct"})

WORKERS = 4                     # the project process cap — do NOT raise
ENGINE_THREADS = BMS.ENGINE_THREADS

BUCKET_ORDER = ("hi_fancy", "hi_pure", "mode_floor", "mid_fancy", "fill")


@dataclass(frozen=True)
class SheetSpec:
    key: str
    batch_id: str
    generator_version: str
    img_prefix: str
    id_salt: str
    max_rows: int
    draw_seed: int
    split_seed: int
    eval_frac: float
    mode_floor: int = MODE_FLOOR
    fancy_mid_share: float = FANCY_MID_SHARE

    @property
    def batch_dir(self) -> Path:
        return CORPUS / "batches" / self.batch_id

    @property
    def work(self) -> Path:
        return ROOT / "scratch" / "mining_correction" / self.key

    @property
    def screen_log(self) -> Path:
        return self.work / "screen.jsonl"

    @property
    def labels_export(self) -> str:
        return f"labels/scores_{self.batch_id}.json"

    @property
    def ui_url(self) -> str:
        return (f"tools/viz/wallpaper_label.html?corpus=render_mode_corpus&tiers=3"
                f"&order=file&batch={self.batch_id}")


SHEETS = {
    "v2": SheetSpec(
        key="v2",
        batch_id="2026-08-10_render_mode_correction_v2",
        generator_version="render_mode_correction_v2",
        img_prefix="mc2",
        id_salt="render_mode_correction_v2/2026-08-10",
        max_rows=MAX_ROWS,
        draw_seed=20260810,
        split_seed=0,            # the July split seed — the design is reused, not re-rolled
        eval_frac=0.40,
    ),
}


def log(msg):
    print(msg, flush=True)


# =========================================================================== #
# 1. Universe — gate passers x roster, minus what sheet v1 already served.
# =========================================================================== #
def served_pairs(batch_id: str = PRIOR_SHEET) -> set:
    """`{(location_key, mode)}` sheet v1 put in front of Matt. A hard failure when the batch
    is absent: an empty exclusion set would silently re-serve 960 judged rows."""
    p = CORPUS / "batches" / batch_id / "images.jsonl"
    if not p.exists():
        raise SystemExit(
            f"[mining-correction] prior sheet absent: {p}\nWithout it the 'nothing is served "
            f"twice' guarantee is vacuous, not merely unchecked.")
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pr = json.loads(line)["provenance"]
            out.add((pr["location_key"], pr["render_mode"]))
    return out


def universe(spec: SheetSpec):
    """`(entries, report)` — every unserved (location, palette-or-cell, mode) candidate.

    Deterministic and pure: the same artifact + spec always gives the same universe, so
    `screen`, `select`, `render` and `write` each recompute it instead of reading a scratch
    file a `rm -r scratch/*` could take out from under a half-finished run."""
    rows, pop_meta = BMS.load_gate_passers()
    sv = served_pairs()
    loc_rows, loc_fam, _fam_locs, loc_rep = BMS.index_population(rows)

    entries = []
    for mode in MR.MODES:
        kind = MR.MODE_KIND[mode]
        for k in sorted(loc_rows):
            if (k, mode) in sv:
                continue
            members = sorted((rows[i] for i in loc_rows[k]), key=lambda r: r["image_id"])
            if kind == "direct":
                # palette-INDIFFERENT: spread the opacity x threshold grid on ONE palette.
                src = members[0]
                for ci, (op, th) in enumerate(MR.DIRECT_GRID):
                    entries.append(_entry(mode, kind, k, loc_fam[k], src,
                                          {"direct_opacity": op, "direct_threshold": th}, ci))
            else:
                for pi, src in enumerate(members):
                    entries.append(_entry(mode, kind, k, loc_fam[k], src, {}, pi))
    entries.sort(key=lambda e: (MR.MODES.index(e["mode"]), e["location_key"], e["variant"]))
    for e in entries:
        e["unit_key"] = f"{e['mode']}|{e['location_key']}|{e['variant']}"
        e["image_id"] = _screen_stem(spec, e["unit_key"])   # the WORKING name; served id is opaque
    report = {
        "population": {
            "artifact": "data/render_mode_corpus/gate_passers_v3.json",
            "rows": len(rows), "locations": len(loc_rows),
            "head": pop_meta["head"], "gate": pop_meta["gate"],
            "source_batch": pop_meta["source_batch"],
        },
        "exclusion": {"prior_sheet": PRIOR_SHEET, "served_location_mode_pairs": len(sv),
                      "rule": "a (location, mode) pair Matt has already judged is never "
                              "served again, whatever palette it would carry"},
        "roster": {"n_modes": len(MR.MODES), "kinds": dict(
            Counter(MR.MODE_KIND[m] for m in MR.MODES))},
        "n_universe": len(entries),
        "universe_by_mode": {m: sum(1 for e in entries if e["mode"] == m) for m in MR.MODES},
        "universe_by_kind": dict(Counter(e["kind"] for e in entries)),
        "direct_rule": "direct_* is palette-INDIFFERENT by construction, so it spreads the "
                       "9-cell DIRECT_GRID on one deterministic palette per location instead "
                       "of spreading palettes (sheet v1's rule, unchanged)",
    }
    return entries, report


def _entry(mode, kind, loc_key, family, src, mode_params, variant):
    return {"mode": mode, "kind": kind, "location_key": loc_key, "family": family,
            "src_image_id": src["image_id"], "palette": src["palette"],
            "src_p_ge3": src["p_ge3"], "color_params": src["params"],
            "render": src["render"], "mode_params": dict(mode_params), "variant": variant}


def _screen_stem(spec: SheetSpec, unit_key: str) -> str:
    """The on-disk name for a candidate's crop — a salted digest of the unit key.

    NOT the served `image_id`, which is assigned at write time from presentation position: a
    re-write in a different order would otherwise rename every file, and a half-renamed tree
    is a batch whose rows point at nothing."""
    return hashlib.blake2b(f"{spec.id_salt}|{unit_key}".encode(), digest_size=8).hexdigest()


# =========================================================================== #
# 2. Screen — the keeper render paths at a SCORING-ONLY geometry.
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
    from tools.mining.mining_gate import MiningScorer

    entries, rep = universe(spec)
    print_universe(rep)
    done = load_screen(spec)
    todo = [e for e in entries if e["unit_key"] not in done]
    if args.limit:
        # SPREAD across the universe, not a prefix: the universe is mode-major, so `todo[:8]`
        # exercises one render path and calls it an end-to-end.
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
        log(f"[screen] --limit {args.limit}: SPREAD -> modes {sorted({e['mode'] for e in todo})}")
    log(f"[screen] universe {len(entries)} · done {len(done)} · todo {len(todo)} · "
        f"{args.workers} workers x {ENGINE_THREADS} threads · geom {SCREEN_GEOM}")
    if not todo:
        return

    crops = spec.work / "screen_crops"
    fields = spec.work / "screen_fields"
    crops.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    scorer = MiningScorer(model_path=MP.ACTIVE_MINING_CKPT)
    log(f"[screen] mining head {MP.HEAD_VERSION} (K={scorer.k}) on {scorer.device}")
    if scorer.k != ST.K_TIERS:
        raise SystemExit(f"[screen] head K={scorer.k} but the suggestion rule is written for "
                         f"K={ST.K_TIERS} — fix the rule, do not coerce.")

    timeout_s = max(90.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    t0, n, errs, pending = time.time(), 0, [], []

    def flush(batch):
        """Score a batch of screen crops on the GPU and append their records."""
        if not batch:
            return
        paths = [crops / f"{e['image_id']}.jpg" for e in batch]
        scores = scorer.score_paths(paths)
        with spec.screen_log.open("a", encoding="utf-8") as fh:
            for e, s in zip(batch, scores):
                fh.write(json.dumps({
                    "unit_key": e["unit_key"], "mode": e["mode"], "kind": e["kind"],
                    "location_key": e["location_key"], "family": e["family"],
                    "variant": e["variant"], "palette": e["palette"],
                    "screen_pred": ST.expected_tier([s.p_ge2, s.p_ge3]),
                    "screen_p_ge2": s.p_ge2, "screen_p_ge3": s.p_ge3,
                    "screen_would_pass_gate": bool(s.passed),
                }) + "\n")
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
            except Exception as exc:                     # noqa: BLE001
                # Recorded in FULL, never head-truncated.
                errs.append({"unit_key": e["unit_key"], "mode": e["mode"],
                             "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                log(f"[screen] ERR {e['unit_key']}: {str(exc)[:160]}")
                continue
            pending.append(by_id[res["image_id"]])
            n += 1
            if len(pending) >= 48:
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
# 3. Select — mode x score, the fancy high cell first.
# =========================================================================== #
def score_bin(pred: float) -> str:
    """`hi` / `mid` / `lo` — the three bands the FROZEN mining cuts cut. The bin is the head's
    own suggested tier by construction, so "the high band" and "the tier the head would
    suggest" cannot drift apart."""
    t = ST.tier_from_pred(float(pred), ST.CUTS)
    return {3: "hi", 2: "mid", 1: "lo"}[t]


def select(spec: SheetSpec, screen_recs, max_rows=None, seed=None):
    max_rows = int(max_rows or spec.max_rows)
    rng = np.random.default_rng([int(seed if seed is not None else spec.draw_seed), 1])
    recs = sorted(screen_recs, key=lambda r: r["unit_key"])
    for r in recs:
        r["bin"] = score_bin(r["screen_pred"])
    unclaimed = {r["unit_key"]: r for r in recs}
    selected, per_bucket = [], []

    def claim(rows, bucket):
        for r in rows:
            if r["unit_key"] in unclaimed:
                del unclaimed[r["unit_key"]]
                r["bucket"] = bucket
                selected.append(r)

    def avail(**kw):
        out = list(unclaimed.values())
        for k, v in kw.items():
            out = [r for r in out if (r[k] in v if isinstance(v, (set, frozenset)) else r[k] == v)]
        return sorted(out, key=lambda r: r["unit_key"])

    def room():
        return max_rows - len(selected)

    # 1-2. the whole HIGH band, fancy first. This is the aimed slice and it is TAKE-ALL: a
    # false positive at the top of a scale is rare by construction, so anything short of
    # take-all puts the same ~15 rows in front of Matt that the last sheet did.
    for bucket, kinds in (("hi_fancy", FANCY_KINDS),
                          ("hi_pure", frozenset({"pure"}))):
        pool = avail(bin="hi", kind=kinds)
        take = pool[:room()]
        claim(take, bucket)
        per_bucket.append(dict(bucket=bucket, rule=f"TAKE ALL — mining pred >= {ST.CUTS[1]:g} "
                                                   f"(the head's own top tier), kinds "
                                                   f"{sorted(kinds)}",
                               available=len(pool), drawn=len(take),
                               by_mode=dict(sorted(Counter(r["mode"] for r in take).items()))))

    # 3. per-mode floor — every mode with supply is represented. Mid band before low, so a
    # mode's floor rows are the most informative ones it still has.
    floor_alloc = {}
    have = Counter(r["mode"] for r in selected)
    for mode in MR.MODES:
        short = spec.mode_floor - have.get(mode, 0)
        if short <= 0 or room() <= 0:
            floor_alloc[mode] = 0
            continue
        pool = avail(mode=mode, bin="mid")
        rng.shuffle(pool)
        lo = avail(mode=mode, bin="lo")
        rng.shuffle(lo)
        take = (pool + lo)[:min(short, room())]
        claim(take, "mode_floor")
        floor_alloc[mode] = len(take)
    per_bucket.append(dict(bucket="mode_floor",
                           rule=f"top every mode up to {spec.mode_floor} rows (mid band "
                                f"before low) — a floor, not a bonus",
                           available=None, drawn=sum(floor_alloc.values()),
                           by_mode=floor_alloc))

    # 4. the declared fancy oversample in the MID band.
    n_fancy = int(round(spec.fancy_mid_share * room()))
    pool = avail(bin="mid", kind=FANCY_KINDS)
    take, alloc, sizes = _apportion_take(pool, min(n_fancy, room()), rng)
    claim(take, "mid_fancy")
    per_bucket.append(dict(bucket="mid_fancy",
                           rule=f"{spec.fancy_mid_share:.0%} of the post-floor remainder to "
                                f"composite+direct in the mid band, balanced over modes "
                                f"(deal_round_robin)",
                           target=n_fancy, available=len(pool), drawn=len(take),
                           mode_available=sizes, mode_alloc=alloc))

    # 5. fill — everything else, balanced over modes, mid band before low.
    fill_alloc = Counter()
    for band in ("mid", "lo"):
        if room() <= 0:
            break
        pool = avail(bin=band)
        take, alloc, _sizes = _apportion_take(pool, room(), rng)
        claim(take, "fill")
        fill_alloc.update(alloc)
    per_bucket.append(dict(bucket="fill",
                           rule="the remainder, balanced over modes (deal_round_robin), mid "
                                "band before low",
                           available=None, drawn=sum(fill_alloc.values()),
                           by_mode=dict(sorted(fill_alloc.items()))))

    selected.sort(key=lambda r: r["unit_key"])
    mode_bin = defaultdict(Counter)
    for r in selected:
        mode_bin[r["mode"]][r["bin"]] += 1
    report = {
        "max_rows": max_rows, "drawn_rows": len(selected),
        "screened": len(recs),
        "cuts": list(ST.CUTS),
        "bucket_order": list(BUCKET_ORDER),
        "per_bucket": per_bucket,
        "drawn_by_bucket": dict(Counter(r["bucket"] for r in selected)),
        "drawn_by_mode": dict(sorted(Counter(r["mode"] for r in selected).items())),
        "drawn_by_kind": dict(sorted(Counter(r["kind"] for r in selected).items())),
        "drawn_by_bin": dict(sorted(Counter(r["bin"] for r in selected).items())),
        "drawn_mode_x_bin": {m: dict(c) for m, c in sorted(mode_bin.items())},
        "screened_by_bin": dict(sorted(Counter(r["bin"] for r in recs).items())),
        "screened_mode_x_bin": {m: dict(sorted(Counter(
            r["bin"] for r in recs if r["mode"] == m).items())) for m in MR.MODES},
        "modes_below_floor": {m: c for m, c in sorted(
            Counter(r["mode"] for r in selected).items()) if c < spec.mode_floor},
        "hi_band_retention": {
            "screened": sum(1 for r in recs if r["bin"] == "hi"),
            "drawn": sum(1 for r in selected if r["bin"] == "hi"),
            "note": "TAKE-ALL, so these agree unless the cap bound first"},
        "seed": int(seed if seed is not None else spec.draw_seed),
    }
    return selected, report


def _apportion_take(pool, n, rng, key=lambda r: r["mode"]):
    """`n` rows out of `pool`, balanced-or-drained over `key` through
    `apportion.deal_round_robin`, seeded-shuffled inside each cell."""
    cells = defaultdict(list)
    for r in pool:
        cells[key(r)].append(r)
    sizes = {k: len(v) for k, v in sorted(cells.items())}
    take_n = apportion.deal_round_robin(sizes, max(0, int(n)))
    out = []
    for k in sorted(cells):
        members = sorted(cells[k], key=lambda r: r["unit_key"])
        rng.shuffle(members)
        out.extend(members[:take_n[k]])
    return out, take_n, sizes


def assign_split(spec: SheetSpec, selected, loc_rep):
    """`split_units.build_split` — union-find over Julia-seed == parent-plane point,
    family-stratified. The July contract, on THIS sheet's location set."""
    keys = {r["location_key"] for r in selected}
    rep = {k: v for k, v in loc_rep.items() if k in keys}
    side, meta = build_split(rep, seed=spec.split_seed, eval_frac=spec.eval_frac)
    ok, why = units_are_disjoint(side, rep)
    if not ok:
        raise SystemExit(f"[mining-correction] split units span both sides: {why}")
    return side, {**meta, "units_disjoint": why}


# =========================================================================== #
# 4. Render — the frozen corpus pins.
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
    entries, _rep = universe(spec)
    by_key = {e["unit_key"]: e for e in entries}
    screen = load_screen(spec)
    if not screen:
        raise SystemExit("[render] no screen records — run `screen` first")
    selected, sel_report = select(spec, list(screen.values()), args.max_rows, args.seed)
    print_composition(sel_report)
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
                except Exception as exc:                 # noqa: BLE001
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
# 5. Write.
# =========================================================================== #
def provenance_block(spec, entry, rec, split_side, transfer_dropped) -> dict:
    p = entry["color_params"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "mining_correction_mode_x_score",
        "family": entry["family"],
        "location_key": entry["location_key"],
        "render_mode": entry["mode"],
        "mode_kind": entry["kind"],
        "mode_params": dict(entry.get("mode_params", {})),
        "rolloff": MR.rolloff_token(entry["mode"]),
        # THE COLORMAP RECIPE — inherited verbatim from the gate-passing wallpaper row; the
        # crop is a pure function of `render` + this block.
        "color_params": {
            "palette": entry["palette"],
            "palette_type": p.get("palette_type"), "palette_source": p.get("palette_source"),
            "reverse": p["reverse"], "log_premap": p["log_premap"], "gamma": p["gamma"],
            "phase": p["phase"], "n_cycles": p["n_cycles"],
            "transfer": p["transfer"], "transfer_gamma": p["transfer_gamma"],
            "interior_color": list(p.get("interior_color", [0.0, 0.0, 0.0])),
        },
        "transfer_dropped": transfer_dropped,
        # stratum
        "bucket": rec["bucket"],
        "score_bin": rec["bin"],
        "screen_pred": rec["screen_pred"],
        "screen_p_ge3": rec["screen_p_ge3"],
        "screen_path": f"the keeper render paths at a SCORING-ONLY geometry "
                       f"{SCREEN_GEOM[0]}x{SCREEN_GEOM[1]}ss{SCREEN_GEOM[2]} "
                       f"(build_mining_sheet.render_one with `geom`)",
        "split_side": split_side,
        "split_origin": "julia_parent_unionfind",
        "source": {
            "batch_id": "2026-07-09_wallpaper_headbatch_dramatic_v1",
            "image_id": entry["src_image_id"], "p_ge3": entry["src_p_ge3"],
            "gate": "wallpaper_head_v3 p_ge3 > 0.90 "
                    "(data/render_mode_corpus/gate_passers_v3.json)",
        },
        "not_served_before": {"prior_sheet": PRIOR_SHEET,
                              "rule": "(location_key, render_mode) excluded"},
    }


def run_write(spec: SheetSpec, args):
    from tools.mining.mining_gate import MiningScorer

    entries, uni_report = universe(spec)
    by_key = {e["unit_key"]: e for e in entries}
    rows_gp, _meta = BMS.load_gate_passers()
    _lr, _lf, _fl, loc_rep = BMS.index_population(rows_gp)

    screen = load_screen(spec)
    selected, sel_report = select(spec, list(screen.values()), args.max_rows, args.seed)
    done = load_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")
    live = [r for r in selected if r["unit_key"] in done]
    sides, split_meta = assign_split(spec, live, loc_rep)

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
        rows.append({
            "_unit_key": rec["unit_key"],
            "_crop_stem": e["image_id"],
            "render": BMS.render_block(e, loc),
            "provenance": provenance_block(
                spec, e, rec, sides[e["location_key"]],
                bool(done[rec["unit_key"]]["transfer_dropped"])),
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

    # PRESENTATION — good -> bad by the CONTINUOUS score, descending; ties on the crop stem.
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

    spec.batch_dir.mkdir(parents=True, exist_ok=True)
    with (spec.batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (spec.batch_dir / "route.json").write_text(json.dumps(route, indent=1), encoding="utf-8")

    sp = np.array([r["provenance"]["screen_p_ge3"] for r in rows])
    fp = np.array([r["p_ge3"] for r in rows])
    from tools.wallpaper import build_fresh_sheet as FS
    agree = {"n": len(rows), "spearman": FS._spearman(sp, fp),
             "mean_abs_delta": float(np.abs(sp - fp).mean()) if len(rows) else None,
             "top_tier_agreement": (float(((sp >= ST.CUTS[1] - 1) == (fp >= ST.CUTS[1] - 1))
                                          .mean()) if len(rows) else None),
             "note": f"screen = the same render path at {SCREEN_GEOM}, SCORING-ONLY; keeper = "
                     f"the stored {BMS.W}x{BMS.H} ss{BMS.SS} crop. head_mining_v1 IS the "
                     f"keeper score; the screen only chose what to render."}

    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
    incomplete = len(rows) < sel_report["drawn_rows"]
    accounted = set(done) | {e["unit_key"] for e in errors}
    unaccounted = sorted(r["unit_key"] for r in selected if r["unit_key"] not in accounted)

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_id": spec.batch_id,
        "generator_version": spec.generator_version,
        "labeler": None,
        "n_rows": len(rows),
        "schema_note":
            "MINING CORRECTION SHEET, stratified by render mode x mining score with the "
            "composite/direct ('fancy') modes deliberately OVER-DRAWN at high score — the "
            "busy-false-positive cell run 25's era gate named. Every row carries the complete "
            "render block, the mode + mode_params, the inherited colormap recipe "
            "(provenance.color_params — the crop is a pure function of the two), the "
            "location-grouped split_side, the bucket/score-bin it was drawn under and the "
            "mining-v1 pre-label (head_mining_v1 / p_ge3 / pred / suggested_tier). "
            "label.score is null on every row: the suggestion is NOT a label and is never "
            "merged as one. Presentation is sorted good->bad by pred, stamped contiguous in "
            "sheet_order, with opaque image_ids.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {sel_report['drawn_rows']} drawn rows are present — a BOUNDED or "
            f"INTERRUPTED run; re-run `render` then `write`.") if incomplete else None,
        "head": {
            "ckpt": MP.ACTIVE_MINING_CKPT, "version": MP.HEAD_VERSION,
            "gate_version": MP.MINING_GATE_VERSION,
            "role": "PRE-LABEL + SELECTION SCREEN — no gate, floor, threshold or pin is "
                    "applied or moved here; would_pass_gate is stamped and gates nothing",
            "scorer": "tools/mining/mining_gate.MiningScorer (fp32, no autocast; marginal "
                      "p_ge = cumprod(sigmoid), NEVER the CORN conditional)",
            "deploy_transform": "classifier.data.Transform(train=False) — 384x224 bicubic "
                                "stretch + the checkpoint's own mean/std",
        },
        "suggested_tier_rule": ST.fit_derivation(ST.CUTS, pred, MP.ACTIVE_MINING_CKPT,
                                                 MP.HEAD_VERSION),
        "universe": uni_report,
        "selection_report": sel_report,
        "screen_vs_keeper": agree,
        "split": {**split_meta,
                  "eval_rows": sum(1 for r in rows
                                   if r["provenance"]["split_side"] == "eval"),
                  "train_rows": sum(1 for r in rows
                                    if r["provenance"]["split_side"] == "train")},
        "seeds": {"draw_seed": sel_report["seed"], "split_seed": spec.split_seed,
                  "id_salt": spec.id_salt},
        "render_defaults": {
            "width": BMS.W, "height": BMS.H, "ss": BMS.SS, "filter": BMS.FILT,
            "jpg_quality": BMS.JPG_Q, "interior_mode": "black", "composition": "center",
            "why_these_pins": "sheet v1's pins, which are the July render-mode batches' own — "
                              "the mining head was trained on crops at these settings, and a "
                              "corpus whose two halves differ in geometry cannot be unioned",
            "screen_geometry": list(SCREEN_GEOM),
        },
        "realized": {
            "rows_by_mode": dict(sorted(Counter(
                r["render"]["render_mode"] for r in rows).items())),
            "rows_by_kind": dict(sorted(Counter(
                r["provenance"]["mode_kind"] for r in rows).items())),
            "rows_by_bucket": dict(Counter(r["provenance"]["bucket"] for r in rows)),
            "rows_by_score_bin": dict(sorted(Counter(
                r["provenance"]["score_bin"] for r in rows).items())),
            "rows_by_family": dict(Counter(r["render"]["fractal_type"] for r in rows)),
            "rows_by_split": dict(Counter(r["provenance"]["split_side"] for r in rows)),
            "suggested_tier_hist": dict(sorted(Counter(
                r["suggested_tier"] for r in rows).items())),
            "would_pass_mining_gate": sum(1 for r in rows
                                          if r["head_mining_v1"]["would_pass_gate"]),
            "transfer_dropped_rows": sum(1 for r in rows
                                         if r["provenance"]["transfer_dropped"]),
            "pred": {"min": round(min(r["pred"] for r in rows), 4),
                     "max": round(max(r["pred"] for r in rows), 4)},
        },
        "presentation": {
            "order": "sheet_order — DESCENDING pred (good -> bad), ties on the crop stem",
            "sorted_on": "the continuous head score (pred), NOT the suggested tier",
            "contiguous": True,
            "image_id": "OPAQUE `<prefix><slot>_<hash8>` — slot is presentation position "
                        "(published anyway: the sheet is sorted), hash is a salted digest of "
                        "the unit key, so the id encodes no mode, kind, bucket or score bin. "
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
            "drawn_rows": sel_report["drawn_rows"], "rendered_rows": len(rows),
            "n_failures": len(errors), "n_unaccounted": len(unaccounted),
            "unaccounted_rows": unaccounted[:50],
            "unaccounted_note": "drawn but neither rendered nor failed — a bounded (--limit) "
                                "or interrupted run; re-run `render` then `write`",
        },
    }
    (spec.batch_dir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print_summary(spec, batch)
    return batch


# =========================================================================== #
# Reporting.
# =========================================================================== #
def print_universe(rep):
    log("-" * 92)
    log(f"POPULATION  {rep['population']['rows']} gate-passer rows / "
        f"{rep['population']['locations']} locations   ·   EXCLUDING "
        f"{rep['exclusion']['served_location_mode_pairs']} (location, mode) pairs already "
        f"served by {rep['exclusion']['prior_sheet']}")
    log(f"UNIVERSE    {rep['n_universe']} candidates   {rep['universe_by_kind']}")
    log("-" * 92)


def print_composition(rep):
    log("-" * 92)
    log(f"{'bucket':<14}{'avail':>8}{'drawn':>8}   rule")
    for b in rep["per_bucket"]:
        log(f"{b['bucket']:<14}{str(b['available']):>8}{b['drawn']:>8}   {b['rule'][:58]}")
    log("-" * 92)
    log(f"drawn {rep['drawn_rows']} / cap {rep['max_rows']}  ·  screened {rep['screened']}")
    log(f"by kind: {rep['drawn_by_kind']}   by bin: {rep['drawn_by_bin']}  "
        f"(screened by bin {rep['screened_by_bin']})")
    log(f"hi band: {rep['hi_band_retention']['drawn']} of "
        f"{rep['hi_band_retention']['screened']} screened")
    log(f"by mode: {rep['drawn_by_mode']}")
    if rep["modes_below_floor"]:
        log(f"BELOW the {MODE_FLOOR}-row mode floor (supply-bound): {rep['modes_below_floor']}")
    log("-" * 92)


def print_summary(spec, batch):
    r = batch["realized"]
    log("\n" + "=" * 92)
    log(f"MINING CORRECTION SHEET — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 92)
    log(f"rows {batch['n_rows']} / drawn {batch['run_status']['drawn_rows']}  ·  failures "
        f"{batch['run_status']['n_failures']}")
    log(f"by mode:   {r['rows_by_mode']}")
    log(f"by kind:   {r['rows_by_kind']}   by bin: {r['rows_by_score_bin']}")
    log(f"by bucket: {r['rows_by_bucket']}")
    log(f"by split:  {r['rows_by_split']}")
    log(f"suggested: {r['suggested_tier_hist']}  (cuts {batch['suggested_tier_rule']['cuts']})")
    log(f"would pass the mining gate: {r['would_pass_mining_gate']}  ·  pred {r['pred']}")
    a = batch["screen_vs_keeper"]
    log(f"screen-vs-keeper: spearman {a['spearman']}  mean|delta| {a['mean_abs_delta']}")
    log(f"-> {spec.batch_dir}")
    log(f"-> serve: uv run python tools/viz/serve.py   then")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Mode x score mining correction sheet builder.")
    ap.add_argument("stage", choices=("estimate", "screen", "select", "render", "write"))
    ap.add_argument("--sheet", default="v2", choices=sorted(SHEETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap units this run (SPREAD across the plan, not a prefix)")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="concurrent engine PROCESSES — the project cap is 4, do not raise")
    ap.add_argument("--dry-run", action="store_true", help="render: composition only")
    ap.add_argument("--unit-timeout", type=float, default=900.0)
    ap.add_argument("--wall-budget-s", type=float, default=4 * 3600.0)
    args = ap.parse_args(argv)

    if args.workers > WORKERS:
        raise SystemExit(f"[mining-correction] --workers {args.workers} exceeds the project "
                         f"process cap of {WORKERS} (CLAUDE.md).")
    missing = MR.missing_recipes()
    if missing:
        raise SystemExit(f"[mining-correction] roster/recipe mismatch: {missing}")

    spec = SHEETS[args.sheet]
    spec.work.mkdir(parents=True, exist_ok=True)
    prio = cc.set_below_normal_priority()
    log(f"[mining-correction] {spec.batch_id} · priority {prio} · {args.workers} workers x "
        f"{ENGINE_THREADS} rayon threads")

    if args.stage == "estimate":
        _e, rep = universe(spec)
        print_universe(rep)
        log(json.dumps(rep, indent=2))
        return 0
    if args.stage == "screen":
        run_screen(spec, args)
        return 0
    if args.stage == "select":
        screen = load_screen(spec)
        if not screen:
            raise SystemExit("[select] no screen records — run `screen` first")
        _sel, rep = select(spec, list(screen.values()), args.max_rows, args.seed)
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
