r"""build_wallpaper_sitting.py — the BUCKETED wallpaper correction sitting.

ONE finished coloured render per intake location, at the shared label-crop pins, pre-labelled
by the deployed wallpaper head and served good->bad for correction. It is the (27) sitting of
the ckpt-37 queue and the motivating slice for the from-scratch wallpaper v4b retrain
(prompts/sittings_27.md).

HOW IT DIFFERS FROM ITS TWO 2026-08-05 PREDECESSORS, which it otherwise reuses wholesale.
`build_fresh_sheet.py` drew 240 locations stratified across five v3 SCORE bins and rendered
four pool-draw palettes each; `build_colorize_sheet.py` took the same draw through the live
colorize path. Both answered "how does the head see the intake". This one answers a different
question — **what does v4b need to see** — so the draw is BUCKETED against named populations
rather than spread over score bins, and each location contributes exactly ONE row: the
argmax-palette render, which is the thing a release actually ships
(`enrich --mode render` selects the same way).

THE BUCKETS, in claim order. A location lands in exactly one; the first bucket to claim it
owns it, and every bucket takes what supply allows rather than forcing its target
(realized counts are recorded per bucket in `batch.json`, never back-filled to look full).

  1. phoenix:classic     TAKE ALL. It is 39 of 2,867 rows and every bucketed cut in this tree
                         skips it by construction, so it is an explicit bucket or it is absent.
  2. minibrot_maneuver   the v4b motivating slice: rows minted by a MANEUVER
                         (`mix_source` `maneuver:*` / `triggered:*` — snap-to-nucleus,
                         neighborhood-expand, lateral-to-sibling) plus the `q4_harvest` vein,
                         whose rows are minibrot-centred by construction (`q4_minibrot_id`).
  3. below_retired_floor the restructure's delta population: v3 p_ge3 in
                         [`floors.GOOD_FLOOR`, the retired `floors.WALLPAPER_RELEASE`) — the
                         rows that now compete for release and would have been cut before
                         2026-08-09. 33 of run 25's 48 release candidates sat here.
  4. top_slice           the highest v3 scores, strictly by rank. No apportionment: "the top
                         slice" means the top slice.
  5. partition_floor     top up every partition with supply to `COVERAGE_FLOOR` rows, counting
                         what buckets 1-4 already gave it. A floor, not a bonus.
  6. remainder           fill to the target, apportioned NEAR-PROPORTIONALLY TO INTAKE through
                         `ranked_intake.partition_slots` — the same rule the release slots use.

STRATIFY BY PARTITION / SCORE / SOURCE-VEIN ONLY, NEVER BY PALETTE. The palette is not a
stratum and is not drawn for coverage: it is whatever the head likes best for that location,
because the row has to be a finished picture somebody would ship. Palette variety across the
sheet is a CONSEQUENCE of 2,867 different locations, and it is reported, not targeted.

THE SCREEN IS NOT THE STAMP (inherited from `build_fresh_sheet`, one geometry cheaper). A
bucketed draw needs a v3 score for every location before any crop exists, and dumping the
label field 2,867 times to find out costs ~6x what the sheet does. So the screen dumps at
`SCREEN_W x SCREEN_H ss SCREEN_SS` and scores through `colormap.coarse_field` +
`render_candidates_coarse` — the same 512x288 scoring grid the beam's pref pick runs behind,
reached from a smaller field. The score STAMPED IN-ROW is re-derived on the real label crop,
both are kept per row (`head_v3.p_ge3` vs `provenance.screen_p_ge3`), and the run prints their
rank agreement so the proxy is audited against the thing it proxied for.

  uv run python -u tools/wallpaper/build_wallpaper_sitting.py estimate
  uv run python -u tools/wallpaper/build_wallpaper_sitting.py screen --limit 8       # smoke
  uv run python -u tools/wallpaper/build_wallpaper_sitting.py screen  > scratch/wallpaper_sitting/screen.log 2>&1
  uv run python -u tools/wallpaper/build_wallpaper_sitting.py select                 # dry composition
  uv run python -u tools/wallpaper/build_wallpaper_sitting.py render --limit 6       # bounded E2E (stamps INCOMPLETE)
  uv run python -u tools/wallpaper/build_wallpaper_sitting.py render > scratch/wallpaper_sitting/render.log 2>&1
  uv run python -u tools/wallpaper/build_wallpaper_sitting.py write
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "queries", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "emission", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                             # noqa: E402  THE two apportionment rules
import colormap as cm                        # noqa: E402
import corpus_common as cc                   # noqa: E402  engine launch defaults + priority
import location as loc_mod                   # noqa: E402
import partitions as P                       # noqa: E402  THE partition map
import query_sampler as qs                   # noqa: E402
import ranked_intake as RI                   # noqa: E402  THE near-proportional slot rule
from label_crop import (                     # noqa: E402  THE shared label-crop pins
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q, ensure_label_field, render_label_crop)
from tools.emission import descriptor as D            # noqa: E402
from tools.emission import floors as F                # noqa: E402  THE cut owner
from tools.emission import ledger_rescore as LR       # noqa: E402  THE intake ledger list
from tools.wallpaper import wallpaper_pins as WP      # noqa: E402  the head pin (torch-free)
from tools.wallpaper.suggest_tier import (            # noqa: E402
    INTAKE_CUTS, INTAKE_DERIVATION, expected_tier, tier_from_pred)

# The 2026-08-05 sibling owns the palette pool, the per-location seeded candidate draw and the
# two scoring helpers. Imported rather than re-derived: a second palette draw is a second
# sheet composition nobody decided, and `test_fresh_sheet.py` pins these.
import build_fresh_sheet as FS               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
WALLPAPER_CORPUS = ROOT / "data" / "wallpaper_corpus"

# --- the screen geometry ----------------------------------------------------
# The scoring grid is `colormap.SCORE_COARSE_W x _H` either way; this is only how big a field
# gets area-downsampled INTO it. 512x288 ss2 is 590k samples against the label field's 3.69M,
# and the field dump is what a 2,867-location screen is priced in.
SCREEN_W, SCREEN_H, SCREEN_SS = cm.SCORE_COARSE_W, cm.SCORE_COARSE_H, 2

# Per-location coverage floor (bucket 5) and the two process caps.
COVERAGE_FLOOR = 40
WORKERS = 4                 # concurrent `fractal-generator.exe` — the project cap, do NOT raise
ENGINE_THREADS = 3          # rayon threads per engine child (12 logical cores / 4 processes)


# =========================================================================== #
# The sitting spec — a frozen dataclass from the start (CLAUDE.md, "Writing a builder for one
# instance"): `build_fresh_sheet` held exactly these fields at module scope and a second
# sitting could not be built without a refactor. This is that refactor, paid once, here.
# =========================================================================== #
@dataclass(frozen=True)
class SittingSpec:
    key: str
    batch_id: str
    generator_version: str
    img_prefix: str
    id_salt: str
    target_rows: int
    n_screen: int               # seeded palette candidates screened per location
    seed: int
    split_seed: int
    eval_frac: float
    minibrot_target: int
    below_floor_target: int
    top_slice_target: int
    coverage_floor: int = COVERAGE_FLOOR

    @property
    def batch_dir(self) -> Path:
        return WALLPAPER_CORPUS / "batches" / self.batch_id

    @property
    def work(self) -> Path:
        return ROOT / "scratch" / "wallpaper_sitting" / self.key

    @property
    def screen_log(self) -> Path:
        return self.work / "screen.jsonl"

    @property
    def labels_export(self) -> str:
        return f"labels/scores_{self.batch_id}.json"

    @property
    def ui_url(self) -> str:
        return (f"tools/viz/wallpaper_label.html?corpus=wallpaper_corpus&tiers=4"
                f"&order=file&batch={self.batch_id}")


SITTINGS = {
    "v2": SittingSpec(
        key="v2",
        batch_id="2026-08-10_wallpaper_correction_v2",
        generator_version="wallpaper_correction_v2",
        img_prefix="wc2",
        id_salt="wallpaper_correction_v2/2026-08-10",
        target_rows=960,
        n_screen=8,
        seed=27,
        split_seed=0,
        eval_frac=0.30,
        minibrot_target=300,
        below_floor_target=150,
        top_slice_target=75,
    ),
}

BUCKET_ORDER = ("phoenix_classic", "minibrot_maneuver", "below_retired_floor",
                "top_slice", "partition_floor", "remainder")


def log(msg: str):
    print(msg, flush=True)


# =========================================================================== #
# 1. Population — the stage-2 admitted intake, and nothing else.
# =========================================================================== #
# The ten-ledger admitted union (`ledger_rescore.LEDGERS`, run 25's three legs included since
# 2026-08-10). The `human_q3plus` library seed that `build_fresh_sheet` unions in is
# DELIBERATELY ABSENT: its 168 looks already carry a human 3-or-4 and were served whole on
# 2026-08-05, and "remainder proportional to intake" is a statement about the intake, which a
# library snapshot is not part of.
def population():
    """`(sources, report)` — one source dict per admitted intake location."""
    srcs, diag = FS._union_sources()
    for s in srcs:
        s["vein"] = vein_of(s)
    report = {
        "source": "the ten-ledger stage-2 admitted union "
                  "(tools/emission/ledger_rescore.LEDGERS -> descriptor.load_union_admitted)",
        "ledgers": [rel for _t, rel in LR.LEDGERS],
        "n_population": len(srcs),
        "union_diag": {k: v for k, v in diag.items()
                       if k in ("n_union", "per_ledger", "n_id_collisions",
                                "n_location_overlaps")},
        "by_partition": dict(sorted(Counter(s["partition"] for s in srcs).items())),
        "by_source_tag": dict(Counter(s["source_tag"] for s in srcs).most_common()),
        "by_vein": dict(sorted(Counter(s["vein"] for s in srcs).items())),
        "human_q3plus_excluded": "the library seed is not intake; its 168 looks already carry "
                                 "a human 3-or-4 and were served whole in "
                                 "2026-08-05_wallpaper_fresh_sheet_v1",
    }
    return srcs, report


# The SOURCE-VEIN axis. Derived from the row's own `mix_source` at read time, never stored:
# `maneuver:`/`triggered:` are the minted-by-a-maneuver prefixes `minibrot_maneuvers.py` emits
# (snap_to_nucleus / neighborhood_expand / lateral_to_sibling; `triggered:` is the same
# operator fired by the walk's own trigger rather than by the scheduler), and `q4_harvest` is
# the near-minibrot harvest, whose every row carries a `q4_minibrot_id`.
MANEUVER_PREFIXES = ("maneuver:", "triggered:")
MINIBROT_VEINS = frozenset({"maneuver", "q4_harvest"})


def vein_of(src: dict) -> str:
    tag = src.get("source_tag") or ""
    if tag.startswith(MANEUVER_PREFIXES):
        return "maneuver"
    if tag == "q4_harvest":
        return "q4_harvest"
    if tag == "dive":
        return "dive"
    if tag in ("classic_phoenix", "phoenix_grid"):
        return tag
    return "descent"


# =========================================================================== #
# 2. Screen — cheap field, coarse recolor, v3 marginals. SCORING-ONLY.
# =========================================================================== #
def screen_field_stem(loc) -> str:
    ptok = loc_mod.maxiter_policy_token()
    suffix = f"|{ptok}" if ptok else ""
    h = hashlib.sha1(
        f"{loc.key()}|{SCREEN_W}x{SCREEN_H}ss{SCREEN_SS}|{loc.maxiter}{suffix}".encode()
    ).hexdigest()[:16]
    return f"{loc.family}_{h}_{SCREEN_W}x{SCREEN_H}ss{SCREEN_SS}"


def ensure_screen_field(loc, fields_dir: Path, timeout_s: float = 900.0):
    """Dump (or reuse) the SCREEN-geometry smooth field. Mirrors `label_crop.ensure_label_field`
    at a smaller geometry; the two stems are disjoint, so a screen field can never be served
    where a label field is meant."""
    fields_dir.mkdir(parents=True, exist_ok=True)
    stem = screen_field_stem(loc)
    bin_path, json_path = fields_dir / f"{stem}.bin", fields_dir / f"{stem}.json"
    if not (bin_path.exists() and json_path.exists()):
        cmd = [str(EXE), "render-one",
               "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
               "--width", str(SCREEN_W), "--height", str(SCREEN_H),
               "--supersample", str(SCREEN_SS), "--maxiter", str(loc.maxiter),
               "--dump-field", str(bin_path)] + loc_mod.render_one_flags(loc)
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=timeout_s, env=dict(os.environ,
                                                       RAYON_NUM_THREADS=str(ENGINE_THREADS)),
                           creationflags=cc.default_creationflags())
        if r.returncode != 0:
            raise RuntimeError(f"screen dump-field failed for {stem}:\n{r.stderr[-400:]}")
    return cm.load_field(str(bin_path), str(json_path))


def wipe_screen_field(loc, fields_dir: Path):
    stem = screen_field_stem(loc)
    for ext in (".bin", ".json"):
        try:
            (fields_dir / f"{stem}{ext}").unlink(missing_ok=True)
        except OSError:
            pass


_W_LIB = _W_SAMPLER = _W_POOL = None


def _init_screen_worker():
    """One engine child per worker, `ENGINE_THREADS` rayon threads each. The palette library
    and pool are loaded once per worker, not once per location."""
    global _W_LIB, _W_SAMPLER, _W_POOL
    os.environ["RAYON_NUM_THREADS"] = str(ENGINE_THREADS)
    _W_SAMPLER = qs.PaletteSampler(qs.load_pool_library())
    _W_LIB = _W_SAMPLER.library
    _W_POOL = FS.palette_pool(_W_SAMPLER)


def screen_one(job):
    """Render the screen field, recolor N candidates on the coarse grid, return the images.

    The v3 SCORE is not taken here: the head lives on one GPU and N worker processes each
    holding a CUDA context is the wrong shape. The worker returns uint8 images and the parent
    scores them in one batched pass."""
    unit_key, render_block, seed_offset, fields_s, n_screen, timeout_s = job
    loc = loc_mod.from_render_block(render_block)
    fields_dir = Path(fields_s)
    try:
        field = ensure_screen_field(loc, fields_dir, timeout_s)
        prep = cm.stretch_field(field)
        coarse = cm.coarse_field(prep)
        cfgs = FS.draw_candidates(loc, _W_POOL, _W_SAMPLER,
                                  seed_offset=seed_offset, n=n_screen)
        imgs = cm.render_candidates_coarse(coarse, cfgs, _W_LIB)
        cand = [{"config": json.loads(c.to_json()), "palette": c.palette,
                 "palette_type": _W_LIB.palette_type(c.palette),
                 "palette_source": _W_SAMPLER.source_of(c.palette)} for c in cfgs]
        return unit_key, cand, np.asarray(imgs), None
    except Exception as e:                                   # noqa: BLE001
        return unit_key, [], np.zeros((0, 1, 1, 3), np.uint8), f"{type(e).__name__}: {e}"
    finally:
        wipe_screen_field(loc, fields_dir)


def load_screen(spec: SittingSpec) -> dict:
    done = {}
    if spec.screen_log.exists():
        for line in spec.screen_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit_key"]] = rec
    return done


def run_screen(spec: SittingSpec, args):
    from classifier.inference import load_scorer

    srcs, pop_report = population()
    fields_dir = spec.work / "screen_fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    done = load_screen(spec)
    todo = [s for s in srcs if s["unit_key"] not in done]
    if args.limit:
        # SPREAD across the population, not the first N: the union is emitted ledger-major, so
        # a prefix exercises one family's render path and calls it an end-to-end.
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
        log(f"[screen] --limit {args.limit}: SPREAD -> partitions "
            f"{sorted({s['partition'] for s in todo})}")
    log(f"[screen] population {len(srcs)} · {len(done)} already screened · {len(todo)} to run "
        f"· {args.workers} workers x {ENGINE_THREADS} threads")
    if not todo:
        return

    scorer = load_scorer(str(WP.HEAD_CKPT))
    log(f"[screen] head {WP.HEAD_VERSION} on {scorer.device} · {spec.n_screen} palettes/loc "
        f"· screen field {SCREEN_W}x{SCREEN_H}ss{SCREEN_SS}")

    by_key = {s["unit_key"]: s for s in todo}
    # A per-unit backstop clamped to the run's own budget (CLAUDE.md, "a backstop longer than
    # the job's budget is not a backstop").
    timeout_s = max(120.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    jobs = [(s["unit_key"], render_block_of(s["loc"]), FS._stable_seed(s["key"]),
             str(fields_dir), spec.n_screen, timeout_s) for s in todo]

    t0, times, n_ok, n_err = time.time(), [], 0, 0
    with spec.screen_log.open("a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_screen_worker) as ex:
            futs = [ex.submit(screen_one, j) for j in jobs]
            for k, fut in enumerate(as_completed(futs)):
                unit_key, cand, imgs, err = fut.result()
                s = by_key[unit_key]
                if err is None and len(cand):
                    marg = FS._marginals(scorer, list(imgs))
                    for j, c in enumerate(cand):
                        c["p_ge2"] = float(marg[j, 0])
                        c["p_ge3"] = float(marg[j, 1])
                        c["p_ge4"] = float(marg[j, 2]) if marg.shape[1] > 2 else None
                        c["pred"] = expected_tier(marg[j])
                    n_ok += 1
                else:
                    n_err += 1
                fh.write(json.dumps({
                    "unit_key": unit_key, "key": s["key"], "family": s["loc"].family,
                    "partition": s["partition"], "vein": s["vein"],
                    "source_tag": s["source_tag"], "floor_admit": s["floor_admit"],
                    "fw": s["loc"].fw, "maxiter": s["loc"].maxiter,
                    "error": err, "candidates": cand}) + "\n")
                fh.flush()
                times.append(time.time() - t0)
                if (k + 1) % 25 == 0 or k < 3:
                    el = time.time() - t0
                    rate = (k + 1) / el
                    log(f"[screen] {k+1}/{len(todo)}  {n_err} failed  "
                        f"{rate:.2f} loc/s -> eta {(len(todo)-k-1)/rate/60:.0f} min "
                        f"(elapsed {el/60:.0f} min)")
    log(f"[screen] done: {n_ok} screened, {n_err} failed, "
        f"{(time.time()-t0)/60:.1f} min")


def render_block_of(loc) -> dict:
    """A Location -> the render dict `location.from_render_block` reads back. Workers get this
    instead of the Location object so the payload is plain JSON."""
    blk = {"cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
           "fractal_type": loc.family, "c_re": loc.c_re, "c_im": loc.c_im}
    blk.update(loc.params)
    return blk


# =========================================================================== #
# 3. Select — the six buckets.
# =========================================================================== #
def _score(rec) -> float:
    """A location's v3 SCREEN score: the best palette's p_ge3. The row this sitting serves is
    the argmax-palette render, so the location's score IS its best candidate's."""
    return max(c["p_ge3"] for c in rec["candidates"])


def _best_candidate(rec) -> dict:
    return max(rec["candidates"], key=lambda c: (c["p_ge3"], c["palette"]))


def _draw_apportioned(pool, n, rng, key=lambda r: r["partition"]):
    """`n` rows out of `pool`, apportioned BALANCED-OR-DRAINED over `key` through
    `apportion.deal_round_robin`, seeded-shuffled inside each cell.

    Balanced, not proportional, and that is the point for a bucket: a bucket exists to make a
    named population visible, and letting julia:multibrot3 own it (673 of 2,867) would undo
    that. The PROPORTIONAL rule is applied once, to the remainder, where it belongs."""
    cells = defaultdict(list)
    for r in pool:
        cells[key(r)].append(r)
    sizes = {k: len(v) for k, v in sorted(cells.items())}
    take = apportion.deal_round_robin(sizes, n)
    out = []
    for k in sorted(cells):
        members = sorted(cells[k], key=lambda r: r["unit_key"])
        rng.shuffle(members)
        out.extend(members[:take[k]])
    return out, take, sizes


def select(spec: SittingSpec, screen_recs, target_rows=None, seed=None):
    """`(selected, report)` — the bucketed draw. Each row carries `bucket`."""
    target_rows = int(target_rows or spec.target_rows)
    rng = np.random.default_rng([int(seed if seed is not None else spec.seed), 1])

    ok = [r for r in screen_recs if not r["error"] and r["candidates"]]
    for r in ok:
        r["score"] = _score(r)
        r["best"] = _best_candidate(r)
    ok.sort(key=lambda r: r["unit_key"])
    unclaimed = {r["unit_key"]: r for r in ok}
    intake_by_part = Counter(r["partition"] for r in ok)
    selected, per_bucket = [], []

    def claim(rows, bucket):
        for r in rows:
            if r["unit_key"] in unclaimed:
                del unclaimed[r["unit_key"]]
                r["bucket"] = bucket
                selected.append(r)

    def avail():
        return list(unclaimed.values())

    def room():
        return target_rows - len(selected)

    # 1. phoenix:classic — TAKE ALL.
    cls = [r for r in avail() if r["partition"] == P.CLASSIC_PHOENIX]
    claim(cls, "phoenix_classic")
    per_bucket.append(dict(bucket="phoenix_classic", target=None, rule="TAKE ALL",
                           available=len(cls), drawn=len(cls)))

    # 2. minibrot / maneuver-view.
    pool = [r for r in avail() if r["vein"] in MINIBROT_VEINS]
    n = min(spec.minibrot_target, room(), len(pool))
    take, alloc, sizes = _draw_apportioned(pool, n, rng)
    claim(take, "minibrot_maneuver")
    per_bucket.append(dict(bucket="minibrot_maneuver", target=spec.minibrot_target,
                           rule="veins " + "/".join(sorted(MINIBROT_VEINS)) +
                                ", balanced over partitions (deal_round_robin)",
                           available=len(pool), drawn=len(take),
                           partition_available=sizes, partition_alloc=alloc,
                           by_vein=dict(sorted(Counter(r["vein"] for r in take).items()))))

    # 3. below the retired release floor. The upper edge goes through `Floor.annotates`, not
    # through `.value`: the comparison carries a HEAD STAMP CHECK, and 0.90 is a point on v3's
    # probability scale that means nothing on v4b's. Reading `.value` here would silently
    # define the band against a head nobody is using.
    lo = F.GOOD_FLOOR
    hi = F.WALLPAPER_RELEASE.value
    pool = [r for r in avail()
            if r["score"] >= lo and not F.WALLPAPER_RELEASE.annotates(r["score"])]
    n = min(spec.below_floor_target, room(), len(pool))
    take, alloc, sizes = _draw_apportioned(pool, n, rng)
    claim(take, "below_retired_floor")
    per_bucket.append(dict(bucket="below_retired_floor", target=spec.below_floor_target,
                           rule=f"v3 screen p_ge3 in [{lo:g}, {hi:g}) — GOOD_FLOOR to the "
                                f"retired {F.WALLPAPER_RELEASE.name} floor, balanced over "
                                f"partitions",
                           band=[lo, hi], available=len(pool), drawn=len(take),
                           partition_available=sizes, partition_alloc=alloc))

    # 4. top slice — strictly by rank.
    pool = sorted(avail(), key=lambda r: (-r["score"], r["unit_key"]))
    n = min(spec.top_slice_target, room())
    take = pool[:n]
    claim(take, "top_slice")
    per_bucket.append(dict(bucket="top_slice", target=spec.top_slice_target,
                           rule="highest v3 screen p_ge3, NO apportionment",
                           available=len(pool), drawn=len(take),
                           score_range=([round(take[0]["score"], 4),
                                         round(take[-1]["score"], 4)] if take else None)))

    # 5. per-partition coverage floor — a FLOOR, counting what buckets 1-4 already gave.
    have = Counter(r["partition"] for r in selected)
    floor_alloc = {}
    for part in sorted(intake_by_part):
        short = spec.coverage_floor - have.get(part, 0)
        if short <= 0 or room() <= 0:
            floor_alloc[part] = 0
            continue
        members = sorted((r for r in avail() if r["partition"] == part),
                         key=lambda r: r["unit_key"])
        rng.shuffle(members)
        take = members[:min(short, room())]
        claim(take, "partition_floor")
        floor_alloc[part] = len(take)
    per_bucket.append(dict(bucket="partition_floor", target=spec.coverage_floor,
                           rule=f"top every partition up to {spec.coverage_floor} rows, "
                                f"counting buckets 1-4 (a floor, not a bonus)",
                           available=None, drawn=sum(floor_alloc.values()),
                           partition_topup=floor_alloc,
                           partition_after={p: c for p, c in sorted(
                               Counter(r["partition"] for r in selected).items())}))

    # 6. remainder — near-proportional to INTAKE.
    n = room()
    shares = {p: c / sum(intake_by_part.values()) for p, c in sorted(intake_by_part.items())}
    slots = RI.partition_slots(shares, n) if n > 0 else {}
    rem_alloc = {}
    for part in sorted(slots, key=lambda p: -slots[p]):
        members = sorted((r for r in avail() if r["partition"] == part),
                         key=lambda r: r["unit_key"])
        rng.shuffle(members)
        take = members[:min(slots[part], room())]
        claim(take, "remainder")
        rem_alloc[part] = len(take)
    # Anything the proportional pass could not seat (a partition drained) is handed to the
    # partitions that still have supply, so a drained cell shrinks the sheet only when the
    # WHOLE population is drained.
    if room() > 0:
        spill = sorted(avail(), key=lambda r: r["unit_key"])
        rng.shuffle(spill)
        take = spill[:room()]
        claim(take, "remainder")
        for r in take:
            rem_alloc[r["partition"]] = rem_alloc.get(r["partition"], 0) + 1
    per_bucket.append(dict(bucket="remainder", target=n,
                           rule="near-proportional to intake via "
                                "ranked_intake.partition_slots (sequence_by_deficit), "
                                "shortfall from a drained partition re-spilled",
                           available=None, drawn=sum(rem_alloc.values()),
                           partition_slots=slots, partition_drawn=rem_alloc,
                           intake_shares={p: round(v, 4) for p, v in shares.items()}))

    selected.sort(key=lambda r: r["unit_key"])
    report = {
        "target_rows": target_rows, "drawn_rows": len(selected),
        "screened": len(screen_recs), "screen_failures": len(screen_recs) - len(ok),
        "eligible": len(ok),
        "bucket_order": list(BUCKET_ORDER),
        "per_bucket": per_bucket,
        "drawn_by_bucket": dict(Counter(r["bucket"] for r in selected)),
        "drawn_by_partition": dict(sorted(Counter(r["partition"] for r in selected).items())),
        "drawn_by_vein": dict(sorted(Counter(r["vein"] for r in selected).items())),
        "drawn_by_source_tag": dict(Counter(r["source_tag"] for r in selected).most_common()),
        "intake_by_partition": dict(sorted(intake_by_part.items())),
        "coverage_floor": spec.coverage_floor,
        "partitions_below_coverage_floor": {
            p: c for p, c in sorted(Counter(r["partition"] for r in selected).items())
            if c < spec.coverage_floor},
        "seed": int(seed if seed is not None else spec.seed),
        "palette_note": "NOT a stratum. Each row is the location's argmax-p_ge3 palette out "
                        "of its own seeded screen draw; palette spread is reported, never "
                        "targeted.",
    }
    return selected, report


def assign_split(spec: SittingSpec, selected):
    """Location-grouped seeded eval assignment, stratified BY BUCKET so each named population
    is represented on both sides. One location -> one row -> one side."""
    rng = np.random.RandomState(spec.split_seed)
    sides, n_eval = {}, 0
    for bucket in BUCKET_ORDER:
        keys = sorted(r["unit_key"] for r in selected if r.get("bucket") == bucket)
        rng.shuffle(keys)
        k = int(round(spec.eval_frac * len(keys)))
        for j, uk in enumerate(keys):
            sides[uk] = "eval" if j < k else "train"
        n_eval += k
    return sides, n_eval


# =========================================================================== #
# 4. Render — one label crop per location, at the shared pins.
# =========================================================================== #
_R_LIB = None


def _init_render_worker():
    global _R_LIB
    os.environ["RAYON_NUM_THREADS"] = str(ENGINE_THREADS)
    _R_LIB = qs.load_pool_library()


def render_one(job):
    unit_key, render_block, cand_json, crop_s, fields_s, timeout_s = job
    t0 = time.time()
    loc = loc_mod.from_render_block(render_block)
    os.environ["RAYON_NUM_THREADS"] = str(ENGINE_THREADS)
    try:
        field = ensure_label_field(loc, fields_dir=Path(fields_s), timeout_s=timeout_s)
        prep = cm.stretch_field(field)
        cfg = cm.CandidateConfig.from_json(json.dumps(cand_json))
        w, h = render_label_crop(field, cfg, _R_LIB, Path(crop_s), prep=prep)
    finally:
        _wipe_label_field(loc, Path(fields_s))
    return dict(unit_key=unit_key, w=w, h=h, secs=time.time() - t0)


def _wipe_label_field(loc, fields_dir: Path):
    """15 MB per label field x ~960 locations is 14 GB of scratch for fields read once."""
    ptok = loc_mod.maxiter_policy_token()
    suffix = f"|{ptok}" if ptok else ""
    h = hashlib.sha1(
        f"{loc.key()}|{LABEL_W}x{LABEL_H}ss{LABEL_SS}|{loc.maxiter}{suffix}".encode()
    ).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{LABEL_W}x{LABEL_H}ss{LABEL_SS}"
    for ext in (".bin", ".json"):
        try:
            (fields_dir / f"{stem}{ext}").unlink(missing_ok=True)
        except OSError:
            pass


def ledger_path(spec: SittingSpec) -> Path:
    return spec.batch_dir / "_progress_ledger.jsonl"


def load_render_ledger(spec: SittingSpec) -> dict:
    """`{unit_key: record}` for units with a ledger row AND an on-disk crop."""
    done, p, crops = {}, ledger_path(spec), spec.batch_dir / "crops"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if (crops / f"{rec['unit_key_hash']}.jpg").exists():
                    done[rec["unit_key"]] = rec
    return done


def crop_stem(spec: SittingSpec, unit_key: str) -> str:
    """The on-disk crop name — a salted hash of the unit key, NOT the served id.

    The served `image_id` is assigned at WRITE time from presentation position, so naming a
    crop after it would make a re-write with a different order rename ~960 files (and a
    half-renamed tree is a batch whose rows point at nothing). The stem is stable under any
    re-order; `images.jsonl` carries the join."""
    return hashlib.blake2b(f"{spec.id_salt}|{unit_key}".encode(), digest_size=8).hexdigest()


def run_render(spec: SittingSpec, args):
    screen = load_screen(spec)
    if not screen:
        raise SystemExit("[render] no screen records — run `screen` first")
    srcs = {s["unit_key"]: s for s in population()[0]}
    selected, sel_report = select(spec, list(screen.values()), args.target_rows, args.seed)
    print_composition(sel_report)
    if args.dry_run:
        return

    crops = spec.batch_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    fields_dir = spec.work / "render_fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    done = load_render_ledger(spec)
    todo = [r for r in selected if r["unit_key"] not in done]
    if args.limit:
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
        log(f"[render] --limit {args.limit}: SPREAD -> buckets "
            f"{sorted({r['bucket'] for r in todo})}")
    log(f"[render] planned {len(selected)} · done {len(done)} · todo {len(todo)} · "
        f"{args.workers} workers x {ENGINE_THREADS} threads")
    if not todo:
        return

    timeout_s = max(120.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    jobs = [(r["unit_key"], render_block_of(srcs[r["unit_key"]]["loc"]), r["best"]["config"],
             str(crops / f"{crop_stem(spec, r['unit_key'])}.jpg"), str(fields_dir), timeout_s)
            for r in todo]

    t0, errors, n = time.time(), [], 0
    with ledger_path(spec).open("a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_render_worker) as ex:
            futs = {ex.submit(render_one, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                uk = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:                     # noqa: BLE001
                    # Recorded in FULL, never head-truncated: the fastest-returning failure
                    # arrives first, so a truncated log describes the wrong failure class.
                    errors.append({"unit_key": uk,
                                   "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                    log(f"[render] ERR {uk}: {str(exc)[:160]}")
                    continue
                res["unit_key_hash"] = crop_stem(spec, uk)
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                n += 1
                if n % 25 == 0 or n <= 3:
                    el = time.time() - t0
                    rate = n / el
                    log(f"[render] {n}/{len(todo)}  {rate:.2f} crop/s -> eta "
                        f"{(len(todo)-n)/rate/60:.0f} min (elapsed {el/60:.0f} min)")
    log(f"[render] done: {n} crops in {(time.time()-t0)/60:.1f} min, {len(errors)} errors")

    ep = spec.batch_dir / "_render_errors.json"
    prior = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else []
    merged = {e["unit_key"]: e for e in prior}
    merged.update({e["unit_key"]: e for e in errors})
    now_done = set(load_render_ledger(spec))
    merged = {k: v for k, v in merged.items() if k not in now_done}
    ep.write_text(json.dumps(sorted(merged.values(), key=lambda e: e["unit_key"]), indent=1),
                  encoding="utf-8")


# =========================================================================== #
# 5. Write — score the stored crops, suggest, sort good->bad, assemble.
# =========================================================================== #
def render_block(loc, palette) -> dict:
    blk = {
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
        "fractal_type": loc.family, "c_re": loc.c_re, "c_im": loc.c_im,
        "palette": palette, "composition": "center",
        "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
        "filter": LABEL_FILTER, "interior_mode": "black",
    }
    for k, v in loc.params.items():
        blk[k] = v
    return blk


def provenance_block(spec, src, rec, loc, cand, split_side) -> dict:
    cfg = cand["config"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "bucketed_intake_sitting",
        "family": loc.family,
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
        "palette": cand["palette"],
        # THE COLORMAP RECIPE — the crop is a pure function of `render` + this block.
        "params": {
            "palette": cand["palette"], "palette_type": cand["palette_type"],
            "palette_source": cand["palette_source"],
            "reverse": cfg["reverse"], "log_premap": cfg["log_premap"],
            "gamma": cfg["gamma"], "phase": cfg["phase"], "n_cycles": cfg["n_cycles"],
            "transfer": cfg.get("transfer"), "transfer_gamma": cfg.get("transfer_gamma"),
            "interior_color": list(cfg["interior_color"]),
            "eval_filter": cfg.get("filter"),
        },
        "render_mode": "smooth",
        # The REGIME axis the two 2026-08-05 batches introduced. This one is neither of them:
        # the palette pool and per-location param law are `pool_draw`'s, but exactly ONE
        # render per location survives and it is the head's own argmax — which is how the
        # emission bridge selects (`enrich --mode render`), not how a pool draw does.
        "coloring_source": "pool_draw_argmax",
        # bucket / stratum
        "bucket": rec["bucket"],
        "vein": rec["vein"],
        "partition": rec["partition"],
        "intake_source": src["intake_source"],
        "source_tag": src["source_tag"],
        "floor_admit": src["floor_admit"],
        "source_ledger": src["source_ledger"],
        "source_oid": src["source_oid"],
        "source_p_good": src["source_p_good"],
        # screen (SELECTION-ONLY — the stamped score is head_v3, off the stored crop)
        "screen_p_ge3": cand["p_ge3"],
        "screen_pred": cand["pred"],
        "loc_screen_p_ge3": rec["score"],
        "n_screened_candidates": len(rec["candidates"]),
        "screen_path": f"render-one --dump-field at {SCREEN_W}x{SCREEN_H}ss{SCREEN_SS} -> "
                       f"colormap.coarse_field + render_candidates_coarse "
                       f"({cm.SCORE_COARSE_W}x{cm.SCORE_COARSE_H}), SCORING-ONLY",
        "split_side": split_side,
        "split_origin": "sitting_bucket_stratified",
    }


def run_write(spec: SittingSpec, args):
    from classifier.inference import load_scorer

    screen = load_screen(spec)
    if not screen:
        raise SystemExit("[write] no screen records — run `screen` first")
    srcs = {s["unit_key"]: s for s in population()[0]}
    selected, sel_report = select(spec, list(screen.values()), args.target_rows, args.seed)
    sides, n_eval = assign_split(spec, selected)
    done = load_render_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")

    crops = spec.batch_dir / "crops"
    live = [r for r in selected if r["unit_key"] in done]
    scorer = load_scorer(str(WP.HEAD_CKPT))
    log(f"[write] head {WP.HEAD_VERSION} on {scorer.device} · scoring {len(live)} crops")
    paths = [crops / f"{crop_stem(spec, r['unit_key'])}.jpg" for r in live]
    marg = FS._marginals_from_paths(scorer, paths)

    rows = []
    for j, rec in enumerate(live):
        src = srcs[rec["unit_key"]]
        loc = src["loc"]
        cand = rec["best"]
        pred = expected_tier(marg[j])
        rows.append({
            "_unit_key": rec["unit_key"],
            "_crop_stem": crop_stem(spec, rec["unit_key"]),
            "render": render_block(loc, cand["palette"]),
            "provenance": provenance_block(spec, src, rec, loc, cand, sides[rec["unit_key"]]),
            # THE HUMAN SLOT. Null on every row: a suggestion is not a label and the merge
            # refuses to read `suggested_tier` into the sidecar.
            "label": {"score": None, "labeler": None, "labeled_at": None},
            # THE PRE-LABEL, off this row's OWN stored crop through the deploy transform.
            "head_v3": {
                "pred": pred,
                "p_ge2": float(marg[j, 0]), "p_ge3": float(marg[j, 1]),
                "p_ge4": float(marg[j, 2]) if marg.shape[1] > 2 else None,
                "ckpt": WP.HEAD_CKPT_REL, "head_version": WP.HEAD_VERSION,
            },
            "p_ge3": float(marg[j, 1]),                    # flat, what the sheet UI reads
            "pred": pred,                                  # flat, the continuous sort key
            "suggested_tier": tier_from_pred(pred, INTAKE_CUTS),
        })

    # PRESENTATION ORDER — good -> bad by the CONTINUOUS score, descending; ties on the crop
    # stem so the order is a pure function of the file. Sorted on `pred`, not on the suggested
    # tier, so within-tier order is stable and meaningful.
    rows.sort(key=lambda r: (-r["pred"], r["_crop_stem"]))
    for i, r in enumerate(rows):
        r["sheet_order"] = i
        # OPAQUE id: `<prefix><slot>_<hash>`. The slot is presentation position — which this
        # sheet publishes anyway, since it is SORTED — and the hash is a salted digest of the
        # unit key, so the id encodes nothing about the bucket, partition, vein or ledger the
        # row came from and a rebuild reproduces it.
        r["image_id"] = f"{spec.img_prefix}{i:04d}_{r['_crop_stem'][:8]}"
    assert len({r["image_id"] for r in rows}) == len(rows), "opaque ids collided"

    # The crops are named by stem; the served rows are named by opaque id. Link them once,
    # here, by copying each crop to its served name — the UI builds
    # `data/<corpus>/batches/<id>/crops/<image_id>.jpg` client-side and cannot be told
    # otherwise.
    import shutil
    for r in rows:
        src_p = crops / f"{r['_crop_stem']}.jpg"
        dst_p = crops / f"{r['image_id']}.jpg"
        if not dst_p.exists():
            shutil.copyfile(src_p, dst_p)

    route = {r["image_id"]: {"unit_key": r.pop("_unit_key"), "crop_stem": r.pop("_crop_stem")}
             for r in rows}

    spec.batch_dir.mkdir(parents=True, exist_ok=True)
    with (spec.batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (spec.batch_dir / "route.json").write_text(json.dumps(route, indent=1), encoding="utf-8")

    # The screen-vs-keeper agreement: the proxy audited against the thing it proxied for.
    sp = np.array([r["provenance"]["screen_p_ge3"] for r in rows])
    fp = np.array([r["p_ge3"] for r in rows])
    agree = {
        "n": len(rows), "spearman": FS._spearman(sp, fp),
        "mean_abs_delta": float(np.abs(sp - fp).mean()) if len(rows) else None,
        "gate_side_agreement_at_0.9": (float(((sp > 0.9) == (fp > 0.9)).mean())
                                       if len(rows) else None),
        "note": f"screen = a {SCREEN_W}x{SCREEN_H}ss{SCREEN_SS} field area-downsampled to the "
                f"{cm.SCORE_COARSE_W}x{cm.SCORE_COARSE_H} scoring grid (SCORING-ONLY); keeper "
                f"= the stored {LABEL_W}x{LABEL_H} ss{LABEL_SS} crop. The stamped head_v3 IS "
                f"the keeper score; the screen only chose what to render and with which "
                f"palette.",
    }

    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
    # INCOMPLETE is DERIVED from the counts, never a flag: a bounded `--limit` run and a killed
    # run both produce a short batch and only one of them would have set one.
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
            "BUCKETED wallpaper CORRECTION SITTING over the stage-2 admitted intake. One "
            "finished coloured render per location — the head's own argmax palette out of "
            "that location's seeded screen draw — at the shared label-crop pins. Every row "
            "carries the complete render block + colormap recipe (provenance.params: the crop "
            "is a pure function of the two), the bucket/vein/partition it was drawn under, "
            "the v3 pre-label (head_v3 / p_ge3 / pred / suggested_tier) and a stamped "
            "bucket-grouped split_side. label.score is null on every row: the suggestion is "
            "NOT a label and is never merged as one. Presentation is sorted good->bad by pred "
            "(descending), stamped contiguous in sheet_order, with opaque image_ids.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {sel_report['drawn_rows']} drawn rows are present — this batch is "
            f"a BOUNDED or INTERRUPTED run and must not be treated as the full sitting. "
            f"Re-run `render` then `write`.") if incomplete else None,
        "head": {"ckpt": WP.HEAD_CKPT_REL, "version": WP.HEAD_VERSION,
                 "role": "PRE-LABEL only — no gate, floor, threshold or pin is applied or "
                         "moved here",
                 "deploy_transform": "classifier.data.Transform(train=False) "
                                     "(1280x720 -> 384x224 bicubic stretch + normalize)"},
        "suggested_tier_rule": INTAKE_DERIVATION,
        "population": args._population_report,
        "selection_report": sel_report,
        "screen_vs_keeper": agree,
        "sampling_metaparameters": {
            "n_screen_candidates": spec.n_screen,
            "picks_per_loc": 1,
            "pick_rule": "the location's argmax-p_ge3 screened candidate — the row a release "
                         "would actually ship (enrich --mode render selects the same way). "
                         "PALETTE IS NEVER A STRATUM.",
            "target_rows": sel_report["target_rows"],
            "palette_pool": FS.PALETTE_POOL,
            "palette_draw": "sample_location.gen0_palettes(sampler, 120) — the deployed "
                            "GEN0_SOURCE_WEIGHTS composition; per-location seeded subsample "
                            "of n_screen, params via query_sampler.sample_candidate",
            "maxiter_policy": loc_mod.maxiter_policy_token(),
            "seed": sel_report["seed"], "split_seed": spec.split_seed,
            "eval_frac": spec.eval_frac, "id_salt": spec.id_salt,
        },
        "split_summary": {
            "eval_rows": sum(1 for r in rows if r["provenance"]["split_side"] == "eval"),
            "train_rows": sum(1 for r in rows if r["provenance"]["split_side"] == "train"),
            "planned_eval_locations": n_eval,
            "rule": "location-grouped (one location = one row), seeded on split_seed, "
                    "stratified BY BUCKET so each named population appears on both sides",
        },
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "jpg_quality": JPG_Q,
            "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(tools/wallpaper/label_crop.py — the locked label-crop pins)",
        },
        "realized": {
            "rows_by_bucket": dict(Counter(r["provenance"]["bucket"] for r in rows)),
            "rows_by_partition": dict(sorted(Counter(
                r["provenance"]["partition"] for r in rows).items())),
            "rows_by_vein": dict(sorted(Counter(r["provenance"]["vein"] for r in rows).items())),
            "rows_by_split": dict(Counter(r["provenance"]["split_side"] for r in rows)),
            "suggested_tier_hist": dict(sorted(Counter(
                r["suggested_tier"] for r in rows).items())),
            "distinct_palettes": len({r["render"]["palette"] for r in rows}),
            "palette_top10": dict(Counter(r["render"]["palette"] for r in rows).most_common(10)),
            "p_ge3": {"min": round(min(r["p_ge3"] for r in rows), 4),
                      "max": round(max(r["p_ge3"] for r in rows), 4)},
            "above_retired_release_floor": sum(
                1 for r in rows if F.WALLPAPER_RELEASE.annotates(r["p_ge3"])),
        },
        "presentation": {
            "order": "sheet_order — DESCENDING pred (good -> bad), ties on the crop stem",
            "sorted_on": "the continuous head score (pred), NOT the suggested tier",
            "contiguous": True,
            "image_id": "OPAQUE `<prefix><slot>_<hash8>` — slot is presentation position "
                        "(published anyway, the sheet is sorted); the hash is a salted digest "
                        "of the unit key, so the id encodes no bucket, partition, vein or "
                        "ledger. route.json maps it back.",
        },
        "labels_export": spec.labels_export,
        "labeling": {
            "ui": spec.ui_url,
            "mode": "correction — every row shows its suggested tier PREFILLED; Enter "
                    "confirms, 1-4 override. Only rows Matt acts on are exported; an "
                    "unreviewed suggestion never leaves the page as a label.",
            "bulk": "accept all remaining, and accept all BELOW THIS ROW (the positional "
                    "sweep a sorted sheet makes natural) — both behind a confirm",
            "blind_rows": 0,
            "calibration_duplicates": 0,
            "merge": f"uv run python tools/wallpaper/merge_sitting.py "
                     f"--corpus wallpaper_corpus --batch {spec.batch_id} "
                     f"--scores {spec.labels_export} --apply",
        },
        "render_failures": errors,
        "run_status": {
            "drawn_rows": sel_report["drawn_rows"],
            "rendered_rows": len(rows),
            "n_failures": len(errors),
            "n_unaccounted": len(unaccounted),
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
def print_composition(rep):
    log("-" * 96)
    log(f"{'bucket':<22}{'target':>8}{'avail':>8}{'drawn':>8}   rule")
    for b in rep["per_bucket"]:
        log(f"{b['bucket']:<22}{str(b['target']):>8}{str(b['available']):>8}"
            f"{b['drawn']:>8}   {b['rule'][:60]}")
    log("-" * 96)
    log(f"drawn {rep['drawn_rows']} / target {rep['target_rows']}  ·  eligible "
        f"{rep['eligible']} of {rep['screened']} screened "
        f"({rep['screen_failures']} screen failures)")
    log(f"by partition: {rep['drawn_by_partition']}")
    log(f"by vein:      {rep['drawn_by_vein']}")
    if rep["partitions_below_coverage_floor"]:
        log(f"BELOW the {rep['coverage_floor']}-row coverage floor (supply-bound): "
            f"{rep['partitions_below_coverage_floor']}")
    log("-" * 96)


def print_summary(spec, batch):
    r = batch["realized"]
    log("\n" + "=" * 96)
    log(f"WALLPAPER CORRECTION SITTING — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 96)
    log(f"rows {batch['n_rows']} / drawn {batch['run_status']['drawn_rows']}  ·  failures "
        f"{batch['run_status']['n_failures']}")
    log(f"by bucket:    {r['rows_by_bucket']}")
    log(f"by partition: {r['rows_by_partition']}")
    log(f"by vein:      {r['rows_by_vein']}")
    log(f"by split:     {r['rows_by_split']}")
    log(f"suggested:    {r['suggested_tier_hist']}  (cuts {batch['suggested_tier_rule']['cuts']})")
    log(f"palettes:     {r['distinct_palettes']} distinct  ·  p_ge3 {r['p_ge3']}  ·  "
        f"above the retired 0.90 floor {r['above_retired_release_floor']}")
    a = batch["screen_vs_keeper"]
    log(f"screen-vs-keeper: spearman {a['spearman']}  gate-side agreement "
        f"{a['gate_side_agreement_at_0.9']}  mean|delta| {a['mean_abs_delta']}")
    log(f"-> {spec.batch_dir}")
    log(f"-> serve: uv run python tools/viz/serve.py   then")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Bucketed wallpaper correction sitting builder.")
    ap.add_argument("stage", choices=("estimate", "screen", "select", "render", "write"))
    ap.add_argument("--sitting", default="v2", choices=sorted(SITTINGS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--target-rows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap units this run (SPREAD across the plan, not a prefix). A short "
                         "batch STAMPS itself sheet_incomplete at write time.")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="concurrent engine PROCESSES — the project cap is 4, do not raise")
    ap.add_argument("--dry-run", action="store_true", help="render: composition only")
    ap.add_argument("--unit-timeout", type=float, default=900.0,
                    help="per-unit backstop, clamped to a quarter of --wall-budget-s")
    ap.add_argument("--wall-budget-s", type=float, default=4 * 3600.0)
    args = ap.parse_args(argv)

    if args.workers > WORKERS:
        raise SystemExit(f"[sitting] --workers {args.workers} exceeds the project process cap "
                         f"of {WORKERS} (CLAUDE.md). In-process threads are the knob.")
    spec = SITTINGS[args.sitting]
    spec.work.mkdir(parents=True, exist_ok=True)
    prio = cc.set_below_normal_priority()
    log(f"[sitting] {spec.batch_id} · priority {prio} · {args.workers} workers x "
        f"{ENGINE_THREADS} rayon threads")

    if args.stage == "estimate":
        _srcs, pop = population()
        log(json.dumps(pop, indent=2))
        return 0
    if args.stage == "screen":
        run_screen(spec, args)
        return 0
    if args.stage == "select":
        screen = load_screen(spec)
        if not screen:
            raise SystemExit("[select] no screen records — run `screen` first")
        _sel, rep = select(spec, list(screen.values()), args.target_rows, args.seed)
        print_composition(rep)
        (spec.work / "selection_report.json").write_text(json.dumps(rep, indent=2),
                                                         encoding="utf-8")
        return 0
    if args.stage == "render":
        run_render(spec, args)
        return 0
    if args.stage == "write":
        args._population_report = population()[1]
        run_write(spec, args)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
