r"""build_mining_sheet.py — the render-mode (mining) CORRECTION SHEET, pre-labeled by v1.

The mining head's label corpus, rebuilt from scratch in the durable pattern after the
original was lost (`data/render_mode_corpus/` was the ONE label corpus with no `.gitignore`
negation; its 1500 human tiers survive in `labels/render_mode_{pilot,scale}_v1.json` and are
orphaned, because every id they key on was defined in an untracked manifest —
`scratch/stage2_label_audit/report.md`). Every design decision below is the July design
REUSED, not reinvented; what changed is where the bytes land and what a row carries.

WHAT MAKES THIS ONE SURVIVABLE (the difference, in one line): the label, the complete render
block, the mode, the mode params, the colormap recipe and the split side are in the SAME
tracked row. A crop is a pure function of that row; a human tier is a pure function of
nothing, so it is the only thing in the pipeline that must never depend on an untracked file.

POPULATION — `data/render_mode_corpus/gate_passers_v3.json` (401 rows / 112 locations),
regenerated and count-verified by `build_gate_passers.py`. One row per (location, palette)
the deployed wallpaper head scores above the emission gate; the render-mode batch INHERITS
that approved palette + colour params verbatim and varies only the MODE, which is exactly
what the head is asked to judge.

ROSTER — all 15 registered non-`smooth` modes (`mining_roster.py`). The July scale sampler
dropped two at sample time and the trainer dropped three at train time; only the second drop
belongs downstream, because a trainer can be re-run and a corpus cannot be un-narrowed.

DRAW, per mode: distinct locations, family-apportioned through `apportion.deal_round_robin`
(so the eight families are balanced-or-drained rather than proportional — mandelbrot supplies
63 of 112 locations and would otherwise own the sheet), one gate-passing palette per drawn
location. `direct_*` is palette-INDIFFERENT by construction, so it is palette-deduped and
spreads a permuted 3x3 opacity x threshold cell per location instead.

SPLIT — `split_units.build_split`: union-find over Julia-seed == parent-plane point,
family-stratified, `EVAL_FRAC=0.40`, seed 0. Stamped in-row as `provenance.split_side`.

PRE-LABEL — every row scored by the pinned mining head v1 (`mining_gate.MiningScorer`, fp32,
marginal `p_ge3`, deploy transform) off its OWN stored crop, and given a `suggested_tier`
from `suggest_tier_mining` (per-batch quantiles at the surviving 1500-label tier prior — the
caveat is in that module and is copied into `batch.json`). A SUGGESTION IS NOT A LABEL:
`label.score` is null on every row and the merge refuses to read the suggestion.

PRESENTATION — sorted good -> bad by the head's CONTINUOUS score (descending `pred`), not by
the suggested tier, so within-tier order is stable and meaningful; `sheet_order` is stamped
contiguous 0..N-1 so the order is auditable later
(`prompts/mining_sheet_addendum_sorted.md`).

    uv run python -u tools/mining/build_mining_sheet.py plan            # dry composition
    uv run python -u tools/mining/build_mining_sheet.py render --limit 6   # bounded E2E
    uv run python -u tools/mining/build_mining_sheet.py write           # (stamps INCOMPLETE)
    uv run python -u tools/mining/build_mining_sheet.py render > scratch/mining_sheet/render.log 2>&1
    uv run python -u tools/mining/build_mining_sheet.py write
"""
from __future__ import annotations

import argparse
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
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                                        # noqa: E402  THE two draw rules
import corpus_common as cc                              # noqa: E402  engine launch defaults
import location as loc_mod                              # noqa: E402
from tools import paths                                 # noqa: E402
from tools.mining import mining_roster as MR            # noqa: E402
from tools.mining import mining_pins as MP              # noqa: E402
from tools.mining import suggest_tier_mining as ST      # noqa: E402
from tools.mining.split_units import build_split, units_are_disjoint  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                        # noqa: BLE001
    pass

EXE = str(ROOT / "target" / "release" / "fractal-generator.exe")
POOL_CMAPS = str(ROOT / "data" / "palettes" / "pool_colormaps.json")
GATE_PASSERS = ROOT / "data" / "render_mode_corpus" / "gate_passers_v3.json"
CORPUS = ROOT / "data" / "render_mode_corpus"

# --- render pins ------------------------------------------------------------- #
# The July render-mode batches' own pins, kept EXACTLY: the mining head was trained on crops
# at these settings (`train_mining_head` config `src_dims: [1280, 720]`), so a fresh corpus
# rendered at anything else would ask v1 to pre-label a distribution it has never seen.
W, H, SS, FILT, JPG_Q = 1280, 720, 2, "lanczos3", 95

# Project-wide cap: 4 concurrent `fractal-generator.exe`, 3 rayon threads each (12 cores).
# This is the PROCESS cap from CLAUDE.md and it is not a tuning knob — do not raise it.
WORKERS = 4
ENGINE_THREADS = 3


# =========================================================================== #
# The sitting spec. A frozen dataclass from the start even though there is one of them
# (CLAUDE.md, "Writing a builder for one instance"): `sitting_cutter` held exactly these
# fields at module scope and a second sitting needed a refactor before it could be built.
# =========================================================================== #
@dataclass(frozen=True)
class SheetSpec:
    key: str
    batch_id: str
    generator_version: str          # also names the tracked labels sidecar
    img_prefix: str
    target_rows: int                # the prompt's ~800-1000 band
    draw_seed: int
    split_seed: int
    eval_frac: float

    @property
    def batch_dir(self) -> Path:
        return CORPUS / "batches" / self.batch_id

    @property
    def labels_export(self) -> str:
        return f"labels/{self.generator_version}.json"

    @property
    def ui_url(self) -> str:
        return (f"tools/viz/wallpaper_label.html?corpus=render_mode_corpus&tiers=3"
                f"&order=file&batch={self.batch_id}")


SHEETS = {
    "v1": SheetSpec(
        key="v1",
        batch_id="2026-08-06_render_mode_fresh_sheet_v1",
        generator_version="render_mode_fresh_sheet_v1",
        img_prefix="rmf",
        target_rows=960,            # 15 modes x 64 locations
        draw_seed=20260806,
        split_seed=0,               # the July split seed — the design is reused, not re-rolled
        eval_frac=0.40,
    ),
}


def log(msg):
    print(msg, flush=True)


# =========================================================================== #
# 1. Population.
# =========================================================================== #

def load_gate_passers() -> tuple:
    """`(rows, meta)` from the durable gate-passer artifact.

    A hard failure with the rebuild command when it is absent — never an empty population.
    An absence-tolerant load here would build a zero-row corpus that reads as a completed
    run (`verification_practice.md` §2)."""
    if not GATE_PASSERS.exists():
        raise SystemExit(
            f"[mining-sheet] population absent: {GATE_PASSERS.relative_to(ROOT)}\n"
            f"Rebuild it (it is deterministic and count-verified):\n"
            f"    uv run python tools/mining/build_gate_passers.py")
    doc = json.loads(GATE_PASSERS.read_text(encoding="utf-8"))
    return doc["rows"], doc["meta"]


def index_population(rows):
    """`(loc_rows, loc_fam, fam_locs, loc_rep)` — the four views the draw needs."""
    loc_rows, loc_fam, fam_locs, loc_rep = defaultdict(list), {}, defaultdict(set), {}
    for i, r in enumerate(rows):
        k = r["location_key"]
        loc_rows[k].append(i)
        loc_fam[k] = r["family"]
        fam_locs[r["family"]].add(k)
        loc_rep.setdefault(k, r)
    return loc_rows, loc_fam, fam_locs, loc_rep


# =========================================================================== #
# 2. Plan — a PURE function of (artifact, spec). No intermediate file.
# =========================================================================== #

def plan(spec: SheetSpec):
    """`(entries, report)`. Deterministic: the same artifact + spec always give the same
    plan, which is what lets `render` and `write` each recompute it instead of reading a
    scratch file that a `rm -r scratch/*` could take out from under a half-finished run."""
    rows, pop_meta = load_gate_passers()
    loc_rows, loc_fam, fam_locs, loc_rep = index_population(rows)
    n_loc = len(loc_rows)
    fam_avail = dict(sorted({f: len(s) for f, s in fam_locs.items()}.items()))

    side, split_meta = build_split(loc_rep, seed=spec.split_seed, eval_frac=spec.eval_frac)
    ok, why = units_are_disjoint(side, loc_rep)
    if not ok:
        raise SystemExit(f"[mining-sheet] split units span both sides: {why}")

    # Rows per mode: equal-ish over the roster, capped by how many distinct locations exist.
    # `deal_round_robin` rather than a local divmod — the rule has ONE owner (apportion.py).
    per_mode = apportion.deal_round_robin({m: n_loc for m in MR.MODES}, spec.target_rows)

    rng = np.random.default_rng(spec.draw_seed)
    entries, mode_report = [], []
    for mode in MR.MODES:
        kind = MR.MODE_KIND[mode]
        target = per_mode[mode]
        alloc = apportion.deal_round_robin(fam_avail, target)
        bal_ok, bal_why = apportion.cells_balanced(alloc, fam_avail)

        drawn = []
        for fam in sorted(fam_avail):
            locs_f = sorted(fam_locs[fam])
            idx = rng.permutation(len(locs_f))[:alloc[fam]]
            drawn += [locs_f[int(j)] for j in idx]
        drawn.sort()                       # deterministic id assignment order

        # direct-trap: a permuted cycle over the 9 grid cells, so every cell is used about
        # equally within the mode and no location's cell correlates with its family.
        if kind == "direct":
            cell_perm = [MR.DIRECT_GRID[int(j)]
                         for j in rng.permutation(len(MR.DIRECT_GRID))]

        for i, k in enumerate(drawn):
            src = rows[int(rng.choice(loc_rows[k]))]     # one gate-passing palette row
            mode_params = {}
            if kind == "direct":
                op, th = cell_perm[i % len(cell_perm)]
                mode_params = {"direct_opacity": op, "direct_threshold": th}
            entries.append({
                "mode": mode, "kind": kind,
                "location_key": k, "family": loc_fam[k],
                "split_side": side[k],
                "src_image_id": src["image_id"],
                "palette": src["palette"],
                "src_p_ge3": src["p_ge3"],
                "color_params": src["params"],
                "render": src["render"],
                "mode_params": mode_params,
            })
        mode_report.append({
            "mode": mode, "kind": kind, "target": target, "drawn": len(drawn),
            "family_alloc": alloc, "family_balanced": bal_ok, "family_balance": bal_why,
            "eval_rows": sum(1 for k in drawn if side[k] == "eval"),
        })

    # ids in plan order (mode-major, then location key) — STABLE across a resume, and
    # deliberately NOT the presentation order (which is derived from scores after render).
    entries.sort(key=lambda e: (MR.MODES.index(e["mode"]), e["location_key"]))
    for i, e in enumerate(entries):
        e["image_id"] = f"{spec.img_prefix}_{i:04d}"

    report = {
        "population": {
            "artifact": str(GATE_PASSERS.relative_to(ROOT)).replace("\\", "/"),
            "rows": len(rows), "locations": n_loc,
            "family_locations": fam_avail,
            "head": pop_meta["head"], "gate": pop_meta["gate"],
            "source_batch": pop_meta["source_batch"],
        },
        "roster": {
            "n_modes": len(MR.MODES), "modes": list(MR.MODES),
            "kinds": dict(Counter(MR.MODE_KIND[m] for m in MR.MODES)),
            "why_15": "all registered non-smooth modes. The July scale sampler dropped "
                      f"{list(MR.SCALE_SAMPLER_DROPPED_2026_07)} at SAMPLE time and the "
                      f"trainer dropped {list(MR.TRAINER_DROPPED_V1)} at TRAIN time; only "
                      "the trainer's drop belongs downstream (a trainer re-runs, a corpus "
                      "does not un-narrow). Decision 2026-08-06, Matt.",
            "direct_grid": [list(c) for c in MR.DIRECT_GRID],
            "rolloff": {m: MR.rolloff_token(m) for m in MR.MODES
                        if MR.rolloff_token(m) != "none"},
        },
        "allocation": {
            "target_rows": spec.target_rows, "planned_rows": len(entries),
            "rule": "equal-ish per mode via apportion.deal_round_robin capped at the "
                    "distinct-location supply; within a mode, families via the same rule "
                    "(balanced-or-drained, NOT proportional — mandelbrot is 63/112 and "
                    "would otherwise own the sheet)",
            "per_mode": mode_report,
            "family_rows": dict(Counter(e["family"] for e in entries).most_common()),
            "distinct_locations_used": len({e["location_key"] for e in entries}),
            "distinct_palettes_used": len({e["palette"] for e in entries}),
        },
        "split": {
            **split_meta,
            "eval_rows": sum(1 for e in entries if e["split_side"] == "eval"),
            "train_rows": sum(1 for e in entries if e["split_side"] == "train"),
            "units_disjoint": why,
        },
        "seeds": {"draw_seed": spec.draw_seed, "split_seed": spec.split_seed},
    }
    return entries, report


# =========================================================================== #
# 3. Render — the July render paths, one per kind.
# =========================================================================== #

_LIB = _CM = _LOC = None


def _init_worker():
    global _LIB, _CM, _LOC
    os.environ["RAYON_NUM_THREADS"] = str(ENGINE_THREADS)
    import colormap as cm
    import location as _lm
    import query_sampler as qs
    _CM, _LOC = cm, _lm
    _LIB = qs.load_pool_library()


def _locflags(loc):
    return _LOC.render_one_flags(loc) + ["--cx", loc.cx, "--cy", loc.cy,
                                         "--fw", loc.fw, "--maxiter", str(loc.maxiter)]


def _run(cmd, timeout_s: float):
    env = dict(os.environ, RAYON_NUM_THREADS=str(ENGINE_THREADS))
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env,
                       timeout=timeout_s, creationflags=cc.default_creationflags())
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-700:])


def _render_pure(entry, loc, crop_path, fields_dir, timeout_s, geom=None):
    """dump-field -> python recolor with the FULL approved colour params. Bit-faithful:
    `transfer=grad` survives, which the Rust coloring path cannot express.

    `geom` is `(w, h, ss)`, defaulting to the FROZEN corpus pins. It exists so a SCORING-ONLY
    screen can drive this exact render path at a smaller geometry instead of a second copy of
    it (`build_mining_correction.py`); a keeper crop never passes it."""
    cm = _CM
    w, h, ss = geom or (W, H, SS)
    binp = fields_dir / f"{entry['image_id']}.bin"
    try:
        _run([EXE, "render-one"] + _locflags(loc)
             + ["--width", str(w), "--height", str(h), "--supersample", str(ss),
                "--coloring", json.dumps(MR.spec_for(entry["mode"])),
                "--dump-field", str(binp)], timeout_s)
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        p = entry["color_params"]
        ptype = _LIB.palette_type(entry["palette"])
        cfg = cm.CandidateConfig(
            palette=entry["palette"], location=fld.location,
            eval_width=ow, eval_height=oh,
            reverse=bool(p["reverse"]), log_premap=p["log_premap"],
            gamma=float(p["gamma"]),
            phase=(p["phase"] if ptype == "cyclic" else 0.0),
            n_cycles=(p["n_cycles"] if ptype == "cyclic" else 1),
            transfer=p["transfer"], transfer_gamma=float(p["transfer_gamma"]),
            filter=FILT)
        prep = cm.stretch_field(fld)
        prof = cm.gradient_transfer_profile(fld, prep) if p["transfer"] == "grad" else None
        img = cm.render_candidate(fld, cfg, _LIB, prep=prep, profile=prof)
        from PIL import Image
        Image.fromarray(img).save(crop_path, quality=JPG_Q)
    finally:
        binp.unlink(missing_ok=True)
        binp.with_suffix(".json").unlink(missing_ok=True)
    return False


def _render_rust(entry, loc, crop_path, fields_dir, timeout_s, geom=None):
    """`render-one --coloring <spec>`. Honors reverse/log_premap/gamma/cycles/offset but
    CANNOT express `transfer=grad`, so that knob is dropped and the row stamps it.

    `geom` — see `_render_pure`."""
    from PIL import Image

    w, h, ss = geom or (W, H, SS)
    mode = entry["mode"]
    spec = MR.spec_for(mode, entry.get("mode_params"))
    p = entry["color_params"]
    ptype = _LIB.palette_type(entry["palette"])
    spec["transform"] = "log" if p["log_premap"] == "log" else "linear"
    spec["gamma"] = float(p["gamma"])
    spec["reverse"] = bool(p["reverse"])
    if ptype == "cyclic":
        spec["palette_cycles"] = float(p["n_cycles"])
        spec["palette_offset"] = float(p["phase"])
    rname, rstrength = MR.rolloff_for(mode)
    if rname != "none":
        spec["rolloff"] = rname
        spec["rolloff_strength"] = rstrength
    transfer_dropped = p["transfer"] == "grad"
    tmp_png = fields_dir / f"{entry['image_id']}.png"
    try:
        _run([EXE, "render-one"] + _locflags(loc)
             + ["--width", str(w), "--height", str(h), "--supersample", str(ss),
                "--filter", FILT, "--palette", entry["palette"],
                "--colormaps", POOL_CMAPS, "--coloring", json.dumps(spec),
                "--out", str(tmp_png)], timeout_s)
        with Image.open(tmp_png) as im:
            im.convert("RGB").save(crop_path, quality=JPG_Q)
    finally:
        tmp_png.unlink(missing_ok=True)
    return transfer_dropped


def render_one(job):
    """One raster. `job` carries the batch paths so the worker holds no module state beyond
    the library/colormap globals its initializer loads.

    A 5th job element is an optional `(w, h, ss)` geometry (see `_render_pure`); absent, the
    frozen corpus pins apply, which is what every keeper crop uses."""
    entry, crops_s, fields_s, timeout_s = job[:4]
    geom = job[4] if len(job) > 4 else None
    t0 = time.time()
    crops, fields_dir = Path(crops_s), Path(fields_s)
    loc = _LOC.from_render_block(entry["render"])
    crop_path = crops / f"{entry['image_id']}.jpg"
    if entry["kind"] == "pure":
        transfer_dropped = _render_pure(entry, loc, crop_path, fields_dir, timeout_s, geom)
    else:
        transfer_dropped = _render_rust(entry, loc, crop_path, fields_dir, timeout_s, geom)
    return {"image_id": entry["image_id"], "mode": entry["mode"], "family": entry["family"],
            "transfer_dropped": transfer_dropped, "secs": time.time() - t0}


# =========================================================================== #
# 4. Rows.
# =========================================================================== #

def render_block(entry, loc):
    blk = {
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
        "fractal_type": loc.family,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "palette": entry["palette"],
        "composition": "center",
        "width": W, "height": H, "ss": SS, "filter": FILT, "interior_mode": "black",
        "render_mode": entry["mode"],
        "mode_params": dict(entry.get("mode_params", {})),
        "rolloff": MR.rolloff_token(entry["mode"]),
    }
    for k, v in loc.params.items():
        blk[k] = v
    return blk


def provenance_block(spec: SheetSpec, entry, transfer_dropped: bool):
    p = entry["color_params"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "render_mode_fresh_sheet",
        "family": entry["family"],
        "location_key": entry["location_key"],
        "render_mode": entry["mode"],
        "mode_kind": entry["kind"],
        "mode_params": dict(entry.get("mode_params", {})),
        "rolloff": MR.rolloff_token(entry["mode"]),
        # THE COLORMAP RECIPE — inherited verbatim from the gate-passing wallpaper row.
        # The crop is a pure function of `render` + this block; nothing else is needed.
        "color_params": {
            "palette": entry["palette"],
            "palette_type": p.get("palette_type"),
            "palette_source": p.get("palette_source"),
            "reverse": p["reverse"], "log_premap": p["log_premap"], "gamma": p["gamma"],
            "phase": p["phase"], "n_cycles": p["n_cycles"],
            "transfer": p["transfer"], "transfer_gamma": p["transfer_gamma"],
            "interior_color": list(p.get("interior_color", [0.0, 0.0, 0.0])),
        },
        "transfer_dropped": transfer_dropped,
        "split_side": entry["split_side"],
        "split_origin": "julia_parent_unionfind",
        # where the (location, palette, colour params) came from
        "source": {
            "batch_id": "2026-07-09_wallpaper_headbatch_dramatic_v1",
            "image_id": entry["src_image_id"],
            "p_ge3": entry["src_p_ge3"],
            "gate": "wallpaper_head_v3 p_ge3 > 0.90 "
                    "(data/render_mode_corpus/gate_passers_v3.json)",
        },
    }


# =========================================================================== #
# 5. Render driver.
# =========================================================================== #

def ledger_path(spec: SheetSpec) -> Path:
    return spec.batch_dir / "_progress_ledger.jsonl"


def load_ledger(spec: SheetSpec) -> dict:
    """`{image_id: record}` for units with a ledger row AND an on-disk crop. Both, because a
    ledger row whose crop was deleted is a row that would land in `images.jsonl` pointing at
    nothing."""
    done, p, crops = {}, ledger_path(spec), spec.batch_dir / "crops"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if (crops / f"{rec['image_id']}.jpg").exists():
                    done[rec["image_id"]] = rec
    return done


def run_render(spec: SheetSpec, args):
    entries, rep = plan(spec)
    print_composition(rep)
    if args.dry_run:
        return

    crops = spec.batch_dir / "crops"
    fields = spec.batch_dir / "_fields"
    crops.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)

    done = load_ledger(spec)
    todo = [e for e in entries if e["image_id"] not in done]
    if args.limit:
        # SPREAD, not the first N. The plan is mode-major, so `todo[:8]` renders eight `tia`
        # rows and exercises only the pure path — a bounded run that never touches the
        # composite/direct branches is not a bounded end-to-end, it is a bounded prefix, and
        # the first execution of the untested half would be the 960-row production run.
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
        log(f"[render] --limit {args.limit}: SPREAD across the plan -> modes "
            f"{sorted({e['mode'] for e in todo})}")
    log(f"[render] planned {len(entries)}  ·  already done {len(done)}  ·  todo {len(todo)}"
        f"  ·  {WORKERS} workers x {ENGINE_THREADS} threads")
    if not todo:
        return

    # A per-unit backstop clamped to the run's own budget: a timeout longer than the job
    # cannot bound it (CLAUDE.md, "a backstop longer than the job's budget is not a
    # backstop"). 25x the observed-typical render, floored so a 6-row smoke is not tripping.
    timeout_s = max(90.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))

    errors, times, n = [], [], 0
    t0 = time.time()
    with open(ledger_path(spec), "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
            futs = {ex.submit(render_one, (e, str(crops), str(fields), timeout_s)): e
                    for e in todo}
            for fut in as_completed(futs):
                e = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:                          # noqa: BLE001
                    # Recorded in FULL, never truncated to the first N: the fastest-returning
                    # failure arrives first, so a truncated error log describes the wrong
                    # failure class (CLAUDE.md, "Four rules").
                    errors.append({"image_id": e["image_id"], "mode": e["mode"],
                                   "kind": e["kind"], "family": e["family"],
                                   "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                    log(f"[render] ERR {e['image_id']} {e['mode']}: {str(exc)[:180]}")
                    continue
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                times.append(res["secs"])
                n += 1
                if n % 20 == 0 or n <= 3:
                    recent = float(np.mean(times[-40:]))
                    # Reprojected from RECENT observed throughput, not the run-to-date mean.
                    eta = recent * (len(todo) - n) / max(args.workers, 1) / 60
                    log(f"[render] {n}/{len(todo)}  {res['mode']:32s} {res['secs']:5.1f}s  "
                        f"recent {recent:.1f}s/render -> eta {eta:.0f} min "
                        f"(elapsed {(time.time()-t0)/60:.0f} min)")

    log(f"[render] done: {n} rendered in {(time.time()-t0)/60:.1f} min, {len(errors)} errors")
    # ACCUMULATE across runs, then drop anything now on disk. A plain overwrite would erase
    # run 1's failures the moment run 2 finished, including failures run 2 never retried
    # (a `--limit` resume touches a subset) — so the file would report "no errors" for a
    # batch that has holes.
    ep = spec.batch_dir / "_render_errors.json"
    prior = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else []
    merged = {e["image_id"]: e for e in prior}
    merged.update({e["image_id"]: e for e in errors})
    now_done = set(load_ledger(spec))
    merged = {k: v for k, v in merged.items() if k not in now_done}
    ep.write_text(json.dumps(sorted(merged.values(), key=lambda e: e["image_id"]), indent=1),
                  encoding="utf-8")
    if merged:
        log(f"[render] outstanding failures by mode (ALL {len(merged)}, never a head): "
            f"{dict(Counter(e['mode'] for e in merged.values()))}")


# =========================================================================== #
# 6. Write — score, suggest, sort, assemble.
# =========================================================================== #

def run_write(spec: SheetSpec, args):
    from tools.mining.mining_gate import MiningScorer

    entries, rep = plan(spec)
    by_id = {e["image_id"]: e for e in entries}
    done = load_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")

    crops = spec.batch_dir / "crops"
    ids = [e["image_id"] for e in entries if e["image_id"] in done]

    scorer = MiningScorer(model_path=MP.ACTIVE_MINING_CKPT)
    log(f"[write] mining head {MP.HEAD_VERSION} (K={scorer.k}) on {scorer.device} "
        f"· scoring {len(ids)} crops")
    if scorer.k != ST.K_TIERS:
        raise SystemExit(
            f"[write] head K={scorer.k} but the suggestion rule is written for "
            f"K={ST.K_TIERS}. A tier scale that moved silently is exactly the "
            f"'equality test against a class ceiling' failure — fix the rule, do not coerce.")
    scores = scorer.score_paths([crops / f"{i}.jpg" for i in ids])
    # expected_tier = 1 + Sum_k sigma(logit_k); MiningScore.score IS that sum.
    pred = [ST.expected_tier([s.p_ge2, s.p_ge3]) for s in scores]

    prior = ST.tier_prior()
    cuts = ST.cuts_from_prior(pred, prior)
    tiers = ST.suggest_all(pred, cuts)

    rows = []
    for j, iid in enumerate(ids):
        e = by_id[iid]
        loc = loc_mod.from_render_block(e["render"])
        s = scores[j]
        rows.append({
            "image_id": iid,
            "render": render_block(e, loc),
            "provenance": provenance_block(spec, e, bool(done[iid]["transfer_dropped"])),
            # THE HUMAN SLOT. Null on every row: a suggestion is not a label, and the merge
            # refuses to read `suggested_tier` into the sidecar.
            "label": {"score": None, "labeler": None, "labeled_at": None},
            # THE PRE-LABEL, off this row's OWN stored crop through the deploy transform.
            "head_mining_v1": {
                "pred": pred[j],
                "p_ge2": s.p_ge2, "p_ge3": s.p_ge3, "score": s.score,
                "ckpt": MP.ACTIVE_MINING_CKPT, "head_version": MP.HEAD_VERSION,
                "gate_version": MP.MINING_GATE_VERSION,
                "would_pass_gate": bool(s.passed),
                "gate_threshold": scorer.threshold,
            },
            "p_ge3": s.p_ge3,                 # flat, what the sheet UI reads
            "pred": pred[j],                  # flat, the continuous sort key
            "suggested_tier": int(tiers[j]),
        })

    # PRESENTATION ORDER — good -> bad by the CONTINUOUS score, descending. Sorted on `pred`,
    # not on `suggested_tier`, so within-tier order is stable and meaningful; ties break on
    # image_id so the order is a pure function of the file.
    rows.sort(key=lambda r: (-r["pred"], r["image_id"]))
    for i, r in enumerate(rows):
        r["sheet_order"] = i
    assert [r["sheet_order"] for r in rows] == list(range(len(rows))), "sheet_order not contiguous"

    spec.batch_dir.mkdir(parents=True, exist_ok=True)
    with (spec.batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # INCOMPLETE is DERIVED from the counts, not from a flag: a bounded `--limit` run and a
    # killed run both produce a short batch, and only one of them would have set a flag.
    incomplete = len(rows) < rep["allocation"]["planned_rows"]
    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
    # Planned rows that are neither rendered nor recorded as a failure. Without this a
    # bounded or killed run is indistinguishable from a complete one with a smaller plan —
    # the count says "short" but nothing says WHICH rows and why they are missing.
    accounted = set(done) | {e["image_id"] for e in errors}
    unaccounted = sorted(e["image_id"] for e in entries if e["image_id"] not in accounted)
    if unaccounted:
        log(f"[write] {len(unaccounted)} planned rows neither rendered nor failed "
            f"(bounded or interrupted run), e.g. {unaccounted[:5]}")

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_id": spec.batch_id,
        "generator_version": spec.generator_version,
        "labeler": None,
        "n_rows": len(rows),
        "schema_note":
            "Render-mode (mining) CORRECTION SHEET, rebuilt after the loss of the original "
            "corpus. Every row carries the human label slot AND the complete render block, "
            "the mode + mode_params, the inherited colormap recipe "
            "(provenance.color_params — the crop is a pure function of the two), the "
            "location-grouped split_side, and the mining-v1 pre-label "
            "(head_mining_v1 / p_ge3 / pred / suggested_tier), in ONE tracked row. "
            "label.score is null on every row: the suggestion is NOT a label and is never "
            "merged as one. Presentation is sorted good->bad by pred (descending), stamped "
            "contiguous in sheet_order.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {rep['allocation']['planned_rows']} planned rows are present — "
            f"this batch is a BOUNDED or INTERRUPTED run and must not be treated as the "
            f"full sheet. Re-run `render` then `write` to complete it."
        ) if incomplete else None,
        "head": {
            "ckpt": MP.ACTIVE_MINING_CKPT, "version": MP.HEAD_VERSION,
            "gate_version": MP.MINING_GATE_VERSION,
            "role": "PRE-LABEL only — no gate, floor, threshold or pin is applied or moved "
                    "here; would_pass_gate is stamped for reference and gates nothing",
            "scorer": "tools/mining/mining_gate.MiningScorer (fp32, no autocast; marginal "
                      "p_ge = cumprod(sigmoid), NEVER the CORN conditional)",
            "deploy_transform": "classifier.data.Transform(train=False) — 384x224 bicubic "
                                "stretch + the checkpoint's own mean/std",
        },
        "suggested_tier_rule": ST.derivation(cuts, prior, pred, MP.ACTIVE_MINING_CKPT,
                                             MP.HEAD_VERSION),
        "population": rep["population"],
        "roster": rep["roster"],
        "allocation": rep["allocation"],
        "split": {**rep["split"],
                  "rendered_eval_rows": sum(1 for r in rows
                                            if r["provenance"]["split_side"] == "eval"),
                  "rendered_train_rows": sum(1 for r in rows
                                             if r["provenance"]["split_side"] == "train")},
        "seeds": rep["seeds"],
        "render_defaults": {
            "width": W, "height": H, "ss": SS, "filter": FILT, "jpg_quality": JPG_Q,
            "interior_mode": "black", "composition": "center",
            "why_these_pins": "the July render-mode batches' own pins — the mining head was "
                              "trained on crops at these settings (train_mining_head config "
                              "src_dims [1280,720]), so a different render would ask v1 to "
                              "pre-label a distribution it has never seen",
            "render_path": {
                "pure": "render-one --dump-field <field spec> -> colormap.render_candidate "
                        "with the full approved colour params (transfer=grad faithful)",
                "composite/direct": "render-one --coloring <specs/*.json> --palette "
                                    "--colormaps (transfer=grad NOT expressible -> "
                                    "provenance.transfer_dropped)",
            },
        },
        "realized": {
            "rows_by_mode": dict(sorted(Counter(
                r["render"]["render_mode"] for r in rows).items())),
            "rows_by_family": dict(Counter(
                r["render"]["fractal_type"] for r in rows).most_common()),
            "rows_by_split": dict(Counter(
                r["provenance"]["split_side"] for r in rows).most_common()),
            "suggested_tier_hist": dict(sorted(Counter(
                r["suggested_tier"] for r in rows).items())),
            "transfer_dropped_rows": sum(1 for r in rows
                                         if r["provenance"]["transfer_dropped"]),
            "would_pass_mining_gate": sum(1 for r in rows
                                          if r["head_mining_v1"]["would_pass_gate"]),
            "pred": {"min": round(min(r["pred"] for r in rows), 4),
                     "max": round(max(r["pred"] for r in rows), 4)},
        },
        "presentation": {
            "order": "sheet_order — DESCENDING pred (good -> bad), ties on image_id",
            "sorted_on": "the continuous head score (pred), NOT the suggested tier, so "
                         "within-tier order is stable and meaningful",
            "contiguous": True,
            "source": "prompts/mining_sheet_addendum_sorted.md",
        },
        "labels_export": spec.labels_export,
        "labeling": {
            "ui": spec.ui_url,
            "mode": "correction — every row shows its suggested tier PREFILLED; Enter "
                    "confirms, 1-3 override. Only rows Matt acts on are exported.",
            "bulk": "accept all remaining, and accept all BELOW THIS ROW (the positional "
                    "sweep a sorted sheet makes natural) — both behind a confirm",
            "merge": f"uv run python tools/wallpaper/merge_sitting.py "
                     f"--corpus render_mode_corpus --batch {spec.batch_id} "
                     f"--scores labels/scores_{spec.batch_id}.json --apply",
        },
        "render_failures": errors,
        "run_status": {
            "planned_rows": rep["allocation"]["planned_rows"],
            "rendered_rows": len(rows),
            "n_failures": len(errors),
            "n_unaccounted": len(unaccounted),
            "unaccounted_rows": unaccounted[:50],
            "unaccounted_note": "planned but neither rendered nor failed — a bounded "
                                "(--limit) or interrupted run; re-run `render` then `write`",
        },
    }
    (spec.batch_dir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print_summary(spec, batch, rows)
    return batch


# =========================================================================== #
# Reporting.
# =========================================================================== #

def print_composition(rep):
    a = rep["allocation"]
    log("-" * 88)
    log(f"POPULATION  {rep['population']['rows']} rows / "
        f"{rep['population']['locations']} locations   "
        f"families {rep['population']['family_locations']}")
    log(f"ROSTER      {rep['roster']['n_modes']} modes  {rep['roster']['kinds']}")
    log(f"SPLIT       {rep['split']['n_units']} units "
        f"(multi-loc {rep['split']['n_multi_loc_units']}, base parents linked "
        f"{rep['split']['linked_base_parents']})  ·  locations "
        f"{rep['split']['n_train_loc']}T/{rep['split']['n_eval_loc']}E  ·  "
        f"{rep['split']['units_disjoint']}")
    log("-" * 88)
    log(f"{'mode':<34}{'kind':<11}{'target':>7}{'drawn':>7}{'eval':>7}  family alloc")
    for m in a["per_mode"]:
        fam = ",".join(f"{k[:4]}{v}" for k, v in sorted(m["family_alloc"].items()) if v)
        flag = "" if m["family_balanced"] else "  UNBALANCED"
        log(f"{m['mode']:<34}{m['kind']:<11}{m['target']:>7}{m['drawn']:>7}"
            f"{m['eval_rows']:>7}  {fam}{flag}")
    log("-" * 88)
    log(f"planned rows {a['planned_rows']} (target {a['target_rows']})  ·  locations used "
        f"{a['distinct_locations_used']}  ·  palettes {a['distinct_palettes_used']}")
    log(f"family rows: {a['family_rows']}")
    log(f"split rows:  train {rep['split']['train_rows']} / eval {rep['split']['eval_rows']}")
    log("-" * 88)


def print_summary(spec, batch, rows):
    r = batch["realized"]
    log("\n" + "=" * 88)
    log(f"MINING CORRECTION SHEET — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 88)
    log(f"rows {batch['n_rows']} / planned {batch['run_status']['planned_rows']}  ·  "
        f"failures {batch['run_status']['n_failures']}")
    log(f"by mode:   {r['rows_by_mode']}")
    log(f"by family: {r['rows_by_family']}")
    log(f"by split:  {r['rows_by_split']}")
    log(f"suggested: {r['suggested_tier_hist']}   "
        f"(prior shares {{{', '.join(f'{k}: {v:.3f}' for k, v in batch['suggested_tier_rule']['prior']['shares'].items())}}})")
    log(f"cuts on pred: {[round(c, 4) for c in batch['suggested_tier_rule']['cuts']]}  ·  "
        f"pred range {r['pred']}")
    log(f"transfer_dropped rows {r['transfer_dropped_rows']}  ·  would pass mining gate "
        f"{r['would_pass_mining_gate']}")
    log(f"-> {spec.batch_dir.relative_to(ROOT)}")
    log(f"-> serve: uv run python tools/viz/serve.py   then")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description="Render-mode (mining) correction sheet builder.")
    ap.add_argument("stage", choices=("plan", "render", "write"))
    ap.add_argument("--sheet", default="v1", choices=sorted(SHEETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="render: cap units this run. A short batch STAMPS itself "
                         "sheet_incomplete at write time (derived from the counts).")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="concurrent engine PROCESSES — the project cap is 4, do not raise")
    ap.add_argument("--dry-run", action="store_true", help="render: composition only")
    ap.add_argument("--unit-timeout", type=float, default=900.0,
                    help="per-render backstop, clamped to a quarter of --wall-budget-s")
    ap.add_argument("--wall-budget-s", type=float, default=6 * 3600.0)
    args = ap.parse_args()

    if args.workers > WORKERS:
        raise SystemExit(f"[mining-sheet] --workers {args.workers} exceeds the project "
                         f"process cap of {WORKERS} (CLAUDE.md). 4+ concurrent engines make "
                         f"the desktop unusable; in-process threads are the knob, not this.")
    missing = MR.missing_recipes()
    if missing:
        raise SystemExit(f"[mining-sheet] roster/recipe mismatch: {missing}")

    spec = SHEETS[args.sheet]
    prio = cc.set_below_normal_priority()
    log(f"[mining-sheet] {spec.batch_id} · priority {prio} · "
        f"{args.workers} workers x {ENGINE_THREADS} rayon threads")

    if args.stage == "plan":
        _entries, rep = plan(spec)
        print_composition(rep)
        return
    if args.stage == "render":
        run_render(spec, args)
        return
    if args.stage == "write":
        run_write(spec, args)
        return


if __name__ == "__main__":
    main()
