"""The v1 emission's LIBRARY HALF — the gate head, the gate+select pass and the emit
field-stem scheme. **This module no longer emits anything.**

WHAT LEFT ON 2026-08-11 (prompts/closure_sweep.md, `docs/design/retired.md`). `main()` and the
whole render tail under it — `emit`, `ensure_emit_field`, `config_from_params`,
`_coloring_recipe`, `_recipe_row`, `contact_sheet`, `_font`. It was the v1 emission driver:
gate the 2026-07-05 humanq3 crops on the wallpaper head, run the Stage-2d selector, and render
the winners at the 2560x1440 ss4 Lanczos-3 canon into `scratch/wallpaper/emit_v1/`. Nothing has
invoked it since the diversity-aware emission replaced it (`build_emission_diversity_v1`), its
output home has been wiped, and it was dead by both liveness methods. The three importers of
this module reach only into the half kept below.

WHAT IS LIVE, and who reads it:
  * `HEAD_CKPT` / `GATE_THRESHOLD` / `HEAD_VERSION` — re-exported from the torch-free
    `wallpaper_pins` under this module's own names, because callers spell them
    `emit_v1.GATE_THRESHOLD` (`build_emission_diversity_v1`, `q4_harvest_readout`,
    `test_floors_one_source`).
  * `load_v2_scorer` — the v2-parity head loader those same two readouts score with.
  * `build_and_select` — the gate+select pass, called by `palettes/viz_render_winners.py`.
  * `_emit_field_stem` — the emit field-identity hash. It outlives its own writer on purpose:
    the stem scheme is a FROZEN contract (`corpus/location.py` names it as the reference for
    which params a field key hashes) and `corpus/test_location.py` asserts its four token axes
    against a frozen digest. Deleting it would have deleted that gate.

CAVEAT, kept because `build_and_select` still carries it: the humanq3 locations were in the
head's training set, so `p_ge3` on that pool is OPTIMISTIC.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "queries"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))

import location as loc_mod           # noqa: E402  from_render_block / *_token (the stem axes)

# emission_selector by file path (module isn't a package).
_spec = importlib.util.spec_from_file_location(
    "emission_selector", REPO / "tools/wallpaper/emission_selector.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["emission_selector"] = es
_spec.loader.exec_module(es)

# ---- config surface (tunable in place, no code changes) -------------------- #
POOL_DEFAULT = REPO / "data/wallpaper_corpus/batches/2026-07-05_wallpaper_humanq3_v1"
# The head pin + gate threshold live in the TORCH-FREE `wallpaper_pins`, so a reader that
# only needs "which head is live, at what gate?" (the stage-2 floor owner's stamp check,
# every pure readout) does not import torch to ask. RE-EXPORTED under this module's own
# names because callers reach them as `emit_v1.HEAD_CKPT` / `emit_v1.GATE_THRESHOLD`; the
# rationale for the 0.90 retune moved with the constant.
from tools.wallpaper.wallpaper_pins import (  # noqa: E402,F401
    HEAD_CKPT, GATE_THRESHOLD, HEAD_VERSION)
PALETTE_CAP_FRAC = 0.05          # selector palette cap = max(2, ceil(frac * N_reachable_cells))
GRID = es.ColorGrid()            # 3x3 a/b x 2 L = 18 color cells; family x cell = behavior space

# The wallpaper canon the emit stems were hashed at: 2560x1440 grid ss4 Lanczos-3. The renderer
# that used it is gone; these survive as the STEM KEY's geometry, which is a frozen contract
# (`_emit_field_stem`, `corpus/test_location.py`) and must not be edited to match a future
# canon — a stem is an identity, not a target.
CANON_W, CANON_H, CANON_SS, CANON_FILTER = 2560, 1440, 4, "lanczos3"


@dataclasses.dataclass(frozen=True)
class RenderSpec:
    """The render geometry a field stem is keyed at. `eval_res` marked a row as NOT the canon
    back when rows were emitted; it is carried because `EVAL_SPEC` is the second geometry the
    stem scheme has to keep disjoint from the canon one."""
    width: int
    height: int
    ss: int
    filter: str
    eval_res: bool = False


CANON_SPEC = RenderSpec(CANON_W, CANON_H, CANON_SS, CANON_FILTER)
EVAL_SPEC = RenderSpec(1024, 576, 2, CANON_FILTER, eval_res=True)

OUT_DIR = REPO / "scratch" / "wallpaper" / "emit_v1"
CELL_CACHE = OUT_DIR / "colorcells.json"    # `load_color_cells`'s own dominant-Lab cache


# ===========================================================================
# 1. Gate-score: v2 inference on the existing crops (no re-render).
# ===========================================================================
def load_v2_scorer(device):
    """Load the v2 wallpaper head + its deterministic deploy transform. Returns a
    callable crops-paths -> (cond, marg, ssum) matching train_wallpaper_v2.predict_all:
    cond = sigmoid(logits) CONDITIONAL, marg = cumprod (the MARGINAL gate probs)."""
    import timm
    from classifier.data import Transform
    from classifier.model import BACKBONE, score_from_logits

    ckpt = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    K = int(cfg["num_classes"])                     # 4
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=K - 1,
                              drop_rate=cfg.get("drop_rate", 0.2),
                              drop_path_rate=cfg.get("drop_path_rate", 0.1))
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval().to(device)
    tf = Transform(geometry=cfg["geometry"], interp=cfg["interpolation"],
                   mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False)

    @torch.no_grad()
    def score(paths, batch_size=32):
        cond = np.zeros((len(paths), K - 1), dtype=np.float64)
        ssum = np.zeros(len(paths), dtype=np.float64)
        i = 0
        while i < len(paths):
            chunk = paths[i:i + batch_size]
            batch = []
            for p in chunk:
                with Image.open(p) as im:
                    im.load()
                    batch.append(tf(im.convert("RGB")))
            x = torch.stack(batch).to(device)
            logits = model(x).float()
            cond[i:i + len(chunk)] = torch.sigmoid(logits).cpu().numpy()
            ssum[i:i + len(chunk)] = score_from_logits(logits, "ordinal").cpu().numpy()
            i += len(chunk)
        marg = np.cumprod(cond, axis=1)             # marg[:,1] = marginal P(>=3)
        return cond, marg, ssum

    return score, cfg


# ===========================================================================
# 2/3. Build candidates (color cells cached) + run the selector.
# ===========================================================================
def _f(x):
    """Decimal-string coord -> float for the fractal-identity guard (None passes through;
    None c_re/c_im means 'no seed axis' — mandelbrot/multibrot/phoenix — and matches only
    other None)."""
    return None if x is None else float(x)


def _thumb_rgb(jpg: Path, w: int = 96) -> np.ndarray:
    with Image.open(jpg) as im:
        im = im.convert("RGB")
        iw, ih = im.size
        im = im.resize((w, max(1, round(w * ih / iw))), Image.BILINEAR)
        return np.asarray(im)


def load_color_cells(rows, crops_dir: Path) -> dict[str, int]:
    cache = json.loads(CELL_CACHE.read_text()) if CELL_CACHE.exists() else {}
    missing = [r["image_id"] for r in rows if r["image_id"] not in cache]
    if missing:
        print(f"[color] dominant Lab for {len(missing)} crops ({len(cache)} cached)")
        for iid in missing:
            cache[iid] = GRID.cell(es.dominant_lab(_thumb_rgb(crops_dir / f"{iid}.jpg"),
                                                   method="median"))
        CELL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CELL_CACHE.write_text(json.dumps(cache))
    return {r["image_id"]: cache[r["image_id"]] for r in rows}


def build_and_select(pool_dir: Path, gate_thr: float):
    rows = [json.loads(l) for l in (pool_dir / "images.jsonl").read_text().splitlines() if l.strip()]
    crops_dir = pool_dir / "crops"
    paths = [str(crops_dir / f"{r['image_id']}.jpg") for r in rows]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_fn, cfg = load_v2_scorer(device)
    print(f"[gate] scoring {len(rows)} crops with head {HEAD_CKPT.parent.name} "
          f"(best_epoch {cfg.get('best_epoch')}) on {device.type}")
    t0 = time.time()
    cond, marg, ssum = score_fn(paths)
    p_ge3 = marg[:, 1]
    print(f"[gate] scored in {time.time()-t0:.0f}s  ·  p_ge3 quantiles "
          f".5/.75/.9/.95/max = {np.round(np.quantile(p_ge3,[.5,.75,.9,.95,1]),4).tolist()}")

    # v2-parity cross-check on any eval crops we happen to score (honest sanity check).
    _parity_check(rows, p_ge3, ssum)

    cells = load_color_cells(rows, crops_dir)
    cands = []
    for i, r in enumerate(rows):
        prov = r.get("provenance", {}) or {}
        loc = loc_mod.from_render_block(r["render"])
        cands.append(es.Candidate(
            location_id=loc.key(),
            palette_id=r["render"]["palette"],
            family=prov.get("family") or r["render"].get("fractal_type") or "mandelbrot",
            fitness=float(ssum[i]),
            color_cell=cells[r["image_id"]],
            image_id=r["image_id"],
            # fractal-identity geometry: feeds the seeder-parity <=1/distinct-fractal
            # guard (sibling recolors converge on the same place at jittered coords).
            cx=_f(loc.cx), cy=_f(loc.cy), fw=_f(loc.fw),
            c_re=_f(loc.c_re), c_im=_f(loc.c_im),
            meta={"p_ge3": float(p_ge3[i]), "row": r},
        ))

    n_pass = sum(1 for c in cands if c.meta["p_ge3"] > gate_thr)
    res = es.select(cands, gate=lambda c: c.meta["p_ge3"] > gate_thr,
                    grid=GRID, palette_cap_frac=PALETTE_CAP_FRAC)
    picks = sorted(res.picks, key=lambda c: -c.fitness)
    print(f"[gate] p_ge3 > {gate_thr}: {n_pass}/{len(cands)} pass  ->  "
          f"[select] {len(picks)} emitted  "
          f"(cells {res.report['cells_filled']}/{res.report['cells_reachable']}, "
          f"{res.report['n_distinct_palettes_picked']} palettes, cap {res.palette_cap})")
    print(f"[select] per-family: {res.report['per_family_spread']}")
    return picks, res, n_pass


def _parity_check(rows, p_ge3, ssum):
    ev_path = HEAD_CKPT.parent / "eval_scores.jsonl"   # track the deployed head (v3), not v2
    if not ev_path.exists():
        return
    ev = {json.loads(l)["image_id"]: json.loads(l)
          for l in ev_path.read_text().splitlines() if l.strip()}
    idx = {r["image_id"]: i for i, r in enumerate(rows)}
    dp, ds = [], []
    for iid, e in ev.items():
        if iid in idx:
            dp.append(abs(p_ge3[idx[iid]] - e["p_ge3"]))
            ds.append(abs(ssum[idx[iid]] - e["score"]))
    if dp:
        print(f"[parity] vs v2 eval_scores on {len(dp)} shared crops: "
              f"max|Δp_ge3|={max(dp):.2e}  max|Δscore|={max(ds):.2e}")


# ===========================================================================
# 4. Emit — full-res render_candidate (Recipe 2) for each selected winner.
# ===========================================================================
def _emit_field_stem(loc, field_mode=None, spec=CANON_SPEC, field_source=None,
                     maxiter_policy=None):
    """Filename stem for `loc`'s emit field dump (pure, no I/O — the parity gate
    tests this directly, so `field_mode` stays the 2nd positional arg; `spec` defaults
    to the canon geometry, keeping the frozen 2560x1440ss4 smooth stem byte-identical).

    `field_mode` is the render-mode / field-identity token (`loc_mod.field_mode_token`):
    the smooth field (default/None) appends NOTHING to the hashed key — so the smooth
    stem is byte-identical to the pre-token scheme — while a strange pure-field mode
    keys distinctly and never collides with the cached smooth field. `field_source`
    (`--dump-field-source`, `loc_mod.field_source_token`) closes the same-shaped axis
    for the field SOURCE: `beautiful` (default) appends nothing; `f64` keys disjointly
    so its offset field can't collide with a beautiful dump. `spec` folds the render
    geometry into the key so an eval-res (1024x576ss2) field never collides with the
    wallpaper-canon field for the same location."""
    import hashlib
    tok = loc_mod.field_mode_token(field_mode)
    suffix = f"|{tok}" if tok else ""
    stok = loc_mod.field_source_token(field_source)
    suffix += f"|{stok}" if stok else ""
    # iteration-CAP policy token — legacy policy appends nothing (frozen stems stay
    # byte-identical), any other policy keys disjointly. See docs/design/auto_maxiter.md.
    ptok = loc_mod.maxiter_policy_token(maxiter_policy)
    suffix += f"|{ptok}" if ptok else ""
    geom = f"{spec.width}x{spec.height}ss{spec.ss}"
    h = hashlib.sha1(f"{loc.key()}|{geom}|{loc.maxiter}{suffix}".encode()).hexdigest()[:16]
    return f"{loc.family}_{h}_{geom}"
