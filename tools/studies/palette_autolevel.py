r"""palette_autolevel.py — PROPOSAL-ONLY study: a Photoshop-style auto-level that is
MEASURED in screen space and APPLIED on the palette.

The lever. A render's per-pixel color is `palette.lookup_linear(tt)` where `tt` is the
mode's final normalized scalar (`render_modes.rs::render_beautiful_composite`), so image
luminance is a pure function of the LUT: push a monotone tone curve `C` through the LUT's
OKLab L and EVERY pixel's lightness moves by exactly `C`. That identity is why the
correction can be measured on the rendered bytes and applied on the palette — the two are
the same map. It holds exactly per subpixel and approximately after AA (downsample averages
COLORS across subpixels, so an edge pixel is a mean of curve-mapped colors, not the curve of
a mean); the measured deviation is reported per example (`pushforward_dev`).

WHAT THIS IS NOT. Nothing here is wired into a production path. The committed LUT seam
(Rust `palette.rs` ↔ the Python coloring tail) is byte-identity-critical and is NOT touched:
the study writes a ONE-ENTRY colormap JSON into `scratch/` carrying the adjusted sRGB8 stops
under the SAME palette name and hands it to `render-one --colormaps`, so the Rust bake, the
mirror flag and the spec are bit-identical to the production call and only the stop COLORS
differ. Stops are densified first (subdivision on a piecewise-linear OKLab segment is exact
identity for the un-curved palette), because a 33-stop palette samples the curve too coarsely.

Four arms per example, all at the mining head's label geometry (1280x720 ss2 lanczos3
jpg q95, `mining_roster`'s own recipe):
  before         the production render (parity-checked against the committed corpus crop)
  autolevel      the same render through the tone-curved LUT (endpoints sent to 0/1 — the
                 textbook auto-level)
  autolevel_soft the same curve with the endpoints sent to the LABELED population's own
                 black/white medians instead. The prompt asked for three arms; this fourth
                 exists because the endpoint policy is the rule's one free parameter and a
                 single setting of it would price the lever, not the lever's idea.
  histeq         the same render with the BASE field's `transform` set to `histeq` — the
                 reference arm, so the new lever is priced against the knob that exists.

    uv run python -u tools/studies/palette_autolevel.py all --limit 2   # bounded end-to-end
    uv run python -u tools/studies/palette_autolevel.py all             # the study
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_common as cc                              # noqa: E402  engine launch defaults
import location as loc_mod                              # noqa: E402
from tools.mining import mining_roster as MR            # noqa: E402  THE mode recipe owner
from tools.palettes.color import (                      # noqa: E402  THE numpy Oklab reference
    oklab_to_srgb, srgb_to_oklab)

EXE = str(ROOT / "target/release/fractal-generator.exe")
POOL_CMAPS = ROOT / "data/palettes/pool_colormaps.json"
CORPUS = ROOT / "data/render_mode_corpus/batches"


# =========================================================================== #
# 1. The study instance (one today; a second is an entry, not a refactor).
# =========================================================================== #
@dataclass(frozen=True)
class StudySpec:
    name: str
    modes: tuple                 # the modes under test
    n_per_mode: int
    min_label: int               # human label floor for the population ("would plausibly ship")
    out_dir: Path
    # render geometry == the mining head's label crops (dataset_v1)
    geom: tuple = (1280, 720, 2)
    filt: str = "lanczos3"
    jpg_q: int = 95
    workers: int = 3             # concurrent engine PROCESSES (<= 4, CLAUDE.md)
    engine_threads: int = 4      # per-process rayon threads (3x4 = 12 logical cores)
    arms: tuple = ("before", "autolevel", "autolevel_soft", "histeq")


SPECS = {
    "exposure_verify_2modes_v2": StudySpec(
        name="exposure_verify_2modes_v2",
        modes=("composite_c7_smooth_trap_circle", "smooth_angle_min"),
        n_per_mode=20,
        min_label=2,
        out_dir=ROOT / "scratch" / "exposure_verify",
    ),
}


# =========================================================================== #
# 2. The auto-level rule.
#
# MEASURE (screen space, on the rendered bytes):
#   black point  b = P_{CLIP_LO}(L)        over ALL pixels — a true-black background keeps
#   white point  w = P_{CLIP_HI}(L)        b at 0, so nothing crushes what is already black
#   midtone      m = median(L | L > MASK_L) — IN-MASK, so a large dead background cannot
#                                             drag the midtone statistic to zero
# CURVE:  C(L) = lo + (hi-lo) * clip((L-b)/(w-b), 0, 1) ** p, with (lo, hi) the OUTPUT
#         endpoints — (0, 1) for the textbook arm, the labeled population's own black/white
#         medians for the `soft` arm. p solves C(m) = MID_TARGET, clamped to
#         [1/EXP_CLAMP, EXP_CLAMP]. Monotone by construction; C(b or below) = lo.
# APPLY (on the LUT, in OKLab): L' = C(L); chroma scaled by (L'/L)**RHO, and RHO IS 0 —
#         the curve moves LIGHTNESS ONLY and leaves (a, b) alone. RHO=1 (constant C/L) was
#         the first choice and was MEASURED on the bounded run to roughly halve in-mask
#         chroma (0.212→0.123, 0.111→0.063 on the first four examples), i.e. it pays for the
#         re-exposure in colorfulness, which is the one thing these palettes are picked for.
#         Out-of-gamut colors are pulled back by chroma reduction, never by clipping RGB
#         (which rotates hue).
# =========================================================================== #
CLIP_LO, CLIP_HI = 0.5, 99.5      # percentiles for the robust black/white points
MASK_L = 0.04                     # OKLab L floor of the "structure" mask
MID_TARGET = None                 # set from the labeled corpus — see MID_TARGET_BASIS
MID_TARGET_BASIS = ROOT / "scratch" / "exposure_verify" / "mid_target.json"
EXP_CLAMP = 2.0                   # p in [0.5, 2.0]
RHO = 0.0                         # chroma is left alone (see the block comment)
MIN_RANGE = 0.05                  # w-b below this ⇒ degenerate, no curve proposed
DENSIFY = 8                       # stop-subdivision factor before the curve is applied


def _mid_target() -> float:
    """The midtone target, DERIVED (never invented): the median in-mask OKLab-L median of the
    human label-3 crops of the mining corpus, measured by `mid_target` and frozen to a file so
    the study's renders and its report read the same number."""
    if MID_TARGET is not None:
        return float(MID_TARGET)
    doc = json.loads(MID_TARGET_BASIS.read_text(encoding="utf-8"))
    return float(doc["mid_target"])


def _soft_ends() -> tuple:
    """The `soft` arm's output endpoints: the labeled population's OWN black/white medians
    (label 3), from the same file as the midtone target. The textbook auto-level sends the
    measured endpoints to (0, 1); good renders in this corpus do not actually sit there."""
    doc = json.loads(MID_TARGET_BASIS.read_text(encoding="utf-8"))
    lab3 = doc["per_label"]["3"]
    return (float(lab3["black_median"]), float(lab3["white_median"]))


def tone_stats(img: np.ndarray) -> dict:
    """Screen-space measurement of one rendered image (uint8 HxWx3)."""
    lab = srgb_to_oklab(img.astype(np.float32) / 255.0)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    mask = L > MASK_L
    b = float(np.percentile(L, CLIP_LO))
    w = float(np.percentile(L, CLIP_HI))
    m = float(np.median(L[mask])) if mask.any() else float(np.median(L))
    return {"black_pt": b, "white_pt": w, "mid": m,
            "mask_frac": float(mask.mean()),
            "in_mask_chroma": float(C[mask].mean()) if mask.any() else 0.0,
            "L_mean": float(L.mean())}


def derive_curve(st: dict, out_ends: tuple = (0.0, 1.0)) -> dict:
    """Measured stats → the tone curve's numbers (+ why, when it declines).

    `out_ends` is where the measured black/white points are SENT. `(0, 1)` is the textbook
    auto-level (full stretch); the `soft` arm sends them to the labeled population's own
    black/white medians instead, which is the same empirical basis as the midtone target and
    the one free parameter of the whole rule."""
    b, w, m = st["black_pt"], st["white_pt"], st["mid"]
    lo, hi = out_ends
    if w - b < MIN_RANGE:
        return {"applies": False, "reason": f"degenerate range w-b={w - b:.3f} < {MIN_RANGE}",
                "black_pt": b, "white_pt": w, "exponent": 1.0, "out_ends": [lo, hi]}
    t = (m - b) / (w - b)
    tgt = _mid_target()
    tgt_n = (tgt - lo) / (hi - lo)
    if not (1e-4 < t < 1 - 1e-4) or not (1e-4 < tgt_n < 1 - 1e-4):
        p = 1.0
    else:
        p = float(np.log(tgt_n) / np.log(t))
    p = float(np.clip(p, 1.0 / EXP_CLAMP, EXP_CLAMP))
    return {"applies": True, "reason": None, "black_pt": b, "white_pt": w,
            "exponent": p, "mid_in": m, "mid_target": tgt, "mid_norm": t,
            "out_ends": [lo, hi], "clamped": abs(p - EXP_CLAMP) < 1e-9 or
            abs(p - 1.0 / EXP_CLAMP) < 1e-9}


def apply_curve_L(L: np.ndarray, cur: dict) -> np.ndarray:
    b, w, p = cur["black_pt"], cur["white_pt"], cur["exponent"]
    lo, hi = cur.get("out_ends", (0.0, 1.0))
    t = np.clip((L - b) / max(w - b, 1e-9), 0.0, 1.0)
    return lo + (hi - lo) * t ** p


# --------------------------------------------------------------------------- #
# LUT surgery. `stops` are the colormap-library `[pos, [r,g,b]]` pairs, i.e. exactly
# what `palette_pick::parse_colormaps` hands to `Palette::from_srgb8_stops_mirrored`.
# --------------------------------------------------------------------------- #
def densify(stops: list, mirror: bool, k: int = DENSIFY) -> list:
    """Subdivide each segment `k`-fold, interpolating in OKLab — the SAME space and the same
    piecewise-linear rule `interp_oklab_cyclic` uses, so this is identity for the palette
    (to sRGB8 rounding) and only makes the tone curve's sampling finer.

    The wrap segment (last→first, through pos 1) is subdivided ONLY for cyclic maps: a
    `mirror_needed` palette is re-based by `mirror_stops` onto [p0, p_last], so a stop placed
    outside that span would change the bake instead of refining it."""
    s = sorted(((float(p) % 1.0, [int(c) for c in rgb]) for p, rgb in stops), key=lambda x: x[0])
    lab = [srgb_to_oklab(np.array(rgb, dtype=np.float64) / 255.0) for _, rgb in s]
    segs = list(range(len(s) - 1))
    out = []
    for i in segs:
        p0, p1 = s[i][0], s[i + 1][0]
        for j in range(k):
            f = j / k
            out.append((p0 + (p1 - p0) * f, lab[i] + (lab[i + 1] - lab[i]) * f))
    if mirror:
        out.append((s[-1][0], lab[-1]))
    else:
        p0, p1 = s[-1][0], s[0][0] + 1.0
        for j in range(k):
            f = j / k
            out.append(((p0 + (p1 - p0) * f) % 1.0, lab[-1] + (lab[0] - lab[-1]) * f))
    return out


def _in_gamut(lab: np.ndarray) -> tuple:
    """(is_in_gamut, srgb). `oklab_to_srgb` CLIPS, so "did it clip?" is asked by round-trip:
    an in-gamut color survives lab→sRGB→lab unchanged, an out-of-gamut one does not. Asking
    the clipped output whether it is in range instead always answers yes — that bug silently
    hard-clipped the darkened stops and cost the curve ~0.06 mean L of fidelity."""
    rgb = oklab_to_srgb(lab.reshape(1, 3)).reshape(3)
    back = srgb_to_oklab(rgb.reshape(1, 3)).reshape(3)
    return float(np.max(np.abs(back - lab))) < 1e-6, rgb


def _gamut_fit(lab: np.ndarray) -> list:
    """OKLab → sRGB8, pulling an out-of-gamut color back by CHROMA reduction (bisection on
    the a/b scale), never by per-channel clipping — clipping rotates hue AND moves L, which
    is the one axis the curve is supposed to control."""
    def at(scale):
        return _in_gamut(np.array([lab[0], lab[1] * scale, lab[2] * scale]))

    ok, rgb = at(1.0)
    if not ok:
        lo, hi = 0.0, 1.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if at(mid)[0]:
                lo = mid
            else:
                hi = mid
        rgb = at(lo)[1]
    return [int(round(float(np.clip(c, 0.0, 1.0)) * 255.0)) for c in rgb]


def curved_stops(stops: list, mirror: bool, cur: dict) -> list:
    """The adjusted stop list: densify, push L through C, scale chroma by (L'/L)**RHO."""
    dense = densify(stops, mirror)
    out = []
    for pos, lab in dense:
        L = float(lab[0])
        Lp = float(apply_curve_L(np.array([L]), cur)[0])
        s = (Lp / L) ** RHO if L > 1e-6 else 0.0
        out.append([round(pos, 9), _gamut_fit(np.array([Lp, lab[1] * s, lab[2] * s]))])
    return out


# =========================================================================== #
# 3. Population — the labeled mining corpus, which IS the strong-q3 material.
# =========================================================================== #
def load_library() -> dict:
    return {c["name"]: c for c in json.loads(POOL_CMAPS.read_text(encoding="utf-8"))}


def select_population(spec: StudySpec) -> list:
    """Per mode: near-dup representatives at label >= `min_label`, best label first, one row
    per location. The corpus's own population is `gate_passers_v3` — wallpaper-head p_ge3 >
    0.90 locations from the dramatic-palette headbatch — so "strong q3 location with a good
    palette" is a property of the source, and the human label is what makes the example one
    that would plausibly ship."""
    from tools.mining import mining_corpus as MC

    raw = {}
    for b in MC.POOL_BATCHES:
        for line in (CORPUS / b / "images.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                raw[r["image_id"]] = r
    pool = MC.load_corpus()
    lib = load_library()
    out = []
    for mode in spec.modes:
        cand = [r for r in pool.rows
                if r.mode == mode and r.is_rep and r.label >= spec.min_label]
        # Half "good" (3), half "okay" (2), the same mix for every mode: a lever is judged
        # on shippable material (the prompt's rule) but it can only SHOW work on the
        # marginal half, and an all-3 population for one mode and a mixed one for the other
        # would price the two modes on different material.
        half = spec.n_per_mode // 2
        by_lab = {}
        for r in sorted(cand, key=lambda r: (-r.v1_p_ge3, r.image_id)):
            by_lab.setdefault(r.label, []).append(r)
        top = by_lab.get(3, [])[:half]
        rest = [r for r in cand if r not in top]
        rest.sort(key=lambda r: (r.label != 2, -r.label, -r.v1_p_ge3, r.image_id))
        seen, picked = set(), []
        for r in top + rest:
            if r.loc in seen:
                continue
            row = raw[r.image_id]
            pal = row["render"]["palette"]
            if pal not in lib:
                continue
            seen.add(r.loc)
            picked.append({
                "example_id": f"{r.image_id}",
                "mode": mode,
                "label": r.label,
                "v1_p_ge3": r.v1_p_ge3,
                "batch": [b for b in MC.POOL_BATCHES if (CORPUS / b / "crops" /
                                                         f"{r.image_id}.jpg").exists()][0],
                "corpus_crop": str((r.jpg).relative_to(ROOT).as_posix()),
                "palette": pal,
                "palette_source": lib[pal].get("source"),
                "fractal_type": row["render"].get("fractal_type"),
                "render": row["render"],
                "color_params": row["provenance"]["color_params"],
                "mode_params": row["render"].get("mode_params") or {},
            })
            if len(picked) >= spec.n_per_mode:
                break
        out.extend(picked)
    return out


# =========================================================================== #
# 4. Render — one code path, three arms, production recipe from `mining_roster`.
# =========================================================================== #
def _run(cmd, threads: int, timeout_s: int = 900):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env,
                       timeout=timeout_s, creationflags=cc.default_creationflags())
    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd[:4])) + "\n" + r.stderr[-700:])


def build_spec(ex: dict, lib: dict, histeq: bool) -> dict:
    """The production composite spec (`build_mining_sheet._render_rust`'s rule), with the
    reference arm's single override."""
    spec = MR.spec_for(ex["mode"], ex.get("mode_params"))
    p = ex["color_params"]
    ptype = lib[ex["palette"]].get("cycle")
    spec["transform"] = "log" if p["log_premap"] == "log" else "linear"
    spec["gamma"] = float(p["gamma"])
    spec["reverse"] = bool(p["reverse"])
    if ptype == "cyclic":
        spec["palette_cycles"] = float(p["n_cycles"])
        spec["palette_offset"] = float(p["phase"])
    rname, rstrength = MR.rolloff_for(ex["mode"])
    if rname != "none":
        spec["rolloff"] = rname
        spec["rolloff_strength"] = rstrength
    if histeq:
        spec["transform"] = "histeq"
    return spec


def render_arm(ex: dict, spec_json: dict, cmaps: Path, out_jpg: Path, sspec: StudySpec,
               tmp_dir: Path):
    w, h, ss = sspec.geom
    loc = loc_mod.from_render_block(ex["render"])
    tmp_png = tmp_dir / f"{out_jpg.stem}.png"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run([EXE, "render-one"] + loc_mod.render_one_flags(loc)
             + ["--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw, "--maxiter", str(loc.maxiter),
                "--width", str(w), "--height", str(h), "--supersample", str(ss),
                "--filter", sspec.filt, "--palette", ex["palette"],
                "--colormaps", str(cmaps), "--coloring", json.dumps(spec_json),
                "--out", str(tmp_png)], sspec.engine_threads)
        with Image.open(tmp_png) as im:
            im.convert("RGB").save(out_jpg, quality=sspec.jpg_q)
    finally:
        tmp_png.unlink(missing_ok=True)


def one_example(job: tuple) -> dict:
    """before → measure → curve → adjusted 1-entry colormap → autolevel → histeq."""
    ex, sspec_d, out_dir_s = job
    sspec = StudySpec(**{**sspec_d, "out_dir": Path(sspec_d["out_dir"])})
    out_dir = Path(out_dir_s)
    arms_dir = out_dir / "arms"
    tmp_dir = out_dir / "_tmp"
    cm_dir = out_dir / "_cmaps"
    for d in (arms_dir, tmp_dir, cm_dir):
        d.mkdir(parents=True, exist_ok=True)
    lib = load_library()
    eid = ex["example_id"]
    t0 = time.time()
    rec = dict(ex)

    before = arms_dir / f"{eid}__before.jpg"
    if not before.exists():
        render_arm(ex, build_spec(ex, lib, False), POOL_CMAPS, before, sspec, tmp_dir)
    img_b = np.asarray(Image.open(before).convert("RGB"))
    st = tone_stats(img_b)
    rec["before_stats"] = st

    # parity of this study's render path against the committed corpus crop
    crop = ROOT / ex["corpus_crop"]
    if crop.exists():
        a = np.asarray(Image.open(crop).convert("RGB")).astype(np.int16)
        if a.shape == img_b.shape:
            d = np.abs(a - img_b.astype(np.int16))
            rec["crop_parity"] = {"mean_abs": float(d.mean()), "max_abs": int(d.max())}

    Lb = srgb_to_oklab(img_b.astype(np.float32) / 255.0)[..., 0]
    for arm, ends in (("autolevel", (0.0, 1.0)), ("autolevel_soft", _soft_ends())):
        cur = derive_curve(st, ends)
        rec["curve" if arm == "autolevel" else "curve_soft"] = cur
        if not cur["applies"]:
            continue
        entry = dict(lib[ex["palette"]])
        entry["stops"] = curved_stops(entry["stops"], bool(entry.get("mirror_needed")), cur)
        cmp_path = cm_dir / f"{eid}__{arm}.json"
        cmp_path.write_text(json.dumps([entry]), encoding="utf-8")
        al = arms_dir / f"{eid}__{arm}.jpg"
        if not al.exists():
            render_arm(ex, build_spec(ex, lib, False), cmp_path, al, sspec, tmp_dir)
        img_a = np.asarray(Image.open(al).convert("RGB"))
        rec[f"{arm}_stats"] = tone_stats(img_a)
        # the pushforward identity: predicted L = C(before L). AA breaks it only at edges.
        La = srgb_to_oklab(img_a.astype(np.float32) / 255.0)[..., 0]
        pred = apply_curve_L(Lb, cur)
        rec[f"{arm}_pushforward_dev"] = {
            "mean_abs": float(np.abs(pred - La).mean()),
            "p99_abs": float(np.percentile(np.abs(pred - La), 99))}

    he = arms_dir / f"{eid}__histeq.jpg"
    if not he.exists():
        render_arm(ex, build_spec(ex, lib, True), POOL_CMAPS, he, sspec, tmp_dir)
    rec["histeq_stats"] = tone_stats(np.asarray(Image.open(he).convert("RGB")))
    rec["secs"] = time.time() - t0
    return rec


# =========================================================================== #
# 5. Sheet.
# =========================================================================== #
TILE_W, TILE_H, CAP_H, PAD = 560, 315, 96, 8


def _font(sz):
    for name in ("consola.ttf", "arial.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _cap(rec: dict, arm: str) -> str:
    st = rec.get(f"{arm}_stats") if arm != "before" else rec.get("before_stats")
    lines = [arm.upper()]
    if st:
        lines.append(f"L med(mask) {st['mid']:.3f}  blk {st['black_pt']:.3f} "
                     f"wht {st['white_pt']:.3f}")
        lines.append(f"in-mask chroma {st['in_mask_chroma']:.4f}  mask {st['mask_frac']:.2f}")
    c = rec.get("curve" if arm == "autolevel" else "curve_soft")
    if arm.startswith("autolevel") and c:
        lines.append(f"curve [{c['black_pt']:.3f},{c['white_pt']:.3f}]->"
                     f"[{c['out_ends'][0]:.2f},{c['out_ends'][1]:.2f}] p={c['exponent']:.3f}"
                     f"{'  CLAMPED' if c.get('clamped') else ''}"
                     if c["applies"] else f"NOT APPLIED — {c['reason']}")
    if arm in rec.get("gate", {}):
        lines[0] += f"   gate p_ge3 {rec['gate'][arm]:.3f}"
    return "\n".join(lines)


def build_sheet(recs: list, mode: str, out_png: Path, sspec: StudySpec):
    rows = [r for r in recs if r["mode"] == mode]
    f = _font(15)
    fh = _font(19)
    head = 58
    W = PAD + len(sspec.arms) * (TILE_W + PAD)
    H = head + len(rows) * (TILE_H + CAP_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 12), f"{mode} — before | palette auto-level (full / soft) | histeq   "
                      f"({len(rows)} examples, {sspec.geom[0]}x{sspec.geom[1]} ss{sspec.geom[2]}, "
                      f"literal render bytes)", font=fh, fill=(235, 235, 235))
    y = head
    for r in rows:
        for i, arm in enumerate(sspec.arms):
            x = PAD + i * (TILE_W + PAD)
            p = out_png.parent / "arms" / f"{r['example_id']}__{arm}.jpg"
            if p.exists():
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS),
                                (x, y))
            else:
                d.rectangle([x, y, x + TILE_W, y + TILE_H], fill=(40, 30, 30))
                d.text((x + 8, y + 8), "(no render)", font=f, fill=(200, 120, 120))
            d.multiline_text((x + 4, y + TILE_H + 4), _cap(r, arm), font=f,
                             fill=(210, 210, 215), spacing=3)
        d.text((PAD, y + TILE_H + CAP_H - 20),
               f"{r['example_id']}  label {r['label']}  {r['fractal_type']}  "
               f"palette {r['palette']} ({r['palette_source']})  "
               f"crop-parity mean|Δ| {r.get('crop_parity', {}).get('mean_abs', float('nan')):.2f}",
               font=f, fill=(150, 150, 160))
        y += TILE_H + CAP_H + PAD
    sheet.save(out_png)


# =========================================================================== #
# 6. Driver.
# =========================================================================== #
def cmd_mid_target(sspec: StudySpec, args):
    """Derive MID_TARGET from the labeled corpus (label-3 crops) and freeze it."""
    from tools.mining import mining_corpus as MC
    pool = MC.load_corpus()
    by = {}
    for lab in (1, 2, 3):
        sub = [r for r in pool.rows if r.label == lab]
        if args.sample and len(sub) > args.sample:
            rng = np.random.default_rng(0)
            sub = [sub[i] for i in rng.choice(len(sub), args.sample, replace=False)]
        mids, blks, whts, chr_ = [], [], [], []
        for r in sub:
            s = tone_stats(np.asarray(Image.open(r.jpg).convert("RGB")))
            mids.append(s["mid"])
            blks.append(s["black_pt"])
            whts.append(s["white_pt"])
            chr_.append(s["in_mask_chroma"])
        by[lab] = {"n": len(mids), "median": float(np.median(mids)),
                   "q25": float(np.percentile(mids, 25)), "q75": float(np.percentile(mids, 75)),
                   "black_median": float(np.median(blks)),
                   "white_median": float(np.median(whts)),
                   "chroma_median": float(np.median(chr_))}
        print(f"label {lab}: {by[lab]}", flush=True)
    sspec.out_dir.mkdir(parents=True, exist_ok=True)
    MID_TARGET_BASIS.write_text(json.dumps({
        "mid_target": by[3]["median"],
        "basis": "median over human label-3 crops of the pooled mining corpus "
                 "(mining_corpus.POOL_BATCHES) of the in-mask OKLab-L median "
                 f"(mask L > {MASK_L})",
        "derived": time.strftime("%Y-%m-%d"),
        "command": "uv run python tools/studies/palette_autolevel.py mid-target",
        "per_label": by}, indent=1), encoding="utf-8")
    print("wrote", MID_TARGET_BASIS)


def cmd_select(sspec: StudySpec, args):
    pop = select_population(sspec)
    sspec.out_dir.mkdir(parents=True, exist_ok=True)
    (sspec.out_dir / "population.json").write_text(json.dumps(pop, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"population {len(pop)}: " +
          ", ".join(f"{m}={sum(1 for p in pop if p['mode'] == m)}" for m in sspec.modes))
    print("labels:", dict(Counter((p["mode"], p["label"]) for p in pop)))
    return pop


def cmd_render(sspec: StudySpec, args):
    pop = json.loads((sspec.out_dir / "population.json").read_text(encoding="utf-8"))
    if args.limit:
        keep = []
        for m in sspec.modes:
            keep += [p for p in pop if p["mode"] == m][:args.limit]
        pop = keep
    sd = {k: (str(v) if isinstance(v, Path) else v) for k, v in sspec.__dict__.items()}
    jobs = [(ex, sd, str(sspec.out_dir)) for ex in pop]
    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=sspec.workers) as pool_x:
        futs = {pool_x.submit(one_example, j): j[0]["example_id"] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                recs.append(fu.result())
            except Exception as e:                                  # noqa: BLE001
                print(f"[fail] {futs[fu]}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"[{i}/{len(jobs)}] {futs[fu]} {recs[-1]['secs']:.1f}s "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    recs.sort(key=lambda r: (r["mode"], -r["label"], r["example_id"]))
    out = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    out.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    if args.limit:
        print(f"BOUNDED RUN (--limit {args.limit}) — records_limited.json is not the study")
    print("wrote", out)
    return recs


def cmd_score(sspec: StudySpec, args):
    """Secondary readout: the LOCKED mining gate on each arm. Not the judgment (the head
    never saw a tone-curved render); it prices the lever against the zero yield it targets."""
    from tools.mining.mining_gate import MiningScorer
    path = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    recs = json.loads(path.read_text(encoding="utf-8"))
    sc = MiningScorer()
    for r in recs:
        g = {}
        for arm in sspec.arms:
            p = sspec.out_dir / "arms" / f"{r['example_id']}__{arm}.jpg"
            if p.exists():
                g[arm] = sc.score_paths([p])[0].p_ge3
        r["gate"] = g
        r["gate_threshold"] = sc.threshold
    path.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    print("scored", len(recs), "examples; threshold", sc.threshold)
    return recs


def cmd_summary(sspec: StudySpec, args):
    """Per-mode derived-parameter spread + the arm deltas, as JSON beside the sheets. The
    spread is the number that decides whether a FIXED per-mode setting could replace the
    per-image derivation."""
    path = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    recs = json.loads(path.read_text(encoding="utf-8"))

    def q(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        return {"n": len(v), "median": float(np.median(v)), "min": float(np.min(v)),
                "max": float(np.max(v)), "iqr": [float(np.percentile(v, 25)),
                                                 float(np.percentile(v, 75))]}

    out = {"mid_target": _mid_target(), "soft_ends": list(_soft_ends()),
           "reference": json.loads(MID_TARGET_BASIS.read_text(encoding="utf-8"))["per_label"],
           "modes": {}}
    for mode in sspec.modes:
        rows = [r for r in recs if r["mode"] == mode]
        d = {"n": len(rows),
             "labels": {str(k): sum(1 for r in rows if r["label"] == k) for k in (2, 3)},
             "black_pt": q([r["before_stats"]["black_pt"] for r in rows]),
             "white_pt": q([r["before_stats"]["white_pt"] for r in rows]),
             "mid_before": q([r["before_stats"]["mid"] for r in rows]),
             "exponent": q([r["curve"]["exponent"] for r in rows]),
             "exponent_soft": q([r["curve_soft"]["exponent"] for r in rows]),
             "n_clamped": sum(1 for r in rows if r["curve"].get("clamped")),
             "n_not_applied": sum(1 for r in rows if not r["curve"]["applies"]),
             "crop_parity_max": max((r.get("crop_parity", {}).get("max_abs", 0)
                                     for r in rows), default=None)}
        for arm in sspec.arms:
            s = [r.get(f"{arm}_stats") if arm != "before" else r["before_stats"] for r in rows]
            s = [x for x in s if x]
            d[arm] = {"mid": q([x["mid"] for x in s]),
                      "black_pt": q([x["black_pt"] for x in s]),
                      "in_mask_chroma": q([x["in_mask_chroma"] for x in s])}
            if arm != "before":
                dev = [r.get(f"{arm}_pushforward_dev", {}).get("mean_abs") for r in rows]
                d[arm]["pushforward_mean_abs"] = q(dev)
            if any("gate" in r for r in rows):
                d[arm]["gate_p_ge3"] = q([r.get("gate", {}).get(arm) for r in rows])
                d[arm]["n_gate_pass"] = sum(
                    1 for r in rows
                    if (r.get("gate", {}).get(arm) or 0) >= r.get("gate_threshold", 1e9))
        out["modes"][mode] = d
    p = sspec.out_dir / "summary.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["modes"], indent=1)[:2000])
    print("wrote", p)
    return out


def cmd_sheet(sspec: StudySpec, args):
    path = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    recs = json.loads(path.read_text(encoding="utf-8"))
    for mode in sspec.modes:
        out = sspec.out_dir / f"sheet_{mode}.png"
        build_sheet(recs, mode, out, sspec)
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["mid-target", "select", "render", "score", "summary",
                                    "sheet", "all"])
    ap.add_argument("--study", default="exposure_verify_2modes_v2")
    ap.add_argument("--limit", type=int, default=0,
                    help="bounded end-to-end: N examples per mode (writes *_limited.json)")
    ap.add_argument("--sample", type=int, default=0, help="mid-target: crops per label")
    args = ap.parse_args()
    sspec = SPECS[args.study]
    if args.cmd in ("mid-target",):
        cmd_mid_target(sspec, args)
    if args.cmd in ("select", "all"):
        cmd_select(sspec, args)
    if args.cmd in ("render", "all"):
        cmd_render(sspec, args)
    if args.cmd in ("score", "all"):
        cmd_score(sspec, args)
    if args.cmd in ("summary", "all"):
        cmd_summary(sspec, args)
    if args.cmd in ("sheet", "all"):
        cmd_sheet(sspec, args)


if __name__ == "__main__":
    main()
