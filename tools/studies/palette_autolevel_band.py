r"""palette_autolevel_band.py — PROPOSAL-ONLY study: BAND-targeting auto-level.

Successor to `palette_autolevel.py` (point-targeting), which Matt's eyeball pass on the
`exposure_verify_2modes_v2` sheets scored as "`autolevel_soft` generally best, histeq worst"
with two named failures. Both are failures of POINT targeting, and both are addressed here:

  1. OVER-CORRECTION OF AN ALREADY-GOOD IMAGE (`rmf_0504`: blacks already at 0.084, and the
     rule darkened a fine bright picture anyway because its midtone was not exactly
     MID_TARGET). A point target has measure zero — every image is "off target", so the
     rule is always-on and never identity. Replaced by a BAND: inside → identity, outside →
     the MINIMUM pull that reaches the nearest band edge. Being always-on but mostly
     identity is now structural, not a hope.
  2. SATURATED DARKS CRUSHED TO BLACK (`mc20453_bc4abf7b`: a deep blue erased). Two causes,
     two guards — see `THE CHROMA GUARD` below.

A third, quieter fix rides along: the v2 curve CLAMPED outside [b, w], so every pixel below
the 0.5th percentile collapsed onto one lightness. The band curve is PIECEWISE with linear
tails ([0,b]→[0,lo] and [w,1]→[hi,1]), so true black stays black, true white stays white,
and the map is EXACTLY the identity when all three statistics are in band.

THE REFERENCE. The band is not invented: it is read off `LevelsCheck` — 35 wallpapers Matt
judges well-leveled, read-only, external to this repo — as the [P10, P90] interval of each
statistic across those images. `mid_target.json`'s label-3 corpus values are the independent
second opinion, reported beside it and never averaged with it.

THE CHROMA GUARD (the mc20453 fix), two mechanisms:
  * MEASUREMENT — the black point is read over NEUTRAL pixels only (OKLab chroma <=
    `CHROMA_NEUTRAL`). If the neutral subset is too small, or its own black point sits far
    ABOVE the all-pixel one, the image's dark tail is chromatic and there is no neutral black
    to read: the black end is declared UNMEASURABLE and left alone. The guard can only ever
    turn a correction OFF — it never manufactures a large one.
  * APPLICATION — a per-LUT-entry darkening cap. Chroma is never scaled by the curve
    (RHO = 0, inherited), so the only way a stop loses colour is the out-of-gamut pullback at
    its new lightness. Each entry's lightness is walked back toward its original value until
    the post-pullback chroma retains at least `CHROMA_RETAIN` of what it had.

WHAT THIS IS NOT: nothing here is wired into a production path. Same containment as v2 — a
one-entry colormap JSON in `scratch/` handed to `render-one --colormaps`, so the Rust bake,
the mirror flag and the spec are bit-identical to the production call and only the stop
COLOURS differ.

    uv run python -u tools/studies/palette_autolevel_band.py reference
    uv run python -u tools/studies/palette_autolevel_band.py all --limit 2   # bounded e2e
    uv run python -u tools/studies/palette_autolevel_band.py all
"""
from __future__ import annotations

import argparse
import json
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

from tools.palettes.color import oklab_to_srgb, srgb_to_oklab      # noqa: E402
from tools.studies import palette_autolevel as PA                  # noqa: E402
# THE RULE NOW SHIPS. What this study proposed was adopted and ported to
# `tools/palettes/autolevel.py` (the production operator) and `tools/palettes/
# levels_reference.py` (the reference deriver); both are imported here rather than kept as a
# second copy, so the rule that was measured on these sheets IS the rule the colorize path
# runs. Everything below is the study DRIVER — populations, arms, sheets, census — which is
# the half nothing in production wants.
from tools.palettes import autolevel as AL                         # noqa: E402
from tools.palettes import levels_reference as LR                  # noqa: E402


# =========================================================================== #
# 1. The study instance.
# =========================================================================== #
@dataclass(frozen=True)
class BandSpec:
    name: str
    ref_dir: Path                     # READ-ONLY reference wallpapers
    out_dir: Path
    v2_dir: Path                      # population.json + the cached before/soft arms
    band_lo_pct: float = 10.0
    band_hi_pct: float = 90.0
    geom: tuple = (1280, 720, 2)
    filt: str = "lanczos3"
    jpg_q: int = 95
    workers: int = 3                  # concurrent engine PROCESSES (<= 4, CLAUDE.md)
    engine_threads: int = 4
    arms: tuple = ("before", "band", "autolevel_soft")
    smooth_jpg_q: int = 90            # the wallpaper LABEL-CROP quality (`label_crop.JPG_Q`),
                                      # not the study's 95 — the smooth arms are rendered
                                      # through the corpus's own recipe so `crop_parity` is a
                                      # real check on the path rather than a JPEG delta
    smooth_n: int = 10                # the smooth-reach slice
    smooth_batches: tuple = ("2026-08-05_wallpaper_fresh_sheet_v1",
                             "2026-08-10_wallpaper_correction_v2",
                             "2026-07-09_wallpaper_headbatch_dramatic_v1")


SPECS = {
    "band_v1": BandSpec(
        name="band_v1",
        ref_dir=Path(r"C:\Users\techm\Desktop\GreatWallpapers\LevelsCheck"),
        out_dir=ROOT / "scratch" / "exposure_band",
        v2_dir=ROOT / "scratch" / "exposure_verify",
    ),
}


# =========================================================================== #
# 2. Measurement (screen space). Same three statistics as v2, plus the chroma guard.
#    ALL of it now lives in the production operator; these are aliases, not copies —
#    `test_palette_autolevel_band.py` reaches every one of them as `PB.<name>` and a second
#    literal would let the study's tests bless a rule the colorize path does not run.
# =========================================================================== #
CLIP_LO, CLIP_HI = AL.CLIP_LO, AL.CLIP_HI
MASK_L = AL.MASK_L
CHROMA_NEUTRAL = AL.CHROMA_NEUTRAL
NEUTRAL_FRAC_MIN = AL.NEUTRAL_FRAC_MIN
DARK_MARGIN = AL.DARK_MARGIN
CHROMA_RETAIN = AL.CHROMA_RETAIN
EXP_CLAMP = AL.EXP_CLAMP
MIN_RANGE = AL.MIN_RANGE
STAT_KEYS = AL.STAT_KEYS

tone_stats = AL.tone_stats


# =========================================================================== #
# 3. The reference band.
# =========================================================================== #
def measure_reference(sspec: BandSpec) -> dict:
    """Per-image stats over the read-only LevelsCheck set -> a band per statistic, written
    into the STUDY's own out_dir.

    The derivation itself is `tools/palettes/levels_reference.py` — the production deriver,
    which writes the committed `data/palettes/levels_reference.json`. This entry point stays
    because the study's `load_bands`/`cmd_census`/`cmd_summary` all read a record beside their
    own sheets; what it must NOT be is a second implementation of the band, which is why the
    per-image measurement, the [P10, P90] cut and the bootstrap edge SE all come from there.
    A study record is a scratch copy of the same derivation, never a rival to it."""
    per = LR.measure_source(sspec.ref_dir)
    doc = LR.build(per, src=sspec.ref_dir)
    doc["command"] = "uv run python tools/studies/palette_autolevel_band.py reference"
    sspec.out_dir.mkdir(parents=True, exist_ok=True)
    (sspec.out_dir / "levels_reference.json").write_text(LR.serialize(doc), encoding="utf-8")
    print("\nbands:", json.dumps({k: doc["bands"][k] and doc["bands"][k]["band"]
                                  for k in STAT_KEYS}))
    print("wrote", sspec.out_dir / "levels_reference.json")
    return doc


def load_bands(sspec: BandSpec) -> dict:
    doc = json.loads((sspec.out_dir / "levels_reference.json").read_text(encoding="utf-8"))
    return {k: tuple(doc["bands"][k]["band"]) for k in STAT_KEYS if doc["bands"][k]}


# =========================================================================== #
# 4. The band rule.
#
# PROJECT each measured statistic onto its band: inside -> itself (identity), outside ->
# the NEAREST edge. That is the MINIMUM move that reaches the acceptable set, which is why
# the pull "strength" is 1.0 and needs no free parameter: a partial pull would leave an
# out-of-band image out of band, and a pull past the edge would assert precision the 35
# reference images do not have.
#
# CURVE, piecewise on [0,1] and continuous:
#     L <= b        L * lo/b                              (tail: 0 -> 0)
#     b < L < w     lo + (hi-lo) * ((L-b)/(w-b))**p       (core)
#     L >= w        hi + (1-hi) * (L-w)/(1-w)             (tail: 1 -> 1)
# with lo = proj(b), hi = proj(w) and p solving C(m) = proj(m). Identity iff all three
# statistics are in band. The black end is skipped entirely (lo = b) when the chroma guard
# calls it unmeasurable.
# =========================================================================== #
project = AL.project
derive_band_curve = AL.derive_band_curve
apply_curve_L = AL.apply_curve_L


# =========================================================================== #
# 5. LUT surgery with the per-entry chroma cap — also the operator's, also aliased.
# =========================================================================== #
_chroma_after = AL._chroma_after
cap_lightness = AL.cap_lightness
curved_stops = AL.curved_stops


# =========================================================================== #
# 6. Arms. `before` and `autolevel_soft` are the v2 renders, REUSED byte-for-byte —
#    re-deriving them would only re-price the reference column.
# =========================================================================== #
def one_example(job: tuple) -> dict:
    ex, sd, out_dir_s, bands = job
    sspec = BandSpec(**{**sd, **{k: Path(sd[k]) for k in ("ref_dir", "out_dir", "v2_dir")}})
    out_dir = Path(out_dir_s)
    arms_dir = out_dir / "arms"
    for d in (arms_dir, out_dir / "_tmp", out_dir / "_cmaps"):
        d.mkdir(parents=True, exist_ok=True)
    lib = PA.load_library()
    eid = ex["example_id"]
    t0 = time.time()
    rec = dict(ex)

    v2_arms = sspec.v2_dir / "arms"
    before = v2_arms / f"{eid}__before.jpg"
    if not before.exists():
        before = arms_dir / f"{eid}__before.jpg"
        PA.render_arm(ex, PA.build_spec(ex, lib, False), PA.POOL_CMAPS, before, sspec,
                      out_dir / "_tmp")
    img_b = np.asarray(Image.open(before).convert("RGB"))
    rec["before_path"] = str(before.relative_to(ROOT).as_posix())
    st = tone_stats(img_b)
    rec["before_stats"] = st

    cur = derive_band_curve(st, bands)
    rec["curve_band"] = cur
    if cur["applies"]:
        entry = dict(lib[ex["palette"]])
        entry["stops"], n_capped = curved_stops(entry["stops"],
                                                bool(entry.get("mirror_needed")), cur)
        rec["n_chroma_capped"] = n_capped
        rec["n_stops"] = len(entry["stops"])
        cmp_path = out_dir / "_cmaps" / f"{eid}__band.json"
        cmp_path.write_text(json.dumps([entry]), encoding="utf-8")
        band_jpg = arms_dir / f"{eid}__band.jpg"
        if not band_jpg.exists():
            PA.render_arm(ex, PA.build_spec(ex, lib, False), cmp_path, band_jpg, sspec,
                          out_dir / "_tmp")
        img_a = np.asarray(Image.open(band_jpg).convert("RGB"))
        rec["band_stats"] = tone_stats(img_a)
        Lb = srgb_to_oklab(img_b.astype(np.float32) / 255.0)[..., 0]
        La = srgb_to_oklab(img_a.astype(np.float32) / 255.0)[..., 0]
        pred = apply_curve_L(Lb, cur)
        rec["band_pushforward_dev"] = {"mean_abs": float(np.abs(pred - La).mean()),
                                       "p99_abs": float(np.percentile(np.abs(pred - La), 99))}
        rec["band_vs_before"] = {"mean_abs_L": float(np.abs(La - Lb).mean()),
                                 "max_abs_L": float(np.abs(La - Lb).max())}

    soft = v2_arms / f"{eid}__autolevel_soft.jpg"
    if soft.exists():
        rec["autolevel_soft_stats"] = tone_stats(np.asarray(Image.open(soft).convert("RGB")))
        rec["autolevel_soft_path"] = str(soft.relative_to(ROOT).as_posix())
    rec["secs"] = time.time() - t0
    return rec


# =========================================================================== #
# 7. Sheet — the v2 inspection rules: literal render bytes, parameters captioned,
#    in-mask chroma printed, clip share never shown.
# =========================================================================== #
TILE_W, TILE_H, CAP_H, PAD = 560, 315, 104, 8


def _font(sz):
    for name in ("consola.ttf", "arial.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _cap(rec: dict, arm: str) -> str:
    st = rec.get("before_stats") if arm == "before" else rec.get(f"{arm}_stats")
    lines = [arm.upper()]
    if st:
        blk = "n/a(guard)" if st["black_pt"] is None else f"{st['black_pt']:.3f}"
        lines.append(f"L med(mask) {st['mid']:.3f}  blk {blk} (all {st['black_pt_all']:.3f}) "
                     f"wht {st['white_pt']:.3f}")
        lines.append(f"in-mask chroma {st['in_mask_chroma']:.4f}  dark chroma "
                     f"{st['dark_chroma']:.4f}  mask {st['mask_frac']:.2f}")
    if arm == "band":
        c = rec.get("curve_band") or {}
        if not c.get("applies"):
            lines.append(f"NOT APPLIED — {c.get('reason')}")
        elif c.get("identity"):
            lines.append("IDENTITY — all three statistics in band")
        else:
            s = c["sides"]
            lines.append(f"curve [{c['black_pt']:.3f},{c['white_pt']:.3f}]->"
                         f"[{c['out_ends'][0]:.3f},{c['out_ends'][1]:.3f}] p={c['exponent']:.3f}"
                         f"{'  CLAMPED' if c.get('clamped') else ''}")
            lines.append(f"sides blk{s['black_pt']:+d} wht{s['white_pt']:+d} mid{s['mid']:+d}"
                         f"{'  BLACK GUARDED' if c.get('black_guarded') else ''}"
                         f"  chroma-capped {rec.get('n_chroma_capped', 0)}/"
                         f"{rec.get('n_stops', 0)}")
    if arm == "autolevel_soft":
        lines.append("v2 reference column (point target, hard clamp)")
    if arm in rec.get("gate", {}):
        lines[0] += f"   gate p_ge3 {rec['gate'][arm]:.3f}"
    return "\n".join(lines)


def build_sheet(recs: list, title: str, out_png: Path, sspec: BandSpec, arms_lookup):
    f = _font(15)
    fh = _font(19)
    head = 58
    W = PAD + len(sspec.arms) * (TILE_W + PAD)
    H = head + len(recs) * (TILE_H + CAP_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 12), title, font=fh, fill=(235, 235, 235))
    y = head
    for r in recs:
        for i, arm in enumerate(sspec.arms):
            x = PAD + i * (TILE_W + PAD)
            p = arms_lookup(r, arm)
            if p and Path(p).exists():
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS),
                                (x, y))
            else:
                d.rectangle([x, y, x + TILE_W, y + TILE_H], fill=(40, 30, 30))
                d.text((x + 8, y + 8), "(no render)", font=f, fill=(200, 120, 120))
            d.multiline_text((x + 4, y + TILE_H + 4), _cap(r, arm), font=f,
                             fill=(210, 210, 215), spacing=3)
        d.text((PAD, y + TILE_H + CAP_H - 18),
               f"{r['example_id']}  label {r.get('label')}  {r.get('fractal_type')}  "
               f"{r.get('mode')}  palette {r.get('palette')}", font=f, fill=(150, 150, 160))
        y += TILE_H + CAP_H + PAD
    sheet.save(out_png)


# =========================================================================== #
# 8. Driver.
# =========================================================================== #
def _sd(sspec: BandSpec) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in sspec.__dict__.items()}


def cmd_render(sspec: BandSpec, args):
    pop = json.loads((sspec.v2_dir / "population.json").read_text(encoding="utf-8"))
    if args.limit:
        modes = sorted({p["mode"] for p in pop})
        keep = []
        for m in modes:
            keep += [p for p in pop if p["mode"] == m][:args.limit]
        pop = keep
    bands = load_bands(sspec)
    jobs = [(ex, _sd(sspec), str(sspec.out_dir), bands) for ex in pop]
    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=sspec.workers) as px:
        futs = {px.submit(one_example, j): j[0]["example_id"] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                recs.append(fu.result())
            except Exception as e:                                    # noqa: BLE001
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


def cmd_score(sspec: BandSpec, args):
    """Secondary readout: the LOCKED mining gate on each arm (the head never saw a
    tone-curved render, so this prices the lever, it does not judge it)."""
    from tools.mining.mining_gate import MiningScorer
    path = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    recs = json.loads(path.read_text(encoding="utf-8"))
    sc = MiningScorer()
    for r in recs:
        g = {}
        for arm in sspec.arms:
            p = _arm_path(sspec, r, arm)
            if p and Path(p).exists():
                g[arm] = sc.score_paths([Path(p)])[0].p_ge3
        r["gate"] = g
        r["gate_threshold"] = sc.threshold
    path.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    print("scored", len(recs), "examples; threshold", sc.threshold)
    return recs


def _arm_path(sspec: BandSpec, r: dict, arm: str):
    if arm == "band":
        return sspec.out_dir / "arms" / f"{r['example_id']}__band.jpg"
    key = f"{arm}_path"
    return ROOT / r[key] if key in r else None


def cmd_sheet(sspec: BandSpec, args):
    path = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    recs = json.loads(path.read_text(encoding="utf-8"))
    for mode in sorted({r["mode"] for r in recs}):
        rows = [r for r in recs if r["mode"] == mode]
        out = sspec.out_dir / f"sheet_{mode}.png"
        build_sheet(rows, f"{mode} — before | BAND rule | v2 autolevel_soft   "
                          f"({len(rows)} examples, {sspec.geom[0]}x{sspec.geom[1]} "
                          f"ss{sspec.geom[2]}, literal render bytes)",
                    out, sspec, lambda r, a: _arm_path(sspec, r, a))
        print("wrote", out)


def cmd_summary(sspec: BandSpec, args):
    path = sspec.out_dir / ("records.json" if not args.limit else "records_limited.json")
    recs = json.loads(path.read_text(encoding="utf-8"))
    doc = json.loads((sspec.out_dir / "levels_reference.json").read_text(encoding="utf-8"))

    def q(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        return {"n": len(v), "median": float(np.median(v)), "min": float(np.min(v)),
                "max": float(np.max(v)),
                "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}

    out = {"bands": {k: doc["bands"][k]["band"] for k in STAT_KEYS if doc["bands"][k]},
           "reference_n": doc["n_images"], "modes": {}}
    for mode in sorted({r["mode"] for r in recs}):
        rows = [r for r in recs if r["mode"] == mode]
        d = {"n": len(rows),
             "n_identity": sum(1 for r in rows if (r["curve_band"] or {}).get("identity")),
             "n_black_guarded": sum(1 for r in rows
                                    if (r["curve_band"] or {}).get("black_guarded")),
             "n_clamped": sum(1 for r in rows if (r["curve_band"] or {}).get("clamped")),
             "n_chroma_capped": q([r.get("n_chroma_capped") for r in rows]),
             "exponent": q([(r["curve_band"] or {}).get("exponent") for r in rows]),
             "band_move_mean_abs_L": q([(r.get("band_vs_before") or {}).get("mean_abs_L")
                                        for r in rows])}
        for arm in sspec.arms:
            s = [r.get("before_stats") if arm == "before" else r.get(f"{arm}_stats")
                 for r in rows]
            s = [x for x in s if x]
            d[arm] = {"mid": q([x["mid"] for x in s]),
                      "black_pt_all": q([x["black_pt_all"] for x in s]),
                      "in_mask_chroma": q([x["in_mask_chroma"] for x in s]),
                      "dark_chroma": q([x["dark_chroma"] for x in s])}
            if any("gate" in r for r in rows):
                d[arm]["gate_p_ge3"] = q([r.get("gate", {}).get(arm) for r in rows])
        out["modes"][mode] = d
    p = sspec.out_dir / "summary.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["modes"], indent=1)[:2500])
    print("wrote", p)
    return out


# =========================================================================== #
# 9. Out-of-range census — numbers only, over crops that ALREADY EXIST.
#
# THE SLICE (stated, because it is a pick): every human-labelled >= 3 crop present on disk
# in the two corpora that still HAVE crops —
#   W  data/wallpaper_corpus/batches/*      score in {3,4}, `labels/<generator_version>.json`
#      -> 1,915 rows, every one of them render_mode `smooth`: the release-shaped material.
#   M  data/render_mode_corpus/batches/*    label 3 (its top class, K=3) -> 148 rows, the
#      strange-mode material, split by mode.
# NOT in the slice, and why: `data/label_corpus/batches/*` carries 2,777 rows at >= 3 and
# ZERO crops (the derived-artifact chain wiped 2026-07-25); the 398 release-record rows carry
# a decision and a location but no image path, so censusing them means re-rendering, which
# this step is forbidden to do.
# =========================================================================== #
def _census_rows(sspec: BandSpec) -> list:
    import glob
    out = []
    for b in sorted(glob.glob(str(ROOT / "data/wallpaper_corpus/batches/*"))):
        b = Path(b)
        rows = [json.loads(x) for x in (b / "images.jsonl").read_text(encoding="utf-8")
                .splitlines() if x.strip()]
        gv = rows[0]["provenance"]["generator_version"]
        lp = ROOT / "labels" / f"{gv}.json"
        lab = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
        for r in rows:
            s = lab.get(r["image_id"])
            jpg = b / "crops" / f"{r['image_id']}.jpg"
            if s and s >= 3 and jpg.exists():
                out.append({"slice": "W", "image_id": r["image_id"], "label": int(s),
                            "mode": r["provenance"].get("render_mode"),
                            "family": r["render"].get("fractal_type"),
                            "palette": r["render"].get("palette"), "jpg": str(jpg)})
    from tools.mining import mining_corpus as MC
    pool = MC.load_corpus()
    for r in pool.rows:
        # near-dup REPRESENTATIVES only — `mining_corpus`'s own rule for an unweighted
        # statistic over distinct pictures (a census of duplicated looks counts one look
        # three times).
        if r.is_rep and r.label >= 3 and Path(r.jpg).exists():
            out.append({"slice": "M", "image_id": r.image_id, "label": int(r.label),
                        "mode": r.mode, "family": None, "palette": None, "jpg": str(r.jpg)})
    return out


def _census_one(path: str) -> dict:
    with Image.open(path) as im:
        return tone_stats(np.asarray(im.convert("RGB")))


def cmd_census(sspec: BandSpec, args):
    doc = json.loads((sspec.out_dir / "levels_reference.json").read_text(encoding="utf-8"))
    bands = {k: tuple(doc["bands"][k]["band"]) for k in STAT_KEYS if doc["bands"][k]}
    alt = {k: tuple(doc["bands"][k]["iqr"]) for k in STAT_KEYS if doc["bands"][k]}
    rows = _census_rows(sspec)
    if args.limit:
        rows = rows[:args.limit]
    print(f"census slice: {len(rows)} crops "
          f"(W={sum(1 for r in rows if r['slice'] == 'W')}, "
          f"M={sum(1 for r in rows if r['slice'] == 'M')})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as px:            # 4 PROCESSES (CLAUDE.md)
        futs = {px.submit(_census_one, r["jpg"]): i for i, r in enumerate(rows)}
        done = 0
        for fu in as_completed(futs):
            rows[futs[fu]]["stats"] = fu.result()
            done += 1
            if done % 250 == 0:
                print(f"  [{done}/{len(rows)}] {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        st = r["stats"]
        sides = {}
        for k in STAT_KEYS:
            v = st[k] if k != "black_pt" else st["black_pt"]
            sides[k] = None if v is None else project(v, bands[k])[1]
        r["sides"] = sides
        r["identity"] = all(s in (0, None) for s in sides.values())
        r["identity_iqr"] = all(
            (st[k] is None) or project(st[k], alt[k])[1] == 0 for k in STAT_KEYS)
    out = _census_report(rows, bands, alt)
    p = sspec.out_dir / "census.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (sspec.out_dir / "census_rows.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "jpg"} for r in rows]),
        encoding="utf-8")
    print(json.dumps(out, indent=1)[:3500])
    print("wrote", p)
    return out


def _census_report(rows: list, bands: dict, alt: dict) -> dict:
    def tally(sub):
        n = len(sub)
        if not n:
            return None
        d = {"n": n, "identity": sum(1 for r in sub if r["identity"]),
             "identity_iqr_band": sum(1 for r in sub if r["identity_iqr"]),
             "black_guarded": sum(1 for r in sub if r["stats"]["black_pt"] is None)}
        for k in STAT_KEYS:
            d[k] = {"below": sum(1 for r in sub if r["sides"][k] == -1),
                    "above": sum(1 for r in sub if r["sides"][k] == +1),
                    "in": sum(1 for r in sub if r["sides"][k] == 0),
                    "unmeasurable": sum(1 for r in sub if r["sides"][k] is None)}
        return d

    out = {"bands": {k: list(v) for k, v in bands.items()},
           "alt_band_iqr": {k: list(v) for k, v in alt.items()},
           "total": tally(rows),
           "by_slice": {s: tally([r for r in rows if r["slice"] == s]) for s in ("W", "M")},
           "by_label": {str(v): tally([r for r in rows if r["label"] == v]) for v in (3, 4)},
           "by_mode": {m: tally([r for r in rows if r["mode"] == m])
                       for m in sorted({r["mode"] for r in rows if r["mode"]})},
           "by_family": {f: tally([r for r in rows if r["family"] == f])
                         for f in sorted({r["family"] for r in rows if r["family"]})}}
    return out


# =========================================================================== #
# 10. Smooth reach — the same three arms on the base carrier, by CACHED-FIELD recolor.
#
# `smooth` is a `pure` mode (`mining_roster.SMOOTH_KIND`): the field is dumped once per
# location by the engine and the whole coloring tail runs in Python (`colormap.render_candidate`),
# so all three arms are three LUTs over ONE field — no re-iteration, and the arms differ by
# nothing but the palette, which is the property the whole lever rests on.
# =========================================================================== #
def _smooth_population(sspec: BandSpec) -> list:
    """~`smooth_n` strong-q3/q4 smooth wallpapers: label 4 first, one per location, spread
    across families and palettes so the reach is not one family's opinion."""
    rows = [r for r in _census_rows(sspec) if r["slice"] == "W"]
    by_batch = {}
    import glob
    for b in sorted(glob.glob(str(ROOT / "data/wallpaper_corpus/batches/*"))):
        b = Path(b)
        if b.name not in sspec.smooth_batches:
            continue
        for line in (b / "images.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_batch[r["image_id"]] = (b, r)
    cand = [r for r in rows if r["image_id"] in by_batch]
    cand.sort(key=lambda r: (-r["label"], r["family"], r["image_id"]))
    picked, seen_f, seen_p = [], {}, set()
    for r in cand:
        b, row = by_batch[r["image_id"]]
        if seen_f.get(r["family"], 0) >= 2 or r["palette"] in seen_p:
            continue
        seen_f[r["family"]] = seen_f.get(r["family"], 0) + 1
        seen_p.add(r["palette"])
        picked.append({"example_id": r["image_id"], "label": r["label"], "mode": "smooth",
                       "fractal_type": r["family"], "palette": r["palette"],
                       "batch": b.name, "corpus_crop": str(Path(r["jpg"])
                                                           .relative_to(ROOT).as_posix()),
                       "render": row["render"],
                       "color_params": row["provenance"]["params"]})
        if len(picked) >= sspec.smooth_n:
            break
    return picked


def _smooth_one(job: tuple) -> dict:
    import location as loc_mod
    from tools import colormap as CM
    ex, sd, bands = job
    sspec = BandSpec(**{**sd, **{k: Path(sd[k]) for k in ("ref_dir", "out_dir", "v2_dir")}})
    out_dir = sspec.out_dir / "smooth"
    arms_dir = out_dir / "arms"
    for d in (arms_dir, out_dir / "_fields", out_dir / "_cmaps"):
        d.mkdir(parents=True, exist_ok=True)
    eid = ex["example_id"]
    t0 = time.time()
    rec = dict(ex)
    w, h, ss = sspec.geom
    loc = loc_mod.from_render_block(ex["render"])
    # ONE field dump per location; the three arms are three LUTs over it (`render_pure`'s
    # recipe, with `smooth`'s own field spec from the roster).
    from tools.mining.mining_roster import SMOOTH_FIELD_SPEC
    fbin = out_dir / "_fields" / f"{eid}.bin"
    if not fbin.exists():
        PA._run([PA.EXE, "render-one"] + loc_mod.render_one_flags(loc)
                + ["--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
                   "--maxiter", str(loc.maxiter), "--width", str(w), "--height", str(h),
                   "--supersample", str(ss), "--coloring", json.dumps(dict(SMOOTH_FIELD_SPEC)),
                   "--dump-field", str(fbin)], sspec.engine_threads)
    fld = CM.load_field(str(fbin))
    ow, oh = fld.out_size
    prep = CM.stretch_field(fld)
    cp = dict(ex["color_params"])
    lib_json = PA.load_library()
    entry = dict(lib_json[ex["palette"]])
    cyclic = lib_json[ex["palette"]].get("cycle") == "cyclic"

    def _render(cmaps_path: Path, tag: str) -> Path:
        library = CM.PaletteLibrary(colormaps_path=str(cmaps_path))
        cfg = CM.CandidateConfig(
            palette=ex["palette"], location=fld.location, eval_width=ow, eval_height=oh,
            reverse=bool(cp.get("reverse", False)),
            log_premap=cp.get("log_premap", "none"), gamma=float(cp.get("gamma", 1.0)),
            n_cycles=(int(cp.get("n_cycles", 1)) if cyclic else 1),
            phase=(float(cp.get("phase", 0.0)) if cyclic else 0.0),
            transfer=cp.get("transfer", "pct"),
            transfer_gamma=float(cp.get("transfer_gamma", 0.0)),
            interior_color=tuple(cp.get("interior_color", (0.0, 0.0, 0.0))),
            filter=sspec.filt)
        prof = (CM.gradient_transfer_profile(fld, prep)
                if cfg.transfer == "grad" else None)
        img = CM.render_candidate(fld, cfg, library, prep=prep, profile=prof)
        p = arms_dir / f"{eid}__{tag}.jpg"
        Image.fromarray(img).convert("RGB").save(p, "JPEG", quality=sspec.smooth_jpg_q)
        return p

    before = _render(PA.POOL_CMAPS, "before")
    img_b = np.asarray(Image.open(before).convert("RGB"))
    st = tone_stats(img_b)
    rec["before_stats"] = st
    rec["before_path"] = str(before.relative_to(ROOT).as_posix())
    crop = ROOT / ex["corpus_crop"]
    if crop.exists():
        a = np.asarray(Image.open(crop).convert("RGB")).astype(np.int16)
        if a.shape == img_b.shape:
            d = np.abs(a - img_b.astype(np.int16))
            rec["crop_parity"] = {"mean_abs": float(d.mean()), "max_abs": int(d.max())}

    for arm, cur in (("band", derive_band_curve(st, bands)),
                     ("autolevel_soft", PA.derive_curve(
                         {"black_pt": st["black_pt_all"], "white_pt": st["white_pt"],
                          "mid": st["mid"]}, PA._soft_ends()))):
        rec["curve_band" if arm == "band" else "curve_soft"] = cur
        if not cur["applies"]:
            continue
        e = dict(entry)
        if arm == "band":
            e["stops"], n_capped = curved_stops(entry["stops"],
                                                bool(entry.get("mirror_needed")), cur)
            rec["n_chroma_capped"] = n_capped
            rec["n_stops"] = len(e["stops"])
        else:
            e["stops"] = PA.curved_stops(entry["stops"], bool(entry.get("mirror_needed")), cur)
        cmp_path = out_dir / "_cmaps" / f"{eid}__{arm}.json"
        cmp_path.write_text(json.dumps([e]), encoding="utf-8")
        p = _render(cmp_path, arm)
        rec[f"{arm}_stats"] = tone_stats(np.asarray(Image.open(p).convert("RGB")))
        rec[f"{arm}_path"] = str(p.relative_to(ROOT).as_posix())
    rec["secs"] = time.time() - t0
    return rec


def cmd_smooth(sspec: BandSpec, args):
    bands = load_bands(sspec)
    pop = _smooth_population(sspec)
    if args.limit:
        pop = pop[:args.limit]
    (sspec.out_dir / "smooth").mkdir(parents=True, exist_ok=True)
    (sspec.out_dir / "smooth" / "population.json").write_text(json.dumps(pop, indent=1),
                                                              encoding="utf-8")
    print(f"smooth population {len(pop)}: " +
          ", ".join(f"{p['example_id']}({p['label']},{p['fractal_type']})" for p in pop))
    recs, t0 = [], time.time()
    jobs = [(ex, _sd(sspec), bands) for ex in pop]
    with ProcessPoolExecutor(max_workers=sspec.workers) as px:
        futs = {px.submit(_smooth_one, j): j[0]["example_id"] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                recs.append(fu.result())
            except Exception as e:                                    # noqa: BLE001
                print(f"[fail] {futs[fu]}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"[{i}/{len(jobs)}] {futs[fu]} {recs[-1]['secs']:.1f}s "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    recs.sort(key=lambda r: (-r["label"], r["example_id"]))
    p = sspec.out_dir / "smooth" / "records.json"
    p.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    out = sspec.out_dir / "sheet_smooth.png"
    build_sheet(recs, f"smooth (base carrier) — before | BAND rule | v2 autolevel_soft   "
                      f"({len(recs)} strong-q3/q4 wallpapers, cached-field recolor, "
                      f"{sspec.geom[0]}x{sspec.geom[1]} ss{sspec.geom[2]}, literal bytes)",
                out, sspec, lambda r, a: ROOT / r[f"{a}_path"] if f"{a}_path" in r else None)
    print("wrote", p, "and", out)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["reference", "render", "score", "summary", "sheet",
                                    "census", "smooth", "all"])
    ap.add_argument("--study", default="band_v1")
    ap.add_argument("--limit", type=int, default=0,
                    help="bounded end-to-end: N examples per mode (writes *_limited.json)")
    args = ap.parse_args()
    sspec = SPECS[args.study]
    if args.cmd == "reference":
        measure_reference(sspec)
    if args.cmd == "census":
        cmd_census(sspec, args)
    if args.cmd == "smooth":
        cmd_smooth(sspec, args)
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
