r"""autolevel.py — THE band-targeting auto-level operator. **SWITCH SHIPS ON** (2026-08-11).

The production port of the rule the band study proposed and Matt adopted
(`scratch/autolevel_band_study_report.md`, commit `1b4af7a`). This module is the ONE owner
of the rule: the study driver (`tools/studies/palette_autolevel_band.py`) re-exports from
here rather than keeping its own copy, so the rule that was measured and the rule that ships
cannot drift apart.

THE LEVER, in one sentence. A render's per-pixel colour is `palette.lookup_linear(tt)`, so
pushing a monotone tone curve `C` through the LUT's OKLab L moves EVERY pixel's lightness by
exactly `C` — which is why the correction is MEASURED on the rendered bytes and APPLIED on
the palette. The two are the same map (exactly per subpixel, approximately after AA).

THE RULE (unchanged from the study; the numbers are read off the committed reference record,
never invented here):
  * Three statistics per render — black point (P0.5 of L over NEUTRAL pixels), white point
    (P99.5 of L), midtone (median L over the structure mask L > MASK_L).
  * Each is PROJECTED onto its band: inside -> itself, outside -> the NEAREST edge. That is
    the minimum move that reaches the acceptable set, so the pull needs no free parameter.
  * IN BAND ON ALL THREE -> EXACTLY THE IDENTITY, and `maybe_level` then returns the base
    render's own bytes rather than re-rendering it. Being always-on but mostly identity is
    structural here, not a hope.
  * The curve is PIECEWISE with linear tails ([0,b]->[0,lo], core, [w,1]->[hi,1]), so 0->0
    and 1->1 always: a true black is never lifted, a true white never dimmed.
  * CHROMA GUARD, two mechanisms. MEASUREMENT — the black point is read over neutral pixels
    and declared UNMEASURABLE (that end left alone) when the neutral subset is thin or its
    own black sits far above the all-pixel one; it can only ever turn a correction OFF.
    APPLICATION — a per-LUT-entry cap walks each stop's new lightness back until at least
    `CHROMA_RETAIN` of its chroma survives the gamut pullback.

WHAT IS NOT TOUCHED. The Rust<->Python LUT seam. The surgery is Python-side and ends in a
STOP LIST: for the Python coloring tail it goes through `OverrideLibrary` (which bakes with
`colormap.build_lut`, the same bake), and for the Rust engine through a one-entry colormap
JSON handed to `render-one --colormaps` — so the bake, the mirror flag and the spec stay
bit-identical to the production call and only the stop COLOURS differ.

THE SWITCH is `enabled()` and it is ONE switch: `SWITCH_DEFAULT` (True since the 2026-08-11
adoption — the build+stage was `43a9328`, the flip is recorded in
`data/palettes/autolevel_adoption.json`) overridden per-run by the `FRACTAL_AUTOLEVEL`
environment variable. Read at CALL time, never at import, so a test monkeypatches it the same
way `floors.active_head_version` is monkeypatched. Every wired production render goes through
`maybe_level` and nothing else — a call site that reaches `plan()` directly would be a second
switch. The OFF path is still a CONTRACT, not dead code: `FRACTAL_AUTOLEVEL=0` returns the
base render's own object with no stamp, no reference load and no re-render, which is how a
before/after pair is produced and how a leveled row is falsified.

    uv run python tools/palettes/autolevel.py            # print the switch + the live band
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.palettes.color import oklab_to_srgb, srgb_to_oklab      # noqa: E402
# The row-wise twins below re-do `color.py`'s two conversions in a stacked-matmul form; they
# BORROW its constants rather than restating them, because a twin built on a second copy of
# the Ottosson matrices is not a twin (and `color.py` already carries four hand-synced copies
# it explicitly refuses to grow). Aliased once here so the borrow is visible.
from tools.palettes.color import (                                 # noqa: E402
    _M1 as _COLOR_M1, _M1_INV as _COLOR_M1_INV,
    _M2 as _COLOR_M2, _M2_INV as _COLOR_M2_INV,
    linear_to_srgb, srgb_to_linear)


# =========================================================================== #
# 0. THE SWITCH.
# =========================================================================== #
OPERATOR_VERSION = "band_autolevel/v1"

SWITCH_ENV = "FRACTAL_AUTOLEVEL"
SWITCH_DEFAULT = True           # <-- THE SWITCH. ADOPTED 2026-08-11; the flip is its own
                                # decision and its own record: data/palettes/autolevel_adoption.json.

_TRUTHY = ("1", "true", "yes", "on")


def enabled() -> bool:
    """Is the operator on for THIS render? Read at call time.

    One switch with two ways to set it, not two switches: `SWITCH_DEFAULT` is what the tree
    ships (ON since the 2026-08-11 adoption), and `FRACTAL_AUTOLEVEL` sets it for one run
    without editing source — which is how the verification sheet runs both arms and how the
    OFF path stays exercisable after the flip. An unparseable value reads as the DEFAULT and
    never as its own state: a typo must not move a production colorize in either direction,
    and after the flip that means it must not silently turn one OFF either."""
    raw = os.environ.get(SWITCH_ENV)
    if raw is None:
        return SWITCH_DEFAULT
    raw = raw.strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in ("0", "false", "no", "off", ""):
        return False
    return SWITCH_DEFAULT


# =========================================================================== #
# 1. Measurement (screen space) — the statistics the band is read on.
# =========================================================================== #
CLIP_LO, CLIP_HI = 0.5, 99.5    # robust black/white percentiles of OKLab L
MASK_L = 0.04                   # OKLab L floor of the "structure" mask (the midtone's mask)
CHROMA_NEUTRAL = 0.06           # OKLab chroma at or below which a pixel reads as neutral
NEUTRAL_FRAC_MIN = 0.05         # a neutral subset thinner than this reads no black point
DARK_MARGIN = 0.10              # neutral black this far ABOVE the all-pixel black => the dark
                                # tail is chromatic -> unmeasurable
CHROMA_RETAIN = 0.85            # per-entry: post-pullback chroma must keep this share
EXP_CLAMP = 2.0                 # p in [0.5, 2.0]
MIN_RANGE = 0.05                # w-b below this => degenerate, no curve proposed
DENSIFY = 8                     # stop-subdivision factor before the curve is applied
STAT_KEYS = ("black_pt", "white_pt", "mid")


def tone_stats(img: np.ndarray) -> dict:
    """One rendered image (uint8 HxWx3) -> the statistics the band is read on.

    `black_pt` is the GUARDED black point and is the one the rule acts on; `black_pt_all`
    (every pixel) is kept beside it so a record can say which renders the guard silenced and
    by how much."""
    lab = srgb_to_oklab(np.asarray(img).astype(np.float32) / 255.0)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    mask = L > MASK_L
    b_all = float(np.percentile(L, CLIP_LO))
    w = float(np.percentile(L, CLIP_HI))
    m = float(np.median(L[mask])) if mask.any() else float(np.median(L))
    neutral = C <= CHROMA_NEUTRAL
    nfrac = float(neutral.mean())
    b_neu = float(np.percentile(L[neutral], CLIP_LO)) if neutral.sum() > 64 else None
    if b_neu is None or nfrac < NEUTRAL_FRAC_MIN:
        black, why = None, f"neutral pixels {nfrac:.3f} < {NEUTRAL_FRAC_MIN}"
    elif b_neu - b_all > DARK_MARGIN:
        black, why = None, f"chromatic dark tail (neutral black {b_neu:.3f} vs all {b_all:.3f})"
    else:
        black, why = b_neu, None
    return {"black_pt": black, "black_unmeasurable": why, "black_pt_all": b_all,
            "black_pt_neutral": b_neu, "neutral_frac": nfrac,
            "white_pt": w, "mid": m,
            "mask_frac": float(mask.mean()),
            "in_mask_chroma": float(C[mask].mean()) if mask.any() else 0.0,
            "dark_chroma": float(C[L <= np.percentile(L, 2.0)].mean()),
            "L_mean": float(L.mean())}


# =========================================================================== #
# 2. The committed reference record (the band). READ here; DERIVED by
#    `tools/palettes/levels_reference.py`, which is the only thing that writes it.
# =========================================================================== #
RECORD_PATH = "data/palettes/levels_reference.json"

_REF_CACHE: dict = {}


def load_reference(path: str | Path | None = None) -> dict:
    """The committed reference record + its identity (`_sha256`, `_path`), memoized on path.

    Memoized because a colorize pass calls this once per render and the record is a frozen
    file; the cache key is the resolved path, so a test pointing at a fixture never sees the
    production record's entry."""
    p = Path(path) if path else (ROOT / RECORD_PATH)
    key = str(p.resolve())
    hit = _REF_CACHE.get(key)
    if hit is not None:
        return hit
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing. It is the committed, tracked levels reference; derive it with "
            f"`uv run python tools/palettes/levels_reference.py --write` (it reads the "
            f"read-only LevelsCheck folder).")
    raw = p.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    doc["_sha256"] = hashlib.sha256(raw).hexdigest()
    doc["_path"] = RECORD_PATH if path is None else str(p)
    _REF_CACHE[key] = doc
    return doc


def bands(ref: dict) -> dict:
    """{statistic: (lo, hi)} from a loaded record. A statistic with no band (nothing
    measurable across the reference set) is ABSENT rather than defaulted — `derive_band_curve`
    then leaves that end alone, which is the same refusal the chroma guard makes."""
    return {k: tuple(ref["bands"][k]["band"]) for k in STAT_KEYS if ref["bands"].get(k)}


# =========================================================================== #
# 3. The band rule.
#
# CURVE, piecewise on [0,1] and continuous:
#     L <= b        L * lo/b                              (tail: 0 -> 0)
#     b < L < w     lo + (hi-lo) * ((L-b)/(w-b))**p       (core)
#     L >= w        hi + (1-hi) * (L-w)/(1-w)             (tail: 1 -> 1)
# with lo = proj(b), hi = proj(w) and p solving C(m) = proj(m). Identity iff all three
# statistics are in band. The black end is skipped entirely (lo = b) when the chroma guard
# calls it unmeasurable.
# =========================================================================== #
def project(x: float, band: tuple) -> tuple:
    """(projected value, side) — side is -1 below the band, +1 above, 0 inside."""
    lo, hi = band
    if x < lo:
        return lo, -1
    if x > hi:
        return hi, +1
    return x, 0


def derive_band_curve(st: dict, band_map: dict) -> dict:
    b_meas, w, m = st["black_pt"], st["white_pt"], st["mid"]
    b = b_meas if b_meas is not None else st["black_pt_all"]
    if w - b < MIN_RANGE:
        return {"applies": False, "reason": f"degenerate range w-b={w - b:.3f} < {MIN_RANGE}",
                "black_pt": b, "white_pt": w, "exponent": 1.0, "out_ends": [b, w]}
    if b_meas is None:
        lo, side_b = b, 0                      # chroma guard: black end left alone
    else:
        lo, side_b = project(b_meas, band_map["black_pt"])
    hi, side_w = project(w, band_map["white_pt"])
    m_t, side_m = project(m, band_map["mid"])
    t = (m - b) / (w - b)
    tgt_n = (m_t - lo) / (hi - lo) if hi > lo else 0.5
    if not (1e-4 < t < 1 - 1e-4) or not (1e-4 < tgt_n < 1 - 1e-4):
        p = 1.0
    else:
        p = float(np.log(tgt_n) / np.log(t))
    p = float(np.clip(p, 1.0 / EXP_CLAMP, EXP_CLAMP))
    ident = (abs(lo - b) < 1e-9 and abs(hi - w) < 1e-9 and abs(p - 1.0) < 1e-9)
    return {"applies": True, "reason": None, "black_pt": b, "white_pt": w, "mid_in": m,
            "exponent": p, "out_ends": [lo, hi], "mid_target": m_t, "mid_norm": t,
            "sides": {"black_pt": side_b, "white_pt": side_w, "mid": side_m},
            "black_guarded": b_meas is None, "identity": ident,
            "clamped": abs(p - EXP_CLAMP) < 1e-9 or abs(p - 1.0 / EXP_CLAMP) < 1e-9}


def apply_curve_L(L: np.ndarray, cur: dict) -> np.ndarray:
    """The piecewise curve. Exact identity when out_ends == (black_pt, white_pt) and p == 1."""
    b, w, p = cur["black_pt"], cur["white_pt"], cur["exponent"]
    lo, hi = cur["out_ends"]
    L = np.asarray(L, dtype=np.float64)
    out = np.empty_like(L)
    below = L <= b
    above = L >= w
    core = ~(below | above)
    out[below] = L[below] * (lo / b) if b > 1e-9 else lo
    t = (L[core] - b) / max(w - b, 1e-9)
    out[core] = lo + (hi - lo) * t ** p
    out[above] = (hi + (1.0 - hi) * (L[above] - w) / (1.0 - w)) if w < 1.0 - 1e-9 else hi
    return out


# =========================================================================== #
# 4. LUT surgery: densify -> curve -> per-entry chroma cap -> gamut fit.
#
# `stops` are the colormap-library `[pos, [r,g,b]]` pairs, i.e. exactly what
# `palette_pick::parse_colormaps` hands to `Palette::from_srgb8_stops_mirrored`.
# =========================================================================== #
def densify(stops: list, mirror: bool, k: int = DENSIFY) -> list:
    """Subdivide each segment `k`-fold, interpolating in OKLab — the SAME space and the same
    piecewise-linear rule `interp_oklab_cyclic` uses, so this is identity for the palette
    (to sRGB8 rounding) and only makes the tone curve's sampling finer.

    The wrap segment (last->first, through pos 1) is subdivided ONLY for cyclic maps: a
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
    an in-gamut color survives lab->sRGB->lab unchanged, an out-of-gamut one does not. Asking
    the clipped output whether it is in range instead always answers yes — that bug silently
    hard-clipped the darkened stops and cost the curve ~0.06 mean L of fidelity."""
    rgb = oklab_to_srgb(lab.reshape(1, 3)).reshape(3)
    back = srgb_to_oklab(rgb.reshape(1, 3)).reshape(3)
    return float(np.max(np.abs(back - lab))) < 1e-6, rgb


def _gamut_fit(lab: np.ndarray) -> list:
    """OKLab -> sRGB8, pulling an out-of-gamut color back by CHROMA reduction (bisection on
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


def _chroma_after(lab: np.ndarray) -> float:
    """Chroma that SURVIVES the gamut pullback at this (L, a, b) — i.e. what `_gamut_fit`
    actually emits, measured back in OKLab so the number is the same quantity as the input."""
    rgb = np.array(_gamut_fit(lab), dtype=np.float64) / 255.0
    out = srgb_to_oklab(rgb.reshape(1, 3)).reshape(3)
    return float(np.hypot(out[1], out[2]))


def cap_lightness(L: float, Lp: float, a: float, b: float, retain: float = CHROMA_RETAIN,
                  iters: int = 18) -> tuple:
    """Walk the new lightness back toward the original until the post-pullback chroma keeps
    at least `retain` of the original. Returns (lightness, capped?).

    Only the direction that COSTS chroma is capped, and the walk-back target is the entry's
    own original lightness — where retention is 1.0 by construction (the stop came from a
    real sRGB8 colour), so the bisection always has a valid bracket."""
    c0 = float(np.hypot(a, b))
    if c0 < 1e-6 or abs(Lp - L) < 1e-9:
        return Lp, False
    if _chroma_after(np.array([Lp, a, b])) >= retain * c0:
        return Lp, False
    good, bad = L, Lp                                  # good retains, bad does not
    for _ in range(iters):
        mid = 0.5 * (good + bad)
        if _chroma_after(np.array([mid, a, b])) >= retain * c0:
            good = mid
        else:
            bad = mid
    return good, True


# --------------------------------------------------------------------------- #
# The ROW-WISE twins of the four scalar helpers above, and the reason they exist.
#
# `curved_stops` is a pure function of (stops, mirror, curve) — it never sees the image — so
# it costs the SAME on a 960x540 colorize render and a 2560x1440 release render, and a pool
# palette carries 33-512 stops that `densify(k=8)` turns into up to 4096 entries. Run per
# entry, each entry's `cap_lightness` is up to 18 bisection steps, each step a `_chroma_after`
# that is itself a `_gamut_fit` of up to 29 `_in_gamut` calls — i.e. up to ~17k numpy calls on
# (1,3) arrays for ONE stop. Measured on the live path (`scratch/colorize_trace/`): 0.13-24.6 s
# per acting render, 31% of every second of a colorize attempt, the largest single term in the
# stage and larger than the engine on the smooth styles.
#
# THE IDENTITY THIS RESTS ON, measured before any of it was written. A plain `(N,3) @ (3,3)`
# matmul is NOT bit-identical to the per-row `(1,3) @ (3,3)` the scalar path does (max delta
# 1.6e-14 on the OKLab round trip, because BLAS blocks the general case differently) — but the
# STACKED `(N,1,3) @ (3,3)` form IS, exactly, because numpy dispatches each row as its own
# small matmul. So every conversion below goes through the stacked form; every bisection runs
# the same FIXED iteration count with masked updates; and the compressed subsets are safe for
# the same reason the batch is (a row's arithmetic cannot depend on who it is batched with).
# Each element therefore sees the identical sequence of float ops as the scalar path, which is
# why `test_autolevel.py` can assert EQUALITY of the stop lists over the whole pool and not a
# tolerance. That equality is the contract: these stops are baked into the LUT, so a one-LSB
# stop is a changed render and a changed head score.
#
# The scalar four are NOT dead and must stay: `tools/studies/palette_autolevel*.py` import
# them, and they are the reference the parity test compares against.
# --------------------------------------------------------------------------- #
def _rows_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """(N,3) sRGB in [0,1] -> (N,3) OKLab. Bit-identical to `color.srgb_to_oklab` called
    once per row on a (1,3) array — see the stacked-matmul note above."""
    lin = srgb_to_linear(np.asarray(srgb, dtype=np.float64))[:, None, :]
    return (np.cbrt(lin @ _COLOR_M1.T) @ _COLOR_M2.T)[:, 0, :]


def _rows_to_srgb(lab: np.ndarray) -> np.ndarray:
    """(N,3) OKLab -> (N,3) clipped sRGB. Bit-identical to per-row `color.oklab_to_srgb`."""
    lab = np.asarray(lab, dtype=np.float64)[:, None, :]
    lin = ((lab @ _COLOR_M2_INV.T) ** 3 @ _COLOR_M1_INV.T)[:, 0, :]
    return np.clip(linear_to_srgb(lin), 0.0, 1.0)


def _in_gamut_rows(lab: np.ndarray) -> tuple:
    """`_in_gamut` over (N,3): (ok (N,), srgb (N,3)). Same round-trip test, same 1e-6."""
    rgb = _rows_to_srgb(lab)
    back = _rows_to_oklab(rgb)
    return np.max(np.abs(back - lab), axis=-1) < 1e-6, rgb


def _gamut_fit_rows(lab: np.ndarray) -> np.ndarray:
    """`_gamut_fit` over (N,3) -> (N,3) int sRGB8. The 28-step chroma bisection runs on the
    COMPRESSED out-of-gamut subset; in-gamut rows keep their scale-1.0 colour untouched."""
    lab = np.asarray(lab, dtype=np.float64)
    ok, rgb = _in_gamut_rows(lab)
    idx = np.flatnonzero(~ok)
    if idx.size:
        sub = lab[idx]
        lo = np.zeros(idx.size)
        hi = np.ones(idx.size)
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            okm, _ = _in_gamut_rows(
                np.stack([sub[:, 0], sub[:, 1] * mid, sub[:, 2] * mid], axis=-1))
            lo = np.where(okm, mid, lo)
            hi = np.where(~okm, mid, hi)
        _, rgb_lo = _in_gamut_rows(
            np.stack([sub[:, 0], sub[:, 1] * lo, sub[:, 2] * lo], axis=-1))
        rgb[idx] = rgb_lo
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.int64)


def _chroma_after_rows(lab: np.ndarray) -> np.ndarray:
    """`_chroma_after` over (N,3) -> (N,)."""
    out = _rows_to_oklab(_gamut_fit_rows(lab).astype(np.float64) / 255.0)
    return np.hypot(out[:, 1], out[:, 2])


def _cap_lightness_rows(L, Lp, a, b, retain: float = CHROMA_RETAIN, iters: int = 18) -> tuple:
    """`cap_lightness` over N entries -> (lightness (N,), capped (N,) bool). The two scalar
    early-outs become the `cand`/`act` compressions, so only the entries that actually cap
    pay the 18-step walk-back — on a typical palette that is a small minority."""
    L = np.asarray(L, dtype=np.float64)
    Lp = np.asarray(Lp, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c0 = np.hypot(a, b)
    out = Lp.copy()
    capped = np.zeros(L.shape, dtype=bool)
    cand = np.flatnonzero((c0 >= 1e-6) & (np.abs(Lp - L) >= 1e-9))
    if cand.size:
        keep = _chroma_after_rows(
            np.stack([Lp[cand], a[cand], b[cand]], axis=-1)) >= retain * c0[cand]
        act = cand[~keep]
        if act.size:
            good, bad = L[act].copy(), Lp[act].copy()
            aa, bb, thr = a[act], b[act], retain * c0[act]
            for _ in range(iters):
                mid = 0.5 * (good + bad)
                ok = _chroma_after_rows(np.stack([mid, aa, bb], axis=-1)) >= thr
                good = np.where(ok, mid, good)
                bad = np.where(~ok, mid, bad)
            out[act] = good
            capped[act] = True
    return out, capped


def curved_stops(stops: list, mirror: bool, cur: dict) -> tuple:
    """Adjusted stop list + how many entries the chroma cap had to hold back.

    Vectorized over the densified entries through the row-wise helpers above; byte-identical
    to the per-entry form, which `test_autolevel.py` pins over the whole production pool."""
    dense = densify(stops, mirror)
    lab = np.array([l for _, l in dense], dtype=np.float64)
    Lp = apply_curve_L(lab[:, 0], cur)
    Lc, capped = _cap_lightness_rows(lab[:, 0], Lp, lab[:, 1], lab[:, 2])
    rgb = _gamut_fit_rows(np.stack([Lc, lab[:, 1], lab[:, 2]], axis=-1))
    out = [[round(float(p), 9), [int(v) for v in row]]
           for (p, _), row in zip(dense, rgb)]
    return out, int(capped.sum())


# =========================================================================== #
# 5. The stamp — what a row produced with the operator ON carries.
#
# TWO THINGS IT HAS TO SUPPORT, and both are checked by `test_autolevel.py`:
#   REPLAY — `stops_from_stamp(stamp, entry)` rebuilds the exact stop list from the stamp
#     alone, with no image and no re-measurement, so a release replays byte-identically off
#     its record. That is why the whole `curve` block is stamped and not just a summary.
#   THE "BEFORE" — `measured` is the base render's own statistics. The before render is the
#     same recipe with the switch OFF (the operator is additive and changes nothing else), and
#     `measured` is what proves a candidate re-render IS that render.
# =========================================================================== #
def make_stamp(ref: dict, cur: dict, st: dict, *, n_capped: int, n_stops: int,
               acted: bool) -> dict:
    return {
        "operator": OPERATOR_VERSION,
        "switch": "on",
        "acted": bool(acted),
        "reference": {
            "path": ref.get("_path", RECORD_PATH),
            "version": ref.get("version"),
            "schema": ref.get("schema"),
            "sha256": ref.get("_sha256"),
            "derived": ref.get("derived"),
            "n_images": ref.get("n_images"),
            "bands": {k: list(v) for k, v in bands(ref).items()},
        },
        "curve": {k: cur.get(k) for k in
                  ("applies", "reason", "identity", "black_pt", "white_pt", "mid_in",
                   "exponent", "out_ends", "mid_target", "mid_norm", "sides",
                   "black_guarded", "clamped")},
        "chroma_cap": {"retain": CHROMA_RETAIN, "n_capped": int(n_capped),
                       "n_stops": int(n_stops)},
        "measured": st,
        "before_recovery": ("the BEFORE render is this same recipe with the switch OFF; "
                            "`measured` are that render's own statistics, so a candidate "
                            "re-render can be checked against them"),
    }


def stops_from_stamp(stamp: dict, entry: dict) -> list:
    """Replay: the leveled stop list from the STAMP alone (no image, no re-measurement).

    Raises on a stamp that did not act — there is no curved stop list for an identity row,
    and returning the palette's own stops would quietly make "replayed" mean two things."""
    if not stamp.get("acted"):
        raise ValueError("stamp records an IDENTITY (or non-applying) row: the render is the "
                         "base palette's own, there is no curved stop list to replay")
    stops, _ = curved_stops(entry["stops"], bool(entry.get("mirror_needed")), stamp["curve"])
    return stops


# =========================================================================== #
# 6. The Python-tail application: a PaletteLibrary with ONE palette's stops replaced.
# =========================================================================== #
class OverrideLibrary:
    """`colormap.PaletteLibrary` with one palette's stops swapped for the leveled ones.

    Everything else is delegated, and the bake goes through `colormap.build_lut` — the same
    function, the same memo, the same mirror flag — so the LUT seam is untouched and only the
    stop COLOURS differ. Deliberately not a subclass: it wraps whatever library the call site
    already built (pool colormaps + features), rather than re-resolving those paths."""

    def __init__(self, lib, name: str, stops: list, mirror: bool):
        self._lib = lib
        self._name = name
        self._stops = [(p, rgb) for p, rgb in stops]
        self._mirror = bool(mirror)

    @property
    def colormaps(self):
        return self._lib.colormaps

    def palette_type(self, name):
        return self._lib.palette_type(name)

    def lut(self, name, reverse=False):
        if name != self._name:
            return self._lib.lut(name, reverse=reverse)
        from tools import colormap as cm                      # noqa: PLC0415
        return cm.build_lut(self._stops, reverse=reverse, mirror=self._mirror)


# =========================================================================== #
# 7. THE production entry. Every wired call site calls this and nothing else.
# =========================================================================== #
STAMP_LOG = "autolevel_stamps.jsonl"


@dataclass(frozen=True)
class Leveled:
    """What one leveled render came out as: the image, and the stamp (None = switch OFF)."""
    img: np.ndarray
    stamp: dict | None

    @property
    def acted(self) -> bool:
        return bool(self.stamp and self.stamp.get("acted"))


def maybe_level(base_img, entry: dict, rerender, *, key: str | None = None,
                log_dir=None, reference: dict | None = None) -> Leveled:
    """THE switch, and the whole operator behind it.

    `base_img`  the render the call site already produced with the unmodified palette.
    `entry`     the palette's colormap-library entry (`stops` + `mirror_needed`).
    `rerender`  `stops -> image`; called ONLY when the curve actually acts, so an in-band
                render costs nothing and comes back as its own bytes.
    `key`/`log_dir`  where the stamp is recorded (`<log_dir>/autolevel_stamps.jsonl`, one row
                per leveled render). Doing it here rather than at each call site is what makes
                "every row produced with the operator ON is stamped" true by construction
                instead of by four remembered edits.

    Returns a `Leveled`. With the switch OFF this is `(base_img, None)` and `rerender` is
    never called — the OFF path is the pre-operator path with one boolean read in front of it.
    """
    if not enabled():
        return Leveled(base_img, None)
    ref = reference if reference is not None else load_reference()
    st = tone_stats(base_img)
    cur = derive_band_curve(st, bands(ref))
    acts = bool(cur.get("applies")) and not cur.get("identity")
    if not acts:
        stamp = make_stamp(ref, cur, st, n_capped=0, n_stops=0, acted=False)
        _log_stamp(log_dir, key, stamp)
        return Leveled(base_img, stamp)                # identity -> the base render's bytes
    stops, n_capped = curved_stops(entry["stops"], bool(entry.get("mirror_needed")), cur)
    img = rerender(stops)
    stamp = make_stamp(ref, cur, st, n_capped=n_capped, n_stops=len(stops), acted=True)
    _log_stamp(log_dir, key, stamp)
    return Leveled(img, stamp)


def append_stamp(log_dir, key, stamp: dict) -> None:
    """THE stamp-log writer, public because a render that does not write its own stamp needs
    somebody who does. `maybe_level` calls it for a render that levels in-process; the
    concurrent release pass calls it from the PARENT for a render that levelled in a worker,
    so `autolevel_stamps.jsonl` has exactly one writer either way and the row is the same row.
    Passing `log_dir=None` is how a call site says "not mine to write" and is a no-op here."""
    if log_dir is None:
        return
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / STAMP_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "autolevel": stamp}) + "\n")


def _log_stamp(log_dir, key, stamp: dict) -> None:
    append_stamp(log_dir, key, stamp)


def one_entry_colormaps(entry: dict, stops: list, out_path) -> Path:
    """The Rust-side application: the SAME library entry with the leveled stops, written as a
    one-entry colormap JSON for `render-one --colormaps`. The name, the `cycle` field and
    `mirror_needed` all ride along unchanged, so the engine's bake and mirror decision are
    bit-identical to the production call and only the stop colours differ."""
    e = dict(entry)
    e["stops"] = stops
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([e]), encoding="utf-8")
    return p


def main() -> int:
    ref = load_reference()
    print(f"operator {OPERATOR_VERSION} · switch {'ON' if enabled() else 'OFF'} "
          f"(default {SWITCH_DEFAULT}, env {SWITCH_ENV}={os.environ.get(SWITCH_ENV)!r})")
    print(f"reference {ref['_path']} · {ref.get('version')} · n={ref.get('n_images')} "
          f"· derived {ref.get('derived')} · sha256 {ref['_sha256'][:16]}…")
    for k, v in bands(ref).items():
        print(f"  {k:9s} band [{v[0]:.4f}, {v[1]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
