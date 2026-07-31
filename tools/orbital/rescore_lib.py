#!/usr/bin/env python
"""Shared helpers for the re-score-at-a-non-clipping-cap study
(prompts/rescore_and_settle_measure.md).

PROMOTED from `scratch/rescore/rescore_lib.py` on 2026-07-31. It was written as
disposable analysis, but it is the only thing that produces
`data/orbital/maxiter_convergence_ladder.json` — the raw evidence behind the base
500 -> 4000 raise — and CLAUDE.md's rule is explicit: a file that is the sole
producer of a durable artifact is not scratch. Left in `scratch/` it would have
been one `rm -r scratch/*` away from making the x8 figure unreproducible in
principle as well as in practice.

Two things live here:

  * A **scoring-only cap policy** `scoring_maxiter(fw)`, decoupled from the
    production `render_core.auto_maxiter` (which governs corpus crops and must
    NOT move). Its parameters would be read from a `scoring_cap.json` beside this
    module; none exists and none ever will, so it always falls back to the fixed
    8x-of-production multiple. That is deliberate, and it is the correction to a
    belief worth naming: `converge.py` FIT a 24x / clamp-67000 "scoring envelope"
    and wrote it to `scratch/rescore/scoring_cap.json`, and nothing outside that
    scratch dir ever read it. **`tools/orbital/` has never run a 24x policy** —
    it measures at 1x through `rc.auto_maxiter`. See docs/design/auto_maxiter.md.

  * A **single-pass, same-rays** computation of BOTH ring measures, `radial_rings`
    (median colour-cycle *crossings*) and `radial_range` (median *monotone range*),
    in one ray walk so the two see byte-identical rays. What each measures, why the
    pair is not redundant, and where they disagree:
    **`docs/design/orbital_field_metrics.md`** §§1,5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "tools" / "orbital", REPO_ROOT / "tools" / "explorer",
          REPO_ROOT / "tools" / "corpus", REPO_ROOT / "tools" / "descent",
          REPO_ROOT / "tools" / "sources", REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import field_metrics as fm     # noqa: E402  (reused: dump_field, _crossings, DENSITY, geoms)
import render_core as rc       # noqa: E402  (production auto_maxiter — reference only)

DENSITY = fm.DENSITY
CAP_JSON = HERE / "scoring_cap.json"


# --------------------------------------------------------------------------- #
# scoring cap policy (decoupled from production auto_maxiter)
# --------------------------------------------------------------------------- #
def prod_maxiter(fw) -> int:
    """The production cap — reference only, never applied to a corpus crop here."""
    return rc.auto_maxiter(fw)


def _cap_params() -> dict:
    if CAP_JSON.exists():
        return json.loads(CAP_JSON.read_text(encoding="utf-8"))
    # pre-convergence fallback: 8x production, generous clamp.
    return {"policy": "fallback", "mult_of_prod": 8.0, "clamp_max": 200000}


def scoring_maxiter(fw) -> int:
    """Scoring-only iteration cap as a function of frame width.

    Policy is data-driven: converge.py fits it and writes scoring_cap.json.
    Supported shapes:
      * {"policy":"logform","base":B,"k":K,"fw_home":3.0,"clamp_min":..,"clamp_max":..}
          maxiter = B*(1 + K*log2(fw_home/fw)), clamped.  (production functional form,
          larger coefficient / base)
      * {"policy":"mult_of_prod","mult_of_prod":M,"clamp_max":..}
          maxiter = M * production_auto_maxiter(fw)
      * {"policy":"fallback",...}  same as mult_of_prod.
    """
    import math
    p = _cap_params()
    pol = p.get("policy")
    if pol == "logform":
        fwf = float(fw)
        home = float(p.get("fw_home", 3.0))
        ratio = home / fwf if fwf > 0 else 1.0
        lz = math.log2(ratio) if ratio > 0 else 0.0
        val = float(p["base"]) * (1.0 + float(p["k"]) * lz)
        val = max(float(p.get("clamp_min", 200)), min(float(p.get("clamp_max", 200000)), val))
        return int(val)
    # mult_of_prod / fallback
    m = float(p.get("mult_of_prod", 8.0))
    val = m * rc.auto_maxiter(fw)
    return int(min(float(p.get("clamp_max", 200000)), val))


# --------------------------------------------------------------------------- #
# both measures, one ray walk, identical geometry to fm.radial_rings
# --------------------------------------------------------------------------- #
def ring_measures(field: np.ndarray, *, n_rays=fm.N_RAYS) -> dict:
    """Return crossings and monotone-range measures over the same 64 rays.

    Ray geometry is byte-identical to `field_metrics.radial_rings`: rays run to
    the inscribed radius, NaN (interior) breaks a ray into segments, and each
    quantity is computed per segment. Crossings sum across segments (reusing
    `fm._crossings`); range takes the max segment span (a genuine radial
    excursion, not summed noise across an interior island).
    """
    h, w = field.shape
    cy_, cx_ = (h - 1) / 2.0, (w - 1) / 2.0
    r_max = min(cx_, cy_)
    n_s = max(32, int(r_max))
    rr = np.linspace(0.0, r_max, n_s)
    crossings, ranges = [], []
    for th in np.linspace(0.0, 2 * np.pi, n_rays, endpoint=False):
        xs = np.clip(np.round(cx_ + rr * np.cos(th)).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy_ + rr * np.sin(th)).astype(int), 0, h - 1)
        t = field[ys, xs] * DENSITY
        ok = np.isfinite(t)
        n_cross = 0
        seg_span_max = 0.0
        seg = []

        def flush(seg, n_cross, seg_span_max):
            if seg:
                n_cross += fm._crossings(seg)
                a = np.asarray(seg)
                seg_span_max = max(seg_span_max, float(a.max() - a.min()))
            return n_cross, seg_span_max

        for val, good in zip(t, ok):
            if good:
                seg.append(val)
            elif seg:
                n_cross, seg_span_max = flush(seg, n_cross, seg_span_max)
                seg = []
        n_cross, seg_span_max = flush(seg, n_cross, seg_span_max)
        crossings.append(n_cross)
        ranges.append(seg_span_max)
    if not crossings:
        return {"radial_rings": 0.0, "radial_rings_p90": 0.0,
                "radial_range": 0.0, "radial_range_p90": 0.0}
    return {
        "radial_rings": float(np.median(crossings)),
        "radial_rings_p90": float(np.percentile(crossings, 90)),
        "radial_range": float(np.median(ranges)),
        "radial_range_p90": float(np.percentile(ranges, 90)),
    }


def measure_both(cx, cy, fw, maxiter, *, family="mandelbrot",
                 width=fm.MEASURE_W, height=fm.MEASURE_H, ss=fm.MEASURE_SS,
                 threads=3, tmpdir=None) -> dict:
    """Dump one field at `maxiter` and return both measures + a few field stats."""
    import tempfile
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        out = Path(td) / "f.bin"
        field, side = fm.dump_field(cx, cy, fw, maxiter, out, width=width,
                                    height=height, ss=ss, family=family, threads=threads)
    m = ring_measures(field)
    v = field[np.isfinite(field)]
    m["cycles_spanned"] = float(fm.cycles_spanned(field))
    m["interior_fraction"] = float((~np.isfinite(field)).mean())
    m["escaped_px"] = int(v.size)
    m["smooth_max"] = (float(v.max()) if v.size else None)
    m["maxiter"] = int(maxiter)
    return m


# --------------------------------------------------------------------------- #
# self-check: crossings from ring_measures must match fm.radial_rings exactly
# --------------------------------------------------------------------------- #
def _selfcheck():
    rng = np.random.default_rng(0)
    for inner in (300.0, 2000.0, 8000.0):
        fy = (np.arange(180) + 0.5) / 180 - 0.5
        fx = (np.arange(320) + 0.5) / 320 - 0.5
        r = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * (180 / 320)) ** 2)
        r /= r.max()
        f = (inner + (100.0 - inner) * r).astype("f4")
        f[r < 0.1] = np.nan
        f += rng.normal(0, 3, f.shape).astype("f4")
        want = fm.radial_rings(f)
        got = ring_measures(f)
        assert abs(got["radial_rings"] - want[0]) < 1e-9, (got["radial_rings"], want[0])
        assert abs(got["radial_rings_p90"] - want[1]) < 1e-9
    print("selfcheck OK — ring_measures crossings == fm.radial_rings")


if __name__ == "__main__":
    _selfcheck()
    print("scoring_maxiter fallback @ fw=8e-10:", scoring_maxiter(8e-10),
          "  prod:", prod_maxiter(8e-10))
