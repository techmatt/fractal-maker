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

One thing lives here: a **single-pass, same-rays** computation of BOTH ring
measures, `radial_rings` (median colour-cycle *crossings*) and `radial_range`
(median *monotone range*), in one ray walk so the two see byte-identical rays.
What each measures, why the pair is not redundant, and where they disagree:
**`docs/design/orbital_field_metrics.md`** §§1,5.

DELETED on 2026-07-31: a scoring-only cap policy `scoring_maxiter(fw)` (plus
`prod_maxiter` and the `scoring_cap.json` loader beside it). It had no caller
anywhere in the tree, and the number it returned was not the one the
`scratch/rescore/` evidence was computed under: that evidence used a fitted
24x-of-legacy-production envelope clamped at 67000, while the committed module found
no `scoring_cap.json` and fell back to 8x of the **raised** production cap — 200000
at `fw = 8e-10` against production's 42165. A dead function returning a wrong number
is a trap for its first real caller, and giving it a caller would have meant adopting
a scoring cap policy that was deliberately never adopted.

The belief that made it look load-bearing is worth keeping even though the code is
gone: `converge.py` FIT that 24x / clamp-67000 envelope and wrote it to
`scratch/rescore/scoring_cap.json`, nothing outside that scratch dir ever read it,
and it became a stated property of the system anyway. **`tools/orbital/` has never
run a 24x policy** — it measures at 1x through `rc.auto_maxiter`. See
docs/design/auto_maxiter.md and docs/design/storage_classes.md ("a proposal must
never leave scratch/ as a fact").
"""
from __future__ import annotations

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

DENSITY = fm.DENSITY


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
                 threads=None, tmpdir=None) -> dict:
    """Dump one field at `maxiter` and return both measures + a few field stats.

    `threads=None` takes the committed single-process engine default
    (`corpus_common.DEFAULT_ENGINE_THREADS`, paired with BELOW_NORMAL). A caller that
    fans out engine processes must size and pass its own — there is no standing number
    for that case."""
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
# self-check fixture: crossings from ring_measures must match fm.radial_rings exactly.
# The assertion itself now lives in tools/orbital/test_rescore_lib.py (a `__main__`
# self-check no suite runs is a memory of a test, not a test); this stays because the
# synthetic ramp field it builds is the fixture both the test and a hand-run share.
# --------------------------------------------------------------------------- #
def selfcheck_field(inner: float, *, seed: int = 0) -> np.ndarray:
    """A 320x180 radial ramp from `inner` at the centre to 100 at the corner, with a
    circular NaN interior island and light noise — enough structure that crossings and
    span are both nonzero and that segment handling is exercised."""
    rng = np.random.default_rng(seed)
    fy = (np.arange(180) + 0.5) / 180 - 0.5
    fx = (np.arange(320) + 0.5) / 320 - 0.5
    r = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * (180 / 320)) ** 2)
    r /= r.max()
    f = (inner + (100.0 - inner) * r).astype("f4")
    f[r < 0.1] = np.nan
    return f + rng.normal(0, 3, f.shape).astype("f4")


if __name__ == "__main__":
    for inner in (300.0, 2000.0, 8000.0):
        f = selfcheck_field(inner)
        want, got = fm.radial_rings(f), ring_measures(f)
        assert abs(got["radial_rings"] - want[0]) < 1e-9, (got["radial_rings"], want[0])
        assert abs(got["radial_rings_p90"] - want[1]) < 1e-9
    print("selfcheck OK — ring_measures crossings == fm.radial_rings "
          "(the committed assertion is tools/orbital/test_rescore_lib.py)")
