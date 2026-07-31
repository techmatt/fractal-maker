#!/usr/bin/env python
"""Measure the **orbital radius / slow-falloff** property of a judging frame.

What these measures are, what they were validated against, which of them FAILED, and
what they are blind to: **`docs/design/orbital_field_metrics.md`**. Do not restate any
of it here.

Fields come from `render-one --dump-field --dump-field-source f64`: raw little-endian
f32, row-major, NaN where the pixel did not escape. **Source `f64` is deliberate** — it
is the fast escape-time backend's smooth channel, i.e. the one the actual render path
shades from. It carries a constant offset relative to the `beautiful` kernel, which is
irrelevant here because every measure below is built from *differences* of the field.

Emitted per field by `measure_field`: `cycles_spanned`, `radial_rings` (+ `_p90`),
`falloff_extent`, `interior_fraction` and its radial profile. `radial_rings` is the one
that survived validation; the others are recorded as failures.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "explorer"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))

import render_core as rc      # noqa: E402
import corpus_common as cc    # noqa: E402
import location as loc_mod    # noqa: E402

# The render path's shading constant. One colour cycle == 1/DENSITY iterations.
DENSITY = 0.025                     # src/cli.rs ShadeArgs::density default
CYCLE_ITERS = 1.0 / DENSITY         # 40


# --------------------------------------------------------------------------- #
# The iteration-CAP provenance axis.
# Why it exists and what it covers: `docs/design/orbital_field_metrics.md` §7
# (cap policy itself: `docs/design/auto_maxiter.md`).
#
# The load-bearing invariant, restated here because code depends on it: the token is
# the EMPTY string for the legacy policy, and a record with the key ABSENT is legacy
# by the same invariant — so records written before this axis existed read correctly
# instead of raising. `loc_mod.maxiter_policy_token` is reused verbatim rather than
# re-derived, so there is one definition of the axis.
# --------------------------------------------------------------------------- #
POLICY_KEY = "maxiter_policy_token"
LEGACY_POLICY_TOKEN = ""            # loc_mod.LEGACY_MAXITER_POLICY == (500, .30, 200, 8000)


class MaxiterPolicyMixError(RuntimeError):
    """Two orbital score records computed under different iteration-cap policies were
    about to be compared or pooled. They are not commensurable — the cap moves the
    measure — so this raises instead of returning a number that silently mixes them."""


def policy_token(policy=None) -> str:
    """The token for `policy` (`None` -> the LIVE production policy). Thin pass-through
    to `loc_mod.maxiter_policy_token` so there is one definition of the axis."""
    return loc_mod.maxiter_policy_token(policy)


def record_policy(rec: dict) -> str:
    """The cap policy a score record was computed under. A MISSING key means legacy —
    the same empty-string invariant the field-cache stems use, so records written
    before this axis existed read correctly instead of raising."""
    return rec.get(POLICY_KEY) or LEGACY_POLICY_TOKEN


def describe_policy(token: str) -> str:
    """Human name for a token, for error messages. The legacy policy's token is the
    empty string, which would otherwise print as nothing at all."""
    if token == LEGACY_POLICY_TOKEN:
        b, k, lo, hi = loc_mod.LEGACY_MAXITER_POLICY
        return f"legacy (base={b}, k={k}, clamp {lo}-{hi})"
    return token


def require_one_policy(*groups, what: str = "these records") -> str:
    """Assert every record across `groups` shares one cap policy; return that token.

    `groups` are iterables of score records, optionally `(label, records)` pairs so the
    error can say WHICH side carried which policy. Raises `MaxiterPolicyMixError`
    naming both policies and their counts. Call this at every point that COMPARES or
    POOLS orbital scores — a percentile over a resumed file, a reference-vs-population
    verdict, a drift ratio across maxiter multipliers."""
    seen: dict[str, dict] = {}
    for i, g in enumerate(groups):
        if isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], str):
            label, recs = g
        else:
            label, recs = f"group{i}", g
        for r in recs or ():
            e = seen.setdefault(record_policy(r), {"n": 0, "labels": [], "ids": []})
            e["n"] += 1
            if label not in e["labels"]:
                e["labels"].append(label)
            if len(e["ids"]) < 3 and r.get("id"):
                e["ids"].append(r["id"])
    if len(seen) <= 1:
        return next(iter(seen), policy_token())
    lines = []
    for tok, e in sorted(seen.items()):
        lines.append(f"    {describe_policy(tok)!r}: {e['n']} record(s) "
                     f"from {', '.join(e['labels'])}"
                     + (f" (e.g. {', '.join(e['ids'])})" if e["ids"] else ""))
    raise MaxiterPolicyMixError(
        f"orbital scores span {len(seen)} iteration-cap policies — refusing to "
        f"compare/pool {what}:\n" + "\n".join(lines) + "\n"
        "These numbers are not commensurable: the cap is an input to every field "
        "measure (see docs/design/auto_maxiter.md). Re-measure one side under the "
        "other's policy, or compare within a policy only."
    )

MEASURE_W, MEASURE_H, MEASURE_SS = 320, 180, 1     # validation fidelity
SCREEN_W, SCREEN_H, SCREEN_SS = 64, 36, 1          # screening fidelity
N_RAYS = 64
N_RADIAL_BINS = 40
FIELD_TIMEOUT_S = 60


# --------------------------------------------------------------------------- #
# field dump
# --------------------------------------------------------------------------- #
def dump_field(cx, cy, fw, maxiter, out: Path, *, width=MEASURE_W, height=MEASURE_H,
               ss=MEASURE_SS, family="mandelbrot", threads=None,
               timeout=FIELD_TIMEOUT_S) -> tuple[np.ndarray, dict]:
    """Render the smooth escape-time field and return `(array[h, w], sidecar)`.

    `threads=None` means the committed single-process engine default
    (`corpus_common.DEFAULT_ENGINE_THREADS` = 7, paired with BELOW_NORMAL priority) — a
    lone `dump_field` should not have to restate it. The fan-out callers here
    (`measure_atoms`, `screen_pool`, `measure_convergence_ladder`) pass `threads=THREADS`
    explicitly, which is required of them: the per-process 7 assumes it has the box."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(rc.RENDER_BIN), "render-one",
            "--cx", rc.dec_str(cx), "--cy", rc.dec_str(cy), "--fw", rc.dec_str(fw),
            "--width", str(width), "--height", str(height), "--supersample", str(ss),
            "--maxiter", str(int(maxiter)),
            "--dump-field", str(out), "--dump-field-source", "f64"]
    if family and family != "mandelbrot":
        argv += ["--family", str(family)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=cc.default_engine_env(threads=threads),
                          creationflags=cc.default_creationflags())
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "dump-field failed").strip()[:300])
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    a = np.fromfile(out, dtype="<f4")
    h, w = side["height"] * side["supersample"], side["width"] * side["supersample"]
    return a.reshape(h, w), side


# --------------------------------------------------------------------------- #
# measures
# --------------------------------------------------------------------------- #
def _plane_radius_grid(h: int, w: int) -> np.ndarray:
    """Per-pixel radius from the frame centre, in units of the FRAME WIDTH — so the
    numbers mean the same thing at every depth and aspect."""
    fy = (np.arange(h) + 0.5) / h - 0.5
    fx = (np.arange(w) + 0.5) / w - 0.5
    ay = fy * (h / w)                       # plane-y in frame-width units (16:9 aware)
    return np.hypot(ax := fx[None, :], ay[:, None]) if False else \
        np.sqrt(fx[None, :] ** 2 + ay[:, None] ** 2)


def cycles_spanned(field: np.ndarray) -> float:
    """(p95 - p05) of smooth_iter over escaping pixels, in colour cycles."""
    v = field[np.isfinite(field)]
    if v.size < 16:
        return 0.0
    return float((np.percentile(v, 95) - np.percentile(v, 5)) * DENSITY)


def radial_rings(field: np.ndarray, *, n_rays=N_RAYS) -> tuple[float, float]:
    """Median (and p90) colour-cycle crossings along rays from the centre outward.

    Rays run to the inscribed radius so every ray has the same length and none leaves
    the frame. Interior (NaN) samples break the ray into segments; crossings are counted
    within segments only, so a black island in the middle costs nothing but its own span.
    """
    h, w = field.shape
    cy_, cx_ = (h - 1) / 2.0, (w - 1) / 2.0
    r_max = min(cx_, cy_)
    n_s = max(32, int(r_max))
    rr = np.linspace(0.0, r_max, n_s)
    counts = []
    for th in np.linspace(0.0, 2 * np.pi, n_rays, endpoint=False):
        xs = np.clip(np.round(cx_ + rr * np.cos(th)).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy_ + rr * np.sin(th)).astype(int), 0, h - 1)
        t = field[ys, xs] * DENSITY
        ok = np.isfinite(t)
        n = 0
        seg = []
        for val, good in zip(t, ok):
            if good:
                seg.append(val)
            elif seg:
                n += _crossings(seg)
                seg = []
        if seg:
            n += _crossings(seg)
        counts.append(n)
    if not counts:
        return 0.0, 0.0
    return float(np.median(counts)), float(np.percentile(counts, 90))


def _crossings(seg) -> int:
    """Integer-boundary crossings of a monotone-ish scalar run == colour cycles seen."""
    if len(seg) < 2:
        return 0
    a = np.asarray(seg)
    return int(np.abs(np.floor(a[1:]) - np.floor(a[:-1])).sum())


def falloff_extent(field: np.ndarray, *, n_bins=N_RADIAL_BINS) -> float:
    """Radial span (in frame widths) over which the binned median smooth value descends
    from its inner plateau to background — measured 90% → 10% of the range, so a single
    outlier bin cannot set it. Slow, wide decoration = large; a thin skin = near zero."""
    h, w = field.shape
    r = _plane_radius_grid(h, w)
    r_max = float(r.max())
    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    v = field.ravel()
    med = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = v[(idx == b) & np.isfinite(v)]
        if sel.size >= 8:
            med[b] = np.median(sel)
    good = np.isfinite(med)
    if good.sum() < 4:
        return 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    mg, cg = med[good], centers[good]
    inner, background = float(mg[0]), float(mg[-1])
    rng = inner - background
    if rng <= 0:
        return 0.0
    hi, lo = background + 0.9 * rng, background + 0.1 * rng
    r_hi = _first_below(cg, mg, hi)
    r_lo = _first_below(cg, mg, lo)
    if r_hi is None or r_lo is None:
        return 0.0
    return float(max(0.0, r_lo - r_hi))


def _first_below(centers, vals, level):
    below = np.nonzero(vals <= level)[0]
    return float(centers[below[0]]) if below.size else None


def interior_profile(field: np.ndarray, *, n_bins=8) -> tuple[float, list[float]]:
    """Overall interior (non-escaping) fraction and its radial distribution."""
    h, w = field.shape
    r = _plane_radius_grid(h, w)
    nan = ~np.isfinite(field)
    edges = np.linspace(0.0, float(r.max()), n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    flat = nan.ravel()
    prof = []
    for b in range(n_bins):
        sel = flat[idx == b]
        prof.append(round(float(sel.mean()), 4) if sel.size else 0.0)
    return float(nan.mean()), prof


def measure_field(field: np.ndarray) -> dict:
    med_rings, p90_rings = radial_rings(field)
    ifrac, iprof = interior_profile(field)
    v = field[np.isfinite(field)]
    return {
        "cycles_spanned": round(cycles_spanned(field), 4),
        "radial_rings": round(med_rings, 2),
        "radial_rings_p90": round(p90_rings, 2),
        "falloff_extent": round(falloff_extent(field), 5),
        "interior_fraction": round(ifrac, 4),
        "interior_radial": iprof,
        "escaped_px": int(v.size),
        "smooth_min": (round(float(v.min()), 2) if v.size else None),
        "smooth_max": (round(float(v.max()), 2) if v.size else None),
    }


def measure_location(cx, cy, fw, maxiter, *, width=MEASURE_W, height=MEASURE_H,
                     ss=MEASURE_SS, family="mandelbrot", threads=None,
                     tmpdir=None, maxiter_policy=None) -> dict:
    """Dump one field and measure it. The .bin is transient — nothing is kept.

    `threads=None` -> the committed single-process engine default (see `dump_field`).

    The result carries `POLICY_KEY`: the iteration-cap policy `maxiter` was sized
    under. `maxiter_policy=None` means the LIVE production policy, which is true for
    every caller here (they all pass `rc.auto_maxiter(fw)`); pass the four constants
    explicitly if the cap came from somewhere else."""
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        out = Path(td) / "f.bin"
        field, side = dump_field(cx, cy, fw, maxiter, out, width=width, height=height,
                                 ss=ss, family=family, threads=threads)
        m = measure_field(field)
        m["maxiter"] = int(maxiter)
        m["dims"] = [side["width"], side["height"], side["supersample"]]
        m[POLICY_KEY] = policy_token(maxiter_policy)
        return m
