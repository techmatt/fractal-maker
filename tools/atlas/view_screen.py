#!/usr/bin/env python
r"""view_screen.py — the VIEW-level screen: spatial participation, not just dynamic range.

WHY IT EXISTS. `maneuver_screen.py` scores the **atom** — one field per nucleus at 64x36
on the 4x atom frame, shared across every `k` row (`minibrot_maneuvers.md` §3.1). That is
the right unit for the thing it was built for and the wrong unit for judging a picture:
the view actually pushed is `k * window_scale` wide (or the parent's `fw` for a `keep`),
which at `k = None` runs to thousands of atom widths. Two failures follow, both named in
`orbital_field_metrics.md` §8 ("Composition ... no scalar settles a composition call") and
both visible on the dry run's Q5 sheet:

  * **the zoomed blob** — a nucleus-centred frame that is mostly dead black interior and
    still scores high `rings`, because rays are cut by NaN and scored on what escapes;
  * **the giant field of blue** — one deep pocket sets a large radial span while the rest
    of the frame is flat, and `radial_range` is a median over rays that ALL start at the
    centre, so a single central well raises every one of them.

Both are the same blind spot: the ring measures describe **dynamic range**, not **spatial
participation**. This module adds the participation half and computes the pair on the
frame that is actually pushed.

WHAT IS NEW HERE, AND WHAT IS REUSED. New: `band_coverage` and `composite`. Reused
verbatim, so there is one definition of each: `rescore_lib.ring_measures` (both ring
measures, one ray walk), `field_metrics.interior_profile` (interior fraction + its radial
profile), `field_metrics.dump_field` (the 64x36 f64 field), and
`maneuver_screen.{screen_maxiter, screen_policy_token}` (the SAME stamped cap policy the
atom screen runs under, `mi12000k0.3c4800-67000`, `retired.md`'s dated UN-RETIRED entry).
`falloff_extent` and `interior_fraction` already existed in `field_metrics` from the
original falloff-criterion work; `falloff_extent` is one of the three measures
`retired.md` lists as failed and is not used here.

`band_coverage` IS NOT `energy::occupancy`. It is deliberately the same SHAPE — grid the
frame, apply a floor per tile, report the occupied fraction — because that shape is the
one the Rust content gate already uses and there is no reason to invent a second one. It
is a different MEASURE: `occupancy` reduces OKLab edge energy over a rendered RGB image,
which needs a render and a palette; this reduces the raw escape-time field, which is what
a 64x36 screen can afford. Do not read one as a proxy for the other
(`measurement_practice.md` §2, "occupancy != mid-detail", "edge-energy != quality").

THE COMPOSITE, AND WHY THE VETO IS NOT A QUALITY AXIS. Interior mass as an independent
quality axis is RETIRED — it measured +0.046 given degree (`minibrot_sourcing.md` §11,
`retired.md`). Nothing here revives it. `interior_fraction` is used as a **sort-to-bottom
composition veto**: a frame that is mostly non-escaping is not a worse picture in
proportion to its interior, it is a picture whose scalars are being computed on the
minority of it that escaped. The veto is a statement about the instrument's domain, not
about quality, and it never excludes: a vetoed candidate keeps every raw measure and keeps
its order among the other vetoed ones.

The veto threshold is expressed RELATIVE TO THE REFERENCES (`VETO_REF_SLACK` x the larger
reference interior fraction) rather than as a bare float, for the same reason the atom
screen ranks against the run's own distribution: an absolute field number means nothing
across geometries and cap policies (`orbital_field_metrics.md` §5, §7). The reference
interiors are measured, frozen in `data/atlas/view_screen_refs.json`, and re-derivable by
`--refs`.

COMPOSITE v3 (2026-08-01) — three re-weightings of recorded measures, no new field. Matt's
verdicts on the v2 Q5 sheet named three defects, each fixed by weighting differently what
is already on the row: a **size band** on `interior_fraction` (a nucleus can be too big
before it is a veto), a **winsorized richness** (the raw product is unbounded and the sweep
argmax exploits it), and an **anchor-retention constraint** on the sweep (its contract is
"frame THIS minibrot well", not "find the richest window near here"). `composite_v2` is
kept beside `composite_v3` because the v2 gate record must stay reproducible from source,
not only from the JSON it was written to.

COMPOSITE v4 (2026-08-01) — PROPOSED, MEASURED, AND **NOT ADOPTED**. `composite_v3` is still
the live sort key; `composite_v4` exists so the 41 formulations that were run against the v4
gate stay re-derivable from source rather than only from the JSON they were written to,
exactly as `composite_v2` does.

The proposal: `band_coverage` calls a tile participating iff it SPANS a colour cycle, and a
slow gradient spans a cycle with one lazy band in it — so a "field of blue" could post
`covq25 = 0.50`. v4 additionally required structure INSIDE the tile (band-boundary crossings
per pixel step), leaving the arithmetic above it — size band, winsorized richness, veto —
untouched.

WHY IT WAS NOT ADOPTED, because the negative result is the useful part. The field does not
support the premise. The tile that motivated it (`snap k16 d5 p16`) is not a frame of lazy
span-only tiles: its dead area is one contiguous diagonal band, and its participating tiles
hold band crossings at population rates. Measured on the 16,440, that tile sits at **p64.6
on the coverage term and p69.4 on richness** — above median on BOTH — and reaches p83.5
because the product of two above-median heavy-tailed factors clears the top quintile. No
refinement of either factor demotes it, and the crossing clause moves it the WRONG way
(p83.5 -> p84.4) while re-promoting the dense k4 frames v3's size band exists to demote
(p77.8 -> p84.3). Full record, every family and floor: `data/atlas/view_screen_gate.json`
§v4 and `orbital_field_metrics.md` §11.7.

What the iteration did leave behind is the field cache (`view_field_cache.py`): the next
per-tile statistic is a numpy pass, not a 17-minute engine pass over the population.

  uv run python tools/atlas/view_screen.py --refs        # re-measure the references
  uv run python tools/atlas/view_screen.py --demo <cx> <cy> <fw>
"""
from __future__ import annotations

import argparse
import decimal
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools" / "explorer",
           ROOT / "tools" / "corpus", ROOT / "tools" / "descent", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import field_metrics as fm          # noqa: E402  dump_field, interior_profile, SCREEN_*, POLICY_KEY
import rescore_lib as rl            # noqa: E402  ring_measures: both measures, one ray walk
import maneuver_screen as ms        # noqa: E402  the stamped screen cap policy

DENSITY = fm.DENSITY                # one colour cycle == 1/DENSITY iterations == 40

# --------------------------------------------------------------------------- #
# band_coverage
# --------------------------------------------------------------------------- #
# The grid divides 64x36 exactly (4x4 px tiles, 144 of them). Integer-division tiling,
# same as `energy::occupancy`, so a geometry that does not divide drops the remainder.
GRID_X, GRID_Y = 16, 9

# A tile PARTICIPATES iff it shows at least one complete colour cycle AND is not mostly
# dead. One cycle is the render-visible unit — a full palette traversal — so the floor is
# phase-independent, which a "does floor(t) change inside the tile" test would not be.
TILE_CYCLE_FLOOR = 1.0
# ...and a tile more than three-quarters non-escaping reads as black whatever the few
# escaping pixels in it do. Not a second interior measure: it is what stops a thin bright
# rim from crediting the black tiles it runs through.
TILE_MIN_FINITE = 0.25

# Blocks: the tile grid pooled into a 4 x 3 grid of REGIONS (4x3 tiles = 16x12 px each).
# `band_coverage` alone is a mean over tiles and so is blind to WHERE the dead tiles are —
# a frame that is one solid black disc plus one solid flat gradient can carry the same
# tile mean as a frame with structure spread evenly through it. Pooling first and then
# taking a low quantile across regions asks the spatial question instead: three quarters
# of the frame's regions participate at least this much. See the report for the measured
# difference (it is the whole reason formulation 1 failed the gate).
BLOCK_X, BLOCK_Y = 4, 3
BLOCK_QUANTILE = 25.0


def _tile_stats(field: np.ndarray, gx: int, gy: int):
    """Per-tile (span in colour cycles, finite share) on a `gy x gx` grid."""
    h, w = field.shape
    tw, th = w // gx, h // gy
    if tw == 0 or th == 0:
        return None, None
    t = np.asarray(field, dtype=np.float64) * DENSITY
    # Trim the remainder, then fold to (gy, gx, th*tw) so each tile is one row.
    t = t[:gy * th, :gx * tw].reshape(gy, th, gx, tw).transpose(0, 2, 1, 3)
    t = t.reshape(gy, gx, th * tw)
    finite = np.isfinite(t)
    n_finite = finite.sum(axis=2)
    # +/-inf sentinels rather than nanmax/nanmin: an all-interior tile is the NORMAL case
    # here, not an edge case, and it must reduce to span 0 without raising a warning.
    hi = np.where(finite, t, -np.inf).max(axis=2)
    lo = np.where(finite, t, np.inf).min(axis=2)
    span = np.where(n_finite > 0, hi - lo, 0.0)
    return span, n_finite / float(th * tw)


# --------------------------------------------------------------------------- #
# v4: STRUCTURE INSIDE THE TILE, not span across it
# --------------------------------------------------------------------------- #
# THE DEFECT v4 FIXES. `span >= 1 cycle` is satisfied by a tile that holds ONE lazy band:
# a slow monotone gradient crossing a single colour boundary spans a cycle with no
# structure in it at all. That is how `snap k16 d5 p16` — "a field of blue" — posted
# `band_coverage_q25 = 0.50`, i.e. half of every region "participating" in banding that is
# not there. The favourite (`neighborhood_expand k16 d2 p43`) differs not in SPAN but in how
# many band boundaries its tiles contain. So the indicator has to ask how much boundary a
# tile holds, not how far its values travel. `[verdict: Matt, on the v3 Q5 sheet]`
#
# THE THREE FAMILIES BELOW ARE ALL "STRUCTURE WITHIN THE TILE" AND THEY ARE NOT THE SAME
# STATISTIC. All are computed here, in one place, so the gate SELECTS among them rather than
# reimplementing any of them (the v3 precedent: only the rejected log compression lives in
# the gate, because it is not what shipped).
#
#   "cross"  — the fraction of a tile's adjacent finite pixel pairs (both axes, within the
#              tile only) whose `floor(cycles)` differs, i.e. how many of this tile's pixel
#              steps cross a colour-band boundary. Bounded in [0, 1] and phase-independent
#              (crossing an integer boundary is what a band edge IS, and the count of
#              boundaries along a path does not move with the palette's phase). BOUNDED IS
#              THE POINT, and it is v3's winsorization lesson applied one level down: one
#              catastrophic jump between two adjacent screen pixels contributes exactly one
#              crossing, not fifty, so an aliased seam cannot manufacture participation.
#   "tv"     — mean |delta cycles| per adjacent finite pixel step: the same shape, UNBOUNDED.
#              Recorded as the alternative that a single discontinuity can carry.
#   "bands"  — the count of distinct `floor(cycles)` values present in the tile. The most
#              literal reading of "distinct rings per tile"; it is phase-DEPENDENT at the
#              margin (a tile spanning 1.0 cycles holds 2 distinct floors or 1, depending on
#              where the boundary falls) which is exactly what `TILE_CYCLE_FLOOR` was written
#              to avoid, and it is recorded so that cost is measured and not asserted.
#
# THE SPAN CLAUSE IS KEPT, NOT REPLACED. v4 participation is `span >= 1 cycle AND
# finite_share >= 0.25 AND structure >= floor` — a strict TIGHTENING, so every v4-
# participating tile also participated under v3 and coverage can only fall. Without the span
# clause the crossing test would credit a tile that merely straddles one boundary with a span
# of 0.01 — flatter than anything v3 admitted — which is the opposite of the fix.
PARTICIPATION_MODES = ("cross", "tv", "bands")

# The shipped structure floor. Selected by the v4 gate under a rule written down first: the
# LEAST DEMANDING floor on a fixed 0.05 grid that satisfies every clause of the v4 gate.
# See `data/atlas/view_screen_gate.json` §v4 for the grid and the floors that failed.
TILE_CROSS_FLOOR = 0.40


def _tile_fold(field: np.ndarray, gx: int, gy: int):
    """`(gy, gx, th, tw)` view of the field in colour cycles, remainder trimmed.

    Tiles are kept 2-D here (unlike `_tile_stats`, which flattens each tile) because every
    v4 statistic is about ADJACENCY inside a tile, and a flattened tile has no adjacency.
    """
    h, w = field.shape
    tw, th = w // gx, h // gy
    if tw == 0 or th == 0:
        return None
    t = np.asarray(field, dtype=np.float64) * DENSITY
    return t[:gy * th, :gx * tw].reshape(gy, th, gx, tw).transpose(0, 2, 1, 3)


def tile_structure(field: np.ndarray, mode: str, *, gx: int = GRID_X, gy: int = GRID_Y):
    """Per-tile structure statistic on a `gy x gx` grid, one of `PARTICIPATION_MODES`.

    Pairs that straddle a tile edge are NOT counted: the question is what one tile holds, and
    borrowing a neighbour's boundary would let a single band smeared across the frame credit
    every tile it passes through — the same error the 25%-escaping clause exists to stop.
    """
    if mode not in PARTICIPATION_MODES:
        raise ValueError(f"unknown participation mode {mode!r}")
    t = _tile_fold(field, gx, gy)
    if t is None:
        return None
    fin = np.isfinite(t)
    if mode == "bands":
        # Distinct finite floors per tile, by counting runs in the sorted tile. `+inf` is
        # the sentinel rather than NaN so the non-escaping pixels sort to the end and drop
        # out of the run count via `isfinite`, instead of comparing unequal to themselves.
        flat = np.sort(np.where(fin, np.floor(t), np.inf).reshape(t.shape[0], t.shape[1], -1),
                       axis=2)
        f = np.isfinite(flat)
        first = f[:, :, :1].sum(axis=2)
        new = (f[:, :, 1:] & (flat[:, :, 1:] != flat[:, :, :-1])).sum(axis=2)
        return (first + new).astype(np.float64)
    b = np.floor(t)
    ph = fin[:, :, :, :-1] & fin[:, :, :, 1:]
    pv = fin[:, :, :-1, :] & fin[:, :, 1:, :]
    if mode == "cross":
        nh = (ph & (b[:, :, :, :-1] != b[:, :, :, 1:])).sum(axis=(2, 3))
        nv = (pv & (b[:, :, :-1, :] != b[:, :, 1:, :])).sum(axis=(2, 3))
    else:                                    # "tv"
        nh = np.where(ph, np.abs(np.diff(t, axis=3)), 0.0).sum(axis=(2, 3))
        nv = np.where(pv, np.abs(np.diff(t, axis=2)), 0.0).sum(axis=(2, 3))
    dh, dv = ph.sum(axis=(2, 3)), pv.sum(axis=(2, 3))
    # PER-AXIS, THEN MAX — not pooled over both axes. Pooling makes the statistic measure
    # ORIENTATION as much as structure: a band running exactly along one axis contributes
    # nothing to the other axis's pairs, so its pooled value is halved and the ceiling for
    # axis-aligned banding is 0.5 while diagonal banding reaches 1.0. A band running along
    # one axis is still a band. Taking the max asks "along its own direction of variation,
    # how many of this tile's steps cross a boundary", which is orientation-free and reaches
    # 1.0 for dense banding at any angle.
    return np.maximum(np.where(dh > 0, nh / np.maximum(dh, 1), 0.0),
                      np.where(dv > 0, nv / np.maximum(dv, 1), 0.0))


def _participating(field: np.ndarray, gx: int, gy: int, cycle_floor: float,
                   min_finite: float, *, mode: str | None = None,
                   structure_floor: float = 0.0):
    """The tile-participation indicator. `mode=None` is the v3 (span-only) rule.

    `mode` adds the v4 structure clause on top of — never instead of — the span and
    escaping clauses, so the v4 indicator is a subset of the v3 one by construction.
    """
    span, finite_share = _tile_stats(field, gx, gy)
    if span is None:
        return None
    ok = (finite_share >= min_finite) & (span >= cycle_floor)
    if mode is not None:
        s = tile_structure(field, mode, gx=gx, gy=gy)
        ok = ok & (s >= structure_floor)
    return ok.astype(np.float64)


def band_coverage(field: np.ndarray, *, gx: int = GRID_X, gy: int = GRID_Y,
                  cycle_floor: float = TILE_CYCLE_FLOOR,
                  min_finite: float = TILE_MIN_FINITE) -> float:
    """Fraction of grid tiles that participate in the banding structure, in [0, 1].

    The target failure is a frame where one deep well sets a large `radial_range` while
    everything else is flat: the well occupies a handful of tiles and every other tile
    spans well under a cycle, so `radial_range` stays high and this goes low. A frame of
    dense filigree goes high at the same `radial_range`, which is the discrimination the
    ring measures cannot make.

    Recorded on every row but NOT the term the composite uses — see `band_coverage_q25`.
    """
    ok = _participating(field, gx, gy, cycle_floor, min_finite)
    return 0.0 if ok is None else float(ok.mean())


def band_coverage_q25(field: np.ndarray, *, gx: int = GRID_X, gy: int = GRID_Y,
                      bx: int = BLOCK_X, by: int = BLOCK_Y,
                      quantile: float = BLOCK_QUANTILE,
                      cycle_floor: float = TILE_CYCLE_FLOOR,
                      min_finite: float = TILE_MIN_FINITE) -> float:
    """The spatially-pooled coverage the composite selects on, in [0, 1].

    Pool the tile-participation indicator into `by x bx` regions, then take the
    `quantile`-th percentile across regions: "at least three quarters of the frame's
    regions participate at least this much". A frame whose dead area is CONCENTRATED —
    one black disc, one flat gradient — puts whole regions at zero and scores far below
    its own tile mean; a frame whose structure is spread scores near it.
    """
    ok = _participating(field, gx, gy, cycle_floor, min_finite)
    return _pool_q(ok, gx, gy, bx, by, quantile)


def _pool_q(ok, gx: int, gy: int, bx: int, by: int, quantile: float) -> float:
    if ok is None or gy % by or gx % bx:
        return 0.0 if ok is None else float(ok.mean())
    blocks = ok.reshape(by, gy // by, bx, gx // bx).mean(axis=(1, 3))
    return float(np.percentile(blocks, quantile))


def coverage_pair(field: np.ndarray, *, mode: str | None = None,
                  structure_floor: float = 0.0, gx: int = GRID_X, gy: int = GRID_Y,
                  bx: int = BLOCK_X, by: int = BLOCK_Y, quantile: float = BLOCK_QUANTILE,
                  cycle_floor: float = TILE_CYCLE_FLOOR,
                  min_finite: float = TILE_MIN_FINITE) -> tuple[float, float]:
    """`(band_coverage, band_coverage_q25)` under one participation rule, in ONE tile pass.

    `mode=None` reproduces the v3 pair exactly (same predicate, same pooling); a `mode`
    adds the v4 structure clause. Returned as a pair because the composite's coverage term
    is the geometric mean of the two and computing them separately walked the field twice.
    """
    ok = _participating(field, gx, gy, cycle_floor, min_finite, mode=mode,
                        structure_floor=structure_floor)
    if ok is None:
        return 0.0, 0.0
    return float(ok.mean()), _pool_q(ok, gx, gy, bx, by, quantile)


# --------------------------------------------------------------------------- #
# the view measures
# --------------------------------------------------------------------------- #
# The formulation grids the v4 gate selects over, recorded per row so the selection is
# arithmetic on the record instead of 21 more passes over the population. Each entry is a
# `(band_coverage, band_coverage_q25)` pair under that participation rule.
#   cross — fraction of a tile's steps crossing a band boundary along its own axis, in
#           [0, 1]. A single boundary through a 4-px tile sits at 1/3; a boundary at every
#           step is 1.0, so the grid runs from just above "one boundary" to saturation.
#   tv    — mean |delta cycles| per pixel step along the same axis. A monotone tile spanning
#           exactly one cycle sits at 1/3, which is why the grid brackets it.
#   bands — distinct floor(cycles) values present in the tile.
COVERAGE_GRID = (
    ("cross", (0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.67, 0.75, 0.85)),
    ("tv", (0.34, 0.40, 0.50, 0.67, 0.85, 1.00)),
    ("bands", (2.0, 3.0, 4.0, 5.0, 6.0)),
)


def coverage_grid(field: np.ndarray) -> dict:
    """`{mode: {floor: [coverage, coverage_q25]}}` over `COVERAGE_GRID`. Pure; no engine."""
    return {mode: {f"{f:g}": [round(a, 4), round(b, 4)]
                   for f in floors
                   for a, b in [coverage_pair(field, mode=mode, structure_floor=f)]}
            for mode, floors in COVERAGE_GRID}


# The SECOND v4 family, and it was added AFTER the participation grid above was run and
# failed — recorded in that order because that is the order it happened in. The tile
# indicator is v3's, unchanged; what moves is the POOLING the q25 is taken over. It is here
# because the field says the participation hypothesis was aimed at the wrong thing: the
# "field of blue"'s dead area is a contiguous diagonal band that the 4x3 region grid cannot
# isolate (every region catches part of the live diagonal), not a frame of lazy tiles.
# `by` must divide GRID_Y = 9 and `bx` must divide GRID_X = 16.
POOLING_GRID = ((4, 3, 25.0), (4, 3, 10.0), (8, 3, 25.0), (8, 3, 10.0),
                (16, 3, 25.0), (16, 3, 10.0), (8, 9, 25.0), (8, 9, 10.0))


def pooling_grid(field: np.ndarray) -> dict:
    """`{"BXxBYqQ": [coverage, coverage_q]}` over `POOLING_GRID`, v3 participation. Pure.

    One tile pass, reused across every pooling: the indicator does not depend on how the
    regions are drawn, so re-deriving it per variant would be eight identical walks.
    """
    ok = _participating(field, GRID_X, GRID_Y, TILE_CYCLE_FLOOR, TILE_MIN_FINITE)
    if ok is None:
        return {f"{bx}x{by}q{q:g}": [0.0, 0.0] for bx, by, q in POOLING_GRID}
    mean = round(float(ok.mean()), 4)
    return {f"{bx}x{by}q{q:g}": [mean, round(_pool_q(ok, GRID_X, GRID_Y, bx, by, q), 4)]
            for bx, by, q in POOLING_GRID}


def view_measures(field: np.ndarray) -> dict:
    """Every raw measure this screen records for one field. Pure; no engine.

    All are kept on every candidate whatever the composite does with them — the screen is a
    recording (`maneuver_screen.py`, "NOT A GATE"). The v3 coverage pair stays on the row
    beside the v4 pair for the same reason `composite_v2` stays live: the record a previous
    gate was written from has to keep re-deriving from source.
    """
    m = rl.ring_measures(field)
    ifrac, iprof = fm.interior_profile(field)
    finite = field[np.isfinite(field)]
    cov4, cov4_q25 = coverage_pair(field, mode="cross", structure_floor=TILE_CROSS_FLOOR)
    return dict(
        interior_fraction=round(float(ifrac), 4),
        interior_radial=iprof,
        band_coverage=round(band_coverage(field), 4),
        band_coverage_q25=round(band_coverage_q25(field), 4),
        band_coverage_v4=round(cov4, 4),
        band_coverage_q25_v4=round(cov4_q25, 4),
        radial_range=round(float(m["radial_range"]), 4),
        radial_rings=round(float(m["radial_rings"]), 2),
        radial_range_p90=round(float(m["radial_range_p90"]), 4),
        radial_rings_p90=round(float(m["radial_rings_p90"]), 2),
        escaped_px=int(finite.size),
        smooth_max=(round(float(finite.max()), 2) if finite.size else None),
    )


def measure_view(cx, cy, fw, *, family: str = "mandelbrot", threads: int = 1,
                 tmpdir=None, timeout: float = fm.FIELD_TIMEOUT_S) -> dict:
    """Dump one 64x36 field AT THE VIEW'S OWN FRAME and measure it. Never raises.

    The frame is `fw` as given — the view actually pushed — not `4 * window_scale`. That
    is the whole point of this module, and it is also why none of the 4x-frame validation
    in `orbital_field_metrics.md` §2 transfers: the ring measures here are the same
    functions on a different frame, and are re-validated against references measured on
    the same frames (`--refs`).
    """
    meta = view_frame_policy(fw)
    if not meta["screened"]:
        return meta
    maxiter = meta["view_maxiter"]
    try:
        with tempfile.TemporaryDirectory(dir=tmpdir) as td:
            field, _side = fm.dump_field(cx, cy, float(fw), maxiter, Path(td) / "f.bin",
                                         width=fm.SCREEN_W, height=fm.SCREEN_H,
                                         ss=fm.SCREEN_SS, family=family, threads=threads,
                                         timeout=max(1.0, float(timeout)))
    except Exception as e:
        return dict(meta, screened=False, screen_reason=f"dump_field:{str(e)[:120]}")
    return measure_view_from_field(fw, field)


def view_frame_policy(fw) -> dict:
    """The engine-INDEPENDENT half of `measure_view`: the spacing guard, the stamped cap and
    the policy token. Shared by the live path, the field cache and the cached re-score, so
    the three cannot disagree about which frame and which cap a row was measured under."""
    tok = ms.screen_policy_token()
    fw = float(fw)
    if not (fw > 0 and math.isfinite(fw)) or (fw / fm.SCREEN_W) <= 1e-13:
        return dict(screened=False, screen_reason="f64_spacing_wall_at_screen_geometry",
                    view_fw=fw, **{fm.POLICY_KEY: tok})
    return dict(screened=True, view_fw=fw, view_maxiter=ms.screen_maxiter(fw),
                **{fm.POLICY_KEY: tok})


def measure_view_from_field(fw, field: np.ndarray) -> dict:
    """`measure_view`'s output for a field that has already been dumped. Pure; no engine.

    This is what makes the field cache a substitute for a measurement rather than an
    approximation of one: the live path calls exactly this function on the array the engine
    just wrote, so a cached row and a live row differ only in where the array came from."""
    meta = view_frame_policy(fw)
    if not meta["screened"]:
        return meta
    m = view_measures(field)
    sm = m["smooth_max"]
    mi = meta["view_maxiter"]
    return dict(**meta,
                cap_headroom=(round(1.0 - sm / mi, 4) if sm is not None else None),
                clamped=bool(mi >= ms.SCREEN_MAXITER_POLICY[3]), **m)


# --------------------------------------------------------------------------- #
# the composite
# --------------------------------------------------------------------------- #
REFS_PATH = "data/atlas/view_screen_refs.json"

# The veto is anchored on the references' ESCAPING share, not on their interior fraction.
# A multiple of the reference interior would be degenerate: the two references measure
# 0.0000 and 0.0104 interior, so ANY multiple of them is a hair above zero and vetoes
# ~70% of the population — a veto that fires on most rows is the main sort, not a veto.
# The escaping share is where the references carry mass (1.0000 and 0.9896), so the
# threshold is stated there: a view is vetoed when its escaping area falls below
# `VETO_ESCAPED_SHARE` of the weaker reference's, i.e. when appreciably more than a third
# of the frame is dead. The share is a judgement; it is frozen alongside the reference
# measurement it is applied to, and moving either moves the veto in code.
VETO_ESCAPED_SHARE = 2.0 / 3.0

# --------------------------------------------------------------------------- #
# v3: the size band on interior
# --------------------------------------------------------------------------- #
# THE VETO IS A DOMAIN STATEMENT; THIS IS A COMPOSITION ONE, AND THEY ARE DIFFERENT.
# The veto says "the scalars are being computed on a minority of the frame". The band says
# something the veto is far too late to say: a nucleus can be TOO BIG for the picture long
# before it is dead area. Matt's verdicts off the v2 Q5 sheet, which is the whole
# calibration set: interior ~= 0 is fine (pure filigree; both references sit there),
# interior up to ~0.12 passed his eye, 0.17 is "good region but minibrot too big", and the
# 0.20-0.25 k4 series reads as dominated. The veto fires at 0.3403 — every one of those
# tiles clears it, which is why v2 ranked four of them into its own top quintile.
#
# This is NOT a revival of interior mass as a quality axis (retired at +0.046 given degree,
# `minibrot_sourcing.md` §11, `retired.md`). Interior mass as a quality axis is monotone and
# unbounded in the claim "less interior is better"; this is a BAND — flat and neutral from 0
# to the edge, so a frame with 0.10 interior is not preferred to one with 0.00 — that only
# expresses "past this share the subject dominates the frame". Above the edge it declines to
# the veto's own behaviour at the veto threshold, so the two are continuous rather than a
# step at 0.3403.
#
# WHERE THE TWO NUMBERS COME FROM, SEPARATELY, BECAUSE THEY WERE NOT FIXED THE SAME WAY.
# The EDGE is Matt's verdict transcribed (the top of what he passed), not a fit. The
# EXPONENT is one scalar fitted against those tiles under a rule written down before the
# grid was run: the LEAST STEEP integer exponent that satisfies every clause of the v3 gate.
# `exp = 6` is recorded as failing (the 0.17 tile survives at p83.9) so the choice reads as
# a fit to six-plus anchor points, which is what it is. `[data/atlas/view_screen_gate.json]`
SIZE_BAND_EDGE = 0.12
SIZE_BAND_EXP = 8.0

# --------------------------------------------------------------------------- #
# v3: the richness cap
# --------------------------------------------------------------------------- #
# `sqrt(range x rings)` is unbounded, and an unbounded term inside an argmax is a term the
# argmax will find the tail of: the framing sweep's worst choice took an antenna-seam window
# at `radial_range = 16603` to a composite of ~1505, against a top-quintile population that
# lives at 3-30. A seam window should score like a rich frame, not like fifty of them.
#
# Both measures are winsorized at `RICHNESS_CAP_REF_MULT` x the STRONGEST reference's value
# — the same anchoring the veto uses, and for the same reason: an absolute field number
# means nothing across geometries and cap policies (`orbital_field_metrics.md` §5, §7).
# Twice the reference is a judgement; what it buys is that the cap lands at the top of the
# ordinary population rather than inside it (on the re-scored 16,440 it clips `range` on
# ~1.5% of rows and `rings` on ~0.5%, i.e. the tail and not the body).
#
# WINSORIZE AND NOT LOG, ON THE STATED CRITERION. Log compression was measured and keeps the
# seam window at ~12x the population's richness, because `c*log1p(x/c)/log2` is still
# growing at x = 78c. Winsorizing is the one that meets "like a rich frame, not 50x one".
# The cost is real and is the reason it is recorded rather than asserted: above the cap the
# ordering is DESTROYED, not compressed, so two genuinely different rich frames tie and fall
# back to the tie-break. On the population gate neither choice is visible at all (all six
# anchors sit far below the cap) — the compression is justified on the SWEEP, and the sweep
# is where its evidence is.
RICHNESS_CAP_REF_MULT = 2.0


class ScreenParams(NamedTuple):
    """Everything the v3 composite needs that is derived from the reference record.

    Derived in code, frozen in the record it reads: re-measuring the references moves the
    veto AND the caps, instead of leaving stale literals in the source. Passed explicitly
    rather than resolved inside `composite_v3`, so a test can inject one and a driver
    resolves it once per run instead of per row.
    """
    veto: float
    cap_range: float
    cap_rings: float
    band_edge: float = SIZE_BAND_EDGE
    band_exp: float = SIZE_BAND_EXP


# The richness term: the geometric mean of the two ring measures.
#
# STATE WHAT THIS IS AND IS NOT FORCED BY. `orbital_field_metrics.md` §4 records that
# `radial_range` FAILS validation on the eye at 320x180 (17.77 against mb19's 70.30, below
# 21 triage atoms) because the eye's richness is dense oscillation rather than radial span,
# while `radial_rings` is the measure both references pass. That is the reason the pair is
# not collapsed to one. It is NOT, on this population, a gate requirement: at the view
# frame all three of `range`, `rings` and their geometric mean put the eye in the top
# decile and mb19 in the top quintile, so the gate does not discriminate between them
# (measured; the sweep is in the report). The geometric mean is chosen because it moves
# when EITHER measure moves and so cannot inherit either one's known single-reference
# failure — a judgement, taken with the alternatives measured, not a forced choice.
def richness(m: dict, p: "ScreenParams | None" = None) -> float:
    """`sqrt(range x rings)`, winsorized at `p`'s caps when one is given.

    `p=None` is the v2 term, uncapped — kept so `composite_v2` is the original function and
    not a re-derivation of it.
    """
    rng = max(0.0, float(m["radial_range"]))
    rings = max(0.0, float(m["radial_rings"]))
    if p is not None:
        rng, rings = min(rng, p.cap_range), min(rings, p.cap_rings)
    return math.sqrt(rng * rings)


def size_factor(m: dict, p: "ScreenParams") -> float:
    """The v3 size band on `interior_fraction`, in [0, 1].

    Neutral (1.0) from 0 through `band_edge`; above it, `((veto - i)/(veto - edge))**exp`,
    reaching 0 exactly at the veto threshold so the graded term and the sort-to-bottom band
    meet rather than step. Nothing above the veto reaches this: those rows are vetoed.
    """
    i = float(m["interior_fraction"])
    if i <= p.band_edge:
        return 1.0
    if i >= p.veto:
        return 0.0
    return ((p.veto - i) / (p.veto - p.band_edge)) ** p.band_exp


COV_KEYS_V3 = ("band_coverage", "band_coverage_q25")
COV_KEYS_V4 = ("band_coverage_v4", "band_coverage_q25_v4")


def coverage_term(m: dict, keys: tuple = COV_KEYS_V3) -> float:
    """The coverage half of the composite: `sqrt(band_coverage * band_coverage_q25)`.

    HOW MUCH of the frame participates, times HOW EVENLY that participation is spread,
    geometrically — so a frame needs both, and neither factor alone can carry it. The two
    ends were each tried on their own against the gate and each failed one half of it: the
    tile mean alone leaves the `snap k16 d2 p18` blob at p62 (it has mid tile coverage, in
    two solid slabs), and the pooled quantile alone drops `mb19_p35` at 16x to p79.9 and
    misses the reference bar by a tenth of a percentile. Full record, including the fact
    that this term was chosen AFTER seeing those two results and what that costs:
    `orbital_field_metrics.md` §11 and `data/atlas/view_screen_gate.json`.

    Computed from the two recorded measures rather than from the field, so a row scored
    before this term existed re-scores without a re-measurement. `keys` selects WHICH
    recorded pair — `COV_KEYS_V3` (span participation) or `COV_KEYS_V4` (span AND in-tile
    structure). The arithmetic is identical; only the indicator underneath it moved.
    """
    return math.sqrt(max(0.0, float(m[keys[0]])) * max(0.0, float(m[keys[1]])))


def load_refs(path=None) -> dict:
    import paths
    p = Path(path) if path else paths.durable(REFS_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def interior_veto(refs: dict, *, share: float = VETO_ESCAPED_SHARE) -> float:
    """The veto threshold on `interior_fraction`, derived from the references at run time.

    Derived in code, frozen in the record it reads (`CLAUDE.md`, "Derive state in code;
    freeze it in records") — so re-measuring the references moves the veto instead of
    leaving a stale literal in the source.
    """
    escaped = [1.0 - float(r["interior_fraction"]) for r in refs["references"].values()
               if r.get("screened")]
    if not escaped:
        raise ValueError(f"{REFS_PATH}: no screened references — cannot derive the veto")
    return round(max(0.0, 1.0 - share * min(escaped)), 4)


def richness_caps(refs: dict, *, mult: float = RICHNESS_CAP_REF_MULT) -> tuple:
    """The winsorization caps, derived from the references at run time (see `interior_veto`).

    The STRONGEST reference sets each cap, not the weakest: the caps say "beyond this much
    dynamic range, more is not more", and anchoring that on the weaker reference would clip
    the stronger one — a reference must never be capped by the screen it calibrates.
    """
    ok = [r for r in refs["references"].values() if r.get("screened")]
    if not ok:
        raise ValueError(f"{REFS_PATH}: no screened references — cannot derive the caps")
    return (round(mult * max(float(r["radial_range"]) for r in ok), 4),
            round(mult * max(float(r["radial_rings"]) for r in ok), 4))


def screen_params(refs: dict, *, share: float = VETO_ESCAPED_SHARE,
                  mult: float = RICHNESS_CAP_REF_MULT, band_edge: float = SIZE_BAND_EDGE,
                  band_exp: float = SIZE_BAND_EXP) -> ScreenParams:
    """Every v3 constant that is derived from the reference record, resolved once."""
    cr, cg = richness_caps(refs, mult=mult)
    return ScreenParams(veto=interior_veto(refs, share=share), cap_range=cr, cap_rings=cg,
                        band_edge=band_edge, band_exp=band_exp)


def _veto_band(c: float) -> float:
    """The sort-to-bottom band: `[-1, 0)`, strictly below every non-vetoed score, and still
    ordered among the vetoed by the same quantity. Shared by v2 and v3 so the two composites
    cannot drift on the one behaviour that is not a re-weighting."""
    return -1.0 / (1.0 + c)


def composite_v2(m: dict, veto: float) -> float:
    """The v2 sort key, FROZEN. `coverage_term * richness`, uncapped, no size band.

    Kept live (not just in the JSON) so `view_screen_gate.py` re-derives the recorded v2
    block from source rather than copying it forward — a record whose producer no longer
    exists cannot be checked (`verification_practice.md` §7).
    """
    if not m.get("screened"):
        return float("-inf")
    c = coverage_term(m) * richness(m)
    return _veto_band(c) if float(m["interior_fraction"]) > veto else c


def composite_v3(m: dict, p: ScreenParams) -> float:
    """The live sort key: size-banded, coverage-weighted, winsorized richness.

    `size_factor * coverage_term * richness(capped)`, with the interior veto sorting to
    bottom exactly as in v2. Non-vetoed candidates score >= 0; a vetoed one scores in
    `[-1, 0)`. Recording, never exclusion: nothing is dropped, and every raw measure
    survives on the row — the band and the caps are weights on the sort, not filters.

    THE BAND IS NOT APPLIED INSIDE THE VETOED BAND, and that is deliberate. `size_factor`
    is 0 for every vetoed row by construction (it reaches 0 at the veto), so banding the
    vetoed branch would collapse all of them to exactly -1.0 and destroy v2's stated
    contract that a vetoed candidate "keeps its order among the other vetoed ones". Below
    the veto the frame is outside the instrument's domain and a composition judgement on it
    means nothing, so what orders those rows stays what ordered them in v2. Caught by
    `test_the_veto_sorts_to_bottom_and_never_excludes`, which runs on both composites.
    """
    if not m.get("screened"):
        return float("-inf")
    c = coverage_term(m) * richness(m, p)
    if float(m["interior_fraction"]) > p.veto:
        return _veto_band(c)
    return size_factor(m, p) * c


def composite_v4(m: dict, p: ScreenParams, *, keys: tuple = COV_KEYS_V4) -> float:
    """v3 EXACTLY, with the v4 participation indicator. **NOT the live sort key** — v4 was
    measured against the gate and rejected (module doc). Kept live so the v4 gate block
    re-derives from source, and callable so the rejected argmax stays reproducible.

    Nothing in the composite's arithmetic changed between v3 and v4 — same size band, same
    winsorized richness, same veto, same sort-to-bottom band. What changed is one boolean
    inside `band_coverage`: a tile now has to hold band BOUNDARIES, not merely span a cycle.
    That is why this is a two-line function and not a re-derivation: writing it any other way
    would let the two versions drift on the parts that did not change.

    `composite_v3` stays live for the same reason `composite_v2` did — the v2 and v3 gate
    blocks are re-run from source on every gate invocation, not copied forward.
    """
    if not m.get("screened"):
        return float("-inf")
    c = coverage_term(m, keys) * richness(m, p)
    if float(m["interior_fraction"]) > p.veto:
        return _veto_band(c)
    return size_factor(m, p) * c


def is_vetoed(m: dict, veto: float) -> bool:
    return bool(m.get("screened")) and float(m["interior_fraction"]) > veto


# --------------------------------------------------------------------------- #
# the framing sweep
# --------------------------------------------------------------------------- #
# Nucleus-centred is the UN-FRAMED case, and the interior-band arc measured framing as the
# killer: uniform-sampled crops averaged 1.07 against G-framed 1.84 at every degree
# (`minibrot_sourcing.md` §4). So the sweep is not a refinement, it is the missing step.
#
# Fully deterministic — a fixed 3x3 offset grid at two scales, no RNG anywhere, so
# "seeded" is satisfied by there being nothing to seed. The chosen window is stamped
# beside the original frame, never in place of it.
SWEEP_OFFSETS = (-0.5, 0.0, 0.5)        # in units of the frame's own width / height
SWEEP_SCALES = (1.0, 2.0)
SWEEP_PREC = 80                          # decimal digits for the centre arithmetic
ASPECT_H_OVER_W = fm.SCREEN_H / fm.SCREEN_W


def sweep_windows(cx, cy, fw) -> list[dict]:
    """The deterministic sweep grid around one view. The first entry is the view itself.

    Offsets are +/- half a FRAME in each axis (so +/-0.5*fw horizontally and
    +/-0.5*fw*h/w vertically) and are fixed in the plane — the scale variants re-window
    the same neighbourhood rather than scaling the offsets with it. Centre arithmetic is
    `Decimal` at 80 digits: at `fw = 1.5e-10` against a 35-digit centre an f64 offset
    would be the same catastrophic cancellation `CLAUDE.md`'s coordinate rule forbids.
    """
    fw = float(fw)
    out = []
    with decimal.localcontext() as ctx:
        ctx.prec = SWEEP_PREC
        cxd, cyd = decimal.Decimal(str(cx)), decimal.Decimal(str(cy))
        fwd = decimal.Decimal(repr(fw))
        fhd = fwd * decimal.Decimal(repr(ASPECT_H_OVER_W))
        for s in SWEEP_SCALES:
            for oy in SWEEP_OFFSETS:
                for ox in SWEEP_OFFSETS:
                    out.append(dict(
                        dx=ox, dy=oy, scale=s,
                        cx=str(cxd + fwd * decimal.Decimal(repr(ox))),
                        cy=str(cyd + fhd * decimal.Decimal(repr(oy))),
                        fw=float(fwd * decimal.Decimal(repr(s))),
                        is_origin=(ox == 0.0 and oy == 0.0 and s == 1.0)))
    out.sort(key=lambda d: (not d["is_origin"], d["scale"], d["dy"], d["dx"]))
    return out


# v3: the anchor-retention constraint. The sweep's contract for a maneuver-anchored
# candidate is "frame THIS minibrot well", not "find the richest window near here" — and
# without a constraint the second is what an argmax does: it walks off the nucleus onto
# whatever seam scores highest, which is content drift dressed as framing. The nucleus must
# stay inside the central `ANCHOR_MARGIN` of the chosen window.
#
# 0.8 is a judgement, and on THIS grid it is not a knife edge: the fixed offsets put the
# nucleus at |0.5| frames (scale 1) or |0.25| (scale 2) from a window's centre, so every
# margin in (0.5, 1.0) selects the same 10 of 18 windows. What that means concretely is
# worth stating because it is the mechanism, not a side effect: for an anchored candidate
# the eligible moves are ZOOM OUT (all nine scale-2 windows) or STAY (the origin) — a
# same-scale lateral shift always drops the nucleus outside the margin. That is the intended
# answer for a frame whose minibrot is too big.
ANCHOR_MARGIN = 0.8


def anchor_retained(anchor_cx, anchor_cy, w: dict, *, margin: float = ANCHOR_MARGIN) -> bool:
    """Is the anchor inside the central `margin` box of window `w`? Pure geometry.

    `Decimal` at `SWEEP_PREC` for the same reason `sweep_windows` is: the difference of two
    35-digit centres at `fw = 1e-10` is exactly the cancellation `CLAUDE.md`'s coordinate
    rule forbids computing in f64.
    """
    with decimal.localcontext() as ctx:
        ctx.prec = SWEEP_PREC
        fwd = decimal.Decimal(repr(float(w["fw"])))
        fhd = fwd * decimal.Decimal(repr(ASPECT_H_OVER_W))
        nx = (decimal.Decimal(str(anchor_cx)) - decimal.Decimal(str(w["cx"]))) / fwd
        ny = (decimal.Decimal(str(anchor_cy)) - decimal.Decimal(str(w["cy"]))) / fhd
        half = decimal.Decimal(repr(margin)) / 2
        return bool(abs(nx) <= half and abs(ny) <= half)


def sweep_best(cx, cy, fw, p: ScreenParams, *, family: str = "mandelbrot", threads: int = 1,
               tmpdir=None, measure=None, anchor=None, composite=None,
               anchor_margin: float = ANCHOR_MARGIN) -> dict:
    """Measure every sweep window and return the argmax by `composite_v3` (the LIVE sort).

    `composite=` swaps the sort key, and it exists because v4 was measured and NOT adopted:
    an argmax under a rejected formulation is evidence about that formulation, so it has to
    be runnable without becoming the default.

    `measure=` is the injection point for tests (and for a cached driver); it defaults to
    the real `measure_view`. Ties break on the fixed window order, which puts the origin
    first — so a sweep that finds nothing better returns the original frame rather than an
    arbitrary neighbour.

    `anchor=(cx, cy)` applies the anchor-retention constraint: an ineligible window is still
    MEASURED and RECORDED with its composite, it is only barred from the argmax. Recording,
    never exclusion — the same contract the veto keeps. `anchor=None` is the non-anchored
    case (ranking arbitrary views), where the constraint has nothing to mean.
    """
    measure = measure or measure_view
    composite = composite or composite_v3
    wins = sweep_windows(cx, cy, fw)
    rows, best, best_c = [], None, None
    for wdw in wins:
        m = measure(wdw["cx"], wdw["cy"], wdw["fw"], family=family, threads=threads,
                    tmpdir=tmpdir)
        c = composite(m, p)
        ok = (True if anchor is None else
              anchor_retained(anchor[0], anchor[1], wdw, margin=anchor_margin))
        rows.append(dict(**{k: wdw[k] for k in ("dx", "dy", "scale", "is_origin")},
                         composite=(None if c == float("-inf") else round(c, 6)),
                         screened=bool(m.get("screened")), anchor_ok=ok,
                         band_coverage=m.get("band_coverage"),
                         band_coverage_q25=m.get("band_coverage_q25"),
                         band_coverage_v4=m.get("band_coverage_v4"),
                         band_coverage_q25_v4=m.get("band_coverage_q25_v4"),
                         radial_range=m.get("radial_range"),
                         radial_rings=m.get("radial_rings"),
                         interior_fraction=m.get("interior_fraction")))
        if ok and c != float("-inf") and (best_c is None or c > best_c):
            best_c, best = c, dict(window=wdw, measures=m)
    origin = next((r for r in rows if r["is_origin"]), None)
    return dict(
        n_windows=len(wins),
        n_anchor_eligible=sum(1 for r in rows if r["anchor_ok"]),
        anchor=(None if anchor is None else dict(cx=str(anchor[0]), cy=str(anchor[1]),
                                                 margin=anchor_margin)),
        origin_composite=(origin or {}).get("composite"),
        chosen=(None if best is None else
                dict(dx=best["window"]["dx"], dy=best["window"]["dy"],
                     scale=best["window"]["scale"], cx=best["window"]["cx"],
                     cy=best["window"]["cy"], fw=best["window"]["fw"])),
        chosen_composite=(None if best_c is None else round(best_c, 6)),
        chosen_measures=(None if best is None else best["measures"]),
        moved=bool(best is not None and not best["window"]["is_origin"]),
        windows=rows,
    )


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
# The eye at its validated 4x frame (`fw = 5.83e-4`, `render-one`'s own default view) and
# mb19 at 16x. 16x for mb19 and not 4x is deliberate and is the prompt's call: at 4x a
# period-35 nucleus IS the zoomed blob this screen exists to sort down, so measuring the
# veto against it would calibrate the veto on the failure.
REF_VIEWS = (
    dict(key="minibroteye", label="minibrot eye", scale=4,
         cx="-0.746339", cy="0.112242", base_scale="1.4575000000e-04"),
    dict(key="mb19_p35_16x", label="mb19_p35", scale=16,
         cx="-0.74977483272365342795786040375088960",
         cy="0.10761724352653678278696798751738616", base_scale="2.0174060071e-10"),
)


def measure_references(*, threads: int = 3) -> dict:
    out = {}
    for r in REF_VIEWS:
        fw = float(decimal.Decimal(r["base_scale"]) * r["scale"])
        m = measure_view(r["cx"], r["cy"], fw, threads=threads)
        # The references are the ONE population the v4 gate compares formulations on that
        # is not in the field cache, so their whole formulation grid is recorded here.
        # BOTH grids: `pooling_grid` was added when the pooling family was measured, and a
        # reference that carries only one of them makes exactly half the gate unrunnable —
        # which is how the pooling and coverage-exponent families ended up measured in a
        # scratch report instead of in the durable record.
        if m.get("screened"):
            with tempfile.TemporaryDirectory() as td:
                field, _ = fm.dump_field(r["cx"], r["cy"], fw, m["view_maxiter"],
                                         Path(td) / "f.bin", width=fm.SCREEN_W,
                                         height=fm.SCREEN_H, ss=fm.SCREEN_SS,
                                         threads=threads)
            m = dict(m, coverage_grid=coverage_grid(field),
                     pooling_grid=pooling_grid(field))
        out[r["key"]] = dict(label=r["label"], scale=r["scale"], cx=r["cx"], cy=r["cy"],
                             fw=fw, **m)
    return dict(
        geometry=[fm.SCREEN_W, fm.SCREEN_H, fm.SCREEN_SS],
        grid=[GRID_X, GRID_Y], blocks=[BLOCK_X, BLOCK_Y],
        block_quantile=BLOCK_QUANTILE, tile_cycle_floor=TILE_CYCLE_FLOOR,
        tile_min_finite=TILE_MIN_FINITE, tile_cross_floor=TILE_CROSS_FLOOR,
        veto_escaped_share=round(VETO_ESCAPED_SHARE, 6),
        richness_cap_ref_mult=round(RICHNESS_CAP_REF_MULT, 6),
        size_band=[SIZE_BAND_EDGE, SIZE_BAND_EXP],
        references=out,
        note=("Reference measurements for the view-level screen. The interior veto AND the "
              "v3 richness caps are derived from these at run time "
              "(view_screen.screen_params), not hardcoded — re-measuring moves both "
              "instead of leaving stale literals."),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", action="store_true", help="re-measure and write the refs")
    ap.add_argument("--demo", nargs=3, metavar=("CX", "CY", "FW"))
    a = ap.parse_args(argv)
    import paths
    if a.refs:
        rep = measure_references()
        p = paths.durable(REFS_PATH, mkparents=True)
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        for k, v in rep["references"].items():
            print(f"  {k:16s} fw={v['fw']:.4g}  interior={v.get('interior_fraction')}  "
                  f"cov={v.get('band_coverage')} cov_q25={v.get('band_coverage_q25')}  "
                  f"range={v.get('radial_range')}  rings={v.get('radial_rings')}")
        print(f"  veto: interior > 1 - {VETO_ESCAPED_SHARE:.4f} x min(ref escaped) "
              f"= {interior_veto(rep)}")
        print(f"  caps: {RICHNESS_CAP_REF_MULT:g} x max(ref) = {richness_caps(rep)}")
        print(f"-> {REFS_PATH}")
        return 0
    if a.demo:
        p = screen_params(load_refs())
        m = measure_view(a.demo[0], a.demo[1], float(a.demo[2]))
        print(json.dumps(m, indent=2))
        print(f"composite v4 = {composite_v4(m, p):.4f}  (v3 {composite_v3(m, p):.4f}, "
              f"v2 {composite_v2(m, p.veto):.4f}; size {size_factor(m, p):.4f}; "
              f"veto {p.veto}, vetoed={is_vetoed(m, p.veto)})")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
