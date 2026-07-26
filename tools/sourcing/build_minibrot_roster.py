#!/usr/bin/env python
"""Durable nucleus roster across degrees 2-5, banded by period and cut by an
`A`-based f64 feasibility predictor.

This is the production successor to the size-band-and-subsample draw that lived in
the (closed) study `tools/studies/q4_multibrot_transfer.py::source_nuclei`. Two
deliberate departures from that study draw:

  1. **Selection is per (degree, period band), not global.** Period is the loop
     variable; we keep up to `TARGET_PER_CELL` atoms in each `(degree, band)` cell
     so degree is never silently confounded with depth by a global subsample. Bands:
     {3-4, 5-6, 7-9, 10-12, 13-15}. Target 8 per cell -> 4 degrees x 5 bands x 8 = 160.

  2. **The global size band is replaced by an `A`-based feasibility cut.** We use
     `deep_center_finder.atom_instrument` (|A| = 1/|size|) to predict the f64
     pixel-spacing wall a priori and admit an atom only if it renders in f64 **at the
     deploy presentation** (1280x720 ss4, the emission wallpaper geometry) with
     **>= MARGIN_MIN_DECADES (1) decade of headroom**. The emission render path is f64
     regardless of the perturbation backend, so "renderable at the deploy presentation
     with margin" is the criterion, not "renderable at all". The 1-decade margin is
     chosen so a sub-window crop (which can push ~1 decade deeper than the whole-atom
     frame) still lands in f64.

Only the `deep_center_finder` LIBRARY is imported here (Newton / size / atom
instrument / symmetry-canonical dedup) — never a study module.

Under-filled cells are REPORTED, never backfilled from adjacent bands (that would
re-confound degree with depth). The roster is DURABLE (`data/minibrot_roster/`): it
records a selection and a train/eval split that later descent crops inherit, so it
must stay stable across future harvests and is written through `paths.durable()`.

Run:  uv run python tools/sourcing/build_minibrot_roster.py [--seeds-ang N --seeds-rad N]
Writes (durable):  data/minibrot_roster/roster.jsonl  +  roster_cells.json
Prints:            the per-cell fill table (target vs filled, medians, excluded).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)                    # deep_center_finder (library)
sys.path.insert(0, os.path.join(_ROOT, "tools"))  # paths (durable())

import mpmath as mp                          # noqa: E402
import deep_center_finder as dcf             # noqa: E402
import paths                                 # noqa: E402


# --------------------------------------------------------------------------- #
# Roster geometry / policy. These are load-bearing for reproducibility: later
# harvests inherit the split, so a band edge or a cut threshold must not drift.
# --------------------------------------------------------------------------- #
DEGREES = [2, 3, 4, 5]
BANDS = [(3, 4), (5, 6), (7, 9), (10, 12), (13, 15)]   # inclusive period ranges
PERIODS = list(range(BANDS[0][0], BANDS[-1][1] + 1))   # 3..15
TARGET_PER_CELL = 8

# Deploy / emission presentation (the f64 wallpaper the feasibility cut protects):
# present.rs full path + CORPUS_SCHEMA render block. 1280x720, grid ss4.
DEPLOY_W, DEPLOY_SS = 1280, 4
# The pilot renders its screen FIELDS through the f64 render-one --dump-field path at
# W=2176 ss1 (q4_stage1_labelset geometry); recorded too so the pilot can tell which
# feasibility-excluded (near-boundary) atoms are still field-renderable.
FIELD_W, FIELD_SS = 2176, 1
MARGIN_MIN_DECADES = 1.0                      # admit iff deploy margin >= this

# Newton / dedup precision (matches the study draw so keys are comparable).
NUCLEUS_DPS = 60
NEWTON_STEPS = 60
DEDUP_DPS = 22
ORIGIN_EPS = mp.mpf("1e-6")                   # reject the c~0 period-1 degenerate

# Split policy — atom-level (each roster atom is one distinct minibrot, so atom-level
# assignment is minibrot-disjoint by construction), 70/30, stratified per (degree,band).
TRAIN_FRAC = 0.70
SPLIT_SEED = 20260726                         # fixed; reshuffling would break inheritance

# How many near-boundary feasibility-excluded atoms to RETAIN per cell as rows (for the
# pilot's "show me the rejects" draw). Excluded rows carry split=null, admitted=false.
KEEP_EXCLUDED_PER_CELL = 8

ROSTER_PATH = "data/minibrot_roster/roster.jsonl"
CELLS_PATH = "data/minibrot_roster/roster_cells.json"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no mpmath / no sourcing).
# --------------------------------------------------------------------------- #
def band_of(period: int):
    """Return the inclusive (lo, hi) band tuple containing `period`, or None."""
    for lo, hi in BANDS:
        if lo <= period <= hi:
            return (lo, hi)
    return None


def band_tag(band) -> str:
    return f"{band[0]}-{band[1]}"


def deploy_wall_log10(width=DEPLOY_W, ss=DEPLOY_SS, spacing_floor=1e-13, k=4.0) -> float:
    """log10 of |A| at which a k*window_scale frame at `width`x`ss` hits the f64 wall.
    Mirrors AtomInstrument.f64_wall_margin_decades' wall term so the test can pin it."""
    return math.log10(k) - math.log10(spacing_floor) - math.log10(width * ss)


def select_spanning(atoms, n):
    """Pick <= n atoms from `atoms` spanning their log10|A| range (scale diversity within
    a band), deterministically. Fewer than n -> return all (sorted by log10|A|)."""
    ordered = sorted(atoms, key=lambda a: (a["log10_abs_A"], a["dedup_key"]))
    if len(ordered) <= n:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(idx.tolist()))]


def _cell_seed(deg, band):
    """Deterministic per-(degree,band) RNG seed, stable across processes/runs."""
    import hashlib
    h = hashlib.sha256(f"{deg}|{band}".encode()).hexdigest()
    return (SPLIT_SEED ^ int(h[:8], 16)) & 0x7FFFFFFF


def assign_splits(admitted):
    """Stratified 70/30 train/eval within each (degree, band) cell, deterministic.
    Mutates each atom dict in place, adding 'split'. Atom-level == minibrot-disjoint."""
    by_cell = defaultdict(list)
    for a in admitted:
        by_cell[(a["degree"], a["band"])].append(a)
    for (deg, band), cell in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        cell.sort(key=lambda a: a["dedup_key"])        # stable, order-independent
        # cell-local seed so adding atoms to one cell never reshuffles another. Derived
        # via sha256 (NOT hash() — str hashing is PYTHONHASHSEED-salted, which would make
        # the durable split non-reproducible across runs).
        seed = _cell_seed(deg, band)
        perm = np.random.default_rng(seed).permutation(len(cell))
        n_train = round(TRAIN_FRAC * len(cell))
        train_idx = set(perm[:n_train].tolist())
        for i, a in enumerate(cell):
            a["split"] = "train" if i in train_idx else "eval"
    return admitted


# --------------------------------------------------------------------------- #
# Sourcing (mpmath; slow-ish -> background it).
# --------------------------------------------------------------------------- #
def ring_seeds(degree, n_ang, n_rad):
    """Seeds on `n_rad` rings between 0.25 and 1.08*R_boundary, `n_ang` angles each.
    R_boundary = 2^(1/(d-1)) is the degree-d escape radius; minibrots cluster near dM."""
    Rb = 2.0 ** (1.0 / (degree - 1))
    radii = np.linspace(0.25, 1.08 * Rb, n_rad)
    seeds = []
    for r in radii:
        for k in range(n_ang):
            th = 2.0 * math.pi * k / n_ang
            seeds.append((float(r * math.cos(th)), float(r * math.sin(th))))
    return seeds


def _is_minimal_nucleus(c, period, degree, tol):
    for q in range(1, period):
        if period % q == 0 and abs(dcf._orbit(c, q, degree)[0]) < tol:
            return False
    return True


def source_degree(degree, n_ang, n_rad, log=print):
    """Source every minimal, deduped nucleus for `degree` across PERIODS. Returns a list
    of atom dicts with margins computed; admission is decided by the caller."""
    mp.mp.dps = NUCLEUS_DPS
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    wall = deploy_wall_log10()
    field_wall = deploy_wall_log10(FIELD_W, FIELD_SS)
    digits = dcf.emit_digits_for_fw(1e-20)     # lossless enough for any f64 frame
    found = {}
    n_solves = 0
    t0 = time.time()
    for sr, si in ring_seeds(degree, n_ang, n_rad):
        seed = mp.mpc(sr, si)
        for p in PERIODS:
            n_solves += 1
            r = dcf.newton_nucleus(seed, p, degree=degree, max_steps=NEWTON_STEPS)
            if not r.converged or abs(r.c) < ORIGIN_EPS:
                continue
            if not _is_minimal_nucleus(r.c, p, degree, tol):
                continue
            # canonicalize into the fundamental sector, then dedup (collapses the
            # (d-1) rotational copies of one atom to a single key).
            cc = dcf.canonical_nucleus_c(r.c, degree)
            key = dcf.nucleus_dedup_key(cc, degree, DEDUP_DPS)
            if key in found:
                continue
            inst = dcf.atom_instrument(cc, p, degree)
            if not math.isfinite(inst.log10_abs_A):
                continue
            size = dcf.nucleus_size_estimate(cc, p, degree)
            sabs = float(abs(size)) if size != 0 else 0.0
            fw = sabs * 4.0
            margin_deploy = wall - inst.log10_abs_A
            margin_field = field_wall - inst.log10_abs_A
            found[key] = dict(
                degree=degree, period=p, band=band_tag(band_of(p)),
                cx=mp.nstr(cc.real, digits, strip_zeros=False),
                cy=mp.nstr(cc.imag, digits, strip_zeros=False),
                abs_A=inst.abs_A, log10_abs_A=round(inst.log10_abs_A, 4),
                arg_A=round(inst.arg_A, 4),
                f64_margin_deploy_decades=round(margin_deploy, 4),
                f64_margin_field_decades=round(margin_field, 4),
                dedup_key=",".join(key),
                size=sabs, fw=f"{fw:.6e}",
                maxiter=dcf._maxiter_for_fw(fw),
                family=("mandelbrot" if degree == 2 else f"multibrot{degree}"),
                newton_res_log10=round(r.residual, 1))
    atoms = [a for a in found.values() if a["band"] is not None]
    log(f"  d{degree}: {n_solves} Newton solves, {len(atoms)} minimal-deduped atoms "
        f"in period band, {time.time()-t0:.1f}s", flush=True)
    return atoms


def build_roster(n_ang, n_rad, log=print):
    """Full roster build. Returns (rows, cell_report). rows = admitted + retained
    near-boundary excluded; cell_report = per-(degree,band) fill status."""
    all_atoms = []
    for deg in DEGREES:
        all_atoms.extend(source_degree(deg, n_ang, n_rad, log=log))

    # Bucket by cell; split admitted (margin >= 1) from feasibility-excluded.
    by_cell = defaultdict(lambda: {"admit": [], "excl": []})
    for a in all_atoms:
        cell = by_cell[(a["degree"], a["band"])]
        if a["f64_margin_deploy_decades"] >= MARGIN_MIN_DECADES:
            cell["admit"].append(a)
        else:
            cell["excl"].append(a)

    admitted, excluded_rows, cell_report = [], [], []
    for deg in DEGREES:
        for band in BANDS:
            cell = by_cell[(deg, band_tag(band))]
            sel = select_spanning(cell["admit"], TARGET_PER_CELL)
            for a in sel:
                a["admitted"] = True
                a["exclusion"] = None
            admitted.extend(sel)
            # retain near-boundary excluded that are still field-renderable, nearest to
            # the 1-decade boundary first (largest deploy margin among the excluded).
            renderable_excl = [a for a in cell["excl"]
                               if a["f64_margin_field_decades"] >= 0.3]
            renderable_excl.sort(key=lambda a: -a["f64_margin_deploy_decades"])
            keep_excl = renderable_excl[:KEEP_EXCLUDED_PER_CELL]
            for a in keep_excl:
                a["admitted"] = False
                a["exclusion"] = "feasibility"
                a["split"] = None
            excluded_rows.extend(keep_excl)
            periods = [a["period"] for a in sel]           # medians describe the roster
            l10 = [a["log10_abs_A"] for a in sel]
            cell_report.append(dict(
                degree=deg, band=band_tag(band), target=TARGET_PER_CELL,
                filled=len(sel), n_admitted_available=len(cell["admit"]),
                n_excluded_feasibility=len(cell["excl"]),
                n_excluded_retained=len(keep_excl),
                median_period=(float(np.median(periods)) if periods else None),
                median_log10_abs_A=(round(float(np.median(l10)), 3) if l10 else None),
                underfilled=(len(sel) < TARGET_PER_CELL)))

    assign_splits(admitted)

    # stable, human-scannable ordering + ids
    rows = admitted + excluded_rows
    rows.sort(key=lambda a: (a["degree"], a["period"], not a["admitted"], a["dedup_key"]))
    per_deg = defaultdict(int)
    for a in rows:
        i = per_deg[a["degree"]]
        per_deg[a["degree"]] += 1
        a["id"] = f"d{a['degree']}_p{a['period']:02d}_{i:03d}"
    return rows, cell_report


# --------------------------------------------------------------------------- #
def print_cell_table(cell_report):
    print("\n" + "=" * 96)
    print("PER-CELL FILL  (target 8; under-filled cells reported, NOT backfilled)")
    print("=" * 96)
    hdr = (f"{'deg':>3} {'band':>6} {'target':>6} {'filled':>6} {'avail':>6} "
           f"{'excl_feas':>9} {'med_per':>7} {'med_log|A|':>10}  status")
    print(hdr)
    print("-" * 96)
    for c in cell_report:
        status = "UNDER" if c["underfilled"] else "full"
        if c["filled"] == 0:
            status = "EMPTY"
        mp_ = "-" if c["median_period"] is None else f"{c['median_period']:.1f}"
        ml = "-" if c["median_log10_abs_A"] is None else f"{c['median_log10_abs_A']:.2f}"
        print(f"{c['degree']:>3} {c['band']:>6} {c['target']:>6} {c['filled']:>6} "
              f"{c['n_admitted_available']:>6} {c['n_excluded_feasibility']:>9} "
              f"{mp_:>7} {ml:>10}  {status}")
    print("-" * 96)
    tot_target = sum(c["target"] for c in cell_report)
    tot_filled = sum(c["filled"] for c in cell_report)
    n_under = sum(1 for c in cell_report if c["underfilled"])
    print(f"TOTAL filled {tot_filled}/{tot_target}   under-filled cells: {n_under}/{len(cell_report)}")


def write_roster(rows, cell_report):
    rpath = paths.durable(ROSTER_PATH, mkparents=True)
    # atomic write (temp + replace): a concurrent or interrupted run can never leave a
    # half-written or clobbered durable roster (the failure that produced a mixed-schema
    # roster once already).
    tmp = rpath.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for a in rows:
            fh.write(json.dumps(a) + "\n")
    os.replace(tmp, rpath)
    cpath = paths.durable(CELLS_PATH)
    admitted = [a for a in rows if a["admitted"]]
    n_train = sum(1 for a in admitted if a["split"] == "train")
    meta = dict(
        policy=dict(degrees=DEGREES, bands=[list(b) for b in BANDS],
                    target_per_cell=TARGET_PER_CELL,
                    deploy_presentation=dict(width=DEPLOY_W, ss=DEPLOY_SS),
                    field_presentation=dict(width=FIELD_W, ss=FIELD_SS),
                    margin_min_decades=MARGIN_MIN_DECADES,
                    deploy_wall_log10_abs_A=round(deploy_wall_log10(), 4),
                    train_frac=TRAIN_FRAC, split_seed=SPLIT_SEED, dedup_dps=DEDUP_DPS),
        totals=dict(n_rows=len(rows), n_admitted=len(admitted),
                    n_excluded_retained=len(rows) - len(admitted),
                    n_train=n_train, n_eval=len(admitted) - n_train),
        cells=cell_report)
    cpath.write_text(json.dumps(meta, indent=1))
    print(f"\n-> roster (durable): {ROSTER_PATH}  ({len(rows)} rows, "
          f"{len(admitted)} admitted [{n_train} train / {len(admitted)-n_train} eval])")
    print(f"-> cells  (durable): {CELLS_PATH}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-ang", type=int, default=64,
                    help="ring seed angles per radius (default 64)")
    ap.add_argument("--seeds-rad", type=int, default=8,
                    help="ring seed radii (default 8)")
    args = ap.parse_args(argv)
    print(f"building minibrot roster: degrees {DEGREES}, bands "
          f"{[band_tag(b) for b in BANDS]}, target {TARGET_PER_CELL}/cell")
    print(f"feasibility: deploy {DEPLOY_W}x ss{DEPLOY_SS}, admit margin "
          f">= {MARGIN_MIN_DECADES} decade (log10|A| <= {deploy_wall_log10():.3f})")
    print(f"seeds: {args.seeds_ang} ang x {args.seeds_rad} rad per degree\n")
    rows, cell_report = build_roster(args.seeds_ang, args.seeds_rad)
    print_cell_table(cell_report)
    write_roster(rows, cell_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
