#!/usr/bin/env python
"""Resumable, **unstratified** nucleus enumeration for the minibrot triage wall.

The descent roster (`tools/sourcing/build_minibrot_roster.py`) stratifies on
(degree, period band) and caps each cell at 8. Period does not predict quality, so
that draw is an unfiltered sample with the good atoms unselected inside it. This
enumerator deliberately does the opposite:

  * **No period stratification, no per-cell cap.** Every minimal, deduped nucleus a
    consumed seed yields is kept. The degree/period mix is whatever the scan
    produces — reported, never engineered.
  * **The `A`-feasibility cut is the ONLY cut.** `f64_margin_deploy_decades >= 1`
    at the deploy presentation (1280 x ss4), imported verbatim from the roster
    builder. That is a rendering constraint, not a quality judgement.
  * **Every covariate the enumeration already computes is recorded** (degree,
    period, |A| / log10|A| / arg A, both f64 margins, size, plus the raw orbit
    statistics and the seed provenance), so Matt's accept/reject can be joined
    against them afterwards to see which axes his eye is actually using.

Machinery is REUSED, not rebuilt: Newton, the sector-canonical nucleus
canonicalization + dedup key, the atom instrument, the size estimate and the
feasibility wall all come from `deep_center_finder` / `build_minibrot_roster`.

Resumability
------------
The seed schedule is a fixed, deterministic list of `(degree, seed_index)` tasks
over a `SEED_ANG x SEED_RAD` ring grid per degree (the roster's `ring_seeds`),
consumed in a seeded permutation so an early stop still covers the whole region.
`enum_state.json` holds a **cursor** into that list. Re-running with a larger
`--target` continues from the cursor: the first 200 atoms are never re-derived, and
the pool grows to 1000+ by consuming more of the same schedule. Ids are content
hashes of the dedup key, so a duplicate can never enter twice even if a run is
interrupted mid-chunk.

Run:
    uv run python tools/descent/build_triage_pool.py --target 200
    uv run python tools/descent/build_triage_pool.py --target 1000    # extends
    uv run python tools/descent/build_triage_pool.py --refs-only      # reference row
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "sourcing"))

import mpmath as mp                       # noqa: E402
import deep_center_finder as dcf          # noqa: E402
import build_minibrot_roster as brs       # noqa: E402  (feasibility policy + ring seeds)
import triage_store as ts                 # noqa: E402

# --------------------------------------------------------------------------- #
# Schedule geometry. Load-bearing for resumability: the cursor indexes THIS
# schedule, so changing a value below invalidates every stored cursor (the run
# refuses to continue unless --force-reschedule).
# --------------------------------------------------------------------------- #
SCHEDULE_VERSION = 1
DEGREES = brs.DEGREES                     # [2, 3, 4, 5] — all four, in whatever mix the scan gives
SEED_ANG = 96                             # ring seed angles per radius, per degree
SEED_RAD = 12                             # ring seed radii, per degree
PERIOD_MIN, PERIOD_MAX = 3, 20            # broad and UNBANDED (the roster's bands stop at 15)

# Feasibility + precision policy: imported, not restated.
MARGIN_MIN_DECADES = brs.MARGIN_MIN_DECADES
NUCLEUS_DPS = brs.NUCLEUS_DPS
NEWTON_STEPS = brs.NEWTON_STEPS
DEDUP_DPS = brs.DEDUP_DPS
ORIGIN_EPS = brs.ORIGIN_EPS

# Neighbour covariate (derived sidecar, see `write_neighbors`).
NEIGHBOR_RADIUS_W = 20.0                  # within 20 x this atom's own window scale
NEIGHBOR_SIZE_DECADES = 1.0               # and within 1 decade of its |A|

# --------------------------------------------------------------------------- #
# The reference row: known-good material rendered through the SAME framing ladder.
# If it does not read as good at 1x/4x/16x, the framing is wrong and that has to be
# visible before 200 tiles are scanned, not after.
#
# `base_scale` is the unit the ladder multiplies, chosen so the **4x tile is exactly
# the canonical known-good frame** (4x is the wall default).
# --------------------------------------------------------------------------- #
REFERENCE_SEEDS = [
    dict(
        id="ref_eye", label="minibrot eye",
        cx="-0.746339", cy="0.112242", degree=2,
        canonical_fw=0.000583,            # render-one's own default --fw
        source="src/render_one.rs RenderOneArgs defaults (--cx/--cy/--fw)",
        note=("render-one's default VIEW, not a nucleus: Newton finds no nucleus "
              "within ~1e-4 of this center, so base_scale is derived as "
              "canonical_fw/4 rather than from an atom instrument."),
        nucleus=False,
    ),
    dict(
        id="ref_mb19", label="mb19_p35",
        cx="-0.74977483272365342795786040375088960",
        cy="0.10761724352653678278696798751738616",
        degree=2, period=35,
        canonical_fw=8.069624e-10,
        source=("data/q4_window_corpus/batches/2026-07-23_q4_g_aimed/windows.jsonl "
                "(minibrot_id mb19_p35); scratch/q4_stage1/minibrots.json is gone"),
        note="a true nucleus: base_scale is its atom instrument's window_scale.",
        nucleus=True,
    ),
]


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def seeds_per_degree() -> int:
    return SEED_ANG * SEED_RAD


def total_tasks() -> int:
    return seeds_per_degree() * len(DEGREES)


def _permutation(n: int) -> np.ndarray:
    """Deterministic scramble of the seed order, so consuming the first K tasks
    samples the whole region instead of the innermost rings."""
    return np.random.default_rng(1000003 * SCHEDULE_VERSION + n).permutation(n)


def schedule_task(i: int, perm: np.ndarray) -> tuple[int, int]:
    """Task `i` -> (degree, seed_index). Degrees round-robin at the outer position
    so an early stop still carries all four."""
    deg = DEGREES[i % len(DEGREES)]
    return deg, int(perm[(i // len(DEGREES)) % len(perm)])


def schedule_meta() -> dict:
    return dict(schedule_version=SCHEDULE_VERSION, degrees=list(DEGREES),
                seed_ang=SEED_ANG, seed_rad=SEED_RAD,
                period_min=PERIOD_MIN, period_max=PERIOD_MAX,
                margin_min_decades=MARGIN_MIN_DECADES,
                dedup_dps=DEDUP_DPS, nucleus_dps=NUCLEUS_DPS,
                newton_steps=NEWTON_STEPS, total_tasks=total_tasks())


def schedule_compatible(state: dict) -> bool:
    old = state.get("schedule")
    if not old:
        return True
    keys = ("schedule_version", "degrees", "seed_ang", "seed_rad",
            "period_min", "period_max", "margin_min_decades", "dedup_dps")
    new = schedule_meta()
    return all(old.get(k) == new.get(k) for k in keys)


# --------------------------------------------------------------------------- #
# Per-atom covariates
# --------------------------------------------------------------------------- #
def orbit_stats(c, period: int, degree: int) -> tuple[float, int]:
    """Raw critical-orbit statistic: min |z_k| over 1 <= k < period, and its index.

    Recorded as a RAW covariate, not a classification. It is *not* a
    primitive-vs-satellite label — see `docs/design` note in the report; the existing
    machinery has no cheap satellite test and none was built here. One extra orbit
    pass of length `period` (negligible beside the ~60 Newton steps already spent)."""
    z = mp.mpc(0)
    best, best_k = None, 0
    for k in range(1, period):
        if degree == 2:
            z = z * z + c
        else:
            z = z ** (degree - 1) * z + c
        a = abs(z)
        if best is None or a < best:
            best, best_k = a, k
    return (float(best) if best is not None else float("nan")), best_k


def make_atom(cc, period: int, degree: int, residual: float, digits: int,
              wall: float, field_wall: float, prov: dict) -> dict | None:
    """Build one pool row from a canonicalized nucleus. Returns None if the atom is
    degenerate or fails the `A`-feasibility cut (the only cut applied)."""
    inst = dcf.atom_instrument(cc, period, degree)
    if not math.isfinite(inst.log10_abs_A) or inst.abs_A <= 0:
        return None
    margin_deploy = wall - inst.log10_abs_A
    if margin_deploy < MARGIN_MIN_DECADES:
        return None
    size = dcf.nucleus_size_estimate(cc, period, degree)
    sabs = float(abs(size)) if size != 0 else 0.0
    min_abs_z, min_abs_z_k = orbit_stats(cc, period, degree)
    key = dcf.nucleus_dedup_key(cc, degree, DEDUP_DPS)
    dedup_key = ",".join(key)
    fw4 = inst.window_scale * 4.0
    return {
        "id": ts.atom_id(degree, dedup_key),
        "degree": degree,
        "period": period,
        "family": ts.family_for(degree),
        "cx": mp.nstr(cc.real, digits, strip_zeros=False),
        "cy": mp.nstr(cc.imag, digits, strip_zeros=False),
        # framing ladder unit: the atom's own size. fw(scale) = window_scale * scale.
        "window_scale": f"{inst.window_scale:.10e}",
        "fw": f"{fw4:.6e}",               # the 4x (default) frame, roster convention
        # --- covariates the enumeration already computes ---
        "abs_A": inst.abs_A,
        "log10_abs_A": round(inst.log10_abs_A, 6),
        "arg_A": round(inst.arg_A, 6),
        "rotation_ambiguity_rad": round(inst.rotation_ambiguity_rad, 6),
        "size": sabs,
        "f64_margin_deploy_decades": round(margin_deploy, 4),
        "f64_margin_field_decades": round(field_wall - inst.log10_abs_A, 4),
        "required_dps": inst.required_dps,
        "newton_res_log10": round(residual, 2),
        "min_abs_z_pre": min_abs_z,       # raw orbit stat; NOT a satellite label
        "min_abs_z_index": min_abs_z_k,
        "dedup_key": dedup_key,
        "provenance": prov,
    }


def run_task(degree: int, seed_idx: int, task_ordinal: int, run_id: str,
             seeds_cache: dict, digits: int, wall: float, field_wall: float,
             have: set[str]) -> tuple[list[dict], Counter]:
    """Consume one (degree, seed) task: Newton over every period, canonicalize,
    dedup, feasibility-cut. Returns (new atom rows, per-outcome counters)."""
    seeds = seeds_cache.setdefault(degree, brs.ring_seeds(degree, SEED_ANG, SEED_RAD))
    sr, si = seeds[seed_idx]
    seed = mp.mpc(sr, si)
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    rows, counts = [], Counter()
    for p in range(PERIOD_MIN, PERIOD_MAX + 1):
        counts["solves"] += 1
        r = dcf.newton_nucleus(seed, p, degree=degree, max_steps=NEWTON_STEPS)
        if not r.converged or abs(r.c) < ORIGIN_EPS:
            counts["no_converge"] += 1
            continue
        if not brs._is_minimal_nucleus(r.c, p, degree, tol):
            counts["not_minimal"] += 1
            continue
        cc = dcf.canonical_nucleus_c(r.c, degree)     # sector-canonical: collapses the
        key = dcf.nucleus_dedup_key(cc, degree, DEDUP_DPS)   # (d-1) rotational copies
        aid = ts.atom_id(degree, ",".join(key))
        if aid in have:
            counts["duplicate"] += 1
            continue
        prov = dict(run_id=run_id, schedule_version=SCHEDULE_VERSION,
                    task_ordinal=task_ordinal, seed_index=seed_idx,
                    seed_re=float(sr), seed_im=float(si),
                    seed_ang=SEED_ANG, seed_rad=SEED_RAD,
                    newton_iters=r.iters)
        row = make_atom(cc, p, degree, r.residual, digits, wall, field_wall, prov)
        if row is None:
            counts["cut_feasibility"] += 1
            continue
        have.add(aid)
        rows.append(row)
        counts["kept"] += 1
    return rows, counts


# --------------------------------------------------------------------------- #
# Derived neighbour sidecar (regenerated whole after every run)
# --------------------------------------------------------------------------- #
def write_neighbors(pool: list[dict]) -> dict:
    """Count comparable-size same-degree pool atoms near each atom.

    Pool-relative and therefore a **lower bound**: it counts what this scan found,
    not what exists. Regenerated in full on every run (never stored per-row, which
    would go stale the moment the pool grows). Different degrees are different
    parameter planes, so neighbours are only ever counted within a degree."""
    by_deg: dict[int, list[dict]] = {}
    for a in pool:
        by_deg.setdefault(a["degree"], []).append(a)
    out: dict[str, dict] = {}
    for deg, atoms in by_deg.items():
        xs = np.array([float(a["cx"]) for a in atoms])
        ys = np.array([float(a["cy"]) for a in atoms])
        la = np.array([float(a["log10_abs_A"]) for a in atoms])
        ws = np.array([float(a["window_scale"]) for a in atoms])
        for i, a in enumerate(atoms):
            d = np.hypot(xs - xs[i], ys - ys[i])
            comparable = np.abs(la - la[i]) <= NEIGHBOR_SIZE_DECADES
            comparable[i] = False
            near = comparable & (d <= NEIGHBOR_RADIUS_W * ws[i])
            if comparable.any():
                nearest = float(d[comparable].min() / ws[i])
            else:
                nearest = None
            out[a["id"]] = {
                "n_comparable_within_20w": int(near.sum()),
                "nearest_comparable_dist_over_w": (round(nearest, 4)
                                                   if nearest is not None else None),
            }
    ts.write_json(ts.NEIGHBORS, {
        "note": ("DERIVED, pool-relative, regenerated on every enumeration run. A LOWER "
                 "BOUND on true neighbour density: it counts only nuclei this scan found. "
                 "Same-degree only (different degrees are different parameter planes)."),
        "radius_window_scales": NEIGHBOR_RADIUS_W,
        "size_window_decades": NEIGHBOR_SIZE_DECADES,
        "n_atoms": len(pool),
        "neighbors": out,
    })
    return out


# --------------------------------------------------------------------------- #
# Reference row
# --------------------------------------------------------------------------- #
def build_references() -> list[dict]:
    mp.mp.dps = NUCLEUS_DPS
    refs = []
    for spec in REFERENCE_SEEDS:
        r = dict(spec)
        if spec.get("nucleus"):
            c = mp.mpc(mp.mpf(spec["cx"]), mp.mpf(spec["cy"]))
            inst = dcf.atom_instrument(c, spec["period"], spec["degree"])
            r["base_scale"] = f"{inst.window_scale:.10e}"
            r["log10_abs_A"] = round(inst.log10_abs_A, 6)
            r["f64_margin_deploy_decades"] = round(
                inst.f64_wall_margin_decades(brs.DEPLOY_W, ss=brs.DEPLOY_SS), 4)
        else:
            r["base_scale"] = f"{spec['canonical_fw'] / 4.0:.10e}"
        r["family"] = ts.family_for(spec["degree"])
        r["scales"] = list(ts.SCALES)
        refs.append(r)
    ts.write_json(ts.REFERENCES, {
        "note": ("Known-good material rendered through the SAME framing ladder as the "
                 "wall (1x/4x/16x of base_scale, vivid blue_orange, nav fidelity). "
                 "base_scale is chosen so the 4x tile IS the canonical known-good frame."),
        "references": refs,
    })
    return refs


# --------------------------------------------------------------------------- #
def report(pool: list[dict], state: dict) -> None:
    print("\n" + "=" * 78)
    print(f"POOL: {len(pool)} atoms   cursor {state['cursor']}/{total_tasks()} tasks")
    print("=" * 78)
    by_deg = Counter(a["degree"] for a in pool)
    print("  degree spread: " + "  ".join(f"d{d}={by_deg.get(d, 0)}" for d in DEGREES))
    per = Counter(a["period"] for a in pool)
    print("  period spread: " + "  ".join(f"p{p}={per[p]}" for p in sorted(per)))
    if pool:
        la = sorted(float(a["log10_abs_A"]) for a in pool)
        mg = sorted(float(a["f64_margin_deploy_decades"]) for a in pool)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]      # noqa: E731
        print(f"  log10|A|: min {la[0]:.2f}  med {q(la, .5):.2f}  max {la[-1]:.2f}")
        print(f"  deploy margin (decades): min {mg[0]:.2f}  med {q(mg, .5):.2f}  max {mg[-1]:.2f}")
    tot = state.get("totals", {})
    if tot:
        print("  cumulative outcomes: " + "  ".join(f"{k}={v}" for k, v in sorted(tot.items())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=200,
                    help="stop once the pool holds at least this many atoms (default 200)")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="hard cap on tasks consumed this run")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="stop after this many seconds (checked per chunk)")
    ap.add_argument("--chunk", type=int, default=8,
                    help="tasks per durable checkpoint (default 8)")
    ap.add_argument("--refs-only", action="store_true",
                    help="(re)build the reference row and exit")
    ap.add_argument("--force-reschedule", action="store_true",
                    help="accept an incompatible stored schedule and reset the cursor")
    args = ap.parse_args(argv)

    ts.ensure_dirs()
    if args.refs_only:
        for r in build_references():
            print(f"  {r['id']:10s} base_scale={r['base_scale']}  "
                  f"4x={4 * float(r['base_scale']):.6e}  ({r['label']})")
        print(f"-> {ts.rel(ts.REFERENCES)}")
        return 0

    state = ts.load_state()
    if state and not schedule_compatible(state):
        if not args.force_reschedule:
            print("stored cursor was built against a DIFFERENT schedule "
                  "(seed grid / period range / feasibility policy changed).\n"
                  f"  stored: {json.dumps(state.get('schedule'))}\n"
                  f"  now   : {json.dumps(schedule_meta())}\n"
                  "Re-run with --force-reschedule to reset the cursor (the pool and "
                  "verdicts are kept; ids are content hashes).", file=sys.stderr)
            return 2
        state["cursor"] = 0
    state.setdefault("cursor", 0)
    state.setdefault("totals", {})
    state["schedule"] = schedule_meta()

    pool = ts.load_pool()
    have = {a["id"] for a in pool}
    n_start, cursor_start = len(pool), state["cursor"]

    mp.mp.dps = NUCLEUS_DPS
    digits = dcf.emit_digits_for_fw(1e-20)
    wall = brs.deploy_wall_log10()                                   # 1280 x ss4
    field_wall = brs.deploy_wall_log10(brs.FIELD_W, brs.FIELD_SS)
    perm = _permutation(seeds_per_degree())
    run_id = time.strftime("%Y%m%dT%H%M%S", time.localtime())

    print(f"triage pool: target {args.target}, have {len(pool)}, "
          f"cursor {state['cursor']}/{total_tasks()}")
    print(f"  schedule v{SCHEDULE_VERSION}: degrees {DEGREES}, "
          f"{SEED_ANG}ang x {SEED_RAD}rad/degree, periods {PERIOD_MIN}-{PERIOD_MAX}")
    print(f"  cut: A-feasibility only (deploy margin >= {MARGIN_MIN_DECADES} decade "
          f"at {brs.DEPLOY_W} x ss{brs.DEPLOY_SS}); NO period stratification")

    t0 = time.time()
    consumed = 0
    while len(pool) < args.target and state["cursor"] < total_tasks():
        if args.max_tasks is not None and consumed >= args.max_tasks:
            break
        if args.time_budget is not None and (time.time() - t0) >= args.time_budget:
            break
        batch: list[dict] = []
        counts = Counter()
        for _ in range(args.chunk):
            if state["cursor"] >= total_tasks():
                break
            deg, sidx = schedule_task(state["cursor"], perm)
            rows, c = run_task(deg, sidx, state["cursor"], run_id, _SEEDS,
                               digits, wall, field_wall, have)
            batch.extend(rows)
            counts.update(c)
            state["cursor"] += 1
            consumed += 1
        # durable order: atoms first, THEN the cursor. An interruption between the two
        # re-runs one chunk, and content-hash ids make the re-run a no-op.
        written = ts.append_atoms(batch)
        pool.extend(batch)
        for k, v in counts.items():
            state["totals"][k] = state["totals"].get(k, 0) + v
        state["last_run_id"] = run_id
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        ts.save_state(state)
        rate = consumed / max(1e-9, time.time() - t0)
        print(f"  cursor {state['cursor']:5d}  pool {len(pool):5d} "
              f"(+{written})  {rate:.2f} task/s  {time.time() - t0:6.1f}s", flush=True)

    elapsed = time.time() - t0
    state["last_run"] = dict(run_id=run_id, tasks=consumed, seconds=round(elapsed, 2),
                             atoms_added=len(pool) - n_start,
                             cursor_from=cursor_start, cursor_to=state["cursor"])
    ts.save_state(state)
    write_neighbors(pool)
    if not ts.REFERENCES.exists():
        build_references()

    report(pool, state)
    print(f"\n  this run: {consumed} tasks, {len(pool) - n_start} atoms, {elapsed:.1f}s "
          f"({consumed / max(1e-9, elapsed):.2f} task/s, "
          f"{(len(pool) - n_start) / max(1e-9, elapsed):.2f} atom/s)")
    print(f"-> {ts.rel(ts.POOL)}")
    return 0


_SEEDS: dict[int, list] = {}     # degree -> ring seed list (built once per process)

if __name__ == "__main__":
    raise SystemExit(main())
