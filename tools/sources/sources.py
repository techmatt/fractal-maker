#!/usr/bin/env python
"""The generation algorithms, one per sheet.

Each `src_*` function returns `(atoms, stats)` where `atoms` is a de-duplicated list
of `atom_lib` records (all descriptors, no feasibility exclusion) and `stats` records
what the source refused to produce and why. A source that cannot reach the target
count returns what it has — **never padded from another source**, because that would
destroy the only comparison this batch exists to make.

Priority order is the prompt's: probe (the baseline everything is read against, and
the cheapest, so it also proves the pipeline end to end) -> label-seeded ->
neighbourhood -> atlas -> complete low-n -> descent -> Misiurewicz -> tuning.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "sourcing", REPO_ROOT / "tools" / "corpus",
          REPO_ROOT / "tools" / "descent"):
    sys.path.insert(0, str(p))

import mpmath as mp                       # noqa: E402
import deep_center_finder as dcf          # noqa: E402
import build_minibrot_roster as brs       # noqa: E402
import atom_lib as al                     # noqa: E402

LABEL_BATCHES = REPO_ROOT / "data" / "label_corpus" / "batches"
ATLAS_ROOT = REPO_ROOT / "data" / "atlas"
TEST_RENDERS = REPO_ROOT / "data" / "test_renders.json"


def _dedup(atoms):
    """Collapse to one record per sector-canonical nucleus, first occurrence wins."""
    seen, out = set(), []
    for a in atoms:
        if a["id"] in seen:
            continue
        seen.add(a["id"])
        out.append(a)
    return out


def _budget_left(deadline):
    return deadline is None or time.time() < deadline


# --------------------------------------------------------------------------- #
# 1. Probe sampling, unstratified — the baseline
# --------------------------------------------------------------------------- #
def src_probe(target=200, *, seed_ang=96, seed_rad=12, period_max=24, deadline=None,
              log=print):
    """Newton over a ring-seed grid across the whole region, every minimal nucleus
    kept. No period stratification, no per-cell cap, **and no feasibility cut** — the
    only difference from the triage enumerator, per the addendum."""
    al.set_precision()
    seeds = brs.ring_seeds(2, seed_ang, seed_rad)
    order = np.random.default_rng(20260730).permutation(len(seeds))
    atoms, stats = [], defaultdict(int)
    t0 = time.time()
    for si in order:
        if len(atoms) >= target or not _budget_left(deadline):
            break
        sr, si_ = seeds[int(si)]
        for p in range(3, period_max + 1):
            stats["solves"] += 1
            rec, why = al.solve_nucleus(
                mp.mpc(sr, si_), p, source="probe",
                provenance={"seed_re": float(sr), "seed_im": float(si_), "seed_index": int(si)})
            if rec is None:
                stats[why] += 1
                continue
            atoms.append(rec)
        atoms = _dedup(atoms)
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"    probe: {len(atoms)} atoms, {stats['solves']} solves, {stats['seconds']}s")
    return _dedup(atoms), dict(stats)


# --------------------------------------------------------------------------- #
# 2. Labeled-location seed
# --------------------------------------------------------------------------- #
def _deg2_good_labels(min_score=3):
    """Degree-2 mandelbrot label-corpus rows resolving to score >= `min_score`.

    Scores are resolved through `label_store.resolve_score` (sidecars + amendment
    overlay), NOT read raw off `row["label"]["score"]` — 9 batches carry null there
    and a raw read undercounts badly."""
    sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
    import label_store as ls
    rows = []
    for bdir in sorted(LABEL_BATCHES.iterdir()):
        f = bdir / "images.jsonl"
        if not f.exists():
            continue
        labels = ls.sidecar_for(bdir.name)
        amend = ls.amendments_for(bdir.name)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rd = r.get("render", {})
            fam = rd.get("family") or rd.get("fractal_type")
            if fam not in (None, "mandelbrot"):
                continue
            if rd.get("c_re") is not None:          # a Julia row in a pre-family batch
                continue
            try:
                score = ls.resolve_score(r, labels, amend)
            except Exception:
                score = (r.get("label") or {}).get("score")
            if score is not None and score >= min_score:
                rows.append({"batch": bdir.name, "image_id": r.get("image_id"),
                             "cx": rd.get("cx"), "cy": rd.get("cy"),
                             "fw": rd.get("fw"), "score": score})
    return rows


def src_label_seeded(target=200, *, min_score=3, period_max=64, deadline=None, log=print):
    """Solve for the nucleus at or near each degree-2 q3/q4 label-corpus location.

    `near` is scaled to the location's OWN frame width: a nucleus counts as "at" a
    labelled view only if it lies within that view, which is what makes this a
    label-seeded source rather than a global scan."""
    al.set_precision()
    rows = _deg2_good_labels(min_score)
    log(f"    supply: {len(rows)} degree-2 label rows with score >= {min_score}")
    atoms, stats = [], defaultdict(int)
    stats["supply"] = len(rows)
    t0 = time.time()
    for i, r in enumerate(rows):
        if not _budget_left(deadline):
            stats["budget_stopped_at"] = i
            break
        try:
            seed = mp.mpc(mp.mpf(r["cx"]), mp.mpf(r["cy"]))
            near = float(r["fw"]) * 0.75 if r.get("fw") else 1e-3
        except Exception:
            stats["bad_coords"] += 1
            continue
        rec, why = al.identify_nucleus(
            seed, period_min=1, period_max=period_max, near=near, source="label_seeded",
            provenance={"batch": r["batch"], "image_id": r["image_id"],
                        "label_score": r["score"], "label_fw": r.get("fw")})
        if rec is None:
            stats[why] += 1
            continue
        atoms.append(rec)
        if len(_dedup(atoms)) >= target and target > 0:
            break
    stats["seconds"] = round(time.time() - t0, 1)
    out = _dedup(atoms)
    stats["distinct"] = len(out)
    log(f"    label_seeded: {len(out)} distinct nuclei from {len(rows)} labelled rows "
        f"({stats['seconds']}s)")
    return out, dict(stats)


# --------------------------------------------------------------------------- #
# 3. Neighbourhood expansion
# --------------------------------------------------------------------------- #
def src_neighborhood(parents, target=200, *, radii=(2.0, 8.0, 32.0), per_parent=6,
                     period_max=None, period_headroom=3.0, period_cap=200,
                     deadline=None, seed=20260730, log=print):
    """Probe a disc around each parent nucleus, at comparable and smaller scale.

    Tests whether richness is locally correlated — assumed everywhere, never checked.
    Probe points are drawn at radii measured in units of the PARENT's own window
    scale, so "nearby" means the same thing for a shallow and a deep parent.

    The period search is scaled to the PARENT's period (`period_headroom` x, capped),
    not fixed: components in the neighbourhood of a period-P minibrot have periods of
    order P and above, so a flat ceiling silently finds nothing around deep parents.
    A first run with a flat `period_max=32` returned 15 atoms from 360 probes (4%) —
    the ceiling, not the source, was the limit."""
    al.set_precision()
    rng = random.Random(seed)
    atoms, stats = [], defaultdict(int)
    parent_ids = {p["id"] for p in parents}
    stats["parents"] = len(parents)
    t0 = time.time()
    for pi, par in enumerate(parents):
        if len(_dedup(atoms)) >= target or not _budget_left(deadline):
            stats["budget_stopped_at"] = pi
            break
        w = float(par["window_scale"])
        pcx, pcy = mp.mpf(par["cx"]), mp.mpf(par["cy"])
        for _ in range(per_parent):
            rad = rng.choice(radii) * w
            th = rng.uniform(0, 2 * math.pi)
            seed_c = mp.mpc(pcx + mp.mpf(rad * math.cos(th)),
                            pcy + mp.mpf(rad * math.sin(th)))
            stats["probes"] += 1
            pmax = period_max or min(period_cap,
                                     max(24, int(period_headroom * par["period"])))
            rec, why = al.identify_nucleus(
                seed_c, period_min=1, period_max=pmax, near=rad * 4,
                source="neighborhood",
                provenance={"parent_id": par["id"], "parent_period": par["period"],
                            "probe_period_max": pmax,
                            "radius_over_parent_w": rad / w if w else None})
            if rec is None:
                stats[why] += 1
                continue
            if rec["id"] in parent_ids:
                stats["hit_parent"] += 1
                continue
            atoms.append(rec)
    stats["seconds"] = round(time.time() - t0, 1)
    out = _dedup(atoms)
    log(f"    neighborhood: {len(out)} distinct from {stats['probes']} probes around "
        f"{len(parents)} parents ({stats['seconds']}s)")
    return out, dict(stats)


# --------------------------------------------------------------------------- #
# 4. Atlas mining — the curated, hand-named supply
# --------------------------------------------------------------------------- #
def curated_seeds() -> list[dict]:
    """Every hand-curated degree-2 mandelbrot location in the repo.

    `data/mandelbrot_named_seeds.json` does NOT exist. The real curated supply is
    source constants, not data files: `emit_deep_pool.SEEDS` (10 hand-named
    seahorse/elephant-valley entries), the one mandelbrot anchor in
    `verify_v6_gate.ANCHORS`, and `data/test_renders.json` (4 hand-picked q3
    locations). Collected here so the sheet's provenance is explicit."""
    out = []
    try:
        sys.path.insert(0, str(REPO_ROOT / "tools" / "sourcing"))
        import emit_deep_pool as edp
        for s in edp.SEEDS:
            kind, sre, sim, period, preperiod, note = s
            out.append({"name": note, "cx": str(sre), "cy": str(sim),
                        "period": period, "preperiod": preperiod, "kind": kind,
                        "origin": "tools/sourcing/emit_deep_pool.py::SEEDS"})
    except Exception as e:                      # pragma: no cover
        out.append({"_error": f"emit_deep_pool: {e}"})
    try:
        sys.path.insert(0, str(REPO_ROOT / "tools" / "atlas"))
        import verify_v6_gate as vg
        for a in vg.ANCHORS:
            label, family = a[0], a[1]
            if family != "mandelbrot":
                continue
            out.append({"name": label, "cx": str(a[2]), "cy": str(a[3]),
                        "period": None, "preperiod": None, "kind": "anchor",
                        "origin": "tools/atlas/verify_v6_gate.py::ANCHORS"})
    except Exception:
        pass
    if TEST_RENDERS.exists():
        doc = json.loads(TEST_RENDERS.read_text(encoding="utf-8"))
        for loc in doc.get("locations", []):
            if loc.get("system") != "mandelbrot":
                continue
            out.append({"name": f'test_render {loc.get("id")}', "cx": loc["cx"],
                        "cy": loc["cy"], "period": None, "preperiod": None,
                        "kind": "hand_picked_q3",
                        "origin": "data/test_renders.json"})
    return [o for o in out if "_error" not in o]


def src_atlas(target=200, *, period_max=64, deadline=None, log=print):
    """Probe each curated location for its nucleus. Small named supply by construction —
    the sheet is filled by neighbourhood expansion around the hits (the prompt's own
    instruction), and the split between the two is reported."""
    al.set_precision()
    seeds = curated_seeds()
    stats = defaultdict(int)
    stats["curated_supply"] = len(seeds)
    atoms = []
    t0 = time.time()
    for s in seeds:
        if not _budget_left(deadline):
            break
        seed = mp.mpc(mp.mpf(s["cx"]), mp.mpf(s["cy"]))
        if s.get("period"):
            rec, why = al.solve_nucleus(
                seed, int(s["period"]), source="atlas",
                provenance={"name": s["name"], "origin": s["origin"], "tier": "curated"})
        else:
            rec, why = al.identify_nucleus(
                seed, period_min=1, period_max=period_max, near=1e-2, source="atlas",
                provenance={"name": s["name"], "origin": s["origin"], "tier": "curated"})
        if rec is None:
            stats[f"curated_{why}"] += 1
            continue
        atoms.append(rec)
    atoms = _dedup(atoms)
    stats["curated_nuclei"] = len(atoms)
    log(f"    atlas: {len(atoms)} nuclei from {len(seeds)} curated locations")
    if len(atoms) < target and _budget_left(deadline):
        grown, gstats = src_neighborhood(atoms, target=target - len(atoms),
                                         per_parent=max(4, (target // max(1, len(atoms))) + 4),
                                         deadline=deadline, log=log)
        for g in grown:
            g["source"] = "atlas"
            g["provenance"] = dict(g["provenance"], tier="neighborhood_of_curated")
        atoms = _dedup(atoms + grown)
        stats["expansion"] = gstats
        stats["expanded_nuclei"] = len(atoms) - stats["curated_nuclei"]
    stats["seconds"] = round(time.time() - t0, 1)
    return atoms, dict(stats)


# --------------------------------------------------------------------------- #
# 5. Low-n complete enumeration
# --------------------------------------------------------------------------- #
# --- exact period-n polynomials: Q_n = p_n / prod_{d|n,d<n} Q_d ------------- #
def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _pdivexact(a, b):
    """Exact integer polynomial division (ascending coefficients); b must divide a."""
    a = a[:]
    q = [0] * (len(a) - len(b) + 1)
    for i in range(len(q) - 1, -1, -1):
        c = a[i + len(b) - 1]
        k = c // b[-1]
        q[i] = k
        if k:
            for j, y in enumerate(b):
                a[i + j] -= k * y
    if any(a):
        raise ValueError("polynomial division left a remainder")
    return q


def period_polynomials(nmax: int) -> dict[int, list[int]]:
    """`Q_n(c)`, the polynomial whose roots are EXACTLY the period-n nuclei.

    `p_1 = c`, `p_{k+1} = p_k^2 + c` (so `p_n(c) = z_n` at the critical orbit), then
    divide out every lower-period factor. `deg Q_n == nu(n)` by construction, which is
    also the completeness guarantee: root-finding `Q_n` returns the whole period-n
    population with no search and no possibility of a miss."""
    ps = {1: [0, 1]}
    for k in range(1, nmax):
        sq = _pmul(ps[k], ps[k])
        sq[1] += 1                                   # + c
        ps[k + 1] = sq
    qs: dict[int, list[int]] = {}
    for n in range(1, nmax + 1):
        num = ps[n]
        for d in range(1, n):
            if n % d == 0:
                num = _pdivexact(num, qs[d])
        qs[n] = num
    return qs


def _approx_roots(coeffs_asc: list[int]) -> list[complex]:
    """Approximate roots of an exact integer polynomial via the companion matrix.

    Exact-arithmetic root-finding (`mp.polyroots`) needs ~10x degree guard bits and is
    far too slow past degree ~60. numpy only has to land each root inside its Newton
    basin — the polish step below recovers full precision — and the ROOT COUNT is what
    guarantees completeness, which the companion matrix gives exactly."""
    scale = max(abs(c) for c in coeffs_asc if c) or 1
    desc = [float(c) / float(scale) for c in reversed(coeffs_asc)]
    return list(np.roots(np.array(desc, dtype=np.float64)))


# NOTE — why there is no cleverer root-finder here.
# The companion matrix locates roots to ~1e-8, which stops being enough to separate
# neighbouring nuclei around degree 63 (n=7): several approximate roots then polish into
# the SAME basin and the population comes out short (53/63 at n=7, 85/120 at n=8).
# Two escalations were measured and rejected:
#   * higher working precision — 60 / 200 / 400 dps give IDENTICAL counts, proving the
#     loss is in the seeds, not the arithmetic;
#   * Aberth-Ehrlich simultaneous root-finding (which does keep roots apart) — O(deg^2)
#     per iteration in pure-Python mpmath, minutes at degree 120, not worth the night.
# So this source SHIPS ONLY THE PERIODS THAT CAME OUT PROVABLY COMPLETE and reports the
# rest as attempted-and-short. That keeps "complete population" a true statement rather
# than an approximate one. Getting n>=7 needs a compiled simultaneous root-finder.


def src_complete_low_n(period_max=8, *, deadline=None, log=print):
    """Every component of exact period n, for n up to `period_max`.

    A **complete population**, not a sample. `Q_n(c)` — the period-n polynomial with
    every lower-period factor divided out — has `deg Q_n = nu(n)` exactly, so its roots
    ARE the period-n nuclei and completeness is a construction, not a search that might
    have missed something. Each approximate root is then polished by the same Newton
    solver every other source uses, and the distinct count is checked back against
    `nu(n)`; any shortfall is reported per period rather than hidden.

    (An earlier grid-and-Newton version passed its own completeness check *spuriously*
    because duplicate real-axis atoms inflated the count — see `atom_lib.SNAP_EPS`.
    That is exactly the failure this construction removes.)

    Completeness also buys the one place an EXACT satellite fraction is available:
    `sum_{q|n,q<n} phi(n/q)*nu(q)` counts them without needing a classifier. (Running
    that count against the classical atom-domain criterion is what proved the criterion
    wrong — 17 satellites called at n=6 where theory says 7. See `atom_lib`.)
    """
    al.set_precision()
    stats = defaultdict(int)
    by_period, report = {}, []
    t0 = time.time()
    qs = period_polynomials(period_max)
    stats["poly_seconds"] = round(time.time() - t0, 1)
    for n in range(1, period_max + 1):
        if not _budget_left(deadline):
            stats["budget_stopped_at_period"] = n
            break
        want = al.nu(n)
        tn = time.time()
        try:
            approx = _approx_roots(qs[n])
        except Exception as e:
            stats[f"roots_failed_n{n}"] = str(e)[:120]
            log(f"    n={n:2d}: root-finding FAILED ({str(e)[:60]}) — period omitted")
            continue
        found: dict[str, dict] = {}
        for z in approx:
            rec, why = al.solve_nucleus(               # polish to full precision
                mp.mpc(float(z.real), float(z.imag)), n, source="complete_low_n",
                provenance={"n": n, "method": "Q_n roots + Newton polish"},
                want_embedding=(n <= 12))
            if rec is None:
                stats[f"polish_{why}_n{n}"] += 1
                continue
            found.setdefault(rec["id"], rec)
        by_period[n] = list(found.values())
        report.append({"period": n, "expected": want, "roots": len(approx),
                       "distinct": len(found), "complete": len(found) == want,
                       "seconds": round(time.time() - tn, 1)})
        log(f"    n={n:2d}: {len(found)}/{want} components "
            f"({time.time()-tn:.1f}s){'' if len(found) == want else '  INCOMPLETE'}")
    # ship ONLY the provably complete periods, so the sheet's premise stays exactly true
    complete = {r["period"] for r in report if r["complete"]}
    short = [{"period": r["period"], "distinct": r["distinct"],
              "expected_total": r["expected"]}
             for r in report if not r["complete"] and r["period"] > 1]
    atoms = [a for n in sorted(by_period) if n in complete for a in by_period[n]]
    stats["per_period"] = report
    stats["shipped_periods"] = sorted(complete)
    stats["attempted_but_short"] = short
    stats["theorem_satellites"] = al.theorem_satellite_fraction(
        {n: v for n, v in by_period.items() if n in complete})
    stats["seconds"] = round(time.time() - t0, 1)
    stats["method"] = ("exact period-n polynomial Q_n (deg = nu(n)) -> companion-matrix "
                       "roots -> Newton polish; complete by construction, count-verified")
    return _dedup(atoms), dict(stats)


# --------------------------------------------------------------------------- #
# 6. Descent-found
# --------------------------------------------------------------------------- #
def _atlas_pool_rows(limit=4000):
    """The guided-descend walker's already-generated candidate pools.
    `data/guided_descend/` does not exist; the pools live under `data/atlas/round*/`."""
    rows = []
    for pool in sorted(ATLAS_ROOT.glob("round*/*_pool/pool.jsonl")):
        for line in pool.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"cx": r["cx"], "cy": r["cy"], "fw": r["fw"],
                         "pool": f"{pool.parent.parent.name}/{pool.parent.name}",
                         "depth": r.get("depth"), "idx": r.get("idx")})
            if len(rows) >= limit:
                return rows
    return rows


def src_descent(target=200, *, period_max=48, deadline=None, seed=20260730, log=print):
    """Solve for the nucleus wherever the existing descent machinery landed."""
    al.set_precision()
    rows = _atlas_pool_rows()
    random.Random(seed).shuffle(rows)
    stats = defaultdict(int)
    stats["supply"] = len(rows)
    atoms = []
    t0 = time.time()
    for i, r in enumerate(rows):
        if len(_dedup(atoms)) >= target or not _budget_left(deadline):
            stats["consumed"] = i
            break
        rec, why = al.identify_nucleus(
            mp.mpc(float(r["cx"]), float(r["cy"])), period_min=1, period_max=period_max,
            near=float(r["fw"]) * 0.75, source="descent",
            provenance={"pool": r["pool"], "depth": r["depth"], "pool_idx": r["idx"],
                        "pool_fw": r["fw"]})
        if rec is None:
            stats[why] += 1
            continue
        atoms.append(rec)
    stats.setdefault("consumed", len(rows))
    stats["seconds"] = round(time.time() - t0, 1)
    out = _dedup(atoms)
    log(f"    descent: {len(out)} distinct nuclei from {stats['consumed']} pool rows "
        f"({stats['seconds']}s)")
    return out, dict(stats)


# --------------------------------------------------------------------------- #
# 7. Misiurewicz-anchored
# --------------------------------------------------------------------------- #
def src_misiurewicz(target=200, *, preperiods=(2, 3, 4, 5, 6, 8), periods=(1, 2, 3, 4, 5, 6),
                    period_max=32, deadline=None, seed=20260730, log=print):
    """Nuclei near preperiodic (Misiurewicz) points.

    Misiurewicz points stay ON the boundary at every scale, so a neighbourhood of one
    is dense in small components — the premise being tested is whether the nuclei that
    live there differ in kind from those a plain scan finds."""
    al.set_precision()
    rng = random.Random(seed)
    stats = defaultdict(int)
    atoms, mis_points = [], []
    t0 = time.time()
    seeds = brs.ring_seeds(2, 48, 6)
    rng.shuffle(seeds)
    for (sr, si) in seeds:
        if len(mis_points) >= 40 or not _budget_left(deadline):
            break
        k, nper = rng.choice(preperiods), rng.choice(periods)
        stats["mis_solves"] += 1
        r = dcf.newton_misiurewicz(mp.mpc(sr, si), k, nper, degree=2,
                                   max_steps=al.NEWTON_STEPS)
        if not r.converged or abs(r.c) > 2.2:
            stats["mis_no_converge"] += 1
            continue
        if not dcf.is_minimal_misiurewicz(r.c, k, nper, degree=2):
            stats["mis_not_minimal"] += 1
            continue
        mis_points.append({"c": r.c, "k": k, "n": nper})
    stats["misiurewicz_points"] = len(mis_points)
    log(f"    misiurewicz: {len(mis_points)} anchor points")
    for mi, m in enumerate(mis_points):
        if len(_dedup(atoms)) >= target or not _budget_left(deadline):
            stats["budget_stopped_at"] = mi
            break
        for scale in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            for _ in range(3):
                th = rng.uniform(0, 2 * math.pi)
                seed_c = m["c"] + mp.mpc(scale * math.cos(th), scale * math.sin(th))
                stats["probes"] += 1
                rec, why = al.identify_nucleus(
                    seed_c, period_min=1, period_max=period_max, near=scale * 4,
                    source="misiurewicz",
                    provenance={"preperiod": m["k"], "mis_period": m["n"],
                                "probe_scale": scale})
                if rec is None:
                    stats[why] += 1
                    continue
                atoms.append(rec)
    stats["seconds"] = round(time.time() - t0, 1)
    out = _dedup(atoms)
    log(f"    misiurewicz: {len(out)} distinct nuclei ({stats['seconds']}s)")
    return out, dict(stats)
