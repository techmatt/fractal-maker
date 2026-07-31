#!/usr/bin/env python
"""Deploy `radial_rings` as a **search screen**: enumerate a large nucleus pool, score
every atom at a tiny screening resolution, and keep the top few hundred for wall-fidelity
rendering.

The point is not to pick winners. It is to stop spending Matt's attention on tiles that
were never candidates — the screen is a **floor**, deliberately loose, and where that
floor sits is his call once he sees the distribution. Nothing here fits a model or
proposes a final threshold.

Two phases, both resumable and both capped by a wall-clock budget:

  * **Enumerate** — Newton over a ring-seed grid across the whole region (the same
    machinery every other source uses), 4 worker PROCESSES, degree 2. Ids are the shared
    content hash, so the pool dedups against itself and cross-references the triage pool
    and the source sheets for free.
  * **Screen** — one `render-one --dump-field` per atom at 64x36 (~2 ms of compute,
    ~76 ms wall including process spawn), then `radial_rings`. Validated against the
    320x180 measure at spearman 0.87, with the reference/triage separation intact at
    screening resolution.

Run:  uv run python tools/orbital/screen_pool.py --target 10000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "sources", REPO_ROOT / "tools" / "sourcing",
          REPO_ROOT / "tools" / "descent", REPO_ROOT / "tools" / "explorer",
          REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

POOL = "data/orbital/screen_pool.jsonl"
SCORES = "data/orbital/screen_scores.jsonl"
REPORT = "data/orbital/screen_report.json"

def _log(*a):
    print(*a, flush=True)


WORKERS = 4                 # concurrent PROCESSES (CLAUDE.md cap), both phases
THREADS = 3
SEED_ANG, SEED_RAD = 256, 24
PERIOD_MIN, PERIOD_MAX = 3, 40
KEEP_TOP = 300              # how many survivors to hand to wall-fidelity rendering


# --------------------------------------------------------------------------- #
# enumeration (worker runs in its own process)
# --------------------------------------------------------------------------- #
def _enumerate_chunk(args):
    seeds, pmin, pmax = args
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "sources"))
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "sourcing"))
    import mpmath as mp
    import atom_lib as al
    al.set_precision()
    out, solves = [], 0
    for (sr, si) in seeds:
        for p in range(pmin, pmax + 1):
            solves += 1
            rec, _why = al.solve_nucleus(mp.mpc(sr, si), p, source="orbital_screen",
                                         provenance={"seed_re": float(sr),
                                                     "seed_im": float(si)},
                                         want_embedding=False)
            if rec is not None:
                out.append({k: rec[k] for k in
                            ("id", "period", "cx", "cy", "window_scale", "family",
                             "degree", "log10_abs_A", "f64_margin_deploy_decades")})
    return out, solves


def enumerate_pool(target, *, workers=WORKERS, time_budget=None, log=_log) -> list[dict]:
    import build_minibrot_roster as brs
    import paths
    ppath = paths.durable(POOL, mkparents=True)
    have: dict[str, dict] = {}
    if ppath.exists():
        for line in ppath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                have[r["id"]] = r
        log(f"  resuming: {len(have)} atoms already enumerated")

    seeds = brs.ring_seeds(2, SEED_ANG, SEED_RAD)
    order = np.random.default_rng(20260730).permutation(len(seeds))
    seeds = [seeds[int(i)] for i in order]
    chunk = 24
    batches = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]
    t0, solves, consumed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        it = iter(batches)
        for _ in range(workers * 2):
            b = next(it, None)
            if b is not None:
                futs.append(ex.submit(_enumerate_chunk, (b, PERIOD_MIN, PERIOD_MAX)))
        while futs:
            done, futs = futs[0], futs[1:]
            recs, s = done.result()
            solves += s
            consumed += 1
            for r in recs:
                have.setdefault(r["id"], r)
            el = time.time() - t0
            if consumed % 8 == 0:
                log(f"  {len(have):6d} atoms  {solves:8d} solves  "
                    f"{len(have)/max(1e-9, el):6.1f} atom/s  {el:6.1f}s")
            stop = (len(have) >= target
                    or (time_budget is not None and el >= time_budget))
            if not stop:
                b = next(it, None)
                if b is not None:
                    futs.append(ex.submit(_enumerate_chunk, (b, PERIOD_MIN, PERIOD_MAX)))
            elif not futs:
                break
    rows = list(have.values())
    with open(ppath, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    log(f"  enumerated {len(rows)} atoms from {solves} Newton solves in "
        f"{time.time()-t0:.1f}s -> {POOL}")
    return rows


# --------------------------------------------------------------------------- #
# screening
# --------------------------------------------------------------------------- #
def screen(rows, *, workers=WORKERS, log=_log, time_budget=None) -> list[dict]:
    import field_metrics as fm
    import render_core as rc
    import paths
    spath = paths.durable(SCORES, mkparents=True)
    done_ids = set()
    scored: list[dict] = []
    if spath.exists():
        for line in spath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                scored.append(r)
                done_ids.add(r["id"])
        log(f"  resuming: {len(done_ids)} already screened")
        # Fail BEFORE spending the screening budget, not after. Rows resumed from disk
        # were measured under whatever cap policy was live when they were written; the
        # rows we are about to append carry today's. Appending across the raise would
        # write one file holding two incommensurable populations.
        fm.require_one_policy((f"resumed from {SCORES}", scored),
                              ("this run", [{fm.POLICY_KEY: fm.policy_token()}]),
                              what="resumed screen scores against this run's cap policy")
    todo = [r for r in rows if r["id"] not in done_ids]
    log(f"  screening {len(todo)} atoms at {fm.SCREEN_W}x{fm.SCREEN_H} ss{fm.SCREEN_SS}")
    t0, n = time.time(), [0]
    errs = []
    lock = __import__("threading").Lock()
    fh = open(spath, "a", encoding="utf-8")

    def work(a):
        if time_budget is not None and (time.time() - t0) > time_budget:
            return
        try:
            fw = float(a["window_scale"]) * 4
            m = fm.measure_location(a["cx"], a["cy"], fw, rc.auto_maxiter(fw),
                                    width=fm.SCREEN_W, height=fm.SCREEN_H,
                                    ss=fm.SCREEN_SS, family=a["family"], threads=THREADS)
            row = {"id": a["id"], "period": a["period"],
                   "log10_abs_A": a["log10_abs_A"],
                   "f64_margin_deploy_decades": a.get("f64_margin_deploy_decades"),
                   "radial_rings": m["radial_rings"],
                   "radial_rings_p90": m["radial_rings_p90"],
                   "cycles_spanned": m["cycles_spanned"],
                   "interior_fraction": m["interior_fraction"],
                   # the iteration-cap policy this score was measured under; every
                   # aggregation below refuses to mix two of them.
                   fm.POLICY_KEY: m[fm.POLICY_KEY]}
            with lock:
                scored.append(row)
                fh.write(json.dumps(row) + "\n")
        except Exception as e:
            with lock:
                errs.append({"id": a["id"], "error": str(e)[:160]})
        n[0] += 1
        if n[0] % 500 == 0 or n[0] == len(todo):
            el = time.time() - t0
            log(f"  {n[0]:6d}/{len(todo)}  {n[0]/max(1e-9, el):6.1f} atom/s  {el:6.1f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    el = time.time() - t0
    log(f"  screened {n[0] - len(errs)} ok / {len(errs)} failed in {el:.1f}s "
        f"({n[0]/max(1e-9, el):.1f} atom/s)")
    return scored, errs, el


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--enum-budget", type=float, default=1800.0, help="seconds")
    ap.add_argument("--screen-budget", type=float, default=1800.0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--keep-top", type=int, default=KEEP_TOP)
    ap.add_argument("--skip-enum", action="store_true")
    args = ap.parse_args(argv)
    if args.workers > 4:
        print("refusing >4 concurrent processes (CLAUDE.md)", file=sys.stderr)
        return 2
    import paths

    print(f"orbital screen — target {args.target} nuclei, degree 2, "
          f"periods {PERIOD_MIN}-{PERIOD_MAX}")
    if args.skip_enum:
        rows = [json.loads(l) for l in
                paths.durable(POOL).read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  loaded {len(rows)} atoms from {POOL}")
    else:
        print("\nphase 1 — enumerate")
        rows = enumerate_pool(args.target, workers=args.workers,
                              time_budget=args.enum_budget)

    print("\nphase 2 — screen")
    scored, errs, el = screen(rows, workers=args.workers, time_budget=args.screen_budget)

    # The aggregation guard: percentiles, the implied floor and the keep-top ranking
    # below are all cross-atom comparisons, so they must be within ONE cap policy.
    import field_metrics as fm
    policy = fm.require_one_policy(("screened", scored),
                                   what="the screen distribution and keep-top ranking")

    v = np.array([s["radial_rings"] for s in scored], dtype=float)
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dist = {f"p{p}": round(float(np.percentile(v, p)), 1) for p in pct}
    top = sorted(scored, key=lambda s: -s["radial_rings"])[:args.keep_top]
    cut = top[-1]["radial_rings"] if top else None
    rep = {
        "measure": "radial_rings (median colour-cycle crossings over 64 rays)",
        "screen_geometry": [64, 36, 1],
        # Provenance, not decoration: every number below is only comparable to another
        # report carrying the same token (docs/design/auto_maxiter.md).
        fm.POLICY_KEY: policy,
        "maxiter_policy": fm.describe_policy(policy),
        "pool_n": len(rows), "scored_n": len(scored), "failed_n": len(errs),
        "screen_seconds": round(el, 1),
        "screen_rate_atoms_per_s": round(len(scored) / max(1e-9, el), 1),
        "distribution": dist,
        "min": round(float(v.min()), 1), "max": round(float(v.max()), 1),
        "keep_top": args.keep_top,
        "implied_cut_at_keep_top": cut,
        "frac_pool_above_cut": round(float((v >= (cut or 0)).mean()), 4),
        "top_ids": [t["id"] for t in top],
        "note": ("The screen is a FLOOR, not a ranker. No threshold is proposed as final "
                 "— the distribution is here so the floor can be chosen after looking at "
                 "it. Nothing was fitted."),
        "errors_sample": errs[:10],
    }
    paths.durable(REPORT, mkparents=True).write_text(json.dumps(rep, indent=2) + "\n",
                                                     encoding="utf-8")
    print(f"\ndistribution of radial_rings over {len(scored)} screened atoms:")
    print("  " + "  ".join(f"{k} {vv}" for k, vv in dist.items()))
    print(f"  min {rep['min']}  max {rep['max']}")
    print(f"  keeping top {args.keep_top} implies a floor at {cut} "
          f"({rep['frac_pool_above_cut']:.1%} of the pool)")
    print(f"-> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
