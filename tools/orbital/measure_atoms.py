#!/usr/bin/env python
"""Measure the orbital-falloff metrics over every atom already enumerated: the 200
triage-wall atoms, the full source-sheet populations, and the two references.

No wall-fidelity re-rendering — each atom's 4x judging frame is re-dumped as a raw
scalar field at 320x180 ss1, which the engine produces in ~28 ms. (The colour PNGs the
sheets already hold cannot be used: the palette wraps `t` mod 1, so a tile pins
`smooth_iter` only modulo 40 iterations, which is exactly the quantity being measured.)

Also runs the **maxiter stability check**: a cap low enough to clip escape times would
depress `cycles_spanned` artificially, so a sample is re-measured at 2x and 4x maxiter
and the drift reported.

Writes `data/orbital/measures.jsonl` (durable — every atom's metrics keyed by the same
content-hash id every other tool uses) and prints the validation outcome.

Run:  uv run python tools/orbital/measure_atoms.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "descent", REPO_ROOT / "tools" / "sources",
          REPO_ROOT / "tools" / "explorer", REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import field_metrics as fm     # noqa: E402
import render_core as rc       # noqa: E402
import triage_store as ts      # noqa: E402
import source_store as ss      # noqa: E402
import paths                   # noqa: E402

OUT = "data/orbital/measures.jsonl"
STABILITY = "data/orbital/maxiter_stability.json"
def _log(*a):
    print(*a, flush=True)


WORKERS = 4                     # concurrent engine processes (CLAUDE.md cap)
THREADS = 3


def collect_atoms() -> list[dict]:
    """Every enumerated atom, tagged with where it came from. Ids are content hashes,
    so an atom in two populations appears once with both tags."""
    seen: dict[str, dict] = {}

    def add(a, group):
        rec = seen.setdefault(a["id"], {
            "id": a["id"], "cx": a["cx"], "cy": a["cy"],
            "window_scale": a["window_scale"], "family": a["family"],
            "degree": a.get("degree"), "period": a.get("period"),
            "log10_abs_A": a.get("log10_abs_A"), "groups": []})
        if group not in rec["groups"]:
            rec["groups"].append(group)

    for a in ts.load_pool():
        add(a, "triage")
    for sid in ss.built_sources():
        for a in ss.load_atoms(sid):
            add(a, f"source:{sid}")
    for r in ts.load_references():
        seen[r["id"]] = {"id": r["id"], "cx": r["cx"], "cy": r["cy"],
                         "window_scale": r["base_scale"], "family": r["family"],
                         "degree": r["degree"], "period": r.get("period"),
                         "log10_abs_A": r.get("log10_abs_A"),
                         "groups": ["reference"], "label": r["label"]}
    return list(seen.values())


def measure_one(a: dict, *, scale=4, width=fm.MEASURE_W, height=fm.MEASURE_H,
                ss_=fm.MEASURE_SS, maxiter_mult=1.0) -> dict:
    fw = float(a["window_scale"]) * scale
    maxiter = max(64, int(rc.auto_maxiter(fw) * maxiter_mult))
    m = fm.measure_location(a["cx"], a["cy"], fw, maxiter, width=width, height=height,
                            ss=ss_, family=a["family"], threads=THREADS)
    return {**{k: a[k] for k in ("id", "degree", "period", "log10_abs_A")},
            "groups": a["groups"], "label": a.get("label"),
            "scale": scale, "fw": f"{fw:.6e}", **m}


def run_batch(atoms, *, workers=WORKERS, log=_log, **kw) -> tuple[list[dict], list[dict]]:
    out, errs, done = [], [], [0]
    t0 = time.time()

    def work(a):
        try:
            out.append(measure_one(a, **kw))
        except Exception as e:
            errs.append({"id": a["id"], "error": str(e)[:200]})
        done[0] += 1
        if done[0] % 100 == 0 or done[0] == len(atoms):
            el = time.time() - t0
            log(f"  {done[0]:5d}/{len(atoms)}  {done[0]/max(1e-9, el):.1f} atom/s  {el:6.1f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, atoms))
    return out, errs


# --------------------------------------------------------------------------- #
# maxiter stability
# --------------------------------------------------------------------------- #
def stability_check(atoms, n=24, log=_log) -> dict:
    """Re-measure a spread of atoms at 2x and 4x maxiter. If the default cap were
    clipping escape times, `cycles_spanned` would rise with the cap — a measure that
    moves here is measuring the cap, not the location."""
    sample = sorted(atoms, key=lambda a: (a.get("log10_abs_A") or 0))[:: max(1, len(atoms) // n)][:n]
    rows = []
    for mult in (1.0, 2.0, 4.0):
        got, _ = run_batch(sample, maxiter_mult=mult, log=lambda *_: None)
        by = {g["id"]: g for g in got}
        rows.append((mult, by))
    base = rows[0][1]
    report = {"n": len(sample), "per_multiplier": [], "worst_drift": {}}
    for mult, by in rows:
        vals = {k: [] for k in ("cycles_spanned", "radial_rings", "falloff_extent")}
        for i, b in base.items():
            if i in by:
                for k in vals:
                    vals[k].append(by[i][k])
        report["per_multiplier"].append(
            {"maxiter_mult": mult,
             **{k: round(float(np.median(v)), 4) for k, v in vals.items() if v}})
    for k in ("cycles_spanned", "radial_rings", "falloff_extent"):
        drift = []
        for i, b in base.items():
            hi = rows[-1][1].get(i)
            if hi and b[k]:
                drift.append(abs(hi[k] - b[k]) / max(1e-9, abs(b[k])))
        report["worst_drift"][k] = round(float(np.percentile(drift, 95)), 4) if drift else None
    log(f"  maxiter stability over {len(sample)} atoms: " +
        ", ".join(f"{k} p95 drift {v:.1%}" for k, v in report["worst_drift"].items() if v is not None))
    return report


# --------------------------------------------------------------------------- #
# validation (§2 of the prompt)
# --------------------------------------------------------------------------- #
MEASURES = ("cycles_spanned", "radial_rings", "radial_rings_p90", "falloff_extent")


def validate(rows: list[dict], log=_log) -> dict:
    refs = [r for r in rows if "reference" in r["groups"]]
    triage = [r for r in rows if "triage" in r["groups"] and "reference" not in r["groups"]]
    out = {"n_refs": len(refs), "n_triage": len(triage), "measures": {}}
    log(f"\nVALIDATION — {len(refs)} references vs {len(triage)} triage atoms, at 4x")
    log(f"  {'measure':20s}{'eye':>10s}{'mb19':>10s}{'triage max':>12s}"
        f"{'triage p99':>11s}{'triage med':>11s}  verdict")
    for k in MEASURES:
        tv = np.array([r[k] for r in triage], dtype=float)
        rv = {r.get("label", r["id"]): r[k] for r in refs}
        eye = rv.get("minibrot eye")
        mb19 = rv.get("mb19_p35")
        above_all = all(v > tv.max() for v in rv.values())
        n_above_eye = int((tv >= (eye or 0)).sum())
        n_above_mb19 = int((tv >= (mb19 or 0)).sum())
        out["measures"][k] = {
            "eye": eye, "mb19": mb19,
            "triage_max": float(tv.max()), "triage_p99": float(np.percentile(tv, 99)),
            "triage_median": float(np.median(tv)),
            "refs_above_all_triage": bool(above_all),
            "triage_atoms_at_or_above_eye": n_above_eye,
            "triage_atoms_at_or_above_mb19": n_above_mb19,
            "separates": bool(above_all),
        }
        log(f"  {k:20s}{(eye or 0):10.2f}{(mb19 or 0):10.2f}{tv.max():12.2f}"
            f"{np.percentile(tv, 99):11.2f}{np.median(tv):11.2f}  "
            f"{'PASS' if above_all else 'FAIL'}"
            f"  ({n_above_eye} triage >= eye, {n_above_mb19} >= mb19)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args(argv)
    if args.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2

    atoms = collect_atoms()
    if args.limit:
        atoms = atoms[:args.limit]
    print(f"measuring {len(atoms)} atoms at 4x, {fm.MEASURE_W}x{fm.MEASURE_H} ss{fm.MEASURE_SS} "
          f"({args.workers} procs x {THREADS} threads)")
    rows, errs = run_batch(atoms, workers=args.workers)
    print(f"  measured {len(rows)}, {len(errs)} failed")

    p = paths.durable(OUT, mkparents=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    print(f"-> {OUT}")

    if not args.skip_stability:
        print("\nmaxiter stability check")
        st = stability_check(atoms)
        paths.durable(STABILITY, mkparents=True).write_text(
            json.dumps(st, indent=2) + "\n", encoding="utf-8")

    v = validate(rows)
    paths.durable("data/orbital/validation.json", mkparents=True).write_text(
        json.dumps(v, indent=2) + "\n", encoding="utf-8")
    if errs:
        print(f"\n{len(errs)} failures, first few: {errs[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
