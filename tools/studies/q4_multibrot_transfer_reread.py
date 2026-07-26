#!/usr/bin/env python
"""q4 multibrot-transfer RE-READ — strip the two confounds from the cross-degree
comparison, post-hoc, from data already on disk.

No re-source, no re-render. The sourced nuclei (`nuclei_d*.json`, with period/|A|/degree
per window) and their f64 fields (`fields/d*/*.bin`) are on disk from the transfer run.
This tool only (a) collapses rotational copies with the symmetry-canonical dedup key
(`dcf.nucleus_dedup_key`), (b) re-screens the CACHED fields per-minibrot — the identical
deployment model and `MT.screen_field`, just per-window bookkeeping the original aggregate
didn't persist (it stored only per-degree pooled G-percentiles + per-minibrot G_max) — and
(c) re-aggregates three ways: raw (reproduces `stats.json`), collapsed, and collapsed +
period-conditioned. It answers: which cross-degree claims survive removing pseudo-replication
(unequal rotational inflation) and period mismatch (a fixed size band admitting systematically
lower periods as degree rises).

Run:  uv run python -m tools.studies.q4_multibrot_transfer_reread
Reads:  scratch/q4_multibrot_transfer/{nuclei_d*.json, fields/d*/*.bin}
Record: docs/design/q4_multibrot_transfer.md  (updated in place)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mpmath as mp  # noqa: E402
import tools.sourcing.deep_center_finder as dcf  # noqa: E402
from tools.studies import q4_multibrot_transfer as MT  # noqa: E402

# Period band populated by EVERY degree (so conditioning on it is apples-to-apples).
# d4 tops out at p10 and d5 is p≤6 plus a p15 cluster, so the common floor is p3..p6.
COND_PERIODS = set(range(3, 7))       # {3,4,5,6}
CACHE = MT.OUT / "reread_screen.json"  # per-minibrot re-screen detail (cache; ~mins to build)
WORKERS = 4                            # repo cap on parallel workers


def _canon_groups(recs, degree):
    """Group `recs` by symmetry-canonical dedup key; return ordered list of groups
    (each a list of records that are rotational copies of one atom)."""
    groups = {}
    with mp.workdps(MT.NUCLEUS_DPS):          # parse cx/cy strings at full precision
        for r in recs:
            key = dcf.nucleus_dedup_key(mp.mpc(r["cx"], r["cy"]), degree, MT.DEDUP_DPS)
            groups.setdefault(key, []).append(r)
    return list(groups.values())


# --- parallel per-minibrot re-screen over CACHED fields (4 workers) -------------- #
_MODEL = None
_CUTOFF = None


def _init_worker(model, cutoff):
    global _MODEL, _CUTOFF
    _MODEL, _CUTOFF = model, cutoff


def _screen_one(task):
    """Screen one cached field. `assert_once=False`: the dense_grid faithfulness of the
    instrumentation is already established by the committed transfer run's own assert;
    re-paying it (a 2nd full featurize pass) here buys nothing."""
    deg, r, fdir_str = task
    field, fw, fh = MT._load_field(Path(fdir_str) / f"{r['id']}.bin")
    res = MT.screen_field(field, fw, fh, _MODEL, _CUTOFF, assert_once=False)
    a = res["agg"]
    return dict(
        id=r["id"], period=r["period"], degree=deg,
        cx=r["cx"], cy=r["cy"], log10_abs_A=r.get("log10_abs_A"),
        n_masked=a["n_masked"], n_surv=a["n_surv"],
        clause={cl: int(a["masked_clause"].get(cl, 0))
                for cl in ("interior", "flat", "speckle")},
        G=res["G"].tolist(),
        n_accepted=sum(1 for c in res["kept"] if c["G"] >= _CUTOFF))


def _screen_all(model, cutoff):
    """Re-screen every sourced field per minibrot (cached .bin; no re-render), 4 workers.
    Returns {degree: [per-minibrot dict]}; G stored as a list (JSON-cacheable)."""
    tasks = []
    for deg in MT.DEGREES:
        recs = json.loads((MT.OUT / f"nuclei_d{deg}.json").read_text())
        fdir = MT.FIELDS / f"d{deg}"
        for r in recs:
            if (fdir / f"{r['id']}.bin").exists():
                tasks.append((deg, r, str(fdir)))
            else:
                print(f"  WARN missing field {r['id']}.bin; skipping", flush=True)
    out = {d: [] for d in MT.DEGREES}
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init_worker, initargs=(model, cutoff)) as ex:
        for rec in ex.map(_screen_one, tasks):
            out[rec["degree"]].append(rec)
            done += 1
            print(f"  screened {done}/{len(tasks)}  ({rec['id']})", flush=True)
    for d in out:
        out[d].sort(key=lambda m: m["id"])
    return out


def _aggregate(per_mb):
    """Pool a list of per-minibrot dicts into degree-level stats (position-weighted)."""
    n_masked = sum(m["n_masked"] for m in per_mb)
    n_surv = sum(m["n_surv"] for m in per_mb)
    n_feat = n_masked + n_surv
    clause = {cl: sum(m["clause"][cl] for m in per_mb)
              for cl in ("interior", "flat", "speckle")}
    G = (np.concatenate([np.asarray(m["G"], float) for m in per_mb])
         if per_mb else np.array([]))
    return dict(
        n_mb=len(per_mb), n_feat=n_feat, n_surv=n_surv,
        ood_reject=(n_masked / n_feat if n_feat else float("nan")),
        clause_frac={cl: (clause[cl] / n_feat if n_feat else float("nan"))
                     for cl in clause},
        G_median=(float(np.median(G)) if len(G) else float("nan")),
        G_p90=(float(np.percentile(G, 90)) if len(G) else float("nan")),
        G_max=(float(G.max()) if len(G) else float("nan")),
        G_min=(float(G.min()) if len(G) else float("nan")),
        frac_G_pos=(float((G > 0).mean()) if len(G) else float("nan")),
        n_accepted=sum(m["n_accepted"] for m in per_mb))


def _fmt(tag, agg):
    cf = agg["clause_frac"]
    return (f"  {tag:<26} n_mb={agg['n_mb']:>2}  feat={agg['n_feat']:>6}  "
            f"reject={agg['ood_reject']:6.1%}  Gmed={agg['G_median']:+6.2f}  "
            f"P90={agg['G_p90']:+5.2f}  max={agg['G_max']:+5.2f}  "
            f"G>0={agg['frac_G_pos']:5.1%}  acc={agg['n_accepted']:>2}  |  "
            f"int={cf['interior']:5.1%} flat={cf['flat']:5.1%} "
            f"spk={cf['speckle']:5.1%}")


def _load_or_screen():
    """Per-minibrot screen detail, from cache if present else a fresh 4-worker pass."""
    if CACHE.exists():
        print(f"loading cached re-screen: {CACHE.relative_to(ROOT)}\n", flush=True)
        raw = json.loads(CACHE.read_text())
        return {int(k): v for k, v in raw.items() if k.isdigit()}, raw["_cutoff"][0]
    print("re-screening cached fields (4 workers; no re-source / no re-render) ...",
          flush=True)
    model, tight = MT._fit_model()
    cutoff = tight["cutoff"]
    screened = _screen_all(model, cutoff)
    payload = {str(d): screened[d] for d in screened}
    payload["_cutoff"] = [cutoff]
    CACHE.write_text(json.dumps(payload))
    print(f"-> cached {CACHE.relative_to(ROOT)}\n", flush=True)
    return screened, cutoff


def main():
    print("re-read: collapsing rotational copies + conditioning on period\n"
          "(no re-source / no re-render; per-minibrot re-screen over cached fields)\n")
    screened, cutoff = _load_or_screen()

    # --- collapse map + period distributions (from the nuclei metadata) --------- #
    collapsed_ids, dropped = {}, {}
    for deg in MT.DEGREES:
        recs = json.loads((MT.OUT / f"nuclei_d{deg}.json").read_text())
        groups = _canon_groups(recs, deg)
        keep = {g[0]["id"] for g in groups}              # one representative per atom
        collapsed_ids[deg] = keep
        dropped[deg] = [x["id"] for g in groups if len(g) > 1 for x in g[1:]]

    print("=" * 108)
    print("EFFECTIVE n AND PERIOD DISTRIBUTION PER DEGREE")
    print("=" * 108)
    for deg in MT.DEGREES:
        mbs = screened[deg]
        keep = collapsed_ids[deg]
        p_all = Counter(m["period"] for m in mbs)
        p_col = Counter(m["period"] for m in mbs if m["id"] in keep)
        p_cond = Counter(m["period"] for m in mbs
                         if m["id"] in keep and m["period"] in COND_PERIODS)
        print(f"d{deg}: raw n={len(mbs)}  ->  collapsed n={len(keep)}"
              + (f"  (dropped {dropped[deg]})" if dropped[deg] else "  (no rotational copies)"))
        print(f"      periods raw       : {dict(sorted(p_all.items()))}")
        print(f"      periods collapsed : {dict(sorted(p_col.items()))}")
        print(f"      periods p3-6 only : {dict(sorted(p_cond.items()))}  "
              f"(n={sum(p_cond.values())})")

    # --- three aggregations per degree ------------------------------------------ #
    print("\n" + "=" * 108)
    print("RE-AGGREGATED STATS  (RAW reproduces stats.json; then confounds removed)")
    print("=" * 108)
    for deg in MT.DEGREES:
        mbs = screened[deg]
        keep = collapsed_ids[deg]
        raw = _aggregate(mbs)
        col = _aggregate([m for m in mbs if m["id"] in keep])
        cond = [m for m in mbs if m["id"] in keep and m["period"] in COND_PERIODS]
        cnd = _aggregate(cond)
        print(f"\nd{deg}:")
        print(_fmt("raw (all sourced)", raw))
        print(_fmt("collapsed", col))
        note = "" if cnd["n_mb"] >= 3 else "   <-- TOO FEW distinct sources; do not compare"
        print(_fmt("collapsed + period 3-6", cnd) + note)

    # --- headline trend tables (collapsed vs collapsed+conditioned) ------------- #
    def trend(select):
        rows = {}
        for deg in MT.DEGREES:
            mbs = [m for m in screened[deg] if select(deg, m)]
            rows[deg] = _aggregate(mbs)
        return rows
    col = trend(lambda deg, m: m["id"] in collapsed_ids[deg])
    cnd = trend(lambda deg, m: m["id"] in collapsed_ids[deg] and m["period"] in COND_PERIODS)

    print("\n" + "=" * 108)
    print("DOES THE DRIFT SURVIVE?  (d2 -> d3 -> d4 -> d5)")
    print("=" * 108)
    def line(name, key, fn=lambda v: f"{v:+.2f}"):
        c = "  ".join(fn(col[d][key]) for d in MT.DEGREES)
        k = "  ".join(fn(cnd[d][key]) for d in MT.DEGREES)
        print(f"  {name:<22} collapsed: {c:<34} | +period3-6: {k}")
    line("OOD reject", "ood_reject", lambda v: f"{v:.1%}")
    line("G median", "G_median")
    line("G P90 (pos tail)", "G_p90")
    line("G max", "G_max")
    # clause fractions are nested — print explicitly
    for cl in ("interior", "flat", "speckle"):
        c = "  ".join(f"{col[d]['clause_frac'][cl]:.1%}" for d in MT.DEGREES)
        k = "  ".join(f"{cnd[d]['clause_frac'][cl]:.1%}" for d in MT.DEGREES)
        print(f"  {cl+' clause':<22} collapsed: {c:<34} | +period3-6: {k}")
    ea = "  ".join(str(cnd[d]["n_mb"]) for d in MT.DEGREES)
    print(f"  effective n (p3-6)     {ea}")


if __name__ == "__main__":
    main()
