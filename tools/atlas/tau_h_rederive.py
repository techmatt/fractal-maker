#!/usr/bin/env python
r"""tau_h_rederive.py — re-derive the per-partition tau_h BASE under the ACTIVE head.

WHY THIS EXISTS. `steered_frontier.TAU_H_FIDELITY_BASE` is a vendored constant stamped
with the model version it was derived under (`TAU_H_FIDELITY_BASE_MODEL`). tau_h is a cut
on the CHEAP-render p_good of a SPECIFIC head, so after a checkpoint flip the vendored
value is a number about nothing and `derive_tau_h` refuses to start a run. This tool is
the re-derivation the failure message points at: it rebuilds the base from the committed
harvest logs under whatever `active_ckpt.ACTIVE_CKPT` currently is.

WHAT IT DOES. The original derivation (`descent_score_fidelity.py` -> the now-lost
`descent_score_fidelity_records.json`) needed PAIRED (cheap, canonical) scores from ONE
head. The harvest logs record every harvest check's geometry (`cx/cy/fw` + the julia seed
c) precisely so every check is re-renderable from the log alone, so the pair is
reconstructible for any head:

    harvest_log row -> Location -> render 384x216 ss1  -> score  => cheap_pgood(active)
                                -> render 640x360 ss2  -> score  => canon_pgood(active)

    tau_h[part] = quantile(cheap_pgood, 1-keep) over rows with canon_pgood >= t_good(part)

which is the fidelity study's estimator verbatim (keep=0.90 -> the 10th percentile of
cheap p_good among frames whose canonical p_good clears the family's t_good), with a
pooled cross-family fallback for partitions too thin to cut on their own.

THE ONE BIAS, STATED. The harvest log only holds checks that ALREADY CLEARED the LIVE
(v7-era) tau_h — it is a left-truncated sample of the candidate population. Cheap p_good
under two heads is correlated, so the surviving rows skew high on the active head's cheap
axis too, and the (1-keep) quantile computed on them is an UPPER bound on the untruncated
value. An upper bound is the aggressive direction (a too-high cut sheds admissions), so
the result is reported alongside a `truncation_floor` cross-check: the same estimator run
on the untruncated `prospect_run1` walk-outcome ledger (walk outcomes are a uniform-random
survivor per rung and are NOT tau-selected). `--combine min` takes the conservative
(lower) of the two per partition, which is the default, and the artifact records both so
the choice is auditable rather than baked in.

Kill-safe: every rendered+scored row is appended to `rows.jsonl` under the work dir and
re-used on a rerun, so a kill loses at most the in-flight batch.

  uv run python tools/atlas/tau_h_rederive.py --per-partition 200
  uv run python tools/atlas/tau_h_rederive.py --score-only        # re-derive from cached rows
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "mining", ROOT / "tools" / "scoring", ROOT / "tools" / "reframe"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import location as loc_mod                                   # noqa: E402
import paths                                                 # noqa: E402
from active_ckpt import (                                    # noqa: E402
    BIN, PALETTE, JPG_Q, auto_maxiter, make_scorer, ACTIVE_CKPT, ACTIVE_VERSION,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- render geometry: the two arms production actually uses --------------------------- #
CAN_W, CAN_H, CAN_SS = 640, 360, 2      # steered_frontier harvest confirmation render
CHEAP_W, CHEAP_H, CHEAP_SS = 384, 216, 1  # guided-descend --expand cheap node presentation

# Harvest logs that reconcile to their summaries at TODAY's t_good era
# (tau_h_retained_readout.RUNS), plus the campaign-2-matched julia parent probe, which is
# the only meaningful supply of julia:mandelbrot checks.
HARVEST_RUNS = [
    "campaign1/breadth", "campaign1/dive", "campaign2/breadth", "campaign2/dive",
    "julia_parent_probe/breadth",
]
# Untruncated cross-check population: walk OUTCOME frames. The walk picks a uniform-random
# gate survivor per rung and never scores, so this ledger is not selected on any tau.
WALK_LEDGER = ROOT / "data/discovery/fresh_runs/prospect_run1/outcome_ledger.jsonl"

# BULK, not scratch. `rows.jsonl` is one rendered+scored (cheap, canonical) pair per
# sampled row — hours of render + GPU scoring — and it is EXACTLY reproducible from the
# committed ledgers plus the active weights, which is the bulk class by definition
# (docs/design/storage_classes.md). It sat under `scratch/` and was consequently
# re-rendered from zero twice after a scratch wipe. `bulk()` resolves it out-of-tree via
# ARTIFACTS_ROOT (registered in artifacts.RELOCATED_PREFIXES), so it survives
# `rm -r scratch/*` and never costs the working tree a traversal.
WORK = paths.bulk("data/atlas/tau_h_rederive")
ARTIFACT = ROOT / "data" / "atlas" / f"tau_h_base_{ACTIVE_VERSION}.json"
WORKERS = 4            # concurrent render-one PROCESSES (project cap)


# --------------------------------------------------------------------------- #
# rows in, from the two populations
# --------------------------------------------------------------------------- #
def _harvest_rows():
    """Every harvest check across the current-era runs, tagged with its source run."""
    out = []
    for run in HARVEST_RUNS:
        p = ROOT / "data/discovery" / run / "harvest_log.jsonl"
        if not p.exists():
            print(f"  WARN missing harvest log: {p}")
            continue
        n, no_geom = 0, 0
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Geometry was added to the harvest row late (the "every reject fate is
            # renderable from the log alone" fix). Older rows carry no cx/cy/fw and are
            # simply not re-scoreable — counted, never guessed at.
            if r.get("cx") is None or r.get("fw") is None:
                no_geom += 1
                continue
            out.append(dict(
                pop="harvest", run=run, partition=r["partition"], depth=int(r["depth"]),
                cx=r["cx"], cy=r["cy"], fw=r["fw"],
                c=(None if r.get("julia_c_re") is None
                   else (str(r["julia_c_re"]), str(r["julia_c_im"]))),
                key=f"h_{run.replace('/', '_')}_{r['node_id']}_{r['batch']}",
            ))
            n += 1
        print(f"  {run}: {n} re-scoreable checks"
              + (f"  ({no_geom} pre-geometry rows skipped)" if no_geom else ""))
    return out


def _walk_rows():
    """Walk-outcome frames — the untruncated cross-check population."""
    if not WALK_LEDGER.exists():
        print(f"  WARN untruncated cross-check ledger absent: {WALK_LEDGER}")
        return []
    out = []
    for line in open(WALK_LEDGER, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fam = r.get("family")
        if not fam or fam == "phoenix":            # phoenix is not a frontier partition
            continue
        c = None
        if fam.startswith("julia:"):
            # walk-era julia rows carry the z-viewport separately; skip the ones that don't.
            if r.get("julia_z_cx") is None:
                continue
            c = (str(r["outcome_cx"]), str(r["outcome_cy"]))
            cx, cy, fw = r["julia_z_cx"], r["julia_z_cy"], r["julia_z_fw"]
        else:
            cx, cy, fw = r["outcome_cx"], r["outcome_cy"], r["outcome_fw"]
        out.append(dict(pop="walk", run="prospect_run1", partition=fam,
                        depth=int(r.get("reached_depth", 0)), cx=cx, cy=cy, fw=fw, c=c,
                        key=f"w_{r['id']}"))
    print(f"  prospect_run1 walk outcomes: {len(out)}")
    return out


def sample(rows, per_partition: int, seed: int):
    """Up to `per_partition` rows per (pop, partition), drawn without replacement."""
    by = defaultdict(list)
    for r in rows:
        by[(r["pop"], r["partition"])].append(r)
    rng = np.random.default_rng(seed)
    picked = []
    for k in sorted(by):
        pool = sorted(by[k], key=lambda r: r["key"])
        n = min(per_partition, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False) if n < len(pool) else range(len(pool))
        picked += [pool[int(i)] for i in sorted(idx)]
    return picked


# --------------------------------------------------------------------------- #
# render + score
# --------------------------------------------------------------------------- #
def render_family_of(partition: str) -> str:
    if partition.startswith("julia:"):
        base = partition.split(":", 1)[1]
        return "julia" if base == "mandelbrot" else "julia_" + base
    return partition


def _render(row, w, h, ss, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    c_re, c_im = (row["c"] if row["c"] else (None, None))
    loc = loc_mod.Location(family=render_family_of(row["partition"]), c_re=c_re, c_im=c_im,
                           cx=str(row["cx"]), cy=str(row["cy"]), fw=str(row["fw"]),
                           family_params={})
    import subprocess
    cmd = [str(BIN), "render-one", "--cx", str(row["cx"]), "--cy", str(row["cy"]),
           "--fw", repr(float(row["fw"])), "--width", str(w), "--height", str(h),
           "--supersample", str(ss), "--maxiter", str(auto_maxiter(float(row["fw"]))),
           "--palette", PALETTE, "--jpg-quality", str(JPG_Q), "--out", str(out)
           ] + loc_mod.render_one_flags(loc)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0 and out.exists()


def render_and_score(rows, scorer, tiles: Path, out_jsonl: Path, chunk: int = 48):
    """Render both arms and score them, chunk by chunk, appending to `out_jsonl`.

    Chunked so a kill loses at most `chunk` rows of work, and so the GPU scoring is
    batched rather than per-row."""
    done = set()
    if out_jsonl.exists():
        for line in open(out_jsonl, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["key"])
    todo = [r for r in rows if r["key"] not in done]
    print(f"[render] {len(done)} cached, {len(todo)} to do", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        blk = todo[i:i + chunk]
        jobs = []
        for r in blk:
            jobs.append((r, CHEAP_W, CHEAP_H, CHEAP_SS, tiles / f"{r['key']}_cheap.jpg"))
            jobs.append((r, CAN_W, CAN_H, CAN_SS, tiles / f"{r['key']}_canon.jpg"))
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(_render, *j): j for j in jobs if not j[4].exists()}
            for f in cf.as_completed(futs):
                f.result()
        ok = [r for r in blk if (tiles / f"{r['key']}_cheap.jpg").exists()
              and (tiles / f"{r['key']}_canon.jpg").exists()]
        if ok:
            # K-AWARE (`score_paths_k`, not `score_paths`). The K=3-shaped reader drops the
            # third cutpoint, so on the K=4 active head a stored row could not reproduce the
            # SERVED decode — `corn_decode(nb, pg, t_good, pg4)`, which is what the harvest
            # gate actually applies (steered_frontier.harvest). Rows written without p_ge4
            # are capped at class 3 by the reader, not by the head. `None` on a K=3 head.
            cheap = scorer.score_paths_k([str(tiles / f"{r['key']}_cheap.jpg") for r in ok])
            canon = scorer.score_paths_k([str(tiles / f"{r['key']}_canon.jpg") for r in ok])
            with open(out_jsonl, "a", encoding="utf-8") as fh:
                for r, ch, cn in zip(ok, cheap, canon):
                    fh.write(json.dumps(dict(
                        key=r["key"], pop=r["pop"], run=r["run"], partition=r["partition"],
                        depth=r["depth"], fw=float(r["fw"]),
                        cheap_eord=float(ch[0]), cheap_nb=float(ch[1]), cheap_pgood=float(ch[2]),
                        cheap_pge4=(float(ch[3]) if len(ch) > 3 else None),
                        canon_eord=float(cn[0]), canon_nb=float(cn[1]), canon_pgood=float(cn[2]),
                        canon_pge4=(float(cn[3]) if len(cn) > 3 else None),
                        model=ACTIVE_VERSION,
                    )) + "\n")
        for r in blk:                                   # tiles are bulk; drop as we go
            for arm in ("cheap", "canon"):
                p = tiles / f"{r['key']}_{arm}.jpg"
                if p.exists():
                    p.unlink()
        el = time.time() - t0
        n = min(i + chunk, len(todo))
        print(f"  {n}/{len(todo)} rows  {el:.0f}s  ({el/max(n,1):.2f}s/row, "
              f"eta {(len(todo)-n)*el/max(n,1)/60:.1f}m)", flush=True)


def assert_rows_current(rows, rows_jsonl):
    """Refuse a cache that is not what THIS head, at THIS row shape, would have written.

    Two ways a resumed `rows.jsonl` can be silently wrong, and neither is detectable from
    the derived number afterwards: rows scored under a different checkpoint, and rows
    written before the scorer went K-aware. The second is tested on KEY PRESENCE, not
    truthiness — `None` is the legitimate value for `p_ge4` on a K=3 head, so a `.get()`
    test would wave through exactly the rows it exists to catch."""
    stale = [r for r in rows if r.get("model") != ACTIVE_VERSION]
    if stale:
        raise SystemExit(f"{len(stale)} cached rows were scored under a different model than "
                         f"{ACTIVE_VERSION} — delete {rows_jsonl} and re-run")
    pre_k = [r for r in rows if "canon_pge4" not in r or "cheap_pge4" not in r]
    if pre_k:
        raise SystemExit(f"{len(pre_k)} cached rows were written by the pre-K-aware scorer "
                         f"(no p_ge4 column) — delete {rows_jsonl} and re-run")


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def derive(rows, partitions, keep, t_good_for, min_n=5, allow_pooled=False):
    """The fidelity-study estimator: per partition, the (1-keep) quantile of cheap p_good
    over rows whose canonical p_good clears that partition's t_good.

    NO POOLED FALLBACK by default (`allow_pooled=False`, changed at the v10 flip). A partition
    too thin to cut on its OWN population yields `None` — the arm is UNAVAILABLE for it, and
    the caller takes its per-partition minimum over the arms that actually produced a number.

    Why the original behaviour had to go. The pooled cut is a cross-family quantile: it is
    dominated by whichever partitions happen to have the most passing rows, and handing it to
    a thin partition is the same category error as serving a v8 threshold on a v10 gate — a
    number derived on a population that is not this one. It was harmless under v8 (every
    partition cut on its own on both arms) and stopped being harmless under v10: the native
    multibrot partitions run at the 0.50 UNCALIBRATED baseline, v10's canonical p_good sits
    lower than v8's, so walk-arm pass counts collapsed (multibrot3 12 -> 1, multibrot5 11 -> 3)
    and BOTH would have silently taken a pooled 0.039 in place of their own ~0.35 — a ~9x
    looser harvest gate, sourced from other families' frames.

    The pooled value is still COMPUTED and reported, because "what would the pooled cut have
    been" is useful context; it is simply never served."""
    q = 1.0 - keep
    by = defaultdict(list)
    for r in rows:
        by[r["partition"]].append(r)

    def cut(sel):
        vals = [r["cheap_pgood"] for r in sel]
        return (float(np.quantile(vals, q)), len(vals)) if len(vals) >= min_n else (None, len(vals))

    pooled_pass = [r for r in rows if r["canon_pgood"] >= t_good_for(r["partition"])]
    pooled, n_pooled = cut(pooled_pass)
    if pooled is None:
        pooled = 0.5
    out, detail = {}, {}
    for p in partitions:
        sel = [r for r in by.get(p, []) if r["canon_pgood"] >= t_good_for(p)]
        v, n = cut(sel)
        if v is None:
            out[p] = pooled if allow_pooled else None
            src = "pooled_fallback" if allow_pooled else f"UNAVAILABLE (<{min_n} own passing rows)"
        else:
            out[p] = v
            src = "own"
        detail[p] = dict(n_rows=len(by.get(p, [])), n_pass=n, t_good=t_good_for(p),
                         value=out[p], source=src,
                         pooled_would_have_been=(round(pooled, 6) if v is None else None))
    return out, detail, dict(pooled=pooled, n_pooled=n_pooled)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-partition", type=int, default=200,
                    help="max rows sampled per (population, partition)")
    ap.add_argument("--keep", type=float, default=0.90,
                    help="fraction of canonical-passing frames the cut retains (default 0.90)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combine", choices=["min", "harvest", "walk"], default="min",
                    help="how to combine the truncated harvest estimate with the untruncated "
                         "walk cross-check (default min = the conservative one)")
    ap.add_argument("--score-only", action="store_true",
                    help="skip rendering; re-derive from the cached rows.jsonl")
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    args = ap.parse_args()

    import production_seeder as ps                        # noqa: E402 (heavy)
    partitions = list(ps.T_GOOD_OVERRIDES) + [
        "mandelbrot", "multibrot3", "multibrot4", "multibrot5",
        "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"]
    partitions = sorted(set(p for p in partitions if p != "phoenix"))

    args.work.mkdir(parents=True, exist_ok=True)
    rows_jsonl = args.work / "rows.jsonl"
    tiles = args.work / "tiles"

    if not args.score_only:
        print(f"[pop] harvest logs ({ACTIVE_VERSION} re-score):")
        pool = _harvest_rows()
        print("[pop] untruncated cross-check:")
        pool += _walk_rows()
        picked = sample(pool, args.per_partition, args.seed)
        cnt = defaultdict(int)
        for r in picked:
            cnt[(r["pop"], r["partition"])] += 1
        print(f"[sample] {len(picked)} rows: {dict(sorted(cnt.items()))}", flush=True)
        scorer = make_scorer(ACTIVE_CKPT)
        print(f"[scorer] {ACTIVE_CKPT} ({ACTIVE_VERSION})", flush=True)
        render_and_score(picked, scorer, tiles, rows_jsonl)

    rows = [json.loads(l) for l in open(rows_jsonl, encoding="utf-8") if l.strip()]
    assert_rows_current(rows, rows_jsonl)
    h_rows = [r for r in rows if r["pop"] == "harvest"]
    w_rows = [r for r in rows if r["pop"] == "walk"]

    h_val, h_det, h_pool = derive(h_rows, partitions, args.keep, ps.t_good_for)
    w_val, w_det, w_pool = derive(w_rows, partitions, args.keep, ps.t_good_for) if w_rows \
        else ({}, {}, {})

    # `combine=min` over the arms that ACTUALLY produced a number on this partition's own
    # population. An arm that came back UNAVAILABLE (too few own passing rows) contributes
    # nothing — it is not replaced by a pooled cross-family cut, and it does not silently
    # become the other arm's value under a `walk`/`harvest` selection either.
    final, arms_used = {}, {}
    for p in partitions:
        cands = {k: v for k, v in (("harvest", h_val.get(p)), ("walk", w_val.get(p)))
                 if v is not None}
        if not cands:
            raise SystemExit(
                f"tau_h: partition {p!r} has NO arm cut on its own population "
                f"(harvest n_pass={h_det[p]['n_pass']}, walk "
                f"n_pass={(w_det.get(p) or {}).get('n_pass', 0)}). Raise --per-partition or "
                f"accept that {p} cannot be cut under {ACTIVE_VERSION}; a pooled cross-family "
                f"quantile is NOT a substitute.")
        if args.combine == "min":
            pick = min(cands, key=cands.get)
        elif args.combine in cands:
            pick = args.combine
        else:                       # requested arm unavailable -> fall to the one that exists
            pick = next(iter(cands))
        final[p], arms_used[p] = cands[pick], {"picked": pick, "available": sorted(cands)}

    art = dict(
        model=ACTIVE_VERSION, ckpt=ACTIVE_CKPT, keep=args.keep, seed=args.seed,
        combine=args.combine, per_partition=args.per_partition,
        n_rows_harvest=len(h_rows), n_rows_walk=len(w_rows),
        harvest_runs=HARVEST_RUNS, walk_ledger=str(WALK_LEDGER),
        t_good={p: ps.t_good_for(p) for p in partitions},
        t_good_status={p: ps.t_good_status(p) for p in partitions},
        tau_h_base=final, harvest_estimate=h_val, walk_estimate=w_val,
        harvest_detail=h_det, walk_detail=w_det, arms_used=arms_used,
        pooled_harvest=h_pool, pooled_walk=w_pool, pooled_fallback_allowed=False,
        caveat=("The harvest population is LEFT-TRUNCATED at the previous head's tau_h, so "
                "its quantile is an UPPER bound on the untruncated value; the walk-outcome "
                "population is untruncated but off-distribution (walk outcomes, not frontier "
                "candidates). combine=min takes the conservative side."),
        no_pooling=("EVERY partition is cut on its OWN population. An arm with fewer than "
                    "min_n=5 own passing rows is UNAVAILABLE and contributes nothing; the "
                    "pooled cross-family quantile is computed and reported but NEVER served. "
                    "`arms_used` names, per partition, which arms existed and which was "
                    "taken — read it before comparing a partition against a prior version."),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(art, indent=2), encoding="utf-8")

    print(f"\n=== tau_h base under {ACTIVE_VERSION} (keep={args.keep}, combine={args.combine}) ===")
    print(f"{'partition':22s} {'harvest':>9s} {'walk':>9s} {'FINAL':>9s}  t_good  n_pass(h/w)")
    def _f(v):
        return f"{v:9.4f}" if v is not None else f"{'n/a':>9s}"

    for p in partitions:
        # Print BOTH arms' availability. The old line printed only the harvest arm's source,
        # so a walk arm that had silently fallen back to the pooled cut still read "[own]".
        print(f"{p:22s} {_f(h_val.get(p))} {_f(w_val.get(p))} "
              f"{final[p]:9.4f}  {ps.t_good_for(p):.2f}   "
              f"{h_det[p]['n_pass']}/{(w_det.get(p, {}) or {}).get('n_pass', 0)}"
              f"  [{'+'.join(arms_used[p]['available'])} -> {arms_used[p]['picked']}]")
    unavail = {p: d for p, d in list(h_det.items()) + list(w_det.items())
               if d["source"].startswith("UNAVAILABLE")}
    if unavail:
        print("\n  arms UNAVAILABLE (cut on own population impossible; NOT pooled):")
        for p in partitions:
            for nm, det in (("harvest", h_det.get(p)), ("walk", w_det.get(p))):
                if det and det["source"].startswith("UNAVAILABLE"):
                    print(f"    {p:20s} {nm:8s} n_pass={det['n_pass']} "
                          f"(pooled would have been {det['pooled_would_have_been']})")
    print(f"\nartifact -> {args.out}")
    print("\nPaste into tools/atlas/steered_frontier.py (BOTH lines together):")
    print(f'TAU_H_FIDELITY_BASE_MODEL = "{ACTIVE_VERSION}"')
    print("TAU_H_FIDELITY_BASE = {")
    for p in partitions:
        print(f'    "{p}": {final[p]!r},')
    print("}")


if __name__ == "__main__":
    main()
