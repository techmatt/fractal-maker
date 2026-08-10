#!/usr/bin/env python
r"""tau_h_rederive.py — re-derive the per-partition tau_h BASE under the ACTIVE head.

WHY THIS EXISTS. `steered_frontier.TAU_H_FIDELITY_BASE` is a vendored constant stamped
with the model version it was derived under (`TAU_H_FIDELITY_BASE_MODEL`). tau_h is a cut
on the CHEAP-render p_good of a SPECIFIC head, so after a checkpoint flip the vendored
value is a number about nothing and `derive_tau_h` refuses to start a run. This tool is
the re-derivation the failure message points at.

WHAT IT DOES.

    walk-outcome row -> Location -> render 384x216 ss1  -> score  => cheap_pgood(active)
                                 -> render 640x360 ss2  -> score  => canon_pgood(active)

    tau_h[part] = quantile(cheap_pgood, 1-keep) over rows with canon_pgood >= GOOD_FLOOR

i.e. the cheap cut that RETAINS ~`keep` of the frames a canonical render would have kept.

TWO THINGS CHANGED ON 2026-08-09 (prompts/selection_restructure_3.md) AND BOTH ARE
SIMPLIFICATIONS OF THE SAME KIND — one definition of "good", one population.

  1. "GOOD OUTCOME" IS `canon_pgood >= floors.GOOD_FLOOR`, not `canon_pgood >=
     t_good_for(partition)`. The per-partition t_good table is gone: the run side admits on
     the fixed floor now, so conditioning this estimator on anything else would derive a
     harvest cut for a gate that does not exist. It also removes the confound that made the
     v10 -> v11 table unreadable — five partitions moved on population alone and three moved
     on population AND a t_good change, and no pair of artifacts could separate them.

  2. THE HARVEST ARM IS GONE, and with it the two-arm minimum, the per-run truncation record
     and the harvest-log registry. The harvest log only ever held checks that had already
     cleared a PREVIOUS head's tau_h, at a level that differed per run — a left-truncated
     sample at a MIXTURE of levels, whose quantile is an upper bound of unknown tightness.
     Every derivation had to carry a paragraph explaining why its own largest number was the
     one to distrust (v11's multibrot4 0.8245 rested on that arm alone). The walk-outcome
     ledger is a uniform-random gate survivor per rung, never tau-selected, so it is the
     only untruncated population in the tree and it is now the only one used. It is smaller
     — hundreds of rows per partition, not thousands — and a smaller unbiased sample is a
     better estimator than a larger one with an unquantifiable bias.

A PARTITION WITH FEWER THAN `MIN_N` GOOD ROWS GETS tau_h = 0.0 — harvest everything. Fail
OPEN, deliberately: tau_h decides who pays for a canonical confirmation render, so a
too-high cut sheds supply invisibly while a too-low one shows up as GPU-minutes in the run's
own telemetry. There is no pooled cross-family fallback and there never was a defensible
one — a cut derived on other families' frames is a number about a different population.

Kill-safe: every rendered+scored row is appended to `rows.jsonl` under the work dir and
re-used on a rerun, so a kill loses at most the in-flight batch.

  uv run python tools/atlas/tau_h_rederive.py
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

import corpus_common as cc                                   # noqa: E402  (engine launch defaults)
import location as loc_mod                                   # noqa: E402
import paths                                                 # noqa: E402
from active_ckpt import (                                    # noqa: E402
    BIN, PALETTE, JPG_Q, auto_maxiter, make_scorer, ACTIVE_CKPT, ACTIVE_VERSION,
)
from tools.emission import floors as F                       # noqa: E402  THE cut owner

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- render geometry: the two arms production actually uses --------------------------- #
CAN_W, CAN_H, CAN_SS = 640, 360, 2      # steered_frontier harvest confirmation render
CHEAP_W, CHEAP_H, CHEAP_SS = 384, 216, 1  # guided-descend --expand cheap node presentation

# THE population: walk OUTCOME frames. The walk picks a uniform-random gate survivor per rung
# and never scores, so this ledger is not selected on any tau — the one untruncated sample of
# the candidate stream in the tree.
WALK_LEDGER = ROOT / "data/discovery/fresh_runs/prospect_run1/outcome_ledger.jsonl"

# Below this many GOOD rows a partition is not cut at all and harvests everything (tau_h=0.0).
# 5 is the smallest n at which a 10th percentile is a statement rather than a restatement of
# the minimum, and the consequence of being under it is spending render time, not losing
# supply — see the fail-open paragraph in the module docstring.
MIN_N = 5

# BULK, not scratch. `rows.jsonl` is one rendered+scored (cheap, canonical) pair per
# sampled row — hours of render + GPU scoring — and it is EXACTLY reproducible from the
# committed ledgers plus the active weights, which is the bulk class by definition
# (docs/design/storage_classes.md). It sat under `scratch/` and was consequently
# re-rendered from zero twice after a scratch wipe. `bulk()` resolves it out-of-tree via
# ARTIFACTS_ROOT (registered in artifacts.RELOCATED_PREFIXES), so it survives
# `rm -r scratch/*` and never costs the working tree a traversal.
#
# It may still hold `pop == "harvest"` rows from a pre-2026-08-09 derivation. They are read
# past, never deleted: re-rendering the walk arm costs an hour and the file is the only
# record of what those renders scored.
WORK = paths.bulk("data/atlas/tau_h_rederive")
ARTIFACT = ROOT / "data" / "atlas" / f"tau_h_base_{ACTIVE_VERSION}.json"
WORKERS = 4            # concurrent render-one PROCESSES (project cap)
# Threads PER engine process. `DEFAULT_ENGINE_THREADS` (7) is the ONE-process default and
# WORKERS=4 of them would oversubscribe this 12-core box 28-to-12; 3 x 4 = 12 fits it exactly.
# Passed explicitly for the reason the convention states: a multi-process fan-out has no
# standing number and must not inherit the single-process one.
RENDER_THREADS = 3


# --------------------------------------------------------------------------- #
# rows in
# --------------------------------------------------------------------------- #
def walk_rows():
    """Walk-outcome frames — THE population."""
    if not WALK_LEDGER.exists():
        raise SystemExit(
            f"tau_h derivation: the walk-outcome ledger is absent ({WALK_LEDGER}). It is the "
            f"only untruncated population in the tree and there is no substitute — a harvest "
            f"log is truncated at whatever tau its own run was serving.")
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
    """Up to `per_partition` rows per partition, drawn without replacement. `0` = uncapped."""
    if not per_partition:
        return list(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["partition"]].append(r)
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
    # Through the committed helpers, not a bare `subprocess.run` (CLAUDE.md § Runtime
    # discipline). This launcher fans out WORKERS=4 engine processes, so the per-process 7 is
    # NOT the number to inherit — `DEFAULT_ENGINE_THREADS` is the one-process default and
    # "multiple parallel engine processes has no standing number", so threads are sized for
    # the actual N against the box's 12 logical cores and passed explicitly. Missing the
    # BELOW_NORMAL flag is what made the adoption run contend with the desktop for 26 min.
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env=cc.default_engine_env(threads=RENDER_THREADS),
                       creationflags=cc.default_creationflags())
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
            # K-AWARE (`score_paths_k`, not `score_paths`). The cut itself reads only P(>=3),
            # but the third cutpoint is what a K=4 head produces and dropping it at write time
            # makes the cached row unable to answer a later question about class 4. `None` on
            # a K=3 head.
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


# The fields that make one derivation a DIFFERENT STATEMENT from another, rather than a
# re-run of the same one. Population size and the estimator's settings; NOT the derived
# values, which are exactly what a legitimate re-run is allowed to move.
SUPERSEDE_KEYS = ("per_partition", "n_rows", "keep", "seed", "good_floor", "min_n")


def assert_not_superseding(out: Path, art: dict, overwrite: bool) -> None:
    """Refuse to overwrite an artifact that records a derivation over a DIFFERENT population.

    A tau_h artifact is the RECORD of a derivation over a stated population, and the file name
    carries only the model version — so a re-derivation at a different size silently destroys
    the record of what production was actually served. It nearly did: the 2026-08-08
    enlargement re-derives v11 over 64,365 rows where the adoption-era v11 artifact was 3,492,
    and both want to be `tau_h_base_v11.json`. The remedy is to ARCHIVE the old one under a
    distinguishing name (`tau_h_base_v11_adoption.json`) and pass `--overwrite`, so replacing a
    record is a deliberate act that leaves the superseded one readable.

    Re-running the SAME derivation is not superseding and passes untouched — the guard compares
    `SUPERSEDE_KEYS`, never the derived values, or every legitimate re-run would trip it."""
    if not out.exists() or overwrite:
        return
    old = json.loads(out.read_text(encoding="utf-8"))
    differs = {k: (old.get(k), art[k]) for k in SUPERSEDE_KEYS if old.get(k) != art[k]}
    if differs:
        raise SystemExit(
            f"{out} already records a derivation over a DIFFERENT population/settings "
            f"({', '.join(f'{k}: {a} -> {b}' for k, (a, b) in differs.items())}). It is the "
            f"record of what production was served; overwriting it loses that. Archive it "
            f"under a distinguishing name and re-run with --overwrite, or write this "
            f"derivation elsewhere with --out.")


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def derive(rows, partitions, keep, *, good_floor=None, min_n=MIN_N):
    """`(tau_h, detail)` — per partition, the (1-keep) quantile of cheap p_good over the rows
    whose CANONICAL p_good clears `good_floor` (default `floors.GOOD_FLOOR`).

    A partition with fewer than `min_n` good rows gets 0.0 and harvests everything. There is
    no pooled cross-family fallback: a cut derived on other families' frames is a number about
    a population that is not this one, and handing it over is the same category error as
    serving a v8 threshold on a v10 gate. Fail OPEN rather than fail-to-a-neighbour."""
    good_floor = F.GOOD_FLOOR if good_floor is None else float(good_floor)
    q = 1.0 - keep
    by = defaultdict(list)
    for r in rows:
        by[r["partition"]].append(r)
    out, detail = {}, {}
    for p in partitions:
        pool = by.get(p, [])
        sel = [r["cheap_pgood"] for r in pool if r["canon_pgood"] >= good_floor]
        if len(sel) >= min_n:
            out[p] = float(np.quantile(sel, q))
            src = "own"
        else:
            out[p] = 0.0
            src = f"FAIL-OPEN (<{min_n} good rows) — harvests everything"
        detail[p] = dict(n_rows=len(pool), n_good=len(sel), value=out[p], source=src)
    return out, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-partition", type=int, default=0,
                    help="max rows sampled per partition (0 = uncapped, the default: the walk "
                         "ledger is small enough to use whole)")
    ap.add_argument("--keep", type=float, default=0.90,
                    help="fraction of canonical-good frames the cut retains (default 0.90)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-only", action="store_true",
                    help="skip rendering; re-derive from the cached rows.jsonl")
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing artifact that recorded a DIFFERENT population "
                         "or settings (archive the old one first — it is the record of what "
                         "production was served)")
    args = ap.parse_args()

    partitions = sorted({
        "mandelbrot", "multibrot3", "multibrot4", "multibrot5",
        "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"})

    args.work.mkdir(parents=True, exist_ok=True)
    rows_jsonl = args.work / "rows.jsonl"
    tiles = args.work / "tiles"

    if not args.score_only:
        print("[pop] untruncated walk-outcome ledger:")
        picked = sample(walk_rows(), args.per_partition, args.seed)
        cnt = defaultdict(int)
        for r in picked:
            cnt[r["partition"]] += 1
        print(f"[sample] {len(picked)} rows: {dict(sorted(cnt.items()))}", flush=True)
        scorer = make_scorer(ACTIVE_CKPT)
        print(f"[scorer] {ACTIVE_CKPT} ({ACTIVE_VERSION})", flush=True)
        render_and_score(picked, scorer, tiles, rows_jsonl)

    cached = [json.loads(l) for l in open(rows_jsonl, encoding="utf-8") if l.strip()]
    assert_rows_current(cached, rows_jsonl)
    # A pre-2026-08-09 cache carries the retired harvest arm's rows too. Read past them.
    rows = [r for r in cached if r["pop"] == "walk"]
    n_ignored = len(cached) - len(rows)
    if n_ignored:
        print(f"[cache] {len(rows)} walk rows; {n_ignored} retired harvest-arm rows ignored")

    tau, detail = derive(rows, partitions, args.keep)

    art = dict(
        model=ACTIVE_VERSION, ckpt=ACTIVE_CKPT, keep=args.keep, seed=args.seed,
        per_partition=args.per_partition, n_rows=len(rows),
        good_floor=F.GOOD_FLOOR, min_n=MIN_N,
        walk_ledger=str(WALK_LEDGER.relative_to(ROOT).as_posix()),
        tau_h_base=tau, detail=detail,
        population=("The prospect_run1 walk-outcome ledger, whole. Walk outcomes are a "
                    "uniform-random gate survivor per rung and are never tau-selected, so "
                    "this is the one UNTRUNCATED population available; the harvest arm and "
                    "its per-run truncation mixture retired on 2026-08-09. It is "
                    "off-distribution in the other direction — walk outcomes, not frontier "
                    "candidates — and that is stated rather than corrected for."),
        definition=("good outcome = canonical p_good >= floors.GOOD_FLOOR (the run side's "
                    "own admission cut, not a per-partition t_good). tau_h = the (1-keep) "
                    "quantile of CHEAP p_good among those frames."),
        fail_open=(f"A partition with fewer than min_n={MIN_N} good rows gets tau_h = 0.0 and "
                   f"harvests everything. No pooled cross-family fallback: fail OPEN, which "
                   f"costs visible GPU-minutes in run telemetry, never invisible supply."),
    )
    assert_not_superseding(args.out, art, args.overwrite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(art, indent=2), encoding="utf-8")

    print(f"\n=== tau_h base under {ACTIVE_VERSION} "
          f"(keep={args.keep}, good_floor={F.GOOD_FLOOR:g}) ===")
    print(f"{'partition':22s} {'n_rows':>7s} {'n_good':>7s} {'tau_h':>9s}  source")
    for p in partitions:
        d = detail[p]
        print(f"{p:22s} {d['n_rows']:7d} {d['n_good']:7d} {tau[p]:9.4f}  {d['source']}")
    print(f"\nartifact -> {args.out}")
    print("\nPaste into tools/atlas/steered_frontier.py (BOTH lines together):")
    print(f'TAU_H_FIDELITY_BASE_MODEL = "{ACTIVE_VERSION}"')
    print("TAU_H_FIDELITY_BASE = {")
    for p in partitions:
        print(f'    "{p}": {tau[p]!r},')
    print("}")


if __name__ == "__main__":
    main()
