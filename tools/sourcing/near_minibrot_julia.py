#!/usr/bin/env python
r"""near_minibrot_julia.py — julia `c`'s sampled on a distance ladder around known nuclei.

THE HYPOTHESIS, STATED SO THE ANSWER IS READABLE. `julia_c_sourcing.md` settles that the
exemplar-grade Julia class — distributed multi-scale filament detail plus composed interior
lakes — sits at ~2% of viable near-∂M `c`, found by a boundary-rejection sampler that knows
nothing about WHERE on the boundary it is. A minibrot nucleus is a very particular place on
that boundary: it is the centre of an atom whose size is known exactly (`1/|A|`, the `A`
instrument). So the question is whether "near ∂M" can be sharpened to "near ∂M **at a known
multiple of a known atom's radius**", and if so at which multiple.

THE LADDER IS THE EXPERIMENT. Each sampled nucleus contributes one `c` per rung at
`r in {1, 4, 16}` atom radii, and **the rung rides every candidate row**. That is the whole
design: a leg that only sampled "near a minibrot" would answer "is this better than the
boundary sampler?" and could not say *at what distance*, which is the part that would
generalise. `1/|A|` and not the naive `lambda^2` law — at `d >= 3` that under-sizes the atom
by 4-2497x (`atom_instrument.md`), and here it would silently move every rung.

NO FRESH ENUMERATION. Supply is the existing roster (`data/minibrot_roster/roster.jsonl`)
plus the atoms the label-seeded harvest already solved — ~517 degree-2 nuclei between them,
which is far more than this leg's budget. Enumeration is ~25x screening
(`minibrot_maneuvers.md` §3), and paying it again for atoms already on disk would spend the
leg's entire budget on supply it already has.

DEGREE 2 ONLY, and it is a definition rather than a restriction: a `julia:mandelbrot` `c` is
a point of the degree-2 parameter plane, so a degree-5 atom's nucleus is not a `c` for this
family at all.

WIDE FRAMES, NOT MID-ZOOMS. `julia_c_sourcing.md`: the class favours whole-Julia framings
(`fw ~ 1.0-1.4`) and "do not mid-zoom-crop the search". The frame is drawn in that band and
recorded per row.

WHAT THIS IS NOT: not an eval, and not a base rate. The nuclei are the ones two prior
searches already surfaced, so the population is conditioned on where those searches went.

  uv run python tools/sourcing/near_minibrot_julia.py sample --n-nuclei 120
  uv run python tools/sourcing/near_minibrot_julia.py score --workers 4 --max-minutes 25
  uv run python tools/sourcing/near_minibrot_julia.py readout
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "atlas",
           ROOT / "tools" / "corpus", ROOT / "tools" / "mining",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import production_seeder as ps                  # noqa: E402
import prescreen                                # noqa: E402
import guard                                    # noqa: E402
from tools.emission import floors as F          # noqa: E402  THE cut owner

STAMP = "2026-08-03"
GEN_VERSION = "q4_near_minibrot_v1"
# SCRATCH, and the class is argued rather than defaulted. `candidates.jsonl` is a pure
# function of two committed inputs (the roster, the harvest's candidates) plus a seed, so it
# regenerates in seconds; `scored.jsonl` adds only the pinned checkpoint and regenerates in
# ~10 engine-minutes. Neither is a durable record of anything — what IS durable is the label
# batch this leg produces (`data/label_corpus/batches/`) and the per-row feature join written
# beside it, which is where a later reader actually needs the ladder rung to survive.
# `bulk()` was the first choice and was WRONG: `data/near_minibrot/` is not a relocated
# prefix, so bulk() resolved it back to an in-tree gitignored path — a file living in the
# repo under a `data/` name that nothing tracks, which is the confusing half-state
# `storage_classes.md` exists to prevent.
DRAW_SEED = 20260803

# The ladder, in units of the atom's own radius `1/|A|`. Three rungs an order of magnitude
# apart, so a monotone trend is visible with three points and the leg does not need a dense
# sweep to say "closer is better" or "closer is worse".
LADDER = (1.0, 4.0, 16.0)

# `julia_c_sourcing.md`: the class favours wide, whole-Julia framings. Drawn per candidate
# from this band rather than pinned, so the frame is not a hidden constant of the leg.
FW_LO, FW_HI = 1.0, 1.4

PARTITION = "julia:mandelbrot"
DEGREE = 2

# Sources of supply. Both are already on disk; neither is enumerated here.
ROSTER = ROOT / "data" / "minibrot_roster" / "roster.jsonl"
HARVEST_CANDIDATES = (ROOT / "data" / "discovery" / "label_seeded_v2_20260802" /
                      "candidates.jsonl")


def _jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# =========================================================================== #
# 1. supply — degree-2 nuclei with a known atom radius
# =========================================================================== #
def load_nuclei() -> tuple[list[dict], dict]:
    """Every distinct degree-2 nucleus on disk, with its `1/|A|` radius. No enumeration.

    Deduped on the rounded coordinate pair rather than on a source id, because the roster and
    the harvest reach the same atoms by different routes and a source-keyed dedup would
    ladder the same nucleus twice under two names."""
    out, seen, rep = [], set(), Counter()
    for src, path, keys in (
            ("roster", ROSTER, ("cx", "cy", "size", "period")),
            ("harvest", HARVEST_CANDIDATES, ("cx", "cy", "window_scale", "period"))):
        if not Path(path).exists():
            rep[f"missing:{src}"] += 1
            continue
        for r in _jl(path):
            rep[f"{src}:rows"] += 1
            if int(r.get("degree", 0)) != DEGREE:
                continue
            size = r.get(keys[2])
            if size is None or float(size) <= 0:
                rep[f"{src}:no_size"] += 1
                continue
            cx, cy = float(r[keys[0]]), float(r[keys[1]])
            key = (round(cx, 12), round(cy, 12))
            if key in seen:
                rep["dup_nucleus"] += 1
                continue
            seen.add(key)
            out.append(dict(source=src, cx=cx, cy=cy, size=float(size),
                            period=int(r.get(keys[3]) or 0),
                            atom_id=r.get("id") or r.get("atom_id")))
            rep[f"{src}:kept"] += 1
    out.sort(key=lambda a: (a["cx"], a["cy"]))     # deterministic, source-order independent
    rep["nuclei"] = len(out)
    return out, dict(rep)


# =========================================================================== #
# 2. the ladder draw
# =========================================================================== #
def sample_candidates(nuclei, *, n_nuclei: int, seed: int = DRAW_SEED) -> list[dict]:
    """One `c` per (nucleus, rung). The angle is drawn; the RADIUS is the experiment.

    The nucleus subset is drawn WITHOUT reference to any score — there is no quality signal
    on an atom here and inventing one would confound the ladder with it. The angle is per
    (nucleus, rung) so two rungs of one atom are not collinear samples of one direction,
    which would make the ladder a radial transect of a single arm rather than a distance
    measurement.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(nuclei))[:min(n_nuclei, len(nuclei))]
    rows = []
    for j in sorted(idx):
        a = nuclei[j]
        for rung in LADDER:
            th = float(rng.uniform(0.0, 2.0 * math.pi))
            r = rung * a["size"]
            cre = a["cx"] + r * math.cos(th)
            cim = a["cy"] + r * math.sin(th)
            rows.append(dict(
                cid=f"nm{len(rows):05d}",
                c_re=cre, c_im=cim,
                ladder_rung=rung, ladder_radius=r, theta=th,
                atom_cx=a["cx"], atom_cy=a["cy"], atom_size=a["size"],
                atom_period=a["period"], atom_id=a["atom_id"], atom_source=a["source"],
                # the wide whole-julia frame, drawn per candidate
                cx=0.0, cy=0.0, fw=float(rng.uniform(FW_LO, FW_HI)),
                family=PARTITION,
            ))
    return rows


def stage_sample(args) -> int:
    nuclei, rep = load_nuclei()
    if not nuclei:
        raise SystemExit(f"no degree-{DEGREE} nuclei found on disk — checked {ROSTER} and "
                         f"{HARVEST_CANDIDATES}. This leg does NOT enumerate; supply must "
                         f"already exist.")
    rows = sample_candidates(nuclei, n_nuclei=args.n_nuclei, seed=args.seed)
    p = paths.scratch("near_minibrot", "candidates.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)   # bulk() resolves, it does not create
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"near-minibrot ladder: {len(rows)} candidates = "
          f"{len(rows)//len(LADDER)} nuclei x {len(LADDER)} rungs {LADDER}")
    print(f"  supply: {json.dumps(rep)}")
    print(f"  atom radius = 1/|A| (the A instrument); rungs are MULTIPLES of it")
    print(f"  -> {p}")
    return 0


# =========================================================================== #
# 3. score — one render per candidate, guarded, decoded
# =========================================================================== #
def stage_score(args) -> int:
    src = paths.scratch("near_minibrot", "candidates.jsonl")
    if not src.exists():
        raise SystemExit(f"{src} missing — run `sample` first.")
    rows = _jl(src)
    out_p = paths.scratch("near_minibrot", "scored.jsonl")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = {r["cid"] for r in _jl(out_p)} if out_p.exists() else set()
    todo = [r for r in rows if r["cid"] not in done]
    print(f"score: {len(todo)} of {len(rows)} to do ({len(done)} already scored); "
          f"{args.workers} engine processes", flush=True)
    if not todo:
        return 0

    scorer = guard.make_guarded_scorer(ps.SCORER_PATH)
    tiles = paths.scratch("near_minibrot", "tiles")
    tiles.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_minutes * 60.0 if args.max_minutes else None

    def render_one(r):
        tile = tiles / f"{r['cid']}.jpg"
        ok, err = prescreen._render(r["cx"], r["cy"], r["fw"], tile,
                                    family="julia", c=(r["c_re"], r["c_im"]),
                                    timeout=args.render_timeout)
        return r, (tile if ok else None), err

    t0, n, fails, stopped = time.time(), 0, [], False
    fh = open(out_p, "a", encoding="utf-8")
    try:
        # Render in blocks, then score each block in ONE batched GPU pass. A per-tile score
        # would pay the transform+forward setup 360 times for a batch the head does in one.
        for i in range(0, len(todo), args.block):
            if deadline and time.time() > deadline:
                stopped = True
                break
            block = todo[i:i + args.block]
            got = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for fut in as_completed([ex.submit(render_one, r) for r in block]):
                    r, tile, err = fut.result()
                    if tile is None:
                        fails.append(dict(cid=r["cid"], err=str(err)[:200]))
                    else:
                        got.append((r, tile))
            if not got:
                continue
            rows_k = scorer.score_paths_k([str(t) for _, t in got])
            for (r, tile), k in zip(got, rows_k):
                eord, nb, pg = float(k[0]), float(k[1]), float(k[2])
                pg4 = float(k[3]) if len(k) > 3 else None
                rec = dict(r, eord=eord, p_notbad=nb, p_good=pg, p_ge4=pg4,
                           decoded_class=F.good_class(pg, pg4),
                           scorer_version=ps.SCORER_VERSION,
                           gen_version=GEN_VERSION)
                fh.write(json.dumps(rec) + "\n")
                n += 1
                tile.unlink(missing_ok=True)
            fh.flush()
            el = time.time() - t0
            print(f"  [{n}/{len(todo)}] {el:.0f}s  {n/max(el,1e-9)*60:.1f}/min  "
                  f"ETA {(len(todo)-n)*el/max(n,1)/60:.1f} min  ({len(fails)} failed)",
                  flush=True)
    finally:
        fh.close()
    if fails:
        # The WHOLE failure population, never a truncated head (`CLAUDE.md`).
        fp = paths.scratch("near_minibrot", "render_failures.json")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(dict(n=len(fails), by_class=dict(Counter(
            f["err"].split(":")[0] for f in fails)), failures=fails), indent=2),
            encoding="utf-8")
        print(f"  !! {len(fails)} render failures -> {fp}")
    print(f"score: {n} scored{' — STOPPED at the time bound' if stopped else ''}; "
          f"-> {out_p}")
    return 0


# =========================================================================== #
# 4. readout — the ladder is the question
# =========================================================================== #
def stage_readout(args) -> int:
    p = paths.scratch("near_minibrot", "scored.jsonl")
    rows = _jl(p)
    if not rows:
        raise SystemExit(f"{p} is empty — run `score` first.")
    out = dict(n=len(rows), gen_version=GEN_VERSION,
               scorer_version=rows[0]["scorer_version"], good_floor=F.GOOD_FLOOR,
               ladder=list(LADDER), fw_band=[FW_LO, FW_HI])
    per = {}
    for rung in LADDER:
        sub = [r for r in rows if r["ladder_rung"] == rung]
        if not sub:
            continue
        e = np.array([r["eord"] for r in sub])
        d = np.array([(r["decoded_class"] or 0) for r in sub])
        k = int((d >= 3).sum())
        per[str(rung)] = dict(
            n=len(sub), eord_median=round(float(np.median(e)), 4),
            eord_p90=round(float(np.percentile(e, 90)), 4),
            eord_max=round(float(e.max()), 4),
            decoded_ge3=k, decoded_ge3_rate=round(k / len(sub), 4),
            wilson95=_wilson(k, len(sub)),
            decoded_4=int((d >= 4).sum()))
    out["per_rung"] = per
    # The comparison that makes the rung readable: the same head on the same family from the
    # run's own boundary-sampled population would be the control, and it is NOT computed
    # here — this leg is train-side and its rate is not a base rate (module doc).
    out["CAVEAT"] = ("rates here are conditioned on nuclei two prior searches surfaced; "
                     "they are NOT a base rate for near-∂M julia c and must not be compared "
                     "to one without matching on how the nuclei were found")
    q = paths.scratch("near_minibrot", "readout.json")
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"  -> {q}")
    return 0


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 1.0]
    ph, d = k / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n-nuclei", type=int, default=120)
    s.add_argument("--seed", type=int, default=DRAW_SEED)
    s.set_defaults(fn=stage_sample)
    c = sub.add_parser("score")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--block", type=int, default=48)
    c.add_argument("--render-timeout", type=float, default=180.0)
    c.add_argument("--max-minutes", type=float, default=0.0)
    c.set_defaults(fn=stage_score)
    o = sub.add_parser("readout")
    o.set_defaults(fn=stage_readout)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 0) and a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
