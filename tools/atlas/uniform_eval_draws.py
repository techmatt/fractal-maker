#!/usr/bin/env python
r"""uniform_eval_draws.py — score-unconditioned draws for the partitions with NO eval rows.

THE GAP THIS FILLS, named by production code rather than by me.
`production_seeder.T_GOOD_UNCALIBRATED` lists the partitions whose `t_good` is the 0.50
baseline rather than a derived number, and its comment separates three different reasons:
multibrot3/4/5 have 24/25/29 unbiased draws each under v10 and are uncalibrated because **not
one of the 78 was a keeper**; julia:mandelbrot and phoenix are the **never-looked** case, with
no unbiased eval rows in v8 or v10 at all; `phoenix:classic` (registered 2026-08-04) is the
third case — a human HAS looked at all 73 of its labels, they are simply below the
estimator's sufficiency floor, and it is NOT DRAWN here (see `NOT_DRAWN`).
Phoenix is flagged there as "the one to watch" —
573 training locations, the only partition where class 4 outnumbers class 3, and running on a
conservative default rather than on evidence. That is what this leg is for.

WHY THIS IS AN EVAL DRAW AND THE SUPPLY CRAWL'S UNIFORM LEG WAS NOT. That leg was uniform
over ONE RUN'S OWN CANDIDATE POPULATION, so it estimates that crawl's base rate and nothing
wider — registering it eval would have moved the instrument to match whatever the crawl
surfaced. Every draw here is taken from the FAMILY'S OWN PARAMETER SPACE by a closed-form or
membership rule, before any run exists and with no score anywhere in the selection:

  * **phoenix** — points on the closed-form neutral-stability skeleton (`phoenix_sampler`,
    spec §2). The curve is exact and score-free; sampling it is sampling the family.
  * **julia:mandelbrot** — boundary-rejection sample of the near-∂M shell: `c` kept iff
    membership is non-constant over `{c} ∪ ring(ε)`. `julia_c_sourcing.md`'s stage 1, WITHOUT
    its stages 2 and 3 — the viability screen and the `|dist_dM|` ranking are both selections,
    and an eval instrument may not have either.
  * **multibrot3/4/5** — the same shell rule at degree `d` (`z^d + c`), which is the honest
    generalisation: the shell is defined by a membership test, not by a fitted boundary.

NO SCORE, NO SCREEN, NO CLASSIFIER — not even to reject a blank frame. A draw that dropped
its own black tiles would be a screened draw wearing an unbiased name, and the black tiles
are exactly the negative mass an eval slice needs. Rows are LABELED after the fact; the
classifier never sees them before Matt does.

PRIORITY ORDER when the budget cannot cover everything, and it is the prompt's:
**phoenix -> native multibrot -> julia:mandelbrot** — never-looked and most anomalous first.

  uv run python tools/atlas/uniform_eval_draws.py draw --n 290
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "phoenix",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import phoenix_sampler as pxs                   # noqa: E402

STAMP = "2026-08-03"
GEN_VERSION = "q4_uniform_eval_v1"
DRAW_SEED = 20260803

# The prompt's priority order, and `T_GOOD_UNCALIBRATED`'s own reasons behind it.
PRIORITY = ("phoenix", "multibrot3", "multibrot4", "multibrot5", "julia:mandelbrot")

# Uncalibrated partitions this leg deliberately does NOT draw, each with the reason. Named
# rather than omitted, and CHECKED against `T_GOOD_UNCALIBRATED` at import (below): a
# partition that silently fell out of both PRIORITY and this map would be an uncalibrated
# family nobody drew and nobody said they were not drawing.
NOT_DRAWN = {
    # Registered 2026-08-04. Its parameter space is ONE point, so there is no parameter-space
    # rule to sample — the family's own closed form degenerates. An unbiased classic draw
    # would have to be uniform over VIEWPORTS inside the fixed plane, which is a different
    # instrument from every one above (those all sample a parameter space before any view
    # exists) and is not built. Drawing it with a viewport rule and reporting it beside the
    # parameter-space draws would pool two instruments into one slice.
    "phoenix:classic": ("parameter space is a single point; an unbiased draw here needs a "
                        "VIEWPORT instrument, which is a different instrument from the "
                        "parameter-space rules this leg uses and is not built"),
}


def _check_uncalibrated_coverage():
    """Every uncalibrated partition is either DRAWN or explicitly NOT_DRAWN with a reason.

    Fail-closed at import, because the failure it guards is silent by construction: a new
    uncalibrated partition that nobody adds here produces a leg that reports full coverage of
    "the partitions with no eval rows" while one of them was never in the draw. Relational —
    it reads the production set rather than a count, so adding a partition to either side
    keeps it true without a re-baselined number."""
    import production_seeder as _ps
    missing = sorted(set(_ps.T_GOOD_UNCALIBRATED) - set(PRIORITY) - set(NOT_DRAWN))
    if missing:
        raise SystemExit(
            f"uniform_eval_draws: uncalibrated partition(s) {missing} are in neither PRIORITY "
            f"nor NOT_DRAWN. This leg exists to draw exactly the uncalibrated set — put each "
            f"one in PRIORITY (with a parameter-space rule in `draw`) or in NOT_DRAWN with the "
            f"reason it cannot be drawn.")
    assert PRIORITY, "PRIORITY is empty — every assertion over it is vacuous"


_check_uncalibrated_coverage()

SHELL_EPS = 0.02        # `q4_decisive_pass.SHELL_EPS` — the near-∂M shell half-width
SHELL_RING_N = 8        # ring samples per candidate for the non-constant-membership test
MEMBER_MAXITER = 400    # membership iterations; a shell test does not need a deep cap
BOX = 2.0               # sampling box half-width in the c-plane (covers every M_d here)

# Viewport policy. WIDE, and fixed per family rather than searched — a search is a selection.
#   z-plane families (julia, phoenix) get the whole-set frame `julia_c_sourcing.md` names.
#   c-plane families get a base-scale frame centred on the origin of the multibrot.
JULIA_FW_LO, JULIA_FW_HI = 1.0, 1.4
CPLANE_FW = 3.0
PHOENIX_FW = 3.0

DEGREE_OF = {"multibrot3": 3, "multibrot4": 4, "multibrot5": 5}


def escapes(c: np.ndarray, d: int, maxiter: int = MEMBER_MAXITER) -> np.ndarray:
    """Vectorised `z <- z^d + c` membership. True = ESCAPED (outside M_d).

    Pure numpy and no engine: a membership test is arithmetic, and spawning the renderer for
    it would make an eval draw depend on render policy."""
    z = np.zeros_like(c)
    out = np.zeros(c.shape, dtype=bool)
    for _ in range(maxiter):
        live = ~out
        if not live.any():
            break
        z[live] = z[live] ** d + c[live]
        out |= np.abs(z) > 2.0
        z[out] = 2.0                      # freeze escaped so |z| cannot overflow
    return out


def near_boundary(c: np.ndarray, d: int, eps: float = SHELL_EPS) -> np.ndarray:
    """True where membership is NON-CONSTANT over `{c} ∪ ring(eps)` — the shell rule.

    This is the whole selection, and it is a statement about the SET, not about the picture:
    it uses no colour, no occupancy, no score. A point deep inside M_d and a point far outside
    both have constant membership on their ring and are both rejected; the boundary is what
    survives."""
    base = escapes(c, d)
    same = np.ones(c.shape, dtype=bool)
    for j in range(SHELL_RING_N):
        th = 2.0 * math.pi * j / SHELL_RING_N
        same &= (escapes(c + eps * complex(math.cos(th), math.sin(th)), d) == base)
    return ~same


def draw_cplane_shell(d: int, n: int, rng, *, max_rounds: int = 400):
    """`n` near-∂M_d points by rejection. Returns `(points, drawn, hits)`.

    `hits` is the number that PASSED the shell test, which is not `len(points)`: the return
    is truncated to `n`. Reporting `n/drawn` as the acceptance rate is how every degree comes
    back at an identical, meaningless 48/4096 — it measures the truncation, not the shell.
    The two are returned separately so the readout cannot conflate them."""
    kept, drawn, hits = [], 0, 0
    while len(kept) < n and max_rounds > 0:
        max_rounds -= 1
        blk = (rng.uniform(-BOX, BOX, 4096) + 1j * rng.uniform(-BOX, BOX, 4096))
        drawn += blk.size
        hit = blk[near_boundary(blk, d)]
        hits += int(hit.size)
        kept.extend(hit.tolist())
    return kept[:n], drawn, hits


def draw(n_per: dict, seed: int = DRAW_SEED) -> tuple[list[dict], dict]:
    """The whole draw, in PRIORITY order so a truncated budget truncates the tail."""
    rng = np.random.default_rng(seed)
    rows, rep = [], {}
    for part in PRIORITY:
        n = int(n_per.get(part, 0))
        if n <= 0:
            continue
        if part == "phoenix":
            # The skeleton is the family's own closed form; the root branch stays IN here,
            # unlike the harvest channel. An eval draw may not inherit a quality verdict —
            # "the root branch is dead to humans" is exactly the kind of claim an unbiased
            # slice should be able to re-test, and excluding it would bake the answer in.
            seeds = pxs.propose_batch(seed + 101, n)
            for i, s in enumerate(seeds):
                rows.append(dict(
                    eid=f"ue_px{i:04d}", partition="phoenix", family="phoenix",
                    cx="0", cy="0", fw=str(PHOENIX_FW),
                    c_re=repr(s.c.real), c_im=repr(s.c.imag),
                    p_re=repr(s.p.real), p_im=repr(s.p.imag),
                    zm1_re=repr(s.z_m1.real), zm1_im=repr(s.z_m1.imag),
                    branch=s.branch, draw_rule="phoenix skeleton (closed form), no screen"))
            rep[part] = dict(n=len(seeds), rule="skeleton", drawn=len(seeds))
        elif part in DEGREE_OF:
            d = DEGREE_OF[part]
            pts, drawn, hits = draw_cplane_shell(d, n, rng)
            for i, c in enumerate(pts):
                rows.append(dict(
                    eid=f"ue_{part}_{i:04d}", partition=part, family=part,
                    cx=repr(c.real), cy=repr(c.imag), fw=str(CPLANE_FW),
                    c_re=None, c_im=None,
                    draw_rule=f"near-∂M_{d} shell eps={SHELL_EPS}, no screen"))
            rep[part] = dict(n=len(pts), rule=f"shell d={d}", drawn=drawn,
                             hits=hits, shell_accept=round(hits / max(1, drawn), 5))
        elif part == "julia:mandelbrot":
            pts, drawn, hits = draw_cplane_shell(2, n, rng)
            for i, c in enumerate(pts):
                rows.append(dict(
                    eid=f"ue_jm_{i:04d}", partition=part, family="julia",
                    cx="0", cy="0", fw=repr(float(rng.uniform(JULIA_FW_LO, JULIA_FW_HI))),
                    c_re=repr(c.real), c_im=repr(c.imag),
                    draw_rule=f"near-∂M shell eps={SHELL_EPS}, no viability screen, "
                              f"no dist_dM ranking"))
            rep[part] = dict(n=len(pts), rule="shell d=2", drawn=drawn,
                             hits=hits, shell_accept=round(hits / max(1, drawn), 5))
        # NO SILENT SHORTFALL: a partition the rejection sampler could not fill is named.
        if rep.get(part, {}).get("n", 0) < n:
            rep.setdefault(part, {})["short_by"] = n - rep.get(part, {}).get("n", 0)
    return rows, rep


def allocate(n_total: int) -> dict:
    """Split `n_total` across the DRAWN partitions in priority order.

    Phoenix gets the double share, and the reason is `T_GOOD_UNCALIBRATED`'s own: it is the
    never-looked partition where class 4 already outnumbers class 3, i.e. the one whose
    conservative default is most likely to be wrong in the expensive direction."""
    w = {"phoenix": 2.0, "multibrot3": 1.0, "multibrot4": 1.0, "multibrot5": 1.0,
         "julia:mandelbrot": 1.0}
    tot = sum(w.values())
    out = {k: int(round(n_total * v / tot)) for k, v in w.items()}
    # fix the rounding drift on the highest-priority partition
    out["phoenix"] += n_total - sum(out.values())
    return out


def stage_draw(args) -> int:
    n_per = allocate(args.n)
    rows, rep = draw(n_per, args.seed)
    p = paths.scratch("uniform_eval", "draws.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"uniform eval draws: {len(rows)} rows over {len(n_per)} partitions "
          f"(requested {args.n})")
    print(f"  allocation (phoenix double share — the never-looked, class-4-heavy one): "
          f"{json.dumps(n_per)}")
    for k in PRIORITY:
        if k in rep:
            print(f"  {k:20s} {json.dumps(rep[k])}")
    print(f"  by partition: {json.dumps(dict(Counter(r['partition'] for r in rows)))}")
    print("  NO score, NO screen, NO classifier anywhere in this selection.")
    print(f"  -> {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--n", type=int, default=290)
    d.add_argument("--seed", type=int, default=DRAW_SEED)
    d.set_defaults(fn=stage_draw)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
