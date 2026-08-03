#!/usr/bin/env python
r"""phoenix_q4_seeds.py — the phoenix channel's seed pool for `steered_frontier`.

WHAT THIS IS. A thin, reproducible driver over `phoenix_sampler.propose_batch` that writes
the `--phoenix-seed-pool` file the walk injects as phoenix roots. It adds no geometry: every
closed form (the cardioid and period-2 neutral-stability curves, the outward normal, the
offset draw, the `z_{-1}` draw) is the sampler's, and `phoenix_seed_sampler_spec.md` owns
them. What this module owns is the SELECTION applied on top, and there are exactly three
choices, each of which is a settled verdict rather than a preference:

  1. **The root branch is dropped.** `spec §8`: measured against human labels, the root
     branch produced **0 good** — it is dead to humans. `root_p=0.0` here, and the sampler's
     own default (0.12) is untouched for every other caller.
  2. **Mid-|p| is the sweet band** (same section), so proposals outside `[P_LO, P_HI]` are
     rejected and redrawn rather than reweighted — rejection keeps the retained proposals
     exactly the sampler's own distribution restricted to the band, where a reweighting
     would make them a different distribution wearing the sampler's name.
  3. **The classic real-c/real-p/z_{-1}=0 sub-mode stays rare** (the sampler's own 0.05).
     It reproduces known results and is not what a scarcity channel is for.

WHY THE POOL IS SMALL, AND WHY THAT IS THE POINT. Phoenix is a **motif-scarcity** channel,
not a volume one: `spec §8` records it as a narrow-motif vein whose good output is
overwhelmingly one log-spiral theme re-framed (`morphology_dedup.md` §4). So the pool is
sized for parameter-space SPREAD — many distinct skeleton points, each descended shallowly —
rather than for many walks per point. A larger pool would not buy more motifs.

  uv run python tools/phoenix/phoenix_q4_seeds.py --n 96
  uv run python tools/phoenix/phoenix_q4_seeds.py --n 96 --out <path> --seed 20260803
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
for _p in (HERE, ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import phoenix_sampler as pxs                   # noqa: E402

STAMP = "2026-08-03"
# Beside `julia_seed_pool.json`, deliberately: the two files are the same artifact class for
# the two z-plane channels — a durable snapshot of the parameter points a run was seeded
# from — and `data/atlas/` is already un-gitignored for exactly that. A new `data/phoenix/`
# prefix would have needed its own `.gitignore` negation to be durable at all, which is what
# `paths.durable()` refuses to let happen silently.
POOL_REL = "data/atlas/phoenix_seed_pool.json"
DRAW_SEED = 20260803

# `spec §8`: mid-|p| is the sweet band. Stated as an interval on |p| and asserted by a test,
# because a band that drifts is a different population under the same name. The upper bound
# also keeps every proposal inside `|p| < 1`, which §2.1 shows is NECESSARY for an attracting
# fixed point to exist at all — outside it the skeleton the sampler draws near stops
# describing the dynamics.
P_LO, P_HI = 0.25, 0.85

# The two live branches. `root` is excluded — see the module docstring.
BRANCHES = ("cardioid", "period2")
MAX_DRAWS_PER_KEEP = 40      # rejection-sampling backstop, so a bad band cannot spin forever


def in_band(s) -> bool:
    """The band predicate, pure so a test can walk it without drawing anything."""
    return s.branch in BRANCHES and P_LO <= abs(s.p) <= P_HI


def draw_pool(n: int, seed: int = DRAW_SEED) -> tuple[list[dict], dict]:
    """`n` in-band seeds as pool records, plus the draw's own accounting.

    Rejection, not reweighting (see the docstring). The rejected count is REPORTED rather
    than swallowed: it is the acceptance rate of the band, and a band that accepts 2% is a
    band worth re-deciding rather than one worth drawing 50x through."""
    rng_seed, kept, seen = seed, [], Counter()
    drawn = 0
    budget = max(64, n * MAX_DRAWS_PER_KEEP)
    while len(kept) < n and drawn < budget:
        # Draw in blocks off ONE seeded stream: `propose_batch` re-seeds per call, so
        # calling it in a loop with the same seed would return the same seed n times.
        block = pxs.propose_batch(rng_seed, 256, root_p=0.0)
        rng_seed += 1
        for s in block:
            drawn += 1
            seen[s.branch] += 1
            if not in_band(s):
                continue
            kept.append(s)
            if len(kept) >= n:
                break
    rep = dict(requested=n, kept=len(kept), drawn=drawn,
               accept_rate=round(len(kept) / max(1, drawn), 4),
               branch_mix_drawn=dict(seen),
               branch_mix_kept=dict(Counter(s.branch for s in kept)),
               band=[P_LO, P_HI], branches=list(BRANCHES), seed=seed)
    if len(kept) < n:
        # NO SILENT CAP: a short pool is named, never inferred from a count downstream.
        rep["short_by"] = n - len(kept)
    return [_record(s, i) for i, s in enumerate(kept)], rep


def _record(s, i: int) -> dict:
    """One pool entry, in the flat shape `steered_frontier.seed_phoenix_pool` reads.

    The key names are the walk's (`c_re`/`p_re`/`zm1_re`), NOT `phoenix_sampler`'s
    `seed_to_record` shape (`phoenix_c_re`) — that one is the LEDGER identity stamp
    (`production_seeder.phoenix_ident_fields`) and reusing it here would make a pool file and
    a ledger row indistinguishable to a reader. The provenance columns ride along and are
    recorded, never selected on."""
    return dict(
        pid=f"px{i:04d}",
        c_re=s.c.real, c_im=s.c.imag, p_re=s.p.real, p_im=s.p.imag,
        zm1_re=s.z_m1.real, zm1_im=s.z_m1.imag,
        branch=s.branch, theta=s.theta, offset=s.offset, classic=s.classic,
        abs_p=abs(s.p), abs_zm1=abs(s.z_m1),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=DRAW_SEED)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    pool, rep = draw_pool(a.n, a.seed)
    out = Path(a.out) if a.out else paths.durable(POOL_REL, mkparents=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=1) + "\n", encoding="utf-8")

    print(f"phoenix q4 seed pool: {rep['kept']} of {rep['requested']} requested "
          f"({rep['drawn']} drawn, accept {rep['accept_rate']:.1%}) "
          f"branches {rep['branches']} band |p| in [{P_LO}, {P_HI}]")
    print(f"  branch mix kept: {json.dumps(rep['branch_mix_kept'])}")
    if rep.get("short_by"):
        print(f"  !! SHORT BY {rep['short_by']} — the band's acceptance rate could not fill "
              f"the request inside {MAX_DRAWS_PER_KEEP}x budget. Named, not silently sized "
              f"down.")
    if pool:
        ap_ = np.array([r["abs_p"] for r in pool])
        nz = sum(1 for r in pool if abs(r["zm1_re"]) + abs(r["zm1_im"]) > 0)
        print(f"  |p| median {np.median(ap_):.3f} range [{ap_.min():.3f}, {ap_.max():.3f}]; "
              f"non-zero z_-1 on {nz}/{len(pool)} "
              f"(z_-1 is a MORPHOLOGY lever, not a fertility one — spec §8)")
        print(f"  distinct (c,p,z_-1) points: "
              f"{len({(round(r['c_re'],9), round(r['c_im'],9), round(r['p_re'],9), round(r['p_im'],9), round(r['zm1_re'],9), round(r['zm1_im'],9)) for r in pool})}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
