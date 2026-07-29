#!/usr/bin/env python
r"""Committed producer for the julia-arm seed population `steered_frontier` draws from.

WHY THIS EXISTS. `steered_frontier --julia-seed-pool <file>` injects sampler-sourced julia
roots from a flat `[{"c_re", "c_im"}, ...]` list. That list was made ONCE, by hand, from
`q4_decisive_pass.py`'s `viable.json` (its `screen` stage output) — drop the exemplar anchor,
keep every viable near-boundary c, project to (c_re, c_im). The 534-row result lived only in
`scratch/q4_decisive/julia_seed_pool.json`, a class whose contract GUARANTEES deletion, and the
filter that produced it was remembered, not written. This module makes both durable: the filter
is `filter_seed_pool` (below), and the output is written through `paths.durable()` to a
git-tracked home, so the population survives `rm -r scratch/*` and cannot be silently discarded.

This is a FORWARD fix (storage_classes.md rule 3): it does not resurrect a wiped copy — the
scratch file is present, and this reproduces it exactly from `viable.json` and lands it durably.

  uv run python tools/atlas/build_julia_seed_pool.py                    # default viable.json
  uv run python tools/atlas/build_julia_seed_pool.py --viable <path>    # explicit input

Output: data/atlas/julia_seed_pool.json (durable). Point `steered_frontier --julia-seed-pool`
at it instead of the scratch path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402   the durability-class declaration

# The durable population snapshot (git-tracked). Readers point here.
SEED_POOL_REL = "data/atlas/julia_seed_pool.json"
SEED_POOL_JSON = ROOT / SEED_POOL_REL
# Default input: the q4_decisive screen stage's viable set (regenerable via that pass).
VIABLE_DEFAULT = ROOT / "scratch" / "q4_decisive" / "viable.json"


def filter_seed_pool(viable_rows) -> list[dict]:
    """The ad-hoc filter, now in code: every VIABLE near-boundary c EXCEPT the exemplar
    anchor, projected to (c_re, c_im), in `viable.json` order. The anchor (`anchor: true`,
    cid `exemplar`) is the positive control, not a sampler root, so it is excluded exactly as
    the hand-built pool did."""
    out = []
    for r in viable_rows:
        if r.get("anchor") or r.get("cid") == "exemplar":
            continue
        out.append({"c_re": r["c_re"], "c_im": r["c_im"]})
    return out


def seed_pool_path() -> Path:
    """The durable output path; `durable()` raises if git would discard it."""
    return paths.durable(SEED_POOL_REL, mkparents=True)


def build(viable_path: Path = VIABLE_DEFAULT) -> tuple[Path, int]:
    viable = json.loads(Path(viable_path).read_text(encoding="utf-8"))
    pool = filter_seed_pool(viable)
    out = seed_pool_path()
    out.write_text(json.dumps(pool), encoding="utf-8")
    return out, len(pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viable", type=Path, default=VIABLE_DEFAULT,
                    help="q4_decisive viable.json (the screen-stage output)")
    args = ap.parse_args()
    if not args.viable.exists():
        raise SystemExit(
            f"viable.json not found: {args.viable}\n"
            f"Regenerate it with: uv run python tools/studies/q4_decisive_pass.py screen")
    out, n = build(args.viable)
    print(f"wrote {out.relative_to(ROOT)}  ({n} julia seed c's, exemplar anchor dropped)")


if __name__ == "__main__":
    main()
