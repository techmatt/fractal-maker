#!/usr/bin/env python
r"""build_julia_supply_pool_v2.py — the harvest-v2 `julia:mandelbrot` supply pool.

WHAT THIS REPLACES. v1 fed `julia:mandelbrot` from one flat list of 534 near-boundary `c`s
(`build_julia_seed_pool.py`) and ran the near-minibrot ladder as a separate labelling leg. The
q4 sitting then priced the three sources against 870 human labels and they are not
interchangeable (`supply_routing.py` carries the table):

    ranked q4 harvest      >=3  79.1%   class-4  16.3%
    near-minibrot ladder   >=3  66.6%   class-4   4.8%
    unscreened boundary    >=3  16.7%   class-4   0.0%

So v2 merges them into ONE pool with the yield ordering as the merge priority, and thins the
result to the measured c-spacing floor. Two properties follow from that shape and neither is
incidental:

  * **First-wins thinning means the ordering IS the policy.** A cluster of near-duplicate `c`s
    collapses to its highest-priced member, so the pool spends its variety budget on the
    channel that earns it. Sorting the pool after thinning instead would keep an arbitrary
    representative of each cluster.
  * **The floor is applied ACROSS channels, not within them.** The saturation is a property of
    c-plane distance, not of which search found the point (`supply_routing.CSPACING_BASIS`) —
    a ladder `c` and a harvest `c` a millidistance apart are the same picture whatever their
    provenance says, and per-channel thinning would keep both.

THE SINGLE RUNG. The ladder contributes ONE `c` per nucleus, at the rung
`supply_routing.rung_choice()` names from its measured cost record — not three. The 1x/4x/16x
ladder buys ~1 look per atom (same-atom pairs at median cos 0.9825, 74.1% at or above the
0.974 cut) for 3x the label cost.

THE TWO MINING CHANNELS are the harvest's own two tiers, not a new split: `rank_tier=2` rows
carry a canonical decode (the RANKED channel) and `rank_tier=1` rows carry only a cheap score
(the RECALL channel, everything at or above `tau_rec` that never earned a confirmation
render). They are never pooled into one ranking — two geometries, two scores — so they enter
the merge as two priority bands with the ranked band first.

  uv run python tools/atlas/build_julia_supply_pool_v2.py
  uv run python tools/atlas/build_julia_supply_pool_v2.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "sourcing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import supply_routing as sr                     # noqa: E402

POOL_REL = "data/atlas/julia_supply_pool_v2.json"
V1_POOL = ROOT / "data" / "atlas" / "julia_seed_pool.json"
Q4_STORE = (ROOT / "data" / "discovery" / "q4_long_harvest_20260803" /
            "q4_candidates.jsonl")
PARTITION = "julia:mandelbrot"

# Merge priority: the measured yield order. First wins under the c-spacing floor, so this
# list is the policy and not a presentation choice.
CHANNEL_ORDER = ("q4_mining_ranked", "q4_mining_recall", "near_minibrot", "seeded_loop")


def _jl(p: Path):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# --------------------------------------------------------------------------- #
# channels
# --------------------------------------------------------------------------- #
def channel_q4_mining(store: Path = Q4_STORE) -> tuple[list, list]:
    """The two mining tiers, each sorted best-first WITHIN the tier.

    Rows are keyed on the julia parameter `c`, not on the viewport: the pool seeds ROOTS, and
    a root is a `c`. Tiers are never pooled — a `rank_tier=2` score comes off a 640x360 ss2
    canonical render and a `rank_tier=1` score off a 384x216 ss1 cheap one, and ordering them
    together is the cap/geometry error `orbital_field_metrics.md` §5 forbids."""
    if not store.exists():
        return [], []
    ranked, recall = [], []
    for r in _jl(store):
        if r.get("partition") != PARTITION:
            continue
        cre, cim = r.get("julia_c_re"), r.get("julia_c_im")
        if cre is None or cim is None:
            continue
        row = dict(c_re=float(cre), c_im=float(cim),
                   channel=None, score=r.get("rank_score"),
                   fate=r.get("fate"), src_batch=r.get("batch"))
        if r.get("rank_tier") == 2:
            row["channel"] = "q4_mining_ranked"
            ranked.append(row)
        elif r.get("rank_tier") == 1:
            row["channel"] = "q4_mining_recall"
            recall.append(row)
    for lst in (ranked, recall):
        lst.sort(key=lambda x: -(x["score"] if x["score"] is not None else -1e18))
    return ranked, recall


def channel_near_minibrot(rung: float, n_nuclei: int | None = None) -> list:
    """ONE `c` per degree-2 nucleus on disk, at `rung` atom radii, angle drawn per nucleus.

    Reuses `near_minibrot_julia`'s own supply loader and draw arithmetic rather than
    reimplementing them — a second copy of "where is the atom and how big is it" is how the
    leg that produced the labels and the channel that consumes them silently diverge. No
    fresh enumeration: enumeration is ~25x screening, and these nuclei are already on disk."""
    import numpy as np
    import near_minibrot_julia as nmj

    nuclei, rep = nmj.load_nuclei()
    if not nuclei:
        return []
    rng = np.random.default_rng(nmj.DRAW_SEED)
    idx = range(len(nuclei)) if n_nuclei is None else \
        sorted(rng.permutation(len(nuclei))[:min(n_nuclei, len(nuclei))])
    out = []
    for j in idx:
        a = nuclei[j]
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        r = rung * a["size"]
        out.append(dict(c_re=a["cx"] + r * math.cos(th),
                        c_im=a["cy"] + r * math.sin(th),
                        channel="near_minibrot", score=None,
                        atom_id=a["atom_id"], atom_size=a["size"], ladder_rung=rung))
    # Deterministic, and ordered by atom size DESCENDING: a bigger atom is a coarser feature,
    # and under first-wins thinning the coarse one should own its neighbourhood.
    out.sort(key=lambda x: (-x["atom_size"], x["c_re"], x["c_im"]))
    return out


def channel_seeded_loop(pool: Path = V1_POOL) -> list:
    """The v1 near-boundary `c`s. Kept — the deficit is large and this is the only channel
    with no selection bias at all — but LAST in the merge, because it measured 16.7% at >=3
    against the ranked harvest's 79.1%."""
    if not pool.exists():
        return []
    return [dict(c_re=float(r["c_re"]), c_im=float(r["c_im"]),
                 channel="seeded_loop", score=None)
            for r in json.loads(pool.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(*, n_nuclei: int | None = None, floor: float | None = None) -> dict:
    rung = sr.rung_choice()
    floor = sr.CSPACING_FLOOR if floor is None else float(floor)
    ranked, recall = channel_q4_mining()
    by_channel = {
        "q4_mining_ranked": ranked,
        "q4_mining_recall": recall,
        "near_minibrot": channel_near_minibrot(rung["rung"], n_nuclei),
        "seeded_loop": channel_seeded_loop(),
    }
    merged = []
    for ch in CHANNEL_ORDER:
        merged.extend(by_channel[ch])
    kept, dropped = sr.thin_by_cspacing(merged, floor=floor)

    def tally(rows):
        out: dict = {}
        for r in rows:
            out[r["channel"]] = out.get(r["channel"], 0) + 1
        return out

    return dict(
        pool=[dict(c_re=r["c_re"], c_im=r["c_im"], channel=r["channel"]) for r in kept],
        report=dict(
            partition=PARTITION, floor=floor, rung=rung["rung"], rung_why=rung["why"],
            merge_order=list(CHANNEL_ORDER),
            proposed=tally(merged), kept=tally(kept), dropped=tally(dropped),
            n_proposed=len(merged), n_kept=len(kept), n_dropped=len(dropped),
            thinning_rate=(round(len(dropped) / len(merged), 4) if merged else None),
            cspacing_basis=sr.CSPACING_BASIS,
            note="first-wins thinning ACROSS channels in measured-yield order; the ordering "
                 "is the policy",
        ))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--n-nuclei", type=int, default=None,
                    help="cap the near-minibrot channel (default: every nucleus on disk)")
    ap.add_argument("--floor", type=float, default=None)
    a = ap.parse_args()
    res = build(n_nuclei=a.n_nuclei, floor=a.floor)
    print(json.dumps(res["report"], indent=2))
    if a.dry_run:
        print("[dry-run] nothing written")
        return
    p = paths.durable(POOL_REL, mkparents=True)
    p.write_text(json.dumps(res["pool"], indent=1) + "\n", encoding="utf-8")
    rp = p.with_name(p.stem + "_report.json")
    rp.write_text(json.dumps(res["report"], indent=2) + "\n", encoding="utf-8")
    print(f"-> {p} ({len(res['pool'])} c's)\n-> {rp}")


if __name__ == "__main__":
    main()
