#!/usr/bin/env python
"""Blind manifest for a maneuver-enabled frontier run.

EVERY admission (no pre-filtering beyond standard admission), shuffled, arm-free, each
rendered at the deploy-canonical 640x360 ss2 twilight presentation. Writes a HIDDEN key
(`manifest_key.json`: tile -> id / operator / k / origin node / depth / **fw** / canonical
p_good / family / coords) plus the blind index the human scores from (`blind_index.json`).
Same format as the dive and run-2 manifests, so the same self-contained labeler
(`tools/atlas/build_blind_labeler.py`) drives the read.

WHY A THIRD MANIFEST AND NOT `dive_manifest.py`. That one hard-requires
`dive_admissions.npz` (morph clusters from the dive report) and stamps the dive's
start-group. A breadth run has neither. What the maneuver read needs stamped instead is the
**operator, its `k`, the origin node, and `fw` alongside depth** — a snap-and-rescale
changes `fw` without changing the walk-rung count, so a read that matches on depth alone is
measuring depth. Morph clusters are OPTIONAL here (folded in if the npz happens to exist).

THE KEY IS HIDDEN AND THE ARMS ARE NOT SEPARATED. Maneuver-originated and ordinary
admissions are shuffled into one pool: the operator label lives only in the key, so the
human scores views, not provenance.

  uv run python tools/atlas/maneuver_manifest.py --run-dir data/discovery/maneuver_shakedown
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))


def admitted(rows):
    """The ledger's own admission predicate: distinct AND guard-passed AND q3+ (class 4
    admits too — v8 is a K=4 head, so `== 3` would silently drop the best rows)."""
    return [r for r in rows if r.get("distinct") and r.get("guard_pass", True)
            and (r.get("decoded_class") or 0) >= 3]


def main():
    import tools.studies.steered_pilot_morph as spm      # noqa: E402 (heavy; render helpers)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap tiles (0 = all admissions)")
    args = ap.parse_args()
    run = args.run_dir
    out_dir = args.out_dir or (ROOT / "scratch" / "maneuver_manifest" / run.name)

    led = run / "outcome_ledger.jsonl"
    if not led.exists():
        raise SystemExit(f"no outcome_ledger.jsonl in {run}")
    rows = admitted([json.loads(l) for l in open(led, encoding="utf-8") if l.strip()])

    cluster_of = {}
    npz = run / "morph_admissions.npz"
    if npz.exists():                       # optional; the read does not depend on it
        z = np.load(npz, allow_pickle=False)
        cluster_of = {str(u): int(c) for u, c in zip(z["uids"], z["cluster_strict"])}

    items = []
    for r in rows:
        man = r.get("maneuver") or {}
        cpg = r.get("canon_pgood")
        items.append(dict(
            tile=None, id=r["id"], family=r["family"],
            cx=r["outcome_cx"], cy=r["outcome_cy"], fw=r["outcome_fw"],
            c=([r["julia_c_re"], r["julia_c_im"]] if r.get("julia_c_re") is not None else None),
            p_good=float(r["p_good"]), p_notbad=float(r["p_notbad"]),
            p_ge4=r.get("p_ge4"), decoded_class=r.get("decoded_class"),
            canon_pgood=(float(cpg) if cpg is not None else None),
            # BOTH axes, always: depth is the walk-rung count and is UNCHANGED by a
            # reframe, while fw moves by orders of magnitude. Match on both or the read
            # measures depth.
            depth=int(r["reached_depth"]), seed_fw=r.get("seed_fw"),
            mix_source=r.get("mix_source"),
            maneuver_op=man.get("op"), maneuver_k=man.get("k"),
            maneuver_origin_node_id=man.get("origin_node_id"),
            maneuver_atom_id=man.get("atom_id"), maneuver_period=man.get("period"),
            maneuver_parent_node_id=man.get("parent_node_id"),
            maneuver_parent_fw=man.get("parent_fw"),
            cluster=cluster_of.get(r["id"], -1),
        ))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(items)
    if args.limit:
        items = items[:args.limit]

    tiles = out_dir / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    key = []
    for i, x in enumerate(items):
        tile = tiles / f"blind_{i:03d}.jpg"
        spm.render_colored(spm.loc_of_row(by_id[x["id"]]), tile)
        if not tile.exists():
            raise SystemExit(f"render failed for {x['id']} -> {tile}")
        x["tile"] = tile.name
        key.append(x)

    (out_dir / "manifest_key.json").write_text(json.dumps(dict(
        run=run.name, n=len(key), n_admissions=len(rows), seed=args.seed,
        note="HIDDEN KEY — do not show the human labeler; maps blind tile -> truth "
             "(operator, k, origin node, depth, fw, canonical p_good).",
        entries=key,
    ), indent=2), encoding="utf-8")
    (out_dir / "blind_index.json").write_text(json.dumps(dict(
        run=run.name, n=len(key),
        instructions="Score each tile 1(bad)/2(okay)/3(good)/4(great). Tiles are shuffled; "
                     "no coords/scores/provenance shown. Return {tile: score}.",
        tiles=[e["tile"] for e in key],
    ), indent=2), encoding="utf-8")

    ops = Counter(e["maneuver_op"] or "ordinary" for e in key)
    dep = Counter(e["depth"] for e in key)
    print(f"maneuver manifest: {len(key)} tiles from {len(rows)} admissions (ALL, no pre-filter)")
    print(f"  by operator: {dict(ops)}")
    print(f"  by depth   : {dict(sorted(dep.items()))}")
    print(f"  fw range   : {min((float(e['fw']) for e in key), default=float('nan')):.3e} .. "
          f"{max((float(e['fw']) for e in key), default=float('nan')):.3e}")
    print(f"wrote {out_dir/'manifest_key.json'} (hidden) + blind_index.json + tiles/")


if __name__ == "__main__":
    main()
