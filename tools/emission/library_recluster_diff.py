#!/usr/bin/env python
r"""library_recluster_diff.py — what the RETROACTIVE merge would change. READ-ONLY.

The forward fix (`descriptor.assign_morph_clusters(..., library=...)`) stops NEW intakes
adding un-deduped seams. It does not touch the seam already in the library: campaign1 and
library_intake_2 clustered separately, so a look held by both sits in two clusters. A
one-pass re-cluster of the whole library measured that at **1268 -> 1258** (10 merges,
0.8%), the method validated by reproducing the committed 745 + 523 split exactly.

Applying it is a different decision from making it. Rewriting those 10 assignments rewrites
COMMITTED library state: `morph_cluster` is an emission cell axis, so cell reachability and
every per-cell deficit shift, and the release record's `morph_cluster` column stops matching
the tags the decisions were taken under. So this module only ever REPORTS. It writes a diff
to scratch/ and never mutates a snapshot, a target measure, or a release record.

  uv run python tools/emission/library_recluster_diff.py [--library DIR] [--out FILE]

What it reports
  * one-pass cluster count vs the library's committed count, per fractal type;
  * every merge as `<absorbed tag> -> <surviving tag>` with member counts and the cosine
    that closed it, flagged when the merge CROSSES a source-batch boundary (the seam) vs
    when it is an intra-pass ordering artifact;
  * the (type, cluster) cell impact: how many committed cells disappear, which is the
    quantity reachability and the deficit model are denominated in.

A merge that crosses a source boundary is the seam the forward fix now prevents; one that
does not is a re-order artifact of single-pass incremental clustering and is NOT evidence
of a second bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.emission import descriptor as D  # noqa: E402


def load_snapshot(library_dir: Path):
    """(rows, embs, committed_tags, source_of) from an intake snapshot.

    `rows` are synthesized from the snapshot's own tags — the snapshot IS the library's
    membership record, and re-deriving families from the ledgers would let a ledger edit
    silently change what "the library" means here."""
    ip = library_dir / D.LIBRARY_INTAKE_NAME
    ep = D.library_emb_source(library_dir)     # npz or the registry's per-look dir
    if not ip.exists() or not ep.exists():
        missing = [p.name for p in (ip, ep) if not p.exists()]
        raise SystemExit(
            f"no library snapshot at {library_dir} (missing {', '.join(missing)}).\n"
            f"The retroactive-merge diff needs the union snapshot's cluster_tags + morph "
            f"embeddings; they are not derivable from the ledgers without re-rendering and "
            f"re-embedding every admitted location (which would itself produce a NEW "
            f"library, not a diff against the committed one).")
    meta = json.loads(ip.read_text(encoding="utf-8"))
    tags = meta["cluster_tags"]
    embs = D.load_embs(ep)
    src = meta.get("source_tags") or {}
    rows = [{"id": i, "family": t.rpartition("#")[0]} for i, t in tags.items() if i in embs]
    return rows, embs, tags, src


def source_of(loc_id: str, src: dict) -> str:
    """Which intake pass a location came from. `source_tags` when the snapshot carries it,
    else the `c1__` id prefix `stage_first_release` uses to disambiguate campaign1."""
    if loc_id in src and src[loc_id]:
        return str(src[loc_id])
    return "campaign1" if loc_id.startswith("c1__") else "library_intake_2"


def diff(library_dir: Path) -> dict:
    rows, embs, committed, src = load_snapshot(library_dir)
    # one pass over the WHOLE library, snapshot order, same threshold, no seed (the seed is
    # what a fresh pass would build for itself).
    onepass = D.assign_morph_clusters(rows, embs)

    members_c = defaultdict(list)
    for i, t in committed.items():
        if i in embs:
            members_c[t].append(i)
    members_1 = defaultdict(list)
    for i, t in onepass.items():
        members_1[t].append(i)

    # committed cluster -> the one-pass cluster its members landed in. A committed cluster
    # whose members split across one-pass clusters is a SPLIT, not a merge; reported apart.
    landed = {t: Counter(onepass[i] for i in ids) for t, ids in members_c.items()}
    splits = {t: dict(c) for t, c in landed.items() if len(c) > 1}
    absorbed = defaultdict(list)          # one-pass cluster -> committed clusters folded in
    for t, c in landed.items():
        if len(c) == 1:
            absorbed[next(iter(c))].append(t)

    merges = []
    for op, comm in sorted(absorbed.items()):
        if len(comm) < 2:
            continue
        # the survivor is the committed cluster owning the one-pass founder (first in order)
        founder = members_1[op][0]
        survivor = committed[founder]
        for t in sorted(comm):
            if t == survivor:
                continue
            ids = members_c[t]
            s_ids = members_c[survivor]
            # best cosine between the two committed clusters' members: what closed the merge
            A = np.stack([embs[i] for i in ids])
            B = np.stack([embs[i] for i in s_ids])
            cosm = float((A @ B.T).max())
            srcs_a = {source_of(i, src) for i in ids}
            srcs_b = {source_of(i, src) for i in s_ids}
            merges.append(dict(
                absorbed=t, survivor=survivor, n_absorbed=len(ids), n_survivor=len(s_ids),
                max_cos=round(cosm, 5),
                absorbed_sources=sorted(srcs_a), survivor_sources=sorted(srcs_b),
                crosses_source_boundary=bool(srcs_a - srcs_b or srcs_b - srcs_a)))

    per_type = {}
    for t in sorted({D.cell_partition(r) for r in rows}):
        per_type[t] = dict(
            committed=len({v for k, v in committed.items() if k in embs and v.startswith(t + "#")}),
            one_pass=len({v for k, v in onepass.items() if v.startswith(t + "#")}))
    return dict(
        library_dir=str(library_dir), n_locations=len(rows),
        n_committed_clusters=len(members_c), n_one_pass_clusters=len(members_1),
        delta=len(members_1) - len(members_c),
        per_type=per_type, n_merges=len(merges),
        n_merges_crossing_source=sum(1 for m in merges if m["crosses_source_boundary"]),
        merges=merges, splits=splits)


def render(d: dict) -> str:
    L = [f"# Retroactive library re-cluster — what the merge WOULD change (read-only)", "",
         f"library snapshot : `{d['library_dir']}`",
         f"locations        : {d['n_locations']}",
         f"committed        : **{d['n_committed_clusters']}** clusters",
         f"one-pass         : **{d['n_one_pass_clusters']}** clusters  "
         f"(**{d['delta']:+d}**)", ""]
    L += ["## per fractal type", "", "| type | committed | one-pass | delta |",
          "|---|--:|--:|--:|"]
    for t, v in d["per_type"].items():
        L.append(f"| {t} | {v['committed']} | {v['one_pass']} | "
                 f"{v['one_pass'] - v['committed']:+d} |")
    L += ["", f"## merges — {d['n_merges']} total, "
              f"{d['n_merges_crossing_source']} crossing a source-batch boundary", "",
          "A merge that CROSSES a source boundary is the intake seam the forward fix now "
          "prevents. One that does not is an ordering artifact of single-pass incremental "
          "clustering, not a second bug.", "",
          "| absorbed | -> survivor | n | max cos | crosses seam | sources |",
          "|---|---|--:|--:|:-:|---|"]
    for m in d["merges"]:
        L.append(f"| `{m['absorbed']}` | `{m['survivor']}` | {m['n_absorbed']}→"
                 f"{m['n_survivor']} | {m['max_cos']:.4f} | "
                 f"{'YES' if m['crosses_source_boundary'] else 'no'} | "
                 f"{'+'.join(m['absorbed_sources'])} → {'+'.join(m['survivor_sources'])} |")
    if d["splits"]:
        L += ["", f"## splits — {len(d['splits'])} committed cluster(s) whose members did NOT "
                  "stay together", "",
              "These are NOT merges and must not be folded in silently.", ""]
        for t, c in sorted(d["splits"].items()):
            L.append(f"- `{t}` -> " + ", ".join(f"`{k}`×{n}" for k, n in sorted(c.items())))
    L += ["", "## downstream impact if APPLIED", "",
          f"- `(type, morph_cluster)` cells lose **{-d['delta'] if d['delta'] < 0 else 0}** "
          f"members of the axis, so cell reachability and every per-cell deficit are "
          f"re-denominated.",
          "- the release record's `morph_cluster` column stops matching the tags the gate/"
          "release decisions were actually taken under.", "",
          "**Not applied.** This module never mutates a snapshot, a target measure, or a "
          "release record."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=str(D.DEFAULT_LIBRARY_DIR))
    ap.add_argument("--out", default=str(ROOT / "scratch" / "emission" /
                                         "library_recluster_diff.md"))
    a = ap.parse_args()
    d = diff(Path(a.library))
    md = render(d)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(md)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
