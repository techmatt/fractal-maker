r"""near_dup_groups.py — THE near-duplicate grouping over the pooled render-mode corpus.

WHY THIS EXISTS. Sheet B (`2026-08-10_render_mode_correction_v2`) draws a 3x3
opacity x threshold sweep per `direct_*` mode, and at many locations two cells of that
sweep are the SAME picture: 631 of its 688 same-location near-dup pairs are a `direct_*`
mode duplicating ITSELF at two cells (`sittings_27c_report.md` correction 3). A retrain
that counts both copies gives one location's one look double the weight of every other,
and a split that puts one copy in train and the other in eval is training-on-eval wearing
two image_ids.

So the pooled corpus is grouped BEFORE it is split or weighted:

  * substrate = colored CLIP (`tools/mining/smooth_equivalence.Embedder` — palette-ON
    `vit_base_patch16_clip_224.openai`), the same producer the (27) measurement used. It is
    the only embedding in the tree that can see a mode/palette difference at all: the
    library near-dup runs on `morph_gray`, whose cosine between two modes of one location
    is identically 1.
  * cut = `smooth_equivalence.STRICT_CUT` (0.974), which is
    `emission.descriptor.NEAR_DUP_THRESHOLD` — imported, never re-typed. It is a BORROWED
    yardstick (grayscale-derived; see that module's caveat), so the artifact records the
    within-location cosine quantiles beside every count and a reader can move the cut.
  * pairs are compared WITHIN A LOCATION only. Two renders at different locations are
    different pictures whatever their cosine says, and the duplication this is about is a
    duplication of the sheet row, not of the fractal.
  * groups are the connected components of the >= cut graph (union-find), so a chain
    a~b~c collapses to one group even where cos(a,c) sits under the cut.

**A group is a subset of a location by construction**, which is what makes the split
constraint free: `split_units` groups whole LOCATIONS into union-find units, so a near-dup
group can never straddle train/eval unless a location does. The property is asserted by the
consumer rather than assumed here (`mining_corpus.check_groups_within_units`).

The artifact is DURABLE (`data/render_mode_corpus/near_dup_groups_v1.json`): it is an input
to a training run's row weights, it costs a GPU pass over every crop in the corpus, and a
retrain that silently regrouped would be a different retrain. Rebuild is deterministic given
the crops, so the file records the batches, the cut, the substrate and the counts that
produced it.

    uv run python tools/mining/near_dup_groups.py            # build (writes the artifact)
    uv run python tools/mining/near_dup_groups.py --limit 40 # bounded end-to-end, STAMPED
                                                            #   incomplete; never a build
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.mining import smooth_equivalence as SE   # noqa: E402  substrate + cut owner
from tools.mining.split_units import UF             # noqa: E402  the one union-find here
from tools.paths import durable                     # noqa: E402  storage class at the write site

ARTIFACT_REL = "data/render_mode_corpus/near_dup_groups_v1.json"
# DURABLE, asserted not-gitignored at import: this is an input to a training run's row
# weights, so a retrain that silently regrouped would be a different retrain.
ARTIFACT = durable(ARTIFACT_REL)

# The pooled corpus, in the order the trainer loads it. Adding a batch here is what makes
# it part of the grouping; the artifact records the list so a stale file is visible.
BATCHES = (
    "2026-08-06_render_mode_fresh_sheet_v1",
    "2026-08-10_render_mode_correction_v2",
    "2026-08-10_render_mode_rare_palette_v1",
)
CORPUS = ROOT / "data" / "render_mode_corpus" / "batches"


def log(m):
    print(m, flush=True)


def load_pool(batches=BATCHES, limit: int | None = None) -> list[dict]:
    """`[{image_id, batch, loc, mode, jpg}]` over every row of every pooled batch.

    Crops are required: a row whose crop is missing cannot be embedded, and dropping it
    silently would leave its duplicate ungrouped and double-counted."""
    rows = []
    for b in batches:
        bdir = CORPUS / b
        n_before = len(rows)
        for line in (bdir / "images.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            jpg = bdir / "crops" / f"{r['image_id']}.jpg"
            if not jpg.exists():
                raise FileNotFoundError(f"crop missing: {jpg}")
            rows.append({"image_id": r["image_id"], "batch": b,
                         "loc": r["provenance"]["location_key"],
                         "mode": r["render"]["render_mode"], "jpg": jpg})
            if limit is not None and len(rows) - n_before >= limit:
                break
    return rows


def group_rows(rows: list[dict], vecs: np.ndarray, cut: float = SE.STRICT_CUT):
    """Connected components of the within-location `cos >= cut` graph.

    Returns `(group_of, pairs, cos_all)` — `group_of` maps image_id -> group id (a group of
    one is still a group, so every row has one and a consumer never has to special-case a
    singleton), `pairs` lists the linking pairs, `cos_all` is every within-location cosine
    (the distribution the cut is read against)."""
    by_loc = defaultdict(list)
    for i, r in enumerate(rows):
        by_loc[r["loc"]].append(i)

    uf = UF()
    for r in rows:
        uf.find(r["image_id"])
    pairs, cos_all = [], []
    for loc, idx in sorted(by_loc.items()):
        if len(idx) < 2:
            continue
        sub = vecs[idx]
        cm = sub @ sub.T
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                c = float(cm[a, b])
                cos_all.append(c)
                if c >= cut:
                    ia, ib = rows[idx[a]], rows[idx[b]]
                    uf.union(ia["image_id"], ib["image_id"])
                    pairs.append({"loc": loc, "cos": c, "a": ia["image_id"],
                                  "b": ib["image_id"],
                                  "modes": [ia["mode"], ib["mode"]],
                                  "batches": [ia["batch"], ib["batch"]]})
    roots = {}
    group_of = {}
    for r in rows:
        root = uf.find(r["image_id"])
        group_of[r["image_id"]] = roots.setdefault(root, f"ndg{len(roots):05d}")
    return group_of, pairs, np.asarray(cos_all, dtype=np.float64)


def build(limit: int | None = None, batches=BATCHES) -> dict:
    rows = load_pool(batches, limit=limit)
    log(f"[near-dup] {len(rows)} rows over {len(set(r['loc'] for r in rows))} locations")
    emb = SE.Embedder()
    log(f"[near-dup] {emb.model_name} on {emb.device}")
    t0 = time.time()
    vecs = emb.embed_paths([r["jpg"] for r in rows])
    log(f"[near-dup] embedded in {time.time() - t0:.1f}s")

    group_of, pairs, cos_all = group_rows(rows, vecs)
    sizes = Counter(group_of.values())
    hist = Counter(sizes.values())
    cross_batch = sum(1 for p in pairs if p["batches"][0] != p["batches"][1])
    same_mode = sum(1 for p in pairs if p["modes"][0] == p["modes"][1])

    doc = {
        "artifact": ARTIFACT_REL,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": "uv run python tools/mining/near_dup_groups.py",
        "incomplete": bool(limit is not None),
        "limit_per_batch": limit,
        "batches": list(batches),
        "substrate": SE.yardstick_block()["substrate"],
        "cut": SE.STRICT_CUT,
        "cut_owner": "tools/emission/descriptor.NEAR_DUP_THRESHOLD "
                     "(via tools/mining/smooth_equivalence.STRICT_CUT)",
        "scope": "WITHIN a location_key only; groups are connected components of the "
                 ">= cut graph, so a chain a~b~c is one group",
        "n_rows": len(rows),
        "n_locations": len(set(r["loc"] for r in rows)),
        "n_groups": len(sizes),
        "n_rows_in_a_multi_group": int(sum(v for k, v in sizes.items() if v > 1)),
        "group_size_hist": {str(k): v for k, v in sorted(hist.items())},
        "largest_group": max(sizes.values()) if sizes else 0,
        "n_linking_pairs": len(pairs),
        "n_pairs_cross_batch": cross_batch,
        "n_pairs_same_mode": same_mode,
        "within_location_cos": ({**SE.quantiles(cos_all)} if cos_all.size else {"n": 0}),
        "unrelated_reference": SE.unrelated_reference(vecs),
        "by_batch": {b: {"rows": sum(1 for r in rows if r["batch"] == b),
                         "rows_in_a_multi_group":
                             sum(1 for r in rows if r["batch"] == b
                                 and sizes[group_of[r["image_id"]]] > 1)}
                     for b in batches},
        "top_mode_pairs": dict(Counter(" + ".join(sorted(p["modes"]))
                                       for p in pairs).most_common(12)),
        "group_of": group_of,
    }
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="rows per batch — bounded end-to-end; the artifact it writes is "
                         "STAMPED incomplete=true and must not be used for a build")
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    a = ap.parse_args(argv)

    doc = build(limit=a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log(f"[near-dup] {doc['n_groups']} groups over {doc['n_rows']} rows "
        f"({doc['n_rows_in_a_multi_group']} rows in a multi-row group); "
        f"sizes {doc['group_size_hist']}")
    log(f"-> {a.out}")


if __name__ == "__main__":
    main()
