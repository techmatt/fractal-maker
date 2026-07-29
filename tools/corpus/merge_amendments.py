"""Route a revision batch's scores.json into the amendment overlay (and fresh rows in-row).

The counterpart to merge_scores.py for REVISIONS. An "anchor"/revision batch is a presentation
batch whose rows re-label ALREADY-labeled source rows on the 1..4 scale (see
docs/design/label_rubric.md, data/label_corpus/CORPUS_SCHEMA.md § Revisions). Each such row
carries a provenance pointer back to the source row it revises
(`revises_batch_id` + `revises_image_id`).

This tool reads the revision batch's images.jsonl + scores.json and, per labeled row:

  * REVISION row (revises_* set): writes the new score to the SOURCE batch's amendment file
    `labels/amend_<source_batch_id>.json` as {source_image_id: score}. This NEVER touches the
    source's original label — `label_store.resolve_score` prefers the amendment, and the
    original stays recoverable (`resolve_score(row, sidecar)` with no amendments). A revision
    MAY change a non-null label (that is the whole point: q3 -> q2 demotion or q3 -> q4
    promotion). Re-revising to the same value is a no-op.

  * FRESH row (revises_* null): merges the score in-row into THIS batch's images.jsonl, exactly
    like merge_scores (null -> value only; refuses to change a non-null in-row label).

After --apply, print the `AMENDMENT_LABELS` registry lines to paste into
tools/corpus/label_store.py (a registered amendment file MUST exist on disk, so register only
after this writes it).

Run:
  uv run python tools/corpus/merge_amendments.py --batch 2026-07-26_anchor_class4_v1 \
      [--scores <path>] [--labeler matt] [--labeled-at 2026-07-26] [--apply]

Without --apply it is a dry run (reports every route, writes nothing).
"""
from __future__ import annotations

import argparse
import json
import os

import corpus_common as cc
import label_store as ls


def _atomic_write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0, sort_keys=True)
    os.replace(tmp, path)


def _amend_filename(source_batch_id: str) -> str:
    return f"amend_{source_batch_id}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="the revision/anchor batch_id")
    ap.add_argument("--corpus-root", default=None,
                    help="batches dir the --batch lives under (default: location corpus)")
    ap.add_argument("--scores", default=None, help="scores.json (default: <batch>/scores.json)")
    ap.add_argument("--max-score", type=int, default=4)
    ap.add_argument("--labeler", default="matt")
    ap.add_argument("--labeled-at", default=None)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    a = ap.parse_args()

    corpus_root = a.corpus_root or cc.BATCHES_DIR
    bdir = os.path.join(corpus_root, a.batch)
    images_path = os.path.join(bdir, "images.jsonl")
    scores_path = a.scores or os.path.join(bdir, "scores.json")

    rows = cc.read_jsonl(images_path)
    by_id = {r["image_id"]: r for r in rows}
    # accept BOTH export shapes (flat {id:int} and the labeler's {id:{score,revealed}}) via the
    # shared loader, so an amendment merge takes the exact file the label UI exports.
    import merge_scores
    scores = merge_scores.load_scores(scores_path)

    # source_batch_id -> {source_image_id: (new_score, original_score)}
    revisions: dict[str, dict] = {}
    fresh_filled, fresh_reaffirmed, fresh_conflicts = 0, 0, []
    out_of_range, unknown = [], []
    demotions, promotions, reaffirmed_rev = [], [], []

    for image_id, new_score in scores.items():
        if new_score is None:
            continue
        if not (1 <= new_score <= a.max_score):
            out_of_range.append((image_id, new_score))
            continue
        row = by_id.get(image_id)
        if row is None:
            unknown.append(image_id)
            continue
        prov = row.get("provenance") or {}
        src_batch = prov.get("revises_batch_id")
        src_image = prov.get("revises_image_id")
        if src_batch and src_image:
            orig = prov.get("original_score")
            revisions.setdefault(src_batch, {})[src_image] = (new_score, orig)
            if orig is None:
                pass
            elif new_score > orig:
                promotions.append((src_batch, src_image, orig, new_score))
            elif new_score < orig:
                demotions.append((src_batch, src_image, orig, new_score))
            else:
                reaffirmed_rev.append((src_batch, src_image, orig))
        else:
            cur = row["label"]["score"]
            if cur is None:
                row["label"]["score"] = new_score
                row["label"]["labeler"] = a.labeler
                row["label"]["labeled_at"] = a.labeled_at
                fresh_filled += 1
            elif cur == new_score:
                fresh_reaffirmed += 1
            else:
                fresh_conflicts.append((image_id, cur, new_score))

    # --- report ---
    n_rev = sum(len(v) for v in revisions.values())
    print(f"batch {a.batch}: {len(rows)} rows, {len(scores)} scores in export "
          f"(valid tiers 1..{a.max_score})")
    print(f"  REVISIONS -> amendment stream: {n_rev} across {len(revisions)} source batch(es)")
    print(f"    promotions:  {len(promotions)}")
    print(f"    demotions:   {len(demotions)}   (these move the >=3 boundary)")
    print(f"    reaffirmed:  {len(reaffirmed_rev)}")
    for sb, si, o, n in (demotions + promotions)[:20]:
        print(f"      {sb}/{si}: {o} -> {n}")
    print(f"  FRESH -> in-row (null->value): filled {fresh_filled}, no-op {fresh_reaffirmed}")
    if fresh_conflicts:
        print(f"  REFUSED (fresh, would change a non-null in-row label): {len(fresh_conflicts)}")
        for iid, c, n in fresh_conflicts[:20]:
            print(f"      {iid}: existing {c} != export {n}")
    if out_of_range:
        print(f"  WARNING: {len(out_of_range)} out of range 1..{a.max_score} (skipped), e.g. {out_of_range[:3]}")
    if unknown:
        print(f"  WARNING: {len(unknown)} unknown image_id (skipped), e.g. {unknown[:3]}")

    # Merge each source batch's revisions into (a possibly-existing) amendment file.
    planned_files = {}   # source_batch_id -> (path, merged_dict, filename)
    for src_batch, revs in sorted(revisions.items()):
        fn = _amend_filename(src_batch)
        path = os.path.join(ls.LABELS_DIR, fn)
        merged = {}
        if os.path.exists(path):
            d = json.load(open(path, encoding="utf-8"))
            merged = d["labels"] if isinstance(d.get("labels"), dict) else dict(d)
            merged = {k: int(v) for k, v in merged.items() if v is not None}
        changed = 0
        for si, (ns, _orig) in revs.items():
            if merged.get(si) != ns:
                changed += 1
            merged[si] = ns
        planned_files[src_batch] = (path, merged, fn)
        print(f"  amendment {fn}: {len(revs)} revised ({changed} new/changed), "
              f"{len(merged)} total after merge")

    if not a.apply:
        print("  DRY RUN — pass --apply to write (nothing written).")
        return

    for src_batch, (path, merged, fn) in planned_files.items():
        _atomic_write_json(merged, path)
        print(f"  WROTE {path}  ({len(merged)} labels)")
    if fresh_filled:
        cc.write_jsonl(rows, images_path)
        print(f"  WROTE {images_path}  ({fresh_filled} fresh in-row labels)")

    if planned_files:
        print("\n  Register these in tools/corpus/label_store.AMENDMENT_LABELS "
              "(the file now exists on disk):")
        for src_batch, (_p, _m, fn) in sorted(planned_files.items()):
            print(f'      "{src_batch}": "{fn}",')


if __name__ == "__main__":
    main()
