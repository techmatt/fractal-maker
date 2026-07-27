"""Part 4 — merge a harness `scores.json` export into a batch's images.jsonl.

The ONE allowed mutation in the store is a label's score going `null -> value`.
This merger enforces that: it fills `label.score` for rows whose score is
currently null, and **warns and refuses** (never silently clobbers) when a
scores.json entry would change an already-non-null label to a different value.
Re-applying the same score is a no-op.

The label scale is 1..4 (bad/okay/good/exceptional); class 4 is a fourth tier on the SAME
quality scale, not a separate head. This merges NEW labels in-row (null -> value). REVISIONS
to already-labeled rows do NOT go through here — they go to the amendment stream via
tools/corpus/merge_amendments.py, which never mutates the original label.

Run (location corpus, 4-tier — default):
  uv run python tools/corpus/merge_scores.py \
      --batch 2026-07-26_minibrot_roster_v2 \
      [--scores <path>]  [--labeler matt] [--labeled-at 2026-07-26] [--apply]

Without --apply it's a dry run (reports what would change, writes nothing).
"""
from __future__ import annotations

import argparse
import json
import os

import corpus_common as cc


def load_scores(path: str) -> dict:
    """Read a label export into `{image_id: score|None}`, accepting BOTH shapes:

      * legacy pair — `scores.json` is `{id: int}` (its reveal flags live in a
        separate `reveals.json` sidecar, which the merge never consumed);
      * new combined — `{id: {"score": int, "revealed": 0|1}}`, one row's score and
        its reveal flag in a single object.

    Only the score is merged into the store either way (a label's reveal flag is an
    audit sidecar, never a store field — see the schema's `label` block). No
    migration: old files stay as-is, this loader reads whichever shape it's given.
    """
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):          # combined form: pull the score out
            v = v.get("score")
        out[k] = int(v) if v is not None else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="batch_id under the corpus root")
    ap.add_argument("--corpus-root", default=None,
                    help="batches dir the --batch lives under "
                         "(default: data/label_corpus/batches — the location corpus)")
    ap.add_argument("--max-score", type=int, default=4,
                    help="max valid ordinal tier (label scale is 1..4: bad/okay/good/exceptional)")
    ap.add_argument("--scores", default=None, help="scores.json (default: <batch>/scores.json)")
    ap.add_argument("--labeler", default="matt")
    ap.add_argument("--labeled-at", default=None)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    a = ap.parse_args()

    corpus_root = a.corpus_root or cc.BATCHES_DIR
    bdir = os.path.join(corpus_root, a.batch)
    images_path = os.path.join(bdir, "images.jsonl")
    scores_path = a.scores or os.path.join(bdir, "scores.json")

    rows = cc.read_jsonl(images_path)
    scores = load_scores(scores_path)

    filled, reaffirmed, conflicts, unknown, out_of_range = 0, 0, [], [], []
    by_id = {r["image_id"]: r for r in rows}

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
        cur = row["label"]["score"]
        if cur is None:
            row["label"]["score"] = new_score
            row["label"]["labeler"] = a.labeler
            row["label"]["labeled_at"] = a.labeled_at
            filled += 1
        elif cur == new_score:
            reaffirmed += 1
        else:
            conflicts.append((image_id, cur, new_score))

    print(f"batch {a.batch}: {len(rows)} rows, {len(scores)} scores in export "
          f"(valid tiers 1..{a.max_score})")
    print(f"  null -> value (fill): {filled}")
    print(f"  already == score (no-op): {reaffirmed}")
    if out_of_range:
        print(f"  WARNING: {len(out_of_range)} scores out of range 1..{a.max_score} (skipped), "
              f"e.g. {out_of_range[:3]}")
    if unknown:
        print(f"  WARNING: {len(unknown)} scores reference unknown image_id (skipped), e.g. {unknown[:3]}")
    if conflicts:
        print(f"  REFUSED: {len(conflicts)} would CHANGE a non-null label - NOT applied:")
        for image_id, cur, new in conflicts[:20]:
            print(f"    {image_id}: existing {cur} != export {new}")
        if len(conflicts) > 20:
            print(f"    ... and {len(conflicts) - 20} more")

    if not a.apply:
        print("  DRY RUN - pass --apply to write (conflicts are never written either way)")
        return

    cc.write_jsonl(rows, images_path)
    labeled = sum(1 for r in rows if r["label"]["score"] is not None)
    print(f"  WROTE {images_path}: {labeled}/{len(rows)} now labeled "
          f"({len(conflicts)} conflicts left untouched)")


if __name__ == "__main__":
    main()
