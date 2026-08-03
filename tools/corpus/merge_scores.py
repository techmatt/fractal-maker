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

ROUTED merges (`--route`)
-------------------------
A single labeling sitting may be served as ONE combined sheet over several registered
batches (a PRESENTATION merge — the batch registrations themselves never move; see
tools/corpus/build_combined_label_sheet.py). Its export is keyed by the sheet's OPAQUE
presentation id, which belongs to no batch, so `--batch` cannot place those rows.

`--route <route.json>` supplies the sheet's `opaque_id -> {batch, image_id}` map (written
at sheet-build time, never served to the browser). Each score is then merged into the
images.jsonl of ITS OWN registered source batch, with the identical null->value rule and
per-batch reporting. Nothing else changes: the same refusal on a non-null change, the same
untouched amendment stream, one images.jsonl write per batch that actually gained a label.

  uv run python tools/corpus/merge_scores.py --max-score 4 \
      --route data/label_corpus/batches/<sheet_id>/route.json \
      --scores <the sitting's labels.json> [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

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


def load_route(path: str) -> dict:
    """Read a combined sheet's routing map into `{opaque_id: (batch_id, image_id)}`.

    The map is the ONLY thing that turns a sheet-keyed export back into per-batch labels,
    so it is validated strictly: every entry must name both a batch and an image_id.
    """
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if not isinstance(v, dict) or not v.get("batch") or not v.get("image_id"):
            raise SystemExit(f"route entry {k!r} must carry both 'batch' and 'image_id': {v!r}")
        out[k] = (v["batch"], v["image_id"])
    return out


def merge_batch(batch, corpus_root, scores, *, labeler, labeled_at, max_score, apply):
    """Fill `null -> value` for one batch from `{image_id: score}`; return a stats dict.

    The ONE allowed store mutation, and the refusal that guards it, live here — both the
    single-batch path and the routed path go through this function so they cannot drift.
    Writes at most one images.jsonl, and only when `apply` and something actually filled.
    """
    images_path = os.path.join(corpus_root, batch, "images.jsonl")
    rows = cc.read_jsonl(images_path)
    by_id = {r["image_id"]: r for r in rows}
    st = dict(batch=batch, n_rows=len(rows), n_scores=len(scores), filled=0, reaffirmed=0,
              conflicts=[], unknown=[], out_of_range=[], images_path=images_path, wrote=False)

    for image_id, new_score in scores.items():
        if new_score is None:
            continue
        if not (1 <= new_score <= max_score):
            st["out_of_range"].append((image_id, new_score))
            continue
        row = by_id.get(image_id)
        if row is None:
            st["unknown"].append(image_id)
            continue
        cur = row["label"]["score"]
        if cur is None:
            row["label"]["score"] = new_score
            row["label"]["labeler"] = labeler
            row["label"]["labeled_at"] = labeled_at
            st["filled"] += 1
        elif cur == new_score:
            st["reaffirmed"] += 1
        else:
            st["conflicts"].append((image_id, cur, new_score))

    if apply and st["filled"]:
        cc.write_jsonl(rows, images_path)
        st["wrote"] = True
    st["labeled"] = sum(1 for r in rows if r["label"]["score"] is not None)
    return st


def report(st, max_score: int) -> None:
    print(f"batch {st['batch']}: {st['n_rows']} rows, {st['n_scores']} scores in export "
          f"(valid tiers 1..{max_score})")
    print(f"  null -> value (fill): {st['filled']}")
    print(f"  already == score (no-op): {st['reaffirmed']}")
    if st["out_of_range"]:
        print(f"  WARNING: {len(st['out_of_range'])} scores out of range 1..{max_score} "
              f"(skipped), e.g. {st['out_of_range'][:3]}")
    if st["unknown"]:
        print(f"  WARNING: {len(st['unknown'])} scores reference unknown image_id (skipped), "
              f"e.g. {st['unknown'][:3]}")
    if st["conflicts"]:
        print(f"  REFUSED: {len(st['conflicts'])} would CHANGE a non-null label - NOT applied:")
        for image_id, cur, new in st["conflicts"][:20]:
            print(f"    {image_id}: existing {cur} != export {new}")
        if len(st["conflicts"]) > 20:
            print(f"    ... and {len(st['conflicts']) - 20} more")
    if st["wrote"]:
        print(f"  WROTE {st['images_path']}: {st['labeled']}/{st['n_rows']} now labeled "
              f"({len(st['conflicts'])} conflicts left untouched)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="batch_id under the corpus root (omit with --route)")
    ap.add_argument("--route", default=None,
                    help="combined-sheet routing map (opaque_id -> {batch,image_id}); "
                         "routes ONE export back to each row's registered source batch")
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

    if bool(a.batch) == bool(a.route):
        raise SystemExit("pass exactly one of --batch (single batch) or --route (combined sheet)")

    corpus_root = a.corpus_root or cc.BATCHES_DIR
    kw = dict(labeler=a.labeler, labeled_at=a.labeled_at,
              max_score=a.max_score, apply=a.apply)

    if a.batch:
        scores_path = a.scores or os.path.join(corpus_root, a.batch, "scores.json")
        report(merge_batch(a.batch, corpus_root, load_scores(scores_path), **kw), a.max_score)
    else:
        if not a.scores:
            raise SystemExit("--route needs --scores (the sitting's single labels.json)")
        route = load_route(a.route)
        export = load_scores(a.scores)
        # Split the one export by the batch each opaque id belongs to. An id the map does
        # not know is UNROUTABLE: it belongs to no batch, so it is reported, never guessed at.
        per_batch, unroutable = defaultdict(dict), []
        for opaque, score in export.items():
            if opaque not in route:
                unroutable.append(opaque)
                continue
            batch, image_id = route[opaque]
            per_batch[batch][image_id] = score
        print(f"routed {len(export) - len(unroutable)}/{len(export)} exported scores over "
              f"{len(per_batch)} batches (route map: {len(route)} ids)")
        if unroutable:
            print(f"  WARNING: {len(unroutable)} exported ids are NOT in the route map "
                  f"(skipped), e.g. {unroutable[:3]}")
        for batch in sorted(per_batch):
            report(merge_batch(batch, corpus_root, per_batch[batch], **kw), a.max_score)

    if not a.apply:
        print("  DRY RUN - pass --apply to write (conflicts are never written either way)")


if __name__ == "__main__":
    main()
