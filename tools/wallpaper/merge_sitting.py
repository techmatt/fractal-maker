r"""merge_sitting.py — a labeling sitting's export -> the batch's tracked labels sidecar.

The wallpaper corpus keeps its human labels in `labels/<generator_version>.json`, a flat
`{image_id: 1..4}` map, NOT in-row (`images.jsonl` keeps `label.score` null; the trainer joins
the sidecar — `train_wallpaper_v3.SOURCES`). The three July batches were placed by hand. This
does it with the checks, because the sidecar is the one artifact in the whole pipeline with no
rebuild path: crops regenerate from the render block, scores do not regenerate from anything.

WHAT IS VERIFIED, AND WHY EACH ONE (all counted from the batch manifest, never from a guard's
say-so — `--verify` re-reads both files and recounts):

  * every exported id EXISTS in the batch's images.jsonl. A key that matches nothing is a
    sitting served from a different batch, and it would land in the sidecar as a label for a
    render nobody can find.
  * every value is an INT in 1..4. The UI cannot emit anything else; a file hand-edited
    between export and merge can.
  * the batch's own row count is reported against the labeled count, so an unlabeled
    remainder is a number in the output rather than an absence nobody looked for.
  * a label that would CHANGE an existing non-null sidecar entry refuses. Same rule as
    `merge_scores.py`: `null -> value` is the one allowed mutation to an original label, and a
    revision belongs in the amendment stream, not in a silent overwrite.
  * `suggested_tier` is never consulted. A correction sheet's machine suggestion is not a
    label and must not reach the sidecar; only what the human exported does.

  uv run python tools/wallpaper/merge_sitting.py \
      --batch 2026-08-05_wallpaper_fresh_sheet_v1 \
      --scores labels/scores_2026-08-05_wallpaper_fresh_sheet_v1.json          # dry run
  uv run python tools/wallpaper/merge_sitting.py --batch ... --scores ... --apply
  uv run python tools/wallpaper/merge_sitting.py --batch ... --verify          # recount only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ROOT / "data" / "wallpaper_corpus" / "batches"
LABELS = ROOT / "labels"
MAX_SCORE = 4


def _rows(batch: str) -> list:
    p = BATCHES / batch / "images.jsonl"
    if not p.exists():
        raise SystemExit(f"[merge] no such batch manifest: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sidecar_for(batch: str) -> Path:
    """`labels/<generator_version>.json` — the name the TRAINER joins on, read out of the
    batch's own manifest rather than derived from the directory name (the two differ: the dir
    carries a date prefix, the sidecar does not)."""
    gv = json.loads((BATCHES / batch / "batch.json").read_text(encoding="utf-8"))["generator_version"]
    return LABELS / f"{gv}.json"


def merge(batch: str, scores_path: Path, apply: bool) -> dict:
    rows = _rows(batch)
    ids = {r["image_id"] for r in rows}
    exported = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    side = sidecar_for(batch)
    existing = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}

    unknown = sorted(k for k in exported if k not in ids)
    bad = {k: v for k, v in exported.items()
           if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= MAX_SCORE)}
    conflicts = {k: (existing[k], v) for k, v in exported.items()
                 if k in existing and existing[k] != v}
    if unknown:
        raise SystemExit(f"[merge] {len(unknown)} exported id(s) are not in {batch}, e.g. "
                         f"{unknown[:5]} — this export belongs to another batch.")
    if bad:
        raise SystemExit(f"[merge] {len(bad)} value(s) outside 1..{MAX_SCORE}: {list(bad.items())[:5]}")
    if conflicts:
        raise SystemExit(f"[merge] REFUSING: {len(conflicts)} label(s) would change an existing "
                         f"non-null score, e.g. {list(conflicts.items())[:5]}. A revision goes to "
                         f"the amendment stream, never over the original.")

    merged = dict(existing)
    merged.update({k: int(v) for k, v in exported.items()})
    new = len(merged) - len(existing)
    if apply:
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps({k: merged[k] for k in sorted(merged)}, separators=(",", ":")),
                        encoding="utf-8")

    return {
        "batch": batch, "sidecar": str(side.relative_to(ROOT)).replace("\\", "/"),
        "manifest_rows": len(rows), "exported": len(exported),
        "already_in_sidecar": len(existing), "newly_written": new,
        "labeled_after": len(merged), "unlabeled_remainder": len(ids - set(merged)),
        "distribution": dict(sorted(Counter(merged.values()).items())),
        "applied": bool(apply),
    }


def verify(batch: str) -> dict:
    """Recount from the two files on disk. Counts rows; trusts nothing that ran earlier."""
    rows = _rows(batch)
    ids = {r["image_id"] for r in rows}
    side = sidecar_for(batch)
    have = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    covered = ids & set(have)
    return {
        "batch": batch, "sidecar": str(side.relative_to(ROOT)).replace("\\", "/"),
        "sidecar_exists": side.exists(),
        "manifest_rows": len(rows), "sidecar_keys": len(have),
        "manifest_rows_labeled": len(covered),
        "unlabeled_remainder": len(ids - set(have)),
        "sidecar_keys_not_in_manifest": len(set(have) - ids),
        "distribution": dict(sorted(Counter(have[k] for k in covered).items())),
        "complete": len(covered) == len(ids) and not (set(have) - ids),
    }


def main():
    ap = argparse.ArgumentParser(description="Merge a wallpaper sitting export into its sidecar.")
    ap.add_argument("--batch", required=True, help="dir name under data/wallpaper_corpus/batches/")
    ap.add_argument("--scores", type=Path, help="the exported scores json (omit with --verify)")
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    ap.add_argument("--verify", action="store_true", help="recount from disk and exit")
    args = ap.parse_args()

    if args.verify:
        print(json.dumps(verify(args.batch), indent=2))
        return
    if not args.scores:
        ap.error("--scores is required unless --verify")
    rep = merge(args.batch, args.scores, args.apply)
    print(json.dumps(rep, indent=2))
    if not args.apply:
        print("\n[dry run] nothing written — re-run with --apply")


if __name__ == "__main__":
    main()
