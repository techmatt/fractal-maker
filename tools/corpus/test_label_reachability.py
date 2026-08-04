"""Every human label on disk must stay reachable through the canonical reader.

The label store's sidecar path depends on a registry (`label_store.SIDECAR_LABELS`)
that has silently gone incomplete TWICE: a batch labeled after the registry was last
touched went unregistered, its labels were never merged in-row, and it resolved to
ZERO through `label_store.resolve_score` — silently dropping the whole batch from any
retrain. This test fails the instant a `labels/*.json` sidecar carries labels that the
canonical resolver cannot reach.

The invariant is BEHAVIORAL, not registry membership (do NOT rewrite this as "every
file appears in SIDECAR_LABELS" — that false-fires on the in-row-merged batches):

  For every sidecar on disk whose keys are label_corpus `image_id`s, every one of
  those labels must resolve non-null through the SAME path the reader uses —
  `resolve_score(row, sidecar_for(batch_id))` = merged `label.score` ELSE the
  REGISTERED sidecar join. A batch merged in-row (blindspot, prospect) reconciles
  because its labels live in `label.score`; a registered sidecar-only batch
  (jm3/jm45/mining/scale/julia_ladder_j0) reconciles because `sidecar_for()` returns
  its map. The failure this catches is an unregistered, unmerged sidecar — reachable
  count 0 while the file holds N.

Out-of-scope sidecars come in two forms, both kept OUT of the v7 reachability assert:
  * By REGISTRATION — a file that belongs to a different corpus with different label
    SEMANTICS (`label_store.FOREIGN_LABEL_FILES`): the q4 WINDOW store's
    `q4_g_aimed.json` / `q4_stage1_windows{,_p2}.json`, whose values are three-way STRING
    classes (accept/reject/filter_leak), not int scores. The integer reader would crash on
    `int('reject')`, so these are skipped by registration and asserted per corpus below
    (`test_q4_window_store_reachable`) — never int-coerced here.
  * By CONTENT — 0 keys match any label_corpus `image_id`: the legacy
    `location_labels.json` (composite `idx|framing|palette` keys, labels live in-store),
    `palette_scores.json`, the wallpaper_corpus sidecars, and the render-mode head sidecars.
    Asserting these against this reader would false-fire on stores it never reads.

Reconciliation is on COUNT reachability, NOT score identity: 616 `image_id`s collide
across sibling batches (e.g. loose0 vs rev4 share `0_center_...`), so a colliding key
can legitimately carry a different in-row score in another batch. The drop we guard is
"label vanished to null", which is a count invariant.

Run either way:
  uv run pytest tools/corpus/test_label_reachability.py
  uv run python tools/corpus/test_label_reachability.py     # prints the reconciliation table
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # tools/corpus

import corpus_common as cc  # noqa: E402
import label_store as ls  # noqa: E402


def _scan_corpus():
    """Walk every label_corpus batch ONCE, mirroring corpus_reader.iter_labeled's
    per-row resolution (same ls.resolve_score, same ls.sidecar_for registry lookup).

    Returns (owners, reachable):
      owners    : {image_id: set(batch_id)}  — every id on disk, labeled or not.
      reachable : {image_id: int score}      — ids the CANONICAL path resolves non-null.
    `reachable` prefers a non-null hit, so a colliding id labeled in one batch and null
    in another still counts as reachable (the store holds it somewhere)."""
    owners: dict[str, set] = {}
    reachable: dict[str, int] = {}
    for images_path in sorted(glob.glob(os.path.join(cc.BATCHES_DIR, "*", "images.jsonl"))):
        batch_id = os.path.basename(os.path.dirname(images_path))
        sidecar = ls.sidecar_for(batch_id)
        with open(images_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = row["image_id"]
                owners.setdefault(iid, set()).add(batch_id)
                sc = ls.resolve_score(row, sidecar)
                if sc is not None and iid not in reachable:
                    reachable[iid] = int(sc)
    return owners, reachable


def reconcile():
    """Per-sidecar reconciliation table. For each labels/*.json:
      {sidecar: {"disk": N, "in_scope": M, "reachable": R, "out_of_scope": bool}}
    In-scope = keys that are a label_corpus image_id; the store's reachability
    invariant is reachable == in_scope for every in-scope sidecar."""
    owners, reachable = _scan_corpus()
    table = {}
    for fn in sorted(os.listdir(ls.LABELS_DIR)):
        if not fn.endswith(".json"):
            continue
        if not ls.is_v7_corpus_label_file(fn):             # a DIFFERENT corpus (q4 window,
            continue                                        # string classes) — asserted per
                                                            # corpus in its own test, not here.
        labels = ls.load_sidecar(fn)                       # {image_id: int}, nulls dropped
        in_scope = [k for k in labels if k in owners]
        reach = sum(1 for k in in_scope if k in reachable)
        table[fn] = {
            "disk": len(labels),
            "in_scope": len(in_scope),
            "reachable": reach,
            "out_of_scope": len(in_scope) == 0,
        }
    return owners, table


def test_every_label_corpus_sidecar_is_fully_reachable():
    """Every sidecar keyed on label_corpus image_ids resolves ALL its labels through
    the canonical reader. A sidecar-only batch that went unregistered (and was never
    merged in-row) reads as reachable=0 here and fails, naming the batch."""
    owners, table = reconcile()
    assert owners, "no label_corpus image_ids scanned — reader/glob broke, test is vacuous"

    in_scope_files = [fn for fn, r in table.items() if not r["out_of_scope"]]
    # Guard against a silently vacuous pass: the registered sidecar-only batches MUST
    # be present and in-scope, else there is nothing meaningful to reconcile.
    assert in_scope_files, "no in-scope sidecars found — the reachability check ran on nothing"

    unreachable = {
        fn: r for fn, r in table.items()
        if not r["out_of_scope"] and r["reachable"] != r["in_scope"]
    }
    assert not unreachable, (
        "label store UNREACHABLE through resolve_score — labels present on disk that the "
        "canonical reader drops to zero (likely an unregistered sidecar-only batch; "
        "register it in tools/corpus/label_store.SIDECAR_LABELS or merge it in-row):\n  "
        + "\n  ".join(
            f"{fn}: {r['reachable']}/{r['in_scope']} in-scope labels reachable "
            f"({r['disk']} on disk)"
            for fn, r in sorted(unreachable.items())
        )
    )


def _q4_foreign_files():
    """The `labels/*.json` files `label_store` hands to the q4-window reader."""
    return {fn: reader for fn, reader in ls.FOREIGN_LABEL_FILES.items()
            if "q4_window_reader" in reader}


def _q4_labels(fn):
    """One q4-window export as {window_id: class}, nulls dropped."""
    raw = json.loads((open(os.path.join(ls.LABELS_DIR, fn), encoding="utf-8")).read())
    body = raw["labels"] if isinstance(raw.get("labels"), dict) else raw
    return {k: v for k, v in body.items() if v is not None}


def test_q4_window_store_keys_are_window_ids():
    """Every key of a registered q4-window export is a `window_id` of a registered batch.

    The weaker half of the window-store guard: it catches an ORPHANED export (keys that
    match no window at all) and keeps the two-corpus split honest — a file gets skipped by
    the v7 integer reader ONLY because it genuinely belongs here. It does NOT prove the
    reader returns those labels; `test_q4_window_labels_resolve` is what does."""
    sys.path.insert(0, os.path.join(HERE))
    import q4_window_reader as q4  # noqa: E402

    foreign = _q4_foreign_files()
    assert foreign, "no q4-window FOREIGN_LABEL_FILES registered — this guard is vacuous"

    window_ids = set()
    for bid in q4.REGISTERED_BATCHES:
        with open(q4.batch_dir(bid) / "windows.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    window_ids.add(json.loads(line)["window_id"])
    assert window_ids, "no q4-window window_ids scanned — window reader/glob broke"

    unreachable = {}
    for fn in foreign:
        keys = list(_q4_labels(fn))
        reach = sum(1 for k in keys if k in window_ids)
        if reach != len(keys):
            unreachable[fn] = (reach, len(keys))
    assert not unreachable, (
        "q4-window label file(s) with keys that are not a window_id in any "
        "REGISTERED_BATCHES batch (orphaned export, or the batch went unregistered in "
        "q4_window_reader.REGISTERED_BATCHES):\n  "
        + "\n  ".join(f"{fn}: {r}/{n} keys reach a window" for fn, (r, n) in sorted(unreachable.items())))


def test_q4_window_labels_resolve():
    """Every label in a registered q4-window export RESOLVES through `resolve_klass`.

    This is the real mirror of the v7 guard, and the version of it that would have caught
    the bug it exists for. The key-validity check above passed for two years' worth of
    exports that the reader could not actually read: `load_scores_sidecar` looked only at
    `<batch_dir>/scores.json`, so `q4_stage1_windows_p2.json` (229) and `q4_g_aimed.json`
    (112) resolved to None with every key perfectly valid `[found 2026-08-04]`. A file
    reachable by KEY but not by VALUE is exactly the "labels present, reader reaches zero"
    failure the v7 guard names, so it is asserted the same way — through the resolver, not
    through the registry."""
    sys.path.insert(0, os.path.join(HERE))
    import q4_window_reader as q4  # noqa: E402

    foreign = _q4_foreign_files()
    assert foreign, "no q4-window FOREIGN_LABEL_FILES registered — this guard is vacuous"

    resolved = {}
    for bid in q4.REGISTERED_BATCHES:
        for row, klass in q4.iter_windows(bid):
            if klass is not None:
                resolved[row["window_id"]] = klass

    unreachable = {}
    for fn in foreign:
        keys = list(_q4_labels(fn))
        reach = sum(1 for k in keys if k in resolved)
        if reach != len(keys):
            unreachable[fn] = (reach, len(keys))
    assert not unreachable, (
        "q4-window label(s) UNREACHABLE through q4_window_reader.resolve_klass — present on "
        "disk, dropped to None by the reader. Register the export in "
        "q4_window_reader.SIDECAR_FILES (batch -> files, oldest first):\n  "
        + "\n  ".join(f"{fn}: {r}/{n} labels resolve" for fn, (r, n) in sorted(unreachable.items())))


def test_q4_sidecar_registries_agree():
    """`label_store.FOREIGN_LABEL_FILES` and `q4_window_reader.SIDECAR_FILES` name the same
    files. The first says "this file belongs to the window reader"; the second is what that
    reader actually opens. When only the first knew about a file, the redirect pointed at a
    reader that then ignored it — a dead end that reads exactly like a registered file."""
    sys.path.insert(0, os.path.join(HERE))
    import q4_window_reader as q4  # noqa: E402

    redirected = set(_q4_foreign_files())
    opened = {fn for files in q4.SIDECAR_FILES.values() for fn in files}
    assert redirected == opened, (
        "the two q4-window registries disagree.\n"
        f"  redirected here by label_store but never opened: {sorted(redirected - opened)}\n"
        f"  opened by the reader but not redirected here:    {sorted(opened - redirected)}")

    unknown = sorted(set(q4.SIDECAR_FILES) - set(q4.REGISTERED_BATCHES))
    assert not unknown, f"SIDECAR_FILES names unregistered batch(es): {unknown}"


def main():
    owners, table = reconcile()
    print("=== labels/ sidecar reachability (disk / in-scope / reachable) ===")
    for fn in sorted(ls.FOREIGN_LABEL_FILES):
        print(f"  {fn:<40}  FOREIGN ({ls.FOREIGN_LABEL_FILES[fn]}) - asserted per corpus")
    width = max(len(fn) for fn in table)
    bad = 0
    for fn, r in table.items():
        if r["out_of_scope"]:
            print(f"  {fn:<{width}}  disk={r['disk']:<5}  OUT-OF-SCOPE "
                  f"(0 keys match any label_corpus image_id)")
            continue
        ok = r["reachable"] == r["in_scope"]
        bad += not ok
        flag = "OK " if ok else "!! "
        print(f"  {flag}{fn:<{width}}  disk={r['disk']:<5} in_scope={r['in_scope']:<5} "
              f"reachable={r['reachable']}")
    print(f"\n{len([r for r in table.values() if not r['out_of_scope']])} in-scope sidecars, "
          f"{bad} unreachable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
