#!/usr/bin/env python3
"""Repo-size guard — the "stays small" constraint, as an enforced registry.

Sibling to `disk_audit.py`. Where that tool classifies data artifacts by
DELETION-SAFETY, this one enforces a different invariant: **the working tree
should stay ~what git tracks — source + irreplaceable metadata + `scratch/`.** Nothing
large lives in-tree without an explicit, written-down reason.

Two independent things live here:

  1. A WORKING-TREE size SCAN (`scan`). Walks the filesystem — not `git ls-files`,
     because a gitignored file can bloat the tree while invisible to git, which is
     the whole point. Flags:
       (a) any FILE >= FILE_THRESHOLD (1 MiB — the same "large" as the index-side
           policy guard, tests/test_large_tracked_blobs.py), and
       (b) any DIRECTORY whose aggregate of SMALL files (< FILE_THRESHOLD) in its
           subtree >= DIR_THRESHOLD (~100 MB), reported at MINIMAL granularity (the
           leaf-most such dir). Rule (b) is deliberately keyed on small files only:
           big files are already caught one-by-one by rule (a), so keying (b) on the
           full aggregate would just re-flag every ancestor of a big file and force
           coarse mixed-disposition registry entries. Small-files-only isolates the
           many-small-file case (label crops, field caches) that no single-file rule
           can see.
       (c) any DIRECTORY whose subtree holds >= DIR_FILE_COUNT_THRESHOLD FILES or
           >= DIR_BYTES_THRESHOLD BYTES (all files, large ones included), again at
           MINIMAL granularity. This is the BULK rule — the enforcement arm of
           storage_classes.md rule 5, "no large training data or results in-tree, even
           if gitignored". Rules (a)/(b) do not cover it: (a) is per-file, so a cache
           of a million 4 KB crops passes cleanly no matter how big the tree gets, and
           (b) caps out at 100 MB of *small* files, which says nothing about file
           COUNT — and traversal cost, the actual harm, is driven by count. The two
           historical bombs (243k aug-cache JPGs, 317k discovery-scratch files) are
           exactly what (c) exists to refuse a second time.
     Excludes {scratch/, .venv/, target/, target-test/, .git/} from flagging. `.git` is
     a history-REWRITE target (git filter-repo), not a relocation one — its size is
     reported as an FYI line, never flagged.

  2. The REGISTRY (`REGISTRY`) — the deliverable. One explicit allowlist, same
     spirit as `tests/test_tracked_artifacts.py`'s `TRACKED_CANARIES`: the
     sanctioned-large-in-tree set. Every current violator is covered by exactly one
     entry, at a stable path-prefix granularity (so intra-dir churn — a new batch, a
     new crop — can't flake the guard). Each entry records size class, tracked-ness,
     and a DISPOSITION:
       KEEP     — legitimately stays in-tree (irreplaceable tracked metadata with no
                  smaller form). Being tracked is NOT an automatic pass; the reason
                  is the written-down "extremely good reason".
       RELOCATE -> <tier> — pending a move; delete the line when the move lands.
                  Tiers are disposition LABELS only (no dirs are created / paths
                  wired here — the precious-store *location* is still undecided):
                    artifacts      regenerable bulk (rebuildable render/cache output)
                    precious-store irreplaceable binaries (trained .pt weights)
                    trash          dead / superseded

     Orthogonal to the disposition, an entry may be marked `forward=True` — a LIVE
     FORWARD DECLARATION. Nothing over-threshold is there now, but a committed writer
     can still put it there, and the line is the disposition that write lands under.
     The test is *can anything still write here*, and the costs are lopsided: a kept
     dead line costs one config line; a pruned live one costs a red build at the worst
     moment, with the disposition re-decided under time pressure.

The guard test (`tests/test_repo_size_guard.py`) fails on any flagged violator not
covered by a registry entry (new bloat caught from today), and ALSO fails on any entry
that has no over-threshold content and is not marked `forward` — i.e. a line nobody
classified. Both are actionable in one edit, so neither is a standing warning. As things
relocate, their RELOCATE lines come out; when only KEEP lines remain, every in-tree
exception is explicit and reviewed.

Relationship to the 20 MB per-commit rule (CLAUDE.md). That rule counts **TREE BYTES** —
the working-tree size of what gets tracked, LFS files at full content size, not the remote
or packed size (settled 2026-08-07). This scan measures the same unit, which is why the two
are commensurable: the rule is the per-commit stop-and-ask, this is the standing check on
the accumulated tree. Neither is the index-side policy guard — that is
`tests/test_large_tracked_blobs.py` (what may be tracked at all, and what may never be).

This module MOVES / DELETES / COMMITS NOTHING. It scans and reports.

Usage
-----
  uv run python tools/audit/size_guard.py            # full report (inventory + guard status)
  uv run python tools/audit/size_guard.py --check    # terse pass/fail (what the test asserts)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds (constants — tune if the report is noisy or sparse).
# ---------------------------------------------------------------------------
FILE_THRESHOLD = 1 * 1024 * 1024        # 1 MiB — same LARGE as test_large_tracked_blobs.py
DIR_THRESHOLD = 100 * 1024 * 1024       # ~100 MB — many-small-file catch

# --- BULK rule (c) thresholds: storage_classes.md rule 5, "no bulk in-tree" ---
# Chosen against the measured shape of THIS tree, not picked round:
#   * FILE COUNT 2,000. Sized against the tree as it was when the rule landed: ~5.2k
#     non-excluded files, so a single 2k subtree was already ~40% of everything a
#     recursive walk had to visit. (Re-measured 2026-07-31: 1,638 non-excluded files
#     against 1,334 tracked — the relocations landed, so 2,000 is now MORE than the whole
#     tree and the rule fires only on a genuine bomb. Dated rather than silently updated:
#     a threshold's rationale is a record of what was true when it was chosen.) It is two
#     orders of magnitude below the bombs that made `grep -r` take >120s (152k / 42k / 27k
#     / 22k aug-cache dirs; 317k discovery scratch) — low enough to fire long before the
#     tree is unusable, high enough that no source or metadata directory comes close.
#   * BYTES 500 MB. Rule (a) already catches single blobs at 1 MiB and rule (b) catches
#     small-file aggregates at 100 MB, so this is not a lower bound on "large" — it is
#     the point at which a directory is unambiguously a DATA STORE rather than source
#     plus metadata. Set above the ~250 MB scale of one label batch on purpose: a
#     per-batch threshold would flap as crops churn inside a directory whose disposition
#     is already registered at the parent prefix.
# Raising either is a policy change; say so in the commit rather than nudging it to make
# a red build green.
DIR_FILE_COUNT_THRESHOLD = 2_000        # files in a subtree — traversal cost
DIR_BYTES_THRESHOLD = 500 * 1024 * 1024  # ~500 MB in a subtree (all files)

# Excluded from FLAGGING (working-tree churn that is regenerable infra, not data
# bloat we relocate). `.git` is a history-rewrite target -> FYI size only.
EXCLUDE_PREFIXES = ("scratch", ".venv", "target", "target-test", ".git", ".pytest_cache")
GIT_DIR = ".git"

# ---------------------------------------------------------------------------
# Dispositions
# ---------------------------------------------------------------------------
KEEP = "KEEP"
RELOCATE = "RELOCATE"
ARTIFACTS = "artifacts"          # regenerable bulk
PRECIOUS = "precious-store"      # irreplaceable binaries (trained weights)
TRASH = "trash"                  # dead / superseded

# Report ordering for disposition groups.
GROUP_ORDER = [
    ("KEEP", None),
    ("RELOCATE", ARTIFACTS),
    ("RELOCATE", PRECIOUS),
    ("RELOCATE", TRASH),
]


@dataclass
class Entry:
    """One registry line. `prefix` is a repo-relative POSIX path; dir prefixes end
    with '/'. A violator is COVERED by this entry iff its path == prefix or starts
    with prefix (dir violators carry a trailing '/', so prefix matching is exact at
    the path-segment boundary).

    `forward` marks a LIVE FORWARD DECLARATION: nothing over-threshold is there right
    now, but a committed writer can still put it there, and this line is the disposition
    that write lands under. It is the answer to a question emptiness cannot answer —
    not-yet-built vs dead — and it is why the stale report can be a hard assertion
    instead of a standing warning. Every entry must be one or the other."""
    prefix: str
    disposition: str                 # KEEP | RELOCATE
    tier: str | None                 # None for KEEP; artifacts|precious-store|trash
    tracked: str                     # 'tracked' | 'ignored' | 'mixed'
    reason: str
    canary: bool = False             # covers a canaried path (move needs a canary update)
    forward: bool = False            # live forward declaration (see docstring)

    def label(self) -> str:
        return self.disposition if self.disposition == KEEP else f"{RELOCATE} -> {self.tier}"


# ---------------------------------------------------------------------------
# THE REGISTRY — every current over-threshold violator, one covering line each.
# Stable path-prefix granularity: intra-dir churn (a new batch / crop / field) is
# absorbed by the prefix and never flakes the guard.
# ---------------------------------------------------------------------------
REGISTRY: list[Entry] = [
    # === KEEP — irreplaceable tracked metadata, legitimately in-tree ===========
    Entry("data/palettes/", KEEP, None, "mixed",
          "committed palette definitions (harvested 746-palette pool + features); "
          "load-bearing config for the palette system, tracked, no smaller form"),
    Entry("data/v8/", KEEP, None, "tracked",
          "v8 classifier build: manifest.jsonl IS the training population + split, and it "
          "is what makes everything else here rebuildable — losing IT is exactly how the "
          "v4..v7 caches became 243k unattributable JPGs. plan/cache_manifest were "
          "DELETED 2026-08-03 (146 MB): unlike the manifest they are DERIVED, and "
          "tools/v8/build_plan.py reproduces both byte-identically from it (proved by "
          "rebuild + sha256). A rollback-to-v8 cache rebuild is now two steps, build_plan "
          "then render_cache. durable() + canaried; the aug_cache JPGs themselves are "
          "bulk() and out-of-tree. CANARY.", canary=True),
    Entry("data/v9/", KEEP, None, "tracked",
          "v9 classifier build — the v8 corpus re-rendered at the raised iteration cap "
          "(docs/design/auto_maxiter.md). What is left here is the RECORD: roster, "
          "colormaps, build_metadata, eval products. It has no manifest.jsonl of its own "
          "on purpose — it reads v8's and records that file's sha256, so 'same corpus' is "
          "a checked claim rather than a copy that can drift. plan.jsonl + "
          "cache_manifest.jsonl (146 MB) were DELETED 2026-08-08: byte-reproducible from "
          "v8's manifest by tools/v9/build_plan.py (proved by rebuild + sha256), and the "
          "v10 recipe-parity gate that was the reason to track them was retired the same "
          "day. The aug_cache tree itself is gone (12.09 GiB, 2026-08-08). FORWARD: nothing "
          "here is over threshold now, and the live writer is tools/v9/build_plan.py, which "
          "still writes the 146 MB pair back to data/v9/{plan,cache_manifest}.jsonl "
          "(bulk(), in-tree, untracked). This line is the disposition that write lands "
          "under.", forward=True),
    Entry("data/v11/", KEEP, None, "tracked",
          "v11 classifier build — ONE tracked row file, unlike v8/v9/v10. The manifest, "
          "plan and cache_manifest are bulk and out-of-tree (the split is a seeded "
          "randomized draw, so they rebuild from the committed corpus); what stays in-tree "
          "is the small record that makes them checkable (build_record / aug_recipe / "
          "prereg / eval_canon_record) plus eval_scores_v11.jsonl (1.4 MB, LFS) — the "
          "frozen eval slice tools/scoring/eval_slice resolves IN-TREE and a later cut "
          "re-cuts from without re-scoring.",),
    Entry("data/v10/", KEEP, None, "tracked",
          "v10 classifier build — v8's manifest APPENDED with 1,267 maneuver-view "
          "locations from the 2026-08 supply crawl and label-seeded harvest. Same shape "
          "and same reason as data/v8/ and data/v9/ above. What is over threshold here now "
          "is manifest.jsonl (2.2 MB) — and unlike v9 it HAS one, because the population "
          "moved: a third forced-eval instrument (maneuver_uniform_v1, 90 loc) joins the "
          "census and the mandelbrot floor, and a split assignment that exists only as a "
          "diff against another file is one nobody can read. plan.jsonl + "
          "cache_manifest.jsonl (179 MB) were DELETED 2026-08-08, last of the three: "
          "byte-reproducible from data/v10/manifest.jsonl plus v9's committed roster and "
          "colormaps by tools/v10/build_plan.py (proved by rebuild + sha256), and their "
          "reason to be tracked — the only map from a cached tile back to a location — went "
          "with data/v{9,10}/aug_cache (14.30 GiB) earlier the same day. FORWARD: the live "
          "writer is tools/v10/build_plan.py, which still writes the 179 MB pair back to "
          "data/v10/{plan,cache_manifest}.jsonl (bulk(), in-tree, untracked); this line is "
          "the disposition that write lands under. aug_roster.json and build_metadata.json "
          "stay tracked and are NOT reproducible — they hold retired GATE A's frozen counts. "
          "CANARY.", canary=True, forward=True),
    Entry("data/library_embeddings/", KEEP, None, "mixed",
          "prospect-library CLIP embeddings (embeddings.npz, tracked): unregenerable "
          "except value-approximate under a verdict-sensitive threshold. CANARY.",
          canary=True),
    Entry("data/sourcing/", KEEP, None, "tracked",
          "DISPOSITION: KEEP — Newton divergence-abort parity table. The violator is "
          "newton_parity_ref.json (1.87 MB, 31,616 rows of [converged, iters, digest16]): "
          "the pinned outcome of the PRE-ABORT reference solver over the two slow-lane "
          "roster grids. It is an INPUT to tools/sourcing/test_newton_divergence_abort.py, "
          "not an output of one. Two reasons it stays in-tree: (1) re-deriving it was "
          "80.6% of the entire `slow` lane (~55 min of the 61-min baseline, 2026-08-08) "
          "and the lane is 2.5 min with it; (2) what it pins is a MATHEMATICAL CONSTANT — "
          "the reference is a reimplementation of code deleted when the abort landed, so "
          "for a fixed (seed, period, degree, max_steps, dps) its outcome can never "
          "change, which is what makes pinning it lossless rather than a staleness risk. "
          "Rebuildable by tools/sourcing/build_newton_parity_ref.py (~4.4 min at 8 "
          "workers), but a rebuild is not free and the file is text and diffable. "
          "durable() + .gitignore negation; rationale in tools/sourcing/newton_parity.py "
          "and docs/design/pytest_suite_cost.md §3."),
    Entry("data/orbital/", KEEP, None, "tracked",
          "DISPOSITION: KEEP — orbital screen. The >=1 MiB violator is "
          "screen_pool.jsonl (1.2 MB, 4,669 rows), and it carries the ENUMERATION "
          "itself — each row is a Newton-solved nucleus (35-digit cx/cy, period, "
          "window_scale, log10|A|), i.e. phase 1 of tools/orbital/screen_pool.py, not "
          "a derived view over a pool committed elsewhere. There IS no other copy of "
          "this pool. Two reasons it stays: (1) enumeration runs ~25x the cost of "
          "screening, so this file is the expensive half of the pipeline in 1.2 MB; "
          "(2) it is not even reproducible on demand — enumeration is wall-clock "
          "budgeted and resumable (--enum-budget), so the exact atom set is a function "
          "of how long the run got, not of the seed alone. Same shape as the "
          "data/label_corpus/ precedent: one tracked file over the per-file rule, no "
          "traversal cost (6 files in the dir), content expensive-to-impossible to "
          "regenerate. The derived half (screen_scores.jsonl, 766 KB) is under the "
          "threshold and is regenerable from this pool; registered at the dir prefix "
          "so it does not flake the guard when the pool grows.", canary=False),

    # === RELOCATE -> artifacts — regenerable bulk (rebuildable render/cache) ====
    # Human-label corpora: the CROP JPGs are a pure function of render coords
    # (render-one; `present` was the other and was deleted 2026-08-10). The tracked
    # scores.json / images.jsonl labels + ledgers
    # are tiny and STAY in-tree (guarded by test_tracked_artifacts.py); only the crop
    # bulk relocates. `_work/` preview+staging subtrees are dead intermediates.
    Entry("data/label_corpus/", KEEP, None, "tracked",
          "crops/+vivid/ (3,822 files) RELOCATED out-of-tree behind artifacts.resolve (the "
          "label-corpus crop class; docs/design/label_corpus_relocation.md). What stays "
          "in-tree is the 71 tracked LABEL files (images.jsonl/scores.json/batch.json) — the "
          "corpus's ONE unrebuildable thing, a human verdict with no regen path — which MUST "
          "remain tracked in-tree; the v2filtered images.jsonl (1.4 MB of per-row provenance) "
          "is the one that trips the >=1 MiB rule. This is a deliberate KEEP, NOT a stale "
          "RELOCATE line to prune: do not relocate the labels. CANARY.",
          canary=True),
    Entry("data/wallpaper_corpus/", RELOCATE, ARTIFACTS, "mixed",
          "wallpaper batch crops (regenerable); tracked images.jsonl/ledgers stay"),
    Entry("data/render_mode_corpus/", RELOCATE, ARTIFACTS, "mixed",
          "render-mode batch crops (regenerable from each row's own render block + colour "
          "recipe); tracked manifests stay. NO LONGER EMPTY as of 2026-08-06: "
          "tools/mining/build_mining_sheet.py writes ~960 crops to "
          "data/render_mode_corpus/batches/<id>/crops/ IN-TREE (it does not route through "
          "artifacts.resolve, same as the retired render_mode_pilot writers), and the v1 "
          "render-mode head they feed is the LIVE strange-mode gate. The tracked half is "
          "gate_passers_<head>.json (v3 462 kB + the 2026-08-11 v4b refresh, 432 kB — one "
          "population per head version, neither replacing the other) + images.jsonl + "
          "batch.json; crops/, _fields/ and "
          "_progress_ledger.jsonl are gitignored by exact path. NO LONGER a forward "
          "declaration: the flag was dropped when the 2026-08-06 batch landed 545 MB of "
          "crops here, which is the condition it was declared against."),
    Entry("data/queries/", RELOCATE, ARTIFACTS, "mixed",
          "query-assembler field/colormap renders + scorer caches (regenerable via "
          "tools/queries); tracked queries/labels/*.json preference tiers stay. FORWARD as "
          "of 2026-08-08: the only over-threshold thing under this prefix was the ranker "
          "weight, and it now has its own narrower line below (longest-prefix coverage), so "
          "this one covers nothing TODAY. It is live, not dead — tools/queries/"
          "diversity_diagnostic.py writes data/queries/diagnostics/ and "
          "launch_query_label_server.py reads/writes data/queries/<batch_id>/, and a "
          "re-assembled batch's images land back here.", forward=True),
    Entry("data/queries/scorer/", RELOCATE, PRECIOUS, "mixed",
          "the palette-preference RANKER weight (pref-v3-gvo, 10.2 MB, LFS) — not "
          "GPU-reproducible, and its retrain input is worse than that: the labels are "
          "tracked but the records/ that join a candidate id to its (location, palette, "
          "coloring) are gone, so it cannot be retrained from what survives at all. A "
          "NARROWER prefix than data/queries/ above on purpose (longest-prefix coverage): "
          "the parent is regenerable render bulk and this is the opposite. Tracked by "
          "force-add at a gitignored path until 2026-08-08, now declared by exact-path "
          ".gitignore negation. CANARY.", canary=True),
    Entry("data/library/", RELOCATE, ARTIFACTS, "mixed",
          "field_cache render bulk (regenerable); tracked library_records.jsonl stays. "
          "FORWARD: only the one tracked record is there today, but "
          "tools/phoenix/phoenix_label_diversity.py still writes its retained field cache "
          "to data/library/field_cache/ in-tree.", forward=True),
    Entry("data/root_field/", RELOCATE, ARTIFACTS, "ignored",
          "root8k f32 score-field cache (4x 256 MB); regenerable via the Rust dump "
          "(src/root_field.rs CACHE_DIR) — needs the Rust-side artifacts resolver first"),
    Entry("data/discovery/", RELOCATE, ARTIFACTS, "mixed",
          "regenerable run-state overlays (campaign*/steered*/shakeout* renders, "
          "logs); tracked ledgers/pools stay in-tree. outcome_feats*.npz LEFT on "
          "2026-08-08 — demoted to bulk() and moved out-of-tree (10.88 MB / 28 files), "
          "rebuildable by tools/atlas/recompute_outcome_feats.py"),
    Entry("dramatic_palettes/", RELOCATE, ARTIFACTS, "mixed",
          "viz_render + viz_render_winners render sheets (regenerable); tracked "
          "palette definitions stay. FORWARD: only the 20 tracked palette definitions are "
          "there today, but tools/palettes/{viz_render,viz_render_winners,viz_batches}.py "
          "all default their sheet output to dramatic_palettes/<viz*>/ in-tree.",
          forward=True),
    Entry("data/guided_descend/", RELOCATE, ARTIFACTS, "mixed",
          "render/field caches; regenerable via enrich (tiny pool.jsonl pools "
          "stay). FORWARD: empty today, but this is the HEAD of the live corpus pipeline — "
          "`guided-descend --out` defaults to data/guided_descend/run4 (src/guided_descend.rs) "
          "and `enrich --pool` reads data/guided_descend/run5/pool.jsonl (src/enrich.rs). "
          "The next run repopulates it in-tree.", forward=True),
    # `data/ranker/` had a FORWARD entry here — "empty today, but train_eval{,_v1}.py and
    # campaign1_manifest.py write into it". That forward declaration is now false in the only
    # way that matters: those three writers were DELETED on 2026-08-08 with the pref_loc_v1
    # rebuild path (docs/design/deferred_recalibration.md § "Ranker growth — CLOSED"), so
    # nothing in the tree can ever populate the directory. A forward entry whose writer is
    # gone is a standing permission for a path that will not appear — the exact fossil shape
    # the note below this block was written about — so it goes out rather than being kept
    # "in case". `data/ranker/` stays ignored by the blanket /data/* rule like any other
    # untracked path.
    # NOTE — the four `data/v4/`..`data/v7/` build-cache lines that used to sit here are
    # DELETED, not left stale. Those caches are gone and will never exist again (the
    # manifests/plans they covered were wiped 2026-07-25 and are unrebuildable; the v8 build
    # is the durable replacement, covered by the KEEP line above). A registry line for a path
    # that can never come back is not an allowlist, it is a fossil — and 19 stale lines is how
    # a soft stale-report gets tuned out. Do NOT resurrect them for a future data/v10/: give
    # v10 its own line describing v10.
    #
    # Nine more lines were pruned on 2026-07-31 for the same reason, after each was tested
    # against "can anything still WRITE here?": data/label_crops/, data_large/label_crops/
    # (readers only — src/palette_probe.rs and tests/occupancy_parity.rs consume them, nothing
    # emits them), data/mining/ (the render bulk moved to scratch/mining/deploy_tail; only
    # sub-threshold JSON configs are still written under data/mining/),
    # data/calibration/maxiter_diag/ (the `maxiter-diag` subcommand was culled in P2 — only
    # two Rust comments still name it), data/classifier/{v2,v3,v4,v5_seed1}/ (superseded
    # weights, already deleted from disk; the retrain protocol gives a retrain its OWN version
    # dir — v9 did exactly that rather than overwrite v8 — so a superseded version is never
    # rebuilt in place), and data/focus_diag/ (no producer anywhere in the tree).
    #
    # POLICY CHANGE (2026-07-31): the stale-entry report was a WARNING, on the argument that
    # `data/v8/` was legitimately empty before its build and emptiness "cannot distinguish
    # not-yet-built from dead". `Entry.forward` now makes exactly that distinction, so the
    # premise no longer holds and tests/test_repo_size_guard.py asserts it HARD: every entry
    # either covers over-threshold content or is marked `forward=True`. A soft red that fires
    # on every run is a guard that gets trained out; 15 permanently-warning lines was that.

    # === RELOCATE -> precious-store — irreplaceable trained binaries (.pt) ======
    # Not GPU-reproducible (float nondeterminism), so no rebuild path. Active +
    # rollback anchors move to the precious store; the classifier weights are CANARY
    # paths — their eventual move needs a deliberate test_tracked_artifacts update.
    # Ladder as at 2026-08-07: v10 LIVE -> v8 -> v7 -> v6 -> v5; v9 is NOT a rung.
    # These notes said "ACTIVE_CKPT still names v8" for five days after the 2026-08-02 flip
    # (found by the pre-distillation census). The version each line DESCRIBES is fine to
    # write down; which one is DEPLOYED is derived state, so read it from
    # production_pins.ACTIVE_CKPT and never re-assert it here — that is the whole "derive
    # state in code, freeze it in records" rule, applied to a comment.
    Entry("data/classifier/v11/", RELOCATE, PRECIOUS, "tracked",
          "v11 model_best.pt — the v10 recipe VERBATIM (itself v9's, itself v8's) retrained "
          "on the v11 corpus under the randomized grouped split. BUILT, STAGED, NOT "
          "ADOPTED: nothing points at it, and the flip is a separate prompt judged against "
          "data/v11/prereg_v11.json. Same position v9 held below, and kept for the same "
          "reason — a not-GPU-reproducible weight cannot be rebuilt if the judgement is "
          "revisited. Declared by exact-path .gitignore negation, so a plain `git add` "
          "reaches it; model_last.pt is deliberately untracked (selection is on best). "
          "NOT a canary: canary status belongs to the deployed head and the rollback rungs, "
          "and a staged candidate is neither (test_tracked_artifacts.py's note on v9). "
          "Written by classifier/train_v11.py; model_best.pt is 34.2 MB, and the "
          "committing of it crossed CLAUDE.md's 20 MB rule and was confirmed by Matt "
          "2026-08-08 before the commit.",
          canary=False),
    Entry("data/classifier/v10/", RELOCATE, PRECIOUS, "tracked",
          "v10 model_best.pt — the v9 recipe (itself v8's, verbatim) retrained on the "
          "corpus EXTENDED with 1,267 maneuver-view locations. This is what "
          "production_pins.ACTIVE_CKPT resolves to (flipped 2026-08-02 against "
          "data/v10/prereg_v10.json). Declared by exact-path .gitignore negation, so a plain "
          "`git add` reaches it. model_last.pt is deliberately untracked (selection is on "
          "best). CANARY.", canary=True),
    # ---- retired heads: RECORDS TRACKED, WEIGHT NOT (2026-08-08 retention pass) ----
    # One entry for all five, because after the de-track they are one thing: a directory
    # holding a kilobyte of config.json (+ metrics.json where the version shipped one) and
    # an untracked 34 MB .pt that only this machine has. The old per-version entries each
    # justified a TRACKED weight; none of those justifications survives the policy, and
    # five stale rationales are worse than one accurate line.
    Entry("data/classifier/v5/", RELOCATE, PRECIOUS, "mixed",
          "v5..v9 model_best.pt were DE-TRACKED on 2026-08-08 under the ACTIVE+PREVIOUS "
          "weights-retention policy (docs/design/storage_classes.md); the ladder is v11 -> "
          "v10 and these are no longer rungs. The .pt files remain in the WORKING TREE on "
          "this machine and are precious there — not GPU-reproducible, and the corpus or "
          "split behind each has moved on. Verified emergency copies sit unreferenced at "
          "<artifacts>/retired_weights/. What stays TRACKED is each version's config.json "
          "(the recipe — v5/v6/v7 had none until extract_retired_config.py lifted it out of "
          "the checkpoint) plus v8/v9's metrics.json. CANARY on the records.", canary=True),
    Entry("data/classifier/v6/", RELOCATE, PRECIOUS, "mixed",
          "see data/classifier/v5/ — same 2026-08-08 retention pass.", canary=True),
    Entry("data/classifier/v7/", RELOCATE, PRECIOUS, "mixed",
          "see data/classifier/v5/. v7 was ALSO the frozen penultimate the pref_loc_v1 "
          "ranker's features were pinned to; that pin was deleted with the ranker on the "
          "same day (deferred_recalibration.md), which is what freed the weight.",
          canary=True),
    Entry("data/classifier/v8/", RELOCATE, PRECIOUS, "mixed",
          "see data/classifier/v5/. v8 was the one-flip rollback anchor until the v11 flip "
          "shortened the ladder to two rungs. Its t_good RECORD "
          "(data/v8/t_good_derivation.json) stays tracked and is unaffected: it is a record "
          "of what v8 served, not a rung.", canary=True),
    Entry("data/classifier/v9/", RELOCATE, PRECIOUS, "mixed",
          "see data/classifier/v5/. v9 was BUILT, STAGED and NEVER ADOPTED, so it was never "
          "a rollback rung even while tracked — a rejected candidate is not a critical "
          "final weight.", canary=True),
    Entry("data/wallpaper_head/", RELOCATE, PRECIOUS, "mixed",
          "trained wallpaper-quality heads — not GPU-reproducible (the corpus that produced "
          "each has moved on). Under ACTIVE+PREVIOUS retention TWO weights are tracked here "
          "since the 2026-08-11 flip (prompts/flip_29.md): v4b/seed_1 is the LIVE head "
          "(wallpaper_pins.HEAD_CKPT — the SEED file, picked on best blind sheet-D AUC>=4, "
          "not the top-level v4b/model_best.pt which is seed 0) and v3 is the one rollback "
          "rung. v4 stays de-tracked: it was staged and never adopted, so only its "
          "config.json + metrics.json remain, and the ladder is v4b -> v3, not v4b -> v4. "
          "Everything else under v4b — the other four seed dirs, eval_scores.jsonl, "
          "train.log — is ignored working state (~170 MB). Was 'ignored': these weights "
          "were tracked by force-add at a gitignored path until 2026-08-08 and are now "
          "declared by exact-path .gitignore negation, so durable() accepts them. CANARY.",
          canary=True),
    Entry("data/render_mode_head/", RELOCATE, PRECIOUS, "mixed",
          "trained render-mode (strange-mode gate) head .pt — not GPU-reproducible. Under "
          "ACTIVE+PREVIOUS retention TWO weights are tracked since the 2026-08-11 flip "
          "(prompts/flip_29.md): v3 (the `dedup_weighted` arm, trained FROM SCRATCH on the "
          "pooled three-batch corpus) is the LIVE gate, and v1 is the one rollback rung — "
          "v1's training corpus is gone, so it cannot be retrained at all. Each carries its "
          "own mining_gate_lock.{json,md}; v3's adds volume_match_mining.{json,md} and "
          "fit_slice_pred.json. v2 (the 2026-08-06 finetune that LOST the winner rule) is "
          "~60 MB of ignored working state, de-tracked once it was a rejected candidate. "
          "The four (28) sibling arms (v3_aug/v3_augx/v3_uniform/v3_ap2) keep v3's OLD "
          "shape: weight and per-seed dirs ignored, run record + winner-rule report tracked "
          "— none is pinnable, so none is a rung. CANARY.", canary=True),

    # === RELOCATE -> trash — dead / superseded ================================
    Entry("scratchpad/", RELOCATE, TRASH, "ignored",
          "canonical disposable temp dir — nothing large should persist here. FORWARD by "
          "construction: it is the ONE directory whose whole purpose is to receive "
          "unannounced writes, so it is empty in exactly the state we want and the line is "
          "the standing disposition for whatever lands next.", forward=True),
]


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
@dataclass
class Violator:
    rel: str            # repo-relative POSIX; dir violators end with '/'
    size: int
    is_dir: bool
    n_files: int = 0            # bulk violators only: files in the subtree
    rules: tuple = ()           # bulk violators only: which of ('files','bytes') fired


@dataclass
class ScanResult:
    file_violators: list[Violator]
    dir_violators: list[Violator]
    git_size: int
    # rule (c) — bulk directories (file-count and/or aggregate-byte), minimal granularity.
    bulk_violators: list[Violator] = field(default_factory=list)
    # populated by check_registry:
    uncovered: list[Violator] = field(default_factory=list)
    stale: list[Entry] = field(default_factory=list)
    # entries with no over-threshold content that are MARKED as live forward declarations
    # (Entry.forward). Reported for information; never counted stale.
    forward: list[Entry] = field(default_factory=list)

    @property
    def violators(self) -> list[Violator]:
        """Every flagged violator, across all three rules. Coverage + the hard fail are
        computed over this. A directory CAN appear twice (once under rule (b), once under
        rule (c)) — that is deliberate: each rule states an independent reason the path
        needs a registry line, and `entry_size` excludes the bulk list so the reported
        byte columns are not double-counted."""
        return self.file_violators + self.dir_violators + self.bulk_violators


def _excluded(rel_parts: tuple[str, ...]) -> bool:
    return bool(rel_parts) and rel_parts[0] in EXCLUDE_PREFIXES


def scan(repo: Path) -> ScanResult:
    """Walk the working tree; return file + directory violators and the .git FYI size."""
    small_sub: dict[Path, int] = {}     # subtree bytes of files < FILE_THRESHOLD
    all_sub: dict[Path, int] = {}       # subtree bytes of ALL files (rule c)
    cnt_sub: dict[Path, int] = {}       # subtree file COUNT (rule c)
    file_viol: list[Violator] = []

    # one pruned top-down walk: collect each dir's own small-file bytes + its kept
    # children, and flag big files inline. Excluded top-level trees are never descended.
    small_own: dict[Path, int] = {}
    all_own: dict[Path, int] = {}
    cnt_own: dict[Path, int] = {}
    kid_map: dict[Path, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(repo, topdown=True):
        d = Path(dirpath)
        rel = d.relative_to(repo)
        if rel == Path("."):
            dirnames[:] = [n for n in dirnames if n not in EXCLUDE_PREFIXES]
        elif _excluded(rel.parts):
            dirnames[:] = []
            continue
        own = 0
        own_all = 0
        own_cnt = 0
        for fn in filenames:
            fp = d / fn
            try:
                sz = fp.stat().st_size
            except OSError:
                continue
            own_all += sz
            own_cnt += 1
            if sz >= FILE_THRESHOLD:
                file_viol.append(Violator(fp.relative_to(repo).as_posix(), sz, False))
            else:
                own += sz
        small_own[d] = own
        all_own[d] = own_all
        cnt_own[d] = own_cnt
        kid_map[d] = [d / n for n in dirnames]

    # bottom-up subtree sums (small-file bytes, all-file bytes, file count)
    for d in sorted(kid_map, key=lambda p: len(p.parts), reverse=True):
        kids = kid_map[d]
        small_sub[d] = small_own.get(d, 0) + sum(small_sub.get(k, 0) for k in kids)
        all_sub[d] = all_own.get(d, 0) + sum(all_sub.get(k, 0) for k in kids)
        cnt_sub[d] = cnt_own.get(d, 0) + sum(cnt_sub.get(k, 0) for k in kids)

    # rule (b): MINIMAL dirs whose small-file subtree >= DIR_THRESHOLD (no child qualifies)
    dir_viol: list[Violator] = []
    for d, sz in small_sub.items():
        if sz < DIR_THRESHOLD:
            continue
        if any(small_sub.get(k, 0) >= DIR_THRESHOLD for k in kid_map.get(d, [])):
            continue
        rel = d.relative_to(repo)
        if rel == Path("."):
            continue
        dir_viol.append(Violator(rel.as_posix() + "/", sz, True))

    # rule (c) BULK: MINIMAL dirs over the file-count OR aggregate-byte threshold.
    # "Minimal" is evaluated on the SAME predicate that flagged the dir, so a 10k-file /
    # 300 MB dir is reported at the leaf-most 10k-file dir, not pushed up to a parent that
    # merely happens to be over on bytes.
    def _bulk(p: Path) -> bool:
        return (cnt_sub.get(p, 0) >= DIR_FILE_COUNT_THRESHOLD
                or all_sub.get(p, 0) >= DIR_BYTES_THRESHOLD)

    bulk_viol: list[Violator] = []
    for d in cnt_sub:
        if not _bulk(d):
            continue
        if any(_bulk(k) for k in kid_map.get(d, [])):
            continue
        rel = d.relative_to(repo)
        if rel == Path("."):
            continue
        rules = tuple(
            r for r, hit in (("files", cnt_sub[d] >= DIR_FILE_COUNT_THRESHOLD),
                             ("bytes", all_sub[d] >= DIR_BYTES_THRESHOLD)) if hit)
        bulk_viol.append(Violator(rel.as_posix() + "/", all_sub[d], True,
                                  n_files=cnt_sub[d], rules=rules))

    git_size = _dir_size(repo / GIT_DIR)
    file_viol.sort(key=lambda v: -v.size)
    dir_viol.sort(key=lambda v: -v.size)
    bulk_viol.sort(key=lambda v: -v.size)
    return ScanResult(file_viol, dir_viol, git_size, bulk_violators=bulk_viol)


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dn, filenames in os.walk(root):
        d = Path(dirpath)
        for fn in filenames:
            try:
                total += (d / fn).stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------
def covering_entry(rel: str, registry: list[Entry] = REGISTRY) -> Entry | None:
    """Most-specific (longest-prefix) registry entry covering `rel`, or None."""
    best: Entry | None = None
    for e in registry:
        if rel == e.prefix or rel.startswith(e.prefix):
            if best is None or len(e.prefix) > len(best.prefix):
                best = e
    return best


def check_registry(res: ScanResult, registry: list[Entry] = REGISTRY) -> ScanResult:
    """Fill res.uncovered (violators no entry covers), res.stale (unmarked entries
    covering no current violator) and res.forward (empty entries deliberately marked as
    live forward declarations).

    The split is the whole point: emptiness alone cannot tell not-yet-built from dead, so
    an unsplit "stale" list is a permanent warning nobody acts on. `Entry.forward` is the
    human judgement — *can anything still write here* — recorded once, which leaves
    `res.stale` meaning only "nobody classified this", i.e. something to fix."""
    covered_prefixes: set[str] = set()
    uncovered: list[Violator] = []
    for v in res.violators:
        e = covering_entry(v.rel, registry)
        if e is None:
            uncovered.append(v)
        else:
            covered_prefixes.add(e.prefix)
    res.uncovered = sorted(uncovered, key=lambda v: -v.size)
    empty = [e for e in registry if e.prefix not in covered_prefixes]
    res.stale = [e for e in empty if not e.forward]
    res.forward = [e for e in empty if e.forward]
    return res


def entry_size(res: ScanResult, entry: Entry) -> int:
    """Total violator bytes assigned (most-specifically) to this entry.

    Rules (a)+(b) only. The rule-(c) bulk list overlaps them by construction (a bulk dir
    usually also holds the small files rule (b) flagged), so summing all three would
    double-count; bulk gets its own report section with its own count/byte columns."""
    tot = 0
    for v in res.file_violators + res.dir_violators:
        if covering_entry(v.rel) is entry:
            tot += v.size
    return tot


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def human(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or u == "TB":
            return f"{int(x)}B" if u == "B" else f"{x:.1f}{u}"
        x /= 1024
    return f"{x:.1f}TB"


def _report(repo: Path) -> int:
    res = check_registry(scan(repo))
    print("=" * 78)
    print(f"REPO-SIZE GUARD   root={repo}")
    print(f"  file threshold >= {human(FILE_THRESHOLD)}   dir(small-file) threshold >= {human(DIR_THRESHOLD)}")
    print(f"  bulk dir threshold >= {DIR_FILE_COUNT_THRESHOLD} files or >= {human(DIR_BYTES_THRESHOLD)}")
    print(f"  excluded from flagging: {', '.join(EXCLUDE_PREFIXES)}")
    print("=" * 78)
    n_v = len(res.violators)
    tot = sum(v.size for v in res.file_violators + res.dir_violators)
    print(f"\n{n_v} violators ({len(res.file_violators)} files + {len(res.dir_violators)} "
          f"small-file dirs + {len(res.bulk_violators)} bulk dirs), {human(tot)} flagged "
          f"(rules a+b; bulk bytes overlap and are listed separately).")
    print(f".git FYI (history-rewrite target, not flagged): {human(res.git_size)}")

    # grouped by disposition
    for disp, tier in GROUP_ORDER:
        entries = [e for e in REGISTRY if e.disposition == disp and e.tier == tier]
        if not entries:
            continue
        head = disp if disp == KEEP else f"{RELOCATE} -> {tier}"
        gtot = sum(entry_size(res, e) for e in entries)
        print(f"\n--- {head}  ({human(gtot)}) ---")
        for e in sorted(entries, key=lambda e: -entry_size(res, e)):
            sz = entry_size(res, e)
            tag = " [CANARY]" if e.canary else ""
            print(f"  {human(sz):>9}  {e.prefix:<52} {e.tracked}{tag}")
            print(f"             -> {e.reason}")

    # --- rule (c): BULK directories (storage_classes.md rule 5) ---
    print(f"\n--- BULK DIRECTORIES  (rule c: >= {DIR_FILE_COUNT_THRESHOLD} files "
          f"or >= {human(DIR_BYTES_THRESHOLD)}, minimal granularity) ---")
    if not res.bulk_violators:
        print("  none — no in-tree directory is bulk.")
    for v in res.bulk_violators:
        e = covering_entry(v.rel)
        cov = f"covered by {e.prefix} [{e.label()}]" if e else "*** UNCOVERED ***"
        print(f"  {v.n_files:>7} files  {human(v.size):>9}  {v.rel:<46} "
              f"({'+'.join(v.rules)})  {cov}")

    print("\n" + "=" * 78)
    if res.uncovered:
        print(f"UNCOVERED VIOLATORS ({len(res.uncovered)}) — new bloat, no registry entry:")
        for v in res.uncovered:
            print(f"  {human(v.size):>9}  {v.rel}")
    else:
        print("OK: every violator is covered by a registry entry.")
    if res.forward:
        print(f"\nLIVE FORWARD DECLARATIONS ({len(res.forward)}) — empty now, but a committed "
              f"writer can still land here; the line is what that write's disposition is:")
        for e in res.forward:
            print(f"  {e.prefix:<40} [{e.label()}]")
    if res.stale:
        print(f"\nSTALE REGISTRY ENTRIES ({len(res.stale)}) — no over-threshold content and "
              f"NOT marked as a forward declaration. Either delete the line (nothing can "
              f"write there any more) or mark it forward=True:")
        for e in res.stale:
            print(f"  {e.prefix}")
    else:
        print("OK: no stale registry entries.")
    print("=" * 78)
    return 1 if res.uncovered else 0


def _check(repo: Path) -> int:
    res = check_registry(scan(repo))
    if res.uncovered:
        print(f"FAIL: {len(res.uncovered)} uncovered violator(s):", file=sys.stderr)
        for v in res.uncovered:
            print(f"  {human(v.size):>9}  {v.rel}", file=sys.stderr)
        return 1
    print(f"PASS: {len(res.violators)} violators, all covered; "
          f"{len(res.stale)} stale entr{'y' if len(res.stale)==1 else 'ies'}, "
          f"{len(res.forward)} live forward declaration(s).")
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="terse pass/fail only (what the pytest guard asserts)")
    args = ap.parse_args()
    return _check(repo) if args.check else _report(repo)


if __name__ == "__main__":
    raise SystemExit(main())
