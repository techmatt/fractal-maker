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
       (a) any FILE >= FILE_THRESHOLD (1 MiB — matches the pre-commit blob hook), and
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

The guard test (`tests/test_repo_size_guard.py`) fails on any flagged violator not
covered by a registry entry (new bloat caught from today), and REPORTS (does not
fail) any registry entry that no longer has over-threshold content (a nudge to
delete the line). As things relocate, their RELOCATE lines come out; when only KEEP
lines remain, every in-tree exception is explicit and reviewed.

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
FILE_THRESHOLD = 1 * 1024 * 1024        # 1 MiB — matches .git/hooks/pre-commit LIMIT
DIR_THRESHOLD = 100 * 1024 * 1024       # ~100 MB — many-small-file catch

# --- BULK rule (c) thresholds: storage_classes.md rule 5, "no bulk in-tree" ---
# Chosen against the measured shape of THIS tree, not picked round:
#   * FILE COUNT 2,000. The whole non-excluded working tree is ~5.2k files. A single
#     subtree holding 2k is already ~40% of everything a recursive walk has to visit,
#     and it is two orders of magnitude below the bombs that made `grep -r` take >120s
#     (152k / 42k / 27k / 22k aug-cache dirs; 317k discovery scratch) — i.e. low enough
#     to fire long before the tree is unusable, high enough that no source or metadata
#     directory here comes close (largest today: tools/ at 504).
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
    the path-segment boundary)."""
    prefix: str
    disposition: str                 # KEEP | RELOCATE
    tier: str | None                 # None for KEEP; artifacts|precious-store|trash
    tracked: str                     # 'tracked' | 'ignored' | 'mixed'
    reason: str
    canary: bool = False             # covers a canaried path (move needs a canary update)

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
          "v8 classifier build: manifest.jsonl IS the training population + split; "
          "plan/cache_manifest are 171,624 rows each (~55/79 MB, LFS) and are the ONLY "
          "thing that maps a cached tile back to a location — losing them is exactly how "
          "the v4..v7 caches became 243k unattributable JPGs. durable() + canaried; the "
          "aug_cache JPGs themselves are bulk() and out-of-tree. CANARY.", canary=True),
    Entry("data/v9/", KEEP, None, "tracked",
          "v9 classifier build — the v8 corpus re-rendered at the raised iteration cap "
          "(docs/design/auto_maxiter.md). Same shape and same reason as data/v8/ above: "
          "plan/cache_manifest are 170,808 rows each (~54/92 MB, LFS) and are the ONLY "
          "thing mapping a cached tile back to a location. It has no manifest.jsonl of "
          "its own on purpose — it reads v8's and records that file's sha256, so 'same "
          "corpus' is a checked claim rather than a copy that can drift. The aug_cache "
          "JPGs are bulk() and out-of-tree. durable() + canaried. CANARY.", canary=True),
    Entry("data/library_embeddings/", KEEP, None, "mixed",
          "prospect-library CLIP embeddings (embeddings.npz, tracked): unregenerable "
          "except value-approximate under a verdict-sensitive threshold. CANARY.",
          canary=True),

    # === RELOCATE -> artifacts — regenerable bulk (rebuildable render/cache) ====
    # Human-label corpora: the CROP JPGs are a pure function of render coords
    # (present/render-one). The tracked scores.json / images.jsonl labels + ledgers
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
          "render-mode batch crops (regenerable via present); tracked manifests stay"),
    Entry("data/label_crops/", RELOCATE, ARTIFACTS, "ignored",
          "early loose label-crop feed (loose0_v2/v3); regenerable render output"),
    Entry("data_large/label_crops/", RELOCATE, ARTIFACTS, "ignored",
          "loose0 crop feed; regenerable render output (tracked data_large/README stays)"),
    Entry("data/queries/", RELOCATE, ARTIFACTS, "mixed",
          "query-assembler field/colormap renders + scorer caches (regenerable via "
          "tools/queries); tracked queries/labels/*.json preference tiers stay"),
    Entry("data/library/", RELOCATE, ARTIFACTS, "mixed",
          "field_cache render bulk (regenerable); tracked library_records.jsonl stays"),
    Entry("data/root_field/", RELOCATE, ARTIFACTS, "ignored",
          "root8k f32 score-field cache (4x 256 MB); regenerable via the Rust dump "
          "(src/root_field.rs CACHE_DIR) — needs the Rust-side artifacts resolver first"),
    Entry("data/discovery/", RELOCATE, ARTIFACTS, "mixed",
          "regenerable run-state overlays (campaign*/steered*/shakeout* renders, "
          "logs); tracked ledgers/pools/outcome_feats provenance stays in-tree"),
    Entry("dramatic_palettes/", RELOCATE, ARTIFACTS, "mixed",
          "viz_render + viz_render_winners render sheets (regenerable); tracked "
          "palette definitions stay"),
    Entry("data/mining/", RELOCATE, ARTIFACTS, "mixed",
          "mining prospect renders (run1); regenerable via tools/mining"),
    Entry("data/guided_descend/", RELOCATE, ARTIFACTS, "mixed",
          "render/field caches (atlas_probe_step0, run5, julia_test_bulb); "
          "regenerable via present/enrich (tiny pool.jsonl pools stay)"),
    Entry("data/ranker/", RELOCATE, ARTIFACTS, "ignored",
          "frozen-feature location-ranker fits + feature caches (pref_loc_v0/v1, "
          "campaign1); regenerable — logistic on committed frozen features"),
    Entry("data/calibration/maxiter_diag/", RELOCATE, ARTIFACTS, "ignored",
          "maxiter diagnostic renders; regenerable (the frozen energy_calibration.json "
          "metric bins are tiny and stay tracked)"),
    # NOTE — the four `data/v4/`..`data/v7/` build-cache lines that used to sit here are
    # DELETED, not left stale. Those caches are gone and will never exist again (the
    # manifests/plans they covered were wiped 2026-07-25 and are unrebuildable; the v8 build
    # is the durable replacement, covered by the KEEP line above). A registry line for a path
    # that can never come back is not an allowlist, it is a fossil — and 19 stale lines is how
    # the soft stale-report gets tuned out. Do NOT resurrect them for a future data/v9/: give
    # v9 its own line describing v9.
    #
    # The stale-entry report stays a WARNING and must not be promoted to a failure:
    # `data/v8/` was legitimately empty before its build, so a hard fail on emptiness would
    # put the guard red during ordinary work — which is the other way guards get tuned out.
    # Emptiness cannot distinguish not-yet-built from dead, so warn-then-prune is correct and
    # the judgement stays with the human.

    # === RELOCATE -> precious-store — irreplaceable trained binaries (.pt) ======
    # Not GPU-reproducible (float nondeterminism), so no rebuild path. Active +
    # rollback anchors move to the precious store; the classifier weights are CANARY
    # paths — their eventual move needs a deliberate test_tracked_artifacts update.
    Entry("data/classifier/v8/", RELOCATE, PRECIOUS, "tracked",
          "v8 model_best.pt — LIVE deployed discovery-gate weight (K=4 ordinal head; the "
          "first version that can decode class 4). CANARY.", canary=True),
    Entry("data/classifier/v7/", RELOCATE, PRECIOUS, "tracked",
          "v7 model_best.pt — one-flip rollback anchor (the role v6 held before the v8 "
          "promotion); ALSO the frozen penultimate the pref_loc_v1 ranker's features are "
          "pinned to, so this weight is load-bearing beyond rollback. CANARY.", canary=True),
    Entry("data/classifier/v6/", RELOCATE, PRECIOUS, "tracked",
          "v6 model_best.pt — deeper rollback anchor. CANARY.", canary=True),
    Entry("data/classifier/v5/", RELOCATE, PRECIOUS, "tracked",
          "v5 model_best.pt — deepest rollback anchor. CANARY.", canary=True),
    Entry("data/wallpaper_head/", RELOCATE, PRECIOUS, "ignored",
          "trained wallpaper-quality heads (v1/v2/v3 .pt) — not GPU-reproducible; "
          "active + rollback -> precious-store, older versions curate to trash at move"),
    Entry("data/render_mode_head/", RELOCATE, PRECIOUS, "ignored",
          "trained render-mode (strange-mode gate) head v1 .pt — not GPU-reproducible"),

    # === RELOCATE -> trash — dead / superseded ================================
    Entry("data/classifier/v2/", RELOCATE, TRASH, "ignored",
          "superseded classifier v2 weight — won't be retrained"),
    Entry("data/classifier/v3/", RELOCATE, TRASH, "ignored",
          "superseded classifier v3 weight — won't be retrained"),
    Entry("data/classifier/v4/", RELOCATE, TRASH, "ignored",
          "superseded classifier v4 weight — won't be retrained"),
    Entry("data/classifier/v5_seed1/", RELOCATE, TRASH, "ignored",
          "v5 seed-1 diagnostic variant — not the live checkpoint, disposable"),
    Entry("data/focus_diag/", RELOCATE, TRASH, "ignored",
          "focus-diagnostic scratch (orbit-space field .npy dumps); dead, regenerable"),
    Entry("scratchpad/", RELOCATE, TRASH, "ignored",
          "canonical disposable temp dir — nothing large should persist here"),
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
    """Fill res.uncovered (violators no entry covers) and res.stale (entries covering
    no current violator)."""
    covered_prefixes: set[str] = set()
    uncovered: list[Violator] = []
    for v in res.violators:
        e = covering_entry(v.rel, registry)
        if e is None:
            uncovered.append(v)
        else:
            covered_prefixes.add(e.prefix)
    res.uncovered = sorted(uncovered, key=lambda v: -v.size)
    res.stale = [e for e in registry if e.prefix not in covered_prefixes]
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
    if res.stale:
        print(f"\nSTALE REGISTRY ENTRIES ({len(res.stale)}) — no over-threshold content, "
              f"delete the line:")
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
          f"{len(res.stale)} stale entr{'y' if len(res.stale)==1 else 'ies'}.")
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
