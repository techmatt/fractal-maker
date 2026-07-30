"""Reappearance tripwire + resolver unit tests for the relocated-artifacts move.

The storage restructure moves regenerable file-count bombs OUT of the working tree
to ARTIFACTS_ROOT, routing readers/writers through ``tools/corpus/artifacts.py``.
Two kinds of relocated family now exist:

* the **live v8 aug-cache**, a fixed versioned literal in ``RELOCATED_PREFIXES``
  (the v4..v7 caches were deleted and their literals dropped);
* the **discovery-scratch class** — any ``data/discovery/**/scratch`` tree — matched
  by *pattern* (``_is_discovery_scratch``), so a new campaign relocates with no new
  registry line. campaign2 breadth/dive scratch (317k files / ~45 GB) was reclaimed;
  ``steered_frontier.py`` now declares this class at its write site so future runs are
  born out-of-tree.

Two guarantees are pinned here:

1. **Reappearance tripwire** (`test_no_relocated_root_repopulated_in_tree`): if a
   missed *writer* silently re-materializes a relocated family under its old
   in-tree path — a literal aug-cache prefix OR any discovery-scratch tree — this
   goes RED and names the offender. This is the backstop the grep-completeness sweep
   can't provide.

2. **Resolver correctness**: relocated prefixes and the discovery-scratch pattern map
   under ARTIFACTS_ROOT, sibling look-alikes and every other path stay in-tree, and the
   env override works.

Run: ``uv run pytest tools/audit/test_relocated_artifacts.py``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
import artifacts as A  # noqa: E402


def _count_files(p: Path, cap: int = 5) -> int:
    """Count up to `cap` regular files under p (cheap; we only need >0 vs 0)."""
    if not p.exists():
        return 0
    n = 0
    for _root, _dirs, files in os.walk(p):
        n += len(files)
        if n >= cap:
            break
    return n


def _scan_in_tree_offenders(repo_root: Path) -> list[str]:
    """Relocated families holding real files under their OLD in-tree path in `repo_root`.

    Covers BOTH relocation mechanisms: the literal ``RELOCATED_PREFIXES`` (v8 aug-cache)
    and the discovery-scratch *class* (any ``data/discovery/**/scratch`` tree, matched by
    pattern — so a new campaign that bombs the tree is caught with no registry edit).

    Empty leftover dirs are tolerated (a move can leave the parent behind); real files
    mean a writer bypassed the resolver and re-bombed the tree. Parameterized on the
    root so the fire direction is testable against a synthetic tree, not only the repo."""
    offenders = []
    for prefix in A.RELOCATED_PREFIXES:
        in_tree = repo_root / prefix
        n = _count_files(in_tree)
        if n:
            offenders.append(f"{prefix} has {n}+ files in-tree at {in_tree}")
    # discovery-scratch class: walk data/discovery for any in-tree `scratch` dir with files.
    disc = repo_root / "data" / "discovery"
    if disc.exists():
        for root, dirs, _files in os.walk(disc):
            for d in dirs:
                full = Path(root) / d
                rel = full.relative_to(repo_root).as_posix()
                if A._is_discovery_scratch(rel):
                    n = _count_files(full)
                    if n:
                        offenders.append(f"{rel} has {n}+ files in-tree at {full}")
    # label-corpus crop class: walk data/label_corpus/batches for in-tree crops/vivid dirs
    # with files. Same conservative rule — a crop tree that reappears in-tree (a batch
    # builder that bypassed corpus_common.crops_dir/vivid_dir) is caught here even for a
    # batch id never registered anywhere.
    lc = repo_root / "data" / "label_corpus" / "batches"
    if lc.exists():
        for root, dirs, _files in os.walk(lc):
            for d in dirs:
                full = Path(root) / d
                rel = full.relative_to(repo_root).as_posix()
                if A._is_label_corpus_crop(rel):
                    n = _count_files(full)
                    if n:
                        offenders.append(f"{rel} has {n}+ files in-tree at {full}")
    return offenders


def test_no_relocated_root_repopulated_in_tree():
    """The real working tree is clean — no relocated family repopulated in-tree."""
    offenders = _scan_in_tree_offenders(REPO_ROOT)
    assert not offenders, (
        "Relocated artifact family repopulated in the working tree (a writer "
        "bypassed tools/corpus/artifacts.resolve): " + "; ".join(offenders)
    )


def test_tripwire_fires_on_synthetic_repopulation(tmp_path):
    """The tripwire FIRES (and names the offender) when a relocated family reappears.

    The clean-tree test above proves it stays quiet; this proves it does not stay
    quiet when it shouldn't. Planted in a tmp repo mirror so the real tree is untouched."""
    # clean mirror -> quiet
    assert _scan_in_tree_offenders(tmp_path) == []
    # plant a real file under one relocated prefix's OLD in-tree path -> must be caught
    victim = A.RELOCATED_PREFIXES[0]
    planted = tmp_path / victim / "sub" / "bomb.jpg"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"x")
    offenders = _scan_in_tree_offenders(tmp_path)
    assert any(victim in o for o in offenders), offenders
    # an empty leftover dir alone is tolerated (a move can leave the parent behind)
    planted.unlink()
    assert _scan_in_tree_offenders(tmp_path) == []


def test_tripwire_fires_on_synthetic_discovery_scratch(tmp_path):
    """The tripwire FIRES on a NEW campaign's scratch bombed in-tree — with NO registry
    line naming it. This is the payoff of matching the discovery-scratch class by pattern:
    campaign3 forgetting to register still gets caught (conservative), not silently 45 GB
    in the tree."""
    assert _scan_in_tree_offenders(tmp_path) == []
    victim = "data/discovery/campaign3/breadth/scratch"        # never in RELOCATED_PREFIXES
    assert not any(victim.startswith(p) for p in A.RELOCATED_PREFIXES)
    planted = tmp_path / victim / "expand_b0001" / "bomb.jpg"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"x")
    offenders = _scan_in_tree_offenders(tmp_path)
    assert any("campaign3" in o and "scratch" in o for o in offenders), offenders
    # a discovery file that is NOT scratch must NOT trip it (the durable ledgers stay in-tree)
    (tmp_path / "data/discovery/campaign3/breadth" / "outcome_ledger.jsonl").write_bytes(b"x")
    planted.unlink()
    assert _scan_in_tree_offenders(tmp_path) == []


def test_tripwire_fires_on_synthetic_label_corpus_crops(tmp_path):
    """The tripwire FIRES on a NEW batch's crops/vivid bombed in-tree — with NO registry
    line naming that batch. Same payoff as the discovery-scratch class test: matching the
    label-corpus crop family by pattern means a batch builder that bypasses
    corpus_common.crops_dir (re-materializing crops in-tree) is caught conservatively,
    even for a batch id the resolver has never seen."""
    assert _scan_in_tree_offenders(tmp_path) == []
    victim = "data/label_corpus/batches/2099-01-01_never_seen/crops"    # unregistered batch
    assert not any(victim.startswith(p) for p in A.RELOCATED_PREFIXES)
    planted = tmp_path / victim / "0_center_pal.jpg"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"x")
    offenders = _scan_in_tree_offenders(tmp_path)
    assert any("never_seen" in o and "crops" in o for o in offenders), offenders
    # vivid companion trips it too
    vivid = tmp_path / "data/label_corpus/batches/2099-01-01_never_seen/vivid" / "0.jpg"
    vivid.parent.mkdir(parents=True)
    vivid.write_bytes(b"x")
    assert any("vivid" in o for o in _scan_in_tree_offenders(tmp_path))
    # the tracked labels beside the crops must NOT trip it — they legitimately stay in-tree
    (tmp_path / "data/label_corpus/batches/2099-01-01_never_seen/images.jsonl").write_bytes(b"x")
    (tmp_path / "data/label_corpus/batches/2099-01-01_never_seen/scores.json").write_bytes(b"x")
    planted.unlink()
    vivid.unlink()
    assert _scan_in_tree_offenders(tmp_path) == []


def test_label_corpus_crop_class_maps_under_artifacts_root():
    """Any data/label_corpus/batches/*/{crops,vivid} relocates by pattern — no registry
    line required, including a batch never seen before."""
    root = A.artifacts_root()
    for rel in [
        "data/label_corpus/batches/2026-07-26_minibrot_roster_v2/crops/0_center_pal.jpg",
        "data/label_corpus/batches/2026-07-26_minibrot_roster_v2/vivid/0_center_pal.jpg",
        "data/label_corpus/batches/some_future_batch/crops",       # unregistered; dir itself
    ]:
        assert A._is_label_corpus_crop(A._norm(rel)), rel
        assert A.is_relocated(rel), rel
        assert A.resolve(rel) == root / rel, rel


def test_label_corpus_labels_and_lookalikes_stay_in_tree():
    """The tracked labels + batch root stay in-tree; only crops/vivid relocate, and a
    `crops_staging`-style sibling component must not match the crop class."""
    for rel in [
        "data/label_corpus/batches/b/images.jsonl",
        "data/label_corpus/batches/b/scores.json",
        "data/label_corpus/batches/b/batch.json",
        "data/label_corpus/batches/b/blind.jsonl",
        "data/label_corpus/batches/b/crops_staging/x.jpg",         # component != crops/vivid
        "data/label_corpus/CORPUS_SCHEMA.md",
    ]:
        assert not A._is_label_corpus_crop(A._norm(rel)), rel
        assert not A.is_relocated(rel), rel
        assert A.resolve(rel) == A.REPO_ROOT / rel, rel


def test_relocated_prefixes_map_under_artifacts_root():
    root = A.artifacts_root()
    for prefix in A.RELOCATED_PREFIXES:
        sample = f"{prefix}/sub/file.jpg"
        assert A.is_relocated(sample)
        resolved = A.resolve(sample)
        assert resolved == root / sample, (prefix, resolved)


def test_discovery_scratch_class_maps_under_artifacts_root():
    """Any data/discovery/**/scratch relocates by pattern — no registry line required,
    including a campaign never seen before."""
    root = A.artifacts_root()
    for rel in [
        "data/discovery/campaign2/breadth/scratch/expand_b0001/x.jpg",
        "data/discovery/campaign2/dive/scratch/roots/x.field.bin",
        "data/discovery/campaign3/breadth/scratch",              # future campaign, unregistered
        "data/discovery/steered_runs/A/scratch/native_mandelbrot/pool.jsonl",
    ]:
        assert A._is_discovery_scratch(A._norm(rel)), rel
        assert A.is_relocated(rel), rel
        assert A.resolve(rel) == root / rel, rel


def test_discovery_non_scratch_stays_in_tree():
    """The durable in-tree ledgers under a discovery run must NOT be relocated, and a
    `scratchpad`-style sibling component must not match the `scratch` class."""
    for rel in [
        "data/discovery/campaign2/breadth/outcome_ledger.jsonl",
        "data/discovery/campaign2/breadth/harvest_log.jsonl",
        "data/discovery/campaign2/summary.json",
        "data/discovery/campaign2/breadth/scratchpad/x.txt",     # component != "scratch"
    ]:
        assert not A._is_discovery_scratch(A._norm(rel)), rel
        assert not A.is_relocated(rel), rel
        assert A.resolve(rel) == A.REPO_ROOT / rel, rel


def test_sibling_lookalike_stays_in_tree():
    # a sibling that merely shares a prefix string must NOT be relocated
    assert not A.is_relocated("data/v4/aug_cache_notes/x.txt")
    assert A.resolve("data/v4/aug_cache_notes/x.txt") == A.REPO_ROOT / "data/v4/aug_cache_notes/x.txt"


def test_non_relocated_paths_resolve_in_tree():
    for p in ["data/v4/cache_manifest.jsonl",
              "data/label_corpus/batches/b/images.jsonl",
              "data/discovery/campaign2/breadth/outcome_ledger.jsonl"]:
        assert not A.is_relocated(p)
        assert A.resolve(p) == A.REPO_ROOT / p


def test_backslash_and_dotslash_normalized():
    assert A.is_relocated("data\\v8\\aug_cache\\1\\x.jpg")
    assert A.is_relocated("./data/v8/aug_cache/1/x.jpg")
    # normalization also applies to the discovery-scratch class
    assert A.is_relocated("data\\discovery\\campaign2\\breadth\\scratch\\x.jpg")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(A.ARTIFACTS_ENV, str(tmp_path))
    assert A.artifacts_root() == tmp_path
    assert A.resolve("data/v8/aug_cache/1/x.jpg") == tmp_path / "data/v8/aug_cache/1/x.jpg"
