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
    # The same walk carries the outcome-FEATURE class, which is the one relocated family
    # that is a FILE and not a directory — so it is checked on `files`, not on `dirs`. A
    # dirs-only scan would have relocated it correctly and never checked it, which is the
    # gap artifacts_resolver.md §3 names ("a class added to the resolver but not to the
    # tripwire is never checked").
    disc = repo_root / "data" / "discovery"
    if disc.exists():
        for root, dirs, files in os.walk(disc):
            for d in dirs:
                full = Path(root) / d
                rel = full.relative_to(repo_root).as_posix()
                if A._is_discovery_scratch(rel):
                    n = _count_files(full)
                    if n:
                        offenders.append(f"{rel} has {n}+ files in-tree at {full}")
            for fn in files:
                full = Path(root) / fn
                rel = full.relative_to(repo_root).as_posix()
                if A._is_discovery_feats(rel):
                    offenders.append(f"{rel} is an in-tree outcome-feature store at {full}")
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
    # descent-harness crop class: walk data/descent_harness for in-tree crops/vivid dirs
    # with files (a tools/descent emit that bypassed store's artifacts-routed crop paths).
    dh = repo_root / "data" / "descent_harness"
    if dh.exists():
        for root, dirs, _files in os.walk(dh):
            for d in dirs:
                full = Path(root) / d
                rel = full.relative_to(repo_root).as_posix()
                if A._is_descent_harness_crop(rel):
                    n = _count_files(full)
                    if n:
                        offenders.append(f"{rel} has {n}+ files in-tree at {full}")
    # versioned bulk trees under data/v<N>/: the aug caches (matched as a class since v10,
    # so v10's and v11's are covered here and not by the v8/v9 literals above) and the
    # eval-canonical tiles v11 introduced. One walk, because both are `data/v<N>/<name>`.
    dv = repo_root / "data"
    if dv.exists():
        for vdir in sorted(p for p in dv.glob("v*") if p.is_dir()):
            for full in sorted(p for p in vdir.iterdir() if p.is_dir()):
                rel = full.relative_to(repo_root).as_posix()
                if A._is_aug_cache(rel) or A._is_eval_canon(rel):
                    n = _count_files(full)
                    if n:
                        offenders.append(f"{rel} has {n}+ files in-tree at {full}")
    # minibrot source-sheet class: tiles/ and sheets/ under data/minibrot_sources.
    msrc = repo_root / "data" / "minibrot_sources"
    if msrc.exists():
        for root, dirs, _files in os.walk(msrc):
            for d in dirs:
                full = Path(root) / d
                rel = full.relative_to(repo_root).as_posix()
                if A._is_minibrot_source_bulk(rel):
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


def test_tripwire_fires_on_synthetic_descent_harness_crops(tmp_path):
    """The tripwire FIRES on descent-harness crops/vivid/thumbs bombed in-tree (a
    tools/descent emit or triage tile that bypassed the artifacts-routed image paths),
    while the durable text records beside them (emits.jsonl / selection.json /
    verified_bad.json / triage/*.jsonl) stay in-tree."""
    assert _scan_in_tree_offenders(tmp_path) == []
    crop = tmp_path / "data/descent_harness/crops" / "d2_p03_001__e1.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    offenders = _scan_in_tree_offenders(tmp_path)
    assert any("descent_harness/crops" in o for o in offenders), offenders
    vivid = tmp_path / "data/descent_harness/vivid" / "d2_p03_001__e1.jpg"
    vivid.parent.mkdir(parents=True)
    vivid.write_bytes(b"x")
    assert any("descent_harness/vivid" in o for o in _scan_in_tree_offenders(tmp_path))
    # the triage wall's thumbnails are the same class (3 per atom over a pool headed
    # past 1000) and trip it the same way
    thumb = tmp_path / "data/descent_harness/thumbs" / "mt0123456789ab__x4.png"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"x")
    assert any("descent_harness/thumbs" in o for o in _scan_in_tree_offenders(tmp_path))
    # the durable text records must NOT trip it (they legitimately stay in-tree)
    (tmp_path / "data/descent_harness/emits.jsonl").write_bytes(b"x")
    (tmp_path / "data/descent_harness/selection.json").write_bytes(b"x")
    (tmp_path / "data/descent_harness/triage").mkdir(parents=True)
    (tmp_path / "data/descent_harness/triage/verdicts.jsonl").write_bytes(b"x")
    crop.unlink()
    vivid.unlink()
    thumb.unlink()
    assert _scan_in_tree_offenders(tmp_path) == []


def test_descent_harness_crop_class_maps_under_artifacts_root():
    """Any data/descent_harness/{crops,vivid,thumbs} relocates by pattern; the records
    stay in-tree, and a `crops_staging`-style sibling component must not match."""
    root = A.artifacts_root()
    for rel in [
        "data/descent_harness/crops/d2_p03_001__e1.jpg",
        "data/descent_harness/vivid/d2_p03_001__e1.jpg",
        "data/descent_harness/thumbs/mt0123456789ab__x4.png",
        "data/descent_harness/crops",                       # the dir itself
        "data/descent_harness/thumbs",
    ]:
        assert A._is_descent_harness_crop(A._norm(rel)), rel
        assert A.is_relocated(rel), rel
        assert A.resolve(rel) == root / rel, rel
    for rel in [
        "data/descent_harness/selection.json",
        "data/descent_harness/selection_triage.json",
        "data/descent_harness/emits.jsonl",
        "data/descent_harness/verified_bad.json",
        "data/descent_harness/triage/pool.jsonl",           # the triage records
        "data/descent_harness/triage/verdicts.jsonl",
        "data/descent_harness/crops_staging/x.jpg",         # component != crops/vivid/thumbs
        "data/descent_harness/thumbs_staging/x.png",
    ]:
        assert not A._is_descent_harness_crop(A._norm(rel)), rel
        assert not A.is_relocated(rel), rel
        assert A.resolve(rel) == A.REPO_ROOT / rel, rel


def test_minibrot_source_bulk_maps_under_artifacts_root():
    """The source-sheet tiles and sheet HTML relocate by pattern; the durable nuclei
    lists and descriptors beside them stay in-tree."""
    root = A.artifacts_root()
    for rel in [
        "data/minibrot_sources/tiles/mt0123456789ab__x4.png",
        "data/minibrot_sources/sheets/probe.html",
        "data/minibrot_sources/sheets/index.html",
        "data/minibrot_sources/tiles",
    ]:
        assert A._is_minibrot_source_bulk(A._norm(rel)), rel
        assert A.is_relocated(rel), rel
        assert A.resolve(rel) == root / rel, rel
    for rel in [
        "data/minibrot_sources/probe/atoms.jsonl",
        "data/minibrot_sources/probe/meta.json",
        "data/minibrot_sources/overlap.json",
        "data/minibrot_sources/index.json",
        "data/minibrot_sources/tiles_staging/x.png",      # component != tiles/sheets
    ]:
        assert not A._is_minibrot_source_bulk(A._norm(rel)), rel
        assert not A.is_relocated(rel), rel
        assert A.resolve(rel) == A.REPO_ROOT / rel, rel


def test_tripwire_fires_on_synthetic_minibrot_source_bulk(tmp_path):
    """A sheet run that bypassed the resolver and wrote tiles in-tree is caught, while
    the durable nuclei lists beside them are not."""
    assert _scan_in_tree_offenders(tmp_path) == []
    tile = tmp_path / "data/minibrot_sources/tiles" / "mt0123456789ab__x4.png"
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b"x")
    assert any("minibrot_sources/tiles" in o for o in _scan_in_tree_offenders(tmp_path))
    (tmp_path / "data/minibrot_sources/probe").mkdir(parents=True)
    (tmp_path / "data/minibrot_sources/probe/atoms.jsonl").write_bytes(b"x")
    tile.unlink()
    assert _scan_in_tree_offenders(tmp_path) == []


def test_relocated_prefixes_map_under_artifacts_root():
    root = A.artifacts_root()
    for prefix in A.RELOCATED_PREFIXES:
        sample = f"{prefix}/sub/file.jpg"
        assert A.is_relocated(sample)
        resolved = A.resolve(sample)
        assert resolved == root / sample, (prefix, resolved)


def test_eval_canon_class_maps_under_artifacts_root():
    """Any ``data/v<N>/eval_canon`` relocates by pattern — v11's is the first, and a future
    version's needs no registry line. Component-exact: the SIBLING manifest and record must
    not be swept in by it (the manifest relocates as a v11 row file, the record is
    committed)."""
    root = A.artifacts_root()
    for rel in ["data/v11/eval_canon/7.jpg", "data/v11/eval_canon",
                "data/v12/eval_canon/1.jpg"]:                     # future version
        assert A._is_eval_canon(A._norm(rel)), rel
        assert A.resolve(rel) == root / A._norm(rel), rel
    for rel in ["data/v11/eval_canon_manifest.jsonl", "data/v11/eval_canon_record.json",
                "data/v11/eval_canonical/1.jpg"]:
        assert not A._is_eval_canon(A._norm(rel)), rel
    # ...and the record it sits beside is COMMITTED, so it must resolve in-tree.
    assert A.resolve("data/v11/eval_canon_record.json") == \
        A.REPO_ROOT / "data/v11/eval_canon_record.json"


def test_tripwire_fires_on_synthetic_eval_canon_in_tree(tmp_path):
    """A canonical-tile tree that reappears in the working tree is caught, for a version
    that is in no registry at all."""
    mirror = tmp_path / "repo"
    (mirror / "data" / "v13" / "eval_canon").mkdir(parents=True)
    assert not _scan_in_tree_offenders(mirror)
    (mirror / "data" / "v13" / "eval_canon" / "1.jpg").write_bytes(b"x")
    offenders = _scan_in_tree_offenders(mirror)
    assert any("v13/eval_canon" in o for o in offenders), offenders


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


def test_discovery_feats_class_maps_under_artifacts_root():
    """The outcome-FEATURE store relocates by pattern, for any run and both members of the
    family (the store and redecode_grid's `_v7_t45` subset) — no registry line required,
    including a run never seen before."""
    root = A.artifacts_root()
    for rel in [
        "data/discovery/steady_state_v2_20260807/outcome_feats.npz",
        "data/discovery/campaign2/breadth/outcome_feats.npz",
        "data/discovery/phoenix_grid/grid/outcome_feats_v7_t45.npz",
        "data/discovery/a_run_that_does_not_exist_yet/outcome_feats.npz",
    ]:
        assert A._is_discovery_feats(A._norm(rel)), rel
        assert A.is_relocated(rel), rel
        assert A.resolve(rel) == root / rel, rel


def test_discovery_feats_lookalikes_stay_in_tree():
    """The ledger the feature store is derived FROM must not follow it out of the tree, and
    neither must the other per-run .npz overlays — they are gitignored, not relocated, and
    conflating the two is how a durable record leaves by accident."""
    for rel in [
        "data/discovery/campaign2/breadth/outcome_ledger.jsonl",
        "data/discovery/phoenix_grid/grid/distinct_looks.npz",
        "data/discovery/campaign2/breadth/morph_mem.npz",
        "data/discovery/campaign2/breadth/node_embs.npz",
        "data/outcome_feats.npz",                      # not under data/discovery/
        "data/discovery/campaign2/breadth/outcome_feats.jsonl",   # not an .npz
    ]:
        assert not A._is_discovery_feats(A._norm(rel)), rel
        assert A.resolve(rel) == A.REPO_ROOT / rel, rel


def test_tripwire_fires_on_synthetic_discovery_feats(tmp_path):
    """The tripwire FIRES on a feature store written in-tree by a writer that bypassed
    `discovery_sinks.feats_path` — for a run with NO registry line naming it. Fires on a
    FILE, which is what makes this branch different from every other family here."""
    assert _scan_in_tree_offenders(tmp_path) == []
    victim = tmp_path / "data/discovery/campaign7/breadth/outcome_feats.npz"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"x")
    offenders = _scan_in_tree_offenders(tmp_path)
    assert any("campaign7" in o and "outcome_feats" in o for o in offenders), offenders
    # the ledger beside it is durable and must NOT trip the wire
    victim.unlink()
    (victim.parent / "outcome_ledger.jsonl").write_bytes(b"x")
    assert _scan_in_tree_offenders(tmp_path) == []


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


def test_live_aug_caches_are_registered():
    """Both live aug-cache trees are literals in RELOCATED_PREFIXES.

    v9 (the v8 corpus re-rendered at the raised iteration cap) was built as a SECOND live
    tree rather than a replacement, with v8's 12.1 GB held as the rollback anchor; v8's
    tree was then deleted on 2026-07-31, once v9 was trained and evaluated. Its prefix
    stays registered regardless: a rebuild must land out-of-tree exactly as the first render
    did (two steps since 2026-08-03 — tools/v8/build_plan.py regenerates the deleted
    plan.jsonl from the committed manifest, then render_cache.py renders through it). Registration must happen BEFORE the first
    render — storage_classes.md rule 5, a new bulk family is born out-of-tree — so this
    asserts the registry rather than the disk, which is why deleting the tiles leaves it
    green."""
    assert "data/v8/aug_cache" in A.RELOCATED_PREFIXES
    assert "data/v9/aug_cache" in A.RELOCATED_PREFIXES
    assert A.is_relocated("data/v9/aug_cache/7116/twilight_shifted__id__s1.0000__sh0.0000__ss2.jpg")
    # ...and the sibling-prefix guard still holds for the new literal
    assert not A.is_relocated("data/v9/aug_cache_notes/x.jpg")


def test_tau_h_rederive_work_dir_is_registered_and_would_not_be_committed_in_tree():
    """`tools/atlas/tau_h_rederive.py` declares its work dir `bulk()`. Two things must hold,
    and the second is specific to THIS family: `data/atlas/` carries a `!/data/atlas/`
    re-include (the fitted theta_hat artifact is tracked), so an in-tree write here would
    not merely sit in the tree — it would be COMMITTED. The resolver keeps it out; the
    .gitignore stanza is the belt-and-braces half for a writer that bypassed the resolver."""
    import subprocess
    assert "data/atlas/tau_h_rederive" in A.RELOCATED_PREFIXES
    root = A.artifacts_root()
    assert A.is_relocated("data/atlas/tau_h_rederive")
    assert A.resolve("data/atlas/tau_h_rederive") == root / "data/atlas/tau_h_rederive"
    # `check-ignore` is asked about FILES: the stanza is directory-scoped, and git cannot
    # apply a dir-only rule to a bare path that does not exist on disk.
    for rel in ("data/atlas/tau_h_rederive/rows.jsonl",
                "data/atlas/tau_h_rederive/tiles/h_campaign1_breadth_1_0_cheap.jpg"):
        assert A.is_relocated(rel), rel
        assert A.resolve(rel) == root / rel, rel
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", rel],
                                 cwd=A.REPO_ROOT, capture_output=True).returncode == 0
        assert ignored, f"{rel} is NOT gitignored — an in-tree straggler would be committed"
    # the tracked artifact the tool WRITES beside it must stay in-tree and committable
    assert not A.is_relocated("data/atlas/tau_h_base_v10.json")
    assert A.resolve("data/atlas/tau_h_base_v10.json") == A.REPO_ROOT / "data/atlas/tau_h_base_v10.json"
    assert subprocess.run(["git", "check-ignore", "-q", "--", "data/atlas/tau_h_base_v10.json"],
                          cwd=A.REPO_ROOT, capture_output=True).returncode != 0


def test_backslash_and_dotslash_normalized():
    assert A.is_relocated("data\\v8\\aug_cache\\1\\x.jpg")
    assert A.is_relocated("./data/v8/aug_cache/1/x.jpg")
    assert A.is_relocated("data\\v9\\aug_cache\\1\\x.jpg")
    # normalization also applies to the discovery-scratch class
    assert A.is_relocated("data\\discovery\\campaign2\\breadth\\scratch\\x.jpg")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(A.ARTIFACTS_ENV, str(tmp_path))
    assert A.artifacts_root() == tmp_path
    assert A.resolve("data/v8/aug_cache/1/x.jpg") == tmp_path / "data/v8/aug_cache/1/x.jpg"
