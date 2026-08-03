"""Resolver for relocated regenerable bulk artifacts.

Why this exists
---------------
~99.98% of the files a recursive tool traverses in this repo were regenerable ML
scratch (augmentation caches, per-node discovery scratch) living *inside* the
source tree, which is what made a plain ``grep -r`` take >120 s. Those two file-
count bombs were physically moved OUT of the working tree to a sibling directory
so traversal (grep/find/editor-indexers/watchers) is fast *by construction*,
independent of gitignore-awareness.

This module is the single seam that maps a **repo-relative** artifact path (the
portable, version-invariant string stored in manifests/plans) to its **real**
on-disk location. Every reader AND writer of a relocated family MUST route
through :func:`resolve` so the data is found where it now lives, and so a rebuild
never re-materializes the bomb in-tree.

ARTIFACTS_ROOT
--------------
Defaults to a *sibling* of the repo (``../fractal-maker-artifacts``), so a
fresh checkout on any machine resolves without configuration. Override with the
``FRACTAL_ARTIFACTS_ROOT`` environment variable (e.g. to point at a different
volume). The relocated tree mirrors the repo-relative layout exactly:
``<ARTIFACTS_ROOT>/data/v4/aug_cache/...`` etc.

Non-relocated paths resolve in-tree, unchanged. This module is deliberately
additive and narrow: it relocates the live aug-cache family (a fixed versioned
path), the discovery-scratch *class* (any ``data/discovery/**/scratch`` tree,
matched by pattern so no per-campaign registration is ever needed), and the
label-corpus crop *class* (any ``data/label_corpus/batches/*/{crops,vivid}`` tree,
matched the same way — see ``_is_label_corpus_crop`` and
``docs/design/label_corpus_relocation.md``), and the descent-harness image *class* (any
``data/descent_harness/{crops,vivid,thumbs}`` tree — see ``_is_descent_harness_crop``),
nothing else.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# tools/corpus/artifacts.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_ENV = "FRACTAL_ARTIFACTS_ROOT"

# Repo-relative prefixes whose contents were relocated to ARTIFACTS_ROOT. Each is
# matched as a whole path component (exact, or followed by "/") so a sibling like
# ``data/v4/aug_cache_notes`` would NOT accidentally match ``data/v4/aug_cache``.
# Keep this list in lockstep with the reappearance tripwire
# (tools/audit/test_relocated_artifacts.py) and the .gitignore stanzas.
#
# LITERAL prefixes are for the aug-cache families, which live at fixed, versioned paths.
# The v4..v7 caches were DELETED (commit 7068839) and will never exist again, so their
# lines were dropped — only the LIVE v8 cache remains. Discovery scratch is NOT listed
# here: it is relocated as a CLASS by pattern (see `_is_discovery_scratch`), so a new
# campaign never needs a new registry line — forgetting costs conservatism, not 45 GB
# in-tree.
RELOCATED_PREFIXES = (
    # v8: registered BEFORE the first render, not after 170k files have landed in the
    # tree (storage_classes.md rule 5 — a new bulk family is born out-of-tree). The tree
    # itself was DELETED on 2026-07-31 (12.13 GB / 171,384 tiles) once v9 was trained; the
    # prefix STAYS, because a v8 rebuild must land out-of-tree exactly as the first render
    # did. That rebuild is TWO steps since 2026-08-03: data/v8/plan.jsonl was deleted with
    # the tiles, so tools/v8/build_plan.py regenerates it (byte-identically, from the
    # committed manifest) before tools/v8/render_cache.py renders through it. Dropping the
    # literal — as was done for v4..v7 — would be right only if v8 could never be rebuilt.
    "data/v8/aug_cache",
    # v9: the same corpus re-rendered at the raised iteration cap
    # (docs/design/auto_maxiter.md). A SEPARATE tree rather than a re-render in place,
    # because v4-render-batch resumes by skipping existing outputs — re-rendering into
    # v8's tree would skip all 170,808 old-cap tiles and silently produce nothing, and
    # the alternative (deleting them) throws away the only rollback anchor. Registered
    # before the first render, same rule as v8.
    "data/v9/aug_cache",
    # The tau_h re-derivation work dir (tools/atlas/tau_h_rederive.py): rows.jsonl (one
    # rendered+scored pair per sampled harvest/walk row) plus its transient tile chunks.
    # A LITERAL rather than a class, because it is one fixed path, not a family that grows
    # a new member per run. It lived under scratch/ and was re-rendered from zero TWICE
    # after a scratch wipe — expensive (GPU scoring + 2 renders/row) but fully
    # deterministic given the committed ledgers and the active weights, i.e. textbook
    # bulk(). Registered here rather than left in-tree because `!/data/atlas/` re-includes
    # this subtree: in-tree it would be COMMITTED, not merely present.
    "data/atlas/tau_h_rederive",
)


def _is_discovery_scratch(r: str) -> bool:
    """True iff ``r`` (normalized repo-relative) is a ``scratch/`` tree under
    ``data/discovery/`` — the per-run render/field scratch that ``steered_frontier.py``
    writes as ``<run_dir>/scratch`` (a file-count bomb: campaign2 breadth/dive were 317k
    files / ~45 GB).

    Matched as a CLASS by pattern rather than a per-campaign literal in
    ``RELOCATED_PREFIXES``: a new campaign's scratch relocates the same way with no
    registry edit, so a forgotten registration fails toward out-of-tree (conservative),
    never toward a silent 45 GB in the source tree. The write site declares the class by
    routing through ``paths.bulk()``; this predicate is what makes that routing relocate.
    Mirrors the ``/data/discovery/**/scratch/`` .gitignore stanza. Component-exact, so a
    sibling like ``data/discovery/x/scratchpad`` does NOT match."""
    parts = r.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "data"
        and parts[1] == "discovery"
        and "scratch" in parts[2:]
    )


def _is_label_corpus_crop(r: str) -> bool:
    """True iff ``r`` (normalized repo-relative) is a label-corpus batch's ``crops/`` or
    ``vivid/`` tree — the regenerable label-crop bulk (a pure function of each row's
    render block via render-one) relocated OUT of the working tree. It was 3,822 files
    (~72% of the working tree) before the move; see
    ``docs/design/label_corpus_relocation.md``.

    Matched as a CLASS by pattern rather than a per-batch literal, exactly like
    ``_is_discovery_scratch``: a new batch relocates the same way with no registry edit,
    so a forgotten registration fails toward out-of-tree (conservative), never toward a
    silent bulk in the source tree. The batch builders declare the class by routing their
    crop writes through ``corpus_common.crops_dir``/``vivid_dir`` (which call ``resolve``);
    this predicate is what makes that routing relocate. Mirrors the
    ``/data/label_corpus/batches/*/{crops,vivid}/`` .gitignore stanzas. Component-exact on
    ``crops``/``vivid``, so a sibling like ``.../crops_staging`` does NOT match — and the
    tracked labels (``images.jsonl``/``scores.json``/``batch.json``) stay in-tree because
    they are not under a ``crops``/``vivid`` component."""
    parts = r.split("/")
    return (
        len(parts) >= 5
        and parts[0] == "data"
        and parts[1] == "label_corpus"
        and parts[2] == "batches"
        and parts[4] in ("crops", "vivid")
    )


DESCENT_HARNESS_IMAGE_DIRS = ("crops", "vivid", "thumbs")


def _is_descent_harness_crop(r: str) -> bool:
    """True iff ``r`` (normalized repo-relative) is one of the descent harness's IMAGE
    trees (``data/descent_harness/{crops,vivid,thumbs}/**``) — the emitted canonical +
    vivid crop pairs and the triage-wall thumbnails (``tools/descent/``). All three are
    regenerable bulk (a pure function of a stored record), and all three grow without
    bound — the crop set from 40 atoms to 163, the thumbnail set 3 per atom over a
    triage pool headed past 1000 — so they relocate OUT of the working tree exactly
    like ``_is_label_corpus_crop``. The durable text records
    (``emits.jsonl``/``selection.json``/``verified_bad.json`` and
    ``triage/{pool,verdicts}.jsonl``) sit under ``data/descent_harness/`` and stay
    in-tree because they are not under one of those components. Matched as a CLASS by
    pattern (component-exact) so a sibling like ``.../crops_staging`` does NOT match;
    mirrors the ``/data/descent_harness/{crops,vivid,thumbs}/`` .gitignore stanzas and
    the reappearance tripwire in ``tools/audit/test_relocated_artifacts.py``."""
    parts = r.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "data"
        and parts[1] == "descent_harness"
        and parts[2] in DESCENT_HARNESS_IMAGE_DIRS
    )


def _is_minibrot_source_bulk(r: str) -> bool:
    """True iff ``r`` is the minibrot source-sheet bulk: ``data/minibrot_sources/tiles``
    (3 scales x ~150 atoms x 8 sources) or ``data/minibrot_sources/sheets`` (the HTML
    pages, written beside the tiles so a sheet addresses them by a plain relative path
    and opens by double-click). Both are regenerable-but-expensive, so they relocate OUT
    of the working tree exactly like the descent-harness images; the durable nuclei lists
    and descriptors sit directly under ``data/minibrot_sources/<source>/`` and stay
    in-tree because they are not under a ``tiles``/``sheets`` component. Matched as a
    CLASS by pattern (component-exact), mirroring the
    ``/data/minibrot_sources/{tiles,sheets}/`` .gitignore stanzas."""
    parts = r.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "data"
        and parts[1] == "minibrot_sources"
        and parts[2] in ("tiles", "sheets")
    )


_AUG_CACHE_VERSION_RE = re.compile(r"^v\d+$")


def _is_aug_cache(r: str) -> bool:
    """True iff ``r`` is a versioned augmentation-cache tree: ``data/v<N>/aug_cache/**``.

    Matched as a CLASS, for the reason ``_is_discovery_scratch`` gives and for one earned
    the hard way: the v10 extension's first dry run resolved
    ``data/v10/aug_cache`` IN-TREE, because the registry above holds a literal per version
    and v10 had not been added yet. Nothing would have failed — 30,408 JPGs would simply
    have landed in the working tree under a gitignored path, which is the silent-bulk
    outcome the whole resolver exists to prevent. A pattern fails toward out-of-tree, so a
    forgotten registration costs nothing.

    The literals in ``RELOCATED_PREFIXES`` stay: they are the record of which versions were
    registered deliberately, and this predicate is a superset, not a replacement.
    Component-exact on ``aug_cache``, so a sibling like ``data/v9/aug_cache_probe`` does
    NOT match."""
    parts = r.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "data"
        and bool(_AUG_CACHE_VERSION_RE.match(parts[1]))
        and parts[2] == "aug_cache"
    )


def artifacts_root() -> Path:
    """Root under which relocated artifacts live (env override or repo sibling)."""
    env = os.environ.get(ARTIFACTS_ENV)
    if env:
        return Path(env)
    return REPO_ROOT.parent / "fractal-maker-artifacts"


def _norm(rel) -> str:
    """Normalize to a forward-slash, repo-relative string (no leading ./ or /)."""
    s = str(rel).replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


def is_relocated(rel) -> bool:
    """True iff ``rel`` (repo-relative) belongs to a relocated family: a literal
    aug-cache prefix, any versioned ``data/v<N>/aug_cache`` tree, any discovery-scratch
    tree, any label-corpus crop/vivid tree, any descent-harness image tree, or the
    minibrot source-sheet tiles/sheets bulk (all but the literals matched by class)."""
    r = _norm(rel)
    if any(r == p or r.startswith(p + "/") for p in RELOCATED_PREFIXES):
        return True
    return (_is_aug_cache(r) or _is_discovery_scratch(r) or _is_label_corpus_crop(r)
            or _is_descent_harness_crop(r) or _is_minibrot_source_bulk(r))


def resolve(rel) -> Path:
    """Map a repo-relative artifact path to its real on-disk location.

    Relocated families -> ``ARTIFACTS_ROOT/<rel>``; every other path ->
    ``REPO_ROOT/<rel>`` (i.e. unchanged in-tree behavior). Accepts str or Path.
    """
    r = _norm(rel)
    base = artifacts_root() if is_relocated(r) else REPO_ROOT
    return base / r
