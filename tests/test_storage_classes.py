"""Guard: the storage-class helper (`tools/paths.py`) refuses a durable path that git
would discard.

The load-bearing assertion is inside `durable()`: a path declared durable must not be
gitignored. This test exercises it with a synthetic ignored path and confirms it fires,
and confirms the complementary cases (a re-included durable path passes; bulk() routes
through the ARTIFACTS_ROOT resolver; scratch() lands under the disposable tree).

Light lane — imports only `tools/paths.py` (+ `tools/corpus/artifacts.py`); pathlib/os
and a `git check-ignore` subprocess, no numpy/torch/GPU.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import paths as P  # noqa: E402


# A path under data/ that matches the blanket `/data/*` ignore with no re-include —
# synthetic (the file need not exist; `git check-ignore` evaluates rules by string).
SYNTHETIC_IGNORED = "data/__synthetic_ignored_for_test__/should_not_exist.bin"
# A durable path re-included by an explicit .gitignore negation.
REINCLUDED_DURABLE = "data/discovery/outcome_ledger.jsonl"


def test_synthetic_ignored_path_is_detected():
    """Sanity: the synthetic path really is gitignored (else the assertion test below
    would pass vacuously)."""
    ab = str(REPO_ROOT / SYNTHETIC_IGNORED)
    assert P._is_gitignored(ab), (
        f"{SYNTHETIC_IGNORED} is unexpectedly NOT gitignored — a negation was added; "
        f"pick a different synthetic ignored path."
    )


def test_durable_raises_on_gitignored_path():
    """The core assertion: declaring a gitignored path durable raises loudly, naming
    the path and the class."""
    with pytest.raises(P.DurabilityError) as ei:
        P.durable(SYNTHETIC_IGNORED)
    msg = str(ei.value)
    assert SYNTHETIC_IGNORED in msg
    assert "durable" in msg.lower()


def test_durable_passes_on_reincluded_path():
    """A durable path re-included by a .gitignore negation is accepted (returns its
    absolute in-tree location)."""
    got = P.durable(REINCLUDED_DURABLE)
    assert got == REPO_ROOT / "data" / "discovery" / "outcome_ledger.jsonl"


def test_durable_mkparents_only_after_assertion(tmp_path, monkeypatch):
    """mkparents must not create the directory when the assertion fails — a rejected
    durable write leaves no trace."""
    with pytest.raises(P.DurabilityError):
        P.durable(SYNTHETIC_IGNORED, mkparents=True)
    assert not (REPO_ROOT / "data" / "__synthetic_ignored_for_test__").exists()


def test_bulk_routes_relocated_family_out_of_tree():
    """bulk() delegates to the ARTIFACTS_ROOT resolver — a relocated family lands
    out-of-tree, not re-materialized in the source tree. Covers both the live v8
    aug-cache literal and the discovery-scratch class (matched by pattern)."""
    for rel, needle in [("data/v8/aug_cache/x.npz", "aug_cache"),
                        ("data/discovery/campaign3/breadth/scratch/x.jpg", "scratch")]:
        got = P.bulk(rel)
        assert needle in str(got)
        assert REPO_ROOT not in got.parents, f"relocated bulk path {got} is still in-tree"


def test_bulk_leaves_non_relocated_in_tree():
    got = P.bulk("data/discovery/foo.jsonl")
    assert got == REPO_ROOT / "data" / "discovery" / "foo.jsonl"


def test_scratch_under_disposable_tree():
    got = P.scratch("atlas", "sheet.png")
    assert got == REPO_ROOT / "scratch" / "atlas" / "sheet.png"
