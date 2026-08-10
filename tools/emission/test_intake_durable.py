"""The two emission intake snapshots resolve durable, and every reader points at that path.

The intake `cluster_tags` snapshot names WHICH locations/clusters the seeded library draws
against — a population that cannot be regenerated once its discovery scratch is cleared (the
campaign-1 and library_intake_2 snapshots were wiped exactly that way).

THE WRITERS ARE GONE (2026-08-10): `campaign1_intake.py` / `library_intake_2.py` were deleted
in the closure sweep — no caller, and `descriptor.load_admitted`'s predicate had changed under
them, so a re-run would have rewritten those two names with a DIFFERENT population. The
snapshots stay; the READERS (`deficit_scheduler`'s library seed, `stage_first_release`'s union)
are live and are now the only thing that names these paths, so the rels below are sourced from
them rather than from the deleted writers' constants. What is pinned is unchanged in substance:
the paths are durable-class (a gitignore would make `paths.durable` raise), disk_audit forces
NEVER on them, and the regenerable field/emb bulk beside them stays out of the tracked tree.

Run:  uv run pytest tools/emission/test_intake_durable.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "emission", ROOT / "tools" / "atlas",
          ROOT / "tools" / "corpus", ROOT / "tools" / "wallpaper", ROOT / "tools" / "mining",
          ROOT / "tools" / "scoring", ROOT / "tools" / "audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import paths  # noqa: E402
import stage_first_release as sfr  # noqa: E402
import deficit_scheduler as dsch  # noqa: E402
import disk_audit as da  # noqa: E402

# The two snapshot addresses, as the LIVE readers spell them. Repo-relative rels for
# `paths.durable()`; the absolute forms are what the readers hold.
C1_REL = "data/emission/campaign1/intake.json"
I2_REL = "data/emission/library_intake_2/intake.json"


def _clear_ignore_cache():
    clear = getattr(paths._is_gitignored, "cache_clear", None)
    if clear:
        clear()


@pytest.mark.parametrize("rel", [C1_REL, I2_REL])
def test_durable_class_refuses_a_gitignored_snapshot_home(rel, monkeypatch):
    """If a snapshot home ever became gitignored, `paths.durable()` must fail on the spot —
    exercised on the exact rels the live readers resolve."""
    _clear_ignore_cache()
    monkeypatch.setattr(paths, "_is_gitignored", lambda _p: True)
    with pytest.raises(paths.DurabilityError):
        paths.durable(rel, mkparents=True)
    _clear_ignore_cache()


@pytest.mark.parametrize("rel,tail", [(C1_REL, C1_REL), (I2_REL, I2_REL)])
def test_real_paths_not_gitignored(rel, tail):
    _clear_ignore_cache()
    assert str(paths.durable(rel)).replace("\\", "/").endswith(tail)


def test_readers_point_at_the_durable_path():
    """No drift between the two live readers, and both resolve to the durable address. With
    the writers deleted this is the whole join: if `deficit_scheduler` and
    `stage_first_release` ever disagreed, one of them would read an empty seed in silence."""
    assert dsch.INTAKE_ARTIFACT == sfr.C1_INTAKE              # scheduler seed == release union
    for reader, rel in ((sfr.C1_INTAKE, C1_REL), (sfr.I2_INTAKE, I2_REL)):
        _clear_ignore_cache()
        assert Path(reader) == paths.durable(rel)


def test_bulk_fields_are_not_promoted_into_the_tracked_tree():
    """Only the snapshot is durable; the regenerable field/emb bulk is NOT, so the split is
    real and we did not promote a file-count bomb into the tree.

    This used to assert `"scratch" in <path>` for all three — and that assertion was
    actively wrong, not merely loose. `scratch/` is one way to be non-durable and it is the
    way that GUARANTEES deletion: the seed embeddings it pinned there were wiped, and the
    campaign-1 pair they belonged to is dark forever as a result. Being under `scratch/` was
    never the property worth pinning; being un-tracked is. `INTAKE_EMB_DIR` is now `bulk()`
    (out-of-tree via the ARTIFACTS_ROOT resolver, survives `rm -r scratch/*`), which the
    old assertion would have called a regression. `C1_DIR`/`I2_DIR` are still under
    `scratch/`, which this still permits — they are inventoried as live scratch references,
    not silently repaired here."""
    for p in (dsch.INTAKE_EMB_DIR, sfr.C1_DIR, sfr.I2_DIR):
        # `is_relative_to`, not `startswith`: the artifacts root is a SIBLING of the repo
        # (`fractal-maker-artifacts` vs `fractal-maker`), so a string-prefix test calls
        # every out-of-tree bulk path in-tree.
        in_tree = Path(p).is_relative_to(ROOT)
        assert (not in_tree) or paths._is_gitignored(str(p)), (
            f"{p} would be COMMITTED — bulk must not land in the tracked tree")
    # The seed's own vectors specifically must be out of the deletable class now — asserted
    # THROUGH the live rule rather than by re-spelling it, so this cannot drift from what
    # the scheduler actually refuses at resolve time.
    dsch._refuse_scratch_class("embeddings", dsch.INTAKE_EMB_DIR)


@pytest.mark.parametrize("rel", [C1_REL, I2_REL])
def test_disk_audit_forces_never(rel):
    assert da.classify(rel).category == da.NEVER
