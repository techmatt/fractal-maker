"""The two emission intake snapshots are written through `paths.durable()`, and every reader
points at that same durable path.

The intake `cluster_tags` snapshot names WHICH locations/clusters the seeded library draws
against — a population that cannot be regenerated once its discovery scratch is cleared (the
campaign-1 and library_intake_2 snapshots were wiped exactly that way). The writers now land the
snapshot under `data/emission/<pass>/intake.json` via `paths.durable()`, which asserts at the
write site that git would keep it. These tests pin that assertion AND that the readers
(deficit_scheduler, library_intake_2's cross-ref, stage_first_release's union) resolve the same
path — an orphaned durable write that no reader follows would be a silent half-fix.

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
import campaign1_intake as c1i  # noqa: E402
import library_intake_2 as li2  # noqa: E402
import stage_first_release as sfr  # noqa: E402
import deficit_scheduler as dsch  # noqa: E402
import disk_audit as da  # noqa: E402


def _clear_ignore_cache():
    clear = getattr(paths._is_gitignored, "cache_clear", None)
    if clear:
        clear()


@pytest.mark.parametrize("rel", [c1i.INTAKE_REL, li2.INTAKE_REL])
def test_write_site_asserts_durability(rel, monkeypatch):
    """If the snapshot home ever became gitignored, the write must fail on the spot — the
    writer routes through paths.durable(rel), so we exercise that exact rel."""
    _clear_ignore_cache()
    monkeypatch.setattr(paths, "_is_gitignored", lambda _p: True)
    with pytest.raises(paths.DurabilityError):
        paths.durable(rel, mkparents=True)
    _clear_ignore_cache()


@pytest.mark.parametrize("rel,tail", [
    (c1i.INTAKE_REL, "data/emission/campaign1/intake.json"),
    (li2.INTAKE_REL, "data/emission/library_intake_2/intake.json"),
])
def test_real_paths_not_gitignored(rel, tail):
    _clear_ignore_cache()
    assert str(paths.durable(rel)).replace("\\", "/").endswith(tail)


def test_readers_point_at_the_durable_write_path():
    """No drift between where the writers land and where the readers look."""
    assert dsch.INTAKE_ARTIFACT == c1i.INTAKE_JSON            # scheduler library seed
    assert li2.C1_INTAKE_JSON == c1i.INTAKE_JSON              # library_intake_2 cross-ref
    assert sfr.C1_INTAKE == c1i.INTAKE_JSON                   # first-release union (campaign1)
    assert sfr.I2_INTAKE == li2.INTAKE_JSON                   # first-release union (intake_2)


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


@pytest.mark.parametrize("rel", [
    "data/emission/campaign1/intake.json",
    "data/emission/library_intake_2/intake.json",
])
def test_disk_audit_forces_never(rel):
    assert da.classify(rel).category == da.NEVER
