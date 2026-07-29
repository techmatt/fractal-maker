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


def test_bulk_fields_stay_scratch():
    """Only the snapshot is durable; the regenerable field/emb bulk stays under scratch/, so
    the split is real and we did not promote a file-count bomb into the tree."""
    assert "scratch" in str(dsch.INTAKE_EMB_DIR).replace("\\", "/")
    assert "scratch" in str(sfr.C1_DIR).replace("\\", "/")
    assert "scratch" in str(sfr.I2_DIR).replace("\\", "/")


@pytest.mark.parametrize("rel", [
    "data/emission/campaign1/intake.json",
    "data/emission/library_intake_2/intake.json",
])
def test_disk_audit_forces_never(rel):
    assert da.classify(rel).category == da.NEVER
