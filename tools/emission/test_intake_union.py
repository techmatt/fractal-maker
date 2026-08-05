"""The stage-2 intake UNION, against the live seven-ledger census.

WHY THIS FILE. `_load_all_admitted` used to abort the whole emission stage on 11 run-scoped
id collisions between campaign1 and campaign2 (`st_<fam>_<arm>_<seq>` minted per campaign,
reused for DIFFERENT locations), and the only offered fix was `stage_first_release`'s
`c1__`-prefixed ledger COPIES — which it wrote to `scratch/` and which are gone. The union is
now namespaced at the reader (`descriptor.load_union_admitted`), and this file pins what that
buys, in one number, against the ledgers actually on disk.

The number is stated ONCE (`UNION_ADMITTED`) and cross-checked against the per-ledger census
rather than written twice, so a real change in the ledgers fails with an arithmetic story
attached ("per-ledger sum moved") instead of a bare mismatch.

  uv run pytest tools/emission/test_intake_union.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.emission import descriptor as D          # noqa: E402
from tools.emission import ledger_rescore as LR     # noqa: E402

# The live census, 2026-08-04, head v10 (`uv run python tools/emission/ledger_rescore.py
# status`). 700 admitted rows over the seven intake ledgers, 0 cross-ledger same-location
# overlaps, 11 bare-id collisions that the namespacing keeps apart. The PRE-fix union was 689 —
# it dropped exactly those 11 distinct locations before aborting on them.
UNION_ADMITTED = 700
ID_COLLISIONS = 11
LOCATION_OVERLAPS = 0

LEDGERS = [LR.ledger_path(rel) for _tag, rel in LR.LEDGERS]
_MISSING = [str(p) for p in LEDGERS if not p.exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"intake ledger absent: {_MISSING}")


@pytest.fixture(scope="module")
def union():
    return D.load_union_admitted(LEDGERS)


def test_the_union_is_reachable_and_matches_the_per_ledger_census(union):
    """THE un-abort: the union resolves, and its size is the per-ledger admitted sum minus the
    genuine same-location overlaps. Nothing is dropped for having a colliding id."""
    rows, diag = union
    per_ledger = sum(len(D.load_admitted(p)) for p in LEDGERS)
    assert diag["n_union"] == per_ledger - diag["n_location_overlaps"]
    assert diag["n_union"] == len(rows) == UNION_ADMITTED
    assert diag["n_location_overlaps"] == LOCATION_OVERLAPS


def test_the_run_scoped_collisions_are_still_counted(union):
    """The count is the evidence the namespacing is doing work. If it silently went to 0 the
    union number would look the same while meaning something else."""
    _rows, diag = union
    assert diag["n_id_collisions"] == ID_COLLISIONS
    assert diag["collision_sample"]


def test_every_union_id_is_unique_and_carries_its_ledger(union):
    rows, _diag = union
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for r in rows:
        assert r["id"] == D.namespaced_id(r["_ledger_ns"], r["_ledger_row_id"])
        assert r["_source_ledger"].endswith(".jsonl")


def test_the_colliding_ids_resolve_to_distinct_locations(union):
    """Non-vacuity for the fix: the rows sharing a bare id really are different wallpapers, so
    an id-keyed union was losing supply, not deduplicating it."""
    rows, _diag = union
    by_bare: dict = {}
    for r in rows:
        by_bare.setdefault(r["_ledger_row_id"], []).append(D.loc_key(r))
    shared = {k: v for k, v in by_bare.items() if len(v) > 1}
    assert len(shared) == ID_COLLISIONS
    for bare, keys in shared.items():
        assert len(set(keys)) == len(keys), f"{bare} aliased two rows at one location"


def test_the_census_and_the_driver_read_the_same_union(union):
    """`ledger_rescore status` reports what stage 2 intakes because it calls the same reader —
    not a mirror of it. A mirror is how "the intake admits N" became false."""
    _rows, diag = union
    u = LR.intake_union()
    assert u["n_union"] == diag["n_union"]
    assert u["n_collisions"] == diag["n_id_collisions"]
    assert u["n_benign_overlaps"] == diag["n_location_overlaps"]


def test_every_admitted_row_resolves_to_a_registered_partition(union):
    """The cell axis has to be able to key every row it will be handed."""
    rows, _diag = union
    from partitions import ALL_FAMS
    parts = {D.cell_partition(r) for r in rows}
    assert parts <= set(ALL_FAMS)
    assert "phoenix:classic" in parts, "the classic supply is in the union and must be keyed"
