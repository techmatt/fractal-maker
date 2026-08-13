"""The stage-2 intake UNION, against the live ten-ledger census.

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
from tools.emission import floors as F              # noqa: E402  THE cut owner

# The live census, 2026-08-12, head v11. 5,716 admitted rows over the FOURTEEN intake ledgers,
# 0 cross-ledger same-location overlaps, 16 bare-id collisions that the namespacing keeps
# apart. Per ledger: c1_breadth 149, c1_dive 140, c2_breadth 157, c2_dive 157,
# phoenix_grid 147, classic_phoenix 22, q4_harvest 108, prod25_breadth 1867,
# prod25_dive 103, prod25_phoenix 17, prod26_breadth 1418, prod26_dive 86,
# prod27_breadth 1143, prod27_dive 202.
#
# 4,371 -> 5,716 (+1,345) ON 2026-08-12 (prompts/run27_launch.md, the closing step), and like
# the run-26 move below it is ONE change: production run 27's two legs joined
# `ledger_rescore.LEDGERS`, breadth 1,143 + dive 202. Run 27 had already emitted from them by
# passing them on the command line alongside the twelve, so this registration changes no
# number the run reported — it makes the fourteen-ledger union the DEFAULT the next run gets
# without an argument. Every row is natively v11: no re-score, no render, no overlay. The
# run's phoenix supply came through `classic_phoenix`, already listed, which is why this is
# two legs and not three. The id-collision count is unchanged at 16 and the overlap count is
# still 0, so 1,345 new rows added no cross-ledger same-location duplicate.
#
# 2,867 -> 4,371 (+1,504) ON 2026-08-12 (prompts/run26_followup.md step 1), and unlike the
# move below it is ONE change: production run 26's two legs joined `ledger_rescore.LEDGERS`,
# breadth 1,418 + dive 86. Run 26 had already emitted from them by passing them on the command
# line alongside the ten, so this registration changes no number the run reported — it makes
# the twelve-ledger union the DEFAULT the next run gets without an argument. Every row is
# natively v11: no re-score, no render, no overlay. The run's phoenix supply came through
# `classic_phoenix`, already listed, which is why this is two legs and not three.
#
# 881 -> 2,867 (+1,986) ON 2026-08-10 (prompts/sittings_27.md step 0), and it is TWO changes
# in one edit, which is why both numbers are stated:
#   +1,987  production run 25's three legs joined `ledger_rescore.LEDGERS`. The run emitted
#           from the seven-ledger union verbatim, so its own night of discovery had never
#           reached an intake. Every row is natively v11 — no re-score, no render.
#      -1   the redundant `classic_phoenix/outcome_ledger.rescored_v11.jsonl` overlay was
#           DELETED in the same commit. It re-reported 24 of the 33 fresh classic rows at the
#           OLD render's probabilities, and one of those rows was distinct-and-admitted only
#           under the stale numbers: classic_phoenix 23 -> 22. The native ledger is the
#           record; the overlay was a second one saying something slightly different.
#
# 862 -> 881 (+19) ON 2026-08-10, and unlike every move below it IS a corpus change:
# production_run_25's classic-phoenix supply leg re-minted all 184 legacy Ushiki coords under
# v11 (`tools/phoenix/classic_phoenix_supply.py`; the ledger's 184 rescored rows were all
# stamped v10, so `purge_stale` re-scored rather than resumed). classic_phoenix 4 -> 23 is the
# whole delta — 33 rows clear GOOD_FLOOR and 23 of those are distinct looks. No other ledger
# moved: the run's own breadth/dive/phoenix-native ledgers are NOT among the seven.
#
# 779 -> 862 (+83) ON 2026-08-09, and it is a PREDICATE change, not a corpus change: not a row
# was added. `load_admitted` cut on `is_current_decoded ∧ decoded_class >= 3` — a frozen class
# against a per-partition `t_good`, plus a version firewall — and now cuts on the row's raw
# P(>=3) against the flat `floors.GOOD_FLOOR` (prompts/selection_restructure_3.md). Both halves
# push the same way here: the four partitions whose t_good sat above 0.50 loosen, and rows an
# older head stamped stop being discarded. `classic_phoenix` 0 -> 4 is the visible end of it
# (its rows decoded class 1-2 under the old rule and four of them clear 0.50).
#
# Earlier: 751 (v10) -> 779 (v11), the head and the per-partition table moving together;
# 700 -> 751 on 2026-08-04 when the v7-era badness floor was deleted from the floor-admit path
# (emission_floors_prompt.md §B), `q4_harvest` 57 -> 108 of its 108 guard-passing rows.
#
# THIS PIN IS SUPPOSED TO BREAK ON A RE-SCORE. It is a census — "what the union IS" — and its
# job is to make a change in the intake population an explicit edit. The standing ALIVE check
# is the relational floor in test_liveness_census.py, which a legitimate re-score leaves green.
UNION_ADMITTED = 5716
Q4_HARVEST_ADMITTED = 108        # the floor-admit ledger, whole (guard ∧ distinct)
ID_COLLISIONS = 16
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


def test_the_floor_admit_ledger_admits_whole(union):
    """§B, in the one number it moved: `q4_harvest` is a floor-admit source, so its admitted
    count IS its guard∧distinct count — no machine quality cut applies to it at all, not the
    v7-era badness floor (deleted 2026-08-04) and not `floors.GOOD_FLOOR` (the bypass). It was
    57/108 under that badness floor read on the v10 scale, and 75/108 when the floor was set
    under v7: a cut that moved by 18 rows with nobody deciding it, which is what an unstamped
    floor does.

    NON-VACUITY matters more than it used to: the bypass now has to hold against a cut that
    would genuinely bite, so the count of rows BELOW the good floor is asserted non-zero."""
    rows, diag = union
    q4 = [p for p in LEDGERS if "q4_harvest" in p.as_posix()]
    assert len(q4) == 1, q4
    resolved = D.resolve_rows(q4[0])
    eligible = [r for r in resolved if D.guard_and_distinct(r)]
    assert len(D.load_admitted(q4[0])) == len(eligible) == Q4_HARVEST_ADMITTED
    n_below = sum(1 for r in eligible if not F.passes_good_floor(r.get("p_good")))
    assert n_below > 0, "the bypass is vacuous if every floor-admit row clears the floor anyway"
    assert sum(1 for r in rows if r["_source_ledger"].endswith(
        "q4_harvest/outcome_ledger.jsonl")) == Q4_HARVEST_ADMITTED
    # ...and it is the ONLY floor-admit ledger in the union, so the +51 is attributable.
    floor_admit = {lbl for lbl, r in ((r["_source_ledger"], r) for r in rows)
                   if D.source_tag_of(r) in D.FLOOR_ADMIT_SOURCES}
    assert floor_admit == {"data/emission/q4_harvest/outcome_ledger.jsonl"}


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


def test_the_classic_split_is_still_resolvable_off_a_row(union):
    """The `phoenix:classic` KEY, checked without depending on the union containing one.

    This used to be `assert "phoenix:classic" in parts` on the union above — which asserted
    two things at once and broke on the weaker of them: at the v11 flip the classic ledger's
    24 rows stopped reaching q3, so the partition left the union and the RESOLVER assertion
    went red for a SUPPLY fact (test_liveness_census.KNOWN_EMPTY carries that one, with its
    cause and remedy). What has to hold here is that `cell_partition` can still tell classic
    from varied phoenix when handed a classic row — the thing that would silently mis-key the
    cell axis — so it is asserted against a row built for the purpose, and it stays true
    whether or not any classic row is currently admissible."""
    import partitions as P
    axes = dict(zip(("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"),
                    P.PHOENIX_CLASSIC_POINT))
    row = {"family": "phoenix", "fractal_type": "phoenix",
           **{f"phoenix_{k}": v for k, v in axes.items()}}
    assert D.cell_partition(row) == P.CLASSIC_PHOENIX, (
        "a row at the pinned Ushiki point does not resolve to phoenix:classic — the cell axis "
        "would pool the classic supply into varied phoenix")
