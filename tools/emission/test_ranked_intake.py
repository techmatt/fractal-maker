#!/usr/bin/env python
"""`tools/emission/ranked_intake.py` — the read-time ranked intake (2026-08-09).

Four properties, each the mirror of something the frozen-verdict intake did:

  1. the DECODE-VERSION predicate is gone — a stale-stamped row is admitted on its raw score,
     where `load_admitted` would drop it. Asserted against `load_admitted` on the SAME file, so
     the difference is demonstrated rather than described.
  2. the stored `decoded_class` is not read at all — a class-1 row with a strong `p_good` is
     admitted, and a class-3 row with a junk `p_good` is not.
  3. the junk floor is the ONE cut, with the floor-admit bypass intact.
  4. guard ∧ distinct still admit, because neither is a head verdict.

Plus the supply arithmetic §3 hangs off (`emit_cap`, `partition_slots`), which is where the
thin-supply rule's zero comes from.

  uv run pytest tools/emission/test_ranked_intake.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_common as cc                       # noqa: E402
import release_mix as RM                         # noqa: E402
from tools.emission import descriptor as D       # noqa: E402
from tools.emission import floors as F           # noqa: E402
from tools.emission import ranked_intake as RI   # noqa: E402


def _row(rid, *, p_good=0.7, ver=None, dc=3, guard=True, distinct=True,
         family="mandelbrot", src="steered", cx=None):
    if cx is None:
        cx = -0.5 - sum(ord(c) for c in rid) * 1e-6
    return {"id": rid, "family": family, "outcome_cx": cx, "outcome_cy": 0.1,
            "outcome_fw": 0.03, "decoded_class": dc, "p_good": p_good,
            "guard_pass": guard, "distinct": distinct, "mix_source": src,
            "scorer_version": cc.active_scorer_version() if ver is None else ver}


def _ledger(tmp_path, rows, name="outcome_ledger.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# 1 + 2. no decode-version predicate, no stored class
# --------------------------------------------------------------------------- #
def test_a_stale_stamped_row_is_admitted_where_the_old_intake_dropped_it(tmp_path):
    """THE change. `load_admitted` refuses a v6-stamped row because its `decoded_class` is a
    verdict from a head that is not live; the ranked intake reads the stored PROBABILITY,
    which is a number and does not go stale into unusability — it just becomes an older head's
    number, which is what a rank degrades to.

    Both sides asserted on one file: the old reader's rejection is what makes the new reader's
    admission a change rather than a restatement."""
    led = _ledger(tmp_path, [_row("cur"), _row("old", ver="v6"), _row("older", ver="v5")])
    assert [r["id"] for r in D.load_admitted(led)] == ["cur"]        # unchanged old path
    ranked, diag = RI.ranked_by_partition([led])
    assert sorted(r["_ledger_row_id"] for r in ranked["mandelbrot"]) == ["cur", "old", "older"]
    assert diag["n_passing"] == 3


def test_the_stored_decoded_class_is_not_read(tmp_path):
    """A class-1 row with a strong raw score is IN; a class-3 row with a junk score is OUT.
    The stored class is `corn_decode` against a per-partition `t_good` frozen at harvest, and
    reading it is exactly the frozen enforcing state this path removed."""
    led = _ledger(tmp_path, [_row("strong_class1", dc=1, p_good=0.93),
                             _row("junk_class3", dc=3, p_good=0.05)])
    ranked, _ = RI.ranked_by_partition([led])
    assert [r["_ledger_row_id"] for r in ranked["mandelbrot"]] == ["strong_class1"]


# --------------------------------------------------------------------------- #
# 3. the junk floor, and the floor-admit bypass
# --------------------------------------------------------------------------- #
def test_the_junk_floor_is_the_one_cut_and_it_is_inclusive(tmp_path):
    led = _ledger(tmp_path, [_row("at", p_good=F.JUNK_FLOOR),
                             _row("under", p_good=F.JUNK_FLOOR - 1e-6),
                             _row("over", p_good=0.9),
                             _row("unscored", p_good=None)])
    ranked, diag = RI.ranked_by_partition([led])
    assert [r["_ledger_row_id"] for r in ranked["mandelbrot"]] == ["over", "at"]
    assert diag["mined_by_partition"]["mandelbrot"] == 4      # the pre-floor denominator
    assert diag["passing_by_partition"]["mandelbrot"] == 2


@pytest.mark.parametrize("src", sorted(D.FLOOR_ADMIT_SOURCES))
def test_a_floor_admit_row_bypasses_the_junk_floor(tmp_path, src):
    """A `q4_harvest` / `human_q3plus` row was selected by a signal ORTHOGONAL to the head —
    the q4 goodness field, or Matt's own 3/4 — so the head must not veto it at 0.20 any more
    than it could at the deleted `FLOOR_PNOTBAD` 0.5 (retired.md, 2026-08-04). It is admitted,
    ranked with everything else, and COUNTED as a bypass so a partition's supply never
    silently becomes "however many humans labelled"."""
    led = _ledger(tmp_path, [_row("machine_junk", p_good=0.05),
                             _row("human_junk", p_good=0.05, src=src)])
    ranked, diag = RI.ranked_by_partition([led])
    assert [r["_ledger_row_id"] for r in ranked["mandelbrot"]] == ["human_junk"]
    assert diag["bypass_by_partition"] == {"mandelbrot": 1}


# --------------------------------------------------------------------------- #
# 4. guard and distinct are not head verdicts and still admit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [{"guard": False}, {"distinct": False}])
def test_guard_and_distinct_still_cut(tmp_path, kw):
    led = _ledger(tmp_path, [_row("ok", p_good=0.9), _row("bad", p_good=0.99, **kw)])
    ranked, diag = RI.ranked_by_partition([led])
    assert [r["_ledger_row_id"] for r in ranked["mandelbrot"]] == ["ok"]
    assert diag["mined_by_partition"]["mandelbrot"] == 1      # not even MINED — pre-floor


def test_a_floor_admit_row_does_not_bypass_the_guard(tmp_path):
    """NON-VACUITY for the bypass: it is a bypass of the HEAD's verdict only. A blank or
    all-interior render is not a wallpaper whoever labelled it."""
    led = _ledger(tmp_path, [_row("h", p_good=0.99, src="human_q3plus", guard=False)])
    ranked, _ = RI.ranked_by_partition([led])
    assert ranked == {}


# --------------------------------------------------------------------------- #
# the rank itself
# --------------------------------------------------------------------------- #
def test_rank_is_raw_p_ge3_descending_tie_broken_on_id(tmp_path):
    led = _ledger(tmp_path, [_row("b", p_good=0.5), _row("a", p_good=0.5),
                             _row("top", p_good=0.9)])
    ranked, _ = RI.ranked_by_partition([led])
    assert [r["_ledger_row_id"] for r in ranked["mandelbrot"]] == ["top", "a", "b"]


def test_partitions_are_cell_identity_not_the_family_token(tmp_path):
    """`phoenix:classic` gets its own ranked list. Keying on `row["family"]` would fold it
    into `phoenix`, and its release share and its supply line would go with it."""
    led = _ledger(tmp_path, [_row("ph", family="phoenix", p_good=0.8),
                             _row("cl", family="phoenix", p_good=0.7, cx=-0.6)])
    ranked, _ = RI.ranked_by_partition([led])
    # both rows are axis-free phoenix -> the pinned Ushiki point -> phoenix:classic
    assert set(ranked) == {"phoenix:classic"}
    assert len(ranked["phoenix:classic"]) == 2


def test_supply_lines_name_every_partition_including_the_empty_ones(tmp_path):
    """The sheet's one-liner. A partition whose whole supply fell below the floor must still
    appear — a partition that vanishes from the readout when its supply dies is the failure
    the mined count exists to prevent."""
    led = _ledger(tmp_path, [_row("m1", p_good=0.9), _row("m2", p_good=0.9),
                             _row("p1", family="phoenix", p_good=0.01),
                             _row("p2", family="phoenix", p_good=0.02)])
    _ranked, diag = RI.ranked_by_partition([led])
    lines = RI.supply_lines(diag)
    assert "mandelbrot: 2 mined, 2 above floor → emits 0 (thin supply)" in lines
    assert "phoenix:classic: 2 mined, 0 above floor → emits 0 (thin supply)" in lines


def test_the_supply_census_reports_both_halves_over_the_SAME_scope(tmp_path):
    """Both counts come from one scoped walk. Taking `mined` from the ledgers and `passing`
    from whatever subset a run actually serves prints two denominators on one line — the sheet
    said "81 mined, 18 above floor" for a partition whose served numbers were 33 and 18,
    because an intake SNAPSHOT had restricted the run and only one of the two counts knew."""
    rows = [_row("keep_a", p_good=0.9), _row("keep_b", p_good=0.05),
            _row("out_a", p_good=0.9), _row("out_b", p_good=0.05)]
    led = _ledger(tmp_path, rows)
    mined_rows, _ = RI.load_mined([led])
    scope = {r["id"] for r in mined_rows if r["_ledger_row_id"].startswith("keep")}
    m_all, p_all = RI.supply_census(mined_rows, None)
    m_scoped, p_scoped = RI.supply_census(mined_rows, scope)
    assert (m_all["mandelbrot"], p_all["mandelbrot"]) == (4, 2)
    assert (m_scoped["mandelbrot"], p_scoped["mandelbrot"]) == (2, 1)


# --------------------------------------------------------------------------- #
# §3 supply arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("supply,cap", [(0, 0), (3, 0), (4, 1), (7, 1), (8, 2), (40, 10)])
def test_emit_cap_is_floor_supply_over_four(supply, cap):
    assert RI.emit_cap(supply) == cap


def test_partition_slots_are_near_proportional_to_the_release_mix():
    """The allocation is `release_mix` through `apportion.sequence_by_deficit`, so a big N
    lands near the shares and a truncating N stays readable. Asserted as the property (each
    partition within one of its proportional share) rather than as a frozen vector, which
    would re-derive the allocation through the code under test."""
    parts = sorted(RM.RATIO)
    shares = RM.shares(parts)
    for n in (12, 24, 100):
        slots = RI.partition_slots(shares, n)
        assert sum(slots.values()) == n
        for p, k in slots.items():
            assert abs(k - shares[p] * n) <= 1.0, (n, p, k, shares[p] * n)


def test_partition_slots_handles_the_degenerate_ends():
    shares = RM.shares(["mandelbrot", "phoenix"])
    assert sum(RI.partition_slots(shares, 0).values()) == 0
    assert set(RI.partition_slots(shares, 0)) == {"mandelbrot", "phoenix"}   # keys survive
    assert RI.partition_slots({}, 5) == {}
    # one slot goes to the biggest share, not to whatever sorts first
    assert RI.partition_slots(shares, 1) == {"mandelbrot": 1, "phoenix": 0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
