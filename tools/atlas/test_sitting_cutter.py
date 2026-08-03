"""The sitting cutter: three NON-OPTIONAL stages, each proved red by injection.

Every stage here exists because its absence cost a real sitting real keystrokes, so each is
tested twice — once that it fires, and once that it does NOT fire on the population it must
leave alone. A filter that removes everything passes the first test and fails the second.

  uv run pytest tools/atlas/test_sitting_cutter.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sitting_cutter as sc      # noqa: E402
import supply_routing as srt     # noqa: E402


def _row(**kw):
    base = dict(partition="julia:mandelbrot", rank_tier=2, rank_score=1.0,
                cx="0", cy="0", fw="1.0", fate="admitted", int_frac=0.1,
                canon_decoded=3)
    base.update(kw)
    return base


def _unit(i, d=8):
    v = np.zeros(d, dtype=np.float32)
    v[i % d] = 1.0
    return v


def _embed_by_key(mapping):
    return lambda r: mapping.get(r.get("cx"))


# =========================================================================== #
# (a) interior > 0.30 — auto-labelled, never presented
# =========================================================================== #
def test_interior_over_the_threshold_is_auto_labelled_and_removed():
    kept, removed, rep = sc.stage_interior([_row(cx="a", int_frac=0.31),
                                            _row(cx="b", int_frac=0.10)], {})
    assert [r["cx"] for r in kept] == ["b"]
    assert [r["cx"] for r in removed] == ["a"]
    al = removed[0]["auto_label"]
    assert al["score"] == 1 and al["rule_id"] == "interior_gt30_v1"
    assert al["labeler"].startswith("rule:")
    assert rep["disposition"].endswith("NEVER presented")


def test_the_interior_comparison_is_strict_so_exactly_030_is_shown():
    """The boundary side is invisible in a count and mirrors `present.rs`'s strict `<` on the
    other side of the same number. A `>=` here would silently delete a whole band."""
    kept, removed, _ = sc.stage_interior([_row(cx="lo", int_frac=0.2999),
                                          _row(cx="eq", int_frac=0.30),
                                          _row(cx="hi", int_frac=0.3001)], {})
    assert [r["cx"] for r in kept] == ["lo", "eq"]
    assert [r["cx"] for r in removed] == ["hi"]


def test_an_unmeasured_interior_is_kept_and_counted_apart():
    """An absent measure is not a high one — `apply_interior_rule.fires`'s own rule."""
    kept, removed, rep = sc.stage_interior([_row(cx="none", int_frac=None)], {})
    assert len(kept) == 1 and not removed and rep["unmeasured_kept"] == 1


def test_the_interior_rule_is_the_SAME_rule_the_label_store_applies():
    """Same id, same threshold, same comparison, imported rather than restated — a second
    literal 0.30 in this tree is how the two drift."""
    import apply_interior_rule as air
    assert sc.INTERIOR_RULE_ID == air.RULE_ID == "interior_gt30_v1"
    assert sc.INTERIOR_THRESHOLD == air.THRESHOLD == 0.30


# =========================================================================== #
# (c) per-partition machine-1 auto-discard
# =========================================================================== #
@pytest.mark.parametrize("part,discarded", [("multibrot3", True), ("multibrot4", True),
                                            ("multibrot5", True), ("phoenix", True),
                                            ("julia:mandelbrot", False),
                                            ("mandelbrot", False)])
def test_machine_1_discard_follows_the_measured_partition_table(part, discarded):
    """The measurement is partition-dependent and the pooled 68.9% is not a decision.
    julia:mandelbrot must survive: 16.5% of its machine-1s are >=3."""
    kept, removed, _ = sc.stage_machine_1([_row(partition=part, canon_decoded=1)], {})
    assert bool(removed) is discarded
    assert bool(kept) is (not discarded)


def test_a_machine_2_or_better_is_never_discarded():
    """The vacuity guard: a stage that discarded every native-multibrot row would pass the
    parametrize above."""
    for dec in (2, 3, 4):
        kept, removed, _ = sc.stage_machine_1(
            [_row(partition="multibrot4", canon_decoded=dec)], {})
        assert kept and not removed, dec


def test_a_cheap_only_row_has_no_machine_1_verdict_to_act_on():
    """A `rank_tier=1` score comes off a 384x216 ss1 render; every P(Matt=1 | decoded 1) rate
    was measured against the 640x360 ss2 canonical decode. Discarding on the cheap score
    would be the cap/geometry error, so a tier-1 row survives whatever its flag says."""
    kept, removed, rep = sc.stage_machine_1(
        [_row(partition="multibrot4", rank_tier=1, canon_decoded=None)], {})
    assert kept and not removed
    assert rep["no_canonical_verdict_kept"]["multibrot4"] == 1


def test_an_unmeasured_partition_fails_closed_to_keep():
    kept, removed, _ = sc.stage_machine_1(
        [_row(partition="julia:multibrot4", canon_decoded=1)], {})
    assert kept and not removed
    assert srt.MACHINE_1_DISCARD["julia:multibrot4"] is False


# =========================================================================== #
# (b) presentation-level morph dedup
# =========================================================================== #
def test_morph_dedup_keeps_one_row_per_look_best_first():
    e = {"a": _unit(0), "b": _unit(0) * 0.999 + _unit(1) * 0.02, "c": _unit(3)}
    rows = [_row(cx="a"), _row(cx="b"), _row(cx="c")]
    kept, removed, rep = sc.stage_morph_dedup(rows, dict(embed=_embed_by_key(e)))
    assert [r["cx"] for r in kept] == ["a", "c"]
    assert [r["cx"] for r in removed] == ["b"] and removed[0]["dup_cos"] >= 0.974
    assert rep["looks_kept"] == 2


def test_morph_dedup_is_first_wins_so_the_incoming_rank_is_the_policy():
    e = {"top": _unit(0), "dup": _unit(0)}
    kept, _r, _ = sc.stage_morph_dedup([_row(cx="top"), _row(cx="dup")],
                                       dict(embed=_embed_by_key(e)))
    assert [r["cx"] for r in kept] == ["top"]
    kept2, _r2, _ = sc.stage_morph_dedup([_row(cx="dup"), _row(cx="top")],
                                         dict(embed=_embed_by_key(e)))
    assert [r["cx"] for r in kept2] == ["dup"]


def test_distinct_looks_are_not_thinned():
    """The vacuity guard. A dedup that dropped everything after the first row would pass the
    test above."""
    e = {str(i): _unit(i) for i in range(6)}
    rows = [_row(cx=str(i)) for i in range(6)]
    kept, removed, _ = sc.stage_morph_dedup(rows, dict(embed=_embed_by_key(e)))
    assert len(kept) == 6 and not removed


def test_an_unembeddable_row_is_kept_and_counted_not_treated_as_a_duplicate():
    e = {"a": _unit(0)}
    kept, removed, rep = sc.stage_morph_dedup([_row(cx="a"), _row(cx="unreachable")],
                                              dict(embed=_embed_by_key(e)))
    assert len(kept) == 2 and not removed and rep["unembeddable_kept"] == 1


def test_a_raising_embedder_costs_the_dedup_verdict_not_the_row():
    def boom(r):
        raise RuntimeError("no field")
    kept, removed, rep = sc.stage_morph_dedup([_row(cx="a")], dict(embed=boom))
    assert len(kept) == 1 and not removed and rep["unembeddable_kept"] == 1


def test_a_missing_embedder_is_a_HARD_failure_never_a_silent_skip():
    """The dedup is not optional. A `ctx` with no embedder must raise, not pass everything
    through — a stage that degrades to a no-op is a stage that will be a no-op on the run
    that needed it."""
    with pytest.raises(ValueError, match="NOT optional"):
        sc.stage_morph_dedup([_row()], {})


def test_the_dedup_threshold_is_the_library_knee():
    assert sc.NEAR_DUP_COS == srt.NEAR_DUP_COS == 0.974


# =========================================================================== #
# the pipeline: non-optional, accounted, capped
# =========================================================================== #
def test_all_three_stages_are_in_the_pipeline_and_there_is_no_way_to_skip_one():
    names = [f.__name__ for f in sc.STAGES]
    assert names == ["stage_interior", "stage_machine_1", "stage_morph_dedup"]
    import inspect
    src = inspect.getsource(sc.cut_sitting)
    assert "for fn in STAGES:" in src
    # no conditional guards the loop body — every stage runs on every cut
    assert "if " not in src.split("for fn in STAGES:")[1].split("sitting, cells")[0]


def test_the_expensive_stage_runs_last():
    """(a) and (c) are free column reads; (b) needs a render and a CLIP pass per row.
    Reversing them would be correct and would pay a morph field for every row the other two
    were about to delete."""
    assert sc.STAGES[-1] is sc.stage_morph_dedup


def test_the_cut_accounts_for_every_row_it_was_given():
    e = {str(i): _unit(i) for i in range(20)}
    rows = ([_row(cx=str(i), partition="multibrot4", canon_decoded=1) for i in range(3)]
            + [_row(cx=str(i), int_frac=0.9) for i in range(3, 6)]
            + [_row(cx=str(i)) for i in range(6, 20)])
    res = sc.cut_sitting(rows, max_rows=5, embed=_embed_by_key(e))
    rep = res["report"]
    assert rep["n_in"] == 20 and rep["n_sitting"] == 5
    removed = sum(len(v) for v in res["removed"].values())
    assert rep["n_in"] == rep["n_sitting"] + removed + rep["n_over_cap"]
    # Each stage removed the population it owns, THROUGH the pipeline — not merely when
    # called directly. Without this, a stage silently dropped from `STAGES` still passes
    # every one of its own unit tests.
    assert len(res["auto_labeled"]) == 3                       # interior
    assert len(res["removed"]["machine_1_discard"]) == 3       # native multibrot machine-1s
    assert set(sc.STAGES) == {sc.stage_interior, sc.stage_machine_1, sc.stage_morph_dedup}


def test_a_cut_that_lost_a_row_would_exit_loud(monkeypatch):
    """The accounting identity is an assertion, not a report line. Proved by injecting a
    stage that eats a row without naming it."""
    def leaky(rows, ctx):
        return rows[:-1], [], dict(stage="leaky", removed=0)
    monkeypatch.setattr(sc, "STAGES", (leaky,))
    with pytest.raises(AssertionError, match="does not balance"):
        sc.cut_sitting([_row(cx="a"), _row(cx="b")], max_rows=10)


def test_the_sitting_is_capped_at_one_page():
    assert sc.MAX_ROWS == 1000
    e = {str(i): _unit(i, d=40) for i in range(40)}
    rows = [_row(cx=str(i)) for i in range(40)]
    res = sc.cut_sitting(rows, max_rows=7, embed=_embed_by_key(e))
    assert res["report"]["n_sitting"] == 7 and res["report"]["n_over_cap"] == 33


def test_the_cut_balances_across_partition_and_tier_cells():
    """One page, so a cell with hundreds of rows must not own it."""
    e = {str(i): _unit(i, d=64) for i in range(60)}
    rows = ([_row(cx=str(i), partition="julia:mandelbrot") for i in range(50)]
            + [_row(cx=str(i), partition="phoenix") for i in range(50, 60)])
    res = sc.cut_sitting(rows, max_rows=10, embed=_embed_by_key(e))
    got = res["report"]["by_partition"]
    assert got == {"julia:mandelbrot": 5, "phoenix": 5}


def test_the_cli_never_serves_a_sitting():
    """v2 builds and dry-runs the cutter; it does not cut one. Pinned so a later
    `--apply` has to be a deliberate edit here as well as there."""
    import inspect
    src = inspect.getsource(sc.main)
    assert '"dry-run"' in src
    assert "--apply" not in src
    for banned in ("write_batch", "blind.jsonl", "images.jsonl"):
        assert banned not in src, banned
