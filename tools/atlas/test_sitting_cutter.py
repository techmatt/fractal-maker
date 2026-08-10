"""The sitting cutter: three NON-OPTIONAL stages, each proved red by injection.

Every stage here exists because its absence cost a real sitting real keystrokes, so each is
tested twice — once that it fires, and once that it does NOT fire on the population it must
leave alone. A filter that removes everything passes the first test and fails the second.

  uv run pytest tools/atlas/test_sitting_cutter.py -q
"""
from __future__ import annotations

import json
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
    import textwrap
    src = textwrap.dedent(inspect.getsource(sc.cut_sitting))
    lines = src.splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.strip() == "for fn in STAGES:")
    indent = len(lines[head]) - len(lines[head].lstrip())
    # THE LOOP BODY, by indentation — not "everything up to the next landmark". The text-span
    # form went red on 2026-08-04 for an `if` that was AFTER the loop (the calibration
    # reservation), i.e. it was guarding a region rather than the thing it names.
    body = []
    for ln in lines[head + 1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        body.append(ln)
    assert [ln for ln in body if ln.strip()], "loop body did not parse — the guard is vacuous"
    # no conditional guards the loop body — every stage runs on every cut
    assert not any(ln.strip().startswith(("if ", "elif ", "try:", "continue"))
                   for ln in body), body


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


# =========================================================================== #
# the calibration reservation (Matt, 2026-08-04)
#
# A GENERAL rule keyed on labelled positives, not a phoenix:classic special case. The tests
# below are in two groups on purpose: the plan (arithmetic, injected counts) and the draw
# (what the sitting actually contains). The one test that reads the LIVE corpus is named as
# such — it is the only thing that can catch the rule being right and inert.
# =========================================================================== #
def test_the_sufficiency_floor_has_exactly_one_owner():
    """15, declared once. It used to be imported from `derive_t_good.MIN_POS` so a harness
    constant could not drift from the estimator's own gate; that estimator was deleted on
    2026-08-09 and this module inherited the number rather than growing a second copy of it
    (`verification_practice.md` §1.8). A re-typed 15 anywhere in the cutter goes red here."""
    assert sc.min_pos() == sc.MIN_POS == 15
    src = (ROOT / "tools/atlas/sitting_cutter.py").read_text(encoding="utf-8")
    assert src.count("MIN_POS = 15") == 1
    assert "derive_t_good import" not in src


def test_a_partition_below_the_floor_is_reserved_and_one_above_it_is_not():
    plan = sc.plan_reservations({"poor": 7, "rich": 626}, 1000, floor=15)
    assert plan == {"poor": 50}                     # 5% of the sitting
    assert "rich" not in plan


def test_the_rule_lapses_by_itself_when_a_partition_crosses_the_floor():
    """No list to edit: the qualifying set is recomputed from the census at every cut, so a
    partition that reaches MIN_POS simply stops being reserved."""
    assert sc.plan_reservations({"p": 14}, 1000, floor=15) == {"p": 50}
    assert sc.plan_reservations({"p": 15}, 1000, floor=15) == {}
    assert sc.plan_reservations({"p": 99}, 1000, floor=15) == {}


def test_multiple_qualifying_partitions_split_evenly_and_the_total_is_capped():
    """Three at 5% is exactly the 15% cap; four must SHRINK each share rather than drop one
    off the end, which would silently pick a favourite among equally starved families."""
    three = sc.plan_reservations({c: 0 for c in "abc"}, 1000, floor=15)
    assert three == {"a": 50, "b": 50, "c": 50}
    assert sum(three.values()) == 150
    four = sc.plan_reservations({c: 0 for c in "abcd"}, 1000, floor=15)
    assert set(four) == set("abcd") and len(set(four.values())) == 1
    assert sum(four.values()) <= int(sc.RESERVE_CAP_FRAC * 1000), four
    assert four == {c: 37 for c in "abcd"}          # truncated, not rounded to 38 (=152)


def test_a_sitting_too_small_to_reserve_a_row_records_a_ZERO_not_an_absence():
    """"nobody qualified" and "the reservation rounded away" are different facts, and only the
    second is a reason to look at the sitting size."""
    assert sc.plan_reservations({"p": 0}, 10, floor=15) == {"p": 0}
    assert sc.plan_reservations({"p": 99}, 10, floor=15) == {}


def _many_cell_population(n_other_parts=8, tiers=(1, 2, 3), per_cell=40, classic=20):
    """A realistic sitting population: several partitions x several rank tiers, plus one thin
    single-cell partition. The cell COUNT is the load-bearing property — see the test below."""
    rows, i = [], 0
    for p in ("mandelbrot", "julia:mandelbrot", "multibrot3", "multibrot4", "multibrot5",
              "julia:multibrot3", "julia:multibrot4", "julia:multibrot5")[:n_other_parts]:
        for t in tiers:
            for _ in range(per_cell):
                rows.append(_row(cx=str(i), partition=p, rank_tier=t))
                i += 1
    for _ in range(classic):
        rows.append(_row(cx=str(i), partition="phoenix:classic", rank_tier=2))
        i += 1
    return rows, {str(k): _unit(k, d=2048) for k in range(i)}


def test_the_reservation_is_a_FLOOR_the_partition_would_not_have_won():
    """The whole point, and the fixture is the real shape: `draw_balanced` round-robins over
    (partition x tier) CELLS, so a partition holding one cell out of C gets ~1/C of the sitting.
    At 25 cells that is 4%, under the 5% reservation — so the reservation binds and the
    partition gains rows it would not have won."""
    rows, e = _many_cell_population()
    bare = sc.cut_sitting(rows, max_rows=100, embed=_embed_by_key(e), reservations={})
    res = sc.cut_sitting(rows, max_rows=100, embed=_embed_by_key(e),
                         positives={"mandelbrot": 626, "phoenix:classic": 7})
    assert res["report"]["by_partition"]["phoenix:classic"] > \
        bare["report"]["by_partition"]["phoenix:classic"], (
            res["report"]["by_partition"], bare["report"]["by_partition"])
    active = res["report"]["calibration_reservations"]["active"]
    assert set(active) == {"phoenix:classic"}
    assert active["phoenix:classic"]["granted"] == 5     # 5% of 100
    assert active["phoenix:classic"]["shortfall"] == 0


def test_a_reservation_a_partition_did_not_need_is_a_FLOOR_and_not_a_BONUS():
    """Two statements, and the second is the scope of the first.

    FLOOR, NOT BONUS: the general fill continues the cell round-robin from the reserved rows
    (`draw_balanced(preseed=)`) rather than restarting at zero, so a partition the draw would
    have served generously anyway ends at `max(natural, reserved)`. The naive two-pass form
    gives it `natural + reserved` — measured here at 36 vs 33 of 100 before the preseed went in.

    SCOPE: the balanced draw is already egalitarian ACROSS CELLS, so a single-cell partition's
    unreserved share is ~1/C, which BEATS a 5% reservation whenever C < 1/RESERVE_FRAC = 20.
    The reservation is a real floor for a wide sitting (10 partitions x 3-4 tiers is 30-40
    cells) and inert for a narrow one. Asserting only "it binds" on a wide fixture would hide
    that; asserting both is what makes the wide result mean the mechanism, not the fixture."""
    rows, e = _many_cell_population(n_other_parts=2, tiers=(1,), per_cell=200, classic=50)
    kw = dict(max_rows=100, embed=_embed_by_key(e))
    bare = sc.cut_sitting(rows, reservations={}, **kw)
    res = sc.cut_sitting(rows, positives={"phoenix:classic": 7}, **kw)
    assert len(bare["report"]["cells"]) == 3 < int(1 / sc.RESERVE_FRAC)
    assert res["report"]["by_partition"] == bare["report"]["by_partition"]
    # ...and it is still RECORDED as active, so an inert reservation is legible, not absent.
    assert res["report"]["calibration_reservations"]["active"]["phoenix:classic"]["granted"] == 5


def test_an_UNFILLABLE_reservation_records_its_shortfall_and_never_fails_the_build():
    """The supply bound. A partition with fewer surviving rows than its reservation is exactly
    the run where refusing to cut a sitting helps nobody: the slot fills from elsewhere and the
    shortfall is the only trace, so it is recorded rather than raised."""
    e = {str(i): _unit(i, d=128) for i in range(103)}
    rows = ([_row(cx=str(i), partition="mandelbrot") for i in range(100)]
            + [_row(cx=str(i), partition="phoenix:classic") for i in range(100, 103)])
    res = sc.cut_sitting(rows, max_rows=100, embed=_embed_by_key(e),
                         positives={"mandelbrot": 626, "phoenix:classic": 7})
    got = res["report"]["calibration_reservations"]["active"]["phoenix:classic"]
    assert got == dict(reserved=5, granted=3, shortfall=2, available=3, capped_by_sitting=False)
    assert res["report"]["n_sitting"] == 100          # silently filled from elsewhere
    assert res["report"]["calibration_reservations"]["shortfall_total"] == 2


def test_a_reservation_for_a_partition_with_NO_rows_grants_nothing_and_says_so():
    e = {str(i): _unit(i, d=64) for i in range(30)}
    rows = [_row(cx=str(i), partition="mandelbrot") for i in range(30)]
    res = sc.cut_sitting(rows, max_rows=20, embed=_embed_by_key(e),
                         positives={"mandelbrot": 626, "phoenix:classic": 7})
    got = res["report"]["calibration_reservations"]["active"]["phoenix:classic"]
    assert got["available"] == 0 and got["granted"] == 0 and got["shortfall"] == 1
    assert res["report"]["n_sitting"] == 20


def test_the_accounting_still_balances_with_a_reservation_in_play():
    """The cut's audit identity is what makes every other number here readable, and a
    reserved slice is a second path rows take into the sitting."""
    e = {str(i): _unit(i, d=128) for i in range(60)}
    rows = ([_row(cx=str(i), partition="mandelbrot") for i in range(50)]
            + [_row(cx=str(i), partition="phoenix:classic") for i in range(50, 60)])
    res = sc.cut_sitting(rows, max_rows=20, embed=_embed_by_key(e),
                         positives={"mandelbrot": 626, "phoenix:classic": 7})
    rep = res["report"]
    removed = sum(len(v) for v in res["removed"].values())
    assert rep["n_in"] == rep["n_sitting"] + removed + rep["n_over_cap"]
    assert len({r["cx"] for r in res["sitting"]}) == rep["n_sitting"], "a row was drawn twice"


def test_the_manifest_records_which_reservations_were_active_and_what_each_GOT():
    e = {str(i): _unit(i, d=64) for i in range(30)}
    rows = [_row(cx=str(i), partition="mandelbrot") for i in range(30)]
    cr = sc.cut_sitting(rows, max_rows=20, embed=_embed_by_key(e),
                        positives={"mandelbrot": 626, "phoenix:classic": 7}
                        )["report"]["calibration_reservations"]
    assert cr["min_pos"] == 15 and cr["frac"] == sc.RESERVE_FRAC
    assert cr["cap_frac"] == sc.RESERVE_CAP_FRAC
    assert cr["positives"] == {"mandelbrot": 626, "phoenix:classic": 7}
    assert set(cr["active"]["phoenix:classic"]) == {
        "reserved", "granted", "shortfall", "available", "capped_by_sitting"}
    assert cr["granted_total"] == 0 and cr["shortfall_total"] == 1


def test_the_rule_fires_on_the_LIVE_corpus_for_phoenix_classic_and_not_for_mandelbrot():
    """OFF LIVE COUNTS, not a fixture. The plan arithmetic above passes whatever the corpus
    holds; this is the one that catches the rule being correct and inert — and it is derived,
    so it self-updates the day Matt labels classic past MIN_POS (at which point the assertion
    below flips to the `lapses` test's territory, which is the intended end state)."""
    pos = sc.positives_census()
    assert pos["phoenix:classic"] < sc.min_pos() <= pos["mandelbrot"], pos
    plan = sc.plan_reservations(pos, sc.MAX_ROWS)
    assert plan == {"phoenix:classic": 50}, plan
    # non-vacuity: the census is a real read, not an empty dict that qualifies nothing
    from partitions import ALL_FAMS
    assert set(pos) == set(ALL_FAMS) and sum(pos.values()) > 100


# =========================================================================== #
# serving: the batch id, the registration, and the bar-readability slice
#
# (The v2 "the CLI never serves a sitting" pin is gone on purpose: it was a statement about
# the prompt that built the cutter, and it was superseded the moment a sitting was served.
# What replaces it is a pin on the decisions that outlive that — which batch, registered
# where, and served through what.)
# =========================================================================== #
def test_the_sitting_batch_id_is_the_same_string_in_all_three_places():
    """The id is declared in three modules that cannot import each other cheaply: the cutter
    (which writes the batch), `build_manifest` (which classifies it) and the sheet SPECS
    (which serves it). A typo in any one of them fails SILENTLY in the worst direction —
    `assign_split` falls closed to `unregistered`, which still returns train/biased, so the
    batch builds, looks right, and records that nobody classified it."""
    from tools.v7 import build_manifest as bm
    sys.path.insert(0, str(ROOT / "tools" / "corpus"))
    import build_combined_label_sheet as bcs
    assert sc.SITTING_BATCH in bm.V2_SITTING_BATCHES
    assert bcs.V2_SITTING.sources == (sc.SITTING_BATCH,)


@pytest.mark.parametrize("name", sorted(sc.SITTINGS))
def test_every_sitting_is_served_by_a_sheet_over_exactly_its_own_legs(name):
    """The generalization of the pin above to N legs. The cutter writes the legs and something
    serves them; if the two disagree about the set, the sitting silently serves a subset — the
    rows of the missing leg exist, are registered, are rendered, and are never shown.

    TWO SERVING PATHS, ONE PROPERTY. A blind sitting is served by a `SheetSpec` whose
    `sources` must equal the leg set. A CORRECTION sitting has no sheet — it is served
    straight off its registered batches by the rig (`SittingSpec.serve_url`, see there for
    why the blinding module is the wrong home for it) — so the check is that its URL names
    every leg and nothing else. Both halves are the same assertion: what is served is exactly
    what was cut."""
    sys.path.insert(0, str(ROOT / "tools" / "corpus"))
    import build_combined_label_sheet as bcs
    spec = sc.SITTINGS[name]
    if spec.correction:
        url = spec.serve_url()
        served = url.split("batch=", 1)[1].split("&", 1)[0].split(",")
        assert set(served) == set(spec.batches) and len(served) == len(spec.batches), (
            f"correction sitting {name!r} has legs {spec.batches} but serves {served}")
        assert "order=file" in url, (
            "a correction sheet stamps its own contiguous sheet_order; without order=file the "
            "rig re-derives an order in the browser and there are two owners for it")
        assert "tiers=4" in url, "the label corpus holds 1..4; the rig must not offer a button "\
                                 "the corpus cannot hold"
        assert not any(set(s.sources) == set(spec.batches) for s in bcs.SPECS.values()), (
            f"{name!r} is a correction sitting AND has a blind SheetSpec over the same legs — "
            f"two serving paths for one cut, and the blind one would drop the suggestion")
        return
    sheet = next((s for s in bcs.SPECS.values() if set(s.sources) == set(spec.batches)), None)
    assert sheet is not None, (
        f"sitting {name!r} has legs {spec.batches} and no SheetSpec serves exactly that set; "
        f"sheets: { {s.name: s.sources for s in bcs.SPECS.values()} }")


@pytest.mark.parametrize("batch", sorted({b for s in sc.SITTINGS.values() for b in s.batches}))
def test_every_sitting_leg_is_registered_explicitly_train_side_and_biased(batch):
    from tools.v7 import build_manifest as bm
    split, biased, source = bm.assign_split({"batch": batch, "ft": "mandelbrot"})
    assert source != "unregistered", "the fail-closed default is not a registration"
    assert (split, biased) == ("train", True)
    assert not bm.registration_contradictions([{"batch": batch, "biased": biased}])


def test_every_leg_of_a_sitting_carries_its_OWN_source_name():
    """Two legs registered under one source name would make the sitting's own contrast
    unrecoverable from the corpus: `batches_with_source` is how a later read finds a leg."""
    import batch_registry as br
    for spec in sc.SITTINGS.values():
        srcs = [br.lookup(b, "mandelbrot").source for b in spec.batches]
        assert len(set(srcs)) == len(srcs), f"{spec.name}: legs share a source {srcs}"


def test_draw_refuses_an_unregistered_leg():
    """The fail-closed default is SAFE (train/biased) but it is not a decision, and a sitting
    built under it records that nobody classified it. So `draw` aborts rather than proceeds —
    and it aborts on EVERY leg before writing any of them, so an unregistered second leg
    cannot be discovered after the first one is already on disk."""
    bad = sc.SittingSpec(
        name="unregistered", gen_version="x", seed=1, id_prefix="xx", crop_ss=2,
        legs=(sc.SittingLeg(batch_id=sc.SITTING_BATCH, run_dir="data/discovery/none",
                            selection_role="a", purpose="a"),
              sc.SittingLeg(batch_id="2099-01-01_never_registered",
                            run_dir="data/discovery/none", selection_role="b", purpose="b")))
    with pytest.raises(SystemExit) as e:
        sc.check_registrations(bad)
    assert "NOT registered" in str(e.value)
    assert "2099-01-01_never_registered" in str(e.value)


# =========================================================================== #
# the union queue: one cut over N legs
# =========================================================================== #
def test_the_union_queue_dedups_ACROSS_legs_and_says_how_many():
    """A dive descends from the crawl's admissions, so the two stores CAN record the same
    place. First occurrence wins in LEG ORDER and the count is reported — a union that
    absorbed the collision would serve one place twice under two registrations, which is the
    one thing `stage_verify`'s no-repeat check exists to catch."""
    import build_q4_harvest_batches as bq
    a = _row(cx="1"), _row(cx="2")
    b = _row(cx="2"), _row(cx="3")          # cx=2 is the collision
    seen, rows, dropped = set(), [], {}
    for tag, leg in (("A", a), ("B", b)):
        for r in leg:
            k = bq.queue_identity(r)
            if k in seen:
                dropped[tag] = dropped.get(tag, 0) + 1
                continue
            seen.add(k)
            rows.append(dict(r, _leg=tag))
    assert [r["cx"] for r in rows] == ["1", "2", "3"]
    assert dropped == {"B": 1}, "the SECOND leg loses the collision, and it is counted"


def test_the_union_queue_and_the_single_run_queue_sort_on_the_SAME_key():
    """The sitting's queue and the v1 batch draw must never disagree about what the queue IS,
    so both call `build_q4_harvest_batches.queue_sort_key`. Asserted on the source rather than
    on an example: a second literal here would agree on this fixture and diverge later."""
    import inspect
    import build_q4_harvest_batches as bq
    src = inspect.getsource(sc.load_union_queue)
    assert "sort(key=bq.queue_sort_key)" in src and "bq.queue_identity(r)" in src
    # the sort key's own shape (descending tier, descending score) restated nowhere here
    assert "-int(" not in src and "-float(" not in src, "the sort key must not be restated"
    # ...and the single-run loader APPLIES that same key rather than carrying its own: the
    # count is on the call, not on the token, so the loader may still name the key in the
    # order it reports (`rep["order"]`) without looking like a second copy.
    q4 = inspect.getsource(bq.build_sorted_queue)
    assert q4.count("sort(key=queue_sort_key)") == 1
    assert "-int(" not in q4 and "-float(" not in q4, "the sort key must not be restated"


def test_queue_rank_is_assigned_over_the_UNION_not_per_leg():
    """A per-leg rank would let a small leg's 3rd-best beat a big leg's best inside a cell,
    because the draw reads `queue_rank` as the within-cell order."""
    import inspect
    src = inspect.getsource(sc.load_union_queue)
    body = src[src.index('"""', src.index('"""') + 3):]     # past the docstring
    i_sort, i_rank = body.index("rows.sort("), body.index('r["queue_rank"]')
    assert i_sort < i_rank, "queue_rank must be stamped AFTER the union is sorted"


# =========================================================================== #
# the dive arm: an ORDER argument, checked rather than trusted
# =========================================================================== #
def _jsonl(p, recs):
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return p


def _arm_case(tmp_path, rows, log):
    return sc.recover_dive_arms(_jsonl(tmp_path / "q4_candidates.jsonl", rows),
                                _jsonl(tmp_path / "dive_log.jsonl", log))


def test_the_dive_arm_is_recovered_by_root_id_ORDER_and_partition_checked(tmp_path):
    m, why = _arm_case(
        tmp_path,
        [_row(cx="a", root_id=7, partition="mandelbrot"),
         _row(cx="b", root_id=7, partition="mandelbrot"),
         _row(cx="c", root_id=9, partition="multibrot3")],
        [dict(dive_id="d0", start_group="control", partition="mandelbrot"),
         dict(dive_id="d1", start_group="top", partition="multibrot3")])
    assert m == {7: "dive:control", 9: "dive:top"}
    assert "partition-checked" in why


def test_a_dive_arm_join_that_does_not_BALANCE_yields_no_arm_and_says_why(tmp_path):
    """One more root_id than dives: the order argument does not hold, so the arm goes NULL.
    A wrong arm would invert the very contrast the leg exists to measure, and unlike a null it
    would never be noticed."""
    m, why = _arm_case(
        tmp_path,
        [_row(cx="a", root_id=7, partition="mandelbrot"),
         _row(cx="c", root_id=9, partition="mandelbrot")],
        [dict(dive_id="d0", start_group="top", partition="mandelbrot")])
    assert m == {} and "2 distinct root_id vs 1 dive_log records" in why


def test_a_dive_arm_join_whose_PARTITIONS_disagree_yields_no_arm(tmp_path):
    """The counts can match while the order is still wrong. The partition cross-check is what
    makes this a checked join rather than a coincidence of two equal lengths — and it is what
    caught the arm being recovered from tier-SORTED rows instead of append order."""
    m, why = _arm_case(
        tmp_path,
        [_row(cx="a", root_id=7, partition="mandelbrot"),
         _row(cx="c", root_id=9, partition="multibrot3")],
        [dict(dive_id="d0", start_group="top", partition="multibrot3"),
         dict(dive_id="d1", start_group="control", partition="mandelbrot")])
    assert m == {} and "spans partitions" in why


def test_the_arm_join_reads_the_APPEND_ORDERED_store_not_a_sorted_queue():
    """The order argument is about append order, and `build_sorted_queue` returns the same rows
    tier-sorted. A signature that can be handed the wrong order will be, so this one takes the
    store path and reads it itself."""
    import inspect
    assert "store" in inspect.signature(sc.recover_dive_arms).parameters
    src = inspect.getsource(sc.load_union_queue)
    assert 'recover_dive_arms(run_dir / "q4_candidates.jsonl"' in src


@pytest.mark.parametrize("prov,ok", [
    ({"fit_model": "view_fit_v1.1", "fit_score": 1.0, "composite": 2.0}, True),
    ({"fit_model": "view_fit_v1.1", "fit_score": 0.0, "composite": 0.0}, True),   # 0 is a score
    ({"fit_model": "view_fit_v1.1", "fit_score": 1.0, "composite": None}, False),
    ({"fit_model": "view_fit_v1.1", "fit_score": None, "composite": 2.0}, False),
    ({"fit_model": None, "fit_score": 1.0, "composite": 2.0}, False),
    ({"fit_model": "view_fit_v1.0", "fit_score": 1.0, "composite": 2.0}, False),
    ({}, False),
])
def test_bar_readability_needs_both_scores_and_the_right_model(prov, ok):
    """BOTH, and from the pre-registered model. A row with one of the two cannot contribute to
    a delta-AP between them, and a zero is a score — the `is not None` is what keeps a legit
    0.0 in the slice, which a truthiness test would silently drop."""
    assert sc.is_bar_readable(prov) is ok


def test_the_screen_columns_ride_on_EXISTING_provenance_keys():
    """Nothing is renamed, which is what lets a v2-screened row pool with a supply-crawl or
    label-seeded one — same view frame, same composite_v3, same terms."""
    sys.path.insert(0, str(ROOT / "tools" / "corpus"))
    import corpus_common as cc
    unknown = [k for k in sc.SCREEN_PROV if k not in cc.PROVENANCE_KEYS]
    assert not unknown, f"{unknown} are not modeled provenance keys"
    assert sc.SCREEN_PROV["fit_score"] == "view_fit"
    assert sc.SCREEN_PROV["composite"] == "composite"


def test_render_writes_through_a_partial_and_renames():
    """A kill mid-render must not leave a truncated jpg — which reads as rendered forever, and
    is the one failure an idempotent skip-if-exists resume cannot recover from."""
    import inspect
    src = inspect.getsource(sc._render_one)
    assert ".part.jpg" in src and "os.replace" in src
    assert "if out.exists():" in src and "continue" in src, "resume must skip finished crops"


def test_the_render_partial_still_ends_in_an_extension_THE_ENGINE_CAN_WRITE():
    """The engine picks the image format off the output EXTENSION. `<id>.jpg.tmp` is not slow
    or lossy, it is `The file extension ."tmp" was not recognized as an image format` on every
    single render — a 100% failure rate that reads as a broken renderer, and it cost 50 renders
    to find. Asserted on the built name, not on the source text, so any future partial scheme
    has to satisfy it too."""
    from pathlib import Path as P
    out = P("x/vs0000_deadbeef.jpg")
    tmp = out.with_name(f"{out.stem}.part.jpg")
    assert tmp.suffix == ".jpg", tmp
    assert tmp != out and tmp.parent == out.parent
    # ...and a partial can never be mistaken for the finished crop by an exact-name reader
    assert tmp.name != out.name


def test_the_dedup_embeds_the_FRAME_THE_CROP_RENDERS():
    """The presentation dedup and the crop must be looking at the same picture.

    A row can carry a reframed `outcome_*` viewport beside its own `cx/cy/fw`; the render
    block uses the latter, so the morph embed must too. Measured on the harvest-v2 population:
    70 rows carried `outcome_*` and 49 were a genuinely different frame, so a dedup reading
    `outcome_*` thinned 1.4% of the sitting on a picture nobody would ever be shown. Asserted
    against `_render_block` itself rather than against a literal — that is the module the
    frame has to agree with."""
    sys.path.insert(0, str(ROOT / "tools" / "sourcing"))
    import build_q4_harvest_batches as bq
    r = dict(partition="julia:mandelbrot", cx="0.25", cy="-0.5", fw="0.125",
             outcome_cx="0.9", outcome_cy="0.9", outcome_fw="0.001",
             julia_c_re="0.3", julia_c_im="0.5", _palette="magma")
    led = sc._ledger_row(r)
    rb = bq._render_block(dict(r))
    assert (led["outcome_cx"], led["outcome_cy"], str(led["outcome_fw"])) == \
           (rb["cx"], rb["cy"], str(float(rb["fw"]))), \
        "the embedded frame and the rendered frame diverged"
    assert led["outcome_cx"] == "0.25", "the reframed outcome_* viewport must NOT win"


def test_the_ss_deviation_is_local_and_recorded_not_a_shared_constant_edit():
    """This sitting renders at ss2 where the corpus renders at ss4. Two things must hold, and
    the second is the one that matters later: the deviation is LOCAL (the shared
    `build_minibrot_batch.CROP_SS` is untouched, so a batch that says nothing still gets the
    corpus default), and it is RECORDED in the version-invariant render block, so a crop is
    still a pure function of its own row rather than of what someone chose that day."""
    sys.path.insert(0, str(ROOT / "tools" / "sourcing"))
    import build_minibrot_batch as BMB
    import build_q4_harvest_batches as bq
    assert BMB.CROP_SS == 4 and bq.CROP_SS == 4, "the corpus default must not be edited"
    assert sc.SITTING_CROP_SS != BMB.CROP_SS
    import corpus_common as cc
    assert "ss" in cc.RENDER_KEYS, "the deviation is only safe because ss is version-invariant"


# =========================================================================== #
# the cheap end-to-end. `dry-run` could be bounded and `draw` could not, so the first
# execution of any change to the draw path WAS the 13.9-minute production run — which is how
# a join bug reached it. `--embed-limit` on `draw` is the 15-second version, and the price of
# having it is that a bounded cut must be impossible to mistake for a real one.
# =========================================================================== #
def test_draw_takes_an_embed_limit_and_it_reaches_the_stage():
    """The flag exists on `draw`, not only on `dry-run` — asked of the real CLI, because the
    parser is built inside `main()` and a test that rebuilt it would be testing its own copy."""
    import subprocess
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "atlas" / "sitting_cutter.py"),
                          "draw", "--help"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "--embed-limit" in out.stdout
    assert "INCOMPLETE" in out.stdout, "the help must say what a bounded draw writes"


def test_an_unbounded_cut_stamps_complete_and_a_bounded_one_stamps_INCOMPLETE():
    """The stamp is a pure function of the bound, so it cannot say `false` about a run that
    was in fact bounded. Both directions, because a stamp that is always true is as useless
    as one that is always false."""
    assert sc.completeness_stamp(None) == dict(INCOMPLETE=False, embed_limit=None)
    assert sc.completeness_stamp(0) == dict(INCOMPLETE=False, embed_limit=None)
    assert sc.completeness_stamp(20) == dict(INCOMPLETE=True, embed_limit=20)
    assert sc.completeness_stamp("20") == dict(INCOMPLETE=True, embed_limit=20)


def test_the_stamp_is_DERIVED_at_the_write_site_and_not_restated():
    """`CLAUDE.md`: derive state in code, freeze it in records. A second literal `INCOMPLETE`
    in the batch.json builder is how a metadata file outlives what it records."""
    import inspect
    src = inspect.getsource(sc.stage_draw)
    assert "completeness_stamp(embed_limit)" in src
    assert "INCOMPLETE=" not in src, "the stamp must come from the pure function, not a literal"


def test_the_bounded_draw_uses_the_limit_it_was_given_not_a_hardcoded_None():
    """The bug this whole item is about, in miniature: a flag that is parsed, stored and then
    not passed to the thing it bounds. `draw` built its embedder with a literal `None`."""
    import inspect
    src = inspect.getsource(sc.stage_draw)
    assert "make_embedder(" in src
    call = src[src.index("make_embedder("):]
    assert "embed_limit" in call[:200], "the draw's embedder ignores --embed-limit"


# =========================================================================== #
# the CORRECTION sheet: bucket apportionment + the pre-label stamp (2026-08-07)
# =========================================================================== #
from collections import Counter        # noqa: E402


def _brow(part, tier=2, rank=1, leg="A", dec=None, mix=None, trig=False, eord=None):
    return dict(partition=part, rank_tier=tier, queue_rank=rank, _leg=leg,
                canon_decoded=dec, mix_source=mix, triggered=trig, canon_eord=eord)


def _bucket(name, target, pick):
    return sc.CorrectionBucket(name, target, pick, why="test")


def test_a_row_is_claimed_by_the_FIRST_bucket_that_wants_it():
    """The buckets PARTITION the sheet. A class-4 mandelbrot row satisfies both the mandelbrot
    bucket and the cross-partition class-4 slice; declaration order decides, and the counts
    then add up to the page instead of double-counting it."""
    rows = [_brow("mandelbrot", dec=4, rank=i) for i in range(10)]
    buckets = (_bucket("mandelbrot", 5, lambda r: r["partition"] == "mandelbrot"),
               _bucket("class4", 5, lambda r: r.get("canon_decoded") == 4))
    out, rep = sc.draw_buckets(rows, sc.cell_of, 10, buckets)
    assert rep["buckets"]["mandelbrot"]["available"] == 10
    assert rep["buckets"]["class4"]["available"] == 0
    assert len(out) == 10 and len(set(map(id, out))) == 10   # each row once, not twice


def test_every_bucket_hits_its_target_when_supply_is_ample():
    rows = ([_brow("mandelbrot", rank=i) for i in range(50)]
            + [_brow("phoenix", rank=100 + i) for i in range(50)])
    buckets = (_bucket("m", 6, lambda r: r["partition"] == "mandelbrot"),
               _bucket("p", 4, lambda r: r["partition"] == "phoenix"))
    out, rep = sc.draw_buckets(rows, sc.cell_of, 10, buckets)
    assert [rep["buckets"][k]["granted"] for k in ("m", "p")] == [6, 4]
    assert rep["shortfall_total"] == 0 and rep["redeal"] == []
    assert Counter(r["partition"] for r in out) == {"mandelbrot": 6, "phoenix": 4}


def test_a_shortfall_is_re_dealt_to_the_SCARCEST_bucket_not_the_most_abundant():
    """The whole point of the re-deal rule. `m` is short by 8; `p` has 3 spare rows and `bulk`
    has 200. Handing the slack to `bulk` is what "never pad with julia:multibrot bulk"
    forbids, and scarcest-first is the rule that makes the forbidding structural."""
    rows = ([_brow("mandelbrot", rank=i) for i in range(2)]
            + [_brow("phoenix", rank=10 + i) for i in range(7)]
            + [_brow("julia:multibrot4", rank=100 + i) for i in range(200)])
    buckets = (_bucket("m", 10, lambda r: r["partition"] == "mandelbrot"),
               _bucket("p", 4, lambda r: r["partition"] == "phoenix"),
               _bucket("bulk", 6, lambda r: r["partition"] == "julia:multibrot4"))
    out, rep = sc.draw_buckets(rows, sc.cell_of, 20, buckets)
    assert rep["buckets"]["m"]["granted"] == 2 and rep["buckets"]["m"]["shortfall"] == 8
    # p was scarcer than bulk, so p absorbs first and drains; only then does bulk take the rest
    assert rep["buckets"]["p"]["redealt_in"] == 3 and rep["buckets"]["p"]["granted"] == 7
    assert [d["bucket"] for d in rep["redeal"]][0] == "p"
    assert len(out) == 20


def test_a_fully_drained_set_returns_a_SHORT_sheet_rather_than_padding_it():
    """Nothing is drawn from outside the buckets. The alternative — a general balanced fill —
    is exactly how the page fills up with the material the sheet is not short of."""
    rows = [_brow("mandelbrot", rank=i) for i in range(3)]
    buckets = (_bucket("m", 10, lambda r: r["partition"] == "mandelbrot"),)
    out, rep = sc.draw_buckets(rows, sc.cell_of, 10, buckets)
    assert len(out) == 3 and rep["shortfall_total"] == 7 and rep["drawn"] == 3


def test_a_later_leg_is_reached_ONLY_after_the_earlier_one_is_exhausted():
    """Backfilled only where a bucket falls short, as a property. Leg B outnumbers leg A
    25:1 and is still untouched while any leg-A row in the cell remains."""
    rows = ([_brow("phoenix", rank=i, leg="B") for i in range(100)]
            + [_brow("phoenix", rank=200 + i, leg="A") for i in range(4)])
    buckets = (_bucket("p", 4, lambda r: r["partition"] == "phoenix"),)
    out, rep = sc.draw_buckets(rows, sc.cell_of, 4, buckets, leg_rank={"A": 0, "B": 1})
    assert rep["buckets"]["p"]["by_leg"] == {"A": 4}
    buckets6 = (_bucket("p", 6, lambda r: r["partition"] == "phoenix"),)
    out2, rep2 = sc.draw_buckets(rows, sc.cell_of, 6, buckets6, leg_rank={"A": 0, "B": 1})
    assert rep2["buckets"]["p"]["by_leg"]["A"] == 4          # A first, then B backfills
    assert rep2["buckets"]["p"]["by_leg"]["B"] == 2


def test_bucket_targets_that_do_not_sum_to_the_cap_are_REFUSED_at_construction():
    with pytest.raises(ValueError, match="sum to"):
        sc.SittingSpec(name="bad", gen_version="x", seed=1, id_prefix="x", crop_ss=2,
                       legs=(sc.SittingLeg("b", "d", "r", "p"),), max_rows=10,
                       buckets=(_bucket("m", 3, lambda r: True),))


def test_the_label_run_buckets_sum_to_the_cap_and_are_the_prompts_numbers():
    spec = sc.LABEL_RUN_SITTING
    assert spec.max_rows == 500 and spec.correction is True
    assert {b.name: b.target for b in spec.buckets} == {
        "mandelbrot": 150, "julia:mandelbrot": 125, "phoenix": 100,
        "native_multibrot_maneuver": 75, "machine_class4_top": 50}
    assert sum(b.target for b in spec.buckets) == spec.max_rows
    # the class-4 slice is declared LAST, so it only sees what the partition buckets left
    assert spec.buckets[-1].name == "machine_class4_top"


def test_julia_mandelbrots_machine_1s_reach_the_sheet():
    """The prompt's explicit requirement, and it is a property of the imported discard table
    rather than of this sitting: P(Matt=1 | decoded 1) is 30.9% for julia:mandelbrot, so the
    table keeps them. A sitting that had to special-case this would be a second owner."""
    assert srt.MACHINE_1_DISCARD.get("julia:mandelbrot") is False
    rows = [_brow("julia:mandelbrot", dec=1), _brow("multibrot4", dec=1)]
    kept, removed, _rep = sc.stage_machine_1(rows, {})
    assert [r["partition"] for r in kept] == ["julia:mandelbrot"]
    assert [r["partition"] for r in removed] == ["multibrot4"]


def test_the_native_multibrot_bucket_takes_maneuver_sourced_rows_ONLY():
    pick = next(b for b in sc.LABEL_RUN_SITTING.buckets
                if b.name == "native_multibrot_maneuver").pick
    assert pick(_brow("multibrot4", mix="maneuver:neighborhood_expand:k=8.0"))
    assert pick(_brow("multibrot3", trig=True))
    assert not pick(_brow("multibrot4", mix="native"))
    assert not pick(_brow("mandelbrot", trig=True))          # native MULTIbrot only


def test_the_correction_stamp_is_good_to_bad_contiguous_and_never_a_label():
    rows = [dict(image_id=f"lr{i:02d}", label=dict(score=None), _pred=p, _sugg=s)
            for i, (p, s) in enumerate([(2.0, 2), (4.1, 4), (None, None), (3.5, 3)])]
    rep = sc.stamp_correction(rows)
    assert [r["image_id"] for r in rows] == ["lr01", "lr03", "lr00", "lr02"]  # 4.1,3.5,2.0,None
    assert [r["sheet_order"] for r in rows] == [0, 1, 2, 3]
    assert rows[-1]["pred"] is None and rows[-1]["suggested_tier"] is None
    assert all(r["label"]["score"] is None for r in rows), "a suggestion is NOT a label"
    assert rep["n_with_pred"] == 3 and rep["n_without_pred"] == 1
    assert "_pred" not in rows[0] and "_sugg" not in rows[0]


def test_head_pred_is_on_the_tier_scale_and_absent_without_a_canonical_decode():
    assert sc.head_pred(dict(canon_eord=2.4)) == pytest.approx(3.4)
    assert sc.head_pred(dict(canon_eord=None)) is None
    assert sc.suggested_tier(dict(canon_decoded=3, reframe_decoded=4)) == 4   # reframe wins
    assert sc.suggested_tier(dict(canon_decoded=3)) == 3
    assert sc.suggested_tier(dict()) is None


def test_a_bucketed_cut_does_not_ALSO_reserve():
    """Five targets already sum to the cap; a sixth claim would silently displace one."""
    rows = [_brow("phoenix", rank=i) for i in range(20)]
    res = sc.cut_sitting(rows, max_rows=5, embed=lambda r: _unit(r["queue_rank"], 64),
                         buckets=(_bucket("p", 5, lambda r: r["partition"] == "phoenix"),))
    assert res["report"]["calibration_reservations"]["active"] == {}
    assert res["report"]["buckets"]["drawn"] == 5


def test_the_reservation_path_is_untouched_when_no_buckets_are_passed():
    rows = [_brow("phoenix", rank=i) for i in range(20)]
    res = sc.cut_sitting(rows, max_rows=5,
                         embed=lambda r: _unit(r["queue_rank"], 64),
                         reservations={})
    assert res["report"]["buckets"] is None


def test_a_no_pad_row_fills_its_OWN_target_but_never_anothers_shortfall():
    """THE DEFECT THE REHEARSAL FOUND, as a regression.

    `bulk` is abundant and `slice` is cross-partition, so "scarcest bucket with supply" walked
    straight into the bulk material: the first bounded end-to-end drew 294 rows into a 50-row
    class-4 slice and padded the page with 244 julia:multibrot rows. Both halves are asserted,
    because a `no_pad` that also blocked the target would silently shrink the slice."""
    rows = ([_brow("mandelbrot", rank=i) for i in range(2)]
            + [_brow("julia:multibrot4", rank=100 + i, dec=4) for i in range(300)]
            + [_brow("multibrot3", rank=500 + i, trig=True) for i in range(600)])
    buckets = (_bucket("m", 10, lambda r: r["partition"] == "mandelbrot"),
               _bucket("native", 5, lambda r: r["partition"] == "multibrot3"),
               _bucket("slice", 5, lambda r: r.get("canon_decoded") == 4))
    no_pad = lambda r: str(r.get("partition") or "").startswith("julia:multibrot")
    out, rep = sc.draw_buckets(rows, sc.cell_of, 20, buckets, no_pad=no_pad)
    assert len(out) == 20
    # the slice still gets its OWN five, from the very material the pad rule blocks
    assert rep["buckets"]["slice"]["granted"] == 5
    assert rep["buckets"]["slice"]["redealt_in"] == 0
    # ...and mandelbrot's 8-row shortfall was paid in native rows, not julia:multibrot bulk
    assert rep["buckets"]["native"]["redealt_in"] == 8
    assert sum(1 for r in out if r["partition"].startswith("julia:multibrot")) == 5
    assert rep["pad_blocked_rows_left"] == 295


def test_without_the_pad_rule_the_bulk_bucket_DOES_swallow_the_shortfall():
    """The negative control for the test above: same population, no `no_pad`, and the slice
    runs away with the page. Without this the guard could pass because the draw is short."""
    rows = ([_brow("mandelbrot", rank=i) for i in range(2)]
            + [_brow("julia:multibrot4", rank=100 + i, dec=4) for i in range(300)]
            + [_brow("multibrot3", rank=500 + i, trig=True) for i in range(600)])
    buckets = (_bucket("m", 10, lambda r: r["partition"] == "mandelbrot"),
               _bucket("native", 5, lambda r: r["partition"] == "multibrot3"),
               _bucket("slice", 5, lambda r: r.get("canon_decoded") == 4))
    out, rep = sc.draw_buckets(rows, sc.cell_of, 20, buckets)
    assert rep["buckets"]["slice"]["redealt_in"] == 8      # the bulk absorbs it
    assert sum(1 for r in out if r["partition"].startswith("julia:multibrot")) == 13


def test_the_label_run_pad_rule_names_julia_multibrot_and_carries_its_reason():
    spec = sc.LABEL_RUN_SITTING
    assert spec.no_pad is not None and spec.no_pad_rule
    for p in ("julia:multibrot3", "julia:multibrot4", "julia:multibrot5"):
        assert spec.no_pad(dict(partition=p)), p
    for p in ("mandelbrot", "julia:mandelbrot", "phoenix", "multibrot4"):
        assert not spec.no_pad(dict(partition=p)), p


def test_a_pad_rule_without_a_stated_reason_is_REFUSED():
    with pytest.raises(ValueError, match="no_pad_rule"):
        sc.SittingSpec(name="bad", gen_version="x", seed=1, id_prefix="x", crop_ss=2,
                       legs=(sc.SittingLeg("b", "d", "r", "p"),),
                       no_pad=lambda r: True)


def test_the_dry_run_cuts_the_SAME_WAY_the_draw_does():
    """A rehearsal that took the balanced path while `draw` took the bucketed one would report
    an apportionment nobody is about to build. Source-scanned because the two call sites are
    what must agree, and there is no cheap way to run both real cuts in a unit test."""
    src = (ROOT / "tools" / "atlas" / "sitting_cutter.py").read_text(encoding="utf-8")
    calls = [src[m:src.index(")", src.index("no_pad", m))]
             for m in [i for i in range(len(src)) if src.startswith("res = cut_sitting(", i)]]
    assert len(calls) == 2, "expected exactly two cut_sitting call sites (draw + dry-run)"
    for c in calls:
        for kw in ("buckets=", "leg_rank=", "no_pad="):
            assert kw in c, f"a cut_sitting call site is missing {kw}: {c[:160]}"
