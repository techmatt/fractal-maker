#!/usr/bin/env python
"""`tools/emission/attempt_budget.py` — colorize volume budgeted from release need (2026-08-09).

The failure these pin is the selrestruct_1 smoke: **3 smooth rows out of 30 attempts against 6
smooth slots**, because colorize volume fell out of a deficit model spreading over a style axis
with one smooth style and N strange ones. Every test below is that failure stated as a
property.

  1. a head's attempts are `ATTEMPT_MULTIPLIER × its own release slots` — never a share of the
     style axis, and never a function of the other head's need;
  2. an over-subscribed budget scales BOTH heads proportionally (`deal_round_robin` would
     level them, which is the same starvation with the sign flipped);
  3. within a head the split is release_mix over the partitions with floor-passing SUPPLY, and
     the fill is rank order;
  4. every PREFIX of the plan is near-proportional, because a wall-clock backstop truncates it;
  5. planned and realized are separate readings — the plan never reports its own execution.

  uv run pytest tools/emission/test_attempt_budget.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import release_mix as RM                             # noqa: E402
from apportion import prefix_deviation                # noqa: E402
from tools.emission import attempt_budget as AB      # noqa: E402
from tools.emission import floors as F               # noqa: E402


def _supply(**per_partition) -> dict:
    """`{partition: [id, ...]}` best-first, the shape `ranked_intake` hands over."""
    return {p: [f"{p}_{k:03d}" for k in range(n)] for p, n in per_partition.items()}


# --------------------------------------------------------------------------- #
# 1. the head split is sized against release slots.
# --------------------------------------------------------------------------- #
def test_each_head_gets_the_multiplier_times_its_own_release_slots():
    """THE rule. N=12 at strange_frac 0.5 is 6 slots each, so each head asks for 4×6 = 24 —
    and gets it whole when the budget covers the pair."""
    att, diag = AB.head_attempts(AB.head_slots(12, 0.5), total_budget=240)
    assert att == {"wallpaper": 24, "mining": 24}
    assert diag["head_slots"] == {"wallpaper": 6, "mining": 6}
    assert diag["scaled_to_budget"] is False


def test_a_head_with_more_slots_gets_more_attempts_not_an_equal_share():
    """The head split follows the RELEASE, not the style axis and not fairness between heads.
    strange_frac 0.25 of 12 is 3 strange / 9 smooth slots → 36 smooth / 12 strange attempts."""
    att, _ = AB.head_attempts(AB.head_slots(12, 0.25), total_budget=240)
    assert att == {"wallpaper": 36, "mining": 12}


def test_a_head_with_no_slots_gets_no_attempts():
    """strange_frac 0 means the release has no strange slots, so no strange colorize is paid
    for. The budget is release need; a head with no need has no budget."""
    att, _ = AB.head_attempts(AB.head_slots(8, 0.0), total_budget=240)
    assert att == {"wallpaper": 32, "mining": 0}


def test_the_multiplier_is_the_owners_constant_not_a_literal_here():
    att, diag = AB.head_attempts({"wallpaper": 5, "mining": 0}, total_budget=1000)
    assert diag["attempt_multiplier"] == F.ATTEMPT_MULTIPLIER
    assert att["wallpaper"] == 5 * F.ATTEMPT_MULTIPLIER


# --------------------------------------------------------------------------- #
# 2. an over-subscribed budget scales BOTH heads, proportionally.
# --------------------------------------------------------------------------- #
def test_an_over_subscribed_budget_scales_both_heads_and_starves_neither():
    """THE regression, at the smoke's own numbers: 6+6 slots want 24+24 = 48 attempts under a
    30-attempt budget. Both come down to 15; smooth does NOT come down to 3."""
    att, diag = AB.head_attempts(AB.head_slots(12, 0.5), total_budget=30)
    assert att == {"wallpaper": 15, "mining": 15}
    assert diag["scaled_to_budget"] is True
    assert sum(att.values()) == 30


def test_the_scale_down_is_proportional_not_levelling():
    """The distinction the module docstring is about, on a shape where the two rules differ:
    want 20/40 under a budget of 24 is 8/16 proportionally and 12/12 by round-robin. Levelling
    the larger need down to the smaller is the same starvation the head split fixes, pointed
    the other way — and it is invisible at the default strange_frac 0.5, where both heads ask
    for the same number and the two rules agree."""
    got = AB.scale_to_budget({"wallpaper": 20, "mining": 40}, 24)
    assert got == {"wallpaper": 8, "mining": 16}
    from apportion import deal_round_robin
    assert deal_round_robin({"wallpaper": 20, "mining": 40}, 24) == {"wallpaper": 12, "mining": 12}


def test_a_budget_that_covers_the_want_leaves_it_alone():
    want = {"wallpaper": 24, "mining": 24}
    assert AB.scale_to_budget(want, 48) == want
    assert AB.scale_to_budget(want, 4800) == want


def test_a_zero_budget_plans_nothing_rather_than_raising():
    plan, budget = AB.plan(12, 0.5, 0, _supply(mandelbrot=50))
    assert plan == [] and budget["planned_total"] == 0


# --------------------------------------------------------------------------- #
# 3. within a head: release_mix over the partitions with supply, filled by rank.
# --------------------------------------------------------------------------- #
def test_the_partition_split_is_the_release_mix_over_the_partitions_with_supply():
    """A partition with no floor-passing supply is not in the solve at all — it must not hold
    attempts hostage, exactly as it does not hold a release slot hostage. The mix reaches the
    attempts through the SEATED SLOTS now (`seated_slots` -> `partition_attempts`), so the
    property is asserted end to end rather than on the old bare-mix call."""
    parts = ["mandelbrot", "multibrot3"]
    seated = AB.seated_slots(parts, 5)
    assert sum(seated.values()) == 5 and set(seated) == set(parts)
    got = AB.partition_attempts(seated, 20)
    assert sum(got.values()) == 20 and set(got) == set(parts)
    sh = RM.shares(parts)
    # the richer share gets the larger budget; the exact split is the apportionment owner's.
    assert (got["mandelbrot"] > got["multibrot3"]) == (sh["mandelbrot"] > sh["multibrot3"])


def test_every_seated_partition_is_budgeted_the_full_multiple_of_its_slots():
    """THE RULE, stated on the function that holds it: attempts are `multiplier ×` seats, not
    the mix re-solved at `multiplier × N`. The second is what shipped until 2026-08-11 and it
    is off by a whole attempt on a 2-seat partition even with no guarantee in play."""
    seated = {"mandelbrot": 2, "multibrot3": 1}
    got = AB.partition_attempts(seated, 12, multiplier=4)
    assert got == {"mandelbrot": 8, "multibrot3": 4}
    for p, k in seated.items():
        assert got[p] >= 4 * k


# --------------------------------------------------------------------------- #
# 3b. the GUARANTEE (2026-08-11) — a partition the release is obliged to seat must be
# budgeted enough colorize to fill that seat.
# --------------------------------------------------------------------------- #
_SMALL = dict(mandelbrot=50, multibrot3=50)
_SMALL["phoenix:classic"] = 50


def _old_split(parts, n):
    """The SUPERSEDED rule — `release_mix` apportioned straight to the head's attempt total,
    with no knowledge of the guarantee. Kept executable rather than described for the reason
    `test_pop_quota._v1_price` is: "old was wrong" is unassertable once the old code is gone
    (`verification_practice.md` §3)."""
    from tools.emission import ranked_intake as RI       # noqa: PLC0415
    return RI.partition_slots(RM.shares(sorted(parts)), n) if n else {p: 0 for p in parts}


def test_the_old_split_starved_a_guaranteed_partition_at_small_release_n():
    """THE DEFECT, asserted. `phoenix:classic` carries the lowest ratio in `release_mix` (0.2),
    so the bare mix hands it almost nothing — while the slot guarantee obliges the release to
    seat it. At N=6 the old rule budgeted it **1 attempt of 12** for a slot it was certain to
    be asked to fill; the guaranteed seat needs `ATTEMPT_MULTIPLIER × 1 = 4`.

    It is not only a small-N effect, which is the correction to the premise: at N=12 — twice
    the 'below ≈7' band — the old rule still budgets it 1 attempt of 24 against the same
    guaranteed seat, because the shortfall is set by the RATIO, not by N."""
    parts = sorted(_SMALL)
    for n, att in ((6, 12), (12, 24)):
        old = _old_split(parts, att)
        assert old["phoenix:classic"] <= 1, (n, old)
        seated = AB.seated_slots(parts, AB.head_slots(n, 0.5)["wallpaper"],
                                 {"phoenix:classic"})
        assert seated["phoenix:classic"] == 1, "the guarantee seats it"
        assert old["phoenix:classic"] < F.ATTEMPT_MULTIPLIER * seated["phoenix:classic"]


def test_every_seated_partition_gets_the_full_multiple_at_small_release_n():
    """THE FIX. Same populations, through `plan` with the guarantee declared: whichever head
    is made to owe a partition its slot budgets `ATTEMPT_MULTIPLIER ×` that slot, at every N
    the old rule starved. Asserted on the SEATED slots the plan recorded, so the property is
    checked against what the run will actually be asked to fill."""
    for n in (2, 4, 6, 8, 12):
        _plan, budget = AB.plan(n, 0.5, 10_000, _supply(**_SMALL), guaranteed=sorted(_SMALL))
        for h in AB.HEADS:
            seated = budget["seated_slots"][h]
            planned = budget["planned_by_partition"][h]
            for p, k in seated.items():
                assert planned.get(p, 0) >= F.ATTEMPT_MULTIPLIER * k, (n, h, p, budget)


def test_the_guarantee_is_split_across_heads_rather_than_billed_to_one():
    """Both heads' mixes are eroded evenly — `_guarantee_head`'s second key, which is the one
    key that survives into a pre-colorize decision. Three guaranteed partitions at N=4 (2 slots
    a head) cannot all be seated by one head, and a head is never asked to owe more guarantees
    than it has slots."""
    _plan, budget = AB.plan(4, 0.5, 10_000, _supply(**_SMALL), guaranteed=sorted(_SMALL))
    owed = budget["guarantee_head"]
    assert set(owed) == set(_SMALL) and not budget["guarantee_unplaced"]
    per_head = {h: sum(1 for v in owed.values() if v == h) for h in AB.HEADS}
    assert per_head == {"wallpaper": 1, "mining": 2} or per_head == {"wallpaper": 2, "mining": 1}
    for h in AB.HEADS:
        assert per_head[h] <= budget["head_slots"][h]


def test_a_guarantee_no_head_has_room_for_is_recorded_not_raised():
    """N=2 is two slots against three guaranteed partitions. The budget is not the place that
    raises — the driver's fixed point knows the candidate counts and can say something true
    about them — but the shortfall must be attributable, so it is in the record."""
    _plan, budget = AB.plan(2, 0.5, 10_000, _supply(**_SMALL), guaranteed=sorted(_SMALL))
    assert budget["guarantee_unplaced"], "an unplaceable guarantee must be visible"
    assert len(budget["guarantee_head"]) + len(budget["guarantee_unplaced"]) == 3


def test_declaring_no_guarantee_still_budgets_the_full_multiple_of_what_is_seated():
    """The default path. With `guaranteed` unset the seats are the plain mix — and attempts are
    still `multiplier ×` them, which is the half of the fix that has nothing to do with the
    guarantee (`4N` apportioned directly ≠ `N` apportioned then multiplied)."""
    _plan, budget = AB.plan(12, 0.5, 10_000, _supply(**_SMALL))
    assert not budget["guaranteed_partitions"] and not budget["guarantee_head"]
    for h in AB.HEADS:
        for p, k in budget["seated_slots"][h].items():
            assert budget["planned_by_partition"][h][p] == F.ATTEMPT_MULTIPLIER * k


def test_a_budget_capped_run_truncates_the_guarantee_proportionally_not_onto_one_partition():
    """`--max-attempts` below the pair's want still scales BOTH heads (the standing rule), and
    inside a head the shortfall is spread by the same `sequence_by_deficit` truncation rather
    than falling entirely on the smallest seat."""
    _plan, budget = AB.plan(12, 0.5, 20, _supply(**_SMALL), guaranteed=sorted(_SMALL))
    assert budget["scaled_to_budget"] is True
    assert sum(sum(v.values()) for v in budget["planned_by_partition"].values()) <= 20
    for h in AB.HEADS:
        planned = budget["planned_by_partition"][h]
        assert sum(1 for p, k in budget["seated_slots"][h].items() if planned.get(p, 0) == 0) \
            <= 1, "a cap must not zero every small seat while a large one keeps its multiple"


def test_a_partition_absent_from_supply_gets_no_attempts():
    plan, budget = AB.plan(12, 0.5, 240, _supply(mandelbrot=200, phoenix=200))
    for h in AB.HEADS:
        assert set(budget["planned_by_partition"][h]) == {"mandelbrot", "phoenix"}
    assert {a.partition for a in plan} == {"mandelbrot", "phoenix"}


def test_each_partitions_attempts_are_filled_in_rank_order():
    """Best-first, and a location is never planned twice for one head. `_supply` ids are
    already in rank order, so 'rank order' is 'the prefix of the list'."""
    plan, budget = AB.plan(12, 0.5, 240, _supply(mandelbrot=200, phoenix=200))
    for h in AB.HEADS:
        for p, k in budget["planned_by_partition"][h].items():
            got = [a.location_id for a in plan if a.head == h and a.partition == p]
            assert got == sorted(got), "not rank order"
            assert got == [f"{p}_{i:03d}" for i in range(k)]


def test_both_heads_may_plan_the_same_top_ranked_location():
    """The two heads render DIFFERENT styles, so the best location in a partition is the right
    answer for both budgets. What must not happen is one head planning it twice."""
    plan, _ = AB.plan(4, 0.5, 240, _supply(mandelbrot=50))
    top = [a.head for a in plan if a.location_id == "mandelbrot_000"]
    assert sorted(top) == ["mining", "wallpaper"]
    for h in AB.HEADS:
        ids = [a.location_id for a in plan if a.head == h]
        assert len(ids) == len(set(ids))


def test_a_partition_thinner_than_its_attempts_short_fills_and_says_so():
    """Supply, not budget — and the record separates them, which is the whole point of keeping
    planned beside realized."""
    plan, budget = AB.plan(12, 0.5, 240, _supply(mandelbrot=3, phoenix=400))
    assert budget["supply_short_total"] > 0
    short = budget["supply_short_by_partition"]["wallpaper"]
    assert short.get("mandelbrot", 0) == budget["planned_by_partition"]["wallpaper"]["mandelbrot"] - 3
    assert sum(1 for a in plan if a.head == "wallpaper" and a.partition == "mandelbrot") == 3
    assert budget["scheduled_total"] == len(plan) < budget["planned_total"]


# --------------------------------------------------------------------------- #
# 4. every prefix is near-proportional (the loop is a truncating consumer).
# --------------------------------------------------------------------------- #
def test_every_prefix_of_the_plan_is_near_proportional_across_heads_and_partitions():
    """The time backstop can cut the loop anywhere, so the ORDER carries the mix. The bound is
    asserted on the order BUILT (apportion.py: ±1 is a check, never a theorem)."""
    plan, budget = AB.plan(12, 0.5, 240,
                           _supply(mandelbrot=300, phoenix=180, multibrot3=170, multibrot5=210))
    sizes: dict = {}
    for a in plan:
        sizes[(a.head, a.partition)] = sizes.get((a.head, a.partition), 0) + 1
    dev = prefix_deviation([(a.head, a.partition) for a in plan], sizes)
    assert dev <= 1.0, f"prefix deviation {dev:.3f} on the built order"


def test_a_truncated_prefix_still_reaches_both_heads_early():
    """The concrete half: the first 8 of a 48-attempt plan are not one head's."""
    plan, _ = AB.plan(12, 0.5, 240, _supply(mandelbrot=300, phoenix=180))
    heads = {a.head for a in plan[:8]}
    assert heads == {"wallpaper", "mining"}


def test_the_plan_is_a_pure_function_of_its_inputs():
    """Batch reproducibility: no seed, no clock, no dict-order dependence."""
    sup = _supply(mandelbrot=300, phoenix=180, multibrot3=170)
    a, _ = AB.plan(12, 0.5, 240, sup)
    b, _ = AB.plan(12, 0.5, 240, dict(reversed(list(sup.items()))))
    assert a == b


# --------------------------------------------------------------------------- #
# 5. realized is DERIVED, never the plan reporting on itself.
# --------------------------------------------------------------------------- #
def _head_of(style):
    return "wallpaper" if style == "smooth" else "mining"


def test_realized_fills_are_read_off_the_pool_not_the_plan():
    rows = [{"render_style": "smooth", "type": "mandelbrot"},
            {"render_style": "smooth", "type": "phoenix"},
            {"render_style": "tia", "type": "mandelbrot"},
            {"render_style": "stripe", "type": "mandelbrot"}]
    got = AB.realized_fills(rows, _head_of)
    assert got == {"wallpaper": {"mandelbrot": 1, "phoenix": 1},
                   "mining": {"mandelbrot": 2}}


def test_realized_can_fall_short_of_planned_without_the_plan_noticing():
    """A render error, a capped cell and a resume all diverge from the plan. `plan` holds no
    realized counts at all, so there is nothing that can go stale."""
    _plan, budget = AB.plan(12, 0.5, 30, _supply(mandelbrot=300))
    assert "realized_by_partition" not in budget
    lines = AB.fill_lines(budget, AB.realized_fills([], _head_of))
    assert any("0 realized" in l for l in lines)


def test_fill_lines_carry_want_budgeted_scheduled_and_realized():
    """The four numbers a short-fill is attributed from. A line missing one of them sends the
    reader back to the pool, which is what this record exists to avoid."""
    plan, budget = AB.plan(12, 0.5, 30, _supply(mandelbrot=300))
    realized = AB.realized_fills(
        [{"render_style": "smooth", "type": "mandelbrot"}], _head_of)
    line = [l for l in AB.fill_lines(budget, realized) if l.startswith("wallpaper")][0]
    assert "24 wanted" in line and "15 budgeted" in line and "15 scheduled" in line
    assert "1 realized" in line
    assert len([a for a in plan if a.head == "wallpaper"]) == 15
