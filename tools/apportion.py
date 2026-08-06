#!/usr/bin/env python
"""apportion.py — the two apportionment rules, once, with the failure mode of each.

Every builder in this tree that spreads a draw over cells reaches for one of two rules, and
until 2026-08-05 there were **seven** copies of them across seven modules — five of the
round-robin, two of the deficit rule, counted by the source scan in `test_apportion.py` rather
than by reading (the sitting notes that prompted this said eight, having counted
`interleave_dive_arms` in both dialects). They are NOT interchangeable — they answer different
questions and only one of them holds the ±1 PREFIX bound — so seven copies is seven chances to
pick the dialect that does not hold the property the caller is relying on. Both are here,
dependency-free (stdlib only, no numpy, no project import), so the layering excuse for an
eighth copy does not exist: importing this costs nothing and pulls nothing.

  `deal_round_robin(sizes, n)`      — DRAW A SUBSET of size n. Floor-then-remainder.
  `sequence_by_deficit(sizes)`      — ORDER A WHOLE SET. Webster/Sainte-Laguë sequencing.

WHICH ONE. Ask what the caller needs to be true:

  * "no cell owns the page" while taking the top `n` of a queue -> `deal_round_robin`. Its
    guarantee is about the FINAL counts: every non-exhausted cell is within 1 of every other
    non-exhausted cell. It says nothing about any prefix, and its emission order is the
    caller's business (the callers here interleave round by round, or concatenate cell
    blocks — both are legal and both are the same take).
  * "the run can be cut short at any point and still be readable" -> `sequence_by_deficit`.
    Its subject is EVERY PREFIX: at every length L each cell is held near its proportional
    share L*n_c/N. That is the only property that survives a truncating budget, and a
    truncating budget is the normal case (the 2026-08-05 dive stopped at 7 of 28).

THE MEASURED REASON THEY ARE NOT INTERCHANGEABLE. The cheap way to order a whole set — lay
each cell's rows at (i+0.5)/n_c and sort by that key, which is the round-robin dialect
generalized to a sequence — does NOT hold ±1: on the label sheet's own cell shape
`{(a,x):43, (a,y):36, (b,x):290, (c,y):48, (c,z):98}` it reaches a prefix deviation of
**1.068** where the deficit rule reaches 0.738, and the accumulation gets worse with cell
count (1.77 vs 1.50 on the 13-cell skew below). Measured 2026-08-05 by
`test_apportion.py::test_the_flat_spread_dialect_MISSES_the_bound_on_the_same_population`,
which keeps it as a live control so the claim is proved rather than remembered.

±1 IS NOT A THEOREM ABOUT THE DEFICIT RULE — IT IS A CHECK EACH CALLER RUNS. Greedy
largest-deficit is provably tight for TWO cells (worst deviation 0.5, exhaustive over all
sizes to 60x60 — so the dive's top/control interleave is safe by construction), and it holds
comfortably on the sheet's real population (0.791 on the built sitting, 0.738 on the shape
above). It is **not** universal: with many cells and two-orders-of-magnitude supply skew it
exceeds 1 — a frozen 13-cell counterexample in the tests reaches **1.495**, and 42% of
randomly-drawn skewed 15-cell populations exceed 1. Both callers therefore assert the bound on
the ORDER THEY BUILT (`build_combined_label_sheet.stage_verify`, `test_steered_frontier`) and
neither trusts the rule. Do not "simplify" those checks away, and do not restate the bound as
a guarantee of this module.

ORDER IS THE CALLER'S. Both functions iterate `sizes` in ITS OWN ORDER and break ties on it,
so a caller that hands over an arbitrarily-ordered dict gets an arbitrary (but deterministic
for that dict) answer. Every caller here passes a sorted mapping; that is the convention, not
a suggestion — pass `dict(sorted(...))` or a mapping you built in a sorted walk.
"""
from __future__ import annotations

from typing import Callable, Hashable, Mapping, Sequence

__all__ = ["deal_round_robin", "cells_balanced", "sequence_by_deficit", "prefix_deviation"]


def deal_round_robin(sizes: Mapping[Hashable, int], n: int, *,
                     preseed: Mapping[Hashable, int] | None = None) -> dict:
    """`{cell: take}` — floor-then-remainder over the cells, capped by supply.

    Every non-empty cell gives up its first row before any cell gives up its second; ties go
    to the cell with the most supply, so a round-robin that cannot fill every cell spends the
    remainder where there is something to spend it on. Stops early when every cell is drained
    (`sum(take) < n` is then the honest answer, and the caller reports the shortfall).

    "BALANCED TO ±1" IS AN INVARIANT ABOUT NON-EXHAUSTED CELLS, and stating it as a flat spread
    over all cells is wrong — measured on the q4 harvest queue, where the flat spread is 78
    while the draw is behaving perfectly. Real cells differ in SUPPLY by two orders of
    magnitude, and a cell that gave up everything it had cannot be faulted for giving up less
    than a cell that did not. So the acceptance predicate is: every cell is within 1 of the
    maximum take, OR it was drained. A flat-spread assertion goes red on a correct draw, which
    is the failure mode `verification_practice.md` §4 calls getting trained out.

    `preseed` credits a cell with rows it was ALREADY given (a reservation), so the round-robin
    continues from that count instead of restarting at zero. That is what makes a reservation a
    FLOOR rather than a bonus: without it, a cell the balanced draw would have served
    generously anyway ends up with its natural share PLUS the reservation. `max(natural,
    reserved)` is the intent; `natural + reserved` is what a naive two-pass draw does, and the
    two differ by exactly the reservation on every cell that did not need one. Preseed counts
    are NOT part of the return — the caller already holds those rows.

    A cell present only in `preseed` (reserved, no remaining supply) must be passed in `sizes`
    with 0 so it participates in the tie-breaks; `sizes` is the authority on which cells exist.
    """
    keys = list(sizes)
    seed = {k: 0 for k in keys}
    for k, v in (preseed or {}).items():
        if k not in seed:
            raise KeyError(f"preseed cell {k!r} is absent from `sizes` — pass it with size 0 "
                           f"if it is drained, so it is visible to the tie-breaks")
        seed[k] = int(v)
    take = {k: 0 for k in keys}
    n = int(n)
    while sum(take.values()) < n:
        cand = [k for k in keys if take[k] < sizes[k]]
        if not cand:
            break
        take[min(cand, key=lambda k: (take[k] + seed[k], -sizes[k]))] += 1
    return take


def cells_balanced(taken: Mapping[Hashable, int],
                   sizes: Mapping[Hashable, int]) -> tuple[bool, str]:
    """The acceptance predicate for a `deal_round_robin` result, as a pure function so a
    `verify` stage and a test share one copy of it.

    Balanced iff every cell is within 1 of the maximum take OR was drained. See the
    "invariant about NON-EXHAUSTED cells" paragraph above for why the obvious flat-spread
    check is the wrong one."""
    if not taken:
        return True, "no cells"
    mx = max(taken.values())
    bad = {k: v for k, v in taken.items() if v < mx - 1 and v < sizes.get(k, 0)}
    return (not bad), (f"max take {mx}; under-taken and NOT drained: {bad}" if bad
                       else f"max take {mx}, all cells within 1 or drained")


def sequence_by_deficit(sizes: Mapping[Hashable, int], *,
                        tie_key: Callable[[Hashable], Sequence] | None = None) -> list:
    """The cell sequence, length `sum(sizes)`, holding every cell NEAR its share in EVERY
    PREFIX. Greedy largest-deficit — Webster/Sainte-Laguë sequencing.

    At position L the next slot goes to the cell whose running count is furthest below its
    proportional share L*n_c/N. Returns the CELL KEY per position; the caller pops its own
    rows off that cell in whatever within-cell order it owns (rank, seeded shuffle, or the
    incoming order untouched), which is why this takes counts and not rows.

    `tie_key(cell) -> tuple` breaks equal deficits and defaults to `(size, cell)` — larger
    cell first, then the key itself, so the sequence is a pure function of `sizes`. Pass a
    SEEDED per-cell jitter where equal-deficit cells must not resolve in the same direction at
    every collision (which would group one cell ahead of another for the whole sequence).

    The ±1 prefix bound is what a truncating consumer needs, and this rule is the best of the
    two by a wide margin — but it is a CHECK, not a theorem (module docstring: tight at two
    cells, 1.495 on a skewed 13-cell population). Callers assert the bound on the order they
    built; `prefix_deviation` below is the metric they assert on.
    """
    keys = list(sizes)
    n = {k: int(sizes[k]) for k in keys}
    N = sum(n.values())
    if N <= 0:
        return []
    tie = tie_key or (lambda k: (n[k], k))
    taken = {k: 0 for k in keys}
    out = []
    for L in range(1, N + 1):
        k = max((k for k in keys if taken[k] < n[k]),
                key=lambda k: (L * n[k] / N - taken[k], *tie(k)))
        taken[k] += 1
        out.append(k)
    return out


def prefix_deviation(seq: Sequence, sizes: Mapping[Hashable, int]) -> float:
    """max over prefixes and cells of |count - L*n_c/N| — the number the ±1 claim is about.

    Exposed rather than inlined in the test because two callers (`build_combined_label_sheet`
    and the dive planner) check it on their BUILT order, which is the assertion that survives
    a change to the rule."""
    keys = list(sizes)
    N = sum(int(sizes[k]) for k in keys)
    if N <= 0:
        return 0.0
    taken = {k: 0 for k in keys}
    worst = 0.0
    for L, k in enumerate(seq, 1):
        taken[k] = taken.get(k, 0) + 1
        for c in keys:
            worst = max(worst, abs(taken[c] - L * int(sizes[c]) / N))
    return worst
