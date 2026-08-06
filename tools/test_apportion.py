#!/usr/bin/env python
"""`tools/apportion.py` is the ONLY copy of the two apportionment rules.

Three things. (1) Each rule does what its docstring claims, stated as a PROPERTY over
populations rather than as a golden output. (2) Each claim is paired with a CONTROL that goes
red on the rule it is being distinguished from — a ±1 assertion that also passes on the wrong
dialect proves nothing, and that pairing is the point of this file. (3) A source scan: no
tracked Python file outside this module may re-declare either rule, the same gate
`test_partitions.py` puts on the partition map, and for the same reason — on 2026-08-05 there
were seven copies in two dialects across seven modules, none of them wrong and all of them
independently editable.

  uv run pytest tools/test_apportion.py -q
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import apportion as A  # noqa: E402

OWNER = "tools/apportion.py"
# This file quotes the copy-shapes it hunts for (in `test_the_scan_would_actually_catch_a_copy`)
# and implements the rejected dialect as a control, so it exempts itself.
EXEMPT = {OWNER, "tools/test_apportion.py"}
# The two rules as they were actually written in the seven copies: the round-robin's
# `key=lambda k: (take[k], -len(cells[k]))` (with or without a preseed term) and the deficit
# rule's `L * n[k] / N - taken[k]`. Both are matched on their ARITHMETIC, not on a function
# name, because a copy is a copy whatever it is called — but anchored on the running-count
# subscript rather than on `min(cand, ...)`, which the first draft matched and which caught
# `atlas_probe/efficiency_depth.py`'s unrelated "shallowest D" pick.
COPY = re.compile(r"key\s*=\s*lambda[^:]*:\s*\(\s*take\[|"
                  r"L\s*\*\s*n(?:_c)?\[\w+\]\s*/\s*N\s*-\s*taken\[")

# The sheet's real cell shape (batch x family), and the 13-cell skew that breaks the bound.
SHEET_SHAPE = {("a", "x"): 43, ("a", "y"): 36, ("b", "x"): 290, ("c", "y"): 48, ("c", "z"): 98}
SKEWED_13 = {"c00": 1, "c01": 13, "c02": 290, "c03": 1, "c04": 5, "c05": 21, "c06": 290,
             "c07": 1, "c08": 1, "c09": 290, "c10": 13, "c11": 40, "c12": 40}


def flat_spread(sizes):
    """THE REJECTED DIALECT, kept executable so every claim about it is measured.

    Lay each cell's rows at (i+0.5)/n_c and sort by that key — the round-robin rule
    generalized to a whole-set ordering. Cheap, obvious, and off the ±1 bound."""
    keyed = [(((i + 0.5) / n, str(c)), c) for c, n in sizes.items() for i in range(n)]
    keyed.sort(key=lambda t: t[0])
    return [c for _, c in keyed]


def _random_pop(rng, k=None):
    k = k or rng.randint(2, 15)
    return {f"c{i:02d}": rng.choice([1, 2, 3, 5, 8, 13, 21, 40, 80, 150, 290]) for i in range(k)}


# =========================================================================== #
# deal_round_robin — drawing a subset
# =========================================================================== #
def test_the_draw_is_balanced_or_drained_over_random_populations():
    """The acceptance predicate, over 500 populations: within 1 of the max take, or drained.
    Stated relationally, because real cells differ in supply by two orders of magnitude."""
    rng = random.Random(0)
    for _ in range(500):
        sizes = _random_pop(rng)
        n = rng.randint(1, sum(sizes.values()) + 20)
        take = A.deal_round_robin(sizes, n)
        ok, why = A.cells_balanced(take, sizes)
        assert ok, f"{why} on {sizes} at n={n}"
        assert sum(take.values()) == min(n, sum(sizes.values()))
        assert all(0 <= take[k] <= sizes[k] for k in sizes)


def test_a_drained_cell_does_not_make_the_draw_unbalanced():
    """Non-vacuity for the predicate: the flat-spread reading of "balanced" would call this
    draw broken, and it is correct. One cell has 1 row, one has 200."""
    sizes = {"tiny": 1, "big": 200}
    take = A.deal_round_robin(sizes, 50)
    assert take == {"tiny": 1, "big": 49}
    assert A.cells_balanced(take, sizes)[0]
    flat = max(take.values()) - min(take.values())
    assert flat == 48, "the flat spread is huge here — that is why it is not the predicate"


def test_preseed_makes_a_reservation_a_FLOOR_and_not_a_BONUS():
    """The whole reason `preseed` exists. A cell already credited with 10 must end at
    max(natural, 10), not natural + 10 — and the difference is exactly the reservation on a
    cell the balanced draw would have served anyway."""
    sizes = {"a": 100, "b": 100, "c": 100}
    natural = A.deal_round_robin(sizes, 30)
    assert natural == {"a": 10, "b": 10, "c": 10}
    seeded = A.deal_round_robin(sizes, 20, preseed={"a": 10})
    assert seeded["a"] == 0, "the credited cell drew again — the reservation acted as a bonus"
    assert {k: seeded[k] + (10 if k == "a" else 0) for k in sizes} == natural
    # ...and a reservation LARGER than the natural share still raises that cell's total
    big = A.deal_round_robin({"a": 100, "b": 100}, 10, preseed={"a": 40})
    assert big == {"a": 0, "b": 10}


def test_a_preseed_cell_absent_from_sizes_is_refused_not_ignored():
    """A drained reserved cell must be passed with size 0 so it is visible to the tie-breaks;
    silently dropping it would let a reservation vanish from the accounting."""
    with pytest.raises(KeyError, match="absent from `sizes`"):
        A.deal_round_robin({"a": 5}, 3, preseed={"gone": 2})
    assert A.deal_round_robin({"a": 5, "gone": 0}, 3, preseed={"gone": 2}) == {"a": 3, "gone": 0}


def test_the_draw_is_a_pure_function_of_the_mapping_order():
    """Determinism is the contract; the mapping's own order is what breaks exact ties, so a
    caller must hand over a sorted mapping (and the same mapping gives the same answer)."""
    sizes = {"b": 3, "a": 3}
    assert A.deal_round_robin(sizes, 3) == A.deal_round_robin(sizes, 3)
    assert A.deal_round_robin(sizes, 1) == {"b": 1, "a": 0}
    assert A.deal_round_robin({"a": 3, "b": 3}, 1) == {"a": 1, "b": 0}


def test_the_larger_cell_breaks_a_tie_so_the_remainder_lands_where_supply_is():
    sizes = {"small": 1, "large": 9}
    assert A.deal_round_robin(sizes, 1) == {"small": 0, "large": 1}


# =========================================================================== #
# sequence_by_deficit — ordering a whole set
# =========================================================================== #
def test_two_cells_hold_the_bound_at_every_size_which_is_what_the_dive_needs():
    """The dive's top/control interleave is two cells, and there the rule is tight: worst
    deviation 0.5 over every pair of sizes up to 60. This is the one place the ±1 statement
    is a property of the rule rather than of the population."""
    worst = 0.0
    for a in range(1, 61):
        for b in range(1, 61):
            sizes = {"top": a, "control": b}
            worst = max(worst, A.prefix_deviation(A.sequence_by_deficit(sizes), sizes))
    assert worst <= 0.5, worst


def test_the_bound_holds_on_the_populations_the_callers_actually_have():
    for sizes in (SHEET_SHAPE, {"top": 20, "control": 8}, {"a": 290, "b": 290, "c": 290},
                  {"a": 500, "b": 5, "c": 1}):
        dev = A.prefix_deviation(A.sequence_by_deficit(sizes), sizes)
        assert dev <= 1.0, f"{dev:.3f} on {sizes}"


def test_the_flat_spread_dialect_MISSES_the_bound_on_the_same_population():
    """THE CONTROL. Without it the assertion above is satisfied by any rule that happens to
    pass, and the two dialects would look interchangeable — which is how the eighth copy got
    written. On the sheet's own shape: deficit 0.738, flat spread 1.068."""
    got = A.prefix_deviation(A.sequence_by_deficit(SHEET_SHAPE), SHEET_SHAPE)
    bad = A.prefix_deviation(flat_spread(SHEET_SHAPE), SHEET_SHAPE)
    assert got == pytest.approx(0.738, abs=0.005)
    assert bad == pytest.approx(1.068, abs=0.005)
    assert bad > 1.0 >= got, "the two dialects stopped being distinguishable — check both"


def test_the_deficit_rule_is_NOT_universally_within_one_and_this_is_the_counterexample():
    """The claim the module refuses to make. 13 cells with two orders of magnitude of skew
    reach 1.495, so every caller asserts the bound on its BUILT order. If this test ever goes
    green-by-passing (deviation <= 1), the rule changed — do not delete the caller checks."""
    dev = A.prefix_deviation(A.sequence_by_deficit(SKEWED_13), SKEWED_13)
    assert dev == pytest.approx(1.495, abs=0.005)
    assert dev > 1.0
    # ...and the rejected dialect is still worse on the same population, so "greedy is not
    # perfect" is not a reason to go back to it.
    assert A.prefix_deviation(flat_spread(SKEWED_13), SKEWED_13) > dev


def test_the_sequence_is_a_permutation_of_the_cells_with_exact_multiplicity():
    rng = random.Random(1)
    for _ in range(200):
        sizes = _random_pop(rng)
        seq = A.sequence_by_deficit(sizes)
        assert len(seq) == sum(sizes.values())
        for k, n in sizes.items():
            assert seq.count(k) == n


def test_the_tie_key_is_what_a_caller_uses_to_stop_cells_resolving_the_same_way():
    """Equal cells tie at every collision; the default resolves by key, a seeded jitter
    resolves differently. Both deterministic — the sequence is a pure function of its inputs."""
    sizes = {"a": 5, "b": 5, "c": 5}
    assert A.sequence_by_deficit(sizes) == A.sequence_by_deficit(sizes)
    jitter = {c: random.Random(f"7|{c}").random() for c in sizes}
    seeded = A.sequence_by_deficit(sizes, tie_key=lambda k: (jitter[k], k))
    assert sorted(seeded) == sorted(A.sequence_by_deficit(sizes))
    assert seeded == A.sequence_by_deficit(sizes, tie_key=lambda k: (jitter[k], k))


def test_degenerate_inputs_are_answers_not_crashes():
    assert A.sequence_by_deficit({}) == []
    assert A.sequence_by_deficit({"a": 0}) == []
    assert A.sequence_by_deficit({"a": 1}) == ["a"]
    assert A.deal_round_robin({}, 5) == {}
    assert A.deal_round_robin({"a": 0}, 5) == {"a": 0}
    assert A.prefix_deviation([], {}) == 0.0


# =========================================================================== #
# the scan — no eighth copy
# =========================================================================== #
def _tracked_python():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [p for p in out.stdout.splitlines() if p.strip()]


def test_no_second_copy_of_either_rule_exists():
    """THE point of this file. Seven copies existed on 2026-08-05; an eighth is a fork with a
    delayed fuse, and the fuse is picking the dialect that does not hold the property."""
    offenders = []
    for rel in _tracked_python():
        if rel.replace("\\", "/") in EXEMPT:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in COPY.finditer(text):
            offenders.append(f"{rel}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        f"{len(offenders)} re-declaration(s) of an apportionment rule outside {OWNER}: "
        f"{offenders}. Import it instead — `import apportion` with tools/ on sys.path "
        f"(dependency-free, so there is no layering reason not to).")


def test_the_scan_would_actually_catch_a_copy():
    """Non-vacuity: the regex has to match the shapes the copies were actually written in —
    all the round-robin spellings and both deficit spellings from the pre-2026-08-05 tree."""
    samples = [
        "k = min(cand, key=lambda k: (take[k], -len(cells[k])))",
        "k = min(cand, key=lambda k: (take[k] + seed[k], -len(cells[k])))",
        "        k = min(cand,\n                key=lambda k: (take[k], -len(cells[k])))",
        "c = max(live, key=lambda c: (L * n_c[c] / N - taken[c], jitter[c], c))",
        "k = max(keys, key=lambda k: (L * n[k] / N - taken[k], n[k], k))",
    ]
    for s in samples:
        assert COPY.search(s), f"the scan would miss this copy:\n{s}"
    for ok in ("import apportion", "take = apportion.deal_round_robin(sizes, n)",
               "seq = apportion.sequence_by_deficit(sizes, tie_key=lambda k: (n[k], k))",
               "best = min(cand)", "# see apportion.py for the L * n_c / N rule",
               # the real false positive the first draft of this scan produced:
               'chosen = min(cand, key=lambda s: s["D"]) if cand else sweep[-1]'):
        assert not COPY.search(ok), f"the scan false-positives on:\n{ok}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
