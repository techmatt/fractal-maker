"""Tests for the cross-run saturation memory (`visited_density.py`).

Three properties, each with the thing that would otherwise let it pass green and useless:
  * the GRID equals the DEFINITION — pinned against `density_brute`, which is the linear scan
    of the same statement, on randomized populations chosen to straddle cell boundaries. Not
    the §1.10 `f(x) == f(x)` defect: the oracle is the definition the index accelerates, and
    it lives in the module precisely so the test is not a second implementation of it.
  * IDENTITY SEPARATES PLANES — two julia views at the same z with different seed c do not
    shadow each other. A z-only index reads them as one place, which is the "over-kill"
    failure `build_cloud`'s `row_ident` gate already exists to prevent.
  * the DISCOUNT IS SOFT — strictly positive at every density, so a saturated region is
    disfavoured and never unreachable.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

import visited_density as vd  # noqa: E402


# --------------------------------------------------------------------------- #
# The grid is exact.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_the_grid_returns_exactly_what_the_linear_definition_returns(seed):
    """The cell rule is `k*2^(o+1) >= k*fw` for every fw in octave o, so a covering disc's
    centre is always inside the 3x3 block. Randomized over four decades of fw and coordinates
    on the scale of the cells themselves, which is where a wrong block size shows."""
    rng = random.Random(seed)
    idx = vd.VisitedIndex(0.3)
    for _ in range(400):
        fw = 10.0 ** rng.uniform(-4, 0.5)
        idx.add("mandelbrot", None, rng.uniform(-2, 2), rng.uniform(-2, 2), fw)
    assert idx.n_visits == 400
    hits = 0
    for _ in range(600):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        got, want = (idx.density("mandelbrot", None, x, y),
                     idx.density_brute("mandelbrot", None, x, y))
        assert got == want, (x, y, got, want)
        hits += got > 0
    # Non-vacuity: an index that never covers anything would agree with the oracle trivially.
    assert hits > 50, f"fixture too sparse to discriminate ({hits} covered queries)"


def test_a_query_exactly_on_the_radius_is_covered_and_one_past_it_is_not():
    """The comparison is `<=` on squared distance, stated once here so a later `<` is a red
    test rather than a silent 1-row shift in every density."""
    idx = vd.VisitedIndex(2.0)
    idx.add("mandelbrot", None, 0.0, 0.0, 0.5)          # radius exactly 1.0
    assert idx.density("mandelbrot", None, 1.0, 0.0) == 1
    assert idx.density("mandelbrot", None, 1.0 + 1e-9, 0.0) == 0


def test_the_radius_is_the_VISITS_fw_not_the_querys():
    """The difference from `near_dup`, which scales on `min(a_fw, b_fw)`. A deep visit shadows
    almost nothing however wide the candidate looking at it is."""
    wide, deep = vd.VisitedIndex(1.0), vd.VisitedIndex(1.0)
    wide.add("mandelbrot", None, 0.0, 0.0, 1.0)
    deep.add("mandelbrot", None, 0.0, 0.0, 1e-6)
    assert wide.density("mandelbrot", None, 0.5, 0.0) == 1
    assert deep.density("mandelbrot", None, 0.5, 0.0) == 0


def test_counts_are_visits_not_places_so_a_re_walked_basin_reads_higher():
    """Density is a COUNT, which is what makes `1/(1+n)` a gradient rather than a boolean."""
    idx = vd.VisitedIndex(0.3)
    for i in range(5):
        idx.add("multibrot3", None, 1e-4 * i, 0.0, 0.1)
    assert idx.density("multibrot3", None, 0.0, 0.0) == 5


# --------------------------------------------------------------------------- #
# Identity separates dynamical planes.
# --------------------------------------------------------------------------- #
def test_two_julia_views_at_one_z_but_different_seed_c_do_not_shadow_each_other():
    idx = vd.VisitedIndex(1.0)
    idx.add("julia:mandelbrot", (-0.75, 0.10), 0.0, 0.0, 1.0)
    assert idx.density("julia:mandelbrot", (-0.75, 0.10), 0.1, 0.0) == 1
    assert idx.density("julia:mandelbrot", (0.30, -0.40), 0.1, 0.0) == 0
    # ...and a c-plane query never picks up a julia visit either.
    assert idx.density("julia:mandelbrot", None, 0.1, 0.0) == 0


def test_identities_inside_the_same_c_epsilon_share_a_bucket():
    """The hash bucket is the O(1) form of `near_dup`'s identity gate, on the SAME epsilon."""
    idx = vd.VisitedIndex(1.0)
    idx.add("julia:multibrot3", (-0.75, 0.10), 0.0, 0.0, 1.0)
    near = (-0.75 + vd.IDENT_EPS / 10, 0.10)
    far = (-0.75 + vd.IDENT_EPS * 100, 0.10)
    assert idx.density("julia:multibrot3", near, 0.0, 0.0) == 1
    assert idx.density("julia:multibrot3", far, 0.0, 0.0) == 0


def test_a_derived_partition_shares_its_base_partitions_memory():
    """`phoenix:classic` rows are stamped `family: phoenix` in every ledger in the tree, so the
    join has to be on the BASE partition — keying on the literal string would give the classic
    channel an empty memory that looked like a fresh plane."""
    idx = vd.VisitedIndex(1.0)
    ident = (0.5, -0.4, -0.5, 0.0, 0.0, 0.0)
    idx.add("phoenix", ident, 0.0, 0.0, 1.0)
    assert idx.density("phoenix:classic", ident, 0.1, 0.0) == 1


def test_add_row_reads_the_ledger_schema_through_row_ident():
    """PRESENCE-FROM-DISK for the row adapter: a julia ledger row's identity is its seed c,
    and string coordinates (the q4_harvest / classic_phoenix ledgers) are coerced."""
    idx = vd.VisitedIndex(1.0)
    assert idx.add_row(dict(family="julia:mandelbrot", julia_c_re="-0.75", julia_c_im="0.1",
                            outcome_cx="0.0", outcome_cy="0.0", outcome_fw="1.0"))
    assert idx.density("julia:mandelbrot", (-0.75, 0.1), 0.2, 0.0) == 1
    assert idx.density("julia:mandelbrot", (0.0, 0.0), 0.2, 0.0) == 0


# --------------------------------------------------------------------------- #
# Unusable rows are counted, not silently dropped.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row", [
    dict(family="mandelbrot", outcome_cx=None, outcome_cy=0.0, outcome_fw=1.0),
    dict(family="mandelbrot", outcome_cx=0.0, outcome_cy=0.0, outcome_fw=0.0),
    dict(family="mandelbrot", outcome_cx=0.0, outcome_cy=0.0, outcome_fw=-1.0),
    dict(family="mandelbrot", outcome_cx=float("nan"), outcome_cy=0.0, outcome_fw=1.0),
    dict(family="mandelbrot", outcome_cx=0.0, outcome_cy=0.0),
])
def test_an_unplaceable_row_is_refused_and_counted(row):
    idx = vd.VisitedIndex(0.3)
    assert idx.add_row(row) is False
    assert (idx.n_visits, idx.n_unusable) == (0, 1)
    assert idx.summary()["unusable_rows"] == 1


def test_a_zero_or_negative_radius_multiple_is_refused_at_construction():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            vd.VisitedIndex(bad)


# --------------------------------------------------------------------------- #
# The discount.
# --------------------------------------------------------------------------- #
def test_the_discount_is_soft_monotone_and_never_zero():
    prev = 1.0
    for d in range(1, 500):
        v = vd.discount(d, 1.0)
        assert 0.0 < v < prev, d
        prev = v
    assert vd.discount(0, 1.0) == 1.0
    assert vd.discount(1, 1.0) == pytest.approx(0.5)
    assert vd.discount(9, 1.0) == pytest.approx(0.10)      # the calibration knee


def test_strength_zero_is_exactly_one_at_every_density():
    """The off switch has to be EXACT, not approximately one: `--sat-strength 0` claims
    byte-identical priorities."""
    for d in (0, 1, 7, 10_000):
        assert vd.discount(d, 0.0) == 1.0


# --------------------------------------------------------------------------- #
# The ledger enumeration — the one owner shared with the freshness prior.
# --------------------------------------------------------------------------- #
def test_ledger_paths_finds_the_real_store_and_excludes_the_named_one():
    paths = vd.ledger_paths(ROOT)
    assert len(paths) > 5, "the committed discovery store is not being found at all"
    victim = paths[0]
    assert victim not in vd.ledger_paths(ROOT, exclude=victim)
    assert len(vd.ledger_paths(ROOT, exclude=victim)) == len(paths) - 1


def test_build_from_ledgers_over_the_real_store_is_non_empty_and_multi_partition():
    """PRESENCE-FROM-DISK for the production entry point. A memory that silently built empty
    would make the discount a no-op that still reported `status: on`."""
    idx = vd.build_from_ledgers(0.3, ROOT)
    s = idx.summary()
    assert s["visits"] > 10_000, s
    assert len(s["partitions"]) >= 8, s
    assert s["ledgers"] == len(vd.ledger_paths(ROOT))


def test_the_index_and_the_oracle_agree_on_the_REAL_store():
    """The randomized grid test proves the cell rule; this proves it on the actual population,
    whose fw spans nine decades and whose identity buckets are real."""
    idx = vd.build_from_ledgers(0.3, ROOT)
    rng = random.Random(7)
    hits = 0
    for _ in range(150):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        got, want = (idx.density("mandelbrot", None, x, y),
                     idx.density_brute("mandelbrot", None, x, y))
        assert got == want, (x, y, got, want)
        hits += got > 0
    assert hits > 0, "no query landed on a visited place — the comparison proved nothing"


def test_the_octave_cell_never_underflows_at_the_deepest_framewidths_in_the_store():
    """fw reaches 1e-9 in the ledgers; the cell size is `k*2^(o+1)` and the cell index is
    `x/cell`, so a deep octave is where an index would overflow or a cell would collapse."""
    idx = vd.VisitedIndex(0.3)
    for fw in (1e-9, 1e-7, 1e-3, 4.25):
        assert idx.add("mandelbrot", None, 1.9, -1.9, fw)
        assert idx.density("mandelbrot", None, 1.9, -1.9) > 0
        assert idx.density("mandelbrot", None, 1.9, -1.9) == \
            idx.density_brute("mandelbrot", None, 1.9, -1.9)
    assert math.isfinite(idx._cell_size(-30))
