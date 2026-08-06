"""The fresh-sheet builder's PURE decisions — binning, quota, pick spread, split, suggestion.

Everything under test here is what decides WHAT Matt ends up labeling, and every one of it
is a rule that would fail silently: a bin that swallows the gate boundary, a quota that
quietly shrinks the draw when one bin is thin, a pick rule that reverts to top-K, a split
that leaks a location across sides, a tier suggestion that collapses onto one class. None of
those raise; they just produce a worse sheet, discovered after the labeling sitting.

The rendering half is not tested here — it is the shared `label_crop.py` tail every existing
wallpaper batch already runs through, and the builder asserts each crop's realized size.

  uv run pytest tools/wallpaper/test_fresh_sheet.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wallpaper import suggest_tier as ST      # noqa: E402


def _load(name):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def B():
    return _load("build_fresh_sheet.py")


@pytest.fixture(scope="module")
def CS():
    return _load("build_colorize_sheet.py")


# --------------------------------------------------------------------------- #
# Synthetic screen population — no engine, no head, no ledgers.
# --------------------------------------------------------------------------- #
def _screen_pop(B, n=600, seed=0):
    """`n` screen records, ROUND-ROBINED across the bins, a third floor-admitted.

    The location's best score is placed INSIDE an assigned bin rather than swept over [0,1]:
    the bins are unequal in width, so a uniform sweep leaves the narrow low bins with a
    handful of members and every "spans the range" assertion below would then be measuring
    the fixture rather than the draw. Each location's 8 candidates fan down from its best so
    the pick rule has a real within-location range to spread over."""
    rng = np.random.default_rng(seed)
    nb = len(B.BIN_LABELS)
    recs = []
    for i in range(n):
        b = i % nb
        lo, hi = B.SCORE_BINS[b], min(B.SCORE_BINS[b + 1], 1.0)
        best = lo + (hi - lo) * ((i // nb) + 0.5) / (n / nb)      # deterministic, inside the bin
        cands = [{"p_ge3": float(np.clip(best * (j + 1) / 8, 0.0, 1.0)),
                  "pred": 1.0 + 3.0 * best * (j + 1) / 8,
                  "palette": f"pal_{i}_{j}", "palette_type": "cyclic",
                  "palette_source": "dramatic", "config": {}}
                 for j in range(8)]
        rng.shuffle(cands)                       # the record's order must not matter
        floor = (i % 3 == 0)
        recs.append({
            "unit_key": f"u{i:04d}", "key": f"k{i:04d}", "family": "mandelbrot",
            "intake_source": "human_q3plus_seed" if floor else "union_ledger",
            "source_tag": "q4_harvest" if floor else "dive",
            "floor_admit": floor, "partition": "mandelbrot",
            "fw": "0.1", "maxiter": 4000, "error": None,
            "candidates": cands,
        })
    return recs


def test_the_fixture_populates_every_bin(B):
    """Non-vacuity for everything below: the spanning assertions can only mean something if
    the synthetic population actually reaches all five bins with room to spare."""
    hist = Counter(B.bin_of(max(c["p_ge3"] for c in r["candidates"])) for r in _screen_pop(B))
    assert set(hist) == set(range(len(B.BIN_LABELS))), hist
    assert min(hist.values()) >= 100, hist


# --------------------------------------------------------------------------- #
# Bins
# --------------------------------------------------------------------------- #
def test_the_bins_tile_the_whole_score_range_with_no_gap(B):
    """Every p_ge3 a head can emit lands in exactly one bin. A gap here silently drops
    locations out of the draw; an overlap double-counts them into a quota."""
    assert len(B.SCORE_BINS) == len(B.BIN_LABELS) + 1
    assert list(B.SCORE_BINS) == sorted(B.SCORE_BINS)
    seen = set()
    for p in np.linspace(0.0, 1.0, 1001):
        b = B.bin_of(float(p))
        assert 0 <= b < len(B.BIN_LABELS), p
        seen.add(b)
    assert seen == set(range(len(B.BIN_LABELS))), "a bin is unreachable across [0,1]"
    assert B.bin_of(0.0) == 0 and B.bin_of(1.0) == len(B.BIN_LABELS) - 1


def test_the_deployed_gate_is_a_bin_EDGE_not_a_bin_interior(B):
    """The sheet exists to span both sides of the emission gate. If 0.90 fell inside a bin,
    that bin would mix "would ship" with "would not" and the stratification could not show
    the boundary at all."""
    from tools.wallpaper.wallpaper_pins import GATE_THRESHOLD
    assert GATE_THRESHOLD in B.SCORE_BINS, (GATE_THRESHOLD, B.SCORE_BINS)
    assert B.bin_of(GATE_THRESHOLD - 1e-9) != B.bin_of(GATE_THRESHOLD)


# --------------------------------------------------------------------------- #
# Quota
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("avail,quota", [
    ([100, 100, 100, 100, 100], 240),      # everyone can pay
    ([3, 400, 400, 400, 400], 240),        # one thin bin — the deficit must be redistributed
    ([0, 0, 500, 0, 0], 240),              # degenerate: only one bin exists
    ([10, 10, 10, 10, 10], 240),           # supply-bound: the draw is smaller, not padded
])
def test_the_quota_conserves_and_never_overdraws(B, avail, quota):
    take = B._apportion(quota, avail)
    assert all(t <= a for t, a in zip(take, avail)), (take, avail)
    assert sum(take) == min(quota, sum(avail)), (take, avail, quota)


def test_a_thin_bin_does_not_shrink_the_draw(B):
    """The failure this guards: a bin with 3 members caps at 3 and the OTHER bins keep their
    original equal share, so a 240-location draw silently lands 195. The redistribution is
    what keeps the target met whenever the population can meet it."""
    take = B._apportion(240, [3, 400, 400, 400, 400])
    assert sum(take) == 240
    assert take[0] == 3
    assert min(take[1:]) >= 59, take          # the deficit went somewhere, evenly


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_the_draw_spans_every_bin_and_hits_the_target(B):
    recs = _screen_pop(B)
    sel, rep = B.select(recs, target_locs=240, seed=7)
    assert len(sel) == 240 == rep["drawn_locations"]
    drawn = Counter(r["bin"] for r in sel)
    assert set(drawn) == set(range(len(B.BIN_LABELS))), drawn
    assert min(drawn.values()) >= 20, drawn   # "spans the range" is a floor, not a whisper


def test_the_floor_admitted_sources_are_oversampled_INSIDE_every_bin(B):
    """The oversample must not be a score shift wearing a source label: it is applied within
    each bin, so the floor-admit share is lifted above its population share everywhere and
    the bin composition is unchanged."""
    recs = _screen_pop(B)
    sel, rep = B.select(recs, target_locs=240, seed=7)
    pop_frac = sum(1 for r in recs if r["floor_admit"]) / len(recs)
    assert rep["floor_admit_frac_realized"] > pop_frac + 0.05, (rep["floor_admit_frac_realized"], pop_frac)
    assert rep["floor_admit_frac_realized"] == pytest.approx(B.FLOOR_ADMIT_FRAC, abs=0.05)
    for b in rep["per_bin"]:
        if b["drawn"] and b["available_floor_admit"] >= b["drawn"] * B.FLOOR_ADMIT_FRAC:
            assert b["drawn_floor_admit"] / b["drawn"] == pytest.approx(B.FLOOR_ADMIT_FRAC, abs=0.06), b


def test_the_draw_is_reproducible_from_its_seed_and_moves_with_it(B):
    """Both halves. Same seed -> same sheet (a resumed build must not re-draw); different
    seed -> a different sheet (else the seed is decorative and the first assert is vacuous)."""
    recs = _screen_pop(B)
    a = [r["unit_key"] for r in B.select(_screen_pop(B), 240, 7)[0]]
    b = [r["unit_key"] for r in B.select(recs, 240, 7)[0]]
    c = [r["unit_key"] for r in B.select(recs, 240, 11)[0]]
    assert a == b
    assert a != c


def test_a_screen_failure_is_excluded_and_counted_not_silently_dropped(B):
    recs = _screen_pop(B, n=120)
    recs[0]["error"] = "RuntimeError: dump-field failed"
    recs[0]["candidates"] = []
    sel, rep = B.select(recs, target_locs=60, seed=7)
    assert rep["screen_failures"] == 1
    assert recs[0]["unit_key"] not in {r["unit_key"] for r in sel}


# --------------------------------------------------------------------------- #
# Per-location picks
# --------------------------------------------------------------------------- #
def test_the_picks_span_the_location_not_its_top_slice(B):
    """The prompt's "negatives and mid-range, not a top slice" applied WITHIN a location:
    the picks must include the location's worst and best screened candidate, on distinct
    palettes. A top-K regression passes every other test in this file."""
    recs = _screen_pop(B, n=20)
    for rec in recs[5:]:                      # skip the all-zero low end
        picks = B.pick_candidates(rec, 4)
        ps = [c["p_ge3"] for c in rec["candidates"]]
        got = [c["p_ge3"] for c in picks]
        assert len(picks) == 4
        assert len({c["palette"] for c in picks}) == 4, "picks must be palette-distinct"
        assert min(got) == pytest.approx(min(ps))
        assert max(got) == pytest.approx(max(ps))


def test_a_location_with_fewer_candidates_than_picks_yields_what_it_has(B):
    rec = _screen_pop(B, n=20)[10]
    rec["candidates"] = rec["candidates"][:2]
    assert len(B.pick_candidates(rec, 4)) == 2


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def test_the_split_is_location_disjoint_and_bin_stratified(B):
    recs = _screen_pop(B)
    sel, _rep = B.select(recs, target_locs=240, seed=7)
    sides, n_eval = B.assign_split(sel)
    assert set(sides) == {r["unit_key"] for r in sel}       # one side per LOCATION, always
    assert set(sides.values()) == {"train", "eval"}
    assert n_eval == sum(1 for v in sides.values() if v == "eval")
    assert n_eval / len(sel) == pytest.approx(B.EVAL_FRAC, abs=0.02)
    # stratified: every bin contributes eval-side locations, so eval spans the range too
    by_bin = Counter(r["bin"] for r in sel if sides[r["unit_key"]] == "eval")
    assert set(by_bin) == set(range(len(B.BIN_LABELS))), by_bin


def test_the_split_is_pinned_to_its_seed(B):
    sel, _ = B.select(_screen_pop(B), 240, 7)
    assert B.assign_split(sel, seed=0)[0] == B.assign_split(sel, seed=0)[0]
    assert B.assign_split(sel, seed=0)[0] != B.assign_split(sel, seed=1)[0]


# --------------------------------------------------------------------------- #
# The tier suggestion
# --------------------------------------------------------------------------- #
def test_the_cut_rule_is_monotone_and_lands_in_range():
    prev = 0
    for p in np.linspace(1.0, 4.0, 601):
        t = ST.tier_from_pred(float(p))
        assert 1 <= t <= ST.K_TIERS
        assert t >= prev, "a higher expected tier must never suggest a lower tier"
        prev = t
    assert ST.tier_from_pred(1.0) == 1
    assert ST.tier_from_pred(4.0) == ST.K_TIERS


def test_the_frozen_cuts_are_ascending_and_agree_with_their_own_deriver():
    assert list(ST.CUTS) == sorted(ST.CUTS)
    assert len(ST.CUTS) == ST.K_TIERS - 1
    assert ST.DERIVATION["cuts"] == list(ST.CUTS), "the record and the constant disagree"
    # `fit_cuts` is prior-matching by construction: re-deriving on a slice whose tiers were
    # ASSIGNED by the frozen cuts must return them (up to the rounding they were frozen at).
    pred = np.linspace(1.0, 4.0, 4001)
    tiers = np.array([ST.tier_from_pred(float(p)) for p in pred])
    assert ST.fit_cuts(pred, tiers) == pytest.approx(ST.CUTS, abs=2e-3)


def test_the_rule_reproduces_a_slices_tier_prior_rather_than_collapsing():
    """THE property the chosen rule was chosen for. The rejected accuracy-maximizing cut put
    69% of the derivation slice on tier 2; prior-matched cuts reproduce the label histogram.
    Stated as an invariant of the DERIVER over an arbitrary slice, not as a claim about the
    frozen cuts on some synthetic distribution — the frozen cuts are absolute and a genuinely
    worse population is SUPPOSED to skew low, so "never collapses" is not true of them and
    must not be asserted."""
    rng = np.random.default_rng(0)
    pred = 1.0 + 3.0 * rng.beta(0.6, 1.4, size=3000)
    tiers = rng.choice([1, 2, 3, 4], size=3000, p=[0.17, 0.43, 0.27, 0.13])
    tiers = np.sort(tiers)[np.argsort(np.argsort(pred))]      # monotone-ish association
    cuts = ST.fit_cuts(pred, tiers)
    got = Counter(ST.tier_from_pred(float(p), cuts) for p in pred)
    want = Counter(int(t) for t in tiers)
    for t in (1, 2, 3, 4):
        assert got[t] == pytest.approx(want[t], rel=0.02, abs=2), (got, want)


def test_the_recorded_derivation_carries_the_non_collapse_it_claims():
    """The frozen cuts' own record: the suggested histogram it produced on the derivation
    slice matches that slice's tier prior and puts no more than half the rows on one tier.
    Keyed here so the claim in the docstring cannot rot into prose."""
    d = ST.DERIVATION
    prior, hist = d["tier_prior"], d["accuracy_on_slice"]["suggested_hist"]
    assert set(prior) == set(hist) == {"1", "2", "3", "4"}
    assert sum(hist.values()) == sum(prior.values()) == d["n"]
    for t in prior:
        assert abs(hist[t] - prior[t]) <= 3, (t, hist, prior)
    assert max(hist.values()) / d["n"] < 0.50, hist
    # ...and it beats the rule it replaced on the axis it was chosen on.
    assert d["accuracy_on_slice"]["exact"] > d["alternatives_rejected"]["corn_0.5"]["exact"]


def test_expected_tier_matches_the_head_readout_definition():
    assert ST.expected_tier([0.0, 0.0, 0.0]) == 1.0
    assert ST.expected_tier([1.0, 1.0, 1.0]) == 4.0
    assert ST.expected_tier([0.9, 0.5, 0.1]) == pytest.approx(2.5)


def test_fit_cuts_refuses_an_empty_or_misaligned_slice():
    with pytest.raises(ValueError):
        ST.fit_cuts([], [])
    with pytest.raises(ValueError):
        ST.fit_cuts([1.0, 2.0], [1])


# --------------------------------------------------------------------------- #
# Wiring the builder cannot be allowed to drift on
# --------------------------------------------------------------------------- #
def test_the_builder_renders_at_the_shared_label_crop_pins(B):
    """The batch is only unionable with the other three if its crops come off the SAME
    geometry. Pinned against `label_crop`'s constants, not restated numbers."""
    from tools.wallpaper import label_crop as LC
    assert (B.LABEL_W, B.LABEL_H, B.LABEL_SS) == (LC.LABEL_W, LC.LABEL_H, LC.LABEL_SS)
    assert B.LABEL_FILTER == LC.LABEL_FILTER


def test_the_builder_reads_the_head_from_the_pin_and_never_a_literal(B):
    """`wallpaper_pins` is the one place the live head is named. A literal here is how a
    batch gets stamped with a head it was not scored by."""
    from tools.wallpaper import wallpaper_pins as WP
    src = (HERE / "build_fresh_sheet.py").read_text(encoding="utf-8")
    assert "data/wallpaper_head/v" not in src, "the head pin is hardcoded in the builder"
    assert WP.HEAD_CKPT.exists(), f"the pinned head is missing: {WP.HEAD_CKPT}"


def test_the_worker_cap_is_the_project_cap(B):
    assert B.LABEL_CROP_WORKERS <= 4


# --------------------------------------------------------------------------- #
# The two coloring regimes (the addendum's whole point)
# --------------------------------------------------------------------------- #
def test_the_two_batches_are_separable_by_coloring_source(B, CS):
    """A retrain that unions the sheets must be able to split them by regime. Nothing else
    in a row distinguishes a pool-draw coloring from a colorize-path one — same intake, same
    population, same label-crop pins, same pre-label — so if these two strings ever collide
    the two regimes become one undifferentiated blob in the corpus."""
    assert B.COLORING_SOURCE == "pool_draw"
    assert CS.COLORING_SOURCE == "colorize_path"
    assert B.COLORING_SOURCE != CS.COLORING_SOURCE
    assert B.BATCH_ID != CS.BATCH_ID
    assert B.IMG_PREFIX != CS.IMG_PREFIX          # ids must not collide in a joint sitting
    assert B.LABELS_EXPORT != CS.LABELS_EXPORT    # ...nor their label exports


def test_both_batches_render_at_the_same_pins_so_the_regime_is_the_only_difference(B, CS):
    """ss-level or geometry correlating with regime would be a batch effect on exactly the
    axis the pair exists to isolate."""
    assert (CS.LABEL_W, CS.LABEL_H, CS.LABEL_SS, CS.LABEL_FILTER) == \
           (B.LABEL_W, B.LABEL_H, B.LABEL_SS, B.LABEL_FILTER)


def test_the_colorize_sheet_reuses_the_siblings_draw_rather_than_copying_it(CS, B):
    """`select`/`assign_split`/the bins are imported, not re-declared: a second copy of the
    stratification is a second sheet composition nobody decided, and only one of them would
    be covered by the tests above."""
    assert CS.FS is not None
    src = (HERE / "build_colorize_sheet.py").read_text(encoding="utf-8")
    for owned in ("SCORE_BINS = (", "def select(", "def assign_split(", "def bin_of("):
        assert owned not in src, f"{owned!r} is re-declared instead of imported from the sibling"
    assert CS.RENDER_STYLES == ("smooth",), "the addendum pins smooth; strange routes to another head"


def test_the_colorize_sheet_applies_no_gate(CS):
    """The pool floor is READ for deficit bookkeeping only. If it ever filtered rows, the
    sheet would stop spanning the range — which is the one thing both batches are for."""
    src = (HERE / "build_colorize_sheet.py").read_text(encoding="utf-8")
    assert "passed_pool_floor" in src
    # no row is dropped on the floor: `passed` only feeds record_fill
    assert "if passed:\n            model.record_fill(cell)" in src
    assert "continue" not in src.split("passed = bool(")[1].split("row = {")[0]
