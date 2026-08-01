"""Acceptance for the view-level screen (`tools/atlas/view_screen.py`) and its retroactive
driver (`view_rescreen.py`).

Two things are being guarded and they fail differently, so they are tested differently:

  * the MEASURES are behavioural — every bracket below asserts a *relation* on a synthetic
    field (a concentrated dead half scores below a spread one at the same tile mean; a deep
    well in a flat field scores high on `radial_range` and low on coverage). Those survive a
    re-measurement, which the frozen-literal kind would not
    (`verification_practice.md` §7).
  * the VALIDATION GATE is an outcome, so it is pinned against the recorded artifact the
    way `test_orbital.py` pins `validation.json` — including the formulation that FAILED,
    so the negative result stays reported rather than being tuned away.

Run:  uv run python -m pytest tools/atlas/test_view_screen.py -q
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import view_screen as vs          # noqa: E402
import field_metrics as fm        # noqa: E402

W, H = fm.SCREEN_W, fm.SCREEN_H                 # 64 x 36
TW, TH = W // vs.GRID_X, H // vs.GRID_Y         # 4 x 4 px tiles
CYCLE = 1.0 / vs.DENSITY                        # 40 iterations == one colour cycle

REFS = ROOT / "data" / "atlas" / "view_screen_refs.json"
GATE = ROOT / "data" / "atlas" / "view_screen_gate.json"


# --------------------------------------------------------------------------- #
# synthetic fields — a tile participates iff it spans >= 1 cycle and is >=25% finite
# --------------------------------------------------------------------------- #
def _field_from_tile_mask(mask: np.ndarray) -> np.ndarray:
    """A field whose participating tiles are exactly `mask` (a `GRID_Y x GRID_X` bool).

    A participating tile is filled with a 1.5-cycle ramp; a non-participating one is
    constant. Both are fully finite, so `interior_fraction` is 0 and only the coverage
    reduction is under test.
    """
    f = np.zeros((H, W), dtype="f4")
    ramp = (np.linspace(0.0, 1.5 * CYCLE, TW)[None, :] * np.ones((TH, 1)))
    for ty in range(vs.GRID_Y):
        for tx in range(vs.GRID_X):
            if mask[ty, tx]:
                f[ty * TH:(ty + 1) * TH, tx * TW:(tx + 1) * TW] = ramp
    return f


def test_a_tile_participates_exactly_when_it_spans_a_colour_cycle():
    """The floor is the render-visible unit and it is phase-independent: 0.9 of a cycle
    does not participate wherever it sits, 1.1 does. A `floor(t)` test would pass or fail
    the same tile depending on where the palette happened to be."""
    for cycles, want in ((0.9, 0.0), (1.1, 1.0)):
        f = np.zeros((H, W), dtype="f4")
        f[:TH, :TW] = np.linspace(0.0, cycles * CYCLE, TW)[None, :]
        assert vs.band_coverage(f) * (vs.GRID_X * vs.GRID_Y) == pytest.approx(want)
        # ...and shifting the whole field by a third of a cycle changes nothing.
        assert vs.band_coverage(f + CYCLE / 3) * (vs.GRID_X * vs.GRID_Y) == pytest.approx(want)


def test_a_flat_field_covers_nothing_and_a_dense_one_covers_everything():
    assert vs.band_coverage(np.full((H, W), 300.0, dtype="f4")) == 0.0
    assert vs.band_coverage_q25(np.full((H, W), 300.0, dtype="f4")) == 0.0
    full = _field_from_tile_mask(np.ones((vs.GRID_Y, vs.GRID_X), bool))
    assert vs.band_coverage(full) == 1.0
    assert vs.band_coverage_q25(full) == 1.0


def test_q25_separates_concentrated_dead_area_from_spread_dead_area():
    """THE reason the composite selects on `band_coverage_q25` and not on the tile mean.

    Two fields with the SAME tile mean (0.5) and opposite spatial arrangements: half the
    frame solidly dead vs a checkerboard. The mean cannot tell them apart — that is the
    defect, asserted here rather than described — and the pooled quantile can.
    """
    solid = np.zeros((vs.GRID_Y, vs.GRID_X), bool)
    solid[:, : vs.GRID_X // 2] = True
    checker = (np.indices((vs.GRID_Y, vs.GRID_X)).sum(axis=0) % 2).astype(bool)
    assert solid.mean() == checker.mean() == 0.5           # the fixture's whole point

    a, b = _field_from_tile_mask(solid), _field_from_tile_mask(checker)
    assert vs.band_coverage(a) == vs.band_coverage(b) == pytest.approx(0.5)
    assert vs.band_coverage_q25(a) == 0.0
    assert vs.band_coverage_q25(b) == pytest.approx(0.5)
    assert vs.band_coverage_q25(a) < vs.band_coverage_q25(b)


def test_a_deep_well_in_a_flat_field_scores_high_range_and_low_coverage():
    """The named Q5 failure, as a synthetic: one deep pocket sets a large radial span
    while the frame is otherwise flat. `radial_range` cannot see it (every ray starts at
    the centre, so the well raises all 64 of them); coverage can."""
    f = np.full((H, W), 200.0, dtype="f4")
    f += np.linspace(0.0, 0.5 * CYCLE, W)[None, :]      # a background too gentle to band
    cy_, cx_ = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    well = ((yy - cy_) ** 2 + (xx - cx_) ** 2) < 4 ** 2
    f[well] = 200.0 + 30 * CYCLE                        # one deep pocket
    import rescore_lib as rl
    assert rl.ring_measures(f)["radial_range"] > 20.0   # "rich" by the old screen
    assert vs.band_coverage(f) < 0.10                   # ...and it is one small pocket
    assert vs.band_coverage_q25(f) == 0.0


def test_an_all_interior_field_reduces_to_zero_without_warning():
    """An all-NaN tile is the normal case here, not an edge case — `nanmax` over one would
    emit a RuntimeWarning per tile per field over 16k fields."""
    f = np.full((H, W), np.nan, dtype="f4")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert vs.band_coverage(f) == 0.0 and vs.band_coverage_q25(f) == 0.0
    assert not caught, [str(c.message) for c in caught]


def test_a_black_tile_with_a_bright_rim_does_not_count_as_participating():
    """`TILE_MIN_FINITE` is what stops a thin escaping rim from crediting the dead tiles it
    runs through. One finite pixel out of sixteen, spanning cycles, is still a black tile."""
    f = np.full((H, W), np.nan, dtype="f4")
    f[0, 0], f[1, 1] = 0.0, 5 * CYCLE           # 2/16 finite = 0.125 < TILE_MIN_FINITE
    assert vs.band_coverage(f) == 0.0
    f[2, 2], f[3, 3] = 0.0, 5 * CYCLE           # 4/16 = 0.25 >= the floor
    assert vs.band_coverage(f) > 0.0


# --------------------------------------------------------------------------- #
# the composite and the veto
# --------------------------------------------------------------------------- #
def _m(interior, cov_q25, rng=4.0, rings=9.0):
    return dict(screened=True, interior_fraction=interior, band_coverage=cov_q25,
                band_coverage_q25=cov_q25, radial_range=rng, radial_rings=rings)


def test_the_veto_sorts_to_bottom_and_never_excludes():
    """"Recording, never exclusion": a vetoed row still scores, still orders against the
    other vetoed rows by the same quantity, and sits strictly below EVERY non-vetoed row —
    including a non-vetoed row that scores exactly zero."""
    veto = 0.34
    good = [_m(0.10, 0.8), _m(0.10, 0.4), _m(0.30, 0.0)]        # 0.0 is the tightest case
    bad = [_m(0.90, 0.8), _m(0.90, 0.4)]
    gs, bs = [vs.composite(m, veto) for m in good], [vs.composite(m, veto) for m in bad]
    assert all(not vs.is_vetoed(m, veto) for m in good)
    assert all(vs.is_vetoed(m, veto) for m in bad)
    assert min(gs) > max(bs), (gs, bs)
    assert bs[0] > bs[1], "order among vetoed rows must be preserved, not collapsed"
    assert all(np.isfinite(x) for x in gs + bs)


def test_an_unscreened_row_is_ranked_below_everything_including_the_vetoed():
    assert vs.composite(dict(screened=False), 0.34) == float("-inf")
    assert not vs.is_vetoed(dict(screened=False), 0.34)


def test_richness_moves_with_either_ring_measure():
    """The property the geometric mean was chosen FOR: it cannot inherit either measure's
    single-reference failure, because it moves when either one moves. Not a claim that a
    `range`-only term fails the gate — on this population it does not
    (`orbital_field_metrics.md` §11); the pair is kept because §4's failure is real and
    the gate does not discriminate between the three candidate terms."""
    base = _m(0.0, 0.5, rng=4.0, rings=9.0)
    assert vs.richness(_m(0.0, 0.5, rng=8.0, rings=9.0)) > vs.richness(base)
    assert vs.richness(_m(0.0, 0.5, rng=4.0, rings=18.0)) > vs.richness(base)
    assert vs.richness(_m(0.0, 0.5, rng=0.0, rings=99.0)) == 0.0


def test_the_veto_is_derived_from_the_references_not_hardcoded():
    """Derive in code, freeze in records: moving the reference measurement must move the
    veto. A literal in the source would not, which is how a threshold outlives the
    calibration it was taken from."""
    mk = lambda i: {"references": {"r": {"screened": True, "interior_fraction": i}}}
    assert vs.interior_veto(mk(0.0)) == pytest.approx(1.0 - vs.VETO_ESCAPED_SHARE, abs=1e-4)
    # A reference with MORE interior relaxes the cut, because the veto is a statement
    # about the references' escaping share and not an absolute interior level. The
    # direction matters: a stricter reference must not silently loosen the screen.
    assert vs.interior_veto(mk(0.5)) > vs.interior_veto(mk(0.0))
    # ...and it is the WEAKEST reference that sets it, not the mean.
    both = {"references": {"a": {"screened": True, "interior_fraction": 0.0},
                           "b": {"screened": True, "interior_fraction": 0.5}}}
    assert vs.interior_veto(both) == vs.interior_veto(mk(0.5))
    with pytest.raises(ValueError):
        vs.interior_veto({"references": {"r": {"screened": False}}})


def test_the_veto_anchored_on_interior_instead_of_escaped_would_be_degenerate():
    """The formulation that was tried FIRST and discarded, pinned so it is not re-proposed:
    both references measure ~0 interior, so ANY multiple of the reference INTERIOR is a
    hair above zero. Recorded as a property of the references, off the committed record."""
    refs = json.loads(REFS.read_text(encoding="utf-8"))["references"]
    worst = max(float(r["interior_fraction"]) for r in refs.values())
    assert worst < 0.05, "if a reference ever gains real interior, re-decide this on purpose"
    assert 2.0 * worst < 0.10 < vs.interior_veto(json.loads(REFS.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# the framing sweep
# --------------------------------------------------------------------------- #
def test_the_sweep_grid_is_deterministic_and_starts_at_the_view_itself():
    a = vs.sweep_windows("-0.5", "0.1", 1e-3)
    b = vs.sweep_windows("-0.5", "0.1", 1e-3)
    assert a == b, "the sweep has no RNG — two calls must be identical"
    assert len(a) == len(vs.SWEEP_OFFSETS) ** 2 * len(vs.SWEEP_SCALES) == 18
    assert a[0]["is_origin"] and sum(w["is_origin"] for w in a) == 1
    assert a[0]["cx"] == "-0.5" or float(a[0]["cx"]) == pytest.approx(-0.5)
    assert float(a[0]["cy"]) == pytest.approx(0.1) and a[0]["fw"] == pytest.approx(1e-3)
    assert {w["scale"] for w in a} == set(vs.SWEEP_SCALES)


def test_the_sweep_offsets_keep_every_digit_of_a_deep_centre():
    """The coordinate rule, one level up. NOT a claim that f64 cannot see this offset — at
    `fw = 1.5e-10` (this population's deepest) against a centre of magnitude ~0.75 it can,
    by five orders of magnitude. The claim is the one that survives the next decade of
    depth: the offset is applied WITHOUT truncating the centre, so all 35 committed digits
    come back, where an f64 round trip keeps 17 and silently discards the rest."""
    from decimal import Decimal
    cx = "-0.74977483272365342795786040375088960"
    fw = 1.5e-10
    wins = vs.sweep_windows(cx, "0.10761724352653678278696798751738616", fw)
    off = [w for w in wins if w["dx"] == 0.5 and w["dy"] == 0.0 and w["scale"] == 1.0][0]
    assert off["cx"] != cx
    assert Decimal(off["cx"]) - Decimal(cx) == Decimal(repr(0.5 * fw))   # exact, not approx
    # the tail the f64 path would have thrown away is still there
    assert off["cx"].endswith("795786040375088960")
    assert len(off["cx"].lstrip("-0.")) >= 30
    assert len(repr(float(cx)).lstrip("-0.")) <= 17


def test_sweep_best_keeps_the_original_frame_when_nothing_beats_it():
    """Ties break on the fixed window order, which puts the origin first — so a flat
    landscape returns the view unmoved rather than an arbitrary neighbour."""
    flat = lambda *a, **k: _m(0.0, 0.5)
    res = vs.sweep_best("-0.5", "0.1", 1e-3, 0.34, measure=flat)
    assert res["moved"] is False and res["chosen"]["dx"] == 0.0
    assert res["chosen"]["scale"] == 1.0 and res["n_windows"] == 18
    assert res["origin_composite"] == res["chosen_composite"]


def test_sweep_best_moves_to_the_argmax_and_records_every_window():
    calls = []

    def measure(cx, cy, fw, **k):
        calls.append((cx, cy, fw))
        # the +0.5/+0.5 window at scale 2 is the only good one
        return _m(0.0, 0.9 if len(calls) == 18 else 0.1)

    res = vs.sweep_best("-0.5", "0.1", 1e-3, 0.34, measure=measure)
    assert len(calls) == 18 and len(res["windows"]) == 18
    assert res["moved"] is True
    assert res["chosen"]["dx"] == 0.5 and res["chosen"]["dy"] == 0.5
    assert res["chosen"]["scale"] == 2.0
    assert res["chosen_composite"] > res["origin_composite"]


def test_the_sweep_records_the_chosen_window_beside_the_original_never_instead():
    res = vs.sweep_best("-0.5", "0.1", 1e-3, 0.34, measure=lambda *a, **k: _m(0.0, 0.5))
    assert res["origin_composite"] is not None
    assert any(w["is_origin"] for w in res["windows"])
    assert all({"dx", "dy", "scale", "composite", "band_coverage"} <= set(w)
               for w in res["windows"])


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_every_view_measure_carries_the_screen_cap_policy():
    """The same stamped policy the atom screen runs under, so a view score and an atom
    score are pairwise-non-poolable in exactly the way `require_one_policy` enforces."""
    import maneuver_screen as ms
    m = vs.measure_view("-0.5", "0.1", 0.0, threads=1)      # fw=0 -> the unscreenable path
    assert m["screened"] is False and m[fm.POLICY_KEY] == ms.screen_policy_token()
    assert m[fm.POLICY_KEY] != fm.LEGACY_POLICY_TOKEN
    with pytest.raises(fm.MaxiterPolicyMixError):
        fm.require_one_policy(("atom-era legacy", [{}]), ("view screen", [m]))


def test_the_resume_guard_is_reached_on_the_live_rescreen_path(tmp_path):
    """The live path that actually bites, same shape as `screen_pool.screen`'s: a resumed
    file written under another cap policy must raise BEFORE the pass is spent, not after."""
    import view_rescreen as vr
    out = tmp_path / "scores.jsonl"
    out.write_text(json.dumps({"key": "old", "band_coverage_q25": 0.5}) + "\n",
                   encoding="utf-8")
    logs = []
    with pytest.raises(fm.MaxiterPolicyMixError) as ei:
        vr.rescreen([], out, log=logs.append)
    assert logs, "the guard must fire on the real path, after the resume log line"
    assert "legacy" in str(ei.value)


# --------------------------------------------------------------------------- #
# the recorded validation gate (the outcome, including what failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not GATE.exists(), reason="gate not run")
def test_the_composite_ranks_both_references_into_the_top_quintile():
    g = json.loads(GATE.read_text(encoding="utf-8"))["formulations"][-1]
    for k, v in g["references"].items():
        assert v["percentile"] >= 80.0, (k, v)
    assert g["passed"] is True


@pytest.mark.skipif(not GATE.exists(), reason="gate not run")
def test_the_composite_sorts_every_named_bad_out_of_the_top_quintile():
    g = json.loads(GATE.read_text(encoding="utf-8"))["formulations"][-1]
    assert len(g["bads"]) == 4
    for k, v in g["bads"].items():
        assert v["percentile"] < 80.0, (k, v)
        assert v["old_quintile"] == 5, (k, v)     # all four came from the OLD Q5


@pytest.mark.skipif(not GATE.exists(), reason="gate not run")
def test_the_eye_still_outranks_mb19_so_the_composite_is_not_depth_in_disguise():
    """`minibroteye` is shallow and not even a nucleus; mb19 is at 8e-10. The standing
    test carried forward from `test_orbital.py` — a composite that ranked the eye low
    would be depth wearing a disguise."""
    g = json.loads(GATE.read_text(encoding="utf-8"))["formulations"][-1]
    assert (g["references"]["minibroteye"]["percentile"]
            >= g["references"]["mb19_p35_16x"]["percentile"])


@pytest.mark.skipif(not GATE.exists(), reason="gate not run")
def test_both_unshipped_formulations_stay_recorded_with_the_half_they_failed():
    """The iteration history is the validation record, not a draft of it. Each unshipped
    formulation is pinned to the SPECIFIC half it lost on, so "we tried the simple one" is
    a checkable claim (`test_orbital.py::test_the_measures_that_failed_are_recorded_as_failed`).

    f1 (tile mean) passes the stated gate and was still not shipped — it leaves the
    `snap k16 d2 p18` blob high in the population, which is a reason a boolean cannot
    carry, so the percentile is asserted directly.
    f2 (pooled quantile) fails G1 outright.
    """
    forms = {f["name"]: f for f in
             json.loads(GATE.read_text(encoding="utf-8"))["formulations"]}
    assert len(forms) == 3, "the iteration history must survive in the record"

    f1 = forms["f1_tile_mean_coverage"]
    assert f1["bads"]["q4_snap_095"]["percentile"] > 55.0, \
        "f1's defect was that it left the blob mid-population — if that moved, re-read this"

    f2 = forms["f2_block_q25_coverage"]
    assert f2["G1_refs_in_top_quintile"] is False and f2["passed"] is False
    assert f2["references"]["mb19_p35_16x"]["percentile"] < 80.0

    shipped = forms["f3_geometric_mean_SHIPPED"]
    assert shipped["passed"] is True
    # ...and shipping it was not free: it beats f1 on the blob by a wide margin.
    assert (shipped["bads"]["q4_snap_095"]["percentile"]
            < f1["bads"]["q4_snap_095"]["percentile"] - 10.0)


@pytest.mark.skipif(not GATE.exists(), reason="gate not run")
def test_no_formulation_reached_the_decile_bar_that_was_written_down_first():
    """The bar that was moved, pinned so the move stays visible. If a later formulation
    DOES reach the decile, this goes red and the bar gets re-decided on purpose rather
    than the quintile quietly outliving its reason."""
    g = json.loads(GATE.read_text(encoding="utf-8"))
    assert not any(f["refs_in_top_decile"] for f in g["formulations"])
    assert g["bar_percentile"] == 80.0


# --------------------------------------------------------------------------- #
# the drivers' pure halves — selection, stratification, the old-vs-new table
# --------------------------------------------------------------------------- #
def _score_row(i, cov, rng, rings=None, interior=0.0, op="snap_to_nucleus", degree=2):
    return dict(key=f"k{i}", screened=True, band_coverage=cov, band_coverage_q25=cov,
                radial_range=rng, radial_rings=(rng * 2 if rings is None else rings),
                interior_fraction=interior, op=op, degree=degree, period=10 + i,
                k=None, cx="-0.5", cy="0.1", fw=1e-3, partition="mandelbrot",
                atom_radial_range=float(i), atom_radial_rings=2.0 * i,
                atom_interior_fraction=0.1)


def test_the_sweep_sample_reaches_the_bottom_quintile_not_only_the_winners():
    """A sweep measured only where the composite already likes the view answers 'does
    framing help the winners'. The stratified fill is what makes it answer the question
    asked, so the bottom quintile is in BY CONSTRUCTION and is asserted, not hoped for."""
    import view_frame_sweep as vfs
    rows = [_score_row(i, cov=0.5, rng=float(i) / 10.0) for i in range(1, 201)]
    sel = vfs.select(rows, veto=0.9, top=10, n=60, seed=1)
    comps = sorted(r["_c"] for r in rows)
    lo = comps[len(comps) // 5]
    assert len(sel) >= 50
    assert any(r["_c"] <= lo for r in sel), "no bottom-quintile row was swept"
    assert sum(1 for r in sel if r["_c"] >= comps[-10]) >= 10, "the top-N was not forced in"
    assert len({r["key"] for r in sel}) == len(sel), "a row was selected twice"


def test_the_sweep_sample_never_includes_a_vetoed_row():
    import view_frame_sweep as vfs
    rows = ([_score_row(i, 0.5, 5.0, interior=0.9) for i in range(1, 40)]
            + [_score_row(100 + i, 0.5, 5.0, interior=0.0) for i in range(1, 40)])
    sel = vfs.select(rows, veto=0.34, top=5, n=50, seed=1)
    assert sel and all(r["interior_fraction"] == 0.0 for r in sel)


def test_a_quintile_sheet_cannot_become_one_operators_showreel():
    """Floor-then-remainder over (operator x degree): every cell present gets a tile before
    any cell gets a second, so a 90%-dominant operator cannot take the whole sheet."""
    import view_screen_sheets as vss
    pool = ([_score_row(i, 0.5, 5.0, op="neighborhood_expand", degree=2)
             for i in range(90)]
            + [_score_row(500 + i, 0.5, 5.0, op="lateral_to_sibling", degree=5)
               for i in range(5)])
    out = vss.stratify(pool, 12, seed=3)
    ops = {r["op"] for r in out}
    assert len(out) == 12 and ops == {"neighborhood_expand", "lateral_to_sibling"}
    assert sum(1 for r in out if r["op"] == "lateral_to_sibling") >= 5, \
        "the rare operator was truncated below its availability"


def test_stratify_does_not_invent_rows_when_the_pool_is_thin():
    import view_screen_sheets as vss
    out = vss.stratify([_score_row(i, 0.5, 5.0) for i in range(3)], 18, seed=1)
    assert len(out) == 3


def test_the_old_vs_new_table_counts_survivors_off_the_real_transition_matrix():
    """The table's two headline numbers (old-Q5 survival, new-Q5 newcomers) must be
    derivable from the transition matrix it prints beside them, or one of the two is
    describing a different population than the other."""
    import view_screen_sheets as vss
    rows = []
    for i in range(1, 101):
        r = _score_row(i, cov=0.5, rng=float(101 - i) / 10.0)   # new sort REVERSES old
        rows.append(r)
    veto = 0.9
    for r in rows:
        r["_comp"] = vs.composite(r, veto)
        r["_vetoed"] = vs.is_vetoed(r, veto)
    nq, _ = vss.quintile_index([r["_comp"] for r in rows])
    oq, _ = vss.quintile_index([r["atom_radial_range"] for r in rows])
    for r, a, b in zip(rows, nq, oq):
        r["new_quintile"], r["old_quintile"] = a, b
    rep = vss.agreement(rows)
    mat = rep["quintile_transition"]
    assert sum(sum(row) for row in mat) == len(rows)
    assert rep["old_Q5_n"] == sum(mat[4])
    assert rep["old_Q5_surviving_new_Q5"] == mat[4][4]
    assert rep["new_Q5_that_were_old_Q1_or_Q2"] == mat[0][4] + mat[1][4]
    # a perfectly reversed sort: nothing survives, and rho is strongly negative
    assert rep["old_Q5_surviving_new_Q5"] == 0
    assert rep["spearman_new_vs_old_atom_range"] < -0.9


def test_the_survivor_count_is_the_top_quintile_and_not_the_top_two():
    """A PARTIAL-agreement fixture, because the reversed one above cannot tell `== 5`
    from `>= 4` (both are zero there) — the vacuity `verification_practice.md` §6 names.
    Here five old-Q5 rows land in new-Q4 exactly, so the two readings differ by five."""
    import view_screen_sheets as vss
    rows = [_score_row(i, cov=0.5, rng=1.0) for i in range(1, 101)]
    for i, r in enumerate(rows, start=1):
        r["atom_radial_range"] = float(i)
        # identity, except the five rows at old ranks 81..85 drop into new-Q4
        r["_comp"] = float(i - 10 if 81 <= i <= 85 else i)
        r["_vetoed"] = False
    nq, _ = vss.quintile_index([r["_comp"] for r in rows])
    oq, _ = vss.quintile_index([r["atom_radial_range"] for r in rows])
    for r, a, b in zip(rows, nq, oq):
        r["new_quintile"], r["old_quintile"] = a, b
    rep = vss.agreement(rows)
    mat = rep["quintile_transition"]
    assert mat[4][4] == 15 and mat[4][3] == 5, mat[4]
    assert rep["old_Q5_surviving_new_Q5"] == 15
    assert rep["old_Q5_surviving_frac"] == pytest.approx(0.75)


def test_the_readout_carries_both_confound_caveats():
    """Prose guards rot, so this anchors on the ROUTING decision — that the degree and
    rank caveats are emitted with the numbers rather than living in a report nobody
    reads beside the JSON (`verification_practice.md` §6)."""
    import view_screen_sheets as vss
    rows = [_score_row(i, 0.5, float(i)) for i in range(1, 21)]
    for r in rows:
        r["_comp"], r["_vetoed"] = vs.composite(r, 0.9), False
        r["new_quintile"] = r["old_quintile"] = 3
    rep = vss.agreement(rows)
    assert "period" in rep["DEGREE_CAVEAT"] and "degree result" in rep["DEGREE_CAVEAT"]
    assert "4x frame" in rep["RANK_CAVEAT"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
