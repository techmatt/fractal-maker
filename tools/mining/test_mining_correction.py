"""The mode x score mining correction sheet: the universe, the draw, and the fitted cuts.

The universe is checked against the TRACKED artifacts (the gate-passer population and sheet
v1's manifest) because "nothing Matt has already judged is served twice" is a claim about
those files and nothing else. The draw is checked on synthetic screen records, because the
cases that matter — a take-all high band larger than the cap, a mode with no supply, the
declared fancy oversample — do not exist in any one real screen log.

  uv run pytest tools/mining/test_mining_correction.py -q
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring", ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import batch_registry as br                                  # noqa: E402
from tools.mining import build_mining_correction as BC       # noqa: E402
from tools.mining import build_mining_sheet as BMS           # noqa: E402
from tools.mining import mining_roster as MR                 # noqa: E402
from tools.mining import suggest_tier_mining as ST           # noqa: E402

SPEC = BC.SHEETS["v2"]


def rec(i, mode, pred):
    return {"unit_key": f"{mode}|loc{i:04d}|0", "mode": mode, "kind": MR.MODE_KIND[mode],
            "location_key": f"loc{i:04d}", "family": "mandelbrot", "variant": 0,
            "palette": "p", "screen_pred": pred, "screen_p_ge2": 0.5, "screen_p_ge3": 0.5,
            "screen_would_pass_gate": False}


def population(per_mode=120, hi_frac=0.10):
    """Synthetic screen records spanning all three bands in every mode."""
    out, i = [], 0
    hi, mid = ST.CUTS[1] + 0.05, ST.CUTS[0] + 0.05
    for mode in MR.MODES:
        for j in range(per_mode):
            pred = hi if j < int(per_mode * hi_frac) else (mid if j % 2 else ST.CUTS[0] - 0.1)
            out.append(rec(i, mode, pred))
            i += 1
    return out


# =========================================================================== #
# registration + cuts
# =========================================================================== #
def test_the_sheet_is_registered_train_side_before_it_is_built():
    assert br.is_registered(SPEC.batch_id)
    split, biased, source = br.assign_split(SPEC.batch_id, "mandelbrot")
    assert (split, biased, source) == ("train", True, "mining_correction_sitting")


@pytest.mark.stage2_pinned
def test_the_frozen_cuts_reproduce_from_the_slice_they_were_fitted_on():
    assert ST.derive_cuts() == ST.CUTS


@pytest.mark.stage2_pinned
def test_the_fitted_cuts_replace_the_per_batch_quantile_fallback_and_differ_from_it():
    """The fallback is kept as the record of what was done while no labeled slice existed; it
    must not silently be what a new sheet uses. Non-vacuous: on the fit slice's own preds the
    two rules give different cutpoints."""
    pred, tiers = ST.fit_slice()
    prior_cuts = ST.cuts_from_prior(pred, ST.tier_prior())
    assert tuple(round(c, 4) for c in prior_cuts) != ST.CUTS
    fitted = Counter(ST.tier_from_pred(p, ST.CUTS) for p in pred)
    true = Counter(tiers)
    for t in range(1, ST.K_TIERS + 1):
        assert abs(fitted[t] - true[t]) <= 2, (t, fitted, true)


def test_the_score_bin_is_the_heads_own_suggested_tier():
    """The bin and the suggestion cannot drift apart — "the high band" IS "the tier the head
    would suggest", by construction rather than by two constants agreeing."""
    for pred, want in ((ST.CUTS[1] + 0.1, "hi"), (ST.CUTS[0] + 0.1, "mid"),
                       (ST.CUTS[0] - 0.1, "lo")):
        assert BC.score_bin(pred) == want
        assert {"hi": 3, "mid": 2, "lo": 1}[want] == ST.tier_from_pred(pred, ST.CUTS)


# =========================================================================== #
# universe
# =========================================================================== #
@pytest.fixture(scope="module")
def uni():
    return BC.universe(SPEC)


def test_nothing_sheet_v1_served_is_in_the_universe(uni):
    entries, rep = uni
    sv = BC.served_pairs()
    assert rep["exclusion"]["served_location_mode_pairs"] == len(sv) == 960
    assert not [e for e in entries if (e["location_key"], e["mode"]) in sv]


def test_every_mode_has_supply_and_the_universe_is_deterministic(uni):
    entries, rep = uni
    assert set(rep["universe_by_mode"]) == set(MR.MODES)
    assert all(v > 0 for v in rep["universe_by_mode"].values())
    again, _ = BC.universe(SPEC)
    assert [e["unit_key"] for e in again] == [e["unit_key"] for e in entries]
    assert len({e["unit_key"] for e in entries}) == len(entries)


def test_direct_modes_spread_the_grid_and_the_others_spread_palettes(uni):
    """Sheet v1's rule, unchanged: `direct_*` is palette-INDIFFERENT, so its axis is the
    9-cell opacity x threshold grid on ONE palette per location."""
    entries, _rep = uni
    for mode in MR.DIRECT_MODES:
        rows = [e for e in entries if e["mode"] == mode]
        assert len({e["palette"] for e in rows if e["location_key"] == rows[0]["location_key"]}) == 1
        cells = {(e["mode_params"]["direct_opacity"], e["mode_params"]["direct_threshold"])
                 for e in rows}
        assert cells == set(MR.DIRECT_GRID)
    pure = [e for e in entries if e["kind"] != "direct"]
    assert all(not e["mode_params"] for e in pure)


def test_an_absent_prior_sheet_is_a_hard_stop_not_an_empty_exclusion(monkeypatch):
    with pytest.raises(SystemExit):
        BC.served_pairs("no_such_batch")


# =========================================================================== #
# the draw
# =========================================================================== #
@pytest.fixture(scope="module")
def drawn():
    pop = population()
    return BC.select(SPEC, pop) + (pop,)


def test_the_high_band_is_taken_whole(drawn):
    """The aimed slice. A false positive at the top of a scale is rare by construction, so
    anything short of take-all shows Matt the same handful the last sheet did."""
    sel, rep, pop = drawn
    hi_pop = [r for r in pop if BC.score_bin(r["screen_pred"]) == "hi"]
    hi_sel = [r for r in sel if r["bin"] == "hi"]
    assert len(hi_sel) == len(hi_pop) > 0
    assert rep["hi_band_retention"]["drawn"] == rep["hi_band_retention"]["screened"]


def test_the_fancy_high_cell_claims_before_the_pure_one(drawn):
    sel, _rep, _pop = drawn
    fancy = [r for r in sel if r["bucket"] == "hi_fancy"]
    pure = [r for r in sel if r["bucket"] == "hi_pure"]
    assert {r["kind"] for r in fancy} <= BC.FANCY_KINDS
    assert {r["kind"] for r in pure} == {"pure"}


def test_every_mode_reaches_the_floor_where_supply_allows(drawn):
    sel, rep, pop = drawn
    got = Counter(r["mode"] for r in sel)
    supply = Counter(r["mode"] for r in pop)
    for mode in MR.MODES:
        assert got[mode] >= min(SPEC.mode_floor, supply[mode]), (mode, got[mode])
    assert not rep["modes_below_floor"]


def test_the_fancy_oversample_is_where_the_prompt_put_it_the_MID_band(drawn):
    """WHERE the oversample can act, and where it cannot.

    In the HIGH band it CANNOT: that band is take-all for both kinds, so the drawn fancy share
    equals the supply's by construction and there is nothing more to give. What the sheet does
    there is take every fancy high row there is (4.3x what sheet v1 carried) and keep the pure
    high rows as the CONTROL — without them "fancy top-tier rows are corrected more often than
    pure ones" is not a measurable sentence.

    In the MID band it can and does: `mid_fancy` claims a declared share of the post-floor
    remainder, so the drawn fancy share must EXCEED the mid band's own supply share.

    The OVERALL fancy share is deliberately NOT asserted. The mode floor and the balanced fill
    both spread over modes, and the six pure modes have far less supply per mode than the nine
    fancy ones (`direct_*` alone spreads a 9-cell grid), so a correct draw comes out BELOW the
    population's fancy share overall — 0.663 against 0.721 on the live screen. Asserting the
    overall share would go red on a draw doing exactly what it was asked to do."""
    # A population where the PURE modes dominate the mid band, so a draw that merely balanced
    # over modes would come out fancy-POOR there. The equal-supply fixture cannot show this:
    # at 9 fancy modes of 15 its fancy share is already 0.60, which is exactly what the
    # declared mid share happens to be, and the test would pass on an equality.
    pop, i = [], 0
    hi, mid, lo = ST.CUTS[1] + 0.05, ST.CUTS[0] + 0.05, ST.CUTS[0] - 0.1
    for mode in MR.MODES:
        n_mid = 60 if MR.MODE_KIND[mode] in BC.FANCY_KINDS else 300
        for j in range(n_mid):
            pop.append(rec(i, mode, mid))
            i += 1
        for j in range(5):
            pop.append(rec(i, mode, hi))
            i += 1
        for j in range(20):
            pop.append(rec(i, mode, lo))
            i += 1
    sel, rep = BC.select(SPEC, pop)
    for r in pop:
        r["bin"] = BC.score_bin(r["screen_pred"])

    def fancy_share(rows):
        return sum(1 for r in rows if r["kind"] in BC.FANCY_KINDS) / max(len(rows), 1)

    mid_pop = [r for r in pop if r["bin"] == "mid"]
    mid_sel = [r for r in sel if r["bin"] == "mid"]
    assert fancy_share(mid_pop) < 0.25                      # the draw starts fancy-poor...
    assert mid_sel and fancy_share(mid_sel) > 0.5           # ...and the bucket pulls it up
    hi_pop = [r for r in pop if r["bin"] == "hi"]
    hi_sel = [r for r in sel if r["bin"] == "hi"]
    assert len(hi_sel) == len(hi_pop)
    assert fancy_share(hi_sel) == pytest.approx(fancy_share(hi_pop)), \
        "the high band is take-all; a fancy share that differs from supply means it is not"
    assert rep["per_bucket"][3]["bucket"] == "mid_fancy"
    assert {r["kind"] for r in sel if r["bucket"] == "mid_fancy"} <= BC.FANCY_KINDS


def test_the_cap_binds_and_nothing_is_drawn_twice(drawn):
    sel, rep, _pop = drawn
    assert rep["drawn_rows"] == len(sel) <= SPEC.max_rows
    assert len({r["unit_key"] for r in sel}) == len(sel)
    assert all(r["bucket"] in BC.BUCKET_ORDER for r in sel)


def test_a_universe_smaller_than_the_cap_yields_a_short_sheet():
    pop = population(per_mode=20)
    sel, rep = BC.select(SPEC, pop)
    assert rep["drawn_rows"] == len(pop) < SPEC.max_rows


def test_a_mode_with_no_supply_is_absent_and_reported_below_floor():
    pop = [r for r in population(per_mode=60) if r["mode"] != "trap_circle"]
    sel, rep = BC.select(SPEC, pop)
    assert "trap_circle" not in rep["drawn_by_mode"]
    assert "trap_circle" not in rep["modes_below_floor"], \
        "a mode with ZERO supply is not 'below its floor' — it is absent, and the report " \
        "must not read as a shortfall the draw could have fixed"


def test_the_draw_is_a_pure_function_of_population_and_seed():
    pop = population()
    a, _ = BC.select(SPEC, [dict(r) for r in pop])
    b, _ = BC.select(SPEC, [dict(r) for r in pop])
    assert [r["unit_key"] for r in a] == [r["unit_key"] for r in b]
    c, _ = BC.select(SPEC, [dict(r) for r in pop], seed=SPEC.draw_seed + 1)
    assert [r["unit_key"] for r in c] != [r["unit_key"] for r in a]


# =========================================================================== #
# the shared render path
# =========================================================================== #
def test_the_screen_geometry_is_a_parameter_of_the_keeper_path_not_a_second_copy():
    """One render path, two geometries. If the screen grew its own renderer, a mode's recipe
    could differ between what was screened and what was served."""
    import inspect
    for fn in (BMS._render_pure, BMS._render_rust):
        assert "geom" in inspect.signature(fn).parameters
    src = Path(BC.__file__).read_text(encoding="utf-8")
    assert "BMS.render_one" in src
    assert "render-one" not in src, "the correction sheet must not invoke the engine itself"


def test_the_keeper_pins_are_sheet_v1s_and_the_screen_is_strictly_smaller():
    assert (BMS.W, BMS.H, BMS.SS) == (1280, 720, 2)
    w, h, ss = BC.SCREEN_GEOM
    assert w * ss < BMS.W * BMS.SS and h * ss < BMS.H * BMS.SS


def test_the_crop_stem_is_opaque_and_stable():
    a = BC._screen_stem(SPEC, "tia|loc0001|0")
    assert a == BC._screen_stem(SPEC, "tia|loc0001|0")
    assert a != BC._screen_stem(SPEC, "tia|loc0002|0")
    for token in ("tia", "loc0001", "direct", "composite"):
        assert token not in a


def test_the_worker_cap_is_the_project_process_cap():
    with pytest.raises(SystemExit):
        BC.main(["select", "--workers", str(BC.WORKERS + 1)])
