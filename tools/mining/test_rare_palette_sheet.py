"""Guards for sheet C — the rare-palette strange sheet.

Everything here runs on SYNTHETIC pools and SYNTHETIC screen records. The one thing that
must not be tested against a fixture is the label corpus itself, and it is not: what is
asserted is the DRAW's behaviour given a population, which is where the failures live
(a partition that silently reaches its human-3s, a classic slice that a bucketed cut skips
by construction, a near-dup filter that only looks across locations).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import location as loc_mod                                # noqa: E402
import partitions as PART                                 # noqa: E402
from tools.mining import build_rare_palette_sheet as C    # noqa: E402
from tools.mining import mining_roster as MR              # noqa: E402
from tools.mining import smooth_equivalence as SE         # noqa: E402
from tools.scoring import batch_registry as BR            # noqa: E402


# --------------------------------------------------------------------------- #
# The smooth baseline is nameable but off-roster.
# --------------------------------------------------------------------------- #
def test_smooth_is_not_on_the_roster_but_has_a_recipe():
    assert MR.SMOOTH_MODE not in MR.MODES
    assert MR.spec_for(MR.SMOOTH_MODE)["field"] == "smooth"
    assert MR.kind_of(MR.SMOOTH_MODE) == "pure"
    assert MR.missing_recipes() == []


def test_kind_of_still_refuses_an_unknown_mode():
    with pytest.raises(KeyError):
        MR.kind_of("not_a_mode")


def test_smooth_spec_matches_the_committed_registry_spec():
    import json
    reg = json.loads((ROOT / "specs" / "smooth.json").read_text(encoding="utf-8"))
    reg.pop("tier", None)
    assert MR.spec_for(MR.SMOOTH_MODE) == reg


# --------------------------------------------------------------------------- #
# Registration precedes the build.
# --------------------------------------------------------------------------- #
def test_sheet_c_is_registered_and_is_train_side():
    spec = C.SHEETS["c1"]
    assert BR.is_registered(spec.batch_id)
    reg = BR.lookup(spec.batch_id, "mandelbrot")
    assert reg.biased is True and reg.eval_eligible is False
    assert BR.split_of(reg) == "train"


# --------------------------------------------------------------------------- #
# The location draw.
# --------------------------------------------------------------------------- #
def _loc(fam, i):
    return loc_mod.Location(family=fam, cx=f"0.{i:04d}", cy="0.0", fw="0.01", maxiter=2000,
                            c_re=("0.1" if fam.startswith("julia") else None),
                            c_im=("0.2" if fam.startswith("julia") else None))


def _pool(counts):
    """counts: {partition: (n_fours, n_threes)}."""
    out, i = {}, 0
    for part, (n4, n3) in counts.items():
        fam = PART.FAM2FT[PART.base_partition(part)]
        for score, n in ((4, n4), (3, n3)):
            for _ in range(n):
                i += 1
                out[f"{part}|{i}"] = {"loc": _loc(fam, i), "score": score,
                                      "partition": part, "batch_ids": {"b"},
                                      "image_ids": ["x"]}
    return out


def _spec(**kw):
    base = C.SHEETS["c1"]
    from dataclasses import replace
    return replace(base, **kw)


def test_fours_are_exhausted_before_any_three_is_used():
    pool = _pool({"mandelbrot": (5, 50), "multibrot3": (50, 50)})
    spec = _spec(n_locations=20, location_caps={})
    drawn, rep = C.draw_locations(spec, pool)
    got = Counter((v["partition"], v["score"]) for _k, v in drawn)
    assert got[("mandelbrot", 4)] == 5        # all of them, then the fallback
    assert got[("mandelbrot", 3)] == rep["drawn_by_partition"]["mandelbrot"] - 5
    assert got[("multibrot3", 3)] == 0, "a partition with fours to spare used a three"


def test_classic_phoenix_is_preseeded_take_all_not_a_natural_share():
    # phoenix:classic supply is tiny; a plain balanced draw over 10 partitions would still
    # give it a share, so the property under test is that the reservation is a FLOOR.
    pool = _pool({"mandelbrot": (400, 0), "phoenix": (100, 0), PART.CLASSIC_PHOENIX: (1, 6)})
    spec = _spec(n_locations=60, location_caps={"phoenix": 5, PART.CLASSIC_PHOENIX: 7})
    _drawn, rep = C.draw_locations(spec, pool)
    assert rep["drawn_by_partition"][PART.CLASSIC_PHOENIX] == 7
    assert rep["drawn_by_partition"]["phoenix"] == 5           # the cost cap holds


def test_a_cap_bounds_a_partition_even_with_supply_to_spare():
    pool = _pool({"mandelbrot": (400, 0), "phoenix": (300, 0)})
    spec = _spec(n_locations=100, location_caps={"phoenix": 9})
    _d, rep = C.draw_locations(spec, pool)
    assert rep["drawn_by_partition"]["phoenix"] == 9
    assert rep["available_after_caps"]["phoenix"] == 9


def test_the_draw_is_deterministic_in_its_seed():
    pool = _pool({"mandelbrot": (40, 10), "multibrot3": (40, 10)})
    spec = _spec(n_locations=30, location_caps={})
    a = [k for k, _v in C.draw_locations(spec, pool)[0]]
    b = [k for k, _v in C.draw_locations(spec, pool)[0]]
    assert a == b


def test_a_pool_with_no_classic_phoenix_does_not_reserve_and_does_not_raise():
    # `deal_round_robin` refuses a preseed key absent from `sizes`; the reservation must
    # therefore be conditional on the cell existing, not on the spec naming it.
    pool = _pool({"mandelbrot": (40, 0), "multibrot3": (40, 0)})
    spec = _spec(n_locations=20)
    _d, rep = C.draw_locations(spec, pool)
    assert rep["reserved_take_all"] == {}
    assert PART.CLASSIC_PHOENIX not in rep["drawn_by_partition"]


def test_the_reservation_is_a_floor_not_a_bonus():
    # A take-all cell must not ALSO win rows from the balanced remainder — that is the
    # `max(natural, reserved)` vs `natural + reserved` distinction apportion documents.
    pool = _pool({"mandelbrot": (400, 0), PART.CLASSIC_PHOENIX: (7, 0)})
    spec = _spec(n_locations=40, location_caps={PART.CLASSIC_PHOENIX: 7})
    drawn, rep = C.draw_locations(spec, pool)
    assert rep["drawn_by_partition"][PART.CLASSIC_PHOENIX] == 7
    assert len(drawn) == 40
    assert len({k for k, _v in drawn}) == 40, "a reserved location was dealt twice"


# --------------------------------------------------------------------------- #
# spread_over — the bounded end-to-end must reach every render path.
# --------------------------------------------------------------------------- #
def test_spread_over_reaches_every_mode_and_partition():
    units = [{"mode": m, "partition": p, "i": i}
             for i in range(8) for p in ("a", "b") for m in MR.MODES]
    got = C.spread_over(units, 2 * len(MR.MODES))
    assert set(u["mode"] for u in got) == set(MR.MODES)
    assert set(u["partition"] for u in got) == {"a", "b"}


def test_spread_over_is_stride_independent_where_a_strided_sample_is_not():
    # 16 units per location, location-major. Any sampler whose stride shares a factor with
    # 16 walks a handful of modes for the whole run; at stride exactly 16 it walks ONE. The
    # cell deal cannot, whatever the layout.
    units = [{"mode": m, "partition": "p", "loc": i}
             for i in range(250) for m in (MR.SMOOTH_MODE,) + MR.MODES]
    strided = units[::16][:250]
    assert len({u["mode"] for u in strided}) == 1                # the failure, demonstrated
    assert len({u["mode"] for u in C.spread_over(units, 250)}) == 16


# --------------------------------------------------------------------------- #
# select — the two filters.
# --------------------------------------------------------------------------- #
def _screen(loc, mode, pred=2.0, **kw):
    return {"unit_key": f"{loc}|{mode}", "mode": mode, "kind": MR.kind_of(mode),
            "location_key": loc, "partition": "mandelbrot", "family": "mandelbrot",
            "palette": kw.get("palette", "p"), "hue_family": kw.get("hue_family", "green"),
            "human_score": 4, "mode_params": {}, "screen_pred": pred,
            "screen_p_ge2": 0.9, "screen_p_ge3": 0.5, "screen_would_pass_gate": True}


def _unit(dim=8, i=0, tilt=0.0):
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    v[(i + 1) % dim] = tilt
    return v / np.linalg.norm(v)


def test_a_smooth_equivalent_mode_is_excluded_not_downweighted():
    modes = ["tia", "stripe", "curv_linear"]
    recs = [_screen("L0", MR.SMOOTH_MODE, 2.5)] + [_screen("L0", m, 2.5) for m in modes]
    emb = {"L0|" + MR.SMOOTH_MODE: _unit(i=0)}
    emb["L0|tia"] = emb["L0|" + MR.SMOOTH_MODE].copy()          # identical -> cos 1.0
    emb["L0|stripe"] = _unit(i=2)
    emb["L0|curv_linear"] = _unit(i=4)
    spec = _spec(target_rows=10, strange_per_location=3, smooth_slice=0, mode_floor=0)
    sel, rep = C.select(spec, recs, emb)
    assert rep["smooth_equivalence"]["excluded_smooth_equivalent"] == 1
    assert "L0|tia" not in {r["unit_key"] for r in sel}


def test_an_unmeasured_row_is_dropped_rather_than_read_as_distinct():
    recs = [_screen("L0", "tia", 2.5)]           # no smooth twin screened at all
    spec = _spec(target_rows=10, strange_per_location=3, smooth_slice=0, mode_floor=0)
    sel, rep = C.select(spec, recs, {"L0|tia": _unit(i=1)})
    assert rep["smooth_equivalence"]["unmeasured_dropped"] == 1
    assert sel == []


def test_near_dup_filter_catches_two_modes_of_ONE_location():
    recs = [_screen("L0", MR.SMOOTH_MODE, 2.0),
            _screen("L0", "tia", 2.9), _screen("L0", "stripe", 2.8)]
    smooth = _unit(i=0)
    twin = _unit(i=3)
    emb = {"L0|" + MR.SMOOTH_MODE: smooth, "L0|tia": twin, "L0|stripe": twin.copy()}
    spec = _spec(target_rows=10, strange_per_location=2, smooth_slice=0, mode_floor=0)
    sel, rep = C.select(spec, recs, emb)
    assert rep["near_dup_filter"]["n_dropped"] == 1
    assert len([r for r in sel if r["mode"] != MR.SMOOTH_MODE]) == 1
    # the better row survives the collision
    assert {r["unit_key"] for r in sel} == {"L0|tia"}


def test_at_most_strange_per_location_rows_reach_the_page():
    modes = [m for m in MR.MODES][:6]
    recs = [_screen("L0", MR.SMOOTH_MODE, 2.0)]
    emb = {"L0|" + MR.SMOOTH_MODE: _unit(dim=32, i=0)}
    for j, m in enumerate(modes):
        recs.append(_screen("L0", m, 2.0 + j * 0.01))
        emb[f"L0|{m}"] = _unit(dim=32, i=j + 3)
    spec = _spec(target_rows=50, strange_per_location=2, smooth_slice=0, mode_floor=0)
    sel, _rep = C.select(spec, recs, emb)
    assert len(sel) == 2


def test_smooth_slice_only_covers_locations_that_kept_a_strange_row():
    recs, emb = [], {}
    for i in range(6):
        loc = f"L{i}"
        recs.append(_screen(loc, MR.SMOOTH_MODE, 2.0))
        emb[f"{loc}|{MR.SMOOTH_MODE}"] = _unit(dim=64, i=2 * i)
        recs.append(_screen(loc, "tia", 2.5))
        emb[f"{loc}|tia"] = _unit(dim=64, i=2 * i + 1)
    spec = _spec(target_rows=50, strange_per_location=1, smooth_slice=3, mode_floor=0)
    sel, rep = C.select(spec, recs, emb)
    smooth_locs = {r["location_key"] for r in sel if r["mode"] == MR.SMOOTH_MODE}
    strange_locs = {r["location_key"] for r in sel if r["mode"] != MR.SMOOTH_MODE}
    assert rep["buckets"].get("smooth_comparison") == 3
    assert smooth_locs <= strange_locs


def test_the_majority_of_a_built_sheet_is_un_smooth_by_construction():
    spec = C.SHEETS["c1"]
    assert spec.smooth_slice < spec.target_rows / 2


def test_selection_report_carries_the_yardstick_it_used():
    recs = [_screen("L0", MR.SMOOTH_MODE, 2.0), _screen("L0", "tia", 2.5)]
    emb = {"L0|" + MR.SMOOTH_MODE: _unit(i=0), "L0|tia": _unit(i=4)}
    spec = _spec(target_rows=10, strange_per_location=1, smooth_slice=0, mode_floor=0)
    _sel, rep = C.select(spec, recs, emb)
    assert rep["smooth_equivalence"]["strict_cut"] == SE.STRICT_CUT
    assert rep["near_dup_filter"]["cut"] == SE.STRICT_CUT


# --------------------------------------------------------------------------- #
# render block.
# --------------------------------------------------------------------------- #
def test_render_block_round_trips_through_the_canonical_location():
    for fam in ("mandelbrot", "julia_multibrot4", "phoenix"):
        loc = _loc(fam, 7)
        blk = C.render_block_of(loc, "viridis")
        back = loc_mod.from_render_block(blk)
        assert loc_mod.location_key(back) == loc_mod.location_key(loc)
