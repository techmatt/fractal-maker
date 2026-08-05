#!/usr/bin/env python
"""Unit tests for the atlas production seeder (control on the pure predicates + the
q3-density rejection rule + a ledger round-trip). The smoke eyeball is the visual gate.

  uv run pytest tools/atlas/test_production_seeder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import production_seeder as ps  # noqa: E402


# --------------------------------------------------------------------------- #
# fw-relative dedup predicate (cloud hygiene)
# --------------------------------------------------------------------------- #
def test_near_dup_within_and_outside():
    # B at origin, fw=1.0 -> dedup radius = 1.5*max(fw).
    assert ps.near_dup(1.4, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5) is True    # 1.4 < 1.5
    assert ps.near_dup(1.6, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5) is False   # 1.6 > 1.5


def test_near_dup_same_center_different_zoom_merges_only_well_inside_the_finer_frame():
    # (nearly) same center, very different fw. Under the calibrated min(fw) rule the FINER
    # frame sets the radius, so this merges only while the offset is small against 1e-3 —
    # the retired max(fw) rule merged the whole 2.0-wide disc, which is what 135 hand
    # verdicts rejected. Both sides asserted so the direction cannot silently flip back.
    assert ps.near_dup(1e-6, 0.0, 1e-3, 0.0, 0.0, 2.0, k=1.5) is True
    assert ps.near_dup(0.5, 0.0, 1e-3, 0.0, 0.0, 2.0, k=1.5) is False        # min-scale: distinct
    assert ps.near_dup(0.5, 0.0, 1e-3, 0.0, 0.0, 2.0, k=1.5, scale="max") is True   # retired rule


def test_near_dup_distant_pair_distinct():
    # genuinely distant centers at small fw -> distinct.
    assert ps.near_dup(5.0, 5.0, 1e-3, 0.0, 0.0, 1e-3, k=1.5) is False


def test_is_distinct_against_cloud():
    cloud = [
        {"id": "a", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 1.0},
        {"id": "b", "outcome_cx": 10.0, "outcome_cy": 0.0, "outcome_fw": 1e-3},
    ]
    d, dup = ps.is_distinct(0.5, 0.0, 1.0, cloud, k=1.5)   # within a's radius
    assert d is False and dup == "a"
    d, dup = ps.is_distinct(3.0, 3.0, 1e-3, cloud, k=1.5)  # far from both
    assert d is True and dup is None


# --------------------------------------------------------------------------- #
# THE PRODUCTION RULE PIN. K and the scale were calibrated TOGETHER against 135 hand
# verdicts (data/atlas/precanon_calibration/adoption.json, 2026-08-04); neither transfers
# to the other scale, so both are pinned, and every assertion below is red under a revert
# of EITHER one alone. The mode pair itself stays differential (the two can only differ on
# ASYMMETRIC pairs) rather than frozen against literals.
# --------------------------------------------------------------------------- #
def test_production_dedup_rule_is_the_calibrated_min_quarter():
    """The adopted pair, and the RESOLVED radius it produces at the admission path.

    Red under either revert, by construction:
      * scale -> "max"  : the asymmetric radius becomes 0.25*2.0 = 0.5, not 0.25*1e-3
      * K     -> 1.5    : the symmetric radius becomes 1.5*1.0, not 0.25*1.0
    A single resolved-radius assertion catches both, so this cannot pass with one of the
    two constants reverted."""
    assert (ps.DEDUP_K, ps.DEDUP_SCALE) == (0.25, "min")
    # resolved with NO explicit argument — what every production call site actually gets.
    assert ps.dedup_radius(ps.DEDUP_K, 1e-3, 2.0) == 0.25 * 1e-3     # scale revert -> 0.5
    assert ps.dedup_radius(ps.DEDUP_K, 1.0, 1.0) == 0.25 * 1.0       # K revert -> 1.5
    # the retired rule is kept NAMED (the record-replaying diagnostics read it) and is not
    # the default: a revert that flips the defaults back also breaks this pair's meaning.
    assert (ps.RETIRED_DEDUP_K, ps.RETIRED_DEDUP_SCALE) == (1.5, "max")
    assert (ps.DEDUP_K, ps.DEDUP_SCALE) != (ps.RETIRED_DEDUP_K, ps.RETIRED_DEDUP_SCALE)


def test_production_dedup_verdicts_move_under_either_revert():
    """The pin at the level that matters: the VERDICT `is_distinct` returns with no explicit
    k/scale. Two fixtures, each isolating one constant — an asymmetric pair only the scale
    moves, and a symmetric pair only K moves (min == max there, so the scale cannot touch
    it). Both are what the calibration bought: a deep zoom inside a wide outcome survives,
    and a same-scale neighbour at 0.5*fw is no longer swallowed."""
    wide = [{"id": "wide", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 2.0}]
    # asymmetric: d=0.4 sits inside 0.25*max(fw)=0.5 but outside 0.25*min(fw)=2.5e-4.
    assert ps.is_distinct(0.4, 0.0, 1e-3, wide) == (True, None)
    assert ps.is_distinct(0.4, 0.0, 1e-3, wide, scale="max") == (False, "wide")
    same = [{"id": "same", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 1.0}]
    # symmetric: d=0.5 is outside 0.25*fw and inside 1.5*fw, and min == max at equal fw.
    assert ps.is_distinct(0.5, 0.0, 1.0, same) == (True, None)
    assert ps.is_distinct(0.5, 0.0, 1.0, same, k=ps.RETIRED_DEDUP_K) == (False, "same")
    assert ps.is_distinct(0.5, 0.0, 1.0, same, scale="max") == (True, None)   # K, not the scale
    # and the rule still fires on what it is FOR: a genuine same-place revisit.
    assert ps.is_distinct(0.5 * ps.DEDUP_K, 0.0, 1.0, same) == (False, "same")


def test_admission_call_sites_resolve_the_live_rule():
    """A source scan over the two admission modules: no `is_distinct`/`near_dup`/`build_cloud`
    call there may pass an explicit `scale=` or a literal `k`, because such a call is exactly
    how production silently keeps running the retired rule after this pin goes green. The
    record-replaying diagnostics (fate sheet, calibration sheet, minfw replay/sheet,
    campaign1_readout) are deliberately NOT scanned — pinning the retired rule is their job.

    Derived + proved non-empty: the scan asserts it actually found call sites."""
    import re
    calls = 0
    for name in ("production_seeder.py", "steered_frontier.py"):
        src = (HERE / name).read_text(encoding="utf-8")
        for m in re.finditer(r"(?<![\w.])(?:ps\.)?(is_distinct|near_dup|build_cloud)\(", src):
            # take the call's argument text up to the matching close paren.
            i, depth = m.end(), 1
            while i < len(src) and depth:
                depth += (src[i] == "(") - (src[i] == ")")
                i += 1
            args = src[m.end():i - 1]
            if "scale=scale" in args or args.lstrip().startswith(("rows", "a_cx")):
                continue        # the definitions and their own forwarding plumbing
            calls += 1
            for pin in ('scale="', "scale='", "RETIRED_DEDUP"):
                assert pin not in args, \
                    f"{name}: {m.group(1)} pins the rule ({pin}): {args[:120]!r}"
            assert not re.search(r"\bk\s*=\s*[\d.]", args), \
                f"{name}: {m.group(1)} pins a literal k: {args[:120]!r}"
    assert calls >= 8, f"scan found only {calls} admission call sites — it has gone vacuous"


def test_dedup_scale_rejects_an_unknown_mode():
    """Fail-closed: a typo'd mode raises rather than silently taking a branch."""
    import pytest
    with pytest.raises(ValueError, match="max.*min"):
        ps.dedup_radius(1.5, 1.0, 1.0, scale="maximum")


def test_symmetric_scale_pairs_decide_identically_under_both_modes():
    """fw_a == fw_b => min == max, so EVERY clause of the predicate agrees. Swept across
    scales, distances and both identity kinds so this cannot pass on one lucky pair; the
    non-vacuity assertion is that the sweep contains both verdicts."""
    jc = (0.3, -0.1)
    ph = (0.3, -0.1, 0.5, 0.2, 0.0, 0.0)
    verdicts = set()
    for fw in (2.0, 1.0, 1e-3, 1e-9):
        for frac in (0.0, 0.5, 0.999, 1.0, 1.4999, 1.5, 2.0, 7.0):
            d = frac * fw
            for a_c, b_c in ((None, None), (jc, jc), (ph, ph)):
                mx = ps.near_dup(d, 0.0, fw, 0.0, 0.0, fw, k=1.5, a_c=a_c, b_c=b_c, scale="max")
                mn = ps.near_dup(d, 0.0, fw, 0.0, 0.0, fw, k=1.5, a_c=a_c, b_c=b_c, scale="min")
                assert mx is mn, f"symmetric pair disagreed at fw={fw} d={d} ident={a_c}"
                verdicts.add(mx)
    assert verdicts == {True, False}       # the sweep actually straddles the cut


def test_min_scale_is_strictly_weaker_than_max_on_asymmetric_pairs():
    """min(fw) can only ever REMOVE collisions (radius shrinks), never add one — and on the
    run's own shape (a deep zoom inside a wide outcome) it does remove them."""
    cloud = [{"id": "wide", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 2.0}]
    # a deep zoom 0.5 from a wide outcome's centre: inside 1.5*2.0, outside 1.5*1e-3.
    assert ps.is_distinct(0.5, 0.0, 1e-3, cloud, k=1.5, scale="max") == (False, "wide")
    assert ps.is_distinct(0.5, 0.0, 1e-3, cloud, k=1.5, scale="min") == (True, None)
    # implication direction, swept: min fires => max fires. Never the reverse.
    for a_fw, b_fw in ((2.0, 1e-3), (1e-3, 2.0), (1.0, 0.4), (0.4, 1.0)):
        for d in (0.0, 1e-4, 0.3, 0.6, 1.2, 1.6, 3.1):
            if ps.near_dup(d, 0.0, a_fw, 0.0, 0.0, b_fw, k=1.5, scale="min"):
                assert ps.near_dup(d, 0.0, a_fw, 0.0, 0.0, b_fw, k=1.5, scale="max")


def test_mode_never_overrides_the_identity_clauses():
    """A mode flip may only change the verdict on a pair that already passed identity —
    distinct-c julias and keyed-vs-c-plane stay non-collidable under both."""
    for scale in ("max", "min"):
        assert ps.near_dup(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, k=1.5,
                           a_c=(0.3, -0.1), b_c=(-0.7, 0.2), scale=scale) is False
        assert ps.near_dup(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, k=1.5,
                           a_c=(0.3, -0.1), b_c=None, scale=scale) is False


def test_build_cloud_forwards_the_scale():
    """The cloud builder dedups with the rule it is handed — a replay that rebuilds a prior
    cloud under the RETIRED "max" must not silently get the live "min", or vice versa."""
    rows = [{"id": "wide", "family": "mandelbrot", "decoded_class": 3,
             "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 2.0},
            {"id": "deep", "family": "mandelbrot", "decoded_class": 3,
             "outcome_cx": 0.5, "outcome_cy": 0.0, "outcome_fw": 1e-3}]
    # live default (min): the deep zoom inside the wide outcome is its own place.
    assert [r["id"] for r in ps.build_cloud(rows, "mandelbrot")] == ["wide", "deep"]
    assert [r["id"] for r in ps.build_cloud(rows, "mandelbrot", scale="min")] == ["wide", "deep"]
    # the retired rule swallowed it — and needs BOTH its constants to reproduce that.
    assert [r["id"] for r in ps.build_cloud(rows, "mandelbrot", k=ps.RETIRED_DEDUP_K,
                                            scale=ps.RETIRED_DEDUP_SCALE)] == ["wide"]


# --------------------------------------------------------------------------- #
# seed-c-aware dup key (the julia over-kill fix). A julia row's dup identity keys on
# BOTH its z-viewport AND its seed c; see docs/design/morphology_dedup.md §5.
# --------------------------------------------------------------------------- #
def test_distinct_c_julias_at_same_view_do_not_collide():
    # (a) two DISTINCT-c julia views at the IDENTICAL shared root z-viewport are distinct
    # (the campaign-1 over-kill was exactly this collision under z-only keying).
    jc_a, jc_b = (0.30, -0.10), (-0.85, 0.20)
    assert ps.near_dup(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, k=1.5, a_c=jc_a, b_c=jc_b) is False
    cloud = [{"id": "ja", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 3.0,
              "julia_c_re": jc_a[0], "julia_c_im": jc_a[1]}]
    d, dup = ps.is_distinct(0.0, 0.0, 3.0, cloud, c=jc_b)
    assert d is True and dup is None


def test_same_c_near_identical_views_collide():
    # (b) same seed c (within eps) + near z-viewport -> genuine dup.
    jc = (0.30, -0.10)
    assert ps.near_dup(1.4, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5, a_c=jc, b_c=jc) is True   # 1.4 < 1.5
    assert ps.near_dup(1.6, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5, a_c=jc, b_c=jc) is False  # z too far
    cloud = [{"id": "ja", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 1.0,
              "julia_c_re": jc[0], "julia_c_im": jc[1]}]
    # distance stated RELATIVE to the live radius: this test is about the identity clause,
    # so it must stay inside the disc whatever (DEDUP_K, DEDUP_SCALE) is.
    d, dup = ps.is_distinct(0.5 * ps.DEDUP_K, 0.0, 1.0, cloud, c=(jc[0] + 1e-9, jc[1]))
    assert d is False and dup == "ja"


def test_julia_never_collides_with_cplane_row():
    # (c) a julia row (has seed c) never collides with a base-family c-plane row (no c),
    # even at the identical viewport.
    jc = (0.30, -0.10)
    assert ps.near_dup(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, k=1.5, a_c=jc, b_c=None) is False
    assert ps.near_dup(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, k=1.5, a_c=None, b_c=jc) is False
    cplane_cloud = [{"id": "m", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 3.0}]
    d, dup = ps.is_distinct(0.0, 0.0, 3.0, cplane_cloud, c=jc)   # julia candidate vs c-plane cloud
    assert d is True and dup is None


def test_cplane_pair_unchanged_when_no_seed_c():
    # regression: with no seed c on either side, the metric is byte-identical to the old z-only.
    assert ps.near_dup(1.4, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5) is True
    assert ps.near_dup(1.6, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5) is False


def test_build_cloud_keeps_distinct_c_julias_as_separate_places():
    # within a julia partition, two distinct-c julias at the same viewport are TWO cloud
    # places (z-only dedup collapsed them to one — the cloud under-count half of the bug).
    rows = [
        {"id": "ja", "family": "julia:multibrot3", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 3.0,
         "julia_c_re": 0.30, "julia_c_im": -0.10},
        {"id": "jb", "family": "julia:multibrot3", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 3.0,
         "julia_c_re": -0.85, "julia_c_im": 0.20},
        # a genuine same-c revisit of ja collapses.
        {"id": "ja2", "family": "julia:multibrot3", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.01, "outcome_cy": 0.0, "outcome_fw": 3.0,
         "julia_c_re": 0.30, "julia_c_im": -0.10},
    ]
    cloud = ps.build_cloud(rows, "julia:multibrot3")
    assert {m["id"] for m in cloud} == {"ja", "jb"}       # ja2 deduped; ja/jb both kept


# --------------------------------------------------------------------------- #
# Phoenix parameter-point identity (the julia seed-c fix lifted to (c, p, z_{-1})). A
# phoenix row's dup identity keys on its z-viewport AND its full parameter point; absent
# axes resolve to the legacy Ushiki values. See docs/design/phoenix_seed_sampler_spec.md §3.
# --------------------------------------------------------------------------- #
def _ph_row(oid, cx, cy, fw, c=(0.5667, 0.0), p=(-0.5, 0.0), z=(0.0, 0.0)):
    return {"id": oid, "family": "phoenix", "guard_pass": True, "decoded_class": 3,
            "outcome_cx": cx, "outcome_cy": cy, "outcome_fw": fw,
            **ps.phoenix_ident_fields(c, p, z)}


def test_distinct_phoenix_params_do_not_collide():
    # distinct parameter points at the IDENTICAL z-viewport are distinct places (opening
    # the axes without keying would recreate the julia over-kill with three axes).
    ka = ps.row_phoenix_key(_ph_row("a", 0, 0, 3.0, p=(-0.5, 0.0), z=(0.0, 0.0)))
    kb = ps.row_phoenix_key(_ph_row("b", 0, 0, 3.0, p=(-0.4, 0.0), z=(0.0, 0.0)))  # differs in p
    kz = ps.row_phoenix_key(_ph_row("z", 0, 0, 3.0, p=(-0.5, 0.0), z=(0.1, 0.0)))  # differs in z_{-1}
    assert ps.near_dup(0, 0, 3.0, 0, 0, 3.0, k=1.5, a_c=ka, b_c=kb) is False
    assert ps.near_dup(0, 0, 3.0, 0, 0, 3.0, k=1.5, a_c=ka, b_c=kz) is False        # z_{-1} keys too
    cloud = [_ph_row("pa", 0, 0, 3.0, p=(-0.5, 0.0))]
    d, dup = ps.is_distinct(0, 0, 3.0, cloud, c=kb)
    assert d is True and dup is None


def test_same_phoenix_params_near_views_collide():
    # identical parameter point + near z-viewport -> genuine dup (dedups as before).
    k = ps.row_phoenix_key(_ph_row("a", 0, 0, 1.0))
    assert ps.near_dup(1.4, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5, a_c=k, b_c=k) is True   # 1.4 < 1.5
    assert ps.near_dup(1.6, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5, a_c=k, b_c=k) is False  # z too far
    cloud = [_ph_row("pa", 0, 0, 1.0)]
    d, dup = ps.is_distinct(0.5 * ps.DEDUP_K, 0.0, 1.0, cloud,      # inside the live radius
                            c=ps.row_phoenix_key(_ph_row("q", 0, 0, 1.0)))
    assert d is False and dup == "pa"


def test_phoenix_never_collides_with_other_family():
    # a phoenix row (6-D identity) never collides with a c-plane row (None) or a julia row
    # (2-D identity) at the identical viewport.
    kp = ps.row_phoenix_key(_ph_row("p", 0, 0, 3.0))
    assert ps.near_dup(0, 0, 3.0, 0, 0, 3.0, k=1.5, a_c=kp, b_c=None) is False        # vs c-plane
    assert ps.near_dup(0, 0, 3.0, 0, 0, 3.0, k=1.5, a_c=kp, b_c=(0.3, -0.1)) is False  # vs julia


def test_build_cloud_keeps_distinct_phoenix_as_separate_places():
    rows = [
        _ph_row("pa", 0.0, 0.0, 3.0, p=(-0.5, 0.0)),
        _ph_row("pb", 0.0, 0.0, 3.0, p=(-0.4, 0.0)),          # distinct p, same viewport
        _ph_row("pa2", 0.01, 0.0, 3.0, p=(-0.5, 0.0)),        # same-params revisit of pa -> collapses
    ]
    cloud = ps.build_cloud(rows, "phoenix")
    assert {m["id"] for m in cloud} == {"pa", "pb"}


def test_coord_overlap_detects_phoenix_dups():
    """campaign1_readout.coord_overlap must key the CANDIDATE by row_ident (family-aware),
    not row_seed_c (julia-only). The bug: row_seed_c returns None for a phoenix admission,
    so near_dup's identity gate treats it as a c-plane row and every phoenix overlap reads
    as DISTINCT — a wrong diversity diagnostic. Regression: a same-params phoenix admission
    at a near viewport must register as an overlap; a distinct-params one must not."""
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    import campaign1_readout as c1

    priors = {"phoenix": ps.build_cloud([_ph_row("pa", 0.0, 0.0, 1.0, p=(-0.5, 0.0))], "phoenix")}
    # coord_overlap replays the RETIRED rule (the campaigns it reads out ran under it), so
    # the fixture distance is stated against that radius, not the live one.
    r = ps.RETIRED_DEDUP_K * 1.0
    dup = _ph_row("cand_dup", 0.5 * r, 0.0, 1.0, p=(-0.5, 0.0))      # same params, inside the disc
    distinct = _ph_row("cand_new", 0.5 * r, 0.0, 1.0, p=(-0.4, 0.0))  # distinct p -> different place

    hit, tot, per_fam = c1.coord_overlap([dup], priors)
    assert (hit, tot) == (1, 1), "same-params phoenix admission must read as an overlap"
    assert per_fam["phoenix"] == [1, 1]

    hit, tot, _ = c1.coord_overlap([distinct], priors)
    assert (hit, tot) == (0, 1), "distinct-params phoenix admission must read as distinct"


def test_legacy_phoenix_row_resolves_ushiki():
    """Backward compat against a REAL legacy ledger row: a pre-axis phoenix row carries no
    phoenix_c/p/zm1 fields, so its identity resolves to the classic Ushiki point byte-for-
    byte — i.e. it dedups identically to an explicit-Ushiki row and never with a distinct one."""
    led = ROOT / "data" / "discovery" / "gather" / "phoenix" / "outcome_ledger.jsonl"
    if not led.exists():
        import pytest
        pytest.skip("no legacy phoenix ledger present")
    import json
    with open(led, encoding="utf-8") as f:
        legacy = json.loads(f.readline())
    assert legacy.get("family") == "phoenix"
    # the real legacy schema predates the axes.
    assert "phoenix_c_re" not in legacy and "phoenix_p_re" not in legacy \
        and "phoenix_zm1_re" not in legacy
    # its resolved identity IS the Ushiki point (byte-for-byte with an explicit stamp).
    assert ps.row_phoenix_key(legacy) == ps.row_phoenix_key(ps.phoenix_ident_fields())
    assert ps.row_phoenix_key(legacy) == (0.5667, 0.0, -0.5, 0.0, 0.0, 0.0)
    # so a legacy row and an explicit-Ushiki row at the same viewport collide (same place),
    # while a distinct-param row does not.
    ushiki = dict(legacy, outcome_cx=legacy["outcome_cx"], outcome_cy=legacy["outcome_cy"],
                  outcome_fw=legacy["outcome_fw"], **ps.phoenix_ident_fields())
    other = dict(ushiki, **ps.phoenix_ident_fields(p=(-0.3, 0.0)))
    assert ps.near_dup(legacy["outcome_cx"], legacy["outcome_cy"], legacy["outcome_fw"],
                       ushiki["outcome_cx"], ushiki["outcome_cy"], ushiki["outcome_fw"],
                       a_c=ps.row_phoenix_key(legacy), b_c=ps.row_phoenix_key(ushiki)) is True
    assert ps.near_dup(legacy["outcome_cx"], legacy["outcome_cy"], legacy["outcome_fw"],
                       other["outcome_cx"], other["outcome_cy"], other["outcome_fw"],
                       a_c=ps.row_phoenix_key(legacy), b_c=ps.row_phoenix_key(other)) is False


# --------------------------------------------------------------------------- #
# q3-density rejection rule (the coverage-control mechanism)
# --------------------------------------------------------------------------- #
def _cloud(pts):
    """Build a cloud of point members at (cx, cy) with tiny fw (points, not zoom)."""
    return [{"id": f"m{i}", "outcome_cx": x, "outcome_cy": y, "outcome_fw": 1e-9}
            for i, (x, y) in enumerate(pts)]


def test_count_within_radius():
    cloud = _cloud([(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.5, 0.5)])
    # radius 0.20 around the origin catches the first three, not the far corner.
    assert ps.count_within(cloud, 0.0, 0.0, radius=0.20) == 3
    assert ps.count_within(cloud, 0.0, 0.0, radius=0.08) == 2   # only (0,0) + (0.05,0)
    assert ps.count_within([], 0.0, 0.0, radius=0.20) == 0


def test_rejection_rule_dense_vs_open(monkeypatch):
    monkeypatch.setattr(ps, "REJECT_RADIUS", 0.20)
    monkeypatch.setattr(ps, "Q3_DENSITY_CAP", 5)
    # 5 distinct members clustered at the origin -> a seed there hits the cap -> reject.
    dense = _cloud([(0.0, 0.0), (0.03, 0.0), (0.0, 0.03), (-0.03, 0.0), (0.0, -0.03)])
    assert ps.count_within(dense, 0.0, 0.0, ps.REJECT_RADIUS) >= ps.Q3_DENSITY_CAP
    # a seed in open space (far from every member) is under the cap -> accept.
    assert ps.count_within(dense, 5.0, 5.0, ps.REJECT_RADIUS) < ps.Q3_DENSITY_CAP


def test_near_dup_does_not_double_count_a_region():
    """A near-dup outcome does not enter the cloud, so it can't push a region over the
    density cap by being counted twice. build_cloud dedups by DEDUP_K*min(fw), so the
    near-dup's offset is stated against that radius rather than a frozen literal."""
    rows = [
        {"id": "a", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # near-dup of a (well inside DEDUP_K*min(fw)): must NOT create a second member.
        {"id": "a2", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.5 * ps.DEDUP_K * 0.01, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # genuinely distinct q3 place.
        {"id": "b", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.10, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # class-2 and guard-failed rows never enter the q3 cloud.
        {"id": "c2", "guard_pass": True, "decoded_class": 2,
         "outcome_cx": 0.11, "outcome_cy": 0.0, "outcome_fw": 0.01},
        {"id": "gf", "guard_pass": False, "decoded_class": None,
         "outcome_cx": 0.12, "outcome_cy": 0.0, "outcome_fw": 0.01},
    ]
    cloud = ps.build_cloud(rows, "mandelbrot")     # keyless rows default to mandelbrot
    ids = {m["id"] for m in cloud}
    assert ids == {"a", "b"}                       # a2 deduped; c2/gf excluded
    # the region around a holds exactly ONE counted member, not two.
    assert ps.count_within(cloud, 0.0, 0.0, radius=0.02) == 1


# --------------------------------------------------------------------------- #
# ledger round-trip (write -> reload -> rows + feats preserved; cross-run cumulative)
# --------------------------------------------------------------------------- #
def _isolate_ledgers(tmp_path, monkeypatch):
    d = tmp_path / "discovery"
    monkeypatch.setattr(ps, "DISCOVERY_DIR", d)
    monkeypatch.setattr(ps, "OUTCOME_LEDGER", d / "outcome_ledger.jsonl")
    monkeypatch.setattr(ps, "OUTCOME_FEATS", d / "outcome_feats.npz")
    monkeypatch.setattr(ps, "PROBE_REJECTS", d / "probe_rejects.jsonl")
    return d


def test_ledger_round_trip(tmp_path, monkeypatch):
    _isolate_ledgers(tmp_path, monkeypatch)
    led = ps.Ledgers()
    row_q3 = {"id": "m_x_000001", "distinct": True, "guard_pass": True, "decoded_class": 3,
              "outcome_cx": 0.1, "outcome_cy": 0.2, "outcome_fw": 0.01, "k3": 1.9}
    row_dup = {"id": "m_x_000002", "distinct": False, "dup_of": "m_x_000001",
               "guard_pass": True, "decoded_class": 3,
               "outcome_cx": 0.1, "outcome_cy": 0.2, "outcome_fw": 0.01, "k3": 1.8}
    led.append_outcome(row_q3, np.arange(1280, dtype=np.float32))
    led.append_outcome(row_dup, np.ones(1280, dtype=np.float32))
    led.save_feats()

    led2 = ps.Ledgers()   # fresh reload
    assert led2.n_outcomes_logged == 2
    assert len(led2.harvested) == 2                        # both guard_pass
    cloud = ps.build_cloud(led2.rows, "mandelbrot")        # keyless rows default to mandelbrot
    assert [m["id"] for m in cloud] == ["m_x_000001"]      # dup collapses to one place
    assert "m_x_000001" in led2.feats and led2.feats["m_x_000001"].shape == (1280,)
    assert float(led2.feats["m_x_000001"][5]) == 5.0       # feature preserved

    # cross-run cumulative: a second run appends and reloads with combined state.
    led2.append_outcome({"id": "m_x_000003", "distinct": True, "guard_pass": True,
                         "decoded_class": 3, "outcome_cx": 9.0, "outcome_cy": 9.0,
                         "outcome_fw": 0.01, "k3": 2.0}, None)
    assert ps.Ledgers().n_outcomes_logged == 3


def test_build_cloud_excludes_pre_decode_rows():
    """No historical backfill: rows predating the decoded_class field (no key) never enter
    the q3 cloud — only rows the new pipeline logged with decoded_class == 3 do."""
    rows = [
        # historical row: guard_pass but no decoded_class key -> excluded.
        {"id": "old", "guard_pass": True,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # new-pipeline q3 row -> included.
        {"id": "new", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 5.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
    ]
    assert [m["id"] for m in ps.build_cloud(rows, "mandelbrot")] == ["new"]


def test_build_cloud_partitions_by_family():
    """The `family` arg is the correctness fix: cross-family outcomes at the SAME (cx, cy)
    are different parameter planes and must never interact. build_cloud returns only the
    active partition; keyless rows count as mandelbrot."""
    rows = [
        # same coords, three different planes -> each partition sees exactly its own row.
        {"id": "m", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},              # keyless
        {"id": "j", "family": "julia", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
        {"id": "mb", "family": "multibrot_d3", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
    ]
    assert [m["id"] for m in ps.build_cloud(rows, "mandelbrot")] == ["m"]
    assert [m["id"] for m in ps.build_cloud(rows, "julia")] == ["j"]
    assert [m["id"] for m in ps.build_cloud(rows, "multibrot_d3")] == ["mb"]
    assert ps.build_cloud(rows, "phoenix") == []   # no member in that partition
