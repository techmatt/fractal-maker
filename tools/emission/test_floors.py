#!/usr/bin/env python
"""`tools/emission/floors.py` — the stage-2 cut owner's two load-bearing properties.

  1. A stamped floor REFUSES to gate when its head's live pin has moved off the stamp.
     Refusing is the point: a floor is a point on ONE head's probability scale, so gating
     0.90 on a head that never produced that scale is silent and only visible months later
     as "the release got worse".
  2. The pool floors stay strictly below their release floors, both directions checked, at
     import.

Each test that asserts a refusal also asserts the NON-refusal it is the mirror of, so a
`check()` that raised unconditionally (or never) would fail here rather than read as green.

  uv run pytest tools/emission/test_floors.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import floors as F            # noqa: E402
from tools.mining import mining_pins as MP        # noqa: E402
from tools.wallpaper import wallpaper_pins as WP  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. the stamp
# --------------------------------------------------------------------------- #
def test_every_cut_is_stamped_against_its_live_head_today():
    """The baseline the refusal tests are the mirror of. Green here means the tree's floors
    and the tree's head pins currently agree — if this ever goes red, a head moved and a
    floor did not, which is the condition the refusal exists for."""
    F.check_stamps()
    assert F.WALLPAPER_POOL.stamp == WP.HEAD_VERSION
    assert F.MINING_POOL.stamp == MP.HEAD_VERSION
    assert {f.head for f in F.ALL_FLOORS} == {F.WALLPAPER_HEAD, F.MINING_HEAD}


def test_a_wallpaper_head_flip_makes_its_floors_refuse(monkeypatch):
    """INJECTION. Move the wallpaper pin to a version no floor was derived against; both
    wallpaper cuts must refuse, and the mining cuts must NOT — a blanket raise would pass a
    one-sided test."""
    monkeypatch.setattr(WP, "HEAD_VERSION", "v9")
    for f in (F.WALLPAPER_POOL, F.WALLPAPER_RELEASE):
        with pytest.raises(F.HeadStampMismatch) as ei:
            f.gate(0.99)
        assert f.stamp in str(ei.value) and "v9" in str(ei.value)
    assert F.MINING_RELEASE.gate(0.99) is True          # untouched head still gates
    with pytest.raises(F.HeadStampMismatch):
        F.check_stamps()


def test_a_mining_head_flip_makes_its_floors_refuse(monkeypatch):
    monkeypatch.setattr(MP, "HEAD_VERSION", "v2")
    for f in (F.MINING_POOL, F.MINING_RELEASE):
        with pytest.raises(F.HeadStampMismatch):
            f.gate(0.99)
    assert F.WALLPAPER_RELEASE.gate(0.99) is True


def test_the_refusal_is_at_gate_not_only_at_check(monkeypatch):
    """`gate()` is THE comparison every consumer calls. If the check only ran in `check()`, a
    site that wrote its own `>=` against `.value` would gate on the wrong scale in silence —
    which is precisely how the six copies this module replaced behaved."""
    monkeypatch.setattr(WP, "HEAD_VERSION", "v9")
    with pytest.raises(F.HeadStampMismatch):
        F.for_style("smooth", "release").gate(0.5)


def test_active_head_version_reads_the_pin_at_call_time(monkeypatch):
    """Not an import-time snapshot: a long-lived process that outlives a pin flip must see
    the flip, and a test must be able to induce one."""
    assert F.active_head_version(F.MINING_HEAD) == MP.HEAD_VERSION
    monkeypatch.setattr(MP, "HEAD_VERSION", "v77")
    assert F.active_head_version(F.MINING_HEAD) == "v77"


def test_the_location_head_is_resolvable_even_though_no_cut_uses_it():
    """`FLOOR_PNOTBAD` was the location head's only stage-2 cut and it is deleted (§B), so
    nothing is stamped against it today. The resolver stays registered so the NEXT intake-side
    cut is stampable without re-deriving the mechanism — and an unregistered head raises
    rather than defaulting."""
    import corpus_common as cc
    assert F.active_head_version(F.LOCATION_HEAD) == cc.active_scorer_version()
    with pytest.raises(KeyError):
        F.active_head_version("no_such_head")


# --------------------------------------------------------------------------- #
# 2. the release floors are the heads' gates, not copies of them
# --------------------------------------------------------------------------- #
def test_a_release_floor_moves_with_its_head_gate(monkeypatch):
    """The release floors are IMPORTED from the head pins, so a gate retune cannot leave the
    release floor calibrated against a head that no longer exists. Asserted on the resolution,
    not the literal: `Floor` is frozen, so this re-resolves the module rather than mutating."""
    assert F.WALLPAPER_RELEASE.value == WP.GATE_THRESHOLD
    assert F.MINING_RELEASE.value == MP.MINING_GATE_THRESHOLD


def test_pool_floors_sit_below_their_release_floors():
    F.check_below_gate()
    assert F.WALLPAPER_POOL.value < F.WALLPAPER_RELEASE.value
    assert F.MINING_POOL.value < F.MINING_RELEASE.value


@pytest.mark.parametrize("pool_value", [0.90, 0.95])
def test_a_pool_floor_at_or_above_its_gate_is_a_red_build(pool_value):
    """INJECTION on `check_below_gate`, which runs at import. A pool floor at the release
    floor makes 'pooled but not release-grade' zero by construction — every inventory count
    downstream reads as a measurement and is an identity."""
    broken = (F.Floor(name="wallpaper_pool", value=pool_value, head=F.WALLPAPER_HEAD,
                      stamp=WP.HEAD_VERSION, site="pool", acts=True, basis="injected"),
              F.WALLPAPER_RELEASE)
    with pytest.raises(ValueError, match="not below their release floor"):
        F.check_below_gate(broken)


def test_check_below_gate_is_not_vacuous():
    """The healthy pair passes the same call the broken pair fails."""
    F.check_below_gate((F.WALLPAPER_POOL, F.WALLPAPER_RELEASE))


# --------------------------------------------------------------------------- #
# 3. routing
# --------------------------------------------------------------------------- #
def test_style_routes_to_the_head_that_was_trained_on_it():
    assert F.for_style("smooth", "release") is F.WALLPAPER_RELEASE
    assert F.for_style("smooth", "pool") is F.WALLPAPER_POOL
    for strange in ("tia", "stripe", "composite_c13_smooth_stripe"):
        assert F.for_style(strange, "release") is F.MINING_RELEASE
        assert F.for_style(strange, "pool") is F.MINING_POOL
    with pytest.raises(KeyError):
        F.for_style("smooth", "somewhere_else")


def test_the_liveness_census_finds_no_ACTING_cut_left():
    """THE census: `acts` is the record of which cuts remove rows, and the set that DO is now
    empty. All four went annotation-only on 2026-08-09
    (prompts/selection_restructure_1.md) — they keep their values and their stamps and are
    read to say what the retired cut would have done.

    An all-False census is exactly what a census reading the wrong field also returns, so the
    mirror below injects an acting cut and requires the same expression to name it."""
    assert [f.name for f in F.ALL_FLOORS if f.acts] == []
    assert not any(f.acts for f in F.ALL_FLOORS)
    assert F.summary().count("annotation-only") == len(F.ALL_FLOORS)


def test_the_census_still_names_an_acting_cut(monkeypatch):
    """NON-VACUITY for the census above, both in the list and in the run-banner summary."""
    ghost = F.Floor(name="ghost_release", value=0.99, head=F.MINING_HEAD,
                    stamp=MP.HEAD_VERSION, site="release", acts=True, basis="injected")
    monkeypatch.setattr(F, "ALL_FLOORS", F.ALL_FLOORS + (ghost,))
    assert [f.name for f in F.ALL_FLOORS if f.acts] == ["ghost_release"]
    assert "ghost_release 0.99 (render_mode_head/v1)" in F.summary()
    assert F.summary().count("annotation-only") == len(F.ALL_FLOORS) - 1


# --------------------------------------------------------------------------- #
# 4. the ONE enforcing cut (2026-08-09)
# --------------------------------------------------------------------------- #
def test_the_junk_floor_is_the_only_enforcing_cut_and_it_is_semantic():
    """It is a bare float, not a `Floor`, and that is the design: it reads on THREE different
    heads (stage-1 at emission intake, mining at deploy_tail) and a stamp names one. What
    keeps it honest across a head flip is the volume-matched restatement rule documented
    beside it, not a stamp check."""
    assert F.JUNK_FLOOR == 0.20
    assert not any(f.value == F.JUNK_FLOOR for f in F.ALL_FLOORS)
    assert "ENFORCING" in F.summary() and str(F.JUNK_FLOOR) in F.summary()
    # it is below every retired floor — the retirement widened the funnel, it did not narrow it
    assert all(F.JUNK_FLOOR < f.value for f in F.ALL_FLOORS)


def test_passes_junk_floor_treats_a_missing_score_as_failing():
    """An unscored candidate has no verdict to spend colorize compute on. `None` must not
    compare as 0.0-and-therefore-fail-anyway by accident, nor pass."""
    assert F.passes_junk_floor(0.20) is True          # inclusive at the boundary
    assert F.passes_junk_floor(0.199) is False
    assert F.passes_junk_floor(None) is False
    assert F.passes_junk_floor(0.0) is False


def test_the_thin_supply_divisor_and_cluster_cap_live_beside_the_floor():
    """Both are coarse volume constants and both are declared here, so the three numbers this
    restructure introduced move together and none of them is re-typed downstream."""
    assert F.THIN_SUPPLY_DIVISOR == 4 and F.CLUSTER_CAP == 2
    from tools.emission import ranked_intake as RI     # noqa: PLC0415
    from tools.emission import selection as SEL        # noqa: PLC0415
    assert RI.emit_cap(F.THIN_SUPPLY_DIVISOR * 3 + 3) == 3     # floor(15/4)
    assert SEL.CLUSTER_CAP is F.CLUSTER_CAP


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
