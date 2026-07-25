"""The julia ledger-schema tag: detection, assertion, and the two-era collision it prevents.

Run: uv run pytest tools/corpus/test_julia_ledger_schema.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import julia_ledger_schema as jls          # noqa: E402
from tools.emission import descriptor as D  # noqa: E402  (location_of — the round-trip target)


# Two rows carrying IDENTICAL stored coordinate numbers, one per era. BOTH store
# outcome_cx/cy = (0.5, 0.6) and a secondary pair (0.1, 0.2) — but the era decides the
# roles: CAMPAIGN reads outcome_* as the viewport (c = the julia_c_* pair), WALK reads
# outcome_* as the parameter c (viewport = the julia_z_* pair). Same numbers, roles swapped
# — the exact confusion the tag prevents.
def _campaign_row():
    return {"id": "camp", "family": "julia:mandelbrot",
            "outcome_cx": 0.5, "outcome_cy": 0.6, "outcome_fw": 0.01,   # viewport
            "julia_c_re": "0.1", "julia_c_im": "0.2",                    # c
            "julia_schema": jls.CAMPAIGN}


def _walk_row():
    return {"id": "walk", "family": "julia:mandelbrot",
            "outcome_cx": 0.5, "outcome_cy": 0.6, "outcome_fw": 9.9,     # c (outcome_fw = c-plane scale)
            "julia_z_cx": 0.1, "julia_z_cy": 0.2, "julia_z_fw": 0.01,    # viewport
            "julia_schema": jls.WALK}


# --------------------------------------------------------------------------- #
# The bug, stated as a test: same stored numbers, different era → different location.
# --------------------------------------------------------------------------- #
def test_identical_numbers_two_eras_resolve_differently():
    camp = D.location_of(_campaign_row())
    walk = D.location_of(_walk_row())
    # Identical stored outcome_cx/cy=(0.5,0.6), yet: campaign viewport=(0.5,0.6) c=(0.1,0.2);
    # walk viewport=(0.1,0.2) c=(0.5,0.6). DIFFERENT canonical locations.
    assert camp.key() != walk.key()
    assert (camp.cx, camp.cy, camp.fw, camp.c_re, camp.c_im) == ("0.5", "0.6", "0.01", "0.1", "0.2")
    assert (walk.cx, walk.cy, walk.fw, walk.c_re, walk.c_im) == ("0.1", "0.2", "0.01", "0.5", "0.6")


def test_each_era_round_trips_to_its_own_fields():
    # CAMPAIGN: viewport = outcome_*, c = julia_c_*.
    cx, cy, fw, cr, ci = jls.viewport_and_c(_campaign_row())
    assert (cx, cy, fw, cr, ci) == (0.5, 0.6, 0.01, "0.1", "0.2")
    # WALK: viewport = julia_z_*, c = outcome_*.
    cx, cy, fw, cr, ci = jls.viewport_and_c(_walk_row())
    assert (cx, cy, fw, cr, ci) == (0.1, 0.2, 0.01, 0.5, 0.6)


# --------------------------------------------------------------------------- #
# Detection (back-stamp only) and assertion (live).
# --------------------------------------------------------------------------- #
def test_detect_from_field_presence():
    assert jls.detect_schema({k: v for k, v in _campaign_row().items()
                              if k != jls.SCHEMA_KEY}) == jls.CAMPAIGN
    assert jls.detect_schema({k: v for k, v in _walk_row().items()
                              if k != jls.SCHEMA_KEY}) == jls.WALK


def test_detect_rejects_contradiction_and_emptiness():
    both = dict(_campaign_row(), julia_z_cx=0.1, julia_z_cy=0.1, julia_z_fw=0.1)
    del both[jls.SCHEMA_KEY]
    with pytest.raises(ValueError):        # both layouts present → ambiguous
        jls.detect_schema(both)
    with pytest.raises(ValueError):        # neither layout → undetectable
        jls.detect_schema({"id": "x", "family": "julia:mandelbrot"})


def test_schema_of_asserts_tag():
    row = {k: v for k, v in _campaign_row().items() if k != jls.SCHEMA_KEY}
    with pytest.raises(ValueError):        # untagged julia row is a loud failure
        jls.schema_of(row)
    with pytest.raises(ValueError):        # unknown tag likewise
        jls.schema_of(dict(row, julia_schema="v2"))
    assert jls.schema_of(_campaign_row()) == jls.CAMPAIGN


def test_location_of_raises_on_untagged_julia():
    row = {k: v for k, v in _walk_row().items() if k != jls.SCHEMA_KEY}
    with pytest.raises(ValueError):
        D.location_of(row)


def test_stamp_is_idempotent_and_native_safe():
    r = {k: v for k, v in _walk_row().items() if k != jls.SCHEMA_KEY}
    assert jls.stamp(r) is True and r[jls.SCHEMA_KEY] == jls.WALK
    assert jls.stamp(r) is False           # already tagged
    native = {"id": "m", "family": "mandelbrot", "outcome_cx": -0.5,
              "outcome_cy": 0.1, "outcome_fw": 0.03}
    assert jls.stamp(native) is False and jls.SCHEMA_KEY not in native
