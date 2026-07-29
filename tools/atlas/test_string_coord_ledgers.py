"""String-coord ledger robustness — q4_harvest / classic_phoenix serialize outcome_cx/cy/fw
as decimal STRINGS (unlike the float-coord discovery ledgers). Every coord consumer must
COERCE (near-dup suppression stays on) rather than silently fail to match OR crash.

Findings this pins (cc_prompt_tier1_closers.md task D):
  * near_dup / is_distinct / build_cloud already coerce -> near-dup suppression + cloud-
    building WORK on string coords (they are NOT silently off).
  * count_within / dup_penalty did raw `str - float` and RAISED TypeError on a string-coord
    cloud (a latent crash). They now coerce like near_dup — these tests are RED before that
    fix (TypeError) and GREEN after (correct numeric result).

  uv run pytest tools/atlas/test_string_coord_ledgers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import production_seeder as ps          # noqa: E402
import steered_frontier as sf           # noqa: E402


def _srow(id, cx, cy, fw, family="mandelbrot"):
    """A guard-passing q3 ledger row with STRING coords (the q4_harvest/classic_phoenix form)."""
    return {"id": id, "family": family, "decoded_class": 3, "guard_pass": True,
            "distinct": True, "outcome_cx": str(cx), "outcome_cy": str(cy),
            "outcome_fw": str(fw)}


# --------------------------------------------------------------------------- #
# near-dup suppression + cloud-building are NOT silently off for string coords.
# --------------------------------------------------------------------------- #
def test_near_dup_coerces_string_coords():
    # Same near_dup, all string args: coerces, so a near pair is still a dup.
    assert ps.near_dup("1.4", "0.0", "1.0", "0.0", "0.0", "1.0", k=1.5) is True
    assert ps.near_dup("1.6", "0.0", "1.0", "0.0", "0.0", "1.0", k=1.5) is False


def test_build_cloud_dedups_string_coord_rows():
    rows = [
        _srow("a", "0.5", "0.0", "0.001"),
        _srow("b", "0.5000005", "0.0", "0.001"),   # within 1.5*fw of a -> deduped
        _srow("c", "2.0", "0.0", "0.001"),          # far -> distinct
    ]
    cloud = ps.build_cloud(rows, "mandelbrot")
    # 3 string-coord rows -> 2 distinct places: suppression RAN (not silently off, no crash).
    assert len(cloud) == 2
    assert {r["id"] for r in cloud} == {"a", "c"}


def test_is_distinct_suppresses_across_string_and_float():
    cloud = [_srow("a", "0.5", "0.0", "1.0")]          # string-coord prior
    # a FLOAT candidate at the same place (a fresh-run admission) must read as a near-dup.
    distinct, dup = ps.is_distinct(0.5, 0.0, 1.0, cloud, k=1.5)
    assert distinct is False and dup == "a"


# --------------------------------------------------------------------------- #
# count_within / dup_penalty — RED (TypeError) before the coercion fix, GREEN after.
# --------------------------------------------------------------------------- #
def test_count_within_coerces_string_cloud():
    cloud = [_srow("a", "0.5", "0.0", "0.001"), _srow("b", "0.5000005", "0.0", "0.001"),
             _srow("c", "2.0", "0.0", "0.001")]
    # pre-fix: `m["outcome_cx"] - cx` raised TypeError('str' and 'float'). Now: correct count.
    assert ps.count_within(cloud, 0.5, 0.0, radius=0.01) == 2
    assert ps.count_within(cloud, 0.5, 0.0, radius=0.0000001) == 1
    # a string candidate coord (e.g. jc from a string-coord row) also coerces.
    assert ps.count_within(cloud, "0.5", "0.0", radius=0.01) == 2


def test_dup_penalty_coerces_string_cloud():
    cloud = [_srow("a", "0.5", "0.0", "0.001")]
    # pre-fix: raised TypeError. Now: finite, large near the cloud point, ~0 far away.
    near = sf.dup_penalty(0.5, 0.0, cloud)
    far = sf.dup_penalty(50.0, 0.0, cloud)
    assert near == pytest.approx(sf.DUP_P0, rel=1e-6)
    assert 0.0 <= far < near


# --------------------------------------------------------------------------- #
# Exercise the REAL committed ledgers when present (belt-and-suspenders).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel,family", [
    ("data/discovery/classic_phoenix/outcome_ledger.jsonl", "phoenix"),
])
def test_real_string_ledger_cloud_builds_without_crash(rel, family):
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not on disk")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert isinstance(rows[0]["outcome_cx"], str)      # confirm the string-coord premise
    cloud = ps.build_cloud(rows, family)               # coerces; no TypeError
    assert 0 < len(cloud) <= len(rows)                 # dedup ran
    # count_within over the real string-coord cloud no longer crashes.
    _ = ps.count_within(cloud, float(cloud[0]["outcome_cx"]), float(cloud[0]["outcome_cy"]), 0.2)
