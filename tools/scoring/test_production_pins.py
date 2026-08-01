"""The production pins resolve to exactly what they resolved to before the split.

`tools/scoring/production_pins.py` was carved out of `active_ckpt.py` on 2026-07-31.
The whole point of the split was that NOTHING changes for the ~41 importers, so the
values below are the pre-split ones, read off the unsplit module and transcribed here
by hand — an independent anchor, not a re-derivation of the code under test.

Two assertions, both load-bearing:
  1. every constant equals its pre-split value;
  2. `active_ckpt` (the old import path, still used by every caller) hands back the
     SAME OBJECT as `production_pins` — a copy that drifted would pass (1) today and
     diverge silently on the next flip.

`auto_maxiter` is pinned at nine fw values spanning the policy's whole range, incl.
both clamps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import production_pins as pins  # noqa: E402

# --- pre-split values, recorded 2026-07-31 before the carve-out ---
# cmd: python -c "import active_ckpt as m; print(m.ACTIVE_CKPT, ...)"  (see the split commit)
PRE_SPLIT = {
    "ACTIVE_CKPT": "data/classifier/v8/model_best.pt",
    "V7_CKPT_ROLLBACK": "data/classifier/v7/model_best.pt",
    "V6_CKPT_ROLLBACK": "data/classifier/v6/model_best.pt",
    "V5_CKPT_ROLLBACK": "data/classifier/v5/model_best.pt",
    "DEFAULT_MODEL": "data/classifier/v8/model_best.pt",
    "ACTIVE_VERSION": "v8",
    "JPG_Q": 90,
    "DEFAULT_SS": 4,
    "PALETTE": "twilight_shifted",
    "FW_HOME": 3.0,
    "MAXITER_BASE": 4000,
    "MAXITER_K": 0.30,
    "MAXITER_MIN": 200,
    "MAXITER_MAX": 67000,
}

# (fw, maxiter) pre-split, incl. both ends: 3.0 hits the base, 1e-15 is under the clamp.
PRE_SPLIT_MAXITER = [
    (3.0, 4000), (1.0, 5901), (0.76, 6377), (0.1, 9888), (1e-3, 17860),
    (1e-6, 29819), (1e-9, 41778), (3.3e-10, 43698), (1e-15, 65696),
]


@pytest.mark.parametrize("name,expected", sorted(PRE_SPLIT.items()))
def test_constant_unchanged_by_the_split(name, expected):
    assert getattr(pins, name) == expected, (
        f"production_pins.{name} is {getattr(pins, name)!r}, was {expected!r} before the "
        "2026-07-31 carve-out — the split was supposed to move code, not values")


@pytest.mark.parametrize("fw,expected", PRE_SPLIT_MAXITER)
def test_auto_maxiter_unchanged_by_the_split(fw, expected):
    assert pins.auto_maxiter(fw) == expected


def test_bin_is_the_release_binary_path():
    assert pins.BIN == ROOT / "target" / "release" / "fractal-generator.exe"


def test_active_version_derives_from_the_checkpoint_path():
    """Derive-in-code: the version token is READ OFF the pin, never restated."""
    assert pins.ACTIVE_VERSION == Path(pins.ACTIVE_CKPT).parent.name


def test_active_ckpt_reexports_the_same_objects():
    """The old import path must be an alias, not a copy."""
    import active_ckpt as legacy
    for name in list(PRE_SPLIT) + ["ROOT", "BIN", "auto_maxiter", "make_scorer"]:
        assert getattr(legacy, name) is getattr(pins, name), (
            f"active_ckpt.{name} is not production_pins.{name} — a second copy will "
            "drift on the next checkpoint flip")
