"""Unit tests for the palette auto-level study's math (no rendering, no GPU).

The three properties the LUT surgery rests on: densification is IDENTITY on the palette,
the curve is MONOTONE and hits its midtone target, and every emitted stop is IN GAMUT at
the requested lightness (the round-trip gamut test, which is the bug this file locks down —
`oklab_to_srgb` clips, so asking the clipped output whether it clipped always says no).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.palettes.color import srgb_to_oklab                      # noqa: E402
from tools.studies import palette_autolevel as PA                   # noqa: E402

IDENT = {"applies": True, "black_pt": 0.0, "white_pt": 1.0, "exponent": 1.0,
         "out_ends": (0.0, 1.0)}


def _some_palettes(n=6):
    lib = PA.load_library()
    names = sorted(lib)[:: max(1, len(lib) // n)][:n]
    return [lib[k] for k in names]


@pytest.mark.parametrize("entry", _some_palettes())
def test_densify_is_identity_under_the_identity_curve(entry):
    """k-fold subdivision of a piecewise-linear OKLab segment reproduces the palette, so the
    only cost of a denser curve sampling is sRGB8 re-quantization (<= 1 code per channel)."""
    mirror = bool(entry.get("mirror_needed"))
    orig = sorted(([float(p) % 1.0, [int(c) for c in rgb]] for p, rgb in entry["stops"]),
                  key=lambda x: x[0])
    got = PA.densify(entry["stops"], mirror, k=1)
    rgb = np.array([PA._gamut_fit(lab) for _, lab in got])
    assert np.abs(rgb - np.array([c for _, c in orig])).max() == 0


@pytest.mark.parametrize("entry", _some_palettes(3))
def test_densify_preserves_the_stop_span_for_mirrored_palettes(entry):
    """`mirror_stops` re-bases onto [p0, p_last]; a stop outside that span would change the
    bake rather than refine it, so the wrap segment is subdivided only for cyclic maps."""
    mirror = bool(entry.get("mirror_needed"))
    pos = [float(p) % 1.0 for p, _ in entry["stops"]]
    got = [p for p, _ in PA.densify(entry["stops"], mirror, k=4)]
    if mirror:
        assert min(got) == pytest.approx(min(pos)) and max(got) == pytest.approx(max(pos))
    else:
        assert max(got) > max(pos) - 1e-9   # the wrap segment is populated


def test_curve_is_monotone_and_hits_the_midtone_target():
    st = {"black_pt": 0.2, "white_pt": 0.9, "mid": 0.7}
    for ends in ((0.0, 1.0), (0.1, 0.93)):
        cur = PA.derive_curve(st, ends)
        assert cur["applies"]
        L = np.linspace(0.0, 1.0, 512)
        y = PA.apply_curve_L(L, cur)
        assert np.all(np.diff(y) >= -1e-12)
        assert y[0] == pytest.approx(ends[0]) and y[-1] == pytest.approx(ends[1])
        # the exponent solves C(mid) = MID_TARGET unless the clamp bound it
        if not cur["clamped"]:
            assert float(PA.apply_curve_L(np.array([st["mid"]]), cur)[0]) == \
                pytest.approx(PA._mid_target(), abs=1e-6)


def test_curve_declines_on_a_degenerate_range():
    cur = PA.derive_curve({"black_pt": 0.5, "white_pt": 0.52, "mid": 0.51})
    assert not cur["applies"] and "degenerate" in cur["reason"]


def test_gamut_fit_keeps_the_requested_lightness():
    """Chroma is what gives way out of gamut — L is the axis the curve controls and must
    survive. A high-chroma color pushed to a dark L is the case that used to hard-clip."""
    lab = srgb_to_oklab(np.array([[0.05, 0.85, 0.15]]))[0]      # saturated green
    for target_L in (0.05, 0.2, 0.5, 0.95):
        rgb = np.array(PA._gamut_fit(np.array([target_L, lab[1], lab[2]])), dtype=float) / 255.0
        got = srgb_to_oklab(rgb.reshape(1, 3))[0, 0]
        assert abs(got - target_L) < 0.01
