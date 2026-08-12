"""Unit tests for the BAND-targeting auto-level's math (no rendering, no GPU).

The four properties the band rule rests on, each one a failure the point-targeting v2 had:
identity inside the band (so an already-good image is untouched), a minimum pull to the
NEAREST edge outside it, non-clamping tails (so true black stays black), and a chroma guard
that can only ever turn the black end OFF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.palettes.color import oklab_to_srgb, srgb_to_oklab       # noqa: E402
from tools.studies import palette_autolevel_band as PB              # noqa: E402

# Stand-in for the LevelsCheck band, rounded — the real one is measured at run time
# (`levels_reference.json`) and is deliberately NOT a constant of the module.
BANDS = {"black_pt": (0.0, 0.25), "white_pt": (0.90, 0.995), "mid": (0.31, 0.72)}


def _stats(black, white, mid, *, guarded=False, neutral_frac=0.5):
    return {"black_pt": None if guarded else black, "black_pt_all": black,
            "black_pt_neutral": None if guarded else black,
            "black_unmeasurable": "test" if guarded else None,
            "neutral_frac": neutral_frac, "white_pt": white, "mid": mid}


def test_all_three_in_band_is_exactly_the_identity():
    """The whole point of a band: an image whose statistics are already acceptable comes out
    bit-unchanged. `rmf_0504` (blacks 0.084, mid 0.671) is this case."""
    cur = PB.derive_band_curve(_stats(0.084, 0.994, 0.671), BANDS)
    assert cur["applies"] and cur["identity"]
    L = np.linspace(0.0, 1.0, 1024)
    assert np.abs(PB.apply_curve_L(L, cur) - L).max() < 1e-12


def test_out_of_band_pulls_to_the_nearest_edge_not_the_centre():
    """A midtone above the band lands ON the upper edge — not on the band's middle, and not
    on the reference median."""
    cur = PB.derive_band_curve(_stats(0.10, 0.95, 0.80), BANDS)
    assert cur["mid_target"] == pytest.approx(BANDS["mid"][1])
    got = float(PB.apply_curve_L(np.array([0.80]), cur)[0])
    assert got == pytest.approx(BANDS["mid"][1], abs=1e-6)
    # ... and a midtone below the band lands on the LOWER edge.
    cur = PB.derive_band_curve(_stats(0.10, 0.95, 0.20), BANDS)
    assert cur["mid_target"] == pytest.approx(BANDS["mid"][0])


def test_the_curve_is_monotone_and_maps_the_full_range():
    """Non-clamping tails: 0 -> 0 and 1 -> 1 always, so a true black is never lifted and a
    true white is never dimmed. v2 clamped, collapsing everything below the 0.5th percentile
    onto one lightness."""
    for st in (_stats(0.35, 0.85, 0.80), _stats(0.02, 0.99, 0.25), _stats(0.30, 0.95, 0.75)):
        cur = PB.derive_band_curve(st, BANDS)
        L = np.linspace(0.0, 1.0, 2048)
        y = PB.apply_curve_L(L, cur)
        assert np.all(np.diff(y) >= -1e-12)
        assert y[0] == pytest.approx(0.0, abs=1e-12)
        assert y[-1] == pytest.approx(1.0, abs=1e-12)


def test_the_curve_is_continuous_at_both_knots():
    cur = PB.derive_band_curve(_stats(0.35, 0.85, 0.80), BANDS)
    for knot in (cur["black_pt"], cur["white_pt"]):
        lo = float(PB.apply_curve_L(np.array([knot - 1e-6]), cur)[0])
        hi = float(PB.apply_curve_L(np.array([knot + 1e-6]), cur)[0])
        assert abs(hi - lo) < 1e-4


def test_the_chroma_guard_only_ever_turns_the_black_end_off():
    """`mc20453_bc4abf7b`: an all-pixel black of 0.333 sits far above the band, but its dark
    tail is a saturated blue and there is no neutral black to read. The guard must leave the
    black endpoint where it is — never send it to the band edge."""
    st = _stats(0.333, 0.947, 0.745, guarded=True, neutral_frac=0.019)
    cur = PB.derive_band_curve(st, BANDS)
    assert cur["black_guarded"] and cur["out_ends"][0] == pytest.approx(0.333)
    # unguarded, the same numbers WOULD move the black end down to the band edge
    cur2 = PB.derive_band_curve(_stats(0.333, 0.947, 0.745), BANDS)
    assert cur2["out_ends"][0] == pytest.approx(BANDS["black_pt"][1])


def test_tone_stats_guards_a_chromatic_dark_tail_and_passes_a_neutral_one():
    """The guard's own measurement, on synthetic images rather than on the rule's inputs."""
    rng = np.random.default_rng(0)
    # (a) a saturated blue field with a neutral highlight: dark tail is chromatic
    lab = np.zeros((64, 64, 3))
    lab[..., 0] = rng.uniform(0.30, 0.55, (64, 64))
    lab[..., 2] = -0.16
    img = np.clip(oklab_to_srgb(lab.reshape(-1, 3)).reshape(64, 64, 3), 0, 1)
    img[:2] = 0.95                                            # a thin neutral highlight
    st = PB.tone_stats((img * 255).round().astype(np.uint8))
    assert st["black_pt"] is None and st["black_unmeasurable"]
    # (b) a neutral ramp: measurable, and close to the all-pixel value
    g = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    st = PB.tone_stats((np.stack([g] * 3, -1) * 255).round().astype(np.uint8))
    assert st["black_pt"] is not None
    assert abs(st["black_pt"] - st["black_pt_all"]) < 0.02


def test_the_chroma_cap_holds_a_saturated_stop_back():
    """Darkening a saturated colour costs chroma to the gamut pullback; the cap walks the
    lightness back until at least CHROMA_RETAIN of the chroma survives."""
    lab = srgb_to_oklab(np.array([[0.0, 0.15, 0.95]]))[0]      # saturated blue
    L, c0 = float(lab[0]), float(np.hypot(lab[1], lab[2]))
    Lp = 0.12                                                  # a hard darkening
    assert PB._chroma_after(np.array([Lp, lab[1], lab[2]])) < PB.CHROMA_RETAIN * c0
    Lc, capped = PB.cap_lightness(L, Lp, float(lab[1]), float(lab[2]))
    assert capped and Lp < Lc <= L
    assert PB._chroma_after(np.array([Lc, lab[1], lab[2]])) >= PB.CHROMA_RETAIN * c0 - 1e-6


def test_the_chroma_cap_is_inert_on_an_in_gamut_move():
    """It must not become a second tone curve: a neutral stop, and a mild move of a
    saturated one, both come through untouched."""
    Lc, capped = PB.cap_lightness(0.5, 0.4, 0.0, 0.0)
    assert not capped and Lc == pytest.approx(0.4)
    lab = srgb_to_oklab(np.array([[0.2, 0.45, 0.6]]))[0]       # modest chroma
    Lc, capped = PB.cap_lightness(float(lab[0]), float(lab[0]) - 0.02,
                                  float(lab[1]), float(lab[2]))
    assert not capped


def test_identity_curve_reproduces_the_palette_through_the_lut_surgery():
    """End to end on real palettes: the identity curve + the chroma cap must return the
    original sRGB8 stops (to at most 1 code of re-quantization from densification)."""
    lib = PB.PA.load_library()
    names = sorted(lib)[:: max(1, len(lib) // 6)][:6]
    ident = {"applies": True, "black_pt": 0.0, "white_pt": 1.0, "exponent": 1.0,
             "out_ends": [0.0, 1.0]}
    for n in names:
        e = lib[n]
        got, n_capped = PB.curved_stops(e["stops"], bool(e.get("mirror_needed")), ident)
        assert n_capped == 0
        orig = sorted(([float(p) % 1.0, [int(c) for c in rgb]] for p, rgb in e["stops"]),
                      key=lambda x: x[0])
        emitted = {round(p, 6): rgb for p, rgb in got}
        for p, rgb in orig:
            if round(p, 6) in emitted:
                assert np.abs(np.array(emitted[round(p, 6)]) - np.array(rgb)).max() <= 1


def test_project_is_the_nearest_point_of_the_band():
    assert PB.project(0.5, (0.3, 0.7)) == (0.5, 0)
    assert PB.project(0.1, (0.3, 0.7)) == (0.3, -1)
    assert PB.project(0.9, (0.3, 0.7)) == (0.7, +1)
