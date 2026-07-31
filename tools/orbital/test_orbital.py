"""Acceptance for the orbital-falloff criterion
(`prompts/orbital_falloff_criterion.md`).

The measures exist to be *falsifiable*, so most of these tests pin outcomes rather than
implementation — including the outcomes that came back negative. A measure that fails
validation is a result, and these tests make sure it stays reported as one instead of
being quietly tuned until it passes.

Run:  uv run python -m pytest tools/orbital/test_orbital.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "descent", REPO_ROOT / "tools" / "sources",
          REPO_ROOT / "tools" / "explorer", REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import field_metrics as fm     # noqa: E402

MEASURES = REPO_ROOT / "data" / "orbital" / "measures.jsonl"
VALIDATION = REPO_ROOT / "data" / "orbital" / "validation.json"


# --------------------------------------------------------------------------- #
# the shading constant the whole criterion rests on
# --------------------------------------------------------------------------- #
def test_density_matches_the_render_path():
    """`coloring::shade` computes t = smooth_iter*density + offset with density fixed at
    the ShadeArgs default. If that default ever moves, every ring count here silently
    changes meaning, so it is pinned against the Rust source."""
    cli = (REPO_ROOT / "src" / "cli.rs").read_text(encoding="utf-8")
    i = cli.index("pub struct ShadeArgs")
    seg = cli[i:i + 400]
    assert "default_value_t = 0.025" in seg, "ShadeArgs::density default moved"
    assert fm.DENSITY == 0.025 and fm.CYCLE_ITERS == 40.0


# --------------------------------------------------------------------------- #
# measures on synthetic fields — behaviour, not magic numbers
# --------------------------------------------------------------------------- #
def _radial_ramp(h=180, w=320, inner=4000.0, outer=100.0, power=1.0):
    """A synthetic minibrot-ish field: high smooth_iter at the centre falling outward."""
    fy = (np.arange(h) + 0.5) / h - 0.5
    fx = (np.arange(w) + 0.5) / w - 0.5
    r = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * (h / w)) ** 2)
    r = r / r.max()
    return (inner + (outer - inner) * r ** power).astype("f4")


def test_cycles_spanned_counts_colour_cycles():
    f = _radial_ramp(inner=4000.0, outer=100.0)
    got = fm.cycles_spanned(f)
    v = f[np.isfinite(f)]
    want = (np.percentile(v, 95) - np.percentile(v, 5)) * fm.DENSITY
    assert got == pytest.approx(want, rel=1e-6)
    # a flat field spans nothing
    assert fm.cycles_spanned(np.full((180, 320), 300.0, dtype="f4")) == pytest.approx(0.0)


def test_radial_rings_scales_with_dynamic_range():
    """More iterations across the frame = more rings crossed going out. This is the
    whole mechanism: one cycle is 40 iterations regardless of depth."""
    lo = fm.radial_rings(_radial_ramp(inner=300.0, outer=100.0))[0]
    hi = fm.radial_rings(_radial_ramp(inner=8000.0, outer=100.0))[0]
    assert hi > lo * 5, (lo, hi)
    assert fm.radial_rings(np.full((180, 320), 300.0, dtype="f4"))[0] == 0


def test_radial_rings_ignores_interior_but_counts_the_rest():
    """A black island in the middle costs its own span and nothing more — NaN breaks a
    ray into segments and crossings are counted within segments."""
    f = _radial_ramp(inner=8000.0, outer=100.0)
    base = fm.radial_rings(f)[0]
    g = f.copy()
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    g[((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.12 * w) ** 2] = np.nan
    holed = fm.radial_rings(g)[0]
    assert holed > 0 and holed <= base


def test_falloff_extent_is_wide_for_a_slow_ramp_and_narrow_for_a_skin():
    slow = fm.falloff_extent(_radial_ramp(inner=4000.0, outer=100.0, power=1.0))
    # a "thin skin": high only in a narrow annulus, background everywhere else
    f = np.full((180, 320), 100.0, dtype="f4")
    fy = (np.arange(180) + 0.5) / 180 - 0.5
    fx = (np.arange(320) + 0.5) / 320 - 0.5
    r = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * (180 / 320)) ** 2)
    f[r < 0.05] = 4000.0
    skin = fm.falloff_extent(f)
    assert slow > skin, (slow, skin)


def test_interior_profile_is_a_fraction_and_a_radial_curve():
    f = _radial_ramp()
    f[:20, :] = np.nan
    frac, prof = fm.interior_profile(f)
    assert 0 < frac < 1 and len(prof) == 8 and all(0 <= x <= 1 for x in prof)


# --------------------------------------------------------------------------- #
# the recorded validation outcomes (§2) — including the failures
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not VALIDATION.exists(), reason="validation not run")
def test_radial_rings_separates_both_references_from_all_triage_atoms():
    """The measure that survived: both references rank above ALL 200 triage atoms."""
    v = json.loads(VALIDATION.read_text(encoding="utf-8"))["measures"]["radial_rings"]
    assert v["refs_above_all_triage"] is True
    assert v["triage_atoms_at_or_above_eye"] == 0
    assert v["triage_atoms_at_or_above_mb19"] == 0


@pytest.mark.skipif(not VALIDATION.exists(), reason="validation not run")
def test_the_minibroteye_test_is_not_depth_in_disguise():
    """`minibroteye` is shallow (fw 5.8e-4, and not even a nucleus) while `mb19_p35` is
    at 8e-10. A measure that ranked the eye low would just be depth wearing a disguise.
    The eye must score at least as high as mb19."""
    v = json.loads(VALIDATION.read_text(encoding="utf-8"))["measures"]["radial_rings"]
    assert v["eye"] >= v["mb19"], (v["eye"], v["mb19"])


@pytest.mark.skipif(not MEASURES.exists(), reason="measures not run")
def test_radial_rings_is_only_weakly_correlated_with_depth():
    """The population-level version of the same check: if the measure were depth in
    disguise it would track log10|A| tightly. It does not."""
    rows = [json.loads(l) for l in MEASURES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("log10_abs_A") is not None]
    la = np.array([r["log10_abs_A"] for r in rows])
    rr = np.array([r["radial_rings"] for r in rows])
    rho = np.corrcoef(np.argsort(np.argsort(la)), np.argsort(np.argsort(rr)))[0, 1]
    assert abs(rho) < 0.6, f"spearman {rho:+.3f} — too close to being depth itself"


@pytest.mark.skipif(not VALIDATION.exists(), reason="validation not run")
def test_the_measures_that_failed_are_recorded_as_failed():
    """`cycles_spanned` and `falloff_extent` did NOT separate the references from the
    triage atoms. Pinned so the negative results stay reported rather than being tuned
    away: if one of these ever passes, that is a real change worth re-reading, not a
    silent improvement."""
    m = json.loads(VALIDATION.read_text(encoding="utf-8"))["measures"]
    assert m["cycles_spanned"]["separates"] is False
    assert m["falloff_extent"]["separates"] is False
    assert m["falloff_extent"]["triage_atoms_at_or_above_eye"] > 50


# --------------------------------------------------------------------------- #
# screening resolution
# --------------------------------------------------------------------------- #
def test_screen_geometry_is_much_cheaper_than_measure_geometry():
    a = fm.SCREEN_W * fm.SCREEN_H * fm.SCREEN_SS ** 2
    b = fm.MEASURE_W * fm.MEASURE_H * fm.MEASURE_SS ** 2
    assert a * 20 < b, "the screen must be far cheaper than the full measure"


def test_measure_keeps_no_field_files(tmp_path):
    """Field dumps are transient — 10k screening renders must not leave 10k .bin files."""
    before = set(tmp_path.rglob("*"))
    fm.measure_location("-0.746339", "0.112242", 5.83e-4, 500,
                        width=fm.SCREEN_W, height=fm.SCREEN_H, ss=fm.SCREEN_SS,
                        tmpdir=str(tmp_path))
    assert set(tmp_path.rglob("*")) == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
