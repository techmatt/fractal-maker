"""Bracket for the recipe-derived palette_renders assertion (v8 trainer unblock).

v4..v7 baked 6 palettes per location and `Loc.palette_renders()` hardcoded
`assert len(got) == 6`. v8b bakes 4 (2 fixed + 2 drawn per location), so that literal
blocks a v8 train. The fix derives the expected count and the always-present palette
names from the recipe (`aug_roster.json`) + the cache manifest, so the next recipe
change is data, not a code edit — and the assertion still rejects a malformed location.

Bracketed three ways (the point of the change):
  * the OLD form (`len == 6`) FAILS on v8-shaped data;
  * the NEW form PASSES on v8-shaped data;
  * the NEW form still REJECTS a location with a missing or duplicated palette render.

REMOVED 2026-08-06: `test_new_form_passes_on_real_v8_locations`, the fourth bracket, which
replayed the first 6 locations out of `data/v8/cache_manifest.jsonl`. That file was DELETED
on 2026-08-03 (`tools/v8/test_v8_cache_alignment.py` records the deletion and the 146 MB it
freed), so the test had skipped every run since. It is deleted rather than left skipping for
the reason that file already states: the manifest is regenerable from the committed manifest
by `tools/v8/build_plan.py`, so "skipped" means only "nobody ran the command", printed once
per suite run forever, and a suite that skips its own brackets reads green. The defect it
caught — the derived spec disagreeing with a genuine v8 row — is carried by
`test_palette_spec_derived_from_roster`, which derives the same spec from the surviving
`data/v8/aug_roster.json`, plus the five synthetic-shape brackets below. `git show` has it
verbatim if the manifest is ever rebuilt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "classifier"))
import data_v4 as D  # noqa: E402

# `load_palette_spec` resolves the recipe as `<cache_dir>/aug_roster.json`, so this is an
# anchor for the DIRECTORY, not a file that has to exist — the manifest itself was deleted
# on 2026-08-03 while `aug_roster.json` (small, committed) stayed. The spec tests below are
# therefore live, not skipped: they read the real v8 recipe.
REAL_CACHE = ROOT / "data" / "v8" / "cache_manifest.jsonl"


# --------------------------------------------------------------------------- #
# Spec derivation from the recipe metadata.
# --------------------------------------------------------------------------- #
def test_palette_spec_derived_from_roster():
    spec = D.load_palette_spec(REAL_CACHE)
    assert spec is not None, "v8 aug_roster.json should be present beside the cache manifest"
    # 2 always + 2 drawn == 4; NOT the literal 6 the old code assumed.
    assert spec.expected_count == 4
    assert set(spec.always) == {"twilight_shifted", "blue_orange"}


def test_no_roster_yields_no_spec(tmp_path):
    """A cache without an aug_roster.json (pre-v8) yields None — palette_renders then
    reports the missing recipe rather than silently accepting any count."""
    (tmp_path / "cache_manifest.jsonl").write_text("", encoding="utf-8")
    assert D.load_palette_spec(tmp_path / "cache_manifest.jsonl") is None


# --------------------------------------------------------------------------- #
# Synthetic v8-shaped location (fast, no 93 MB manifest parse).
# --------------------------------------------------------------------------- #
def _render(palette, aa="antialiased", scale=D.CANON_SCALE, shift=D.CANON_SHIFT):
    return D.Render(path=Path(f"{palette}.jpg"), palette=palette, palette_family=palette,
                    scale=scale, shift_id=shift, aa_level=aa)


def _v8_loc(palettes, spec):
    """A location whose ss4/center/scale-1.0 set is exactly `palettes`, plus some
    off-axis renders (aliased / jittered) that palette_renders must ignore."""
    renders = [_render(p) for p in palettes]
    renders.append(_render("twilight_shifted", aa="aliased"))          # aa_twin axis
    renders.append(_render("blue_orange", scale=1.1, shift="jit0"))    # jittered geometry
    return D.Loc(location_id=1, label=3, split="train", group_id=1, source="s",
                 biased=False, fractal_type="mandelbrot", renders=renders,
                 palette_spec=spec)


V8_PALETTES = ["twilight_shifted", "blue_orange", "viridis", "magma"]  # 2 always + 2 drawn


def _old_form(loc):
    """The pre-fix assertion, reproduced verbatim for the bracket."""
    got = list(loc._pick(aa="antialiased", scale=D.CANON_SCALE, shift=D.CANON_SHIFT))
    assert len(got) == 6, f"loc {loc.location_id}: {len(got)} ss4 palette renders"
    return got


def test_old_form_fails_on_v8_shape():
    spec = D.load_palette_spec(REAL_CACHE)
    loc = _v8_loc(V8_PALETTES, spec)
    with pytest.raises(AssertionError):
        _old_form(loc)                       # 4 != 6


def test_new_form_passes_on_v8_shape():
    spec = D.load_palette_spec(REAL_CACHE)
    loc = _v8_loc(V8_PALETTES, spec)
    got = loc.palette_renders()
    assert [r.palette for r in got] == sorted(V8_PALETTES)     # 4, distinct, sorted
    for always in spec.always:
        assert always in {r.palette for r in got}


def test_new_form_rejects_missing_palette():
    spec = D.load_palette_spec(REAL_CACHE)
    loc = _v8_loc(V8_PALETTES[:-1], spec)    # only 3 of the 4
    with pytest.raises(AssertionError, match="recipe expects 4"):
        loc.palette_renders()


def test_new_form_rejects_duplicated_palette():
    spec = D.load_palette_spec(REAL_CACHE)
    # count is still 4, but two renders share a palette (a genuinely malformed cache).
    loc = _v8_loc(["twilight_shifted", "blue_orange", "viridis", "viridis"], spec)
    with pytest.raises(AssertionError, match="duplicated palette"):
        loc.palette_renders()


def test_new_form_rejects_missing_always_palette():
    spec = D.load_palette_spec(REAL_CACHE)
    # 4 distinct palettes, but the vivid companion (an always-palette) is absent.
    loc = _v8_loc(["twilight_shifted", "viridis", "magma", "inferno"], spec)
    with pytest.raises(AssertionError, match="missing always-palette"):
        loc.palette_renders()


def test_palette_renders_needs_spec():
    """Without a recipe spec the assertion refuses rather than accepting any count."""
    loc = _v8_loc(V8_PALETTES, spec=None)
    with pytest.raises(AssertionError, match="needs the recipe palette spec"):
        loc.palette_renders()
