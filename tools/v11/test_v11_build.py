#!/usr/bin/env python
r"""v11 build guards — the claims the committed records make about themselves.

Everything here is cheap (no engine, no corpus rebuild) and reads the two COMMITTED files,
`data/v11/{build_record,aug_recipe}.json`, against the modules that wrote them. That pairing
is the point: the bulk manifest/plan/cache are regenerable and out-of-tree, so the records
are the only thing standing between "rebuildable" and "rebuildable into something else."

The expensive halves live elsewhere and are marked `slow`: `test_crop_batch.py` proves the
executor's draw, and `verify_cache.py` proves the rendered tree.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tools/corpus", "tools/scoring"):
    sys.path.insert(0, str(ROOT / sub))

import artifacts        # noqa: E402
import location as lm   # noqa: E402
import paths            # noqa: E402

RECORD = ROOT / "data" / "v11" / "build_record.json"
RECIPE = ROOT / "data" / "v11" / "aug_recipe.json"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _committed(path: Path, rebuild: str) -> dict:
    """Read a COMMITTED record, or fail loudly naming the rebuild.

    Not `pytest.skip` — `verification_practice.md` §2 is explicit that an absence-tolerant
    guard un-guards exactly when its subject is removed, and these two files are the whole
    reason the bulk manifest/plan/cache are allowed to live out-of-tree. If one is gone, the
    guard must say so, not go quiet."""
    assert path.exists(), (
        f"{path.relative_to(ROOT)} is MISSING. It is a committed record, not an output — "
        f"either it was never re-added after a rebuild (`{rebuild}`, then `git add` it) or "
        f"it was deleted. The bulk artifacts it describes are unverifiable without it.")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recipe():
    return _committed(RECIPE, "uv run python tools/v11/build_plan.py")


@pytest.fixture(scope="module")
def record():
    return _committed(RECORD, "uv run python tools/v11/build_manifest.py")


@pytest.fixture(scope="module")
def plan_mod():
    return _load("v11_build_plan", "tools/v11/build_plan.py")


# --------------------------------------------------------------------------- #
# The palette pools
# --------------------------------------------------------------------------- #
def test_draw_pool_is_the_curated_pool_minus_the_held_out_eight(recipe):
    pool = set(recipe["palettes"]["draw_pool"])
    held = set(recipe["palettes"]["held_out"])
    assert len(held) == 8
    assert len(pool) == recipe["palettes"]["curated_pool"] - len(held) == 68
    assert not (pool & held), f"a held-out palette is in the draw pool: {sorted(pool & held)}"
    src = json.loads((ROOT / "data/palettes/score3_colormaps.json").read_text(encoding="utf-8"))
    curated = {c["name"] for c in src}
    assert pool | held == curated, "the two pools do not partition the curated set"
    # twilight_shifted IS drawable in v11 (v9 pinned it and drew from 67 without it).
    assert "twilight_shifted" in pool


def test_the_held_out_eight_are_the_same_names_v9_and_v10_held_out(recipe):
    """The invariance read only spans versions if the same eight were never trainable in
    any of them — so this is held against the committed rosters, not re-derived from the
    seed (a re-derivation that happens to match is a different claim)."""
    for rel in ("data/v9/aug_roster.json", "data/v10/aug_roster.json"):
        r = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert r["palettes"]["held_out"] == recipe["palettes"]["held_out"], rel


def test_blue_orange_is_floor_only(recipe):
    """The labeler's map is not one of the curated 76, so it can only reach a location
    through the floor — which is exactly 2 tiles, on every location."""
    assert recipe["fan_out"]["floor_per_location"]["blue_orange"] == 2
    assert "blue_orange" not in recipe["palettes"]["draw_pool"]
    assert recipe["palettes"]["floor_only"] == ["blue_orange"]


# --------------------------------------------------------------------------- #
# The fan-out
# --------------------------------------------------------------------------- #
def test_the_floor_comes_out_of_the_thirty_two(recipe):
    fan = recipe["fan_out"]
    assert fan["tiles_per_location"] == 32
    assert sum(fan["floor_per_location"].values()) <= fan["tiles_per_location"]
    assert fan["floor_identity_geometries"] <= fan["tiles_per_location"]
    assert fan["floor_per_location"] == {"twilight_shifted": 2, "blue_orange": 2}


def test_the_extended_field_contains_the_widest_shifted_crop(recipe):
    """The containment bound the executor validates, restated on the RECIPE's own numbers.

    `extend >= 1 + 2*shift_max + max(1, H/W)*(scale_hi - 1)`. The shipped triple sits
    exactly on it with nothing to spare, so a nudge to any of the three without the others
    is a silent under-extension — and `build_taps_scaled` clamps rather than failing, so the
    symptom would be edge-smeared training tiles, not an error."""
    f, g = recipe["field"], recipe["geometry"]
    need = (1.0 + 2.0 * g["shift_frac_max"]
            + max(1.0, f["tile_h"] / f["tile_w"]) * (g["scale"][1] - 1.0))
    assert f["extend"] >= need - 1e-9, f"extend {f['extend']} < required {need}"


def test_aa_levels_are_the_two_cache_manifest_labels(recipe):
    labels = [lvl.split(":")[0] for lvl in recipe["aa"]["levels"]]
    assert labels == ["aliased", "antialiased"], labels
    assert [lvl.split(":")[1] for lvl in recipe["aa"]["levels"]] == ["point", "lanczos3"]


def test_the_recorded_render_command_is_the_module_constants(recipe, plan_mod):
    """The record must be DERIVED from the builder, not a transcription of it.

    Every flag below is read out of `build_plan`'s own constants, so a constant that moves
    without a rebuild goes red here instead of leaving the committed record describing a
    recipe nobody rendered."""
    cmd = recipe["render_command"]

    def flag(name):
        return cmd[cmd.index(name) + 1]

    assert flag("--tiles") == str(plan_mod.TILES_PER_LOCATION)
    assert flag("--seed-tag") == plan_mod.SEED_TAG == recipe["seed_tag"]
    assert flag("--field-ss") == str(plan_mod.FIELD_SS)
    assert flag("--extend") == str(plan_mod.EXTEND)
    assert flag("--scale-lo") == str(plan_mod.SCALE_LO)
    assert flag("--scale-hi") == str(plan_mod.SCALE_HI)
    assert flag("--shift-frac-max") == str(plan_mod.SHIFT_FRAC_MAX)
    assert flag("--jpg-quality-lo") == str(plan_mod.JPG_QUALITY_LO)
    assert flag("--jpg-quality-hi") == str(plan_mod.JPG_QUALITY_HI)
    assert flag("--floor-identity") == str(plan_mod.FLOOR_IDENTITY)
    assert flag("--aa").split() == list(plan_mod.AA_LEVELS)
    assert flag("--floor-palette").split() == [f"{n}:{c}" for n, c in plan_mod.FLOOR]
    assert flag("--palette-pool").split() == recipe["palettes"]["draw_pool"]


def test_the_cap_policy_token_is_the_live_one(recipe):
    """A stamped policy token that no longer matches the live policy means the cache and
    the deploy crop resolve different numbers — the defect v9 was built to remove."""
    assert recipe["maxiter"]["token"] == lm.maxiter_policy_token()
    assert recipe["maxiter"]["range_over_plan"]["max"] < recipe["maxiter"]["constants"]["max"]


# --------------------------------------------------------------------------- #
# Storage class
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", ["data/v11/manifest.jsonl", "data/v11/eval_slice.jsonl",
                                 "data/v11/plan.jsonl", "data/v11/cache_manifest.jsonl",
                                 "data/v11/aug_cache"])
def test_the_bulk_artifacts_resolve_out_of_tree(rel):
    """Rule 5 — a new bulk family is born out-of-tree. Registered BEFORE the first render,
    so a forgotten registration cannot leave 23 GiB of tiles in the working tree."""
    assert artifacts.is_relocated(rel), f"{rel} is not a registered bulk family"
    assert artifacts.artifacts_root() in paths.bulk(rel).parents


@pytest.mark.parametrize("rel", ["data/v11/build_record.json", "data/v11/aug_recipe.json",
                                 "data/v11/colormaps.json"])
def test_the_committed_records_are_durable(rel):
    """`durable()` raises if git would discard the path — the assertion is the call."""
    assert paths.durable(rel) == ROOT / rel


def test_the_record_and_the_recipe_do_not_disagree_about_the_population(record, recipe):
    assert record["population"]["manifest_rows"] == recipe["plan"]["locations"]
    assert (recipe["plan"]["tiles"]
            == recipe["plan"]["locations"] * recipe["fan_out"]["tiles_per_location"])
    assert record["config"]["split_seed"] == _load(
        "v11_build_manifest", "tools/v11/build_manifest.py").SPLIT_SEED


def test_the_split_is_two_eval_roles_and_the_instrument_half_is_unbiased(record):
    """The whole point of the v11 split: `holdout` is biased on purpose and `instrument` is
    not, so a consumer that reads a base rate off the wrong one is reading a ranker."""
    pop = record["population"]
    assert pop["eval_instrument"] + pop["eval_holdout"] == pop["eval"]
    assert pop["train"] + pop["eval"] == pop["manifest_rows"]
    assert set(record["eval_roles"]) == {"instrument", "holdout"}
    assert pop["eval_instrument"] == 1050, "the four instruments moved"
    assert pop["dropped_biased_in_forced_eval_group"] == 0
