"""Guards for the render-mode (mining) corpus rebuild.

The corpus this replaces was lost because its labels lived in one file and their meaning
lived in another, untracked one. So the properties worth guarding are the ones that make a
row SELF-SUFFICIENT and the draw REPRODUCIBLE — not that the builder ran.

Every derived-set assertion here is paired with a non-vacuity assertion, and the two
structural claims (the split's union-find, the roster/recipe coverage) are each paired with a
CONTROL that fails on the unfixed input, so "the guard works" is distinguishable from "this
population would have passed anyway" (`verification_practice.md` §3, §6, §9).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.mining import build_mining_sheet as BMS      # noqa: E402
from tools.mining import mining_roster as MR            # noqa: E402
from tools.mining import split_units as SU              # noqa: E402
from tools.mining import suggest_tier_mining as ST      # noqa: E402
from tools.wallpaper import merge_sitting as MS         # noqa: E402

SPEC = BMS.SHEETS["v1"]


# =========================================================================== #
# The roster and its recipes.
# =========================================================================== #
def test_every_roster_mode_has_a_recipe_and_every_recipe_a_roster_entry():
    assert MR.missing_recipes() == []
    assert len(MR.MODES) == 15, MR.MODES          # non-vacuity: the set is not empty/short


def test_the_coverage_check_CATCHES_a_mode_with_no_recipe(monkeypatch):
    """The control. A one-way or absent check passes on exactly the failure that matters —
    a mode added to the roster and not to the recipe table renders nothing."""
    monkeypatch.setattr(MR, "MODES", MR.MODES + ("invented_mode",))
    assert any("invented_mode" in m for m in MR.missing_recipes())


def test_the_corpus_keeps_the_modes_the_TRAINER_drops():
    """The whole point of the 15-mode decision: a trainer re-runs, a corpus does not
    un-narrow. If a future edit trims the roster to the trainer's set, the corpus can no
    longer be used to re-examine the drop."""
    for mode in MR.TRAINER_DROPPED_V1 + MR.SCALE_SAMPLER_DROPPED_2026_07:
        assert mode in MR.MODES, f"{mode} was dropped downstream, not upstream"


def test_an_unknown_mode_RAISES_instead_of_rendering_a_default():
    """A typo'd mode that silently rendered `smooth` would put a mislabeled class in the
    corpus — the label IS the mode here."""
    with pytest.raises(KeyError):
        MR.spec_for("smooth")                      # deliberately not on the roster
    assert MR.spec_for("tia")["field"] == "tia"    # non-vacuity: the happy path works


def test_rolloff_is_gated_to_the_screen_family_alone():
    on = [m for m in MR.MODES if MR.rolloff_token(m) != "none"]
    assert on == ["direct_trap_screen"], on
    assert MR.rolloff_token("direct_trap_screen") == "soft_knee@0.35"


# =========================================================================== #
# The split. The property is about UNITS, so the fixture is the real population.
# =========================================================================== #
@pytest.fixture(scope="module")
def population():
    if not BMS.GATE_PASSERS.exists():
        # NOT a skip. The artifact is tracked and count-verified; its absence is the exact
        # condition these guards exist for (`verification_practice.md` §2).
        pytest.fail(f"{BMS.GATE_PASSERS.relative_to(ROOT)} is absent — rebuild with "
                    f"`uv run python tools/mining/build_gate_passers.py`")
    rows, meta = BMS.load_gate_passers()
    rep = {}
    for r in rows:
        rep.setdefault(r["location_key"], r)
    return rows, meta, rep


def test_the_gate_passer_artifact_still_matches_its_own_recorded_census(population):
    """Relational, not a frozen literal: the artifact carries the census it was verified
    against, so this compares the file with its own claim rather than with a number here
    that would need re-baselining."""
    rows, meta, rep = population
    v = meta["verified_against_census"]
    assert len(rows) == v["expected_rows"] == v["realized_rows"]
    assert len(rep) == v["expected_locations"] == v["realized_locations"]
    assert all(r["p_ge3"] > meta["gate"]["threshold"] for r in rows)


def test_a_julia_child_and_its_parent_plane_location_land_on_ONE_side(population):
    _rows, _meta, rep = population
    side, meta = SU.build_split(rep, seed=SPEC.split_seed, eval_frac=SPEC.eval_frac)
    ok, why = SU.units_are_disjoint(side, rep)
    assert ok, why
    # Non-vacuity: the union-find must actually have merged something, or "each unit is
    # wholly on one side" is a statement about singletons and proves nothing.
    assert meta["n_units"] < meta["n_locations"], meta
    assert meta["n_multi_loc_units"] >= 1 and meta["linked_base_parents"] >= 1, meta


def test_a_NAIVE_per_location_split_SEPARATES_a_unit_on_this_same_population(population):
    """The control on the fixture. If a plain family-stratified split of raw locations kept
    every unit together anyway, the test above would pass without the union-find and this
    population could not tell the two apart."""
    _rows, _meta, rep = population
    # the same seeded scheme, but over locations rather than components
    rng = np.random.default_rng(SPEC.split_seed)
    strata = {}
    for k, r in sorted(rep.items()):
        strata.setdefault(r["family"], []).append(k)
    naive = {}
    for fam in sorted(strata):
        ks = sorted(strata[fam])
        order = rng.permutation(len(ks))
        ev = set(order[:int(round(SPEC.eval_frac * len(ks)))].tolist())
        for i, k in enumerate(ks):
            naive[k] = "eval" if i in ev else "train"
    ok, why = SU.units_are_disjoint(naive, rep)
    assert not ok, ("the naive split kept every unit together on this population, so it "
                    f"cannot distinguish the two rules here: {why}")


def test_a_julia_row_with_no_seed_is_REFUSED_not_left_a_singleton():
    """Silently leaving it unlinked is the leak this module exists to stop."""
    rep = {"j1": {"family": "julia",
                  "render": {"cx": "0", "cy": "0", "fw": "1", "c_re": None, "c_im": None}}}
    with pytest.raises(ValueError, match="parent point"):
        SU.build_split(rep)


def test_coordinates_that_differ_only_in_FORMATTING_stay_in_one_unit():
    """`"0.25" != "0.250"` would split a unit in half; the population reaches this function
    with decimal strings from one row and floats from another."""
    rep = {
        "m": {"family": "mandelbrot",
              "render": {"cx": "0.250", "cy": "-0.5000", "fw": "1", "c_re": None, "c_im": None}},
        "j": {"family": "julia",
              "render": {"cx": "0", "cy": "0", "fw": "1", "c_re": 0.25, "c_im": -0.5}},
    }
    side, meta = SU.build_split(rep)
    assert meta["n_units"] == 1 and meta["linked_base_parents"] == 1, meta
    assert side["m"] == side["j"]


def test_no_second_copy_of_the_split_rule_exists():
    """The July sampler's copy was deleted and repointed at this owner; a third would be two
    split designs that are supposed to be one."""
    owner = "tools/mining/split_units.py"
    exempt = {owner, "tools/mining/test_mining_sheet.py"}
    pat = re.compile(r"seed::|def\s+build_split\s*\(")
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    tracked = [p for p in out.stdout.splitlines() if p.strip()]
    assert tracked, "git ls-files returned nothing — the scan would be vacuous"
    offenders = [rel for rel in tracked
                 if rel.replace("\\", "/") not in exempt
                 and pat.search((ROOT / rel).read_text(encoding="utf-8", errors="replace"))]
    assert not offenders, (f"split rule re-declared outside {owner}: {offenders} — import "
                          f"`tools.mining.split_units.build_split` instead")


def test_the_split_scan_would_actually_catch_a_copy():
    pat = re.compile(r"seed::|def\s+build_split\s*\(")
    assert pat.search('node = f"seed::{pfam}::{a}::{b}"')
    assert pat.search("def build_split(locs):")
    assert not pat.search("from tools.mining.split_units import build_split")


# =========================================================================== #
# The pre-label rule.
# =========================================================================== #
def test_the_suggestion_histogram_REPRODUCES_the_prior_it_was_matched_to():
    prior = ST.tier_prior()
    rng = np.random.default_rng(3)
    pred = rng.uniform(1.0, 3.0, 2000)
    cuts = ST.cuts_from_prior(pred, prior)
    hist = Counter(ST.suggest_all(pred, cuts))
    for t in range(1, ST.K_TIERS + 1):
        got = hist.get(t, 0) / len(pred)
        want = prior["shares"][str(t)]
        assert abs(got - want) < 0.02, (t, got, want, hist)
    # non-vacuity: the cuts must be interior, i.e. the rule is not collapsing onto one class
    assert len(set(hist)) == ST.K_TIERS, hist
    assert cuts == tuple(sorted(cuts)) and len(cuts) == ST.K_TIERS - 1


def test_the_prior_is_DERIVED_from_the_files_and_an_absent_file_is_a_HARD_failure():
    p = ST.tier_prior()
    assert p["n"] == sum(p["counts"].values()) == 1500, p
    assert set(p["per_file"]) == set(ST.PRIOR_LABEL_FILES)
    with pytest.raises(SystemExit, match="absent"):
        ST.tier_prior(files=("labels/there_is_no_such_file.json",))


def test_the_rule_itself_is_IMPORTED_from_the_wallpaper_owner_not_restated():
    from tools.wallpaper import suggest_tier as WST
    assert ST.expected_tier is WST.expected_tier
    assert ST.tier_from_pred is WST.tier_from_pred
    assert ST.fit_cuts is WST.fit_cuts
    # and it is genuinely k-agnostic on this head's readout
    assert ST.expected_tier([0.5, 0.25]) == pytest.approx(1.75)


def test_cuts_from_an_empty_batch_RAISE_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="empty"):
        ST.cuts_from_prior([], ST.tier_prior())


# =========================================================================== #
# The draw.
# =========================================================================== #
@pytest.fixture(scope="module")
def planned(population):
    return BMS.plan(SPEC)


def test_the_plan_is_a_pure_function_of_the_artifact_and_the_spec(planned):
    """Recomputed by BOTH `render` and `write`, so a plan that drifted between the two would
    render one set of rows and describe another."""
    a, _ = planned
    b, _ = BMS.plan(SPEC)
    assert [x["image_id"] for x in a] == [x["image_id"] for x in b]
    assert [(x["mode"], x["location_key"], x["palette"], x["mode_params"]) for x in a] == \
           [(x["mode"], x["location_key"], x["palette"], x["mode_params"]) for x in b]


def test_every_mode_is_drawn_and_the_families_are_balanced_or_drained(planned):
    entries, rep = planned
    by_mode = Counter(e["mode"] for e in entries)
    assert set(by_mode) == set(MR.MODES), set(MR.MODES) ^ set(by_mode)
    assert min(by_mode.values()) >= 1
    for m in rep["allocation"]["per_mode"]:
        assert m["family_balanced"], (m["mode"], m["family_balance"])


def test_the_draw_uses_every_available_location(planned):
    """A draw that quietly used a third of the supply would look identical in the row count
    and give the head far less geometric variety."""
    entries, rep = planned
    assert rep["allocation"]["distinct_locations_used"] == rep["population"]["locations"]
    assert len({e["location_key"] for e in entries}) == rep["population"]["locations"]


def test_within_a_mode_every_location_is_distinct(planned):
    entries, _ = planned
    for mode in MR.MODES:
        ks = [e["location_key"] for e in entries if e["mode"] == mode]
        assert len(ks) == len(set(ks)), mode


def test_only_direct_modes_carry_grid_cells_and_they_span_the_whole_grid(planned):
    entries, _ = planned
    for e in entries:
        if e["kind"] == "direct":
            assert set(e["mode_params"]) == {"direct_opacity", "direct_threshold"}
            assert (e["mode_params"]["direct_opacity"],
                    e["mode_params"]["direct_threshold"]) in MR.DIRECT_GRID
        else:
            assert e["mode_params"] == {}, e["mode"]
    for mode in MR.DIRECT_MODES:
        cells = {(e["mode_params"]["direct_opacity"], e["mode_params"]["direct_threshold"])
                 for e in entries if e["mode"] == mode}
        assert cells == set(MR.DIRECT_GRID), (mode, sorted(cells))


def test_every_planned_row_carries_a_split_side_and_a_complete_colour_recipe(planned):
    """The self-sufficiency property — the one whose absence cost the original corpus."""
    entries, _ = planned
    need = {"reverse", "log_premap", "gamma", "phase", "n_cycles", "transfer", "transfer_gamma"}
    for e in entries:
        assert e["split_side"] in ("train", "eval")
        assert need <= set(e["color_params"]), (e["image_id"], sorted(need - set(e["color_params"])))
        for k in ("cx", "cy", "fw", "maxiter", "fractal_type"):
            assert e["render"].get(k) is not None or k == "fractal_type", (e["image_id"], k)


def test_image_ids_are_unique_and_stable_across_a_resume(planned):
    entries, _ = planned
    ids = [e["image_id"] for e in entries]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids), "ids are assigned in plan order; a resume must reproduce it"


# =========================================================================== #
# The written batch, when one exists.
# =========================================================================== #
@pytest.fixture(scope="module")
def written():
    p = SPEC.batch_dir / "images.jsonl"
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} not built yet — the render/write stages own it")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_sheet_order_is_contiguous_and_sorted_good_to_bad(written):
    """The addendum's contract: sorted on the CONTINUOUS score descending, stamped
    contiguous so the order is auditable later."""
    assert [r["sheet_order"] for r in written] == list(range(len(written)))
    preds = [r["pred"] for r in written]
    assert preds == sorted(preds, reverse=True), "sheet_order is not descending in pred"
    assert len(set(preds)) > 1, "a constant pred makes the order claim vacuous"


def test_the_BUILDER_emits_a_null_label_slot(written):
    """A suggestion is not a label. If `label.score` were pre-filled, the merge's null->value
    rule would accept the machine tier as human on the first pass.

    Asserted on the SOURCE, not on the store. This used to read `label.score is None` off the
    committed images.jsonl, which states the property only until the first merge — and
    null->value is the ONE mutation the store permits, so the guard fired on the sitting
    landing (2026-08-06) rather than on any defect. The build-time claim is about what the
    writer constructs; the store-side claim is the test below."""
    src = (ROOT / "tools" / "mining" / "build_mining_sheet.py").read_text(encoding="utf-8")
    assert re.search(r'"label":\s*\{\s*"score":\s*None,\s*"labeler":\s*None,\s*'
                     r'"labeled_at":\s*None\s*\}', src), \
        "the row constructor no longer emits an all-null label slot"
    for r in written:
        assert 1 <= r["suggested_tier"] <= ST.K_TIERS
        assert r["head_mining_v1"]["head_version"] == "v1"


def test_a_SCORED_row_carries_human_attribution_and_is_not_the_suggestion_copied(written):
    """The store-side half, and the one that survives a merge.

    Two ways a machine tier becomes a label: written into the slot at build time (the source
    check above), or merged in from an export that was really the suggestion column. The
    second is invisible per-row — a merged suggestion carries a labeler like any other — so
    it is caught in the aggregate: an export that WAS the suggestions agrees with them on
    every row. The real sitting agrees on 92.9%, so the check has margin and is not vacuous.
    """
    scored = [r for r in written if r["label"]["score"] is not None]
    if not scored:
        pytest.skip("batch not labeled yet — the null slot is covered by the source check")
    for r in scored:
        assert 1 <= r["label"]["score"] <= ST.K_TIERS, r["image_id"]
        assert r["label"]["labeler"] and r["label"]["labeled_at"], \
            f"{r['image_id']} has a score with no labeler/date — no merge path writes that"
    assert any(r["label"]["score"] != r["suggested_tier"] for r in scored), \
        "every scored row equals its suggested tier: the suggestion column was merged as labels"


def test_the_merge_REFUSES_a_tier_above_the_corpus_ceiling(tmp_path, written):
    assert MS.CORPORA["render_mode_corpus"] == ST.K_TIERS
    iid = written[0]["image_id"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({iid: 4}), encoding="utf-8")
    with pytest.raises(SystemExit, match="outside 1..3"):
        MS.merge(SPEC.batch_id, bad, apply=False, corpus="render_mode_corpus")
    # the control: the same call with an in-range tier goes through (dry run, writes nothing)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({iid: 3}), encoding="utf-8")
    rep = MS.merge(SPEC.batch_id, good, apply=False, corpus="render_mode_corpus")
    assert rep["exported"] == 1 and rep["applied"] is False


def test_the_merge_never_reads_the_suggestion(written):
    """`suggested_tier` must not reach the sidecar. Asserted on the SOURCE because the
    property is 'this field is not consulted', which no input can demonstrate."""
    src = (ROOT / "tools" / "wallpaper" / "merge_sitting.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]          # skip the module docstring, which names it
    assert "suggested_tier" not in body
