"""The bucketed wallpaper sitting's DRAW, pinned on synthetic screen records.

WHY SYNTHETIC. The draw is the only part of this builder that decides anything, and it is a
pure function of (screen records, spec). Pinning it against the live screen log would tie the
test to a scratch artifact that a `rm -r scratch/*` deletes, and would say nothing about the
cases that matter — a drained bucket, a partition below the coverage floor, a take-all
population larger than the target. Those are constructed here.

The two things that are NOT synthetic are the frozen suggestion cuts (re-derived from the
tracked batches + sidecars they were fitted on) and the batch registration.

  uv run pytest tools/wallpaper/test_wallpaper_sitting.py -q
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring", ROOT / "tools" / "corpus",
          ROOT / "tools" / "queries", ROOT / "tools" / "wallpaper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import batch_registry as br                              # noqa: E402
import partitions as P                                   # noqa: E402
from tools.emission import floors as F                   # noqa: E402
from tools.wallpaper import build_wallpaper_sitting as BS  # noqa: E402
from tools.wallpaper import suggest_tier as ST           # noqa: E402

SPEC = BS.SITTINGS["v2"]


def rec(i, partition, score, vein="descent", err=None):
    """One screen record. `score` is the location's best candidate's p_ge3."""
    return {
        "unit_key": f"u{i:05d}", "key": f"k{i:05d}", "family": "mandelbrot",
        "partition": partition, "vein": vein, "source_tag": vein, "floor_admit": False,
        "fw": "1e-3", "maxiter": 4000, "error": err,
        "candidates": [{"palette": f"p{j}", "palette_type": "cyclic", "palette_source": "x",
                        "config": {}, "p_ge2": 1.0, "p_ge3": score * (1.0 - 0.1 * j),
                        "p_ge4": 0.0, "pred": 1.0 + score}
                       for j in range(3)],
    }


def population(n_per_part=200, classic=39, maneuver_frac=0.25):
    """A synthetic screened population with a controlled partition and vein shape."""
    out, i = [], 0
    for part in P.ALL_FAMS:
        if part == P.CLASSIC_PHOENIX:
            continue
        for j in range(n_per_part):
            vein = "maneuver" if j % int(1 / maneuver_frac) == 0 else "descent"
            # scores spread over [0, 1) so every band is populated
            out.append(rec(i, part, (j % 100) / 100.0, vein))
            i += 1
    for j in range(classic):
        out.append(rec(i, P.CLASSIC_PHOENIX, (j % 100) / 100.0))
        i += 1
    return out


# =========================================================================== #
# registration
# =========================================================================== #
def test_the_sitting_is_registered_train_side_before_it_is_built():
    """The prompt's own precondition, and `sitting_cutter.check_registrations`' rule applied
    to a corpus that has no such check: an unregistered batch classifies biased/train by the
    fail-closed default, which is indistinguishable from "nobody thought about it"."""
    assert br.is_registered(SPEC.batch_id)
    split, biased, source = br.assign_split(SPEC.batch_id, "mandelbrot")
    assert (split, biased) == ("train", True)
    assert source == "wallpaper_correction_sitting"
    assert not br.lookup(SPEC.batch_id, "mandelbrot").eval_eligible


# =========================================================================== #
# the suggestion cuts
# =========================================================================== #
@pytest.mark.stage2_pinned
def test_the_frozen_intake_cuts_reproduce_from_the_slice_they_were_fitted_on():
    """Freeze in records, DERIVE IN CODE. The constant is the record; `derive_intake_cuts` is
    the derivation, and this asserts they have not drifted apart."""
    assert ST.derive_intake_cuts() == ST.INTAKE_CUTS


@pytest.mark.stage2_pinned
def test_the_intake_cuts_reproduce_the_intake_prior_and_the_dramatic_cuts_do_not():
    """The whole reason for a second cut set: prior-matching is the objective, and only the
    fitted cuts achieve it ON THIS POPULATION. Non-vacuous — it also asserts the two rules
    genuinely disagree, so a copy-paste of CUTS into INTAKE_CUTS goes red.

    The tolerance is 2 rows of 1,140 and it is NOT slack: the cuts are frozen at 4 decimal
    places and two rows sit exactly on the tier-1/2 quantile, so the frozen rule reproduces the
    prior up to its own rounding. `INTAKE_DERIVATION` records the realized histogram, not the
    exact fit, for the same reason."""
    pred, tiers = ST.intake_slice()
    true_hist = Counter(tiers)
    fitted = Counter(ST.tier_from_pred(p, ST.INTAKE_CUTS) for p in pred)
    dramatic = Counter(ST.tier_from_pred(p, ST.CUTS) for p in pred)
    for t in range(1, ST.K_TIERS + 1):
        assert abs(fitted[t] - true_hist[t]) <= 2, (t, fitted, true_hist)
        assert fitted[t] == ST.INTAKE_DERIVATION["accuracy_on_slice"]["suggested_hist"][str(t)]
    assert max(abs(dramatic[t] - true_hist[t]) for t in range(1, ST.K_TIERS + 1)) > 2
    assert ST.INTAKE_CUTS != ST.CUTS


def test_the_cuts_are_absolute_not_re_quantiled(monkeypatch):
    """A deliberately over-drawn population must come out with MORE tier-3/4 suggestions, which
    is the property a per-batch quantile destroys and the reason these are frozen."""
    good = [3.6] * 100
    bad = [1.2] * 100
    assert all(ST.tier_from_pred(p, ST.INTAKE_CUTS) == 4 for p in good)
    assert all(ST.tier_from_pred(p, ST.INTAKE_CUTS) <= 2 for p in bad)


# =========================================================================== #
# the draw
# =========================================================================== #
@pytest.fixture(scope="module")
def drawn():
    pop = population()
    return BS.select(SPEC, pop) + (pop,)


def test_the_draw_hits_the_target_and_every_row_carries_its_bucket(drawn):
    sel, rep, _pop = drawn
    assert rep["drawn_rows"] == rep["target_rows"] == SPEC.target_rows
    assert len(sel) == SPEC.target_rows
    assert all(r["bucket"] in BS.BUCKET_ORDER for r in sel)
    assert len({r["unit_key"] for r in sel}) == len(sel), "a location was claimed twice"


def test_phoenix_classic_is_taken_whole(drawn):
    """The bucket exists because every OTHER rule skips classic by construction: 39 of 2,867
    is below every proportional share and below the coverage floor."""
    sel, rep, pop = drawn
    supply = sum(1 for r in pop if r["partition"] == P.CLASSIC_PHOENIX)
    got = sum(1 for r in sel if r["partition"] == P.CLASSIC_PHOENIX)
    assert supply > 0 and got == supply
    assert all(r["bucket"] == "phoenix_classic" for r in sel
               if r["partition"] == P.CLASSIC_PHOENIX)


def test_the_minibrot_bucket_draws_only_from_its_veins_and_hits_its_target(drawn):
    sel, rep, _pop = drawn
    mb = [r for r in sel if r["bucket"] == "minibrot_maneuver"]
    assert len(mb) == SPEC.minibrot_target
    assert {r["vein"] for r in mb} <= BS.MINIBROT_VEINS


def test_the_below_floor_bucket_is_exactly_the_restructure_delta_band(drawn):
    sel, _rep, _pop = drawn
    band = [r for r in sel if r["bucket"] == "below_retired_floor"]
    assert len(band) == SPEC.below_floor_target
    for r in band:
        assert F.GOOD_FLOOR <= r["score"] < F.WALLPAPER_RELEASE.value


def test_the_top_slice_is_the_top_of_what_it_could_still_see(drawn):
    """It claims AFTER the three buckets above it, so it is the top of the REMAINDER — which
    is the intended reading of a claim order, and is asserted rather than assumed."""
    sel, _rep, _pop = drawn
    top = sorted((r for r in sel if r["bucket"] == "top_slice"),
                 key=lambda r: -r["score"])
    assert len(top) == SPEC.top_slice_target
    earlier = {"phoenix_classic", "minibrot_maneuver", "below_retired_floor"}
    later = [r["score"] for r in sel if r["bucket"] not in earlier | {"top_slice"}]
    assert min(r["score"] for r in top) >= max(later), \
        "a later bucket claimed a row the top slice should have had"


def test_the_coverage_floor_is_a_floor_not_a_bonus(drawn):
    """`max(natural, reserved)`, not `natural + reserved` — the distinction `deal_round_robin`'s
    `preseed` paragraph and `partition_slots`' guarantee fixed point are both about."""
    sel, rep, pop = drawn
    got = Counter(r["partition"] for r in sel)
    supply = Counter(r["partition"] for r in pop)
    for part, n in supply.items():
        assert got[part] >= min(SPEC.coverage_floor, n), \
            f"{part}: {got[part]} rows against supply {n} and floor {SPEC.coverage_floor}"
    # ...and a partition the draw would have served generously anyway gains nothing from it:
    # the floor bucket only ever tops SHORT partitions up.
    topup = rep["per_bucket"][4]["partition_topup"]
    for part, added in topup.items():
        if added:
            assert got[part] == SPEC.coverage_floor or any(
                r["bucket"] == "remainder" and r["partition"] == part for r in sel)


def test_a_drained_bucket_records_its_shortfall_instead_of_borrowing(drawn):
    """"Strata take what supply allows" — asserted on a population with almost no maneuver
    supply, where the bucket must under-fill and SAY so rather than pulling in descent rows."""
    pop = [r for r in population() if r["vein"] != "maneuver"][:1200]
    for r in pop[:10]:
        r["vein"] = "maneuver"
    sel, rep, = BS.select(SPEC, pop)
    mb = rep["per_bucket"][1]
    assert mb["bucket"] == "minibrot_maneuver"
    assert mb["available"] == 10 and mb["drawn"] == 10 < SPEC.minibrot_target
    assert {r["vein"] for r in sel if r["bucket"] == "minibrot_maneuver"} == {"maneuver"}


def test_a_population_smaller_than_the_target_yields_a_short_sitting_not_a_repeat(drawn):
    pop = population(n_per_part=20, classic=5)
    sel, rep = BS.select(SPEC, pop)
    assert rep["drawn_rows"] == len(pop) == len(set(r["unit_key"] for r in sel))
    assert rep["drawn_rows"] < SPEC.target_rows


def test_screen_failures_never_reach_the_draw():
    pop = population(n_per_part=60, classic=5)
    for r in pop[:50]:
        r["error"] = "RuntimeError: dump-field failed"
    sel, rep = BS.select(SPEC, pop)
    assert rep["screen_failures"] == 50
    assert all(r["error"] is None for r in sel)


def test_the_draw_is_a_pure_function_of_population_and_seed():
    # The population MUST exceed the target, or the seed has nothing to choose between and
    # the second half of this test is vacuous (it was, at n_per_part=40).
    pop = population(n_per_part=200, classic=39)
    assert len(pop) > SPEC.target_rows
    a, _ = BS.select(SPEC, [dict(r) for r in pop])
    b, _ = BS.select(SPEC, [dict(r) for r in pop])
    assert [r["unit_key"] for r in a] == [r["unit_key"] for r in b]
    c, _ = BS.select(SPEC, [dict(r) for r in pop], seed=SPEC.seed + 1)
    assert [r["unit_key"] for r in c] != [r["unit_key"] for r in a]


def test_the_split_is_bucket_stratified_and_one_row_per_location(drawn):
    sel, _rep, _pop = drawn
    sides, n_eval = BS.assign_split(SPEC, sel)
    assert set(sides.values()) <= {"train", "eval"}
    assert len(sides) == len(sel)
    for bucket in BS.BUCKET_ORDER:
        keys = [r["unit_key"] for r in sel if r["bucket"] == bucket]
        if not keys:
            continue
        got = sum(1 for k in keys if sides[k] == "eval")
        assert got == int(round(SPEC.eval_frac * len(keys)))


# =========================================================================== #
# presentation contract
# =========================================================================== #
def test_the_opaque_crop_stem_is_stable_and_says_nothing_about_the_row():
    """The stem is what the crop is NAMED, and the served id is `<prefix><slot>_<stem[:8]>`.
    Neither may encode the bucket, partition or vein — an id that leaks the stratum re-creates
    the anchoring the opaque convention exists to remove."""
    a = BS.crop_stem(SPEC, "u00001")
    assert a == BS.crop_stem(SPEC, "u00001")
    assert a != BS.crop_stem(SPEC, "u00002")
    other = BS.SittingSpec(**{**SPEC.__dict__, "id_salt": "different"})
    assert BS.crop_stem(other, "u00001") != a
    for token in ("mandelbrot", "phoenix", "maneuver", "top_slice", "remainder"):
        assert token not in a


def test_the_veins_are_derived_from_the_row_not_stored():
    assert BS.vein_of({"source_tag": "maneuver:snap_to_nucleus:k=8.0"}) == "maneuver"
    assert BS.vein_of({"source_tag": "triggered:neighborhood_expand:k=16.0"}) == "maneuver"
    assert BS.vein_of({"source_tag": "q4_harvest"}) == "q4_harvest"
    assert BS.vein_of({"source_tag": "steered"}) == "descent"
    assert BS.vein_of({"source_tag": "dive"}) == "dive"
    assert BS.vein_of({}) == "descent"
    assert BS.MINIBROT_VEINS == {"maneuver", "q4_harvest"}


def test_the_screen_and_label_field_stems_can_never_collide():
    """A screen field is a SCORING-ONLY proxy at a smaller geometry. Serving one where a label
    field is meant would put a 512x288 render in the corpus."""
    import location as loc_mod
    loc = loc_mod.Location(family="mandelbrot", cx="0", cy="0", fw="1", maxiter=1000)
    screen = BS.screen_field_stem(loc)
    label = f"{loc.family}_x_{BS.LABEL_W}x{BS.LABEL_H}ss{BS.LABEL_SS}"
    assert screen.endswith(f"{BS.SCREEN_W}x{BS.SCREEN_H}ss{BS.SCREEN_SS}")
    assert not screen.endswith(f"{BS.LABEL_W}x{BS.LABEL_H}ss{BS.LABEL_SS}")
    assert label.split("_")[-1] != screen.split("_")[-1]


def test_the_worker_cap_is_the_project_process_cap():
    with pytest.raises(SystemExit):
        BS.main(["select", "--workers", str(BS.WORKERS + 1)])
