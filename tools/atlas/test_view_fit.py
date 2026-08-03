"""Acceptance for the fitted neighborhood score (`tools/atlas/view_fit.py`).

Three things are guarded and they fail differently:

  * the DERIVED MEASURES are behavioural — each bracket asserts a *relation* on a synthetic
    field (a field that descends outward scores a positive `falloff_rate`; a flat one does
    not; a flat one reports the frame radius rather than `falloff_extent`'s colliding
    `0.0`). Those survive a re-measurement (`verification_practice.md` §7).
  * the MODEL is an outcome, so it is pinned against `data/atlas/view_fit_v1.json` the way
    `test_view_screen` pins the gate record — including the reads that came back NULL
    (exemplar similarity), so a negative result stays reported rather than being tuned away.
  * the STAGING is a fact about the tree: nothing production imports this module, and
    `composite_v3` is still what every live sort calls. That assertion is the "BUILD ONLY,
    no flip" contract, and it is derived from the source rather than asserted in prose.

Run:  uv run python -m pytest tools/atlas/test_view_fit.py -q
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import view_fit as vf              # noqa: E402
import field_metrics as fm         # noqa: E402

W, H = fm.SCREEN_W, fm.SCREEN_H
CYCLE = 1.0 / fm.DENSITY
RECORD = ROOT / "data" / "atlas" / "view_fit_v1.json"


@pytest.fixture(scope="module")
def record() -> dict:
    # NOT absence-tolerant: the record is a tracked durable artifact, and a skip here would
    # un-guard the model exactly when it went missing (`verification_practice.md` §2).
    assert RECORD.exists(), (f"{RECORD} is missing — rebuild with "
                             f"`uv run python tools/atlas/view_fit.py fit`")
    return json.loads(RECORD.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the derived axis: falloff
# --------------------------------------------------------------------------- #
def _radial_field(f):
    """A field whose value is `f(r)` with `r` the shared plane radius, in ITERATIONS."""
    r = fm._plane_radius_grid(H, W)
    return (f(r) / fm.DENSITY).astype("f4")


def test_a_field_that_descends_outward_has_a_positive_falloff_rate():
    m = vf.falloff_features(_radial_field(lambda r: 8.0 - 12.0 * r))
    assert m["falloff_rate"] > 0.0
    # ...and the sign is the discrimination, not the magnitude: the same field flipped must
    # cross zero, or the statistic is measuring the offset instead of the slope.
    up = vf.falloff_features(_radial_field(lambda r: 1.0 + 12.0 * r))
    assert up["falloff_rate"] < 0.0 < m["falloff_rate"]


def test_a_flat_field_falls_off_at_neither_rate_nor_scale():
    m = vf.falloff_features(_radial_field(lambda r: np.full_like(r, 3.0)))
    assert abs(m["falloff_rate"]) < 1e-6


def test_a_field_with_no_descent_reports_the_frame_radius_not_zero():
    """The stated difference from the retired `falloff_extent`, which returns 0.0 both for
    an instant descent and for no descent at all and so cannot tell them apart."""
    flat = vf.falloff_features(_radial_field(lambda r: np.full_like(r, 3.0)))
    steep = vf.falloff_features(_radial_field(lambda r: np.where(r < 0.05, 40.0, 0.0)))
    assert flat["falloff_half"] > steep["falloff_half"] >= 0.0
    assert flat["falloff_half"] > 0.3, "a frame with no falloff must not report a tiny scale"


def test_a_tight_skin_has_a_smaller_half_scale_than_wide_decoration():
    tight = vf.falloff_features(_radial_field(lambda r: 20.0 * np.exp(-r / 0.05)))
    wide = vf.falloff_features(_radial_field(lambda r: 20.0 * np.exp(-r / 0.40)))
    assert tight["falloff_half"] < wide["falloff_half"]


def test_the_falloff_pass_survives_an_all_interior_field():
    m = vf.falloff_features(np.full((H, W), np.nan, dtype="f4"))
    assert m == dict(falloff_rate=0.0, falloff_half=0.0)


# --------------------------------------------------------------------------- #
# the fitted score
# --------------------------------------------------------------------------- #
def test_the_record_and_the_code_agree_on_the_feature_design(record):
    """A feature added to the code and not refitted would otherwise score silently under a
    stale coefficient vector."""
    assert tuple(record["models"]["primary"]["features"]) == vf.FEATURE_ORDER
    assert tuple(record["models"]["no_family"]["features"]) == vf.FEATURES_NO_FAMILY
    for name in ("primary", "no_family"):
        m = record["models"][name]
        n = len(m["features"])
        assert len(m["coef"]) == len(m["mean"]) == len(m["scale"]) == n
    # the no-family variant is the primary MINUS the family terms, and non-vacuously so
    assert set(vf.FEATURE_ORDER) - set(vf.FEATURES_NO_FAMILY) == set(vf.FAMILY_FEATURES)
    assert vf.FAMILY_FEATURES, "an empty family set would make the variant a duplicate"


def test_the_score_is_linear_in_its_features(record):
    """It is a linear score or it is not the thing that was fitted: a midpoint of two
    feature vectors must score the midpoint of their scores."""
    m = vf.FittedScore(record["models"]["primary"])
    rng = np.random.default_rng(7)
    a = {f: float(v) for f, v in zip(m.features, m.mean + rng.normal(size=len(m.features)))}
    b = {f: float(v) for f, v in zip(m.features, m.mean - rng.normal(size=len(m.features)))}
    mid = {f: 0.5 * (a[f] + b[f]) for f in m.features}
    assert m.score(mid) == pytest.approx(0.5 * (m.score(a) + m.score(b)), abs=1e-9)
    assert m.score(a) != pytest.approx(m.score(b)), "fixture too easy: the two tie"


def test_p_notbad_is_the_score_through_a_sigmoid(record):
    m = vf.FittedScore(record["models"]["primary"])
    x = {f: float(v) for f, v in zip(m.features, m.mean)}
    assert m.score(x) == pytest.approx(m.intercept, abs=1e-9)   # mean row == intercept
    assert m.p_notbad(x) == pytest.approx(1 / (1 + math.exp(-m.intercept)), abs=1e-9)


def test_a_missing_modelled_feature_takes_the_recorded_imputation(record):
    """`cap_headroom` is null exactly when a frame had no escaping pixel. The imputed value
    is the FIT's median, frozen in the record — not a zero invented at score time."""
    m = vf.FittedScore(record["models"]["primary"])
    x = {f: float(v) for f, v in zip(m.features, m.mean)}
    x["cap_headroom"] = float("nan")
    assert math.isfinite(m.score(x))
    filled = dict(x, cap_headroom=m.impute["cap_headroom"])
    assert m.score(x) == pytest.approx(m.score(filled), abs=1e-9)


def test_an_unmodelled_missing_value_raises_rather_than_scoring(record):
    m = vf.FittedScore(record["models"]["primary"])
    spec = dict(record["models"]["primary"], impute={})
    x = {f: float(v) for f, v in zip(m.features, m.mean)}
    x["band_coverage"] = float("nan")
    with pytest.raises(KeyError):
        vf.FittedScore(spec).score(x)


def test_an_arity_mismatch_is_refused_at_load(record):
    spec = dict(record["models"]["primary"])
    spec = dict(spec, coef=spec["coef"][:-1])
    with pytest.raises(ValueError):
        vf.FittedScore(spec)


# --------------------------------------------------------------------------- #
# the outcome, pinned to the record
# --------------------------------------------------------------------------- #
def test_the_population_the_fit_reads_is_the_one_it_claims(record):
    p = record["population"]
    assert (p["n"], p["fit"], p["uniform"], p["exemplar"]) == (730, 580, 90, 60)
    assert p["rule_rows"] == 81 and p["fit_positives"] == 149
    assert p["groups"] > 100, "grouped CV with a handful of groups is not grouped CV"


def test_the_fit_beats_composite_v3_on_the_fit_leg_and_the_interval_excludes_zero(record):
    r = record["readout"]
    assert record["models"]["primary"]["oof"]["ap"] > r["composite_v3_on_fit"]["ap"]
    assert r["ap_delta_fit_vs_composite"]["lo"] > 0.0


def test_the_rule_labeled_rows_are_not_what_the_fit_is_measuring(record):
    """The 81 rule rows are `interior_fraction > 0.30 -> 1`, and `interior_fraction` is a
    fitted feature — so the sensitivity line exists to say the result is not the rule read
    back. It is pinned because a refit that lost the property must go red."""
    s = record["readout"]["sensitivity_rule_rows_dropped"]
    assert s["dropped"] > 0
    assert s["oof"]["ap"] > s["composite_v3"]["ap"]
    both = (record["models"]["primary"]["coef"][vf.FEATURE_ORDER.index("interior_fraction")],
            s["coef"]["interior_fraction"])
    assert both[0] < 0 and both[1] < 0, "the interior term must survive the rule drop"


def test_the_exemplar_similarity_null_result_stays_reported(record):
    """Read (a) came back NULL — the coefficient CI straddles zero and dropping the two
    columns does not move AP. Pinned so it cannot quietly become a positive claim."""
    a = record["readout"]["exemplar_read_a"]
    lo, hi = a["coef_ci95"]["exemplar_sim_max"]
    assert lo < 0.0 < hi, "the CI no longer straddles zero — the null result changed"
    assert a["delta_ap"]["lo"] < 0.0 < a["delta_ap"]["hi"]


def test_the_uniform_leg_is_never_in_the_fit(record):
    """The one score-unconditioned draw. If it ever enters the fit its check is circular."""
    assert set(vf.FIT_LEGS) == {"strat_a", "strat_b"}
    assert record["models"]["primary"]["n"] == record["population"]["fit"] == 580
    assert record["readout"]["uniform_leg"]["fitted"]["n"] == 90


# --------------------------------------------------------------------------- #
# BUILD ONLY, no flip
# --------------------------------------------------------------------------- #
LIVE_SORT_MODULES = ("view_screen.py", "maneuver_view_screen.py", "steered_frontier.py",
                     "view_frame_sweep.py", "view_screen_gate.py", "view_rescreen.py")

# RE-AIMED A SECOND TIME (harvest v2 §3, 2026-08-03), and for the same class of reason the
# comment below records for the first re-aiming.
#
# The contract has always been "the fitted score does not ORDER anything until its bar is
# read". The guard implemented that as "no live sort path IMPORTS it", which held only while
# nobody needed the score for anything else. Harvest v2 needs exactly that: `view_fit_v1.1`
# recorded as a COLUMN on every screened row, because the pre-registered bar reads at a
# SITTING's labels and the q4 sitting could not read it — neither score existed on any row,
# so NOT-ADOPT was the absence of evidence rather than a measured loss. Recording requires
# importing, so an import ban now forbids the one thing that would let the bar be read, while
# still not testing the thing it cares about.
#
# So the modules split by DUTY:
#   * IMPORT_BANNED — pure sort / gate / sweep paths with no recording duty. The import ban
#     stays exactly as it was, and it is still a real tripwire.
#   * RECORDING — the two modules that must write the column. They may import; what they may
#     NOT do is let it order anything, and that is asserted FUNCTIONALLY below (a row with a
#     huge `view_fit` and a poor `composite` must still sort last) rather than by grep.
RECORDING_MODULES = ("maneuver_view_screen.py", "steered_frontier.py")
IMPORT_BANNED_MODULES = tuple(m for m in LIVE_SORT_MODULES if m not in RECORDING_MODULES)


# The contract is "no live sort path IMPORTS the fitted score", and the guard used to test
# it with `"view_fit" not in source`. That is a guard pinned to PROSE, and it went red on
# 2026-08-03 for a module that merely NAMED view_fit in a pre-registration record — the
# written-down statement of the very bar the score is staged against. A guard that a
# correctly-written pre-registration cannot coexist with is aimed at the wrong thing
# (`verification_practice.md` §6), so it is re-aimed at the routing decision: any import
# FORM, plus the positive assertion that `composite_v3` is still the key. It stays red for a
# real adoption — proved by injection in the test below — and no longer fires on a mention.
_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+(?:[\w.]+\.)?view_fit\b"          # import view_fit / tools.atlas.view_fit
    r"|from\s+(?:[\w.]+\s+import\s+[^\n]*\bview_fit\b"  # from tools.atlas import view_fit
    r"|[\w.]*view_fit\s+import\b))",                    # from view_fit import ...
    re.MULTILINE)
_SPEC_LOAD_RE = re.compile(r"spec_from_file_location\([^)]*view_fit", re.DOTALL)


def imports_view_fit(src: str) -> bool:
    """True iff `src` actually pulls the fitted score in, by any of the three forms this
    tree uses (dotted import, bare import, `spec_from_file_location` path load)."""
    return bool(_IMPORT_RE.search(src) or _SPEC_LOAD_RE.search(src))


def test_no_pure_sort_path_imports_the_fitted_score():
    """The staging contract for every module with no recording duty. `composite_v3` stays
    the live sort key until an adoption decision is made against its own pre-registered bar;
    this is what makes that a fact about the tree rather than a sentence in a module doc."""
    seen = 0
    for name in IMPORT_BANNED_MODULES:
        p = HERE / name
        assert p.exists(), f"{name} moved — this guard must be re-aimed, not deleted"
        seen += 1
        assert not imports_view_fit(p.read_text(encoding="utf-8")), (
            f"{name} IMPORTS view_fit — the fitted score is staged, not adopted")
    assert seen == len(IMPORT_BANNED_MODULES) >= 4


def test_the_recording_modules_record_the_column_and_do_not_order_by_it():
    """The two modules that MAY import it. Non-vacuous in both directions: the column must
    actually be produced (a "staged" score nobody records is the state that made the bar
    unreadable), and the sort key must still be `composite`."""
    import maneuver_view_screen as mvs
    src = (HERE / "maneuver_view_screen.py").read_text(encoding="utf-8")
    assert imports_view_fit(src), "the recording module must import the score it records"
    assert "view_fit" in mvs.STATE_KEYS, "the column must survive a checkpoint"
    sort_src = inspect.getsource(mvs.composite_sort_key)
    assert "composite" in sort_src and "view_fit" not in sort_src


def test_a_high_view_fit_cannot_outrank_a_high_composite():
    """The FUNCTIONAL half, and the one that would survive a refactor a grep would not:
    two screened rows, the first with a far better `view_fit` and a far worse `composite`.
    `composite_sort_key` must still put the second one on top. If the fitted score were ever
    quietly promoted to the key, this inverts."""
    import maneuver_view_screen as mvs
    fit_wins = dict(screened=True, composite=0.10, view_fit=99.0)
    comp_wins = dict(screened=True, composite=0.90, view_fit=-99.0)
    assert mvs.composite_sort_key(comp_wins) > mvs.composite_sort_key(fit_wins)
    # ... and an unscreened row still sorts below both, which is the older contract this
    # must not have broken on the way past.
    assert mvs.composite_sort_key(dict(screened=False)) < mvs.composite_sort_key(fit_wins)


def test_the_import_guard_is_red_for_a_real_adoption():
    """Prove it red on purpose: a carried red is a guard that is off, and a guard loosened
    from a substring match to a regex has to show it still catches the thing it exists for."""
    for adoption in ("import view_fit\n",
                     "import view_fit as vf\n",
                     "from tools.atlas import view_fit\n",
                     "from view_fit import load_model_v11\n",
                     "    import tools.atlas.view_fit\n",
                     'spec_from_file_location("view_fit", p)\n'):
        assert imports_view_fit(adoption), adoption
    # ... and green for a MENTION, which is what a pre-registration record is.
    for mention in ('"adopted only if view_fit beats composite_v3 by +0.1181"\n',
                    "# view_fit is staged, not adopted\n"):
        assert not imports_view_fit(mention), mention
