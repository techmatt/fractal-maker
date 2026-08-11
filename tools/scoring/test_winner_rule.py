"""The winner rule's OTHER branches — the ones the real outcome will not exercise.

A rule only ever evaluated on one outcome is a rule whose failure paths have never run, and
this one decides whether a head is a candidate. Every clause combination is constructed here,
plus the three refusals the module makes explicit: an unmeasurable cell votes neither way,
multiplicity is counted, and the two readings of clause (a) are both produced.

Also: a source scan, because the rule had a second copy once already
(`mining_v2_reads.apply_winner_rule`, written for the K=3 head) and a second copy is two
rules that are supposed to agree.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scoring.winner_rule import (  # noqa: E402
    Metric, paired_bootstrap, point_block, value, verdict)


def ci(lo, hi, n_draws=4000):
    return {"n_draws": n_draws, "lo": lo, "hi": hi, "median": (lo + hi) / 2,
            "significantly_worse": hi < 0.0, "significantly_better": lo > 0.0}


NEUTRAL = ci(-0.02, 0.03)
BETTER = ci(0.01, 0.06)
WORSE = ci(-0.09, -0.01)
DEAD = {"n_draws": 0, "lo": None, "hi": None, "median": None,
        "significantly_worse": None, "significantly_better": None}


def run(no_worse, motivating, pooled="pooled"):
    return verdict(no_worse, motivating, pooled_arm=pooled, baseline="old", candidate="new")


def test_both_clauses_pass_gives_the_candidate():
    v = run({"pooled": {"auc3": NEUTRAL}, "slice": {"auc3": BETTER}},
            {"motiv": {"auc3": BETTER, "ap3": NEUTRAL}})
    assert v["winner"] == "new" and v["clause_a"]["pass"] and v["clause_b"]["pass"]


def test_a_regression_anywhere_in_the_no_worse_set_loses_it():
    v = run({"pooled": {"auc3": NEUTRAL}, "slice": {"auc3": WORSE}},
            {"motiv": {"auc3": BETTER}})
    assert v["winner"] == "old"
    assert [f["arm"] for f in v["clause_a"]["failures"]] == ["slice"]
    # ...but the pooled-only reading still passes, and both are reported.
    assert v["winner_pooled_only"] == "new"


def test_no_improvement_on_the_motivating_arm_loses_it():
    v = run({"pooled": {"auc3": NEUTRAL}}, {"motiv": {"auc3": NEUTRAL, "ap3": NEUTRAL}})
    assert v["winner"] == "old" and not v["clause_b"]["pass"]
    assert v["clause_a"]["pass"], "clause (a) must be independent of clause (b)"


def test_an_improvement_beside_a_regression_on_the_motivating_arm_loses_it():
    v = run({"pooled": {"auc3": NEUTRAL}}, {"motiv": {"auc3": BETTER, "ap3": WORSE}})
    assert v["winner"] == "old" and not v["clause_b"]["pass"]
    assert v["clause_b"]["improvements"] and v["clause_b"]["regressions"]


def test_an_unmeasurable_cell_votes_neither_way():
    """A dead boundary must not pass clause (a) as 'not worse' nor clause (b) as 'better'."""
    v = run({"pooled": {"auc3": NEUTRAL}, "dead": {"auc4": DEAD}},
            {"motiv": {"auc3": BETTER, "auc4": DEAD}})
    assert v["clause_a"]["unmeasurable"] == ["dead.auc4"]
    assert v["clause_a"]["n_tests"] == 1, "the dead cell must not be counted as a test"
    assert v["clause_b"]["unmeasurable"] == ["motiv.auc4"]
    assert v["winner"] == "new"


def test_an_entirely_unmeasurable_motivating_arm_cannot_win():
    v = run({"pooled": {"auc3": NEUTRAL}}, {"motiv": {"auc3": DEAD}})
    assert v["winner"] == "old" and v["clause_b"]["n_tests"] == 0


def test_multiplicity_is_counted():
    v = run({"pooled": {"auc3": NEUTRAL, "ap3": NEUTRAL},
             "a": {"auc3": NEUTRAL}, "b": {"auc3": NEUTRAL}},
            {"motiv": {"auc3": BETTER}})
    assert v["clause_a"]["n_tests"] == 4
    assert "4 arm x metric cells" in v["multiplicity_note"]


def test_pooled_arm_must_be_declared():
    with pytest.raises(KeyError):
        run({"slice": {"auc3": NEUTRAL}}, {"motiv": {"auc3": BETTER}}, pooled="pooled")


# --------------------------------------------------------------------------- #
# the statistics
# --------------------------------------------------------------------------- #
M3 = Metric("auc3", "AUC>=3", "p_ge3", 3, "auc")
A3 = Metric("ap3", "AP>=3", "p_ge3", 3, "ap")


def test_auc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(5):
        lb = rng.integers(1, 4, 200)
        s = np.round(rng.random(200), 2)          # deliberate ties
        got = value(lb, {"p_ge3": s}, M3)
        assert abs(got - roc_auc_score((lb >= 3).astype(int), s)) < 1e-12


def test_ap_matches_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    lb = rng.integers(1, 4, 300)
    s = rng.random(300)
    got = value(lb, {"p_ge3": s}, A3)
    assert abs(got - average_precision_score((lb >= 3).astype(int), s)) < 1e-9


def test_paired_bootstrap_finds_a_real_gap_and_not_a_zero_one():
    rng = np.random.default_rng(2)
    lb = rng.integers(1, 4, 400)
    y = (lb >= 3).astype(float)
    weak = rng.random(400)
    strong = 0.75 * y + 0.25 * rng.random(400)
    same = paired_bootstrap(lb, {"p_ge3": weak}, {"p_ge3": weak}, [M3], draws=400)
    assert not same["auc3"]["significantly_better"] and not same["auc3"]["significantly_worse"]
    gap = paired_bootstrap(lb, {"p_ge3": weak}, {"p_ge3": strong}, [M3], draws=400)
    assert gap["auc3"]["significantly_better"] and not gap["auc3"]["significantly_worse"]
    flip = paired_bootstrap(lb, {"p_ge3": strong}, {"p_ge3": weak}, [M3], draws=400)
    assert flip["auc3"]["significantly_worse"]


def test_degenerate_boundary_yields_zero_draws():
    lb = np.full(50, 3)                       # every row positive at >=3
    s = np.random.default_rng(3).random(50)
    out = paired_bootstrap(lb, {"p_ge3": s}, {"p_ge3": s}, [M3], draws=50)
    assert out["auc3"]["n_draws"] == 0


def test_point_block_carries_its_denominators():
    lb = np.array([1, 2, 3, 3, 2])
    b = point_block(lb, {"p_ge3": np.array([0.1, 0.2, 0.9, 0.8, 0.3])}, [M3])
    assert b["n"] == 5 and b["auc3__n_pos"] == 2 and b["auc3__n_neg"] == 3


# --------------------------------------------------------------------------- #
def test_no_second_copy_of_the_rule():
    """The rule has ONE implementation. `mining_v2_reads` keeps its own August copy as the
    record of the v1-vs-v2 sitting — that file is frozen evidence, so it is named as the
    single permitted exception rather than silently skipped."""
    allowed = {Path("tools/mining/mining_v2_reads.py"), Path("tools/scoring/winner_rule.py")}
    # The rule's signature is a file that DEFINES the significance flag (writes it as a
    # dict key) *and* structures clauses off it. Reading `ci["significantly_worse"]` and
    # printing a clause is what every legitimate CONSUMER does — the two reads harnesses do
    # exactly that — so the scan must separate producing the verdict from reporting it, or
    # it fires on its own callers. `sat_radius_calibrate` has an unrelated `verdict()` about
    # saturation radii and matches neither half.
    sig = re.compile(r'["\']significantly_worse["\']\s*:')
    clause = re.compile(r"clause_[ab]")
    hits = []
    for p in list((ROOT / "tools").rglob("*.py")) + list((ROOT / "classifier").rglob("*.py")):
        rel = Path(p).relative_to(ROOT)
        if rel in allowed or "__pycache__" in rel.parts or rel.name.startswith("test_"):
            continue
        src = p.read_text(encoding="utf-8", errors="ignore")
        if sig.search(src) and clause.search(src):
            hits.append(rel.as_posix())
    assert not hits, f"a second winner-rule implementation: {hits}"
