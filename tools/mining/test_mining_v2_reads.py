"""Guards for the v1-vs-v2 mining-head readout.

The readout's job is to pick a calibration candidate and then derive cuts on it, so the
properties worth guarding are the ones that would let it pick wrongly or overstate:

  * the WINNER RULE must have both branches exercised. Evaluated on one real outcome it is
    a rule whose other branch has never run, and the branch that never runs is the one that
    would have refused an adoption (§6 — a fixture that cannot fail).
  * `boundary` must report an unmeasurable cell as UNMEASURABLE, not as 0.5. Two of the
    three modes v1's trainer dropped have zero labeled tier-3 on the eval side, so this is
    the live path, not a hypothetical: a cell silently reported at chance would enter the
    per-mode table as a real number.
  * the paired bootstrap must find a real improvement AND must not manufacture one from
    identical scores (§3 — every derived assertion paired with a control).
  * the volume-matched comparison must take the SAME count from both heads; that is the
    entire reason it exists beside the fixed-threshold table.
  * a candidate cut must never be reported `supported` on a Wilson bound short of its target.
  * `load_eval_rows` must fail closed on an unlabeled row — a silently smaller n reads
    exactly like a complete eval side (§2).

The committed-batch tests are the non-vacuity half: they fail if the merge is un-applied or
the split stamp is lost, which are the two ways every number in the report stops describing
a complete, held-out slice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.mining import mining_v2_reads as MR                  # noqa: E402
from tools.mining.mining_roster import MODES, TRAINER_DROPPED_V1  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ci(lo, hi, n_draws=1000):
    return {"n_draws": n_draws, "lo": lo, "hi": hi, "median": (lo + hi) / 2,
            "significantly_worse": hi < 0.0, "significantly_better": lo > 0.0}


def all_ci(lo, hi, **over):
    out = {k: ci(lo, hi) for k, _ in MR.OVERALL_METRICS}
    out.update(over)
    return out


def scores(p3, p2=None):
    p3 = np.asarray(p3, dtype=float)
    p2 = p3 if p2 is None else np.asarray(p2, dtype=float)
    return {"p_ge3": p3, "p_ge2": p2, "rank": p2 + p3}


# =========================================================================== #
# The winner rule — both branches.
# =========================================================================== #
def test_v2_wins_only_when_it_holds_overall_AND_improves_the_dropped_modes():
    r = MR.apply_winner_rule(all_ci(-0.01, +0.02), all_ci(+0.05, +0.15))
    assert r["winner"] == "v2"
    assert r["clause_a_pass"] and r["clause_b_pass"]
    assert r["calibration_candidate_ckpt"] == MR.V2_CKPT


def test_one_significantly_worse_overall_metric_loses_it_for_v2():
    """Clause (a) is a veto, not a vote — a dropped-mode sweep cannot buy back an overall
    regression. Everything but AP>=3 is flat and the dropped modes improve on all four."""
    r = MR.apply_winner_rule(all_ci(-0.01, +0.02, ap_ge3=ci(-0.09, -0.02)),
                             all_ci(+0.05, +0.15))
    assert r["winner"] == "v1"
    assert r["clause_a_pass"] is False and r["clause_b_pass"] is True
    assert r["clause_a_holds"] == {"auc_ge3": True, "ap_ge3": False,
                                   "auc_ge2": True, "ap_ge2": True}
    assert r["calibration_candidate_ckpt"] == MR.V1_CKPT


def test_holding_overall_is_not_enough_the_dropped_modes_must_actually_improve():
    """The finetune's stated purpose is the three modes v1 never saw. A v2 that merely ties
    everywhere is not a reason to move; the rule says so and this is the branch that says it."""
    r = MR.apply_winner_rule(all_ci(-0.01, +0.02), all_ci(-0.04, +0.04))
    assert r["winner"] == "v1"
    assert r["clause_a_pass"] is True
    assert r["clause_b_improves_any"] is False and r["clause_b_pass"] is False


def test_an_improvement_beside_a_regression_inside_the_dropped_slice_still_loses():
    """No cherry-picking WITHIN clause (b) either: AUC>=3 up, AP>=2 down -> not adopted."""
    r = MR.apply_winner_rule(all_ci(-0.01, +0.02),
                             all_ci(+0.05, +0.15, ap_ge2=ci(-0.20, -0.05)))
    assert r["clause_b_improves_any"] is True and r["clause_b_worse_any"] is True
    assert r["clause_b_pass"] is False and r["winner"] == "v1"


def test_an_unmeasurable_dropped_boundary_contributes_neither_verdict():
    """`n_draws == 0` means the pooled slice could not measure that boundary at all. It must
    not count as an improvement (which would adopt on a cell nobody measured) and must not
    count as a regression (which would refuse on one)."""
    dead = {"n_draws": 0, "lo": None, "hi": None, "median": None,
            "significantly_worse": None, "significantly_better": None}
    r = MR.apply_winner_rule(all_ci(-0.01, +0.02),
                             all_ci(+0.05, +0.15, auc_ge2=dead, ap_ge2=dead))
    assert r["clause_b_measurable"] == ["auc_ge3", "ap_ge3"]
    assert r["winner"] == "v2"                    # decided by the two live boundaries

    only_dead = {k: dead for k, _ in MR.OVERALL_METRICS}
    r2 = MR.apply_winner_rule(all_ci(-0.01, +0.02), only_dead)
    assert r2["clause_b_measurable"] == [] and r2["winner"] == "v1"


# =========================================================================== #
# boundary — unmeasurable is not chance.
# =========================================================================== #
def test_a_cell_with_no_positives_is_unmeasurable_not_at_chance():
    """Live path: trap_circle and direct_trap_screen carry zero labeled tier-3 on the eval
    side, so >=3 does not exist for them. Reporting 0.5 there would put a fabricated number
    in the per-mode table for exactly the modes this finetune is being judged on."""
    lb = np.array([1, 1, 2, 2, 1])
    b = MR.boundary(lb, np.linspace(0, 1, 5), 3)
    assert b["measurable"] is False
    assert b["auc"] is None and b["ap"] is None and b["at_chance"] is None
    assert b["n_pos"] == 0 and b["n"] == 5


def test_a_perfect_ranker_scores_one_and_an_inverted_one_scores_zero():
    lb = np.array([1, 1, 3, 3])
    assert MR.boundary(lb, np.array([0.1, 0.2, 0.8, 0.9]), 3)["auc"] == pytest.approx(1.0)
    assert MR.boundary(lb, np.array([0.9, 0.8, 0.2, 0.1]), 3)["auc"] == pytest.approx(0.0)


# =========================================================================== #
# paired bootstrap — finds a real gap, invents none.
# =========================================================================== #
@pytest.fixture(scope="module")
def rng_labels():
    rng = np.random.default_rng(7)
    lb = rng.choice([1, 2, 3], size=180, p=[0.6, 0.25, 0.15])
    return lb, rng


def test_identical_scores_give_a_zero_delta_and_neither_flag(rng_labels):
    """The control. A bootstrap that reported a difference between a head and ITSELF would
    make every winner-rule verdict unfalsifiable."""
    lb, rng = rng_labels
    s = scores(rng.random(len(lb)))
    out = MR.paired_bootstrap(lb, s, s, draws=200, seed=1)
    for k, _ in MR.OVERALL_METRICS:
        assert out[k]["n_draws"] > 150                  # non-vacuity: the draws survived
        assert out[k]["lo"] == 0.0 and out[k]["hi"] == 0.0
        assert not out[k]["significantly_better"] and not out[k]["significantly_worse"]


def test_a_strictly_better_ranker_reads_as_significantly_better(rng_labels):
    lb, rng = rng_labels
    noise = rng.random(len(lb))
    weak = scores(noise)
    strong = scores(np.where(lb >= 3, 0.9, 0.1) + 0.05 * noise,
                    np.where(lb >= 2, 0.9, 0.1) + 0.05 * noise)
    out = MR.paired_bootstrap(lb, weak, strong, draws=200, seed=1)
    assert all(out[k]["significantly_better"] for k, _ in MR.OVERALL_METRICS)
    flipped = MR.paired_bootstrap(lb, strong, weak, draws=200, seed=1)
    assert all(flipped[k]["significantly_worse"] for k, _ in MR.OVERALL_METRICS)


# =========================================================================== #
# volume matching + scale shift.
# =========================================================================== #
def test_both_heads_take_the_same_count_at_every_matched_point():
    """The whole point of the volume-matched view: if the two heads selected different
    counts it would be the fixed-threshold comparison again, under a different name."""
    lb = np.array([3] * 20 + [2] * 30 + [1] * 50)
    a = scores(np.linspace(0.99, 0.01, 100))                  # v1: perfect ordering
    b = scores(np.linspace(0.55, 0.45, 100)[::-1])            # v2: compressed AND inverted
    vm = MR.volume_matched(lb, a, b)
    cells = list(vm["by_v1_live_cut"].values()) + list(vm["by_fixed_rate"].values())
    assert cells, "no matched cells — the comparison would be vacuous"
    for c in cells:
        assert c["v1"]["n_selected"] == c["v2"]["n_selected"] == c["matched_volume"]
    top20 = vm["by_fixed_rate"]["0.20"]
    assert top20["matched_volume"] == 20
    assert top20["v1"]["precision"] == pytest.approx(1.0)     # perfect ranker takes the 20
    assert top20["v2"]["precision"] == pytest.approx(0.0)     # inverted takes the worst 20


def test_a_compressed_marginal_reads_as_a_scale_shift_and_an_identical_one_does_not():
    """A shifted marginal is what makes a fixed threshold mean different things on the two
    heads — the same argument floors.py's head stamp is built on."""
    same = scores(np.linspace(0.01, 0.99, 300))
    assert MR.scale_shift(same, same)["shifted"] is False
    squashed = scores(np.linspace(0.40, 0.60, 300))
    sh = MR.scale_shift(same, squashed)
    assert sh["shifted"] is True and sh["ks"] > 0.3
    assert sh["quantiles"]["q90"]["v1"] > sh["quantiles"]["q90"]["v2"]


# =========================================================================== #
# candidate cuts — the Wilson bound is not decoration.
# =========================================================================== #
def test_a_target_met_on_three_passers_is_reported_unsupported():
    """3/3 is a precision of 1.000 whose Wilson lower bound is ~0.44 — a cut this slice
    cannot buy. `supported` is what makes that visible instead of arguable."""
    lb = np.array([3, 3, 3] + [1] * 97)
    s = np.array([0.99, 0.98, 0.97] + list(np.linspace(0.0, 0.5, 97)))
    cand = MR.candidates(MR.ladder(lb, s, thr=3))
    hit = cand["0.90"]
    assert hit is not None and hit["fires"] == 3
    assert hit["precision"] == pytest.approx(1.0)
    assert hit["precision_lo"] < 0.90 and hit["supported"] is False


def test_a_target_backed_by_enough_passers_is_reported_supported():
    """The control for the test above — `supported` must be reachable, or it is a constant."""
    lb = np.array([3] * 90 + [1] * 210)
    s = np.array([0.99] * 90 + list(np.linspace(0.0, 0.38, 210)))
    hit = MR.candidates(MR.ladder(lb, s, thr=3))["0.90"]
    assert hit["threshold"] == pytest.approx(0.40)
    assert hit["fires"] == 90 and hit["precision"] == pytest.approx(1.0)
    assert hit["precision_lo"] > 0.90 and hit["supported"] is True


def test_the_pool_candidate_is_the_HIGHEST_cut_still_keeping_the_target_recall():
    """The mirror of the release candidate, and it must not silently become the same table:
    `candidates` walks up and stops at the first precision, `recall_candidates` walks DOWN
    from the top and stops at the last threshold still retaining the target."""
    lb = np.array([3] * 10 + [1] * 90)
    s = np.array(list(np.linspace(0.95, 0.55, 10)) + list(np.linspace(0.5, 0.0, 90)))
    lad = MR.ladder(lb, s, thr=3)
    pool = MR.recall_candidates(lad)
    assert pool["0.95"]["threshold"] == pytest.approx(0.55)   # 10/10 kept, the last rung
    assert pool["0.95"]["recall"] == pytest.approx(1.0)
    assert pool["0.80"]["threshold"] > pool["0.95"]["threshold"]  # looser recall, higher cut
    assert pool["0.80"]["recall"] >= 0.80
    # and it is genuinely a different answer from the precision table
    assert MR.candidates(lad)["0.90"]["threshold"] < pool["0.80"]["threshold"]


def test_an_unreachable_recall_target_is_None_not_the_closest_rung():
    lad = MR.ladder(np.array([3, 1, 1, 1]), np.array([0.0, 0.9, 0.8, 0.7]), thr=3)
    assert MR.recall_candidates(lad, targets=(1.01,)) == {"1.01": None}


def test_the_two_live_cuts_are_exact_rows_of_the_sweep_never_nearest_bin():
    from tools.emission import floors as F
    assert F.MINING_POOL.value in MR.SWEEP and F.MINING_RELEASE.value in MR.SWEEP
    lad = MR.ladder(np.array([3, 1, 2, 3]), np.array([0.9, 0.1, 0.3, 0.6]), thr=3)
    marked = {r["threshold"]: r["marks"] for r in lad if r["marks"]}
    assert marked == {F.MINING_POOL.value: ["mining_pool"],
                      F.MINING_RELEASE.value: ["mining_release"]}


# =========================================================================== #
# the loader fails closed.
# =========================================================================== #
def _row(image_id, side, score, mode="tia"):
    return {"image_id": image_id, "render": {"palette": "p"},
            "provenance": {"split_side": side, "render_mode": mode, "mode_kind": "pure",
                           "family": "mandelbrot", "location_key": f"L{image_id}"},
            "label": {"score": score}, "suggested_tier": 1, "sheet_order": 0,
            "head_mining_v1": {"p_ge3": 0.4, "p_ge2": 0.7}}


def test_an_unlabeled_eval_row_raises_rather_than_shrinking_n(tmp_path):
    (tmp_path / "images.jsonl").write_text(
        "\n".join(json.dumps(r) for r in
                  [_row("a", "eval", 3), _row("b", "eval", None), _row("c", "train", 1)]),
        encoding="utf-8")
    with pytest.raises(SystemExit, match="unlabeled"):
        MR.load_eval_rows(tmp_path)


def test_train_side_rows_are_excluded_from_the_eval_slice(tmp_path):
    """The control: the loader must actually read `split_side`, not return everything. A
    readout over both sides would report v2 on rows v2 trained on."""
    (tmp_path / "images.jsonl").write_text(
        "\n".join(json.dumps(r) for r in
                  [_row("a", "eval", 3), _row("b", "train", 1), _row("c", "eval", 2)]),
        encoding="utf-8")
    rows = MR.load_eval_rows(tmp_path)
    assert [r["id"] for r in rows] == ["a", "c"]


# =========================================================================== #
# the committed batch — non-vacuity for everything above.
# =========================================================================== #
def test_the_committed_eval_slice_is_complete_and_covers_every_roster_mode():
    rows = MR.load_eval_rows()
    assert len(rows) == 422, "eval-side row count moved — batch.json says 422"
    assert {r["mode"] for r in rows} == set(MODES)
    assert set(TRAINER_DROPPED_V1) <= {r["mode"] for r in rows}
    labels = np.array([r["label"] for r in rows])
    assert set(labels.tolist()) <= {1, 2, 3}
    assert (labels >= 3).sum() > 0, "no tier-3 on the eval side — every >=3 read is vacuous"


def test_two_of_the_three_dropped_modes_have_no_eval_tier3_which_is_why_ge2_is_reported():
    """Pins the fact that drives the report's shape: the dropped-mode verdict cannot rest on
    the >=3 boundary alone. If a future sitting adds tier-3 rows to these modes this test
    goes red, which is the correct time to re-read the report's clause-(b) framing."""
    rows = MR.load_eval_rows()
    n3 = {m: sum(1 for r in rows if r["mode"] == m and r["label"] >= 3)
          for m in TRAINER_DROPPED_V1}
    assert n3 == {"trap_circle": 0, "exp_smoothing": 20, "direct_trap_screen": 0}
