"""The v1-vs-v3 harness's SHAPE, on synthetic scores — no checkpoints, no GPU.

What this pins is not a number: it is that the pre-declared arms exist on the real eval
slice, that the per-mode arms vote on AUCs only (the prompt's wording, and a doubling of
clause (a)'s multiplicity if it drifted), and that the markdown renders for both outcomes.
A report writer that raises on the losing branch is a report writer that has only been run
on the winning one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.mining.mining_v3_reads as MR                        # noqa: E402
from tools.mining.mining_corpus import load_corpus                # noqa: E402

DRAWS = 60


@pytest.fixture(scope="module")
def pool():
    return load_corpus(require_crops=False)


def _scores(rows, quality, seed):
    """`quality` in [0,1]: 0 = pure noise, 1 = the labels themselves."""
    rng = np.random.default_rng(seed)
    lb = np.array([r.label for r in rows])
    n = len(rows)
    return {"p_ge2": np.clip(quality * (lb >= 2) + (1 - quality) * rng.random(n), 0, 1),
            "p_ge3": np.clip(quality * (lb >= 3) + (1 - quality) * rng.random(n), 0, 1),
            "rank": rng.random(n)}


def test_every_pre_declared_arm_is_populated_on_the_real_eval_slice(pool):
    rows = pool.eval_rows
    motiv, no_worse, diag = MR.slice_masks(rows)
    assert motiv["busy_fp"].sum() >= 50
    assert no_worse["pooled"].sum() == len(rows)
    assert no_worse["rare_palette"].sum() >= 50
    assert sum(1 for k in no_worse if k.startswith("mode:")) >= 10
    assert all(m.sum() > 0 for m in diag.values())


def test_the_motivating_arm_is_exactly_sheetB_hi_fancy_plus_sheetC_fancy(pool):
    rows = pool.eval_rows
    motiv, _n, diag = MR.slice_masks(rows)
    assert (motiv["busy_fp"] == (diag["sheetB_hi_fancy"] | diag["sheetC_fancy"])).all()
    assert not (diag["sheetB_hi_fancy"] & diag["sheetC_fancy"]).any()


def test_per_mode_arms_vote_on_aucs_only():
    keys = [m.key for m in MR.voting_metrics("mode:tia")]
    assert keys == ["auc_ge3", "auc_ge2"]
    assert [m.key for m in MR.voting_metrics("pooled")] == [m.key for m in MR.METRICS]


def test_a_clearly_better_candidate_wins_and_the_report_renders(pool):
    rows = pool.eval_rows
    R = MR.build(rows, _scores(rows, 0.0, 1), _scores(rows, 0.9, 2), {}, pool,
                 draws=DRAWS, seed=3)
    assert R["winner_rule"]["winner"] == "v3"
    assert R["winner_rule"]["clause_a"]["n_tests"] > 20
    md = MR.md(R)
    assert "WINNER: v3" in md and "`busy_fp`" in md and "`rare_palette`" in md


def test_a_clearly_worse_candidate_loses_and_the_report_still_renders(pool):
    rows = pool.eval_rows
    R = MR.build(rows, _scores(rows, 0.9, 1), _scores(rows, 0.0, 2), {}, pool,
                 draws=DRAWS, seed=3)
    assert R["winner_rule"]["winner"] == "v1"
    assert R["winner_rule"]["clause_a"]["failures"]
    md = MR.md(R)
    assert "WINNER: v1" in md and "clause (a) failures" in md


def test_the_v1_unseen_diagnostic_exists_and_excludes_the_v1_sitting(pool):
    rows = pool.eval_rows
    R = MR.build(rows, _scores(rows, 0.3, 1), _scores(rows, 0.4, 2), {}, pool,
                 draws=DRAWS, seed=3)
    d = R["diagnostic"]["v1_unseen_locations"]
    assert 0 < d["n"] < len(rows)
    assert R["diagnostic"]["v1_sitting"]["n"] + d["n"] <= len(rows)


def test_the_report_never_compares_at_a_raw_shared_threshold(pool):
    """`score_scale` is a DESCRIPTION of the two scales; the precision comparison must come
    from `volume_matched`, because a fixed cut is a point on one head's scale only."""
    rows = pool.eval_rows
    R = MR.build(rows, _scores(rows, 0.3, 1), _scores(rows, 0.6, 2), {}, pool,
                 draws=DRAWS, seed=3)
    assert "volume_matched" in R and "by_v1_live_cut" in R["volume_matched"]
    for blk in R["volume_matched"]["by_v1_live_cut"].values():
        assert blk["v1"]["n_selected"] == blk["v3"]["n_selected"]
