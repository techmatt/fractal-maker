"""The v3-vs-v4b harness's SHAPE, on synthetic scores — no checkpoints, no GPU.

Same job as `tools/mining/test_mining_v3_reads.py`: the arms are the ones the TRAINER
declared (imported, not restated), the report renders on both branches, and no cross-head
comparison in it is taken at a shared raw threshold — which for a train-prior-calibrated
CORN marginal is the difference between comparing two heads and comparing two volumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.train_wallpaper_v4b import PRE_DECLARED, load_union, split_v4b  # noqa: E402
import tools.wallpaper.wallpaper_v4b_reads as WR                                # noqa: E402

DRAWS = 40


@pytest.fixture(scope="module")
def built():
    prior, sheet_a = load_union(require_crops=False)
    _tr, ev, meta = split_v4b(prior, sheet_a)
    return ev, meta


def _scores(rows, quality, seed):
    rng = np.random.default_rng(seed)
    lb = np.array([r.label for r in rows])
    n = len(rows)
    return {"p_ge2": np.clip(quality * (lb >= 2) + (1 - quality) * rng.random(n), 0, 1),
            "p_ge3": np.clip(quality * (lb >= 3) + (1 - quality) * rng.random(n), 0, 1),
            "p_ge4": np.clip(quality * (lb >= 4) + (1 - quality) * rng.random(n), 0, 1),
            "rank": rng.random(n)}


def test_the_arms_are_the_trainers_arms(built):
    ev, _m = built
    R = WR.build(ev, _scores(ev, 0.2, 1), _scores(ev, 0.3, 2), {}, _m, draws=DRAWS, seed=3)
    assert set(R["motivating"]) == set(PRE_DECLARED["motivating"])
    assert set(R["no_worse"]) == set(PRE_DECLARED["no_worse"])
    assert "overall" in R["no_worse"], "the pooled arm must be a no-worse arm"
    assert R["diagnostic"], "the diagnostics must still be reported"


def test_a_clearly_better_candidate_wins_and_the_report_renders(built):
    ev, m = built
    R = WR.build(ev, _scores(ev, 0.0, 1), _scores(ev, 0.9, 2), {}, m, draws=DRAWS, seed=3)
    assert R["winner_rule"]["winner"] == "v4b"
    md = WR.md(R)
    assert "WINNER: v4b" in md and "sheet_a_minibrot_maneuver" in md


def test_a_clearly_worse_candidate_loses_and_the_report_still_renders(built):
    ev, m = built
    R = WR.build(ev, _scores(ev, 0.9, 1), _scores(ev, 0.0, 2), {}, m, draws=DRAWS, seed=3)
    assert R["winner_rule"]["winner"] == "v3"
    assert R["winner_rule"]["clause_a"]["failures"]
    assert "clause (a) failures" in WR.md(R)


def test_every_cross_head_precision_read_is_volume_matched(built):
    ev, m = built
    R = WR.build(ev, _scores(ev, 0.3, 1), _scores(ev, 0.6, 2), {}, m, draws=DRAWS, seed=3)
    g = R["volume_matched"]["by_deployed_gate"]
    assert g["v3"]["n_selected"] == g["v4b"]["n_selected"]
    for blk in R["volume_matched"]["by_fixed_rate"].values():
        assert blk["v3"]["n_selected"] == blk["v4b"]["n_selected"]
    # the live gate appears only as a VOLUME, never as a threshold applied to both heads
    md = WR.md(R)
    assert "v4b is read at that same volume" in md


def test_the_tier4_boundary_is_measured_not_dropped(built):
    ev, m = built
    R = WR.build(ev, _scores(ev, 0.3, 1), _scores(ev, 0.6, 2), {}, m, draws=DRAWS, seed=3)
    ov = R["no_worse"]["overall"]
    assert ov["v3"]["auc_ge4"] is not None and ov["delta_ci"]["ap_ge4"]["n_draws"] > 0
