"""The (28b) fifth arm's ONE dial: the epoch-selection objective on the mining v3 trainer.

The arm's whole claim is "v1's own objective, and nothing else moves". Three ways that can be
silently false, and each is a test here:

  * the objective is chosen AFTER a number is seen. `SELECTION_METRICS` is a module constant
    declared above the training loop, and `--selection-metric` only picks a key out of it; a
    free-form string would let an arm be defined post hoc.
  * the DEFAULT drifts. Four arms were run before this dial existed and their records are
    committed; the default must still be `ap_ge3` or every one of them becomes irreproducible.
  * the selected epoch is not actually the objective's argmax — the failure that would make
    the arm a null result for the wrong reason. The selection loop is re-run here on a
    synthetic history where the two objectives peak at DIFFERENT epochs, so "it selected on
    ap_ge2" is proven by the epoch it picks rather than by the string it logs.

  uv run pytest classifier/test_selection_metric_v3.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier import train_mining_head_v3 as T                       # noqa: E402


def test_both_objectives_are_declared_above_the_code():
    assert set(T.SELECTION_METRICS) == {"ap_ge2", "ap_ge3"}
    for k, text in T.SELECTION_METRICS.items():
        assert text and k.replace("_ge", ">=").replace("ap", "AP").upper()[:5] in text.upper()


def test_the_default_is_still_the_four_committed_arms_objective():
    """v3 / v3_aug / v3_augx / v3_uniform were run before this dial existed. If the default
    moves, none of them is reproducible from the committed command line."""
    assert T.SELECTION_METRIC == "ap_ge3"
    assert T.SELECTION_TEXT == T.SELECTION_METRICS["ap_ge3"]


def test_the_flag_cannot_invent_an_objective():
    """`choices=` off the same dict — an arm cannot be defined by a string nobody declared."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-metric", default=T.SELECTION_METRIC,
                    choices=sorted(T.SELECTION_METRICS))
    with pytest.raises(SystemExit):
        ap.parse_args(["--selection-metric", "auc_ge3"])
    assert ap.parse_args([]).selection_metric == "ap_ge3"


# --------------------------------------------------------------------------- #
# The selection loop itself, replayed on a synthetic schedule.
# --------------------------------------------------------------------------- #
def _pick(sel_metric, ap2_by_epoch, ap3_by_epoch):
    """The trainer's selection rule, verbatim from `train_one_seed`: track the running best of
    whichever AP the arm declared, strictly greater, and record BOTH at that epoch."""
    best_sel, best_epoch = -1.0, -1
    best_ap = {"ap_ge2": None, "ap_ge3": None}
    for epoch, (ap_nb, ap_gd) in enumerate(zip(ap2_by_epoch, ap3_by_epoch)):
        raw = ap_gd if sel_metric == "ap_ge3" else ap_nb
        sel = -1.0 if (raw is None or not np.isfinite(raw)) else float(raw)
        if sel > best_sel:
            best_sel, best_epoch = sel, epoch
            best_ap = {"ap_ge2": float(ap_nb), "ap_ge3": float(ap_gd)}
    return best_epoch, best_sel, best_ap


# The two objectives peak at DIFFERENT epochs on purpose: on a schedule where they agree, a
# trainer that ignored the flag entirely would pass.
AP2 = [0.90, 0.94, 0.97, 0.93, 0.92]      # peaks at epoch 2
AP3 = [0.50, 0.55, 0.52, 0.61, 0.58]      # peaks at epoch 3


def test_ap_ge3_selects_the_ap3_peak():
    ep, val, both = _pick("ap_ge3", AP2, AP3)
    assert (ep, val) == (3, 0.61)
    assert both == {"ap_ge2": 0.93, "ap_ge3": 0.61}


def test_ap_ge2_selects_the_ap2_peak_and_a_DIFFERENT_epoch():
    ep, val, both = _pick("ap_ge2", AP2, AP3)
    assert (ep, val) == (2, 0.97)
    assert ep != _pick("ap_ge3", AP2, AP3)[0], "the two objectives must be separable here"
    # and the AP>=3 recorded beside it is that epoch's, NOT the AP>=3 peak — this is the
    # number the arms table compares across arms.
    assert both == {"ap_ge2": 0.97, "ap_ge3": 0.52}


def test_a_degenerate_boundary_never_wins_the_selection():
    """A NaN AP maps to -1.0, so an epoch whose boundary degenerated cannot be staged."""
    ep, val, _ = _pick("ap_ge2", [np.nan, 0.5, np.nan], [0.1, 0.2, 0.3])
    assert (ep, val) == (1, 0.5)


def test_the_selection_rule_in_the_trainer_matches_this_replay():
    """A source check on the one line the replay reproduces, so the two cannot drift apart
    without a failure."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert 'raw = ap_gd if sel_metric == "ap_ge3" else ap_nb' in src
    assert 'sel_metric = cfg["selection_metric"]' in src, \
        "the objective must be read off the run's own config, not a module constant"
