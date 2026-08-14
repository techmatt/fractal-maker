"""Guards for the backbone comparison — the arm table, the recipe guard, the arithmetic.

None of these need a GPU, a checkpoint or the aug cache: what they pin is the part that
would silently produce a WRONG comparison rather than a failed one — a drifted
hyperparameter, an AUC that disagrees with sklearn's, a "paired" bootstrap that is not
paired, a decode that is not CORN's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in (".", "tools", "tools/scoring"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from backbone_search import arms as A  # noqa: E402
from backbone_search import eval_arms as E  # noqa: E402
from backbone_search import train_arm as T  # noqa: E402


# ------------------------------- the arm table ------------------------------- #
def test_arm_table_is_well_formed():
    names = [a.name for a in A.ARMS]
    assert len(names) == len(set(names)), "duplicate arm name"
    assert sum(a.is_control for a in A.ARMS) == 1
    assert A.CONTROL.timm_model.startswith("mobilenetv4_conv_medium"), \
        "the control must be the shipped v5..v11 backbone, or nothing is a control"
    for a in A.ARMS:
        assert "." in a.timm_model, f"{a.name}: pin a pretrained TAG, not a bare arch"
        assert a.why.strip(), f"{a.name}: an arm with no stated reason is not an arm"


def test_run_order_covers_every_arm_and_starts_with_the_control():
    from backbone_search import run_round
    assert run_round.ORDER[0] == A.CONTROL.name
    assert set(run_round.ORDER) == set(A.ARMS_BY_NAME)


def test_weights_are_out_of_tree_and_records_are_in_it():
    """The two storage classes of this study, asserted rather than described.

    A weight under the tracked `!/data/backbone_search/` negation would be COMMITTED, not
    merely present — the trap `data/atlas/tau_h_rederive` names."""
    a = A.CONTROL
    assert ROOT not in a.weights_dir(0).parents, \
        "arm weights must resolve OUT of the repo (artifacts.RELOCATED_PREFIXES)"
    assert ROOT in a.record_dir(0).parents, "arm records must be in-tree and tracked"


# ------------------------------ the recipe guard ----------------------------- #
def _v11_like():
    return {"num_classes": 4, "epochs": 40, "batch_size": 32, "backbone_lr": 2e-4,
            "head_lr": 1e-3, "geometry": "stretch", "interpolation": "bicubic",
            "drop_rate": 0.2, "drop_path_rate": 0.1, "seed": 0, "grad_clip": 1.0,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "input_size": [3, 384, 384], "src_dims": [512, 288], "target_dims": [384, 224],
            "backbone": "mobilenetv4_conv_medium.e250_r384_in12k",
            "beta_biased": 0.4, "class_balance": "sqrt", "best_epoch": 21}


def _arm_cfg(arm, seed=0):
    dc = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "interpolation": "bilinear",
          "input_size": (3, 224, 384)}
    return T.build_arm_config(_v11_like(), arm, seed, dc, rnd=1)


@pytest.mark.parametrize("arm", A.ARMS, ids=[a.name for a in A.ARMS])
def test_every_arm_config_passes_the_drift_guard(arm):
    cfg = _arm_cfg(arm)
    T.assert_recipe_untouched(_v11_like(), cfg)          # raises SystemExit on drift
    assert cfg["backbone"] == arm.timm_model
    assert cfg["grad_checkpointing"] == arm.grad_checkpointing


def test_the_deploy_transform_does_not_follow_the_backbone():
    """Geometry AND interpolation stay v11's even when the arm's data config disagrees —
    two image pipelines would make the comparison a render comparison."""
    cfg = _arm_cfg(A.ARMS_BY_NAME["vit_small_p16"])
    assert cfg["interpolation"] == "bicubic" and cfg["geometry"] == "stretch"
    assert cfg["target_dims"] == [384, 224] and cfg["input_size"] == (3, 224, 384)
    # normalization, by contrast, DOES follow the pretrained weights
    assert tuple(cfg["mean"]) == (0.5, 0.5, 0.5)


@pytest.mark.parametrize("key,bad", [("batch_size", 16), ("epochs", 10),
                                     ("backbone_lr", 1e-3), ("drop_path_rate", 0.0),
                                     ("class_balance", "none")])
def test_drift_on_a_behavioural_key_is_refused(key, bad):
    cfg = _arm_cfg(A.CONTROL)
    cfg[key] = bad
    with pytest.raises(SystemExit, match="DRIFTED"):
        T.assert_recipe_untouched(_v11_like(), cfg)


def test_seed_and_backbone_are_the_only_experiment_keys():
    """`_ARM_KEYS` is the exemption list; a key added to it stops being guarded, so the
    set is pinned here rather than left to whoever edits it next."""
    assert set(T._ARM_KEYS) == {
        "backbone", "backbone_kwargs", "seed", "mean", "std", "input_size",
        "grad_checkpointing", "arm", "arm_pretrain", "arm_why", "frozen_from", "round"}


# -------------------------------- arithmetic -------------------------------- #
def test_fast_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 200)
        s = rng.normal(size=200) + y * 0.7
        if y.min() == y.max():
            continue
        assert E.fast_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_fast_auc_handles_ties_by_midrank():
    y = np.array([1, 1, 0, 0])
    assert E.fast_auc(y, np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.5)
    assert E.fast_auc(y, np.ones(4)) is not None
    assert E.fast_auc(np.zeros(4), np.arange(4)) is None      # one class empty -> None


def test_decode_tier_is_corns_own_decode():
    p = np.array([[0.9, 0.8, 0.7],      # all three cutpoints fire -> 4
                  [0.9, 0.8, 0.4],      # two -> 3
                  [0.6, 0.2, 0.1],      # one -> 2
                  [0.4, 0.3, 0.2]])     # none -> 1
    assert list(E.decode_tier(p)) == [4, 3, 2, 1]


def test_metrics_block_reports_counts_and_agreement():
    labels = np.array([1, 2, 3, 4] * 5)
    probs = np.tile(np.array([[0.9, 0.8, 0.7]]), (20, 1))     # everything decodes to 4
    m = E.metrics_block(labels, probs)
    assert m["n"] == 20 and m["n_ge3"] == 10 and m["n_ge4"] == 5
    assert m["exact_agree"] == pytest.approx(0.25)
    assert m["adj_agree"] == pytest.approx(0.5)               # 3s and 4s
    assert m["auc_ge3"] == pytest.approx(0.5)                 # constant score -> chance


def test_paired_bootstrap_of_an_arm_against_itself_is_exactly_zero():
    """The pairing IS the claim: two identical arms must give a degenerate interval, which
    only happens if both are recomputed on the SAME resampled rows each draw."""
    rng = np.random.default_rng(1)
    labels = rng.integers(1, 5, 300)
    groups = rng.integers(0, 40, 300)
    p = rng.random((300, 3))
    ci = E.paired_cluster_boot(labels, groups, p, p,
                               lambda lb, pr: E.fast_auc((lb >= 3).astype(int), pr[:, 1]),
                               B=200, seed=0)
    assert ci["lo"] == 0.0 and ci["hi"] == 0.0


def test_paired_bootstrap_resamples_whole_groups():
    """A cluster bootstrap must draw GROUPS, not rows: with one group the only possible
    resample is the whole population, so any statistic's interval is degenerate."""
    rng = np.random.default_rng(2)
    labels = rng.integers(1, 5, 120)
    groups = np.zeros(120, dtype=int)
    a, c = rng.random((120, 3)), rng.random((120, 3))
    ci = E.paired_cluster_boot(labels, groups, a, c,
                               lambda lb, pr: E.fast_auc((lb >= 3).astype(int), pr[:, 1]),
                               B=50, seed=0)
    assert ci["lo"] == pytest.approx(ci["hi"])


# --------------------------- the pre-registration --------------------------- #
def test_prereg_exists_and_declares_its_bars():
    p = ROOT / "data/backbone_search/prereg_backbone_v1.json"
    if not p.exists():
        pytest.skip("prereg not written yet")
    import json
    d = json.loads(p.read_text())
    assert d["eval_populations"]["PRIMARY (unseen)"]["n"] > 0
    assert "julia:mandelbrot" in d["declared_slices"]["named_up_front"]
    assert "phoenix" in d["declared_slices"]["named_up_front"]
    assert d["honesty_rule"]["bootstrap"]["unit"].startswith("split_group")
    assert {a["name"] for a in d["arms"]} == set(A.ARMS_BY_NAME), \
        "the prereg's arm set and the arm table disagree — one of them was edited after"
    assert d["adoption"].startswith("NOTHING IS ADOPTED")
