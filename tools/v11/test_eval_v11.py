#!/usr/bin/env python
r"""Guards for the v11 certification arithmetic and for the pre-registration contract.

Cheap: no model, no tile, no GPU. What is tested is the part of a certification that can be
silently wrong — the cutpoint/ordering/calibration arithmetic, and the mechanical property
that makes "pre-registered" mean anything, namely that the eval script LOADS its bars
instead of restating them.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tools/corpus", "tools/scoring"):
    sys.path.insert(0, str(ROOT / sub))

PREREG = ROOT / "data" / "v11" / "prereg_v11.json"
EVAL_SRC = ROOT / "tools" / "v11" / "eval_v11.py"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prereg():
    assert PREREG.exists(), (
        "data/v11/prereg_v11.json is MISSING. It is the committed record every v11 bar is "
        "loaded from — without it the eval script has no thresholds and 'pre-registered' "
        "means nothing. Rebuild: uv run python tools/v11/prereg.py")
    return json.loads(PREREG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def E():
    return _load("v11_eval_mod", "tools/v11/eval_v11.py")


@pytest.fixture(scope="module")
def PR():
    return _load("v11_prereg_mod", "tools/v11/prereg.py")


# --------------------------------------------------------------------------- #
# the power bar is a function of the counts — which is what lets it be pre-registered
# --------------------------------------------------------------------------- #
def test_uniform90_bar_reproduces_v10s(PR):
    """v10 derived 0.64 at (22 pos, 68 neg) by hand; v11 recomputes it. If this moved, the
    instrument's labels moved and the 'standing read' is not standing."""
    bar, se = PR.min_detectable_auc(22, 68)
    assert bar == 0.64
    assert 0.070 <= se <= 0.072


def test_min_detectable_auc_clears_chance_and_is_minimal(PR):
    for n_pos, n_neg in ((22, 68), (62, 228), (26, 500)):
        bar, _ = PR.min_detectable_auc(n_pos, n_neg)
        lo = bar - 1.96 * PR.hanley_mcneil_se(bar, n_pos, n_neg)
        assert lo > 0.50, (n_pos, n_neg, bar, lo)
        prev = round(bar - 0.01, 4)
        lo_prev = prev - 1.96 * PR.hanley_mcneil_se(prev, n_pos, n_neg)
        assert lo_prev <= 0.50, ("not minimal", n_pos, n_neg, prev, lo_prev)


def test_more_positives_lowers_the_bar(PR):
    assert PR.min_detectable_auc(62, 228)[0] < PR.min_detectable_auc(22, 68)[0]


# --------------------------------------------------------------------------- #
# cutpoint vs ordering — the arm's whole point is that they are separable
# --------------------------------------------------------------------------- #
def test_cutpoint_read_counts_a_loose_cut_as_loose(E):
    y = [1, 1, 0, 0]
    loose = [0.9, 0.9, 0.8, 0.7]      # calls 4/4 -> precision 0.5, rate 1.0
    tight = [0.9, 0.4, 0.2, 0.1]      # calls 1/4 -> precision 1.0, rate 0.25
    a, b = E.cutpoint_read(y, loose, 0.5), E.cutpoint_read(y, tight, 0.5)
    assert (a["n_pred_pos"], a["precision"], a["recall"], a["predicted_rate"]) == \
        (4, 0.5, 1.0, 1.0)
    assert (b["n_pred_pos"], b["precision"], b["recall"], b["predicted_rate"]) == \
        (1, 1.0, 0.5, 0.25)


def test_cutpoint_is_blind_to_ordering_and_ordering_to_the_cut(E):
    """A monotone squash keeps the ORDER identical while moving every prediction below the
    cut. That is exactly v10's reported defect shape, inverted — and the reason the arm
    reports both numbers: neither one alone can see it."""
    y = np.array([1, 1, 0, 0])
    p = np.array([0.95, 0.85, 0.75, 0.55])
    squashed = p * 0.5
    assert E.cutpoint_read(y, p, 0.5)["n_pred_pos"] == 4
    assert E.cutpoint_read(y, squashed, 0.5)["n_pred_pos"] == 0
    from scipy.stats import spearmanr
    assert spearmanr(p, squashed).correlation == pytest.approx(1.0)


def test_cutpoint_read_survives_a_degenerate_prediction(E):
    r = E.cutpoint_read([1, 0], [0.1, 0.1], 0.5)
    assert r["n_pred_pos"] == 0 and r["precision"] is None and r["f1"] is None


# --------------------------------------------------------------------------- #
# calibration reads
# --------------------------------------------------------------------------- #
def test_reliability_is_zero_ece_on_a_perfectly_calibrated_slice(E):
    rng = np.random.default_rng(0)
    p = np.repeat([0.05, 0.25, 0.45, 0.65, 0.85], 400)
    y = (rng.random(p.size) < p).astype(int)
    r = E.reliability(y, p)
    assert r["ece"] < 0.03
    assert r["mean_p"] == pytest.approx(r["base_rate"], abs=0.03)


def test_reliability_names_an_overconfident_head(E):
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    p = np.full(100, 0.9)
    r = E.reliability(y, p)
    assert r["base_rate"] == 0.1 and r["mean_p"] == 0.9
    assert r["ece"] == pytest.approx(0.8, abs=1e-6)


def test_fbeta_argmax_ties_toward_the_higher_t_and_reports_the_plateau(E):
    """Protocol §4: tie-break high, which puts the pick at the plateau's UPPER edge by
    construction — so the plateau width is the only honest read on how knife-edged it is."""
    y = np.array([1] * 10 + [0] * 10)
    p = np.array([0.9] * 10 + [0.1] * 10)
    r = E.fbeta_argmax(y, p, 0.5)
    assert r["f_at_argmax"] == pytest.approx(1.0)
    assert r["t_argmax"] == r["plateau_hi"] > r["plateau_lo"]
    assert r["plateau_width"] > 0
    assert r["precision_at_t"] == 1.0 and r["recall_at_t"] == 1.0


def test_f2_admits_more_than_f05_on_the_same_slice(E):
    """The supply argument in one assertion: recall-weighting cuts lower and admits more."""
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.15).astype(int)
    p = np.clip(0.25 * y + rng.normal(0.3, 0.2, 400), 0.001, 0.999)
    f05, f2 = E.fbeta_argmax(y, p, 0.5), E.fbeta_argmax(y, p, 2.0)
    assert f2["t_argmax"] <= f05["t_argmax"]
    assert f2["n_pred_pos"] >= f05["n_pred_pos"]


# --------------------------------------------------------------------------- #
# the verdict rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cand,base,p,expect", [
    (0.70, 0.75, 0.40, True),    # inside the margin, not significant
    (0.69, 0.75, 0.40, False),   # outside the margin
    (0.74, 0.75, 0.01, False),   # inside the margin but significantly below
    (0.80, 0.75, 0.01, True),    # significantly ABOVE is not an inferiority
])
def test_noninferiority_rule(E, cand, base, p, expect):
    block = {"auc_cand": cand, "auc_base": base, "delong_p": p,
             "delta_cand_minus_base": cand - base}
    assert E.noninferior(block, 0.05) is expect


def test_separation_needs_both_the_bar_and_the_ci(E):
    assert E.separates(0.70, 0.55, 0.64)
    assert not E.separates(0.63, 0.55, 0.64)      # below the bar
    assert not E.separates(0.70, 0.49, 0.64)      # CI touches chance


# --------------------------------------------------------------------------- #
# the pre-registration contract
# --------------------------------------------------------------------------- #
def test_every_arm_declares_a_bar_and_a_gating_flag(prereg):
    for name, arm in prereg["arms"].items():
        assert "gating" in arm, name
        assert ("bar" in arm) or ("cutpoint_bar" in arm), name
        if arm["gating"]:
            assert arm.get("bar"), f"{name} gates on a null bar"


def test_the_gating_set_is_the_three_pinned_instruments(prereg):
    gating = {k for k, a in prereg["arms"].items() if a["gating"]}
    assert gating == {"primary_census144", "floor_loose0_v3", "uniform90"}


def test_eval_script_loads_the_bars_instead_of_restating_them():
    """A bar in the eval script can be edited after seeing the results; a bar in a committed
    artifact the script loads cannot. So the script must not carry a margin or a separation
    threshold of its own."""
    src = EVAL_SRC.read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#") and '"' not in l and "'" not in l)
    for pat in (r"noninf_margin\s*=\s*0", r"separation_bar\s*=\s*0",
                r"MARGIN\s*=\s*0", r"BAR\s*=\s*0"):
        assert not re.search(pat, body), f"eval_v11.py restates a bar: {pat}"
    assert 'arms["primary_census144"]' in src and 'prereg["arms"]' in src


def test_the_motivating_slice_is_out_of_sample_for_both_heads(prereg):
    pop = prereg["eval_population"]["correction_sitting"]
    assert pop["in_v10_corpus"] == 0, (
        "a correction-sitting row is in v10's corpus — the arm's premise is false")
    assert pop["holdout_out_of_sample_for_both"]["n_eq4"] >= 15, "too few fours to read 3|4"
    assert (pop["holdout_out_of_sample_for_both"]["n"]
            + pop["train_side_v11"]["n"]) == pop["all"]["n"]


def test_the_new_partitions_clear_min_pos(prereg):
    arm = prereg["arms"]["per_partition_calibration_first_reads"]
    for name, c in arm["populations"].items():
        assert c["n_ge3"] >= arm["min_pos"], (name, c)


def test_held_out_palettes_are_the_v9_v10_set(prereg):
    recipe = json.loads((ROOT / "data/v11/aug_recipe.json").read_text(encoding="utf-8"))
    assert prereg["arms"]["palette_invariance"]["held_out_palettes"] == \
        recipe["palettes"]["held_out"]
    assert len(recipe["palettes"]["held_out"]) == 8
    assert not set(recipe["palettes"]["held_out"]) & set(recipe["palettes"]["draw_pool"]), \
        "a held-out palette is in the draw pool — the invariance battery is not held out"
