"""Tests for the FAIL-CLOSED split classifier in tools/v7/build_manifest.py.

`assign_split` used to fall every UNREGISTERED batch through to
`("train", False, "loose0_v3")` — tagged unbiased and sourced to the unbiased bucket —
so five intentionally-biased batches added after mid-July classified as unbiased. The
fix inverts the default: unregistered => biased/train; unbiased/eval-eligible requires
explicit registration. These tests bracket both sides of that inversion.

Run:  uv run python -m pytest tools/v7/test_build_manifest_split.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v7"))
import build_manifest as bm  # noqa: E402


def split(batch, ft):
    return bm.assign_split({"batch": batch, "ft": ft})


# The five intentionally-biased batches the old default silently misclassified as unbiased
# (cc_prompt_tier1_closers.md task B). Each paired with a representative fractal_type.
FIVE_BIASED = [
    ("2026-07-21_phoenix_grid", "phoenix"),
    ("2026-07-22_native_multibrot_band_v1", "multibrot4"),
    ("2026-07-26_anchor_class4_v1", "mandelbrot"),
    ("2026-07-26_minibrot_roster_v2", "mandelbrot"),
    ("2026-07-27_interior_band_v1", "multibrot4"),
]

# Batches that already classified correctly BEFORE the fix — their (split, biased, source)
# must be byte-identical after it (zero-change proof). These are the exact tuples the OLD
# assign_split produced for these inputs.
UNCHANGED = [
    # census julia -> eval (unbiased instrument)
    (("2026-07-17_prospect_run1_baserate_v1", "julia_multibrot3"),
     ("eval", False, "prospect_census")),
    (("2026-07-17_prospect_run1_baserate_R_v1", "julia_multibrot4"),
     ("eval", False, "prospect_census")),
    # census native-plane multibrot -> train (biased)
    (("2026-07-17_prospect_run1_baserate_v1", "multibrot3"),
     ("train", True, "prospect_native")),
    # model-band batches -> train (biased)
    (("2026-07-11_jm3_band_v1", "julia_multibrot3"), ("train", True, "jm_band")),
    (("2026-07-12_jm45_band_v1", "julia_multibrot4"), ("train", True, "jm_band")),
    # blindspot negatives -> train (biased)
    (("2026-07-12_blindspot_v6reject_v1", "mandelbrot"),
     ("train", True, "blindspot_v6reject")),
    # the one explicitly-registered unbiased train source -> train (unbiased)
    (("2026-06-23_flat_generate_loose0_v3", "mandelbrot"),
     ("train", False, "loose0_v3")),
]


# 1. An unregistered batch name lands biased / train.
@pytest.mark.parametrize("ft", ["mandelbrot", "multibrot5", "julia", "julia_multibrot3", "phoenix"])
def test_unregistered_batch_is_biased_train(ft):
    sp, biased, source = split("2026-08-01_some_brand_new_batch", ft)
    assert sp == "train"
    assert biased is True
    assert source == "unregistered"


# 2. Each of the five named batches classifies as biased.
@pytest.mark.parametrize("batch,ft", FIVE_BIASED)
def test_five_named_batches_are_biased(batch, ft):
    sp, biased, source = split(batch, ft)
    assert biased is True, f"{batch} must be biased"
    assert sp == "train", f"{batch} must be train-side"


# 3. Prove the OLD default was wrong on purpose (RED before the fix). Under the old
#    fall-through these five returned exactly ("train", False, "loose0_v3") — unbiased.
#    This assertion FAILS against the pre-fix code and passes now.
@pytest.mark.parametrize("batch,ft", FIVE_BIASED)
def test_old_unbiased_default_is_gone(batch, ft):
    res = split(batch, ft)
    assert res != ("train", False, "loose0_v3"), (
        f"{batch} still hits the retired unbiased fall-through")
    assert res[1] is True and res[2] != "loose0_v3"


# 4. Zero-change proof: every batch that already classified correctly is identical.
@pytest.mark.parametrize("inp,expected", UNCHANGED)
def test_correctly_classified_batches_unchanged(inp, expected):
    assert split(*inp) == expected


# ---- the label_store cross-check gate (registration_contradictions) ----

def test_no_self_contradiction_for_real_batches():
    """assign_split never produces a location that is unbiased yet train-side-only in
    label_store — so the real classifications carry no contradiction."""
    locs = []
    for batch in bm.ls.TRAIN_SIDE_ONLY_BATCHES:
        sp, biased, source = split(batch, "multibrot4")
        locs.append({"batch": batch, "split": sp, "biased": biased, "source": source})
    assert bm.registration_contradictions(locs) == []


def test_contradiction_gate_fires():
    """A batch registered train-side-only but classified unbiased is caught."""
    victim = next(iter(bm.ls.TRAIN_SIDE_ONLY_BATCHES))
    locs = [
        {"batch": victim, "split": "eval", "biased": False, "source": "bogus_unbiased"},
        {"batch": "2026-06-23_flat_generate_loose0_v3", "split": "train",
         "biased": False, "source": "loose0_v3"},   # unbiased but NOT train-side-only -> ok
    ]
    contra = bm.registration_contradictions(locs)
    assert len(contra) == 1
    assert contra[0]["batch"] == victim
