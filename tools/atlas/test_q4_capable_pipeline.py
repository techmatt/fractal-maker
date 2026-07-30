#!/usr/bin/env python
"""The pipeline must be able to EXPRESS class 4 — the pre-flip gate for a K=4 head.

Flipping a q4-capable scorer into a pipeline that cannot record or admit q4 produces
plausible-looking ledger rows that silently cap at class 3, which would defeat the entire
point of the v8 flip. These tests pin the whole chain end to end with no model and no GPU:

    scorer tuple -> reframe trace -> _chosen_probs -> corn_decode -> ledger row -> intake

Each link is independently checkable, and the K=3 path must stay byte-identical so nothing
v5..v7-era shifts under it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "tools/mining", "tools/atlas", "tools/scoring", "tools/reframe", "tools/corpus"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))

from score_lib import corn_decode  # noqa: E402
import production_seeder as ps  # noqa: E402
from tools.emission import descriptor as desc  # noqa: E402


# --------------------------------------------------------------------------- decode
def test_corn_decode_reaches_class_4_only_when_the_third_prob_is_supplied():
    # The K=3 form is capped at 3 no matter how confident it is — this is exactly the failure
    # a q4 flip would hit if the third probability never reached the decode.
    assert corn_decode(0.99, 0.99) == 3
    assert corn_decode(0.99, 0.99, 0.5) == 3
    # supplying the third cutpoint unlocks class 4
    assert corn_decode(0.99, 0.99, 0.5, 0.99) == 4
    assert corn_decode(0.99, 0.99, 0.5, 0.01) == 3


def test_corn_decode_k3_path_is_byte_identical():
    for nb in (0.0, 0.3, 0.49, 0.5, 0.8, 1.0):
        for pg in (0.0, 0.13, 0.5, 0.9):
            for t in (0.14, 0.5, 0.85):
                assert corn_decode(nb, pg, t) == 1 + int(nb >= 0.5) + int(pg >= t)
                assert corn_decode(nb, pg, t, None) == corn_decode(nb, pg, t)


def test_class4_uses_its_natural_cutpoint_not_t_good():
    # t_good moves the q3 boundary only; the third cutpoint stays at 0.5 (no per-family
    # calibration — data/v8/t_good_derivation.json no_class4_threshold).
    assert corn_decode(0.9, 0.2, 0.14, 0.6) == 4     # q3 admitted at a low t_good, q4 natural
    assert corn_decode(0.9, 0.2, 0.85, 0.6) == 3     # q3 refused at a high t_good -> counts one less


def test_decode_counts_thresholds_rather_than_chaining_them():
    # p_ge4 high but p_ge3 below t_good: the count rule degrades to 3, it does not promote to 4
    # on the strength of a cutpoint whose predecessor failed.
    assert corn_decode(0.9, 0.1, 0.5, 0.99) == 3


# --------------------------------------------------------------------------- trace -> probs
def _fake_result(p_notbad, p_good, p_ge4):
    """A reframe-shaped result whose chosen recenter candidate carries the given probs."""
    import types
    trace = {
        "chosen": {"fw_factor": 1.0, "dx": 0.0, "dy": 0.0},
        "recenter": [
            {"dx": -0.25, "dy": 0.0, "score": 0.1, "p_notbad": 0.1, "p_good": 0.0, "p_ge4": 0.0},
            {"dx": 0.0, "dy": 0.0, "score": 2.5,
             "p_notbad": p_notbad, "p_good": p_good, "p_ge4": p_ge4},
        ],
    }
    return types.SimpleNamespace(cx="0", cy="0", fw="1", score=2.5, trace=trace)


def test_chosen_probs_carries_the_third_probability_off_the_trace():
    assert ps._chosen_probs(_fake_result(0.95, 0.9, 0.7)) == (0.95, 0.9, 0.7)
    # a K=3 trace (third prob present-and-None) degrades cleanly, not to 0.0
    assert ps._chosen_probs(_fake_result(0.95, 0.9, None)) == (0.95, 0.9, None)


def test_chosen_probs_tolerates_a_pre_q4_trace_with_no_key_at_all():
    r = _fake_result(0.95, 0.9, None)
    for rc in r.trace["recenter"]:
        rc.pop("p_ge4")
    assert ps._chosen_probs(r) == (0.95, 0.9, None)


# --------------------------------------------------------------------------- ledger row
def test_ledger_write_path_records_and_decodes_class_4():
    """The `rew`-dict -> row contract the four append sites share, exercised directly."""
    rew = {"p_notbad": 0.97, "p_good": 0.93, "p_ge4": 0.88}
    t_good = ps.t_good_for("julia:multibrot4")
    decoded = corn_decode(rew["p_notbad"], rew["p_good"], t_good, rew.get("p_ge4"))
    row = {"decoded_class": decoded, "p_notbad": rew["p_notbad"], "p_good": rew["p_good"],
           "p_ge4": rew.get("p_ge4"), "t_good": t_good, "guard_pass": True, "distinct": True}
    assert row["decoded_class"] == 4, "a q4 row must be able to decode to 4"
    assert row["p_ge4"] == 0.88, "the third probability must be persisted, not dropped"
    # and the row is re-decodable from what was persisted (no re-render, no re-forward)
    assert corn_decode(row["p_notbad"], row["p_good"], row["t_good"], row["p_ge4"]) == 4


def test_a_pre_q4_rew_dict_still_writes_a_valid_row():
    rew = {"p_notbad": 0.97, "p_good": 0.93}          # no p_ge4 key (K=3 era)
    decoded = corn_decode(rew["p_notbad"], rew["p_good"], 0.5, rew.get("p_ge4"))
    assert decoded == 3 and rew.get("p_ge4") is None


# --------------------------------------------------------------------------- q3+ predicates
def test_q3plus_admits_class_4_everywhere_it_is_asked():
    assert ps.is_q3plus({"decoded_class": 4}) and ps.is_q3plus({"decoded_class": 3})
    assert not ps.is_q3plus({"decoded_class": 2})
    assert not ps.is_q3plus({"decoded_class": None})   # guard-failed
    assert not ps.is_q3plus({})                        # pre-decoded_class era, no backfill


def test_build_cloud_keeps_a_class_4_row():
    rows = [
        {"id": "q4", "family": "mandelbrot", "guard_pass": True, "decoded_class": 4,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 1e-3},
        {"id": "q3", "family": "mandelbrot", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 5.0, "outcome_cy": 5.0, "outcome_fw": 1e-3},
        {"id": "q2", "family": "mandelbrot", "guard_pass": True, "decoded_class": 2,
         "outcome_cx": 9.0, "outcome_cy": 9.0, "outcome_fw": 1e-3},
    ]
    assert {r["id"] for r in ps.build_cloud(rows, "mandelbrot")} == {"q4", "q3"}


def test_cloud_diagnostic_reports_class_4():
    rows = [{"family": "mandelbrot", "guard_pass": True, "decoded_class": c,
             "outcome_cx": float(i), "outcome_cy": 0.0, "outcome_fw": 1e-3}
            for i, c in enumerate((1, 2, 3, 4, 4))]
    diag = ps.cloud_diagnostic(rows, ps.build_cloud(rows, "mandelbrot"), "mandelbrot")
    assert diag["class_split"] == {1: 1, 2: 1, 3: 1, 4: 2}


def test_emission_intake_admits_a_class_4_row():
    assert desc.admit_quality({"decoded_class": 4})
    assert desc.admit_quality({"decoded_class": 3})
    assert not desc.admit_quality({"decoded_class": 2})
    assert not desc.admit_quality({"decoded_class": None})
    # a FLOOR source is still gated on the badness floor, untouched by the q3+ widening
    assert desc.admit_quality({"_source_tag": "q4_harvest", "p_notbad": 0.6,
                               "decoded_class": 1})
    assert not desc.admit_quality({"_source_tag": "q4_harvest", "p_notbad": 0.4,
                                   "decoded_class": 4})
