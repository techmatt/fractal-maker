"""Tests for the post-training quantization recipe.

WHAT THESE HAVE TO CATCH, stated before the assertions, because each one is a way this tool
could ship green and be worthless (verification_practice.md §1):

  * A rung that quantizes NOTHING. Every agreement number in the study would then be a
    perfect 0.0000 — "not a null result, a measurement of nothing" (§1.11). So the round-trip
    tests assert the weights MOVED, not only that they came back.
  * A dequantizer that is right on average and wrong per channel. A per-TENSOR scale passes
    an aggregate-error assertion on these backbones and destroys the small depthwise
    channels, so the per-channel test is built on a fixture whose channels differ by 1000x —
    a fixture that CANNOT pass under a shared scale (§6, "the fixture is too easy").
  * A bar checker that reports PASS for a metric it never computed. `check_bars` is asserted
    to return MISSING, and MISSING is asserted not to count as a pass.
  * An artifact that a checkpoint loader would silently accept. `read_artifact` must refuse
    anything that is not this format, and the artifact must carry no `state_dict` key.

None of these need a GPU or a real head: the heavy path (`heads.py`) is exercised by the
acceptance run itself, and a test that spends 40 s loading v11 to assert arithmetic is a test
that gets moved to the slow lane and then never runs (§4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "quant"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_quant as EQ        # noqa: E402
import quantize_head as Q      # noqa: E402


def _fixture_state_dict():
    """A miniature state_dict with the four shapes the recipe distinguishes.

    `conv.weight` is the load-bearing one: its three output channels span 1e-3 .. 1e0, so a
    per-TENSOR scale would quantize channel 0 to zero. That is the defect the per-channel
    axis exists to prevent, and this fixture is the only thing that can tell the two apart.
    """
    g = torch.Generator().manual_seed(7)
    conv = torch.stack([
        torch.rand(2, 3, 3, generator=g) * 1e-3,
        torch.rand(2, 3, 3, generator=g) * 1e-1,
        torch.rand(2, 3, 3, generator=g) * 1e0,
    ])
    return {
        "conv_stem.weight": conv,                                    # 4-D, per-channel int8
        "conv_stem.bias": torch.rand(3, generator=g),                # 1-D -> fp16
        "blocks.0.bn.weight": torch.rand(3, generator=g),            # 1-D -> fp16
        "blocks.0.bn.num_batches_tracked": torch.tensor(17),         # int -> raw
        "classifier.weight": torch.rand(4, 3, generator=g),          # 2-D, per-channel int8
    }


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def test_group_of_splits_blocks_by_stage_and_leaves_everything_else_at_the_top():
    assert Q.group_of("conv_stem.weight") == "conv_stem"
    assert Q.group_of("blocks.2.1.conv_pw.weight") == "blocks.2"
    assert Q.group_of("blocks.10.0.bn.bias") == "blocks.10"
    assert Q.group_of("classifier.bias") == "classifier"


def test_groups_are_listed_in_state_dict_order_without_duplicates():
    gs = Q.groups_of(_fixture_state_dict())
    assert gs == ["conv_stem", "blocks.0", "classifier"]


# --------------------------------------------------------------------------- #
# the rungs
# --------------------------------------------------------------------------- #
def test_int8_is_per_output_channel_and_the_small_channel_survives():
    """The whole point of the axis. Under one shared scale the 1e-3 channel quantizes to all
    zeros; under per-channel it keeps ~1% relative accuracy like every other channel."""
    sd = _fixture_state_dict()
    q, scale = Q.quantize_tensor_int8(sd["conv_stem.weight"])
    assert scale.shape == (3,)
    assert (scale[2] / scale[0]) > 100, "the fixture must span scales or it proves nothing"
    deq = Q.dequantize_tensor_int8(q, scale, sd["conv_stem.weight"].shape, torch.float32)
    for c in range(3):
        a, b = sd["conv_stem.weight"][c], deq[c]
        rel = float((a - b).abs().amax() / a.abs().amax())
        assert rel < 0.01, f"channel {c} lost accuracy: {rel}"
    # ... and the control: one scale for the whole tensor DOES destroy channel 0.
    shared = sd["conv_stem.weight"].abs().amax() / Q.INT8_QMAX
    naive = torch.round(sd["conv_stem.weight"] / shared) * shared
    assert float(naive[0].abs().amax()) == 0.0


def test_int8_rung_puts_each_tensor_in_the_kind_the_recipe_declares():
    p = Q.quantize_state_dict(_fixture_state_dict(), "int8")
    kinds = {k: v["kind"] for k, v in p["tensors"].items()}
    assert kinds == {
        "conv_stem.weight": "int8_per_channel",
        "conv_stem.bias": "fp16",
        "blocks.0.bn.weight": "fp16",
        "blocks.0.bn.num_batches_tracked": "raw",
        "classifier.weight": "int8_per_channel",
    }


def test_fp16_rung_touches_every_float_and_no_integer_buffer():
    p = Q.quantize_state_dict(_fixture_state_dict(), "fp16")
    kinds = {v["kind"] for k, v in p["tensors"].items() if k.endswith("num_batches_tracked")}
    assert kinds == {"raw"}
    assert all(v["kind"] == "fp16" for k, v in p["tensors"].items()
               if not k.endswith("num_batches_tracked"))


def test_dequantize_restores_shape_dtype_and_the_untouched_integer_buffer_exactly():
    sd = _fixture_state_dict()
    for rung in ("fp16", "int8"):
        deq = Q.dequantize(Q.quantize_state_dict(sd, rung))
        assert set(deq) == set(sd)
        for k, v in sd.items():
            assert deq[k].shape == v.shape and deq[k].dtype == v.dtype, (rung, k)
        assert torch.equal(deq["blocks.0.bn.num_batches_tracked"],
                           sd["blocks.0.bn.num_batches_tracked"])


def test_a_rung_actually_moves_the_weights():
    """§1.11 aimed at ourselves: a rung that changed nothing would make every downstream
    agreement number a measurement of nothing, and it would LOOK like a perfect pass."""
    sd = _fixture_state_dict()
    for rung, floor in (("fp16", 0.0), ("int8", 1e-4)):
        err = Q.weight_error(sd, Q.dequantize(Q.quantize_state_dict(sd, rung)))
        assert err["n_tensors_changed"] >= 3
        assert err["max_rel_err_per_tensor"] > floor


def test_int8_moves_the_weights_strictly_more_than_fp16():
    sd = _fixture_state_dict()
    e16 = Q.weight_error(sd, Q.dequantize(Q.quantize_state_dict(sd, "fp16")))
    e8 = Q.weight_error(sd, Q.dequantize(Q.quantize_state_dict(sd, "int8")))
    assert e8["max_rel_err_per_tensor"] > e16["max_rel_err_per_tensor"]


def test_int8_max_relative_error_is_bounded_by_half_a_quantization_step():
    """Half a step of 127 levels is 1/254; anything above it means the scale is wrong."""
    sd = _fixture_state_dict()
    err = Q.weight_error(sd, Q.dequantize(Q.quantize_state_dict(sd, "int8")))
    assert err["max_rel_err_per_tensor"] <= 1.0 / (2 * Q.INT8_QMAX) + 1e-6


def test_hybrid_keeps_the_named_group_out_of_int8_and_quantizes_the_rest():
    p = Q.quantize_state_dict(_fixture_state_dict(), "hybrid", keep_fp16_groups=["conv_stem"])
    assert p["tensors"]["conv_stem.weight"]["kind"] == "fp16"
    assert p["tensors"]["classifier.weight"]["kind"] == "int8_per_channel"


def test_hybrid_without_an_exception_list_is_refused_rather_than_silently_int8():
    with pytest.raises(ValueError, match="just 'int8' under another name"):
        Q.quantize_state_dict(_fixture_state_dict(), "hybrid")


def test_the_sweep_arm_quantizes_one_group_and_leaves_the_others_bit_identical():
    """The sensitivity sweep's whole claim is attribution: if a sweep arm perturbed anything
    outside its own group, the group it names would not be the group it measured."""
    sd = _fixture_state_dict()
    deq = Q.dequantize(Q.quantize_state_dict(sd, "int8", only_groups=["classifier"]))
    assert torch.equal(deq["conv_stem.weight"], sd["conv_stem.weight"])
    assert torch.equal(deq["conv_stem.bias"], sd["conv_stem.bias"])
    assert not torch.equal(deq["classifier.weight"], sd["classifier.weight"])


def test_an_unknown_rung_is_refused():
    with pytest.raises(ValueError, match="unknown rung"):
        Q.quantize_state_dict(_fixture_state_dict(), "int4")


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #
def _write(tmp_path, rung="int8", **kw):
    src = tmp_path / "src.pt"
    torch.save({"state_dict": _fixture_state_dict(), "config": {"num_classes": 4}}, src)
    out = tmp_path / f"{rung}.qpt"
    return src, out, Q.write_artifact(src, rung, out, **kw)


def test_the_artifact_round_trips_off_disk_bit_identically(tmp_path):
    src, out, meta = _write(tmp_path)
    a, cfg, m = Q.read_artifact(out)
    b, _c, _m = Q.read_artifact(out)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert cfg == {"num_classes": 4}
    assert m["rung"] == "int8" and m["recipe_version"] == Q.RECIPE_VERSION
    assert meta["sha256"] == Q.sha256_file(out)
    assert meta["source_sha256"] == Q.sha256_file(src)


def test_the_artifact_carries_no_state_dict_key_so_a_checkpoint_loader_cannot_eat_it(tmp_path):
    _src, out, _meta = _write(tmp_path)
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert "state_dict" not in payload
    with pytest.raises(KeyError):
        _ = payload["state_dict"]


def test_read_artifact_refuses_a_plain_checkpoint(tmp_path):
    """The one failure that would make the whole study meaningless: a plain checkpoint read
    as a quantized artifact would score identically and be filed as a perfect rung."""
    p = tmp_path / "plain.pt"
    torch.save({"state_dict": _fixture_state_dict(), "config": {}}, p)
    with pytest.raises(ValueError, match="not a fractal-quant-1 artifact"):
        Q.read_artifact(p)


def _stored_bytes(payload):
    """Tensor payload bytes only. The ORDERING claim is about the weights, and a miniature
    fixture's file size is dominated by torch.save's own dict/pickle overhead — which is how
    a first cut of this test asserted int8 was BIGGER than fp16 on 60 parameters."""
    n = 0
    for d in ("int8", "scale", "fp16", "raw"):
        for t in payload[d].values():
            n += t.numel() * t.element_size()
    return n


def test_int8_stores_fewer_bytes_than_fp16_stores_fewer_than_the_source():
    sd = _fixture_state_dict()
    src = sum(t.numel() * t.element_size() for t in sd.values())
    b8 = _stored_bytes(Q.quantize_state_dict(sd, "int8"))
    b16 = _stored_bytes(Q.quantize_state_dict(sd, "fp16"))
    assert b8 < b16 < src


def test_a_real_sized_state_dict_lands_near_the_expected_size_ratios(tmp_path):
    """On a state_dict whose weights dominate — as every real head's does, 98% of float
    params are ndim>=2 — the FILE sizes have to land near 1/2 and 1/4, or the recipe is not
    doing what the record will claim it does."""
    g = torch.Generator().manual_seed(3)
    sd = {"conv_stem.weight": torch.randn(64, 64, 3, 3, generator=g),
          "conv_stem.bias": torch.randn(64, generator=g)}
    src = tmp_path / "big.pt"
    torch.save({"state_dict": sd, "config": {}}, src)
    m16 = Q.write_artifact(src, "fp16", tmp_path / "b16.qpt")
    m8 = Q.write_artifact(src, "int8", tmp_path / "b8.qpt")
    assert 0.48 < m16["bytes_after"] / m16["bytes_before"] < 0.53
    assert 0.24 < m8["bytes_after"] / m8["bytes_before"] < 0.29


def test_apply_to_model_loads_the_dequantized_weights_into_a_live_module(tmp_path):
    src = tmp_path / "src.pt"
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Linear(4, 2))
    torch.save({"state_dict": m.state_dict(), "config": None}, src)
    out = tmp_path / "q.qpt"
    Q.write_artifact(src, "int8", out)
    before = m[0].weight.detach().clone()
    Q.apply_to_model(m, out)
    assert not torch.equal(before, m[0].weight)
    assert float((before - m[0].weight).abs().amax() / before.abs().amax()) < 0.01


# --------------------------------------------------------------------------- #
# bars
# --------------------------------------------------------------------------- #
def test_check_bars_reads_the_direction_off_the_suffix():
    v = EQ.check_bars({"agreement_min": 0.98, "delta_max": 0.01},
                      {"agreement": 0.99, "delta": 0.02})
    assert v["agreement_min"]["verdict"] == "PASS"
    assert v["delta_max"]["verdict"] == "FAIL"


def test_a_bar_whose_metric_was_never_computed_is_MISSING_and_MISSING_is_not_a_pass():
    v = EQ.check_bars({"nonexistent_min": 0.5}, {"agreement": 1.0})
    assert v["nonexistent_min"]["verdict"] == "MISSING"
    assert all(x["verdict"] == "PASS" for x in v.values()) is False


def test_a_bar_key_with_no_direction_suffix_is_MALFORMED_rather_than_ignored():
    v = EQ.check_bars({"agreement": 0.98}, {"agreement": 0.99})
    assert v["agreement"]["verdict"] == "MALFORMED"


def test_the_committed_prereg_declares_a_bar_set_for_every_head_the_harness_knows():
    """The prereg and the head registry must not drift: a head with no bars would run and
    report PASS on an empty conjunction."""
    import heads as H

    prereg = json.loads((ROOT / EQ.PREREG_REL).read_text(encoding="utf-8"))
    assert set(prereg["heads"]) == set(H.HEADS)
    for k, blk in prereg["heads"].items():
        assert blk["bars"], f"{k} declares no bar"
        for bar in blk["bars"]:
            assert bar.endswith(("_min", "_max")), (k, bar)


def test_every_prereg_bar_names_a_metric_the_harness_actually_computes():
    """The paired control on the test above: bars that exist and metrics that exist are two
    sets, and a bar naming a metric nobody computes is a bar that can only report MISSING."""
    import heads as H

    prereg = json.loads((ROOT / EQ.PREREG_REL).read_text(encoding="utf-8"))
    n = 8
    for key, spec in H.HEADS.items():
        if spec.kind == "ranker":
            items = [H.QItem(key=f"{i}", group=f"loc{i // 4}", extra={"variant_id": f"v{i}"})
                     for i in range(n)]
            P = np.random.default_rng(0).random((n, 1))
            m = EQ.pref_metrics(spec, items, P, P + 1e-6, P[:, 0], P[:, 0] + 1e-6)
        else:
            K = 4
            items = [H.QItem(key=f"{i}", group=f"g{i}", label=1 + i % K) for i in range(n)]
            P = np.random.default_rng(0).random((n, K - 1))
            m = EQ.corn_metrics(spec, items, P, P + 1e-6, P.sum(1), P.sum(1) + 1e-6)
        for bar in prereg["heads"][key]["bars"]:
            assert bar[:-4] in m, f"{key}: bar {bar} names no computed metric"


def test_the_adoption_rule_takes_the_smallest_rung_that_passes_everywhere():
    ladder = ["fp16", "int8", "hybrid"]
    passing = {"fp16": ["a", "b"], "int8": ["a", "b"], "hybrid": ["a"]}
    assert EQ.choose_default(ladder, passing, 2) == ("fp16", True)


def test_when_nothing_passes_everywhere_the_widest_pass_wins_and_says_so():
    """The branch that actually fired. The flag matters as much as the rung: a default that
    does NOT cover every head is a default plus a named exception, not a clean adoption."""
    ladder = ["fp16", "int8", "hybrid"]
    passing = {"fp16": ["a", "b", "c"], "int8": [], "hybrid": ["b"]}
    assert EQ.choose_default(ladder, passing, 4) == ("fp16", False)


def test_a_tie_on_pass_width_goes_to_the_smaller_rung():
    ladder = ["fp16", "int8", "hybrid"]
    passing = {"fp16": ["a"], "int8": ["b"], "hybrid": []}
    assert EQ.choose_default(ladder, passing, 2)[0] == "fp16"


def test_no_rung_passing_anywhere_still_returns_a_rung_and_flags_it_non_universal():
    """Degenerate but reachable: every head would then be an exception, and the caller has to
    be able to say that rather than crash on an empty max()."""
    ladder = ["fp16", "int8", "hybrid"]
    rung, universal = EQ.choose_default(ladder, {r: [] for r in ladder}, 3)
    assert rung in ladder and universal is False


def test_the_committed_recipe_record_agrees_with_the_committed_agreement_record():
    """The decision is DERIVED, so it must still be derivable: re-running the rule over the
    frozen agreement table has to reproduce the rung the record claims. A record that has
    drifted from its own evidence is the 'hardcoded True' failure with more steps."""
    prereg = json.loads((ROOT / EQ.PREREG_REL).read_text(encoding="utf-8"))
    rec = json.loads((ROOT / EQ.OUT_REL).read_text(encoding="utf-8"))
    recipe = json.loads((ROOT / EQ.RECIPE_REL).read_text(encoding="utf-8"))
    ladder = [r["name"] for r in prereg["candidate_ladder"]["rungs"]]
    passing = {r: [k for k, b in rec["heads"].items()
                   if b["rungs"].get(r, {}).get("verdict") == "PASS"] for r in ladder}
    rung, universal = EQ.choose_default(ladder, passing, len(rec["heads"]))
    assert recipe["default_rung"] == rung
    assert recipe["default_passes_everywhere"] == universal
    assert sorted(recipe["default_covers"]) == sorted(passing[rung])
    for key in rec["heads"]:
        assert key in recipe["per_head"], f"{key} missing from the recipe record"


def test_the_recipe_record_names_an_exception_for_every_head_the_default_does_not_cover():
    recipe = json.loads((ROOT / EQ.RECIPE_REL).read_text(encoding="utf-8"))
    uncovered = set(recipe["per_head"]) - set(recipe["default_covers"])
    assert set(recipe["per_head_exceptions"]) == uncovered
    for key in uncovered:
        assert recipe["per_head"][key]["adopted_rung"] != recipe["default_rung"]


# --------------------------------------------------------------------------- #
# metric primitives
# --------------------------------------------------------------------------- #
def test_agreement_reports_its_denominator_not_only_a_rate():
    a = EQ.agree([1, 2, 3, 4], [1, 2, 3, 9])
    assert a == {"n_agree": 3, "n": 4, "rate": 0.75}


def test_decode_tier_counts_thresholds_met_rather_than_chaining_them():
    """The canonical CORN decode (`score_lib.corn_decode`): a frame that meets a later
    cutpoint but fails an earlier one is degraded by one rank, never promoted."""
    P = np.array([[0.9, 0.9, 0.9], [0.9, 0.4, 0.9], [0.4, 0.4, 0.4]])
    assert list(EQ.decode_tier(P)) == [4, 3, 1]


def test_spearman_is_None_on_a_constant_column_rather_than_1():
    assert EQ.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert EQ.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_auc_is_None_when_one_class_is_empty():
    assert EQ.auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert EQ.auc([0, 1], [0.1, 0.9]) == 1.0


def test_pref_metrics_are_within_location_and_a_cross_location_reorder_does_not_move_them():
    """The pref head's scores are comparable only inside a candidate set. Adding a constant
    to ONE location's scores must leave every reported quantity untouched — if it does not,
    something is being pooled that cannot be."""
    import heads as H

    spec = H.HEADS["pref"]
    items = [H.QItem(key=f"c{i}", group=f"loc{i // 4}", extra={"variant_id": f"v{i}"})
             for i in range(8)]
    s0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    s1 = s0.copy()
    s1[:4] += 100.0                       # shift one location wholesale; order preserved
    m = EQ.pref_metrics(spec, items, s0.reshape(-1, 1), s1.reshape(-1, 1), s0, s1)
    assert m["n_locations"] == 2
    assert m["mean_within_location_spearman"] == pytest.approx(1.0)
    assert m["argmax_agreement"] == 1.0
    assert m["top3_set_agreement"] == 1.0


def test_pref_metrics_catch_a_within_location_top1_flip():
    import heads as H

    spec = H.HEADS["pref"]
    items = [H.QItem(key=f"c{i}", group="loc0", extra={"variant_id": f"v{i}"})
             for i in range(4)]
    s0 = np.array([0.1, 0.2, 0.3, 0.4])
    s1 = np.array([0.1, 0.2, 0.5, 0.4])   # candidate 2 overtakes candidate 3
    m = EQ.pref_metrics(spec, items, s0.reshape(-1, 1), s1.reshape(-1, 1), s0, s1)
    assert m["argmax_agreement"] == 0.0
    assert m["per_location"][0]["top1_fp32"] == "v3"
    assert m["per_location"][0]["top1_quant"] == "v2"


def test_corn_metrics_report_the_gate_verdict_only_where_a_gate_is_pinned():
    import heads as H

    n, K = 12, 4
    items = [H.QItem(key=f"{i}", group=f"g{i}", label=1 + i % K) for i in range(n)]
    P = np.random.default_rng(1).random((n, K - 1))
    with_gate = EQ.corn_metrics(H.HEADS["wallpaper"], items, P, P, P.sum(1), P.sum(1))
    without = EQ.corn_metrics(H.HEADS["location"], items, P, P, P.sum(1), P.sum(1))
    assert "gate_verdict_agreement" in with_gate
    assert with_gate["gate"]["threshold"] == H.HEADS["wallpaper"].gate_threshold
    assert "gate_verdict_agreement" not in without


def test_identical_scores_give_perfect_agreement_and_that_is_why_vacuity_is_checked_elsewhere():
    """The control for every agreement column: feed it the same array twice. It reports a
    flawless rung — which is exactly why the weight-error check runs BEFORE the scores and
    a zero-error rung is stamped VACUOUS rather than PASS."""
    import heads as H

    n, K = 10, 4
    items = [H.QItem(key=f"{i}", group=f"g{i}", label=1 + i % K) for i in range(n)]
    P = np.random.default_rng(2).random((n, K - 1))
    m = EQ.corn_metrics(H.HEADS["location"], items, P, P.copy(), P.sum(1), P.sum(1))
    assert m["max_abs_delta_p"] == 0.0
    assert m["decoded_tier_agreement"] == 1.0
    assert m["abs_delta_auc_ge3"] == 0.0


# --------------------------------------------------------------------------- #
# pins
# --------------------------------------------------------------------------- #
def test_every_head_resolves_its_checkpoint_from_its_own_pin_module_and_the_file_exists():
    import heads as H

    for key, spec in H.HEADS.items():
        assert (ROOT / spec.ckpt_rel).exists(), f"{key}: {spec.ckpt_rel} absent"
        assert spec.pin.split(".")[-1] in ("ACTIVE_CKPT", "HEAD_CKPT_REL",
                                           "ACTIVE_MINING_CKPT", "ACTIVE_SCORER_DIR")


def test_no_head_restates_a_checkpoint_path_as_a_literal():
    """The property IS about the source (verification_practice.md §9): `ckpt_rel` must be
    computed from the pin module, so a pin flip moves this file's inputs with it. Anchored on
    the assignment shape, not on a bare name that also appears in prose."""
    import inspect

    import heads as H

    src = inspect.getsource(H)
    for spec in H.HEADS.values():
        assert f'"{spec.ckpt_rel}"' not in src, (
            f"{spec.key}'s checkpoint path is a literal in heads.py — resolve it from "
            f"{spec.pin} instead")


def test_the_quantized_weights_path_is_a_registered_bulk_prefix():
    """A quantized head must be born OUT of the tree (storage_classes.md rule 5). If this
    goes red, `artifact_path` is about to write a second copy of a live weight into git."""
    sys.path.insert(0, str(ROOT / "tools" / "corpus"))
    import artifacts

    assert Q.WEIGHTS_DIR in artifacts.RELOCATED_PREFIXES
    assert artifacts.is_relocated(f"{Q.WEIGHTS_DIR}/location_int8.qpt")
