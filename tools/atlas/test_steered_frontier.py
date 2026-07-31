#!/usr/bin/env python
"""Unit tests for the steered-frontier v1.1 priority terms + the keeper cut.

Pure / fast — no render, no GPU, no binary. The control for the two new priority terms
(morph-novelty + depth) and the acceptance guarantee that BOTH coefficients at zero
reproduce the pilot priority exactly, plus the F0.5 keeper-cut metric math.

  uv run pytest tools/atlas/test_steered_frontier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import steered_frontier as sf   # noqa: E402
import keeper_cut as kc         # noqa: E402


# =========================================================================== #
# novelty penalty — the re-anchored ramp.
# =========================================================================== #
def test_novelty_zero_when_disabled():
    # lambda_m == 0 -> identically zero for ANY cosine (incl. a perfect dup).
    for cos in (-1.0, 0.0, 0.85, 0.974, 1.0):
        assert sf.novelty_penalty(cos, 0.0, 0.80, 0.97) == 0.0


def test_novelty_ramp_anchors():
    lo, hi, lam = 0.80, 0.90, 0.5
    assert sf.novelty_penalty(lo - 0.01, lam, lo, hi) == 0.0          # below lo: zero
    assert sf.novelty_penalty(lo, lam, lo, hi) == 0.0                 # at lo: zero
    assert abs(sf.novelty_penalty((lo + hi) / 2, lam, lo, hi) - lam / 2) < 1e-9  # midpoint: half
    assert abs(sf.novelty_penalty(hi, lam, lo, hi) - lam) < 1e-9      # at hi: full
    assert abs(sf.novelty_penalty(1.0, lam, lo, hi) - lam) < 1e-9     # above hi: clamped full


# =========================================================================== #
# priority decomposition — the acceptance guarantee.
# =========================================================================== #
def test_priority_reduces_to_pilot_at_zero_coeffs():
    # With lambda_m == 0 AND beta == 0, priority == pilot's eord + gumbel - dup_pen, and the
    # novelty/depth terms vanish REGARDLESS of cos_max or depth (so the frontier order, hence
    # the whole run, is byte-identical to the pilot).
    for eord, g, dup, cos, depth in [
        (1.3, 0.02, 0.0, 0.99, 7), (0.4, -0.05, 0.8, 0.5, 2), (1.9, 0.1, 0.3, 0.9741, 13),
    ]:
        prio, terms = sf.priority_terms(eord, g, dup, cos, 0.0, 0.0, depth, 0.85, 0.974)
        assert prio == eord + g - dup
        assert terms["nov_pen"] == 0.0
        assert terms["depth_bonus"] == 0.0
        assert terms["priority"] == prio


def test_priority_full_terms_contribute():
    # Both terms live: a near-dup at high depth is penalised by novelty and lifted by depth.
    lo, hi, lam, beta = 0.80, 0.90, 0.5, 0.02
    prio, terms = sf.priority_terms(1.0, 0.0, 0.0, hi, lam, beta, 10, lo, hi)
    assert abs(terms["nov_pen"] - lam) < 1e-9            # full novelty penalty
    assert abs(terms["depth_bonus"] - beta * 10) < 1e-9  # depth bonus
    assert abs(prio - (1.0 - lam + beta * 10)) < 1e-9


# =========================================================================== #
# anchor resolution — CLI override > file > fallback; degenerate ramp guarded.
# =========================================================================== #
def test_anchor_cli_override_and_guard():
    lo, hi, src = sf.load_morph_anchors(cli_lo=0.6, cli_hi=0.95)
    assert (lo, hi) == (0.6, 0.95) and "cli_lo" in src and "cli_hi" in src
    # hi <= lo is repaired to a positive-width ramp.
    lo2, hi2, _ = sf.load_morph_anchors(cli_lo=0.9, cli_hi=0.8)
    assert hi2 > lo2


# =========================================================================== #
# keeper cut — F0.5 metric math + calibration gate.
# =========================================================================== #
def test_fbeta_precision_weighted():
    # F0.5 weights precision over recall: at equal-ish P/R it sits between them, closer to P.
    p_heavy = kc.prf_beta(tp=8, fp=1, fn=4)   # P=0.889 R=0.667
    prec, rec, f = p_heavy
    assert prec > rec
    # F0.5 must lie strictly between recall and precision (precision-leaning).
    assert rec < f < prec


def test_fbeta_beta_half_formula():
    # explicit check against the closed form (1+0.25)*P*R / (0.25*P + R).
    prec, rec, f = kc.prf_beta(tp=6, fp=2, fn=3)
    p, r = 6 / 8, 6 / 9
    expect = 1.25 * p * r / (0.25 * p + r)
    assert abs(f - expect) < 1e-9


def test_keeper_cuts_rederive_matches_the_committed_constant():
    """The re-derivation drift gate, RESTORED now that its input is durable.

    This check was retired in the v7 era for a good reason: `kc.derive()` read
    `data/classifier/v7/eval_scores_v7.jsonl`, which was gitignored, was never committed, and
    is gone — so the check skip-plumbed itself quiet and was a gate that could never fire. The
    v8 recut moved the population to `data/v8/eval_scores_v8.jsonl`, which is `paths.durable()`
    and committed, so the reason to retire it no longer holds and the gate comes back: the
    committed constant must be exactly what the derivation code produces from the committed
    slice. A hand-edited threshold, a changed objective, or a changed keeper-positive predicate
    all surface here instead of silently shipping."""
    assert kc.EVAL.exists(), f"{kc.EVAL} missing — the keeper population must stay durable"
    fresh = kc.derive()
    committed = json.loads(kc.OUT.read_text(encoding="utf-8"))["cuts"]
    assert set(fresh) == set(committed)
    for part in fresh:
        assert float(fresh[part]["t"]) == float(committed[part]["t"]), (
            f"{part}: re-derived t={fresh[part]['t']} != committed {committed[part]['t']} — "
            f"re-run tools/atlas/keeper_cut.py")
        assert fresh[part]["calibrated"] == committed[part]["calibrated"], part
        assert fresh[part]["pos"] == committed[part]["pos"], part


def test_keeper_positive_is_label_ge_3_not_eq_3():
    # Under v8's 1..4 labels a class-4 location is the BEST kind of keeper. Scoring it as a
    # negative would push the precision-weighted cut in exactly the wrong direction, and the
    # v7-era `label == 3` predicate did precisely that once class 4 existed.
    rows = kc.read_jsonl(kc.EVAL)
    assert any(r["label"] == 4 for r in rows), "no class-4 rows — this test would be vacuous"
    triples = kc.load_triples()
    n_pos = sum(1 for rs in triples.values() for _, _, pos in rs if pos)
    n_ge3 = sum(1 for r in rows if r["label"] >= 3 and kc.FT2FAM.get(r["fractal_type"]))
    assert n_pos == n_ge3 > sum(1 for r in rows if r["label"] == 3)


def test_keeper_cuts_committed_shape_partitions_and_provenance():
    # LIVE gate on the committed report-only constant data/atlas/keeper_cuts.json — the thing
    # actually consumed (steered_run2_*/keeper_calibrate read it via kc.load_keeper_cuts). It
    # does NOT re-assert threshold VALUES (the re-derivation test above does that). It guards the
    # three things that would silently rot the constant: parseable shape, live partition coverage,
    # and a provenance stamp whose named model is the active checkpoint.
    import json
    import active_ckpt  # single source of truth for the live scorer version (tools/scoring)

    doc = json.loads(kc.OUT.read_text(encoding="utf-8"))
    cuts = doc["cuts"]

    # (1) shape — every partition row parses to the fields consumers rely on.
    for part, row in cuts.items():
        assert isinstance(row["calibrated"], bool), part
        t = row["t"]
        assert isinstance(t, (int, float)) and 0.0 <= float(t) <= 1.0, (part, t)
        assert isinstance(row["n"], int) and isinstance(row["pos"], int), part
        if not row["calibrated"]:
            assert float(t) == kc.T_GOOD_BASELINE, part   # uncalibrated => discovery baseline

    # (2) partition coverage — the live set is FT2FAM's targets (derived from code, not hardcoded,
    # so adding a family to the derivation without recutting keeper_cuts.json fails loudly here).
    live = set(kc.FT2FAM.values())
    assert set(cuts) == live, f"keeper_cuts partitions {set(cuts)} != live {live}"

    # (3) provenance — must carry a stamp naming the model + population the cuts came from. A stamp
    # that NAMES a model must name the ACTIVE checkpoint; a null model is an accepted "unverified"
    # stamp (used only if the basis were not cheaply determinable — it is, so today model=='v8').
    prov = doc["provenance"]
    assert prov["population"], "provenance must name the population the cuts were derived from"
    model = prov.get("model")
    if model is not None:
        assert model == active_ckpt.ACTIVE_VERSION, (
            f"keeper_cuts provenance names model {model!r} but active checkpoint is "
            f"{active_ckpt.ACTIVE_VERSION!r} (tools/scoring/active_ckpt.py) — recut or restamp")


def test_pop_batch_evicts_capped_root_nodes():
    # A node whose root is at M_CAP must be EVICTED (removed from the frontier + its cached
    # embedding dropped), not merely skipped-and-retained — else capped nodes clog the frontier.
    import types
    obj = types.SimpleNamespace(
        B=2, maneuvers=False, man_quota=0,             # maneuver floor off => plain top-B
        expansions_per_root={"1": sf.M_CAP, "2": 0},   # root 1 capped, root 2 open
        node_embs={101: None, 102: None, 201: None},
        totals={"cap_hits": 0},
        frontier=[
            {"node_id": 101, "root_id": 1, "priority": 5.0},   # capped -> evicted
            {"node_id": 102, "root_id": 1, "priority": 4.0},   # capped -> evicted
            {"node_id": 201, "root_id": 2, "priority": 3.0},   # open  -> popped
        ],
    )
    obj._split_reserved = types.MethodType(sf.SteeredFrontier._split_reserved, obj)
    batch = sf.SteeredFrontier.pop_batch(obj)
    assert [n["node_id"] for n in batch] == [201]
    assert obj.frontier == []                                  # capped nodes gone, not retained
    assert 101 not in obj.node_embs and 102 not in obj.node_embs
    assert obj.expansions_per_root["2"] == 1                   # popped root incremented by 1


def test_add_julia_root_hook_spacing_and_durable_log(tmp_path):
    # item 2 + 3: a hook within JULIA_HOOK_SPACING of an already-hooked c is SKIPPED, and every
    # hook decision (accepted + skipped) is durably logged with its seed c.
    import types, json
    import numpy as np
    from collections import defaultdict
    obj = types.SimpleNamespace(
        julia_hook_spacing=0.20, hooked_c=defaultdict(list), batch_i=1,
        julia_hooks_path=tmp_path / "julia_hooks.jsonl",
        totals={"julia_roots": 0, "julia_hooks_skipped": 0},
        node_ctr=0, frontier=[], rng=np.random.default_rng(0),
    )
    obj.new_node_id = types.MethodType(sf.SteeredFrontier.new_node_id, obj)
    obj._log_julia_hook = types.MethodType(sf.SteeredFrontier._log_julia_hook, obj)
    add = types.MethodType(sf.SteeredFrontier.add_julia_root, obj)

    assert add("multibrot3", ("0.30", "-0.10"), "p0") is True     # first hook: accepted
    assert add("multibrot3", ("0.31", "-0.10"), "p1") is False    # within 0.20 of the first: skip
    assert add("multibrot3", ("0.90", "0.50"), "p2") is True      # distinct-c: accepted
    assert obj.totals == {"julia_roots": 2, "julia_hooks_skipped": 1}
    assert len(obj.hooked_c["julia:multibrot3"]) == 2             # only accepted c's tracked

    logged = [json.loads(l) for l in open(obj.julia_hooks_path, encoding="utf-8")]
    assert len(logged) == 3                                       # all decisions durably logged
    assert [r["hooked"] for r in logged] == [True, False, True]
    assert logged[0]["nearest_c_dist"] is None                    # first hook has no neighbour
    assert logged[1]["hooked"] is False and logged[1]["parent_oid"] == "p1"


def test_is_keeper_uses_corn_decode():
    cuts = {"mandelbrot": {"t": 0.5}}
    # p_notbad>=0.5 AND p_good>=0.5 -> keeper; either failing -> not.
    assert kc.is_keeper("mandelbrot", 0.9, 0.9, cuts) is True
    assert kc.is_keeper("mandelbrot", 0.9, 0.4, cuts) is False
    assert kc.is_keeper("mandelbrot", 0.4, 0.9, cuts) is False


# =========================================================================== #
# MorphMemory — the v1.2 novelty-memory fix (legacy vs recency semantics).
# =========================================================================== #
def _unit(i, d=768):
    import numpy as np
    v = np.zeros(d, np.float32); v[i] = 1.0
    return v


def test_morph_memory_legacy_is_all_permanent(tmp_path):
    # recency_k == 0: admitted AND expanded looks are permanent; end_batch is a no-op.
    import numpy as np
    m = sf.MorphMemory("cpu", tmp_path / "m.npz", recency_k=0)
    m.add_admitted(_unit(0))
    m.add_expanded(_unit(1))
    m.end_batch()                                   # no-op in legacy
    assert m.n_perm == 2 and m.n_recency == 0 and len(m) == 2
    # a perfect dup of either look reads cos_max ~ 1.
    cm = m.cos_max(np.stack([_unit(0), _unit(1), _unit(5)]))
    assert cm[0] > 0.999 and cm[1] > 0.999 and cm[2] < 1e-6


def test_morph_memory_recency_window_evicts(tmp_path):
    # recency_k == 2: admitted permanent; expanded looks live in a 2-batch rolling window.
    import numpy as np
    m = sf.MorphMemory("cpu", tmp_path / "m.npz", recency_k=2)
    m.add_admitted(_unit(0))                        # permanent
    m.add_expanded(_unit(1)); m.end_batch()         # block A = {e1}
    m.add_expanded(_unit(2)); m.end_batch()         # block B = {e2}
    # both windows + the admitted look are live here.
    cm = m.cos_max(np.stack([_unit(0), _unit(1), _unit(2), _unit(3)]))
    assert cm[0] > 0.999 and cm[1] > 0.999 and cm[2] > 0.999 and cm[3] < 1e-6
    m.add_expanded(_unit(3)); m.end_batch()         # block C = {e3}; e1's block evicted (>K)
    cm = m.cos_max(np.stack([_unit(0), _unit(1), _unit(2), _unit(3)]))
    assert cm[0] > 0.999                            # admitted look survives (permanent)
    assert cm[1] < 1e-6                             # e1 evicted from the window
    assert cm[2] > 0.999 and cm[3] > 0.999          # e2, e3 still in window
    assert m.n_perm == 1 and m.n_recency == 2


def test_morph_memory_current_batch_excluded_until_end(tmp_path):
    # A look expanded THIS batch is NOT visible to cos_max until end_batch finalizes it into a
    # block — comparing a candidate to its own just-expanded parent would trivially saturate.
    import numpy as np
    m = sf.MorphMemory("cpu", tmp_path / "m.npz", recency_k=4)
    m.add_expanded(_unit(7))                         # pending, not yet in the window
    assert m.cos_max(np.stack([_unit(7)]))[0] < 1e-6
    m.end_batch()                                    # finalized -> visible to the NEXT batch
    assert m.cos_max(np.stack([_unit(7)]))[0] > 0.999


def test_morph_memory_roundtrip_persists_window(tmp_path):
    # save() + reload preserves permanent + window blocks (so a resume evicts on the same K).
    import numpy as np
    p = tmp_path / "m.npz"
    m = sf.MorphMemory("cpu", p, recency_k=2)
    m.add_admitted(_unit(0))
    m.add_expanded(_unit(1)); m.end_batch()
    m.add_expanded(_unit(2)); m.end_batch()
    m.save()
    m2 = sf.MorphMemory("cpu", p, recency_k=2)
    assert m2.n_perm == 1 and m2.n_recency == 2
    cm = m2.cos_max(np.stack([_unit(0), _unit(1), _unit(2)]))
    assert cm[0] > 0.999 and cm[1] > 0.999 and cm[2] > 0.999


# =========================================================================== #
# tau_h derivation — decoupled from the disposable fidelity-records artifact.
# Fires on the absence of scratch/descent_score_fidelity_records.json so the launch-critical
# derivation surfaces in pytest, not only at a campaign launch.
# =========================================================================== #
def test_derive_tau_h_falls_back_to_vendored_base_when_records_absent(monkeypatch, tmp_path):
    # With the disposable records file gone, derive_tau_h must NOT SystemExit — it uses the
    # vendored base and still returns a floored tau_h for every production partition.
    # The version gate is satisfied explicitly here so THIS test stays about the fallback;
    # the gate itself is tested separately below.
    monkeypatch.setattr(sf, "FIDELITY_RECORDS", tmp_path / "records_do_not_exist.json")
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: sf.TAU_H_FIDELITY_BASE_MODEL)
    parts = list(sf.TAU_H_FIDELITY_BASE)
    tau = sf.derive_tau_h(parts)
    assert set(tau) == set(parts)
    # campaign floors are applied on top (max — only ever raise)
    for p, floor in sf.TAU_H_CAMPAIGN_FLOOR.items():
        assert tau[p] >= floor - 1e-12
    # a floor-free partition passes the vendored base through unchanged
    assert tau["julia:multibrot3"] == sf.TAU_H_FIDELITY_BASE["julia:multibrot3"]


def test_campaign_floor_mechanism_still_raises_when_a_floor_exists(monkeypatch, tmp_path):
    """The floor MECHANISM, tested independently of the (currently empty) live table.

    `TAU_H_CAMPAIGN_FLOOR` was emptied at the v8 flip — its values were cuts on v7's cheap
    p_good and cannot be re-derived until a v8-era campaign produces admissions. Keying this
    test on the live table would have deleted coverage of the mechanism along with the data,
    so it injects a floor instead: max-only, never lowering."""
    monkeypatch.setattr(sf, "FIDELITY_RECORDS", tmp_path / "records_do_not_exist.json")
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: sf.TAU_H_FIDELITY_BASE_MODEL)
    base = sf.TAU_H_FIDELITY_BASE["mandelbrot"]
    monkeypatch.setattr(sf, "TAU_H_CAMPAIGN_FLOOR",
                        {"mandelbrot": base + 0.1, "multibrot3": 0.0})
    tau = sf.derive_tau_h(["mandelbrot", "multibrot3"])
    assert tau["mandelbrot"] == base + 0.1                              # raised
    assert tau["multibrot3"] == sf.TAU_H_FIDELITY_BASE["multibrot3"]    # never lowered


def test_the_retired_v7_campaign_floor_is_not_on_the_live_path():
    # Kept for the record, never read. A v7 floor applied to a v8 base is the same category
    # error the version stamp exists to stop.
    assert sf.TAU_H_CAMPAIGN_FLOOR == {}
    assert sf.TAU_H_CAMPAIGN_FLOOR_V7_RETIRED                    # the record survives
    assert sf.TAU_H_CAMPAIGN_FLOOR_MODEL == sf.TAU_H_FIDELITY_BASE_MODEL


def test_vendored_tau_h_stamp_matches_the_active_checkpoint():
    """The stamp is the whole guard, so it must actually be current in the committed tree —
    otherwise every run aborts and the gate reads as broken rather than as protective."""
    import active_ckpt
    assert sf.TAU_H_FIDELITY_BASE_MODEL == active_ckpt.ACTIVE_VERSION, (
        f"vendored tau_h is stamped {sf.TAU_H_FIDELITY_BASE_MODEL!r} but the active scorer "
        f"is {active_ckpt.ACTIVE_VERSION!r} — re-run tools/atlas/tau_h_rederive.py and "
        f"update TAU_H_FIDELITY_BASE + TAU_H_FIDELITY_BASE_MODEL together")


def test_vendored_tau_h_matches_its_provenance_artifact():
    """The committed constants must be exactly what the re-derivation wrote — a hand-edited
    threshold is otherwise indistinguishable from a derived one."""
    art = Path(sf.ROOT) / "data" / "atlas" / f"tau_h_base_{sf.TAU_H_FIDELITY_BASE_MODEL}.json"
    assert art.exists(), f"{art} missing — the tau_h provenance must stay durable"
    doc = json.loads(art.read_text(encoding="utf-8"))
    assert doc["model"] == sf.TAU_H_FIDELITY_BASE_MODEL
    assert set(doc["tau_h_base"]) == set(sf.TAU_H_FIDELITY_BASE)
    for p, v in doc["tau_h_base"].items():
        assert float(v) == float(sf.TAU_H_FIDELITY_BASE[p]), p


def test_derive_tau_h_loud_fail_for_unvendored_partition_when_records_absent(monkeypatch, tmp_path):
    # A partition with neither a record-derived nor a vendored base aborts loudly, naming the
    # regenerator — immediately, not deep in a frontier run.
    import pytest
    monkeypatch.setattr(sf, "FIDELITY_RECORDS", tmp_path / "records_do_not_exist.json")
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: sf.TAU_H_FIDELITY_BASE_MODEL)
    with pytest.raises(SystemExit, match="descent_score_fidelity"):
        sf.derive_tau_h(["mandelbrot", "julia:bogus_unvendored"])


def test_vendored_tau_h_is_version_stamped_and_refuses_a_foreign_head(monkeypatch, tmp_path):
    """The vendored base must never quietly serve a stale value after a head change.

    tau_h is a cut on the CHEAP-render p_good of a specific scorer; nothing about a float says
    which model it describes, which is exactly why a vendored constant survives a version flip
    looking authoritative. The stamp + this gate make the mismatch fatal instead."""
    import pytest
    assert isinstance(sf.TAU_H_FIDELITY_BASE_MODEL, str) and sf.TAU_H_FIDELITY_BASE_MODEL
    monkeypatch.setattr(sf, "FIDELITY_RECORDS", tmp_path / "records_do_not_exist.json")
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: "v99_some_future_head")
    with pytest.raises(SystemExit, match="derived under"):
        sf.derive_tau_h(["mandelbrot"])


def test_live_fidelity_records_bypass_the_version_gate(monkeypatch, tmp_path):
    """A re-run study was run under the ACTIVE checkpoint by construction, so the live
    record-derived path is not gated — only the vendored fallback is. Otherwise re-deriving
    correctly would still be blocked by a stamp describing the constant it replaced."""
    recs = tmp_path / "records.json"
    recs.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sf, "FIDELITY_RECORDS", recs)
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: "v99_some_future_head")
    # an injected floor above the record-derived base, so the floor's raise is observable
    # independently of the (retired, empty) live table
    monkeypatch.setattr(sf, "TAU_H_CAMPAIGN_FLOOR", {"mandelbrot": 0.30})
    monkeypatch.setattr(sf, "_derive_tau_h_base_from_records",
                        lambda parts, keep: {p: 0.10 for p in parts})
    tau = sf.derive_tau_h(["mandelbrot", "julia:multibrot3"])
    assert tau["julia:multibrot3"] == 0.10                     # no gate, no raise
    assert tau["mandelbrot"] == 0.30                           # floor still applies


# =========================================================================== #
# The unseeded-run preflight — the "before any work" half of the guard.
#
# The scheduler-level guard is tested in test_deficit_scheduler.py; what MUST be pinned here
# is the ORDER: the driver's __init__ mkdirs the run dir and opens the ledger, so a guard that
# only fires inside the scheduler would already have written to disk. These assert the abort
# happens ahead of SteeredFrontier(...) entirely.
# =========================================================================== #
def _sched_args(tmp_path, **kw):
    from types import SimpleNamespace
    return SimpleNamespace(scheduler=True, resume=False, allow_unseeded=False,
                           run_dir=str(tmp_path / "run"), **kw)


def _no_library(monkeypatch, tmp_path):
    monkeypatch.setattr(sf.dsched, "INTAKE_ARTIFACT", tmp_path / "gone" / "intake.json")
    monkeypatch.setattr(sf.dsched, "INTAKE_EMB_DIR", tmp_path / "gone_embs")


def test_preflight_aborts_unseeded_and_touches_nothing(monkeypatch, tmp_path):
    import pytest
    _no_library(monkeypatch, tmp_path)
    args = _sched_args(tmp_path)
    with pytest.raises(SystemExit) as ei:
        sf.preflight_library_seed(args)
    msg = str(ei.value)
    assert "REFUSING TO START UNSEEDED" in msg
    assert str(tmp_path / "gone" / "intake.json") in msg      # names the missing path
    assert not (tmp_path / "run").exists()                    # nothing written


def test_preflight_is_a_noop_without_the_scheduler(monkeypatch, tmp_path):
    # Scoped to --scheduler only: the seed matters when the scheduler is allocating.
    _no_library(monkeypatch, tmp_path)
    args = _sched_args(tmp_path)
    args.scheduler = False
    assert sf.preflight_library_seed(args) is None            # no abort, no seed loaded


def test_preflight_override_proceeds_and_marks_the_record(monkeypatch, tmp_path):
    _no_library(monkeypatch, tmp_path)
    args = _sched_args(tmp_path)
    args.allow_unseeded = True
    rec = sf.preflight_library_seed(args)
    assert rec["status"] == "unseeded" and rec["allow_unseeded"] is True
    assert args._library_seed is rec                          # rides on args for the driver


def test_preflight_skips_a_resume_with_a_tally_on_disk(monkeypatch, tmp_path):
    # A resumed distinct-look tally IS the seed; a since-moved artifact must not abort it.
    _no_library(monkeypatch, tmp_path)
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "distinct_looks.npz").write_bytes(b"")
    args = _sched_args(tmp_path)
    args.resume = True
    assert sf.preflight_library_seed(args) is None


def test_main_runs_the_preflight_before_constructing_the_driver(monkeypatch, tmp_path):
    """Order proof: with no seed, main() must exit WITHOUT ever instantiating SteeredFrontier
    (whose __init__ creates the run dir and the ledger). A sentinel records instantiation."""
    import pytest
    _no_library(monkeypatch, tmp_path)
    built = []

    class _Sentinel:
        def __init__(self, args):
            built.append(args)

        def run(self):
            built.append("ran")

    monkeypatch.setattr(sf, "SteeredFrontier", _Sentinel)
    monkeypatch.setattr(sys, "argv", ["steered_frontier.py",
                                      "--run-dir", str(tmp_path / "run"), "--scheduler"])
    with pytest.raises(SystemExit):
        sf.main()
    assert built == []                                        # never constructed, never ran
    assert not (tmp_path / "run").exists()


# =========================================================================== #
# Per-unit reconcile + the wall-clock hard-kill backstop (the unattended-run guards).
# =========================================================================== #
def _recon_obj(totals):
    import types
    o = types.SimpleNamespace(totals=dict(totals), batch_i=7)
    o._reconcile_snapshot = types.MethodType(sf.SteeredFrontier._reconcile_snapshot, o)
    o._reconcile_batch = types.MethodType(sf.SteeredFrontier._reconcile_batch, o)
    o.RECONCILE_KEYS = sf.SteeredFrontier.RECONCILE_KEYS
    return o


_ZERO = {k: 0 for k in sf.SteeredFrontier.RECONCILE_KEYS}


def test_reconcile_passes_when_every_check_lands_in_exactly_one_fate():
    o = _recon_obj(_ZERO)
    before = o._reconcile_snapshot()
    o.totals.update(candidates=10, frontier_pushed=10, harvest_checks=6,
                    precanon_dup=2, canonical_q3=3, canon_not_q3=1,
                    admitted=1, q3_dup=1, guarded=1, reframe_not_q3=0)
    o._reconcile_batch(before, 10)          # closes: 6 == 2+3+1 and 3 == 1+1+1+0


def test_reconcile_exits_loud_on_a_lost_candidate():
    o = _recon_obj(_ZERO)
    before = o._reconcile_snapshot()
    o.totals.update(candidates=10, frontier_pushed=9)     # one candidate vanished
    with pytest.raises(SystemExit) as e:
        o._reconcile_batch(before, 10)
    assert "frontier" in str(e.value)


def test_reconcile_exits_loud_on_an_uncounted_q3_fate():
    # The historical gap: guard passed but the REFRAMED frame decoded below q3 and nothing
    # counted it, so the q3 identity could not close.
    o = _recon_obj(_ZERO)
    before = o._reconcile_snapshot()
    o.totals.update(candidates=1, frontier_pushed=1, harvest_checks=1, canonical_q3=1)
    with pytest.raises(SystemExit) as e:
        o._reconcile_batch(before, 1)
    assert "q3 fates" in str(e.value)


def test_unit_timeout_is_clamped_by_the_remaining_budget():
    import types
    # No budget -> the historical 900s standing backstop, unchanged.
    o = types.SimpleNamespace(budget_s=0.0, active_s=0.0)
    assert sf.SteeredFrontier.unit_timeout_s(o) == float(sf.EXPAND_TIMEOUT_S)
    # 15-minute budget, nothing spent: the backstop is the run, never longer than it.
    o = types.SimpleNamespace(budget_s=900.0, active_s=0.0)
    assert sf.SteeredFrontier.unit_timeout_s(o) == 900.0
    # Near the end of the budget the backstop shrinks with it ...
    o = types.SimpleNamespace(budget_s=900.0, active_s=880.0)
    assert sf.SteeredFrontier.unit_timeout_s(o) == sf.MIN_UNIT_TIMEOUT_S
    # ... but never below the floor, so a legitimately slow last unit is not shot for
    # being slow (this is the branch that keeps the clamp from becoming a zero timeout).
    o = types.SimpleNamespace(budget_s=900.0, active_s=100000.0)
    assert sf.SteeredFrontier.unit_timeout_s(o) == float(sf.MIN_UNIT_TIMEOUT_S)
