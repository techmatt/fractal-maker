#!/usr/bin/env python
"""Unit tests for the steered-frontier v1.1 priority terms + the keeper cut.

Pure / fast — no render, no GPU, no binary. The control for the two new priority terms
(morph-novelty + depth) and the acceptance guarantee that BOTH coefficients at zero
reproduce the pilot priority exactly, plus the F0.5 keeper-cut metric math.

  uv run pytest tools/atlas/test_steered_frontier.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_record             # noqa: E402  (the segmented run-record reader)
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


@pytest.mark.version_pinned
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


@pytest.mark.version_pinned
def test_keeper_positive_is_label_ge_3_not_eq_3():
    # Under v8's 1..4 labels a class-4 location is the BEST kind of keeper. Scoring it as a
    # negative would push the precision-weighted cut in exactly the wrong direction, and the
    # v7-era `label == 3` predicate did precisely that once class 4 existed.
    rows = kc.read_jsonl(kc.EVAL)
    assert any(r["label"] == 4 for r in rows), "no class-4 rows — this test would be vacuous"
    triples = kc.load_triples()
    n_pos = sum(1 for rs in triples.values() for _, _, pos in rs if pos)
    # Counted over the SAME population `load_triples` cut, not over the whole slice: since
    # v11 that population is the grouped holdout with an instrument fallback, so "every row
    # whose fractal_type maps" is a different and larger set. The invariant under test is the
    # PREDICATE (`>= 3`, not `== 3`), and it is only visible if both sides count the same rows.
    kept, _ = kc._select_population(rows, instrument=kc.INSTRUMENT)         if all("eval_role" in r for r in rows) else (rows, None)
    n_ge3 = sum(1 for r in kept if r["label"] >= 3)
    assert n_pos == n_ge3 > sum(1 for r in kept if r["label"] == 3)


@pytest.mark.version_pinned
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

    # (2) partition coverage — the live set is `partitions.ALL_FAMS` (derived from code, not
    # hardcoded, so adding a family to the derivation without recutting keeper_cuts.json fails
    # loudly here). ALL_FAMS, not `FT2FAM.values()`: a DERIVED partition has no fractal_type of
    # its own, so the value list silently omitted `phoenix:classic` — which then read as "not a
    # partition" rather than "uncalibrated". Fixed 2026-08-08 with the v11 recut, which is the
    # first slice to carry rows for it.
    live = set(kc.ALL_FAMS)
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


def _push_children_node(cand, **over):
    """Drive `push_children` on ONE candidate and return the frontier node it rebuilt.

    The subject is exactly the node-rebuild in `push_children`; everything it touches
    besides the frontier append (morph memory, the prio/saturation logs, the prune) is
    stubbed to a no-op so a failure here can only mean the rebuild dropped a field."""
    import types
    import numpy as np
    obj = types.SimpleNamespace(
        frontier=[], node_embs={}, clouds={}, rng=np.random.default_rng(0),
        lambda_m=0.0, beta=0.0, morph_lo=0.85, morph_hi=0.974, sat_cos=0.9666,
        batch_i=3, totals=collections.Counter(),
        prio_log=over.pop("prio_log"), sat_log=over.pop("sat_log"),
        _stream_writers={},
    )
    obj.prune_frontier = types.MethodType(lambda s: None, obj)
    # The prio log goes through the REAL `_writer` (run_record.SegmentWriter), not a
    # stub: it is the append path the stub is standing in front of, and a fake that
    # skipped it would let the rebuild write to a stream nothing rotates.
    obj._writer = types.MethodType(sf.SteeredFrontier._writer, obj)
    sf.SteeredFrontier.push_children(obj, [cand])
    assert len(obj.frontier) == 1
    return obj.frontier[0]


def _cand(**kw):
    base = dict(node_id=7, root_id=2, partition="multibrot3", c=None,
                cx=0.1, cy=0.2, fw=1e-3, depth=4, branch="policy",
                cheap_eord=1.0, cheap_pgood=0.4, cos_max=0.0, emb=None,
                mix_source="triggered:snap:k=16", man={"op": "snap", "k": 16.0},
                triggered=True, phoenix={"branch": "A", "theta": 0.5})
    base.update(kw)
    return base


@pytest.mark.parametrize("field,value", [("triggered", True),
                                         ("phoenix", {"branch": "A", "theta": 0.5})])
def test_push_children_carries_the_lineage_stamps_onto_the_rebuilt_node(tmp_path, field, value):
    """RED BY CONSTRUCTION on the pre-fix code: `push_children` rebuilds the frontier node
    from scratch, so a stamp it does not name is dropped — and `expand_group` reads these
    back OFF THE NODE to stamp the next generation. The stamp therefore does not go missing
    once; it truncates at generation 1 and every descendant is written with the default,
    which is a positive claim about the wrong population.

    Both fields are asserted through one parametrize so adding a third stamp to the node is
    one line here, not a new test that someone forgets to write.

    (Measured on q4_long_harvest_20260803, the run this fix is owed to: 616 of 794
    triggered-lineage rows and 1,221 of 1,238 phoenix rows were written stamp-less.)"""
    node = _push_children_node(_cand(), prio_log=tmp_path / "prio_terms.jsonl",
                               sat_log=tmp_path / "s.jsonl")
    assert node[field] == value


def test_push_children_does_not_invent_a_stamp_the_candidate_never_had(tmp_path):
    """The other side of the straddle. Without this, a rebuild that hardcoded
    `triggered=True` would pass the test above — and a FRESH descendant mislabelled
    triggered corrupts the split in the opposite direction, where it is harder to see
    because the triggered arm is the small one."""
    node = _push_children_node(_cand(triggered=None, phoenix=None, mix_source="sampler",
                                     man=None),
                               prio_log=tmp_path / "prio_terms.jsonl", sat_log=tmp_path / "s.jsonl")
    assert not node["triggered"] and node["phoenix"] is None


def test_expand_group_reads_the_stamps_back_off_the_node():
    """Why the test above is about the NODE and not the candidate: the round trip closes
    through `expand_group`, which stamps a child from `parent.get(...)`. Pinned by source
    inspection rather than by spawning the engine — the assertion is that the two halves
    name the SAME key set, which is the thing that silently drifts."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.expand_group)
    for field in ("triggered", "phoenix", "man", "mix_source"):
        assert f'parent.get("{field}")' in src, field


# =========================================================================== #
# per-partition root low-water — the supply half of the allocator.
#
# The scenario every test here reproduces is arm B of `allocator_prereg_v1`: a 944-node
# frontier that is 97% one partition, with the other eight queues empty and `draw_roots`
# firing zero times in-loop because the GLOBAL low-water (B = 32) never saw a frontier under
# 32. The allocator's intent was correct and unservable.
# =========================================================================== #
def _refill_obj(frontier, families, *, drawn=None, low_water=32, cooldown=10,
                share=0.25, batch_i=100, active_s=600.0, root_draw_s=0.0):
    """The minimum object the refill path touches. `draw_roots` is REPLACED by a recorder —
    the native seeder spawns the engine, and what is under test is which families get asked
    and whether they get asked at all."""
    import types
    obj = types.SimpleNamespace(
        frontier=list(frontier), families=list(families), batch_i=batch_i,
        partition_low_water=low_water, root_refill_cooldown=cooldown,
        root_refill_share=share, root_draw_s=root_draw_s, active_s=active_s,
        last_refill_batch={}, totals={})
    calls = [] if drawn is None else drawn
    obj.draw_roots = lambda only=None: (calls.append(list(only) if only else None), 7)[1]
    obj._draw_calls = calls
    obj.partitions = list(dict.fromkeys(
        list(families) + [n["partition"] for n in frontier]))
    obj.REFILL_DEFERRAL = sf.SteeredFrontier.REFILL_DEFERRAL
    for m in ("queue_lens", "starved_families", "refill_affordable", "refill_starved",
              "deferred_partitions"):
        setattr(obj, m, types.MethodType(getattr(sf.SteeredFrontier, m), obj))
    return obj


def _armb_frontier():
    """944 nodes, 862 of them one partition — arm B at b381, to the node."""
    f = [{"node_id": i, "partition": "julia:mandelbrot"} for i in range(862)]
    f += [{"node_id": 900 + i, "partition": "mandelbrot"} for i in range(82)]
    return f


def test_the_global_low_water_is_blind_to_the_collapse_the_fix_is_for():
    """OLD BEHAVIOUR WAS WRONG — asserted against the real numbers rather than described.
    A 944-node frontier is nowhere near a 32-node global mark, so the v1 condition is False
    while three of four families hold zero nodes."""
    frontier = _armb_frontier()
    assert len(frontier) == 944
    assert not (len(frontier) < 32), "the global low-water cannot fire on this frontier"
    obj = _refill_obj(frontier, ["mandelbrot", "multibrot3", "multibrot4", "multibrot5"])
    q = obj.queue_lens()
    assert q.get("multibrot3", 0) == 0 and q.get("multibrot4", 0) == 0
    assert q["julia:mandelbrot"] / len(frontier) > 0.9


def test_the_per_partition_mark_fires_on_exactly_the_starved_families():
    """NEW BEHAVIOUR IS RIGHT, and does not over-correct: `mandelbrot` holds 82 nodes and is
    NOT redrawn, so the refill is targeted rather than a full draw wearing a new name."""
    obj = _refill_obj(_armb_frontier(),
                      ["mandelbrot", "multibrot3", "multibrot4", "multibrot5"])
    assert obj.refill_starved() == 7
    assert obj._draw_calls == [["multibrot3", "multibrot4", "multibrot5"]]
    assert obj.totals["root_refills"] == 1 and obj.totals["root_refill_families"] == 3


def test_a_healthy_frontier_draws_nothing():
    """Non-vacuity from the other end: the fixture can fail. Every family above the mark ->
    no draw at all, so a refill that fired unconditionally would not pass."""
    frontier = [{"node_id": i, "partition": p}
                for p in ("mandelbrot", "multibrot3") for i in range(40)]
    obj = _refill_obj(frontier, ["mandelbrot", "multibrot3"])
    assert obj.refill_starved() == 0 and obj._draw_calls == []
    assert obj.totals.get("root_refills", 0) == 0


def test_julia_twins_and_phoenix_are_never_asked_for_a_root_draw():
    """SCOPE, and it is a correctness property: `draw_roots` is the c-plane native seeder, so
    a starved julia twin routed to it would produce a refill request no draw can serve — a
    silent no-op consuming the cooldown and the share budget every time it fired."""
    frontier = [{"node_id": i, "partition": "mandelbrot"} for i in range(100)]
    obj = _refill_obj(frontier, ["mandelbrot"])
    obj.frontier += [{"node_id": 500, "partition": "julia:mandelbrot"}]
    assert obj.starved_families() == []          # julia:mandelbrot at 1 node is NOT requested
    assert obj.refill_starved() == 0


def test_a_starved_partition_the_refill_will_not_serve_is_REPORTED_with_its_reason():
    """The other half of the scope statement above. `starved_families` correctly omits every
    non-c-plane partition, but an omitted starved partition and a healthy one read identically
    in a run record — which is exactly how arm B reported `root_refills=0` with eight empty
    queues.

    `phoenix:classic` is NOT in this report (2026-08-07): it is EXTERNALLY SUPPLIED, so an
    empty queue is its normal state rather than a starvation — see `deferred_partitions`
    SKIP SITE 1 and the companion assertions below."""
    frontier = [{"node_id": i, "partition": "mandelbrot"} for i in range(100)]
    frontier += [{"node_id": 500, "partition": "julia:mandelbrot"},
                 {"node_id": 501, "partition": "phoenix"},
                 {"node_id": 502, "partition": "phoenix:classic"}]
    obj = _refill_obj(frontier, ["mandelbrot"])
    assert obj.starved_families() == []                 # nothing is asked for a draw...
    d = obj.deferred_partitions()                        # ...and the crawl-fed ones are reported
    assert set(d) == {"julia:mandelbrot", "phoenix"}
    assert d["phoenix"]["queue"] == 1 and d["phoenix"]["low_water"] == 32
    assert "phoenix-seed-pool" in d["phoenix"]["reason"]
    assert "julia-hook" in d["julia:mandelbrot"]["reason"]
    # a HEALTHY partition is never reported (non-vacuity: the report can be empty)
    assert "mandelbrot" not in d
    assert _refill_obj([{"node_id": i, "partition": "phoenix"} for i in range(40)],
                       ["mandelbrot"]).deferred_partitions() == {}


def test_an_externally_supplied_partition_is_not_censused_as_starved(monkeypatch):
    """SKIP SITE 1 of 3. `phoenix:classic` has one channel and it is a standalone job, so an
    empty queue is not news — reporting it every batch is a permanent false alarm on the one
    dict whose job is to make a real empty queue loud.

    The CONTROL is the same partition with the flag off: it must come straight back, so the
    green here is the flag being honoured and not `phoenix:classic` being special-cased."""
    frontier = [{"node_id": i, "partition": "mandelbrot"} for i in range(100)]
    frontier += [{"node_id": 502, "partition": "phoenix:classic"}]
    obj = _refill_obj(frontier, ["mandelbrot"])
    assert "phoenix:classic" not in obj.deferred_partitions()

    monkeypatch.setattr(sf.srt, "is_externally_supplied", lambda p: False)
    d = _refill_obj(frontier, ["mandelbrot"]).deferred_partitions()
    assert set(d) == {"phoenix:classic"}
    assert "PINNED" in d["phoenix:classic"]["reason"]
    assert "classic_plane_descent" in d["phoenix:classic"]["reason"]


def test_a_phoenix_seed_pool_routes_each_point_to_its_own_partition():
    """The split is DERIVED from the parameter point, at the reader, in the one place that
    labels a phoenix node — so the tracked partition set and the nodes produced cannot
    disagree. Pinned Ushiki point -> `phoenix:classic`; anything else -> `phoenix`."""
    classic = ("0.5667", "0", "-0.5", "0", "0", "0")
    varied = ("-1.089", "0.481", "-0.222", "0.172", "-0.224", "-0.347")
    assert sf.partition_for_phoenix_c(classic) == "phoenix:classic"
    assert sf.partition_for_phoenix_c(varied) == "phoenix"
    # one ulp off the pinned p is NOT classic — exact equality, no tolerance
    import math
    near = ("0.5667", "0", repr(math.nextafter(-0.5, 0.0)), "0", "0", "0")
    assert sf.partition_for_phoenix_c(near) == "phoenix"
    # and both normalize back to the same render family / walk grammar
    assert sf.render_family_of("phoenix:classic") == sf.render_family_of("phoenix") == "phoenix"
    assert sf.descend_flags("phoenix:classic", classic) == sf.descend_flags("phoenix", classic)
    assert sf.render_args_for("phoenix:classic", classic)["family"] == "phoenix"


def test_the_tracked_phoenix_partitions_are_read_off_the_pool_not_declared(tmp_path):
    """A partition the pool cannot feed would hold a permanent 5% quota floor it can never
    spend; a partition the pool DOES feed but nobody tracks gets nodes with no cloud, no
    tau_h and no quota row. So the set is derived from the pool, and it is the SAME function
    that labels the nodes."""
    import types
    def pool_parts(entries):
        p = tmp_path / f"pool_{len(entries)}_{entries[0]['p_re']}.json"
        p.write_text(json.dumps(entries), encoding="utf-8")
        obj = types.SimpleNamespace(phoenix_seed_pool_path=p)
        obj.phoenix_pool_partitions = types.MethodType(
            sf.SteeredFrontier.phoenix_pool_partitions, obj)
        return obj.phoenix_pool_partitions()

    C = dict(c_re="0.5667", c_im="0", p_re="-0.5", p_im="0", zm1_re="0", zm1_im="0")
    V = dict(c_re="-1.089", c_im="0.481", p_re="-0.222", p_im="0.172",
             zm1_re="-0.224", zm1_im="-0.347")
    assert pool_parts([V]) == ["phoenix"]
    assert pool_parts([C]) == ["phoenix:classic"]
    # mixed pool -> both, in ALL_FAMS order (base before derived), never duplicated
    assert pool_parts([C, V, dict(V, p_im="0.3")]) == ["phoenix", "phoenix:classic"]
    # a pool entry with no zm1 axes defaults them to the legacy z_-1=0, exactly as the node
    # builder does — so an old pool file keeps routing where it always did
    assert pool_parts([{k: v for k, v in C.items() if not k.startswith("zm1")}]) \
        == ["phoenix:classic"]


def test_the_cooldown_stops_a_barren_family_being_redrawn_every_batch():
    """A family whose depth-2 probe survives nothing stays starved, so without a cooldown it
    is redrawn every batch at ~2 min a draw against a ~0.34 min batch — the run becomes a
    root-draw loop. Second batch: refused. After the cooldown: allowed again."""
    frontier = [{"node_id": i, "partition": "mandelbrot"} for i in range(100)]
    obj = _refill_obj(frontier, ["mandelbrot", "multibrot3"], cooldown=10, batch_i=100)
    assert obj.refill_starved() == 7 and obj._draw_calls == [["multibrot3"]]
    for b in range(101, 110):
        obj.batch_i = b
        assert obj.refill_starved() == 0, f"b{b} is inside the cooldown"
    assert len(obj._draw_calls) == 1
    obj.batch_i = 110
    assert obj.refill_starved() == 7
    assert obj._draw_calls == [["multibrot3"], ["multibrot3"]]


def test_the_share_cap_defers_a_refill_and_counts_it():
    """A cooldown spaces retries; it does not bound their total. The share cap does, and a
    deferral is COUNTED — a refill that silently did not happen reads exactly like a frontier
    that was never starved."""
    frontier = [{"node_id": i, "partition": "mandelbrot"} for i in range(100)]
    # 400 s of root draws against 600 s of active work = 40% of loop wall, over a 25% cap.
    obj = _refill_obj(frontier, ["mandelbrot", "multibrot3"],
                      active_s=600.0, root_draw_s=400.0, share=0.25)
    assert obj.refill_affordable() is False
    assert obj.refill_starved() == 0 and obj._draw_calls == []
    assert obj.totals["root_refill_deferred"] == 1
    # ...and it re-opens once the run has mined enough to afford it again.
    obj.active_s = 3000.0
    assert obj.refill_affordable() is True
    assert obj.refill_starved() == 7


def test_the_first_refill_of_a_run_is_always_affordable():
    """The share is defined against total loop wall, not against `active_s`, so it is
    well-defined at zero. A cap that blocked the first refill would make the fix unreachable
    on exactly the run that starts starved."""
    obj = _refill_obj([], ["mandelbrot"], active_s=0.0, root_draw_s=0.0)
    assert obj.refill_affordable() is True


def test_the_loop_calls_the_per_partition_refill_at_the_replenishment_seam():
    """The tests above drive the mechanism; this one asserts the LOOP reaches it, so they
    cannot all pass against a run() that still only checks the global mark."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.run)
    assert "self.refill_starved()" in src
    assert "len(self.frontier) < ROOT_LOW_WATER" in src, "the global emergency draw stays"


def test_draw_roots_restricted_to_a_subset_reallocates_over_that_subset():
    """A scheduler-driven refill must recompute `root_allocation` over the STARVED families.
    Slicing the full allocation would hand a starved family its proportional share of B — a
    rounding error — which is the starvation the refill exists to end."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.draw_roots)
    assert "root_allocation(fams" in src
    assert "for fam in fams:" in src


# =========================================================================== #
# pop quota — the driver seam (the allocator's own arithmetic is in test_pop_quota.py)
# =========================================================================== #
def _quota_obj(tmp_path, frontier, currency):
    """The minimum object `pop_batch_quota` touches, wired to a real PopQuota."""
    import types
    import pop_quota as pq
    cen = pq.CurrencyCensus(counts={}, currency=currency, defaulted_rows=0, sources={},
                            partitions=list(currency))
    # EQUAL ratios: these are synthetic partition names with no release-mix entry, and this
    # block is about the DRIVER SEAM, not the mix policy. Equal ratios reproduce the
    # pre-2026-08-04 uniform target exactly, so what these tests assert is unchanged.
    q = pq.PopQuota(list(currency), tmp_path, census=cen,
                    prices_config=dict(cap_minutes=1e9),
                    ratios={p: 1.0 for p in currency})
    obj = types.SimpleNamespace(
        B=2, maneuvers=False, man_quota=0, quota=q, batch_i=1,
        expansions_per_root={}, node_embs={}, totals={"cap_hits": 0},
        frontier=list(frontier), partitions=list(currency), _served_partition=None)
    obj._split_reserved = types.MethodType(sf.SteeredFrontier._split_reserved, obj)
    return obj, q


def test_pop_batch_quota_serves_the_partition_furthest_below_its_intent(tmp_path):
    """The seam, end to end: `poor` carries all the deficit, so it is popped even though
    `rich` sits at the head of a pooled priority sort. A regression here is precisely the v1
    failure — an allocator that names a partition and a pop that ignores it."""
    frontier = [
        {"node_id": 1, "root_id": 1, "partition": "rich", "priority": 9.9},
        {"node_id": 2, "root_id": 2, "partition": "poor", "priority": 0.1},
        {"node_id": 3, "root_id": 3, "partition": "poor", "priority": 0.2},
    ]
    obj, q = _quota_obj(tmp_path, frontier, {"rich": 100.0, "poor": 0.0})
    batch = sf.SteeredFrontier.pop_batch_quota(obj)
    assert {n["partition"] for n in batch} == {"poor"}
    assert obj._served_partition == "poor"
    assert [n["node_id"] for n in batch] == [3, 2]        # priority order WITHIN the partition
    assert [n["node_id"] for n in obj.frontier] == [1]


def test_pop_batch_quota_returns_to_the_rich_partition_once_the_poor_one_is_served(tmp_path):
    """The floor doing its job across two pops. After `poor` has taken enough realized time
    to overshoot its intent, the next pop goes to `rich` — which is what a 5% floor on a
    zero-deficit partition means operationally."""
    frontier = [{"node_id": i, "root_id": i, "partition": p, "priority": 1.0}
                for i, p in enumerate(["rich", "poor", "rich", "poor"])]
    obj, q = _quota_obj(tmp_path, frontier, {"rich": 100.0, "poor": 0.0})
    sf.SteeredFrontier.pop_batch_quota(obj)
    q.charge("poor", 100.0)                               # poor now holds 100% of the time
    obj.batch_i = 2
    batch = sf.SteeredFrontier.pop_batch_quota(obj)
    assert {n["partition"] for n in batch} == {"rich"}


def test_pop_batch_quota_evicts_capped_root_dead_weight(tmp_path):
    """Inherited from `pop_batch`: a capped root's nodes must leave the frontier, or they
    accumulate faster than they drain and eventually starve every partition."""
    frontier = [{"node_id": 1, "root_id": 1, "partition": "a", "priority": 5.0},
                {"node_id": 2, "root_id": 2, "partition": "a", "priority": 1.0}]
    obj, q = _quota_obj(tmp_path, frontier, {"a": 0.0})
    obj.expansions_per_root = {"1": sf.M_CAP, "2": 0}
    obj.node_embs = {1: None, 2: None}
    batch = sf.SteeredFrontier.pop_batch_quota(obj)
    assert [n["node_id"] for n in batch] == [2] and 1 not in obj.node_embs


def test_pop_batch_quota_logs_every_choice_with_its_bucket(tmp_path):
    frontier = [{"node_id": 1, "root_id": 1, "partition": "a", "priority": 1.0}]
    obj, q = _quota_obj(tmp_path, frontier, {"a": 0.0, "b": 5.0})
    sf.SteeredFrontier.pop_batch_quota(obj)
    rec = run_record.read_rows(tmp_path / "quota_trace.jsonl")[0]
    assert rec["chosen"] == "a" and rec["queue_lens"] == {"a": 1, "b": 0}


def test_the_two_allocators_refuse_to_coexist(tmp_path, monkeypatch):
    """Two owners of the pop is two mixes and no readable number. Constructing both is a hard
    exit, not a precedence rule — a precedence rule is how a run silently uses the allocator
    nobody meant to enable."""
    import types
    args = types.SimpleNamespace(scheduler=True, pop_quota=True)
    obj = types.SimpleNamespace(scheduler=object(), partitions=["a"], run_dir=tmp_path)
    with pytest.raises(SystemExit, match="both name the pop"):
        # the guard is the first thing the --pop-quota branch does
        if getattr(args, "pop_quota", False):
            if obj.scheduler is not None:
                raise SystemExit("--pop-quota and --scheduler both name the pop; pick one "
                                 "(--pop-quota is the harvest-v2 allocator)")


def test_the_mutual_exclusion_guard_is_actually_in_the_constructor():
    """The test above rehearses the guard; this one asserts the guard EXISTS at the seam,
    so the rehearsal cannot pass against a constructor that lost it."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.__init__)
    assert "both name the pop" in src and 'getattr(args, "pop_quota"' in src


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


@pytest.mark.version_pinned
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
    """Point the seed REGISTRY at nothing.

    Patching `INTAKE_ARTIFACT` / `INTAKE_EMB_DIR` is no longer sufficient and the failure was
    silent-green in the dangerous direction: `SEED_SOURCES` is built from those constants at
    import time, so a patched constant left the registry resolving to the REAL relit seed and
    the "aborts unseeded" tests stopped testing an unseeded run. The registry is the
    authority, so the fixture patches the authority."""
    monkeypatch.setattr(sf.dsched, "SEED_SOURCES",
                        (("test_absent", tmp_path / "gone" / "intake.json",
                          tmp_path / "gone_embs"),))


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


# =========================================================================== #
# set_below_normal_priority — the call must actually SUCCEED, and a failure must
# be reportable.
# =========================================================================== #
@pytest.mark.skipif(sys.platform != "win32", reason="win32 priority path")
def test_below_normal_priority_actually_succeeds_and_restores():
    """The pseudo-handle regression, pinned. `GetCurrentProcess` returns (HANDLE)-1; without
    `restype = c_void_p` ctypes truncates it to a 32-bit int and `SetPriorityClass` rejects
    the call — so the driver printed a failure (or worse, "err 0") while every child kept
    running at NORMAL and stole the desktop. Assert the real return value, not the plumbing.

    Restores the original class so a test run does not leave the pytest process demoted."""
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k32.GetPriorityClass.restype = ctypes.c_uint
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k32.SetPriorityClass.restype = ctypes.c_int
    before = k32.GetPriorityClass(k32.GetCurrentProcess())
    try:
        assert sf.set_below_normal_priority() == "BELOW_NORMAL"
        import corpus_common as cc
        assert k32.GetPriorityClass(k32.GetCurrentProcess()) == cc.BELOW_NORMAL_PRIORITY_CLASS
    finally:
        if before:
            k32.SetPriorityClass(k32.GetCurrentProcess(), before)


@pytest.mark.skipif(sys.platform != "win32", reason="win32 priority path")
def test_the_failure_branch_can_report_a_real_error_code():
    """`ctypes.windll.kernel32` is a cached library object built WITHOUT use_last_error, so
    on it `ctypes.get_last_error()` always reads 0 and the helper's failure branch could only
    ever print "FAILED (err 0)" — a silent failure wearing a report. The helper now opens its
    own `WinDLL(..., use_last_error=True)`; this pins that that is what makes an error code
    observable, by forcing a genuine failure on an invalid handle."""
    import ctypes
    shared = ctypes.windll.kernel32
    shared.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    shared.SetPriorityClass.restype = ctypes.c_int
    ctypes.set_last_error(0)
    assert shared.SetPriorityClass(ctypes.c_void_p(0), 0x00004000) == 0     # fails
    assert ctypes.get_last_error() == 0, "the cached windll DID populate get_last_error"

    private = ctypes.WinDLL("kernel32", use_last_error=True)
    private.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    private.SetPriorityClass.restype = ctypes.c_int
    assert private.SetPriorityClass(ctypes.c_void_p(0), 0x00004000) == 0    # fails
    assert ctypes.get_last_error() == 6   # ERROR_INVALID_HANDLE


PRIORITY_HELPER_HOME = "tools/corpus/corpus_common.py"


def test_the_priority_helper_is_not_duplicated_anywhere():
    """The loose end this closes: if the ctypes priority path had been COPIED into other
    drivers, each copy would have carried the truncation and every one of their run records'
    "BELOW_NORMAL" claims would have been false. There is exactly one definition, and every
    other BELOW_NORMAL site goes through `creationflags` on subprocess (a CreateProcess flag,
    which never involved a handle). Keep it that way: share this function, don't re-derive it.

    The one definition MOVED (2026-08-03) from this driver to `corpus_common`, beside the rest
    of the same pairing (`BELOW_NORMAL_PRIORITY_CLASS`, `default_creationflags`,
    `DEFAULT_ENGINE_THREADS`), when a second driver needed it — `sitting_cutter` copied it
    first, this test caught the copy, and the copy did carry the truncation exactly as
    predicted. `steered_frontier.set_below_normal_priority` is now a delegating wrapper, so
    what this scan asserts is unchanged: ONE file contains the ctypes call."""
    import subprocess
    repo = Path(__file__).resolve().parents[2]
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.split()
    self_rel = Path(__file__).resolve().relative_to(repo).as_posix()
    hits = []
    for rel in files:
        if rel == self_rel:
            continue
        t = (repo / rel).read_text(encoding="utf-8", errors="ignore")
        if "SetPriorityClass" in t:
            hits.append(rel)
    assert hits == [PRIORITY_HELPER_HOME], (
        "SetPriorityClass appears outside the one helper — a copy carries the "
        f"pseudo-handle truncation with it: {hits}")


def test_every_caller_reaches_the_one_definition():
    """The scan above is satisfied by a helper nobody can reach. Both live entry points must
    resolve to the SAME function object, or one driver is quietly running at NORMAL."""
    import corpus_common as cc
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas"))
    import sitting_cutter as sc  # noqa: F401  (its call sites go through corpus_common)
    assert sf.set_below_normal_priority.__module__ != cc.set_below_normal_priority.__module__
    assert sf.set_below_normal_priority() == cc.set_below_normal_priority()
    src = Path(__file__).resolve().parents[1] / "atlas" / "sitting_cutter.py"
    assert "cc.set_below_normal_priority()" in src.read_text(encoding="utf-8")


# =========================================================================== #
# The julia c-supply pool: wired to v3, and the floor enforced at LOAD.
#
# `seed_julia_pool` bypasses the hook-spacing gate by design (the pool is far denser than
# the 0.2 hook radius), and nothing replaced that gate — so before this, the c-spacing a run
# applied was whatever the passed FILE carried, with no check anywhere and nothing in the run
# config recording it. harvest_v2_proving_20260803 ran on v2, i.e. the superseded 1e-2 floor.
# =========================================================================== #
def _pool_loader(tmp_path, rows):
    """Drive `load_julia_supply_pool` off a stub rather than a constructed SteeredFrontier:
    the method reads exactly two attributes, and building a real frontier would drag a
    scorer, a ledger and a run dir into a test about a closest-pair check."""
    import types
    p = tmp_path / "pool.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    stub = types.SimpleNamespace(julia_seed_pool_path=p, _julia_pool_cache=None)
    return stub, lambda: sf.SteeredFrontier.load_julia_supply_pool(stub)


def test_the_default_julia_supply_is_the_pool_built_at_the_current_floor():
    """v3 was built, committed, and loaded by NOTHING. The default is the wiring."""
    import build_julia_supply_pool_v2 as b
    import supply_routing as sr
    assert sf.JULIA_SUPPLY_POOL == sf.ROOT / b.POOL_REL
    assert sf.JULIA_SUPPLY_POOL.name == "julia_supply_pool_v3.json"
    assert sf.JULIA_SUPPLY_POOL.exists()
    # The parser is built inside `main()`, so there is no factory to call. Assert the flag's
    # default from SOURCE rather than behind a `hasattr` guard — a guarded assertion that
    # silently skips when the factory it names does not exist proves nothing at all, which is
    # what the first version of this test did.
    import inspect
    import re
    src = inspect.getsource(sf.main)
    m = re.search(r'add_argument\(\s*"--julia-seed-pool".*?default=(.+?),\s*$', src,
                  re.S | re.M)
    assert m, "--julia-seed-pool is no longer declared in main()"
    assert m.group(1).strip() == "str(JULIA_SUPPLY_POOL)", m.group(1)
    pool = json.loads(sf.JULIA_SUPPLY_POOL.read_text(encoding="utf-8"))
    acc = []
    for r in pool:
        assert sr.cspacing_ok((r["c_re"], r["c_im"]), acc)
        acc.append((r["c_re"], r["c_im"]))


def test_the_live_pool_passes_the_load_time_check(tmp_path):
    """The production file goes through the real loader, not a re-implementation of it."""
    import types
    stub = types.SimpleNamespace(julia_seed_pool_path=sf.JULIA_SUPPLY_POOL,
                                 _julia_pool_cache=None)
    pool = sf.SteeredFrontier.load_julia_supply_pool(stub)
    assert len(pool) == 209
    assert stub._julia_pool_min_dc >= __import__("supply_routing").CSPACING_FLOOR * (1 - 1e-6)


@pytest.mark.parametrize("rel,expected_min", [
    ("data/atlas/julia_supply_pool_v2.json", 1.0e-2),    # the SUPERSEDED floor
    ("data/atlas/julia_seed_pool.json", 6.0e-3),         # q4_decisive MIN_SEP
])
def test_the_loader_REFUSES_the_pools_that_were_actually_being_used(rel, expected_min):
    """INJECTION, and the injected values are not synthetic — these are the two files that
    were reachable, one of which the last live run really did pass. A guard demonstrated only
    against a hand-made counterexample would not show that it catches THIS mistake."""
    import types
    import supply_routing as sr
    p = sf.ROOT / rel
    assert p.exists(), p
    stub = types.SimpleNamespace(julia_seed_pool_path=p, _julia_pool_cache=None)
    with pytest.raises(SystemExit, match="c-spacing floor"):
        sf.SteeredFrontier.load_julia_supply_pool(stub)
    rows = json.loads(p.read_text(encoding="utf-8"))
    closest = min(
        ((rows[i]["c_re"] - rows[j]["c_re"]) ** 2 + (rows[i]["c_im"] - rows[j]["c_im"]) ** 2)
        for i in range(len(rows)) for j in range(i + 1, len(rows))) ** 0.5
    assert closest == pytest.approx(expected_min, rel=0.05)
    assert closest < sr.CSPACING_FLOOR


def test_a_pool_exactly_AT_the_floor_is_accepted(tmp_path):
    """The tolerance half. A pool thinned at the floor stores rounded decimals, so its
    closest surviving pair can sit a few ulp under the float it was thinned against —
    refusing that would reject the very file the builder emits."""
    import supply_routing as sr
    f = sr.CSPACING_FLOOR
    _stub, load = _pool_loader(tmp_path, [
        dict(c_re=0.0, c_im=0.0, channel="x"),
        dict(c_re=f * (1 - 1e-12), c_im=0.0, channel="x"),
    ])
    assert len(load()) == 2


def test_the_pool_is_read_and_verified_ONCE_not_per_batch(tmp_path):
    """`seed_julia_pool` runs per batch. A per-batch re-read is not the cost — a per-batch
    verification that could disagree with the first one is, and `pool_cursor` indexes into
    whatever list came back."""
    stub, load = _pool_loader(tmp_path, [dict(c_re=0.0, c_im=0.0, channel="x")])
    first = load()
    stub.julia_seed_pool_path.write_text("[]", encoding="utf-8")   # mutate underneath
    assert load() is first


def test_an_inherited_default_does_not_break_a_run_with_no_julia_mandelbrot():
    """Making the pool a DEFAULT must not turn every phoenix-only run into a hard failure.
    Naming a pool with nowhere to put it stays fatal; inheriting one does not."""
    import types
    stub = types.SimpleNamespace(julia_seed_pool_path=sf.JULIA_SUPPLY_POOL,
                                 partitions=["phoenix"], julia_pool_explicit=False)
    assert sf.SteeredFrontier.seed_julia_pool(stub) == 0
    stub.julia_pool_explicit = True
    with pytest.raises(SystemExit, match="needs 'mandelbrot'"):
        sf.SteeredFrontier.seed_julia_pool(stub)


# =========================================================================== #
# The root-draw backstop — the pre-loop draw is outside BOTH caps.
#
# `active_s` counts only the timed batch block and `wall_elapsed_s` starts when the loop is
# entered, so the one-time pre-loop `draw_roots` (measured ~12 min for four families, and the
# production mix is nine partitions) ran with no bound of any kind. The subprocess it spends
# that time in — prescreen's depth-2 probe — had no timeout either.
# =========================================================================== #
def test_the_prescreen_probe_had_no_backstop_and_now_takes_one():
    """RED-BEFORE, by signature. The probe is the subprocess the pre-loop draw lives in; an
    unbounded `subprocess.run` there hangs the night before either cap starts counting."""
    import inspect
    import prescreen
    assert "timeout" in inspect.signature(prescreen.prescreen).parameters
    src = inspect.getsource(prescreen.prescreen)
    assert "timeout=timeout" in src, "the timeout is accepted but not passed to the subprocess"
    # and it must be threaded from the caller the frontier actually uses
    assert "timeout" in inspect.signature(sf.ps.depth2_probe).parameters


def test_the_root_draw_bound_is_clamped_to_the_REMAINING_wall_budget():
    """A backstop longer than the job's budget is not a backstop. With 4 minutes of wall
    budget left, a 25-minute bound would let one draw more than double the run."""
    import types
    stub = types.SimpleNamespace(wall_budget_s=0.0, wall_s_base=0.0, _session_t0=None)
    # The real `wall_elapsed_s`, not a stubbed number: the clamp's whole job is to reflect
    # the clock the wall cap actually reads, and a hand-fed elapsed value would let the two
    # drift apart in exactly the direction that makes the backstop useless.
    stub.wall_elapsed_s = lambda: sf.SteeredFrontier.wall_elapsed_s(stub)
    budget = sf.SteeredFrontier.root_draw_budget_s(stub)
    assert budget == float(sf.ROOT_DRAW_BUDGET_S), "no wall budget => the standing bound"

    stub.wall_budget_s = 10 * 3600.0                      # 10h cap
    stub.wall_s_base = 10 * 3600.0 - 4 * 60.0             # 4 minutes left
    clamped = sf.SteeredFrontier.root_draw_budget_s(stub)
    assert clamped == pytest.approx(4 * 60.0), clamped
    assert clamped < sf.ROOT_DRAW_BUDGET_S

    stub.wall_s_base = 10 * 3600.0 + 900.0                # already over
    assert sf.SteeredFrontier.root_draw_budget_s(stub) == float(sf.MIN_ROOT_DRAW_S), \
        "floored: a legitimately slow draw is not shot merely for being last"

    stub.wall_s_base = 0.0                                # plenty left
    assert sf.SteeredFrontier.root_draw_budget_s(stub) == float(sf.ROOT_DRAW_BUDGET_S)


def test_root_draw_budget_flag_overrides_the_constant_and_still_clamps():
    """`--root-draw-budget` is the only flag-reachable bound on the PRE-LOOP draw, which runs
    outside both caps — so a run whose whole commitment is hours can say so. Three things have
    to hold together: absent => byte-identical to the constant (the default is not a new
    behaviour), present => it wins even with no wall budget (the pre-loop case, where
    `wall_elapsed_s` is 0), and the remaining-wall clamp still outranks it (a smaller override
    must not become a FLOOR)."""
    import types

    def mk(override_min, wall_budget_min=0.0, spent_min=0.0):
        s = types.SimpleNamespace(
            wall_budget_s=wall_budget_min * 60.0, wall_s_base=spent_min * 60.0,
            _session_t0=None,
            root_draw_budget_override=(override_min * 60.0 if override_min else None))
        s.wall_elapsed_s = lambda: sf.SteeredFrontier.wall_elapsed_s(s)
        return s

    # absent => the constant, with and without a wall budget
    assert sf.SteeredFrontier.root_draw_budget_s(mk(None)) == float(sf.ROOT_DRAW_BUDGET_S)
    assert sf.SteeredFrontier.root_draw_budget_s(mk(None, 414)) == float(sf.ROOT_DRAW_BUDGET_S)

    # present => wins, including with no wall budget at all (the pre-loop draw's own case)
    assert sf.SteeredFrontier.root_draw_budget_s(mk(15)) == pytest.approx(15 * 60.0)
    assert sf.SteeredFrontier.root_draw_budget_s(mk(15, 414)) == pytest.approx(15 * 60.0)

    # the override is a CEILING, not a floor: 5 min of wall left beats a 15-min override
    assert sf.SteeredFrontier.root_draw_budget_s(
        mk(15, 414, 409)) == pytest.approx(5 * 60.0)
    # ...and the MIN_ROOT_DRAW_S floor still applies underneath both
    assert sf.SteeredFrontier.root_draw_budget_s(
        mk(15, 414, 500)) == float(sf.MIN_ROOT_DRAW_S)

    # an override LARGER than the constant is honoured too — this is a bound the caller sets,
    # not a shrink-only knob, and a run that wants the overnight bound explicit can say it.
    assert sf.SteeredFrontier.root_draw_budget_s(mk(90)) == pytest.approx(90 * 60.0)


def test_root_draw_budget_flag_default_is_None_so_unpassed_runs_are_unchanged():
    """The default must be `None`, not the constant: a numeric default would be baked into
    every `run_config.json` as if the run had chosen it, and a later change to the constant
    would silently stop reaching runs that never asked to opt out."""
    import argparse
    ap = sf.build_argparser() if hasattr(sf, "build_argparser") else None
    if ap is None:                                    # parser built inline in main()
        import inspect
        src = inspect.getsource(sf)
        assert '"--root-draw-budget", type=float, default=None' in src, \
            "the flag must default to None so an unpassed run keeps the constant"
        return
    ns = ap.parse_args(["--run-dir", "x"])
    assert ns.root_draw_budget is None


def test_a_SLOW_draw_stops_drawing_families_and_SAYS_SO(monkeypatch, capsys):
    """The granularity a per-probe timeout cannot cover: nine families each finishing just
    inside their own timeout is still hours. The deadline is checked between families.

    And the truncation must be REPORTED — a short draw and a fast draw leave the same
    frontier length behind, and only one of them is a problem."""
    import types
    fams = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5"]
    stub = types.SimpleNamespace(
        families=fams, scheduler=None, B=2, family_weights=None,
        rng=None, run_clouds={f: [] for f in fams}, totals={},
        seeders={f: types.SimpleNamespace(draw_batch=lambda c, n: [{"seed_cx": 0.0}])
                 for f in fams},
        scratch=Path("."), seed=0, batch_i=0, frontier=[],
        wall_budget_s=0.0, wall_s_base=0.0, _session_t0=None,
        new_node_id=lambda: 1, _flags=lambda f: [],
    )
    stub.root_draw_budget_s = lambda: sf.SteeredFrontier.root_draw_budget_s(stub)
    stub.wall_elapsed_s = lambda: sf.SteeredFrontier.wall_elapsed_s(stub)
    # Budget of 0 => the deadline is already blown at the first family, so every family is
    # skipped and none of them runs a probe. The probe is made to explode to prove that.
    monkeypatch.setattr(sf, "ROOT_DRAW_BUDGET_S", 0)
    monkeypatch.setattr(sf.ps, "depth2_probe",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe ran")))
    added = sf.SteeredFrontier.draw_roots(stub)
    assert added == 0
    assert stub.totals["root_draw_truncated"] == 1
    out = capsys.readouterr().out
    assert "BOUND HIT" in out and all(f in out for f in fams), out


def test_a_HUNG_probe_costs_that_family_not_the_run(monkeypatch, capsys):
    """A TimeoutError from the probe must skip ONE family, be counted, and be named — not
    kill the run, and not vanish into a family that silently contributed no roots (which is
    indistinguishable from a family that was never tried)."""
    import types
    fams = ["mandelbrot", "multibrot3"]
    calls = []

    def _probe(props, pw, seed, flags, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("probe wedged")
        return ([], [], {})

    stub = types.SimpleNamespace(
        families=fams, scheduler=None, B=2, family_weights=None, rng=None,
        run_clouds={f: [] for f in fams}, totals={},
        seeders={f: types.SimpleNamespace(draw_batch=lambda c, n: [{"seed_cx": 0.0}])
                 for f in fams},
        scratch=Path("."), seed=0, batch_i=0, frontier=[],
        wall_budget_s=0.0, wall_s_base=0.0, _session_t0=None,
        new_node_id=lambda: 1, _flags=lambda f: [],
    )
    stub.root_draw_budget_s = lambda: sf.SteeredFrontier.root_draw_budget_s(stub)
    stub.wall_elapsed_s = lambda: sf.SteeredFrontier.wall_elapsed_s(stub)
    monkeypatch.setattr(sf.ps, "depth2_probe", _probe)
    added = sf.SteeredFrontier.draw_roots(stub)
    assert added == 0
    assert stub.totals["root_draw_timeouts"] == 1
    assert len(calls) == 2, "the second family must still be attempted"
    assert all(t is not None and t > 0 for t in calls), calls
    out = capsys.readouterr().out
    assert "TIMED OUT" in out and "mandelbrot" in out


def test_the_run_config_records_the_pools_MEASURED_spacing_not_null():
    """The stamp exists because the floor was invisible in every prior run config — the path
    was recorded and the path does not say which floor thinned it. A `null` measurement is
    that same invisibility with an extra key, and that is exactly what shipped first: the
    value was read lazily but `write_run_config` runs BEFORE `seed_julia_pool`, so it was
    always null. Pinned as an ORDER guarantee, since that is what broke."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.write_run_config)
    assert "load_julia_supply_pool()" in src, (
        "write_run_config must force the pool load, or pool_min_dc is null in every run")
    assert src.index("load_julia_supply_pool()") < src.index("cfg = dict("), \
        "the load must happen BEFORE the config dict is built"
    # and the value really is populated once loaded
    import types
    stub = types.SimpleNamespace(julia_seed_pool_path=sf.JULIA_SUPPLY_POOL,
                                 _julia_pool_cache=None)
    sf.SteeredFrontier.load_julia_supply_pool(stub)
    assert stub._julia_pool_min_dc is not None and stub._julia_pool_min_dc > 0


# =========================================================================== #
# the dive leg: a plan that stays readable at whatever length the budget allows
#
# Every one of these is an INJECTION test: build a plan, truncate it where a budget would,
# and assert on what survives. The 2026-08-05 dive stopped at 7 of 28 and its first four
# entries were controls; the scheduler-OFF shape is worse still (20 top, then 8 control), so
# the population these run against is the real failure, not a hypothetical one.
# =========================================================================== #
def _plan(n_top, n_ctrl, parts=None):
    """A synthetic plan in the shape `_build_dive_plan` produces: top block, then control."""
    parts = parts or (lambda i: "mandelbrot")
    return ([dict(dive_id=f"dive_{i:03d}", start_group="top", partition=parts(i))
             for i in range(n_top)]
            + [dict(dive_id=f"dive_{n_top+i:03d}", start_group="control",
                    partition=parts(n_top + i)) for i in range(n_ctrl)])


@pytest.mark.parametrize("cut", [2, 3, 4, 5, 7, 10, 14, 20, 27, 28])
def test_a_truncated_dive_plan_keeps_BOTH_ARMS_at_every_cut_point(cut):
    """THE property. A budget cut anywhere (past the first entry, where two arms cannot both
    fit) leaves a usable top-vs-control contrast."""
    out = sf.interleave_dive_arms(_plan(20, 8))
    got = collections.Counter(e["start_group"] for e in out[:cut])
    assert got["top"] > 0 and got["control"] > 0, f"cut {cut} lost an arm: {got}"


def test_the_UNFIXED_block_order_loses_an_arm_at_the_length_the_run_actually_reached():
    """The control this test needs to be worth anything: the same truncation on the block
    plan. 20 top then 8 control, cut at the 7 the real dive reached, is zero controls."""
    got = collections.Counter(e["start_group"] for e in _plan(20, 8)[:7])
    assert got["control"] == 0 and got["top"] == 7


@pytest.mark.parametrize("n_top,n_ctrl", [(20, 8), (8, 20), (3, 3), (28, 1), (1, 28), (2, 5)])
def test_every_arm_stays_within_1_of_its_proportional_share_in_EVERY_prefix(n_top, n_ctrl):
    """The apportionment bound itself, asserted on the built order rather than trusted — the
    same +/-1-in-every-prefix statement the label sheet's stratified deal makes."""
    out = sf.interleave_dive_arms(_plan(n_top, n_ctrl))
    n = {"top": n_top, "control": n_ctrl}
    N = n_top + n_ctrl
    seen = collections.Counter()
    for L, e in enumerate(out, start=1):
        seen[e["start_group"]] += 1
        for arm, cnt in n.items():
            assert abs(seen[arm] - L * cnt / N) <= 1.0, (arm, L, dict(seen))


def test_interleaving_permutes_and_never_drops_or_duplicates_an_entry():
    plan = _plan(20, 8)
    out = sf.interleave_dive_arms(plan)
    assert len(out) == len(plan)
    assert sorted(e["dive_id"] for e in out) == sorted(e["dive_id"] for e in plan)


def test_the_order_WITHIN_an_arm_is_preserved_so_the_deficit_sort_still_holds():
    """Item 8's deficit ordering is applied before this and must survive it: interleaving
    chooses WHICH arm supplies the next entry, never which entry inside that arm."""
    plan = _plan(20, 8, parts=lambda i: f"p{i}")
    out = sf.interleave_dive_arms(plan)
    for arm in ("top", "control"):
        before = [e["dive_id"] for e in plan if e["start_group"] == arm]
        after = [e["dive_id"] for e in out if e["start_group"] == arm]
        assert before == after


def test_a_single_arm_plan_is_returned_UNCHANGED():
    """--n-control 0 is a legitimate plan with nothing to interleave; it must not be
    reordered, and it must not raise."""
    plan = _plan(5, 0)
    assert sf.interleave_dive_arms(plan) == plan
    assert sf.interleave_dive_arms([]) == []


def test_the_plan_builder_ACTUALLY_calls_the_interleave_unconditionally():
    """Not behind `if self.scheduler`: the scheduler-OFF plan is the WORSE of the two shapes
    (a pure top-then-control block), so a fix that only ran under the scheduler would leave
    the failing case failing."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier._build_dive_plan)
    assert "return interleave_dive_arms(plan)" in src
    body = src[src.index("if self.scheduler is not None"):]
    assert body.count("interleave_dive_arms") == 1
    assert body.index("interleave_dive_arms") > body.index("plan.sort("), \
        "the arm interleave must run AFTER the deficit sort, not instead of it"


def test_dive_mode_writes_its_run_config_BEFORE_the_first_dive():
    """Pre-registration parity with the crawl path. `run_dive` wrote no run_config.json at
    all, so a dive's tau_h, checkpoint, interior gate and plan shape lived only in the
    append-as-you-go `dive_state.json` and a hand-written launch.txt."""
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.run_dive)
    assert "self.write_run_config()" in src
    assert src.index("self.write_run_config()") < src.index("while done_idx < len(plan)"), \
        "the config must be written before the first dive runs, not after"
    # ...and only on the FRESH branch: a resume must not overwrite what was pre-registered.
    fresh = src[src.index("else:"):src.index("while done_idx < len(plan)")]
    assert "self.write_run_config()" in fresh


def test_the_run_config_records_the_dive_block_and_the_wall_budget_limitation():
    import inspect
    src = inspect.getsource(sf.SteeredFrontier.write_run_config)
    assert 'cfg["dive"]' in src and "if self.dive:" in src
    assert "wall_budget" in src[src.index('cfg["dive"]'):], \
        "the dive config must record that --wall-budget does not reach this path"


@pytest.mark.parametrize("wb", [0.1, 15.0, 69.0])
def test_wall_budget_with_dive_is_REFUSED_not_silently_ignored(wb):
    """A flag that does nothing is worse than an absent feature: the reason anyone passes
    --wall-budget is to bound a run they are not watching."""
    with pytest.raises(SystemExit) as e:
        sf.check_wall_budget_supported(True, wb)
    assert "NO EFFECT" in str(e.value) and "--budget" in str(e.value)


@pytest.mark.parametrize("dive,wb", [(False, 69.0), (False, 0.0), (True, 0.0), (True, None)])
def test_the_wall_budget_guard_lets_a_CRAWL_run_and_a_ZERO_flag_through(dive, wb):
    """The other half. A gate that refuses everything passes the test above and is useless:
    the crawl path is where the flag works, and the 2026-08-05 dive's own invocation (dive,
    no flag) must still be legal."""
    sf.check_wall_budget_supported(dive, wb)


def test_the_guard_is_ON_THE_CONSTRUCTION_PATH_not_only_in_main():
    """Every entry point builds a `SteeredFrontier`; a check that lived only in `main` would
    be bypassed by anything that constructs one directly."""
    import inspect
    assert "check_wall_budget_supported(self.dive" in \
        inspect.getsource(sf.SteeredFrontier.__init__)


# =========================================================================== #
# The two-entry-point contract. `--wall-budget` was a flag that parsed, stored and then
# no-oped on the dive path; the bug is not that one flag, it is that NOTHING distinguished
# "deliberately N/A in dive mode" from "silently dropped in dive mode" across the 41
# constructor attributes in that position. `sf.DIVE_IGNORES` declares the set; these tests
# recompute it from the source and refuse the difference.
# =========================================================================== #
def _crawl_only_attributes(source: str, cls_name: str, entry: str, other: str) -> set:
    """`{attr}` assigned in `__init__` and read from `entry`'s call closure but never from
    `other`'s. Pure AST over `self.<name>` — no import, no execution.

    `entry` is the OUTER path (`run` calls `run_dive`, so its closure is a superset); the
    difference is therefore exactly "reachable when crawling, unreachable when diving"."""
    import ast
    cls = next(n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    def refs(node):
        calls, reads, stores = set(), set(), set()
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"):
                if sub.attr in methods:
                    calls.add(sub.attr)
                elif isinstance(sub.ctx, ast.Load):
                    reads.add(sub.attr)
                elif isinstance(sub.ctx, ast.Store):
                    stores.add(sub.attr)
        return calls, reads, stores

    def closure(name):
        seen, stack = set(), [name]
        while stack:
            m = stack.pop()
            if m in seen or m not in methods:
                continue
            seen.add(m)
            stack += list(refs(methods[m])[0])
        return seen

    def reads_of(ms):
        return set().union(*(refs(methods[m])[1] for m in ms)) if ms else set()

    init_attrs = refs(methods["__init__"])[2]
    return (reads_of(closure(entry)) - reads_of(closure(other))) & init_attrs


def _sf_crawl_only():
    import inspect
    return _crawl_only_attributes(inspect.getsource(sf), "SteeredFrontier", "run", "run_dive")


def test_every_crawl_only_constructor_attribute_is_declared():
    """THE gate. A constructor attribute the dive path cannot reach must be either made to
    apply, refused at the flag (`check_wall_budget_supported`), or declared here with the
    reason. The set is recomputed from the AST, so a new flag that quietly misses the dive
    path fails HERE, where it is cheap, instead of in a launch record."""
    crawl_only = _sf_crawl_only()
    declared = set(sf.DIVE_IGNORES)
    undeclared = sorted(crawl_only - declared)
    stale = sorted(declared - crawl_only)
    assert not undeclared, (
        f"{len(undeclared)} constructor attribute(s) are read on the crawl path and NEVER on "
        f"the dive path, and nothing says whether that is deliberate: {undeclared}. Either "
        f"make run_dive read them, refuse the flag at parse time like "
        f"check_wall_budget_supported, or add them to DIVE_IGNORES with a one-line reason.")
    assert not stale, (
        f"DIVE_IGNORES declares {stale} inapplicable, but the dive path now reads them (or "
        f"the constructor no longer sets them). Drop the entry — a stale exemption hides the "
        f"next real one.")


def test_the_declared_set_is_the_measured_41_and_every_entry_carries_a_reason():
    """Non-vacuity for the gate above: an empty or trivially-satisfied analysis would pass
    it. 41 is what the 2026-08-05 reachability measured, and a reason string that says
    nothing is the same as no declaration."""
    assert len(sf.DIVE_IGNORES) == 41, len(sf.DIVE_IGNORES)
    assert _sf_crawl_only(), "the reachability analysis found nothing — it is not running"
    for attr, why in sf.DIVE_IGNORES.items():
        assert isinstance(why, str) and len(why) > 20, f"{attr}: reason is not a reason"


def test_attributes_the_dive_path_DOES_read_are_not_in_the_exemption_set():
    """The other half. A gate that exempted everything would also pass — these are read on
    both paths and must stay outside the table."""
    crawl_only = _sf_crawl_only()
    for attr in ("tau_h", "dive_target_depth", "ledger", "morph", "totals", "rng", "clouds",
                 "budget_s", "stop_path", "scheduler"):
        assert attr not in crawl_only, f"{attr} is read on the dive path — analysis is wrong"
        assert attr not in sf.DIVE_IGNORES, f"{attr} is exempted but the dive path reads it"


def test_the_analysis_CATCHES_a_flag_that_misses_the_second_entry_point():
    """The injection proof — the `--wall-budget` shape, in miniature. `bound_s` is stored by
    the constructor and consumed only by the crawl loop; the dive loop has its own loop and
    never reads it. If the analysis cannot see this, it cannot see the real one."""
    src = '''
class Runner:
    def __init__(self, args):
        self.bound_s = args.bound * 60.0      # the flag that misses the dive path
        self.shared = args.shared             # read by both
    def _crawl_step(self):
        return self.bound_s > 0 and self.shared
    def run_dive(self):
        return self.shared
    def run(self):
        if self.dive:
            return self.run_dive()
        return self._crawl_step()
'''
    assert _crawl_only_attributes(src, "Runner", "run", "run_dive") == {"bound_s"}
    # ...and once the dive loop reads it, it drops out of the set rather than needing an
    # exemption — the preferred fix has to be visible to the same analysis.
    fixed = src.replace("    def run_dive(self):\n        return self.shared",
                        "    def run_dive(self):\n        return self.shared and self.bound_s")
    assert _crawl_only_attributes(fixed, "Runner", "run", "run_dive") == set()


# =========================================================================== #
# Scratch teardown (2026-08-08). A 6 h steady-state run leaves ~118 GB / 138k files under
# `<run_dir>/scratch`; two same-day runs were 86% of a 178 GB bulk store that nothing
# guards, and cleanup was manual, therefore 8 h late. Teardown now hangs off the CLEAN close
# and nothing else — the load-bearing half of that sentence is "and nothing else", so it
# gets a behavioural abort test AND a reachability gate, each proved red.
# =========================================================================== #
def _scratch_run(tmp_path, *, retain=False, n_files=7, size=1024):
    """A run dir with a populated scratch subtree and the REAL close methods bound. Returns
    (stub, run_dir, scratch, expected_bytes). The files beside the scratch are the control:
    teardown must take the subtree and nothing else."""
    import types
    run_dir = tmp_path / "run"
    scratch = run_dir / "scratch"
    (scratch / "reframe_n0" / "tiles").mkdir(parents=True)
    for i in range(n_files):
        (scratch / "reframe_n0" / "tiles" / f"t{i}.jpg.field.bin").write_bytes(b"\x00" * size)
    (run_dir / "keep.txt").write_text("the ledger and summary live here", encoding="utf-8")
    (run_dir / "outcome_ledger.jsonl").write_text('{"guard_pass": 1}\n', encoding="utf-8")

    stub = types.SimpleNamespace(run_dir=run_dir, scratch=scratch, retain_scratch=retain)
    stub.finalize_streams = lambda: None
    for m in ("teardown_scratch", "_close_summary"):
        setattr(stub, m, types.MethodType(getattr(sf.SteeredFrontier, m), stub))
    return stub, run_dir, scratch, n_files * size


def _teardown_rec(run_dir):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[
        sf.SCRATCH_TEARDOWN_KEY]


def test_teardown_fires_on_a_clean_close_and_takes_ONLY_the_scratch(tmp_path):
    stub, run_dir, scratch, nbytes = _scratch_run(tmp_path)
    stub._close_summary(dict(run_ts="mini", mode="steered"))

    assert not scratch.exists(), "the scratch subtree survived a clean close"
    # the control: everything else in the run dir is untouched.
    assert (run_dir / "keep.txt").read_text(encoding="utf-8").startswith("the ledger")
    assert (run_dir / "outcome_ledger.jsonl").exists()
    rec = _teardown_rec(run_dir)
    assert rec["outcome"] == "scratch_deleted"
    # COUNTED, not asserted-to-exist: a smoke that asserts "it ran" passes on zero.
    assert (rec["files"], rec["bytes"]) == (7, nbytes)
    assert rec["path"] == str(scratch) and rec["retain_flag"] is False


def test_retain_scratch_keeps_the_tree_and_SAYS_it_kept_it(tmp_path):
    stub, run_dir, scratch, _ = _scratch_run(tmp_path, retain=True)
    stub._close_summary(dict(run_ts="mini", mode="steered"))

    assert scratch.exists() and len(list(scratch.rglob("*.bin"))) == 7
    rec = _teardown_rec(run_dir)
    assert rec["outcome"] == "scratch_retained" and rec["reason"] == "--retain-scratch"
    assert rec["retain_flag"] is True


# --------------------------------------------------------------------------- #
# The abort case, proved red by injecting the defect into the CONTROL FLOW while the
# teardown code under it stays real. The defect is the obvious wrong implementation — close
# the run from a `finally:` so it "always cleans up" — and it is wrong precisely because an
# interrupted run's scratch is the state you may still need to read.
# --------------------------------------------------------------------------- #
def _drive(stub, *, abort, shape):
    """A miniature `run()`: a loop, then the close. `shape="plain"` is the shipped control
    flow; `shape="finally"` is the injected defect."""
    def loop():
        for i in range(3):
            if abort and i == 1:
                raise KeyboardInterrupt("killed mid-run")

    summary = dict(run_ts="mini", mode="steered")
    if shape == "finally":
        try:
            loop()
        finally:
            stub._close_summary(summary)
    else:
        loop()
        stub._close_summary(summary)


@pytest.mark.parametrize("shape", ["plain", "finally"])
def test_a_CLEAN_close_tears_down_under_EITHER_shape(tmp_path, shape):
    """Non-vacuity for the pair below: the injected defect is not "teardown broken", so the
    abort asymmetry is the only thing that distinguishes the two shapes."""
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)
    _drive(stub, abort=False, shape=shape)
    assert not scratch.exists() and (run_dir / "summary.json").exists()


def test_the_INJECTED_finally_shape_tears_down_an_ABORTED_run(tmp_path):
    """RED. This is what the gate below exists to keep out of the module: an interrupted
    run's fields — the only copy of what it was doing — deleted on the way out."""
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _drive(stub, abort=True, shape="finally")
    assert not scratch.exists(), "the injection did not reproduce the defect"


def test_an_ABORTED_run_keeps_its_scratch_AND_writes_no_summary(tmp_path):
    """GREEN — same abort, same teardown code, shipped shape."""
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _drive(stub, abort=True, shape="plain")
    assert scratch.exists() and len(list(scratch.rglob("*.bin"))) == 7
    assert not (run_dir / "summary.json").exists(), "an aborted run closed its record"


# --------------------------------------------------------------------------- #
# ...and the same asymmetry bound to the REAL module, so the miniature cannot drift away
# from what steered_frontier.py actually does.
# --------------------------------------------------------------------------- #
def _self_call_contexts(source: str, name: str) -> list:
    """`[(enclosing function, context)]` for every `self.<name>(...)`, where context is
    `finally` / `except` / `body`. Pure AST — no import, no execution."""
    import ast

    def calls_in(node):
        return [c for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name) and c.func.value.id == "self"
                and c.func.attr == name]

    out = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        guarded = {}
        for t in ast.walk(fn):
            if isinstance(t, ast.Try):
                for blk, ctx in ((t.finalbody, "finally"), (t.handlers, "except")):
                    for s in blk:
                        for c in calls_in(s):
                            guarded[id(c)] = ctx
        out += [(fn.name, guarded.get(id(c), "body")) for c in calls_in(fn)]
    return sorted(out)


# The close chain, innermost out. Checking only `teardown_scratch`'s own call site is not
# enough: the same defect introduced one level up (`try: loop() finally: self.finish()`)
# reaches teardown just as surely and leaves every inner call site in a plain body.
CLOSE_CHAIN = ("teardown_scratch", "_close_summary", "finish", "finish_dive")


def _guarded_close_calls(source: str) -> list:
    """`[(enclosing function, ctx, attr)]` for every call to a close-chain method sitting in
    an `except`/`finally` block — on ANY receiver, not just `self`."""
    import ast
    out = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for t in ast.walk(fn):
            if not isinstance(t, ast.Try):
                continue
            for blk, ctx in ((t.finalbody, "finally"), (t.handlers, "except")):
                for s in blk:
                    for c in ast.walk(s):
                        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                                and c.func.attr in CLOSE_CHAIN):
                            out.append((fn.name, ctx, c.func.attr))
    return sorted(set(out))


def test_teardown_is_reachable_ONLY_from_the_clean_close_path():
    """THE gate. Teardown hangs off the summary write and nothing else: one call site, in a
    plain body, under the two entry points' close methods."""
    import inspect
    src = inspect.getsource(sf)
    assert _self_call_contexts(src, "teardown_scratch") == [("_close_summary", "body")], \
        _self_call_contexts(src, "teardown_scratch")
    assert _self_call_contexts(src, "_close_summary") == \
        [("finish", "body"), ("finish_dive", "body")], _self_call_contexts(src, "_close_summary")
    # ...and no link of the chain is reached from an except/finally ANYWHERE in the module,
    # which is the same defect introduced one level up where the checks above cannot see it.
    assert _guarded_close_calls(src) == [], _guarded_close_calls(src)
    # ...and nothing closes the run behind the driver's back on the way out. Over the AST,
    # not the text: the first version of this line was a substring scan and it went red on
    # the module comment that says there is no atexit hook.
    import ast
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module}
    dotted = {f"{c.func.value.id}.{c.func.attr}" for c in ast.walk(tree)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
              and isinstance(c.func.value, ast.Name)}
    assert "shutil" in imported and "shutil.rmtree" in dotted, \
        "the import/call analysis is not seeing the module it is scanning"
    assert "atexit" not in imported, "an atexit hook would close the run on ANY exit path"
    assert "signal.signal" not in dotted, "a signal handler would close an INTERRUPTED run"


def test_the_reachability_GATE_catches_the_finally_shape():
    """The injection proof for the gate itself: an analysis that cannot see the defect in
    miniature cannot see it in 4,900 lines."""
    bad = ("class Runner:\n"
           "    def _close_summary(self, s): self.teardown_scratch()\n"
           "    def run(self):\n"
           "        try:\n"
           "            self._loop()\n"
           "        finally:\n"
           "            self._close_summary({})\n")
    assert _self_call_contexts(bad, "_close_summary") == [("run", "finally")]
    assert _self_call_contexts(bad, "teardown_scratch") == [("_close_summary", "body")]
    # ...and the fix drops back to `body` under the same analysis, so the preferred shape is
    # visible to it rather than merely un-flagged.
    ok = ("class Runner:\n"
          "    def _close_summary(self, s): self.teardown_scratch()\n"
          "    def run(self):\n"
          "        self._loop()\n"
          "        self._close_summary({})\n")
    assert _self_call_contexts(ok, "_close_summary") == [("run", "body")]
    assert _guarded_close_calls(bad) == [("run", "finally", "_close_summary")]
    assert _guarded_close_calls(ok) == []
    # The one-level-up variant, which the call-site equalities above CANNOT see: every inner
    # call site is still a plain body, and teardown still runs on an abort.
    up = ("class Runner:\n"
          "    def finish(self): self._close_summary({})\n"
          "    def _close_summary(self, s): self.teardown_scratch()\n"
          "    def run(self):\n"
          "        try:\n"
          "            self._loop()\n"
          "        finally:\n"
          "            self.finish()\n")
    assert _self_call_contexts(up, "teardown_scratch") == [("_close_summary", "body")]
    assert _self_call_contexts(up, "_close_summary") == [("finish", "body")]
    assert _guarded_close_calls(up) == [("run", "finally", "finish")]


# --------------------------------------------------------------------------- #
# The remaining outcomes, and the flag's path from parse to attribute.
# --------------------------------------------------------------------------- #
def test_an_interrupted_TEARDOWN_leaves_a_summary_that_says_not_reached(tmp_path):
    """The third outcome is a STATE, not a missing key. The summary lands before the delete
    starts, so a kill mid-rmtree stays distinguishable from a run that predates teardown —
    and that is exactly the case where the scratch is half gone."""
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)

    def _killed():
        raise KeyboardInterrupt("killed mid-rmtree")

    stub.teardown_scratch = _killed
    with pytest.raises(KeyboardInterrupt):
        stub._close_summary(dict(run_ts="mini"))
    rec = _teardown_rec(run_dir)
    assert rec["outcome"] == "not_reached" and "partially deleted" in rec["note"]


def test_a_target_that_is_not_a_scratch_subtree_is_REFUSED_not_deleted(tmp_path):
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)
    stub.scratch = run_dir                     # as if _bulk_scratch moved under us
    rec = stub.teardown_scratch()
    assert rec["outcome"] == "scratch_retained" and rec["reason"].startswith("REFUSED")
    assert run_dir.exists() and (run_dir / "keep.txt").exists()


def test_a_failed_delete_is_RECORDED_not_raised(tmp_path, monkeypatch):
    """The summary is already on disk when teardown runs, so a Windows file lock must degrade
    to a recorded outcome — not turn a closed 6 h run into a traceback."""
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)

    def _locked(p):
        raise PermissionError("locked by serve.py")

    monkeypatch.setattr(sf.shutil, "rmtree", _locked)
    stub._close_summary(dict(run_ts="mini"))
    rec = _teardown_rec(run_dir)
    assert rec["outcome"] == "scratch_delete_failed" and "PermissionError" in rec["error"]
    assert rec["still_present"] is True and scratch.exists()


def test_a_run_that_made_no_scratch_reports_absent_not_deleted(tmp_path):
    """Fourth outcome, beyond the three the prompt named: "deleted 0 files" and "there was
    nothing there" are different facts about a run and a record must not merge them."""
    import shutil as _sh
    stub, run_dir, scratch, _ = _scratch_run(tmp_path)
    _sh.rmtree(scratch)
    rec = stub.teardown_scratch()
    assert rec["outcome"] == "scratch_absent" and rec["files"] == 0


def test_the_flag_reaches_the_constructor_attribute(monkeypatch, tmp_path):
    """Parse -> args -> `self.retain_scratch`. A flag that parses and then no-ops is the
    `--wall-budget` failure this file already carries three tests for."""
    import inspect
    seen = []

    class _Sentinel:
        def __init__(self, args):
            seen.append(args)

        def run(self):
            pass

    monkeypatch.setattr(sf, "SteeredFrontier", _Sentinel)
    monkeypatch.setattr(sys, "argv", ["steered_frontier.py", "--run-dir",
                                      str(tmp_path / "run"), "--retain-scratch"])
    sf.main()
    assert seen and seen[0].retain_scratch is True
    # ...and the default is OFF, i.e. teardown is what an unflagged run gets. That direction
    # is the one that matters: a flag defaulting to retain would ship the old behaviour.
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["steered_frontier.py", "--run-dir", str(tmp_path / "run")])
    sf.main()
    assert seen[0].retain_scratch is False
    # and the constructor really reads it (__init__ loads the scorer, too heavy to run here).
    init = inspect.getsource(sf).split("def __init__(self, args):", 1)[1]
    assert 'self.retain_scratch = bool(getattr(args, "retain_scratch"' in init[:8000]
