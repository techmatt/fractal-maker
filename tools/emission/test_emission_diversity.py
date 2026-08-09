"""Tests for diversity-aware emission v1 — pure logic + the two acceptance proofs
(current-decode rejection of an old-ledger v6 row; append-only pool resume).

All tests are torch-free / render-free: the descriptor module's clustering + Location +
admitted-loader, the deficit machinery, the selector, and the pool are exercised directly.
Run: uv run pytest tools/emission/test_emission_diversity.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.emission import cells as C          # noqa: E402
from tools.emission import selection as SEL     # noqa: E402
from tools.emission import descriptor as D     # noqa: E402
from tools.emission import floors as FL        # noqa: E402
from tools.emission import ranked_intake as RI  # noqa: E402
from tools.emission.pool import Pool           # noqa: E402
import corpus_common as cc                     # noqa: E402


# --------------------------------------------------------------------------- #
# cells.py — target measure, feasible cells, deficit, attempt cap, colorizer choice.
# --------------------------------------------------------------------------- #
def _measure(cells, **kw):
    """A `TargetMeasure` over `cells` from EQUAL per-partition release shares — the
    test-local stand-in for the deleted `from_config({"mode": "uniform"})`."""
    parts = sorted({c[0] for c in cells})
    return C.TargetMeasure.from_partition_shares({p: 1.0 for p in parts}, cells, **kw)


def _share_of(tm, feasible, match_pred):
    """Realized fraction of the total measure held by cells satisfying `match_pred`."""
    w = {c: tm.weight(c) for c in feasible}
    tot = sum(w.values())
    return sum(v for c, v in w.items() if match_pred(c)) / tot


def test_measure_gives_each_partition_exactly_its_release_share():
    """The contract: a partition's cells hold its intended share of the measure, and the
    per-cell weight is that share divided by ITS OWN feasible-cell count."""
    obs = [("mandelbrot", f"mandelbrot#{i}") for i in range(5)] + [("phoenix", "phoenix#0")]
    feasible = C.build_feasible_cells(obs, ["k16:1", "k16:2"], ["smooth", "tia"])
    tm = C.TargetMeasure.from_partition_shares({"mandelbrot": 3.0, "phoenix": 1.0}, feasible)
    assert tm.partition_shares() == pytest.approx({"mandelbrot": 0.75, "phoenix": 0.25})
    assert _share_of(tm, feasible, lambda c: c[0] == "mandelbrot") == pytest.approx(0.75)
    # 20 mandelbrot cells vs 4 phoenix cells: the per-cell weights differ by exactly that ratio
    assert tm.weight(("mandelbrot", "mandelbrot#0", "k16:1", "smooth")) == pytest.approx(0.75 / 20)
    assert tm.weight(("phoenix", "phoenix#0", "k16:1", "smooth")) == pytest.approx(0.25 / 4)


def test_measure_is_denominator_invariant_in_morph_clusters():
    """The property the deleted `target_share` solver existed to give ONE partition, now
    structural for every partition: growing a partition's cluster count does not grow its
    share of the release, it spreads the same share over more cells. This is the campaign-2
    inversion (102 mandelbrot clusters swamping julia:mandelbrot's 4) made impossible."""
    def share(n_mandel):
        obs = ([("mandelbrot", f"mandelbrot#{i}") for i in range(n_mandel)]
               + [("phoenix:classic", "phoenix:classic#0")])
        feasible = C.build_feasible_cells(obs, ["k16:1"], ["smooth"])
        tm = C.TargetMeasure.from_partition_shares(
            {"mandelbrot": 3.0, "phoenix:classic": 0.2}, feasible)
        return _share_of(tm, feasible, lambda c: c[0] == "phoenix:classic")
    assert share(3) == pytest.approx(0.2 / 3.2)
    assert share(300) == pytest.approx(0.2 / 3.2)


def test_classic_phoenix_is_addressable_as_its_own_partition():
    """Injection proof for the cell-axis re-key: `phoenix:classic` cells carry the classic
    ratio and the `phoenix` cells alongside them carry phoenix's — a measure that could not
    tell them apart gave both the same weight, which is the state this replaces."""
    obs = [("phoenix", "phoenix#0"), ("phoenix:classic", "phoenix:classic#0")]
    feasible = C.build_feasible_cells(obs, ["k16:1"], ["smooth"])
    tm = C.TargetMeasure.from_partition_shares({"phoenix": 1.0, "phoenix:classic": 0.2}, feasible)
    w_varied = tm.weight(("phoenix", "phoenix#0", "k16:1", "smooth"))
    w_classic = tm.weight(("phoenix:classic", "phoenix:classic#0", "k16:1", "smooth"))
    assert w_varied / w_classic == pytest.approx(5.0)


def test_a_cell_whose_partition_has_no_share_is_refused():
    """No zero default: an unregistered partition would be permanently starved AND read as
    "no demand" rather than as a missing policy decision."""
    feasible = C.build_feasible_cells([("mandelbrot", "m#0"), ("nope", "nope#0")],
                                      ["k16:1"], ["smooth"])
    with pytest.raises(C.UnknownPartitionCell):
        C.TargetMeasure.from_partition_shares({"mandelbrot": 1.0}, feasible)
    tm = C.TargetMeasure.from_partition_shares({"mandelbrot": 1.0},
                                               [("mandelbrot", "m#0", "k16:1", "smooth")])
    with pytest.raises(C.UnknownPartitionCell):
        tm.weight(("nope", "nope#0", "k16:1", "smooth"))


def test_a_share_with_no_feasible_cell_is_reported_not_absorbed():
    """A partition with demand and no supply this intake is a supply fact; a renormalized
    measure alone cannot say it."""
    feasible = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1"], ["smooth"])
    shares = {"mandelbrot": 3.0, "phoenix:classic": 0.2}
    tm = C.TargetMeasure.from_partition_shares(shares, feasible)
    assert tm.partition_shares() == pytest.approx({"mandelbrot": 1.0})
    assert tm.unrealized_shares(shares) == pytest.approx({"phoenix:classic": 0.2})


def test_feasible_cells_and_deficit_sign():
    observed = [("mandelbrot", "m#0"), ("multibrot3", "x#0")]
    flavors = ["k16:1", "k16:2"]
    styles = ["smooth", "tia"]
    cells = C.build_feasible_cells(observed, flavors, styles)
    assert len(cells) == 2 * 2 * 2
    m = C.DeficitModel(cells, _measure(cells))
    # empty pool: every cell deficit == its target fraction (all equal, uniform)
    d0 = m.deficit(cells[0])
    assert d0 == pytest.approx(1.0 / len(cells))
    # fill one cell → its deficit drops below an unfilled cell's
    m.record_fill(cells[0])
    assert m.deficit(cells[0]) < m.deficit(cells[1])


def test_attempt_cap_evicts_cell():
    cells = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1"], ["smooth", "tia"])
    m = C.DeficitModel(cells, _measure(cells, attempt_cap=3))
    target = ("mandelbrot", "m#0", "k16:1", "smooth")
    assert m.record_attempt(target) is False   # 1
    assert m.record_attempt(target) is False   # 2
    assert m.record_attempt(target) is True    # 3 → capped (zero fills)
    assert target in m.capped and target not in m.support
    # a filled cell is never capped no matter how many attempts
    other = ("mandelbrot", "m#0", "k16:1", "tia")
    m.record_fill(other)
    for _ in range(10):
        assert m.record_attempt(other) is False


def test_range_normalized_softmax_prefers_max():
    p = C.range_normalized_softmax([0.1, 0.0, 0.0], temp=0.2)
    assert p[0] > p[1] and p[0] > p[2]
    assert p[1] == pytest.approx(p[2])
    assert sum(p) == pytest.approx(1.0)
    # all equal → uniform
    q = C.range_normalized_softmax([0.5, 0.5, 0.5], temp=0.2)
    assert all(x == pytest.approx(1 / 3) for x in q)


def test_choose_option_avoids_filled():
    cells = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1", "k16:2"], ["smooth"])
    m = C.DeficitModel(cells, _measure(cells, softmax_temp=0.05))
    # fill (k16:1, smooth) heavily so the deficit strongly favors (k16:2, smooth)
    for _ in range(5):
        m.record_fill(("mandelbrot", "m#0", "k16:1", "smooth"))
    rng = np.random.default_rng(0)
    picks = [C.choose_option(m, "mandelbrot", "m#0", ["k16:1", "k16:2"], ["smooth"], rng)[0]
             for _ in range(200)]
    from collections import Counter
    ct = Counter(picks)
    assert ct["k16:2"] > ct["k16:1"]      # deficit steers away from the filled flavor


# --------------------------------------------------------------------------- #
# select.py — kernel, niche percentile, greedy coverage.
# --------------------------------------------------------------------------- #
def _entry(id, type, cluster, flavor, style, score, emb):
    return {"id": id, "type": type, "cluster": cluster, "flavor": flavor,
            "style": style, "score": score, "emb": emb}


def test_kernel_continuous_cos_across_cells():
    # continuous morph cos, NO categorical gate: a near-identical look is discounted even
    # across cells (this is the coverage-engages fix — the old kernel returned 0 for c).
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.9, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#0", "k16:1", "smooth", 0.8, [1.0, 0.0])   # same cell, cos 1
    c = _entry("c", "mandelbrot", "m#0", "k16:2", "smooth", 0.8, [1.0, 0.0])   # diff flavor, cos 1
    d = _entry("d", "mandelbrot", "m#0", "k16:2", "smooth", 0.8, [0.0, 1.0])   # diff flavor, cos 0
    assert SEL.kernel(a, b) == pytest.approx(1.0)
    assert SEL.kernel(a, c) == pytest.approx(1.0)   # was 0.0 under the categorical gate
    assert SEL.kernel(a, d) == pytest.approx(0.0)


def test_kernel_style_weight_floors_same_mode():
    # morph-distinct (orthogonal) tiles of the SAME render style are floored at style_weight;
    # a different style stays at the (here 0) cosine — how the strange pass spreads modes.
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "tia", 0.6, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#1", "k16:2", "tia", 0.6, [0.0, 1.0])       # same style, cos 0
    c = _entry("c", "mandelbrot", "m#2", "k16:3", "stripe", 0.6, [0.0, 1.0])    # diff style, cos 0
    assert SEL.kernel(a, b) == pytest.approx(0.0)                # no floor → 0
    assert SEL.kernel(a, b, style_weight=0.5) == pytest.approx(0.5)
    assert SEL.kernel(a, c, style_weight=0.5) == pytest.approx(0.0)


def test_greedy_style_weight_spreads_modes():
    # 3 tia + 1 stripe, all morph-distinct, N=2; the style floor makes the 2nd pick switch
    # modes to stripe rather than take a 2nd (lower-score) tia.
    e = [_entry("t0", "mandelbrot", "m#0", "k16:1", "tia", 0.90, [1.0, 0.0, 0.0, 0.0]),
         _entry("t1", "mandelbrot", "m#1", "k16:1", "tia", 0.80, [0.0, 1.0, 0.0, 0.0]),
         _entry("t2", "mandelbrot", "m#2", "k16:1", "tia", 0.70, [0.0, 0.0, 1.0, 0.0]),
         _entry("s0", "mandelbrot", "m#3", "k16:1", "stripe", 0.60, [0.0, 0.0, 0.0, 1.0])]
    sel, _log = SEL.greedy_select(e, 2, style_weight=0.5)
    styles = {x["style"] for x in sel}
    assert styles == {"tia", "stripe"}         # spread, not two tia
    assert sel[0]["id"] == "t0"                # best tia first


def test_greedy_prefers_distinct_cells():
    # two near-duplicate entries in ONE cell + one entry in another cell; N=2 → one per cell.
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.95, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#0", "k16:1", "smooth", 0.90, [1.0, 0.0])
    c = _entry("c", "mandelbrot", "m#0", "k16:2", "smooth", 0.80, [0.0, 1.0])
    selected, log = SEL.greedy_select([a, b, c], 2)
    cells = {(e["type"], e["cluster"], e["flavor"], e["style"]) for e in selected}
    assert len(cells) == 2                     # spread across cells, not two from the crowded one
    assert {e["id"] for e in selected} == {"a", "c"}


def _rentry(id, part, cluster, score):
    """A `rank_select` entry (the greedy one above is `_entry`, with the coverage kernel's
    extra axes)."""
    return {"id": id, "type": part, "cluster": cluster, "flavor": "k16:1",
            "style": "smooth", "score": score}


def test_rank_select_takes_the_slot_map_as_the_authority_on_partitions():
    """A partition with candidates but NO slot emits nothing. The slot map is the allocation,
    and a selector that quietly served an unallocated partition would make the release-mix
    apportionment advisory."""
    entries = [_rentry("a", "mandelbrot", "m#0", 0.9), _rentry("b", "phoenix", "p#0", 0.95)]
    sel, _log = SEL.rank_select(entries, {"mandelbrot": 1}, {})
    assert [e["id"] for e in sel] == ["a"]


def test_rank_select_treats_an_absent_cap_as_uncapped_and_a_present_one_as_binding():
    """Both directions. A caller with no supply census must not be silently capped to zero;
    a caller that passes one must be held to it."""
    entries = [_rentry(f"e{k}", "mandelbrot", f"m#{k}", 0.9 - k * 0.01) for k in range(5)]
    assert len(SEL.rank_select(entries, {"mandelbrot": 3}, {})[0]) == 3
    assert len(SEL.rank_select(entries, {"mandelbrot": 3}, {"mandelbrot": 1})[0]) == 1
    assert SEL.rank_select(entries, {"mandelbrot": 3}, {"mandelbrot": 0})[0] == []


def test_rank_select_cluster_counter_is_shared_across_calls():
    """The per-RUN cap, as the pure property: one dict threaded through two calls."""
    used: dict = {}
    e1 = [_rentry("a", "mandelbrot", "m#0", 0.9), _rentry("b", "mandelbrot", "m#0", 0.8)]
    e2 = [_rentry("c", "mandelbrot", "m#0", 0.7)]
    assert len(SEL.rank_select(e1, {"mandelbrot": 2}, {}, used)[0]) == 2
    assert SEL.rank_select(e2, {"mandelbrot": 2}, {}, used)[0] == []
    assert used["m#0"] == SEL.CLUSTER_CAP
    # a fresh counter takes it again — the cap is state, not a property of the entries
    assert len(SEL.rank_select(e2, {"mandelbrot": 2}, {}, {})[0]) == 1


def test_niche_percentile_singleton_is_one():
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.5, [1.0])
    pct = SEL.niche_percentiles([a])
    assert pct["a"] == 1.0


# --------------------------------------------------------------------------- #
# descriptor.py — clustering + Location mapping.
# --------------------------------------------------------------------------- #
def test_cluster_incremental_join_and_new():
    items = [("a", np.array([1.0, 0.0, 0.0], np.float32)),
             ("b", np.array([1.0, 0.0, 0.0], np.float32)),   # cos 1 → joins a
             ("c", np.array([0.0, 1.0, 0.0], np.float32))]    # cos 0 → new
    assign = D.cluster_incremental(items, threshold=0.974)
    assert assign["a"] == assign["b"]
    assert assign["c"] != assign["a"]


def test_assign_morph_clusters_within_type():
    rows = [{"id": "a", "family": "mandelbrot"}, {"id": "b", "family": "mandelbrot"},
            {"id": "c", "family": "multibrot3"}]
    embs = {"a": np.array([1.0, 0.0], np.float32),
            "b": np.array([1.0, 0.0], np.float32),
            "c": np.array([1.0, 0.0], np.float32)}
    tags = D.assign_morph_clusters(rows, embs)
    assert tags["a"] == tags["b"] == "mandelbrot#0"
    assert tags["c"] == "multibrot3#0"          # different type → own namespace


def test_location_of_partition_mapping():
    m = D.location_of({"family": "mandelbrot", "outcome_cx": -0.5, "outcome_cy": 0.1,
                       "outcome_fw": 0.03})
    assert m.family == "mandelbrot" and m.c_re is None
    j = D.location_of({"family": "julia:multibrot3", "outcome_cx": 0.0, "outcome_cy": 0.0,
                       "outcome_fw": 3.0, "julia_c_re": 0.28, "julia_c_im": 0.008,
                       "julia_schema": "campaign"})
    assert j.family == "julia_multibrot3" and j.c_re == "0.28"


# --------------------------------------------------------------------------- #
# ACCEPTANCE — current-decode rejects an old-ledger v6 row.
# --------------------------------------------------------------------------- #
def _row(id, ver=None, dc=3, guard=True, distinct=True, cx=None, p_good=0.7, **extra):
    """A ledger row. `ver=None` means the CURRENT scorer version, resolved from the single
    source of truth (tools/scoring/active_ckpt) rather than hardcoded — these tests are about
    stale-decode semantics, so pinning the live version in them just breaks the suite at every
    flip for no signal.

    `p_good` is the STORED RAW P(>=3) the read-time ranked intake reads (`ranked_intake`); it
    defaults comfortably above the junk floor so a fixture that is not about the floor is not
    silently emptied by it."""
    # Distinct ids get distinct COORDINATES by default: the cross-ledger union dedups by
    # location identity, so a fixture that gives every row the same viewport is a fixture in
    # which every row is the same location.
    if cx is None:
        cx = -0.5 - sum(ord(ch) for ch in str(id)) * 1e-6
    row = {"id": id, "family": "mandelbrot", "outcome_cx": cx, "outcome_cy": 0.1,
           "outcome_fw": 0.03, "decoded_class": dc, "guard_pass": guard,
           "distinct": distinct, "p_good": p_good,
           "scorer_version": cc.active_scorer_version() if ver is None else ver}
    row.update(extra)
    return row


def test_stale_scorer_version_rows_rejected(tmp_path):
    cur = cc.active_scorer_version()
    assert cur and cur not in ("v6", "v5")   # sanity: the tokens below really are stale
    led = tmp_path / "outcome_ledger.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in [
        _row("cur"), _row("old", "v6"), _row("older", "v5"),
    ]) + "\n", encoding="utf-8")

    # soft form: stale rows silently skipped, only the current row admitted.
    admitted = D.load_admitted(led)
    assert [r["id"] for r in admitted] == ["cur"]

    # strict form: a v6 row RAISES rather than being consumed as a current verdict.
    only_v6 = tmp_path / "v6_only.jsonl"
    only_v6.write_text(json.dumps(_row("old", "v6")) + "\n", encoding="utf-8")
    with pytest.raises(cc.StaleDecodeError):
        D.load_admitted(only_v6, require_current=True)


# --------------------------------------------------------------------------- #
# ACCEPTANCE — append-only pool resume (no lost / duplicated entries).
# --------------------------------------------------------------------------- #
def _prec(id, loc, passed, cell):
    return {"id": id, "location_id": loc, "cell": list(cell), "passed": passed,
            "p_ge3": 0.8 if passed else 0.4}


def test_pool_resume_no_loss_no_dup(tmp_path):
    p = Pool(tmp_path)
    assert p.next_id() == "em_000000"
    cell = ("mandelbrot", "m#0", "k16:1", "smooth")
    p.append(_prec(p.next_id(), "loc0", True, cell))
    p.append(_prec(p.next_id(), "loc0", False, cell))
    p.append(_prec(p.next_id(), "loc1", True, cell))
    assert p.next_id() == "em_000003"

    # simulate kill + resume: a brand-new Pool over the same dir replays the durable log.
    q = Pool(tmp_path)
    assert q.n_attempts() == 3
    assert q.next_id() == "em_000003"                 # sequence continues, no collision
    assert [r["id"] for r in q.gated()] == ["em_000000", "em_000002"]
    assert q.attempts_per_location() == {"loc0": 2, "loc1": 1}
    # ids are unique (no duplication of a logged row)
    ids = [r["id"] for r in q.rows]
    assert len(ids) == len(set(ids))

    # a resumed append does not rewrite or duplicate prior rows.
    q.append(_prec(q.next_id(), "loc2", True, cell))
    r = Pool(tmp_path)
    assert [x["id"] for x in r.rows] == ["em_000000", "em_000001", "em_000002", "em_000003"]


# --------------------------------------------------------------------------- #
# The location-ranker tests LIVED HERE and were deleted on 2026-08-08 with their subject.
# `tools/ranker/` is gone: `pref_loc_v1` never existed on this checkout and its rebuild was
# DELETED permanently, not left blocked (deferred_recalibration.md § "Ranker rebuild —
# DELETED"). What they covered — `rank_percentiles` ties, and cache-hit/direct scoring parity
# — is not behaviour this tree has any more. The within-partition SEEDED SHUFFLE that
# replaced the ranker ordering is covered by `test_round_order_*` below, which is the live
# rule now rather than the fallback half of one.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# driver — per-head release floors + short-fill + multi-ledger intake dedup.
# --------------------------------------------------------------------------- #
from tools.emission import build_emission_diversity_v1 as B     # noqa: E402


def _args(tmp_path, **over):
    import argparse
    a = argparse.Namespace(
        ledger=["x.jsonl"], out=str(tmp_path / "scratch"), report=None, release_n=5,
        target_gated=0, floor=B.DEFAULT_FLOOR, mining_floor=B.DEFAULT_MINING_FLOOR,
        release_floor=B.DEFAULT_RELEASE_FLOOR, mining_release_floor=B.DEFAULT_MINING_RELEASE_FLOOR,
        strange_frac=B.DEFAULT_STRANGE_FRAC,
        max_attempts=240, time_budget_min=45.0, seed=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _gate_rec(id, loc, style, p_ge3, cell):
    return {"id": id, "location_id": loc, "type": cell[0], "morph_cluster": cell[1],
            "palette_flavor": cell[2], "render_style": style, "cell": list(cell),
            "p_ge3": p_ge3, "passed": True, "head": B.head_for_style(style)}


def _supply(eng, **per_partition):
    """Give a hand-built engine the intake supply census `select_release` reads. A pool-only
    fixture never ran `intake()`, and the thin-supply cap has to come from somewhere."""
    eng.passing_supply = dict(per_partition)
    eng.emit_caps = {p: RI.emit_cap(n) for p, n in per_partition.items()}
    return eng


def test_sub_floor_rows_are_eligible_now_and_the_retired_floors_annotate(tmp_path):
    """THE restructure, at the eligibility boundary. Both per-head release floors are
    ANNOTATION-ONLY as of 2026-08-09: a 0.80 smooth (below the retired 0.90) and a 0.30 strange
    (below the retired 0.50) are release-eligible and can ship, and the run still REPORTS what
    those cuts would have done.

    This is the inverse of the assertion that stood here from 2026-08-06, and it is written as
    a pair on purpose — eligibility widened AND the counterfactual survived. Widening alone
    would have deleted the old cut's value rather than retiring it."""
    eng = _supply(B.EmissionDiversity(_args(tmp_path)), mandelbrot=40)
    eng.embs = {}
    recs = [
        _gate_rec("em_0", "l0", "smooth", 0.95, ("mandelbrot", "m#0", "k16:1", "smooth")),
        _gate_rec("em_1", "l1", "smooth", 0.80, ("mandelbrot", "m#1", "k16:2", "smooth")),
        _gate_rec("em_2", "l2", "tia",    0.60, ("mandelbrot", "m#2", "k16:3", "tia")),
        _gate_rec("em_3", "l3", "tia",    0.30, ("mandelbrot", "m#3", "k16:4", "tia")),
    ]
    for r in recs:
        eng.pool.append(r)
    assert {r["id"] for r in eng.release_eligible()} == {"em_0", "em_1", "em_2", "em_3"}
    acct = eng.target_accounting()
    assert acct["post_floor"] == 4
    # the retired floors' verdict, kept as a number: 0.95 smooth and 0.60 strange clear them.
    assert acct["would_pass_release_floor"] == 2
    assert acct["would_pass_release_floor_smooth"] == 1
    assert acct["would_pass_release_floor_strange"] == 1
    assert acct["below_retired_release_floor"] == 2
    assert acct["cut_by_release_floor_strange"] == 1          # the 0.30 tia
    # and the annotation is per row, per head
    by_id = {r["id"]: r for r in eng.pool.rows}
    assert eng.would_pass_release_floor(by_id["em_1"]) is False   # 0.80 smooth, retired 0.90
    assert eng.would_pass_release_floor(by_id["em_2"]) is True    # 0.60 tia,    retired 0.50


def test_a_release_now_ships_rows_the_retired_floors_would_have_cut(tmp_path):
    """The consequence at SELECTION, not just at eligibility: with 4 slots and one partition
    with ample supply, all four rows ship — including the two below the retired floors. Under
    the pre-2026-08-09 rule this release was 2 tiles."""
    eng = _supply(B.EmissionDiversity(_args(tmp_path, release_n=4)), mandelbrot=40)
    eng.embs = {}
    for r in (_gate_rec("em_0", "l0", "smooth", 0.95, ("mandelbrot", "m#0", "k16:1", "smooth")),
              _gate_rec("em_1", "l1", "smooth", 0.80, ("mandelbrot", "m#1", "k16:2", "smooth")),
              _gate_rec("em_2", "l2", "tia",    0.60, ("mandelbrot", "m#2", "k16:3", "tia")),
              _gate_rec("em_3", "l3", "tia",    0.30, ("mandelbrot", "m#3", "k16:4", "tia"))):
        eng.pool.append(r)
    selected, _log = eng.select_release()
    assert {e["_rec"]["id"] for e in selected} == {"em_0", "em_1", "em_2", "em_3"}
    # head split intact — two per head, never compared in one step
    assert eng.release_split["smooth_selected"] == 2
    assert eng.release_split["strange_selected"] == 2


def test_a_thin_partition_emits_nothing(tmp_path):
    """§3. `emit = min(slots, floor(passing_supply / 4))`. A partition with 3 floor-passing
    candidates mined emits ZERO even though it has a 0.99 row sitting in the pool — the point
    of the rule is that it refuses to ship a partition's least-bad row. Paired with the ample
    case above, which is the same code path with supply 40."""
    eng = _supply(B.EmissionDiversity(_args(tmp_path, release_n=4)), mandelbrot=3)
    eng.embs = {}
    eng.pool.append(_gate_rec("em_0", "l0", "smooth", 0.99,
                              ("mandelbrot", "m#0", "k16:1", "smooth")))
    selected, _log = eng.select_release()
    assert selected == []
    assert eng.release_short_fill["short_by"] == 4
    # ...and one more mined-and-passing row flips it on: 4 // 4 == 1
    eng2 = _supply(B.EmissionDiversity(_args(tmp_path / "b", release_n=4)), mandelbrot=4)
    eng2.embs = {}
    eng2.pool.append(_gate_rec("em_0", "l0", "smooth", 0.99,
                               ("mandelbrot", "m#0", "k16:1", "smooth")))
    assert len(eng2.select_release()[0]) == 1


def test_at_most_two_picks_per_morph_cluster_per_run(tmp_path):
    """§4. Three smooth rows in ONE morph cluster, four slots, ample supply: two ship and the
    third is passed over by the cluster cap, with the skip named in the log. The fourth slot
    goes unfilled rather than being handed back to the saturated cluster."""
    eng = _supply(B.EmissionDiversity(_args(tmp_path, release_n=8, strange_frac=0.0)),
                  mandelbrot=40)
    eng.embs = {}
    for k, p in enumerate((0.9, 0.8, 0.7)):
        eng.pool.append(_gate_rec(f"em_{k}", f"l{k}", "smooth", p,
                                  ("mandelbrot", "m#0", f"k16:{k}", "smooth")))
    selected, log = eng.select_release()
    assert [e["_rec"]["id"] for e in selected] == ["em_0", "em_1"]      # best two, by rank
    skips = [l for l in log if l.get("skip") == "cluster_cap"]
    assert [l["id"] for l in skips] == ["em_2"]
    assert eng.release_split["n_cluster_cap_skips"] == 1


def test_the_cluster_cap_is_per_run_not_per_head_pass(tmp_path):
    """The counter is threaded through BOTH head passes. Two smooth and two strange rows share
    one morph cluster; without a shared counter each pass would take its own two and the run
    would ship four tiles of one look."""
    eng = _supply(B.EmissionDiversity(_args(tmp_path, release_n=8, strange_frac=0.5)),
                  mandelbrot=40)
    eng.embs = {}
    for k, (style, p) in enumerate((("smooth", 0.9), ("smooth", 0.8),
                                    ("tia", 0.7), ("tia", 0.6))):
        eng.pool.append(_gate_rec(f"em_{k}", f"l{k}", style, p,
                                  ("mandelbrot", "m#0", f"k16:{k}", style)))
    selected, _log = eng.select_release()
    assert len(selected) == 2, [e["_rec"]["id"] for e in selected]
    assert {e["_rec"]["id"] for e in selected} == {"em_0", "em_1"}


def test_selection_is_rank_order_within_a_partition(tmp_path):
    """Top-N by the head's own score. Not the greedy coverage rule, which could prefer a
    lower-scoring row for being unlike what was already picked."""
    eng = _supply(B.EmissionDiversity(_args(tmp_path, release_n=2, strange_frac=0.0)),
                  mandelbrot=40)
    eng.embs = {}
    for k, p in enumerate((0.30, 0.95, 0.60)):
        eng.pool.append(_gate_rec(f"em_{k}", f"l{k}", "smooth", p,
                                  ("mandelbrot", f"m#{k}", "k16:1", "smooth")))
    selected, _log = eng.select_release()
    assert [e["_rec"]["id"] for e in selected] == ["em_1", "em_2"]      # 0.95 then 0.60


def test_multi_ledger_intake_dedups_by_location_and_namespaces_ids(tmp_path):
    """A row appearing at the SAME location in two ledgers is one location: dropped,
    first-ledger wins. Surviving ids are namespaced by ledger and carry their source."""
    l1 = tmp_path / "a.jsonl"
    l2 = tmp_path / "b.jsonl"
    l1.write_text(json.dumps(_row("shared")) + "\n"
                  + json.dumps(_row("only_a")) + "\n", encoding="utf-8")
    l2.write_text(json.dumps(_row("shared")) + "\n"      # same id AND same location
                  + json.dumps(_row("only_b")) + "\n", encoding="utf-8")
    eng = B.EmissionDiversity(_args(tmp_path, ledger=[str(l1), str(l2)]))
    rows = eng._load_all_admitted()
    assert [r["_ledger_row_id"] for r in rows] == ["shared", "only_a", "only_b"]
    ns1, ns2 = D.ledger_namespace(l1), D.ledger_namespace(l2)
    assert ns1 != ns2
    assert [r["id"] for r in rows] == [D.namespaced_id(ns1, "shared"),
                                       D.namespaced_id(ns1, "only_a"),
                                       D.namespaced_id(ns2, "only_b")]
    src = {r["_ledger_row_id"]: r["_source_ledger"] for r in rows}
    assert src["shared"].endswith("a.jsonl") and src["only_b"].endswith("b.jsonl")


def test_run_scoped_id_collision_no_longer_aliases_two_locations(tmp_path):
    """THE un-abort. The same run-scoped id naming DIFFERENT locations in two ledgers used to
    raise (and before that would have silently dropped a distinct wallpaper). Namespacing by
    ledger keeps both, and the collision is still counted so the fix stays visible.

    Injection proof of the aliasing it prevents: with the namespace forced to a constant, the
    two locations collapse to one row and the union under-counts."""
    def _at(id, cx):
        r = _row(id)
        r["outcome_cx"] = cx
        return r
    l1 = tmp_path / "a.jsonl"
    l2 = tmp_path / "b.jsonl"
    l1.write_text(json.dumps(_at("st_x", -0.5)) + "\n", encoding="utf-8")
    l2.write_text(json.dumps(_at("st_x", 0.9)) + "\n", encoding="utf-8")   # SAME id, other coord
    rows, diag = D.load_union_admitted([l1, l2])
    assert diag["n_union"] == 2 and diag["n_id_collisions"] == 1
    assert len({r["id"] for r in rows}) == 2
    assert {r["outcome_cx"] for r in rows} == {-0.5, 0.9}     # both locations survive
    eng = B.EmissionDiversity(_args(tmp_path, ledger=[str(l1), str(l2)]))
    assert len(eng._load_all_admitted()) == 2

    # same id + IDENTICAL location is one location, deduped (0 today, kept dedupable)
    l2.write_text(json.dumps(_at("st_x", -0.5)) + "\n", encoding="utf-8")
    rows2, diag2 = D.load_union_admitted([l1, l2])
    assert diag2["n_union"] == 1 and diag2["n_location_overlaps"] == 1
    assert rows2[0]["_ledger_row_id"] == "st_x"


def test_a_constant_namespace_would_alias_the_collision(tmp_path, monkeypatch):
    """The injection the test above rests on: namespacing is what separates them, not luck."""
    def _at(id, cx):
        return _row(id, cx=cx)
    l1, l2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    l1.write_text(json.dumps(_at("st_x", -0.5)) + "\n", encoding="utf-8")
    l2.write_text(json.dumps(_at("st_x", 0.9)) + "\n", encoding="utf-8")
    monkeypatch.setattr(D, "ledger_namespace", lambda _p: "same")
    with pytest.raises(D.LedgerNamespaceCollision):
        D.load_union_admitted([l1, l2])


def test_two_ledgers_in_one_directory_get_distinct_namespaces(tmp_path):
    """`outcome_ledger.jsonl` beside `outcome_ledger_v7_t45.jsonl` is a real shape in the
    tree; a parent-directory-only namespace would collide on it."""
    d = tmp_path / "run"
    d.mkdir()
    assert D.ledger_namespace(d / "outcome_ledger.jsonl") \
        != D.ledger_namespace(d / "outcome_ledger_v7_t45.jsonl")


def test_deficit_rebuild_from_pool_log(tmp_path):
    """The build_axes resume path: replaying the pool log reproduces fill+attempt counts."""
    cells = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1"], ["smooth", "tia"])
    tm = _measure(cells, attempt_cap=99)
    p = Pool(tmp_path)
    recs = [_prec(p.next_id(), "loc0", True, cells[0]),
            _prec(p.next_id(), "loc0", False, cells[0]),
            _prec(p.next_id(), "loc0", True, cells[1])]
    for rc in recs:
        p.append(rc)
    q = Pool(tmp_path)
    m = C.DeficitModel(cells, tm)
    for rr in q.rows:
        cell = tuple(rr["cell"])
        m.record_attempt(cell)
        if rr["passed"]:
            m.record_fill(cell)
    assert m.attempt_counts[cells[0]] == 2 and m.fill_counts[cells[0]] == 1
    assert m.attempt_counts[cells[1]] == 1 and m.fill_counts[cells[1]] == 1


# --------------------------------------------------------------------------- #
# --target-gated: the POST-FLOOR surplus contract (emission_floors_prompt.md D).
# --------------------------------------------------------------------------- #
def _target_pool(tmp_path, **over):
    """A pool holding one post-floor smooth, one post-floor strange, and three sub-release-
    floor strange rows — the shape that made `--target-gated` count rows no floor had
    vouched for, back when the mining release floor was report-only."""
    eng = B.EmissionDiversity(_args(tmp_path, **over))
    eng.embs = {}
    recs = [
        _gate_rec("em_0", "l0", "smooth", 0.95, ("mandelbrot", "m#0", "k16:1", "smooth")),
        _gate_rec("em_1", "l1", "tia",    0.70, ("mandelbrot", "m#1", "k16:1", "tia")),
        _gate_rec("em_2", "l2", "tia",    0.30, ("mandelbrot", "m#2", "k16:2", "tia")),
        _gate_rec("em_3", "l3", "tia",    0.26, ("mandelbrot", "m#3", "k16:3", "tia")),
        _gate_rec("em_4", "l4", "stripe", 0.40, ("mandelbrot", "m#4", "k16:4", "stripe")),
    ]
    for r in recs:
        eng.pool.append(r)
    return eng


def test_the_target_counts_every_scored_row_now(tmp_path):
    """What `--target-gated` counts moved with the floors. All five rows scored, so all five
    count — and the accounting still says how many of them the RETIRED floors would have kept
    (the 0.95 smooth and the 0.70 tia), which is the number that makes the weaker surplus
    readable instead of invisible.

    The inverse assertion (`post_floor == 2`) stood here from 2026-08-06 to 2026-08-09."""
    eng = _target_pool(tmp_path)
    assert {r["id"] for r in eng.release_eligible()} == {f"em_{k}" for k in range(5)}
    acct = eng.target_accounting()
    assert acct["post_floor"] == 5 and acct["release_eligible"] == 5
    assert acct["post_floor_smooth"] == 1 and acct["post_floor_strange"] == 4
    assert acct["would_pass_release_floor"] == 2              # 0.95 smooth + 0.70 tia
    assert acct["below_retired_release_floor"] == 3
    assert acct["cut_by_release_floor_strange"] == 3          # 0.30 / 0.26 / 0.40
    assert {r["id"] for r in eng.post_floor()} == {f"em_{k}" for k in range(5)}


def test_post_floor_is_an_identity_on_eligible(tmp_path):
    """`post_floor() == release_eligible()`. It was an identity maintained by two separately
    computed floor predicates until 2026-08-09; it is now the same list, and the divergence
    that identity was watching for is reported by name (`would_pass_release_floor`)."""
    eng = _target_pool(tmp_path)
    assert [r["id"] for r in eng.post_floor()] == [r["id"] for r in eng.release_eligible()]


def test_an_unscored_row_is_not_eligible(tmp_path):
    """NON-VACUITY for "every scored row is eligible": the absence of a verdict is still not a
    verdict. A render error must not enter a draw ordered by score."""
    eng = B.EmissionDiversity(_args(tmp_path))
    eng.embs = {}
    ok = _gate_rec("em_0", "l0", "smooth", 0.10, ("mandelbrot", "m#0", "k16:1", "smooth"))
    bad = _gate_rec("em_1", "l1", "smooth", None, ("mandelbrot", "m#1", "k16:2", "smooth"))
    bad["passed"], bad["error"] = False, "render failed"
    eng.pool.append(ok)
    eng.pool.append(bad)
    assert {r["id"] for r in eng.release_eligible()} == {"em_0"}   # 0.10 is BELOW every floor
    assert eng.target_accounting()["post_floor"] == 1


def test_a_target_of_six_is_not_met_by_five_scored_rows(tmp_path):
    """THE break condition (`target_met`, which `run_colorize` reads verbatim)."""
    assert _target_pool(tmp_path, target_gated=6).target_met() is False
    assert _target_pool(tmp_path / "b", target_gated=5).target_met() is True
    assert _target_pool(tmp_path / "c", target_gated=2).target_met() is True


def test_the_default_target_is_three_times_release_n(tmp_path):
    """The default is unchanged; only what it COUNTS moved."""
    eng = B.EmissionDiversity(_args(tmp_path, release_n=12, target_gated=0))
    assert eng.target_gated == 36


def test_the_retired_floor_annotation_is_per_head(tmp_path):
    """Non-vacuity for the annotation: it is by HEAD, not one global number. A strange row at
    0.70 clears the retired mining 0.50; a smooth row at 0.70 does not clear the retired
    wallpaper 0.90. Both are eligible either way — that is the difference from before."""
    eng = B.EmissionDiversity(_args(tmp_path))
    eng.embs = {}
    for r in (_gate_rec("em_a", "la", "smooth", 0.70, ("mandelbrot", "a#0", "k16:1", "smooth")),
              _gate_rec("em_b", "lb", "tia",    0.70, ("mandelbrot", "b#0", "k16:1", "tia"))):
        eng.pool.append(r)
    by_id = {r["id"]: r for r in eng.pool.rows}
    assert eng.would_pass_release_floor(by_id["em_a"]) is False
    assert eng.would_pass_release_floor(by_id["em_b"]) is True
    assert {r["id"] for r in eng.post_floor()} == {"em_a", "em_b"}


# --------------------------------------------------------------------------- #
# gate_report: the POOL-site pairing (emission_floors_prompt.md C).
# --------------------------------------------------------------------------- #
def test_gate_report_pairs_the_pool_site_like_the_release_site(tmp_path, monkeypatch):
    from tools.mining import gate_report as GR
    monkeypatch.setattr(GR, "GATE_LOG_DIR", tmp_path / "gr")
    rows = [
        GR.gate_report_row(site="s", key="k_below", location={}, style="tia", palette="p",
                           p_ge3=0.10, release_threshold=0.50, pool_floor=0.25,
                           pooled=False, selected=True, selection_stage="release"),
        GR.gate_report_row(site="s", key="k_mid", location={}, style="tia", palette="p",
                           p_ge3=0.40, release_threshold=0.50, pool_floor=0.25,
                           pooled=True, selected=False, selection_stage="release"),
    ]
    below, mid = rows
    # release site — unchanged
    assert below["would_cut"] and mid["would_cut"]
    # pool site — the new pairing, both fields present and joinable
    assert below["would_cut_pool"] is True and below["pooled"] is False
    assert mid["would_cut_pool"] is False and mid["pooled"] is True
    path, n_tot, n_cut, n_cut_sel, pool_c = GR.write_gate_report("s", rows)
    assert (n_tot, n_cut, n_cut_sel) == (2, 2, 1)
    assert pool_c == {"n_with_pool_site": 2, "n_would_cut_pool": 1,
                      "n_would_cut_pool_pooled": 0, "n_would_cut_pool_selected": 1}
    assert path.exists()


def test_a_site_with_no_pool_stage_logs_no_pool_pairing(tmp_path, monkeypatch):
    """`deploy_tail` has no pool stage. Its rows must not grow a `pooled` field claiming an
    outcome nobody reported, and its pool counts must read zero rather than zero-of-zero
    dressed as a measurement."""
    from tools.mining import gate_report as GR
    monkeypatch.setattr(GR, "GATE_LOG_DIR", tmp_path / "gr2")
    row = GR.gate_report_row(site="deploy_tail", key="k", location={}, style="tia",
                             palette="p", p_ge3=0.10, release_threshold=0.50,
                             selected=False, selection_stage="keeper")
    assert "pooled" not in row and "would_cut_pool" not in row
    _p, _t, _c, _cs, pool_c = GR.write_gate_report("deploy_tail", [row])
    assert pool_c["n_with_pool_site"] == 0


def test_an_unreported_pool_outcome_stays_null_not_false(tmp_path):
    """`pooled=None` means "nobody said", which is not the same claim as "not pooled" — a
    calibration pass that cannot tell them apart counts silence as a negative."""
    from tools.mining import gate_report as GR
    row = GR.gate_report_row(site="s", key="k", location={}, style="tia", palette="p",
                             p_ge3=0.10, release_threshold=0.50, pool_floor=0.25,
                             selected=False, selection_stage="release")
    assert row["pooled"] is None and row["would_cut_pool"] is True


def test_a_legacy_pool_row_still_counts_as_would_cut(tmp_path, monkeypatch):
    """A gate-report file is a MIX of formats after any partial re-run: the upsert preserves
    keys it did not touch, so rows accrued before the pool pairing landed carry
    `would_pass_pool` and no `would_cut_pool`. `write_gate_report` derives the complement at
    read time; trusting the stored field would count every legacy row as "not would-cut" and
    quietly shrink the denominator a calibration pass reads."""
    import json as _json
    from tools.mining import gate_report as GR
    d = tmp_path / "gr3"
    d.mkdir()
    monkeypatch.setattr(GR, "GATE_LOG_DIR", d)
    legacy = {"site": "s", "key": "old", "location": {}, "style": "tia", "palette": "p",
              "gate_version": "mining_v1", "p_ge3": 0.10, "release_threshold": 0.5,
              "would_pass": False, "would_cut": True, "selected": True,
              "selection_stage": "release", "pool_floor": 0.25, "would_pass_pool": False}
    (d / "s.jsonl").write_text(_json.dumps(legacy) + "\n", encoding="utf-8")
    fresh = GR.gate_report_row(site="s", key="new", location={}, style="tia", palette="p",
                               p_ge3=0.05, release_threshold=0.50, pool_floor=0.25,
                               pooled=False, selected=False, selection_stage="release")
    _p, _t, _c, _cs, pool_c = GR.write_gate_report("s", [fresh])
    assert pool_c["n_with_pool_site"] == 2
    assert pool_c["n_would_cut_pool"] == 2          # 1 legacy (derived) + 1 fresh (stored)
    assert pool_c["n_would_cut_pool_selected"] == 1


# --------------------------------------------------------------------------- #
# pick_location — within-round order.
#
# Fewest-attempts-first only preserves coverage when the budget covers the whole union;
# under any smaller budget the run never leaves round 0 and the WITHIN-round order IS the
# selection. It used to be `id`, so the 200-attempt smoke colorized `ids[0:200]` — campaign1
# only, four source ledgers at exactly zero. These pin the replacement: seeded round-robin
# across the partitions present, ranker-ordered inside a partition when the ranker artifact
# resolved and seeded-shuffled when it did not.
#
# The ids below are deliberately `<partition>_<k>`, so alphabetical order == one partition at
# a time: the old behaviour is exactly the failure these must not allow.
# --------------------------------------------------------------------------- #
class _FakePool:
    """`attempts_per_location` is all `pick_location` reads of the pool."""

    def __init__(self):
        self.counts: dict = {}

    def attempts_per_location(self) -> dict:
        return dict(self.counts)

    def record(self, rid) -> None:
        self.counts[rid] = self.counts.get(rid, 0) + 1


def _driver(sizes: dict, seed=7, scores=None):
    """A bare EmissionDiversity carrying only what pick_location touches (no sinks, no heads,
    no intake) — `object.__new__` deliberately, so this stays torch-free and render-free.

    `scores` is `{id: raw p_good}`; the rank index is built from it exactly as `_index_ranks`
    does. Default: score DESCENDING in id order, so "rank order" and "id order" coincide and a
    test that means to distinguish them has to say so (the shuffle tests used to rely on the
    opposite convention, which is why this is spelled out)."""
    from tools.emission.build_emission_diversity_v1 import EmissionDiversity
    d = object.__new__(EmissionDiversity)
    d.rows, d.partition_of = [], {}
    for p in sorted(sizes):
        for k in range(sizes[p]):
            rid = f"{p}_{k:04d}"
            d.rows.append({"id": rid, "p_good": (scores or {}).get(rid, 1.0 - k * 1e-4)})
            d.partition_of[rid] = p
    d.ranker_score = {}      # permanently empty since the ranker was deleted (2026-08-08)
    d.seed = seed
    d.pool = _FakePool()
    d._round_idx = None
    d._round_queue = None
    # no supply census here: `_index_ranks` also builds one, and these fixtures use invented
    # partition names that `cell_partition` would (correctly) refuse. These tests are about
    # the ORDER; the census is exercised in test_ranked_intake.py.
    d.mined_rows, d.intake_scope = [], None
    d._index_ranks()
    return d


def _run(d, budget: int, exhausted=None) -> list:
    exhausted = set() if exhausted is None else exhausted
    picked = []
    for _ in range(budget):
        row = d.pick_location(exhausted)
        if row is None:
            break
        d.pool.record(row["id"])
        picked.append(row["id"])
    return picked


def _counts_by_partition(d, picked) -> dict:
    out = {p: 0 for p in set(d.partition_of.values())}
    for i in picked:
        out[d.partition_of[i]] += 1
    return out


SIZES = {"aaa": 100, "bbb": 60, "ccc": 30, "ddd": 8, "eee": 3}      # 201 locations


def test_a_budget_smaller_than_the_union_still_reaches_every_partition():
    """THE regression. Budget 40 of 201: every partition with admitted rows gets a share, and
    the share is max-min fair — a partition may only trail another by more than one attempt if
    it has been exhausted. Stated as the fairness property rather than as an expected vector,
    so it does not re-derive the allocation through the code under test."""
    d = _driver(SIZES)
    picked = _run(d, 40)
    assert len(picked) == 40 and len(set(picked)) == 40
    got = _counts_by_partition(d, picked)
    assert all(v >= 1 for v in got.values()), got
    for p, a in got.items():
        for q, b in got.items():
            if a < b - 1:
                assert a == SIZES[p], (
                    f"{p} got {a} and {q} got {b} while {p} still had "
                    f"{SIZES[p] - a} unattempted rows")
    # and it is NOT the old alphabetical prefix
    assert picked != sorted(d.partition_of)[:40]
    assert got["aaa"] < SIZES["aaa"]


def test_the_old_alphabetical_prefix_would_have_starved_four_of_the_five():
    """The contrast that makes the assertion above non-vacuous: the order this replaced puts
    all 40 attempts in one partition. If this ever stops being true the fixture got easy."""
    old = sorted(f"{p}_{k:04d}" for p in SIZES for k in range(SIZES[p]))[:40]
    assert {i.split("_")[0] for i in old} == {"aaa"}


def test_partitions_are_derived_from_the_rows_not_a_literal():
    """A partition name that appears nowhere in the codebase gets its round-robin share. The
    ten base partitions are already a moving set (classic-phoenix splits off `family`), and a
    hardcoded roster would send a new one to the tail — where a bounded budget never arrives."""
    sizes = {"mandelbrot": 50, "a_brand_new_partition_v9": 20}
    d = _driver(sizes)
    got = _counts_by_partition(d, _run(d, 20))
    assert got["a_brand_new_partition_v9"] == 10 and got["mandelbrot"] == 10


# `test_within_a_partition_the_ranker_orders_when_it_resolves` stood here until 2026-08-08.
# It asserted that a populated `ranker_score` made the within-partition order
# ranker-descending — true of the code and true of nothing else: the ranker artifact never
# resolved on this checkout, so the branch it covered ran zero times in production while the
# test kept it green. Deleted with the ranker (deferred_recalibration.md § "Ranker growth —
# CLOSED").
#
# `test_the_within_partition_order_is_a_seeded_shuffle` stood here until 2026-08-09. The
# shuffle was replaced by READ-TIME RANK (raw P(>=3) descending) once the intake stopped
# discarding the probability in favour of the frozen `decoded_class`; the two tests below are
# the shuffle test's three properties restated for the rule that replaced it.


def test_the_within_partition_order_is_rank_order():
    """Best-first on the stored raw P(>=3), inside each partition. Deliberately scored so rank
    order is the REVERSE of id order — the two coincide under the fixture default, and a test
    that cannot tell them apart is the one that would have passed under the old shuffle too."""
    scores = {f"{p}_{k:04d}": k * 1e-3 for p in SIZES for k in range(SIZES[p])}
    d = _driver(SIZES, scores=scores)
    picked = _run(d, 60)
    for p in SIZES:
        seq = [i for i in picked if d.partition_of[i] == p]
        if len(seq) >= 4:
            assert seq == sorted(seq, reverse=True), f"{p} was not served best-first"


def test_the_within_partition_rank_order_is_deterministic_and_seed_independent():
    """Reproducibility, and the half that CHANGED: the within-partition order no longer moves
    with the seed, because it is not random any more. The seed still shuffles which partition
    leads each round, so the interleaved sequence may differ — the per-partition subsequence
    must not."""
    def per_partition(seed):
        d = _driver(SIZES, seed=seed)
        picked = _run(d, 60)
        return {p: [i for i in picked if d.partition_of[i] == p] for p in SIZES}
    a, b = per_partition(7), per_partition(7)
    assert a == b, "same seed must reproduce the run (batch reproducibility)"
    c = per_partition(8)
    for p in SIZES:
        n = min(len(a[p]), len(c[p]))
        assert a[p][:n] == c[p][:n], f"{p}'s rank order moved with the seed"


def test_an_unranked_location_goes_to_the_tail_not_the_front():
    """A row the rank index does not know (a resume against a pool whose intake snapshot has
    moved on) must sort LAST. Sorting an unknown to the front would let a location with no
    score outrank every scored one, silently, on exactly the path a resume takes."""
    d = _driver({"aaa": 4})
    d.rank_of.pop("aaa_0000")                      # the best-ranked row loses its rank
    picked = _run(d, 4)
    assert picked[-1] == "aaa_0000"


def test_coverage_still_comes_before_any_second_colorize():
    """The property the round-robin must not cost: no location gets a 2nd attempt while any
    location has none. `--cover-all`'s stop condition reads exactly this."""
    sizes = {"aaa": 7, "bbb": 5, "ccc": 3}
    d = _driver(sizes)
    picked = _run(d, 15)
    assert sorted(picked) == sorted(d.partition_of)
    picked += _run(d, 5)
    assert all(v == 1 for v in d.pool.counts.values() if v) or len(picked) == 20
    assert max(d.pool.counts.values()) == 2 and min(d.pool.counts.values()) == 1


def test_a_rebuilt_queue_mid_round_still_serves_every_partition():
    """A `--resume` (or a location going `exhausted`) drops the in-memory queue mid-round; the
    order is rebuilt from the DURABLE pool counts. Coverage must survive that, and it must be
    deterministic — the same interruption twice gives the same run."""
    def interrupted():
        d = _driver(SIZES)
        out = _run(d, 17)
        d._round_idx, d._round_queue = None, None       # the resume
        return d, out + _run(d, 23)
    d, picked = interrupted()
    _d2, again = interrupted()
    assert picked == again
    assert len(set(picked)) == 40
    assert all(v >= 1 for v in _counts_by_partition(d, picked).values())


def test_exhausted_locations_are_skipped_without_stalling_the_round():
    """A location whose cells are all capped comes back as `exhausted`; the round must step
    over it rather than re-offering it or ending early."""
    sizes = {"aaa": 6, "bbb": 6}
    d = _driver(sizes)
    dead = {f"aaa_{k:04d}" for k in range(4)}
    picked = _run(d, 8, exhausted=dead)
    assert not (set(picked) & dead)
    assert len(set(picked)) == 8, "round 0 must cover every LIVE location exactly once"
    assert _counts_by_partition(d, picked) == {"aaa": 2, "bbb": 6}
    d.pick_location(dead)                       # the round advances rather than ending
    assert d.pick_location(dead)["id"] not in dead
