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
def _row(id, ver=None, dc=3, guard=True, distinct=True, cx=None):
    """A ledger row. `ver=None` means the CURRENT scorer version, resolved from the single
    source of truth (tools/scoring/active_ckpt) rather than hardcoded — these tests are about
    stale-decode semantics, so pinning the live version in them just breaks the suite at every
    flip for no signal."""
    # Distinct ids get distinct COORDINATES by default: the cross-ledger union dedups by
    # location identity, so a fixture that gives every row the same viewport is a fixture in
    # which every row is the same location.
    if cx is None:
        cx = -0.5 - sum(ord(ch) for ch in str(id)) * 1e-6
    return {"id": id, "family": "mandelbrot", "outcome_cx": cx, "outcome_cy": 0.1,
            "outcome_fw": 0.03, "decoded_class": dc, "guard_pass": guard,
            "distinct": distinct,
            "scorer_version": cc.active_scorer_version() if ver is None else ver}


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
# ranker (pref_loc_v0) — percentiles + cache-only scoring parity (render-free).
# --------------------------------------------------------------------------- #
from tools.ranker.score_locations import rank_percentiles, LocationRanker, DEFAULT_FEATURES  # noqa: E402


def test_rank_percentiles_ties_share_higher_rank():
    pct = rank_percentiles({"a": 1.0, "b": 2.0, "c": 2.0})
    assert pct["a"] == pytest.approx(1 / 3)      # smallest → bottom third
    assert pct["b"] == pct["c"] == 1.0           # ties both count each other as <= → top
    assert rank_percentiles({}) == {}
    assert rank_percentiles({"solo": 5.0})["solo"] == 1.0


@pytest.mark.skipif(not (ROOT / "data/ranker/pref_loc_v0/model.npz").exists()
                    or not DEFAULT_FEATURES.exists(),
                    reason="pref_loc_v0 artifacts absent")
def test_location_ranker_cache_hit_matches_direct_scoring():
    from tools.ranker.scorer import RankerScorer
    z = np.load(DEFAULT_FEATURES, allow_pickle=True)
    s = RankerScorer.load()
    direct = {str(z["ids"][k]): float(v)
              for k, v in enumerate(s.score_matrix({b: z[b] for b in s.sets}))}
    lr = LocationRanker()
    rows = [{"id": str(i)} for i in z["ids"]]
    mine = lr.score_rows(rows, ROOT / "scratch" / "_test_ranker_tiles")   # all cache hits
    assert lr._stack is None                     # torch feature stack never loaded
    assert max(abs(mine[i] - direct[i]) for i in direct) < 1e-9


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
        intake_floor=None,
        strange_frac=B.DEFAULT_STRANGE_FRAC,
        max_attempts=240, time_budget_min=45.0, seed=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _gate_rec(id, loc, style, p_ge3, cell):
    return {"id": id, "location_id": loc, "type": cell[0], "morph_cluster": cell[1],
            "palette_flavor": cell[2], "render_style": style, "cell": list(cell),
            "p_ge3": p_ge3, "passed": True, "head": B.head_for_style(style)}


def test_release_floors_exclude_subfloor_and_short_fill(tmp_path):
    # Wallpaper-head RELEASE floor (0.90) still CUTS a sub-floor smooth row (em_1). The mining
    # head is REPORT-ONLY since b515017: its 0.50 release floor no longer cuts, so a sub-floor
    # strange row (em_3) is release-eligible anyway (would-cut is logged, not acted on). See
    # prompts/mining_gate_report_only.md / release_eligible().
    eng = B.EmissionDiversity(_args(tmp_path))
    eng.embs = {}
    # 4 pool-admitted rows in distinct cells: smooth floor cuts em_1; strange admits all scored.
    recs = [
        _gate_rec("em_0", "l0", "smooth", 0.95, ("mandelbrot", "m#0", "k16:1", "smooth")),  # ≥0.90 ✓
        _gate_rec("em_1", "l1", "smooth", 0.80, ("mandelbrot", "m#1", "k16:2", "smooth")),  # <0.90 CUT
        _gate_rec("em_2", "l2", "tia",    0.60, ("mandelbrot", "m#2", "k16:3", "tia")),     # scored ✓
        _gate_rec("em_3", "l3", "tia",    0.30, ("mandelbrot", "m#3", "k16:4", "tia")),     # <0.50 but report-only ✓
    ]
    for r in recs:
        eng.pool.append(r)
    elig = {r["id"] for r in eng.release_eligible()}
    assert elig == {"em_0", "em_2", "em_3"}              # smooth floor cuts em_1; strange admits all scored
    selected, _log = eng.select_release()
    sel_ids = {e["_rec"]["id"] for e in selected}
    assert sel_ids == {"em_0", "em_2", "em_3"}           # smooth head still short-fills below its floor
    sf = eng.release_short_fill
    assert (sf["requested"], sf["eligible"], sf["selected"], sf["short_by"]) == (5, 3, 3, 2)
    # head-split: one smooth (wallpaper) + two strange (mining), never compared in one step
    assert eng.release_split["smooth_selected"] == 1 and eng.release_split["strange_selected"] == 2


def test_release_floor_per_head_boundary(tmp_path):
    # a mining tile at exactly 0.50 is eligible; a smooth at 0.50 is NOT (its floor is 0.90).
    eng = B.EmissionDiversity(_args(tmp_path))
    eng.embs = {}
    eng.pool.append(_gate_rec("em_0", "l0", "tia", 0.50, ("mandelbrot", "m#0", "k16:1", "tia")))
    eng.pool.append(_gate_rec("em_1", "l1", "smooth", 0.50, ("mandelbrot", "m#1", "k16:2", "smooth")))
    assert {r["id"] for r in eng.release_eligible()} == {"em_0"}


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
