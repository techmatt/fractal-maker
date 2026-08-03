"""The pop quota: the currency, the floor, and the property v1's root weights did not have.

The load-bearing test here is `test_the_quota_converges_the_realized_mix_...`: it drives the
pure pop decision through a simulated run in which one partition's queue MULTIPLIES far faster
than the others (the julia-hook / seed-pool shape that realized 19.6% against an intended 70%
in `discovery_pipeline.md` §3.1) and asserts the realized mix still lands on the intent. A
quota that only *steers* passes every other test in this file and fails that one.

  uv run pytest tools/atlas/test_pop_quota.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pop_quota as pq   # noqa: E402

PARTS = ["a", "b", "c", "d"]


# =========================================================================== #
# 1. the currency
# =========================================================================== #
def test_currency_weights_are_the_stated_ones():
    """n4 + 0.1*n3, and nothing else contributes. A class-2 leaking in at any weight would
    make the deficit a volume measure instead of a quality one."""
    assert pq.CLASS_WEIGHT == {4: 1.0, 3: 0.1}
    assert pq.CLASS_WEIGHT.get(2, 0.0) == 0.0 and pq.CLASS_WEIGHT.get(1, 0.0) == 0.0


def test_uniform_target_levels_to_the_richest_and_the_richest_has_zero_deficit():
    cur = {"a": 10.0, "b": 4.0, "c": 0.0}
    d = pq.deficits_from_currency(cur, ["a", "b", "c"])
    assert d == {"a": 0.0, "b": 6.0, "c": 10.0}


def test_a_partition_absent_from_the_census_carries_the_full_deficit():
    """Not-yet-mined and mined-nothing are the same demand, and a KeyError here would be the
    silent kind — the partition would simply never be allocated."""
    d = pq.deficits_from_currency({"a": 8.0}, ["a", "new"])
    assert d["new"] == 8.0


def test_label_currency_reads_the_amendment_overlay_and_reports_the_default_route(tmp_path,
                                                                                  monkeypatch):
    """PRESENCE-FROM-DISK, not an injected dict: the census must go through
    `corpus_reader.iter_labeled`, which is where the amendment overlay lives. A row whose
    render block has no `fractal_type` takes the tree's `mandelbrot` default AND is counted,
    so a default that carries currency can never be invisible."""
    corpus = tmp_path / "corpus"
    b = corpus / "batches" / "2026-08-03_t"
    b.mkdir(parents=True)
    (corpus / "batches" / "2026-08-03_t" / "images.jsonl").write_text("".join(
        json.dumps(r) + "\n" for r in [
            dict(image_id="i1", render=dict(fractal_type="phoenix"), label=dict(score=4)),
            dict(image_id="i2", render=dict(fractal_type="phoenix"), label=dict(score=3)),
            dict(image_id="i3", render=dict(fractal_type="julia"), label=dict(score=4)),
            dict(image_id="i4", render=dict(cx=0, cy=0), label=dict(score=4)),   # no type
            dict(image_id="i5", render=dict(fractal_type="phoenix"), label=dict(score=1)),
        ]), encoding="utf-8")
    (b / "batch.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("corpus_common.crops_dir", lambda bid: str(b / "crops"), raising=False)

    cen = pq.label_currency(["phoenix", "julia:mandelbrot", "mandelbrot"],
                            corpus_dir=str(corpus), library_globs=[])
    assert cen.currency["phoenix"] == pytest.approx(1.1)      # one 4 + one 3
    assert cen.currency["julia:mandelbrot"] == pytest.approx(1.0)
    assert cen.currency["mandelbrot"] == pytest.approx(1.0)   # the untyped row
    assert cen.defaulted_rows == 1
    assert cen.sources == {"label_corpus": 5, "library": 0}


def test_the_library_leg_is_derived_not_asserted(tmp_path, monkeypatch):
    """The wallpaper library carries zero scored rows TODAY. That zero has to come from
    reading it, or the census silently stops counting the library the day it starts holding
    human verdicts (`measurement_practice.md`: derive state in code)."""
    corpus = tmp_path / "corpus"
    (corpus / "batches" / "b").mkdir(parents=True)
    (corpus / "batches" / "b" / "images.jsonl").write_text("", encoding="utf-8")
    lib = tmp_path / "lib.jsonl"
    lib.write_text("".join(json.dumps(r) + "\n" for r in [
        dict(image_id="L1", render=dict(fractal_type="multibrot4"), label=dict(score=4)),
        dict(image_id="L2", render=dict(fractal_type="multibrot4"), label=dict(score=None)),
    ]), encoding="utf-8")
    cen = pq.label_currency(["multibrot4"], corpus_dir=str(corpus), library_globs=[str(lib)])
    assert cen.currency["multibrot4"] == pytest.approx(1.0)
    assert cen.sources["library"] == 1


# =========================================================================== #
# 2. the floor — the addendum's two sentences, as two tests
# =========================================================================== #
def test_a_zero_deficit_partition_still_gets_its_floor():
    a = pq.allocate({"a": 0.0, "b": 10.0, "c": 10.0, "d": 10.0},
                    {p: 1.0 for p in PARTS}, PARTS, floor=0.05)
    assert a.share["a"] == pytest.approx(0.05)
    assert "a" in a.floored and a.bucket("a") == "floor"


def test_a_partition_above_the_floor_gets_NOTHING_EXTRA():
    """The other half of the addendum's sentence, and the one a naive
    "reserve n*floor, then split the rest proportionally" implementation gets wrong: it hands
    a high-deficit partition its floor ON TOP of its proportional share.

    The deficits are DELIBERATELY UNEQUAL. With equal deficits the two formulas are
    algebraically identical — (1-kf)/(n-k) == f + (1-nf)/(n-k) — so an equal-deficit fixture
    cannot fail and would make this test vacuous (`verification_practice.md` §6)."""
    defs = {"a": 0.0, "b": 6.0, "c": 3.0, "d": 1.0}
    a = pq.allocate(defs, {p: 1.0 for p in PARTS}, PARTS, floor=0.05)
    pool = 1.0 - 0.05
    for p, w in (("b", 0.6), ("c", 0.3), ("d", 0.1)):
        assert a.share[p] == pytest.approx(pool * w)
        assert p not in a.floored
    naive_b = 0.05 + (1 - 4 * 0.05) * 0.6
    assert abs(a.share["b"] - naive_b) > 0.02, (a.share["b"], naive_b)


def test_shares_sum_to_one_and_every_partition_clears_the_floor():
    for defs in ({"a": 0, "b": 0, "c": 0, "d": 0},
                 {"a": 1e6, "b": 1.0, "c": 0.0, "d": 0.0},
                 {"a": 3, "b": 2, "c": 1, "d": 0.001}):
        a = pq.allocate(defs, {p: 1.0 for p in PARTS}, PARTS, floor=0.05)
        assert sum(a.share.values()) == pytest.approx(1.0)
        assert min(a.share.values()) >= 0.05 - 1e-9, (defs, a.share)


def test_the_floor_cascades_when_pinning_one_partition_pushes_another_under():
    """Water-filling, not one pass. Pinning the small partitions up to the floor shrinks the
    pool for everyone else, which can push a borderline partition under the floor in turn — a
    single-pass implementation leaves it there."""
    parts = [f"p{i}" for i in range(9)]
    defs = {p: 0.0 for p in parts}
    defs["p0"] = 100.0
    defs.update({p: 0.4 for p in parts[1:]})
    a = pq.allocate(defs, {p: 1.0 for p in parts}, parts, floor=0.05)
    assert min(a.share.values()) >= 0.05 - 1e-9
    assert sum(a.share.values()) == pytest.approx(1.0)
    assert len(a.floored) == 8 and "p0" not in a.floored


def test_the_floor_claim_at_nine_partitions_is_bounded_by_the_addendums_forty_five_percent():
    """"With 9 partitions this floors UP TO ~45% of total time." Up to: 0.05 x 9 is the
    ceiling, reached only when every partition is floor-pinned, which cannot happen while the
    shares sum to 1. The reachable maximum is 8 pinned = 40%, and that case is constructed
    here rather than described."""
    parts = [f"p{i}" for i in range(9)]
    defs = {p: 1e-6 for p in parts}
    defs["p0"] = 1000.0
    a = pq.allocate(defs, {p: 1.0 for p in parts}, parts, floor=0.05)
    assert len(a.floored) == 8
    assert a.summary()["floor_share_total"] == pytest.approx(0.40)
    assert a.summary()["floor_share_total"] <= 0.05 * 9 + 1e-9
    assert sum(a.share.values()) == pytest.approx(1.0)


def test_a_degenerate_uniform_allocation_is_not_reported_as_floor_driven():
    """With no deficit anywhere the 1/9 share is not something the floor set, and tagging it
    floor-driven would report a 100% floor share on a run where the floor never bound."""
    parts = [f"p{i}" for i in range(9)]
    a = pq.allocate({p: 0.0 for p in parts}, {p: 1.0 for p in parts}, parts, floor=0.05)
    assert a.share["p0"] == pytest.approx(1 / 9) and a.floored == set()
    assert a.summary()["floor_share_total"] == 0.0


def test_an_infeasible_floor_degrades_to_uniform_and_says_so():
    parts = [f"p{i}" for i in range(9)]
    bad = pq.allocate({p: 1.0 for p in parts}, {p: 1.0 for p in parts}, parts, floor=0.2)
    assert bad.floored == set(parts) and sum(bad.share.values()) == pytest.approx(1.0)


def test_a_cheaper_partition_gets_more_of_the_same_deficit():
    """The price weighting, isolated: identical deficits, one partition half the price."""
    a = pq.allocate({p: 10.0 for p in PARTS},
                    {"a": 1.0, "b": 2.0, "c": 2.0, "d": 2.0}, PARTS, floor=0.05)
    assert a.share["a"] > a.share["b"] == pytest.approx(a.share["c"])


# =========================================================================== #
# 3. the pop decision
# =========================================================================== #
def test_pop_is_pure_and_takes_the_largest_gap():
    intended = {"a": 0.5, "b": 0.3, "c": 0.2}
    realized = {"a": 50.0, "b": 10.0, "c": 0.0}     # shares .833 / .167 / 0
    # gaps: a -0.333, b +0.133, c +0.200 -> c, the partition that has had NOTHING.
    assert pq.choose_partition(intended, realized, {"a", "b", "c"}) == "c"
    # ... and once c has caught up past b's gap, b is next. Two calls, so the test asserts
    # the ORDERING the rule produces and not one lucky argmax.
    realized["c"] = 20.0                            # shares .625 / .125 / .25
    assert pq.choose_partition(intended, realized, {"a", "b", "c"}) == "b"


def test_pop_returns_none_when_nothing_is_servable():
    assert pq.choose_partition({"a": 1.0}, {"a": 0.0}, set()) is None
    assert pq.choose_partition({"a": 1.0}, {"a": 0.0}, {"a"}, capped={"a"}) is None


def test_pop_is_deterministic():
    """No RNG: the same inputs give the same answer, which is what makes a realized-mix
    number evidence about the allocator rather than about a seed."""
    args = ({"a": 0.4, "b": 0.4, "c": 0.2}, {"a": 1.0, "b": 1.0, "c": 1.0}, {"a", "b", "c"})
    assert len({pq.choose_partition(*args) for _ in range(50)}) == 1


def test_unservable_julia_intent_folds_into_its_cplane_parent_and_the_original_is_untouched():
    intended = {"multibrot3": 0.2, "julia:multibrot3": 0.3, "phoenix": 0.5}
    eff = pq.fold_julia_intent(intended, {"multibrot3": 5, "julia:multibrot3": 0, "phoenix": 3},
                               list(intended))
    assert eff["multibrot3"] == pytest.approx(0.5) and eff["julia:multibrot3"] == 0.0
    assert intended["julia:multibrot3"] == 0.3, "the reported intent must not be mutated"


def test_a_julia_partition_with_its_own_queue_is_not_folded():
    intended = {"multibrot3": 0.2, "julia:multibrot3": 0.3}
    eff = pq.fold_julia_intent(intended, {"multibrot3": 5, "julia:multibrot3": 7},
                               list(intended))
    assert eff == intended


# =========================================================================== #
# 4. THE PROPERTY. This is the test the v1 mechanism fails.
# =========================================================================== #
def _simulate(intended, growth, n_batches=600, batch_min=1.0):
    """Drive the pure pop decision through a run where `growth[p]` children per served batch
    land in p's own queue — so a partition that multiplies fast keeps a huge queue no matter
    how rarely it is served. Returns the realized minute share."""
    realized = {p: 0.0 for p in intended}
    queue = {p: 10 for p in intended}
    for _ in range(n_batches):
        part = pq.choose_partition(intended, realized, {p for p, n in queue.items() if n > 0})
        if part is None:
            break
        queue[part] = queue[part] - 1 + growth[part]
        realized[part] += batch_min
    tot = sum(realized.values())
    return {p: v / tot for p, v in realized.items()}


def test_the_quota_converges_the_realized_mix_despite_a_runaway_queue():
    """`discovery_pipeline.md` §3.1: the julia hook manufactures z-plane supply from every
    native admission and injected pools out-number native roots, so the fast-multiplying
    partition dominated a globally-popped frontier — 70% intended, 19.6% realized.

    Here `julia` multiplies 8x per service and the natives 1x, and the realized mix still
    lands within 1pp of the intent. THIS is what makes it a quota rather than a steer."""
    intended = {"mb3": 0.30, "mb4": 0.30, "phoenix": 0.10, "julia": 0.30}
    growth = {"mb3": 1, "mb4": 1, "phoenix": 1, "julia": 8}
    got = _simulate(intended, growth)
    for p, want in intended.items():
        assert abs(got[p] - want) < 0.01, (p, got)


def test_the_simulation_would_expose_a_priority_pop(monkeypatch):
    """VACUITY GUARD for the test above: the fixture must be able to fail. A global
    "always pop the biggest queue" rule — the v1 shape — is run through the same simulation
    and must miss the intent badly, or the convergence result proves nothing about the
    mechanism."""
    intended = {"mb3": 0.30, "mb4": 0.30, "phoenix": 0.10, "julia": 0.30}
    growth = {"mb3": 1, "mb4": 1, "phoenix": 1, "julia": 8}
    monkeypatch.setattr(pq, "choose_partition",
                        lambda intended, realized, servable, capped=None:
                        max(sorted(servable), key=lambda p: _QUEUE[p]) if servable else None)
    global _QUEUE
    realized = {p: 0.0 for p in intended}
    _QUEUE = {p: 10 for p in intended}
    for _ in range(600):
        part = max(sorted(_QUEUE), key=lambda p: _QUEUE[p])
        _QUEUE[part] += growth[part] - 1
        realized[part] += 1.0
    tot = sum(realized.values())
    got = {p: v / tot for p, v in realized.items()}
    assert got["julia"] > 0.6, got          # the runaway takes the run
    assert got["mb3"] < 0.2, got


_QUEUE: dict = {}


# =========================================================================== #
# 5. price model
# =========================================================================== #
def test_price_is_minutes_per_currency_unit_and_a_class2_is_not_a_credit():
    c = pq.CostToMine(["a"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9))
    c.charge("a", 10.0)
    c.credit("a", pq.CLASS_WEIGHT.get(2, 0.0))       # a decoded 2: zero units, no credit
    assert c.price("a") == pytest.approx(3.0)
    assert c.min_since_credit["a"] == pytest.approx(10.0), "a non-credit must not reset the clock"
    c.credit("a", pq.CLASS_WEIGHT[4])                # 10 minutes for 1.0 unit
    assert c.price("a") == pytest.approx(10.0)


def test_a_class3_costs_ten_times_a_class4_per_unit():
    c = pq.CostToMine(["a", "b"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9,
                                       price_clamp=1e9))
    c.charge("a", 10.0)
    c.credit("a", pq.CLASS_WEIGHT[4])
    c.charge("b", 10.0)
    c.credit("b", pq.CLASS_WEIGHT[3])
    assert c.price("b") == pytest.approx(10.0 * c.price("a"))


def test_the_price_clamp_bounds_what_a_miscalibrated_head_can_buy():
    c = pq.CostToMine(["a"], dict(seed_price=3.0, price_ema=1.0, price_clamp=4.0,
                                  cap_minutes=1e9))
    c.charge("a", 0.001)
    c.credit("a", 1.0)                                # absurdly cheap raw price
    assert c.raw["a"] < 0.75 and c.price("a") == pytest.approx(3.0 / 4.0)
    assert "a" in c.summary()["clamped"], "a clamped price must be visible, not silent"


def test_a_dry_partition_caps_and_a_credit_reopens_it():
    c = pq.CostToMine(["a"], dict(cap_minutes=5.0))
    assert c.charge("a", 4.0) is False and not c.capped
    assert c.charge("a", 2.0) is True and c.capped == {"a"}
    c.credit("a", 1.0)
    assert not c.capped


# =========================================================================== #
# 6. the object: accounting, buckets, resume
# =========================================================================== #
def _quota(tmp_path, currency, floor=0.05):
    cen = pq.CurrencyCensus(counts={}, currency=currency, defaulted_rows=0,
                            sources={}, partitions=list(currency))
    return pq.PopQuota(list(currency), tmp_path, floor=floor, census=cen,
                       prices_config=dict(cap_minutes=1e9))


def test_charge_tags_the_bucket_that_bought_the_time(tmp_path):
    q = _quota(tmp_path, {"rich": 100.0, "poor": 0.0})
    q.pick({"rich": 5, "poor": 5})
    q.charge("rich", 2.0)                    # rich has zero deficit -> floor-driven
    q.pick({"rich": 5, "poor": 5})
    q.charge("poor", 3.0)                    # poor carries the deficit -> deficit-driven
    fvd = q.floor_vs_deficit()
    assert fvd["per_partition"]["rich"] == {"floor": 2.0, "deficit": 0.0}
    assert fvd["per_partition"]["poor"] == {"floor": 0.0, "deficit": 3.0}
    assert fvd["floor_min"] == 2.0 and fvd["deficit_min"] == 3.0


def test_mix_report_carries_all_three_denominations(tmp_path):
    q = _quota(tmp_path, {"a": 10.0, "b": 0.0})
    q.pick({"a": 1, "b": 1})
    q.charge("a", 1.0)
    q.note_candidates("a", 30)
    q.note_admission("a")
    m = q.mix_report()
    assert set(m) >= {"minutes", "candidates", "admitted", "pops", "l1_gap_minutes"}
    assert m["minutes"]["a"]["realized"] == 1.0
    assert m["candidates"]["a"]["realized"] == 1.0


def test_state_round_trips_and_the_deficit_is_recensused_not_restored(tmp_path):
    q = _quota(tmp_path, {"a": 10.0, "b": 0.0})
    q.pick({"a": 1, "b": 1})
    q.charge("a", 7.5)
    q.credit_decode("a", 4)
    d = json.loads(json.dumps(q.state_dict()))

    q2 = _quota(tmp_path, {"a": 10.0, "b": 4.0})      # the corpus moved between sessions
    q2.load_state(d)
    assert q2.state.realized_min["a"] == 7.5
    assert q2.cost.units["a"] == pytest.approx(1.0)
    assert q2.deficit["b"] == pytest.approx(6.0), "resume must re-census, not restore, demand"


def test_credit_decode_weights_by_class(tmp_path):
    q = _quota(tmp_path, {"a": 1.0})
    assert q.credit_decode("a", 4) == 1.0
    assert q.credit_decode("a", 3) == pytest.approx(0.1)
    assert q.credit_decode("a", 2) == 0.0
    assert q.credit_decode("a", None) == 0.0


def test_log_choice_writes_intent_realized_and_bucket(tmp_path):
    q = _quota(tmp_path, {"a": 10.0, "b": 0.0})
    q.pick({"a": 1, "b": 1})
    q.log_choice(1, "a", {"a": 1, "b": 1})
    rec = json.loads((tmp_path / "quota_trace.jsonl").read_text(encoding="utf-8").strip())
    assert rec["chosen"] == "a" and rec["bucket"] in ("floor", "deficit")
    assert set(rec) >= {"intended", "realized", "deficit", "price", "capped", "queue_lens"}


# =========================================================================== #
# 7. the boundary the whole module inherits
# =========================================================================== #
def test_no_p_good_or_classifier_score_reaches_the_pop_decision():
    """`deficit_scheduler.py`'s HARD SCOPE INVARIANT, re-asserted for the replacement: the
    cross-partition choice is a function of intended shares and realized minutes only. Pinned
    by signature so a later kwarg smuggling a score in is a red test, not a review miss."""
    import inspect
    sig = inspect.signature(pq.choose_partition)
    assert list(sig.parameters) == ["intended", "realized_min", "servable", "capped"]
    # The DOCSTRING names the banned quantities on purpose (it explains why they are absent),
    # so the scan is over the code body only — a guard pinned to prose goes red when the
    # prose is corrected (`verification_practice.md` §6).
    body = inspect.getsource(pq.choose_partition).split('"""')[-1]
    for banned in ("p_good", "eord", "pgood", "score"):
        assert banned not in body, banned


# =========================================================================== #
# 8. the EFFECTIVE intent — the vector the pop actually acts on
# =========================================================================== #
def test_the_effective_intent_is_what_the_pop_acted_on(tmp_path):
    """A julia partition with no queue cannot be popped; §3's routing folds its demand into
    its c-plane parent. So a run that serves the parent is following instructions, and
    scoring it against the STATED vector charges it for obeying them."""
    q = _quota(tmp_path, {"multibrot3": 0.0, "julia:multibrot3": 0.0})
    q.pick({"multibrot3": 5, "julia:multibrot3": 0})       # twin unservable -> folds
    eff = q.effective_intent()
    assert eff["julia:multibrot3"] == 0.0
    assert eff["multibrot3"] == pytest.approx(1.0)


def test_the_effective_intent_is_time_weighted_not_batch_weighted(tmp_path):
    """The effective vector changes with queue occupancy, and an intent that held while an
    expensive batch ran governed more of the run than one that held through a cheap one."""
    q = _quota(tmp_path, {"multibrot3": 0.0, "julia:multibrot3": 0.0})
    q.pick({"multibrot3": 5, "julia:multibrot3": 0})       # folded: mb3 gets it all
    q.charge("multibrot3", 90.0)                            # ... for 90 minutes
    q.pick({"multibrot3": 5, "julia:multibrot3": 5})       # both servable: 50/50
    q.charge("julia:multibrot3", 10.0)                      # ... for 10
    eff = q.effective_intent()
    assert eff["multibrot3"] == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)


def test_the_mix_report_carries_both_gaps_and_they_can_disagree(tmp_path):
    """The proving run's headline correction: L1 0.352 against the stated intent and 0.091
    against the effective one, on the same run. Only the second is a statement about the pop.
    Both are reported, because the first is what a fixed target would have wanted."""
    q = _quota(tmp_path, {"multibrot3": 0.0, "julia:multibrot3": 0.0})
    q.pick({"multibrot3": 5, "julia:multibrot3": 0})
    q.charge("multibrot3", 60.0)
    m = q.mix_report()
    assert m["l1_gap_minutes"] == pytest.approx(0.5)            # vs stated 50/50
    assert m["l1_gap_minutes_effective"] == pytest.approx(0.0)  # vs effective 100/0
    assert m["effective_intent"]["multibrot3"] == pytest.approx(1.0)


def test_the_trace_logs_the_effective_vector_rather_than_leaving_it_derivable(tmp_path):
    """Telemetry, not arithmetic. The first proving run's effective intent had to be
    recomputed offline from `intended` + `queue_lens` by someone who knew the fold rule."""
    q = _quota(tmp_path, {"multibrot3": 0.0, "julia:multibrot3": 0.0})
    q.pick({"multibrot3": 5, "julia:multibrot3": 0})
    q.log_choice(1, "multibrot3", {"multibrot3": 5, "julia:multibrot3": 0})
    rec = json.loads((tmp_path / "quota_trace.jsonl").read_text(encoding="utf-8").strip())
    assert rec["effective"]["multibrot3"] == pytest.approx(1.0)
    assert rec["effective"]["julia:multibrot3"] == 0.0
    assert rec["intended"] != rec["effective"]
