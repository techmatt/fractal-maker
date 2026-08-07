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
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import run_record  # noqa: E402

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


def test_at_equal_ratios_the_target_levels_to_the_richest_and_the_richest_has_zero_deficit():
    """The ratio-weighted rule REDUCES to the uniform one it replaced when every ratio is
    equal — kept as the base case, so a change to the weighting that also broke the levelling
    is two failures rather than one."""
    cur = {"a": 10.0, "b": 4.0, "c": 0.0}
    d = pq.deficits_from_currency(cur, ["a", "b", "c"], {p: 1.0 for p in cur})
    assert d == {"a": 0.0, "b": 6.0, "c": 10.0}


def test_the_target_is_the_anchor_scaled_by_the_ratio_and_the_anchor_is_the_richest_holding():
    """Matt, 2026-08-04. The maximum-ratio partitions keep the target the uniform rule gave
    them (so their deficits do not move at all) and everything below falls proportionally."""
    cur = {"big": 100.0, "small": 5.0, "mid": 20.0}
    ratios = {"big": 3.0, "small": 0.2, "mid": 1.0}
    tgt, anchor = pq.currency_targets(cur, list(cur), ratios)
    assert anchor == 100.0
    assert tgt["big"] == pytest.approx(100.0)                  # ratio 3 == max -> the anchor
    assert tgt["mid"] == pytest.approx(100.0 / 3.0)
    assert tgt["small"] == pytest.approx(100.0 / 15.0)         # 0.2/3
    d = pq.deficits_from_currency(cur, list(cur), ratios)
    assert d["big"] == 0.0
    assert d["small"] == pytest.approx(100.0 / 15.0 - 5.0)
    # ...and a partition ABOVE its (low) target has no deficit, which uniform could not express:
    assert pq.deficits_from_currency({"big": 100.0, "small": 90.0}, ["big", "small"],
                                     {"big": 3.0, "small": 0.2})["small"] == 0.0


def test_a_ratio_change_moves_the_target_vector_with_no_stale_cache(monkeypatch):
    """The table is read at CALL time, not baked at import: `PopQuota` re-allocates every pop,
    and a cached table would keep a running frontier on the mix it was launched with."""
    import release_mix as rm
    from partitions import ALL_FAMS
    cur = {p: 0.0 for p in ALL_FAMS}
    cur[ALL_FAMS[0]] = 100.0
    before = pq.deficits_from_currency(cur, ALL_FAMS)
    monkeypatch.setitem(rm.RATIO, "phoenix:classic", 3.0)
    after = pq.deficits_from_currency(cur, ALL_FAMS)
    assert after["phoenix:classic"] > before["phoenix:classic"]
    assert after["phoenix:classic"] == pytest.approx(100.0)


def test_a_partition_with_no_declared_ratio_RAISES_rather_than_defaulting():
    """The `partitions._registered` failure one layer down: a defaulted ratio would give an
    unregistered partition a plausible target nobody decided, and every downstream read of its
    quiet quota would be a read of the default."""
    with pytest.raises(KeyError, match="release_mix"):
        pq.deficits_from_currency({"mandelbrot": 1.0}, ["mandelbrot", "not_a_partition"])


def test_a_partition_absent_from_the_census_carries_the_full_target_as_its_deficit():
    """Not-yet-mined and mined-nothing are the same demand, and a KeyError here would be the
    silent kind — the partition would simply never be allocated."""
    d = pq.deficits_from_currency({"a": 8.0}, ["a", "new"], {"a": 1.0, "new": 1.0})
    assert d["new"] == 8.0


# A VARIED phoenix render block. Spelled out rather than `dict(fractal_type="phoenix")`
# because a phoenix row's PARTITION is decided by its parameter point since 2026-08-04, so a
# param-free fixture is not a varied-phoenix row — it is the pinned classic point, and the
# census would (correctly) route it to `phoenix:classic`.
VARIED_PH = dict(cx="0.36", cy="-0.41", fw="0.28",
                 c_re="-1.089", c_im="0.481", p_re="-0.222", p_im="0.172",
                 zm1_re="-0.224", zm1_im="-0.347")
# The legacy CLASSIC shape: a phoenix token, a viewport, and not one parameter field. This is
# the on-disk form of all 84 classic rows in the corpus (73 of them labeled).
CLASSIC_PH = dict(cx="0.374", cy="0.378", fw="0.0114")


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
            dict(image_id="i1", render=dict(fractal_type="phoenix", **VARIED_PH),
                 label=dict(score=4)),
            dict(image_id="i2", render=dict(fractal_type="phoenix", **VARIED_PH),
                 label=dict(score=3)),
            dict(image_id="i3", render=dict(fractal_type="julia", cx="0", cy="0", fw="3.0"),
                 label=dict(score=4)),
            dict(image_id="i4", render=dict(cx=0, cy=0), label=dict(score=4)),   # no type
            dict(image_id="i5", render=dict(fractal_type="phoenix", **VARIED_PH),
                 label=dict(score=1)),
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


def test_the_census_splits_classic_phoenix_out_of_varied_phoenix(tmp_path, monkeypatch):
    """The demand signal the whole tenth partition rests on. Two rows sharing one
    `fractal_type` must land in two partitions on their parameter point alone, and the
    currency must SPLIT rather than double-count — pooled, `phoenix` would read 2.0 here and
    `phoenix:classic` would read 0, i.e. exactly the pre-split behaviour."""
    corpus = tmp_path / "corpus"
    b = corpus / "batches" / "2026-08-04_t"
    b.mkdir(parents=True)
    (b / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        dict(image_id="v1", render=dict(fractal_type="phoenix", **VARIED_PH), label=dict(score=4)),
        dict(image_id="c1", render=dict(fractal_type="phoenix", **CLASSIC_PH), label=dict(score=4)),
        dict(image_id="c2", render=dict(fractal_type="phoenix", **CLASSIC_PH,
                                        palette="RdBu"), label=dict(score=3)),
    ]), encoding="utf-8")
    (b / "batch.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("corpus_common.crops_dir", lambda bid: str(b / "crops"), raising=False)

    cen = pq.label_currency(["phoenix", "phoenix:classic"], corpus_dir=str(corpus),
                            library_globs=[])
    assert cen.currency["phoenix"] == pytest.approx(1.0)
    assert cen.currency["phoenix:classic"] == pytest.approx(1.1)
    assert cen.counts["phoenix:classic"] == {4: 1, 3: 1}
    assert cen.defaulted_rows == 0        # every row carried a token; none took the default


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


def test_the_universal_floor_is_feasible_at_the_live_partition_count():
    """The floor is a fraction of TOTAL time and every partition gets one, so `floor * n` is
    its total claim — and `allocate` DEGRADES TO UNIFORM once that reaches 1.0. At the live
    count the floor must still bind something, or every share becomes 1/n and the deficits
    stop steering the run entirely.

    Derived from `ALL_FAMS` rather than pinned to a number: this is the assertion that has to
    fire when the eleventh, not the tenth, partition is registered. At 10 partitions the claim
    is 10 x 5% = 50%, leaving half the budget deficit-driven; the ceiling is 20 partitions."""
    from partitions import ALL_FAMS
    n = len(ALL_FAMS)
    assert pq.FLOOR_FRAC * n < 1.0, (
        f"{n} partitions x {pq.FLOOR_FRAC} floor >= 100% of total time — `allocate` degrades "
        f"to a uniform 1/n split and the deficit stops steering. Lower FLOOR_FRAC (Matt's "
        f"call) or stop adding partitions.")
    # ...and it is still a REAL floor at this count: a zero-deficit partition is floored, not
    # given 1/n. (Non-vacuity — the assertion above passes trivially at small n.)
    defs = {p: (0.0 if p == ALL_FAMS[0] else 100.0) for p in ALL_FAMS}
    a = pq.allocate(defs, {p: 1.0 for p in ALL_FAMS}, ALL_FAMS)
    assert a.floored == {ALL_FAMS[0]}
    assert a.share[ALL_FAMS[0]] == pytest.approx(pq.FLOOR_FRAC)
    assert sum(a.share.values()) == pytest.approx(1.0)


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
# 4b. THE FLOOR CARRY. The property `steady_state_v2_20260807` failed.
#
# Run 2 held julia:mandelbrot's 5% floor for all 361 batches against a queue pinned full at
# 209 nodes and popped it ZERO times (mandelbrot 1 pop / 1.54 min and phoenix 1 pop / 0.37 min
# against a 17.8-minute floor each). The shape below is that run's, reduced to its mechanism:
# a c-plane parent and a julia twin whose queue drains to empty on every service, so
# `fold_julia_intent` keeps swinging the pair's whole intent onto whichever member is
# momentarily unservable while the realized minutes stay SPLIT between the two. One of the
# pair therefore always shows a gap above the floor, and a floored partition's own gap
# saturates at 0.05 no matter how long it starves.
#
# `_run2_sim` is driven twice — carry off is the CONTROL and must starve forever
# (`verification_practice.md` §3: the fixture has to be able to fail), carry on must serve.
# =========================================================================== #
_RUN2_INTENT = {"mandelbrot": 0.05, "julia:mandelbrot": 0.05,
                "multibrot3": 0.30, "julia:multibrot3": 0.60}


def _run2_sim(carry: bool, n_batches=400, cost=None, floor=0.05):
    """The run-2 shape through the SHIPPED rules (`fold_julia_intent`, `choose_partition`,
    `FloorLedger`). `julia:mandelbrot` holds a queue it can never exhaust and is never fed by
    anyone, exactly as in the run. Returns `(pops, minute share, first batch served)`."""
    parts = list(_RUN2_INTENT)
    cost = cost or {p: 1.0 for p in parts}
    queue = {"mandelbrot": 60, "julia:mandelbrot": 209, "multibrot3": 60,
             "julia:multibrot3": 0}
    realized = {p: 0.0 for p in parts}
    pops = {p: 0 for p in parts}
    first: dict = {}
    led = pq.FloorLedger(floor=floor)
    for b in range(n_batches):
        servable = {p for p, k in queue.items() if k > 0}
        eff = pq.fold_julia_intent(_RUN2_INTENT, queue, parts)
        part = pq.choose_partition(
            eff, realized, servable,
            floor_debt=(led.debts(realized) if carry else None),
            debt_trigger=(led.trigger() if carry else 0.0))
        if part is None:
            break
        if part == "julia:multibrot3":
            queue[part] = 0                      # the twin drains on service -> refolds
        else:
            queue[part] -= 1
        if part == "multibrot3":                 # serving the parent manufactures one twin
            queue["julia:multibrot3"] += 1
            queue["multibrot3"] += 1
        if part in ("mandelbrot", "julia:mandelbrot"):
            queue[part] += 1                     # pinned full, as both were for all 361 batches
        realized[part] += cost[part]
        pops[part] += 1
        first.setdefault(part, b)
        led.settle(servable, cost[part])
    tot = sum(realized.values())
    return pops, {p: v / tot for p, v in realized.items()}, first


def test_run2_a_floored_partition_with_a_full_queue_is_NEVER_served_without_the_carry():
    """THE CONTROL, and it is the run-2 defect itself: zero pops, forever, on a partition
    that was servable in every single batch. Without this arm the fixed arm below proves
    nothing about the mechanism — a rule that served the floor anyway would pass it."""
    pops, share, _first = _run2_sim(carry=False, n_batches=400)
    assert pops["julia:mandelbrot"] == 0, pops
    assert pops["mandelbrot"] == 0, pops
    assert share["julia:mandelbrot"] == 0.0
    # ... and it is not a slow start: the competitors have taken the whole run.
    assert share["multibrot3"] + share["julia:multibrot3"] == pytest.approx(1.0)


def test_the_floor_carry_serves_both_floored_partitions_within_a_bounded_number_of_batches():
    """Carry on, same fixture. The bound is exact and cost-free: `debt = floor * T` and
    `trigger = T / pops`, so a partition servable throughout comes due at `pops >= 1/floor`
    = batch 20 at a 5% floor, whatever a batch costs. Both floored partitions must be served
    by then plus one for the second of them — and the realized minute share must land ON the
    floor, not above it."""
    pops, share, first = _run2_sim(carry=True, n_batches=400)
    assert first["julia:mandelbrot"] <= 21, first
    assert first["mandelbrot"] <= 21, first
    assert pops["julia:mandelbrot"] > 0 and pops["mandelbrot"] > 0
    for p in ("julia:mandelbrot", "mandelbrot"):
        assert abs(share[p] - 0.05) < 0.01, (p, share)


def test_the_carry_is_in_MINUTES_so_a_cheap_floored_partition_gets_pops_not_time():
    """"The floor stays a floor" is a claim about the CLOCK, and this is what enforces it.
    A floored partition whose batches cost a fifth of the mean triggers five times as often
    and repays a fifth as much each time, so it takes ~5x the POPS and the SAME minutes."""
    cheap = {p: 1.0 for p in _RUN2_INTENT} | {"julia:mandelbrot": 0.2}
    pops, share, _ = _run2_sim(carry=True, n_batches=3000, cost=cheap)
    assert abs(share["julia:mandelbrot"] - 0.05) < 0.01, share
    pop_share = pops["julia:mandelbrot"] / sum(pops.values())
    assert pop_share > 4 * share["julia:mandelbrot"], (pop_share, share)


def test_an_EXPENSIVE_floored_partition_is_served_regardless_of_its_per_pop_cost():
    """The half of the fix the prompt's leading hypothesis was about: a partition whose batch
    costs 5x the mean cannot be priced out of its floor, because the debt grows without bound
    while a batch's cost does not. It overshoots by at most one batch when it takes its turn."""
    dear = {p: 1.0 for p in _RUN2_INTENT} | {"julia:mandelbrot": 5.0}
    pops, share, first = _run2_sim(carry=True, n_batches=3000, cost=dear)
    # Same `pops >= 1/floor` bound as the flat-cost case, because both sides of the trigger
    # scale with total minutes — a 5x batch does not buy a 5x wait.
    assert pops["julia:mandelbrot"] > 0 and first["julia:mandelbrot"] <= 21, first
    assert abs(share["julia:mandelbrot"] - 0.05) < 0.01, share


def test_with_no_ledger_the_pop_rule_is_the_pre_carry_share_gap_verbatim():
    """The carry is additive: pass no ledger and the shipped rule is the one every other
    test in §3/§4 was written against."""
    intended = {"a": 0.5, "b": 0.3, "c": 0.2}
    realized = {"a": 50.0, "b": 10.0, "c": 0.0}
    assert pq.choose_partition(intended, realized, {"a", "b", "c"}) == "c"
    assert pq.choose_partition(intended, realized, {"a", "b", "c"},
                               floor_debt={}, debt_trigger=0.0) == "c"
    # a trigger of zero must not make every partition "owed" — a zero debt is not a claim.
    assert pq.choose_partition(intended, realized, {"a", "b"},
                               floor_debt={"a": 0.0, "b": 0.0}, debt_trigger=0.0) == "b"


def test_the_carry_preempts_the_gap_only_once_the_debt_has_bought_a_batch():
    cand = ["a", "b"]
    assert pq.floor_carry_pick(cand, {"a": 0.9, "b": 0.0}, 1.0) is None    # not yet due
    assert pq.floor_carry_pick(cand, {"a": 1.0, "b": 0.0}, 1.0) == "a"     # exactly due
    assert pq.floor_carry_pick(cand, {"a": 3.0, "b": 9.0}, 1.0) == "b"     # most owed first
    assert pq.floor_carry_pick(["a"], {"b": 9.0}, 1.0) is None             # not a candidate


def test_entitlement_accrues_only_over_the_minutes_a_partition_COULD_have_been_served():
    """A partition nobody could feed must not bank arrears and spend them in a burst when its
    queue refills. `settle` is handed the set the pop could have chosen from, not everyone."""
    led = pq.FloorLedger(floor=0.05)
    for _ in range(100):
        led.settle({"a"}, 1.0)                   # b unservable for the whole stretch
    assert led.debts({"a": 0.0, "b": 0.0}) == {"a": pytest.approx(5.0)}
    led.settle({"a", "b"}, 1.0)
    assert led.debts({"a": 0.0, "b": 0.0})["b"] == pytest.approx(0.05)


def test_an_externally_supplied_partition_banks_no_floor_claim():
    """SKIP SITE 2's consequence: a partition the crawl cannot serve is allocated 0.0 on
    purpose, so it must not accrue a claim it would later preempt with."""
    led = pq.FloorLedger(floor=0.05, external={"ext"})
    for _ in range(100):
        led.settle({"a", "ext"}, 1.0)
    assert "ext" not in led.debts({"a": 0.0, "ext": 0.0})
    rep = led.unspent_floor({"a": 5.0, "ext": 0.0}, ["a", "ext"])
    assert rep["alarms"] == [] and "ext" not in rep["per_partition"]


# --------------------------------------------------------------------------- #
# The OBSERVABILITY half — independent of the fix, and it must fire on run 2's numbers.
# --------------------------------------------------------------------------- #
def test_the_unspent_floor_alarm_fires_on_the_injected_run2_case():
    """Run 2's own figures: 356.7 charged minutes, a 5% floor = 17.8 allocated minutes each,
    against which julia:mandelbrot spent 0.00, mandelbrot 1.54 and phoenix 0.37. All three
    must be named; multibrot4, which spent 112.15, must not be."""
    led = pq.FloorLedger(floor=0.05)
    parts = ["julia:mandelbrot", "mandelbrot", "phoenix", "multibrot4"]
    for _ in range(361):
        led.settle(set(parts), 356.7 / 361)
    realized = {"julia:mandelbrot": 0.0, "mandelbrot": 1.54, "phoenix": 0.37,
                "multibrot4": 112.15}
    rep = led.unspent_floor(realized, parts)
    assert rep["alarms"] == ["julia:mandelbrot", "mandelbrot", "phoenix"], rep["alarms"]
    assert rep["allocated_min_per_partition"] == pytest.approx(17.835, abs=1e-3)
    d = rep["per_partition"]["julia:mandelbrot"]
    assert d["unspent_frac"] == pytest.approx(1.0)
    # SERVABLE the whole run: that is what makes it the pop rule's failure and not the
    # frontier's, and it is why the number is reported beside the spend.
    assert d["servable_frac"] == pytest.approx(1.0) and d["never_servable"] is False
    assert rep["per_partition"]["mandelbrot"]["unspent_frac"] == pytest.approx(0.9137, abs=1e-3)


def test_the_unspent_floor_alarm_is_silent_when_the_floor_was_actually_spent():
    """VACUITY GUARD: an alarm that cannot stay quiet is not an alarm."""
    led = pq.FloorLedger(floor=0.05)
    for _ in range(100):
        led.settle({"a", "b"}, 1.0)
    assert led.unspent_floor({"a": 5.0, "b": 95.0}, ["a", "b"])["alarms"] == []


def test_the_alarm_separates_a_starved_partition_from_an_unfeedable_one():
    """Both spend zero. `servable_min` is the only thing that tells them apart, so it rides
    in the same row rather than in a different file."""
    led = pq.FloorLedger(floor=0.05)
    for _ in range(100):
        led.settle({"served", "starved"}, 1.0)      # `dead` never servable
    rep = led.unspent_floor({"served": 100.0, "starved": 0.0, "dead": 0.0},
                            ["served", "starved", "dead"])
    assert rep["alarms"] == ["dead", "starved"]
    assert rep["per_partition"]["starved"]["never_servable"] is False
    assert rep["per_partition"]["dead"]["never_servable"] is True


def test_the_quota_reports_the_alarm_and_the_trace_says_which_rule_popped(tmp_path):
    """End to end through `PopQuota`: the ledger accrues off `pick`/`charge`, the alarm lands
    in `summary()`, and the trace stamps `via` so a reader can tell a floor that is being HELD
    from one that is merely never tested."""
    q = _quota(tmp_path, {"mandelbrot": 100.0, "julia:mandelbrot": 100.0,
                          "multibrot3": 40.0, "julia:multibrot3": 0.0}, floor=0.05)
    queue = {"mandelbrot": 60, "julia:mandelbrot": 209, "multibrot3": 60,
             "julia:multibrot3": 0}
    for i in range(200):
        part = q.pick(dict(queue))
        q.log_choice(i, part, dict(queue))
        q.charge(part, 1.0)
        queue["julia:multibrot3"] = 0 if part == "julia:multibrot3" else \
            queue["julia:multibrot3"] + (1 if part == "multibrot3" else 0)
    rows = run_record.read_rows(tmp_path / "quota_trace.jsonl")
    assert {r["via"] for r in rows} == {"gap", "floor_carry"}
    carried = [r for r in rows if r["via"] == "floor_carry"]
    assert all(r["floor_debt"][r["chosen"]] >= r["floor_trigger_min"] for r in carried)
    assert {r["chosen"] for r in carried} == {"mandelbrot", "julia:mandelbrot"}
    s = q.summary()
    assert s["unspent_floor"]["alarms"] == []          # the carry held both floors
    assert q.state.pops["julia:mandelbrot"] > 0 and q.state.pops["mandelbrot"] > 0


def test_the_floor_ledger_survives_a_resume(tmp_path):
    """A resumed run that reset the ledger would re-offer the floor from scratch every
    session — the same defect on a longer period."""
    q = _quota(tmp_path, {"rich": 100.0, "poor": 0.0})
    for _ in range(10):
        q.charge(q.pick({"rich": 5, "poor": 5}), 1.0)
    st = json.loads(json.dumps(q.state_dict()))
    q2 = _quota(tmp_path, {"rich": 100.0, "poor": 0.0})
    q2.load_state(st)
    assert q2.floor_ledger.total_min == pytest.approx(q.floor_ledger.total_min)
    assert q2.floor_ledger.pops == q.floor_ledger.pops
    assert q2.floor_ledger.debts(q2.state.realized_min) == \
        pytest.approx(q.floor_ledger.debts(q.state.realized_min))


# =========================================================================== #
# 5. price model
# =========================================================================== #
def test_price_is_minutes_per_currency_unit_and_a_class2_is_not_a_credit():
    c = pq.CostToMine(["a"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9))
    c.charge("a", 10.0)
    c.credit("a", pq.CLASS_WEIGHT.get(2, 0.0))       # a decoded 2: zero units, no credit
    c.end_window()
    assert c.price("a") == pytest.approx(3.0)
    assert c.min_since_credit["a"] == pytest.approx(10.0), "a non-credit must not reset the clock"
    c.credit("a", pq.CLASS_WEIGHT[4])                # 10 minutes for 1.0 unit
    c.end_window()
    assert c.price("a") == pytest.approx(10.0)


def test_a_class3_costs_ten_times_a_class4_per_unit():
    c = pq.CostToMine(["a", "b"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9,
                                       price_clamp=1e9))
    c.charge("a", 10.0)
    c.credit("a", pq.CLASS_WEIGHT[4])
    c.charge("b", 10.0)
    c.credit("b", pq.CLASS_WEIGHT[3])
    c.end_window()
    assert c.price("b") == pytest.approx(10.0 * c.price("a"))


def test_the_price_clamp_bounds_what_a_miscalibrated_head_can_buy():
    c = pq.CostToMine(["a"], dict(seed_price=3.0, price_ema=1.0, price_clamp=4.0,
                                  cap_minutes=1e9))
    c.charge("a", 0.001)
    c.credit("a", 1.0)                                # absurdly cheap raw price
    c.end_window()
    assert c.raw["a"] < 0.75 and c.price("a") == pytest.approx(3.0 / 4.0)
    assert "a" in c.summary()["clamped"], "a clamped price must be visible, not silent"


# --------------------------------------------------------------------------- #
# 5b. batch-aggregated sampling — the fix, and the defect it replaced.
#
# `_v1_price` below is the SUPERSEDED sampler, kept as an executable reference rather than as
# prose: the bracket the fix needs is "old was wrong AND new is right AND new does not
# over-correct" (`verification_practice.md` §3), and the first half of that is unassertable
# once the old code is deleted. It is eight lines and it is the exact arithmetic arm B ran.
# --------------------------------------------------------------------------- #
def _v1_price(stream, seed=3.0, ema=0.30):
    """v1: one EMA sample per DECODE = (minutes since last credit) / (units of THAT decode).

    `stream` is a list of batches, each `(minutes, [units, ...])`."""
    raw, since = seed, 0.0
    for minutes, decodes in stream:
        # v1 charged the batch AFTER harvesting it, exactly as the driver does.
        for u in decodes:
            sample = since / u
            if sample > 0:
                raw = (1 - ema) * raw + ema * sample
            since = 0.0
        since += minutes
    return raw


def _v2_price(stream, seed=3.0, ema=0.30):
    """The live sampler, driven through the same stream."""
    c = pq.CostToMine(["p"], dict(seed_price=seed, price_ema=ema, cap_minutes=1e9,
                                  price_clamp=1e9))
    for minutes, decodes in stream:
        for u in decodes:
            c.credit("p", u)
        c.charge("p", minutes)
        c.end_window()
    return c.raw["p"]


def _truth(stream):
    m = sum(minutes for minutes, _ in stream)
    u = sum(sum(d) for _, d in stream)
    return m / u


# Two streams of IDENTICAL true cost (720 active minutes buying 60 units = 12.0 min/unit),
# differing ONLY in whether the decodes arrive clustered in one batch or one per batch.
CLUSTERED = ([(20.0, []), (20.0, []), (20.0, []), (20.0, []), (20.0, []),
              (20.0, [1.0] * 10)] * 6)
DRIP = [(12.0, [1.0])] * 60


def test_v1_priced_two_identical_cost_streams_an_order_of_magnitude_apart():
    """OLD BEHAVIOUR WAS WRONG — the bracket's first half, asserted rather than described.

    v1 divides a batch's whole accumulated gap by the FIRST decode's units; the other nine
    decodes in the burst find the counter already reset, sample 0.0, and are dropped by the
    `sample > 0` guard. Their units never reach any denominator, so the sample overstates the
    true rate by the burst's unit ratio — here 10x. The drip stream has one decode per batch,
    no burst, and v1 gets it right.

    So v1's price ordering across partitions carries their BURSTINESS, not their cost. Since
    allocation share is deficit/price, that ordering is what routed arm B's intent."""
    assert _truth(CLUSTERED) == pytest.approx(12.0)
    assert _truth(DRIP) == pytest.approx(12.0)
    v1_cl, v1_dr = _v1_price(CLUSTERED), _v1_price(DRIP)
    assert v1_cl > 8.0 * 12.0, f"clustered read {v1_cl:.2f} against a truth of 12.0"
    assert v1_dr == pytest.approx(12.0, rel=0.05), f"drip read {v1_dr:.2f}"
    assert v1_cl > 8.0 * v1_dr, "the same cost, priced 8x apart, on clustering alone"


def test_batch_aggregation_prices_both_streams_at_the_true_per_unit_cost():
    """NEW BEHAVIOUR IS RIGHT, AND DOES NOT OVER-CORRECT — the bracket's other two halves.

    Both streams land within 15% of 12.0 and within 15% of each other. The drip stream is the
    over-correction check: v1 already priced it correctly, so a fix that moved it would be
    trading one bias for another."""
    got_cl, got_dr = _v2_price(CLUSTERED), _v2_price(DRIP)
    for name, got in (("clustered", got_cl), ("drip", got_dr)):
        assert 0.85 * 12.0 <= got <= 1.15 * 12.0, f"{name} read {got:.2f}, truth 12.0"
    assert abs(got_cl - got_dr) / max(got_cl, got_dr) < 0.15
    assert abs(got_cl - 12.0) < abs(_v1_price(CLUSTERED) - 12.0)


def test_every_sample_the_window_emits_is_the_windows_own_aggregate_rate():
    """The property underneath both tests above, stated exactly rather than through an EMA:
    each emitted sample IS window-minutes / window-units. The EMA is then a smoother over
    unbiased samples instead of over first-decode ratios."""
    c = pq.CostToMine(["p"], dict(seed_price=3.0, price_ema=0.3, cap_minutes=1e9,
                                  price_clamp=1e9))
    samples = []
    for minutes, decodes in CLUSTERED:
        for u in decodes:
            c.credit("p", u)
        c.charge("p", minutes)
        samples.append(c.end_window().get("p"))
    emitted = [s for s in samples if s is not None]
    assert len(emitted) == 6, "one sample per batch that carried units, and no others"
    assert emitted == pytest.approx([12.0] * 6)


def test_a_repeatedly_charged_dry_partition_samples_its_whole_gap_at_the_next_credit():
    """The sparse-partition case the seed-pinning bias hid. Twenty dry batches then one
    class-3: the sample is all twenty batches' minutes against 0.1 units, not the last
    batch's."""
    c = pq.CostToMine(["p"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9,
                                  price_clamp=1e9))
    for _ in range(20):
        c.charge("p", 1.0)
        c.end_window()                       # no units -> no sample, minutes carry
    assert c.raw["p"] == pytest.approx(3.0), "a dry window must not emit a sample"
    c.credit("p", pq.CLASS_WEIGHT[3])
    c.charge("p", 1.0)
    c.end_window()
    assert c.raw["p"] == pytest.approx(21.0 / 0.1)


def test_units_credited_with_no_charged_minutes_are_never_priced_at_zero():
    """A window with units but no minutes must NOT flush: a zero sample prices a partition as
    free, which is the one direction the allocator amplifies (share = deficit/price)."""
    c = pq.CostToMine(["p"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9,
                                  price_clamp=1e9))
    c.credit("p", 1.0)
    assert c.end_window() == {}
    assert c.raw["p"] == pytest.approx(3.0)
    assert c.win_units["p"] == pytest.approx(1.0), "the units are held, not discarded"
    c.charge("p", 4.0)                        # the minutes those units actually cost
    assert c.end_window() == pytest.approx({"p": 4.0})
    assert c.raw["p"] == pytest.approx(4.0)


def test_the_window_survives_a_resume():
    """`state_dict`/`load_state` carry the open window. A resume that dropped it would throw
    away the dry minutes a sparse partition had accumulated — pricing it cheap exactly at the
    moment it is most expensive."""
    a = pq.CostToMine(["p"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9,
                                  price_clamp=1e9))
    for _ in range(5):
        a.charge("p", 2.0)
        a.end_window()
    b = pq.CostToMine(["p"], dict(seed_price=3.0, price_ema=1.0, cap_minutes=1e9,
                                  price_clamp=1e9))
    b.load_state(json.loads(json.dumps(a.state_dict())))
    assert b.win_min["p"] == pytest.approx(10.0)
    b.credit("p", 1.0)
    b.charge("p", 2.0)
    b.end_window()
    assert b.raw["p"] == pytest.approx(12.0)


def test_the_quota_closes_the_price_window_once_per_charged_batch(tmp_path):
    """The seam, not the arithmetic: `PopQuota.charge` IS the window boundary, so a driver
    that charges cannot forget to price. Two batches, one credit each -> two samples."""
    q = _quota(tmp_path, {"a": 0.0, "b": 10.0})
    q.pick({"a": 5, "b": 5})
    q.credit_decode("a", 4)
    q.charge("a", 6.0)
    assert q.cost.samples["a"] == 1 and q.cost.win_units["a"] == 0.0
    q.credit_decode("a", 4)
    q.charge("a", 6.0)
    assert q.cost.samples["a"] == 2
    assert q.cost.summary()["price_aggregate"]["a"] == pytest.approx(6.0)


def test_a_dry_partition_caps_and_a_credit_reopens_it():
    c = pq.CostToMine(["a"], dict(cap_minutes=5.0))
    assert c.charge("a", 4.0) is False and not c.capped
    assert c.charge("a", 2.0) is True and c.capped == {"a"}
    c.credit("a", 1.0)
    assert not c.capped


# =========================================================================== #
# 6. the object: accounting, buckets, resume
# =========================================================================== #
def _quota(tmp_path, currency, floor=0.05, ratios=None):
    """A quota over SYNTHETIC partitions, at EQUAL ratios unless a test says otherwise.

    Equal ratios reproduce the pre-2026-08-04 uniform target exactly, so every test below is
    still about the thing it was written for (buckets, resume, the price window) and not about
    the mix policy. The real table is exercised where it belongs — §1's target tests and
    `tools/scoring/test_release_mix.py` — over the real partition names; a synthetic partition
    has no ratio and `currency_targets` refuses to invent one, which is the point."""
    cen = pq.CurrencyCensus(counts={}, currency=currency, defaulted_rows=0,
                            sources={}, partitions=list(currency))
    return pq.PopQuota(list(currency), tmp_path, floor=floor, census=cen,
                       prices_config=dict(cap_minutes=1e9),
                       ratios=ratios or {p: 1.0 for p in currency})


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
    rec = run_record.read_rows(tmp_path / "quota_trace.jsonl")[0]
    assert rec["chosen"] == "a" and rec["bucket"] in ("floor", "deficit")
    assert set(rec) >= {"intended", "realized", "deficit", "price", "capped", "queue_lens"}


# =========================================================================== #
# 7. the boundary the whole module inherits
# =========================================================================== #
def test_no_p_good_or_classifier_score_reaches_the_pop_decision():
    """`deficit_scheduler.py`'s HARD SCOPE INVARIANT, re-asserted for the replacement: the
    cross-partition choice is a function of intended shares, realized minutes and the floor
    ledger only. Pinned by signature so a later kwarg smuggling a score in is a red test, not
    a review miss. `floor_debt`/`debt_trigger` (2026-08-07) are MINUTES out of `FloorLedger`
    and widen the pin deliberately — the invariant is that nothing per-node or per-image
    arrives here, not that the signature never changes."""
    import inspect
    sig = inspect.signature(pq.choose_partition)
    assert list(sig.parameters) == ["intended", "realized_min", "servable", "capped",
                                    "floor_debt", "debt_trigger"]
    assert list(inspect.signature(pq.floor_carry_pick).parameters) == \
        ["cand", "floor_debt", "debt_trigger"]
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
    rec = run_record.read_rows(tmp_path / "quota_trace.jsonl")[0]
    assert rec["effective"]["multibrot3"] == pytest.approx(1.0)
    assert rec["effective"]["julia:multibrot3"] == 0.0
    assert rec["intended"] != rec["effective"]


# =========================================================================== #
# The default-fractal_type routing, over the REAL corpus.
#
# mandelbrot's cost-to-mine rests on rows that were routed to it by a default rather than by
# a token, and mandelbrot is the partition that SETS the uniform target level — so a row that
# defaults there without earning it moves every other partition's deficit too.
# =========================================================================== #
def _token_less_rows():
    """Every render block in the corpus with no family token — labeled or not.

    Deliberately NOT `iter_labeled`: the census only sees SCORED rows, and that is exactly
    what made the population look like six batches when eleven carry token-less renders. The
    five unlabeled ones are a future census's rows, and the invariant has to hold for them
    before they are labeled, not after."""
    import json
    base = ROOT / "data" / "label_corpus" / "batches"
    for bd in sorted(base.iterdir()):
        f = bd / "images.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = (json.loads(line).get("render") or {})
            if not (r.get("fractal_type") or r.get("family")):
                yield bd.name, r


def test_every_defaulted_row_is_mandelbrot_SHAPED():
    """The two reasons that survive as testable INVARIANTS, over the whole corpus.

    The third reason in `_partition_of_render` — "no token dates the row to before any other
    family existed" — is a fact about six batches created 2026-06-23..25 and it does NOT
    generalize: five later batches carry token-less renders too. So the dating argument is
    not what is asserted here. These two are:

      (a) a parameter-plane family writes `c_re`/`c_im`; a token-less row that carries them
          is not a degree-2 c-plane row and must not default to mandelbrot;
      (b) `flat_generate`'s sampling box IS the mandelbrot c-plane rectangle, so a center
          outside it did not come from that sampler.

    The COUNT is not pinned on purpose: it moves the day the revisit batches are labeled,
    and that is a labeling event, not a regression."""
    import math
    bad_c, bad_box, n = [], [], 0
    per_batch = {}
    for batch, r in _token_less_rows():
        n += 1
        per_batch[batch] = per_batch.get(batch, 0) + 1
        if r.get("c_re") is not None or r.get("c_im") is not None:
            bad_c.append((batch, r.get("c_re"), r.get("c_im")))
        cx, cy = float(r["cx"]), float(r["cy"])
        if not (-2.05 <= cx <= 0.75 and abs(cy) <= 1.25):
            bad_box.append((batch, cx, cy))
    assert n > 4000, f"only {n} token-less rows found — is the corpus reachable?"
    assert not bad_c, (
        "token-less render blocks carrying c_re/c_im would default to mandelbrot but are "
        f"parameter-plane rows: {bad_c[:5]}")
    assert not bad_box, (
        "token-less render blocks centered outside the mandelbrot c-plane box "
        f"(re in [-2.05, 0.75], |im| <= 1.25): {bad_box[:5]}")


def test_no_defaulted_row_carries_a_DYNAMICAL_writer_stamp():
    """The WRITER-side half of the invariant (added 2026-08-05).

    The shape test above is reader-side and cannot exclude a julia z-plane row: those
    viewports centre near the origin, so they sit inside the mandelbrot box, and for a
    PRE-extension row the schema had no `c_re`/`c_im` field for them to be missing from.
    The walker's own stamp does exclude them — `guided_descend.rs` sets `root_src` to
    "julia"/"phoenix" in its dynamical modes and "8k"/"flat" on the c-plane — so that is
    what is asserted. `None` is allowed (writers that stamp no root at all: flat_generate,
    mining_v3guided, the post-extension anchor/revisit batches), because a null is the
    absence of a claim, not a dynamical one."""
    C_PLANE = {"8k", "flat", "injected", None}
    bad, n = [], 0
    for bd in sorted((ROOT / "data" / "label_corpus" / "batches").iterdir()):
        f = bd / "images.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rend = row.get("render") or {}
            if rend.get("fractal_type") or rend.get("family"):
                continue
            n += 1
            rs = (row.get("provenance") or {}).get("root_src")
            if rs not in C_PLANE:
                bad.append((bd.name, row.get("image_id"), rs))
    assert n > 4000, f"only {n} token-less rows found — is the corpus reachable?"
    assert not bad, (
        "token-less render blocks stamped with a DYNAMICAL root by the walker would "
        f"default to mandelbrot but are julia/phoenix z-plane rows: {bad[:5]}")


def test_the_labeled_defaulted_population_still_carries_mandelbrots_currency():
    """The number the price rests on, re-derived rather than quoted. Asserted as a SHARE with
    room to move — labeling adds rows — because pinning 159.2 would make every labeling
    session a red suite, and the decision this supports is only "does the default carry
    enough currency to matter", which a band answers."""
    import corpus_reader as cr
    cur_def = cur_all = 0.0
    for lc in cr.iter_labeled(None):
        r = lc.render or {}
        w = pq.CLASS_WEIGHT.get(int(lc.score), 0.0)   # only 3s and 4s are currency
        if not (r.get("fractal_type") or r.get("family")):
            cur_def += w
        if pq._partition_of_render(r) == "mandelbrot":
            cur_all += w
    assert cur_all > 0
    share = cur_def / cur_all
    assert 0.5 < share < 1.0, (
        f"defaulted rows carry {cur_def:.1f} of mandelbrot's {cur_all:.1f} currency "
        f"({share:.1%}). Outside the band this default stopped being the thing that "
        f"determines mandelbrot's price — re-read pop_quota._partition_of_render.")


# =========================================================================== #
# 8. the RUN-SCOPED currency-target override (--currency-targets, 2026-08-07)
# =========================================================================== #
def _targets_file(tmp_path, targets, **extra):
    p = tmp_path / "targets.json"
    p.write_text(json.dumps(dict(targets=targets, **extra)), encoding="utf-8")
    return p


def test_an_explicit_target_vector_is_read_verbatim_and_the_file_comes_back_whole(tmp_path):
    """The resolved vector is the file's numbers, unscaled and unanchored — and the raw file
    rides along so `run_config.json` can record what was passed rather than what was parsed."""
    f = _targets_file(tmp_path, {p: 10.0 * (i + 1) for i, p in enumerate(PARTS)},
                      ratios={p: 1.0 for p in PARTS}, note="hi")
    tgt, raw = pq.load_currency_targets(f, PARTS)
    assert tgt == {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0}
    assert raw["note"] == "hi" and raw["ratios"] == {p: 1.0 for p in PARTS}


def test_an_explicit_target_defeats_the_anchor_that_zeroes_the_RICHEST_partition(tmp_path):
    """THE REASON THE FLAG EXISTS, as a property rather than as prose.

    Under the derived rule the census-maximum partition lands at exactly zero deficit whenever
    it also carries the maximum ratio, and NO reweighting of the ratio table moves it — raising
    its ratio cannot raise it above the anchor it itself defines. That is run 2's
    `julia:mandelbrot` (190.6 currency, ratio 3.0, zero pops in 361 batches). The override is
    the only thing that can give that partition demand, so this asserts both halves."""
    have = {"a": 100.0, "b": 40.0, "c": 40.0, "d": 40.0}          # 'a' is the census maximum
    rich = dict(a=3.0, b=1.0, c=1.0, d=1.0)
    assert pq.deficits_from_currency(have, PARTS, rich)["a"] == 0.0
    # ...and cranking its ratio to the sky does not help, because it IS the anchor.
    assert pq.deficits_from_currency(have, PARTS, dict(a=99.0, b=1.0, c=1.0, d=1.0))["a"] == 0.0
    f = _targets_file(tmp_path, {"a": 150.0, "b": 40.0, "c": 40.0, "d": 40.0})
    tgt, _ = pq.load_currency_targets(f, PARTS)
    assert tgt["a"] - have["a"] == 50.0


def test_a_missing_or_stray_partition_RAISES_in_both_directions(tmp_path):
    """`release_mix.check_complete`'s two failures, one layer down. An absent target reads
    downstream as a measured zero demand; a target for an untracked partition reads as applied
    while reaching no allocation."""
    with pytest.raises(KeyError, match="tracked with no target"):
        pq.load_currency_targets(_targets_file(tmp_path, {"a": 1.0, "b": 1.0, "c": 1.0}), PARTS)
    with pytest.raises(KeyError, match="untracked partition"):
        pq.load_currency_targets(
            _targets_file(tmp_path, {p: 1.0 for p in PARTS} | {"zz": 1.0}), PARTS)


def test_a_file_with_no_targets_object_RAISES_rather_than_falling_back(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"ratios": {"a": 1.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty top-level `targets`"):
        pq.load_currency_targets(p, PARTS)


def test_the_two_target_paths_are_MUTUALLY_EXCLUSIVE_not_blended(tmp_path, monkeypatch):
    """Passing both an explicit vector and a ratio table is a third rule nobody wrote down."""
    cen = pq.CurrencyCensus(counts={}, currency={p: 1.0 for p in PARTS}, defaulted_rows=0,
                            sources={}, partitions=PARTS)
    with pytest.raises(ValueError, match="does not reweight"):
        pq.PopQuota(PARTS, tmp_path, census=cen, external=set(),
                    targets={p: 5.0 for p in PARTS}, ratios={p: 1.0 for p in PARTS})


def test_an_overridden_quota_reports_NO_anchor_and_names_the_override_rule(tmp_path):
    """The anchor is a fact about the DERIVED rule. Reporting max(target) as one would invent a
    derivation the run never performed, so it is None and every reader formats it."""
    cen = pq.CurrencyCensus(counts={}, currency={p: 1.0 for p in PARTS}, defaulted_rows=0,
                            sources={}, partitions=PARTS)
    src = {"targets": {p: 5.0 for p in PARTS}, "ratios": {p: 2.0 for p in PARTS}}
    q = pq.PopQuota(PARTS, tmp_path, census=cen, external=set(),
                    targets={p: 5.0 for p in PARTS}, targets_source=src)
    assert q.anchor is None
    assert q.target_rule == pq.TARGET_RULE_OVERRIDE and q.target_rule != pq.TARGET_RULE
    assert q.deficit == {p: 4.0 for p in PARTS}
    s = q.summary()
    assert s["anchor"] is None and s["target_rule"] == pq.TARGET_RULE_OVERRIDE
    assert s["currency_targets_file"] == src
    # declared ratios are PROVENANCE — carried, never multiplied by
    assert q.ratios == {p: 2.0 for p in PARTS}


def test_the_derived_path_is_untouched_when_no_override_is_passed(tmp_path):
    cen = pq.CurrencyCensus(counts={}, currency={"a": 9.0, "b": 3.0, "c": 3.0, "d": 3.0},
                            defaulted_rows=0, sources={}, partitions=PARTS)
    q = pq.PopQuota(PARTS, tmp_path, census=cen, external=set(),
                    ratios={p: 1.0 for p in PARTS})
    assert q.anchor == 9.0 and q.target_rule == pq.TARGET_RULE
    assert q.summary()["currency_targets_file"] is None


def test_the_shipped_label_run_targets_file_resolves_against_the_live_partitions():
    """The instrument this run actually launches with. It is a committed artifact, so a
    partition registered or retired after it was written must go red HERE rather than at
    launch — and the canonical ratio table must NOT have been edited to produce it."""
    import release_mix                                            # noqa: E402
    f = ROOT / "data" / "atlas" / "currency_targets_label_run_20260807.json"
    parts = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5", "julia:mandelbrot",
             "julia:multibrot3", "julia:multibrot4", "julia:multibrot5", "phoenix"]
    tgt, raw = pq.load_currency_targets(f, parts)
    assert set(tgt) == set(parts) and all(v > 0 for v in tgt.values())
    # the scale rule, re-derived from the file's own recorded inputs
    k = raw["_provenance"]["scale_k"]
    for p in parts:
        assert abs(tgt[p] - raw["ratios"][p] * k) < 1e-3
    assert abs(sum(tgt.values()) - raw["_provenance"]["derived_total_target"]) < 1e-2
    # the canonical policy table is NOT what this file says
    assert release_mix.RATIO["mandelbrot"] == 3.0 and release_mix.RATIO["phoenix"] == 1.0
    assert raw["ratios"]["mandelbrot"] == 30.0 and raw["ratios"]["phoenix"] == 20.0
