#!/usr/bin/env python
"""Tests for the v1.5 VIEW screen: the run-local field cache, the composite selection, and
the seam where the walk reads them.

WHAT THESE TESTS ARE FOR, and what they deliberately are not. The measures themselves
(`band_coverage`, the composite, the veto, the size band) are tested in
`test_view_screen.py` against synthetic and reference fields; nothing here re-tests them.
What is new in v1.5 is a *plumbing* claim — that the run screens the frame it pushes, files
the result under an identity three separate structures agree on, survives a kill mid-write,
and never lets a caching failure cost a candidate. Those are the claims here.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "orbital")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import field_metrics as fm                 # noqa: E402
import maneuver_view_screen as mvs         # noqa: E402
import view_field_cache as vfc             # noqa: E402
import view_screen as vs                   # noqa: E402

H, W = fm.SCREEN_H * fm.SCREEN_SS, fm.SCREEN_W * fm.SCREEN_SS


def _field(seed=0):
    return (np.arange(H * W, dtype=np.float32).reshape(H, W) + float(seed)) * 0.01


def _params():
    return vs.ScreenParams(veto=0.3403, cap_range=21.3236, cap_rings=66.0)


# =========================================================================== #
# the run-local field cache
# =========================================================================== #
def test_run_cache_round_trips_and_refuses_to_overwrite_a_key(tmp_path):
    c = vfc.RunFieldCache(tmp_path / "vf", policy="tok")
    assert c.put("a|16.0", _field(1), cx="1", cy="2", fw=1e-3) is True
    # A second put under the same key is a NO-OP that says so, not an append and not a
    # silent overwrite: a view is screened once and re-screening it would mean the two
    # measurements disagreed, which the caller has to be able to see.
    assert c.put("a|16.0", _field(99)) is False
    assert c.put("b|None", _field(2)) is True
    assert c.n == 2
    assert np.array_equal(c.get("a|16.0"), _field(1))
    assert np.array_equal(c.get("b|None"), _field(2))
    assert c.get("never-stored") is None
    c.close()


def test_run_cache_reopens_and_finalizes_into_the_retrospective_format(tmp_path):
    root = tmp_path / "vf"
    c = vfc.RunFieldCache(root, policy="tok")
    c.put("a|16.0", _field(1))
    c.put("b|8.0", _field(2))
    c.close()

    reopened = vfc.RunFieldCache(root)
    assert reopened.n == 2 and np.array_equal(reopened.get("b|8.0"), _field(2))
    rep = reopened.finalize()
    reopened.close()
    assert rep["n"] == 2

    # ONE FORMAT: the finalized store opens as the retrospective FieldCache, which is what
    # the post-label numpy pass uses. If these ever diverge the run's fields become a
    # private format nothing else can read, which is the whole failure the shared layout
    # exists to prevent.
    static = vfc.FieldCache(root)
    assert static.keys == ["a|16.0", "b|8.0"] and static.n_valid == 2
    assert np.array_equal(np.asarray(static.get("a|16.0")), _field(1))


def test_run_cache_drops_an_index_line_whose_field_is_not_on_disk(tmp_path):
    """A kill between the index append and the field flush cannot invent a row."""
    root = tmp_path / "vf"
    c = vfc.RunFieldCache(root, policy="tok")
    c.put("a|16.0", _field(1))
    c.put("b|8.0", _field(2))
    c.close()
    # Truncate the store to ONE record while the index still names two.
    with open(root / vfc.FIELDS_NAME, "r+b") as f:
        f.truncate(H * W * 4)

    reopened = vfc.RunFieldCache(root)
    assert reopened.n == 1 and reopened.has("a|16.0") and not reopened.has("b|8.0")
    # ...and the recovered cache is writable again at the SAME index, so the next screen
    # does not leave a permanent hole.
    assert reopened.put("b|8.0", _field(3)) is True
    assert np.array_equal(reopened.get("b|8.0"), _field(3))
    reopened.close()


def test_run_cache_overwrites_an_orphan_field_no_index_line_claims(tmp_path):
    """The other half of the same kill: the field landed, the index line did not."""
    root = tmp_path / "vf"
    c = vfc.RunFieldCache(root, policy="tok")
    c.put("a|16.0", _field(1))
    c.close()
    # Simulate the orphan: append a second record's bytes without its index line.
    with open(root / vfc.FIELDS_NAME, "ab") as f:
        f.write(_field(77).tobytes(order="C"))

    reopened = vfc.RunFieldCache(root)
    assert reopened.n == 1                       # the orphan is not a row
    reopened.put("b|8.0", _field(2))             # ...and its SLOT is written straight over
    assert np.array_equal(reopened.get("b|8.0"), _field(2))
    assert np.array_equal(reopened.get("a|16.0"), _field(1))
    reopened.close()
    # The reuse is the substantive claim and it is a claim about the file, not about the
    # lookup: a store that skipped the orphan slot would still read back correctly while
    # leaking one record per kill, and after enough resumes the leak is the store.
    assert (root / vfc.FIELDS_NAME).stat().st_size == 2 * H * W * 4
    assert vfc.RunFieldCache(root).finalize()["n"] == 2


def test_run_cache_stops_at_a_torn_final_index_line(tmp_path):
    root = tmp_path / "vf"
    c = vfc.RunFieldCache(root, policy="tok")
    c.put("a|16.0", _field(1))
    c.put("b|8.0", _field(2))
    c.close()
    txt = (root / vfc.RUN_INDEX_NAME).read_text(encoding="utf-8")
    (root / vfc.RUN_INDEX_NAME).write_text(txt[:-8], encoding="utf-8")   # half a line
    reopened = vfc.RunFieldCache(root)
    assert reopened.n == 1 and reopened.has("a|16.0")
    reopened.close()


def test_static_cache_refuses_a_population_whose_key_order_differs(tmp_path):
    """The retrospective cache's frozen key order, pinned. Appending to a grown log would
    re-index every row and hand one candidate's field back for another's key."""
    root = tmp_path / "static"
    vfc.FieldCache(root, ["a|16.0", "b|8.0"], policy="tok", mode="r+")
    with pytest.raises(SystemExit):
        vfc.FieldCache(root, ["b|8.0", "a|16.0"], policy="tok", mode="r+")   # reordered
    with pytest.raises(SystemExit):
        vfc.FieldCache(root, ["a|16.0", "b|8.0", "c|None"], policy="tok", mode="r+")
    with pytest.raises(SystemExit):
        vfc.FieldCache(root, ["a|16.0", "b|8.0"], policy="OTHER", mode="r+")  # cap policy


# =========================================================================== #
# one identity for one candidate
# =========================================================================== #
def test_the_view_key_is_the_same_string_in_all_three_structures():
    """The screen cache, the field cache and the walk's visited set must agree on what one
    candidate IS. Three modules form this key; a divergence would file a screen under a key
    the cache never looks up, and the failure would look like a 100% cache miss rather than
    like a bug."""
    for atom, k in (("abc", 16.0), ("abc", None), ("abc", 8.0)):
        key = mvs.view_key(atom, k)
        assert key == vfc.row_key(dict(atom_key=atom, k=k))
        assert key == f"{atom}|{k}"          # steered_frontier's man_visited key


# =========================================================================== #
# selection
# =========================================================================== #
def test_composite_sort_key_orders_screened_over_vetoed_over_unscreened():
    hi = mvs.composite_sort_key(dict(screened=True, composite=12.0))
    lo = mvs.composite_sort_key(dict(screened=True, composite=0.5))
    vetoed = mvs.composite_sort_key(dict(screened=True, composite=-0.9))
    unscreened = mvs.composite_sort_key(dict(screened=False))
    assert hi > lo > vetoed > unscreened
    # A vetoed row scores in [-1, 0) and MUST still outrank an unmeasured one: conflating
    # them would let the veto band act as a floor and quietly promote unscreenable rows.
    assert vetoed > unscreened
    # ...and a screened row whose composite is somehow absent is not treated as good.
    assert mvs.composite_sort_key(dict(screened=True, composite=None)) < vetoed


def test_compact_keeps_selection_columns_and_drops_the_radial_profile():
    rec = dict(screened=True, composite=1.0, vetoed=False, size_factor=1.0,
               radial_range=3.0, radial_rings=4.0, interior_fraction=0.1,
               band_coverage=0.5, band_coverage_q25=0.4, view_fw=1e-3,
               interior_radial=[0.0] * 8, escaped_px=2304, smooth_max=900.0,
               **{fm.POLICY_KEY: "tok"})
    out = mvs.compact(rec)
    assert out["composite"] == 1.0 and out[fm.POLICY_KEY] == "tok"
    # The checkpoint is not a second copy of the record — the profile and the raw pixel
    # counts are already durable on the maneuver row and in the field.
    assert "interior_radial" not in out and "escaped_px" not in out
    assert set(out) <= set(mvs.STATE_KEYS)


# =========================================================================== #
# the cache never costs a candidate
# =========================================================================== #
class _ExplodingFields:
    def put(self, *a, **k):
        raise OSError("disk full")


def test_a_field_cache_failure_does_not_fail_the_screen(monkeypatch):
    """The cache is an accelerator for LATER work. Losing a row of it costs one numpy pass;
    raising here would cost the candidate, which is the thing the run exists to produce."""
    def fake_screen_view(cx, cy, fw, **kw):
        return dict(screened=True, radial_range=5.0, radial_rings=10.0,
                    interior_fraction=0.0, band_coverage=0.8, band_coverage_q25=0.7,
                    view_fw=float(fw)), _field(0)

    monkeypatch.setattr(mvs, "screen_view", fake_screen_view)
    c = mvs.ViewScreenCache(_params(), workers=1, fields=_ExplodingFields())
    out = c.screen_many([dict(view_key="a|16.0", cx="0", cy="0", fw=1e-3)])
    assert out["a|16.0"]["screened"] and out["a|16.0"]["composite"] > 0
    assert c.n_fields_cached == 0             # counted honestly as not cached


def test_an_unscreened_row_gets_no_composite_rather_than_a_sentinel(monkeypatch):
    monkeypatch.setattr(mvs, "screen_view", lambda *a, **k: (
        dict(screened=False, screen_reason="f64_spacing_wall_at_screen_geometry"), None))
    c = mvs.ViewScreenCache(_params(), workers=1)
    rec = c.screen_many([dict(view_key="a|None", cx="0", cy="0", fw=1e-30)])["a|None"]
    assert rec["screened"] is False and rec["composite"] is None
    assert rec["vetoed"] is None and rec["screen_reason"]


def test_the_screen_cache_pays_once_per_view_and_hits_thereafter(monkeypatch):
    calls = []

    def fake(cx, cy, fw, **kw):
        calls.append((str(cx), float(fw)))
        return dict(screened=True, radial_range=1.0, radial_rings=1.0,
                    interior_fraction=0.0, band_coverage=0.5, band_coverage_q25=0.5,
                    view_fw=float(fw)), None

    monkeypatch.setattr(mvs, "screen_view", fake)
    c = mvs.ViewScreenCache(_params(), workers=1)
    jobs = [dict(view_key="a|16.0", cx="0", cy="0", fw=1e-3),
            dict(view_key="a|8.0", cx="0", cy="0", fw=5e-4),
            dict(view_key="a|16.0", cx="0", cy="0", fw=1e-3)]   # dup inside one batch
    c.screen_many(jobs)
    assert len(calls) == 2                       # one field per DISTINCT view
    # The in-batch duplicate is collapsed before any screen runs, so it is not a cache HIT
    # — it never reached the cache. Stated because the two are easy to conflate and only
    # the second is evidence that the cross-batch cache is doing anything.
    assert c.n_hits == 0
    c.screen_many(jobs)                          # a later batch re-enumerates the same views
    assert len(calls) == 2 and c.n_hits == 3     # all three now served from the cache


def test_the_pass_budget_bounds_the_whole_screen_not_each_field(monkeypatch):
    """A backstop longer than the job's budget is not a backstop (`CLAUDE.md`). The walk
    checks its cap BETWEEN batches, so an unbounded screen inside one is outside the cap."""
    monkeypatch.setattr(mvs, "screen_view", lambda *a, **k: (
        dict(screened=True, radial_range=1.0, radial_rings=1.0, interior_fraction=0.0,
             band_coverage=0.5, band_coverage_q25=0.5, view_fw=1.0), None))
    c = mvs.ViewScreenCache(_params(), workers=1)
    out = c.screen_many([dict(view_key=f"a|{i}", cx="0", cy="0", fw=1e-3)
                         for i in range(4)], budget_s=-1.0)     # already exhausted
    # Every job comes back as a ROW with a named reason — never as a silent absence, which
    # downstream cannot tell apart from "never proposed".
    assert len(out) == 4
    assert all(r["screened"] is False and r["screen_reason"] == "screen_budget_exhausted"
               for r in out.values())


def test_state_round_trips_through_the_checkpoint(monkeypatch):
    monkeypatch.setattr(mvs, "screen_view", lambda *a, **k: (
        dict(screened=True, radial_range=1.0, radial_rings=1.0, interior_fraction=0.0,
             band_coverage=0.5, band_coverage_q25=0.5, view_fw=1.0), None))
    c = mvs.ViewScreenCache(_params(), workers=1)
    c.screen_many([dict(view_key="a|16.0", cx="0", cy="0", fw=1e-3)])
    blob = json.loads(json.dumps(c.state_dict()))       # must survive JSON, not just copy
    c2 = mvs.ViewScreenCache(_params(), workers=1)
    c2.load_state(blob)
    assert c2.get("a|16.0")["composite"] == c.get("a|16.0")["composite"]
    assert c2.n_screened == 1


# =========================================================================== #
# the seam: what the WALK does with the view screen
# =========================================================================== #
import minibrot_maneuvers as mnv          # noqa: E402
import steered_frontier as sf             # noqa: E402

_FLOOR_TOTALS = ("man_quota_bound", "man_quota_unfilled", "man_quota_passed_over")


def _floor_obj(B, quota, *, view_prior=True, logged=None):
    return types.SimpleNamespace(B=B, man_quota=quota, maneuvers=True,
                                 man_range_prior=False, man_view_prior=view_prior,
                                 batch_i=1, man_passed_logged=set(),
                                 _log_maneuver=(logged.append if logged is not None
                                                else (lambda row: None)),
                                 totals={k: 0 for k in _FLOOR_TOTALS})


def _comp_pool(spec):
    """spec: (node_id, priority, composite|None). None == unscreened."""
    return [dict(node_id=i, priority=p, partition="mandelbrot",
                 man={"op": "snap_to_nucleus", "k": 16.0, "atom_key": f"a{i}",
                      "screen_frame": "view", "screened": c is not None, "composite": c})
            for i, p, c in spec]


def test_the_quota_is_filled_by_descending_composite_not_by_priority():
    pool = _comp_pool([(1, 9.0, 0.4), (2, 8.0, 31.5), (3, 7.0, None), (4, 6.0, 12.0)])
    o = _floor_obj(3, 2)
    batch, rest = sf.SteeredFrontier._split_reserved(o, pool)
    assert [n["node_id"] for n in batch[:2]] == [2, 4]        # the two best PICTURES
    assert len(batch) == 3 and len(rest) == 1


def test_a_vetoed_candidate_still_outranks_an_unscreenable_one_for_a_slot():
    """The veto sorts to bottom AMONG MEASURED rows; it is not a statement that the frame is
    worse than one nobody could measure. Conflating them would let the veto band act as a
    floor and quietly promote the deep tail."""
    o = _floor_obj(2, 1)
    batch, _ = sf.SteeredFrontier._split_reserved(
        o, _comp_pool([(1, 9.0, None), (2, 1.0, -0.5)]))
    assert batch[0]["node_id"] == 2


def test_the_view_sort_never_changes_how_many_slots_the_floor_takes():
    """The v1.5 change is WHICH maneuver fills a slot, never how many — the same invariant
    the v1.4 range sort holds, restated because a new sort key is a new chance to break it."""
    spec = [(1, 9.0, 0.1), (2, 8.0, 99.0), (3, 7.0, None), (4, 6.0, 50.0)]
    off = _floor_obj(3, 2, view_prior=False)
    on = _floor_obj(3, 2, view_prior=True)
    b_off, r_off = sf.SteeredFrontier._split_reserved(off, _comp_pool(spec))
    b_on, r_on = sf.SteeredFrontier._split_reserved(on, _comp_pool(spec))
    assert len(b_off) == len(b_on) == 3 and len(r_off) == len(r_on) == 1
    assert off.totals["man_quota_unfilled"] == on.totals["man_quota_unfilled"] == 0
    assert {n["node_id"] for n in b_on} != {n["node_id"] for n in b_off}   # not vacuous


def test_passed_over_rows_carry_the_composite_and_the_frame_they_were_scored_on():
    """Part of the run's contract: every candidate is RECORDED with all measures, pushed or
    not — the low bins are what supply the negative class to the label batches. A
    passed-over row that omits the score is a row the batch builder cannot stratify on."""
    logged = []
    o = _floor_obj(2, 1, logged=logged)
    sf.SteeredFrontier._split_reserved(o, _comp_pool([(1, 9.0, 5.0), (2, 8.0, 0.2)]))
    rows = [r for r in logged if r.get("unused_reason") == "quota_passed_over"]
    assert len(rows) == 1 and rows[0]["node_id"] == 2
    assert rows[0]["composite"] == 0.2 and rows[0]["screen_frame"] == "view"


def _consume_obj(view_prior, dist=None):
    pushed = []
    return pushed, types.SimpleNamespace(
        maneuvers=True, man_visited=set(), man_range_prior=False, man_range_gain=0.5,
        man_range_dist=None, man_view_prior=view_prior, man_view_gain=0.5,
        man_comp_dist=(dist if dist is not None else sf.msc.RangeDistribution()),
        batch_i=1, beta=0.0, rng=np.random.default_rng(0), frontier=pushed,
        totals={k: 0 for k in sf.MAN_TOTALS},
        new_node_id=lambda: len(pushed) + 1, _log_maneuver=lambda row: None)


def _mnv(k=16.0, atom="KEY"):
    return mnv.Maneuver(op="snap_to_nucleus", available=True, k=k, cx="0.1", cy="0.2",
                        fw=1e-3, depth=3, atom_id="x", atom_key=atom, period=5,
                        log10_abs_A=2.0, window_scale=1e-4, extra={"degree": 2})


_PARENT = dict(root_id=1, c=None, partition="mandelbrot")
_SCREEN = dict(screened=True, composite=9.0, vetoed=False, size_factor=1.0,
               radial_range=4.0, radial_rings=20.0, interior_fraction=0.05,
               band_coverage=0.8, band_coverage_q25=0.7, view_fw=1e-3,
               maxiter_policy_token="mi12000k0.3c4800-67000")


def test_the_pushed_node_records_the_composite_and_names_the_frame():
    pushed, o = _consume_obj(True)
    assert sf.SteeredFrontier._consume_maneuver(o, _mnv(), _PARENT, screen=_SCREEN) == 1
    man = pushed[0]["man"]
    assert man["composite"] == 9.0 and man["screen_frame"] == "view"
    assert man["band_coverage_q25"] == 0.7 and man["interior_fraction"] == 0.05
    # ... and under the ATOM screen the same node names the other frame, so a readout that
    # joins two runs can never pool a 4x radial_range with a view-frame one.
    pushed2, o2 = _consume_obj(False)
    sf.SteeredFrontier._consume_maneuver(o2, _mnv(), _PARENT, screen=_SCREEN)
    assert pushed2[0]["man"]["screen_frame"] == "atom4x"
    assert "composite" not in pushed2[0]["man"]


def test_the_composite_prior_is_bounded_and_symmetric_about_the_neutral_prior():
    """The bound is the whole design: an ordinary node's cheap_eord runs over [0, K-1], so a
    maneuver can out-compete a SCORED node only through the quota floor, never the prior."""
    dist = sf.msc.RangeDistribution()
    for v in range(20):
        dist.add(float(v))
    top, o = _consume_obj(True, dist)
    sf.SteeredFrontier._consume_maneuver(o, _mnv(atom="A"), _PARENT,
                                         screen=dict(_SCREEN, composite=1e9))
    bottom, o2 = _consume_obj(True, dist)
    sf.SteeredFrontier._consume_maneuver(o2, _mnv(atom="B"), _PARENT,
                                         screen=dict(_SCREEN, composite=-1e9))
    assert top[0]["man"]["composite_pct"] == 1.0
    assert bottom[0]["man"]["composite_pct"] == 0.0
    # +/- gain/2 around NEUTRAL_PRIOR, with beta=0 and the Gumbel drawn from the same seed.
    assert top[0]["priority"] - bottom[0]["priority"] == pytest.approx(0.5, abs=1e-9)


def test_an_unscreened_candidate_draws_the_neutral_prior_not_a_penalty():
    dist = sf.msc.RangeDistribution()
    for v in range(20):
        dist.add(float(v))
    got, o = _consume_obj(True, dist)
    sf.SteeredFrontier._consume_maneuver(o, _mnv(), _PARENT,
                                         screen=dict(screened=False,
                                                     screen_reason="f64_spacing_wall"))
    assert got[0]["man"]["composite_pct"] == 0.5          # exactly neutral: delta 0


def test_the_screen_key_is_the_view_under_v1_5_and_the_atom_under_v1_4():
    m = _mnv(k=8.0, atom="KEY")
    on = types.SimpleNamespace(man_view_prior=True)
    off = types.SimpleNamespace(man_view_prior=False)
    assert sf.SteeredFrontier._screen_key(on, m) == "KEY|8.0"
    assert sf.SteeredFrontier._screen_key(off, m) == "KEY"


def test_the_neighbourhood_top_n_ranks_an_atom_by_its_BEST_framing():
    """Selection is per DISTINCT ATOM but the score is per VIEW, so the reduction has to be
    chosen rather than fallen into. max answers the question the selection asks - is there
    a good picture here - where a mean would let two weak framings outvote one strong one."""
    o = types.SimpleNamespace(man_view_prior=True, man_nbh_n=1,
                              totals={"man_nbh_passed_over": 0})
    o._screen_key = lambda m: sf.SteeredFrontier._screen_key(o, m)
    produced = [dict(m=_mnv(k=kk, atom=a), parent=None, nbh_group=7)
                for a, kk in (("A", 8.0), ("A", 16.0), ("B", 8.0), ("B", 16.0))]
    scores = {"A|8.0": dict(screened=True, composite=1.0),
              "A|16.0": dict(screened=True, composite=50.0),    # A's best is the best
              "B|8.0": dict(screened=True, composite=9.0),
              "B|16.0": dict(screened=True, composite=9.0)}     # B's mean is higher
    keep = sf.SteeredFrontier._nbh_top_n(o, produced, scores)
    assert keep[7] == {"A"}
    assert o.totals["man_nbh_passed_over"] == 1


def test_the_two_screen_flags_refuse_to_run_together():
    """Two sort keys for one seam, measured on different frames. Running both would screen
    twice and let WHICH one selected depend on argument order. Asserted on the SOURCE
    because building a whole SteeredFrontier needs a model, a ledger and a run dir - so the
    alternative was a fixture that reimplements the guard and could not fail with it."""
    src = (HERE / "steered_frontier.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    hit = [i for i, ln in enumerate(lines)
           if "--maneuver-range-prior and --maneuver-view-prior both set" in ln]
    assert len(hit) == 1
    before = "\n".join(lines[max(0, hit[0] - 6):hit[0]])
    assert "if self.man_range_prior:" in before and "raise SystemExit(" in before
    assert "if self.maneuvers and self.man_view_prior:" in before
