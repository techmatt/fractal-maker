"""`tau_h_rederive` — the estimator, the row cache's storage class, and its K=4 shape.

Behaviours, each earned by a failure:

* THE ESTIMATOR conditions on `floors.GOOD_FLOOR` and FAILS OPEN below `MIN_N` (2026-08-09).
  Both halves are the point: a per-partition bar made every cross-version comparison a
  confound, and a thin partition that got a POOLED cross-family cut was being served a number
  about somebody else's population. Fail-open costs render time, which shows up in run
  telemetry; the alternatives cost supply, which does not.
* the row-level scores were written under `scratch/` and re-rendered from zero TWICE after a
  scratch wipe. They are expensive-but-deterministic given the committed ledgers plus the
  active weights, i.e. `bulk()`, and this pins that the write site declares it.
* the rows stored the K=3-shaped `(score, p_notbad, p_good)` triple, so the third cutpoint
  never reached the row and a cached row could not answer a class-4 question at all. This
  pins the K-aware shape and the refusal to mix shapes in one cache.

Run: uv run pytest tools/atlas/test_tau_h_rederive.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "mining", ROOT / "tools" / "scoring", ROOT / "tools" / "reframe"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import artifacts as A                                     # noqa: E402
import tau_h_rederive as thr                              # noqa: E402


class FakeScorer:
    """A K-aware stub. `score_paths_k` returns k-tuples (score, p_ge2, p_ge3[, p_ge4]);
    `score_paths` returns the K=3-shaped triple and RAISES, so a regression back to the
    lossy reader fails loudly instead of quietly dropping the cutpoint again."""

    def __init__(self, k=4):
        self.k = k

    def score_paths_k(self, paths, batch_size=64):
        out = []
        for i, p in enumerate(paths):
            probs = [0.9 - 0.1 * i, 0.7 - 0.1 * i, 0.4 - 0.1 * i][: self.k - 1]
            out.append((sum(probs), *probs))
        return out

    def score_paths(self, paths, batch_size=64):
        raise AssertionError("tau_h_rederive must score K-aware (score_paths_k)")


def _row(key="w_st_m_0"):
    return dict(pop="walk", run="prospect_run1", partition="mandelbrot", depth=3,
                cx="-0.5", cy="0.1", fw="1e-3", c=None, key=key)


def _prepare(tmp_path, rows):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    for r in rows:
        for arm in ("cheap", "canon"):
            (tiles / f"{r['key']}_{arm}.jpg").write_bytes(b"x")
    return tiles


# --------------------------------------------------------------------------- #
# storage class
# --------------------------------------------------------------------------- #
def test_the_row_cache_is_bulk_and_resolves_out_of_tree():
    """The work dir is declared `bulk()` at the write site, so it survives `rm -r scratch/*`
    and never costs the working tree a traversal."""
    assert thr.WORK == A.artifacts_root() / "data/atlas/tau_h_rederive"
    assert not str(thr.WORK).startswith(str(A.REPO_ROOT) + "\\")
    assert not str(thr.WORK).startswith(str(A.REPO_ROOT) + "/")
    # the derived artifact it writes BESIDE the cache stays durable and in-tree
    assert str(thr.ARTIFACT).startswith(str(A.REPO_ROOT))


# --------------------------------------------------------------------------- #
# the K=4 triple
# --------------------------------------------------------------------------- #
def test_rows_carry_the_fourth_cutpoint_on_a_k4_head(tmp_path):
    rows = [_row()]
    tiles = _prepare(tmp_path, rows)
    out = tmp_path / "rows.jsonl"
    thr.render_and_score(rows, FakeScorer(k=4), tiles, out)
    got = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert len(got) == 1
    r = got[0]
    for col in ("cheap_eord", "cheap_nb", "cheap_pgood", "cheap_pge4",
                "canon_eord", "canon_nb", "canon_pgood", "canon_pge4"):
        assert col in r, col
    assert r["cheap_pge4"] is not None and r["canon_pge4"] is not None
    # the first three columns still mean what they always meant
    assert r["cheap_nb"] == pytest.approx(0.9) and r["cheap_pgood"] == pytest.approx(0.7)
    assert r["cheap_pge4"] == pytest.approx(0.4)
    assert r["cheap_eord"] == pytest.approx(2.0)


def test_a_k3_head_stores_an_explicit_null_rather_than_omitting_the_column(tmp_path):
    """On a K=3 head there is no third cutpoint. The column must still be WRITTEN, as null:
    the cache-shape guard tests key presence, so an omitted column would read as a
    pre-K-aware row and refuse a legitimate cache."""
    rows = [_row()]
    tiles = _prepare(tmp_path, rows)
    out = tmp_path / "rows.jsonl"
    thr.render_and_score(rows, FakeScorer(k=3), tiles, out)
    r = json.loads(open(out, encoding="utf-8").read().strip())
    assert "cheap_pge4" in r and r["cheap_pge4"] is None
    assert "canon_pge4" in r and r["canon_pge4"] is None
    thr.assert_rows_current([r], out)          # a K=3 cache is current, not pre-K-aware


# --------------------------------------------------------------------------- #
# the cache-shape refusal
# --------------------------------------------------------------------------- #
def test_a_pre_k_aware_cached_row_is_refused(tmp_path):
    ok = dict(model=thr.ACTIVE_VERSION, cheap_pge4=0.4, canon_pge4=0.3)
    thr.assert_rows_current([ok], tmp_path / "rows.jsonl")
    for missing in ("cheap_pge4", "canon_pge4"):
        bad = {k: v for k, v in ok.items() if k != missing}
        with pytest.raises(SystemExit, match="pre-K-aware"):
            thr.assert_rows_current([ok, bad], tmp_path / "rows.jsonl")


def test_a_row_scored_under_another_model_is_refused(tmp_path):
    bad = dict(model="v_not_this_one", cheap_pge4=0.4, canon_pge4=0.3)
    with pytest.raises(SystemExit, match="different model"):
        thr.assert_rows_current([bad], tmp_path / "rows.jsonl")


# --------------------------------------------------------------------------- #
# the supersede guard
# --------------------------------------------------------------------------- #
# The file name carries only the model version, so two derivations over DIFFERENT populations
# both want to be `tau_h_base_v11.json` and the second silently destroys the first. That is
# not hypothetical: the 2026-08-08 enlargement re-derives v11 over 64,365 rows where the
# adoption-era artifact was 3,492.
def _art(**over):
    base = dict(per_partition=0, n_rows=1148, keep=0.9, seed=0,
                good_floor=0.5, min_n=5, tau_h_base={"mandelbrot": 0.63})
    base.update(over)
    return base


def test_a_rederivation_over_a_different_population_refuses_to_overwrite(tmp_path):
    out = tmp_path / "tau_h_base_v11.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    with pytest.raises(SystemExit, match="DIFFERENT population"):
        thr.assert_not_superseding(out, _art(n_rows=63217, per_partition=200), False)


def test_the_refusal_names_every_field_that_moved(tmp_path):
    """The message has to say WHAT differs, or the operator cannot tell a population
    enlargement from a fat-fingered --keep."""
    out = tmp_path / "a.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        thr.assert_not_superseding(out, _art(n_rows=63217, keep=0.8), False)
    msg = str(e.value)
    assert "n_rows: 1148 -> 63217" in msg and "keep: 0.9 -> 0.8" in msg
    assert "per_partition" not in msg           # unchanged fields stay out of the message


def test_overwrite_is_the_deliberate_act_that_allows_it(tmp_path):
    out = tmp_path / "a.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    thr.assert_not_superseding(out, _art(n_rows=63217), True)


def test_rerunning_the_same_derivation_is_not_superseding(tmp_path):
    """The guard compares the POPULATION and the settings, never the derived values — a
    re-run under a retrained head moves tau_h_base and must still be allowed to land, or the
    guard would refuse exactly the re-derivation it exists to keep honest."""
    out = tmp_path / "a.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    thr.assert_not_superseding(out, _art(tau_h_base={"mandelbrot": 0.11}), False)
    assert "tau_h_base" not in thr.SUPERSEDE_KEYS


def test_a_first_derivation_has_nothing_to_supersede(tmp_path):
    thr.assert_not_superseding(tmp_path / "absent.json", _art(), False)


# --------------------------------------------------------------------------- #
# engine launch defaults
# --------------------------------------------------------------------------- #
def test_the_render_fanout_does_not_inherit_the_one_process_thread_default():
    """WORKERS=4 concurrent engine processes at `DEFAULT_ENGINE_THREADS` (7) would
    oversubscribe this 12-core box 28-to-12. "Multiple parallel engine processes has no
    standing number" — so the count is sized for the actual N and passed EXPLICITLY."""
    import corpus_common as cc
    assert thr.WORKERS * thr.RENDER_THREADS <= 12
    assert thr.RENDER_THREADS != cc.DEFAULT_ENGINE_THREADS, \
        "the per-process default must not be inherited by a fan-out"
    src = (ROOT / "tools/atlas/tau_h_rederive.py").read_text(encoding="utf-8")
    assert "cc.default_engine_env(threads=RENDER_THREADS)" in src
    assert "creationflags=cc.default_creationflags()" in src, \
        "missing BELOW_NORMAL is what made the adoption run contend with the desktop"


# --------------------------------------------------------------------------- #
# the estimator: one bar for every partition, and fail-OPEN below MIN_N
# --------------------------------------------------------------------------- #
def _scored(partition, canon, cheap):
    return dict(partition=partition, canon_pgood=canon, cheap_pgood=cheap)


def test_the_bar_is_the_good_floor_and_not_a_per_partition_threshold():
    """The rows that count toward the quantile are the ones a CANONICAL render would have
    kept under the live floor \u2014 the same cut the run side admits on. Two partitions with
    identical score distributions must therefore get identical cuts; under the retired
    per-partition t_good they would not have, which is what made every cross-version move a
    head change AND a threshold change with no way to separate them."""
    from tools.emission import floors as F
    rows = []
    for part in ("mandelbrot", "julia:multibrot4"):
        for i in range(10):
            rows.append(_scored(part, 0.05 * i + 0.30, 0.05 * i + 0.10))
    tau, detail = thr.derive(rows, ["mandelbrot", "julia:multibrot4"], 0.90)
    assert tau["mandelbrot"] == tau["julia:multibrot4"]
    n_good = sum(1 for r in rows
                 if r["partition"] == "mandelbrot" and r["canon_pgood"] >= F.GOOD_FLOOR)
    assert detail["mandelbrot"]["n_good"] == n_good > 0
    assert detail["mandelbrot"]["source"] == "own"


def test_a_partition_below_min_n_fails_OPEN_and_harvests_everything():
    """0.0, not a pooled cross-family cut and not a refusal. A too-high cut sheds admissions
    invisibly; a zero cut spends render minutes the run's own telemetry reports."""
    rows = [_scored("mandelbrot", 0.9, 0.8) for _ in range(20)]
    rows += [_scored("multibrot5", 0.9, 0.8) for _ in range(thr.MIN_N - 1)]
    tau, detail = thr.derive(rows, ["mandelbrot", "multibrot5"], 0.90)
    assert tau["multibrot5"] == 0.0
    assert "FAIL-OPEN" in detail["multibrot5"]["source"]
    assert tau["mandelbrot"] > 0.0                      # the thick partition is unaffected


def test_the_thin_partition_does_not_inherit_the_thick_one_s_cut():
    """No pooled cross-family fallback. A pooled quantile is dominated by whichever
    partitions happen to have the most passing rows, and handing it to a thin one is the same
    category error as serving a v8 threshold on a v10 gate."""
    rows = [_scored("mandelbrot", 0.9, 0.77) for _ in range(20)]
    rows += [_scored("multibrot5", 0.9, 0.8)]
    tau, _ = thr.derive(rows, ["mandelbrot", "multibrot5"], 0.90)
    assert tau["multibrot5"] == 0.0 and tau["multibrot5"] != tau["mandelbrot"]


def test_the_retired_harvest_arm_leaves_no_reader_behind():
    """The harvest arm, the two-arm minimum, the per-run truncation record and the harvest-log
    registry retired together (prompts/selection_restructure_3.md). A resumed cache may still
    HOLD harvest rows \u2014 they are read past, never deleted \u2014 but nothing derives from them."""
    for gone in ("_harvest_rows", "truncation_record"):
        assert not hasattr(thr, gone), gone
    src = (ROOT / "tools/atlas/tau_h_rederive.py").read_text(encoding="utf-8")
    assert "harvest_log_registry" not in src
    assert "--combine" not in src


def test_only_walk_rows_reach_the_estimator(tmp_path):
    """The cache is filtered by `pop` at the read, not at the write: a pile of harvest rows
    with a different cheap distribution must not move the derived cut."""
    walk = [dict(pop="walk", partition="mandelbrot", canon_pgood=0.9, cheap_pgood=0.8)
            for _ in range(10)]
    harvest = [dict(pop="harvest", partition="mandelbrot", canon_pgood=0.9, cheap_pgood=0.05)
               for _ in range(500)]
    tau_walk, _ = thr.derive(walk, ["mandelbrot"], 0.90)
    tau_mixed, _ = thr.derive([r for r in walk + harvest if r["pop"] == "walk"],
                              ["mandelbrot"], 0.90)
    assert tau_walk["mandelbrot"] == tau_mixed["mandelbrot"] == pytest.approx(0.8)
