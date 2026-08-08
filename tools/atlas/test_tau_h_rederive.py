"""`tau_h_rederive` — the row cache is bulk, and it carries the K=4 triple.

Two behaviours, each earned by a failure:

* the row-level scores were written under `scratch/` and re-rendered from zero TWICE after a
  scratch wipe. They are expensive-but-deterministic given the committed ledgers plus the
  active weights, i.e. `bulk()`, and this pins that the write site declares it.
* the rows stored the K=3-shaped `(score, p_notbad, p_good)` triple, which cannot reproduce
  the SERVED decode on a K=4 head (`corn_decode(nb, pg, t_good, pg4)`): the third cutpoint
  never reached the row, so a stored row was capped at class 3 by the reader rather than by
  the head. This pins the K-aware shape and the refusal to mix shapes in one cache.

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


def _row(key="h_run_1_0"):
    return dict(pop="harvest", run="campaign1/breadth", partition="mandelbrot", depth=3,
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
    base = dict(per_partition=200, n_rows_harvest=2400, n_rows_walk=1092,
                keep=0.9, seed=0, combine="min", tau_h_base={"mandelbrot": 0.63})
    base.update(over)
    return base


def test_a_rederivation_over_a_different_population_refuses_to_overwrite(tmp_path):
    out = tmp_path / "tau_h_base_v11.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    with pytest.raises(SystemExit, match="DIFFERENT population"):
        thr.assert_not_superseding(out, _art(n_rows_harvest=63217, per_partition=100000), False)


def test_the_refusal_names_every_field_that_moved(tmp_path):
    """The message has to say WHAT differs, or the operator cannot tell a population
    enlargement from a fat-fingered --keep."""
    out = tmp_path / "a.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        thr.assert_not_superseding(out, _art(n_rows_harvest=63217, keep=0.8), False)
    msg = str(e.value)
    assert "n_rows_harvest: 2400 -> 63217" in msg and "keep: 0.9 -> 0.8" in msg
    assert "per_partition" not in msg           # unchanged fields stay out of the message


def test_overwrite_is_the_deliberate_act_that_allows_it(tmp_path):
    out = tmp_path / "a.json"
    out.write_text(json.dumps(_art()), encoding="utf-8")
    thr.assert_not_superseding(out, _art(n_rows_harvest=63217), True)


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
# the truncation mixture
# --------------------------------------------------------------------------- #
class _FakeRun:
    def __init__(self, name, rows):
        self.name, self.log, self._rows = name, name, rows


def test_truncation_record_reports_the_mixture_not_a_single_level(monkeypatch):
    """`caveat` used to say the harvest arm is left-truncated at "the previous head's tau_h",
    singular. It never was: every row carries the tau_h LIVE FOR ITS OWN RUN, so pooling runs
    pools tau eras and the bound's tightness varies by partition. This pins that the mixture
    is reported rather than collapsed."""
    runs = [_FakeRun("old", [dict(cx=1, fw=1e-3, partition="mandelbrot", tau_h=0.70)] * 3),
            _FakeRun("new", [dict(cx=1, fw=1e-3, partition="mandelbrot", tau_h=0.02)] * 1)]
    monkeypatch.setattr(thr.run_record, "require_rows", lambda log: dict(
        old=runs[0]._rows, new=runs[1]._rows)[log])
    rec = thr.truncation_record(runs)
    mb = rec["by_partition"]["mandelbrot"]
    assert mb["n_levels"] == 2 and mb["n_rows"] == 4
    assert mb["min"] == 0.02 and mb["max"] == 0.70
    assert mb["row_weighted_mean"] == pytest.approx((0.70 * 3 + 0.02) / 4)
    assert rec["by_run"] == {"old": {"mandelbrot": 0.70}, "new": {"mandelbrot": 0.02}}


def test_untagged_rows_are_counted_never_guessed(monkeypatch):
    """A row written before the tau_h stamp existed has no level. Imputing one would invent
    a truncation the pool does not have, so it is counted separately and excluded."""
    run = _FakeRun("r", [dict(cx=1, fw=1e-3, partition="mandelbrot", tau_h=None),
                         dict(cx=1, fw=1e-3, partition="mandelbrot", tau_h=0.5)])
    monkeypatch.setattr(thr.run_record, "require_rows", lambda log: run._rows)
    mb = thr.truncation_record([run])["by_partition"]["mandelbrot"]
    assert mb["unstamped_rows"] == 1 and mb["n_rows"] == 1
    assert mb["row_weighted_mean"] == pytest.approx(0.5)


def test_phoenix_and_geometryless_rows_are_excluded_as_the_deriver_excludes_them(monkeypatch):
    """The record must describe the population the estimator actually cuts on, not the log."""
    run = _FakeRun("r", [dict(cx=None, fw=1e-3, partition="mandelbrot", tau_h=0.5),
                         dict(cx=1, fw=None, partition="mandelbrot", tau_h=0.5),
                         dict(cx=1, fw=1e-3, partition="phoenix", tau_h=0.5)])
    monkeypatch.setattr(thr.run_record, "require_rows", lambda log: run._rows)
    assert thr.truncation_record([run])["by_partition"] == {}
