#!/usr/bin/env python
"""Sink isolation for the emission record stores — the guard, and its non-vacuity.

`tools/atlas/test_discovery_sinks.py`'s shape, applied to the other durable store. The
property under test is not "a smoke run overwrites nothing" (it never did — the stores upsert
by a run-id-prefixed key). It is that a throwaway run cannot ADD rows to a file whose whole
purpose is that a later calibration pass reads accumulated real releases out of it.

  uv run pytest tools/emission/test_emission_sinks.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import emission_sinks as S       # noqa: E402
from tools.emission import release_record as RR      # noqa: E402
from tools.mining import gate_report as GR           # noqa: E402

SITE = "emission_diversity_v1"


@pytest.fixture(autouse=True)
def _unbind():
    """`use()` is process-global state that is NOT a monkeypatch target (it is a plain
    setter), so every test here restores it — an autouse fixture at least as broad as
    anything that binds it. Without this the binding leaks and unrelated tests write their
    records wherever the last test here pointed."""
    S.use(None)
    yield
    S.use(None)


# --------------------------------------------------------------------------- #
# 1. the resolution decision
# --------------------------------------------------------------------------- #
def test_production_is_the_default_and_lands_in_data():
    root = S.resolve_record_root(ROOT, smoke=False, explicit=None)
    assert root == S.default_record_root(ROOT).resolve()
    assert S.resolves_under_data(ROOT, root, SITE), \
        "the production root MUST resolve under data/ — otherwise this guard tests nothing"


def test_a_throwaway_run_is_redirected_out_of_data():
    root = S.resolve_record_root(ROOT, smoke=True, explicit=None, run_id="smoke_x")
    assert S.resolves_under_data(ROOT, root, SITE) == []
    assert "scratch" in root.as_posix()
    assert S.assert_isolated(ROOT, root, SITE)          # returns the sink list, non-empty


def test_an_explicit_root_wins_over_the_throwaway_redirect(tmp_path):
    root = S.resolve_record_root(ROOT, smoke=True, explicit=tmp_path / "rec", run_id="ignored")
    assert root == (tmp_path / "rec").resolve()


def test_two_throwaway_runs_do_not_share_a_root():
    """These files accumulate BY RUN. A fixed smoke dir (which is right for `discovery_sinks`)
    would make the second smoke's readout a population it did not produce."""
    a = S.resolve_record_root(ROOT, smoke=True, explicit=None, run_id="run_a")
    b = S.resolve_record_root(ROOT, smoke=True, explicit=None, run_id="run_b")
    assert a != b


def test_the_assertion_fires_on_a_root_under_data():
    """Non-vacuity: the guard is red for exactly the input it exists to reject."""
    bad = S.default_record_root(ROOT)
    with pytest.raises(S.SinkNotIsolated) as e:
        S.assert_isolated(ROOT, bad, SITE)
    assert "release_records" in str(e.value) or "data/" in str(e.value).replace("\\", "/")


def test_derive_sinks_covers_every_file_a_run_writes():
    """The completeness half. A sink missing here is a sink the assertion is blind to — the
    failure `discovery_sinks`' `gather` entry had. Cross-checked against the writers' OWN path
    functions under a binding, not against a second list."""
    root = ROOT / "scratch" / "emission" / "_ephemeral_records" / "coverage_probe"
    derived = {p.resolve() for p in S.derive_sinks(root, SITE).values()}
    S.use(root)
    actual = {RR.record_path(SITE).resolve(), RR.runs_path(SITE).resolve(),
              (GR.gate_log_dir() / f"{SITE}.jsonl").resolve()}
    assert actual == derived


# --------------------------------------------------------------------------- #
# 2. the writers follow the binding
# --------------------------------------------------------------------------- #
def test_release_record_writes_under_the_binding_and_not_data(tmp_path):
    S.use(tmp_path / "rec")
    row = RR.decision_row(run_id="r1", stage=RR.STAGE_GATE, join_key="k", location_id="L",
                          location={}, partition="mandelbrot", morph_cluster="m#0",
                          decision="admitted", score=0.9)
    path, n_total, n_new = RR.write_decisions(SITE, [row])
    assert path.resolve() == (tmp_path / "rec" / S.RELEASE_RECORDS / f"{SITE}.jsonl").resolve()
    assert (n_total, n_new) == (1, 1)
    assert S.resolves_under_data(ROOT, tmp_path / "rec", SITE) == []
    # ...and the read side resolves the same binding, so an ephemeral run can read its own log.
    assert len(RR.read_decisions(SITE, run_id="r1")) == 1


def test_gate_report_writes_under_the_binding_and_not_data(tmp_path):
    S.use(tmp_path / "rec")
    row = GR.gate_report_row(site=SITE, key="k", location={}, style="tia", palette="p",
                             p_ge3=0.4, release_threshold=0.5, pool_floor=0.25,
                             pooled=True, selected=False, selection_stage="release")
    path, n_total, _n_cut, _n_cut_sel, _pool = GR.write_gate_report(SITE, [row])
    assert path.resolve() == (tmp_path / "rec" / S.MINING_GATE_REPORTS / f"{SITE}.jsonl").resolve()
    assert n_total == 1


def test_unbinding_restores_the_production_paths():
    """The binding is not a one-way door: with nothing bound, both writers resolve the durable
    store through `paths.durable()` exactly as before."""
    S.use(None)
    assert RR.record_path(SITE) == (ROOT / RR.RECORD_DIR_REL / f"{SITE}.jsonl")
    assert GR.gate_log_dir() == GR.GATE_LOG_DIR
    assert S.resolves_under_data(ROOT, S.default_record_root(ROOT), SITE)


def test_the_ephemeral_run_never_touches_the_production_file(tmp_path):
    """End-to-end of the property, on the real production path: a bound run writes its rows,
    and the production file's bytes are untouched (including when it does not exist)."""
    prod = RR.record_path(SITE) if S.is_production() else None
    before = prod.read_bytes() if prod and prod.exists() else None
    S.use(tmp_path / "rec")
    RR.write_decisions(SITE, [RR.decision_row(
        run_id="ephemeral_run", stage=RR.STAGE_RELEASE, join_key="k", location_id="L",
        location={}, partition="phoenix", morph_cluster="m#1", decision="selected", score=0.95)])
    S.use(None)
    after = prod.read_bytes() if prod and prod.exists() else None
    assert after == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
