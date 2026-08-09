"""The gate/release record must actually land, carry its run, and ACCUMULATE.

This exists because the failure it guards is silent and total. The emission stage's decisions
were written under `--out` (a scratch/ path), campaign-2's output was wiped, and the loss only
surfaced months later as two unanswerable questions. A record that quietly writes to the wrong
place, or that overwrites the previous run instead of accumulating, reproduces that failure
exactly while looking healthy — so the properties are asserted, not assumed:

  * rows land, one per decision, at both stages;
  * every row carries its run id (a row that cannot be attributed to a run is not evidence);
  * a SECOND run accumulates alongside the first rather than overwriting it — the property
    that distinguishes a record from a snapshot;
  * a re-run of the SAME run upserts in place (a --resume must not double-count itself);
  * the population is recorded, not just the survivors (a rate needs its denominator);
  * the path is durable-registered — `paths.durable()` raises if git would discard it.

The driver's own gate/release plumbing is exercised through a stub Engine that reuses the REAL
`EmissionDiversity` row-building methods, so a change to the row shape or the funnel counts
fails here rather than at the next campaign.

Run: uv run pytest tools/emission/test_release_record.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths  # noqa: E402
from tools.emission import release_record as RR  # noqa: E402

SITE = "test_release_record_site"


def _clear_ignore_cache():
    clear = getattr(paths._is_gitignored, "cache_clear", None)
    if clear:
        clear()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the record dir at a tmp tree, keeping the real durable() assertion in play by
    pointing paths.REPO_ROOT at a location git does not ignore."""
    monkeypatch.setattr(RR, "RECORD_DIR_REL", "data/emission/release_records")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    _clear_ignore_cache()
    yield tmp_path
    _clear_ignore_cache()   # a test may have swapped the function out; undo runs after this


def _rows(run_id, n_gate=3, n_rel=2, first_selected=True):
    out = []
    for i in range(n_gate):
        out.append(RR.decision_row(
            run_id=run_id, stage=RR.STAGE_GATE, join_key=f"loc{i}|smooth|viridis",
            location_id=f"loc{i}", location={"cx": "0.1", "cy": "0.2", "fw": "0.3"},
            partition="julia:mandelbrot", morph_cluster=f"julia#{i}",
            decision=("admitted" if i < 2 else "rejected"),
            score=0.9 - 0.3 * i, reason=(None if i < 2 else "below pool floor"),
            head="wallpaper", floor=0.75, style="smooth", palette="viridis"))
    for i in range(n_rel):
        out.append(RR.decision_row(
            run_id=run_id, stage=RR.STAGE_RELEASE, join_key=f"loc{i}|smooth|viridis",
            location_id=f"loc{i}", location={"cx": "0.1", "cy": "0.2", "fw": "0.3"},
            partition="julia:mandelbrot", morph_cluster=f"julia#{i}",
            decision=("selected" if (i == 0 and first_selected) else "not_selected"),
            score=0.9 - 0.1 * i, head="wallpaper", floor=0.90,
            style="smooth", palette="viridis"))
    return out


def test_rows_land_with_their_run_id(store):
    path, n_total, n_new = RR.write_decisions(SITE, _rows("run_a"))
    assert Path(path).exists()
    assert (n_total, n_new) == (5, 5)
    rows = RR.read_decisions(SITE)
    assert len(rows) == 5
    assert {r["run_id"] for r in rows} == {"run_a"}
    assert {r["stage"] for r in rows} == {RR.STAGE_GATE, RR.STAGE_RELEASE}
    assert all(r["partition"] and r["morph_cluster"] and r["join_key"] for r in rows)
    # the score that a decision was taken on, and the absence of one, are both preserved
    assert [r["decision"] for r in rows if r["stage"] == RR.STAGE_GATE].count("admitted") == 2


def test_second_run_accumulates_rather_than_overwriting(store):
    """The property that makes this a record and not a snapshot."""
    RR.write_decisions(SITE, _rows("run_a"))
    _, n_total, n_new = RR.write_decisions(SITE, _rows("run_b"))
    assert (n_total, n_new) == (10, 5)
    rows = RR.read_decisions(SITE)
    assert {r["run_id"] for r in rows} == {"run_a", "run_b"}
    assert len(RR.read_decisions(SITE, run_id="run_a")) == 5
    assert len(RR.read_decisions(SITE, run_id="run_b")) == 5


def test_same_run_upserts_in_place(store):
    """A --resume re-derives the same keys; it must not double-count its own rows."""
    RR.write_decisions(SITE, _rows("run_a"))
    _, n_total, n_new = RR.write_decisions(SITE, _rows("run_a", first_selected=False))
    assert (n_total, n_new) == (5, 0)
    rel = [r for r in RR.read_decisions(SITE) if r["stage"] == RR.STAGE_RELEASE]
    assert all(r["decision"] == "not_selected" for r in rel)   # replaced, not appended


def test_population_is_recorded_and_accumulates(store):
    """Survivors alone cannot answer 'out of how many' — the denominator is the half that was
    lost with campaign-2."""
    counts = {"intake_admitted": 40, "intake_by_partition": {"julia:mandelbrot": 25,
                                                             "mandelbrot": 15},
              "colorized": 12, "gate_admitted": 8, "release_eligible": 5, "released": 2}
    RR.write_run(SITE, RR.run_row(run_id="run_a", site=SITE, out_dir="scratch/x",
                                  ledgers=["data/discovery/x/outcome_ledger.jsonl"],
                                  counts=counts, floors={"pool_wallpaper": 0.75}))
    path, n_total, n_new = RR.write_run(SITE, RR.run_row(
        run_id="run_b", site=SITE, out_dir="scratch/y", ledgers=[], counts=counts, floors={}))
    assert (n_total, n_new) == (2, 1)
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["run_id"] for r in rows} == {"run_a", "run_b"}
    assert rows[0]["counts"]["intake_by_partition"]["julia:mandelbrot"] == 25


def test_record_path_is_durable_registered(store, monkeypatch):
    """`durable()` asserts git would keep the path. If the record ever moves somewhere
    gitignored it must fail at the write site, not months later when it is needed."""
    monkeypatch.setattr(paths, "_is_gitignored", lambda _p: True)
    with pytest.raises(paths.DurabilityError):
        RR.record_path(SITE)


def test_real_record_dir_is_not_gitignored():
    """The production path, against the real repo — the registration this task asked for."""
    _clear_ignore_cache()
    p = paths.durable(f"{RR.RECORD_DIR_REL}/emission_diversity_v1.jsonl")
    assert str(p).replace("\\", "/").endswith(
        "data/emission/release_records/emission_diversity_v1.jsonl")


# --------------------------------------------------------------------------- #
# End-to-end over the driver's OWN row builders (tiny population, no rendering) #
# --------------------------------------------------------------------------- #

class _StubPool:
    def __init__(self, rows):
        self.rows = rows


class _Engine:
    """Minimal stand-in that borrows the real EmissionDiversity methods, so the row shape and
    the funnel counts are the driver's, not a copy that can drift from it."""

    def __init__(self, tmp_path):
        from tools.emission.build_emission_diversity_v1 import EmissionDiversity as ED
        self.ED = ED
        self.out = tmp_path / "run_smoke"
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledgers = [Path("data/discovery/fake/outcome_ledger.jsonl")]
        self.floor, self.mining_floor = 0.75, 0.25
        self.release_floor, self.mining_release_floor = 0.90, 0.50
        self.release_n = 1
        self.rows = [{"id": "L0", "family": "julia:mandelbrot"},
                     {"id": "L1", "family": "julia:mandelbrot"},
                     {"id": "L2", "family": "phoenix"}]
        self.by_id = {r["id"]: dict(r, outcome_cx="0.1", outcome_cy="0.2", outcome_fw="0.3",
                                    julia_c_re="-0.4", julia_c_im="0.6") for r in self.rows}
        # the driver's cell axis (`descriptor.cell_partition`), which the funnel counts by.
        # L2 is an axis-free phoenix row, i.e. the pinned Ushiki point -> `phoenix:classic`.
        self.partition_of = {"L0": "julia:mandelbrot", "L1": "julia:mandelbrot",
                             "L2": "phoenix:classic"}
        self.pool = _StubPool([
            dict(id="e0", location_id="L0", type="julia:mandelbrot", morph_cluster="j#0",
                 render_style="smooth", palette="viridis", p_ge3=0.95, passed=True,
                 head="wallpaper", floor=0.75, error=None),
            # `passed` means SCORED since 2026-08-09 (the pool floor annotates), so a 0.60 row
            # below the retired 0.75 is pooled like any other.
            dict(id="e1", location_id="L1", type="julia:mandelbrot", morph_cluster="j#1",
                 render_style="smooth", palette="magma", p_ge3=0.60, passed=True,
                 head="wallpaper", floor=0.75, error=None),
            dict(id="e2", location_id="L2", type="phoenix", morph_cluster="p#0",
                 render_style="smooth", palette="cubehelix", p_ge3=None, passed=False,
                 head=None, floor=0.75, error="RuntimeError('render died')"),
        ])

    # borrow the real implementations
    RECORD_SITE = SITE

    def _run_id(self):
        return self.ED._run_id(self)

    def _record_location(self, lid):
        return self.ED._record_location(self, lid)

    _record_join_key = staticmethod(
        lambda r: "|".join(str(x) for x in (r["location_id"], r["render_style"], r["palette"])))

    def release_floor_for(self, style):
        return self.ED.release_floor_for(self, style)

    def floor_for(self, style):
        return self.ED.floor_for(self, style)

    def above_pool_floor(self, r):
        return self.ED.above_pool_floor(self, r)

    def would_pass_release_floor(self, r):
        return self.ED.would_pass_release_floor(self, r)

    def release_eligible(self):
        return self.ED.release_eligible(self)

    def _gate_decision_rows(self):
        return self.ED._gate_decision_rows(self)

    def _release_decision_rows(self, selected):
        return self.ED._release_decision_rows(self, selected)

    def _record_counts(self, g, r, s):
        return self.ED._record_counts(self, g, r, s)

    def write_release_record(self, selected):
        return self.ED.write_release_record(self, selected)


def test_driver_end_to_end_records_gate_and_release(store, monkeypatch):
    import tools.emission.build_emission_diversity_v1 as drv
    monkeypatch.setattr(drv, "ROOT", store)
    eng = _Engine(store)
    selected = [{"_rec": {"id": "e0"}}]
    path, n_total, n_new, runs_path = eng.write_release_record(selected)

    rows = RR.read_decisions(SITE)
    gate = {r["join_key"]: r for r in rows if r["stage"] == RR.STAGE_GATE}
    rel = {r["join_key"]: r for r in rows if r["stage"] == RR.STAGE_RELEASE}

    assert len(gate) == 3, "every colorized candidate is gated, including the crashed one"
    assert gate["L0|smooth|viridis"]["decision"] == "admitted"
    # SCORED IS ADMITTED since 2026-08-09: the 0.60 row is below the retired 0.75 pool floor
    # and is pooled anyway, with the retired cut's verdict recorded beside it.
    assert gate["L1|smooth|magma"]["decision"] == "admitted"
    assert gate["L1|smooth|magma"]["would_pass_floor"] is False
    assert gate["L0|smooth|viridis"]["would_pass_floor"] is True
    # a render error is a decision with a REASON and NO score — not a zero, which would be
    # indistinguishable from a genuinely bad wallpaper.
    assert gate["L2|smooth|cubehelix"]["decision"] == "rejected"
    assert gate["L2|smooth|cubehelix"]["score"] is None
    assert "render_error" in gate["L2|smooth|cubehelix"]["reason"]
    # ...and it gets NO floor verdict. `False` there would say "the old cut would have removed
    # this", which is a different claim from "there was nothing to compare".
    assert gate["L2|smooth|cubehelix"]["would_pass_floor"] is None

    # both SCORED rows are release-eligible and recorded; only L0 was selected, and the
    # passed-over row carries the retired floor's verdict so the old cut stays inspectable.
    assert sorted(rel) == ["L0|smooth|viridis", "L1|smooth|magma"]
    assert rel["L0|smooth|viridis"]["decision"] == "selected"
    assert rel["L0|smooth|viridis"]["would_pass_floor"] is True
    assert rel["L1|smooth|magma"]["decision"] == "not_selected"
    assert rel["L1|smooth|magma"]["would_pass_floor"] is False     # 0.60 < the retired 0.90
    assert all(r["run_id"] == "run_smoke" for r in rows)
    assert all(r["partition"] and r["morph_cluster"] for r in rows)

    runs = [json.loads(l) for l in Path(runs_path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    c = runs[0]["counts"]
    assert c["intake_admitted"] == 3
    # the funnel is keyed by the CELL PARTITION, so the classic-phoenix row is counted under
    # `phoenix:classic` and not folded into `phoenix` — the split it is owed a share for.
    assert c["intake_by_partition"] == {"julia:mandelbrot": 2, "phoenix:classic": 1}
    assert (c["colorized"], c["gate_admitted"], c["release_eligible"], c["released"]) == (3, 2, 2, 1)
    assert c["gate_admitted_by_partition"] == {"julia:mandelbrot": 2}

    # ...and a second run of the same shape accumulates rather than replacing it
    eng2 = _Engine(store)
    eng2.out = store / "run_smoke_2"
    eng2.out.mkdir(parents=True, exist_ok=True)
    _, n_total2, n_new2, _ = eng2.write_release_record(selected)
    assert n_new2 == 5 and n_total2 == n_total + 5
    assert {r["run_id"] for r in RR.read_decisions(SITE)} == {"run_smoke", "run_smoke_2"}
