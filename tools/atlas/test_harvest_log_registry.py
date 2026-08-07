#!/usr/bin/env python
"""Guards for `harvest_log_registry` — the discovery that replaced `tau_h_rederive`'s
five-entry hand list of harvest runs.

The defect this replaces was not a wrong number, it was a SILENT population: nine runs
written after the list was typed were never read, and nothing said so. So the guards here
are about what enters and what does not, proved by INJECTION in both directions
(`verification_practice.md` §3) — a discovery test that only ever asserts "the five are
found" passes just as well on a hand list, which is the thing being removed.

Kept light on purpose: `tau_h_rederive` is imported (it is cheap — the torch-side scorer is
built lazily inside `make_scorer`), but nothing here renders or scores.
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

import deficit_scheduler as D                    # noqa: E402
import discovery_sinks                           # noqa: E402
import harvest_log_registry as H                 # noqa: E402
import tau_h_rederive as T                       # noqa: E402
from tools import run_record                     # noqa: E402
from tools import paths as _paths                # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _row(i: int, partition="mandelbrot", geom=True) -> str:
    r = dict(node_id=f"n{i}", batch=i, partition=partition, depth=3,
             cheap_pgood=0.7, canon_pgood=0.8, canon_nb=0.9, admitted=True)
    if geom:
        r.update(cx="-0.5", cy="0.1", fw="0.01")
    return json.dumps(r)


def _plant(run_dir: Path, n=3, **kw) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / H.LOG_NAME
    log.write_text("\n".join(_row(i, **kw) for i in range(n)) + "\n", encoding="utf-8")
    return log


# =========================================================================== #
# 1. The live registry: invariants over the REAL table, not a synthetic one.
# =========================================================================== #
def test_the_live_registry_holds_no_disposable_path():
    """Read `HARVEST_STORES` / `PINNED_RUNS` directly rather than through discovery.

    Discovery skips a store that does not exist, so routing through it would leave a
    dormant `scratch/` entry unchecked until the day something creates it — which is the
    shape that cost both seed sources (`deficit_scheduler.SEED_SOURCES`, same rule, same
    owner)."""
    for name, store in H.HARVEST_STORES:
        H._refuse_disposable(f"store ({name})", store)
        for run in H.PINNED_RUNS:
            H._refuse_disposable(f"run ({run})", store / run)


def test_every_pinned_run_resolves_and_discovery_is_not_vacuous():
    """Derive-and-prove-non-empty (`verification_practice.md` §5). A discovery that returned
    nothing would satisfy every "X is not discovered" assertion below."""
    runs, missing = H.discover_run_dirs()
    assert not missing, f"pinned harvest runs with no log: {missing}"
    assert {r.name for r in runs} >= set(H.PINNED_RUNS)
    assert len(runs) > len(H.PINNED_RUNS), (
        "discovery found nothing beyond the pinned set — either the store moved or this is "
        "the hand list again with extra steps")
    for r in runs:
        # `r.log` is the LOGICAL stream path, which is what readers hand to run_record —
        # NOT a promise that a plain `harvest_log.jsonl` is on disk. A finalized run has
        # only `harvest_log.000.jsonl.gz` (SegmentWriter.finalize), so `Path.exists()` here
        # went red the first time a segmented run finished, on a healthy registry that was
        # already resolving it correctly. Assert the STREAM has rows, which is the property
        # the derivation actually depends on and the one that holds in either layout.
        assert r.log.name == H.LOG_NAME
        assert run_record.exists(r.log), (
            f"{r.name}: no rows under either layout at {r.log} — a discovered run whose "
            f"stream is empty would silently shrink the derivation population")


def test_a_production_run_lands_in_a_registered_store():
    """THE mechanism behind "new runs auto-feed the curves", asserted rather than claimed.

    A production (non-throwaway) run's sinks resolve to `discovery_sinks`'
    `default_discovery_dir`, which IS a registered store — so writing a harvest log at all
    is what registers a run; nobody has to remember to. The same call for a throwaway run
    resolves under `scratch/`, which the registry refuses by class. Both directions, or the
    claim is only half true."""
    live = discovery_sinks.resolve_discovery_dir(ROOT, smoke=False, time_only=False,
                                                 explicit=None)
    assert live in [p.resolve() for _n, p in H.stores()]

    smoke = discovery_sinks.resolve_discovery_dir(ROOT, smoke=True, time_only=False,
                                                  explicit=None)
    with pytest.raises(H.HarvestPathClassError):
        H._refuse_disposable("store (smoke)", smoke)


# =========================================================================== #
# 2. Injection, both directions: registered enters, unregistered does not.
# =========================================================================== #
@pytest.mark.parametrize("rel", ["newrun_20260805", "campaign3/breadth"])
def test_a_newly_registered_run_dir_ENTERS_and_an_unregistered_one_does_NOT(
        tmp_path, monkeypatch, rel):
    """One fixture, two registries. The SAME populated run dir must enter when its parent
    store is registered and stay out when it is not — a one-sided test cannot tell
    discovery from a hand list that happens to agree.

    Parametrized over both run-dir shapes the store actually holds: flat
    (`popquota_v2_20260804`) and legged (`campaign2/breadth`)."""
    monkeypatch.setattr(H, "ROOT", tmp_path)
    registered = tmp_path / "data" / "discovery"
    elsewhere = tmp_path / "data" / "not_a_store"
    _plant(registered / rel, n=3)
    _plant(elsewhere / rel, n=3)

    runs, missing = H.discover_run_dirs(registry=[("discovery", registered)], pinned=())
    assert not missing
    assert [r.name for r in runs] == [rel]
    assert not runs[0].pinned

    # ...and the rows actually reach the derivation's population, tagged with the run.
    rows = T._harvest_rows(runs)
    assert len(rows) == 3 and {r["run"] for r in rows} == {rel}
    assert all(r["key"].startswith(f"h_{rel.replace('/', '_')}_") for r in rows)

    # The unregistered copy is identical in every way except its parent, and it stays out.
    assert not any(r.dir == elsewhere / rel for r in runs)
    only_other, _ = H.discover_run_dirs(registry=[("other", elsewhere)], pinned=())
    assert [r.name for r in only_other] == [rel], (
        "the fixture cannot fail — the unregistered dir was not discoverable at all, so "
        "its absence above proved nothing")


def test_a_run_dir_deeper_than_MAX_RUN_DEPTH_is_not_discovered(tmp_path, monkeypatch):
    """The bound is a decision, not an accident: `data/discovery` also holds per-run
    subtrees (`gather/`, `runs/`) and an unbounded walk would start reading whatever lands
    in them. Proved red by planting one rung too deep."""
    monkeypatch.setattr(H, "ROOT", tmp_path)
    store = tmp_path / "data" / "discovery"
    _plant(store / "a" / "b", n=2)                      # depth 2 — in
    _plant(store / "a" / "b" / "c", n=2)                # depth 3 — out
    runs, _ = H.discover_run_dirs(registry=[("discovery", store)], pinned=())
    assert [r.name for r in runs] == ["a/b"]


@pytest.mark.parametrize("victim", ["store", "run"])
def test_an_INJECTED_disposable_path_is_refused_at_RESOLVE_time(tmp_path, monkeypatch,
                                                                victim):
    """A `scratch/` log is refused for its CLASS, while it still exists and reads fine.

    Made to EXIST and to hold real rows on purpose: a guard that only fires on a missing
    path is a presence check, and the failure being prevented is a healthy disposable path
    that derives a number and is then deleted underneath it."""
    monkeypatch.setattr(H, "ROOT", tmp_path)
    if victim == "store":
        store = tmp_path / "scratch" / "discovery"
        _plant(store / "run_a")
        with pytest.raises(H.HarvestPathClassError, match="scratch"):
            H.discover_run_dirs(registry=[("planted", store)], pinned=())
    else:
        store = tmp_path / "data" / "discovery"
        _plant(store / "scratchpad")                 # a run dir in the disposable class
        with pytest.raises(H.HarvestPathClassError, match="scratchpad"):
            H.discover_run_dirs(registry=[("discovery", store)], pinned=())

    # ...and the same fixture one directory sideways is fine, so the refusal is the class
    # and not the fixture.
    ok = tmp_path / "data" / "discovery2"
    _plant(ok / "run_a")
    runs, _ = H.discover_run_dirs(registry=[("discovery2", ok)], pinned=())
    assert [r.name for r in runs] == ["run_a"]


def test_a_registered_run_with_NO_log_records_its_absence_LOUDLY(tmp_path, monkeypatch):
    """A pinned run whose log has gone is a hard failure naming it, not a silent −1 on the
    population (`verification_practice.md` §2). The total `discover_run_dirs` still returns
    it in `missing` so a reporter can say what is gone without raising."""
    monkeypatch.setattr(H, "ROOT", tmp_path)
    store = tmp_path / "data" / "discovery"
    _plant(store / "present/leg")
    (store / "gone" / "leg").mkdir(parents=True)      # dir exists, log does not

    runs, missing = H.discover_run_dirs(registry=[("discovery", store)],
                                        pinned=("present/leg", "gone/leg"))
    assert [r.name for r in runs] == ["present/leg"]
    assert missing == ["gone/leg"]

    with pytest.raises(H.MissingHarvestLogError, match="gone/leg"):
        H.require_harvest_runs(registry=[("discovery", store)],
                               pinned=("present/leg", "gone/leg"))
    # and it goes green again only when the log comes back, not when the entry is deleted
    _plant(store / "gone/leg")
    assert len(H.require_harvest_runs(registry=[("discovery", store)],
                                      pinned=("present/leg", "gone/leg"))) == 2


# =========================================================================== #
# 3. Equivalence with the hand list it replaced, and the by-construction exclusions.
# =========================================================================== #
HAND_LIST = ["campaign1/breadth", "campaign1/dive", "campaign2/breadth", "campaign2/dive",
             "julia_parent_probe/breadth"]


def _reference_rows():
    """The REPLACED reader, re-implemented here from the run names alone.

    Differential over a frozen literal (`verification_practice.md` §7): the expectation is
    the old code path's actual output, not a number someone typed. It reads the logs
    itself — it does not call anything in `tau_h_rederive`, or it would be asserting
    f(x) == f(x) (§1.10)."""
    out = []
    for run in HAND_LIST:
        p = ROOT / "data/discovery" / run / H.LOG_NAME
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("cx") is None or r.get("fw") is None:
                continue
            out.append((run, r["partition"],
                        f"h_{run.replace('/', '_')}_{r['node_id']}_{r['batch']}"))
    return out


def test_discovery_over_the_PINNED_set_reproduces_the_hand_lists_population():
    """The pinned five are exactly the population `tau_h_base_v10.json` was derived on, so
    discovery restricted to them must reproduce that population row-for-row and key-for-key
    — otherwise the plumbing change moves the adopted number for a reason nobody chose."""
    pinned = [r for r in H.require_harvest_runs() if r.pinned]
    assert [r.name for r in pinned] == HAND_LIST
    got = [(r["run"], r["partition"], r["key"]) for r in T._harvest_rows(pinned)]
    assert got == _reference_rows()
    assert len(got) > 20_000                      # non-vacuity: the pool is not empty


def test_campaign1_is_excluded_from_the_harvest_arm_BY_CONSTRUCTION():
    """campaign1 is DISCOVERED (it is pinned) and still contributes zero rows, because
    every one of its checks predates the `cx/cy/fw` field and is not re-renderable.

    That is the point of asserting it: the exclusion is the geometry filter doing its job
    on a run the derivation genuinely looked at, not a name someone left off a list. If a
    future backfill gives campaign1 geometry, this goes red and the population change is
    a decision instead of a surprise."""
    c1 = [r for r in H.require_harvest_runs() if r.name.startswith("campaign1/")]
    assert len(c1) == 2
    for r in c1:
        assert sum(1 for ln in open(r.log, encoding="utf-8") if ln.strip()) > 5_000
    assert T._harvest_rows(c1) == []


def test_phoenix_rows_are_excluded_from_the_harvest_arm_like_the_walk_arm(tmp_path,
                                                                          monkeypatch):
    """`derive` never cuts phoenix, so a phoenix row would be rendered and scored for
    nothing and would still land in the pooled cross-family figure the artifact reports.
    The walk arm already dropped them; discovery is what first brings them to this arm."""
    monkeypatch.setattr(H, "ROOT", tmp_path)
    store = tmp_path / "data" / "discovery"
    run = store / "mixed"
    run.mkdir(parents=True)
    (run / H.LOG_NAME).write_text(
        "\n".join([_row(0, partition="mandelbrot"), _row(1, partition="phoenix"),
                   _row(2, partition="julia:multibrot3")]) + "\n", encoding="utf-8")
    runs, _ = H.discover_run_dirs(registry=[("discovery", store)], pinned=())
    assert sorted(r["partition"] for r in T._harvest_rows(runs)) == [
        "julia:multibrot3", "mandelbrot"]


# =========================================================================== #
# 4. One owner for the disposable-class rule.
# =========================================================================== #
def test_the_disposable_class_rule_has_exactly_one_owner(monkeypatch):
    """Behavioural single-source proof, stronger than grepping for a second literal: stub
    the shared predicate and BOTH refusals must change together. Two copies of the rule is
    two policies about which paths guarantee deletion, which is how one of them drifts."""
    monkeypatch.setattr(_paths, "disposable_component", lambda p, roots: "planted")
    with pytest.raises(H.HarvestPathClassError, match="planted"):
        H._refuse_disposable("x", ROOT / "data" / "discovery")
    with pytest.raises(D.SeedPathClassError, match="planted"):
        D._refuse_scratch_class("x", ROOT / "data" / "emission")


def test_the_artifact_stamp_records_the_registry_not_just_the_result():
    """"five runs" and "five runs out of a registry that offered five" are different facts,
    and only one of them survives in a JSON artifact by accident."""
    rec = H.registry_record(H.require_harvest_runs())
    assert rec["pinned"] == list(H.PINNED_RUNS)
    assert [s["name"] for s in rec["stores"]] == [n for n, _ in H.HARVEST_STORES]
    assert rec["n_pinned"] == len(H.PINNED_RUNS) and rec["n_unpinned"] > 0
    assert all(d["log"].endswith(H.LOG_NAME) for d in rec["discovered"])
