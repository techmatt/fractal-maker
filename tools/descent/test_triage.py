"""Acceptance for the minibrot triage wall (`prompts/minibrot_triage_wall.md`).

The four things the wall is only worth building if they hold:

  * **Enumeration is resumable and extends without re-running.** A second run picks up
    at the stored cursor, the already-enumerated prefix is byte-identical, and no atom
    is ever enumerated twice (ids are content hashes of the sector-canonical dedup key).
  * **Verdicts survive a restart.** They are appended and fsync'd before the UI
    advances, and re-reading from disk reproduces them; the latest event per atom wins,
    so "back" and re-verdict are appends, never edits.
  * **No metadata is visible during triage.** Nothing the browser can see carries
    period, degree, `|A|`, or any score — including the tile id itself, which is an
    opaque content hash precisely so it cannot smuggle one. If the numbers were on
    screen the verdicts would be partly a response to them and the covariate join
    afterwards would mean nothing.
  * **The `A`-feasibility cut is the ONLY cut**, and the pool is NOT period-stratified
    (the failure the wall exists to correct).

Store paths are redirected to a temp dir, so the tests never touch the committed
`data/descent_harness/triage/` records.

Run:  uv run python -m pytest tools/descent/test_triage.py -q
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))

import artifacts as A          # noqa: E402
import triage_store as ts      # noqa: E402
import build_triage_pool as btp  # noqa: E402
import build_triage_selection as bts  # noqa: E402
brs = btp.brs                  # the roster builder, for its band/cap policy

# Every covariate name the pool records. NONE of these may reach the browser.
METADATA_KEYS = {
    "degree", "period", "family", "cx", "cy", "fw", "window_scale", "abs_A",
    "log10_abs_A", "arg_A", "rotation_ambiguity_rad", "size", "required_dps",
    "f64_margin_deploy_decades", "f64_margin_field_decades", "newton_res_log10",
    "min_abs_z_pre", "min_abs_z_index", "dedup_key", "provenance",
    "score", "p_good", "split",
}


def _redirect(tmp: Path):
    """Point the durable triage store (and the out-of-tree thumb family) at `tmp`."""
    tmp = Path(tmp)
    ts.TRIAGE_DIR = tmp / "data" / "descent_harness" / "triage"
    ts.POOL = ts.TRIAGE_DIR / "pool.jsonl"
    ts.VERDICTS = ts.TRIAGE_DIR / "verdicts.jsonl"
    ts.ENUM_STATE = ts.TRIAGE_DIR / "enum_state.json"
    ts.REFERENCES = ts.TRIAGE_DIR / "references.json"
    ts.NEIGHBORS = ts.TRIAGE_DIR / "neighbors.json"
    os.environ["FRACTAL_ARTIFACTS_ROOT"] = str(tmp)
    ts.ensure_dirs()


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def test_schedule_covers_every_seed_exactly_once():
    """The cursor indexes a bijection over (degree, seed) — so advancing it never
    re-runs a seed and never skips one."""
    perm = btp._permutation(btp.seeds_per_degree())
    seen = [btp.schedule_task(i, perm) for i in range(btp.total_tasks())]
    assert len(set(seen)) == btp.total_tasks() == len(seen)
    assert {d for d, _ in seen} == set(btp.DEGREES)
    # degrees round-robin, so any prefix carries all four
    assert {d for d, _ in seen[:8]} == set(btp.DEGREES)


def test_schedule_is_deterministic_across_processes():
    """The permutation must not be PYTHONHASHSEED- or clock-dependent: a stored
    cursor from a previous process has to mean the same thing."""
    a = btp._permutation(64).tolist()
    b = btp._permutation(64).tolist()
    assert a == b


# --------------------------------------------------------------------------- #
# enumeration: resumable, extends, never duplicates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", [2])
def test_enumeration_resumes_and_extends(tmp_path, chunk):
    _redirect(tmp_path)
    btp.main(["--target", "10", "--chunk", str(chunk), "--max-tasks", "2"])
    first = ts.load_pool()
    state1 = ts.load_state()
    assert first, "first run produced no atoms"
    assert state1["cursor"] == 2

    # a second run CONTINUES from the cursor rather than re-deriving the first atoms
    btp.main(["--target", "10000", "--chunk", str(chunk), "--max-tasks", "4"])
    second = ts.load_pool()
    state2 = ts.load_state()
    assert state2["cursor"] == 6, "cursor did not advance past the first run's tasks"
    assert len(second) > len(first), "the extension added nothing"
    # the already-enumerated prefix is untouched, row for row
    assert second[:len(first)] == first
    # and no atom is enumerated twice, ever
    ids = [a["id"] for a in second]
    assert len(ids) == len(set(ids))
    # every row carries a task_ordinal below the cursor that produced it
    assert all(a["provenance"]["task_ordinal"] < state2["cursor"] for a in second)


def test_interrupted_chunk_replay_is_a_noop(tmp_path):
    """Ids are content hashes, so re-running a chunk whose atoms landed but whose
    cursor did not cannot double-insert."""
    _redirect(tmp_path)
    btp.main(["--target", "10", "--chunk", "2", "--max-tasks", "2"])
    pool = ts.load_pool()
    assert ts.append_atoms(pool) == 0, "replaying the same atoms inserted duplicates"
    assert ts.load_pool() == pool


@pytest.fixture(scope="module")
def full_pool_dir(tmp_path_factory):
    """ONE full-size enumeration (~120 atoms, ~14s at 4.6 tasks/s), shared by the two
    tests below that need a pool big enough to be statistically meaningful.

    Enumeration is the dominant cost in this file and it is pure — the pool is a
    deterministic function of the cursor schedule — so the two consumers can read the
    same one. They re-`_redirect` at it on entry, which is what keeps them independent
    of the function-scoped builds elsewhere in the file (`_redirect` repoints module
    globals with no teardown, so LAST writer wins and every test must set its own)."""
    d = tmp_path_factory.mktemp("triage_full_pool")
    _redirect(d)
    btp.main(["--target", "120", "--chunk", "8"])
    return d


def test_only_cut_is_a_feasibility_and_no_period_stratification(full_pool_dir):
    _redirect(full_pool_dir)
    pool = ts.load_pool()
    assert len(pool) >= 120
    # the ONLY cut: every atom clears the deploy f64 wall with >= 1 decade
    assert all(a["f64_margin_deploy_decades"] >= btp.MARGIN_MIN_DECADES for a in pool)
    # NOT period-stratified, two independent ways the roster's draw cannot be:
    #   (a) some (degree, band) cell blows past the roster's per-cell cap, and
    from collections import Counter
    cells = Counter((a["degree"], brs.band_of(a["period"])) for a in pool
                    if brs.band_of(a["period"]) is not None)
    assert max(cells.values()) > brs.TARGET_PER_CELL, f"pool looks capped: {cells}"
    #   (b) periods outside the roster's bands (3..15) survive at all.
    assert any(a["period"] > brs.BANDS[-1][1] for a in pool), "period range looks banded"
    # all four degrees are represented, in whatever proportion the scan produced
    assert {a["degree"] for a in pool} == set(btp.DEGREES)


def test_pool_rows_carry_every_covariate(tmp_path):
    _redirect(tmp_path)
    btp.main(["--target", "6", "--chunk", "2", "--max-tasks", "2"])
    a = ts.load_pool()[0]
    for k in ("degree", "period", "abs_A", "log10_abs_A", "arg_A",
              "f64_margin_deploy_decades", "f64_margin_field_decades",
              "size", "window_scale", "dedup_key", "min_abs_z_pre", "min_abs_z_index"):
        assert k in a, f"covariate {k} not recorded"
    for k in ("run_id", "task_ordinal", "seed_index", "seed_re", "seed_im"):
        assert k in a["provenance"], f"provenance {k} not recorded"
    # the derived neighbour sidecar is regenerated, never stored per-row (it goes stale)
    assert "n_comparable_within_20w" not in a
    nb = ts.load_neighbors()["neighbors"]
    assert set(nb) == {r["id"] for r in ts.load_pool()}


# --------------------------------------------------------------------------- #
# verdicts: durable, latest-wins, reversible
# --------------------------------------------------------------------------- #
def test_verdicts_survive_a_restart(tmp_path):
    _redirect(tmp_path)
    ts.append_verdict("mtaaaaaaaaaaaa", "accept", session_id="s1")
    ts.append_verdict("mtbbbbbbbbbbbb", "reject", session_id="s1")
    # "restart": nothing cached — read the durable log back from disk
    assert ts.load_verdicts() == {"mtaaaaaaaaaaaa": "accept", "mtbbbbbbbbbbbb": "reject"}
    # re-verdict is an APPEND; latest wins, history is preserved
    ts.append_verdict("mtaaaaaaaaaaaa", "reject", session_id="s2")
    assert ts.load_verdicts()["mtaaaaaaaaaaaa"] == "reject"
    assert len(ts.load_verdict_events()) == 3
    # clearing is reversible and removes the atom from the collapsed view
    ts.append_verdict("mtaaaaaaaaaaaa", None)
    assert "mtaaaaaaaaaaaa" not in ts.load_verdicts()
    with pytest.raises(ValueError):
        ts.append_verdict("mtcccccccccccc", "maybe")


# --------------------------------------------------------------------------- #
# the no-metadata invariant
# --------------------------------------------------------------------------- #
def test_atom_ids_are_opaque(tmp_path):
    """A tile id must not encode degree or period — the roster's `d2_p03_001` style
    would put both on screen the moment an id is rendered or inspected."""
    _redirect(tmp_path)
    btp.main(["--target", "6", "--chunk", "2", "--max-tasks", "2"])
    for a in ts.load_pool():
        assert re.fullmatch(r"mt[0-9a-f]{12}", a["id"]), a["id"]
        assert ts.atom_id(a["degree"], a["dedup_key"]) == a["id"]   # content-derived


def _assert_no_metadata(obj, where="payload"):
    if isinstance(obj, dict):
        leaked = METADATA_KEYS & set(obj)
        assert not leaked, f"{where} leaks metadata: {sorted(leaked)}"
        for k, v in obj.items():
            _assert_no_metadata(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_metadata(v, f"{where}[{i}]")


def test_browser_payloads_carry_no_metadata(tmp_path):
    _redirect(tmp_path)
    btp.main(["--target", "6", "--chunk", "2", "--max-tasks", "2"])
    btp.build_references()
    import triage_app as ta
    ta.load_all()
    client = ta.app.test_client()

    state = client.post("/api/state").get_json()
    page = client.post("/api/page", json={"page": 0}).get_json()
    _assert_no_metadata(state, "/api/state")
    _assert_no_metadata(page, "/api/page")
    # a tile row is EXACTLY id + verdict, nothing more
    assert page["tiles"] and all(set(t) == {"id", "verdict"} for t in page["tiles"])
    # and the raw JSON text does not contain a covariate name anywhere
    blob = json.dumps(state) + json.dumps(page)
    for k in ("period", "degree", "log10_abs_A", "abs_A", "dedup_key", "score"):
        assert k not in blob, f"/api payload text mentions {k}"


def test_page_size_bounds_what_loads(full_pool_dir):
    """A 1000-tile pass must not load at once."""
    _redirect(full_pool_dir)
    btp.build_references()
    import triage_app as ta
    ta.load_all()
    ta.PAGE_SIZE = 25
    client = ta.app.test_client()
    p0 = client.post("/api/page", json={"page": 0}).get_json()
    assert len(p0["tiles"]) == 25
    assert p0["progress"]["pages"] >= 4
    p1 = client.post("/api/page", json={"page": 1}).get_json()
    assert {t["id"] for t in p0["tiles"]}.isdisjoint({t["id"] for t in p1["tiles"]})
    ta.PAGE_SIZE = 60


def test_verdict_route_records_durably_and_resumes(tmp_path):
    _redirect(tmp_path)
    btp.main(["--target", "10", "--chunk", "4"])
    btp.build_references()
    import triage_app as ta
    ta.load_all()
    client = ta.app.test_client()

    first = ta.POOL_ORDER[0]
    r = client.post("/api/verdict", json={"atom_id": first, "verdict": "accept"})
    assert r.status_code == 200 and r.get_json()["progress"]["accepted"] == 1
    assert ts.load_verdicts()[first] == "accept"          # on disk, not just in memory
    assert ta.first_untriaged() == 1                      # the wall reopens past it
    assert client.post("/api/verdict",
                       json={"atom_id": "mtdeadbeefdead", "verdict": "accept"}).status_code == 404
    assert client.post("/api/verdict",
                       json={"atom_id": first, "verdict": "great"}).status_code == 400


# --------------------------------------------------------------------------- #
# selection hand-off
# --------------------------------------------------------------------------- #
def test_accepted_set_becomes_a_selection(tmp_path):
    _redirect(tmp_path)
    btp.main(["--target", "10", "--chunk", "4"])
    pool = ts.load_pool()
    ts.append_verdict(pool[0]["id"], "accept")
    ts.append_verdict(pool[1]["id"], "reject")
    ts.append_verdict(pool[2]["id"], "accept")
    doc = bts.build()
    assert doc["n_accepted"] == 2 and doc["n_rejected"] == 1
    assert {a["id"] for a in doc["atoms"]} == {pool[0]["id"], pool[2]["id"]}
    # schema the descent harness reads (store.load_selection -> app.ATOMS)
    for a in doc["atoms"]:
        assert set(a) >= {"id", "degree", "period", "split", "family", "cx", "cy", "fw"}
        assert a["split"] in ("train", "eval")
    # the split is growth-stable: it depends on the id alone
    assert bts.split_for(pool[0]["id"]) == doc["atoms"][0]["split"] or True
    assert all(bts.split_for(a["id"]) == a["split"] for a in doc["atoms"])


def test_existing_selection_is_left_in_place():
    """The 40-atom study set must remain reproducible — the triage set is additional."""
    assert (REPO_ROOT / "data" / "descent_harness" / "selection.json").exists()
    assert bts.OUT.name == "selection_triage.json"


# --------------------------------------------------------------------------- #
# storage classes
# --------------------------------------------------------------------------- #
def test_thumbs_relocate_out_of_tree_records_stay_in():
    root = A.artifacts_root()
    rel = ts.thumb_rel("mt0123456789ab", 4)
    assert rel == "data/descent_harness/thumbs/mt0123456789ab__x4.png"
    assert A.is_relocated(rel) and A.resolve(rel) == root / rel
    for keep in ("data/descent_harness/triage/pool.jsonl",
                 "data/descent_harness/triage/verdicts.jsonl",
                 "data/descent_harness/selection_triage.json"):
        assert not A.is_relocated(keep)
        assert A.resolve(keep) == A.REPO_ROOT / keep


def test_durable_records_are_not_gitignored():
    """`data/*` is ignored by default; the triage records must be re-included by path,
    or every verdict Matt records would be silently discarded."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import paths
    for rel in ("data/descent_harness/triage/pool.jsonl",
                "data/descent_harness/triage/verdicts.jsonl",
                "data/descent_harness/triage/enum_state.json",
                "data/descent_harness/selection_triage.json"):
        paths.durable(rel)          # raises DurabilityError if git would drop it


# --------------------------------------------------------------------------- #
# read-time dedup of real-axis Newton-noise copies (the consume-side fix)
# --------------------------------------------------------------------------- #
import mpmath as mp                       # noqa: E402
import deep_center_finder as dcf          # noqa: E402


@pytest.fixture
def restore_triage_store():
    """`_redirect` permanently repoints `triage_store` module globals at a tmp dir with no
    teardown; snapshot + restore them so a redirected test cannot leak a deleted path into
    a later test file (e.g. `test_sources`' reference loader reads `ts.REFERENCES`)."""
    attrs = ("TRIAGE_DIR", "POOL", "VERDICTS", "ENUM_STATE", "REFERENCES", "NEIGHBORS")
    saved = {a: getattr(ts, a) for a in attrs}
    saved_env = os.environ.get("FRACTAL_ARTIFACTS_ROOT")
    try:
        yield
    finally:
        for a, v in saved.items():
            setattr(ts, a, v)
        if saved_env is None:
            os.environ.pop("FRACTAL_ARTIFACTS_ROOT", None)
        else:
            os.environ["FRACTAL_ARTIFACTS_ROOT"] = saved_env


def _axis_dup_rows():
    """Three sibling rows for ONE real-axis nucleus, each with a different Newton-noise
    imaginary tail (so distinct stored dedup_key -> distinct `mt…` id), plus one
    unrelated off-axis atom. Built through the real key/id functions, no `snap`, exactly
    as the pre-fix write path produced them."""
    mp.mp.dps = 80
    cx = "-1.3107026413368328835635707974121808"
    noise = ["0.0", "2.4439490907996837456415835145708942e-150",
             "4.1676726552845202814550876517573002e-162"]
    rows = []
    for cy in noise:
        key = ",".join(dcf.nucleus_dedup_key(mp.mpc(mp.mpf(cx), mp.mpf(cy)), 2, ts.DEDUP_DPS))
        rows.append({"id": ts.atom_id(2, key), "degree": 2, "period": 4,
                     "family": "mandelbrot", "cx": cx, "cy": cy, "fw": "1e-3",
                     "window_scale": "2.5e-4", "dedup_key": key,
                     "f64_margin_deploy_decades": 3.0, "f64_margin_field_decades": 4.0})
    off = "0.35277690561250861882966838891343329"
    offy = "-0.35267856477628614081660023405873145"
    key = ",".join(dcf.nucleus_dedup_key(mp.mpc(mp.mpf(off), mp.mpf(offy)), 2, ts.DEDUP_DPS))
    rows.append({"id": ts.atom_id(2, key), "degree": 2, "period": 5,
                 "family": "mandelbrot", "cx": off, "cy": offy, "fw": "1e-3",
                 "window_scale": "2.5e-4", "dedup_key": key,
                 "f64_margin_deploy_decades": 3.0, "f64_margin_field_decades": 4.0})
    return rows


def test_collapse_population_keeps_first_and_counts_dups():
    rows = _axis_dup_rows()
    kept, dropped, id_map = dcf.collapse_population(rows, dps=ts.DEDUP_DPS)
    assert len(kept) == 2 and len(dropped) == 2      # 3 axis siblings -> 1, plus the off-axis
    assert kept[0]["id"] == rows[0]["id"]            # first-row-wins
    assert set(id_map) == {rows[1]["id"], rows[2]["id"]}
    assert set(id_map.values()) == {rows[0]["id"]}   # both point at the survivor


def test_load_pool_canonical_collapses_and_carries_a_verdict(tmp_path, restore_triage_store):
    _redirect(tmp_path)
    rows = _axis_dup_rows()
    ts.append_atoms(rows)
    # a verdict on a row that the collapse DROPS must survive on the kept survivor
    ts.append_verdict(rows[1]["id"], "accept")
    pool, id_map, conflicts = ts.load_pool_canonical(ts.load_verdicts())
    assert conflicts == []
    assert len(pool) == 2                             # collapsed
    v = ts.verdict_for_canonical(ts.load_verdicts(), id_map)
    assert v[rows[0]["id"]] == "accept"              # carried from the dropped sibling
    # and the selection builder sees exactly one accepted atom, not three
    doc = bts.build()
    assert doc["n_accepted"] == 1 and doc["pool_size"] == 2


def test_load_pool_canonical_leaves_conflicting_verdicts_alone(tmp_path, restore_triage_store):
    _redirect(tmp_path)
    rows = _axis_dup_rows()
    ts.append_atoms(rows)
    ts.append_verdict(rows[0]["id"], "accept")
    ts.append_verdict(rows[1]["id"], "reject")        # conflict on the same atom
    pool, id_map, conflicts = ts.load_pool_canonical(ts.load_verdicts())
    assert len(conflicts) == 1
    assert conflicts[0]["ids"] == {rows[0]["id"]: "accept", rows[1]["id"]: "reject",
                                   rows[2]["id"]: None}
    # left uncollapsed: all three siblings still present, nothing remapped
    ids = {a["id"] for a in pool}
    assert {rows[0]["id"], rows[1]["id"], rows[2]["id"]} <= ids
    assert id_map == {}


def test_snapped_key_leaves_off_axis_atoms_alone():
    mp.mp.dps = 80
    k_raw = ",".join(dcf.nucleus_dedup_key(
        mp.mpc(mp.mpf("-0.12256117"), mp.mpf("0.74486177")), 2, ts.DEDUP_DPS))
    k_snap = dcf.snapped_dedup_key("-0.12256117", "0.74486177", 2, ts.DEDUP_DPS)
    assert k_raw == k_snap                            # a genuine imag part is never snapped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
