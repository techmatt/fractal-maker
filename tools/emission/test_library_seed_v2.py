"""The relit library look-seed: the floor, the look rule, and the durability split.

  uv run pytest tools/emission/test_library_seed_v2.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "emission"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import library_seed_v2 as ls2                    # noqa: E402
import paths                                     # noqa: E402


def _queue(tmp_path, rows) -> Path:
    p = tmp_path / "q.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _row(**kw):
    base = dict(batch="B", image_id="i0", partition="julia:mandelbrot", human=3,
                first_of_look=True)
    base.update(kw)
    return base


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A corpus the render-block join can actually reach, so the build is exercised through
    its real join rather than through an injected dict."""
    import corpus_common as cc
    bd = tmp_path / "batches" / "B"
    bd.mkdir(parents=True)
    rows = [dict(image_id=f"i{i}", render=dict(cx="0", cy="0", fw="1.0", maxiter=500,
                                               fractal_type="julia", c_re="0.1", c_im="0.2"),
                 label=dict(score=None))
            for i in range(6)]
    (bd / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                     encoding="utf-8")
    monkeypatch.setattr(cc, "batch_dir", lambda b: str(tmp_path / "batches" / b))
    return tmp_path


# =========================================================================== #
# the floor
# =========================================================================== #
def test_the_floor_is_a_HUMAN_label_and_no_decode_is_consulted(tmp_path, corpus):
    """The q4_harvest floor-admit precedent with a stronger floor. A row Matt scored 3 or 4
    is in whatever any head thinks of it; a row he scored 2 is out however high its decode."""
    q = _queue(tmp_path, [_row(image_id="i0", human=4),
                          _row(image_id="i1", human=3),
                          _row(image_id="i2", human=2)])
    snap = ls2.build(q, write=False)
    assert snap["n_looks"] == 2 and snap["floor_rejected"] == 1
    assert set(snap["entries"]) == {"i0", "i1"}
    assert snap["admission"]["decode_consulted"] is False
    assert snap["admission"]["floor"] == 3


def test_only_the_first_row_of_a_look_seeds(tmp_path, corpus):
    """The queue already clustered at cos 0.974 — the SAME knee the tally uses. Re-clustering
    would be a second opinion on a question already answered with the same metric."""
    q = _queue(tmp_path, [_row(image_id="i0", first_of_look=True),
                          _row(image_id="i1", first_of_look=False),
                          _row(image_id="i2", first_of_look=True)])
    snap = ls2.build(q, write=False)
    assert set(snap["entries"]) == {"i0", "i2"}
    assert "not recomputed" in snap["admission"]["looks_from"]


def test_every_row_is_tagged_with_its_human_provenance(tmp_path, corpus):
    q = _queue(tmp_path, [_row(image_id="i0", human=4)])
    snap = ls2.build(q, write=False)
    e = snap["entries"]["i0"]
    assert e["mix_source"] == "human_q3plus" and e["human"] == 4
    assert snap["mix_source"] == "human_q3plus"


def test_cluster_tags_are_partition_hash_index_which_is_what_the_loader_parses(tmp_path,
                                                                               corpus):
    """`load_library_seed_embeddings` splits the tag on the LAST '#' to recover the
    partition. A julia partition's own colon must survive that."""
    import deficit_scheduler as dsched
    q = _queue(tmp_path, [_row(image_id="i0", partition="julia:mandelbrot"),
                          _row(image_id="i1", partition="julia:mandelbrot"),
                          _row(image_id="i2", partition="phoenix")])
    snap = ls2.build(q, write=False)
    tags = sorted(snap["medoid_id"])
    assert tags == ["julia:mandelbrot#0", "julia:mandelbrot#1", "phoenix#0"]
    assert all(t.rsplit("#", 1)[0] in ("julia:mandelbrot", "phoenix") for t in tags)
    assert dsched.EMB_DIM == 768


def test_a_queue_row_with_no_corpus_row_is_a_HARD_failure(tmp_path, corpus):
    """A look with no coordinates cannot be re-embedded, so a partial join must not become a
    smaller seed — that is the shape that makes a seed silently non-reproducible."""
    q = _queue(tmp_path, [_row(image_id="not_in_corpus")])
    with pytest.raises(SystemExit, match="no corpus row"):
        ls2.build(q, write=False)


# =========================================================================== #
# the loader round-trip — the thing the guard actually reads
# =========================================================================== #
def test_the_snapshot_plus_embeddings_are_what_require_library_seed_reads(tmp_path, corpus):
    """PRESENCE-FROM-DISK. The snapshot and the `.npy` files must satisfy the real guard,
    not a mock of it: the seed's whole job is to make `require_library_seed` return looks."""
    import deficit_scheduler as dsched
    q = _queue(tmp_path, [_row(image_id="i0", partition="julia:mandelbrot"),
                          _row(image_id="i1", partition="phoenix")])
    snap = ls2.build(q, write=False)
    ip = tmp_path / "intake.json"
    ip.write_text(json.dumps(snap), encoding="utf-8")
    ed = tmp_path / "embs"
    ed.mkdir()
    for lid in snap["entries"]:
        np.save(ed / f"{lid}.npy", np.random.default_rng(0).normal(size=768).astype("f4"))
    rec = dsched.require_library_seed(intake_path=ip, emb_dir=ed)
    assert rec["status"] == "seeded" and rec["library_looks"] == 2
    assert set(rec["library_partitions"]) == {"julia:mandelbrot", "phoenix"}


def test_a_snapshot_with_no_embeddings_still_fails_closed(tmp_path, corpus):
    """The half-built state. A snapshot on disk with an empty embedding dir must abort a
    scheduler run, not seed it with nothing — which is exactly what a `{}` return would do."""
    import deficit_scheduler as dsched
    q = _queue(tmp_path, [_row(image_id="i0")])
    ip = tmp_path / "intake.json"
    ip.write_text(json.dumps(ls2.build(q, write=False)), encoding="utf-8")
    (tmp_path / "embs").mkdir()
    with pytest.raises(dsched.UnseededRunError):
        dsched.require_library_seed(intake_path=ip, emb_dir=tmp_path / "embs")


# =========================================================================== #
# durability
# =========================================================================== #
def test_the_snapshot_is_durable_and_the_embeddings_are_bulk():
    """The tree's own split (`test_intake_durable.py` pins it for the campaign-1 pair): the
    snapshot is git-tracked, the regenerable per-look vectors are not. This one is SAFE to
    split that way because `embed` rebuilds the vectors from the snapshot's own render
    blocks — the property the campaign-1 seed did not have, which is why it is dark."""
    assert str(paths.durable(ls2.INTAKE_REL)).replace("\\", "/").endswith(
        "data/emission/library_seed_v2/intake.json")
    assert "scratch" in ls2.EMB_REL
    snap = json.loads(ls2.INTAKE_JSON.read_text(encoding="utf-8"))
    for e in snap["entries"].values():
        assert e["render"].get("cx") is not None and e["render"].get("fw") is not None


def test_the_write_site_refuses_a_gitignored_home(monkeypatch):
    monkeypatch.setattr(paths, "_is_gitignored", lambda _p: True)
    if hasattr(paths, "_IGNORE_CACHE"):
        paths._IGNORE_CACHE.clear()
    with pytest.raises(paths.DurabilityError):
        paths.durable(ls2.INTAKE_REL, mkparents=True)
    if hasattr(paths, "_IGNORE_CACHE"):
        paths._IGNORE_CACHE.clear()


# =========================================================================== #
# the committed seed
# =========================================================================== #
def test_the_committed_snapshot_holds_the_queues_looks():
    """Not absence-tolerant: the snapshot is the durable half and a missing one means the
    seed is dark again. 322 queue rows collapse to 168 looks."""
    assert ls2.INTAKE_JSON.exists(), (
        f"{ls2.INTAKE_JSON} missing — rebuild with "
        f"`uv run python tools/emission/library_seed_v2.py build`")
    snap = json.loads(ls2.INTAKE_JSON.read_text(encoding="utf-8"))
    assert snap["n_queue_rows"] == 322 and snap["n_looks"] == 168
    assert len(snap["medoid_id"]) == len(snap["entries"]) == 168
    assert sum(snap["by_partition"].values()) == 168


# =========================================================================== #
# the seed-source registry — a documented resolution order, not a fallback
# =========================================================================== #
def test_the_registry_resolves_by_existence_first_match_wins(tmp_path, monkeypatch):
    import deficit_scheduler as dsched
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    monkeypatch.setattr(dsched, "SEED_SOURCES",
                        (("A", a, tmp_path / "ae"), ("B", b, tmp_path / "be")))
    b.write_text("{}", encoding="utf-8")
    assert dsched.resolve_seed_source()[0] == "B"
    a.write_text("{}", encoding="utf-8")
    assert dsched.resolve_seed_source()[0] == "A", "first match must win once it exists"


def test_with_nothing_present_the_registry_names_the_PRIMARY_artifact(tmp_path, monkeypatch):
    """The error message has to send a reader to rebuild the right thing. Naming the last
    source tried would send them to rebuild the fallback."""
    import deficit_scheduler as dsched
    monkeypatch.setattr(dsched, "SEED_SOURCES",
                        (("A", tmp_path / "a.json", tmp_path / "ae"),
                         ("B", tmp_path / "b.json", tmp_path / "be")))
    assert dsched.resolve_seed_source()[0] == "A"
    with pytest.raises(dsched.UnseededRunError, match="a.json"):
        dsched.require_library_seed()


def test_the_record_says_WHICH_source_seeded_it(tmp_path, monkeypatch):
    """That is what makes an ordered registry different from a silent fallback: a reader of
    a run summary months later can tell which artifact the deficits were measured against."""
    import deficit_scheduler as dsched
    ip, ed = tmp_path / "b.json", tmp_path / "be"
    ed.mkdir()
    ip.write_text(json.dumps(dict(medoid_id={"phoenix#0": "L1"})), encoding="utf-8")
    np.save(ed / "L1.npy", np.ones(768, dtype="f4"))
    monkeypatch.setattr(dsched, "SEED_SOURCES",
                        (("A", tmp_path / "a.json", tmp_path / "ae"), ("B", ip, ed)))
    rec = dsched.require_library_seed()
    assert rec["resolved_from"] == "B" and rec["library_looks"] == 1
    assert [r["name"] for r in rec["registry"]] == ["A", "B"]
    assert [r["exists"] for r in rec["registry"]] == [False, True]


def test_an_explicit_path_is_never_mixed_with_a_resolved_one(tmp_path, monkeypatch):
    """Half an explicit pair would pair one source's snapshot with another's vectors — the
    embeddings would silently not join and the seed would read as empty."""
    import deficit_scheduler as dsched
    monkeypatch.setattr(dsched, "SEED_SOURCES", (("A", tmp_path / "a.json", tmp_path / "ae"),))
    ip, ed = dsched.library_seed_paths(intake_path=tmp_path / "mine.json")
    assert ip == tmp_path / "mine.json" and ed == tmp_path / "ae"


def test_the_relit_seed_is_what_the_live_registry_resolves_to_today():
    """The production assertion. campaign1's snapshot is gone, so the registry must resolve
    to the relit seed and it must be non-empty — 168 looks across all nine partitions."""
    import deficit_scheduler as dsched
    name, ip, _ed = dsched.resolve_seed_source()
    assert name == "library_seed_v2" and ip == ls2.INTAKE_JSON
    rec = dsched.require_library_seed(allow_unseeded=True)
    assert rec["status"] == "seeded" and rec["library_looks"] == 168
    assert len(rec["library_partitions"]) == 9
