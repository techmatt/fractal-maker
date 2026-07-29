#!/usr/bin/env python
"""Tests for library-seeded intake clustering (`descriptor.assign_morph_clusters`).

The bug: medoids started EMPTY on every call, so an intake batch was deduplicated only
against itself and never against the released library. The measured cost is small today
(10 clusters, 0.8%) only because the library was assembled as two big jointly-clustered
passes and therefore holds exactly ONE un-deduped seam — the error is proportional to the
number of intake seams, and a campaign adds seams.

Five tests, bracketing both sides of the fix:
  1. a new batch's known near-duplicate of a library row JOINS that row's cluster;
  2. genuinely novel material still FOUNDS new clusters;
  3. no existing library row changes cluster under either of the above;
  4. RED-before-the-fix — unseeded, case 1 founds a fresh parallel cluster (the bug);
  5. zero-change — re-running intake over an already-ingested batch reproduces its
     existing assignments exactly.

Pure numpy: hand-built unit embeddings, no torch, no CLIP, no library artifact on disk
(the on-disk loader is exercised against a synthetic snapshot).

  uv run pytest tools/emission/test_descriptor_library_seed.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.emission import descriptor as D  # noqa: E402

DIM = 768


# --------------------------------------------------------------------------- #
# helpers — embeddings with a controlled cosine to a base direction
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _base(seed: int):
    return _unit(np.random.default_rng(seed).normal(size=DIM))


def _near(base, cos: float, seed: int):
    """A unit vector at exactly `cos` from `base` (orthogonal perturbation)."""
    r = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    perp = _unit(r - np.dot(r, base) * base)
    return _unit(cos * base + np.sqrt(max(0.0, 1.0 - cos * cos)) * perp)


def _row(i, family="mandelbrot"):
    return {"id": i, "family": family}


# the library: 3 mandelbrot clusters + 1 multibrot3 cluster, keyed with an OFFSET and a
# GAP, because the real union offsets campaign1's tags past library_intake_2's per-family
# count — a seeded key space is not guaranteed to be 0..N-1.
LIB_BASES = {("mandelbrot", 5): 11, ("mandelbrot", 6): 12, ("mandelbrot", 9): 13,
             ("multibrot3", 2): 21}


def _library():
    lib = {}
    for (fam, k), sd in LIB_BASES.items():
        lib.setdefault(fam, []).append((k, _base(sd)))
    return {f: sorted(v) for f, v in lib.items()}


def _library_prior():
    """The library's own {location_id: tag} assignment (one member per cluster is enough
    for the never-moved guard)."""
    return {f"lib_{fam}_{k}": f"{fam}#{k}" for (fam, k) in LIB_BASES}


# --------------------------------------------------------------------------- #
# 1. a near-duplicate of a library row joins THAT row's cluster
# --------------------------------------------------------------------------- #
def test_a_near_duplicate_of_a_library_row_joins_that_rows_cluster():
    lib = _library()
    dup = _near(_base(12), 0.99, seed=101)          # 0.99 > 0.974 -> same look as mandelbrot#6
    rows = [_row("new_dup")]
    tags = D.assign_morph_clusters(rows, {"new_dup": dup}, library=lib)
    assert tags["new_dup"] == "mandelbrot#6"


def test_the_join_respects_the_threshold_not_merely_the_nearest_medoid():
    """Just below the knee must still found a new cluster — seeding must not turn the
    strict 0.974 near-dup test into a nearest-medoid assignment."""
    lib = _library()
    just_under = _near(_base(12), 0.97, seed=102)
    tags = D.assign_morph_clusters([_row("x")], {"x": just_under}, library=lib)
    assert tags["x"] == "mandelbrot#10"             # past max seeded key 9, not #6


def test_seeding_is_within_type_a_library_look_of_another_type_is_not_a_match():
    """multibrot3's medoid must not capture a mandelbrot row even at cosine 1.0 — the
    dedup convention is within-family."""
    lib = _library()
    same_as_mb3 = _base(21)
    tags = D.assign_morph_clusters([_row("m", family="mandelbrot")], {"m": same_as_mb3},
                                   library=lib)
    assert tags["m"] == "mandelbrot#10"             # new mandelbrot cluster, not multibrot3#2


# --------------------------------------------------------------------------- #
# 2. genuinely novel material founds new clusters
# --------------------------------------------------------------------------- #
def test_novel_material_founds_new_clusters_past_the_library_key_space():
    lib = _library()
    embs = {f"n{i}": _base(500 + i) for i in range(4)}     # 4 unrelated directions
    rows = [_row(i) for i in embs]
    tags = D.assign_morph_clusters(rows, embs, library=lib)
    assert sorted(tags.values()) == ["mandelbrot#10", "mandelbrot#11",
                                     "mandelbrot#12", "mandelbrot#13"]
    # no new tag collides with a seeded one, even though the seeded space has a gap at 7-8
    assert not (set(tags.values()) & set(_library_prior().values()))


def test_novel_rows_that_duplicate_EACH_OTHER_still_collapse():
    """Seeding must not disable in-batch dedup: two novel rows that near-duplicate one
    another share one new cluster."""
    lib = _library()
    b = _base(601)
    embs = {"a": b, "b": _near(b, 0.995, seed=602), "c": _base(603)}
    tags = D.assign_morph_clusters([_row(i) for i in ("a", "b", "c")], embs, library=lib)
    assert tags["a"] == tags["b"] == "mandelbrot#10"
    assert tags["c"] == "mandelbrot#11"


def test_a_type_absent_from_the_library_starts_at_zero():
    lib = _library()
    tags = D.assign_morph_clusters([_row("p", family="phoenix")], {"p": _base(700)},
                                   library=lib)
    assert tags["p"] == "phoenix#0"


# --------------------------------------------------------------------------- #
# 3. no existing library row changes cluster
# --------------------------------------------------------------------------- #
def test_no_library_row_changes_cluster_under_a_mixed_batch():
    """A batch of joins + novel material + in-batch duplicates: every library location keeps
    its tag, and the guard agrees."""
    lib, prior = _library(), _library_prior()
    embs = {"j1": _near(_base(11), 0.99, seed=801),      # joins mandelbrot#5
            "j2": _near(_base(13), 0.98, seed=802),      # joins mandelbrot#9
            "n1": _base(803), "n2": _base(804)}
    tags = D.assign_morph_clusters([_row(i) for i in embs], embs, library=lib)
    D.verify_library_unmoved(prior, tags)                 # must not raise
    assert tags["j1"] == "mandelbrot#5" and tags["j2"] == "mandelbrot#9"
    # the library's own ids are untouched: they are not in the returned map at all
    assert not (set(tags) & set(prior))


def test_a_joining_row_does_not_move_the_seeded_medoid():
    """The medoid is the FOUNDER's embedding and is never updated. If a join re-pointed the
    medoid, a second row near the JOINER (but far from the library founder) would be pulled
    into the library cluster."""
    lib = _library()
    base = _base(11)
    joiner = _near(base, 0.9741, seed=901)                # just inside the knee
    drifted = _near(joiner, 0.9741, seed=902)             # near the joiner, far from `base`
    assert D._cos(drifted, base) < D.NEAR_DUP_THRESHOLD   # precondition of the test
    tags = D.assign_morph_clusters([_row("j"), _row("d")], {"j": joiner, "d": drifted},
                                   library=lib)
    assert tags["j"] == "mandelbrot#5"
    assert tags["d"] == "mandelbrot#10"                   # founds its own, medoid unmoved


def test_the_guard_raises_when_a_library_row_would_move():
    prior = {"loc_a": "mandelbrot#5"}
    with pytest.raises(D.LibraryRowMoved) as e:
        D.verify_library_unmoved(prior, {"loc_a": "mandelbrot#42"})
    assert "mandelbrot#5 -> mandelbrot#42" in str(e.value)


def test_the_guard_passes_a_disjoint_batch():
    D.verify_library_unmoved({"loc_a": "mandelbrot#5"}, {"new": "mandelbrot#10"})


# --------------------------------------------------------------------------- #
# 4. RED before the fix — unseeded, the near-duplicate founds a parallel cluster
# --------------------------------------------------------------------------- #
def test_RED_without_the_library_seed_the_near_duplicate_founds_a_fresh_cluster():
    """The old behaviour, reproduced exactly by omitting `library`: the SAME embedding that
    joins `mandelbrot#6` when seeded instead founds `mandelbrot#0` — a second cluster for a
    look the library already holds. That duplicate is the per-seam error the fix removes."""
    dup = _near(_base(12), 0.99, seed=101)                # identical input to test 1
    unseeded = D.assign_morph_clusters([_row("new_dup")], {"new_dup": dup})
    assert unseeded["new_dup"] == "mandelbrot#0"          # BUG: parallel cluster
    seeded = D.assign_morph_clusters([_row("new_dup")], {"new_dup": dup}, library=_library())
    assert seeded["new_dup"] == "mandelbrot#6"            # FIXED: joins the library's
    assert unseeded["new_dup"] != seeded["new_dup"]


def test_RED_the_seam_error_grows_with_the_number_of_intakes():
    """Three sequential single-row intakes of the SAME look: unseeded each founds its own
    cluster (3 duplicates of one library look); seeded, all three join it."""
    lib = _library()
    looks = [_near(_base(12), 0.99, seed=1000 + i) for i in range(3)]
    unseeded = {D.assign_morph_clusters([_row("r")], {"r": e})["r"] for e in looks}
    seeded = {D.assign_morph_clusters([_row("r")], {"r": e}, library=lib)["r"] for e in looks}
    assert unseeded == {"mandelbrot#0"}          # 3 separate intakes -> 3 new clusters
    assert seeded == {"mandelbrot#6"}            # all three collapse onto the library's


# --------------------------------------------------------------------------- #
# 5. zero-change — re-ingesting an already-ingested batch reproduces its assignments
# --------------------------------------------------------------------------- #
def test_zero_change_re_running_intake_over_an_ingested_batch_reproduces_it():
    """The load-bearing idempotence property. Cluster a batch cold (no library), publish the
    result AS the library, then re-run intake over the identical batch seeded from it: every
    row must land back on its own cluster and no new cluster may be founded."""
    embs = {"a": _base(11), "b": _near(_base(11), 0.99, seed=1101),
            "c": _base(12), "d": _base(13), "e": _base(21)}
    fams = {"a": "mandelbrot", "b": "mandelbrot", "c": "mandelbrot", "d": "mandelbrot",
            "e": "multibrot3"}
    rows = [_row(i, fams[i]) for i in embs]
    first = D.assign_morph_clusters(rows, embs)
    # publish: the library's medoids are the FOUNDERS, exactly as `library_medoids` recovers
    # them from a snapshot's cluster_tags in stable order.
    lib, founder = {}, {}
    for i in embs:
        founder.setdefault(first[i], i)
    for tag, fid in founder.items():
        f, _, k = tag.rpartition("#")
        lib.setdefault(f, []).append((int(k), embs[fid]))
    lib = {f: sorted(v) for f, v in lib.items()}

    second = D.assign_morph_clusters(rows, embs, library=lib)
    assert second == first                              # byte-for-byte the same assignment
    D.verify_library_unmoved(first, second)
    assert len(set(second.values())) == len(set(first.values()))   # nothing founded


def test_zero_change_holds_through_the_on_disk_snapshot_loader():
    """Same property, but the library is loaded from a real snapshot pair on disk
    (intake.json + morph_embs.npz), so the founder recovery and key parsing are covered."""
    import tempfile
    embs = {"a": _base(11), "b": _near(_base(11), 0.99, seed=1201), "c": _base(12),
            "e": _base(21)}
    fams = {"a": "mandelbrot", "b": "mandelbrot", "c": "mandelbrot", "e": "multibrot3"}
    rows = [_row(i, fams[i]) for i in embs]
    first = D.assign_morph_clusters(rows, embs)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / D.LIBRARY_INTAKE_NAME).write_text(
            json.dumps({"cluster_tags": first, "n_admitted": len(rows)}), encoding="utf-8")
        D._save_embs(embs, d / D.LIBRARY_EMBS_NAME)
        lib, prior, note = D.load_library_seed(d)
        assert "library seed:" in note and prior == first
        assert sum(len(v) for v in lib.values()) == len(set(first.values()))
        second = D.assign_morph_clusters(rows, embs, library=lib)

    assert second == first
    D.verify_library_unmoved(prior, second)


def test_an_absent_snapshot_is_reported_not_silently_empty():
    """A missing library must produce a LOUD note, because a silently-empty seed is
    indistinguishable from a working one and re-opens the seam."""
    lib, prior, note = D.load_library_seed(ROOT / "scratch" / "_no_such_library_dir")
    assert lib == {} and prior == {}
    assert note.startswith("LIBRARY SEED ABSENT")
    assert "deduplicated against ITSELF ONLY" in note


# --------------------------------------------------------------------------- #
# the unseeded path is unchanged (no behaviour drift for a first-ever intake)
# --------------------------------------------------------------------------- #
def test_unseeded_clustering_is_bit_identical_to_the_pre_fix_behaviour():
    """`library=None` must reproduce the old dense-from-zero keys exactly, so the historical
    tags (and everything keyed off them) are unaffected by the fix."""
    embs = {f"r{i}": _base(1300 + i) for i in range(5)}
    embs["r5"] = _near(embs["r2"], 0.99, seed=1399)
    rows = [_row(i) for i in embs]
    tags = D.assign_morph_clusters(rows, embs)
    assert tags == {"r0": "mandelbrot#0", "r1": "mandelbrot#1", "r2": "mandelbrot#2",
                    "r3": "mandelbrot#3", "r4": "mandelbrot#4", "r5": "mandelbrot#2"}
