#!/usr/bin/env python
"""Fail-closed intake guards: the library seed cannot go silently empty, and the relit
seed's rows admit on the HUMAN label they were selected by.

Two failures these pin, both found by the 2026-08-04 stage-2 survey:

  1. `descriptor.load_library_seed` returned `({}, {}, "LIBRARY SEED ABSENT ...")` and both
     intake callers printed the note and carried on. The seed had been dark since the
     `scratch/` wipe, so every intake since deduplicated against ITSELF ONLY while its
     cluster counts went on record as library-wide. A warning that a run does not act on is
     not a guard (`verification_practice.md` §1.4/§2).
  2. `FLOOR_ADMIT_SOURCES` named only `q4_harvest`, so a `human_q3plus` row — a location
     Matt scored 3 or 4 with no decode consulted — would have been gated on
     `decoded_class>=3`, i.e. the head vetoing material it never judged. That is the exact
     inversion the floor rule exists to prevent.
  3. (2026-08-04, §B) The fix for 2 left the head a SECOND veto on the same rows — the
     `p_notbad >= FLOOR_PNOTBAD` badness floor, a v7-era number still being applied on the
     v10 scale. That floor is deleted; a floor-admit row now takes no machine quality cut at
     all, and the tests below pin both halves (bypassed for the tagged source, NOT bypassed
     for anything else).

  uv run pytest tools/emission/test_intake_fail_closed.py -q
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "scoring", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import descriptor as D            # noqa: E402
from tools.emission import campaign1_intake as C1     # noqa: E402
from tools.emission import library_seed_v2 as LS2     # noqa: E402
import corpus_common as cc                            # noqa: E402

DIM = 768


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _emb(seed):
    return _unit(np.random.default_rng(seed).normal(size=DIM))


def _seeded_library(d: Path, embs: dict, tags: dict):
    (d / D.LIBRARY_INTAKE_NAME).write_text(json.dumps({"cluster_tags": tags}),
                                           encoding="utf-8")
    D._save_embs(embs, d / D.LIBRARY_EMBS_NAME)


# --------------------------------------------------------------------------- #
# 1. the intake path itself aborts — bracketed on both sides
# --------------------------------------------------------------------------- #
def test_the_intake_clustering_path_ABORTS_on_a_missing_seed():
    """`campaign1_intake.cluster` is one of the two live intake call sites and is pure
    numpy, so the whole path runs here. Before the fix this returned tags and logged a
    note."""
    rows = [{"id": "n0", "family": "mandelbrot"}]
    with pytest.raises(D.LibrarySeedUnavailable):
        C1.cluster(rows, {"n0": _emb(1)}, library_dir=ROOT / "data" / "emission" / "_absent")


def test_the_intake_clustering_path_ABORTS_on_a_present_but_empty_seed():
    """The harder half: the snapshot EXISTS. An `.exists()` check would pass here and the
    pass would be unseeded anyway (`verification_practice.md` §2)."""
    rows = [{"id": "n0", "family": "mandelbrot"}]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _seeded_library(d, {}, {})
        with pytest.raises(D.LibrarySeedUnavailable):
            C1.cluster(rows, {"n0": _emb(1)}, library_dir=d)


def test_the_intake_clustering_path_SUCCEEDS_and_actually_seeds_when_the_seed_is_there(
        tmp_path, monkeypatch):
    """The over-correction check: the guard must not have turned every intake into an
    abort. A new row that near-duplicates a library look joins the LIBRARY's cluster, which
    is only observable if the seed reached the clustering.

    `C1.OUT` is redirected because `campaign1_intake.log` appends to a progress file under
    it — the success path writes, the two abort paths above never get that far."""
    monkeypatch.setattr(C1, "OUT", tmp_path)
    lib_emb = _emb(7)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _seeded_library(d, {"lib0": lib_emb}, {"lib0": "mandelbrot#4"})
        rows = [{"id": "n0", "family": "mandelbrot"}]
        tags, _medoid_id = C1.cluster(rows, {"n0": lib_emb.copy()}, library_dir=d)
    assert tags == {"n0": "mandelbrot#4"}          # joined, did not found #0


def test_no_call_site_can_swallow_the_abort():
    """A raise a caller catches is the warning it replaced. Mechanical, over the whole tree:
    no `load_library_seed(` call may sit inside a `try:` body."""
    offenders = []
    for py in sorted(ROOT.joinpath("tools").rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:                                   # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if (isinstance(sub, ast.Call) and
                        getattr(sub.func, "attr", getattr(sub.func, "id", None))
                        == "load_library_seed"):
                    offenders.append(f"{py.relative_to(ROOT)}:{sub.lineno}")
    assert not offenders, f"load_library_seed inside a try block: {offenders}"


def test_the_scan_above_is_not_vacuous():
    """§5: a derived set can pass by evaluating empty. Prove the tree actually holds call
    sites for the scan to have looked at."""
    n = 0
    for py in sorted(ROOT.joinpath("tools").rglob("*.py")):
        if py.name.startswith("test_"):
            continue
        if "load_library_seed(" in py.read_text(encoding="utf-8"):
            n += 1
    assert n >= 2, f"expected the two intake call sites, found {n}"


# --------------------------------------------------------------------------- #
# 1b. the never-moved guard actually runs where the seeding happens
# --------------------------------------------------------------------------- #
def test_every_seeded_clustering_call_site_runs_the_never_moved_verifier():
    """`verify_library_unmoved` is the only mechanical statement of "nothing already in the
    library moves", and it protects committed state (cell reachability, per-cell deficits,
    the release record's `morph_cluster` column). A seeded clustering that forgets it would
    rewrite that state silently, so the pairing is checked structurally: any
    `assign_morph_clusters(..., library=...)` must sit in a function that also calls the
    verifier. (An UNseeded call — `library_recluster_diff`'s one-pass diff — has no prior to
    verify against and is correctly not matched.)"""
    seeded, verified, checked = [], [], 0
    for py in sorted(ROOT.joinpath("tools").rglob("*.py")):
        if py.name.startswith("test_"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = [getattr(c.func, "attr", getattr(c.func, "id", None))
                     for c in ast.walk(fn) if isinstance(c, ast.Call)]
            kwargs_seeded = any(
                getattr(c.func, "attr", None) == "assign_morph_clusters"
                and any(k.arg == "library" for k in c.keywords)
                for c in ast.walk(fn) if isinstance(c, ast.Call))
            if kwargs_seeded:
                checked += 1
                where = f"{py.relative_to(ROOT)}:{fn.name}"
                (verified if "verify_library_unmoved" in names else seeded).append(where)
    assert checked >= 2, f"expected the two seeded intake call sites, found {checked}"
    assert not seeded, f"seeded clustering with no verify_library_unmoved: {seeded}"


def test_a_seeded_medoid_is_frozen_and_joiners_do_not_displace_it(tmp_path, monkeypatch):
    """The freeze, observed rather than asserted about: two new rows that each near-dup a
    seeded medoid but NOT each other must both land in the seeded cluster. If a join updated
    the medoid, the second row would be compared against the first and would found its own."""
    monkeypatch.setattr(C1, "OUT", tmp_path)
    med = _emb(101)
    r = np.random.default_rng(202).normal(size=DIM).astype(np.float32)
    perp = _unit(r - np.dot(r, med) * med)
    j1 = _unit(0.98 * med + np.sqrt(1 - 0.98 ** 2) * perp)
    j2 = _unit(0.98 * med - np.sqrt(1 - 0.98 ** 2) * perp)   # cos(j1,j2) ~ 0.92 < 0.974
    assert float(np.dot(j1, j2)) < D.NEAR_DUP_THRESHOLD      # the fixture can fail

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _seeded_library(d, {"lib0": med}, {"lib0": "mandelbrot#9"})
        rows = [{"id": "n1", "family": "mandelbrot"}, {"id": "n2", "family": "mandelbrot"}]
        tags, medoid_id = C1.cluster(rows, {"n1": j1, "n2": j2}, library_dir=d)
    assert tags == {"n1": "mandelbrot#9", "n2": "mandelbrot#9"}
    assert "mandelbrot#9" not in medoid_id      # a joiner never becomes the medoid


def test_the_verifier_raises_rather_than_rewriting():
    """It must be a refusal, not a repair: `LibraryRowMoved` names the moves and returns
    nothing, so no caller can 'handle' it by taking the new tags."""
    with pytest.raises(D.LibraryRowMoved) as ei:
        D.verify_library_unmoved({"x": "mandelbrot#1"}, {"x": "mandelbrot#2"})
    assert "refusing rather than rewriting" in str(ei.value)


# --------------------------------------------------------------------------- #
# 2. the relit seed's rows admit on the HUMAN label, not on a decode
# --------------------------------------------------------------------------- #
def _seed_row(**over):
    """A row shaped like the relit seed's, current-stamped so `is_current_decoded` passes."""
    row = {"id": "s0", "family": "mandelbrot",
           "outcome_cx": "0.0", "outcome_cy": "0.0", "outcome_fw": "1.0",
           "mix_source": LS2.MIX_SOURCE, "scorer_version": cc.active_scorer_version(),
           "decoded_class": 1, "p_notbad": 0.80, "p_good": 0.01,
           "guard_pass": True, "distinct": True}
    row.update(over)
    return row


def test_the_seeds_source_tag_is_registered_as_a_floor_admit_source():
    """Derived from the seed builder's own constant, so a rename there cannot leave this
    passing against a stale literal."""
    assert LS2.MIX_SOURCE in D.FLOOR_ADMIT_SOURCES


def test_a_seed_row_admits_on_the_floor_with_a_class_1_decode():
    """The whole point: `decoded_class` is 1 and the row is still admitted, because the
    selection signal was a human 3/4 and the head never judged it."""
    assert D.admit_quality(_seed_row()) is True
    assert D.admit_quality(_seed_row(decoded_class=None)) is True


def test_a_seed_row_the_head_calls_bad_is_admitted_anyway():
    """The §B bypass. A `human_q3plus` row carries a HUMAN 3 or 4; a machine `p_notbad` of 0
    on such a row is the head disagreeing with Matt, and the head does not get to resolve
    that at intake. The v7-era badness floor that used to cut here was DELETED (2026-08-04),
    not lowered — so there is no threshold left to drift.

    Injection: this is red under the old `p_notbad >= FLOOR_PNOTBAD` branch for every value
    below the floor, which is what it is here to catch coming back."""
    for nb in (0.0, 0.01, 0.49, 0.4999):
        assert D.admit_quality(_seed_row(p_notbad=nb)) is True, nb
    assert D.admit_quality(_seed_row(p_notbad=None)) is True
    # ...and the constant itself is gone, so a re-add cannot be silent.
    assert not hasattr(D, "FLOOR_PNOTBAD"), (
        "descriptor.FLOOR_PNOTBAD is back. A floor-admit source's selection signal is "
        "orthogonal to the head; re-applying the head's badness verdict to it is the veto "
        "the floor-admit rule exists to prevent. If a machine cut is genuinely wanted here, "
        "it belongs in tools/emission/floors.py with a head stamp and a derivation.")


def test_the_source_tag_is_what_switches_the_rule_not_the_row_shape():
    """Same row, same numbers, different `mix_source` -> the q3 gate applies and it is cut.
    Without this the bypass test passes for any row at all — which is exactly the failure
    mode a bypass introduces, so this is the non-vacuity half of §B."""
    assert D.admit_quality(_seed_row(mix_source="steered_frontier")) is False
    assert D.admit_quality(_seed_row(mix_source=None)) is False
    assert D.admit_quality(_seed_row(mix_source="steered_frontier", decoded_class=3)) is True


def test_load_admitted_admits_the_seed_row_end_to_end(tmp_path):
    """Through the real loader, not just the predicate — the guard/distinct/current checks
    are the caller's, and a floor-admit source must clear THOSE like anything else even
    though no machine quality cut applies to it.

    `s1` carries `p_notbad=0.1` and is admitted (it used to be the row the badness floor cut);
    `s2` fails the guard and `s3` is not distinct, and both are still rejected — which is what
    keeps "bypasses the quality cut" from meaning "bypasses everything"."""
    led = tmp_path / "seed_ledger.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in (
        _seed_row(),
        _seed_row(id="s1", p_notbad=0.1),
        _seed_row(id="s2", guard_pass=False),
        _seed_row(id="s3", distinct=False),
        _seed_row(id="s4", scorer_version="v6"),
    )) + "\n", encoding="utf-8")
    got = [r["id"] for r in D.load_admitted(led)]
    assert got == ["s0", "s1"]
