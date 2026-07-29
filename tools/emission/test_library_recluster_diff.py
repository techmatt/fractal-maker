#!/usr/bin/env python
"""Tests for the retroactive-merge diff.

The real library snapshot was wiped with `scratch/emission/`, so the 1268 -> 1258 number
cannot be recomputed today. This pins the diff's LOGIC against a synthetic two-pass library
built to hold exactly the seam the real one has: two intake passes clustered separately,
one look present in both, plus one look duplicated WITHIN a pass (an ordering artifact,
which must be reported as a merge that does NOT cross the seam).

  uv run pytest tools/emission/test_library_recluster_diff.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.emission import descriptor as D                    # noqa: E402
from tools.emission import library_recluster_diff as RD       # noqa: E402

DIM = 768


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _base(seed):
    return _unit(np.random.default_rng(seed).normal(size=DIM))


def _near(base, cos, seed):
    r = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    perp = _unit(r - np.dot(r, base) * base)
    return _unit(cos * base + np.sqrt(max(0.0, 1.0 - cos * cos)) * perp)


def _snapshot(tmp: Path):
    """A two-pass library with one CROSS-PASS duplicate look and one WITHIN-pass one.

    pass A (library_intake_2): a0 (look L), a1 (look M)
    pass B (campaign1, `c1__` prefixed): b0 (look L again -> cross-seam dup), b1 (look N)
    Each pass numbered its clusters from 0 independently; the union offsets B past A, which
    is exactly what stage_first_release does.
    """
    L, M, N = _base(1), _base(2), _base(3)
    embs = {"a0": L, "a1": M,
            "c1__b0": _near(L, 0.99, seed=10),      # same look as a0, different pass
            "c1__b1": N}
    committed = {"a0": "mandelbrot#0", "a1": "mandelbrot#1",
                 "c1__b0": "mandelbrot#2", "c1__b1": "mandelbrot#3"}
    (tmp / D.LIBRARY_INTAKE_NAME).write_text(
        json.dumps({"cluster_tags": committed, "n_admitted": len(embs)}), encoding="utf-8")
    D._save_embs(embs, tmp / D.LIBRARY_EMBS_NAME)
    return committed


def test_the_diff_finds_the_cross_pass_merge_and_names_the_seam():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _snapshot(tmp)
        d = RD.diff(tmp)
    assert d["n_committed_clusters"] == 4
    assert d["n_one_pass_clusters"] == 3        # a0 and c1__b0 collapse
    assert d["delta"] == -1
    assert d["n_merges"] == 1
    m = d["merges"][0]
    assert m["survivor"] == "mandelbrot#0" and m["absorbed"] == "mandelbrot#2"
    assert m["crosses_source_boundary"] is True
    assert m["absorbed_sources"] == ["campaign1"]
    assert m["survivor_sources"] == ["library_intake_2"]
    assert m["max_cos"] > D.NEAR_DUP_THRESHOLD
    assert not d["splits"]


def test_a_within_pass_duplicate_is_reported_as_NOT_crossing_the_seam():
    """A merge that does not cross a source boundary is an ordering artifact of single-pass
    incremental clustering, not a second bug — the report must distinguish them."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        L = _base(1)
        embs = {"a0": L, "a1": _near(L, 0.99, seed=11)}
        # committed as two clusters within ONE pass (possible when a third, since-removed
        # member founded the second): both ids carry the same source.
        committed = {"a0": "mandelbrot#0", "a1": "mandelbrot#1"}
        (tmp / D.LIBRARY_INTAKE_NAME).write_text(
            json.dumps({"cluster_tags": committed}), encoding="utf-8")
        D._save_embs(embs, tmp / D.LIBRARY_EMBS_NAME)
        d = RD.diff(tmp)
    assert d["n_merges"] == 1
    assert d["merges"][0]["crosses_source_boundary"] is False
    assert d["n_merges_crossing_source"] == 0


def test_source_tags_in_the_snapshot_win_over_the_id_prefix_heuristic():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        L = _base(1)
        embs = {"x": L, "y": _near(L, 0.99, seed=12)}
        (tmp / D.LIBRARY_INTAKE_NAME).write_text(json.dumps({
            "cluster_tags": {"x": "mandelbrot#0", "y": "mandelbrot#1"},
            "source_tags": {"x": "c2_breadth", "y": "phoenix_grid"}}), encoding="utf-8")
        D._save_embs(embs, tmp / D.LIBRARY_EMBS_NAME)
        d = RD.diff(tmp)
    m = d["merges"][0]
    assert m["survivor_sources"] == ["c2_breadth"]
    assert m["absorbed_sources"] == ["phoenix_grid"]
    assert m["crosses_source_boundary"] is True


def test_a_clean_library_reports_zero_merges():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        embs = {f"r{i}": _base(100 + i) for i in range(5)}
        (tmp / D.LIBRARY_INTAKE_NAME).write_text(json.dumps(
            {"cluster_tags": {f"r{i}": f"mandelbrot#{i}" for i in range(5)}}), encoding="utf-8")
        D._save_embs(embs, tmp / D.LIBRARY_EMBS_NAME)
        d = RD.diff(tmp)
    assert d["delta"] == 0 and d["n_merges"] == 0 and not d["splits"]
    assert "0 crossing a source-batch boundary" in RD.render(d)


def test_a_missing_snapshot_explains_why_the_number_is_not_derivable():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(SystemExit) as e:
            RD.diff(Path(td))
    assert "no library snapshot" in str(e.value)
    assert "re-rendering and re-embedding" in str(e.value)


def test_the_report_never_claims_to_have_applied_anything():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _snapshot(tmp)
        md = RD.render(RD.diff(tmp))
    assert "**Not applied.**" in md
    assert "1 crossing a source-batch boundary" in md
