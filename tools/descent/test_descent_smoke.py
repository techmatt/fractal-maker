"""Scripted acceptance for the descent harness.

  * **Two-step descent + emit** drives the Flask routes (open → box → box →
    quality → emit) and asserts the emit record's lineage length, parentage, and a
    **non-zero** solution count.
  * **Round-trip identity** re-renders the canonical crop from the STORED record's
    `render` block through the same sanctioned path and asserts it reproduces the
    saved crop **byte-for-byte** — the check that catches coordinate truncation,
    the failure mode that would quietly ruin the dataset.

The store paths are redirected to a temp dir so the test never touches the
committed `data/descent_harness/` store.

Run:  uv run python -m pytest tools/descent/test_descent_smoke.py -q
"""
import os
import sys
from decimal import Decimal
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BIN = REPO_ROOT / "target" / "release" / "fractal-generator.exe"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
import store          # noqa: E402
from tools.descent import app as dh   # noqa: E402
import corpus_common as cc  # noqa: E402

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="release binary not built")


@pytest.fixture(autouse=True)
def _no_artifacts_root_leak():
    """`_redirect_store` sets FRACTAL_ARTIFACTS_ROOT process-wide with no teardown, so it
    leaked a tmp artifacts root into every later test in the same process.
    `artifacts_root()` is read at CALL time while modules bake `bulk()` paths at IMPORT
    time (`tau_h_rederive.WORK`), so the leak decided that test's verdict by file order.
    Harmless under the serial ordering; `-n 4 --dist loadfile` assigns files to workers
    dynamically and surfaced it."""
    old = os.environ.get("FRACTAL_ARTIFACTS_ROOT")
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("FRACTAL_ARTIFACTS_ROOT", None)
        else:
            os.environ["FRACTAL_ARTIFACTS_ROOT"] = old


def _redirect_store(tmp: Path):
    tmp = Path(tmp)
    store.DH_DIR = tmp / "data" / "descent_harness"      # in-tree records analog
    store.EMITS = store.DH_DIR / "emits.jsonl"
    store.VBAD = store.DH_DIR / "verified_bad.json"
    store.SCRATCH_DIR = tmp / "scratch" / "descent_harness"
    store.THUMBS_DIR = store.SCRATCH_DIR / "thumbs"
    store.QUALITY_DIR = store.SCRATCH_DIR / "quality"
    # crops/vivid resolve OUT of the tree via artifacts.resolve → point the artifacts
    # root at the temp dir so the emit crops land under tmp, not the real sibling.
    os.environ["FRACTAL_ARTIFACTS_ROOT"] = str(tmp)
    store._ensure_dirs()
    # NB: store.REPO_ROOT stays the REAL repo root so rel() keeps resolving the
    # committed palette libraries repo-relative.


def _first_d2():
    return next(a["id"] for a in store.load_selection() if a["degree"] == 2)


def _deepest_d2():
    """The d2 atom already closest to the f64 wall. The wall-guard test walks boxes down
    until the guard refuses, and every rung on the way is a real nav render whose maxiter
    grows with depth — starting from the shallowest atom (what `_first_d2` returns) spends
    three renders getting to the wall instead of one. Same route, same refusal, chosen by
    the property under test rather than by file order."""
    return min((a for a in store.load_selection() if a["degree"] == 2),
               key=lambda a: Decimal(a["fw"]))["id"]


def test_two_step_descent_emit_and_roundtrip(tmp_path, monkeypatch):
    _redirect_store(tmp_path)
    dh.SESSIONS.clear()
    client = dh.app.test_client()

    atom_id = _first_d2()
    r = client.post("/open", json={"atom_id": atom_id})
    assert r.status_code == 200

    # step 1: a box descent (center panel, ~100px horizontal radius)
    r = client.post("/nav_box", json={"atom_id": atom_id,
                                      "down_px": 360, "down_py": 202, "cur_px": 460})
    assert r.status_code == 200, r.get_json()
    v1 = r.get_json()["current_view"]

    # step 2: a second box descent, deeper
    r = client.post("/nav_box", json={"atom_id": atom_id,
                                      "down_px": 360, "down_py": 202, "cur_px": 430})
    assert r.status_code == 200, r.get_json()
    v2 = r.get_json()["current_view"]
    assert v1 != v2

    # quality render (enables emit for this exact view)
    r = client.post("/quality", json={"atom_id": atom_id})
    assert r.status_code == 200
    assert r.get_json()["quality_ready"] is True

    # emit as q4
    r = client.post("/emit", json={"atom_id": atom_id, "class": 4})
    assert r.status_code == 200, r.get_json()
    st = r.get_json()
    assert st.get("emitted")

    # --- lineage / parentage assertions ---
    emits = store.load_emits()
    assert len(emits) >= 1                       # NON-ZERO count, not merely "ran"
    rec = emits[-1]
    lin = rec["lineage"]
    assert len(lin) == 3                         # base + 2 box steps
    assert lin[0]["step_kind"] == "base" and lin[0]["parent_id"] is None
    assert lin[1]["step_kind"] == "box" and lin[1]["parent_id"] == lin[0]["view_id"]
    assert lin[2]["step_kind"] == "box" and lin[2]["parent_id"] == lin[1]["view_id"]
    # scale-invariant deltas are present on the descent steps
    for step in lin[1:]:
        assert step["zoom_factor"] is not None and step["zoom_factor"] > 1.0
        assert step["center_dx_over_parent_fw"] is not None

    # --- round-trip identity: re-render canonical from the stored record ---
    # The record stores a portable repo-relative crop path; resolve it through the
    # artifacts seam (out-of-tree, under the temp artifacts root here).
    assert rec["canonical_crop"] == "data/descent_harness/crops/" + rec["emit_id"] + ".jpg"
    stored_crop = store.resolve(rec["canonical_crop"])
    assert stored_crop.exists()
    palette_source = REPO_ROOT / rec["palette_source"]   # palette lib lives in the real tree
    rerender = tmp_path / "roundtrip_canonical.jpg"
    cc.render_corpus_crop(rec["render"], str(rerender),
                          palette_source=str(palette_source),
                          jpg_quality=rec["jpg_quality"])
    assert rerender.read_bytes() == stored_crop.read_bytes(), \
        "re-render from stored record does not reproduce the emitted crop byte-for-byte"


def test_box_guard_refuses_below_f64_wall(tmp_path):
    _redirect_store(tmp_path)
    dh.SESSIONS.clear()
    client = dh.app.test_client()
    atom_id = _deepest_d2()
    client.post("/open", json={"atom_id": atom_id})
    # a 1-pixel-radius box at a shallow atom is still well above the wall; instead
    # drive the current view very deep by many boxes until the guard fires, bounded.
    refused = False
    for _ in range(60):
        r = client.post("/nav_box", json={"atom_id": atom_id,
                                          "down_px": 360, "down_py": 202, "cur_px": 361})
        if r.status_code == 400:
            assert "f64-wall margin" in r.get_json()["error"]
            refused = True
            break
    assert refused, "expected the f64-wall guard to eventually refuse a sub-wall box"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_two_step_descent_emit_and_roundtrip(Path(td), None)
        print("two-step descent + emit + round-trip: PASS")
    with tempfile.TemporaryDirectory() as td:
        test_box_guard_refuses_below_f64_wall(Path(td))
        print("f64-wall guard: PASS")
