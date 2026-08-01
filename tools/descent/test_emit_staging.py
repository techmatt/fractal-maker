"""An approval must survive a `scratch/` wipe between judging and emitting.

The failure this file exists to stop, reproduced 2026-07-31 against the pre-fix code:
`/quality` renders the crop pair into `scratch/descent_harness/quality/`, the human
judges it, and `/emit` **copied** those two files to durable storage *before* writing
the verdict row. A `scratch/` wipe in between made `shutil.copyfile` raise
`FileNotFoundError` inside the view → an unhandled **500**, and `emits.jsonl` was
never written: the approval was gone. `scratch/` is ephemeral by construction, so a
verdict that only exists there is not a verdict.

The fix has two halves, and this file brackets both:

  * the **verdict** is the artifact nothing can regenerate, so it is written durably
    **first**, before anything that can fail;
  * the staged crops are a **cache**. The session holds the render *parameters* (the
    canonical/vivid render blocks), not paths, so a cache miss rebuilds the judged
    image from the record — byte-for-byte, because `render_corpus_crop` reads every
    pixel-affecting input off the block (the round-trip property `test_descent_smoke`
    already asserts).

Three cases, bracketing the fix on both sides:

  1. cache PRESENT — emit uses the staged bytes.
  2. cache ABSENT  — the wipe. The verdict survives, and the rebuilt crop is
     byte-identical to what was judged.
  3. cache ABSENT and unrebuildable — must fail LOUDLY with a named error and write
     no crop, rather than over-correcting into silently emitting a different image.
     The verdict still survives.

Run:  uv run python -m pytest tools/descent/test_emit_staging.py -q
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BIN = REPO_ROOT / "target" / "release" / "fractal-generator.exe"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
import store          # noqa: E402
import app as dh      # noqa: E402

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="release binary not built")


def _redirect_store(tmp: Path):
    """Same redirection test_descent_smoke uses: records, scratch and the out-of-tree
    artifacts root all land under `tmp`, so the committed store is never touched."""
    tmp = Path(tmp)
    store.DH_DIR = tmp / "data" / "descent_harness"
    store.EMITS = store.DH_DIR / "emits.jsonl"
    store.VBAD = store.DH_DIR / "verified_bad.json"
    store.SCRATCH_DIR = tmp / "scratch" / "descent_harness"
    store.THUMBS_DIR = store.SCRATCH_DIR / "thumbs"
    store.QUALITY_DIR = store.SCRATCH_DIR / "quality"
    os.environ["FRACTAL_ARTIFACTS_ROOT"] = str(tmp)
    store._ensure_dirs()


def _stage(tmp_path):
    """Drive open → one box descent → quality, and return the client, the atom id and
    the bytes of the two staged crops as the human judged them."""
    _redirect_store(tmp_path)
    dh.SESSIONS.clear()
    client = dh.app.test_client()
    atom_id = next(a["id"] for a in store.load_selection() if a["degree"] == 2)

    assert client.post("/open", json={"atom_id": atom_id}).status_code == 200
    r = client.post("/nav_box", json={"atom_id": atom_id,
                                      "down_px": 360, "down_py": 202, "cur_px": 460})
    assert r.status_code == 200, r.get_json()
    view_id = r.get_json()["current_view"]

    r = client.post("/quality", json={"atom_id": atom_id})
    assert r.status_code == 200 and r.get_json()["quality_ready"] is True

    canon_staged, vivid_staged = store.quality_scratch_paths(atom_id, view_id)
    assert canon_staged.exists() and vivid_staged.exists()
    judged = (canon_staged.read_bytes(), vivid_staged.read_bytes())
    return client, atom_id, view_id, judged


def _emits():
    return [json.loads(l) for l in store.EMITS.read_text().splitlines() if l.strip()] \
        if store.EMITS.exists() else []


# --------------------------------------------------------------------------- #
# 1 · cache PRESENT — the ordinary path still emits exactly what was judged
# --------------------------------------------------------------------------- #
def test_emit_uses_the_staged_cache_and_matches_the_judged_bytes(tmp_path):
    client, atom_id, _view, (canon_judged, vivid_judged) = _stage(tmp_path)

    r = client.post("/emit", json={"atom_id": atom_id, "class": 4})
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    emit_id = r.get_json()["emitted"]

    rows = _emits()
    assert len(rows) == 1 and rows[0]["emit_id"] == emit_id
    assert store.canonical_crop_path(emit_id).read_bytes() == canon_judged
    assert store.vivid_crop_path(emit_id).read_bytes() == vivid_judged


# --------------------------------------------------------------------------- #
# 2 · cache ABSENT — the guard. RED against the pre-fix code (unhandled 500,
#     zero rows in emits.jsonl).
# --------------------------------------------------------------------------- #
def test_emit_survives_a_scratch_wipe_between_judging_and_emitting(tmp_path):
    client, atom_id, _view, (canon_judged, vivid_judged) = _stage(tmp_path)

    shutil.rmtree(store.SCRATCH_DIR)              # `rm -r scratch/*`
    assert not store.SCRATCH_DIR.exists()

    r = client.post("/emit", json={"atom_id": atom_id, "class": 3})
    # never an unhandled 500: the response is JSON either way
    assert r.is_json, r.get_data(as_text=True)[:300]
    assert r.status_code == 200, r.get_json()

    # the approval survives
    rows = _emits()
    assert len(rows) == 1, "the verdict must be durable regardless of the crop cache"
    rec = rows[0]
    assert rec["class"] == 3 and rec["emit_id"] == r.get_json()["emitted"]

    # and the reconstruction reproduces the image that was judged
    assert store.canonical_crop_path(rec["emit_id"]).read_bytes() == canon_judged, \
        "rebuilt canonical crop is not the crop that was judged"
    assert store.vivid_crop_path(rec["emit_id"]).read_bytes() == vivid_judged, \
        "rebuilt vivid crop is not the crop that was judged"


def test_session_holds_render_parameters_not_scratch_paths(tmp_path):
    """(b) of the fix: a path is not a record. The staged file is derivable from
    (atom_id, view_id); what the session must carry is the parameter set the judged
    image is a pure function of."""
    _client, atom_id, view_id, _judged = _stage(tmp_path)
    qq = dh.SESSIONS[atom_id]["quality"][view_id]
    assert set(qq) == {"canonical_block", "vivid_block"}, (
        f"session quality entry carries more than the render parameters: {sorted(qq)}")
    for blk in (qq["canonical_block"], qq["vivid_block"]):
        # every pixel-affecting input `render_corpus_crop` reads, present and frozen
        for k in ("cx", "cy", "fw", "maxiter", "palette", "width", "height",
                  "ss", "filter", "fractal_type"):
            assert blk.get(k) is not None, f"{k} missing from the recorded block"
    # maxiter is FROZEN in the block, not re-derived at rebuild time — a later policy
    # change must not silently make the rebuild a different image.
    assert isinstance(qq["canonical_block"]["maxiter"], int)


# --------------------------------------------------------------------------- #
# 3 · the other bracket — do not over-correct into silently emitting something else
# --------------------------------------------------------------------------- #
def test_unrebuildable_emit_fails_loudly_and_still_keeps_the_verdict(tmp_path):
    client, atom_id, view_id, _judged = _stage(tmp_path)

    # corrupt the recorded parameters so the crop genuinely cannot be reconstructed
    # (`location.render_one_flags` refuses an unknown family)
    dh.SESSIONS[atom_id]["quality"][view_id]["canonical_block"]["fractal_type"] = "nonesuch"
    shutil.rmtree(store.SCRATCH_DIR)

    r = client.post("/emit", json={"atom_id": atom_id, "class": 4})
    assert r.is_json, "must be a NAMED error, not an unhandled 500 HTML page"
    assert r.status_code != 200
    err = r.get_json()["error"]
    assert "canonical" in err and "nonesuch" in err, err

    rows = _emits()
    assert len(rows) == 1, "the verdict is durable even when the crop cannot be built"
    emit_id = rows[0]["emit_id"]
    assert not store.canonical_crop_path(emit_id).exists(), \
        "a failed rebuild must leave no crop rather than emit a different image"


if __name__ == "__main__":
    import tempfile
    for fn in (test_emit_uses_the_staged_cache_and_matches_the_judged_bytes,
               test_emit_survives_a_scratch_wipe_between_judging_and_emitting,
               test_session_holds_render_parameters_not_scratch_paths,
               test_unrebuildable_emit_fails_loudly_and_still_keeps_the_verdict):
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
            print(f"{fn.__name__}: PASS")
