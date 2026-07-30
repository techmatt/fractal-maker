#!/usr/bin/env python
"""Minibrot descent harness — a browser tool for hand-finding artist-quality
locations near minibrot centers.

Matt drives it; the recorded **descent path is the product** (training data for a
learned "descend to the good window" function), not a side effect. Built as an
extension of `tools/explorer/` and sharing its pixel→plane math + render-one
invocation via `tools/explorer/render_core.py` (the math must not exist twice).

Deliberately shows **no model output** anywhere — no score, no p_good, no period,
no ranking. Degree is fine (visible from symmetry). The whole point is to capture
Matt's judgement uncontaminated by the machine.

Run:  uv run python tools/descent/app.py
"""
from __future__ import annotations

import base64
import os
import shutil
import sys
import threading
import time
from decimal import Decimal, getcontext
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "explorer"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "sourcing"))
sys.path.insert(0, str(HERE))

import render_core as rc              # noqa: E402  (shared coord + render-one)
import corpus_common as cc           # noqa: E402  (sanctioned corpus-crop render)
from deep_center_finder import atom_instrument  # noqa: E402  (f64-wall guard)
import store                          # noqa: E402

getcontext().prec = 60

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
NAV_W, NAV_H, NAV_SS = 720, 405, 1        # navigation viewport (16:9, cheap ss1)
THUMB_W, THUMB_H, THUMB_SS = 240, 135, 1  # library thumbnails (16:9, ss1)
NAV_PALETTE = store.CANONICAL_PALETTE     # neutral, model-blind
BOX_MIN_PX = 5                            # drag below this falls through to a click
IDLE_CUTOFF_S = 120.0                     # gaps longer than this are idle (not active time)
GUARD_W, GUARD_SS = store.CROP_W, store.CROP_SS   # f64-wall guard at quality fidelity (1280 wide)
MAX_ENGINE = 3                            # concurrent render processes (<= CLAUDE.md's 4)

app = Flask(__name__)

_lock = threading.RLock()
_engine_sem = threading.BoundedSemaphore(MAX_ENGINE)

# in-memory, session-scoped per-atom navigation state (the durable product goes to store)
SESSIONS: dict[str, dict] = {}
ATOMS: dict[str, dict] = {a["id"]: a for a in store.load_selection()}


# --------------------------------------------------------------------------- #
# render helpers (all engine subprocesses go through the semaphore)
# --------------------------------------------------------------------------- #
def _b64(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def render_nav(atom, cx, cy, fw) -> str:
    """Cheap ss1 navigation render of one view → PNG data URL."""
    maxiter = rc.auto_maxiter(fw)
    out = store.SCRATCH_DIR / "nav" / f"{atom['id']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = rc.render_one_argv(cx, cy, fw, maxiter, NAV_W, NAV_H, NAV_SS,
                              NAV_PALETTE, rc.CLEAN_COLORMAPS, out,
                              family=atom["family"])
    with _engine_sem:
        rc.run_render_one(argv, out, low_priority=True)
    return _b64(out, "image/png")


def render_thumb(atom) -> Path:
    out = store.thumb_path(atom["id"])
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fw = Decimal(atom["fw"])
    maxiter = rc.auto_maxiter(fw)
    argv = rc.render_one_argv(atom["cx"], atom["cy"], fw, maxiter,
                              THUMB_W, THUMB_H, THUMB_SS,
                              NAV_PALETTE, rc.CLEAN_COLORMAPS, out,
                              family=atom["family"])
    with _engine_sem:
        rc.run_render_one(argv, out, low_priority=True)
    return out


def canonical_block(atom, cx, cy, fw) -> dict:
    """Canonical (model-facing) label-crop render block — DERIVED from the production
    source (store.canonical_render_block → active_ckpt), NOT the explorer nav heuristic.
    `rc.auto_maxiter` is for navigation renders only and never touches this path."""
    return store.canonical_render_block(cx, cy, fw, atom["family"])


def vivid_block(atom, cx, cy, fw) -> dict:
    """Vivid judging companion — identical to the canonical block except palette."""
    return store.vivid_render_block(cx, cy, fw, atom["family"])


def render_quality_crop(block: dict, out: Path, palette_source: Path) -> Path:
    """Render one quality crop through the sanctioned byte-reproducible path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with _engine_sem:
        cc.render_corpus_crop(block, str(out), palette_source=str(palette_source),
                              jpg_quality=store.JPG_QUALITY)
    return out


# --------------------------------------------------------------------------- #
# session / view-tree
# --------------------------------------------------------------------------- #
def _session(atom_id: str) -> dict:
    sess = SESSIONS.get(atom_id)
    if sess is None:
        atom = ATOMS[atom_id]
        inst = atom_instrument(complex(float(atom["cx"]), float(atom["cy"])),
                               atom["period"], atom["degree"])
        base_fw = Decimal(atom["fw"])
        base = {"id": "v0", "parent_id": None, "step_kind": "base",
                "cx": Decimal(atom["cx"]), "cy": Decimal(atom["cy"]), "fw": base_fw}
        sess = {
            "views": {"v0": base},
            "counter": 1,
            "base_id": "v0",
            "current_id": "v0",
            "quality_view_id": None,          # view with a fresh quality render
            "instrument": inst,
            "activity": {"last_unix": None, "active_seconds": 0.0,
                         "box_descents": 0, "base_fw": base_fw, "min_fw": base_fw},
            "touched": False,
        }
        SESSIONS[atom_id] = sess
    return sess


def _new_view(sess, parent, cx, cy, fw, step_kind) -> dict:
    vid = f"v{sess['counter']}"
    sess["counter"] += 1
    node = {"id": vid, "parent_id": parent["id"], "step_kind": step_kind,
            "cx": cx, "cy": cy, "fw": fw}
    sess["views"][vid] = node
    return node


def _touch(sess):
    now = time.time()
    a = sess["activity"]
    last = a["last_unix"]
    if last is not None and (now - last) <= IDLE_CUTOFF_S:
        a["active_seconds"] += now - last
    a["last_unix"] = now
    sess["touched"] = True


def _note_view_depth(sess, fw: Decimal):
    a = sess["activity"]
    if fw < a["min_fw"]:
        a["min_fw"] = fw


def deepest_zoom(sess) -> float:
    a = sess["activity"]
    return float(a["base_fw"] / a["min_fw"]) if a["min_fw"] > 0 else 1.0


# --------------------------------------------------------------------------- #
# lineage
# --------------------------------------------------------------------------- #
def walk_lineage(sess, leaf_id: str) -> list[dict]:
    """Walk parent pointers leaf→base, return base→leaf steps with scale-invariant
    zoom/offset expressed in units of the PARENT view's frame width."""
    chain = []
    vid = leaf_id
    while vid is not None:
        node = sess["views"][vid]
        chain.append(node)
        vid = node["parent_id"]
    chain.reverse()   # base ... leaf
    out = []
    for i, node in enumerate(chain):
        entry = {
            "index": i, "view_id": node["id"], "parent_id": node["parent_id"],
            "step_kind": node["step_kind"],
            "cx": rc.dec_str(node["cx"]), "cy": rc.dec_str(node["cy"]),
            "fw": rc.dec_str(node["fw"]),
            "zoom_factor": None,
            "center_dx_over_parent_fw": None,
            "center_dy_over_parent_fw": None,
        }
        if node["parent_id"] is not None:
            p = sess["views"][node["parent_id"]]
            entry["zoom_factor"] = float(p["fw"] / node["fw"]) if node["fw"] > 0 else None
            entry["center_dx_over_parent_fw"] = float((node["cx"] - p["cx"]) / p["fw"])
            entry["center_dy_over_parent_fw"] = float((node["cy"] - p["cy"]) / p["fw"])
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# rehydrate a stored lineage back into the live view tree (restore for re-emit)
# --------------------------------------------------------------------------- #
def rehydrate(sess, lineage: list[dict]) -> str:
    parent = sess["views"][sess["base_id"]]
    # lineage[0] is the base; reuse the session base (same coords by construction)
    for entry in lineage[1:]:
        node = _new_view(sess, parent,
                         Decimal(entry["cx"]), Decimal(entry["cy"]), Decimal(entry["fw"]),
                         entry["step_kind"])
        _note_view_depth(sess, node["fw"])
        parent = node
    return parent["id"]


# --------------------------------------------------------------------------- #
# state payloads
# --------------------------------------------------------------------------- #
def coord_str(node) -> dict:
    return {"cx": rc.dec_str(node["cx"]), "cy": rc.dec_str(node["cy"]),
            "fw": f"{float(node['fw']):.6e}"}


def solutions_payload(atom_id: str) -> list[dict]:
    out = []
    for e in store.emits_for(atom_id):
        f = e["render"]
        out.append({
            "emit_id": e["emit_id"], "class": e["class"],
            "fw": f"{float(f['fw']):.6e}",
            "steps": len(e["lineage"]) - 1,
            "created_at": e.get("created_at"),
            "vivid_crop": e["vivid_crop"],
        })
    return out


def atom_state(atom_id: str) -> dict:
    sess = _session(atom_id)
    cur = sess["views"][sess["current_id"]]
    vb = store.load_vbad().get(atom_id)
    return {
        "atom_id": atom_id,
        "current_view": sess["current_id"],
        "coord": coord_str(cur),
        "depth_from_base": float(sess["activity"]["base_fw"] / cur["fw"]),
        "quality_ready": sess["quality_view_id"] == sess["current_id"],
        "at_base": sess["current_id"] == sess["base_id"],
        "solutions": solutions_payload(atom_id),
        "verified_bad": bool(vb),
        "activity": {
            "active_seconds": round(sess["activity"]["active_seconds"], 1),
            "box_descents": sess["activity"]["box_descents"],
            "deepest_zoom": round(deepest_zoom(sess), 2),
        },
    }


def library_payload() -> list[dict]:
    emits = store.load_emits()
    vbad = store.load_vbad()
    counts: dict[str, int] = {}
    for e in emits:
        counts[e["atom_id"]] = counts.get(e["atom_id"], 0) + 1
    rows = []
    for a in ATOMS.values():
        aid = a["id"]
        rows.append({
            "id": aid, "degree": a["degree"], "split": a["split"],
            "family": a["family"],
            "emit_count": counts.get(aid, 0),
            "verified_bad": aid in vbad,
            "touched": aid in SESSIONS and SESSIONS[aid]["touched"],
        })
    rows.sort(key=lambda r: (r["degree"], r["id"]))
    return rows


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/library", methods=["POST"])
def library():
    with _lock:
        return jsonify({"atoms": library_payload(), "total": len(ATOMS)})


@app.route("/file/<path:relpath>")
def durable_file(relpath):
    """Serve an emitted crop for the solutions list. The record carries a portable
    repo-relative path (data/descent_harness/{crops,vivid}/…); the bytes live out of
    the tree via the artifacts resolver. Restricted to that crop class (no traversal)."""
    norm = relpath.replace("\\", "/")
    if not (norm.startswith("data/descent_harness/crops/")
            or norm.startswith("data/descent_harness/vivid/")):
        return ("forbidden", 403)
    target = store.resolve(norm)
    if not target.exists():
        return ("not found", 404)
    return send_file(target, max_age=0)


@app.route("/thumb/<atom_id>")
def thumb(atom_id):
    if atom_id not in ATOMS:
        return ("unknown atom", 404)
    path = render_thumb(ATOMS[atom_id])
    return send_file(path, mimetype="image/png", max_age=0)


@app.route("/open", methods=["POST"])
def open_atom():
    atom_id = request.get_json()["atom_id"]
    if atom_id not in ATOMS:
        return ("unknown atom", 404)
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        cur = sess["views"][sess["current_id"]]
        cx, cy, fw = cur["cx"], cur["cy"], cur["fw"]
        atom = dict(ATOMS[atom_id])
    img = render_nav(atom, cx, cy, fw)
    with _lock:
        st = atom_state(atom_id)
    st["img"] = img
    st["atom"] = {"id": atom_id, "degree": atom["degree"], "split": atom["split"],
                  "family": atom["family"]}
    return jsonify(st)


@app.route("/nav_click", methods=["POST"])
def nav_click():
    d = request.get_json()
    atom_id = d["atom_id"]
    px, py = float(d["px"]), float(d["py"])
    recenter_only = bool(d.get("recenter_only", False))
    zoom = Decimal(str(d.get("zoom", 2)))
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        cur = sess["views"][sess["current_id"]]
        wx, wy = rc.click_to_world(px, py, cur["cx"], cur["cy"], cur["fw"], NAV_W, NAV_H)
        new_fw = cur["fw"] if recenter_only else cur["fw"] / zoom
        node = _new_view(sess, cur, wx, wy, new_fw, "click")
        sess["current_id"] = node["id"]
        sess["quality_view_id"] = None
        _note_view_depth(sess, new_fw)
        atom = dict(ATOMS[atom_id])
        cx, cy, fw = node["cx"], node["cy"], node["fw"]
    img = render_nav(atom, cx, cy, fw)
    with _lock:
        st = atom_state(atom_id)
    st["img"] = img
    return jsonify(st)


@app.route("/nav_box", methods=["POST"])
def nav_box():
    d = request.get_json()
    atom_id = d["atom_id"]
    down_px, down_py, cur_px = float(d["down_px"]), float(d["down_py"]), float(d["cur_px"])
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        cur = sess["views"][sess["current_id"]]
        ncx, ncy, nfw = rc.box_commit(down_px, down_py, cur_px,
                                      cur["cx"], cur["cy"], cur["fw"], NAV_W, NAV_H)
        # f64-wall guard (atom_instrument, in-process): refuse a box whose quality-
        # fidelity pixel spacing would cross the f64 wall. Atoms carry 8–9.5 decades
        # of headroom, so this should almost never fire.
        inst = sess["instrument"]
        margin = inst.f64_wall_margin_decades(GUARD_W, ss=GUARD_SS,
                                              k=float(nfw) * inst.abs_A)
        if margin < 0:
            return jsonify({"error": (
                f"box refused: predicted f64-wall margin {margin:.2f} decades "
                f"(< 0). fw {float(nfw):.3e} at {GUARD_W}×ss{GUARD_SS} would cross "
                f"pixel-spacing 1e-13. Draw a larger box.")}), 400
        node = _new_view(sess, cur, ncx, ncy, nfw, "box")
        sess["current_id"] = node["id"]
        sess["quality_view_id"] = None
        sess["activity"]["box_descents"] += 1
        _note_view_depth(sess, nfw)
        atom = dict(ATOMS[atom_id])
        cx, cy, fw = node["cx"], node["cy"], node["fw"]
    img = render_nav(atom, cx, cy, fw)
    with _lock:
        st = atom_state(atom_id)
    st["img"] = img
    return jsonify(st)


@app.route("/undo", methods=["POST"])
def undo():
    atom_id = request.get_json()["atom_id"]
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        cur = sess["views"][sess["current_id"]]
        if cur["parent_id"] is not None:
            sess["current_id"] = cur["parent_id"]
            sess["quality_view_id"] = None
        node = sess["views"][sess["current_id"]]
        atom = dict(ATOMS[atom_id])
        cx, cy, fw = node["cx"], node["cy"], node["fw"]
    img = render_nav(atom, cx, cy, fw)
    with _lock:
        st = atom_state(atom_id)
    st["img"] = img
    return jsonify(st)


@app.route("/to_base", methods=["POST"])
def to_base():
    atom_id = request.get_json()["atom_id"]
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        sess["current_id"] = sess["base_id"]
        sess["quality_view_id"] = None
        node = sess["views"][sess["base_id"]]
        atom = dict(ATOMS[atom_id])
        cx, cy, fw = node["cx"], node["cy"], node["fw"]
    img = render_nav(atom, cx, cy, fw)
    with _lock:
        st = atom_state(atom_id)
    st["img"] = img
    return jsonify(st)


@app.route("/quality", methods=["POST"])
def quality():
    """Render the current view at full label-crop quality (canonical twilight_shifted
    + vivid blue_orange companion). Display the vivid one for judging; the canonical
    is what gets stored. Enables the emit buttons for THIS view only."""
    atom_id = request.get_json()["atom_id"]
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        cur = sess["views"][sess["current_id"]]
        view_id = cur["id"]
        atom = dict(ATOMS[atom_id])
        canon_blk = canonical_block(atom, cur["cx"], cur["cy"], cur["fw"])
        vivid_blk = vivid_block(atom, cur["cx"], cur["cy"], cur["fw"])
    canon_scratch, vivid_scratch = store.quality_scratch_paths(atom_id, view_id)
    render_quality_crop(canon_blk, canon_scratch, store.CLEAN_COLORMAPS)
    render_quality_crop(vivid_blk, vivid_scratch, store.VIVID_COLORMAPS)
    with _lock:
        sess = _session(atom_id)
        if sess["current_id"] == view_id:      # view unchanged during the render
            sess["quality_view_id"] = view_id
            sess.setdefault("quality", {})[view_id] = {
                "canonical_block": canon_blk, "vivid_block": vivid_blk,
                "canonical_scratch": str(canon_scratch), "vivid_scratch": str(vivid_scratch),
            }
        st = atom_state(atom_id)
    st["vivid_img"] = _b64(vivid_scratch, "image/jpeg")
    st["canonical_img"] = _b64(canon_scratch, "image/jpeg")
    return jsonify(st)


@app.route("/emit", methods=["POST"])
def emit():
    d = request.get_json()
    atom_id = d["atom_id"]
    klass = int(d["class"])
    if klass not in (3, 4):
        return jsonify({"error": "class must be 3 or 4"}), 400
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        view_id = sess["current_id"]
        if sess["quality_view_id"] != view_id:
            return jsonify({"error": "render quality for this view before emitting"}), 400
        qq = sess.get("quality", {}).get(view_id)
        if not qq:
            return jsonify({"error": "quality render missing; re-render"}), 400
        atom = dict(ATOMS[atom_id])
        seq = store.next_emit_seq(atom_id)
        emit_id = f"{atom_id}__e{seq}"
        lineage = walk_lineage(sess, view_id)
        canon_scratch = Path(qq["canonical_scratch"])
        vivid_scratch = Path(qq["vivid_scratch"])
        canon_blk = qq["canonical_block"]
        vivid_blk = qq["vivid_block"]

    # copy the approved quality crops to durable storage (approve == saved, no re-render).
    # Crops resolve OUT of the tree via the artifacts seam; the record stores the portable
    # repo-relative string.
    canon_dst = store.canonical_crop_path(emit_id)
    vivid_dst = store.vivid_crop_path(emit_id)
    canon_dst.parent.mkdir(parents=True, exist_ok=True)
    vivid_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canon_scratch, canon_dst)
    shutil.copyfile(vivid_scratch, vivid_dst)

    record = {
        "emit_id": emit_id,
        "atom_id": atom_id,
        "split": atom["split"],
        "degree": atom["degree"],
        "period": atom["period"],
        "family": atom["family"],
        "class": klass,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "created_unix": time.time(),
        "final_view_id": view_id,
        "render": canon_blk,                    # canonical, model-facing
        "vivid_render": vivid_blk,              # judging companion
        "palette_source": store.rel(store.CLEAN_COLORMAPS),
        "vivid_palette_source": store.rel(store.VIVID_COLORMAPS),
        "jpg_quality": store.JPG_QUALITY,
        "canonical_crop": store.canonical_crop_rel(emit_id),   # portable repo-relative
        "vivid_crop": store.vivid_crop_rel(emit_id),
        "lineage": lineage,
    }
    store.append_emit(record)
    with _lock:
        st = atom_state(atom_id)
    st["emitted"] = emit_id
    return jsonify(st)


@app.route("/solution/delete", methods=["POST"])
def solution_delete():
    d = request.get_json()
    atom_id, emit_id = d["atom_id"], d["emit_id"]
    store.delete_emit(emit_id)
    with _lock:
        st = atom_state(atom_id)
    return jsonify(st)


@app.route("/solution/restore", methods=["POST"])
def solution_restore():
    d = request.get_json()
    atom_id, emit_id = d["atom_id"], d["emit_id"]
    rec = next((e for e in store.emits_for(atom_id) if e["emit_id"] == emit_id), None)
    if rec is None:
        return jsonify({"error": "no such solution"}), 404
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        leaf = rehydrate(sess, rec["lineage"])
        sess["current_id"] = leaf
        sess["quality_view_id"] = None
        node = sess["views"][leaf]
        atom = dict(ATOMS[atom_id])
        cx, cy, fw = node["cx"], node["cy"], node["fw"]
    img = render_nav(atom, cx, cy, fw)
    with _lock:
        st = atom_state(atom_id)
    st["img"] = img
    return jsonify(st)


@app.route("/verified_bad", methods=["POST"])
def verified_bad():
    d = request.get_json()
    atom_id = d["atom_id"]
    value = bool(d["value"])
    with _lock:
        sess = _session(atom_id)
        _touch(sess)
        if value:
            rec = {
                "atom_id": atom_id,
                "verified_bad": True,
                "active_seconds": round(sess["activity"]["active_seconds"], 1),
                "box_descents": sess["activity"]["box_descents"],
                "deepest_zoom_factor": round(deepest_zoom(sess), 3),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "recorded_unix": time.time(),
            }
            store.set_vbad(atom_id, rec)
        else:
            store.set_vbad(atom_id, None)
        st = atom_state(atom_id)
    return jsonify(st)


def smoke():
    """Non-HTTP sanity: render a nav frame + a quality crop for the first atom."""
    a = next(iter(ATOMS.values()))
    print(f"smoke: atom {a['id']} deg{a['degree']} {a['family']}")
    img = render_nav(a, Decimal(a["cx"]), Decimal(a["cy"]), Decimal(a["fw"]))
    assert img.startswith("data:image/png;base64,") and len(img) > 1000
    blk = canonical_block(a, Decimal(a["cx"]), Decimal(a["cy"]), Decimal(a["fw"]))
    out = store.QUALITY_DIR / "_smoke.jpg"
    render_quality_crop(blk, out, store.CLEAN_COLORMAPS)
    assert out.exists() and out.stat().st_size > 1000
    print("smoke PASSED")


if __name__ == "__main__":
    if not rc.RENDER_BIN.exists():
        sys.exit(f"render binary not found: {rc.RENDER_BIN}")
    store._ensure_dirs()
    # `--selection <path>` (or the DESCENT_SELECTION env var) swaps the atom set —
    # notably for `selection_triage.json`, the triage wall's accept-set. ATOMS is built
    # at import, so re-load it here once the flag has been read.
    if "--selection" in sys.argv:
        sel = sys.argv[sys.argv.index("--selection") + 1]
        ATOMS = {a["id"]: a for a in store.load_selection(store.REPO_ROOT / sel
                                                          if not Path(sel).is_absolute() else sel)}
        print(f"selection: {sel}")
    elif os.environ.get("DESCENT_SELECTION"):
        print(f"selection: {os.environ['DESCENT_SELECTION']}")
    if "--smoke" in sys.argv:
        smoke()
        sys.exit(0)
    print(f"Descent harness: {len(ATOMS)} atoms")
    print("Running at: http://127.0.0.1:5006\n")
    app.run(host="127.0.0.1", port=5006, debug=False, threaded=True)
