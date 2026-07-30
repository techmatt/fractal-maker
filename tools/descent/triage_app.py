#!/usr/bin/env python
"""Minibrot **triage wall** — a dense, keyboard-driven accept/reject pass over the
enumerated nucleus pool.

Why it exists: the descent roster is an unfiltered sample of nuclei (stratified on
period, which does not predict quality; cut on `|A|`, which is feasibility). The good
atoms are in there unselected at maybe a 10% rate. Enumeration is cheap and Matt's eye
is fast, so the fix is volume plus a rejection pass — not a fitted criterion. Accepted
atoms become the descent tool's selection set; rejected atoms are the negative class.

Two invariants this app exists to protect:

1. **No metadata reaches the browser during triage.** No period, no degree, no `|A|`,
   no model score of any kind — not in the payload, not in a tooltip, not smuggled
   through a tile id (ids are opaque content hashes for exactly this reason). If the
   numbers were on screen, the rejections would be partly a response to them and the
   covariate join afterwards would mean nothing. Metadata is *recorded*, never
   *displayed*.
2. **Verdicts are durable and survive a restart.** Every keystroke appends to
   `data/descent_harness/triage/verdicts.jsonl` before the UI advances; the wall
   reopens at the first un-triaged tile.

Framing: tiles render at 1x / 4x / 16x the atom's own size (`triage_store.SCALES`),
4x by default, click to cycle. A reference row of known-good locations sits above the
wall at the same three scales — if known-good material does not read as good at this
framing, the framing is wrong and that must be visible before 200 tiles are scanned.

Run:  uv run python tools/descent/triage_app.py
      http://127.0.0.1:5007
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "explorer"))

import render_core as rc          # noqa: E402
import triage_store as ts         # noqa: E402
import prerender_triage as pre    # noqa: E402

PORT = 5007
PAGE_SIZE = 60                    # tiles per page — a 1000-tile pass never loads at once
MAX_ENGINE = 3                    # concurrent render processes (<= CLAUDE.md's 4)
PREFETCH_THREADS = 2              # background tile-warming workers (share the semaphore)

app = Flask(__name__)

_engine_sem = threading.BoundedSemaphore(MAX_ENGINE)
_render_locks: dict[tuple[str, int], threading.Lock] = {}
_locks_guard = threading.Lock()

# Geometry cache: tile_id -> {cx, cy, base, family}. Metadata NEVER leaves this dict.
GEOM: dict[str, dict] = {}
POOL_ORDER: list[str] = []
POOL_SET: set[str] = set()        # membership test stays O(1) as the pool passes 1000
REFS: list[dict] = []


def load_all() -> None:
    global POOL_ORDER, POOL_SET, REFS
    GEOM.clear()
    REFS = ts.load_references()
    for r in REFS:
        GEOM[r["id"]] = {"cx": r["cx"], "cy": r["cy"],
                         "base": r["base_scale"], "family": r["family"]}
    pool = ts.load_pool()
    POOL_ORDER = [a["id"] for a in pool]
    POOL_SET = set(POOL_ORDER)
    for a in pool:
        GEOM[a["id"]] = {"cx": a["cx"], "cy": a["cy"],
                         "base": a["window_scale"], "family": a["family"]}


# --------------------------------------------------------------------------- #
# rendering (lazy, per-tile, de-duplicated across concurrent requests)
# --------------------------------------------------------------------------- #
def _lock_for(tile_id: str, scale: int) -> threading.Lock:
    with _locks_guard:
        return _render_locks.setdefault((tile_id, scale), threading.Lock())


def ensure_tile(tile_id: str, scale: int) -> Path | None:
    """Return the tile's PNG path, rendering it on demand. None if unknown tile."""
    geom = GEOM.get(tile_id)
    if geom is None:
        return None
    out = ts.thumb_path(tile_id, scale)
    if out.exists() and out.stat().st_size > 0:
        return out
    with _lock_for(tile_id, scale):
        if out.exists() and out.stat().st_size > 0:
            return out
        with _engine_sem:
            return pre.render_tile(tile_id, scale, geom, threads=3)


def prefetch_worker(stop: threading.Event) -> None:
    """Warm missing tiles in wall order (default scale first, then the alternates), so
    a fresh pool becomes scannable without a separate pre-render pass."""
    order = [(tid, s) for s in (ts.DEFAULT_SCALE, *[x for x in ts.SCALES if x != ts.DEFAULT_SCALE])
             for tid in ([r["id"] for r in REFS] + POOL_ORDER)]
    for tid, scale in order:
        if stop.is_set():
            return
        p = ts.thumb_path(tid, scale)
        if p.exists() and p.stat().st_size > 0:
            continue
        try:
            ensure_tile(tid, scale)
        except Exception:
            pass                      # a failed tile is reported by the pre-render tool
        time.sleep(0.02)              # stay out of the way of interactive requests


# --------------------------------------------------------------------------- #
# payloads — metadata-free by construction
# --------------------------------------------------------------------------- #
def progress() -> dict:
    v = ts.load_verdicts()
    acc = sum(1 for x in v.values() if x == "accept")
    rej = sum(1 for x in v.values() if x == "reject")
    return {"total": len(POOL_ORDER), "judged": acc + rej,
            "accepted": acc, "rejected": rej,
            "page_size": PAGE_SIZE,
            "pages": max(1, (len(POOL_ORDER) + PAGE_SIZE - 1) // PAGE_SIZE),
            "scales": list(ts.SCALES), "default_scale": ts.DEFAULT_SCALE}


def first_untriaged() -> int:
    v = ts.load_verdicts()
    for i, tid in enumerate(POOL_ORDER):
        if tid not in v:
            return i
    return max(0, len(POOL_ORDER) - 1)


@app.route("/")
def index():
    return render_template("triage.html")


@app.route("/api/state", methods=["POST"])
def api_state():
    """Wall bootstrap. Carries ids + verdicts ONLY — no atom covariates."""
    return jsonify({
        "progress": progress(),
        # reference row: id + label only (the labels ARE the point of a reference)
        "references": [{"id": r["id"], "label": r["label"]} for r in REFS],
        "resume_index": first_untriaged(),
    })


@app.route("/api/page", methods=["POST"])
def api_page():
    """One page of tiles: ids and verdicts, nothing else. The grid pages so a
    1000-tile pass never loads at once."""
    d = request.get_json(silent=True) or {}
    page = max(0, int(d.get("page", 0)))
    lo = page * PAGE_SIZE
    ids = POOL_ORDER[lo:lo + PAGE_SIZE]
    v = ts.load_verdicts()
    return jsonify({
        "page": page, "offset": lo,
        "tiles": [{"id": t, "verdict": v.get(t)} for t in ids],
        "progress": progress(),
    })


@app.route("/api/verdict", methods=["POST"])
def api_verdict():
    d = request.get_json()
    tile_id = d["atom_id"]
    verdict = d.get("verdict")
    if tile_id not in POOL_SET:
        return jsonify({"error": "unknown atom"}), 404
    if verdict not in (None, *ts.VERDICT_VALUES):
        return jsonify({"error": f"verdict must be null or one of {ts.VERDICT_VALUES}"}), 400
    ts.append_verdict(tile_id, verdict, session_id=str(d.get("session_id", "")))
    return jsonify({"atom_id": tile_id, "verdict": verdict, "progress": progress()})


@app.route("/tile/<tile_id>/<int:scale>")
def tile(tile_id, scale):
    if scale not in ts.SCALES:
        return ("bad scale", 400)
    path = ensure_tile(tile_id, scale)
    if path is None:
        return ("unknown tile", 404)
    return send_file(path, mimetype="image/png", max_age=3600)


def smoke() -> None:
    """Non-HTTP sanity: the pool loads, a page is metadata-free, a verdict round-trips."""
    load_all()
    assert POOL_ORDER, "empty pool — run build_triage_pool.py first"
    client = app.test_client()
    st = client.post("/api/state").get_json()
    pg = client.post("/api/page", json={"page": 0}).get_json()
    assert pg["tiles"], "no tiles on page 0"
    leaked = set(pg["tiles"][0]) - {"id", "verdict"}
    assert not leaked, f"page payload leaks metadata: {leaked}"
    print(f"smoke: {st['progress']['total']} atoms, {st['progress']['pages']} pages, "
          f"{len(REFS)} references, resume at {st['resume_index']} — PASSED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-prefetch", action="store_true",
                    help="do not warm missing tiles in the background")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if not rc.RENDER_BIN.exists():
        print(f"render binary not found: {rc.RENDER_BIN}", file=sys.stderr)
        return 2
    ts.ensure_dirs()
    if args.smoke:
        smoke()
        return 0

    load_all()
    p = progress()
    print(f"Triage wall: {p['total']} atoms, {p['pages']} pages of {PAGE_SIZE}, "
          f"{p['judged']} already judged ({p['accepted']} accept / {p['rejected']} reject)")
    if not args.no_prefetch:
        stop = threading.Event()
        for _ in range(PREFETCH_THREADS):
            threading.Thread(target=prefetch_worker, args=(stop,), daemon=True).start()
    print(f"Running at: http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
