#!/usr/bin/env python
"""Durable store for the **minibrot triage wall** (`tools/descent/triage_app.py`).

The wall exists because the descent roster is an *unfiltered* sample of nuclei: it
stratifies on period (which does not predict quality) and cuts on `|A|` (a
feasibility axis), so the good atoms sit in it unselected at maybe a 10% rate. The
fix is volume plus a human rejection pass. Matt accepts/rejects tiles; the accepted
set becomes the descent tool's selection set and the rejected set is the negative
class for later work.

Storage classes (mirrors `store.py`, the emit-record store next door):

  * `data/descent_harness/triage/pool.jsonl`     — enumerated atoms, append-only.
    Durable: the ids are content-derived and the verdicts key on them, so a
    re-enumeration that renumbered would orphan every verdict.
  * `data/descent_harness/triage/verdicts.jsonl` — append-only accept/reject events.
    **Durable and irreplaceable** — a human judgement nothing can regenerate.
    Latest event per atom wins (so a re-verdict is an append, never an edit).
  * `data/descent_harness/triage/enum_state.json` — the enumeration cursor, so a
    later run extends the pool instead of re-running the seeds already consumed.
  * `data/descent_harness/triage/references.json` — the known-good reference row.
  * `data/descent_harness/triage/neighbors.json` — DERIVED (regenerated on every
    enumeration run); pool-relative neighbour counts, a lower bound by construction.
  * `data/descent_harness/thumbs/<atom_id>__x<scale>.png` — the wall images. Bulk
    regenerable (a pure function of the atom row + the constants below), so they
    resolve **out of the working tree** through `artifacts.resolve`, exactly like the
    emit crops. 200 atoms x 3 scales = 600 files today and this scales to 1000+
    atoms / 3000+ files; that never lands in the source tree.

Nothing load-bearing goes to `scratch/`.

**No metadata reaches the browser.** Atom ids are opaque content hashes (`mt<hex>`)
precisely so a tile cannot leak period or degree through its own name — if the
numbers were on screen, the accept/reject verdicts would be partly a response to
them and the covariate join afterwards would mean nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "sourcing"))
sys.path.insert(0, str(HERE))

import artifacts as _artifacts   # noqa: E402  (the out-of-tree resolver seam)
import store                     # noqa: E402  (palette constants; the emit store next door)
import deep_center_finder as dcf  # noqa: E402  (shared read-time dedup canonicalization)

# The significant-digit rounding the stored dedup_key/id was built at
# (build_minibrot_roster.DEDUP_DPS). The read-time snapped key must round at the SAME
# width so a snapped real-axis nucleus lands on the stored key of its noise-free sibling.
DEDUP_DPS = 22

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
TRIAGE_DIR = REPO_ROOT / "data" / "descent_harness" / "triage"
POOL = TRIAGE_DIR / "pool.jsonl"
VERDICTS = TRIAGE_DIR / "verdicts.jsonl"
ENUM_STATE = TRIAGE_DIR / "enum_state.json"
REFERENCES = TRIAGE_DIR / "references.json"
NEIGHBORS = TRIAGE_DIR / "neighbors.json"

# Wall images: repo-relative string in records, bytes out-of-tree via the resolver.
THUMBS_REL = "data/descent_harness/thumbs"

# --------------------------------------------------------------------------- #
# render constants — the framing ladder
#
# Tiles are framed at 1x / 4x / 16x the atom's own size (`window_scale` = 1/|A|),
# NOT at 1x alone: at 1x the island fills the frame and every atom looks alike.
# 4x is the wall default (it is also the roster's `fw = 4*|size|` convention, so a
# 4x tile is the same frame the descent harness opens an atom at).
#
# Fidelity is *navigation* class, not crop class: cheap, no lanczos JPG corpus path.
# Vivid `blue_orange` throughout — the judging palette, not the model-facing one.
# --------------------------------------------------------------------------- #
SCALES = (1, 4, 16)
DEFAULT_SCALE = 4
THUMB_W, THUMB_H, THUMB_SS = 320, 180, 2
THUMB_PALETTE = store.VIVID_PALETTE          # "blue_orange"
THUMB_COLORMAPS = store.VIVID_COLORMAPS

_io_lock = threading.RLock()


def resolve(relpath) -> Path:
    """Repo-relative artifact path -> real on-disk location (out-of-tree for thumbs)."""
    return Path(_artifacts.resolve(relpath))


def rel(p) -> str:
    """Repo-relative path string (forward slashes), falling back to the absolute path
    when `p` is outside the repo (e.g. a store redirected to a temp dir under test).
    Same contract as `store.rel` — a print must never crash a run."""
    rp = Path(p).resolve()
    try:
        return str(rp.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(rp).replace("\\", "/")


def thumbs_dir() -> Path:
    return resolve(THUMBS_REL)


def ensure_dirs() -> None:
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    thumbs_dir().mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# atom identity — content-derived, so it is stable under re-enumeration AND
# opaque (a tile id must not leak degree/period to the eye doing the triage).
# --------------------------------------------------------------------------- #
def atom_id(degree: int, dedup_key: str) -> str:
    h = hashlib.sha256(f"{degree}|{dedup_key}".encode()).hexdigest()
    return "mt" + h[:12]


def family_for(degree: int) -> str:
    return "mandelbrot" if degree == 2 else f"multibrot{degree}"


# --------------------------------------------------------------------------- #
# pool
# --------------------------------------------------------------------------- #
def load_pool() -> list[dict]:
    if not POOL.exists():
        return []
    return [json.loads(l) for l in POOL.read_text(encoding="utf-8").splitlines() if l.strip()]


def pool_ids() -> set[str]:
    return {a["id"] for a in load_pool()}


def load_pool_canonical(verdicts: dict | None = None):
    """Read-time-deduped pool. The stored `pool.jsonl` is append-only and UNCHANGED;
    this collapses the per-solve real-axis Newton-noise copies of one atom (distinct
    stored `dedup_key` -> distinct `mt…` id -> the same atom sitting in the pool more
    than once) at read, via `dcf.snapped_dedup_key`.

    Verdict safety — pass `verdicts` (`load_verdicts()`), which keys on the noise-derived
    id, and the collapse never loses one:
      * a group whose merged rows carry MORE THAN ONE DISTINCT verdict is a CONFLICT:
        it is left UNCOLLAPSED (all its rows kept) and reported in `conflicts`, so
        nothing is auto-resolved and no human judgement is dropped;
      * otherwise the group collapses to its first row and `id_map` re-points the merged
        ids to the survivor, so a verdict on a dropped id carries to the kept one.

    Returns `(rows, id_map, conflicts)`. `conflicts` is a list of
    `{"snapped_key", "ids": {id: verdict}}` (empty when every merge is verdict-clean).
    """
    verdicts = {} if verdicts is None else verdicts
    raw = load_pool()
    # group by snapped key, preserving file order for first-wins
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for a in raw:
        k = dcf.snapped_dedup_key(a["cx"], a["cy"], int(a.get("degree", 2)), DEDUP_DPS)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(a)

    rows, id_map, conflicts = [], {}, []
    for k in order:
        g = groups[k]
        if len(g) == 1:
            rows.append(g[0])
            continue
        distinct_verdicts = {verdicts[a["id"]] for a in g if a["id"] in verdicts}
        if len(distinct_verdicts) > 1:                 # conflict: leave the group alone
            conflicts.append({"snapped_key": k,
                              "ids": {a["id"]: verdicts.get(a["id"]) for a in g}})
            rows.extend(g)
            continue
        keep = g[0]                                    # verdict-clean: collapse, first wins
        rows.append(keep)
        for a in g[1:]:
            id_map[a["id"]] = keep["id"]
    return rows, id_map, conflicts


def verdict_for_canonical(verdicts: dict, id_map: dict) -> dict:
    """Verdict lookup that follows a `load_pool_canonical` `id_map`: a survivor inherits a
    verdict recorded on any of the ids that collapsed into it (verdict-clean by
    construction — conflicting groups are never collapsed, so never appear in `id_map`)."""
    out = dict(verdicts)
    for dropped_id, keep_id in id_map.items():
        if dropped_id in verdicts and keep_id not in out:
            out[keep_id] = verdicts[dropped_id]
    return out


def append_atoms(rows: list[dict]) -> int:
    """Append atom rows, skipping ids already present. Returns how many were written."""
    if not rows:
        return 0
    with _io_lock:
        ensure_dirs()
        have = pool_ids()
        fresh = [r for r in rows if r["id"] not in have]
        if not fresh:
            return 0
        with open(POOL, "a", encoding="utf-8") as f:
            for r in fresh:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return len(fresh)


# --------------------------------------------------------------------------- #
# verdicts — append-only event log, latest event per atom wins.
# `verdict` is "accept" | "reject" | None (None clears, so "back" is reversible).
# --------------------------------------------------------------------------- #
VERDICT_VALUES = ("accept", "reject")


def load_verdict_events() -> list[dict]:
    if not VERDICTS.exists():
        return []
    return [json.loads(l) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_verdicts() -> dict[str, str]:
    """Collapse the event log to {atom_id: verdict}. Cleared atoms are absent."""
    out: dict[str, str] = {}
    for e in load_verdict_events():
        v = e.get("verdict")
        if v in VERDICT_VALUES:
            out[e["atom_id"]] = v
        else:
            out.pop(e["atom_id"], None)
    return out


def append_verdict(atom_id_: str, verdict: str | None, *, session_id: str = "") -> dict:
    """Record one verdict event (append-only; never mutates a prior row)."""
    if verdict is not None and verdict not in VERDICT_VALUES:
        raise ValueError(f"verdict must be one of {VERDICT_VALUES} or None, got {verdict!r}")
    rec = {
        "atom_id": atom_id_,
        "verdict": verdict,
        "session_id": session_id,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "recorded_unix": time.time(),
    }
    with _io_lock:
        ensure_dirs()
        with open(VERDICTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return rec


# --------------------------------------------------------------------------- #
# enumeration state (the resumable cursor)
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if not ENUM_STATE.exists():
        return {}
    return json.loads(ENUM_STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    with _io_lock:
        ensure_dirs()
        tmp = ENUM_STATE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, ENUM_STATE)


def write_json(path: Path, doc) -> None:
    with _io_lock:
        ensure_dirs()
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def load_references() -> list[dict]:
    if not REFERENCES.exists():
        return []
    return json.loads(REFERENCES.read_text(encoding="utf-8"))["references"]


def load_neighbors() -> dict:
    if not NEIGHBORS.exists():
        return {}
    return json.loads(NEIGHBORS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# thumbnails
# --------------------------------------------------------------------------- #
def thumb_rel(tile_id: str, scale: int) -> str:
    return f"{THUMBS_REL}/{tile_id}__x{scale}.png"


def thumb_path(tile_id: str, scale: int) -> Path:
    return resolve(thumb_rel(tile_id, scale))


def frame_width(window_scale, scale: int):
    """Frame width for a tile: `scale` x the atom's own size."""
    return float(window_scale) * float(scale)
