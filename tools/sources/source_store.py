#!/usr/bin/env python
"""Storage for the minibrot source sheets.

Two classes, per the prompt:

* **Durable** (`data/minibrot_sources/`, git-tracked, in-tree): the nuclei lists and
  their descriptors — a computed population everything downstream keys off, and the
  thing that makes the sheets reproducible and the overlap matrix meaningful.
  `<source_id>/atoms.jsonl` + `<source_id>/meta.json`, plus `overlap.json` and
  `index.json` at the root.
* **Bulk, regenerable-but-expensive** (`data/minibrot_sources/{tiles,sheets}/`):
  the rendered tiles (3 scales x ~150 atoms x 8 sources) and the sheet HTML. These
  resolve **out of the working tree** through `artifacts.resolve`, exactly like the
  descent-harness crops and the triage thumbnails.

Nothing load-bearing goes to `scratch/`.

The sheets live in the SAME out-of-tree root as the tiles they reference, so a sheet
addresses its images by a plain relative path (`../tiles/<id>__x4.png`) and opens by
double-click with **no manual path fixing** — which is the actual requirement.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "descent"))

import artifacts as _artifacts     # noqa: E402
import triage_store as ts          # noqa: E402

ROOT = REPO_ROOT / "data" / "minibrot_sources"      # durable, in-tree
OVERLAP = ROOT / "overlap.json"
INDEX_JSON = ROOT / "index.json"

TILES_REL = "data/minibrot_sources/tiles"            # bulk, out-of-tree
SHEETS_REL = "data/minibrot_sources/sheets"

# Framing is IMPORTED from the triage wall, not restated — identical framing across
# every sheet is the only thing that makes the sources comparable, so there must be
# exactly one definition of it. (`test_sources.py` pins this.)
SCALES = ts.SCALES                   # (1, 4, 16)
DEFAULT_SCALE = ts.DEFAULT_SCALE     # 4
TILE_W, TILE_H, TILE_SS = ts.THUMB_W, ts.THUMB_H, ts.THUMB_SS
TILE_PALETTE = ts.THUMB_PALETTE      # "blue_orange"
TILE_COLORMAPS = ts.THUMB_COLORMAPS

_io_lock = threading.RLock()


def resolve(relpath) -> Path:
    return Path(_artifacts.resolve(relpath))


def tiles_dir() -> Path:
    return resolve(TILES_REL)


def sheets_dir() -> Path:
    return resolve(SHEETS_REL)


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tiles_dir().mkdir(parents=True, exist_ok=True)
    sheets_dir().mkdir(parents=True, exist_ok=True)


def rel(p) -> str:
    rp = Path(p).resolve()
    try:
        return str(rp.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(rp).replace("\\", "/")


# --------------------------------------------------------------------------- #
# per-source durable records
# --------------------------------------------------------------------------- #
def source_dir(source_id: str) -> Path:
    return ROOT / source_id


def atoms_path(source_id: str) -> Path:
    return source_dir(source_id) / "atoms.jsonl"


def meta_path(source_id: str) -> Path:
    return source_dir(source_id) / "meta.json"


def load_atoms(source_id: str) -> list[dict]:
    p = atoms_path(source_id)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_atoms(source_id: str, atoms: list[dict]) -> Path:
    with _io_lock:
        ensure_dirs()
        source_dir(source_id).mkdir(parents=True, exist_ok=True)
        p = atoms_path(source_id)
        tmp = p.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for a in atoms:
                f.write(json.dumps(a) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p


def write_json(path: Path, doc) -> Path:
    with _io_lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return Path(path)


def load_meta(source_id: str) -> dict:
    p = meta_path(source_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def built_sources() -> list[str]:
    if not ROOT.exists():
        return []
    return sorted(d.name for d in ROOT.iterdir()
                  if d.is_dir() and (d / "meta.json").exists())


# --------------------------------------------------------------------------- #
# tiles + sheets (bulk, out-of-tree)
# --------------------------------------------------------------------------- #
def tile_rel(atom_id: str, scale: int) -> str:
    return f"{TILES_REL}/{atom_id}__x{scale}.png"


def tile_path(atom_id: str, scale: int) -> Path:
    return resolve(tile_rel(atom_id, scale))


def sheet_path(name: str) -> Path:
    return sheets_dir() / f"{name}.html"


def index_path() -> Path:
    return sheets_dir() / "index.html"


def frame_width(window_scale, scale: int) -> float:
    """Frame width for a tile: `scale` x the atom's own size. Same rule as the wall."""
    return float(window_scale) * float(scale)
