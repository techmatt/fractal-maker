#!/usr/bin/env python
"""Durable store for the minibrot descent harness.

These records capture human choices nothing can regenerate, so they are
**durable** (committed under `data/descent_harness/`, negated out of the
`/data/*` gitignore by exact path):

  * `selection.json`  — the data-driven 40-atom subset (built by build_selection.py).
                        `selection_triage.json` (the triage wall's accept-set) is an
                        alternative in the same schema, chosen via `DESCENT_SELECTION`.
  * `triage/`         — the triage wall's durable pool + accept/reject verdicts
                        (see triage_store.py)
  * `emits.jsonl`     — one row per emitted q3/q4 solution, with its full descent
                        lineage and the render blocks needed to reproduce the crop
  * `verified_bad.json` — per-atom "no easy q3/q4 here" verdicts (reversible)
  * `crops/<emit_id>.jpg`  — the model-facing canonical (twilight_shifted) crop
  * `vivid/<emit_id>.jpg`  — the vivid (blue_orange) judging companion

Thumbnails and navigation previews are **scratch** (regenerable) — they live under
`scratch/descent_harness/`, never here.

The emit crop is produced by `corpus_common.render_corpus_crop` (the sanctioned,
byte-reproducible corpus-crop path), so a re-render from a stored record's
`render` block reproduces the saved crop bit-for-bit (the round-trip acceptance).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "scoring"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "mining"))
import artifacts as _artifacts          # noqa: E402  (the out-of-tree resolver seam)
import corpus_common as cc              # noqa: E402
import active_ckpt as _prod             # noqa: E402  (production render-block source of truth)

DH_DIR = REPO_ROOT / "data" / "descent_harness"
SELECTION = DH_DIR / "selection.json"
EMITS = DH_DIR / "emits.jsonl"
VBAD = DH_DIR / "verified_bad.json"

# Emitted crop pairs are destined for the label corpus, so — like every other corpus
# crop — they resolve OUT of the working tree through artifacts.resolve (the set grows
# 40→163). Records store the portable REPO-RELATIVE string; the bytes live at resolve().
CROPS_REL = "data/descent_harness/crops"
VIVID_REL = "data/descent_harness/vivid"

# scratch (regenerable previews) — stay in-tree under scratch/
SCRATCH_DIR = REPO_ROOT / "scratch" / "descent_harness"
THUMBS_DIR = SCRATCH_DIR / "thumbs"
QUALITY_DIR = SCRATCH_DIR / "quality"     # per-view quality renders awaiting emit

CLEAN_COLORMAPS = REPO_ROOT / "data" / "palettes" / "clean_colormaps.json"
VIVID_COLORMAPS = REPO_ROOT / "data" / "palettes" / "vivid_blue_orange.json"

# --- Canonical label-crop render block: DERIVED from the production source, not copied.
# `tools/corpus/build_native_multibrot_band.py` is the authoritative producer of the
# 640×360 ss2 twilight_shifted native-multibrot/mandelbrot label crop; it imports
# `PALETTE`, `JPG_Q` and `auto_maxiter` from `tools/scoring/active_ckpt.py`. We import
# the SAME production primitives so the block moves in lockstep with production (maxiter
# DERIVES from active_ckpt.auto_maxiter — it is not fixed). auto_maxiter must NOT come
# from the explorer's navigation copy (render_core); that copy is for nav renders only.
CANONICAL_PALETTE = _prod.PALETTE          # "twilight_shifted"
VIVID_PALETTE = "blue_orange"
JPG_QUALITY = _prod.JPG_Q                   # 90
# Geometry is the corpus canonical for this crop class (matches build_native_multibrot_band).
CROP_W, CROP_H, CROP_SS = 640, 360, 2
CROP_FILTER = "lanczos3"
CROP_INTERIOR = "black"
CROP_COMPOSITION = "center"


def production_maxiter(fw) -> int:
    """The production canonical-crop maxiter derivation (active_ckpt.auto_maxiter)."""
    return int(_prod.auto_maxiter(float(fw)))


def canonical_render_block(cx, cy, fw, family) -> dict:
    """The version-invariant canonical label-crop render block, derived from the
    production source (active_ckpt) — geometry + interior/filter/composition are the
    corpus canonical; maxiter and palette come from production."""
    blk = cc.render_block(
        cx=cc.hp_str(cx), cy=cc.hp_str(cy), fw=cc.hp_str(fw),
        maxiter=production_maxiter(fw), palette=CANONICAL_PALETTE,
        composition=CROP_COMPOSITION,
        width=CROP_W, height=CROP_H, ss=CROP_SS,
        filter=CROP_FILTER, interior_mode=CROP_INTERIOR,
    )
    blk["fractal_type"] = family            # native family in the block, no c
    blk["c_re"] = None
    blk["c_im"] = None
    return blk


def vivid_render_block(cx, cy, fw, family) -> dict:
    """The vivid judging companion: identical to the canonical block in every field
    except the palette (the colormap library, a render arg not a block field, also
    differs — VIVID_COLORMAPS)."""
    blk = canonical_render_block(cx, cy, fw, family)
    blk["palette"] = VIVID_PALETTE
    return blk


_io_lock = threading.RLock()


def _ensure_dirs():
    DH_DIR.mkdir(parents=True, exist_ok=True)               # in-tree records
    for d in (crops_dir(), vivid_dir(), THUMBS_DIR, QUALITY_DIR):
        d.mkdir(parents=True, exist_ok=True)                # crops/vivid resolve out-of-tree


def rel(p) -> str:
    """Repo-relative path string (forward slashes). Falls back to the absolute path
    when `p` is outside the repo (e.g. a redirected temp store under test)."""
    rp = Path(p).resolve()
    try:
        return str(rp.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(rp).replace("\\", "/")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def selection_path() -> Path:
    """Which selection file the harness opens.

    Defaults to the roster-drawn 40-atom `selection.json`. The `DESCENT_SELECTION`
    env var (repo-relative or absolute) points it at another file in the same schema —
    notably `selection_triage.json`, the accept-set from the triage wall
    (`tools/descent/build_triage_selection.py`). The original file is never replaced,
    so the 40-atom study stays reproducible."""
    override = os.environ.get("DESCENT_SELECTION")
    if not override:
        return SELECTION
    p = Path(override)
    return p if p.is_absolute() else REPO_ROOT / p


def load_selection(path=None) -> list[dict]:
    doc = json.loads(Path(path or selection_path()).read_text())
    return doc["atoms"]


# --------------------------------------------------------------------------- #
# emits
# --------------------------------------------------------------------------- #
def load_emits() -> list[dict]:
    if not EMITS.exists():
        return []
    return [json.loads(l) for l in EMITS.read_text().splitlines() if l.strip()]


def emits_for(atom_id: str) -> list[dict]:
    return [e for e in load_emits() if e["atom_id"] == atom_id]


def next_emit_seq(atom_id: str) -> int:
    return len(emits_for(atom_id)) + 1


def append_emit(record: dict) -> None:
    with _io_lock:
        _ensure_dirs()
        with open(EMITS, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def delete_emit(emit_id: str) -> bool:
    """Remove an emit row (and its crop pair). Returns True if something was removed."""
    with _io_lock:
        rows = load_emits()
        keep = [e for e in rows if e["emit_id"] != emit_id]
        if len(keep) == len(rows):
            return False
        gone = [e for e in rows if e["emit_id"] == emit_id]
        with open(EMITS, "w", encoding="utf-8") as f:
            for e in keep:
                f.write(json.dumps(e) + "\n")
        for e in gone:
            for key in ("canonical_crop", "vivid_crop"):
                p = resolve(e[key])          # crops live out-of-tree via the resolver
                if p.exists():
                    p.unlink()
        return True


# --------------------------------------------------------------------------- #
# verified-bad
# --------------------------------------------------------------------------- #
def load_vbad() -> dict:
    if not VBAD.exists():
        return {}
    return json.loads(VBAD.read_text())


def set_vbad(atom_id: str, record: dict | None) -> None:
    """record=None clears the verdict (the toggle is reversible)."""
    with _io_lock:
        _ensure_dirs()
        data = load_vbad()
        if record is None:
            data.pop(atom_id, None)
        else:
            data[atom_id] = record
        VBAD.write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# crop paths — records store the REPO-RELATIVE string (*_rel); the bytes live at
# resolve(*_rel), out of the working tree via the artifacts seam.
# --------------------------------------------------------------------------- #
def resolve(relpath) -> Path:
    """Repo-relative artifact path → real on-disk location (out-of-tree for crops)."""
    return Path(_artifacts.resolve(relpath))


def crops_dir() -> Path:
    return resolve(CROPS_REL)


def vivid_dir() -> Path:
    return resolve(VIVID_REL)


def canonical_crop_rel(emit_id: str) -> str:
    return f"{CROPS_REL}/{emit_id}.jpg"


def vivid_crop_rel(emit_id: str) -> str:
    return f"{VIVID_REL}/{emit_id}.jpg"


def canonical_crop_path(emit_id: str) -> Path:
    return resolve(canonical_crop_rel(emit_id))


def vivid_crop_path(emit_id: str) -> Path:
    return resolve(vivid_crop_rel(emit_id))


def quality_scratch_paths(atom_id: str, view_id: str) -> tuple[Path, Path]:
    stem = f"{atom_id}__{view_id}"
    return QUALITY_DIR / f"{stem}__canonical.jpg", QUALITY_DIR / f"{stem}__vivid.jpg"


def thumb_path(atom_id: str) -> Path:
    return THUMBS_DIR / f"{atom_id}.png"
