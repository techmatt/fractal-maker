#!/usr/bin/env python
r"""morph_embed_cache.py — compute each location's morph CLIP embedding ONCE, ever.

WHY. The sitting cutter's presentation morph-dedup (`sitting_cutter.stage_morph_dedup`)
embeds every surviving row on every invocation, at a measured **0.93 s/row** (a render, a
robust-z transfer and a CLIP forward pass each time). Dedup necessarily runs BEFORE the draw
— the cap is denominated in looks — so a live cut pays that on the whole post-(a)-post-(c)
population, not on the 1,000 rows that reach the page. Two sittings over overlapping material
paid it twice. This store makes the second pass free, and it is the same recipe every morph
consumer uses, so the vectors are not a private format.

WHAT THE KEY IS, AND WHY EACH PART OF IT (`morph_key`)
------------------------------------------------------
A cache key that omits an axis is a SILENT WRONG ANSWER: a hit returns a vector computed
under a different render or a different embedder and nothing anywhere goes red. So the key is
DERIVED from the objects that decide those two things, never a hand-kept list of literals:

  loc=    `location.location_key(loc)` — the canonical join_key-grade identity: family, cx,
          cy, **fw**, c, and the family's declared extra constants, in registry order. Two
          locations that differ anywhere in that string are different pictures.
  maxiter= the iteration cap actually rendered at.
  field=  `library_store.field_stem(loc, "smooth", w, h, ss, policy)` — the SAME token the
          smooth-field cache keys by. Carrying it means a future axis added to field identity
          (the `field_mode_token` and `maxiter_policy_token` seams both landed this way)
          propagates into this key with no edit here. Geometry rides it and is also spelled
          out, because a stem is a hash and a reader cannot see the geometry in one.
  src=    `--dump-field-source`. `beautiful` and `f64` are the same geometry with values
          offset by a constant; `field_source_token` exists precisely because a geometry-only
          key serves one where the other was meant.
  gray=   the grayscale transfer: producer tag + its two constants. `MORPH_PRODUCER` is the
          seam marker whose absence created the mixed-store parity risk once already.
  clip=   the embedder: model name + a digest of the timm PRETRAINED preprocessing config
          (input size, interpolation, mean/std, crop). The model name alone would not notice
          a timm release that re-specifies this model's transform, which changes every vector
          while leaving every key identical.

The three mirrored constant groups (geometry, transfer, model) are pinned to their owners by
`test_morph_embed_cache.py` rather than imported: `colored_clip` pulls torch, and a key
builder that loads a GPU stack to spell a model name would cost seconds per process. Same
reason `build_q4_harvest_batches.render_family_of` is pinned rather than imported.

THE FILE FORMAT, AND WHAT AN INTERRUPTED WRITE COSTS
-----------------------------------------------------
ONE append-only file. Each record is self-describing and independently checkable:

    file header  b"MORPH-EMBED-CACHE v1\n"
    record       REC_MAGIC(4) klen(u32) dim(u32) crc32(u32) key_utf8[klen] f32[dim]

`crc32` covers the key AND the vector, so a torn record is detected rather than served. There
is no separate index to fall out of sync with the data — the index IS the scan, rebuilt at
open. A kill mid-append can therefore only damage the TAIL, and `open()` recovers by
truncating back to the last record that verifies, reporting `bytes_dropped` (never silently:
a recovery nobody prints is a store that quietly shrank). Damage further in is handled the
same conservative way — everything from the first bad record on is dropped — because a
resync-by-scanning-for-magic could re-admit a record whose length field is garbage.

Appends are a single `write()` of the whole record followed by `flush()` + `os.fsync()`, so a
crash after the syscall returns cannot leave a hit that is not on disk. SINGLE WRITER: two
processes appending to one store is not made safe here, and the store is per-machine bulk, so
the contract is one cutter at a time rather than a lock file whose staleness is its own
failure mode.

STORAGE CLASS: bulk. Expensive (hours) but deterministic to rebuild from committed code plus
the locations themselves, so it goes out-of-tree through the ONE ARTIFACTS_ROOT resolver
(`paths.bulk` -> `artifacts.resolve`), registered in `artifacts.RELOCATED_PREFIXES`. It is
not durable: nothing in it records a population that no longer exists.

    uv run python tools/wallpaper/morph_embed_cache.py stat
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                        # noqa: E402
from tools.corpus import location as loc_mod        # noqa: E402
from tools.wallpaper import library_store as store  # noqa: E402

# --------------------------------------------------------------------------- #
# The store path. A LITERAL relocated prefix (one fixed path, not a family that grows a
# member per run) — the same shape as `data/atlas/tau_h_rederive`.
# --------------------------------------------------------------------------- #
STORE_REL = "data/morph_embed_cache/morph_clip_v1.mec"

# --------------------------------------------------------------------------- #
# Mirrored identity constants. Each is pinned to its owner by a test in
# test_morph_embed_cache.py; NONE of them may be edited without bumping SCHEMA_TAG,
# because every one of them changes the vector a key stands for.
# --------------------------------------------------------------------------- #
SCHEMA_TAG = "morph_embed/v1"

# geometry — tools/wallpaper/library_annotate.py (W, H, SS)
W, H, SS = 640, 360, 2
FIELD_MODE = "smooth"

# grayscale transfer — library_annotate.MORPH_PRODUCER / MORPH_K / MORPH_MAD_SCALE
MORPH_PRODUCER = "robustz_tanh_k2_v1"
MORPH_K = 2.0
MORPH_MAD_SCALE = 1.4826

# embedder — tools/curation/colored_clip.CLIP_MODEL
CLIP_MODEL = "vit_base_patch16_clip_224.openai"

# The preprocessing fields of timm's pretrained cfg for CLIP_MODEL that actually decide the
# pixels the model sees. Digested (not listed) into the key so the key stays short; the field
# LIST is here so a reader can see what is covered, and `clip_preproc_digest` recomputes it
# from timm in the test. Deliberately excludes `license`/`notes`/`url`/`hf_hub_id`: those move
# without changing a single vector, and invalidating an hours-long store on a docstring edit
# is how a cache gets turned off.
CLIP_PREPROC_FIELDS = ("input_size", "interpolation", "mean", "std",
                       "crop_pct", "crop_mode", "fixed_input_size")
# blake2b-8 of the canonicalized cfg subset, recorded 2026-08-03 against timm's registry.
# `test_clip_preproc_digest_still_matches_timm` recomputes it live and goes RED on a timm
# release that re-specifies this model's transform — at which point the fix is to bump
# SCHEMA_TAG (every cached vector is stale), never to re-baseline this string.
CLIP_PREPROC_DIGEST = "69973089c56af81d"


def clip_preproc_digest(model_name: str = CLIP_MODEL) -> str:
    """Recompute the preprocessing digest from timm's registry (imports timm — test-side)."""
    import timm
    cfg = timm.get_pretrained_cfg(model_name)
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
    parts = [f"{k}={d.get(k)!r}" for k in CLIP_PREPROC_FIELDS]
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def embedder_tag() -> str:
    """The `clip=` component: what produced the vector, independent of what it looked at."""
    return f"{CLIP_MODEL}:{CLIP_PREPROC_DIGEST}"


def morph_key(loc, *, w: int = W, h: int = H, ss: int = SS, maxiter_policy=None) -> str:
    """The cache key for one location's morph embedding. See the module doc for each part.

    Pure and cheap (no torch, no render), so it is safe to call once per row on a fully-warm
    pass — which is the pass this whole module exists to make free."""
    return "|".join([
        SCHEMA_TAG,
        f"loc={loc_mod.location_key(loc)}",
        f"maxiter={int(loc.maxiter)}",
        f"field={store.field_stem(loc, FIELD_MODE, w, h, ss, maxiter_policy)}",
        f"geom={w}x{h}ss{ss}",
        f"src={loc_mod.field_source_token(None) or loc_mod.BEAUTIFUL_SOURCE}",
        f"gray={MORPH_PRODUCER}:k{MORPH_K:g}:mad{MORPH_MAD_SCALE:g}",
        f"clip={embedder_tag()}",
    ])


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
FILE_HEADER = b"MORPH-EMBED-CACHE v1\n"
REC_MAGIC = b"MREC"
_REC_HDR = struct.Struct("<4sIII")          # magic, klen, dim, crc32
_HDR_LEN = _REC_HDR.size                    # 16


class CacheError(RuntimeError):
    """The store on disk is not a morph-embed cache (wrong header). Never a silent reset:
    the one thing that must not happen is overwriting somebody else's file."""


class MorphEmbedCache:
    """Persistent key -> float32 vector store. Hit -> reuse; miss -> compute + append.

    Usage is deliberately not a context manager only, because the cutter holds one open
    across a whole pass and reports its counters afterwards::

        cache = MorphEmbedCache().open()
        v = cache.get(key)
        if v is None:
            v = expensive(...)
            cache.put(key, v)
        cache.close()
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path(paths.bulk(STORE_REL))
        self._index: dict[str, tuple[int, int]] = {}   # key -> (offset into _blob, dim)
        self._blob: bytes = b""
        self._new: dict[str, np.ndarray] = {}
        self._fh = None
        self.opened = False
        # counters — every one of these is reported, because a cache that cannot say what it
        # did is indistinguishable from one that did nothing.
        self.hits = 0
        self.misses = 0
        self.appends = 0
        self.records_loaded = 0
        self.bytes_dropped = 0
        self.truncated_at = None
        self.created = False

    # -- open / scan / recover ---------------------------------------------- #
    def open(self):
        """Scan the store, verify every record, recover a damaged tail, index the rest."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "wb") as f:
                f.write(FILE_HEADER)
                f.flush()
                os.fsync(f.fileno())
            self.created = True
        raw = self.path.read_bytes()
        if not raw.startswith(FILE_HEADER):
            raise CacheError(
                f"{self.path} does not start with the morph-embed-cache header. Refusing to "
                f"treat it as one (it would be appended to and then truncated). Move it "
                f"aside if it is really scratch.")
        good_end = len(FILE_HEADER)
        pos = good_end
        n = len(raw)
        index: dict[str, tuple[int, int]] = {}
        while pos < n:
            if n - pos < _HDR_LEN:
                break                                   # torn header
            magic, klen, dim, crc = _REC_HDR.unpack_from(raw, pos)
            if magic != REC_MAGIC or dim == 0 or klen == 0:
                break                                   # desync / garbage
            end = pos + _HDR_LEN + klen + 4 * dim
            if end > n:
                break                                   # torn payload
            payload = raw[pos + _HDR_LEN:end]
            if zlib.crc32(payload) != crc:
                break                                   # bit rot / interrupted write
            try:
                key = payload[:klen].decode("utf-8")
            except UnicodeDecodeError:
                break
            index[key] = (pos + _HDR_LEN + klen, dim)    # last writer wins
            pos = end
            good_end = end
        if good_end != n:
            # SELF-HEAL. Truncating is safe (the store is bulk — regenerable) and is what
            # makes the next append land at a record boundary instead of after garbage.
            self.bytes_dropped = n - good_end
            self.truncated_at = good_end
            with open(self.path, "r+b") as f:
                f.truncate(good_end)
                f.flush()
                os.fsync(f.fileno())
            raw = raw[:good_end]
        self._blob = raw
        self._index = index
        self.records_loaded = len(index)
        self._fh = open(self.path, "ab")
        self.opened = True
        return self

    # -- read / write -------------------------------------------------------- #
    def get(self, key: str):
        """The cached vector for `key`, or None. Counts a hit/miss either way."""
        v = self._new.get(key)
        if v is None:
            ent = self._index.get(key)
            if ent is not None:
                off, dim = ent
                v = np.frombuffer(self._blob, dtype="<f4", count=dim, offset=off)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return v

    def put(self, key: str, vec) -> None:
        """Append `vec` under `key`, durably. A re-put of a present key is a no-op — the
        vector is a pure function of the key, so a second value for one key would mean the
        key is missing an axis, and appending it would hide that."""
        if not self.opened:
            raise CacheError("put() before open()")
        if key in self._index or key in self._new:
            return
        a = np.ascontiguousarray(np.asarray(vec, dtype="<f4").reshape(-1))
        if a.size == 0:
            raise ValueError("refusing to cache an empty vector")
        kb = key.encode("utf-8")
        payload = kb + a.tobytes()
        rec = _REC_HDR.pack(REC_MAGIC, len(kb), a.size, zlib.crc32(payload)) + payload
        self._fh.write(rec)                    # ONE write, then fsync: no torn hit
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._new[key] = a
        self.appends += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.opened = False

    def __enter__(self):
        return self.open() if not self.opened else self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- reporting ----------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._index) + len(self._new)

    def report(self) -> dict:
        return dict(path=str(self.path), created=self.created,
                    records_at_open=self.records_loaded, records_now=len(self),
                    hits=self.hits, misses=self.misses, appends=self.appends,
                    bytes_dropped=self.bytes_dropped, truncated_at=self.truncated_at,
                    size_bytes=(self.path.stat().st_size if self.path.exists() else 0))


def wrap(embed, cache: MorphEmbedCache, key_of):
    """Turn a row-embedder into a cached one: `key_of(row) -> str`, hit -> reuse, miss ->
    compute + append. The wrapper does NOT swallow exceptions — a row the embedder cannot
    reach must still reach `stage_morph_dedup`'s per-row counter, which is what keeps "we
    could not measure this" distinct from "this is a duplicate"."""
    def cached(row):
        k = key_of(row)
        v = cache.get(k)
        if v is not None:
            return v
        v = embed(row)
        if v is not None:
            cache.put(k, v)
        return v
    return cached


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stat", help="open the store, report its contents and any recovery")
    ap.parse_args()
    import json
    c = MorphEmbedCache().open()
    print(json.dumps(c.report(), indent=2))
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
