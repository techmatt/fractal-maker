"""The morph-embed cache: the KEY must cover every axis, and a killed write must not poison it.

Three families here, failing for different reasons:

  * the KEY-AXIS tests. A cache key that omits an axis returns a vector computed under a
    different render or a different embedder, and nothing goes red — the whole class of bug
    this store can introduce. So each axis is moved on its own and the key is asserted to
    move with it, and the mirrored constants are pinned to the modules that own them
    (`verification_practice.md` §1.8 — two copies exist because nothing structural keeps them
    equal, so a test does).
  * the DURABILITY tests. Interrupted writes are simulated by truncating and by corrupting
    bytes, and the store is required to recover the prefix and REPORT the loss — a silent
    shrink is the failure that would be invisible.
  * the STORAGE-CLASS test. The store is bulk and must resolve out-of-tree.

Run: uv run pytest tools/wallpaper/test_morph_embed_cache.py -q
"""
from __future__ import annotations

import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import morph_embed_cache as M  # noqa: E402
from tools.corpus import location as loc_mod  # noqa: E402


def _loc(**kw):
    base = dict(family="mandelbrot", cx="-0.5", cy="0.0", fw="3.0", maxiter=1000)
    base.update(kw)
    return loc_mod.Location(**base)


# --------------------------------------------------------------------------- #
# the key covers every axis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,a,b", [
    ("cx", "-0.5", "-0.50000000000001"),
    ("cy", "0.0", "1e-30"),
    ("fw", "3.0", "3.0000000001"),        # fw is the axis a location-only key drops
    ("maxiter", 1000, 1001),
])
def test_location_axes_each_move_the_key(field, a, b):
    """join_key-grade identity, one axis at a time. `fw` is called out by name because a
    'same centre' key that ignores frame width serves a zoomed-in crop's vector for a
    zoomed-out one, and both are real rows in the same sitting."""
    assert M.morph_key(_loc(**{field: a})) != M.morph_key(_loc(**{field: b}))


def test_family_and_family_constants_move_the_key():
    """Two locations at identical coordinates on different planes are different pictures."""
    m = M.morph_key(_loc())
    j = M.morph_key(_loc(family="julia", c_re="0.3", c_im="0.5"))
    j2 = M.morph_key(_loc(family="julia", c_re="0.3", c_im="0.6"))
    assert len({m, j, j2}) == 3
    ph = M.morph_key(_loc(family="phoenix", c_re="0.5667", c_im="0.0",
                          family_params={"p_re": "-0.5", "p_im": "0.0"}))
    ph2 = M.morph_key(_loc(family="phoenix", c_re="0.5667", c_im="0.0",
                           family_params={"p_re": "-0.4", "p_im": "0.0"}))
    assert ph != ph2, "a declared family constant must enter the key"


def test_presentation_geometry_moves_the_key():
    """The crop presentation parameters: a 640x360 ss2 field and a 640x360 ss1 field are
    different pixels through the same transfer."""
    keys = {M.morph_key(_loc(), w=w, h=h, ss=ss)
            for w, h, ss in [(640, 360, 2), (640, 360, 1), (320, 180, 2), (640, 384, 2)]}
    assert len(keys) == 4


def test_maxiter_policy_moves_the_key():
    """The iteration-cap POLICY is a field-identity axis (`auto_maxiter.md`); it rides in
    through `field_stem` rather than being restated here, which is the point."""
    legacy = loc_mod.LEGACY_MAXITER_POLICY
    other = (legacy[0] + 1, legacy[1], legacy[2], legacy[3])
    assert M.morph_key(_loc(), maxiter_policy=legacy) != \
        M.morph_key(_loc(), maxiter_policy=other)


def test_embedder_and_transfer_tags_are_in_the_key():
    """The two axes that change every vector without touching a single coordinate."""
    k = M.morph_key(_loc())
    assert M.MORPH_PRODUCER in k and M.CLIP_MODEL in k and M.CLIP_PREPROC_DIGEST in k
    assert M.SCHEMA_TAG in k


def test_key_is_stable_across_calls():
    assert M.morph_key(_loc()) == M.morph_key(_loc())


# --------------------------------------------------------------------------- #
# the mirrored constants are pinned to their owners
# --------------------------------------------------------------------------- #
def test_geometry_and_transfer_pinned_to_library_annotate():
    """Mirrored, not imported (see the module doc). Nothing structural keeps them equal."""
    from tools.wallpaper import library_annotate as la
    assert (M.W, M.H, M.SS) == (la.W, la.H, la.SS)
    assert M.MORPH_PRODUCER == la.MORPH_PRODUCER
    assert M.MORPH_K == la.MORPH_K and M.MORPH_MAD_SCALE == la.MORPH_MAD_SCALE


def test_clip_model_pinned_to_colored_clip():
    torch = pytest.importorskip("torch")          # colored_clip imports the GPU stack
    assert torch is not None
    sys.path.insert(0, str(ROOT / "tools" / "curation"))
    from tools.curation import colored_clip as cc
    assert M.CLIP_MODEL == cc.CLIP_MODEL
    assert (cc.W, cc.H, cc.SS) == (M.W, M.H, M.SS)


def test_clip_preproc_digest_still_matches_timm():
    """The committed digest vs timm's live registry for this model.

    RED means a timm release re-specified this model's preprocessing: every cached vector is
    stale. The fix is to bump SCHEMA_TAG (which invalidates the store by key), NEVER to
    re-baseline this string — re-baselining keeps serving the old vectors under the new
    transform, which is the exact silent-wrong-answer this key exists to prevent."""
    pytest.importorskip("timm")
    assert M.clip_preproc_digest() == M.CLIP_PREPROC_DIGEST


# --------------------------------------------------------------------------- #
# storage class
# --------------------------------------------------------------------------- #
def test_store_is_bulk_and_resolves_out_of_tree():
    import artifacts as A
    assert A.is_relocated(M.STORE_REL), "the store must relocate out of the working tree"
    assert Path(A.resolve(M.STORE_REL)) == Path(A.artifacts_root()) / M.STORE_REL
    # ...and an in-tree straggler (a writer that bypassed the resolver) would not be committed
    import subprocess
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", M.STORE_REL],
                             cwd=A.REPO_ROOT, capture_output=True).returncode == 0
    assert ignored, f"{M.STORE_REL} is NOT gitignored — a straggler would be committed"
    # the default path really is the resolved one, not a hand-built sibling
    assert Path(M.MorphEmbedCache().path) == Path(A.resolve(M.STORE_REL))


# --------------------------------------------------------------------------- #
# hit / miss / append
# --------------------------------------------------------------------------- #
def test_roundtrip_across_two_opens(tmp_path):
    p = tmp_path / "s.mec"
    v = np.arange(8, dtype=np.float32) / 3.0
    with M.MorphEmbedCache(p) as c:
        assert c.get("k") is None and c.misses == 1
        c.put("k", v)
    with M.MorphEmbedCache(p) as c:
        assert c.records_loaded == 1
        got = c.get("k")
        assert got is not None and np.allclose(got, v) and got.dtype == np.dtype("<f4")
        assert c.hits == 1 and c.appends == 0


def test_a_second_put_of_a_present_key_is_a_no_op(tmp_path):
    """One key, one vector, forever. A second value for one key would mean the key is missing
    an axis; appending it silently would hide exactly that."""
    p = tmp_path / "s.mec"
    with M.MorphEmbedCache(p) as c:
        c.put("k", np.ones(4, dtype=np.float32))
        c.put("k", np.zeros(4, dtype=np.float32))
        assert c.appends == 1
    with M.MorphEmbedCache(p) as c:
        assert np.allclose(c.get("k"), 1.0) and c.records_loaded == 1


def test_many_keys_survive_and_are_not_confused(tmp_path):
    p = tmp_path / "s.mec"
    vs = {f"key-{i}": np.full(5 + (i % 3), i, dtype=np.float32) for i in range(50)}
    with M.MorphEmbedCache(p) as c:
        for k, v in vs.items():
            c.put(k, v)
    with M.MorphEmbedCache(p) as c:
        assert c.records_loaded == 50
        for k, v in vs.items():
            assert np.array_equal(c.get(k), v), k


# --------------------------------------------------------------------------- #
# interrupted writes — proved by injection
# --------------------------------------------------------------------------- #
def _fill(p, n=12):
    with M.MorphEmbedCache(p) as c:
        for i in range(n):
            c.put(f"k{i}", np.full(6, i, dtype=np.float32))
    return p.stat().st_size


@pytest.mark.parametrize("chop", [1, 5, 17, 40])
def test_a_torn_tail_is_recovered_and_reported(tmp_path, chop):
    """Simulates a kill mid-append: the last record is short. The prefix must survive, the
    damage must be REPORTED (a store that silently shrank is the invisible failure), and the
    store must stay appendable — the next put has to land at a record boundary."""
    p = tmp_path / "s.mec"
    size = _fill(p)
    with open(p, "r+b") as f:
        f.truncate(size - chop)
    with M.MorphEmbedCache(p) as c:
        assert c.bytes_dropped > 0 and c.truncated_at is not None
        assert c.records_loaded == 11, "the 11 intact records must survive"
        assert c.get("k11") is None
        c.put("k11", np.full(6, 11, dtype=np.float32))
    with M.MorphEmbedCache(p) as c:
        assert c.records_loaded == 12 and c.bytes_dropped == 0
        assert np.allclose(c.get("k11"), 11.0)
        assert np.allclose(c.get("k0"), 0.0)


def test_a_corrupted_vector_byte_is_caught_by_the_crc(tmp_path):
    """The CRC covers the VECTOR, not just the key — a flipped float must not be served."""
    p = tmp_path / "s.mec"
    _fill(p, 3)
    raw = bytearray(p.read_bytes())
    raw[-3] ^= 0xFF                                  # inside the last record's payload
    p.write_bytes(bytes(raw))
    with M.MorphEmbedCache(p) as c:
        assert c.records_loaded == 2 and c.bytes_dropped > 0
        assert c.get("k2") is None


def test_a_garbage_record_header_does_not_desync_the_scan(tmp_path):
    """A length field that is garbage must stop the scan, not be trusted into a wild read."""
    p = tmp_path / "s.mec"
    _fill(p, 4)
    raw = bytearray(p.read_bytes())
    off = len(M.FILE_HEADER)
    # walk to the 3rd record and smash its magic
    for _ in range(2):
        _m, klen, dim, _c = M._REC_HDR.unpack_from(raw, off)
        off += M._HDR_LEN + klen + 4 * dim
    raw[off:off + 4] = b"XXXX"
    p.write_bytes(bytes(raw))
    with M.MorphEmbedCache(p) as c:
        assert c.records_loaded == 2 and c.truncated_at == off


def test_a_zero_length_store_is_created_with_its_header(tmp_path):
    p = tmp_path / "sub" / "s.mec"
    with M.MorphEmbedCache(p) as c:
        assert c.created and c.records_loaded == 0
    assert p.read_bytes() == M.FILE_HEADER


def test_a_foreign_file_is_refused_not_overwritten(tmp_path):
    """The one thing a self-healing store must never do is heal somebody else's file."""
    p = tmp_path / "s.mec"
    p.write_bytes(b"this is not a cache\n" * 10)
    with pytest.raises(M.CacheError):
        M.MorphEmbedCache(p).open()
    assert p.read_bytes().startswith(b"this is not a cache")


def test_put_survives_a_process_kill_after_fsync(tmp_path):
    """`put` returns only after fsync, so a record that a caller saw accepted is on disk.
    Simulated by re-reading the bytes from a SEPARATE handle mid-session — no buffered tail."""
    p = tmp_path / "s.mec"
    c = M.MorphEmbedCache(p).open()
    c.put("k", np.arange(4, dtype=np.float32))
    raw = p.read_bytes()                              # separate read, nothing flushed for us
    magic, klen, dim, crc = M._REC_HDR.unpack_from(raw, len(M.FILE_HEADER))
    assert magic == M.REC_MAGIC and dim == 4
    payload = raw[len(M.FILE_HEADER) + M._HDR_LEN:]
    assert zlib.crc32(payload) == crc
    c.close()


# --------------------------------------------------------------------------- #
# the wrapper
# --------------------------------------------------------------------------- #
def test_wrap_computes_once_then_never_again(tmp_path):
    p = tmp_path / "s.mec"
    calls = []

    def embed(row):
        calls.append(row["id"])
        return np.full(4, row["id"], dtype=np.float32)

    rows = [{"id": i % 3} for i in range(9)]
    with M.MorphEmbedCache(p) as c:
        f = M.wrap(embed, c, lambda r: f"key-{r['id']}")
        out = [f(r) for r in rows]
    assert len(calls) == 3, "each distinct key computed exactly once within a pass"
    assert all(np.allclose(o, r["id"]) for o, r in zip(out, rows))
    calls.clear()
    with M.MorphEmbedCache(p) as c:                   # SECOND pass: the whole point
        f = M.wrap(embed, c, lambda r: f"key-{r['id']}")
        out = [f(r) for r in rows]
        assert c.hits == 9 and c.appends == 0
    assert calls == [], "a warm pass must do ZERO embed work"
    assert all(np.allclose(o, r["id"]) for o, r in zip(out, rows))


def test_wrap_does_not_swallow_an_embedder_exception(tmp_path):
    """A row the embedder cannot reach must still reach the stage's per-row counter: 'we
    could not measure this' and 'this is a duplicate' are different facts."""
    p = tmp_path / "s.mec"
    def boom(row):
        raise RuntimeError("no field")
    with M.MorphEmbedCache(p) as c:
        f = M.wrap(boom, c, lambda r: "k")
        with pytest.raises(RuntimeError):
            f({})
        assert c.appends == 0


def test_wrap_does_not_cache_a_none(tmp_path):
    p = tmp_path / "s.mec"
    with M.MorphEmbedCache(p) as c:
        f = M.wrap(lambda r: None, c, lambda r: "k")
        assert f({}) is None
        assert c.appends == 0 and len(c) == 0
