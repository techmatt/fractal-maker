"""Tests for the Python coloring tail (tools/colormap.py).

Run:  uv run python -m pytest tools/test_colormap.py -v

The reference-match test (the load-bearing one) shells out to the release binary; it
is skipped if the binary isn't built. The rest are pure-Python unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colormap as cm  # noqa: E402
import colormap_acceptance as acc  # noqa: E402


def _synthetic_field(h=48, w=64, ss=2, interior_frac=0.2, seed=0):
    """A deterministic super-res smooth field with a NaN interior blob."""
    rng = np.random.default_rng(seed)
    hs, ws = h * ss, w * ss
    vals = rng.uniform(3.0, 40.0, size=(hs, ws)).astype(np.float64)
    n_int = int(interior_frac * hs * ws)
    idx = rng.choice(hs * ws, size=n_int, replace=False)
    flat = vals.ravel()
    flat[idx] = np.nan
    loc = cm.LocationRef(kind="mandelbrot", cx="-0.75", cy="0.1", fw="0.01", maxiter=500)
    return cm.FieldData(values=flat.reshape(hs, ws), supersample=ss, location=loc)


@pytest.fixture(scope="module")
def library():
    return cm.PaletteLibrary()


# --------------------------------------------------------------------------- #
# Determinism — same (field, config) -> identical image.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("filt", ["box", "mitchell", "lanczos3"])
def test_determinism(library, filt):
    field = _synthetic_field()
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(palette="twilight", location=field.location,
                             eval_width=ow, eval_height=oh, filter=filt)
    a = cm.render_candidate(field, cfg, library)
    b = cm.render_candidate(field, cfg, library)
    assert np.array_equal(a, b)
    assert a.shape == (oh, ow, 3) and a.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Type dispatch — inapplicable params rejected on the wrong palette type.
# --------------------------------------------------------------------------- #

def test_type_dispatch_cyclic_only(library):
    field = _synthetic_field()
    ow, oh = field.out_size
    base = dict(location=field.location, eval_width=ow, eval_height=oh)
    # phase/n_cycles on a non-cyclic palette (magma = sequential) -> reject.
    for kw in ({"phase": 0.3}, {"n_cycles": 2}):
        cfg = cm.CandidateConfig(palette="magma", **base, **kw)
        with pytest.raises(ValueError, match="cyclic"):
            cm.render_candidate(field, cfg, library)
    # Same params on a cyclic palette (twilight) -> allowed.
    cfg = cm.CandidateConfig(palette="twilight", **base, phase=0.3, n_cycles=2)
    cm.render_candidate(field, cfg, library)


def test_type_dispatch_domain_checks(library):
    field = _synthetic_field()
    ow, oh = field.out_size
    base = dict(palette="twilight", location=field.location, eval_width=ow, eval_height=oh)
    # n_cycles=3 is a *valid* value on cyclic twilight (cf. test_type_dispatch_cyclic_only,
    # which allows n_cycles=2); the domain rule is "positive integer", so probe with 0.
    with pytest.raises(ValueError, match="n_cycles"):
        cm.render_candidate(field, cm.CandidateConfig(**base, n_cycles=0), library)
    with pytest.raises(ValueError, match="log_premap"):
        cm.render_candidate(field, cm.CandidateConfig(**base, log_premap="bogus"), library)
    with pytest.raises(ValueError, match="filter"):
        cm.render_candidate(field, cm.CandidateConfig(**base, filter="bogus"), library)


# --------------------------------------------------------------------------- #
# Recipe round-trip — CandidateConfig -> JSON -> back -> identical render.
# --------------------------------------------------------------------------- #

def test_recipe_roundtrip_json():
    loc = cm.LocationRef(kind="julia", cx="0.0", cy="0.0", fw="0.75", maxiter=800,
                         c_re="0.27", c_im="0.48")
    cfg = cm.CandidateConfig(palette="twilight", location=loc, eval_width=100, eval_height=60,
                             reverse=True, log_premap="log", gamma=1.5, phase=0.25,
                             n_cycles=2, interior_color=(0.1, 0.2, 0.3),
                             filter="lanczos3")
    back = cm.CandidateConfig.from_json(cfg.to_json())
    assert back == cfg


def test_recipe_roundtrip_render(library):
    field = _synthetic_field()
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(palette="twilight", location=field.location, eval_width=ow,
                             eval_height=oh, reverse=True, log_premap="log", gamma=1.3,
                             phase=0.2, n_cycles=2, filter="box")
    back = cm.CandidateConfig.from_json(cfg.to_json())
    a = cm.render_candidate(field, cfg, library)
    b = cm.render_candidate(field, back, library)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Interior fill — NaN pixels take the configured interior color.
# --------------------------------------------------------------------------- #

def test_interior_fill(library):
    # An all-interior field must render to a flat color = the interior fill.
    loc = cm.LocationRef(kind="mandelbrot", cx="0", cy="0", fw="0.01", maxiter=500)
    field = cm.FieldData(values=np.full((8, 8), np.nan), supersample=1, location=loc)
    cfg = cm.CandidateConfig(palette="twilight", location=loc, eval_width=8, eval_height=8,
                             interior_color=(1.0, 0.0, 0.0), filter="box")
    img = cm.render_candidate(field, cfg, library)
    # linear (1,0,0) -> sRGB (255,0,0).
    assert np.array_equal(img, np.tile([255, 0, 0], (8, 8, 1)))


# --------------------------------------------------------------------------- #
# LUT parity — a baked twilight LUT reproduces its control stops (OKLab interp).
# --------------------------------------------------------------------------- #

def test_lut_reproduces_stops(library):
    import json
    cms = {c["name"]: c for c in json.loads(Path("data/palettes/score3_colormaps.json").read_text())}
    stops = cms["twilight"]["stops"]
    lut = library.lut("twilight")
    for pos, rgb in stops[:8]:
        got = cm.lookup_linear(lut, np.array([pos % 1.0]))[0]
        want = cm.srgb_to_linear(np.asarray(rgb) / 255.0)
        assert np.max(np.abs(got - want)) < 5e-3, f"stop {pos}: {got} vs {want}"


# --------------------------------------------------------------------------- #
# LUT memo — the module-level cache is PURE: byte-identical to an uncached bake. #
# --------------------------------------------------------------------------- #

def test_lut_memo_byte_identical(library):
    """`build_lut` (memoized) must equal `_bake_lut` (uncached) exactly, and a render
    with the memo warm must equal one that re-bakes fresh — for varied palette + reverse
    + mirror. Any nonzero delta means the memo key is wrong (silent color corruption)."""
    import json
    cms = {c["name"]: c for c in json.loads(Path("data/palettes/score3_colormaps.json").read_text())}
    field = _synthetic_field()
    ow, oh = field.out_size
    # (a) LUT-level: memoized == fresh bake, both reverse states, both mirror states.
    for name in ("twilight", "magma", "viridis"):
        stops = [(p, rgb) for p, rgb in cms[name]["stops"]]
        for reverse in (False, True):
            for mirror in (False, True):
                cm._LUT_MEMO.clear()
                fresh = cm._bake_lut(stops, reverse=reverse, mirror=mirror)
                memo1 = cm.build_lut(stops, reverse=reverse, mirror=mirror)   # miss -> bake
                memo2 = cm.build_lut(stops, reverse=reverse, mirror=mirror)   # hit
                assert np.array_equal(fresh, memo1)
                assert memo1 is memo2                                         # cached object reused
    # (b) render-level: memo-warm == re-bake-every-render, across coloring knobs.
    configs = [
        dict(palette="twilight", reverse=False, gamma=1.0, phase=0.0, n_cycles=1),
        dict(palette="twilight", reverse=False, gamma=1.7, phase=0.35, n_cycles=2),
        dict(palette="magma", reverse=True, gamma=0.8, log_premap="log"),
        dict(palette="viridis", reverse=False, gamma=1.2),
    ]
    for kw in configs:
        cfg = cm.CandidateConfig(location=field.location, eval_width=ow, eval_height=oh, **kw)
        cm._LUT_MEMO.clear(); library._lut_cache.clear()
        warm = cm.render_candidate(field, cfg, library)                      # populates caches
        again = cm.render_candidate(field, cfg, library)                     # memo hit
        cm._LUT_MEMO.clear(); library._lut_cache.clear()
        fresh = cm.render_candidate(field, cfg, library)                     # cold re-bake
        assert np.array_equal(warm, again), kw
        assert np.array_equal(warm, fresh), kw


# --------------------------------------------------------------------------- #
# Streamed tail — the fast paths must be BYTE-IDENTICAL to the whole-array form.
#
# `render_candidate` shades and downsamples in row chunks, and each banded pass takes a
# basic-slice shortcut over the shift-invariant interior (`_uniform_span`). Both are pure
# speedups whose ONLY contract is that they change nothing, so the gate is a differential
# against a reference implementation kept here — the whole-array, no-fast-path code the
# optimization replaced (verification_practice.md §7 "differential over frozen literals",
# §8 "byte-identity only where ... depends on it": here it is the whole claim).
# --------------------------------------------------------------------------- #

def _ref_banded_pass(src, starts, weights):
    """Reference: every destination through the general gather, no uniform-span path."""
    N, src_len, C = src.shape
    dst_len, K = weights.shape
    out = np.zeros((N, dst_len, C), dtype=np.float64)
    for k in range(K):
        cols = np.clip(starts + k, 0, src_len - 1)
        out += src[:, cols, :] * weights[:, k][None, :, None]
    return out


def _ref_render_candidate(field, config, library, prep=None, profile=None):
    """Reference: the whole-array tail — one 3-D linear buffer, both passes over all of it."""
    cm.validate_config(config, library)
    if prep is None:
        prep = cm.stretch_field(field)
    x, valid = prep.x, prep.valid
    if config.transfer == "grad":
        if profile is None:
            profile = cm.gradient_transfer_profile(field, prep)
        base = cm._apply_transfer(x, profile, config.transfer_gamma)
    else:
        base = x
    gray = cm.apply_transform(base, config.log_premap, config.gamma)
    t = np.mod(gray * config.n_cycles, 1.0)
    t = np.mod(t + config.phase, 1.0)
    lut = library.lut(config.palette, reverse=config.reverse)
    t = np.mod(np.asarray(t, dtype=np.float64), 1.0)
    xi = t * cm.LUT_SIZE
    i0 = np.floor(xi).astype(np.int64)
    f = (xi - i0)[..., None]
    i0 = i0 % cm.LUT_SIZE
    i1 = (i0 + 1) % cm.LUT_SIZE
    linear = lut[i0] * (1.0 - f) + lut[i1] * f
    linear[~valid] = np.asarray(config.interior_color, dtype=np.float64)

    ss, name = field.supersample, config.filter
    Hs, Ws, _ = linear.shape
    out_h, out_w = Hs // ss, Ws // ss
    if name == "box":
        r = linear[: out_h * ss, : out_w * ss].reshape(out_h, ss, out_w, ss, 3).mean(axis=(1, 3))
        return cm._encode_srgb8(r)
    hstart, hw = cm._build_banded_taps(out_w, Ws, ss, name)
    vstart, vw = cm._build_banded_taps(out_h, Hs, ss, name)
    inter = _ref_banded_pass(linear, hstart, hw).astype(np.float32).astype(np.float64)
    out = _ref_banded_pass(np.transpose(inter, (1, 0, 2)), vstart, vw)
    return cm._encode_srgb8(np.transpose(out, (1, 0, 2)))


# Deliberately ragged so neither chunk loop divides evenly: at ss=2 the field is 106 rows
# against SHADE_CHUNK_ROWS, and the output is 53 rows against VPASS_BAND_ROWS.
_STREAM_KNOBS = [
    dict(palette="twilight", filter="lanczos3"),
    dict(palette="twilight", filter="mitchell"),
    dict(palette="twilight", filter="box"),
    dict(palette="twilight", filter="lanczos3", gamma=1.7, log_premap="log"),
    dict(palette="twilight", filter="lanczos3", reverse=True, interior_color=(0.2, 0.05, 0.4)),
    dict(palette="twilight", filter="lanczos3", n_cycles=3, phase=0.37),
    dict(palette="twilight", filter="lanczos3", phase=0.5),
    dict(palette="twilight", filter="lanczos3", transfer="grad", transfer_gamma=1.5),
    dict(palette="twilight", filter="lanczos3", transfer="grad", transfer_gamma=0.0),
    dict(palette="magma", filter="lanczos3"),
]


@pytest.mark.parametrize("ss", [1, 2, 4])
@pytest.mark.parametrize("kw", _STREAM_KNOBS, ids=lambda k: f"{k['filter']}-{len(k)}")
def test_streamed_tail_is_byte_identical_to_the_whole_array_form(library, ss, kw):
    field = _synthetic_field(h=53, w=71, ss=ss, seed=ss)
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(location=field.location, eval_width=ow, eval_height=oh, **kw)
    prep = cm.stretch_field(field)
    prof = cm.gradient_transfer_profile(field, prep) if kw.get("transfer") == "grad" else None
    got = cm.render_candidate(field, cfg, library, prep=prep, profile=prof)
    want = _ref_render_candidate(field, cfg, library, prep=prep, profile=prof)
    assert np.array_equal(got, want), (
        f"ss={ss} {kw}: max delta "
        f"{np.abs(got.astype(int) - want.astype(int)).max()}")


def test_the_uniform_span_fast_path_is_actually_taken(library):
    """Non-vacuity for the test above: if `_uniform_span` returned "no span" everywhere the
    differential would still pass while the fast path went dead. Assert it FIRES at the two
    production geometries, and that it covers the interior while excluding the truncated
    edge rows — a span of the whole axis would mean the edge detection is broken."""
    for src_len, ss in ((10240, 4), (1920, 2)):
        dst_len = src_len // ss
        starts, weights = cm._build_banded_taps(dst_len, src_len, ss, "lanczos3")
        lo, hi, W = cm._uniform_span(starts, weights, ss, src_len)
        assert W is not None, (src_len, ss)
        assert 0 < lo < hi < dst_len, (src_len, ss, lo, hi, dst_len)
        assert (hi - lo) / dst_len > 0.99, (src_len, ss, lo, hi)
        # the excluded rows are exactly the ones whose padded window runs off an end
        for d in list(range(lo)) + list(range(hi, dst_len)):
            assert starts[d] == 0 or starts[d] + weights.shape[1] > src_len, (d, starts[d])


def test_the_chunk_loops_actually_iterate(library):
    """Second non-vacuity leg: the fixture must be taller than BOTH chunk sizes, or the
    streamed path degenerates to one pass and covers none of the boundary handling."""
    field = _synthetic_field(h=53, w=71, ss=2)
    assert field.values.shape[0] > cm.SHADE_CHUNK_ROWS
    assert field.out_size[1] > cm.VPASS_BAND_ROWS


# --------------------------------------------------------------------------- #
# Reference-match — the headline gate (shells out to the release binary).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not acc.BIN.exists(), reason="release binary not built")
@pytest.mark.parametrize("test_id,filt", [
    ("test_01", "box"),        # mandelbrot
    ("test_01", "lanczos3"),   # mandelbrot, production filter
    ("test_03", "box"),        # julia
])
def test_reference_match(test_id, filt):
    m = acc.run_gate(test_id, filt=filt, width=480, height=270, ss=2)
    assert m["max_diff"] <= acc.TOL_MAX, m
    assert m["frac_gt1"] <= acc.TOL_FRAC_GT1, m


# --------------------------------------------------------------------------- #
# The SCORING-ONLY coarse path. `render_candidates_coarse` claims its k-th slice is
# byte-identical to `render_candidate_coarse` on the same config, and until now nothing
# asserted it — the pref pick has run on that claim since it was written. It matters twice
# over: the batched form takes a shared-t fast path AND memoizes the t-plane on the config's
# transform knobs, so a collapse that is wrong would silently colour K candidates alike and
# the palette ranker would pick by nothing.
# --------------------------------------------------------------------------- #
def _coarse(library, ss=2):
    prep = cm.stretch_field(_synthetic_field(h=24, w=32, ss=ss))
    return cm.coarse_field(prep, out_w=32, out_h=24)


_COARSE_PALETTES = ["twilight", "viridis", "magma", "cividis"]


def test_the_batched_coarse_recolor_matches_the_per_candidate_form(library):
    """The canonical pref-pick case: K configs differing in NOTHING but palette, which is
    exactly when the shared-t branch and the knob memo both fire."""
    coarse = _coarse(library)
    cfgs = [cm.CandidateConfig(palette=p, location=_synthetic_field().location,
                               eval_width=32, eval_height=24, filter="box")
            for p in _COARSE_PALETTES]
    got = cm.render_candidates_coarse(coarse, cfgs, library)
    assert got.shape == (len(cfgs), 24, 32, 3) and got.dtype == np.uint8
    for k, cfg in enumerate(cfgs):
        want = cm.render_candidate_coarse(coarse, cfg, library)
        assert np.array_equal(got[k], want), f"slice {k} ({cfg.palette}) differs"
    # Non-vacuity: distinct palettes must give distinct images, or "identical" is free.
    assert len({got[k].tobytes() for k in range(len(cfgs))}) == len(cfgs)


def test_the_coarse_knob_memo_never_collapses_two_different_t_planes(library):
    """The memo's failure mode is the dangerous one: a key too coarse would hand candidate B
    candidate A's plane and the batch would still look self-consistent. Every knob in the key
    gets its own config here, all on ONE palette, so any collapse shows up as two identical
    slices AND as a mismatch against the per-candidate form."""
    coarse = _coarse(library)
    loc = _synthetic_field().location
    base = dict(palette="twilight", location=loc, eval_width=32, eval_height=24, filter="box")
    cfgs = [
        cm.CandidateConfig(**base),
        cm.CandidateConfig(**base, gamma=1.7),
        cm.CandidateConfig(**base, log_premap="log"),
        cm.CandidateConfig(**base, n_cycles=3),
        cm.CandidateConfig(**base, phase=0.37),
    ]
    got = cm.render_candidates_coarse(coarse, cfgs, library)
    for k, cfg in enumerate(cfgs):
        assert np.array_equal(got[k], cm.render_candidate_coarse(coarse, cfg, library)), \
            f"slice {k} differs from the per-candidate form"
    assert len({got[k].tobytes() for k in range(len(cfgs))}) == len(cfgs), \
        "two knob settings produced the same image — the memo key is too coarse"

    # `transfer_gamma` is in the key but INERT under `transfer='pct'` (only `_apply_transfer`,
    # i.e. the grad path, reads it) — so it is a key that is finer than it has to be, never
    # coarser. Asserted rather than left as a surprise, because the leg above deliberately
    # requires every listed knob to move the image and this one does not.
    pct_g = cm.CandidateConfig(**base, transfer_gamma=0.5)
    assert np.array_equal(cm.render_candidate_coarse(coarse, pct_g, library), got[0])


def test_a_grad_transfer_config_is_kept_out_of_the_knob_memo(library):
    """`transfer='grad'` planes also depend on the per-config profile, which is NOT in the
    key, so two grad configs sharing the knob tuple must still get their own planes."""
    field = _synthetic_field(h=24, w=32)
    prep = cm.stretch_field(field)
    coarse = cm.coarse_field(prep, out_w=32, out_h=24)
    prof = cm.gradient_transfer_profile(field, prep)
    other = cm.gradient_transfer_profile(_synthetic_field(h=24, w=32, seed=3),
                                         cm.stretch_field(_synthetic_field(h=24, w=32, seed=3)))
    cfgs = [cm.CandidateConfig(palette="twilight", location=field.location, eval_width=32,
                               eval_height=24, filter="box", transfer="grad",
                               transfer_gamma=0.0) for _ in range(2)]
    got = cm.render_candidates_coarse(coarse, cfgs, library, profiles=[prof, other])
    for k, (cfg, p) in enumerate(zip(cfgs, [prof, other])):
        assert np.array_equal(got[k], cm.render_candidate_coarse(coarse, cfg, library, profile=p))
    with pytest.raises(ValueError):
        cm.render_candidates_coarse(coarse, cfgs[:1], library, profiles=None)
