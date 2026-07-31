"""Tests for the canonical location + family-params slot (tools/corpus/location.py).

The linchpin is KEY-STABILITY: adding the general family-params slot must not move a
single existing mandelbrot/julia key, or it would break the corpus splits/manifest and
orphan every cached field. These tests pin that byte-identity, the cache-key
non-collision for the new families, the five-family render-one arg builder, the
render-block/sidecar parse round-trip (Phoenix's `p` survives), and that a synthetic
new-family location keys/loads without tripping the v5 manifest guard.

Run either way:
  uv run pytest tools/corpus/test_location.py
  uv run python tools/corpus/test_location.py     # prints PASS/FAIL summary (+ optional
                                                    binary-backed Phoenix round-trip)
"""
import hashlib
import itertools
import os
import subprocess
import sys
from pathlib import Path

_TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _TOOLS)
import _bootstrap  # noqa: E402,F401  (tools/{palettes,corpus,queries} on sys.path)

import location as loc_mod  # noqa: E402
import colormap as cm  # noqa: E402
import corpus_reader as cr  # noqa: E402
import assemble_queries as aq  # noqa: E402  (the field-cache key under test)
import query_sampler as qs  # noqa: E402

ROOT = Path(_TOOLS).parent


# --------------------------------------------------------------------------- #
# The pre-slot field-cache key formula, frozen here as the byte-identity oracle.
# --------------------------------------------------------------------------- #
def _old_field_key(ref):
    parts = [ref.kind, ref.cx, ref.cy, ref.fw, str(ref.maxiter),
             ref.c_re or "", ref.c_im or "",
             str(qs.CANDIDATE_SS), str(qs.EVAL_WIDTH), str(qs.EVAL_HEIGHT)]
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return f"{ref.kind}_{h}"


def _old_loc_key(ref):
    """The pre-slot dedup key (the 6-tuple query_sampler/query_batch_gen used)."""
    return (ref.kind, ref.cx, ref.cy, ref.fw, ref.c_re, ref.c_im)


def _sample_existing_locations(limit=200):
    """A representative sample of existing (mandelbrot+julia) corpus locations as
    canonical Locations, deduped by key."""
    out = {}
    for lc in itertools.islice(cr.iter_labeled(), limit * 4):
        canon = loc_mod.from_render_block(lc.render)
        out.setdefault(canon.key(), canon)
        if len(out) >= limit:
            break
    return list(out.values())


# --------------------------------------------------------------------------- #
# 1. Arg builder — the five-family location -> flags mapping.
# --------------------------------------------------------------------------- #
def test_render_one_flags_five_families():
    L = loc_mod.Location
    assert loc_mod.render_one_flags(
        L(family="mandelbrot", cx="0", cy="0", fw="3")) == ["--family", "mandelbrot"]

    assert loc_mod.render_one_flags(
        L(family="julia", cx="0", cy="0", fw="0.75", c_re="-0.8", c_im="0.156")) == \
        ["--family", "mandelbrot", "--julia", "--c", "-0.8", "0.156"]

    for n in ("multibrot3", "multibrot4", "multibrot5"):
        assert loc_mod.render_one_flags(L(family=n, cx="0", cy="0", fw="3")) == ["--family", n]

    assert loc_mod.render_one_flags(
        L(family="phoenix", cx="0", cy="0", fw="3", c_re="0.5667", c_im="0.0",
          family_params={"p_re": "-0.5", "p_im": "0.0"})) == \
        ["--family", "phoenix", "--c", "0.5667", "0.0", "--p", "-0.5", "0.0"]

    # Phoenix with the constants omitted (acceptance "test the Rust default" case):
    assert loc_mod.render_one_flags(L(family="phoenix", cx="0", cy="0", fw="3")) == \
        ["--family", "phoenix"]

    # Unknown family is a loud error, not a silent mandelbrot.
    try:
        loc_mod.render_one_flags(L(family="nope", cx="0", cy="0", fw="1"))
        assert False, "expected ValueError for unknown family"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# 2. Key-stability (HARD GATE) — every existing m/j key + field-cache filename
#    is byte-identical to the pre-slot scheme.
# --------------------------------------------------------------------------- #
def test_existing_keys_byte_identical():
    locs = _sample_existing_locations()
    assert locs, "no corpus locations found"
    for canon in locs:
        # identity key == the pre-slot 6-tuple, joined the same way (empty params append nothing)
        old = "|".join("" if v is None else str(v) for v in _old_loc_key(canon))
        assert canon.key() == old, (canon.family, canon.key(), old)
        # field-cache filename unchanged (this is what would orphan the cache if it moved).
        # Evaluated at the LEGACY cap policy: `_old_field_key` is the pre-slot formula,
        # which predates the cap axis, so the comparable key is the one the legacy policy
        # produces. That the LIVE policy moves the stem is deliberate and is asserted by
        # test_field_key_moves_with_cap_policy — not something this test should mask.
        assert aq._field_key(canon, maxiter_policy=loc_mod.LEGACY_MAXITER_POLICY) == \
            _old_field_key(canon), canon.key()


def test_existing_families_are_only_m_j():
    """Sanity: the current corpus is mandelbrot+julia only, so the sample above really
    is exercising the empty-params path."""
    fams = {canon.family for canon in _sample_existing_locations()}
    assert fams <= {"mandelbrot", "julia"}, fams


# --------------------------------------------------------------------------- #
# 3. Cache-key non-collision for the new families (and m/j still stable).
# --------------------------------------------------------------------------- #
def test_cache_key_noncollision():
    L = loc_mod.Location
    vp = dict(cx="0.0", cy="0.0", fw="3.0", maxiter=400)

    mand = L(family="mandelbrot", **vp)
    mb3 = L(family="multibrot3", **vp)
    assert mand.key() != mb3.key()
    assert aq._field_key(mand) != aq._field_key(mb3)

    ph_a = L(family="phoenix", c_re="0.5667", c_im="0.0",
             family_params={"p_re": "-0.5", "p_im": "0.0"}, **vp)
    ph_b = L(family="phoenix", c_re="0.5667", c_im="0.0",
             family_params={"p_re": "-0.4", "p_im": "0.0"}, **vp)   # differs ONLY in p
    assert ph_a.key() != ph_b.key()
    assert aq._field_key(ph_a) != aq._field_key(ph_b)

    # ...and the multibrot key/filename does NOT accidentally collide with a julia at
    # the same viewport carrying a c that stringifies into the same slot.
    jul = L(family="julia", c_re="0.0", c_im="0.0", **vp)
    assert len({mand.key(), mb3.key(), ph_a.key(), jul.key()}) == 4


# --------------------------------------------------------------------------- #
# 3b. Field-mode token — the render-mode / field-identity token that keeps a
#     strange pure-field dump (tia/stripe/…) from colliding with the cached
#     SMOOTH field. Parity gate: the smooth key is byte-identical to its
#     pre-token value (both beam + emit paths), and distinct modes key distinctly.
# --------------------------------------------------------------------------- #
def test_field_mode_token_semantics():
    # smooth (default / None / explicit "smooth") -> empty token; strange -> itself.
    assert loc_mod.field_mode_token(None) == ""
    assert loc_mod.field_mode_token("smooth") == ""
    assert loc_mod.field_mode_token("tia") == "tia"
    assert loc_mod.field_mode_token("stripe") == "stripe"


def test_field_source_token_semantics():
    # beautiful (default / None) -> empty token so smooth stems are unchanged;
    # f64 (the offset field) -> its own token so it keys disjointly.
    assert loc_mod.field_source_token(None) == ""
    assert loc_mod.field_source_token("beautiful") == ""
    assert loc_mod.field_source_token("f64") == "f64"


def test_maxiter_policy_token_semantics():
    """The iteration-CAP axis (docs/design/auto_maxiter.md).

    Same shape as the two tokens above: the LEGACY policy appends nothing, so every
    stem dumped before the 2026-07-31 raise is byte-identical and no cached field is
    orphaned by adding the axis; anything else keys disjointly."""
    assert loc_mod.maxiter_policy_token(loc_mod.LEGACY_MAXITER_POLICY) == ""
    assert loc_mod.LEGACY_MAXITER_POLICY == (500, 0.30, 200, 8000)
    # the ADOPTED policy is not the legacy one, so it carries a token
    live = loc_mod.current_maxiter_policy()
    assert live != loc_mod.LEGACY_MAXITER_POLICY, \
        "the cap raise did not land in active_ckpt — this test is vacuous"
    assert loc_mod.maxiter_policy_token(live) == "mi4000k0.3c200-67000"
    # every constant is in the token: perturb each of the four in turn, all distinct
    toks = {loc_mod.maxiter_policy_token(p) for p in (
        (4000, 0.30, 200, 67000), (4001, 0.30, 200, 67000),
        (4000, 0.31, 200, 67000), (4000, 0.30, 201, 67000),
        (4000, 0.30, 200, 67001))}
    assert len(toks) == 5


# Known input, frozen pre-token stems (computed on the live smooth path). These
# literals are the invariant: if a key edit moves the smooth stem, every cached
# field is orphaned — this test must fail before that ships.
#
# They are pinned at the LEGACY cap policy, which is what they were computed under
# and what every field on disk before 2026-07-31 was dumped at. Passing
# `maxiter_policy=LEGACY` is not a way to keep an old oracle green: it is the
# statement that the old fields keep their old names. The live policy MUST move the
# stem, and `test_field_key_moves_with_cap_policy` asserts exactly that.
_KNOWN = loc_mod.Location(family="mandelbrot", cx="-0.743643887",
                          cy="0.131825904", fw="1e-6", maxiter=2000)
_LEGACY = loc_mod.LEGACY_MAXITER_POLICY
_BEAM_SMOOTH = "mandelbrot_90d714081a89180f"
_EMIT_SMOOTH = "mandelbrot_3c882a9fb29412d4_2560x1440ss4"


def test_field_key_smooth_parity_beam():
    # default and explicit "smooth" both reproduce the frozen pre-token stem.
    assert aq._field_key(_KNOWN, maxiter_policy=_LEGACY) == _BEAM_SMOOTH
    assert aq._field_key(_KNOWN, "smooth", maxiter_policy=_LEGACY) == _BEAM_SMOOTH
    assert aq._field_key(_KNOWN, None, maxiter_policy=_LEGACY) == _BEAM_SMOOTH
    # distinct modes -> distinct stems, pairwise disjoint (incl. vs smooth).
    stems = {aq._field_key(_KNOWN, m, maxiter_policy=_LEGACY)
             for m in (None, "smooth", "tia", "stripe", "curvature")}
    assert len(stems) == 4  # {smooth, tia, stripe, curvature}
    assert _BEAM_SMOOTH in stems


def test_field_key_source_parity_beam():
    # default `beautiful` source (None / explicit) leaves the frozen smooth stem
    # byte-identical — no cached field is orphaned by adding the source axis.
    assert aq._field_key(_KNOWN, None, None, _LEGACY) == _BEAM_SMOOTH
    assert aq._field_key(_KNOWN, "smooth", "beautiful", _LEGACY) == _BEAM_SMOOTH
    # f64 source keys DISJOINTLY (its constant-offset field must not collide).
    assert aq._field_key(_KNOWN, None, "f64", _LEGACY) != _BEAM_SMOOTH
    # the source axis is orthogonal to the mode axis: smooth/beautiful, smooth/f64,
    # tia/beautiful, tia/f64 are four distinct stems.
    quad = {aq._field_key(_KNOWN, m, s, _LEGACY)
            for m in (None, "tia") for s in (None, "f64")}
    assert len(quad) == 4


def test_field_key_moves_with_cap_policy():
    """The §3 bracket, both sides.

    (a) an old-cap and a new-cap key must DIFFER — otherwise a field iterated to the
        old clipped cap is served silently under the raised one; and
    (b) two same-cap keys must be unchanged from today's value — the legacy policy
        still reproduces the frozen literal, so the axis orphans nothing.

    Run on both the beam and the emit stem, since they hash the parts in different
    orders and a token appended to only one of them is the failure this catches."""
    import importlib
    sys.path.insert(0, os.path.join(_TOOLS, "wallpaper"))
    ev = importlib.import_module("emit_v1")
    new = loc_mod.current_maxiter_policy()
    assert new != _LEGACY, "cap raise absent — the bracket below would be vacuous"

    # (a) old != new, on every stem builder that keys a field
    assert aq._field_key(_KNOWN, maxiter_policy=_LEGACY) != \
        aq._field_key(_KNOWN, maxiter_policy=new)
    assert ev._emit_field_stem(_KNOWN, maxiter_policy=_LEGACY) != \
        ev._emit_field_stem(_KNOWN, maxiter_policy=new)
    # ...and the LIVE default (no argument) is the new-cap key, not the old one —
    # without this the token could be plumbed but never actually reached.
    assert aq._field_key(_KNOWN) == aq._field_key(_KNOWN, maxiter_policy=new)
    assert ev._emit_field_stem(_KNOWN) == ev._emit_field_stem(_KNOWN, maxiter_policy=new)

    # (b) same cap -> same key, and equal to today's committed literal
    assert aq._field_key(_KNOWN, maxiter_policy=_LEGACY) == \
        aq._field_key(_KNOWN, maxiter_policy=_LEGACY) == _BEAM_SMOOTH
    assert ev._emit_field_stem(_KNOWN, maxiter_policy=_LEGACY) == \
        ev._emit_field_stem(_KNOWN, maxiter_policy=_LEGACY) == _EMIT_SMOOTH

    # the cap axis is orthogonal to mode and source: 2 policies x 2 modes x 2 sources
    # are eight distinct stems, so no combination aliases onto another.
    oct_ = {aq._field_key(_KNOWN, m, s, p)
            for p in (_LEGACY, new) for m in (None, "tia") for s in (None, "f64")}
    assert len(oct_) == 8


def test_field_key_smooth_parity_emit():
    import importlib
    sys.path.insert(0, os.path.join(_TOOLS, "wallpaper"))
    ev = importlib.import_module("emit_v1")   # lazy: keeps torch off the collection path
    K = dict(maxiter_policy=_LEGACY)
    assert ev._emit_field_stem(_KNOWN, **K) == _EMIT_SMOOTH
    assert ev._emit_field_stem(_KNOWN, "smooth", **K) == _EMIT_SMOOTH
    assert ev._emit_field_stem(_KNOWN, None, **K) == _EMIT_SMOOTH
    stems = {ev._emit_field_stem(_KNOWN, m, **K)
             for m in (None, "smooth", "tia", "stripe", "curvature")}
    assert len(stems) == 4
    assert _EMIT_SMOOTH in stems
    # field-SOURCE axis: default `beautiful` keeps the frozen emit stem byte-identical;
    # f64 keys disjointly, orthogonal to the mode axis.
    assert ev._emit_field_stem(_KNOWN, None, field_source=None, **K) == _EMIT_SMOOTH
    assert ev._emit_field_stem(_KNOWN, "smooth", field_source="beautiful", **K) == _EMIT_SMOOTH
    assert ev._emit_field_stem(_KNOWN, None, field_source="f64", **K) != _EMIT_SMOOTH
    quad = {ev._emit_field_stem(_KNOWN, m, field_source=s, **K)
            for m in (None, "tia") for s in (None, "f64")}
    assert len(quad) == 4


# --------------------------------------------------------------------------- #
# 4. Render-block / sidecar parse round-trip — the new families load correctly,
#    Phoenix's p survives parse -> key -> flags.
# --------------------------------------------------------------------------- #
def test_from_render_block_families():
    m = loc_mod.from_render_block({"cx": "0", "cy": "0", "fw": "3", "maxiter": 400})
    assert m.family == "mandelbrot" and m.c_re is None and m.family_params == ()

    j = loc_mod.from_render_block({"fractal_type": "julia", "cx": "0", "cy": "0",
                                   "fw": "0.75", "maxiter": 800,
                                   "c_re": "0.27", "c_im": "0.48"})
    assert j.family == "julia" and (j.c_re, j.c_im) == ("0.27", "0.48")

    p = loc_mod.from_render_block({"fractal_type": "phoenix", "cx": "0", "cy": "0",
                                   "fw": "3", "maxiter": 400, "c_re": "0.5667",
                                   "c_im": "0.0", "p_re": "-0.5", "p_im": "0.0"})
    assert p.family == "phoenix" and p.params == {"p_re": "-0.5", "p_im": "0.0"}


def test_sidecar_phoenix_p_survives():
    """render -> sidecar -> parse -> key -> re-render flags: Phoenix's p is never lost."""
    meta = {"location": {"kind": "phoenix", "cx": "0.0", "cy": "0.0", "fw": "3.0",
                         "maxiter": 400, "c_re": "0.5667", "c_im": "0.0",
                         "p_re": "-0.5", "p_im": "0.0"}}
    loc = loc_mod.from_sidecar(meta)
    assert loc.params == {"p_re": "-0.5", "p_im": "0.0"}
    # p survives into the key (now followed by the empty zm1_* slots, so it is no longer
    # the key's tail — assert it is present, not that it terminates the key).
    assert "|-0.5|0.0" in loc.key()
    flags = loc_mod.render_one_flags(loc)
    assert flags[-4:] == ["--p", "-0.5", "0.0"] or ("--p" in flags and "-0.5" in flags)
    # and the key is stable under a re-parse of an equivalent sidecar
    assert loc_mod.from_sidecar(meta).key() == loc.key()


# --------------------------------------------------------------------------- #
# 5. Coloring bridge — m/j Location -> LocationRef leaves the recipe byte-identical.
# --------------------------------------------------------------------------- #
def test_to_location_ref_recipe_stable():
    canon = loc_mod.Location(family="julia", cx="0.0", cy="0.0", fw="0.75",
                             maxiter=800, c_re="0.27", c_im="0.48")
    ref_direct = cm.LocationRef(kind="julia", cx="0.0", cy="0.0", fw="0.75",
                                maxiter=800, c_re="0.27", c_im="0.48")
    ref_bridge = loc_mod.to_location_ref(canon)
    # A CandidateConfig built from either serializes byte-identically (recipe schema).
    def _recipe(ref):
        return cm.CandidateConfig(palette="twilight", location=ref,
                                  eval_width=1024, eval_height=576).to_json()
    assert _recipe(ref_bridge) == _recipe(ref_direct)
    # An already-LocationRef passes straight through.
    assert loc_mod.to_location_ref(ref_direct) is ref_direct


# --------------------------------------------------------------------------- #
# 6. Manifest — the v5 guard still passes; a synthetic new-family location neither
#    changes the julia count nor trips assert_matches_v5.
# --------------------------------------------------------------------------- #
def test_manifest_untripped_by_new_family():
    pool = qs.LocationPool.from_corpus(verbose=False)
    base_v5_julia = pool.v5_julia_count()                       # julia_ladder_j0-scoped invariant
    base_julia_family = pool.family_counts().get("julia", 0)
    base_phoenix = pool.family_counts().get("phoenix", 0)       # delta-based: corpus may already
                                                                # hold phoenix locations (it now does)
    assert pool.assert_matches_v5() == base_v5_julia

    # Inject a synthetic phoenix location; it must load, key, and leave the guard green.
    ph = loc_mod.Location(family="phoenix", cx="0.0", cy="0.0", fw="3.0", maxiter=400,
                          c_re="0.5667", c_im="0.0",
                          family_params={"p_re": "-0.5", "p_im": "0.0"})
    pool.locations.append(qs.PooledLocation(ref=ph, scores=[3], batch_ids={"synthetic"}))
    assert pool.family_counts().get("phoenix", 0) == base_phoenix + 1   # exactly one added
    assert pool.family_counts().get("julia", 0) == base_julia_family   # julia family unchanged
    assert pool.v5_julia_count() == base_v5_julia               # v5-era julia count unchanged
    assert pool.assert_matches_v5() == base_v5_julia            # still green
    assert aq._field_key(ph)                                    # keys/loads fine


# --------------------------------------------------------------------------- #
# Optional binary-backed Phoenix round-trip (Step-6 ≤1 LSB): render -> dump-field ->
# sidecar -> color-in-Python == Rust ref. Script-only (skipped when the binary is
# absent), mirroring the reframe GPU check being kept out of the pytest gate.
# --------------------------------------------------------------------------- #
def _phoenix_acceptance():
    import colormap_acceptance as ca
    if not ca.BIN.exists():
        print("SKIP  phoenix_acceptance (release binary not built)")
        return True
    m = ca.run_gate("test_06", palette="twilight", filt="box", width=640, height=360, ss=2)
    ok = m["passed"]
    print(f"{'PASS' if ok else 'FAIL'}  phoenix_acceptance "
          f"(max_diff={m['max_diff']} frac_gt1={m['frac_gt1']:.2e})")
    return ok


def main():
    tests = [
        test_render_one_flags_five_families,
        test_existing_keys_byte_identical,
        test_existing_families_are_only_m_j,
        test_cache_key_noncollision,
        test_field_mode_token_semantics,
        test_field_source_token_semantics,
        test_maxiter_policy_token_semantics,
        test_field_key_smooth_parity_beam,
        test_field_key_source_parity_beam,
        test_field_key_moves_with_cap_policy,
        test_field_key_smooth_parity_emit,
        test_from_render_block_families,
        test_sidecar_phoenix_p_survives,
        test_to_location_ref_recipe_stable,
        test_manifest_untripped_by_new_family,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL  %s: %s" % (t.__name__, e))
    if not _phoenix_acceptance():
        failed += 1
    print("\n%d/%d python tests passed" % (len(tests) - failed if failed <= len(tests)
                                           else 0, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
