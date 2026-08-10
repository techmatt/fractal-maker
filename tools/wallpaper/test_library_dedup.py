"""Tests for `library_dedup` — the identity-aware coordinate dedup under the location store.

GPU-free (pure identity logic — no torch, no render, no seeder subprocess): the coord dedup
honors julia `c` and phoenix z-plane scale, and accumulates within a batch.

The reconciliation-assert and c-plane-budget-knob halves of this file were `prospect_orchestrator`
tests and were deleted with it on 2026-08-10 (docs/design/retired.md, legacy A/B path).

Run either way:
  uv run pytest tools/wallpaper/test_library_dedup.py
  uv run python tools/wallpaper/test_library_dedup.py     # prints PASS/FAIL summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "tools" / "corpus"))

import library_dedup as dedup            # noqa: E402
from tools.corpus import location as loc_mod  # noqa: E402


# =========================================================================== #
# Fixtures.
# =========================================================================== #
def _record(family, cx, cy, fw, c=None, p=None):
    """A minimal store record (only the identity block the dedup index reads)."""
    return {"identity": {"family": family, "cx": str(cx), "cy": str(cy), "fw": str(fw),
                         "c": c, "p": p}}


def _store(tmp_path, records):
    p = tmp_path / "records.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _loc(family, cx, cy, fw, c_re=None, c_im=None, **fp):
    return loc_mod.Location(family=family, cx=str(cx), cy=str(cy), fw=str(fw), maxiter=1500,
                            c_re=c_re, c_im=c_im, family_params=fp)


# =========================================================================== #
# 1. Identity-aware coordinate dedup.
# =========================================================================== #
def test_cplane_proximity_scale_aware(tmp_path):
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("mandelbrot", "0.10", "0.20", "0.010")]))
    # within 0.5*min(fw) of the same spot -> dup; far -> not.
    assert idx.is_dup("mandelbrot", "0.1004", "0.2003", "0.010", None, None)
    assert not idx.is_dup("mandelbrot", "0.20", "0.20", "0.010", None, None)
    # a MUCH tighter incoming fw shrinks the tolerance (min(fw)): the same 0.004 offset that
    # merged at fw=0.01 no longer merges when the incoming frame is 100x tighter.
    assert not idx.is_dup("mandelbrot", "0.1004", "0.2003", "0.0001", None, None)


def test_julia_requires_matching_c(tmp_path):
    # Two julia locations sharing a z-plane viewport but DIFFERENT c are different fractals.
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("julia", "0.0", "0.0", "0.01", c={"re": "0.233", "im": "0.538"})]))
    same_c = dedup.coord_of_location(_loc("julia", "0.0", "0.0", "0.01",
                                          c_re="0.233", c_im="0.538"))
    diff_c = dedup.coord_of_location(_loc("julia", "0.0", "0.0", "0.01",
                                          c_re="-0.4", c_im="0.6"))
    assert idx.is_dup(*same_c)          # same viewport AND same c -> dup
    assert not idx.is_dup(*diff_c)      # same viewport, different c -> NOT a dup


def test_julia_multibrot_degree_and_c(tmp_path):
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("julia_multibrot3", "1.0", "1.0", "0.02", c={"re": "-0.387", "im": "-0.629"})]))
    # different render-family (degree) never collides even at identical coords+c
    assert not idx.is_dup("julia_multibrot4", "1.0", "1.0", "0.02", "-0.387", "-0.629")
    assert idx.is_dup("julia_multibrot3", "1.0001", "1.0001", "0.02", "-0.387", "-0.629")


def test_phoenix_scale_aware_zplane(tmp_path):
    # Phoenix carries the fixed Ushiki c, so its c always matches: the test reduces to a
    # SCALE-AWARE z-plane viewport proximity. A deep spot under a shallow neighbour must NOT
    # over-merge (the min(fw), not 1.5*max(fw), rule).
    ph_c = {"re": "0.5667", "im": "0.0"}
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("phoenix", "0.30", "0.40", "1.0", c=ph_c),      # shallow
        _record("phoenix", "0.300000", "0.400000", "1e-4", c=ph_c)]))  # deep, ~same center
    # a NEW deep spot 0.02 away from the shallow one: 0.5*min(fw)=0.5*1e-4 tolerance -> NOT a dup
    assert not idx.is_dup("phoenix", "0.32", "0.42", "1e-4", "0.5667", "0.0")
    # a new spot truly on top of the deep record (within 0.5*1e-4) IS a dup
    assert idx.is_dup("phoenix", "0.3000003", "0.4000002", "1e-4", "0.5667", "0.0")
    # and the shallow record still merges a nearby shallow spot (its own scale)
    assert idx.is_dup("phoenix", "0.31", "0.41", "1.0", "0.5667", "0.0")


def test_within_batch_accumulation():
    # add_location makes a second same-spot q3 in the SAME cycle collapse onto the first.
    idx = dedup.StoreIndex()
    a = _loc("mandelbrot", "0.5", "0.5", "0.01")
    assert not idx.is_location_dup(a)   # empty index
    idx.add_location(a)
    b = _loc("mandelbrot", "0.5001", "0.5001", "0.01")
    assert idx.is_location_dup(b)       # now a within-batch dup of `a`


# =========================================================================== #
# Standalone runner.
# =========================================================================== #
def _run_standalone():
    import tempfile
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    npass = 0
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS {name}")
            npass += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{npass}/{len(tests)} passed")
    return npass == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_standalone() else 1)
