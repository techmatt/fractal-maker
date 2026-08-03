#!/usr/bin/env python
"""`tools/scoring/partitions.py` is the ONLY copy of the fractal_type ⟷ partition map.

Two things, and the second is the one that matters. The first is that the map is internally
coherent (injective, total over `ALL_FAMS`, a real inverse). The second is a **source scan**:
no tracked Python file outside this module may define `FT2FAM` / `FAM2FT` as a literal again.

Why a source scan rather than trust. On 2026-08-02 these nine pairs had seven literal copies
across `derive_t_good_{v7,v8}`, `keeper_cut.py`, `v7/build_manifest.py` and three test files —
none of them wrong, all of them independently editable, and one of them (`keeper_cut.py`'s)
carrying a comment that said it mirrored another. A duplicated constant does not fail when it
is created; it fails years later when someone adds a family to one copy. The scan is what
turns "please import it" into a gate.

  uv run pytest tools/scoring/test_partitions.py -q
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import partitions as P  # noqa: E402

OWNER = "tools/scoring/partitions.py"
# This file quotes the copy-shapes it hunts for (in `test_the_scan_would_actually_catch_a_copy`),
# so it exempts itself — the exemption list is exactly two files and both are named here.
EXEMPT = {OWNER, "tools/scoring/test_partitions.py"}
# `NAME = {` — a literal dict binding. `FT2FAM = {v: k for ...}` (a derived inverse) is
# matched too, on purpose: the inverse also has exactly one correct home.
LITERAL = re.compile(r"^\s*(FT2FAM|FAM2FT)\s*(:[^=]*)?=\s*\{", re.M)


def test_the_map_is_injective_and_its_inverse_is_real():
    assert len(set(P.FT2FAM.values())) == len(P.FT2FAM), "two fractal_types share a partition"
    assert P.FAM2FT == {v: k for k, v in P.FT2FAM.items()}
    for ft, fam in P.FT2FAM.items():
        assert P.FAM2FT[fam] == ft


def test_every_partition_is_in_ALL_FAMS_and_vice_versa():
    """`ALL_FAMS` is what the derivations walk to stamp partitions with no eval rows. A
    partition reachable in production but missing from that list would be silently absent
    from every t_good table rather than stamped UNCALIBRATED."""
    assert set(P.ALL_FAMS) == set(P.FT2FAM.values())
    assert len(P.ALL_FAMS) == len(set(P.ALL_FAMS)), "ALL_FAMS has a duplicate"


def test_julia_planes_are_namespaced_and_native_ones_are_not():
    """The one substantive thing the map does: keep `julia_multibrot4` (a julia-plane view)
    from colliding with `multibrot4` (its native twin). They are different supply, different
    scarcity and different objectives, so they must be different partitions."""
    for d in (3, 4, 5):
        assert P.FT2FAM[f"julia_multibrot{d}"] == f"julia:multibrot{d}"
        assert P.FT2FAM[f"multibrot{d}"] == f"multibrot{d}"
    assert P.FT2FAM["julia"] == "julia:mandelbrot"


def test_partition_of_default_is_explicit_at_the_call_site():
    assert P.partition_of("julia_multibrot4") == "julia:multibrot4"
    assert P.partition_of("not_a_family") is None                 # keeper_cut's convention
    assert P.partition_of("not_a_family", "not_a_family") == "not_a_family"   # derivations'


def _tracked_python():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [p for p in out.stdout.splitlines() if p.strip()]


def test_no_second_literal_copy_of_the_map_exists():
    """THE point of this file. A second literal is a fork with a delayed fuse."""
    offenders = []
    for rel in _tracked_python():
        if rel.replace("\\", "/") in EXEMPT:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in LITERAL.finditer(text):
            line = text[:m.start()].count("\n") + 1
            offenders.append(f"{rel}:{line}")
    assert not offenders, (
        f"{len(offenders)} literal redefinition(s) of the partition map outside {OWNER}: "
        f"{offenders}. Import it instead — `from partitions import FT2FAM` with "
        f"tools/scoring on sys.path.")


def test_the_scan_would_actually_catch_a_copy(tmp_path):
    """Non-vacuity: the regex has to match the shape a copy is actually written in. All four
    real spellings from the pre-2026-08-02 tree, plus the derived inverse."""
    samples = [
        'FT2FAM = {\n    "mandelbrot": "mandelbrot",\n}',
        'FT2FAM = {"mandelbrot": "mandelbrot", "julia": "julia:mandelbrot"}',
        'FAM2FT = {\n    "mandelbrot": "mandelbrot",\n}',
        'FT2FAM = {v: k for k, v in FAM2FT.items()}',
        'FT2FAM: dict = {"mandelbrot": "mandelbrot"}',
    ]
    for s in samples:
        assert LITERAL.search(s), f"the scan would miss this copy:\n{s}"
    # ...and does not fire on the legitimate uses that must stay legal
    for ok in ("from partitions import FT2FAM", "part = FT2FAM.get(ft)",
               "live = set(kc.FT2FAM.values())", "# mirrors FT2FAM = {...}\n"):
        assert not LITERAL.search(ok), f"the scan false-positives on:\n{ok}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
