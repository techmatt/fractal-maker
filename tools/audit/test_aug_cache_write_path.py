#!/usr/bin/env python
r"""End-to-end exercise of the aug_cache WRITE path (resolver -> Rust batch).

The write path had never been run for real, and the first classifier rebuild should
not be its first test. The path is two seams stitched together:

  1. `tools/v{5,6,7}/build_plan.emit_location` sets each plan row's `out` to
     `artifacts.resolve(<repo-relative>)`, which for a relocated family
     (`data/v7/aug_cache/...`) maps under ARTIFACTS_ROOT (out of the working tree).
  2. The Rust `v4-render-batch` executor writes each rendered JPEG to `spec.out`
     VERBATIM (src/v4_cache.rs) — it does no resolving of its own.

So the guarantee "a cache rebuild lands out-of-tree, never re-bombing the source
tree" depends entirely on step 1 feeding the right `out` to step 2. This test drives
both against a throwaway ARTIFACTS_ROOT with a 2-row plan (one Mandelbrot = settled
path, one julia_multibrot3 = smooth path), then asserts:

  * the JPEGs materialize under ARTIFACTS_ROOT/data/v7/aug_cache/... (routed), and
  * NOTHING appears under the in-tree data/v7/aug_cache (did not bomb the tree).

The reappearance tripwire's quiet/fire behavior is covered separately and cheaply in
test_relocated_artifacts.py; this is the heavy render half.

Requires a built release binary; SKIPS loudly if absent. Slow (real renders) -> opt-in.

  uv run pytest -m slow tools/audit/test_aug_cache_write_path.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN = REPO_ROOT / "target" / "release" / "fractal-generator.exe"
COLORMAPS = REPO_ROOT / "data" / "palettes" / "clean_colormaps.json"
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
import artifacts as A  # noqa: E402

# A relocated (out-of-tree) cache prefix — the v7 post-freeze cache that has never
# actually been written. Kept in lockstep with A.RELOCATED_PREFIXES.
REL_PREFIX = "data/v7/aug_cache"


def _plan_rows():
    """Two rows exercising both render families, `out` set exactly as emit_location does."""
    specs = [
        dict(cx="-0.743643887", cy="0.131825904", fw=3.0e-3, palette="magma", ss=1,
             filter="box", fractal_type="mandelbrot",
             _rel=f"{REL_PREFIX}/write_path_test/mb_0.jpg"),
        dict(cx="0.0", cy="0.0", fw=3.0, palette="magma", ss=1, filter="box",
             fractal_type="julia_multibrot3", c_re="-0.4", c_im="0.6",
             _rel=f"{REL_PREFIX}/write_path_test/jm3_0.jpg"),
    ]
    for s in specs:
        rel = s.pop("_rel")
        s["out"] = A.resolve(rel).as_posix()   # the write path under test (step 1)
    return specs


def test_aug_cache_write_routes_out_of_tree(tmp_path, monkeypatch):
    if not BIN.exists():
        pytest.skip(f"release binary not built ({BIN}); run `cargo build --release` first")
    assert COLORMAPS.exists(), f"colormap library missing: {COLORMAPS}"

    artroot = tmp_path / "fractal-maker-artifacts"
    monkeypatch.setenv(A.ARTIFACTS_ENV, str(artroot))

    rows = _plan_rows()
    # every out must resolve under the throwaway ARTIFACTS_ROOT, never in the repo tree
    # (compare in posix form — `out` is .as_posix(), matching emit_location)
    for s in rows:
        assert artroot.as_posix() in s["out"], s["out"]
        assert (REPO_ROOT / REL_PREFIX).as_posix() not in s["out"], s["out"]

    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps(s) + "\n" for s in rows), encoding="utf-8")

    proc = subprocess.run(
        [str(BIN), "v4-render-batch", "--plan", str(plan),
         "--colormaps", str(COLORMAPS), "--width", "64", "--height", "36",
         "--maxiter", "200"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**__import__("os").environ, A.ARTIFACTS_ENV: str(artroot)},
    )
    assert proc.returncode == 0, f"batch failed (rc={proc.returncode}):\n{proc.stderr}"

    # (1) rendered JPEGs landed under ARTIFACTS_ROOT
    landed = sorted((artroot / REL_PREFIX / "write_path_test").glob("*.jpg"))
    assert len(landed) == 2, f"expected 2 JPEGs under {artroot}, found {landed}\n{proc.stderr}"
    for f in landed:
        assert f.stat().st_size > 0, f"empty JPEG {f}"

    # (2) the in-tree relocated path was NOT bombed
    in_tree = REPO_ROOT / REL_PREFIX
    assert not in_tree.exists() or not any(in_tree.rglob("*")), (
        f"write bombed the working tree at {in_tree} — a plan `out` bypassed the resolver")
