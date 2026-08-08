#!/usr/bin/env python
r"""`crop-batch` seed→tile determinism and manifest replay.

Three properties, each a real end-to-end run of the release binary (no fixtures, no
injected dependency — `verification_practice.md` §1.10: a test that derives its expectation
from the code under test asserts `f(x) == f(x)`):

  1. **Seed → tile is a pure function.** Two invocations with the same seed tag emit
     byte-identical JPGs and identical draw geometry.
  2. **A different seed tag moves the draw.** The control for (1): without it, (1) passes
     on any implementation that ignores the seed entirely — including one that emits the
     identity crop 24 times. Asserted on the COUNT OF DISTINCT geometries, not merely on
     direction (§6).
  3. **A manifest row replays byte-identically**, and does so from the RECORDED geometry —
     `--replay` is run with a seed tag and draw bounds that would produce a different
     fan-out, so a replay that secretly re-drew would come out different.

Marked `slow`: each test spawns the engine, and a run of this file spawns 5 of them. That
is inside the 4-process cap (they are sequential), but it is seconds of render per test,
which is not default-lane material.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tools/corpus", "tools/scoring"):
    sys.path.insert(0, str(ROOT / sub))

import corpus_common as cc            # noqa: E402
import production_pins as pins        # noqa: E402

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
COLORMAPS = ROOT / "data" / "v10" / "colormaps.json"

pytestmark = pytest.mark.slow

# Two locations, one per colour path: a plain Mandelbrot (location profile — the
# `palette.lookup_linear(nu * density)` map) and a Phoenix (beautiful smooth — the
# per-crop percentile stretch). A determinism test on one path would not cover the other,
# and the beautiful path is the one with a frame-global stage in it.
LOCS = [
    {"loc_id": 0, "cx": "-0.746339", "cy": "0.112242", "fw": "0.000583",
     "fractal_type": "mandelbrot"},
    {"loc_id": 1, "cx": "0.0", "cy": "0.0", "fw": "2.5", "fractal_type": "phoenix",
     "c_re": "0.5667", "c_im": "0", "p_re": "-0.5", "p_im": "0"},
]


def _require_inputs():
    if not BIN.exists():
        pytest.skip(f"release binary missing: {BIN} (cargo build --release)")
    if not COLORMAPS.exists():
        pytest.skip(f"colormap library missing: {COLORMAPS}")


def write_locs(d: Path) -> Path:
    rows = []
    for r in LOCS:
        rows.append({**r, "maxiter": int(pins.auto_maxiter(float(r["fw"]))),
                     "maxiter_policy": "test"})
    p = d / "locations.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def run(d: Path, *extra: str) -> subprocess.CompletedProcess:
    """One engine invocation through the committed launch defaults."""
    proc = subprocess.run(
        [str(BIN), "crop-batch", "--colormaps", str(COLORMAPS), *extra],
        cwd=str(ROOT), env=cc.default_engine_env(),
        creationflags=cc.default_creationflags(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc


def emit(d: Path, tag: str, name: str, **kw) -> tuple[Path, Path, list]:
    lp = write_locs(d)
    root, mf = d / name, d / f"{name}.jsonl"
    args = ["--locations", str(lp), "--out-root", str(root), "--manifest", str(mf),
            "--seed-tag", tag, "--geoms", "3", "--aa", "aliased:point antialiased:lanczos3",
            "--palettes", "twilight_shifted blue_orange", "--no-resume", "--log-every", "100"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    run(d, *args)
    rows = [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()]
    return root, mf, rows


def digests(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.jpg"))}


def geoms_of(rows: list) -> set:
    return {(r["loc_id"], r["crop"]["geom"], r["crop"]["scale"], r["crop"]["shift_frac"],
             r["crop"]["shift_angle"]) for r in rows}


def test_same_seed_gives_byte_identical_tiles(tmp_path):
    """(1) seed -> tile is a pure function of (seed_tag, loc_id, slot)."""
    _require_inputs()
    _r1, _m1, rows1 = emit(tmp_path, "unit-A", "a")
    _r2, _m2, rows2 = emit(tmp_path, "unit-A", "b")
    d1, d2 = digests(tmp_path / "a"), digests(tmp_path / "b")

    assert d1, "no tiles emitted — the fixture cannot fail (verification_practice §6)"
    assert len(d1) == len(LOCS) * 3 * 2 * 2, f"expected 24 tiles, got {len(d1)}"
    assert set(d1) == set(d2), "slot filenames differ between identical-seed runs"
    assert d1 == d2, ("byte mismatch on " +
                      ", ".join(k for k in d1 if d1[k] != d2.get(k)))
    assert geoms_of(rows1) == geoms_of(rows2)
    # Both colour paths are actually exercised, or the test covers half of what it claims.
    assert {r["field"]["profile"] for r in rows1} == {"location", "beautiful_smooth"}


def test_a_different_seed_tag_moves_the_draw(tmp_path):
    """(2) THE CONTROL for (1). Without this, an implementation that ignores the seed —
    or emits the identity crop every time — passes the determinism test."""
    _require_inputs()
    _r1, _m1, rows1 = emit(tmp_path, "unit-A", "a")
    _r2, _m2, rows2 = emit(tmp_path, "unit-B", "c")

    # Geometry 0 is the identity in BOTH runs by construction; the jittered ones must move.
    def jitter(rows):
        return {(r["loc_id"], r["crop"]["geom"]): (r["crop"]["scale"], r["crop"]["shift_frac"])
                for r in rows if r["crop"]["geom"] > 0}

    j1, j2 = jitter(rows1), jitter(rows2)
    assert j1 and j2
    assert all(j1[k] != j2[k] for k in j1), "a jittered crop did not move with the seed tag"
    # Distinct-value count, not direction: 2 locations x 2 jittered geoms = 4 draws, and a
    # sampler that drew one value and reused it would still differ from run A.
    assert len(set(j1.values())) == len(j1) == 4, f"draws are not independent: {j1}"
    for r in rows1 + rows2:
        if r["crop"]["geom"] == 0:
            assert (r["crop"]["scale"], r["crop"]["shift_frac"]) == (1, 0)

    assert digests(tmp_path / "a") != digests(tmp_path / "c")


def test_manifest_row_replays_byte_identically(tmp_path):
    """(3) A tile regenerates from its manifest row alone.

    Replay is invoked with a DIFFERENT seed tag and draw bounds than the forward run, so a
    replay that re-drew instead of reading the recorded geometry produces different tiles
    and this goes red."""
    _require_inputs()
    root, mf, rows = emit(tmp_path, "unit-A", "a")
    before = digests(root)
    assert before

    run(tmp_path, "--replay", str(mf), "--replay-out-root", str(tmp_path / "replay"),
        "--seed-tag", "totally-different", "--scale-lo", "0.5", "--scale-hi", "1.0",
        "--shift-frac-max", "0.0", "--jpg-quality-lo", "60", "--jpg-quality-hi", "60")
    after = digests(tmp_path / "replay")

    assert set(after) == set(before), (
        f"replay emitted {len(after)} tiles for {len(before)} manifest slots")
    bad = [k for k in before if before[k] != after[k]]
    assert not bad, f"{len(bad)} replayed tile(s) differ: {bad[:4]}"

    # And the replay honoured the RECORDED per-tile quality, not the flag's 60.
    assert {r["jpg_quality"] for r in rows} - {60}, "fixture never exercised a quality != 60"

    # PROVE IT RED (§3): perturb one row's recorded crop origin by a single field subpixel
    # and the replayed tile must change. Without this the equality above passes for any
    # replay that reads the geometry, ignores it, and happens to redraw the same thing —
    # and, more to the point, it passes on a comparison that is not actually comparing.
    row = dict(rows[0])
    key = str(Path(row["out"]).parent.name) + "/" + Path(row["out"]).name
    row["crop"] = {**row["crop"], "src_x0": row["crop"]["src_x0"] + 1.0}
    one = tmp_path / "one.jsonl"
    one.write_text(json.dumps(row) + "\n", encoding="utf-8")
    run(tmp_path, "--replay", str(one), "--replay-out-root", str(tmp_path / "moved"))
    moved = digests(tmp_path / "moved")
    assert len(moved) == 1
    assert moved[key] != before[key], (
        "a 1-subpixel shift of the RECORDED crop origin produced a byte-identical tile — "
        "the replay is not reading the manifest geometry, so the equality above is vacuous")


def test_extend_smaller_than_the_draw_is_a_loud_error(tmp_path):
    """The containment precondition fails OUT LOUD rather than emitting edge-clamped tiles.

    `build_taps_scaled` clamps at the source bounds, so an under-extended field would
    silently produce a smeared tile that still looks like a fractal — exactly the
    green-and-useless shape §1 warns about."""
    _require_inputs()
    lp = write_locs(tmp_path)
    proc = subprocess.run(
        [str(BIN), "crop-batch", "--colormaps", str(COLORMAPS), "--locations", str(lp),
         "--out-root", str(tmp_path / "x"), "--manifest", str(tmp_path / "x.jsonl"),
         "--extend", "1.05", "--scale-hi", "1.10", "--shift-frac-max", "0.05"],
        cwd=str(ROOT), env=cc.default_engine_env(),
        creationflags=cc.default_creationflags(), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "cannot contain" in proc.stderr, proc.stderr[-2000:]
    assert not list((tmp_path / "x").rglob("*.jpg")) if (tmp_path / "x").exists() else True


def test_limit_stamps_every_row_incomplete(tmp_path):
    """A bounded run that WRITES must stamp itself unusable (CLAUDE.md's bounded-end-to-end
    rule), and the stamp must be derived from the flag rather than hardcoded — so the
    unbounded run in the same fixture must carry `false`."""
    _require_inputs()
    _r, _m, full = emit(tmp_path, "unit-A", "a")
    assert all(r["batch_incomplete"] is False for r in full)

    _r2, _m2, part = emit(tmp_path, "unit-A", "lim", limit=1)
    assert part, "--limit 1 emitted no rows"
    assert {r["loc_id"] for r in part} == {LOCS[0]["loc_id"]}
    assert all(r["batch_incomplete"] is True for r in part)
