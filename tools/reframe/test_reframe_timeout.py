"""`reframe_location`'s per-render backstop: default is byte-identical, bounded is safe.

`reframe_location` spawns up to 12 `render-one` processes per location and was the last
unbounded subprocess on the discovery path — its two siblings (`--expand` in
`steered_frontier`, `prescreen._render`) were bounded during the maneuvers work and this
one was left as a known residual gap. It is the way an unattended run hangs forever.

What these tests pin, in the order that matters:

  1. **The default changes nothing.** `timeout=None` reaches `subprocess.run` as `None`,
     so the call cannot raise and none of the tolerance code is reachable. This is the
     test that lets the change land in front of a live discovery run.
  2. **A timed-out candidate is dropped, not fatal.** No tile on disk means the framing
     cannot be scored; the argmax skips it and the trace records it.
  3. **Losing the ORIGINAL framing IS fatal.** The returned score is only a valid
     cross-location re-ranking key because the input framing (fw x1.0, recenter 0,0) is
     always in the search space and therefore bounds the result from below. If that rung
     is the one that timed out, MONOTONE-NON-DECREASING silently stops being true and the
     caller has no way to notice — so refuse instead.
  4. **The guard-field dump is bounded too**, since it is a second subprocess per tile.

No GPU, no engine: `_render` is monkeypatched and the scorer is a stub, so this runs in
the default pytest lane.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import reframe  # noqa: E402

LOC = reframe.Location(family="mandelbrot", c_re=None, c_im=None,
                       cx="-0.5", cy="0.0", fw="1e-3", family_params={})


class StubScorer:
    """Scores by tile name so the argmax is deterministic and the shared center tile is
    visibly deduped. K=3 shape: (E[ord], p_ge2, p_ge3)."""

    def __init__(self):
        self.calls = []

    def score_paths_k(self, paths):
        self.calls.append([Path(p).name for p in paths])
        out = []
        for p in paths:
            # "c2_dx+0.25_dy+0.00.jpg" -> a stable score; the +0.25/+0.25 recenter wins.
            n = Path(p).name
            s = 1.0 + 0.1 * n.count("+0.25")
            out.append((s, 0.9, s / 2))
        return out


def _fake_render(*, expire: set[str] = frozenset(), record: list | None = None):
    """Stand-in for `reframe._render`: writes the tile (so the exists() check passes) or
    raises TimeoutExpired for tile names in `expire`, mirroring a real hang — which
    leaves NO file behind."""

    def inner(loc, c, out, w, h, ss, *, timeout=None):
        if record is not None:
            record.append(timeout)
        if reframe._tile_name(c) in expire:
            raise subprocess.TimeoutExpired(cmd=["render-one"], timeout=timeout)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return True, ""

    return inner


# --------------------------------------------------------------------------- #
# 1. the default is byte-identical
# --------------------------------------------------------------------------- #
def test_default_passes_timeout_none_to_every_render(monkeypatch, tmp_path):
    seen: list = []
    monkeypatch.setattr(reframe, "_render", _fake_render(record=seen))
    res = reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path, workers=2)
    assert seen, "no renders were attempted"
    assert set(seen) == {None}, f"a bound leaked into the default path: {set(seen)}"
    assert res.trace["render"]["timeout_s"] is None
    assert res.trace["timed_out"] == []
    # 12 distinct renders, not 13 — the (best_fw, center) tile is shared between steps.
    assert len(seen) == 12, seen


def test_explicit_timeout_reaches_every_render(monkeypatch, tmp_path):
    seen: list = []
    monkeypatch.setattr(reframe, "_render", _fake_render(record=seen))
    res = reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path,
                                  workers=2, timeout=45)
    assert set(seen) == {45}
    assert res.trace["render"]["timeout_s"] == 45


def test_real_render_forwards_timeout_to_subprocess(monkeypatch, tmp_path):
    """The plumbing, not the policy: `_render` itself must hand the bound to
    subprocess.run, or every test above passes while nothing is actually bounded."""
    got = {}

    class R:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        got.update(kw)
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        return R()

    monkeypatch.setattr(reframe.subprocess, "run", fake_run)
    c = reframe._candidate(LOC, 1.0, 0.0, 0.0)
    ok, err = reframe._render(LOC, c, tmp_path / "t.jpg", 64, 36, 1, timeout=12.5)
    assert ok, err
    assert got["timeout"] == 12.5
    got.clear()
    reframe._render(LOC, c, tmp_path / "t2.jpg", 64, 36, 1)
    assert got["timeout"] is None


def test_guard_field_dump_is_bounded_too(monkeypatch, tmp_path):
    """The guard hook adds a SECOND subprocess per tile; leaving it unbounded would leave
    the hang in place for every guarded discovery run."""
    seen = []

    class R:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        seen.append(kw.get("timeout"))
        target = ("--dump-field" in cmd) and cmd[cmd.index("--dump-field") + 1] \
            or cmd[cmd.index("--out") + 1]
        Path(target).write_bytes(b"")
        return R()

    monkeypatch.setattr(reframe.subprocess, "run", fake_run)
    monkeypatch.setattr(reframe, "DUMP_GUARD_FIELD", True)
    c = reframe._candidate(LOC, 1.0, 0.0, 0.0)
    ok, err = reframe._render(LOC, c, tmp_path / "t.jpg", 64, 36, 1, timeout=30)
    assert ok, err
    assert seen == [30, 30], f"guard-field dump was not bounded: {seen}"


# --------------------------------------------------------------------------- #
# 2. a timed-out candidate is dropped
# --------------------------------------------------------------------------- #
def test_timed_out_candidate_is_dropped_not_fatal(monkeypatch, tmp_path):
    # Kill the x1.414 rung (index 3) — not the original framing.
    doomed = reframe._tile_name(reframe._candidate(LOC, reframe.FW_FACS[3], 0.0, 0.0))
    monkeypatch.setattr(reframe, "_render", _fake_render(expire={doomed}))
    res = reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path,
                                  workers=2, timeout=1)
    assert [t["fw_factor"] for t in res.trace["timed_out"]] == [reframe.FW_FACS[3]]
    # the hole is a None score, never a zero that would look like a real bad frame
    hole = [r for r in res.trace["fw_ladder"] if r["fw_factor"] == reframe.FW_FACS[3]][0]
    assert hole["score"] is None and hole["timed_out"] is True
    assert hole["p_good"] is None
    assert res.score > 0 and res.trace["original_score"] > 0


def test_timed_out_recenter_still_yields_the_cached_center(monkeypatch, tmp_path):
    """Every recenter but the shared center dies. The center tile is cached from step 1,
    so the search still returns — at recenter (0,0)."""
    doomed = {reframe._tile_name(reframe._candidate(LOC, f, dx, dy))
              for f in reframe.FW_FACS for dx in reframe.RECENTER
              for dy in reframe.RECENTER if not (dx == 0.0 and dy == 0.0)}
    monkeypatch.setattr(reframe, "_render", _fake_render(expire=doomed))
    res = reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path,
                                  workers=2, timeout=1)
    assert res.trace["chosen"] == {"fw_factor": res.trace["best_fw_factor"],
                                   "dx": 0.0, "dy": 0.0}
    # no centered framing is ever in the timed-out set — those are what survived
    assert all((t["dx"], t["dy"]) != (0.0, 0.0) for t in res.trace["timed_out"])


# --------------------------------------------------------------------------- #
# 3. losing the original framing, or everything, is fatal
# --------------------------------------------------------------------------- #
def test_original_framing_timeout_refuses(monkeypatch, tmp_path):
    orig = reframe._tile_name(reframe._candidate(LOC, 1.0, 0.0, 0.0))
    monkeypatch.setattr(reframe, "_render", _fake_render(expire={orig}))
    with pytest.raises(SystemExit) as e:
        reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path,
                                 workers=2, timeout=1)
    assert "monotone" in str(e.value).lower()


def test_every_candidate_timeout_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(reframe, "_render", _fake_render(
        expire={reframe._tile_name(reframe._candidate(LOC, f, 0.0, 0.0))
                for f in reframe.FW_FACS}))
    with pytest.raises(SystemExit) as e:
        reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path,
                                 workers=2, timeout=1)
    # the original rung is among them, so the monotonicity refusal fires first
    assert "timed out" in str(e.value).lower()


# --------------------------------------------------------------------------- #
# 4. a real render FAILURE is still fatal (unchanged behaviour)
# --------------------------------------------------------------------------- #
def test_render_failure_is_still_fatal(monkeypatch, tmp_path):
    def failing(loc, c, out, w, h, ss, *, timeout=None):
        return False, "engine exploded"

    monkeypatch.setattr(reframe, "_render", failing)
    with pytest.raises(SystemExit) as e:
        reframe.reframe_location(LOC, scorer=StubScorer(), workdir=tmp_path, workers=2)
    assert "reframe render failed" in str(e.value)
