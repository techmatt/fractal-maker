#!/usr/bin/env python
"""The committed single-process engine-launch defaults: 7 rayon threads at BELOW_NORMAL.

These two knobs were being carried by hand in prompts, which is how a number drifts run to
run and how a long render ends up either starving the desktop or leaving half the box idle.
They now live in `corpus_common` and these pin them, including the part that is easy to get
wrong: the 7 is a PER-PROCESS number, so a caller that fans out engine processes must set its
own and nothing may silently inherit a figure that assumed it had the box to itself.

No renders, no binary — pure environment/flag construction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

import corpus_common as cc  # noqa: E402


def test_single_process_default_is_seven_threads():
    assert cc.DEFAULT_ENGINE_THREADS == 7
    env = cc.default_engine_env({})
    assert env["RAYON_NUM_THREADS"] == "7"


def test_default_pairs_with_below_normal_priority():
    # The pairing is the point: threads buy throughput, the priority class buys interactivity.
    # A default that set one without the other would be half a decision.
    if sys.platform == "win32":
        assert cc.default_creationflags() == cc.BELOW_NORMAL_PRIORITY_CLASS
    else:
        assert cc.default_creationflags() == 0
    assert cc.BELOW_NORMAL_PRIORITY_CLASS == 0x00004000


def test_thread_count_stays_under_the_core_count():
    # 7 of 12 logical cores — real headroom for the desktop and for the Python/GPU side of a
    # scoring run. A default at or above the core count would make BELOW_NORMAL the only thing
    # standing between a batch render and an unusable machine.
    assert 1 < cc.DEFAULT_ENGINE_THREADS < 12


def test_explicit_threads_wins_over_the_default_and_over_the_environment():
    # A fan-out caller MUST be able to size its own processes; there is deliberately no standing
    # number for the multi-process case.
    assert cc.default_engine_env({}, threads=3)["RAYON_NUM_THREADS"] == "3"
    assert cc.default_engine_env({"RAYON_NUM_THREADS": "9"}, threads=3)["RAYON_NUM_THREADS"] == "3"


def test_a_caller_supplied_environment_setting_is_respected():
    # An existing RAYON_NUM_THREADS is never overwritten by the default — a caller that already
    # decided keeps its decision.
    assert cc.default_engine_env({"RAYON_NUM_THREADS": "2"})["RAYON_NUM_THREADS"] == "2"


def test_default_env_inherits_os_environ_and_does_not_mutate_it():
    before = dict(os.environ)
    env = cc.default_engine_env()
    assert env["PATH"] == os.environ["PATH"]        # inherits, so the engine still resolves
    env["SOME_PROBE_VAR"] = "x"
    assert "SOME_PROBE_VAR" not in os.environ       # a copy, not os.environ itself
    assert dict(os.environ) == before


def test_render_corpus_crop_accepts_the_thread_override():
    # Signature-level: the batch-builder seam must expose `threads`/`env` or a fan-out caller
    # has no way to opt out of the per-process default.
    import inspect
    sig = inspect.signature(cc.render_corpus_crop)
    for p in ("threads", "env", "creationflags"):
        assert p in sig.parameters, p
        assert sig.parameters[p].default is None, f"{p} must default to None (= use the default)"
