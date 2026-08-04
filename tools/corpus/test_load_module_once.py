"""`location._load_module_once` never publishes a half-built module.

The defect it closes: the loader used to bind `sys.modules[name]` BEFORE `exec_module`,
copying what the import system does for circular imports. These modules have no cycle, so
that bought nothing — and it cost a live crash. Three render threads reached
`current_maxiter_policy()` at once; the losers read the partially-executed `active_ckpt` and
died on `AttributeError: module 'active_ckpt' has no attribute 'MAXITER_BASE'`.

Made DETERMINISTIC rather than raced: the fixture module sleeps inside its own body, so a
second caller is guaranteed to arrive mid-exec. The old spelling is reproduced here as
`_publish_first` and asserted to FAIL the same invariant — the guard is proved able to tell
the two apart, without keeping the broken code in production
(verification_practice.md §3, §6 "the fixture is too easy").
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import location as loc_mod  # noqa: E402

# A module body that takes long enough that a concurrent caller MUST land inside it, and
# that binds its two names far enough apart to be observably partial in between.
FIXTURE_SRC = """
import time
FIRST = 1
time.sleep(0.4)
SECOND = 2
"""

N_THREADS = 6


def _fixture(tmp_path, name):
    p = tmp_path / f"{name}.py"
    p.write_text(FIXTURE_SRC, encoding="utf-8")
    return p


def _publish_first(name, src):
    """The PRE-FIX spelling, verbatim: bind the name, then exec. Present only so the
    invariant below is shown to discriminate."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, src)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _hammer(loader, name, src):
    """Run `loader` on N threads released together, with one extra thread POLLING
    sys.modules. Returns (results, errors, partial_observations)."""
    results, errors, partial = [], [], []
    ready = threading.Barrier(N_THREADS + 1)
    stop = threading.Event()

    def work():
        ready.wait()
        try:
            results.append(loader(name, src))
        except BaseException as e:      # noqa: BLE001 — the point is to catch whatever
            errors.append(e)

    def poll():
        ready.wait()
        while not stop.is_set():
            m = sys.modules.get(name)
            # The invariant: whenever the name is BOUND, the module is COMPLETE.
            if m is not None and not hasattr(m, "SECOND"):
                partial.append(m)
            time.sleep(0.002)

    ts = [threading.Thread(target=work) for _ in range(N_THREADS)]
    pt = threading.Thread(target=poll, daemon=True)
    pt.start()
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    stop.set()
    pt.join(timeout=5)
    return results, errors, partial


@pytest.fixture
def clean_sys_modules():
    names = []
    yield names
    for n in names:
        sys.modules.pop(n, None)
        loc_mod._LOADING.discard(n)


def test_a_concurrent_caller_never_sees_a_partial_module(tmp_path, clean_sys_modules):
    name = "_lmo_fixture_green"
    clean_sys_modules.append(name)
    results, errors, partial = _hammer(loc_mod._load_module_once, name,
                                       _fixture(tmp_path, name))
    assert not errors, f"loader raised under contention: {errors[:3]}"
    assert len(results) == N_THREADS
    assert all(m is results[0] for m in results), "threads got different module objects"
    assert results[0].SECOND == 2
    assert not partial, "sys.modules held a half-built module"


def test_the_pre_fix_spelling_fails_that_same_invariant(tmp_path, clean_sys_modules):
    """RED arm. If this ever passes, the test above has stopped discriminating and is
    green for a reason other than the one it claims."""
    name = "_lmo_fixture_red"
    clean_sys_modules.append(name)
    _results, errors, partial = _hammer(_publish_first, name, _fixture(tmp_path, name))
    assert partial or errors, (
        "publishing before exec did NOT produce an observable partial module — the "
        "green test proves nothing")


def test_the_body_executes_exactly_once(tmp_path, clean_sys_modules):
    """A lock that serialized without deduplicating would still pass the invariant above
    while running the module body N times."""
    name = "_lmo_fixture_once"
    clean_sys_modules.append(name)
    p = tmp_path / f"{name}.py"
    marker = tmp_path / "runs.txt"
    p.write_text(f"import time\nopen(r'{marker}','a').write('x')\n"
                 f"time.sleep(0.3)\nFIRST=1\nSECOND=2\n", encoding="utf-8")
    results, errors, _ = _hammer(loc_mod._load_module_once, name, p)
    assert not errors and len(results) == N_THREADS
    assert marker.read_text() == "x", "module body ran more than once"


def test_reentry_raises_instead_of_recursing(tmp_path, clean_sys_modules):
    """There is no import cycle through this loader today. A future one must be loud."""
    name = "_lmo_fixture_cycle"
    clean_sys_modules.append(name)
    p = tmp_path / f"{name}.py"
    p.write_text(
        "import sys, os\n"
        f"sys.path.insert(0, r'{_TOOLS}')\n"
        "import location as _loc\n"
        f"_loc._load_module_once({name!r}, r'{p}')\n", encoding="utf-8")
    with pytest.raises(ImportError, match="circular import"):
        loc_mod._load_module_once(name, p)
    assert name not in loc_mod._LOADING, "the in-progress marker leaked after the raise"


def test_the_live_policy_load_is_still_correct_and_cached():
    """The real caller: same object every time, and it is the module the ~41 `import
    active_ckpt` sites resolve to."""
    a = loc_mod._active_ckpt()
    assert a is loc_mod._active_ckpt()
    assert a is sys.modules["active_ckpt"]
    assert Path(a.__file__).name == "active_ckpt.py"
    assert loc_mod.current_maxiter_policy() == (a.MAXITER_BASE, a.MAXITER_K,
                                                a.MAXITER_MIN, a.MAXITER_MAX)
