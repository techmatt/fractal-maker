#!/usr/bin/env python
"""The v9 recipe-parity check must FAIL when its v8 input is missing, not skip.

`assert_recipe_parity` is the load-bearing claim of the whole v9 build: v9's corpus is
v8's corpus at a different iteration cap, and nothing else moved. It makes that claim by
comparing three committed v8 artifacts — the colormap library, `plan.jsonl` and
`cache_manifest.jsonl` — against what this build just produced.

All three reads used to be `if p.exists():`. A rebuild on a tree without them therefore
printed a parity block with the corresponding lines simply absent, and an absent line is
not something anyone reads a table for. That is the failure mode
`verification_practice.md` §2 names: a gate that degrades to silence cannot protect
against the removal of its own input — and the removal of its own input is exactly what
was about to happen, since `data/v8/{plan,cache_manifest}.jsonl` are 146 MB of derived
data that a deletion pass would want back.

So the guard is pinned from three directions, because one alone rots:
  * the raise happens, and its text names the missing file AND the rebuild command;
  * no `.exists()` guard is reintroduced into `assert_recipe_parity`'s source;
  * every path the guard is armed with actually exists today — otherwise it is a guard
    against a typo, and the real v9 rebuild would die on a name nobody can rebuild.

  uv run pytest tools/v9/test_recipe_parity_guard.py -q
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("tools", "tools/v9", "tools/corpus", "tools/scoring", "tools/mining"):
    sys.path.insert(0, str(ROOT / p))


def _load(name: str, path: Path):
    """Load a sibling under a UNIQUE module name. `import build_plan` would be ambiguous —
    seven versions ship one, and the first import in the interpreter wins for the rest of
    the session (`tests/test_import_hygiene.py`). Same pattern as the v5/v6 parity tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load("v9_build_plan", HERE / "build_plan.py")
v8bp = _load("v8_build_plan_for_parity_guard", ROOT / "tools" / "v8" / "build_plan.py")


def test_a_missing_v8_artifact_raises_and_names_the_rebuild_command():
    """The red case. `_require_v8` is what the three reads go through; a missing file
    must stop the build with an actionable message, not return None and let the caller
    omit a line."""
    rel = "data/v8/__absent_for_this_test__.jsonl"
    with pytest.raises(SystemExit) as ei:
        bp._require_v8(rel)
    msg = str(ei.value)
    assert rel in msg, msg
    assert "tools/v8/build_plan.py" in msg, (
        f"the message must name what rebuilds the file, not just that it is gone:\n{msg}")


def test_no_exists_guard_survives_in_assert_recipe_parity():
    """The regression that reintroduces the silence is a three-character edit
    (`if p.exists():`), and it would leave every assertion above still green — the raise
    still raises, the paths still exist — while the build once again skips its own check.
    So the source is asserted, not just the behaviour."""
    src = inspect.getsource(bp.assert_recipe_parity)
    hits = [ln.strip() for ln in src.splitlines() if re.search(r"\.exists\(\)", ln)]
    assert not hits, (
        "assert_recipe_parity must not gate a v8 comparison on the file being present — "
        f"route it through _require_v8 instead. Found: {hits}")


def test_the_guard_is_armed_with_paths_the_v8_builder_actually_produces():
    """A guard pointed at a path nothing produces is a guard against a typo.

    NOT "the file is on disk": two of the three are `data/v8/{plan,cache_manifest}.jsonl`,
    deleted 2026-08-03 precisely because `tools/v8/build_plan.py` reproduces them
    byte-identically on demand — being absent is the intended steady state, and the error
    message tells you to run that builder. What must hold is that the builder really does
    write them. So every rel handed to `_require_v8` is checked against v8's own output
    constants, and a rename on either side fails here rather than at the top of a rebuild
    that then cannot be completed."""
    src = inspect.getsource(bp)
    rels = set(re.findall(r'_require_v8\(\s*"([^"]+)"\s*\)', src))
    # ...plus the ones passed by module constant
    for name in re.findall(r"_require_v8\(\s*([A-Z_][A-Z_0-9]*)\s*\)", src):
        rels.add(getattr(bp, name))
    rels.discard("data/v8/__absent_for_this_test__.jsonl")
    assert rels, "no _require_v8 call sites found — the parity check was restructured"

    produced = {v for k, v in vars(v8bp).items()
                if k.endswith(("_OUT", "_PATH")) and isinstance(v, str) and v.startswith("data/v8/")}
    assert produced, "tools/v8/build_plan.py declares no data/v8/ outputs — constants renamed"
    unproducible = sorted(r for r in rels if r not in produced and not (ROOT / r).exists())
    assert not unproducible, (
        f"the parity guard is armed with {unproducible}, which are neither on disk nor "
        f"written by tools/v8/build_plan.py ({sorted(produced)}). The rebuild command in "
        f"the guard's own error message would not restore them.")
