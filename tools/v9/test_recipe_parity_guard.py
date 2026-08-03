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

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in ("tools", "tools/v9", "tools/corpus", "tools/scoring", "tools/mining"):
    sys.path.insert(0, str(ROOT / p))

import build_plan as bp                  # noqa: E402  (tools/v9/build_plan.py)


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


def test_the_guard_is_armed_with_paths_that_actually_exist():
    """A guard pointed at a path that never existed is a guard against a typo. Every rel
    handed to `_require_v8` is read out of the source and checked on disk, so a rename of
    a v8 artifact fails here rather than at the top of a rebuild."""
    src = inspect.getsource(bp.assert_recipe_parity) + inspect.getsource(bp)
    rels = set(re.findall(r'_require_v8\(\s*"([^"]+)"\s*\)', src))
    # ...plus the ones passed by module constant
    for name in re.findall(r"_require_v8\(\s*([A-Z_][A-Z_0-9]*)\s*\)", src):
        rels.add(getattr(bp, name))
    rels.discard("data/v8/__absent_for_this_test__.jsonl")
    assert rels, "no _require_v8 call sites found — the parity check was restructured"
    missing = sorted(r for r in rels if not (ROOT / r).exists())
    assert not missing, (
        f"the parity guard is armed with {missing}, which are not on disk. Either they "
        f"were deleted (rebuild: uv run python tools/v8/build_plan.py) or renamed.")
