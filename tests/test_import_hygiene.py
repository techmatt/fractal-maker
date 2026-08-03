#!/usr/bin/env python
"""No module whose BASENAME is ambiguous may be imported by that bare name.

There is no package root (`tools/README.md` § "Two standing facts"), so imports resolve by
`sys.path` order and every module is reachable by its bare basename. That is survivable
while basenames are unique. It stops being survivable the moment two files share one:

    import build_plan          # -> v4? v5? v6? v7? v8? v9? v10?

The answer is "whichever directory was inserted at sys.path[0] most recently", which is a
property of the whole process, not of this line. Worse, `sys.modules` caches the FIRST
resolution: once any module in the interpreter has imported `build_plan`, every later
`import build_plan` gets that one regardless of what the importer inserted. Under pytest,
where one process imports the whole tree, that is a real ordering dependency between
unrelated files.

**The fix is not renaming — it is the dotted form**, which `tools/` supports today with no
package root, no `__init__.py` and no editable install, because a directory without
`__init__.py` is a PEP 420 namespace package:

    from tools.v9 import build_plan as v9p      # unambiguous, resolves from the repo root

The importer must have the repo root on `sys.path` (most already insert it; the rest gained
one line). Ten names were ambiguous on 2026-08-02 and 22 call sites resolved by luck of
insert order; this holds that at zero. It also fails on a NEW collision — which is how it
would have caught `tools/scoring/derive_t_good.py` colliding with `tools/v7/derive_t_good.py`
the moment the former was created, instead of a rename after the fact.

Note this does NOT ban ambiguous basenames (the `tools/vN/` families are copy-forward by
design). It bans *resolving one by luck*.

  uv run pytest tests/test_import_hygiene.py -q
"""
from __future__ import annotations

import ast
import collections
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked_python() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [Path(p) for p in out.stdout.split() if p.strip()]


def _by_basename() -> dict[str, list[str]]:
    mods = collections.defaultdict(list)
    for f in _tracked_python():
        mods[f.stem].append(str(f).replace("\\", "/"))
    return mods


def _bare_imports(path: Path) -> list[tuple[str, int]]:
    """Every top-level-name import in `path`: `import x` / `from x import y`, x undotted.

    Dotted forms (`from tools.v9 import build_plan`) and relative ones are exactly what this
    file is asking for, so they are not reported."""
    try:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found += [(a.name, n.lineno) for a in n.names if "." not in a.name]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and "." not in n.module:
            found.append((n.module, n.lineno))
    return found


def test_no_ambiguous_basename_is_imported_bare():
    mods = _by_basename()
    ambiguous = {m for m, v in mods.items() if len(v) > 1}
    assert ambiguous, "no duplicate basenames at all — this test would be vacuous"
    offenders = []
    for f in _tracked_python():
        for name, line in _bare_imports(f):
            if name in ambiguous:
                offenders.append(
                    f"{str(f).replace(chr(92), '/')}:{line}  `{name}` -> one of {mods[name]}")
    assert not offenders, (
        f"{len(offenders)} bare import(s) of an ambiguous module basename — which file each "
        f"resolves to depends on sys.path order and sys.modules caching:\n  "
        + "\n  ".join(offenders)
        + "\nUse the dotted namespace-package form instead, e.g. "
          "`from tools.v9 import build_plan`, and make sure the repo root is on sys.path.")


def test_the_dotted_form_this_test_demands_actually_works():
    """Non-vacuity, and a live check on the mechanism: PEP 420 namespace packages resolve
    `tools.<sub>.<mod>` from the repo root with no `__init__.py` anywhere on the path. If a
    stray `tools/__init__.py` ever turned `tools` into a regular package this would be the
    first thing to notice."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.v9 import build_plan as v9p
    from tools.sources import sheet as sh

    assert Path(v9p.__file__).parent.name == "v9", v9p.__file__
    assert Path(sh.__file__).parent.name == "sources", sh.__file__
    assert not (ROOT / "tools" / "__init__.py").exists(), (
        "tools/ became a regular package — that changes resolution for the whole tree and "
        "breaks `uv run python tools/<sub>/<mod>.py` sibling imports; see tools/README.md")


def test_every_ambiguous_name_is_a_known_copy_forward_family():
    """The duplicates that remain are deliberate (`tools/vN/` is copy-forward by design, and
    the scorer/classifier pairs are two different models' code). A NEW duplicate outside
    these families is worth a second look rather than an automatic pass, so it is listed."""
    mods = _by_basename()
    ambiguous = sorted(m for m, v in mods.items() if len(v) > 1)
    known = {"__init__", "app", "build_features", "build_manifest", "build_plan", "data",
             "render_cache", "report", "sheet", "train", "train_v3", "verify_cache_alignment"}
    new = set(ambiguous) - known
    assert not new, (
        f"new duplicate basename(s) {sorted(new)}: "
        + "; ".join(f"{m} -> {mods[m]}" for m in sorted(new))
        + ". Either give it a distinct name, or add it here and make sure every importer "
          "uses the dotted form.")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
