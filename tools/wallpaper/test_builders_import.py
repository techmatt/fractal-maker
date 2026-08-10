"""Module-load smoke for the wallpaper-head batch builders.

These front-ends (`build_headbatch_dramatic` especially) share a web of cross-module
helpers via sys.path.insert + bare imports — `build_fresh_discovery`, `sample_location`,
`pool_rule`, `label_crop`, `corpus_common`, … A helper renamed or dropped in one of those
surfaces only as an AttributeError at module-load time, and nothing in CI imported these
builders, so the rot was invisible until a manual run. (`build_headbatch_dramatic` broke
exactly this way: build_fresh_discovery widened to all-families and dropped DEG2_FAMILIES,
which the builder referenced at top level.)

This guard imports each builder by file path (they are scripts, not a package) and asserts
module-load succeeds — cheap, and it fails LOUD the next time a shared API drifts.

  uv run pytest tools/wallpaper/test_builders_import.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BUILDERS = [
    "build_headbatch_dramatic.py",
    "build_humanq3.py",
    "build_fresh_discovery.py",
    "build_bootstrap.py",
    "build_fresh_sheet.py",
    "build_colorize_sheet.py",
    "build_wallpaper_sitting.py",
]


@pytest.mark.parametrize("fname", BUILDERS)
def test_builder_module_loads(fname):
    path = HERE / fname
    assert path.exists(), path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # REGISTER BEFORE EXEC. A module carrying `from __future__ import annotations` AND a
    # `@dataclass` resolves its string annotations through `sys.modules[cls.__module__]`, so
    # exec'ing an unregistered module raises `AttributeError: 'NoneType' has no attribute
    # '__dict__'` from inside dataclasses — a failure of THIS LOADER, not of the builder.
    # Removed afterwards so a builder is never left shadowing a real import.
    import sys
    prev = sys.modules.get(path.stem)
    sys.modules[path.stem] = module
    try:
        spec.loader.exec_module(module)   # raises if any top-level import/const is stale
    finally:
        if prev is None:
            sys.modules.pop(path.stem, None)
        else:
            sys.modules[path.stem] = prev
