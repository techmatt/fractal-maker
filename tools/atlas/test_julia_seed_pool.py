"""The julia-arm seed population has a committed producer, a reproducible filter, and a
durable home whose write site ASSERTS durability rather than trusting the path string.

The 534-row `julia_seed_pool.json` was made once by hand from q4_decisive's `viable.json`
(drop the exemplar anchor, project to c_re/c_im) and lived only under `scratch/`, a class whose
contract guarantees deletion. `build_julia_seed_pool` puts the filter in code and the output
through `paths.durable()`. These tests pin: the filter, the durable-registration (durable()
raises if git would discard the path), and — when the scratch input is present — that the
committed durable file is exactly what the filter reproduces.

Run:  uv run pytest tools/atlas/test_julia_seed_pool.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import paths  # noqa: E402
import build_julia_seed_pool as B  # noqa: E402
import disk_audit as da  # noqa: E402


def _clear_ignore_cache():
    clear = getattr(paths._is_gitignored, "cache_clear", None)
    if clear:
        clear()


def test_filter_drops_anchor_and_projects_in_order():
    viable = [
        {"cid": "exemplar", "anchor": True, "c_re": 9.0, "c_im": 9.0, "klass": "viable"},
        {"cid": "b0", "anchor": False, "c_re": 0.1, "c_im": 0.2, "scr_occ": 0.3},
        {"cid": "b1", "anchor": False, "c_re": 0.3, "c_im": 0.4},
    ]
    pool = B.filter_seed_pool(viable)
    assert pool == [{"c_re": 0.1, "c_im": 0.2}, {"c_re": 0.3, "c_im": 0.4}]
    assert all(set(r) == {"c_re", "c_im"} for r in pool)   # projected, no extra fields
    assert not any(r["c_re"] == 9.0 for r in pool)          # exemplar anchor excluded


def test_seed_pool_path_is_durable_registered(monkeypatch):
    """If the durable home ever became gitignored, the WRITE must fail on the spot."""
    _clear_ignore_cache()
    monkeypatch.setattr(paths, "_is_gitignored", lambda _p: True)
    with pytest.raises(paths.DurabilityError):
        B.seed_pool_path()
    _clear_ignore_cache()


def test_real_seed_pool_path_not_gitignored():
    _clear_ignore_cache()
    p = paths.durable(B.SEED_POOL_REL)
    assert str(p).replace("\\", "/").endswith("data/atlas/julia_seed_pool.json")


def test_disk_audit_forces_never():
    assert da.classify("data/atlas/julia_seed_pool.json").category == da.NEVER


def test_committed_file_is_what_the_filter_reproduces():
    """Reproducibility, end to end: the committed durable file equals filter(viable.json).
    Skips only if the scratch input has been wiped (the filter/​durable-path tests still run)."""
    committed = B.SEED_POOL_JSON
    if not committed.exists():
        pytest.skip("durable seed pool not committed yet")
    if not B.VIABLE_DEFAULT.exists():
        pytest.skip("scratch viable.json wiped — cannot re-derive (regenerate via q4_decisive)")
    viable = json.loads(B.VIABLE_DEFAULT.read_text(encoding="utf-8"))
    assert json.loads(committed.read_text(encoding="utf-8")) == B.filter_seed_pool(viable)
