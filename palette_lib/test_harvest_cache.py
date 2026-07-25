"""Cache-before-network guard for the gnofract4d palette harvest.

`harvest_gnofract4d` used to GET the GitHub tree API unconditionally — before any
cache read — so even a warm cache needed connectivity. The contract now is
**cache before network**: a warm cache is served from disk with zero network, and
the tree/raw endpoints are touched only on a cold cache (or force=True).

These tests pin both directions without any live network: the network seam
`download._get` is replaced by a fake that RAISES (simulating offline) and counts
calls. The warm path also proves the harder guarantee — a warm cache imports and
runs even though the `requests` package is not installed in this env (the import is
lazy inside `_get`, which the warm path never reaches).

  uv run pytest palette_lib/test_harvest_cache.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from palette_lib import download


class _OfflineGuard:
    """Stand-in for download._get that fails loudly — any call means we hit network."""
    def __init__(self):
        self.calls = 0

    def __call__(self, url, **kw):
        self.calls += 1
        raise AssertionError(f"network was contacted while offline: GET {url}")


def _warm(cache: Path, n=3):
    cache.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (cache / f"pal_{i}.map").write_text("0 0 0\n255 255 255\n", encoding="utf-8")
    # a non-palette file must be ignored by the manifest-from-cache logic
    (cache / "README.txt").write_text("not a palette", encoding="utf-8")


def test_warm_cache_serves_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "gnofract4d"
    _warm(cache, n=3)
    guard = _OfflineGuard()
    monkeypatch.setattr(download, "HARVEST", cache)
    monkeypatch.setattr(download, "_get", guard)

    files = download.harvest_gnofract4d()

    assert guard.calls == 0, "warm cache must not touch the network"
    assert len(files) == 3, files
    assert all(f.suffix == ".map" for f in files), files       # README.txt excluded


def test_force_resync_does_hit_network_even_when_warm(tmp_path, monkeypatch):
    cache = tmp_path / "gnofract4d"
    _warm(cache, n=3)
    guard = _OfflineGuard()
    monkeypatch.setattr(download, "HARVEST", cache)
    monkeypatch.setattr(download, "_get", guard)

    # force=True must reach the tree API (the guard raises -> proves the call happened)
    with pytest.raises(AssertionError, match="network was contacted"):
        download.harvest_gnofract4d(force=True)
    assert guard.calls == 1


def test_cold_cache_reaches_for_network(tmp_path, monkeypatch):
    cache = tmp_path / "gnofract4d"    # absent -> cold
    guard = _OfflineGuard()
    monkeypatch.setattr(download, "HARVEST", cache)
    monkeypatch.setattr(download, "_get", guard)

    with pytest.raises(AssertionError, match="network was contacted"):
        download.harvest_gnofract4d()
    assert guard.calls == 1
