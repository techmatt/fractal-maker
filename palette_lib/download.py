"""Downloader for public, reachable palette collections (personal-use harvest).

Discovered reachable sources (probed 2026-06-21; guessed URLs were verified, not
hard-coded blindly):
  - fract4d/gnofract4d  maps/  -> 206 Fractint .map + 2 UltraFractal .ugr
    (~100 .ugr blocks). Reachable via raw.githubusercontent.com. GPL project;
    bundled Fractint maps. Treated as a THIRD-PARTY HARVEST: cached under
    palette_cache/harvest/ which is gitignored and never redistributed.
  - matplotlib / colorcet / cmasher: pip libraries, not downloaded — read live
    by importer.py. Clean / zero-restriction; the committable backbone.

UltraFractal's own formula DB (formulas.ultrafractal.com) did not resolve (SSL);
artist DeviantArt packs are per-artist copyright and deliberately not scraped —
hand-drop those into palette_cache/harvest/ if wanted.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = "fract4d/gnofract4d"
BRANCH = "master"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"

CACHE = Path(__file__).resolve().parent.parent / "palette_cache"
HARVEST = CACHE / "harvest" / "gnofract4d"

_H = {"User-Agent": "fractal-generator-palette-harvest"}


def _get(url, **kw):
    # Imported lazily so a warm (offline) cache needs neither the network NOR the
    # `requests` package — only a cold-cache / force re-sync reaches this line.
    import requests
    return requests.get(url, headers=_H, timeout=30, **kw)


def _cached_files():
    """Harvested .ugr/.map already on disk (the cache doubles as its own manifest)."""
    if not HARVEST.exists():
        return []
    return sorted(
        p for p in HARVEST.iterdir()
        if p.is_file() and p.suffix.lower() in (".ugr", ".map")
    )


def harvest_gnofract4d(force=False):
    """Download all maps/*.ugr and maps/*.map into the gitignored harvest cache.

    **Cache before network.** A warm cache is served straight from disk with **zero
    network** — a warm run works fully offline. The GitHub tree API (to enumerate the
    file list) and the raw per-file fetches are consulted ONLY on a **cold cache**
    (nothing harvested yet) or an explicit ``force=True`` re-sync; the tree call used
    to run unconditionally, so even a warm run needed connectivity. Discovering NEW
    upstream files therefore requires ``force=True`` (the cache is otherwise trusted
    as its own manifest). Returns the list of local file paths.
    """
    cached = _cached_files()
    if cached and not force:
        print(f"[harvest] gnofract4d: {len(cached)} files in cache (warm, no network)")
        return cached

    # Cold cache (or forced re-sync): now — and only now — reach for the network.
    HARVEST.mkdir(parents=True, exist_ok=True)
    tree = _get(TREE_URL).json().get("tree", [])
    wanted = [
        t["path"]
        for t in tree
        if t["type"] == "blob"
        and t["path"].startswith("maps/")
        and t["path"].lower().endswith((".ugr", ".map"))
    ]
    local = []
    fetched = 0
    for path in wanted:
        name = path.split("/", 1)[1]  # strip "maps/"
        dest = HARVEST / name
        if dest.exists() and not force:
            local.append(dest)
            continue
        r = _get(RAW + path)
        if r.status_code != 200:
            continue
        dest.write_text(r.text, encoding="utf-8", errors="replace")
        local.append(dest)
        fetched += 1
    print(f"[harvest] gnofract4d: {len(local)} files in cache ({fetched} newly fetched)")
    return local


if __name__ == "__main__":
    harvest_gnofract4d(force=bool(os.environ.get("FORCE")))
