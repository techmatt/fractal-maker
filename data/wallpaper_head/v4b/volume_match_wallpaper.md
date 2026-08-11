# Volume-matched restatement — wallpaper_head: v3 -> seed_1

Generated 2026-08-11T10:30:17 · `uv run python tools/scoring/volume_match.py wallpaper`

Reference pool: **1337 rows** over 527 locations — the (28) six-batch eval union (train_wallpaper_v4b.split_v4b); tiers {'1': 284, '2': 495, '3': 369, '4': 189}; base rate ≥3 0.417.


| cut | owner | old | **new** | volume | rate | precision≥3 old → new |
|---|---|---:|---:|---:|---:|---|
| `wallpaper_release` | `tools/wallpaper/wallpaper_pins.GATE_THRESHOLD (== tools/emission/floors.WALLPAPER_RELEASE)` | 0.9 | **0.6052** | 416/1337 | 0.311 | 0.798 → 0.748 |
| `wallpaper_pool` | `tools/emission/floors.WALLPAPER_POOL` | 0.75 | **0.4698** | 503/1337 | 0.376 | 0.748 → 0.706 |

Realized volume equals matched volume for every cut.

Volume-matching keeps VOLUME invariant on purpose; the precision column is what the head changed, not what the cut bought.

