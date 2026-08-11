# Volume-matched restatement — render_mode_head: v1 -> v3

Generated 2026-08-11T10:29:13 · `uv run python tools/scoring/volume_match.py mining`

Reference pool: **827 rows** over 136 locations — the (28) deduplicated mining eval side (mining_corpus.load_corpus); tiers {'1': 261, '2': 352, '3': 214}; base rate ≥3 0.259.


| cut | owner | old | **new** | volume | rate | precision≥3 old → new |
|---|---|---:|---:|---:|---:|---|
| `mining_release` | `tools/mining/mining_pins.MINING_GATE_THRESHOLD (== tools/emission/floors.MINING_RELEASE)` | 0.5 | **0.6691** | 129/827 | 0.156 | 0.636 → 0.760 |
| `mining_pool` | `tools/emission/floors.MINING_POOL` | 0.25 | **0.3402** | 322/827 | 0.389 | 0.534 → 0.525 |

Realized volume equals matched volume for every cut.

Volume-matching keeps VOLUME invariant on purpose; the precision column is what the head changed, not what the cut bought.

