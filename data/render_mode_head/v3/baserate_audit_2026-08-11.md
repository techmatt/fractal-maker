# Sheet F — base-rate audit and the label/score crossover

Generated 2026-08-11T16:03:36 · `uv run python tools/mining/baserate_audit_reads.py`


Population: **200 rows** over 131 locations — sheet F — 2026-08-11_render_mode_baserate_audit_v1, the base-rate audit sitting; basis `[human n=200, prefill-anchored — ceiling]`.


> EVERY rate in this block is a CEILING. The page was v3-prefilled and score-sorted, so label and score are coupled by construction; a correction rate here measures agreement with v3 and never quality (classifier_retrain_protocol.md §2b). Sheet E is the unanchored bound on the same draw rule.


## Tier mix, and the two bounds on it

| slice | elicitation | n | tiers | ≥2 | ≥3 |
|---|---|--:|---|--:|--:|
| F human | CORRECTION — v3's tier PREFILLED, page sorted by v3's readout | 200 | {'1': 93, '2': 88, '3': 19} | 107 (53.5%) | 19 (9.5%) |
| F v3-prefill | the suggested_tier each row was SERVED with (suggest_tier_mining.CUTS on v3) | 200 | {'1': 94, '2': 86, '3': 20} | 106 (53.0%) | 20 (10.0%) |
| E blind | BLIND, shuffled | 150 | {'1': 74, '2': 70, '3': 6} | 76 (50.7%) | 6 (4.0%) |

## Correction rate against the v3 prefill

Exact tier agreement **176/200 = 88.0%**. Flips across ≥2: **5 up** (served <2, human ≥2), **4 down**; across ≥3: **7 up**, **8 down**.


| served ↓ / human → | 1 | 2 | 3 |
|---|--:|--:|--:|
| **1** | 89 | 5 | 0 |
| **2** | 4 | 75 | 7 |
| **3** | 0 | 8 | 12 |

## The crossover

Isotonic fit of `1[label >= 2]` against `p_ge3`, 200 rows, base rate 53.5%. Fitted probability reaches 0.5 between **0.09475** and **0.09514** → constant **0.0949**, realized volume **119/200**.


The crossing lands on a TIE BLOCK fitted at exactly 0.5, so the reading is ambiguous by one block: `> 0.5` instead of `>= 0.5` gives **0.1039** at 111/200. Adopted convention: first fitted value AT OR ABOVE 0.5.


| fitted P(label≥2) | rows | positives | score range |
|--:|--:|--:|---|
| 0.0000 | 81 | 0 | 0.00000 – 0.09475 |
| 0.5000 | 8 | 4 | 0.09514 – 0.10301 |
| 0.6000 | 5 | 3 | 0.10474 – 0.11018 |
| 0.8000 | 10 | 8 | 0.11554 – 0.12884 |
| 0.9394 | 66 | 62 | 0.13780 – 0.40144 |
| 1.0000 | 30 | 30 | 0.42555 – 0.88075 |

## The cuts

| cut | owner | old | **new** | fires (F) | pass rate | precision≥3 old → new | precision≥2 old → new |
|---|---|--:|--:|--:|--:|---|---|
| `mining_release` | `tools/mining/mining_pins.MINING_GATE_THRESHOLD (== tools/emission/floors.MINING_RELEASE)` | 0.6691 | **0.0949** | 119/200 | 59.5% | 55.6% → 16.0% | 100.0% → 89.9% |
| `mining_pool` | `tools/emission/floors.MINING_POOL` | 0.3402 | **0** | 200/200 | 100.0% | 41.7% → 9.5% | 97.2% → 53.5% |

REALIZED, not matched — a crossover holds the label MEANING fixed and lets the volume move. The 'matched' spelling is the field the lock builder reads; §5a's invariant does not apply.


## The same cuts on the flip's reference pool (827 rows)

the (28) deduplicated mining eval side (mining_corpus.load_corpus), re-scored under the live pin; base rate ≥3 25.9%, ≥2 68.4%.


| cut | old | new | fires old → new | precision≥3 old → new | recall≥3 old → new |
|---|--:|--:|---|---|---|
| `mining_release` | 0.6691 | 0.0949 | 129 → **587** | 76.0% → 36.3% | 45.8% → 99.5% |
| `mining_pool` | 0.3402 | 0 | 322 → **827** | 52.5% → 25.9% | 79.0% → 100.0% |

**JUNK_FLOOR inversion.** JUNK_FLOOR is PERMANENT shared-scale and was not moved (it is read on two heads' scales). With the gate below it, the colorize-pool draw now cuts rows the gate passes; the count above is how many of this pool survive it. 455/827 clear 0.2.


## Ladder on sheet F — ≥3, base rate 9.5%

| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |
|--:|--:|--:|--:|--:|--:|--:|---|
| 0.0000 | 200 | 100.0% | 19 | 9.5% | 6.2%–14.4% | 100.0% | mining_pool |
| 0.0500 | 140 | 70.0% | 19 | 13.6% | 8.9%–20.2% | 100.0% |  |
| 0.0949 | 119 | 59.5% | 19 | 16.0% | 10.5%–23.6% | 100.0% | mining_release |
| 0.1000 | 116 | 58.0% | 19 | 16.4% | 10.7%–24.2% | 100.0% |  |
| 0.1500 | 89 | 44.5% | 19 | 21.3% | 14.1%–31.0% | 100.0% |  |
| 0.2000 | 76 | 38.0% | 17 | 22.4% | 14.5%–32.9% | 89.5% |  |
| 0.2500 | 56 | 28.0% | 16 | 28.6% | 18.4%–41.5% | 84.2% |  |
| 0.3000 | 45 | 22.5% | 16 | 35.6% | 23.2%–50.2% | 84.2% |  |
| 0.3500 | 35 | 17.5% | 15 | 42.9% | 28.0%–59.1% | 78.9% |  |
| 0.4000 | 31 | 15.5% | 15 | 48.4% | 32.0%–65.2% | 78.9% |  |
| 0.4500 | 25 | 12.5% | 15 | 60.0% | 40.7%–76.6% | 78.9% |  |
| 0.5000 | 19 | 9.5% | 12 | 63.2% | 41.0%–80.9% | 63.2% |  |
| 0.5500 | 15 | 7.5% | 8 | 53.3% | 30.1%–75.2% | 42.1% |  |
| 0.6000 | 11 | 5.5% | 6 | 54.5% | 28.0%–78.7% | 31.6% |  |
| 0.6500 | 9 | 4.5% | 5 | 55.6% | 26.7%–81.1% | 26.3% |  |
| 0.7000 | 9 | 4.5% | 5 | 55.6% | 26.7%–81.1% | 26.3% |  |
| 0.7500 | 5 | 2.5% | 3 | 60.0% | 23.1%–88.2% | 15.8% |  |
| 0.8000 | 3 | 1.5% | 2 | 66.7% | 20.8%–93.9% | 10.5% |  |
| 0.8500 | 1 | 0.5% | 0 | 0.0% | 0.0%–79.3% | 0.0% |  |
| 0.9000 | 0 | 0.0% | 0 | — | — | 0.0% |  |
| 0.9500 | 0 | 0.0% | 0 | — | — | 0.0% |  |

## Ladder on sheet F — ≥2 (the crossover's boundary), base rate 53.5%

| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |
|--:|--:|--:|--:|--:|--:|--:|---|
| 0.0000 | 200 | 100.0% | 107 | 53.5% | 46.6%–60.3% | 100.0% | mining_pool |
| 0.0500 | 140 | 70.0% | 107 | 76.4% | 68.8%–82.7% | 100.0% |  |
| 0.0949 | 119 | 59.5% | 107 | 89.9% | 83.2%–94.1% | 100.0% | mining_release |
| 0.1000 | 116 | 58.0% | 105 | 90.5% | 83.8%–94.6% | 98.1% |  |
| 0.1500 | 89 | 44.5% | 85 | 95.5% | 89.0%–98.2% | 79.4% |  |
| 0.2000 | 76 | 38.0% | 72 | 94.7% | 87.2%–97.9% | 67.3% |  |
| 0.2500 | 56 | 28.0% | 53 | 94.6% | 85.4%–98.2% | 49.5% |  |
| 0.3000 | 45 | 22.5% | 43 | 95.6% | 85.2%–98.8% | 40.2% |  |
| 0.3500 | 35 | 17.5% | 34 | 97.1% | 85.5%–99.5% | 31.8% |  |
| 0.4000 | 31 | 15.5% | 30 | 96.8% | 83.8%–99.4% | 28.0% |  |
| 0.4500 | 25 | 12.5% | 25 | 100.0% | 86.7%–100.0% | 23.4% |  |
| 0.5000 | 19 | 9.5% | 19 | 100.0% | 83.2%–100.0% | 17.8% |  |
| 0.5500 | 15 | 7.5% | 15 | 100.0% | 79.6%–100.0% | 14.0% |  |
| 0.6000 | 11 | 5.5% | 11 | 100.0% | 74.1%–100.0% | 10.3% |  |
| 0.6500 | 9 | 4.5% | 9 | 100.0% | 70.1%–100.0% | 8.4% |  |
| 0.7000 | 9 | 4.5% | 9 | 100.0% | 70.1%–100.0% | 8.4% |  |
| 0.7500 | 5 | 2.5% | 5 | 100.0% | 56.6%–100.0% | 4.7% |  |
| 0.8000 | 3 | 1.5% | 3 | 100.0% | 43.8%–100.0% | 2.8% |  |
| 0.8500 | 1 | 0.5% | 1 | 100.0% | 20.7%–100.0% | 0.9% |  |
| 0.9000 | 0 | 0.0% | 0 | — | — | 0.0% |  |
| 0.9500 | 0 | 0.0% | 0 | — | — | 0.0% |  |
