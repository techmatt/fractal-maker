# Mining gate lock — `mining_v3` @ data/render_mode_head/v3/model_best.pt

Frozen operating point of the render-mode (strange) quality gate. Written by `tools/mining/lock_mining_gate.py` from the committed volume-match record `data/render_mode_head/v3/volume_match_mining.json` (sha256 `436717ae995094d3…`); nothing here is re-measured, and a reader that finds the pin moved off `render_mode_head/v3` must refuse the whole file.


**Population.** the (28) deduplicated mining eval side (mining_corpus.load_corpus) — tools.scoring.volume_match.mining_pool - {'sheetB': 227, 'v1_sitting': 393, 'sheetC': 207}, **n = 827** (136 locations), labels {'1': 261, '2': 352, '3': 214} on K=3 (1 bad / 2 okay / 3 good): base rate **25.9%** at >=3, **68.4%** at >=2.


## The two cuts

| cut | value | was | site | acts | fires | pass rate | precision | 95% CI | recall |
|---|--:|--:|---|:-:|--:|--:|--:|--:|--:|
| `mining_pool` | 0.3402 | 0.25 (render_mode_head/v1) | pool | no | 322/827 | 38.9% | 52.5% | 47.0%–57.9% | 79.0% |
| `mining_release` | 0.6691 | 0.5 (render_mode_head/v1) | release | no | 129/827 | 15.6% | 76.0% | 67.9%–82.5% | 45.8% |

Both are on the gate signal — p_ge3 (marginal P(label>=3)). Every value is a VOLUME-MATCHED restatement of the `was` column: the same number of reference-pool rows passes, and only the precision beside it moved. Precision is of PASSERS and carries a Wilson interval: the top of the ladder is estimated from a handful of rows, and a bare 1.000 over 3 and a 0.90 over 90 are the same column otherwise.


## What this is an optimistic bound on

- **incumbent_trained_at_these_locations** — mining v1 trained at the 112 gate-passer locations the v1 sitting and sheet B draw from, and its dataset is gone so the exact rows cannot be excluded: 630 of the 827 rows sit at a location v1 has seen.
- **labels_are_anchored_to_v1** — every sheet in this corpus is a CORRECTION sheet - rows were served with v1's own suggested tier prefilled and the page sorted by its score, so label and v1's score are coupled by construction (0.929 of the v1 sitting's labels came back equal to what was served).
- **staged_is_eval_selected** — the pinned checkpoint is the best of five seeds by eval AP>=3 on this very slice, so a number read here is optimistic for it. The five-seed band is in the (28) report.
- **direction** — the first two lean toward the INCUMBENT and the third toward the pinned head. They do not cancel and none is subtractable.

**OPTIMISTIC. Two of the three leans above inflate the INCUMBENT and one inflates the pinned head; none is subtractable from these numbers. Every precision here is a bound on what the same cut buys at a FRESH (location, mode) pair, and the honest use of this record is as a ceiling, not an estimate. The unanchored read is tools/mining/sheet_e_reverdict.py.**


**Harness parity.** BY CONSTRUCTION, not by a separate check: the volume-match pass scores through mining_gate.MiningScorer - the gate's own scorer - so there is no sibling harness for these numbers to disagree with. (scorer: `mining_scorer`)


## Frozen ladder — >=3 (the gate boundary), base rate 25.9%

| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |
|--:|--:|--:|--:|--:|--:|--:|---|
| 0.0000 | 827 | 100.0% | 214 | 25.9% | 23.0%–29.0% | 100.0% |  |
| 0.0500 | 644 | 77.9% | 213 | 33.1% | 29.6%–36.8% | 99.5% |  |
| 0.1000 | 581 | 70.3% | 213 | 36.7% | 32.8%–40.7% | 99.5% |  |
| 0.1500 | 518 | 62.6% | 210 | 40.5% | 36.4%–44.8% | 98.1% |  |
| 0.2000 | 455 | 55.0% | 205 | 45.1% | 40.5%–49.6% | 95.8% |  |
| 0.2500 | 404 | 48.9% | 194 | 48.0% | 43.2%–52.9% | 90.7% |  |
| 0.3000 | 356 | 43.0% | 182 | 51.1% | 45.9%–56.3% | 85.0% |  |
| 0.3402 | 322 | 38.9% | 169 | 52.5% | 47.0%–57.9% | 79.0% | mining_pool |
| 0.3500 | 318 | 38.5% | 169 | 53.1% | 47.7%–58.6% | 79.0% |  |
| 0.4000 | 283 | 34.2% | 165 | 58.3% | 52.5%–63.9% | 77.1% |  |
| 0.4500 | 256 | 31.0% | 151 | 59.0% | 52.9%–64.8% | 70.6% |  |
| 0.5000 | 219 | 26.5% | 136 | 62.1% | 55.5%–68.3% | 63.6% |  |
| 0.5500 | 196 | 23.7% | 124 | 63.3% | 56.3%–69.7% | 57.9% |  |
| 0.6000 | 166 | 20.1% | 112 | 67.5% | 60.0%–74.1% | 52.3% |  |
| 0.6500 | 138 | 16.7% | 102 | 73.9% | 66.0%–80.5% | 47.7% |  |
| 0.6691 | 129 | 15.6% | 98 | 76.0% | 67.9%–82.5% | 45.8% | mining_release |
| 0.7000 | 118 | 14.3% | 91 | 77.1% | 68.8%–83.8% | 42.5% |  |
| 0.7500 | 92 | 11.1% | 70 | 76.1% | 66.4%–83.6% | 32.7% |  |
| 0.8000 | 70 | 8.5% | 52 | 74.3% | 63.0%–83.1% | 24.3% |  |
| 0.8500 | 47 | 5.7% | 38 | 80.9% | 67.5%–89.6% | 17.8% |  |
| 0.9000 | 26 | 3.1% | 22 | 84.6% | 66.5%–93.9% | 10.3% |  |
| 0.9500 | 14 | 1.7% | 13 | 92.9% | 68.5%–98.7% | 6.1% |  |

## Frozen ladder — >=2 (not-bad), base rate 68.4%

| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |
|--:|--:|--:|--:|--:|--:|--:|---|
| 0.0000 | 827 | 100.0% | 566 | 68.4% | 65.2%–71.5% | 100.0% |  |
| 0.0500 | 759 | 91.8% | 565 | 74.4% | 71.2%–77.4% | 99.8% |  |
| 0.1000 | 742 | 89.7% | 564 | 76.0% | 72.8%–78.9% | 99.6% |  |
| 0.1500 | 719 | 86.9% | 561 | 78.0% | 74.9%–80.9% | 99.1% |  |
| 0.2000 | 704 | 85.1% | 558 | 79.3% | 76.1%–82.1% | 98.6% |  |
| 0.2500 | 685 | 82.8% | 555 | 81.0% | 77.9%–83.8% | 98.1% |  |
| 0.3000 | 658 | 79.6% | 552 | 83.9% | 80.9%–86.5% | 97.5% |  |
| 0.3500 | 638 | 77.1% | 548 | 85.9% | 83.0%–88.4% | 96.8% |  |
| 0.4000 | 625 | 75.6% | 543 | 86.9% | 84.0%–89.3% | 95.9% |  |
| 0.4500 | 605 | 73.2% | 534 | 88.3% | 85.5%–90.6% | 94.3% |  |
| 0.5000 | 584 | 70.6% | 522 | 89.4% | 86.6%–91.6% | 92.2% |  |
| 0.5500 | 550 | 66.5% | 502 | 91.3% | 88.6%–93.4% | 88.7% |  |
| 0.6000 | 517 | 62.5% | 480 | 92.8% | 90.3%–94.8% | 84.8% |  |
| 0.6500 | 490 | 59.3% | 460 | 93.9% | 91.4%–95.7% | 81.3% |  |
| 0.7000 | 450 | 54.4% | 430 | 95.6% | 93.2%–97.1% | 76.0% |  |
| 0.7500 | 401 | 48.5% | 386 | 96.3% | 93.9%–97.7% | 68.2% |  |
| 0.8000 | 353 | 42.7% | 345 | 97.7% | 95.6%–98.8% | 61.0% |  |
| 0.8500 | 296 | 35.8% | 292 | 98.6% | 96.6%–99.5% | 51.6% |  |
| 0.9000 | 226 | 27.3% | 224 | 99.1% | 96.8%–99.8% | 39.6% |  |
| 0.9500 | 127 | 15.4% | 126 | 99.2% | 95.7%–99.9% | 22.3% |  |

## Provenance

- **Head** render_mode_head/v3 — LIVE mining gate (mining_pins.ACTIVE_MINING_CKPT). Threshold 0.6691 on marginal p_ge3 = cumprod(sigma(logits)) - NEVER the CORN conditional. Rollback: data/render_mode_head/v1/model_best.pt.
- **Source** `data/render_mode_head/v3/volume_match_mining.json`, generated 2026-08-11T10:29:13 by `uv run python tools/scoring/volume_match.py mining`.
- **Adoption** — prompts/flip_29.md - mining v1 -> v3 (the `dedup_weighted` arm), 2026-08-11. Both cuts were restated volume-matched at this flip; the v1 lock stays at data/render_mode_head/v1/ as the record of what 0.50 and 0.25 bought on v1.
