# Mining gate lock — `mining_v1` @ data/render_mode_head/v1/model_best.pt

Frozen operating point of the render-mode (strange) quality gate. Written by `tools/mining/lock_mining_gate.py` from the committed sitting record `data/render_mode_head/v2/report.json` (sha256 `ced58a675c68607f…`); nothing here is re-measured, and a reader that finds the pin moved off `render_mode_head/v1` must refuse the whole file.


**Population.** `2026-08-06_render_mode_fresh_sheet_v1` — eval side (provenance.split_side == 'eval'); NOT re-derived, **n = 422** (48 locations, 15 roster modes), labels {'1': 246, '2': 113, '3': 63} on K=3 (1 bad / 2 okay / 3 good): base rate **14.9%** at >=3, **41.7%** at >=2.


## The two cuts

| cut | value | site | acts | fires | pass rate | precision | 95% CI | recall |
|---|--:|---|:-:|--:|--:|--:|--:|--:|
| `mining_pool` | 0.25 | pool | YES | 70/422 | 16.6% | 75.7% | 64.5%–84.2% | 84.1% |
| `mining_release` | 0.50 | release | YES | 33/422 | 7.8% | 97.0% | 84.7%–99.5% | 50.8% |

Both are on the gate signal — p_ge3 (marginal P(label>=3)). Precision is of PASSERS and carries a Wilson interval: the top of the ladder is estimated from a handful of rows, and a bare 1.000 over 3 and a 0.90 over 90 are the same column otherwise.


## What this is an optimistic bound on

- **eval_is_held_out_for_v2_only** — location-disjoint and unseen by v2's trainer; v1 trained on renders at these same 112 gate-passer locations, so v1 is read on a population it has partly memorised.
- **labels_are_anchored_to_v1** — correction sheet — every row was served with v1's suggested tier prefilled, sorted good->bad, Enter confirming. label and v1's score are coupled by construction.
- **direction** — BOTH caveats inflate v1 and neither touches v2. A v2 win is understated; a v1 win is partly an artifact this sitting cannot subtract.

**OPTIMISTIC. Both caveats above inflate v1 and neither is subtractable from these numbers: the head trained at these locations, and the labels were prefilled with its own suggestions. Every precision here is an upper bound on what the same cut buys at a FRESH location, and the honest use of this record is as a ceiling, not an estimate.**


**Harness parity.** v1 re-scored here vs head_mining_v1.p_ge3 stamped into images.jsonl when the sheet was built. Same checkpoint, same deploy transform, months apart. Max abs diff over 422 rows: 0.00e+00 (tolerance 1e-06) — **PASS**. These are the gate's own numbers, not a sibling scorer's.


## Frozen ladder — >=3 (the gate boundary), base rate 14.9%

| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |
|--:|--:|--:|--:|--:|--:|--:|---|
| 0.00 | 422 | 100.0% | 63 | 14.9% | 11.8%–18.6% | 100.0% |  |
| 0.05 | 187 | 44.3% | 63 | 33.7% | 27.3%–40.7% | 100.0% |  |
| 0.10 | 140 | 33.2% | 63 | 45.0% | 37.0%–53.3% | 100.0% |  |
| 0.15 | 107 | 25.4% | 62 | 57.9% | 48.5%–66.9% | 98.4% |  |
| 0.20 | 87 | 20.6% | 55 | 63.2% | 52.7%–72.6% | 87.3% |  |
| 0.25 | 70 | 16.6% | 53 | 75.7% | 64.5%–84.2% | 84.1% | mining_pool |
| 0.30 | 60 | 14.2% | 49 | 81.7% | 70.1%–89.4% | 77.8% |  |
| 0.35 | 52 | 12.3% | 45 | 86.5% | 74.7%–93.3% | 71.4% |  |
| 0.40 | 42 | 10.0% | 39 | 92.9% | 81.0%–97.5% | 61.9% |  |
| 0.45 | 38 | 9.0% | 37 | 97.4% | 86.5%–99.5% | 58.7% |  |
| 0.50 | 33 | 7.8% | 32 | 97.0% | 84.7%–99.5% | 50.8% | mining_release |
| 0.55 | 31 | 7.3% | 30 | 96.8% | 83.8%–99.4% | 47.6% |  |
| 0.60 | 22 | 5.2% | 21 | 95.5% | 78.2%–99.2% | 33.3% |  |
| 0.65 | 18 | 4.3% | 17 | 94.4% | 74.2%–99.0% | 27.0% |  |
| 0.70 | 15 | 3.6% | 14 | 93.3% | 70.2%–98.8% | 22.2% |  |
| 0.75 | 12 | 2.8% | 11 | 91.7% | 64.6%–98.5% | 17.5% |  |
| 0.80 | 9 | 2.1% | 8 | 88.9% | 56.5%–98.0% | 12.7% |  |
| 0.85 | 6 | 1.4% | 5 | 83.3% | 43.6%–97.0% | 7.9% |  |
| 0.90 | 3 | 0.7% | 3 | 100.0% | 43.8%–100.0% | 4.8% |  |
| 0.95 | 0 | 0.0% | 0 | — | — | 0.0% |  |

## Frozen ladder — >=2 (not-bad), base rate 41.7%

| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |
|--:|--:|--:|--:|--:|--:|--:|---|
| 0.00 | 422 | 100.0% | 176 | 41.7% | 37.1%–46.5% | 100.0% |  |
| 0.05 | 323 | 76.5% | 176 | 54.5% | 49.0%–59.8% | 100.0% |  |
| 0.10 | 256 | 60.7% | 172 | 67.2% | 61.2%–72.6% | 97.7% |  |
| 0.15 | 216 | 51.2% | 166 | 76.9% | 70.8%–82.0% | 94.3% |  |
| 0.20 | 187 | 44.3% | 162 | 86.6% | 81.0%–90.8% | 92.0% |  |
| 0.25 | 173 | 41.0% | 161 | 93.1% | 88.3%–96.0% | 91.5% | mining_pool |
| 0.30 | 156 | 37.0% | 152 | 97.4% | 93.6%–99.0% | 86.4% |  |
| 0.35 | 143 | 33.9% | 142 | 99.3% | 96.1%–99.9% | 80.7% |  |
| 0.40 | 122 | 28.9% | 122 | 100.0% | 96.9%–100.0% | 69.3% |  |
| 0.45 | 101 | 23.9% | 101 | 100.0% | 96.3%–100.0% | 57.4% |  |
| 0.50 | 91 | 21.6% | 91 | 100.0% | 95.9%–100.0% | 51.7% | mining_release |
| 0.55 | 77 | 18.2% | 77 | 100.0% | 95.2%–100.0% | 43.8% |  |
| 0.60 | 70 | 16.6% | 70 | 100.0% | 94.8%–100.0% | 39.8% |  |
| 0.65 | 60 | 14.2% | 60 | 100.0% | 94.0%–100.0% | 34.1% |  |
| 0.70 | 43 | 10.2% | 43 | 100.0% | 91.8%–100.0% | 24.4% |  |
| 0.75 | 32 | 7.6% | 32 | 100.0% | 89.3%–100.0% | 18.2% |  |
| 0.80 | 24 | 5.7% | 24 | 100.0% | 86.2%–100.0% | 13.6% |  |
| 0.85 | 21 | 5.0% | 21 | 100.0% | 84.5%–100.0% | 11.9% |  |
| 0.90 | 12 | 2.8% | 12 | 100.0% | 75.7%–100.0% | 6.8% |  |
| 0.95 | 6 | 1.4% | 6 | 100.0% | 61.0%–100.0% | 3.4% |  |

## Provenance

- **Head** render_mode_head/v1 — LIVE mining gate (mining_pins.ACTIVE_MINING_CKPT). Threshold 0.5 on marginal p_ge3 = cumprod(sigma(logits)) — NEVER the CORN conditional. Rollback: None.
- **Winner rule** — the calibration ran on **v1**. v2 (a finetune of v1 on this batch's train side) lost this rule, so the calibration — and this lock — are on the incumbent.
- **Supersedes** — the July lock at this path (frozen PR curve + deployed-scorer parity over data/render_mode_corpus/dataset_v1/) did not survive the corpus loss; its inputs are gone and it cannot be re-derived. Its operating point (precision 0.548 / recall 0.195 / pass-rate 0.050 at 0.50, base 0.139) survives only as prose quoted in fresh_sheet_reads.JULY_LOCK, measured on a DIFFERENT and genuinely held-out population.
- **Adoption** — prompts/mining_adoption_prompt.md — the release floor went from report-only to enforcing on this record's numbers.
