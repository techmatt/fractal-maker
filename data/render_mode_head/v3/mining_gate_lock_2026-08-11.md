# Mining gate lock — `mining_v3` @ data/render_mode_head/v3/model_best.pt

Frozen operating point of the render-mode (strange) quality gate. Written by `tools/mining/lock_mining_gate.py` from the committed measurement record `data/render_mode_head/v3/baserate_audit_2026-08-11.json` (sha256 `c5d241f3296e093f…`); nothing here is re-measured, and a reader that finds the pin moved off `render_mode_head/v3` must refuse the whole file.


**Population.** sheet F — 2026-08-11_render_mode_baserate_audit_v1, the base-rate audit sitting — tools.mining.baserate_audit_reads.load_sheet - {'2026-08-11_render_mode_baserate_audit_v1': 200}, **n = 200** (131 locations), labels {'1': 93, '2': 88, '3': 19} on K=3 (1 bad / 2 okay / 3 good): base rate **9.5%** at >=3, **53.5%** at >=2.


## The two cuts

| cut | value | was | site | acts | fires | pass rate | precision | 95% CI | recall |
|---|--:|--:|---|:-:|--:|--:|--:|--:|--:|
| `mining_pool` | 0 | 0.3402 (render_mode_head/v3) | pool | no | 200/200 | 100.0% | 9.5% | 6.2%–14.4% | 100.0% |
| `mining_release` | 0.0949 | 0.6691 (render_mode_head/v3) | release | no | 119/200 | 59.5% | 16.0% | 10.5%–23.6% | 100.0% |

Both are on the gate signal — p_ge3 (marginal P(label>=3)). **CROSSOVER:** The `was` column is NOT volume-matched to the new value and the two do not pass the same rows: a crossover holds the label MEANING fixed and lets the volume move, which here it does by 4.6x on the flip's reference pool (129 -> 587 of 827). That is the audit's finding, not a side effect of it. Precision is of PASSERS and carries a Wilson interval: the top of the ladder is estimated from a handful of rows, and a bare 1.000 over 3 and a 0.90 over 90 are the same column otherwise.


## What this is an optimistic bound on

- **the_page_was_prefilled_by_the_head_being_cut** — sheet F is a CORRECTION page - every row was served with v3's own suggested tier prefilled and the page sorted by v3's readout, and 176 of 200 labels came back equal to what was served. Label and score are coupled by construction, so the crossover is where the human agreed with the head, not only where the head is right.
- **the_draw_however_was_score_unconditioned** — no mining head touched the SELECTION - sheet E's population imported, flat mode apportionment, a pool palette draw, near-dup ties broken by draw order. That is what makes the TIER MIX a base rate over the population the gate sees; it does nothing to un-anchor the labels, because the anchoring is in the page and not in the draw.
- **nineteen_positives_at_the_gate_boundary** — the >=3 columns are estimated from 19 rows of 200. The cut is read at >=2 (107 of 200) where the sheet has power; every >=3 precision beside it is a wide interval and the Wilson bounds in the ladder are the honest width.
- **direction** — the first lean inflates agreement and therefore the sharpness of the crossover; the third widens intervals rather than moving them. The second is not a lean at all, it is what makes the base rate readable. None is subtractable. Sheet E (tools/mining/sheet_e_reverdict.py) is the unanchored bound on the same draw rule.

**OPTIMISTIC, and more so than its predecessor. The crossover is read off a page the cut head prefilled and sorted, so it is where the human AGREED with v3 - an upper bound on the separation a fresh (location, mode) pair would show. Basis [human n=200, prefill-anchored - ceiling]. The unanchored bound on the same draw rule is tools/mining/sheet_e_reverdict.py.**


**Harness parity.** BY CONSTRUCTION, not by a separate check: the measurement pass scores through mining_gate.MiningScorer - the gate's own scorer - so there is no sibling harness for these numbers to disagree with. (scorer: `mining_scorer`)


## What this cut forced elsewhere

THE JUNK-FLOOR INVERSION, resolved 2026-08-11 (prompts/junk_floor_repoint.md). Landing the gate at 0.0949 put it BELOW floors.JUNK_FLOOR (0.20), the one enforcing stage-2 cut, which tools/mining/deploy_tail.py had read since 2026-08-09 to draw its mining-side colorize pool. The permissive cut had become the strictest one on this head: the pool draw removed 132 of the 587 gate-good rows on the 827-row reference pool (455 clear 0.20 against 587 clearing the gate), so the compute-saving floor was silently overruling the cut this record freezes. MATT'S DECISION: REPOINT THE READER, not the number. deploy_tail filters its allocation input through mining_gate.MiningScorer.gate - the gate's own comparison, so this lock's threshold IS what draws that pool and a future pin flip moves both together. Realized mining-side colorize pool on the reference pool: 455 -> 587 of 827 (55.0% -> 71.0%), precision>=2 of the drawn set 0.952 -> 0.893, recall>=3 0.958 -> 0.995 - measured 2026-08-11 on tools.scoring.volume_match.mining_pool scored through MiningScorer under this pin. That is the flip's 827-row reference pool, the population the 129 -> 587 volume claim above is read on, NOT sheet F (n=200), which is what the ladders below are read on. THE THREE ALTERNATIVES, refused. (1) LEAVE IT STANDING - a documented inversion is still an inversion, and it makes the gate advisory at the only mining site that spends compute on the answer. (2) LOWER JUNK_FLOOR to sit under the gate - it is PERMANENT shared-scale (floors.py; a coarse semantic 'the judging head is confident this is junk', not an operating point) and moving it would have shifted the stage-1 intake draw, on a different head's scale, by an amount nobody measured. (3) SPLIT it per head - buys exactly the per-head operating point the cut was deliberately chosen not to be, and doubles a constant to avoid changing a reader. JUNK_FLOOR is untouched at 0.20 and still filters the stage-1 emission intake; it keeps one live reader, and deploy_tail now only COUNTS with it (the counterfactual in its run report).


## Frozen ladder — >=3 (the gate boundary), base rate 9.5%

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

## Frozen ladder — >=2 (not-bad), base rate 53.5%

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

## Provenance

- **Head** render_mode_head/v3 — LIVE mining gate (mining_pins.ACTIVE_MINING_CKPT). Threshold 0.0949 on marginal p_ge3 = cumprod(sigma(logits)) - NEVER the CORN conditional. Rollback: data/render_mode_head/v1/model_best.pt; the lock this one supersedes, kept as the record of what the previous cuts bought: `data/render_mode_head/v3/mining_gate_lock.json`.
- **Source** `data/render_mode_head/v3/baserate_audit_2026-08-11.json`, generated 2026-08-11T16:03:36 by `uv run python tools/mining/baserate_audit_reads.py`.
- **Adoption** — prompts/audit_mining_process.md - the sheet F base-rate audit, 2026-08-11. Matt's decision, pre-stated before the labels were read: land the gate at the crossover. The pool floor followed to 0.0 because a pool floor is defined relative to its release floor and floors.check_below_gate refuses the inversion. The un-suffixed lock beside this one stays as the record of what 0.6691 and 0.3402 bought, i.e. the rollback record (mining_pins.MINING_LOCK_ROLLBACK).
