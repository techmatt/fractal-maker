# mining head v2 — finetune from v1, eval + calibration

Batch `2026-08-06_render_mode_fresh_sheet_v1` · **eval side, n = 422** (48 locations, all 15 roster modes) · labels {'1': 246, '2': 113, '3': 63} (>=3 base rate **14.9%**, >=2 **41.7%**).


`v1` = `data/render_mode_head/v1/model_best.pt` (the LIVE pin, `mining_pins.ACTIVE_MINING_CKPT`) · `v2` = `data/render_mode_head/v2/model_best.pt` (finetuned from v1 on this batch's train side, 538 rows).


**Report only. No pin flip, no floor move, no gate change.** The two live cuts (pool **0.25**, acting; release **0.50**, report-only) are marked for reference and are not touched.


## 0 · which way every gap leans

- **eval_is_held_out_for_v2_only** — location-disjoint and unseen by v2's trainer; v1 trained on renders at these same 112 gate-passer locations, so v1 is read on a population it has partly memorised.
- **labels_are_anchored_to_v1** — correction sheet — every row was served with v1's suggested tier prefilled, sorted good->bad, Enter confirming. label and v1's score are coupled by construction.
- **direction** — BOTH caveats inflate v1 and neither touches v2. A v2 win is understated; a v1 win is partly an artifact this sitting cannot subtract.
- **staged_is_eval_selected** — v2's staged checkpoint is the best of 5 seeds BY eval AP>=3 on this very slice, so the staged number is optimistic. The 5-seed band is reported beside it and is the honest read.

**Harness parity.** v1 re-scored here vs head_mining_v1.p_ge3 stamped into images.jsonl when the sheet was built. Same checkpoint, same deploy transform, months apart. Max abs diff over 422 rows: **0.00e+00** (mean 0.00e+00); tolerance 1e-06 — **PASS**. Both heads are scored through this same path.


## 1 · overall, eval side (n = 422)

AUC/AP at each tier boundary, each on the marginal probability that boundary's gate uses. `Δ` is v2 − v1 with a 95% **paired** bootstrap CI (4000 draws, seed 20260806) — paired because both heads score identical rows.

| boundary | n pos | base | v1 | v2 | Δ (v2 − v1), 95% CI |
|---|--:|--:|--:|--:|--:|
| AUC >=3 | 63 | 14.9% | 0.978 | 0.971 | -0.008 [-0.023, +0.007] |
| AP >=3 | 63 | 14.9% | 0.877 | 0.830 | -0.044 [-0.108, +0.013] |
| AUC >=2 | 176 | 41.7% | 0.977 | 0.937 | -0.039 [-0.058, -0.022] **worse** |
| AP >=2 | 176 | 41.7% | 0.975 | 0.921 | -0.053 [-0.084, -0.031] **worse** |

Rank-score (Σσ) at >=3: v1 AUC 0.982, v2 0.970. Smallest AUC distinguishable from 0.50 at this n: 0.580 (>=3), 0.560 (>=2).


**v2's five seeds** (the staged checkpoint is the best of these BY eval AP>=3 on this slice, so the staged row above is optimistic; this band is not).

| seed | AUC >=3 | AP >=3 | AUC >=2 | AP >=2 |
|---|--:|--:|--:|--:|
| 0 | 0.957 | 0.798 | 0.936 | 0.921 |
| 1 | 0.965 | 0.790 | 0.936 | 0.923 |
| 2 | 0.971 | 0.830 | 0.937 | 0.921 |
| 3 | 0.964 | 0.822 | 0.943 | 0.929 |
| 4 | 0.961 | 0.815 | 0.953 | 0.937 |
| **mean ± SD** | 0.964 ± 0.004 | 0.811 ± 0.015 | 0.941 ± 0.006 | 0.926 ± 0.006 |


## 2 · per-mode, eval side

`v1 saw?` is whether v1's TRAINER included the mode — a low AUC on a mode v1 never trained on is a gap, not a failure. `—` means the boundary is not measurable in this cell (no positive or no negative at that boundary), which is a stronger statement than "at chance", not a missing one.

| mode | kind | v1 saw? | n | 1 | 2 | 3 | v1 AUC>=3 | v2 AUC>=3 | Δ | v1 AUC>=2 | v2 AUC>=2 | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tia | pure | yes | 27 | 9 | 8 | 10 | 0.965 | 0.976 | +0.012 | 0.994 | 0.969 | -0.025 |
| stripe | pure | yes | 28 | 14 | 10 | 4 | 1.000 | 0.958 | -0.042 | 0.985 | 0.867 | -0.117 |
| exp_smoothing | pure | NO | 27 | 1 | 6 | 20 | 0.814 | 0.729 | -0.086 | 1.000 | 1.000 | +0.000 |
| gaussian_int | pure | yes | 30 | 16 | 10 | 4 | 0.990 | 1.000 | +0.010 | 1.000 | 0.879 | -0.121 |
| trap_circle | pure | NO | 26 | 20 | 6 | 0 | — | — | — | 0.950 | 0.883 | -0.067 |
| curv_linear | pure | yes | 27 | 23 | 2 | 2 | 0.960 | 1.000 | +0.040 | 1.000 | 0.946 | -0.054 |
| smooth_mean_angle | composite | yes | 28 | 21 | 5 | 2 | 1.000 | 0.962 | -0.038 | 0.952 | 0.878 | -0.075 |
| smooth_angle_min | composite | yes | 28 | 12 | 11 | 5 | 0.904 | 0.965 | +0.061 | 1.000 | 0.969 | -0.031 |
| composite_c7_smooth_trap_circle | composite | yes | 29 | 23 | 4 | 2 | 1.000 | 0.963 | -0.037 | 0.993 | 1.000 | +0.007 |
| composite_c13_smooth_stripe | composite | yes | 28 | 15 | 9 | 4 | 1.000 | 1.000 | +0.000 | 0.949 | 0.897 | -0.051 |
| composite_c17_smooth_curvature | composite | yes | 29 | 18 | 9 | 2 | 1.000 | 1.000 | +0.000 | 0.919 | 0.929 | +0.010 |
| direct_trap_ring | direct | yes | 29 | 20 | 6 | 3 | 1.000 | 0.987 | -0.013 | 0.967 | 0.944 | -0.022 |
| direct_trap_screen | direct | NO | 30 | 23 | 7 | 0 | — | — | — | 1.000 | 0.969 | -0.031 |
| direct_trap_multiply | direct | yes | 26 | 13 | 11 | 2 | 1.000 | 1.000 | +0.000 | 0.970 | 1.000 | +0.030 |
| direct_trap_lines | direct | yes | 30 | 18 | 9 | 3 | 0.988 | 1.000 | +0.012 | 0.981 | 0.977 | -0.005 |

| mode | v1 AP>=3 | v2 AP>=3 | Δ | v1 AP>=2 | v2 AP>=2 | Δ |
|---|--:|--:|--:|--:|--:|--:|
| tia | 0.952 | 0.967 | +0.015 | 0.997 | 0.986 | -0.011 |
| stripe | 1.000 | 0.817 | -0.183 | 0.985 | 0.884 | -0.101 |
| exp_smoothing | 0.855 | 0.828 | -0.027 | 1.000 | 1.000 | +0.000 |
| gaussian_int | 0.950 | 1.000 | +0.050 | 1.000 | 0.861 | -0.139 |
| trap_circle | — | — | — | 0.896 | 0.719 | -0.178 |
| curv_linear | 0.583 | 1.000 | +0.417 | 1.000 | 0.861 | -0.139 |
| smooth_mean_angle | 1.000 | 0.750 | -0.250 | 0.900 | 0.815 | -0.085 |
| smooth_angle_min | 0.737 | 0.853 | +0.116 | 1.000 | 0.978 | -0.022 |
| composite_c7_smooth_trap_circle | 1.000 | 0.750 | -0.250 | 0.976 | 1.000 | +0.024 |
| composite_c13_smooth_stripe | 1.000 | 1.000 | +0.000 | 0.956 | 0.881 | -0.075 |
| composite_c17_smooth_curvature | 1.000 | 1.000 | +0.000 | 0.922 | 0.916 | -0.006 |
| direct_trap_ring | 1.000 | 0.917 | -0.083 | 0.940 | 0.879 | -0.061 |
| direct_trap_screen | — | — | — | 1.000 | 0.874 | -0.126 |
| direct_trap_multiply | 1.000 | 1.000 | +0.000 | 0.979 | 1.000 | +0.021 |
| direct_trap_lines | 0.917 | 1.000 | +0.083 | 0.971 | 0.965 | -0.005 |

## 3 · the three modes v1's trainer dropped

`trap_circle`, `exp_smoothing`, `direct_trap_screen` — v1 never trained on a row of any of them; v2 trained on all three. Pooled eval slice: **n = 83**, labels {'1': 44, '2': 19, '3': 20} (>=3 base 24.1%).


**Individually.**


- **`trap_circle`** (n=26, labels {'1': 20, '2': 6, '3': 0}) — >=3 is **not measurable** here (no labeled tier-3 on the eval side), so the finetune's effect on this mode can only be read at >=2. >=2: v1 AUC 0.950 / AP 0.896 → v2 AUC 0.883 / AP 0.719.

- **`exp_smoothing`** (n=27, labels {'1': 1, '2': 6, '3': 20}) — **the rich one: this mode is the qualitative pass/fail of the finetune.** >=3: v1 AUC 0.814 / AP 0.855 → v2 AUC 0.729 / AP 0.828. >=2: v1 AUC 1.000 / AP 1.000 → v2 AUC 1.000 / AP 1.000.

- **`direct_trap_screen`** (n=30, labels {'1': 23, '2': 7, '3': 0}) — >=3 is **not measurable** here (no labeled tier-3 on the eval side), so the finetune's effect on this mode can only be read at >=2. >=2: v1 AUC 1.000 / AP 1.000 → v2 AUC 0.969 / AP 0.874.


**Pooled, with paired CIs** (the slice the winner rule's clause (b) reads):

| metric | n pos | v1 | v2 | Δ (v2 − v1), 95% CI |
|---|--:|--:|--:|--:|
| AUC >=3 | 20 | 0.967 | 0.970 | +0.003 [-0.020, +0.029] |
| AP >=3 | 20 | 0.823 | 0.828 | +0.005 [-0.057, +0.072] |
| AUC >=2 | 39 | 0.995 | 0.976 | -0.018 [-0.048, -0.001] **worse** |
| AP >=2 | 39 | 0.995 | 0.974 | -0.019 [-0.056, -0.001] **worse** |


## 4 · score-scale check

A fixed threshold is a point on ONE head's probability scale. If the marginals differ, `p_ge3 >= 0.50` selects different VOLUMES from the two heads and a fixed-threshold table is comparing two different operating points.


Two-sample KS on eval `p_ge3`: **D = 0.118, p = 4.82e-03** — the marginal distribution **HAS** shifted. The volume-matched view below is therefore the load-bearing comparison; the fixed-threshold view is kept beside it.

| quantile of p_ge3 | v1 | v2 |
|---|--:|--:|
| q10 | 0.0003 | 0.0002 |
| q25 | 0.0063 | 0.0075 |
| q50 | 0.0340 | 0.0641 |
| q75 | 0.1530 | 0.2371 |
| q90 | 0.3935 | 0.4741 |
| q95 | 0.6090 | 0.5735 |
| q99 | 0.8743 | 0.8090 |
| mean | 0.1284 | 0.1581 |

**Fixed thresholds — pass rate of each head**

| p_ge3 >= | v1 pass rate | v2 pass rate |
|---|--:|--:|
| 0.25 | 16.6% | 24.4% |
| 0.50 | 7.8% | 8.8% |
| 0.75 | 2.8% | 1.7% |
| 0.90 | 0.7% | 0.2% |

**Volume-matched — both heads take the SAME number of rows**

| matched at | volume | v1 precision | v1 recall | v2 precision | v2 recall | v2 cut on p_ge3 |
|---|--:|--:|--:|--:|--:|--:|
| v1 @ 0.25 (mining_pool) | 70 | 75.7% [64.5%–84.2%] | 84.1% | 74.3% [63.0%–83.1%] | 82.5% | 0.3701 |
| v1 @ 0.50 (mining_release) | 33 | 97.0% [84.7%–99.5%] | 50.8% | 90.9% [76.4%–96.9%] | 47.6% | 0.5105 |
| fixed 5% pass rate | 21 | 95.2% [77.3%–99.2%] | 31.7% | 95.2% [77.3%–99.2%] | 31.7% | 0.5989 |
| fixed 10% pass rate | 42 | 92.9% [81.0%–97.5%] | 61.9% | 85.7% [72.2%–93.3%] | 57.1% | 0.4840 |
| fixed 20% pass rate | 84 | 65.5% [54.8%–74.8%] | 87.3% | 65.5% [54.8%–74.8%] | 87.3% | 0.2996 |


## 5 · the winner rule, applied

> v2 is the calibration candidate iff (a) no overall eval metric is significantly worse than v1 (95% paired-bootstrap CI on the delta not entirely below 0) AND (b) on the pooled three-dropped-mode slice at least one boundary is significantly BETTER and none is significantly worse. Evaluated on the two pre-declared slices; no per-slice cherry-picking, and the losing head keeps the candidacy rather than the tie being resolved by whichever number looks best.


**(a) overall — no metric significantly worse:** AUC >=3 OK, AP >=3 OK, AUC >=2 **FAIL**, AP >=2 **FAIL** → **FAIL**.


**(b) dropped modes improve:** measurable boundaries ['auc_ge3', 'ap_ge3', 'auc_ge2', 'ap_ge2']; at least one significantly better: **False**; any significantly worse: **True** → **FAIL**.


### → the calibration candidate is **v1** (`data/render_mode_head/v1/model_best.pt`)


## 6 · calibration on the winner (v1), eval side (n = 422)

>=3 base rate **14.9%**. Precision is of PASSERS and carries a Wilson interval — the top of any ladder is estimated from few passers, and a bare 1.000 over 3 rows and a 0.90 over 90 are the same column otherwise.

| p_ge3 >= | fires | pass rate | TP | precision | 95% CI | recall | mark |
|---|--:|--:|--:|--:|--:|--:|--:|
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

**Today's two cuts, for reference only.**

- `mining_pool` = 0.25 — fires 70/422 (16.6%), precision 75.7% [64.5%–84.2%], recall 84.1%.
- `mining_release` = 0.50 — fires 33/422 (7.8%), precision 97.0% [84.7%–99.5%], recall 50.8%.

**RELEASE-floor candidates — DERIVED, NOT ADOPTED.** The release floor is a precision question — what may ship. Lowest swept threshold reaching each target.

| target precision | lowest p_ge3 | achieved | 95% CI | recall | fires | supported by the CI? |
|---|--:|--:|--:|--:|--:|--:|
| 0.70 | 0.25 | 75.7% | 64.5%–84.2% | 84.1% | 70 | NO |
| 0.80 | 0.30 | 81.7% | 70.1%–89.4% | 77.8% | 60 | NO |
| 0.90 | 0.40 | 92.9% | 81.0%–97.5% | 61.9% | 42 | NO |

**POOL-floor candidates — DERIVED, NOT ADOPTED.** The pool floor is capacity ordering, not curation (`floors.py`), so its question is the mirror one: the HIGHEST threshold that still keeps each share of the good rows.

| retain recall >= | highest p_ge3 | recall kept | pass rate | fires | precision there | 95% CI |
|---|--:|--:|--:|--:|--:|--:|
| 0.95 | 0.15 | 98.4% | 25.4% | 107 | 57.9% | 48.5%–66.9% |
| 0.90 | 0.15 | 98.4% | 25.4% | 107 | 57.9% | 48.5%–66.9% |
| 0.80 | 0.25 | 84.1% | 16.6% | 70 | 75.7% | 64.5%–84.2% |

**The `>=2` ladder** (base rate 41.7%), for the pool cut's not-bad question.

| p_ge2 >= | fires | pass rate | precision | 95% CI | recall |
|---|--:|--:|--:|--:|--:|
| 0.00 | 422 | 100.0% | 41.7% | 37.1%–46.5% | 100.0% |
| 0.05 | 323 | 76.5% | 54.5% | 49.0%–59.8% | 100.0% |
| 0.10 | 256 | 60.7% | 67.2% | 61.2%–72.6% | 97.7% |
| 0.15 | 216 | 51.2% | 76.9% | 70.8%–82.0% | 94.3% |
| 0.20 | 187 | 44.3% | 86.6% | 81.0%–90.8% | 92.0% |
| 0.25 | 173 | 41.0% | 93.1% | 88.3%–96.0% | 91.5% |
| 0.30 | 156 | 37.0% | 97.4% | 93.6%–99.0% | 86.4% |
| 0.35 | 143 | 33.9% | 99.3% | 96.1%–99.9% | 80.7% |
| 0.40 | 122 | 28.9% | 100.0% | 96.9%–100.0% | 69.3% |
| 0.45 | 101 | 23.9% | 100.0% | 96.3%–100.0% | 57.4% |
| 0.50 | 91 | 21.6% | 100.0% | 95.9%–100.0% | 51.7% |
| 0.55 | 77 | 18.2% | 100.0% | 95.2%–100.0% | 43.8% |
| 0.60 | 70 | 16.6% | 100.0% | 94.8%–100.0% | 39.8% |
| 0.65 | 60 | 14.2% | 100.0% | 94.0%–100.0% | 34.1% |
| 0.70 | 43 | 10.2% | 100.0% | 91.8%–100.0% | 24.4% |
| 0.75 | 32 | 7.6% | 100.0% | 89.3%–100.0% | 18.2% |
| 0.80 | 24 | 5.7% | 100.0% | 86.2%–100.0% | 13.6% |
| 0.85 | 21 | 5.0% | 100.0% | 84.5%–100.0% | 11.9% |
| 0.90 | 12 | 2.8% | 100.0% | 75.7%–100.0% | 6.8% |
| 0.95 | 6 | 1.4% | 100.0% | 61.0%–100.0% | 3.4% |

DERIVED AND RECORDED, ADOPTED NOTHING. The two live cuts are marked for reference only. A cut set from this slice inherits both caveats above and would be an optimistic bound on a fresh location.
