# SHEET E re-verdict — mining v1 vs the five staged arms on BLIND labels

Generated 2026-08-11T09:31:34 · `uv run python tools/mining/sheet_e_reverdict.py`

> no pin, gate, floor, lock or annotation is written by this file. Adoption is a separate prompt.

Slice: **150 labeled rows** of 150 in `2026-08-11_render_mode_blind_v1` over 110 locations; tiers {'1': 74, '2': 70, '3': 6}; 100 rows on the four contested modes. Blind, eval-only.


## (a) Do the contested-cell regressions survive?

every clause-(a) cell any staged arm failed on the ANCHORED corpus, and what it does on the blind slice. A cell absent from an arm's row failed nothing for that arm.

| cell | contested mode | arms that failed it anchored | arms where it SURVIVES blind | per-arm verdict |
|---|:--:|---:|---:|---|
| `mode:composite_c17_smooth_curvature.auc_ge2` | no | 3 | **0** | v3_ap2: NOT SIGNIFICANT (underpowered) · v3_aug: NOT SIGNIFICANT (underpowered) · v3_uniform: NOT SIGNIFICANT (underpowered) |
| `mode:composite_c17_smooth_curvature.auc_ge3` | no | 1 | **0** | v3_ap2: UNMEASURABLE |
| `mode:curv_linear.auc_ge2` | yes | 5 | **0** | v3: DOES NOT SURVIVE · v3_ap2: DOES NOT SURVIVE · v3_aug: DOES NOT SURVIVE · v3_augx: DOES NOT SURVIVE · v3_uniform: DOES NOT SURVIVE |
| `mode:direct_trap_lines.auc_ge2` | yes | 1 | **0** | v3: DOES NOT SURVIVE |
| `mode:direct_trap_lines.auc_ge3` | yes | 5 | **1** | v3: DOES NOT SURVIVE · v3_ap2: DOES NOT SURVIVE · v3_aug: SURVIVES · v3_augx: DOES NOT SURVIVE · v3_uniform: DOES NOT SURVIVE |
| `mode:direct_trap_ring.auc_ge2` | yes | 5 | **1** | v3: DOES NOT SURVIVE · v3_ap2: DOES NOT SURVIVE · v3_aug: DOES NOT SURVIVE · v3_augx: SURVIVES · v3_uniform: DOES NOT SURVIVE |
| `mode:direct_trap_screen.auc_ge2` | yes | 4 | **0** | v3: DOES NOT SURVIVE · v3_aug: DOES NOT SURVIVE · v3_augx: DOES NOT SURVIVE · v3_uniform: DOES NOT SURVIVE |
| `mode:direct_trap_screen.auc_ge3` | yes | 1 | **0** | v3: UNMEASURABLE |
| `mode:gaussian_int.auc_ge2` | no | 1 | **0** | v3_augx: UNMEASURABLE |
| `mode:gaussian_int.auc_ge3` | no | 2 | **0** | v3_ap2: UNMEASURABLE · v3_aug: UNMEASURABLE |
| `mode:smooth_angle_min.auc_ge2` | no | 1 | **0** | v3_ap2: NOT SIGNIFICANT (underpowered) |
| `mode:smooth_mean_angle.auc_ge2` | no | 1 | **0** | v3_augx: NOT SIGNIFICANT (underpowered) |
| `mode:stripe.auc_ge2` | no | 1 | **0** | v3_augx: NOT SIGNIFICANT (underpowered) |
| `mode:tia.auc_ge2` | no | 1 | **0** | v3_uniform: NOT SIGNIFICANT (underpowered) |
| `pooled.ap_ge2` | no | 3 | **0** | v3_aug: DOES NOT SURVIVE · v3_augx: DOES NOT SURVIVE · v3_uniform: DOES NOT SURVIVE |
| `pooled.auc_ge2` | no | 5 | **0** | v3: DOES NOT SURVIVE · v3_ap2: DOES NOT SURVIVE · v3_aug: DOES NOT SURVIVE · v3_augx: DOES NOT SURVIVE · v3_uniform: DOES NOT SURVIVE |

## (b) Pooled reads, per arm

| arm | AUC≥2 v1 | AUC≥2 arm | Δ 95% CI | AUC≥3 v1 | AUC≥3 arm | Δ 95% CI | clause (a) on this slice |
|---|---:|---:|---|---:|---:|---|---|
| `v3` (v3) | 0.676 | 0.709 | [-0.031, +0.095]  | 0.818 | 0.954 | [-0.043, +0.341]  | PASS (0/21 cells) |
| `v3_aug` (aug_gentle) | 0.676 | 0.663 | [-0.093, +0.062]  | 0.818 | 0.843 | [-0.305, +0.321]  | FAIL (1/21 cells) |
| `v3_augx` (aug_strong) | 0.676 | 0.674 | [-0.060, +0.057]  | 0.818 | 0.922 | [-0.116, +0.329]  | FAIL (1/21 cells) |
| `v3_uniform` (uniform) | 0.676 | 0.674 | [-0.083, +0.079]  | 0.818 | 0.928 | [-0.128, +0.347]  | PASS (0/21 cells) |
| `v3_ap2` (ap2_selected) | 0.676 | 0.711 | [-0.023, +0.098]  | 0.818 | 0.929 | [-0.066, +0.313]  | PASS (0/21 cells) |

| arm | AP≥2 v1 | AP≥2 arm | Δ 95% CI | AP≥3 v1 | AP≥3 arm | Δ 95% CI |
|---|---:|---:|---|---:|---:|---|
| `v3` | 0.686 | 0.740 | [-0.018, +0.129]  | 0.306 | 0.684 | [-0.038, +0.784]  |
| `v3_aug` | 0.686 | 0.715 | [-0.046, +0.101]  | 0.306 | 0.481 | [-0.133, +0.601]  |
| `v3_augx` | 0.686 | 0.715 | [-0.042, +0.102]  | 0.306 | 0.635 | [-0.065, +0.744]  |
| `v3_uniform` | 0.686 | 0.711 | [-0.056, +0.109]  | 0.306 | 0.596 | [-0.056, +0.710]  |
| `v3_ap2` | 0.686 | 0.734 | [-0.025, +0.121]  | 0.306 | 0.400 | [-0.105, +0.311]  |

## (c) The anchoring price

v1's pooled AUC>=2 on rows whose labels it SUGGESTED, against v1's pooled AUC>=2 on fresh (location, mode) pairs of the same population labeled blind. The gap is the anchoring price — how much of the 0.953 was agreement rather than quality (classifier_retrain_protocol.md 2b). AUC>=2 is the boundary named because it is the one every staged arm loses on.

| slice | labels elicited | n | v1 AUC≥2 | v1 AUC≥3 |
|---|---|---:|---:|---:|
| anchored corpus `pooled` | v1's tier PREFILLED, page sorted by v1 | 827 | 0.953 | 0.839 |
| sheet E `pooled` | BLIND, shuffled | 150 | 0.676 | 0.818 |

**Δ AUC≥2 (blind − anchored) = -0.278** · Δ AUC≥3 = -0.021 — the blind slice is the LOWER number; the difference is what the prefilled suggestion bought v1 on a slice it was measured on


## Per-mode reads (v1 vs each arm, AUC cells only — the (28) voting rule)

| mode | n | ≥3 | v1 AUC≥2 | v1 AUC≥3 | v3 AUC≥2 | v3_aug AUC≥2 | v3_augx AUC≥2 | v3_uniform AUC≥2 | v3_ap2 AUC≥2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `composite_c13_smooth_stripe` ⚠ | 5 | 0 | 0.750 | — | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 |
| `composite_c17_smooth_curvature` ⚠ | 5 | 0 | 1.000 | — | 1.000 | 0.667 | 0.667 | 0.833 | 1.000 |
| `composite_c7_smooth_trap_circle` ⚠ | 5 | 1 | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `curv_linear` | 25 | 0 | 0.600 | — | 0.680 | 0.520 | 0.573 | 0.653 | 0.540 |
| `direct_trap_lines` | 25 | 2 | 0.583 | 0.848 | 0.763 | 0.692 | 0.731 | 0.667 | 0.686 |
| `direct_trap_multiply` ⚠ | 5 | 0 | 0.500 | — | 0.500 | 0.500 | 0.833 | 0.333 | 0.667 |
| `direct_trap_ring` | 25 | 0 | 0.747 | — | 0.727 | 0.620 | 0.620 | 0.647 | 0.660 |
| `direct_trap_screen` | 25 | 0 | 0.757 | — | 0.794 | 0.846 | 0.728 | 0.765 | 0.801 |
| `gaussian_int` ⚠ | 5 | 0 | — | — | — | — | — | — | — |
| `smooth_angle_min` ⚠ | 5 | 1 | 1.000 | 0.750 | 0.833 | 0.833 | 0.833 | 0.667 | 0.833 |
| `smooth_mean_angle` ⚠ | 5 | 0 | 0.000 | — | 0.500 | 0.250 | 0.000 | 0.500 | 0.250 |
| `stripe` ⚠ | 5 | 1 | 0.833 | 1.000 | 0.667 | 0.833 | 0.833 | 0.833 | 1.000 |
| `tia` ⚠ | 5 | 1 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 0.750 | 0.750 |
| `trap_circle` ⚠ | 5 | 0 | — | — | — | — | — | — | — |

⚠ = under the 20-row floor; its CI is reported and votes, but read it as underpowered.


## Cells that vote neither way

- `mode:composite_c13_smooth_stripe.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:composite_c17_smooth_curvature.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:curv_linear.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=25) — one class only, so this cell votes neither way`
- `mode:direct_trap_multiply.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:direct_trap_ring.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=25) — one class only, so this cell votes neither way`
- `mode:direct_trap_screen.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=25) — one class only, so this cell votes neither way`
- `mode:gaussian_int.auc_ge2: no positives at label >= 2 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:gaussian_int.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:smooth_mean_angle.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:trap_circle.auc_ge2: no positives at label >= 2 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`
- `mode:trap_circle.auc_ge3: no positives at label >= 3 (n_pos=0, n_neg=5) — one class only, so this cell votes neither way`


### `v3_aug` clause (a) failures on this slice

- `mode:direct_trap_lines` **auc_ge3** (n=25): Δ median -0.348, 95% CI [-0.667, -0.042]

### `v3_augx` clause (a) failures on this slice

- `mode:direct_trap_ring` **auc_ge2** (n=25): Δ median -0.122, 95% CI [-0.267, -0.013]

clause (a) is a conjunction over 21 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


NOT DECIDED HERE. The (28) motivating arm (`busy_fp`) is defined by sheets B and C's own draw buckets and has no analogue on this slice; choosing one after seeing these rows would be choosing it from the data.

