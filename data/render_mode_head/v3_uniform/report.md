# mining v1 vs v3 [arm: uniform] — winner-rule verdict (STAGED, nothing adopted)

Generated 2026-08-10T23:58:41 · `uv run python tools/mining/mining_v3_reads.py --candidate-dir data/render_mode_head/v3_uniform`

Arm dials: `{'border_crop': 0.05, 'axis_crop': 0.0, 'uniform_weights': True, 'row_weighting': 'UNIFORM — the 1/group_size weights are LIFTED for this arm; the grouping still governs the split and the eval dedup'}`

**WINNER: v1** (pooled-only reading: v1)  
clause (a) no-worse FAIL over 38 arm x metric cells · clause (b) motivating PASS

> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Eval slice: **827 rows** (973 before near-dup dedup) over 136 locations, {'v1_sitting': 393, 'sheetB': 227, 'sheetC': 207}; tiers {'1': 261, '2': 352, '3': 214}.

Harness parity (v1 re-scored vs stamped): max |Δ| 4.30e-08 over 827 rows — OK.


## MOTIVATING arm

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `busy_fp` | 146 | 51 | 0.567 | 0.685 | [-0.010, +0.237]  | 0.424 | 0.556 | [+0.013, +0.256] **better** |

## NO-WORSE arms

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `pooled` | 827 | 214 | 0.839 | 0.856 | [-0.016, +0.049]  | 0.613 | 0.664 | [-0.011, +0.116]  |
| `rare_palette` | 207 | 96 | 0.652 | 0.756 | [+0.014, +0.191] **better** | 0.606 | 0.725 | [+0.025, +0.205] **better** |
| `mode:composite_c13_smooth_stripe` | 60 | 17 | 0.871 | 0.903 | [-0.070, +0.142]  | 0.731 | 0.825 | [-0.084, +0.285]  |
| `mode:composite_c17_smooth_curvature` | 39 | 3 | 0.889 | 0.963 | [-0.057, +0.316]  | 0.627 | 0.589 | [-0.583, +0.866]  |
| `mode:composite_c7_smooth_trap_circle` | 40 | 6 | 0.956 | 0.975 | [-0.054, +0.108]  | 0.733 | 0.877 | [-0.211, +0.520]  |
| `mode:curv_linear` | 43 | 7 | 0.877 | 0.937 | [-0.012, +0.150]  | 0.585 | 0.784 | [+0.006, +0.417] **better** |
| `mode:direct_trap_lines` | 69 | 10 | 0.895 | 0.649 | [-0.441, -0.053] **worse** | 0.565 | 0.220 | [-0.613, -0.074] **worse** |
| `mode:direct_trap_multiply` | 74 | 15 | 0.776 | 0.820 | [-0.078, +0.167]  | 0.444 | 0.539 | [-0.095, +0.335]  |
| `mode:direct_trap_ring` | 49 | 8 | 0.890 | 0.854 | [-0.160, +0.076]  | 0.507 | 0.555 | [-0.285, +0.340]  |
| `mode:direct_trap_screen` | 56 | 8 | 0.940 | 0.740 | [-0.427, +0.014]  | 0.708 | 0.405 | [-0.689, +0.251]  |
| `mode:exp_smoothing` | 70 | 43 | 0.670 | 0.609 | [-0.163, +0.037]  | 0.710 | 0.744 | [-0.073, +0.114]  |
| `mode:gaussian_int` | 48 | 7 | 0.916 | 0.819 | [-0.267, +0.019]  | 0.725 | 0.596 | [-0.358, +0.124]  |
| `mode:smooth` | 16 | 10 | 0.883 | 0.833 | [-0.283, +0.182]  | 0.929 | 0.905 | [-0.186, +0.126]  |
| `mode:smooth_angle_min` | 40 | 10 | 0.730 | 0.793 | [-0.117, +0.250]  | 0.640 | 0.608 | [-0.267, +0.192]  |
| `mode:smooth_mean_angle` | 39 | 8 | 0.855 | 0.815 | [-0.250, +0.148]  | 0.636 | 0.555 | [-0.302, +0.181]  |
| `mode:stripe` | 59 | 17 | 0.833 | 0.776 | [-0.216, +0.088]  | 0.666 | 0.656 | [-0.183, +0.154]  |
| `mode:tia` | 91 | 45 | 0.744 | 0.847 | [-0.010, +0.221]  | 0.736 | 0.814 | [-0.042, +0.202]  |
| `mode:trap_circle` | 34 | 0 | — | — | n/a | — | — | n/a |

## Diagnostics (vote on nothing)

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fancy_all` | 466 | 85 | 0.836 | 0.835 | [-0.052, +0.050]  | 0.516 | 0.545 | [-0.079, +0.137]  |
| `pure_all` | 361 | 129 | 0.833 | 0.854 | [-0.021, +0.065]  | 0.693 | 0.738 | [-0.023, +0.110]  |
| `sheetB` | 227 | 47 | 0.782 | 0.838 | [-0.019, +0.133]  | 0.536 | 0.615 | [-0.038, +0.196]  |
| `sheetB_hi_fancy` | 32 | 1 | 1.000 | 1.000 | [+0.000, +0.000]  | 1.000 | 1.000 | [+0.000, +0.000]  |
| `sheetC_fancy` | 114 | 50 | 0.599 | 0.675 | [-0.060, +0.212]  | 0.527 | 0.629 | [-0.027, +0.237]  |
| `v1_sitting` | 393 | 71 | 0.945 | 0.910 | [-0.077, +0.007]  | 0.812 | 0.684 | [-0.226, -0.035] **worse** |
| `v1_unseen_locations` | 197 | 89 | 0.648 | 0.755 | [+0.016, +0.202] **better** | 0.580 | 0.717 | [+0.043, +0.231] **better** |

### clause (a) failures

- `mode:composite_c17_smooth_curvature` **auc_ge2**: Δ median -0.095, 95% CI [-0.198, -0.016]
- `mode:curv_linear` **auc_ge2**: Δ median -0.185, 95% CI [-0.329, -0.076]
- `mode:direct_trap_lines` **auc_ge3**: Δ median -0.244, 95% CI [-0.441, -0.053]
- `mode:direct_trap_ring` **auc_ge2**: Δ median -0.262, 95% CI [-0.428, -0.120]
- `mode:direct_trap_screen` **auc_ge2**: Δ median -0.148, 95% CI [-0.319, -0.022]
- `mode:tia` **auc_ge2**: Δ median -0.105, 95% CI [-0.224, -0.019]
- `pooled` **ap_ge2**: Δ median -0.033, 95% CI [-0.053, -0.016]
- `pooled` **auc_ge2**: Δ median -0.061, 95% CI [-0.085, -0.038]

### clause (b) improvements

- `busy_fp` **ap_ge3**: Δ median +0.130, 95% CI [+0.013, +0.256]

clause (a) is a conjunction over 38 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


## v3 five-seed band (staged = the max of these, on this same slice)

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.858 ± 0.005 | 0.855 0.852 0.863 0.856 0.867 |
| AP>=3 | 0.649 ± 0.008 | 0.644 0.642 0.650 0.664 0.643 |
| AUC>=2 | 0.911 ± 0.015 | 0.927 0.894 0.925 0.892 0.919 |
| AP>=2 | 0.952 ± 0.009 | 0.962 0.945 0.962 0.938 0.954 |

## Volume-matched (a fixed threshold is a point on ONE head's scale)

| volume | v1 precision≥3 | v3 precision≥3 | n |
|---|---:|---:|---:|
| mining_pool (v1 @ 0.25) | 0.534 | 0.547 | 322 |
| mining_release (v1 @ 0.5) | 0.636 | 0.705 | 129 |
| top 0.05 | 0.707 | 0.854 | 41 |
| top 0.10 | 0.699 | 0.723 | 83 |
| top 0.20 | 0.636 | 0.655 | 165 |
