# mining v1 vs v3 [arm: ap2_selected] — winner-rule verdict (STAGED, nothing adopted)

Generated 2026-08-11T01:24:21 · `uv run python tools/mining/mining_v3_reads.py --candidate-dir data/render_mode_head/v3_ap2`

Arm dials: `{'border_crop': 0.05, 'axis_crop': 0.0, 'uniform_weights': False, 'row_weighting': '1/near_dup_group_size on the TRAIN side; the eval side keeps one row per group (data/render_mode_corpus/near_dup_groups_v1.json)', 'selection_metric': 'ap_ge2'}`

**WINNER: v1** (pooled-only reading: v1)  
clause (a) no-worse FAIL over 38 arm x metric cells · clause (b) motivating PASS

> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Eval slice: **827 rows** (973 before near-dup dedup) over 136 locations, {'v1_sitting': 393, 'sheetB': 227, 'sheetC': 207}; tiers {'1': 261, '2': 352, '3': 214}.

Harness parity (v1 re-scored vs stamped): max |Δ| 4.30e-08 over 827 rows — OK.


## MOTIVATING arm

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `busy_fp` | 146 | 51 | 0.567 | 0.675 | [-0.025, +0.233]  | 0.424 | 0.551 | [-0.011, +0.265]  |

## NO-WORSE arms

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `pooled` | 827 | 214 | 0.839 | 0.833 | [-0.041, +0.027]  | 0.613 | 0.601 | [-0.081, +0.060]  |
| `rare_palette` | 207 | 96 | 0.652 | 0.708 | [-0.039, +0.151]  | 0.606 | 0.658 | [-0.053, +0.148]  |
| `mode:composite_c13_smooth_stripe` | 60 | 17 | 0.871 | 0.910 | [-0.104, +0.182]  | 0.731 | 0.876 | [-0.084, +0.398]  |
| `mode:composite_c17_smooth_curvature` | 39 | 3 | 0.889 | 0.694 | [-0.386, -0.032] **worse** | 0.627 | 0.164 | [-0.875, -0.016] **worse** |
| `mode:composite_c7_smooth_trap_circle` | 40 | 6 | 0.956 | 0.946 | [-0.086, +0.066]  | 0.733 | 0.691 | [-0.427, +0.341]  |
| `mode:curv_linear` | 43 | 7 | 0.877 | 0.873 | [-0.131, +0.108]  | 0.585 | 0.738 | [-0.078, +0.433]  |
| `mode:direct_trap_lines` | 69 | 10 | 0.895 | 0.607 | [-0.489, -0.100] **worse** | 0.565 | 0.244 | [-0.573, -0.037] **worse** |
| `mode:direct_trap_multiply` | 74 | 15 | 0.776 | 0.781 | [-0.140, +0.152]  | 0.444 | 0.430 | [-0.274, +0.244]  |
| `mode:direct_trap_ring` | 49 | 8 | 0.890 | 0.753 | [-0.323, +0.028]  | 0.507 | 0.471 | [-0.398, +0.294]  |
| `mode:direct_trap_screen` | 56 | 8 | 0.940 | 0.859 | [-0.258, +0.070]  | 0.708 | 0.592 | [-0.504, +0.341]  |
| `mode:exp_smoothing` | 70 | 43 | 0.670 | 0.645 | [-0.170, +0.109]  | 0.710 | 0.693 | [-0.137, +0.095]  |
| `mode:gaussian_int` | 48 | 7 | 0.916 | 0.742 | [-0.386, -0.005] **worse** | 0.725 | 0.556 | [-0.465, +0.159]  |
| `mode:smooth` | 16 | 10 | 0.883 | 0.783 | [-0.467, +0.209]  | 0.929 | 0.817 | [-0.375, +0.114]  |
| `mode:smooth_angle_min` | 40 | 10 | 0.730 | 0.717 | [-0.191, +0.179]  | 0.640 | 0.423 | [-0.429, +0.048]  |
| `mode:smooth_mean_angle` | 39 | 8 | 0.855 | 0.907 | [-0.071, +0.207]  | 0.636 | 0.604 | [-0.303, +0.251]  |
| `mode:stripe` | 59 | 17 | 0.833 | 0.772 | [-0.211, +0.086]  | 0.666 | 0.564 | [-0.317, +0.148]  |
| `mode:tia` | 91 | 45 | 0.744 | 0.704 | [-0.155, +0.073]  | 0.736 | 0.650 | [-0.213, +0.040]  |
| `mode:trap_circle` | 34 | 0 | — | — | n/a | — | — | n/a |

## Diagnostics (vote on nothing)

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fancy_all` | 466 | 85 | 0.836 | 0.818 | [-0.072, +0.039]  | 0.516 | 0.510 | [-0.127, +0.112]  |
| `pure_all` | 361 | 129 | 0.833 | 0.821 | [-0.057, +0.030]  | 0.693 | 0.661 | [-0.116, +0.051]  |
| `sheetB` | 227 | 47 | 0.782 | 0.863 | [+0.008, +0.157] **better** | 0.536 | 0.634 | [-0.063, +0.261]  |
| `sheetB_hi_fancy` | 32 | 1 | 1.000 | 0.968 | [-0.100, +0.000]  | 1.000 | 0.500 | [-0.750, +0.000]  |
| `sheetC_fancy` | 114 | 50 | 0.599 | 0.684 | [-0.048, +0.220]  | 0.527 | 0.636 | [-0.031, +0.242]  |
| `v1_sitting` | 393 | 71 | 0.945 | 0.866 | [-0.125, -0.033] **worse** | 0.812 | 0.531 | [-0.381, -0.176] **worse** |
| `v1_unseen_locations` | 197 | 89 | 0.648 | 0.701 | [-0.043, +0.150]  | 0.580 | 0.640 | [-0.044, +0.160]  |

### clause (a) failures

- `mode:composite_c17_smooth_curvature` **auc_ge2**: Δ median -0.127, 95% CI [-0.251, -0.037]
- `mode:composite_c17_smooth_curvature` **auc_ge3**: Δ median -0.189, 95% CI [-0.386, -0.032]
- `mode:curv_linear` **auc_ge2**: Δ median -0.039, 95% CI [-0.105, -0.004]
- `mode:direct_trap_lines` **auc_ge3**: Δ median -0.287, 95% CI [-0.489, -0.100]
- `mode:direct_trap_ring` **auc_ge2**: Δ median -0.132, 95% CI [-0.261, -0.035]
- `mode:gaussian_int` **auc_ge3**: Δ median -0.165, 95% CI [-0.386, -0.005]
- `mode:smooth_angle_min` **auc_ge2**: Δ median -0.086, 95% CI [-0.219, -0.003]
- `pooled` **auc_ge2**: Δ median -0.021, 95% CI [-0.039, -0.005]

### clause (b) improvements

- `busy_fp` **ap_ge2**: Δ median +0.026, 95% CI [+0.007, +0.054]
- `busy_fp` **auc_ge2**: Δ median +0.366, 95% CI [+0.186, +0.526]

clause (a) is a conjunction over 38 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


## v3 five-seed band (staged = the max of these, on this same slice)

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.846 ± 0.012 | 0.839 0.852 0.865 0.833 0.840 |
| AP>=3 | 0.611 ± 0.020 | 0.577 0.627 0.630 0.601 0.621 |
| AUC>=2 | 0.930 ± 0.002 | 0.932 0.926 0.928 0.932 0.930 |
| AP>=2 | 0.965 ± 0.002 | 0.966 0.962 0.963 0.967 0.965 |

## Volume-matched (a fixed threshold is a point on ONE head's scale)

| volume | v1 precision≥3 | v3 precision≥3 | n |
|---|---:|---:|---:|
| mining_pool (v1 @ 0.25) | 0.534 | 0.516 | 322 |
| mining_release (v1 @ 0.5) | 0.636 | 0.628 | 129 |
| top 0.05 | 0.707 | 0.683 | 41 |
| top 0.10 | 0.699 | 0.651 | 83 |
| top 0.20 | 0.636 | 0.630 | 165 |
