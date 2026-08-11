# mining v1 vs v3 [arm: aug_gentle] — winner-rule verdict (STAGED, nothing adopted)

Generated 2026-08-10T21:56:53 · `uv run python tools/mining/mining_v3_reads.py --candidate-dir data/render_mode_head/v3_aug`

Arm dials: `{'border_crop': 0.0, 'axis_crop': 0.03, 'uniform_weights': False, 'row_weighting': '1/near_dup_group_size on the TRAIN side; the eval side keeps one row per group (data/render_mode_corpus/near_dup_groups_v1.json)'}`

**WINNER: v1** (pooled-only reading: v1)  
clause (a) no-worse FAIL over 38 arm x metric cells · clause (b) motivating PASS

> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Eval slice: **827 rows** (973 before near-dup dedup) over 136 locations, {'v1_sitting': 393, 'sheetB': 227, 'sheetC': 207}; tiers {'1': 261, '2': 352, '3': 214}.

Harness parity (v1 re-scored vs stamped): max |Δ| 4.30e-08 over 827 rows — OK.


## MOTIVATING arm

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `busy_fp` | 146 | 51 | 0.567 | 0.693 | [-0.003, +0.250]  | 0.424 | 0.586 | [+0.035, +0.283] **better** |

## NO-WORSE arms

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `pooled` | 827 | 214 | 0.839 | 0.871 | [-0.001, +0.064]  | 0.613 | 0.682 | [+0.002, +0.137] **better** |
| `rare_palette` | 207 | 96 | 0.652 | 0.733 | [-0.007, +0.169]  | 0.606 | 0.657 | [-0.040, +0.144]  |
| `mode:composite_c13_smooth_stripe` | 60 | 17 | 0.871 | 0.941 | [-0.024, +0.179]  | 0.731 | 0.840 | [-0.100, +0.336]  |
| `mode:composite_c17_smooth_curvature` | 39 | 3 | 0.889 | 0.926 | [-0.158, +0.316]  | 0.627 | 0.443 | [-0.800, +0.866]  |
| `mode:composite_c7_smooth_trap_circle` | 40 | 6 | 0.956 | 0.971 | [-0.054, +0.090]  | 0.733 | 0.882 | [-0.173, +0.496]  |
| `mode:curv_linear` | 43 | 7 | 0.877 | 0.925 | [-0.023, +0.144]  | 0.585 | 0.773 | [-0.024, +0.454]  |
| `mode:direct_trap_lines` | 69 | 10 | 0.895 | 0.680 | [-0.404, -0.041] **worse** | 0.565 | 0.268 | [-0.563, -0.023] **worse** |
| `mode:direct_trap_multiply` | 74 | 15 | 0.776 | 0.769 | [-0.148, +0.130]  | 0.444 | 0.514 | [-0.180, +0.303]  |
| `mode:direct_trap_ring` | 49 | 8 | 0.890 | 0.835 | [-0.275, +0.106]  | 0.507 | 0.535 | [-0.255, +0.336]  |
| `mode:direct_trap_screen` | 56 | 8 | 0.940 | 0.768 | [-0.396, +0.024]  | 0.708 | 0.392 | [-0.642, +0.132]  |
| `mode:exp_smoothing` | 70 | 43 | 0.670 | 0.780 | [-0.027, +0.244]  | 0.710 | 0.863 | [-0.001, +0.274]  |
| `mode:gaussian_int` | 48 | 7 | 0.916 | 0.739 | [-0.381, -0.009] **worse** | 0.725 | 0.502 | [-0.526, +0.005]  |
| `mode:smooth` | 16 | 10 | 0.883 | 0.850 | [-0.444, +0.312]  | 0.929 | 0.848 | [-0.358, +0.177]  |
| `mode:smooth_angle_min` | 40 | 10 | 0.730 | 0.820 | [-0.108, +0.292]  | 0.640 | 0.644 | [-0.254, +0.269]  |
| `mode:smooth_mean_angle` | 39 | 8 | 0.855 | 0.863 | [-0.135, +0.157]  | 0.636 | 0.571 | [-0.259, +0.180]  |
| `mode:stripe` | 59 | 17 | 0.833 | 0.828 | [-0.158, +0.147]  | 0.666 | 0.612 | [-0.211, +0.152]  |
| `mode:tia` | 91 | 45 | 0.744 | 0.796 | [-0.048, +0.151]  | 0.736 | 0.721 | [-0.138, +0.112]  |
| `mode:trap_circle` | 34 | 0 | — | — | n/a | — | — | n/a |

## Diagnostics (vote on nothing)

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fancy_all` | 466 | 85 | 0.836 | 0.844 | [-0.050, +0.064]  | 0.516 | 0.544 | [-0.090, +0.144]  |
| `pure_all` | 361 | 129 | 0.833 | 0.870 | [-0.002, +0.076]  | 0.693 | 0.752 | [-0.019, +0.133]  |
| `sheetB` | 227 | 47 | 0.782 | 0.903 | [+0.049, +0.192] **better** | 0.536 | 0.736 | [+0.067, +0.338] **better** |
| `sheetB_hi_fancy` | 32 | 1 | 1.000 | 1.000 | [+0.000, +0.000]  | 1.000 | 1.000 | [+0.000, +0.000]  |
| `sheetC_fancy` | 114 | 50 | 0.599 | 0.681 | [-0.052, +0.216]  | 0.527 | 0.629 | [-0.026, +0.229]  |
| `v1_sitting` | 393 | 71 | 0.945 | 0.932 | [-0.046, +0.023]  | 0.812 | 0.759 | [-0.162, +0.047]  |
| `v1_unseen_locations` | 197 | 89 | 0.648 | 0.731 | [-0.007, +0.180]  | 0.580 | 0.645 | [-0.028, +0.165]  |

### clause (a) failures

- `mode:composite_c17_smooth_curvature` **auc_ge2**: Δ median -0.106, 95% CI [-0.211, -0.029]
- `mode:curv_linear` **auc_ge2**: Δ median -0.071, 95% CI [-0.173, -0.009]
- `mode:direct_trap_lines` **auc_ge3**: Δ median -0.215, 95% CI [-0.404, -0.041]
- `mode:direct_trap_ring` **auc_ge2**: Δ median -0.176, 95% CI [-0.317, -0.073]
- `mode:direct_trap_screen` **auc_ge2**: Δ median -0.094, 95% CI [-0.229, -0.008]
- `mode:gaussian_int` **auc_ge3**: Δ median -0.166, 95% CI [-0.381, -0.009]
- `pooled` **ap_ge2**: Δ median -0.017, 95% CI [-0.028, -0.006]
- `pooled` **auc_ge2**: Δ median -0.036, 95% CI [-0.054, -0.020]

### clause (b) improvements

- `busy_fp` **ap_ge2**: Δ median +0.021, 95% CI [+0.006, +0.044]
- `busy_fp` **ap_ge3**: Δ median +0.160, 95% CI [+0.035, +0.283]
- `busy_fp` **auc_ge2**: Δ median +0.282, 95% CI [+0.148, +0.431]

clause (a) is a conjunction over 38 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


## v3 five-seed band (staged = the max of these, on this same slice)

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.860 ± 0.008 | 0.869 0.855 0.853 0.871 0.852 |
| AP>=3 | 0.650 ± 0.022 | 0.651 0.665 0.623 0.682 0.631 |
| AUC>=2 | 0.917 ± 0.009 | 0.924 0.925 0.916 0.917 0.901 |
| AP>=2 | 0.954 ± 0.007 | 0.962 0.961 0.949 0.955 0.944 |

## Volume-matched (a fixed threshold is a point on ONE head's scale)

| volume | v1 precision≥3 | v3 precision≥3 | n |
|---|---:|---:|---:|
| mining_pool (v1 @ 0.25) | 0.534 | 0.553 | 322 |
| mining_release (v1 @ 0.5) | 0.636 | 0.713 | 129 |
| top 0.05 | 0.707 | 0.732 | 41 |
| top 0.10 | 0.699 | 0.747 | 83 |
| top 0.20 | 0.636 | 0.691 | 165 |
