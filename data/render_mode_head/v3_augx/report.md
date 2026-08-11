# mining v1 vs v3 [arm: aug_strong] — winner-rule verdict (STAGED, nothing adopted)

Generated 2026-08-10T23:00:36 · `uv run python tools/mining/mining_v3_reads.py --candidate-dir data/render_mode_head/v3_augx`

Arm dials: `{'border_crop': 0.1, 'axis_crop': 0.03, 'uniform_weights': False, 'row_weighting': '1/near_dup_group_size on the TRAIN side; the eval side keeps one row per group (data/render_mode_corpus/near_dup_groups_v1.json)'}`

**WINNER: v1** (pooled-only reading: v1)  
clause (a) no-worse FAIL over 38 arm x metric cells · clause (b) motivating PASS

> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Eval slice: **827 rows** (973 before near-dup dedup) over 136 locations, {'v1_sitting': 393, 'sheetB': 227, 'sheetC': 207}; tiers {'1': 261, '2': 352, '3': 214}.

Harness parity (v1 re-scored vs stamped): max |Δ| 4.30e-08 over 827 rows — OK.


## MOTIVATING arm

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `busy_fp` | 146 | 51 | 0.567 | 0.695 | [+0.007, +0.243] **better** | 0.424 | 0.548 | [+0.010, +0.237] **better** |

## NO-WORSE arms

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `pooled` | 827 | 214 | 0.839 | 0.873 | [+0.005, +0.065] **better** | 0.613 | 0.689 | [+0.019, +0.134] **better** |
| `rare_palette` | 207 | 96 | 0.652 | 0.777 | [+0.042, +0.211] **better** | 0.606 | 0.741 | [+0.046, +0.222] **better** |
| `mode:composite_c13_smooth_stripe` | 60 | 17 | 0.871 | 0.877 | [-0.095, +0.105]  | 0.731 | 0.705 | [-0.166, +0.137]  |
| `mode:composite_c17_smooth_curvature` | 39 | 3 | 0.889 | 0.935 | [+0.000, +0.189]  | 0.627 | 0.667 | [+0.000, +0.167]  |
| `mode:composite_c7_smooth_trap_circle` | 40 | 6 | 0.956 | 0.966 | [-0.070, +0.086]  | 0.733 | 0.889 | [-0.139, +0.507]  |
| `mode:curv_linear` | 43 | 7 | 0.877 | 0.917 | [-0.005, +0.100]  | 0.585 | 0.745 | [-0.009, +0.407]  |
| `mode:direct_trap_lines` | 69 | 10 | 0.895 | 0.764 | [-0.259, -0.017] **worse** | 0.565 | 0.338 | [-0.458, +0.012]  |
| `mode:direct_trap_multiply` | 74 | 15 | 0.776 | 0.816 | [-0.094, +0.178]  | 0.444 | 0.539 | [-0.124, +0.336]  |
| `mode:direct_trap_ring` | 49 | 8 | 0.890 | 0.918 | [-0.100, +0.153]  | 0.507 | 0.738 | [-0.146, +0.516]  |
| `mode:direct_trap_screen` | 56 | 8 | 0.940 | 0.880 | [-0.219, +0.064]  | 0.708 | 0.511 | [-0.527, +0.243]  |
| `mode:exp_smoothing` | 70 | 43 | 0.670 | 0.730 | [-0.037, +0.157]  | 0.710 | 0.793 | [+0.001, +0.168] **better** |
| `mode:gaussian_int` | 48 | 7 | 0.916 | 0.878 | [-0.200, +0.114]  | 0.725 | 0.605 | [-0.420, +0.247]  |
| `mode:smooth` | 16 | 10 | 0.883 | 0.900 | [-0.292, +0.312]  | 0.929 | 0.926 | [-0.203, +0.186]  |
| `mode:smooth_angle_min` | 40 | 10 | 0.730 | 0.763 | [-0.182, +0.256]  | 0.640 | 0.499 | [-0.436, +0.243]  |
| `mode:smooth_mean_angle` | 39 | 8 | 0.855 | 0.835 | [-0.243, +0.182]  | 0.636 | 0.625 | [-0.223, +0.195]  |
| `mode:stripe` | 59 | 17 | 0.833 | 0.794 | [-0.199, +0.105]  | 0.666 | 0.709 | [-0.149, +0.214]  |
| `mode:tia` | 91 | 45 | 0.744 | 0.844 | [-0.003, +0.206]  | 0.736 | 0.805 | [-0.050, +0.190]  |
| `mode:trap_circle` | 34 | 0 | — | — | n/a | — | — | n/a |

## Diagnostics (vote on nothing)

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fancy_all` | 466 | 85 | 0.836 | 0.847 | [-0.037, +0.059]  | 0.516 | 0.543 | [-0.068, +0.123]  |
| `pure_all` | 361 | 129 | 0.833 | 0.878 | [+0.007, +0.085] **better** | 0.693 | 0.769 | [+0.017, +0.140] **better** |
| `sheetB` | 227 | 47 | 0.782 | 0.817 | [-0.029, +0.103]  | 0.536 | 0.598 | [-0.042, +0.171]  |
| `sheetB_hi_fancy` | 32 | 1 | 1.000 | 1.000 | [+0.000, +0.000]  | 1.000 | 1.000 | [+0.000, +0.000]  |
| `sheetC_fancy` | 114 | 50 | 0.599 | 0.713 | [-0.015, +0.233]  | 0.527 | 0.640 | [-0.016, +0.240]  |
| `v1_sitting` | 393 | 71 | 0.945 | 0.933 | [-0.046, +0.025]  | 0.812 | 0.758 | [-0.130, +0.019]  |
| `v1_unseen_locations` | 197 | 89 | 0.648 | 0.774 | [+0.040, +0.219] **better** | 0.580 | 0.726 | [+0.053, +0.243] **better** |

### clause (a) failures

- `mode:curv_linear` **auc_ge2**: Δ median -0.065, 95% CI [-0.152, -0.011]
- `mode:direct_trap_lines` **auc_ge3**: Δ median -0.127, 95% CI [-0.259, -0.017]
- `mode:direct_trap_ring` **auc_ge2**: Δ median -0.199, 95% CI [-0.339, -0.089]
- `mode:direct_trap_screen` **auc_ge2**: Δ median -0.048, 95% CI [-0.116, -0.004]
- `mode:gaussian_int` **auc_ge2**: Δ median -0.082, 95% CI [-0.188, -0.011]
- `mode:smooth_mean_angle` **auc_ge2**: Δ median -0.064, 95% CI [-0.153, -0.003]
- `mode:stripe` **auc_ge2**: Δ median -0.105, 95% CI [-0.210, -0.032]
- `pooled` **ap_ge2**: Δ median -0.016, 95% CI [-0.028, -0.005]
- `pooled` **auc_ge2**: Δ median -0.042, 95% CI [-0.060, -0.024]

### clause (b) improvements

- `busy_fp` **ap_ge2**: Δ median +0.022, 95% CI [+0.004, +0.048]
- `busy_fp` **ap_ge3**: Δ median +0.123, 95% CI [+0.010, +0.237]
- `busy_fp` **auc_ge2**: Δ median +0.298, 95% CI [+0.083, +0.508]
- `busy_fp` **auc_ge3**: Δ median +0.126, 95% CI [+0.007, +0.243]

clause (a) is a conjunction over 38 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


## v3 five-seed band (staged = the max of these, on this same slice)

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.862 ± 0.013 | 0.873 0.865 0.837 0.865 0.869 |
| AP>=3 | 0.651 ± 0.027 | 0.689 0.642 0.611 0.642 0.670 |
| AUC>=2 | 0.917 ± 0.008 | 0.912 0.917 0.905 0.926 0.927 |
| AP>=2 | 0.956 ± 0.005 | 0.956 0.953 0.950 0.964 0.959 |

## Volume-matched (a fixed threshold is a point on ONE head's scale)

| volume | v1 precision≥3 | v3 precision≥3 | n |
|---|---:|---:|---:|
| mining_pool (v1 @ 0.25) | 0.534 | 0.556 | 322 |
| mining_release (v1 @ 0.5) | 0.636 | 0.729 | 129 |
| top 0.05 | 0.707 | 0.854 | 41 |
| top 0.10 | 0.699 | 0.783 | 83 |
| top 0.20 | 0.636 | 0.679 | 165 |
