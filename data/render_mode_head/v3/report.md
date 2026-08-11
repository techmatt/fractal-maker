# mining v1 vs v3 — winner-rule verdict (STAGED, nothing adopted)

Generated 2026-08-10T18:31:36 · `uv run python tools/mining/mining_v3_reads.py`

**WINNER: v1** (pooled-only reading: v1)  
clause (a) no-worse FAIL over 38 arm x metric cells · clause (b) motivating PASS

> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Eval slice: **827 rows** (973 before near-dup dedup) over 136 locations, {'v1_sitting': 393, 'sheetB': 227, 'sheetC': 207}; tiers {'1': 261, '2': 352, '3': 214}.

Harness parity (v1 re-scored vs stamped): max |Δ| 4.30e-08 over 827 rows — OK.


## MOTIVATING arm

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `busy_fp` | 146 | 51 | 0.567 | 0.672 | [-0.014, +0.217]  | 0.424 | 0.569 | [+0.024, +0.260] **better** |

## NO-WORSE arms

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `pooled` | 827 | 214 | 0.839 | 0.870 | [+0.004, +0.061] **better** | 0.613 | 0.679 | [+0.008, +0.129] **better** |
| `rare_palette` | 207 | 96 | 0.652 | 0.752 | [+0.014, +0.190] **better** | 0.606 | 0.724 | [+0.030, +0.203] **better** |
| `mode:composite_c13_smooth_stripe` | 60 | 17 | 0.871 | 0.932 | [-0.019, +0.153]  | 0.731 | 0.812 | [-0.115, +0.288]  |
| `mode:composite_c17_smooth_curvature` | 39 | 3 | 0.889 | 0.972 | [-0.029, +0.289]  | 0.627 | 0.756 | [-0.500, +0.667]  |
| `mode:composite_c7_smooth_trap_circle` | 40 | 6 | 0.956 | 0.966 | [-0.049, +0.079]  | 0.733 | 0.813 | [-0.260, +0.425]  |
| `mode:curv_linear` | 43 | 7 | 0.877 | 0.940 | [+0.014, +0.135] **better** | 0.585 | 0.781 | [+0.033, +0.393] **better** |
| `mode:direct_trap_lines` | 69 | 10 | 0.895 | 0.746 | [-0.276, -0.032] **worse** | 0.565 | 0.261 | [-0.550, -0.060] **worse** |
| `mode:direct_trap_multiply` | 74 | 15 | 0.776 | 0.806 | [-0.077, +0.136]  | 0.444 | 0.525 | [-0.123, +0.291]  |
| `mode:direct_trap_ring` | 49 | 8 | 0.890 | 0.845 | [-0.159, +0.056]  | 0.507 | 0.412 | [-0.358, +0.111]  |
| `mode:direct_trap_screen` | 56 | 8 | 0.940 | 0.828 | [-0.232, -0.002] **worse** | 0.708 | 0.335 | [-0.638, -0.042] **worse** |
| `mode:exp_smoothing` | 70 | 43 | 0.670 | 0.699 | [-0.073, +0.131]  | 0.710 | 0.752 | [-0.045, +0.120]  |
| `mode:gaussian_int` | 48 | 7 | 0.916 | 0.801 | [-0.282, +0.000]  | 0.725 | 0.520 | [-0.435, +0.040]  |
| `mode:smooth` | 16 | 10 | 0.883 | 0.917 | [-0.214, +0.300]  | 0.929 | 0.956 | [-0.111, +0.214]  |
| `mode:smooth_angle_min` | 40 | 10 | 0.730 | 0.793 | [-0.069, +0.227]  | 0.640 | 0.678 | [-0.154, +0.250]  |
| `mode:smooth_mean_angle` | 39 | 8 | 0.855 | 0.871 | [-0.179, +0.188]  | 0.636 | 0.740 | [-0.166, +0.383]  |
| `mode:stripe` | 59 | 17 | 0.833 | 0.818 | [-0.137, +0.103]  | 0.666 | 0.712 | [-0.137, +0.231]  |
| `mode:tia` | 91 | 45 | 0.744 | 0.783 | [-0.063, +0.141]  | 0.736 | 0.744 | [-0.099, +0.116]  |
| `mode:trap_circle` | 34 | 0 | — | — | n/a | — | — | n/a |

## Diagnostics (vote on nothing)

| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fancy_all` | 466 | 85 | 0.836 | 0.857 | [-0.023, +0.065]  | 0.516 | 0.584 | [-0.040, +0.168]  |
| `pure_all` | 361 | 129 | 0.833 | 0.859 | [-0.011, +0.065]  | 0.693 | 0.732 | [-0.024, +0.102]  |
| `sheetB` | 227 | 47 | 0.782 | 0.857 | [+0.016, +0.139] **better** | 0.536 | 0.650 | [-0.019, +0.251]  |
| `sheetB_hi_fancy` | 32 | 1 | 1.000 | 1.000 | [+0.000, +0.000]  | 1.000 | 1.000 | [+0.000, +0.000]  |
| `sheetC_fancy` | 114 | 50 | 0.599 | 0.682 | [-0.044, +0.205]  | 0.527 | 0.643 | [-0.015, +0.234]  |
| `v1_sitting` | 393 | 71 | 0.945 | 0.926 | [-0.053, +0.018]  | 0.812 | 0.707 | [-0.194, -0.021] **worse** |
| `v1_unseen_locations` | 197 | 89 | 0.648 | 0.746 | [+0.010, +0.191] **better** | 0.580 | 0.699 | [+0.023, +0.211] **better** |

### clause (a) failures

- `mode:curv_linear` **auc_ge2**: Δ median -0.057, 95% CI [-0.139, -0.009]
- `mode:direct_trap_lines` **auc_ge2**: Δ median -0.051, 95% CI [-0.129, -0.001]
- `mode:direct_trap_lines` **auc_ge3**: Δ median -0.146, 95% CI [-0.276, -0.032]
- `mode:direct_trap_ring` **auc_ge2**: Δ median -0.130, 95% CI [-0.253, -0.042]
- `mode:direct_trap_screen` **auc_ge2**: Δ median -0.036, 95% CI [-0.090, -0.004]
- `mode:direct_trap_screen` **auc_ge3**: Δ median -0.110, 95% CI [-0.232, -0.002]
- `pooled` **auc_ge2**: Δ median -0.019, 95% CI [-0.034, -0.005]

### clause (b) improvements

- `busy_fp` **ap_ge2**: Δ median +0.020, 95% CI [+0.000, +0.046]
- `busy_fp` **ap_ge3**: Δ median +0.142, 95% CI [+0.024, +0.260]
- `busy_fp` **auc_ge2**: Δ median +0.282, 95% CI [+0.010, +0.502]

clause (a) is a conjunction over 38 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


## v3 five-seed band (staged = the max of these, on this same slice)

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.861 ± 0.005 | 0.870 0.862 0.862 0.859 0.854 |
| AP>=3 | 0.654 ± 0.026 | 0.679 0.669 0.617 0.675 0.629 |
| AUC>=2 | 0.918 ± 0.014 | 0.934 0.927 0.907 0.897 0.927 |
| AP>=2 | 0.956 ± 0.007 | 0.966 0.957 0.956 0.944 0.958 |

## Volume-matched (a fixed threshold is a point on ONE head's scale)

| volume | v1 precision≥3 | v3 precision≥3 | n |
|---|---:|---:|---:|
| mining_pool (v1 @ 0.25) | 0.534 | 0.525 | 322 |
| mining_release (v1 @ 0.5) | 0.636 | 0.760 | 129 |
| top 0.05 | 0.707 | 0.829 | 41 |
| top 0.10 | 0.699 | 0.747 | 83 |
| top 0.20 | 0.636 | 0.679 | 165 |
