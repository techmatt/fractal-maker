# wallpaper v3 vs v4b — winner-rule verdict (STAGED, nothing adopted)

Generated 2026-08-10T20:31:59 · `uv run python tools/wallpaper/wallpaper_v4b_reads.py`

**WINNER: v3** (pooled-only reading: v3)  
clause (a) no-worse FAIL over 18 arm x metric cells · clause (b) motivating FAIL

> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Eval slice: **1337 rows** over 527 locations, {'humanq3': 287, 'dramatic': 399, 'fresh_sheet': 296, 'colorize_path': 61, 'correction_v2': 294}; tiers {'1': 284, '2': 495, '3': 369, '4': 189}.


## MOTIVATING arm

| arm | n | ≥3 | v3 AUC≥3 | v4b AUC≥3 | Δ 95% CI | v3 AP≥3 | v4b AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `sheet_a_minibrot_maneuver` | 89 | 66 | 0.965 | 0.823 | [-0.243, -0.058] **worse** | 0.981 | 0.925 | [-0.113, -0.015] **worse** |

## NO-WORSE arms

| arm | n | ≥3 | v3 AUC≥3 | v4b AUC≥3 | Δ 95% CI | v3 AP≥3 | v4b AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fresh_colorize_path` | 61 | 29 | 0.934 | 0.885 | [-0.120, +0.011]  | 0.939 | 0.855 | [-0.198, +0.000]  |
| `fresh_pool_draw` | 296 | 45 | 0.853 | 0.874 | [-0.024, +0.069]  | 0.580 | 0.598 | [-0.090, +0.127]  |
| `overall` | 1337 | 558 | 0.845 | 0.831 | [-0.032, +0.002]  | 0.797 | 0.772 | [-0.053, +0.003]  |

## Diagnostics (vote on nothing)

| arm | n | ≥3 | v3 AUC≥3 | v4b AUC≥3 | Δ 95% CI | v3 AP≥3 | v4b AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fresh_era` | 357 | 74 | 0.886 | 0.893 | [-0.027, +0.039]  | 0.741 | 0.712 | [-0.107, +0.041]  |
| `fresh_sheet` | 296 | 45 | 0.853 | 0.874 | [-0.024, +0.069]  | 0.580 | 0.598 | [-0.090, +0.127]  |
| `old_dramatic` | 399 | 179 | 0.750 | 0.783 | [-0.004, +0.070]  | 0.713 | 0.738 | [-0.029, +0.079]  |
| `old_era` | 686 | 275 | 0.748 | 0.760 | [-0.016, +0.040]  | 0.669 | 0.679 | [-0.035, +0.055]  |
| `old_humanq3` | 287 | 96 | 0.746 | 0.714 | [-0.087, +0.022]  | 0.583 | 0.551 | [-0.109, +0.040]  |
| `sheet_a` | 294 | 209 | 0.950 | 0.851 | [-0.144, -0.056] **worse** | 0.975 | 0.930 | [-0.073, -0.019] **worse** |
| `sheet_a_maneuver_vein` | 95 | 79 | 0.957 | 0.776 | [-0.301, -0.078] **worse** | 0.987 | 0.945 | [-0.087, -0.012] **worse** |

### clause (a) failures

- `fresh_pool_draw` **ap_ge2**: Δ median -0.047, 95% CI [-0.083, -0.017]
- `fresh_pool_draw` **auc_ge2**: Δ median -0.036, 95% CI [-0.068, -0.007]
- `overall` **ap_ge2**: Δ median -0.012, 95% CI [-0.019, -0.005]
- `overall` **ap_ge4**: Δ median -0.075, 95% CI [-0.143, -0.010]
- `overall` **auc_ge2**: Δ median -0.032, 95% CI [-0.048, -0.015]
- `overall` **auc_ge4**: Δ median -0.032, 95% CI [-0.059, -0.007]

clause (a) is a conjunction over 18 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.


## Volume-matched (never a shared raw threshold)

The deployed gate passes **416** of 1337 eval rows on v3 (`p_ge3 > 0.9`); v4b is read at that same volume.

| volume | v3 precision≥3 | v4b precision≥3 | v3 %tier4 | v4b %tier4 | n |
|---|---:|---:|---:|---:|---:|
| deployed gate | 0.798 | 0.764 | 0.358 | 0.334 | 416 |
| top 0.05 | 0.910 | 0.896 | 0.716 | 0.552 | 67 |
| top 0.10 | 0.918 | 0.881 | 0.597 | 0.515 | 134 |
| top 0.20 | 0.850 | 0.824 | 0.476 | 0.397 | 267 |

## v4b five-seed band (staged = the max of these, on this same slice)

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.821 ± 0.006 | 0.831 0.813 0.824 0.817 0.819 |
| AP>=3 | 0.763 ± 0.008 | 0.772 0.750 0.769 0.763 0.763 |
| AUC>=2 | 0.877 ± 0.012 | 0.877 0.876 0.897 0.876 0.860 |
| AP>=2 | 0.961 ± 0.004 | 0.960 0.960 0.967 0.962 0.954 |
| AUC>=4 | 0.829 ± 0.011 | 0.816 0.828 0.839 0.843 0.819 |
| AP>=4 | 0.470 ± 0.023 | 0.454 0.449 0.513 0.468 0.466 |
