# SHEET D re-verdict — wallpaper v3 vs v4b on BLIND minibrot labels

Generated 2026-08-11T08:35:19 · `uv run python tools/wallpaper/sheet_d_reverdict.py`

**Clause (a) no-worse PASS · clause (b) motivating FAIL → WINNER v3**

> THIS SLICE ONLY. The (28) winner rule ran over the whole six-batch eval union; this is the motivating arm re-drawn without the anchoring, and it decides that arm, not the flip.  
> NOT decided here. BUILD != FLIP: adoption is a separate prompt after Matt reads this verdict.

Slice: **197 labeled rows** of 197 in `2026-08-11_wallpaper_blind_minibrot_v1`; tiers {'1': 0, '2': 6, '3': 95, '4': 96}; veins {'maneuver': 186, 'q4_harvest': 11}; partitions {'mandelbrot': 37, 'multibrot3': 160}. Blind, eval-only.


## THE MOTIVATING ARM, re-drawn blind

| arm | n | ≥3 | v3 AUC≥3 | v4b AUC≥3 | Δ 95% CI | v3 AP≥3 | v4b AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `blind_minibrot` | 197 | 191 | 0.741 | 0.622 | [-0.341, +0.034]  | 0.984 | 0.978 | [-0.024, +0.007]  |

## Diagnostics (vote on nothing)

| arm | n | ≥3 | v3 AUC≥3 | v4b AUC≥3 | Δ 95% CI | v3 AP≥3 | v4b AP≥3 | Δ 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `partition:mandelbrot` | 37 | 35 | 0.929 | 0.814 | [-0.314, +0.000]  | 0.996 | 0.987 | [-0.034, +0.000]  |
| `partition:multibrot3` | 160 | 156 | 0.671 | 0.513 | [-0.506, +0.038]  | 0.982 | 0.976 | [-0.026, +0.008]  |
| `vein:maneuver` | 186 | 181 | 0.740 | 0.601 | [-0.389, +0.027]  | 0.984 | 0.979 | [-0.024, +0.007]  |

### Every metric on the arm

| metric | v3 | v4b | Δ 95% CI |
|---|---:|---:|---|
| AUC>=3 | 0.741 | 0.622 | [-0.341, +0.034] |
| AP>=3 | 0.984 | 0.978 | [-0.024, +0.007] |
| AUC>=2 | — | — | n/a |
| AP>=2 | — | — | n/a |
| AUC>=4 | 0.510 | 0.572 | [-0.018, +0.141] |
| AP>=4 | 0.480 | 0.526 | [-0.016, +0.116] |

## The anchoring price

v3's AUC>=3 on rows whose labels it SUGGESTED, against v3's AUC>=3 on fresh rows of the same population labeled blind. The gap is the anchoring price — how much of the 0.965 was agreement rather than quality (classifier_retrain_protocol.md §2b).

| slice | labels elicited | n | v3 AUC≥3 | v3 AP≥3 |
|---|---|---:|---:|---:|
| sheet A `sheet_a_minibrot_maneuver` | v3's tier PREFILLED, page sorted by v3 | 89 | 0.965 | 0.981 |
| sheet D `blind_minibrot` | BLIND, shuffled | 197 | 0.741 | 0.984 |

**Δ AUC≥3 (blind − anchored) = -0.224** — the blind slice is the LOWER number; the difference is what the prefilled suggestion bought v3 on a slice it was measured on


## Volume-matched (never a shared raw threshold)

The deployed gate passes **111** of 197 rows on v3 (`p_ge3 > 0.9`); v4b is read at that same volume.

| volume | v3 precision≥3 | v4b precision≥3 | v3 %tier4 | v4b %tier4 | n |
|---|---:|---:|---:|---:|---:|
| deployed gate | 0.991 | 0.982 | 0.523 | 0.541 | 111 |
| top 0.05 | 1.000 | 1.000 | 0.400 | 0.700 | 10 |
| top 0.10 | 0.950 | 1.000 | 0.450 | 0.750 | 20 |
| top 0.20 | 0.974 | 0.949 | 0.513 | 0.590 | 39 |

## v4b five-seed band on this slice

every v4b seed was selected on the (28) POOLED eval, not on this slice — so sheet D is the first held-out population any of them has seen and the staged max is NOT optimistic here, unlike in the (28) report.

| metric | mean ± sd | per seed |
|---|---|---|
| AUC>=3 | 0.697 ± 0.053 | 0.622 0.755 0.689 0.757 0.661 |
| AP>=3 | 0.983 ± 0.004 | 0.978 0.990 0.984 0.978 0.983 |
| AUC>=2 | undefined on this slice | — — — — — |
| AP>=2 | undefined on this slice | — — — — — |
| AUC>=4 | 0.567 ± 0.032 | 0.572 0.609 0.557 0.585 0.512 |
| AP>=4 | 0.540 ± 0.027 | 0.526 0.580 0.541 0.552 0.499 |

clause (a) is a conjunction over 4 arm x metric cells; at 95% per cell the chance one crosses by luck alone is material, so read `failures` before reading `pass`.

