# Feature bake-off — is "interior mass" the real axis, not degree?

*2026-07-27. Batch `2026-07-26_minibrot_roster_v2`, the same 487 labels. No new labels, no
new renders beyond re-deriving the 487 crops' own escape-time fields.*

Reproduce with:

```bash
uv run python -m tools.studies.interior_bakeoff features   # ~2 min (the only rendering)
uv run python -m tools.studies.interior_bakeoff board      # Part B, ~10 s
uv run python -m tools.studies.interior_bakeoff audit      # Part C, ~7 min (background it)
```

Durable feature table: `data/minibrot_roster/batch_v1/interior_features.jsonl` (487 rows ×
13 crop features + the deployed screen's own 15 + covariates). Tests:
`tools/studies/test_interior_bakeoff.py`.

**Nothing was changed.** No cutoff, no screen, no OOD mask, no draw, no production feature.
Every deployed module (`q4_stage1_linear_fit`, `q4_harvest_tight`, `q4_multibrot_transfer`)
was imported read-only.

---

## Headline

**The hypothesis is refuted in its stated form, and a narrower version of it survives.**

1. **Interior *mass* is not the axis — it is the proxy.** The prompt's claim was that degree
   correlates with label without being causal, because higher degree puts more bodies in a
   window. The conditioning runs the other way. Interior mass's entire pooled correlation
   is degree: `int_frac` ρ = +0.328 raw → **+0.046** given degree (train); +0.454 → +0.151
   (eval). Degree barely moves the other way: ρ = +0.498 raw → **+0.399** given `int_frac`
   (train); +0.753 → +0.683 (eval). Every interior-mass feature behaves the same way. Inside
   a fixed degree, interior mass goes flat or slightly **negative** (deg-5 train: `int_frac`
   ρ = −0.121, AUC 0.421).
2. **Two features do survive conditioning, and they are the *shape* half of the read, not
   the *mass* half**: `int_perim_area` (interior boundary length per unit interior area) and
   `coh_scale_drop` (how much local orientation coherence falls when the analysis window
   grows — a scroll/turning measure). `coh_scale_drop` is a textbook Simpson reversal: ρ =
   −0.183 pooled, but **+0.139 / +0.356 / +0.320** inside degrees 3/4/5 (train AUC 0.70 /
   0.76 / 0.73). It is the strongest within-degree signal on the board, G included.
3. **The eye was reading something real, but it was reading degree.** `hi_g_lo` tiles carry
   an interior body in 33% of frames (median 0 components); `sub_hi` tiles in **81%**
   (median 2). That contrast is exactly as described — and their median degrees are 3 and 5.
4. **Part C: yes, an interior guard exists, and it is large.** The OOD mask drops any window
   whose frame is ≥10% in-set, outright, unscored; and `interior_worst` carries **−1.278**,
   the second-largest weight in G. Over a 24-atom sweep of every position the screen looks
   at, the interior ceiling is the **sole** cause of masking for **20.2%** of all
   featurizable positions (34.0% of everything masked); removing it would enlarge the
   scoreable pool by **49.8%**. **But the labels do not show it costing quality** — every
   drawn window above the ceiling scored 1, and the corpus contains almost no labeled
   examples in the band it cuts, which is itself the circularity: the screen that built the
   corpus ensured the band was never labeled.

---

## Part A — the candidate features

Each of the 487 crops had its escape-time field re-derived at its **exact label geometry**
(1280×720, ss1, f64 source, the crop's own maxiter) — the same field the labeled JPG was
shaded from. NaN marks in-set. Features are pure functions of that field, so they are
palette-invariant exactly like the screen's own.

| feature | what it is |
|---|---|
| `int_frac` | in-set pixel fraction |
| `int_largest_frac` | largest interior component, as a frame fraction |
| `int_n_comp_a4/a3/a2` | interior components with area ≥ 1e-4 / 1e-3 / 1e-2 of frame (the "counts at 2–3 scales" — nested motifs) |
| `int_perim_area` | total boundary length ÷ total area of components ≥1e-4. **The dendrite-vs-body discriminator** (high = filament, low = blob) |
| `int_compactness` | area-weighted isoperimetric ratio 4πA/P² per component (1 = disc, →0 = filament) |
| `int_max_inradius`, `int_mean_inradius` | Euclidean distance transform inside the mask, in frame heights — "how thick is the fattest body" |
| `coh_s3`, `coh_s8` | gradient-energy-weighted mean structure-tensor coherence at σ = 3 and 8 px |
| `coh_scale_drop` | `coh_s3 − coh_s8`. A straight filament stays coherent at every scale; a **scroll turns**, so its coherence falls as the window grows |
| `edge_energy` | mean ‖∇field‖ — the reference "ink" line |

Undefined-for-lack-of-interior is emitted as **NaN, not 0** (a crop with no in-set pixels has
no perimeter-to-area ratio, and coding it 0 would fake a maximally-blobby reading). Hence
`int_perim_area` / `int_compactness` have n = 223 train / 63 eval, not 377 / 110.

**G and its components.** The deployed screen's own 15 T2 features were recomputed on the
**exact** drawn window out of the cached parent atom field, and the resulting G reproduces
the stored deployed G to **max |ΔG| = 0.0000 over all 462 scored rows** — so every statement
below about G's components is about the window that was actually accepted or rejected.
Crop-resolution `int_frac` and screen-resolution `g_interior` agree at ρ = **+0.984**, so
nothing here is a re-derivation artifact.

---

## Part B — the whole board

Atom-level split, inherited from the roster: **train n = 377** (56 with label ≥3, 120 atoms),
**eval n = 110** (19 with label ≥3, 25 atoms). Select on train, confirm on eval. Every
feature computed is listed, including the ones that did nothing.

| feature | ρ train | AUC train | n tr | ρ eval | AUC eval | n ev |
|---|---|---|---|---|---|---|
| `int_frac` | +0.328 | 0.544 | 377 | +0.454 | 0.659 | 110 |
| `int_largest_frac` | +0.311 | 0.524 | 377 | +0.470 | 0.671 | 110 |
| `int_n_comp_a4` | +0.286 | 0.546 | 377 | +0.402 | 0.675 | 110 |
| `int_n_comp_a3` | +0.103 | 0.459 | 377 | +0.082 | 0.473 | 110 |
| `int_n_comp_a2` | −0.065 | 0.492 | 377 | — (no variance) | 0.500 | 110 |
| `int_perim_area` | +0.185 | **0.652** | 223 | +0.284 | **0.683** | 63 |
| `int_compactness` | −0.024 | 0.549 | 223 | −0.067 | 0.486 | 63 |
| `int_max_inradius` | +0.297 | 0.521 | 377 | +0.466 | 0.665 | 110 |
| `int_mean_inradius` | −0.057 | 0.383 | 306 | +0.086 | 0.569 | 87 |
| `coh_s3` | −0.371 | 0.493 | 377 | −0.346 | 0.461 | 110 |
| `coh_s8` | −0.420 | 0.438 | 377 | −0.333 | 0.448 | 110 |
| `coh_scale_drop` | −0.142 | 0.585 | 377 | −0.323 | 0.479 | 110 |
| `edge_energy` | +0.372 | 0.652 | 377 | +0.171 | 0.478 | 110 |
| `s_g_speckle` | +0.471 | **0.711** | 377 | +0.317 | 0.607 | 110 |
| `s_speckle_worst` | +0.362 | 0.687 | 377 | +0.333 | 0.570 | 110 |
| `s_g_mid` | +0.360 | 0.677 | 377 | +0.149 | 0.462 | 110 |
| `s_g_occ` | +0.360 | 0.672 | 377 | +0.159 | 0.471 | 110 |
| `s_lapvar_mean` | +0.368 | 0.656 | 377 | +0.190 | 0.495 | 110 |
| `s_speckle_spread` | +0.325 | 0.652 | 377 | +0.357 | 0.617 | 110 |
| `s_lapvar_worst` | +0.301 | 0.606 | 377 | +0.250 | 0.536 | 110 |
| `s_flat_spread` | +0.246 | 0.588 | 377 | +0.306 | 0.572 | 110 |
| `s_lapvar_spread` | +0.289 | 0.587 | 377 | +0.254 | 0.519 | 110 |
| `s_flat_edge_minus_center` | +0.251 | 0.560 | 377 | +0.372 | 0.622 | 110 |
| `s_detail_spread` | +0.253 | 0.548 | 377 | +0.384 | 0.636 | 110 |
| `s_g_interior` | +0.298 | 0.530 | 377 | +0.400 | 0.634 | 110 |
| `s_detail_worst` | +0.149 | 0.521 | 377 | −0.156 | 0.463 | 110 |
| `s_interior_spread` | +0.285 | 0.513 | 377 | +0.403 | 0.639 | 110 |
| `s_interior_worst` | +0.286 | 0.513 | 377 | +0.390 | 0.627 | 110 |
| `s_g_high` | +0.119 | 0.505 | 377 | +0.082 | 0.440 | 110 |
| `s_flat_worst` | −0.166 | 0.461 | 377 | +0.090 | 0.509 | 110 |
| `s_g_flat` | −0.360 | 0.328 | 377 | −0.159 | 0.529 | 110 |
| **`G` (reference line)** | +0.185 | 0.641 | 352 | +0.038 | 0.489 | 110 |
| **`degree`** | **+0.498** | **0.701** | 377 | **+0.753** | **0.880** | 110 |
| `period` | +0.125 | 0.619 | 377 | −0.208 | 0.359 | 110 |
| `log10|A|` | +0.135 | 0.660 | 377 | +0.017 | 0.589 | 110 |
| `maxiter` | +0.133 | 0.657 | 377 | +0.027 | 0.604 | 110 |

**Nothing beats degree**, on either split. Atom-bootstrap 90% CIs (resampling *atoms*, since
up to 3 windows share one): degree AUC eval 0.880 [0.780, 0.957]; the best interior feature
`int_perim_area` 0.683 [0.528, 0.824] on 63 rows.

Note the ρ/AUC divergence for the mass features (ρ +0.33 but AUC 0.54): interior mass
separates label 1 from label 2 but does almost nothing at the 3-boundary, which is the
boundary that matters.

### 2. The conditioning pair — the point of the exercise

Rank partial correlations (rank all three, regress out the control's ranks, correlate
residuals).

**Train** (raw ρ(degree, label) = +0.498):

| feature F | ρ(F, y) | ρ(F, y \| degree) | ρ(degree, y \| F) | n |
|---|---|---|---|---|
| `int_frac` | +0.328 | **+0.046** | +0.399 | 377 |
| `int_largest_frac` | +0.311 | +0.060 | +0.413 | 377 |
| `int_n_comp_a4` | +0.286 | −0.012 | +0.425 | 377 |
| `int_max_inradius` | +0.297 | +0.048 | +0.421 | 377 |
| `int_perim_area` | +0.185 | **+0.177** | +0.377 | 223 |
| `int_compactness` | −0.024 | +0.090 | +0.389 | 223 |
| `int_mean_inradius` | −0.057 | −0.165 | +0.389 | 306 |
| `int_n_comp_a3` | +0.103 | −0.087 | +0.495 | 377 |
| `int_n_comp_a2` | −0.065 | −0.112 | +0.504 | 377 |
| `coh_s3` | −0.371 | −0.135 | +0.379 | 377 |
| `coh_s8` | −0.420 | −0.200 | +0.351 | 377 |
| `coh_scale_drop` | −0.142 | +0.110 | +0.492 | 377 |
| `edge_energy` | +0.372 | +0.295 | +0.451 | 377 |
| `G` | +0.185 | +0.274 | +0.544 | 352 |

**Eval** (raw ρ(degree, label) = +0.753): `int_frac` +0.454 → **+0.151** given degree, while
ρ(degree, y | int_frac) = **+0.683**. `int_perim_area` +0.284 → +0.204; degree | it = +0.594.
`int_n_comp_a4` +0.402 → +0.210; degree | it = +0.712.

**The answer to both halves of the question:**
- *Does degree retain signal after conditioning on the best interior feature?* **Yes,
  overwhelmingly.** +0.498 → +0.351…+0.504 (train); +0.753 → +0.594…+0.753 (eval). No
  interior feature removes more than ~30% of degree's rank correlation, and most remove
  under 15%.
- *Does that feature retain signal after conditioning on degree?* **Interior mass: no**
  (+0.328 → +0.046 train). **Boundary shape and scroll: partly yes** — `int_perim_area`
  keeps +0.177 (train) / +0.204 (eval), and `coh_scale_drop` *gains* (see below).

### 2b. Within each degree — where the reversal lives

Pooled ρ vs ρ inside each degree, train (degree 2 has zero label ≥3, so its AUC is
undefined by construction):

| feature | pooled ρ | d3 ρ / AUC | d4 ρ / AUC | d5 ρ / AUC |
|---|---|---|---|---|
| `int_frac` | +0.355 | −0.067 / 0.301 | +0.040 / 0.423 | −0.121 / 0.421 |
| `int_largest_frac` | +0.344 | −0.066 / 0.307 | +0.081 / 0.460 | −0.132 / 0.394 |
| `int_n_comp_a4` | +0.309 | −0.188 / 0.272 | +0.040 / 0.446 | −0.049 / 0.456 |
| `int_max_inradius` | +0.333 | −0.077 / 0.304 | +0.037 / 0.437 | −0.151 / 0.389 |
| `int_perim_area` | +0.211 | +0.120 / **0.718** | +0.180 / **0.666** | +0.231 / **0.639** |
| `int_compactness` | −0.015 | +0.068 / 0.615 | +0.131 / 0.626 | +0.144 / 0.624 |
| `coh_s3` | −0.365 | −0.085 / 0.615 | +0.100 / 0.714 | +0.083 / 0.662 |
| `coh_s8` | −0.400 | −0.214 / 0.535 | −0.022 / 0.619 | +0.065 / 0.643 |
| **`coh_scale_drop`** | **−0.183** | **+0.139 / 0.701** | **+0.356 / 0.755** | **+0.320 / 0.727** |
| `edge_energy` | +0.329 | +0.320 / 0.567 | +0.374 / 0.672 | +0.317 / 0.641 |
| `G` | +0.152 | +0.259 / 0.588 | +0.361 / 0.775 | +0.346 / 0.679 |

Two things fall out. **Interior mass inverts**: positive pooled, ≈0 or negative in every
degree with positives. **`coh_scale_drop` reverses sign** — pooled it looks mildly harmful,
inside every degree it is the best feature measured. Higher-degree fields are intrinsically
more coherent (ρ(degree, `coh_s8`) = −0.566), which is what buries it in the pool.

`G` also survives within degree (AUC 0.59 / 0.78 / 0.68) — consistent with the earlier
readout's finding that degree was acting as a *suppressor* for G.

Eval, degree 5 (n = 24, 14 positives, ~6 atoms — small, treat as directional only): all
features go positive, `coh_scale_drop` ρ +0.635 / AUC 0.871, `edge_energy` +0.781 / 0.957,
`int_perim_area` +0.436 / 0.786, `G` −0.037 / 0.479.

### 3. hi_g_lo vs sub_hi — the two populations the screen gets backwards

`hi_g_lo` = the sheet's top 24 by G among the 189 accepts labeled ≤2. `sub_hi` = the 27
sub-cutoff/OOD crops labeled ≥3. AUC = P(sub_hi ranks above hi_g_lo); 0.5 = no separation.
`G` separates them at 0.000 **by construction** (the sets are defined by G), so it is not
evidence.

| feature | AUC (sheet 24 v 27) | AUC (all 189 v 27) |
|---|---|---|
| `degree` | 0.884 | 0.782 |
| `s_detail_spread` | 0.873 | 0.786 |
| `s_g_flat` | 0.898 | 0.546 |
| `s_g_mid` / `s_g_occ` | 0.099 / 0.102 | 0.448 / 0.454 |
| `int_largest_frac` | 0.773 | 0.645 |
| `int_max_inradius` | 0.764 | 0.646 |
| `int_n_comp_a4` | 0.751 | 0.644 |
| `int_frac` | 0.743 | 0.656 |
| `s_g_interior` | 0.731 | 0.643 |
| `edge_energy` | 0.153 | 0.522 |
| `int_compactness` | 0.153 | 0.451 |
| `coh_scale_drop` | 0.577 | 0.539 |
| `int_perim_area` | 0.466 | 0.524 |

The interior-mass block does separate the two sheets (0.74–0.77), and the descriptive
contrast is exactly as described: **33% of `hi_g_lo` tiles contain an interior body ≥1e-4
(median 0 components) vs 81% of `sub_hi` (median 2)**. But `degree` separates them just as
well or better (0.884 / 0.782) and their median degrees are 3 vs 5 — so the sheets are also
a degree contrast, and given §2 the interior separation is what degree looks like. Note the
sheet-set numbers collapse toward 0.5 on the full 189-vs-27 comparison; the 24-tile
selection is the extreme tail of G, not a representative accept sample.

### 4. Near-duplicate windows — the batch is clean

- Same-atom pairs at IoU ≥ 0.25: **0**. At IoU > 0: **6 pairs**, max IoU 0.182.
- Effective independent windows at IoU ≥ 0.50: **487 of 487** (largest cluster 1).
- **The two class-4s are one atom but not one window.** `mb0095` box centre (0.067, 0.790)
  vs `mb0159` (0.299, 0.064) — **IoU = 0.000**, opposite corners of the parent field. They
  are the same *motif* repeated by the d5 symmetry, not the same picture. The prompt's
  "heavily overlapping windows" premise is wrong at the window level; the "one independent
  example" conclusion still holds at the **atom** level, which is why every AUC CI here
  bootstraps atoms rather than crops. Class 4 is treated as anecdote throughout.

### 5. Confound checks

- **maxiter ↔ interior.** ρ(maxiter, `int_frac`) = −0.570 (deeper crops iterate longer, so
  less unresolved "false" interior). It does not explain the interior signal: partial
  ρ(`int_frac`, label | maxiter) = +0.499, larger than the raw +0.355. ρ(maxiter, label) =
  +0.094.
- **Resolution.** ρ(crop `int_frac` @1280×720, screen `g_interior` @~195×110) = +0.984.
- **Clustering.** 145 atoms, up to 3 windows each; all CIs resample atoms. Nothing here is
  tested for significance and most cells are small.

---

## Part C — is the bias self-inflicted? **Partly yes, and it is large — but the labels do
not show it costing quality**

### The guard exists, in two places

1. **The OOD mask** (`q4_stage1_linear_fit._v2_drop`, deployed): a window is dropped
   outright — **never scored at all** — if `g_interior ≥ 0.10`, or `g_flat ≥ 0.88`, or
   `g_speckle ≥ 0.30`. A frame that is ≥10% in-set is unscoreable **by construction**.
2. **G itself**: `interior_worst` = **−1.278**, the **second-largest of 11 nonzero weights**
   (only `detail_spread` at −1.758 is larger). The single worst cell's in-set fraction
   pushes G down. `g_interior` itself has weight 0, and `interior_spread` 0 — so the penalty
   is specifically on *concentrated* interior, i.e. a body sitting in one cell.

Consequence over the drawn windows: ρ(`g_interior`, G) = **−0.163** and
ρ(`interior_worst`, G) = **−0.167** among the 462 windows the mask *did* let through, and
**no drawn accept exceeds `g_interior` = 0.037** — barely a third of the mask's own 0.10
ceiling. The effective interior ceiling on the accept arm is set by G's weight, not by the
mask.

### How much of the split it accounts for

A 24-atom seeded sweep (6 per degree) over **every position the deployed screen looks at**,
at the deployed scales and stride — 296,233 featurizable positions:

| clause | n | % featurizable | % of masked |
|---|---|---|---|
| **interior ONLY (sole cause)** | **59,858** | **20.2%** | **34.0%** |
| flat only | 93,214 | 31.5% | 52.9% |
| speckle only | 4,223 | 1.4% | 2.4% |
| two or more | 18,772 | 6.3% | 10.7% |
| interior (any, incl. shared) | 78,630 | 26.5% | 44.7% |

59.4% of positions are masked; **34.0% of that is the interior clause acting alone**.
Dropping the interior clause would enlarge the scoreable pool by **49.8%**. 26.5% of all
positions sit at or above the ceiling.

**So the guard is real, it is the second-biggest thing the screen does, and it selects
against interior content globally — not just against the all-black failure it was written
for.** The prompt's suspicion that a guard overshot is confirmed as a description of the
mechanism.

### But the labels do not convict it

- Every drawn window above the ceiling scored **1**: `g_interior` ∈ [0.10, 1.01) has n = 6,
  mean label **1.00**, zero label ≥3. The `[0.05, 0.10)` bin (n = 4) also has zero label ≥3.
- Of the 25 OOD-masked crops that *were* drawn and labeled: 18 tripped `flat`, 5 `interior`,
  1 `interior+speckle`, 1 `speckle`. Every interior-tripped one scored 1. **The single
  label-3 masked crop tripped `speckle`, not interior.**
- Within degree, interior mass carries no positive residual signal at all (§2b).

### The circularity, which is the real limitation

The labeled corpus cannot adjudicate the ceiling, because the screen that built the corpus
made sure the band was never populated:

| crop `int_frac` band | n labeled | mean label | count label ≥3 |
|---|---|---|---|
| [0.00, 0.01) | 440 | 1.82 | 68 |
| [0.01, 0.05) | 38 | 2.08 | 7 |
| [0.05, 0.10) | 3 | 2.00 | 0 |
| **[0.10, 0.25)** | **2** | 1.00 | 0 |
| **[0.25, 0.50)** | **2** | 1.00 | 0 |
| [0.50, 1.01) | 2 | 1.00 | 0 |

**Four labeled crops** exist in the entire [0.10, 0.50] band the guard cuts, and they are
near-degenerate framings, not "distinct black body with scroll structure". 59,858 swept
positions were removed by the interior clause alone; roughly six of them ever reached a
label. The correct statement is: *the interior guard removes a fifth of the search space,
and the existing labels give no evidence that what it removes is good — but they also give
almost no evidence either way, by the guard's own doing.*

---

## What this says (report only — nothing was changed)

1. **Interior mass is not the axis.** It is degree's shadow: ρ +0.33 pooled → +0.05 given
   degree, and negative inside deg-3 and deg-5. Replacing G with an interior-mass term would
   be re-deriving degree the expensive way.
2. **Degree is not a proxy for interior content.** It survives conditioning on every
   interior feature on both splits. The prompt's causal claim does not hold in this corpus.
3. **The surviving candidates are shape, not mass** — `int_perim_area` (train AUC 0.652 /
   eval 0.683, defined on the 59% of crops that have a body) and especially
   `coh_scale_drop`, which is sign-reversed pooled and the best within-degree feature
   measured (train AUC 0.70 / 0.76 / 0.73 across d3/d4/d5). If anything augments G, these
   are the two with evidence, and both need conditioning on degree to be visible at all.
4. **G is less dead than it looked**, once degree is held: within-degree train AUC 0.59 /
   0.78 / 0.68. The prior readout's "flat above the cutoff" (within-accept AUC 0.511) is a
   statement about the post-cutoff tail, not about G's ordering across the full range.
5. **The interior guard is a genuine, quantified, self-inflicted restriction** — 20.2% of
   all swept positions removed by the interior clause alone, +49.8% pool if dropped — **but
   this corpus cannot say whether that costs quality**, because it contains four labeled
   crops in the band. That is the one measurement this exercise could not make, and it is
   the one that would settle it.
6. **The batch has no near-duplicate windows** (zero pairs at IoU ≥ 0.25). The class-4 pair
   is one atom, two disjoint windows — anecdote at the atom level, as treated.

**Decision deferred to Matt with the numbers in hand:** what, if anything, replaces or
augments G. No cutoff, screen, mask, draw, or feature was touched.
