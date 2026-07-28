# Interior-band batch v1 — build notes, and two follow-ups on the bake-off

*2026-07-27. Companion to `interior_feature_bakeoff.md`. Part A builds a label batch;
Part B answers two questions off data already on disk. **Nothing deployed was changed** —
no cutoff, screen, OOD mask, draw rule, or production feature. `q4_stage1_linear_fit`,
`q4_multibrot_transfer` and `q4_harvest_tight` are imported read-only throughout.*

Reproduce with:

```bash
# Part A (the batch)
uv run python tools/sourcing/build_interior_band_batch.py sweep    # ~75 min, 4 workers — background it
uv run python tools/sourcing/build_interior_band_batch.py draw     # ~5 s
uv run python tools/sourcing/build_interior_band_batch.py feat     # ~2 min
uv run python tools/sourcing/build_interior_band_batch.py render   # ~15 min, 4 workers
uv run python tools/sourcing/build_interior_band_batch.py report   # ~10 s

# Part B (no new data)
uv run python -m tools.studies.interior_bakeoff questions          # ~40 s
```

Tests: `tools/sourcing/test_build_interior_band_batch.py`,
`tools/studies/test_interior_bakeoff.py`.

---

## Part B — two follow-ups on the bake-off

### B1. `coh_scale_drop`'s per-degree AUCs 0.70 / 0.76 / 0.73 — **those are TRAIN**

They came from the bake-off's §2b table, whose per-degree columns are train (the caption
says so); the eval confirmation printed beneath it covered degree 5 alone. Both splits,
split out, with cell sizes and atom-bootstrap 90% CIs:

| feature | deg | split | n | pos | atoms | ρ | AUC | AUC 90% CI |
|---|---|---|---|---|---|---|---|---|
| `coh_scale_drop` | 2 | train | 103 | 0 | 30 | −0.272 | — | 0 positives |
| | 2 | eval | 25 | 0 | 7 | −0.082 | — | 0 positives |
| | 3 | **train** | 95 | 17 | 30 | +0.139 | **0.701** | [0.532, 0.833] |
| | 3 | **eval** | 33 | 1 | 6 | +0.466 | **0.281** | [0.065, 0.559] |
| | 4 | **train** | 98 | 17 | 30 | +0.356 | **0.755** | [0.624, 0.852] |
| | 4 | **eval** | 28 | 4 | 6 | +0.307 | **0.802** | [0.632, 0.871] |
| | 5 | **train** | 81 | 22 | 30 | +0.320 | **0.727** | [0.491, 0.844] |
| | 5 | **eval** | 24 | 14 | 6 | +0.635 | **0.871** | [0.753, 0.972] |

`int_perim_area` and `G` on the same footing:

| feature | deg | train AUC (n, pos) | eval AUC (n, pos) |
|---|---|---|---|
| `int_perim_area` | 3 | 0.718 (40, 1) | 0.955 (23, 1) |
| | 4 | 0.666 (76, 15) | 0.559 (21, 4) |
| | 5 | 0.639 (77, 22) | 0.786 (19, 14) |
| `G` | 3 | 0.588 (90, 17) | 0.500 (33, 1) |
| | 4 | 0.775 (93, 17) | 0.510 (28, 4) |
| | 5 | 0.679 (76, 21) | 0.479 (24, 14) |

**What this changes.**

1. **The headline number was train.** The finding's phrasing ("train AUC 0.70 / 0.76 /
   0.73") was accurate, but the number that survives selection is the eval column, and it
   is not the same story: **d4 0.802 and d5 0.871 hold and are the strongest cells on the
   board, d3 inverts to 0.281.**
2. **d3 eval has one positive.** An AUC over 1 positive vs 32 negatives is a statement
   about a single crop. Its CI is [0.065, 0.559] — it does not exclude 0.5, and neither
   does the train d3 cell's lower bound of 0.532 by much. Read d3 as unresolved, not as a
   reversal.
3. **The cells that count are d4 and d5**, and there `coh_scale_drop` is the only feature
   on the board whose train and eval agree, both above 0.75, with eval CIs excluding 0.5
   (d4 [0.632, 0.871], d5 [0.753, 0.972]). **`G` does the opposite** — train 0.775 / 0.679,
   eval 0.510 / 0.479, i.e. it collapses to chance on held-out atoms in exactly the cells
   where `coh_scale_drop` holds.
4. **Selection exposure, stated.** `coh_scale_drop` was picked on train out of **13 crop
   features × 3 degrees = 39 AUCs**; the max of a board that size is biased upward on
   train by construction. The full board is printed by the `questions` stage. That bias is
   *why* the eval column above matters, and the eval column is what survives — with 6 eval
   atoms per degree, which is the real limitation.

**Bottom line:** the 0.70/0.76/0.73 triple is train. The eval read is 0.28 / 0.80 / 0.87,
the d3 cell is one positive wide, and on d4/d5 the feature confirms while `G` does not.
That is a weaker claim than the pooled train triple looked, and a more interesting one.

### B2. Atom-bootstrapped CIs on degree's conditional rank correlation

Clustered 90% intervals (2000 reps, resampling **atoms**, since up to 3 windows share one):

| control F | split | n | atoms | ρ(deg, y) | ρ(deg, y \| F) | 90% CI |
|---|---|---|---|---|---|---|
| — (raw) | train | 377 | 120 | +0.498 | +0.498 | [+0.394, +0.584] |
| — (raw) | eval | 110 | 25 | +0.753 | +0.753 | [+0.609, +0.844] |
| `int_frac` | **train** | 377 | 120 | +0.498 | **+0.399** | **[+0.304, +0.478]** |
| `int_frac` | **eval** | 110 | 25 | +0.753 | **+0.683** | **[+0.541, +0.763]** |
| `int_perim_area` | train | 223 | 96 | +0.381 | +0.377 | [+0.220, +0.509] |
| `int_perim_area` | eval | 63 | 17 | +0.615 | +0.594 | [+0.383, +0.765] |
| `coh_scale_drop` | train | 377 | 120 | +0.498 | +0.492 | [+0.397, +0.571] |
| `coh_scale_drop` | eval | 110 | 25 | +0.753 | +0.743 | [+0.583, +0.822] |
| `G` | train | 352 | 120 | +0.514 | +0.544 | [+0.437, +0.628] |
| `G` | eval | 110 | 25 | +0.753 | +0.753 | [+0.608, +0.843] |

Reverse direction (does the control survive degree?):

| feature F | split | ρ(F, y) | ρ(F, y \| deg) | 90% CI |
|---|---|---|---|---|
| `int_frac` | train | +0.328 | +0.046 | [−0.060, +0.164] |
| `int_frac` | eval | +0.454 | +0.151 | [−0.039, +0.288] |
| `int_perim_area` | train | +0.185 | +0.177 | [+0.039, +0.309] |
| `int_perim_area` | eval | +0.284 | +0.204 | [−0.091, +0.468] |
| `coh_scale_drop` | train | −0.142 | +0.110 | [−0.006, +0.223] |
| `coh_scale_drop` | eval | −0.323 | +0.270 | [+0.071, +0.423] |

**What this changes.**

1. **Degree's conditional effect is comfortably above zero on both splits.** Train
   [+0.304, +0.478], eval [+0.541, +0.763]. Neither interval comes near 0. The direction of
   the finding is not a small-sample artifact.
2. **The train/eval gap is real, not noise.** The two intervals **do not overlap**
   (train upper +0.478 < eval lower +0.541). So the unusual direction is not explained
   away by eval's 25 atoms — eval genuinely has a stronger degree effect.
3. **But the gap is a composition difference, not a validation.** Eval's *raw* ρ is also
   higher (+0.753 vs +0.498), and the reason is visible in B1's cell counts: **14 of eval's
   19 positives sit at degree 5**, so on eval "is it degree 5?" is very nearly "is it
   good?". A split where one degree carries three quarters of the positives will report a
   large degree correlation whether or not degree is doing the work. Eval is confirming
   with 25 atoms, six of them at degree 5.
4. **Before degree becomes a draw axis** the honest statement is: degree survives
   conditioning on every interior feature, at ρ ≈ +0.40 [+0.30, +0.48] on the split with
   120 atoms, and the eval number is not independent corroboration of its *size*. Interior
   mass still does not survive the reverse conditioning on either split (train CI spans
   zero; eval CI spans zero) — that half of the bake-off is unchanged and now has intervals.
5. `coh_scale_drop` and `G` remove essentially none of degree's correlation (+0.492 /
   +0.544 train), which is consistent with the bake-off: they are measuring something else,
   which is exactly why they are the interesting candidates.

---

## Part A — the interior-band batch

*(built by `tools/sourcing/build_interior_band_batch.py`; batch
`data/label_corpus/batches/2026-07-27_interior_band_v1`, manifest
`data/minibrot_roster/interior_band_v1/`)*

### Why this batch exists

The bake-off's Part C established that the deployed OOD mask's interior clause
(`g_interior >= 0.10`) removes **20.2% of every position the screen sweeps** — 34.0% of
everything it masks — and that dropping the clause would enlarge the scoreable pool by
**49.8%**. It also established that the labeled corpus cannot adjudicate that, because the
screen that built the corpus made sure the band was never populated. Counting the 487 on
the screen-resolution quantity the mask actually cuts on:

| `g_interior` band | n labeled | mean label | n ≥ 3 |
|---|---|---|---|
| [0.00, 0.10) — below the ceiling | 481 | 1.84 | 75 |
| **[0.10, 0.20)** | **2** | 1.00 | 0 |
| **[0.20, 0.35)** | **1** | 1.00 | 0 |
| **[0.35, 0.50)** | **1** | 1.00 | 0 |
| [0.50, 1.00] | 2 | 1.00 | 0 |

Four crops in the whole [0.10, 0.50] band. This batch puts 60 there.

### The two confounds it is built to avoid

**1. Interior vs degree.** In the 487 the two moved together, which is why the bake-off
could only separate them by conditioning after the fact. Here they are crossed
explicitly — degree {2,3,4,5} × band {0.10–0.20, 0.20–0.35, 0.35–0.50}, 5 crops per cell.

**2. Interior vs framing method — the subtler one.** The 487 were G-maxima framed, and G
carries `interior_worst` = −1.278 (its second-largest weight), so G-maxima framing
*physically cannot* produce a high-interior window. Any framing rule usable here therefore
differs from the 487's, and a naive new-vs-old comparison would confound interior with the
framing method. So the batch carries its **own** low-interior control arm (`g_interior` <
0.10, same degrees, 5 per degree) drawn by the **identical sampler** — same swept grid,
same scale mix, same uniform-random draw, same per-atom cap, no additional predicate.
Within this batch, interior fraction is the only thing that varies between the arms.

### The sampler

Uniform-random over exactly the positions the deployed screen sweeps: `LF.FIELD_SCALES`
(0.06 / 0.09 / 0.14) × `DENSE_STRIDE_FRAC` = 0.12 stride × 16:9 windows, on the same cached
2176×1224 parent atom fields the 487 came off. The grid geometry is copied verbatim from
`MT._sweep_fates`, so the candidate universe **is** the screen's swept set; the sampler
differs from the screen only in what it selects on. Per atom, every swept position is
featurized once (~14.6k positions/atom, 160 atoms) and reservoir-sampled into
(band × scale) buckets, so each bucket holds a uniform random sample of that atom's
positions in that band.

- **The only selection predicate is the interior band.** G is never used to frame or to
  filter. Each drawn candidate's counterfactual G *is* recorded (`G_counterfactual`) for
  the analysis afterward; nothing reads it before the labels exist.
- **Scale is drawn to the 487's realized mix** (0.867 / 0.103 / 0.031 at 0.06 / 0.09 / 0.14
  — measured off `batch_v1/draw.jsonl`, pinned by a test), so scale is not a second thing
  varying between this batch and the old one.
- **≤ 3 crops per atom**, and two windows from one atom must clear the screen's own
  elliptical separation (`HT.SEP`) — same NMS metric the deployed framing uses.
- **Split inherited** from the source roster atom, never reassigned. The per-cell *atom
  order* offers an eval atom every 4th slot, so both arms carry the same eval share rather
  than drawing it by luck (a partial-sweep dry run gave 0.43 vs 0.10 without it).
- **Presentation**: blind by default and seeded-shuffled, the same rig as the 487
  (`presentation_seed` = 0x1B0DE5 in `batch.json`). Beyond that, the `image_id` itself is
  opaque — `ib<shuffled slot>_<content hash>`. The 487's ids encoded the screen's fate;
  these encode nothing, because the id is the one string that reaches the browser as a URL
  even when the UI is blind.
- **Recorded, never selected on**: `int_perim_area` and `coh_scale_drop` (the two features
  that survived degree-conditioning in the bake-off) are computed on every drawn crop at
  draw time, on the crop's own re-derived 1280×720 f64 escape-time field, via
  `interior_bakeoff.crop_features` — the same function that produced the 487's numbers, so
  the two batches are directly comparable.

### Verification

Drawn, featurized, rendered (all 80 crops: canonical + vivid), reported. Sheet:
`scratch/interior_band_batch/band_sheet.png`; full text: `scratch/interior_band_batch/report.txt`.

**1. The crossing filled.** Every one of the 16 (band × degree) cells holds exactly 5
crops — no under-fill, no backfill across bands. 60 in the interior arm, 20 in the control.
Both arms carry an identical **20% eval share** (12/60 and 4/20), by the every-4th-slot atom
ordering rather than by luck.

| arm | n | train | eval | per-cell |
|---|---|---|---|---|
| `interior_band` (3 bands × 4 deg) | 60 | 48 | 12 | 5/5/5/5 all cells |
| `low_interior_control` (4 deg) | 20 | 16 | 4 | 5/5/5/5 |

**2. Interior is the only thing that varies between the arms.** Same sampler, same grid,
so on everything *except* interior the two arms line up:

| quantity | interior arm | control arm |
|---|---|---|
| `g_interior` (median) | **0.291** | **0.0001** |
| scale mix 0.06 / 0.09 / 0.14 | 0.883 / 0.100 / 0.017 | 0.950 / 0.000 / 0.050 |
| mean degree | 3.50 | 3.50 |
| mean period | 8.53 | 7.55 |
| eval share | 0.200 | 0.200 |
| `G_counterfactual` (median) | −11.20 | −3.52 |

Interior fraction separates cleanly (0.291 vs ~0); degree, eval share are identical; scale
mix tracks the 487's target (0.867 / 0.103 / 0.031) on both arms modulo the 0.09/0.14 tail at
n=20. The `G_counterfactual` gap (−11.2 vs −3.5) is the confound this batch is built to expose,
not a leak: G would rank the interior arm far below the control **because G penalizes interior**
— which is exactly why G-maxima framing could never have produced these windows, and why the
counterfactual is recorded and never selected on.

**3. No duplicate pictures.** ≤ 2 crops per atom (cap 3), 64 atoms; **zero same-atom window
pairs with IoU > 0**.

**4. Mask clauses (recorded, never selected on).** All 60 interior-arm windows trip the
`interior` clause by construction. The control arm carries the swept grid's own base rate —
15 unmasked, 5 `flat` — because it is drawn by the identical sampler with no extra predicate.

**5. Recorded crop features (computed at draw time, never a selector).** Medians by band:

| band | int_perim_area | coh_scale_drop |
|---|---|---|
| control | 0.147 | 0.079 |
| i10_20 | 0.050 | 0.081 |
| i20_35 | 0.036 | 0.080 |
| i35_50 | 0.026 | 0.081 |

`coh_scale_drop` is flat across the interior axis (~0.08 everywhere) — consistent with the
bake-off's read that it measures something orthogonal to interior mass. `int_perim_area`
*falls* as the band deepens (thicker interior → lower perimeter-to-area).

**6. Resolution parity.** Spearman(screen `g_interior`, crop `int_frac` @1280×720) = **+0.999**
(n=80): the screen-resolution quantity the mask cuts on is a near-perfect proxy for the
crop-resolution interior fraction, so selecting on the screen quantity is selecting on what the
labeler sees.

Ready to label (`tools/viz/corpus_label.html`, blind + shuffled). What to do about the mask is
Matt's call once the labels are in.
