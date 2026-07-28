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

Filled in below once the batch is drawn — see the verification section.
