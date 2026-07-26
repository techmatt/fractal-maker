# Does the q4 stage-1 screen transfer to d3/d4/d5 multibrot minibrots?

**Question.** The stage-1 goodness screen (coarse pre-filter → OOD mask → L1 goodness
field G → G-maxima framing) was fitted on **degree-2** minibrot windows. Before the
minibrot pipeline is designed we need to know whether it works on **d3/d4/d5
multibrot** minibrots, and whether there is a q4 vein in those neighborhoods at all.

**Answer: yes, it transfers.** The fitted screen — run *unchanged* — accepts, rejects,
and OOD-masks multibrot windows at rates comparable to d2; the G field keeps its shape
and positive tail; and G-maxima framing produces q4-quality candidates at every degree
(d3 finds *more* above the d2 cutoff than d2 does). The eye confirms it: the accepted
sheets are ornate multibrot dendrite/spiral filigree, not degenerate crops.

**One load-bearing caveat, stated up front:** the transfer only holds once the
**minibrot atom-size law is corrected for degree** (exponent `d/(d-1)` on λ, not the
degree-2 `λ²`). With the naïve d2 law, a `4·|size|` frame lands *inside* the d≥3
minibrot body → all-black fields → the screen (correctly) OOD-masks ~everything. That
first result was an artifact of **sourcing scale**, not the screen. See "The trap we
fell into" below — it is exactly the artifact-vs-inherent confound this read exists to
avoid, and it bit at the sourcing layer rather than the screen layer.

## Method (apples-to-apples, nothing refit)

- **Sourcing.** `deep_center_finder` generalized to `z^d+c` (degree threaded through
  the orbit/derivative recurrences + Newton; d=2 kept byte-identical). 12 minibrot
  nuclei per degree, spread across periods 3–15, Newton-refined, kept in the same
  f64-dumpable size band as the d2 label-set. A freshly-sourced **d2** set (not the
  training corpus — that would be in-sample) is the control.
- **Fields.** Each nucleus → an f64 field via the **same** `render-one --dump-field`
  path, degree carried by `--family multibrot{d}` (`F64Backend` is already
  degree-parametric). 2176×1224 ss1, `fw = 4·|size|`. (Rendered with
  `--dump-field-source f64` — offset-invariant to the screen and ~35× faster; see
  `beautiful_perf_report.md`.)
- **Screen — unchanged.** The deployment model is refit exactly as `q4_harvest_tight`
  does (`LF.surviving_weights(.., "T2_cells", 2.0)` over the 340 d2 labels, 33 corpus
  minibrots; tight cutoff **G ≥ 1.390**, labeled precision 0.85). Every gating decision
  is the screen's own function — `LF.featurize`, `LF._v2_drop` (the coarse-prefilter ≡
  OOD-mask ceilings: interior 0.10 / flat 0.88 / speckle 0.30), `clf.decision_function`,
  `HT._all_peaks` + elliptical NMS. The instrumentation only *counts* fates; it is
  asserted to reproduce `LF.dense_grid`'s survivor set exactly.
- Code: `tools/studies/q4_multibrot_transfer.py`. Disposable outputs under
  `scratch/q4_multibrot_transfer/` (`stats.json`, `sheet_d{2..5}.png`).

Note on the two named stages: in the **deployed** harvest path the "coarse pre-filter"
and the "OOD mask" are the *same* three-ceiling `_v2_drop` predicate (the coarse
`filter_v2` metrics and the featurize globals are the same quantities). So the two rates
the brief asks for coincide by construction; the informative split is the **per-clause**
rejection (interior / flat / speckle), which is where the degree signal lives.

## Results (d2 control vs d3/d4/d5)

Aggregated over 12 fields/degree, 3 scales, dense position sweep. "Featurizable" =
swept positions with ≥64 finite pixels (the rest are all-interior crops).

| degree | featurizable | OOD-mask reject | G median | G range (min…max) | maxima kept | accepted (G≥1.39) |
|---|---|---|---|---|---|---|
| **d2** (control) | 158 789 | **65.0 %** | −2.81 | −13.2 … 2.5 | 48 | **8** |
| **d3** | 154 891 | **55.2 %** | −2.97 | −13.7 … 2.6 | 48 | **23** |
| **d4** | 150 475 | **60.4 %** | −3.41 | −14.2 … 4.1 | 48 | **16** |
| **d5** | 140 008 | **64.0 %** | −4.36 | −14.6 … 3.3 | 48 | **12** |

OOD-mask rejection **by clause** (fraction of featurizable positions each ceiling trips):

| degree | interior ≥0.10 | flat ≥0.88 | speckle ≥0.30 |
|---|---|---|---|
| d2 | 19.1 % | 51.3 % | 1.8 % |
| d3 | 23.5 % | 36.6 % | 2.7 % |
| d4 | 27.6 % | 36.9 % | 4.4 % |
| d5 | 32.1 % | 36.6 % | 6.5 % |

### Reading

1. **Coarse pre-filter / OOD-mask pass rate — comparable, not degenerate.** Multibrot
   is rejected at **55–64 %**, *at or below* d2's 65 %. The mask is **not degree-bound**:
   multibrot windows survive it at the same rate d2 windows do. (The failure mode the
   brief warned about — the mask rejecting all multibrot — appeared only under the
   broken size law, and was a black-field artifact, not the screen.)

2. **Per-clause degree signal is inherent geometry, not screen failure.** `interior`
   rises monotonically with degree (19 → 24 → 28 → 32 %) — the (d−1)-fold nucleus has a
   proportionally larger interior body, exactly as predicted. `flat` *drops* (51 → 37 %)
   — multibrot decorations are busier, less barren. `speckle` rises modestly (1.8 → 6.5 %)
   — higher-degree ornament is finer. All three are real differences in what a multibrot
   neighborhood *is*, and the mask responds to them correctly.

3. **G distribution — shifted, not compressed or degenerate.** The median drifts down
   with degree (−2.81 → −4.36) but the **range and shape are preserved** (≈[−14, +3] at
   every degree) and the **positive tail is intact** — d4 and d5 reach *higher* max G
   (4.1, 3.3) than d2 (2.5). The screen is not collapsing multibrot G to a dead point;
   it is scoring a full spread, top end included.

4. **G-maxima framing yields q4 candidates at every degree.** 48 kept framings per
   degree (PER_MB_CAP 4 × 12 fields), and windows clear the **d2-fitted** cutoff G≥1.39
   at every degree: d2 = 8, **d3 = 23**, d4 = 16, d5 = 12. The screen doesn't just
   tolerate multibrot — it *finds* q4-quality windows there, d3 more abundantly than d2.

5. **The eye agrees (fate sheets).** `sheet_d{2..5}.png`, vivid blue/orange field
   colorize (not `twilight_shifted`). **Accepted** = ornate dendrite/spiral filigree,
   well-framed — genuine wallpaper candidates. **Rejected (survived, G<cutoff)** =
   sparser / less-balanced structure. **OOD-masked** = interior-heavy black-blob or
   barren-gradient crops. The three fates look the same across degrees; the higher-degree
   accepted windows are the deep (p15) minibrots, where the atom-size law is most
   accurate and the framing tightest.

**There is a q4 vein in d3/d4/d5 multibrot neighborhoods, and the degree-2 screen finds
it.** The minibrot pipeline can reuse the stage-1 screen across degrees without a refit.

## The trap we fell into (artifact vs. inherent, at the sourcing layer)

The first full run reported d3 OOD-reject **100 %**, d4 86 %, d5 ~all — an apparently
damning "screen is degree-bound." It was not. Interior-fraction of the raw fields was
**~1.00** for d3/d4/d5 vs **~0.16** for d2: the fields were **solid black**. Cause: the
degree-2 minibrot size estimate (`size = 1/(b·λ²)`) under-estimates the d≥3 atom by
~4–11×, so `fw = 4·|size|` framed a region *entirely inside* the minibrot body. The
screen was correctly OOD-masking black windows — the artifact was in the *sourcing
scale*, not the screen.

Fix: the multibrot renormalization scaling puts the atom's linear size at `|λ|^{-d/(d-1)}`
(the p-fold iterate near a period-p nucleus is a small `w→w^d+c` copy). `d/(d-1)` reduces
to 2 at d=2 (d2 untouched, byte-identical). Validated by rendered interior-fraction:
with the corrected law a `4·|size|` frame lands at interior-frac ≈0.2–0.5, comparable to
d2's 0.16, and the fields show real exterior decoration — which is what the numbers above
are computed on. This correction now lives in `deep_center_finder.nucleus_size_estimate`.

## Spec deviations (flagged, with reasons)

- **`--dump-field-source f64` instead of the default `beautiful`.** The two smooth
  fields differ only by the constant `ln(ln B)/ln d`, which is washed out by
  `featurize`'s per-crop percentile stretch — verified: interior mask identical,
  features agree to 1.3e-9, `_v2_drop` identical. ~35× faster. Both the corpus refit
  fields and the transfer fields use it, so the model is unchanged. Full write-up:
  `beautiful_perf_report.md`.
- **Size-law exponent `d/(d-1)`** (above) — required to source non-degenerate multibrot
  fields at all; without it the read is confounded by black fields. d=2 unchanged.
- **Coarse pre-filter and OOD mask reported as one rate** — they are the identical
  `_v2_drop` predicate in the deployed harvest; the per-clause breakdown carries the
  degree signal the two-number split was meant to expose.
- The d2 baseline is **freshly sourced** (out-of-sample), not the training corpus, so the
  control is not inflated by in-sample G.
