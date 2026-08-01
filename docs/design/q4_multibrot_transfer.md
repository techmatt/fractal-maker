# Does the q4 stage-1 screen transfer to d3/d4/d5 multibrot minibrots?

**Question.** The stage-1 goodness screen (coarse pre-filter → OOD mask → L1 goodness
field G → G-maxima framing) was fitted on **degree-2** minibrot windows. Before the
minibrot pipeline is designed we need to know whether it works on **d3/d4/d5
multibrot** minibrots, and whether there is a q4 vein in those neighborhoods at all.

**Answer: the screen transfers — the mask is not degree-bound and multibrot
neighborhoods do yield accepted q4 windows — but the cross-degree comparison had to be
re-read after removing two confounds (rotational-copy pseudo-replication and period
mismatch), and one headline claim did not survive.** What holds up: the OOD mask accepts
multibrot windows *at or below* the rate it rejects d2 (so the vein is not masked away);
the G field keeps its shape and positive tail; and G-maxima framing produces q4-quality
candidates at every degree, which the eye confirms as ornate multibrot filigree. What did
**not** hold up: the raw "d3 finds *more* accepted windows than d2 (23 vs 8)" ranking —
that was a period-coverage artifact (see "The confound re-read" below), not a degree
effect. The accepted-count *ranking across degrees* is retracted; the *existence* of a q4
vein at each degree stands.

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
- **Symmetry-canonical dedup (added in the re-read).** `z^d+c` has **(d−1)-fold
  rotational symmetry** about the origin: `c` and `c·ω^k` (`ω = exp(2πi/(d−1))`) are the
  *same* atom under the conjugacy `z→ωz` — same period, same |size|, rotated field.
  Rounded-coordinate dedup alone let rotational copies survive as separate "minibrots".
  `deep_center_finder.nucleus_dedup_key` now canonicalizes `c` into the fundamental
  sector `arg c ∈ [0, 2π/(d−1))` *before* rounding (d=2 = identity, byte-identical), so a
  clean per-degree distinct count is a guard, not luck. Regression-covered
  (`test_rotational_copies_collapse_to_one_key`).
- **Fields.** Each nucleus → an f64 field via the **same** `render-one --dump-field`
  path, degree carried by `--family multibrot{d}` (`F64Backend` is already
  degree-parametric). 2176×1224 ss1, `fw = 4·|size|`. (Rendered with
  `--dump-field-source f64` — offset-invariant to the screen and ~35× faster; see
  `render_coloring_surface.md` §7.)
- **Screen — unchanged.** The deployment model is refit exactly as `q4_harvest_tight`
  does (`LF.surviving_weights(.., "T2_cells", 2.0)` over the 340 d2 labels, 33 corpus
  minibrots; tight cutoff **G ≥ 1.390**, labeled precision 0.85). Every gating decision
  is the screen's own function — `LF.featurize`, `LF._v2_drop` (the coarse-prefilter ≡
  OOD-mask ceilings: interior 0.10 / flat 0.88 / speckle 0.30), `clf.decision_function`,
  `HT._all_peaks` + elliptical NMS. The instrumentation only *counts* fates; it is
  asserted to reproduce `LF.dense_grid`'s survivor set exactly.
- Code: `tools/studies/q4_multibrot_transfer.py`; the post-hoc confound re-read is
  `tools/studies/q4_multibrot_transfer_reread.py` (no re-source, no re-render — it
  collapses rotational copies and re-screens the **cached** `.bin` fields per-minibrot,
  recovering the per-window G/clause detail the aggregate `stats.json` didn't persist).
  Disposable outputs under `scratch/q4_multibrot_transfer/`.

Note on the two named stages: in the **deployed** harvest path the "coarse pre-filter"
and the "OOD mask" are the *same* three-ceiling `_v2_drop` predicate (the coarse
`filter_v2` metrics and the featurize globals are the same quantities). So the two rates
the brief asks for coincide by construction; the informative split is the **per-clause**
rejection (interior / flat / speckle), which is where the degree signal lives.

## The confound re-read (what changed, and why)

The first cross-degree table (12 fields/degree, size-band-matched) had two problems that
can each manufacture a monotone trend across degree. Both are fixed from data already on
disk — no re-source, no re-render.

**1. Rotational-copy pseudo-replication.** The rounded-coordinate dedup kept rotational
copies of one atom as separate minibrots, and it did so **unequally** by degree. With the
symmetry-canonical key, the distinct counts are **d2 12/12, d3 12/12, d4 10/12
(one 3-fold p4 family → 1), d5 8/12 (four copies dropped)**. So d3's clean result was
luck; d4/d5 were inflated. Collapsing the copies removes that weight.

**2. Period mismatch.** The 12/degree were matched on **size band, not period.** Because
`P_n′` has degree `d^(n−1)`, |A| grows faster with period at higher degree, so the fixed
f64-dumpable size band admits systematically *lower* periods as degree rises. It did — the
period coverage is badly mismatched:

| degree | period distribution (collapsed) | note |
|---|---|---|
| d2 | 3, 5, 6, 6, 7, 7, 7, 8, 8, 9, 11, 15 | mass at 7–9, reaches 15; **no p4** |
| d3 | 3, 4, 5, 5, 6, 6, 7, 8, 9, 10, 12, 15 | broad, well-spread |
| d4 | 3, 3, 4, 5, 6, 7, 8, 9, 10, 10 | **caps at period 10** |
| d5 | 3, 3, 4, 5, 5, 6, 15, 15 | **bimodal**: 3–6 plus a p15 cluster |

So the raw comparison partly compared *different period regimes*. To remove it, condition
on the period band **every** degree populates — periods **3–6** — and recompute. (This
also drains the q4 signal, because at every degree the accepted windows live in the
*deeper*, higher-period minibrots — see reading point 4. The p3–6 cut therefore tests the
mask/G-shape drifts cleanly but has too few high-G positions to compare abundance.)

## Results (confound-corrected)

Re-aggregated over the **collapsed** (distinct-atom) sets; "featurizable" = swept
positions with ≥64 finite pixels. RAW (all sourced, pre-collapse) reproduces the original
`stats.json` exactly — the per-minibrot re-screen is faithful.

**Collapsed, all periods:**

| degree | eff. n | featurizable | OOD-mask reject | G median | G P90 | G max | accepted (G≥1.39) |
|---|---|---|---|---|---|---|---|
| **d2** (control) | 12 | 158 789 | **65.0 %** | −2.81 | −1.01 | +2.45 | 8 |
| **d3** | 12 | 154 891 | **55.2 %** | −2.97 | −0.69 | +2.63 | 23 |
| **d4** | 10 | 124 950 | **57.7 %** | −3.33 | +0.02 | +4.06 | 16 |
| **d5** | 8 | 92 766 | **64.1 %** | −4.50 | −0.28 | +3.28 | 8 |

**Collapsed + conditioned on period ∈ {3,4,5,6}** (the common band):

| degree | eff. n | featurizable | OOD-mask reject | G median | G max | accepted |
|---|---|---|---|---|---|---|
| d2 | 4 | 53 048 | **75.1 %** | −3.10 | +0.52 | 0 |
| d3 | 6 | 76 965 | **61.5 %** | −3.50 | +2.60 | 5 |
| d4 | 5 | 62 267 | **70.6 %** | −4.02 | +0.48 | 0 |
| d5 | 6 | 67 553 | **71.7 %** | −5.54 | +0.37 | 0 |

OOD-mask rejection **by clause** (fraction of featurizable positions each ceiling trips):

| degree | interior ≥0.10 (collapsed / p3–6) | flat ≥0.88 (collapsed / p3–6) | speckle ≥0.30 (collapsed / p3–6) |
|---|---|---|---|
| d2 | 19.1 % / 19.1 % | 51.3 % / 61.8 % | 1.8 % / 1.7 % |
| d3 | 23.5 % / 24.3 % | 36.6 % / 41.9 % | 2.7 % / 3.0 % |
| d4 | 28.0 % / 28.0 % | 33.7 % / 47.4 % | 4.7 % / 4.0 % |
| d5 | 32.8 % / 35.2 % | 35.6 % / 41.9 % | 6.7 % / 6.6 % |

### Reading — which claims survive

1. **Headline: OOD-mask is not degree-bound — SURVIVES, and strengthens.** Multibrot is
   rejected at **55–64 %** (collapsed), *at or below* d2's 65 %; when period-matched, d2's
   shallow fields are rejected **hardest of all** (75 % vs multibrot's 61–72 %). So under
   every cut the multibrot vein survives the mask at least as well as d2. The failure mode
   the brief warned about (mask rejecting all multibrot) appeared only under the broken
   size law, and was a black-field artifact. Stated loudly, as requested: **this survives.**

2. **G-median downward drift with degree — SURVIVES as a genuine degree effect.** The raw
   drift (−2.81 / −2.97 / −3.41 / −4.36) was **not** duplication: collapsing barely moves
   it (−2.81 / −2.97 / −3.33 / −4.50). And it was **not** period mismatch: conditioning on
   period 3–6 *steepens* it (−3.10 / −3.50 / −4.02 / −5.54). Higher-degree neighborhoods
   genuinely score lower-median G at matched period. The **range and positive tail are
   still preserved** (max G ≈ +2.5…+4.1 at every degree, collapsed) — the screen is not
   collapsing multibrot G to a dead point, it is shifting a full spread downward.

3. **Interior-clause rise (19 → 33 %) — SURVIVES.** Monotone under collapse
   (19.1 / 23.5 / 28.0 / 32.8) *and* under period conditioning (19.1 / 24.3 / 28.0 /
   35.2). The (d−1)-fold nucleus has a proportionally larger interior body; the mask reads
   it as more interior at matched period, so this is inherent geometry, not confound.
   **Flat-clause "51 → 37 %" survives as a d2→multibrot step, not a smooth gradient:** d2
   is the flat/barren one (51 %, and 62 % at low period); every multibrot degree sits
   ~34–37 % (collapsed) with no monotone ordering among d3/d4/d5. Speckle rises modestly
   with degree (finer ornament) and survives both cuts.

4. **Accepted-window abundance ranking (raw d2=8, d3=23, d4=16, d5=12) — DOES NOT
   SURVIVE; retracted.** Two problems compound: (a) pseudo-replication inflated d5 (its
   raw 12 → **8** after collapse, the dropped accepts being rotational copies of its p15
   minibrots); (b) the accepts live almost entirely in the **deeper, higher-period**
   minibrots at *every* degree — at matched period 3–6 the accepted counts are
   **0 / 5 / 0 / 0** and G>0 ≈ 0 % everywhere, **including d2**. So the raw per-degree
   accepted counts track each degree's period *mix* (d2 skews high 7–15; d4 caps at 10;
   d5 is 3–6 plus p15), not a degree property. The correct statement is **existential, not
   comparative**: q4-quality windows *do* appear in multibrot neighborhoods (from their
   deeper minibrots), but this data cannot rank the vein's *abundance* across degree.

5. **The eye agrees (fate sheets).** `sheet_d{2..5}.png`, vivid blue/orange field
   colorize. **Accepted** = ornate dendrite/spiral filigree, well-framed — genuine
   wallpaper candidates. **Rejected (survived, G<cutoff)** = sparser / less-balanced
   structure. **OOD-masked** = interior-heavy black-blob or barren-gradient crops. The
   three fates look the same across degrees; the accepted windows are the deeper
   (higher-period) minibrots, consistent with reading 4.

**Effective-n caveat.** Position counts are large (53–90 k per period-conditioned cell),
so the pooled *rates and medians* are well-powered; but atom-level replication after
conditioning is thin — **4 / 6 / 5 / 6** distinct minibrots — so treat the
period-conditioned medians as directional (d2's p3–6 numbers rest on just 4 atoms), and do
not read a per-degree *count* (e.g. accepted) off the conditioned cut. No degree was left
with too few distinct sources to make the *rate* comparisons, but the accepted-abundance
comparison is exactly the one the thin high-G tail cannot support (reading 4).

**Bottom line.** There is a q4 vein in d3/d4/d5 multibrot neighborhoods and the
degree-2 screen finds it: the mask is not degree-bound (survives, strengthened), and the
G field keeps its shape and top end. The minibrot pipeline can reuse the stage-1 screen
across degrees without a refit. The genuine degree signals are the **downward G-median
drift** and the **rising interior fraction** (both real at matched period); the apparent
degree *ranking of accepted-window abundance* was a period-coverage artifact and is
dropped.

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
  `render_coloring_surface.md` §7.
- **Size-law exponent `d/(d-1)`** (above) — required to source non-degenerate multibrot
  fields at all; without it the read is confounded by black fields. d=2 unchanged.
- **Coarse pre-filter and OOD mask reported as one rate** — they are the identical
  `_v2_drop` predicate in the deployed harvest; the per-clause breakdown carries the
  degree signal the two-number split was meant to expose.
- The d2 baseline is **freshly sourced** (out-of-sample), not the training corpus, so the
  control is not inflated by in-sample G.
- **Confound re-read is post-hoc on cached data** — the collapsed/conditioned figures come
  from `q4_multibrot_transfer_reread.py` re-screening the on-disk `.bin` fields (no
  re-source, no re-render); the sourcing guard (`nucleus_dedup_key`) fixes *future* runs
  but was **not** re-run here, so the field set is the original one, collapsed post-hoc.
