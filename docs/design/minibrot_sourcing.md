# Minibrot Sourcing — the Atom Roster, the q4 Screen, and What the 487 Measured

**Status:** the sourcing and labeling legs are built and measured. No q4 network exists and none is planned — "what is the emit quality of this image" is the question the quality head already answers, so q4 is **class 4 on the 1–4 label scale**, not a separate model. What exists separately is the **screen**: a heuristic field over windows, used for sourcing, never for judging.

**Boundary — this file is the MODERATE-DEPTH arc, inside f64.** Everything measured here ran at moderate nucleus depths on the f64 path. The *deep* regime — beyond-f64 precision, the perturbation tier, ∂M tracking, and how valid deep centers are produced — is `deep_zoom_sourcing.md`, which is PARKED. The two files touch at exactly two places: that file's framing rule depends on the atom-size law used here (`size ≡ 1/|A|`), and the "own fitted objective" it calls for is the screen `G` characterised below. Do not restate its content here.

**Scope — what this file owns.** The atom roster, the stage-1 screen `G` and its measured worth, the pre-filter, the atom- and window-level signals found in the labeled data, and the measurement discipline that makes those numbers mean anything. It does **not** own: the `A` instrument's mathematics (engine; `atom_instrument` / `deep_center_finder`), julia c-sourcing (`julia_c_sourcing.md`), the harvest→emission path (`q4_harvest_emission.md`), or the classifier retrain protocol (`classifier_retrain_protocol.md`). Those are referenced by name here and their values are not restated.

---

## 1. Architecture — the search tree

Find minibrot atoms → screen their neighbourhoods → propose candidate windows → **one classifier decides**.

Generation is a q4 detector run over a search tree: classify the base window → emit if q4; **descend regardless**, field-steered; classify each descent crop with the same net; q4 nodes descend further.

Two properties are load-bearing and should survive any redesign:

- **Descend-worthiness is SEARCH, not prediction.** Deciding "is there more below" by descending and looking deletes an expensive label that would otherwise have to be collected, and search is robust exactly where a predictor is weakest.
- **Steering uses the q4 field, never the location head.** The location head is uncertified on minibrot neighbourhoods and at deeper zoom; using it to steer here would be an out-of-domain application of a model whose in-domain certification does not transfer.

---

## 2. The roster — supply

`tools/sourcing/build_minibrot_roster.py`, with `pilot_harvest.py` downstream. Output: `data/minibrot_roster/roster.jsonl`, durable.

Shape: **4 degrees × 5 period bands × 8 atoms**, all 160 cells filled. Selection is per-(degree, period band); the size band that used to be a third axis is replaced by the `A` feasibility cut (§6).

Rules that must not be relaxed:

- **Atom-level split is assigned at build time and inherited by every crop thereafter, never reshuffled.** A crop's split follows its atom, always.
- **Under-filled cells are reported, never backfilled across bands.** Backfilling re-confounds degree with depth, which is the exact confound the grid exists to break.

The old belief that "d4 tops out near period 10 and d5 is bimodal" was a **seeding-plus-subsample artifact**, not a degree ceiling. Minimal p13–15 nuclei exist at every degree.

### 2.1 How a nucleus is actually seeded — as-built, and it is a DRAW, not a census

Four properties of `deep_center_finder` that constrain every rate computed off its output. There is no polynomial root-finding (no Aberth/Durand–Kerner) and no atom-domain detection from a rendered field.

1. **Newton is seeded from a ring grid, blind.** `_ring_seeds(degree)` = 3 radii from `0.30` to `1.05·R_boundary` (`R_boundary = 2^{1/(d−1)}`) × 24 angles = **72 seeds**, each tried at **every** period in `range(3,16)` — 936 solves per degree. (`emit_deep_pool.py` instead uses a hand-curated `SEEDS` list, d2 only.)
2. **Period is a loop variable, not a measurement.** Nothing reads a period estimate off a field or an orbit; the sweep keeps whatever converges.
3. **Minimality IS confirmed; largeness is not.** After `|z_n| < tol`, `_is_minimal_nucleus` rejects the solution if any proper divisor `q | period` also closes, so the reported period is exact rather than a multiple. A `|c| < 1e-6` guard drops the `z=0` period-1 degenerate.
4. **Dedup is by rounded coordinate only** (`DEDUP_DPS = 22`), with **no symmetry dedup** — so the draw contains rotational copies (`c ↦ ωc`, `ω^{d−1}=1`), which a coordinate key cannot collapse. Any "distinct look" count off a raw ring draw is inflated by up to `(d−1)×`. Canonicalizing each nucleus into one symmetry sector before the coordinate dedup is the fix when this becomes a supply pipeline. `q4_multibrot_transfer.md` owns the measured pseudo-replication and its removal.

**Consequence: screen-transfer rates computed over this population are not base rates.** The draw is {ring seeds} → Newton at 3–15 → keep minimal → size-band → coordinate-dedup → explicit subsample. Read those rates as *"the d2 screen treats comparably-sourced d≥3 atoms comparably"*, never as a yield. `[measured: 2026-07-26/27; code: deep_center_finder.newton_nucleus / _is_minimal_nucleus, q4_multibrot_transfer.source_nuclei]`

**Three atom counts appear in this arc and they are different populations — do not reconcile them by assumption.** The roster holds 160 filled cells; the `A` feasibility cut was evaluated over 163 atoms; the labeled roster's realized split covers 145 atoms. Quote the one that matches your population, and reconcile them properly before quoting an atom-level rate.

---

## 3. The stage-1 screen `G`

**Pipeline:** HP sourcer → render minibrot fields → sweep windows → coarse pre-filter → OOD mask → L1 goodness field → G-maxima framing → harvest.

**Corpus:** `data/q4_window_corpus/`, reader `tools/corpus/q4_window_reader.py`. Registered **separately** from the location corpus. Its rows are window-level and distribution-bound; **do not pool them with location-corpus rows.** Its label files carry 3-way string classes and are excluded from the location-corpus scan by registration (`FOREIGN_LABEL_FILES` / `is_v7_corpus_label_file`), while remaining physically under `labels/` because that path prefix is what confers never-delete protection.

### 3.1 What G is worth — measured

[human n=487, blind]

| quantity | value |
|---|---|
| pooled AUC(label ≥ 3) | 0.605 |
| AUC **within its own accepts** | 0.511 |
| Spearman within accepts | 0.023 |
| mean label, accepts | 1.96 |
| mean label, rejects | 1.71 |
| mean label, OOD-masked windows | 1.08 |

Read plainly: **G is a weak gate and a dead ranker.** Its accepts average *below* "good." The part that works is the **OOD mask**, which produces a clean floor of 1s. Use G to discard junk; never to order candidates.

**11% of what G rejects is good** — 27 of 250 sub-cutoff crops scored ≥3. Hunting outside the accepts pays.

**The screen accepts nothing at all from the shallowest period band.** Every p3–4 atom at every degree screens to zero accepts, which is why the positive arm of the 487 came in at **237, not 250** — accepts are genuinely scarce outside the mid/deep bands. That is a property of the deployed screen, not a draw shortfall, and it is not fixed by padding with low-quality accepts. A low-period minibrot is large and simple: its `4·size` frame is mostly black body plus plain surround, which the d2-trained screen dislikes.

**G is less dead than the accept-arm number alone suggests — but only on train.** Held at fixed degree, train AUC recovers to **0.59 / 0.78 / 0.68** (d3/d4/d5), so "flat above the cutoff" (within-accept AUC 0.511) is a statement about the post-cutoff tail, not about G's ordering across its full range. **On held-out atoms it collapses back to chance: eval 0.500 / 0.510 / 0.479** — in exactly the cells where `coh_scale_drop` (§8) holds at 0.80 / 0.87. The verdict stands on the eval column.

**The recomputation is verified, so these are statements about the windows actually drawn.** The deployed screen's 15 T2 features recomputed on the exact drawn window reproduce the stored deployed G to **max |ΔG| = 0.0000 over all 462 scored rows**, and crop-resolution interior agrees with screen-resolution interior at ρ = **+0.984** (487 crops) / **+0.999** (the 80-crop interior band) — so nothing here is a re-derivation or resolution artifact. `[measured: 2026-07-27; cmd: uv run python -m tools.studies.interior_bakeoff board]`

**Superseded at sourcing by the orbital ring measures** (`orbital_field_metrics.md`), which
own that verdict and the evidence for it. G's numbers above stand; do not invest further in
G. Nothing is wired — the supersession is a judgement, not a pipeline change.

### 3.2 Diagnosis — what G actually measures

The accepts that scored 1–2 share one look: a lone crisp filament in a flat field. G is an **edge-energy statistic with no occupancy term**, and dendrites maximise edge per unit area. G is not broken; it is measuring something anticorrelated with the target inside its own accept region.

The cheapest known improvement follows directly: **add an occupancy/fill term.** That is much cheaper than replacing the screen and attacks the diagnosed failure mode by name.

### 3.3 Invariants any successor inherits

1. **The OOD mask is load-bearing and permanent.** G is defined only on the pre-filter manifold — filter and field are one system, not two.
2. **`p` is not calibrated.** Gate from labeled precision, never from a p-value.
3. **G-maxima auto-framing replaces score_A NMS.** It is the framing mechanism, not an add-on.
4. **Deterministic seed before any weight-stability claim.**
5. **The referee is a minibrot-disjoint LOMO split.**

---

## 4. The pre-filter, and the interior-clause catch-22

Rejection ceilings: `interior_frac ≥ 0.10` · `flat_frac ≥ 0.88` · `speckle_ratio ≥ 0.30`.

The interior clause is large and expensive. Over 296k swept positions it alone removes **20.2%** of what the screen looks at, and dropping it would grow the scoreable pool by **49.8%**.

**It is also not honestly adjudicated, and the evidence that looks like a verdict is not one.** A dedicated 80-crop labeling of the `[0.10, 0.50]` interior band returned **zero** rows ≥3 — including its own low-interior control. That looks decisive until you notice the same batch showed the killer is the **absence of G-framing**, not interior mass: uniform-sampled crops averaged 1.07 against G-framed 1.84, at every degree. So the batch tested "uniform sampling vs G-framing," not "high interior vs low interior."

**Do not move the threshold on that evidence.**

The catch-22 is structural: G-framing cannot produce high-interior windows (`interior_worst` = −1.278 inside G), and uniform sampling produces high-interior windows but no good material. Neither arm can convict or clear the clause.

**The binding ceiling is G's weight, not the mask's threshold.** No drawn accept exceeds `g_interior` = **0.037** — barely a third of the mask's own 0.10 cut. So moving the mask threshold would change almost nothing about what gets accepted; `interior_worst` = −1.278 is what actually holds the accept arm at near-zero interior. Any experiment aimed at the clause has to move the *weight*, or bypass G-framing entirely (which is what the interior-band batch did).

**The way out is constructible and designed but unrun: the `G_cf` framing experiment.** Frame by maximising G with the interior clause removed — the `G_cf` objective is already computed. Note that the reported non-overlap of `G_cf` between the existing arms is an artifact of uniform sampling (uniform windows score low regardless of interior), *not* evidence that high-interior windows cannot score well under a de-interiored objective. Worth ~20% of swept positions and ~50% pool growth.

---

## 5. Atom-level signal — degree, and two live objections

**Degree is the strongest measured atom-level signal.** Spearman(degree, label) = **+0.55** raw; conditional on interior mass **+0.399 CI[+0.304, +0.478]** train and **+0.683 CI[+0.541, +0.763]** eval. Mean label runs d2 1.22 → d5 2.27. Degree is orthogonal to G (−0.07), and conditioning on degree *sharpens* G.

The reverse does not hold: interior mass given degree is **+0.046 / +0.151**, both CIs spanning zero. **Interior mass was degree's shadow, not the reverse.**

Two objections are live and a "degree is the draw axis" decision has to survive both:

- **Objection 1 — the `A` feasibility cut fired only on d2**, and removed d2's best material: the three excluded atoms are among the most beautiful sourced. d2's mean is biased downward by construction. Quantify the size of that bias before trusting the gradient.
- **Objection 2 — the cross-family anchor batch points the opposite way.** Promotions to class 4: mandelbrot 3/12 plus julia 2/10 = **5/22**, against **1/22** across all six high-degree families. Two labeled sets, one eye, one week, opposite sign. Not strictly contradictory — anchors are curated whole-set locations while the 487 are minibrot-neighbourhood crops at one scale — but the strongest single counter-example is that the human's own exemplar of a great minibrot appears to be d2.

- **Objection 3 — eval's larger degree effect is a composition difference, not corroboration.** The train and eval conditional intervals **do not overlap** (train upper +0.478 < eval lower +0.541), which looks like independent confirmation and is not: **14 of eval's 19 positives sit at degree 5**, so on eval "is it degree 5?" is very nearly "is it good?". A split where one degree carries three quarters of the positives reports a large degree correlation whether or not degree is doing the work. The honest statement before degree becomes a draw axis is the **train** number — ρ ≈ +0.40 [+0.30, +0.48] on the split with 120 atoms — with eval corroborating the *direction* and not the *size*.

- **Objection 4 — degree has never been measured free of PERIOD either.** The balanced-supply maneuver walk (four c-plane degrees, equal root supply, `data/discovery/maneuver_degree_probe/`) found the degrees do not sample the same period population at all: **median pushed period d2 22 / d3 16 / d4 9 / d5 8**, with median `log₁₀|A|` 4.17 → 3.36 → 2.78 → 2.75. Higher degrees reach *lower* periods, and low-period atoms are large and easy at every degree. And the mechanism usually offered for a degree effect — the `(d−1)`-th root damping how fast `|A|` grows with period, so a degree-`d` atom **of a given period** is intrinsically larger — **does not show once period is controlled**: within period bands, median `log₁₀|A|` is flat within noise in the low bands and inverted or too thin to read in the high ones. `[measured: 1,180 distinct operator decisions, 44 batches, 2026-07-31; `minibrot_maneuvers.md` §8]`

**Conclusion — degree has never been measured free of period or of composition.** Objection 3 is the eval-slice composition confound; Objection 4 is the period confound. Neither population was constructed to isolate degree, and a walk confounds its own axes by construction (it chooses where it goes). **A degree decision requires a draw matched on degree AND period.** Until then the direction is reportable and the size is not.

**Novelty is a live alternative reading of the degree gradient:** far more d2 has been looked at than d5 over the life of this project, so a preference for d5 may be a preference for unfamiliarity.

**Period is dead as a quality axis** [+0.06 pooled, and **−0.21** inside the period-matched eval slice]. The earlier "deep = good" signal was the screen's bias, not the human's judgement.

---

## 6. Feasibility — the `A` cut as a sourcing gate

**Admission rule:** admit an atom iff its predicted f64 pixel-spacing margin is **≥ 1 decade** at the deploy presentation. The predictor is a priori — it falls out of the same recursion Newton already runs, at ~zero cost. Mathematics and the identity `|A| ≡ 1/|size|` live with the engine (`atom_instrument`, `deep_center_finder`); do not restate them here.

**In practice it is a safety rail, not a selector** [measured, 163 atoms]: it fired on 3. Ring-seeding converges to the large-basin nucleus at each period, so the median admitted margin is 6.66 decades. The two margin≈0 exclusions genuinely fail f64 quantisation on render, 9 times out of 9; a margin of 0.95 renders fine.

**The cut only becomes a selector if you deliberately source the deep tail, and ring-seeding cannot.** Uniform rings converge to the large-basin (large-size) nucleus at each period, so even p13–15 atoms sourced this way are the prominent ones (size ≈ 2.5e-5, log|A| ≈ 4.6), not the small/deep tail. Reaching that tail needs **targeted near-∂M seeding**, not more rings — and only there does the `A` cut start doing work.

The wall is on **pixel spacing**, not centre precision, and the emission render path is f64 regardless. Consequently the deep/perturbation tier cannot rescue material this cut excludes: a location that can be rendered deep but not *scored* deep is not a candidate. See `deep_zoom_sourcing.md` §4 for why that is one dependency chain rather than four separate gaps.

**The same size law governs framing in the deep file.** `size ≡ 1/|A|`; the naïve degree-2 λ² law under-sizes d≥3 atoms by 4–2497× and frames them all-black. That law is forbidden in both files.

---

## 7. Measurement discipline for this subsystem

These are the rules that the arc's mistakes were made of. They cost real batches to learn.

- **The atom is the SEARCH unit; the window is the QUALITY unit.** Within-atom ICC is 0.68, but 25 of 105 multi-crop atoms straddle low and high labels. The per-arm rule that each atom contributes both a positive and a negative crop turned out to be ideal case-control — it forces discrimination onto the window. Keep it.
- **A hard negative has to be sourced deliberately; a deep atom will not volunteer one.** Cache only the screen's top-4 G-maxima per atom and a deep atom yields ~zero rejects — its best windows are good — so accepts land at p13–15 and rejects at p3–4, nearly disjoint, and "deep" becomes a free predictor of "good" purely by construction. The fix is to keep going past the peaks: extract the **sub-cutoff, OOD-surviving** windows the screen already surveyed and NMS them into ≤8 distinct near-miss framings per atom. Deep atoms have thousands of surviving windows (`n_surv` ≈ 7–8k), so deep-and-structured negatives are abundant once you look below the cutoff. That is what filled the deep-band negative floor with 125 genuine near-misses rather than 10 featureless masked frames.
- **Nominal splits are not realized splits.** The roster's nominal 70/30 realized as **17.2% eval — 25 of 145 atoms.** Every eval-side number in this arc rests on ~25 atoms. Check the realized share before quoting a CI.
- **Measure the axis you intend to conclude about.** The 487 answered "what is auto-framed yield at one scale per atom." It was very nearly read as "do minibrot neighbourhoods contain q4s." State the population and the axis in the same breath as the number.
- **A control arm must differ from the treatment in one thing.** The interior batch's low-interior control conflated "no interior" with "empty space," because a 0%-interior window is either a dendrite-rich field or dead space. Where a clean control is impossible, run two comparisons that fail in *opposite* directions and report whether they agree.
- **A statistic that reverses sign when pooled is not disqualified — it is conditional.** See `coh_scale_drop` below.
- **Sourcing artifacts hit TAIL statistics, not BULK statistics.** A median over tens of thousands of swept positions shrugs off biased source atoms; a count of rare accepts is dominated by which atoms were drawn. Any q4 abundance number is a tail statistic — audit the sourcing before trusting it.
- **A cheap proxy's usefulness is partition-dependent, not global.** Never set one threshold from a pooled measurement.

---

## 8. Candidate window features from the 487

Selected on train, confirmed on eval, atom-bootstrapped, out of a board of 39. Both are **promising, not established** — max-of-39 on ~6 eval atoms.

**`int_perim_area`** — interior-boundary length per unit interior area; the dendrite-vs-body discriminator. Train AUC 0.652, eval 0.683, **+0.177 given degree**. Replicated out-of-sample on the interior band, but only at the dead-vs-not boundary, where a trivial ink measure beats it. Real but weak.

**`coh_scale_drop`** — orientation coherence lost as the analysis window grows; scrolls turn, filaments don't. **A sign-reversing Simpson case:** ρ = −0.183 pooled, but eval AUC **d4 0.802 CI[0.632, 0.871]** and **d5 0.871 CI[0.753, 0.972]**, both excluding chance — and in exactly the cells where G collapses to 0.510 / 0.479. Higher-degree fields are intrinsically more coherent (ρ = −0.566), which buries the feature in the pool. **Usable only conditioned on degree**, which is always known at draw time. Strongest positive result of the arc.

**Quote the eval column, not the train triple.** The per-degree numbers this feature is usually cited by — 0.70 / 0.76 / 0.73 across d3/d4/d5 — are **train**. Eval is **0.28 / 0.80 / 0.87**: d4 and d5 hold and are the strongest cells on the board, while **d3 has one positive against 32 negatives** (CI [0.065, 0.559]) and must be read as unresolved, not as a reversal. This matters because the feature was the max of a 13-feature × 3-degree board, so the train column is biased upward by selection and the eval column is the one that survives it — on six eval atoms per degree.

Both features are recomputed at draw time on every later batch and recorded without ever being selected on, so batches stay directly comparable. `[artifact: data/minibrot_roster/batch_v1/interior_features.jsonl — 487 rows × 13 crop features + the screen's own 15 + covariates]` `[cmd: uv run python -m tools.studies.interior_bakeoff {features,board,audit,questions}; tests tools/studies/test_interior_bakeoff.py]`

---

## 9. What was measured, batch by batch

| batch | n | what it is | usable as |
|---|---|---|---|
| `minibrot_roster_v2` | 487 | minibrot-neighbourhood, G-framed, screen-selected, one scale per atom | train-side |
| `interior_band_v1` | 80 | uniform-sampled high-interior | train-side hard negatives |
| `anchor_class4_v1` | 60 | cross-family class-4 anchor (52 rows are revisions) | anchor / cross-family bar |

All blind. Class-4 yield from the 487: **2, and both are windows of a single atom** (`d5_p11_029` — disjoint windows of one motif, reachable twice via d5 symmetry). That is **one independent example**, not two.

**The correction that matters most.** "Minibrot mining hasn't demonstrated value" is the wrong verdict and has been reached in error once. The 487 measured **auto-framed yield at one scale per atom**. Direct experience is that good q4s sit *within* many of those neighbourhoods, reachable by zooming in — so a minibrot's value is as a **marker of an interesting, high-density region**, not as a window that is itself shippable.

**Scale is the axis that was never varied.** Three results already pointed there and were under-read at the time: the 27 sub-cutoff crops that scored ≥3, the 25 of 105 straddling atoms, and the ICC leaving substantial within-atom spread.

---

## 10. Open, and deliberately not built

**The open question for the atom-selection function: what makes a great minibrot?** Degree is the strongest measured signal and carries the two objections in §5.

- **Hypothesis to test first: the atom's EMBEDDING, not its parameters.** Where an atom sits in its parent determines the decoration field around it. Two period-15 atoms in different regions look nothing alike, and `(degree, period, |A|)` cannot distinguish them. Measurable as neighbourhood statistics at 10–100× atom scale — that is, *outside* the crop window everything has so far been scored on.
- **Proposed instrument: time-boxed human exploration.** Stratified atom draw, fixed minutes per atom, record q4 yield. The eye is the ground truth and the exploration happens anyway; the fixed budget is what stops effort confounding yield. Blinding is impractical — degree is visible from symmetry — and that is accepted.

**Not built, deliberately: the min|z| atom-domain detector.** Complete enumeration is bounded by d^(n−1): fine at d=2 across the period range, but only to about n=7 at d=5. Past that, track `min|z_k|` and its argmin index in the backend sample loop, ride a second dumped channel, and feed `newton_nucleus` unchanged; field gating makes an extra accumulator ~free when unrequested. **Its output is SUPPLY, not a base rate.** Bonus: its false positives cluster near Misiurewicz points, giving one detector two useful outputs.

---

## 11. Retired — do not relitigate

- **A separate q4 network.** A net trained only on minibrot crops could never learn that q4-ness ≠ minibrot-ness, since every positive it ever saw would be a minibrot. q4 is class 4 on the existing scale.
- **Depth / period as a quality axis.** Measured −0.21 in the period-matched slice; the apparent signal was the screen's bias.
- **Interior MASS as an independent quality axis.** It is degree's shadow: +0.046 given degree.
- **"The q4 screen might be degree-bound."** Tested; it isn't.
- **The deep-floor negative-draw rule.** It did its job — the shortcut it guarded against does not exist in the labels.
- **A written class-4 rubric.** Depth and complexity are contributors and belong in the predictors, not in prose.
