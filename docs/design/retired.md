# Retired approaches — an append-only register

**Check here before proposing an approach.** It may have been tried, and the reason it was
dropped may still hold. This is an index, not an argument: where a design doc already owns
an item's evidence, the line points at it rather than restating it.

**Rules.**

- **Append only.** Entries are never deleted and never edited.
- **A reversal is a new dated `UN-RETIRED` entry**, appended below. The original line stays
  exactly as written — that is the whole point of the register.
- One item per line, with an optional one-clause reason.
- "Retired" is a statement about a decision that was made, not a proof that the idea is
  wrong. `measurement_practice.md` §1 has the standing caution: a retirement measured under
  range restriction is scoped to the range, not to the axis — which is how the first
  `UN-RETIRED` entry below came about.

---

## Retired

- Forced palette diversity
- Uniform-per-category targets
- Hard CLIP novelty thresholds
- Hue-histogram clustering
- `morph_v6` in the library
- Morph gate at discovery
- Cross-head absolute-score selection
- Categorical-gated morph-coverage
- "Singleton degeneracy eases at scale"
- Within-flavor v3-gvo-argmax palette pick
- Per-mode compute-weighting of the measure
- The 1280×720 pool render res
- Hook-spacing as the julia lever
- The skip log as a c-diversity pool
- Parent c-diversity as the julia lever — real cause: intra-`c` z-plane self-saturation
  (`julia_parent_sourcing_probe.md`)
- The H3 julia ↔ M correspondence
- The q4-frontier-budget-fraction idea
- "The mining head scores strange near zero"
- τ_h as an unopened lever
- "The q4 screen might be degree-bound" — tested; it isn't (`minibrot_sourcing.md` §11)
- Recovering anything from the old `out/` tree — the class guaranteed deletion in advance
  (`storage_classes.md` rules 2–3)
- A SEPARATE q4 network — q4 is class 4 on the 1–4 label scale, not a second model
  (`minibrot_sourcing.md` §11)
- The deep-floor negative-draw rule — the shortcut it guarded against is not in the labels
  (`minibrot_sourcing.md` §11)
- Interior MASS as an independent quality axis — degree's shadow, +0.046 given degree
  (`minibrot_sourcing.md` §11)
- The interior-clause question — neither arm can convict or clear it (`minibrot_sourcing.md` §4)
- The phoenix surrogate seed loop — a METHOD, never a TYPE
- "The sampler pool is finite"
- A class-4-specific eval requirement
- A written class-4 rubric — depth and complexity belong in the predictors, not in prose
  (`minibrot_sourcing.md` §11)
- Append-don't-rebuild for a manifest
- Recovering the v4–v7 augmentation caches — the chain, not the producers, was the blocker
  (`storage_classes.md`)
- A UNIFORM `t_good` objective across partitions
- "Palette slots are cheaper than geometry slots"
- `cycles_spanned` / `falloff_extent` / `radial_rings_p90` as quality measures
  (`orbital_field_metrics.md` §5)
- Complete low-n enumeration as a minibrot source
- First-principles minibrot enumeration as a source at all (`minibrot_maneuvers.md` §1)
- The period-band-stratified roster — no quality parameter
- The 24× / clamp-67000 scoring cap policy — a fitted proposal that was never adopted and
  became a stated property of the system anyway (`auto_maxiter.md`, "Where it lives")
- A client-side pre-commit hook — replaced by a suite assertion
  (`verification_practice.md` §4)
- `selection_triage.json` regeneration
- Calibration exemplars beside the labeling rig
- Tile-mean band coverage as the view screen's coverage term — blind to *where* the dead
  area is, so a solid black disc plus a solid flat gradient scores like structure spread
  evenly (`orbital_field_metrics.md` §11)
- A veto threshold expressed as a multiple of the references' `interior_fraction` — both
  references measure ~0 interior, so any multiple of them vetoes ~70% of the population
  (`orbital_field_metrics.md` §11)
- The unbounded `sqrt(range × rings)` richness term — an unbounded factor inside an argmax
  is the tail the argmax finds; winsorized at 2× the strongest reference in v3
  (`orbital_field_metrics.md` §11.6)
- Log-compressing the richness term instead of winsorizing — measured, and at the seam
  window it still returns ~6.3× the cap, so the pathology survives at ~12× the population
  (`orbital_field_metrics.md` §11.6)
- An unconstrained framing argmax for a maneuver-anchored candidate — "find the richest
  window near here" is not the sweep's contract and is content drift dressed as framing
  (`orbital_field_metrics.md` §11.6)

## Un-retired

- **2026-07-31 — the 24× / clamp-67000 scoring cap policy, for ONE narrow use.** Adopted as
  the cap policy of the maneuver richness screen (`tools/atlas/maneuver_screen.py`
  `SCREEN_MAXITER_POLICY`, token `mi12000k0.3c4800-67000`), and for nothing else. The
  original retirement stands as written and its reason still holds where it was aimed: it
  was a *fitted* proposal that became a stated property of the system without ever being
  adopted, and it is still not a **scoring** policy — production scoring is untouched. What
  changed is that a screen now needs a cap that does not move when the production cap moves,
  and at 64×36 the extra iterations cost nothing. Scope: screening only, stamped on every
  record, and pairwise-disjoint from both the legacy and the live tokens so
  `field_metrics.require_one_policy` raises rather than pooling across them.
  `[code: docs/design/minibrot_maneuvers.md §3.1;
  tools/atlas/test_maneuver_screen.py::test_the_screen_policy_is_24x_the_legacy_envelope_until_the_clamp_binds]`

- **2026-08-01 — the 24× / clamp-67000 screening cap policy, scope EXTENDED by one module.**
  The 2026-07-31 entry above adopted it for `tools/atlas/maneuver_screen.py` "and for nothing
  else"; `tools/atlas/view_screen.py` now runs under the same policy, reading
  `maneuver_screen.{screen_maxiter, screen_policy_token}` rather than restating it. The
  extension is deliberate and is the whole point: a view score and an atom score describe the
  same population at two frames, and giving them different cap tokens would make
  `require_one_policy` refuse a comparison that is legitimate while permitting nothing new.
  Everything else in the original scope stands — screening only, stamped on every record,
  pairwise-disjoint from the legacy and live tokens, production scoring untouched.
  `[code: tools/atlas/view_screen.py::measure_view;
  tools/atlas/test_view_screen.py::test_every_view_measure_carries_the_screen_cap_policy]`

- **2026-08-01 — "interior mass as a quality axis", re-scoped to a bounded SIZE BAND and
  to nothing else.** The original retirement (`minibrot_sourcing.md` §11: +0.046 given
  degree) stands as written and its reason still holds where it was aimed — interior mass
  is not a monotone quality signal and nothing here treats it as one. What v3 adds is a
  *band*: `view_screen.size_factor` is exactly 1.0 from interior 0 through 0.12, so a frame
  at 0.10 scores identically to the same frame at 0.00 and "less interior is better" is
  never asserted; above the edge it declines to the veto's own behaviour at the veto
  threshold. The claim is only that past some share the subject dominates the frame, which
  is a composition statement about one picture and not a quality axis over a population.
  Scope: the view-level composite's sort key, and nothing else — no discovery module reads
  it, and `--maneuver-range-prior` is untouched. The distinction is asserted, not asserted
  in prose: a test pins the flat interval and would go red if the band ever became monotone.
  `[code: tools/atlas/view_screen.py::size_factor;
  tools/atlas/test_view_screen.py::test_the_size_band_is_a_band_and_not_an_interior_quality_axis]`

- **2026-07 — "depth/period as a quality axis."** Retired on a roster spanning period ~3–15,
  where it measured +0.06 pooled and −0.21 inside the period-matched eval slice. Across
  periods 2–74 `period` correlates **+0.87** with `radial_rings`. The retirement was scoped
  to a range, not to the axis. Evidence and the sharp-end caveat (top-100 overlap between a
  `rings` sort and a period sort is only 12%) are owned by `orbital_field_metrics.md` §6;
  the original retirement stands as written in `minibrot_sourcing.md` §11.

- **2026-08-01 — the v4 "in-tile participation" refinement of `band_coverage`, RETIRED
  BEFORE ADOPTION, together with the two families measured after it.** The proposal was
  that a tile calling itself participating merely because it SPANS a colour cycle can be
  satisfied by one lazy band, and that requiring band *structure* inside the tile would
  demote the "field of blue" (`snap k16 d5 p16`) without touching the references. Three
  structure statistics were run against the extended v4 gate — `cross` (band-boundary
  crossings per in-tile pixel step, 9 floors), `tv` (mean |Δcycles| per step, 6) and
  `bands` (distinct `floor(cycles)` per tile, 5) — under a selection rule written down
  first: the least demanding `cross` floor satisfying G1–G7. **None of the 21 passed**, and
  the reason is that the premise is false on the field: 42% of that tile's tiles ALREADY
  fail v3 participation, and it reaches p83.5 because it is above median on BOTH factors
  (coverage p64.6, richness p69.4), not because either is over-read. The crossing clause
  moves it the WRONG way (p83.5 → p84.4 at floor 0.45) while re-promoting the k4 frames the
  v3 size band exists to demote (p77.8 → p84.3).

  Two further families were then measured and are retired with it. **Region pooling** (8
  `BX×BY qQ` variants, v3's tile indicator unchanged) is the family that actually addresses
  the measured mechanism — the dead area is one contiguous diagonal band that the 4×3 region
  grid cannot isolate — and it separates the two anchors ~5× where the participation clause
  separates them ~1.2×; its best variant `16×3q25` still cannot get the field of blue under
  the bar, missing G7 at **p81.0** against 80.0. **Coverage exponents** (`cov^{1…4}`, 6) buy
  p83.5 → p80.8 at `cov^4` but break G4 from `cov^1.5` onward, and the 6 exponent×crossing
  combinations fail both clauses at once. G4 and G7 are 5.7 percentile points apart and
  every coverage-side lever moves them together — that, not any one formulation, is the
  result.

  Scope of the retirement: the participation INDICATOR and these two re-weightings, as
  candidates for the live sort key. `composite_v4`, `tile_structure`, `pooling_grid` and
  `coverage_grid` all stay live in code — a record whose producer no longer exists cannot be
  checked — and `composite_v3` remains the live sort key. What the iteration did leave behind
  and is NOT retired is the field cache: the next per-tile statistic is a numpy pass rather
  than a 17-minute engine pass over the population.
  `[measured: 41 formulations, 16,440 candidates, 2026-08-01;
  data/atlas/view_screen_gate.json §v4; orbital_field_metrics.md §11.7]`
  `[code: tools/atlas/view_screen.py::{tile_structure,composite_v4,pooling_grid};
  tools/atlas/test_view_screen.py::{test_the_v4_gate_block_is_pinned_to_its_record,
  test_the_live_sort_key_is_composite_v3}]`

- **2026-08-02 — exemplar similarity as an ORDERING / STEERING feature, retired on two null
  pre-registered reads.** The hypothesis was "closer to the tiles Matt liked = better": each
  of the supply crawl's 7,063 candidates carried a cosine to an 8-exemplar set on the
  deterministic colour-mapped 64×36 field, and a 60-row mini-chunk was drawn top-by-similarity
  to test it against the stratified chunks. Both reads were written down before the labels
  came back and both came back null.

  **Read (a) — does the column carry a coefficient?** In the `view_fit_v1` fit (580 rows, 149
  positives, GroupKFold on the walk root) the standardized coefficients are `sim_max` +0.118
  and `sim_mean` −0.086 against a largest-coefficient magnitude of 3.58, and their
  group-bootstrap 95% CIs span zero in both directions: **[−0.369, +0.706]** and **[−0.788,
  +0.417]**. Dropping both columns does not cost anything — OOF AP **0.7158 dropped against
  0.7118 with them in**, i.e. the drop is *above* the full model, paired-bootstrap
  ΔAP −0.0044 **[−0.0155, +0.0056]**.

  **Read (b) — is the held-out leg's rate anything but its other features?** The model was
  fitted on the stratified legs only and asked to predict the exemplar leg it had never seen.
  Predicted `label >= 2` rate **0.6465**, realized **0.6333** (Wilson **[0.507, 0.744]**),
  expected 38.8 against 38 realized on 60 rows. The leg's yield is exactly what its ordinary
  screen features already say it should be; similarity adds nothing on top.

  **And the one number that says why the mini-chunk could never have settled it on its own:**
  `Spearman(composite_v3, exemplar_sim_max) = +0.442` over all 7,063, with the top-60-by-
  similarity sitting at median composite **4.23** against the population's **0.58** — 55 of 60
  rows in the top two composite bins. "Closer to the exemplars = better" is not separable from
  "higher composite = better" out of that chunk; what makes the question answerable at all is
  the stratified chunks spanning every bin, which is where both reads above were taken.

  **Scope, and it is narrower than "the feature is useless".** Retired as an axis anything may
  ORDER or STEER on: it is out of `view_fit_v1.1` (`FEATURES_V11`), which is the score the
  label-seeded harvest's queue is ordered by, and the harvest does not compute it at all. Raw
  similarity remains a perfectly good RECORDABLE feature if a future question wants it —
  `tools/atlas/exemplar_similarity.py` stays live and the 730 supply-crawl rows keep their
  recorded values, so the reads above stay re-derivable. Also unretired-by-omission: the
  substrate itself (a deterministic colour map of the cached field) is close to blind to
  palette-level qualities by construction, so this is a null on COMPOSITION similarity and
  says nothing about a similarity measured on rendered crops.
  `[measured: data/atlas/view_fit_v1.json §readout.exemplar_read_a / exemplar_read_b;
  730 labeled rows, 2026-08-02]`
  `[code: tools/atlas/view_fit.py::{FEATURES_V11,EXEMPLAR_FEATURES};
  tools/atlas/test_label_seeded_harvest.py::test_v11_drops_the_family_and_the_exemplar_columns]`

- **2026-08-02 — `tools/coevo/` (the guard-OFF v6-gap co-evolution round), deleted.** Two
  modules — `coevo_round.py` (the driver) and `analyze_round.py` (its readout) — that ran
  one guard-OFF diagnostic round into `data/discovery/gather/` and were never run again.
  An all-channels liveness sweep (imports, `sys.path` inserts, `spec_from_file_location`
  path loads, subprocess argv, launcher scripts) found **no consumer of either module**;
  the only references left in the tree were prose. Deleted rather than archived, so the
  cost is stated plainly: `analyze_round.py` was **B-live** in `tools/README.md`'s sense —
  it produced a committed artifact once — and its deletion removes the only committed
  description of how that round's gather ledger was made. The artifact stays; the recipe
  does not. `[code: removed — tools/coevo/{coevo_round,analyze_round}.py]`

- **2026-08-03 — the near-minibrot 1×/4×/16× distance LADDER as a three-rung emitter.** The
  ladder was the experiment that answered "at what multiple of an atom's own radius should a
  julia `c` be drawn?", and it answered it: **nowhere in particular**. Yields are flat across
  the rungs (labelled ≥3: 68.0 / 63.5 / 68.0%, one-per-cluster 61.8 / 65.3 / 66.7%, every
  pair's Wilson interval overlapping) and the three rungs of one atom are **one look** —
  same-atom different-rung pairs sit at median cos **0.9825** with **74.1%** at or above the
  0.974 near-dup cut. So the ladder bought ~1 distinct look per atom for 3× the label cost.
  A paired render-cost measurement was taken to break the tie the prompt expected cost to
  break, and **cost is flat too** (3.9% spread, 24 atoms × 3 rungs interleaved), so the single
  rung was chosen on the one column that separates — one-per-cluster class-4, 8.6% / 4.0% /
  3.7% — giving **rung 1**.
  **Scope: the LADDER, not the channel.** Near-minibrot sourcing is very much alive and is one
  of the four `julia:mandelbrot` channels; it now emits one `c` per nucleus. The rung constant
  and the flat-yield table stay in code (`supply_routing.LADDER_YIELD`) so a later question
  about distance is a read rather than a re-run.
  `[measured: data/label_corpus/batches/2026-08-03_q4_near_minibrot_v1, 290 labelled rows /
  103 atoms; data/atlas/near_minibrot_rung_v2.json, 2026-08-03]`
  `[code: tools/atlas/supply_routing.py::rung_choice; tools/atlas/near_minibrot_rung.py]`

- **2026-08-03 — the unscreened ∂M_d-shell draw for native multibrot3/4/5, priced at zero.**
  The score-unconditioned leg of the q4 sitting drew from the degree-`d` boundary shell at
  ε = 0.02 with no screen, and **0 of 144 rows reached ≥2** (48 per family, Wilson [0, 7.4%]
  each; no class-4 anywhere in the leg). It is not built in harvest v2.
  **Scope: a verdict on the DRAW, not a ceiling on the families.** The same three partitions
  reach ≥3 at 55.0% through triggered maneuvers against a partition-matched 25.5% fresh, so
  native multibrot supply is routed through seeds + triggered maneuvers instead. Named in
  `supply_routing.RETIRED_CHANNELS` rather than omitted, because a channel that is absent and
  a channel that measured zero read identically in a config.
  `[measured: data/label_corpus/batches/2026-08-03_q4_uniform_eval_v1, 2026-08-03]`
  `[code: tools/atlas/supply_routing.py::RETIRED_CHANNELS]`

- **2026-08-04 — the DURABLE classification of the q4 stage-1 fields, and the 336 MB of
  git-LFS it bought.** Tracked as durable that morning and **deleted by decision** the same
  day (Matt), under usefulness-before-recoverability: a superseded screen's fit input is not
  worth 336 MB of tracked bytes even if it were irreplaceable. What retires here is the
  **classification**, not the fields — and the classification was **wrong on its own terms**,
  which is the reusable part. The contract test re-dumped a field through the **`beautiful`
  kernel** (default bailout 2^16), got a constant **+3.4712** offset on every escaped sample,
  and read that as irreproducibility. But nothing in the path uses that writer: these fields
  are dumped by **`--dump-field-source f64`** (`q4_multibrot_transfer._dump_field`), and that
  path reproduces them **byte-identically** — sha256 equal on 4/4 spot-checked files, NaN and
  interior mask identical, **100%** of escaped samples exactly equal, and the re-dump sidecar
  carries the same `bailout_b` 1e6 the stored ones do. Downstream the question could not have
  gone the other way either: `LF.featurize` percentile-stretches every crop by its own
  `lo`/`hi`, so a constant offset cancels **exactly** — worst feature difference **0.0** over
  the real labeled windows, zero `_v2_drop` disagreements. **The rule: a reproducibility test
  must re-run the writer the artifact actually came from, not a plausible neighbour.** This
  one measured a kernel no caller invokes and priced a regenerable set as unrecoverable.
  **Correcting the record on recoverability, twice.** The prompt authorizing this deletion
  called the set "partially unreconstructable" and its selection "already unrecoverable";
  both are false. The bytes rebuild in **~60 s** (~1.7 s × 33) via
  `q4_multibrot_transfer.py corpus-fields`, and the **selection** is derived by `HT.mb_info()`
  from the tracked window store `data/q4_window_corpus/batches/` with **33/33** coverage. The
  deletion is right for the reason Matt gave — the screen is superseded and the bytes are not
  worth tracking — and it costs nothing recoverable, so it did not need the stronger claim.
  `LS.FIELDS` is now `paths.bulk()`, every reader funnels through `LS._require_field` (which
  raises naming the rebuild command), and a **fourth** hardcoded copy of the path in
  `q4_richness_grid.py` — missed by the original three-reader collapse, and dangling since —
  was collapsed onto it. Local LFS objects deliberately **not** pruned. Holding copy of the
  deleted bytes at `C:\Code\fractal-maker-holding\q4_stage1_fields` (66 files, 351,581,091 B,
  count + sha256 spot-checked), expected lifetime one to two checkpoints — the rebuild
  command, not that copy, is the recovery path.
  `[measured: 4/4 fields sha256-identical vs `--dump-field-source f64`; featurize/_v2_drop
  parity over 8 windows of mb00_p04; 2026-08-04]`
  `[code: tools/studies/q4_stage1_labelset.py::{FIELDS,_require_field}; .gitattributes;
  .gitignore; tools/audit/{size_guard,durability_map}.py; tests/test_large_tracked_blobs.py]`

- **2026-08-04 — `--scheduler` retired as the PRODUCTION allocator. `--pop-quota` is the
  standing one (Matt's decision).** The retirement is **policy, not deletion**: the flag still
  parses, the deficit scheduler still runs, and `tools/atlas/steered_frontier.py` is unchanged
  by this entry. What is retired is its standing as the allocator a production discovery run
  reaches for by default.
  **Basis, stated exactly.** Two things, and neither is a comparison. (1) `--pop-quota` is the
  stated allocation design — demand denominated in human labels (n4 + 0.1·n3 through the
  amendment overlay), which is the currency the corpus is actually short of. (2) The 2026-08-04
  mid-run **mechanism** read showed it steering correctly over the set it could serve: batch-
  weighted **L1 0.041** against effective intent renormalized over the SERVABLE partitions,
  tighter than the proving run's 0.093.
  **What this is NOT based on.** The pre-registered scheduler-vs-pop-quota comparison
  (`data/discovery/allocator_prereg_v1.json`) produced **no result**: amendment 1, same date,
  voids it as an allocator read — both arms were supply-bound rather than allocation-bound
  (globally-blind `ROOT_LOW_WATER`, inverted price sampling, a julia hook closed by a pool/hook
  spacing conflict), so the estimand could not see the allocator. Arm B's 92 admissions against
  arm A's 336 are **not** the reason and are not comparable; arm B was stopped on its sentinel
  at 143 of 510 active minutes once the confound was established. This decision is a design
  call taken on the mechanism, and it is written down that way so a later reader does not
  reconstruct a verdict the record explicitly withdrew.
  **Re-use needs a new dated `UN-RETIRED` entry**, per this file's rules — including a re-run
  of the comparison, which would first need the three supply-loop fixes named in the amendment.
  `[decision: Matt, 2026-08-04]`
  `[measured: data/discovery/allocator_prereg_v1_mechanism_read_20260804.md — mechanism read at
  b381 / 129.5 active min, 2026-08-04]`
  `[code: data/discovery/allocator_prereg_v1.json amendment 1;
  tools/atlas/compare_allocator_runs.py::voided_windows (the void binds the reader);
  tools/atlas/steered_frontier.py `--scheduler` — unchanged, deliberately]`

- **2026-08-04 — the hand-placed emission target measure (`data/emission/target_measure.json`)
  and the config machinery that read it.** The emission per-cell target now DERIVES from the
  canonical release-mix ratio table (`tools/scoring/release_mix.RATIO`) at intake time:
  `weight_p = share_p / n_feasible_cells_p`, re-solved against the live feasible cells
  (`cells.TargetMeasure.from_partition_shares`). The discovery order book
  (`deficit_scheduler.target_shares`) reads the same derived shares, so the two stages cannot
  hold two policies about the same partitions again.
  **Deleted with the file, not kept as dead code:** `TargetMeasure.{from_config,
  resolve_source_tags,solve_target_shares,weight_overrides}` and
  `deficit_scheduler.{load_target,project_type_marginals}`. They existed only to read that
  file's nine literal multipliers, its `source_tag` indirection and its one `target_share`
  override; nothing else called them. `library_intake_2`'s own `CLASSIC_RELEASE_SHARE = 0.02`
  went with them (a second copy of the same policy) and its `--target-measure` /
  `--scheduler-target` flags are gone.
  **Basis.** The file and the ratio table disagreed about the same partitions — mandelbrot
  9.0% vs 22.7% intended, multibrot4 1.9% vs 7.6%, julia:multibrot4 18.9% vs 7.6% (basis: the
  9 `weight_overrides` projected per-type vs `release_mix.shares()`, 2026-08-04). The two
  properties the machinery bought are kept and are now structural for EVERY partition rather
  than solved for one: absolute share (a partition's cells hold exactly its share) and
  denominator-invariance (growing a partition's morph-cluster count spreads the same share
  over more cells instead of enlarging it).
  **The file was deleted rather than made a derived, regenerable artifact** because what would
  have been left in it after the weights moved out is `attempt_cap` and `softmax_temp` — two
  mechanism knobs that are already code defaults. Absence fails loud by there being no reader:
  `first_release_readout` had an `if MEASURE.exists() else {}` that turned an absent measure
  into a silently UNIFORM target, and `tools/scoring/test_release_mix_one_source.py` scans the
  tree for any reader coming back.
  `[decision: prompts/emission_releasemix_prompt.md §B, Matt, 2026-08-04]`
  `[code: tools/emission/cells.py::TargetMeasure.from_partition_shares;
  tools/atlas/deficit_scheduler.py::target_shares;
  tools/scoring/test_release_mix_one_source.py]`

- **2026-08-04 — the `c1__` prefixed-ledger-COPY scheme for run-scoped id collisions.**
  Campaign1 and campaign2 mint `st_<fam>_<arm>_<seq>` per campaign and reuse 11 of them for
  different locations across the seven intake ledgers, which aborted the emission union.
  `stage_first_release.py` handled it by writing id-prefixed COPIES of the campaign1 ledgers —
  to `scratch/`, where they were deleted, taking the only reachable union with them.
  Replaced by namespacing row identity BY LEDGER at the reader
  (`descriptor.load_union_admitted`): no ledger row is rewritten, no copy is minted, and
  deduplication moves onto location identity (`descriptor.loc_key`, unchanged), which is the
  axis that can actually carry it. The union went from unreachable to **700** admitted rows.
  `[code: tools/emission/descriptor.py::{ledger_namespace,load_union_admitted};
  tools/emission/test_intake_union.py; tools/emission/stage_first_release.py — kept, marked
  SUPERSEDED, deliberately not deleted]`

- **2026-08-04 — the machine BADNESS floor on floor-admit sources (`descriptor.FLOOR_PNOTBAD`,
  `p_notbad >= 0.5`).** A floor-admit source (`q4_harvest`, `human_q3plus`) is one whose
  selection signal is ORTHOGONAL to the quality head — the q4 goodness field, or a human 3/4
  label taken with no decode consulted — so it bypasses the head's q3 gate. It did not bypass
  the head's *badness* verdict, on the reading that "reject clear junk" was a weaker claim
  than "judge quality". It is not weaker; it is the same claim at a lower threshold, made by
  the same head, and on a `human_q3plus` row it resolved a head-vs-Matt disagreement in the
  head's favour, silently, at intake.
  **Deleted, not zeroed.** A `0.0` floor is still a floor: it reads as a policy somebody chose
  and gets re-tuned by whoever finds it next. `admit_quality` now returns True for a
  floor-admit row outright; `guard_pass ∧ distinct ∧ current-decode` still apply to every
  source alike, which is the difference between bypassing the quality verdict and bypassing
  the intake.
  **Basis (the second reason, and the general one).** `0.5` was set on the **v7** `p_notbad`
  scale and was still being read under **v10**. On the `q4_harvest` ledger's 108 guard-passing
  rows: the v7-era floor admitted **75**; the same `0.5` against the v10 rescore admitted
  **57**. An 18-row move in what the intake accepts, with no decision taken about it. That is
  the standing hazard of an unstamped cut, and it is why every stage-2 cut that REMAINS now
  lives in `tools/emission/floors.py` carrying the head version it was set against and
  refusing to gate when the live pin disagrees.
  **What it moved:** `q4_harvest` 57 → **108** admitted; the seven-ledger stage-2 union
  700 → **751**. The other six ledgers admit on the q3 gate and did not move.
  `[decision: prompts/emission_floors_prompt.md §B, Matt, 2026-08-04]`
  `[code: tools/emission/descriptor.py::admit_quality; tools/emission/floors.py;
  tools/emission/test_intake_fail_closed.py; tools/emission/test_intake_union.py;
  docs/design/q4_harvest_emission.md]`

- **2026-08-04 — `DEDUP_K = 1.5 x max(fw)` as the precanon / q3-cloud coordinate dedup.**
  Replaced by the calibrated `0.25 x min(fw)` (`production_seeder.{DEDUP_K,DEDUP_SCALE}`).
  On the 135 pairs Matt judged, the retired rule merged ~**8 pairs he would keep per 1 it
  merged correctly** — `max(fw)` is set by the wider frame in 28/28 top-tier cases, so what
  it deleted was the deep zoom inside a wide outcome's disc, i.e. the output guided descent
  exists to produce. At the same verdicts, K=0.25 on the min scale merges 8 SAME with **0**
  false merges against 1.5's 9 SAME / **35** false. The two constants were calibrated
  together and neither transfers to the other scale.
  Record (sweep, boundary, the n=14 SAME ceiling, the unexamined outliers, and the named
  escape valve — K may move inside [0.25, 0.376] as a priced decision, never above without a
  new calibration): `data/atlas/precanon_calibration/adoption.json`, beside the verdicts.
  `1.5 x max` is NOT deleted: it stays reachable as `RETIRED_DEDUP_K`/`RETIRED_DEDUP_SCALE`
  and as explicit `k=`/`scale=` arguments, because the diagnostics that replay records made
  under it must keep replaying it.
  `[decision: prompts/precanon_adopt_calibrated_prompt.md, Matt, 2026-08-04]`
  `[code: tools/atlas/production_seeder.py::{DEDUP_K,DEDUP_SCALE,dedup_radius};
  tools/atlas/test_production_seeder.py::{test_production_dedup_rule_is_the_calibrated_min_quarter,
  test_production_dedup_verdicts_move_under_either_revert,
  test_admission_call_sites_resolve_the_live_rule}; docs/design/morphology_dedup.md §6]`

- **2026-08-04 — `1.5 x max(fw)` as the EMISSION-side c-plane fractal identity.** The entry
  above retired the rule at the precanon gate; it stayed live one layer down, in
  `emission_selector.same_fractal`'s c-plane branch, as a private `DEDUP_K = 1.5`. Its safety
  argument was that such pairs were already deduped upstream — which the adoption falsified in
  the same commit that made it, since the seeder now KEEPS the deep-zoom-inside-a-wide-outcome
  pair. What retires here is the emission layer's **copy**, not a separately-calibrated
  constant: the branch now CALLS `production_seeder.near_dup` under the owner's live
  `(DEDUP_K, DEDUP_SCALE)`, read at call time, so the next recalibration cannot leave it
  behind. Nothing was recalibrated by this entry; the same 135 verdicts are the whole basis.
  **NOT retired:** the emission-side *z-plane viewport* rule (`d < K*min(fw)` AND
  `max(fw) <= ZOOM_RATIO*min(fw)`), which is a different rule on a different population and has
  never been calibrated. It keeps K = 1.5 and is renamed `VIEWPORT_K` so the coordinate gate's
  name has one owner.
  `[decision: prompts/emission_smoke_prompt.md §A, 2026-08-04]`
  `[code: tools/wallpaper/emission_selector.py::{same_place_c_plane,VIEWPORT_K};
  tools/atlas/test_production_seeder.py::{test_no_module_declares_a_coordinate_gate_constant_of_its_own,
  test_the_emission_selector_c_plane_branch_resolves_the_owners_pair};
  tools/wallpaper/test_emission_selector.py; docs/design/morphology_dedup.md §6]`
