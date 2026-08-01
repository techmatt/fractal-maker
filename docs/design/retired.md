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
