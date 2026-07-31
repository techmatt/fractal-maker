# docs/design — canonical design layer

The curated, checked-in design docs. Each holds durable, design-informing knowledge
distilled from the (now largely deleted) `docs/findings` research ledger — the timeless
core, stripped of run logs and dated readouts. `CLAUDE.md` remains the top-level map of
the render core and pipeline; these docs go a level deeper on specific decisions.

| doc | governs |
|---|---|
| [phoenix_seed_sampler_spec.md](phoenix_seed_sampler_spec.md) | Phoenix seed proposal: the closed-form stability skeleton, the `(c,p,z₋₁)` axes, the corrected real-axis-reflection symmetry (§7.1), the fertility-aware surrogate loop, and the settled human-label verdicts (§8). |
| [deep_zoom_sourcing.md](deep_zoom_sourcing.md) | **Deep regime (PARKED).** Deep-center sourcing (Newton nucleus/Misiurewicz → decimal-string centers), depth-band locality, the f64-bound walker, the standing perturbation-engine gaps, and why deep-Mandelbrot curation needs its own objective. |
| [minibrot_sourcing.md](minibrot_sourcing.md) | **Moderate depth, inside f64.** The minibrot-atom arc: the atom roster, the stage-1 screen `G` and its measured worth, the pre-filter and its interior catch-22, the `A` feasibility cut, and the atom/window-level signals from the labeled evidence. |
| [julia_c_sourcing.md](julia_c_sourcing.md) | The ~2% Julia "corner" exemplar class and the three-stage `c`-selection screen; the interior-lake determinant. |
| [aesthetic_scoring.md](aesthetic_scoring.md) | How to read the signal: occupancy/busy-ness is anti-quality (report-only), and `p_good` is a badness filter, not a goodness ranker. |
| [classifier_retrain_protocol.md](classifier_retrain_protocol.md) | Append-don't-rebuild manifests, forced eval/train split rules, pre-registered paired-eval bar, and per-version `t_good` re-derivation. |
| [morphology_dedup.md](morphology_dedup.md) | Visual dedup: coordinate dedup is not visual dedup, the cone-compressed CLIP space, grayscale-CLIP descriptor choice, and dedup-key identity rules. |
| [render_coloring_surface.md](render_coloring_surface.md) | The two coloring algorithms, three palette namespaces, the silent density fork, the open corpus-coloring hazard (P4), why the OKLab ports must not be merged, and (§6) the Ultra Fractal originals four `render_modes` fields were ported against — incl. the three `DirectShape` names that collide with UF's while computing something different. |
| [auto_maxiter.md](auto_maxiter.md) | The depth-aware iteration cap: the closed form and its two load-bearing sites, the 32-atom measurement that showed base 500 was ×8 too low, the adopted base 4000 / clamp 67000, the **median-clean-not-clean** residual (tail runs to ×24), the aug cache's separate flat-8000 cap, and the four things a cap change moves. |
| [discovery_pipeline.md](discovery_pipeline.md) | The guided-descend walk + reward split, the freshness-prior/dive incompatibility, and the distinct-look deficit scheduler (incl. julia routing). |
| [storage_classes.md](storage_classes.md) | The durability **contract**: the four storage classes, usefulness-before-cost, why a missing `scratch/` artifact is not data loss, `scratch/` as a liveness rule (not just a rebuild-cost one), reproducible-on-demand as the KEEP test, derive-in-code/freeze-in-records, and the no-bulk-in-tree rule (traversal cost, not repo size). |
| [artifacts_resolver.md](artifacts_resolver.md) | The storage **mechanism**, named for `tools/corpus/artifacts.py`: the `ARTIFACTS_ROOT` resolver and its seam, relocation classes matched by pattern + the reappearance tripwire, the size-guard registry (dispositions, prefix-vs-file granularity as a trade, `forward` declarations and the end of the standing stale warning), LFS + exact-path `.gitignore` negation + why verification runs `--no-index`, the derived canary set and its non-vacuity pairing, and the small-tree / docs-are-source guards. |
| [orbital_field_metrics.md](orbital_field_metrics.md) | The ring measures `radial_rings` / `radial_range` (`tools/orbital/`): what they compute, the validation record incl. the failures (`cycles_spanned`, `falloff_extent`, p90), the span-vs-oscillation two-axis reading, the screening/validation resolution split, the cap-provenance axis, and what the instrument is blind to. |
| [label_corpus_relocation.md](label_corpus_relocation.md) | Relocating the label corpus's `crops/`+`vivid/` bulk (3,822 files, ~72% of the working tree) out of tree behind `artifacts.resolve`: why the scope is crops/vivid only (labels stay tracked in-tree), the silent-zero hazard across 34 construction sites (2 load-bearing readers), the seam, and the staged move + before/after `(image_id,score)` gate. |
| [minibrot_maneuvers.md](minibrot_maneuvers.md) | Minibrot moves as candidate MOVES inside a descent (`tools/atlas/minibrot_maneuvers.py`): the two operators (`snap_to_nucleus(k)` / `lateral_to_sibling`) and why five collapse to two, the superattracting-argmin correction that makes the atom-domain probe work, unavailability as the normal case, the reserved frontier FLOOR (of available) vs the probability used only as a cost governor, and the provenance a later read needs. |

**Two sourcing docs, one boundary — pick by depth regime.** `deep_zoom_sourcing.md` is
the **deep, beyond-f64 tier** (perturbation, ∂M-tracking, deep-center production) and is
**parked**; `minibrot_sourcing.md` is the **moderate-depth minibrot-atom arc that runs
entirely inside f64** (roster, screen, pre-filter, labeled evidence). They touch at only
two places — the atom-size law `size ≡ 1/|A|` and the fitted-objective status — and each
doc's boundary note names those explicitly. If you want roster / screen / pre-filter /
window-label content, read `minibrot_sourcing.md`; if you want precision, the
perturbation tier or deep-center production, read `deep_zoom_sourcing.md`.

**The table above indexes 15 of the 33 docs here** (measured 2026-07-31,
`grep -c '^| \[' README.md` vs `ls docs/design/*.md`). The other 18 fall in two groups, and
listing them is the point — an unindexed doc is invisible, which is how one rots unnoticed.

**Dated readouts** (a run's numbers, not a timeless rule) also live here — there is nowhere
else. They are named for the run and carry their own date: `interior_feature_bakeoff.md`,
`interior_band_batch_v1.md`, `minibrot_roster_v2_{pilot,readout}.md`,
`minibrot_label_batch_v2.md`, `nucleus_seeding_and_atom_A.md`,
`closeout_pre_distillation_2026-07-25.md`, `migration_to_fractal_maker.md`,
`walk_era_julia_resolution_audit.md`. **Note the tension with "What belongs here" below:**
this is a declared carve-out for exactly the category the test excludes, and it covers ~9 of
33 docs. Whether the carve-out should exist is an open decision, not a settled rule.

**Indexed by neither the table nor that list** — nine docs, recorded here so they are at
least findable: `atom_instrument.md` (the `A` instrument: size / orientation / required
precision / f64-wall predictor — owned by `deep_center_finder.atom_instrument`),
`label_rubric.md` (the 1–4 labeling rubric, cited from the corpus batch builders),
`deferred_recalibration.md` (four recalibrations designed but deliberately unbuilt, with
their unpark triggers), `v8_training.md` (the v8 train-split population, read off
`data/v8/manifest.jsonl`), `beautiful_perf_report.md` (why the `beautiful` field dump was
~35× the `f64` twin, and the field-gate that fixed it), `q4_harvest_emission.md`,
`q4_multibrot_transfer.md`, `julia_parent_sourcing_probe.md`, `prio_terms_park_note.md`.

## What belongs here

**A document belongs in `docs/design/` only if something in the code owns it and it stays
true as the code changes.** Two corollaries, both learned the hard way:

- **A repo measurement owns nothing, and is false the moment the cleanup it drove
  succeeds.** `repo_size_audit.md` and `repo_structure_audit.md` were one-off cleanup
  snapshots; they were committed, went ~20× numerically stale while sitting where they read
  as current, and were eventually deleted outright (`27b68a4`). `repo_analysis.md` was their
  successor and the same kind of document — briefly moved *into* this directory by the
  2026-07-31 hygiene pass, which made it *more* authoritative-looking, not less. It was
  extracted and deleted on 2026-07-31: the LFS-prune trap and the `.gitignore`
  interleaving cause went to [`artifacts_resolver.md`](artifacts_resolver.md) §1/§5, the
  `sys.path` and copy-forward-`vN` facts to `tools/README.md`, and the storage-class
  violations were already in [`storage_classes.md`](storage_classes.md). Nothing else
  survived the test. Recover any of the three with
  `git show <commit>^:<file>` — that is derivable from git and needs no home here.
- **A reference with no owner in the code goes the same way — but one whose subject the
  code owns gets FOLDED IN, not filed beside.** `uf_coloring_algorithms.md` (the four Ultra
  Fractal `Standard.ucl` colorings) was real design knowledge, since all four are
  implemented as `render_modes::Field` variants with `specs/*.json`. But its technical body
  was already carried by the code's own doc comments, and its planning half addressed a
  numpy rasterizer that never existed. Its residue — provenance, the UF execution model, and
  where our port deliberately stops short — is now
  [`render_coloring_surface.md`](render_coloring_surface.md) §6. The file is gone.

A measurement that survives into a doc here carries its **date and the command that produced
it** (`[measured: …]` / `[cmd: …]`). If it cannot carry those, it is a snapshot and does not
belong.

**Provenance.** These docs were promoted in the 2026-07-24 docs-hygiene pass; the
research ledger they distil (`docs/findings/`, `docs/rescued/`) was retired in the same
pass and is recoverable from git history. See `scratch/docs_cleanup_summary.md` for the
per-file dispositions.

**`docs/findings/` is retired and stays retired.** The 2026-07-25 retirement moved two
stragglers and left no guard, so five later runs recreated the directory and wrote into it;
those six files were folded in here on 2026-07-27. `tests/test_docs_tree.py` now enforces
three things mechanically: `docs/findings/` does not exist, no tracked source file names it
as a path, and **every file under `docs/` is git-tracked** — so a generated sheet cannot be
parked in `docs/` and gitignored (which is what left the repo-size guard permanently red).
Generated views go to `scratch/`, per the generated-output convention in `CLAUDE.md`.
