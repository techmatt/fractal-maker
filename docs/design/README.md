# docs/design — canonical design layer

The curated, checked-in design docs. Each holds durable, design-informing knowledge
distilled from the (now largely deleted) `docs/findings` research ledger — the timeless
core, stripped of run logs and dated readouts. `CLAUDE.md` remains the top-level map of
the render core and pipeline; these docs go a level deeper on specific decisions.

| doc | governs |
|---|---|
| [phoenix_seed_sampler_spec.md](phoenix_seed_sampler_spec.md) | Phoenix seed proposal: the closed-form stability skeleton, the `(c,p,z₋₁)` axes, the corrected real-axis-reflection symmetry (§7.1), the fertility-aware surrogate loop, and the settled human-label verdicts (§8). |
| [deep_zoom_sourcing.md](deep_zoom_sourcing.md) | Deep-center sourcing (Newton nucleus/Misiurewicz → decimal-string centers), depth-band locality, the f64-bound walker, the standing perturbation-engine gaps, and why deep-Mandelbrot curation needs its own objective. |
| [julia_c_sourcing.md](julia_c_sourcing.md) | The ~2% Julia "corner" exemplar class and the three-stage `c`-selection screen; the interior-lake determinant. |
| [aesthetic_scoring.md](aesthetic_scoring.md) | How to read the signal: occupancy/busy-ness is anti-quality (report-only), and `p_good` is a badness filter, not a goodness ranker. |
| [classifier_retrain_protocol.md](classifier_retrain_protocol.md) | Append-don't-rebuild manifests, forced eval/train split rules, pre-registered paired-eval bar, and per-version `t_good` re-derivation. |
| [morphology_dedup.md](morphology_dedup.md) | Visual dedup: coordinate dedup is not visual dedup, the cone-compressed CLIP space, grayscale-CLIP descriptor choice, and dedup-key identity rules. |
| [render_coloring_surface.md](render_coloring_surface.md) | The two coloring algorithms, three palette namespaces, the silent density fork, the open corpus-coloring hazard (P4), and why the OKLab ports must not be merged. |
| [discovery_pipeline.md](discovery_pipeline.md) | The guided-descend walk + reward split, the freshness-prior/dive incompatibility, and the distinct-look deficit scheduler (incl. julia routing). |
| [storage_classes.md](storage_classes.md) | The durability contract: the four storage classes, why a missing `scratch/` artifact is not data loss, and that durability is claimed only by `durable()` at the write site. |

**Dated readouts** (a run's numbers, not a timeless rule) also live here — there is nowhere
else. They are named for the run and carry their own date: `interior_feature_bakeoff.md`,
`interior_band_batch_v1.md`, `minibrot_roster_v2_{pilot,readout}.md`,
`minibrot_label_batch_v2.md`, `nucleus_seeding_and_atom_A.md`,
`closeout_pre_distillation_2026-07-25.md`, `migration_to_fractal_maker.md`,
`walk_era_julia_resolution_audit.md`.

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
