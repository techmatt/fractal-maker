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

**Provenance.** These docs were promoted in the 2026-07-24 docs-hygiene pass; the
research ledger they distil (`docs/findings/`, `docs/rescued/`) was retired in the same
pass and is recoverable from git history. See `scratch/docs_cleanup_summary.md` for the
per-file dispositions.
