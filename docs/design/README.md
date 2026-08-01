# docs/design — canonical design layer

The curated, checked-in design docs. Each holds durable, design-informing knowledge
distilled from the (now largely deleted) `docs/findings` research ledger — the timeless
core, stripped of run logs and dated readouts. `CLAUDE.md` remains the top-level map of
the render core and pipeline; these docs go a level deeper on specific decisions.

| doc | governs |
|---|---|
| [phoenix_seed_sampler_spec.md](phoenix_seed_sampler_spec.md) | Phoenix seed proposal: the closed-form stability skeleton, the `(c,p,z₋₁)` axes, the corrected real-axis-reflection symmetry (§7.1), the fertility-aware surrogate loop, and the settled human-label verdicts (§8). |
| [deep_zoom_sourcing.md](deep_zoom_sourcing.md) | **Deep regime (PARKED).** Deep-center sourcing (Newton nucleus/Misiurewicz → decimal-string centers), depth-band locality, the f64-bound walker, the standing perturbation-engine gaps, and why deep-Mandelbrot curation needs its own objective. |
| [minibrot_sourcing.md](minibrot_sourcing.md) | **Moderate depth, inside f64.** The minibrot-atom arc: the atom roster and (§2.1) how nuclei are actually seeded — a ring-grid *draw*, not a census — the stage-1 screen `G` and its measured worth, the pre-filter and its interior catch-22, the `A` feasibility cut, and the atom/window-level signals from the labeled evidence. |
| [julia_c_sourcing.md](julia_c_sourcing.md) | The ~2% Julia "corner" exemplar class and the three-stage `c`-selection screen; the interior-lake determinant. |
| [aesthetic_scoring.md](aesthetic_scoring.md) | How to read the signal: occupancy/busy-ness is anti-quality (report-only), and `p_good` is a badness filter, not a goodness ranker. |
| [classifier_retrain_protocol.md](classifier_retrain_protocol.md) | Append-don't-rebuild manifests, forced eval/train split rules, pre-registered paired-eval bar, and per-version `t_good` re-derivation. |
| [morphology_dedup.md](morphology_dedup.md) | Visual dedup: coordinate dedup is not visual dedup, the cone-compressed CLIP space, grayscale-CLIP descriptor choice, and dedup-key identity rules. |
| [render_coloring_surface.md](render_coloring_surface.md) | The two coloring algorithms, three palette namespaces, the silent density fork, the open corpus-coloring hazard (P4), why the OKLab ports must not be merged, (§6) the Ultra Fractal originals four `render_modes` fields were ported against — incl. the three `DirectShape` names that collide with UF's while computing something different — and (§7) the two field-dump sources, the `FieldNeeds` gate that closed the ~35× `beautiful` gap, and why the field source belongs in the cache key. |
| [auto_maxiter.md](auto_maxiter.md) | The depth-aware iteration cap: the closed form and its two load-bearing sites, the 32-atom measurement that showed base 500 was ×8 too low, the adopted base 4000 / clamp 67000, the **median-clean-not-clean** residual (tail runs to ×24), the aug cache's separate flat-8000 cap, and the four things a cap change moves. |
| [discovery_pipeline.md](discovery_pipeline.md) | The guided-descend walk + reward split, the freshness-prior/dive incompatibility, and the distinct-look deficit scheduler (incl. julia routing). |
| [storage_classes.md](storage_classes.md) | The durability **contract**: the four storage classes, usefulness-before-cost, why a missing `scratch/` artifact is not data loss, `scratch/` as a liveness rule (not just a rebuild-cost one), reproducible-on-demand as the KEEP test (incl. the chain shape that cost `data/v4..v7`), git history as a durability tier with a 2026-07-24 floor and where the pre-migration record actually lives, derive-in-code/freeze-in-records, and the no-bulk-in-tree rule (traversal cost, not repo size). |
| [artifacts_resolver.md](artifacts_resolver.md) | The storage **mechanism**, named for `tools/corpus/artifacts.py`: the `ARTIFACTS_ROOT` resolver and its seam, relocation classes matched by pattern + the reappearance tripwire, the size-guard registry (dispositions, prefix-vs-file granularity as a trade, `forward` declarations and the end of the standing stale warning), LFS + exact-path `.gitignore` negation + why verification runs `--no-index`, the derived canary set and its non-vacuity pairing, and the small-tree / docs-are-source guards. |
| [orbital_field_metrics.md](orbital_field_metrics.md) | The ring measures `radial_rings` / `radial_range` (`tools/orbital/`): what they compute, the validation record incl. the failures (`cycles_spanned`, `falloff_extent`, p90), the span-vs-oscillation two-axis reading, the screening/validation resolution split, the cap-provenance axis, and what the instrument is blind to. |
| [label_corpus_relocation.md](label_corpus_relocation.md) | Relocating the label corpus's `crops/`+`vivid/` bulk (3,822 files, ~72% of the working tree) out of tree behind `artifacts.resolve`: why the scope is crops/vivid only (labels stay tracked in-tree), the silent-zero hazard across 34 construction sites (2 load-bearing readers), the seam, and the staged move + before/after `(image_id,score)` gate. |
| [minibrot_maneuvers.md](minibrot_maneuvers.md) | Minibrot moves as candidate MOVES inside a descent (`tools/atlas/minibrot_maneuvers.py`): the two operators (`snap_to_nucleus(k)` / `lateral_to_sibling`) and why five collapse to two, the superattracting-argmin correction that makes the atom-domain probe work, unavailability as the normal case, the reserved frontier FLOOR (of available) vs the probability used only as a cost governor, and the provenance a later read needs. |
| [atom_instrument.md](atom_instrument.md) | The atom instrument `A` (`deep_center_finder.atom_instrument`): the recursion for size / orientation / required precision, and the f64-wall predictor derived from it. |
| [label_rubric.md](label_rubric.md) | The 1–4 human quality scale, cited by every corpus batch builder: the one question a labeler answers, judge-from-the-vivid-render rule, and class 4 as a fourth tier that ranks the top of "good" without moving the `>=3` emit floor. **Class-4 aesthetic criteria are a stub** pending the anchor pass. |
| [deferred_recalibration.md](deferred_recalibration.md) | Four recalibrations designed but deliberately unbuilt (v8 head retrain, ranker growth, location blind reads, mining-head calibration), the release-review gate that unparks them, and where to start on each. |
| [v8_training.md](v8_training.md) | The v8 train-split population read off `data/v8/manifest.jsonl`: locations per (fractal partition × quality class), and the ×24 augmentation-tile expansion under the v8b recipe. |
| [pytest_suite_cost.md](pytest_suite_cost.md) | The Python suite's cost model: five files hold ~all of it, the default/`-m slow` lane split and the two ways to get it wrong (`-m` is a filter not a path rule; mark the test not the module), the three shared-fixture fixes, the four costs measured to be **at their floor** (incl. the tripwire's worker/thread sweep), and the prototyped-but-unadopted xdist 1.88×. |
| [q4_multibrot_transfer.md](q4_multibrot_transfer.md) | Whether the degree-2-fitted q4 stage-1 screen transfers to d3/d4/d5 multibrot minibrots: it does, but only after removing rotational-copy pseudo-replication and period mismatch, and one headline claim did not survive. Imported read-only as the deployed OOD mask. |
| [q4_harvest_emission.md](q4_harvest_emission.md) | Wiring the q4 tight harvest (`tools/studies/q4_harvest_tight.py`, G-gated framings) into emission as a first-class source through intake → cells/deficit → colorize → gate/pool → select; and why `current-decode` in `load_admitted` is a firewall that bounds the blast radius of any resolver bug behind it. |
| [julia_parent_sourcing_probe.md](julia_parent_sourcing_probe.md) | **Negative result.** Sourcing julia roots from the c-diverse near-∂M sampler does NOT reduce `precanon_dup` (93.6% vs 90.3% baseline): the churn is intra-`c` z-plane self-saturation, not cross-parent c-crowding. |
| [prio_terms_park_note.md](prio_terms_park_note.md) | Park note for `prio_terms.jsonl` (one row per *pushed* candidate, incl. the never-admitted majority): what it is, where it lives, why it is retained unprocessed. |

**Two sourcing docs, one boundary — pick by depth regime.** `deep_zoom_sourcing.md` is
the **deep, beyond-f64 tier** (perturbation, ∂M-tracking, deep-center production) and is
**parked**; `minibrot_sourcing.md` is the **moderate-depth minibrot-atom arc that runs
entirely inside f64** (roster, screen, pre-filter, labeled evidence). They touch at only
two places — the atom-size law `size ≡ 1/|A|` and the fitted-objective status — and each
doc's boundary note names those explicitly. If you want roster / screen / pre-filter /
window-label content, read `minibrot_sourcing.md`; if you want precision, the
perturbation tier or deep-center production, read `deep_zoom_sourcing.md`.

**The table indexes all 24 docs here** — every file gets a row, because an unindexed doc is
invisible, which is how one rots unnoticed. A new doc lands with its row or it does not land.
`[measured: 24 docs, 24 rows; 2026-07-31]`
`[cmd: ls docs/design/*.md | grep -vc README; grep -c '^| \[' docs/design/README.md]`

**The dated-readouts carve-out is withdrawn, and the readouts are gone.** This README used to
declare that "dated readouts (a run's numbers, not a timeless rule) also live here — there is
nowhere else", naming nine documents. That was a standing exception for exactly the category
the test below excludes, and it is what made a one-off repo snapshot look at home in the
curated layer. There *is* somewhere else: analysis goes to `scratch/`, what survives is
extracted into the doc that owns the subject, and the analysis is deleted.

**All nine were extracted and deleted on 2026-07-31, plus `beautiful_perf_report.md`, which
was folded rather than dropped** (its subject — the cost of the two field-dump sources — is
owned by `render_coloring_surface.md` §7). Where each one's residue went:

| deleted | residue landed in |
|---|---|
| `minibrot_roster_v2_pilot.md` · `minibrot_roster_v2_readout.md` · `minibrot_label_batch_v2.md` · `interior_feature_bakeoff.md` · `interior_band_batch_v1.md` | [`minibrot_sourcing.md`](minibrot_sourcing.md) already carried most of this arc; the split added §2.1 (seeding as a draw), the zero-accept shallowest band, G's train-vs-eval split, the `g_interior` = 0.037 effective ceiling, the near-miss negative mechanism, and the eval-composition objection to the degree gradient |
| `nucleus_seeding_and_atom_A.md` | §2 was already [`atom_instrument.md`](atom_instrument.md); §1 → `minibrot_sourcing.md` §2.1; §3 (min\|z\|) was already `minibrot_sourcing.md` §10; §4 (field-source cache key) → [`render_coloring_surface.md`](render_coloring_surface.md) §7 |
| `beautiful_perf_report.md` | [`render_coloring_surface.md`](render_coloring_surface.md) §7 — **folded, not deleted** |
| `walk_era_julia_resolution_audit.md` | [`q4_harvest_emission.md`](q4_harvest_emission.md) — the decode-version firewall, which is the durable half of a verdict of "nothing to do" |
| `closeout_pre_distillation_2026-07-25.md` · `migration_to_fractal_maker.md` | [`storage_classes.md`](storage_classes.md) — the chain-regenerability failure that cost `data/v4..v7`, and git history as a durability tier with this repo's 2026-07-24 floor |

Recover any of them with `git show <commit>^:<file>` — mind that this repo's history floor is
the 2026-07-24 import (`storage_classes.md`).

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
