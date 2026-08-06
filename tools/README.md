# tools/ — index

413 tracked files, 381 of them `.py`, across 35 subdirectories. This file is the entry
point for **"what produces X?"**. It is an index, not a contract: nothing here is
enforced by a test, so treat a verdict as evidence to check, not as permission to delete.

**File counts re-derived 2026-07-31 at this commit**; the `A`/`B`/`∪` liveness numbers are
from the 2026-07-31 @ `fad68df` pass and are **not** re-derived here, because the analysis
scripts that produced them do not exist (§[Method](#method)). The two are stamped
separately on purpose: the counts were **stale on arrival** — measured at `fad68df` but
committed in `b6ce6ee`, which itself added six test files, so six rows and the header were
wrong the moment they landed. Regenerate the counts with the commands stamped in
[§Method](#method); re-deriving the liveness numbers is a bigger job and says so.

## How liveness was established (and why not by date)

Two traps make the obvious approaches produce a confidently wrong index:

1. **`git log` cannot date liveness here.** Every path's first commit is the 2026-07-24
   `fractal-maker` migration import, so last-touch dates cluster into eight days and rank
   retired scaffolding alongside current drivers.
2. **An import-graph test alone marks almost everything dead.** 280 of the 373 tracked
   `.py` files are standalone `__main__` entry points with no importer *by construction*.
   Absence of an importer is the normal case here, not evidence of death.

So liveness is established by **two methods that fail differently**, and the table
reports their union:

- **A — import graph.** Who imports this module? Resolved *to a file*, not a basename:
  `from tools.v7 import build_manifest` resolves exactly; a bare `import build_plan` is
  resolved against the importer's own directory and its `sys.path.insert` targets; and a
  second pass follows `spec_from_file_location` path loads, which no AST import walk sees.
  All three refinements changed verdicts. Basename matching credits all six
  `tools/v*/build_plan.py` with an importer, when in fact v5/v6 only *mention*
  `tools/v4/build_plan.py` in prose ("MUST match verbatim") — **the vN recipe chain is a
  copy-forward, not an import** — and package-member resolution has to try
  `tools/emission/descriptor.py` before `tools/emission/__init__.py` or every member of
  that package loses its real importers.
- **B — artifact map.** Does this entry point *produce* a committed artifact? A module is
  B-live if it both names a git-tracked path (under `data/`, `labels/`, `specs/`, …) and
  contains a write call. A driver whose outputs are live in `data/` is live regardless of
  who imports it.

### What neither method sees

Four blind spots they share. All four show up inside the "dead by both" pool, which is
why that pool is a **candidate** list and nothing more:

1. **Interactive tools invoked by a launcher, not an import.** `explorer/app.py` scores
   dead by both — and is launched by the tracked `explorer.cmd` at the repo root.
   `viz/serve.py` likewise: 0 importers, no writes, and it is the static server the
   documented labeling flow runs against.
2. **Closed runs vs retired code.** B-live means "produced a committed artifact *once*".
   It cannot distinguish a driver that will run again from a one-shot version driver whose
   artifact is frozen. Most of `v8/`/`v9/` is B-live in exactly this frozen sense.
3. **Frozen-by-design dependencies.** Several closed studies are deliberately imported
   read-only by live code and must not be edited — `studies/q4_stage1_linear_fit.py` (11
   importers) is the deployed OOD mask, imported by the live `sourcing/` batch builders
   precisely so screen parity holds. "Closed" and "dead" are different states.
4. **Dynamic loads assembled from path *components*.** The `spec_from_file_location` pass
   catches single-literal targets, but not paths built piecewise. Two real edges it still
   misses, both confirmed by hand: `tests/test_repo_size_guard.py` execs
   `audit/size_guard.py` via `REPO_ROOT / "tools" / "audit" / "size_guard.py"`, and
   `corpus/location.py` execs `scoring/active_ckpt.py` the same way. So an A count of 0
   is weaker evidence than an A count of 0 *plus* a grep for the filename.

## The index

`n` = non-test modules; `A`/`B` per the two methods; `∪` = union live.

| dir | files | drives | verdict | n / A / B / ∪ |
|---|---:|---|---|---|
| *(top level)* | 11 | Shared helpers every subdir imports: `paths.py` (storage-class write gate, 42 importers), `colormap.py` (the Python coloring tail for the field⊗colormap split, 32), `_bootstrap.py` (`sys.path` for 4 of 36 subdirs), `kill_run.py`. | **live** | 6 / 5 / 3 / 5 |
| `atlas/` | 48 | **The standing discovery flow.** `production_seeder.py` (26 importers) → `guided-descend` → reward; `steered_frontier.py` (classifier-steered descent), `guard.py` (degenerate-outcome gate), `deficit_scheduler.py`, `minibrot_maneuvers.py`, `prescreen.py`. Plus ~9 closed per-campaign readouts and the maneuver measurement tools added 2026-07-31 — `bench_lateral_seeding.py` (replay bench), `bench_neighborhood_subsumption.py` (is operator 2 subsumed by operator 3), `maneuver_degree_readout.py` (per-degree availability/cost) and `maneuver_inspection_sheet.py` (captioned richness sheets). `maneuver_screen.py` is different — it is imported by `steered_frontier.py` and is the field half of the richness screen. Both are hand-run and write only `scratch/`, so both are **dead by both methods by construction** (blind spot 1), which is what the pool is for. Plus the 2026-08-01 view-screen group — `view_screen.py` (the view-level composition screen; imported by the other four), `view_rescreen.py`, `view_screen_gate.py`, `view_frame_sweep.py`, `view_screen_sheets.py` — all hand-run. They stopped being **retroactive** on 2026-08-01: `--maneuver-view-prior` (v1.5) makes `view_screen.composite_v3` the live maneuver sort key, so `view_screen.py` is now on the discovery path through `maneuver_view_screen.py` (the run-time view screen) and `view_field_cache.RunFieldCache` (the run-local f32 field store the walk writes as it screens). Three more modules landed the same day — `maneuver_view_screen.py` (A-live: imported by `steered_frontier`), `exemplar_similarity.py` and `build_supply_crawl_batches.py` (the supply crawl's label batches; B-live, it writes `data/label_corpus/batches/2026-08-01_supply_crawl_*` and `data/supply_crawl/`). `view_fit.py` (2026-08-02) fits the labeled crawl into a linear replacement for `composite_v3` and is **staged, not live** — B-live (it writes `data/atlas/view_fit_v1.json`) and deliberately A-dead, the contract `test_view_fit.py::test_no_live_sort_path_imports_the_fitted_score` asserts; an importer appearing is the signal that adoption happened. `--maneuver-range-prior` (the 4× atom score) is unchanged and still the default. The three modules added 2026-08-02 are the **label-seeded harvest**, a sourcing path that does NOT go through the walk at all: `label_seeded_harvest.py` (seed pool from the corpus's own class-3/4 locations → `atom_lib.identify_nucleus` at the judged view → `minibrot_maneuvers.neighborhood_expand` around it → view screen → interior>0.30 discarded at sourcing; B-live, it writes `data/label_seeded_harvest/`), `build_label_seeded_batches.py` (the fit-ordered queue and the two label chunks; B-live, `data/label_corpus/batches/2026-08-02_label_seeded_v2_*`) and `test_label_seeded_harvest.py`. `livefire_harvest_budget.py` (2026-08-02) is the hand-run end-to-end proof of that harvest's three halt mechanisms (active-time cap, wall budget, STOP sentinel) — ~13 min, drives the real engine, writes only `scratch/budget_livefire/`, so it is past even the `slow` lane; the branch logic itself is in the suite. `view_fit.py` gains a second record (`view_fit_v1_1.json`) which IS read at run time by the batch builder — so `view_fit` is now **A-live**, and by that contract this importer is exactly the signal that adoption happened. It is adoption as a **sourcing queue's sort key only**: `composite_v3` remains the live sort key on the discovery path, and `test_label_seeded_harvest.py::test_the_live_sort_key_elsewhere_is_still_composite_v3` asserts neither `steered_frontier` nor `maneuver_view_screen` imports it. `view_fit_bar_read.py` (2026-08-04) TAKES the pre-registered `+0.1181` delta-AP bar against `composite_v3` on the v2 sitting's 268-row readable slice and freezes the verdict (`data/atlas/view_fit_v1_1_bar_read.json`); B-live and A-dead, and it is a READ — the import ban's module list is unchanged because it orders nothing. Two more landed 2026-08-05 with the first steady-state run, both post-run readers of a finished run dir: `derive_quota_prices.py` (regenerates the pop-quota **cost-to-mine seed table** — `data/atlas/quota_prices_v1.json`, fed back via `--quota-prices`; B-live, A-dead) and `harvest_log_reconcile.py` (dedups the append-only `harvest_log.jsonl` by `node_id` and ties it to the run's checkpointed `totals` before any rate is quoted; it is the gate in front of the precanon-skip-rate read). `harvest_log_registry.py` (2026-08-05) is the third of that group and the one on the τ_h path — it replaced `tau_h_rederive`'s five-entry hand list with discovery over registered stores. `regularize_quota_prices.py` (2026-08-05) is the fourth: it shrinks that measured price table geometrically toward its own median (α=0.7) into `data/atlas/quota_prices_regularized_v1.json`, which is now the **default** `--quota-prices` seed and fatal when absent — so it is B-live, and A-live by reference (`steered_frontier.QUOTA_PRICES_DEFAULT` names its artifact). | **live** | 40 / 14 / 22 / 28 |
| `atlas_probe/` | 5 | Step-0 atlas measurement probes (a closed study) — **but `step0_reanalysis.py` is the k3 reward primitive `production_seeder` imports** via an explicit `sys.path.insert`. Not retirable. | **live** | 5 / 3 / 2 / 4 |
| `audit/` | 6 | `size_guard.py` (the repo-size registry — exec'd by `tests/test_repo_size_guard.py` through a piecewise path the A pass cannot see), `disk_audit.py` (safe-delete classifier), `durability_map.py` (declared-vs-actual storage class). | **live** | 3 / 1 / 3 / 3 |
| `corpus/` | 44 | **The label-corpus contract.** `corpus_common.py` (row shape, 59 stem-refs), `location.py` (canonical location identity, 57), `label_store.py`, `corpus_reader.py` (the trainer's version-blind view), `merge_scores.py` (single-batch **and** `--route`d combined-sheet merges), `build_combined_label_sheet.py` (serve N registered batches as ONE blind sitting; the sheet is a presentation alias, never a batch), `artifacts.py` (the `ARTIFACTS_ROOT` resolver), `enrich_score.py`/`enrich_select.py`, `q4_window_reader.py`, + ~15 closed batch builders. | **live** | 32 / 12 / 14 / 23 |
| `curation/` | 4 | `colored_clip` palette-appearance descriptors + collision-aware palette placement (`colorize_assign`, soft-spread). | **live** | 4 / 3 / 2 / 4 |
| `descent/` | 14 | The minibrot **descent harness** and **triage wall** — two Flask apps + their durable stores (`data/descent_harness/`). Matt-driven; the recorded path is the product. | **live** | 8 / 7 / 5 / 8 |
| `descent_ablation/` | 3 | One overnight ablation + percentile-strategy campaign. Has its own `README.md`. Closed. | retired | 2 / 1 / 1 / 2 |
| `eda/` | 13 | One-off render/coloring eyeball studies (direct-trap sweeps, interior-trap validation, deband) + the whole closed scale-2×2 corpus arc. **0 of 12 live by either method.** | retired | 12 / 0 / 0 / **0** |
| `emission/` | 26 | **Diversity-aware emission v1**: `cells.py` (joint-count cells + the target measure DERIVED from `scoring/release_mix.RATIO`), `descriptor.py` (intake → namespaced cross-ledger union → partition-keyed morph cluster), `palette_deficit.py`, `pool.py`, `selection.py`, `release_record.py`, driven by `build_emission_diversity_v1.py`. `floors.py` (2026-08-04) is THE stage-2 cut owner — the four live floors (wallpaper pool 0.75 / release 0.90, mining pool 0.25 / release 0.50 report-only), each stamped with the head VERSION it was set against and refusing to gate when the live pin disagrees; the release floors are imported from the heads' own gates, and `test_floors_one_source.py` scans the stage-2 surface for a re-typed literal (there were six copies of four numbers). `ledger_rescore.py` (2026-08-04) brings the seven intake ledgers current under the active head — a sibling `<stem>.rescored_<version>.jsonl` that `descriptor.resolve_rows` overlays, never an in-place edit; hand-run at a flip (`classifier_retrain_protocol.md` §5). | **live** | 16 / 12 / 6 / 13 |
| `explorer/` | 4 | The Flask Mandelbrot/Julia explorer + `render_core.py` (shared pixel→plane math, 13 importers, also used by the descent apps). `app.py` scores dead by both methods and is launched by `explorer.cmd`. | **live** | 2 / 1 / 0 / 1 |
| `julia_ladder/` | 1 | The J0 unlabeled Julia batch generator. One batch, produced and labeled; nothing imports it. | retired | 1 / 0 / 0 / **0** |
| `mining/` | 16 | The strange/render-mode quality gate: `mining_pins.py` (the TORCH-FREE pin block — ckpt, threshold, derived version tag) + `mining_gate.py` (the pinned `mining_v1`, re-exporting it), `score_lib.py` (the v3 deploy-transform scorer, 28 importers), `dedup.py`, `tail_alloc.py`, `deploy_tail.py`. **The corpus rebuild (2026-08-06)** lives here too, since the head's label corpus was lost and `render_mode_pilot/` is retired: `build_gate_passers.py` (regenerates the wallpaper-v3 gate-passer population, count-verified 401/112, → the tracked `data/render_mode_corpus/gate_passers_v3.json`), `mining_roster.py` (THE 15-mode roster **and** its per-mode render recipe — one owner, four copies before), `split_units.py` (THE Julia-parent union-find split, lifted out of `render_mode_pilot/build_scale_sample.py`), `suggest_tier_mining.py` (the K=3 pre-label rule; imports `wallpaper/suggest_tier`'s rule rather than restating it), `build_mining_sheet.py` (plan/render/write). | **live** | 10 / 7 / 2 / 7 |
| `orbital/` | 8 | The ring measures `radial_rings`/`radial_range` and the cap-policy stamping — owned by `docs/design/orbital_field_metrics.md`. | **live** | 6 / 6 / 5 / 6 |
| `palettes/` | 11 | The durable palette pool / features / k-cut categories, authored-palette densification, and the preview-sheet harness. | **live** | 10 / 6 / 3 / 7 |
| `phoenix/` | 12 | The phoenix seed sampler (`phoenix_sampler.py`, spec: `docs/design/phoenix_seed_sampler_spec.md`) + the closed Phase-B grid and its label analysis. Sampler live, grid closed. | **live** | 10 / 4 / 3 / 6 |
| `queries/` | 20 | The **palette-preference** corpus and scorer (distinct from the location classifier): `query_sampler.py` (20 importers), `scorer/` (v1 → v3-gvo; `data/queries/scorer/v3_gvo` is the deployed pref head), `query_label.html`. | **live** | 18 / 11 / 2 / 12 |
| `ranker/` | 8 | The location preference ranker, deployed head `pref_loc_v1` (`scorer.py`, `score_locations.py` — the consumer-side seam). | **live** | 6 / 3 / 3 / 6 |
| `readout/` | 1 | One morning diversity readout over an overnight emit manifest. | retired | 1 / 0 / 0 / **0** |
| `reframe/` | 2 | The promoted coarse-reframing step of discovery (`reframe_location`, 12 importers) — runs on every discovery survivor. | **live** | 1 / 1 / 1 / 1 |
| `reframe_probe/` | 1 | One coarse-reframe speed diagnostic. Closed. | retired | 1 / 0 / 1 / 1 |
| `render_mode_pilot/` | 8 | The 500 + 1000 render-mode label batches that produced the mining-head dataset (`labels/render_mode_pilot_v1.json`). **0 importers, 5 committed producers** — the clearest B-only block in the tree. Closed, and now UNRUNNABLE: both its inputs (`scratchpad/gate_passers_v3.json`, the pilot `images.jsonl`) are gone. Its split rule moved to `mining/split_units.py` (2026-08-06) and `build_scale_sample.py` imports it; the live sampler is `mining/build_mining_sheet.py`. | retired | 8 / 0 / 5 / 5 |
| `scoring/` | 14 | `production_pins.py` — **`ACTIVE_CKPT`, the single source of truth for the live classifier pin**, plus `PALETTE`/`JPG_Q`/`BIN`/`auto_maxiter` and `COUPLED_ARTIFACTS` (the revert-together set as data, walked by `test_coupled_artifacts.py`). `active_ckpt.py` re-exports all of it (~41 importers use that name) and is otherwise the retired reframe probe, two of whose helpers are still imported. Gained the two version-following flip gates at the v10 flip: `test_t_good_adoption.py` (adopted table ⟷ the ACTIVE version's derivation) and `test_flip_end_to_end.py` (`slow`; the rendered proof of whichever head is live). Both were `v8/`-scoped files that hardcoded the version and would have gone red *for* a flip rather than for a fault. The 2026-08-02 cleanup added three shared owners the version dirs used to each re-declare: `derive_t_good.py` (THE t_good estimator, imported by v8/v9/v10), `partitions.py` (the `fractal_type` ⟷ partition map — seven literal copies collapsed to one, guarded by a source scan) and `eval_slice.py` (a version's frozen slice: its path and its `<v>_p_geN` columns). The 2026-08-04 split-machinery unification added a fourth: `batch_registry.py`, THE batch->split classification table (one owner for `v7/build_manifest.assign_split` and `v8|v10/build_manifest.classify_batch`, which had drifted apart on `loose0_v3`), guarded by a source scan in `test_batch_registry.py`. A fifth landed 2026-08-04: `release_mix.py`, THE canonical per-partition release-mix ratio table (`mandelbrot : multibrot3 : phoenix:classic = 3 : 1 : 0.2`), keyed off `ALL_FAMS` with a two-directional completeness assertion at import and its own source scan; `pop_quota`'s deficit target is its only reader today — emission wires to it at the next checkpoint. | **live** | 2 / 2 / 1 / 2 |
| `sources/` | 7 | The minibrot **source sheets**: seven generation algorithms → `data/minibrot_sources/`, one HTML sheet per algorithm. | **live** | 6 / 5 / 1 / 5 |
| `sourcing/` | 11 | The durable minibrot **roster** (`build_minibrot_roster.py`), `deep_center_finder.py` (19 importers), and the live label-batch builders (`build_minibrot_batch`, `build_interior_band_batch`, `build_gcf_arm_batch`). | **live** | 7 / 6 / 5 / 7 |
| `specs/` | 1 | Generates `specs/REGISTRY.md` from `specs/modes_registry.json`; parity enforced by `cargo test --test modes_registry`. 0 importers — A alone would kill it. | **live** | 1 / 0 / 1 / 1 |
| `studies/` | 40 | 37 closed measurement passes (the q4 stage-1 arc, morphology dedup, descent score fidelity, …) plus `archive/`. **13 are frozen-by-design dependencies of live code** and must not be edited — `q4_stage1_linear_fit.py` (11 importers) is the deployed OOD mask. | **live** (frozen) | 37 / 13 / 7 / 17 |
| `v4/` | 6 | Origin of the 42-slot augmentation-cache recipe. **0 importers, 0 committed outputs — dead by both methods.** The recipe is carried forward by textual copy, not import. | retired | 6 / 0 / 0 / **0** |
| `v5/` | 4 | v5 manifest + plan (freeze v4, fold J0 Julia). Only `build_plan.py` is live, and only because `test_recipe_parity_v5.py` execs it by path. | retired (1 pinned) | 3 / 1 / 0 / 1 |
| `v6/` | 7 | v6 manifest + plan + the monitored-harvest drivers. Same shape: `build_plan.py` pinned by `test_recipe_parity_v6.py`, the rest dead by both. | retired (1 pinned) | 6 / 1 / 0 / 1 |
| `v7/` | 6 | Mostly closed — **except `build_manifest.py`** (imported by the live `sourcing/build_gcf_arm_batch.py` for `assign_split`) and **`eval_delong.py`** (the acceptance battery v8 and v9 both import). | **live** (2 of 4) | 4 / 2 / 0 / 2 |
| `v8/` | 9 | The 2026-07 training generation: `data/v8/{manifest,plan,cache_manifest}.jsonl`, `eval_v8.py`, the render supervisor. `ACTIVE_CKPT` is v10 as of 2026-08-02; v8 is the one-flip rollback anchor. Its two version-pinned tests moved to `scoring/` at that flip, and so did the estimator: `derive_t_good_v8.py` had outlived its version (v10 imported `build_table` from it) and is now the thin v8 population/objective wrapper over `scoring/derive_t_good.py`. | **live** | 8 / 3 / 7 / 7 |
| `v9/` | 9 | The raised-cap re-render generation (`docs/design/auto_maxiter.md`). Artifacts committed; `t_good` and keeper cuts derived but **STAGED, and now permanently so** — v10 was adopted over it. Never deployed, and explicitly **not a rollback rung**. | **live** (staged) | 8 / 1 / 6 / 6 |
| `v10/` | 11 | **The LIVE training generation** (`ACTIVE_CKPT` = v10, flipped 2026-08-02). v9's cache extended with 1,267 appended maneuver-view locations; `build_manifest.py`/`build_plan.py`/`render_cache.py`/`verify_cache_alignment.py`, `prereg.py` + `eval_v10.py` (the pre-registered acceptance battery), `diagnose_selection.py`. `derive_t_good_v10.py` is the adopted discovery table and `test_v10_flip.py` the flip's no-GPU proof. | **live** | — |
| `viz/` | 26 | 23 committed HTML inspection/label UIs — including **`corpus_label.html`, the labeling harness** named in `CLAUDE.md` — plus `serve.py` (the single-host static server for it) and two sheet builders. The `.py` side scores dead by both; the directory is not. | **live** | 3 / 0 / 1 / 1 |
| `wallpaper/` | 24 | The emission / location-library tail: `library_store.py`, `library_annotate.py` (12 importers), `prospect_orchestrator.py`, `wallpaper_pins.py` (the TORCH-FREE head pin + gate threshold) + `emit_v1.py` (re-exporting it), `pool_rule.py`, `label_crop.py`, `emission_selector.py`. Four batch builders are held live only by `test_builders_import.py`, a module-load smoke that exists because exactly this rot went unnoticed once. | **live** | 19 / 13 / 7 / 17 |

## Candidate retirements

**Nothing was deleted or archived in this pass** — this is the evidence, not the decision.
101 non-test modules score dead by *both* methods. Ranked by how safe the call looks:

| candidate | evidence | caveat |
|---|---|---|
| **`v4/` (6 files)** | 0 importers, 0 committed outputs, and the only directory left where **every** module is dead by both. The "reuses the EXACT v4 recipe" claim in v5–v9 is **prose in a docstring**, not an import. | The recipe constants are duplicated into v5–v9, so deleting v4 loses the origin comment, not a code path. But `test_recipe_parity_v5.py` compares regenerated rows against `data/v4/cache_manifest.jsonl` — that artifact is already gone (the test skips), so confirm you are not deleting the last description of a manifest nobody can rebuild. |
| **`v5/` (2 of 3), `v6/` (5 of 6)** | Dead by both, superseded by v7 → v8 → v9. | **Not the whole directory:** each one's `build_plan.py` is exec'd by its committed recipe-parity test. `v6/threshold_sweep.py` documents how the q3 operating point was set; move that reasoning to `docs/design/` before deleting. |
| **`eda/` (12)** | 0 / 0 across all 12. Closed one-off render studies + the finished scale-2×2 arc. | `scale_2x2_label_analysis.py` reads `labels/scale_2x2_labelset.json`, which stays. |
| **`julia_ladder/` (1), `readout/` (1), `descent_ablation/` (0 of 2), `reframe_probe/` (0 of 1)** | Single-purpose closed runs. **Only the first two are dead by both** — the counts here contradicted the index's own `∪` column until 2026-07-31 and now match it. `reframe_probe/speed.py` is **B-live**: it produced a committed artifact once (blind spot 2 — B cannot tell a one-shot from a driver that will run again), so "closed" is a judgement about the future, not something the numbers say. `descent_ablation/` is dead by both to the *rest of the tree*: its `A=1` edge is `run_campaign.py` importing its own sibling `finalize.py`, which the method counts and a retirement decision should not. | `descent_ablation/` has its own README explaining the campaign — retire the code, keep the README or fold it into `docs/design/`. Retiring a B-live module deletes the only committed description of how its artifact was made. |
| `studies/archive/` (6 of them dead by both) | Already explicitly an archive with its own README. | Six *other* `studies/` modules are frozen live dependencies. Do not retire the directory wholesale. |
| `atlas/` closed readouts (9) | Per-campaign reports (`campaign2_readout`, `dive_read_adjudicate`, `julia_fix_readout`, `confirm_report`, `v5_v6_anchor_diff`, …). Each answered one question, once. | `cross_family_shakeout.py` imports `step0_reanalysis` — retiring it does not free `atlas_probe/`. |
| `corpus/` closed builders (9) | rev4 / anchor / revisit batch builders; their batches are built and labeled. | `merge_amendments.py` is the amendment-overlay writer — verify no future revision batch needs it before retiring. |
| `render_mode_pilot/` (3 of 8) | 0 importers dir-wide; the 3 with no committed output are `build_sample`, `integrate_dataset`, `smooth_pass`. | The other 5 wrote `labels/render_mode_pilot_v1.json` and the mining-head dataset — retiring those makes the mining head's training set unrebuildable from committed code. |
| `wallpaper/` (2 of 19) | Only `family_entropy_trace.py` and `rerender_bootstrap_ss2.py` are dead by both. | The two builders that *look* retirable (`build_headbatch_dramatic`, `build_humanq3`) are pinned live by `test_builders_import.py`. |

**Explicitly NOT retirement candidates**, despite appearing on earlier candidate lists:
`atlas_probe/` (`step0_reanalysis` is on the production reward path), `studies/`
(13 frozen-by-design dependencies), `v7/` (2 of 4 live), `specs/` and `audit/`
(0 importers, both load-bearing), `explorer/` and `viz/` (launcher- and browser-driven).

## Known defects found while indexing

- ~~**`scoring/active_ckpt.py` carries the wrong module docstring.**~~ **Fixed 2026-07-31.**
  The production constants moved to `scoring/production_pins.py`; `active_ckpt.py` re-exports
  every name (its ~41 importers are unchanged, including `corpus/location.py`'s and
  `classifier/train_v9.py`'s by-path execs, which resolve the sibling off `__file__`) and
  keeps only the probe body. `tools/scoring/test_production_pins.py` pins the resolved values
  to their pre-split ones and asserts the re-export is the same object, not a copy. The stale
  usage lines and the `# "v7"` comment on a v8 pin went with it.
  **The probe half was NOT deleted** — `active_ckpt.select_anchors` is imported by
  `reframe_probe/speed.py` and `active_ckpt._unique_score3_locations` by the live
  `reframe/reframe.py`. Only the probe's own CLI (`main` and the sweep/render/sheet functions
  reachable only from it) is entry-point-dead: no `if __name__` block anywhere else invokes
  it and nothing subprocesses the path.
- **`hooks/` is gone (2026-07-31).** It held one tracked `pre-commit` that refused staged
  blobs over 1 MiB, and this checkout never installed it — so the recurrence guard was not
  guarding, and nothing said so except the line that used to be here. A client-side hook is
  the wrong instrument for a standing policy: it fires during ordinary work (training people
  to reach for `--no-verify`) and it is invisible when disabled. Replaced by
  `tests/test_large_tracked_blobs.py`, an allowlist of what may be large-and-tracked that
  reads the whole index instead of one staged blob.
- **`data_large/` was deleted in this pass**, leaving two files naming `data_large/…` paths
  whose corpus was already absent. `tests/occupancy_parity.rs` was **deleted on 2026-07-31**
  rather than repointed: it compared `energy::occupancy` against a persisted
  `complexity_scores.json`, and both the crops and the `score_complexity.py` that produced
  the reference numbers are gone, so there is no counterpart left to be parity *with*.
  `tools/viz/complexity_sort.html` still names them.

## Two standing facts about this directory's shape

Both are why the index above is hand-maintained rather than generated, so they belong here
rather than in a doc about storage or coloring.

- **There is no package root, and imports work by `sys.path` mutation.** `pyproject.toml`
  declares dependencies only — no build backend, no packages, no `src` layout — so every
  module is run *both* as `uv run python tools/<sub>/<mod>.py` and collected by pytest, and
  imports are position-dependent. `tools/_bootstrap.py` exists to centralize it, and covers
  **4 paths** (`tools/`, `palettes`, `corpus`, `queries`) with **3 importers**; its own
  docstring says the rest "still carry their own inserts — migrating them is a follow-up."
  The reasoning is sound — real packages would break `python tools/x/y.py` invocation — but
  the current state is the worst of both: a centralizing module almost nothing uses, and a
  convention every new file re-derives. This is also the direct cause of method A above
  being hard: there is no dotted name to resolve against.
  **Correction, 2026-08-02: there IS a dotted name, and it costs nothing.** A directory
  without `__init__.py` is a PEP 420 *namespace* package, so `from tools.v9 import
  build_plan` already resolves with the repo root on `sys.path` — no `__init__.py`, no build
  backend, no editable install, and `uv run python tools/x/y.py` keeps working (a script's
  own dir is still `sys.path[0]`). The bare-name convention is a habit, not a constraint.
  It is now **required** wherever the basename is ambiguous: `tests/test_import_hygiene.py`
  fails on any bare import of a name owned by more than one file, because those resolve by
  `sys.path` order and `sys.modules` first-write — 10 names and 22 call sites were doing so
  until that pass converted them. The remaining ~600 bare imports of *unique* names are an
  ergonomic tax, not a correctness risk, and are unchanged.
  `[measured: 305 of 439 tracked .py (69%) call sys.path.insert directly; _bootstrap has 3
  importers; 2026-07-31]` `[cmd: git ls-files '*.py' | xargs grep -l 'sys.path.insert' | wc -l;
  grep -rlE '^\s*(import|from)\s+_bootstrap' --include='*.py' tools/ | wc -l]`
- **The `tools/vN/` families are copy-forward, not parameterized — and that is a standing
  recommendation to retire.** `build_plan.py` exists **6 times** (`v4`…`v9`),
  `build_manifest.py` **4 times**, `render_cache.py` and `verify_cache_alignment.py` twice
  each. Crucially the reuse is *textual*: v5/v6 only **mention** `tools/v4/build_plan.py` in
  a docstring ("MUST match verbatim"), so nothing enforces the recipe parity the comment
  claims, and the next version adds a 7th copy. One parameterized driver taking the
  generation as an argument would replace the family. Unexecuted; inherited from the deleted
  `repo_structure_audit.md`, which is where it was first raised.
  `[cmd: git ls-files 'tools/v*/*.py' | xargs -n1 basename | sort | uniq -c | sort -rn]`

## Method

```bash
# file / subdir counts
git ls-files tools/ | awk -F/ 'NF>2{print $2} NF==2{print "(top-level)"}' | sort | uniq -c
# entry points (no importer by construction)
git ls-files tools/ | grep '\.py$' | xargs grep -l '__main__' | wc -l
```

Methods A and B were one-off analysis scripts, not committed machinery — deliberately, since
a liveness index is read by a human and decided by a human. **They no longer exist**, so
refreshing the counts means re-deriving them from the description above: resolve each import
to a *file* (dotted path first, then the importer's own directory, then its
`sys.path.insert` targets, then a `spec_from_file_location` pass); and extract path literals
under a committed root, keeping those that co-occur with a write call. Re-stamp the date at
the top of this file when you do. **The per-directory counts are the perishable part of this
document; the one-line-per-subdirectory index is the part worth maintaining.**
