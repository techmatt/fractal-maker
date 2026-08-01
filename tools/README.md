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

**They disagree on 155 of 308 non-test modules (50%).** Neither is authoritative:

| | count | what it means |
|---|---:|---|
| A and B agree live | 52 | a module that is both imported and produces a committed artifact |
| **A only** | **89** | B alone would kill these — incl. `corpus/corpus_common.py` (**59 importers**), `corpus/location.py` (**57**, the canonical location identity), `mining/score_lib.py` (28), `corpus/label_store.py` (18). Shared libraries write nothing. |
| **B only** | **66** | A alone would kill these — incl. `specs/gen_registry.py` (0 importers; it generates `specs/REGISTRY.md`), `v9/eval_v9.py`, all 5 live `render_mode_pilot/` builders. |
| union live | 207 | |
| dead by **both** | 101 | the candidate-retirement pool ([below](#candidate-retirements)) |

**This block is one below the per-row sums**, and which side is right cannot be recovered:
summing the index's own `n / A / ∪` columns gives **309 / 142 / 208** against the 308 / 141 /
207 above, while `B` (118) and dead-by-both (101) agree exactly. That shape — off by one in
`n`, `A` and `∪` together, consistent everywhere else — reads as one A-live non-test module
counted in a row and missed by the summary, but the analysis scripts are gone (§Method), so
the discrepancy is **recorded rather than papered over by editing one number until it adds
up**. The two maneuver tools added 2026-07-31 make the row sums 311 / 142 / 208; the four v1.4 maneuver modules added later the same day (`atlas/maneuver_screen.py` + its test, `atlas/bench_neighborhood_subsumption.py`, `atlas/maneuver_inspection_sheet.py`) make them 315 / 143 / 209 — `maneuver_screen.py` is A-live (imported by `steered_frontier`), the other three are hand-run and write only `scratch/`, i.e. dead by both methods by construction (blind spot 1) exactly like the two before them. The six view-screen modules added 2026-08-01 (`atlas/view_screen.py` + `view_rescreen.py`, `view_screen_gate.py`, `view_frame_sweep.py`, `view_screen_sheets.py`, `test_view_screen.py`) make the row sums **321 / 144 / 210**: `view_screen.py` is A-live (imported by the other four), and the four drivers are hand-run — but unlike the four above, two of them write `data/atlas/` and are therefore **B-live**, not dead-by-both.

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
| `atlas/` | 48 | **The standing discovery flow.** `production_seeder.py` (26 importers) → `guided-descend` → reward; `steered_frontier.py` (classifier-steered descent), `guard.py` (degenerate-outcome gate), `deficit_scheduler.py`, `minibrot_maneuvers.py`, `prescreen.py`. Plus ~9 closed per-campaign readouts and the maneuver measurement tools added 2026-07-31 — `bench_lateral_seeding.py` (replay bench), `bench_neighborhood_subsumption.py` (is operator 2 subsumed by operator 3), `maneuver_degree_readout.py` (per-degree availability/cost) and `maneuver_inspection_sheet.py` (captioned richness sheets). `maneuver_screen.py` is different — it is imported by `steered_frontier.py` and is the field half of the richness screen. Both are hand-run and write only `scratch/`, so both are **dead by both methods by construction** (blind spot 1), which is what the pool is for. Plus the 2026-08-01 view-screen group — `view_screen.py` (the view-level composition screen; imported by the other four), `view_rescreen.py`, `view_screen_gate.py`, `view_frame_sweep.py`, `view_screen_sheets.py` — all hand-run, all **retroactive**: nothing in the discovery flow imports any of them, and `--maneuver-range-prior` still selects on the 4× atom score. | **live** | 37 / 12 / 20 / 26 |
| `atlas_probe/` | 5 | Step-0 atlas measurement probes (a closed study) — **but `step0_reanalysis.py` is the k3 reward primitive `production_seeder` imports** via an explicit `sys.path.insert`. Not retirable. | **live** | 5 / 3 / 2 / 4 |
| `audit/` | 6 | `size_guard.py` (the repo-size registry — exec'd by `tests/test_repo_size_guard.py` through a piecewise path the A pass cannot see), `disk_audit.py` (safe-delete classifier), `durability_map.py` (declared-vs-actual storage class). | **live** | 3 / 1 / 3 / 3 |
| `coevo/` | 2 | One guard-OFF v6-gap diagnostic round into `data/discovery/gather/`. Closed. | retired | 2 / 0 / 1 / 1 |
| `corpus/` | 40 | **The label-corpus contract.** `corpus_common.py` (row shape, 59 stem-refs), `location.py` (canonical location identity, 57), `label_store.py`, `corpus_reader.py` (the trainer's version-blind view), `merge_scores.py`, `artifacts.py` (the `ARTIFACTS_ROOT` resolver), `enrich_score.py`/`enrich_select.py`, `q4_window_reader.py`, + ~15 closed batch builders. | **live** | 32 / 12 / 14 / 23 |
| `curation/` | 4 | `colored_clip` palette-appearance descriptors + collision-aware palette placement (`colorize_assign`, soft-spread). | **live** | 4 / 3 / 2 / 4 |
| `descent/` | 14 | The minibrot **descent harness** and **triage wall** — two Flask apps + their durable stores (`data/descent_harness/`). Matt-driven; the recorded path is the product. | **live** | 8 / 7 / 5 / 8 |
| `descent_ablation/` | 3 | One overnight ablation + percentile-strategy campaign. Has its own `README.md`. Closed. | retired | 2 / 1 / 1 / 2 |
| `eda/` | 13 | One-off render/coloring eyeball studies (direct-trap sweeps, interior-trap validation, deband) + the whole closed scale-2×2 corpus arc. **0 of 12 live by either method.** | retired | 12 / 0 / 0 / **0** |
| `emission/` | 23 | **Diversity-aware emission v1**: `cells.py` (joint-count cells + target measure), `descriptor.py` (intake → morph cluster), `palette_deficit.py`, `pool.py`, `selection.py`, `release_record.py`, driven by `build_emission_diversity_v1.py`. | **live** | 16 / 12 / 6 / 13 |
| `explorer/` | 4 | The Flask Mandelbrot/Julia explorer + `render_core.py` (shared pixel→plane math, 13 importers, also used by the descent apps). `app.py` scores dead by both methods and is launched by `explorer.cmd`. | **live** | 2 / 1 / 0 / 1 |
| `julia_ladder/` | 1 | The J0 unlabeled Julia batch generator. One batch, produced and labeled; nothing imports it. | retired | 1 / 0 / 0 / **0** |
| `mining/` | 12 | The strange/render-mode quality gate: `mining_gate.py` (the pinned `mining_v1`), `score_lib.py` (the v3 deploy-transform scorer, 28 importers), `dedup.py`, `tail_alloc.py`, `deploy_tail.py`. | **live** | 10 / 7 / 2 / 7 |
| `orbital/` | 8 | The ring measures `radial_rings`/`radial_range` and the cap-policy stamping — owned by `docs/design/orbital_field_metrics.md`. | **live** | 6 / 6 / 5 / 6 |
| `palettes/` | 11 | The durable palette pool / features / k-cut categories, authored-palette densification, and the preview-sheet harness. | **live** | 10 / 6 / 3 / 7 |
| `phoenix/` | 12 | The phoenix seed sampler (`phoenix_sampler.py`, spec: `docs/design/phoenix_seed_sampler_spec.md`) + the closed Phase-B grid and its label analysis. Sampler live, grid closed. | **live** | 10 / 4 / 3 / 6 |
| `queries/` | 20 | The **palette-preference** corpus and scorer (distinct from the location classifier): `query_sampler.py` (20 importers), `scorer/` (v1 → v3-gvo; `data/queries/scorer/v3_gvo` is the deployed pref head), `query_label.html`. | **live** | 18 / 11 / 2 / 12 |
| `ranker/` | 8 | The location preference ranker, deployed head `pref_loc_v1` (`scorer.py`, `score_locations.py` — the consumer-side seam). | **live** | 6 / 3 / 3 / 6 |
| `readout/` | 1 | One morning diversity readout over an overnight emit manifest. | retired | 1 / 0 / 0 / **0** |
| `reframe/` | 2 | The promoted coarse-reframing step of discovery (`reframe_location`, 12 importers) — runs on every discovery survivor. | **live** | 1 / 1 / 1 / 1 |
| `reframe_probe/` | 1 | One coarse-reframe speed diagnostic. Closed. | retired | 1 / 0 / 1 / 1 |
| `render_mode_pilot/` | 8 | The 500 + 1000 render-mode label batches that produced the mining-head dataset (`labels/render_mode_pilot_v1.json`). **0 importers, 5 committed producers** — the clearest B-only block in the tree. Closed. | retired | 8 / 0 / 5 / 5 |
| `scoring/` | 4 | `production_pins.py` — **`ACTIVE_CKPT`, the single source of truth for the live classifier pin**, plus `PALETTE`/`JPG_Q`/`BIN`/`auto_maxiter`. `active_ckpt.py` re-exports all of it (~41 importers use that name) and is otherwise the retired reframe probe, two of whose helpers are still imported. | **live** | 2 / 2 / 1 / 2 |
| `sources/` | 7 | The minibrot **source sheets**: seven generation algorithms → `data/minibrot_sources/`, one HTML sheet per algorithm. | **live** | 6 / 5 / 1 / 5 |
| `sourcing/` | 11 | The durable minibrot **roster** (`build_minibrot_roster.py`), `deep_center_finder.py` (19 importers), and the live label-batch builders (`build_minibrot_batch`, `build_interior_band_batch`, `build_gcf_arm_batch`). | **live** | 7 / 6 / 5 / 7 |
| `specs/` | 1 | Generates `specs/REGISTRY.md` from `specs/modes_registry.json`; parity enforced by `cargo test --test modes_registry`. 0 importers — A alone would kill it. | **live** | 1 / 0 / 1 / 1 |
| `studies/` | 40 | 37 closed measurement passes (the q4 stage-1 arc, morphology dedup, descent score fidelity, …) plus `archive/`. **13 are frozen-by-design dependencies of live code** and must not be edited — `q4_stage1_linear_fit.py` (11 importers) is the deployed OOD mask. | **live** (frozen) | 37 / 13 / 7 / 17 |
| `v4/` | 6 | Origin of the 42-slot augmentation-cache recipe. **0 importers, 0 committed outputs — dead by both methods.** The recipe is carried forward by textual copy, not import. | retired | 6 / 0 / 0 / **0** |
| `v5/` | 4 | v5 manifest + plan (freeze v4, fold J0 Julia). Only `build_plan.py` is live, and only because `test_recipe_parity_v5.py` execs it by path. | retired (1 pinned) | 3 / 1 / 0 / 1 |
| `v6/` | 7 | v6 manifest + plan + the monitored-harvest drivers. Same shape: `build_plan.py` pinned by `test_recipe_parity_v6.py`, the rest dead by both. | retired (1 pinned) | 6 / 1 / 0 / 1 |
| `v7/` | 6 | Mostly closed — **except `build_manifest.py`** (imported by the live `sourcing/build_gcf_arm_batch.py` for `assign_split`) and **`eval_delong.py`** (the acceptance battery v8 and v9 both import). | **live** (2 of 4) | 4 / 2 / 0 / 2 |
| `v8/` | 11 | The shipped training generation: `data/v8/{manifest,plan,cache_manifest}.jsonl`, `derive_t_good_v8.py` (the **adopted** discovery table), `eval_v8.py`, the render supervisor. `ACTIVE_CKPT` = v8. | **live** | 8 / 3 / 7 / 7 |
| `v9/` | 9 | The raised-cap re-render generation (`docs/design/auto_maxiter.md`). Artifacts committed; `t_good` and keeper cuts derived but **STAGED, not adopted**. | **live** (staged) | 8 / 1 / 6 / 6 |
| `viz/` | 26 | 23 committed HTML inspection/label UIs — including **`corpus_label.html`, the labeling harness** named in `CLAUDE.md` — plus `serve.py` (the single-host static server for it) and two sheet builders. The `.py` side scores dead by both; the directory is not. | **live** | 3 / 0 / 1 / 1 |
| `wallpaper/` | 24 | The emission / location-library tail: `library_store.py`, `library_annotate.py` (12 importers), `prospect_orchestrator.py`, `emit_v1.py`, `pool_rule.py`, `label_crop.py`, `emission_selector.py`. Four batch builders are held live only by `test_builders_import.py`, a module-load smoke that exists because exactly this rot went unnoticed once. | **live** | 19 / 13 / 7 / 17 |

## Candidate retirements

**Nothing was deleted or archived in this pass** — this is the evidence, not the decision.
101 non-test modules score dead by *both* methods. Ranked by how safe the call looks:

| candidate | evidence | caveat |
|---|---|---|
| **`v4/` (6 files)** | 0 importers, 0 committed outputs, and the only directory left where **every** module is dead by both. The "reuses the EXACT v4 recipe" claim in v5–v9 is **prose in a docstring**, not an import. | The recipe constants are duplicated into v5–v9, so deleting v4 loses the origin comment, not a code path. But `test_recipe_parity_v5.py` compares regenerated rows against `data/v4/cache_manifest.jsonl` — that artifact is already gone (the test skips), so confirm you are not deleting the last description of a manifest nobody can rebuild. |
| **`v5/` (2 of 3), `v6/` (5 of 6)** | Dead by both, superseded by v7 → v8 → v9. | **Not the whole directory:** each one's `build_plan.py` is exec'd by its committed recipe-parity test. `v6/threshold_sweep.py` documents how the q3 operating point was set; move that reasoning to `docs/design/` before deleting. |
| **`eda/` (12)** | 0 / 0 across all 12. Closed one-off render studies + the finished scale-2×2 arc. | `scale_2x2_label_analysis.py` reads `labels/scale_2x2_labelset.json`, which stays. |
| **`julia_ladder/` (1), `readout/` (1), `coevo/` (1 of 2), `descent_ablation/` (0 of 2), `reframe_probe/` (0 of 1)** | Single-purpose closed runs. **Only the first two are dead by both** — the counts here contradicted the index's own `∪` column until 2026-07-31 and now match it. `coevo/analyze_round.py` and `reframe_probe/speed.py` are **B-live**: they produced a committed artifact once (blind spot 2 — B cannot tell a one-shot from a driver that will run again), so "closed" is a judgement about the future, not something the numbers say. `descent_ablation/` is dead by both to the *rest of the tree*: its `A=1` edge is `run_campaign.py` importing its own sibling `finalize.py`, which the method counts and a retirement decision should not. | `descent_ablation/` has its own README explaining the campaign — retire the code, keep the README or fold it into `docs/design/`. Retiring a B-live module deletes the only committed description of how its artifact was made. |
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
