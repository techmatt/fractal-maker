#!/usr/bin/env python
r"""Durability map: every path a committed tool writes, vs the class it declares.

WHY THIS EXISTS. Two misclassified artifacts surfaced incidentally inside a single task
(`data/emission/mining_gate_reports/`, the julia seed pool that was living at
`scratch/q4_decisive/julia_seed_pool.json`), on top of two wiped intake snapshots. That hit
rate says stumbling over them one at a time is the wrong method — so this enumerates the
whole surface at once and computes the mechanical half of the classification, rather than
anyone re-deriving it by hand per finding.

WHAT IT IS NOT. This is not `tools/audit/disk_audit.py`. That tool classifies what EXISTS on
disk by DELETION SAFETY (never / ambiguous / regenerable / scratch) to answer "what can I
reclaim". This one classifies what committed tools WRITE against the storage-class contract
in `tools/paths.py` (scratch() / bulk() / durable()) to answer "is this artifact's declared
class the one it actually needs". The two disagree in exactly the interesting places: a file
disk_audit marks NEVER is a bug if the writer's declared class is scratch(), because the
scratch contract GUARANTEES deletion.

THE MISMATCH TEST (the two conditions from the durability contract):
  (a) POPULATION-DEFINING — other measurements are quoted against it. A number computed over
      a population nobody can re-enumerate is not checkable, only repeatable.
  (b) NOT REGENERABLE from surviving inputs — re-running produces a DIFFERENT artifact
      (stochastic draw, GPU nondeterminism, or an input that no longer exists), not this one.
An artifact that is either, and is not durable(), is a MISMATCH.

MECHANICAL vs JUDGED COLUMNS. Everything git can answer is computed live at run time —
ignore status (`git check-ignore --no-index`, which unlike a bare check-ignore does NOT lie
about a force-added path), tracked count, in-tree and out-of-tree file counts, and what
`paths.durable()` would actually do with the path. The two columns git cannot answer —
regenerable-from-what and population-defining — are the registry below, and are judgements.

Inventory only. This writes a report and changes nothing.

  uv run python tools/audit/durability_map.py

Writes: scratch/durability_map/{report.txt,rows.jsonl}   (regenerable view)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths  # noqa: E402

ARTIFACTS = ROOT.parent / "fractal-maker-artifacts"
OUT = paths.scratch("durability_map")

# Classes as DECLARED (what the writer says / where it writes), not what it needs.
DUR = "durable()"            # data/ + a .gitignore negation: git keeps it
DUR_F = "durable(force-add)"  # tracked, but the path is gitignored: LFS force-add practice
BULK = "bulk()"              # ARTIFACTS_ROOT-relocated (out-of-tree, regenerable-by-contract)
SCR = "scratch()"            # scratch/: contract GUARANTEES deletion
UND = "undeclared"           # under data/ but gitignored and untracked: no class binds

Y, N = "yes", "no"

# path | writer | declared class | regenerable from what | population-defining | verdict
REGISTRY = [
    # ---------------- the corpus flow head ---------------------------------------
    ("data/guided_descend/<run>/pool.jsonl", "guided-descend (src/, Rust)", UND,
     "NOTHING — a stochastic guided descent; a re-run yields different locations", Y,
     "MISMATCH. CLAUDE.md's own flow starts here and every batch's provenance quotes a "
     "pool idx against it. run5 is gone in-tree AND out-of-tree; run4 is out-of-tree only. "
     "bulk() resolves it in-tree, where it does not exist."),
    ("data/guided_descend/<run>/{crops,fields}/", "present / enrich (src/, Rust)", UND,
     "the pool + the render core (pure function of the render block)", N,
     "OK as a class — regenerable bulk. Moot while the pool it needs is gone."),

    # ---------------- emission intake + decision records --------------------------
    ("data/emission/campaign1/intake.json", "tools/emission/campaign1_intake.py", DUR,
     "NOTHING — built from discovery scratch that has since been cleared", Y,
     "MISMATCH (realized). Class is correct on paper — negated, durable-eligible, "
     "disk_audit says NEVER — but it was never committed, so the declaration never bound. "
     "cluster_tags names WHICH clusters the seeded library draws against; absent, the "
     "scheduler seed is empty (this is what the unseeded-run guard now refuses to start on)."),
    ("data/emission/library_intake_2/intake.json", "tools/emission/library_intake_2.py", DUR,
     "NOTHING — same shape as campaign1 (incl. phoenix cluster tags)", Y,
     "MISMATCH (realized). Same as above: declared durable, never committed, gone."),
    ("data/emission/release_records/*.jsonl", "tools/emission/release_record.py", DUR,
     "NOTHING — the pool a release decision was taken against dies with --out", Y,
     "OK as of 2026-08-06. The slot stopped being empty: the post-flip run wrote 302 "
     "decisions + 1 population row, and they are now COMMITTED. Until that add they were the "
     "exact shape that lost campaign1/intake.json — declared durable at a negated path and "
     "untracked, where the declaration reads as coverage and binds nothing. Both stores are "
     "keyed by SITE, so a future run's file is a new PATH: covered by the relational half of "
     "tests/test_tracked_artifacts.py (every file present in the store must be tracked), not "
     "by the static canary list, which by construction cannot name a file that does not "
     "exist yet."),
    ("data/emission/mining_gate_reports/*.jsonl", "tools/mining/gate_report.py", DUR,
     "NOTHING — a would-cut log paired with the realized selections of one run", Y,
     "OK as of 2026-08-06 (this is one of the two that surfaced and was fixed). Same story "
     "as the row above: 240 gate verdicts from the post-flip run, untracked until now, "
     "committed and covered by the same relational guard."),

    # ---------------- classifier training population + weights --------------------
    ("data/v7/manifest.jsonl", "tools/v7/build_manifest.py", UND,
     "the label corpus (committed) + assign_split — but the v4/v5/v6 lineage it unions "
     "over is out-of-tree only", Y,
     "MISMATCH. This IS the training population and its split assignment; every v7 metric "
     "is quoted against it. Gitignored, and 0 files in-tree — data/v7/ is an empty dir."),
    ("data/v7/{plan,cache_manifest}.jsonl", "tools/v7/build_plan.py", UND,
     "the manifest + the render core, under a byte-identical recipe-parity gate", N,
     "MISMATCH-adjacent: regenerable in principle, but only from the manifest above, "
     "which is gone. Not population-defining itself."),
    ("data/v8/aug_cache/", "classifier/train_v*.py + tools/v8/build_plan.py", BULK,
     "full render compute under the recipe-parity gate", N,
     "OK. Correctly bulk() — the LIVE cache (171,384 tiles) relocated to ARTIFACTS_ROOT, "
     "and the reappearance tripwire (test_relocated_artifacts.py) holds it there. The "
     "v4..v7 caches were DELETED (commit 7068839) and their RELOCATED_PREFIXES literals "
     "dropped — dead machinery for caches that will never exist again."),
    ("data/classifier/v{5,6,7}/model_best.pt", "classifier/train_v{5,6,7}.py", DUR_F,
     "a full GPU retrain — which needs the v7 manifest above (gone)", N,
     "MISMATCH (fragile). Tracked via LFS force-add, but the PATH is gitignored, so "
     "durable() REFUSES it and every new sibling. (Until 2026-08-03 it gave a false OK on "
     "the existing file — `check-ignore` without `--no-index` answers about the index, not "
     "the rules; see tools/paths._is_gitignored.) The live discovery gate's weights are in "
     "a directory the contract cannot extend."),
    ("data/classifier/v7/eval_scores_v7.jsonl", "classifier/train_v7.py (eval-freeze)", UND,
     "a full GPU eval-freeze over the frozen eval slice — whose manifest is gone", Y,
     "MISMATCH. .gitattributes documents it as the frozen input the keeper-calibration "
     "gate derives and tests against, and says it must be force-added. It never was: not "
     "tracked, not on disk, in-tree or out."),
    ("data/wallpaper_head/v3/model_best.pt", "classifier/train_wallpaper_v3.py", DUR_F,
     "a full GPU retrain from the wallpaper corpus (committed)", N,
     "MISMATCH (fragile). Same force-add shape as the classifier weights."),
    ("data/queries/scorer/v3_gvo/model_best.pt", "tools/queries/scorer/*", DUR_F,
     "a retrain from the committed query labels", N,
     "MISMATCH (fragile). Same force-add shape."),
    ("data/render_mode_head/v1/model_best.pt", "tools/render_mode_pilot/*", DUR,
     "NOTHING — its training corpus (data/render_mode_corpus/dataset_v1/) is gone, so "
     "'retrain' would produce a different head, not this one", Y,
     "RESOLVED 2026-08-06. Was the force-add shape; .gitignore now carries an exact-path "
     "negation (!/data/render_mode_head/v1/model_best.pt), so durable() accepts it and a "
     "fresh clone's `git add -A` re-adds it. Reclassified population-defining as well as "
     "unregenerable: it is the LIVE gate AND the head whose suggestions seeded the 960 "
     "labels of the 2026-08-06 correction sheet, so every read on that sheet is quoted "
     "against this exact weight file."),
    ("data/render_mode_head/v1/mining_gate_lock.json",
     "tools/mining/lock_mining_gate.py --write", DUR,
     "re-derivable from data/render_mode_head/v2/report.json (pure Python, byte-identical) "
     "— but ONLY while that report survives; the sitting behind it is a GPU pass over crops",
     N,
     "OK. Negated by exact path. The frozen operating point the LIVE 0.50 release floor is "
     "set against; its reader (`read_lock`) refuses when the pin moves off v1. Its .md face "
     "is generated beside it and is a view of the same record, not a second source."),
    ("data/render_mode_head/v2/{config.json,metrics.json,per_seed.json,report.{md,json}}",
     "classifier/train_mining_head_v2.py, tools/mining/mining_v2_reads.py", DUR,
     "a full GPU finetune + a GPU eval pass — reproducible in recipe from the committed "
     "sheet + v1's weights, but not bit-identical (GPU nondeterminism)", N,
     "OK. Negated by exact path from the start rather than force-added — the wiring v9/v10 "
     "established. The WEIGHT (model_best.pt) was de-tracked 2026-08-06: v2 lost the winner "
     "rule, and a rejected candidate is not a critical final weight. The record it left "
     "behind is load-bearing in a way the weight is not — mining_gate_lock.json is derived "
     "from report.json. The per-seed dirs were always ignored."),

    # ---------------- calibration: the contract's own flagship --------------------
    ("data/calibration/energy_calibration.json", "src/energy.rs (`calibrate`)", DUR_F,
     "re-run `calibrate` over the 746-image corpus (the corpus must survive)", Y,
     "MISMATCH (fragile). CLAUDE.md names this THE durable persistent-store example and "
     "the frozen quantile bins ARE the metric's definition — yet the path is gitignored. "
     "It survives only by force-add; durable() refuses any new artifact in this dir."),
    ("data/calibration/{buffet_histograms,control_histograms,collision_distances,\n"
     "     palette_muster,rescore_{archetype,buffet,controls}}.json",
     "NONE — producers retired in the P2 subcommand cull", DUR_F,
     "NOTHING — the six scoring-experiment subcommands that wrote them are deleted", Y,
     "MISMATCH (orphaned). Committed by force-add, gitignored by rule, and 6 of the 7 "
     "have zero references anywhere in the repo — no producer, no consumer, no doc. "
     "Population-defining if anything ever quotes them; unrebuildable either way."),
    ("data/calibration/dedup_droplist.json", "palette_extractor/harvest_dedup.py", DUR_F,
     "re-run the dedup harvest over the palette cache (gitignored, local-only inputs)", Y,
     "MISMATCH (fragile). Its only consumer is outside tools/; inputs are the "
     "never-redistributable palette_cache/, so it is not regenerable on a fresh checkout."),

    # ---------------- discovery ledgers: the class that works ---------------------
    ("data/discovery/**/{outcome_ledger,probe_rejects}.jsonl", "tools/atlas/production_seeder.py", DUR,
     "NOTHING — append-only record of one run's admissions", Y,
     "OK. Negated, committed (18 tracked), disk_audit NEVER. The reference case."),
    # `<stem>.jsonl` here names the STREAM, not one file: since 2026-08-07 the four LFS
    # streams rotate into `<stem>.<nnn>.jsonl.gz` segments (tools/run_record.py) and a
    # finished run has no plain file at all. Read them with `run_record.iter_rows`.
    ("data/discovery/**/harvest_log.jsonl", "tools/atlas/steered_frontier.py", DUR,
     "NOTHING — re-deriving needs the cheap scorer AND canonical render over every "
     "candidate; a full re-derivation, not a ledger replay", Y,
     "OK. Explicitly negated + LFS-tracked (9 files). The tau_h curve's data on record."),
    ("data/discovery/**/prio_terms.jsonl", "tools/atlas/steered_frontier.py", DUR,
     "NOTHING — would require re-running the walk", Y,
     "OK. LFS-tracked (6 files). The steering-side counterpart to harvest_log."),
    ("data/discovery/**/summary.json", "tools/atlas/steered_frontier.py", DUR,
     "NOTHING — a run's realized totals", Y,
     "OK. 184 tracked. (Its library_seed stamp is new as of the unseeded-run guard; "
     "the campaign-2 summary predates it and cannot say whether it was seeded.)"),
    ("data/discovery/**/{state,dive_state}.json, morph_mem.npz, node_embs.npz,\n"
     "     scheduler_trace.jsonl, distinct_looks.npz, saturation.jsonl", "steered_frontier.py", UND,
     "rebuildable from the ledger + generator (crash-safety is by ledger-replay)", N,
     "OK. Deliberately ignored run-internal overlays, each with a .gitignore rationale."),
    ("data/discovery/**/scratch/", "steered_frontier.py", BULK,
     "nothing — completed-campaign discovery scratch, no named future use", N,
     "RECLAIMED. campaign2 breadth/dive scratch (317k files / ~46 GB) was deleted. "
     "steered_frontier.py now declares this class bulk() at its write site, and the "
     "resolver relocates any data/discovery/**/scratch by PATTERN (not a per-campaign "
     "literal), so a future campaign is born out-of-tree with no registry edit."),

    # ---------------- corpora + configs: the well-classified majority -------------
    ("data/label_corpus/batches/*/{images.jsonl,batch.json,scores.json}",
     "tools/corpus/*, tools/sourcing/build_*_batch.py", DUR,
     "NOTHING — human labels", Y,
     "OK. Negated by path, 71 tracked; crops/ and vivid/ correctly excluded as pure "
     "functions of the render block."),
    ("data/wallpaper_corpus/, data/q4_window_corpus/", "tools/wallpaper/*, q4_stage1_labelset.py", DUR,
     "NOTHING — human labels", Y, "OK. Same policy, negated and tracked."),
    ("data/minibrot_roster/{roster.jsonl,roster_cells.json,*/draw.jsonl}",
     "tools/sourcing/build_minibrot_roster.py, build_*_batch.py", DUR,
     "NOTHING — a per-(degree,band) atom selection + train/eval split later crops inherit", Y,
     "OK. Negated by path with an explicit rationale; 9 tracked."),
    ("data/palettes/*.json", "tools/palettes/build_*.py, tools/colormap.py", DUR,
     "rebuildable from palette_lib/build_sheet.py, but load-bearing downstream", N,
     "OK. Eight files negated individually by exact path."),
    ("data/atlas/{julia_seed_pool,morph_anchors,scheduler_prices,keeper_cuts,\n"
     "     guard_tripwire,mandelbrot_tgood_steered}.json, round{1,2}/",
     "tools/atlas/build_julia_seed_pool.py, steered_run2_report.py, ...", DUR,
     "the julia pool re-derives from a q4_decisive pass; the rest are fitted/config", Y,
     "OK — and this is the OTHER artifact that surfaced: the julia seed pool used to live "
     "at scratch/q4_decisive/julia_seed_pool.json and now has a committed producer writing "
     "it durably. Fixed already; listed so the map is complete."),
    ("data/test_{renders,palettes}.json", "hand-edited", DUR,
     "hand-authored config, not derived", N, "OK. Negated, committed."),
    ("data/emission/target_measure.json", "hand-edited", "DELETED",
     "the emission target is DERIVED from tools/scoring/release_mix.py at intake "
     "(cells.TargetMeasure.from_partition_shares); the file's nine hand-placed multipliers "
     "were a second policy about the same partitions", N,
     "GONE 2026-08-04. Not an artifact any more — no reader, no rebuild command."),
    ("data/library/library_records.jsonl, data/library_embeddings/embeddings.npz",
     "tools/wallpaper/library_records_build.py, tools/curation/colored_clip.py", DUR,
     "re-embed the library (deterministic given the records)", Y,
     "OK. Negated by exact path; the per-cycle shards/ overlay stays ignored on purpose."),

    # ---------------- families whose writers still exist but whose data does not ---
    ("data/enrich/run5/{scored,score_meta,selection_full}.jsonl",
     "tools/corpus/enrich_score.py, enrich_select.py", UND,
     "re-run enrich --mode score over the pool — which is gone", Y,
     "MISMATCH. The selection record for the ~1.1k enriched rows: which location won which "
     "palette, out of what scored set. Gitignored, and absent in-tree and out."),
    ("data/mining/run1/descent/pool.jsonl", "tools/mining/harvest.py", UND,
     "NOTHING — another stochastic descent pool", Y,
     "MISMATCH. Same shape as the guided-descend pool; gone in-tree and out."),
    ("data/ranker/{pref_loc_v0,campaign1}/{model.npz,features.npz}",
     "tools/ranker/build_features.py, report.py", UND,
     "re-derivable from the discovery ledgers (committed) + the corpus", N,
     "Regenerable, not population-defining -> the ignored class is defensible. Listed "
     "because the artifacts are absent, so nothing currently depends on them."),
    ("data/render_mode_corpus/batches/*/images.jsonl, rms_split_map.json",
     "tools/render_mode_pilot/*", UND,
     "NOTHING for the labels; the split map re-derives from them", Y,
     "MISMATCH, RESOLVED FORWARD 2026-08-06 — the loss stands, the hole is closed. Held the "
     "render-mode pilot+scale labels and their split map, ignored under the blanket /data/* "
     "rule with NO negation, unlike every other label corpus. Still absent in-tree and out, "
     "and data/render_mode_head/v1 was trained on it, so the 1500 tiers in "
     "labels/render_mode_{pilot,scale}_v1.json remain permanently orphaned — their ids are "
     "defined nowhere. The tree now carries an exact negation (!/data/render_mode_corpus/ "
     "with crops/, _fields/ and _progress_ledger.jsonl re-excluded) and a REBUILT corpus "
     "under it (see the two rows below), so the next batch cannot repeat this."),
    ("data/render_mode_corpus/gate_passers_v3.json",
     "tools/mining/build_gate_passers.py", DUR,
     "re-derivable EXACTLY: score the tracked 2026-07-09_wallpaper_headbatch_dramatic_v1 "
     "crops with the pinned wallpaper v3 head at p_ge3>0.90 (fp32), count-verified against "
     "the 401-row/112-location census the July samplers printed", N,
     "OK. The population the render-mode samplers draw from, which used to live at "
     "scratchpad/gate_passers_v3.json — the disposable tree — and went with the wipe. "
     "Regenerable, so not population-defining in the un-rebuildable sense, but committed "
     "anyway: its regeneration depends on gitignored crops, and it is 462 kB."),
    ("data/render_mode_corpus/batches/2026-08-06_render_mode_fresh_sheet_v1/"
     "{images.jsonl,batch.json}",
     "tools/mining/build_mining_sheet.py", DUR,
     "NOTHING once labeled — a human tier regenerates from nothing. The crops regenerate "
     "from each row's own render block + colour recipe (both in-row).", Y,
     "OK, and it is the fix for the row above: label slot, complete render block, mode + "
     "mode_params, colormap recipe and split_side are all in ONE tracked row, so an id can "
     "never again outlive its meaning. crops/, _fields/ and _progress_ledger.jsonl are "
     "gitignored by exact path — the rebuildable half."),
    ("data/queries/{coldstart_v2,warmstart_v1,prefv2_dramatic_v1}/{records/*.json,images/},\n"
     "     data/queries/scorer/v{1,2,3}/",
     "tools/queries/{assemble_queries,regenerate_coldstart_v2,prefv2_dramatic_v1}.py, "
     "tools/queries/scorer/train*.py", UND,
     "NOTHING — coldstart_v2's records are drawn from data/queries/coldstart_v1/records/, "
     "which is gone too; the scorer dirs need a retrain over the union those records define", Y,
     "MISMATCH. This is the PREF-HEAD JOIN, and it is the counterpart to the render_mode "
     "loss above with the halves swapped: there the labels went and the head survived, here "
     "the labels survive and their join went. The three label stores ARE tracked "
     "(data/queries/labels/*.json — 1,100 labeled passes over 1,000 queries) but they key "
     "their tiers by candidate id alone; the (location, palette, coloring params) each id "
     "names lives ONLY in records/, so the tracked labels are orphaned and the pref head "
     "cannot be retrained from them. launch_query_label_server exits on the missing dir, "
     "prefv2_dramatic_v1 reads these as 'durable query records' for its dedup exclusion, and "
     "scorer/data.py's documented one-flip rollback ladder (v3_gvo -> v3 -> v2) points at "
     "scorer dirs that do not exist — only v3_gvo survives, by force-add. Ignored under "
     "/data/queries/* with negations only for labels/ and sampler_eval/; absent in-tree "
     "and out."),
    ("data/curation/recolor_pass/, data/label_crops/loose0_v3/", "tools/curation/recolor_pass.py, "
     "tools/corpus/import_loose0_v3.py", UND,
     "re-render from the committed locations manifest", N,
     "OK as a class (regenerable crop feeds). Absent; label_crops has 1 file out-of-tree."),
    ("data/generated/loose0/{locations.jsonl,manifest.json}", "generate (src/) + import_loose0_v3.py",
     DUR_F, "re-run generate — a different draw, not this one", Y,
     "MISMATCH (fragile). It is the location list the loose0_v3 batch is keyed to — since "
     "2026-08-04 the mandelbrot EVAL FLOOR in batch_registry, not an unbiased-train entry "
     "(that category is now empty). Tracked by force-add at a gitignored path."),
    ("data/root_field/field_8192x8192_m1000*.{f32,json}", "dump-field (src/, Rust)", UND,
     "re-dump the field (deterministic, pure compute)", N,
     "OK. Ignored, untracked, 8 files in-tree — a genuine regenerable cache. Arguably "
     "belongs in bulk() rather than an ignored data/ path, but nothing is at risk."),

    # ---------------- scratch/ that is on a dependency path -----------------------
    ("data/minibrot_roster/interior_band_v1/cand/<atom>.json",
     "tools/sourcing/build_interior_band_batch.py", DUR,
     "re-sweep every position on 160 atoms' fields — a DIFFERENT reservoir draw, not this "
     "one (the full swept grid was never kept)", Y,
     "RESOLVED 2026-08-04. Was scratch/interior_band_batch/cand and was wiped; the copy "
     "that survived did so in a trash directory. Two committed tools read it "
     "(build_gcf_arm_batch draws the parked G_cf batch from it; the interior-clause "
     "inertness study quotes its argmax) and it IS the candidate population both sets of "
     "numbers are quoted against — impossible-class, so durable(). Restored byte-identical "
     "(160 files, sha256 over the set matches the trash copy) beside the manifest it "
     "defines, and durable() now asserts at write time that git will keep it."),
    ("data/minibrot_batch/fields/<atom>.bin", "tools/sourcing/build_minibrot_batch.py", BULK,
     "re-dump each atom's screen field (deterministic, expensive: 160 full-frame f64 renders)", N,
     "RESOLVED 2026-08-04. Was scratch/minibrot_batch/fields and was wiped (all 320 files). "
     "Regenerable, so scratch() was survivable ON PAPER — in fact it is the input three "
     "committed tools need (interior_band, interior_bakeoff, gcf_arm) and recovering it "
     "cost a trash directory rather than a re-dump. Registered in "
     "artifacts.RELOCATED_PREFIXES and restored out-of-tree byte-identical (1.6 GB / 320 "
     "files, sha256 over the .bin set and the .json set both match)."),
    ("data/q4_stage1/fields/<mb_id>.bin", "tools/studies/q4_stage1_labelset.py", BULK,
     "`q4_multibrot_transfer.py corpus-fields` — byte-identical, ~1.7 s/field, ~60 s for 33",
     N,
     "DELETED-BY-DECISION 2026-08-04 (Matt), and the RESOLVED-durable verdict it replaces "
     "was WRONG — recorded here because the error is the reusable part. That verdict "
     "re-dumped through the `beautiful` kernel (default bailout 2^16), got a constant "
     "+3.4712 offset on every escaped sample, and read that as irreproducibility. But the "
     "writer these fields actually come from is `--dump-field-source f64` "
     "(q4_multibrot_transfer._dump_field), and it reproduces them BYTE-IDENTICALLY: sha256 "
     "equal on 4/4 spot-checked files, identical NaN/interior mask, 100% of escaped samples "
     "exactly equal, re-dump sidecar bailout_b = 1e6 like the stored ones. On the real "
     "labeled windows the worst LF.featurize feature difference is 0.0 with zero _v2_drop "
     "disagreements — and that was never in doubt mechanically, since featurize "
     "percentile-stretches each crop by its own lo/hi, which cancels any constant offset "
     "exactly. THE LESSON: a reproducibility test must re-run the writer the artifact came "
     "from, not a plausible neighbour — this one compared against a kernel nothing in the "
     "path uses and cost 336 MB of LFS. The selection is recoverable too: HT.mb_info() "
     "derives all 33 ids from the tracked store data/q4_window_corpus/batches/ (33/33). So: "
     "untracked (.gitattributes LFS rule and .gitignore negation both removed), removed from "
     "disk, LS.FIELDS -> paths.bulk(). Local LFS objects deliberately NOT pruned. Every "
     "reader now funnels through LS._require_field, which raises with the rebuild command; "
     "a fourth hardcoded copy of the path in q4_richness_grid.py (missed by the original "
     "collapse, and dangling since) was collapsed onto LS.FIELDS at the same time. Holding "
     "copy of the deleted bytes: C:\\Code\\fractal-maker-holding\\q4_stage1_fields."),
    ("data/emission/campaign1/embs/*.npy", "tools/emission/campaign1_intake.py", BULK,
     "re-embed the intake's medoids — needs intake.json, which is gone", N,
     "Was scratch/emission/campaign1/embs and that is why campaign1 is DARK: the vectors "
     "were wiped and the snapshot went with them, so 'regenerable by contract' was false in "
     "fact. Repointed to a registered bulk family on 2026-08-03 so a relight cannot land in "
     "the deletable class again; nothing writes here today."),
    ("data/emission/library_seed_v2/embs/*.npy", "tools/emission/library_seed_v2.py", BULK,
     "re-embed the snapshot's 168 medoids from their own render blocks (~13 min, and "
     "verified BYTE-IDENTICAL on the 2026-08-03 regeneration)", N,
     "OK. This is the seed the scheduler actually resolves to. It was "
     "scratch/emission/library_seed_v2/embs — declared bulk() at the write site while the "
     "path itself said scratch/, so bulk() resolved it in-tree under scratch/ and the wipe "
     "took all 168. Registered in artifacts.RELOCATED_PREFIXES now, and "
     "deficit_scheduler._refuse_scratch_class refuses a scratch path at resolve time."),
    ("scratch/<subcommand>/** (renders, sheets, reports, logs)",
     "every subcommand + most tools/", SCR,
     "re-run the producing command", N,
     "OK. The convention holds — this is the large majority of write sites."),
]


def _ignored(rel: str) -> bool:
    """git check-ignore WITH --no-index: a bare check-ignore reports a force-added path as
    not-ignored, which is exactly the state we are trying to surface."""
    return subprocess.run(["git", "check-ignore", "--no-index", "-q", "--", rel],
                          cwd=ROOT, capture_output=True).returncode == 0


def _count(base: Path, rel: str) -> int:
    p = base / rel
    if not p.exists():
        return -1
    if p.is_file():
        return 1
    return sum(len(f) for _, _, f in os.walk(p))


def _durable_verdict(rel: str) -> str:
    try:
        paths.durable(rel)
        return "accepts"
    except paths.DurabilityError:
        return "REFUSES"


# Concrete probe paths for the mechanical columns (the registry keys use globs/braces).
PROBES = {
    "data/guided_descend/<run>/pool.jsonl": "data/guided_descend/run5/pool.jsonl",
    "data/v7/manifest.jsonl": "data/v7/manifest.jsonl",
    "data/classifier/v7/eval_scores_v7.jsonl": "data/classifier/v7/eval_scores_v7.jsonl",
    "data/calibration/energy_calibration.json": "data/calibration/energy_calibration.json",
    "data/emission/campaign1/intake.json": "data/emission/campaign1/intake.json",
    "data/minibrot_roster/interior_band_v1/cand/<atom>.json":
        "data/minibrot_roster/interior_band_v1/cand",
    "data/minibrot_batch/fields/<atom>.bin": "data/minibrot_batch/fields",
    "data/q4_stage1/fields/<mb_id>.bin": "data/q4_stage1/fields",
}


def main():
    rows = []
    for path, writer, cls, regen, popdef, verdict in REGISTRY:
        mismatch = verdict.lstrip().startswith("MISMATCH")
        probe = PROBES.get(path)
        rec = dict(path=path, writer=writer, declared_class=cls, regenerable_from=regen,
                   population_defining=popdef, verdict=verdict, mismatch=mismatch)
        if probe:
            rec.update(probe=probe, gitignored=_ignored(probe),
                       files_in_tree=_count(ROOT, probe),
                       files_out_of_tree=_count(ARTIFACTS, probe),
                       durable_call=_durable_verdict(probe))
        rows.append(rec)

    n = len(rows)
    nm = sum(1 for r in rows if r["mismatch"])
    L = ["=" * 100,
         "DURABILITY MAP — every path a committed tool writes, vs the class it declares",
         "=" * 100, "",
         f"{n} path families. MISMATCHES: {nm}.", "",
         "A mismatch = the artifact either DEFINES A POPULATION other measurements are quoted",
         "against, or CANNOT BE REGENERATED from surviving inputs — and is not durable().",
         "Classes: durable() | durable(force-add) = tracked at a gitignored path | bulk() |",
         "scratch() = contract guarantees deletion | undeclared = ignored, untracked.", ""]
    for r in rows:
        flag = "  <<< MISMATCH" if r["mismatch"] else ""
        L.append("-" * 100)
        L.append(f"PATH   {r['path']}{flag}")
        L.append(f"WRITER {r['writer']}")
        L.append(f"CLASS  {r['declared_class']}"
                 + (f"   [probe {r['probe']}: gitignored={r['gitignored']} "
                    f"in_tree={r['files_in_tree']} out_of_tree={r['files_out_of_tree']} "
                    f"durable()={r['durable_call']}]" if "probe" in r else ""))
        L.append(f"REGEN  {r['regenerable_from']}")
        L.append(f"POPDEF {r['population_defining']}")
        L.append(f"VERDICT {r['verdict']}")
    L += ["-" * 100, "", f"MISMATCH COUNT: {nm} of {n} path families.", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                    encoding="utf-8")
    txt = "\n".join(L)
    (OUT / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"rows   -> {OUT / 'rows.jsonl'}\nreport -> {OUT / 'report.txt'}")


if __name__ == "__main__":
    main()
