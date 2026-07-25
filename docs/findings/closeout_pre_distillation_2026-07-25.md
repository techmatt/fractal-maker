# Closeout — pre-distillation settle (2026-07-25)

Verification-only unattended window. Two units: (§1) un-dormant the keeper calibration
gate by regenerating `data/classifier/v7/eval_scores_v7.jsonl`; (§2) exercise the
loud-late orchestrator surface pytest can't see.

## §1 — eval slice regen: STOPPED, guard stays dormant (correct outcome)

**Verdict: cannot regenerate `eval_scores_v7.jsonl` without a multi-stage pipeline
reconstruction, and no oracle survives to verify one. Per the prompt's explicit escape
hatch ("leaving the guard dormant one more cycle is the correct outcome — do not force
it"), §1 is deliberately left undone.**

The eval-*only* path is trivial (load the pinned `data/classifier/v7/model_best.pt`,
score the eval locations through `Transform(train=False)`, score v6 likewise, write the
11-column artifact — lines 388–400 of `classifier/train_v7.py` lifted out of `main()`).
It is **not** entangled with training. The blocker is the **input data is gone**:

The eval-freeze needs, per eval location: `label/source/fractal_type/group_id` +
canonical render JPG. That comes from the v7 **cache manifest**, which is deleted, and
its whole regeneration chain is deleted too:

| Artifact | Path | State |
|---|---|---|
| v7 cache manifest | `data/v7/cache_manifest.jsonl` | **GONE** (`data/v7/` empty) |
| v7 render plan / manifest | `data/v7/plan.jsonl`, `data/v7/manifest.jsonl` | **GONE** |
| v6 manifest (byte-parity oracle for build_manifest) | `data/v6/manifest.jsonl` | **GONE** |
| v6 cache manifest (byte-parity oracle for build_plan) | `data/v6/cache_manifest.jsonl` | **GONE** |
| aug roster | `data/v4/aug_roster.json` | **GONE** |
| all of `data/v4 .. data/v7` (plan side) | — | **empty dirs** |
| prior `eval_scores_v7.jsonl` / metrics.json (any copy) | git history, sibling artifacts | **not found** |

What **does** survive: the trained checkpoints (`data/classifier/{v6,v7}/model_best.pt`),
and the raw augmentation-cache JPGs in the sibling
`../fractal-maker-artifacts/data/{v4/aug_cache, v5/aug_cache_julia, v6/aug_cache_gather,
v7/aug_cache}` (3622 / 1000 / 639 / 536 location dirs). The canonical render per location
(`twilight_shifted__s1.0__shcenter__ss4.jpg`) is present on disk.

**Why no shortcut.** The census-144 eval slice could in principle be rebuilt from the
surviving `data/label_corpus/batches/` (15 batches present) + the canonical JPGs. But the
keeper gate (`test_keeper_derivation_calibration_gate`) asserts calibration over
`mandelbrot` and `julia:mandelbrot` (≥15 positives each) — those positives live in the
**frozen v6 eval split**, which only exists inside the deleted `data/v6/manifest.jsonl`.
So the full eval split (not just the post-freeze census) is required, and that means
re-running `build_manifest → build_plan → cache_manifest`, each of whose ABORT-level
byte-parity gates checks against an artifact that no longer exists. That is
"reimplementing the pipeline," not "an eval-only entry point."

**To actually revive this later** (out of scope for a verification window — it is a
retrain-adjacent data rebuild):
1. Restore or rebuild `data/v6/manifest.jsonl` + `data/v6/cache_manifest.jsonl` +
   `data/v4/aug_roster.json` (the frozen oracles), then
   `tools/v7/build_manifest.py` → `tools/v7/build_plan.py` to reproduce
   `data/v7/cache_manifest.jsonl`.
2. Add an eval-only freeze entry point to `classifier/train_v7.py` (extract the
   eval-battery + freeze tail behind an arg that loads `model_best.pt` instead of
   training) and run it. **Do not retrain**; the pinned checkpoint is the score source.
3. Force-add the LFS pointer (`.gitattributes` rule already present), verify pointer +
   >1 MiB hook, then the dormant gate executes.

The dormant test already behaves correctly: it `pytest.skip`s loudly with the regen
command rather than crashing, so it is visible in the summary and never silently passes.
No change was made to it.

## §2 — loud-late orchestrator surface

Four entry points, capped smoke runs. **3 GREEN, 1 RED (blocked on a missing
regenerable input, not a code regression).** Logs under `out/smoke_closeout/` and each
run's own `orchestrator.log`.

| Entry point | Verdict | Evidence |
|---|---|---|
| `production_seeder --smoke` | **GREEN** | exit 0, 355s, 3 batches. Seed rejection fired (draws 142, rejected 106 = 74.7% — the smoke's stated purpose), probe-rejected 5. Scorer = v7 `model_best.pt`. 1 distinct q3 harvested, guard telemetry clean (19 clean / 12 salvaged / 0 dropped), saturation false. `data/discovery/runs/20260725_085438/summary.json`. |
| `overnight_orchestrator --mini --cap-hours 0.25` | **GREEN** | exit 0, "RUN COMPLETE: **2 cycles, 4 wallpapers, 21 fresh q3, 0 phase failures**". Every phase rc=0 (discovery → pool → present/emit). Emission gate exercised: `[gate] p_ge3 > 0.05: 12/24 pass -> [select] 2 emitted`. julia-hook active. Scratch auto-purged on exit. |
| `prospect_orchestrator --mini --cap-hours 0.25` | **GREEN** | exit 0, "RUN COMPLETE: **2 cycles (0 failed), +7 library records, 15 fresh q3, 0 phase failures**". All 6 phases rc=0 (discovery/pool/annotate ×2). The prospect-distinctive **library-store + embedding sink** ran: 7 records, 2 embedding shards (69 vecs), 7 thumbs, field-cache written. Reconciliation clean: `q3_found=15 = records=7 + coord_dup=8 + field_fail=0 + unexplained=0`. |
| `steered_frontier` smoke | **RED — blocked on missing input** | exit 1, immediately: `missing C:\Code\fractal-maker\out\descent_score_fidelity_records.json — run tools/studies/descent_score_fidelity.py`. `derive_tau_h()` (steered_frontier.py:198) hard-`SystemExit`s when the fidelity records are absent, before any GPU/render work. **This is not a structural regression** — the module imports cleanly and reaches τ_h derivation; the τ_h fidelity-records input was wiped along with the rest of `out/`/`data/`. Regenerating it means running the `descent_score_fidelity` study (renders + dual-scorer fidelity pass), which is out of scope for a verification window. Not fixed (fix is not "small"); recorded. |

**On the recent structural changes named in §2.** All exercised except where noted:
- **τ_h campaign floors** — `TAU_H_CAMPAIGN_FLOOR` (steered_frontier.py:191, mandelbrot/mb3/mb5;
  mb4 deliberately absent, matching commit c879da3) is applied inside `derive_tau_h`, which
  the steered smoke could not reach past due to the missing fidelity records. **Not
  exercised at runtime this window** — but the dict is present and the pure keeper/priority
  math is covered green by `tools/atlas/test_steered_frontier.py` in the pytest suite.
- **report-only mining gate + `tools/mining/gate_report.py` writer** — these live on the
  **emission/mining side** (`tools/mining/deploy_tail.py`, `build_emission_diversity_v1.py`),
  not on any of the four discovery orchestrators, so this smoke set does not touch them. The
  emission *selection* gate (`emit_v1`) WAS exercised green inside overnight (`[gate] … ->
  [select] 2 emitted`). If the mining-gate report path specifically needs loud-late
  coverage, `deploy_tail` is the entry point to smoke — flagged for a follow-up window.
- **harvest-log tracking + guard telemetry** — exercised green in all three passing runs
  (`guard telemetry: clean/salvaged/dropped` lines, `outcome_ledger.jsonl` written).
- **aug_cache tripwire refactor** — this one has a **committed pytest test**
  (`tools/audit/test_aug_cache_write_path.py`), so it is NOT loud-late; the standard suite
  covers it. Out of scope for the four-orchestrator smoke.

## Cross-cutting note — the working corpus under `data/` and `out/` was cleared

Both §1's blocker and the steered_frontier RED trace to the same root cause: the disposable
`out/` tree and large parts of the (committed) `data/` tree are empty in this checkout —
all v4–v7 manifests/plans/cache manifests, `aug_roster.json`, and
`out/descent_score_fidelity_records.json` are gone. The **checkpoints**
(`data/classifier/{v6,v7}/model_best.pt`) and the **sibling aug_cache JPGs**
(`../fractal-maker-artifacts/…`) survive, which is why the render/score path smokes green
while every "load a frozen derived artifact" path is blocked. This is a data-state issue,
not a code regression — the code paths that could run, ran clean.

