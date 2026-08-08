# Storage classes — the durability contract

Every file this project writes belongs to exactly one durability class. The class is
declared **at the write site** through `tools/paths.py`, and the choice is binding:
it decides whether the file is expected to survive, and therefore whether its absence
is ever a problem. This note is the contract those functions enforce. It is rules, not
commentary.

**Boundary — this is one half of a pair.** This doc owns the **contract**: which class an
artifact belongs to, and why. [`artifacts_resolver.md`](artifacts_resolver.md), named for
`tools/corpus/artifacts.py`, owns the **mechanism**: the `ARTIFACTS_ROOT` resolver and its
seam, the size-guard registry, the reappearance tripwire, the LFS / `.gitignore` wiring,
the derived canary set, and the small-tree and docs guards. Read this one at a write site;
read that one when changing the wiring. Where they touch, this doc states the rule and
that one states what enforces it. `[verdict: Matt]`

**Tags.** `[code: path]` — true because the tree says so. `[measured: population]` — a
number, with the population it is true *of*. `[verdict: who]` — a judgement call.
`[unverified]` — supplied from outside the repo and not checkable here.

## The first question is usefulness, not cost

The contract's question — *what would it cost to get the identical content back?* — is
only ever the **second** question. The **first** is whether we would ever want it back
at all. As written, the rules below weigh preservation by rebuild cost, which quietly
implies that anything expensive to reproduce deserves to be kept. It does not. A recipe
for regenerating something no one will ever ask for is itself clutter — and keeping the
apparatus (index, manifest, plan, registry line) for such a thing is the specific
failure this note exists to stop. `[verdict: Matt]`

So classify in this order:

1. **Name a concrete future use.** Not "this could in principle be regenerated," but
   what will actually want it. If none can be named, **delete it — and delete its
   regeneration machinery with it**: the index, the manifest, the recipe, the registry
   line. Keeping the machinery for reproducing something nobody will run again is the
   clutter, not the saving.
2. **Only then** apply the cost question. If a use *can* be named and the thing records
   a population that no longer exists and cannot be re-observed (labels, current-decoded
   ledgers, the library), keep it. If it is regenerable and something will genuinely want
   it again, keep the **minimum** needed to regenerate and state that cost.

The one safety rail: for **unclassified** items — no write site, no declaration — do not
delete silently, but do record a judgement of whether anything will ever want it, so the
set can be resolved in one pass instead of sitting open.

**This rule is applied, not just stated.** Nine size-guard registry lines were deleted on
2026-07-31 rather than left as fossils, on exactly this test — nothing will ever write
there again, so the line goes with the data.
`[code: tools/audit/size_guard.py, the prune note in REGISTRY]`

## The classes

| class | writer | lives in | survives `rm -r scratch/*`? |
|---|---|---|---|
| **scratch** | `scratch(...)` | `scratch/` (gitignored) | no — deletion is the design |
| **bulk** | `bulk(rel)` | out-of-tree via `ARTIFACTS_ROOT` | rebuilt on demand |
| **durable** | `durable(rel)` | `data/` (git-tracked) | yes — git keeps it |
| **vendored** | committed by hand | config/code, with provenance | yes — it's source |

`[code: tools/paths.py]`

## Rules

1. **`scratch/` is ephemeral by definition.** Deleting it — in whole or in part, at
   any time — is expected, routine, and correct. It requires no warning, no backup,
   and no notice to anyone. A tool that writes there is stating that its output is
   cheap or free to rebuild and is not worth keeping.

2. **A missing scratch artifact is not data loss.** Everything that historically lived
   under `out/` was scratch by construction — that is what the directory meant, and it
   has since been renamed to `scratch/` so the name can no longer imply otherwise. The
   absence of any such file is a designed outcome, not a loss, and **must not be
   described as data loss** in code, commits, docs, or tasks.
   `[code: tests/test_no_out_dir.py]`

3. **Never open a recovery, reconstruction, or investigation task against vanished
   scratch contents.** There is nothing to recover: the class guaranteed deletion in
   advance. If a specific use case genuinely needs an artifact to persist, the answer
   is to declare it `durable()` at its write site **going forward** — prospectively,
   for future runs. Never retroactively, and never by trying to resurrect what a
   previous run discarded.

4. **Durability is claimed only by `durable()`, and only for a git-tracked path.** No
   module may treat a `scratch/` (or otherwise gitignored) path as durable. `durable()`
   is the sole way to assert that a file must survive, and it verifies the claim at
   write time: if the path would be gitignored, it raises `DurabilityError` on the spot
   rather than letting the write succeed and the file silently disappear later.
   `[code: tools/paths.py::durable; tests/test_storage_classes.py]`

5. **No large training data or results inside the `fractal-maker` folder — even if
   gitignored.** Bulk regenerable data lives out-of-tree, under the artifacts root
   (`ARTIFACTS_ROOT`, default `../fractal-maker-artifacts`), reached through
   `paths.bulk(rel)` → `tools/corpus/artifacts.resolve`. **The reason is tool and
   `grep` traversal cost, not repo size.** A gitignored directory is invisible to
   `git status` and fully visible to every recursive walk — `grep -r`, `find`, editor
   indexers, file watchers, and this agent's own search tools. Two file-count bombs
   (243k aug-cache JPGs, 317k discovery-scratch files) once made a plain `grep -r` take
   over two minutes; being ignored bought nothing, because none of those tools consult
   `.gitignore`. So "it's ignored" is not a defense, and neither is "it's small on
   disk": a million 4 KB crops cost more traversal than one 10 GB blob.
   `[measured: the two named caches, pre-relocation]`

   The corollary for a *new* bulk family is that it must be born out-of-tree — declare
   it `bulk()` at the write site and register its prefix (or, better, match it as a
   *class* by pattern) **before** the first run, not after it has already materialized
   170k files in the tree. `[code: docs/design/artifacts_resolver.md §3]`

6. **`bulk()` bounds *where*, not *how much* — a producer of per-run bulk owns its own
   teardown.** Nothing guards the artifacts root's size: `tools/audit/size_guard.py`
   walks `REPO_ROOT` only, and rule 5 above exists to push bulk out to exactly the place
   no registry, threshold or test looks. A discovery run's scratch is ~18 GB per
   engine-hour, so on 2026-08-07 two same-day `steered_frontier` runs were 154 GB — 86% of
   a 178 GB store — while their conclusions already sat in the tracked
   `outcome_ledger.jsonl`. `steered_frontier` therefore deletes its own scratch subtree on
   a **clean close** (`--retain-scratch` opts out) and stamps the outcome into
   `summary.json`. Two properties are load-bearing and both are guarded: teardown hangs off
   the summary write and **nothing else** — no `finally`, no `atexit`, no signal handler,
   so an interrupted run keeps the intermediate state you may still need to read; and the
   summary lands *before* the delete begins, so a kill mid-teardown is distinguishable
   (`outcome="not_reached"`) from a run that predates the feature.
   `[measured: 2026-08-07, scratch/artifacts_audit_report.md; code:
   steered_frontier.SCRATCH_TEARDOWN_KEY, tools/atlas/test_steered_frontier.py]`

## `scratch/` is about liveness, not just recovery cost

Rules 1–3 read as if cheapness of rebuild were the test. It is not the whole test. A
thing can be free to regenerate and still be **load-bearing right now**, and the class
says nothing about that.

**`scratch()` is for what a *finished* process leaves behind.** Live state — anything a
later step in the same piece of work depends on — belongs in memory, or in a named cache
under `ARTIFACTS_ROOT`, not in a tree whose contract is "deleting this at any time is
correct and requires no notice". An approved render staged in `scratch/` between rendering
it and committing it is a violation of this contract even though re-rendering costs
seconds: the cost of the loss is not the render, it is the approval. `[verdict: Matt]`

**This is not hypothetical — the descent harness does exactly it, today.** This clause was
carried as `[unverified — no committed code path does this staging today]`; that was wrong,
and the correction is the whole point of the clause. `tools/descent/app.py` stages the
approval in `scratch/` across a human-length window:

1. `POST /quality` renders the label-crop-quality canonical + vivid pair to
   `scratch/descent_harness/quality/` (`store.QUALITY_DIR`).
2. Matt looks at the vivid companion and decides. **That judgement is the artifact**; the
   render is worth seconds and the judgement is worth the session.
3. `POST /emit` `shutil.copyfile`s those two files out of `scratch/` into the durable
   store — the comment there reads *"approve == saved, no re-render"*, which is precisely
   the guarantee `scratch/` does not offer.

The session dict holds only the two **paths**, not the bytes, so nothing can re-supply
them. `/emit` validates that the *session* entry exists but never that the *files* do, so
a `rm -r scratch/*` between steps 1 and 2 does not produce the harness's own
`"quality render missing; re-render"` 400 — it produces an unhandled `FileNotFoundError`
from `copyfile`, i.e. a 500 with the approval lost. The class is still `scratch()` in the
tree; correcting that is a harness change, not a doc change, and is not done here.
`[code: tools/descent/app.py::{quality,emit}; tools/descent/store.py::QUALITY_DIR]`
`[verified 2026-07-31 by reading both handlers]`

**Nothing load-bearing may live in `scratch/`, and it fails in two directions.**

- *Evidence must leave the moment it justifies a durable decision.* The 32-atom maxiter
  convergence ladder — the whole basis of the production cap's base 500 → 4000 raise — sat
  in `scratch/rescore/converge.json` with its producer, one `rm -r scratch/*` from gone. It
  was promoted to `data/orbital/maxiter_convergence_ladder.json` and canaried on
  2026-07-31. `[code: data/orbital/maxiter_convergence_ladder.json `promoted_from`;
  tests/test_tracked_artifacts.py]`
- *A proposal must never leave as a fact.* `converge.py` fitted a 24× / clamp-67000
  "scoring envelope" to `scratch/rescore/scoring_cap.json`; nothing outside that scratch
  directory ever read it, and it was nonetheless carried as a stated property of the
  system for two checkpoints. The function that would have loaded it had no caller and, in
  its absence, returned a different number entirely. It was deleted on 2026-07-31.
  `[code: tools/orbital/rescore_lib.py module docstring; scratch/rescore/scoring_cap.json]`

The asymmetry to remember: the first failure loses something real; the second manufactures
something that was never real. Both come from the same habit of treating `scratch/` as a
place where work lives rather than a place where finished work is discarded.
`[verdict: Matt]`

## Reproducible-on-demand is the KEEP test, and it is stricter than "the producer exists"

A surviving producer is not reproducibility. Two failure shapes, both live in this tree:

- **Wall-clock-budgeted resumable production.** `data/orbital/screen_pool.jsonl` holds the
  Newton-solved atom enumeration, produced under `--enum-budget` (default 1800 s) with
  resume. The output is a function of **how long the run got**, not of the seed — so the
  exact atom set is not reproducible even though `screen_pool.py` is right there. This is
  half of why the file is a registry `KEEP`.
  `[code: tools/orbital/screen_pool.py `--enum-budget`; tools/audit/size_guard.py REGISTRY
  "data/orbital/"]`
- **A measurement whose producer reads live constants.** The convergence ladder's producer
  survived, but every ratio in it was a multiple of the *then*-production cap, and the cap
  has since moved — so re-running it measured the new policy and reported ratios near 1. A
  different quantity wearing the same producer. (The producer now takes the policy as a
  parameter defaulting to live, which makes the legacy measurement repeatable; the
  committed artifact is unchanged and still stamped legacy.)
  `[code: tools/orbital/measure_convergence_ladder.py::{POLICY_LIVE, POLICY_LEGACY,
  policy_maxiter}; data/orbital/maxiter_convergence_ladder.json
  `not_reproducible_under_current_policy: true`]`

- **A chain, where the producer of each link needs the link before it.** The third shape, and
  the most expensive one this tree has actually paid. `data/v{4,5,6,7}/` were classed
  regenerable at the 2026-07-24 migration on exactly the "the producer exists" reasoning, and
  cleared. They are not regenerable: `build_manifest → build_plan → cache_manifest` each carry
  an **ABORT-level byte-parity gate against the previous version's artifact**, so rebuilding v7
  requires v6's manifest and cache_manifest, which require v4's `aug_roster.json`, which is also
  gone. What that cost concretely: `data/classifier/v7/eval_scores_v7.jsonl` cannot be
  regenerated, because the eval freeze needs the v7 cache manifest and the full frozen v6 eval
  split. The eval-only *code* path is trivial and was never the blocker — the inputs were. The
  gate that depended on it was retired in favour of one that guards the committed constant
  directly, with no dead-machinery input. **A per-file regenerability judgement is wrong for a
  chain; classify the chain.** `[measured: 2026-07-25 — data/v4..v7 empty, aug_roster.json and
  both v6 oracles absent, no prior eval_scores_v7 copy in git history or siblings]`

So the KEEP question is not *does something still produce this*, it is *would running it
again produce this*. `[verdict: Matt]`

### The v8 `plan` + `cache_manifest` deletion — TAKEN 2026-08-03

`data/v8/{plan,cache_manifest}.jsonl`, **139.6 MiB** (53,391,652 + 92,985,596 B), one row per
augmentation tile, are **gone**. What follows is the record of the preconditions and how each
was met; v9's and v10's pairs are NOT in the same position and are covered at the end.

**The rebuild is byte-identical, and that was measured, not argued.** `uv run python
tools/v8/build_plan.py` (~15 s) regenerates the pair from `data/v8/manifest.jsonl` plus the
committed colormap sources. Proved by rebuilding over the originals and comparing sha256:
`plan.jsonl`, `cache_manifest.jsonl`, `colormaps.json` and `aug_roster.json` all matched.
`[measured 2026-08-03; the one file that did NOT match is `build_metadata.json` — see the
frozen-measurement hazard below]`

**A rollback-to-v8 cache rebuild is now two steps**, `build_plan.py` then
`render_cache.py` (~4.7 h at 6 workers, unchanged) — the plan regeneration adds 15 s to a
4.7-hour job. `data/v8/aug_cache` stays in `RELOCATED_PREFIXES` for exactly that reason.

**The pair is `bulk()`, not `durable()` — the deletion moved its class.** Deleting the files
took their `.gitignore` negations with them ("the negation goes with the file"), and
`durable()` asserts its target is not ignored — so from that commit `build_plan.py` raised
`DurabilityError` on `plan.jsonl` and **the rebuild this deletion rests on could not
complete**. A byte-reproducible artifact that is deliberately untracked is `bulk()`;
`data/v8/plan` is not a `RELOCATED_PREFIXES` entry, so it resolves in-tree at the same path
every reader already opens, merely untracked. `[fixed 2026-08-03; rebuild re-verified:
plan.jsonl 53,391,652 B and cache_manifest.jsonl 92,985,596 B, both matching the byte counts
recorded above, then deleted again]`

**The frozen-measurement hazard is fixed.** `tools/v8/build_plan.py::amend_metadata` used to
rewrite `data/v8/build_metadata.json` unconditionally, and a plain re-run set
`aug_recipe.marginal_cost` to `null` — the committed palette-vs-geometry timing measurement,
which only `--measure-marginal` produces. A plain rebuild now **carries the committed value
forward** (`carry_marginal`), per target: `build_metadata.json` and `aug_roster.json` hold
different values and each keeps its own, so a no-flag rebuild leaves both byte-identical.
Only `--measure-marginal` replaces the measurement. `data/v9/build_plan.py` carries the same
guard on `data/v9/build_metadata.json`, whose committed value is `null` — the guard is there
because v9 is permanently staged, so a re-run could only ever destroy the record of what was
staged. `[fixed 2026-08-03, bracketed red and green; tools/audit/test_frozen_record_writes.py.
Same shape as the derive_t_good_v8/v9/v10, keeper_cut_v9 and prereg hazards, all fixed by
making the durable write take an explicit flag]`

#### The preconditions, as they stood before the deletion

**(a) Make the guards hard-fail, naming the rebuild command.** The two `.exists()` guards
that read these files are both in `tools/v9/build_plan.py::assert_recipe_parity` — the
v8-plan comparison and the v8-cache-manifest recipe-field comparison. Each is the
**load-bearing check of the whole v9 rebuild**: a recipe that drifted by one seeded draw
would make v9's corpus non-comparable to v8's, and the v9-vs-v8 eval bar would be measuring
the drift. Absent the input, both simply do not run and the rebuild reports success. They
must raise, naming `tools/v8/build_plan.py` (which rebuilds v8's pair) and
`tools/v9/build_plan.py` (v9's). Note the chain: deleting **v8's** pair is what disarms the
gate on a **v9** rebuild.

**(b) DELETE the alignment tests whose referent is gone; do not let them skip.**
`tools/v8/test_v8_cache_alignment.py`'s module fixture `pytest.skip`s if any of the four
v8 artifacts is absent, so BACKWARD / FIELDS / COUNTS would go quietly green-by-absence.
**Keep the census test** (`test_the_eval_slice_holds_the_full_144_location_census`) with a
fixture narrowed to `eval_slice.jsonl`, which is small, plain-text and not part of the
deletion. `[code: tools/v8/test_v8_cache_alignment.py::v8]`

**[RE-CHECKED 2026-08-02, after the v10 cache extension] Still blocked — and the set shrank
from four files to two.** v10 does not rebuild v9's cache; it **extends** it (7,115 prefix
locations keep naming `data/v9/aug_cache`, 1,267 appended ones render into `data/v10/`), and
the legitimacy of reusing those 170,760 tiles rests on GATE A in
`tools/v10/build_plan.py::assert_recipe_parity`: every prefix plan row must be byte-identical
to its **v9** row. So the pair that arms the live recipe-parity gate is now **v9's**, not
v8's. **It is no longer a deletion candidate at all**: it is the referent the current
training generation is verified against. `data/v9/cache_manifest.jsonl` (96 MB) is read only
by `train_v9`/`eval_v9`/v9's own `verify_cache_alignment`, all v9-scoped.

**[SUPERSEDED 2026-08-08 — the v9 pair is DELETED and the gate is RETIRED.]** See the next
section. One correction to the paragraph above, because the miscount is the reusable part:
it said `data/v9/plan.jsonl` had **three** non-absence-tolerant readers. It had **eight** —
`tools/v10/{build_plan (the gate),prereg,verify_cache_alignment}.py`, the `slow`
`test_v10_build.py::test_prefix_plan_rows_are_byte_identical_to_v9s`, and
`tools/v9/{render_cache,verify_cache_alignment,measure_cap_effect,estimate_cap_cost}.py`.
The three were the readers of the *gate's* concern, not the readers of the path; the
one-line grep that finds all eight is the same one the 2026-08-02 re-check ran for v8.

**[2026-08-03] Both preconditions met, plus one the list did not name.**

  * **(a) met.** All THREE `.exists()` comparisons in
    `tools/v9/build_plan.py::assert_recipe_parity` (the colormap library as well as the two
    the list named) now route through `_require_v8`, which raises naming the missing file and
    `tools/v8/build_plan.py`. Proved red by hiding `data/v8/plan.jsonl`: exit 1, no plan
    written. `tools/v9/test_recipe_parity_guard.py` pins the raise, pins that no `.exists()`
    guard returns to that function, and pins that the paths it is armed with exist.
  * **(b) met.** BACKWARD / FIELDS / COUNTS are DELETED from
    `tools/v8/test_v8_cache_alignment.py`, not skipped, and its fixture is gone — the census
    test reads `eval_slice.jsonl` directly and asserts its presence rather than skipping on it.
  * **(c) the one the list did not name: the coverage derivation.** `test_tracked_artifacts.py`
    derives its guarded set FROM the `.gitignore` negations, so removing two negations shrinks
    the parametrization — and a smaller parametrization is a quieter run, not a red. Removing a
    negation while leaving its file would therefore have silently dropped coverage. Now
    asserted from git's side too: every TRACKED file under a versioned build tree must carry an
    exact-path negation or be one of the three known force-added classifier weights
    (`test_every_tracked_build_artifact_is_covered_by_a_negation`, proved red by deleting a
    negation whose file remains). Guarded set went 36 → 34, LFS rules 9 → 7, both accounted.

`[re-checked 2026-08-02 at this commit: `rg -n "v8/plan|v8/cache_manifest|v9/plan|v9/cache_manifest" --glob '*.py'` over tools/, classifier/, tests/, then reading each reader for absence tolerance. Re-run 2026-08-03 before the deletion: the eight remaining readers of v8's pair — `tools/v8/{render_cache,dump_fanout,estimate_runtime,verify_cache_alignment,eval_v8}.py`, `tools/v9/{verify_cache_alignment,eval_v9}.py`, `classifier/train_v8.py` — all read with an unguarded `read_text`/`open`, so every one fails loudly rather than going stale. None is exercised by a test.]`

**The reclaim is working-tree only, and that is what was taken.** These were `filter=lfs` in
`.gitattributes` and re-included by exact-path `.gitignore` negations; `git rm` freed the
139.6 MiB working copy and `.git/lfs` reclaimed **nothing**, because that needs a prune — and
`git lfs prune --verify-remote` is not usable here (it reports "missing on remote" when it
merely could not authenticate, the one condition under which you must not prune). It was
**not run**. Any actual `.git/lfs` reclaim goes through the batch-API + sha256 procedure in
[`artifacts_resolver.md`](artifacts_resolver.md) §5.

### The v9 + v10 aug caches, and v9's `plan` + `cache_manifest` — TAKEN 2026-08-08

**What went.** `data/v{9,10}/aug_cache` under `ARTIFACTS_ROOT` — **201,216 tiles /
15,353,500,833 B (14.299 GiB)**, measured immediately before the delete so that the trees
removed were provably the trees classified (170,808 / 12.090 GiB and 30,408 / 2.209 GiB).
No backup copies. And `data/v9/{plan,cache_manifest}.jsonl` — **146.0 MB**, de-tracked and
deleted. `[measured 2026-08-08, scratch/storage_cleanup/aug_cache_deletion.json]`

**Why the KEEP verdict of 2026-08-02 does not survive.** It rested on "deleting v9's tree
blocks any v10 retrain". Under ACTIVE + PREVIOUS (the section above) **v10 will never
retrain**: a rollback to v10 uses its *weights*, and v11 is a fresh crop-batch build whose
manifest/plan/cache are all `bulk()` with no chain to either tree. A cache kept to enable a
retrain nobody will run is the "keep the machinery" failure this doc opens with.

**Liveness was checked two ways before anything was removed, because one way misses the
readers that build their paths from data.**

  * *Path-literal grep over source.* Every non-doc hit on `data/v{9,10}/aug_cache` is v9- or
    v10-scoped machinery, a synthetic resolver test (`tmp_path`, no disk dependency), or
    `tools/v11/eval_v11.py::tile_path_diagnostic` — the one cross-version reader, and it is
    **absence-tolerant**: it returns `{"error": "...v10 cache tiles are not on disk"}` rather
    than raising. Its verdict is already banked in `data/v11/eval_results_v11.json`, so what
    the deletion costs is the ability to *re-run* v11's instrument check, not the check.
  * *Which committed RECORDS dereference into the trees.* Grep over all 1,108 tracked
    `data/**`+`labels/**` JSON/JSONL: exactly six files, all of them v9's and v10's own build
    artifacts (`v{9,10}/plan.jsonl`, `v{9,10}/cache_manifest.jsonl`, `v10/aug_roster.json`,
    `v10/build_metadata.json`). No label batch, no discovery ledger, no classifier config
    names a tile. This is the method that catches `classifier/data_v4.py`, which opens
    `row["path"]` and contains no `aug_cache` literal at all.

**The pair's rebuild is byte-identical, and it was measured with the caches already
deleted.** `uv run python tools/v9/build_plan.py` regenerates `plan.jsonl`
(56,424,452 B) and `cache_manifest.jsonl` (96,018,396 B) sha256-equal to the committed
originals. Two things the run also established:

  * **It is a TWO-STEP rebuild, and the first step is v8's.** v9's parity gate compares
    against `data/v8/{plan,cache_manifest}.jsonl`, themselves deleted 2026-08-03, so
    `tools/v8/build_plan.py` (~15 s) runs first or `_require_v8` raises. v8's three tracked
    outputs came back byte-identical in the same run, which re-proves the 2026-08-03 record.
  * **The other three v9 outputs are NOT byte-reproducible, and that is why they stay
    tracked.** A plain re-run rewrites `aug_roster.json` and `build_metadata.json` with a
    different `reuse_audit`: `v8_tree_retained_as_rollback` flips `true` → `false` and
    `prior_cache_trees_on_disk` empties, because that block is a **live disk probe frozen
    into a record** and the disk moved under it. Same shape as `marginal_cost`, which
    `carry_marginal` already protects; `reuse_audit` has no such guard, so a rebuild must
    restore these two from git (`git checkout -- data/v9/`) or it falsifies what was true on
    2026-07-31. `[measured 2026-08-08, scratch/storage_cleanup/v9_rebuild_parity.json]`

**The gate is retired, not weakened.** GATE A in
`tools/v10/build_plan.py::assert_recipe_parity` proved every prefix plan row byte-identical
to its v9 row — the claim that made reusing 170,760 v9 tiles legitimate. With both trees
gone it can only compare a plan against a plan, and arming it was the whole reason the v9
pair was tracked. GATE B (colormap library identity, two committed files) is untouched.
Retired alongside, and **deleted rather than left to skip**: the `slow`
`test_v10_build.py::test_prefix_plan_rows_are_byte_identical_to_v9s` and the `slow`
`test_frozen_record_writes.py::test_prereg_build_still_reproduces_the_committed_record`
(it re-ran `prereg.build()`, which reads v9's plan). A guard that goes green because its
input vanished is the failure both files exist to catch.

**The machinery went with the data.** Seven modules deleted, each one either a driver of a
render into a deleted tree or a measurement of one: `tools/v{9,10}/render_cache.py`,
`tools/v{9,10}/verify_cache_alignment.py`, `tools/v9/{estimate_cap_cost,measure_cap_effect}.py`,
`tools/v10/estimate_extend_cost.py`. `tools/v9/build_plan.py` **stays** — it is what makes
the pair `bulk()` rather than lost.

**Deliberately NOT done.** `data/v10/{plan,cache_manifest}.jsonl` (179 MB, LFS) stay
tracked: the same argument applies to them, but their producer's inputs are v10's own
manifest and the demotion was not in scope. `data/v9/aug_cache` keeps its
`RELOCATED_PREFIXES` literal even though `render_cache.py` is gone — `_is_aug_cache` matches
it as a *class* regardless, so dropping the literal changes no behaviour and removing it
would only make an accidental in-tree rebuild depend on one predicate instead of two.
`git lfs prune` was **not** run, for the reason in the v8 section above.
`[measured: 298,820,096 B across the four files, 2026-07-31, `ls -l data/v8 data/v9`;
146,377,248 B of that removed 2026-08-03]`

## The 20 MB gate is a sanity cutoff; the test is future usefulness — SET 2026-08-08

**The principle, stated once:**

> **The 20 MB commit gate is a sanity cutoff, not the test. The test is future usefulness.
> Labels, and the generative provenance of what was rendered, are the critical record;
> most else is optional scaffolding toward an eventual compact repo.**

It is the operational form of "the first question is usefulness, not cost" at the top of
this doc, and it settles the case that section leaves open: what to do when something is
*small enough to keep* and *has no named future use*. Under a size test it stays, because
nothing complains. Under this one it goes, and its machinery — index, manifest, plan,
registry line — goes with it.

Two corollaries, both exercised on 2026-08-08:

- **A record can outlive the thing it records and still be worth keeping**, when it is the
  provenance of a render someone may want to explain: `data/v9/aug_roster.json` stays after
  its 170,808 tiles are deleted, because it is the recipe those tiles were drawn under.
- **The scaffolding around a record does not inherit that.** `data/v9/{plan,cache_manifest}.jsonl`
  are 146 MB of per-tile rows that the roster plus `tools/v9/build_plan.py` regenerate
  byte-identically, so they are `bulk()` — the minimum needed to regenerate is what gets kept,
  and that minimum is the recipe, not the expansion of it.

`[verdict: Matt, 2026-08-08]`

## Weights retention: ACTIVE + PREVIOUS per model family — SET 2026-08-08

**The policy, stated once:**

> **Tracked weights = the ACTIVE head plus the PREVIOUS one, per model family. Everything
> older, and everything REJECTED, de-tracks at the next flip. An emergency copy may sit
> unreferenced in the artifacts store.**

A model family is a pin, not a directory: `data/classifier/` (the location head,
`production_pins.ACTIVE_CKPT`), `data/wallpaper_head/` (`wallpaper_pins.HEAD_CKPT`),
`data/render_mode_head/`, `data/queries/scorer/`. Each keeps two.

**Why two and not one.** One rung is not a rollback — it is a hope. Two is the smallest
number that lets a bad flip be undone by an edit rather than by a retrain, and the retrain is
the thing that cannot be done: none of these weights is GPU-reproducible, because in every
case the corpus or the split that produced it has moved on.

**Why two and not five.** The five-rung ladder v10 carried was a *fiction that read as a
plan*. Rolling back to v7 means reverting `t_good`, the keeper cut and τ_h to v7's `p_good`
scale — and v7's own eval slice was gitignored and is gone, so its table cannot be re-derived
and copying it forward reinstates a cut chosen against a predicate that is no longer served
(`production_pins`, the v8 t_good record). A rung you could not correctly take is worse than
an absent one: it costs 34 MB, and it invites a rollback that silently serves numbers about
nothing.

**A REJECTED candidate is not a rung at any age.** `render_mode_head/v2` lost its winner rule
on 2026-08-06 and de-tracked the same day; `classifier/v9` was built, evaluated, staged and
never deployed. Neither was ever a rollback target, because a rollback to a version that was
never served restores a gate that never ran. Their **run record and report stay tracked** —
those are small, and they are the reason the rejection does not get re-litigated.

**The emergency copy is deliberately unreferenced.** De-tracked weights are copied to
`C:\Code\fractal-maker-artifacts\retired_weights\<name>\` and verified by SHA-256 against the
LFS pointer's own `oid` — the strongest available check, since it compares against what git
had rather than against what happens to be on the disk that did the de-tracking. Nothing in
the repo resolves, registers or documents that path *as a path*: no resolver entry, no size
guard registry line, no pin. That is the point. A referenced backup is a rung with extra
steps, and the policy above is only meaningful if the tree cannot quietly grow a sixth one.

**What de-tracking a weight touches**, all five, or the guards go red — and they are the
checklist because each was found the hard way: `.gitattributes` (the LFS rule),
`tests/test_large_tracked_blobs.py` (`ALLOWLIST` + `BINARY_ALLOWLIST`; a dead entry fails
just as loudly as an undeclared blob), `tools/audit/size_guard.py` (the working-tree
registry), `tests/test_tracked_artifacts.py` (the canaries), and the exact-path `.gitignore`
negations. `git rm --cached` leaves the file in the working tree, which is why every
rung-existence assertion in this repo now reads the **index**, not `Path.exists()`: on the
machine that did the de-tracking those are the same answer, and on a fresh clone they are not.

`[code: tools/scoring/production_pins.V10_CKPT_ROLLBACK; test_production_pins.LADDER;
data/v11/adoption_record.json:rollback_ladder]` `[verdict: Matt, 2026-08-08]`

### The force-add class is empty — CLOSED 2026-08-08

Every model-family artifact in the tree is now declared by an **exact-path `.gitignore`
negation**. The last four were `data/wallpaper_head/v3/model_best.pt`,
`data/wallpaper_head/v4/{config,metrics}.json` and
`data/queries/scorer/v3_gvo/model_best.pt`, closed the same way `data/classifier/`
(2026-08-08) and `data/render_mode_head/v1` (2026-08-06) were. `durability_map` MISMATCHes
went **16 → 14 of 50**.

**A force-add WORKS, which is the problem.** Nothing is red and the file is in the index;
all three ways it bites are silent. `paths.durable()` refuses the path *and every new
sibling*, so the contract cannot extend a directory the live head lives in. A fresh clone's
`git add -A` would not re-add it, and nothing would notice. And a bare `git check-ignore`
reports a force-added path as not-ignored regardless of the rules — how a real false accept
got through on the v7 checkpoint. A negation is a **rule**; a force-add is an **event** that
leaves no rule behind.

**Never a directory negation.** `!/data/queries/scorer/v3_gvo/` would sweep in the untracked
training state beside the weight; `!/data/wallpaper_head/v4/` would re-include the 34 MB
weight the retention policy just de-tracked, plus five per-seed dirs and an eval montage.
Walk the chain down and negate each file.

**Why these two families outlived the others**, and the reusable part:
`test_every_tracked_build_artifact_is_covered_by_a_negation` asserts the same invariant from
git's side — but only over `BUILD_PREFIX_RE = ^data/(?:classifier/)?v\d+/`. The stage-2 heads
are not versioned that way (`v3_gvo` is not `v<N>`), so they sat outside every coverage
assertion. The fix is a second assertion over the four **model-family roots** of the
retention policy rather than a wider regex, since a family is a pin, not a path shape.
`[code: tests/test_tracked_artifacts.py::{MODEL_FAMILY_ROOTS,
test_no_model_family_artifact_survives_by_force_add} — proved red by removing one negation
while its file stayed tracked]`

**Two corrections found while doing it.** `.gitattributes` claimed
`data/wallpaper_head/v4/metrics.json` was untracked "as it always was"; it is tracked, and
so is `config.json` — they are the run record the policy keeps on purpose. And the
`durability_map` row for the query ranker said it was regenerable by "a retrain from the
committed query labels": it is not. The labels are tracked but key their tiers by candidate
id alone, and the `records/` that join an id to its (location, palette, coloring) are gone —
the same loss the PREF-HEAD JOIN row states from the other side. Re-classed unregenerable
and population-defining.

## `outcome_feats.npz` is the ledger's SIDECAR, not a second ledger — TAKEN 2026-08-08

The two files sat side by side in every run dir, in the same `.gitignore` re-include, under
the same `disk_audit` `NEVER` neighbourhood, and that adjacency was doing the classifying.
They are not the same class. `outcome_ledger.jsonl` records a population that cannot be
re-walked. `outcome_feats.npz` is that population plus a forward pass — one 1280-D
penultimate vector per admitted q3 — and `production_seeder.py`'s own header says the
feature is *"logged, never gates"*. It is `bulk()` as of 2026-08-08, matched as a class by
`artifacts._is_discovery_feats`, and the 28 existing files (10.88 MB) were **moved**
out-of-tree, not deleted.

**The committed byte split, per banked run** (tree bytes, `git ls-files data/discovery` +
`stat`, 2026-08-08; full table in `scratch/storage_cleanup/feats_split.json`):

| | bytes | share |
|---|---:|---:|
| all tracked `data/discovery/` (215 run dirs, 586 files) | 485,835,461 | |
| — `outcome_ledger*` | 15,721,887 | 3.2% |
| — `outcome_feats*` | 10,878,152 | **2.2%** |
| — everything else (the four LFS `.jsonl.gz` streams, pools, walks, summaries) | 459,235,422 | 94.5% |

The 2.2% is the misleading number and is why the per-run figure is the one to quote: the
485 MB total is dominated by a handful of pre-segmentation campaign runs. On a **modern**
run the sidecar is the largest single file — `steady_state_v2_20260807` (363.63 active
minutes, clean close) is 10,765,771 tree bytes, of which **3,234,169 is the npz (30.0%)**
against 2,725,616 for the ledger.

**An 8 h run's committed tree bytes, projected linearly off that run** (×1.32):

| | bytes | vs the 20 MB gate |
|---|---:|---:|
| as it stood | 14,211,066 | 71% |
| with the npz out | 9,941,889 | 50% |
| the ledger alone | 3,597,876 | 18% |

**So the ledger alone does not approach 20 MB, and segmentation stays a separate
decision.** The demotion buys back 30% of the per-run footprint and moves an 8 h run from
71% of the gate to 50%. `[measured 2026-08-08, scratch/storage_cleanup/feats_split.py;
consistent with `discovery_pipeline.md`'s independently-projected 17.0 MB worst case]`

**The recompute is named, and it is not a byte-restore.**
`tools/atlas/recompute_outcome_feats.py` rebuilds a run's store from its own ledger,
reaching `prescreen._render` / `prescreen.embed_paths` by import so there is no second copy
of the recipe, over exactly the `distinct: true` rows (derived, and verified against two
banked stores: 2,244 == 2,244 and 311 == 311). But each banked vector was pulled through the
head that was ACTIVE when its run walked — every ledger row records that in
`scorer_version` — and those weights are de-tracked under ACTIVE+PREVIOUS. A rebuild embeds
through today's head and stamps which one into the npz's `__meta__` key. **That is why the
existing files were moved rather than deleted**, and it is the "producer reads live
constants" hazard in its exact form: the producer survives, and re-running it measures
something else.

**The reader that would have gone quiet.** `redecode_grid.py` subset the store under an
`if feats_src.exists():` and, absent it, wrote an empty subset with `n_feats: 0` into its
readout — which reads as *"this intake unit has no surviving features"*, not as *"the input
was missing"*. It now routes through `discovery_sinks._require_feats`, which raises naming
the rebuild. `[code: tools/atlas/discovery_sinks.py::{feats_path,_require_feats}]`

**One relocated family is a FILE.** Every other class names a directory, so the reappearance
tripwire scanned `dirs` and was complete. This one sits beside the ledger it derives from, so
the tripwire's discovery walk grew a `files` branch — otherwise the class would relocate
correctly and never be checked, which is the gap `artifacts_resolver.md` §3 warns about.

## Git history is a durability tier too, and this repo's floor is 2026-07-24

`fractal-maker` begins at a **single squashed import commit** (`ff88da4`, 1247 tracked files,
byte-identical to the old HEAD by blob-OID diff). Nothing before 2026-07-24 is reachable with
`git log` here, so "recoverable from git history" — a phrase this contract and
`docs/design/README.md` both lean on — means *since the import*, and a `git show <old>^:<file>`
against an earlier date silently finds nothing rather than failing.

The pre-migration record is not lost, it is **elsewhere**: the old GitHub repo (renamed
`fractal-generator-deprecated`) holds the full 609-commit history with all branches confirmed
merged, and a local mirror sits at `C:\code\fractal-generator-prewrite-backup-20260724`
(`repo-mirror.git` = all refs, plus raw weight copies with verified sha256 and a HEAD
manifest). Check there before concluding something never existed.

Why the squash: the old `.git` had reached 365 MB, dominated by raw-committed `v6`/`v7`
weights (34 MB each) and dead `to_delete/**` + superseded `pool_colormaps.json` blobs. A
premise correction worth keeping, because it is the kind that gets re-guessed: there was
**never ~300 MB of dead weight** — committed weight was 68 MB single-copy, and a
"single-digit MB `.git`" was unreachable while preserving the live tracked palettes and corpus.
A fresh repo was chosen over a history rewrite for that reason. All classifier weights are LFS
from the import forward. `[measured: 2026-07-24; verified by blob-OID diff of the 1247-file
tracked set]`

## Derive in code, freeze in records

**A generator must read the state it reports from the state itself.** A number restated by
hand in a generator drifts silently the day the thing it describes moves.
`stamp_cap_policy.py` derives the legacy policy from `location.LEGACY_MAXITER_POLICY`
rather than restating its four constants, so it cannot drift from the token;
`verify_v6_gate.py` reads mean/std from the checkpoint rather than a constant;
`deploy_tail.py` derives its candidate roster from the mode registry rather than a list.
`[code: tools/orbital/stamp_cap_policy.py; tools/atlas/verify_v6_gate.py;
tools/mining/deploy_tail.py]`

**A committed record may keep what was true when it was written**, and rewriting it would
falsify history. `data/v9/build_metadata.json` records the sha256 of v8's `manifest.jsonl`
so "same corpus" is a checked claim rather than a copy that can drift; the convergence
ladder records the policy it was measured under and asserts it is not reproducible today.
Those are frozen on purpose. `[code: .gitignore's v9 stanza; data/v9/build_metadata.json;
data/orbital/maxiter_convergence_ladder.json]`

**Corollary: a hardcoded fact is how a metadata file outlives the thing it records.** The
symptom is always the same — the fact is still there, still confidently phrased, and no
longer true. `redecode_grid.py` documents a grid that ran at "a hardcoded provisional
`t_good=0.18` that nobody ordered (a stale copy…)"; the size guard's own threshold
rationale carried "the whole non-excluded working tree is ~5.2k files" long after the tree
had fallen to 1,638. The rule that follows: if a generator can compute it, compute it; if
it is a record, date it. `[code: tools/phoenix/redecode_grid.py;
tools/audit/size_guard.py]` `[measured: 1,638 non-excluded working-tree files, 2026-07-31]`

## Known exceptions the contract does not cover

The four classes are exhaustive over what `tools/paths.py` writes. They are **not**
exhaustive over the tree. Recording the gaps here so the contract stops silently
disagreeing with what is on disk — an unstated exception reads as an oversight, and the
next pass either "fixes" it or trips over it.

- **`labels/` — durable corpus metadata living outside the durable store.** 54 flat
  `.json` files, 505 KB (2026-08-06: `git ls-files labels/`), git-tracked and **not**
  gitignored, under three competing naming schemes: 37 bare (`location_labels.json`), 16
  `amend_<date>_…`, 1 `<date>_…`. Substantively these are `durable` — hand labels
  recording a human judgement that cannot be re-observed — but they are reached through
  `label_store.LABELS_DIR` (`os.path.join(ROOT, "labels")`), not `paths.durable()`, so
  **no `durable()` write-time gitignore assertion ever runs over them** and they are
  invisible to the class the tree would otherwise put them in.

  *Why they are still there, and why that is defensible for now:* `label_store.py` already
  owns the harder half of the problem — `SIDECAR_LABELS` (batch_id → sidecar filename),
  `FOREIGN_LABEL_FILES` (files sitting here that belong to a *different* corpus and must
  not be read), and `SIDECAR_OWNER` (the join rule that stops cross-contamination when one
  sidecar is shared by two batches). 30 modules reference the directory. `disk_audit.py`
  additionally protects it with `Rule(r"^labels/", NEVER, …)` — deliberately **by path
  prefix and registry-independent**, its comment noting that keying off `SIDECAR_LABELS`
  would make an unregistered sidecar (which has already happened twice) look deletable. So
  relocating the tree silently drops that protection unless the pattern moves with it. The
  exception is a *naming and reach* problem, not a durability risk today.

  *What would close it:* route the sidecars through `paths.durable()` and fold the
  directory under `data/labels/`, moving the `disk_audit` prefix rule and the `label_store`
  registries in the same commit. Not attempted in the 2026-07-31 hygiene pass — 46 files
  with 30 referencing modules and a location-keyed protection rule is a seam, not a
  cleanup. `[code: tools/corpus/label_store.py::{LABELS_DIR,SIDECAR_LABELS,SIDECAR_OWNER,
  FOREIGN_LABEL_FILES}; tools/audit/disk_audit.py `Rule(r"^labels/", NEVER)`]`
  `[measured: 46 tracked files / 448 KB, 2026-07-31, `git ls-files labels/` + `du -sh labels/`]`

- **`data/root_field/*.f32` — bulk regenerable inside the durable store.** 8 files, 1.1 GB,
  untracked; an expensive-but-deterministic cache, i.e. textbook `bulk()`. It sits in-tree
  because the resolver is Python-only and this path is a **Rust** constant
  (`src/root_field.rs::CACHE_DIR = "data/root_field"`); relocating it needs a Rust-side
  `ARTIFACTS_ROOT` twin. Deliberately out of scope of the 2026-07-31 pass, which named it a
  seam rather than a cleanup. `[code: src/root_field.rs::CACHE_DIR]`
  `[measured: 8 files / 1.1 GB, 2026-07-31, `du -sh data/root_field`]`

- **A view is not durable content, and a view whose images are gone is not content at all.**
  *Closed 2026-07-31 — the eight `pool_sheet.html` files under `data/atlas/round{1,2}/*/`
  (2.6 MB) were deleted.* Each was an `<img src="tiles/tile_NNNN.png">` index over a `tiles/`
  directory that no longer existed anywhere: not beside the sheet, not out-of-tree
  (`data/atlas/` is not in `artifacts.RELOCATED_PREFIXES`, so `artifacts.resolve` returns the
  in-tree path unchanged), and nowhere else in the working tree. Eight pages of broken images
  sitting in the class reserved for records that cannot be rebuilt.

  The durable content of those rounds — `pool.jsonl`, `walks.jsonl`, `REPORT.md` — stays and
  is unaffected. The sheet is a **view over** `pool.jsonl`, which is the point: the row
  survives, the rendering of it does not have to. `[measured: 8 files / 2.6 MB, 0 tiles
  present, 2026-07-31, `find . -type d -name tiles`]`

  *The generator is alive, and that does not save the sheets.* An earlier draft of this entry
  said "the builder went with the atlas round-1/round-2 cluster when it was deleted" — that
  was wrong. `src/guided_descend.rs` still writes `pool_sheet.html` and `tiles/tile_%04d.png`
  in the same pass, and those directories are its output set. But a *sheet* rebuild would mean
  re-running the seeded descent walk end to end, and nothing re-renders tiles from an existing
  `pool.jsonl` — so the sheets were regenerable only in the sense that the whole run was, which
  is not a reason to keep 2.6 MB of broken HTML in `data/`.
  `[code: src/guided_descend.rs — `tiles_dir.join("tile_{:04}.png")` and
  `out_dir.join("pool_sheet.html")` in one function]`

## Where this is enforced

The mechanism is [`artifacts_resolver.md`](artifacts_resolver.md); this is the index into
it.

- `tools/paths.py` — the four class functions; `durable()` carries the write-time
  gitignore assertion. **This is the primary mechanism**: a family that declares
  `bulk()` never lands in-tree in the first place. → resolver doc §1–2
- `tools/audit/size_guard.py` — the **backstop**: three independent scan rules against one
  `REGISTRY` allowlist (per-file ≥ 1 MiB; per-dir small-file aggregate ≥ 100 MB; per-dir
  bulk ≥ 2,000 files or ≥ 500 MB). The bulk rule is the one that matches rule 5's harm.
  → resolver doc §4
- `tools/audit/test_relocated_artifacts.py` — reappearance tripwire: a relocated family
  that re-materializes under its old in-tree path goes red and names the offender.
  → resolver doc §3
- `tests/test_repo_size_guard.py` — hard-fails on any flagged violator with no registry
  entry, and on any entry that covers nothing and is not marked a live forward
  declaration. → resolver doc §4
- `tests/test_storage_classes.py` — guards the `durable()` assertion (a durable path git
  would discard is rejected). → resolver doc §2
- `tests/test_tracked_artifacts.py` — unregenerable artifacts stay tracked; the versioned
  build trees are guarded relationally off the `.gitignore` negations, with a non-vacuity
  pairing. → resolver doc §6
- `tests/test_no_out_dir.py` — tripwire: the retired `out/` directory must not exist
  or be recreated.
- `tests/test_docs_tree.py` — `docs/` is source only; `docs/findings/` stays retired.
  → resolver doc §7
