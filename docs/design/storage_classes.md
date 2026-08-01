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

- **`labels/` — durable corpus metadata living outside the durable store.** 45 flat
  `.json` files (+1 stray `.md`, `cc_julia_dup_audit.md`), 448 KB, git-tracked and **not**
  gitignored, under three competing naming schemes: 30 bare (`location_labels.json`), 15
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
