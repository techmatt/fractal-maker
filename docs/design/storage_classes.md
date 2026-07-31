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
seconds: the cost of the loss is not the render, it is the approval.
`[verdict: Matt]` `[unverified — no committed code path does this staging today; the
nearest live instance is a *proposal* left in scratch, below]`

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

So the KEEP question is not *does something still produce this*, it is *would running it
again produce this*. `[verdict: Matt]`

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
