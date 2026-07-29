# Storage classes — the durability contract

Every file this project writes belongs to exactly one durability class. The class is
declared **at the write site** through `tools/paths.py`, and the choice is binding:
it decides whether the file is expected to survive, and therefore whether its absence
is ever a problem. This note is the contract those functions enforce. It is rules, not
commentary.

## The first question is usefulness, not cost

The contract's question — *what would it cost to get the identical content back?* — is
only ever the **second** question. The **first** is whether we would ever want it back
at all. As written, the rules below weigh preservation by rebuild cost, which quietly
implies that anything expensive to reproduce deserves to be kept. It does not. A recipe
for regenerating something no one will ever ask for is itself clutter — and keeping the
apparatus (index, manifest, plan, registry line) for such a thing is the specific
failure this note exists to stop.

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

## The classes

| class | writer | lives in | survives `rm -r scratch/*`? |
|---|---|---|---|
| **scratch** | `scratch(...)` | `scratch/` (gitignored) | no — deletion is the design |
| **bulk** | `bulk(rel)` | out-of-tree via `ARTIFACTS_ROOT` | rebuilt on demand |
| **durable** | `durable(rel)` | `data/` (git-tracked) | yes — git keeps it |
| **vendored** | committed by hand | config/code, with provenance | yes — it's source |

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

   The corollary for a *new* bulk family is that it must be born out-of-tree — declare
   it `bulk()` at the write site and register its prefix in
   `artifacts.RELOCATED_PREFIXES` **before** the first run, not after it has already
   materialized 170k files in the tree.

## Where this is enforced

- `tools/paths.py` — the four class functions; `durable()` carries the write-time
  gitignore assertion. **This is the primary mechanism**: a family that declares
  `bulk()` never lands in-tree in the first place.
- `tools/audit/size_guard.py` — the **backstop**. Three independent scan rules, each
  checked against one shared `REGISTRY` allowlist:
    - per-**file** ≥ `FILE_THRESHOLD` (1 MiB),
    - per-**directory** small-file aggregate ≥ `DIR_THRESHOLD` (100 MB),
    - per-**directory** bulk: ≥ `DIR_FILE_COUNT_THRESHOLD` (2,000 files) **or**
      ≥ `DIR_BYTES_THRESHOLD` (500 MB) in its subtree, at minimal granularity.
  The bulk rule is the one that matches rule 5's harm: the older per-file rule passes a
  cache of a million small crops cleanly, because no single file is large.
- `tools/audit/test_relocated_artifacts.py` — reappearance tripwire: a relocated family
  that re-materializes under its old in-tree path goes red and names the offender.
- `tests/test_repo_size_guard.py` — hard-fails on any flagged violator with no registry
  entry; warns on registry entries that no longer cover anything.
- `tests/test_storage_classes.py` — guards the `durable()` assertion (a durable path git
  would discard is rejected).
- `tests/test_no_out_dir.py` — tripwire: the retired `out/` directory must not exist
  or be recreated.
