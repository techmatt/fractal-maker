# Repo analysis — file counts, byte weight, and structural notes

Measured **2026-07-31** on `C:\Code\fractal-maker` @ `73afd8b`. **`scratch/` is excluded
from every count below** (803 files / 1.1 GB) at the owner's request — it's being
emptied. Successor to `repo_structure_audit.md` (file-count bomb) and
`repo_size_audit.md` (byte bomb); this one re-measures both and audits what's left. Both
predecessors were **deleted** in `27b68a4` ("deleting old files"), so there is nothing left
at the root to mistake for current — see "Predecessors" at the foot of this file.

> **Every measurement below carries the command that produced it.** A measurement in prose
> with no date and no command is what produced three false premises this month, so the
> `[cmd: …]` stamps are part of the content, not decoration. Where a figure has **moved
> since 2026-07-31**, the change is stated inline rather than silently overwritten — a
> reader must be able to tell a re-measurement from an edit.
>
> **Moved since first measurement:** the LFS-cache figures in "Git object storage is fine"
> (`git lfs prune` ran in the 2026-07-31 hygiene pass and reclaimed 268 MiB), and the
> root-clutter and empty-directory items in "Structural observations" §5–§7 (acted on in
> the same pass). Each is annotated in place.

## Headline: the grep problem is solved

The audits that triggered this line of work measured **~640,000 files / 85 GB** in
tree and a `grep -rl` that took **>120 s**. Today:

| metric | then | now |
|---|---:|---:|
| files in tree (excl. `scratch/`) | ~620,000 | **30,236** |
| bytes in tree (excl. `scratch/`) | ~75 GB | **7.93 GB** |
| files a gitignore-*unaware* tool must walk | ~620,000 | **30,236** |
| full recursive walk of the whole tree | >120 s | **1.2 s** |
| `git ls-files` | — | 0.04 s |

`[cmd, 2026-07-31: files = find . -path ./scratch -prune -o -type f -print | wc -l;
bytes = find . -path ./scratch -prune -o -type f -printf '%s\n' | awk '{s+=$1} END{print s}';
walk = time find . -type f > /dev/null]`
`[re-measured 2026-07-31 after the hygiene pass: 30,284 files / 7.67 GB / 0.73 s walk —
the byte drop is the 268 MiB LFS prune, the file delta is churn under target*/]`

**Nothing needs doing for greppability.** Even the naive walk (no ignore rules,
including `.venv`, `.git`, `target*`) is ~1 s. The part a human or a tool actually
reads is far smaller:

| slice | files | size |
|---|---:|---:|
| whole tree, excl. `scratch/` | 30,236 | 7.93 GB |
| ├─ `.venv/` | 25,053 | 4.82 GB |
| ├─ `.git/` | 1,787 | 951 MB |
| ├─ `target/` + `target-test/` | 1,778 | 484 MB |
| └─ **everything else ("source-ish")** | **1,613** | 1.86 GB |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `__pycache__`/`.pyc` | 256 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;**real source + metadata** | **1,360** | — |

`[cmd, 2026-07-31: for d in .venv .git target target-test; do find $d -type f | wc -l;
du -sh $d; done — "everything else" is the whole-tree figure minus those four]`

`.venv` is 83% of the remaining inode count and is a non-problem (regenerable,
skipped by every ignore-aware tool, and cheap even when it isn't).

## Tracked content

**1,336 tracked files, 749 MB of working-copy bytes.** Composition:
`[cmd, 2026-07-31: git ls-files | wc -l; git ls-files | sed 's/.*\.//' | sort | uniq -c | sort -rn;
git ls-files | cut -d/ -f1 | sort | uniq -c | sort -rn]`
`[re-measured 2026-07-31 post-pass: 1,346 tracked — +11 from this pass's new docs/tests,
−1 from deleting data_large/README.md]`

| ext | count | | top-level dir | tracked files |
|---|---:|---|---|---:|
| `.py` | 434 | | `data/` | **706** |
| `.json` | 433 | | `tools/` | 401 |
| `.jsonl` | 301 | | `labels/` | 46 |
| `.md` | 50 | | `docs/` | 33 |
| `.html` | 35 | | `classifier/` | 27 |
| `.rs` | 32 | | `src/` | 24 |
| `.npz` | 23 | | `palette_extractor/` | 21 |
| `.pt` | 8 | | `dramatic_palettes/` | 20 |
| `.COMPLETE` | 6 | | `specs/` | 18 |

LOC: **Rust 18.1k** (24 files), **Python 120.3k** (434 files), Markdown 7.1k (50 files).

**53% of tracked files are data, not code** (706 in `data/` + 46 in `labels/`, almost
all `.json`/`.jsonl`/`.npz`/`.pt`). That is a deliberate choice — the durability
contract in `tools/paths.py` says `data/` records populations that cannot be rebuilt —
but it means the tracked tree is majority ledger. It's the reason `.gitignore` is what
it is (below) and the reason `git` needed LFS.

### Git object storage is fine; the LFS cache is not

| store | size |
|---|---:|
| pack + loose objects | **26 MB** (16.8 MiB pack, 1,472 objects) |
| `.git/lfs/` local cache | **925 MB** (33 files) |
| LFS working copies on disk | 656 MB (28 files) |

`[cmd, 2026-07-31: git count-objects -vH; du -sh .git/lfs/objects;
git lfs ls-files -s]`

> **SUPERSEDED 2026-07-31 (same day, later): `git lfs prune` ran.** The cache is now
> **658 MB / 31 objects**; 4 superseded objects totalling **281,512,322 B (268 MiB)** were
> deleted. The recommendation below is therefore **done**, not pending. Two things about
> how it was verified are worth carrying forward, because the guard nearly stopped it for
> the wrong reason: `git lfs prune --verify-remote` (git-lfs 2.11.0) reported all four
> objects *missing on remote* and aborted — a **false negative**, because it resolves the
> endpoint via SSH and SSH auth fails in this checkout. All four were then confirmed
> present by querying the LFS batch API over anonymous HTTPS and **downloading and
> sha256-verifying each one byte-for-byte** before pruning.
> `[cmd, 2026-07-31: git lfs prune --dry-run --verify-remote; curl -X POST
> .../info/lfs/objects/batch; sha256sum each object; git lfs prune]`

The `git filter-repo` recommendation from `repo_size_audit.md` (Phase 5, "~360 MB of
dead binary weight") is **moot** — the migration to `fractal-maker` on 2026-07-24 was a
clean re-import (111 commits, 8 days of history), so there is no legacy weight. The
real object DB is 26 MB.

What *is* live: **the LFS cache stores 925 MB to back 656 MB of checked-out content**,
i.e. it retains superseded versions (`model_best.pt` for v5…v9, the v8 *and* v9
`plan.jsonl`/`cache_manifest.jsonl` pairs at ~50–90 MB each), and every one of those
also exists as a full working copy. Net ~1.6 GB on disk for 656 MB of data.
`git lfs prune` reclaims the superseded part; that is the single cheapest byte win
available and is zero-risk (it only drops objects reachable from old commits that are
also on the remote — verify a remote exists first).

## Where the remaining bytes are

| item | size | class | note |
|---|---:|---|---|
| `.venv/` | 4.82 GB | regenerable | `uv sync`; leave it |
| `data/root_field/` (8 `.f32`) | **1.02 GB** | **misfiled** | see below |
| `.git/lfs/` | 925 MB | see above | `git lfs prune` |
| `target/` + `target-test/` | 484 MB | build | `target-test` is the CLAUDE.md exe-lock workaround — a permanent 235 MB duplicate |
| `data/classifier/` | 228 MB | durable (`.pt`) | keep — owner's stated exception |
| `data/discovery/` | 197 MB | mixed | 467 files; LFS `harvest_log`/`prio_terms` |
| `data/v9/` + `data/v8/` | 288 MB | durable | the two LFS plan/manifest pairs |

`[cmd, 2026-07-31: du -sh .venv .git/lfs target target-test data/root_field
data/classifier data/discovery data/v8 data/v9]`
`[superseded: .git/lfs is 658 MB after the prune, so the total is ~268 MiB lower than the
table implies. data/root_field re-measured at 1.1 GB / 8 files.]`

**`data/root_field/*.f32` (1.02 GB, untracked) is the one clear storage-class
violation left.** By the contract in `tools/paths.py`, `data/` is the *durable,
git-tracked* class; this is an untracked, expensive-but-deterministic cache, which is
exactly the `bulk()` class that belongs out-of-tree under `ARTIFACTS_ROOT`. It is still
there because the resolver is Python-only and the path is a **Rust** constant
(`src/root_field.rs` `CACHE_DIR`) — Phase 2 of `repo_size_audit.md`, never executed.
It is 55% of the non-`.venv`, non-build byte weight of the source-ish tree.

## Structural observations

Ordered by how much they'd bite someone navigating the repo.

**1. `tools/` is an append-only research ledger with no index.** 401 tracked files
across **36 subdirectories**, no `README`, no live/dead marker. **318 of 434 tracked
`.py` files are standalone entry-point scripts** (`__main__`), not importable modules —
so "what produces X?" has no entry point to start from. The versioned families are
copy-forward rather than parameterized: `build_plan.py` exists **6 times**
(`tools/v4…v9`), `build_manifest.py` **4 times**, `render_cache.py` and
`verify_cache_alignment.py` twice each. This was recommendation #5 in
`repo_structure_audit.md` and is untouched.

Note for anyone trying to date this: **`git log` cannot tell you what's live here.**
Every path's first commit is 2026-07-24 (the migration import), so last-touch dates
cluster in an 8-day window and rank retired scaffolding alongside current drivers.
Liveness has to be established by import graph or by hand, which is the cost of having
no boundary.

**2. Python has no package root, and 300 of 434 files hand-patch `sys.path`.**
`pyproject.toml` declares dependencies only — no build backend, no packages, no `src`
layout. Imports work by CWD plus `sys.path.insert`. `tools/_bootstrap.py` centralizes
this **for 4 of the 36 subdirs** and is imported by 21 files; its own docstring says
the rest "still carry their own inserts — migrating them is a follow-up."
**300/434 files (69%) mutate `sys.path` directly.** The reasoning in that docstring is
sound (real packages would break `python tools/x/y.py` invocation), but the current
state is the worst of both: a centralizing module that covers 12% of the tree, and a
convention that every new file must re-derive.

**3. Tests live in three places, and `pytest` has no `testpaths`.** `tests/` holds
8 Rust integration tests **and** 6 Python guard tests; ~60 more Python tests live
beside their code (`tools/*/test_*.py` ×52, `classifier/` ×2, `palette_lib/`,
`palette_extractor/`). Co-location is defensible; mixing Rust `tests/*.rs` (which cargo
*requires* there) with Python guards in the same dir is not, and with no `testpaths`
in `pyproject.toml` a bare `pytest` collects from CWD downward — including `.venv`'s
25k files unless pytest's default `norecursedirs` happens to save it. One
`testpaths = ["tests", "tools", "classifier", "palette_lib", "palette_extractor"]`
line makes the suite's extent explicit instead of emergent.

**4. Eight top-level dirs are really subsystems.** `classifier/`, `palette_lib/`,
`palette_extractor/`, `dramatic_palettes/`, `labels/`, `specs/`, `prompts/`,
`data_large/` all sit as peers of `src/` and `data/`. Two are near-empty
(`prompts/` = 1 file; `data_large/` = **1 file, a README** — a tombstone for a
relocated tree, still occupying the top level and still colliding by name with
`data/`, which `repo_structure_audit.md` flagged and nobody removed). Two are
data stores parallel to `data/`: **`labels/`** (46 flat JSON files under three
competing naming schemes — bare, `amend_<date>_…`, and `<date>_…`) is durable corpus
metadata living *outside* the durable store, so it is invisible to the four-class
contract that `tools/paths.py` enforces.
`[cmd, 2026-07-31: git ls-files labels/ | wc -l; naming split by regex on the basenames]`
`[RESOLVED 2026-07-31 (partly): data_large/ deleted (it held one README and no data).
labels/ is NOT moved — it is now recorded as a named exception with its reason in
storage_classes.md §"Known exceptions the contract does not cover", so the contract
stops silently disagreeing with the tree. Re-measured: 45 .json + 1 stray .md, 448 KB,
30 bare / 15 amend_<date>_ / 1 <date>_.]`

**5. Two empty directories persist:** `scratchpad/` (0 files — governed by a
~15-line CLAUDE.md rule that exists because something load-bearing once vanished from
it) and `data/v7/` (0 files, a leftover of the v7 → v8 artifact rebuild).
`[RESOLVED 2026-07-31: data/v7/ removed (confirmed empty first — `find data/v7 -mindepth 1`
returned nothing). `scratchpad/` deliberately KEPT: it is empty by design and is a live
forward declaration in the size-guard registry.]`

**6. Root-level clutter contradicts its own rule.** CLAUDE.md: *"The root holds only
source, config, docs, and committed `assets/`"*, and *"Findings/analysis text goes to
`docs/design/`, committed"* — enforced by `tests/test_docs_tree.py`. Yet the root
carries **three loose analysis documents** (`repo_size_audit.md`,
`repo_structure_audit.md`, `uf_coloring_algorithms.md`) that are exactly the kind of
prose `docs/design/` owns. This file makes four. They should all move.
`[RESOLVED 2026-07-31: the two audit docs were already gone — deleted in `27b68a4`, after
this analysis was written and before it was acted on. `uf_coloring_algorithms.md` and this
file moved to `docs/design/` and are indexed in its README. The root now holds only source,
config, docs and LICENSE.]`

**7. The rename to `fractal-maker` is half-applied.** `README.md` (2 lines, its entire
content) says *"# fractal-generator"*; `pyproject.toml` declares
`name = "fractal-generator-ml"`; both root audit docs measure and cite
`C:\Code\fractal-generator` paths. Those audits are also **numerically stale by ~20×**
(they report 640k files / 21 GB against today's 30k / 7.9 GB) while sitting at the root
where they read as current. Stale measurements that look authoritative are worse than
no measurements — either restamp them with a "superseded" header pointing here, or
retire them into `docs/design/`.
`[RESOLVED 2026-07-31: README.md → `# fractal-maker`; pyproject `name = "fractal-maker-ml"`
and its description likewise. The stale audits needed neither treatment — `27b68a4` had
already deleted them, which is the stronger fix. **Still half-applied on purpose:** the
Rust crate is `fractal-generator` and the binary `fractal-generator.exe`, named in
CLAUDE.md, in ~30 tools and in every `tools/*/render_*` argv. Renaming that is a
mechanical sweep with a real breakage surface, not a hygiene item.]`

**8. `.gitignore` is still the negation machine.** 364 lines: 216 comments, 127 rules,
**64 negations (`!`)**. That's the structural tax of interleaving tracked metadata with
ignored bulk in the *same directory* (`data/label_corpus/batches/<id>/` holds both
committed `images.jsonl` and ignored `crops/`). The comments are genuinely good — they
document their own traps — but 64 negations means every new experiment must
hand-author a correct stanza or silently commit scratch / lose metadata. The
`repo_structure_audit.md` fix (all disposable under one ignored root, all committed
metadata under one tracked root, never interleaved) is still the right one and is
still unexecuted.

**9. Minor:** 8 `.html` viz sheets are committed *into* `data/atlas/` (regenerable
views in the durable store — the convention says those belong in `scratch/`); 6
`.COMPLETE` zero-byte sentinels are tracked as ledger state; `labels/` contains a
stray `.md` (`cc_julia_dup_audit.md`) among 46 JSON files.
`[CORRECTED 2026-07-31: the 8 sheets are not "regenerable views in the wrong class" —
they are **dangling**. Each references `tiles/tile_NNNN.png`; no `tiles/` dir exists,
`data/atlas/` is not in `artifacts.RELOCATED_PREFIXES` so nothing resolves out-of-tree,
and their builder went with the deleted atlas round-1/round-2 cluster. Neither viewable
nor rebuildable. Recorded with a recommendation (delete the files, not the convention) in
storage_classes.md; not acted on unilaterally.]`

## What's well-built (calibration)

The Rust core remains the healthy part: **24 files / 18.1k LOC**, two clean seams,
real `//!` rationale, f64-vs-perturbation ground-truth tests. The storage-class
regime (`tools/paths.py` → `tools/corpus/artifacts.py`) is a genuinely good design —
naming durability at the *write site* and asserting it at call time is the right
shape, and `paths.py` explicitly delegating to the one existing `ARTIFACTS_ROOT`
resolver instead of building a second is the right instinct. (One nit: the general
helper `tools/paths.py` depends on the more specific `tools/corpus/artifacts.py` — a
layering inversion. If `artifacts.py` moved up to `tools/`, both would sit at the tier
their scope implies.) The guard tests (`test_repo_size_guard.py`,
`test_tracked_artifacts.py`, `test_docs_tree.py`, `test_storage_classes.py`) are how
the file-count bomb stayed defused.

## Recommended, cheapest first

> **Status, 2026-07-31 hygiene pass: items 1–4 are DONE.** Items 5–6 were explicitly held
> back as seams. See the pass report for what was done and what was deliberately not.

1. ~~**`git lfs prune`**~~ — **DONE**, 268 MiB reclaimed. Note the correction above: the
   remote-verification guard gave a *false negative* and had to be re-checked by hand
   against the LFS batch API before pruning was safe.
2. ~~**Delete `data/v7/` and `data_large/`**; move the loose analysis docs; fix `README.md`
   and the `pyproject` `name`~~ — **DONE**, minus the two audit docs, which `27b68a4` had
   already deleted.
3. ~~**Add `testpaths` to `pyproject.toml`**~~ — **DONE**; collection verified unchanged at
   **754 tests** before and after, over exactly the five dirs that hold tracked test files.
4. ~~**Give `tools/` a README index**~~ — **DONE**: `tools/README.md`, one line per
   subdirectory, liveness by two methods that fail differently (import graph + committed
   artifact map) with their union *and their disagreement* reported. **The candidate list
   guessed above did not survive contact with the evidence**: `atlas_probe/` is on the
   production reward path (`production_seeder` imports `step0_reanalysis`), `v5`/`v6` are
   not wholly dead (each `build_plan.py` is exec'd by its recipe-parity test), and
   `render_mode_pilot/` has 5 committed producers. `v4/` is the one directory where every
   module is dead by both methods. Nothing was deleted — the list is evidence, not a
   decision.
5. **Finish Phase 2 of `repo_size_audit.md`:** route `src/root_field.rs`'s `CACHE_DIR`
   through a Rust-side `ARTIFACTS_ROOT` twin and relocate the 1.02 GB of `.f32`. This
   is the last real byte offender in the source tree *and* the last storage-class
   violation.
6. **Longer-term:** stop interleaving tracked metadata with ignored bulk in the same
   directory, which is what would let `.gitignore` drop from 64 negations to a handful
   of rules. And retire the copy-forward `tools/vN/` pattern in favor of one
   parameterized driver — the next version otherwise adds a 7th `build_plan.py`.

## Predecessors — where the two root audits went, and why nothing was restored

Both were **deleted** in `27b68a4` ("deleting old files") on 2026-07-31, which is *after*
this analysis was written (@ `73afd8b`) and *before* it was acted on. So §6/§7 above, and
the standing instruction to give each a superseded header, describe a root that no longer
exists.

The hygiene pass **did not restore them to stamp them**. The reason is the point those
sections were making: the harm was stale measurements sitting where they read as current,
and deletion removes that harm more completely than a header does. Restoring two documents
reporting ~640k files / 21 GB — against a tree of 30k / 7.7 GB — in order to write
"superseded" at the top of each would put the wrong numbers back in `docs/design/`, where
they would read as curated rather than retired.

What they were, so a reader who finds them in history knows what they are:

| doc | its subject | its headline number | status |
|---|---|---|---|
| `repo_structure_audit.md` | the **file-count** bomb behind a `grep -rl` that took >120 s | ~620k files in tree | deleted `27b68a4`; the count is now 30,284 and a full walk is 0.73 s |
| `repo_size_audit.md` | the **byte** weight left after the file-count restructure | ~21 GB on disk | deleted `27b68a4`; the tree is now 7.67 GB |

Both are recoverable: `git show 27b68a4^:repo_size_audit.md` and
`git show 27b68a4^:repo_structure_audit.md`. Two of their recommendations are still live and
are carried forward in this file rather than in them: the Rust-side `ARTIFACTS_ROOT` twin for
`src/root_field.rs::CACHE_DIR` (item 5) and the de-interleaving that would let `.gitignore`
shed its 64 negations (item 6). **Anything else either of them recommends should be checked
against this file first** — they measured a different tree.
