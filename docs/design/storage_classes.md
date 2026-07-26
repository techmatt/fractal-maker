# Storage classes — the durability contract

Every file this project writes belongs to exactly one durability class. The class is
declared **at the write site** through `tools/paths.py`, and the choice is binding:
it decides whether the file is expected to survive, and therefore whether its absence
is ever a problem. This note is the contract those functions enforce. It is rules, not
commentary.

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

## Where this is enforced

- `tools/paths.py` — the four class functions; `durable()` carries the write-time
  gitignore assertion.
- `tests/test_storage_classes.py` — guards that assertion (a durable path git would
  discard is rejected).
- `tests/test_no_out_dir.py` — tripwire: the retired `out/` directory must not exist
  or be recreated.
