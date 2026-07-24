# Repo-size guard — the "stays small" constraint (registry mode, no moves)

Standing goal: fractal-generator's working tree should be ~what's checked into git — source + irreplaceable metadata + `out/`. Large things don't live in-tree without an explicit, written-down reason. Turn that into an enforced, self-documenting constraint. **This prompt builds the guard only — no relocations, no deletes, no commits, no touching tracked files.** Its populated registry doubles as the authoritative worklist for the moves that follow.

## Build the scan
Extend `tools/audit/disk_audit.py` (or a sibling under `tools/audit/`) with a working-tree size scan:
- Walk the **working tree** (the filesystem, not `git ls-files` — a file can bloat the tree while gitignored; that's the whole point).
- Flag: (a) any **file** ≥ 1 MiB (match the pre-commit blob hook), and (b) any **directory** whose aggregate ≥ ~100 MB (catches many-small-file dirs like label crops that no single-file rule would). Both thresholds are constants — tune if the report is noisy or sparse.
- Exclude these prefixes from flagging: `out/`, `.venv/`, `target/`, `target-test/`, and `.git/`. `.git` is a history-rewrite target, not a relocation one — report its size as an FYI line, don't flag it.
- The sibling trees (`../fractal-generator-artifacts`, `-trash`) are outside the walk root by construction.

## The registry — this IS the deliverable
One explicit allowlist, same spirit as the canary's `TRACKED_CANARIES`: the sanctioned-large-in-tree registry. Populate it with **every current violator** the scan finds; each entry, at a stable granularity (path prefix / dir root, so intra-dir churn can't flake the test), records size + tracked-or-not + a disposition:
- `KEEP: <reason>` — legitimately stays in-tree (irreplaceable tracked metadata with no smaller form). This is the "extremely good reason," written down. Being tracked is **not** an automatic pass — every large thing needs a stated reason.
- `RELOCATE -> <tier>` — pending a move; the line is deleted when the move lands. Tiers are disposition labels only (don't create dirs or wire paths — the precious-store *location* is still undecided): `artifacts` (regenerable bulk), `precious-store` (irreplaceable binaries), `trash` (dead).
- The `.pt` checkpoints: active model + its rollback anchor → `precious-store`; superseded versions → `trash` (old versions won't be retrained). If a checkpoint is a **canary path** (the v5 weight), tag it and note in the registry that its eventual move needs a deliberate canary update — but change nothing now.

## The test — live today, ratchets to fully-enforcing
A pytest test that **fails on any flagged violator not covered by a registry entry**, so new bloat is caught from today. Separately **report** (don't hard-fail) any registry entry that no longer has over-threshold content, as a nudge to delete the line. As things relocate, their `RELOCATE` lines come out; when only `KEEP` lines remain, the guard is fully enforcing and every in-tree exception is explicit and reviewed.

Prove it goes **RED on purpose**: drop a temp ≥1 MiB file outside the allowed prefixes → the test fails and names it → remove it → green. Then confirm the registry covers the real tree exactly — green with no uncovered violators and no stale entries.

## Report + leave for Matt
Write `docs/findings/repo_size_guard.md`: the large-in-tree inventory grouped by disposition (KEEP / →artifacts / →precious-store / →trash), with sizes, per-group and running totals, and the `.git` FYI line. That table is the worklist for the relocation phases and supersedes the size-audit's by being guard-backed and live. **Nothing committed, nothing moved** — leave it all for Matt.
