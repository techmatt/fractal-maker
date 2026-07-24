# Overnight storage restructure — move regenerable bulk out of the working tree

You recently wrote `repo_structure_audit.md`. This executes its core finding: ~640k files / ~85 GB sit in the working tree, ~99.98% regenerable ML scratch, which is why a plain `grep -r` takes >120 s and the `.gitignore` is a 150-line negation machine. Fix it by getting the regenerable bulk out of the source tree — **without losing anything, and with zero irreversible operations tonight.** This runs unattended; safety rails matter more than completeness.

## The one invariant — read this twice
**Every move/delete operates ONLY on git-untracked paths.** `git ls-files --error-unmatch <path>` is the gate; the tracked-artifacts canary and `tools/audit/disk_audit.py` are the backstops. If git tracks it, it is untouchable tonight — no move, no delete, no edit.
- **No literal `rm` anywhere.** Destructive == a *move* to a trash sibling (below). The only irreversible act tonight is one Matt runs himself in the morning.
- **No git operations.** No `git add`, no commit, no stage. Leave every git-visible change (new resolver code, `.gitignore` edits, the pre-commit hook, rescued docs) uncommitted for Matt to review.
- **When unsure whether something is regenerable or live: relocate it (keeper side), never trash it.** Reversible beats irreversible.
- Before starting: confirm no orchestrator / seeder / render process is live (something could be mid-write). If one is, abort and report.

## Two destinations, both siblings on the same volume (so moves are fast renames)
- `ARTIFACTS_ROOT` = `C:\code\fractal-generator-artifacts` (override via env `FRACTAL_ARTIFACTS_ROOT`). Keepers land here and stay wired into the pipeline.
- `TRASH_ROOT` = `C:\code\fractal-generator-trash`. Cruft lands here; Matt deletes it in the morning.

Create both if absent. If either would cross a volume boundary (making moves a slow copy+delete), background the move, checkpoint, and cap it (see runtime).

## Per-path decision table (classify each subtree, then act)
1. **git-TRACKED** (source, committed metadata, canary paths) → **NEVER TOUCH.**
2. **untracked + irreplaceable** (hand-authored `.md`/analysis notes/config not regenerable — e.g. durable docs living in gitignored `out/`) → **RESCUE** to a tracked home under `docs/` (or `store/`). Leave uncommitted. Apply the canary's own criterion: unregenerable ⇒ rescue.
3. **untracked + regenerable + kept/live-read** (current aug_cache; campaign ledgers+scratch; corpora caches) → **RELOCATE** to `ARTIFACTS_ROOT`, wired through the resolver.
4. **untracked + regenerable + disposable-by-convention** (`out/` scratch, scratchpad temp, build caches like `target/` `target-test/`) → **stage to `TRASH_ROOT`** (after rescuing any tier-2 stragglers out of it first).
5. **INTERLEAVED dirs** — committed metadata + regenerable scratch in the SAME directory (`data/label_corpus/batches/<id>/` = tracked `images.jsonl` beside ignored `crops/`, and similar for `library/`, `library_embeddings/`) → **LEAVE ENTIRELY.** Per-file separation is a later, attended pass. Do not attempt it tonight.
6. **unsure** → tier 3 (relocate), never tier 4.

## Named big targets (apply the rule broadly, but these are the point)
- `data/**/aug_cache*` (all classifier versions, ~395k files) → `ARTIFACTS_ROOT`.
- `data/discovery/campaign2/` (~317k files; holds live-intake-read ledgers + dead per-node scratch — move the whole dir, ledgers travel and stay resolver-readable) → `ARTIFACTS_ROOT`.
- `out/` (rescue tier-2 first, then stage the rest) and scratchpad temp → `TRASH_ROOT`.
- These two relocations alone are ~87% of the inodes — they kill the grep bug on their own.
- Any *other* cleanly-gitignored regenerable subtree you find: same rule. Interleaved dirs stay.

## Mechanism to build (all uncommitted)
- **Resolver** — one small module resolving `ARTIFACTS_ROOT` (env or default sibling), cross-platform via pathlib. Route the RELOCATED path families through it. Sweep for **readers AND writers** of those families (grep the path strings + the constants that build them); a missed *writer* silently regenerates the bomb in-tree, so be thorough. Leave all non-relocated paths exactly as they are — the resolver is additive and narrow.
- **`.gitignore`** — remove only the now-dead ignore/negation lines for the relocated roots. Leave the rest (the interleaved-dir stanzas stay until the split).
- **Pre-commit hook** — reject blobs over ~1 MB (portable to the Windows git environment). This is the recurrence guard for the "oops committed 300 MB" class.
- **Reappearance tripwire** — a test that goes red if any relocated root repopulates under its old in-tree path. This is the backstop for a missed writer.
- Prove the hook and the tripwire can each go **RED on purpose**, then revert — this repo has shipped guards that measured nothing; a guard unproven-red is not trusted here.

## Verification gate (green before AND after)
Full `uv run pytest` → `overnight_orchestrator --mini` → `prospect_orchestrator --mini` → `production_seeder --smoke`. Wait past each phase's gate so pool/emit actually fire (an exit-0 from a skipped phase is a false green).
- **Honest gap to state in the report, not paper over:** the aug_cache *write* path is not exercised by the minis (they don't train). Its rewiring is verified by grep-completeness + the reappearance tripwire only — say so plainly.
- If any gate goes red you can't cleanly explain as a pre-existing/unrelated issue: **STOP, leave the tree recoverable, write the report.** Do not attempt to auto-fix a broken tree overnight.

## Runtime discipline
The two big moves are ~300k+ files each. Estimate runtime first; if a unit exceeds ~30 s, background it and poll. Checkpoint between subtrees; don't start a move-unit you can't finish within a sane wall-clock budget, and add a hard-kill backstop for a hung move. Same-volume moves should be near-instant renames — if one is crawling, that signals a cross-volume copy; stop and report rather than grind.

## Deliverable — `docs/findings/storage_restructure.md` (uncommitted)
Every classified subtree → tier → action → destination → size/file-count. Plus: the resolver's reader+writer sweep coverage (what you routed, what you confirmed has no other references); the exact contents staged in `TRASH_ROOT` with the one-line `rm -rf` for Matt to finalize; what you rescued and where; the interleaved dirs left for the split with why; and any honest gaps (starting with the aug_cache write-path one). Do not commit it — leave the whole tree for Matt.
