# Weights → git-LFS (active) + trash (superseded)

The size-guard registry currently tags the trained `.pt` weights `RELOCATE -> precious-store` (and superseded ones `-> trash`). The decision has since changed: the **active** weight set stays **in-tree via git-LFS** — a runnable checkout wants the weights, LFS keeps the blob out of the main pack (pointer in history), and the canary keeps guarding them. There is no out-of-tree precious store. **No commits. No relocation of any tracked file. No history rewrite in this run** (see the Step 1 branch). Trash moves go to `TRASH_ROOT` (`C:\code\fractal-generator-trash`), never `rm`.

## Step 1 — establish the current LFS state (this decides the whole job)
Check `.gitattributes` and the stored form of the tracked classifier weights `data/classifier/v{5,6,7}/model_best.pt`: are they **git-LFS** (committed as ~130-byte pointers) or **raw blobs** in history? Use `git check-attr filter <path>`, inspect the `.gitattributes` lfs rules, and confirm against the stored object. As a byproduct, report how much of the 352 MB `.git` is current tracked weights vs dead old blobs.

**Branch:**
- **All of v5/v6/v7 already LFS** → proceed to Step 2 (clean, safe path — no tracked weight moves, no history rewrite).
- **Any tracked weight is raw-committed** → **STOP.** Migrating it is a `git lfs migrate` history rewrite that merges with the P5 `filter-repo` job — attended and coordinated, not an unattended step. Report which weights are affected and halt; we'll design that separately.

## Step 2 — pin the LIVE set from the CODE, not the tree
From `ACTIVE_CKPT` and what the live gate / pool / mining path actually loads, determine:
- **Classifier:** v7 (live gate) + v6 (one-flip rollback) + v5 (deeper rollback) = the KEEP set. These are already tracked + canaried, so they need **no file action** — only a registry re-tag (Step 5).
- **Trained heads:** which version each live stage wires — `data/wallpaper_head/` (the active wallpaper-quality head is v3; confirm from code) and `data/render_mode_head/` (confirm the live version). These are currently **untracked** → Step 3.
Anything trained but outside that live set is superseded → Step 4.

## Step 3 — LFS-track + canary the active untracked heads
The live `wallpaper_head` and `render_mode_head` weights are untracked, and a runnable checkout needs them. Bring each under LFS: add the `.gitattributes` lfs rule for its path, stage it (leave uncommitted for Matt), and add it to `tests/test_tracked_artifacts.py`. Confirm each commits as an LFS pointer, not a raw blob. **The >1 MiB pre-commit hook must check the *staged blob* (the pointer), not the working-tree file** — verify it passes these LFS files while still rejecting a raw large file; fix the hook to check the staged blob if it currently measures the working file.

## Step 4 — trash the superseded weights + dead scratch
Move to `TRASH_ROOT` (all git-untracked → pure moves, zero git impact; verify each with `git ls-files --error-unmatch` failing, and confirm none is in the Step-2 live/rollback set):
- superseded classifiers `data/classifier/v{2,3,4}/`, `data/classifier/v5_seed1/`
- superseded head versions (`wallpaper_head` v1/v2, any non-live `render_mode_head`)
- the non-weight trash lines: `data/focus_diag/`, oversized `scratchpad/` content.

## Step 5 — registry + verify
- Update the guard `REGISTRY`: the `-> precious-store` tier is gone — v5/v6/v7 + the two LFS'd heads become `KEEP: git-LFS <reason>`; delete the executed `-> trash` lines.
- Verify, in order: `production_seeder --smoke` loads the active model (proves the live weights resolve) → `test_tracked_artifacts` canary green with the new head entries → `test_repo_size_guard` green (registry matches tree) → full `uv run pytest`.
- Report: the LFS state found, the live-set determination with its code evidence, what was LFS'd / canaried / trashed, and the remaining worklist (should be just the 12.4 GB `-> artifacts` bulk). Nothing committed; trash reversible until Matt's `rm`.
