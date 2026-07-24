# History rewrite — v6/v7 weights → LFS + drop the ~300 MB dead history (ATTENDED)

`data/classifier/v{6,7}/model_best.pt` are raw-committed 34 MB blobs (v5 is already LFS). One coordinated history rewrite converts them to LFS pointers and drops the ~300 MB of dead history, taking `.git` from 365 MB to single-digit MB. **This is the highest-risk operation in the whole storage arc — a wrong path/blob selection can delete a currently-tracked file from all history. It runs ATTENDED, with hard stops before anything irreversible, and every destructive step is preceded by the backup below.** Repo is solo: one machine + a personal GitHub remote, no other clones or forks.

## Rule 0 — back up before touching anything
Full mirror backup of the entire repo including `.git` to a timestamped dir outside the tree (e.g. `C:\code\fractal-generator-prewrite-backup-<date>`) — `git clone --mirror` plus a copy of the working tree, or a full filesystem copy of `C:\code\fractal-generator`. This is the undo for the whole operation (and what makes `filter-repo --force` on a non-fresh clone safe). Do not proceed until it exists and is verified non-empty.

## Step 1 — enumerate the sets precisely, then STOP for confirmation
Produce three explicit lists and present them before executing anything:
- **MIGRATE → LFS:** the tracked classifier weight paths (`data/classifier/**/model_best.pt`) — all historical versions convert. v5 already LFS; v6/v7 convert.
- **PURGE from history:** `to_delete/**` plus the specific superseded artifacts you found (`pool_colormaps.json`, the old contact sheets, the dead embeddings) — list each by path/blob-id. **Every purge target must be confirmed gone-from-HEAD** (`git ls-files <path>` empty / blob unreachable from HEAD). Anything still tracked at HEAD is excluded and flagged.
- **PRESERVE-allowlist (never purge):** every currently-tracked large file — the migrated weights, `data/library_embeddings/embeddings.npz` (canary), `data/palettes/**`, and everything else at HEAD. **No blunt `--strip-blobs-bigger-than`** — it would vaporize the tracked `embeddings.npz`. Purge only by the enumerated paths/blobs.
- **HARD STOP:** show both lists + the preserve-allowlist + the projected `.git` size, and get explicit confirmation before Step 2.

## Step 2 — execute the rewrite (local, still reversible via the backup)
`git lfs migrate import` for the weight paths, then `git filter-repo` to purge the enumerated dead paths/blobs. Ensure `.gitattributes` carries the LFS rule for the weight paths so the pointers stick. Nothing pushed yet.

## Step 3 — verify BEFORE pushing (the go/no-go)
All must pass, in order:
- **Canary green** (`test_tracked_artifacts.py`) — the net proving no tracked artifact was dropped. This is the gate.
- Weights are now LFS pointers — `git check-attr filter` = lfs and the HEAD blob is a ~133-byte pointer for v6/v7 (and v5).
- **HEAD working tree byte-identical to the backup's** — content-compare (hash) every tracked file in both checkouts; zero differences. Proves the rewrite changed history/storage-form only, never file content.
- `.git` shrank to ~single-digit MB (`git count-objects -vH` after `git gc`).
- Full `uv run pytest` green (incl. the size guard).
- `git lfs fsck` / `git lfs ls-files` clean.
- Report every result. **HARD STOP** — the push is the point of no return; get explicit confirmation before Step 4.

## Step 4 — push (irreversible for the remote)
Solo repo, and the working copy is already the rewritten one, so there's nothing to re-clone. If branch protection is on the default branch, Matt toggles it off in the GitHub UI (call this out), then: `git push --force-with-lease` for all branches and tags, `git lfs push --all`, and Matt re-enables protection. Note for Matt: GitHub keeps the old objects server-side until its own GC — every clone is clean immediately, but GitHub's reported repo size lags; that's expected, not a failure.

## Not in this pass (safe follow-up on the clean base)
Once verified and pushed, the rest of the weight tier is rewrite-independent and safe: LFS-add + canary the two live untracked heads (`wallpaper_head` v3, the live `render_mode_head`), trash the untracked superseded weights (`classifier/v{2,3,4}`, `v5_seed1`, `wallpaper_head` v1/v2, plus `focus_diag` / dead scratch) to `C:\code\fractal-generator-trash`, and re-tag the size-guard `REGISTRY` (`KEEP: git-LFS` for the weights, retire the executed trash lines). Runs in the same attended session after Step 4, or as the next prompt. Leave all additive changes uncommitted for Matt.
