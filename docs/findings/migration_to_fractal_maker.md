# Repo migration: `fractal-generator` → `fractal-maker` (2026-07-24)

**Status: DONE and verified.** The active repository is now **`fractal-maker`**
(`git@github.com:techmatt/fractal-maker.git`, local `C:\Code\fractal-maker`). The
old `fractal-generator` repo is being renamed **`fractal-generator-deprecated`**
and will eventually be moved off this machine. This doc records exactly what moved,
what is verified working, and — critically — **what has NOT been carried over yet**
so nothing irreplaceable is lost when the old tree leaves.

## Why this happened

`fractal-generator`'s `.git` had bloated to **365 MB**. Root causes, from a full
history enumeration:
- Classifier weights `v6/v7/model_best.pt` were **raw-committed** (34 MB each,
  single copy). v5 was already git-LFS.
- **Dead history**: `to_delete/**` (~67 MB), 4 superseded `pool_colormaps.json`
  blobs (~40 MB), old discovery `contact_sheet.png`s, `reject_corridor/draws.jsonl`.
- The rest was legitimately-tracked corpus metadata churn (`images.jsonl`, ledgers)
  plus the live 20 MB `pool_colormaps.json`.

A premise correction worth recording: there was **never ~300 MB of dead weight** —
committed weight was only 68 MB, single-copy. "Single-digit MB `.git`" was
unreachable while preserving the live tracked palettes/corpus. So rather than a
force-push history rewrite, we started a **fresh repo** with the full tracked tree
as one clean commit; the old repo keeps the full history as an archive.

## What was done

1. **Safety backup** (still on disk): full mirror at
   `C:\code\fractal-generator-prewrite-backup-20260724` (`repo-mirror.git` = all
   history/branches/refs, + raw weight copies with verified sha256, + a 1247-file
   HEAD manifest). This is the undo for the entire operation.
2. **LFS-migrated** `v6/v7/model_best.pt` (glob `data/classifier/*/model_best.pt`)
   so all three weights are now LFS pointers; `.gitattributes` carries the rule.
3. Built **`fractal-maker`** as a single "Initial commit" of the **1247 tracked
   files** (byte-identical set to the old HEAD; verified by blob-OID diff — only
   `.gitattributes` + the two migrated weights differ, as intended).
4. **Pushed** to GitHub. Fresh-clone verified clean; weights smudge to full 34 MB.
5. **Recreated both environments** in `fractal-maker` from committed lockfiles
   (`uv sync` for `.venv`, `cargo build --release` for the engine) — *regenerated,
   not copied*, since a uv venv is path-bound and both are reproducible from
   `uv.lock`/`Cargo.lock`.

## `fractal-maker` — verified working (2026-07-24)

| Check | Result |
|---|---|
| Python env (`uv sync`) | `torch 2.6.0+cu124`, CUDA available, timm/sklearn import |
| Rust release build | compiles → `target/release/fractal-generator.exe` (crate still named `fractal-generator`) |
| Smoke render | valid 640×427 PNG |
| `cargo test --release` | all pass |
| LFS weights | v5/v6/v7 present, `git lfs fsck` OK, `torch.load` succeeds (real checkpoints) |
| Tracked-artifacts canary | 28/28 |
| `.git` (after gc) | 16.8 MiB objects (+~98 MB local LFS cache) |

**What `fractal-maker` contains:** all source, tooling, docs, palettes
(`pool_colormaps.json`, `palette_features.json`), the full label/wallpaper/
render-mode/q4 corpus **labels + `images.jsonl`/`scores.json`/`windows.jsonl`**,
classifier weights **v5/v6/v7 (LFS)**, the three **live trained heads** (wallpaper v3,
render-mode v1, palette pref-v3-gvo — LFS, carried 2026-07-24; see below), and the
`KEEP` artifacts (`library_embeddings/embeddings.npz`, `library/library_records.jsonl`,
`calibration/energy_calibration.json`, `data/atlas/**` seeder state, test fixtures).
The q4 workstream labels/scores and atlas seeder state are **fully carried**.

## Trained models — CARRIED (done 2026-07-24)

The three live trained heads had **no rebuild path and existed nowhere in git** — the
single highest-risk gap. They are now in `fractal-maker` as **git-LFS + canary**,
exactly like the classifier weights. Per decision, only the **latest canonical weight
of each** was taken — no v1/v2 history, no seed variants, no `model_last` — one file
per model. Configs are embedded in each checkpoint (`state_dict` + `config`), so the
`.pt` is independently loadable; the sidecar `config.json`/`metrics.json` were left
behind as redundant provenance.

| path (now LFS-tracked + canaried) | size | what |
|---|---:|---|
| `data/wallpaper_head/v3/model_best.pt` | 32 MB | LIVE cross-location wallpaper-quality head (CORN) |
| `data/render_mode_head/v1/model_best.pt` | 9.7 MB | LIVE strange-mode `mining_v1` gate |
| `data/queries/scorer/v3_gvo/model_best.pt` | 9.7 MB | LIVE palette-preference ranker (pref-v3-gvo, used in emission colorize) |

Wiring: `.gitattributes` carries an LFS rule per path; `tests/test_tracked_artifacts.py`
`TRACKED_CANARIES` lists all three (canary now 31/31). `git lfs fsck` OK.

**Left behind (intentionally):** the superseded `wallpaper_head` v1/v2 + all
`seed_*/model_best.pt`, `render_mode_head` seeds, the older scorer towers
(v1/v2/v3, `v3_gvo/model_last`), and per-head `config/metrics/train.log` provenance.
Recoverable from the archive if ever needed.

### Regenerable — rebuild in `fractal-maker` on demand (carry only to skip recompute)

- **Crops (regenerable via `present`/`render-one`):** `data/label_corpus/` (6.0 GB),
  `data/wallpaper_corpus/` (938 MB), `data/render_mode_corpus/` (851 MB),
  `data/label_crops/` + `data/label_crops` feeds (~1.1 GB).
- **Caches:** `data/root_field/` (1.1 GB, Rust dump), `data/library/field_cache`
  (814 MB), `data/queries/` renders (757 MB), build manifests `data/v{4,5,6,7}/`
  (~377 MB), `data/ranker/**` fits (frozen-feature logistic), discovery run-state
  (`node_embs`/`morph_mem`/`distinct_looks` npz — per-run, rebuilt each run).
- **Trash / dead:** classifier `v2/v3/v4/v5_seed1` + `model_last` variants (~270 MB),
  `data/focus_diag/` (16 MB dead scratch).

## Rollback / safety state

- Old **`fractal-generator` GitHub repo** (→ `-deprecated`) still has the **full
  609-commit history + all branches** (`feat/render-mode-pilot`,
  `guard/v6-ledger-decode`, `label-corpus-run4-batch` — all confirmed **fully
  merged into `main`**, zero unique code).
- Local **mirror backup** at `C:\code\fractal-generator-prewrite-backup-20260724`.
- Keep both until the CRITICAL carry-list above is confirmed moved and
  `fractal-maker` is confirmed reproducible on another machine.

## Suggested next steps (in order)

1. ~~Move the three CRITICAL model trees into `fractal-maker`; LFS-add + canary.~~
   **DONE 2026-07-24** — latest single weight each, LFS + canary (31/31).
2. **Verify** the three heads load and score in `fractal-maker` (torch load done;
   a one-inference smoke through each head's deploy path still worth running).
3. Decide which regenerable bulk (if any) is cheaper to copy than recompute; leave
   the rest to rebuild on demand.
4. Refresh `docs/findings/repo_size_guard.md` + `CLAUDE.md` in `fractal-maker` to
   the new baseline (the old `.git`-rewrite worklist is retired by this migration).
5. Only after 1–4: rename old repo to `-deprecated`, then move/delete it and the
   mirror backup.

## Working-tree size note

`fractal-maker` on disk is ~5.4 GB, but that is **entirely gitignored/regenerable**:
`.venv` (~4.8 GB, torch cu124 + CUDA wheels), `target/` (~193 MB, Rust build), and
`.git` (~167 MB = 17 MB objects + ~150 MB local LFS cache). Tracked content is ~150 MB,
dominated by the six LFS weight objects + `pool_colormaps.json` (22 MB). The GitHub
repo is small; `rm -rf .venv target && uv sync && cargo build --release` reconstructs
the environment.
