# Carry-completeness audit — old-tree deletion gate (2026-07-24)

**Verdict: PASS (deletion gate met).** Every trained-weight file in the old tree
is CARRIED-by-content or INTENTIONALLY-LEFT — **zero MISSING weights**. The three
live heads **load AND score sanely** through their actual deploy paths. The broader
untracked-irreplaceable sweep surfaced exactly **one** uncarried hand-authored class
(4 gitignored `prompts/*.md`), now **safety-copied** to `C:\Code\fractal-maker-carry-review\`
so old-tree deletion cannot lose it. Read-only on the old tree; nothing deleted,
moved (beyond the safety copy), committed, or wired — the prompts and their disposition
are Matt's call.

Scope: `C:\Code\fractal-generator` (old, rename pending) + `C:\Code\fractal-generator-artifacts`.
Target of record: `C:\Code\fractal-maker`. This closes step 2 ("verify the heads
load and score") of `migration_to_fractal_maker.md`, plus a full untracked-file sweep.

---

## Part A — the weight sweep (primary)

Enumerated every `*.pt *.pth *.ckpt *.safetensors *.onnx` under both old trees
(excluding `.venv/*.pth`, which are Python path files, not weights).
`fractal-generator-artifacts` holds **zero** weight files (a `data/` tree only).
`fractal-generator` holds **38** weight files. Matched by **SHA-256 content hash**
against fractal-maker's 6 carried weights.

**Result: 8 CARRIED-by-content, 30 INTENTIONALLY-LEFT, 0 MISSING.**

Two content cross-checks fell out of the hashing and confirm the migration's "latest
single weight each" claim: the promoted `model_best.pt` is **byte-identical** to the
seed that produced it —
`render_mode_head/v1/seed_0` ≡ `render_mode_head/v1/model_best` (`a1a4ea11…`), and
`wallpaper_head/v3/seed_1` ≡ `wallpaper_head/v3/model_best` (`e2f53a25…`).

### CARRIED (same-hash object present in fractal-maker, LFS)

| old-tree path | sha256 (head) | carried as |
|---|---|---|
| `data/classifier/v5/model_best.pt` | `0c68c74b` | `data/classifier/v5/model_best.pt` |
| `data/classifier/v6/model_best.pt` | `57a94224` | `data/classifier/v6/model_best.pt` |
| `data/classifier/v7/model_best.pt` | `5050b085` | `data/classifier/v7/model_best.pt` |
| `data/queries/scorer/v3_gvo/model_best.pt` | `745cab27` | (same path) — LIVE palette-pref ranker |
| `data/render_mode_head/v1/model_best.pt` | `a1a4ea11` | (same path) — LIVE mining_v1 gate |
| `data/render_mode_head/v1/seed_0/model_best.pt` | `a1a4ea11` | ≡ v1/model_best (promoted seed) |
| `data/wallpaper_head/v3/model_best.pt` | `e2f53a25` | (same path) — LIVE wallpaper head |
| `data/wallpaper_head/v3/seed_1/model_best.pt` | `e2f53a25` | ≡ v3/model_best (promoted seed) |

### INTENTIONALLY LEFT (matches migration's known-left set)

Reason per group (all recoverable from the archive / mirror backup):

- **Superseded classifier towers** — `classifier/v2,v3,v4` (`model_best`+`model_last`)
  and `classifier/v5_seed1` (`model_best`+`model_last`): rollback/dead tier; only the
  v5 rollback anchor + v6/v7 are live.
- **`model_last` variants** — `classifier/{v5,v6,v7}/model_last`,
  `queries/scorer/v3_gvo/model_last`: last-epoch checkpoints; `model_best` is the
  canonical weight. (v2/v3/v4/v5_seed1 `model_last` covered above.)
- **Older scorer towers** — `queries/scorer/{v1,v2,v3}` (`model_best`+`model_last`):
  superseded by the carried `v3_gvo`.
- **Seed variants** — `render_mode_head/v1/seed_{1,2,3,4}`,
  `wallpaper_head/v3/seed_{0,2,3,4}`: per-seed eval variants; the promoted seed is
  carried (see cross-check above).
- **Superseded head versions** — `wallpaper_head/v1,v2` (`model_best`+`model_last`):
  pre-v3 wallpaper heads.

**⚠ MISSING (Part A): none.**

---

## Part B — untracked-irreplaceable backstop

Risk universe = old-tree files NOT git-tracked, **including gitignored** (the heads
were ignored). Built as `full working tree − tracked` via
`git ls-files --others` (4 unignored) + `git ls-files --others --ignored` (58 740
ignored) — not relying on `--exclude-standard` to filter. After excluding the
known-regenerable prefixes (`.venv`, `target*`, `out`, `data_large`, `*_corpus`,
`label_crops`, `library`, `root_field`, `data/v{4567}`, `discovery` run-state,
`__pycache__`), **1 183 candidates** remained. Classification below; two prior passes
did most of this work already — `migration_to_fractal_maker.md` (explicit
regenerable/left lists) and `docs/rescued/` (the `out/`-tree analysis-doc rescue).

### ⚠ MISSING → preserved (the actionable gap)

| path (old tree, gitignored) | why irreplaceable | action |
|---|---|---|
| `prompts/history-rewrite.md` | hand-authored task spec, no producer, not carried | preserved → **committed** to `docs/rescued/prompts/` |
| `prompts/repo-size-guard.md` | " | " |
| `prompts/storage-restructure.md` | " | " |
| `prompts/weights-lfs.md` | " | " |

The old tree's `prompts/` dir is gitignored; these 4 hand-authored prompts were never
committed and are not in fractal-maker (whose `prompts/` holds only this audit's
`carry-completeness.md`). They describe **already-executed** migrations (their outcomes
are committed as `docs/findings/*` + commits), so intrinsic residual value is low —
but they are hand-authored with no rebuild path, so per "err toward flagging" they were
first **safety-copied** to `C:\Code\fractal-maker-carry-review\` (sha256 below) and,
on Matt's call, **committed** into `docs/rescued/prompts/` (the same convention as the
`out/`-tree analysis-doc rescue) so they survive independently of the old tree.

```
474af947…  history-rewrite.md      509c0fc5…  repo-size-guard.md
0814ec2f…  storage-restructure.md  fe5f852f…  weights-lfs.md
```

### Cleared — regenerable / intentionally-left (no preservation needed)

- **The pref-ranker precedent is accounted for.** The "pref-ranker" named in the
  migration = `queries/scorer/v3_gvo` (palette-preference ranker) → **CARRIED** (Part A).
  A *different* `data/ranker/pref_loc_v{0,1}/{model,prior,features}.npz` also exists in
  the old tree — this is the **"frozen-feature logistic fits"** the migration explicitly
  marks regenerable; its producer/report tool is committed (`tools/ranker/report.py`,
  and `docs/rescued/out/pref_loc_v0_report.md`). LEFT/regenerable, **not** a fourth head.
- **Scratchpad is genuinely disposable** (the `visual_dup/embed.py` failure mode does
  NOT recur here). Both CLAUDE.md mechanical greps are **empty**: nothing outside
  `scratchpad/` imports a scratchpad module, and no scratchpad file writes a durable
  `data/` artifact. The 20 `scratchpad/*.py` are diagnostics with no dependents.
- **GPU embeddings** — `library_embeddings/shards/*.npz` (13, 520 KB total) are raw
  per-cycle shards **subsumed** by the carried consolidated `library_embeddings/embeddings.npz`
  (a KEEP artifact); `wallpaper_harvest/dedup_curves.npz` and `data/ranker/*.npz` have
  committed producers (`palette_extractor/harvest_dedup.py`, `tools/ranker/*`);
  `focus_diag/*.npy` are "dead scratch" (migration). No live threshold depends on an
  uncarried embedding.
- **Weight sidecars** — `data/*/inference.py` + `config.json`/`metrics.json` next to each
  weight: redundant provenance (configs are embedded in each `.pt`'s `state_dict`+`config`,
  as Part C confirms); left intentionally per migration.
- **`out/`-tree analysis docs** — already handled by `docs/rescued/` (hand-authored vs
  regenerable split done there).
- **Run outputs** — `data/curation/recolor_pass/report.md` (deterministic-pass report),
  `data/atlas_probe/*.csv`, `data/guided_descend/*` logs + `run_2x2.sh` (one-off
  experiment driver; params documented in its own header + `run_config.json`),
  `data/{mining,curation}/*.json`, ~808 `.png/.jpg` renders: regenerable views / run-state.
- **Uncommitted-but-ignored discovery** — the 4 `git ls-files --others` files are
  `data/discovery/runs/*/summary.json`+`guard_telemetry.jsonl`: per-run state, regenerable.

**⚠ MISSING (Part B): the 4 prompts above — all preserved.**

---

## Part C — the three heads LOAD **and SCORE** (in fractal-maker)

One inference each through the head's **actual deploy path** (not a bare `torch.load`),
on a real 1280×720 crop rendered in-repo via `render-one` at test location `test_01`
(`data/test_renders.json`: cx=-0.0566…, cy=0.6673…, fw=1.18e-4, maxiter=8000, palette
`default`). Every head loaded its config from its own checkpoint and produced a sane,
in-range score:

| head | deploy path exercised | output | range | verdict |
|---|---|---|---|---|
| `wallpaper_head/v3/model_best.pt` | `classifier.inference.load_scorer` → `score_paths` (target=ordinal, **num_classes=4** ⇒ 3 CORN logits, geometry=stretch) | `score_from_logits = 1.970` | [0,3] | ✅ sane |
| `render_mode_head/v1/model_best.pt` | `tools.mining.mining_gate.MiningScorer` → `score_paths` (marginal `p_ge` gate) | `p≥2=0.346, p≥3=0.270, E[ord]=1.127` | [0,2] | ✅ sane |
| `queries/scorer/v3_gvo/model_best.pt` | `tools.queries.scorer.train.build_model` + `data.build_transform(train=False)` (mobilenetv4_conv_small, squash-224, num_classes=1 margin ranker) | `pref = 5.275` | ℝ (order-only) | ✅ finite/sane |

Notes that confirm the deploy paths are the *right* ones (not just "a" load):
- The wrong loader **fails loudly** — `tools.mining.score_lib.Scorer` hardcodes the
  ordinal K−1=2 head and raised a `size mismatch [3,1280] vs [2,1280]` on wallpaper_head
  v3; the canonical `classifier.inference.load_scorer` reads `num_classes=4` from the
  checkpoint and builds the correct 3-logit head. This is exactly the "loads-≠-scores"
  trap the audit targets, caught and routed to the real path.
- The v3_gvo scorer is a **distinct architecture** (mobilenetv4_conv_small single-output
  pairwise-margin ranker, not a CORN head); scored through its own
  `build_model`+`build_transform`, output is a finite scalar whose only contract is
  relative order — sane.
- `render_mode_head` "fails" the `mining_v1` gate (thr=0.5) on this `default`-palette
  render. That is the gate **working** (a legitimate not-strange verdict), not a load error.

**Caveat:** single sample input (one location, one palette) — sufficient for the
load-and-score-sanely gate, not a distributional check.

---

## Deletion gate

- **Part A:** 0 MISSING weights. ✅
- **Part B:** 4 hand-authored prompts were uncarried; all **safety-copied** to
  `C:\Code\fractal-maker-carry-review\` → nothing irreplaceable is lost when the old
  tree is deleted, and are now **committed** into `docs/rescued/prompts/`. ✅
- **Part C:** all three live heads load and score sanely through their real deploy
  paths. ✅

**The old `fractal-generator` tree is safe to delete on carry-completeness grounds**,
pending only the separate "reproducible on another machine" check
(`migration_to_fractal_maker.md` §Rollback). The audit itself was read-only on the old
tree; the only follow-up write is this report + the 4 rescued prompts, committed into
`fractal-maker` (nothing deleted, moved, or wired).
