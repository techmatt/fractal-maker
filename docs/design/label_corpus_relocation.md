# Label-corpus crop relocation — moving `crops/` + `vivid/` out of the working tree

**Status:** plan, recorded before execution (2026-07-29). Scope: relocate the label
corpus's regenerable crop bulk out of the working tree behind the existing
`tools/corpus/artifacts.py` resolver, exactly as the aug-cache and discovery-scratch
families already are. The **labels stay in-tree**; only the rebuildable JPGs move.

This doc is the durable half of the work. The migration report (file counts, gate
results) is disposable and lives under `scratch/`; the reasoning below is what a fresh
reader could not cheaply rederive, so it is committed.

---

## 1. The tree, as measured

Under `data/label_corpus/batches/` (23 batches), on 2026-07-29:

| set | files | bytes |
|---|---:|---:|
| **`crops/`** JPGs (the model-facing canonical crop) | 1,918 | — |
| **`vivid/`** JPGs (the blue/orange companion the labeler judges from) | 1,904 | — |
| **crops + vivid together** (the move set) | **3,822** | **~1,470 MB** |
| everything else (the labels — see below) | 71 | tiny |
| **total** | **3,893** | **~1.5 GB** |

So **98.2% of the files and effectively all of the bytes** under the corpus are the two
crop trees. The whole non-excluded working tree is ~5,293 files; this one subtree is
**~72% of everything a recursive `grep`/`find`/editor-index has to walk.** That traversal
cost — not repo size — is the harm the no-bulk-in-tree rule (`storage_classes.md` rule 5)
exists to prevent, and it is why relocation is worth the seam.

### The 71 files that stay

All 71 non-crop files are git-**tracked** and are the one unrebuildable thing in the
corpus — the labels and the coordinates they dereference:

- 23 × `images.jsonl` (the version-invariant render block: cx/cy/fw + geometry — the crop
  is a *pure function* of this, so the crop is rebuildable and the row is not),
- 23 × `batch.json`,
- 17 × `scores.json` (the human 1–4 verdicts),
- 5 × `blind.jsonl`, 1 × `reveals.json`, 1 × `probe_manifest.jsonl`,
  1 × `mb4_gather_v7_rescore.jsonl`.

**Zero** tracked files live under any `crops/` or `vivid/` dir — confirmed by
`git ls-files data/label_corpus/batches | grep -E '/(crops|vivid)/'` returning nothing.
The move therefore touches only gitignored files and cannot destage a tracked artifact.

**17 of the 71 are canary-listed** (`tests/test_tracked_artifacts.py::TRACKED_CANARIES`):
the 8 `scores.json`+`images.jsonl` pairs for the labeled batches plus the blindspot
`images.jsonl` (labels live only there). The canary guards *de-tracking*; this migration
must not touch the canary list, because it moves nothing the canary names.

## 2. Why the scope is `crops/` + `vivid/` only

The split is already declared in three independent places, and this migration just makes
the resolver honor it:

- **`.gitignore`** ignores `/data/label_corpus/batches/*/crops/` and `.../vivid/` while
  negating `!/data/label_corpus/` — i.e. the batch root is committed, the crop trees are
  not. (The same stanza block also ignores `_work/` and `sanity_contact_sheet.html`.)
- **The size-guard registry** (`tools/audit/size_guard.py`) carried one line:
  `data/label_corpus/` → `RELOCATE → artifacts`, reason "batch crops … labels stay
  in-tree". Once the move lands that disposition flips to **KEEP**, not stale: the crops
  leave the tree, but one tracked label file — the v2filtered `images.jsonl`, 1.4 MB of
  per-row provenance — still trips the ≥1 MiB rule and legitimately stays in-tree, so the
  entry is reworded to cover exactly that remaining KEEP-class violator.
- **The crop-rebuild contract** (`corpus_common.render_corpus_crop`, `CORPUS_SCHEMA.md`):
  every crop is a byte-reproducible function of its `images.jsonl` render block through
  `render-one --palette … --colormaps …`. Regenerable ⇒ `artifacts` tier ⇒ out of tree.

The labels are the exception to every one of those: no script rebuilds a human verdict.
So the batch root (71 files) stays in-tree and tracked; the crop bulk (3,822 files)
relocates. Moving the labels too would put irreplaceable data behind the resolver's
sibling directory for no traversal benefit — they are 71 tiny files.

## 3. The silent-zero hazard — the whole reason this is staged

A crop path is constructed by **hand-joining `"crops"`/`"vivid"` onto a batch dir** in
many places; there is no seam today (`corpus_common.py` stops at `batch_dir()`). If the
JPGs move while those constructions still point in-tree, **nothing crashes** — a glob over
`batches/*/crops/*.jpg` simply yields *zero* files, an `iter_labeled()` census yields
*zero* labeled crops, and **a training run on zero crops looks exactly like a successful
training run.** The failure is silent, and it corrupts the one thing the corpus exists to
feed.

### The in-scope construction sites

Enumerated 2026-07-29 (label corpus only — `data/wallpaper_corpus`,
`data/render_mode_corpus`, `data/q4_window_corpus`, the coevo round dirs, and the loose0
`data/label_crops` feed are **separate** stores with their own relocation story and are
explicitly out of scope):

- **30 in-scope Python sites** were migrated — 34 distinct crops/vivid construction
  statements across them (a few sites build a `crops`+`vivid` pair, or touch the tree in
  both a build and a report/verify stage): roughly 18 read, 16 write. An earlier scratch
  analysis put the figure at **62**. Record both so the next reader isn't left reconciling
  them: **62** counted at per-literal granularity (each `crops`/`vivid` string, plus the
  browser JS and the Rust `--crops-dir` sink, tallied separately); **30** is the count of
  Python files actually migrated under the crops/vivid-only scope. The load-bearing
  conclusion is identical either way.
- **How the scope was verified — the part worth keeping.** The first enumeration was
  delegated, and it **missed 3 in-scope sites** (`tools/eda/scale_2x2_{build_batch,
  label_analysis,labelbatch}.py`, all label-corpus readers/writers). They were caught **not
  by rechecking that enumeration** but by a **second, independent method**: a base-dir sweep
  for every file that *both* constructs a `crops`/`vivid` path *and* references a
  `data/label_corpus` base. Generalize this past the migration: **establish a scope by two
  methods that fail differently and trust their union — never one list rechecked against
  itself.** A single missed reader is the whole silent-zero failure mode, and re-running the
  method that missed it will miss it again.
- Most sites already derive the batch dir via `cc.batch_dir(<id>)`; the rest build it from a
  module-level literal, a CLI path arg, or the `CORPUS_DIR`/`BATCHES_DIR` glob's `dirname`,
  and route through the id-based seam once the bare `batch_id` is extracted
  (`os.path.basename` of the batch dir) — a trivial edit at each site.

**The one reader that is actually load-bearing — `tools/corpus/corpus_reader.py::iter_labeled`.**
This is the version-blind labeled reader the **v8 pipeline actually uses**: the anchor set
(`tools/scoring/active_ckpt`), `query_sampler`, and the batch builders all consume it. It
resolves labels through `label_store` (merged score ELSE registered sidecar) and now yields
crop paths through the seam. A silent zero here is *the* catastrophic case, so it is the
primary subject of the census gate (§5, and the invariant below).

**`classifier/corpus_data.py::load_corpus_rows` is NOT runnable on the current corpus** —
do not read it as a live loader (an earlier draft of this doc wrongly named it a second
load-bearing reader; that is the misleading direction, because a dead loader documented as
critical is what the next person protects and reasons around). It requires a
`provenance.seed_index` on every row (it keys CV/holdout grouping on it), and **no
crop-bearing batch added after v3 carries one**, so it raises before returning; on the full
corpus it raises even earlier — a `FileNotFoundError` on the loose0 batch, whose crops were
never materialized in the store. It fed `train_v2`/`train_v3` only, both superseded by v8.
Its single crop-path construction was migrated to the seam identically to `iter_labeled`'s,
and the loader was made to **fail loudly** naming the missing `seed_index` (in place of the
prior cryptic `int(None)` `TypeError`) — the message is its documentation. The census gate
(§5) therefore exercises `iter_labeled`, never this loader.

Two more readers are the most likely *silent*-miss sites because they never call
`cc.batch_dir` (so a naive "grep for `cc.batch_dir` + `/crops`" sweep skips them):
`tools/ranker/build_features.py:166` (prior-embedding feature cache, batch id baked into a
literal path) and `tools/corpus/verify_render_path.py:94` (per-batch render-parity check,
generic `batch_dir` CLI arg).

### Sites the Python seam cannot reach — handled elsewhere, by design

- **The browser** (`tools/viz/corpus_label.html:149,152`) builds crop URLs client-side:
  `ROOT + BATCH_DIR + '/crops/' + image_id + '.jpg'`. A Python helper cannot reach a
  JavaScript string. This is resolved **server-side**: `serve.py` transparently maps a
  request path under `data/label_corpus/batches/*/crops|vivid/` to the artifacts root
  (§4), so the page keeps requesting the in-tree URL and gets the relocated bytes.
- **The Rust engine** (`src/enrich.rs`, `--crops-dir`) writes crops to a path it is
  *handed*. Its label-corpus caller is `tools/mining/harvest.py:214`, which constructs that
  path — so routing the caller through the seam makes the engine write out-of-tree with no
  Rust change. `present.rs` is the same shape: it writes to `--out-dir`/`--flat-out`, both
  supplied by Python.

So there is **no genuinely unreachable site**: every Python construction routes through the
seam, the JS routes through the server, and the Rust routes through its Python caller. If
execution turns up one that does not fit this, the rule is **stop and report it** — one
unreachable reader is the entire failure mode.

## 4. The seam and the resolver

- **Seam** (`corpus_common`): `crops_dir(batch_id)` / `vivid_dir(batch_id)`, each returning
  `artifacts.resolve("data/label_corpus/batches/<batch_id>/crops|vivid")`. One place the
  crop-path string is formed; every reader and writer calls it.
- **Resolver** (`artifacts.py`): register the family as a **pattern**, not a literal — a
  predicate `_is_label_corpus_crop(rel)` true iff `rel` is
  `data/label_corpus/batches/<id>/{crops,vivid}[/…]` (component-exact on `crops`/`vivid`,
  so a `crops_staging` sibling does not match). Pattern, not per-batch literal, for the
  same reason discovery-scratch is a pattern: a new batch relocates with no registry edit,
  and a forgotten registration fails toward *out-of-tree* (conservative), never toward a
  silent bulk back in the tree.

Because the seam always goes through `resolve`, the registration flip (§step 2) is what
moves the resolved path from in-tree to out-of-tree — the seam code does not change between
steps.

## 5. Staged order, and why this order

The ordering exists so the risky, mechanical part (the 34-site sweep) is done and *proven*
while the old paths still work, and the irreversible part (the move) happens only after the
seam is proven and the resolver knows where the bytes went.

- **Step 0 — this doc.** Make the analysis durable before touching code, so it survives a
  half-finished migration.
- **Step 1 — seam + migrate all sites, files still in place.** Add `crops_dir`/`vivid_dir`,
  route all 34 sites through them. The family is *not yet registered*, so `resolve` still
  returns in-tree paths — every site keeps working. **Baseline gate:** record the full
  `(image_id, score)` set from a real `iter_labeled()` census to a file, and prove the suite
  is green and a census *through the seam* reproduces that set exactly.
- **Step 2 — register + tripwire.** Add the pattern to `RELOCATED_PREFIXES`'s pattern arm
  and extend `tools/audit/test_relocated_artifacts.py` to (a) scan `batches/*/crops|vivid`
  for in-tree reappearance and (b) prove it fires on an unregistered batch's crops. After
  this, `resolve` returns out-of-tree paths — the tree is transiently inconsistent (paths
  point out, files still in) until step 3, which follows immediately.
- **Step 3 — move + prove the training path.** Physically move the 3,822 files to
  `<ARTIFACTS_ROOT>/data/label_corpus/batches/<id>/{crops,vivid}`. Then **re-run the census
  and assert it equals the recorded baseline** (same `(image_id, score)` *set*, not merely
  the same count), and **run a one-epoch classifier smoke that asserts a non-zero
  labeled-crop count matching the census** — the assertion is on the number of crops loaded,
  because a smoke that "passes" on zero crops is precisely this migration's trap.
- **Step 4 — the browser.** Transparent `artifacts.resolve` in `serve.py` plus a `?crops=`
  override, then re-verify the four recorded fetches (the page, `images.jsonl`, `batch.json`,
  a crop) by loading a batch and confirming crops render — the labeling rig is in daily use,
  so it is verified by loading it, not by reading the diff.

### The before/after gate, concretely

The `(image_id, score)` set from `iter_labeled()` is the invariant: it must be **identical**
before step 1 and after step 3. Count-equality is not enough — a set comparison catches a
crop that silently moved to the wrong batch id or a label that changed resolution. The
working-tree file count (§1) is the *outcome* measure: it should drop by 3,822.

## 6. What this migration must NOT do

- Not move, restage, or touch any of the 71 tracked label files, and not touch the canary
  list (`TRACKED_CANARIES`) — the move relocates nothing it names.
- Not stage anything under a batch's `labels/`.
- Not special-case around an unreachable site — stop and report instead.
- Commit only the migration's own files, by explicit path.
