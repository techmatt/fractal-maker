# Label corpus — durable schema (schema_version 1)

A **permanent, version-spanning** store of labeled fractal crops. Its one job: let the
aesthetic classifier train across every generator version's output **even though each
version's sampling metaparameters differ.** It does this by separating *what a crop is*
(version-invariant) from *how a sampler found it* (version-tagged).

## The core contract

A label attaches to a **location + render spec**, never to **how the sampler found it.**
Every row in `batches/<batch_id>/images.jsonl` is one labeling unit with three independent
blocks:

```json
{
  "image_id": "<batch-unique, filesystem-safe stem; the crop is crops/<image_id>.jpg>",

  "render": {
    "cx": "-0.223773887498737",   // high-precision DECIMAL STRINGS, never floats —
    "cy": "-0.735076228354636",   //   the frame center/width that re-renders this crop
    "fw": "0.009992443630274168", //   forever. (Shallow f64 batches store the f64's
    "maxiter": 2000,              //   exact decimal; deep batches store true arb-prec.)
    "palette": "RdGy",
    "composition": "center",      // center | thirds | golden  (the present offset name)
    "width": 1280,
    "height": 720,
    "ss": 4,                      // grid supersampling factor
    "filter": "lanczos3",         // downsample reconstruction filter
    "interior_mode": "black"      // non-escaped pixel fill
  },

  "provenance": {
    "generator_version": "guided_descend_rev4",   // ALWAYS present
    "batch_id": "2026-06-24_guided_descend_rev4",  // ALWAYS present
    "root_src": "flat",           // everything below MAY be null or absent per version
    "branch": "foci",
    "depth": 3,
    "target_depth": 6,
    "walk_id": 1,
    "placement": "center",
    "focus_score": 5.737703364348315,
    "draw_index": 2,
    "seed_index": 2,
    "black_fraction": 0.041,
    "interior_frac": 0.000035,
    "occupancy": 0.4149,
    "void_guard": false
  },

  "label": { "score": null, "labeler": null, "labeled_at": null }   // score ∈ {null,1,2,3,4}
}
```

### `render` — version-invariant. The classifier sees ONLY this (+ the crop).
**Always present, identical field set across every batch.** It is a pure function from
`render` → crop: `crops/<image_id>.jpg` is rebuildable from `render` via `present`/`render-one`.
`cx`/`cy`/`fw` are **decimal strings**, because an f64 center is meaningless at deep zoom and
the store must outlive the depth regime that produced any one batch.

### `provenance` — version-tagged. **Freely allowed to differ or be absent across batches.**
This is *how the sampler found the location*: root mixture source, descent branch/placement,
depth, walk id, draw/seed indices, the per-frame measures (`black_fraction`, `interior_frac`,
`occupancy`), `void_guard`, and the batch's sampling metaparameters **by reference** (see
`batch.json`). For the **bias loop only** — analysing which sampler settings yield good labels.
Fields a given generator version never produced are `null`. **Nothing is fabricated:** a flat
sample does not get invented rev4-style provenance.

### `label` — the human verdict.
`score`: `null` (unlabeled) | `1` (bad) | `2` (okay) | `3` (good) | `4` (exceptional wallpaper
emission). Class 4 is a **fourth tier on the same quality scale**, not a separate head or a new
floor — final emissions can be class 3 or class 4; class 4 is preferred where available. See
`docs/design/label_rubric.md`. `labeler`, `labeled_at` filled when scored. **`null → value` is
the ONE allowed mutation to an original label anywhere in the store.**

### Revisions — the amendment overlay.
An existing label can be **revised** (a q3 demoted to q2, or promoted to q4). Because a revision
can move the `>=3` (good) boundary — demotions as well as promotions — and at least one batch is
the frozen evaluation census for the current model, **revisions never modify the original label
file.** They go to a separate registered amendment stream (`labels/<revision>.json`, registered
in `label_store.AMENDMENT_LABELS`, written by `tools/corpus/merge_amendments.py`).
`label_store.resolve_score` **prefers the amendment when one exists and falls back to the original
otherwise**, so the pre-revision label stays recoverable for any row and reconstructing the
original `>=3` boundary is the one-liner `resolve_score(row, sidecar) >= 3` (no `amendments`
argument). The amendment is authored against a source row's coordinate identity, so a revision
follows render identity, not image_id.

## Hard invariants

1. **Nothing labeled is ever deleted or overwritten.** Every sample is precious.
2. The **only** allowed mutation to an original label is its `score` going `null → value`. A
   merge that would change a non-null score **warns and refuses** — it never silently clobbers
   (`merge_scores.py`). A *revision* of a non-null label is not an exception to this: it is not
   written in place at all, but to the separate amendment stream (`merge_amendments.py`), leaving
   the original file byte-for-byte intact and recoverable.
3. The classifier trainer (`corpus_reader.py`) reads **only `(render-crop, label.score)`** and
   **unions every batch blind to `generator_version`.** Provenance NEVER enters training — which
   is exactly why v4's metaparameters not matching v1's costs nothing on the training side.
4. The bias loop reads `(provenance, label)` **within compatible batches** only.

## Layout

```
data/label_corpus/
  CORPUS_SCHEMA.md          # this file — the durable spec
  schema_version.txt        # "1"
  batches/<batch_id>/
    batch.json              # batch-level manifest (below)
    images.jsonl            # one row per labeling unit: {image_id, render, provenance, label}
    crops/<image_id>.jpg    # the 1280×720 q90 crop (a pure function of `render`)
    scores.json             # harness export: { image_id: 1|2|3|4 }, merged into images.jsonl labels
labels/<revision>.json      # (repo root) amendment stream: revised labels, resolve_score prefers it
```

`batch_id` = `<date>_<generator_version>` (e.g. `2026-06-24_guided_descend_rev4`).

`batch.json` carries: `created`, `labeler`, `generator_version`, `source_run`, the full
`sampling_metaparameters` block (root_mix, target_mix, zoom band, depth range, black_cap,
candidates, fps_radius, root_zoom_8k, orbit band, seed, … — whatever the version had;
absent/`null` for versions that lack a concept), `present_gates` (black_thresh, occupancy_floor,
edge_floor, tile_grid), and `render_defaults` (the `render` block's batch-constant fields).

## Git policy

**Commit** (tiny, irreplaceable): `CORPUS_SCHEMA.md`, `schema_version.txt`, every batch's
`batch.json`, `images.jsonl`, `scores.json`.

**Gitignore** `crops/` — a crop is a pure function of its row's `render` block, rebuildable via
`present` / `render-one`. (See the repo `.gitignore` rule for `data/label_corpus/batches/*/crops/`.)
