"""Shared label resolution — the single source of truth for "what score a row has".

A batch's hand labels live in ONE of two places:
  (a) merged into its images.jsonl `label.score` (via tools/corpus/merge_scores.py)
      — the loose0 / rev4 / rev4occfix batches; or
  (b) ONLY in a `labels/*.json` sidecar keyed by image_id, because the merge into
      images.jsonl was never run — the Julia (`julia_ladder_j0`), `mining`, `scale`,
      and the two jm-band revival batches (`jm3_band`, `jm45_band`).
A loader that reads `label.score` alone silently drops the (b) batches: for Julia
that wiped out the entire family (0 Julia locations), and it dropped the mining/scale
Mandelbrot labels too.

THE JOIN KEY IS COORDINATES, NOT image_id. image_id (e.g. `A_<idx>_<comp>_<palette>`) is a
slug that does NOT encode render scale, so it is NOT unique across batches built at
different scales — and `scale_2x2_labelset.json` is deliberately shared by two such batches.
Joining a label to a row by image_id could therefore hand a label to a same-id crop at a
different scale. So the resolution keys on `join_key` (the canonical location identity —
family + cx/cy/fw + c — plus palette/composition), and a sidecar's image_id→score map is
re-keyed onto that coordinate key through its OWNER batch's images.jsonl (`sidecar_for`).
This was a prose invariant (`CORPUS_SCHEMA.md`, the v5 build_manifest recipe); it is now
enforced in code and tested (`test_label_store_join.py`).

Every consumer that turns a corpus row into a label MUST route through this module —
`corpus_reader.iter_labeled` (the version-blind trainer view) and
`query_sampler.LocationPool.from_corpus` (the q2+q3 location universe) both do — so the
resolution logic + the `SIDECAR_LABELS`/`SIDECAR_OWNER` registries live in exactly ONE place
and the two can never drift. NEW unmerged batches MUST be registered in `SIDECAR_LABELS` (or
have their labels merged into images.jsonl); `assert_sidecars_joined` makes a broken join
loud at load.

REVISIONS go to a separate `AMENDMENT_LABELS` stream, never in-place: `resolve_score` prefers
the amendment when one exists and falls back to the original otherwise, so the pre-revision
label stays recoverable (`resolve_score(row, sidecar)` with no amendments = the original). See
the `AMENDMENT_LABELS` block below and `amendments_for`.

REFERENCE for the complete label set: the v5 unified classifier's training-data
assembly, tools/v5/build_manifest.py. It recovers the J0 Julia labels from
labels/location_labels_julia_ladder_j0.json JOINED to the batch's images.jsonl — the same
join mirrored here (image_id is unique WITHIN the julia batch, so it agrees with the
coordinate key). See data/label_corpus/CORPUS_SCHEMA.md.
"""
from __future__ import annotations

import json
import os

import location as _loc   # canonical Location.key() — the coordinate identity of a render

# repo root = two levels up from tools/corpus/
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LABELS_DIR = os.path.join(ROOT, "labels")
# The standard label corpus. `sidecar_for` reads the OWNER batch's images.jsonl from here to
# turn image_id-keyed sidecars into coordinate-keyed label maps; a non-default corpus (tests)
# passes its own batches dir.
BATCHES_DIR = os.path.join(ROOT, "data", "label_corpus", "batches")

# The (b)-case batches: batch_id -> its labels/*.json sidecar file. The registry is
# the single source of truth for "which batches carry labels only in a sidecar".
SIDECAR_LABELS = {
    "julia_ladder_j0": "location_labels_julia_ladder_j0.json",
    "2026-06-25_mining_v3guided_v1": "mining_v3guided_v1.json",
    "2026-06-25_scale_2x2_labelset": "scale_2x2_labelset.json",
    "2026-06-25_scale_controlled_2x2": "scale_2x2_labelset.json",
    # jm-band revival batches (labeled 2026-07-11/12; never merged in-row, empty
    # scores.json). Registering them recovers 58+71 crop labels for EVERY canonical
    # consumer (corpus_reader trainer view, query_sampler pool, atlas guard) at once.
    "2026-07-11_jm3_band_v1": "jm3_band_v1.json",
    "2026-07-12_jm45_band_v1": "jm45_band_v1.json",
    # Phoenix Phase-B seed-grid batch (500 items, labeled 2026-07-21; empty scores.json,
    # never merged in-row). Stratified FROM grid output (biased) -> train-side only for any
    # future CORN manifest; the varied-phoenix v7 calibration read lives in its own
    # stratification. See docs/design/phoenix_seed_sampler_spec.md §8.
    "2026-07-21_phoenix_grid": "phoenix_grid.json",
    # Native multibrot band batch (300 items, labeled 2026-07-22; empty scores.json,
    # never merged in-row). Stratified across v7 p_good bands incl. sub-threshold/rejects
    # -> train-side only for any future CORN manifest, never an unbiased base-rate source.
    # Sidecar name is the labeling export (2026-07-22_*), NOT the empty placeholder the
    # batch builder pre-created.
    "2026-07-22_native_multibrot_band_v1": "2026-07-22_native_multibrot_band_v1.json",
    # Interior-band batch (80 items, labeled 2026-07-28; combined reveal-audit export,
    # empty scores.json, never merged in-row). Deliberately samples the g_interior>=0.10
    # band the deployed OOD mask discards -> 74 bad / 6 okay / 0 good, all masked-band
    # material; a train-side hard-negative source, never an unbiased base-rate.
    "2026-07-27_interior_band_v1": "interior_band_v1.json",
}

# Label files that live in labels/ but belong to a DIFFERENT corpus and MUST NOT be read
# by this INTEGER reader. The q4 WINDOW store (tools/corpus/q4_window_reader.py) uses
# three-way STRING classes (accept/reject/filter_leak) keyed by window_id; int-coercing
# them is a category error (`int('reject')`), and pooling the two corpora would poison the
# version-blind v7 training distribution (see q4_window_reader.__doc__). This is the EXPLICIT,
# registration-driven boundary between the two stores at the filesystem level: the files stay
# in labels/ (so disk_audit's `^labels/` NEVER-delete protection still covers them in place —
# do NOT relocate to escape a crash), and this reader skips them by registration. Value = the
# owning corpus's canonical reader module, for the error message that redirects a mis-call.
FOREIGN_LABEL_FILES = {
    "q4_g_aimed.json": "tools/corpus/q4_window_reader.py",
    "q4_stage1_windows.json": "tools/corpus/q4_window_reader.py",
    "q4_stage1_windows_p2.json": "tools/corpus/q4_window_reader.py",
}


def is_v7_corpus_label_file(filename) -> bool:
    """True iff `filename` (a bare labels/*.json name) is in scope for THIS v7 location-corpus
    reader. A registered FOREIGN file (q4 window store, string classes) returns False and must
    be routed through its own reader, never int-coerced here. The explicit in-scope predicate
    every labels/ walk (the reachability guard included) filters on."""
    return filename not in FOREIGN_LABEL_FILES


# A sidecar file's labels are authored against ONE batch's crops — its OWNER. The join
# re-keys the image_id→score sidecar into coord_key→score using the OWNER batch's
# images.jsonl (see `sidecar_for`), so the label follows the RENDER IDENTITY, not the
# image_id slug. That is what stops cross-contamination: `scale_2x2_labelset.json` is shared
# by two batches built at DIFFERENT scales, and image_id (`A_<idx>_<comp>_<palette>`) does
# not encode the scale — so an `A_100_center_cmr.fusion` in one batch and the same id at a
# different fw in the other are different images that must NOT trade labels. Keying on the
# owner's coordinates makes a label reach only the crop whose (location, palette,
# composition) actually matches. Default owner = the sole batch that registers the file; a
# SHARED file MUST name its owner here (else `_owner_of` raises).
SIDECAR_OWNER = {
    "scale_2x2_labelset.json": "2026-06-25_scale_2x2_labelset",
}

# ---------------------------------------------------------------------------
# Amendment overlay — the revision stream.
#
# A human label is authored ONCE, in a batch's images.jsonl `label.score` (merged) or in a
# registered SIDECAR file. That original is NEVER modified. When a label is REVISED (a q3
# demoted to q2, or promoted to the new q4 tier), the new value goes to a SEPARATE amendment
# file registered here — batch_id -> labels/*.json — and `resolve_score` PREFERS the amendment
# over the original. This exists because:
#   * a revision can move the >=3 (good) boundary (demotions as well as promotions), and at
#     least one batch is the FROZEN eval census for the current model — rewriting labels in
#     place would silently destroy that comparison; and
#   * the pre-revision label must stay recoverable for any row. It is: the amendment lives in
#     a distinct file, the original is untouched, and calling `resolve_score(row, sidecar)`
#     with NO `amendments` argument reconstructs the original label. Reconstructing the
#     original >=3 boundary is the one-liner `resolve_score(row, sidecar) >= 3`.
# An amendment file is authored against ONE batch's crops (its OWNER = the amended batch),
# re-keyed onto the coordinate `join_key` through that batch's images.jsonl exactly like a
# sidecar (`amendments_for`), so revisions follow render identity, not the collision-prone
# image_id. Amendment files live in labels/, so disk_audit's `^labels/` NEVER-delete rule
# protects them in place. Registered here, they are NOT re-checked by `assert_sidecars_joined`
# (that guards the original sidecar stream); an amended batch already resolves non-null through
# its original label, so the reachability guard stays green.
AMENDMENT_LABELS: dict[str, str] = {
    # batch_id -> labels/<revision>.json  (populated by tools/corpus/merge_amendments.py)
    # 2026-07-26_anchor_class4_v1 revision pass: 52 previously-q3 rows re-judged blind on the
    # 1..4 scale; 14 demotions + 7 promotions + 31 reaffirmed, routed here per source batch.
    "2026-06-23_flat_generate_loose0_v3": "amend_2026-06-23_flat_generate_loose0_v3.json",
    "2026-06-24_guided_descend_rev4": "amend_2026-06-24_guided_descend_rev4.json",
    "2026-06-24_guided_descend_rev4occfix_v2filtered": "amend_2026-06-24_guided_descend_rev4occfix_v2filtered.json",
    "2026-06-25_mining_v3guided_v1": "amend_2026-06-25_mining_v3guided_v1.json",
    "2026-06-25_scale_2x2_labelset": "amend_2026-06-25_scale_2x2_labelset.json",
    "2026-07-05_gather_v6": "amend_2026-07-05_gather_v6.json",
    "2026-07-11_jm3_band_v1": "amend_2026-07-11_jm3_band_v1.json",
    "2026-07-12_blindspot_v6reject_v1": "amend_2026-07-12_blindspot_v6reject_v1.json",
    "2026-07-12_jm45_band_v1": "amend_2026-07-12_jm45_band_v1.json",
    "2026-07-17_prospect_run1_baserate_R_v1": "amend_2026-07-17_prospect_run1_baserate_R_v1.json",
    "2026-07-17_prospect_run1_baserate_v1": "amend_2026-07-17_prospect_run1_baserate_v1.json",
    "2026-07-21_phoenix_grid": "amend_2026-07-21_phoenix_grid.json",
    "2026-07-22_native_multibrot_band_v1": "amend_2026-07-22_native_multibrot_band_v1.json",
    "julia_ladder_j0": "amend_julia_ladder_j0.json",
}

# Parallel to SIDECAR_OWNER: a shared amendment file MUST name the batch its coordinate
# identities are authored against. Default owner = the amended batch itself (the sole
# registrant), so this is usually empty.
AMENDMENT_OWNER: dict[str, str] = {}


def join_key(render):
    """The explicit coordinate join key for a corpus render block: the canonical location
    identity (`location.Location.key()` — family + cx/cy/fw + c, so SCALE via fw is part of
    the key) plus the per-crop recolor/reframe axes (palette, composition). Verified unique
    per image_id in every registered sidecar batch, so it distinguishes same-location recolor
    crops (mining) while collapsing nothing, and it differs across scales where image_id
    collides."""
    loc = _loc.from_render_block(render)
    return (loc.key(), render.get("palette"), render.get("composition"))


def _owner_of(filename):
    """The batch whose images.jsonl authors the coordinate identities for `filename`.
    Explicit `SIDECAR_OWNER` wins; otherwise the sole batch registering the file. A file
    shared by >1 batch with no explicit owner is a registry error and raises."""
    if filename in SIDECAR_OWNER:
        return SIDECAR_OWNER[filename]
    owners = sorted(b for b, fn in SIDECAR_LABELS.items() if fn == filename)
    if len(owners) == 1:
        return owners[0]
    raise RuntimeError(
        f"sidecar {filename!r} is registered by {owners} — add an explicit SIDECAR_OWNER "
        f"entry naming the batch its labels were authored against.")


def _owner_keymap(owner_batch_id, batches_dir):
    """image_id → join_key for every row of the owner batch's images.jsonl. This is the
    authoritative image_id→coordinate mapping the sidecar's labels are re-keyed through."""
    path = os.path.join(batches_dir, owner_batch_id, "images.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"owner batch images.jsonl missing for sidecar re-key: {path}")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["image_id"]] = join_key(row["render"])
    return out


def load_sidecar(filename):
    """Load a labels/*.json sidecar as {image_id: int score}, dropping nulls.

    Tolerates the three on-disk shapes a label export takes (same value forms as
    `merge_scores.load_scores`, so the two loaders can never disagree on a format):
      * bare `{image_id: int}` — the legacy flat sidecar (jm3/jm45/native_multibrot/phoenix);
      * `{"labels": {...}}` wrapper around either value form;
      * combined reveal-audit export `{image_id: {"score": int, "revealed": 0|1}}` — the
        blind-labeling UI's current output (`interior_band_v1`). Only the score is a store
        value; the reveal flag is an audit field, dropped here.
    Raises on a registered FOREIGN file (a different corpus, string classes) rather than
    int-crashing on it — the loud, actionable form of the two-corpus boundary."""
    if filename in FOREIGN_LABEL_FILES:
        raise ValueError(
            f"{filename!r} belongs to a different corpus (read it through "
            f"{FOREIGN_LABEL_FILES[filename]}), not the v7 integer sidecar loader — its labels "
            f"are string classes, not int scores. See label_store.FOREIGN_LABEL_FILES.")
    d = json.loads((open(os.path.join(LABELS_DIR, filename), encoding="utf-8")).read())
    body = d["labels"] if isinstance(d.get("labels"), dict) else d
    out = {}
    for k, v in body.items():
        if isinstance(v, dict):          # combined reveal-audit form: pull the score out
            v = v.get("score")
        if v is not None:
            out[k] = int(v)
    return out


def _rekey_onto_join(raw, owner_batch_id, batches_dir, fn):
    """Re-key an on-disk `{image_id: score}` file onto the coordinate `join_key` via the
    OWNER batch's images.jsonl. Shared by `sidecar_for` and `amendments_for` so the two
    streams re-key identically. Raises if a labeled image_id is absent from the owner batch,
    or if two entries collide on one join_key with different scores."""
    id2key = _owner_keymap(owner_batch_id, batches_dir or BATCHES_DIR)
    labels = {}
    for iid, sc in raw.items():
        key = id2key.get(iid)
        if key is None:
            raise RuntimeError(
                f"label file {fn!r} labels image_id {iid!r} that is absent from its owner "
                f"batch {owner_batch_id!r} images.jsonl — image_id keys diverged.")
        if key in labels and labels[key] != sc:
            raise RuntimeError(
                f"label file {fn!r}: two image_ids collide on one coordinate join_key {key!r} "
                f"with different scores — the owner batch is not render-unique.")
        labels[key] = sc
    return labels


def sidecar_for(batch_id, batches_dir=None):
    """The `{join_key: score}` label map for a batch, or None if it isn't registered.

    The on-disk sidecar is `{image_id: score}`; this re-keys it onto the coordinate
    `join_key` via the OWNER batch's images.jsonl (`_owner_keymap`), so resolution follows
    the render identity rather than the collision-prone image_id. `batches_dir` overrides the
    standard corpus (a non-default corpus threads its own batches dir through). Raises if a
    labeled image_id is absent from the owner batch, or if two entries collide on one
    join_key — both are registry/key errors, not silent drops."""
    fn = SIDECAR_LABELS.get(batch_id)
    if fn is None:
        return None
    return _rekey_onto_join(load_sidecar(fn), _owner_of(fn), batches_dir, fn)


def _amendment_owner_of(batch_id, fn):
    """Owner batch whose images.jsonl authors an amendment file's coordinate identities.
    Explicit `AMENDMENT_OWNER` wins; otherwise the amended batch itself (the default —
    an amendment is authored against the very batch it revises)."""
    return AMENDMENT_OWNER.get(fn, batch_id)


def amendments_for(batch_id, batches_dir=None):
    """The `{join_key: revised_score}` REVISION map for a batch, or None if unregistered.

    Same coordinate re-key as `sidecar_for`, but the owner defaults to the amended batch
    itself. `resolve_score` prefers this over both the in-row label and the sidecar, so a
    revision overrides the original WITHOUT touching the original file. Called only where the
    REVISED truth is wanted (the canonical trainer view); pass nothing to read originals."""
    fn = AMENDMENT_LABELS.get(batch_id)
    if fn is None:
        return None
    return _rekey_onto_join(load_sidecar(fn), _amendment_owner_of(batch_id, fn), batches_dir, fn)


def resolve_score(row, labels, amendments=None):
    """A row's label: the REVISION (amendment) if one exists, ELSE merged `label.score`,
    ELSE the sidecar join by coordinate `join_key`.

    `labels` is the map from `sidecar_for(batch_id)` (or None for a merged batch);
    `amendments` is the map from `amendments_for(batch_id)` (or None — the default). Called
    WITHOUT `amendments` this yields the ORIGINAL, pre-revision label, so reconstructing the
    original >=3 boundary is the one-liner `resolve_score(row, sidecar) >= 3`. Returns None if
    the row is unlabeled everywhere. This is the ONE resolution rule; every consumer calls it
    so they cannot disagree on a row. Joining on `join_key(row["render"])` (not image_id) is
    what makes a shared sidecar/amendment safe across scale batches."""
    if amendments is not None:
        amd = amendments.get(join_key(row["render"]))
        if amd is not None:
            return amd
    sc = (row.get("label") or {}).get("score")
    if sc is None and labels is not None:
        sc = labels.get(join_key(row["render"]))
    return sc


def assert_sidecars_joined(joined):
    """Raise if a REGISTERED sidecar batch present in `joined` contributed 0 rows.

    `joined`: {batch_id: rows_resolved_via_this_batch} accumulated over a full pass. A
    registered sidecar that resolves nothing is a corpus/registry error (image_id keys
    diverged / wrong file) — not an empty batch. Batches absent from `joined` (not on
    disk) are skipped."""
    for bid, fn in SIDECAR_LABELS.items():
        if bid in joined and joined[bid] == 0:
            raise RuntimeError(
                f"batch {bid!r} has a registered label sidecar ({fn}) but joined 0 "
                f"rows — image_id keys likely diverged. Fix SIDECAR_LABELS in "
                f"tools/corpus/label_store.py.")
