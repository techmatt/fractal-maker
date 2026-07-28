"""Tests for the version-blind reader's shared label resolution.

Guards against the latent sidecar-drop: `corpus_reader.iter_labeled` used to read
`label.score` alone, so the sidecar-only batches (Julia/mining/scale) silently
resolved to no label. It now routes through `label_store` — the SAME primitive the
query sampler uses — so (a) it recovers those labels and (b) the two consumers can
never disagree on a row.

The expectation in `_independent_join` is EXTERNAL to that primitive on purpose: raw
`labels/*.json` read straight off disk, joined by image_id, with the revision overlay
reconstructed in two lines here rather than delegated to `resolve_score`. Since the reader
applies amendments, the ground truth has to as well — and applying them by CALLING
`resolve_score` would turn the check into a restatement. `test_independent_ground_truth_is_
not_tautological` proves by mutation that it has not.

Run either way:
  uv run pytest tools/corpus/test_corpus_reader.py
  uv run python tools/corpus/test_corpus_reader.py     # prints PASS/FAIL summary
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # tools/corpus
sys.path.insert(0, os.path.join(HERE, "..", "queries"))   # tools/queries

import corpus_reader as cr  # noqa: E402
import label_store as ls  # noqa: E402


# The (b)-case batches whose labels live ONLY in a labels/*.json sidecar — the ones a
# `label.score`-only reader dropped. julia_ladder_j0 is the whole Julia family; the
# other three are Mandelbrot mining/scale.
SIDECAR_BATCHES = list(ls.SIDECAR_LABELS)


def _reader_labels_by_batch():
    """{batch_id: {image_id: score}} as corpus_reader.iter_labeled resolves them."""
    out = {}
    for lc in cr.iter_labeled():
        out.setdefault(lc.batch_id, {})[lc.image_id] = lc.score
    return out


def _raw_json_map(filename):
    """Load a labels/*.json as {image_id: int}, nulls dropped, WITHOUT going through
    label_store — the external side of the check. Reimplements the same value-shape
    tolerance independently (flat `int`, or the combined `{"score", "revealed"}` reveal-audit
    export) so the ground truth stays a genuine second implementation, not a call to the code
    under test."""
    raw = json.loads(open(os.path.join(ls.LABELS_DIR, filename), encoding="utf-8").read())
    body = raw["labels"] if isinstance(raw.get("labels"), dict) else raw
    out = {}
    for k, v in body.items():
        if isinstance(v, dict):
            v = v.get("score")
        if v is not None:
            out[k] = int(v)
    return out


def _independent_join(batch_id):
    """Reconstruct a sidecar batch's labels WITHOUT label_store: raw sidecar file
    JOINED to the batch's images.jsonl image_ids, then the REVISION amendment overlaid
    (if the batch has one registered). This is the external ground truth that the shared
    resolver must reproduce — and iter_labeled applies amendments (revised truth wins),
    so the ground truth must too, else a demoted/promoted sidecar row (e.g. the
    julia_ladder_j0 anchor revisions) reads as a spurious mismatch. The overlay is by
    image_id: for these sidecar batches the amendment's owner is the batch itself, so its
    keys are that batch's own image_ids (merge_amendments keyed them by revises_image_id)."""
    sidecar = _raw_json_map(ls.SIDECAR_LABELS[batch_id])
    jl = os.path.join(cr.cc.BATCHES_DIR, batch_id, "images.jsonl")
    ids = set()
    with open(jl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["image_id"])
    joined = {iid: sc for iid, sc in sidecar.items() if iid in ids}
    amend_file = ls.AMENDMENT_LABELS.get(batch_id)
    if amend_file:
        for iid, sc in _raw_json_map(amend_file).items():
            if iid in ids:
                joined[iid] = sc          # revision wins, exactly as resolve_score prefers it
    return joined


def test_recovers_sidecar_only_labels():
    """iter_labeled now sees the previously-dropped sidecar batches (non-zero each),
    and its recovered labels match an independent sidecar x images.jsonl join."""
    by_batch = _reader_labels_by_batch()
    for bid in SIDECAR_BATCHES:
        got = by_batch.get(bid, {})
        assert got, f"{bid}: iter_labeled recovered 0 labels (sidecar drop regressed)"
        assert got == _independent_join(bid), (
            f"{bid}: reader labels diverge from the raw sidecar join")


def test_the_amendment_overlay_in_the_ground_truth_is_load_bearing():
    """Non-vacuity: at least one sidecar batch carries an amendment that CHANGES a score.

    If every amendment merely reaffirmed its original, the overlay in `_independent_join`
    would be a no-op and the test below would pass for the wrong reason."""
    changed = 0
    for bid, sidecar_file in ls.SIDECAR_LABELS.items():
        amend_file = ls.AMENDMENT_LABELS.get(bid)
        if not amend_file:
            continue
        orig = _raw_json_map(sidecar_file)
        for iid, sc in _raw_json_map(amend_file).items():
            if iid in orig and orig[iid] != sc:
                changed += 1
    assert changed > 0, (
        "no sidecar-batch amendment changes a score — the amendment overlay in the "
        "independent ground truth is currently vacuous, so the test below proves nothing")


def test_independent_ground_truth_is_not_tautological():
    """The "independent" expectation must RECONSTRUCT the amendment overlay, not delegate
    to the resolver it is checking.

    `_independent_join` reads the raw `labels/*.json` files and overlays them by image_id
    with its own two lines of dict logic; `resolve_score` reads `label.score` and joins on
    the coordinate `join_key`. Different code, different join key, same expected answer —
    that is what makes it a check rather than a restatement.

    Proven by mutation, because "looks independent" is not a guarantee: break
    `resolve_score`'s amendment preference and the sidecar test must go RED. A ground truth
    that called `resolve_score` would stay green under the same break, which is exactly the
    tautology this pins shut."""
    orig = ls.resolve_score
    ls.resolve_score = lambda row, labels, amendments=None: orig(row, labels, None)
    try:
        caught = False
        try:
            test_recovers_sidecar_only_labels()
        except AssertionError:
            caught = True
    finally:
        ls.resolve_score = orig
    assert caught, (
        "breaking resolve_score's amendment preference did NOT fail the sidecar test — "
        "the 'independent' ground truth is delegating to the resolver it is supposed to "
        "check, so that test can no longer detect a resolution bug")
    test_recovers_sidecar_only_labels()          # and green again once restored


def test_both_consumers_share_the_resolver():
    """corpus_reader and query_sampler resolve through the SAME label_store object —
    the structural guarantee that they cannot drift. Also assert they agree row-for-row
    on the sidecar batches (the concrete cross-consumer check).

    This one IS resolver-based on both sides, deliberately: it checks that the two
    CONSUMERS agree, which is a wiring question, not a ground-truth question. The
    independent ground truth lives in `_independent_join` above, and
    `test_independent_ground_truth_is_not_tautological` pins it there."""
    sys.path.insert(0, os.path.join(HERE, "..", "palettes"))
    import query_sampler as qs  # noqa: E402  (heavy import: colormap + numpy)

    # Same primitive, one registry.
    assert cr.ls is qs.ls, "corpus_reader and query_sampler bound different label_store"
    assert qs.SIDECAR_LABELS is ls.SIDECAR_LABELS

    reader = _reader_labels_by_batch()
    # Reproduce the sampler's per-row resolution (its from_corpus loop delegates to
    # iter_labeled, which resolves through the SAME ls.resolve_score WITH amendments) and
    # confirm it agrees with the reader on the q2/q3 rows. Passing `amendments` here is
    # load-bearing: a revised sidecar row (e.g. a julia_ladder_j0 3->2 demotion) resolves
    # to the revised value in BOTH consumers, and dropping the arg would falsely flag it.
    for bid in SIDECAR_BATCHES:
        sidecar = ls.sidecar_for(bid)
        amendments = ls.amendments_for(bid)
        jl = os.path.join(cr.cc.BATCHES_DIR, bid, "images.jsonl")
        with open(jl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sc = ls.resolve_score(row, sidecar, amendments)
                if sc in (2, 3):   # q2/q3: the sampler's keep-set
                    assert reader.get(bid, {}).get(row["image_id"]) == sc, (
                        f"{bid}/{row['image_id']}: reader != sampler resolution")


def main():
    tests = [
        test_recovers_sidecar_only_labels,
        test_the_amendment_overlay_in_the_ground_truth_is_load_bearing,
        test_independent_ground_truth_is_not_tautological,
        test_both_consumers_share_the_resolver,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (t.__name__, e))
    print("\n%d/%d tests passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
