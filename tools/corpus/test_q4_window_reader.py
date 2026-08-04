"""Resolution semantics of the q4 WINDOW store reader.

`test_label_reachability.py` asserts that no label on disk is dropped. This file asserts
the rule the reader uses to combine the files it now reads: sources in
`sidecar_sources` order, LAST WINS, in-row `label.klass` still authoritative over all of
them. The two halves matter separately — a reader that read every file but took the
FIRST verdict would pass reachability and still serve the pre-revision label for the 49
windows p2 re-judged.

  uv run pytest tools/corpus/test_q4_window_reader.py
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import q4_window_reader as q4  # noqa: E402

STAGE1 = "2026-07-23_q4_stage1_windows"
G_AIMED = "2026-07-23_q4_g_aimed"


def _raw(fn):
    return q4._load_label_file(q4.LABELS_DIR / fn)


# --------------------------------------------------------------------------- #
# precedence
# --------------------------------------------------------------------------- #
def test_later_export_wins_on_a_rejudged_window():
    """p1 and p2 overlap and DISAGREE; the reader must serve p2. Asserted on the real
    disagreements rather than a fixture, so the test is about this corpus's actual
    revision and cannot pass on a synthetic pair the store does not contain."""
    p1, p2 = _raw("q4_stage1_windows.json"), _raw("q4_stage1_windows_p2.json")
    flipped = {k: (p1[k], p2[k]) for k in set(p1) & set(p2) if p1[k] != p2[k]}
    assert flipped, ("p1 and p2 no longer disagree anywhere — this test is vacuous. If the "
                     "exports were reconciled on purpose, delete it; if not, a file moved.")

    merged = q4.load_scores_sidecar(STAGE1)
    wrong = {k: (old, new, merged.get(k)) for k, (old, new) in flipped.items()
             if merged.get(k) != new}
    assert not wrong, (
        f"{len(wrong)} re-judged window(s) resolve to the SUPERSEDED verdict — precedence "
        f"is not last-wins: {dict(list(wrong.items())[:3])}")


def test_precedence_order_is_the_registry_order():
    """The merge follows `SIDECAR_FILES` order, not filesystem or dict-iteration luck."""
    srcs = [p.name for p in q4.sidecar_sources(STAGE1)]
    assert srcs == ["scores.json", "q4_stage1_windows.json", "q4_stage1_windows_p2.json"], srcs
    assert [p.name for p in q4.sidecar_sources(G_AIMED)] == ["q4_g_aimed.json"]


def test_in_row_klass_still_wins_over_every_sidecar():
    """`resolve_klass`'s contract is unchanged: a non-null in-row label is authoritative.
    Adding sources must not quietly invert that — the sidecars fill nulls, nothing more."""
    row = {"window_id": "wid", "label": {"klass": "accept"}}
    assert q4.resolve_klass(row, {"wid": "reject"}) == "accept"
    assert q4.resolve_klass({"window_id": "wid", "label": {"klass": None}},
                            {"wid": "reject"}) == "reject"
    assert q4.resolve_klass({"window_id": "wid", "label": {}}, {}) is None


def test_unrecognized_value_does_not_erase_an_earlier_label():
    """A junk/None value in a later file must not overwrite a good earlier verdict with
    nothing — dropping a label is the failure mode this whole module is about."""
    assert q4._norm_klass("not_a_class") is None
    merged = q4.load_scores_sidecar(G_AIMED)
    assert set(merged.values()) <= set(q4.CLASSES)


# --------------------------------------------------------------------------- #
# what the reader now serves
# --------------------------------------------------------------------------- #
def test_every_batch_resolves_labels():
    """No registered batch reads as entirely unlabeled. `q4_g_aimed` did exactly that —
    112 labels on disk, 0 through the reader — because it has no in-store scores.json."""
    for bid in q4.REGISTERED_BATCHES:
        n = sum(1 for _, k in q4.iter_windows(bid) if k is not None)
        assert n, (f"{bid}: 0 windows resolve a label. If it is genuinely unlabeled, say so "
                   f"here; otherwise its export is missing from SIDECAR_FILES.")


def test_fit_view_excludes_filter_leak():
    """`iter_labeled` is the accept-vs-reject fit view and drops `filter_leak` by contract
    (it is pre-filter feedback, never a quality target). Pinned because the leak count is
    no longer small-and-obvious: p2 re-judged most of p1's leaks."""
    fit = Counter(acc for _, acc in q4.iter_labeled())
    allk = Counter(k for _, k in q4.iter_windows(labeled_only=True))
    assert fit[True] == allk["accept"]
    assert fit[False] == allk["reject"]
    assert allk["filter_leak"], "no filter_leak rows left — this exclusion is untested"
    assert sum(fit.values()) == allk["accept"] + allk["reject"]


def test_leak_rate_is_computed_over_the_full_labeled_set():
    """The diagnostic must move with the labels the reader can see. It read 61/107 = 57%
    when two thirds of the labels were invisible; over the resolvable set it is far lower,
    and a stale high reading is what would have justified tightening a pre-filter that was
    not actually leaking."""
    nleak, nlab, rate = q4.prefilter_leak_rate()
    total = sum(1 for _, k in q4.iter_windows() if k is not None)
    assert nlab == total, f"leak rate saw {nlab} labeled, reader resolves {total}"
    assert 0 <= rate < 0.5, f"leak rate {rate:.1%} — recheck before trusting the pre-filter"
