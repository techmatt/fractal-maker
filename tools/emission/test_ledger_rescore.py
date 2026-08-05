#!/usr/bin/env python
"""The re-score sibling record and its reader resolution.

A ledger's decode block is one head's verdict, so a pin flip strands every row in it. The
re-score writes a SIBLING (`<stem>.rescored_<version>.jsonl`) and the reader overlays it;
the original is only ever read. What these pin:

  * the sibling name carries the ACTIVE version and is derived, never a literal — so the
    NEXT flip looks for a file that does not exist, falls through to the original rows, and
    the current-decode predicate rejects them. Failing to an empty intake is the correct
    failure; silently serving v10 verdicts under v11 is not.
  * the overlay is what makes a stale ledger admissible again, and the original file is
    byte-identical afterwards.
  * the seven-ledger population is on disk and its siblings do not collide with each other
    or with the two pre-existing `rescored.jsonl` RESUME files, which are a different
    artifact belonging to a different producer.

  uv run pytest tools/emission/test_ledger_rescore.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import descriptor as D          # noqa: E402
from tools.emission import ledger_rescore as LR     # noqa: E402
import corpus_common as cc                          # noqa: E402


def _row(rid, version, **over):
    row = {"id": rid, "family": "mandelbrot",
           "outcome_cx": "0.0", "outcome_cy": "0.0", "outcome_fw": "1.0",
           "scorer_version": version, "decoded_class": 3, "p_notbad": 0.9, "p_good": 0.8,
           "guard_pass": True, "distinct": True}
    row.update(over)
    return row


def _write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
def test_the_sibling_name_is_derived_from_the_active_head_not_a_literal():
    v = cc.active_scorer_version()
    p = D.rescore_path("data/discovery/x/outcome_ledger.jsonl")
    assert p.name == f"outcome_ledger.rescored_{v}.jsonl"
    assert p.parent == Path("data/discovery/x")


def test_the_sibling_is_never_the_producers_own_rescored_resume_file():
    """`classic_phoenix` and `q4_harvest` each already keep a `rescored.jsonl` — their own
    per-candidate resume state, not an overlay. A collision would make the reader consume a
    file with different semantics."""
    for _tag, rel in LR.LEDGERS:
        sib = D.rescore_path(LR.ledger_path(rel))
        assert sib.name != "rescored.jsonl"
        assert sib != LR.ledger_path(rel)


def test_the_seven_ledger_population_is_on_disk_and_its_siblings_are_distinct():
    assert len(LR.LEDGERS) == 7
    missing = [rel for _t, rel in LR.LEDGERS if not LR.ledger_path(rel).exists()]
    assert not missing, f"intake ledgers missing: {missing}"
    sibs = [D.rescore_path(LR.ledger_path(rel)) for _t, rel in LR.LEDGERS]
    assert len(set(sibs)) == len(sibs)


# --------------------------------------------------------------------------- #
# reader resolution
# --------------------------------------------------------------------------- #
def test_a_stale_ledger_admits_nothing_and_the_overlay_is_what_revives_it(tmp_path):
    """Both sides. The stale ledger is the state the v10 flip left every non-classic ledger
    in; the sibling is the fix; and the ids must be the ledger's own."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7"), _row("b", "v7", decoded_class=1)])
    assert D.load_admitted(led) == []

    _write(D.rescore_path(led), [_row("a", cc.active_scorer_version()),
                                 _row("b", cc.active_scorer_version(), decoded_class=2)])
    assert [r["id"] for r in D.load_admitted(led)] == ["a"]      # b decodes below q3


def test_the_original_ledger_is_byte_identical_after_a_resolve(tmp_path):
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7")])
    _write(D.rescore_path(led), [_row("a", cc.active_scorer_version(), p_good=0.99)])
    before = hashlib.sha256(led.read_bytes()).hexdigest()
    rows = D.resolve_rows(led)
    assert rows[0]["p_good"] == 0.99 and rows[0]["scorer_version"] == cc.active_scorer_version()
    assert hashlib.sha256(led.read_bytes()).hexdigest() == before


def test_a_sibling_for_another_version_is_not_consulted(tmp_path):
    """The fail-correct property. A v10 record must not answer a v11 reader's question."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7")])
    other = led.with_name("outcome_ledger.rescored_v99.jsonl")
    _write(other, [_row("a", "v99")])
    assert D.load_rescored(led) == {}
    assert D.load_admitted(led) == []
    # ...and it IS readable when that version is the one being asked about (not vacuous).
    assert set(D.load_rescored(led, version="v99")) == {"a"}


def test_the_ledger_defines_the_population_not_the_sibling(tmp_path):
    """A re-score row whose id the ledger does not hold is ignored — a sibling cannot add
    locations to an intake."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7")])
    _write(D.rescore_path(led), [_row("a", cc.active_scorer_version()),
                                 _row("ghost", cc.active_scorer_version())])
    assert [r["id"] for r in D.resolve_rows(led)] == ["a"]


def test_an_absent_sibling_leaves_the_rows_exactly_as_they_are(tmp_path):
    led = tmp_path / "outcome_ledger.jsonl"
    rows = [_row("a", "v7"), _row("b", "v7")]
    _write(led, rows)
    assert D.resolve_rows(led) == rows


# --------------------------------------------------------------------------- #
# the writer
# --------------------------------------------------------------------------- #
def test_the_writer_never_touches_the_original_and_resumes_by_id(tmp_path, monkeypatch):
    """`_score_row` is stubbed (it renders through a subprocess); everything around it —
    the sibling path, the merge, the resume set, the untouched original — is the subject."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row(f"r{i}", "v7") for i in range(4)])
    before = hashlib.sha256(led.read_bytes()).hexdigest()
    rel = led.relative_to(led.anchor).as_posix()      # tag/label only; path comes from LEDGERS
    monkeypatch.setattr(LR, "ledger_path", lambda _r: led)
    monkeypatch.setattr(LR, "_score_row", lambda _s, row, _t: {
        "decoded_class": 3, "p_notbad": 0.9, "p_good": 0.9, "p_ge4": 0.1,
        "scorer_version": cc.active_scorer_version()})

    got = LR.rescore_ledger("t", rel, scorer=None, limit=2)
    assert got["n_rescored"] == 2 and got["already_current"] is False
    assert len(D.load_rescored(led)) == 2

    got2 = LR.rescore_ledger("t", rel, scorer=None)       # resume: only the remaining 2
    assert got2["n_rescored"] == 2
    assert len(D.load_rescored(led)) == 4
    assert hashlib.sha256(led.read_bytes()).hexdigest() == before


def test_an_already_current_ledger_is_verified_not_re_rendered(tmp_path, monkeypatch):
    """`classic_phoenix` is already v10. Re-scoring it would burn the budget reproducing its
    own numbers, so the pass must skip it — and must say so, not silently do nothing."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", cc.active_scorer_version())])
    monkeypatch.setattr(LR, "ledger_path", lambda _r: led)
    monkeypatch.setattr(LR, "_score_row",
                        lambda *a, **k: pytest.fail("re-rendered an already-current ledger"))
    got = LR.rescore_ledger("t", "rel", scorer=None)
    assert got["already_current"] is True and got["n_rescored"] == 0
    assert not D.rescore_path(led).exists()
