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
from production_pins import ACTIVE_VERSION as _ACTIVE
from tools.emission import floors as F
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


def test_the_intake_ledger_population_is_on_disk_and_its_siblings_are_distinct():
    # TEN since 2026-08-10 (prompts/sittings_27.md step 0): production run 25's three legs —
    # breadth, dive, native phoenix — joined the seven. The count is pinned so a ledger
    # appearing or vanishing is an explicit edit; `test_intake_union.UNION_ADMITTED` carries
    # what the population then admits.
    assert len(LR.LEDGERS) == 10
    missing = [rel for _t, rel in LR.LEDGERS if not LR.ledger_path(rel).exists()]
    assert not missing, f"intake ledgers missing: {missing}"
    sibs = [D.rescore_path(LR.ledger_path(rel)) for _t, rel in LR.LEDGERS]
    assert len(set(sibs)) == len(sibs)


# --------------------------------------------------------------------------- #
# reader resolution
# --------------------------------------------------------------------------- #
def test_the_overlay_is_what_moves_a_rows_score_not_its_admissibility(tmp_path):
    """What a re-score buys, after 2026-08-09. It used to buy ADMISSIBILITY: a stale ledger
    admitted NOTHING (that is the state the v10 flip left every non-classic ledger in) and the
    sibling revived it. The decode-version predicate is gone, so a stale row is admitted on
    its own older-scale `p_good` — and what the overlay changes is the NUMBER, which can move
    a row across the floor in either direction.

    Both directions asserted: `a` starts below the floor and the re-score lifts it over, `b`
    starts above and the re-score drops it under. A re-score that could only add rows would
    look like an improvement whatever it did."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7", p_good=0.2), _row("b", "v7", p_good=0.8)])
    assert [r["id"] for r in D.load_admitted(led)] == ["b"]       # stale, but judged on merit

    _write(D.rescore_path(led), [_row("a", _ACTIVE, p_good=0.8),
                                 _row("b", _ACTIVE, p_good=0.2)])
    assert [r["id"] for r in D.load_admitted(led)] == ["a"]


def test_the_original_ledger_is_byte_identical_after_a_resolve(tmp_path):
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7")])
    _write(D.rescore_path(led), [_row("a", cc.active_scorer_version(), p_good=0.99)])
    before = hashlib.sha256(led.read_bytes()).hexdigest()
    rows = D.resolve_rows(led)
    assert rows[0]["p_good"] == 0.99 and rows[0]["scorer_version"] == cc.active_scorer_version()
    assert hashlib.sha256(led.read_bytes()).hexdigest() == before


def test_a_sibling_for_another_version_is_not_consulted(tmp_path):
    """The fail-correct property. A v99 record must not answer the live reader's question —
    it falls through to the ORIGINAL rows, which is a worse estimate rather than an empty
    intake (that distinction is the 2026-08-09 change; before it, falling through meant every
    row was refused)."""
    led = tmp_path / "outcome_ledger.jsonl"
    _write(led, [_row("a", "v7", p_good=0.8)])
    other = led.with_name("outcome_ledger.rescored_v99.jsonl")
    _write(other, [_row("a", "v99", p_good=0.1)])
    assert D.load_rescored(led) == {}
    assert [r["id"] for r in D.load_admitted(led)] == ["a"]   # the v7 score, not the v99 one
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


# --------------------------------------------------------------------------- #
# The externally-supplied supply check at intake (`classic_supply_note`).
#
# `phoenix:classic` is the one partition no crawl produces, so its supply is invisible until
# something looks. This is that something — and the low-water it compares against is derived
# from the committed release mix, not declared, so these pin the derivation and not a literal.
# --------------------------------------------------------------------------- #
def _classic_row(rid, p_notbad, p_good, **over):
    """A classic-phoenix ledger row: `family: phoenix` plus the pinned Ushiki axes, which is
    what makes `partitions.partition_of_row` resolve it to `phoenix:classic`."""
    row = {"id": rid, "family": "phoenix", "outcome_cx": "0.1", "outcome_cy": "0.2",
           "outcome_fw": "0.01", "decoded_class": 3, "p_notbad": p_notbad, "p_good": p_good,
           "t_good": 0.5, "guard_pass": True, "distinct": True, "dup_of": None,
           "scorer_version": cc.active_scorer_version(), "mix_source": "classic_phoenix",
           "phoenix_c_re": 0.5667, "phoenix_c_im": 0.0, "phoenix_p_re": -0.5,
           "phoenix_p_im": 0.0, "phoenix_zm1_re": 0.0, "phoenix_zm1_im": 0.0}
    row.update(over)
    return row


def _classic_ledger(tmp_path, rows):
    p = tmp_path / "outcome_ledger.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_the_servable_classic_count_is_read_at_the_live_floor(tmp_path):
    """SERVABLE is a live read, never a stored one. A ledger row used to carry the `t_good` it
    was MINTED under, so counting its stamped `decoded_class` reported against a threshold that
    might no longer be served; there is one flat `floors.GOOD_FLOOR` now and it is applied here
    at read time for the same reason — a count against a retired cut is worse than no count.

    `n_admitted` is guard ∧ distinct — what the ledger holds — and `n_servable` is the subset
    over the floor, so the pair says "how much supply, and how much of it counts"."""
    t = F.GOOD_FLOOR
    led = _classic_ledger(tmp_path, [
        _classic_row("a", 0.99, min(0.99, t + 0.2)),                  # clears the live cut
        _classic_row("b", 0.99, min(0.99, t + 0.2)),
        # a row a stamped class-3 count would have over-reported: below the live floor.
        _classic_row("c", 0.99, max(0.0, t - 0.2), decoded_class=3),
    ])
    n = LR.classic_supply_note(ledger=led, n_union=100)
    assert n["n_admitted"] == 3 and n["n_servable"] == 2
    assert n["good_floor"] == t and n["partition"] == "phoenix:classic"


def test_a_varied_phoenix_row_is_not_counted_as_classic_supply(tmp_path):
    """The split is on the PARAMETER POINT. Counting a swept-grid row here would report
    supply for a plane it does not belong to — and the grid is the abundant one."""
    varied = _classic_row("v", 0.99, 0.99, phoenix_c_re=-1.089, phoenix_c_im=0.481)
    led = _classic_ledger(tmp_path, [_classic_row("a", 0.99, 0.99), varied])
    n = LR.classic_supply_note(ledger=led, n_union=100)
    assert n["n_admitted"] == 1 and n["n_servable"] == 1


def test_the_low_water_is_derived_from_the_release_mix_not_declared(tmp_path):
    """`wanted` must move with the intake size and with the ratio table. A literal would go
    stale the first time either moved, and would read as a decision nobody made."""
    import math
    import release_mix as rm
    share = rm.shares()["phoenix:classic"]
    led = _classic_ledger(tmp_path, [_classic_row(f"r{i}", 0.99, 0.99) for i in range(5)])
    small = LR.classic_supply_note(ledger=led, n_union=100)
    big = LR.classic_supply_note(ledger=led, n_union=10_000)
    assert small["wanted"] == math.ceil(share * 100)
    assert big["wanted"] == math.ceil(share * 10_000)
    assert big["wanted"] > small["wanted"]
    assert small["low"] is False and big["low"] is True   # same supply, different demand


def test_the_low_note_names_the_manual_job_from_the_routing_table(tmp_path):
    """No automated top-up: the note has to say what to RUN, and it reads that from
    `supply_routing` rather than restating a command."""
    import supply_routing as srt
    led = _classic_ledger(tmp_path, [_classic_row("a", 0.99, 0.99)])
    n = LR.classic_supply_note(ledger=led, n_union=10_000)
    assert n["low"] and n["externally_supplied"]
    assert n["command"] == srt.supply_command("phoenix:classic")
    assert "run-phoenix" in n["command"]


def test_an_empty_classic_ledger_reads_as_zero_supply_not_as_healthy(tmp_path):
    """The absence case. A partition with no rows at all is the LOUDEST version of the
    problem, and must not fall out of the check by producing no row to look at."""
    led = _classic_ledger(tmp_path, [])
    n = LR.classic_supply_note(ledger=led, n_union=751)
    assert n["n_admitted"] == 0 and n["n_servable"] == 0 and n["low"] is True


def test_the_note_prints_without_raising_and_says_which_way_it_went(tmp_path, capsys):
    led = _classic_ledger(tmp_path, [_classic_row("a", 0.99, 0.99)])
    LR.print_classic_supply_note(LR.classic_supply_note(ledger=led, n_union=10_000))
    out = capsys.readouterr().out
    assert "phoenix:classic servable: 1" in out and "LOW" in out
    assert "Top up manually" in out
    LR.print_classic_supply_note(LR.classic_supply_note(ledger=led, n_union=10))
    assert "Top up manually" not in capsys.readouterr().out
