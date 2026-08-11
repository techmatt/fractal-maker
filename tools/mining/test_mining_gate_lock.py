#!/usr/bin/env python
"""`tools/mining/lock_mining_gate.py` — the gate lock's four load-bearing properties.

  1. THE RECORD IS THE CODE'S OUTPUT. The committed lock (and its .md face) is exactly what
     the committed sitting + the live pin/floors derive — so a hand-edit, a floor move or a
     pin move shows up here rather than as a record quietly describing a gate nobody runs.
  2. A READER REFUSES ON A PIN MOVE. `read_lock` raises when the live mining head is not the
     one the numbers were measured on, mirroring `floors.Floor.gate`.
  3. THE DERIVATION REFUSES rather than interpolating: a cut the sitting never swept at that
     exact value, or a sitting that calibrated a different checkpoint, is an error.
  4. A DEFAULT RUN DOES NOT WRITE. The frozen-record rule — the durable write takes --write.

Each refusal is bracketed by the non-refusal it is the mirror of, so a check that raised
unconditionally would fail here rather than read as green.

  uv run pytest tools/mining/test_mining_gate_lock.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import floors as F                 # noqa: E402
from tools.mining import lock_mining_gate as L         # noqa: E402
from tools.mining import mining_pins as MP             # noqa: E402


# EVERY test here reads a stage-2 pin: the lock IS the pinned head's record.
pytestmark = pytest.mark.stage2_pinned


@pytest.fixture(scope="module")
def lock():
    return L.read_lock()


# --------------------------------------------------------------------------- #
# 1. the committed record IS the derivation
# --------------------------------------------------------------------------- #
def test_the_committed_lock_matches_what_the_sitting_and_the_live_floors_derive():
    """Derive-in-code / freeze-in-record, checked in the direction that catches drift: the
    record is regenerated in memory and compared byte-for-byte with the tree's copy."""
    derived = L.serialize(L.derive())
    assert (ROOT / L.LOCK_PATH).read_text(encoding="utf-8") == derived
    assert (ROOT / L.MD_PATH).read_text(encoding="utf-8") == L.write_md(L.derive())


def test_the_lock_quotes_the_owner_cuts_and_their_measured_operating_points(lock):
    """The record may not carry its own copy of a cut value: both come from `floors.py`, and
    the numbers beside them are the swept rows at exactly those values."""
    assert lock["cuts"]["mining_pool"]["value"] == F.MINING_POOL.value
    assert lock["cuts"]["mining_release"]["value"] == F.MINING_RELEASE.value
    rel = lock["cuts"]["mining_release"]
    # `acts` is DERIVED from the owner, so the record tracks the retirement instead of
    # outliving it: the cut went annotation-only on 2026-08-09 and the frozen MEASUREMENT
    # beside it is unchanged, which is the whole reason the record survives the retirement.
    assert rel["acts"] is False and rel["site"] == "release"
    # The COUNTS are read off the same source the record derives from, not restated: they
    # move with every head flip (the cut is volume-matched) and a literal here would go red
    # FOR the flip rather than for a fault.
    vm = json.loads((ROOT / L.SOURCE_REPORT).read_text(encoding="utf-8"))
    src = next(c for c in vm["cuts"] if c["name"] == "mining_release")
    assert rel["n"] == vm["reference_pool"]["n"]
    assert rel["fires"] == src["incoming"]["n_selected"] == src["matched_volume"]
    assert rel["tp"] == src["incoming"]["tp"]
    lo, hi = rel["precision_ci95"]
    assert lo < rel["precision"] < hi                       # a Wilson interval, not a point
    row = next(r for r in lock["ladder_ge3"]
               if abs(r["threshold"] - F.MINING_RELEASE.value) < 1e-9)
    assert (row["fires"], row["precision"]) == (rel["fires"], rel["precision"])


def test_every_cut_records_the_value_it_was_restated_FROM_and_what_kind_of_claim_that_is(lock):
    """A restatement that keeps no trace of what it restated is indistinguishable from a
    retune, and the KIND matters as much as the trace: volume-matching asserts the cut's
    VOLUME is unchanged, a crossover asserts the opposite (the label meaning is what is held
    fixed and the volume is free to move). The record must say which — read from the live
    `LockSpec`, never hardcoded, so the check follows a source change instead of going red for
    one."""
    kind = L.SPEC.restatement_kind
    assert kind in ("VOLUME-MATCHED", "CROSSOVER"), kind
    assert lock["restatement"]["kind"] == kind
    for name, cut in lock["cuts"].items():
        rf = cut["restated_from"]
        assert rf["kind"] == kind, name
        assert kind in rf["how"], name
        assert rf["value"] != cut["value"], name
        assert 0.0 <= rf["precision"] <= 1.0, name
        # `fires` is read off the ladder and `matched_volume` off the source's own count, so
        # their agreement is a cross-check of the record against itself under BOTH kinds — it
        # is the volume the cut REALIZES either way.
        assert cut["fires"] == cut["matched_volume"], name
    # And the two kinds differ in exactly one checkable place: a volume match restates from the
    # PREVIOUS head, a crossover from the same one (the head did not move).
    prev = {c["restated_from"]["head"].rsplit("/", 1)[-1] for c in lock["cuts"].values()}
    assert prev == ({MP.HEAD_VERSION} if kind == "CROSSOVER" else {"v1"}), prev


def test_both_boundaries_are_frozen_whole_not_only_the_cut_rows(lock):
    """A record holding only the two cut rows cannot answer "what would 0.40 have bought"
    without re-running a sitting whose crops may be gone by then."""
    assert len(lock["ladder_ge3"]) >= 20 and len(lock["ladder_ge2"]) >= 20
    # the two LIVE cuts must be exact rows of the >=3 ladder (that is what makes `_row_at`
    # able to refuse rather than interpolate); read from the owner, never restated.
    assert {r["threshold"] for r in lock["ladder_ge3"]} >= {F.MINING_POOL.value,
                                                            F.MINING_RELEASE.value}


def test_the_record_states_what_its_numbers_are_an_optimistic_bound_on(lock):
    """The two caveats are the reason this is a ceiling and not an estimate; a lock that
    dropped them would read as a measurement of a fresh population."""
    assert set(lock["caveats"]) == set(L.CAVEATS)
    assert "direction" in lock["caveats"]        # which way each lean points, stated
    assert "OPTIMISTIC" in lock["bound"]
    assert lock["corpus"]["n"] > 0 and lock["corpus"]["n_locations"] > 0
    assert lock["head"]["version"] == MP.HEAD_VERSION
    assert lock["provenance"]["source_report"] == L.SOURCE_REPORT
    # Parity is BY CONSTRUCTION now: the source pass scores through the gate's own scorer,
    # so there is no sibling harness for the numbers to disagree with. The record has to say
    # which scorer that was, or the claim is unfalsifiable.
    assert lock["harness_parity"]["scorer"] == "mining_scorer"


# --------------------------------------------------------------------------- #
# 2. the reader refuses when the pin moves
# --------------------------------------------------------------------------- #
def test_read_lock_refuses_when_the_live_pin_is_not_the_locked_head():
    with pytest.raises(L.LockHeadMismatch) as ei:
        L.read_lock(live_version="v2")
    assert "v2" in str(ei.value) and MP.HEAD_VERSION in str(ei.value)


def test_read_lock_serves_the_record_while_the_pin_matches():
    """The mirror of the refusal — without it, a `read_lock` that always raised would pass."""
    assert L.read_lock(live_version=MP.HEAD_VERSION)["gate"]["version"] == \
        MP.MINING_GATE_VERSION


def test_the_refusal_follows_the_pin_module_at_call_time(monkeypatch):
    """Not an import-time snapshot: moving the pin makes the DEFAULT call refuse too."""
    assert L.read_lock()["head"]["version"] == MP.HEAD_VERSION
    monkeypatch.setattr(MP, "HEAD_VERSION", "v77")
    with pytest.raises(L.LockHeadMismatch):
        L.read_lock()


# --------------------------------------------------------------------------- #
# 3. the derivation refuses rather than interpolating
# --------------------------------------------------------------------------- #
def test_a_cut_the_sitting_never_swept_is_an_error_not_a_nearest_bin():
    ladder = [{"threshold": 0.45, "fires": 1, "tp": 1, "pass_rate": 0.1, "precision": 1.0,
               "precision_lo": 0.1, "precision_hi": 1.0, "recall": 0.1}]
    assert L._row_at(ladder, 0.45)["fires"] == 1                     # non-vacuous
    with pytest.raises(L.LockDerivationError, match="not a swept row"):
        L._row_at(ladder, 0.50)


def test_a_pass_that_scored_another_checkpoint_cannot_lock_this_gate():
    vm = json.loads((ROOT / L.SOURCE_REPORT).read_text(encoding="utf-8"))
    L.build_lock(vm, source_sha="x")                                 # non-vacuous
    vm["head"]["incoming"] = "data/render_mode_head/v2/model_best.pt"
    with pytest.raises(L.LockDerivationError, match="DEPLOYED head"):
        L.build_lock(vm, source_sha="x")


def test_a_BOUNDED_pass_cannot_lock_this_gate():
    """`volume_match.py --limit` stamps `incomplete`, and a lock derived from a partial pass
    would state an operating point nobody measured. Same shape as the `sitting_cut.INCOMPLETE`
    rule: a bounded run that writes real files stamps itself unusable."""
    vm = json.loads((ROOT / L.SOURCE_REPORT).read_text(encoding="utf-8"))
    assert vm["incomplete"] is False                                 # non-vacuous
    vm["incomplete"] = True
    with pytest.raises(L.LockDerivationError, match="incomplete"):
        L.build_lock(vm, source_sha="x")


def test_a_floor_moved_off_the_sitting_cannot_be_quoted(monkeypatch):
    """A value nobody measured cannot acquire a precision by being written into the record."""
    moved = F.Floor(name="mining_release", value=0.55, head=F.MINING_HEAD,
                    stamp=MP.HEAD_VERSION, site="release", basis="injected")
    monkeypatch.setattr(L, "LOCKED_CUTS", (F.MINING_POOL, moved))
    with pytest.raises(L.LockDerivationError, match="only\n?\\s*quote a cut"):
        L.derive()


# --------------------------------------------------------------------------- #
# 4. the frozen-record write rule
# --------------------------------------------------------------------------- #
def test_a_default_run_verifies_and_writes_nothing(tmp_path, monkeypatch):
    """The durable write takes --write. Proved on BOTH sides: the default leaves the file
    byte-identical, and the verify still reports OK on the tree's real record."""
    before = (ROOT / L.LOCK_PATH).read_bytes()
    proc = subprocess.run([sys.executable, str(ROOT / "tools/mining/lock_mining_gate.py")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (ROOT / L.LOCK_PATH).read_bytes() == before


def test_the_verify_is_red_on_drift(tmp_path, monkeypatch):
    """INJECTION: point the writer's outputs at a copy holding a hand-edited number. Without
    this, a verify that always returned 0 would pass the test above."""
    edited = json.loads((ROOT / L.LOCK_PATH).read_text(encoding="utf-8"))
    edited["cuts"]["mining_release"]["precision"] = 0.5              # a number nobody measured
    (tmp_path / "mining_gate_lock.json").write_text(json.dumps(edited, indent=2) + "\n",
                                                    encoding="utf-8")
    (tmp_path / "mining_gate_lock.md").write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(L.P, "durable", lambda rel, **kw: tmp_path / Path(rel).name)
    monkeypatch.setattr(sys, "argv", ["lock_mining_gate.py"])
    assert L.main() == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
