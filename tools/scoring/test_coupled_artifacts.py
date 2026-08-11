#!/usr/bin/env python
"""The revert-together set, walked from ONE list instead of asserted in four places.

`production_pins.COUPLED_ARTIFACTS` enumerates everything that must name the live head: the
pin, the discovery table and its derivation, the keeper cut, the two vendored tau_h bases and
the tau_h provenance record. Each already had a guard somewhere; what did not exist was a way
to ASK the question. "What must move with the pin?" was answerable only by flipping the pin
and seeing what went red — which is how the v10 flip discovered nine failures across four
files, three of them tests that had hardcoded "v8" and went red *for* the flip rather than
for a fault.

Three properties, all of them about the list rather than about any one artifact:

  1. every entry that declares a stamp reads back the ACTIVE version;
  2. every entry names a guard file that exists, so an entry cannot be added as a comment;
  3. the enumeration in `data/<v>/build_metadata.json:rollback_ladder.must_revert_together`
     is a SUBSET of it — the JSON record and the code agree, which is the property that had
     no owner when the same set lived in prose, in JSON and in four test files.

`pytest -m version_pinned` lists this file and its siblings — that is the answer to the
question above, delivered without flipping anything.

  uv run pytest tools/scoring/test_coupled_artifacts.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.version_pinned

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools" / "atlas", ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import production_pins as pins  # noqa: E402
from production_pins import ACTIVE_VERSION, COUPLED_ARTIFACTS  # noqa: E402

def _ladder_record() -> dict:
    """The live version's `rollback_ladder` block, wherever that version keeps it.

    Two homes on purpose: v10's BUILD wrote a "what a future adoption would have to revert"
    note into `build_metadata.json`, and v11's ADOPTION writes the real thing into
    `adoption_record.json` — a different object with a different date. Resolved rather than
    branched on a version literal, which is what this whole file exists to avoid."""
    for name in ("adoption_record.json", "build_metadata.json"):
        p = ROOT / "data" / ACTIVE_VERSION / name
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            if "rollback_ladder" in doc:
                return doc["rollback_ladder"]
    raise AssertionError(f"no rollback_ladder record for the live head {ACTIVE_VERSION}")

STAMPED = [e for e in COUPLED_ARTIFACTS if e.get("stamp") is not None]


@pytest.mark.parametrize("entry", STAMPED, ids=[e["what"] for e in STAMPED])
def test_every_coupled_artifact_names_the_active_head(entry):
    got = pins.coupled_stamp(entry)
    assert got == ACTIVE_VERSION, (
        f"{entry['what']} is stamped {got!r} but the active head is {ACTIVE_VERSION!r} — "
        f"{entry['why']}. A partial rollback is what this catches; revert the whole set.")


@pytest.mark.parametrize("entry", COUPLED_ARTIFACTS, ids=[e["what"] for e in COUPLED_ARTIFACTS])
def test_every_entry_names_a_guard_that_exists(entry):
    """An entry whose guard file is gone is an entry nothing enforces — which is exactly the
    state the prose list was in before this file existed."""
    guard = ROOT / entry["guard"]
    assert guard.exists(), f"{entry['what']} names a guard that is not on disk: {entry['guard']}"
    assert entry["why"], entry["what"]


# Subjects a COMMITTED RECORD still names and the live registry deliberately does not, each
# because the thing itself was deleted rather than uncoupled (2026-08-09,
# prompts/selection_restructure_3.md). The committed json files survive as records of what
# their heads served; nothing reads them, so nothing about them has to move at a flip. An
# entry here must name something genuinely GONE from the tree — asserted below.
RETIRED_SUBJECTS = {
    "tools/atlas/production_seeder.T_GOOD_OVERRIDES",
    "data/{v}/t_good_derivation.json",
    "data/atlas/keeper_cuts.json",
}


def test_the_retired_subjects_are_actually_gone():
    """The other half of `RETIRED_SUBJECTS`: an exemption that outlives its subject's
    deletion is how a registry quietly stops covering something live again."""
    import subprocess
    src = (ROOT / "tools" / "atlas" / "production_seeder.py").read_text(encoding="utf-8")
    assert "T_GOOD_OVERRIDES = {" not in src
    out = subprocess.run(["git", "ls-files", "tools/scoring/derive_t_good.py",
                          "tools/atlas/keeper_cut.py"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.stdout.strip() == "", f"still tracked: {out.stdout!r}"


def test_every_entry_carries_a_stamp():
    """Non-vacuity in the other direction: `stamp: None` must be a fact about the artifact,
    not a way to opt out of the check. There is no such entry any more — the only one there
    ever was, `production_seeder.T_GOOD_OVERRIDES` (a bare dict of floats in a .py file, whose
    version was carried by the derivation artifact beside it), was deleted with the rest of
    the per-partition t_good machinery on 2026-08-09."""
    unstamped = [e["what"] for e in COUPLED_ARTIFACTS if e.get("stamp") is None]
    assert unstamped == [], unstamped
    assert len(STAMPED) == len(COUPLED_ARTIFACTS) >= 4, \
        "the set lost stamped entries — did someone opt one out?"


def test_the_two_enforcing_floors_are_deliberately_NOT_in_this_set():
    """`floors.GOOD_FLOOR` / `JUNK_FLOOR` are cuts on the same train-prior-calibrated scale
    as everything above and are just as scale-bound — and they are still not registered here,
    because a stamp check is the wrong instrument for them. An artifact in this set is
    RE-DERIVED at a flip and its stamp says which head it was derived under; `GOOD_FLOOR` is
    RESTATED volume-matched against the re-scored pool, and the correct new value depends on a
    measurement rather than on a version. Registering them would make a flip pass by moving a
    string. `JUNK_FLOOR` is not restated at all as of 2026-08-11 — permanent shared-scale,
    since it is read on two heads at once (`test_floors.py` pins that declaration) — which is a
    second, independent reason it does not belong in a set whose whole content is "re-derive
    this when the head moves". The procedure that does hold them is prose plus a human:
    docs/design/classifier_retrain_protocol.md §5."""
    from tools.emission import floors as F
    assert not any("GOOD_FLOOR" in e["what"] or "JUNK_FLOOR" in e["what"] or
                   "emission.floors" in e["what"] for e in COUPLED_ARTIFACTS)
    assert (F.GOOD_FLOOR, F.JUNK_FLOOR) == (0.50, 0.20)
    proto = (ROOT / "docs" / "design" / "classifier_retrain_protocol.md").read_text(
        encoding="utf-8")
    assert "GOOD_FLOOR" in proto and "volume-match" in proto.lower(), \
        "the flip procedure must name the restatement the registry deliberately cannot check"


def test_the_build_metadata_record_is_a_subset_of_the_registry():
    """The JSON record and the code must name the same things.

    `must_revert_together` is a frozen record — through v10 it kept the four items that were
    known when the BUILD ran and could lag the registry (which has since added the t_good
    derivation, the campaign floor and the tau_h provenance file). From v11 the record is
    written BY the adoption off `COUPLED_ARTIFACTS` itself, so it is the whole set rather than
    a lagging subset. Either way what it may NOT do is name something the registry has
    forgotten, and that is the direction asserted.

    RETIRED subjects are the one exemption, and they are LISTED rather than pattern-matched.
    A committed record keeps what was true when it was written (`storage_classes.md`), so the
    v11 adoption record still names the three artifacts the 2026-08-09 restructure deleted.
    Naming them here is what stops "the registry forgot it" and "the subject no longer exists"
    from reading the same."""
    meta = _ladder_record()
    known = {e["what"] for e in COUPLED_ARTIFACTS}
    for item in meta["must_revert_together"]:
        what = item["what"]
        if what in RETIRED_SUBJECTS:
            continue
        # the record writes the tau_h pair and its provenance file as one line; the registry
        # splits them, so match on the leading constant rather than the whole string.
        head = what.split(",")[0].strip()
        assert any(head in k or k in head for k in known), (
            f"build_metadata names {what!r} in must_revert_together, but "
            f"production_pins.COUPLED_ARTIFACTS does not cover it — the record and the "
            f"registry have drifted apart")


def test_the_prose_block_and_the_registry_have_not_drifted_apart():
    """The comment beside ACTIVE_CKPT numbers the same set. Nothing can hold a comment to a
    list, but it can hold the COUNT, which is what silently rots when someone adds an artifact
    to one and not the other. The registry splits the tau_h pair (constant + stamp) from its
    provenance file where the prose numbers them 2 and 3, and adds the campaign floor, so the
    two counts differ by one BY CONSTRUCTION — asserted as a relation, not as a literal."""
    src = (ROOT / "tools" / "scoring" / "production_pins.py").read_text(encoding="utf-8")
    block = src.split("THE REVERT-TOGETHER SET", 1)[1].split("=" * 32, 1)[0]
    numbered = [ln for ln in block.splitlines()
                if ln.lstrip("# ").strip()[:2] in {f"{i}." for i in range(1, 10)}]
    assert len(numbered) == len(COUPLED_ARTIFACTS) - 1, (
        f"the prose block numbers {len(numbered)} items against "
        f"{len(COUPLED_ARTIFACTS)} registry entries; if the revert-together set changed, "
        f"change BOTH it and COUPLED_ARTIFACTS")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
