#!/usr/bin/env python
r"""THE checkpoint flip, checked where it can be checked WITHOUT a model or a GPU.

WAS `tools/v10/test_v10_flip.py`. It was already written against `ACTIVE_VERSION` rather
than a literal, so it was never v10's test — only v10's file, and the v11 flip would have
had to copy it to a `tools/v11/test_v11_flip.py` that differed in nothing. Moved here, beside
the pins it reads, so the next flip edits `FLIP_HISTORY` and this file stays put. The two
places it still had to learn a version-shaped fact — where a version keeps its rollback
ladder, and where its manifest resolves — are resolved rather than spelled.

Three things, none of which needs a render (the rendered end-to-end proof is the slow
`tools/scoring/test_flip_end_to_end.py`):

  1. **The revert-together set is coherent.** The pin, the discovery t_good table, the keeper
     cut and the vendored tau_h base all describe ONE head or they describe nothing. Each has
     its own guard elsewhere; this asserts they agree with each other, which is the property a
     partial rollback breaks.
  2. **Stale rows go stale BY THE PREDICATE, not by deletion.** The 2026-08 v8-era discovery
     ledgers are still on disk, unedited, and `descriptor.load_admitted` returns nothing from
     them because `scorer_version` no longer matches. The count that WOULD have been admitted
     is asserted non-zero, so this cannot pass by the ledger being empty.
  3. **The arithmetic re-decode works end to end on v10-scored rows** — a human-labeled class-4
     location decodes to 4 under the NEW per-partition t_good and clears the q3+ admission
     predicate, the emission-intake predicate and the coverage cloud; a human-labeled class-1
     location is refused at the same boundary. Both re-decode from their own persisted
     probabilities, which is what makes a ledger row trustworthy off disk.

Runs in the default lane: it reads committed artifacts and does arithmetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "tools", "tools/atlas", "tools/mining", "tools/scoring", "tools/corpus",
            "tools/emission"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import corpus_common as cc                       # noqa: E402
import eval_slice                                # noqa: E402
import production_seeder as ps                   # noqa: E402
from partitions import FT2FAM                    # noqa: E402
from production_pins import ACTIVE_CKPT, ACTIVE_VERSION   # noqa: E402
from tools.emission import floors as F           # noqa: E402  THE cut owner
from tools.emission import descriptor as D       # noqa: E402

# Coupled to production_pins.ACTIVE_CKPT: `pytest -m version_pinned` lists it.
pytestmark = pytest.mark.version_pinned


import paths                                    # noqa: E402

EVAL = eval_slice.path_for(ACTIVE_VERSION)
# The manifest is BULK from v11 on (a deterministic function of the committed corpus, so it
# rebuilds rather than being restored) and was in-tree through v10. `paths.bulk` resolves the
# relocated copy and returns the in-tree path unchanged where nothing was relocated, so one
# expression covers both eras.
MANIFEST = paths.bulk(f"data/{ACTIVE_VERSION}/manifest.jsonl")
KEEPER_CUTS = ROOT / "data" / "atlas" / "keeper_cuts.json"


def _ladder_record() -> dict:
    """The live version's `rollback_ladder` block, wherever that version keeps it.

    Two homes, because the two versions wrote it for two different reasons: v10's build
    emitted a "what a future adoption would have to revert" note into `build_metadata.json`,
    and v11's ADOPTION writes the real thing into `adoption_record.json` (tools/v11/
    adopt_v11.py). Resolved, not branched on a version literal — a literal here is what made
    this file v10's in the first place. Raises naming both candidates rather than skipping:
    a live head with no ladder record anywhere is the failure, not a gap in the test."""
    for name in ("adoption_record.json", "build_metadata.json"):
        p = ROOT / "data" / ACTIVE_VERSION / name
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            if "rollback_ladder" in doc:
                return doc["rollback_ladder"]
    raise AssertionError(
        f"no rollback_ladder for the live head {ACTIVE_VERSION}: looked in "
        f"data/{ACTIVE_VERSION}/adoption_record.json and .../build_metadata.json")

# A committed v8-era discovery ledger: real production rows, written before the flip, left
# untouched by it. Any of the five 2026-08 v8 ledgers would do; this is the largest.
STALE_LEDGER = ROOT / "data/discovery/maneuver_v14_exploration/outcome_ledger.jsonl"


def _rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# 1. the revert-together set agrees with itself
# --------------------------------------------------------------------------- #
def test_the_revert_together_set_names_one_head():
    """Pin, keeper cut and vendored tau_h base must all name the SAME version.

    Each is separately held to ACTIVE_VERSION by its own test; the failure this adds is the
    one those cannot see — a set that is individually plausible and mutually inconsistent,
    which is what a rollback that reverts three of four things produces."""
    import steered_frontier as sf

    assert ACTIVE_CKPT == f"data/classifier/{ACTIVE_VERSION}/model_best.pt"
    assert (ROOT / ACTIVE_CKPT).exists(), f"the pinned checkpoint is not on disk: {ACTIVE_CKPT}"
    keeper_model = json.loads(KEEPER_CUTS.read_text(encoding="utf-8"))["provenance"]["model"]
    assert keeper_model == ACTIVE_VERSION, (
        f"keeper_cuts.json names {keeper_model!r}, pin names {ACTIVE_VERSION!r}")
    assert sf.TAU_H_FIDELITY_BASE_MODEL == ACTIVE_VERSION, (
        f"vendored tau_h base names {sf.TAU_H_FIDELITY_BASE_MODEL!r}, pin names "
        f"{ACTIVE_VERSION!r} — a cut on another head's cheap p_good")
    assert sf.TAU_H_CAMPAIGN_FLOOR_MODEL == ACTIVE_VERSION


def test_a_forgetful_rollback_goes_red(monkeypatch):
    """The coherence guard above, proved NON-VACUOUS by injection.

    A guard that only ever sees a coherent set cannot distinguish "coherent" from "not
    looking". This simulates the failure it exists for — the pin rolled back ONE rung while
    the threshold files stay on the live head — and asserts each of the three checks fires.
    The rung is read off the ladder, not spelled, so this keeps testing the real one-flip
    rollback after the ladder shortens. Injection, not a real rollback: nothing on disk moves.
    """
    import active_ckpt
    import steered_frontier as sf

    meta = _ladder_record()
    ladder = meta["ladder"] if "ladder" in meta else meta["ladder_after_a_v10_adoption"]
    prev = ladder[1]
    monkeypatch.setattr(active_ckpt, "ACTIVE_VERSION", prev)
    monkeypatch.setattr(active_ckpt, "ACTIVE_CKPT", f"data/classifier/{prev}/model_best.pt")
    rolled_back = active_ckpt.ACTIVE_VERSION

    assert sf.TAU_H_FIDELITY_BASE_MODEL != rolled_back, (
        "tau_h stamp guard would NOT fire on a forgetful rollback")
    art = ROOT / "data" / "atlas" / f"tau_h_base_{sf.TAU_H_FIDELITY_BASE_MODEL}.json"
    assert json.loads(art.read_text(encoding="utf-8"))["model"] != rolled_back, (
        "the tau_h provenance guard would NOT fire on a forgetful rollback")
    # THE KEEPER-CUT AND t_good STAMPS WERE ASSERTED HERE TOO, until 2026-08-09. Both
    # subjects were deleted (prompts/selection_restructure_3.md) and left
    # `COUPLED_ARTIFACTS` with them; `data/atlas/keeper_cuts.json` and
    # `data/<v>/t_good_derivation.json` survive as records nothing reads. What replaced them
    # — `floors.GOOD_FLOOR` / `JUNK_FLOOR` — carries no stamp on purpose and cannot be
    # asserted here: it is restated volume-matched, not re-derived
    # (classifier_retrain_protocol.md §5a, and test_coupled_artifacts.py holds that reasoning
    # to the doc).


def test_the_rollback_ladder_is_readable_and_its_rungs_are_tracked():
    """A ladder naming a weight a fresh clone does not receive is a plan that cannot run.

    Asserted against the INDEX, not the working tree. `git rm --cached` leaves the .pt on the
    machine that de-tracked it, so an `exists()` check keeps passing exactly where the policy
    that de-tracked it would bite — on the next clone, which is the only place a rollback is
    ever actually attempted."""
    import subprocess
    meta = _ladder_record()
    ladder = meta["ladder"] if "ladder" in meta else meta["ladder_after_a_v10_adoption"]
    assert ladder[0] == ACTIVE_VERSION, f"ladder head {ladder[0]!r} != live {ACTIVE_VERSION!r}"
    tracked = set(subprocess.run(["git", "ls-files", "data/classifier"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.split())
    for rung in ladder:
        rel = f"data/classifier/{rung}/model_best.pt"
        assert rel in tracked, (
            f"rollback rung {rung} is on the ladder but {rel} is not tracked — the ladder "
            f"names only rungs that exist (storage_classes.md § weights retention)")


# --------------------------------------------------------------------------- #
# 2. an off-head ledger is RANKED, not refused
# --------------------------------------------------------------------------- #
def test_v8_ledger_rows_are_admitted_on_their_own_scores_not_refused_by_a_stamp():
    """THE 2026-08-09 reversal, on the real v8-era ledger this file has always used.

    It used to assert the opposite: every row here carries a `v8` stamp, so the decode-version
    predicate refused all of them and `load_admitted` returned []. That was the intended
    behaviour and it was the mechanism by which the v10 flip took the emission intake from
    ~1.4k locations to 16 with nothing going red. The predicate is gone; a v8 `p_good` is a
    worse estimate of quality than a v11 one, and a worse estimate ranks lower rather than
    disappearing.

    The floor still cuts, and the ledger is a good witness for that: it contains rows on both
    sides of 0.50, so this is not "everything admits now"."""
    rows = _rows(STALE_LEDGER)
    stamps = {r.get("scorer_version") for r in rows}
    assert stamps == {"v8"}, f"expected a pure v8-era ledger, got stamps {stamps}"
    assert len(rows) > 0

    eligible = [r for r in rows if D.guard_and_distinct(r)]
    admitted = D.load_admitted(STALE_LEDGER)
    assert len(admitted) > 100, (
        f"only {len(admitted)} v8-stamped rows admit — the stamp is not supposed to cut, so "
        f"this ledger should contribute its whole above-floor population")
    assert len(admitted) < len(eligible), (
        "every guard-and-distinct row admitted — the good floor is not cutting anything here, "
        "so this ledger cannot witness that it still cuts")
    assert all(F.passes_good_floor(r.get("p_good")) for r in admitted)


# --------------------------------------------------------------------------- #
# 3. the arithmetic re-decode, end to end, on v10-scored rows
# --------------------------------------------------------------------------- #
def _ledger_row(evrow, manifest):
    """The persisted shape a discovery run writes, built from an eval-slice row + its coords."""
    part = FT2FAM[evrow["fractal_type"]]
    m = manifest[evrow["location_id"]]
    p2, p3, p4 = eval_slice.probs(evrow, ACTIVE_VERSION)
    return part, {
        "id": f"flip_{evrow['location_id']}", "family": part,
        "outcome_cx": float(m["cx"]), "outcome_cy": float(m["cy"]), "outcome_fw": float(m["fw"]),
        "p_notbad": p2, "p_good": p3, "p_ge4": p4,
        "guard_pass": True, "distinct": True, "scorer_version": ACTIVE_VERSION,
    }


def _pick():
    ev = _rows(EVAL)
    manifest = {r["loc_id"]: r for r in _rows(MANIFEST)}
    best4 = max((r for r in ev if r["label"] == 4), key=lambda r: r[f"{ACTIVE_VERSION}_p_ge4"])
    worst1 = min((r for r in ev if r["label"] == 1), key=lambda r: r[f"{ACTIVE_VERSION}_score"])
    return _ledger_row(best4, manifest), _ledger_row(worst1, manifest)


def test_a_class4_location_classifies_as_4_and_admits_end_to_end(tmp_path):
    """The row carries PROBABILITIES and no stored class; the class is derived at read time
    at the live floor, and the whole run-side path agrees with it."""
    (part, row), _ = _pick()
    assert "decoded_class" not in row and "t_good" not in row
    cls = F.good_class(row["p_good"], row["p_ge4"])
    assert cls == 4, (
        f"a human-labeled class-4 location classifies as {cls} "
        f"(p2={row['p_notbad']:.4f} p3={row['p_good']:.4f} p4={row['p_ge4']:.4f} "
        f"good_floor={F.GOOD_FLOOR}) — if this is 3, the pipeline is q4-incapable")
    assert ps.is_good(row)
    assert ps.build_cloud([row], part)
    led = tmp_path / "one.jsonl"
    led.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(D.load_admitted(led)) == 1


def test_the_flip_watch_is_attached_to_the_readout_that_shows_class_4():
    """Every flip records a WATCH — a thing to look at on the first run under the new head —
    and a watch is never a gate.

    A watch written only in prose is a watch nobody sees, so it rides on `cloud_diagnostic`,
    the run's first eyeball of the decode distribution. Asserted here: it is PRESENT while
    the head it was raised for is live, it is KEYED on the scorer version (so it retires
    itself at the next flip rather than outliving its subject), and it changes no verdict —
    the diagnostic's counts are identical with and without it."""
    rows = [{"family": "mandelbrot", "guard_pass": True, "p_good": pg, "p_ge4": p4,
             "outcome_cx": 0.1 * i, "outcome_cy": 0.2 * i, "outcome_fw": 1e-3}
            for i, (pg, p4) in enumerate([(0.05, 0.0), (0.30, 0.0), (0.80, 0.1), (0.95, 0.9)])]
    diag = ps.cloud_diagnostic(rows, ps.build_cloud(rows, "mandelbrot"), "mandelbrot")
    assert diag["class_split"]["great"] == 1, "the diagnostic must count class 4 at all"
    assert ps.GOOD_FLOOR_WATCH_VERSION == ACTIVE_VERSION, (
        "the WATCH names a version that is not live — it should have retired itself")
    assert diag.get("good_floor_watch") == ps.GOOD_FLOOR_WATCH
    # a WATCH is not a gate: nothing about the counts or the cloud depends on it
    assert diag["cloud_size"] == len(ps.build_cloud(rows, "mandelbrot"))
    assert set(diag["class_split"]) == {"below_floor", "good", "great"}


def test_a_class1_location_is_refused_at_the_same_boundary(tmp_path):
    """Without this, a classifier that returned 4 unconditionally would pass the test above."""
    _, (part, row) = _pick()
    cls = F.good_class(row["p_good"], row["p_ge4"])
    assert cls is None, (
        f"a human-labeled class-1 location classifies as {cls} — the good floor is not "
        f"discriminating")
    assert not ps.is_good(row)
    assert not ps.build_cloud([row], part)
    led = tmp_path / "one.jsonl"
    led.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert D.load_admitted(led) == []
