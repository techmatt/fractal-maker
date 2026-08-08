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
from score_lib import corn_decode                # noqa: E402
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

    keeper_model = json.loads(KEEPER_CUTS.read_text(encoding="utf-8"))["provenance"]["model"]
    assert keeper_model != rolled_back, "keeper-cut guard would NOT fire on a forgetful rollback"
    assert sf.TAU_H_FIDELITY_BASE_MODEL != rolled_back, (
        "tau_h stamp guard would NOT fire on a forgetful rollback")
    live_tgood = ROOT / "data" / rolled_back / "t_good_derivation.json"
    adopted = json.loads(live_tgood.read_text(encoding="utf-8"))["adopted"]
    assert adopted != {k: float(v) for k, v in ps.T_GOOD_OVERRIDES.items()}, (
        f"the adopted t_good table equals {rolled_back}'s — the t_good guard would NOT fire "
        f"on a forgetful rollback")


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
# 2. stale by predicate, not by deletion
# --------------------------------------------------------------------------- #
def test_v8_ledger_rows_are_refused_by_the_version_predicate_not_deleted():
    rows = _rows(STALE_LEDGER)
    stamps = {r.get("scorer_version") for r in rows}
    assert stamps == {"v8"}, f"expected a pure v8-era ledger, got stamps {stamps}"
    assert not any(cc.is_current_decoded(r) for r in rows)

    # The rows are still THERE — nothing was hand-deleted.
    assert len(rows) > 0
    # ...and they are substantive: without the version predicate most would admit. If this
    # count were 0 the test below would pass on an empty set and prove nothing.
    would_admit = [r for r in rows
                   if r.get("guard_pass") and r.get("distinct") and D.admit_quality(r)]
    assert len(would_admit) > 100, (
        f"only {len(would_admit)} rows would admit ignoring the stamp — pick a ledger with "
        f"real admissions or this asserts nothing")

    assert D.load_admitted(STALE_LEDGER) == [], (
        "a v8-decoded row survived load_admitted after the flip")


def test_require_current_raises_on_a_stale_row():
    import pytest
    with pytest.raises(cc.StaleDecodeError):
        D.load_admitted(STALE_LEDGER, require_current=True)


# --------------------------------------------------------------------------- #
# 3. the arithmetic re-decode, end to end, on v10-scored rows
# --------------------------------------------------------------------------- #
def _ledger_row(evrow, manifest):
    """The persisted shape a discovery run writes, built from an eval-slice row + its coords."""
    part = FT2FAM[evrow["fractal_type"]]
    m = manifest[evrow["location_id"]]
    p2, p3, p4 = eval_slice.probs(evrow, ACTIVE_VERSION)
    t_good = ps.t_good_for(part)
    return part, {
        "id": f"flip_{evrow['location_id']}", "family": part,
        "outcome_cx": float(m["cx"]), "outcome_cy": float(m["cy"]), "outcome_fw": float(m["fw"]),
        "p_notbad": p2, "p_good": p3, "p_ge4": p4, "t_good": t_good,
        "decoded_class": corn_decode(p2, p3, t_good, p4),
        "guard_pass": True, "distinct": True, "scorer_version": ACTIVE_VERSION,
    }


def _pick():
    ev = _rows(EVAL)
    manifest = {r["loc_id"]: r for r in _rows(MANIFEST)}
    best4 = max((r for r in ev if r["label"] == 4), key=lambda r: r[f"{ACTIVE_VERSION}_p_ge4"])
    worst1 = min((r for r in ev if r["label"] == 1), key=lambda r: r[f"{ACTIVE_VERSION}_score"])
    return _ledger_row(best4, manifest), _ledger_row(worst1, manifest)


def test_a_class4_location_decodes_to_4_and_admits_under_the_new_t_good():
    (part, row), _ = _pick()
    assert row["t_good"] == ps.t_good_for(part)
    assert corn_decode(row["p_notbad"], row["p_good"], row["t_good"], row["p_ge4"]) \
        == row["decoded_class"], "the persisted row does not re-decode to its own class"
    assert row["decoded_class"] == 4, (
        f"a human-labeled class-4 location decoded to {row['decoded_class']} "
        f"(p2={row['p_notbad']:.4f} p3={row['p_good']:.4f} p4={row['p_ge4']:.4f} "
        f"t_good={row['t_good']}) — if this is 3, the pipeline is q4-incapable")
    assert ps.is_q3plus(row)
    assert D.admit_quality(row)
    assert ps.build_cloud([row], part)
    assert cc.is_current_decoded(row)


def test_the_flip_watch_is_attached_to_the_readout_that_shows_class_4():
    """Every flip records a WATCH — a thing to look at on the first run under the new head —
    and a watch is never a gate.

    A watch written only in prose is a watch nobody sees, so it rides on `cloud_diagnostic`,
    the run's first eyeball of the decode distribution. Asserted here: it is PRESENT while
    the head it was raised for is live, it is KEYED on the scorer version (so it retires
    itself at the next flip rather than outliving its subject), and it changes no verdict —
    the diagnostic's counts are identical with and without it."""
    rows = [{"family": "mandelbrot", "guard_pass": True, "decoded_class": c,
             "outcome_cx": 0.1 * i, "outcome_cy": 0.2 * i, "outcome_fw": 1e-3}
            for i, c in enumerate((1, 2, 3, 4))]
    diag = ps.cloud_diagnostic(rows, ps.build_cloud(rows, "mandelbrot"), "mandelbrot")
    assert diag["class_split"][4] == 1, "the diagnostic must count class 4 at all"
    assert ps.Q4_WATCH_VERSION == ACTIVE_VERSION, (
        "the q4 WATCH names a version that is not live — it should have retired itself")
    assert diag.get("q4_watch") == ps.Q4_WATCH
    # a WATCH is not a gate: nothing about the counts or the cloud depends on it
    assert diag["cloud_size"] == len(ps.build_cloud(rows, "mandelbrot"))
    assert set(diag["class_split"]) == {1, 2, 3, 4}


def test_a_class1_location_is_refused_at_the_same_boundary():
    """Without this, a decode that returned 4 unconditionally would pass the test above."""
    _, (part, row) = _pick()
    assert row["decoded_class"] < 3, (
        f"a human-labeled class-1 location decoded to {row['decoded_class']} — the q3+ "
        f"boundary is not discriminating")
    assert not ps.is_q3plus(row)
    assert not D.admit_quality(row)
    assert not ps.build_cloud([row], part)
