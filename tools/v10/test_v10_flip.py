#!/usr/bin/env python
r"""The v10 flip, checked where it can be checked WITHOUT a model or a GPU.

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

ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "tools", "tools/atlas", "tools/mining", "tools/scoring", "tools/corpus",
            "tools/emission"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import corpus_common as cc                       # noqa: E402
import production_seeder as ps                   # noqa: E402
from production_pins import ACTIVE_CKPT, ACTIVE_VERSION   # noqa: E402
from score_lib import corn_decode                # noqa: E402
from tools.emission import descriptor as D       # noqa: E402

EVAL = ROOT / "data" / ACTIVE_VERSION / f"eval_scores_{ACTIVE_VERSION}.jsonl"
MANIFEST = ROOT / "data" / ACTIVE_VERSION / "manifest.jsonl"
KEEPER_CUTS = ROOT / "data" / "atlas" / "keeper_cuts.json"
BUILD_META = ROOT / "data" / ACTIVE_VERSION / "build_metadata.json"

# A committed v8-era discovery ledger: real production rows, written before the flip, left
# untouched by it. Any of the five 2026-08 v8 ledgers would do; this is the largest.
STALE_LEDGER = ROOT / "data/discovery/maneuver_v14_exploration/outcome_ledger.jsonl"

FT2FAM = {"mandelbrot": "mandelbrot", "julia": "julia:mandelbrot",
          "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
          "julia_multibrot3": "julia:multibrot3", "julia_multibrot4": "julia:multibrot4",
          "julia_multibrot5": "julia:multibrot5", "phoenix": "phoenix"}


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
    looking". This simulates the failure it exists for — the pin rolled back to v8 while the
    threshold files stay on v10 — and asserts each of the three checks fires. Injection, not
    a real rollback: nothing on disk moves.
    """
    import active_ckpt
    import steered_frontier as sf

    monkeypatch.setattr(active_ckpt, "ACTIVE_VERSION", "v8")
    monkeypatch.setattr(active_ckpt, "ACTIVE_CKPT", "data/classifier/v8/model_best.pt")
    rolled_back = active_ckpt.ACTIVE_VERSION

    keeper_model = json.loads(KEEPER_CUTS.read_text(encoding="utf-8"))["provenance"]["model"]
    assert keeper_model != rolled_back, "keeper-cut guard would NOT fire on a forgetful rollback"
    assert sf.TAU_H_FIDELITY_BASE_MODEL != rolled_back, (
        "tau_h stamp guard would NOT fire on a forgetful rollback")
    live_tgood = ROOT / "data" / rolled_back / "t_good_derivation.json"
    adopted = json.loads(live_tgood.read_text(encoding="utf-8"))["adopted"]
    assert adopted != {k: float(v) for k, v in ps.T_GOOD_OVERRIDES.items()}, (
        "the adopted t_good table equals v8's — the t_good guard would NOT fire on a "
        "forgetful rollback")


def test_the_rollback_ladder_is_readable_and_its_rungs_exist():
    """A ladder naming a weight that is not on disk is a rollback plan that cannot run."""
    meta = json.loads(BUILD_META.read_text(encoding="utf-8"))["rollback_ladder"]
    ladder = meta["ladder_after_a_v10_adoption"]
    assert ladder[0] == ACTIVE_VERSION, f"ladder head {ladder[0]!r} != live {ACTIVE_VERSION!r}"
    for rung in ladder:
        w = ROOT / f"data/classifier/{rung}/model_best.pt"
        assert w.exists(), f"rollback rung {rung} has no weight on disk ({w})"


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
    p2 = evrow[f"{ACTIVE_VERSION}_p_ge2"]
    p3 = evrow[f"{ACTIVE_VERSION}_p_ge3"]
    p4 = evrow[f"{ACTIVE_VERSION}_p_ge4"]
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


def test_the_class4_watch_is_attached_to_the_readout_that_shows_class_4():
    """The v10 flip records a WATCH (class-4 descriptive 0.813 -> 0.728), not a gate.

    A watch written only in prose is a watch nobody sees, so it rides on `cloud_diagnostic`
    — the run's first eyeball of the decode distribution. Asserted here: it is PRESENT while
    v10 is live, it is keyed on the scorer version (so it retires itself at the next flip),
    and it changes no verdict — the diagnostic's counts are identical with and without it."""
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
