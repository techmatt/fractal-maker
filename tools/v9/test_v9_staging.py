"""§7 staging invariants: v9's thresholds are BUILT, and provably NOT deployed.

Build is not flip. Every threshold in this pass is derived against the v9 eval slice and
written to a staged path; the live gate keeps running on v8 until the ACTIVE_CKPT flip,
which is its own pass and is conditional on the pre-registered bar. That is easy to say and
easy to violate by one overwrite, so it is checked:

  * `ACTIVE_CKPT` has never pointed at v9;
  * the LIVE keeper cut names the ACTIVE version, so `test_steered_frontier`'s guard
    is intact and a v9 threshold is not sitting on another head's gate;
  * `production_seeder.T_GOOD_OVERRIDES` mirrors the ACTIVE head's derived table, not v9's;
  * the staged v9 artifacts, when present, are stamped STAGED and carry the v9 model —
    except the keeper recut, which was deleted outright once v10 was adopted over v9.

And the τ_h half of §7: τ_h is left FATAL and was NOT re-derived by THIS pass. What is
confirmed is that the version-mismatch raise reports **v9** by name once v9 is the active
head — checked by simulating the active version rather than by flipping it, because flipping
it is exactly what this pass must not do.

  UPDATE 2026-08-02 — the flip happened, to **v10**, not to v9. Three checks here named "v8"
  as a literal and went red for the flip rather than for a violation; that is a test measuring
  its own edit. They now assert what this file is actually about and what stays true for every
  future flip: **v9 was never deployed**, and the live thresholds name whichever head IS. The
  premise did not weaken — v9 remains the one version built, staged and skipped.

  UPDATE 2026-07-31 — the launch-time re-derivation the τ_h tests always pointed at has now
  happened, under the ACTIVE (v8) head, for the minibrot-maneuver shakedown
  (`tools/atlas/tau_h_rederive.py`; `docs/design/minibrot_maneuvers.md`). The staging
  invariant is untouched by it — v9 is still staged, not deployed — but the τ_h check below
  no longer pins the literal "v7"; it pins the RELATION to `ACTIVE_CKPT`, which is what the
  stamp was always for.

Run:  uv run python -m pytest tools/v9/test_v9_staging.py -q
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in ("tools", "tools/atlas", "tools/mining", "tools/scoring"):
    sys.path.insert(0, str(ROOT / p))

import active_ckpt                      # noqa: E402
import steered_frontier as sf           # noqa: E402

# Coupled to production_pins.ACTIVE_CKPT: `pytest -m version_pinned` lists it.
pytestmark = pytest.mark.version_pinned


LIVE_CUTS = ROOT / "data/atlas/keeper_cuts.json"
STAGED_CUTS = ROOT / "data/atlas/keeper_cuts_v9.json"
STAGED_TGOOD = ROOT / "data/v9/t_good_derivation.json"
V8_TGOOD = ROOT / "data/v8/t_good_derivation.json"


# --------------------------------------------------------------------------- #
# The flip has NOT happened.
# --------------------------------------------------------------------------- #
def test_v9_was_never_deployed():
    """v9 is the version that was built, evaluated, staged — and skipped. Its primary arm
    passed on inputs byte-identical to the baseline's, so the verdict was true and empty.
    Deploying it later, by accident or by a rollback that mistakes it for a rung, would
    point every discovery gate at a head no certification ever cleared."""
    assert active_ckpt.ACTIVE_VERSION != "v9", (
        "ACTIVE_CKPT names v9 — v9 was never certified for deployment and is explicitly not "
        "a rollback rung (data/v10/build_metadata.json:rollback_ladder.why_not_v9)")


def test_live_keeper_cut_still_names_the_active_head():
    """The live cuts describe the ACTIVE head's p_good scale. Overwriting them with the v9
    recut would put a v9 threshold on another head's gate — a number about nothing, exactly
    as a v7 cut on a v8 gate was."""
    doc = json.loads(LIVE_CUTS.read_text(encoding="utf-8"))
    active = active_ckpt.ACTIVE_VERSION
    assert doc["provenance"]["model"] == active
    assert doc["eval"] == f"data/{active}/eval_scores_{active}.jsonl"
    assert doc["provenance"]["model"] != "v9", "the STAGED v9 cut was promoted to the live path"


def test_no_t_good_table_can_be_adopted_from_v9_or_anywhere_else():
    """This used to assert that `production_seeder.T_GOOD_OVERRIDES` mirrored the DEPLOYED
    head's derivation and never the staged v9 one — the negative half of the adoption gate.

    The whole per-partition table went on 2026-08-09 (prompts/selection_restructure_3.md), so
    the hazard it guarded (a staged table reaching production by being copied) is unwritable:
    there is no table to copy into. `data/v9/t_good_derivation.json` stays as v9's record and
    the staged marker on it stays true. The assertion follows the hazard rather than the
    file."""
    import production_seeder as ps
    assert not hasattr(ps, "T_GOOD_OVERRIDES") and not hasattr(ps, "t_good_for")
    if STAGED_TGOOD.exists():
        doc = json.loads(STAGED_TGOOD.read_text(encoding="utf-8"))
        assert doc.get("staged") or "STAGED" in json.dumps(doc), (
            "the v9 derivation record must stay marked staged even now that nothing reads it")


# --------------------------------------------------------------------------- #
# The staged artifacts are stamped as staged (skipped until §7 produces them).
# --------------------------------------------------------------------------- #
def test_the_staged_v9_keeper_cut_is_gone_and_stays_gone():
    """`data/atlas/keeper_cuts_v9.json` was DELETED 2026-08-02.

    It was checked here by a `skipif(not exists)` guard, which was right while the file was
    a pending §7 output — but once v10 was adopted over v9 that guard degrades to a
    permanent silent skip, and a test that can only skip is a memory of a test. The file is
    a keeper threshold on a head no gate will ever run; the invariant worth keeping is not
    "it is stamped STAGED" but "it is not there at all", which is also the strongest form of
    `test_live_keeper_cut_still_names_the_active_head`'s negative. `tools/v9/keeper_cut_v9.py`
    still exists, so re-running it is the way to get the file back — and if that ever happens
    this test is the thing that says out loud that it did."""
    assert not STAGED_CUTS.exists(), (
        f"{STAGED_CUTS} is back. It was deleted as a threshold on a never-deployed head "
        f"(v9 is not a rollback rung — data/v10/build_metadata.json:rollback_ladder). If a "
        f"question genuinely needs it, say so here rather than letting it sit unexplained.")


@pytest.mark.skipif(not STAGED_TGOOD.exists(), reason="v9 t_good not derived yet")
def test_staged_t_good_is_v9_same_objectives_and_keeps_uncalibrated_uncalibrated():
    doc = json.loads(STAGED_TGOOD.read_text(encoding="utf-8"))
    assert doc["model"] == "v9"
    assert "STAGED" in doc.get("status", "")
    v8 = json.loads(V8_TGOOD.read_text(encoding="utf-8"))
    # SAME objectives — re-deriving on a different one would confound the cap with the
    # objective, which is the one thing this pass must not do.
    assert doc["objective_by_partition"] == v8["objective_by_partition"]
    assert doc["default_objective"] == v8["default_objective"]
    # every derived partition reports BOTH objectives, as v8 does
    for fam, blk in doc["derived"].items():
        assert set(blk["by_objective"]) == {"F0.5", "F2"}, fam
    # a partition with no eval rows is UNCALIBRATED, never a derived 0.50
    for fam, blk in doc["uncalibrated"].items():
        assert blk["status"] == "UNCALIBRATED", fam
        assert fam not in doc["adopted"], (
            f"{fam} is uncalibrated but appears in `adopted` — a baseline 0.50 and a "
            f"derived 0.50 are indistinguishable as a bare number, which is the whole "
            f"reason the distinction is carried explicitly")


# --------------------------------------------------------------------------- #
# τ_h — left fatal, and it names v9.
# --------------------------------------------------------------------------- #
def test_tau_h_mismatch_raise_names_v9_as_the_active_version(monkeypatch, tmp_path):
    """§7: leave τ_h fatal; confirm the version-mismatch raise reports v9.

    τ_h is a cut on the CHEAP-render p_good of a specific head; the vendored base is stamped
    with whatever head it was derived under (v7-era when this test was written, v8 since the
    2026-07-31 re-derivation). Either way it is NOT re-derived by THIS pass. What must hold
    is that the moment v9 is the active head, the vendored fallback refuses BY NAME rather
    than serving a stale float that looks authoritative.

    The active version is SIMULATED, not flipped: flipping it is precisely what this pass
    must not do, and a test that required the flip to prove the post-flip behaviour would
    be unrunnable until it was too late to matter."""
    monkeypatch.setattr(sf, "FIDELITY_RECORDS", tmp_path / "records_do_not_exist.json")
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: "v9")
    with pytest.raises(SystemExit) as ei:
        sf.derive_tau_h(["mandelbrot"])
    msg = str(ei.value)
    assert "v9" in msg, msg
    assert sf.TAU_H_FIDELITY_BASE_MODEL in msg, msg
    assert "tau_h_rederive" in msg, "the raise should name the re-derivation route"
    # non-vacuity: the same call under the stamped head does NOT raise
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: sf.TAU_H_FIDELITY_BASE_MODEL)
    assert sf.derive_tau_h(["mandelbrot"])["mandelbrot"] > 0


def test_tau_h_vendored_base_tracks_the_ACTIVE_head_not_a_frozen_version_string():
    """τ_h's stamp must name the LIVE gate's head, and base+stamp move together.

    This test used to pin `TAU_H_FIDELITY_BASE_MODEL == "v7"` — correct while the v9 staging
    pass was the only thing that had touched τ_h, and its own docstring said so: the v7 pair
    stands "until a launch-time re-derivation replaces BOTH together". That re-derivation
    happened (2026-07-31, `tools/atlas/tau_h_rederive.py` -> `data/atlas/tau_h_base_v8.json`),
    so pinning the literal would now assert the staleness the stamp exists to prevent.

    The invariant that actually survives a re-derivation is the RELATION: the vendored base
    is stamped with whatever `ACTIVE_CKPT` currently is, and it is non-empty. That still
    fails loudly on the thing this test was written to catch — a head flip that leaves τ_h
    behind — and it now also catches a re-derivation that updates one of the pair.
    """
    assert sf.TAU_H_FIDELITY_BASE_MODEL == active_ckpt.ACTIVE_VERSION, (
        f"τ_h is stamped {sf.TAU_H_FIDELITY_BASE_MODEL!r} against active "
        f"{active_ckpt.ACTIVE_VERSION!r} — re-run tools/atlas/tau_h_rederive.py and update "
        f"TAU_H_FIDELITY_BASE + TAU_H_FIDELITY_BASE_MODEL together")
    assert sf.TAU_H_FIDELITY_BASE, "vendored base emptied — τ_h would fail at launch"
    # v9 is STAGED, not deployed, so the stamp must not have run ahead of the flip either.
    assert sf.TAU_H_FIDELITY_BASE_MODEL != "v9"
