"""§7 staging invariants: v9's thresholds are BUILT, and provably NOT deployed.

Build is not flip. Every threshold in this pass is derived against the v9 eval slice and
written to a staged path; the live gate keeps running on v8 until the ACTIVE_CKPT flip,
which is its own pass and is conditional on the pre-registered bar. That is easy to say and
easy to violate by one overwrite, so it is checked:

  * `ACTIVE_CKPT` still points at v8 (the flip has not happened here);
  * the LIVE keeper cut still names the ACTIVE version, so `test_steered_frontier`'s guard
    is intact and a v9 threshold is not sitting on a v8 gate;
  * `production_seeder.T_GOOD_OVERRIDES` still mirrors v8's derived table, not v9's;
  * the staged v9 artifacts, when present, are stamped STAGED and carry the v9 model.

And the τ_h half of §7: τ_h is left FATAL and is NOT re-derived here (that happens from
harvest logs at hunt launch). What is confirmed is that the version-mismatch raise reports
**v9** by name once v9 is the active head — checked by simulating the active version rather
than by flipping it, because flipping it is exactly what this pass must not do.

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

LIVE_CUTS = ROOT / "data/atlas/keeper_cuts.json"
STAGED_CUTS = ROOT / "data/atlas/keeper_cuts_v9.json"
STAGED_TGOOD = ROOT / "data/v9/t_good_derivation.json"
V8_TGOOD = ROOT / "data/v8/t_good_derivation.json"


# --------------------------------------------------------------------------- #
# The flip has NOT happened.
# --------------------------------------------------------------------------- #
def test_active_ckpt_is_still_v8():
    """The cap raise + retrain does not deploy anything. If this goes red, either the flip
    happened (and this file belongs to the flip pass, not this one) or ACTIVE_CKPT was
    edited by accident — which would silently point every discovery gate at an unmeasured
    head."""
    assert active_ckpt.ACTIVE_VERSION == "v8", (
        f"ACTIVE_CKPT names {active_ckpt.ACTIVE_VERSION!r}; the v9 build pass must NOT flip "
        f"it (the flip is conditional on the pre-registered bar and is its own pass)")


def test_live_keeper_cut_still_names_the_active_head():
    """The live cuts describe v8's p_good scale. Overwriting them with the v9 recut would
    put a v9 threshold on a v8 gate — a number about nothing, exactly as a v7 cut on a v8
    gate was."""
    doc = json.loads(LIVE_CUTS.read_text(encoding="utf-8"))
    assert doc["provenance"]["model"] == active_ckpt.ACTIVE_VERSION
    assert doc["eval"] == "data/v8/eval_scores_v8.jsonl"


def test_production_seeder_t_good_still_mirrors_v8():
    """`T_GOOD_OVERRIDES` is the ADOPTED table. It must keep mirroring v8's derivation
    while v8 is deployed; the v9 table is staged in data/v9/t_good_derivation.json and is
    mirrored by the flip pass."""
    import production_seeder as ps
    v8 = json.loads(V8_TGOOD.read_text(encoding="utf-8"))["adopted"]
    for fam, t in v8.items():
        assert ps.T_GOOD_OVERRIDES.get(fam) == t, (
            f"production_seeder.T_GOOD_OVERRIDES[{fam!r}]={ps.T_GOOD_OVERRIDES.get(fam)!r} "
            f"but v8's derived table says {t!r} — the adopted table drifted off the "
            f"deployed head's derivation")


# --------------------------------------------------------------------------- #
# The staged artifacts are stamped as staged (skipped until §7 produces them).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not STAGED_CUTS.exists(), reason="v9 keeper recut not produced yet")
def test_staged_keeper_cuts_are_v9_and_marked_staged():
    doc = json.loads(STAGED_CUTS.read_text(encoding="utf-8"))
    assert doc["provenance"]["model"] == "v9"
    assert doc["eval"] == "data/v9/eval_scores_v9.jsonl"
    assert "STAGED" in doc.get("status", "")
    assert doc.get("keeper_predicate") == "label >= 3"
    # ...and it is genuinely a different file from the live one
    assert STAGED_CUTS.resolve() != LIVE_CUTS.resolve()


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

    τ_h is a cut on the CHEAP-render p_good of a specific head, and the vendored base is
    v7-era. It is deliberately NOT re-derived here — that happens from harvest logs at hunt
    launch. What must hold is that the moment v9 is the active head, the vendored fallback
    refuses BY NAME rather than serving a stale float that looks authoritative.

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
    assert "harvest logs" in msg, "the raise should name the re-derivation route"
    # non-vacuity: the same call under the stamped head does NOT raise
    monkeypatch.setattr(sf, "_active_scorer_version", lambda: sf.TAU_H_FIDELITY_BASE_MODEL)
    assert sf.derive_tau_h(["mandelbrot"])["mandelbrot"] > 0


def test_tau_h_vendored_base_was_not_rederived_by_this_pass():
    """τ_h is left alone. The vendored base and its stamp are v7-era and must stay that
    way until a launch-time re-derivation replaces BOTH together."""
    assert sf.TAU_H_FIDELITY_BASE_MODEL == "v7"
    assert sf.TAU_H_FIDELITY_BASE, "vendored base emptied — τ_h would fail at launch"
