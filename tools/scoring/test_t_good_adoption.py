#!/usr/bin/env python
"""The t_good derivation ⟷ production table agreement gate, for whatever version is LIVE.

`data/<ACTIVE_VERSION>/t_good_derivation.json` is the DERIVATION and
`production_seeder.T_GOOD_OVERRIDES` is the ADOPTED copy. Two copies of the same numbers
drift; this holds them equal. It also pins the thing the numbers alone cannot say: which
partitions are UNCALIBRATED (running at the baseline without ever having been derived) vs
DERIVED-at-0.50, because in a config file those are the same character sequence.

WAS `tools/v8/test_derive_t_good_v8.py`, which hardcoded v8 in three places — the artifact
path, the model stamp and the eval-slice name. That is exactly what the pins module exists to
prevent, and it showed at the v10 flip: the gate would have gone red *for the flip* rather
than for a drift. It now resolves the version from `production_pins.ACTIVE_VERSION`, so it
follows the pin instead of racing it — and a flip that forgets to re-derive t_good is
precisely what it now catches.

Runs with no model and no GPU — it reads two committed artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools" / "atlas", ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    sys.path.insert(0, str(_p))

import production_seeder as ps  # noqa: E402
import derive_t_good as est  # noqa: E402  the shared estimator (fbeta, build_table)
from partitions import ALL_FAMS  # noqa: E402
from production_pins import ACTIVE_VERSION  # noqa: E402

# Coupled to production_pins.ACTIVE_CKPT: `pytest -m version_pinned` lists it.
pytestmark = pytest.mark.version_pinned


DERIVATION = ROOT / "data" / ACTIVE_VERSION / "t_good_derivation.json"


def _doc():
    assert DERIVATION.exists(), (
        f"{DERIVATION} missing — the ACTIVE checkpoint is {ACTIVE_VERSION} but no t_good "
        f"derivation exists for it. t_good is scale-bound (protocol §4): re-derive before "
        f"serving, do not carry the previous version's table.")
    return json.loads(DERIVATION.read_text(encoding="utf-8"))


def test_adopted_table_matches_the_derivation_artifact():
    doc = _doc()
    assert doc["adopted"] == {k: float(v) for k, v in ps.T_GOOD_OVERRIDES.items()}, (
        f"production_seeder.T_GOOD_OVERRIDES {ps.T_GOOD_OVERRIDES} != derivation adopted "
        f"{doc['adopted']} — re-run the {ACTIVE_VERSION} derivation and mirror the table")


def test_uncalibrated_partitions_are_named_not_merely_absent():
    doc = _doc()
    assert set(doc["uncalibrated"]) == set(ps.T_GOOD_UNCALIBRATED), (
        f"derivation says UNCALIBRATED={sorted(doc['uncalibrated'])} but "
        f"T_GOOD_UNCALIBRATED={sorted(ps.T_GOOD_UNCALIBRATED)}")
    # every uncalibrated partition runs at the baseline, and says so
    for part, row in doc["uncalibrated"].items():
        assert row["status"] == "UNCALIBRATED", part
        assert float(row["t_good"]) == ps.T_GOOD_BASELINE, part
        assert ps.t_good_for(part) == ps.T_GOOD_BASELINE, part
        assert ps.t_good_status(part) == "UNCALIBRATED", part
        # "we looked and found no keepers" and "we have never looked" are different states
        # and must not reduce to the same string (v10 is the first version carrying both).
        assert row.get("reason"), part


def test_every_live_partition_is_classified_either_way():
    # No partition may be silently UNKNOWN: a family that reaches production without being
    # either derived or explicitly stamped uncalibrated is the failure this guards.
    for part in ALL_FAMS:
        assert ps.t_good_status(part) in ("DERIVED", "UNCALIBRATED"), part
    assert not (set(ps.T_GOOD_OVERRIDES) & set(ps.T_GOOD_UNCALIBRATED)), "a partition is both"


def test_derived_partitions_carry_the_supply_weighted_objective():
    # The split objective is the point of the derivation: precision where supply is
    # abundant (mandelbrot), recall where it saturates (julia:multibrot). If someone
    # re-uniformizes the objective, this fails.
    doc = _doc()
    assert doc["derived"]["mandelbrot"]["objective"] == "F0.5"
    for d in (3, 4, 5):
        assert doc["derived"][f"julia:multibrot{d}"]["objective"] == "F2"
    # and both objectives stay REPORTED for each, so the choice remains auditable
    for part, row in doc["derived"].items():
        assert {"F0.5", "F2"} <= set(row["by_objective"]), part
        for blk in row["by_objective"].values():
            assert "fbeta" in blk and "fbeta_oof" in blk and "plateau" in blk, part


def test_class4_has_no_per_family_threshold():
    # The head is q4-capable; the derivation deliberately does NOT calibrate the third cutpoint.
    doc = _doc()
    assert doc["no_class4_threshold"] is True
    for row in doc["derived"].values():
        assert "t_great" not in row and "t_class4" not in row


def test_fbeta_matches_the_closed_form():
    p, r = 0.4, 0.9
    for beta in (0.5, 1.0, 2.0):
        b2 = beta * beta
        assert abs(est.fbeta(p, r, beta) - (1 + b2) * p * r / (b2 * p + r)) < 1e-12
    assert est.fbeta(0.0, 0.0, 2.0) == 0.0


def test_derivation_is_stamped_with_the_model_it_came_from():
    doc = _doc()
    assert doc["model"] == ACTIVE_VERSION, (
        f"the live t_good derivation is stamped {doc['model']!r} but the active checkpoint is "
        f"{ACTIVE_VERSION!r} — a threshold from another head is a number about nothing")
    assert doc["eval_slice"] == f"data/{ACTIVE_VERSION}/eval_scores_{ACTIVE_VERSION}.jsonl"


def test_the_derivation_reruns_to_the_committed_numbers():
    """Re-derivation drift gate: the committed artifact must be what the code produces from
    the committed slice. A hand-edited threshold, a changed objective, a changed population
    rule or a changed keeper predicate all surface here instead of silently shipping.

    Skips (does not fail) if the live version has no re-runnable deriver module — the stamp
    checks above still apply, and a missing deriver is a build-side gap, not a drift."""
    import importlib

    import pytest

    modname = f"derive_t_good_{ACTIVE_VERSION}"
    sys.path.insert(0, str(ROOT / "tools" / ACTIVE_VERSION))
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError:
        pytest.skip(f"no {modname} module — nothing to re-run")
    if not mod.EVAL.exists():
        pytest.skip(f"{mod.EVAL} absent")
    rows = [json.loads(l) for l in mod.EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, _ = mod.select_population(rows)
    # Every knob the deriver declares is passed through. `WITHHOLD` is optional (v8/v10 have
    # none) but it is NOT ignorable: a re-run that dropped it would re-adopt exactly the
    # partitions the pass deliberately withheld and then report the difference as drift.
    fresh = est.build_table(kept, version=mod.VERSION, eval_slice=mod.EVAL_REL,
                            objective=mod.OBJECTIVE, uncal_reason=mod.UNCAL_REASON,
                            withhold=getattr(mod, "WITHHOLD", None))
    assert fresh["adopted"] == _doc()["adopted"], (
        f"re-derived {fresh['adopted']} != committed {_doc()['adopted']} — "
        f"re-run tools/{ACTIVE_VERSION}/{modname}.py")
