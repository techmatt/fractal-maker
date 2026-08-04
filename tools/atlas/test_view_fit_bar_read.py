"""Acceptance for the pre-registered bar read (`tools/atlas/view_fit_bar_read.py`).

Three guards, failing differently:

  * the MARGIN is not a literal anyone can edit — it must still equal the number the
    run pre-registered before batch 1. A bar restated after the readout is not a bar.
  * the READ is an outcome, so it is pinned against the frozen record
    `data/atlas/view_fit_v1_1_bar_read.json`, INCLUDING the fact that it came back
    NOT MET and uninformative. A negative result that can be quietly re-run into a
    positive one is not a record.
  * the QUALIFIER is behavioural, not prose: `attainable_range` is asserted on
    synthetic label vectors, and a verdict is asserted never to be emitted without
    the attainable-ratio beside it. That is what stops "NOT MET" from being read as
    "view_fit lost" on a slice where no ordering could have won.

Run:  uv run python -m pytest tools/atlas/test_view_fit_bar_read.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                      # noqa: E402
import view_fit_bar_read as bar   # noqa: E402


def _record():
    p = paths.durable(bar.RECORD_REL)
    if not p.exists():
        pytest.skip(f"{bar.RECORD_REL} not frozen yet")
    return json.loads(p.read_text(encoding="utf-8"))


def test_margin_still_equals_the_pre_registration():
    """The module's MARGIN is the run's pre-registered one, read from the run's own object."""
    import steered_frontier as sf
    assert sf.SteeredFrontier.PREREG["view_fit_v1_1_vs_composite_v3"]["margin"] == bar.MARGIN


def test_attainable_range_shrinks_as_the_slice_saturates():
    """Behavioural: AP's whole range narrows toward 0 as one class empties out. This is why a
    fixed delta-AP margin cannot be carried from one base rate to another."""
    balanced = bar.attainable_range(np.array([1] * 50 + [0] * 50))
    lopsided = bar.attainable_range(np.array([1] * 95 + [0] * 5))
    degenerate = bar.attainable_range(np.array([1] * 100))
    assert balanced > lopsided > 0.0
    assert degenerate == 0.0


def test_the_frozen_read_is_the_read_this_code_takes():
    """Re-running the read reproduces the frozen record's numbers. The bootstrap is seeded
    (view_fit.FIT_SEED) so this is an equality, not a tolerance."""
    rec = _record()
    got = bar.read(rec["batch"])
    for k in ("target", "n_pos", "n_neg", "base_rate", "ap", "bootstrap", "verdict"):
        assert got[k] == rec[k], f"{k} drifted: {got[k]!r} != frozen {rec[k]!r}"
    assert got["slice"]["n"] == rec["slice"]["n"] == 268


def test_the_slice_agrees_with_what_the_cut_stamped():
    """The read and the sitting cut must not disagree about which rows are bar-readable —
    the read uses `sitting_cutter.is_bar_readable`, and the cut stamped its own count."""
    rec = _record()
    stamped = json.loads(
        paths.durable(f"data/label_corpus/batches/{rec['batch']}/batch.json")
        .read_text(encoding="utf-8"))["bar_readability"]
    assert rec["slice"]["n"] == stamped["n"]
    assert rec["slice"]["by_partition"] == stamped["by_partition"]


def test_verdict_is_never_recorded_without_what_it_is_worth():
    """A NOT MET on a saturated slice is uninformative, so the record carries the attainable
    range and the ratio beside the verdict. Guarding the PAIR is the point: dropping the
    qualifier would leave a bare 'NOT MET' to be read as evidence against view_fit."""
    rec = _record()
    assert rec["verdict"] in ("MET", "NOT MET")
    assert rec["attainable"]["max_abs_delta_ap"] > 0
    assert rec["attainable"]["bar_attainable_ratio"] == pytest.approx(
        bar.MARGIN / rec["attainable"]["max_abs_delta_ap"], abs=1e-4)
    if rec["attainable"]["bar_attainable_ratio"] > 0.5:
        assert "UNINFORMATIVE" in rec["verdict_qualifier"]


def test_the_read_is_scoped_to_the_population_it_ran_on():
    """The slice is entirely native-plane; the record must say so rather than leave the
    verdict sounding pipeline-wide."""
    rec = _record()
    assert not any(f.startswith("julia") or f == "phoenix" for f in rec["slice"]["by_partition"])
    assert "MANEUVER" in rec["population_caveat"]


def test_the_reader_changes_no_order():
    """The staging contract, derived from the source: this module does not import or touch a
    live sort key, and `composite_v3` remains what the sourcing order calls."""
    src = (HERE / "view_fit_bar_read.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "SORT_KEY" not in body and "composite_v3 =" not in body
