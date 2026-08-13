"""`autolevel_read.py`, promoted out of `scratch/production_run26/` for run 27.

Covers the two things a report quotes and that are easy to get silently wrong:

  * THE RELEASED SPLIT. Identity share over ALL scored rows and over the rows that SHIPPED
    are different populations and answered differently in run 26 (28/48 scored vs 8/12
    released) — the selector prefers already-in-band renders, so quoting the scored share as
    "what ships" is wrong in a knowable direction. Run 26's own report published 9/12 for
    that split against the record's 8/12, which is exactly the slip an assertion catches.
  * THE ACTED/IDENTITY DISAGREEMENT COUNT. `acted` and `curve.identity` are written by the
    same operator and must be complements; the reader counts rows where they AGREE, because
    that is the bug condition. A test that only fed it consistent rows would pass on a
    reader that had the polarity backwards, so an inconsistent row is fed deliberately.

Light lane: builds a tiny record on disk, no engine, no GPU.

  uv run pytest tools/emission/test_autolevel_read.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "emission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import autolevel_read as ar  # noqa: E402


def _row(run_id, style, identity, decision, *, acted=None, capped=0, stops=100):
    return {
        "run_id": run_id, "stage": "release", "render_style": style, "decision": decision,
        "schema_version": 3, "slot_source": "mix" if decision == "selected" else None,
        "autolevel": {
            "switch": "on", "operator": "band_v1",
            "acted": (not identity) if acted is None else acted,
            "curve": {"identity": identity, "sides": {} if identity else {"mid": 1}},
            "chroma_cap": {"n_capped": capped, "n_stops": stops, "retain": 0.85},
            "reference": {"sha256": "abc"},
        },
    }


@pytest.fixture
def record(tmp_path):
    rows = [
        # 4 selected: 3 identity, 1 acting.   4 not_selected: 1 identity, 3 acting.
        _row("r27", "smooth", True, "selected"),
        _row("r27", "smooth", True, "selected"),
        _row("r27", "tia", True, "selected"),
        _row("r27", "tia", False, "selected", capped=40),
        _row("r27", "smooth", True, "not_selected"),
        _row("r27", "smooth", False, "not_selected", capped=5),
        _row("r27", "tia", False, "not_selected"),
        _row("r27", "tia", False, "not_selected", capped=90),
        # another run's rows, and this run's gate mirror — neither may be counted
        _row("r26", "smooth", False, "selected"),
        {**_row("r27", "smooth", True, "selected"), "stage": "gate"},
    ]
    p = tmp_path / "rec.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_released_share_is_a_separate_population_from_the_scored_share(record, tmp_path):
    rep = ar.read_report(record, "r27", tmp_path / "absent.jsonl")
    assert rep["n_scored_rows"] == 8                    # gate mirror and r26 excluded
    assert rep["identity_overall"] == {"n": 8, "identity": 4, "share": 0.5}
    assert rep["identity_released"] == {"n": 4, "identity": 3, "share": 0.75}


def test_by_style_and_chroma_cap_are_over_acting_rows_only(record, tmp_path):
    rep = ar.read_report(record, "r27", tmp_path / "absent.jsonl")
    c = rep["chroma_cap"]
    assert c["n_acting"] == 4                           # the 4 non-identity rows
    assert c["n_rows_capped"] == 3                      # one acting row capped 0 stops
    assert c["max_capped_stops"] == 90
    assert rep["render_cost"]["expected_multiplier_on_colorize"] == pytest.approx(1.5)


def test_an_acted_identity_disagreement_is_counted(tmp_path):
    """`acted=True` on an identity row is a stamp bug. The reader must count rows where the
    two AGREE — a reader with the polarity flipped would report 0 here and 8 on the fixture
    above, and both are silently plausible."""
    rows = [_row("r27", "smooth", True, "selected", acted=True),      # inconsistent
            _row("r27", "smooth", True, "selected")]                  # consistent
    p = tmp_path / "rec.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rep = ar.read_report(p, "r27", tmp_path / "absent.jsonl")
    assert rep["acted_identity_disagreements"] == 1


def test_absent_run_id_raises_and_names_what_is_there(record, tmp_path):
    """An empty population must not read as a run with no acting rows."""
    with pytest.raises(SystemExit, match="r27"):
        ar.read_report(record, "prod99", tmp_path / "absent.jsonl")
