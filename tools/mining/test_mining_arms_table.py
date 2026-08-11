"""The (28)/(28b) arms table — the two things it can get wrong without anyone noticing.

The table is a read-only renderer over committed run records, so there is nothing to test
about its arithmetic. What there IS to test is the pair of couplings that would let it
describe the wrong experiment:

  1. an arm's DECLARED dial disagreeing with the run record it renders (the table says
     `ap_ge3`, the record says the run selected on `ap_ge2`), and
  2. the arm list drifting from the trainer's own objective vocabulary.

Both are silent failures — the table renders happily either way — so both are asserted here
rather than left to a reader noticing a wrong column.

  uv run pytest tools/mining/test_mining_arms_table.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.train_mining_head_v3 import SELECTION_METRICS          # noqa: E402
from tools.mining import mining_arms_table as AT                       # noqa: E402


def test_every_arm_declares_a_real_objective():
    for a in AT.ARMS:
        assert a.objective in SELECTION_METRICS, (
            f"{a.key} declares objective {a.objective!r}, which the trainer cannot run")


def test_exactly_one_arm_per_dial_combination():
    """The arms discipline: each arm changes exactly ONE thing against `dedup_weighted`."""
    base = AT.ARMS[0]
    assert base.key == "dedup_weighted"
    for a in AT.ARMS[1:]:
        moved = sum(getattr(a, f) != getattr(base, f)
                    for f in ("geometry", "weights", "objective"))
        assert moved == 1, (f"{a.key} moves {moved} dials against the base arm — the arms "
                            f"discipline is one variable each")


def test_the_fifth_arm_is_v1s_objective_and_nothing_else():
    arm = next(a for a in AT.ARMS if a.key == "ap2_selected")
    base = AT.ARMS[0]
    assert arm.objective == "ap_ge2"
    assert (arm.geometry, arm.weights) == (base.geometry, base.weights)


@pytest.mark.parametrize("arm", AT.ARMS, ids=lambda a: a.key)
def test_declared_objective_matches_the_run_record(arm):
    """The coupling the table's own assert enforces at render time, checked per arm so a
    mismatch names WHICH arm instead of failing the whole build."""
    r = AT.load(arm)
    if r is None:
        pytest.skip(f"{arm.key} has not been run")
    stamped = (r["metrics"].get("selection") or {}).get("metric")
    if stamped is None:
        pytest.skip(f"{arm.key} predates the selection stamp")
    assert stamped == arm.objective


def test_build_renders_from_whatever_arms_exist():
    rows = [r for r in (AT.load(a) for a in AT.ARMS) if r]
    if not rows:
        pytest.skip("no arm has been run")
    md = AT.build(rows)
    assert "pooled ≥2 boundary" in md
    for r in rows:
        assert f"`{r['arm'].key}`" in md
    missing = [a.key for a in AT.ARMS if not any(x["arm"].key == a.key for x in rows)]
    # an unrun arm must be NAMED as unrun, not silently absent
    for k in missing:
        assert k in md
