"""An amendment that voids a window must bind the READER, not just the file.

`allocator_prereg_v1.json` is append-only: amendment 1 (2026-08-04) withdrew the
scheduler-vs-pop-quota read as an allocator comparison. Prose alone would leave
`compare_allocator_runs.py` still printing ADOPT/KEEP/DISAGREE off the same numbers, and a
withdrawn read that the tool still scores is a withdrawn read that gets quoted.

Bracketed on both sides (verification_practice §3): the same synthetic arms score a real
verdict with no amendment and VOID with one, so the void is what changes the answer rather
than a ratio that happened to be UNKNOWN. Synthetic run dirs on purpose — the committed
ledgers are ~11 MB and this belongs in the fast lane.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "data/discovery/allocator_prereg_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_cmp_alloc", ROOT / "tools/atlas/compare_allocator_runs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cmp_alloc"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cmp_mod():
    return _load()


def _arm(d: Path, n_batches: int, adm_per_batch: float, active_min: float):
    """A run dir with just what Arm reads: harvest_log rows, summary.json, run.log."""
    d.mkdir(parents=True, exist_ok=True)
    admitted = 0
    with (d / "harvest_log.jsonl").open("w", encoding="utf-8") as fh, \
            (d / "run.log").open("w", encoding="utf-8") as lg:
        for b in range(1, n_batches + 1):
            hit = int(b * adm_per_batch) > int((b - 1) * adm_per_batch)
            fh.write(json.dumps({"batch": b, "admitted": bool(hit)}) + "\n")
            admitted += int(hit)
            lg.write(f"  batch {b}: cand=1 | 9s active={active_min * b / n_batches:.1f}m\n")
    (d / "summary.json").write_text(json.dumps({
        "totals": {"admitted": admitted}, "active_min": active_min,
        "wall_min": 520.0, "wall_budget_min": 520.0, "wall_over_active": 520.0 / active_min,
    }), encoding="utf-8")
    return admitted


@pytest.fixture
def synthetic(tmp_path):
    """Arm B admits 2x arm A per batch over the same span — an unambiguous ADOPT, so a VOID
    cannot be mistaken for the rule merely failing to reach a margin."""
    a, b = tmp_path / "armA", tmp_path / "armB"
    _arm(a, 400, 0.25, 200.0)
    _arm(b, 400, 0.50, 200.0)
    return {"A": {"run_id": "synthA", "allocator": "--scheduler", "run_dir": str(a)},
            "B": {"run_id": "synthB", "allocator": "--pop-quota", "run_dir": str(b)}}


def _doc(synthetic, amendments=None):
    d = json.loads(PREREG.read_text(encoding="utf-8"))
    d["arms"] = synthetic
    d.pop("amendments", None)
    if amendments is not None:
        d["amendments"] = amendments
    return d


AMEND = [{"n": 1, "date": "2026-08-04", "voids": ["matched_wall", "decision_rule"],
          "verdict": "VOID as an allocator comparison.",
          "what_was_wrong": "the estimand is confounded",
          "descriptive_line_NOT_A_VERDICT": "A 100, B 200 — not a result."}]


def test_without_the_amendment_the_rule_is_evaluated(cmp_mod, synthetic):
    """RED side. Strip `amendments` and the very same arms produce a scored verdict — which
    is what makes the VOID below a real gate and not a tool that was already silent."""
    out = cmp_mod.compare(_doc(synthetic))
    assert out["voided_windows"] == {}
    assert out["verdict"]["verdict"] == "ADOPT_POP_QUOTA"
    assert out["verdict"]["primary_ratio_B_over_A"] > 1.5
    assert "[VOIDED]" not in cmp_mod.render(out)


def test_a_voiding_amendment_withdraws_the_verdict(cmp_mod, synthetic):
    out = cmp_mod.compare(_doc(synthetic, AMEND))
    v = out["verdict"]
    assert v["verdict"] == "VOID"
    assert v["voided_by_amendment"] == 1
    # VOID is not DISAGREE, and it must not carry ratios: the rule never became applicable,
    # so a number computed against it has no standing even as a footnote.
    assert "primary_ratio_B_over_A" not in v and "marginal_ratio_B_over_A" not in v
    txt = cmp_mod.render(out)
    assert "VERDICT: VOID" in txt and "ADOPT" not in txt.split("VERDICT:")[1]
    assert "DESCRIPTIVE ONLY" in txt


def test_only_the_named_windows_are_stamped(cmp_mod, synthetic):
    """A void is per-window. `matched_batch` is not in AMEND's list, so it must print clean —
    otherwise the stamp is decoration rather than a reading of the file."""
    txt = cmp_mod.render(cmp_mod.compare(_doc(synthetic, AMEND)))
    assert "MATCHED WALL" in txt and "[VOIDED]" in txt.split("MATCHED WALL")[1].split("\n")[0]
    assert "[VOIDED]" not in txt.split("MATCHED BATCH INDEX")[1].split("\n")[0]
    # the numbers survive the void — a withdrawn read is still on the record, just not scored
    assert "admitted=" in txt


def test_committed_prereg_carries_the_void_and_the_reader_honours_it(cmp_mod):
    """The live file, not a fixture: amendment 1 must actually void the decision rule, and
    `voided_windows` must find it. Guards against the amendment being reworded into
    inertness by a later append."""
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    voided = cmp_mod.voided_windows(doc)
    assert "decision_rule" in voided, "amendment 1 no longer voids the decision rule"
    assert voided["decision_rule"]["n"] == 1
    ams = doc["amendments"]
    assert [a["n"] for a in ams] == list(range(1, len(ams) + 1)), "amendments must stay ordered"
    ev = ams[0]["evidence"]
    frozen = ROOT / "data/discovery/allocator_prereg_v1_mechanism_read_20260804.md"
    assert any(frozen.name in e for e in ev) and frozen.exists(), \
        "the amendment cites a frozen mechanism read that is not committed beside it"
