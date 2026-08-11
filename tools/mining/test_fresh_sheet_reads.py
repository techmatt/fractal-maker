"""Guards for the render-mode fresh-sheet reads.

The readout's job is to say what the sitting can and cannot adjudicate, so the properties
worth guarding are the ones that would let it overstate:

  * the bulk-sweep detector must FIND a sweep and must NOT invent one (§3 — every derived-set
    assertion paired with a control that fails on the unfixed input);
  * `cut_block` must go through `Floor.annotates()`, so a head-stamp mismatch REFUSES rather than
    reporting a number on a scale the live pin no longer serves;
  * `load()` must fail closed on an unlabeled row — a silently smaller n reads exactly like a
    complete sitting (§2);
  * a candidate cut must never be reported as supported on a Wilson bound that does not
    clear its target.

The committed-batch tests are the non-vacuity half: they fail if the merge is ever
un-applied, which is the one way every number above becomes a lie about a complete sitting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import floors as F                      # noqa: E402
from tools.mining import fresh_sheet_reads as FSR           # noqa: E402
from tools.mining import mining_pins as MP                  # noqa: E402


def row(order, label, suggested, p3=0.0, p2=0.0, split="train", mode="tia"):
    return {"id": f"x{order:04d}", "crop": Path("nonexistent.jpg"), "order": order,
            "label": label, "suggested": suggested, "p_ge3": p3, "p_ge2": p2,
            "pred": 1.0 + p2 + p3, "rank_score": p2 + p3, "mode": mode, "kind": "pure",
            "family": "mandelbrot", "split": split, "palette": "p", "loc": f"L{order}"}


# =========================================================================== #
# The bulk-sweep detector.
# =========================================================================== #
def test_it_finds_an_accept_all_below_sweep_at_the_tail():
    """80 individually-touched rows then a 100-row confirm-everything tail.

    The last adjudicated row (79) is a CHANGE on purpose: the tail is the pure-agreement
    suffix, so trailing confirmations inside the touched region belong to it and the boundary
    is only unambiguous when the region ends on a change."""
    rows = ([row(i, 2 if i % 5 == 4 else 1, 1) for i in range(80)]
            + [row(80 + i, 1, 1) for i in range(100)])
    sw = FSR.find_sweeps(rows)
    assert sw["tail_len"] == 100
    assert sw["adjudicated_n"] == 80
    assert sw["boundary_order"] == 80
    assert sw["changed_in_adjudicated"] == 16          # i % 5 == 4 over 0..79
    assert [r["len"] for r in sw["runs"]] == [100]


def test_the_detector_does_NOT_invent_a_sweep_when_labels_move_throughout():
    """The control. A detector that reports a tail on independently-labeled rows would make
    every sitting look bulk-confirmed, and the caveat it drives would be unfalsifiable."""
    rows = [row(i, 2 if i % 7 == 0 else 1, 1) for i in range(400)]
    sw = FSR.find_sweeps(rows)
    assert sw["runs"] == []
    assert sw["tail_len"] < FSR.MIN_SWEEP_RUN
    assert sw["adjudicated_n"] > 390                  # non-vacuity: it saw the rows


def test_a_fully_confirmed_sheet_is_all_tail_and_no_boundary():
    """The degenerate end: nothing was adjudicated, so there is no boundary order to name —
    reported as None rather than as row 0, which would read as 'one row was judged'."""
    rows = [row(i, 1, 1) for i in range(120)]
    sw = FSR.find_sweeps(rows)
    assert sw["tail_len"] == 120 and sw["adjudicated_n"] == 0
    assert sw["boundary_order"] is None


# =========================================================================== #
# The live cuts are read through the stamp check, not around it.
# =========================================================================== #
def test_cut_block_reports_the_live_pool_and_release_points():
    rows = [row(i, 3 if i < 10 else 1, 1, p3=0.9 if i < 12 else 0.0) for i in range(40)]
    cb = FSR.cut_block(rows, F.MINING_RELEASE)
    assert cb["fires"] == 12 and cb["tp"] == 10
    assert cb["precision"] == pytest.approx(10 / 12)
    assert cb["recall"] == pytest.approx(1.0)
    assert cb["precision_lo"] < cb["precision"] < cb["precision_hi"]


@pytest.mark.stage2_pinned
def test_a_head_stamp_mismatch_REFUSES_instead_of_reporting_a_number(monkeypatch):
    """The control for the whole read: 0.25 and 0.50 are points on v1's probability scale.
    If the mining pin ever moves, a readout that quietly kept comparing would annotate the
    pool against a head that no longer exists."""
    monkeypatch.setattr(MP, "HEAD_VERSION", "v2")
    with pytest.raises(F.HeadStampMismatch):
        FSR.cut_block([row(0, 3, 3, p3=0.9)], F.MINING_POOL)


@pytest.mark.stage2_pinned
def test_the_sweep_grid_contains_both_live_cuts_exactly():
    """Marked as exact rows, never nearest-bin: a ladder that reported the pool cut at 0.25
    while sweeping 0.2/0.3 would attribute a precision to a threshold nobody set."""
    assert F.MINING_POOL.value in FSR.SWEEP
    assert F.MINING_RELEASE.value in FSR.SWEEP
    lad = FSR.ladder([row(i, 1, 1, p3=i / 100) for i in range(100)])
    marked = {r["threshold"]: r["marks"] for r in lad if r["marks"]}
    # READ from the owner, never restated: these two values are volume-matched at every head
    # flip (protocol section 5a), and a literal here would go red FOR the flip rather than
    # for a fault -- which is the whole reason `version_pinned` exists.
    assert marked == {F.MINING_POOL.value: ["mining_pool"],
                      F.MINING_RELEASE.value: ["mining_release"]}


# =========================================================================== #
# Ladders and candidate cuts.
# =========================================================================== #
def test_recall_is_non_increasing_and_precision_is_of_passers():
    rows = [row(i, 3 if i >= 90 else 1, 1, p3=i / 100) for i in range(100)]
    lad = FSR.ladder(rows)
    rec = [r["recall"] for r in lad]
    assert rec == sorted(rec, reverse=True)
    top = [r for r in lad if r["threshold"] == 0.9][0]
    assert top["fires"] == 10 and top["tp"] == 10 and top["precision"] == pytest.approx(1.0)


def test_a_candidate_is_the_LOWEST_threshold_reaching_the_target():
    rows = [row(i, 3 if i >= 50 else 1, 1, p3=i / 100) for i in range(100)]
    c = FSR.candidates(FSR.ladder(rows), targets=(0.70,))["0.70"]
    assert c["threshold"] == pytest.approx(0.30)     # 50 pos / 70 passers = 0.714
    assert c["precision"] == pytest.approx(50 / 70)


def test_a_target_met_only_on_the_point_estimate_is_reported_UNSUPPORTED():
    """3 passers, 3 true — precision 1.000, Wilson lower bound ~0.44. `supported` is what
    stops a 3-row cell from being quoted as a 90%-precision cut."""
    rows = [row(i, 3 if i >= 95 else 1, 1, p3=i / 100) for i in range(98)]
    c = FSR.candidates(FSR.ladder(rows), targets=(0.90,))["0.90"]
    assert c["precision"] == pytest.approx(1.0)
    assert c["supported"] is False and c["fires"] <= 5


def test_an_unreachable_target_is_None_not_the_best_available():
    rows = [row(i, 1, 1, p3=i / 100) for i in range(100)]     # no positives at all
    assert FSR.candidates(FSR.ladder(rows), targets=(0.70,))["0.70"] is None


# =========================================================================== #
# Fail-closed load, and the committed batch itself.
# =========================================================================== #
def test_load_refuses_a_batch_with_an_unlabeled_row(tmp_path):
    src = FSR.SPEC.batch_dir / "images.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()
    doc = json.loads(lines[0])
    doc["label"]["score"] = None
    (tmp_path / "images.jsonl").write_text("\n".join([json.dumps(doc)] + lines[1:]),
                                           encoding="utf-8")
    with pytest.raises(SystemExit, match="label.score = null"):
        FSR.load(tmp_path)


def test_the_committed_batch_is_fully_merged_and_matches_its_manifest():
    """Non-vacuity for every read above: they describe a COMPLETE sitting or they describe
    nothing. This is also the merge's own row-count verification, kept in the suite."""
    manifest = json.loads((FSR.SPEC.batch_dir / "batch.json").read_text(encoding="utf-8"))
    rows = FSR.load()
    assert len(rows) == manifest["n_rows"] == 960
    assert {r["mode"] for r in rows} == set(FSR.MODES)
    assert all(1 <= r["label"] <= FSR.K_TIERS for r in rows)


def test_the_swept_tail_reaches_exactly_the_cuts_that_sit_below_it():
    """Read E's ladders are invariant to the bulk-accept sweep for a cut the swept tail never
    reaches, and are NOT for one it does. The test asserts the boundary rather than a fixed
    answer, because the answer moved.

    Until 2026-08-11 both live cuts (0.25/0.3402 pool, 0.50/0.6691 release) sat above the
    tail's ceiling and the whole read was invariant. The base-rate audit dropped the pool floor
    to 0.0, so all 295 swept rows now clear it and the POOL row of every ladder is a statement
    about ~665 adjudicated decisions dressed as 960 — precision 0.154 on the full sheet against
    0.223 on the adjudicated one. The release cut is unaffected: the tail's highest `p_ge3` is
    0.032, well under the 0.0949 crossover.

    So the invariance claim is now per-cut, and the module already says so in its own
    `verdict`. What this pins is that the verdict TRACKS the cuts instead of being a sentence
    somebody wrote once."""
    E = FSR.build_E(FSR.load())
    both, ts = E["slices"], E["tail_sensitivity"]
    ceiling = E["swept_tail"]["max_p_ge3"]
    for name, cut in (("pool", F.MINING_POOL), ("release", F.MINING_RELEASE)):
        reached = cut.value <= ceiling
        for k in ("precision", "recall"):
            same = both["all_960"][name][k] == both["adjudicated"][name][k]
            # A cut the tail reaches MAY still read the same (the tail could be unanimous);
            # a cut it does not reach MUST. Only the second direction is an invariant.
            assert same or reached, (name, k, cut.value, ceiling)
    assert (ts["pool_fires_inside_tail"] > 0) == (F.MINING_POOL.value <= ceiling)
    assert ("NOT invariant" in ts["verdict"]) == (ts["pool_fires_inside_tail"] > 0)
    assert both["adjudicated"]["n"] < both["all_960"]["n"]      # the slices really differ
