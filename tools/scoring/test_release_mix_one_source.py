"""`release_mix.RATIO` is the ONLY source of a per-partition release share.

Two consumers read it at very different scales — the discovery order book
(`atlas/deficit_scheduler`, denominated in distinct looks) and the emission target measure
(`emission/cells`, denominated in joint cells) — and until 2026-08-04 they read two different
things: the scheduler projected `data/emission/target_measure.json` down to per-partition
marginals, emission solved the same file's nine hand-placed multipliers plus a `target_share`
override, and a fourth site (`library_intake_2`) carried its own `CLASSIC_RELEASE_SHARE = 0.02`
literal. Those numbers said the same KIND of thing as the ratio table and disagreed with it
(mandelbrot 9.0% vs 22.7% intended).

So this file asserts the two properties that keep them one policy:
  1. both consumers resolve IDENTICAL partition shares from one input, and
  2. no tracked source reads a second target source (a measure file, or its own share literal).

  uv run pytest tools/scoring/test_release_mix_one_source.py -q
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "scoring", ROOT / "tools" / "atlas"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import partitions as P                              # noqa: E402
import release_mix as RM                            # noqa: E402
import deficit_scheduler as DS                      # noqa: E402
from tools.emission import cells as C               # noqa: E402


def _feasible(parts, clusters_per_part):
    """Feasible cells with a DELIBERATELY lopsided cluster count per partition — the axis the
    two consumers used to disagree on."""
    obs = [(p, f"{p}#{k}") for p in parts for k in range(clusters_per_part[p])]
    return C.build_feasible_cells(obs, ["k16:1", "k16:5"], ["smooth", "tia"])


def test_both_consumers_resolve_identical_partition_shares():
    """One input (the ratio table over a partition list) -> one answer, on both sides."""
    parts = ["mandelbrot", "julia:mandelbrot", "multibrot4", "phoenix", "phoenix:classic"]
    counts = {"mandelbrot": 102, "julia:mandelbrot": 4, "multibrot4": 1,
              "phoenix": 196, "phoenix:classic": 41}
    order_book = DS.target_shares(parts)                       # discovery side
    measure = C.TargetMeasure.from_partition_shares(           # emission side
        RM.shares(parts), _feasible(parts, counts))
    assert measure.partition_shares() == pytest.approx(order_book)
    assert sum(order_book.values()) == pytest.approx(1.0)
    # ...and both are the ratio table, not merely each other.
    assert order_book == pytest.approx(RM.shares(parts))


def test_the_agreement_is_not_an_artifact_of_equal_cluster_counts():
    """Non-vacuity. The counts above are lopsided on purpose; here they are moved by 100x and
    the shares must not budge, which is the property that broke when the emission side summed
    cell weights instead of dividing the count out."""
    parts = ["mandelbrot", "julia:mandelbrot"]
    a = C.TargetMeasure.from_partition_shares(
        RM.shares(parts), _feasible(parts, {"mandelbrot": 1, "julia:mandelbrot": 1}))
    b = C.TargetMeasure.from_partition_shares(
        RM.shares(parts), _feasible(parts, {"mandelbrot": 300, "julia:mandelbrot": 3}))
    assert a.partition_shares() == pytest.approx(b.partition_shares())
    assert b.partition_shares() == pytest.approx(DS.target_shares(parts))


def test_classic_phoenix_carries_its_own_ratio_end_to_end():
    """The share that used to be hand-placed in two files: it now comes from the one ratio and
    lands on cells that are addressable as `phoenix:classic`."""
    parts = ["phoenix", "phoenix:classic"]
    feasible = _feasible(parts, {"phoenix": 196, "phoenix:classic": 41})
    tm = C.TargetMeasure.from_partition_shares(RM.shares(parts), feasible)
    expect = RM.ratio_of("phoenix:classic") / (RM.ratio_of("phoenix")
                                               + RM.ratio_of("phoenix:classic"))
    assert tm.partition_shares()["phoenix:classic"] == pytest.approx(expect)
    assert P.CLASSIC_PHOENIX in tm.weights_by_partition


def test_a_ratio_edit_moves_both_consumers_together(monkeypatch):
    """Injection: change the policy in its one home and BOTH sides move. If either had kept a
    private copy this would move one of them."""
    parts = ["mandelbrot", "multibrot4"]
    before = DS.target_shares(parts)
    monkeypatch.setitem(RM.RATIO, "multibrot4", 9.0)
    after = DS.target_shares(parts)
    measure = C.TargetMeasure.from_partition_shares(
        RM.shares(parts), _feasible(parts, {"mandelbrot": 7, "multibrot4": 2}))
    assert after["multibrot4"] > before["multibrot4"]
    assert measure.partition_shares() == pytest.approx(after)


# --------------------------------------------------------------------------- #
# 2. No second target source.
# --------------------------------------------------------------------------- #
OWN = {"tools/scoring/test_release_mix_one_source.py",
       "tools/audit/durability_map.py"}          # the durability map RECORDS the deletion
# The deleted measure file, and the shape of a hand-placed share literal that used to sit
# beside it. Both are quoted here, so this file exempts itself.
#
# The measure scan hunts a READ of the file, not a mention of it: prose recording the
# retirement (`cells.py`'s docstring, `durability_map`'s row) is the point of that prose, and a
# scan that forbids naming a retired thing forbids explaining why it is retired. A read is a
# path expression or a CLI default, so the line must also carry one of those tokens.
MEASURE_READ = re.compile(
    r"^(?!\s*#).*(ROOT\s*/|Path\(|open\(|read_text|add_argument).*target_measure\.json", re.M)
SHARE_LITERAL = re.compile(r"^\s*(CLASSIC_RELEASE_SHARE|RELEASE_SHARE)\s*(:[^=]*)?=", re.M)


def _tracked_python():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [p for p in out.stdout.splitlines() if p.strip()]


def test_the_deleted_measure_file_is_gone_and_unreferenced():
    """Deleted rather than kept as a derived artifact: its remaining content was two mechanism
    knobs that are code defaults. Absence must fail LOUD at a missed reader, and the way it
    fails loud is that there is no reader — asserted, because an `if path.exists() else {}`
    reader (which `first_release_readout` had) turns absence into a silently uniform target."""
    assert not (ROOT / "data" / "emission" / "target_measure.json").exists()
    offenders = []
    for rel in _tracked_python():
        norm = rel.replace("\\", "/")
        if norm in OWN:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in MEASURE_READ.finditer(text):
            offenders.append(f"{norm}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        f"{len(offenders)} reference(s) to the deleted target measure: {offenders}. The "
        f"emission target derives from release_mix.RATIO at intake.")


def test_no_module_carries_its_own_release_share_literal():
    offenders = []
    for rel in _tracked_python():
        norm = rel.replace("\\", "/")
        if norm in OWN:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in SHARE_LITERAL.finditer(text):
            offenders.append(f"{norm}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        f"{len(offenders)} module-level release-share literal(s) outside the ratio table: "
        f"{offenders}. `library_intake_2` carried CLASSIC_RELEASE_SHARE = 0.02 beside the "
        f"measure file's own 0.02 — two copies of one policy.")


def test_the_scan_would_catch_a_copy():
    """Non-vacuity for both scans."""
    assert SHARE_LITERAL.search("CLASSIC_RELEASE_SHARE = 0.02   # the classic target")
    assert SHARE_LITERAL.search("RELEASE_SHARE: float = 0.02")
    assert not SHARE_LITERAL.search("from release_mix import shares")
    assert MEASURE_READ.search('P = ROOT / "data" / "emission" / "target_measure.json"')
    assert MEASURE_READ.search('ap.add_argument("--target-measure", default=str(target_measure.json))')
    # ...and does not fire on prose that names the retired file
    assert not MEASURE_READ.search("# it used to read data/emission/target_measure.json")
    assert not MEASURE_READ.search("It used to be a hand-edited `target_measure.json` carrying")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
