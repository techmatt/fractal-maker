#!/usr/bin/env python
"""The release-candidate sheet's derivations — fate assignment, the realized-vs-target read,
and the gate-report accrual — on synthetic runs.

The page itself is eyeball output and is not asserted here. What IS asserted is everything
the eye cannot check: that a tile's stratum is the run's OWN decision (read off the release
record, not re-derived from the floors), that the floor-admit rows stay identifiable, and that
the two gate-report sites are counted separately. A sheet that mislabels a reject as an
admission is worse than no sheet — it is a wrong caption under a real picture.

  uv run pytest tools/emission/test_release_candidate_sheet.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import descriptor as D                   # noqa: E402
from tools.emission import emission_sinks as ESINKS          # noqa: E402
from tools.emission import release_candidate_sheet as RCS    # noqa: E402


def prow(rid, style="smooth", p=0.95, passed=True, err=None, loc=None, typ="mandelbrot",
         ledger="data/discovery/campaign1/breadth/outcome_ledger.jsonl"):
    return {"id": rid, "location_id": loc or f"L_{rid}", "type": typ, "morph_cluster": f"{typ}#0",
            "palette_flavor": "cool", "render_style": style, "palette": "pal",
            "cell": [typ, f"{typ}#0", "cool", style],
            "head": "wallpaper" if style == "smooth" else "mining",
            "p_ge3": p, "passed": passed, "error": err, "floor": 0.75,
            "jpg": None, "provenance": {"source_ledger": ledger}}


# --------------------------------------------------------------------------- #
# fate assignment
# --------------------------------------------------------------------------- #
def test_every_row_gets_exactly_one_fate_and_the_fates_partition_the_pool():
    rows = [prow("a"), prow("b"), prow("c", passed=False, p=0.4),
            prow("d", p=None, passed=False, err="boom"), prow("e", p=0.80)]
    fates = RCS.assign_fates(rows, {"a"}, {"a", "b"}, lambda s: 0.90)
    assert set(fates) == {r["id"] for r in rows}
    assert set(fates.values()) <= {k for k, _ in RCS.FATES}
    assert fates == {"a": "selected", "b": "eligible_not_selected",
                     "c": "below_pool_floor", "d": "render_error",
                     "e": "pooled_below_release_floor"}


def test_fate_follows_the_release_RECORD_not_a_re_derivation_from_the_floors():
    """The property that keeps a caption honest. A strange row can be release-eligible while
    sitting BELOW the mining release floor — that was every strange row during the report-only
    period, and those runs' records are still read by this sheet — so a sheet that decided
    eligibility from TODAY's floors would file a genuinely-selected row under 'rejected'. The
    floor going enforcing on 2026-08-06 makes this property MORE load-bearing, not less: it is
    now the only thing standing between an old record and a re-derivation that contradicts
    it."""
    ungated_strange = prow("s", style="tia", p=0.31)          # below the 0.50 mining floor
    fates = RCS.assign_fates([ungated_strange], {"s"}, {"s"}, lambda st: 0.50)
    assert fates["s"] == "selected"
    # ...and with the record saying it was passed over, it is still an ADMISSION-side stratum.
    assert RCS.assign_fates([ungated_strange], set(), {"s"}, lambda st: 0.50)["s"] \
        == "eligible_not_selected"


def test_the_floor_admit_source_stays_identifiable():
    """`q4_harvest` is the one FLOOR_ADMIT source; the sheet's whole comparison depends on the
    tag surviving from the intake snapshot to the caption."""
    r = prow("q", ledger="data/emission/q4_harvest/outcome_ledger.jsonl")
    ledger, tag = RCS.source_of(r, {r["location_id"]: "q4_harvest"})
    assert ledger == "q4_harvest" and tag == "q4_harvest"
    assert tag in D.FLOOR_ADMIT_SOURCES                       # non-vacuous: it IS a floor-admit
    html = RCS.tile(r, "scratch/x.jpg", "selected", ledger, tag, "(floors)")
    assert 'class="q4"' in html and "FLOOR-ADMIT" in html
    # an ordinary discovery row must NOT be marked, or the mark says nothing.
    o = prow("o")
    l2, t2 = RCS.source_of(o, {})
    assert 'class="q4"' not in RCS.tile(o, "scratch/y.jpg", "selected", l2, t2, "(floors)")


# --------------------------------------------------------------------------- #
# realized vs target
# --------------------------------------------------------------------------- #
def test_realized_vs_target_reports_both_columns_and_invents_no_verdict():
    sel = [prow("a", typ="mandelbrot"), prow("b", typ="mandelbrot"), prow("c", typ="phoenix")]
    out = RCS.realized_vs_target(sel, {"mandelbrot": 0.5, "phoenix": 0.3, "multibrot3": 0.2})
    by = {r["partition"]: r for r in out}
    assert by["mandelbrot"]["selected"] == 2
    assert abs(by["mandelbrot"]["realized_share"] - 2 / 3) < 1e-12
    assert by["multibrot3"]["selected"] == 0            # a demanded partition with no pick SHOWS
    assert abs(sum(r["realized_share"] for r in out) - 1.0) < 1e-12
    # no residual, no test statistic, no verdict — a shape read at this n carries none.
    assert set(out[0]) == {"partition", "target_share", "selected", "realized_share"}


def test_realized_vs_target_is_empty_shares_not_a_crash_on_an_empty_selection():
    out = RCS.realized_vs_target([], {"mandelbrot": 1.0})
    assert out == [{"partition": "mandelbrot", "target_share": 1.0,
                    "selected": 0, "realized_share": 0.0}]


# --------------------------------------------------------------------------- #
# gate-report accrual
# --------------------------------------------------------------------------- #
def test_gate_report_accrual_counts_the_two_sites_separately(tmp_path):
    d = tmp_path / ESINKS.MINING_GATE_REPORTS
    d.mkdir(parents=True)
    rows = [
        # would-cut at RELEASE (0.31 < 0.50) and selected anyway — the false-cut signal.
        {"key": "k1", "p_ge3": 0.31, "would_cut": True, "selected": True,
         "pool_floor": 0.25, "would_pass_pool": True, "would_cut_pool": False, "pooled": True},
        # would-cut at BOTH sites, not selected.
        {"key": "k2", "p_ge3": 0.10, "would_cut": True, "selected": False,
         "pool_floor": 0.25, "would_pass_pool": False, "would_cut_pool": True, "pooled": False},
        # clears both.
        {"key": "k3", "p_ge3": 0.80, "would_cut": False, "selected": True,
         "pool_floor": 0.25, "would_pass_pool": True, "would_cut_pool": False, "pooled": True},
        # LEGACY shape: no `would_cut_pool` key. Must be DERIVED, not read as False — a file
        # is a mix of both formats after any partial re-run.
        {"key": "k4", "p_ge3": 0.05, "would_cut": True, "selected": False,
         "pool_floor": 0.25, "would_pass_pool": False, "pooled": False},
    ]
    (d / f"{RCS.SITE}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    acc = RCS.gate_report_accrual(tmp_path)
    assert acc["n_rows"] == 4
    assert acc["release_site"] == {"n_would_cut": 3, "n_would_cut_selected": 1}
    assert acc["pool_site"] == {"n_with_pool_site": 4, "n_would_cut_pool": 2,
                                "n_would_cut_pool_pooled": 0, "n_would_cut_pool_selected": 0}


def test_gate_report_accrual_is_zeroes_not_a_crash_when_the_log_is_absent(tmp_path):
    acc = RCS.gate_report_accrual(tmp_path / "nothing")
    assert acc["n_rows"] == 0 and acc["pool_site"]["n_would_cut_pool"] == 0


# --------------------------------------------------------------------------- #
# colorize behavior
# --------------------------------------------------------------------------- #
def test_colorize_behavior_reports_a_render_span_not_a_run_time(tmp_path):
    """The span is measured off render mtimes and EXCLUDES intake. Asserted here so the
    number cannot quietly start being called run time."""
    (tmp_path / "renders").mkdir()
    for i in range(3):
        p = tmp_path / "renders" / f"em_{i}.jpg"
        p.write_bytes(b"x")
        import os
        os.utime(p, (1_700_000_000 + i * 10, 1_700_000_000 + i * 10))
    out = RCS.colorize_behavior(tmp_path, [prow("a"), prow("b"), prow("c")],
                                {"target_accounting": {"target_gated": 60, "post_floor": 12,
                                                       "ungated_strange": 5,
                                                       "release_eligible": 17}})
    assert out["render_span_s"] == 20.0 and out["attempts"] == 3
    assert out["s_per_attempt"] == 10.0
    assert out["ungated_strange"] == 5 and out["post_floor"] == 12
    assert "run_time_s" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
