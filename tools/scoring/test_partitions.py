#!/usr/bin/env python
"""`tools/scoring/partitions.py` is the ONLY copy of the fractal_type ⟷ partition map.

Two things, and the second is the one that matters. The first is that the map is internally
coherent (injective, total over `ALL_FAMS`, a real inverse). The second is a **source scan**:
no tracked Python file outside this module may define `FT2FAM` / `FAM2FT` as a literal again.

Why a source scan rather than trust. On 2026-08-02 these nine pairs had seven literal copies
across `derive_t_good_{v7,v8}`, `keeper_cut.py`, `v7/build_manifest.py` and three test files —
none of them wrong, all of them independently editable, and one of them (`keeper_cut.py`'s)
carrying a comment that said it mirrored another. A duplicated constant does not fail when it
is created; it fails years later when someone adds a family to one copy. The scan is what
turns "please import it" into a gate.

  uv run pytest tools/scoring/test_partitions.py -q
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import partitions as P  # noqa: E402

OWNER = "tools/scoring/partitions.py"
# This file quotes the copy-shapes it hunts for (in `test_the_scan_would_actually_catch_a_copy`),
# so it exempts itself — the exemption list is exactly two files and both are named here.
EXEMPT = {OWNER, "tools/scoring/test_partitions.py"}
# `NAME = {` — a literal dict binding. `FT2FAM = {v: k for ...}` (a derived inverse) is
# matched too, on purpose: the inverse also has exactly one correct home.
LITERAL = re.compile(r"^\s*(FT2FAM|FAM2FT)\s*(:[^=]*)?=\s*\{", re.M)


def test_the_map_is_injective_and_its_inverse_is_real():
    assert len(set(P.FT2FAM.values())) == len(P.FT2FAM), "two fractal_types share a partition"
    assert P.FAM2FT == {v: k for k, v in P.FT2FAM.items()}
    for ft, fam in P.FT2FAM.items():
        assert P.FAM2FT[fam] == ft


def test_every_partition_is_in_ALL_FAMS_and_vice_versa():
    """`ALL_FAMS` is what the derivations walk to stamp partitions with no eval rows. A
    partition reachable in production but missing from that list would be silently absent
    from every t_good table rather than stamped UNCALIBRATED.

    `ALL_FAMS` is the base partitions PLUS the derived ones — expressed relationally so
    adding either kind keeps the assertion true without a bumped number."""
    assert set(P.ALL_FAMS) == set(P.FT2FAM.values()) | set(P.DERIVED_FAMS)
    assert len(P.ALL_FAMS) == len(set(P.ALL_FAMS)), "ALL_FAMS has a duplicate"
    assert P.DERIVED_FAMS, "the derived layer evaluated empty — every assertion over it is vacuous"


def test_a_derived_partition_has_no_fractal_type_of_its_own():
    """The structural claim of the derived layer: `phoenix:classic` is NOT a render family,
    so it must be absent from both directions of the token map and reachable only through
    `base_partition`. If it ever acquires a `fractal_type`, it stops being derived and the
    row-side resolver becomes dead code that still runs."""
    for d, base in P.DERIVED_FAMS.items():
        assert d not in P.FAM2FT, f"{d} is derived but has a fractal_type — pick one"
        assert d not in P.FT2FAM.values()
        assert base in P.FAM2FT, f"{d}'s base {base} is not a real partition"
        assert P.base_partition(d) == base
    for f in P.FT2FAM.values():
        assert P.base_partition(f) == f, "base_partition must be identity on base partitions"


def test_julia_planes_are_namespaced_and_native_ones_are_not():
    """The one substantive thing the map does: keep `julia_multibrot4` (a julia-plane view)
    from colliding with `multibrot4` (its native twin). They are different supply, different
    scarcity and different objectives, so they must be different partitions."""
    for d in (3, 4, 5):
        assert P.FT2FAM[f"julia_multibrot{d}"] == f"julia:multibrot{d}"
        assert P.FT2FAM[f"multibrot{d}"] == f"multibrot{d}"
    assert P.FT2FAM["julia"] == "julia:mandelbrot"


def test_partition_of_default_is_explicit_at_the_call_site():
    assert P.partition_of("julia_multibrot4") == "julia:multibrot4"
    assert P.partition_of("not_a_family") is None                 # keeper_cut's convention
    assert P.partition_of("not_a_family", "not_a_family") == "not_a_family"   # derivations'


# --------------------------------------------------------------------------- #
# The phoenix:classic split. Injection proofs, both directions.
# --------------------------------------------------------------------------- #
CLASSIC_LEDGER_ROW = {          # the discovery-ledger schema (phoenix_* stamp)
    "family": "phoenix", "phoenix_c_re": 0.5667, "phoenix_c_im": 0.0,
    "phoenix_p_re": -0.5, "phoenix_p_im": 0.0,
    "phoenix_zm1_re": 0.0, "phoenix_zm1_im": 0.0,
}
# The pre-axis corpus render block — verbatim shape of all 73 labeled classic rows in
# 2026-07-05_gather_v6: a phoenix token, a viewport, and NOT ONE parameter field.
LEGACY_ABSENT_AXES_ROW = {
    "fractal_type": "phoenix", "cx": "0.3746259253780486", "cy": "0.3789509981745586",
    "fw": "0.011492717668006597", "maxiter": 8000, "palette": "RdBu",
}
VARIED_RENDER_ROW = {           # the corpus render-block schema (family_params)
    "fractal_type": "phoenix", "cx": "0.36332145469349897", "cy": "-0.4138261338719179",
    "fw": "0.28371162512578074",
    "c_re": "-1.0891829835364482", "c_im": "0.48116538489449995",
    "p_re": "-0.2224834566864348", "p_im": "0.1723706161191405",
    "zm1_re": "-0.22447924416366366", "zm1_im": "-0.34704670572868107",
}


def test_a_varied_phoenix_row_never_resolves_classic():
    """The direction that costs labels if it breaks: varied phoenix annexed into a partition
    whose entire content is one parameter value. Includes a row that differs on ONE axis by
    one ulp — exact equality, no tolerance, ever."""
    assert P.partition_of_row(VARIED_RENDER_ROW) == "phoenix"
    for axis in ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"):
        i = ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im").index(axis)
        row = {"fractal_type": "phoenix", "cx": "0", "cy": "0", "fw": "3.0"}
        for j, k in enumerate(("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im")):
            row[k] = P.PHOENIX_CLASSIC_POINT[j]
        row[axis] = math.nextafter(float(P.PHOENIX_CLASSIC_POINT[i]), 1e9)
        assert P.partition_of_row(row) == "phoenix", f"one ulp on {axis} resolved classic"


def test_a_legacy_absent_axes_row_resolves_classic():
    """The direction that LOSES the whole classic population if it breaks: every classic-era
    row in the tree (84 corpus rows on 2026-08-04) carries no parameter axes at all, so
    "absent" must resolve to the pinned point exactly as `row_phoenix_key` resolves it."""
    assert P.partition_of_row(LEGACY_ABSENT_AXES_ROW) == P.CLASSIC_PHOENIX
    assert P.partition_of_row(CLASSIC_LEDGER_ROW) == P.CLASSIC_PHOENIX
    # explicitly-stamped and absent-axes forms are the SAME point, both schemas
    assert P.phoenix_point(LEGACY_ABSENT_AXES_ROW) == P.phoenix_point(CLASSIC_LEDGER_ROW)


def test_the_two_on_disk_schemas_are_read_as_schemas_not_per_axis():
    """A ledger-stamped VARIED row must not fall back to the render-block keys axis by axis
    (and vice versa) — mixing them synthesizes a parameter point neither writer wrote."""
    # A ledger-schema row whose `p` axes are absent, carrying a VARIED `p_re` in the render
    # spelling. Schema-at-a-time: the ledger stamp wins, absent `p` -> the legacy default ->
    # classic. Per-axis fallback: the stray `p_re` is picked up -> varied. They disagree, so
    # this fixture can tell them apart — which the obvious one (both keys present on the same
    # axis) cannot, since both readers return the same value there.
    mixed = {k: v for k, v in CLASSIC_LEDGER_ROW.items() if not k.startswith("phoenix_p_")}
    mixed["p_re"] = "-0.3"
    assert P.phoenix_point(mixed)[2] == -0.5, "an axis leaked across the two on-disk schemas"
    assert P.partition_of_row(mixed) == P.CLASSIC_PHOENIX
    # ...and the reverse: a render-schema row is not rescued by a stray ledger key
    stray = dict(VARIED_RENDER_ROW)
    assert P.phoenix_point(stray) == tuple(float(stray[k]) for k in
                                           ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"))


def test_a_schema_with_no_parameter_axes_is_refused_not_guessed():
    """The eval slice (`data/<v>/eval_scores_<v>.jsonl`) carries `fractal_type` and no
    parameter axes whatsoever. Its absent axes mean UNKNOWN, not classic — reading them as
    classic would annex every varied-phoenix eval row into a 41-look partition. So the row
    resolver refuses, and the token resolver (which cannot see a point) still answers."""
    eval_row = {"fractal_type": "phoenix", "location_id": "x", "source": "prospect_census",
                "label": 3, "v10_p_ge3": 0.4}
    with pytest.raises(ValueError, match="neither on-disk parameter schema"):
        P.partition_of_row(eval_row)
    assert P.partition_of(eval_row["fractal_type"]) == "phoenix"
    # a NON-phoenix row in the same schema is unaffected — the refusal is phoenix-scoped
    assert P.partition_of_row(dict(eval_row, fractal_type="julia_multibrot4")) == "julia:multibrot4"


def test_the_structural_markers_recognize_a_legacy_axis_free_row_in_both_schemas():
    """Non-vacuity for the marker rule: the two axis-free shapes that DO mean classic."""
    assert P.partition_of_row({"fractal_type": "phoenix", "cx": "0", "cy": "0", "fw": "3.0",
                               "palette": "viridis"}) == P.CLASSIC_PHOENIX
    assert P.partition_of_row({"family": "phoenix", "outcome_cx": "0.42",
                               "outcome_cy": "-0.67", "outcome_fw": "0.51"}) == P.CLASSIC_PHOENIX


def test_non_phoenix_rows_are_untouched_by_the_split():
    for ft, fam in P.FT2FAM.items():
        if fam == "phoenix":
            continue
        assert P.partition_of_row({"fractal_type": ft}) == fam
    assert P.partition_of_row({"fractal_type": "not_a_family"}) is None
    assert P.partition_of_row({"fractal_type": "nope"}, "nope") == "nope"


def test_token_only_partition_of_still_resolves_classic_rows_to_the_base():
    """`partition_of` takes a TOKEN and cannot see a parameter point, so it must keep
    answering `phoenix` — a caller with only the token has not been silently upgraded."""
    assert P.partition_of("phoenix") == "phoenix"


def test_registration_is_proved_to_come_before_resolution(monkeypatch):
    """The ordering guard, proved red. With `phoenix:classic` de-registered from `ALL_FAMS`,
    resolving a classic row must RAISE — not quietly emit a tenth partition key that no
    per-partition table has a row for. This is the failure mode the whole registration-first
    ordering exists to prevent, so it is asserted rather than trusted."""
    monkeypatch.setattr(P, "ALL_FAMS", [f for f in P.ALL_FAMS if f != P.CLASSIC_PHOENIX])
    with pytest.raises(KeyError, match="not registered"):
        P.partition_of_row(LEGACY_ABSENT_AXES_ROW)
    # ...and the base partition still resolves, so the guard is specific, not a blanket break
    assert P.partition_of_row(VARIED_RENDER_ROW) == "phoenix"


def test_the_classic_point_has_exactly_one_owner():
    """`production_seeder`'s dup-identity defaults are RE-EXPORTS of the partition owner's
    point, not a second literal. Two copies would be a dup rule and a partition rule that can
    disagree about which points are classic."""
    sys.path.insert(0, str(ROOT / "tools" / "atlas"))
    import production_seeder as ps
    assert (tuple(ps.PHOENIX_C_DEFAULT) + tuple(ps.PHOENIX_P_DEFAULT)
            + tuple(ps.PHOENIX_ZM1_DEFAULT)) == P.PHOENIX_CLASSIC_POINT
    # the two resolvers agree on every axis-absent form
    assert ps.row_phoenix_key(LEGACY_ABSENT_AXES_ROW) == P.phoenix_point(LEGACY_ABSENT_AXES_ROW)
    assert ps.row_phoenix_key(CLASSIC_LEDGER_ROW) == P.phoenix_point(CLASSIC_LEDGER_ROW)


def _tracked_python():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [p for p in out.stdout.splitlines() if p.strip()]


def test_no_second_literal_copy_of_the_map_exists():
    """THE point of this file. A second literal is a fork with a delayed fuse."""
    offenders = []
    for rel in _tracked_python():
        if rel.replace("\\", "/") in EXEMPT:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in LITERAL.finditer(text):
            line = text[:m.start()].count("\n") + 1
            offenders.append(f"{rel}:{line}")
    assert not offenders, (
        f"{len(offenders)} literal redefinition(s) of the partition map outside {OWNER}: "
        f"{offenders}. Import it instead — `from partitions import FT2FAM` with "
        f"tools/scoring on sys.path.")


def test_the_scan_would_actually_catch_a_copy(tmp_path):
    """Non-vacuity: the regex has to match the shape a copy is actually written in. All four
    real spellings from the pre-2026-08-02 tree, plus the derived inverse."""
    samples = [
        'FT2FAM = {\n    "mandelbrot": "mandelbrot",\n}',
        'FT2FAM = {"mandelbrot": "mandelbrot", "julia": "julia:mandelbrot"}',
        'FAM2FT = {\n    "mandelbrot": "mandelbrot",\n}',
        'FT2FAM = {v: k for k, v in FAM2FT.items()}',
        'FT2FAM: dict = {"mandelbrot": "mandelbrot"}',
    ]
    for s in samples:
        assert LITERAL.search(s), f"the scan would miss this copy:\n{s}"
    # ...and does not fire on the legitimate uses that must stay legal
    for ok in ("from partitions import FT2FAM", "part = FT2FAM.get(ft)",
               "live = set(kc.FT2FAM.values())", "# mirrors FT2FAM = {...}\n"):
        assert not LITERAL.search(ok), f"the scan false-positives on:\n{ok}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
