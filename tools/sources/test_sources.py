"""Acceptance for the minibrot source sheets (`prompts/minibrot_source_sheets.md`
plus its addendum).

The invariants that make the batch mean anything:

  * **Identical framing across every sheet.** If two sheets rendered at different
    geometry or a different palette, the comparison the batch exists to make would be
    void. Pinned as a byte-identical argv against the triage wall's own renderer.
  * **No `A`-feasibility exclusion** (addendum §1). The known-good `mb19_p35` fails the
    roster's 1-decade floor; a cut that would drop the canonical good example cannot
    gate a fertility race. The margin is recorded and the atom is kept.
  * **Depth-spanning sampling** (addendum §2) that never pads a short sheet.
  * **Primitive vs satellite is NOT claimed.** Two cheap criteria were tried and both
    were falsified by the counting theorem on a complete period-n population; the tests
    below pin the falsification so the claim cannot creep back in.
  * **No per-tile metadata** on a sheet; the aggregate mix lives in the header only.

Run:  uv run python -m pytest tools/sources/test_sources.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "corpus", REPO_ROOT / "tools" / "descent",
          REPO_ROOT / "tools" / "explorer", REPO_ROOT / "tools" / "sourcing"):
    sys.path.insert(0, str(p))

import mpmath as mp                 # noqa: E402
import artifacts as A               # noqa: E402
import atom_lib as al               # noqa: E402
import source_store as ss           # noqa: E402
import render_tiles as rt           # noqa: E402
import sheet as sh                  # noqa: E402
import sources as S                 # noqa: E402
import triage_store as ts           # noqa: E402
import prerender_triage as pre      # noqa: E402
import deep_center_finder as dcf    # noqa: E402

BIN = REPO_ROOT / "target" / "release" / "fractal-generator.exe"


# --------------------------------------------------------------------------- #
# framing: identical across sheets, and identical to the triage wall
# --------------------------------------------------------------------------- #
def test_framing_constants_are_the_wall_s():
    """One definition of framing, imported — not a second copy that can drift."""
    assert ss.SCALES == ts.SCALES == (1, 4, 16)
    assert ss.DEFAULT_SCALE == ts.DEFAULT_SCALE == 4
    assert (ss.TILE_W, ss.TILE_H, ss.TILE_SS) == (ts.THUMB_W, ts.THUMB_H, ts.THUMB_SS)
    assert ss.TILE_PALETTE == ts.THUMB_PALETTE == "blue_orange"
    assert ss.TILE_COLORMAPS == ts.THUMB_COLORMAPS


def test_tile_argv_matches_the_triage_wall_byte_for_byte(tmp_path):
    """A source-sheet tile and a triage-wall tile of the SAME atom must be the same
    render command — otherwise no sheet is comparable to the wall or to another sheet."""
    atom = {"id": "mt000000000000", "cx": "-0.75", "cy": "0.1",
            "window_scale": "1.0000000000e-04", "family": "mandelbrot"}
    for scale in ss.SCALES:
        out = tmp_path / f"t{scale}.png"
        mine = rt.tile_argv(atom, scale, out)
        geom = {"cx": atom["cx"], "cy": atom["cy"],
                "base": atom["window_scale"], "family": atom["family"]}
        fw = ts.frame_width(geom["base"], scale)
        import render_core as rc
        theirs = rc.render_one_argv(geom["cx"], geom["cy"], f"{fw:.17e}",
                                    rc.auto_maxiter(fw), ts.THUMB_W, ts.THUMB_H,
                                    ts.THUMB_SS, ts.THUMB_PALETTE, ts.THUMB_COLORMAPS,
                                    out, family=geom["family"])
        assert mine == theirs, f"scale {scale}: sheet argv diverged from the wall's"


def test_frame_width_is_scale_times_atom_size():
    assert ss.frame_width("2.0e-10", 4) == pytest.approx(8.0e-10)
    assert ss.frame_width("2.0e-10", 1) == pytest.approx(2.0e-10)
    assert ss.frame_width("2.0e-10", 16) == pytest.approx(3.2e-9)


# --------------------------------------------------------------------------- #
# addendum §1 — feasibility is recorded, never an exclusion
# --------------------------------------------------------------------------- #
def test_sub_floor_atom_is_kept_not_cut():
    """`mb19_p35` — the canonical known-good reference — fails the roster's 1-decade
    deploy floor. It must survive `make_atom` with the margin recorded."""
    al.set_precision()
    c = mp.mpc(mp.mpf("-0.74977483272365342795786040375088960"),
               mp.mpf("0.10761724352653678278696798751738616"))
    rec = al.make_atom(c, 35, "test")
    assert rec is not None, "the known-good reference was excluded by make_atom"
    import build_minibrot_roster as brs
    assert rec["f64_margin_deploy_decades"] < brs.MARGIN_MIN_DECADES, \
        "expected mb19 to sit BELOW the roster floor — if not, the test lost its point"
    assert rec["period"] == 35


def test_describe_counts_sub_floor_atoms():
    al.set_precision()
    c = mp.mpc(mp.mpf("-0.74977483272365342795786040375088960"),
               mp.mpf("0.10761724352653678278696798751738616"))
    d = al.describe([al.make_atom(c, 35, "test")])
    assert d["n_below_feasibility_floor"] == 1     # surfaced in the sheet header


# --------------------------------------------------------------------------- #
# primitive vs satellite — the claim is NOT shipped; these tests keep it that way
# --------------------------------------------------------------------------- #
def test_no_shape_is_claimed_only_raw_quantities():
    """The prompt asks for primitive-vs-satellite. No cheap criterion survived
    verification (see atom_lib's docstring), so the record carries the RAW atom-domain
    quantities and `shape is None`. This test exists so the claim cannot creep back in
    without someone deleting it deliberately."""
    al.set_precision()
    r = dcf.newton_nucleus(mp.mpc(-1.7548776662, 0.0), 3, degree=2, max_steps=200)
    rec = al.make_atom(r.c, 3, "test")
    assert rec["shape"] is None and rec["satellite_of_period"] is None
    assert rec["embedding_depth"] is None
    assert isinstance(rec["atom_domain_index"], int)
    assert isinstance(rec["atom_domain_divides_period"], bool)
    assert "satellite_frac" not in al.describe([rec])
    assert al.describe([rec])["shape_available"] is False


def test_the_atom_domain_criterion_really_does_over_call():
    """Pins the FALSIFICATION itself, on a complete period-6 population: the classical
    atom-domain guess flags far more atoms than the counting theorem allows. If this
    ever stops failing, the criterion deserves another look — but until then, shipping
    it as a satellite count would be shipping a wrong number."""
    atoms, stats = S.src_complete_low_n(period_max=6, log=lambda *a: None)
    p6 = [a for a in atoms if a["period"] == 6]
    th = stats["theorem_satellites"]
    row6 = next(r for r in th["per_period"] if r["period"] == 6)
    assert row6["complete"], "need a complete period-6 population for this check"
    guess = sum(1 for a in p6 if a["atom_domain_divides_period"])
    assert guess > row6["satellites_expected"], (
        f"atom-domain guess {guess} vs theorem {row6['satellites_expected']}")


def test_complete_enumeration_is_complete_and_exact():
    """Q_n has degree nu(n), so its roots ARE the period-n population. Completeness is a
    construction; this verifies the implementation delivers it, and that the EXACT
    satellite fraction (the one number the theorem gives without a classifier) is
    reported."""
    atoms, stats = S.src_complete_low_n(period_max=6, log=lambda *a: None)
    th = stats["theorem_satellites"]
    for r in th["per_period"]:
        if r["period"] == 1:
            continue          # Q_1 root is c=0, excluded as degenerate by ORIGIN_EPS
        assert r["complete"], f"period {r['period']}: {r['found']}/{r['expected_total']}"
    assert th["satellite_frac"] is not None and 0 < th["satellite_frac"] < 1
    by_p = {}
    for a in atoms:
        by_p.setdefault(a["period"], []).append(a)
    assert len(by_p[6]) == al.nu(6) == 27


def test_period_polynomial_degree_equals_component_count():
    qs = S.period_polynomials(10)
    for n in range(1, 11):
        assert len(qs[n]) - 1 == al.nu(n), n


def test_nu_matches_known_component_counts():
    assert [al.nu(n) for n in range(1, 13)] == [1, 1, 3, 6, 15, 27, 63, 120, 252, 495, 1023, 2010]


# --------------------------------------------------------------------------- #
# the dedup-noise defect (what caught the classifier error in the first place)
# --------------------------------------------------------------------------- #
def test_real_axis_nucleus_dedups_to_one_atom():
    """`dcf.nucleus_dedup_key` rounds to significant digits, so a real-axis nucleus
    whose imaginary part is Newton noise got a DIFFERENT key on every solve — c=-1.3107
    entered a 262-atom probe population ten times. `atom_lib.snap` fixes it locally."""
    al.set_precision()
    ids = set()
    for seed in [(-1.31, 0.0), (-1.3107, 1e-9), (-1.3107, -1e-9), (-1.305, 0.002)]:
        r = dcf.newton_nucleus(mp.mpc(*seed), 4, degree=2, max_steps=150)
        rec = al.make_atom(r.c, 4, "t")
        if rec:
            ids.add(rec["id"])
    assert len(ids) == 1, f"the same real-axis nucleus produced {len(ids)} ids"


def test_snap_leaves_genuine_off_axis_atoms_alone():
    c = mp.mpc(mp.mpf("-0.12256117"), mp.mpf("0.74486177"))
    assert al.snap(c) == c
    assert al.snap(mp.mpc(mp.mpf("-1.31"), mp.mpf("1e-40"))).imag == 0


# --------------------------------------------------------------------------- #
# addendum §2 — depth-spanning, never padded
# --------------------------------------------------------------------------- #
def _fake(n, lo=0.0, hi=8.0):
    return [{"id": f"mt{i:012x}", "log10_abs_A": lo + (hi - lo) * i / max(1, n - 1),
             "period": 3 + i % 7, "shape": None, "satellite_of_period": None,
             "embedding_depth": None, "atom_domain_index": 2,
             "atom_domain_divides_period": False, "on_real_axis": False,
             "f64_margin_deploy_decades": 5.0}
            for i in range(n)]


def test_span_by_depth_spans_the_range():
    picked = al.span_by_depth(_fake(500), 50)
    assert len(picked) <= 50
    assert picked[0]["log10_abs_A"] == pytest.approx(0.0, abs=0.05)
    assert picked[-1]["log10_abs_A"] == pytest.approx(8.0, abs=0.05)
    # spread, not the natural head: every depth decade is represented
    hist = al.depth_histogram(picked)
    assert sum(1 for h in hist if h["n"] > 0) >= 7


def test_span_by_depth_never_pads_a_short_source():
    short = _fake(12)
    picked = al.span_by_depth(short, 150)
    assert len(picked) == 12, "a short source must ship short, never padded"


def test_describe_reports_the_mix():
    d = al.describe(_fake(40))
    for k in ("period_min", "period_max", "log10_abs_A_min", "log10_abs_A_max",
              "depth_histogram", "n_below_feasibility_floor", "on_real_axis_n",
              "shape_available"):
        assert k in d


# --------------------------------------------------------------------------- #
# sheets: aggregate in the header, nothing per tile
# --------------------------------------------------------------------------- #
def test_sheet_has_no_per_tile_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("FRACTAL_ARTIFACTS_ROOT", str(tmp_path))
    atoms = _fake(20)
    for a in atoms:                       # descriptors a leak would most likely expose
        a.update({"source": "probe", "period": 11, "cx": "-0.75", "cy": "0.1",
                  "window_scale": "1e-4", "family": "mandelbrot", "size": 1e-4,
                  "dedup_key": "x,y", "abs_A": 1e4})
    p = sh.build_sheet("t", "T", "b", atoms, al.describe(atoms))
    doc = p.read_text(encoding="utf-8")
    tiles = re.findall(r'<div class="tile"[^>]*>.*?</div>\s*</div>', doc, re.S)
    body = doc.split('<div id="wall">', 1)[1]
    for leak in ("period", "log10_abs_A", "dedup_key", "satellite", "abs_A",
                 "window_scale", "-0.75", "probe", "atom_domain"):
        assert leak not in body, f"per-tile region leaks {leak!r}"
    # a tile carries an opaque id and nothing else
    for t in re.findall(r'data-id="([^"]+)"', body):
        assert re.fullmatch(r"mt[0-9a-f]{12}", t), t
    # ...while the HEADER does carry the aggregate mix (that is the point)
    head = doc.split('<div id="wall">', 1)[0]
    assert "log10|A|" in head and "depth histogram" in head and "atom size" in head


def test_sheet_image_paths_are_relative_to_the_sheet(tmp_path, monkeypatch):
    """Sheets must open by double-click with no manual path fixing, so every image
    reference is relative and resolves from the sheet's own directory."""
    monkeypatch.setenv("FRACTAL_ARTIFACTS_ROOT", str(tmp_path))
    atoms = _fake(3)
    p = sh.build_sheet("t2", "T", "b", atoms, al.describe(atoms))
    for src in re.findall(r'src="([^"]+)"', p.read_text(encoding="utf-8")):
        assert src.startswith("../tiles/"), src
        assert not Path(src).is_absolute()


# --------------------------------------------------------------------------- #
# storage classes
# --------------------------------------------------------------------------- #
def test_tiles_and_sheets_relocate_records_stay_in_tree():
    root = A.artifacts_root()
    for rel in ("data/minibrot_sources/tiles/mt0123456789ab__x4.png",
                "data/minibrot_sources/sheets/probe.html",
                "data/minibrot_sources/tiles", "data/minibrot_sources/sheets"):
        assert A._is_minibrot_source_bulk(A._norm(rel)), rel
        assert A.is_relocated(rel) and A.resolve(rel) == root / rel
    for rel in ("data/minibrot_sources/probe/atoms.jsonl",
                "data/minibrot_sources/probe/meta.json",
                "data/minibrot_sources/overlap.json",
                "data/minibrot_sources/tiles_staging/x.png"):
        assert not A._is_minibrot_source_bulk(A._norm(rel)), rel
        assert not A.is_relocated(rel)
        assert A.resolve(rel) == A.REPO_ROOT / rel


def test_durable_nuclei_lists_are_not_gitignored():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import paths
    for rel in ("data/minibrot_sources/probe/atoms.jsonl",
                "data/minibrot_sources/probe/meta.json",
                "data/minibrot_sources/overlap.json",
                "data/minibrot_sources/index.json"):
        paths.durable(rel)          # raises if git would silently drop it


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def test_atom_ids_agree_with_the_triage_pool():
    """Same id function as the triage wall, so an atom found twice collapses and the
    overlap matrix is meaningful."""
    al.set_precision()
    pool = ts.load_pool()
    d2 = [a for a in pool if a["degree"] == 2]
    if not d2:
        pytest.skip("triage pool not built")
    a = d2[0]
    rec = al.make_atom(mp.mpc(mp.mpf(a["cx"]), mp.mpf(a["cy"])), a["period"], "test")
    assert rec is not None and rec["id"] == a["id"]


def test_curated_seed_supply_is_real_and_named():
    seeds = S.curated_seeds()
    assert seeds, "no curated locations found at all"
    assert all("cx" in s and "cy" in s and "origin" in s for s in seeds)
    # the prompt's mandelbrot_named_seeds.json does not exist; the supply is code constants
    assert not (REPO_ROOT / "data" / "mandelbrot_named_seeds.json").exists()
    assert any("emit_deep_pool" in s["origin"] for s in seeds)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
