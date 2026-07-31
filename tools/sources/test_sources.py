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
def test_the_comparability_framing_is_shared_with_the_wall():
    """What makes the sources comparable is the LADDER and the PALETTE — imported from
    the triage wall so there is one definition. Pixel geometry is deliberately not
    shared (see below)."""
    assert ss.SCALES == ts.SCALES == (1, 4, 16)
    assert ss.DEFAULT_SCALE == ts.DEFAULT_SCALE == 4
    assert ss.TILE_PALETTE == ts.THUMB_PALETTE == "blue_orange"
    assert ss.TILE_COLORMAPS == ts.THUMB_COLORMAPS


def test_sheet_geometry_is_its_own_and_costs_the_same_as_the_wall_s():
    """Sheets render bigger than the wall on purpose: the wall is a dense keyboard grid
    tuned for ~1 s per tile, a sheet is read at leisure with all three rungs side by
    side. 640x360 ss1 carries the SAME sample count as the wall's 320x180 ss2, so the
    change buys 2x linear detail for zero extra render cost — that equality is the
    justification, so it is pinned."""
    assert (ss.TILE_W, ss.TILE_H, ss.TILE_SS) == (640, 360, 1)
    sheet_samples = ss.TILE_W * ss.TILE_H * ss.TILE_SS ** 2
    wall_samples = ts.THUMB_W * ts.THUMB_H * ts.THUMB_SS ** 2
    assert sheet_samples == wall_samples, (sheet_samples, wall_samples)
    assert ss.TILE_W == 2 * ts.THUMB_W and ss.TILE_H == 2 * ts.THUMB_H


def test_every_sheet_renders_at_one_geometry(tmp_path):
    """The load-bearing invariant: two atoms on two different sheets must produce the
    same render command apart from their coordinates. One geometry, one palette, one
    ladder — otherwise no sheet is comparable to another."""
    import render_core as rc
    a = {"id": "mt000000000000", "cx": "-0.75", "cy": "0.1",
         "window_scale": "1.0000000000e-04", "family": "mandelbrot"}
    b = {**a, "id": "mt111111111111", "cx": "-0.5", "cy": "0.6"}
    for scale in ss.SCALES:
        av = rt.tile_argv(a, scale, tmp_path / "a.png")
        bv = rt.tile_argv(b, scale, tmp_path / "b.png")
        strip = lambda v: [x for i, x in enumerate(v)                      # noqa: E731
                           if v[i - 1] not in ("--cx", "--cy", "--out")
                           and x not in ("--cx", "--cy", "--out")]
        assert strip(av) == strip(bv), f"scale {scale}: two atoms rendered differently"
        # ...and the geometry in that command is the sheet geometry
        assert av[av.index("--width") + 1] == str(ss.TILE_W)
        assert av[av.index("--supersample") + 1] == str(ss.TILE_SS)
        assert av[av.index("--palette") + 1] == ss.TILE_PALETTE
        # ...and the frame width is exactly `scale` x the atom's own size
        fw = float(av[av.index("--fw") + 1])
        assert fw == pytest.approx(float(a["window_scale"]) * scale, rel=1e-9)


def test_reference_tiles_render_at_the_same_geometry_as_atom_tiles(tmp_path):
    """The reference row exists to be compared against, so it must be drawn at the same
    size as what it is compared with. It once was not: a geometry change re-rendered
    every atom tile at 640x360 and left the references at 320x180, because
    `ensure_reference_tiles` did not thread `force` through."""
    import inspect
    src = inspect.getsource(rt.ensure_reference_tiles)
    assert "force=force" in src, "ensure_reference_tiles must pass force to render_atoms"
    refs = rt.reference_atoms()
    assert refs, "no references — run build_triage_pool.py --refs-only"
    for r in refs:
        for scale in ss.SCALES:
            argv = rt.tile_argv(r, scale, tmp_path / "r.png")
            assert argv[argv.index("--width") + 1] == str(ss.TILE_W)
            assert argv[argv.index("--height") + 1] == str(ss.TILE_H)
            assert argv[argv.index("--supersample") + 1] == str(ss.TILE_SS)


def test_rebuild_rederives_the_sample_instead_of_narrowing_it():
    """A rebuild must be idempotent. Reusing the previous run's `sheet_ids` is not: each
    pass drops whatever failed to render, so a bad pass ratchets a sheet toward empty and
    no later rebuild can recover it. Two concurrent rebuilds did exactly that, taking
    three sheets to 0 atoms — the sample is now re-derived from the durable population."""
    import inspect
    src = inspect.getsource(__import__("run_sheets").rebuild_only)
    assert "span_by_depth" in src
    assert 'meta.get("sheet_ids"' not in src, "rebuild must not seed from stored sheet_ids"


def test_span_by_depth_is_deterministic():
    """Which is what makes re-deriving the sample safe."""
    pop = _fake(400)
    assert [a["id"] for a in al.span_by_depth(pop, 150)] ==            [a["id"] for a in al.span_by_depth(list(reversed(pop)), 150)]


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
    body = doc.split('<div id="wall">', 1)[1]
    for leak in ("period", "log10_abs_A", "dedup_key", "satellite", "abs_A",
                 "window_scale", "-0.75", "probe", "atom_domain"):
        assert leak not in body, f"per-tile region leaks {leak!r}"
    for tid in re.findall(r"/tiles/([^_]+)__", body):
        assert re.fullmatch(r"mt[0-9a-f]{12}", tid), tid
    head = doc.split('<div class="colhead">', 1)[0]
    assert "log10|A|" in head and "depth histogram" in head and "atom size" in head


def test_every_atom_gets_one_row_showing_all_three_rungs(tmp_path, monkeypatch):
    """All three ladder rungs on screen at once, one row per atom — comparing 1x against
    16x must not be a memory test, which is what click-to-cycle made it."""
    monkeypatch.setenv("FRACTAL_ARTIFACTS_ROOT", str(tmp_path))
    atoms = _fake(7)
    for a in atoms:                       # every rung available
        for s in ss.SCALES:
            q = ss.tile_path(a["id"], s)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(b"x")
    doc = sh.build_sheet("t3", "T", "b", atoms, al.describe(atoms)).read_text(encoding="utf-8")
    body = doc.split('<div id="wall">', 1)[1]
    assert body.count('class="row"') == len(atoms)
    assert body.count('class="cell') == len(atoms) * len(ss.SCALES)
    assert "cell na" not in body
    for a in atoms:
        for s in ss.SCALES:
            assert f'{a["id"]}__x{s}.png' in body
    assert "<script" not in doc, "the sheet should need no JavaScript"
    assert doc.count('class="colhead"') == 1 and "the sheet frame" in doc


def test_a_rung_that_cannot_render_becomes_an_empty_cell_not_a_broken_image(tmp_path,
                                                                            monkeypatch):
    """The deepest atoms clear the f64 wall at 4x and 16x but not at 1x, where the frame
    is four times narrower. That atom still earns its row on the rungs that rendered, and
    the missing rung must not show as a broken image."""
    monkeypatch.setenv("FRACTAL_ARTIFACTS_ROOT", str(tmp_path))
    atoms = _fake(3)
    for a in atoms:
        for s in ss.SCALES:
            if s == 1:
                continue                   # 1x unavailable, as at the wall
            q = ss.tile_path(a["id"], s)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(b"x")
    doc = sh.build_sheet("t4", "T", "b", atoms, al.describe(atoms)).read_text(encoding="utf-8")
    body = doc.split('<div id="wall">', 1)[1]
    assert body.count('class="row"') == len(atoms)          # rows survive
    assert body.count("cell na") == len(atoms)              # exactly the 1x rung
    for a in atoms:
        assert f'{a["id"]}__x1.png' not in body             # no broken <img>
        assert f'{a["id"]}__x4.png' in body


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
