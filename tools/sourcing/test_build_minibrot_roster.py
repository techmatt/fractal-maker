"""Fast unit tests for the roster builder's pure policy helpers (no mpmath sourcing).

The slow part (Newton sourcing) is exercised by running the tool; these lock the
load-bearing *policy* — band assignment, the feasibility wall constant, cell-spanning
selection, and the determinism + minibrot-disjointness of the train/eval split — so a
future edit that would silently reshuffle inherited splits fails here.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_minibrot_roster as R  # noqa: E402


def test_bands_partition_period_range():
    # every period 3..15 lands in exactly one band; nothing outside leaks in
    for p in range(3, 16):
        assert R.band_of(p) is not None
    assert R.band_of(2) is None
    assert R.band_of(16) is None
    seen = set()
    for lo, hi in R.BANDS:
        for p in range(lo, hi + 1):
            assert p not in seen, "bands overlap"
            seen.add(p)
    assert seen == set(range(3, 16))


def test_deploy_wall_matches_atom_instrument():
    # the roster's admission wall must equal the instrument's own wall term, or the
    # margin recorded in the roster disagrees with the predictor it claims to use.
    import deep_center_finder as dcf
    inst = dcf.AtomInstrument(
        degree=2, period=5, A=None, abs_A=1.0, arg_A=0.0, log10_abs_A=0.0,
        window_scale=1.0, rotation_rad=0.0, rotation_ambiguity_rad=0.0, required_dps=50)
    # margin at |A|=1 (log10=0) equals the wall constant
    m = inst.f64_wall_margin_decades(R.DEPLOY_W, ss=R.DEPLOY_SS)
    assert math.isclose(m, R.deploy_wall_log10(), rel_tol=0, abs_tol=1e-9)


def test_select_spanning_keeps_extremes_and_count():
    atoms = [dict(log10_abs_A=float(i), dedup_key=f"k{i}") for i in range(20)]
    sel = R.select_spanning(atoms, 8)
    assert len(sel) == 8
    logs = [a["log10_abs_A"] for a in sel]
    assert logs[0] == 0.0 and logs[-1] == 19.0        # spans full range (near-boundary kept)
    assert logs == sorted(logs)
    # under target -> return all
    assert len(R.select_spanning(atoms[:5], 8)) == 5


def _cell(deg, band, n):
    return [dict(degree=deg, band=band, dedup_key=f"{deg}_{band}_{i:03d}") for i in range(n)]


def test_split_is_deterministic_and_stratified_70_30():
    atoms = _cell(2, (3, 4), 10) + _cell(3, (5, 6), 10)
    R.assign_splits(atoms)
    splits = {a["dedup_key"]: a["split"] for a in atoms}
    # re-run on a fresh copy -> identical assignment
    atoms2 = _cell(2, (3, 4), 10) + _cell(3, (5, 6), 10)
    R.assign_splits(atoms2)
    assert splits == {a["dedup_key"]: a["split"] for a in atoms2}
    # 70/30 within each cell
    for deg, band in [(2, (3, 4)), (3, (5, 6))]:
        cell = [a for a in atoms if a["degree"] == deg and a["band"] == band]
        n_train = sum(1 for a in cell if a["split"] == "train")
        assert n_train == round(R.TRAIN_FRAC * len(cell)) == 7
        assert all(a["split"] in ("train", "eval") for a in cell)


def test_split_disjoint_and_stable_when_other_cell_grows():
    # adding atoms to one cell must not reshuffle another cell (cell-local seed)
    base = _cell(2, (3, 4), 8)
    R.assign_splits([dict(a) for a in base] + _cell(3, (5, 6), 4))
    a1 = _cell(2, (3, 4), 8)
    R.assign_splits(a1 + _cell(3, (5, 6), 4))
    a2 = _cell(2, (3, 4), 8)
    R.assign_splits(a2 + _cell(3, (5, 6), 99))          # other cell much larger
    s1 = {a["dedup_key"]: a["split"] for a in a2 if a["degree"] == 2}
    s0 = {a["dedup_key"]: a["split"] for a in a1 if a["degree"] == 2}
    assert s0 == s1
