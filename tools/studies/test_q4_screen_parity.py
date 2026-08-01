r"""Stage-1 screen extension is ADDITIVE-ONLY: deployed accept verdicts are bit-identical.

The stage-1 minibrot screen (`q4_stage1_linear_fit.dense_grid` -> `q4_harvest_tight`
G-maxima + NMS framing, filtered at the tight cutoff) sits in production under descent
scoring. It was extended for the class-4 negative arm: `q4_multibrot_transfer._sweep_fates`
now ALSO records the OOD-surviving sub-cutoff windows and the OOD-masked windows
(`surv_box` / `masked_box` / `masked_clause`) so `build_minibrot_batch` can draw
structured near-misses. That collection is pure bookkeeping laid alongside the deployed
gating — it must not move a single ACCEPT verdict, because accepts feed descent scoring and
a drift there moves everywhere.

This asserts additive-only two ways, on a representative field across the deployed scales:

  1. ROOT INVARIANT — the scored field is untouched. `_sweep_fates`'s survivor G grid
     (`G2`) is bit-identical to the unextended deployed screen `LF.dense_grid`'s G grid at
     every scale (same geometry, same NaN/survivor mask, same G values). Both call the same
     `featurize` / `_v2_drop` / `clf.decision_function` in the same position order, so the
     equality is exact, not approximate.

  2. ACCEPT VERDICTS — the accept box set from the EXTENDED path (`screen_field`'s `kept`
     peaks filtered `G >= cutoff`) equals, box-for-box and G-for-G, the accept set built by
     the DEPLOYED harvest path (`HT._all_peaks` + `harvest_minibrot`'s cross-scale
     elliptical-separation NMS on `dense_grid` grids, same filter). Checked at a cutoff that
     admits a non-empty accept set (guarding against a vacuous empty==empty pass) and at a
     permissive cutoff that admits every peak.

Parity is model-agnostic BY CONSTRUCTION — both paths apply the identical model to the
identical survivor set — so a deterministic synthetic field plus a real sklearn
(StandardScaler + L1 LogisticRegression) model exercises the exact production arithmetic
without depending on the wiped q4 label corpus or the gitignored field cache. That is what
lets this live in the default suite.

Run:
  uv run pytest tools/studies/test_q4_screen_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.studies import q4_stage1_linear_fit as LF   # noqa: E402  the deployed screen
from tools.studies import q4_harvest_tight as HT       # noqa: E402  the deployed harvest
from tools.studies import q4_multibrot_transfer as MT  # noqa: E402  the extended screen


def _synthetic_field(seed=0, W=176, H=112):
    """A deterministic field (NaN=interior) with mixed fate: a structured high-gradient
    band (survives the v2 pre-filter), a flat quadrant (v2-dropped as `flat`), and an
    interior NaN blob (v2-dropped as `interior`). Values are escape-time-like floats."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # low-frequency ramp + structured ripples over most of the field -> survivors
    field = (0.5 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
             + 0.3 * np.sin((xx + yy) / 4.0)
             + 0.15 * rng.standard_normal((H, W)))
    field += xx / W  # gentle gradient so crops differ
    # a flat quadrant (near-constant -> g_flat high -> masked)
    field[:H // 3, :W // 3] = 0.02 * rng.standard_normal((H // 3, W // 3))
    # an interior blob -> NaN (g_interior high -> masked)
    cy, cx, r = H // 2, 2 * W // 3, min(H, W) // 6
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    field[mask] = np.nan
    return field.astype(np.float64), int(W), int(H)   # deployed screen indexes with int dims


def _fit_synthetic_model(field, fw, fh, seed=0):
    """A real (StandardScaler, L1-LogisticRegression, keys) over the deployed feature set,
    fit on featurized crops of the field. Mirrors LF._fit_logit's estimator config so the
    decision_function arithmetic is the production one; labels are synthetic (both classes
    present) — only the arithmetic path matters for a code-parity check."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    keys = LF.FEATURES[HT.TIER]
    rng = np.random.default_rng(seed + 1)
    H, W = field.shape
    rows = []
    for _ in range(80):
        wp = int(rng.integers(40, 90))
        hp = int(round(wp * 9 / 16))
        if wp >= W or hp >= H:
            continue
        x = int(rng.integers(0, W - wp))
        y = int(rng.integers(0, H - hp))
        f = LF.featurize(field[y:y + hp, x:x + wp])
        if f is not None:
            rows.append(f)
    assert len(rows) >= 20, "synthetic field produced too few featurizable crops"
    X = np.array([[r[k] for k in keys] for r in rows], dtype=np.float64)
    # a deterministic 2-class split on a real feature so the fit is well-posed
    thr = np.median(X[:, keys.index("g_high")])
    y = (X[:, keys.index("g_high")] > thr).astype(int)
    if y.min() == y.max():                       # degenerate split guard
        y = (np.arange(len(y)) % 2)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=HT.C,
                             class_weight="balanced", max_iter=2000, random_state=0)
    clf.fit(sc.transform(X), y)
    return (sc, clf, keys)


def _box(c):
    """The accept-verdict identity of one framing: scale + center + window size + G."""
    return (float(c["scale"]), float(c["cu"]), float(c["cv"]),
            float(c["wu"]), float(c["wv"]), float(c["G"]))


def _deployed_kept(grids, fw, fh):
    """The PRE-EXTENSION deployed kept-framing set: HT._all_peaks on each precomputed
    LF.dense_grid G grid, then harvest_minibrot's cross-scale elliptical-separation NMS +
    PER_MB_CAP. NMS block copied verbatim from q4_harvest_tight.harvest_minibrot (91-117);
    `grids` is {scale: dense_grid(...)} so the field is swept exactly once."""
    peaks = []
    for s, res in grids.items():
        if res is None:
            continue
        gx, gy, G, (Wp, Hp) = res
        for (iy, ix, gv) in HT._all_peaks(G):
            peaks.append(dict(scale=s, cu=float(gx[ix]), cv=float(gy[iy]),
                              wu=Wp / fw, wv=Hp / fh, G=gv))
    peaks.sort(key=lambda c: -c["G"])
    kept = []
    for c in peaks:
        clash = False
        for k in kept:
            du = (c["cu"] - k["cu"]) / (0.5 * (c["wu"] + k["wu"]))
            dv = (c["cv"] - k["cv"]) / (0.5 * (c["wv"] + k["wv"]))
            if du * du + dv * dv < HT.SEP * HT.SEP:
                clash = True
                break
        if not clash:
            kept.append(c)
        if len(kept) >= HT.PER_MB_CAP:
            break
    return kept


def _accepts(kept, cutoff):
    return sorted(_box(c) for c in kept if c["G"] >= cutoff)


# --------------------------------------------------------------------------- #
# Shared inputs. Both tests need the same field, the same fitted model, and the same
# `LF.dense_grid` sweep at every scale — and a dense sweep is ~2.2s per scale (one
# `featurize` per sliding window), which was the whole cost of this file. The field is
# deterministic (seeded) and the model fit is a pure function of it, so a module-scoped
# build is the same object each test would have built for itself. Nothing here is
# mutated by either test: `dense_grid` returns fresh arrays and both tests only read.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic():
    field, fw, fh = _synthetic_field()
    return field, fw, fh, _fit_synthetic_model(field, fw, fh)


@pytest.fixture(scope="module")
def deployed_grids(synthetic):
    """{scale: LF.dense_grid(...)} — the UNEXTENDED deployed screen's output, which is
    the reference side of both parity assertions."""
    field, fw, fh, model = synthetic
    return {s: LF.dense_grid(field, fw, fh, s, model) for s in LF.FIELD_SCALES}


def test_sweep_fates_G_grid_bit_identical_to_dense_grid(synthetic, deployed_grids):
    """Root invariant: the extended sweep's scored field == the deployed screen's, exactly,
    at every deployed scale (shape, NaN/survivor mask, and finite G values)."""
    field, fw, fh, model = synthetic
    checked_finite = 0
    for s in LF.FIELD_SCALES:
        dg = deployed_grids[s]
        fr = MT._sweep_fates(field, fw, fh, s, model)
        assert (dg is None) == (fr is None), f"scale {s}: one path returned None, the other did not"
        if dg is None:
            continue
        _, _, G_dep, _ = dg
        G_ext = fr["G2"]
        assert G_dep.shape == G_ext.shape, f"scale {s}: grid shape differs"
        # identical NaN (masked/too-small) pattern
        assert np.array_equal(np.isnan(G_dep), np.isnan(G_ext)), f"scale {s}: survivor mask differs"
        # identical finite G values, exactly (same ops, same order)
        assert np.array_equal(G_dep, G_ext, equal_nan=True), f"scale {s}: survivor G values differ"
        checked_finite += int(np.isfinite(G_dep).sum())
    assert checked_finite > 0, "no survivors scored at any scale — test field is vacuous"


def test_accept_verdicts_bit_identical_across_the_extension(synthetic, deployed_grids):
    """The deployed accept box set (dense_grid + harvest NMS) equals the extended accept box
    set (screen_field), box-for-box and G-for-G, at a non-empty cutoff and a permissive one.
    The field is swept once per path: `grids` for the deployed side, one `screen_field` call
    for the extended side."""
    field, fw, fh, model = synthetic

    grids = deployed_grids
    dep_kept = _deployed_kept(grids, fw, fh)
    ext_kept = MT.screen_field(field, fw, fh, model, 0.0, assert_once=False)["kept"]

    all_G = [g for res in grids.values() if res is not None
             for (_, _, g) in HT._all_peaks(res[2])]
    assert all_G, "no peaks on the synthetic field — cannot exercise the accept filter"
    admits_some = float(np.percentile(all_G, 40.0))   # a cutoff that keeps a non-empty subset
    admits_all = float(min(all_G)) - 1.0              # a cutoff that keeps everything

    for cutoff in (admits_some, admits_all):
        dep = _accepts(dep_kept, cutoff)
        ext = _accepts(ext_kept, cutoff)
        assert dep == ext, (
            f"accept verdicts moved under the extension at cutoff {cutoff:.4f}: "
            f"deployed {len(dep)} vs extended {len(ext)}")

    # guard against a vacuous empty==empty pass: the non-empty cutoff must admit >=1 accept
    assert _accepts(ext_kept, admits_some), (
        "the 'non-empty' cutoff admitted zero accepts — parity check was vacuous")


if __name__ == "__main__":
    _field, _fw, _fh = _synthetic_field()
    _syn = (_field, _fw, _fh, _fit_synthetic_model(_field, _fw, _fh))
    _grids = {s: LF.dense_grid(_field, _fw, _fh, s, _syn[3]) for s in LF.FIELD_SCALES}
    test_sweep_fates_G_grid_bit_identical_to_dense_grid(_syn, _grids)
    test_accept_verdicts_bit_identical_across_the_extension(_syn, _grids)
    print("PASS  stage-1 screen extension is additive-only (G grid + accept verdicts identical)")
