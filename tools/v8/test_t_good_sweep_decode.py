"""The t_good SWEEP's admission predicate must be the SERVED one, elementwise.

`scoring/derive_t_good.keeper_pred` is a vectorized twin of `score_lib.corn_decode(...) >= 3`.
It exists only because LOO-OOF makes O(n^2 * |GRID|) predictions and a per-row call into
`corn_decode` is the wrong shape for that — but a twin is a duplication, and `corn_decode`'s
own docstring says in as many words: "reuse it, don't reimplement the >= threshold counting
inline". So the duplication gets a real check rather than a comment.

What was wrong before 2026-08-02: the sweep searched `(p_ge2 >= 0.5) & (p_ge3 >= t)`, an
AND. The served rule COUNTS thresholds met — `class = 1 + #{p_ge2>=0.5, p_ge3>=t,
p_ge4>=0.5}` — and on a K=4 head (v8 onward) those are different predicates. They differ on
rows where the count reaches 2 without the `p_ge3` leg, which requires the cumulative
probabilities to be non-monotone; CORN does not guarantee monotonicity (the check in
`tools/v6/threshold_sweep.py`) and the live slice does contain such rows.

**Measured 2026-08-02** over `data/v10/eval_scores_v10.jsonl` (760 rows), by re-running
`build_table` under both predicates:

  * 92 rows (12%) are non-monotone, `p_ge4 > p_ge3`; 28 rows have `p_ge4 >= 0.5`, and none
    of those has `p_ge2 < 0.5` — so on this slice only the first divergence case can fire.
  * the two predicates disagree at **68 of the 97 grid points**, on up to 22 rows at once
    (worst at t=0.98). "The gap is zero" is a statement about the ADOPTED cuts, not about
    the rules: at t = 0.03 / 0.03 / 0.06 / 0.27 the disagreement is exactly 0 rows.
  * so the **v10 adopted table is byte-identical** under both — mandelbrot 0.03,
    julia:multibrot{3,4,5} 0.27/0.03/0.06 either way. No live number moves.
  * two *reported* counterfactual (non-adopted) F0.5 argmaxes do move, and are re-derived
    into the artifact: julia:multibrot3 0.47 -> 0.48, julia:multibrot5 0.55 -> 0.06.

**And it is NOT free on the v8 slice** — see
`test_the_v8_anchor_is_the_known_divergence_and_is_left_alone` below. That is the finding
this alignment turned up, not a side effect of it.

Run:  uv run python -m pytest tools/v8/test_t_good_sweep_decode.py -q
"""
import json
from collections import Counter
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import derive_t_good as est             # noqa: E402
from partitions import partition_of     # noqa: E402
from production_pins import ACTIVE_VERSION   # noqa: E402
from score_lib import corn_decode       # noqa: E402

# Coupled to production_pins.ACTIVE_CKPT: `pytest -m version_pinned` lists it.
pytestmark = pytest.mark.version_pinned



def _served(nb, gd, gr, t):
    """The rule as production runs it, one row at a time, through the real function."""
    return corn_decode(nb, gd, t, gr, est.T_GREAT) >= est.KEEPER_CLASS


# --------------------------------------------------------------------------- #
# Agreement on constructed rows, including the cases where an AND would differ.
# --------------------------------------------------------------------------- #
# (p_ge2, p_ge3, p_ge4, t) -> the fixture spans both sides of all three cutpoints, and the
# last four rows are the DIVERGENCE cases: an AND rejects them, the served rule admits.
FIXTURE = [
    # ordinary, monotone rows — AND and counting agree
    (0.90, 0.80, 0.70, 0.50),      # all three met
    (0.90, 0.60, 0.30, 0.50),      # two met, the p_ge3 leg among them
    (0.90, 0.20, 0.10, 0.50),      # nb only -> class 2, not a keeper
    (0.10, 0.05, 0.01, 0.50),      # nothing met -> class 1
    (0.90, 0.03, 0.02, 0.03),      # a LOW t (the live mandelbrot cut) admits
    (0.4999, 0.4999, 0.4999, 0.50),  # just below every cutpoint
    (0.50, 0.50, 0.50, 0.50),      # exactly ON every cutpoint (>= is inclusive)
    # NON-MONOTONE rows: p_ge4 above p_ge3. These are where the predicates split.
    (0.90, 0.10, 0.60, 0.50),      # nb + great, no good  -> served: keeper. AND: no.
    (0.40, 0.80, 0.60, 0.50),      # good + great, no nb  -> served: keeper. AND: no.
    (0.90, 0.02, 0.99, 0.03),      # same at the live low t
    (0.10, 0.10, 0.99, 0.50),      # great alone -> count 1 -> class 2, NOT a keeper
]


@pytest.mark.parametrize("nb,gd,gr,t", FIXTURE)
def test_vectorized_twin_equals_the_served_decode(nb, gd, gr, t):
    got = bool(est.keeper_pred(np.array([nb]), np.array([gd]), np.array([gr]), t)[0])
    assert got == _served(nb, gd, gr, t), (nb, gd, gr, t)


def test_the_fixture_actually_contains_divergence_cases():
    """Non-vacuity: if every fixture row agreed under BOTH rules the test above would pass
    against the old AND too, and would be pinning nothing."""
    diverge = [row for row in FIXTURE
               if _served(*row) != bool((row[0] >= est.NB_GATE) and (row[1] >= row[3]))]
    assert len(diverge) >= 3, f"fixture lost its divergence cases: {diverge}"


def test_k3_slice_decodes_exactly_as_the_historical_and():
    """`p_ge4=None` is the K=3 decode, and on K=3 counting and chaining ARE the same rule —
    so re-deriving a v5..v7 slice through this estimator is byte-unchanged."""
    nb = np.array([0.9, 0.9, 0.4, 0.4, 0.5])
    gd = np.array([0.8, 0.2, 0.8, 0.2, 0.5])
    for t in (0.03, 0.5, 0.97):
        counting = est.keeper_pred(nb, gd, None, t)
        chaining = (nb >= est.NB_GATE) & (gd >= t)
        assert np.array_equal(counting, chaining), t


# --------------------------------------------------------------------------- #
# Agreement on the LIVE eval slice, and the measured size of the change.
# --------------------------------------------------------------------------- #
def _live_rows():
    p = ROOT / "data" / ACTIVE_VERSION / f"eval_scores_{ACTIVE_VERSION}.jsonl"
    if not p.exists():
        pytest.skip(f"{p} absent (bulk/relocated checkout)")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_sweep_and_served_agree_on_every_row_of_the_live_slice():
    """The fixture proves the twin is the same FUNCTION; this proves it on the actual
    population the threshold is chosen from, at every grid point the sweep visits."""
    rows = _live_rows()
    nb = np.array([r[f"{ACTIVE_VERSION}_p_ge2"] for r in rows])
    gd = np.array([r[f"{ACTIVE_VERSION}_p_ge3"] for r in rows])
    gr = est.great_column(rows, ACTIVE_VERSION)
    assert gr is not None, f"{ACTIVE_VERSION} is K=4 — the p_ge4 column must be there"
    for t in est.GRID:
        vec = est.keeper_pred(nb, gd, gr, t)
        ref = np.array([_served(a, b, c, t) for a, b, c in zip(nb, gd, gr)])
        assert np.array_equal(vec, ref), f"twin diverges from corn_decode at t={t}"


# The measured served-vs-AND gap on the LIVE slice, per version. THE POINT OF THIS NUMBER IS
# THAT IT IS NOT ZERO ANY MORE, and it stopped being zero for a reason worth writing down.
#
# Under v10 it WAS zero, and this test asserted that. That was a fact about v10's ADOPTED
# CUTS, not about the rules: v10's table was very loose (mandelbrot t=0.03, julia:multibrot4
# t=0.03), so essentially every row that met `p_ge4 >= 0.5` had already met `p_ge3 >= t`, and
# the counting rule and the AND could not come apart. v11's table is strict (mandelbrot 0.90,
# julia:mandelbrot 0.85, phoenix 0.77), so there is now real room between the two legs.
#
# EVERY DIVERGENCE IS `served_only` — admitted by the SERVED rule, refused by the AND — and
# that is the expected direction: counting reaches 2 via `p_ge4 >= 0.5` without the `p_ge3`
# leg. These are locations the head is confident are class 4 while being unsure they are
# class 3, which CORN's cumulative probabilities do not forbid. The served rule is
# `corn_decode`, so these 34 rows admit in production; the sweep that chose the thresholds
# swept the same rule, so the cuts already account for them.
LIVE_ALIGNMENT_GAP = {
    "v10": {"total": 0, "by_partition": {}},
    "v11": {"total": 34, "by_partition": {
        "mandelbrot": 13, "julia:mandelbrot": 8, "phoenix": 5, "multibrot3": 4,
        "julia:multibrot3": 2, "multibrot4": 2}},
}


def test_the_alignment_gap_on_the_live_slice_is_the_recorded_one():
    """The served-vs-AND gap, MEASURED and pinned per version rather than asserted zero.

    A moving number pinned to its measurement is the honest instrument here. Asserting zero
    was right while it was a live invariant; keeping that assertion after v11's cuts tightened
    would have meant either a red build for a correct change, or quietly deleting the check.
    What must not move without someone looking is the COUNT and its DIRECTION — a divergence
    in the other direction (`and_only`: refused by the served rule, admitted by the AND) would
    mean a row met `p_ge3 >= t` and `p_ge2 >= 0.5` and still decoded under 3, which cannot
    happen and would indicate the twin has drifted from `corn_decode`."""
    rows = _live_rows()
    doc = json.loads((ROOT / "data" / ACTIVE_VERSION / "t_good_derivation.json")
                     .read_text(encoding="utf-8"))
    adopted, baseline = doc["adopted"], doc["baseline"]
    served_only, and_only = Counter(), Counter()
    for r in rows:
        fam = est.fam_of(r)
        t = adopted.get(fam, baseline)
        nb, gd = r[f"{ACTIVE_VERSION}_p_ge2"], r[f"{ACTIVE_VERSION}_p_ge3"]
        gr = r[f"{ACTIVE_VERSION}_p_ge4"]
        a, b = _served(nb, gd, gr, t), bool(nb >= est.NB_GATE and gd >= t)
        if a and not b:
            served_only[fam] += 1
        elif b and not a:
            and_only[fam] += 1

    assert not and_only, (
        f"{sum(and_only.values())} rows are admitted by the superseded AND and REFUSED by the "
        f"served counting rule ({dict(and_only)}) — that direction is impossible if the twin "
        f"still tracks corn_decode, so read this as the twin having drifted, not as a "
        f"threshold question")

    expected = LIVE_ALIGNMENT_GAP.get(ACTIVE_VERSION)
    assert expected is not None, (
        f"no recorded alignment gap for {ACTIVE_VERSION} — measure it and add a "
        f"LIVE_ALIGNMENT_GAP entry rather than deleting this check")
    assert (sum(served_only.values()), dict(served_only)) ==            (expected["total"], expected["by_partition"]), (
        f"{ACTIVE_VERSION} served-vs-AND gap is {sum(served_only.values())} "
        f"{dict(served_only)}, recorded {expected['total']} {expected['by_partition']} — "
        f"the adopted table or the slice moved; re-read both, then update the record")


# --------------------------------------------------------------------------- #
# The v8 anchor: where the alignment is NOT free, pinned so a rollback finds it here.
# --------------------------------------------------------------------------- #
def test_the_v8_anchor_is_the_known_divergence_and_is_left_alone():
    """v8's committed mandelbrot t_good was derived under the AND, and the served rule
    picks a different one. This is the finding, so it is a test and not a comment.

    Re-deriving `data/v8/eval_scores_v8.jsonl` through the aligned estimator moves
    mandelbrot **0.85 -> 0.14**; the other three partitions are unchanged. The cause is 8
    of v8's 526 mandelbrot eval rows that have `p_ge4 >= 0.5` with `p_ge3 < 0.85` — under
    the rule that actually served they are keepers, and admitting them at a cut that high
    is what collapses the F0.5 argmax. So v8's live mandelbrot cut was chosen against a
    predicate the gate did not run. (Hiding the `p_ge4` column reproduces 0.85 exactly,
    which is how the move is attributed to the predicate and to nothing else.)

    **The v8 artifact is deliberately NOT re-derived.** 0.85 is what v8 actually served
    while v8 was live, and a record keeps what was true when it was written. What must not
    happen is discovering this during a rollback: v8 is the one-flip rollback anchor, and
    `tools/scoring/test_t_good_adoption.py::test_the_derivation_reruns_to_the_committed_numbers`
    would go red on mandelbrot the moment ACTIVE_VERSION became v8. This test is that
    warning, delivered now. A rollback to v8 must re-derive its table, not copy it.
    """
    import contextlib
    import io

    p = ROOT / "data" / "v8" / "eval_scores_v8.jsonl"
    doc = ROOT / "data" / "v8" / "t_good_derivation.json"
    if not (p.exists() and doc.exists()):
        pytest.skip("v8 slice/derivation absent (bulk/relocated checkout)")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    committed = json.loads(doc.read_text(encoding="utf-8"))["adopted"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        served = est.build_table(rows, version="v8", eval_slice="data/v8/eval_scores_v8.jsonl")
        # the same slice with the third cutpoint hidden IS the superseded AND
        no_g4 = [{k: v for k, v in r.items() if k != "v8_p_ge4"} for r in rows]
        anded = est.build_table(no_g4, version="v8", eval_slice="data/v8/eval_scores_v8.jsonl")

    assert anded["adopted"] == committed, (
        "v8's committed table no longer reproduces even under the AND — the divergence "
        "below would then not be attributable to the predicate")
    assert served["adopted"]["mandelbrot"] == 0.14 and committed["mandelbrot"] == 0.85, (
        f"the v8 mandelbrot divergence changed: served {served['adopted']['mandelbrot']} vs "
        f"committed {committed['mandelbrot']} (was 0.14 vs 0.85 on 2026-08-02)")
    for fam in ("julia:multibrot3", "julia:multibrot4", "julia:multibrot5"):
        assert served["adopted"][fam] == committed[fam], fam

    mb = [r for r in rows if r["fractal_type"] == "mandelbrot"]
    flippers = [r for r in mb if r["v8_p_ge4"] >= est.T_GREAT and r["v8_p_ge3"] < 0.85]
    assert len(flippers) == 8, f"{len(flippers)} rows drive the v8 move, was 8 on 2026-08-02"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
