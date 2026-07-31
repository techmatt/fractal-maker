"""Coverage for `rescore_lib` and `measure_convergence_ladder` — the two orbital modules
that had none, and for `radial_range`, which had no test at all.

Both modules were PROMOTED out of `scratch/rescore/` on 2026-07-31 because one of them is
the sole producer of a committed artifact. Promotion moved the code; it did not bring a
suite with it, and `rescore_lib`'s only assertion lived in a `__main__` self-check that
nothing ran. A test CI never runs and git never sees is a memory of a test.

Everything here is synthetic numpy or pure arithmetic — no engine, no GPU, no corpus
reads, nothing over a few milliseconds. The parts that are NOT covered and why:

  * `ladder_for_atom` / `measure_both` / `dump_field` spawn `fractal-generator.exe` per
    cap step (the ladder is 9 steps x 32 atoms). Covering them means either a live-binary
    test in the mould of `test_measure_keeps_no_field_files`, or mocking the subprocess —
    which would assert the mock. Not cheap, so it is stated rather than implied.
  * `load_reachable_pool` reads the two committed `data/orbital/` jsonl files. Its logic
    is a join plus one `fw >= FW_MIN_320` filter; the interesting half (the filter) is
    covered indirectly by nothing, and pinning it would pin the committed pool's contents.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "explorer", REPO_ROOT / "tools" / "corpus",
          REPO_ROOT / "tools" / "descent", REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import field_metrics as fm                    # noqa: E402
import rescore_lib as rl                      # noqa: E402
import render_core as rc                      # noqa: E402
import measure_convergence_ladder as mcl      # noqa: E402


# --------------------------------------------------------------------------- #
# ring_measures — the equality that used to live in a __main__ self-check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inner", [300.0, 2000.0, 8000.0])
def test_ring_measures_crossings_equal_field_metrics_radial_rings(inner):
    """The single-pass walk must reproduce `fm.radial_rings` EXACTLY, or the two measures
    are not being read off the same rays and the pair's whole premise is gone."""
    f = rl.selfcheck_field(inner)
    want_med, want_p90 = fm.radial_rings(f)
    got = rl.ring_measures(f)
    assert got["radial_rings"] == pytest.approx(want_med, abs=1e-9)
    assert got["radial_rings_p90"] == pytest.approx(want_p90, abs=1e-9)


# --------------------------------------------------------------------------- #
# radial_range — the measure that had no test at all
# --------------------------------------------------------------------------- #
def _ramp(hi: float, *, h=180, w=320) -> np.ndarray:
    """A clean radial ramp: 0 at the centre, `hi` at the inscribed radius, no NaN."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / min(cx, cy)
    return (hi * np.clip(r, 0, 1)).astype("f4")


def test_radial_range_is_the_span_in_colour_cycles():
    """`range` is `max - min` of `smooth_iter * DENSITY` along a ray. A ramp of `hi`
    smooth-iters therefore spans `hi * DENSITY` cycles, independent of the ray count."""
    hi = 4000.0
    m = rl.ring_measures(_ramp(hi))
    assert m["radial_range"] == pytest.approx(hi * rl.DENSITY, rel=0.02)


def test_radial_range_is_flat_for_a_flat_field():
    m = rl.ring_measures(np.full((180, 320), 1234.0, dtype="f4"))
    assert m["radial_range"] == 0.0
    assert m["radial_rings"] == 0.0


def test_radial_range_takes_the_max_segment_not_the_sum():
    """The one deliberate asymmetry with crossings: an interior island splits a ray into
    segments, crossings SUM over them and span takes the MAX. Two identical half-ramps
    either side of a NaN band must therefore read as ONE half-ramp's span, not two."""
    hi = 4000.0
    f = _ramp(hi)
    r = np.sqrt((np.mgrid[0:180, 0:320][1] - 159.5) ** 2
                + (np.mgrid[0:180, 0:320][0] - 89.5) ** 2) / 89.5
    split = f.copy()
    split[(r > 0.45) & (r < 0.55)] = np.nan          # annular NaN band cuts every ray
    whole = rl.ring_measures(f)["radial_range"]
    cut = rl.ring_measures(split)["radial_range"]
    # each surviving segment covers ~half the ramp, so the max segment is ~half the span
    assert cut < whole
    assert cut == pytest.approx(whole * 0.5, rel=0.15)


def test_a_dithering_ray_racks_up_crossings_without_span():
    """The failure mode `range` was added to expose: a field that oscillates across one
    colour boundary accumulates crossings on every transition while its span stays ~one
    cycle. This is the measured basis of the two-axis reading in
    docs/design/orbital_field_metrics.md §5."""
    h, w = 180, 320
    base = 1.0 / rl.DENSITY                            # exactly one colour boundary
    yy, xx = np.mgrid[0:h, 0:w]
    dither = (base + 0.6 / rl.DENSITY * np.sin(0.9 * np.hypot(xx - 159.5, yy - 89.5))
              ).astype("f4")
    m = rl.ring_measures(dither)
    assert m["radial_rings"] > 20, "a dithering ray should accumulate many crossings"
    assert m["radial_range"] < 2.0, "...while spanning barely more than one cycle"


def test_ring_measures_returns_all_four_keys_on_an_all_nan_field():
    """An all-interior frame must return zeros, not raise — `screen_pool` scores whatever
    the enumeration hands it."""
    m = rl.ring_measures(np.full((180, 320), np.nan, dtype="f4"))
    assert m == {"radial_rings": 0.0, "radial_rings_p90": 0.0,
                 "radial_range": 0.0, "radial_range_p90": 0.0}


def test_the_dead_scoring_cap_policy_is_gone():
    """`scoring_maxiter` was deleted on 2026-07-31: no caller, and it returned 8x of the
    RAISED production cap (200000 at fw=8e-10) rather than the 24x-of-legacy envelope the
    scratch evidence was computed under. A dead function returning a wrong number is a
    trap for its first real caller — this pins the deletion so it does not come back by
    reflex."""
    for gone in ("scoring_maxiter", "prod_maxiter", "_cap_params", "CAP_JSON"):
        assert not hasattr(rl, gone), (
            f"rescore_lib.{gone} is back. If a scoring-only cap policy is genuinely "
            f"wanted, it needs a caller and a stated policy, not a fallback multiple.")


# --------------------------------------------------------------------------- #
# measure_convergence_ladder — the cap policy as a PARAMETER
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fw", [3.0, 1e-2, 1e-5, 8e-10, 1e-20])
def test_policy_maxiter_at_the_live_policy_is_production_auto_maxiter(fw):
    """The parameterization must be a strict generalization: at `POLICY_LIVE` it has to
    return exactly what `rc.auto_maxiter` returns, or the default run silently measures
    something other than production."""
    assert mcl.policy_maxiter(fw, mcl.POLICY_LIVE) == rc.auto_maxiter(fw)


def test_policy_maxiter_default_is_live():
    assert mcl.POLICY_LIVE == (rc.MAXITER_BASE, rc.MAXITER_K, rc.MAXITER_MIN, rc.MAXITER_MAX)
    assert mcl.policy_maxiter(1e-6) == mcl.policy_maxiter(1e-6, mcl.POLICY_LIVE)


def test_the_legacy_policy_is_measurable_and_is_the_pre_raise_one():
    """The point of the parameter: the committed ladder's measurement is repeatable. The
    legacy cap must be ~8x smaller than today's at the same fw (base 500 vs 4000) and
    must clamp at 8000."""
    assert mcl.POLICY_LEGACY == (500, 0.30, 200, 8000)
    for fw in (3.0, 1e-5, 8e-10):
        legacy = mcl.policy_maxiter(fw, mcl.POLICY_LEGACY)
        live = mcl.policy_maxiter(fw, mcl.POLICY_LIVE)
        assert legacy < live
        assert live / legacy == pytest.approx(8.0, rel=0.02)
    assert mcl.policy_maxiter(1e-30, mcl.POLICY_LEGACY) == 8000


def test_default_out_is_scratch_and_carries_the_policy_token():
    """A run must not be able to land on the committed artifact by default — that is how
    a legacy-policy measurement gets replaced by a new-policy one under one filename."""
    a = mcl.parse_args([])
    assert a.policy == mcl.POLICY_LIVE
    assert a.out.parent == mcl.SCRATCH_DIR
    assert "mi4000" in a.out.name
    b = mcl.parse_args(["--legacy-policy"])
    assert b.policy == mcl.POLICY_LEGACY
    assert b.out.name.endswith("legacy.json")
    assert b.out.resolve() != mcl.COMMITTED_OUT.resolve()


def test_writing_a_new_policy_over_the_committed_ladder_is_refused():
    """The committed artifact holds the LEGACY ladder the base 500 -> 4000 raise rests on.
    Naming it as `--out` under any other policy must raise, not overwrite."""
    with pytest.raises(SystemExit) as ei:
        mcl.main(["--out", str(mcl.COMMITTED_OUT)])
    assert "refusing to overwrite" in str(ei.value)


def test_explicit_policy_flags_override_the_live_default():
    a = mcl.parse_args(["--policy-base", "500", "--policy-clamp-max", "8000"])
    assert a.policy == (500, mcl.POLICY_LIVE[1], mcl.POLICY_LIVE[2], 8000)
    assert mcl.policy_maxiter(3.0, a.policy) == 500


# --- analyze_ladder: pure, and it decides what "converged" means -------------- #
def _pts(*rings):
    return [{"mult": float(2 ** i), "maxiter": 1000 * 2 ** i, "rings": r}
            for i, r in enumerate(rings)]


def test_analyze_ladder_picks_the_first_step_within_tolerance_of_the_asymptote():
    rec = mcl.analyze_ladder({"points": _pts(38.0, 48.5, 48.5, 57.5, 57.5, 57.5)})
    assert rec["converged"] is True
    assert rec["asymptote"] == 57.5
    assert rec["conv_mult"] == 8.0            # index 3 — the first at the asymptote
    assert rec["rings_at_prod"] == 38.0
    assert rec["clip_ratio"] == pytest.approx(57.5 / 38.0, abs=1e-3)


def test_analyze_ladder_marks_a_still_climbing_atom_not_converged():
    """A ladder still moving at its top step has NOT converged; recording it as converged
    would understate the cap the measurement calls for."""
    rec = mcl.analyze_ladder({"points": _pts(10.0, 40.0, 90.0, 200.0)})
    assert rec["converged"] is False
    assert rec["top_mult_reached"] == 8.0


def test_analyze_ladder_handles_an_atom_that_errored_out_immediately():
    """`ladder_for_atom` appends an `error` point and breaks; fewer than two good points
    must degrade to a stated non-result rather than an IndexError."""
    rec = mcl.analyze_ladder({"points": [{"mult": 1.0, "maxiter": 1000,
                                          "error": "spacing guard"}]})
    assert rec["converged"] is False and rec["conv_maxiter"] is None
    assert rec["note"] == "too few points"


def test_stratified_sample_spreads_over_fw_and_is_seeded():
    """The sample is stratified over log10(fw) deciles, and seeded — two calls on the same
    rows must give the same atoms, or the ladder is not repeatable even at a fixed cap."""
    rows = [{"id": f"a{i}", "window_scale": 10.0 ** (-i / 4.0)} for i in range(60)]
    s1 = mcl.stratified_sample(rows)
    s2 = mcl.stratified_sample(rows)
    assert [r["id"] for r in s1] == [r["id"] for r in s2]
    assert len(s1) == mcl.N_BINS * mcl.N_PER_BIN
    fws = [float(r["window_scale"]) * 4 for r in s1]
    assert max(fws) / min(fws) > 1e10, "the sample must span the pool's fw range"
