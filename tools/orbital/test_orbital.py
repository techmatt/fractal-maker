"""Acceptance for the orbital-falloff criterion
(`prompts/orbital_falloff_criterion.md`).

The measures exist to be *falsifiable*, so most of these tests pin outcomes rather than
implementation — including the outcomes that came back negative. A measure that fails
validation is a result, and these tests make sure it stays reported as one instead of
being quietly tuned until it passes.

Run:  uv run python -m pytest tools/orbital/test_orbital.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "descent", REPO_ROOT / "tools" / "sources",
          REPO_ROOT / "tools" / "explorer", REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import field_metrics as fm     # noqa: E402

MEASURES = REPO_ROOT / "data" / "orbital" / "measures.jsonl"
VALIDATION = REPO_ROOT / "data" / "orbital" / "validation.json"


# --------------------------------------------------------------------------- #
# the shading constant the whole criterion rests on
# --------------------------------------------------------------------------- #
def test_density_matches_the_render_path():
    """`coloring::shade` computes t = smooth_iter*density + offset with density fixed at
    the ShadeArgs default. If that default ever moves, every ring count here silently
    changes meaning, so it is pinned against the Rust source."""
    cli = (REPO_ROOT / "src" / "cli.rs").read_text(encoding="utf-8")
    i = cli.index("pub struct ShadeArgs")
    seg = cli[i:i + 400]
    assert "default_value_t = 0.025" in seg, "ShadeArgs::density default moved"
    assert fm.DENSITY == 0.025 and fm.CYCLE_ITERS == 40.0


# --------------------------------------------------------------------------- #
# measures on synthetic fields — behaviour, not magic numbers
# --------------------------------------------------------------------------- #
def _radial_ramp(h=180, w=320, inner=4000.0, outer=100.0, power=1.0):
    """A synthetic minibrot-ish field: high smooth_iter at the centre falling outward."""
    fy = (np.arange(h) + 0.5) / h - 0.5
    fx = (np.arange(w) + 0.5) / w - 0.5
    r = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * (h / w)) ** 2)
    r = r / r.max()
    return (inner + (outer - inner) * r ** power).astype("f4")


def test_cycles_spanned_counts_colour_cycles():
    f = _radial_ramp(inner=4000.0, outer=100.0)
    got = fm.cycles_spanned(f)
    v = f[np.isfinite(f)]
    want = (np.percentile(v, 95) - np.percentile(v, 5)) * fm.DENSITY
    assert got == pytest.approx(want, rel=1e-6)
    # a flat field spans nothing
    assert fm.cycles_spanned(np.full((180, 320), 300.0, dtype="f4")) == pytest.approx(0.0)


def test_radial_rings_scales_with_dynamic_range():
    """More iterations across the frame = more rings crossed going out. This is the
    whole mechanism: one cycle is 40 iterations regardless of depth."""
    lo = fm.radial_rings(_radial_ramp(inner=300.0, outer=100.0))[0]
    hi = fm.radial_rings(_radial_ramp(inner=8000.0, outer=100.0))[0]
    assert hi > lo * 5, (lo, hi)
    assert fm.radial_rings(np.full((180, 320), 300.0, dtype="f4"))[0] == 0


def test_radial_rings_ignores_interior_but_counts_the_rest():
    """A black island in the middle costs its own span and nothing more — NaN breaks a
    ray into segments and crossings are counted within segments."""
    f = _radial_ramp(inner=8000.0, outer=100.0)
    base = fm.radial_rings(f)[0]
    g = f.copy()
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    g[((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.12 * w) ** 2] = np.nan
    holed = fm.radial_rings(g)[0]
    assert holed > 0 and holed <= base


def test_falloff_extent_is_wide_for_a_slow_ramp_and_narrow_for_a_skin():
    slow = fm.falloff_extent(_radial_ramp(inner=4000.0, outer=100.0, power=1.0))
    # a "thin skin": high only in a narrow annulus, background everywhere else
    f = np.full((180, 320), 100.0, dtype="f4")
    fy = (np.arange(180) + 0.5) / 180 - 0.5
    fx = (np.arange(320) + 0.5) / 320 - 0.5
    r = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * (180 / 320)) ** 2)
    f[r < 0.05] = 4000.0
    skin = fm.falloff_extent(f)
    assert slow > skin, (slow, skin)


def test_interior_profile_is_a_fraction_and_a_radial_curve():
    f = _radial_ramp()
    f[:20, :] = np.nan
    frac, prof = fm.interior_profile(f)
    assert 0 < frac < 1 and len(prof) == 8 and all(0 <= x <= 1 for x in prof)


# --------------------------------------------------------------------------- #
# the recorded validation outcomes (§2) — including the failures
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not VALIDATION.exists(), reason="validation not run")
def test_radial_rings_separates_both_references_from_all_triage_atoms():
    """The measure that survived: both references rank above ALL 200 triage atoms."""
    v = json.loads(VALIDATION.read_text(encoding="utf-8"))["measures"]["radial_rings"]
    assert v["refs_above_all_triage"] is True
    assert v["triage_atoms_at_or_above_eye"] == 0
    assert v["triage_atoms_at_or_above_mb19"] == 0


@pytest.mark.skipif(not VALIDATION.exists(), reason="validation not run")
def test_the_minibroteye_test_is_not_depth_in_disguise():
    """`minibroteye` is shallow (fw 5.8e-4, and not even a nucleus) while `mb19_p35` is
    at 8e-10. A measure that ranked the eye low would just be depth wearing a disguise.
    The eye must score at least as high as mb19."""
    v = json.loads(VALIDATION.read_text(encoding="utf-8"))["measures"]["radial_rings"]
    assert v["eye"] >= v["mb19"], (v["eye"], v["mb19"])


@pytest.mark.skipif(not MEASURES.exists(), reason="measures not run")
def test_radial_rings_is_only_weakly_correlated_with_depth():
    """The population-level version of the same check: if the measure were depth in
    disguise it would track log10|A| tightly. It does not."""
    rows = [json.loads(l) for l in MEASURES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("log10_abs_A") is not None]
    la = np.array([r["log10_abs_A"] for r in rows])
    rr = np.array([r["radial_rings"] for r in rows])
    rho = np.corrcoef(np.argsort(np.argsort(la)), np.argsort(np.argsort(rr)))[0, 1]
    assert abs(rho) < 0.6, f"spearman {rho:+.3f} — too close to being depth itself"


@pytest.mark.skipif(not VALIDATION.exists(), reason="validation not run")
def test_the_measures_that_failed_are_recorded_as_failed():
    """`cycles_spanned` and `falloff_extent` did NOT separate the references from the
    triage atoms. Pinned so the negative results stay reported rather than being tuned
    away: if one of these ever passes, that is a real change worth re-reading, not a
    silent improvement."""
    m = json.loads(VALIDATION.read_text(encoding="utf-8"))["measures"]
    assert m["cycles_spanned"]["separates"] is False
    assert m["falloff_extent"]["separates"] is False
    assert m["falloff_extent"]["triage_atoms_at_or_above_eye"] > 50


# --------------------------------------------------------------------------- #
# the iteration-CAP provenance axis (docs/design/auto_maxiter.md)
#
# `tools/orbital/` sizes every field with `rc.auto_maxiter(fw)`, which reads the LIVE
# production constants — so this stack followed the 2026-07-31 cap raise silently.
# These brackets pin all three halves: the token rides on the record, a same-policy
# comparison is unaffected, a cross-policy one raises, and the guard is actually
# REACHED on the live path rather than merely defined.
# --------------------------------------------------------------------------- #
POOL = REPO_ROOT / "data" / "orbital" / "screen_pool.jsonl"
SCREEN_SCORES = REPO_ROOT / "data" / "orbital" / "screen_scores.jsonl"


def _rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_missing_token_reads_as_legacy():
    """The load-bearing back-compat invariant, same as the field-cache stems: a record
    written before this axis existed is LEGACY, not 'unknown'. If this flipped, every
    pre-2026-07-31 record would start raising against itself."""
    assert fm.record_policy({}) == fm.LEGACY_POLICY_TOKEN == ""
    assert fm.record_policy({fm.POLICY_KEY: ""}) == fm.LEGACY_POLICY_TOKEN
    assert "legacy" in fm.describe_policy("").lower()
    # ...and the live policy is NOT legacy, or this whole axis is vacuous.
    assert fm.policy_token() != fm.LEGACY_POLICY_TOKEN


def test_same_policy_comparison_is_unaffected():
    """Half one of the bracket: within one policy nothing changed. Two same-token
    groups pool cleanly and return that token — including the all-legacy case, which
    is what every committed file is."""
    legacy = [{"id": "a"}, {"id": "b", fm.POLICY_KEY: ""}]
    assert fm.require_one_policy(("refs", legacy[:1]), ("triage", legacy[1:])) == ""
    live = fm.policy_token()
    now = [{"id": "c", fm.POLICY_KEY: live}, {"id": "d", fm.POLICY_KEY: live}]
    assert fm.require_one_policy(("a", now), ("b", now)) == live
    assert fm.require_one_policy([]) == fm.policy_token()      # empty pools cleanly


def test_cross_policy_comparison_raises_naming_both():
    """Half two: mixing raises, and the message NAMES both policies — an error that
    says only 'policy mismatch' would leave you guessing which side to re-measure."""
    old = [{"id": "old1"}, {"id": "old2"}]
    new = [{"id": "new1", fm.POLICY_KEY: fm.policy_token()}]
    with pytest.raises(fm.MaxiterPolicyMixError) as ei:
        fm.require_one_policy(("committed", old), ("this run", new),
                              what="the separation verdict")
    msg = str(ei.value)
    assert "legacy" in msg and fm.policy_token() in msg
    assert "committed" in msg and "this run" in msg
    assert "2 record(s)" in msg and "1 record(s)" in msg
    assert "the separation verdict" in msg


def test_the_guard_is_reached_on_the_live_validate_path():
    """Defining a guard nobody calls is the failure mode this test exists to refuse.
    Drive the REAL `measure_atoms.validate` — the function that emits validation.json —
    with a reference measured under one policy and triage under another, and assert it
    refuses. Rows carry every field validate reads, so it fails at the GUARD, not
    incidentally on a KeyError."""
    import measure_atoms as ma

    def row(i, groups, tok=None):
        r = {"id": i, "groups": groups, "label": i,
             **{k: 5.0 for k in ma.MEASURES}}
        if tok is not None:
            r[fm.POLICY_KEY] = tok
        return r

    ref, tri = row("r", ["reference"]), row("t", ["triage"])
    assert ma.validate([ref, tri], log=lambda *_: None)[fm.POLICY_KEY] == ""  # same: fine
    with pytest.raises(fm.MaxiterPolicyMixError):
        ma.validate([ref, row("t", ["triage"], fm.policy_token())], log=lambda *_: None)


def test_the_guard_is_reached_on_the_live_screen_resume_path(tmp_path, monkeypatch):
    """The other live path, and the one that actually bites: `screen()` RESUMES from an
    existing scores file and appends. Across the raise that silently writes one file
    holding two populations, so the guard must fire on the resume load — before the
    screening budget is spent, not after.

    Driven against a planted scores file rather than the committed one, so the bracket
    keeps testing the guard after the day `screen_scores.jsonl` is legitimately
    re-measured under the live policy."""
    import paths
    import screen_pool as sp

    fake = tmp_path / "screen_scores.jsonl"
    monkeypatch.setattr(paths, "durable", lambda rel, **kw: fake)

    # (a) resumed rows are LEGACY, this run is live -> refuse, before any screening.
    fake.write_text(json.dumps({"id": "old", "radial_rings": 5.0}) + "\n", encoding="utf-8")
    calls = []
    with pytest.raises(fm.MaxiterPolicyMixError) as ei:
        sp.screen([], log=lambda *a: calls.append(a))
    assert calls, "guard must fire AFTER the resume log line, i.e. on the real path"
    assert "legacy" in str(ei.value) and fm.policy_token() in str(ei.value)

    # (b) same file stamped with THIS run's policy -> resumes normally, no raise.
    fake.write_text(json.dumps({"id": "old", "radial_rings": 5.0,
                                fm.POLICY_KEY: fm.policy_token()}) + "\n", encoding="utf-8")
    scored, errs, _ = sp.screen([], log=lambda *_: None)
    assert [r["id"] for r in scored] == ["old"] and not errs


def test_committed_score_records_are_stamped():
    """Every committed orbital SCORE record states the policy it was computed under.
    All of them are legacy: git puts these files at 2026-07-30, the raise at 07-31."""
    import stamp_cap_policy as scp
    assert scp.audit(check=True)["jsonl"], "nothing audited — paths moved?"
    for rel, s in scp.audit(check=True)["jsonl"].items():
        assert s["needed_stamp"] == 0, f"{rel}: {s['needed_stamp']} unstamped rows"
    for rel in (MEASURES, SCREEN_SCORES):
        if rel.exists():
            assert all(fm.record_policy(r) == "" for r in _rows(rel))


@pytest.mark.skipif(not POOL.exists(), reason="pool not enumerated")
def test_the_enumeration_is_not_stamped_with_a_cap_policy():
    """`screen_pool.jsonl` is the ENUMERATION, not a score: Newton nuclei from
    `atom_lib.solve_nucleus` (mpmath), whose fields are analytic properties of the atom.
    Nothing on that path renders a field or reads an iteration cap, so a cap token there
    would be a FALSE provenance claim. Pinned in both directions — the day someone adds
    a rendered quantity to the pool, this goes red and the disposition gets re-decided
    on purpose."""
    rows = _rows(POOL)
    assert rows and not any(fm.POLICY_KEY in r for r in rows)
    assert set(rows[0]) == {"id", "period", "cx", "cy", "window_scale", "family",
                            "degree", "log10_abs_A", "f64_margin_deploy_decades"}


# --------------------------------------------------------------------------- #
# screening resolution
# --------------------------------------------------------------------------- #
def test_screen_geometry_is_much_cheaper_than_measure_geometry():
    a = fm.SCREEN_W * fm.SCREEN_H * fm.SCREEN_SS ** 2
    b = fm.MEASURE_W * fm.MEASURE_H * fm.MEASURE_SS ** 2
    assert a * 20 < b, "the screen must be far cheaper than the full measure"


def test_measure_keeps_no_field_files(tmp_path):
    """Field dumps are transient — 10k screening renders must not leave 10k .bin files."""
    before = set(tmp_path.rglob("*"))
    fm.measure_location("-0.746339", "0.112242", 5.83e-4, 500,
                        width=fm.SCREEN_W, height=fm.SCREEN_H, ss=fm.SCREEN_SS,
                        tmpdir=str(tmp_path))
    assert set(tmp_path.rglob("*")) == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
