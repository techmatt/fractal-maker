#!/usr/bin/env python
"""`tools/emission/floors.py` — the stage-2 cut owner's two load-bearing properties.

  1. A stamped floor REFUSES to gate when its head's live pin has moved off the stamp.
     Refusing is the point: a floor is a point on ONE head's probability scale, so gating
     0.90 on a head that never produced that scale is silent and only visible months later
     as "the release got worse".
  2. The pool floors stay strictly below their release floors, both directions checked, at
     import.

Each test that asserts a refusal also asserts the NON-refusal it is the mirror of, so a
`check()` that raised unconditionally (or never) would fail here rather than read as green.

  uv run pytest tools/emission/test_floors.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import floors as F            # noqa: E402
from tools.mining import mining_pins as MP        # noqa: E402
from tools.wallpaper import wallpaper_pins as WP  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. the stamp
# --------------------------------------------------------------------------- #
@pytest.mark.stage2_pinned
def test_every_cut_is_stamped_against_its_live_head_today():
    """The baseline the refusal tests are the mirror of. Green here means the tree's floors
    and the tree's head pins currently agree — if this ever goes red, a head moved and a
    floor did not, which is the condition the refusal exists for."""
    F.check_stamps()
    assert F.WALLPAPER_POOL.stamp == WP.HEAD_VERSION
    assert F.MINING_POOL.stamp == MP.HEAD_VERSION
    assert {f.head for f in F.ALL_FLOORS} == {F.WALLPAPER_HEAD, F.MINING_HEAD}


def test_a_wallpaper_head_flip_makes_its_floors_refuse(monkeypatch):
    """INJECTION. Move the wallpaper pin to a version no floor was derived against; both
    wallpaper cuts must refuse, and the mining cuts must NOT — a blanket raise would pass a
    one-sided test."""
    monkeypatch.setattr(WP, "HEAD_VERSION", "v9")
    for f in (F.WALLPAPER_POOL, F.WALLPAPER_RELEASE):
        with pytest.raises(F.HeadStampMismatch) as ei:
            f.annotates(0.99)
        assert f.stamp in str(ei.value) and "v9" in str(ei.value)
    assert F.MINING_RELEASE.annotates(0.99) is True     # untouched head still annotates
    with pytest.raises(F.HeadStampMismatch):
        F.check_stamps()


def test_a_mining_head_flip_makes_its_floors_refuse(monkeypatch):
    monkeypatch.setattr(MP, "HEAD_VERSION", "v2")
    for f in (F.MINING_POOL, F.MINING_RELEASE):
        with pytest.raises(F.HeadStampMismatch):
            f.annotates(0.99)
    assert F.WALLPAPER_RELEASE.annotates(0.99) is True


def test_the_refusal_is_at_the_comparison_not_only_at_check(monkeypatch):
    """`annotates()` is THE comparison every consumer calls. If the check only ran in
    `check()`, a site that wrote its own `>=` against `.value` would annotate on the wrong
    scale in silence — which is precisely how the six copies this module replaced behaved."""
    monkeypatch.setattr(WP, "HEAD_VERSION", "v9")
    with pytest.raises(F.HeadStampMismatch):
        F.for_style("smooth", "release").annotates(0.5)


def test_active_head_version_reads_the_pin_at_call_time(monkeypatch):
    """Not an import-time snapshot: a long-lived process that outlives a pin flip must see
    the flip, and a test must be able to induce one."""
    assert F.active_head_version(F.MINING_HEAD) == MP.HEAD_VERSION
    monkeypatch.setattr(MP, "HEAD_VERSION", "v77")
    assert F.active_head_version(F.MINING_HEAD) == "v77"


def test_the_location_head_is_resolvable_even_though_no_cut_uses_it():
    """`FLOOR_PNOTBAD` was the location head's only stage-2 cut and it is deleted (§B), so
    nothing is stamped against it today. The resolver stays registered so the NEXT intake-side
    cut is stampable without re-deriving the mechanism — and an unregistered head raises
    rather than defaulting."""
    import corpus_common as cc
    assert F.active_head_version(F.LOCATION_HEAD) == cc.active_scorer_version()
    with pytest.raises(KeyError):
        F.active_head_version("no_such_head")


# --------------------------------------------------------------------------- #
# 2. the release floors are the heads' gates, not copies of them
# --------------------------------------------------------------------------- #
@pytest.mark.stage2_pinned
def test_a_release_floor_moves_with_its_head_gate(monkeypatch):
    """The release floors are IMPORTED from the head pins, so a gate retune cannot leave the
    release floor calibrated against a head that no longer exists. Asserted on the resolution,
    not the literal: `Floor` is frozen, so this re-resolves the module rather than mutating."""
    assert F.WALLPAPER_RELEASE.value == WP.GATE_THRESHOLD
    assert F.MINING_RELEASE.value == MP.MINING_GATE_THRESHOLD


@pytest.mark.stage2_pinned
def test_pool_floors_sit_below_their_release_floors():
    F.check_below_gate()
    assert F.WALLPAPER_POOL.value < F.WALLPAPER_RELEASE.value
    assert F.MINING_POOL.value < F.MINING_RELEASE.value


@pytest.mark.parametrize("pool_value", [0.90, 0.95])
def test_a_pool_floor_at_or_above_its_gate_is_a_red_build(pool_value):
    """INJECTION on `check_below_gate`, which runs at import. A pool floor at the release
    floor makes 'pooled but not release-grade' zero by construction — every inventory count
    downstream reads as a measurement and is an identity."""
    broken = (F.Floor(name="wallpaper_pool", value=pool_value, head=F.WALLPAPER_HEAD,
                      stamp=WP.HEAD_VERSION, site="pool", basis="injected"),
              F.WALLPAPER_RELEASE)
    with pytest.raises(ValueError, match="not below their release floor"):
        F.check_below_gate(broken)


def test_check_below_gate_is_not_vacuous():
    """The healthy pair passes the same call the broken pair fails."""
    F.check_below_gate((F.WALLPAPER_POOL, F.WALLPAPER_RELEASE))


# --------------------------------------------------------------------------- #
# 3. routing
# --------------------------------------------------------------------------- #
def test_style_routes_to_the_head_that_was_trained_on_it():
    assert F.for_style("smooth", "release") is F.WALLPAPER_RELEASE
    assert F.for_style("smooth", "pool") is F.WALLPAPER_POOL
    for strange in ("tia", "stripe", "composite_c13_smooth_stripe"):
        assert F.for_style(strange, "release") is F.MINING_RELEASE
        assert F.for_style(strange, "pool") is F.MINING_POOL
    with pytest.raises(KeyError):
        F.for_style("smooth", "somewhere_else")


def test_a_Floor_cannot_cut_at_all():
    """`acts` and `gate()` are GONE (2026-08-09, prompts/selection_restructure_3.md), and the
    absence is the census. `acts` had been False on all four since the day before — a field
    whose only legal value is False, sitting next to a method named for flipping it, is an
    invitation rather than a record. What removes rows is the two module-level constants; a
    `Floor` offers `annotates()` and nothing else.

    Asserted on the CLASS, not on the four instances: a fifth Floor added later inherits the
    property instead of needing to be remembered."""
    assert not hasattr(F.Floor, "gate")
    assert hasattr(F.Floor, "annotates")
    assert "acts" not in F.Floor.__dataclass_fields__
    assert F.summary().count("annotation-only") == len(F.ALL_FLOORS)


@pytest.mark.stage2_pinned
def test_the_summary_still_names_an_added_cut(monkeypatch):
    """NON-VACUITY for the census above: the run-banner summary is derived from `ALL_FLOORS`,
    so an added cut appears in it — and appears as annotation-only, because there is no longer
    any other kind."""
    ghost = F.Floor(name="ghost_release", value=0.99, head=F.MINING_HEAD,
                    stamp=MP.HEAD_VERSION, site="release", basis="injected")
    monkeypatch.setattr(F, "ALL_FLOORS", F.ALL_FLOORS + (ghost,))
    assert f"ghost_release 0.99 ({F.MINING_HEAD}/{MP.HEAD_VERSION}, annotation-only)" \
        in F.summary()
    assert F.summary().count("annotation-only") == len(F.ALL_FLOORS)


# --------------------------------------------------------------------------- #
# 4. the ONE enforcing cut (2026-08-09)
# --------------------------------------------------------------------------- #
def test_the_junk_floor_is_the_only_enforcing_cut_and_it_is_semantic():
    """It is a bare float, not a `Floor`, and that is the design: it was written to read on TWO
    different heads (stage-1 at emission intake, mining at `deploy_tail`) and a stamp names one.
    What keeps it honest across a head flip is that it is DECLARED SCALE-FREE and left alone (see
    the test below), not a stamp check and not a restatement. The mining-scale reader is gone
    twice over now — repointed 2026-08-11, retired with the deploy-tail driver 2026-08-12 — and
    the constant is unchanged by either, which IS the property."""
    assert F.JUNK_FLOOR == 0.20
    assert not any(f.value == F.JUNK_FLOOR for f in F.ALL_FLOORS)
    assert "ENFORCING" in F.summary() and str(F.JUNK_FLOOR) in F.summary()
    # It sits below both WALLPAPER cuts — on that head the retirement widened the funnel and
    # did not narrow it, which is what the annotation-only claim needs to stay true.
    assert all(F.JUNK_FLOOR < f.value for f in (F.WALLPAPER_POOL, F.WALLPAPER_RELEASE))


def test_the_junk_floor_still_sits_above_both_mining_cuts_and_no_mining_cut_acts_at_all():
    """The 2026-08-11 inversion and the 2026-08-12 retirement that ended it, pinned together.

    The sheet-F crossover put the mining gate at 0.0949 and the pool floor at 0.0, both BELOW
    0.20, so the colorize-pool draw started removing rows the gate passes — 132 of them, 455 of
    the 827 reference-pool rows clearing 0.20 against 587 clearing the gate. The arithmetic
    inversion is still there (first assert) and was never what was fixed: on 2026-08-11 Matt
    moved the READER, repointing `deploy_tail` at `mining_gate.MiningScorer.gate`; on 2026-08-12
    the deploy-tail DRIVER was retired and took that call site with it.

    So the mining scale now has NO acting cut anywhere: `MINING_RELEASE` is a `Floor` and a
    `Floor` cannot remove a row, `MINING_POOL` is 0.0, `rank_select` holds no floor, and
    `MiningScorer.gate` — the last comparison that removed anything on this head — has no caller
    left. That is asserted by an AST scan rather than a text one, because the retirement's own
    record NAMES the retired call site in prose and a text scan cannot tell a eulogy from a call.
    Records, both untouched as provenance: data/render_mode_head/v3/mining_gate_lock_2026-08-11.md
    (the 0.0949 crossover) and scratch/deploy_tail_recon_report.md (the retirement)."""
    assert F.JUNK_FLOOR > F.MINING_RELEASE.value > F.MINING_POOL.value
    assert F.MINING_POOL.value < F.MINING_RELEASE.value        # the invariant that still holds
    assert "0.0949" in F.MINING_RELEASE.basis and "CROSSOVER" in F.MINING_RELEASE.basis
    assert "CONSEQUENCE" in F.MINING_POOL.basis

    callers = _gate_call_sites()
    assert callers == [], (
        f"`MiningScorer.gate` has a caller again: {callers}. It has had NO acting site since "
        f"2026-08-12, so a new one is a decision to make 0.0949 cut somewhere — re-derive it "
        f"against the head that site scores on, or route the row past it.")


def _gate_call_sites() -> list:
    """`["<file>:<line>", ...]` for every `<x>.gate(...)` call in tracked non-test Python.

    AST, not regex: the only remaining textual mentions of the retired call are in the prose
    that records its retirement, and a scan that cannot distinguish those from a live call is a
    scan that will be edited away the next time someone writes the history down."""
    import ast
    import subprocess
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable")
    hits = []
    for rel in out.stdout.splitlines():
        norm = rel.replace("\\", "/")
        if Path(norm).name.startswith("test_"):
            continue
        try:
            tree = ast.parse((ROOT / norm).read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:                    # not ours to police here
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "gate"):
                hits.append(f"{norm}:{node.lineno}")
    return sorted(hits)


def test_the_gate_call_scan_is_not_vacuous(tmp_path, monkeypatch):
    """The assertion above passes by evaluating EMPTY, which is exactly the shape that goes
    quiet when the scanner breaks (verification_practice.md §5). Point it at a planted tree and
    it must SEE the call it is supposed to be watching for."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "planted.py").write_text(
        "def f(scorer, p):\n    return [c for c in p if scorer.gate(c)]\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert _gate_call_sites() == ["tools/planted.py:2"]


def test_the_junk_floor_removes_a_row_at_ONE_site_and_it_is_the_location_scale_one():
    """The consequence of the repoint above, stated as a census so a second mining-scale reader
    cannot appear unnoticed. `ranked_intake` (stage-1 location head `p_good`) is THE site that
    drops a row on this floor, and since 2026-08-12 it is the ONLY caller of the comparison at
    all: `deploy_tail` kept one counterfactual COUNT of it after the repoint (`n_above_junk`, a
    report field), and that went with the deploy-tail driver.

    Not "no other file may mention it": `floors.py`'s own docs, the readouts that print the
    number and the audit that measures the inversion all name it and none of them removes a row.
    What is counted is CALLS to the comparison, and what each caller does with the answer."""
    import re
    import subprocess
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable")
    call = re.compile(r"\b(?:F|floors|FLOORS)\.passes_junk_floor\s*\(")
    callers = {}
    for rel in out.stdout.splitlines():
        norm = rel.replace("\\", "/")
        if Path(norm).name.startswith("test_"):
            continue
        src = (ROOT / norm).read_text(encoding="utf-8", errors="ignore")
        hits = [ln.strip() for ln in src.splitlines() if call.search(ln)]
        if hits:
            callers[norm] = hits
    assert set(callers) == {"tools/emission/ranked_intake.py"}, \
        (f"the junk floor's callers are {sorted(callers)}; since 2026-08-12 it is read at ONE "
         f"colorize-pool draw (stage-1) and nowhere else. A new caller is a decision about what "
         f"0.20 means on that site's head, not a detail — and on the MINING head specifically "
         f"it would re-open the 2026-08-11 inversion, since 0.20 sits above both mining cuts.")


def test_the_junk_floor_is_declared_PERMANENT_shared_scale_at_the_constant_and_in_the_protocol():
    """Matt, 2026-08-11: `JUNK_FLOOR` is a coarse semantic floor read on BOTH heads' scales and
    is never restated at a head flip — the settled answer to the residual §5a used to carry.

    This is a PROSE assertion on purpose and it is the honest instrument available: the
    decision is "do not change this number at a flip", and no runtime check can see a change
    that does not happen. What it does catch is the failure that actually threatens it — the
    next flip reading §5a's step 2, volume-matching all the floors it names, and leaving the
    two declarations disagreeing about which ones those are. Both texts must carry the
    exemption, and `GOOD_FLOOR` must still carry the opposite instruction, or the pair has
    drifted (`verification_practice.md` §5: pin the claim where it can rot)."""
    src = (Path(__file__).resolve().parent / "floors.py").read_text(encoding="utf-8")
    junk, good = src.split("JUNK_FLOOR = 0.20")[0], src.split("GOOD_FLOOR = 0.50")[0]
    assert "PERMANENT SHARED-SCALE" in junk, \
        "floors.py must declare the exemption at the constant it exempts"
    # the paragraph immediately above GOOD_FLOOR still orders the restatement for THAT floor
    assert "VOLUME-MATCHED" in good.rsplit("JUNK_FLOOR = 0.20", 1)[-1]

    proto = (ROOT / "docs" / "design" / "classifier_retrain_protocol.md").read_text(
        encoding="utf-8")
    sec = proto.split("### 5a.")[1].split("### 5b.")[0]
    assert "PERMANENT shared-scale" in sec and "exempt" in sec, \
        "protocol §5a must state the exemption, or a flip will volume-match it back"
    assert "GOOD_FLOOR`** (and each stamped floor)" in sec, \
        "§5a's restatement step must name what it DOES apply to, not 'each floor'"


def test_passes_junk_floor_treats_a_missing_score_as_failing():
    """An unscored candidate has no verdict to spend colorize compute on. `None` must not
    compare as 0.0-and-therefore-fail-anyway by accident, nor pass."""
    assert F.passes_junk_floor(0.20) is True          # inclusive at the boundary
    assert F.passes_junk_floor(0.199) is False
    assert F.passes_junk_floor(None) is False
    assert F.passes_junk_floor(0.0) is False


def test_the_thin_supply_divisor_and_cluster_cap_live_beside_the_floor():
    """Both are coarse volume constants and both are declared here, so the three numbers this
    restructure introduced move together and none of them is re-typed downstream."""
    assert F.THIN_SUPPLY_DIVISOR == 4 and F.CLUSTER_CAP == 2
    from tools.emission import ranked_intake as RI     # noqa: PLC0415
    from tools.emission import selection as SEL        # noqa: PLC0415
    assert RI.emit_cap(F.THIN_SUPPLY_DIVISOR * 3 + 3) == 3     # floor(15/4)
    assert SEL.CLUSTER_CAP is F.CLUSTER_CAP


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------- #
# 5. the good floor (2026-08-09, prompts/selection_restructure_3.md)
# --------------------------------------------------------------------------- #
def test_the_good_floor_is_the_run_side_cut_and_shares_the_junk_floor_s_shape():
    """Two heights of ONE comparison, both on the stored raw P(>=3), both bare floats rather
    than `Floor`s. Not `Floor`s for the reason the junk floor is not: they read on whichever
    head owns the score at that site, and a stamp names one head."""
    assert F.GOOD_FLOOR == 0.50 and F.JUNK_FLOOR == 0.20
    assert F.JUNK_FLOOR < F.GOOD_FLOOR, "junk must be the permissive end of the same scale"
    assert F.passes_good_floor(0.50) and F.passes_good_floor(0.99)
    assert not F.passes_good_floor(0.4999) and not F.passes_good_floor(None)
    # everything the junk floor passes is not automatically good, and vice versa is total
    assert F.passes_junk_floor(0.30) and not F.passes_good_floor(0.30)


def test_good_class_answers_None_below_the_floor_rather_than_a_bad_class():
    """Below the floor there is NO class. The run keeps no verdict about how bad a thing it
    did not keep is, which is what stops a stored class from being read as a decode of the
    retired 1..4 kind."""
    assert F.good_class(0.4) is None
    assert F.good_class(0.4, 0.99) is None, "P(>=4) cannot promote a below-floor frame"
    assert F.good_class(0.6) == 3
    assert F.good_class(0.6, 0.49) == 3
    assert F.good_class(0.6, 0.50) == 4
    assert F.good_class(None) is None


def test_the_two_natural_cutpoints_are_declared_here_not_at_their_call_sites():
    """`NOTBAD_CUT` / `GREAT_CUT` are CORN's own uncalibrated rank cutpoints, and they only
    NAME a frame — the run-side sites that need a class rather than a yes/no read them from
    here so the surface of "numbers that decide something about a frame" stays one file."""
    assert F.NOTBAD_CUT == F.GREAT_CUT == 0.50
    assert not any(f.value == F.GOOD_FLOOR and f.site == "pool" for f in F.ALL_FLOORS)


def test_no_module_outside_the_owner_declares_the_run_side_floors():
    """The run side IMPORTS the constants and never restates them. `test_floors_one_source`
    scans stage 2 and the two heads for a re-typed stage-2 cut; the run side is in
    `tools/atlas`, outside that surface, so the two names are scanned for by hand here."""
    import re
    import subprocess
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable")
    pat = re.compile(r"^\s*(GOOD_FLOOR|JUNK_FLOOR|GREAT_CUT|NOTBAD_CUT)\s*(?::[^=\n]*)?=", re.M)
    offenders = []
    for rel in out.stdout.splitlines():
        norm = rel.replace("\\", "/")
        if norm in ("tools/emission/floors.py", "tools/emission/test_floors.py"):
            continue
        if pat.search((ROOT / rel).read_text(encoding="utf-8", errors="replace")):
            offenders.append(norm)
    assert not offenders, f"the run-side cuts are re-declared in {offenders}"
    assert pat.search("GOOD_FLOOR = 0.5")          # non-vacuous
