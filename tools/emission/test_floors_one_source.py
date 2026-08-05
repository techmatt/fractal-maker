#!/usr/bin/env python
"""`tools/emission/floors.py` is the ONLY source of a stage-2 emission cut.

The pattern is `tools/scoring/test_release_mix_one_source.py`'s, applied to the other family
of numbers stage 2 was carrying in duplicate. Before 2026-08-04 the four cuts lived in six
places:

  tools/emission/build_emission_diversity_v1.py   DEFAULT_FLOOR / DEFAULT_MINING_FLOOR /
                                                  DEFAULT_RELEASE_FLOOR /
                                                  DEFAULT_MINING_RELEASE_FLOOR
  tools/emission/first_release_readout.py         WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50
  tools/emission/reselect_readout.py              WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50
  tools/emission/q4_harvest_readout.py            STRICT_WP, STRICT_MN = 0.90, 0.50
  tools/emission/report.py                        two bare literals inside the surplus counts
  (+ the head gates themselves in emit_v1 / mining_gate, which the release floors now IMPORT)

They all agreed, which is what made it dangerous: nothing would have noticed the day one of
them stopped agreeing, and a readout that annotates a pool against a floor the driver is not
using reads as a measurement.

So this file asserts:
  1. every consumer resolves the SAME value the owner declares, and
  2. no tracked source declares a floor of its own.

  uv run pytest tools/emission/test_floors_one_source.py -q
"""
from __future__ import annotations

import re
import subprocess
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
# 1. every consumer resolves the owner's value.
# --------------------------------------------------------------------------- #
def test_the_driver_defaults_are_the_owners_values():
    from tools.emission import build_emission_diversity_v1 as B
    assert B.DEFAULT_FLOOR == F.WALLPAPER_POOL.value
    assert B.DEFAULT_MINING_FLOOR == F.MINING_POOL.value
    assert B.DEFAULT_RELEASE_FLOOR == F.WALLPAPER_RELEASE.value
    assert B.DEFAULT_MINING_RELEASE_FLOOR == F.MINING_RELEASE.value


def test_the_readouts_annotate_against_the_owners_values():
    from tools.emission import first_release_readout as FR
    from tools.emission import reselect_readout as RS
    from tools.emission import q4_harvest_readout as Q4
    for mod in (FR, RS):
        assert mod.WP_RELEASE_FLOOR == F.WALLPAPER_RELEASE.value
        assert mod.MN_RELEASE_FLOOR == F.MINING_RELEASE.value
    assert (Q4.STRICT_WP, Q4.STRICT_MN) == (F.WALLPAPER_RELEASE.value, F.MINING_RELEASE.value)
    assert (Q4.POOL_WP, Q4.POOL_MN) == (F.WALLPAPER_POOL.value, F.MINING_POOL.value)


def test_an_owner_edit_moves_every_consumer_together():
    """INJECTION, the release_mix pattern: change the policy in its one home and every
    consumer moves. A private copy anywhere would leave one of these behind.

    The release floors resolve from the HEAD pin, so this moves the pin — the stronger form
    of the same property (a gate retune must carry its release floor with it). Run in a
    SUBPROCESS: the patch has to be in place before `floors` is first imported, and a reload
    dance in-process leaks a 0.42 floor into every test that runs after it (it did)."""
    prog = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from tools.wallpaper import wallpaper_pins as WP\n"
        "from tools.mining import mining_pins as MP\n"
        # moved UP, not down: `floors.check_below_gate` runs at import and would (correctly)
        # abort on a release floor pushed below a pool floor, which is a different guard.
        "WP.GATE_THRESHOLD = 0.95; MP.MINING_GATE_THRESHOLD = 0.85\n"
        "from tools.emission import floors as F\n"
        "from tools.emission import first_release_readout as FR\n"
        "from tools.emission import reselect_readout as RS\n"
        "from tools.emission import q4_harvest_readout as Q4\n"
        "from tools.emission import build_emission_diversity_v1 as B\n"
        "vals = [F.WALLPAPER_RELEASE.value, F.MINING_RELEASE.value,\n"
        "        FR.WP_RELEASE_FLOOR, FR.MN_RELEASE_FLOOR,\n"
        "        RS.WP_RELEASE_FLOOR, RS.MN_RELEASE_FLOOR,\n"
        "        Q4.STRICT_WP, Q4.STRICT_MN,\n"
        "        B.DEFAULT_RELEASE_FLOOR, B.DEFAULT_MINING_RELEASE_FLOOR]\n"
        "assert vals == [0.95, 0.85] * 5, vals\n"
        "print('MOVED')\n") % (ROOT,)
    out = subprocess.run([sys.executable, "-c", prog], cwd=ROOT,
                         capture_output=True, text=True)
    assert "MOVED" in out.stdout, out.stdout + out.stderr
    # ...and this process is untouched by it.
    assert F.WALLPAPER_RELEASE.value == WP.GATE_THRESHOLD


def test_the_head_gates_are_the_release_floors_not_a_second_copy():
    """`emit_v1` / `mining_gate` re-export their pins; the owner imports the SAME objects.
    Asserted by value AND by the pin modules being the ones the heads use, so a head that
    re-declared its own threshold would show up here."""
    from tools.wallpaper import emit_v1  # noqa: PLC0415  (heavy; only this test needs it)
    from tools.mining import mining_gate  # noqa: PLC0415
    assert emit_v1.GATE_THRESHOLD is WP.GATE_THRESHOLD
    assert mining_gate.MINING_GATE_THRESHOLD is MP.MINING_GATE_THRESHOLD
    assert F.WALLPAPER_RELEASE.value == emit_v1.GATE_THRESHOLD
    assert F.MINING_RELEASE.value == mining_gate.MINING_GATE_THRESHOLD
    # the version tag stamped into every durable gate-report row is derived, not restated
    from tools.mining import gate_report as GR  # noqa: PLC0415
    assert GR.MINING_GATE_VERSION is MP.MINING_GATE_VERSION


# --------------------------------------------------------------------------- #
# 2. no second source.
# --------------------------------------------------------------------------- #
# The owner, the head pins the release floors are imported FROM, and this file (which quotes
# the shapes it forbids). `mining_gate`/`emit_v1` are NOT exempt: they re-export now, and a
# re-declared literal there is exactly the regression this catches.
OWN = {
    "tools/emission/floors.py",                  # THE owner
    "tools/emission/test_floors_one_source.py",  # quotes the forbidden shapes
    "tools/mining/mining_pins.py",               # the mining head's own operating point
    "tools/wallpaper/wallpaper_pins.py",         # the wallpaper head's own operating point
}

# The SURFACE the scan covers: stage 2 and the two heads it gates against. Deliberately not
# the whole tree — `FLOOR`/`THRESHOLD` is a common word and every unrelated structure floor,
# occupancy floor and interior cap in `tools/atlas` would be a false positive, which is how a
# scan gets an exemption list long enough to hide a real one.
SURFACE = ("tools/emission/", "tools/mining/", "tools/wallpaper/")

# A module-level constant named like a stage-2 cut, assigned a bare numeric literal. Catches
# `DEFAULT_FLOOR = 0.75`, `WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50`,
# `STRICT_WP, STRICT_MN = ...`, `POOL_FLOOR: float = 0.75`. Does NOT catch an assignment from
# the owner (`DEFAULT_FLOOR = F.WALLPAPER_POOL.value`) — the ban is on re-typing the number,
# not on naming it.
FLOOR_LITERAL = re.compile(
    r"^(?!\s)"                                         # module level only
    r"(?P<names>[A-Z0-9_,\s]*"                         # possibly a tuple target
    r"\b(?:[A-Z0-9_]*FLOOR[A-Z0-9_]*|STRICT_WP|STRICT_MN|[A-Z0-9_]*GATE_THRESHOLD)\b"
    r"[A-Z0-9_,\s]*)"
    r"(?::[^=\n]*)?=\s*(?P<value>[-+0-9.][-+0-9.eE,\s]*)", re.M)

# A cut is a point on a HEAD's probability scale, so a literal outside [0,1] is not one of
# these and is not what this scan is for. Two live names in the surface land there or are
# otherwise not head scores; both are named, because an unnamed exclusion is a hole:
#   * `descriptor.NEAR_DUP_THRESHOLD` (0.974) — a morph-CLIP COSINE, not a head score.
#   * `library_seed_v2.FLOOR_LABEL` (3)       — a HUMAN label, excluded by the [0,1] rule.
NOT_A_HEAD_SCORE = {"NEAR_DUP_THRESHOLD"}


def _probability_scale(value_text: str) -> bool:
    """True if every literal on the right-hand side is a float in [0, 1] — i.e. could be a
    point on a head's probability scale."""
    parts = [p.strip() for p in value_text.split(",") if p.strip()]
    if not parts:
        return False
    try:
        return all(0.0 <= float(p) <= 1.0 for p in parts)
    except ValueError:
        return False


def _tracked_python():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [p for p in out.stdout.splitlines() if p.strip()]


def _scan(text: str):
    """(line_offset, names) for each module-level stage-2-floor-shaped literal in `text`."""
    hits = []
    for m in FLOOR_LITERAL.finditer(text):
        names = {n for n in re.split(r"[,\s]+", m.group("names")) if n}
        if names & NOT_A_HEAD_SCORE:
            continue
        if not _probability_scale(m.group("value")):
            continue
        hits.append((m.start(), sorted(names)))
    return hits


def test_no_module_declares_a_stage_2_floor_of_its_own():
    offenders = []
    for rel in _tracked_python():
        norm = rel.replace("\\", "/")
        if norm in OWN or not norm.startswith(SURFACE):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for start, names in _scan(text):
            offenders.append(f"{norm}:{text[:start].count(chr(10)) + 1} {names}")
    assert not offenders, (
        f"{len(offenders)} module-level stage-2 floor literal(s) outside "
        f"tools/emission/floors.py: {offenders}. Import the cut from the owner — a re-typed "
        f"floor carries no head stamp, so nothing can tell whether it is a point on the "
        f"scale the live head produces.")


def test_the_scan_would_catch_a_copy():
    """Non-vacuity, in both directions — every historical duplicate shape fires, and the
    import-from-owner shapes that replaced them do not."""
    for planted in ("DEFAULT_FLOOR = 0.75          # wallpaper-head POOL floor",
                    "WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50",
                    "STRICT_WP, STRICT_MN = 0.90, 0.50   # production release floors",
                    "DEFAULT_MINING_RELEASE_FLOOR = 0.50",
                    "MINING_GATE_THRESHOLD = 0.50",
                    "GATE_THRESHOLD = 0.90",
                    "POOL_FLOOR: float = 0.75",
                    "DEFAULT_FLOOR = 0.80   # a DIVERGED copy, not merely a duplicated one"):
        assert _scan(planted), planted
    for ok in ("DEFAULT_FLOOR = F.WALLPAPER_POOL.value",
               "WP_RELEASE_FLOOR = F.WALLPAPER_RELEASE.value",
               "STRICT_WP = F.WALLPAPER_RELEASE.value",
               "from tools.emission import floors as F",
               "    release_floor = 0.9    # a local, not a module-level policy",
               "# the pool floor used to be 0.75 here",
               "NEAR_DUP_THRESHOLD = 0.974",     # a cosine, named exclusion
               "FLOOR_LABEL = 3"):               # a human label, outside [0,1]
        assert not _scan(ok), ok


def test_the_deleted_badness_floor_has_no_source_at_all():
    """§B. `descriptor.FLOOR_PNOTBAD` was deleted, not moved into the owner — a floor-admit
    source takes no machine quality cut. A re-add anywhere goes red here rather than quietly
    re-introducing a v7-era number on a v10 scale."""
    from tools.emission import descriptor as D  # noqa: PLC0415
    assert not hasattr(D, "FLOOR_PNOTBAD")
    assert not any(f.head == F.LOCATION_HEAD for f in F.ALL_FLOORS)
    pat = re.compile(r"^\s*FLOOR_PNOTBAD\s*(?::[^=\n]*)?=", re.M)
    offenders = []
    for rel in _tracked_python():
        norm = rel.replace("\\", "/")
        if norm == "tools/emission/test_floors_one_source.py":
            continue
        if pat.search((ROOT / rel).read_text(encoding="utf-8", errors="replace")):
            offenders.append(norm)
    assert not offenders, f"FLOOR_PNOTBAD re-declared in {offenders}"
    assert pat.search("FLOOR_PNOTBAD = 0.5")           # non-vacuous


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
