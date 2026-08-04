#!/usr/bin/env python
"""`tools/scoring/release_mix.py` is the ONLY copy of the intended release mix.

Two halves. The first is that the table and `partitions.ALL_FAMS` cover exactly each other, in
BOTH directions and at import — the guard that stops a newly registered partition from silently
getting no share of the release, and stops a retired one from silently deflating everyone
else's. The second is a **source scan**: no other tracked Python file may bind a per-partition
ratio table of its own. `partitions.py` earned that scan the hard way (seven literal copies of
nine pairs); a policy table that every stage wants to consult earns it before the fact.

  uv run pytest tools/scoring/test_release_mix.py -q
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import partitions as P        # noqa: E402
import release_mix as RM      # noqa: E402

OWNER = "tools/scoring/release_mix.py"
EXEMPT = {OWNER, "tools/scoring/test_release_mix.py"}
LITERAL = re.compile(r"^\s*(RATIO|RELEASE_MIX|MIX_RATIO)\s*(:[^=]*)?=\s*\{", re.M)


# =========================================================================== #
# 1. the table itself
# =========================================================================== #
def test_the_table_is_matts_dictated_ratios():
    """The numbers as Matt gave them, 2026-08-04. Pinned literally because they are a DECISION,
    not a derivation — nothing in the tree can re-derive them, so nothing else can catch a
    typo in one."""
    assert RM.RATIO == {
        "mandelbrot": 3.0, "julia:mandelbrot": 3.0,
        "multibrot3": 1.0, "julia:multibrot3": 1.0,
        "multibrot4": 1.0, "julia:multibrot4": 1.0,
        "multibrot5": 1.0, "julia:multibrot5": 1.0,
        "phoenix": 1.0, "phoenix:classic": 0.2,
    }


def test_the_table_covers_the_partition_registry_exactly():
    assert set(RM.RATIO) == set(P.ALL_FAMS)
    assert set(RM.ratios()) == set(P.ALL_FAMS)


def test_a_registered_partition_with_no_ratio_is_a_RED_BUILD():
    """Injection, both directions — the guard reads both tables at call time precisely so it
    can be proved red without editing the file."""
    with pytest.raises(KeyError, match="registered with no ratio"):
        RM.check_complete(fams=list(P.ALL_FAMS) + ["multibrot9"])


def test_a_ratio_for_an_unregistered_partition_is_a_RED_BUILD():
    with pytest.raises(KeyError, match="unregistered partition"):
        RM.check_complete(ratio=dict(RM.RATIO, retired_family=1.0))


def test_a_zero_ratio_is_refused_because_it_leaves_the_partition_registered_and_starved():
    with pytest.raises(ValueError, match="non-positive"):
        RM.check_complete(ratio=dict(RM.RATIO, phoenix=0.0))


def test_the_import_time_check_is_the_one_that_runs():
    """A guard defined and never called is prose. Prove it fires on a FRESH interpreter — a
    shared one has already imported the module (`verification_practice.md` §3)."""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import partitions as P\n"
        "P.ALL_FAMS.append('multibrot9')\n"
        "import release_mix\n" % (ROOT / "tools" / "scoring"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode != 0, "release_mix imported cleanly with an unrationed partition"
    assert "multibrot9" in r.stderr


# =========================================================================== #
# 2. the derived reads
# =========================================================================== #
def test_shares_are_derived_and_sum_to_one():
    s = RM.shares()
    assert sum(s.values()) == pytest.approx(1.0)
    assert s["mandelbrot"] == pytest.approx(3.0 / sum(RM.RATIO.values()))
    # ...and phoenix:classic is a garnish, not a family: strictly under 2% of the release.
    assert s["phoenix:classic"] < 0.02


def test_ratios_hands_back_a_copy_not_the_policy():
    r = RM.ratios()
    r["mandelbrot"] = 99.0
    assert RM.RATIO["mandelbrot"] == 3.0


def test_ratio_of_raises_on_an_unregistered_partition():
    assert RM.ratio_of("phoenix:classic") == 0.2
    with pytest.raises(KeyError, match="release-mix ratio"):
        RM.ratio_of("multibrot9")


def test_the_reads_follow_a_monkeypatched_table(monkeypatch):
    """Everything is read at call time, so a policy change reaches a live process. This is the
    property `pop_quota` depends on to re-allocate on the CURRENT mix every pop."""
    monkeypatch.setitem(RM.RATIO, "phoenix:classic", 2.0)
    assert RM.ratio_of("phoenix:classic") == 2.0
    assert RM.ratios()["phoenix:classic"] == 2.0
    assert RM.shares()["phoenix:classic"] > 0.02


# =========================================================================== #
# 3. one copy, enforced
# =========================================================================== #
def _tracked_py():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [p for p in out if p not in EXEMPT]


def test_no_second_literal_ratio_table_exists():
    files = _tracked_py()
    assert files, "git ls-files returned nothing — the scan would pass vacuously"
    hits = [p for p in files
            if LITERAL.search((ROOT / p).read_text(encoding="utf-8", errors="ignore"))]
    assert not hits, (f"a per-partition ratio table is bound outside {OWNER}: {hits}. "
                      f"Import it — a second copy does not fail when it is written, it fails "
                      f"the year someone changes one of them.")


def test_the_scan_would_actually_catch_a_copy():
    """Non-vacuity: the regex must match the shape it hunts for."""
    assert LITERAL.search("RATIO = {\n    'mandelbrot': 3.0,\n}")
    assert LITERAL.search("MIX_RATIO: dict = {")
    assert not LITERAL.search("ratio = RATIO['mandelbrot']")
