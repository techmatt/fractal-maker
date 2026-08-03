#!/usr/bin/env python
"""The one disk-free check of `verify_cache_alignment.py` whose inputs still exist.

That script is the pre-train gate: it asserts tile<->location agreement in both
directions, because the aug cache is keyed on `loc_id` and a silent renumber trains on
tiles belonging to a different location — plausible numbers, wrong model. It runs by hand,
needs `--prior-plan`/`--prior-eval` backups that are not committed, and half its checks
read the ~12 GB tile cache, which has been DELETED since v9 was trained.

WHAT THIS FILE USED TO HOLD, and why three of its four tests are gone rather than skipped.
It carried four relations among the committed v8 artifacts. Three of them — BACKWARD
(no orphan cache row), FIELDS (cache split/group/label/biased == the manifest's) and
COUNTS (exactly 24 plan rows per loc_id) — read `data/v8/{plan,cache_manifest}.jsonl`,
which were DELETED on 2026-08-03: 146 MB of derived rows that `tools/v8/build_plan.py`
reproduces byte-for-byte from the committed manifest (verified by rebuild + sha256 in that
pass). Their referent is gone, so they are DELETED. They were not left behind a
`pytest.skip`, because a skip on this path is worse than nothing: the file whose absence
triggers it is regenerable on demand, so "skipped" would mean "nobody ran the command",
printed once per suite run forever, and a suite that skips its own alignment checks reads
green. If the pair is ever rebuilt, `git show` has these three verbatim.

  CENSUS  the eval slice holds exactly 144 `prospect_census` locations

is what remains, and it reads only `data/v8/eval_slice.jsonl` — committed, small, and not
a rebuild candidate. The census is a fixed, deliberately-constructed eval population; a
re-split that drops or duplicates any of it changes what every AP number since means.

WHAT STAYS IN THE CLI. The other checks are irreducibly external: FORWARD plan-vs-prior
and the census loc_id-preservation check compare against a PRIOR build that only exists as
a hand-made backup, and both tiles-on-disk checks need the cache. A test cannot assert
those without inventing an input, and a gate that invents its input is not a gate.
`verify_cache_alignment.py` is unchanged, and it now needs the plan pair rebuilt first.

  uv run pytest tools/v8/test_v8_cache_alignment.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / "data" / "v8"
EVAL_SLICE = V8 / "eval_slice.jsonl"
CENSUS_N = 144        # prospect_census locations in the eval slice
CENSUS_SOURCE = "prospect_census"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_the_eval_slice_is_present_and_nonempty():
    """No skip. `data/v8/eval_slice.jsonl` is committed and LFS-negated by exact path; if
    it is missing, that is the durability wiring failing, which is a red — not a reason to
    quietly not run the census check below."""
    assert EVAL_SLICE.exists(), (
        f"{EVAL_SLICE.relative_to(ROOT)} is absent. It is committed and exact-path negated "
        f"in .gitignore — check LFS smudging, do not skip this file.")
    assert len(_load(EVAL_SLICE)) > 100


def test_the_eval_slice_holds_the_full_144_location_census():
    """The census is a fixed, deliberately-constructed eval population; a re-split that
    drops or duplicates any of it changes what every AP number since means. Only the COUNT
    is assertable here — identity-preservation vs the PRIOR build needs the uncommitted
    backup, and that check stays in verify_cache_alignment.py."""
    n = sum(1 for r in _load(EVAL_SLICE) if r.get("source") == CENSUS_SOURCE)
    assert n == CENSUS_N, f"{n} {CENSUS_SOURCE} rows in the eval slice, expected {CENSUS_N}"
