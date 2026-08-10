"""The frozen-record write gates: a default run must not rewrite a record of a past state.

Shape guarded (`scratch/cleanup/item4_frozen_overwrite_sweep.md`): a `main()`/stage that
writes a durable path UNCONDITIONALLY over a value that RECORDS something — a pre-registered
bar, a timing measurement, an adopted threshold table — rather than deriving it from committed
inputs. The fixed precedents invert the default: the durable write takes an explicit flag.

Four sites, each proved in BOTH directions here (verification_practice.md §3, "bracket a fix
on both sides"): the old behaviour is still reachable and still destroys the record, the new
default preserves it byte-identically, and the explicit flag still writes.

Cross-cutting on purpose. These live in three version directories that are otherwise
independent, and a per-directory copy of the same assertion is how one of them silently loses
it at the next flip.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    """Import a tools/v*/ module by path under a unique name — these directories are not
    packages and three of them define `build_plan`."""
    for p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring",
              ROOT / "tools" / "atlas", ROOT / "tools" / "mining", (ROOT / rel).parent):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# v8 / v9 build_plan — aug_recipe.marginal_cost survives a plain rebuild        #
# --------------------------------------------------------------------------- #
MARGINAL_SITES = [("v8", "tools/v8/build_plan.py", "data/v8/build_metadata.json"),
                  ("v9", "tools/v9/build_plan.py", "data/v9/build_metadata.json")]


@pytest.mark.parametrize("ver,rel,meta", MARGINAL_SITES)
def test_carry_marginal_preserves_unless_the_run_measured(ver, rel, meta):
    """`marginal_cost` is the one recipe key that records a past timing run: it is None
    unless --measure-marginal was passed, so an unguarded rebuild overwrote it with null."""
    bp = _load(f"_bp_{ver}", rel)
    committed = {"marginal_cost": {"arms": {"pal": {"s_per_tile": 0.03}}}, "other": 1}
    rebuilt = {"marginal_cost": None, "other": 1}

    # GREEN — a plain rebuild carries the committed measurement forward.
    assert bp.carry_marginal(committed, rebuilt, False, meta)["marginal_cost"] \
        == committed["marginal_cost"]
    # RED — the pre-fix behaviour is exactly this call with the flag set; it still nulls the
    # record, which is what makes the green case a real guard rather than a no-op.
    assert bp.carry_marginal(committed, rebuilt, True, meta)["marginal_cost"] is None
    # A real --measure-marginal run must still be able to REPLACE the measurement.
    fresh = {"marginal_cost": {"arms": {"pal": {"s_per_tile": 0.07}}}, "other": 1}
    assert bp.carry_marginal(committed, fresh, True, meta) == fresh
    # No committed record yet (first build) — pass through untouched, never invent a value.
    assert bp.carry_marginal(None, rebuilt, False, meta) is rebuilt
    # Everything OTHER than marginal_cost is left alone.
    assert bp.carry_marginal(committed, rebuilt, False, meta)["other"] == 1


@pytest.mark.parametrize("ver,rel,meta", MARGINAL_SITES)
def test_both_recipe_writers_are_gated_on_the_measure_flag(ver, rel, meta):
    """Source scan. `recipe_block` is written to TWO durable targets (the metadata file and
    aug_roster.json) and they already hold different values, so each must carry forward its
    OWN committed one. A gate on only the first is the defect one file over."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    assert "amend_metadata(recipe_block, a.measure_marginal)" in src, \
        f"{rel}: amend_metadata must be told whether this run measured"
    assert "carry_marginal(roster_committed, recipe_block, a.measure_marginal" in src, \
        f"{rel}: the aug_roster write must carry its own committed marginal_cost too"
    assert "def amend_metadata(recipe_block: dict, measured: bool)" in src


# Every version whose {plan,cache_manifest} pair has been de-tracked, with the date. All
# three went for the same reason — byte-reproducible from the manifest, and the aug_cache the
# pair mapped is gone — so the guard is one parametrized assertion rather than three copies.
DETRACKED_PAIRS = [
    ("tools/v8/build_plan.py", "2026-08-03"),
    ("tools/v9/build_plan.py", "2026-08-08"),
    ("tools/v10/build_plan.py", "2026-08-08"),
]


@pytest.mark.parametrize("rel,when", DETRACKED_PAIRS)
def test_a_detracked_derived_pair_is_not_declared_durable(rel, when):
    """`data/v{8,9,10}/{plan,cache_manifest}.jsonl` were deleted and their .gitignore
    negations went with them. `durable()` asserts its target is not ignored, so leaving a
    pair declared durable makes THAT BUILDER — the rebuild the deletion's argument rests on —
    raise before it can finish. v8 sprang the trap first (2026-08-03); the guard is
    parametrized so v9's and v10's de-tracks cannot re-spring it. Class follows the decision:
    byte-reproducible and deliberately untracked is bulk()."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    assert "paths.durable(PLAN_OUT" not in src and "paths.durable(CACHE_MANIFEST_OUT" not in src, \
        f"{rel}: the pair was de-tracked {when}; durable() would refuse an ignored path"
    assert "paths.bulk(rel)" in src, f"{rel}: the pair must be written through bulk()"


# --------------------------------------------------------------------------- #
# tools/v10/prereg.py — a post-hoc run cannot produce a pre-hoc file            #
# --------------------------------------------------------------------------- #
PREREG_REC = ROOT / "data/v10/prereg_v10.json"


@pytest.fixture(scope="module")
def prereg():
    return _load("_prereg_v10", "tools/v10/prereg.py")


def _run_prereg(prereg, monkeypatch, argv, doc):
    """Run main() with `build()` pinned to `doc` — the guard lives after build(), and
    pinning it keeps this test off the 9-second path that reads 116 MB of plan rows."""
    monkeypatch.setattr(prereg, "build", lambda: doc)
    monkeypatch.setattr(sys, "argv", ["prereg.py"] + argv)
    return prereg.main()


@pytest.mark.skipif(not PREREG_REC.exists(), reason="no committed v10 pre-registration")
def test_prereg_default_run_does_not_touch_the_committed_bars(prereg, monkeypatch, capsys):
    doc = json.loads(PREREG_REC.read_text(encoding="utf-8"))
    before = _sha(PREREG_REC)
    assert _run_prereg(prereg, monkeypatch, [], doc) == 0
    assert _sha(PREREG_REC) == before
    assert "NOT written" in capsys.readouterr().out


@pytest.mark.skipif(not PREREG_REC.exists(), reason="no committed v10 pre-registration")
@pytest.mark.parametrize("argv", [[], ["--adopt"]])
def test_prereg_refuses_to_rewrite_a_registered_bar(prereg, monkeypatch, capsys, argv):
    """The injection: a bar that moved. `build()` derives the bars FROM the eval slice, so a
    re-run after the slice moves recomputes them post-hoc while the artifact still says it
    was written first. --adopt must NOT be an escape hatch for that."""
    doc = json.loads(PREREG_REC.read_text(encoding="utf-8"))
    doc["arms"]["primary_census144"]["noninf_margin"] = 0.20
    before = _sha(PREREG_REC)
    assert _run_prereg(prereg, monkeypatch, argv, doc) == 2
    assert _sha(PREREG_REC) == before, "the record was rewritten despite the refusal"
    out = capsys.readouterr().out
    assert "REFUSING to rewrite data/v10/prereg_v10.json" in out
    assert "keys that differ: arms" in out


@pytest.mark.skipif(not PREREG_REC.exists(), reason="no committed v10 pre-registration")
def test_prereg_refuses_a_rewritten_amendment_but_allows_an_appended_one(prereg, monkeypatch,
                                                                        tmp_path, capsys):
    """"Append only; never rewrite an entry" is prose in the AMENDMENTS block. This is that
    sentence made checkable in both directions."""
    doc = json.loads(PREREG_REC.read_text(encoding="utf-8"))
    edited = json.loads(json.dumps(doc))
    edited["amendments"][0]["what_was_wrong"] = "rewritten after the fact"
    before = _sha(PREREG_REC)
    assert _run_prereg(prereg, monkeypatch, ["--adopt"], edited) == 2
    assert _sha(PREREG_REC) == before
    assert "is not an extension" in capsys.readouterr().out

    appended = json.loads(json.dumps(doc))
    appended["amendments"].append({"n": 99, "date": "2099-01-01"})
    dest = tmp_path / "prereg_v10.json"
    monkeypatch.setattr(prereg.paths, "durable", lambda rel, **kw: dest)
    assert _run_prereg(prereg, monkeypatch, ["--adopt"], appended) == 0
    assert json.loads(dest.read_text(encoding="utf-8"))["amendments"][-1]["n"] == 99


# RETIRED 2026-08-08 — `test_prereg_build_still_reproduces_the_committed_record`.
#
# It re-ran `prereg.build()` and asserted it reproduced the committed record byte for byte,
# which is what made `--adopt` safe. `build()` reads `data/v9/plan.jsonl`, demoted to bulk()
# on 2026-08-08 with the aug-cache trees it described, so the input is absent by design and
# the test could only pass after a rebuild. DELETED rather than made skip-on-missing: this
# file's whole subject is writes that silently falsify a record, and a guard that goes green
# because its input vanished is that failure in test form. The refusal half of the contract
# is untouched and still fast — `--adopt` will not rewrite an existing record, proved above
# with `build()` pinned.
#
# THE derive_t_good_v10 CASE WENT THE SAME WAY on 2026-08-09. It proved that the durable
# t_good table was written only under `--adopt`; the estimator and all five per-version
# drivers were deleted with the per-partition machinery, and `data/v10/t_good_derivation.json`
# is now a record nothing writes. There is no write left to gate.
