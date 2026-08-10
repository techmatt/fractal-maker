#!/usr/bin/env python
"""The classic-phoenix ledger's RESUME predicate — the thing that makes a model flip re-mint.

One property, and it is the one a per-coord resume gets wrong by default: "done" means
DECODED BY THE ACTIVE HEAD, not "an id I have seen". `p_notbad` / `p_good` / `decoded_class`
are a specific head's verdict, so resuming on the id alone makes a flip a no-op — every coord
looks finished and the ledger keeps serving the previous head's admissions under the new
head's name. That is exactly the state `data/discovery/classic_phoenix` was in on 2026-08-04
(184/184 rows stamped v7 against an active v10 pin).

No model, no GPU, no engine: `purge_stale` is file arithmetic over the shared
`corpus_common.is_current_decoded` predicate.

  uv run pytest tools/phoenix/test_classic_phoenix_supply.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_common as cc                       # noqa: E402
import partitions as P                           # noqa: E402

# Coupled to production_pins.ACTIVE_CKPT: `pytest -m version_pinned` lists it.
pytestmark = pytest.mark.version_pinned


def _mod():
    """Imported lazily and per call: the module pulls the whole scoring tail at import, and
    the tests below only need two pure functions off it."""
    import classic_phoenix_supply as cps
    return cps


def _write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(i, version):
    return dict(id=f"clphx_{i:04d}", family="phoenix", decoded_class=3, p_good=0.6,
                p_notbad=0.9, guard_pass=True, scorer_version=version,
                outcome_cx="0.42", outcome_cy="-0.67", outcome_fw="0.51")


def test_a_stale_stamped_row_is_NOT_done_and_its_sidecars_go_with_it(tmp_path):
    """The whole point. A row from an older head is re-scored, and the two DERIVED sidecars
    are deleted rather than carried: the look tally is order-dependent (a near-dup test
    against whatever was admitted before it), so a tally built from the old head's admissions
    would report distinctness against looks this run never admits."""
    cps = _mod()
    active = cc.active_scorer_version()
    rescored = tmp_path / "rescored.jsonl"
    feats, looks = tmp_path / "outcome_feats.npz", tmp_path / "distinct_looks.npz"
    _write(rescored, [_row(0, "v7"), _row(1, "v7"), _row(2, active)])
    np.savez(feats, clphx_0000=np.zeros(4, "f4"))
    np.savez(looks, phoenix=np.zeros((1, 4), "f4"))

    done, dropped = cps.purge_stale(rescored, feats, looks)
    assert dropped == 2 and done == {"clphx_0002"}
    surviving = [json.loads(l) for l in rescored.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["id"] for r in surviving] == ["clphx_0002"]
    assert not feats.exists() and not looks.exists(), (
        "the derived sidecars were carried across a head flip — the tally is order-dependent")


def test_an_all_current_ledger_is_a_true_no_op(tmp_path):
    """Bracketing the fix on the other side: it must not over-correct. Nothing stale ->
    nothing rewritten, nothing deleted, every id still done. A purge that fired
    unconditionally would re-score the whole ledger on every resume."""
    cps = _mod()
    active = cc.active_scorer_version()
    rescored = tmp_path / "rescored.jsonl"
    feats, looks = tmp_path / "outcome_feats.npz", tmp_path / "distinct_looks.npz"
    _write(rescored, [_row(0, active), _row(1, active)])
    np.savez(feats, clphx_0000=np.zeros(4, "f4"))
    np.savez(looks, phoenix=np.zeros((1, 4), "f4"))
    before = rescored.read_bytes()

    done, dropped = cps.purge_stale(rescored, feats, looks)
    assert dropped == 0 and done == {"clphx_0000", "clphx_0001"}
    assert rescored.read_bytes() == before
    assert feats.exists() and looks.exists()


def test_an_unstamped_row_is_stale_too(tmp_path):
    """A row with NO `scorer_version` predates the stamp entirely; `is_current_decoded` is
    false for it, and treating absence as "probably fine" is how a pre-stamp verdict survives
    every future flip."""
    cps = _mod()
    rescored = tmp_path / "rescored.jsonl"
    r = _row(0, "x")
    r.pop("scorer_version")
    _write(rescored, [r])
    done, dropped = cps.purge_stale(rescored, tmp_path / "f.npz", tmp_path / "l.npz")
    assert dropped == 1 and done == set()


def test_a_missing_resume_file_is_an_empty_resume_not_a_crash(tmp_path):
    cps = _mod()
    assert cps.purge_stale(tmp_path / "nope.jsonl", tmp_path / "f.npz",
                           tmp_path / "l.npz") == (set(), 0)


def test_the_ledger_resolves_to_its_OWN_partition_and_the_pinned_point(tmp_path):
    """`classic_phoenix` IS `phoenix:classic` (registered 2026-08-04). That used to also decide
    WHICH THRESHOLD admitted its rows, and getting the key wrong once minted all 24 against
    `phoenix`'s 0.77 and took the partition to zero supply. There is one flat
    `floors.GOOD_FLOOR` now, so the partition no longer selects a cut and that class of bug is
    unwritable; the partition still decides the CELL, the release share and the supply note."""
    import production_seeder as ps
    cps = _mod()
    # every row this ledger writes resolves to the derived partition
    assert P.partition_of_row(dict(family="phoenix", outcome_cx="0", outcome_cy="0",
                                   outcome_fw="3.0")) == P.CLASSIC_PHOENIX
    # ...and nothing anywhere can pick a per-partition threshold for it any more
    assert not hasattr(ps, "t_good_for") and not hasattr(ps, "T_GOOD_UNCALIBRATED")
    # the module's classic identity is the registry's pinned point, not a second literal
    s = cps.CLASSIC_SEED
    assert (s.c.real, s.c.imag, s.p.real, s.p.imag, s.z_m1.real, s.z_m1.imag) \
        == P.PHOENIX_CLASSIC_POINT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
