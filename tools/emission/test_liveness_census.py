#!/usr/bin/env python
"""Stage-2 LIVENESS CENSUS — the standing test that turns "stage 2 silently collapsed" into a
red build.

WHY. Stage 2 has collapsed silently twice, and both times it was found by a human going
looking. The v10 head flip took the intake from ~1.4k admissible locations to **16** (every
non-classic ledger was still v7-stamped, so `is_current_decoded` correctly refused it) and
nothing went red — the driver ran, admitted 16 rows, and reported a healthy-looking run over
them. The union then aborted outright on 11 run-scoped id collisions. Neither is a bug in any
one function; both are the pipeline's INPUTS going away underneath code that still works.

WHAT THIS ASSERTS. The four things that must be true for stage 2 to be able to run at all,
checked against whatever is on disk today:

  1. the union loads, and every intake ledger still contributes admitted rows;
  2. the library seed loads FAIL-CLOSED-POSITIVE (it raises when absent, and it is not empty);
  3. every partition the ratio table demands has admitted supply;
  4. `release_mix` shares resolve over the live partition set and re-solve into a target
     measure without raising.

FLOORS ARE RELATIONAL, NEVER PINS. Every threshold here is expressed against something that
moves with the corpus — the per-ledger census, the partition registry, the ledgers' own
current-decode denominator — so a re-score that legitimately moves the counts leaves this
green while a collapse goes red. The one number that IS pinned (`test_intake_union.py`'s
`UNION_ADMITTED = 751`) is a census, deliberately exact, and is a different job: it says what
the union IS, this says the union is ALIVE. A pin breaks on every re-score; a floor does not,
which is why the standing guard is the floor and not the pin.

COST: no renders, no GPU, no head. Seconds.

  uv run pytest tools/emission/test_liveness_census.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring",
           ROOT / "tools" / "atlas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import cells as C                # noqa: E402
from tools.emission import descriptor as D           # noqa: E402
from tools.emission import ledger_rescore as LR      # noqa: E402
import corpus_common as cc                           # noqa: E402
import partitions as P                               # noqa: E402
import release_mix as RM                             # noqa: E402

LEDGERS = [(tag, LR.ledger_path(rel)) for tag, rel in LR.LEDGERS]
_MISSING = [str(p) for _t, p in LEDGERS if not p.exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"intake ledger absent: {_MISSING}")

# The collapse floor, as a FRACTION of the live current-decode denominator rather than a count.
# Calibrated against the failure it exists to catch, not against today's healthy number: the
# v10 flip admitted 16 of 1657 current rows (0.97%). Today's union sits at ~45%. 5% is far
# below any plausible healthy corpus and an order of magnitude above the collapse, so a real
# re-score moves freely inside it and a firewall shutting out six of seven ledgers does not.
MIN_ADMITTED_FRACTION = 0.05


@pytest.fixture(scope="module")
def census():
    """Per-ledger admitted / current counts + the union, from THE union reader the driver
    uses (`descriptor.load_union_admitted`) — not a mirror of it. A mirror is how "the intake
    admits N" became false while both numbers looked computed."""
    per_ledger = {}
    for tag, path in LEDGERS:
        resolved = D.resolve_rows(path)
        per_ledger[tag] = {
            "rows": len(resolved),
            "current": sum(1 for r in resolved if cc.is_current_decoded(r)),
            "admitted": len(D.load_admitted(path)),
        }
    rows, diag = D.load_union_admitted([p for _t, p in LEDGERS])
    return {"per_ledger": per_ledger, "rows": rows, "diag": diag}


# --------------------------------------------------------------------------- #
# 1. the union is reachable, and no ledger has silently stopped contributing
# --------------------------------------------------------------------------- #
def test_the_union_loads_and_is_the_per_ledger_sum(census):
    diag, rows = census["diag"], census["rows"]
    per_ledger_sum = sum(v["admitted"] for v in census["per_ledger"].values())
    assert len(rows) == diag["n_union"] == per_ledger_sum - diag["n_location_overlaps"]
    assert diag["n_union"] > 0


# Intake ledgers KNOWN to contribute zero, each with the date, the cause and the remedy.
# A registration here is NOT an exemption — it is the opposite of one. The assertion below is
# "no ledger is UNEXPECTEDLY empty", and a registered entry still has to be true: an entry that
# starts contributing again goes red, so the note cannot outlive its subject.
#
# `classic_phoenix` (2026-08-08, the v11 flip). Its 24 rows re-score cleanly under v11 —
# 19 decode class 1, 5 class 2 — and NONE reaches q3, so `load_admitted` refuses all of them.
# This is the HEAD, not a stale decode: the partition runs at the 0.50 UNCALIBRATED baseline
# under v11 exactly as it did under v10, and the re-score is current. The first read of it was
# worse and wrong — a partition-key bug in `ledger_rescore` minted all 24 against `phoenix`'s
# new 0.77 and decoded every one to class 1; that is fixed, and the remaining zero is real.
# `phoenix:classic` is EXTERNALLY SUPPLIED — no crawl produces it — so the remedy is a supply
# run, not a threshold: `production_seeder.py --run-phoenix` then `classic_phoenix_supply.py`.
# The release mix asks ~12 of a 779-row intake (1.52%) and gets 0.
KNOWN_EMPTY = {"classic_phoenix"}


def test_every_intake_ledger_still_contributes(census):
    """THE v10-flip detector. A version firewall shutting a ledger out is invisible in the
    total (the others absorb it) and total here: six of seven went to zero and the run
    proceeded. Stated per ledger so the failure names which one died."""
    dead = {tag: v for tag, v in census["per_ledger"].items()
            if v["admitted"] == 0 and tag not in KNOWN_EMPTY}
    assert not dead, (
        f"{len(dead)} intake ledger(s) contribute NO admitted rows: {sorted(dead)}. Either the "
        f"decode block went stale under a head flip (re-run tools/emission/ledger_rescore.py) "
        f"or the ledger itself is gone. Stage 2 will run either way, on a corpus that quietly "
        f"lost a family.")


def test_no_known_empty_ledger_has_quietly_come_back(census):
    """The other half of KNOWN_EMPTY, and the half that keeps it from rotting into a blanket
    exemption. A registered ledger that starts admitting again is good news the registration
    must not swallow — delete its entry (and its prose) rather than leaving a note that
    describes a state the tree has left."""
    revived = {tag: census["per_ledger"][tag]["admitted"] for tag in KNOWN_EMPTY
               if census["per_ledger"].get(tag, {}).get("admitted", 0) > 0}
    assert not revived, (
        f"{revived} — these are registered in KNOWN_EMPTY as contributing nothing and they "
        f"now contribute. Remove the registration; the guard above should carry them again.")
    assert KNOWN_EMPTY <= set(census["per_ledger"]), (
        f"KNOWN_EMPTY names a ledger the census does not: "
        f"{sorted(KNOWN_EMPTY - set(census['per_ledger']))}")


def test_the_admitted_share_is_above_the_collapse_floor(census):
    """Relational scale floor: admitted as a fraction of the ledgers' own current-decode rows.
    Not a count — a count is a pin that a legitimate re-score breaks."""
    current = sum(v["current"] for v in census["per_ledger"].values())
    assert current > 0, "no ledger row decodes as current — the whole intake is stale"
    frac = census["diag"]["n_union"] / current
    assert frac >= MIN_ADMITTED_FRACTION, (
        f"union admits {census['diag']['n_union']}/{current} = {frac:.2%} of current-decode "
        f"rows, under the {MIN_ADMITTED_FRACTION:.0%} collapse floor. The v10 flip looked like "
        f"this (0.97%).")


# --------------------------------------------------------------------------- #
# 2. the seed is fail-closed POSITIVE
# --------------------------------------------------------------------------- #
def test_the_library_seed_loads_and_is_not_empty():
    """Fail-closed-POSITIVE: both halves. It must RAISE when the snapshot is absent (the
    negative half, below) *and* actually yield medoids here — a seed loader that silently
    returns `{}` is an unseeded intake wearing the seeded intake's report."""
    med, prior, note = D.load_library_seed(None)
    assert med and sum(len(v) for v in med.values()) > 0, note
    assert prior, "the seed has medoids but no prior assignment — nothing can be held unmoved"
    # every seeded type is a registered partition, or the seed cannot key the cell axis.
    assert set(med) <= set(P.ALL_FAMS), sorted(set(med) - set(P.ALL_FAMS))


def test_an_absent_seed_raises_rather_than_degrading(tmp_path):
    """The negative half, in the suite rather than in a comment. This used to print a note and
    continue, which is how campaign-1's seeding degraded silently."""
    with pytest.raises(D.LibrarySeedUnavailable):
        D.load_library_seed(tmp_path / "nothing_here")


# --------------------------------------------------------------------------- #
# 3. every partition the ratio table demands has supply
# --------------------------------------------------------------------------- #
def test_every_demanded_partition_has_admitted_supply(census):
    """The ratio table is a claim about what a release should contain. A partition with a
    positive ratio and zero admitted rows is demand with no supply — servable only by
    renormalizing it away, which reads as "that partition had no demand"."""
    from collections import Counter
    have = Counter(D.cell_partition(r) for r in census["rows"])
    demanded = [p for p, v in RM.RATIO.items() if v > 0]
    # The partitions a KNOWN_EMPTY ledger is the sole supplier of are starved BY THAT ENTRY,
    # not independently — reporting them here would be the same fact a second time, and it is
    # the ledger registration that carries the date, the cause and the remedy.
    known_starved = {P.CLASSIC_PHOENIX} if "classic_phoenix" in KNOWN_EMPTY else set()
    starved = sorted(p for p in demanded if have[p] == 0 and p not in known_starved)
    assert not starved, (
        f"partition(s) {starved} are demanded by release_mix.RATIO and have zero admitted "
        f"rows. Their share is silently renormalized away at every target solve. Live counts: "
        f"{dict(sorted(have.items()))}")
    # and nothing admitted is OFF the registry — the cell axis has to key every row it is handed.
    assert set(have) <= set(P.ALL_FAMS), sorted(set(have) - set(P.ALL_FAMS))


# --------------------------------------------------------------------------- #
# 4. release_mix shares resolve, and re-solve into a measure
# --------------------------------------------------------------------------- #
def test_release_mix_shares_resolve_over_the_live_partitions(census):
    observed = sorted({D.cell_partition(r) for r in census["rows"]})
    shares = RM.shares(observed)
    assert set(shares) == set(observed)
    assert all(v > 0 for v in shares.values()), shares
    assert abs(sum(shares.values()) - 1.0) < 1e-12


def test_the_target_measure_solves_and_leaves_nothing_unserved(census):
    """The end of the chain: shares → per-cell weights, against a feasible set built the way
    the driver builds it. Uses one synthetic flavor × one style — the target's per-partition
    realized share is denominator-invariant, so it does not need the live palette roster (and
    must not, or this test would drag the palette library into the light lane)."""
    observed_pc = sorted({(D.cell_partition(r), r["id"]) for r in census["rows"]})
    feasible = C.build_feasible_cells(observed_pc, ["f"], ["smooth"])
    observed = sorted({p for p, _c in observed_pc})
    shares = RM.shares(observed)
    target = C.TargetMeasure.from_partition_shares(shares, feasible)
    assert target.unrealized_shares(shares) == {}, \
        "a demanded partition has no feasible cell this intake"
    realized = target.partition_shares()
    for p, want in shares.items():
        assert abs(realized[p] - want) < 1e-12, (p, realized[p], want)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
