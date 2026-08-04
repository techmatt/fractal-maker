#!/usr/bin/env python
"""The forced-eval group cascade and its score-unconditioned exemption.

`assign_split_by_group` is where a registration turns into a split, and it is the only
place in the pipeline that DELETES a labeled location. Until 2026-08-04 a biased location
sharing a neighborhood group with a forced-eval one was dropped outright; under the live
registry that cost 687 train locations (193 threes, 50 fours). The exemption keeps them as
TRAIN for instruments whose draw was score-unconditioned.

Every test here brackets the change on both sides (`verification_practice.md` §3): the old
behaviour on an unflagged instrument, the new behaviour on a flagged one, and the cases
where the exemption must NOT reach. The fixtures are synthetic because the live corpus
cannot produce a non-exempt instrument at all — all four current instruments are flagged,
so the drop branch would be untested by any corpus-driven fixture (§6, the fixture that
cannot fail).

  uv run pytest tools/v8/test_group_cascade.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import batch_registry as br                      # noqa: E402
from tools.v8 import build_manifest as v8b       # noqa: E402


def loc(gid, *, instrument=False, biased=False, unconditioned=False, tag=""):
    """A location as `classify_location` leaves it, reduced to what the cascade reads."""
    return {"group_id": gid, "forced_eval": instrument, "biased": biased,
            "score_unconditioned": instrument and unconditioned,
            "exempt_group_mate": False, "tag": tag}


def by_tag(rows):
    return {d["tag"]: d for d in rows}


# =========================================================================== #
# the exemption
# =========================================================================== #
def test_a_biased_group_mate_of_a_FLAGGED_instrument_lands_train():
    """The injected case, positive side. Previously this location was deleted."""
    members = [loc(1, instrument=True, unconditioned=True, tag="instrument"),
               loc(1, biased=True, tag="mate")]
    kept, dropped = v8b.assign_split_by_group(members)
    assert dropped == []
    k = by_tag(kept)
    assert k["instrument"]["split"] == "eval"
    assert k["mate"]["split"] == "train"
    assert k["mate"]["exempt_group_mate"] is True


def test_the_same_group_mate_is_DROPPED_when_the_instrument_is_not_flagged():
    """The injected case, negative side — one field differs from the test above. Without
    this the test above would pass against code that never drops anything."""
    members = [loc(1, instrument=True, unconditioned=False, tag="instrument"),
               loc(1, biased=True, tag="mate")]
    kept, dropped = v8b.assign_split_by_group(members)
    assert [d["tag"] for d in dropped] == ["mate"]
    assert dropped[0]["split"] is None
    assert by_tag(kept)["instrument"]["split"] == "eval"


def test_a_group_with_a_MIX_of_flagged_and_unflagged_instruments_still_drops():
    """`all`, not `any`: one instrument in the group that did not claim the exemption is
    enough to keep the protection, because it is that instrument's read being protected."""
    members = [loc(1, instrument=True, unconditioned=True, tag="flagged"),
               loc(1, instrument=True, unconditioned=False, tag="unflagged"),
               loc(1, biased=True, tag="mate")]
    kept, dropped = v8b.assign_split_by_group(members)
    assert [d["tag"] for d in dropped] == ["mate"]


def test_a_biased_INSTRUMENT_location_is_dropped_even_under_the_exemption():
    """The exemption moves a biased NEIGHBOUR to train. It must never move an instrument's
    own location there — that would put an eval row in the training population, which no
    reading of the decision asks for."""
    members = [loc(1, instrument=True, unconditioned=True, tag="clean"),
               loc(1, instrument=True, unconditioned=True, biased=True, tag="tainted")]
    kept, dropped = v8b.assign_split_by_group(members)
    assert [d["tag"] for d in dropped] == ["tainted"]
    assert by_tag(kept)["clean"]["split"] == "eval"


def test_a_group_with_no_instrument_is_all_train_and_nothing_is_exempted():
    members = [loc(2, biased=True, tag="a"), loc(2, tag="b")]
    kept, dropped = v8b.assign_split_by_group(members)
    assert dropped == []
    assert all(d["split"] == "train" for d in kept)
    assert not any(d["exempt_group_mate"] for d in kept)


def test_an_unbiased_non_instrument_neighbour_still_follows_its_group_to_eval():
    """Unchanged behaviour, asserted because the exemption edits this branch's neighbour.
    GATE 5 is what refuses such a location downstream — the cascade does not silently
    reclassify it."""
    members = [loc(3, instrument=True, unconditioned=True, tag="instrument"),
               loc(3, tag="clean_neighbour")]
    kept, _ = v8b.assign_split_by_group(members)
    assert by_tag(kept)["clean_neighbour"]["split"] == "eval"


# =========================================================================== #
# the straddle gate the exemption relaxes
# =========================================================================== #
def test_an_exempt_straddle_is_reported_exempt_and_any_other_straddle_is_illegal():
    """GATE 3 stops being unconditional, so the two halves are asserted apart: the
    exemption's own straddle is counted, and a straddle with a non-exempt eval member is
    still an abort."""
    ok = [loc(1, instrument=True, unconditioned=True, tag="i"), loc(1, biased=True, tag="m")]
    kept_ok, _ = v8b.assign_split_by_group(ok)
    illegal, exempt = v8b.straddle_report(kept_ok)
    assert illegal == [] and exempt == [1]

    # hand-built: an unflagged instrument in eval beside a train member in the same group
    bad = [dict(loc(9, instrument=True, tag="i"), split="eval"),
           dict(loc(9, biased=True, tag="m"), split="train")]
    illegal, exempt = v8b.straddle_report(bad)
    assert illegal == [9] and exempt == []


def test_straddle_report_is_silent_on_a_group_that_does_not_straddle():
    rows = [dict(loc(4, tag="a"), split="train"), dict(loc(4, tag="b"), split="train")]
    assert v8b.straddle_report(rows) == ([], [])


# =========================================================================== #
# the flag is registry data, not a batch list in the cascade
# =========================================================================== #
def test_the_cascade_reads_the_flag_off_the_location_and_names_no_batch():
    """The decision is data on a registration (`batch_registry`), so the cascade must be
    able to run on locations whose batches do not exist. If it ever grew a batch list the
    fixtures above — which name no batch at all — would stop working."""
    src = (ROOT / "tools/v8/build_manifest.py").read_text(encoding="utf-8")
    body = src[src.index("def assign_split_by_group"):src.index("def straddle_report")]
    assert "score_unconditioned" in body
    assert not re.search(r"""["']20\d\d-\d\d-\d\d_[A-Za-z0-9_]+["']""", body), \
        "the cascade names a corpus batch"


@pytest.mark.parametrize("bid", sorted(br.eval_eligible_batches()))
def test_every_live_instrument_is_flagged_so_the_drop_branch_is_currently_unreachable(bid):
    """Why the fixtures above are synthetic, stated as an assertion rather than a comment.
    When this goes red a real instrument has declined the exemption and the drop rule is
    live again for its groups."""
    assert any(r.score_unconditioned for r in br.REGISTRY[bid] if r.eval_eligible), bid


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
