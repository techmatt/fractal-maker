#!/usr/bin/env python
"""The derived atom key must agree with every stored one, on the LIVE corpus.

This is the whole warrant for `atom_identity.atom_key_of_row`: the v11 manifest unions
split groups on a key it computes from the render block instead of one a batch builder may
or may not have written, and that substitution is only legitimate while the two agree
EVERYWHERE they are both defined. So the comparison is re-run over whatever the corpus holds
rather than pinned to the 1310 rows that held at the time (`verification_practice.md` §5 —
a number frozen into an assertion stops tracking the population it describes).

The paired negative control matters as much as the agreement: a derivation that agreed by
returning the stored value would pass the first assertion trivially. `atom_key_of_row` never
reads the column, and `test_derivation_does_not_consult_the_stored_column` proves it by
mutating the column and requiring the derived key not to move.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

import atom_identity as ai  # noqa: E402

BATCHES_GLOB = str(ROOT / "data" / "label_corpus" / "batches" / "*" / "images.jsonl")


def _corpus_rows():
    for p in sorted(glob.glob(BATCHES_GLOB)):
        bid = os.path.basename(os.path.dirname(p))
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield bid, json.loads(line)


@pytest.fixture(scope="module")
def carriers():
    """Every corpus row that stores an `atom_key`, with its derived key beside it."""
    out = []
    for bid, row in _corpus_rows():
        stored = ai.stored_atom_key(row)
        if stored:
            out.append((bid, row["image_id"], stored, ai.atom_key_of_row(row)))
    return out


def test_derived_key_matches_every_stored_key(carriers):
    assert carriers, "no corpus row stores an atom_key — the comparison is vacuous"
    missing = [(b, i) for b, i, _s, d in carriers if d is None]
    assert not missing, (
        f"{len(missing)} row(s) store an atom_key but do not carry the maneuver signature "
        f"{ai.MANEUVER_SIGNATURE}, so the derivation declines to produce one: {missing[:5]}")
    drift = [(b, i, s, d) for b, i, s, d in carriers if s != d]
    assert not drift, (
        f"{len(drift)} of {len(carriers)} stored atom_keys disagree with the key derived "
        f"from the row's own render block. The v11 split-group union keys on the DERIVED "
        f"value, so a disagreement means some location's atom identity is now a different "
        f"one than the batch that wrote it believed: {drift[:3]}")


def test_derivation_does_not_consult_the_stored_column(carriers):
    """The proved-red control. Corrupt the column; the derived key must be unmoved."""
    bid = carriers[0][0]
    path = ROOT / "data" / "label_corpus" / "batches" / bid / "images.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert ai.stored_atom_key(row), f"{bid}'s first row does not store a key"
    before = ai.atom_key_of_row(row)
    row["provenance"]["atom_key"] = "0,0"           # in memory only; nothing is written back
    assert ai.atom_key_of_row(row) == before, \
        "atom_key_of_row moved when the stored column moved — it is reading the column"


def test_scope_excludes_non_maneuver_rows():
    """A row without the maneuver signature gets no key, however much else it carries.

    `2026-08-03_q4_near_minibrot_v1` is the live instance: 290 julia rows that carry an
    `atom_id` and no `degree`. Their nucleus is the seed `c`, not the frame centre, so a
    key derived from the centre would be a wrong answer rather than a missing one."""
    scoped = [(b, r) for b, r in _corpus_rows()
              if (r.get("provenance") or {}).get("atom_id")
              and not ai.has_maneuver_signature(r.get("provenance"))]
    assert scoped, "no in-scope negative case in the corpus — the guard is vacuous"
    assert all(ai.atom_key_of_row(r) is None for _b, r in scoped)
