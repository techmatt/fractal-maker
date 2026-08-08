#!/usr/bin/env python
"""THE atom identity of a corpus row — DERIVED from the row, never read from a column.

WHY THIS EXISTS. `classifier_retrain_protocol.md` §2's standing rule is that children
inherit their seed's split, and for a maneuver view the seed is the **minibrot nucleus** it
was framed around. The spatial union-find in `v8/build_manifest.assign_groups` cannot see
that relation: it unions only when two frame widths are within 1.5x, and two views of ONE
atom at different maneuver `k` (k=4 vs k=16 is 4x in `fw`) are two framings of the same
subject at four times the width. So the manifest build unions groups on an atom key, and
that key has to come from somewhere.

Until now it came from `provenance.atom_key`, a column a batch builder opts into. Six
batches (2026-08-01/02) carry it; the four maneuver batches built since do not, and
`tools/atlas/sitting_cutter.py` withholds it DELIBERATELY — its note says enlisting a fresh
1,000-row train batch in the union is an eval-contamination question nobody had re-argued.
The column being optional is the actual defect: whether a location's atom participates in
the split-group rule ends up depending on which builder wrote it.

THE KEY IS IN THE ROW ALREADY. A maneuver view's frame is centred ON the nucleus — the
framing operators (`neighborhood_expand`, `snap_at_seed`, `snap_to_nucleus`) set `fw =
k · window_scale` about the atom and do not displace the centre. So

    atom_key == deep_center_finder.snapped_dedup_key(render.cx, render.cy, degree, DEDUP_DPS)

is a pure read of the render block, which is version-invariant and cannot be forgotten by a
builder. Verified against every stored key in the corpus: **1310/1310 exact string
agreement**, zero near-misses (`test_atom_identity.py`, which re-runs the comparison over
whatever the corpus holds rather than pinning the number).

SCOPE, and it is narrow on purpose. The derivation is applied ONLY to rows carrying the
maneuver-view signature — `provenance.degree` AND `provenance.window_scale` — because that
pair is what asserts "this frame is `k·window_scale` about a nucleus". Run on an arbitrary
location the same call returns a 22-significant-digit rounding of an ordinary centre: never
colliding with anything, so never unioning anything, but a meaningless key dressed as an
atom identity. Two consequences worth naming:

  * `2026-08-03_q4_near_minibrot_v1` carries `atom_id` and no `degree` — its 290 rows are
    JULIA locations near a minibrot, so the nucleus is the seed `c`, not the frame centre.
    They are correctly out of scope here; seed-`c` overlap is already a manifest gate.
  * `atom_id` (`triage_store.atom_id`, a sha256 of the WRITE-time dedup key) is NOT usable
    as the union key in its place: it hashes the pre-snap key, so the per-solve axis-noise
    copies that `snapped_dedup_key` collapses land on different ids. It is also unresolvable
    for most of the corpus — 38 of 348 distinct ids survive in any store on disk.

    from atom_identity import atom_key_of_row, MANEUVER_SIGNATURE
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "tools" / "sourcing"))

import deep_center_finder as dcf  # noqa: E402

# The significant-digit width every stored dedup key / atom id was formed at. Taken from
# `build_minibrot_roster`, which is the ROOT owner — `atom_lib.DEDUP_DPS` and
# `triage_store.DEDUP_DPS` are both re-exports of this one. The read-time key MUST round at
# the same width or it lands beside the stored key instead of on it, so it is imported
# rather than restated: a second literal 22 is a second policy.
from build_minibrot_roster import DEDUP_DPS  # noqa: E402

#: The provenance keys whose joint presence means "this frame is k*window_scale about a
#: nucleus". Named so a caller can state the scope instead of re-deriving it.
MANEUVER_SIGNATURE = ("degree", "window_scale")


def has_maneuver_signature(prov: dict | None) -> bool:
    """True iff this row's provenance asserts a maneuver framing (see MANEUVER_SIGNATURE)."""
    p = prov or {}
    return all(p.get(k) is not None for k in MANEUVER_SIGNATURE)


def atom_key_of_row(row: dict) -> str | None:
    """The canonical atom key of a corpus row, or None if it is not a maneuver view.

    Derived from `render.cx/cy` + `provenance.degree`. Deliberately does NOT consult
    `provenance.atom_key`: a derived read that silently falls back to the stored column
    would hide exactly the disagreement the test exists to detect."""
    prov = row.get("provenance") or {}
    if not has_maneuver_signature(prov):
        return None
    rd = row.get("render") or {}
    if rd.get("cx") is None or rd.get("cy") is None:
        return None
    return dcf.snapped_dedup_key(rd["cx"], rd["cy"], int(prov["degree"]), DEDUP_DPS)


def stored_atom_key(row: dict) -> str | None:
    """The row's stored `provenance.atom_key`, if it opted into the column."""
    return (row.get("provenance") or {}).get("atom_key") or None
