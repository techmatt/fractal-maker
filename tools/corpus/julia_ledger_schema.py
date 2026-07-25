"""Julia ledger-row schema tag — the enforced form of a previously prose-only invariant.

A julia discovery row stores its (viewport, parameter-c) pair in one of two field
layouts, and the SAME field names mean different things in each:

  * CAMPAIGN  (steered_frontier arm): `outcome_cx/cy` is the z-plane VIEWPORT, the
                parameter `c` is `julia_c_re/julia_c_im`, and there are no `julia_z_*`
                fields.
  * WALK      (production_seeder / julia-hook arm): `outcome_cx/cy` IS the parameter
                `c` (the cloud repels on c), and the z-plane VIEWPORT is `julia_z_cx/
                julia_z_cy/julia_z_fw`.

Both writing arms are live — this is not a migration to a dead format — so for years the
only thing distinguishing the two was a reader that inferred the era from which fields
were present, plus a paragraph of documentation. `schema_of` replaces that inference with
an explicit per-row tag (`julia_schema`): both writers stamp it, the existing ledgers were
back-stamped (`tools/corpus/backstamp_julia_schema.py`, era detected once from field
presence and round-trip-verified), and every reader now ASSERTS the tag. An untagged or
unknown-tagged julia row is a loud failure, never a guess.

`detect_schema` (field-presence inference) exists ONLY for the one-time back-stamp; the
live path is `schema_of` (asserting) + `viewport_and_c` (the canonical resolver). Native
(mandelbrot/multibrot) and phoenix rows carry no ambiguity and are untouched.
"""
from __future__ import annotations

# The per-row tag key and its two legal values.
SCHEMA_KEY = "julia_schema"
CAMPAIGN = "campaign"   # outcome_cx/cy = viewport; c = julia_c_re/julia_c_im
WALK = "walk"           # outcome_cx/cy = c;        viewport = julia_z_cx/cy/fw
KNOWN = frozenset({CAMPAIGN, WALK})


def is_julia_row(row: dict) -> bool:
    """True for a julia ledger row (ledger `family` is `julia:mandelbrot`,
    `julia:multibrot{d}`, …). Native and phoenix rows return False and are never tagged
    or resolved through this module."""
    return str(row.get("family", "")).startswith("julia")


def _has(row: dict, *keys) -> bool:
    return all(row.get(k) is not None for k in keys)


def detect_schema(row: dict) -> str:
    """Infer a julia row's schema from field presence. BACK-STAMP ONLY — the live path is
    `schema_of`. A row that carries BOTH layouts' discriminating fields, or NEITHER, is a
    contradiction and raises (the risk the back-stamp verification is meant to catch)."""
    if not is_julia_row(row):
        raise ValueError(f"detect_schema on a non-julia row (family={row.get('family')!r})")
    has_walk = _has(row, "julia_z_cx", "julia_z_cy", "julia_z_fw")
    has_campaign = _has(row, "julia_c_re", "julia_c_im")
    if has_walk and has_campaign:
        raise ValueError(
            f"ambiguous julia row {row.get('id')!r}: carries BOTH julia_z_* (walk) and "
            f"julia_c_* (campaign) fields — cannot infer era")
    if has_walk:
        return WALK
    if has_campaign:
        return CAMPAIGN
    raise ValueError(
        f"undetectable julia row {row.get('id')!r}: neither julia_z_* (walk) nor "
        f"julia_c_* (campaign) fields present")


def schema_of(row: dict) -> str:
    """The ASSERTED schema of a julia row: read the stamped `julia_schema` tag. Raises on an
    untagged or unknown-tagged row — the enforcement that lets the two-schema warning stop
    being carried. This is what every live reader calls (never `detect_schema`)."""
    tag = row.get(SCHEMA_KEY)
    if tag is None:
        raise ValueError(
            f"untagged julia row {row.get('id')!r}: missing {SCHEMA_KEY!r}. Back-stamp the "
            f"ledger via tools/corpus/backstamp_julia_schema.py (both writers stamp new rows).")
    if tag not in KNOWN:
        raise ValueError(
            f"unknown {SCHEMA_KEY}={tag!r} on row {row.get('id')!r}; expected one of {sorted(KNOWN)}")
    return tag


def viewport_and_c(row: dict):
    """Canonical resolver for a julia row → `(cx, cy, fw, c_re, c_im)` as stored (types
    untouched; the caller stringifies). Asserts the tag via `schema_of`, then reads the
    fields the tag names — no shape inference. CAMPAIGN reads the viewport from `outcome_*`
    and c from `julia_c_*`; WALK reads the viewport from `julia_z_*` and c from `outcome_*`."""
    schema = schema_of(row)
    if schema == CAMPAIGN:
        return (row["outcome_cx"], row["outcome_cy"], row["outcome_fw"],
                row["julia_c_re"], row["julia_c_im"])
    # WALK
    return (row["julia_z_cx"], row["julia_z_cy"], row["julia_z_fw"],
            row["outcome_cx"], row["outcome_cy"])


def stamp(row: dict) -> bool:
    """Ensure a julia row carries its schema tag, inferring the era from field presence when
    absent (back-stamp helper). Returns True iff the row was mutated. Non-julia rows and
    already-tagged rows are left untouched. Raises (via `detect_schema`) on a contradictory
    row so the back-stamp fails loudly rather than guessing."""
    if not is_julia_row(row) or SCHEMA_KEY in row:
        return False
    row[SCHEMA_KEY] = detect_schema(row)
    return True
