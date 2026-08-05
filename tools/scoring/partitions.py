"""THE fractal_type ⟷ ledger-partition map. One copy, imported everywhere.

`fractal_type` is the Rust render-family token that lands in a manifest / eval-slice row
(`mandelbrot`, `julia_multibrot4`, …). A **partition** (a.k.a. "family" in the ledgers) is
what everything downstream is keyed on — `production_seeder.t_good_for`, the keeper cut,
the coverage clouds, the per-partition derivations. They differ in TWO places now:
the julia planes are namespaced (`julia_multibrot4` -> `julia:multibrot4`) so a julia-plane
partition is never confused with its native twin, and `phoenix` splits on its PARAMETER
POINT into `phoenix:classic` (the pinned Ushiki plane) and `phoenix` (varied) — see the
DERIVED PARTITIONS block below.

WHY THIS MODULE EXISTS. These nine pairs had **seven literal copies** on 2026-08-02 —
`derive_t_good_{v7,v8}`, `keeper_cut.py` (whose comment said "mirrors
derive_t_good.FT2FAM", which is the duplication admitting itself), `v7/build_manifest.py`
(as an inverted `FAM2FT`), and three test files — plus an eighth inlined into throwaway
analysis scripts every time someone needed to group eval rows by family. Nine key-value
pairs that have not changed across four classifier versions do not need seven owners; what
they need is one, and a test that says so. `tools/scoring/test_partitions.py` fails if a
second literal copy appears.

It lives under `tools/scoring/` because the partition is a *scoring* concept: every live
consumer is a threshold, a decode or a ledger keyed on it. It has no imports, so any module
that can put `tools/scoring` on `sys.path` can take it.

    from partitions import FT2FAM, ALL_FAMS, partition_of
    from partitions import partition_of_row      # the ROW-aware resolver (phoenix:classic)
"""
from __future__ import annotations

# fractal_type (Rust kind_str) -> ledger partition key.
FT2FAM = {
    "mandelbrot": "mandelbrot",
    "julia": "julia:mandelbrot",
    "multibrot3": "multibrot3",
    "multibrot4": "multibrot4",
    "multibrot5": "multibrot5",
    "julia_multibrot3": "julia:multibrot3",
    "julia_multibrot4": "julia:multibrot4",
    "julia_multibrot5": "julia:multibrot5",
    "phoenix": "phoenix",
}

# The inverse. Derived, never hand-written — a hand-written inverse is how two maps that are
# supposed to be inverses stop being inverses. The map is injective (asserted in the tests).
# It inverts FT2FAM and ONLY FT2FAM: a DERIVED partition has no fractal_type of its own and
# is deliberately absent here, so `FAM2FT[p]` on one raises instead of guessing. Route a
# derived partition through `base_partition` first.
FAM2FT = {v: k for k, v in FT2FAM.items()}

# =========================================================================== #
# DERIVED PARTITIONS — a partition with no `fractal_type` of its own.
#
# `phoenix:classic` (Matt, 2026-08-04). Classic phoenix is the full pinned Ushiki parameter
# point c=(0.5667, 0), p=(-0.5, 0), z_{-1}=0. It is structurally a pinned-parameter
# DYNAMICAL-plane family — the same shape as the julia twins, which are already namespaced
# for exactly this reason — while `phoenix` from now on means VARIED phoenix only. They are
# different supply (41 distinct looks total vs a swept 6-D grid), different scarcity and
# different objectives, so they are different partitions.
#
# The split is on the PARAMETER POINT, not on a token: both render as `fractal_type ==
# "phoenix"` and both are rendered by the same Rust family. So `FT2FAM` stays a bijection
# over render families (nine pairs, untouched, still the thing `FAM2FT` inverts) and the
# derived partition is resolved ROW-SIDE by `partition_of_row`. `DERIVED_FAMS` maps a derived
# partition to the base partition it splits off, which is also how a consumer that needs a
# RENDER family for it gets one (`base_partition` -> `FAM2FT`).
#
# READER-SIDE ONLY. Nothing rewrites, re-keys or re-stamps a stored row — every classic row
# in the tree today (84 corpus rows, 58 ledger rows) is either explicitly stamped with the
# point or carries NO parameter axes at all, and the absent-axes form resolves to these same
# legacy defaults byte-for-byte (`production_seeder.row_phoenix_key`, the identity machinery
# this mirrors). Verified 2026-08-04: 84/84 corpus classic rows and 58/58 ledger rows land on
# the pinned point under EXACT float equality, with zero near-misses — so there is no
# tolerance here and there must never be one.
# =========================================================================== #
CLASSIC_PHOENIX = "phoenix:classic"

# The pinned Ushiki point as a flat 6-tuple in canonical axis order
# (c_re, c_im, p_re, p_im, zm1_re, zm1_im). THE one copy — `production_seeder`'s
# PHOENIX_{C,P,ZM1}_DEFAULT are re-exports of these, so the dup-identity machinery and the
# partition resolver cannot drift onto two different "classic" points.
PHOENIX_CLASSIC_POINT = (0.5667, 0.0, -0.5, 0.0, 0.0, 0.0)

# derived partition -> the base partition it splits off.
DERIVED_FAMS = {CLASSIC_PHOENIX: "phoenix"}

# Every partition that can reach production, in the canonical report order (native planes
# first, then the julia planes, then phoenix, then the derived phoenix split). Derivations
# walk this to stamp the partitions that got NO eval rows at all — a family that is silently
# absent from a table and a family explicitly stamped UNCALIBRATED are different states.
ALL_FAMS = [
    "mandelbrot", "julia:mandelbrot",
    "multibrot3", "multibrot4", "multibrot5",
    "julia:multibrot3", "julia:multibrot4", "julia:multibrot5",
    "phoenix", CLASSIC_PHOENIX,
]

# The two on-disk spellings of a phoenix parameter point, per axis, in the canonical order of
# `PHOENIX_CLASSIC_POINT`: (discovery-ledger key, corpus-render-block key). A row carries ONE
# of the two schemas — the ledger stamp (`production_seeder.phoenix_ident_fields`) or the
# render block's `family_params` (`location.FAMILY_PARAM_KEYS['phoenix']` plus the primary
# constant in `c_re`/`c_im`). They are read as a SCHEMA, never per-axis: mixing one axis from
# each is how a row acquires a parameter point neither writer ever wrote.
_PHOENIX_AXIS_KEYS = (
    ("phoenix_c_re", "c_re"),
    ("phoenix_c_im", "c_im"),
    ("phoenix_p_re", "p_re"),
    ("phoenix_p_im", "p_im"),
    ("phoenix_zm1_re", "zm1_re"),
    ("phoenix_zm1_im", "zm1_im"),
)

# Positive markers that a row is IN one of those two schemas at all, even with every
# parameter axis absent. This is the load-bearing distinction and it is not pedantry:
#   * a legacy corpus row has no axes because it PREDATES them, and absent there means the
#     pinned point — that is the whole classic population (84 of 84 rows on 2026-08-04);
#   * an eval-slice row (`data/<v>/eval_scores_<v>.jsonl`) has no axes because the SCHEMA
#     has no such field, and absent there means UNKNOWN.
# Reading the second as the first would annex every varied-phoenix eval row into a partition
# holding 41 looks. So a schema that cannot express a parameter point is refused rather than
# guessed at, and its consumers stay on the token-only `partition_of`.
_LEDGER_MARKERS = ("outcome_cx", "outcome_fw")     # discovery outcome ledger
_RENDER_MARKERS = ("cx", "fw")                     # corpus images.jsonl render block


def _phoenix_schema(row):
    """`0` (ledger stamp), `1` (render block), or `None` — which of the two on-disk schemas
    `row` is in. An explicit axis wins over the structural marker, so a row carrying both
    markers and one schema's axes is read as that schema."""
    for i in (0, 1):
        if any(row.get(axis[i]) is not None for axis in _PHOENIX_AXIS_KEYS):
            return i
    if all(row.get(m) is not None for m in _LEDGER_MARKERS):
        return 0
    if all(row.get(m) is not None for m in _RENDER_MARKERS):
        return 1
    return None


def base_partition(partition: str) -> str:
    """The base partition a (possibly derived) partition belongs to. Identity for the nine
    base partitions; `phoenix:classic` -> `phoenix`. This is what a consumer needing a
    RENDER family, a walk grammar or a `fractal_type` for a derived partition goes through
    (`FAM2FT[base_partition(p)]`)."""
    return DERIVED_FAMS.get(partition, partition)


def _registered(partition: str) -> str:
    """Return `partition`, having proved it is REGISTERED in `ALL_FAMS` first.

    The ordering guard. A resolver that can emit a partition key the per-partition tables
    were never extended for produces a silent tenth bucket: the quota never floors it, the
    low-water never covers it, the t_good table has no row for it, and every one of those
    reads as "that partition had nothing" rather than as a missing registration. So
    resolution refuses to invent a partition — raise here and the omission is a crash at the
    first row, not a run that quietly under-serves one family. Reads `ALL_FAMS` at call time
    on purpose, so the guard is provable red by removing the registration."""
    if partition not in ALL_FAMS:
        raise KeyError(
            f"partition {partition!r} is not registered in partitions.ALL_FAMS. Register it "
            f"there (and extend every per-partition table — pop-quota floors, low-water, "
            f"MACHINE_1_DISCARD, ROUTES, t_good/tau_h) BEFORE any resolver can emit it.")
    return partition


def phoenix_point(row) -> tuple:
    """A phoenix row's `(c_re, c_im, p_re, p_im, zm1_re, zm1_im)` parameter point as floats,
    resolving each ABSENT axis to its legacy Ushiki value.

    Absent-axes resolution is not a convenience: every classic-era row in the tree predates
    the c/p/z_{-1} axes entirely and carries no parameter fields at all, so "absent" IS the
    classic point and any other reading loses the whole classic population. Mirrors
    `production_seeder.row_phoenix_key` exactly (that one is the dup-identity key; this one
    is the partition key, and they must agree on what "classic" means)."""
    i = _phoenix_schema(row)
    if i is None:
        raise ValueError(
            "phoenix row is in neither on-disk parameter schema (no phoenix_* stamp, no "
            "p_re/zm1_* family_params, and no cx/fw or outcome_cx/outcome_fw marker), so its "
            "parameter point is UNKNOWN, not classic. An eval-slice row is the live instance: "
            "its schema has no parameter axes at all. Use the token-only `partition_of` "
            "there — it answers `phoenix` for both halves, which is honest.")
    return tuple(
        default if row.get(keys[i]) is None else float(row.get(keys[i]))
        for keys, default in zip(_PHOENIX_AXIS_KEYS, PHOENIX_CLASSIC_POINT))


def is_classic_phoenix(row) -> bool:
    """True iff `row`'s phoenix parameter point IS the pinned Ushiki point. EXACT equality —
    the stamped values round-trip through JSON as the same floats, and a tolerance here would
    quietly annex varied points near the classic one into a partition whose whole content is
    one parameter value. Does NOT check the family: callers reach it through
    `partition_of_row`, which does."""
    return phoenix_point(row) == PHOENIX_CLASSIC_POINT


def partition_of_token(token, default=None):
    """Partition for a family token that may ALREADY be a partition key.

    Two on-disk spellings name the same thing and one resolver has to read both. A corpus
    render block carries the Rust `fractal_type` (`julia_multibrot4`); a DISCOVERY LEDGER's
    `family` field carries the PARTITION (`julia:multibrot4`) — the ledgers have been written
    that way since the julia planes were namespaced, which is why `production_seeder.t_good_for`
    takes `row["family"]` straight. Passing a ledger `family` through `FT2FAM` alone answers
    `default` for every julia row, so the emission cell axis silently loses them.

    A token already in `ALL_FAMS` passes through (including a DERIVED one — a writer that
    stamped `phoenix:classic` is telling us the answer). Anything else is `default`, exactly as
    `partition_of`. TOKEN-ONLY: the phoenix split needs the whole row (`partition_of_row`)."""
    if token in FT2FAM:
        return FT2FAM[token]
    if token in ALL_FAMS:
        return token
    return default


def partition_of(fractal_type: str, default=None):
    """Partition key for a row's `fractal_type`.

    `default` is what an UNKNOWN token maps to, and the two live conventions differ on
    purpose, so it is explicit at every call site rather than baked in here:
    `partition_of(ft)` -> None drops the row (keeper_cut: an unrecognised family is not
    cut), `partition_of(ft, ft)` passes the token through (the derivations: an unrecognised
    family becomes its own partition and shows up in the table rather than vanishing).

    TOKEN-ONLY, so a phoenix row resolves to the BASE `phoenix` partition here whatever its
    parameter point. Any consumer that has the whole row must call `partition_of_row`."""
    return FT2FAM.get(fractal_type, default)


def partition_of_row(row, default=None):
    """Partition key for a whole row — the resolver every label/ledger/corpus consumer wants.

    `row` is a discovery-ledger row, a corpus `images.jsonl` render block, or anything else
    carrying a family token (`fractal_type` or `family`) plus, for phoenix, its parameter
    axes in either on-disk schema. `default` behaves exactly as in `partition_of`.

    The token is read through `partition_of_token`, so a ledger row whose `family` is already
    a partition key resolves to itself instead of falling to `default`.

    The only thing this adds over `partition_of` is the phoenix split, and it adds it at the
    READER: nothing on disk is re-keyed, and a writer that never heard of `phoenix:classic`
    still produces rows that resolve correctly."""
    ft = row.get("fractal_type") or row.get("family")
    part = partition_of_token(ft, default)
    if part == "phoenix" and is_classic_phoenix(row):
        return _registered(CLASSIC_PHOENIX)
    return part
