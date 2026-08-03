"""THE fractal_type ⟷ ledger-partition map. One copy, imported everywhere.

`fractal_type` is the Rust render-family token that lands in a manifest / eval-slice row
(`mandelbrot`, `julia_multibrot4`, …). A **partition** (a.k.a. "family" in the ledgers) is
what everything downstream is keyed on — `production_seeder.t_good_for`, the keeper cut,
the coverage clouds, the per-partition derivations. They differ in exactly one place: the
julia planes are namespaced (`julia_multibrot4` -> `julia:multibrot4`) so a julia-plane
partition is never confused with its native twin.

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
FAM2FT = {v: k for k, v in FT2FAM.items()}

# Every partition that can reach production, in the canonical report order (native planes
# first, then the julia planes, then phoenix). Derivations walk this to stamp the partitions
# that got NO eval rows at all — a family that is silently absent from a table and a family
# explicitly stamped UNCALIBRATED are different states.
ALL_FAMS = [
    "mandelbrot", "julia:mandelbrot",
    "multibrot3", "multibrot4", "multibrot5",
    "julia:multibrot3", "julia:multibrot4", "julia:multibrot5",
    "phoenix",
]


def partition_of(fractal_type: str, default=None):
    """Partition key for a row's `fractal_type`.

    `default` is what an UNKNOWN token maps to, and the two live conventions differ on
    purpose, so it is explicit at every call site rather than baked in here:
    `partition_of(ft)` -> None drops the row (keeper_cut: an unrecognised family is not
    cut), `partition_of(ft, ft)` passes the token through (the derivations: an unrecognised
    family becomes its own partition and shows up in the table rather than vanishing)."""
    return FT2FAM.get(fractal_type, default)
