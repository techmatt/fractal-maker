#!/usr/bin/env python
"""visited_density.py — CROSS-RUN SATURATION MEMORY, read straight off the committed ledgers.

The breadth leg of the steered frontier has no memory of where earlier runs already went.
Each run starts with an empty dup cloud (the freshness prior is off by default, and it was
turned off for a reason — it sterilized the native-seed rejection sampler), so a region three
runs have already mined ranks exactly like untouched territory. This module is the memory,
and it is deliberately NOT a new persistent store: `data/**/outcome_ledger.jsonl` already
records every place a run confirmed, durably and per-run, so the memory IS the ledgers.
Nothing to maintain, nothing that can go stale, nothing to migrate at a head flip.

WHAT A VISIT IS. Every ledger row with usable coordinates, with NO quality and NO distinctness
filter — deliberately unlike `production_seeder.build_cloud`, which keeps one row per distinct
GOOD place. Two reasons: a place that was checked and rejected was still visited (the descent
spent its budget there), and `is_good` is a cut on a stored probability whose meaning moves
with the active head, so a quality-filtered cross-run memory would silently re-shape itself at
every classifier flip. A visit is a visit.

SCALE-AWARE, AND THE SCALE IS THE VISIT'S. A row at framewidth `fw` shadows a disc of radius
`k * fw` **centred on itself** — the candidate's own fw does not enter (that is the difference
from `near_dup`, whose radius is `k * min(a_fw, b_fw)`). A deep visit therefore shadows almost
nothing and a base-scale visit shadows a neighbourhood, which is the intended reading of
"already saturated": one run passing through a wide frame does not exhaust everything inside
it, but a hundred deep confirmations in one basin do exhaust that basin.

IDENTITY-AWARE, and this is load-bearing rather than tidy. Within a julia or phoenix partition
the coordinate is a Z-PLANE point; two views at the same z with different seed parameters are
different fractals, and collapsing them is the "over-kill" failure `build_cloud`'s
`row_ident` gate already exists to prevent. Measured on the calibration population
(`sat_radius_calibrate.py`): at k=0.05 a z-only index reads julia:mandelbrot as 46.3% shadowed
where the identity-aware index reads 9.7%, and julia:multibrot5 as 8.5% where the true answer
is 0.0% — i.e. z-only would discount a channel that has never been visited twice. The
identity is `production_seeder.row_ident` (imported, never restated), bucketed on a grid of
`JULIA_SAME_C_EPS`; two identities within eps but straddling a bucket edge land in different
buckets and simply do not shadow each other, which errs toward LESS discount.

THE INDEX. One uniform grid per (partition, identity, fw-octave). A visit in octave
`o = floor(log2(fw))` has radius `k*fw <= k*2^(o+1)`, which is the octave's cell size — so
every disc covering a query point has its centre inside the 3x3 cell block around that point,
and the 3x3 scan is EXACT, not approximate. `test_visited_density.py` pins it against the
brute-force definition on random populations.

Pure: stdlib + `production_seeder` (the identity rule) + `partitions` (the base-partition
map). No numpy, no torch, no GPU — it has to stay in the light pytest lane.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import production_seeder as ps          # noqa: E402  THE identity rule + the same-c epsilon
import partitions as P                  # noqa: E402  THE partition registry (base_partition)

LEDGER_NAME = "outcome_ledger.jsonl"

# The identity bucket grid. `near_dup` treats two parameter points as the same fractal when
# they are within this of each other; a hash bucket is the O(1) form of that test, and the
# constant is the SAME one — imported, so a move to the eps moves both together.
IDENT_EPS = ps.JULIA_SAME_C_EPS


def discount(density: float, strength: float) -> float:
    """The soft saturation discount: `1 / (1 + strength * density)`.

    SOFT AND NEVER EXCLUSIONARY, which is the whole design constraint. It falls to 1/(1+n) at
    n shadowing visits and reaches zero only in the limit, so a fully-saturated region is
    strongly disfavoured and still reachable — a partition whose entire frontier is saturated
    keeps picking its best candidate rather than stalling. `strength <= 0` returns 1.0
    exactly (the mechanism off, byte-identical priorities)."""
    if strength <= 0.0 or density <= 0.0:
        return 1.0
    return 1.0 / (1.0 + strength * float(density))


def _quant_ident(ident) -> tuple | None:
    """A parameter identity as an exact-match hash key, on the `IDENT_EPS` grid."""
    if ident is None:
        return None
    return tuple(int(math.floor(float(v) / IDENT_EPS + 0.5)) for v in ident)


class VisitedIndex:
    """Scale-aware count of prior visits shadowing a point, per (partition, identity).

    Built once at run start and never mutated afterwards — this is CROSS-run memory, and the
    current run's own coverage is already what the dup cloud and the morph memory carry. That
    also makes it resume-identical: a killed run rebuilds the same index from the same files.
    """

    def __init__(self, k: float):
        if not (float(k) > 0.0):
            raise ValueError(f"radius multiple must be positive; got {k!r}")
        self.k = float(k)
        self._cells: dict = defaultdict(dict)     # key -> {(octave,i,j): [(x,y,r2)]}
        self._octaves: dict = defaultdict(set)    # key -> {octave}
        self.n_visits = 0
        self.n_unusable = 0                       # rows with no/degenerate coordinates
        self.per_partition: Counter = Counter()
        self.sources: list[str] = []

    # ---------------------------------------------------------------- build
    @staticmethod
    def _key(partition: str, ident):
        return (P.base_partition(partition), _quant_ident(ident))

    def _cell_size(self, octave: int) -> float:
        return self.k * (2.0 ** (octave + 1))

    def add(self, partition: str, ident, cx, cy, fw) -> bool:
        """Register one visit. Returns False (and counts it) for a row that cannot be placed —
        an absent or non-finite coordinate, or a non-positive framewidth. Counted rather than
        raised: the ledgers span years of schema and one unusable legacy row must not take a
        run's start-up down, but a silently dropped population is how a memory quietly
        becomes empty, so the count is reported in `summary()`."""
        try:
            cx, cy, fw = float(cx), float(cy), float(fw)
        except (TypeError, ValueError):
            self.n_unusable += 1
            return False
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(fw) and fw > 0.0):
            self.n_unusable += 1
            return False
        key = self._key(partition, ident)
        octave = math.floor(math.log2(fw))
        cs = self._cell_size(octave)
        cell = (octave, math.floor(cx / cs), math.floor(cy / cs))
        self._cells[key].setdefault(cell, []).append((cx, cy, (self.k * fw) ** 2))
        self._octaves[key].add(octave)
        self.n_visits += 1
        self.per_partition[key[0]] += 1
        return True

    def add_row(self, row: dict) -> bool:
        """Register one LEDGER row: partition from `family`, identity from `row_ident`."""
        return self.add(row.get("family", "mandelbrot"), ps.row_ident(row),
                        row.get("outcome_cx"), row.get("outcome_cy"), row.get("outcome_fw"))

    @classmethod
    def from_rows(cls, rows, k: float) -> "VisitedIndex":
        idx = cls(k)
        for r in rows:
            idx.add_row(r)
        return idx

    # ---------------------------------------------------------------- query
    def density(self, partition: str, ident, cx, cy) -> int:
        """How many prior visits shadow `(cx, cy)` in this partition-and-identity.

        Exact: the octave cell size bounds every radius in that octave, so a covering disc's
        centre cannot lie outside the 3x3 block around the query cell."""
        key = self._key(partition, ident)
        cells = self._cells.get(key)
        if not cells:
            return 0
        cx, cy = float(cx), float(cy)
        n = 0
        for octave in self._octaves[key]:
            cs = self._cell_size(octave)
            i0, j0 = math.floor(cx / cs), math.floor(cy / cs)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for (x, y, r2) in cells.get((octave, i0 + di, j0 + dj), ()):
                        if (cx - x) ** 2 + (cy - y) ** 2 <= r2:
                            n += 1
        return n

    def density_brute(self, partition: str, ident, cx, cy) -> int:
        """The DEFINITION, scanned linearly — the oracle `density` is pinned against. Not a
        second implementation of the index: it is the statement the index accelerates, and
        keeping it here (rather than in the test) is what lets the calibration tool and the
        test share one referent."""
        key = self._key(partition, ident)
        cells = self._cells.get(key)
        if not cells:
            return 0
        cx, cy = float(cx), float(cy)
        return sum(1 for bucket in cells.values() for (x, y, r2) in bucket
                   if (cx - x) ** 2 + (cy - y) ** 2 <= r2)

    def summary(self) -> dict:
        """What went into the memory, for `run_config.json`. A memory whose size nobody can
        read afterwards is a memory nobody can tell was empty."""
        return dict(radius_k=self.k, visits=self.n_visits,
                    unusable_rows=self.n_unusable,
                    partitions=dict(sorted(self.per_partition.items())),
                    identity_buckets=len(self._cells),
                    ledgers=len(self.sources))


# --------------------------------------------------------------------------- #
# The ledger enumeration. ONE owner: `steered_frontier.load_prior_library_rows` (the
# coordinate freshness prior) and the saturation memory read the same files under the same
# exclusion rule, and they used to be one rglob written twice.
# --------------------------------------------------------------------------- #
def ledger_paths(root: Path = ROOT, exclude: Path | None = None) -> list[Path]:
    """Every committed discovery ledger under `data/`, minus one (this run's own).

    `exclude` is resolved before comparison because the caller's path is the run dir's and
    ours came off an rglob — two spellings of one file is how a run ends up seeded with its
    own admissions."""
    own = Path(exclude).resolve() if exclude is not None else None
    return [p for p in sorted((Path(root) / "data").rglob(LEDGER_NAME))
            if own is None or p.resolve() != own]


def iter_prior_ledger_rows(root: Path = ROOT, exclude: Path | None = None):
    """Yield every prior-ledger row, coordinates coerced to float ONCE at ingestion.

    Some ledgers (deep / q4_harvest / phoenix) serialize outcome coords as high-precision
    decimal STRINGS. Every downstream consumer here — the dedup clouds, `dup_penalty`,
    `count_within`, this module's index — is float arithmetic, so the coercion belongs at the
    read rather than at each reader (float64 is ample for O(1) dedup coords, and a prior row
    is never re-rendered)."""
    for led in ledger_paths(root, exclude):
        with open(led, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for key in ("outcome_cx", "outcome_cy", "outcome_fw"):
                    v = r.get(key)
                    if isinstance(v, str):
                        r[key] = float(v)
                yield r


def build_from_ledgers(k: float, root: Path = ROOT,
                       exclude: Path | None = None) -> VisitedIndex:
    """THE production entry point: the cross-run memory, straight off the committed store."""
    paths = ledger_paths(root, exclude)
    idx = VisitedIndex(k)
    idx.sources = [str(Path(p).relative_to(Path(root))) for p in paths]
    for led in paths:
        with open(led, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    idx.add_row(json.loads(line))
    return idx
