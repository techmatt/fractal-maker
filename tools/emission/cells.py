"""cells.py — joint-count cells, target measure, and deficit (pure; no torch/GPU).

A wallpaper's full descriptor is a point in the product space

    cell = (partition, morph_cluster, palette_flavor, render_style)

The first two are FIXED by a location's intake; the last two are chosen at colorize
time. This module maintains, for the *gated pool*, the joint count over these cells,
the target measure, and the resulting per-cell deficit that drives the
conditional-deficit colorizer.

Joint counts (not per-axis marginals) are the whole point: a marginal view ("plenty
of warm palettes, plenty of spirals") cannot see the hole "warm spirals plentiful,
cold spirals absent" that the joint count exposes directly.

THE TARGET IS DERIVED, NOT HAND-PLACED. `TargetMeasure` is built by
`from_partition_shares` from the canonical release-mix ratio table
(`tools/scoring/release_mix.py`) re-solved against THIS intake's feasible cells:

    weight(cell) = share[partition(cell)] / n_feasible_cells[partition(cell)]

so each partition's cells carry exactly its intended share of the measure, whatever its
morph-cluster count happens to be this intake. That division is the whole content of the
old `target_share` solver, done once for every partition instead of by hand for one of
them. It is also what makes the measure DENOMINATOR-INVARIANT: a partition that gains
clusters does not gain release share, it spreads the same share over more cells.

It used to be a hand-edited `data/emission/target_measure.json` carrying nine literal
multipliers plus a `target_share` override for classic phoenix, and a fourth consumer
(`library_intake_2`) carried its own second copy of the classic share. Those numbers said
the same KIND of thing as the ratio table and disagreed with it (mandelbrot 9.0% vs 22.7%),
which is what two policies about the same partitions always become. See
`docs/design/retired.md`.

A cell that repeatedly fails to fill (attempt cap reached with zero fills) leaves the
support and is logged.

Everything here is pure Python + stdlib — the shares come in as a plain dict, so the
deficit logic is unit-testable without loading a model or rendering a frame, and this
module keeps no opinion about which partitions exist.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

# A cell is a 4-tuple of strings. Axis order is fixed and load-bearing (the report and the
# per-axis marginals both address axes by this name order). The first axis is the PARTITION
# (`partitions.partition_of_row`) — it is still spelled `fractal_type` in reports and pool
# rows because that is what every persisted record calls it.
AXES = ("fractal_type", "morph_cluster", "palette_flavor", "render_style")
Cell = tuple  # (partition, cluster, flavor, style)


# --------------------------------------------------------------------------- #
# Target measure — hand-editable config.
# --------------------------------------------------------------------------- #
DEFAULT_ATTEMPT_CAP = 6
DEFAULT_SOFTMAX_TEMP = 0.35


class UnknownPartitionCell(KeyError):
    """A feasible cell names a partition the derived share table has no entry for.

    Refused rather than defaulted to zero: a zero-weight partition is never chosen by the
    colorizer, its cells report a target of nothing, and the readout says "that partition had
    no demand" instead of "that partition has no policy" — the same failure
    `release_mix.check_complete` exists to prevent, one layer down."""


@dataclass
class TargetMeasure:
    """Target measure over feasible cells, DERIVED from per-partition release shares.

    `weights_by_partition[p]` is the weight of ONE cell of partition `p`, so a partition's
    total measure over the feasible support is exactly its intended share. Built by
    `from_partition_shares`; there is no config file and no hand-placed override — the policy
    lives in `tools/scoring/release_mix.RATIO` and nowhere else.

    `attempt_cap` / `softmax_temp` are mechanism, not policy (per-cell colorize attempts before
    eviction; the colorizer's range-normalized choice temperature), so they are plain defaults
    here rather than the last two fields of a deleted config file."""
    weights_by_partition: dict = field(default_factory=dict)
    attempt_cap: int = DEFAULT_ATTEMPT_CAP
    softmax_temp: float = DEFAULT_SOFTMAX_TEMP
    shares: dict = field(default_factory=dict)          # partition -> intended share (report)
    n_cells_by_partition: dict = field(default_factory=dict)   # partition -> feasible cells

    @staticmethod
    def from_partition_shares(shares: dict, feasible_cells: Iterable[Cell],
                              attempt_cap: int = DEFAULT_ATTEMPT_CAP,
                              softmax_temp: float = DEFAULT_SOFTMAX_TEMP) -> "TargetMeasure":
        """Re-solve `shares` (partition -> intended fraction of the release) against the LIVE
        feasible cells: `weight_p = share_p / n_cells_p`.

        The division is load-bearing and is the one thing a substituted multiplier gets wrong.
        A partition's measure over its own cells is `n_cells_p × weight_p = share_p`, i.e. its
        realized target share is its intended share whatever its morph-cluster count — where a
        raw multiplier would scale each partition's share BY that count, so the family with the
        most clusters swamps the order book regardless of the policy.

        A partition present in `feasible_cells` with no share raises (`UnknownPartitionCell`).
        A share for a partition with no feasible cell is dropped and reported in
        `unrealized_shares` — it cannot be served this intake — and the remaining shares are
        renormalized so the measure still sums to 1."""
        feasible = list(feasible_cells)
        n_cells = Counter(c[0] for c in feasible)
        missing = sorted(p for p in n_cells if p not in shares)
        if missing:
            raise UnknownPartitionCell(
                f"feasible cells name partition(s) {missing} with no release-mix share. "
                f"Register them in release_mix.RATIO (and partitions.ALL_FAMS) before an "
                f"intake can allocate against them.")
        live = {p: float(shares[p]) for p in n_cells}
        tot = sum(live.values())
        if tot <= 0.0:
            raise ValueError(f"release shares over the live partitions {sorted(live)} sum to "
                             f"{tot}; no measure can be derived from them")
        live = {p: v / tot for p, v in live.items()}
        return TargetMeasure(
            weights_by_partition={p: live[p] / n_cells[p] for p in n_cells},
            attempt_cap=int(attempt_cap), softmax_temp=float(softmax_temp),
            shares=live, n_cells_by_partition=dict(n_cells))

    def unrealized_shares(self, shares: dict) -> dict:
        """`{partition: share}` for partitions the caller asked for that this intake cannot
        serve (no feasible cell). Reported, never silently absorbed: a partition with demand
        and no supply is a supply fact, and a renormalized measure alone cannot say it."""
        return {p: float(v) for p, v in shares.items() if p not in self.weights_by_partition}

    def weight(self, cell: Cell) -> float:
        try:
            return self.weights_by_partition[cell[0]]
        except KeyError:
            raise UnknownPartitionCell(
                f"cell {cell!r} names partition {cell[0]!r}, which is not in this measure's "
                f"derived share table {sorted(self.weights_by_partition)}. The measure is "
                f"solved against a specific feasible set — re-derive it, do not extend it."
            ) from None

    def partition_shares(self) -> dict:
        """The realized per-partition target share of this measure: `n_cells_p × weight_p`.
        THE number both consumers are asserted to agree on."""
        return {p: self.n_cells_by_partition[p] * w
                for p, w in self.weights_by_partition.items()}


# --------------------------------------------------------------------------- #
# Feasible-cell enumeration.
# --------------------------------------------------------------------------- #
def build_feasible_cells(observed_type_cluster: Iterable[tuple],
                         flavors: Iterable[str],
                         styles: Iterable[str]) -> list:
    """Feasible cells = (observed (partition, cluster) pairs) × all flavors × all styles.

    A morph cluster that no location realizes cannot be produced, so only OBSERVED
    (partition, cluster) pairs anchor the grid; palette flavor and render style are free
    choices at colorize time, so every one of them is feasible for every observed
    (partition, cluster)."""
    flavors = list(flavors)
    styles = list(styles)
    cells = []
    for (t, cl) in observed_type_cluster:
        for f in flavors:
            for s in styles:
                cells.append((t, cl, f, s))
    return cells


# --------------------------------------------------------------------------- #
# Deficit model — maintained over the GATED pool.
# --------------------------------------------------------------------------- #
class DeficitModel:
    """Joint-count deficit over feasible cells for the gated pool.

    deficit(cell) = target_frac(cell) − pool_frac(cell)

    where target_frac is the target measure normalized to sum 1 over the live support,
    and pool_frac is the gated-pool joint count normalized to sum 1 (0 when empty).
    Both fill counts and attempt counts are rebuilt from the durable pool log on
    resume; nothing here is trusted from a checkpoint."""

    def __init__(self, feasible_cells: list, target: TargetMeasure):
        self.target = target
        self.support: set = set(feasible_cells)
        self.weights: dict = {c: target.weight(c) for c in feasible_cells}
        self.fill_counts: Counter = Counter()
        self.attempt_counts: Counter = Counter()
        self.capped: set = set()

    # ---- rebuild-from-log entry points (resume safety) -------------------- #
    def record_fill(self, cell: Cell):
        """A gated (floor-passing) wallpaper landed in `cell`."""
        self.fill_counts[cell] += 1

    def record_attempt(self, cell: Cell) -> bool:
        """A colorize attempt targeted `cell` (whether or not it passed the floor).
        Returns True iff this attempt tips the cell over the attempt cap with zero
        fills, evicting it from the support."""
        self.attempt_counts[cell] += 1
        if (cell in self.support and self.fill_counts[cell] == 0
                and self.attempt_counts[cell] >= self.target.attempt_cap):
            self.support.discard(cell)
            self.capped.add(cell)
            return True
        return False

    # ---- deficit -------------------------------------------------------- #
    def _target_frac(self) -> dict:
        tot = sum(self.weights[c] for c in self.support)
        if tot <= 0:
            return {c: 0.0 for c in self.support}
        return {c: self.weights[c] / tot for c in self.support}

    def _pool_total(self) -> int:
        return sum(self.fill_counts.values())

    def deficit(self, cell: Cell, target_frac: dict | None = None,
                pool_total: int | None = None) -> float:
        if cell not in self.support:
            return float("-inf")
        tf = target_frac if target_frac is not None else self._target_frac()
        pt = pool_total if pool_total is not None else self._pool_total()
        pool_frac = (self.fill_counts[cell] / pt) if pt > 0 else 0.0
        return tf.get(cell, 0.0) - pool_frac

    def feasible_options(self, ftype: str, cluster: str,
                         flavors: Iterable[str], styles: Iterable[str]) -> list:
        """The (flavor, style) options still in support for a fixed (type, cluster)."""
        opts = []
        for f in flavors:
            for s in styles:
                if (ftype, cluster, f, s) in self.support:
                    opts.append((f, s))
        return opts

    # ---- diagnostics ---------------------------------------------------- #
    def occupancy(self) -> dict:
        """Report snapshot: how many feasible cells are populated / capped / empty."""
        feasible = len(self.support) + len(self.capped)
        populated = sum(1 for c, n in self.fill_counts.items() if n > 0)
        return {
            "feasible_cells": feasible,
            "in_support": len(self.support),
            "capped": len(self.capped),
            "populated_cells": populated,
            "empty_in_support": len(self.support) - sum(
                1 for c in self.support if self.fill_counts[c] > 0),
            "pool_total": self._pool_total(),
        }


# --------------------------------------------------------------------------- #
# Conditional-deficit colorizer choice (softmax over per-option deficit).
# --------------------------------------------------------------------------- #
def range_normalized_softmax(deficits: list, temp: float) -> list:
    """Softmax over deficits after normalizing them to [0,1] by their own range, so the
    distribution is scale-free: the best option maps to 1, the worst to 0, ties to a
    flat (uniform) distribution. `temp` then controls how sharply the best option is
    preferred (small → argmax-like, large → uniform)."""
    if not deficits:
        return []
    lo, hi = min(deficits), max(deficits)
    span = hi - lo
    if span <= 1e-12:                       # all equal → uniform
        return [1.0 / len(deficits)] * len(deficits)
    norm = [(d - lo) / span for d in deficits]
    t = max(temp, 1e-6)
    exps = [math.exp(v / t) for v in norm]
    z = sum(exps)
    return [e / z for e in exps]


def choose_option(model: DeficitModel, ftype: str, cluster: str,
                  flavors: Iterable[str], styles: Iterable[str],
                  rng) -> tuple | None:
    """Pick a (flavor, style) for a location whose (type, cluster) is fixed, by
    softmax over the per-option joint deficit (softmax tie-break, not strict argmax).
    Returns (flavor, style, deficit, n_options, probs) or None if nothing feasible."""
    opts = model.feasible_options(ftype, cluster, flavors, styles)
    if not opts:
        return None
    tf = model._target_frac()
    pt = model._pool_total()
    defs = [model.deficit((ftype, cluster, f, s), tf, pt) for (f, s) in opts]
    probs = range_normalized_softmax(defs, model.target.softmax_temp)
    idx = int(rng.choice(len(opts), p=probs))
    f, s = opts[idx]
    return f, s, defs[idx], len(opts), probs
