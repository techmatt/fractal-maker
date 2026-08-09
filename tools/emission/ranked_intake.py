"""ranked_intake.py — READ-TIME ranked intake: the union ledgers, ranked per partition.

WHAT CHANGED AND WHY (2026-08-09, prompts/selection_restructure_1.md). Stage-2 intake used to
be a FROZEN VERDICT filter: a ledger row was admitted iff its stored `decoded_class` was `>= 3`
AND that class had been stamped by the checkpoint that is live right now
(`corpus_common.is_current_decoded`). Both halves are enforcing frozen state:

  * `decoded_class` is `corn_decode(p_notbad, p_good, t_good_p)` — a per-partition DERIVED
    threshold, baked into the row at harvest time. Moving `t_good` for a partition does not
    move a single stored class, so a row's admissibility is a fact about the day it was minted.
  * the stamp then throws away every row whose class came from an older head, which is correct
    for a frozen verdict and catastrophic as an intake rule: the v10 flip took this intake from
    ~1.4k locations to **16**, and the repair was a re-score pass costing hours of GPU.

This module replaces both with a READ-TIME choice. The row's raw `p_good` — the head's stored
P(>=3), a number, not a verdict — is read as-is, the population is ranked by it best-first, and
the only cut is the one coarse junk floor (`floors.JUNK_FLOOR`). Nothing is frozen: a different
floor, a different order, a different budget are all a re-read away, and a head flip degrades
the RANK QUALITY of the old rows instead of deleting them.

WHAT IS STILL ENFORCED, AND WHY IT IS NOT THE SAME KIND OF THING.
`guard_pass` and `distinct` (`descriptor.guard_and_distinct`) still admit. Neither is a head
verdict: the guard is the degenerate-outcome prior on the FIELD (flat / all-interior), and
`distinct` is the run's own morphology dedup against its own cloud. They are properties of the
location and of the population, they do not go stale when a checkpoint moves, and dropping them
would put blank tiles and near-duplicates into the ranking.

THE FLOOR-ADMIT BYPASS IS UNCHANGED. A `FLOOR_ADMIT_SOURCES` row (`q4_harvest`,
`human_q3plus`) was selected by a signal ORTHOGONAL to the head — the q4 goodness field, or
Matt's own 3/4 — so the head's own verdict must not veto it. That is why
`descriptor.FLOOR_PNOTBAD` was DELETED rather than re-derived (retired.md, 2026-08-04), and the
junk floor would be the same veto at a lower number if it applied to those rows. They are
admitted through the floor, ranked by their `p_good` like everything else, and COUNTED
separately (`n_bypass`) so a partition's supply never silently becomes "however many humans
labelled".

WHAT THIS MODULE IS NOT. It does not decide colorize VOLUME (unchanged, still the deficit
model + `--target-gated`) and it does not select a release (`selection.rank_select`). It
answers exactly one question — *in what order should this partition's locations be offered* —
plus the supply arithmetic that question implies (`emit_cap`, `partition_slots`).

    from tools.emission import ranked_intake as RI
    ranked, diag = RI.ranked_by_partition([Path("data/.../outcome_ledger.jsonl")])
    ranked["mandelbrot"][0]        # the best-scoring mandelbrot candidate, floor-passing
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apportion import sequence_by_deficit         # noqa: E402  THE apportionment owner
from tools.emission import descriptor as D        # noqa: E402
from tools.emission import floors as F            # noqa: E402  THE cut owner (junk floor, r)

# The row field holding the stage-1 head's raw P(>=3). ONE spelling, named here rather than
# typed at four call sites: `p_good` is the CORN marginal the ledger persists, and
# `decoded_class` is what this module exists NOT to read.
RAW_P_GE3_KEY = "p_good"


def raw_p_ge3(row: dict):
    """The row's STORED raw P(>=3) — the ranking key. `None` when the row carries none (an
    unscored or guard-zeroed row), which sorts last and never passes the junk floor."""
    v = row.get(RAW_P_GE3_KEY)
    return None if v is None else float(v)


def is_floor_admit(row: dict) -> bool:
    """This row's selection signal is orthogonal to the head (see the module docstring)."""
    return D.source_tag_of(row) in D.FLOOR_ADMIT_SOURCES


def passes(row: dict) -> bool:
    """Junk floor, with the floor-admit bypass. THE admission comparison of this path."""
    return is_floor_admit(row) or F.passes_junk_floor(raw_p_ge3(row))


def admit(row: dict) -> bool:
    """The whole read-time predicate handed to `descriptor.load_union_admitted`:
    guard ∧ distinct ∧ (junk floor ∨ floor-admit). No decode-version predicate, no q3 gate."""
    return D.guard_and_distinct(row) and passes(row)


def rank_key(row: dict):
    """Sort key for BEST-FIRST order: raw P(>=3) descending, then the namespaced id.

    The id tie-break is not cosmetic — a run must be reproducible, and float scores tie often
    enough (a guard-zeroed 0.0, a floor-admit row with no score) that leaving the order to
    whatever the union walk happened to produce would make two runs over the same ledgers pick
    different locations."""
    p = raw_p_ge3(row)
    return (-(p if p is not None else -1.0), str(row.get("id")))


def rank_rows(rows) -> list:
    """`rows` best-first. Pure; takes and returns the row dicts themselves."""
    return sorted(rows, key=rank_key)


# --------------------------------------------------------------------------- #
# The intake read.
# --------------------------------------------------------------------------- #
def load_mined(ledger_paths) -> tuple:
    """`(mined_rows, diag)` — the union at guard ∧ distinct, PRE-floor, in ledger order.

    Separate from the ranking so a caller can hold the pre-floor population: the mined count
    is the denominator of every "N mined, M above floor" line, and a caller that only kept the
    passing rows cannot recover it. THE union reader underneath
    (`descriptor.load_union_admitted`), with this path's predicate."""
    return D.load_union_admitted(ledger_paths, admit=guard_and_distinct)


guard_and_distinct = D.guard_and_distinct     # re-exported so a caller needs one import


def ranked_from_mined(mined_rows, mdiag: dict | None = None) -> tuple:
    """`(ranked, diag)` from an already-loaded PRE-FLOOR union (`load_mined`).

    `ranked` is `{partition: [row, ...]}`, each list best-first by raw P(>=3). Partition is
    `descriptor.cell_partition` — the CELL identity, so `phoenix:classic` is its own list and
    is not absorbed by `phoenix`.

    `diag` carries the union diagnostics verbatim plus the per-partition supply census this
    path adds: `mined` (guard ∧ distinct, PRE-floor), `passing` (what is in the ranked list),
    `bypass` (floor-admit rows admitted through the floor). `mined` is what makes the sheet's
    one-liner honest — "24 mined, 0 above floor" is a statement about a population, and
    without the pre-floor count the zero has no denominator."""
    ranked: dict = {}
    mined: dict = {}
    bypass: dict = {}
    for row in mined_rows:
        part = D.cell_partition(row)
        mined[part] = mined.get(part, 0) + 1
        if not passes(row):
            continue
        ranked.setdefault(part, []).append(row)
        if is_floor_admit(row):
            bypass[part] = bypass.get(part, 0) + 1
    for part in ranked:
        ranked[part] = rank_rows(ranked[part])
    diag = dict(mdiag or {})
    diag.update({
        "junk_floor": F.JUNK_FLOOR,
        "mined_by_partition": dict(sorted(mined.items())),
        "passing_by_partition": {p: len(v) for p, v in sorted(ranked.items())},
        "bypass_by_partition": dict(sorted(bypass.items())),
        "n_mined": len(mined_rows),
        "n_passing": sum(len(v) for v in ranked.values()),
    })
    return ranked, diag


def ranked_by_partition(ledger_paths) -> tuple:
    """`load_mined` then `ranked_from_mined` — the one-call form."""
    mined_rows, mdiag = load_mined(ledger_paths)
    return ranked_from_mined(mined_rows, mdiag)


def supply_census(mined_rows, scope: set | None = None) -> tuple:
    """`(mined_by_partition, passing_by_partition)` over `mined_rows`, restricted to `scope`.

    ONE walk producing BOTH halves, because the two must describe the SAME population and the
    obvious way to get them wrong is to take each from wherever it was already lying around:
    the sheet printed "81 mined, 18 above floor" for julia:mandelbrot when the mined count came
    from the whole 1470-row union and the passing count from the 720 rows an intake SNAPSHOT
    restricted the run to. `scope` is that restriction (None = the whole union)."""
    mined: dict = {}
    passing: dict = {}
    for row in mined_rows:
        if scope is not None and row["id"] not in scope:
            continue
        part = D.cell_partition(row)
        mined[part] = mined.get(part, 0) + 1
        if passes(row):
            passing[part] = passing.get(part, 0) + 1
    return dict(sorted(mined.items())), dict(sorted(passing.items()))


def supply_lines(diag: dict) -> list:
    """One line per partition — `"<partition>: <mined> mined, <passing> above floor"` — for a
    run banner and for the sheet. Every partition the union SAW gets a line, including the ones
    that emit nothing: a partition that vanishes from the readout because its supply died is
    the failure this line exists to make visible."""
    mined = diag.get("mined_by_partition", {})
    passing = diag.get("passing_by_partition", {})
    out = []
    for part in sorted(set(mined) | set(passing)):
        n_pass = passing.get(part, 0)
        line = f"{part}: {mined.get(part, 0)} mined, {n_pass} above floor"
        cap = emit_cap(n_pass)
        if not cap:
            line += " → emits 0 (thin supply)"
        out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Supply arithmetic (§3).
# --------------------------------------------------------------------------- #
def emit_cap(passing_supply: int) -> int:
    """`floor(passing_supply / THIN_SUPPLY_DIVISOR)` — the most a partition may emit.

    The rule is "show me one only if there were four to choose from". Its whole point is the
    ZERO: a partition with three floor-passing candidates emits nothing rather than shipping
    its own least-bad row into a release, and the sheet says so in one line
    (`supply_lines`). The divisor is `floors.THIN_SUPPLY_DIVISOR`, declared beside the floor."""
    return int(passing_supply) // F.THIN_SUPPLY_DIVISOR


# Granularity of the share→slot apportionment below. Shares are fractions and
# `sequence_by_deficit` counts whole items, so the shares are quantized to per-mille. 1000 is
# far finer than any release N (a 0.2-ratio partition is 15/1000, i.e. resolvable at N>=67)
# and the sequence is built once per pass.
SLOT_GRANULARITY = 1000


def partition_slots(shares: dict, n: int) -> dict:
    """`{partition: slots}` — `n` release slots apportioned over `shares`, near-proportionally.

    Through `apportion.sequence_by_deficit`, THE owner of the ordering rule, and through it
    rather than a hand-rolled largest-remainder for the stated reason: this is a TRUNCATING
    consumer (n is small, often smaller than the partition count), and the prefix bound is the
    only property that survives truncation. The sequence is built over per-mille share weights
    and the first `n` positions are counted.

    The ±1 prefix bound is a CHECK, not a theorem (apportion.py) — `selection.rank_select`'s
    log carries the realized slots so the allocation is inspectable on the run that used it,
    and no caller here asserts a bound it did not measure."""
    n = max(0, int(n))
    weights = {p: int(round(float(s) * SLOT_GRANULARITY)) for p, s in sorted(shares.items())}
    weights = {p: w for p, w in weights.items() if w > 0}
    if not weights or n == 0:
        return {p: 0 for p in sorted(shares)}
    seq = sequence_by_deficit(weights)[:n]
    out = {p: 0 for p in sorted(shares)}
    for p in seq:
        out[p] += 1
    return out
