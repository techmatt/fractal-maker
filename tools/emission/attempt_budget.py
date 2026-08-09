"""attempt_budget.py — WHO gets colorized, and how many times, budgeted from release need.

WHAT CHANGED AND WHY (2026-08-09, prompts/selection_restructure_2.md). Colorize volume used to
be a side effect of the joint deficit model: `cells.choose_option` picked a (palette_flavor,
render_style) pair per location by largest joint deficit, and the render-style axis carries ONE
smooth style against N registry-promoted strange ones. So the smooth head drew roughly 1/(N+1)
of the attempts no matter how many smooth slots the release asked for. The selrestruct_1 smoke
made the cost concrete: **3 smooth rows out of 30 attempts against 6 smooth slots**, short-filling
3 of them off a supply that had 300+ floor-passing smooth-capable locations. A release starved by
an allocation rule that had no opinion about the release.

The rule here is the opposite direction of causation — **the release budgets the colorize**:

  1. HEAD FIRST. `attempts_head = ATTEMPT_MULTIPLIER × that head's release slots`. The two
     heads are sized against their own release need and never against each other. When the
     total attempt budget cannot cover both at 4x, BOTH scale down proportionally
     (`scale_to_budget`) — never one head starved to keep the other whole, which is the exact
     failure this replaces.
  2. PARTITION SECOND. Within a head, attempts split over the partitions by `release_mix`
     re-solved over the partitions that actually have floor-passing supply — the same table and
     the same apportionment rule the release SLOTS use (`ranked_intake.partition_slots` ->
     `apportion.sequence_by_deficit`), so the colorize is spent in the mix the release will be
     selected in.
  3. RANK THIRD. A partition's attempts are filled in `ranked_intake` rank order, best-first.
     A partition with fewer floor-passing locations than attempts short-fills and SAYS SO
     (`supply_short`), which is the whole point of recording planned beside realized: a thin
     release is then attributable to supply or to budget at a glance, without re-deriving
     either from the pool.

WHAT THIS MODULE DOES NOT DECIDE. The STYLE within a head is unchanged — the driver still asks
`cells.choose_option` for a (flavor, style) over that head's styles, so the deficit spread over
the promoted strange modes is untouched; this budget only decides how many attempts that spread
is run inside. It does not select a release (`selection.rank_select`), it does not admit a
location (`ranked_intake`), and it holds no floor.

THE ORDER IS ITSELF A DECISION. The plan is emitted through `apportion.sequence_by_deficit` over
the (head, partition) cells, so EVERY PREFIX of it is near-proportional: a run killed by the
time backstop half way through has spent its half in the planned mix rather than in whatever
order the cells were built. That is the truncating-consumer property, and the colorize loop is
a truncating consumer by construction (a wall-clock backstop it does not control).

    from tools.emission import attempt_budget as AB
    plan, budget = AB.plan(release_n=12, strange_frac=0.5, total_budget=30,
                           ranked_ids={"mandelbrot": [...], ...})
    budget["head_attempts"]        # {"wallpaper": 15, "mining": 15}
    plan[0]                        # Attempt(head="mining", partition="mandelbrot", ...)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import release_mix as RM                          # noqa: E402  THE release-mix ratio table
from apportion import sequence_by_deficit         # noqa: E402  THE apportionment owner
from tools.emission import floors as F            # noqa: E402  THE coarse-constant owner
from tools.emission import ranked_intake as RI    # noqa: E402  the share->count apportionment

# The two heads, in the spelling every pool row's `head` column and the driver's
# `head_for_style` use. Declared here because this module allocates BETWEEN them and a
# mis-spelled head is a budget that silently goes nowhere.
WALLPAPER, MINING = "wallpaper", "mining"
HEADS = (WALLPAPER, MINING)


@dataclass(frozen=True)
class Attempt:
    """One planned colorize: a location, and the head whose slots paid for it.

    `head` picks the STYLE SET the driver offers `cells.choose_option` (smooth for the
    wallpaper head, the promoted strange modes for the mining head) — it does not pick the
    style. `rank` is the location's 0-based position in its partition's ranked list, carried so
    the realized log can say how deep the budget reached without re-deriving the ranking."""
    head: str
    partition: str
    location_id: str
    rank: int


def head_slots(release_n: int, strange_frac: float) -> dict:
    """`{head: release slots}` — the release split the attempt budget is sized against.

    THE SAME arithmetic `select_release` uses (`strange_slots = round(N·frac)`, smooth takes
    the rest), read from here so the budget and the selection cannot disagree about how many
    slots a head is filling. That disagreement is not hypothetical: a budget sized against a
    different N than the selection spends is exactly a short-fill nobody can attribute."""
    n = max(0, int(release_n))
    strange = int(round(n * float(strange_frac)))
    strange = max(0, min(n, strange))
    return {WALLPAPER: n - strange, MINING: strange}


def scale_to_budget(want: dict, budget: int) -> dict:
    """`want` truncated to `budget` PROPORTIONALLY, via `apportion.sequence_by_deficit`.

    Returns `want` unchanged when the budget covers it. Otherwise every cell keeps its share of
    what there is — the deficit rule is exact for two cells (apportion.py: worst deviation 0.5,
    exhaustive to 60x60), which is the case that matters here.

    NOT `deal_round_robin`, and the difference is the whole rule: round-robin equalizes COUNTS
    (up to each cell's supply), so a 20-smooth / 40-strange want under a budget of 24 comes back
    12/12 instead of 8/16. That is levelling the head with the larger need down to the other —
    the mirror image of the failure this module exists to fix, and it would be invisible in any
    run where both heads happen to ask for the same number (`strange_frac 0.5`, the default)."""
    budget = max(0, int(budget))
    total = sum(int(v) for v in want.values())
    if budget >= total:
        return {k: int(v) for k, v in want.items()}
    out = {k: 0 for k in want}
    for k in sequence_by_deficit({k: int(v) for k, v in sorted(want.items())})[:budget]:
        out[k] += 1
    return out


def head_attempts(slots: dict, total_budget: int,
                  multiplier: int = F.ATTEMPT_MULTIPLIER) -> tuple:
    """`(attempts, diag)` — the per-head attempt budget, sized against release need.

    `attempts_head = multiplier × slots_head`, scaled down proportionally (both heads, never
    one) when `total_budget` cannot cover the pair. `diag` carries the want and whether the
    scale-down fired, because "smooth got 15" and "smooth got 15 because the run was capped at
    30" are different facts and only the second one explains a short-fill."""
    want = {h: max(0, int(multiplier) * int(slots.get(h, 0))) for h in HEADS}
    attempts = scale_to_budget(want, total_budget)
    scaled = attempts != want
    return attempts, {
        "attempt_multiplier": int(multiplier),
        "total_budget": max(0, int(total_budget)),
        "head_slots": {h: int(slots.get(h, 0)) for h in HEADS},
        "head_want": want,
        "scaled_to_budget": scaled,
    }


def partition_attempts(parts, n: int) -> dict:
    """`{partition: attempts}` — `n` attempts over `parts` by the release-mix shares.

    `parts` is the set of partitions with floor-passing SUPPLY, not the registry: a partition
    with nothing to offer must not hold attempts hostage, exactly as `_slot_plan` re-solves the
    slot shares over the partitions a head pass has candidates for. Same table (`release_mix`),
    same apportionment (`ranked_intake.partition_slots`), so the colorize is spent in the mix
    the release is selected in."""
    parts = sorted(set(parts))
    if not parts or n <= 0:
        return {p: 0 for p in parts}
    return RI.partition_slots(RM.shares(parts), n)


def plan(release_n: int, strange_frac: float, total_budget: int, ranked_ids: dict,
         multiplier: int = F.ATTEMPT_MULTIPLIER) -> tuple:
    """`(attempts, budget)` — THE plan: an ordered list of `Attempt`, plus its record.

    `ranked_ids` is `{partition: [location_id, ...]}` best-first — `ranked_intake`'s output,
    already floor-passing. A partition with an empty list is not in the supply set and gets no
    attempts.

    The returned order is `sequence_by_deficit` over the (head, partition) cells, each cell's
    ids popped in rank order, so every prefix is near-proportional across both heads and all
    partitions (see the module docstring). `budget` is the durable record: planned per head and
    per (head, partition), the supply shortfall the plan already knows about, and the diag from
    `head_attempts`. Realized fills are NOT here — they are derived from the pool by
    `realized_fills` after the run, because a plan that reports its own execution is a
    hardcoded `True` waiting to happen."""
    supply = {p: list(v) for p, v in ranked_ids.items() if v}
    slots = head_slots(release_n, strange_frac)
    att, diag = head_attempts(slots, total_budget, multiplier)

    planned: dict = {}
    short: dict = {}
    cells: dict = {}
    for h in HEADS:
        per_part = partition_attempts(supply.keys(), att[h])
        planned[h] = {p: int(k) for p, k in sorted(per_part.items())}
        take = {p: min(int(k), len(supply.get(p, ()))) for p, k in per_part.items()}
        short[h] = {p: int(per_part[p]) - int(t) for p, t in take.items() if per_part[p] > t}
        for p, t in sorted(take.items()):
            if t > 0:
                cells[(h, p)] = t

    order = sequence_by_deficit(dict(sorted(cells.items())))
    cursor = {c: 0 for c in cells}
    out: list = []
    for cell in order:
        h, p = cell
        k = cursor[cell]
        cursor[cell] = k + 1
        out.append(Attempt(head=h, partition=p, location_id=supply[p][k], rank=k))

    budget = dict(diag)
    budget.update({
        "head_attempts": {h: int(att[h]) for h in HEADS},
        "planned_by_partition": planned,
        "planned_total": sum(sum(v.values()) for v in planned.values()),
        "supply_short_by_partition": {h: v for h, v in short.items() if v},
        "supply_short_total": sum(sum(v.values()) for v in short.values()),
        "scheduled_total": len(out),
        "supply_partitions": {p: len(v) for p, v in sorted(supply.items())},
    })
    return out, budget


def realized_fills(rows, head_of_style, partition_of_row=None) -> dict:
    """`{head: {partition: attempts}}` DERIVED from the pool rows, never tracked alongside.

    `rows` are pool rows (`render_style` + `type`); `head_of_style` is the driver's style->head
    router. Derived rather than counted into the plan as it executes for the reason
    `CLAUDE.md` states as a rule: a generator must read the state it reports from the state
    itself, or the record outlives what it records — a resumed run, an errored render and a
    location the deficit model had no feasible cell for all diverge from the plan, and only the
    pool knows which."""
    out: dict = {h: {} for h in HEADS}
    for r in rows:
        h = head_of_style(r.get("render_style"))
        p = (partition_of_row(r) if partition_of_row else r.get("type"))
        if h not in out:
            out[h] = {}
        out[h][p] = out[h].get(p, 0) + 1
    return {h: dict(sorted(v.items())) for h, v in out.items()}


def fill_lines(budget: dict, realized: dict) -> list:
    """One line per head — `planned N (want M) → realized K` with the per-partition detail —
    for the run banner and the report. The three numbers together are what makes a short-fill
    attributable: `want > planned` is the budget cap, `planned > realized` is supply (or an
    error / a capped cell), and `realized < 4x slots` with neither is a bug."""
    lines = []
    for h in HEADS:
        want = (budget.get("head_want") or {}).get(h, 0)
        got = (budget.get("head_attempts") or {}).get(h, 0)
        planned = (budget.get("planned_by_partition") or {}).get(h, {})
        real = (realized or {}).get(h, {})
        n_real = sum(real.values())
        line = (f"{h}: {budget.get('attempt_multiplier')}x{(budget.get('head_slots') or {}).get(h, 0)}"
                f" slots = {want} wanted, {got} budgeted, {sum(planned.values())} scheduled, "
                f"{n_real} realized")
        shortp = (budget.get("supply_short_by_partition") or {}).get(h) or {}
        if shortp:
            line += f" · supply-short {shortp}"
        lines.append(line)
    return lines
