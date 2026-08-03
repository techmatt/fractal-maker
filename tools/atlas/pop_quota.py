#!/usr/bin/env python
r"""pop_quota.py — per-partition POP QUOTA in Matt's currency, with a universal floor.

WHY THIS EXISTS, AND WHY IT IS NOT `deficit_scheduler.py` v2
-----------------------------------------------------------
`discovery_pipeline.md` §3.1 measures the v1 failure end to end: `--family-weights` sizes the
per-family ROOT DRAW, and an intended 70% native-multibrot share realized **19.6%** over 149
batches. Two mechanisms defeat it and neither is a bad draw — the julia hook MANUFACTURES
z-plane supply from every native admission, and injected seed pools out-number native roots
inside the frontier. The conclusion there is the whole design of this module:

    the mix is decided WHERE THE BATCH IS POPPED, and anything that only changes what enters
    the frontier is diluted by whatever multiplies fastest inside it.

`deficit_scheduler.pick_partition` already pops rather than draws, but it is not a quota: it
is a per-batch stochastic argmax on price-weighted deficit, so it steers the mix without ever
MEASURING it. A stale price, a partition that happens to expand cheaply, or a run whose early
batches are unrepresentative all move the realized share with nothing to pull it back. This
module closes that loop: it computes an INTENDED share vector once per pop from the standing
deficits, tracks the REALIZED share of active minutes, and serves whichever servable partition
is furthest below its intent. That is a quota enforced at the population level — the realized
mix converges on the intent no matter what multiplies inside the frontier — and it is the
run's headline metric rather than a hope.

THE CURRENCY IS MATT'S, AND IT IS HUMAN LABELS
----------------------------------------------
    currency(partition) = count(label == 4) + 0.1 * count(label == 3)

counted through the amendment overlay (`label_store.resolve_score`, so a revision counts as
the revised value and the original stays byte-identical) plus the library. The target is
UNIFORM: every partition should hold the same currency, so the target level is the RICHEST
partition's holding and a partition's deficit is what it would take to reach it. That makes
the richest partition's deficit exactly zero — which is not a degenerate case here but the
one the addendum's floor is written for.

NO MACHINE SCORE ENTERS THE DEFICIT. This is the same hard boundary
`deficit_scheduler.py` states and it survives verbatim: a q3/q4 COUNT from the classifier
measures the classifier, not the family (`measurement_practice.md` §2), so the demand side is
human labels only. The classifier is allowed into the PRICE — see the next section, and the
clamp that bounds what it can do there.

THE PRICE IS MEASURED COST-TO-MINE, AND IT IS CLAMPED
-----------------------------------------------------
    price(partition) = active-minutes per unit of currency mined

estimated from the run's own telemetry: a canonical decode of 4 credits 1.0 unit, a decode of
3 credits 0.1, and the minutes spent since that partition's last credit are the cost. Seeded
from config, EMA-smoothed, and **clamped to a bounded band around the seed**. The clamp is
not tidiness: the numerator of a price is the classifier's decode, so a head that over-calls
4s in one family would make that family look cheap and buy it more service — winner's curse
climbing a level (`measurement_practice.md` §2). The clamp bounds that to a factor, and the
universal floor below bounds it again from the other side.

THE UNIVERSAL FLOOR (addendum, Matt)
------------------------------------
Every partition — including the currency-rich ones — receives a floor of ~5% of TOTAL TIME.
It is a **floor, not a quota**: a partition whose deficit allocation already exceeds 5% gets
nothing extra, a partition with zero deficit still gets its 5%. Implemented as
floor-constrained proportional water-filling (`allocate`), which is what makes both halves of
that sentence true simultaneously and keeps the shares summing to exactly 1.

Recorded rationale: spending 100% of the time on a stubborn deficit partition means never
learning anything new about the rich ones. The floor keeps every partition's cost-to-mine
price FRESH for the scheduler, keeps rich-type material flowing to emission's diversity
targets, and keeps the cross-feed alive — rich-base admissions are what trigger maneuvers and
julia hooks into the deficit partitions.

Every allocation is stamped floor-driven or deficit-driven per partition, so the realized
floor-vs-deficit split is a read rather than a reconstruction.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ------------------------------------------------------------------------- #
# Currency constants. Named here because three readers (the census, the in-run credit, the
# readout) must weight a class-3 identically or the deficit and the price stop being
# denominated in the same thing.
# ------------------------------------------------------------------------- #
CLASS_WEIGHT = {4: 1.0, 3: 0.1}          # Matt's currency: n4 + 0.1*n3
FLOOR_FRAC = 0.05                        # addendum: every partition floors at 5% of TOTAL time
SEED_PRICE = 3.0                         # neutral seed: active-minutes per currency unit
PRICE_EMA = 0.30                         # weight on the newest per-unit sample
PRICE_CLAMP = 4.0                        # price stays within [seed/4, seed*4]
CAP_MINUTES = 25.0                       # dry-time before a partition is capped out of service
JULIA_ROUTE_GAIN = 1.0                   # unservable julia intent folded into its c-plane parent


def is_julia(partition: str) -> bool:
    return partition.startswith("julia:")


def cplane_of(partition: str) -> str | None:
    return partition.split(":", 1)[1] if is_julia(partition) else None


# ========================================================================== #
# 1. The currency census — human labels through the amendment overlay + library.
# ========================================================================== #
@dataclass
class CurrencyCensus:
    """What each partition already holds, and enough provenance to argue about it."""
    counts: dict                    # partition -> {1: n, 2: n, 3: n, 4: n}
    currency: dict                  # partition -> n4 + 0.1*n3
    defaulted_rows: int             # rows with NO fractal_type, routed by the tree's default
    sources: dict                   # source name -> labeled rows contributed
    partitions: list

    def summary(self) -> dict:
        return dict(currency={p: round(self.currency.get(p, 0.0), 3) for p in self.partitions},
                    counts={p: self.counts.get(p, {}) for p in self.partitions},
                    defaulted_rows=self.defaulted_rows, sources=self.sources,
                    weights={str(k): v for k, v in CLASS_WEIGHT.items()})


def _partition_of_render(render: dict) -> str:
    """The tree's ONE settled rule for a render block with no `fractal_type`.

    `tools/corpus/location.py` (and `build_anchor_batch` / `build_revisit_batches` after it)
    resolve an absent token to `mandelbrot`, because the pre-2026-06-26 batches predate the
    multi-family schema and are all c-plane degree 2. That default is inherited rather than
    re-decided here — but the COUNT of rows that took it is reported, because a silent default
    carrying ~160 currency units into one partition is exactly the kind of thing a deficit
    should not be allowed to hide."""
    from partitions import partition_of                     # noqa: E402 (tools/scoring)
    ft = render.get("fractal_type") or render.get("family")
    return partition_of(ft, ft) if ft else "mandelbrot"


def label_currency(partitions: list[str], corpus_dir: str | None = None,
                   library_globs: list[str] | None = None) -> CurrencyCensus:
    """Census the human-label currency per partition.

    `corpus_dir` -> the label corpus, read through `corpus_reader.iter_labeled`, which applies
    the amendment overlay (a registered revision wins over the merged label). `library_globs`
    -> the library's own `images.jsonl` files; today the wallpaper library carries zero scored
    rows, and that zero is DERIVED here rather than asserted, so the census self-updates the
    day the library starts carrying human verdicts."""
    import glob
    import corpus_reader as cr                              # noqa: E402 (tools/corpus)

    counts: dict = {p: Counter() for p in partitions}
    sources: dict = {}
    defaulted = 0

    n = 0
    for lc in cr.iter_labeled(corpus_dir):
        render = lc.render or {}
        if not (render.get("fractal_type") or render.get("family")):
            defaulted += 1
        p = _partition_of_render(render)
        counts.setdefault(p, Counter())[int(lc.score)] += 1
        n += 1
    sources["label_corpus"] = n

    globs = library_globs if library_globs is not None else [
        str(ROOT / "data" / "wallpaper_corpus" / "batches" / "*" / "images.jsonl")]
    n = 0
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                score = (row.get("label") or {}).get("score")
                if score is None:
                    continue
                render = row.get("render") or {}
                if not (render.get("fractal_type") or render.get("family")):
                    defaulted += 1
                counts.setdefault(_partition_of_render(render), Counter())[int(score)] += 1
                n += 1
    sources["library"] = n

    currency = {p: sum(CLASS_WEIGHT.get(k, 0.0) * v for k, v in counts.get(p, {}).items())
                for p in set(partitions) | set(counts)}
    return CurrencyCensus(counts={p: dict(c) for p, c in counts.items()},
                          currency=currency, defaulted_rows=defaulted, sources=sources,
                          partitions=list(partitions))


def deficits_from_currency(currency: dict, partitions: list[str]) -> dict:
    """Shortfall against a UNIFORM target.

    Uniform means every partition should hold the same currency, so the target LEVEL is the
    richest partition's holding and a deficit is the distance up to it. Two properties this
    buys over "target = the mean": every deficit is >= 0 without a clamp (a clamp would make
    the allocation depend on how many partitions happen to be above the mean), and the
    richest partition lands at exactly zero — which is the case the universal floor exists to
    serve, so the two rules meet cleanly instead of fighting."""
    have = {p: float(currency.get(p, 0.0)) for p in partitions}
    level = max(have.values()) if have else 0.0
    return {p: max(0.0, level - v) for p, v in have.items()}


# ========================================================================== #
# 2. The allocation — floor-constrained proportional water-filling.
# ========================================================================== #
@dataclass
class Allocation:
    share: dict                     # partition -> intended fraction of total active time
    floored: set                    # partitions whose share is the floor (floor-driven)
    pwd: dict                       # price-weighted deficit that drove the proportional part
    floor: float

    def bucket(self, partition: str) -> str:
        return "floor" if partition in self.floored else "deficit"

    def summary(self) -> dict:
        return dict(share={p: round(v, 4) for p, v in sorted(self.share.items())},
                    floored=sorted(self.floored), floor=self.floor,
                    floor_share_total=round(sum(self.share[p] for p in self.floored), 4),
                    pwd={p: round(v, 5) for p, v in sorted(self.pwd.items())})


def allocate(deficits: dict, prices: dict, partitions: list[str],
             floor: float = FLOOR_FRAC) -> Allocation:
    """Intended time-share per partition: every partition >= `floor`, the rest proportional to
    price-weighted deficit. Shares sum to exactly 1.

    WATER-FILLING, not "reserve n*floor then split the rest", and the difference is the
    addendum's own wording. Reserving 9x5% and splitting the other 55% would hand a partition
    with a huge deficit its 5% ON TOP of its proportional share — "nothing extra" is exactly
    what the addendum says a rich-allocation partition gets. So instead: normalize the
    price-weighted deficits, pin every partition that falls below the floor UP to the floor,
    and redistribute what is left among the unpinned in proportion. Iterate to a fixpoint
    (pinning one partition lowers the pool, which can push another under the floor).

    Degenerate cases, each with a reason rather than a fallback:
      * every deficit zero (a fresh uniform corpus) -> nothing to be proportional to, so the
        budget is spread uniformly. Those partitions are NOT tagged floor-driven unless 1/n
        is itself below the floor: `floored` means "the floor is what set this share", and a
        degenerate-uniform allocation was not set by the floor. Tagging it floor-driven would
        report a 100% floor share on a run where the floor never bound anything.
      * floor * n > 1 -> the floor is infeasible; it degrades to uniform rather than raising,
        because a run that cannot honour the floor should still run, and every partition IS
        tagged floored there, since the floor is exactly what could not be honoured.

    `floor * |floored|` is therefore the floor's total claim, and it is bounded by
    `floor * n` — the addendum's "up to ~45%" at nine partitions."""
    parts = list(partitions)
    n = len(parts)
    if n == 0:
        return Allocation(share={}, floored=set(), pwd={}, floor=floor)
    if floor * n >= 1.0:
        share = {p: 1.0 / n for p in parts}
        return Allocation(share=share, floored=set(parts), pwd={p: 0.0 for p in parts},
                          floor=floor)

    pwd = {p: max(0.0, float(deficits.get(p, 0.0)))
              / max(float(prices.get(p, SEED_PRICE)), 1e-9) for p in parts}
    total = sum(pwd.values())
    if total <= 0.0:
        share = {p: 1.0 / n for p in parts}
        return Allocation(share=share, floored=(set(parts) if 1.0 / n < floor else set()),
                          pwd=pwd, floor=floor)

    pinned: set = set()
    share = {p: pwd[p] / total for p in parts}
    while True:
        below = [p for p in parts if p not in pinned and share[p] < floor - 1e-12]
        if not below:
            break
        pinned.update(below)
        rest = [p for p in parts if p not in pinned]
        pool = 1.0 - floor * len(pinned)
        for p in pinned:
            share[p] = floor
        sub = sum(pwd[p] for p in rest)
        if not rest or pool <= 0.0:
            break
        for p in rest:
            share[p] = pool * (pwd[p] / sub) if sub > 0 else pool / len(rest)
    # numerical tidy-up: renormalize the UNPINNED mass only, so the floor stays exact.
    rest = [p for p in parts if p not in pinned]
    if rest:
        pool = 1.0 - floor * len(pinned)
        s = sum(share[p] for p in rest)
        if s > 0:
            for p in rest:
                share[p] = share[p] * pool / s
    return Allocation(share=share, floored=pinned, pwd=pwd, floor=floor)


# ========================================================================== #
# 3. The price model — measured cost-to-mine, clamped.
# ========================================================================== #
class CostToMine:
    """Per-partition active-minutes per unit of Matt's currency, measured in-run.

    `charge` accounts a batch's active minutes to the partition it served. `credit` is called
    with the currency a canonical decode just produced (a decoded 4 -> 1.0, a decoded 3 ->
    0.1); the minutes since that partition's last credit divided by the units is one price
    sample, EMA'd in. A partition that burns `cap_minutes` with zero credit is CAPPED out of
    service until something re-opens it — an unbounded stall on a partition whose queue is
    full of dead ground would otherwise eat the whole run's quota for that partition.

    THE CLAMP. `price` never leaves `[seed/clamp, seed*clamp]`. The raw EMA is kept and
    reported so the clamp is visible rather than silent."""

    def __init__(self, partitions: list[str], config: dict | None = None):
        cfg = config or {}
        self.seed = {p: float((cfg.get("prices") or {}).get(p, cfg.get("seed_price", SEED_PRICE)))
                     for p in partitions}
        self.ema = float(cfg.get("price_ema", PRICE_EMA))
        self.clamp = float(cfg.get("price_clamp", PRICE_CLAMP))
        self.cap_minutes = float(cfg.get("cap_minutes", CAP_MINUTES))
        self.raw = dict(self.seed)
        self.min_since_credit = {p: 0.0 for p in partitions}
        self.min_spent = {p: 0.0 for p in partitions}
        self.units = {p: 0.0 for p in partitions}
        self.capped: set = set()

    def ensure(self, p: str):
        if p not in self.raw:
            self.seed[p] = SEED_PRICE
            self.raw[p] = SEED_PRICE
            self.min_since_credit[p] = 0.0
            self.min_spent[p] = 0.0
            self.units[p] = 0.0

    def price(self, p: str) -> float:
        self.ensure(p)
        lo, hi = self.seed[p] / self.clamp, self.seed[p] * self.clamp
        return min(max(self.raw[p], lo), hi)

    def prices(self) -> dict:
        return {p: self.price(p) for p in self.raw}

    def charge(self, p: str, minutes: float) -> bool:
        self.ensure(p)
        self.min_spent[p] += minutes
        self.min_since_credit[p] += minutes
        if p not in self.capped and self.min_since_credit[p] >= self.cap_minutes:
            self.capped.add(p)
            return True
        return False

    def credit(self, p: str, units: float):
        """`units` of currency just mined in `p`. Zero units is not a credit — it must not
        reset the dry-time counter, or a partition producing only class-1s would never cap."""
        self.ensure(p)
        if units <= 0:
            return
        self.units[p] += units
        sample = self.min_since_credit[p] / units
        if sample > 0:
            self.raw[p] = (1 - self.ema) * self.raw[p] + self.ema * sample
        self.min_since_credit[p] = 0.0
        self.capped.discard(p)

    def reopen_caps(self):
        for p in list(self.capped):
            self.min_since_credit[p] = 0.0
        self.capped.clear()

    def state_dict(self) -> dict:
        return dict(seed=self.seed, raw=self.raw, min_since_credit=self.min_since_credit,
                    min_spent=self.min_spent, units=self.units, capped=sorted(self.capped),
                    ema=self.ema, clamp=self.clamp, cap_minutes=self.cap_minutes)

    def load_state(self, d: dict):
        for k in ("seed", "raw", "min_since_credit", "min_spent", "units"):
            getattr(self, k).update({p: float(v) for p, v in (d.get(k) or {}).items()})
        self.capped = set(d.get("capped", []))

    def summary(self) -> dict:
        return dict(price={p: round(v, 3) for p, v in sorted(self.prices().items())},
                    price_raw={p: round(v, 3) for p, v in sorted(self.raw.items())},
                    seed={p: round(v, 3) for p, v in sorted(self.seed.items())},
                    clamped=sorted(p for p in self.raw
                                   if abs(self.price(p) - self.raw[p]) > 1e-9),
                    units_mined={p: round(v, 3) for p, v in sorted(self.units.items())},
                    min_spent={p: round(v, 2) for p, v in sorted(self.min_spent.items())},
                    capped=sorted(self.capped), clamp_factor=self.clamp)


# ========================================================================== #
# 4. The pop decision — PURE, and it is the quota.
# ========================================================================== #
def choose_partition(intended: dict, realized_min: dict, servable: set,
                     capped: set | None = None) -> str | None:
    """Serve the servable partition furthest BELOW its intended share of realized time.

    Pure, deterministic, and a function of (intended shares, realized minutes, servability)
    ONLY — no per-node score, no p_good, no RNG. Determinism is the point: v1's stochastic
    argmax could steer a mix but never converge one, and a quota that is allowed to be lucky
    cannot be read as evidence about the allocator.

    The gap is measured in SHARE space (`intended - realized/total`) rather than in minutes,
    so it is scale-free: the same decision is made in minute one and hour six. Before any time
    has been spent the realized share is zero everywhere, so the first pop goes to the largest
    intended share — which is correct, and is why there is no special case for it.

    A capped partition is excluded but keeps its intent, so its unserved share shows up in the
    realized-vs-intended report as a miss with a named cause rather than quietly redistributing.
    """
    cand = sorted(p for p in servable if p not in (capped or set()))
    if not cand:
        return None
    total = sum(max(0.0, v) for v in realized_min.values())
    def gap(p):
        got = (realized_min.get(p, 0.0) / total) if total > 0 else 0.0
        return intended.get(p, 0.0) - got
    return max(cand, key=lambda p: (gap(p), p))


def fold_julia_intent(intended: dict, queue_lens: dict, partitions: list[str],
                      gain: float = JULIA_ROUTE_GAIN) -> dict:
    """A `julia:X` partition CANNOT be popped into existence — it is fed only by descending
    c-plane X and firing the hook on a qualifying parent (`discovery_pipeline.md` §3). So when
    a julia twin has intent but no queue, its intent is folded into its c-plane parent's
    EFFECTIVE intent: serving the parent is what manufactures the twin's supply.

    Returns a NEW dict; the original `intended` is what the realized-vs-intended report is
    scored against, and must not be mutated into the thing the pop actually used — reporting
    against the folded vector would grade the run on a target it moved."""
    eff = dict(intended)
    for jp in partitions:
        if not is_julia(jp) or queue_lens.get(jp, 0) > 0:
            continue
        cp = cplane_of(jp)
        if cp in eff:
            eff[cp] = eff[cp] + gain * intended.get(jp, 0.0)
            eff[jp] = 0.0
    return eff


# ========================================================================== #
# 5. The object the driver holds.
# ========================================================================== #
@dataclass
class QuotaState:
    realized_min: dict = field(default_factory=dict)
    realized_by_bucket: dict = field(default_factory=dict)   # partition -> {floor, deficit}
    pops: dict = field(default_factory=dict)                 # partition -> batches served
    candidates: dict = field(default_factory=dict)           # partition -> candidates produced
    admitted: dict = field(default_factory=dict)             # partition -> admissions


class PopQuota:
    """Owns the currency census, the price model, the intended allocation and the realized
    tally; names the partition to pop; and reports realized-vs-intended.

    Re-allocation happens every pop (the prices move, and a censused deficit does not — the
    human labels this run produces do not arrive until a sitting comes back). That is cheap:
    `allocate` is O(n^2) on nine partitions."""

    def __init__(self, partitions: list[str], run_dir: Path, *,
                 floor: float = FLOOR_FRAC, prices_config: dict | None = None,
                 census: CurrencyCensus | None = None,
                 julia_route_gain: float = JULIA_ROUTE_GAIN):
        self.partitions = list(partitions)
        self.run_dir = Path(run_dir)
        self.floor = float(floor)
        self.julia_route_gain = float(julia_route_gain)
        self.census = census if census is not None else label_currency(self.partitions)
        self.deficit = deficits_from_currency(self.census.currency, self.partitions)
        self.cost = CostToMine(self.partitions, prices_config)
        self.state = QuotaState(
            realized_min={p: 0.0 for p in self.partitions},
            realized_by_bucket={p: {"floor": 0.0, "deficit": 0.0} for p in self.partitions},
            pops={p: 0 for p in self.partitions},
            candidates={p: 0 for p in self.partitions},
            admitted={p: 0 for p in self.partitions})
        self.trace_path = self.run_dir / "quota_trace.jsonl"
        self._last_alloc: Allocation | None = None
        self._last_eff: dict | None = None
        self._served: str | None = None
        # TIME-WEIGHTED MEAN EFFECTIVE INTENT, accumulated as the run goes.
        #
        # This is the vector the realized mix must actually be scored against, and leaving it
        # to be recomputed offline is what made the first proving run read as a MISS. The
        # STATED intent contains demand for julia partitions that cannot be popped into
        # existence; by the §3 routing rule that demand folds into the c-plane parent, so a
        # run that serves the parent correctly is scored as over-serving a native. Measured on
        # harvest_v2_proving_20260803: L1 gap 0.352 against the stated intent, 0.091 against
        # the effective one — the same run, and only the second is a statement about the pop.
        #
        # Weighted by MINUTES rather than by batches because the effective vector changes with
        # queue occupancy, and an intent that held while an expensive batch ran governed more
        # of the run than one that held through a cheap one.
        self._eff_accum: dict = {p: 0.0 for p in self.partitions}
        self._eff_weight: float = 0.0

    # ---- allocation ----------------------------------------------------- #
    def allocation(self) -> Allocation:
        return allocate(self.deficit, self.cost.prices(), self.partitions, self.floor)

    def pick(self, queue_lens: dict) -> str | None:
        alloc = self.allocation()
        self._last_alloc = alloc
        servable = {p for p, n in queue_lens.items() if n > 0}
        eff = fold_julia_intent(alloc.share, queue_lens, self.partitions,
                                self.julia_route_gain)
        self._last_eff = eff
        part = choose_partition(eff, self.state.realized_min, servable, self.cost.capped)
        self._served = part
        return part

    def effective_intent(self) -> dict:
        """The time-weighted mean of the vector the pop actually acted on. Falls back to the
        current allocation before any time has been charged."""
        if self._eff_weight <= 0:
            return dict(self._last_eff or self.allocation().share)
        return {p: v / self._eff_weight for p, v in self._eff_accum.items()}

    # ---- accounting ----------------------------------------------------- #
    def charge(self, partition: str, minutes: float) -> bool:
        """Account a served batch's active minutes, tagged with the bucket that bought it."""
        if partition is None:
            return False
        st = self.state
        st.realized_min[partition] = st.realized_min.get(partition, 0.0) + minutes
        st.pops[partition] = st.pops.get(partition, 0) + 1
        bucket = (self._last_alloc.bucket(partition) if self._last_alloc else "deficit")
        b = st.realized_by_bucket.setdefault(partition, {"floor": 0.0, "deficit": 0.0})
        b[bucket] = b.get(bucket, 0.0) + minutes
        if self._last_eff:
            for p, v in self._last_eff.items():
                self._eff_accum[p] = self._eff_accum.get(p, 0.0) + v * minutes
            self._eff_weight += minutes
        return self.cost.charge(partition, minutes)

    def credit_decode(self, partition: str, decoded_class):
        """One canonical decode landed in `partition`. Currency weight only — a decoded 1 or 2
        is not a credit and deliberately does NOT reset the partition's dry-time clock."""
        if decoded_class is None:
            return 0.0
        units = CLASS_WEIGHT.get(int(decoded_class), 0.0)
        if units:
            self.cost.credit(partition, units)
        return units

    def note_candidates(self, partition: str, n: int):
        self.state.candidates[partition] = self.state.candidates.get(partition, 0) + int(n)

    def note_admission(self, partition: str):
        self.state.admitted[partition] = self.state.admitted.get(partition, 0) + 1

    # ---- reporting ------------------------------------------------------ #
    def realized_share(self, key: str = "minutes") -> dict:
        src = {"minutes": self.state.realized_min, "candidates": self.state.candidates,
               "admitted": self.state.admitted, "pops": self.state.pops}[key]
        tot = sum(src.get(p, 0) for p in self.partitions)
        if tot <= 0:
            return {p: 0.0 for p in self.partitions}
        return {p: src.get(p, 0) / tot for p in self.partitions}

    def mix_report(self) -> dict:
        """REALIZED vs INTENDED, the run's headline. Three denominations, because they answer
        different questions and v1's failure was quoted in the third: minutes is what the
        quota allocates and therefore what it can be held to; candidates is what
        `discovery_pipeline.md` §3.1 measured the 19.6% in; admissions is what the corpus
        actually gains."""
        alloc = self._last_alloc or self.allocation()
        eff = self.effective_intent()
        out = {}
        for denom in ("minutes", "candidates", "admitted", "pops"):
            got = self.realized_share(denom)
            out[denom] = {p: dict(intended=round(alloc.share.get(p, 0.0), 4),
                                  effective=round(eff.get(p, 0.0), 4),
                                  realized=round(got.get(p, 0.0), 4),
                                  delta=round(got.get(p, 0.0) - alloc.share.get(p, 0.0), 4),
                                  delta_effective=round(got.get(p, 0.0) - eff.get(p, 0.0), 4))
                          for p in self.partitions}
        out["l1_gap_minutes"] = round(
            sum(abs(out["minutes"][p]["delta"]) for p in self.partitions) / 2.0, 4)
        # THE GAP THAT IS A STATEMENT ABOUT THE POP. The stated intent carries demand for
        # julia partitions that cannot be popped into existence; §3's routing folds that
        # demand into the c-plane parent, so scoring against the stated vector charges a run
        # for serving the parent exactly as instructed.
        out["l1_gap_minutes_effective"] = round(
            sum(abs(out["minutes"][p]["delta_effective"]) for p in self.partitions) / 2.0, 4)
        out["effective_intent"] = {p: round(v, 4) for p, v in sorted(eff.items())}
        return out

    def floor_vs_deficit(self) -> dict:
        """The addendum's separate read: how much realized time each bucket bought, per
        partition and in total."""
        per = {p: {k: round(v, 3) for k, v in b.items()}
               for p, b in self.state.realized_by_bucket.items()}
        tf = sum(b.get("floor", 0.0) for b in self.state.realized_by_bucket.values())
        td = sum(b.get("deficit", 0.0) for b in self.state.realized_by_bucket.values())
        tot = tf + td
        return dict(per_partition=per, floor_min=round(tf, 2), deficit_min=round(td, 2),
                    floor_share=(round(tf / tot, 4) if tot else None),
                    deficit_share=(round(td / tot, 4) if tot else None))

    def log_choice(self, batch: int, chosen: str | None, queue_lens: dict):
        alloc = self._last_alloc or self.allocation()
        rec = dict(batch=batch, chosen=chosen,
                   bucket=(alloc.bucket(chosen) if chosen else None),
                   intended={p: round(alloc.share.get(p, 0.0), 4) for p in self.partitions},
                   # The vector the pop ACTED on, logged rather than left to be recomputed
                   # from `intended` + `queue_lens` by a reader who knows the fold rule.
                   effective={p: round(v, 4) for p, v in (self._last_eff or {}).items()},
                   realized={p: round(v, 4)
                             for p, v in self.realized_share("minutes").items()},
                   deficit={p: round(self.deficit.get(p, 0.0), 3) for p in self.partitions},
                   price={p: round(v, 3) for p, v in self.cost.prices().items()},
                   capped=sorted(self.cost.capped),
                   queue_lens={p: int(queue_lens.get(p, 0)) for p in self.partitions})
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- state ---------------------------------------------------------- #
    def state_dict(self) -> dict:
        return dict(partitions=self.partitions, floor=self.floor,
                    julia_route_gain=self.julia_route_gain,
                    deficit=self.deficit, cost=self.cost.state_dict(),
                    realized_min=self.state.realized_min,
                    realized_by_bucket=self.state.realized_by_bucket,
                    pops=self.state.pops, candidates=self.state.candidates,
                    admitted=self.state.admitted)

    def load_state(self, d: dict, reopen_caps: bool = False):
        self.floor = float(d.get("floor", self.floor))
        self.julia_route_gain = float(d.get("julia_route_gain", self.julia_route_gain))
        # The DEFICIT is re-censused on resume rather than restored: the label corpus can gain
        # a sitting between sessions, and a resumed run reading a stale deficit would allocate
        # against a corpus that no longer exists. The checkpointed copy is kept only for the
        # readout's record of what the previous session ran on.
        self.state.realized_min.update({p: float(v)
                                        for p, v in (d.get("realized_min") or {}).items()})
        self.state.realized_by_bucket.update(d.get("realized_by_bucket") or {})
        self.state.pops.update({p: int(v) for p, v in (d.get("pops") or {}).items()})
        self.state.candidates.update({p: int(v)
                                      for p, v in (d.get("candidates") or {}).items()})
        self.state.admitted.update({p: int(v) for p, v in (d.get("admitted") or {}).items()})
        self.cost.load_state(d.get("cost") or {})
        if reopen_caps:
            self.cost.reopen_caps()

    def summary(self) -> dict:
        alloc = self.allocation()
        return dict(currency=self.census.summary(),
                    deficit={p: round(self.deficit.get(p, 0.0), 3) for p in self.partitions},
                    target_rule="uniform: level every partition to the richest holding",
                    allocation=alloc.summary(), cost=self.cost.summary(),
                    mix=self.mix_report(), floor_vs_deficit=self.floor_vs_deficit(),
                    realized_min={p: round(v, 2) for p, v in self.state.realized_min.items()},
                    trace=str(self.trace_path))


# ========================================================================== #
# CLI — the census + the allocation it implies, without running anything.
# ========================================================================== #
def main():
    import argparse
    from partitions import ALL_FAMS                          # noqa: E402
    ap = argparse.ArgumentParser(description="census the label currency and the allocation "
                                             "it implies")
    ap.add_argument("--floor", type=float, default=FLOOR_FRAC)
    ap.add_argument("--partitions", default=",".join(ALL_FAMS))
    args = ap.parse_args()
    parts = [p for p in args.partitions.split(",") if p]
    cen = label_currency(parts)
    defc = deficits_from_currency(cen.currency, parts)
    alloc = allocate(defc, {p: SEED_PRICE for p in parts}, parts, args.floor)
    print(json.dumps(dict(currency=cen.summary(), deficits={p: round(defc[p], 2) for p in parts},
                          allocation_at_seed_prices=alloc.summary()), indent=2))


if __name__ == "__main__":
    main()
