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
the revised value and the original stays byte-identical) plus the library.

THE TARGET IS RATIO-WEIGHTED (Matt, 2026-08-04). It was UNIFORM until then — every partition
levelled to the RICHEST partition's holding — which said that a pinned single-parameter-point
plane with 16 distinct looks and the mandelbrot c-plane are owed the same number of labels.
They are not: the intended release mix is one table (`release_mix.RATIO`) and the demand side
reads it, so `target_p ∝ ratio_p` with the maximum-ratio partitions anchored at the level the
uniform rule used. The richest maximum-ratio partition still lands at exactly zero deficit —
which is not a degenerate case here but the one the addendum's floor is written for — and a
low-ratio partition's demand is now bounded by what it is meant to be worth rather than by
what the biggest family holds. Nothing else moves: the deficit definition, the price weighting
and the universal 5% floor are unchanged; only the target vector.

NO MACHINE SCORE ENTERS THE DEFICIT. This is the same hard boundary
`deficit_scheduler.py` states and it survives verbatim: a q3/q4 COUNT from the classifier
measures the classifier, not the family (`measurement_practice.md` §2), so the demand side is
human labels only. The classifier is allowed into the PRICE — see the next section, and the
clamp that bounds what it can do there.

THE PRICE IS MEASURED COST-TO-MINE, AND IT IS CLAMPED
-----------------------------------------------------
    price(partition) = active-minutes per unit of currency mined

estimated from the run's own telemetry: a canonical decode of 4 credits 1.0 unit, a decode of
3 credits 0.1, and the minutes charged to that partition are the cost. Seeded
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
    should not be allowed to hide.

    VERIFIED 2026-08-03, and RE-VERIFIED independently the same day (`corpus_reader.iter_labeled`,
    census by batch; the re-run reproduced 4,935 / 159.2 / max |c| = 1.7762 exactly): all 4,935
    defaulted rows are genuinely mandelbrot, and they carry 159.2 of mandelbrot's 186.8
    currency — so this default is load-bearing for the partition that SETS the uniform target
    level (its deficit is 0 by construction), not just for its own share. The price does not
    move. They come from exactly six batches — of the LABELED population; see the caveat
    below — all created 2026-06-23..25: `flat_generate_loose0_v3`, `guided_descend_rev4`,
    `guided_descend_rev4occfix_v2filtered`, `mining_v3guided_v1`, `scale_2x2_labelset`,
    `scale_controlled_2x2`. Three independent reasons, not one:
      * `julia_ladder_j0` (created 2026-06-25, the same day as the last of them) is the batch
        whose `batch.json` records `schema_extension: render block adds fractal_type/c_re/c_im
        for Julia rows` — the token entered the schema WITH the first non-mandelbrot family, so
        "no token" dates a row to before any other family existed in the corpus;
      * every one of the six writes a render block with no `c_re`/`c_im` at all (0 of 5,050
        rows), so none of them is a parameter-plane family;
      * all 5,050 centers lie inside the mandelbrot c-plane box (re ∈ [-2.05, 0.75],
        |im| <= 1.25; max |c| = 1.78) — `flat_generate`'s own sampling box is that rectangle.
    No re-render was needed. Nothing to correct: mandelbrot's price and its floored 5% share
    stand as the proving run reported them.

    THE SIX IS A FACT ABOUT THE LABELED POPULATION, NOT ABOUT THE CORPUS. Eleven batches
    carry token-less render blocks; the other five (`anchor_class4_v1`, `revisit_class3_c1..c4`,
    541 rows) hold ZERO scored rows today, which is the only reason the census sees six. They
    were created 2026-07-26..28, a month AFTER the schema extension, so reason (1) above — the
    dating argument — does NOT cover them. Reasons (2) and (3) do: 0 of the 541 carry
    `c_re`/`c_im` and all 541 centers are inside the mandelbrot box (max |c| = 1.353), so they
    would route correctly if labeled. That is checked, not assumed:
    `test_pop_quota.py::test_every_defaulted_row_is_mandelbrot_SHAPED` asserts the shape
    invariant over every token-less row in the corpus, labeled or not, so a future batch that
    defaults to mandelbrot without earning it goes red. The COUNT is deliberately not pinned —
    it moves the day Matt labels the revisit batches, and that is not a regression.

    RE-VERIFIED FROM THE WRITER SIDE 2026-08-05, because the three reasons above are all
    READER-side and two of them are weaker than they look: for a pre-extension row the
    schema could not express `c_re`/`c_im` at all, so their absence states nothing; and the
    mandelbrot box does not exclude a julia z-viewport (those centre near the origin) or a
    multibrot c-plane (`WalkFamily::flat_box_default` -> `root_field::degree_bbox`, an
    origin-centred square of half-width 2^(1/(d-1))*1.2 = ±1.70/1.51/1.43 for d=3/4/5).
    What the WRITERS say, over all 5,591 token-less render blocks in the corpus:
      * `flat_generate` (1,043 rows): its writer is `src/generate.rs`, which contains no
        family, degree, julia or phoenix axis of any kind — mandelbrot by construction, not
        by default.
      * guided-descend (3,557 rows): the walker stamps `root_src` = "julia"/"phoenix" in
        its dynamical modes and "8k"/"flat" on the c-plane. Every token-less row carries
        "8k", "flat" or null; not one carries a dynamical stamp. So the dynamical planes
        are excluded by a POSITIVE writer record, not by an absent field.
      * post-extension batches (541 rows, `anchor_class4_v1` + `revisit_class3_c1..c4`,
        unlabeled today): their schema CAN express a family and their batches are
        explicitly mixed-family, so absence there IS the writer's statement. All 9
        `anchor_class4_v1` rows are `anchor_mandelbrot_*` by image_id, alongside siblings
        of the same family that DO carry the token — the token is simply optional for the
        default family.
      * `mining_v3guided_v1` (450 rows, 9.1% of the 4,935): its writer and its source pool
        lived in `C:/Code/fractal-generator`, which no longer exists. Nothing writer-side
        survives for these; they rest on the reader-side reasons alone.
    THE ONE GAP THE WRITERS LEAVE IS DEGREE: neither writer records `--family`, so no row
    is writer-attested as d=2 rather than d=3/4/5. That is closed geometrically instead —
    ~44/49/52% of the d=3/4/5 root boxes lie outside the mandelbrot rectangle, and 0 of
    5,591 centres do (max |c| = 1.7762). VERDICT: genuinely mandelbrot, not
    family-ambiguous. The price does not move.

    THE PHOENIX SPLIT (2026-08-04) IS RESOLVED HERE, and it is the reason this reads the whole
    render block rather than the token: `phoenix:classic` and `phoenix` share one
    `fractal_type` and are told apart only by the parameter point (`partitions.partition_of_row`).
    A corpus render block always carries `cx`/`fw`, so it is always in a schema that can answer
    — an axis-free legacy row resolves classic (which is what all 84 classic rows in the corpus
    are), and a varied row resolves `phoenix`. Nothing on disk is re-keyed."""
    from partitions import partition_of_row                 # noqa: E402 (tools/scoring)
    ft = render.get("fractal_type") or render.get("family")
    return partition_of_row(render, ft) if ft else "mandelbrot"


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


TARGET_RULE = ("ratio-weighted (release_mix.RATIO): target_p = anchor * ratio_p / max(ratio), "
               "anchor = the richest partition's holding")


def currency_targets(currency: dict, partitions: list[str],
                     ratios: dict | None = None) -> tuple[dict, float]:
    """`(target per partition, the anchor level)` under the ratio table.

    THE ANCHOR IS THE RICHEST HOLDING, unchanged from the uniform rule, and that is what makes
    this a re-weighting rather than a rescaling: the maximum-ratio partitions keep exactly the
    target they had, so mandelbrot's deficit (0) and julia:mandelbrot's (27.0) do not move at
    all, and everything below ratio 3 falls proportionally. `phoenix:classic` at 0.2/3 lands at
    a fifteenth of the anchor.

    `ratios` is read at CALL TIME (default: the whole `release_mix` table) so a change to the
    policy moves the next allocation — `PopQuota` re-allocates every pop, and a table cached at
    import would keep a running frontier on the mix it was launched with.

    A partition with no declared ratio RAISES. It is the `partitions._registered` failure one
    layer down: a defaulted ratio would give it a plausible target nobody decided, and every
    downstream read ("that partition had little demand") would be about the default."""
    have = {p: float(currency.get(p, 0.0)) for p in partitions}
    level = max(have.values()) if have else 0.0
    if ratios is None:
        from release_mix import ratios as _table              # noqa: E402 (tools/scoring)
        ratios = _table()
    missing = [p for p in partitions if p not in ratios]
    if missing:
        raise KeyError(
            f"no release-mix ratio for {missing} — register them in release_mix.RATIO (and in "
            f"partitions.ALL_FAMS). The target vector must not default: a defaulted ratio "
            f"reads downstream as a measured demand.")
    rmax = max(float(ratios[p]) for p in partitions) if partitions else 0.0
    if rmax <= 0:
        return {p: 0.0 for p in partitions}, level
    return {p: level * float(ratios[p]) / rmax for p in partitions}, level


def deficits_from_currency(currency: dict, partitions: list[str],
                           ratios: dict | None = None) -> dict:
    """Shortfall against the RATIO-WEIGHTED target (`currency_targets`).

    Two properties this keeps from the uniform rule it replaces: every deficit is >= 0 without
    a clamp (a clamp would make the allocation depend on how many partitions happen to be above
    the target), and the partition that sets the anchor lands at exactly zero — which is the
    case the universal floor exists to serve, so the two rules meet cleanly instead of
    fighting. What it adds is that a partition can now be AT its target while holding far less
    than the richest one, which is the whole point: `phoenix:classic` is 0.2 of a release, not
    a tenth of it."""
    have = {p: float(currency.get(p, 0.0)) for p in partitions}
    target, _level = currency_targets(currency, partitions, ratios)
    return {p: max(0.0, target[p] - have.get(p, 0.0)) for p in partitions}


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
    0.1). A partition that burns `cap_minutes` with zero credit is CAPPED out of
    service until something re-opens it — an unbounded stall on a partition whose queue is
    full of dead ground would otherwise eat the whole run's quota for that partition.

    THE SAMPLE IS BATCH-AGGREGATED, AND THAT IS A FIX, NOT A REFINEMENT
    ------------------------------------------------------------------
    v1 took one EMA sample PER DECODE: `minutes-since-last-credit / units of THAT decode`,
    with the dry-time counter reset by every credit. Two biases compound, and they push in
    opposite directions on different partitions, so the price table came out INVERTED on the
    biggest spenders (`allocator_prereg_v1_mechanism_read_20260804.md` §4; the run's own
    aggregate `min_spent/units` against the EMA it was quoting: multibrot4 **14.3 vs 4.69**,
    multibrot5 11.4 vs 2.48, mandelbrot 1.80 vs 9.24, julia:mandelbrot 3.0 vs 13.54):

      * within a burst of k decodes in one batch, the whole gap is divided by the FIRST
        decode's units only — the other k-1 decodes reset the counter to zero, sample 0.0,
        and are dropped by the `sample > 0` guard. Their units never appear in any
        denominator, so a productive clustered partition reads k times too EXPENSIVE.
      * the number of samples is the number of decode EVENTS, not the amount of work. A
        sparse-but-costly partition gets a handful of EMA updates and stays pinned near its
        seed price, reading too CHEAP.

    Since allocation share is deficit/price, an inverted table sends intent to the wrong
    partitions — which is the whole failure this class is upstream of.

    So: minutes and units accumulate per partition into a WINDOW, and `end_window` (one call
    per batch, from `PopQuota.charge`) emits at most ONE sample from the aggregate,
    `window minutes / window units`. That is the estimand — total cost over total currency —
    computed directly instead of approximated by a first-decode ratio. A window with no
    units does not flush: its minutes carry forward, so a partition that goes dry for twenty
    batches and then credits once samples the full twenty batches against that one credit.

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
        # The price WINDOW: minutes and units since this partition's last emitted sample.
        # Separate from `min_since_credit`, which is the CAP's dry-time clock and is reset by
        # every credit — the two counters answer different questions and sharing one is how
        # v1's burst bias got in.
        self.win_min = {p: 0.0 for p in partitions}
        self.win_units = {p: 0.0 for p in partitions}
        self.samples = {p: 0 for p in partitions}
        self.capped: set = set()

    def ensure(self, p: str):
        if p not in self.raw:
            self.seed[p] = SEED_PRICE
            self.raw[p] = SEED_PRICE
            self.min_since_credit[p] = 0.0
            self.min_spent[p] = 0.0
            self.units[p] = 0.0
            self.win_min[p] = 0.0
            self.win_units[p] = 0.0
            self.samples[p] = 0

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
        self.win_min[p] += minutes
        if p not in self.capped and self.min_since_credit[p] >= self.cap_minutes:
            self.capped.add(p)
            return True
        return False

    def credit(self, p: str, units: float):
        """`units` of currency just mined in `p`. Zero units is not a credit — it must not
        reset the dry-time counter, or a partition producing only class-1s would never cap.

        Accumulates into the price window; the EMA moves in `end_window`, never here. A
        decode is an EVENT and the price is a RATE, so one decode is not one sample."""
        self.ensure(p)
        if units <= 0:
            return
        self.units[p] += units
        self.win_units[p] += units
        self.min_since_credit[p] = 0.0
        self.capped.discard(p)

    def end_window(self) -> dict:
        """Close the price window: one EMA sample per partition that has both minutes and
        units in it. Called once per served batch (`PopQuota.charge`).

        A partition with units but no charged minutes does NOT flush — a zero sample would
        drag the EMA toward zero and price it as free. Its units stay in the window and are
        spent against the minutes that follow, which is the aggregate the estimand asks for.
        Returns {partition: sample} for the samples actually taken, so a caller can log what
        moved rather than diff two price tables."""
        taken = {}
        for p in list(self.win_units):
            u, m = self.win_units[p], self.win_min[p]
            if u <= 0 or m <= 0:
                continue
            sample = m / u
            self.raw[p] = (1 - self.ema) * self.raw[p] + self.ema * sample
            self.samples[p] = self.samples.get(p, 0) + 1
            self.win_units[p] = 0.0
            self.win_min[p] = 0.0
            taken[p] = sample
        return taken

    def reopen_caps(self):
        for p in list(self.capped):
            self.min_since_credit[p] = 0.0
        self.capped.clear()

    def state_dict(self) -> dict:
        return dict(seed=self.seed, raw=self.raw, min_since_credit=self.min_since_credit,
                    min_spent=self.min_spent, units=self.units, capped=sorted(self.capped),
                    win_min=self.win_min, win_units=self.win_units, samples=self.samples,
                    ema=self.ema, clamp=self.clamp, cap_minutes=self.cap_minutes)

    def load_state(self, d: dict):
        for k in ("seed", "raw", "min_since_credit", "min_spent", "units",
                  "win_min", "win_units"):
            getattr(self, k).update({p: float(v) for p, v in (d.get(k) or {}).items()})
        self.samples.update({p: int(v) for p, v in (d.get("samples") or {}).items()})
        self.capped = set(d.get("capped", []))

    def summary(self) -> dict:
        return dict(price={p: round(v, 3) for p, v in sorted(self.prices().items())},
                    price_raw={p: round(v, 3) for p, v in sorted(self.raw.items())},
                    seed={p: round(v, 3) for p, v in sorted(self.seed.items())},
                    clamped=sorted(p for p in self.raw
                                   if abs(self.price(p) - self.raw[p]) > 1e-9),
                    units_mined={p: round(v, 3) for p, v in sorted(self.units.items())},
                    min_spent={p: round(v, 2) for p, v in sorted(self.min_spent.items())},
                    # The aggregate the EMA is an estimate OF. Quoted beside it because the
                    # v1 defect was invisible until the two were read together.
                    price_aggregate={p: round(self.min_spent[p] / self.units[p], 3)
                                     for p in sorted(self.units) if self.units.get(p, 0) > 0},
                    price_samples={p: v for p, v in sorted(self.samples.items()) if v},
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
                 julia_route_gain: float = JULIA_ROUTE_GAIN,
                 ratios: dict | None = None):
        self.partitions = list(partitions)
        self.run_dir = Path(run_dir)
        self.floor = float(floor)
        self.julia_route_gain = float(julia_route_gain)
        self.census = census if census is not None else label_currency(self.partitions)
        # The target vector is resolved HERE, from the live table, and kept beside the deficit
        # it produced — the deficit alone cannot say whether a partition is quiet because it is
        # near its target or because its target is small.
        self.target, self.anchor = currency_targets(self.census.currency, self.partitions,
                                                    ratios)
        if ratios is None:
            import release_mix                                # noqa: E402 (tools/scoring)
            ratios = release_mix.ratios(self.partitions)
        self.ratios = {p: float(ratios[p]) for p in self.partitions}
        self.deficit = {p: max(0.0, self.target[p] - float(self.census.currency.get(p, 0.0)))
                        for p in self.partitions}
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
        """Account a served batch's active minutes, tagged with the bucket that bought it.

        This is also the price WINDOW BOUNDARY, and it is here rather than in a separate
        `end_batch` the driver must remember to call: `charge` is already the once-per-served-
        batch call (it is what increments `pops`), so hanging the flush off it makes
        "one price sample per batch" structural instead of a convention. A driver that
        charges is a driver that prices."""
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
        capped = self.cost.charge(partition, minutes)
        self.cost.end_window()
        return capped

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
                    target={p: round(self.target.get(p, 0.0), 3) for p in self.partitions},
                    ratio={p: self.ratios.get(p) for p in self.partitions},
                    anchor=round(self.anchor, 3),
                    target_rule=TARGET_RULE,
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
    from release_mix import ratios as ratio_table              # noqa: E402
    parts = [p for p in args.partitions.split(",") if p]
    cen = label_currency(parts)
    rt = ratio_table(parts)
    tgt, anchor = currency_targets(cen.currency, parts, rt)
    defc = deficits_from_currency(cen.currency, parts, rt)
    alloc = allocate(defc, {p: SEED_PRICE for p in parts}, parts, args.floor)
    print(json.dumps(dict(currency=cen.summary(), target_rule=TARGET_RULE, anchor=round(anchor, 3),
                          ratio=rt, target={p: round(tgt[p], 2) for p in parts},
                          deficits={p: round(defc[p], 2) for p in parts},
                          allocation_at_seed_prices=alloc.summary()), indent=2))


if __name__ == "__main__":
    main()
