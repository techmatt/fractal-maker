#!/usr/bin/env python
r"""sitting_cutter.py — cut a continuous harvest's record into ONE labelling sitting.

WHAT A SITTING IS. Harvest v1 produced three registered batches and three exports off one
run, and Matt then sat through 870 tiles of which a large minority were things nobody should
ever have been shown: 126 all-label-1 unscreened native multibrots in one cluster, 52
julia:mandelbrot rows that were 17 atoms x 3 rungs of the same look, and an unmeasured number
of >30%-interior frames the rule already says are class 1 by construction. A sitting is the
fix: ONE cut, ONE manifest, ONE export, capped at `MAX_ROWS`, with everything the record
already knows to be worthless removed BEFORE a human sees it.

THE THREE FILTER STAGES ARE NON-OPTIONAL, AND THAT IS THE DESIGN
----------------------------------------------------------------
They are entries in `STAGES`, walked unconditionally by `cut_sitting`. There is no flag to
skip one. That is deliberate and it is not tidiness: every one of them exists because its
absence cost a real sitting real keystrokes, and a filter with an off switch is a filter that
will be off on the run that needed it (`verification_practice.md` §2 — a gate that degrades
to silence cannot protect against the removal of its own input). Each is proved red by
injection in `test_sitting_cutter.py`.

  (a) INTERIOR > 0.30 -> auto-labelled `interior_gt30_v1`, NEVER PRESENTED.
      Matt's rule, dictated 2026-08-01 and firm: a frame more than 30% black is class 1 for
      wallpaper emission, no gray zone. `apply_interior_rule.py` already applies it to the
      label store AFTER a batch is built and seeds the score into the served manifest so the
      rig skips the row. This is the stronger form the sitting is owed: the row never enters
      the served manifest at all. Same rule id, same threshold, same strict `>`, same measure,
      imported from that module rather than restated.

  (b) PRESENTATION-LEVEL MORPH-DEDUP at cos 0.974 — one row per look.
      NEVER A DISCOVERY GATE, and the distinction is the whole point. The discovery record
      keeps every candidate; what is thinned is what gets SHOWN. A near-duplicate is not
      evidence of anything after the first one, and 870 labelled rows collapsed to 367 looks
      (2.37 labels per look) — the sitting's cost is denominated in looks, so the dedup is
      what makes the cap mean something. Best-first, so a look is represented by its
      highest-ranked member.

  (c) PER-PARTITION MACHINE-1 AUTO-DISCARD.
      ON for native multibrot and phoenix, OFF for julia:mandelbrot, because the measurement
      is partition-dependent and the pooled number is not a decision: P(Matt=1 | v10 decoded
      1) is 94-100% in multibrot3/4/5 and 72.0% in phoenix (P(>=3 | decoded 1) = 0/82 there),
      but 30.9% in julia:mandelbrot, where 16.5% of machine-1s are >=3. The per-partition
      table is `supply_routing.MACHINE_1_DISCARD`, imported, and every partition with no
      measurement of its own fails CLOSED to KEEP — spending labels is recoverable, throwing
      away one good picture in six is not.

ORDER IS COST-DESCENDING IN THE OTHER DIRECTION. (a) and (c) are free reads off columns the
record already carries; (b) needs a render and a CLIP pass per surviving row. So the two free
stages run first and the expensive one sees the smallest population. Reversing them would be
correct and would cost a morph field for every row the other two were about to delete.

THE CALIBRATION RESERVATION (Matt, 2026-08-04)
----------------------------------------------
A partition whose LABELLED POSITIVES (human score >= 3, amendment overlay applied) are below
`MIN_POS` is one nothing can be said about: no amount of discovery fixes that, because the
missing thing is human labels. Left to the balanced draw, such a partition gets whatever its
cell count earns — which for a scarce partition is close to nothing, so it stays unmeasurable
forever. So `cut_sitting` RESERVES a slice of each sitting for it (`plan_reservations`,
`draw_reserved`). The floor came from the retired t_good estimator's own sufficiency gate and
this module now owns it; the reservation outlived the sweep because it is a statement about
the label corpus, not about a threshold.

It is a GENERAL RULE, not a `phoenix:classic` special case, and it lapses per-partition by
itself: the qualifying set is recomputed from the live corpus at every cut, so a partition that
crosses MIN_POS stops being reserved without anyone editing a list. Today exactly one qualifies
(`phoenix:classic`, 7 positives); mandelbrot at 626 gets nothing.

Bounded on three sides, because a reservation is time taken from the rest of the sitting:
`RESERVE_FRAC` per qualifying partition, `RESERVE_CAP_FRAC` across all of them together (split
evenly once more than three qualify), and SUPPLY — a reservation is a claim on rows that
survived the three stages, so an unfillable one records its shortfall and the balanced draw
silently fills the slot from elsewhere. It never fails the build: a partition with no candidate
rows this run is exactly the run where refusing to cut a sitting helps nobody.

A SITTING MAY SPAN SEVERAL LEGS (2026-08-05). It is one cut, one cap and one page — not one
batch. `SittingSpec` names N `SittingLeg`s, each its own registered batch; the cut runs ONCE
over their union and the drawn rows are split back to their own batches. See the block above
`SITTINGS` for why a leg is a registration.

  uv run python tools/atlas/sitting_cutter.py dry-run --run-dir data/discovery/<run>
  uv run python tools/atlas/sitting_cutter.py dry-run --sitting steady_state
  uv run python tools/atlas/sitting_cutter.py draw    --sitting steady_state
  uv run python tools/atlas/sitting_cutter.py render  --sitting steady_state
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining",
           ROOT / "tools" / "sourcing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tools import run_record            # noqa: E402  (segments-aware run-record layer)

import apportion                                # noqa: E402  (THE apportionment rules)
import apply_interior_rule as air               # noqa: E402  (the rule id + threshold)
import supply_routing as srt                    # noqa: E402  (the per-partition discard table)
from tools.emission import floors as F          # noqa: E402  THE cut owner (GOOD_FLOOR/NOTBAD_CUT)

MAX_ROWS = 1000              # one sitting


class BudgetExhausted(Exception):
    """A BOUNDED morph pass hit its limit. Raised by an embedder built with a `limit`, and
    counted by `stage_morph_dedup` as `budget_not_reached` — never as `unembeddable`, which
    is a claim about the row rather than about the pass."""


NEAR_DUP_COS = srt.NEAR_DUP_COS
INTERIOR_RULE_ID = air.RULE_ID
INTERIOR_THRESHOLD = air.THRESHOLD

# --- the calibration reservation ------------------------------------------------------- #
RESERVE_FRAC = 0.05          # per qualifying partition, as a fraction of the sitting (~50/1000)
RESERVE_CAP_FRAC = 0.15      # ...and never more than this across ALL of them together
POSITIVE_CLASSES = (3, 4)    # a "positive" is a human keeper — the deriver's own label>=3


# THE sufficiency floor for the calibration reservation. It USED to be imported from
# `tools/scoring/derive_t_good.MIN_POS`, the t_good estimator's own "can this partition be
# calibrated at all" gate, precisely so a second literal 15 could not diverge from production
# (`verification_practice.md` §1.8). That estimator was deleted on 2026-08-09 with the rest of
# the per-partition t_good machinery (prompts/selection_restructure_3.md), and this module
# INHERITED the number rather than losing it: the reservation is about having enough human
# keepers in a partition to say anything about it, which is a question about the LABEL CORPUS
# and outlived the threshold sweep that used to ask it. One owner, still — this one.
MIN_POS = 15


def min_pos() -> int:
    """THE sufficiency floor (see `MIN_POS` above)."""
    return int(MIN_POS)


@functools.lru_cache(maxsize=8)
def _positives_cached(parts: tuple, corpus_dir) -> tuple:
    import pop_quota as pq                                   # noqa: E402 (pure, torch-free)
    cen = pq.label_currency(list(parts), corpus_dir)
    return tuple((p, sum(int(v) for k, v in (cen.counts.get(p) or {}).items()
                         if int(k) in POSITIVE_CLASSES)) for p in parts)


def positives_census(partitions=None, corpus_dir=None) -> dict:
    """partition -> count of HUMAN positives (score >= 3) in the label corpus + library.

    Counted through `pop_quota.label_currency`, which is THE census — same amendment overlay,
    same phoenix split, same default-route rule — rather than a second corpus walk that could
    disagree with the deficit about what a partition holds. It reads a different projection of
    the same counts: the deficit weights 4s ten times a 3, this counts keepers, because the
    sufficiency floor is about POSITIVES and does not weight them.

    Memoized per (partitions, corpus_dir): the corpus does not change inside one cut, and a
    cut calls this once per invocation of a function that is also unit-tested."""
    from partitions import ALL_FAMS                          # noqa: E402 (tools/scoring)
    parts = tuple(partitions if partitions is not None else ALL_FAMS)
    return dict(_positives_cached(parts, corpus_dir))


def plan_reservations(positives: dict, max_rows: int, *, floor: int | None = None,
                      frac: float = RESERVE_FRAC,
                      cap_frac: float = RESERVE_CAP_FRAC) -> dict:
    """{partition: rows reserved} for every partition below the sufficiency floor.

    `frac` of the sitting each, and if more than `cap_frac / frac` partitions qualify the TOTAL
    is held at `cap_frac` and split evenly — so the cap binds by shrinking each share rather
    than by dropping partitions off the end, which would silently pick a favourite among equally
    starved families.

    TRUNCATES rather than rounds (`int`), so the cap is a hard bound: four qualifying partitions
    at 15%/4 = 3.75% of 1000 give 37 rows each (148 total), never 38 each (152, over the cap).

    Returns an entry for every qualifying partition even when its size works out to ZERO — a
    sitting too small to reserve a row is a fact about the sitting, and recording it is what
    distinguishes "nobody qualified" from "the reservation rounded away"."""
    floor = min_pos() if floor is None else int(floor)
    qual = sorted(p for p, n in positives.items() if int(n) < floor)
    if not qual:
        return {}
    per = min(float(frac), float(cap_frac) / len(qual))
    n = int(per * int(max_rows) + 1e-9)
    return {p: n for p in qual}


# =========================================================================== #
# the stages. Each takes (rows, ctx) and returns (kept, removed, report).
# =========================================================================== #
def stage_interior(rows, ctx):
    """(a) Matt's >0.30-interior rule, as an AUTO-LABEL that is never presented.

    A row with NO measure is KEPT and counted apart — an absent measure is not a high one,
    which is `apply_interior_rule.fires`'s own rule and the sourcing gate's. Strict `>`, so a
    frame at exactly 0.30 is shown; the boundary side is invisible in a count, which is why
    it is asserted rather than described."""
    kept, removed, unmeasured = [], [], 0
    for r in rows:
        v = r.get("int_frac")
        if v is None:
            unmeasured += 1
            kept.append(r)
        elif float(v) > INTERIOR_THRESHOLD:
            r = dict(r, auto_label=dict(score=air.RULE_SCORE, labeler=air.LABELER,
                                        rule_id=INTERIOR_RULE_ID, measure="int_frac",
                                        value=float(v), threshold=INTERIOR_THRESHOLD,
                                        comparison="strict >"))
            removed.append(r)
        else:
            kept.append(r)
    return kept, removed, dict(stage="interior_gt30", removed=len(removed),
                               unmeasured_kept=unmeasured, rule_id=INTERIOR_RULE_ID,
                               threshold=INTERIOR_THRESHOLD, comparison="strict >",
                               disposition="auto-labelled class 1, NEVER presented")


def stage_machine_1(rows, ctx):
    """(c) Per-partition machine-1 auto-discard.

    Only a CANONICAL decode counts. A row carrying just a cheap score (`rank_tier=1`) has no
    machine-1 verdict to act on — the cheap score comes off a 384x216 ss1 render and the
    measured P(Matt=1 | decoded 1) rates were all taken against the 640x360 ss2 canonical
    decode. Treating the two as one number is the cap/geometry error, so a tier-1 row is
    never discarded here whatever its partition's flag says."""
    # "MACHINE CLASS 1" IS `canon_nb < floors.NOTBAD_CUT`, read from the raw probability
    # rather than from a stored class. It used to be `canon_decoded == 1`, and that column
    # stopped being able to say 1 on 2026-08-09: it is `floors.good_class` now, which answers
    # None / 3 / 4 (below the good floor there is no class, because the run keeps no verdict
    # about how bad a thing it did not keep is). Reading it would have turned a narrow
    # "the head is confident this is BAD" discard into "everything below the good floor",
    # which is the whole of class 2 as well — a silent widening of a discard rule.
    table = ctx.get("machine_1_discard") or srt.MACHINE_1_DISCARD
    kept, removed = [], []
    no_verdict = Counter()
    for r in rows:
        part = r.get("partition")
        nb = r.get("canon_nb")
        dec = r.get("canon_decoded")
        if int(r.get("rank_tier") or 0) < 2 or (nb is None and dec is None):
            no_verdict[part] += 1
            kept.append(r)
            continue
        is_class_1 = (float(nb) < F.NOTBAD_CUT) if nb is not None else (int(dec) == 1)
        if is_class_1 and table.get(part, False):
            removed.append(dict(r, discard_reason=f"machine_1:{part}"))
        else:
            kept.append(r)
    return kept, removed, dict(stage="machine_1_discard", removed=len(removed),
                               no_canonical_verdict_kept=dict(no_verdict),
                               table={p: bool(v) for p, v in sorted(table.items())},
                               by_partition=dict(Counter(r["partition"] for r in removed)),
                               disposition="discarded from the SITTING; the discovery record "
                                           "keeps them")


def stage_morph_dedup(rows, ctx):
    """(b) Presentation-level morph-dedup: one row per look at cos 0.974, best-first.

    NOT A DISCOVERY GATE. Nothing is removed from the run's record; what is thinned is the
    served page. Greedy leader-radius against the accepted set, in the order the caller hands
    them over — so the caller's ranking is the policy and a look is represented by its
    highest-ranked member, exactly as `supply_routing.thin_by_cspacing` works one layer up.

    `ctx["embed"]` maps a row to an L2-normalized vector (the library morph recipe: 640x360
    ss2 smooth field -> robust-z tanh gray -> CLIP vit_base_patch16_clip_224.openai — the same
    recipe emission clusters at 0.974, so this threshold means what it means everywhere else).
    A row the embedder cannot reach is KEPT and counted, because "we could not measure this"
    is not "this is a duplicate".

    THE EMBED IS NOW COMPUTED ONCE, EVER (`tools/wallpaper/morph_embed_cache.py`). The cost
    below is what a COLD population pays; a re-cut of overlapping material pays ~nothing,
    because a location's morph vector is a pure function of (location, morph recipe, embedder)
    and that triple is the store's key. MEASURED on the harvest-v2 proving population,
    2026-08-03 (3,527 rows surviving (a)+(c); `dry-run --run-dir
    data/discovery/harvest_v2_proving_20260803`): **cold 2,033 s / 3,527 embeds -> warm 3.95 s
    / 0 embeds, 515x**. The store is 12.2 MB for 3,576 vectors. Two notes a future sizing
    decision needs: the cold rate here is **0.576 s/row, not the 0.93 below** — same recipe,
    shallower population (median maxiter 3,000), so size from YOUR population's fw/maxiter
    profile and not from either number; and a fully-warm pass never loads the model, so it
    never checks the embedder digest (see `morph_embed_cache`).

    MEASURED AT FULL SITTING SCALE, 2026-08-03 (`sitting_cutter.py dry-run --run-dir
    data/discovery/q4_long_harvest_20260803 --embed-limit 1000`): **15 m 36 s for 1,000
    embeds**, 587 removed, **413 looks kept** (2.42 rows per look — the harvest-v1 sitting
    measured 2.37, so the knee holds at 2.5x the population). Nothing degraded: 0
    unembeddable, 0 exceptions, the accounting closed.

    Two numbers that change how a live sitting is sized:
      * **0.93 s/row, not 0.26.** A 25-row calibration off the head of the queue said 0.26;
        the queue is tier-sorted and the expensive rows are later, so the prefix sample
        underestimated by 3.6x. `CLAUDE.md`'s run-order rule, hit again.
      * **the cap does not bound this stage.** Dedup runs BEFORE `draw_balanced`, and must —
        the cap is denominated in looks. So a live cut embeds the whole post-(a)-post-(c)
        population, 7,244 rows here, not the 1,000 that reach the page: **~1.9 h**, not the
        15 minutes this bounded run took. `--embed-limit` is a dry-run instrument only.
      * the duplicate rate is NOT a population constant — 30.5% at 400 embeds, 58.7% at
        1,000. A leader-radius accumulates leaders, so a cut sized from a small pilot will
        over-estimate how many looks survive."""
    import numpy as np
    embed = ctx.get("embed")
    if embed is None:
        raise ValueError("stage_morph_dedup needs ctx['embed'] — the dedup is NOT optional, "
                         "so a missing embedder is a hard failure and never a silent skip")
    thr = float(ctx.get("near_dup_cos", NEAR_DUP_COS))
    # A pass that can run for an hour must report on itself WHILE it runs, and must reproject
    # from its own recent throughput rather than restate a pre-run estimate (`CLAUDE.md`,
    # "Projecting a long run's wall clock"): the queue is tier-sorted and the expensive rows
    # are late, so a rate taken over the run-to-date average reads optimistic all the way down.
    import time as _time
    progress = ctx.get("progress")
    every = int(ctx.get("progress_every") or 100)
    t_start = _time.time()
    t_window, n_window = t_start, 0
    acc: list = []
    kept, removed, unembeddable, not_reached = [], [], 0, 0
    reasons: Counter = Counter()
    for i, r in enumerate(rows, 1):
        if progress and i % every == 0:
            now = _time.time()
            recent = (now - t_window) / max(i - n_window, 1)
            progress(dict(seen=i, of=len(rows), looks=len(acc), removed=len(removed),
                          elapsed_s=round(now - t_start, 1),
                          recent_s_per_row=round(recent, 3),
                          eta_min=round((len(rows) - i) * recent / 60.0, 1)))
            t_window, n_window = now, i
        try:
            e = embed(r)
        except BudgetExhausted:
            # A BOUNDED pass (a dry run) reached its limit. Counted APART from a row the
            # embedder could not reach: "we stopped early" and "this row has no field" are
            # different facts, and a run that reported the first as the second would look
            # like a population property. The rows after the bound pass through untouched.
            not_reached += 1
            kept.append(r)
            continue
        except Exception as exc:                             # noqa: BLE001
            # PER-ROW tolerance, NOT a silent one. One bad row must not kill a cut, but the
            # reason is counted and reported — the first dry-run of this stage embedded ZERO
            # of 7,264 rows and reported it as "unembeddable_kept", which reads as a property
            # of the population and was actually one exception repeated 7,264 times.
            reasons[f"{type(exc).__name__}: {str(exc)[:80]}"] += 1
            e = None
        if e is None:
            unembeddable += 1
            if not reasons:
                reasons["embedder returned None"] += 1
            kept.append(r)
            continue
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        e = e / (float(np.linalg.norm(e)) + 1e-9)
        if acc:
            cos = float(np.max(np.stack(acc) @ e))
            if cos >= thr:
                removed.append(dict(r, dup_cos=round(cos, 4)))
                continue
        acc.append(e)
        kept.append(r)
    return kept, removed, dict(stage="morph_dedup", removed=len(removed),
                               unembeddable_kept=unembeddable, threshold=thr,
                               budget_not_reached=not_reached,
                               embedded=len(acc) + len(removed),
                               unembeddable_reasons=dict(reasons.most_common(5)),
                               recipe="library morph CLIP (640x360 ss2 -> robustz_tanh_k2_v1 "
                                      "-> vit_base_patch16_clip_224.openai)",
                               looks_kept=len(acc),
                               disposition="PRESENTATION only — the discovery record keeps "
                                           "every candidate")


# The pipeline. Walked unconditionally; there is no flag that removes an entry.
STAGES = (stage_interior, stage_machine_1, stage_morph_dedup)


# =========================================================================== #
# the cut
# =========================================================================== #
def draw_balanced(rows, cell_of, n: int, preseed: dict | None = None):
    """`n` rows, round-robin over cells, best-first inside each cell — the caller's order is
    the within-cell rank. The allocation is `apportion.deal_round_robin`, the same
    floor-then-remainder rule the v1 batch draw uses; only the CELLS differ here (partition x
    tier rather than fate x partition), because a sitting is one page and a fate-balanced page
    would spend the cap on rejects.

    `preseed` credits a cell with rows it was ALREADY given (the calibration reservation), so
    the round-robin continues from that count instead of restarting at zero. That is what makes
    a reservation a FLOOR rather than a bonus: without it a partition the balanced draw would
    have served generously anyway ends up with its natural share PLUS the reservation, which is
    an over-service nobody asked for. `max(natural, reserved)` is the intent; `natural +
    reserved` is what the naive two-pass draw does, and the two differ by exactly the
    reservation on every partition that did not need one."""
    cells = defaultdict(list)
    for r in rows:
        cells[cell_of(r)].append(r)
    seed = {k: int(v) for k, v in (preseed or {}).items()}
    keys = sorted(set(cells) | set(seed), key=str)
    # A reserved cell with no remaining supply still enters `sizes` at 0 — `deal_round_robin`
    # refuses a preseed cell it cannot see, because a reservation that vanishes from the
    # tie-breaks also vanishes from the accounting.
    take = apportion.deal_round_robin({k: len(cells[k]) for k in keys}, n,
                                      preseed=seed)
    seed = {k: seed.get(k, 0) for k in keys}
    out = []
    for i in range(max(take.values(), default=0)):
        for k in keys:
            if i < take[k]:
                out.append(cells[k][i])
    rep = {str(k): dict(taken=take[k] + seed[k], available=len(cells[k]) + seed[k],
                        reserved=seed[k], drained=take[k] >= len(cells[k])) for k in keys}
    return out, rep


def cell_of(r) -> tuple:
    return (r.get("partition"), int(r.get("rank_tier") or 0))


def draw_reserved(rows, cell_of, n: int, reservations: dict):
    """`draw_balanced`, with the calibration reservations honoured FIRST.

    Each reserved partition's slice is itself drawn by `draw_balanced` over that partition's own
    rows — so a reservation is spread across its tiers and taken best-first inside each, exactly
    like the general draw — and what is left over fills the rest of the sitting. A reservation is
    a FLOOR, not a cap: a reserved partition can still win more rows in the general draw.

    Three ways a reservation is smaller than asked, and each is a separate number in the report
    rather than one "granted" that means all of them: `available` (the partition has fewer
    surviving rows than the reservation), `capped_by_sitting` (the sitting is smaller than the
    reservations), and the plan's own cap. None of them raises — the slot fills from elsewhere
    and the shortfall is recorded (`CLAUDE.md`: no silent caps).

    Returns `(sitting, cells_report, reservation_report)`."""
    if not reservations:
        out, rep = draw_balanced(rows, cell_of, n)
        return out, rep, {}
    taken, reserved_rows, res_rep = set(), [], {}
    preseed: dict = defaultdict(int)
    for p in sorted(reservations):
        want_full = int(reservations[p])
        want = max(0, min(want_full, n - len(reserved_rows)))
        pool = [r for r in rows if r.get("partition") == p]
        got, _ = draw_balanced(pool, cell_of, want)
        reserved_rows += got
        taken |= {id(r) for r in got}
        for r in got:
            preseed[cell_of(r)] += 1
        res_rep[p] = dict(reserved=want_full, granted=len(got),
                          shortfall=want_full - len(got), available=len(pool),
                          capped_by_sitting=want < want_full)
    rest = [r for r in rows if id(r) not in taken]
    fill, cells = draw_balanced(rest, cell_of, n - len(reserved_rows), dict(preseed))
    return reserved_rows + fill, cells, res_rep


# =========================================================================== #
# BUCKET APPORTIONMENT — the correction sheet's draw (2026-08-07)
# =========================================================================== #
@dataclass(frozen=True)
class CorrectionBucket:
    """One named slice of a correction sheet, with a target and a membership rule.

    WHY THIS IS NOT `reservations`. A calibration reservation is a FLOOR under a balanced draw
    that fills everything else; these buckets ARE the sheet — their targets sum to the cap, so
    there is no "everything else" for a balanced draw to fill. Using reservations for this
    would leave the general draw with zero rows to deal and the whole apportionment would read
    as five floors that happened to add up, which is not what makes it correct.

    `pick` is a predicate over a surviving queue row. It is a function rather than a partition
    list because two of the five buckets are not partition slices at all: one is
    source-conditioned (native multibrot, maneuver-sourced only) and one is score-conditioned
    and deliberately cross-partition (the machine-class-4 top slice)."""
    name: str
    target: int
    pick: object                  # row -> bool
    why: str


def draw_buckets(rows, cell_of, n: int, buckets, leg_rank=None, no_pad=None):
    """`n` rows apportioned across named BUCKETS, each drawn best-first, shortfalls recorded
    and re-dealt to the SCARCEST bucket that can still absorb them.

    ORDER OF DRAW IS THE DECLARED ORDER, and a row goes to the FIRST bucket that claims it, so
    the buckets partition the sheet rather than overlapping it. That is what makes the
    per-bucket counts add up to the page a human sits through.

    THE CROSS-PARTITION CLASS-4 SLICE IS DECLARED LAST ON PURPOSE. A class-4 mandelbrot row
    satisfies both the mandelbrot bucket and the class-4 slice, so the declaration order
    decides which one spends it. Last means the slice surfaces the class-4s the partition
    buckets did NOT already take — which is the only way it adds anything, and is also the one
    channel through which abundant julia:multibrot material can legitimately reach the sheet
    (gated on being a machine 4, never as bulk).

    WITHIN A BUCKET the existing `draw_balanced` rule runs unchanged — round-robin over
    (partition x tier) cells, caller's order as the within-cell rank — so a bucket spanning
    three partitions is spread across them and across tiers instead of taking one cell's head.

    BACKFILL IS AN ORDERING, NOT A SEPARATE PASS. `leg_rank` maps a row's leg to a priority;
    each bucket's pool is sorted by it before the draw, so a later leg's rows are reached only
    once the earlier leg's are exhausted IN THAT CELL. That is what "backfilled only where a
    bucket falls short" means operationally, and it is per (bucket, cell) rather than per
    bucket — worth stating, because a bucket can be short in one cell while another cell of
    the same bucket still has first-leg supply.

    THE RE-DEAL IS SCARCEST-FIRST AND BOUNDED BY SUPPLY. An unfilled target is offered to the
    bucket with the least remaining supply first, because the abundant buckets are exactly the
    ones whose material the sheet is not short of. Nothing fills from OUTSIDE the buckets: if
    every bucket is drained the sheet comes back under `n`, short, and says so
    (`CLAUDE.md`: no silent caps — the shortfall is a number, not a quiet truncation).

    `no_pad` IS A SEPARATE RULE FROM `pick`, AND THE REHEARSAL IS WHY. A row it names may
    still fill its own bucket's TARGET; it may never fill someone else's SHORTFALL. Without
    the split, the first bounded end-to-end of this function drew 294 rows into a 50-row
    cross-partition class-4 slice — it held 508 rows against native multibrot's 1,287, so
    "scarcest with supply" picked it, and 244 rows of julia:multibrot bulk padded a page whose
    whole purpose was the three partitions that came up short. Absolute remaining supply is
    the wrong scarcity measure the moment one bucket is cross-partition, and the constraint
    that actually matters ("never pad with julia:multibrot bulk") is a statement about the
    ROWS, not about the bucket that happens to hold them. So it is enforced on the rows, at
    the only place padding happens. The 50 the slice is owed still come from anywhere."""
    rank = leg_rank or {}
    pools: dict = {b.name: [] for b in buckets}
    unclaimed = 0
    for r in rows:
        for b in buckets:
            if b.pick(r):
                pools[b.name].append(r)
                break
        else:
            unclaimed += 1
    for name in pools:
        pools[name].sort(key=lambda r: (rank.get(r.get("_leg"), 99),
                                        int(r.get("queue_rank") or 1 << 30)))

    out, rep, taken_ids = [], {}, set()
    for b in buckets:
        want = min(b.target, n - len(out))
        got, cells = draw_balanced(pools[b.name], cell_of, want)
        out += got
        taken_ids |= {id(r) for r in got}
        rep[b.name] = dict(target=b.target, granted=len(got),
                           shortfall=b.target - len(got), available=len(pools[b.name]),
                           capped_by_sheet=want < b.target, why=b.why,
                           by_leg=dict(Counter(r.get("_leg") for r in got)), cells=cells,
                           redealt_in=0)

    # --- the re-deal: scarcest-first, supply-bounded, never from outside the buckets, and
    #     never from rows `no_pad` names (see the docstring — this is the julia:multibrot rule)
    block = no_pad or (lambda r: False)
    short = n - len(out)
    redeal: list = []
    blocked = 0
    while short > 0:
        left = {}
        for b in buckets:
            avail = [r for r in pools[b.name] if id(r) not in taken_ids]
            left[b.name] = [r for r in avail if not block(r)]
        cand = sorted((nm for nm in left if left[nm]), key=lambda nm: (len(left[nm]), nm))
        if not cand:
            break
        nm = cand[0]
        got, _ = draw_balanced(left[nm], cell_of, min(short, len(left[nm])))
        if not got:
            break
        out += got
        taken_ids |= {id(r) for r in got}
        rep[nm]["granted"] += len(got)
        rep[nm]["redealt_in"] += len(got)
        for r in got:
            rep[nm]["by_leg"][r.get("_leg")] = rep[nm]["by_leg"].get(r.get("_leg"), 0) + 1
        redeal.append(dict(bucket=nm, rows=len(got), pad_eligible_left=len(left[nm])))
        short = n - len(out)
    blocked = sum(1 for r in rows if id(r) not in taken_ids and block(r))

    summary = dict(
        buckets=rep, unclaimed_by_any_bucket=unclaimed,
        drawn=len(out), cap=n, shortfall_total=max(0, n - len(out)),
        redeal=redeal,
        # The rows the pad rule held back. Reported because a sheet that came back short and a
        # sheet that could have been filled from forbidden material are different facts, and
        # only one of them is a supply problem.
        pad_blocked_rows_left=blocked,
        rule=("declared order claims a row once; within a bucket draw_balanced over "
              "(partition x tier); shortfall re-dealt to the SCARCEST bucket with "
              "PAD-ELIGIBLE supply left; a `no_pad` row may fill its own target but never "
              "another bucket's shortfall; NOTHING is drawn from outside the buckets, so a "
              "fully drained set returns a short sheet rather than padding it"))
    return out, summary


def cut_sitting(rows, *, max_rows: int = MAX_ROWS, embed=None,
                machine_1_discard=None, near_dup_cos: float = NEAR_DUP_COS,
                progress=None, positives=None, reservations=None,
                buckets=None, leg_rank=None, no_pad=None) -> dict:
    """Run every stage, then cut to one sitting. Returns the sitting and its full accounting.

    The accounting closes: `n_in == n_sitting + sum(removed per stage) + n_over_cap`. A cut
    that can lose a row without a stage naming it is a cut nobody can audit, which is the
    same identity `steered_frontier._reconcile_batch` enforces per batch.

    THE CALIBRATION RESERVATION IS ON BY DEFAULT and is planned from the LIVE corpus — the same
    reason the three stages have no off switch. `positives` / `reservations` exist to inject a
    census or a plan (tests, and a re-cut reproducing an earlier sitting); passing
    `reservations={}` is the explicit way to cut without one, which is a decision that shows up
    in the report as an empty plan rather than as an absent feature.
    """
    ctx = dict(embed=embed, machine_1_discard=machine_1_discard, near_dup_cos=near_dup_cos,
               progress=progress)
    n_in = len(rows)
    stage_reports, removed_by_stage = [], {}
    cur = list(rows)
    for fn in STAGES:
        cur, removed, rep = fn(cur, ctx)
        stage_reports.append(rep)
        removed_by_stage[rep["stage"]] = removed
    # PLANNED AFTER THE STAGES, DRAWN AGAINST WHAT SURVIVED THEM: a reservation sized against
    # the raw queue would look filled while every one of its rows was about to be auto-labelled
    # or deduped away.
    floor_used = None
    bucket_rep = None
    if buckets:
        # A BUCKETED CUT DOES NOT ALSO RESERVE. The buckets already sum to the cap, so a
        # calibration reservation on top would be a sixth claim on a page with no slack —
        # it would silently displace one of the five targets Matt sized. Recorded as an
        # explicit empty plan rather than an absent feature (`cut_sitting`'s own contract).
        reservations, res_rep = {}, {}
        sitting, bucket_rep = draw_buckets(cur, cell_of, max_rows, buckets, leg_rank,
                                           no_pad=no_pad)
        cells = {k: v["cells"] for k, v in bucket_rep["buckets"].items()}
    else:
        if reservations is None:
            positives = positives_census() if positives is None else positives
            floor_used = min_pos()
            reservations = plan_reservations(positives, max_rows, floor=floor_used)
        sitting, cells, res_rep = draw_reserved(cur, cell_of, max_rows, reservations)
    over_cap = len(cur) - len(sitting)

    total_removed = sum(len(v) for v in removed_by_stage.values())
    assert n_in == len(sitting) + total_removed + over_cap, (
        f"sitting cut does not balance: {n_in} in != {len(sitting)} sitting + "
        f"{total_removed} removed + {over_cap} over cap")

    return dict(
        sitting=sitting,
        auto_labeled=removed_by_stage["interior_gt30"],
        removed=removed_by_stage,
        report=dict(
            n_in=n_in, n_sitting=len(sitting), n_over_cap=over_cap, max_rows=max_rows,
            stages=stage_reports,
            balances=True,
            calibration_reservations=dict(
                min_pos=floor_used,
                positives=(dict(positives) if positives is not None else None),
                frac=RESERVE_FRAC, cap_frac=RESERVE_CAP_FRAC,
                active=res_rep,
                granted_total=sum(v["granted"] for v in res_rep.values()),
                shortfall_total=sum(v["shortfall"] for v in res_rep.values())),
            by_partition=dict(Counter(r.get("partition") for r in sitting)),
            by_tier=dict(Counter(str(r.get("rank_tier")) for r in sitting)),
            by_fate=dict(Counter(r.get("fate") for r in sitting)),
            triggered=sum(1 for r in sitting if r.get("triggered")),
            cells=cells,
            # None on the reservation path; the whole apportionment on the bucketed one.
            buckets=bucket_rep,
        ))


# =========================================================================== #
# the default embedder — the library morph recipe, one field per row
# =========================================================================== #
def morph_key_of(row) -> str:
    """The persistent morph-embed cache key for one queue row.

    Pure and cheap — a ledger-row reshape and a string join, no torch and no render — which
    is what makes a fully-warm dedup pass cost seconds instead of hours."""
    from tools.emission import descriptor as D                # noqa: E402
    from tools.wallpaper import morph_embed_cache as mec      # noqa: E402
    return mec.morph_key(D.location_of(_ledger_row(row)))


def make_embedder(scratch_dir: Path, limit: int | None = None, cache=None):
    """The real embedder: 640x360 ss2 smooth field -> morph gray -> CLIP. Heavy imports are
    lazy so the stage functions stay unit-testable with hand-built vectors.

    `cache` is a `morph_embed_cache.MorphEmbedCache`; when given, the returned embedder is the
    cached one (hit -> reuse, miss -> compute + append). The cache wraps the OUTSIDE of the
    budget check on purpose: `--embed-limit` bounds embed WORK, and a hit is not work, so a
    bounded dry-run over an already-warm population runs to completion rather than reporting a
    budget it never spent.

    `limit` bounds how many rows are actually embedded; beyond it the embedder raises
    `BudgetExhausted`, which the stage counts as `budget_not_reached` — a fact about the PASS,
    kept separate from `unembeddable_kept`, which is a fact about a ROW. That is for a bounded
    dry-run, and the separate count is what stops it being a silent truncation
    (`CLAUDE.md`: no silent caps — log what was dropped)."""
    from tools.emission import descriptor as D                # noqa: E402
    from tools.wallpaper import library_annotate as la        # noqa: E402
    import numpy as np

    state = dict(model=None, tf=None, n=0)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def embed(row):
        # The GPU stack is imported on the FIRST MISS, not at build time: a fully-warm pass
        # never embeds anything, and paying ~7 s of `import torch` to answer 3,500 dict
        # lookups would be most of that pass's wall clock.
        from tools.curation.colored_clip import load_clip, embed_clip   # noqa: E402
        if limit is not None and state["n"] >= limit:
            raise BudgetExhausted(f"morph pass bounded at {limit} rows")
        if state["model"] is None:
            state["model"], state["tf"] = load_clip()
        loc = D.location_of(_ledger_row(row))
        field = la.ensure_field(loc, retain=False, tmp_dir=scratch_dir,
                                cache_root=scratch_dir)
        gray = la.morph_gray_image(field)
        e = embed_clip(state["model"], state["tf"], [gray])[0].astype(np.float32)
        state["n"] += 1
        return e / (float(np.linalg.norm(e)) + 1e-9)

    if cache is None:
        return embed
    from tools.wallpaper import morph_embed_cache as mec      # noqa: E402
    return mec.wrap(embed, cache, morph_key_of)


def _ledger_row(r) -> dict:
    """A record-and-rank row in the shape `emission.descriptor.location_of` expects.

    That function reads a LEDGER row — `family`, the reframed `outcome_*` viewport, and for a
    julia twin the ASSERTED schema tag that says which of `outcome_*` / `julia_*` is the
    viewport and which is the parameter. It is not a corpus render block, and handing it one
    raises `KeyError: 'family'` — which the first dry-run of this stage did, 7,264 times, and
    reported as an unembeddable population.

    THE VIEWPORT IS THE CANDIDATE'S OWN FRAME (`cx`/`cy`/`fw`), NOT THE REFRAMED `outcome_*`,
    because that is the frame `build_q4_harvest_batches._render_block` renders the crop at —
    and a PRESENTATION dedup that measures a different picture from the one it is thinning is
    not thinning looks, it is thinning something else. This read the admitted frame when there
    was one; on the harvest-v2 population 70 rows carry `outcome_*` and 49 of them are a
    genuinely different viewport (a reframe halves `fw`), so 1.4% of the population was being
    deduped on a frame nobody would ever see. Derived from the same fields the render block
    reads, so the two cannot drift apart again (`test_sitting_cutter.py` pins it).

    The julia tag is stamped CAMPAIGN because that is the schema these rows were written in
    (viewport in the position fields, parameter in `julia_c_*`) — asserted rather than
    inferred, as `location_of` requires."""
    import julia_ledger_schema as jls                          # noqa: E402
    cx, cy, fw = r["cx"], r["cy"], r["fw"]
    out = dict(family=r["partition"], outcome_cx=str(cx), outcome_cy=str(cy),
               outcome_fw=float(fw))
    if r.get("julia_c_re") is not None and r["partition"].startswith("julia:"):
        out["julia_c_re"], out["julia_c_im"] = r["julia_c_re"], r["julia_c_im"]
        out[jls.SCHEMA_KEY] = jls.CAMPAIGN
    for k in ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
              "phoenix_zm1_re", "phoenix_zm1_im"):
        if r.get(k) is not None:
            out[k] = r[k]
    return out


# =========================================================================== #
# serving a sitting: ONE registered batch PER LEG, then ONE blind sheet over them
#
# The two layers are not redundant and the split is the corpus contract, not tidiness.
# The BATCH is what the corpus owns: `assign_split`-registered, full provenance, an
# `images.jsonl` every consumer globs, and the only place a label may ever land. The SHEET is
# what the labeler is shown: presentation-only, no `images.jsonl` (a sheet that grew one would
# be unioned into training as a second copy of every row), opaque post-shuffle ids, provenance
# DROPPED rather than nulled, and an apportionment-sequenced order. Both halves already exist
# and are already guarded — `build_combined_label_sheet` holds the sheet rules and
# `test_combined_label_sheet.py` the tripwires over them — so the sitting declares an instance
# in that module's `SPECS` and runs it, rather than growing a second copy that can drift.
#
# ONE SITTING MAY SPAN SEVERAL LEGS, AND A LEG IS A REGISTRATION (2026-08-05)
# --------------------------------------------------------------------------
# A sitting is ONE cut, ONE cap and ONE page. It is not one *batch*: the registry classifies
# a GENERATION METHOD, and a run's crawl residue and that run's own dive are two of those —
# different selection stories, different bias arguments, and the dive's whole readable
# property (top-vs-control) disappears if its rows are pooled into a batch whose registration
# describes something else. So a `SittingSpec` names N `SittingLeg`s; the CUT runs once over
# their union (the cap is denominated in the page a human sits through, not in a leg), and the
# drawn rows are then split back to their own registered batches. The sheet unions them again
# for presentation and sequences (source_batch x family) to +/-1.
#
# Everything above this line is the CODE — the three stages, the reservation, the accounting.
# A spec is instance constants only, exactly as `build_combined_label_sheet.SheetSpec` is.
# =========================================================================== #
SITTING_BATCH = "2026-08-03_v2_sitting_v1"      # pinned to the registrations by a test
GEN_VERSION = "v2_sitting_v1"
PRESENTATION_SEED = 0x5177_0803

# SUPERSAMPLE, AND THIS BATCH DEVIATES FROM THE CORPUS. Every other label-corpus batch renders
# its crops at `build_minibrot_batch.CROP_SS` = 4; this sitting renders at 2, at Matt's call
# (2026-08-03), to buy back roughly 4x the sample count on ~1000 rows x 2 crops. Consequences,
# stated rather than left to be discovered:
#   * `ss` is part of the VERSION-INVARIANT render block, so each crop stays self-describing
#     and rebuildable from its own row — this is a recorded difference, never a silent one.
#   * it is a real batch-level difference from the rest of the corpus. The classifier's deploy
#     transform stretches 1280x720 -> 384x224 bicubic, which absorbs most of an ss4-vs-ss2
#     antialiasing difference, but "most" is not "all" and no one has measured it here.
# The shared constant is NOT edited: the deviation is local to this batch, so a later batch
# that says nothing still gets the corpus default.
SITTING_CROP_SS = 2

# The v2 view screen's columns, as they ride onto a corpus row. Names are the label-seeded /
# supply-crawl block's EXISTING provenance keys, deliberately unrenamed: a screened row here
# and a screened row there are the same measurement on the same frame, so they pool
# (`corpus_common.PROVENANCE_KEYS`, the label-seeded block: "same view frame, same
# composite_v3, same terms — which is the whole reason nothing is renamed").
SCREEN_PROV = {
    "composite": "composite",              # view_screen.composite_v3, the LIVE sort key
    "fit_score": "view_fit",               # view_fit_v1.1's logit — RECORDED, never the order
    "fit_model": "view_fit_model",
    "screen_frame": "screen_frame",
    "screen_policy": "screen_policy",
    "vetoed": "vetoed",
    "size_factor": "size_factor",
    "band_coverage": "band_coverage",
    "band_coverage_q25": "band_coverage_q25",
    "radial_range": "radial_range",
    "radial_rings": "radial_rings",
    "interior_fraction": "interior_fraction",
    "op": "op", "k": "k", "degree": "degree", "period": "period",
    "log10_abs_A": "log10_abs_A", "window_scale": "window_scale",
    "parent_depth": "parent_depth", "atom_id": "atom_id",
    # `atom_key` is DELIBERATELY WITHHELD, and this is the note that says so rather than
    # leaving it to read as an oversight. It is not just a column: `v10/build_manifest.
    # collect_atom_keys` unions groups across every row that records one, and GATE 14's scope
    # argument is "only the six 2026-08 batches record provenance.atom_key, so this pass can
    # only union APPENDED locations". A fresh 1,000-row train batch carrying it silently
    # enlists in that union — and the union's own rationale records that it already pulled 18
    # train-side rows into a group with a uniform-leg EVAL row, so enlisting a new batch is an
    # eval-contamination question, not a serving one. Nobody has re-argued that scope, so this
    # batch does not opt in. `atom_id` still names the atom for analysis; only the union KEY is
    # withheld. Pinned by tools/v10/test_v10_build.py::test_only_the_2026_08_batches_record_an
    # _atom_key, which is what caught it.
}
VIEW_FIT_MODEL = "view_fit_v1.1"

# The bar-readability slice: a served row is readable for the pre-registered +0.1181 delta-AP
# margin iff it carries BOTH scores. Checked on the BUILT provenance, not on the queue, because
# what matters is what survived the cut and the apportionment.
def is_bar_readable(prov: dict) -> bool:
    return (prov.get("fit_model") == VIEW_FIT_MODEL
            and prov.get("fit_score") is not None
            and prov.get("composite") is not None)


# --------------------------------------------------------------------------------------- #
# the instance: which legs, which seed, which id prefix. Everything else is code.
# --------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SittingLeg:
    """One registered batch inside a sitting, and the run whose record it is cut from.

    `run_dir` is repo-relative so a spec is portable across the artifacts resolver.
    `dive_log` names a dive leg's own log: its q4 rows do NOT carry the arm (`mix_source` is
    null on every one), and the arm is the whole readable property of a dive, so it is
    recovered by a CHECKED join (`recover_dive_arms`) rather than left absent."""
    batch_id: str
    run_dir: str
    selection_role: str
    purpose: str
    dive_log: str | None = None


@dataclass(frozen=True)
class SittingSpec:
    name: str
    gen_version: str
    seed: int
    id_prefix: str
    crop_ss: int
    legs: tuple
    max_rows: int = MAX_ROWS
    # A CORRECTION SHEET (2026-08-07). The rows are served pre-labelled with the head's own
    # decode, ordered good -> bad by its continuous score, and the human corrects what it got
    # wrong. `correction=False` is every earlier sitting, byte-identical: blind, shuffled, no
    # suggestion on the row. The invariant that does not bend either way is that a SUGGESTION
    # IS NOT A LABEL — `label.score` stays null and the merge refuses to read the suggestion.
    correction: bool = False
    # BUCKET APPORTIONMENT. Empty => the calibration-reservation + balanced draw, unchanged.
    buckets: tuple = ()
    # ROWS THAT MAY FILL THEIR OWN BUCKET BUT MAY NEVER PAD ANOTHER'S SHORTFALL, and the
    # prose that says why. Same pairing rule as `SheetSpec.row_filter`/`filter_rule`: a
    # constraint on the draw whose reason is not written down is indistinguishable from a bug.
    no_pad: object = None
    no_pad_rule: str = ""
    # THE POPULATION THIS SITTING CUTS FROM, applied to the union queue before any stage.
    # `None` is the whole queue, which is every earlier sitting. Paired with its prose for the
    # same reason as `no_pad`, and for the same reason `SheetSpec` pairs `row_filter`: a
    # subset whose rule is not written down is indistinguishable from a lossy build.
    population: object = None
    population_rule: str = ""

    @property
    def batches(self) -> tuple:
        return tuple(l.batch_id for l in self.legs)

    def leg(self, batch_id: str) -> SittingLeg:
        return next(l for l in self.legs if l.batch_id == batch_id)

    def serve_url(self) -> str:
        """WHERE A CORRECTION SITTING IS SERVED — derived from the legs, never written out.

        A blind sitting is served by a `build_combined_label_sheet.SheetSpec`: a separate
        directory with opaque post-shuffle ids, `provenance` DROPPED, and a route map back to
        the source batches. A CORRECTION sitting cannot use that path and it is not an
        oversight — every property that module exists to enforce is the negation of what a
        correction sheet is. It shows the head's decode where the sheet drops the score
        columns; it is ordered by that decode where the sheet deals a seeded apportionment
        specifically so the head's ranking cannot anchor the labeler. Adding a non-blind mode
        to the blinding module is exactly the drift its own docstring warns about.

        So a correction sitting is served straight off its registered batches by the rig,
        which already has the mode: `wallpaper_label.html` auto-detects CORRECTION from
        `suggested_tier` and honours a builder-stamped contiguous `sheet_order` under
        `order=file`. The legs go in as `?batch=id1,id2`, so the served set IS the leg set by
        construction — which is the property the sheet's route map buys the blind path, bought
        here by there being no second artifact to disagree with.

        Derived from `self.legs` so a leg added to the spec is served without anyone
        remembering to edit a URL (`CLAUDE.md`: derive state in code, freeze it in records)."""
        if not self.correction:
            raise ValueError(f"{self.name} is a BLIND sitting — it is served by a "
                             f"build_combined_label_sheet.SheetSpec over {self.batches}, "
                             f"not by a rig URL")
        return (f"tools/viz/wallpaper_label.html?corpus=label_corpus"
                f"&batch={','.join(self.batches)}&tiers=4&order=file")

    @property
    def leg_rank(self) -> dict:
        """Declaration order as a draw priority — leg 0 is exhausted before leg 1 is reached.
        This is the whole mechanism behind "backfilled only where a bucket falls short"."""
        return {l.batch_id: i for i, l in enumerate(self.legs)}

    def __post_init__(self):
        if bool(self.population) != bool(self.population_rule):
            raise ValueError(f"{self.name}: population and population_rule must be set "
                             f"together (a subset with no stated rule reads as a lossy build)")
        if bool(self.no_pad) != bool(self.no_pad_rule):
            raise ValueError(f"{self.name}: no_pad and no_pad_rule must be set together (a "
                             f"draw constraint with no stated reason reads as a bug)")
        if self.buckets and sum(b.target for b in self.buckets) != self.max_rows:
            raise ValueError(
                f"{self.name}: bucket targets sum to "
                f"{sum(b.target for b in self.buckets)} but the cap is {self.max_rows}. A "
                f"bucketed sheet has no balanced draw to absorb the difference, so targets "
                f"that do not sum to the cap silently under- or over-fill the page.")


V2_SITTING = SittingSpec(
    name="v2_sitting",
    gen_version=GEN_VERSION,
    seed=PRESENTATION_SEED,
    id_prefix="vs",
    crop_ss=SITTING_CROP_SS,
    legs=(SittingLeg(
        batch_id=SITTING_BATCH,
        run_dir="data/discovery/harvest_v2_proving_20260803",
        selection_role="v2_sitting",
        purpose=("The harvest-v2 proving run's ONE labelling sitting. TRAIN-side and BIASED "
                 "more than once: the cheap CORN ordinal decided which candidates earned a "
                 "canonical confirmation, the rank is built from those scores, and part of "
                 "the supply was itself selected on view_screen.composite_v3. No rate "
                 "measured on this batch is a base rate.")),),
)

# --- the steady-state telemetry run's sitting (2026-08-05), TWO legs ---------------------- #
# The crawl leg's ranked record-and-rank residue and that same run's dive, cut ONCE against a
# single 1000-row cap and served as one page. Two registrations because they are two
# generation methods (see `batch_registry`), one cut because a sitting is one human's time.
STEADY_STATE_SITTING = SittingSpec(
    name="steady_state",
    gen_version="steady_state_sitting_v1",
    seed=0x57ED_0805,
    id_prefix="ss",
    crop_ss=SITTING_CROP_SS,          # the same recorded ss2 deviation; see SITTING_CROP_SS
    legs=(
        SittingLeg(
            batch_id="2026-08-05_steady_state_ranked_v1",
            run_dir="data/discovery/steady_state_v1_20260805",
            selection_role="steady_state_ranked",
            purpose=("The 2026-08-05 steady-state telemetry run's CRAWL leg: its whole "
                     "record-and-rank residue, tier-sorted. TRAIN-side and BIASED more than "
                     "once — the cheap v10 ordinal decided which candidates earned a "
                     "canonical confirmation, the rank is built from those scores, and part "
                     "of the supply was selected on view_screen.composite_v3 "
                     "(--maneuver-view-prior). No rate measured on it is a base rate.")),
        SittingLeg(
            batch_id="2026-08-05_steady_state_dive_v1",
            run_dir="data/discovery/steady_state_v1_20260805_dive",
            selection_role="steady_state_dive",
            dive_log="dive_log.jsonl",
            purpose=("The same run's DIVE leg: single-track descent off the crawl's own "
                     "admissions, 7 of 28 planned dives inside a 15-minute active budget. "
                     "TRAIN-side and BIASED at the source and at every rung; the control arm "
                     "is unbiased only WITHIN the admitted set, which is itself screened, so "
                     "it is not an instrument either. `provenance.mix_source` carries the arm "
                     "(`dive:top` / `dive:control`) — the contrast the leg exists to read.")),
    ),
)

# --- the label-collection run's CORRECTION sheet (2026-08-07), <=500 rows, TWO legs -------- #
# Cut from `label_run_20260807`'s ranked residue, with `steady_state_v2_20260807`'s 1,544-row
# residue declared SECOND so it is reached only where a bucket runs short (`SittingSpec.
# leg_rank` -> `draw_buckets`). Both legs are the same generation method — a pop-quota steered
# crawl's record-and-rank residue — so they are two registrations only because they are two
# runs, and the bias argument is identical for both.
NATIVE_MULTIBROT = ("multibrot3", "multibrot4", "multibrot5")


def _maneuver_sourced(r) -> bool:
    """A row the maneuver machinery produced, by the record's own stamp rather than by
    inference. `triggered` is the on-admission trigger; `mix_source` carries the maneuver kind
    for the quota-driven ones. Native multibrot supply is seeds + triggered maneuvers only
    (`supply_routing`: raw unscreened dM-shell draws yield 0/48 at >=2), so a native bucket
    that did not condition on this would be a bucket of draws nobody expects to be good."""
    return bool(r.get("triggered")) or str(r.get("mix_source") or "").startswith("maneuver")


LABEL_RUN_BUCKETS = (
    CorrectionBucket(
        "mandelbrot", 150, lambda r: r.get("partition") == "mandelbrot",
        why="the run's largest currency target (ratio 30) and a FLOORED partition all run — "
            "whatever the floor carry bought is the whole mandelbrot supply there is"),
    CorrectionBucket(
        "julia:mandelbrot", 125, lambda r: r.get("partition") == "julia:mandelbrot",
        why="ratio 25, also floored. Its MACHINE-1s are included by construction: "
            "supply_routing.MACHINE_1_DISCARD is False for this partition because "
            "P(Matt=1 | decoded 1) is 30.9% here against 94-100% on the natives, so "
            "stage_machine_1 never removes them and this bucket sees them"),
    CorrectionBucket(
        "phoenix", 100, lambda r: r.get("partition") == "phoenix",
        why="ratio 20 and the run's deficit-driven engine (58% of intent) — the bucket most "
            "likely to be supply-rich rather than short"),
    CorrectionBucket(
        "native_multibrot_maneuver", 75,
        lambda r: r.get("partition") in NATIVE_MULTIBROT and _maneuver_sourced(r),
        why="natives 5 each, and MANEUVER-SOURCED only: their root supply is seeds + "
            "triggered maneuvers, so an unscreened raw draw is not what this bucket is for"),
    CorrectionBucket(
        "machine_class4_top", 50,
        lambda r: r.get("canon_decoded") is not None and int(r["canon_decoded"]) == 4,
        why="the cross-partition class-4 top slice. DECLARED LAST so it surfaces the class-4s "
            "the four partition buckets did not already spend — including julia:multibrot "
            "ones, which is the only way that material legitimately reaches the page"),
)

LABEL_RUN_SITTING = SittingSpec(
    name="label_run",
    gen_version="label_run_correction_v1",
    seed=0x1AB0_0807,
    id_prefix="lr",
    crop_ss=SITTING_CROP_SS,
    max_rows=500,
    correction=True,
    buckets=LABEL_RUN_BUCKETS,
    population=lambda r: int(r.get("rank_tier") or 0) >= 2,
    population_rule=(
        "THE RANKED RESIDUE: rank_tier >= 2, i.e. rows that earned a CANONICAL decode. This "
        "is what 'residue' already means everywhere else — steady_state_v2's readout reports "
        "its 1,544 rankable rows as 'tier-2 minus admitted', over the fates q3_dup / "
        "canon_not_q3 / reframe_not_q3 — so the union queue's tier-1 rows (below_tau_h, "
        "precanon_dup) were never part of it. It is also what makes this a CORRECTION sheet "
        "at all: a tier-1 row has no canonical verdict to prefill, and the first bounded draw "
        "put 223 such rows on a 500-row page, 45% of it unprefilled. The cheap score is NOT a "
        "substitute — it comes off a 384x216 ss1 render where every canonical rate was "
        "measured at 640x360 ss2, which is the cap/geometry error `stage_machine_1` refuses "
        "for exactly the same reason."),
    no_pad=lambda r: str(r.get("partition") or "").startswith("julia:multibrot"),
    no_pad_rule=(
        "julia:multibrot{3,4,5} rows may fill the cross-partition class-4 slice's own 50, "
        "and may never pad another bucket's shortfall. They are the cheapest partitions to "
        "mine and the most abundant in both legs' residue (2,600 of the union's 5,718 rows "
        "at rehearsal), so 'fill from the next-scarcest bucket' measured on absolute supply "
        "reaches them almost immediately — the first bounded end-to-end put 244 of them onto "
        "a page whose three short buckets were mandelbrot, julia:mandelbrot and phoenix. "
        "Matt, prompt-label-run.md Part 3: 'never pad with julia:multibrot bulk'."),
    legs=(
        SittingLeg(
            batch_id="2026-08-07_label_run_correction_v1",
            run_dir="data/discovery/label_run_20260807",
            selection_role="label_run_ranked",
            purpose=("The 2026-08-07 LABEL-COLLECTION run's record-and-rank residue. The run "
                     "allocated against an EXPLICIT run-scoped currency-target vector "
                     "(--currency-targets), not the standing release mix, so its partition "
                     "mix is a per-run instrument and no share of it is a policy statement. "
                     "TRAIN-side and BIASED more than once — the cheap v10 ordinal decided "
                     "which candidates earned a canonical confirmation, the rank is built "
                     "from those scores, and part of the supply was selected on "
                     "view_screen.composite_v3 (--maneuver-view-prior). Served as a "
                     "CORRECTION sheet, so every row also carries the head's own decode as a "
                     "suggestion: no rate measured on it is a base rate, and the labels are "
                     "anchored to the head as well as selected by it.")),
        SittingLeg(
            batch_id="2026-08-07_steady_state_v2_backfill_v1",
            run_dir="data/discovery/steady_state_v2_20260807",
            selection_role="steady_state_v2_backfill",
            purpose=("BACKFILL ONLY. steady_state_v2_20260807's 1,544-row ranked residue, "
                     "declared second so `draw_buckets` reaches it only in a (bucket, cell) "
                     "where the label run itself ran short. Same generation method and the "
                     "same bias argument as the first leg; it is a separate registration "
                     "because it is a separate run, and its per-bucket contribution is "
                     "recorded in `sitting_cut.buckets[*].by_leg` rather than inferred.")),
    ),
)

SITTINGS = {s.name: s for s in (V2_SITTING, STEADY_STATE_SITTING, LABEL_RUN_SITTING)}


def _spec(args) -> SittingSpec:
    """The sitting this invocation is about. Named, never positional."""
    return SITTINGS[getattr(args, "sitting", None) or V2_SITTING.name]


def recover_dive_arms(store: Path, dive_log: Path) -> tuple[dict, str]:
    """{root_id: 'dive:top'|'dive:control'} for a dive leg, or ({}, why) if the join cannot be
    VERIFIED.

    A dive's q4 rows carry no arm: `_q4_record` stamps `mix_source`/`dive_id` on the OUTCOME
    ledger row, and a q4 candidate row is written on a different path, so all 276 rows here
    have `mix_source = null`. What they do carry is `root_id`, and `one_dive` mints exactly one
    fresh root node id per dive and appends exactly one `dive_log.jsonl` record per dive, both
    in execution order. So the i-th distinct root_id in APPEND order is the i-th dive.

    IT READS THE STORE ITSELF rather than taking rows, and that is the whole safety of it: the
    argument is about APPEND order, and `build_sorted_queue` returns the same rows tier-SORTED
    (its name now says so). Handed
    the sorted rows this returned no mapping at all (the partition check caught it), which is
    the failure working — but a signature that can be handed the wrong order is a signature
    that will be.

    AN ORDER ARGUMENT IS NOT A JOIN KEY, so it is CHECKED rather than trusted: the counts must
    match exactly, and every row under a root_id must carry the partition its dive_log record
    names. A mismatch returns no mapping and the reason — the arm goes null, which is
    recoverable, where a wrong arm would silently invert the contrast the leg exists to
    measure."""
    def _rd(p):
        # run_record: `store` is q4_candidates.jsonl, which rotates into .jsonl.gz segments —
        # and APPEND ORDER across segments is exactly what the join argument below rests on.
        return run_record.require_rows(p)
    rows, log = _rd(store), _rd(dive_log)
    seen: list = []
    for r in rows:
        rid = r.get("root_id")
        if rid is not None and rid not in seen:
            seen.append(rid)
    if len(seen) != len(log):
        return {}, (f"{len(seen)} distinct root_id vs {len(log)} dive_log records — "
                    f"the order argument does not hold, arm left null")
    by_root = defaultdict(set)
    for r in rows:
        by_root[r.get("root_id")].add(r.get("partition"))
    for rid, rec in zip(seen, log):
        if by_root[rid] != {rec["partition"]}:
            return {}, (f"root_id {rid} spans partitions {sorted(by_root[rid])} but its dive "
                        f"records {rec['partition']!r} — arm left null")
    return ({rid: f"dive:{rec['start_group']}" for rid, rec in zip(seen, log)},
            f"{len(log)} dives joined on root_id order, partition-checked")


def load_union_queue(spec: SittingSpec) -> tuple[list[dict], dict]:
    """The sitting's queue: every leg's record-and-rank store, unioned and tier-sorted ONCE.

    The per-leg load, the row identity and the sort are the v1 batch builder's own
    (`build_q4_harvest_batches`), imported rather than re-derived, so the sitting and the batch
    draw can never disagree about what the queue IS.

    CROSS-LEG DEDUP IS NOT A FORMALITY HERE: a dive descends from the crawl's admissions, so
    the two stores can record the same place. First occurrence wins in LEG ORDER (the crawl is
    declared first, so a place both legs saw is attributed to the one that found it), and the
    count is reported rather than absorbed.

    `queue_rank` is assigned over the UNION, because that is the order the draw reads
    best-first inside a cell — a per-leg rank would make a dive row's 3rd-best beat a crawl
    row's best."""
    import build_q4_harvest_batches as bq
    rows, seen, dropped, per_leg = [], set(), Counter(), Counter()
    arms = {}
    for leg in spec.legs:
        run_dir = ROOT / leg.run_dir
        leg_rows, _rep = bq.build_sorted_queue(run_dir)
        if leg.dive_log:
            m, why = recover_dive_arms(run_dir / "q4_candidates.jsonl",
                                       run_dir / leg.dive_log)
            arms[leg.batch_id] = why
            for r in leg_rows:
                if r.get("mix_source") is None and r.get("root_id") in m:
                    r["mix_source"] = m[r["root_id"]]
        for r in leg_rows:
            key = bq.queue_identity(r)
            if key in seen:
                dropped[leg.batch_id] += 1
                continue
            seen.add(key)
            r["_leg"] = leg.batch_id
            rows.append(r)
            per_leg[leg.batch_id] += 1
    rows.sort(key=bq.queue_sort_key)
    for i, r in enumerate(rows, 1):
        r["queue_rank"] = i
    rep = dict(n=len(rows), by_leg=dict(per_leg),
               cross_leg_duplicates_dropped=dict(dropped),
               dive_arm_join=arms,
               by_tier=dict(Counter(str(r.get("rank_tier")) for r in rows)),
               by_fate=dict(Counter(r.get("fate") for r in rows)),
               by_partition=dict(Counter(r.get("partition") for r in rows)))
    return rows, rep


def apply_population(rows: list[dict], spec: SittingSpec) -> tuple[list, dict]:
    """Narrow the union queue to the population the SPEC cuts from, and report the narrowing.

    Applied BEFORE the three stages, so what they report removing is denominated in the
    population rather than in a queue half of which was never a candidate. `None` filter =>
    the whole queue, byte-identical to every sitting before this field existed.

    The counts are kept per tier and per fate because "5,946 -> 1,900" on its own reads as a
    cut that lost most of its material; what it actually is here is the difference between the
    record-and-rank QUEUE and the ranked RESIDUE, and only the by-fate split says so."""
    if spec is None or spec.population is None:
        return rows, dict(rule=None, kept=len(rows), dropped=0)
    keep = [r for r in rows if spec.population(r)]
    drop = [r for r in rows if not spec.population(r)]
    return keep, dict(
        rule=spec.population_rule, kept=len(keep), dropped=len(drop),
        dropped_by_tier=dict(Counter(str(r.get("rank_tier")) for r in drop)),
        dropped_by_fate=dict(Counter(r.get("fate") for r in drop)),
        kept_by_tier=dict(Counter(str(r.get("rank_tier")) for r in keep)),
        kept_by_fate=dict(Counter(r.get("fate") for r in keep)))


def load_queue(run_dir: Path) -> list[dict]:
    """One run's record-and-rank store, tier-sorted, first-occurrence-wins on identity."""
    import build_q4_harvest_batches as bq
    rows, _rep = bq.build_sorted_queue(Path(run_dir))
    return rows


def _provenance(r: dict, cc, spec: SittingSpec, leg: SittingLeg) -> dict:
    """One served row's full selection trail. The classifier never sees any of it."""
    fields = {k: r.get(k) for k in (
        "fate", "rank_tier", "rank_score", "queue_rank", "cheap_eord", "cheap_pgood",
        "canon_eord", "canon_pgood", "canon_decoded", "reframe_decoded", "triggered",
        "mix_source", "int_frac", "occ", "tau_h", "tau_rec", "t_good", "scorer_version",
        "depth", "branch") if r.get(k) is not None}
    man = r.get("maneuver")
    if isinstance(man, dict):
        for prov_key, man_key in SCREEN_PROV.items():
            v = man.get(man_key)
            if v is not None:
                fields[prov_key] = v
    return cc.provenance_block(spec.gen_version, leg.batch_id,
                               family=r.get("partition"),
                               selection_role=leg.selection_role,
                               stratum=str(cell_of(r)), **fields)


def head_pred(r: dict) -> float | None:
    """The head's CONTINUOUS readout for one row, on the 1..K tier scale.

    `canon_eord` is `model.score_from_logits` = sum of the CORN marginals in [0, K-1]; the
    served `pred` is `1 + eord` so it is denominated in the same units as the tier a human is
    about to press, which is what makes a sorted correction sheet readable as a quality ramp.
    Same convention as the mining sheet (`build_mining_sheet`), not a second one.

    None when the row never earned a canonical decode. Such a row cannot be pre-labelled, so
    it must not be ORDERED as if it had been either — `stamp_correction` sorts it last rather
    than letting a defaulted 0 put it at the good end of the page."""
    e = r.get("canon_eord")
    return (1.0 + float(e)) if e is not None else None


def suggested_tier(r: dict) -> int | None:
    """The head's own decode, served as the PREFILLED suggestion.

    Taken straight from the record — no quantile mapping, unlike the mining sheet, and the
    difference is that this head already emits a class. `reframe_decoded` wins when present:
    the reframe is the decode of the frame the row would actually be SHOWN in, so it is the
    verdict that matches the crop under the suggestion.

    A SUGGESTION IS NOT A LABEL. `label.score` stays null on every row; the rig holds the
    suggestion in a separate map and exports only what Matt actually pressed."""
    for k in ("reframe_decoded", "canon_decoded"):
        v = r.get(k)
        if v is not None:
            return int(v)
    # Below `floors.GOOD_FLOOR` the class column is null (see `machine_1_discard`), and a row
    # that DID earn a canonical verdict still deserves a prefill rather than being served
    # unsuggested. Same two natural cutpoints the class column uses above the floor: 2 if the
    # head calls it not-bad, else 1. This reproduces exactly what the retired hard-class
    # decode returned for these rows.
    nb = r.get("canon_nb")
    if nb is not None and r.get("canon_pgood") is not None:
        return 2 if float(nb) >= F.NOTBAD_CUT else 1
    return None


def stamp_correction(rows: list[dict]) -> dict:
    """Add `pred` / `suggested_tier` / `sheet_order` to a correction sheet's served rows.

    ORDER IS GOOD -> BAD on the continuous score, descending, ties on `image_id`, and it is
    stamped in the FILE rather than derived in the browser: `wallpaper_label.html?order=file`
    honours a contiguous `sheet_order` and derives nothing, so the order is reproducible from
    the artifact and auditable later instead of being a sort the page happens to perform. Two
    owners for one decision is the failure that convention avoids.

    Rows with no canonical decode sort LAST and carry `suggested_tier: null`. The rig shows
    them unprefilled, which is the honest presentation — a row the head never judged has no
    suggestion to correct.

    These three keys sit beside `render`/`provenance`/`label` rather than inside them: the
    classifier reads only `render` (CLAUDE.md, the label-corpus contract), so a presentation
    column here cannot reach training, and `provenance` is dropped wholesale by any blind
    sheet — which a correction sheet is not, but the next sitting off these batches may be."""
    for r in rows:
        r["pred"] = r.pop("_pred", None)
        r["suggested_tier"] = r.pop("_sugg", None)
    rows.sort(key=lambda r: (r["pred"] is None,
                             -(r["pred"] if r["pred"] is not None else 0.0),
                             r["image_id"]))
    for i, r in enumerate(rows):
        r["sheet_order"] = i
    assert [r["sheet_order"] for r in rows] == list(range(len(rows))), "sheet_order not contiguous"
    have = [r for r in rows if r["pred"] is not None]
    return dict(
        order="sheet_order — DESCENDING pred (good -> bad), ties on image_id; rows with no "
              "canonical decode sort last and carry no suggestion",
        pred="1 + canon_eord (sum of the v10 CORN marginals), the 1..K tier scale",
        suggestion="reframe_decoded if present else canon_decoded — the head's own class, "
                   "NOT a quantile mapping. label.score stays null on every row.",
        served_via=("tools/viz/wallpaper_label.html?corpus=label_corpus&batch=<ids>&tiers=4"
                    "&order=file — CORRECTION mode is auto-detected from suggested_tier"),
        n_with_pred=len(have), n_without_pred=len(rows) - len(have),
        pred_range=([round(have[0]["pred"], 4), round(have[-1]["pred"], 4)] if have else None),
        suggested_tier_hist=dict(sorted(Counter(
            r["suggested_tier"] for r in rows).items(), key=lambda kv: str(kv[0]))))


def check_registrations(spec: SittingSpec) -> dict:
    """EVERY leg registered EXPLICITLY, before anything is built.

    The fail-closed default is safe (train/biased) but it is not a decision: a leg built under
    it records that nobody classified its generation method, and the sitting would land it
    train-side silently. Checked for all legs up front rather than per leg as it is written,
    so an unregistered second leg cannot be discovered after the first one is on disk."""
    from tools.v7 import build_manifest as bm
    out = {}
    for leg in spec.legs:
        split, biased, source = bm.assign_split({"batch": leg.batch_id, "ft": "mandelbrot"})
        if source == "unregistered":
            raise SystemExit(
                f"{leg.batch_id} is NOT registered in tools/scoring/batch_registry. "
                f"Register it BEFORE building — the fail-closed default lands it train-side "
                f"silently, which records 'nobody thought about this batch'.")
        contra = bm.registration_contradictions([{"batch": leg.batch_id, "biased": biased}])
        if contra:
            raise SystemExit(f"registration contradiction for {leg.batch_id}: {contra}")
        out[leg.batch_id] = [split, biased, source]
    return out


def completeness_stamp(embed_limit) -> dict:
    """The `INCOMPLETE` / `embed_limit` pair a cut records about itself.

    A pure function of the bound, shared by the writer and the test, so the stamp cannot drift
    from what actually happened — the hazard `CLAUDE.md` states as "derive state in code;
    freeze it in records" (a hardcoded `True` is how a metadata file outlives what it records).
    `INCOMPLETE` is true iff the morph pass was bounded, because that is the ONE thing that
    makes the written batches partial: the dedup runs on whatever the embedder reached."""
    n = int(embed_limit) if embed_limit else 0
    return dict(INCOMPLETE=n > 0, embed_limit=(n or None))


def stage_draw(args) -> int:
    """Cut the sitting's union queue ONCE and write each leg as its own registered batch.

    Nothing is rendered here and no sheet is built: the cut has to be readable — and its
    bar-readability slice reported — BEFORE hours of rendering are committed to it.

    `--embed-limit` IS THE CHEAP END-TO-END, and it exists because there was none. `dry-run`
    could be bounded and `draw` could not, so the first execution of any change to this path
    was the production run itself: a 400-line refactor's first run was 13.9 minutes long and
    carried a join bug (`recover_dive_arms` handed a tier-SORTED queue where its contract is
    append order) that a 20-row draw in ~15 s would have surfaced before any of it ran.

    A BOUNDED DRAW WRITES REAL BATCH FILES, so it must be impossible to mistake for a real
    cut: every leg's `batch.json` carries `sitting_cut.INCOMPLETE = true` plus the limit that
    produced it, and the stage prints the same in its closing lines. The stamp is DERIVED from
    the argument at the write site, never hardcoded (`CLAUDE.md`: derive state in code, freeze
    it in records) — an unbounded draw stamps `false` and means it."""
    import time
    import paths
    import corpus_common as cc
    import build_q4_harvest_batches as bq
    import build_minibrot_batch as BMB
    import numpy as np
    from tools.wallpaper import morph_embed_cache as mec

    spec = _spec(args)
    cc.set_below_normal_priority()
    reg = check_registrations(spec)
    max_rows = getattr(args, "max_rows", None) or spec.max_rows

    rows, qrep = load_union_queue(spec)
    rows, qrep["population"] = apply_population(rows, spec)
    print(f"queue: {qrep['n']} rows over {len(spec.legs)} leg(s) {json.dumps(qrep['by_leg'])}"
          f"; cross-leg dups dropped {json.dumps(qrep['cross_leg_duplicates_dropped'])}")
    for b, why in qrep["dive_arm_join"].items():
        print(f"  dive arm ({b}): {why}")
    embed_limit = getattr(args, "embed_limit", None)
    if embed_limit:
        print(f"!! BOUNDED DRAW: --embed-limit {embed_limit}. The morph pass stops there, so "
              f"the dedup is PARTIAL and the batches written below are a smoke test, not a "
              f"sitting. Every batch.json will say so (sitting_cut.INCOMPLETE = true).")
    cache = mec.MorphEmbedCache().open()
    t0 = time.time()
    res = cut_sitting(rows, max_rows=max_rows,
                      embed=make_embedder(Path(paths.scratch("sitting_cutter", "fields")),
                                          embed_limit, cache),
                      progress=lambda d: print(json.dumps(d), flush=True),
                      buckets=(spec.buckets or None), leg_rank=spec.leg_rank,
                      no_pad=spec.no_pad)
    cut_wall = time.time() - t0
    cache_rep = cache.report()
    cache.close()

    sitting = res["sitting"]
    # Opaque ids assigned POST-shuffle over the drawn set — over the WHOLE sitting, not per
    # leg, so an id encodes nothing about which leg a row came from; the hash makes an id a
    # stable function of the row, so a rebuild reproduces it.
    order = list(range(len(sitting)))
    np.random.default_rng(spec.seed ^ BMB._stable_seed(spec.name)).shuffle(order)
    for slot, oi in enumerate(order):
        h = BMB._stable_seed(json.dumps([sitting[oi].get("cx"), sitting[oi].get("cy"),
                                         sitting[oi].get("fw"), sitting[oi].get("julia_c_re"),
                                         sitting[oi].get("phoenix_c_re")], default=str))
        sitting[oi]["image_id"] = f"{spec.id_prefix}{slot:04d}_{h:08x}"
    sitting.sort(key=lambda r: r["image_id"])

    bq._PHOENIX_POOL_CACHE.update(bq._phoenix_points())
    names = BMB._palette_names()
    by_leg: dict = defaultdict(list)
    built: list = []
    for r in sitting:
        leg = spec.leg(r.get("_leg") or spec.legs[0].batch_id)
        r["_palette"] = names[BMB._stable_seed(r["image_id"]) % len(names)]
        render = bq._render_block(r)
        render["ss"] = spec.crop_ss         # the recorded deviation; see SITTING_CROP_SS
        row = cc.make_row(r["image_id"], render, _provenance(r, cc, spec, leg),
                          cc.label_block())
        if spec.correction:
            row["_pred"], row["_sugg"] = head_pred(r), suggested_tier(r)
        built.append(row)
        by_leg[leg.batch_id].append(row)

    # The sheet spans every leg, so `sheet_order` is stamped over the UNION and is contiguous
    # across the batches the rig loads together (`?batch=id1,id2`). A per-leg stamp would
    # restart at 0 in the second file and the page would interleave two ramps.
    correction_rep = stamp_correction(built) if spec.correction else None
    if correction_rep:
        print(f"[correction] {json.dumps(correction_rep)}", flush=True)

    cut = res["report"]
    # The cut is ONE event over the union, so every leg's batch.json carries the SAME cut
    # record. A leg-local slice of it would read as "this is what the cut did to me", which is
    # false — the cap, the dedup and the reservation are all denominated in the whole page.
    sitting_cut = dict(
        sitting=spec.name, legs=list(spec.batches),
        run_dirs={l.batch_id: l.run_dir for l in spec.legs},
        # DERIVED from the flag, so a bounded smoke-test cut can never be read as a real one.
        # `embed_limit` is the whole story: the morph pass stopped there, so the dedup that
        # follows it saw only part of the population.
        **completeness_stamp(embed_limit),
        max_rows=max_rows, queue=qrep,
        n_in=cut["n_in"], n_sitting=cut["n_sitting"], n_over_cap=cut["n_over_cap"],
        stages=cut["stages"], cells=cut["cells"], balances=cut["balances"],
        # THE BUCKET APPORTIONMENT — targets, what each bucket actually got, every shortfall
        # and every re-deal, plus the per-bucket leg split that says how much of it is
        # backfill. None on a reservation-drawn sitting.
        buckets=cut["buckets"],
        correction=correction_rep,
        # WHICH RESERVATIONS WERE ACTIVE AND WHAT EACH ACTUALLY GOT. A reservation that
        # went unfilled for lack of supply records its shortfall here; the sitting was
        # filled from elsewhere, so the only trace it ever existed is this block.
        calibration_reservations=cut["calibration_reservations"],
        cut_wall_s=round(cut_wall, 1), morph_cache=cache_rep,
        by_leg={b: len(v) for b, v in sorted(by_leg.items())},
        auto_labeled_never_presented=[
            dict(cx=r["cx"], cy=r["cy"], fw=r["fw"], partition=r["partition"],
                 leg=r.get("_leg"), **r["auto_label"]) for r in res["auto_labeled"]],
    )

    written = []
    for leg in spec.legs:
        full = by_leg.get(leg.batch_id, [])
        readable = [r for r in full if is_bar_readable(r["provenance"])]
        bdir = Path(cc.batch_dir(leg.batch_id))
        bdir.mkdir(parents=True, exist_ok=True)
        bj = dict(
            schema_version=1, batch_id=leg.batch_id, generator_version=spec.gen_version,
            created=None, labeler=None,
            presentation_seed=spec.seed,
            vivid_companion=BMB.VIVID_PALETTE,
            served_manifest=None,
            served_via=(
                # DERIVED from the spec, so the two serving paths cannot be described by the
                # wrong one: a correction sitting has no blind sheet to point at.
                f"THIS DIRECTORY, as a CORRECTION sheet: {spec.serve_url()} . Rows carry the "
                f"head's own decode as `suggested_tier` and a contiguous `sheet_order` "
                f"(descending `pred`, good -> bad); `label.score` is null on every one and "
                f"the merge refuses to read the suggestion."
                if spec.correction else
                f"a PRESENTATION SHEET, not this directory: "
                f"build_combined_label_sheet.py --spec {spec.name}. This batch holds "
                f"the rows and the provenance; the sheet holds the blind order the "
                f"labeler sees, and the single export routes back here."),
            queued_for_labeling=False,
            purpose=leg.purpose,
            counts=dict(total=len(full),
                        by_partition=dict(Counter(r["provenance"]["family"] for r in full)),
                        by_tier=dict(Counter(str(r["provenance"].get("rank_tier"))
                                             for r in full)),
                        bar_readable=len(readable)),
            registration=dict(assign_split=reg[leg.batch_id],
                              registered_explicitly=True,
                              NOTE="registered in tools/scoring/batch_registry BEFORE the cut"),
            render_defaults=dict(width=bq.CROP_W, height=bq.CROP_H, ss=spec.crop_ss,
                                 ss_deviates_from_corpus_default=dict(
                                     corpus_default=bq.CROP_SS, this_batch=spec.crop_ss,
                                     why="Matt's call 2026-08-03 — ~4x fewer samples over "
                                         "~1000 rows x 2 crops; see sitting_cutter."
                                         "SITTING_CROP_SS for what it costs"),
                                 filter=bq.CROP_FILTER, interior_mode=bq.INTERIOR_MODE,
                                 composition=bq.COMPOSITION,
                                 palette_roster="data/palettes/score3_colormaps.json",
                                 vivid_companion=BMB.VIVID_PALETTE,
                                 maxiter="deep_center_finder._maxiter_for_fw(fw)"),
            render_recipe=cc.render_recipe_stamp(bq.PALETTE_SOURCE),
            sitting_cut=sitting_cut,
            bar_readability=dict(
                n=len(readable), of=len(full),
                definition=("served rows carrying BOTH view_fit_v1.1 (provenance.fit_score / "
                            "fit_model) and composite_v3 (provenance.composite) — the slice "
                            "the pre-registered +0.1181 delta-AP margin reads on"),
                by_partition=dict(Counter(r["provenance"]["family"] for r in readable))),
            calibration_aids="NONE — no exemplars, no reference strip, no score shown",
        )
        cc.write_jsonl(full, str(bdir / "images.jsonl"))
        (bdir / "batch.json").write_text(json.dumps(bj, indent=2, default=str) + "\n",
                                         encoding="utf-8")
        if not (bdir / "scores.json").exists():
            (bdir / "scores.json").write_text("{}", encoding="utf-8")
        written.append((leg.batch_id, len(full), len(readable), bdir))

    print(f"\ncut: {cut['n_in']} in -> {cut['n_sitting']} sitting "
          f"(+{cut['n_over_cap']} over cap); wall {cut_wall/60:.1f} min")
    for s in cut["stages"]:
        print(f"  {s['stage']:18s} removed {s['removed']:5d}  "
              + json.dumps({k: v for k, v in s.items()
                            if k in ("looks_kept", "unembeddable_kept",
                                     "no_canonical_verdict_kept", "unmeasured_kept")}))
    print(f"  morph cache: {json.dumps(cache_rep)}")
    cr = cut["calibration_reservations"]
    print(f"  calibration reservations (< {cr['min_pos']} positives): "
          + (json.dumps(cr["active"]) if cr["active"]
             else "NONE — every partition is calibratable"))
    print(f"  by partition: {json.dumps(cut['by_partition'])}")
    print(f"  by tier:      {json.dumps(cut['by_tier'])}")
    for b, n, nr, bdir in written:
        print(f"  {b:38s} n={n:4d}  bar-readable={nr:4d}  assign_split={reg[b]}  -> {bdir}")
    if embed_limit:
        print(f"\n!! INCOMPLETE CUT — the morph pass was bounded at {embed_limit} embeds and "
              f"every batch.json above is stamped sitting_cut.INCOMPLETE = true. Re-run "
              f"`draw` with no --embed-limit before rendering or labelling any of it.")
    print("\n(NOTHING RENDERED — run `render` next)")
    return 0


def _render_one(job):
    """Both crops for one row. Atomic per file: render to a partial beside the target and
    rename, so a kill mid-render can never leave a TRUNCATED jpg that reads as done forever.

    THE PARTIAL MUST STILL END IN `.jpg`. The engine infers the image format from the output
    extension, so the obvious `<id>.jpg.tmp` is not a slower path or a warning — it is a hard
    `failed to write ...: The file extension ."tmp" was not recognized as an image format`,
    i.e. a 100% failure rate that looks exactly like a broken renderer. It cost 50 renders to
    find. `<id>.part.jpg` is the same atomicity with an extension the engine can write, and it
    cannot be mistaken for a finished crop because every reader (`needs`, the completeness
    count, the sheet's route walk) addresses crops by exact `<image_id>.jpg`."""
    import corpus_common as cc
    import build_q4_harvest_batches as bq
    import build_minibrot_batch as BMB
    row, crops, vivid, timeout = job
    iid, render = row["image_id"], row["render"]
    for out, pal, src in ((crops / f"{iid}.jpg", render["palette"], bq.PALETTE_SOURCE),
                          (vivid / f"{iid}.jpg", BMB.VIVID_PALETTE, BMB.VIVID_SOURCE)):
        if out.exists():
            continue
        tmp = out.with_name(f"{out.stem}.part.jpg")
        try:
            cc.render_corpus_crop(dict(render, palette=pal), str(tmp), palette_source=src,
                                  timeout=timeout, threads=bq.RENDER_THREADS)
            os.replace(tmp, out)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return iid


def stage_render(args) -> int:
    """Render both crops for every row of every leg. IDEMPOTENT — an existing crop is skipped,
    so a kill and a relaunch resume exactly where the last one stopped, and the per-unit
    checkpoint is the crop file itself (there is no separate progress file to fall out of sync
    with disk).

    The legs render in ONE pass with ONE deadline, because a sitting is not servable until
    every one of its legs is: a per-leg time bound would finish leg 1 and leave the page
    unbuildable, which reads as progress and is not."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import corpus_common as cc
    import build_q4_harvest_batches as bq

    spec = _spec(args)
    cc.set_below_normal_priority()
    jobs, per_leg = [], {}
    for leg in spec.legs:
        bdir = Path(cc.batch_dir(leg.batch_id))
        if not (bdir / "images.jsonl").exists():
            raise SystemExit(f"{bdir/'images.jsonl'} missing — run `draw` first.")
        rows = cc.read_jsonl(str(bdir / "images.jsonl"))
        crops = Path(cc.crops_dir(leg.batch_id))
        vivid = Path(cc.vivid_dir(leg.batch_id))
        crops.mkdir(parents=True, exist_ok=True)
        vivid.mkdir(parents=True, exist_ok=True)
        per_leg[leg.batch_id] = (rows, crops, vivid)
        jobs += [(r, crops, vivid) for r in rows]

    def needs(r, crops, vivid):
        return not (crops / f"{r['image_id']}.jpg").exists() or \
            not (vivid / f"{r['image_id']}.jpg").exists()

    todo = [j for j in jobs if needs(*j)]
    deadline = (time.time() + args.max_minutes * 60.0) if args.max_minutes else None
    print(f"render {spec.name}: {len(jobs)} rows over {len(spec.legs)} leg(s), "
          f"{len(todo)} need crops = up to {2*len(todo)} renders, "
          f"{args.workers}x{bq.RENDER_THREADS} threads", flush=True)
    t0, done, fails, stopped = time.time(), 0, [], False
    t_win, n_win = t0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_render_one, (r, crops, vivid, args.render_timeout)): r
                for r, crops, vivid in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:                               # noqa: BLE001
                fails.append(dict(image_id=futs[fut]["image_id"],
                                  err=f"{type(e).__name__}: {str(e)[:200]}"))
            done += 1
            if done % 25 == 0 or done == len(todo):
                now = time.time()
                recent = (now - t_win) / max(done - n_win, 1)
                # Reproject from RECENT throughput, never the run-to-date average: the draw is
                # cell-round-robin so deep rows are spread through, but a rate that is falling
                # must be read off the tail (`CLAUDE.md`, "Projecting a long run's wall clock").
                print(json.dumps(dict(done=done, of=len(todo),
                                      elapsed_min=round((now - t0) / 60, 1),
                                      recent_s_per_row=round(recent, 2),
                                      eta_min=round((len(todo) - done) * recent / 60, 1),
                                      failed=len(fails))), flush=True)
                t_win, n_win = now, done
            if deadline and time.time() > deadline:
                stopped = True
                for f2 in futs:
                    f2.cancel()
                break
    if fails:
        # The WHOLE failure list, never a head slice: a truncated error log describes the
        # fastest-returning failure class, not the population (`CLAUDE.md`, "Four rules").
        p = Path(__import__("paths").scratch(f"sitting_{spec.name}", "render_failures.json"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(n=len(fails),
                                     by_class=dict(Counter(f["err"].split(":")[0]
                                                           for f in fails)),
                                     failures=fails), indent=2), encoding="utf-8")
        print(f"  !! {len(fails)} render failures -> {p}")
    print(f"render: {done} rows this pass"
          + ("  [STOPPED at the time bound]" if stopped else ""))
    total_miss = 0
    for b, (rows, crops, vivid) in per_leg.items():
        miss = sum(1 for r in rows if needs(r, crops, vivid))
        total_miss += miss
        print(f"  {b:38s} {len(rows)-miss:4d}/{len(rows):4d}"
              + ("  COMPLETE" if miss == 0 else f"  INCOMPLETE — {miss} still need crops"))
    return 0 if total_miss == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="report what a sitting WOULD contain; serve nothing")
    d.add_argument("--run-dir", default=None,
                   help="a SINGLE run's record store. Omit and pass --sitting to dry-run the "
                        "declared spec's whole union.")
    d.add_argument("--sitting", choices=sorted(SITTINGS), default=None)
    # None => the SPEC's cap (or MAX_ROWS for a bare --run-dir). A hardcoded 1000 default made
    # the rehearsal cut a different page size from the draw it rehearses.
    d.add_argument("--max-rows", type=int, default=None)
    d.add_argument("--embed-limit", type=int, default=None,
                   help="bound the morph pass. The unembedded remainder is counted as "
                        "budget_not_reached, never silently dropped.")
    d.add_argument("--no-cache", action="store_true",
                   help="bypass the persistent morph-embed store (a COLD timing arm)")
    d.add_argument("--out", default=None)

    w = sub.add_parser("draw", help="cut the sitting and write each leg's registered batch")
    w.add_argument("--sitting", choices=sorted(SITTINGS), default=V2_SITTING.name)
    w.add_argument("--max-rows", type=int, default=None)
    w.add_argument("--embed-limit", type=int, default=None,
                   help="SMOKE TEST ONLY: bound the morph pass so the whole draw path runs "
                        "end to end in seconds. The dedup is then PARTIAL, so every batch.json "
                        "it writes is stamped sitting_cut.INCOMPLETE = true; re-run without "
                        "the flag before rendering or labelling.")
    w.set_defaults(fn=stage_draw)

    r = sub.add_parser("render", help="render both crops per row; resumable, idempotent")
    r.add_argument("--sitting", choices=sorted(SITTINGS), default=V2_SITTING.name)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--render-timeout", type=float, default=600.0)
    r.add_argument("--max-minutes", type=float, default=0.0)
    r.set_defaults(fn=stage_render)

    a = ap.parse_args()
    if getattr(a, "fn", None) is not None:
        if getattr(a, "workers", 0) and a.workers > 4:
            print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
            return 2
        return a.fn(a)

    import time                                               # noqa: E402
    import paths                                              # noqa: E402
    import corpus_common as cc                                # noqa: E402
    from tools.wallpaper import morph_embed_cache as mec      # noqa: E402
    # A cold morph pass is ~an hour of engine renders launched through a helper that takes no
    # creationflags; the child inherits this class, which is the only lever that reaches them.
    cc.set_below_normal_priority()
    if not a.run_dir and not a.sitting:
        raise SystemExit("dry-run needs --run-dir (one run) or --sitting (a declared union)")
    # THE DRY RUN MUST CUT THE SAME WAY THE DRAW DOES. A rehearsal that took the balanced
    # path while `draw` took the bucketed one would report an apportionment nobody is about to
    # build — the bounded end-to-end exists to be the same path, cheaply, or it is not one
    # (`CLAUDE.md`: give a long path a bounded end-to-end). `--run-dir` names a single run and
    # no spec, so it has no buckets and no leg order to honour, and keeps the balanced draw.
    spec = None
    if a.run_dir:
        rows, qrep = load_queue(a.run_dir), dict(source="--run-dir", run_dir=str(a.run_dir))
    else:
        spec = SITTINGS[a.sitting]
        rows, qrep = load_union_queue(spec)
        rows, qrep["population"] = apply_population(rows, spec)
        qrep["source"] = f"--sitting {a.sitting}"
    max_rows = a.max_rows or (spec.max_rows if spec else MAX_ROWS)
    scratch = Path(paths.scratch("sitting_cutter", "fields"))
    cache = None if a.no_cache else mec.MorphEmbedCache().open()
    t0 = time.time()
    res = cut_sitting(rows, max_rows=max_rows,
                      embed=make_embedder(scratch, a.embed_limit, cache),
                      progress=lambda d: print(json.dumps(d), flush=True),
                      buckets=((spec.buckets or None) if spec else None),
                      leg_rank=(spec.leg_rank if spec else None),
                      no_pad=(spec.no_pad if spec else None))
    elapsed = time.time() - t0
    rep = res["report"]
    rep["queue"] = qrep
    rep["by_leg"] = dict(Counter(r.get("_leg") for r in res["sitting"]))
    rep["SERVED"] = False
    rep["wall_s"] = round(elapsed, 2)
    rep["morph_cache"] = cache.report() if cache else "DISABLED (--no-cache)"
    if cache:
        cache.close()
    rep["note"] = ("DRY RUN: no manifest written, no export, no crop rendered. `serve` is "
                   "what builds a sitting.")
    out = Path(a.out) if a.out else Path(paths.scratch("sitting_cutter", "dry_run.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    print(f"\n-> {out}   (NOTHING SERVED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
