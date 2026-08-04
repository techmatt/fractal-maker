#!/usr/bin/env python
r"""supply_routing.py — WHICH supply channel feeds WHICH partition, and what priced it.

THE POINT OF THIS BEING DATA. Harvest v1 ran every channel into every partition and let the
frontier sort it out; the q4 sitting then priced the channels against 870 human labels and
the prices are *sharply* partition-dependent — one channel is worth 4x the base rate in one
partition and measures literally zero in another. A routing that lives in a launch script is
a routing nobody can diff against the labels that set it, so it lives here as a table, each
row carrying the measurement that justifies it, and `tools/atlas/test_supply_routing.py`
asserts the table against the run config the launcher actually writes.

`[measured: scratch/q4_readout/REPORT.md + readout.txt, 870 labels, 2026-08-03]` unless a row
says otherwise. Every rate below is a HUMAN-label rate, not a decode.

THE FOUR ROUTES
---------------
**julia:mandelbrot — the rich partition, and the only one with three live channels.**
Ranked harvest (>=3 79.1%, class-4 16.3%) is the top-end channel; the near-minibrot ladder
(>=3 66.6%, class-4 4.8%) beats unscreened supply 4x but is beaten at the top end; the
unscreened boundary draw is 16.7% / 0%. All three stay, because the deficit is large and the
ladder's cost per label is low. **Machine-1s are NOT discarded here**: P(Matt=1 | v10
decoded 1) is 30.9% in this partition and P(>=3 | decode 1) is **16.5%** (16 of 97) — an
auto-discard would throw away one good-or-better picture in six.

**native multibrot3/4/5 — seeds and TRIGGERED maneuvers only.** The unscreened dM_d-shell
draw measured **0 of 144 at >=2** ([0, 7.4%] each at n=48), i.e. it is priced at zero and is
not built. That is a verdict on the DRAW (eps=0.02 shell, no screen), not a ceiling on the
family — the same families reach >=3 through triggered maneuvers at 55.0% against a
partition-matched 25.5% fresh.

**phoenix — parameter-space spread off the neutral-stability skeleton.** >=2 39.8%, >=3 6.1%
unscreened, but 57 class-4s in the corpus: the constraint here is MOTIF SCARCITY, not volume.
Spread the skeleton draw; do not deepen it.

**maneuvers-on-admissions — promoted from an experiment to a budgeted channel.** 55.0% vs
25.5% at >=3, partition-matched, same direction in all four c-plane partitions (n=60, single
run). It is c-plane only by construction (a julia/phoenix viewport has no nucleus in the
parameter-plane sense), and its share is governed by the quota like any other spend rather
than by a fixed per-batch count.

THE c-SPACING FLOOR — 3.2e-2, on a FIXED-VIEWPORT re-measurement
----------------------------------------------------------------
`CSPACING_FLOOR` is the minimum |delta c| between two accepted julia parameters. It is not a
saturation point, and the 1e-2 it replaces was read as though it were.

The first derivation (labelled ladder, 290 rows / 103 atoms) paired images rendered at their
OWN framings, so framing dissimilarity was scored as look dissimilarity, and it reported the
near-dup rate falling to a different-atom baseline of 2.3% at 1e-2. Re-measured with both
members of every pair at the SAME z-viewport, on the canonical morph_clip substrate, that
knee does not exist — the rate decays smoothly across five decades and reaches the baseline
nowhere below ~3e-1:

    |delta c|         constructed pairs        v2 pool's own pairs
                    n     med cos    >=.974     n      >=.974
    1e-5 - 3.2e-5    507   0.9990     1.000      -          -    (pool is thinned at 1e-2,
    1e-4 - 3.2e-4   1161   0.9984     0.977      -          -     so it HAS no sub-floor
    1e-3 - 3.2e-3   1888   0.9902     0.673      -          -     pairs to contribute)
    3.2e-3 - 1e-2   3528   0.9797     0.538      -          -
    1e-2 - 3.2e-2   4231   0.9487     0.354    484       0.130
    3.2e-2 - 1e-1   4175   0.9091     0.153   1828       0.045
    1e-1 - 3.2e-1   3479   0.8430     0.029   7010       0.019
    different-region reference, any distance: 26368 pairs, med 0.8454, >=.974 = 0.0036

So a floor is a TOLERANCE against pool cost, not a point where the signal ends, and it has to
be quoted at the floor rather than averaged over a bin the rate is falling through. At
quarter-decade resolution the rate a floor actually admits — its closest surviving pairs —
is, on the production-accepted population: 0.196 at 1e-2, **0.074 at 3.2e-2**, 0.037 at
5.6e-2. **Matt's decision: 3.2e-2** — 2.6x fewer near-dup closest pairs than the old floor,
at 210/539 of the committed v2 pool retained under naive re-thinning (the v3 build re-thins
the full candidate set instead and lands above that).

Two reads that constrain how this generalizes, both in `julia_c_sourcing.md`:
  * The floor is VIEWPORT-CONDITIONAL. At 1e-2 the same pairs read 0.354 near-dup at the wide
    whole-julia framing and 0.130 at a mid-zoom. Wide is what the class is emitted at, so
    wide is the read and it is the conservative one.
  * Distance to dM moves the knee about half a decade: at 1e-2-3.2e-2 the near-dup rate is
    0.294 for c near the boundary against 0.547 for c further out, consistent in direction in
    every bin. Not adopted as a covariate-scaled rule — every production channel selects onto
    the knife edge, which is the half the absolute floor is already set on.

The atom-level correction the first derivation earned still stands and is why the floor is an
absolute distance rather than a per-atom cap: the roster's atoms sit a median 9.1e-4 apart,
and different atoms that close are near-duplicates of each other whatever their provenance
says. "One c per atom" was never enough.

Compare the julia HOOK's spacing (0.20, or 0.10 after the campaign-2 resume): that is 3-6x
coarser and was set on a different population, so the two are not interchangeable and this
one is not derived from it.

THE SINGLE RUNG
---------------
The ladder's three rungs are one look (`LADDER_RUNGS_MEASURED`), and their yields are flat,
so the choice is a cost decision — see `RUNG_CHOICE` for the measured basis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# The near-minibrot channel's two derived constants.
# --------------------------------------------------------------------------- #
NEAR_DUP_COS = 0.974          # the library/emission near-dup knee; not re-decided here

CSPACING_FLOOR = 3.2e-2
CSPACING_BASIS = dict(
    adopted="2026-08-03, Matt's decision. The measurement below sizes the choice; it does "
            "not make it — there is no knee to read a floor off (see `rule`).",
    supersedes=dict(
        floor=1e-2,
        why="its pairs were rendered at their OWN viewports, so framing dissimilarity was "
            "scored as look dissimilarity and produced an apparent return-to-baseline at "
            "1e-2. At a fixed viewport the same distance reads 0.354 near-dup on constructed "
            "pairs and 0.130 on the pool's own, against a 0.0036 baseline.",
    ),
    measured_on="1421 c (14 regions x 63 satellites over 10 half-decade annuli 1e-5..3e-1, "
                "+ the 539-c committed v2 pool) x 3 SHARED z-viewports = 4263 embeddings",
    command="uv run python tools/studies/julia_c_stationarity.py all",
    recipe="library morph CLIP (robustz_tanh_k2_v1, 640x360ss2, "
           "vit_base_patch16_clip_224.openai) at cos >= 0.974",
    rule="a TOLERANCE against pool cost, not a saturation point: the near-dup rate decays "
         "smoothly over five decades and reaches the different-region baseline nowhere below "
         "~3e-1, so no bucket boundary can be read as a knee",
    # Quarter-decade, canonical `wide` viewport, viable pairs, production-accepted population
    # — the rate among the CLOSEST pairs a floor still admits, not a bin average.
    near_dup_rate_at_floor=0.074,
    near_dup_rate_at_old_floor=0.196,
    different_region_baseline=0.0036,
    atom_nn_median_dc=9.05e-4,
    viewport_conditional="0.354 (wide, fw 1.3) vs 0.130 (mid, fw 0.55) at 1e-2 on the same "
                         "pairs; wide is the framing the class is emitted at, so wide is the "
                         "read and it is the conservative one",
    dM_distance_split="0.294 near dM vs 0.547 further out at 1e-2-3.2e-2, same direction in "
                      "every bin; NOT adopted as a covariate rule — every channel selects "
                      "onto the knife edge, which is the half this floor is set on",
    note="one c per ATOM is NOT sufficient: the roster's atoms sit a median 9.1e-4 apart, "
         "two decades inside the floor",
)

LADDER_RUNGS_MEASURED = (1.0, 4.0, 16.0)
LADDER_YIELD = {         # human labels, one-per-cluster (ball) where the readout gives both
    1.0: dict(n=97, ge3=0.680, ge3_1pc=0.618, eq4=0.062, eq4_1pc=0.086),
    4.0: dict(n=96, ge3=0.635, ge3_1pc=0.653, eq4=0.031, eq4_1pc=0.040),
    16.0: dict(n=97, ge3=0.680, ge3_1pc=0.667, eq4=0.052, eq4_1pc=0.037),
}
SAME_ATOM_SATURATION = dict(pairs=278, median_cos=0.9825, frac_at_or_above_cut=0.741)

# The single rung. `cost_basis` is filled from a measurement, never asserted — see
# `rung_choice()`, which reads the cost record and refuses to answer without one.
RUNG_CHOICE_RECORD = ROOT / "data" / "atlas" / "near_minibrot_rung_v2.json"


def rung_choice(record: Path | None = None) -> dict:
    """The single rung the v2 near-minibrot channel emits, and why.

    Yields are flat across the three rungs (>=3 at 68.0/63.5/68.0, one-per-cluster
    61.8/65.3/66.7 — every pair's Wilson intervals overlap), so yield cannot decide and the
    prompt's instruction is that cost does. This reads the measured per-rung cost record and
    picks the cheapest; if two rungs are within `TIE_BAND` of each other it falls back to the
    rung with the best measured one-per-cluster class-4 rate, which is the only yield column
    where the three rungs differ by more than noise (8.6% / 4.0% / 3.7%).

    NOT ABSENCE-TOLERANT. Without the cost record there is no basis for the choice, and
    picking a rung anyway would be exactly the "decided in a launch script" failure this
    module exists to prevent."""
    p = Path(record) if record else RUNG_CHOICE_RECORD
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing — the single-rung choice is a COST decision and there is no "
            f"cost measurement to make it from. Rebuild with "
            f"`uv run python tools/atlas/near_minibrot_rung.py measure`.")
    rec = json.loads(p.read_text(encoding="utf-8"))
    # Keyed off the RUNGS, not off "every key that isn't metadata": a metadata key added to
    # the record later would otherwise be parsed as a rung and crash the routing.
    missing = [r for r in LADDER_RUNGS_MEASURED if str(r) not in rec]
    if missing:
        raise ValueError(f"{p} has no cost for rung(s) {missing}; it measures {sorted(rec)}")
    costs = {r: float(rec[str(r)]["mean_s"]) for r in LADDER_RUNGS_MEASURED}
    cheapest = min(costs, key=lambda r: (costs[r], r))
    spread = (max(costs.values()) - min(costs.values())) / max(1e-12, min(costs.values()))
    if spread <= TIE_BAND:
        pick = max(LADDER_YIELD, key=lambda r: (LADDER_YIELD[r]["eq4_1pc"], -r))
        why = (f"cost is a tie ({spread:.1%} spread across rungs, band {TIE_BAND:.0%}), so "
               f"the tie-break is the one yield column where the rungs separate: "
               f"one-per-cluster class-4 rate")
    else:
        pick, why = cheapest, f"cheapest measured render ({spread:.1%} spread across rungs)"
    return dict(rung=pick, why=why, cost_s=costs, spread=round(spread, 4),
                yields=LADDER_YIELD, record=str(p))


TIE_BAND = 0.10        # <=10% cost spread across rungs is a tie, not a cost signal


# --------------------------------------------------------------------------- #
# The routing table.
# --------------------------------------------------------------------------- #
# `machine_1_discard`: whether a v10 decode of class 1 may auto-discard the row before a
# human sees it. Partition-dependent because the measurement is: P(Matt=1 | decoded 1) runs
# 94-100% in native multibrot and 72.0% in phoenix (with P(>=3 | decoded 1) of 0/82 there),
# but only 30.9% in julia:mandelbrot, where 16.5% of machine-1s are >=3.
MACHINE_1_DISCARD = {
    "mandelbrot": False,          # no per-partition measurement; fail-closed to KEEP
    "julia:mandelbrot": False,    # measured: P(>=3 | decoded 1) = 16.5% (16/97)
    "multibrot3": True,           # measured: P(Matt=1 | decoded 1) 94-100%, P(>=3|dec1) <= 2%
    "multibrot4": True,
    "multibrot5": True,
    "julia:multibrot3": False,    # no per-partition measurement; fail-closed to KEEP
    "julia:multibrot4": False,
    "julia:multibrot5": False,
    "phoenix": True,              # measured: P(Matt=1 | decoded 1) 72.0%, P(>=3|dec1) 0/82
}

ROUTES = {
    "julia:mandelbrot": dict(
        channels=("seeded_loop", "q4_mining_ranked", "q4_mining_recall", "near_minibrot"),
        near_minibrot=dict(single_rung=True, cspacing_floor=CSPACING_FLOOR),
        evidence="ranked harvest >=3 79.1% / class-4 16.3%; ladder >=3 66.6% / 4.8%; "
                 "unscreened 16.7% / 0%",
    ),
    "multibrot3": dict(channels=("seeds", "triggered_maneuvers"),
                       evidence="unscreened dM shell 0/48 at >=2; triggered >=3 55.6% vs 9.1%"),
    "multibrot4": dict(channels=("seeds", "triggered_maneuvers"),
                       evidence="unscreened dM shell 0/48 at >=2; triggered >=3 17.9% vs 9.1%"),
    "multibrot5": dict(channels=("seeds", "triggered_maneuvers"),
                       evidence="unscreened dM shell 0/48 at >=2; triggered >=3 62.0% vs 50.0%"),
    "mandelbrot": dict(channels=("seeds", "triggered_maneuvers"),
                       evidence="triggered >=3 87.5% vs 0.0% (n small); c-plane, same route "
                                "as the multibrots"),
    "phoenix": dict(channels=("skeleton_spread",),
                    evidence="unscreened >=2 39.8% / >=3 6.1%, but 57 class-4s in corpus: "
                             "motif scarcity, not volume"),
    "julia:multibrot3": dict(channels=("julia_hook",), evidence="hook-fed only"),
    "julia:multibrot4": dict(channels=("julia_hook",), evidence="hook-fed only"),
    "julia:multibrot5": dict(channels=("julia_hook",), evidence="hook-fed only"),
}

# Channels that are PRICED AT ZERO and therefore not built. Named rather than omitted: a
# channel that is absent and a channel that was measured worthless read identically in a
# config, and only the second is a decision.
RETIRED_CHANNELS = {
    "unscreened_dM_shell": dict(
        partitions=("multibrot3", "multibrot4", "multibrot5"),
        measured="0 of 144 rows reached >=2 (48 per family, [0, 7.4%] each); no class-4 "
                 "anywhere in the leg",
        scope="a verdict on the DRAW (dM_d shell eps=0.02, no screen), NOT a ceiling on the "
              "family — the same families reach >=3 at 55% through triggered maneuvers",
        population="2026-08-03_q4_uniform_eval_v1, the score-unconditioned leg",
    ),
    "near_minibrot_multi_rung": dict(
        partitions=("julia:mandelbrot",),
        measured="same-atom different-rung pairs sit at median cos 0.9825 with 74.1% at or "
                 "above the 0.974 near-dup cut; yields flat across rungs",
        scope="the 1x/4x/16x ladder buys ~1 look per atom for 3x the label cost",
        population="2026-08-03_q4_near_minibrot_v1, 290 labelled rows / 103 atoms",
    ),
}


def cspacing_ok(c, accepted, floor: float = CSPACING_FLOOR) -> bool:
    """True iff julia parameter `c` clears the floor against every already-accepted `c`.

    Plain euclidean distance in the c-plane, which is the metric the saturation was measured
    in. Deliberately NOT the ledger's `is_distinct` (that one is a viewport-coordinate dedup
    with a seed-c-aware term and a different radius): this floor is about which PARAMETER
    values to draw, and it applies before any frame exists."""
    cr, ci = float(c[0]), float(c[1])
    for a in accepted:
        if math.hypot(cr - float(a[0]), ci - float(a[1])) < floor:
            return False
    return True


def thin_by_cspacing(cands, key=lambda r: (r["c_re"], r["c_im"]),
                     floor: float = CSPACING_FLOOR):
    """Greedy first-wins thinning of a candidate list to the c-spacing floor.

    FIRST WINS, so the caller's ordering is the policy — hand it a list already sorted
    best-first and the floor keeps the best of each cluster. Returns `(kept, dropped)` rather
    than filtering in place, because "how much did the floor cost?" has to be a read: the
    ladder leg would have lost ~2 of every 3 rows to this and that number belongs in the run
    record, not in a diff of two lengths."""
    kept, dropped, acc = [], [], []
    for r in cands:
        c = key(r)
        if cspacing_ok(c, acc, floor):
            acc.append((float(c[0]), float(c[1])))
            kept.append(r)
        else:
            dropped.append(r)
    return kept, dropped


def summary(rung_record: Path | None = None) -> dict:
    """The whole routing decision as one JSON-able record, for the run config."""
    try:
        rc = rung_choice(rung_record)
    except FileNotFoundError as e:
        rc = dict(rung=None, why=f"UNMEASURED: {e}")
    return dict(routes=ROUTES, machine_1_discard=MACHINE_1_DISCARD,
                retired_channels=RETIRED_CHANNELS,
                cspacing_floor=CSPACING_FLOOR, cspacing_basis=CSPACING_BASIS,
                near_dup_cos=NEAR_DUP_COS, rung=rc,
                same_atom_saturation=SAME_ATOM_SATURATION)


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
