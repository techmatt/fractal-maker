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

THE c-SPACING FLOOR (derived here, not inherited)
-------------------------------------------------
`CSPACING_FLOOR` is the minimum |delta c| between two accepted julia parameters in the
near-minibrot channel. It is DERIVED from the labelled ladder's own morph embeddings
(290 rows / 103 atoms, `scratch/q4_readout/morph_emb_870.npz`, the library CLIP recipe at
cos 0.974), and it corrects the atom-level framing the q4 readout stopped at:

    |delta c| bucket      pairs   median cos   frac >= 0.974
    [1e-5, 1e-4)             75       0.9824           0.813
    [1e-4, 1e-3)            659       0.9708           0.417
    [1e-3, 1e-2)           2196       0.9622           0.239
    [1e-2, 1e-1)           1866       0.9267           0.024   <-- baseline reached
    >= 1e-1               36964       0.8992           0.004
    (different-atom pairs, any distance, as the reference: 0.023)

The readout's finding was "same atom, different rung => same look" (median cos 0.9825, 74.1%
at/above the cut). Restricting to DIFFERENT-atom pairs shows the saturation is a property of
the c-plane distance and not of atom identity: different atoms at 1e-4..1e-3 are still 38%
near-dup, sixteen times the different-atom baseline. So "one c per atom" would NOT have been
enough — two neighbouring atoms are near-duplicates of each other, and the roster's atoms sit
a median 9.1e-4 apart.

**Floor = 1e-2**: the coarsest bucket boundary at which the near-dup rate falls to the
different-atom baseline (2.4% against 2.3%). One bucket finer it is 23.9%, ten times that.
Stated as a bucket boundary rather than a fitted knee because the measurement is bucketed —
quoting three significant figures off five bins would be precision the data does not carry.

Compare the julia HOOK's spacing (0.20, or 0.10 after the campaign-2 resume): that is 10-20x
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

CSPACING_FLOOR = 1e-2
CSPACING_BASIS = dict(
    measured_on="data/label_corpus/batches/2026-08-03_q4_near_minibrot_v1 (290 rows, "
                "103 atoms) x scratch/q4_readout/morph_emb_870.npz",
    recipe="library morph CLIP (robustz_tanh_k2_v1, 640x360ss2, "
           "vit_base_patch16_clip_224.openai) at cos >= 0.974",
    rule="the coarsest |delta c| bucket boundary at which the near-dup rate reaches the "
         "different-atom baseline",
    near_dup_rate_at_or_above_floor=0.024,
    near_dup_rate_one_bucket_below=0.239,
    different_atom_baseline=0.023,
    atom_nn_median_dc=9.05e-4,
    note="one c per ATOM is NOT sufficient: different atoms at 1e-4..1e-3 are 38% near-dup",
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
