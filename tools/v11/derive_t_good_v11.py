#!/usr/bin/env python
r"""Derive v11 per-partition t_good — the ADOPTED discovery table at the v11 flip.

t_good is a cut on ONE head's `P(>=3)`; v11's probability scale is not v10's, so the v10
table is a set of numbers about nothing on a v11 gate (protocol §4). This re-derives it.

THE ESTIMATOR IS IMPORTED, NOT COPIED. `scoring/derive_t_good.build_table` is the derivation —
grid, F_beta-argmax, tie-break-toward-higher-t, LOO-OOF, plateau width, the >=15-positive
sufficiency floor and the UNCALIBRATED stamping all run from that module. A copied deriver is
how two thresholds that are supposed to be comparable stop being comparable.

THREE THINGS THIS PASS DECIDES, none inherited:

1. THE POPULATION RULE — **the randomized location-GROUPED holdout, instrument as fallback**
   (`derive_t_good.select_population`, the shared default from v11 on; Matt, 2026-08-06).
   This is the conformance item of the flip and it is what makes the table cover the corpus
   instead of covering the instruments. v10's eval side WAS the four score-unconditioned
   draws — 1,050 rows, none of them julia:mandelbrot, phoenix or classic phoenix — so six of
   ten partitions ran at the 0.50 baseline while the corpus held 809 julia:mandelbrot and 375
   phoenix positives. The positives existed; no eval slice contained any of them.

   v11's build adds a seeded stratified holdout over the location-grouped split, biased
   exactly as training is, which is what a calibration cut needs and what a base rate must
   never be read from. STILL NEVER POOLED: a partition is cut on the holdout OR on its
   instrument, never their union. Exactly one partition takes the fallback —
   `julia:multibrot3`, 3 holdout positives against 19 in the `prospect_census` — and that is
   the same census v10 cut it on, so its number stays readable across the flip.

2. THE OBJECTIVE, re-read from CURRENT supply (recall where scarce, precision where
   abundant). The evidence is the label-corpus draw since the v10 table was cut on
   2026-08-02 — every batch dated 2026-08-03 or later, by partition:

     julia:mandelbrot 814 · phoenix 525 · multibrot3 419 · julia:multibrot5 281 ·
     multibrot5 269 · julia:multibrot4 266 · julia:multibrot3 198 · multibrot4 191 ·
     mandelbrot 155

   (`julia_ladder_j0`'s 1,000 julia:mandelbrot rows are NOT in that count — the directory has
   no date prefix and sorts last, but the batch was created 2026-06-25 and predates v10.)

     * mandelbrot — ABUNDANT, and its 155 is the smallest draw in the table precisely
       BECAUSE of that. Mandelbrot is the anchor partition that sets the release-mix target
       and carried +0.0 deficit through the 2026-08-03 sitting, so the scheduler steered
       every subsequent draw elsewhere. That is a selection artifact, not scarcity — the
       interpretation guard in deferred_recalibration.md, applied in the direction it was
       written for. -> F0.5 (precision), UNCHANGED.
     * julia:multibrot{3,4,5} — SCARCE. The v10 read rested on a hard zero (both 2026-08
       supply efforts were 100% native-plane), and that zero is gone: 198/266/281 rows have
       been drawn since. The BASIS moved and the verdict does not. These are still the three
       smallest parameter-plane families, every one of those rows came through a quality gate
       rather than from an exhaustion test, and a miss is still gone. -> F2 (recall),
       UNCHANGED NUMBER ON CHANGED EVIDENCE, which is the read protocol §4 asks for.
     * julia:mandelbrot — NEW, and ABUNDANT: 814 rows, the largest parameter-plane draw by
       1.6x, generated systematically by the near-minibrot and ladder builders rather than
       harvested opportunistically. A miss costs the next hunt. -> F0.5 (precision).
     * phoenix — NEW, and ABUNDANT: 525 rows since 2026-08-03, and phoenix supply is a
       PARAMETER GRID (`2026-07-21_phoenix_grid` drew 500 in a single pass), so it is
       generable at will rather than found. -> F0.5 (precision).
       Phoenix is the partition where class 4 outnumbers class 3, which is an argument about
       the VALUE of a phoenix keeper, not about the cost of missing one when the next sweep
       produces more. The objective keys on supply; that is the principle as written.
     * multibrot{3,4,5} (native) — ABUNDANT (419/191/269). Recorded, and the partitions are
       WITHHELD (see 3), so the objective never fires.

3. THE NATIVE MULTIBROTS — WITHHELD AT THE FLIP, ADOPTED THE SAME DAY. Under the new
   population rule the three NATIVE multibrots clear MIN_POS for the first time (49/32/38
   holdout positives against zero keeper positives in v10's uniform instrument). Adopting
   them is a native multibrot tightening, and that decision was FORK-SCHEDULED — taken at
   the fork launch together with tau_h, not as a side effect of a flip
   (deferred_recalibration.md § "Related, but not part of this cluster"). So they first went
   through `withhold`: derived, written to the artifact's `withheld` block for the owner of
   the decision, and running at the 0.50 baseline with a reason that said "withheld", not
   "no data".

   **The fork ran on 2026-08-08 and the owner approved the tightening**, so `WITHHOLD` is now
   empty and `multibrot{3,4,5}` are DERIVED at 0.61 / 0.85 / 0.61 — byte-for-byte the values
   the `withheld` block had already recorded, which is what that path exists to guarantee.
   0.50 -> 0.61/0.85/0.61 is a TIGHTENING on all three; read the admitted-count deltas in
   `scratch/tau_h_enlargement_report.md`, not the thresholds, for what it costs.

   Two stamps that ride along and are NOT weakened by adoption: multibrot3 and multibrot5 sit
   on 1-step plateaus (knife-edge — re-derive rather than nudge), and the holdout is biased
   exactly as training is, so 0.583 / 0.704 / 0.667 precision is NOT what the gate delivers on
   a discovery frontier. Both are emitted automatically by the shared estimator.

   `phoenix:classic` is the one partition the rule still cannot reach: 8 holdout rows with 1
   positive, no instrument rows at all. It stays UNCALIBRATED at 0.50, below MIN_POS.

THE DURABLE WRITE TAKES `--adopt`, matching derive_t_good_v{8,9,10}: `data/v11/
t_good_derivation.json` is the record the live cuts in `production_seeder.T_GOOD_OVERRIDES`
are MIRRORED from, so a re-derivation over it moves the record while the running gate keeps
the old numbers — and the desync is silent in both directions.

  uv run python tools/v11/derive_t_good_v11.py            # print + scratch/v11/, no write
  uv run python tools/v11/derive_t_good_v11.py --adopt    # write data/v11/t_good_derivation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas",
           ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import derive_t_good as est      # noqa: E402  THE estimator, imported not copied
import paths                     # noqa: E402

VERSION = "v11"
EVAL_REL = "data/v11/eval_scores_v11.jsonl"
EVAL = ROOT / EVAL_REL
OUT_REL = "data/v11/t_good_derivation.json"

# --- population rule: the shared grouped-holdout default (see §1) ------------------------- #
# `INSTRUMENT` names the ONE fallback source per partition, kept from v10's table so a
# partition that falls back lands on the population v10 cut it on rather than on whatever
# instrument rows happen to exist. Partitions absent here fall back to any instrument row.
INSTRUMENT = {
    "mandelbrot": "loose0_v3_floor",
    "julia:multibrot3": "prospect_census",
    "julia:multibrot4": "prospect_census",
    "julia:multibrot5": "prospect_census",
    "multibrot3": "maneuver_uniform_v1",
    "multibrot4": "maneuver_uniform_v1",
    "multibrot5": "maneuver_uniform_v1",
}

# --- objective, re-read from post-2026-08-02 supply (see §2 above) ------------------------ #
OBJECTIVE = {
    "mandelbrot": 0.5,          # abundant (anchor partition; small draw is a deficit artifact)
    "julia:mandelbrot": 0.5,    # NEW — abundant (814 since 2026-08-03, largest param-plane draw)
    "phoenix": 0.5,             # NEW — abundant (525; supply is a parameter grid)
    "multibrot3": 0.5,          # abundant (419) — ADOPTED 2026-08-08 (was WITHHELD at the flip)
    "multibrot4": 0.5,          # abundant (191) — ADOPTED 2026-08-08 (was WITHHELD at the flip)
    "multibrot5": 0.5,          # abundant (269) — ADOPTED 2026-08-08 (was WITHHELD at the flip)
    "julia:multibrot3": 2.0,    # scarce — 198 drawn, gate-limited, smallest param-plane families
    "julia:multibrot4": 2.0,    # scarce — 266
    "julia:multibrot5": 2.0,    # scarce — 281
}

# --- the withhold, and its 2026-08-08 release (see §3 above) ------------------------------ #
# EMPTY ON PURPOSE, and it is not empty because nothing was withheld — the three native
# multibrots WERE withheld at the flip earlier the same day, and the fork that owned the
# decision released them the same day (Matt-approved, `tau_h_enlargement.md` §1). They are now
# DERIVED at exactly the numbers the `withheld` block recorded (0.61/0.85/0.61), off the same
# holdout population, same F0.5 objective, same estimator — the withhold path's whole point
# was that the owner would not have to re-derive, and it did not.
#
# Do NOT re-add these three to buy back the old behaviour: the pre-adoption record is
# `tau_h_base_v11.json`'s superseded sibling and the v11-flip artifact in git history, and a
# re-withhold would be a NEW loosening decision needing its own justification.
WITHHOLD: dict = {}

UNCAL_REASON = {
    # The ONE partition the grouped holdout still cannot reach. NOT the never-looked case: a
    # human has labeled all of it; the eval side simply draws 8 rows with 1 positive, and the
    # partition has no instrument rows of any kind (the instruments carry no parameter axes,
    # so they cannot separate classic from varied phoenix at any n).
    "phoenix:classic": (f"8 holdout rows / 1 positive and 0 instrument rows — below "
                        f"MIN_POS={est.MIN_POS} on both eval roles, the only partition still "
                        f"short of it under the grouped-holdout rule"),
}


def select_population(rows):
    """The shared grouped-holdout rule, with v10's instrument map as the fallback source.

    Named `select_population` on purpose: `tools/scoring/test_t_good_adoption.py`'s drift
    gate re-runs the LIVE version's deriver through this name, so the entry point is part of
    the per-version deriver contract, not an implementation detail."""
    return est.select_population(rows, instrument=INSTRUMENT)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    adopt = "--adopt" in argv
    if not EVAL.exists():
        sys.exit(f"missing {EVAL} — run tools/v11/eval_v11.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, report = select_population(rows)
    print(f"[population] grouped holdout, instrument fallback: {len(kept)}/{len(rows)} "
          f"eval rows kept")
    print(f"  {'partition':22s} {'cut on':>11}  {'n':>5} {'pos':>4}   "
          f"{'holdout n[pos]':>16} {'instrument n[pos]':>19}")
    for fam, d in sorted(report.items()):
        h, i = d["holdout"], d["instrument"]
        print(f"  {fam:22s} {str(d['population']):>11}  {d['n']:5d} {d['pos']:4d}   "
              f"{h['n']:11d}[{h['pos']:3d}] {i['n']:14d}[{i['pos']:3d}]"
              + (f"  src={i['source']}" if d["population"] == "instrument" else ""))
    print()

    out = est.build_table(kept, version=VERSION, eval_slice=EVAL_REL,
                          objective=OBJECTIVE, uncal_reason=UNCAL_REASON, withhold=WITHHOLD)
    out["population_rule"] = (
        "THE RANDOMIZED LOCATION-GROUPED HOLDOUT, frozen instrument as the per-partition "
        "fallback, NEVER pooled (derive_t_good.select_population — the shared default from "
        "v11 on; Matt, 2026-08-06). Replaces v10's one-frozen-instrument-per-partition rule, "
        "which left six of ten partitions with no eval row of any kind while the corpus held "
        "809 julia:mandelbrot and 375 phoenix positives.")
    out["population_detail"] = report
    out["objective_basis"] = (
        "Re-read from the label-corpus draw since the v10 table was cut (batches dated "
        "2026-08-03+): julia:mandelbrot 814, phoenix 525, multibrot3 419, julia:multibrot5 "
        "281, multibrot5 269, julia:multibrot4 266, julia:multibrot3 198, multibrot4 191, "
        "mandelbrot 155. mandelbrot's small draw is a DEFICIT artifact — it is the anchor "
        "partition at +0.0 deficit, so the scheduler steered elsewhere — not scarcity. The "
        "julia:multibrot leg's F2 is re-affirmed on CHANGED evidence: v10 read a hard zero "
        "julia-plane supply and that zero is gone, but the three remain the smallest "
        "parameter-plane families and every row came through a gate, not an exhaustion test.")
    out["withheld_note"] = (
        "`withheld` holds partitions the estimator CAN cut and this pass does not adopt. They "
        "appear in `uncalibrated` at the baseline with a reason naming the owner of the "
        "decision, and their derived value is preserved so the owner is not re-deriving it.")

    prev = ROOT / "data" / "v10" / "t_good_derivation.json"
    if prev.exists():
        old = json.loads(prev.read_text(encoding="utf-8"))["adopted"]
        out["vs_v10"] = {p: {"v10": old.get(p), VERSION: out["adopted"].get(p)}
                         for p in sorted(set(old) | set(out["adopted"]))}
        print("\n  v10 -> v11 adopted:")
        for p, d in out["vs_v10"].items():
            print(f"    {p:22s} {d['v10']}  ->  {d[VERSION]}")

    blob = json.dumps(out, indent=2)
    if adopt:
        paths.durable(OUT_REL, mkparents=True).write_text(blob, encoding="utf-8")
        print(f"\nwrote {OUT_REL} (durable) — ADOPTED table; mirror into "
              f"production_seeder.T_GOOD_OVERRIDES in the SAME pass, or the live gate and "
              f"its provenance disagree")
        return 0

    dest = paths.scratch("v11", "t_good_rederived.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(blob, encoding="utf-8")
    print(f"\nwrote {dest} (scratch). {OUT_REL} is UNTOUCHED. Pass --adopt to write it, and "
          f"mirror the new cuts in the same pass.")
    if (ROOT / OUT_REL).exists():
        cur = json.loads((ROOT / OUT_REL).read_text(encoding="utf-8")).get("adopted", {})
        moved = {p: (cur.get(p), out["adopted"].get(p))
                 for p in sorted(set(cur) | set(out["adopted"]))
                 if cur.get(p) != out["adopted"].get(p)}
        print(f"  vs the committed table: "
              + ("IDENTICAL" if not moved else
                 ", ".join(f"{p} {a} -> {b}" for p, (a, b) in moved.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
