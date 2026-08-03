#!/usr/bin/env python
r"""Derive v10 per-partition t_good — the ADOPTED discovery table at the v10 flip.

t_good is a cut on ONE head's `P(>=3)`; v10's probability scale is not v8's, so the v8 table
is a set of numbers about nothing on a v10 gate (protocol §4). This re-derives it.

THE ESTIMATOR IS IMPORTED, NOT COPIED. `scoring/derive_t_good.build_table` is the derivation —
grid, F_beta-argmax, tie-break-toward-higher-t, LOO-OOF, plateau width, the >=15-positive
sufficiency floor and the UNCALIBRATED stamping all run from that module. A copied deriver is
how two thresholds that are supposed to be comparable stop being comparable.

TWO THINGS THIS PASS DECIDES, both required by protocol §4 and neither inherited:

1. THE POPULATION RULE — **one instrument per partition, never pooled.**
   v10's eval slice carries THREE unbiased instruments where v8 carried two:

     prospect_census      144  julia:multibrot{3,4,5}   (byte-identical to v8's/v9's)
     loose0_v3_floor      526  mandelbrot               (byte-identical to v8's/v9's)
     maneuver_uniform_v1   90  mandelbrot 12 + native multibrot3/4/5 24/25/29   (NEW)

   Each partition is cut on its own instrument. The only rows this drops are the **12
   maneuver_uniform mandelbrot rows**: mandelbrot already has a dedicated instrument, and
   folding a second population's base rate into one precision denominator is a pooled cut by
   another name. It is not a rounding decision — pooling them moves the argmax 0.03 -> 0.08
   and collapses the OOF F0.5 from 0.357 to 0.100, which is the instability speaking, not a
   better threshold. Keeping mandelbrot on the identical 526 rows v8 used is also what makes
   the v8 -> v10 threshold move readable as a head change rather than a population change.

   The native multibrot partitions KEEP their uniform rows even though they carry zero keeper
   positives, because "we looked at 24 unbiased draws and none was a keeper" and "we have
   never had an unbiased draw" are different states and the artifact must not print the same
   string for both. Their `reason` says so.

2. THE OBJECTIVE, re-read from CURRENT supply (recall where scarce, precision where
   abundant). The 2026-08 supply record, which is the only supply evidence generated since
   the v8 table was cut:

     * mandelbrot     — ABUNDANT. The 2026-08-01 supply crawl drew 156 mandelbrot atoms and
       the 2026-08-02 label-seeded harvest produced 374 mandelbrot seeds, both without
       exhausting their sources. A miss still costs nothing. -> F0.5 (precision), UNCHANGED.
     * native multibrot{3,4,5} — ABUNDANT on the same evidence (the crawl drew 181/193/200).
       Recorded as F0.5, though all three are UNCALIBRATED (0 keeper positives), so the
       objective never fires; it is recorded so the supply read is not lost.
     * julia:multibrot{3,4,5} — SCARCE, and MORE scarce than at v8. BOTH 2026-08 supply
       efforts were 100% native-plane: zero julia-plane locations were drawn or harvested,
       so this family's supply has not grown at all since the v8 table was cut. A miss is
       still gone. -> F2 (recall), UNCHANGED.

   Unchanged is not uninspected: the choices are re-affirmed against 2026-08 evidence that
   did not exist when v8 was cut, and the julia leg is re-affirmed *because* the intervening
   supply work bought it nothing.

  uv run python tools/v10/derive_t_good_v10.py
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
from partitions import partition_of   # noqa: E402  THE fractal_type -> partition map

VERSION = "v10"
EVAL_REL = "data/v10/eval_scores_v10.jsonl"
EVAL = ROOT / EVAL_REL
OUT_REL = "data/v10/t_good_derivation.json"

# --- population rule: partition -> the ONE instrument it is cut on (see §1 above) --------- #
INSTRUMENT = {
    "mandelbrot": "loose0_v3_floor",
    "julia:multibrot3": "prospect_census",
    "julia:multibrot4": "prospect_census",
    "julia:multibrot5": "prospect_census",
    "multibrot3": "maneuver_uniform_v1",
    "multibrot4": "maneuver_uniform_v1",
    "multibrot5": "maneuver_uniform_v1",
}

# --- objective, re-read from 2026-08 supply (see §2 above) -------------------------------- #
OBJECTIVE = {
    "mandelbrot": 0.5,          # abundant (crawl 156 + harvest 374 in 2026-08)
    "multibrot3": 0.5,          # abundant (crawl 181) — recorded; partition is UNCALIBRATED
    "multibrot4": 0.5,          # abundant (crawl 193) — recorded; partition is UNCALIBRATED
    "multibrot5": 0.5,          # abundant (crawl 200) — recorded; partition is UNCALIBRATED
    "julia:multibrot3": 2.0,    # scarce — zero julia-plane supply drawn in 2026-08
    "julia:multibrot4": 2.0,
    "julia:multibrot5": 2.0,
}

UNCAL_REASON = {
    "multibrot3": "unbiased eval slice present (maneuver_uniform_v1) but 0 keeper positives",
    "multibrot4": "unbiased eval slice present (maneuver_uniform_v1) but 0 keeper positives",
    "multibrot5": "unbiased eval slice present (maneuver_uniform_v1) but 0 keeper positives",
    "julia:mandelbrot": "no unbiased eval slice",
    "phoenix": "no unbiased eval slice",
}


def select_population(rows) -> tuple[list, dict]:
    """Apply the one-instrument-per-partition rule; return (kept_rows, dropped_report)."""
    kept, dropped = [], {}
    for r in rows:
        part = partition_of(r["fractal_type"], r["fractal_type"])
        want = INSTRUMENT.get(part)
        if want is None or r.get("source") == want:
            kept.append(r)
        else:
            dropped.setdefault(f"{part}<-{r.get('source')}", 0)
            dropped[f"{part}<-{r.get('source')}"] += 1
    return kept, dropped


def main() -> int:
    if not EVAL.exists():
        sys.exit(f"missing {EVAL} — run tools/v10/eval_v10.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, dropped = select_population(rows)
    print(f"[population] one instrument per partition: {len(kept)}/{len(rows)} rows kept")
    for k, v in sorted(dropped.items()):
        print(f"  dropped {v:4d}  {k}  (partition is cut on {INSTRUMENT[k.split('<-')[0]]})")
    print()

    out = est.build_table(kept, version=VERSION, eval_slice=EVAL_REL,
                          objective=OBJECTIVE, uncal_reason=UNCAL_REASON)
    out["population_rule"] = (
        "ONE INSTRUMENT PER PARTITION, never pooled. " + json.dumps(INSTRUMENT)
        + f". Dropped: {json.dumps(dropped)} — mandelbrot is cut on the identical 526 "
          "loose0_v3_floor rows v8 used, so the v8->v10 move reads as a head change.")
    out["objective_basis"] = (
        "Re-read from 2026-08 supply, not inherited. Crawl 2026-08-01 drew mandelbrot 156 / "
        "multibrot3 181 / multibrot4 193 / multibrot5 200; label-seeded harvest 2026-08-02 "
        "produced mandelbrot 374 / mb5 53 / mb4 44 / mb3 40. BOTH were 100% native-plane: "
        "zero julia:multibrot supply was generated since the v8 table was cut, so that "
        "family is if anything scarcer than it was. Choices re-affirmed, not copied.")

    prev = ROOT / "data" / "v8" / "t_good_derivation.json"
    if prev.exists():
        old = json.loads(prev.read_text(encoding="utf-8"))["adopted"]
        out["vs_v8"] = {p: {"v8": old.get(p), VERSION: out["adopted"].get(p)}
                        for p in sorted(set(old) | set(out["adopted"]))}
        print("\n  v8 -> v10 adopted:")
        for p, d in out["vs_v8"].items():
            print(f"    {p:20s} {d['v8']}  ->  {d[VERSION]}")

    paths.durable(OUT_REL, mkparents=True).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_REL} (durable) — ADOPTED table; mirror into "
          f"production_seeder.T_GOOD_OVERRIDES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
