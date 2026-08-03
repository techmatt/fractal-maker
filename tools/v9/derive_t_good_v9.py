#!/usr/bin/env python
r"""Derive v9 per-partition t_good — STAGED, never adopted, and v9 is not a rollback rung.

Every threshold derived before the cap raise sits on a moved score distribution: v9 reads
raised-cap renders, so its `P(>=3)` is not v8's `P(>=3)` and a v8 cut applied to a v9 gate is
a number about nothing. This re-derives the table on the SAME objectives, from the durable v9
eval slice.

  * SAME OBJECTIVES AS v8, deliberately. F0.5 (precision-weighted) for mandelbrot, F2
    (recall-weighted) for julia:multibrot{3,4,5}. Re-deriving on a different objective would
    confound "the cap moved" with "the objective moved", and reading the cap is the whole
    point of this pass. Taken by reference from the estimator, not restated.
  * POPULATION: the whole slice, as v8 did — v9's slice carries one unbiased instrument per
    partition, so the one-instrument rule v10 introduced selects the identical rows.

STAGED. This writes `data/v9/t_good_derivation.json` and **does not touch**
`production_seeder.T_GOOD_OVERRIDES`. v9 was built, staged and skipped: the deployed head went
v8 -> v10 and `data/v10/build_metadata.json:rollback_ladder` records v9 as explicitly NOT a
rung, so nothing here can ever gate. It is kept as the recipe behind a committed artifact.

THE ESTIMATOR IS IMPORTED, NOT COPIED — `tools/scoring/derive_t_good.build_table`. This file
used to carry its own copy of the per-partition loop, calling `best_t(p_nb, p_gd, y, beta)`;
when the sweep was aligned to the SERVED decode on 2026-08-02 that signature grew a `p_gr`
argument and this module became a **TypeError waiting for whoever ran it next**. That is what
a copied deriver buys: the copy did not go red with the original, it went stale beside it.

RE-RUNNING DOES NOT REPRODUCE `data/v9/t_good_derivation.json`, for the same reason v8's does
not: the committed v9 table was swept under an AND, which is not the rule `corn_decode` serves
on a K=4 head. The artifact stays as the record of the staged pass. `main()` therefore prints
and writes to scratch; `--adopt` is required to overwrite it, and there is no live reason to.

  uv run python tools/v9/derive_t_good_v9.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import derive_t_good as est   # noqa: E402  THE estimator, imported not copied
import paths                  # noqa: E402

VERSION = "v9"
EVAL_REL = "data/v9/eval_scores_v9.jsonl"
EVAL = ROOT / EVAL_REL
OUT_REL = "data/v9/t_good_derivation.json"

OBJECTIVE = est.OBJECTIVE        # unchanged from v8, deliberately — see the docstring
UNCAL_REASON: dict = {}


def select_population(rows) -> tuple[list, dict]:
    """v9 takes the whole slice (one unbiased instrument per partition); nothing is dropped."""
    return list(rows), {}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    adopt = "--adopt" in argv
    if not EVAL.exists():
        sys.exit(f"missing {EVAL} — run tools/v9/eval_v9.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, _ = select_population(rows)
    out = est.build_table(kept, version=VERSION, eval_slice=EVAL_REL,
                          objective=OBJECTIVE, uncal_reason=UNCAL_REASON)
    out["objective_unchanged_from_v8"] = True
    out["objective_unchanged_rationale"] = (
        "Re-deriving on a different objective would confound 'the cap moved' with 'the "
        "objective moved'. Same F0.5/F2 assignment as v8, deliberately.")
    out["status"] = "STAGED — NOT adopted; v9 is not a rollback rung"

    v8_path = ROOT / "data/v8/t_good_derivation.json"
    if v8_path.exists():
        v8 = json.loads(v8_path.read_text(encoding="utf-8")).get("adopted", {})
        moved = {fam: {"v8": v8[fam], "v9": t, "delta": round(t - v8[fam], 4)}
                 for fam, t in out["adopted"].items() if fam in v8}
        out["vs_v8"] = moved
        if moved:
            print("\n  vs v8 (both tables' own predicates apply — see the docstring):")
            for fam, m in sorted(moved.items()):
                print(f"    {fam:20s} v8 {m['v8']:.2f} -> v9 {m['v9']:.2f}  ({m['delta']:+.2f})")

    if adopt:
        paths.durable(OUT_REL, mkparents=True).write_text(json.dumps(out, indent=2),
                                                          encoding="utf-8")
        print(f"\nwrote {OUT_REL} (durable) — STAGED table; v9 can never gate")
    else:
        dest = paths.scratch("v9", "t_good_rederived.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {dest} (scratch). {OUT_REL} is UNTOUCHED — it records the staged pass "
              f"under the superseded AND predicate. Pass --adopt to overwrite it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
