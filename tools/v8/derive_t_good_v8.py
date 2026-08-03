#!/usr/bin/env python
r"""Derive v8 per-partition t_good — v8's population + objective over the shared estimator.

THE ESTIMATOR IS IMPORTED, NOT COPIED: `tools/scoring/derive_t_good.build_table` is the
derivation (grid, F_beta-argmax, tie-break, LOO-OOF, plateau, sufficiency floor, UNCALIBRATED
stamping). This module is only what is v8's own — its eval slice, its population rule and its
objective — which is the same shape `tools/v10/derive_t_good_v10.py` has. Until 2026-08-02
this file WAS the estimator and every other version imported it from here; the estimator moved
to `tools/scoring/` when it outlived the version in its name.

  * POPULATION: every row of the slice. v8 predates the one-instrument-per-partition rule
    (protocol §4), which only bites when a partition has two unbiased instruments — v8's
    slice has one each (`prospect_census` for julia:multibrot{3,4,5}, `loose0_v3_floor` for
    mandelbrot), so "take everything" and "one instrument each" select the identical rows.
    `select_population` is here rather than absent so the interface matches v10's and the
    rerun gate in `tools/scoring/test_t_good_adoption.py` can drive either version.
  * OBJECTIVE: the estimator's default assignment, which is the one v8 adopted — F0.5 for
    mandelbrot (abundant supply -> precision), F2 for julia:multibrot{3,4,5} (saturating
    supply -> recall). Taken by reference, not restated, so there is one copy.

RE-RUNNING THIS DOES NOT REPRODUCE `data/v8/t_good_derivation.json`, ON PURPOSE. The
committed v8 table was swept under an AND (`p_ge2>=0.5 & p_ge3>=t`), which is not the rule
`corn_decode` serves on a K=4 head; through the aligned estimator v8's mandelbrot cut is 0.14,
not the committed 0.85. The artifact is left as the record of what v8 actually served. See
`tools/v8/test_t_good_sweep_decode.py::test_the_v8_anchor_is_the_known_divergence_and_is_left_alone`
— and note that a rollback to v8 (the one-flip anchor) must RE-DERIVE its table, not copy it.

  uv run python tools/v8/derive_t_good_v8.py
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

VERSION = "v8"
EVAL_REL = "data/v8/eval_scores_v8.jsonl"
EVAL = ROOT / EVAL_REL
OUT_REL = "data/v8/t_good_derivation.json"

# v8 adopted the estimator's default per-partition objective — referenced, not restated.
OBJECTIVE = est.OBJECTIVE
UNCAL_REASON: dict = {}


def select_population(rows) -> tuple[list, dict]:
    """v8 takes the whole slice; nothing is dropped. Returns (kept, dropped_report) to match
    the interface v10 introduced, so one rerun gate drives every version."""
    return list(rows), {}


def main(argv=None) -> int:
    """Re-derive and PRINT. Writing the durable artifact takes `--adopt`, because the
    committed one is a RECORD of what v8 served under the superseded AND and re-deriving
    over it would destroy that record for a version that is not even live. `--adopt` is a
    rollback action: it is correct only in the same pass that points ACTIVE_CKPT at v8."""
    argv = sys.argv[1:] if argv is None else argv
    adopt = "--adopt" in argv
    if not EVAL.exists():
        sys.exit(f"missing {EVAL} — run tools/v8/eval_v8.py first (freezes the eval scores)")
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, _ = select_population(rows)
    out = est.build_table(kept, version=VERSION, eval_slice=EVAL_REL,
                          objective=OBJECTIVE, uncal_reason=UNCAL_REASON)
    if adopt:
        paths.durable(OUT_REL, mkparents=True).write_text(json.dumps(out, indent=2),
                                                          encoding="utf-8")
        print(f"\nwrote {OUT_REL} (durable) — mirror into production_seeder.T_GOOD_OVERRIDES; "
              f"this is only correct in a pass that also points ACTIVE_CKPT at v8")
    else:
        dest = paths.scratch("v8", "t_good_rederived.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {dest} (scratch). {OUT_REL} is UNTOUCHED — it records what v8 served "
              f"under the superseded AND predicate. Pass --adopt to overwrite it (rollback only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
