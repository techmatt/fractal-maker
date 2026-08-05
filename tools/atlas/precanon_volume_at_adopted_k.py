#!/usr/bin/env python
r"""precanon_volume_at_adopted_k.py — size a production run under the ADOPTED dedup rule.

WHY THIS EXISTS SEPARATELY FROM THE REPLAY. `precanon_minfw_replay.py` answered "what does
the SCALE flip do", so it holds K at the run's own 1.5 and moves only max->min; its
self-check would fail at any other K, by design. The adoption then moved BOTH constants
(0.25 x min, 2026-08-04), and the M0/M1/M2 admission bracket in the adoption record was
therefore measured at a radius 6x larger than the one production now runs. That bracket is
a floor, not a forecast. This module re-runs the same replay at the live
`production_seeder.DEDUP_K` and reports the counterfactual beside it.

WHAT IT DOES NOT DO. It measures the rule on a FIXED candidate stream. A real run under the
new rule admits differently, fires different julia hooks and therefore sources a different
stream — every caveat in `precanon_minfw_replay`'s docstring applies unchanged, and the
first production run's own telemetry remains the actual read.

The run's-rule rows are printed first and MUST reproduce the replay's published numbers
(2151 / 1184 / 943 newly surviving at M0 / M2 / M1); they are the harness check, and the
0.25 rows mean nothing without them.

  uv run python tools/atlas/precanon_volume_at_adopted_k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import production_seeder as ps          # noqa: E402
import precanon_minfw_replay as R       # noqa: E402


def main() -> int:
    rows, _ledger = R.load_population(R.RUN_DIR)
    print(f"population {len(rows)} harvest checks of {R.RUN_DIR.name}; "
          f"replay K {R.REPLAY_K} x {ps.RETIRED_DEDUP_SCALE} (the run's) vs "
          f"live {ps.DEDUP_K} x {ps.DEDUP_SCALE} (adopted)")

    base = R.replay(rows, "max", admit_frac=0.0, strict=True)
    print(f"self-check at the run's rule: precanon_dup {base['precanon_dup']} / "
          f"admitted {base['admitted']} -> OK")
    obs_rate = base["admitted"] / base["survived_precanon"]

    run_k = R.REPLAY_K
    try:
        for k in (run_k, ps.DEDUP_K):
            R.REPLAY_K = k                      # the replay reads this at call time
            for tag, frac in (("M0_determinate", 0.0),
                              ("M2_observed_rate", obs_rate),
                              ("M1_saturated", 1.0)):
                m = R.replay(rows, "min", admit_frac=frac, strict=False)
                newly = len(m["newly_survived"])
                print(f"  K={k:<5g} {tag:<17s} survivors {m['survived_precanon']:>5d} "
                      f"(newly {newly:>5d} of {base['precanon_dup']}, "
                      f"{100 * newly / base['precanon_dup']:.1f}%)  "
                      f"determinate admissions {m['admitted']}  "
                      f"canonical renders {m['survived_precanon']} "
                      f"(the run made {base['survived_precanon']})")
    finally:
        R.REPLAY_K = run_k                      # never leave the module re-pinned
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
