#!/usr/bin/env python
r"""Write `data/v11/adoption_record.json` — the v11 flip's own record.

WHY THIS IS NOT IN `build_record.json`. v11's build adopted nothing: its record says so in
`deploy_note` ("ACTIVE_CKPT NOT switched; no threshold touched"). The rollback ladder and the
revert-together set are facts about the ADOPTION, which happened on 2026-08-08, six days
after the build. v10 kept them in `build_metadata.json` only because v10's build wrote a
`rollback_ladder` block stamped "RECORD ONLY — this build adopts nothing", and a
"what a future adoption would have to revert" note is a different object from "what this
adoption did revert". Two records, two dates, neither rewriting the other.

EVERY VERSION TOKEN IS READ, NOT DECLARED. Same rule as `tools/v10/build_manifest.
rollback_ladder`: a rollback note that states the deployed version from memory is the same
species of bug as a metadata file with a hardcoded `True` — it outlives the fact it records.
The ladder head, the checkpoint, the t_good stamp, the keeper stamp and the tau_h stamp are
all read off the modules and artifacts that own them, so a set that has drifted apart cannot
be written as a coherent one.

THE LADDER IS TWO RUNGS AND THAT IS A POLICY OUTCOME, not attrition. The standing weights
retention policy (docs/design/storage_classes.md § weights retention) tracks ACTIVE +
PREVIOUS per model family; v5..v9 de-tracked at this flip, so naming them as rungs would
name a rollback a fresh clone cannot perform.

  uv run python tools/v11/adopt_v11.py            # print, no write
  uv run python tools/v11/adopt_v11.py --write    # write data/v11/adoption_record.json
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

import paths                       # noqa: E402
import production_pins as pp       # noqa: E402

OUT_REL = "data/v11/adoption_record.json"
ADOPTED_ON = "2026-08-08"

# The rung BELOW the live head, read off the pin module rather than spelled here.
PREVIOUS = Path(pp.V10_CKPT_ROLLBACK).parent.name


def _stamp(entry: dict):
    """One COUPLED_ARTIFACTS stamp, or the exception text if it cannot be read.

    Never a silent None: an artifact the registry names and this record cannot read is
    exactly the state the record exists to make visible."""
    try:
        return pp.coupled_stamp(entry)
    except Exception as e:                                   # noqa: BLE001
        return f"UNREADABLE: {type(e).__name__}: {e}"


def rollback_ladder() -> dict:
    """What a rollback off v11 would have to revert, READ from the live pins."""
    return {
        "adopted_on": ADOPTED_ON,
        "note": ("The v11 ADOPTION record. Unlike data/v10/build_metadata.json:"
                 "rollback_ladder — which was a build-time note about a flip that had not "
                 "happened — this is written by the flip itself."),
        "deployed_now": pp.ACTIVE_VERSION,
        "ladder": [pp.ACTIVE_VERSION, PREVIOUS],
        "why_two_rungs": (
            "The rollback ladder names only rungs whose weight a fresh clone receives. The "
            "standing weights-retention policy (docs/design/storage_classes.md) tracks ACTIVE "
            "+ PREVIOUS per model family and de-tracks everything older or rejected, so "
            "v5/v6/v7/v8/v9 left the index at this flip and left the ladder with them. "
            "Emergency copies of all five sit unreferenced outside the repo; they are not "
            "resolvable from any pin, on purpose."),
        "why_not_v9": (
            "v9 was built, evaluated and STAGED but never adopted, so it was never a rung "
            "even while it was tracked: a rollback to a version that was never deployed "
            "restores a gate that never ran."),
        "must_revert_together": [
            {"what": e["what"], "why": e["why"], "now": _stamp(e), "guard": e["guard"]}
            for e in pp.COUPLED_ARTIFACTS
        ],
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    lad = rollback_ladder()
    stamps = {e["now"] for e in lad["must_revert_together"] if e["now"] is not None}
    lad["coherence"] = (f"all stamped entries agree on {pp.ACTIVE_VERSION!r}"
                        if stamps == {pp.ACTIVE_VERSION}
                        else f"INCOHERENT — stamps present: {sorted(stamps)}")
    record = {
        "adoption": pp.ACTIVE_VERSION,
        "adopted_on": ADOPTED_ON,
        "checkpoint": pp.ACTIVE_CKPT,
        "certified": ("non-inferior on all three pre-registered gating arms (census-144, "
                      "floor-526, uniform-90); bars in data/v11/prereg_v11.json, results in "
                      "data/v11/eval_results_v11.json"),
        "wrote": {
            "t_good": "data/v11/t_good_derivation.json (+ production_seeder.T_GOOD_OVERRIDES)",
            "keeper_cuts": "data/atlas/keeper_cuts.json",
            "tau_h": (f"data/atlas/tau_h_base_{pp.ACTIVE_VERSION}.json (+ "
                      "steered_frontier.TAU_H_FIDELITY_BASE / TAU_H_CAMPAIGN_FLOOR)"),
        },
        "known_flip_cost": (
            "The emission LIBRARY SEED is unreconstructable until the next real run. It is "
            "built from run-local counts and embeddings under the head that produced them, "
            "so a head flip retires it and only a v11-era run can rebuild it. Known cost of "
            "any flip, not a defect of this one."),
        "rollback_ladder": lad,
    }
    blob = json.dumps(record, indent=2)
    print(blob)
    if "--write" in argv:
        paths.durable(OUT_REL, mkparents=True).write_text(blob, encoding="utf-8")
        print(f"\nwrote {OUT_REL} (durable)")
    else:
        print(f"\n(not written — pass --write to land {OUT_REL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
