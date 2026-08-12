#!/usr/bin/env python
r"""Write `data/palettes/autolevel_adoption.json` — the band auto-level flip's own record.

WHY A RECORD AT ALL. The flip is one boolean, and one boolean in a source file says WHAT the
tree does and nothing about what it was decided against. This is the same object a head flip
writes (`tools/v11/adopt_v11.py`): the date, the evidence the decision was taken on, what the
flip does NOT touch, and how to undo it. Sized to fit — there is no ladder here, no coupled
artifact set, and nothing to re-score.

EVERY STATE TOKEN IS READ, NOT DECLARED. The switch state, the operator version and the whole
identity of the reference record (version, sha256, n, the three bands) come off the live
modules and the committed file, so a record that disagrees with the tree cannot be written —
and `test_autolevel.py::test_the_adoption_record_matches_the_live_switch_and_reference` fails
if the tree moves out from under an already-written one. A hardcoded `"switch": "on"` here
would be the metadata file that outlives what it records (CLAUDE.md).

  uv run python tools/palettes/adopt_autolevel.py            # print, no write
  uv run python tools/palettes/adopt_autolevel.py --write    # land the record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                          # noqa: E402
from tools.palettes import autolevel as AL            # noqa: E402

OUT_REL = "data/palettes/autolevel_adoption.json"
ADOPTED_ON = "2026-08-11"
BUILD_COMMIT = "43a9328"        # the build+stage commit this record adopts


def record() -> dict:
    ref = AL.load_reference()
    return {
        "adoption": AL.OPERATOR_VERSION,
        "adopted_on": ADOPTED_ON,
        "decided_by": "Matt, on the build-stage report and its production sheet",
        "switch": {
            "where": "tools/palettes/autolevel.py::SWITCH_DEFAULT",
            "default_now": AL.SWITCH_DEFAULT,
            "env_override": AL.SWITCH_ENV,
            "read_at": "call time, via autolevel.enabled()",
        },
        "reference_record": {
            "path": ref.get("_path"),
            "version": ref.get("version"),
            "schema": ref.get("schema"),
            "sha256": ref.get("_sha256"),
            "derived": ref.get("derived"),
            "n_images": ref.get("n_images"),
            "bands": {k: list(v) for k, v in AL.bands(ref).items()},
            "note": ("The band is READ off this file, never off this record. A re-derivation "
                     "that moves an edge is a new band and therefore a new adoption "
                     "question — which is what the coherence test turns into a red suite."),
        },
        "evidence": {
            "build_stage_commit": BUILD_COMMIT,
            "report": "scratch/autolevel_build_stage_report.md",
            "sheet": "scratch/autolevel_verify/sheet_autolevel_production.png",
            "verdict": "scratch/autolevel_build_stage_verdict.json",
            "what_it_showed": (
                "12 production-path rows: 9 identity rows at max |delta| = 0 exactly (the "
                "operator returns the base render's own array), 3 acting rows all replaying "
                "to max |delta| = 0 from the stamp alone, pre-screen vs production render "
                "disagreeing on 0/12."),
            "rebuild": "uv run python -u tools/palettes/build_autolevel_verify_sheet.py",
            "caveat": ("The evidence lives in scratch/ and scratch is wiped freely. The sheet "
                       "rebuilds from the command above; the numbers above are the record."),
        },
        "wired_sites": [
            "tools/mining/deploy_tail.py::render_pure (pure modes)",
            "tools/mining/deploy_tail.py::render_rust (composites; level=(kind != 'direct'))",
            "tools/emission/build_emission_diversity_v1.py::render_smooth (base carrier)",
        ],
        "does_not_touch": [
            "the mining gate and every floor in tools/emission/floors.py",
            "the Rust<->Python LUT seam (the surgery ends in a stop list; the bake is "
            "colormap.build_lut on the Python tail and render-one --colormaps on the Rust "
            "tail, both unchanged)",
            "the direct-trap family, which is palette-indifferent and excluded where the "
            "kind is known rather than disabled inside the render",
            "any stored label row, ledger or release record written before this date",
        ],
        "not_covered": (
            "`enrich --mode render` (the mining harvest colorizes inside Rust, src/enrich.rs) "
            "is the one palette-mapped production path the operator does not reach: it needs "
            "a measured base render before it can pick a curve. Named in the build-stage "
            "report's NOT-done list and still open at adoption."),
        "known_flip_cost": (
            "Renders stop being byte-comparable across the flip: an out-of-band location now "
            "ships a different image than it would have on 2026-08-10, and rows carry a new "
            "`autolevel` key. The 2,439-crop exposure census was measured against the earlier "
            "35-image band and is stale in BOTH directions (band and switch); the 48-image "
            "band's own share was read on a 50-crop pre-screen (34/50 identity), not a census."),
        "rollback": {
            "one_run": f"{AL.SWITCH_ENV}=0 in the run's environment — no source edit",
            "permanent": ("set SWITCH_DEFAULT back to False and re-run this writer; the OFF "
                          "path is a live contract (test_switch_off_is_the_pre_operator_path), "
                          "not dead code, so the revert is the boolean and nothing else"),
            "artifacts_to_revert": [],
            "why_none": ("Nothing was re-scored, re-derived or rewritten by this flip. The "
                         "operator is additive per render: rows produced after it carry a "
                         "stamp, rows produced before it do not."),
        },
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    blob = json.dumps(record(), indent=2)
    print(blob)
    if "--write" in argv:
        paths.durable(OUT_REL, mkparents=True).write_text(blob, encoding="utf-8")
        print(f"\nwrote {OUT_REL} (durable)")
    else:
        print(f"\n(not written — pass --write to land {OUT_REL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
