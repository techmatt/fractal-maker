r"""adopt_head.py — write a stage-2 head's ADOPTION RECORD, both heads, one shape.

THE OBJECT. `data/<family>/<version>/adoption_record.json`: what was adopted, on what
evidence, what moved with it, and what a rollback would have to revert. It is the stage-2
sibling of `tools/v11/adopt_v11.py` (the location head's) and copies its two load-bearing
rules rather than inventing a second shape:

  EVERY VERSION TOKEN AND EVERY LIVE CUT IS READ, NOT DECLARED. The pin, the gate, the two
  floors, the suggestion cuts and the rollback rung all come off the modules that own them,
  so a set that has drifted apart cannot be written as a coherent one. A record that states
  the deployed version from memory is the same species of bug as a metadata file with a
  hardcoded `True`.

  THE MEASURED HALF IS QUOTED FROM COMMITTED RECORDS, never restated. The blind re-verdict,
  the anchoring price and the volume-matched precisions are read out of
  `sheet_{d,e}_reverdict/report.json` and `volume_match_*.json` by key. A number pasted here
  would outlive the run it describes.

WHAT IS PROSE AND WHY. `rationale` and `not_established` are Matt's decision and its stated
limits (prompts/flip_29.md). They are the one part of an adoption that is not derivable from
the tree — an adoption is a judgement, and the judgement has to be written down somewhere or
the record is just a diff.

  uv run python tools/scoring/adopt_head.py wallpaper           # print, no write
  uv run python tools/scoring/adopt_head.py mining --write      # write the record
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import paths                                       # noqa: E402
from tools.emission import floors as F             # noqa: E402

ADOPTED_ON = "2026-08-11"
PROMPT = "prompts/flip_29.md"

# The regenerability argument, stated once because it is the SAME argument on both heads and
# it is the whole of why two losing challengers were adopted. Matt, 2026-08-11.
REGENERABILITY = (
    "Both incumbents are pre-corpus-discipline heads that cannot be regenerated — their "
    "training-data provenance is incomplete — while both challengers are from-scratch "
    "retrains off fully tracked corpora. Comparable-plus-regenerable is sufficient: a head "
    "that cannot be rebuilt is a rung that decays, and holding an unrebuildable incumbent "
    "against a rebuildable challenger that is not measurably worse trades a permanent cost "
    "for no measured gain.")


@dataclass(frozen=True)
class AdoptSpec:
    key: str
    family: str                 # data/<family>/
    pin_module: str
    reverdict_rel: str
    volume_match_rel: str
    cut_names: tuple
    rationale: tuple
    not_established: tuple = ()


WALLPAPER = AdoptSpec(
    key="wallpaper", family="wallpaper_head", pin_module="tools.wallpaper.wallpaper_pins",
    reverdict_rel="data/wallpaper_head/sheet_d_reverdict/report.json",
    volume_match_rel="data/wallpaper_head/v4b/volume_match_wallpaper.json",
    cut_names=("wallpaper_release", "wallpaper_pool"),
    rationale=(
        "The (28) verdict made v3 the winner on an eval union whose MOTIVATING arm was "
        "sheet A's minibrot bucket — a correction sheet v3 itself served, 84.9% of whose "
        "labels came back equal to the suggestion. Sheet D re-drew that arm BLIND at fresh "
        "locations (protocol §2b) and priced the anchoring at -0.224 AUC>=3: v3's 0.965 on "
        "the anchored bucket against 0.741 on the blind one. On the blind slice v4b is not "
        "distinguishable from v3 at >=3 and is AHEAD at the job boundary.",
        "SEED PICK: best blind sheet-D AUC>=4. Production sees post-floor 3/4 material, so "
        ">=4 is the boundary the job actually turns on, and all five v4b seeds beat v3's "
        "0.510 there. Seed 1 took it at 0.609 (band 0.567 ± 0.032).",
        REGENERABILITY),
    not_established=(
        "THE SEED PICK SPENDS A 197-ROW SELECTION. Sheet D was the only unanchored read of "
        "the minibrot population that will ever exist, and choosing among five seeds on it "
        "consumes it: the picked seed's 0.609 is a selected maximum, not a held-out number, "
        "and the band is the honest read of what a fresh seed would do. Nothing here is a "
        "held-out estimate of v4b/seed_1 any more.",
        "ONE SLICE, ~200 ROWS. A null on sheet D is 'not distinguishable at this n', not "
        "'identical'.",),
)

MINING = AdoptSpec(
    key="mining", family="render_mode_head", pin_module="tools.mining.mining_pins",
    reverdict_rel="data/render_mode_head/sheet_e_reverdict/report.json",
    volume_match_rel="data/render_mode_head/v3/volume_match_mining.json",
    cut_names=("mining_release", "mining_pool"),
    rationale=(
        "The (28)/(28b) clause-(a) verdict was measured entirely against a baseline the "
        "labels are coupled to: ALL THREE labeled render-mode batches are correction sheets "
        "served with mining v1's suggested tier prefilled and ordered by its score, so there "
        "is no clean arm anywhere in that corpus (protocol §2b, 'a whole corpus can be "
        "anchored, and the mining corpus is'). Sheet E is the unanchored slice — 150 blind "
        "(location, mode) pairs no prior sheet served. Clause (a) passes 0/7 there for v3, "
        "38 of the 40 anchored failures across the five arms do not reproduce, and v1's "
        "anchoring price at the boundary every arm lost on is -0.278 (pooled AUC>=2 0.953 "
        "anchored -> 0.676 blind).",
        "The anchor-broken motivating slices favour v3: busy_fp +0.142 AP>=3 [+0.024, "
        "+0.260] and +0.366 median AUC>=2, rare_palette +0.100 AUC>=3 [+0.014, +0.190], "
        "v1_unseen_locations +0.098 AUC>=3 [+0.010, +0.191].",
        "At the release floor's own volume — 129 of 827 rows, the cut restated to hold that "
        "volume fixed — precision>=3 goes 0.636 -> 0.760.",
        REGENERABILITY),
    not_established=(
        "PER-MODE ANCHORED COMPARISONS ARE NOT ESTABLISHED EITHER WAY. The (28) clause-(a) "
        "cells that v3 failed (curv_linear, direct_trap_lines, direct_trap_ring, "
        "direct_trap_screen) were measured on anchored labels and are therefore not evidence "
        "against v3; sheet E's per-mode cells are 25 rows each with 0-2 positives and are "
        "not evidence for it. Neither direction is shown. THE CORRECTION LOOP IS THE SAFETY "
        "NET: a mode v3 is wrong about surfaces as corrections on the next sheet.",
        "THE BASE-RATE AUDIT IS DELIBERATELY NOT HERE. Sheet E's blind >=3 rate is ~4% "
        "against 21-46% on the anchored sheets, so the calibrations may be 5-11x off. That "
        "is next session's own prompt. Volume-matching preserves current volumes through "
        "this flip regardless of how it lands.",),
)

SPECS = {s.key: s for s in (WALLPAPER, MINING)}


def _read(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(
            f"adopt_head: {rel} is missing. An adoption record may not cite evidence it "
            f"cannot read — that is the 'reports absent where it means could not look' "
            f"failure the four rules name.")
    return json.loads(p.read_text(encoding="utf-8"))


def pin_block(spec: AdoptSpec) -> dict:
    """Everything the pin module owns, READ off it."""
    import importlib
    m = importlib.import_module(spec.pin_module)
    if spec.key == "wallpaper":
        return {"module": spec.pin_module, "ckpt": m.HEAD_CKPT_REL,
                "version": m.HEAD_VERSION, "gate": m.GATE_THRESHOLD,
                "rollback": m.V3_CKPT_ROLLBACK}
    return {"module": spec.pin_module, "ckpt": m.ACTIVE_MINING_CKPT,
            "version": m.HEAD_VERSION, "gate": m.MINING_GATE_THRESHOLD,
            "gate_version": m.MINING_GATE_VERSION, "lock": m.LOCK_PATH,
            "rollback": m.MINING_V1_ROLLBACK}


def suggestion_cuts(spec: AdoptSpec) -> dict:
    """The re-derived correction-sheet cuts, read off their owners AND re-derived, so a
    record cannot claim a constant its own deriver disagrees with."""
    if spec.key == "wallpaper":
        from tools.wallpaper import suggest_tier as ST                 # noqa: PLC0415
        return {
            "CUTS": {"value": list(ST.CUTS), "rederived": list(ST.derive_cuts()),
                     "slice": ST.DERIVATION["slice"], "n": ST.DERIVATION["n"],
                     "was": ST.DERIVATION["supersedes"]["cuts"]},
            "INTAKE_CUTS": {"value": list(ST.INTAKE_CUTS),
                            "rederived": list(ST.derive_intake_cuts()),
                            "slice": ST.INTAKE_DERIVATION["slice"],
                            "n": ST.INTAKE_DERIVATION["n"],
                            "was": ST.INTAKE_DERIVATION["supersedes"]["cuts"]},
        }
    from tools.mining import suggest_tier_mining as MT                 # noqa: PLC0415
    return {"CUTS": {"value": list(MT.CUTS), "rederived": list(MT.derive_cuts()),
                     "slice": MT.FIT_SLICE_BATCH, "n": len(MT.fit_slice()[0]),
                     "was": [1.317, 1.8727]}}


def build(spec: AdoptSpec) -> dict:
    pin = pin_block(spec)
    vm = _read(spec.volume_match_rel)
    rv = _read(spec.reverdict_rel)
    by_name = {c["name"]: c for c in vm["cuts"]}
    live = {f.name: f for f in F.ALL_FLOORS}

    cuts = {}
    for name in spec.cut_names:
        c, f = by_name[name], live[name]
        if abs(f.value - c["incoming_value"]) > 1e-12:
            raise SystemExit(
                f"[adopt-head] {name} is {f.value} in floors.py but the volume-match record "
                f"placed it at {c['incoming_value']}. The record may not claim a restatement "
                f"nobody applied.")
        cuts[name] = {
            "old": c["outgoing_value"], "new": f.value,
            "restated_how": "VOLUME-MATCHED (classifier_retrain_protocol.md §5a) — the score "
                            "that admits the same NUMBER of reference-pool rows",
            "matched_volume": c["matched_volume"], "n": c["n"],
            "matched_rate": c["matched_rate"],
            "precision_ge3_old": c["outgoing"]["precision_ge3"],
            "precision_ge3_new": c["incoming"]["precision_ge3"],
            "volume_preserved": c["volume_preserved"],
        }

    return {
        "adoption": pin["version"],
        "family": spec.family,
        "adopted_on": ADOPTED_ON,
        "prompt": PROMPT,
        "checkpoint": pin["ckpt"],
        "pin": pin,
        "rationale": list(spec.rationale),
        "not_established": list(spec.not_established),
        "evidence": {
            "blind_reverdict": {"record": spec.reverdict_rel,
                                "generated": rv.get("generated"),
                                "command": rv.get("command"),
                                "slice": rv.get("slice")},
            "volume_match": {"record": spec.volume_match_rel,
                             "generated": vm["generated"],
                             "reference_pool": {k: vm["reference_pool"][k]
                                                for k in ("what", "n", "n_locations",
                                                          "tiers", "base_rate_ge3")}},
        },
        "moved_with_the_pin": {
            "cuts": cuts,
            "suggestion_cuts": suggestion_cuts(spec),
            "floors_summary": F.summary(),
        },
        "rollback_ladder": {
            "ladder": [pin["version"], Path(pin["rollback"]).parent.name],
            "why_two_rungs": (
                "ACTIVE + PREVIOUS per model family (docs/design/storage_classes.md § "
                "weights retention). Nothing de-tracked at this flip: each family gained a "
                "rung and the outgoing head became the rung below."),
            "must_revert_together": [
                {"what": f"{spec.pin_module}", "why": "the pin; HEAD_VERSION and every "
                                                      "stamp derive from it"},
                *[{"what": f"tools/emission/floors.{n.upper()}"
                           if n.endswith("pool") else f"{spec.pin_module} gate threshold",
                   "why": f"{n} is a point on this head's probability scale",
                   "revert_to": cuts[n]["old"]} for n in spec.cut_names],
                {"what": ("tools/wallpaper/suggest_tier.{CUTS,INTAKE_CUTS}"
                          if spec.key == "wallpaper"
                          else "tools/mining/suggest_tier_mining.CUTS"),
                 "why": "cutpoints on `expected_tier`, a sum of CORN marginals — as "
                        "scale-bound as a probability floor"},
                *([{"what": "data/render_mode_head/v3/mining_gate_lock.json",
                    "why": "read_lock() refuses when the pin moves off the head it describes; "
                           "v1's lock is still on disk and becomes live again on a rollback"}]
                  if spec.key == "mining" else []),
            ],
        },
        "known_flip_cost": (
            "Every accumulated record that carries this head's score — "
            "data/emission/release_records/*.jsonl and "
            "data/emission/mining_gate_reports/*.jsonl — holds numbers on the OUTGOING "
            "head's scale. They are records of decisions taken, not inputs to a decision, so "
            "they are not rewritten; the `floor` column beside each score says which cut it "
            "was judged against, which is what keeps an old row readable. A cross-flip "
            "precision read off them must not pool the two eras."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head", choices=sorted(SPECS))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    spec = SPECS[args.head]
    rec = build(spec)
    for k, v in rec["moved_with_the_pin"]["suggestion_cuts"].items():
        if v["value"] != v["rederived"]:
            raise SystemExit(f"[adopt-head] {k} frozen {v['value']} != deriver "
                             f"{v['rederived']} — refusing to record a drifted constant.")
    blob = json.dumps(rec, indent=2) + "\n"
    print(blob)
    out_rel = f"data/{spec.family}/{rec['adoption']}/adoption_record.json"
    if args.write:
        paths.durable(out_rel, mkparents=True).write_text(blob, encoding="utf-8")
        print(f"wrote {out_rel} (durable)")
    else:
        print(f"(not written — pass --write to land {out_rel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
