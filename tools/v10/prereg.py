#!/usr/bin/env python
r"""v10 certification — the PRE-REGISTRATION. Run and commit this BEFORE any eval runs.

`docs/design/classifier_retrain_protocol.md` §3 says to set the bar before training, and
adds a second obligation earned by v9: **verify the instrument's inputs actually change**,
because a NON-INFERIOR verdict computed on inputs identical to the baseline's is true and
empty. This script writes both — the bars and the instrument check — into
`data/v10/prereg_v10.json`, which `eval_v10.py` READS rather than restates. A bar that
lives in the eval script is a bar that can be edited after seeing the numbers; a bar in a
committed artifact the eval script loads cannot.

WHAT THE INTERVENTION IS, and therefore what the instrument check means here. v9's
intervention was the RENDER PATH (the iteration cap), so the question "did the eval slice's
pixels move?" was the right one, and the answer — zero of 144 census tiles differed — is
why v9's verdict was worthless. v10's intervention is the TRAINING DATA: 1,267 appended
locations, of which 1,265 are train-side. The eval slice's tiles are SUPPOSED to be
unchanged; what has to move is the MODEL. So the check inverts:

  census-144 / floor-526 : tiles byte-identical to v9's, deliberately. These are pure
                           non-regression arms — the frozen instruments, read on a new
                           model. They CANNOT see the intervention's target, because every
                           appended location is a native-plane maneuver view and the census
                           is julia:multibrot. A null here is the expected outcome and is
                           a non-regression result, not a null result about the data.
  uniform-90             : 90 newly rendered tiles, all eval, none in training. This is
                           the ONLY arm that reads the population the appended labels come
                           from, and it is the arm the retrain is actually for. Both models
                           score the SAME tiles, so it is model-vs-model on identical
                           inputs — the cleanest of the three.

Per §3's closing instruction the arm whose inputs moved would normally be ranked PRIMARY.
It is not, and that is Matt's call recorded rather than argued: the census-144 is the
unchanged instrument the v7->v8->v9 chain is comparable on, and breaking that chain to
promote a 90-row arm with 22 positives would trade a comparable series for a noisier read.
The uniform-90 carries its own pre-registered bar and is reported at equal prominence.

THE DEFAULT DOES NOT WRITE THE RECORD, and that is the whole point of the file. `build()`
derives `eval_population`, the instrument-check delta and the uniform-90's power bar FROM the
eval slice — so a re-run after the slice moves silently recomputes the "pre-registered" bar
from post-hoc inputs, and the artifact still says "written before any v10 eval ran". A bar
that lives in the eval script can be edited after seeing the numbers; a bar in a committed
artifact that any re-run rewrites is the same defect one file further out.

So: default prints and writes only to `scratch/prereg/`. `--adopt` writes the record, and
even then it will not REWRITE one — an existing record may only gain APPENDED amendments,
with every other key byte-identical. That is the invariant the AMENDMENTS block below states
in prose ("Append only; never rewrite an entry") made checkable.

  uv run python tools/v10/prereg.py            # print + scratch/prereg/prereg_v10.json
  uv run python tools/v10/prereg.py --show     # print the registered bars and exit
  uv run python tools/v10/prereg.py --adopt    # write data/v10/prereg_v10.json (guarded)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

EVAL_SLICE = ROOT / "data/v10/eval_slice.jsonl"
V9_PLAN = ROOT / "data/v9/plan.jsonl"
V10_PLAN = ROOT / "data/v10/plan.jsonl"
OUT = "data/v10/prereg_v10.json"

# n=144 cannot resolve an AUC gap inside ~0.05. Same margin v8 and v9 were judged on — NOT
# re-chosen for this run, which is the point of reusing it.
NONINF_MARGIN = 0.05


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUC. Used here for ONE purpose: to derive the
    smallest AUC the uniform-90 can distinguish from chance, so that arm's bar is a
    consequence of its power rather than a number someone liked."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(var)


def min_detectable_auc(n_pos: int, n_neg: int, step: float = 0.005) -> float:
    """Smallest AUC whose 95% Hanley-McNeil interval clears 0.50."""
    a = 0.50 + step
    while a < 0.99:
        if a - 1.96 * hanley_mcneil_se(a, n_pos, n_neg) > 0.50:
            return round(a, 3)
        a += step
    return 0.99


def eval_tile_delta() -> dict:
    """Per-instrument: how many of the eval slice's canonical tiles differ between the v9
    plan and the v10 plan. Computed from the plans (identical `out` path AND identical row
    => identical tile; the prefix parity gate in build_plan.py proves that implication)."""
    ev = [json.loads(l) for l in EVAL_SLICE.read_text(encoding="utf-8").splitlines()
          if l.strip()]
    by_src = {}
    v9 = {r["out"]: r for r in
          (json.loads(l) for l in V9_PLAN.read_text(encoding="utf-8").splitlines() if l.strip())}
    v10 = [json.loads(l) for l in V10_PLAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_loc = {}
    for r in v10:
        lid = int(Path(r["out"]).parent.name)
        if "twilight_shifted__id__s1.0000__sh0.0000__ss2.jpg" in r["out"]:
            by_loc[lid] = r
    for row in ev:
        s = row["source"]
        d = by_src.setdefault(s, {"n": 0, "identical_to_v9": 0, "new": 0})
        d["n"] += 1
        r = by_loc.get(row["loc_id"])
        if r is None:
            continue
        o = v9.get(r["out"])
        if o is not None and o == r:
            d["identical_to_v9"] += 1
        else:
            d["new"] += 1
    return by_src


def build() -> dict:
    ev = [json.loads(l) for l in EVAL_SLICE.read_text(encoding="utf-8").splitlines()
          if l.strip()]
    by_src = Counter(r["source"] for r in ev)
    pos = {}
    for src in by_src:
        rows = [r for r in ev if r["source"] == src]
        pos[src] = {"n": len(rows),
                    "n_ge2": sum(1 for r in rows if r["label"] >= 2),
                    "n_ge3": sum(1 for r in rows if r["label"] >= 3),
                    "n_eq4": sum(1 for r in rows if r["label"] == 4)}

    u = pos["maneuver_uniform_v1"]
    u_bar = min_detectable_auc(u["n_ge2"], u["n"] - u["n_ge2"])

    return {
        "version": "v10",
        "written": "before any v10 eval ran; eval_v10.py loads this file rather than "
                   "restating its constants",
        "claim_available": (
            "NON-INFERIORITY on the two frozen instruments (label noise forbids reading "
            "small AUC differences on the >=3 boundary, protocol §3); SEPARATION-vs-chance "
            "on the new instrument, whose bar is derived from its own power below."),
        "baseline": (
            "v8 RE-SCORED on the v10 tiles, not v8's own numbers from eval_results_v8.json. "
            "v8's flat-8000 cache was deleted 2026-07-31, and the deployed system today "
            "renders through the live auto_maxiter policy anyway — so v8-on-v10-tiles IS "
            "the deployed system, and it is what a deploy decision must be made against."),
        "eval_population": pos,
        "instrument_check": {
            "rule": "classifier_retrain_protocol.md §3, the v9 block",
            "intervention": "TRAINING DATA (1,267 appended locations; 1,265 train-side)",
            "why_the_check_inverts": (
                "v9's intervention was the render path, so identical eval tiles meant a "
                "blind instrument. v10's intervention is the data, so identical eval tiles "
                "are CORRECT — what must move is the model. The failure mode to guard "
                "instead is an eval slice that cannot see the population the new labels "
                "come from, which is why the uniform-90 exists."),
            "canonical_tile_delta_vs_v9": eval_tile_delta(),
            "expected": (
                "census-144 and floor-526: 100% identical to v9's tiles. uniform-90: 100% "
                "new, because those locations did not exist in v9. Anything else means the "
                "recipe moved and the whole battery is measuring that instead."),
            "what_the_frozen_arms_cannot_see": (
                "every appended location is a NATIVE-plane maneuver view (mandelbrot / "
                "multibrot3/4/5); the census is julia:multibrot only. A null on the census "
                "is therefore the EXPECTED outcome and reads as non-regression, not as a "
                "null result about the appended data."),
        },
        "arms": {
            "primary_census144": {
                "instrument": "prospect_census, 144 locations",
                "metric": "AUC(label>=3 vs rest), paired DeLong, v10 vs v8 on the same tiles",
                "n": pos["prospect_census"]["n"],
                "n_pos": pos["prospect_census"]["n_ge3"],
                "bar": (f"NON-INFERIOR iff AUC_v10 >= AUC_v8_rescored - {NONINF_MARGIN} AND "
                        f"the paired DeLong does not put v10 significantly below v8 "
                        f"(p < 0.05 AND delta < 0)"),
                "noninf_margin": NONINF_MARGIN,
                "gating": True,
                "style": "identical construction to the v8-vs-v7 and v9-vs-v8 primaries",
            },
            "floor_loose0_v3": {
                "instrument": "loose0_v3_floor, 526 unbiased base-rate mandelbrot locations",
                "metric": "AUC(label>=3 vs rest), paired DeLong, v10 vs v8 on the same tiles",
                "n": pos["loose0_v3_floor"]["n"],
                "n_pos": pos["loose0_v3_floor"]["n_ge3"],
                "bar": (f"NON-INFERIOR iff AUC_v10 >= AUC_v8_rescored - {NONINF_MARGIN} AND "
                        f"the paired DeLong does not put v10 significantly below v8 "
                        f"(p < 0.05 AND delta < 0)"),
                "noninf_margin": NONINF_MARGIN,
                "gating": True,
                "note": ("SYMMETRIC: v8 and v10 both trained on these locations, so neither "
                         "arm is flattered."),
            },
            "new_uniform90": {
                "instrument": "maneuver_uniform_v1, 90 locations — the only "
                              "score-unconditioned draw over the maneuver-view population",
                "metric": "AUC(label>=2 vs rest); AUC + bootstrap CI reported for BOTH "
                          "models, plus paired DeLong",
                "n": u["n"],
                "n_pos": u["n_ge2"],
                "boundary_choice": (
                    ">=2, not >=3: the leg came back 0/90 at the >=3 emission floor, so a "
                    ">=3 read has zero positives and no power at all. >=2 has "
                    f"{u['n_ge2']} positives, which is the only boundary on this "
                    "population with any power."),
                "power": {
                    "method": "Hanley-McNeil SE at n_pos=%d, n_neg=%d" % (
                        u["n_ge2"], u["n"] - u["n_ge2"]),
                    "min_detectable_auc_vs_chance": u_bar,
                    "se_at_that_auc": round(
                        hanley_mcneil_se(u_bar, u["n_ge2"], u["n"] - u["n_ge2"]), 4),
                },
                "bar": (f"SEPARATES iff AUC_v10 >= {u_bar} AND the 95% bootstrap CI's lower "
                        f"bound > 0.50. Below that the arm is UNDERPOWERED, which per "
                        f"protocol §3 means 'label more', NOT 'the model failed'."),
                "separation_bar": u_bar,
                "gating": False,
                "gating_note": (
                    "NOT gating for adoption — it is a 90-row arm and this is its first "
                    "run, so it has no prior version to be non-inferior to. It is the "
                    "reason the retrain exists and is reported at equal prominence; a "
                    "future v11 can make it gating once a v10 number exists to beat."),
                "held_out_scope": (
                    "as of amendment 1: fully held out — this leg touches neither training "
                    "nor checkpoint selection, and is scored only at certification."),
                "v8_baseline_expectation": (
                    "v8 is measured non-separating on this population. That flat result is "
                    "the number to beat, and it is COMPUTED here rather than assumed — "
                    "v8's AUC and CI on these same 90 tiles are reported."),
            },
            "class4_descriptive": {
                "instrument": "the 22 class-4 census locations (all julia:multibrot)",
                "metric": "AUC(label==4 vs rest), reported for v10 and v8",
                "n_class4_eval": pos["prospect_census"]["n_eq4"],
                "bar": None,
                "gating": False,
                "note": ("DESCRIPTIVE, no bar. The 23 appended class-4 locations are ALL "
                         "train-side and must not appear in any eval number; eval_v10.py "
                         "asserts the eval slice holds exactly "
                         f"{pos['prospect_census']['n_eq4']} class-4 rows, all census."),
            },
        },
        "label_noise_discipline": (
            "No small-AUC-difference reads on the >=3 boundary. Only the bars above are "
            "verdicts; every other number in the battery is descriptive and is labeled so."),
        "adoption": (
            "THIS PROMPT DOES NOT ADOPT. ACTIVE_CKPT stays v8 and no threshold file is "
            "touched. Adoption is a separate prompt judged against these bars."),
        "amendments": AMENDMENTS,
    }


# --------------------------------------------------------------------------- #
# Amendments. A pre-registration is only worth the discipline of NOT editing it, so a
# second run against these bars is recorded here BEFORE it happens, with what failed and
# what changed. Append only; never rewrite an entry.
# --------------------------------------------------------------------------- #
AMENDMENTS = [
    {
        "n": 1,
        "date": "2026-08-02",
        "declared": "BEFORE the re-run, and after attempt 1's numbers were known",
        "attempt_1_outcome": {
            "primary_census144": {"v8": 0.7509, "v10": 0.6598, "delta": -0.0911,
                                  "delong_p": 0.0126, "verdict": "INFERIOR"},
            "floor_loose0_v3": {"v8": 0.8673, "v10": 0.8701, "verdict": "NON-INFERIOR"},
            "new_uniform90": {"v8": 0.8483, "v10": 0.8222, "both": "SEPARATE"},
            "frozen_at": "commit c590070; data/v10/eval_results_v10.json as of that commit",
        },
        "what_was_wrong": (
            "NOT the data — the build. Promoting the uniform leg to EVAL also moved the "
            "MODEL-SELECTION objective, because train_resumable selects on not-bad AP over "
            "the WHOLE eval split (cfg['eval_split_is_val']). v8 selected over 670 "
            "locations; attempt 1 selected over 760, 12% of them a population v8's "
            "selection never saw. So the build's own premise — 'the labels are the only "
            "variable' — was false: the criterion that picks the checkpoint moved with "
            "them."),
        "evidence": (
            "tools/v10/diagnose_selection.py: model_last (epoch 40, selected by NO "
            "criterion) scores 0.7634 on census-144, ABOVE v8's 0.7509, against "
            "model_best's 0.6598 — paired DeLong p=0.0024. The run passed through "
            "checkpoints that would have certified and the changed objective rejected "
            "them. model_last was NOT adopted: swapping in a checkpoint chosen by nothing, "
            "after seeing the numbers, is the bar-moving this file exists to prevent."),
        "change": (
            "Model selection is restricted to the v8-COMPARABLE eval subset — "
            "prospect_census + loose0_v3_floor, 670 locations, exactly v8's and v9's "
            "selection population. The uniform-90 becomes a fully held-out instrument: it "
            "is scored at certification only and touches neither training nor checkpoint "
            "selection. This is strictly better for that arm's credibility than attempt 1, "
            "where it influenced the pick."),
        "what_is_NOT_changed": (
            "Every bar, margin and boundary above is untouched, and so is the corpus, the "
            "cache, the recipe and the baseline. Re-running the same bars is the reason "
            "this amendment exists rather than a quiet retrain: attempt 1's failure is on "
            "the record at commit c590070 and stays there."),
        "authorized_by": "Matt, 2026-08-02 — asked explicitly before the re-run",
        "if_it_fails_again": (
            "then the selection objective was NOT the cause and the appended native-plane "
            "data genuinely costs julia:multibrot accuracy. That is a real finding about "
            "the corpus mix and the answer is a mix decision, not a third retrain."),
    },
]


def _diff_keys(old: dict, new: dict) -> list:
    """Top-level keys, EXCLUDING `amendments`, on which the two records disagree."""
    keys = (set(old) | set(new)) - {"amendments"}
    return sorted(k for k in keys if old.get(k) != new.get(k))


def _amendments_extend(old: list, new: list) -> bool:
    """True iff `new` is `old` plus zero or more APPENDED entries. Append-only, made
    checkable: a rewritten amendment is not an extension and must not adopt."""
    return len(new) >= len(old) and new[:len(old)] == old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print and exit; write nothing")
    ap.add_argument("--adopt", action="store_true",
                    help="write data/v10/prereg_v10.json. Refuses to REWRITE an existing "
                         "record: only appended amendments are permitted.")
    a = ap.parse_args()
    d = build()
    print("=" * 80)
    print("v10 CERTIFICATION — PRE-REGISTERED BARS (written before any eval ran)")
    print("=" * 80)
    for name, arm in d["arms"].items():
        print(f"\n  {name}   n={arm.get('n')}  n_pos={arm.get('n_pos')}  "
              f"gating={arm['gating']}")
        print(f"    metric: {arm['metric']}")
        print(f"    bar   : {arm['bar']}")
    print("\n  instrument check — canonical eval tile delta vs v9:")
    for src, v in d["instrument_check"]["canonical_tile_delta_vs_v9"].items():
        print(f"    {src:<24} n={v['n']:>4}  identical_to_v9={v['identical_to_v9']:>4}  "
              f"new={v['new']:>4}")
    if a.show:
        return 0

    blob = json.dumps(d, indent=2)
    record = paths.durable(OUT)
    old = json.loads(record.read_text(encoding="utf-8")) if record.exists() else None

    if old is not None:
        changed = _diff_keys(old, d)
        appended = _amendments_extend(old.get("amendments", []), d.get("amendments", []))
        if changed or not appended:
            print(f"\nREFUSING to rewrite {OUT}: this run does not merely APPEND an "
                  f"amendment to the committed pre-registration.")
            if changed:
                print(f"  keys that differ: {', '.join(changed)}")
            if not appended:
                print(f"  amendments: committed {len(old.get('amendments', []))} entries; "
                      f"this run's list is not an extension of them")
            print("  The bars were registered before v10 was evaluated. `build()` derives "
                  "them FROM the eval slice, so a slice that has moved since produces a "
                  "post-hoc bar wearing a pre-hoc label. Nothing was written.")
            return 2
        if not a.adopt:
            print(f"\n{OUT} exists and this run only appends amendments — pass --adopt to "
                  f"write it.")

    prev = paths.scratch("prereg", "prereg_v10.json")
    prev.parent.mkdir(parents=True, exist_ok=True)
    prev.write_text(blob, encoding="utf-8")
    print(f"\nwrote {prev} (preview)")

    if not a.adopt:
        state = "does not exist" if old is None else "exists"
        print(f"NOT written: {OUT} ({state}). Pass --adopt to write the committed record.")
        return 0

    p = paths.durable(OUT, mkparents=True)
    p.write_text(blob, encoding="utf-8")
    print(f"WROTE {OUT}  — commit this BEFORE running eval_v10.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
