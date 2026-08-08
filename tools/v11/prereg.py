#!/usr/bin/env python
r"""Write `data/v11/prereg_v11.json` — the v11 bars, BEFORE any v11 number exists.

`measurement_practice.md` and protocol §3: a bar living in the eval script is a bar that can
be edited after seeing the results; a bar in a committed artifact the eval script LOADS
cannot. So every threshold, margin and power-derived separation bar is computed here, from
LABEL COUNTS and closed-form power only, and `tools/v11/eval_v11.py` reads them.

Everything this file computes is a property of the corpus and the split — no model is
loaded, no tile is scored. Run it, read it, commit it, then run the eval.

WHAT IS NEW ABOUT THE v11 BATTERY, in one place:

  * The BASELINE is v10 re-scored on the v11 canonical tiles, for v10's own reason: v10 is
    the deployed head, and pairing both arms on identical inputs is what leaves the model as
    the only difference. A DIAGNOSTIC arm quantifies the tile-path change (v10 on its own
    v10 cache tile vs v10 on the v11 canonical tile) so a reader can tell how much of any
    gap could be the renderer rather than the head.
  * The eval split has TWO ROLES. `instrument` (1,050, score-unconditioned, unbiased) is
    where every verdict is read. `holdout` (1,810, a stratified random draw over the
    remaining split groups, biased exactly as training is) carries NO verdict against v10 —
    1,426 of its 1,810 rows were v10 TRAINING rows, so any v10-vs-v11 comparison there is
    structurally rigged for v10. It is used for two things v11 built it for: the 3|4
    boundary on rows absent from BOTH heads' training, and the first per-partition
    calibration reads.
  * The MOTIVATING slice is the correction sitting. Its 500 rows postdate v10's build
    entirely, so all 500 are out-of-sample for v10; 87 of them fell to v11's holdout and are
    out-of-sample for v11 too. THOSE 87 are the instrument. The other 413 are v11-train and
    are reported CONTAMINATED, never as a verdict.

  uv run python tools/v11/prereg.py [--out data/v11/prereg_v11.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tools/corpus", "tools/scoring"):
    sys.path.insert(0, str(ROOT / sub))

import partitions as P  # noqa: E402
import paths  # noqa: E402

MANIFEST = "data/v11/manifest.jsonl"
V10_MANIFEST = ROOT / "data" / "v10" / "manifest.jsonl"
V10_RESULTS = ROOT / "data" / "v10" / "eval_results_v10.json"
OUT = ROOT / "data" / "v11" / "prereg_v11.json"

CENSUS, FLOOR = "prospect_census", "loose0_v3_floor"
UNIFORM, Q4_UNIFORM = "maneuver_uniform_v1", "q4_uniform_eval"
CORRECTION_BATCHES = ("2026-08-07_label_run_correction_v1",
                      "2026-08-07_steady_state_v2_backfill_v1")
NONINF_MARGIN = 0.05
PINV_INVESTIGATE_DELTA = 0.10
CLASS4_DECODE_T = 0.50          # sigma(logit2) >= t  =>  "class 4"
MIN_POS = 15                    # protocol §4's calibration floor


def rows(rel):
    p = paths.bulk(rel)
    if not p.exists():
        raise SystemExit(f"missing {rel} -> {p}")
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def counts(rs) -> dict:
    c = collections.Counter(r["label"] for r in rs)
    return {"n": len(rs), "n_ge2": sum(v for k, v in c.items() if k >= 2),
            "n_ge3": sum(v for k, v in c.items() if k >= 3), "n_eq4": c[4],
            "by_label": {str(k): c[k] for k in (1, 2, 3, 4)}}


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUC — the same closed form v10's uniform-90 bar
    was derived from, restated here only because a bar must be computable from the counts."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                      + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def min_detectable_auc(n_pos: int, n_neg: int, step: float = 0.01) -> tuple[float, float]:
    """Smallest AUC on a 0.01 grid whose 95% Hanley-McNeil lower bound clears chance.

    This is the separation bar's definition, and it is a FUNCTION OF THE COUNTS — which is
    what lets it be pre-registered without seeing a score. Reproduces v10's 0.64 at
    (22, 68) exactly, which is the check that this is the same instrument, not a new one."""
    a = 0.50
    while a < 1.0:
        a = round(a + step, 4)
        if a - 1.96 * hanley_mcneil_se(a, n_pos, n_neg) > 0.50:
            return a, round(hanley_mcneil_se(a, n_pos, n_neg), 4)
    raise SystemExit(f"no detectable AUC at n_pos={n_pos}, n_neg={n_neg}")


def key(r) -> tuple:
    return (r["fractal_type"], r["cx"], r["cy"], r["fw"],
            str(r.get("c_re")), str(r.get("c_im")))


def build() -> dict:
    man = rows(MANIFEST)
    v10_keys = {key(r) for r in
                (json.loads(l) for l in V10_MANIFEST.open(encoding="utf-8") if l.strip())}
    v10res = json.loads(V10_RESULTS.read_text(encoding="utf-8"))

    by_src = collections.defaultdict(list)
    for r in man:
        by_src[r["source"]].append(r)
    inst = {s: counts(by_src[s]) for s in (CENSUS, FLOOR, UNIFORM, Q4_UNIFORM)}
    for s, c in inst.items():
        if c["n"] == 0:
            raise SystemExit(f"instrument {s} is empty in {MANIFEST}")

    corr = [r for r in man if any(b in r["source"] for b in CORRECTION_BATCHES)]
    corr_ho = [r for r in corr if r["eval_role"] == "holdout"]
    corr_tr = [r for r in corr if r["split"] == "train"]
    leaked = [r["loc_id"] for r in corr if key(r) in v10_keys]
    if leaked:
        raise SystemExit(f"{len(leaked)} correction-sitting rows are in v10's corpus — the "
                         f"'out-of-sample for both heads' premise of the motivating slice "
                         f"is false, e.g. {leaked[:5]}")

    holdout = [r for r in man if r["eval_role"] == "holdout"]
    part_ho = collections.defaultdict(list)
    for r in holdout:
        part_ho[P.partition_of_row(r, r["fractal_type"])].append(r)
    new_parts = {p: counts(part_ho[p]) for p in ("julia:mandelbrot", "phoenix")}

    u_bar, u_se = min_detectable_auc(inst[UNIFORM]["n_ge2"],
                                     inst[UNIFORM]["n"] - inst[UNIFORM]["n_ge2"])
    q_bar, q_se = min_detectable_auc(inst[Q4_UNIFORM]["n_ge2"],
                                     inst[Q4_UNIFORM]["n"] - inst[Q4_UNIFORM]["n_ge2"])
    if u_bar != 0.64:
        raise SystemExit(f"the uniform-90 separation bar re-derives to {u_bar}, not v10's "
                         f"0.64 — the instrument's label counts moved, so it is not the "
                         f"same instrument and the standing read is not standing")

    noninf = (f"NON-INFERIOR iff AUC_v11 >= AUC_v10 - {NONINF_MARGIN} AND the paired DeLong "
              f"does not put v11 significantly below v10 (p < 0.05 AND delta < 0)")

    return {
        "version": "v11",
        "written": "BEFORE any v11 eval ran; tools/v11/eval_v11.py loads this file rather "
                   "than restating its constants. No model is loaded here — every number "
                   "below is a label count or a closed-form power calculation.",
        "adoption": "THIS PROMPT DOES NOT ADOPT. ACTIVE_CKPT stays v10, t_good is not "
                    "re-derived, no keeper cut or ledger is touched. Adoption is a separate "
                    "prompt judged against these bars.",
        "baseline": (
            "v10 RE-SCORED on the v11 canonical tiles, not v10's own numbers from "
            "data/v10/eval_results_v10.json. v10 is the deployed head; pairing both arms on "
            "identical inputs is what leaves the MODEL as the only difference between them. "
            "v10's own tiles are a different render path (v4-render-batch direct vs "
            "crop-batch's extended field), which the diagnostic arm below quantifies."),
        "canonical_view": {
            "cell": "twilight_shifted / identity framing / antialiased:lanczos3 / q85",
            "why_it_is_produced": (
                "v4..v10 fanned out as a PRODUCT so this cell existed for every location. "
                "v11 draws its 32 tiles independently: tile 0 is twilight_shifted at the "
                "identity framing by the recipe floor, but its AA level is a 50/50 draw, so "
                "only 1,448 of 2,860 eval locations carry it. tools/v11/build_eval_canon.py "
                "replays those tile-0 rows at antialiased/q85 — the cache's own code path, "
                "verified byte-identical on the 38 locations that already had the cell."),
            "manifest": "data/v11/eval_canon_manifest.jsonl (bulk)"},
        "instrument_check": {
            "rule": "classifier_retrain_protocol.md §3, the v9 block",
            "intervention": ("CORPUS + SPLIT RULE + AUG RECIPE — 11,303 labeled locations "
                             "against v10's 8,382, a stratified randomized holdout over "
                             "leakage-closure groups instead of v8's frozen prefix, and 32 "
                             "independently drawn tiles instead of a 24-slot product."),
            "why_the_check_inverts": (
                "v9's intervention was the render path, so identical eval tiles meant a "
                "blind instrument. v11's is the data and the split, so what must move is "
                "the MODEL — and the arms are paired on identical tiles by construction. "
                "The failure mode to guard instead is a tile-path change masquerading as a "
                "model change, which is what the diagnostic arm measures."),
            "diagnostic_tile_path": {
                "what": "v10 scored on its OWN v10 cache canonical tile vs v10 scored on "
                        "the v11 canonical tile, census-144, AUC(label>=3)",
                "expectation": "|delta| <= 0.02. The two paths were measured at functional "
                               "parity (tools/v11/parity_crop_mode.py: 0/30 decision flips "
                               "at the identity crop), so a larger move means the battery "
                               "is reading the renderer and every verdict below is void.",
                "gating": False, "abort_if": "abs(delta) > 0.02"}},
        "eval_population": {
            "instrument": {"prospect_census": inst[CENSUS], "loose0_v3_floor": inst[FLOOR],
                           "maneuver_uniform_v1": inst[UNIFORM],
                           "q4_uniform_eval": inst[Q4_UNIFORM]},
            "holdout_partitions_first_calibration": new_parts,
            "correction_sitting": {
                "holdout_out_of_sample_for_both": counts(corr_ho),
                "train_side_v11": counts(corr_tr),
                "all": counts(corr),
                "in_v10_corpus": len(leaked)}},
        "arms": {
            "primary_census144": {
                "instrument": f"{CENSUS}, {inst[CENSUS]['n']} locations",
                "metric": "AUC(label>=3 vs rest), paired DeLong, v11 vs v10 on the same tiles",
                "n": inst[CENSUS]["n"], "n_pos": inst[CENSUS]["n_ge3"],
                "bar": noninf, "noninf_margin": NONINF_MARGIN, "gating": True,
                "style": "identical construction to the v8-vs-v7, v9-vs-v8 and v10-vs-v8 "
                         "primaries; eval-side in every build since v7",
                "reads": "julia:multibrot only. v11 changes the split everywhere, so unlike "
                         "v10 this arm is not blind to the intervention — but it is still "
                         "the frozen comparability instrument, not the motivating one."},
            "floor_loose0_v3": {
                "instrument": f"{FLOOR}, {inst[FLOOR]['n']} unbiased base-rate mandelbrot "
                              f"locations",
                "metric": "AUC(label>=3 vs rest), paired DeLong, v11 vs v10 on the same tiles",
                "n": inst[FLOOR]["n"], "n_pos": inst[FLOOR]["n_ge3"],
                "bar": noninf, "noninf_margin": NONINF_MARGIN, "gating": True,
                "note": "SYMMETRIC: eval-side in both builds, so neither arm is flattered."},
            "uniform90": {
                "instrument": f"{UNIFORM}, {inst[UNIFORM]['n']} locations — the "
                              f"score-unconditioned draw over the maneuver-view population",
                "metric": "AUC(label>=2 vs rest); AUC + bootstrap CI for both models, plus "
                          "paired DeLong",
                "n": inst[UNIFORM]["n"], "n_pos": inst[UNIFORM]["n_ge2"],
                "boundary_choice": ">=2, not >=3: 0 positives at >=3 (v10's reasoning, and "
                                   "the counts are unchanged)",
                "power": {"method": "Hanley-McNeil SE",
                          "min_detectable_auc_vs_chance": u_bar, "se_at_that_auc": u_se,
                          "reproduces_v10_bar": True},
                "bar": (f"SEPARATES iff AUC_v11 >= {u_bar} AND the 95% bootstrap CI's lower "
                        f"bound > 0.50; AND {noninf}"),
                "separation_bar": u_bar, "noninf_margin": NONINF_MARGIN, "gating": True,
                "gating_note": "GATING now, unlike in v10's battery, for the reason v10's "
                               "prereg gave: 'a future v11 can make it gating once a v10 "
                               "number exists to beat.' It does — 0.8282, SEPARATES.",
                "v10_measured": v10res["new_uniform90"]["auc_cand"]},
            "q4_uniform290": {
                "instrument": f"{Q4_UNIFORM}, {inst[Q4_UNIFORM]['n']} locations — "
                              f"registered 2026-08-03, FIRST eval use",
                "metric": "AUC(label>=2 vs rest); AUC + bootstrap CI for both models",
                "n": inst[Q4_UNIFORM]["n"], "n_pos": inst[Q4_UNIFORM]["n_ge2"],
                "boundary_choice": f">=2: {inst[Q4_UNIFORM]['n_ge3']} positives at >=3 is "
                                   f"below any usable power; >=2 has "
                                   f"{inst[Q4_UNIFORM]['n_ge2']}",
                "power": {"method": "Hanley-McNeil SE",
                          "min_detectable_auc_vs_chance": q_bar, "se_at_that_auc": q_se},
                "bar": f"SEPARATES iff AUC >= {q_bar} AND the 95% CI's lower bound > 0.50",
                "separation_bar": q_bar, "gating": False,
                "gating_note": "NOT gating: this is the arm's first run and it has no prior "
                               "version to be non-inferior to — exactly the position the "
                               "uniform-90 was in at v10. Reported for BOTH heads because "
                               "these 290 locations postdate v10's build and so are "
                               "out-of-sample for it as well.",
                "clean_for_both": True},
            "motivating_class4_correction87": {
                "instrument": (f"the {counts(corr_ho)['n']} correction-sitting locations "
                               f"({' + '.join(CORRECTION_BATCHES)}) that v11's stratified "
                               f"holdout drew into eval"),
                "why_this_population": (
                    "the prompt names 'the correction-sitting rows'. All 500 postdate v10's "
                    "build, so all 500 are out-of-sample for v10 — but 413 are v11 TRAIN, "
                    "and an in-sample-for-v11 vs out-of-sample-for-v10 comparison is rigged "
                    "for v11 in exactly the direction the claim runs. These 87 are the "
                    "subset out-of-sample for BOTH."),
                "n": counts(corr_ho)["n"], "n_pos_class4": counts(corr_ho)["n_eq4"],
                "observed_class4_rate": round(counts(corr_ho)["n_eq4"]
                                              / counts(corr_ho)["n"], 4),
                "defect_under_test": (
                    "v10's class-4 CUTPOINT is loose while its ORDERING is sound — of the "
                    "320 class-4 decodes the sheet served as prefills, 172 survived as "
                    "human 4s (53.8%) and 96.6% survived as >=3, and 68% of all corrections "
                    "sat on the 3|4 boundary alone (commit a66dd56)."),
                "cutpoint_metric": (f"the decode rule sigma(logit_2) >= {CLASS4_DECODE_T} => "
                                    f"'class 4': precision, recall, F1 and predicted-4 rate, "
                                    f"for v10 and v11 on the same tiles"),
                "cutpoint_bar": (
                    f"TIGHTENED iff precision_v11 > precision_v10 AND the predicted-4 rate "
                    f"moves toward the observed rate "
                    f"({round(counts(corr_ho)['n_eq4']/counts(corr_ho)['n'], 4)}), i.e. "
                    f"|rate_v11 - observed| < |rate_v10 - observed|. Direction "
                    f"pre-registered, magnitude not."),
                "ordering_metric": "AUC(label==4 vs rest), paired DeLong, same 87 tiles",
                "ordering_bar": (f"NOT DAMAGED iff AUC_v11 >= AUC_v10 - {NONINF_MARGIN} AND "
                                 f"the paired DeLong does not put v11 significantly below "
                                 f"v10. A cutpoint fix bought with ordering damage FAILS "
                                 f"here and must be reported as a failure."),
                "calibration_metric": "mean predicted P(label>=4) vs the observed rate",
                "noninf_margin": NONINF_MARGIN, "class4_decode_t": CLASS4_DECODE_T,
                "gating": False,
                "gating_note": "NOT gating for adoption — it is an 87-row arm on a biased "
                               "draw. It is the arm the retrain is FOR and is reported at "
                               "equal prominence.",
                "companion_contaminated": (
                    f"the same reads over all {counts(corr)['n']} correction rows, stamped "
                    f"CONTAMINATED: {counts(corr_tr)['n']} are v11 TRAIN, so v11's number "
                    f"there is in-sample and v10's is not. Descriptive only, never a verdict.")},
            "class4_census_descriptive": {
                "instrument": f"the {inst[CENSUS]['n_eq4']} class-4 census locations",
                "metric": "AUC(label==4 vs rest) for v11 and v10",
                "n_class4_eval_census": inst[CENSUS]["n_eq4"],
                "bar": None, "gating": False,
                "note": "DESCRIPTIVE, no bar. Carried forward from v10's battery so the "
                        "julia:multibrot 3|4 read stays continuous across versions."},
            "palette_invariance": {
                "instrument": f"{CENSUS} under twilight_shifted vs the 8 held-out palettes",
                "metric": "Spearman(twilight, palette) per palette; mean / range / pooled",
                "held_out_palettes": json.loads(
                    (ROOT / "data/v11/aug_recipe.json").read_text(encoding="utf-8")
                )["palettes"]["held_out"],
                "held_out_asserted_equal_to_v9_v10": True,
                "bar": None, "gating": False,
                "v10_measured_mean": v10res["palette_invariance"]["mean_spearman"],
                "investigate_delta": PINV_INVESTIGATE_DELTA,
                "read_rule": (f"INVESTIGATE iff |mean_v11 - "
                              f"{v10res['palette_invariance']['mean_spearman']}| > "
                              f"{PINV_INVESTIGATE_DELTA} — a move that large is not a "
                              f"corpus effect. Otherwise DESCRIPTIVE."),
                "note": "the 8 palettes are held out of the DRAW POOL, so no v11 cache tile "
                        "uses one; the battery renders them fresh at the canonical geometry "
                        "and the live cap, as v10's did."},
            "per_partition_calibration_first_reads": {
                "instrument": "the v11 HOLDOUT rows of julia:mandelbrot and phoenix — "
                              "partitions that have never had a calibration read, because "
                              "before v11's grouped random split they had no eval "
                              "population at all",
                "populations": new_parts, "min_pos": MIN_POS,
                "metric": "base rate, reliability (ECE over deciles of P(label>=3)), Brier, "
                          "and the F_beta-argmax over the P(>=3) grid with its plateau "
                          "width, at BOTH beta=0.5 and beta=2",
                "bar": None, "gating": False,
                "reported_for": "v11 ONLY as a first instrument; v10's numbers on the same "
                                "rows are printed CONTAMINATED (v10 trained on much of this "
                                "population) and are not a comparison.",
                "why_no_beta_is_chosen": (
                    "protocol §4 sets beta per family from SUPPLY, and neither of these two "
                    "has that argument made. Both are reported so the choice is a later "
                    "decision with the numbers already in hand."),
                "explicitly_not_adopted": (
                    "no t_good is derived, no threshold is written, production_seeder."
                    "T_GOOD_OVERRIDES is not touched. These are reads, not cuts."),
                "holdout_caveat": (
                    "the holdout is a stratified random draw over the split groups, biased "
                    "exactly as training is. Precision/recall on it are statements about "
                    "the ranker over the population training is drawn from — NOT base "
                    "rates, which may only be read off the score-unconditioned instruments.")},
        },
        "label_noise_discipline": (
            "No small-AUC-difference reads on the >=3 boundary. Only the bars above are "
            "verdicts; every other number in the battery is descriptive and labeled so. A "
            "wash is a reportable outcome and per protocol §3 means 'label more', not "
            "'the model failed'."),
        "no_cherry_picking": (
            "The arms above are the whole battery. Every one is reported with its verdict "
            "whatever the result, and no per-slice arm is added after the fact."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    doc = build()
    Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    for name, arm in doc["arms"].items():
        print(f"  {name:<42} n={arm.get('n')} gating={arm.get('gating')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
