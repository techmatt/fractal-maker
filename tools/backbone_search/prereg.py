#!/usr/bin/env python
"""Write the backbone-comparison PRE-REGISTRATION. Run once, BEFORE any arm is scored.

Same mechanism as `tools/v11/prereg.py`: the metrics, populations, slices, honesty rule
and round-2 advancement rule go into a committed artifact, and `eval_arms.py` LOADS them
rather than restating them. A bar in an eval script can be edited after seeing the
numbers; a bar in a committed artifact the script reads cannot.

No model is loaded here. Every number below is a label count off `data/v11/manifest.jsonl`
or a cost projection off `scratch/backbone_search/cost_smoke.json` — nothing that could be
an eval result.

  uv run python tools/backbone_search/prereg.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import partitions as P  # noqa: E402
import paths  # noqa: E402
from backbone_search.arms import ARMS, CONTROL  # noqa: E402

OUT_REL = "data/backbone_search/prereg_backbone_v1.json"
SMOKE = "backbone_search/cost_smoke.json"
SELECTION_SOURCES = ("prospect_census", "loose0_v3_floor")
MIN_POS_SLICE = 10          # a slice with fewer positives at a cutpoint gets NO verdict


def label_block(rows):
    by = collections.Counter(r["label"] for r in rows)
    return {"n": len(rows), "n_ge2": sum(r["label"] >= 2 for r in rows),
            "n_ge3": sum(r["label"] >= 3 for r in rows),
            "n_ge4": sum(r["label"] >= 4 for r in rows),
            "by_label": {str(k): by.get(k, 0) for k in (1, 2, 3, 4)}}


def main():
    rows = [json.loads(l) for l in
            paths.bulk("data/v11/manifest.jsonl").open(encoding="utf-8") if l.strip()]
    ev = [r for r in rows if r["split"] == "eval"]
    primary = [r for r in ev if r["source"] not in SELECTION_SOURCES]
    selection = [r for r in ev if r["source"] in SELECTION_SOURCES]

    by_part = collections.defaultdict(list)
    for r in primary:
        by_part[P.partition_of(r["fractal_type"])].append(r)

    smoke_p = paths.scratch(*SMOKE.split("/"))
    smoke = json.loads(smoke_p.read_text()) if smoke_p.exists() else {}

    doc = {
      "study": "backbone_search_v1",
      "written": "BEFORE any arm was trained or scored. tools/backbone_search/eval_arms.py "
                 "LOADS this file; it restates no threshold, population or rule.",
      "amendments": [
        {"n": 1, "when": "2026-08-14, during round 1, before ANY arm was scored",
         "who": "Matt", "what": "round 2 cut from per-arm seed bands to control-only seed "
         "replication; the study's question narrowed from ranking to screening. Full text "
         "and what it gives up: round_2_rule.AMENDED.",
         "why_it_is_not_moving_a_bar": "No eval number existed when it was made — the "
         "metrics, populations, slices and honesty rule below are untouched, and the "
         "amendment REMOVES evidence rather than lowering a bar."}],
      "adoption": "NOTHING IS ADOPTED. This is measurement: no production_pins edit, no "
                  "ACTIVE_CKPT move, no floor restatement, no ledger rescore. If an arm "
                  "ever becomes a flip candidate the Winner Rule and "
                  "classifier_retrain_protocol.md §5 apply then, not here.",
      "design_law": {
        "one_variable": "The backbone, and only the backbone. Every behavioural "
                        "hyperparameter is read out of data/classifier/v11/model_best.pt"
                        "[config] at train time and asserted unchanged "
                        "(train_arm.assert_recipe_untouched).",
        "frozen": ["the v11 crop-batch aug cache (361,696 tiles, NOT rebuilt)",
                   "the v11 registered split — identical train/eval populations across "
                   "arms, so every comparison is PAIRED on the same locations",
                   "the deploy transform: stretch to 384x224, bicubic, no jitter. Both "
                   "geometry AND interpolation are v11's, not the arm's timm data config "
                   "(all 7 arms happen to resolve bicubic anyway — cost_smoke.json)",
                   "CORN ordinal head, K=4, Sum sigma(logit_k)",
                   "the model-selection objective: max not-bad AP over the frozen 670 "
                   "census+floor, identical for every arm (a controlled variable)"],
        "moves_with_the_backbone": {
          "mean_std": "Normalization belongs to the pretrained weights — feeding in1k "
                      "statistics to a 0.5/0.5-normalized checkpoint is a handicap, not a "
                      "control. Stamped per arm.",
          "head_pooling": "timm's own classifier for that backbone at num_classes=K-1=3; "
                          "penultimate dim varies, everything downstream of features is "
                          "identical.",
          "vit_pos_embed": "vit_small_p16 is created with img_size=(224,384) so timm "
                           "resamples the pretrained pos-embed to a 14x24 grid. The deploy "
                           "transform is NOT changed to suit it.",
          "grad_checkpointing": "effnetv2_s and convnextv2_tiny only. A memory-time trade "
                                "with identical gradients — the alternative at 8 GB was "
                                "dropping both arms or shrinking the frozen batch. Their "
                                "TRAIN-TIME column is therefore not a clean architecture "
                                "cost and is flagged in the table; their SCORE-TIME column "
                                "is unaffected (inference runs unchecked)."},
        "control": f"{CONTROL.name} ({CONTROL.timm_model}) RETRAINED FRESH under these "
                   f"conditions. Every delta is measured against THAT run, never against "
                   f"shipped v11, so backbone effect separates from retrain variance."},
      "arms": [{"name": a.name, "timm_model": a.timm_model, "pretrain": a.pretrain,
                "create_kwargs": {k: list(v) if isinstance(v, tuple) else v
                                  for k, v in a.create_kwargs.items()},
                "grad_checkpointing": a.grad_checkpointing,
                "is_control": a.is_control, "why": a.why} for a in ARMS],
      "arms_dropped": [
        {"candidate": "repvit_m1_5.dist_450e_in1k",
         "reason": "timm's RepVit takes no drop_path_rate, so running it would silently "
                   "drop stochastic depth from the frozen recipe — a second moved "
                   "variable. Replaced by fastvit_sa12.apple_dist_in1k, which accepts it.",
         "when": "before any training"},
        {"candidate": "dinov2 vit_small (patch 14)",
         "reason": "384 is not divisible by 14, so it could only be fed by changing the "
                   "deploy transform. Adapting pos-embeds is allowed; changing the deploy "
                   "transform is not. Replaced by vit_small_patch16_224.augreg_in21k_ft_in1k.",
         "when": "before any training"}],
      "eval_populations": {
        "PRIMARY (unseen)": {
          "what": "every eval location EXCEPT the 670 the checkpoint is selected on — the "
                  "1,810-row grouped holdout + uniform-90 + q4-uniform-290.",
          "why": "Each arm's checkpoint is picked on the 670, and the size of that "
                 "optimism differs by arm; the 2,190 touch neither training nor the pick "
                 "for ANY arm, so they are the population on which arms are comparable to "
                 "each other. NOTE the holdout is biased exactly as training is "
                 "(protocol §1) — it is a model-comparison population, and NO base rate is "
                 "read from it here.",
          **label_block(primary),
          "by_role": dict(collections.Counter(r.get("eval_role") for r in primary))},
        "SELECTION (contaminated, reported separately)": {
          "what": "census-144 + loose0_v3_floor-526, the frozen v8..v11 selection objective.",
          "why": "Reported for continuity with every previous version's battery, and "
                 "labelled in-sample: an arm that overfits the pick is flattered here.",
          **label_block(selection)}},
      "declared_metrics": {
        "primary": "location-level AUC of P(label>=3) — rank by sigma(logit_1) — on the "
                   "PRIMARY population, pooled.",
        "secondary": [
          "AUC of P(label>=4) (sigma(logit_2)), pooled and per slice",
          "AUC of P(label>=2) (sigma(logit_0)) — the selection objective's own cutpoint",
          "exact-tier agreement and adjacent (within-one-tier) agreement, decoded "
          "PARAMETER-FREE as tier = 1 + sum_k 1[sigma(logit_k) > 0.5] (CORN's own "
          "rank-consistent decode; no per-arm threshold is fitted, which would be a "
          "second moved variable)",
          "calibration / prior reproduction: (a) total-variation distance between the "
          "decoded tier histogram and the label histogram, (b) per-cutpoint gap between "
          "mean sigma(logit_k) and the observed rate"],
        "cost": ["params (M)", "peak train VRAM (MB) at the frozen batch 32",
                 "train wall clock (h)",
                 "END-TO-END score seconds per 1,000 canonical tiles — JPEG decode + "
                 "deploy transform + forward, batch 64, the cost a ledger rescore pays",
                 "GPU-only score seconds per 1,000 tiles"]},
      "declared_slices": {
        "pooled": "the whole PRIMARY population",
        "per_partition": {k: label_block(v) for k, v in sorted(by_part.items())},
        "named_up_front": ["julia:mandelbrot", "phoenix"],
        "why_named": "the known weak partitions — v11 is the first build in which either "
                     "has an eval population at all (protocol §1), so they are where a "
                     "backbone change is most likely to show and most likely to be "
                     "over-read.",
        "min_positives_for_a_slice_verdict": MIN_POS_SLICE,
        "underpowered_slices_are": "REPORTED WITH THEIR COUNTS AND NO VERDICT — never "
                                   "dropped, which would print 'no data' about data there is."},
      "honesty_rule": {
        "statement": "Label noise forbids calling small differences at the >=3 boundary. "
                     "Every quality delta is reported as a PAIRED bootstrap CI against the "
                     "control, and an arm whose CI covers 0 is a TIE — not a rank.",
        "bootstrap": {"kind": "paired cluster bootstrap", "unit": "split_group (the "
                      "leakage-closure group the v11 holdout was drawn over — locations "
                      "inside one are not independent)", "B": 5000, "seed": 0,
                      "statistic": "delta = metric(arm) - metric(control), recomputed on "
                      "each resample with BOTH arms scored on the same resampled rows",
                      "interval": "percentile 2.5 / 97.5"},
        "secondary_test": "paired DeLong p for AUC deltas (tools/v7/eval_delong.delong_"
                          "paired), reported alongside — the bootstrap is the declared rule.",
        "multiplicity": "The pooled PRIMARY AUC(>=3) is the ONE confirmatory comparison. "
                        "Per-partition deltas are DESCRIPTIVE: CIs are reported unadjusted "
                        "and no winner is declared off them. A per-partition CI that "
                        "excludes 0 is reported as 'suggestive, unadjusted, 9 slices'.",
        "no_staged_pick": "Round-2 arms are summarized by their 3-seed BAND (min..max and "
                          "mean), never by their best seed."},
      "round_2_rule": {
        "AMENDED": "AMENDMENT 1 (Matt, 2026-08-14, BEFORE any arm was scored — round 1 was "
                   "mid-flight and no eval number existed). The question this study has to "
                   "answer was narrowed to SCREENING: is each backbone viable as an "
                   "alternative worth pursuing, not which one wins. Per-arm seed bands are "
                   "what a WINNER needs; a screen needs the NOISE FLOOR, which is one arm's "
                   "band and not seven. So the per-arm seed replication is cut and the "
                   "CONTROL alone is replicated. Cost: 2 runs (~1.7 h) instead of 6 "
                   "(~5-12 h). What is GIVEN UP is stated rather than hidden: a single-seed "
                   "arm delta cannot be decomposed into backbone effect and that ARM's own "
                   "seed variance, so no arm may be RANKED against another here, and an arm "
                   "that clears the bars below is a candidate for a multi-seed study rather "
                   "than a winner of one.",
        "superseded_rule": "top 2-3 non-control arms by pooled PRIMARY AUC(>=3) x 3 seeds, "
                           "control always included, win = 3-seed min above the control's "
                           "3-seed max AND every matched-pair CI excluding 0.",
        "control_seeds": [0, 1, 2],
        "arm_seeds": [0],
        "what_a_seed_moves": "the fresh-head init, the WeightedRandomSampler draw and the "
                             "per-epoch tile/aug draw. The corpus, split and selection "
                             "population do not move.",
        "noise_floor": "The control's 3-seed spread on the pooled PRIMARY AUC(>=3) IS the "
                       "retrain-variance floor, and it is assumed to apply to every arm — "
                       "an assumption this design cannot check, and the one place a "
                       "cheap-but-wrong screen would break (a backbone with a genuinely "
                       "wider seed spread would be read as a mover). Stated as a limitation "
                       "in the report, not buried.",
        "screen_verdict": "Each arm gets one of three, and NEVER a rank against another arm:"
                          " VIABLE — the delta clears BOTH the control's seed spread and a "
                          "paired cluster-bootstrap CI excluding 0; TIE — inside either; "
                          "WORSE — the delta is negative under both. A VIABLE verdict is a "
                          "recommendation to run the multi-seed study that was cut, not an "
                          "adoption case."},
      "cost_projection": {
        "source": "scratch/backbone_search/cost_smoke.json (measured on synthetic tensors "
                  "before round 1)",
        "cpu_floor_note": "v11's real epoch was 76.0 s for 8,443 locations = 111 img/s, "
                          "and the train transform decodes+crops+resizes+RE-ENCODES a JPG "
                          "on 4 workers. The control is CPU-bound, so 76 s/epoch is a "
                          "floor every arm inherits and projections are max(gpu, floor).",
        "per_arm_h": {r["arm"]: r["proj_train_h"] for r in smoke.get("arms", [])},
        "round1_total_h": smoke.get("round1_total_h")},
      "what_would_falsify_the_whole_study": [
        "The control arm failing to reproduce shipped v11's selection AP to within "
        "retrain noise — that would mean the harness, not the backbone, moved.",
        "Any arm's eval reading a different tile set: all arms score the SAME 2,190 "
        "canonical renders from data/v11/eval_canon_manifest.jsonl, and eval_arms.py "
        "asserts the location id list is identical across arms before comparing."],
    }
    out = paths.durable(OUT_REL, mkparents=True)
    out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {out}")
    print(f"  PRIMARY n={len(primary)} (ge3 {sum(r['label']>=3 for r in primary)}), "
          f"SELECTION n={len(selection)}, partitions {len(by_part)}, arms {len(ARMS)}")


if __name__ == "__main__":
    main()
