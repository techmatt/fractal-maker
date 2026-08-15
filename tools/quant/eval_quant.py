#!/usr/bin/env python
"""eval_quant.py — the acceptance run: quantized-vs-full agreement on committed eval material.

    uv run python tools/quant/eval_quant.py run --rungs fp16 int8
    uv run python tools/quant/eval_quant.py run --heads pref --rungs int8 --limit 4

EVERY BAR IS LOADED FROM `data/quant/prereg_quant_v1.json`, which was committed before this
file existed. Nothing here restates a threshold — the same discipline `eval_arms.py` runs
under, and for the same reason: a bar that lives in the harness is a bar the harness can move.

WHAT ONE ARM IS. For one (head, rung): build the head's OWN production scorer off its OWN
live pin, score the material, score it a SECOND time unchanged (the instrument's noise
floor), then load the quantized weights into that same model and score a third time. Decode,
transform, row order, device and process are shared, so the only thing that differs between
the compared passes is the weight values.

THE READS, per `docs/design/measurement_practice.md`:

  * Agreement is a COUNT over its denominator, never a bare rate, because a rate without a
    denominator is not a decision.
  * Every delta is reported beside the fp32-vs-fp32 noise floor measured in the same run. A
    delta below the floor is the instrument, not the rung.
  * The weight error is reported BEFORE any score, so a vacuous rung (max relative weight
    error 0) cannot be read as a passing one — an exact 0.0000 is a measurement of nothing.
  * The blind sheets are READ-ONLY instruments here: this file computes agreement between two
    versions of one head and draws no verdict about any head's quality from them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "quant"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import heads as H                       # noqa: E402
import quantize_head as Q               # noqa: E402

PREREG_REL = "data/quant/prereg_quant_v1.json"
OUT_REL = "data/quant/agreement_quant_v1.json"
RECIPE_REL = "data/quant/quant_recipe_v1.json"

# The AUTHORED half of the decision. Everything else in the recipe record is derived from the
# agreement record by `cmd_decide` — which rung passed where, which bar bound, what the sizes
# were — because a hand-copied number is how a record outlives what it records. These are the
# sentences a table cannot write, and they are here rather than in the JSON so the JSON stays
# regenerable.
NOTES = {
    "why_not_speed": "The objective was stored bytes, never latency. backbone_search_v1 "
                     "measured the backbone at 5-17% of end-to-end score time, so the deploy "
                     "path is decode-bound and no rung here would be visible in throughput.",
    "why_weight_only": "No activation quantization and no backend engine (fbgemm / qnnpack / "
                       "TensorRT). The recipe must re-apply in a clean-room repo with standard "
                       "torch on any platform, and a weight-only rung dequantized at load is "
                       "the only form of that which leaves the compute graph untouched.",
    "why_per_channel": "These are MobileNetV4 backbones. Their depthwise convolutions have "
                       "per-channel weight ranges spanning orders of magnitude, so one scale "
                       "per tensor quantizes the small channels into noise.",
    "int8_verdict": "int8 weight-only is NOT near-free on any of the four heads, and on the "
                    "location head it is not close: it moves AUC(>=4) by more than the entire "
                    "control 3-seed retrain band the bars were calibrated against.",
    "hybrid_verdict": "The per-group sweep found ONE dominant group per backbone family — "
                      "conv_head on mnv4_conv_medium, blocks.2 on mnv4_conv_small — and "
                      "exempting the top three recovers most of the loss. It is still not "
                      "enough on three of four heads, and it cannot be pushed further: the "
                      "group carrying the residual error on the location head is blocks.3, "
                      "which is 63% of that backbone's parameters, so exempting it would "
                      "collapse hybrid back into fp16 at a worse size.",
    "pref_exception": "The pref ranker is the one head where NO rung passes. fp16 keeps its "
                      "top-1 palette on every location and its top-3 set on all but two, but "
                      "misses the declared mean within-location rank agreement. A single-tower "
                      "margin head scores a location's candidates within a fraction of a point "
                      "of each other, so full-order rank agreement is the strictest read any "
                      "of the four heads is held to, and it is the read that decides this head.",
    "what_this_does_not_do": "No production pin moves. Every ACTIVE checkpoint is untouched "
                             "and the big-repo deploy path stays fp32. This record says what a "
                             "FUTURE ship step should do with a newly-trained head.",
}


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# metric primitives
# --------------------------------------------------------------------------- #
def auc(y_bin, s):
    """Binary AUC by mid-rank, or None when one class is empty (a normal state of a small
    slice — report the hole, do not abort the battery)."""
    from scipy.stats import rankdata

    y = np.asarray(y_bin)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(s, float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a, b):
    """Spearman rho, or None when either side is constant (rho is undefined, not 1.0)."""
    from scipy.stats import rankdata

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def decode_tier(P):
    """CORN's own rank-consistent decode: tier = 1 + #{k: P_k >= 0.5}.

    Parameter-free and identical on both arms, so the column reads the WEIGHTS and not a
    per-arm calibration — the argument `eval_arms.decode_tier` makes for the same choice.
    Counting rather than chaining is the canonical decode (`score_lib.corn_decode`): CORN's
    cumulative probabilities are not guaranteed monotone, and counting degrades such a frame
    by one rank rather than promoting it on a cutpoint whose predecessor it failed."""
    return 1 + (np.asarray(P) >= 0.5).sum(axis=1)


def agree(a, b) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    k = int((a == b).sum())
    return {"n_agree": k, "n": n, "rate": (k / n) if n else None}


# --------------------------------------------------------------------------- #
# per-head metric blocks
# --------------------------------------------------------------------------- #
def corn_metrics(spec, items, P0, P1, rank0, rank1) -> dict:
    """Agreement between two weight versions of one CORN head, on one population."""
    labels = np.array([it.label for it in items])
    K = P0.shape[1] + 1
    dp = np.abs(P1 - P0)
    out = {
        "n": len(items),
        "n_by_label": {str(t): int((labels == t).sum()) for t in range(1, K + 1)},
        "mean_abs_delta_p": float(dp.mean()),
        "max_abs_delta_p": float(dp.max()),
        "max_abs_delta_rank": float(np.abs(rank1 - rank0).max()),
        "decoded_tier_agreement": agree(decode_tier(P0), decode_tier(P1))["rate"],
        "decoded_tier_agreement_count": agree(decode_tier(P0), decode_tier(P1)),
    }
    for k in range(K - 1):
        t = k + 2
        y = (labels >= t).astype(int)
        a0, a1 = auc(y, P0[:, k]), auc(y, P1[:, k])
        out[f"n_ge{t}"] = int(y.sum())
        out[f"auc_ge{t}_fp32"] = a0
        out[f"auc_ge{t}_quant"] = a1
        out[f"delta_auc_ge{t}"] = (None if a0 is None or a1 is None else a1 - a0)
        out[f"abs_delta_auc_ge{t}"] = (None if a0 is None or a1 is None else abs(a1 - a0))
        out[f"spearman_p_ge{t}"] = spearman(P0[:, k], P1[:, k])
    if spec.gate_threshold is not None:
        g0 = P0[:, 1] > spec.gate_threshold
        g1 = P1[:, 1] > spec.gate_threshold
        out["gate"] = {"threshold": spec.gate_threshold, "pin": spec.gate_pin,
                       "n_pass_fp32": int(g0.sum()), "n_pass_quant": int(g1.sum())}
        out["gate_verdict_agreement_count"] = agree(g0, g1)
        out["gate_verdict_agreement"] = agree(g0, g1)["rate"]
    return out


def pref_metrics(spec, items, P0, P1, rank0, rank1) -> dict:
    """WITHIN-LOCATION agreement for the ranking head.

    The pref head is a single tower trained on a margin loss: its scores are comparable only
    inside one location's candidate set (identical geometry), never across locations. So
    there is no pooled correlation here, by construction — every quantity is computed inside
    a candidate set and then summarized over sets."""
    by = defaultdict(list)
    for i, it in enumerate(items):
        by[it.group].append(i)
    rhos, flips, top3, per_loc = [], 0, 0, []
    worst = None
    for loc in sorted(by):
        idx = np.array(by[loc])
        s0, s1 = rank0[idx], rank1[idx]
        rho = spearman(s0, s1)
        a0, a1 = int(np.argmax(s0)), int(np.argmax(s1))
        same_top1 = items[idx[a0]].key == items[idx[a1]].key
        o0 = {items[idx[j]].key for j in np.argsort(-s0)[:3]}
        o1 = {items[idx[j]].key for j in np.argsort(-s1)[:3]}
        same_top3 = (o0 == o1)
        blk = {"location_id": loc, "n_candidates": len(idx), "spearman": rho,
               "top1_same": bool(same_top1), "top3_set_same": bool(same_top3),
               "top1_fp32": items[idx[a0]].extra.get("variant_id"),
               "top1_quant": items[idx[a1]].extra.get("variant_id"),
               # the margin a top-1 flip crossed: a swap between two candidates the fp32
               # head itself separated by ~0 is a different event from one that reorders a
               # clear winner, and only the record can tell them apart afterwards
               "fp32_margin_top1_top2": float(np.diff(np.sort(s0)[-2:])[0]),
               "max_abs_delta_score": float(np.abs(s1 - s0).max())}
        per_loc.append(blk)
        if rho is not None:
            rhos.append(rho)
        flips += (not same_top1)
        top3 += same_top3
        if worst is None or (rho is not None and rho < worst["spearman"]):
            worst = blk
    n = len(per_loc)
    return {"n_locations": n, "n_frames": len(items),
            "mean_within_location_spearman": float(np.mean(rhos)) if rhos else None,
            "min_within_location_spearman": float(np.min(rhos)) if rhos else None,
            "n_spearman_defined": len(rhos),
            "argmax_agreement": (n - flips) / n if n else None,
            "argmax_agreement_count": {"n_agree": n - flips, "n": n},
            "top3_set_agreement": top3 / n if n else None,
            "top3_set_agreement_count": {"n_agree": top3, "n": n},
            "max_abs_delta_score": float(np.abs(rank1 - rank0).max()),
            "mean_abs_delta_score": float(np.abs(rank1 - rank0).mean()),
            "worst_location": worst, "per_location": per_loc}


def noise_floor(P0, P0b, rank0, rank0b) -> dict:
    """fp32 scored twice, unchanged. Everything below this is the instrument."""
    return {"max_abs_delta_p": float(np.abs(P0b - P0).max()),
            "mean_abs_delta_p": float(np.abs(P0b - P0).mean()),
            "max_abs_delta_rank": float(np.abs(rank0b - rank0).max()),
            "bit_identical": bool(np.array_equal(P0, P0b))}


# --------------------------------------------------------------------------- #
# bars
# --------------------------------------------------------------------------- #
def check_bars(bars: dict, metrics: dict) -> dict:
    """Every declared bar, against the measured value. `*_max` is an upper bound, `*_min` a
    lower one; the metric name is the bar key minus that suffix, so a bar cannot be declared
    for a metric that is not computed (it reports MISSING rather than passing silently)."""
    out = {}
    for bar_key, bound in bars.items():
        if bar_key.endswith("_max"):
            name, cmp = bar_key[:-4], "<="
        elif bar_key.endswith("_min"):
            name, cmp = bar_key[:-4], ">="
        else:
            out[bar_key] = {"verdict": "MALFORMED", "why": "bar key ends in neither _max nor _min"}
            continue
        val = metrics.get(name)
        if val is None:
            out[bar_key] = {"verdict": "MISSING", "metric": name, "bar": bound,
                            "why": "the harness computed no such metric (or it is undefined "
                                   "on this population)"}
            continue
        ok = (val <= bound) if cmp == "<=" else (val >= bound)
        out[bar_key] = {"verdict": "PASS" if ok else "FAIL", "metric": name,
                        "value": float(val), "bar": bound, "cmp": cmp}
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_head(spec, rungs, prereg, *, limit=None, device=None, keep_fp16=()) -> dict:
    import torch

    t0 = time.time()
    items = spec.items(limit)
    log(f"[{spec.key}] {len(items)} items · ckpt {spec.ckpt_rel}")
    scorer = H.load_scorer(spec, device)
    src = ROOT / spec.ckpt_rel
    src_sd = {k: v.clone() for k, v in
              torch.load(src, map_location="cpu", weights_only=False)["state_dict"].items()}

    P0, rank0 = scorer.score(items)
    P0b, rank0b = scorer.score(items)
    floor = noise_floor(P0, P0b, rank0, rank0b)
    log(f"[{spec.key}] noise floor max|dp| {floor['max_abs_delta_p']:.2e} "
        f"(fp32 scored twice, unchanged)")

    bars = prereg["heads"][spec.key]["bars"]
    metric_fn = pref_metrics if spec.kind == "ranker" else corn_metrics
    blk = {"head": spec.key, "label": spec.label, "pin": spec.pin, "ckpt": spec.ckpt_rel,
           "material": spec.material, "n_items": len(items),
           "source_bytes": src.stat().st_size, "source_sha256": Q.sha256_file(src),
           "device": scorer.device, "noise_floor_fp32_vs_fp32": floor,
           "bars": bars, "rungs": {}}
    if spec.kind == "ranker":
        blk["draw"] = {
            "what": "every palette_candidates[] entry of every committed library record",
            "source": "data/library/library_records.jsonl",
            "n_locations": len({it.group for it in items}), "n_frames": len(items),
            "locations": sorted({it.group for it in items}),
            "recolor_path": "tools/curation/colored_clip.render_candidates (the live path)",
        }

    for rung in rungs:
        art = Q.artifact_path(spec.key, rung)
        # The exception list belongs to the HYBRID rung and to nothing else. It was applied
        # to every rung for one run, which silently turned the int8 arm into a second copy
        # of hybrid — a control arm that is not the control is worse than a missing one,
        # because the table still reads as a ladder.
        meta = Q.write_artifact(src, rung, art,
                                keep_fp16_groups=(keep_fp16 if rung == "hybrid" else ()))
        deq, _cfg, _m = Q.read_artifact(art)
        werr = Q.weight_error(src_sd, deq)
        Q.apply_to_model(scorer.model, art)
        P1, rank1 = scorer.score(items)
        m = metric_fn(spec, items, P0, P1, rank0, rank1)
        verdicts = check_bars(bars, m)
        blk["rungs"][rung] = {
            "artifact": art.as_posix(), "sha256": meta["sha256"],
            "bytes_before": meta["bytes_before"], "bytes_after": meta["bytes_after"],
            "mb_before": round(meta["bytes_before"] / 1e6, 2),
            "mb_after": round(meta["bytes_after"] / 1e6, 2),
            "size_ratio": round(meta["bytes_after"] / meta["bytes_before"], 4),
            "keep_fp16_groups": meta["keep_fp16_groups"], "tensor_kinds": meta["kinds"],
            # BEFORE any score is read: a rung that moved nothing cannot pass a bar
            "weight_error": werr,
            "vacuous": werr["max_rel_err_per_tensor"] == 0.0,
            "metrics": m, "bar_verdicts": verdicts,
            "verdict": ("VACUOUS" if werr["max_rel_err_per_tensor"] == 0.0 else
                        "PASS" if all(v["verdict"] == "PASS" for v in verdicts.values())
                        else "FAIL"),
        }
        log(f"[{spec.key}] {rung:6s} {meta['bytes_after']/1e6:6.2f} MB "
            f"({blk['rungs'][rung]['size_ratio']:.3f}x)  "
            f"max_rel_werr {werr['max_rel_err_per_tensor']:.2e}  "
            f"-> {blk['rungs'][rung]['verdict']}")
        for k, v in verdicts.items():
            if v["verdict"] != "PASS":
                log(f"    {v['verdict']:7s} {k}: {v.get('value')} vs bar {v.get('bar')}")
        # restore fp32 for the next rung: every rung is measured against the SAME reference
        scorer.model.load_state_dict(src_sd)

    blk["wall_s"] = round(time.time() - t0, 1)
    return blk


def cmd_run(a):
    prereg = json.loads((ROOT / PREREG_REL).read_text(encoding="utf-8"))
    heads = a.heads or list(H.HEADS)
    out = {"study": prereg["study"], "prereg": PREREG_REL,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "command": "uv run python tools/quant/eval_quant.py run "
                      f"--heads {' '.join(heads)} --rungs {' '.join(a.rungs)}",
           "recipe_version": Q.RECIPE_VERSION,
           "limit": a.limit, "heads": {}}
    for key in heads:
        out["heads"][key] = run_head(H.HEADS[key], a.rungs, prereg, limit=a.limit,
                                     device=a.device, keep_fp16=a.keep_fp16)
    # smallest rung passing EVERYWHERE — the prereg's adoption rule, computed rather than read
    passing = {r: [k for k in out["heads"]
                   if out["heads"][k]["rungs"].get(r, {}).get("verdict") == "PASS"]
               for r in a.rungs}
    out["rung_pass_by_head"] = passing
    out["smallest_rung_passing_everywhere"] = next(
        (r for r in a.rungs if len(passing[r]) == len(heads)), None)

    import paths
    dst = (paths.durable(OUT_REL, mkparents=True) if not a.out else Path(a.out))
    if a.limit:                      # a bounded run must stamp itself unusable
        out["incomplete"] = True
        out["incomplete_why"] = (f"--limit {a.limit}: this is a bounded end-to-end, not the "
                                 f"acceptance population")
        from paths import scratch
        dst = scratch("quant") / "agreement_quant_v1.INCOMPLETE.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log(f"wrote {dst}")
    log(f"smallest rung passing everywhere: {out['smallest_rung_passing_everywhere']}")
    return out


def choose_default(ladder, passing: dict, n_heads: int):
    """`(default_rung, passes_everywhere)` — the prereg's adoption rule as a function.

    The declared rule is "adopt the SMALLEST rung that passes, everywhere; a head that fails
    it takes a NAMED EXCEPTION rather than a lowered bar". `ladder` is in smallest-first
    order, so the first universal pass wins.

    When nothing passes everywhere the rule still decides, and this is the branch that
    actually fired: take the rung with the WIDEST pass (smallest first on a tie) and ship the
    heads it does not cover unquantized. That is not a tie-break invented after the fact — it
    is the only move left once "no bar is relaxed after a number is seen" removes the other
    one."""
    universal = next((r for r in ladder if len(passing[r]) == n_heads), None)
    if universal:
        return universal, True
    return max(ladder, key=lambda r: (len(passing[r]), -ladder.index(r))), False


def cmd_decide(a):
    """Turn the agreement record into the dated decision. No model is loaded, so the recipe
    can be re-cut from a frozen table in a second — the same score/report split
    `eval_arms.py` uses, and for the same reason."""
    import paths

    prereg = json.loads((ROOT / PREREG_REL).read_text(encoding="utf-8"))
    rec = json.loads((ROOT / OUT_REL).read_text(encoding="utf-8"))
    if rec.get("incomplete"):
        raise SystemExit("the agreement record is stamped incomplete (a --limit run); "
                         "re-run the acceptance population before deciding anything off it")
    ladder = [r["name"] for r in prereg["candidate_ladder"]["rungs"]]

    per_head, passing = {}, {r: [] for r in ladder}
    for key, blk in rec["heads"].items():
        order = [r for r in ladder if r in blk["rungs"]]
        for r in order:
            if blk["rungs"][r]["verdict"] == "PASS":
                passing[r].append(key)
        smallest = next((r for r in order if blk["rungs"][r]["verdict"] == "PASS"), None)
        per_head[key] = {
            "pin": blk["pin"], "ckpt": blk["ckpt"], "material": blk["material"],
            "n": blk["n_items"],
            "smallest_passing_rung": smallest,
            "adopted_rung": smallest or "fp32 (unquantized)",
            "verdict_by_rung": {r: blk["rungs"][r]["verdict"] for r in order},
            "sizes_mb": {r: blk["rungs"][r]["mb_after"] for r in order},
            "size_ratio": {r: blk["rungs"][r]["size_ratio"] for r in order},
            "mb_before": round(blk["source_bytes"] / 1e6, 2),
            "source_sha256": blk["source_sha256"],
            "sha256_by_rung": {r: blk["rungs"][r]["sha256"] for r in order},
            "failed_bars_by_rung": {
                r: {k: {kk: v[kk] for kk in ("value", "bar", "cmp") if kk in v}
                    for k, v in blk["rungs"][r]["bar_verdicts"].items()
                    if v["verdict"] != "PASS"}
                for r in order},
            "noise_floor_fp32_vs_fp32": blk["noise_floor_fp32_vs_fp32"],
        }

    default, universal = choose_default(ladder, passing, len(per_head))
    exceptions = {k: v for k, v in per_head.items() if v["adopted_rung"] != default}

    out = {
        "study": prereg["study"], "decided": time.strftime("%Y-%m-%d"),
        "prereg": PREREG_REL, "agreement_record": OUT_REL,
        "recipe_version": Q.RECIPE_VERSION, "artifact_format": Q.ARTIFACT_FORMAT,
        "tool": "tools/quant/quantize_head.py",
        "command": "uv run python tools/quant/eval_quant.py decide",
        "default_rung": default,
        "default_passes_everywhere": bool(universal),
        "default_covers": passing[default],
        "per_head_exceptions": {k: v["adopted_rung"] for k, v in exceptions.items()},
        "rung_pass_by_head": passing,
        "rung_spec": {r["name"]: r["spec"] for r in prereg["candidate_ladder"]["rungs"]},
        "hybrid_exception_groups": sorted(
            {g for b in rec["heads"].values()
             for g in b["rungs"].get("hybrid", {}).get("keep_fp16_groups", [])}),
        "sensitivity_sweep": "scratch/quant/sweep_<head>.json "
                             "(uv run python tools/quant/quantize_head.py sweep --head <head>)",
        "per_head": per_head,
        "notes": NOTES,
        "quantized_weights_are_not_tracked": (
            f"{Q.WEIGHTS_DIR} is a registered bulk prefix; a quantized head is born "
            f"out-of-tree and is reproducible from its source sha256 plus its rung"),
    }
    dst = paths.durable(RECIPE_REL, mkparents=True)
    dst.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log(f"wrote {dst}")
    log(f"DEFAULT RUNG: {default} "
        f"({'passes on every head' if universal else 'passes on ' + ', '.join(passing[default])})")
    for k, v in exceptions.items():
        log(f"  EXCEPTION {k}: {v['adopted_rung']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="agreement record -> the dated recipe decision")
    d.set_defaults(fn=cmd_decide)
    r = sub.add_parser("run")
    r.add_argument("--heads", nargs="*", default=None, choices=list(H.HEADS))
    r.add_argument("--rungs", nargs="+", default=["fp16", "int8"], choices=Q.RUNGS)
    r.add_argument("--keep-fp16", action="append", default=[], metavar="GROUP",
                   help="hybrid only: layer group kept at fp16 (repeatable)")
    r.add_argument("--limit", type=int, default=None,
                   help="bounded end-to-end; writes an INCOMPLETE-stamped file to scratch/")
    r.add_argument("--device", default=None)
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
