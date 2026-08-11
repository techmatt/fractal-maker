r"""wallpaper_v4b_reads.py — wallpaper v3 vs v4b on ONE harness, winner rule, STAGED ONLY.

`prompts/retrains_28.md`, wallpaper half. Both checkpoints are re-scored over the SAME crops
on disk right now, through `report_v4_eval.load_head` (head-agnostic: K, geometry, interp and
mean/std all come from the checkpoint's own config). Neither head's frozen `eval_scores.jsonl`
is read: v3's was produced against crops that have since been deleted and re-rendered, and the
in-row `head_v3.p_ge3` stamps were produced at sheet-build time. Either would be a comparison
across two rendering events.

THE ARMS, declared in `classifier.train_wallpaper_v4b.PRE_DECLARED` — the same object the
TRAINER logged before it trained, imported here rather than restated so the two cannot drift:

  MOTIVATING  `sheet_a_minibrot_maneuver`  sheet A's minibrot-centered / maneuver-view
              stratum, the 300-row bucket the sitting was built around.
  NO-WORSE    `fresh_colorize_path` (the v4 regression slice — the regime a live emission
              actually colours through), `fresh_pool_draw`, `overall`.
  DIAGNOSTIC  the July halves, sheet A as a whole, the maneuver VEIN (a different cut of the
              same idea: bucket is how the row was drawn, vein is where the location came
              from), and the tier-4 boundary everywhere.

EVERY CROSS-HEAD COMPARISON IS VOLUME-MATCHED. A CORN marginal is calibrated to its own
training prior, so `p_ge3 > 0.90` is a point on v3's scale and means something else on
v4b's: comparing precision at a shared raw threshold compares two different operating
points. `wallpaper_pins.GATE_THRESHOLD` is therefore reported as the VOLUME v3 passes at
0.90, and v4b is read at that same volume. AUC and AP are rank statistics and are immune,
which is why they carry the winner rule.

WHAT LEANS THIS COMPARISON. Only one thing, and it leans toward v4b's DISADVANTAGE being
understated rather than the reverse: v4b's staged checkpoint is the best of five seeds by
pooled-eval AP>=3 on this very slice, so the staged number is optimistic — the five-seed band
is reported beside it. The eval side itself is clean for BOTH heads: the five prior batches
keep v4's split verbatim (v3 trained on none of their eval rows) and sheet A's rows were
never seen by v3 at all.

DERIVED AND RECORDED. `wallpaper_pins.HEAD_CKPT` is read, never written; no gate, floor or
pin moves here.

Outputs -> data/wallpaper_head/v4b/report.{md,json}

  uv run python tools/wallpaper/wallpaper_v4b_reads.py
  uv run python tools/wallpaper/wallpaper_v4b_reads.py --limit 64    # bounded end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classifier.train_wallpaper_v4b import (PRE_DECLARED, load_union,   # noqa: E402
                                            slices_of, split_v4b)
from tools.corpus.q4_combined_readout import wilson                     # noqa: E402
from tools.scoring.winner_rule import (Metric, paired_bootstrap,        # noqa: E402
                                       point_block, verdict)
from tools.wallpaper import wallpaper_pins as WP                        # noqa: E402

V3_CKPT = ROOT / WP.HEAD_CKPT_REL
V4B_DIR = ROOT / "data" / "wallpaper_head" / "v4b"
V4B_CKPT = V4B_DIR / "model_best.pt"

# K=4 — three boundaries, two statistics each. Fixed above the code that computes them.
METRICS = (Metric("auc_ge3", "AUC>=3", "p_ge3", 3, "auc"),
           Metric("ap_ge3", "AP>=3", "p_ge3", 3, "ap"),
           Metric("auc_ge2", "AUC>=2", "p_ge2", 2, "auc"),
           Metric("ap_ge2", "AP>=2", "p_ge2", 2, "ap"),
           Metric("auc_ge4", "AUC>=4", "p_ge4", 4, "auc"),
           Metric("ap_ge4", "AP>=4", "p_ge4", 4, "ap"))

VOLUME_RATES = (0.05, 0.10, 0.20)


def log(m):
    print(m, flush=True)


def score_with(ckpt: Path, rows, device: str) -> dict:
    from tools.wallpaper.report_v4_eval import load_head    # noqa: PLC0415 (torch import)
    score, cfg = load_head(ckpt, device)
    _cond, marg, ssum = score([r.jpg for r in rows])
    return {"p_ge2": marg[:, 0], "p_ge3": marg[:, 1], "p_ge4": marg[:, 2],
            "rank": ssum, "_cfg": cfg}


def tier_dist(labels) -> dict:
    c = Counter(int(x) for x in labels)
    n = len(labels)
    return {"n": n, "hist": {str(t): c.get(t, 0) for t in (1, 2, 3, 4)},
            "frac_ge2": (n - c.get(1, 0)) / n if n else None,
            "frac_ge3": (c.get(3, 0) + c.get(4, 0)) / n if n else None,
            "frac_ge4": c.get(4, 0) / n if n else None}


def arm_block(labels, base, cand, mask, *, draws, seed) -> dict:
    lb = labels[mask]
    b = {k: v[mask] for k, v in base.items() if not k.startswith("_")}
    c = {k: v[mask] for k, v in cand.items() if not k.startswith("_")}
    return {"n": int(mask.sum()), "tiers": tier_dist(lb),
            "v3": point_block(lb, b, METRICS), "v4b": point_block(lb, c, METRICS),
            "delta_ci": paired_bootstrap(lb, b, c, METRICS, draws=draws, seed=seed)}


def top_block(labels, s, k) -> dict | None:
    n = len(s)
    k = int(min(max(k, 0), n))
    if not k:
        return None
    order = np.argsort(-s, kind="stable")[:k]
    good = int((labels >= 3).sum())
    exc = int((labels >= 4).sum())
    tp = int((labels[order] >= 3).sum())
    p, lo, hi = wilson(tp, k)
    return {"n_selected": k, "pass_rate": k / n, "tp": tp,
            "precision_ge3": p, "precision_lo": lo, "precision_hi": hi,
            "recall_ge3": tp / good if good else None,
            "frac_tier4": float((labels[order] == 4).mean()),
            "recall_ge4": (int((labels[order] >= 4).sum()) / exc) if exc else None,
            "cut_at": float(s[order[-1]])}


def volume_matched(labels, base, cand) -> dict:
    """v3 and v4b at EQUAL selected volume — the only honest cross-head precision read.

    Volumes are (i) whatever the deployed gate passes on v3, so "the same number of renders
    as today" is directly comparable, and (ii) three fixed pass rates, so the comparison does
    not depend on v3's calibration at all."""
    n = len(labels)
    v = int((base["p_ge3"] > WP.GATE_THRESHOLD).sum())
    out = {"by_deployed_gate": {
        "threshold_on_v3": WP.GATE_THRESHOLD, "matched_volume": v,
        "note": "v4b is read at v3's VOLUME, never at v3's threshold — the two marginals "
                "are calibrated to different training priors.",
        "v3": top_block(labels, base["p_ge3"], v),
        "v4b": top_block(labels, cand["p_ge3"], v)}, "by_fixed_rate": {}}
    for rate in VOLUME_RATES:
        k = int(round(rate * n))
        out["by_fixed_rate"][f"{rate:.2f}"] = {
            "matched_volume": k,
            "v3": top_block(labels, base["p_ge3"], k),
            "v4b": top_block(labels, cand["p_ge3"], k)}
    return out


def build(rows, base, cand, seed_scores, meta, *, draws, seed) -> dict:
    labels = np.array([r.label for r in rows])
    sl = slices_of(rows)
    motiv = {k: sl[k] for k in PRE_DECLARED["motivating"]}
    noworse = {k: sl[k] for k in PRE_DECLARED["no_worse"]}
    diag = {k: v for k, v in sl.items()
            if k not in motiv and k not in noworse}

    R = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": "uv run python tools/wallpaper/wallpaper_v4b_reads.py",
        "heads": {"v3": {"ckpt": WP.HEAD_CKPT_REL, "role": "incumbent (LIVE pin)"},
                  "v4b": {"ckpt": "data/wallpaper_head/v4b/model_best.pt",
                          "role": "from-scratch candidate, STAGED"}},
        "live_pin": WP.HEAD_CKPT_REL, "live_gate": WP.GATE_THRESHOLD,
        "moves_nothing": "no pin, gate or floor is written by this file",
        "eval_slice": {"n": len(rows), "n_locations": len({r.loc for r in rows}),
                       "by_batch": dict(Counter(r.batch for r in rows)),
                       "by_coloring_source": dict(Counter(r.coloring_source for r in rows)),
                       "tiers": tier_dist(labels)},
        "split": meta,
        "bootstrap": {"draws": draws, "seed": seed, "kind": "paired over eval rows"},
        "pre_declared": PRE_DECLARED,
    }

    def arms(masks):
        return {k: arm_block(labels, base, cand, m, draws=draws, seed=seed)
                for k, m in masks.items() if m.sum() > 0}

    R["motivating"] = arms(motiv)
    R["no_worse"] = arms(noworse)
    R["diagnostic"] = arms(diag)

    if seed_scores:
        band = []
        for s, sc in sorted(seed_scores.items()):
            pb = point_block(labels, {k: v for k, v in sc.items() if not k.startswith("_")},
                             METRICS)
            band.append({"seed": s, **{m.key: pb[m.key] for m in METRICS}})
        R["v4b_seed_band"] = {
            "per_seed": band,
            "mean_sd": {m.key: {"mean": float(np.mean([b[m.key] for b in band])),
                                "sd": float(np.std([b[m.key] for b in band], ddof=0))}
                        for m in METRICS},
            "note": "every seed selected on THIS eval side; the staged max is optimistic "
                    "against this band."}

    R["score_scale"] = {
        "why": "CORN marginals are calibrated to the training prior, so the two heads' "
               "p_ge3 are not on one scale and no raw-threshold comparison appears in this "
               "report.",
        "quantiles": {f"q{int(q*100)}": {"v3": float(np.quantile(base["p_ge3"], q)),
                                         "v4b": float(np.quantile(cand["p_ge3"], q))}
                      for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "pass_rate_at_the_live_gate": {
            "threshold": WP.GATE_THRESHOLD,
            "v3": float((base["p_ge3"] > WP.GATE_THRESHOLD).mean()),
            "v4b": float((cand["p_ge3"] > WP.GATE_THRESHOLD).mean()),
            "read_as": "volume, NOT quality — the two rates are what makes the volume "
                       "matching necessary, not a comparison in themselves."}}
    R["volume_matched"] = volume_matched(labels, base, cand)

    R["winner_rule"] = verdict(
        {k: v["delta_ci"] for k, v in R["no_worse"].items()},
        {k: v["delta_ci"] for k, v in R["motivating"].items()},
        pooled_arm="overall", baseline="v3", candidate="v4b")
    R["winner_rule"]["candidate_ckpt"] = (
        "data/wallpaper_head/v4b/model_best.pt" if R["winner_rule"]["winner"] == "v4b"
        else WP.HEAD_CKPT_REL)
    R["winner_rule"]["adoption"] = ("NOT decided here. BUILD != FLIP: adoption is a separate "
                                    "prompt after Matt reads this verdict.")
    return R


def md(R) -> str:
    L = []
    A = L.append
    wr = R["winner_rule"]
    A("# wallpaper v3 vs v4b — winner-rule verdict (STAGED, nothing adopted)\n")
    A(f"Generated {R['generated']} · `{R['command']}`\n")
    A(f"**WINNER: {wr['winner']}** (pooled-only reading: {wr['winner_pooled_only']})  ")
    A(f"clause (a) no-worse {'PASS' if wr['clause_a']['pass'] else 'FAIL'} over "
      f"{wr['clause_a']['n_tests']} arm x metric cells · clause (b) motivating "
      f"{'PASS' if wr['clause_b']['pass'] else 'FAIL'}\n")
    A(f"> {wr['adoption']}\n")
    e = R["eval_slice"]
    A(f"Eval slice: **{e['n']} rows** over {e['n_locations']} locations, {e['by_batch']}; "
      f"tiers {e['tiers']['hist']}.\n")

    def table(title, arms, order):
        A(f"\n## {title}\n")
        A("| arm | n | ≥3 | v3 AUC≥3 | v4b AUC≥3 | Δ 95% CI | v3 AP≥3 | v4b AP≥3 | Δ 95% CI |")
        A("|---|---:|---:|---:|---:|---|---:|---:|---|")
        for name in order:
            b = arms.get(name)
            if b is None:
                continue
            def cell(x):
                return "—" if x is None else f"{x:.3f}"
            def ic(c):
                if not c or c["n_draws"] == 0:
                    return "n/a"
                tag = "**worse**" if c["significantly_worse"] else (
                    "**better**" if c["significantly_better"] else "")
                return f"[{c['lo']:+.3f}, {c['hi']:+.3f}] {tag}"
            n3 = int(b["tiers"]["hist"]["3"]) + int(b["tiers"]["hist"]["4"])
            A(f"| `{name}` | {b['n']} | {n3} | {cell(b['v3']['auc_ge3'])} "
              f"| {cell(b['v4b']['auc_ge3'])} | {ic(b['delta_ci']['auc_ge3'])} "
              f"| {cell(b['v3']['ap_ge3'])} | {cell(b['v4b']['ap_ge3'])} "
              f"| {ic(b['delta_ci']['ap_ge3'])} |")

    table("MOTIVATING arm", R["motivating"], sorted(R["motivating"]))
    table("NO-WORSE arms", R["no_worse"], PRE_DECLARED["no_worse"])
    table("Diagnostics (vote on nothing)", R["diagnostic"], sorted(R["diagnostic"]))

    if wr["clause_a"]["failures"]:
        A("\n### clause (a) failures\n")
        for f in wr["clause_a"]["failures"]:
            A(f"- `{f['arm']}` **{f['metric']}**: Δ median {f['median']:+.3f}, "
              f"95% CI [{f['lo']:+.3f}, {f['hi']:+.3f}]")
    if wr["clause_b"]["improvements"]:
        A("\n### clause (b) improvements\n")
        for f in wr["clause_b"]["improvements"]:
            A(f"- `{f['arm']}` **{f['metric']}**: Δ median {f['median']:+.3f}, "
              f"95% CI [{f['lo']:+.3f}, {f['hi']:+.3f}]")
    A(f"\n{wr['multiplicity_note']}\n")

    vm = R["volume_matched"]["by_deployed_gate"]
    A("\n## Volume-matched (never a shared raw threshold)\n")
    A(f"The deployed gate passes **{vm['matched_volume']}** of {R['eval_slice']['n']} eval "
      f"rows on v3 (`p_ge3 > {vm['threshold_on_v3']}`); v4b is read at that same volume.\n")
    A("| volume | v3 precision≥3 | v4b precision≥3 | v3 %tier4 | v4b %tier4 | n |")
    A("|---|---:|---:|---:|---:|---:|")
    if vm["v3"] and vm["v4b"]:
        A(f"| deployed gate | {vm['v3']['precision_ge3']:.3f} | {vm['v4b']['precision_ge3']:.3f} "
          f"| {vm['v3']['frac_tier4']:.3f} | {vm['v4b']['frac_tier4']:.3f} "
          f"| {vm['matched_volume']} |")
    for name, blk in R["volume_matched"]["by_fixed_rate"].items():
        if blk["v3"] and blk["v4b"]:
            A(f"| top {name} | {blk['v3']['precision_ge3']:.3f} "
              f"| {blk['v4b']['precision_ge3']:.3f} | {blk['v3']['frac_tier4']:.3f} "
              f"| {blk['v4b']['frac_tier4']:.3f} | {blk['matched_volume']} |")

    if "v4b_seed_band" in R:
        A("\n## v4b five-seed band (staged = the max of these, on this same slice)\n")
        A("| metric | mean ± sd | per seed |")
        A("|---|---|---|")
        for m in METRICS:
            b = R["v4b_seed_band"]["mean_sd"][m.key]
            vals = " ".join(f"{s[m.key]:.3f}" if s[m.key] is not None else "—"
                            for s in R["v4b_seed_band"]["per_seed"])
            A(f"| {m.label} | {b['mean']:.3f} ± {b['sd']:.3f} | {vals} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="eval rows — bounded end-to-end; writes to scratch/, stamped "
                         "incomplete")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    for p in (V3_CKPT, V4B_CKPT):
        if not p.exists():
            raise SystemExit(f"[v4b-reads] missing checkpoint {p} — train it first "
                             f"(uv run python -m classifier.train_wallpaper_v4b)")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    prior, sheet_a = load_union()
    _tr, ev, meta = split_v4b(prior, sheet_a)
    rows = ev[:a.limit] if a.limit else ev
    log(f"[v4b-reads] eval slice {len(rows)} rows on {device}")

    base = score_with(V3_CKPT, rows, device)
    cand = score_with(V4B_CKPT, rows, device)
    seed_scores = {}
    if not a.limit:
        for d in sorted(V4B_DIR.glob("seed_*")):
            ck = d / "model_best.pt"
            if ck.exists():
                seed_scores[int(d.name.split("_")[1])] = score_with(ck, rows, device)
                log(f"[v4b-reads] scored {d.name}")

    R = build(rows, base, cand, seed_scores, meta, draws=a.draws, seed=a.seed)
    R["incomplete"] = bool(a.limit)
    out = (ROOT / "scratch" / "wallpaper_v4b_reads") if a.limit else V4B_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(md(R), encoding="utf-8")
    wr = R["winner_rule"]
    log(f"[v4b-reads] WINNER {wr['winner']} (pooled-only {wr['winner_pooled_only']}) — "
        f"clause a {wr['clause_a']['pass']} / clause b {wr['clause_b']['pass']}")
    log(f"-> {out / 'report.md'}")


if __name__ == "__main__":
    main()
