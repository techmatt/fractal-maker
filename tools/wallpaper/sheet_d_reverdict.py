r"""sheet_d_reverdict.py — the ONE-COMMAND re-verdict for wallpaper v3 vs v4b on SHEET D.

    uv run python tools/wallpaper/sheet_d_reverdict.py

That is the whole invocation. It runs AFTER Matt labels sheet D and merges the export
(`merge_sitting.py --batch 2026-08-11_wallpaper_blind_minibrot_v1 ... --apply`); before that
it exits with the merge command rather than half a report.

WHAT IT ANSWERS, and it is exactly the two questions the (28) verdict could not:

  (a) **THE MOTIVATING ARM, ON UNANCHORED LABELS.** v4b lost clause (b) because its
      motivating arm was sheet A's minibrot bucket, whose labels came back 84.9% equal to the
      v3 suggestion they were served with. Sheet D is that same population — the same
      `MINIBROT_VEINS` cut of the same intake — at FRESH locations, served BLIND. So the
      slice IS the arm, and the verdict here is the one the retrain wanted.

  (b) **THE ANCHORING PRICE.** v3's AUC>=3 on the anchored sheet-A bucket is read out of the
      committed v4b report (0.965 there, against 0.746/0.750 on the two batches labeled
      before v3 existed) and set beside v3's AUC>=3 here. The gap is what one head's
      suggestion is worth as a predictor of the label it suggested, measured rather than
      argued. DERIVED from `data/wallpaper_head/v4b/report.json`, never restated as a
      constant — a number pasted here would outlive the run it describes.

BOTH HEADS ARE RE-SCORED HERE, over the same crops, through one loader
(`report_v4_eval.load_head`, head-agnostic: K, geometry, interp and mean/std all come from the
checkpoint's own config). The five staged v4b seeds are scored too, so the staged pick can be
read against its own band — every one of those seeds was selected on the (28) pooled eval, NOT
on this slice, which makes sheet D the first held-out population any of them has seen.

WHAT LEANS THIS COMPARISON: nothing that favours either head. Sheet D's locations are fresh to
both, its labels were elicited with neither head's suggestion on the page, its palette came
from the pref head, and its rows are stamped eval-only. The one caveat worth stating is the
opposite of a lean: it is ONE slice of ~200 rows, so a null here is "not distinguishable at
this n", not "identical".

MOVES NOTHING. `wallpaper_pins.HEAD_CKPT` is read and never written; no pin, gate or floor
changes. Adoption stays a separate prompt.

Outputs -> data/wallpaper_head/sheet_d_reverdict/report.{md,json}

    uv run python tools/wallpaper/sheet_d_reverdict.py
    uv run python tools/wallpaper/sheet_d_reverdict.py --limit 32   # bounded, writes scratch/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.corpus.q4_combined_readout import wilson                     # noqa: E402
from tools.scoring.winner_rule import (Metric, paired_bootstrap,        # noqa: E402
                                       point_block, verdict)
from tools.wallpaper import wallpaper_pins as WP                        # noqa: E402
from tools.wallpaper.build_blind_minibrot_sheet import SHEETS           # noqa: E402

SHEET = SHEETS["d"]
V3_CKPT = ROOT / WP.HEAD_CKPT_REL
V4B_DIR = ROOT / "data" / "wallpaper_head" / "v4b"
V4B_CKPT = V4B_DIR / "model_best.pt"
OUT_DIR = ROOT / "data" / "wallpaper_head" / "sheet_d_reverdict"

# The anchored bucket this slice exists to replace. The NUMBER is read from the report, not
# from here: this names WHERE, and `anchoring_price()` reads WHAT.
ANCHORED_REPORT = V4B_DIR / "report.json"
ANCHORED_ARM = "sheet_a_minibrot_maneuver"

# K=4 — three boundaries, two statistics each. Identical to `wallpaper_v4b_reads.METRICS`, so
# a cell here is the same cell there.
METRICS = (Metric("auc_ge3", "AUC>=3", "p_ge3", 3, "auc"),
           Metric("ap_ge3", "AP>=3", "p_ge3", 3, "ap"),
           Metric("auc_ge2", "AUC>=2", "p_ge2", 2, "auc"),
           Metric("ap_ge2", "AP>=2", "p_ge2", 2, "ap"),
           Metric("auc_ge4", "AUC>=4", "p_ge4", 4, "auc"),
           Metric("ap_ge4", "AP>=4", "p_ge4", 4, "ap"))

VOLUME_RATES = (0.05, 0.10, 0.20)

# The arm the rule reads. ONE population, and that is the point: sheet D was drawn to BE the
# motivating arm, so the motivating and pooled arms are the same rows and clause (a) reads
# "nothing on this slice is significantly worse" while clause (b) reads "something on it is
# significantly better". Declared here, above any number.
MOTIVATING_ARM = "blind_minibrot"


def log(m):
    print(m, flush=True)


@dataclass(frozen=True)
class DRow:
    image_id: str
    label: int
    jpg: Path
    vein: str
    partition: str
    flavor: str
    family: str


def load_rows(require_crops: bool = True) -> tuple[list, dict]:
    """`(rows, meta)` — sheet D's labeled rows. A missing sidecar is a HARD STOP with the
    merge command, not a short slice: half a labeled sheet silently answering a verdict
    question is the failure this whole file exists to avoid."""
    bdir = SHEET.batch_dir
    imgs = bdir / "images.jsonl"
    if not imgs.exists():
        raise SystemExit(f"[sheet-d] {imgs} absent — build the sheet first "
                         f"(tools/wallpaper/build_blind_minibrot_sheet.py)")
    sidecar = ROOT / SHEET.labels_export
    if not sidecar.exists():
        raise SystemExit(
            f"[sheet-d] no labels at {sidecar}.\nLabel the sheet, then merge:\n"
            f"  uv run python tools/wallpaper/merge_sitting.py --corpus wallpaper_corpus "
            f"--batch {SHEET.batch_id} --scores {SHEET.labels_export} --apply")
    labels = json.loads(sidecar.read_text(encoding="utf-8"))

    rows, n_total, n_unlabeled = [], 0, 0
    for line in imgs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_total += 1
        iid = r["image_id"]
        s = labels.get(iid)
        if s is None:
            n_unlabeled += 1
            continue
        jpg = bdir / "crops" / f"{iid}.jpg"
        if require_crops and not jpg.exists():
            raise FileNotFoundError(f"crop missing: {jpg}")
        prov = r["provenance"]
        rows.append(DRow(image_id=iid, label=int(s), jpg=jpg,
                         vein=prov.get("vein") or "?",
                         partition=prov.get("partition") or "?",
                         flavor=(prov.get("colorize") or {}).get("palette_flavor") or "?",
                         family=r["render"]["fractal_type"]))
    if not rows:
        raise SystemExit(f"[sheet-d] sidecar {sidecar} labels none of the batch's rows")
    return rows, {"batch_id": SHEET.batch_id, "sidecar": SHEET.labels_export,
                  "n_batch_rows": n_total, "n_labeled": len(rows),
                  "n_unlabeled": n_unlabeled,
                  "partial": bool(n_unlabeled),
                  "partial_note": (f"{n_unlabeled} of {n_total} rows carry no label — this "
                                   f"verdict is on the labeled subset and the subset is NOT "
                                   f"a random one (labeling runs in sheet order)")
                                  if n_unlabeled else None}


def score_with(ckpt: Path, rows, device: str) -> dict:
    from tools.wallpaper.report_v4_eval import load_head    # noqa: PLC0415 (torch import)
    score, cfg = load_head(ckpt, device)
    _cond, marg, ssum = score([str(r.jpg) for r in rows])
    return {"p_ge2": marg[:, 0], "p_ge3": marg[:, 1], "p_ge4": marg[:, 2], "rank": ssum,
            "_cfg": cfg}


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
    tp = int((labels[order] >= 3).sum())
    p, lo, hi = wilson(tp, k)
    return {"n_selected": k, "pass_rate": k / n, "tp": tp, "precision_ge3": p,
            "precision_lo": lo, "precision_hi": hi,
            "recall_ge3": tp / good if good else None,
            "frac_tier4": float((labels[order] == 4).mean()),
            "cut_at": float(s[order[-1]])}


def volume_matched(labels, base, cand) -> dict:
    """v3 and v4b at EQUAL selected volume — the only honest cross-head precision read.

    A CORN marginal is calibrated to its own training prior, so `p_ge3 > 0.90` is a point on
    v3's scale and something else on v4b's. The deployed gate is reported as the VOLUME v3
    passes at it; v4b is read at that same volume, never at that threshold."""
    n = len(labels)
    v = int((base["p_ge3"] > WP.GATE_THRESHOLD).sum())
    out = {"by_deployed_gate": {
        "threshold_on_v3": WP.GATE_THRESHOLD, "matched_volume": v,
        "v3": top_block(labels, base["p_ge3"], v),
        "v4b": top_block(labels, cand["p_ge3"], v)}, "by_fixed_rate": {}}
    for rate in VOLUME_RATES:
        k = int(round(rate * n))
        out["by_fixed_rate"][f"{rate:.2f}"] = {
            "matched_volume": k,
            "v3": top_block(labels, base["p_ge3"], k),
            "v4b": top_block(labels, cand["p_ge3"], k)}
    return out


def anchoring_price(here: dict) -> dict:
    """THE anchored-vs-blind comparison, with the anchored side DERIVED off the committed
    (28) report rather than restated. A missing report is reported as UNKNOWN, not as absent:
    the price is the headline of this file and a silently-dropped one reads as no price."""
    out = {"blind_slice": {"arm": MOTIVATING_ARM,
                           "v3_auc_ge3": here.get("auc_ge3"),
                           "v3_ap_ge3": here.get("ap_ge3")},
           "anchored_slice": {"report": ANCHORED_REPORT.relative_to(ROOT).as_posix(),
                              "arm": ANCHORED_ARM},
           "what": "v3's AUC>=3 on rows whose labels it SUGGESTED, against v3's AUC>=3 on "
                   "fresh rows of the same population labeled blind. The gap is the "
                   "anchoring price — how much of the 0.965 was agreement rather than "
                   "quality (classifier_retrain_protocol.md §2b)."}
    if not ANCHORED_REPORT.exists():
        out["anchored_slice"]["status"] = "UNKNOWN — the (28) v4b report is not on disk"
        return out
    R = json.loads(ANCHORED_REPORT.read_text(encoding="utf-8"))
    blk = (R.get("motivating") or {}).get(ANCHORED_ARM) or \
          (R.get("diagnostic") or {}).get(ANCHORED_ARM)
    if blk is None:
        out["anchored_slice"]["status"] = f"UNKNOWN — no arm {ANCHORED_ARM!r} in that report"
        return out
    out["anchored_slice"].update({
        "status": "read", "n": blk["n"], "tiers": blk["tiers"]["hist"],
        "v3_auc_ge3": blk["v3"]["auc_ge3"], "v3_ap_ge3": blk["v3"]["ap_ge3"]})
    a, b = out["anchored_slice"]["v3_auc_ge3"], out["blind_slice"]["v3_auc_ge3"]
    if a is not None and b is not None:
        out["delta_auc_ge3"] = float(b - a)
        out["reading"] = ("the blind slice is the LOWER number; the difference is what the "
                          "prefilled suggestion bought v3 on a slice it was measured on"
                          if b < a else
                          "the blind slice is NOT lower — the anchoring did not inflate v3 "
                          "on this population, which is itself the finding")
    return out


def build(rows, base, cand, seed_scores, meta, *, draws, seed) -> dict:
    labels = np.array([r.label for r in rows])
    all_mask = np.ones(len(rows), bool)
    vein = np.array([r.vein for r in rows])
    part = np.array([r.partition for r in rows])

    R = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": "uv run python tools/wallpaper/sheet_d_reverdict.py",
        "heads": {"v3": {"ckpt": WP.HEAD_CKPT_REL, "role": "incumbent (LIVE pin)"},
                  "v4b": {"ckpt": "data/wallpaper_head/v4b/model_best.pt",
                          "role": "the (28) from-scratch candidate, STAGED and unadopted"}},
        "live_pin": WP.HEAD_CKPT_REL, "live_gate": WP.GATE_THRESHOLD,
        "moves_nothing": "no pin, gate or floor is written by this file",
        "slice": {**meta, "n": len(rows),
                  "tiers": tier_dist(labels),
                  "by_vein": dict(sorted(Counter(r.vein for r in rows).items())),
                  "by_partition": dict(sorted(Counter(r.partition for r in rows).items())),
                  "by_family": dict(sorted(Counter(r.family for r in rows).items())),
                  "blind": "no suggestion was served, no score ordered the page",
                  "eval_only": True},
        "bootstrap": {"draws": draws, "seed": seed, "kind": "paired over slice rows"},
        "pre_declared": {"motivating": [MOTIVATING_ARM], "no_worse": [MOTIVATING_ARM],
                         "note": "sheet D was drawn to BE the motivating arm, so the two are "
                                 "the same rows: clause (a) reads 'nothing here is "
                                 "significantly worse', clause (b) 'something here is "
                                 "significantly better'."},
    }

    R["motivating"] = {MOTIVATING_ARM: arm_block(labels, base, cand, all_mask,
                                                 draws=draws, seed=seed)}
    R["no_worse"] = R["motivating"]
    diag = {}
    for name, arr in (("vein", vein), ("partition", part)):
        for v in sorted(set(arr.tolist())):
            m = arr == v
            if m.sum() >= 20:            # below this a paired CI on 6 metrics says nothing
                diag[f"{name}:{v}"] = arm_block(labels, base, cand, m, draws=draws, seed=seed)
    R["diagnostic"] = diag

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
            "note": "every v4b seed was selected on the (28) POOLED eval, not on this slice — "
                    "so sheet D is the first held-out population any of them has seen and the "
                    "staged max is NOT optimistic here, unlike in the (28) report."}

    R["score_scale"] = {
        "why": "CORN marginals are calibrated to the training prior, so no raw-threshold "
               "comparison appears in this report.",
        "quantiles": {f"q{int(q*100)}": {"v3": float(np.quantile(base["p_ge3"], q)),
                                         "v4b": float(np.quantile(cand["p_ge3"], q))}
                      for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "pass_rate_at_the_live_gate": {
            "threshold": WP.GATE_THRESHOLD,
            "v3": float((base["p_ge3"] > WP.GATE_THRESHOLD).mean()),
            "v4b": float((cand["p_ge3"] > WP.GATE_THRESHOLD).mean()),
            "read_as": "volume, NOT quality"}}
    R["volume_matched"] = volume_matched(labels, base, cand)

    R["winner_rule"] = verdict(
        {k: v["delta_ci"] for k, v in R["no_worse"].items()},
        {k: v["delta_ci"] for k, v in R["motivating"].items()},
        pooled_arm=MOTIVATING_ARM, baseline="v3", candidate="v4b")
    R["winner_rule"]["scope"] = (
        "THIS SLICE ONLY. The (28) winner rule ran over the whole six-batch eval union; this "
        "is the motivating arm re-drawn without the anchoring, and it decides that arm, not "
        "the flip.")
    R["winner_rule"]["adoption"] = ("NOT decided here. BUILD != FLIP: adoption is a separate "
                                    "prompt after Matt reads this verdict.")
    R["anchoring_price"] = anchoring_price(R["motivating"][MOTIVATING_ARM]["v3"])
    return R


def md(R) -> str:
    L = []
    A = L.append
    wr = R["winner_rule"]
    s = R["slice"]
    A("# SHEET D re-verdict — wallpaper v3 vs v4b on BLIND minibrot labels\n")
    A(f"Generated {R['generated']} · `{R['command']}`\n")
    A(f"**Clause (a) no-worse {'PASS' if wr['clause_a']['pass'] else 'FAIL'} · "
      f"clause (b) motivating {'PASS' if wr['clause_b']['pass'] else 'FAIL'} → "
      f"WINNER {wr['winner']}**\n")
    A(f"> {wr['scope']}  \n> {wr['adoption']}\n")
    A(f"Slice: **{s['n']} labeled rows** of {s['n_batch_rows']} in `{s['batch_id']}`; "
      f"tiers {s['tiers']['hist']}; veins {s['by_vein']}; partitions {s['by_partition']}. "
      f"Blind, eval-only.\n")
    if s.get("partial_note"):
        A(f"**PARTIAL:** {s['partial_note']}\n")

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

    table("THE MOTIVATING ARM, re-drawn blind", R["motivating"], [MOTIVATING_ARM])
    if R["diagnostic"]:
        table("Diagnostics (vote on nothing)", R["diagnostic"], sorted(R["diagnostic"]))

    A("\n### Every metric on the arm\n")
    b = R["motivating"][MOTIVATING_ARM]
    A("| metric | v3 | v4b | Δ 95% CI |")
    A("|---|---:|---:|---|")
    for m in METRICS:
        c = b["delta_ci"][m.key]
        tag = "" if not c or c["n_draws"] == 0 else (
            " **worse**" if c["significantly_worse"] else
            (" **better**" if c["significantly_better"] else ""))
        rng = "n/a" if not c or c["n_draws"] == 0 else f"[{c['lo']:+.3f}, {c['hi']:+.3f}]"
        def cell(x):
            return "—" if x is None else f"{x:.3f}"
        A(f"| {m.label} | {cell(b['v3'][m.key])} | {cell(b['v4b'][m.key])} | {rng}{tag} |")

    ap = R["anchoring_price"]
    A("\n## The anchoring price\n")
    A(f"{ap['what']}\n")
    an, bl = ap["anchored_slice"], ap["blind_slice"]
    A("| slice | labels elicited | n | v3 AUC≥3 | v3 AP≥3 |")
    A("|---|---|---:|---:|---:|")
    if an.get("status") == "read":
        A(f"| sheet A `{an['arm']}` | v3's tier PREFILLED, page sorted by v3 | {an['n']} "
          f"| {an['v3_auc_ge3']:.3f} | {an['v3_ap_ge3']:.3f} |")
    else:
        A(f"| sheet A `{an['arm']}` | {an.get('status', 'UNKNOWN')} | — | — | — |")
    bl_auc = "—" if bl["v3_auc_ge3"] is None else f"{bl['v3_auc_ge3']:.3f}"
    bl_ap = "—" if bl["v3_ap_ge3"] is None else f"{bl['v3_ap_ge3']:.3f}"
    A(f"| sheet D `{bl['arm']}` | BLIND, shuffled | {R['slice']['n']} "
      f"| {bl_auc} | {bl_ap} |")
    if "delta_auc_ge3" in ap:
        A(f"\n**Δ AUC≥3 (blind − anchored) = {ap['delta_auc_ge3']:+.3f}** — {ap['reading']}\n")

    vm = R["volume_matched"]["by_deployed_gate"]
    A("\n## Volume-matched (never a shared raw threshold)\n")
    A(f"The deployed gate passes **{vm['matched_volume']}** of {R['slice']['n']} rows on v3 "
      f"(`p_ge3 > {vm['threshold_on_v3']}`); v4b is read at that same volume.\n")
    A("| volume | v3 precision≥3 | v4b precision≥3 | v3 %tier4 | v4b %tier4 | n |")
    A("|---|---:|---:|---:|---:|---:|")
    if vm["v3"] and vm["v4b"]:
        A(f"| deployed gate | {vm['v3']['precision_ge3']:.3f} "
          f"| {vm['v4b']['precision_ge3']:.3f} | {vm['v3']['frac_tier4']:.3f} "
          f"| {vm['v4b']['frac_tier4']:.3f} | {vm['matched_volume']} |")
    for name, blk in R["volume_matched"]["by_fixed_rate"].items():
        if blk["v3"] and blk["v4b"]:
            A(f"| top {name} | {blk['v3']['precision_ge3']:.3f} "
              f"| {blk['v4b']['precision_ge3']:.3f} | {blk['v3']['frac_tier4']:.3f} "
              f"| {blk['v4b']['frac_tier4']:.3f} | {blk['matched_volume']} |")

    if "v4b_seed_band" in R:
        A("\n## v4b five-seed band on this slice\n")
        A(f"{R['v4b_seed_band']['note']}\n")
        A("| metric | mean ± sd | per seed |")
        A("|---|---|---|")
        for m in METRICS:
            bb = R["v4b_seed_band"]["mean_sd"][m.key]
            vals = " ".join(f"{x[m.key]:.3f}" if x[m.key] is not None else "—"
                            for x in R["v4b_seed_band"]["per_seed"])
            A(f"| {m.label} | {bb['mean']:.3f} ± {bb['sd']:.3f} | {vals} |")
    A(f"\n{wr['multiplicity_note']}\n")
    return "\n".join(L) + "\n"


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="rows — bounded end-to-end; writes to scratch/, stamped incomplete")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    for p in (V3_CKPT, V4B_CKPT):
        if not p.exists():
            raise SystemExit(f"[sheet-d] missing checkpoint {p}")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows, meta = load_rows()
    if a.limit:
        rows = rows[:a.limit]
        meta = dict(meta, n_labeled=len(rows))
    log(f"[sheet-d] {len(rows)} labeled rows on {device}")

    base = score_with(V3_CKPT, rows, device)
    cand = score_with(V4B_CKPT, rows, device)
    seed_scores = {}
    if not a.limit:
        for d in sorted(V4B_DIR.glob("seed_*")):
            ck = d / "model_best.pt"
            if ck.exists():
                seed_scores[int(d.name.split("_")[1])] = score_with(ck, rows, device)
                log(f"[sheet-d] scored {d.name}")

    R = build(rows, base, cand, seed_scores, meta, draws=a.draws, seed=a.seed)
    R["incomplete"] = bool(a.limit)
    out = (ROOT / "scratch" / "sheet_d_reverdict") if a.limit else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(md(R), encoding="utf-8")
    wr = R["winner_rule"]
    log(f"[sheet-d] clause a {wr['clause_a']['pass']} / clause b {wr['clause_b']['pass']} "
        f"-> WINNER {wr['winner']}")
    if "delta_auc_ge3" in R["anchoring_price"]:
        log(f"[sheet-d] anchoring price: Δ AUC>=3 (blind - anchored) "
            f"{R['anchoring_price']['delta_auc_ge3']:+.3f}")
    log(f"-> {out / 'report.md'}")


if __name__ == "__main__":
    main()
