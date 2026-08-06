"""v3 vs v4 wallpaper-head eval report — same slices, same harness, n stated everywhere.

Why this is not inside the trainer: a head-to-head needs BOTH checkpoints scored on the
SAME crops through the SAME deploy transform. v3's own eval scores were frozen against
crops that have since been deleted and re-rendered, and the fresh-era rows carry a
stamped `head_v3.p_ge3` produced at sheet-build time — either would be a comparison
across two rendering events. So both heads are re-scored here, from their .pt files,
over the crops on disk right now.

What it reports (report.json + report.md under scratch/wallpaper_v4_report/):

  * The two eval slices, each for BOTH heads: AP at >=2/>=3/>=4, AUC>=3, and the
    precision-of-passers ladder at p_ge3 > 0.5 / 0.90 / 0.99 (the deployed gate sits at
    0.90; the ladder is what `wallpaper_pins.GATE_THRESHOLD` records).
      - old_era: the 686-row July slice, byte-identical across v2/v3/v4.
      - fresh_era: the stamped 2026-08-05 eval side.
  * The fresh side broken out by coloring regime (pool_draw vs colorize_path) and by
    intake vein (human_q3plus / q4_harvest / machine_admitted).
  * THE BLIND-SPOT CHECK: every fresh-era row the deployed v3 gated out at p_ge3 < 0.05
    despite a human label >=3, listed individually with its v4 score. Rows on the TRAIN
    side are marked and excluded from the headline — v4 was fit on them, so their score
    is memory, not evidence. The eval-side subset is the qualitative pass/fail.
  * Fresh-era tier-4 counts (both batches, train+eval).

    uv run python tools/wallpaper/report_v4_eval.py
    uv run python tools/wallpaper/report_v4_eval.py --limit 64     # bounded end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from classifier.data import Transform                              # noqa: E402
from classifier.model import BACKBONE, score_from_logits           # noqa: E402
from classifier.eval import _ap                                    # noqa: E402
from classifier.train_wallpaper_v4 import load_rows, split_union   # noqa: E402

OUT = ROOT / "scratch" / "wallpaper_v4_report"
HEADS = {"v3": ROOT / "data" / "wallpaper_head" / "v3" / "model_best.pt",
         "v4": ROOT / "data" / "wallpaper_head" / "v4" / "model_best.pt"}
LADDER = (0.5, 0.90, 0.99)      # the operating points wallpaper_pins.GATE_THRESHOLD records
BLINDSPOT_V3_MAX = 0.05         # "v3 gated it out" for the blind-spot population


# --------------------------------------------------------------------------- #
def load_head(path: Path, device):
    """Checkpoint -> (score(paths) -> (cond, marg, ssum), config). Mirrors
    emit_v1.load_v2_scorer: cond = sigmoid(logits) CONDITIONAL, marg = cumprod."""
    import timm
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    K = int(cfg["num_classes"])
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=K - 1,
                              drop_rate=cfg.get("drop_rate", 0.2),
                              drop_path_rate=cfg.get("drop_path_rate", 0.1))
    model.load_state_dict(ck["state_dict"])
    model = model.eval().to(device)
    tf = Transform(geometry=cfg["geometry"], interp=cfg["interpolation"],
                   mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False)

    @torch.no_grad()
    def score(paths, batch_size=32):
        cond = np.zeros((len(paths), K - 1), dtype=np.float64)
        ssum = np.zeros(len(paths), dtype=np.float64)
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            batch = []
            for p in chunk:
                with Image.open(p) as im:
                    im.load()
                    batch.append(tf(im.convert("RGB")))
            logits = model(torch.stack(batch).to(device)).float()
            cond[i:i + len(chunk)] = torch.sigmoid(logits).cpu().numpy()
            ssum[i:i + len(chunk)] = score_from_logits(logits, "ordinal").cpu().numpy()
        return cond, np.cumprod(cond, axis=1), ssum

    return score, cfg


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return None if y.min() == y.max() else float(roc_auc_score(y, s))


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def slice_block(labels, marg, mask):
    """One head on one slice. `n` is carried on every block — a rate without its
    denominator is the thing this report exists to avoid."""
    if mask.sum() == 0:
        return None
    lb = np.asarray(labels)[mask]
    m = marg[mask]
    nb, gd, ex = (lb >= 2).astype(int), (lb >= 3).astype(int), (lb >= 4).astype(int)
    ladder = {}
    for t in LADDER:
        fires = m[:, 1] > t
        ladder[f"{t:g}"] = {
            "n_fire": int(fires.sum()),
            "fire_frac": float(fires.mean()),
            "precision_ge3": (float(gd[fires].mean()) if fires.sum() else None),
            "precision_ge2": (float(nb[fires].mean()) if fires.sum() else None),
            "recall_ge3": (float((fires & (gd == 1)).sum() / gd.sum()) if gd.sum() else None),
            "frac_tier4": (float((lb[fires] == 4).mean()) if fires.sum() else None),
        }
    return {
        "n": int(mask.sum()), "n_ge2": int(nb.sum()), "n_ge3": int(gd.sum()),
        "n_ge4": int(ex.sum()),
        "tier_hist": {int(k): int(v) for k, v in sorted(Counter(lb.tolist()).items())},
        "ap_ge2": _f(_ap(nb, m[:, 0])),
        "ap_ge3": _f(_ap(gd, m[:, 1])),
        "ap_ge4": (_f(_ap(ex, m[:, 2])) if ex.sum() else None),
        "auc_ge3": _auc(gd, m[:, 1]),
        "precision_of_passers": ladder,
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0,
                    help="cap scored eval rows (bounded end-to-end; report is NOT usable)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    missing = {v: str(p) for v, p in HEADS.items() if not p.exists()}
    if missing:
        raise SystemExit(f"checkpoint(s) missing: {missing}")

    rows = load_rows()
    tr, ev, _, _, _, old_slice_ids, conflicts = split_union(rows)
    train_ids = {r.image_id for r in tr}
    if args.limit:
        ev = ev[:args.limit]
        print(f"[report] --limit {args.limit}: BOUNDED — numbers are not the report", flush=True)

    # The blind-spot population: fresh-era, human label >=3, deployed-v3 stamped
    # p_ge3 < 0.05. Drawn from ALL rows (train + eval) because that is the population
    # the question is about; the sides are reported separately, never pooled.
    blind = [r for r in rows
             if r.era == "fresh" and r.label >= 3
             and r.v3_p_ge3 is not None and r.v3_p_ge3 < BLINDSPOT_V3_MAX]
    blind_ids = {r.image_id for r in blind}
    print(f"[report] eval n={len(ev)}  blind-spot n={len(blind)} "
          f"({sum(1 for r in blind if r.image_id in train_ids)} on the TRAIN side)", flush=True)

    # Score once per head over eval ∪ blind-spot, then index by image_id.
    scored = {r.image_id: r for r in ev}
    for r in blind:
        scored.setdefault(r.image_id, r)
    todo = list(scored.values())
    paths = [str(r.jpg) for r in todo]

    per_head = {}
    for ver, ckpt in HEADS.items():
        score, cfg = load_head(ckpt, args.device)
        print(f"[report] scoring {len(paths)} crops with {ver} "
              f"({ckpt.relative_to(ROOT).as_posix()}, K={cfg['num_classes']}, "
              f"seed={cfg.get('seed')}, best_epoch={cfg.get('best_epoch')})", flush=True)
        cond, marg, ssum = score(paths)
        per_head[ver] = {"marg": marg, "ssum": ssum,
                         "cfg": {k: cfg.get(k) for k in
                                 ("model", "num_classes", "seed", "best_epoch", "selection")}}

    idx = {r.image_id: i for i, r in enumerate(todo)}
    ev_i = np.asarray([idx[r.image_id] for r in ev])
    labels = np.asarray([r.label for r in ev])
    era = np.asarray([r.era for r in ev])
    cs = np.asarray([r.coloring_source for r in ev])
    sg = np.asarray([r.source_group for r in ev])
    slices = {
        "overall": np.ones(len(ev), bool),
        "old_era_686": era == "july",
        "fresh_era": era == "fresh",
        "fresh_pool_draw": cs == "pool_draw",
        "fresh_colorize_path": cs == "colorize_path",
        "fresh_human_q3plus": sg == "human_q3plus",
        "fresh_q4_harvest": sg == "q4_harvest",
        "fresh_machine_admitted": sg == "machine_admitted",
    }
    report = {"slices": {}, "heads": {v: per_head[v]["cfg"] for v in HEADS},
              "gate_note": "precision_of_passers is cut on MARGINAL p_ge3; deployed gate = 0.90",
              "split_conflicts": {"n_locations": len(conflicts),
                                  "n_rows_moved": sum(len(c["moved_image_ids"]) for c in conflicts)}}
    for name, mask in slices.items():
        report["slices"][name] = {
            v: slice_block(labels, per_head[v]["marg"][ev_i], mask) for v in HEADS}

    # --- score-scale comparison (the confound that makes a fixed-t table misread) ---
    # A CORN head's marginal p_ge3 is calibrated to the TRAIN prior, and v4's train prior
    # moved (the fresh sheet is bin-stratified and 49.5% tier-1). So a fixed threshold
    # compares two different scales, and "v4 fires less" is not "v4 is more selective".
    # The scale-free reading: hold the FIRING VOLUME fixed at what v3 does at 0.90 and
    # compare precision there. This is a report statistic, not a re-derived floor.
    report["score_scale"] = {}
    for name, mask in slices.items():
        lb = labels[mask]
        gd = (lb >= 3).astype(int)
        ent = {"n": int(mask.sum()), "n_ge3": int(gd.sum())}
        for v in HEADS:
            p = per_head[v]["marg"][ev_i][mask][:, 1]
            ent[v] = {"p_ge3_quantiles": {q: float(np.quantile(p, q))
                                          for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                      "frac_over_0.9": float((p > 0.90).mean())}
        # volume-matched operating point: k = #{v3 fires at 0.90}; take each head's top k.
        p3 = per_head["v3"]["marg"][ev_i][mask][:, 1]
        k = int((p3 > 0.90).sum())
        ent["volume_matched"] = {"k": k, "note": "top-k by each head's own p_ge3, k = v3's 0.90 volume"}
        if k:
            for v in HEADS:
                p = per_head[v]["marg"][ev_i][mask][:, 1]
                top = np.argsort(-p)[:k]
                ent["volume_matched"][v] = {
                    "threshold": float(p[top].min()),
                    "precision_ge3": float(gd[top].mean()),
                    "recall_ge3": (float(gd[top].sum() / gd.sum()) if gd.sum() else None),
                    "frac_tier4": float((lb[top] == 4).mean()),
                }
        report["score_scale"][name] = ent

    # --- blind-spot listing ---
    # Each row also gets its PERCENTILE within the head's own fresh-eval-side score
    # distribution. Absolute p_ge3 is not comparable across the two heads (see
    # score_scale above); "where does this row sit in the ranking" is, and the blind
    # spot was a claim about ranking — v3 put these at the very bottom.
    fresh_mask = era == "fresh"
    ref = {v: np.sort(per_head[v]["marg"][ev_i][fresh_mask][:, 1]) for v in HEADS}

    def pct(v, x):
        return float(np.searchsorted(ref[v], x, side="right") / len(ref[v]))

    bs = []
    for r in blind:
        i = idx[r.image_id]
        bs.append({
            "v3_pct_in_fresh_eval": pct("v3", per_head["v3"]["marg"][i, 1]),
            "v4_pct_in_fresh_eval": pct("v4", per_head["v4"]["marg"][i, 1]),
            "image_id": r.image_id, "batch": r.batch, "label": r.label,
            "coloring_source": r.coloring_source, "source_group": r.source_group,
            "family": r.family, "side": "train" if r.image_id in train_ids else "eval",
            "v3_stamped_p_ge3": r.v3_p_ge3,
            "v3_p_ge3": float(per_head["v3"]["marg"][i, 1]),
            "v4_p_ge3": float(per_head["v4"]["marg"][i, 1]),
            "v4_p_ge2": float(per_head["v4"]["marg"][i, 0]),
            "v4_score": float(per_head["v4"]["ssum"][i]),
        })
    bs.sort(key=lambda d: (d["side"] != "eval", -d["v4_p_ge3"]))
    ev_bs = [d for d in bs if d["side"] == "eval"]
    report["blind_spot"] = {
        "definition": (f"fresh-era rows with human label >=3 and the STAMPED deployed-v3 "
                       f"p_ge3 < {BLINDSPOT_V3_MAX}"),
        "n_total": len(bs), "n_eval_side": len(ev_bs), "n_train_side": len(bs) - len(ev_bs),
        "eval_side_v4_raised": sum(1 for d in ev_bs if d["v4_p_ge3"] > d["v3_p_ge3"]),
        "eval_side_recovered_at_0.90": sum(1 for d in ev_bs if d["v4_p_ge3"] > 0.90),
        "eval_side_recovered_at_0.50": sum(1 for d in ev_bs if d["v4_p_ge3"] > 0.50),
        "eval_side_still_under_0.05": sum(1 for d in ev_bs if d["v4_p_ge3"] < 0.05),
        "eval_side_median_pct_v3": float(np.median([d["v3_pct_in_fresh_eval"] for d in ev_bs])),
        "eval_side_median_pct_v4": float(np.median([d["v4_pct_in_fresh_eval"] for d in ev_bs])),
        # The one number that answers "would the retrain have EMITTED these?" without
        # re-deriving a floor: each head's own fresh-era threshold at v3's 0.90 volume.
        # v3's count is 0 by construction — the population is defined by v3 rejecting it.
        "eval_side_over_volume_matched_v4": sum(
            1 for d in ev_bs
            if d["v4_p_ge3"] >= report["score_scale"]["fresh_era"]["volume_matched"]["v4"]["threshold"]),
        "volume_matched_threshold_v4": report["score_scale"]["fresh_era"]["volume_matched"]["v4"]["threshold"],
        "threshold_caveat": ("the *_at_0.90 / *_at_0.50 counts compare v4 against a "
                             "threshold tuned on v3's score scale, and the two scales are "
                             "not the same (report.score_scale). The percentile columns "
                             "are the scale-free reading, and the blind spot was a claim "
                             "about ranking."),
        "train_side_caveat": ("v4 was FIT on the train-side rows — their scores are "
                              "memorization, listed for completeness and excluded from "
                              "every count above."),
        "rows": bs,
    }

    # --- fresh-era tier-4 census (train+eval, both batches) ---
    fresh = [r for r in rows if r.era == "fresh"]
    ev_ids = {r.image_id for r in ev}
    report["fresh_tier4_census"] = {
        "n_fresh_rows": len(fresh),
        "n_tier4": sum(1 for r in fresh if r.label == 4),
        "by_batch": dict(Counter(r.batch for r in fresh if r.label == 4)),
        "by_coloring_source": dict(Counter(r.coloring_source for r in fresh if r.label == 4)),
        "by_side": dict(Counter(("eval" if r.image_id in ev_ids else "train")
                                for r in fresh if r.label == 4)),
        "by_source_group": dict(Counter(r.source_group for r in fresh if r.label == 4)),
        "tier4_rate": round(sum(1 for r in fresh if r.label == 4) / len(fresh), 4),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    write_md(report, out / "report.md")
    print(f"[report] wrote {out / 'report.json'} and {out / 'report.md'}", flush=True)


def write_md(rep, path: Path):
    def n(x, d=3):
        return "n/a" if x is None else f"{x:.{d}f}"
    L = ["# Wallpaper head v3 vs v4 — same slices, one harness", ""]
    L.append(f"Both heads re-scored from their `.pt` over the crops on disk now "
             f"(v4 seed {rep['heads']['v4']['seed']}, best epoch "
             f"{rep['heads']['v4']['best_epoch']}; v3 seed {rep['heads']['v3']['seed']}, "
             f"best epoch {rep['heads']['v3']['best_epoch']}). "
             f"Precision-of-passers is cut on marginal `p_ge3`; the deployed gate is 0.90.")
    L += ["", "## Rank metrics", "",
          "| slice | n | n>=3 | n>=4 | head | AP>=2 | AP>=3 | AP>=4 | AUC>=3 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, per in rep["slices"].items():
        b3, b4 = per["v3"], per["v4"]
        if b3 is None:
            continue
        for ver, b in (("v3", b3), ("v4", b4)):
            L.append(f"| {name if ver=='v3' else ''} | {b['n'] if ver=='v3' else ''} | "
                     f"{b['n_ge3'] if ver=='v3' else ''} | {b['n_ge4'] if ver=='v3' else ''} | "
                     f"**{ver}** | {n(b['ap_ge2'])} | {n(b['ap_ge3'])} | {n(b['ap_ge4'])} | "
                     f"{n(b['auc_ge3'])} |")
    L += ["", "## Precision of passers (marginal p_ge3 > t)", "",
          "| slice | head | t | fires | frac | precision>=3 | precision>=2 | recall>=3 |",
          "|---|---|---|---|---|---|---|---|"]
    for name, per in rep["slices"].items():
        for ver in ("v3", "v4"):
            b = per[ver]
            if b is None:
                continue
            for t, d in b["precision_of_passers"].items():
                L.append(f"| {name} | {ver} | {t} | {d['n_fire']}/{b['n']} | "
                         f"{n(d['fire_frac'])} | {n(d['precision_ge3'])} | "
                         f"{n(d['precision_ge2'])} | {n(d['recall_ge3'])} |")
    L += ["", "## Score scale — why a fixed threshold misreads this pair", "",
          "A CORN marginal is calibrated to the TRAIN prior, and v4's moved (the fresh "
          "sheet is bin-stratified and 49.5% tier-1). The volume-matched columns hold "
          "firing volume at what v3 does at 0.90 and compare precision there — the "
          "scale-free read. No production floor is re-derived here.", "",
          "| slice | n | head | median p_ge3 | p90 | p99 | frac>0.9 | vol-matched k | "
          "vm threshold | vm precision>=3 | vm frac tier4 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in rep["score_scale"].items():
        vm = e["volume_matched"]
        for ver in ("v3", "v4"):
            q = e[ver]["p_ge3_quantiles"]
            m = vm.get(ver)
            L.append(
                f"| {name if ver=='v3' else ''} | {e['n'] if ver=='v3' else ''} | **{ver}** | "
                f"{n(q['0.5'] if '0.5' in q else q[0.5],4)} | {n(q.get('0.9', q.get(0.9)),4)} | "
                f"{n(q.get('0.99', q.get(0.99)),4)} | {n(e[ver]['frac_over_0.9'])} | "
                f"{vm['k'] if ver=='v3' else ''} | "
                f"{n(m['threshold'],4) if m else 'n/a'} | "
                f"{n(m['precision_ge3']) if m else 'n/a'} | "
                f"{n(m['frac_tier4']) if m else 'n/a'} |")
    bsd = rep["blind_spot"]
    L += ["", "## Blind-spot check", "", bsd["definition"] + ".",
          f"n={bsd['n_total']} ({bsd['n_eval_side']} eval-side, {bsd['n_train_side']} train-side).",
          "",
          f"**Scale-free (the claim the blind spot actually made):** v4 raises "
          f"{bsd['eval_side_v4_raised']}/{bsd['n_eval_side']} of the eval-side rows. Median "
          f"percentile within each head's own fresh-eval score distribution moves "
          f"{bsd['eval_side_median_pct_v3']:.1%} -> {bsd['eval_side_median_pct_v4']:.1%}. "
          f"At v4's volume-matched fresh-era threshold "
          f"({bsd['volume_matched_threshold_v4']:.4f} — the point where v4 fires as often "
          f"as v3 does at 0.90), **{bsd['eval_side_over_volume_matched_v4']}/"
          f"{bsd['n_eval_side']}** of them would now be emitted; v3's count there is 0 by "
          f"construction, since the population is defined by v3 rejecting it.",
          "",
          f"**Against v3's thresholds** (see the caveat): "
          f"{bsd['eval_side_recovered_at_0.90']}/{bsd['n_eval_side']} clear 0.90, "
          f"{bsd['eval_side_recovered_at_0.50']}/{bsd['n_eval_side']} clear 0.50, "
          f"{bsd['eval_side_still_under_0.05']} still below 0.05.",
          "", bsd["threshold_caveat"], "", bsd["train_side_caveat"], "",
          "| image_id | side | label | coloring | source | v3 p_ge3 | v4 p_ge3 | v3 pct | "
          "v4 pct | v4 score |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in bsd["rows"]:
        L.append(f"| {d['image_id']} | {d['side']} | {d['label']} | {d['coloring_source']} | "
                 f"{d['source_group']} | {d['v3_p_ge3']:.4f} | {d['v4_p_ge3']:.4f} | "
                 f"{d['v3_pct_in_fresh_eval']:.1%} | {d['v4_pct_in_fresh_eval']:.1%} | "
                 f"{d['v4_score']:.3f} |")
    c = rep["fresh_tier4_census"]
    L += ["", "## Fresh-era tier-4 census (train + eval, both batches)", "",
          f"{c['n_tier4']} / {c['n_fresh_rows']} rows ({c['tier4_rate']:.1%}).",
          f"By batch: {c['by_batch']}. By coloring: {c['by_coloring_source']}. "
          f"By side: {c['by_side']}. By source: {c['by_source_group']}.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
