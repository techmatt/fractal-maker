#!/usr/bin/env python
"""Score the backbone-comparison arms and build the chart. Bars LOADED from the prereg.

Two subcommands, because scoring is the expensive half and a partial round must be
reportable without re-running it:

  score   one forward pass per arm over the SAME 2,860 v11 canonical eval tiles, frozen
          into data/backbone_search/eval_scores_backbone_v1.jsonl with one column block
          per (arm, seed). Also times the END-TO-END score path (decode + deploy
          transform + forward) on 1,000 tiles — the cost a ledger rescore actually pays.
  report  reads that file and emits results.json + results.md + the figure. No model is
          loaded, so a cut can be re-cut in seconds.

Everything the report is allowed to claim comes out of `prereg_backbone_v1.json`: the
PRIMARY population, the slice list, the min-positives rule, the bootstrap spec and the
honesty rule. Nothing here restates a threshold.

  uv run python tools/backbone_search/eval_arms.py score
  uv run python tools/backbone_search/eval_arms.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for sub in ("tools", "tools/scoring", "tools/v7"):
    sys.path.insert(0, str(ROOT / sub))

import partitions as P  # noqa: E402
import paths  # noqa: E402
from backbone_search.arms import ARMS, CONTROL  # noqa: E402
from eval_delong import delong_paired  # reuse — do NOT reimplement DeLong  # noqa: E402
from eval_model import load_model  # THE shared eval loader  # noqa: E402

PREREG_REL = "data/backbone_search/prereg_backbone_v1.json"
SCORES_REL = "data/backbone_search/eval_scores_backbone_v1.jsonl"
RESULTS_REL = "data/backbone_search/results_backbone_v1.json"
RESULTS_MD_REL = "data/backbone_search/results_backbone_v1.md"
FIGURE_REL = "data/backbone_search/quality_vs_throughput.png"
SELECTION_SOURCES = ("prospect_census", "loose0_v3_floor")
THROUGHPUT_N = 1000


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def fast_auc(y, s):
    """Binary AUC by mid-rank (ties handled), or None when one class is empty.

    Local rather than `eval_model.q_auc` because the bootstrap calls it ~10^5 times and
    sklearn's per-call overhead dominates there; checked equal to it in test_eval_arms.py.
    """
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(np.asarray(s, float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def decode_tier(probs):
    """CORN's own rank-consistent decode: tier = 1 + #{k: sigma(logit_k) > 0.5}.

    Parameter-free ON PURPOSE — a per-arm fitted cutpoint would be a second moved
    variable, and the agreement columns are meant to read the head, not a calibration."""
    return 1 + (np.asarray(probs) > 0.5).sum(axis=1)


def metrics_block(labels, probs):
    """Every declared metric for one arm on one slice. probs: (N, K-1) = P(>=2,>=3,>=4)."""
    labels = np.asarray(labels)
    tier = decode_tier(probs)
    out = {"n": int(len(labels))}
    for k, thr in enumerate((2, 3, 4)):
        y = (labels >= thr).astype(int)
        out[f"n_ge{thr}"] = int(y.sum())
        out[f"auc_ge{thr}"] = fast_auc(y, probs[:, k])
        out[f"calib_gap_ge{thr}"] = (float(probs[:, k].mean() - y.mean())
                                     if len(y) else None)
    out["exact_agree"] = float((tier == labels).mean())
    out["adj_agree"] = float((np.abs(tier - labels) <= 1).mean())
    hp = np.array([(tier == c).mean() for c in (1, 2, 3, 4)])
    hl = np.array([(labels == c).mean() for c in (1, 2, 3, 4)])
    out["prior_tv_distance"] = float(np.abs(hp - hl).sum() / 2)
    out["decoded_hist"] = {str(c): int((tier == c).sum()) for c in (1, 2, 3, 4)}
    return out


def paired_cluster_boot(labels, groups, probs_a, probs_c, stat, B, seed):
    """CI of stat(arm) - stat(control) under a cluster bootstrap over `groups`.

    Both arms are recomputed on the SAME resampled rows every draw, which is what makes
    the interval paired: the shared sampling noise cancels instead of being added twice.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_g[uniq[p]] for p in pick])
        va = stat(labels[idx], probs_a[idx])
        vc = stat(labels[idx], probs_c[idx])
        if va is None or vc is None:
            continue
        deltas.append(va - vc)
    if not deltas:
        return None
    d = np.asarray(deltas)
    return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "boot_mean": float(d.mean()), "n_draws": int(len(d))}


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def discover_runs():
    """(arm, seed, ckpt) for every FULL arm run whose weights exist."""
    out = []
    for arm in ARMS:
        for seed in (0, 1, 2):
            ck = arm.weights_dir(seed) / "model_best.pt"
            rec = arm.record_dir(seed) / "metrics.json"
            if ck.exists() and rec.exists():
                m = json.loads(rec.read_text())
                if m.get("bounded_run"):
                    continue
                out.append((arm, seed, ck))
    return out


def cmd_score(a):
    import torch
    from scipy.special import expit
    from torch.utils.data import DataLoader

    from classifier.data_v11 import load_locations_v11
    from classifier.train_v2 import detect_device
    from classifier.train_v4 import _RenderSet
    from classifier.train_v8 import score_renders_k

    device = detect_device(a.device)
    locs = load_locations_v11(verify_paths=False)
    ev = sorted([l for l in locs if l.split == "eval"], key=lambda l: l.location_id)
    renders = [l.canonical() for l in ev]
    print(f"eval locations {len(ev)}  (primary "
          f"{sum(1 for l in ev if l.source not in SELECTION_SOURCES)})")

    scores_path = paths.durable(SCORES_REL, mkparents=True)
    rows = {}
    if scores_path.exists() and not a.fresh:
        for line in scores_path.open(encoding="utf-8"):
            r = json.loads(line)
            rows[int(r["loc_id"])] = r
    for l in ev:
        r = rows.setdefault(l.location_id, {})
        r.update({"loc_id": l.location_id, "label": l.label, "source": l.source,
                  "fractal_type": l.fractal_type,
                  "partition": P.partition_of(l.fractal_type),
                  "eval_role": l.eval_role, "split_group": l.split_group,
                  "population": ("selection" if l.source in SELECTION_SOURCES
                                 else "primary")})

    thr_path = paths.durable("data/backbone_search/throughput.json", mkparents=True)
    thr = json.loads(thr_path.read_text()) if thr_path.exists() else {}

    for arm, seed, ck in discover_runs():
        key = f"{arm.name}_s{seed}"
        if any(f"{key}_p3" in r for r in rows.values()) and not a.fresh:
            print(f"  {key:28s} already scored — skip")
            continue
        t0 = time.time()
        model, tf, K, cfg = load_model(ck, device)
        model.eval()
        logits = score_renders_k(model, renders, tf, device, K - 1,
                                 batch_size=64, num_workers=4)
        probs = expit(logits)
        for i, l in enumerate(ev):
            r = rows[l.location_id]
            for k, t in enumerate((2, 3, 4)):
                r[f"{key}_p{t}"] = float(probs[i, k])
            r[f"{key}_score"] = float(probs[i].sum())
        # END-TO-END throughput: decode + deploy transform + forward, the ledger-rescore
        # cost. Timed on its own pass so the scoring pass above cannot flatter it.
        sub = renders[:THROUGHPUT_N]
        loader = DataLoader(_RenderSet(sub, tf), batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=(device == "cuda"))
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            for x, _idx in loader:
                model(x.to(device, non_blocking=True))
        if device == "cuda":
            torch.cuda.synchronize()
        e2e = (time.time() - t1) * 1000 / len(sub)
        thr[key] = {"arm": arm.name, "seed": seed,
                    "e2e_s_per_1k": round(e2e, 2), "n_tiles": len(sub),
                    "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3)}
        print(f"  {key:28s} scored {len(renders)} tiles in {time.time()-t0:.0f}s   "
              f"end-to-end {e2e:.2f} s/1k")
        del model, loader
        if device == "cuda":
            torch.cuda.empty_cache()

    with scores_path.open("w", encoding="utf-8") as f:
        for lid in sorted(rows):
            f.write(json.dumps(rows[lid]) + "\n")
    thr_path.write_text(json.dumps(thr, indent=2))
    print(f"wrote {scores_path}  ({len(rows)} rows)  and {thr_path}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def arm_keys(rows):
    ks = sorted({k[:-3] for r in rows for k in r if k.endswith("_p3")})
    return ks


def cmd_report(a):
    prereg = json.loads(paths.durable(PREREG_REL).read_text())
    min_pos = prereg["declared_slices"]["min_positives_for_a_slice_verdict"]
    boot = prereg["honesty_rule"]["bootstrap"]
    B = a.boot or int(boot["B"])
    rows = [json.loads(l) for l in
            paths.durable(SCORES_REL).open(encoding="utf-8") if l.strip()]
    thr = json.loads(paths.durable("data/backbone_search/throughput.json").read_text())
    keys = arm_keys(rows)
    ctrl_keys = [k for k in keys if k.startswith(CONTROL.name + "_s")]
    if not ctrl_keys:
        raise SystemExit("no control arm scored — every delta is measured against it")
    print(f"arms scored: {keys}")

    prim = [r for r in rows if r["population"] == "primary"]
    sel = [r for r in rows if r["population"] == "selection"]
    assert len(prim) == prereg["eval_populations"]["PRIMARY (unseen)"]["n"], \
        "PRIMARY population moved from the pre-registration"

    def probs_of(rs, key):
        return np.array([[r[f"{key}_p2"], r[f"{key}_p3"], r[f"{key}_p4"]] for r in rs])

    def slices(rs):
        yield "pooled", rs
        by = defaultdict(list)
        for r in rs:
            by[r["partition"]].append(r)
        for p in sorted(by):
            yield p, by[p]

    labels_all = {id(rs): np.array([r["label"] for r in rs]) for rs in ()}  # noqa: F841
    results = {"prereg": PREREG_REL, "arms": {}, "control": CONTROL.name,
               "bootstrap": {**boot, "B": B}, "throughput": thr}

    # per-arm metrics on both populations + every slice
    for key in keys:
        blk = {"populations": {}}
        for pop_name, rs in (("primary", prim), ("selection", sel)):
            pb = {}
            for sname, srs in slices(rs):
                labels = np.array([r["label"] for r in srs])
                pb[sname] = metrics_block(labels, probs_of(srs, key))
            blk["populations"][pop_name] = pb
        blk["throughput"] = thr.get(key, {})
        rec = next((arm.record_dir(int(key.rsplit("_s", 1)[1])) / "metrics.json"
                    for arm in ARMS if key.startswith(arm.name + "_s")), None)
        if rec and rec.exists():
            m = json.loads(rec.read_text())
            blk["train"] = {k: m.get(k) for k in
                            ("params_m", "best_epoch", "val_best_not_bad_ap",
                             "train_wall_s", "peak_train_vram_mb", "n_epochs_run")}
        results["arms"][key] = blk

    # paired deltas vs the control, seed-matched where both exist
    deltas = {}
    for key in keys:
        if key in ctrl_keys:
            continue
        seed = key.rsplit("_s", 1)[1]
        ck = f"{CONTROL.name}_s{seed}" if f"{CONTROL.name}_s{seed}" in keys else ctrl_keys[0]
        d = {"vs": ck, "slices": {}}
        for sname, srs in slices(prim):
            labels = np.array([r["label"] for r in srs])
            groups = np.array([r["split_group"] if r["split_group"] is not None else -r["loc_id"]
                               for r in srs])
            pa, pc = probs_of(srs, key), probs_of(srs, ck)
            entry = {"n": len(srs)}
            for k, t in zip((0, 1, 2), (2, 3, 4)):
                y = (labels >= t).astype(int)
                npos = int(y.sum())
                entry[f"n_ge{t}"] = npos
                if npos < min_pos or npos == len(y):
                    entry[f"auc_ge{t}"] = {"verdict": "UNDERPOWERED",
                                           "why": f"{npos} positives < {min_pos}"}
                    continue
                aa, ac = fast_auc(y, pa[:, k]), fast_auc(y, pc[:, k])
                ci = paired_cluster_boot(
                    labels, groups, pa, pc,
                    (lambda lb, pr, kk=k, tt=t: fast_auc((lb >= tt).astype(int), pr[:, kk])),
                    B, int(boot["seed"]))
                _, _, z, p = delong_paired(y, pa[:, k], pc[:, k])
                entry[f"auc_ge{t}"] = {
                    "arm": aa, "control": ac, "delta": aa - ac, "ci95": ci,
                    "delong_p": p, "delong_z": z,
                    "verdict": ("TIE" if ci is None or (ci["lo"] <= 0 <= ci["hi"])
                                else ("ARM" if ci["lo"] > 0 else "CONTROL"))}
            for stat_name, fn in (("exact_agree",
                                   lambda lb, pr: float((decode_tier(pr) == lb).mean())),
                                  ("adj_agree",
                                   lambda lb, pr: float((np.abs(decode_tier(pr) - lb) <= 1).mean()))):
                va, vc = fn(labels, pa), fn(labels, pc)
                ci = paired_cluster_boot(labels, groups, pa, pc, fn, B, int(boot["seed"]))
                entry[stat_name] = {"arm": va, "control": vc, "delta": va - vc, "ci95": ci,
                                    "verdict": ("TIE" if ci is None or ci["lo"] <= 0 <= ci["hi"]
                                                else ("ARM" if ci["lo"] > 0 else "CONTROL"))}
            d["slices"][sname] = entry
        deltas[key] = d
        pooled = d["slices"]["pooled"]["auc_ge3"]
        print(f"  {key:28s} pooled AUC>=3 {pooled['arm']:.4f} vs {pooled['control']:.4f}  "
              f"delta {pooled['delta']:+.4f}  CI [{pooled['ci95']['lo']:+.4f}, "
              f"{pooled['ci95']['hi']:+.4f}]  {pooled['verdict']}")
    results["deltas_vs_control"] = deltas

    # seed bands (round 2)
    bands = defaultdict(list)
    for key in keys:
        arm = key.rsplit("_s", 1)[0]
        bands[arm].append(results["arms"][key]["populations"]["primary"]["pooled"]["auc_ge3"])
    results["seed_bands_primary_auc_ge3"] = {
        k: {"n_seeds": len(v), "min": min(v), "mean": float(np.mean(v)), "max": max(v)}
        for k, v in bands.items()}

    out = paths.durable(RESULTS_REL, mkparents=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    write_markdown(results, prereg)
    if not a.no_figure:
        write_figure(results)
    print(f"wrote {out}")


def _fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_markdown(res, prereg):
    keys = sorted(res["arms"], key=lambda k: -(res["arms"][k]["populations"]["primary"]
                                               ["pooled"]["auc_ge3"] or 0))
    L = ["# Backbone comparison — results", "",
         f"Control: **{res['control']}**. Population: PRIMARY = "
         f"{prereg['eval_populations']['PRIMARY (unseen)']['n']} eval locations that touch "
         f"neither training nor the checkpoint pick "
         f"({prereg['eval_populations']['PRIMARY (unseen)']['n_ge3']} at label>=3). "
         f"Deltas are paired cluster-bootstrap (B={res['bootstrap']['B']}, over "
         f"{res['bootstrap']['unit']}); a CI covering 0 is a TIE.", "",
         "| arm | params M | pretrain | ckpt | train h | VRAM MB | score s/1k | "
         "AUC>=3 | delta vs control | AUC>=4 | AUC>=2 | exact | adj |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    from backbone_search.arms import ARMS_BY_NAME
    for k in keys:
        b = res["arms"][k]
        p = b["populations"]["primary"]["pooled"]
        tr = b.get("train", {})
        th = b.get("throughput", {})
        arm_name = k.rsplit("_s", 1)[0]
        spec = ARMS_BY_NAME.get(arm_name)
        d = res["deltas_vs_control"].get(k, {}).get("slices", {}).get("pooled", {}).get("auc_ge3")
        dstr = "— (control)" if d is None else (
            f"{d['delta']:+.4f} [{d['ci95']['lo']:+.4f}, {d['ci95']['hi']:+.4f}] {d['verdict']}"
            if isinstance(d.get("ci95"), dict) else "n/a")
        gc = " ⧗" if (spec and spec.grad_checkpointing) else ""
        L.append(f"| {k} | {tr.get('params_m','?')} | {spec.pretrain if spec else '?'} | "
                 f"e{tr.get('best_epoch','?')} | "
                 f"{(tr.get('train_wall_s') or 0)/3600:.2f}{gc} | "
                 f"{tr.get('peak_train_vram_mb','?')} | {th.get('e2e_s_per_1k','?')} | "
                 f"{_fmt(p['auc_ge3'])} | {dstr} | {_fmt(p['auc_ge4'])} | "
                 f"{_fmt(p['auc_ge2'])} | {_fmt(p['exact_agree'],3)} | "
                 f"{_fmt(p['adj_agree'],3)} |")
    L += ["", "⧗ = gradient checkpointing (memory-time trade, identical gradients): the "
          "train-h column is not a clean architecture cost for that arm.", "",
          "## Per-partition delta vs control — DESCRIPTIVE, unadjusted over 9 slices", ""]
    parts = sorted({s for k in res["deltas_vs_control"]
                    for s in res["deltas_vs_control"][k]["slices"] if s != "pooled"})
    L += ["| arm | " + " | ".join(parts) + " |", "|---|" + "---|" * len(parts)]
    for k in keys:
        if k not in res["deltas_vs_control"]:
            continue
        cells = []
        for p_ in parts:
            e = res["deltas_vs_control"][k]["slices"].get(p_, {}).get("auc_ge3", {})
            if e.get("verdict") == "UNDERPOWERED":
                cells.append(f"— ({e['why'].split()[0]} pos)")
            elif "delta" in e:
                mark = "*" if e["verdict"] != "TIE" else ""
                cells.append(f"{e['delta']:+.3f}{mark}")
            else:
                cells.append("n/a")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    bands = res.get("seed_bands_primary_auc_ge3", {})
    if any(v["n_seeds"] > 1 for v in bands.values()):
        L += ["", "## Round-2 seed bands — pooled PRIMARY AUC>=3", "",
              "| arm | seeds | min | mean | max |", "|---|---|---|---|---|"]
        for k, v in sorted(bands.items(), key=lambda kv: -kv[1]["mean"]):
            L.append(f"| {k} | {v['n_seeds']} | {v['min']:.4f} | {v['mean']:.4f} | "
                     f"{v['max']:.4f} |")
    paths.durable(RESULTS_MD_REL, mkparents=True).write_text("\n".join(L) + "\n")


def write_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = defaultdict(list)
    for k, b in res["arms"].items():
        arm = k.rsplit("_s", 1)[0]
        bands[arm].append((b["populations"]["primary"]["pooled"]["auc_ge3"],
                           b.get("throughput", {}).get("e2e_s_per_1k")))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for arm, pts in bands.items():
        ys = [p[0] for p in pts if p[0] is not None]
        xs = [p[1] for p in pts if p[1] is not None]
        if not ys or not xs:
            continue
        x = float(np.mean(xs))
        is_ctrl = arm == res["control"]
        ax.scatter([x], [float(np.mean(ys))], s=170 if is_ctrl else 90,
                   marker="*" if is_ctrl else "o", zorder=3,
                   edgecolor="black", linewidth=0.8)
        if len(ys) > 1:                       # round-2 band whisker
            ax.plot([x, x], [min(ys), max(ys)], lw=2, alpha=0.7, zorder=2)
        ax.annotate(f"{arm}{' (control)' if is_ctrl else ''}", (x, float(np.mean(ys))),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel("end-to-end score time, seconds per 1,000 canonical tiles (lower = better)")
    ax.set_ylabel("pooled PRIMARY AUC(label>=3)")
    ax.set_title("Backbone comparison — quality vs deploy throughput\n"
                 "star = control; whisker = 3-seed band (round 2)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = paths.durable(FIGURE_REL, mkparents=True)
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--device", default="auto")
    s.add_argument("--fresh", action="store_true", help="re-score arms already in the file")
    s.set_defaults(fn=cmd_score)
    r = sub.add_parser("report")
    r.add_argument("--boot", type=int, default=None, help="override B (prereg says 5000)")
    r.add_argument("--no-figure", action="store_true")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
