#!/usr/bin/env python
r"""v11 certification battery — bars LOADED from the pre-registration, never restated.

`data/v11/prereg_v11.json` was written and committed before this file was run
(`tools/v11/prereg.py`, commit 4894ff8). Every threshold, margin and power-derived
separation bar comes out of that file; nothing here invents a number. That is the mechanical
form of "pre-registered": a bar in the eval script can be edited after seeing the results, a
bar in a committed artifact the script loads cannot.

THE BASELINE IS v10 RE-SCORED on the tiles v11 reads. v10 is the deployed head, and pairing
both arms on identical inputs is what leaves the MODEL as the only difference. Those tiles
are the v11 canonical view (`tools/v11/build_eval_canon.py`), which is a different RENDER
path from v10's own cache — so a DIAGNOSTIC arm scores v10 on both and reports the delta.
The two paths were measured at functional parity (0/30 decision flips,
`tools/v11/parity_crop_mode.py`); if the diagnostic says otherwise, the battery is reading
the renderer and every verdict in it is void.

THE ARMS, and what each can and cannot see:
  PRIMARY   census-144, AUC(>=3), non-inferiority vs v10. The frozen comparability
            instrument, eval-side in every build since v7.
  FLOOR     loose0_v3-526, same construction. Eval-side in both builds, so symmetric.
  UNIFORM   uniform-90, AUC(>=2). GATING now — v10 measured 0.8282 on it, which is the
            number v10's own prereg said would make this arm gating.
  Q4-UNIFORM q4-uniform-290, AUC(>=2). FIRST use; registered 2026-08-03, i.e. after v10's
            build, so it is out-of-sample for BOTH heads. Non-gating, like uniform-90 was.
  MOTIVATING correction-87 at the 3|4 boundary — cutpoint AND ordering together, because a
            cutpoint fix bought with ordering damage has to be visible.
  CLASS-4   census fours, descriptive, carried forward from v10's battery.
  PALETTE   census-144 under the 8 held-out palettes, Spearman, descriptive.
  PARTITION julia:mandelbrot and phoenix holdout calibration — FIRST reads, no bar, and
            nothing derived from them is adopted.

Freezes `data/v11/eval_scores_v11.jsonl` (durable, `tools/scoring/eval_slice`'s conventions)
with per-location v10_/v11_ columns, so a later cut can be re-cut without re-scoring.

  uv run python tools/v11/eval_v11.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for sub in ("tools", "tools/v7", "tools/v8", "tools/corpus", "tools/scoring"):
    sys.path.insert(0, str(ROOT / sub))

from scipy.stats import spearmanr  # noqa: E402

import partitions as P  # noqa: E402
import paths  # noqa: E402
from eval_delong import boot_ci, delong_paired  # reuse — do NOT reimplement DeLong  # noqa: E402
from eval_v8 import load_model, q_auc  # reuse v8's loader verbatim  # noqa: E402

from classifier.data_v11 import load_locations_v11  # noqa: E402
from classifier.train_v2 import detect_device  # noqa: E402
from classifier.train_v8 import derive_k, score_renders_k  # noqa: E402

PREREG = ROOT / "data/v11/prereg_v11.json"
V11_CKPT = ROOT / "data/classifier/v11/model_best.pt"
V10_CKPT = ROOT / "data/classifier/v10/model_best.pt"
V10_CACHE = ROOT / "data/v10/cache_manifest.jsonl"
COLORMAPS = ROOT / "data/v11/colormaps.json"
BIN = ROOT / "target/release/fractal-generator.exe"
EVAL_SCORES_OUT = "data/v11/eval_scores_v11.jsonl"
EVAL_RESULTS_OUT = "data/v11/eval_results_v11.json"

CENSUS, FLOOR = "prospect_census", "loose0_v3_floor"
UNIFORM, Q4_UNIFORM = "maneuver_uniform_v1", "q4_uniform_eval"
CORRECTION_BATCHES = ("2026-08-07_label_run_correction_v1",
                      "2026-08-07_steady_state_v2_backfill_v1")
NEUTRAL_PALETTE = "twilight_shifted"


# --------------------------------------------------------------------------- #
# shared readouts
# --------------------------------------------------------------------------- #
def paired_block(name, labels, s_base, s_cand, thr, label_base, label_cand, *, eq=False):
    """AUC for two paired score vectors + paired DeLong + bootstrap CIs.

    `eq=True` cuts `label == thr` instead of `label >= thr` — the 3|4 arm asks whether a row
    is a FOUR, and on a population that is 83% >=3 the `>=4` and `==4` cuts are the same set
    but the intent is not, so it is spelled."""
    y = ((np.asarray(labels) == thr) if eq else (np.asarray(labels) >= thr)).astype(int)
    ab, ac_, z, p = delong_paired(y, s_base, s_cand)
    ci_b, ci_c = boot_ci(y, np.asarray(s_base)), boot_ci(y, np.asarray(s_cand))
    return {"name": name, "n": int(len(labels)), "thr": thr, "cut": "==" if eq else ">=",
            "n_pos": int(y.sum()), "arm_base": label_base, "arm_cand": label_cand,
            "auc_base": round(ab, 4), "auc_base_ci95": [round(ci_b[0], 4), round(ci_b[1], 4)],
            "auc_cand": round(ac_, 4), "auc_cand_ci95": [round(ci_c[0], 4), round(ci_c[1], 4)],
            "delta_cand_minus_base": round(ac_ - ab, 4), "delong_z": round(z, 3),
            "delong_p": round(p, 4)}


def noninferior(block, margin):
    """The pre-registered non-inferiority rule, applied to a paired block."""
    return (block["auc_cand"] >= block["auc_base"] - margin) and not (
        block["delta_cand_minus_base"] < 0 and block["delong_p"] < 0.05)


def separates(auc, ci_lo, bar):
    return bool(auc >= bar and ci_lo > 0.50)


def cutpoint_read(y_true, p4, t):
    """Precision / recall / F1 / predicted-rate of the `P(label>=4) >= t` decode."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(p4) >= t
    tp = int((pred & y).sum())
    prec = tp / int(pred.sum()) if pred.sum() else None
    rec = tp / int(y.sum()) if y.sum() else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"t": t, "n_pred_pos": int(pred.sum()), "tp": tp,
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec is None else round(rec, 4),
            "f1": None if f1 is None else round(f1, 4),
            "predicted_rate": round(float(pred.mean()), 4),
            "mean_predicted_prob": round(float(np.mean(p4)), 4)}


def reliability(y_true, p, bins=10):
    """Decile reliability + ECE + Brier — the calibration read, not a cut."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, obs, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        table.append({"bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})", "n": n,
                      "mean_p": round(conf, 4), "observed": round(obs, 4)})
        ece += n / len(y) * abs(conf - obs)
    return {"bins": table, "ece": round(ece, 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "base_rate": round(float(y.mean()), 4), "mean_p": round(float(p.mean()), 4)}


def fbeta_argmax(y_true, p, beta, grid=None):
    """F_beta-argmax over a P-grid with the plateau width, per protocol §4's method.

    Tie-break toward higher `t`, which is what puts the argmax at the plateau's UPPER edge
    by construction — so the plateau is the only honest read on how knife-edged the pick is.
    Reported, never adopted: no threshold file is written by this module."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(p, dtype=float)
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2) if grid is None else grid
    best, rows = None, []
    for t in grid:
        pred = p >= t
        tp = int((pred & y).sum())
        if not pred.sum() or not y.sum():
            rows.append((float(t), None))
            continue
        prec, rec = tp / int(pred.sum()), tp / int(y.sum())
        f = (0.0 if (prec + rec) == 0 else
             (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))
        rows.append((float(t), f))
        if best is None or f >= best[1]:            # >= ties toward higher t
            best = (float(t), f)
    if best is None:
        return None
    plateau = [t for t, f in rows if f is not None and abs(f - best[1]) < 1e-9]
    pred = p >= best[0]
    tp = int((pred & y).sum())
    return {"beta": beta, "t_argmax": round(best[0], 2), "f_at_argmax": round(best[1], 4),
            "plateau_lo": round(min(plateau), 2), "plateau_hi": round(max(plateau), 2),
            "plateau_width": round(max(plateau) - min(plateau), 2),
            "precision_at_t": round(tp / max(int(pred.sum()), 1), 4),
            "recall_at_t": round(tp / max(int(y.sum()), 1), 4),
            "n_pred_pos": int(pred.sum())}


# --------------------------------------------------------------------------- #
# the tile-path diagnostic
# --------------------------------------------------------------------------- #
def v10_cache_canonical(locs):
    """v10's OWN canonical cache tile for each of `locs`, matched by coordinate key.

    The join is on coordinates, not on loc_id: v11 renumbered loc_id densely over its own
    row order and `build_record.json` says so explicitly ('no cross-version meaning'), so an
    id join here would silently pair 144 unrelated locations."""
    v11man = {r["loc_id"]: r for r in
              (json.loads(x) for x in
               paths.bulk("data/v11/manifest.jsonl").open(encoding="utf-8") if x.strip())}

    def key(r):
        return (r["fractal_type"], r["cx"], r["cy"], r["fw"],
                str(r.get("c_re")), str(r.get("c_im")))

    want_keys = {key(v11man[l.location_id]): l.location_id for l in locs}
    out = {}
    with V10_CACHE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["palette"] != NEUTRAL_PALETTE or r["aa_level"] != "antialiased"
                    or r["scale"] != 1.0 or r["shift_id"] != "center"):
                continue
            out.setdefault(r["location_id"], r)
    # v10's cache manifest carries no coordinates, so go through v10's manifest.
    v10man = {}
    for line in (ROOT / "data/v10/manifest.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            v10man[r["loc_id"]] = r
    matched = {}
    for lid, row in out.items():
        k = key(v10man[lid])
        if k in want_keys:
            matched[want_keys[k]] = paths.bulk(row["path"])
    return matched


# --------------------------------------------------------------------------- #
# palette invariance
# --------------------------------------------------------------------------- #
def palette_invariance(census, held_out, model, tf, K, device):
    """Render census-144 under twilight + each held-out palette at the canonical geometry
    and the LIVE cap, score with v11, report Spearman(twilight, palette)."""
    if not BIN.exists():
        return {"error": f"release binary missing: {BIN}"}
    import active_ckpt as ac
    import location as loc_mod
    from classifier.data_v4 import Render

    man = {r["loc_id"]: r for r in
           (json.loads(x) for x in
            paths.bulk("data/v11/manifest.jsonl").open(encoding="utf-8") if x.strip())}
    rows = [man[l.location_id] for l in census]
    palettes = [NEUTRAL_PALETTE] + list(held_out)
    with tempfile.TemporaryDirectory(prefix="v11_pinv_") as td:
        troot = Path(td)
        plan, index = [], {}
        for r in rows:
            lid, ftype = r["loc_id"], r.get("fractal_type", "mandelbrot")
            extra = {k: r[k] for k in loc_mod.family_param_keys(ftype) if r.get(k) is not None}
            mit = int(ac.auto_maxiter(float(r["fw"])))
            for pal in palettes:
                out = (troot / f"{lid}__{pal}.jpg").as_posix()
                row = {"cx": r["cx"], "cy": r["cy"], "fw": r["fw"], "palette": pal,
                       "ss": 2, "filter": "lanczos3", "maxiter": mit, "out": out,
                       "fractal_type": ftype}
                if r.get("c_re") is not None:
                    row["c_re"], row["c_im"] = r["c_re"], r["c_im"]
                row.update(extra)
                plan.append(row)
                index[(lid, pal)] = out
        pf = troot / "pinv_plan.jsonl"
        pf.write_text("\n".join(json.dumps(x) for x in plan) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [str(BIN), "v4-render-batch", "--plan", str(pf), "--colormaps", str(COLORMAPS),
             "--log-every", "100000"], cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": f"render failed: {proc.stderr[-1500:]}"}

        def score_palette(pal):
            renders = [Render(path=Path(index[(r["loc_id"], pal)]), palette=pal,
                              palette_family=pal, scale=1.0, shift_id="center",
                              aa_level="antialiased") for r in rows]
            _, s = derive_k(score_renders_k(model, renders, tf, device, K - 1, num_workers=0))
            return s

        s_twi = score_palette(NEUTRAL_PALETTE)
        per_pal, rhos, held_scores = {}, [], []
        for pal in held_out:
            s = score_palette(pal)
            held_scores.append(s)
            rho = spearmanr(s_twi, s).correlation
            rho = float(rho) if np.isfinite(rho) else None
            per_pal[pal] = None if rho is None else round(rho, 4)
            if rho is not None:
                rhos.append(rho)
        pooled = spearmanr(np.tile(s_twi, len(held_out)),
                           np.concatenate(held_scores)).correlation
    return {"n_census": len(census), "held_out_palettes": held_out,
            "instrument": NEUTRAL_PALETTE,
            "rendered_at_cap": "auto_maxiter(fw) — the live policy, per row",
            "spearman_twilight_vs_heldout": per_pal,
            "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
            "range": [round(min(rhos), 4), round(max(rhos), 4)] if rhos else None,
            "pooled_spearman": round(float(pooled), 4) if np.isfinite(pooled) else None}


# --------------------------------------------------------------------------- #
def main() -> int:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    arms = prereg["arms"]
    device = detect_device("auto")
    for p in (V11_CKPT, V10_CKPT):
        if not p.exists():
            sys.exit(f"checkpoint missing: {p}")
    m11, tf11, K11, _ = load_model(V11_CKPT, device)
    m10, tf10, K10, _ = load_model(V10_CKPT, device)
    print(f"v11 K={K11} | v10 K={K10} | device {device}")
    print_bar(prereg)

    # The scored population is the eval slice PLUS the correction sitting's train-side rows
    # — `build_eval_canon` canonicalizes both, because the pre-registration promises a
    # CONTAMINATED companion read over all 500 correction rows. Train-side rows carry
    # `split == "train"` and every number computed on them is stamped; they are excluded
    # from every verdict by the masks below, which key on `source` and `eval_role`.
    allocs = load_locations_v11(verify_paths=False)
    locs = [l for l in allocs
            if l.split == "eval"
            or any(b in l.source for b in CORRECTION_BATCHES)]
    labels = np.array([l.label for l in locs])
    src = np.array([l.source for l in locs])
    role = np.array([l.eval_role for l in locs])
    split = np.array([l.split for l in locs])
    canon = [l.canonical() for l in locs]
    print(f"scored {len(locs)}: eval {(split=='eval').sum()} "
          f"(instrument {(role=='instrument').sum()} + holdout {(role=='holdout').sum()}) "
          f"+ {(split=='train').sum()} train-side correction rows (companion only)")

    p11, s11 = derive_k(score_renders_k(m11, canon, tf11, device, K11 - 1, num_workers=0))
    p10, s10 = derive_k(score_renders_k(m10, canon, tf10, device, K10 - 1, num_workers=0))

    results = {"score_key": "sum_sigma (rank score)",
               "prereg_source": "data/v11/prereg_v11.json", "prereg": prereg,
               "arms_note": {"baseline": "v10 re-scored on the v11 canonical tiles — THE "
                                         "DEPLOYED SYSTEM",
                             "candidate": "v11 on the same tiles"}}

    # ---------------- DIAGNOSTIC: the tile path ----------------
    results["diagnostic_tile_path"] = tile_path_diagnostic(
        [l for l in locs if l.source == CENSUS], m10, tf10, K10, device,
        labels[src == CENSUS], s10[src == CENSUS], arms)

    # ---------------- PRIMARY / FLOOR ----------------
    for armkey, source, thr, tag in (("primary_census144", CENSUS, 3, "census-144 q3"),
                                     ("floor_loose0_v3", FLOOR, 3, "mandelbrot-floor q3")):
        a = arms[armkey]
        m = src == source
        b = paired_block(tag, labels[m], s10[m], s11[m], thr, "v10", "v11")
        b["bar"] = a["bar"]
        b["verdict"] = ("NON-INFERIOR" if noninferior(b, a["noninf_margin"])
                        else "INFERIOR / inconclusive")
        b["gating"] = a["gating"]
        results[armkey] = b

    # ---------------- UNIFORM-90 (separation AND non-inferiority, both gating) ----------
    a = arms["uniform90"]
    m = src == UNIFORM
    ub = paired_block("maneuver-uniform-90 q2", labels[m], s10[m], s11[m], 2, "v10", "v11")
    ub["bar"] = a["bar"]
    ub["separation_bar"] = a["separation_bar"]
    ub["separates"] = separates(ub["auc_cand"], ub["auc_cand_ci95"][0], a["separation_bar"])
    ub["v10_separates"] = separates(ub["auc_base"], ub["auc_base_ci95"][0],
                                    a["separation_bar"])
    ub["noninferior"] = noninferior(ub, a["noninf_margin"])
    ub["verdict"] = ("SEPARATES + NON-INFERIOR" if (ub["separates"] and ub["noninferior"])
                     else "FAILS: " + ", ".join(
                         ([] if ub["separates"] else ["does not separate"])
                         + ([] if ub["noninferior"] else ["inferior to v10"])))
    ub["gating"] = a["gating"]
    results["uniform90"] = ub

    # ---------------- Q4-UNIFORM-290 (first read, non-gating) ----------------
    a = arms["q4_uniform290"]
    m = src == Q4_UNIFORM
    qb = paired_block("q4-uniform-290 q2", labels[m], s10[m], s11[m], 2, "v10", "v11")
    qb["bar"] = a["bar"]
    qb["separation_bar"] = a["separation_bar"]
    qb["separates"] = separates(qb["auc_cand"], qb["auc_cand_ci95"][0], a["separation_bar"])
    qb["v10_separates"] = separates(qb["auc_base"], qb["auc_base_ci95"][0],
                                    a["separation_bar"])
    qb["verdict"] = ("SEPARATES" if qb["separates"]
                     else "UNDERPOWERED / does not separate")
    qb["gating"] = a["gating"]
    qb["note"] = a["gating_note"]
    results["q4_uniform290"] = qb

    # ---------------- MOTIVATING: the 3|4 boundary on the correction sitting ----------
    results["motivating_class4_correction87"] = class4_arm(
        locs, labels, p10, p11, s10, s11, K10, K11, arms)

    # ---------------- CLASS-4 on the census (descriptive) ----------------
    cm = src == CENSUS
    y4 = (labels[cm] == 4).astype(int)
    results["class4_census_descriptive"] = {
        "n_census": int(cm.sum()), "n_class4": int(y4.sum()),
        "auc_v11_q4_vs_rest": q_auc(y4, s11[cm]),
        "auc_v11_q4_by_pnext": (None if K11 < 4 else q_auc(y4, p11[cm][:, 2])),
        "auc_v10_q4_vs_rest": q_auc(y4, s10[cm]),
        "auc_v10_q4_by_pnext": (None if K10 < 4 else q_auc(y4, p10[cm][:, 2])),
        "class4_by_family": dict(Counter(np.array([l.fractal_type for l in locs])[cm]
                                         [y4.astype(bool)].tolist())),
        "note": "DESCRIPTIVE, no bar."}
    for k, v in list(results["class4_census_descriptive"].items()):
        if isinstance(v, float):
            results["class4_census_descriptive"][k] = round(v, 4)

    # ---------------- freeze the durable per-location slice ----------------
    # EVAL ROWS ONLY. `eval_slice` is the frozen instrument a later keeper cut or t_good
    # derivation re-cuts from without re-scoring, and a train-side row in it is one nobody
    # downstream would know to exclude.
    with paths.durable(EVAL_SCORES_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for i, l in enumerate(locs):
            if l.split != "eval":
                continue
            row = {"location_id": l.location_id, "label": l.label, "source": l.source,
                   "eval_role": l.eval_role, "group_id": l.group_id,
                   "split_group": l.split_group, "fractal_type": l.fractal_type,
                   "partition": P.partition_of_row(
                       {"fractal_type": l.fractal_type}, l.fractal_type),
                   "v11_score": float(s11[i]), "v10_score": float(s10[i])}
            for k in range(K11 - 1):
                row[f"v11_p_ge{k+2}"] = float(p11[i, k])
            for k in range(K10 - 1):
                row[f"v10_p_ge{k+2}"] = float(p10[i, k])
            f.write(json.dumps(row) + "\n")

    # ---------------- PER-PARTITION calibration, first reads ----------------
    results["per_partition_calibration_first_reads"] = partition_calibration(
        locs, labels, p11, p10, K11, arms)

    # ---------------- PALETTE INVARIANCE ----------------
    a = arms["palette_invariance"]
    pinv = palette_invariance([l for l in locs if l.source == CENSUS],
                              a["held_out_palettes"], m11, tf11, K11, device)
    if "error" not in pinv and pinv.get("mean_spearman") is not None:
        d = pinv["mean_spearman"] - a["v10_measured_mean"]
        pinv["v10_reference_mean"] = a["v10_measured_mean"]
        pinv["delta_vs_v10"] = round(d, 4)
        pinv["read"] = ("INVESTIGATE — a move this large is not a corpus effect"
                        if abs(d) > a["investigate_delta"] else "consistent with v10")
    results["palette_invariance"] = pinv

    paths.durable(EVAL_RESULTS_OUT, mkparents=True).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print_report(results)
    print(f"\nwrote {EVAL_SCORES_OUT} and {EVAL_RESULTS_OUT} (durable)")
    return 0


def tile_path_diagnostic(census, m10, tf10, K10, device, labels, s10_v11tiles, arms):
    """v10 on its OWN cache tile vs v10 on the v11 canonical tile, census-144.

    The one thing that could make every verdict above a measurement of the renderer instead
    of the head. Protocol §3's instrument check, in the form v11's intervention calls for."""
    a = arms["primary_census144"]
    try:
        matched = v10_cache_canonical(census)
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"could not resolve v10 cache tiles: {e}"}
    have = [l for l in census if l.location_id in matched]
    missing = [l.location_id for l in census if l.location_id not in matched]
    if not have:
        return {"error": "no census location matched a v10 cache canonical tile",
                "n_missing": len(missing)}
    from classifier.data_v4 import Render
    renders = [Render(path=matched[l.location_id], palette=NEUTRAL_PALETTE,
                      palette_family=NEUTRAL_PALETTE, scale=1.0, shift_id="center",
                      aa_level="antialiased") for l in have]
    present = [l.path.exists() for l in renders]
    if not all(present):
        return {"error": f"{present.count(False)} v10 cache tiles are not on disk "
                         f"(the v9/v10 aug_cache tree)", "n_matched": len(have)}
    _, s_own = derive_k(score_renders_k(m10, renders, tf10, device, K10 - 1, num_workers=0))
    idx = {l.location_id: i for i, l in enumerate(census)}
    s_v11 = np.array([s10_v11tiles[idx[l.location_id]] for l in have])
    y = (np.array([l.label for l in have]) >= 3).astype(int)
    auc_own, auc_v11 = q_auc(y, s_own), q_auc(y, s_v11)
    delta = None if (auc_own is None or auc_v11 is None) else auc_v11 - auc_own
    rho = spearmanr(s_own, s_v11).correlation
    return {"what": "v10 scored on its OWN v10 cache canonical tile vs on the v11 "
                    "canonical tile, census-144, AUC(label>=3)",
            "n_matched": len(have), "n_unmatched": len(missing),
            "auc_v10_on_v10_tiles": None if auc_own is None else round(auc_own, 4),
            "auc_v10_on_v11_tiles": None if auc_v11 is None else round(auc_v11, 4),
            "delta": None if delta is None else round(delta, 4),
            "spearman_scores": round(float(rho), 4) if np.isfinite(rho) else None,
            "mean_abs_score_delta": round(float(np.mean(np.abs(s_own - s_v11))), 4),
            "expectation": "|delta| <= 0.02 (prereg instrument_check.diagnostic_tile_path)",
            "verdict": ("BATTERY VOID — the tile path moved the instrument"
                        if (delta is not None and abs(delta) > 0.02)
                        else "tile path is not what the arms are reading")}


def class4_arm(locs, labels, p10, p11, s10, s11, K10, K11, arms):
    """The motivating slice: the 3|4 boundary on the correction sitting.

    Cutpoint AND ordering, on the same rows, in the same block — the whole point of the arm
    is that a cutpoint fix bought with ordering damage must be visible rather than
    reportable separately."""
    a = arms["motivating_class4_correction87"]
    t = a["class4_decode_t"]
    is_corr = np.array([any(b in l.source for b in CORRECTION_BATCHES) for l in locs])
    is_ho = np.array([l.eval_role == "holdout" for l in locs])
    m = is_corr & is_ho
    if K11 < 4 or K10 < 4:
        return {"error": f"a K<4 head cannot decode class 4 (v11 K={K11}, v10 K={K10})"}
    y4 = (labels[m] == 4).astype(int)
    obs = float(y4.mean())
    cut10, cut11 = cutpoint_read(y4, p10[m][:, 2], t), cutpoint_read(y4, p11[m][:, 2], t)
    ordering = paired_block("correction-87 class-4 ordering", labels[m], s10[m], s11[m],
                            4, "v10", "v11", eq=True)
    ordering["by_pnext"] = {"v10": q_auc(y4, p10[m][:, 2]), "v11": q_auc(y4, p11[m][:, 2])}
    tighter = (cut11["precision"] is not None and cut10["precision"] is not None
               and cut11["precision"] > cut10["precision"]
               and abs(cut11["predicted_rate"] - obs) < abs(cut10["predicted_rate"] - obs))
    ord_ok = noninferior(ordering, a["noninf_margin"])
    return {
        "n": int(m.sum()), "n_class4": int(y4.sum()), "observed_class4_rate": round(obs, 4),
        "population": a["instrument"],
        "why_this_population": a["why_this_population"],
        "cutpoint_bar": a["cutpoint_bar"], "ordering_bar": a["ordering_bar"],
        "cutpoint": {"decode_t": t, "v10": cut10, "v11": cut11,
                     "verdict": "TIGHTENED" if tighter else "NOT TIGHTENED"},
        "ordering": ordering,
        "ordering_verdict": "NOT DAMAGED" if ord_ok else "DAMAGED",
        "calibration_p_ge4": {
            "v10": reliability(y4, p10[m][:, 2]), "v11": reliability(y4, p11[m][:, 2])},
        "verdict": ("CUTPOINT TIGHTENED, ORDERING INTACT" if (tighter and ord_ok) else
                    "CUTPOINT TIGHTENED BUT ORDERING DAMAGED" if tighter else
                    "CUTPOINT NOT TIGHTENED"),
        "companion_contaminated": contaminated_companion(locs, labels, p10, p11, t),
    }


def contaminated_companion(locs, labels, p10, p11, t):
    """The same cutpoint read over ALL 500 correction rows, split by whether v11 trained on
    it. Stamped, and never a verdict: v11's number on a train-side row is in-sample and
    v10's is not — all 500 postdate v10's build — so the train-side block is a comparison
    rigged in exactly the direction the claim runs. It is here because it is what the
    pre-registration promised beside the 87-row instrument, and because the size of the
    in-sample/out-of-sample gap is itself worth seeing."""
    is_corr = np.array([any(b in l.source for b in CORRECTION_BATCHES) for l in locs])
    if not is_corr.any():
        return {"note": "no correction rows scored"}
    tr = np.array([l.split == "train" for l in locs]) & is_corr
    out = {"STAMP": "CONTAMINATED — descriptive only, never a verdict",
           "scope": "all correction-sitting rows; the train-side block is IN-SAMPLE for v11 "
                    "and out-of-sample for v10"}
    for tag, m in (("all", is_corr), ("v11_train_side", tr),
                   ("v11_holdout", is_corr & ~tr)):
        y4 = (labels[m] == 4).astype(int)
        out[tag] = {"n": int(m.sum()), "n_class4": int(y4.sum()),
                    "v10": cutpoint_read(y4, p10[m][:, 2], t),
                    "v11": cutpoint_read(y4, p11[m][:, 2], t)}
    return out


def partition_calibration(locs, labels, p11, p10, K11, arms):
    """First calibration reads for the partitions the grouped split gave an eval population.

    HOLDOUT rows only, and the caveat travels with the number: the holdout is a stratified
    random draw over the split groups, biased exactly as training is, so these are statements
    about the ranker over the population training is drawn from and NOT base rates."""
    a = arms["per_partition_calibration_first_reads"]
    part = np.array([P.partition_of_row({"fractal_type": l.fractal_type}, l.fractal_type)
                     for l in locs])
    is_ho = np.array([l.eval_role == "holdout" for l in locs])
    out = {"caveat": a["holdout_caveat"], "explicitly_not_adopted": a["explicitly_not_adopted"],
           "min_pos": a["min_pos"], "partitions": {}}
    for name in ("julia:mandelbrot", "phoenix"):
        m = is_ho & (part == name)
        if not m.any():
            out["partitions"][name] = {"error": "no holdout rows"}
            continue
        y3 = (labels[m] >= 3).astype(int)
        p3_11, p3_10 = p11[m][:, 1], p10[m][:, 1]
        block = {"n": int(m.sum()), "n_pos_ge3": int(y3.sum()),
                 "clears_min_pos": bool(int(y3.sum()) >= a["min_pos"]),
                 "v11": {"reliability": reliability(y3, p3_11),
                         "auc_ge3": q_auc(y3, p3_11),
                         "fbeta": {"F0.5": fbeta_argmax(y3, p3_11, 0.5),
                                   "F2": fbeta_argmax(y3, p3_11, 2.0)}},
                 "v10_CONTAMINATED": {"reliability": reliability(y3, p3_10),
                                      "auc_ge3": q_auc(y3, p3_10),
                                      "note": "v10 trained on much of this population — "
                                              "printed for orientation, NOT a comparison"}}
        if K11 >= 4:
            y4 = (labels[m] == 4).astype(int)
            block["v11"]["class4"] = {"n_class4": int(y4.sum()),
                                      "reliability": reliability(y4, p11[m][:, 2])}
        out["partitions"][name] = block
    return out


# --------------------------------------------------------------------------- #
def print_bar(prereg):
    print("\n" + "=" * 82)
    print("PRE-REGISTERED BARS — loaded from data/v11/prereg_v11.json, not restated here")
    print("=" * 82)
    print(f"  baseline: {prereg['baseline'][:150]}...")
    for name, a in prereg["arms"].items():
        print(f"  {name:<40} n={a.get('n')} gating={a.get('gating')}")
    print("=" * 82)


def _pb(tag, b):
    print(f"\n--- {tag} (n={b['n']}, pos={b['n_pos']} at label{b['cut']}{b['thr']}) ---")
    print(f"  v10 AUC {b['auc_base']} CI{b['auc_base_ci95']}   (baseline, re-scored)")
    print(f"  v11 AUC {b['auc_cand']} CI{b['auc_cand_ci95']}")
    print(f"  paired DeLong delta(v11-v10)={b['delta_cand_minus_base']:+.4f} "
          f"z={b['delong_z']} p={b['delong_p']}")
    if "verdict" in b:
        print(f"  ->  {b['verdict']}")


def print_report(r):
    print("\n" + "=" * 82)
    print("v11 CERTIFICATION — v11 vs v10 re-scored, on identical tiles")
    print("=" * 82)
    d = r["diagnostic_tile_path"]
    print(f"\n--- DIAGNOSTIC tile path --- {d.get('verdict', d.get('error'))}")
    if "delta" in d:
        print(f"  v10 on v10 tiles {d['auc_v10_on_v10_tiles']}  on v11 tiles "
              f"{d['auc_v10_on_v11_tiles']}  delta {d['delta']}  "
              f"spearman {d['spearman_scores']}  (n={d['n_matched']})")
    _pb("PRIMARY census-144", r["primary_census144"])
    _pb("FLOOR loose0_v3-526", r["floor_loose0_v3"])
    _pb("UNIFORM-90 (gating)", r["uniform90"])
    print(f"  separation bar {r['uniform90']['separation_bar']}  "
          f"v11 separates={r['uniform90']['separates']}  "
          f"v10 separates={r['uniform90']['v10_separates']}")
    _pb("Q4-UNIFORM-290 (first read, not gating)", r["q4_uniform290"])
    c = r["motivating_class4_correction87"]
    if "error" in c:
        print(f"\n--- MOTIVATING 3|4: ERROR {c['error']}")
    else:
        print(f"\n--- MOTIVATING: the 3|4 boundary, correction-87 (out-of-sample for BOTH) ---")
        print(f"  n={c['n']}  fours={c['n_class4']}  observed rate {c['observed_class4_rate']}")
        for who in ("v10", "v11"):
            k = c["cutpoint"][who]
            print(f"  {who} decode P>={c['cutpoint']['decode_t']}: predicts {k['n_pred_pos']:3d} "
                  f"({k['predicted_rate']})  precision {k['precision']}  recall {k['recall']}"
                  f"  F1 {k['f1']}  mean P {k['mean_predicted_prob']}")
        print(f"  cutpoint -> {c['cutpoint']['verdict']}")
        o = c["ordering"]
        print(f"  ordering AUC(==4): v10 {o['auc_base']} -> v11 {o['auc_cand']} "
              f"(delta {o['delta_cand_minus_base']:+.4f}, p={o['delong_p']})"
              f"  -> {c['ordering_verdict']}")
        print(f"  ==> {c['verdict']}")
    c4 = r["class4_census_descriptive"]
    print(f"\n--- CLASS-4 census (descriptive) --- n={c4['n_class4']}/{c4['n_census']}")
    print(f"  v11 AUC(q4) {c4['auc_v11_q4_vs_rest']} by P>=4 {c4['auc_v11_q4_by_pnext']}   "
          f"v10 {c4['auc_v10_q4_vs_rest']} by P>=4 {c4['auc_v10_q4_by_pnext']}")
    pc = r["per_partition_calibration_first_reads"]
    print(f"\n--- PER-PARTITION CALIBRATION (holdout, FIRST reads, nothing adopted) ---")
    for name, b in pc["partitions"].items():
        if "error" in b:
            print(f"  {name}: {b['error']}")
            continue
        v = b["v11"]
        print(f"  {name:<18} n={b['n']:4d} pos={b['n_pos_ge3']:3d} "
              f"(>=MIN_POS {b['clears_min_pos']})  AUC {v['auc_ge3']}  "
              f"ECE {v['reliability']['ece']}  base {v['reliability']['base_rate']} "
              f"vs mean P {v['reliability']['mean_p']}")
        for bn, fb in v["fbeta"].items():
            if fb:
                print(f"      {bn}: t*={fb['t_argmax']} plateau [{fb['plateau_lo']},"
                      f"{fb['plateau_hi']}] w={fb['plateau_width']} prec {fb['precision_at_t']}"
                      f" rec {fb['recall_at_t']}")
    pinv = r["palette_invariance"]
    if "error" in pinv:
        print(f"\n--- PALETTE-INVARIANCE: ERROR {pinv['error']}")
    else:
        print(f"\n--- PALETTE-INVARIANCE (census-144, 8 held-out palettes, descriptive) ---")
        print(f"  mean Spearman {pinv['mean_spearman']} range {pinv['range']} pooled "
              f"{pinv['pooled_spearman']}   vs v10 {pinv.get('delta_vs_v10')} "
              f"({pinv.get('read')})")


if __name__ == "__main__":
    raise SystemExit(main())
