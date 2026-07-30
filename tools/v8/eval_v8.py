#!/usr/bin/env python
r"""v8 evaluation battery — re-scores v7 and v8 on the v8 eval slice and applies the
pre-registered acceptance reads. Nothing here trains; ACTIVE_CKPT is not touched.

Reads: data/classifier/v8/model_best.pt (K=4) and data/classifier/v7/model_best.pt (K=3),
the v8 cache (canonical twilight_shifted views), and — for the palette-invariance read —
renders the census-144 under twilight + the held-out palettes on the fly.

Reads (all through the SAME deploy transform, Transform(train=False), 512x288 ss2 tile ->
384x224 stretch + normalize):

  PRIMARY — census-144, q3(=label>=3)-vs-rest AUC. v7 re-scored on the identical slice, paired
            DeLong vs v8. Pre-registered bar stated in the report before any number.
  SECONDARY — mandelbrot FLOOR (loose0_v3, 526 loc), q3-vs-rest AUC, paired DeLong
            (non-regression). v7 TRAINED on these locations, so its score is flattered — the
            comparison is biased AGAINST v8 (conservative for a regression check).
  CLASS-4 — census q4(=label>=4)-vs-rest AUC for v8 (descriptive only; 22 class-4 census loc,
            all julia:multibrot; native-multibrot class-4 behaviour is extrapolation).
  PALETTE-INVARIANCE — census-144 v8 score under twilight_shifted vs each held-out palette;
            Spearman rank correlation. twilight_shifted stays the pinned instrument for every
            number above; this is additive.

Freezes data/v8/eval_scores_v8.jsonl (durable) with per-location v7_/v8_ columns so the
keeper cut can be recut against it without re-scoring.

  uv run python tools/v8/eval_v8.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "v7"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import paths  # noqa: E402
from eval_delong import boot_ci, delong_paired  # reuse — do NOT reimplement DeLong  # noqa: E402
from classifier.data import Transform  # noqa: E402
from classifier.data_v4 import load_locations, NEUTRAL_PALETTE, CANON_SCALE, CANON_SHIFT  # noqa: E402
from classifier.model import build_model, data_config  # noqa: E402
from classifier.train_v2 import detect_device  # noqa: E402
from classifier.train_v8 import score_renders_k, derive_k, CENSUS_SOURCE, FLOOR_SOURCE  # noqa: E402

V8_CACHE = ROOT / "data/v8/cache_manifest.jsonl"
V8_CKPT = ROOT / "data/classifier/v8/model_best.pt"
V7_CKPT = ROOT / "data/classifier/v7/model_best.pt"
BUILD_META = ROOT / "data/v8/build_metadata.json"
COLORMAPS = ROOT / "data/v8/colormaps.json"
BIN = ROOT / "target/release/fractal-generator.exe"
EVAL_SCORES_OUT = "data/v8/eval_scores_v8.jsonl"
EVAL_RESULTS_OUT = "data/v8/eval_results_v8.json"

# Pre-registered reference: v7's census-144 q3-vs-rest AUC (score-based), as reported.
V7_CENSUS_Q3_REFERENCE = 0.705
# Pre-registered non-inferiority margin (see report banner). The n=144 census cannot resolve
# an AUC gap inside the ~0.05 DeLong-indistinguishable band (protocol §3 / deferred_recalibration
# 0.55-0.65 "label more"); a drop within this margin is non-inferior, larger needs "label more".
NONINF_MARGIN = 0.05


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    K = int(cfg.get("num_classes", 3))
    m = build_model(target="ordinal", drop_rate=cfg.get("drop_rate", 0.2),
                    drop_path_rate=cfg.get("drop_path_rate", 0.1), pretrained=False,
                    num_classes=K).to(device)
    m.load_state_dict(ck["state_dict"])
    tf = Transform(cfg["geometry"], cfg["interpolation"], tuple(cfg["mean"]), tuple(cfg["std"]),
                   train=False)
    return m, tf, K, cfg


def q_auc(y_bin, s):
    from sklearn.metrics import roc_auc_score
    y_bin = np.asarray(y_bin)
    if y_bin.min() == y_bin.max():
        return None
    return float(roc_auc_score(y_bin, s))


def paired_block(name, labels, s_v7, s_v8, thr=3):
    """AUC(label>=thr vs rest) for v7 and v8 + paired DeLong (v7 vs v8) + boot CIs."""
    y = (np.asarray(labels) >= thr).astype(int)
    a7, a8, z, p = delong_paired(y, s_v7, s_v8)
    ci7 = boot_ci(y, np.asarray(s_v7)); ci8 = boot_ci(y, np.asarray(s_v8))
    return {"name": name, "n": int(len(labels)), "thr": thr, "n_pos": int(y.sum()),
            "auc_v7": round(a7, 4), "auc_v7_ci95": [round(ci7[0], 4), round(ci7[1], 4)],
            "auc_v8": round(a8, 4), "auc_v8_ci95": [round(ci8[0], 4), round(ci8[1], 4)],
            "delta_v8_minus_v7": round(a8 - a7, 4), "delong_z": round(z, 3),
            "delong_p": round(p, 4)}


def main():
    device = detect_device("auto")
    if not V8_CKPT.exists():
        sys.exit(f"v8 checkpoint missing: {V8_CKPT} (train first)")
    v8, tf8, K8, cfg8 = load_model(V8_CKPT, device)
    v7, tf7, K7, cfg7 = load_model(V7_CKPT, device)
    print(f"v8 K={K8} (cutpoints {K8-1}) | v7 K={K7} (cutpoints {K7-1}) | device {device}")

    locs = load_locations(cache_path=V8_CACHE)
    eval_locs = [l for l in locs if l.split == "eval"]
    census = [l for l in eval_locs if l.source == CENSUS_SOURCE]
    floor = [l for l in eval_locs if l.source == FLOOR_SOURCE]
    print(f"eval {len(eval_locs)}: census {len(census)} (julia:mb) + floor {len(floor)} (mandelbrot)")

    canon = [l.canonical() for l in eval_locs]
    lg8 = score_renders_k(v8, canon, tf8, device, K8 - 1, num_workers=0)
    lg7 = score_renders_k(v7, canon, tf7, device, K7 - 1, num_workers=0)
    p8, s8 = derive_k(lg8)
    p7, s7 = derive_k(lg7)
    labels = np.array([l.label for l in eval_locs])
    src = np.array([l.source for l in eval_locs])
    ft = np.array([l.fractal_type for l in eval_locs])

    results = {"score_key": "sum_sigma (rank score)",
               "v7_census_q3_reference": V7_CENSUS_Q3_REFERENCE,
               "noninf_margin": NONINF_MARGIN}

    # ---------------- PRIMARY: census q3-vs-rest ----------------
    cm = src == CENSUS_SOURCE
    cb = paired_block("census-144 q3", labels[cm], s7[cm], s8[cm], thr=3)
    non_inferior = (cb["auc_v8"] >= V7_CENSUS_Q3_REFERENCE - NONINF_MARGIN) and not (
        cb["delta_v8_minus_v7"] < 0 and cb["delong_p"] < 0.05)
    cb["v8_vs_reference"] = round(cb["auc_v8"] - V7_CENSUS_Q3_REFERENCE, 4)
    cb["non_inferior_verdict"] = "NON-INFERIOR" if non_inferior else "INFERIOR / inconclusive"
    results["census_q3"] = cb

    # ---------------- SECONDARY: mandelbrot floor non-regression ----------------
    fm = src == FLOOR_SOURCE
    fb = paired_block("mandelbrot-floor q3", labels[fm], s7[fm], s8[fm], thr=3)
    fb["verdict"] = ("REGRESSION" if (fb["delta_v8_minus_v7"] < 0 and fb["delong_p"] < 0.05)
                     else "non-inferior")
    fb["note"] = ("v7 TRAINED on these locations -> its AUC here is flattered; the comparison "
                  "is biased against v8 (conservative for a regression check). A v8 shortfall "
                  "here is ambiguous, not damning.")
    results["mandelbrot_floor_q3"] = fb

    # ---------------- CLASS-4 discrimination (descriptive) ----------------
    y4 = (labels[cm] >= 4).astype(int)
    c4 = {"n_census": int(cm.sum()), "n_class4": int(y4.sum()),
          "auc_v8_q4_vs_rest": (None if y4.min() == y4.max() else round(q_auc(y4, s8[cm]), 4)),
          "auc_v8_q4_by_pnext": (None if y4.min() == y4.max() or K8 < 4 else
                                 round(q_auc(y4, p8[cm][:, 2]), 4)),
          "note": ("Descriptive only, NOT gating. 22 class-4 census loc, all julia:multibrot. "
                   "Native-multibrot class-4 is EXTRAPOLATION: mb4 has 5 class-4 training loc, "
                   "mb3 has 9 (build_metadata per_family).")}
    # census class-4 by native family (all julia:multibrot here)
    c4["class4_by_family"] = dict(Counter(ft[cm][y4.astype(bool)].tolist()))
    results["class4_descriptive"] = c4

    # ---------------- freeze eval scores (v7_/v8_ per location) ----------------
    with paths.durable(EVAL_SCORES_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for i, l in enumerate(eval_locs):
            row = {"location_id": l.location_id, "label": l.label, "source": l.source,
                   "group_id": l.group_id, "fractal_type": l.fractal_type,
                   "v8_score": float(s8[i]), "v7_score": float(s7[i])}
            for k in range(K8 - 1):
                row[f"v8_p_ge{k+2}"] = float(p8[i, k])
            for k in range(K7 - 1):
                row[f"v7_p_ge{k+2}"] = float(p7[i, k])
            f.write(json.dumps(row) + "\n")

    # ---------------- PALETTE-INVARIANCE: census under held-out palettes ----------------
    meta = json.loads(BUILD_META.read_text(encoding="utf-8"))
    held_out = meta["aug_recipe"]["palettes"]["held_out"]
    pinv = palette_invariance(census, held_out, v8, tf8, K8, device)
    results["palette_invariance"] = pinv

    paths.durable(EVAL_RESULTS_OUT, mkparents=True).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print_report(results)
    print(f"\nwrote {EVAL_SCORES_OUT} and {EVAL_RESULTS_OUT} (durable)")


def _coord_rows_for(locs):
    """Pull raw coord rows (cx/cy/fw/c_re/c_im/family params) for these Loc objects from the
    manifest, keyed by location_id."""
    import location as loc_mod
    man = {}
    for line in (ROOT / "data/v8/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            man[r["loc_id"]] = r
    rows = []
    for l in locs:
        r = man[l.location_id]
        rows.append(r)
    return rows, loc_mod


def palette_invariance(census, held_out, model, tf, K, device):
    """Render census-144 under twilight_shifted + each held-out palette at the canonical
    geometry, score with the model, and report Spearman(twilight, palette) rank correlation."""
    if not BIN.exists():
        return {"error": f"release binary missing: {BIN}"}
    rows, loc_mod = _coord_rows_for(census)
    palettes = [NEUTRAL_PALETTE] + list(held_out)
    with tempfile.TemporaryDirectory(prefix="v8_pinv_") as td:
        troot = Path(td)
        plan = []
        index = {}  # (loc_id, palette) -> out path
        for r in rows:
            lid = r["loc_id"]
            ftype = r.get("fractal_type", "mandelbrot")
            extra = {k: r[k] for k in loc_mod.family_param_keys(ftype) if r.get(k) is not None}
            for pal in palettes:
                out = (troot / f"{lid}__{pal}.jpg").as_posix()
                row = {"cx": r["cx"], "cy": r["cy"], "fw": r["fw"], "palette": pal,
                       "ss": 2, "filter": "lanczos3", "out": out, "fractal_type": ftype}
                if r.get("c_re") is not None:
                    row["c_re"] = r["c_re"]; row["c_im"] = r["c_im"]
                row.update(extra)
                plan.append(row)
                index[(lid, pal)] = out
        pf = troot / "pinv_plan.jsonl"
        pf.write_text("\n".join(json.dumps(x) for x in plan) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [str(BIN), "v4-render-batch", "--plan", str(pf), "--colormaps", str(COLORMAPS),
             "--log-every", "100000"],
            cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": f"render failed: {proc.stderr[-1500:]}"}

        # score each palette's set in census order
        from classifier.data_v4 import Render
        def score_palette(pal):
            renders = [Render(path=Path(index[(r["loc_id"], pal)]), palette=pal,
                              palette_family=pal, scale=CANON_SCALE, shift_id=CANON_SHIFT,
                              aa_level="antialiased") for r in rows]
            lg = score_renders_k(model, renders, tf, device, K - 1, num_workers=0)
            _, s = derive_k(lg)
            return s

        s_twi = score_palette(NEUTRAL_PALETTE)
        per_pal = {}
        rhos = []
        for pal in held_out:
            s = score_palette(pal)
            rho = spearmanr(s_twi, s).correlation
            rho = float(rho) if np.isfinite(rho) else None
            per_pal[pal] = round(rho, 4) if rho is not None else None
            if rho is not None:
                rhos.append(rho)
        # pooled: concatenate all held-out palette scores against twilight repeated
        s_all_held = np.concatenate([score_palette(p) for p in held_out])
        s_twi_rep = np.tile(s_twi, len(held_out))
        pooled = spearmanr(s_twi_rep, s_all_held).correlation
    return {"n_census": len(census), "held_out_palettes": held_out,
            "instrument": NEUTRAL_PALETTE,
            "spearman_twilight_vs_heldout": per_pal,
            "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
            "pooled_spearman": round(float(pooled), 4) if np.isfinite(pooled) else None,
            "note": ("twilight_shifted remains the pinned instrument for every number above; "
                     "this is additive. High rho => palette-broad training bought presentation "
                     "independence; low rho is a genuine negative result, not a failure to bury.")}


def print_report(r):
    print("\n" + "=" * 80)
    print("v8 EVALUATION")
    print("=" * 80)
    print(f"PRE-REGISTERED BAR (stated before the numbers): census-144 q3-vs-rest AUC on")
    print(f"  twilight_shifted. v8 is NON-INFERIOR iff (a) AUC_v8 >= {V7_CENSUS_Q3_REFERENCE} "
          f"- {NONINF_MARGIN} = {V7_CENSUS_Q3_REFERENCE-NONINF_MARGIN:.3f} AND")
    print(f"  (b) the paired DeLong does NOT show v8 significantly below v7 (p<0.05 & delta<0).")
    print(f"  Rationale: n=144 cannot resolve an AUC gap within ~{NONINF_MARGIN} (protocol S3;")
    print(f"  a drop inside the band means 'label more', not 'v8 failed').")
    cb = r["census_q3"]
    print(f"\n--- PRIMARY census-144 q3 (n={cb['n']}, q3-pos={cb['n_pos']}) ---")
    print(f"  v7 AUC {cb['auc_v7']} CI{cb['auc_v7_ci95']}  (reference {V7_CENSUS_Q3_REFERENCE})")
    print(f"  v8 AUC {cb['auc_v8']} CI{cb['auc_v8_ci95']}")
    print(f"  paired DeLong delta(v8-v7)={cb['delta_v8_minus_v7']:+.4f} z={cb['delong_z']} p={cb['delong_p']}")
    print(f"  v8 vs reference {cb['v8_vs_reference']:+.4f}  ->  {cb['non_inferior_verdict']}")
    fb = r["mandelbrot_floor_q3"]
    print(f"\n--- SECONDARY mandelbrot floor q3 (n={fb['n']}, q3-pos={fb['n_pos']}) ---")
    print(f"  v7 AUC {fb['auc_v7']} / v8 AUC {fb['auc_v8']}  delta={fb['delta_v8_minus_v7']:+.4f} "
          f"p={fb['delong_p']}  -> {fb['verdict']}")
    print(f"  ({fb['note']})")
    c4 = r["class4_descriptive"]
    print(f"\n--- CLASS-4 (descriptive, NOT gating) ---")
    print(f"  census class-4 n={c4['n_class4']}/{c4['n_census']}  "
          f"AUC(q4-vs-rest, score)={c4['auc_v8_q4_vs_rest']}  "
          f"AUC(by P>=4)={c4['auc_v8_q4_by_pnext']}")
    print(f"  {c4['note']}")
    pinv = r["palette_invariance"]
    if "error" in pinv:
        print(f"\n--- PALETTE-INVARIANCE: ERROR {pinv['error']}")
    else:
        print(f"\n--- PALETTE-INVARIANCE (census-144, {NEUTRAL_PALETTE} vs {len(pinv['held_out_palettes'])} held-out) ---")
        print(f"  mean Spearman {pinv['mean_spearman']}  pooled {pinv['pooled_spearman']}")
        for pal, rho in pinv["spearman_twilight_vs_heldout"].items():
            print(f"    {pal:52s} rho={rho}")


if __name__ == "__main__":
    main()
