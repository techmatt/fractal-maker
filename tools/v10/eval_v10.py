#!/usr/bin/env python
r"""v10 certification battery — bars LOADED from the pre-registration, never restated.

`data/v10/prereg_v10.json` was written and committed before this file was run
(`tools/v10/prereg.py`). Every threshold, every margin and the uniform-90's
power-derived separation bar come out of that file. Nothing here invents a number, which
is the mechanical form of "pre-registered": a bar in the eval script is a bar that can be
edited after seeing the results, and a bar in a committed artifact the script loads cannot.

THE BASELINE IS v8 RE-SCORED, on the same tiles v10 reads. v8's own flat-8000 cache was
deleted 2026-07-31, and the deployed system today renders through the live `auto_maxiter`
policy anyway — so "v8 on the v10 tiles" IS the deployed system, and it is what a deploy
decision has to be made against. Every arm is therefore paired on identical inputs, and the
only thing that differs between the arms is the model. (v9's battery needed a third
diagnostic arm to separate "model improved" from "inputs improved"; v10 does not, because
the inputs are held fixed by construction.)

THREE ARMS, and what each can and cannot see:
  PRIMARY   census-144, AUC(label>=3), non-inferiority vs v8. The unchanged instrument the
            v7->v8->v9 chain is comparable on. It is julia:multibrot and every appended
            location is native-plane, so it CANNOT see the intervention's target: a null is
            the expected outcome and reads as non-regression.
  FLOOR     loose0_v3, 526 unbiased base-rate mandelbrot locations, same construction.
            Native-plane, so unlike the census it is at least in the right half-plane —
            but its locations predate the maneuver sweep.
  NEW       uniform-90, AUC(label>=2), separation-vs-chance at the power-derived bar. The
            only arm that reads the population the appended labels come from.
  CLASS-4   descriptive, no bar. Asserted to contain exactly the 22 census fours — the 23
            appended fours are train-side and must not reach any eval number.
  PALETTE   census-144 under twilight_shifted vs the 8 held-out palettes, Spearman.
            Descriptive; a LARGE move means something other than the labels changed.

Freezes data/v10/eval_scores_v10.jsonl (durable) with per-location v8_/v10_ columns so a
keeper cut can be recut without re-scoring.

  uv run python tools/v10/eval_v10.py
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
sys.path.insert(0, str(ROOT / "tools" / "v8"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

from scipy.stats import spearmanr  # noqa: E402

import paths  # noqa: E402
from eval_delong import boot_ci, delong_paired  # reuse — do NOT reimplement DeLong  # noqa: E402
from classifier.data_v4 import (CANON_SCALE, CANON_SHIFT, NEUTRAL_PALETTE,  # noqa: E402
                                load_locations)
from classifier.train_v2 import detect_device  # noqa: E402
from classifier.train_v8 import derive_k, score_renders_k  # noqa: E402
from eval_v8 import load_model, q_auc  # reuse v8's loader verbatim  # noqa: E402

PREREG = ROOT / "data/v10/prereg_v10.json"
V10_CACHE = ROOT / "data/v10/cache_manifest.jsonl"
V10_CKPT = ROOT / "data/classifier/v10/model_best.pt"
V8_CKPT = ROOT / "data/classifier/v8/model_best.pt"
BUILD_META = ROOT / "data/v10/build_metadata.json"
COLORMAPS = ROOT / "data/v10/colormaps.json"
MANIFEST = ROOT / "data/v10/manifest.jsonl"
BIN = ROOT / "target/release/fractal-generator.exe"
EVAL_SCORES_OUT = "data/v10/eval_scores_v10.jsonl"
EVAL_RESULTS_OUT = "data/v10/eval_results_v10.json"

CENSUS_SOURCE = "prospect_census"
FLOOR_SOURCE = "loose0_v3_floor"
UNIFORM_SOURCE = "maneuver_uniform_v1"
PINV_INVESTIGATE_DELTA = 0.10


def paired_block(name, labels, s_base, s_cand, thr, label_base, label_cand):
    """AUC(label>=thr vs rest) for two paired score vectors + paired DeLong + boot CIs."""
    y = (np.asarray(labels) >= thr).astype(int)
    ab, ac_, z, p = delong_paired(y, s_base, s_cand)
    ci_b, ci_c = boot_ci(y, np.asarray(s_base)), boot_ci(y, np.asarray(s_cand))
    return {"name": name, "n": int(len(labels)), "thr": thr, "n_pos": int(y.sum()),
            "arm_base": label_base, "arm_cand": label_cand,
            "auc_base": round(ab, 4), "auc_base_ci95": [round(ci_b[0], 4), round(ci_b[1], 4)],
            "auc_cand": round(ac_, 4), "auc_cand_ci95": [round(ci_c[0], 4), round(ci_c[1], 4)],
            "delta_cand_minus_base": round(ac_ - ab, 4), "delong_z": round(z, 3),
            "delong_p": round(p, 4)}


def noninferior(block, margin):
    """The pre-registered non-inferiority rule, applied to a paired block."""
    return (block["auc_cand"] >= block["auc_base"] - margin) and not (
        block["delta_cand_minus_base"] < 0 and block["delong_p"] < 0.05)


def main():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    arms = prereg["arms"]
    device = detect_device("auto")
    for p in (V10_CKPT, V8_CKPT):
        if not p.exists():
            sys.exit(f"checkpoint missing: {p}")
    m10, tf10, K10, _ = load_model(V10_CKPT, device)
    m8, tf8, K8, _ = load_model(V8_CKPT, device)
    print(f"v10 K={K10} | v8 K={K8} | device {device}")

    print_bar(prereg)

    locs = [l for l in load_locations(cache_path=V10_CACHE) if l.split == "eval"]
    labels = np.array([l.label for l in locs])
    src = np.array([l.source for l in locs])
    ft = np.array([l.fractal_type for l in locs])
    canon = [l.canonical() for l in locs]
    print(f"eval {len(locs)}: census {(src==CENSUS_SOURCE).sum()} + floor "
          f"{(src==FLOOR_SOURCE).sum()} + uniform {(src==UNIFORM_SOURCE).sum()}")

    # The class-4 discipline, asserted before any number is reported.
    q4_nonc = int(((labels == 4) & (src != CENSUS_SOURCE)).sum())
    assert q4_nonc == 0, (f"{q4_nonc} class-4 eval locations are not census — the appended "
                          f"fours were supposed to be train-side only")
    n_q4 = int((labels == 4).sum())
    assert n_q4 == arms["class4_descriptive"]["n_class4_eval"], (
        f"{n_q4} class-4 eval rows, pre-registration says "
        f"{arms['class4_descriptive']['n_class4_eval']}")

    # --- both models over the SAME renders; each scored exactly once ---
    p10, s10 = derive_k(score_renders_k(m10, canon, tf10, device, K10 - 1, num_workers=0))
    p8, s8 = derive_k(score_renders_k(m8, canon, tf8, device, K8 - 1, num_workers=0))

    results = {"score_key": "sum_sigma (rank score)",
               "prereg_source": "data/v10/prereg_v10.json",
               "prereg": prereg,
               "arms": {"baseline": "v8 re-scored on the v10 tiles — THE DEPLOYED SYSTEM",
                        "candidate": "v10 on the same tiles",
                        "note": ("inputs are held identical between arms by construction, "
                                 "so no diagnostic arm is needed: the only thing that "
                                 "differs is the model.")}}

    # ---------------- PRIMARY: census-144 ----------------
    cm = src == CENSUS_SOURCE
    a = arms["primary_census144"]
    cb = paired_block("census-144 q3", labels[cm], s8[cm], s10[cm], 3, "v8", "v10")
    cb["bar"] = a["bar"]
    cb["verdict"] = ("NON-INFERIOR" if noninferior(cb, a["noninf_margin"])
                     else "INFERIOR / inconclusive")
    cb["reads"] = ("julia:multibrot only — cannot see the appended native-plane "
                   "population; a null is non-regression, not a null about the data")
    results["primary_census144"] = cb

    # ---------------- FLOOR: loose0_v3 ----------------
    fm = src == FLOOR_SOURCE
    a = arms["floor_loose0_v3"]
    fb = paired_block("mandelbrot-floor q3", labels[fm], s8[fm], s10[fm], 3, "v8", "v10")
    fb["bar"] = a["bar"]
    fb["verdict"] = ("NON-INFERIOR" if noninferior(fb, a["noninf_margin"])
                     else "INFERIOR / inconclusive")
    fb["note"] = a["note"]
    results["floor_loose0_v3"] = fb

    # ---------------- NEW: uniform-90 ----------------
    um = src == UNIFORM_SOURCE
    a = arms["new_uniform90"]
    ub = paired_block("maneuver-uniform-90 q2", labels[um], s8[um], s10[um], 2, "v8", "v10")
    ub["bar"] = a["bar"]
    ub["separation_bar"] = a["separation_bar"]
    separates = (ub["auc_cand"] >= a["separation_bar"]
                 and ub["auc_cand_ci95"][0] > 0.50)
    v8_separates = (ub["auc_base"] >= a["separation_bar"] and ub["auc_base_ci95"][0] > 0.50)
    ub["verdict"] = "SEPARATES" if separates else "UNDERPOWERED / does not separate"
    ub["v8_verdict"] = "SEPARATES" if v8_separates else "does not separate (as measured)"
    ub["gating"] = False
    ub["reads"] = ("the maneuver-view population the appended labels come from — the only "
                   "arm that does")
    results["new_uniform90"] = ub

    # ---------------- CLASS-4 (descriptive) ----------------
    y4 = (labels[cm] >= 4).astype(int)
    c4 = {"n_census": int(cm.sum()), "n_class4": int(y4.sum()),
          "auc_v10_q4_vs_rest": q_auc(y4, s10[cm]),
          "auc_v10_q4_by_pnext": (None if K10 < 4 else q_auc(y4, p10[cm][:, 2])),
          "auc_v8_q4_vs_rest": q_auc(y4, s8[cm]),
          "class4_by_family": dict(Counter(ft[cm][y4.astype(bool)].tolist())),
          "appended_class4_train_side": 23,
          "note": ("DESCRIPTIVE, no bar. All 23 appended class-4 locations are train-side "
                   "and are asserted absent from every eval number above.")}
    for k in ("auc_v10_q4_vs_rest", "auc_v10_q4_by_pnext", "auc_v8_q4_vs_rest"):
        if c4[k] is not None:
            c4[k] = round(c4[k], 4)
    results["class4_descriptive"] = c4

    # ---------------- freeze the durable per-location slice ----------------
    with paths.durable(EVAL_SCORES_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for i, l in enumerate(locs):
            row = {"location_id": l.location_id, "label": l.label, "source": l.source,
                   "group_id": l.group_id, "fractal_type": l.fractal_type,
                   "v10_score": float(s10[i]), "v8_score": float(s8[i])}
            for k in range(K10 - 1):
                row[f"v10_p_ge{k+2}"] = float(p10[i, k])
            for k in range(K8 - 1):
                row[f"v8_p_ge{k+2}"] = float(p8[i, k])
            f.write(json.dumps(row) + "\n")

    # ---------------- PALETTE-INVARIANCE (descriptive) ----------------
    meta = json.loads(BUILD_META.read_text(encoding="utf-8"))
    held_out = meta["aug_recipe"]["palettes"]["held_out"]
    results["palette_invariance"] = palette_invariance(
        [l for l in locs if l.source == CENSUS_SOURCE], held_out, m10, tf10, K10, device)

    paths.durable(EVAL_RESULTS_OUT, mkparents=True).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print_report(results)
    print(f"\nwrote {EVAL_SCORES_OUT} and {EVAL_RESULTS_OUT} (durable)")
    return 0


def _coord_rows_for(locs):
    import location as loc_mod
    man = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            man[r["loc_id"]] = r
    return [man[l.location_id] for l in locs], loc_mod


def palette_invariance(census, held_out, model, tf, K, device):
    """Render census-144 under twilight_shifted + each held-out palette at the canonical
    geometry AND THE LIVE CAP, score with v10, report Spearman(twilight, palette)."""
    if not BIN.exists():
        return {"error": f"release binary missing: {BIN}"}
    sys.path.insert(0, str(ROOT / "tools" / "scoring"))
    import active_ckpt as ac
    rows, loc_mod = _coord_rows_for(census)
    palettes = [NEUTRAL_PALETTE] + list(held_out)
    with tempfile.TemporaryDirectory(prefix="v10_pinv_") as td:
        troot = Path(td)
        plan, index = [], {}
        for r in rows:
            lid = r["loc_id"]
            ftype = r.get("fractal_type", "mandelbrot")
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
             "--log-every", "100000"],
            cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": f"render failed: {proc.stderr[-1500:]}"}

        from classifier.data_v4 import Render

        def score_palette(pal):
            renders = [Render(path=Path(index[(r["loc_id"], pal)]), palette=pal,
                              palette_family=pal, scale=CANON_SCALE, shift_id=CANON_SHIFT,
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
            per_pal[pal] = round(rho, 4) if rho is not None else None
            if rho is not None:
                rhos.append(rho)
        pooled = spearmanr(np.tile(s_twi, len(held_out)),
                           np.concatenate(held_scores)).correlation
    v9_ref = 0.9002     # v9's measured mean, for orientation only — NOT a bar
    out = {"n_census": len(census), "held_out_palettes": held_out,
           "instrument": NEUTRAL_PALETTE,
           "rendered_at_cap": "auto_maxiter(fw) — the live policy, per row",
           "spearman_twilight_vs_heldout": per_pal,
           "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
           "range": [round(min(rhos), 4), round(max(rhos), 4)] if rhos else None,
           "pooled_spearman": round(float(pooled), 4) if np.isfinite(pooled) else None,
           "v9_reference_mean": v9_ref,
           "note": "DESCRIPTIVE, no bar."}
    if out["mean_spearman"] is not None:
        d = out["mean_spearman"] - v9_ref
        out["delta_vs_v9"] = round(d, 4)
        out["read"] = ("INVESTIGATE — a move this large is not a label effect"
                       if abs(d) > PINV_INVESTIGATE_DELTA else "consistent with v9")
    return out


def print_bar(prereg):
    print("\n" + "=" * 80)
    print("PRE-REGISTERED BARS — loaded from data/v10/prereg_v10.json, not restated here")
    print("=" * 80)
    print(f"  baseline: {prereg['baseline']}")
    for name, a in prereg["arms"].items():
        print(f"  {name:<22} n={a.get('n')} n_pos={a.get('n_pos')} gating={a['gating']}")
        print(f"      {a['bar']}")
    print("=" * 80)


def _pb(tag, b):
    print(f"\n--- {tag} (n={b['n']}, pos={b['n_pos']} at label>={b['thr']}) ---")
    print(f"  v8  AUC {b['auc_base']} CI{b['auc_base_ci95']}   (baseline, re-scored)")
    print(f"  v10 AUC {b['auc_cand']} CI{b['auc_cand_ci95']}")
    print(f"  paired DeLong delta(v10-v8)={b['delta_cand_minus_base']:+.4f} "
          f"z={b['delong_z']} p={b['delong_p']}")
    print(f"  ->  {b['verdict']}")


def print_report(r):
    print("\n" + "=" * 80)
    print("v10 CERTIFICATION — v10 vs v8 re-scored, on identical tiles")
    print("=" * 80)
    _pb("PRIMARY census-144", r["primary_census144"])
    print(f"  ({r['primary_census144']['reads']})")
    _pb("FLOOR loose0_v3", r["floor_loose0_v3"])
    _pb("NEW maneuver-uniform-90", r["new_uniform90"])
    ub = r["new_uniform90"]
    print(f"  separation bar {ub['separation_bar']}   v8 on the same 90: {ub['v8_verdict']}")
    print(f"  ({ub['reads']}; NOT gating for adoption)")
    c4 = r["class4_descriptive"]
    print(f"\n--- CLASS-4 (descriptive, NOT gating) ---")
    print(f"  n={c4['n_class4']}/{c4['n_census']}  v10 AUC(q4-vs-rest)={c4['auc_v10_q4_vs_rest']}"
          f"  AUC(by P>=4)={c4['auc_v10_q4_by_pnext']}   v8 {c4['auc_v8_q4_vs_rest']}")
    print(f"  {c4['note']}")
    pinv = r["palette_invariance"]
    if "error" in pinv:
        print(f"\n--- PALETTE-INVARIANCE: ERROR {pinv['error']}")
    else:
        print(f"\n--- PALETTE-INVARIANCE (census-144, descriptive) ---")
        print(f"  mean Spearman {pinv['mean_spearman']} range {pinv['range']}  "
              f"pooled {pinv['pooled_spearman']}   vs v9 {pinv.get('delta_vs_v9')} "
              f"({pinv.get('read')})")


if __name__ == "__main__":
    raise SystemExit(main())
