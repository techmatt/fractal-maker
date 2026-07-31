#!/usr/bin/env python
r"""v9 evaluation battery — SYSTEM against SYSTEM, on a pre-registered bar.

v9 is the v8 recipe trained verbatim on the same corpus re-rendered at the raised
iteration cap (docs/design/auto_maxiter.md). The question is therefore not "is the model
better" but "is the SYSTEM — model plus the renders it consumes — at least as good", so
the primary comparison is:

    v9 scoring the NEW (raised-cap) renders   vs   deployed v8 scoring its OWN (old) renders

paired by location. Both arms are what you would actually ship, which is the only
comparison a deploy decision can be made from.

THE DIAGNOSTIC ARM, and why it is worth its cost: v8 scored on the NEW renders. That is
the train/deploy-mismatch arm — a model trained on flat-8000 tiles reading raised-cap
tiles. It separates "the model improved" from "the inputs improved", and the two
comparisons fail in OPPOSITE directions:

  * if v9-on-new > v8-on-old AND v8-on-new < v8-on-old, the gain is the model learning
    from better data (v8 cannot exploit the new inputs; v9 can);
  * if v9-on-new > v8-on-old AND v8-on-new > v8-on-old, the raised-cap renders are simply
    easier to score and the retrain may be carrying little of the gain.

Whether they agree is reported explicitly rather than left for a reader to reconstruct.

PRE-REGISTERED, and pre-registered means committed to source before the numbers exist —
the constants below were written and committed while the v9 corpus was still rendering.
Label noise forbids reading small AUC differences on the >=3 boundary (protocol S3;
deferred_recalibration's 0.55-0.65 "label more" band), so the claim available is
NON-INFERIORITY, not superiority.

  PRIMARY      census-144 q3(label>=3)-vs-rest AUC. v9-on-new vs v8-on-old, paired DeLong.
               v8's deployed number: 0.751. Non-inferior iff AUC_v9 >= 0.751 - 0.05 AND
               the paired DeLong does not put v9 significantly below v8 (p<0.05 & delta<0).
  SECONDARY    the 526-row unbiased mandelbrot floor (26 positives), same construction.
               v8: 0.868. NOTE this floor is NOT flattered for v8 the way v7's was — both
               v8 and v9 trained on these locations, so the arms are symmetric.
  DIAGNOSTIC   v8 on the new renders (above). Reported, never gating.
  DESCRIPTIVE  class-4 discrimination on the 22 class-4 census locations. v8: 0.813.
  PALETTE      census-144 under twilight_shifted vs the 8 held-out palettes, Spearman.
               v8: mean 0.896, range 0.767-0.975. A LARGE move here means something other
               than the cap changed — investigate before reporting.

Freezes data/v9/eval_scores_v9.jsonl (durable) with per-location v8_/v9_ columns so the
keeper cut can be recut against it without re-scoring.

  uv run python tools/v9/eval_v9.py
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

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import paths  # noqa: E402
from eval_delong import boot_ci, delong_paired  # reuse — do NOT reimplement DeLong  # noqa: E402
from classifier.data_v4 import (CANON_SCALE, CANON_SHIFT, NEUTRAL_PALETTE,  # noqa: E402
                                load_locations)
from classifier.train_v2 import detect_device  # noqa: E402
from classifier.train_v8 import (CENSUS_SOURCE, FLOOR_SOURCE, derive_k,  # noqa: E402
                                 score_renders_k)
from eval_v8 import load_model, q_auc  # reuse v8's loader verbatim  # noqa: E402

V8_CACHE = ROOT / "data/v8/cache_manifest.jsonl"     # the OLD (flat-8000) renders
V9_CACHE = ROOT / "data/v9/cache_manifest.jsonl"     # the NEW (raised-cap) renders
V9_CKPT = ROOT / "data/classifier/v9/model_best.pt"
V8_CKPT = ROOT / "data/classifier/v8/model_best.pt"
BUILD_META = ROOT / "data/v9/build_metadata.json"
COLORMAPS = ROOT / "data/v9/colormaps.json"
MANIFEST = ROOT / "data/v8/manifest.jsonl"           # v9 has none; it reads v8's
BIN = ROOT / "target/release/fractal-generator.exe"
EVAL_SCORES_OUT = "data/v9/eval_scores_v9.jsonl"
EVAL_RESULTS_OUT = "data/v9/eval_results_v9.json"

# ======================= PRE-REGISTERED (committed before the numbers) ============= #
V8_CENSUS_Q3_REFERENCE = 0.751      # deployed v8 on its own renders (eval_results_v8.json)
V8_FLOOR_Q3_REFERENCE = 0.868       # 526-row unbiased mandelbrot floor, 26 positives
V8_CLASS4_REFERENCE = 0.813         # 22 class-4 census locations (descriptive)
V8_PINV_MEAN_REFERENCE = 0.896      # mean Spearman, twilight vs 8 held-out
V8_PINV_RANGE_REFERENCE = (0.767, 0.975)
# n=144 cannot resolve an AUC gap inside ~0.05; a drop inside the band means "label more",
# not "v9 failed". Same margin v8 was judged on — not re-chosen for this run.
NONINF_MARGIN = 0.05
# A palette-invariance move larger than this is not a cap effect and wants investigation
# before the rest of the report is believed.
PINV_INVESTIGATE_DELTA = 0.10
# ================================================================================== #


def paired_block(name, labels, s_a, s_b, thr, label_a, label_b):
    """AUC(label>=thr vs rest) for two paired score vectors + paired DeLong + boot CIs."""
    y = (np.asarray(labels) >= thr).astype(int)
    aa, ab, z, p = delong_paired(y, s_a, s_b)
    ci_a, ci_b = boot_ci(y, np.asarray(s_a)), boot_ci(y, np.asarray(s_b))
    return {"name": name, "n": int(len(labels)), "thr": thr, "n_pos": int(y.sum()),
            "arm_a": label_a, "arm_b": label_b,
            "auc_a": round(aa, 4), "auc_a_ci95": [round(ci_a[0], 4), round(ci_a[1], 4)],
            "auc_b": round(ab, 4), "auc_b_ci95": [round(ci_b[0], 4), round(ci_b[1], 4)],
            "delta_b_minus_a": round(ab - aa, 4), "delong_z": round(z, 3),
            "delong_p": round(p, 4)}


def _aligned_canonicals(cache_path, want_ids):
    """Canonical renders for `want_ids`, in that exact order, from one cache manifest.

    Alignment is by location_id and asserted, not assumed: the two caches are supposed to
    hold the same locations under the same ids, and if they ever do not, a paired test
    would silently compare different fractals."""
    locs = {l.location_id: l for l in load_locations(cache_path=cache_path)}
    missing = [i for i in want_ids if i not in locs]
    if missing:
        raise SystemExit(f"{cache_path}: {len(missing)} eval loc_ids absent "
                         f"(e.g. {missing[:5]}) — the two caches are not the same corpus")
    return [locs[i].canonical() for i in want_ids], [locs[i] for i in want_ids]


def main():
    device = detect_device("auto")
    for p in (V9_CKPT, V8_CKPT):
        if not p.exists():
            sys.exit(f"checkpoint missing: {p}")
    m9, tf9, K9, _ = load_model(V9_CKPT, device)
    m8, tf8, K8, _ = load_model(V8_CKPT, device)
    print(f"v9 K={K9} | v8 K={K8} | device {device}")

    # eval slice identity comes from the v9 cache; the v8 cache supplies the OLD renders
    v9_locs = [l for l in load_locations(cache_path=V9_CACHE) if l.split == "eval"]
    ids = [l.location_id for l in v9_locs]
    labels = np.array([l.label for l in v9_locs])
    src = np.array([l.source for l in v9_locs])
    ft = np.array([l.fractal_type for l in v9_locs])
    canon_new = [l.canonical() for l in v9_locs]
    canon_old, _ = _aligned_canonicals(V8_CACHE, ids)
    n_census = int((src == CENSUS_SOURCE).sum())
    n_floor = int((src == FLOOR_SOURCE).sum())
    print(f"eval {len(ids)}: census {n_census} + mandelbrot-floor {n_floor}")

    print_bar()

    # --- the three arms; each model x render-set scored exactly once ---
    p9_new, s9_new = derive_k(                       # CANDIDATE system
        score_renders_k(m9, canon_new, tf9, device, K9 - 1, num_workers=0))
    p8_old, s8_old = derive_k(                       # DEPLOYED system
        score_renders_k(m8, canon_old, tf8, device, K8 - 1, num_workers=0))
    _, s8_new = derive_k(                            # DIAGNOSTIC: inputs moved, model fixed
        score_renders_k(m8, canon_new, tf8, device, K8 - 1, num_workers=0))

    results = {"score_key": "sum_sigma (rank score)",
               "arms": {"primary_a": "v8 on v8 (old, flat-8000) renders — DEPLOYED SYSTEM",
                        "primary_b": "v9 on v9 (new, raised-cap) renders — CANDIDATE SYSTEM",
                        "diagnostic": "v8 on v9 (new) renders — train/deploy mismatch"},
               "pre_registered": {
                   "v8_census_q3": V8_CENSUS_Q3_REFERENCE,
                   "v8_floor_q3": V8_FLOOR_Q3_REFERENCE,
                   "v8_class4": V8_CLASS4_REFERENCE,
                   "v8_palette_mean_spearman": V8_PINV_MEAN_REFERENCE,
                   "v8_palette_range": list(V8_PINV_RANGE_REFERENCE),
                   "noninf_margin": NONINF_MARGIN,
                   "claim": "non-inferiority, not superiority (label noise on the >=3 boundary)"}}

    # ---------------- PRIMARY ----------------
    cm = src == CENSUS_SOURCE
    cb = paired_block("census-144 q3", labels[cm], s8_old[cm], s9_new[cm], 3,
                      "v8-on-old", "v9-on-new")
    non_inferior = (cb["auc_b"] >= V8_CENSUS_Q3_REFERENCE - NONINF_MARGIN) and not (
        cb["delta_b_minus_a"] < 0 and cb["delong_p"] < 0.05)
    cb["v9_vs_reference"] = round(cb["auc_b"] - V8_CENSUS_Q3_REFERENCE, 4)
    cb["non_inferior_verdict"] = "NON-INFERIOR" if non_inferior else "INFERIOR / inconclusive"
    results["census_q3"] = cb

    # ---------------- SECONDARY ----------------
    fm = src == FLOOR_SOURCE
    fb = paired_block("mandelbrot-floor q3", labels[fm], s8_old[fm], s9_new[fm], 3,
                      "v8-on-old", "v9-on-new")
    fb["v9_vs_reference"] = round(fb["auc_b"] - V8_FLOOR_Q3_REFERENCE, 4)
    fb["verdict"] = ("REGRESSION" if (fb["delta_b_minus_a"] < 0 and fb["delong_p"] < 0.05)
                     else "non-inferior")
    fb["note"] = ("Unlike the v7-vs-v8 floor comparison, this one is SYMMETRIC: v8 and v9 "
                  "both trained on these locations, so neither arm is flattered.")
    results["mandelbrot_floor_q3"] = fb

    # ---------------- DIAGNOSTIC: v8 on the new renders ----------------
    db = paired_block("census-144 q3 (diagnostic)", labels[cm], s8_old[cm], s8_new[cm], 3,
                      "v8-on-old", "v8-on-new")
    d_inputs = db["delta_b_minus_a"]          # inputs alone, model held fixed
    d_system = cb["delta_b_minus_a"]          # model + inputs
    if d_system > 0 and d_inputs <= 0:
        agree = ("AGREE: the system gained while v8 alone LOST on the new renders — the "
                 "gain is v9 learning from better data, not the renders being easier.")
    elif d_system > 0 and d_inputs > 0:
        agree = ("PARTIAL: both moved up, so some of the system gain is the renders being "
                 "easier to score rather than the retrain. Attribute with care: the "
                 f"inputs-alone arm carries {d_inputs:+.4f} of the {d_system:+.4f}.")
    elif d_system <= 0 and d_inputs > 0:
        agree = ("DISAGREE: the inputs alone helped v8 but the system did not improve — "
                 "that points at the v9 TRAIN, not at the cap.")
    else:
        agree = ("AGREE (negative): neither the inputs alone nor the system improved on "
                 "this instrument.")
    db["interpretation"] = agree
    db["delta_inputs_only"] = d_inputs
    db["delta_system"] = d_system
    db["note"] = "DIAGNOSTIC ONLY — never gating."
    results["diagnostic_v8_on_new"] = db

    # ---------------- CLASS-4 (descriptive) ----------------
    y4 = (labels[cm] >= 4).astype(int)
    c4 = {"n_census": int(cm.sum()), "n_class4": int(y4.sum()),
          "auc_v9_q4_vs_rest": (None if y4.min() == y4.max() else round(q_auc(y4, s9_new[cm]), 4)),
          "auc_v9_q4_by_pnext": (None if y4.min() == y4.max() or K9 < 4 else
                                 round(q_auc(y4, p9_new[cm][:, 2]), 4)),
          "auc_v8_q4_vs_rest_on_old": (None if y4.min() == y4.max()
                                       else round(q_auc(y4, s8_old[cm]), 4)),
          "v8_reference": V8_CLASS4_REFERENCE,
          "note": ("Descriptive only, NOT gating. Class-4 census locations are all "
                   "julia:multibrot; native-multibrot class-4 remains extrapolation.")}
    c4["class4_by_family"] = dict(Counter(ft[cm][y4.astype(bool)].tolist()))
    results["class4_descriptive"] = c4

    # ---------------- freeze the durable eval slice ----------------
    with paths.durable(EVAL_SCORES_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for i, l in enumerate(v9_locs):
            row = {"location_id": l.location_id, "label": l.label, "source": l.source,
                   "group_id": l.group_id, "fractal_type": l.fractal_type,
                   "v9_score": float(s9_new[i]),          # v9 on the NEW renders
                   "v8_score": float(s8_old[i]),          # deployed v8 on its OWN renders
                   "v8_score_on_new": float(s8_new[i])}   # diagnostic arm
            for k in range(K9 - 1):
                row[f"v9_p_ge{k+2}"] = float(p9_new[i, k])
            for k in range(K8 - 1):
                row[f"v8_p_ge{k+2}"] = float(p8_old[i, k])
            f.write(json.dumps(row) + "\n")

    # ---------------- PALETTE-INVARIANCE ----------------
    meta = json.loads(BUILD_META.read_text(encoding="utf-8"))
    held_out = meta["aug_recipe"]["palettes"]["held_out"]
    census_locs = [l for l in v9_locs if l.source == CENSUS_SOURCE]
    pinv = palette_invariance(census_locs, held_out, m9, tf9, K9, device)
    if "error" not in pinv and pinv.get("mean_spearman") is not None:
        d = pinv["mean_spearman"] - V8_PINV_MEAN_REFERENCE
        pinv["delta_vs_v8"] = round(d, 4)
        pinv["verdict"] = ("INVESTIGATE — a move this large is not a cap effect"
                           if abs(d) > PINV_INVESTIGATE_DELTA else "consistent with v8")
    results["palette_invariance"] = pinv

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
    geometry AND THE LIVE CAP, score with v9, report Spearman(twilight, palette).

    The cap matters here: these renders are made on the fly, so if they were left at the
    subcommand's flat default the invariance read would be taken on old-cap frames while
    every other number in this battery is on new-cap frames."""
    if not BIN.exists():
        return {"error": f"release binary missing: {BIN}"}
    sys.path.insert(0, str(ROOT / "tools" / "scoring"))
    import active_ckpt as ac
    rows, loc_mod = _coord_rows_for(census)
    palettes = [NEUTRAL_PALETTE] + list(held_out)
    with tempfile.TemporaryDirectory(prefix="v9_pinv_") as td:
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
    return {"n_census": len(census), "held_out_palettes": held_out,
            "instrument": NEUTRAL_PALETTE,
            "rendered_at_cap": "auto_maxiter(fw) — the live policy, per row",
            "spearman_twilight_vs_heldout": per_pal,
            "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
            "range": [round(min(rhos), 4), round(max(rhos), 4)] if rhos else None,
            "pooled_spearman": round(float(pooled), 4) if np.isfinite(pooled) else None,
            "v8_reference": {"mean": V8_PINV_MEAN_REFERENCE,
                             "range": list(V8_PINV_RANGE_REFERENCE)}}


def print_bar():
    lo = V8_CENSUS_Q3_REFERENCE - NONINF_MARGIN
    print("\n" + "=" * 80)
    print("PRE-REGISTERED BAR — stated before any number is computed")
    print("=" * 80)
    print("Claim available is NON-INFERIORITY, not superiority: label noise forbids reading")
    print("small AUC differences on the >=3 boundary (protocol S3).")
    print(f"  PRIMARY   census-144 q3 AUC, v9-on-NEW vs deployed v8-on-OLD, paired DeLong.")
    print(f"            v8 = {V8_CENSUS_Q3_REFERENCE}. NON-INFERIOR iff AUC_v9 >= {lo:.3f}")
    print(f"            AND DeLong does not put v9 significantly below v8 (p<0.05 & delta<0).")
    print(f"  SECONDARY 526-row unbiased mandelbrot floor (26 pos), same construction. "
          f"v8 = {V8_FLOOR_Q3_REFERENCE}.")
    print(f"  DIAGNOSTIC v8 on the NEW renders — separates 'model improved' from 'inputs")
    print(f"            improved'. Reported, NEVER gating.")
    print(f"  DESCRIPTIVE class-4 AUC on the 22 class-4 census loc. v8 = {V8_CLASS4_REFERENCE}.")
    print(f"  PALETTE   census-144 twilight vs 8 held-out, Spearman. v8 mean "
          f"{V8_PINV_MEAN_REFERENCE}, range {V8_PINV_RANGE_REFERENCE}.")
    print(f"            |delta mean| > {PINV_INVESTIGATE_DELTA} => something other than the "
          f"cap changed; investigate first.")
    print("=" * 80)


def print_report(r):
    print("\n" + "=" * 80)
    print("v9 EVALUATION — system against system")
    print("=" * 80)
    cb = r["census_q3"]
    print(f"--- PRIMARY census-144 q3 (n={cb['n']}, q3-pos={cb['n_pos']}) ---")
    print(f"  v8-on-old AUC {cb['auc_a']} CI{cb['auc_a_ci95']}   (reference {V8_CENSUS_Q3_REFERENCE})")
    print(f"  v9-on-new AUC {cb['auc_b']} CI{cb['auc_b_ci95']}")
    print(f"  paired DeLong delta(v9-v8)={cb['delta_b_minus_a']:+.4f} z={cb['delong_z']} "
          f"p={cb['delong_p']}")
    print(f"  vs reference {cb['v9_vs_reference']:+.4f}  ->  {cb['non_inferior_verdict']}")
    fb = r["mandelbrot_floor_q3"]
    print(f"\n--- SECONDARY mandelbrot floor q3 (n={fb['n']}, q3-pos={fb['n_pos']}) ---")
    print(f"  v8-on-old {fb['auc_a']} / v9-on-new {fb['auc_b']}  "
          f"delta={fb['delta_b_minus_a']:+.4f} p={fb['delong_p']}  -> {fb['verdict']}")
    print(f"  ({fb['note']})")
    db = r["diagnostic_v8_on_new"]
    print(f"\n--- DIAGNOSTIC v8 on the NEW renders (not a bar) ---")
    print(f"  v8-on-old {db['auc_a']} / v8-on-new {db['auc_b']}  "
          f"delta(inputs only)={db['delta_inputs_only']:+.4f} p={db['delong_p']}")
    print(f"  system delta {db['delta_system']:+.4f}")
    print(f"  {db['interpretation']}")
    c4 = r["class4_descriptive"]
    print(f"\n--- CLASS-4 (descriptive, NOT gating) ---")
    print(f"  n={c4['n_class4']}/{c4['n_census']}  v9 AUC(q4-vs-rest)={c4['auc_v9_q4_vs_rest']}  "
          f"AUC(by P>=4)={c4['auc_v9_q4_by_pnext']}   (v8 reference {V8_CLASS4_REFERENCE})")
    pinv = r["palette_invariance"]
    if "error" in pinv:
        print(f"\n--- PALETTE-INVARIANCE: ERROR {pinv['error']}")
    else:
        print(f"\n--- PALETTE-INVARIANCE (census-144, {NEUTRAL_PALETTE} vs "
              f"{len(pinv['held_out_palettes'])} held-out) ---")
        print(f"  mean Spearman {pinv['mean_spearman']} range {pinv['range']}  "
              f"pooled {pinv['pooled_spearman']}")
        print(f"  v8 reference mean {V8_PINV_MEAN_REFERENCE} range "
              f"{list(V8_PINV_RANGE_REFERENCE)}  ->  delta {pinv.get('delta_vs_v8')} "
              f"({pinv.get('verdict')})")
        for pal, rho in pinv["spearman_twilight_vs_heldout"].items():
            print(f"    {pal:52s} rho={rho}")


if __name__ == "__main__":
    raise SystemExit(main())
