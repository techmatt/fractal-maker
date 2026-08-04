#!/usr/bin/env python
"""q4 stage-1 screen TRANSFER READ — does the degree-2 goodness screen carry to
d3/d4/d5 multibrot minibrots?

The stage-1 screen (coarse pre-filter -> OOD mask -> L1 goodness field G ->
G-maxima framing) was fitted on **degree-2** minibrot windows. A degree-d nucleus
has (d-1)-fold local symmetry, so cell-dispersion statistics genuinely differ.
This asks, apples-to-apples: run the *unchanged* fitted screen over freshly-sourced
d3/d4/d5 minibrot fields and see whether it accepts, rejects, or OOD-masks them —
with a freshly-sourced **d2** set as the out-of-sample control.

The trap this read exists to avoid: if the OOD mask rejects multibrot windows, that
is evidence the SCREEN is degree-bound, not evidence the vein is empty (artifact vs.
inherent). So the fate-stratified sheets show the eye the actual fields regardless of
what the screen says.

Nothing here is refit or retuned. The model is `LF.surviving_weights(.., "T2_cells",
2.0)` — the exact deployment fit from `q4_harvest_tight` — and the mask is the exact
`LF._v2_drop` ceilings. The ONLY new code is (a) degree-parametric nucleus sourcing
(via the generalized `deep_center_finder`), (b) multibrot field rendering through the
same `render-one --dump-field` path (`--family multibrot{d}`), and (c) instrumentation
that records per-position FATE using the screen's own functions (asserted to reproduce
`dense_grid`'s survivor set exactly).

Stages:
  corpus-fields  regenerate the 33 d2 label-corpus fields (needed to fit the model)
  source         source ~N minibrot nuclei per degree {2,3,4,5} (generalized Newton)
  fields         render each sourced nucleus to an f64 field (.bin) via render-one
  screen         fit the d2 model, run the screen over every field, aggregate stats
  sheets         fate-stratified sheets (accepted / rejected / OOD-masked) per degree
  all

Run:  uv run python -m tools.studies.q4_multibrot_transfer all
Outputs (all disposable -> scratch/q4_multibrot_transfer/):
  nuclei_d{2,3,4,5}.json · fields/d{d}/*.bin · stats.json · sheet_d{d}.png
Record: docs/design/q4_multibrot_transfer.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mpmath as mp  # noqa: E402
import tools.sourcing.deep_center_finder as dcf  # noqa: E402
from tools.corpus import q4_window_reader as qr  # noqa: E402
from tools.studies import q4_stage1_labelset as LS  # noqa: E402
from tools.studies import q4_stage1_linear_fit as LF  # noqa: E402
from tools.studies import q4_stage1_refit as R  # noqa: E402
from tools.studies import q4_harvest_tight as HT  # noqa: E402

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "scratch" / "q4_multibrot_transfer"
FIELDS = OUT / "fields"
CORPUS_FIELDS = LS.FIELDS                      # data/q4_stage1/fields — DURABLE fit inputs
                                               # (reclassified 2026-08-04; see LS.FIELDS)
FINDINGS = ROOT / "docs" / "design" / "q4_multibrot_transfer.md"

DEGREES = [2, 3, 4, 5]
W, H = LS.W, LS.H                              # 2176 x 1224 (same field geometry as d2)
TIER, C = HT.TIER, HT.C                        # "T2_cells", 2.0 — deployment fit
# Same f64-dumpable size band as the d2 label-set (render-one --dump-field is f64-only;
# fw = 4*size must keep pixel spacing > ~1e-13 at W=2176). Multibrot has no perturbation
# path, so this band bounds every degree identically -> apples-to-apples.
SIZE_LO, SIZE_HI = LS.SIZE_LO, LS.SIZE_HI      # 1e-10, 3e-2
DEDUP_DPS = LS.DEDUP_DPS
N_PER_DEGREE = 12                              # sourced minibrots kept per degree
PERIODS = list(range(3, 16))                  # Newton period band (spread across scales)
NUCLEUS_DPS = 60
NEWTON_STEPS = 60                             # cap: a good seed converges well under this


# --------------------------------------------------------------------------- #
# Sourcing — degree-parametric minibrot nuclei via the generalized Newton.
# Seeds are sampled on rings spanning the degree-d boundary annulus. The multibrot
# of degree d is contained in |c| <= 2^(1/(d-1)); minibrots cluster near ∂M, so we
# ring-sample radii from a small inner radius out just past the escape radius.
# --------------------------------------------------------------------------- #
def _boundary_radius(degree):
    return 2.0 ** (1.0 / (degree - 1))         # d2->2, d3->1.414, d4->1.26, d5->1.19


def _ring_seeds(degree, n_ang=24, n_rad=3):
    """Seeds on n_rad rings between 0.30 and 1.05*R_boundary, n_ang angles each."""
    Rb = _boundary_radius(degree)
    radii = np.linspace(0.30, 1.05 * Rb, n_rad)
    seeds = []
    for r in radii:
        for k in range(n_ang):
            th = 2.0 * np.pi * k / n_ang
            seeds.append((float(r * np.cos(th)), float(r * np.sin(th))))
    return seeds


def _is_minimal_nucleus(c, period, degree, tol):
    """No proper divisor q|period also closes z_q(c)=0 (period is minimal)."""
    for q in range(1, period):
        if period % q == 0 and abs(dcf._orbit(c, q, degree)[0]) < tol:
            return False
    return True


def source_nuclei(degree, *, n_target=N_PER_DEGREE):
    """Source ~n_target valid minibrot nuclei for `degree`, spread across periods."""
    mp.mp.dps = NUCLEUS_DPS
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    origin_eps = mp.mpf("1e-6")                # reject the c=0 period-1 degenerate
    found = {}
    t0 = time.time()
    n_solves = 0
    for sr, si in _ring_seeds(degree):
        seed = mp.mpc(sr, si)
        for p in PERIODS:
            n_solves += 1
            r = dcf.newton_nucleus(seed, p, degree=degree, max_steps=NEWTON_STEPS)
            if not r.converged:
                continue
            if abs(r.c) < origin_eps:          # z=0 fixed point closes every period
                continue
            if not _is_minimal_nucleus(r.c, p, degree, tol):
                continue
            size = dcf.nucleus_size_estimate(r.c, p, degree)
            sabs = float(abs(size)) if size != 0 else 0.0
            if not (SIZE_LO <= sabs <= SIZE_HI):
                continue
            key = dcf.nucleus_dedup_key(r.c, degree, DEDUP_DPS)   # symmetry-canonical
            if key in found:
                continue
            r.newton_residual_log10 = r.residual
            dc = dcf.make_deep_center(r)       # fw_suggest = 4*size, render_maxiter
            inst = dcf.atom_instrument(r.c, p, degree)   # A: |A|≡1/size, arg, req precision
            found[key] = dict(
                degree=degree, period=p, cx=dc.cx, cy=dc.cy,
                fw=dc.fw_suggest, maxiter=dc.render_maxiter,
                size=sabs, newton_res_log10=round(r.residual, 1),
                # §2 covariates — n/|A|/degree logged per window so period-vs-quality
                # reads later without re-running. Nothing keys off them yet.
                abs_A=inst.abs_A, log10_abs_A=round(inst.log10_abs_A, 3),
                arg_A=round(inst.arg_A, 4), required_dps=inst.required_dps,
                f64_wall_margin=round(inst.f64_wall_margin_decades(W), 3))
    recs = list(found.values())
    # spread across the period range (proxy for scale diversity), like the d2 label-set
    recs.sort(key=lambda d: (d["period"], d["cx"]))
    if len(recs) > n_target:
        idx = np.linspace(0, len(recs) - 1, n_target).round().astype(int)
        recs = [recs[i] for i in sorted(set(idx.tolist()))]
    fam = "mandelbrot" if degree == 2 else f"multibrot{degree}"
    for i, d in enumerate(recs):
        d["id"] = f"d{degree}_mb{i:02d}_p{d['period']:02d}"
        d["family"] = fam
    periods = sorted(d["period"] for d in recs)
    print(f"  d{degree}: {len(recs)} nuclei kept ({n_solves} Newton solves, "
          f"{time.time()-t0:.1f}s)  periods {periods[0] if periods else '-'}.."
          f"{periods[-1] if periods else '-'}  "
          f"size [{min((d['size'] for d in recs), default=0):.1e}, "
          f"{max((d['size'] for d in recs), default=0):.1e}]")
    return recs


def stage_source():
    OUT.mkdir(parents=True, exist_ok=True)
    print("sourcing nuclei per degree (generalized z^d+c Newton):")
    for deg in DEGREES:
        recs = source_nuclei(deg)
        (OUT / f"nuclei_d{deg}.json").write_text(json.dumps(recs, indent=2))
    print(f"-> {OUT.relative_to(ROOT)}/nuclei_d*.json")


# --------------------------------------------------------------------------- #
# Field rendering — the SAME render-one --dump-field path as the d2 label-set,
# only the family carries the degree (F64Backend is already degree-parametric).
# --------------------------------------------------------------------------- #
def _dump_field(cx, cy, fw, maxiter, family, out_bin):
    # --dump-field-source f64 (fast, rayon-parallel) instead of the default `beautiful`
    # kernel. The two differ ONLY by the constant offset ln(ln B)/ln d, and LF.featurize
    # percentile-stretches every crop ((v-lo)/span), which is exactly offset-invariant —
    # verified: interior mask identical, features agree to ~1e-9, _v2_drop identical. The
    # screen sees the same features; f64 is ~35x faster (1.1s vs 39s at W=2176). Both the
    # corpus refit fields and the transfer fields use f64, so the model is unchanged too.
    cmd = [str(EXE), "render-one", "--cx", cx, "--cy", cy, "--fw", fw,
           "--family", family, "--maxiter", str(maxiter),
           "--width", str(W), "--height", str(H), "--supersample", "1",
           "--dump-field-source", "f64", "--dump-field", str(out_bin)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump-field {out_bin.name} failed: {r.stderr[-400:]}")


def _load_field(bin_path):
    meta = json.loads(bin_path.with_suffix(".json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    a = np.frombuffer(bin_path.read_bytes(), dtype="<f4")
    return a.reshape(h, w).astype(np.float64), w, h


def stage_corpus_fields():
    """Regenerate the 33 d2 label-corpus fields the model fit reads (into the path
    LS.load_field_values expects). Resumable; skips any already on disk."""
    CORPUS_FIELDS.mkdir(parents=True, exist_ok=True)
    info = HT.mb_info()                          # {minibrot_id: render dict} from store
    todo = [(mbid, rd) for mbid, rd in sorted(info.items())
            if not (CORPUS_FIELDS / f"{mbid}.bin").exists()]
    print(f"corpus fields: {len(info)} total, {len(todo)} to render (fit inputs)")
    t0 = time.time()
    for i, (mbid, rd) in enumerate(todo):
        b = CORPUS_FIELDS / f"{mbid}.bin"
        ts = time.time()
        _dump_field(rd["cx"], rd["cy"], rd["fw"], rd["maxiter"],
                    rd.get("family", "mandelbrot"), b)
        print(f"  [{i+1}/{len(todo)}] {mbid} fw={rd['fw']} mi={rd['maxiter']} "
              f"-> {time.time()-ts:.1f}s", flush=True)
    print(f"corpus fields done in {time.time()-t0:.1f}s")


def stage_fields():
    """Render every sourced multibrot/d2-control nucleus to an f64 field."""
    for deg in DEGREES:
        recs = json.loads((OUT / f"nuclei_d{deg}.json").read_text())
        fdir = FIELDS / f"d{deg}"
        fdir.mkdir(parents=True, exist_ok=True)
        todo = [r for r in recs if not (fdir / f"{r['id']}.bin").exists()]
        print(f"d{deg}: {len(recs)} fields, {len(todo)} to render")
        for i, r in enumerate(todo):
            b = fdir / f"{r['id']}.bin"
            ts = time.time()
            _dump_field(r["cx"], r["cy"], r["fw"], r["maxiter"], r["family"], b)
            print(f"  [{i+1}/{len(todo)}] {r['id']} fw={r['fw']} mi={r['maxiter']} "
                  f"-> {time.time()-ts:.1f}s", flush=True)


# --------------------------------------------------------------------------- #
# The screen — instrumented. Every gating decision below is the deployed screen's
# OWN function (LF.featurize, LF._v2_drop, clf.decision_function); only the
# per-position bookkeeping is new. `_sweep_fates` mirrors LF.dense_grid's grid
# geometry EXACTLY and its survivor set is asserted equal to dense_grid's finite-G
# set, so the instrumentation is provably faithful to the screen.
# --------------------------------------------------------------------------- #
def _clause_reasons(f):
    """Which v2 ceilings this window trips (a window can trip several)."""
    reasons = []
    if f["g_interior"] >= LF.V2_INTERIOR:
        reasons.append("interior")
    if f["g_flat"] >= LF.V2_FLAT:
        reasons.append("flat")
    if f["g_speckle"] >= LF.V2_SPECKLE:
        reasons.append("speckle")
    return reasons


# One-time proof that _sweep_fates reproduces LF.dense_grid's survivor set exactly.
# The two share featurize/_v2_drop/clf and grid geometry, so a single confirming run
# stands for all; re-featurizing every field through dense_grid too (a 2nd full pass)
# is what made the first run intractable.
_ASSERTED = {"done": False}


def _sweep_fates(field, fw, fh, scale, model):
    """Per-position fate over one field at one scale, building the SAME 2D G grid as
    LF.dense_grid in ONE featurize pass. Geometry copied verbatim from dense_grid so
    swept positions and the (gx, gy, G, (Wp, Hp)) layout are identical. Returns fate
    counts + the survivor G array + boxes + the dense_grid-shaped G (for _all_peaks)."""
    sc, clf, keys = model
    Wp = max(8, int(round(scale * fw)))
    Hp = max(8, int(round(Wp * 9 / 16)))
    if Hp >= fh or Wp >= fw:
        return None
    st = max(4, int(round(LF.DENSE_STRIDE_FRAC * Wp)))
    ys = list(range(0, fh - Hp + 1, st))
    xs = list(range(0, fw - Wp + 1, st))
    G2 = np.full((len(ys), len(xs)), np.nan)      # dense_grid-shaped (NaN = masked/small)
    n_swept = n_toosmall = n_masked = 0
    masked = defaultdict(int)                     # clause -> count (union over clauses)
    surv_feat, surv_box, surv_ij, masked_box = [], [], [], []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            n_swept += 1
            f = LF.featurize(field[y:y + Hp, x:x + Wp])
            if f is None:
                n_toosmall += 1
                continue
            if LF._v2_drop(f):
                n_masked += 1
                for cl in _clause_reasons(f):
                    masked[cl] += 1
                masked_box.append(((x + Wp / 2) / fw, (y + Hp / 2) / fh, Wp / fw, Hp / fh))
                continue
            surv_feat.append([f[k] for k in keys])
            surv_box.append(((x + Wp / 2) / fw, (y + Hp / 2) / fh, Wp / fw, Hp / fh))
            surv_ij.append((iy, ix))
    G = np.array([])
    if surv_feat:
        G = clf.decision_function(sc.transform(np.array(surv_feat)))
        for (iy, ix), gv in zip(surv_ij, G):
            G2[iy, ix] = gv
    gx = (np.array(xs) + Wp / 2) / fw
    gy = (np.array(ys) + Hp / 2) / fh
    return dict(scale=scale, Wp=Wp, Hp=Hp, n_swept=n_swept, n_toosmall=n_toosmall,
                n_masked=n_masked, masked_clause=dict(masked),
                G=G, surv_box=surv_box, masked_box=masked_box,
                G2=G2, gx=gx, gy=gy)


def screen_field(field, fw, fh, model, cutoff, assert_once=True):
    """Run the instrumented screen over one field across all scales, reusing
    HT.harvest_minibrot's G-maxima+NMS framing. A single one-time faithfulness assert
    confirms _sweep_fates reproduces LF.dense_grid's survivor set exactly."""
    sc, clf, keys = model
    agg = dict(n_swept=0, n_toosmall=0, n_masked=0, n_surv=0,
               masked_clause=defaultdict(int))
    all_G = []
    peaks = []                          # replicate HT.harvest_minibrot's peaks+NMS
    surv_boxes_by_scale = {}
    masked_boxes = []
    for s in LF.FIELD_SCALES:
        fr = _sweep_fates(field, fw, fh, s, model)
        if fr is None:
            continue
        if assert_once and not _ASSERTED["done"]:
            dg = LF.dense_grid(field, fw, fh, s, model)   # the unchanged screen
            n_finite = int(np.isfinite(dg[2]).sum())
            assert n_finite == len(fr["G"]) and np.allclose(
                np.nan_to_num(dg[2]), np.nan_to_num(fr["G2"]), atol=1e-9), (
                f"instrumentation drift vs dense_grid: {n_finite} vs {len(fr['G'])}")
            _ASSERTED["done"] = True
        Gdg = fr["G2"]
        Wp, Hp = fr["Wp"], fr["Hp"]
        gx, gy = fr["gx"], fr["gy"]
        agg["n_swept"] += fr["n_swept"]
        agg["n_toosmall"] += fr["n_toosmall"]
        agg["n_masked"] += fr["n_masked"]
        agg["n_surv"] += len(fr["G"])
        for cl, n in fr["masked_clause"].items():
            agg["masked_clause"][cl] += n
        all_G.append(fr["G"])
        surv_boxes_by_scale[s] = (fr["surv_box"], fr["G"])
        masked_boxes.extend(fr["masked_box"])
        # G-maxima framing (HT._all_peaks on the same G grid), same as the harvest
        for (iy, ix, gv) in HT._all_peaks(Gdg):
            peaks.append(dict(scale=s, cu=float(gx[ix]), cv=float(gy[iy]),
                              wu=Wp / fw, wv=Hp / fh, G=gv))
    # elliptical center-separation NMS across scales (verbatim from HT.harvest_minibrot)
    peaks.sort(key=lambda c: -c["G"])
    kept = []
    for c in peaks:
        clash = False
        for k in kept:
            du = (c["cu"] - k["cu"]) / (0.5 * (c["wu"] + k["wu"]))
            dv = (c["cv"] - k["cv"]) / (0.5 * (c["wv"] + k["wv"]))
            if du * du + dv * dv < HT.SEP * HT.SEP:
                clash = True
                break
        if not clash:
            kept.append(c)
        if len(kept) >= HT.PER_MB_CAP:
            break
    G = np.concatenate(all_G) if all_G else np.array([])
    agg["masked_clause"] = dict(agg["masked_clause"])
    return dict(agg=agg, G=G, kept=kept, cutoff=cutoff,
                surv_boxes_by_scale=surv_boxes_by_scale, masked_boxes=masked_boxes)


def _fit_model():
    """The exact deployment fit (q4_harvest_tight): refit-union labels, T2_cells, C=2.0,
    plus the label-derived tight G cutoff on the same in-sample G scale."""
    labels = R.load_labels()
    rows = R.build_dataset(labels)
    lab = [r for r in rows if r[3] in ("accept", "reject")]
    _, _, sc, clf = LF.surviving_weights(rows, TIER, C)
    keys = LF.FEATURES[TIER]
    Xl = np.array([[r[2][k] for k in keys] for r in lab])
    yl = np.array([1 if r[3] == "accept" else 0 for r in lab])
    gl = clf.decision_function(sc.transform(Xl))
    tight = HT.pick_cutoff(gl, yl, HT.TARGET_PREC, HT.MIN_N)
    print(f"deployment model: {len(lab)} labeled windows over "
          f"{len({r[1] for r in lab})} minibrots; TIGHT cutoff G>={tight['cutoff']:.3f} "
          f"(prec {tight['precision']:.2f}, {tight['n_above']} labeled above)")
    return (sc, clf, keys), tight


def _pct(x):
    return {q: (float(np.percentile(x, q)) if len(x) else None)
            for q in (0, 10, 25, 50, 75, 90, 100)}


def stage_screen():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tight = _fit_model()
    cutoff = tight["cutoff"]
    per_degree = {}
    for deg in DEGREES:
        recs = json.loads((OUT / f"nuclei_d{deg}.json").read_text())
        fdir = FIELDS / f"d{deg}"
        d_swept = d_masked = d_surv = d_toosmall = 0
        clause = defaultdict(int)
        allG, kept_all = [], []
        n_fields = 0
        per_mb = []
        fates = {}                      # id -> {kept:[...], masked_sample:[...]} for sheets
        srng = np.random.default_rng(0)
        for r in recs:
            b = fdir / f"{r['id']}.bin"
            if not b.exists():
                continue
            field, fw, fh = _load_field(b)
            res = screen_field(field, fw, fh, model, cutoff)
            a = res["agg"]
            d_swept += a["n_swept"]; d_masked += a["n_masked"]
            d_surv += a["n_surv"]; d_toosmall += a["n_toosmall"]
            for cl, n in a["masked_clause"].items():
                clause[cl] += n
            allG.append(res["G"])
            n_acc = sum(1 for c in res["kept"] if c["G"] >= cutoff)
            kept_all.extend(res["kept"])
            per_mb.append(dict(id=r["id"], period=r["period"], size=r["size"],
                               degree=deg, log10_abs_A=r.get("log10_abs_A"),
                               f64_wall_margin=r.get("f64_wall_margin"),
                               n_surv=a["n_surv"], n_masked=a["n_masked"],
                               n_kept=len(res["kept"]), n_accepted=n_acc,
                               G_max=(float(res["G"].max()) if len(res["G"]) else None)))
            # persist fate boxes for the sheets stage (so it never re-screens)
            mb = res["masked_boxes"]
            msample = ([mb[j] for j in srng.choice(len(mb), size=min(3, len(mb)),
                                                    replace=False)] if mb else [])
            fates[r["id"]] = dict(
                kept=[dict(box=[c["cu"], c["cv"], c["wu"], c["wv"]],
                           G=c["G"], scale=c["scale"]) for c in res["kept"]],
                masked_sample=[list(x) for x in msample])
            n_fields += 1
        (OUT / f"fates_d{deg}.json").write_text(json.dumps(fates))
        G = np.concatenate(allG) if allG else np.array([])
        # denominator for pass rates: positions that were featurizable (exclude too-small)
        n_eff = d_swept - d_toosmall
        n_kept = len(kept_all)
        n_acc = sum(1 for c in kept_all if c["G"] >= cutoff)
        per_degree[deg] = dict(
            n_fields=n_fields,
            n_positions_swept=d_swept, n_toosmall=d_toosmall, n_featurizable=n_eff,
            n_masked=d_masked, n_survivors=d_surv,
            coarse_ood_pass_rate=(d_surv / n_eff if n_eff else None),
            ood_mask_reject_rate=(d_masked / n_eff if n_eff else None),
            masked_clause_frac={cl: (clause[cl] / n_eff if n_eff else None)
                                for cl in ("interior", "flat", "speckle")},
            G_n=int(len(G)), G_pct=_pct(G),
            G_mean=(float(G.mean()) if len(G) else None),
            G_std=(float(G.std()) if len(G) else None),
            n_maxima_kept=n_kept, n_accepted=n_acc,
            frac_kept_accepted=(n_acc / n_kept if n_kept else None),
            per_minibrot=per_mb)
        pd = per_degree[deg]
        gm = lambda q: (float("nan") if pd["G_pct"][q] is None else pd["G_pct"][q])
        orr = pd["ood_mask_reject_rate"]
        print(f"\nd{deg}: {n_fields} fields | featurizable {n_eff} | "
              f"OOD-mask reject {(orr if orr is not None else float('nan')):.1%} | "
              f"survivors {d_surv} | G median {gm(50):.2f} "
              f"[{gm(0):.1f},{gm(100):.1f}] | "
              f"maxima {n_kept} accepted {n_acc} (>= {cutoff:.2f})")
        mc = lambda cl: (float("nan") if pd["masked_clause_frac"][cl] is None
                         else pd["masked_clause_frac"][cl])
        print(f"     mask by clause (frac of featurizable): "
              f"interior {mc('interior'):.1%}  flat {mc('flat'):.1%}  "
              f"speckle {mc('speckle'):.1%}")

    stats = dict(cutoff=cutoff, tight=tight, tier=TIER, C=C,
                 scales=LF.FIELD_SCALES,
                 v2_ceilings=dict(interior=LF.V2_INTERIOR, flat=LF.V2_FLAT,
                                  speckle=LF.V2_SPECKLE),
                 per_degree=per_degree)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"\n-> {(OUT / 'stats.json').relative_to(ROOT)}")
    return stats


# --------------------------------------------------------------------------- #
# Fate-stratified sheets. Colorize the f64 field crop with a VIVID blue/orange
# colormap (NOT twilight_shifted, which crushes multibrot fields). Per degree, one
# sheet with rows accepted / rejected / OOD-masked so the eye judges the sourcing
# independently of the screen's verdict.
# --------------------------------------------------------------------------- #
def _blue_orange():
    from matplotlib.colors import LinearSegmentedColormap
    # deep blue -> teal -> cream -> orange -> near-black-orange; vivid, high-contrast.
    return LinearSegmentedColormap.from_list("blue_orange", [
        (0.03, 0.05, 0.25), (0.05, 0.35, 0.62), (0.55, 0.82, 0.86),
        (0.98, 0.93, 0.78), (0.95, 0.55, 0.12), (0.35, 0.12, 0.02)])


def _colorize(crop, cmap):
    """f64 field crop (NaN=interior) -> RGB uint8, per-crop percentile stretch."""
    finite = np.isfinite(crop)
    if finite.sum() < 4:
        return np.zeros((*crop.shape, 3), np.uint8)
    lo, hi = np.percentile(crop[finite], [1, 99])
    span = max(hi - lo, 1e-9)
    norm = np.clip((crop - lo) / span, 0.0, 1.0)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[~finite] = (10, 10, 14)                 # interior -> near-black
    return rgb


def _crop_box(field, fw, fh, box):
    cu, cv, wu, wv = box
    x0 = int(round((cu - wu / 2) * fw)); x1 = int(round((cu + wu / 2) * fw))
    y0 = int(round((cv - wv / 2) * fh)); y1 = int(round((cv + wv / 2) * fh))
    return field[max(0, y0):y1, max(0, x0):x1]


def stage_sheets():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = json.loads((OUT / "stats.json").read_text())
    cutoff = stats["cutoff"]
    cmap = _blue_orange()
    rng = np.random.default_rng(0)
    N_COL = 6

    for deg in DEGREES:
        recs = json.loads((OUT / f"nuclei_d{deg}.json").read_text())
        fates = json.loads((OUT / f"fates_d{deg}.json").read_text())  # screen-stage boxes
        fdir = FIELDS / f"d{deg}"
        accepted, rejected, ood = [], [], []      # (crop_rgb, caption)
        for r in recs:
            b = fdir / f"{r['id']}.bin"
            if not b.exists() or r["id"] not in fates:
                continue
            field, fw, fh = _load_field(b)          # load for crops only — NO re-screen
            ft = fates[r["id"]]
            for c in ft["kept"]:
                crop = _crop_box(field, fw, fh, tuple(c["box"]))
                cap = f"{r['id']} s{c['scale']} G={c['G']:.2f}"
                (accepted if c["G"] >= cutoff else rejected).append((crop, cap))
            for box in ft["masked_sample"]:
                ood.append((_crop_box(field, fw, fh, tuple(box)), f"{r['id']} masked"))

        rows = [("accepted (G>=cutoff)", accepted),
                ("rejected (survived, G<cutoff)", rejected),
                ("OOD-masked (v2 pre-filter)", ood)]
        fig, axes = plt.subplots(3, N_COL, figsize=(2.3 * N_COL, 7.4))
        fam = "mandelbrot" if deg == 2 else f"multibrot{deg}"
        fig.suptitle(f"d{deg} ({fam}) — stage-1 screen fate  |  cutoff G>={cutoff:.2f}  "
                     f"|  vivid blue/orange field colorize", y=0.99, fontsize=11)
        for ri, (label, items) in enumerate(rows):
            rng.shuffle(items)
            for ci in range(N_COL):
                ax = axes[ri, ci]
                ax.axis("off")
                if ci == 0:
                    ax.text(-0.08, 0.5, label, rotation=90, va="center", ha="right",
                            transform=ax.transAxes, fontsize=8, weight="bold")
                if ci < len(items):
                    crop, cap = items[ci]
                    if crop.size and min(crop.shape[:2]) >= 2:
                        ax.imshow(_colorize(crop, cmap))
                        ax.set_title(cap, fontsize=6)
        fig.tight_layout(rect=[0.02, 0, 1, 0.97])
        out_png = OUT / f"sheet_d{deg}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"d{deg}: sheet -> {out_png.relative_to(ROOT)}  "
              f"(accepted {len(accepted)} / rejected {len(rejected)} / ood {len(ood)})")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["corpus-fields", "source", "fields", "screen", "sheets", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("corpus-fields", "all"):
        stage_corpus_fields()
    if args.stage in ("source", "all"):
        stage_source()
    if args.stage in ("fields", "all"):
        stage_fields()
    if args.stage in ("screen", "all"):
        stage_screen()
    if args.stage in ("sheets", "all"):
        stage_sheets()
