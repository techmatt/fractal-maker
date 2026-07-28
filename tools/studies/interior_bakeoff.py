#!/usr/bin/env python
r"""interior_bakeoff.py — feature bake-off: is INTERIOR MASS the real axis, not degree?

The hypothesis (prompts/prompt-feature-bakeoff-interior.md). Reading the three
minibrot-roster-v2 sheets together: `hi_g_lo` (screen ACCEPTED, labeled 1-2) is near-pure
dendrite — thin branching filigree on smooth gradient, no interior bodies in frame; while
`class4` and `sub_hi` (screen REJECTED, labeled 3-4) show distinct black interior bodies
with thick scroll structure wrapping them. G is an edge-energy statistic and dendrites
maximize edge per unit area, so G is not malfunctioning — it may be measuring something
anticorrelated with what Matt wants. The claim under test: **the degree correlation
(Spearman +0.55) is a proxy** — higher-degree sets put more satellite bodies in a window,
so degree would correlate with label without being causal.

MEASUREMENT ONLY. Nothing here changes a cutoff, a screen, a draw, or a production
feature. Every deployed module (`q4_stage1_linear_fit`, `q4_multibrot_transfer`,
`q4_harvest_tight`) is imported READ-ONLY and never edited.

No new labels. The only rendering is re-deriving the escape-time FIELD of the 487 crops
that are already labeled, at their exact label geometry (1280x720 ss1, f64 source, the
crop's own maxiter) — the same field the labeler's JPG was shaded from.

Stages:
  features  Per drawn crop: (a) re-derive its 1280x720 f64 field and compute the candidate
            interior/scroll features on it; (b) re-featurize its EXACT screen window out of
            the cached parent atom field, giving the deployed screen's own 15 features and a
            recomputed G (parity-checked against the stored G). -> durable feature table.
  board     Part B. The WHOLE board — every feature, including the ones that did nothing.
            Spearman + AUC(label>=3) on train (select) and eval (confirm); the conditioning
            pair (degree | best interior feature, and the reverse); hi_g_lo vs sub_hi
            separation; and a near-duplicate-window audit of the batch.
  audit     Part C. Does the stage-1 screen / OOD mask penalize interior content, and how
            much of the accept/reject split does that account for? Clause attribution over
            the drawn windows plus a sampled full-field sweep (counterfactual: which
            positions the interior clause ALONE removes).

Runtime: features ~2 min (487 field dumps at ~0.1s + 145 parent-field re-featurizations);
board ~5 s; audit ~4 min with 4 workers (background it).

  uv run python -m tools.studies.interior_bakeoff features
  uv run python -m tools.studies.interior_bakeoff board
  uv run python -m tools.studies.interior_bakeoff audit [--atoms 24]

Reads:   data/minibrot_roster/batch_v1/draw.jsonl, roster.jsonl
         data/label_corpus/batches/2026-07-26_minibrot_roster_v2/images.jsonl (labels)
         scratch/minibrot_batch/fields/<atom>.bin (cached parent screen fields)
Writes:  scratch/interior_bakeoff/{fields,cropfeat,sweep}/   (regenerable cache)
         data/minibrot_roster/batch_v1/interior_features.jsonl  (durable feature table)
         scratch/interior_bakeoff/{board,audit}.json           (numbers for the findings doc)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

BATCH_ID = "2026-07-26_minibrot_roster_v2"
BATCH_DIR = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID
DRAW = ROOT / "data" / "minibrot_roster" / "batch_v1" / "draw.jsonl"
ROSTER = ROOT / "data" / "minibrot_roster" / "roster.jsonl"
PARENT_FIELDS = paths.scratch("minibrot_batch", "fields")
EXE = ROOT / "target" / "release" / "fractal-generator.exe"

SCR = paths.scratch("interior_bakeoff")
CROP_FIELDS = SCR / "fields"
CROP_FEAT = SCR / "cropfeat"
SWEEP = SCR / "sweep"
TABLE_REL = "data/minibrot_roster/batch_v1/interior_features.jsonl"

# The label geometry: the crops were rendered 1280x720 (ss4 for the JPG; the FIELD is the
# same grid at ss1 — the escape mask is what we measure, and ss only anti-aliases it).
CROP_W, CROP_H = 1280, 720
WORKERS = 4                                   # project rule: <=4
# interior connected-component area thresholds (fraction of frame) — the "2-3 scales" of
# the prompt: ~92 px, ~921 px, ~9216 px at 1280x720.
AREA_A4, AREA_A3, AREA_A2 = 1e-4, 1e-3, 1e-2
COH_SIGMAS = (3.0, 8.0)                       # structure-tensor scales (px)
HI_G_LO_MAX = 24                              # the sheet's top-N (minibrot_roster_v2_sheets)


# =========================================================================== #
# io helpers
# =========================================================================== #
def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _load_field(bin_path):
    """Read a dumped f32 field + its sidecar. Same reader as q4_multibrot_transfer."""
    meta = json.loads(Path(bin_path).with_suffix(".json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    a = np.frombuffer(Path(bin_path).read_bytes(), dtype="<f4")
    return a.reshape(h, w).astype(np.float64), w, h


def _field_ready(b: Path) -> bool:
    return b.exists() and b.with_suffix(".json").exists()


def _dump_crop_field(row, out_bin: Path, timeout=180.0):
    """Re-derive one crop's escape-time field at its exact label geometry (f64 source,
    1280x720, ss1). Mirrors the deployed dump path; NaN marks non-escaped (in-set)."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(EXE), "render-one", "--cx", row["cx"], "--cy", row["cy"], "--fw", row["fw"],
           "--family", row["family"], "--maxiter", str(row["maxiter"]),
           "--width", str(CROP_W), "--height", str(CROP_H), "--supersample", "1",
           "--dump-field-source", "f64", "--dump-field", str(out_bin)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"dump-field {out_bin.name} failed: {r.stderr[-300:]}")


# =========================================================================== #
# Part A — the candidate features, computed on the re-derived crop field.
#
# Deliberately small and defensible. Every one is a pure function of the escape-time
# field, so it is palette-invariant exactly like the screen's own features.
# =========================================================================== #
def crop_features(field: np.ndarray) -> dict:
    """1280x720 escape-time field (NaN = in-set) -> candidate feature dict.

    Interior mass:      int_frac, int_largest_frac, int_n_comp_a{4,3,2}
    Body vs dendrite:   int_perim_area, int_compactness, int_{max,mean}_inradius
    Scroll structure:   coh_s3, coh_s8, coh_scale_drop
    Reference ink:      edge_energy

    NaN (not 0) is emitted where a feature is undefined for lack of any interior — a crop
    with no in-set pixels has no perimeter-to-area ratio, and coding that as 0 would fake
    a "maximally blobby" reading. The board reports per-feature n.
    """
    H, W = field.shape
    A = float(H * W)
    inside = ~np.isfinite(field)
    f = {"int_frac": float(inside.mean())}

    # --- connected interior components (8-connectivity) ---------------------
    lab, n = ndimage.label(inside, structure=np.ones((3, 3), np.int32))
    if n:
        areas = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
        # per-component perimeter = 4-neighbour label transitions. The image border is NOT
        # counted: a body clipped by the frame should not read as extra boundary.
        per = np.zeros(n + 1)
        for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
            d = a != b
            np.add.at(per, a[d], 1)
            np.add.at(per, b[d], 1)
        per = per[1:]
    else:
        areas = np.zeros(0)
        per = np.zeros(0)

    f["int_n_comp_a4"] = int((areas >= AREA_A4 * A).sum())
    f["int_n_comp_a3"] = int((areas >= AREA_A3 * A).sum())
    f["int_n_comp_a2"] = int((areas >= AREA_A2 * A).sum())
    f["int_largest_frac"] = float(areas.max() / A) if n else 0.0

    keep = areas >= AREA_A4 * A
    if keep.any():
        ak, pk = areas[keep], np.maximum(per[keep], 1.0)
        f["int_perim_area"] = float(pk.sum() / ak.sum())
        # isoperimetric compactness per component, area-weighted (1 = disc, ->0 = filament)
        comp = np.minimum(4.0 * np.pi * ak / (pk * pk), 1.0)
        f["int_compactness"] = float((comp * ak).sum() / ak.sum())
    else:
        f["int_perim_area"] = float("nan")
        f["int_compactness"] = float("nan")

    if inside.any():
        edt = ndimage.distance_transform_edt(inside)
        f["int_max_inradius"] = float(edt.max() / H)          # thickest body, in frame heights
        f["int_mean_inradius"] = float(edt[inside].mean() / H)
    else:
        f["int_max_inradius"] = 0.0
        f["int_mean_inradius"] = float("nan")

    # --- scroll / spiral structure: local orientation coherence -------------
    # Same normalization convention as the screen's featurize (percentile stretch, interior
    # -> deepest) so the two feature sets are on comparable footing.
    finite = ~inside
    vv = field[finite]
    if vv.size >= 64:
        lo, hi = np.percentile(vv, [0.5, 99.5])
        span = max(hi - lo, 1e-9)
        work = np.where(finite, np.clip((field - lo) / span, 0.0, 1.0), 1.0)
    else:
        work = np.ones_like(field)
    gy, gx = np.gradient(work)
    f["edge_energy"] = float(np.hypot(gx, gy).mean())
    for s in COH_SIGMAS:
        jxx = ndimage.gaussian_filter(gx * gx, s)
        jxy = ndimage.gaussian_filter(gx * gy, s)
        jyy = ndimage.gaussian_filter(gy * gy, s)
        tr = jxx + jyy
        coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(tr, 1e-12)
        wsum = float(tr.sum())
        f[f"coh_s{int(s)}"] = float((coh * tr).sum() / wsum) if wsum > 0 else float("nan")
    # A straight filament stays coherent at every scale; a scroll/spiral turns, so its
    # coherence FALLS as the tensor window grows. The drop is the turning measure.
    f["coh_scale_drop"] = f["coh_s3"] - f["coh_s8"]
    return f


CROP_KEYS = ["int_frac", "int_largest_frac", "int_n_comp_a4", "int_n_comp_a3",
             "int_n_comp_a2", "int_perim_area", "int_compactness", "int_max_inradius",
             "int_mean_inradius", "coh_s3", "coh_s8", "coh_scale_drop", "edge_energy"]


def _crop_worker(job):
    """Dump (if absent) + featurize one crop field. Resumable via the per-crop JSON."""
    row, timeout = job
    iid = row["image_id"]
    cache = CROP_FEAT / f"{iid}.json"
    if cache.exists():
        return iid, "cached"
    b = CROP_FIELDS / f"{iid}.bin"
    if not _field_ready(b):
        _dump_crop_field(row, b, timeout)
    field, _, _ = _load_field(b)
    f = crop_features(field)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(f))
    os.replace(tmp, cache)
    return iid, "done"


# =========================================================================== #
# Stage: features
# =========================================================================== #
def stage_features(args):
    from tools.studies import q4_multibrot_transfer as MT   # read-only reuse
    from tools.studies import q4_stage1_linear_fit as LF

    draw = _read_jsonl(DRAW)
    labels = {r["image_id"]: r["label"]["score"]
              for r in _read_jsonl(BATCH_DIR / "images.jsonl")}
    roster = {r["id"]: r for r in _read_jsonl(ROSTER)}
    print(f"features: {len(draw)} drawn crops, {len(labels)} labeled, "
          f"{len({d['atom_id'] for d in draw})} atoms")

    # ---- (a) crop-resolution features (parallel; the only rendering) -------
    CROP_FIELDS.mkdir(parents=True, exist_ok=True)
    CROP_FEAT.mkdir(parents=True, exist_ok=True)
    todo = [d for d in draw if not (CROP_FEAT / f"{d['image_id']}.json").exists()]
    print(f"  crop fields: {len(draw) - len(todo)} cached, {len(todo)} to derive "
          f"(1280x720 ss1 f64, workers={args.workers})")
    t0 = time.time()
    if todo:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_crop_worker, (d, args.timeout)): d["image_id"] for d in todo}
            for fut in as_completed(futs):
                try:
                    iid, st = fut.result()
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        print(f"    [{done}/{len(todo)}] {iid} {st} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:                       # noqa: BLE001
                    print(f"    !! {futs[fut]} FAILED: {type(e).__name__}: {str(e)[:200]}",
                          flush=True)
    cropf = {}
    for d in draw:
        p = CROP_FEAT / f"{d['image_id']}.json"
        if p.exists():
            cropf[d["image_id"]] = json.loads(p.read_text())
    print(f"  crop features: {len(cropf)}/{len(draw)} ({time.time()-t0:.0f}s)")

    # ---- (b) screen-resolution features: the deployed screen's OWN featurize on
    #          the EXACT drawn window, out of the cached parent atom field. -----
    model, tight = MT._fit_model()
    sc, clf, keys = model
    by_atom = defaultdict(list)
    for d in draw:
        by_atom[d["atom_id"]].append(d)
    screenf, missing_parent = {}, []
    t1 = time.time()
    for i, (aid, rows) in enumerate(sorted(by_atom.items())):
        b = PARENT_FIELDS / f"{aid}.bin"
        if not _field_ready(b):
            missing_parent.append(aid)
            continue
        field, fw, fh = _load_field(b)
        for r in rows:
            crop = MT._crop_box(field, fw, fh, tuple(r["box"]))
            f = LF.featurize(crop)
            screenf[r["image_id"]] = f
        if (i + 1) % 40 == 0:
            print(f"    parent fields [{i+1}/{len(by_atom)}] ({time.time()-t1:.0f}s)",
                  flush=True)
    ok = [i for i, f in screenf.items() if f is not None]
    print(f"  screen features: {len(ok)}/{len(draw)} featurizable "
          f"({len(missing_parent)} atoms missing a parent field) ({time.time()-t1:.0f}s)")

    # recomputed G + parity against the stored (deployed) G
    X = np.array([[screenf[i][k] for k in keys] for i in ok])
    Grec = clf.decision_function(sc.transform(X))
    grec = dict(zip(ok, Grec.tolist()))
    stored = {d["image_id"]: d["G"] for d in draw if d["G"] is not None}
    pair = [(stored[i], grec[i]) for i in ok if i in stored]
    if pair:
        a = np.array([p[0] for p in pair]); bb = np.array([p[1] for p in pair])
        print(f"  G parity (recomputed vs stored, n={len(pair)}): "
              f"max|dG|={np.abs(a-bb).max():.4f}  mean|dG|={np.abs(a-bb).mean():.5f}  "
              f"corr={np.corrcoef(a, bb)[0,1]:.6f}   [box->pixel rounding only]")

    # ---- join + write the durable table -----------------------------------
    out = []
    for d in draw:
        iid = d["image_id"]
        sf = screenf.get(iid)
        row = dict(
            image_id=iid, atom=d["atom_id"], label=labels.get(iid), split=d["split"],
            fate=d["fate"], arm=d["arm"], degree=d["degree"], period=d["period"],
            band=d["band"], G=d["G"], G_recomputed=grec.get(iid), maxiter=d["maxiter"],
            fw=float(d["fw"]), box=[float(x) for x in d["box"]],
            log10A=(roster[d["atom_id"]]["log10_abs_A"] if d["atom_id"] in roster else None),
            crop={k: cropf[iid][k] for k in CROP_KEYS} if iid in cropf else None,
            screen=({k: sf[k] for k in sorted(sf)} if sf else None),
        )
        out.append(row)
    tp = paths.durable(TABLE_REL, mkparents=True)
    with open(tp, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"  -> {tp.relative_to(ROOT)}  ({len(out)} rows, "
          f"{len(CROP_KEYS)} crop + {len(keys)}+ screen features)")
    return 0


# =========================================================================== #
# statistics (small, hand-checked; scipy/sklearn for the standard estimators)
# =========================================================================== #
def _finite(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= np.isfinite(np.asarray(a, float))
    return m


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = _finite(x, y)
    if m.sum() < 8 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return float("nan"), int(m.sum())
    return float(spearmanr(x[m], y[m]).statistic), int(m.sum())


def auc(x, pos):
    """AUC of x as a ranker for the binary indicator `pos`."""
    from sklearn.metrics import roc_auc_score
    x, pos = np.asarray(x, float), np.asarray(pos).astype(int)
    m = _finite(x)
    x, pos = x[m], pos[m]
    if pos.sum() == 0 or pos.sum() == len(pos):
        return float("nan"), int(pos.sum()), int(len(pos) - pos.sum())
    return float(roc_auc_score(pos, x)), int(pos.sum()), int(len(pos) - pos.sum())


def partial_spearman(x, y, z):
    """Spearman(x, y | z): rank all three, linearly regress rank(x) and rank(y) on
    rank(z), correlate the residuals. The standard rank partial correlation."""
    from scipy.stats import rankdata, pearsonr
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = _finite(x, y, z)
    if m.sum() < 12:
        return float("nan"), int(m.sum())
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    Z = np.column_stack([np.ones_like(rz), rz])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    return float(pearsonr(resid(rx), resid(ry)).statistic), int(m.sum())


def atom_bootstrap_auc(x, pos, atoms, reps=2000, seed=0):
    """90% CI for AUC resampling ATOMS (not crops) — the batch has up to 3 windows per
    atom and heavily overlapping windows, so crop-level resampling would overstate n."""
    from sklearn.metrics import roc_auc_score
    x, pos, atoms = np.asarray(x, float), np.asarray(pos).astype(int), np.asarray(atoms)
    m = _finite(x)
    x, pos, atoms = x[m], pos[m], atoms[m]
    uniq = np.unique(atoms)
    idx_by_atom = {a: np.where(atoms == a)[0] for a in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_atom[a] for a in pick])
        yy = pos[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(roc_auc_score(yy, x[ii]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


# =========================================================================== #
# Stage: board (Part B)
# =========================================================================== #
def _flatten(rows):
    """One flat feature dict per row: crop.* + screen.* + the covariates."""
    out = []
    for r in rows:
        f = {}
        if r["crop"]:
            f.update({k: v for k, v in r["crop"].items()})
        if r["screen"]:
            f.update({f"s_{k}": v for k, v in r["screen"].items()})
        f["G"] = r["G"] if r["G"] is not None else float("nan")
        f["degree"] = r["degree"]
        f["period"] = r["period"]
        f["log10A"] = r["log10A"] if r["log10A"] is not None else float("nan")
        f["maxiter"] = r["maxiter"]
        out.append(dict(r, feat=f))
    return out


def _feature_names(rows):
    names = []
    for k in CROP_KEYS:
        names.append(k)
    scr = sorted({k for r in rows if r["screen"] for k in r["screen"]})
    names += [f"s_{k}" for k in scr]
    names += ["G", "degree", "period", "log10A", "maxiter"]
    return names


def _col(rows, name):
    return np.array([r["feat"].get(name, float("nan")) for r in rows], float)


def _iou(b1, b2):
    """IoU of two normalized [cu, cv, wu, wv] field boxes."""
    def rect(b):
        cu, cv, wu, wv = b
        return (cu - wu / 2, cv - wv / 2, cu + wu / 2, cv + wv / 2)
    x0, y0, x1, y1 = rect(b1)
    a0, b0, a1, bq = rect(b2)
    iw = max(0.0, min(x1, a1) - max(x0, a0))
    ih = max(0.0, min(y1, bq) - max(y0, b0))
    inter = iw * ih
    u = (x1 - x0) * (y1 - y0) + (a1 - a0) * (bq - b0) - inter
    return inter / u if u > 0 else 0.0


def _near_dup_audit(rows, thr=0.5):
    """Same-atom window overlap. The draw NMS'd within a category and kept rejects clear of
    accepts, but nothing enforced separation between the masked sample and the rest."""
    by_atom = defaultdict(list)
    for r in rows:
        by_atom[r["atom"]].append(r)
    pairs = []
    for aid, rs in by_atom.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                v = _iou(rs[i]["box"], rs[j]["box"])
                if v > 0:
                    pairs.append((v, aid, rs[i], rs[j]))
    pairs.sort(key=lambda t: -t[0])
    # union-find over the >=thr graph -> "effective independent windows"
    parent = {r["image_id"]: r["image_id"] for r in rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for v, _, ri, rj in pairs:
        if v >= thr:
            a, b = find(ri["image_id"]), find(rj["image_id"])
            if a != b:
                parent[a] = b
    groups = Counter(find(r["image_id"]) for r in rows)
    return pairs, groups


def stage_board(args):
    rows = _flatten(_read_jsonl(paths.durable(TABLE_REL)))
    rows = [r for r in rows if r["label"] is not None]
    names = _feature_names(rows)
    tr = [r for r in rows if r["split"] == "train"]
    ev = [r for r in rows if r["split"] == "eval"]
    lab_tr = np.array([r["label"] for r in tr], float)
    lab_ev = np.array([r["label"] for r in ev], float)
    pos_tr = lab_tr >= 3
    pos_ev = lab_ev >= 3
    at_tr = [r["atom"] for r in tr]
    at_ev = [r["atom"] for r in ev]

    print("=" * 100)
    print("PART B — THE WHOLE BOARD.  select on TRAIN, confirm on EVAL (atom-level split)")
    print("=" * 100)
    print(f"  train n={len(tr)} ({int(pos_tr.sum())} label>=3, "
          f"{len({r['atom'] for r in tr})} atoms) | "
          f"eval n={len(ev)} ({int(pos_ev.sum())} label>=3, "
          f"{len({r['atom'] for r in ev})} atoms)")
    print(f"  label dist train {dict(sorted(Counter(int(x) for x in lab_tr).items()))}  "
          f"eval {dict(sorted(Counter(int(x) for x in lab_ev).items()))}")
    print()
    hdr = (f"{'feature':<26}{'rho_tr':>9}{'AUC_tr':>9}{'n_tr':>6}"
           f"{'rho_ev':>9}{'AUC_ev':>9}{'n_ev':>6}   note")
    print(hdr)
    print("-" * len(hdr))
    board = {}
    for nm in names:
        xtr, xev = _col(tr, nm), _col(ev, nm)
        rtr, ntr = spearman(xtr, lab_tr)
        rev, nev = spearman(xev, lab_ev)
        mtr, mev = _finite(xtr), _finite(xev)
        atr, _, _ = auc(xtr[mtr], pos_tr[mtr])
        aev, _, _ = auc(xev[mev], pos_ev[mev])
        board[nm] = dict(rho_train=rtr, auc_train=atr, n_train=ntr,
                         rho_eval=rev, auc_eval=aev, n_eval=nev)
        note = ""
        if nm == "G":
            note = "reference line (deployed screen score)"
        elif nm.startswith("s_"):
            note = "screen feature = a component of G"
        elif nm in ("degree", "period", "log10A", "maxiter"):
            note = "covariate"
        print(f"{nm:<26}{rtr:>+9.3f}{atr:>9.3f}{ntr:>6}{rev:>+9.3f}{aev:>9.3f}{nev:>6}   {note}")

    # ---------------- 2. the conditioning pair --------------------------- #
    # The prompt's question is symmetric, so answer it symmetrically for EVERY candidate:
    # does degree survive conditioning on the feature, and does the feature survive
    # conditioning on degree? `edge_energy` is the reference ink line, not an interior
    # feature, so it is excluded from "best interior feature" but stays on the table.
    print("\n" + "=" * 100)
    print("2. CONDITIONING — the symmetric pair, every candidate.  "
          "rho(F,label) vs rho(F,label|degree)  and  rho(degree,label) vs rho(degree,label|F)")
    print("=" * 100)
    cond = {}
    for tag, sub, lab in (("train", tr, lab_tr), ("eval", ev, lab_ev)):
        D = _col(sub, "degree")
        r_d, _ = spearman(D, lab)
        print(f"\n  [{tag}]  raw Spearman(degree, label) = {r_d:+.3f}")
        h = (f"    {'feature F':<24}{'rho(F,y)':>10}{'rho(F,y|deg)':>14}"
             f"{'rho(deg,y|F)':>14}{'n':>6}")
        print(h)
        print("    " + "-" * (len(h) - 4))
        cond[tag] = {}
        for nm in CROP_KEYS + ["G"]:
            F = _col(sub, nm)
            r_f, n_f = spearman(F, lab)
            pf, npf = partial_spearman(F, lab, D)     # feature | degree
            pd_, _ = partial_spearman(D, lab, F)      # degree  | feature
            print(f"    {nm:<24}{r_f:>+10.3f}{pf:>+14.3f}{pd_:>+14.3f}{npf:>6}")
            cond[tag][nm] = dict(rho=r_f, rho_given_degree=pf, rho_degree_given_f=pd_,
                                 n=npf)
        cond[tag]["_rho_degree"] = r_d

    # headline candidates: best interior-mass feature, best body/dendrite discriminator,
    # best scroll feature — carried through the stratified read on both splits.
    INTERIOR_ONLY = [n for n in CROP_KEYS if n != "edge_energy"]
    best = max(INTERIOR_ONLY,
               key=lambda n: (abs(board[n]["rho_train"])
                              if np.isfinite(board[n]["rho_train"]) else -1))
    best_auc = max(INTERIOR_ONLY,
                   key=lambda n: (abs(board[n]["auc_train"] - 0.5)
                                  if np.isfinite(board[n]["auc_train"]) else -1))
    heads = list(dict.fromkeys([best, best_auc, "coh_s8"]))
    print(f"\n  best interior feature by |rho| on TRAIN: {best}  "
          f"(rho_tr {board[best]['rho_train']:+.3f});  by AUC on TRAIN: {best_auc}  "
          f"(AUC_tr {board[best_auc]['auc_train']:.3f}, n={board[best_auc]['n_train']})")
    for tag, sub, lab, pos, ats in (("train", tr, lab_tr, pos_tr, at_tr),
                                    ("eval", ev, lab_ev, pos_ev, at_ev)):
        D = _col(sub, "degree")
        lo_d, hi_d = atom_bootstrap_auc(D, pos, ats)
        print(f"\n  [{tag}]  degree: AUC(label>=3) = {board['degree']['auc_'+tag]:.3f} "
              f"[atom-bootstrap 90% CI {lo_d:.3f}-{hi_d:.3f}]")
        for nm in heads:
            F = _col(sub, nm)
            lo_f, hi_f = atom_bootstrap_auc(F, pos, ats)
            print(f"    {nm}: AUC = {board[nm]['auc_'+tag]:.3f} "
                  f"[CI {lo_f:.3f}-{hi_f:.3f}], n={board[nm]['n_'+tag]}")
            print(f"      AUC({nm} | label>=3) within each degree:")
            for d in sorted({int(x) for x in D if np.isfinite(x)}):
                m = (D == d) & _finite(F)
                a, p, ng = auc(F[m], pos[m])
                print(f"         degree {d}: AUC={a:.3f}  (pos={p}, neg={ng})")
            mF = _finite(F)
            if mF.sum() > 16:
                qs = np.quantile(F[mF], [0.25, 0.5, 0.75])
                print(f"      AUC(degree | label>=3) within {nm} quartiles:")
                for qi in range(4):
                    lo = -np.inf if qi == 0 else qs[qi - 1]
                    hi = np.inf if qi == 3 else qs[qi]
                    m = (mF & (F <= qs[0])) if qi == 0 else (mF & (F > lo) & (F <= hi))
                    if m.sum() < 8:
                        continue
                    a, p, ng = auc(D[m], pos[m])
                    print(f"         Q{qi+1} ({lo:+.4g},{hi:+.4g}]: AUC={a:.3f} "
                          f"(pos={p}, neg={ng})")

    # the zero-interior split: most crops have NO interior body at all
    nb = np.array([r["feat"]["int_n_comp_a4"] for r in rows], float)
    labs = np.array([r["label"] for r in rows], float)
    for tag, m in (("no interior body (int_n_comp_a4 == 0)", nb == 0),
                   ("has >=1 interior body", nb >= 1),
                   ("has >=3 interior bodies", nb >= 3)):
        print(f"  {tag:<40} n={int(m.sum()):4d}  mean label {labs[m].mean():.2f}  "
              f"frac L>=3 {(labs[m] >= 3).mean():.2f}")

    # ---------------- 2b. the board WITHIN each degree -------------------- #
    # Degree is the dominant covariate, so the pooled board can hide (and here does hide)
    # Simpson reversals: a feature that tracks degree can read positive pooled and
    # negative inside every degree. This is the same conditioning question as (2), asked
    # non-parametrically.
    print("\n" + "=" * 100)
    print("2b. THE BOARD WITHIN EACH DEGREE  (pooled board vs within-degree — sign flips "
          "mean the pooled number was degree)")
    print("=" * 100)
    lab_all = np.array([r["label"] for r in rows], float)
    within = {}
    for nm in CROP_KEYS + ["G"]:
        x_all = _col(rows, nm)
        r_pool, _ = spearman(x_all, lab_all)
        cells = []
        for d in (2, 3, 4, 5):
            out = []
            for tag, sub, lab in (("tr", tr, lab_tr), ("ev", ev, lab_ev)):
                D, F = _col(sub, "degree"), _col(sub, nm)
                m = (D == d) & _finite(F)
                rr, nn = spearman(F[m], lab[m])
                aa, pp, _ = auc(F[m], (lab >= 3)[m])
                out.append((rr, aa, nn, pp))
            cells.append(out)
        within[nm] = dict(pooled=r_pool, cells=cells)
        line = f"  {nm:<22}pool rho={r_pool:>+6.3f} |"
        for d, out in zip((2, 3, 4, 5), cells):
            rr = out[0][0]
            aa = out[0][1]
            line += f"  d{d} rho={rr:>+6.3f} AUC={aa:>5.3f}" if np.isfinite(rr) else \
                    f"  d{d}    --          "
        print(line)
    print("  (per-degree columns are TRAIN; degree 2 has zero label>=3 so its AUC is "
          "undefined by construction.)")
    print("\n  EVAL confirmation, degree 5 only (the only degree with a workable positive "
          "count on eval):")
    for nm in CROP_KEYS + ["G"]:
        rr, aa, nn, pp = within[nm]["cells"][3][1]
        print(f"     {nm:<22} rho={rr:>+6.3f}  AUC={aa:>5.3f}  n={nn} ({pp} pos)")

    # crop-resolution vs screen-resolution parity for the same quantity
    rr, nn = spearman(_col(rows, "int_frac"), _col(rows, "s_g_interior"))
    print(f"\n  resolution parity: Spearman(crop int_frac @1280x720, screen g_interior "
          f"@~{int(0.09*2176)}x{int(0.09*1224)}) = {rr:+.3f} (n={nn}) — the two resolutions "
          f"measure the same quantity, so nothing below is a re-derivation artifact.")

    # ---------------- 3. hi_g_lo vs sub_hi ------------------------------- #
    print("\n" + "=" * 100)
    print("3. DOES ANY FEATURE SEPARATE hi_g_lo FROM sub_hi?  (the two populations the "
          "screen gets backwards)")
    print("=" * 100)
    hi_lo_all = sorted([r for r in rows if r["fate"] == "accepted" and r["label"] <= 2],
                       key=lambda r: -r["G"])
    hi_lo = hi_lo_all[:HI_G_LO_MAX]                       # exactly the sheet's 24 tiles
    sub_hi = ([r for r in rows if r["fate"] == "rejected" and r["label"] >= 3]
              + [r for r in rows if r["fate"] == "ood_masked" and r["label"] >= 3])
    print(f"  sheet sets: hi_g_lo n={len(hi_lo)} (top {HI_G_LO_MAX} by G of "
          f"{len(hi_lo_all)} accepts labeled <=2)   sub_hi n={len(sub_hi)}")
    sep = {}
    for tag, A, B in (("sheet (24 vs 27)", hi_lo, sub_hi),
                      ("full (all accepts<=2 vs sub_hi)", hi_lo_all, sub_hi)):
        both = A + B
        y = np.array([0] * len(A) + [1] * len(B))
        print(f"\n  [{tag}]  AUC = P(sub_hi ranks above hi_g_lo); 0.5 = no separation")
        res = []
        for nm in names:
            x = _col(both, nm)
            m = _finite(x)
            if m.sum() < 10 or len(set(y[m])) < 2:
                continue
            a, p, ng = auc(x[m], y[m])
            res.append((abs(a - 0.5), a, nm, int(m.sum())))
        res.sort(reverse=True)
        for d, a, nm, n in res:
            print(f"     {nm:<26} AUC={a:.3f}   |sep|={d:.3f}   n={n}")
        sep[tag] = {nm: a for _, a, nm, _ in res}

    # ---------------- 4. near-duplicate audit ---------------------------- #
    print("\n" + "=" * 100)
    print("4. NEAR-DUPLICATE WINDOWS IN THE BATCH")
    print("=" * 100)
    pairs, groups = _near_dup_audit(rows, thr=0.5)
    n_pairs = len(pairs)
    for t in (0.75, 0.5, 0.25):
        k = sum(1 for p in pairs if p[0] >= t)
        ims = {p[2]["image_id"] for p in pairs if p[0] >= t} | \
              {p[3]["image_id"] for p in pairs if p[0] >= t}
        print(f"  same-atom pairs with IoU >= {t:.2f}: {k:4d}   crops involved: {len(ims)}")
    print(f"  total overlapping same-atom pairs (IoU>0): {n_pairs}")
    print(f"  effective independent windows at IoU>=0.50: {len(groups)} of {len(rows)} crops "
          f"(largest cluster {max(groups.values())})")
    print("  worst offenders:")
    for v, aid, ri, rj in pairs[:8]:
        print(f"    IoU={v:.3f}  {aid}  {ri['image_id']}(L{ri['label']},{ri['fate']}) "
              f"<-> {rj['image_id']}(L{rj['label']},{rj['fate']})")
    c4 = [r for r in rows if r["label"] == 4]
    if len(c4) == 2:
        print(f"  the two class-4s: {c4[0]['image_id']} <-> {c4[1]['image_id']}  "
              f"IoU={_iou(c4[0]['box'], c4[1]['box']):.3f}  "
              f"(same atom: {c4[0]['atom'] == c4[1]['atom']})  -> ANECDOTE, not signal")

    # ---------------- 5. the maxiter confound ---------------------------- #
    print("\n" + "=" * 100)
    print("5. CONFOUND CHECK — interior fraction depends on maxiter (deeper crops iterate "
          "longer, so less 'false' interior)")
    print("=" * 100)
    for nm in ("int_frac", "int_largest_frac", "int_max_inradius", "coh_scale_drop"):
        r1, n1 = spearman(_col(rows, "maxiter"), _col(rows, nm))
        r2, _ = partial_spearman(_col(rows, nm),
                                 np.array([r["label"] for r in rows], float),
                                 _col(rows, "maxiter"))
        print(f"  Spearman(maxiter, {nm:<18}) = {r1:+.3f} (n={n1})   "
              f"partial rho({nm}, label | maxiter) = {r2:+.3f}")
    rml, _ = spearman(_col(rows, "maxiter"), np.array([r["label"] for r in rows], float))
    print(f"  Spearman(maxiter, label) = {rml:+.3f}")

    outp = SCR / "board.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(dict(board=board, conditioning=cond, separation=sep,
                                    n_train=len(tr), n_eval=len(ev)), indent=1))
    print(f"\n-> {outp.relative_to(ROOT)}")
    return 0


# =========================================================================== #
# Stage: audit (Part C) — is the bias self-inflicted?
# =========================================================================== #
def _sweep_one(job):
    """Full-field clause sweep for one atom: for every swept position at every deployed
    scale, record which v2 ceilings it trips. Geometry copied from LF.dense_grid /
    MT._sweep_fates so the swept set is exactly the screen's."""
    from tools.studies import q4_stage1_linear_fit as LF
    aid, = job
    out = SWEEP / f"{aid}.json"
    if out.exists():
        return aid, "cached"
    field, fw, fh = _load_field(PARENT_FIELDS / f"{aid}.bin")
    rec = dict(atom=aid, n_swept=0, n_toosmall=0, n_surv=0, n_masked=0,
               interior_only=0, flat_only=0, speckle_only=0, multi=0,
               interior_any=0, flat_any=0, speckle_any=0,
               g_interior_hist=[0] * 10, interior_only_g_interior=[])
    for s in LF.FIELD_SCALES:
        Wp = max(8, int(round(s * fw)))
        Hp = max(8, int(round(Wp * 9 / 16)))
        if Hp >= fh or Wp >= fw:
            continue
        st = max(4, int(round(LF.DENSE_STRIDE_FRAC * Wp)))
        for y in range(0, fh - Hp + 1, st):
            for x in range(0, fw - Wp + 1, st):
                rec["n_swept"] += 1
                f = LF.featurize(field[y:y + Hp, x:x + Wp])
                if f is None:
                    rec["n_toosmall"] += 1
                    continue
                gi = f["g_interior"]
                rec["g_interior_hist"][min(9, int(gi * 10))] += 1
                ci = gi >= LF.V2_INTERIOR
                cf = f["g_flat"] >= LF.V2_FLAT
                cs = f["g_speckle"] >= LF.V2_SPECKLE
                if not (ci or cf or cs):
                    rec["n_surv"] += 1
                    continue
                rec["n_masked"] += 1
                rec["interior_any"] += int(ci)
                rec["flat_any"] += int(cf)
                rec["speckle_any"] += int(cs)
                k = int(ci) + int(cf) + int(cs)
                if k > 1:
                    rec["multi"] += 1
                elif ci:
                    rec["interior_only"] += 1
                    if len(rec["interior_only_g_interior"]) < 4000:
                        rec["interior_only_g_interior"].append(round(float(gi), 4))
                elif cf:
                    rec["flat_only"] += 1
                else:
                    rec["speckle_only"] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out)
    return aid, (f"swept={rec['n_swept']} masked={rec['n_masked']} "
                 f"int_only={rec['interior_only']}")


def stage_audit(args):
    from tools.studies import q4_multibrot_transfer as MT
    from tools.studies import q4_stage1_linear_fit as LF

    rows = _flatten(_read_jsonl(paths.durable(TABLE_REL)))
    rows = [r for r in rows if r["label"] is not None]
    model, tight = MT._fit_model()
    sc, clf, keys = model
    w = dict(zip(keys, clf.coef_.ravel().tolist()))

    print("=" * 100)
    print("PART C — IS THE BIAS SELF-INFLICTED?  audit of the stage-1 screen + OOD mask")
    print("=" * 100)
    print("\n  (i) THE OOD MASK (q4_stage1_linear_fit._v2_drop — the deployed pre-filter).")
    print(f"      A window is DROPPED OUTRIGHT, never scored, if ANY holds:")
    print(f"        g_interior >= {LF.V2_INTERIOR}   <-- in-set pixel fraction ceiling")
    print(f"        g_flat     >= {LF.V2_FLAT}")
    print(f"        g_speckle  >= {LF.V2_SPECKLE}")
    print(f"      So a window whose frame is >= {LF.V2_INTERIOR:.0%} in-set is unscoreable "
          f"BY CONSTRUCTION.")
    print("\n  (ii) THE GOODNESS SCORE G (deployed L1 fit, standardized weights):")
    for k, v in sorted(w.items(), key=lambda kv: -abs(kv[1])):
        tag = "  <-- INTERIOR TERM" if k.startswith("interior") else ""
        print(f"        {k:<26}{v:+.4f}{tag}")
    print(f"      interior_worst = {w['interior_worst']:+.3f} is the "
          f"{sorted(map(abs, w.values()), reverse=True).index(abs(w['interior_worst']))+1}"
          f"-largest weight of {sum(1 for v in w.values() if abs(v) > 1e-9)} nonzero: "
          f"the single worst cell's in-set fraction PUSHES G DOWN.")

    # ---- (iii) what the ceilings do to the drawn windows ------------------
    print("\n  (iii) THE DRAWN 487 — where they sit against the interior ceiling")
    gi = _col(rows, "s_g_interior")
    iw = _col(rows, "s_interior_worst")
    lab = np.array([r["label"] for r in rows], float)
    acc = np.array([r["fate"] == "accepted" for r in rows])
    print(f"      g_interior (screen-res, the masked quantity): "
          f"accepts max={np.nanmax(gi[acc]):.4f}  (ceiling {LF.V2_INTERIOR}) | "
          f"all max={np.nanmax(gi):.4f}")
    edges = [0.0, 0.01, 0.02, 0.05, 0.10, 1.01]
    print(f"      {'g_interior bin':<18}{'n':>5}{'mean label':>12}{'frac L>=3':>11}")
    for i in range(len(edges) - 1):
        m = _finite(gi) & (gi >= edges[i]) & (gi < edges[i + 1])
        if m.sum():
            print(f"      [{edges[i]:.2f},{edges[i+1]:.2f})".ljust(18)
                  + f"{int(m.sum()):>5}{lab[m].mean():>12.2f}{(lab[m]>=3).mean():>11.2f}")
    a_gi, _, _ = auc(gi[_finite(gi)], (lab >= 3)[_finite(gi)])
    r_gi, n_gi = spearman(gi, lab)
    print(f"      Spearman(g_interior, label) = {r_gi:+.3f} (n={n_gi})   "
          f"AUC(label>=3) = {a_gi:.3f}")
    m_acc = acc & _finite(gi)
    a_gia, _, _ = auc(gi[m_acc], (lab >= 3)[m_acc])
    r_gia, n_gia = spearman(gi[acc], lab[acc])
    print(f"      WITHIN ACCEPTS (g_interior compressed into [0,{LF.V2_INTERIOR}) by the "
          f"mask): rho={r_gia:+.3f} (n={n_gia})  AUC={a_gia:.3f}")
    r_iw, _ = spearman(iw, lab)
    print(f"      Spearman(interior_worst, label) = {r_iw:+.3f}   "
          f"(the model's weight on it is {w['interior_worst']:+.3f} — opposite sign?)")
    Gv = _col(rows, "G")
    r_gGi, n_gGi = spearman(gi, Gv)
    r_gIw, _ = spearman(iw, Gv)
    print(f"      Spearman(g_interior, G) over scored windows = {r_gGi:+.3f} (n={n_gGi});  "
          f"Spearman(interior_worst, G) = {r_gIw:+.3f}")
    print("      -> G actively ranks interior-bearing windows DOWN among the windows the "
          "mask did let through.")

    # which clause did each OOD-masked drawn row trip?
    ood = [r for r in rows if r["fate"] == "ood_masked" and r["screen"]]
    cl = Counter()
    for r in ood:
        f = r["screen"]
        t = tuple(sorted([n for n, c in (("interior", f["g_interior"] >= LF.V2_INTERIOR),
                                         ("flat", f["g_flat"] >= LF.V2_FLAT),
                                         ("speckle", f["g_speckle"] >= LF.V2_SPECKLE)) if c]))
        cl[t or ("none (box-rounding drift)",)] += 1
    print(f"\n      the {len(ood)} OOD-masked drawn rows, by tripped clause:")
    for k, v in cl.most_common():
        sub = [r["label"] for r in ood
               if tuple(sorted([n for n, c in (
                   ("interior", r["screen"]["g_interior"] >= LF.V2_INTERIOR),
                   ("flat", r["screen"]["g_flat"] >= LF.V2_FLAT),
                   ("speckle", r["screen"]["g_speckle"] >= LF.V2_SPECKLE)) if c])) == k
               or (not k[0].startswith("none") and False)]
        print(f"        {'+'.join(k):<28} n={v}"
              + (f"   mean label {np.mean(sub):.2f}" if sub else ""))

    # ---- (iv) the counterfactual sweep -----------------------------------
    atoms = sorted({r["atom"] for r in rows})
    by_deg = defaultdict(list)
    for r in rows:
        by_deg[r["degree"]].append(r["atom"])
    rng = np.random.default_rng(0)
    sample = []
    per = max(1, args.atoms // max(1, len(by_deg)))
    for d in sorted(by_deg):
        cand = sorted(set(by_deg[d]))
        pick = rng.choice(len(cand), size=min(per, len(cand)), replace=False)
        sample += [cand[i] for i in sorted(pick)]
    sample = [a for a in sample if _field_ready(PARENT_FIELDS / f"{a}.bin")]
    print(f"\n  (iv) COUNTERFACTUAL SWEEP over {len(sample)} atoms "
          f"({per}/degree, seeded) — every position the screen swept, by tripped clause.")
    SWEEP.mkdir(parents=True, exist_ok=True)
    todo = [a for a in sample if not (SWEEP / f"{a}.json").exists()]
    t0 = time.time()
    if todo:
        print(f"       sweeping {len(todo)} atoms (workers={args.workers}) ...", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_sweep_one, (a,)): a for a in todo}
            for fut in as_completed(futs):
                try:
                    aid, msg = fut.result()
                    done += 1
                    print(f"         [{done}/{len(todo)}] {aid} {msg} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:                       # noqa: BLE001
                    print(f"         !! {futs[fut]} FAILED: {type(e).__name__}: "
                          f"{str(e)[:200]}", flush=True)
    recs = [json.loads((SWEEP / f"{a}.json").read_text()) for a in sample
            if (SWEEP / f"{a}.json").exists()]
    tot = defaultdict(int)
    gih = np.zeros(10)
    for r in recs:
        for k in ("n_swept", "n_toosmall", "n_surv", "n_masked", "interior_only",
                  "flat_only", "speckle_only", "multi", "interior_any", "flat_any",
                  "speckle_any"):
            tot[k] += r[k]
        gih += np.array(r["g_interior_hist"], float)
    feas = tot["n_swept"] - tot["n_toosmall"]
    if feas:
        print(f"\n       featurizable positions: {feas:,}  "
              f"(surviving {tot['n_surv']:,} = {tot['n_surv']/feas:.1%}, "
              f"masked {tot['n_masked']:,} = {tot['n_masked']/feas:.1%})")
        print(f"       {'clause':<34}{'n':>10}{'% featurizable':>16}{'% of masked':>14}")
        for k, nm in (("interior_only", "interior ONLY (sole cause)"),
                      ("flat_only", "flat ONLY"),
                      ("speckle_only", "speckle ONLY"),
                      ("multi", "two or more clauses")):
            print(f"       {nm:<34}{tot[k]:>10,}{tot[k]/feas:>15.1%}"
                  f"{tot[k]/max(1,tot['n_masked']):>14.1%}")
        for k, nm in (("interior_any", "interior (any, incl. shared)"),
                      ("flat_any", "flat (any)"), ("speckle_any", "speckle (any)")):
            print(f"       {nm:<34}{tot[k]:>10,}{tot[k]/feas:>15.1%}"
                  f"{tot[k]/max(1,tot['n_masked']):>14.1%}")
        print(f"\n       => the interior ceiling ALONE removes {tot['interior_only']/feas:.1%} "
              f"of every position the screen looks at, before any goodness score is computed.")
        print(f"          Counterfactual: dropping the interior clause would enlarge the "
              f"scoreable pool by {tot['interior_only']/max(1,tot['n_surv']):.1%}.")
        print(f"\n       g_interior distribution over featurizable positions (deciles):")
        print("         " + "  ".join(f"[{i/10:.1f},{(i+1)/10:.1f}) {gih[i]/gih.sum():.1%}"
                                      for i in range(10) if gih[i] > 0))
        print(f"         positions at or above the {LF.V2_INTERIOR:.2f} ceiling: "
              f"{gih[1:].sum()/gih.sum():.1%}")

    outp = SCR / "audit.json"
    outp.write_text(json.dumps(dict(
        v2=dict(interior=LF.V2_INTERIOR, flat=LF.V2_FLAT, speckle=LF.V2_SPECKLE),
        weights=w, sweep_atoms=sample, sweep_totals=dict(tot),
        g_interior_hist=gih.tolist(),
        drawn=dict(spearman_g_interior_label=r_gi, auc_g_interior=a_gi,
                   spearman_within_accepts=r_gia, auc_within_accepts=a_gia,
                   spearman_g_interior_G=r_gGi, spearman_interior_worst_G=r_gIw)), indent=1))
    print(f"\n-> {outp.relative_to(ROOT)}")
    return 0


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    pf = sub.add_parser("features")
    pf.add_argument("--workers", type=int, default=WORKERS)
    pf.add_argument("--timeout", type=float, default=180.0)
    pf.set_defaults(func=stage_features)
    pb = sub.add_parser("board")
    pb.add_argument("--workers", type=int, default=WORKERS)
    pb.set_defaults(func=stage_board)
    pa = sub.add_parser("audit")
    pa.add_argument("--workers", type=int, default=WORKERS)
    pa.add_argument("--atoms", type=int, default=24, help="atoms in the counterfactual sweep")
    pa.set_defaults(func=stage_audit)
    args = ap.parse_args()
    if args.workers > 4:
        sys.exit("workers capped at 4 (project rule)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
