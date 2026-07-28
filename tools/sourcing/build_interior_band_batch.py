#!/usr/bin/env python
r"""Interior-band label batch — degree x interior crossed, drawn from MASKED positions.

WHY. The deployed OOD mask (`q4_stage1_linear_fit._v2_drop`) discards any window whose
frame is >= 10% in-set, unscored. The bake-off measured that clause: it alone removes
20.2% of every position the screen sweeps (34.0% of everything masked; +49.8% pool if
dropped) — and the labeled corpus contains **four** crops in the entire [0.10, 0.50]
band it cuts, which is the circularity: the screen that built the corpus made sure the
band was never populated. Nobody has ever looked. This batch looks.

THE DESIGN — two things it must not confound.

  1. **Interior vs degree.** In the 487 the two moved together and the analysis could not
     separate them. Here they are crossed EXPLICITLY: degree {2,3,4,5} x interior band
     {0.10-0.20, 0.20-0.35, 0.35-0.50}, ~5 crops per cell.

  2. **Interior vs FRAMING METHOD.** The 487 were G-maxima framed and G penalizes interior
     (`interior_worst` = -1.278, its 2nd-largest weight), so G-maxima framing physically
     cannot produce a high-interior window. Any framing rule used here therefore differs
     from the 487's, and a naive new-vs-old comparison would confound interior with the
     framing method. So this batch carries its own **low-interior control arm** (<0.10,
     same degrees, ~5/degree) drawn by the **identical sampler** — same swept grid, same
     scale mix, same uniform-random draw, no other predicate. Within this batch, interior
     fraction is the only thing that varies between the arms.

THE SAMPLER (both arms, identically). Uniform-random over the positions the DEPLOYED
screen actually sweeps: `LF.FIELD_SCALES` x `LF.DENSE_STRIDE_FRAC` stride x 16:9 windows
on the same cached 2176x1224 parent atom fields as the 487 — geometry copied verbatim
from `MT._sweep_fates`. The per-position scale is drawn to match the 487's realized scale
mix, so scale is not a confound either. **The ONLY selection predicate is the interior
band.** G is never used to frame or to filter; the counterfactual G of each candidate is
recorded for the analysis afterward and nothing reads it before the labels exist.

`int_perim_area` and `coh_scale_drop` (the two features that survived degree-conditioning
in the bake-off) are computed on every drawn crop at draw time and stored with the
manifest — **recorded, never selected on**.

Stages (resumable; each a subcommand):

  sweep    Per admitted roster atom: featurize every swept position out of its cached
           parent field, reservoir-sample candidates per (interior band, scale). Long
           pole (~2M featurize calls over 160 atoms). Parallel, <=4 workers.
  draw     Stratified degree x band draw off the candidate cache -> durable manifest +
           the corpus batch (images.jsonl / batch.json). <=3 crops/atom, split INHERITED
           from the source atom (never reassigned).
  feat     Re-derive each drawn crop's 1280x720 f64 field and compute the bake-off's crop
           features on it (`interior_bakeoff.crop_features`, imported — not reimplemented)
           -> the durable feature table beside the manifest.
  render   Canonical corpus crop (1280x720 ss4 lanczos3 q90) + vivid blue_orange
           companion, exactly the 487's spec so the labeler's eye stays calibrated.
  report   Verification: realized degree x band counts per arm and split, crops-per-atom
           histogram, scale/clause composition, window-overlap audit, and the vivid
           band-stratified sheet.

Run (background sweep + render; both are long):
  uv run python tools/sourcing/build_interior_band_batch.py sweep  [--workers 4] [--wall-seconds N]
  uv run python tools/sourcing/build_interior_band_batch.py draw
  uv run python tools/sourcing/build_interior_band_batch.py feat   [--workers 4]
  uv run python tools/sourcing/build_interior_band_batch.py render [--workers 4]
  uv run python tools/sourcing/build_interior_band_batch.py report

Reads:   data/minibrot_roster/roster.jsonl
         scratch/minibrot_batch/fields/<atom>.bin   (the 487's own cached parent fields)
Writes:  scratch/interior_band_batch/{cand,fields,report.txt,band_sheet.png}  (regenerable)
         data/minibrot_roster/interior_band_v1/{draw.jsonl,interior_features.jsonl}
         data/label_corpus/batches/2026-07-27_interior_band_v1/{images.jsonl,batch.json}

NOTHING DEPLOYED IS CHANGED. No cutoff, screen, mask, draw rule, or production feature is
touched; `q4_stage1_linear_fit` / `q4_multibrot_transfer` / `q4_harvest_tight` are imported
read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "tools", "corpus"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                            # noqa: E402
import corpus_common as cc                              # noqa: E402
import build_minibrot_batch as BMB                      # noqa: E402 (reuse: coords/seed/io)
from tools.studies import q4_stage1_linear_fit as LF    # noqa: E402 (deployed screen, read-only)
from tools.studies import q4_multibrot_transfer as MT   # noqa: E402 (read-only)

# ---- geometry + spec: everything inherited, nothing re-invented ------------------
FIELD_W, FIELD_H = MT.W, MT.H                  # 2176 x 1224, same parent fields as the 487
WORKERS = 4                                    # project rule: <=4
CROP_W, CROP_H, CROP_SS = BMB.CROP_W, BMB.CROP_H, BMB.CROP_SS
CROP_FILTER, INTERIOR_MODE, COMPOSITION = BMB.CROP_FILTER, BMB.INTERIOR_MODE, BMB.COMPOSITION
PALETTE_SOURCE = BMB.PALETTE_SOURCE            # data/palettes/score3_colormaps.json
VIVID_PALETTE, VIVID_SOURCE = BMB.VIVID_PALETTE, BMB.VIVID_SOURCE

BATCH_ID = "2026-07-27_interior_band_v1"
GEN_VERSION = "interior_band_v1"
PRESENTATION_SEED = 0x1B0DE5                   # UI blind-shuffle seed, recorded in batch.json
DRAW_SEED = 20260727                           # candidate-draw seed

# ---- the arms ------------------------------------------------------------------
# `key`, lo, hi (half-open on the SCREEN-resolution in-set fraction `g_interior` — the
# exact quantity the deployed mask cuts on), arm. The interior arm's three bands are the
# prompt's; `control` is the same sampler with the band predicate moved below the ceiling.
BANDS = [
    ("control", 0.00, 0.10, "low_interior_control"),
    ("i10_20", 0.10, 0.20, "interior_band"),
    ("i20_35", 0.20, 0.35, "interior_band"),
    ("i35_50", 0.35, 0.50, "interior_band"),
]
BAND_LO = {b[0]: b[1] for b in BANDS}
BAND_HI = {b[0]: b[2] for b in BANDS}
BAND_ARM = {b[0]: b[3] for b in BANDS}
DEGREES = (2, 3, 4, 5)
PER_CELL = 5                                   # 4 degrees x 4 bands x 5 = 80 crops
PER_ATOM_CAP = 3
CAND_CAP = 24                                  # reservoir size per (atom, band, scale)

# The 487's realized window-scale mix (draw.jsonl box widths: 422 / 50 / 15 at
# 0.06 / 0.09 / 0.14). Sampling scale to this mix is what makes "same scale and geometry
# as the 487 crops" true of the DISTRIBUTION, not just the scale set.
SCALE_MIX = {0.06: 422 / 487, 0.09: 50 / 487, 0.14: 15 / 487}

SCR = paths.scratch("interior_band_batch")
CAND = SCR / "cand"
CROP_FIELDS = SCR / "fields"
FIELDS = BMB.FIELDS                            # scratch/minibrot_batch/fields
DIR_REL = "data/minibrot_roster/interior_band_v1"
DRAW_REL = f"{DIR_REL}/draw.jsonl"
FEAT_REL = f"{DIR_REL}/interior_features.jsonl"


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def band_of(gi: float):
    """Interior band key for a screen-resolution in-set fraction, or None if >= 0.50
    (above every band this batch draws — deliberately not backfilled into i35_50)."""
    for key, lo, hi, _arm in BANDS:
        if lo <= gi < hi:
            return key
    return None


def clauses_of(gi, gflat, gspeck):
    """Which deployed OOD-mask clauses this window trips (recorded, never selected on)."""
    c = []
    if gi >= LF.V2_INTERIOR:
        c.append("interior")
    if gflat >= LF.V2_FLAT:
        c.append("flat")
    if gspeck >= LF.V2_SPECKLE:
        c.append("speckle")
    return c


# ======================================================================= #
# STAGE: sweep — candidate harvest, uniform-random within each band
# ======================================================================= #
def _sweep_one(job):
    """Every position the deployed screen sweeps on one atom's field, reservoir-sampled
    into (band, scale) buckets. Grid geometry copied verbatim from MT._sweep_fates, so the
    candidate universe IS the screen's swept set — the sampler differs from the screen only
    in what it selects on (interior band, not G)."""
    atom, model = job
    aid = atom["id"]
    out = CAND / f"{aid}.json"
    if out.exists():
        return aid, "cached"
    sc, clf, keys = model
    field, fw, fh = MT._load_field(FIELDS / f"{aid}.bin")
    rng = np.random.default_rng(BMB._stable_seed(aid))
    seen = Counter()                 # (band, scale) -> positions seen (the reservoir's n)
    keep = defaultdict(list)         # (band, scale) -> up to CAND_CAP sampled candidates
    n_swept = n_toosmall = n_over50 = 0

    for s in LF.FIELD_SCALES:
        Wp = max(8, int(round(s * fw)))
        Hp = max(8, int(round(Wp * 9 / 16)))
        if Hp >= fh or Wp >= fw:
            continue
        st = max(4, int(round(LF.DENSE_STRIDE_FRAC * Wp)))
        for y in range(0, fh - Hp + 1, st):
            for x in range(0, fw - Wp + 1, st):
                n_swept += 1
                f = LF.featurize(field[y:y + Hp, x:x + Wp])
                if f is None:
                    n_toosmall += 1
                    continue
                b = band_of(f["g_interior"])
                if b is None:
                    n_over50 += 1
                    continue
                k = (b, s)
                seen[k] += 1
                rec = dict(box=[(x + Wp / 2) / fw, (y + Hp / 2) / fh, Wp / fw, Hp / fh],
                           scale=s, gi=f["g_interior"], gflat=f["g_flat"],
                           gspeck=f["g_speckle"], _feat=[f[kk] for kk in keys])
                slot = keep[k]
                if len(slot) < CAND_CAP:
                    slot.append(rec)
                else:                                   # reservoir replacement
                    j = int(rng.integers(0, seen[k]))
                    if j < CAND_CAP:
                        slot[j] = rec

    # Counterfactual G for the KEPT candidates only — recorded for the post-label analysis
    # (does this batch's material rank the way G says it would?). Never read by `draw`.
    for k, slot in keep.items():
        X = np.array([r.pop("_feat") for r in slot], float)
        g = clf.decision_function(sc.transform(X))
        for r, gv in zip(slot, g):
            r["G_cf"] = round(float(gv), 5)

    rec = dict(atom_id=aid, degree=atom["degree"], period=atom["period"],
               period_band=atom["band"], split=atom["split"], family=atom["family"],
               cx=atom["cx"], cy=atom["cy"], fw=atom["fw"], maxiter=atom["maxiter"],
               n_swept=n_swept, n_toosmall=n_toosmall, n_over_050=n_over50,
               seen={f"{b}|{s}": n for (b, s), n in seen.items()},
               cands={f"{b}|{s}": v for (b, s), v in keep.items()})
    BMB._atomic_write_json(rec, out)
    tot = {b: sum(n for (bb, _s), n in seen.items() if bb == b) for b, *_ in BANDS}
    return aid, f"swept={n_swept} " + " ".join(f"{b}={tot[b]}" for b, *_ in BANDS)


def stage_sweep(args):
    atoms = BMB.load_admitted()
    have = [a for a in atoms if BMB._field_ready(FIELDS / f"{a['id']}.bin")]
    if len(have) != len(atoms):
        print(f"  NOTE: {len(atoms) - len(have)} admitted atoms have no cached parent field "
              f"(run build_minibrot_batch screen first) — skipping them.")
    CAND.mkdir(parents=True, exist_ok=True)
    todo = [a for a in have if not (CAND / f"{a['id']}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"sweep: {len(have)} atoms with fields, {len(have) - len(todo)} cached, "
          f"{len(todo)} this run. workers={args.workers} wall={args.wall_seconds}s", flush=True)
    if not todo:
        print("sweep: all atoms cached — nothing to do.")
        return 0
    print("fitting the deployed model (unchanged q4_harvest_tight fit; used ONLY for the "
          "recorded counterfactual G) ...", flush=True)
    model, _tight = MT._fit_model()

    t0, done = time.time(), 0
    jobs = iter([(a, model) for a in todo])
    per_unit = 60.0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for _ in range(min(args.workers, len(todo))):
            j = next(jobs, None)
            if j is not None:
                futs[ex.submit(_sweep_one, j)] = j[0]["id"]
        while futs:
            fut = next(as_completed(list(futs)))
            aid = futs.pop(fut)
            try:
                rid, msg = fut.result()
                done += 1
                print(f"  [{done}/{len(todo)}] {rid} {msg} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:                       # noqa: BLE001 — stay resumable
                print(f"  !! {aid} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if args.wall_seconds - (time.time() - t0) > per_unit:
                j = next(jobs, None)
                if j is not None:
                    futs[ex.submit(_sweep_one, j)] = j[0]["id"]
    left = sum(1 for a in have if not (CAND / f"{a['id']}.json").exists())
    print(f"sweep: {done} this run, {left} left ({time.time()-t0:.0f}s)." if left else
          f"sweep: COMPLETE — {len(have)} atoms cached ({time.time()-t0:.0f}s).", flush=True)
    return 0


# ======================================================================= #
# STAGE: draw — stratified degree x band, one sampler for both arms
# ======================================================================= #
def _clash(c, picked, sep):
    """Elliptical center-separation test, the same metric the screen's own NMS uses —
    keeps two windows drawn off ONE atom from being the same picture."""
    for k in picked:
        du = (c["box"][0] - k["box"][0]) / (0.5 * (c["box"][2] + k["box"][2]))
        dv = (c["box"][1] - k["box"][1]) / (0.5 * (c["box"][3] + k["box"][3]))
        if du * du + dv * dv < sep * sep:
            return True
    return False


def _pick_scale(rng, avail):
    """Draw a scale from the 487's realized mix, restricted to the scales that actually
    have a candidate left here; falls back to uniform if none of the mix is available."""
    w = np.array([SCALE_MIX.get(s, 0.0) for s in avail], float)
    if w.sum() <= 0:
        w = np.ones(len(avail))
    return avail[int(rng.choice(len(avail), p=w / w.sum()))]


EVAL_EVERY = 4          # every 4th atom offered to a cell is an eval atom


def _cell_atom_order(atoms, A, rng):
    """Seeded atom order for one (band, degree) cell, with eval atoms slotted every
    EVAL_EVERY positions.

    The split is INHERITED from the roster and never reassigned — this only changes the
    ORDER in which atoms are offered. Without it a 5-pick cell draws its eval share by
    luck, and the two arms end up with different train/eval mixes (0.43 vs 0.10 on a
    partial-sweep dry run), which would be a second thing varying between them."""
    tr = [a for a in atoms if A[a]["split"] == "train"]
    ev = [a for a in atoms if A[a]["split"] == "eval"]
    rng.shuffle(tr)
    rng.shuffle(ev)
    order, i, j = [], 0, 0
    while i < len(tr) or j < len(ev):
        take_eval = (len(order) % EVAL_EVERY == EVAL_EVERY - 1 and j < len(ev)) or i >= len(tr)
        if take_eval and j < len(ev):
            order.append(ev[j]); j += 1
        else:
            order.append(tr[i]); i += 1
    return order


def stage_draw(args):
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CAND.glob("*.json"))]
    if not recs:
        sys.exit("no candidate cache — run the `sweep` stage first.")
    A = {r["atom_id"]: r for r in recs}
    print(f"draw: {len(A)} swept atoms "
          f"({sum(1 for r in recs if r['split']=='eval')} eval / "
          f"{sum(1 for r in recs if r['split']=='train')} train)")

    # pools[(atom, band, scale)] = shuffled candidate list
    rng = np.random.default_rng(DRAW_SEED)
    pools = defaultdict(list)
    for r in recs:
        for key, lst in r["cands"].items():
            b, s = key.split("|")
            pool = list(lst)
            rng.shuffle(pool)
            pools[(r["atom_id"], b, float(s))] = pool

    budget = defaultdict(lambda: PER_ATOM_CAP)
    picked_by_atom = defaultdict(list)
    drawn, shortfall = [], {}
    for band, _lo, _hi, arm in BANDS:
        for deg in DEGREES:
            atoms = _cell_atom_order(sorted(a for a in A if A[a]["degree"] == deg), A, rng)
            sel, ai, stall = [], 0, 0
            while len(sel) < PER_CELL and stall < len(atoms):
                a = atoms[ai % len(atoms)]
                ai += 1
                got = False
                if budget[a] > 0:
                    avail = [s for s in LF.FIELD_SCALES if pools[(a, band, s)]]
                    while avail and not got:
                        s = _pick_scale(rng, avail)
                        pool = pools[(a, band, s)]
                        while pool:
                            c = pool.pop()
                            if _clash(c, picked_by_atom[a], MT.HT.SEP):
                                continue
                            sel.append(dict(c, atom_id=a, band=band, arm=arm))
                            picked_by_atom[a].append(c)
                            budget[a] -= 1
                            got = True
                            break
                        if not got:
                            avail = [x for x in avail if pools[(a, band, x)]]
                stall = 0 if got else stall + 1
            if len(sel) < PER_CELL:
                shortfall[(band, deg)] = (len(sel), PER_CELL)
            drawn += sel

    # crop coordinates + maxiter, from the atom's own geometry (BMB._crop_coords verbatim)
    out = []
    for c in drawn:
        a = A[c["atom_id"]]
        cx, cy, fws, fwm = BMB._crop_coords(a, tuple(c["box"]))
        out.append(dict(
            atom_id=a["atom_id"], degree=a["degree"], period=a["period"],
            period_band=a["period_band"], split=a["split"], family=a["family"],
            arm=c["arm"], band=c["band"], box=[float(x) for x in c["box"]],
            scale=float(c["scale"]), g_interior=float(c["gi"]), g_flat=float(c["gflat"]),
            g_speckle=float(c["gspeck"]), clauses=clauses_of(c["gi"], c["gflat"], c["gspeck"]),
            G_counterfactual=float(c["G_cf"]),
            cx=cx, cy=cy, fw=fws, maxiter=int(MT.dcf._maxiter_for_fw(float(fwm)))))

    # Opaque image_id: a seeded shuffle assigns the index, and the suffix is a content hash
    # of the window. Nothing in the FILENAME encodes arm, band, degree, period or interior —
    # so the one identifier that reaches the labeler's browser carries no answer.
    order = list(range(len(out)))
    np.random.default_rng(PRESENTATION_SEED).shuffle(order)
    for slot, oi in enumerate(order):
        c = out[oi]
        h = BMB._stable_seed(f"{c['atom_id']}|{c['box']}|{c['scale']}")
        c["image_id"] = f"ib{slot:04d}_{h:08x}"
    out.sort(key=lambda c: c["image_id"])

    dp = paths.durable(DRAW_REL, mkparents=True)
    with open(dp, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    _write_batch(out)

    print(f"  drawn: {len(out)} crops over {len({c['atom_id'] for c in out})} atoms")
    for band, _lo, _hi, arm in BANDS:
        n = sum(1 for c in out if c["band"] == band)
        print(f"    {band:<8} [{BAND_LO[band]:.2f},{BAND_HI[band]:.2f})  {arm:<21} n={n}")
    if shortfall:
        print("  UNDER-FILLED CELLS (not backfilled across bands, by design):")
        for (b, d), (got, want) in sorted(shortfall.items()):
            print(f"    band {b} degree {d}: {got}/{want}")
    else:
        print("  every (band, degree) cell filled to target.")
    print(f"  -> {dp}")
    return 0


def _write_batch(rows_in):
    names = BMB._palette_names()
    rows = []
    for c in rows_in:
        pal = names[BMB._stable_seed(c["image_id"]) % len(names)]     # seeded score-3 draw
        render = cc.render_block(cx=c["cx"], cy=c["cy"], fw=c["fw"], maxiter=c["maxiter"],
                                 palette=pal, composition=COMPOSITION, width=CROP_W,
                                 height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                                 interior_mode=INTERIOR_MODE)
        render["fractal_type"] = c["family"]
        render["c_re"] = None
        render["c_im"] = None
        prov = cc.provenance_block(
            GEN_VERSION, BATCH_ID, family=c["family"],
            selection_role=c["arm"],
            stratum=f"{c['band']}[{BAND_LO[c['band']]:.2f},{BAND_HI[c['band']]:.2f})",
            interior_frac=c["g_interior"],
            focus_score=c["G_counterfactual"],     # RECORDED counterfactual; not a selector
            decoded_class="+".join(c["clauses"]) or "unmasked",
            descend_mode=f"minibrot_d{c['degree']}_p{c['period']}",
        )
        rows.append(cc.make_row(c["image_id"], render, prov, cc.label_block()))
    bdir = Path(cc.batch_dir(BATCH_ID))
    cc.write_jsonl(rows, str(bdir / "images.jsonl"))
    bj = dict(
        schema_version=1, batch_id=BATCH_ID, generator_version=GEN_VERSION,
        created=None, labeler=None,
        presentation_seed=PRESENTATION_SEED,
        vivid_companion=VIVID_PALETTE,
        purpose=("degree x interior-band crossed batch drawn from positions the deployed OOD "
                 "mask currently discards, plus a same-sampler low-interior control arm"),
        counts=dict(total=len(rows), **{b: sum(1 for c in rows_in if c["band"] == b)
                                        for b, *_ in BANDS}),
        bands={b: [lo, hi, arm] for b, lo, hi, arm in BANDS},
        render_defaults=dict(width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE, composition=COMPOSITION,
                             palette_roster="data/palettes/score3_colormaps.json",
                             vivid_companion=VIVID_PALETTE,
                             maxiter="per-crop deploy maxiter (dcf._maxiter_for_fw)"),
        render_recipe=cc.render_recipe_stamp(PALETTE_SOURCE),
        sampling_metaparameters=dict(
            sampler="uniform-random over the deployed swept grid (LF.FIELD_SCALES x "
                    "DENSE_STRIDE_FRAC, 16:9), reservoir-sampled per (atom, band, scale)",
            selection_predicate="screen-resolution g_interior band ONLY",
            g_used_for_selection=False,
            scale_mix="matched to the 487-crop batch's realized mix "
                      f"{ {k: round(v, 3) for k, v in SCALE_MIX.items()} }",
            per_cell_target=PER_CELL, per_atom_cap=PER_ATOM_CAP,
            draw_seed=DRAW_SEED,
            split="inherited from the source roster atom, never reassigned; the per-cell "
                  f"atom ORDER offers an eval atom every {EVAL_EVERY} slots so both arms "
                  "carry the same eval share",
            recorded_not_selected=["G_counterfactual", "int_perim_area", "coh_scale_drop"]),
    )
    (bdir / "batch.json").write_text(json.dumps(bj, indent=2), encoding="utf-8")
    if not (bdir / "scores.json").exists():
        (bdir / "scores.json").write_text("{}", encoding="utf-8")
    print(f"  batch -> {bdir}  ({len(rows)} rows, images.jsonl + batch.json)")


# ======================================================================= #
# STAGE: feat — crop-resolution features, recorded at draw time
# ======================================================================= #
def _feat_one(job):
    """Dump one crop's 1280x720 f64 escape-time field and run the bake-off's feature set
    on it. Imported, not reimplemented — the numbers must be comparable to the 487's."""
    from tools.studies import interior_bakeoff as IB
    row, timeout = job
    iid = row["image_id"]
    cache = CROP_FIELDS / f"{iid}.json"
    if cache.exists():
        return iid, "cached"
    b = CROP_FIELDS / f"{iid}.bin"
    if not (b.exists() and b.with_suffix(".json").exists()):
        IB._dump_crop_field(row, b, timeout)
    field, _, _ = IB._load_field(b)
    f = IB.crop_features(field)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(f))
    os.replace(tmp, cache)
    return iid, "done"


def stage_feat(args):
    from tools.studies import interior_bakeoff as IB
    rows = _read_jsonl(paths.durable(DRAW_REL))
    CROP_FIELDS.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (CROP_FIELDS / f"{r['image_id']}.json").exists()]
    print(f"feat: {len(rows)} crops, {len(rows)-len(todo)} cached, {len(todo)} to derive "
          f"(1280x720 ss1 f64 field + crop features, workers={args.workers})", flush=True)
    t0, done = time.time(), 0
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_feat_one, (r, args.timeout)): r["image_id"] for r in todo}
            for fut in as_completed(futs):
                try:
                    iid, st = fut.result()
                    done += 1
                    if done % 20 == 0 or done == len(todo):
                        print(f"  [{done}/{len(todo)}] {iid} {st} ({time.time()-t0:.0f}s)",
                              flush=True)
                except Exception as e:                   # noqa: BLE001
                    print(f"  !! {futs[fut]} FAILED: {type(e).__name__}: {str(e)[:200]}",
                          flush=True)
    tbl, missing = [], 0
    for r in rows:
        p = CROP_FIELDS / f"{r['image_id']}.json"
        if not p.exists():
            missing += 1
            tbl.append(dict(r, crop=None))
            continue
        cf = json.loads(p.read_text())
        tbl.append(dict(r, crop={k: cf[k] for k in IB.CROP_KEYS}))
    fp = paths.durable(FEAT_REL, mkparents=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in tbl:
            f.write(json.dumps(r) + "\n")
    print(f"  -> {fp}  ({len(tbl)} rows, {missing} without crop features)")
    ipa = [r["crop"]["int_perim_area"] for r in tbl if r["crop"]
           and np.isfinite(r["crop"]["int_perim_area"])]
    csd = [r["crop"]["coh_scale_drop"] for r in tbl if r["crop"]
           and np.isfinite(r["crop"]["coh_scale_drop"])]
    print(f"  int_perim_area: n={len(ipa)} median {np.median(ipa):.4f}" if ipa else
          "  int_perim_area: n=0")
    print(f"  coh_scale_drop: n={len(csd)} median {np.median(csd):.4f}" if csd else
          "  coh_scale_drop: n=0")
    print("  (both RECORDED for the analysis; neither was used to select anything.)")
    return 0


# ======================================================================= #
# STAGE: render
# ======================================================================= #
def _render_row(job):
    row, crops_dir, vivid_dir, timeout = job
    iid, render = row["image_id"], row["render"]
    made = []
    canon = crops_dir / f"{iid}.jpg"
    if not canon.exists():
        cc.render_corpus_crop(render, str(canon), palette_source=PALETTE_SOURCE, timeout=timeout)
        made.append("canon")
    vivid = vivid_dir / f"{iid}.jpg"
    if not vivid.exists():
        vr = dict(render)
        vr["palette"] = VIVID_PALETTE
        cc.render_corpus_crop(vr, str(vivid), palette_source=VIVID_SOURCE, timeout=timeout)
        made.append("vivid")
    return iid, made


def stage_render(args):
    bdir = Path(cc.batch_dir(BATCH_ID))
    rows = cc.read_jsonl(str(bdir / "images.jsonl"))
    crops_dir, vivid_dir = bdir / "crops", bdir / "vivid"
    crops_dir.mkdir(parents=True, exist_ok=True)
    vivid_dir.mkdir(parents=True, exist_ok=True)

    def needs(r):
        iid = r["image_id"]
        return not (crops_dir / f"{iid}.jpg").exists() or not (vivid_dir / f"{iid}.jpg").exists()
    todo = [r for r in rows if needs(r)]
    print(f"render: {len(rows)} rows, {len(todo)} need crops (canonical + vivid). "
          f"workers={args.workers}", flush=True)
    if not todo:
        print("render: all crops present.")
        return 0
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_render_row, (r, crops_dir, vivid_dir, args.render_timeout)): r
                for r in todo}
        for fut in as_completed(futs):
            try:
                iid, made = fut.result()
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"  [{done}/{len(todo)}] {iid} {'+'.join(made) or 'cached'} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  !! {futs[fut]['image_id']} FAILED: {type(e).__name__}: "
                      f"{str(e)[:200]}", flush=True)
    left = sum(1 for r in rows if needs(r))
    print(f"render: {done} this run, {left} still missing ({time.time()-t0:.0f}s)." if left
          else f"render: COMPLETE — all {len(rows)} rows ({time.time()-t0:.0f}s).", flush=True)
    return 0


# ======================================================================= #
# STAGE: report — the verification Matt reads BEFORE labeling
# ======================================================================= #
def stage_report(args):
    fp = paths.durable(FEAT_REL)
    rows = _read_jsonl(fp) if Path(fp).exists() else _read_jsonl(paths.durable(DRAW_REL))
    L = []

    def emit(s=""):
        L.append(s)
        print(s)

    emit(f"interior-band batch {BATCH_ID} — {len(rows)} crops")
    emit(f"selection predicate: screen-resolution g_interior band ONLY (deployed mask "
         f"ceiling = {LF.V2_INTERIOR}); G never used to frame or filter.")

    # --- 1. realized degree x band, per arm and split -----------------------
    emit("\n[1] REALIZED degree x interior-band counts (the crossing this batch exists for)")
    for arm in ("interior_band", "low_interior_control"):
        bands = [b for b, *_r in BANDS if BAND_ARM[b] == arm]
        emit(f"\n  arm = {arm}")
        for split in ("train", "eval", "ALL"):
            sub = [r for r in rows if r["arm"] == arm and (split == "ALL" or r["split"] == split)]
            emit(f"    {split:<6} n={len(sub):3d}   " + "  ".join(
                f"{b}:[" + ",".join(
                    f"d{d}={sum(1 for r in sub if r['band']==b and r['degree']==d)}"
                    for d in DEGREES) + "]" for b in bands))
    emit("\n  per-cell realized (target %d/cell):" % PER_CELL)
    under = []
    for b, *_r in BANDS:
        line = f"    {b:<8} " + "  ".join(
            f"d{d}={sum(1 for r in rows if r['band']==b and r['degree']==d)}" for d in DEGREES)
        emit(line)
        for d in DEGREES:
            n = sum(1 for r in rows if r["band"] == b and r["degree"] == d)
            if n < PER_CELL:
                under.append((b, d, n))
    emit("  UNDER-FILLED: " + (", ".join(f"{b}/d{d}={n}" for b, d, n in under) if under
                               else "none — every cell at target"))

    # --- 2. the two arms must differ ONLY in interior -----------------------
    emit("\n[2] ARM COMPARABILITY — the control must differ from the interior arm in "
         "interior fraction and nothing else")
    hdr = f"    {'quantity':<26}{'interior arm':>16}{'control arm':>16}"
    emit(hdr)
    emit("    " + "-" * (len(hdr) - 4))
    I = [r for r in rows if r["arm"] == "interior_band"]
    C = [r for r in rows if r["arm"] == "low_interior_control"]

    def med(sub, fn):
        v = [fn(r) for r in sub]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return np.median(v) if v else float("nan")
    for nm, fn in (("g_interior (median)", lambda r: r["g_interior"]),
                   ("scale 0.06 share", lambda r: 1.0 if r["scale"] == 0.06 else 0.0),
                   ("scale 0.09 share", lambda r: 1.0 if r["scale"] == 0.09 else 0.0),
                   ("scale 0.14 share", lambda r: 1.0 if r["scale"] == 0.14 else 0.0),
                   ("mean degree", lambda r: r["degree"]),
                   ("mean period", lambda r: r["period"]),
                   ("eval share", lambda r: 1.0 if r["split"] == "eval" else 0.0),
                   ("G_counterfactual (median)", lambda r: r["G_counterfactual"])):
        a = np.mean([fn(r) for r in I]) if "share" in nm or "mean" in nm else med(I, fn)
        b = np.mean([fn(r) for r in C]) if "share" in nm or "mean" in nm else med(C, fn)
        emit(f"    {nm:<26}{a:>16.4f}{b:>16.4f}")
    emit(f"    scale mix target (the 487's realized mix): "
         + ", ".join(f"{s}={p:.3f}" for s, p in SCALE_MIX.items()))

    # --- 3. crops per atom --------------------------------------------------
    per_atom = Counter(r["atom_id"] for r in rows)
    hist = Counter(per_atom.values())
    emit(f"\n[3] CROPS PER ATOM — {len(per_atom)} atoms used; cap {PER_ATOM_CAP}")
    emit("    " + ", ".join(f"{k} crop(s): {v} atoms" for k, v in sorted(hist.items()))
         + f"   (max {max(per_atom.values())})")
    assert max(per_atom.values()) <= PER_ATOM_CAP, "per-atom cap violated"

    # --- 4. what the mask says about these windows --------------------------
    emit("\n[4] DEPLOYED-MASK CLAUSE COMPOSITION (recorded, never selected on)")
    for arm in ("interior_band", "low_interior_control"):
        sub = [r for r in rows if r["arm"] == arm]
        cl = Counter("+".join(r["clauses"]) or "unmasked" for r in sub)
        emit(f"    {arm:<22} " + ", ".join(f"{k}={v}" for k, v in cl.most_common()))
    emit("    (the control arm is drawn by the SAME sampler with no extra predicate, so it "
         "carries the flat/speckle base rate of the swept grid — that is the point.)")

    # --- 5. window overlap ---------------------------------------------------
    from tools.studies.interior_bakeoff import _iou
    by_atom = defaultdict(list)
    for r in rows:
        by_atom[r["atom_id"]].append(r)
    pairs = []
    for aid, rs in by_atom.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                v = _iou(rs[i]["box"], rs[j]["box"])
                if v > 0:
                    pairs.append((v, aid, rs[i]["image_id"], rs[j]["image_id"]))
    pairs.sort(reverse=True)
    emit(f"\n[5] SAME-ATOM WINDOW OVERLAP — {len(pairs)} overlapping pairs (IoU>0); "
         f"max IoU {pairs[0][0]:.3f}" if pairs else
         "\n[5] SAME-ATOM WINDOW OVERLAP — zero overlapping pairs")
    for v, aid, a, b in pairs[:5]:
        emit(f"    IoU={v:.3f}  {aid}  {a} <-> {b}")

    # --- 6. the recorded (never-selected) crop features ----------------------
    if rows and rows[0].get("crop"):
        emit("\n[6] RECORDED CROP FEATURES (computed at draw time; NOT used for selection)")
        h = f"    {'band':<8}{'n':>4}{'int_perim_area med':>20}{'coh_scale_drop med':>20}"
        emit(h)
        emit("    " + "-" * (len(h) - 4))
        for b, *_r in BANDS:
            sub = [r for r in rows if r["band"] == b and r.get("crop")]
            ipa = [r["crop"]["int_perim_area"] for r in sub
                   if np.isfinite(r["crop"]["int_perim_area"])]
            csd = [r["crop"]["coh_scale_drop"] for r in sub
                   if np.isfinite(r["crop"]["coh_scale_drop"])]
            emit(f"    {b:<8}{len(sub):>4}"
                 f"{(np.median(ipa) if ipa else float('nan')):>20.4f}"
                 f"{(np.median(csd) if csd else float('nan')):>20.4f}")
        # crop-resolution vs screen-resolution parity for the selection quantity
        gi = np.array([r["g_interior"] for r in rows if r.get("crop")])
        ifr = np.array([r["crop"]["int_frac"] for r in rows if r.get("crop")])
        if len(gi) > 8:
            from scipy.stats import spearmanr
            emit(f"    resolution parity Spearman(screen g_interior, crop int_frac @1280x720) "
                 f"= {spearmanr(gi, ifr).statistic:+.3f} (n={len(gi)})")

    rp = SCR / "report.txt"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {rp}")
    _band_sheet(rows)
    return 0


def _band_sheet(rows):
    """Vivid band-stratified sheet: one row per interior band, control first. Every drawn
    crop appears — this is the pre-label eyeball, so completeness beats prettiness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    vivid = Path(cc.batch_dir(BATCH_ID)) / "vivid"
    groups = []
    for b, lo, hi, arm in BANDS:
        items = sorted([r for r in rows if r["band"] == b
                        and (vivid / f"{r['image_id']}.jpg").exists()],
                       key=lambda r: (r["degree"], r["g_interior"]))
        groups.append((f"{b}\n[{lo:.2f},{hi:.2f})", items))
    if not any(items for _t, items in groups):
        print("band sheet: no vivid crops rendered yet — run `render` first.")
        return
    N = max(len(items) for _t, items in groups)
    fig, axes = plt.subplots(len(groups), N, figsize=(1.7 * N, 1.35 * len(groups) + 1),
                             squeeze=False)
    fig.suptitle(f"{BATCH_ID} — interior-band stratified (vivid {VIVID_PALETTE}); "
                 f"control arm on top, then the three masked bands. "
                 f"Tile caption = degree · screen g_interior.", y=0.995, fontsize=9)
    for ri, (title, items) in enumerate(groups):
        for ci in range(N):
            ax = axes[ri][ci]
            ax.axis("off")
            if ci == 0:
                ax.text(-0.04, 0.5, title, rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=7, weight="bold")
            if ci < len(items):
                r = items[ci]
                ax.imshow(mpimg.imread(vivid / f"{r['image_id']}.jpg"))
                ax.set_title(f"d{r['degree']} · {r['g_interior']:.3f}", fontsize=5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    out = SCR / "band_sheet.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"band sheet -> {out}")


# ======================================================================= #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("sweep")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--wall-seconds", type=float, default=7200.0)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=stage_sweep)
    p = sub.add_parser("draw")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.set_defaults(func=stage_draw)
    p = sub.add_parser("feat")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=stage_feat)
    p = sub.add_parser("render")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--render-timeout", type=float, default=300.0)
    p.set_defaults(func=stage_render)
    p = sub.add_parser("report")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.set_defaults(func=stage_report)
    args = ap.parse_args()
    if args.workers > 4:
        sys.exit("workers capped at 4 (project rule)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
