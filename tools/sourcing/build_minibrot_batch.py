#!/usr/bin/env python
"""Draw the 500-crop minibrot label batch off the durable roster (two arms).

Reuses the DEPLOYED stage-1 screen / OOD mask / G-maxima framing unchanged, imported
read-only from the closed study `tools/studies/q4_multibrot_transfer` (never edited) —
so screen parity with the pilot + deployed harvest is exact.

Stages (resumable; each is a subcommand):

  screen   For every ADMITTED roster atom: dump its f64 field (cached) and run the
           screen, caching per-atom results (kept framings + a sample of OOD-masked
           boxes) to scratch. Parallel (<=4 workers). The long pole (~55s/atom screen).

  draw     From the screen cache, sample the two arms into a durable draw manifest and
           write the corpus batch (images.jsonl / batch.json). Also emits the anchor
           batch's ~8 minibrot picks (accepted framings) as a side file.
             Positive arm ~250: screen-accepted windows (G>=cutoff), spread across the
               four degrees and five period bands as the draw allows.
             Negative arm ~250: ~90% screen-REJECTED-but-OOD-surviving + ~10% OOD-masked,
               with >=50% from the deep bands (10-12, 13-15) so "deep" != "good".
             Both arms: <=3 crops/atom; split INHERITED from the source atom (never
               reassigned). Eval slice period-matched between arms; the match is reported.

  render   Render each drawn row's canonical corpus crop (1280x720 ss4 Lanczos3 q90, the
           existing label-crop spec) AND a vivid (blue/orange) companion. Parallel,
           resumable.

  report   Realized period x band x degree distribution per arm per split, crops-per-atom
           histogram, and the fate-stratified vivid sheet (negatives next to positives).

Run (background the screen; it is >30s):
  uv run python tools/sourcing/build_minibrot_batch.py screen  [--wall-seconds N] [--workers 4]
  uv run python tools/sourcing/build_minibrot_batch.py draw
  uv run python tools/sourcing/build_minibrot_batch.py render  [--workers 4] [--wall-seconds N]
  uv run python tools/sourcing/build_minibrot_batch.py report

Reads:   data/minibrot_roster/roster.jsonl
Writes:  scratch/minibrot_batch/{fields,screen}/           (regenerable cache)
         data/minibrot_roster/batch_v1/draw.jsonl          (durable draw manifest)
         data/minibrot_roster/batch_v1/anchor_minibrot_picks.jsonl (for the anchor batch)
         data/label_corpus/batches/<BATCH_ID>/{images.jsonl,batch.json}
         crops/ + vivid/ under the batch dir (gitignored, rebuildable)
         scratch/minibrot_batch/{distribution_report.txt,fate_sheet.png}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
sys.path.insert(0, os.path.join(_ROOT, "tools", "corpus"))
sys.path.insert(0, _ROOT)

import mpmath as mp                                    # noqa: E402
import paths                                           # noqa: E402
import build_minibrot_roster as RB                     # noqa: E402
import corpus_common as cc                             # noqa: E402
from tools.studies import q4_multibrot_transfer as MT  # noqa: E402 (read-only reuse)

# --- geometry / spec constants (read off the deployed paths, not re-invented) ---
FIELD_W, FIELD_H = MT.W, MT.H                          # 2176 x 1224 screen field
WORKERS = 4                                            # project rule: <=4 workers
DEEP_BANDS = ("10-12", "13-15")
EMIT_DIGITS_GUARD = 12
MAX_MASKED_PER_ATOM = 8                                # sampled OOD-masked boxes cached/atom
REJECT_CAP = 8                                         # NMS'd sub-cutoff reject framings/atom

# Batch identity + the canonical label-crop spec (mirrors build_enrich_batch.py).
BATCH_ID = "2026-07-26_minibrot_roster_v2"
GEN_VERSION = "minibrot_roster_v2"
CROP_W, CROP_H, CROP_SS = 1280, 720, 4
CROP_FILTER, INTERIOR_MODE, COMPOSITION = "lanczos3", "black", "center"
CROP_MAXITER = 8000
PALETTE_SOURCE = os.path.join(_ROOT, "data", "palettes", "score3_colormaps.json")
# canonical (model-facing) palette per degree — a fixed, legible score-3 map, so the
# classifier sees a consistent colorway; the labeler judges from the vivid companion.
CANON_PALETTE = "magma"
VIVID_PALETTE = "cmr.fusion"                           # vivid blue/orange companion

# arm targets
POS_TARGET = 250
NEG_TARGET = 250
NEG_MASKED_FRAC = 0.10
NEG_DEEP_FLOOR = 0.50
PER_ATOM_CAP = 3
ANCHOR_MINIBROT_N = 8

SCR_DIR = paths.scratch("minibrot_batch")
FIELDS = SCR_DIR / "fields"
SCREEN = SCR_DIR / "screen"
DRAW_DIR_REL = "data/minibrot_roster/batch_v1"


# ======================================================================= #
# helpers
# ======================================================================= #
def _atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def _stable_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def load_admitted():
    rpath = paths.durable(RB.ROSTER_PATH)
    rows = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
    return [r for r in rows if r["admitted"]]


def _field_ready(out_bin: Path) -> bool:
    """A field is usable only if BOTH the .bin and its .json metadata sidecar exist (a
    hard-killed dump can leave a partial .bin with no sidecar — force a re-dump then)."""
    return out_bin.exists() and out_bin.with_suffix(".json").exists()


def _dump_field_timed(atom, out_bin: Path, timeout: float):
    """Mirror MT._dump_field (f64 source, W x H, ss1) with a hard per-dump timeout so a
    hung render can't wedge the run. Writes to the final paths (render-one emits the .json
    metadata sidecar itself); a killed dump is caught by `_field_ready` on the next run."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(MT.EXE), "render-one", "--cx", atom["cx"], "--cy", atom["cy"],
           "--fw", atom["fw"], "--family", atom["family"], "--maxiter", str(atom["maxiter"]),
           "--width", str(FIELD_W), "--height", str(FIELD_H), "--supersample", "1",
           "--dump-field-source", "f64", "--dump-field", str(out_bin)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"dump-field {out_bin.name} failed: {r.stderr[-300:]}")


def _crop_coords(atom, box):
    """Normalized field box -> (cx, cy, fw string, fw mpf) for a re-render. Mirrors
    pilot_harvest._crop_coords exactly (row 0 = top = max imag)."""
    cu, cv, wu, wv = box
    fw_field = mp.mpf(atom["fw"])
    aspect = mp.mpf(FIELD_H) / FIELD_W
    cre = mp.mpf(atom["cx"]) + (mp.mpf(cu) - mp.mpf("0.5")) * fw_field
    cim = mp.mpf(atom["cy"]) + (mp.mpf("0.5") - mp.mpf(cv)) * fw_field * aspect
    crop_fw = mp.mpf(wu) * fw_field
    digits = MT.dcf.emit_digits_for_fw(float(crop_fw), guard=EMIT_DIGITS_GUARD)
    return (mp.nstr(cre, digits, strip_zeros=False),
            mp.nstr(cim, digits, strip_zeros=False),
            f"{float(crop_fw):.8e}", crop_fw)


# ======================================================================= #
# STAGE: screen
# ======================================================================= #
def _nms_pick(cands, sep, cap, exclude=()):
    """Greedy elliptical center-separation NMS (same metric as the screen's kept-NMS),
    keeping the highest-G distinct framings clear of each other AND of `exclude`."""
    kept = [dict(c) for c in exclude]
    out = []
    for c in cands:
        clash = False
        for k in kept:
            du = (c["cu"] - k["cu"]) / (0.5 * (c["wu"] + k["wu"]))
            dv = (c["cv"] - k["cv"]) / (0.5 * (c["wv"] + k["wv"]))
            if du * du + dv * dv < sep * sep:
                clash = True
                break
        if not clash:
            out.append(c)
            kept.append(c)
        if len(out) >= cap:
            break
    return out


def _screen_one(job):
    atom, model, cutoff, dump_timeout = job
    aid = atom["id"]
    out_json = SCREEN / f"{aid}.json"
    if out_json.exists():
        return (aid, "cached")
    binp = FIELDS / f"{aid}.bin"
    if not _field_ready(binp):
        _dump_field_timed(atom, binp, dump_timeout)
    field, fw, fh = MT._load_field(binp)
    res = MT.screen_field(field, fw, fh, model, cutoff, assert_once=False)

    # ACCEPTS: the screen's top-4 G-maxima framings at/above the cutoff.
    accepts = [dict(cu=float(c["cu"]), cv=float(c["cv"]), wu=float(c["wu"]),
                    wv=float(c["wv"]), G=round(float(c["G"]), 5), scale=float(c["scale"]))
               for c in res["kept"] if c["G"] >= cutoff]

    # REJECTS: structured near-misses = OOD-surviving windows the screen scored BELOW the
    # cutoff. The top-4 kept peaks alone give ~zero rejects for deep atoms (their peaks are
    # all accepts), which is exactly the "deep == good" confound; drawing sub-cutoff survivors
    # here is what puts deep-and-bland windows in the corpus. NMS'd, highest-G (most
    # structured) first, kept clear of the accepts.
    rej_cands = []
    for s, (boxes, Gs) in res["surv_boxes_by_scale"].items():
        for (cu, cv, wu, wv), g in zip(boxes, np.asarray(Gs)):
            if g < cutoff:
                rej_cands.append(dict(cu=float(cu), cv=float(cv), wu=float(wu),
                                      wv=float(wv), G=round(float(g), 5), scale=float(s)))
    rej_cands.sort(key=lambda c: -c["G"])
    rejects = _nms_pick(rej_cands, MT.HT.SEP, REJECT_CAP, exclude=accepts)

    # OOD-MASKED: a small sample of the (plentiful) v2-pre-filter drops — featureless.
    mb = res["masked_boxes"]
    if len(mb) > MAX_MASKED_PER_ATOM:
        rng = np.random.default_rng(_stable_seed(aid))
        idx = sorted(rng.choice(len(mb), MAX_MASKED_PER_ATOM, replace=False))
        mb = [mb[i] for i in idx]

    rec = dict(atom_id=aid, degree=atom["degree"], period=atom["period"], band=atom["band"],
               split=atom.get("split"), family=atom["family"],
               cx=atom["cx"], cy=atom["cy"], fw=atom["fw"], maxiter=atom["maxiter"],
               cutoff=round(float(cutoff), 5),
               n_masked=int(res["agg"]["n_masked"]), n_surv=int(res["agg"]["n_surv"]),
               accepts=accepts, rejects=rejects,
               masked_boxes=[[float(x) for x in b] for b in mb])
    _atomic_write_json(rec, out_json)
    return (aid, f"acc={len(accepts)} rej={len(rejects)} masked={len(mb)} "
                 f"(n_surv={rec['n_surv']})")


def stage_screen(args):
    atoms = load_admitted()
    FIELDS.mkdir(parents=True, exist_ok=True)
    SCREEN.mkdir(parents=True, exist_ok=True)
    uncached = [a for a in atoms if not (SCREEN / f"{a['id']}.json").exists()]
    todo = uncached[:args.limit] if args.limit else uncached
    print(f"screen: {len(atoms)} admitted atoms, {len(atoms)-len(uncached)} cached, "
          f"{len(todo)} to screen this run. workers={args.workers} "
          f"wall_budget={args.wall_seconds}s", flush=True)
    if not todo:
        print("screen: all atoms cached — nothing to do.")
        return 0
    print("fitting deployment model (unchanged q4_harvest_tight fit) ...", flush=True)
    model, tight = MT._fit_model()
    cutoff = tight["cutoff"]

    t0 = time.time()
    done, stopped_early = 0, False
    # Wall budget: stop SUBMITTING once we can't fit another ~90s unit; let in-flight finish.
    per_unit = 90.0
    jobs = iter([(a, model, cutoff, args.dump_timeout) for a in todo])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for _ in range(min(args.workers, len(todo))):     # prime the pool
            j = next(jobs, None)
            if j is not None:
                futs[ex.submit(_screen_one, j)] = j[0]["id"]
        while futs:
            fut = next(as_completed(list(futs)))
            aid = futs.pop(fut)
            try:
                rid, msg = fut.result()
                done += 1
                print(f"  [{done}/{len(todo)}] {rid} {msg}  ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001 — report + continue, keep resumable
                print(f"  !! {aid} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if args.wall_seconds - (time.time() - t0) > per_unit:
                j = next(jobs, None)
                if j is not None:
                    futs[ex.submit(_screen_one, j)] = j[0]["id"]
            elif not stopped_early:
                stopped_early = True
                print(f"  wall budget {args.wall_seconds}s nearly reached — draining "
                      f"in-flight, not starting new units.", flush=True)
    left = sum(1 for a in atoms if not (SCREEN / f"{a['id']}.json").exists())
    print(f"screen: {done} screened this run, {left} still uncached "
          f"({time.time()-t0:.0f}s). Re-run to resume." if left else
          f"screen: COMPLETE — all {len(atoms)} atoms cached ({time.time()-t0:.0f}s).",
          flush=True)
    return 0


# ======================================================================= #
# STAGE: draw
# ======================================================================= #
def _load_screens():
    recs = [json.loads(p.read_text()) for p in sorted(SCREEN.glob("*.json"))]
    if not recs:
        sys.exit("no screen cache — run the `screen` stage first.")
    return recs


def _atom_framings(rec):
    """(accepts, rejects, masked) crop-records for one atom, coords precomputed."""
    base = dict(atom_id=rec["atom_id"], degree=rec["degree"], period=rec["period"],
                band=rec["band"], split=rec["split"], family=rec["family"])

    def mk(box, fate, G, scale):
        cx, cy, fws, fwm = _crop_coords(rec, box)
        return dict(base, fate=fate, G=G, scale=scale, cx=cx, cy=cy, fw=fws,
                    maxiter=int(MT.dcf._maxiter_for_fw(float(fwm))),
                    box=[float(x) for x in box])
    acc = [mk((f["cu"], f["cv"], f["wu"], f["wv"]), "accepted", f["G"], f["scale"])
           for f in rec["accepts"]]
    rej = [mk((f["cu"], f["cv"], f["wu"], f["wv"]), "rejected", f["G"], f["scale"])
           for f in rec["rejects"]]
    mask = [mk(tuple(b), "ood_masked", None, None) for b in rec["masked_boxes"]]
    return acc, rej, mask


def _rr_take(atoms, cat, A, want, budget, cursor):
    """Round-robin take up to `want` framings of category `cat` ('acc'|'rej'|'mask')
    across `atoms`, one per atom per pass (spreads the draw across atoms), respecting the
    per-arm `budget[atom]` and the per-(atom,cat) `cursor`."""
    sel = []
    progress = True
    while len(sel) < want and progress:
        progress = False
        for aid in atoms:
            if len(sel) >= want:
                break
            if budget[aid] <= 0:
                continue
            pool = A[aid][cat]
            c = cursor[(aid, cat)]
            if c < len(pool):
                sel.append(pool[c])
                cursor[(aid, cat)] += 1
                budget[aid] -= 1
                progress = True
    return sel


def _interleave_by_cell(atoms, A):
    """Order atoms so a round-robin cycles (degree,band) cells -> spreads the draw."""
    bycell = defaultdict(list)
    for aid in atoms:
        bycell[(A[aid]["degree"], A[aid]["band"])].append(aid)
    order, cells = [], sorted(bycell)
    while any(bycell[c] for c in cells):
        for c in cells:
            if bycell[c]:
                order.append(bycell[c].pop(0))
    return order


def stage_draw(args):
    recs = _load_screens()
    A = {}
    for rec in recs:
        acc, rej, mask = _atom_framings(rec)
        A[rec["atom_id"]] = dict(acc=acc, rej=rej, mask=mask,
                                 degree=rec["degree"], band=rec["band"],
                                 period=rec["period"], split=rec["split"])
    print(f"draw: {len(A)} screened atoms "
          f"({sum(1 for a in A.values() if a['split']=='eval')} eval / "
          f"{sum(1 for a in A.values() if a['split']=='train')} train)")

    pos_budget = defaultdict(lambda: PER_ATOM_CAP)
    neg_budget = defaultdict(lambda: PER_ATOM_CAP)
    cur = defaultdict(int)

    eval_atoms = [a for a in A if A[a]["split"] == "eval"]
    train_atoms = [a for a in A if A[a]["split"] == "train"]

    # ---- EVAL slice: period-matched positives vs negatives ----
    eval_by_period = defaultdict(list)
    for a in eval_atoms:
        eval_by_period[A[a]["period"]].append(a)
    eval_pos, eval_neg = [], []
    for p in sorted(eval_by_period):
        ats = eval_by_period[p]
        avail_pos = sum(min(PER_ATOM_CAP, len(A[a]["acc"])) for a in ats)
        avail_neg = sum(min(PER_ATOM_CAP, len(A[a]["rej"])) for a in ats)
        n = min(avail_pos, avail_neg)                 # matched count for this period
        if n == 0:
            continue
        eval_pos += _rr_take(ats, "acc", A, n, pos_budget, cur)
        eval_neg += _rr_take(ats, "rej", A, n, neg_budget, cur)

    # ---- TRAIN positives: fill to POS_TARGET, spread across cells ----
    train_order = _interleave_by_cell(train_atoms, A)
    train_pos = _rr_take(train_order, "acc", A, max(0, POS_TARGET - len(eval_pos)),
                         pos_budget, cur)

    # ---- TRAIN negatives: deep floor + ~10% masked ----
    masked_quota = round(NEG_MASKED_FRAC * NEG_TARGET)
    deep_floor = -(-NEG_TARGET // 2)                  # ceil(0.5*NEG_TARGET)
    deep_train = _interleave_by_cell([a for a in train_atoms if A[a]["band"] in DEEP_BANDS], A)
    nondeep_train = _interleave_by_cell(
        [a for a in train_atoms if A[a]["band"] not in DEEP_BANDS], A)
    eval_deep = sum(1 for x in eval_neg if x["band"] in DEEP_BANDS)

    train_all_order = _interleave_by_cell(train_atoms, A)
    train_neg = []
    # 1. deep near-misses first, to the deep floor.
    train_neg += _rr_take(deep_train, "rej", A, max(0, deep_floor - eval_deep),
                          neg_budget, cur)
    # 2. RESERVE the ~10% masked quota now, before the bulk reject fill eats the per-atom
    #    neg budget (masked shares each atom's budget with rejects).
    train_masked = _rr_take(train_all_order, "mask", A, masked_quota, neg_budget, cur)
    # 3. broaden rejects to fill the remaining reject slots.
    reject_quota = NEG_TARGET - masked_quota - len(eval_neg)
    have_rej = len(train_neg)
    train_neg += _rr_take(nondeep_train, "rej", A, max(0, reject_quota - have_rej),
                          neg_budget, cur)
    train_neg += train_masked
    # 4. top up to NEG_TARGET if pools were thin (any rejects, then any masked).
    short = (NEG_TARGET - len(eval_neg)) - len(train_neg)
    if short > 0:
        train_neg += _rr_take(train_all_order, "rej", A, short, neg_budget, cur)
    short = (NEG_TARGET - len(eval_neg)) - len(train_neg)
    if short > 0:
        train_neg += _rr_take(train_all_order, "mask", A, short, neg_budget, cur)

    pos = [dict(x, arm="positive") for x in (eval_pos + train_pos)]
    neg = [dict(x, arm="negative") for x in (eval_neg + train_neg)]
    allc = pos + neg
    # stable, batch-unique image_id (image_id collides across scale batches by design —
    # the label store joins on coordinates, per test_label_store_join).
    for gi, c in enumerate(allc):
        c["image_id"] = f"mb{gi:04d}_{c['atom_id']}_{c['fate']}"

    # ---- durable draw manifest ----
    draw_path = paths.durable(f"{DRAW_DIR_REL}/draw.jsonl", mkparents=True)
    with open(draw_path, "w", encoding="utf-8") as f:
        for c in allc:
            f.write(json.dumps(c) + "\n")

    # ---- anchor batch's ~8 minibrot picks: accepts spread across DEGREES (round-robin)
    #      and distinct (degree,band) cells, so all four degrees are represented ----
    accepts_all = [c for c in pos if c["fate"] == "accepted"]
    by_deg = defaultdict(list)
    for c in sorted(accepts_all, key=lambda c: (c["band"], -(c["G"] or 0))):
        by_deg[c["degree"]].append(c)
    degs = sorted(by_deg)
    anchor, seen_cell = [], set()
    while len(anchor) < ANCHOR_MINIBROT_N and any(by_deg[d] for d in degs):
        for d in degs:
            if len(anchor) >= ANCHOR_MINIBROT_N:
                break
            while by_deg[d]:                                   # next unseen-cell accept
                c = by_deg[d].pop(0)
                if (c["degree"], c["band"]) not in seen_cell:
                    seen_cell.add((c["degree"], c["band"]))
                    anchor.append(c)
                    break
    for c in accepts_all:                                      # backfill if pools were thin
        if len(anchor) >= ANCHOR_MINIBROT_N:
            break
        if c not in anchor:
            anchor.append(c)
    anchor_path = paths.durable(f"{DRAW_DIR_REL}/anchor_minibrot_picks.jsonl", mkparents=True)
    with open(anchor_path, "w", encoding="utf-8") as f:
        for c in anchor:
            f.write(json.dumps(c) + "\n")

    # ---- corpus batch images.jsonl / batch.json ----
    _write_batch(allc)

    # ---- console summary ----
    deep_neg = sum(1 for c in neg if c["band"] in DEEP_BANDS)
    masked_neg = sum(1 for c in neg if c["fate"] == "ood_masked")
    print(f"  positives: {len(pos)}  (eval {len(eval_pos)} / train {len(train_pos)})")
    print(f"  negatives: {len(neg)}  (eval {len(eval_neg)} / train {len(train_neg)})")
    print(f"    deep-band negatives: {deep_neg}/{len(neg)} "
          f"({100*deep_neg/max(1,len(neg)):.0f}%, floor {NEG_DEEP_FLOOR:.0%})")
    print(f"    OOD-masked negatives: {masked_neg}/{len(neg)} "
          f"({100*masked_neg/max(1,len(neg)):.0f}%, target {NEG_MASKED_FRAC:.0%})")
    print(f"  total crops: {len(allc)}  -> {draw_path}")
    print(f"  anchor minibrot picks: {len(anchor)} -> {anchor_path}")
    return 0


def _palette_names():
    lib = json.load(open(PALETTE_SOURCE, encoding="utf-8"))
    return [e["name"] for e in lib if isinstance(e, dict) and e.get("name")]


def _row_render(c, palette):
    r = cc.render_block(cx=c["cx"], cy=c["cy"], fw=c["fw"], maxiter=c["maxiter"],
                        palette=palette, composition=COMPOSITION, width=CROP_W,
                        height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                        interior_mode=INTERIOR_MODE)
    r["fractal_type"] = c["family"]                   # native multibrot: family in render block
    r["c_re"] = None
    r["c_im"] = None
    return r


def _write_batch(allc):
    names = _palette_names()
    rows = []
    for c in allc:
        pal = names[_stable_seed(c["image_id"]) % len(names)]   # seeded score-3 draw
        render = _row_render(c, pal)
        prov = cc.provenance_block(
            GEN_VERSION, BATCH_ID, family=c["family"],
            selection_role=c["arm"], focus_score=c["G"],
            stratum=c["band"], seed_index=None,
        )
        prov["draw_index"] = None
        # minibrot-specific provenance stashed in existing keys (never fabricated):
        prov["decoded_class"] = c["fate"]             # screen fate: accepted/rejected/ood_masked
        prov["descend_mode"] = f"minibrot_d{c['degree']}_p{c['period']}"
        rows.append(cc.make_row(c["image_id"], render, prov, cc.label_block()))
    bdir = Path(cc.batch_dir(BATCH_ID))
    cc.write_jsonl(rows, str(bdir / "images.jsonl"))
    batch_json = dict(
        schema_version=1, batch_id=BATCH_ID, generator_version=GEN_VERSION,
        created=None, labeler=None,
        source="data/minibrot_roster/roster.jsonl (v2) via the deployed stage-1 screen",
        counts=dict(total=len(rows),
                    positive=sum(1 for c in allc if c["arm"] == "positive"),
                    negative=sum(1 for c in allc if c["arm"] == "negative")),
        render_defaults=dict(width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE, composition=COMPOSITION,
                             palette_roster="data/palettes/score3_colormaps.json",
                             vivid_companion=VIVID_PALETTE,
                             maxiter="per-crop deploy maxiter (dcf._maxiter_for_fw)"),
        render_recipe=cc.render_recipe_stamp(PALETTE_SOURCE),
        sampling_metaparameters=dict(
            screen="q4_multibrot_transfer (deployed stage-1, unchanged)",
            cutoff_rule="q4_harvest_tight tight cutoff (prec>=0.85, n>=12)",
            pos_target=POS_TARGET, neg_target=NEG_TARGET,
            neg_masked_frac=NEG_MASKED_FRAC, neg_deep_floor=NEG_DEEP_FLOOR,
            per_atom_cap_per_arm=PER_ATOM_CAP,
            split="inherited from source atom (roster v2), never reassigned"),
    )
    (bdir / "batch.json").write_text(json.dumps(batch_json, indent=2), encoding="utf-8")
    (bdir / "scores.json").write_text("{}", encoding="utf-8")
    print(f"  batch -> {bdir}  ({len(rows)} rows, images.jsonl + batch.json)")


# ======================================================================= #
# STAGE: render  (canonical crop + vivid companion, resumable, <=4 threads)
# ======================================================================= #
def _render_row(job):
    row, crops_dir, vivid_dir, timeout = job
    iid = row["image_id"]
    render = row["render"]
    made = []
    canon = crops_dir / f"{iid}.jpg"
    if not canon.exists():
        cc.render_corpus_crop(render, str(canon), palette_source=PALETTE_SOURCE, timeout=timeout)
        made.append("canon")
    vivid = vivid_dir / f"{iid}.jpg"
    if not vivid.exists():
        vr = dict(render)
        vr["palette"] = VIVID_PALETTE
        cc.render_corpus_crop(vr, str(vivid), palette_source=PALETTE_SOURCE, timeout=timeout)
        made.append("vivid")
    return iid, made


def stage_render(args):
    from concurrent.futures import ThreadPoolExecutor
    bdir = Path(cc.batch_dir(BATCH_ID))
    rows = cc.read_jsonl(str(bdir / "images.jsonl"))
    crops_dir = bdir / "crops"
    vivid_dir = bdir / "vivid"
    crops_dir.mkdir(parents=True, exist_ok=True)
    vivid_dir.mkdir(parents=True, exist_ok=True)

    def needs(r):
        iid = r["image_id"]
        return not (crops_dir / f"{iid}.jpg").exists() or not (vivid_dir / f"{iid}.jpg").exists()
    todo = [r for r in rows if needs(r)]
    print(f"render: {len(rows)} rows, {len(todo)} need crops (2 renders each: canonical + "
          f"vivid). workers={args.workers} wall_budget={args.wall_seconds}s", flush=True)
    if not todo:
        print("render: all crops present — nothing to do.")
        return 0

    t0 = time.time()
    done = 0
    jobs = iter([(r, crops_dir, vivid_dir, args.render_timeout) for r in todo])
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for _ in range(min(args.workers, len(todo))):
            j = next(jobs, None)
            if j is not None:
                futs[ex.submit(_render_row, j)] = 1
        while futs:
            for fut in list(futs):
                if fut.done():
                    futs.pop(fut)
                    try:
                        iid, made = fut.result()
                        done += 1
                        if done % 25 == 0 or made:
                            print(f"  [{done}/{len(todo)}] {iid} {'+'.join(made) or 'cached'} "
                                  f"({time.time()-t0:.0f}s)", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! render failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    if args.wall_seconds - (time.time() - t0) > 15:
                        j = next(jobs, None)
                        if j is not None:
                            futs[ex.submit(_render_row, j)] = 1
            time.sleep(0.2)
    left = sum(1 for r in rows if needs(r))
    print(f"render: {done} rows this run, {left} still missing crops ({time.time()-t0:.0f}s)."
          if left else f"render: COMPLETE — all {len(rows)} rows have crop + vivid "
          f"({time.time()-t0:.0f}s).", flush=True)
    return 0


# ======================================================================= #
# STAGE: report  (distribution tables + fate-stratified vivid sheet)
# ======================================================================= #
def _dist_table(crops, key):
    out = defaultdict(lambda: defaultdict(int))
    for c in crops:
        out[(c["arm"], c["split"])][c[key]] += 1
    return out


def stage_report(args):
    draw_path = paths.durable(f"{DRAW_DIR_REL}/draw.jsonl")
    crops = [json.loads(l) for l in open(draw_path, encoding="utf-8") if l.strip()]
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit(f"minibrot batch {BATCH_ID} — {len(crops)} crops")
    for arm in ("positive", "negative"):
        for split in ("train", "eval"):
            sub = [c for c in crops if c["arm"] == arm and c["split"] == split]
            if not sub:
                continue
            emit(f"\n[{arm} / {split}]  n={len(sub)}")
            emit("  degree: " + ", ".join(f"d{d}:{n}" for d, n in
                                          sorted(Counter(c["degree"] for c in sub).items())))
            emit("  band:   " + ", ".join(f"{b}:{n}" for b, n in
                                          sorted(Counter(c["band"] for c in sub).items())))
            emit("  period: " + ", ".join(f"p{p}:{n}" for p, n in
                                           sorted(Counter(c["period"] for c in sub).items())))
            emit("  fate:   " + ", ".join(f"{fa}:{n}" for fa, n in
                                          sorted(Counter(c["fate"] for c in sub).items())))

    # deep floor + masked frac
    neg = [c for c in crops if c["arm"] == "negative"]
    deep = sum(1 for c in neg if c["band"] in DEEP_BANDS)
    masked = sum(1 for c in neg if c["fate"] == "ood_masked")
    emit(f"\nnegative arm: deep-band {deep}/{len(neg)} ({100*deep/max(1,len(neg)):.1f}%, "
         f"floor {NEG_DEEP_FLOOR:.0%}); OOD-masked {masked}/{len(neg)} "
         f"({100*masked/max(1,len(neg)):.1f}%, target {NEG_MASKED_FRAC:.0%})")

    # eval period match (positive vs negative period distributions)
    ev_pos = Counter(c["period"] for c in crops if c["arm"] == "positive" and c["split"] == "eval")
    ev_neg = Counter(c["period"] for c in crops if c["arm"] == "negative" and c["split"] == "eval")
    periods = sorted(set(ev_pos) | set(ev_neg))
    npos, nneg = sum(ev_pos.values()), sum(ev_neg.values())
    tv = 0.5 * sum(abs(ev_pos[p] / max(1, npos) - ev_neg[p] / max(1, nneg)) for p in periods)
    emit(f"\neval period match (pos vs neg): TV distance = {tv:.3f}  (0 = identical)")
    emit("  period:  " + ", ".join(f"p{p}[{ev_pos[p]}/{ev_neg[p]}]" for p in periods)
         + "   (pos/neg)")

    # crops-per-atom histogram
    per_atom = Counter(c["atom_id"] for c in crops)
    hist = Counter(per_atom.values())
    emit(f"\ncrops-per-atom: {len(per_atom)} atoms used; histogram "
         + ", ".join(f"{k}crop:{v}atoms" for k, v in sorted(hist.items()))
         + f"  (max {max(per_atom.values())})")

    rep_path = SCR_DIR / "distribution_report.txt"
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {rep_path}")
    _fate_sheet(crops)
    return 0


def _fate_sheet(crops):
    """Fate-stratified VIVID sheet: positives (accepts) above negatives (rejects, masked),
    read from the rendered vivid companions so Matt sees exactly what he judges from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    vivid_dir = Path(cc.batch_dir(BATCH_ID)) / "vivid"
    rng = np.random.default_rng(7)

    def sample(pred, n):
        pool = [c for c in crops if pred(c) and (vivid_dir / f"{c['image_id']}.jpg").exists()]
        rng.shuffle(pool)
        return pool[:n]
    N = 10
    rows = [
        ("POS accept", sample(lambda c: c["arm"] == "positive", N)),
        ("NEG reject", sample(lambda c: c["arm"] == "negative" and c["fate"] == "rejected", N)),
        ("NEG masked", sample(lambda c: c["fate"] == "ood_masked", N)),
        ("POS deep", sample(lambda c: c["arm"] == "positive" and c["band"] in DEEP_BANDS, N)),
        ("NEG deep", sample(lambda c: c["arm"] == "negative" and c["band"] in DEEP_BANDS, N)),
    ]
    fig, axes = plt.subplots(len(rows), N, figsize=(2.0 * N, 1.55 * len(rows) + 1))
    fig.suptitle(f"{BATCH_ID} — fate-stratified (vivid cmr.fusion) · negatives next to "
                 f"positives · POS accept | NEG reject=near-miss | NEG masked=featureless",
                 y=0.997, fontsize=10)
    for ri, (label, items) in enumerate(rows):
        for ci in range(N):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci == 0:
                ax.text(-0.02, 0.5, label, rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=8, weight="bold")
            if ci < len(items):
                c = items[ci]
                ax.imshow(mpimg.imread(vivid_dir / f"{c['image_id']}.jpg"))
                g = f"{c['G']:.2f}" if c["G"] is not None else "—"
                ax.set_title(f"d{c['degree']}p{c['period']} {c['band']} G={g}", fontsize=5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = SCR_DIR / "fate_sheet.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"fate sheet -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--wall-seconds", type=float, default=7200.0)
    p.add_argument("--dump-timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0, help="cap atoms this run (0 = all)")
    p.set_defaults(func=stage_screen)
    pd = sub.add_parser("draw")
    pd.add_argument("--workers", type=int, default=WORKERS)
    pd.set_defaults(func=stage_draw)
    pr = sub.add_parser("render")
    pr.add_argument("--workers", type=int, default=WORKERS)
    pr.add_argument("--wall-seconds", type=float, default=3600.0)
    pr.add_argument("--render-timeout", type=float, default=180.0,
                    help="per-crop hard timeout (raise for deep multibrot5 ss4 crops)")
    pr.set_defaults(func=stage_render)
    prp = sub.add_parser("report")
    prp.add_argument("--workers", type=int, default=WORKERS)
    prp.set_defaults(func=stage_report)
    args = ap.parse_args()
    if args.workers > 4:
        sys.exit("workers capped at 4 (project rule)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
