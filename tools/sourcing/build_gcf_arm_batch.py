#!/usr/bin/env python
r"""The `G_cf` arm — a paired HIGH/LOW interior batch, BOTH arms framed by maximising G_cf.

WHY. The deployed OOD mask's interior clause (`g_interior >= 0.10`) removes 20.2% of every
position the screen sweeps; dropping it would grow the scoreable pool ~50%. It has never
been honestly adjudicated:

  * G-framing physically CANNOT produce a high-interior window — `interior_worst` is G's
    second-largest weight (-1.278) — so the 487 could never populate the band; and
  * the 80-crop interior-band batch that DID populate it framed both arms by uniform-random
    draw, so it tested *uniform vs G-framed*, not *high vs low interior*. Its own report says
    the G_counterfactual gap (-11.2 vs -3.5) is the confound it exists to expose.

`G_cf` — the same objective with the interior clause removed — is the arm nobody has. It was
already computed for every candidate the interior-band sweep kept (`build_interior_band_batch`
records `G_cf` on each reservoir-sampled window), so this batch needs no new sweep.

THE DESIGN.

  * TWO ARMS, ONE OBJECTIVE. Both arms are framed by argmax `G_cf`. The arms differ in ONE
    predicate: HIGH takes `interior_frac >= 0.10` (the currently-rejected band, properly
    framed for the first time), LOW takes `interior_frac < 0.10` (the control).
  * PAIRED PER ATOM, AND WITHIN A SCALE. Each selected atom contributes exactly one window to
    EACH arm, so the contrast is WITHIN-atom — the case-control structure the 487's ICC 0.68
    established as the right one (the atom explains ~2/3 of label variance, so an unpaired
    contrast spends most of its power on atom identity), and exactly what the 80-crop batch
    lacked. The two windows of a pair additionally share a window SCALE: an unconstrained
    argmax `G_cf` picks a different window size in each arm (measured on the first draw: HIGH
    7/8/45 vs LOW 22/10/28 at scale 0.06/0.09/0.14), and size is plainly visible to the
    labeler, so the arms would have differed in two things. Pairing within a scale keeps the
    objective identical — still argmax `G_cf`, now argmax within the pair's scale — and
    leaves interior mass as the only difference.
  * TRAIN-SIDE ONLY, SPLIT INHERITED. Atoms are drawn from the roster's train-side atoms and
    each row carries the atom-level split as assigned at roster build time. Nothing is
    reassigned.
  * DESIGNED CONTRAST => BIASED, TRAIN-SIDE. Registered as its own window-level batch under
    `data/label_corpus/batches/`; `tools/v7/build_manifest.assign_split` classifies it
    biased->train through its FAIL-CLOSED default, with no edit to any registration list.
    `verify` asserts that rather than assuming it.

HONEST LIMITS (both reported by `verify`, neither hideable):
  1. `G_cf` is maximised over the interior-band sweep's RESERVOIR SAMPLE (<=24 uniformly
     drawn windows per (atom, band, scale)), not over every swept position. It is a
     sample-argmax, and the two arms are sampled identically, so the CONTRAST is sound while
     the absolute framing is weaker than a full-grid argmax would be.
  2. The sweep never cached a window with `interior_frac >= 0.50` (`band_of` returns None
     there), so the HIGH arm is [0.10, 0.50), not [0.10, 1.0].

PRESENTATION / BLINDING — as in the class-3 revisit chunks:
  * opaque `image_id` `gc<slot>_<hash>` encoding nothing (no arm, no atom, no degree);
  * a seeded presentation shuffle assigning the slot;
  * a BLINDED served manifest `blind.jsonl` (provenance reduced to batch identity), so the
    arm is not in the bytes the browser fetches under any reveal state. The full-provenance
    `images.jsonl` is the analysis-side file;
  * canonical crop + the VIVID `blue_orange` companion, colormaps read straight off the
    committed library (`data/palettes/{score3_colormaps,vivid_blue_orange}.json`).

NOT QUEUED FOR LABELING. Built, verified, parked. It competes with the class-3 revisit for
the only scarce resource here and the revisit comes first.

Stages:
  draw     pair every selected atom's argmax-G_cf HIGH and LOW window -> durable manifest +
           the corpus batch (images.jsonl / blind.jsonl / batch.json). Seconds.
  render   canonical + vivid crop for every row. ~240 renders; ONE worker (the machine is
           labeling), BELOW_NORMAL priority via corpus_common's engine default.
  verify   acceptance: arm separation, atom-by-atom pairing, blind safety, batch
           registration + biased/train classification. Seconds.

  uv run python tools/sourcing/build_gcf_arm_batch.py draw
  uv run python tools/sourcing/build_gcf_arm_batch.py render [--workers 1]
  uv run python tools/sourcing/build_gcf_arm_batch.py verify

Reads:  scratch/interior_band_batch/cand/<atom>.json   (G_cf already computed there)
        data/minibrot_roster/roster.jsonl
Writes: data/minibrot_roster/gcf_arm_v1/draw.jsonl                       (durable manifest)
        data/label_corpus/batches/2026-07-28_gcf_arm_v1/{images,blind}.jsonl, batch.json
        scratch/gcf_arm_batch/{report.txt,arm_sheet.png}                 (regenerable)

NOTHING DEPLOYED IS CHANGED. No cutoff, screen, mask, draw rule, config or production
feature is touched; `q4_stage1_linear_fit` / `q4_multibrot_transfer` are imported read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "tools", "corpus"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                            # noqa: E402
import corpus_common as cc                              # noqa: E402
import build_minibrot_batch as BMB                      # noqa: E402 (coords / palettes / io)
import build_interior_band_batch as IBB                 # noqa: E402 (the G_cf candidate cache)
from tools.studies import q4_stage1_linear_fit as LF    # noqa: E402 (deployed screen, read-only)
from tools.studies import q4_multibrot_transfer as MT   # noqa: E402 (read-only)

BATCH_ID = "2026-07-28_gcf_arm_v1"
GEN_VERSION = "gcf_arm_v1"
PRESENTATION_SEED = 0x6CF001                   # UI blind-shuffle seed, recorded in batch.json
DRAW_SEED = 20260728                           # atom-selection seed

# The arms. `interior_frac` is the SCREEN-resolution in-set fraction `g_interior` — the exact
# quantity the deployed mask cuts on, and the quantity the interior-band batch proved is a
# +0.999 Spearman proxy for the crop-resolution interior fraction the labeler sees.
INTERIOR_CUT = LF.V2_INTERIOR                  # 0.10 — read off the deployed screen, not set here
ARM_HIGH, ARM_LOW = "gcf_high_interior", "gcf_low_interior"
N_ATOMS = 60                                   # ~60 atoms x 2 arms = ~120 crops = ~240 renders
DEGREES = (2, 3, 4, 5)

CROP_W, CROP_H, CROP_SS = BMB.CROP_W, BMB.CROP_H, BMB.CROP_SS
CROP_FILTER, INTERIOR_MODE, COMPOSITION = BMB.CROP_FILTER, BMB.INTERIOR_MODE, BMB.COMPOSITION
PALETTE_SOURCE = BMB.PALETTE_SOURCE            # data/palettes/score3_colormaps.json
VIVID_PALETTE, VIVID_SOURCE = BMB.VIVID_PALETTE, BMB.VIVID_SOURCE

SCR = paths.scratch("gcf_arm_batch")
CAND = IBB.CAND                                # scratch/interior_band_batch/cand
DIR_REL = "data/minibrot_roster/gcf_arm_v1"
DRAW_REL = f"{DIR_REL}/draw.jsonl"


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def arm_of(gi: float) -> str:
    """Which arm a candidate belongs to. The ONLY predicate that differs between the arms."""
    return ARM_HIGH if gi >= INTERIOR_CUT else ARM_LOW


# ======================================================================= #
# STAGE: draw
# ======================================================================= #
def _candidates(rec):
    """Flatten one atom's cached reservoir into a list of candidate dicts, tagged by arm."""
    out = []
    for key, lst in rec["cands"].items():
        _band, _, s = key.partition("|")
        for c in lst:
            out.append(dict(c, scale=float(s), arm=arm_of(c["gi"])))
    return out


def _pair_one_atom(rec):
    """(high, low, rule) — the atom's argmax-`G_cf` window in each arm, AT A SHARED WINDOW
    SCALE.

    Why scale is matched. Framing both arms by an unconstrained argmax `G_cf` lets the
    objective pick a different window SIZE in each arm (measured: HIGH skewed to 0.14 and LOW
    to 0.06), and window size is plainly visible to the labeler — so the arms would differ in
    two things, not one, and the batch would repeat the confound it exists to remove. Pairing
    within a scale keeps "both arms framed by maximising G_cf" exactly (the objective is still
    argmax G_cf, now argmax *within the pair's scale*) while leaving interior mass as the only
    difference.

    Which scale. The one whose HIGH-arm best `G_cf` is largest — i.e. the scale at which the
    currently-rejected band is at its strongest, since that arm is the one being framed
    properly for the first time. Scales without a usable window in BOTH arms are skipped.

    The LOW pick clears the screen's own elliptical separation against the HIGH pick
    (`MT.HT.SEP`, the metric the deployed framing's NMS uses) so a pair is never two views of
    the same picture; a step down the LOW ranking is recorded, not hidden."""
    by_scale = defaultdict(list)
    for c in _candidates(rec):
        by_scale[c["scale"]].append(c)
    best = None
    for s, cands in by_scale.items():
        hi = sorted([c for c in cands if c["arm"] == ARM_HIGH], key=lambda c: -c["G_cf"])
        lo = sorted([c for c in cands if c["arm"] == ARM_LOW], key=lambda c: -c["G_cf"])
        if not hi or not lo:
            continue
        h = hi[0]
        for rank, l in enumerate(lo):
            if not IBB._clash(l, [h], MT.HT.SEP):
                rule = f"scale{s}" + ("" if rank == 0 else f"+sep(rank{rank})")
                if best is None or h["G_cf"] > best[0]["G_cf"]:
                    best = (h, l, rule)
                break
    if best is None:
        return None, None, ("empty_arm" if not by_scale else "no_scale_with_both_arms")
    return best


def stage_draw(args):
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CAND.glob("*.json"))]
    if not recs:
        sys.exit(f"no G_cf candidate cache at {CAND} — this batch reuses the interior-band "
                 f"sweep's already-computed G_cf; run that sweep first.")
    # TRAIN-SIDE ONLY, split inherited from the roster and never reassigned.
    train = [r for r in recs if r["split"] == "train"]
    print(f"draw: {len(recs)} swept atoms, {len(train)} train-side "
          f"({len(recs) - len(train)} eval atoms excluded by design)")

    # Atom selection: seeded, degree-balanced. It is blind to G_cf and to interior — selecting
    # atoms on either would make the paired contrast a selected one.
    rng = np.random.default_rng(DRAW_SEED)
    by_deg = defaultdict(list)
    for r in train:
        by_deg[r["degree"]].append(r)
    per_deg = args.n_atoms // len(DEGREES)
    chosen = []
    for d in DEGREES:
        pool = sorted(by_deg[d], key=lambda r: r["atom_id"])
        idx = rng.permutation(len(pool))[:per_deg]
        chosen += [pool[i] for i in sorted(idx)]

    rows, skipped = [], []
    for rec in chosen:
        h, l, how = _pair_one_atom(rec)
        if h is None:
            skipped.append((rec["atom_id"], how))
            continue
        for c, arm in ((h, ARM_HIGH), (l, ARM_LOW)):
            cx, cy, fws, fwm = BMB._crop_coords(rec, tuple(c["box"]))
            rows.append(dict(
                atom_id=rec["atom_id"], degree=rec["degree"], period=rec["period"],
                period_band=rec["period_band"], split=rec["split"], family=rec["family"],
                arm=arm, pair_rule=how,
                box=[float(x) for x in c["box"]], scale=float(c["scale"]),
                interior_frac=float(c["gi"]), g_flat=float(c["gflat"]),
                g_speckle=float(c["gspeck"]),
                clauses=IBB.clauses_of(c["gi"], c["gflat"], c["gspeck"]),
                G_cf=float(c["G_cf"]),
                cx=cx, cy=cy, fw=fws, maxiter=int(MT.dcf._maxiter_for_fw(float(fwm)))))
    if skipped:
        print(f"  {len(skipped)} atom(s) dropped (no pairable window): {skipped[:5]}")

    # Opaque image_id: a seeded shuffle assigns the slot, the suffix is a content hash of the
    # window. Nothing in the filename — the one identifier that reaches the browser as a URL —
    # encodes arm, interior, atom or degree.
    order = list(range(len(rows)))
    np.random.default_rng(PRESENTATION_SEED).shuffle(order)
    for slot, oi in enumerate(order):
        c = rows[oi]
        h = BMB._stable_seed(f"{c['atom_id']}|{c['arm']}|{c['box']}|{c['scale']}")
        c["image_id"] = f"gc{slot:04d}_{h:08x}"
    rows.sort(key=lambda c: c["image_id"])

    dp = paths.durable(DRAW_REL, mkparents=True)
    with open(dp, "w", encoding="utf-8") as f:
        for c in rows:
            f.write(json.dumps(c) + "\n")
    _write_batch(rows)

    n_atoms = len({c["atom_id"] for c in rows})
    print(f"  drawn: {len(rows)} crops over {n_atoms} atoms "
          f"({len(rows)//2} pairs) -> {len(rows) * 2} renders")
    for arm in (ARM_HIGH, ARM_LOW):
        sub = [c for c in rows if c["arm"] == arm]
        gi = np.array([c["interior_frac"] for c in sub])
        g = np.array([c["G_cf"] for c in sub])
        print(f"    {arm:<18} n={len(sub):3d}  interior_frac med={np.median(gi):.4f} "
              f"[{gi.min():.4f},{gi.max():.4f}]   G_cf med={np.median(g):+.3f}")
    print(f"  -> {dp}")
    return 0


def _write_batch(rows_in):
    names = BMB._palette_names()
    full, blind = [], []
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
            stratum=(f"{c['arm']}|interior_frac"
                     f"{'>=' if c['arm'] == ARM_HIGH else '<'}{INTERIOR_CUT}"),
            interior_frac=c["interior_frac"],
            focus_score=c["G_cf"],                  # the framing objective, for the analysis
            decoded_class="+".join(c["clauses"]) or "unmasked",
            descend_mode=f"minibrot_d{c['degree']}_p{c['period']}",
        )
        full.append(cc.make_row(c["image_id"], render, prov, cc.label_block()))
        # browser-side row: batch identity only. `arm`, `interior_frac`, `focus_score` and
        # `family` are absent entirely — not even as null keys — so the arm is not in the
        # served bytes, the DOM or JS memory under any reveal state (the revisit-chunk rig).
        blind.append(cc.make_row(c["image_id"], dict(render),
                                 {"generator_version": GEN_VERSION, "batch_id": BATCH_ID},
                                 cc.label_block()))
    bdir = Path(cc.batch_dir(BATCH_ID))
    bdir.mkdir(parents=True, exist_ok=True)
    cc.write_jsonl(full, str(bdir / "images.jsonl"))
    cc.write_jsonl(blind, str(bdir / "blind.jsonl"))
    hi = [c for c in rows_in if c["arm"] == ARM_HIGH]
    bj = dict(
        schema_version=1, batch_id=BATCH_ID, generator_version=GEN_VERSION,
        created=None, labeler=None,
        presentation_seed=PRESENTATION_SEED,
        vivid_companion=VIVID_PALETTE,
        served_manifest="blind.jsonl",
        queued_for_labeling=False,
        purpose=("paired two-arm within-atom contrast, BOTH arms framed by maximising G_cf "
                 "(the deployed objective with the interior clause removed), differing only "
                 "in interior mass: HIGH interior_frac >= %.2f (the currently-rejected band, "
                 "properly framed for the first time) vs LOW < %.2f (control). BUILT AND "
                 "PARKED — not queued for labeling." % (INTERIOR_CUT, INTERIOR_CUT)),
        counts=dict(total=len(full), pairs=len(hi),
                    **{ARM_HIGH: len(hi), ARM_LOW: len(rows_in) - len(hi)}),
        arms={ARM_HIGH: [INTERIOR_CUT, None], ARM_LOW: [0.0, INTERIOR_CUT]},
        render_defaults=dict(width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE, composition=COMPOSITION,
                             palette_roster="data/palettes/score3_colormaps.json",
                             vivid_companion=VIVID_PALETTE,
                             maxiter="per-crop deploy maxiter (dcf._maxiter_for_fw)"),
        render_recipe=cc.render_recipe_stamp(PALETTE_SOURCE),
        sampling_metaparameters=dict(
            framing="argmax G_cf within the arm, at the pair's shared window scale "
                    "(both arms, one objective)",
            selection_predicate=f"interior_frac vs the deployed ceiling {INTERIOR_CUT}",
            g_used_for_selection=True,
            pairing="one window per arm per atom, both at the SAME window scale "
                    "(within-atom, within-scale case-control); the shared scale is the one "
                    "whose HIGH-arm best G_cf is largest",
            candidate_universe=("the interior-band sweep's reservoir sample: <=24 uniformly "
                                "drawn windows per (atom, band, scale) over the deployed "
                                "swept grid, so G_cf is a SAMPLE-argmax, identically sampled "
                                "in both arms"),
            high_arm_ceiling=("0.50 — the sweep never cached interior_frac >= 0.50 "
                              "(band_of returns None there)"),
            atom_selection=f"seeded, degree-balanced over train-side roster atoms "
                           f"(seed {DRAW_SEED}); blind to G_cf and to interior",
            split="inherited from the source roster atom, never reassigned; train-side only",
            separation="LOW clears MT.HT.SEP against HIGH (the screen's own NMS metric)",
            draw_seed=DRAW_SEED),
    )
    (bdir / "batch.json").write_text(json.dumps(bj, indent=2), encoding="utf-8")
    if not (bdir / "scores.json").exists():
        (bdir / "scores.json").write_text("{}", encoding="utf-8")
    print(f"  batch -> {bdir}  ({len(full)} rows; images.jsonl + blind.jsonl + batch.json)")


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
    print(f"render: {len(rows)} rows, {len(todo)} need crops (canonical + vivid) = "
          f"{2 * len(todo)} renders. workers={args.workers} "
          f"(engine runs BELOW_NORMAL — the machine stays responsive for labeling)", flush=True)
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
                    el = time.time() - t0
                    eta = (len(todo) - done) * el / max(done, 1)
                    print(f"  [{done}/{len(todo)}] {iid} {'+'.join(made) or 'cached'} "
                          f"({el:.0f}s, ETA {eta/60:.1f} min)", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  !! {futs[fut]['image_id']} FAILED: {type(e).__name__}: "
                      f"{str(e)[:200]}", flush=True)
    left = sum(1 for r in rows if needs(r))
    print(f"render: {done} this run, {left} still missing ({time.time()-t0:.0f}s)." if left
          else f"render: COMPLETE — all {len(rows)} rows ({time.time()-t0:.0f}s).", flush=True)
    return 0


# ======================================================================= #
# STAGE: verify — the acceptance criteria, checked not asserted in prose
# ======================================================================= #
def stage_verify(args):
    rows = _read_jsonl(paths.durable(DRAW_REL))
    bdir = Path(cc.batch_dir(BATCH_ID))
    L, ok = [], True

    def emit(s=""):
        L.append(s)
        print(s)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"    [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    H = [r for r in rows if r["arm"] == ARM_HIGH]
    Lo = [r for r in rows if r["arm"] == ARM_LOW]
    emit(f"G_cf arm batch {BATCH_ID} — {len(rows)} crops, {len(rows)//2} within-atom pairs")
    emit(f"both arms framed by argmax G_cf; the ONLY differing predicate is interior_frac "
         f"vs the deployed ceiling {INTERIOR_CUT}.")

    # --- 1. the arms' interior_frac distributions genuinely separate ---------
    emit("\n[1] ARM SEPARATION on interior_frac (the one thing that differs)")
    gh = np.array([r["interior_frac"] for r in H])
    gl = np.array([r["interior_frac"] for r in Lo])
    emit(f"    {'arm':<20}{'n':>4}{'min':>10}{'median':>10}{'max':>10}{'mean':>10}")
    for nm, v in ((ARM_HIGH, gh), (ARM_LOW, gl)):
        emit(f"    {nm:<20}{len(v):>4}{v.min():>10.4f}{np.median(v):>10.4f}"
             f"{v.max():>10.4f}{v.mean():>10.4f}")
    check("arms do not overlap", gl.max() < gh.min(),
          f"LOW max {gl.max():.4f} < HIGH min {gh.min():.4f}")
    check(f"HIGH is entirely in the rejected band (>= {INTERIOR_CUT})", (gh >= INTERIOR_CUT).all())
    check(f"LOW is entirely below the ceiling (< {INTERIOR_CUT})", (gl < INTERIOR_CUT).all())
    # AUC of interior_frac as an arm separator: 1.0 means a clean cut
    sep = float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in gl] for a in gh]))
    check("separation AUC == 1.0", sep == 1.0, f"AUC={sep:.4f}")

    # What maximising G_cf does INSIDE each arm — two consequences that are inherent to the
    # brief ("both arms framed by maximising G_cf"), not build errors, but load-bearing for
    # how the batch can be read.
    emit("\n    WHERE IN THE REJECTED BAND THE HIGH ARM LANDS")
    emit("    (the interior-band batch's own strata, for comparison — it stratified to cover "
         "the band; this batch lets the objective choose)")
    for lo, hi in ((INTERIOR_CUT, 0.20), (0.20, 0.35), (0.35, 0.50)):
        n = sum(1 for v in gh if lo <= v < hi)
        emit(f"      [{lo:.2f},{hi:.2f}): {n:3d}/{len(gh)}")
    emit(f"    CONSEQUENCE 1: G_cf still penalises interior, so within the HIGH arm the "
         f"argmax runs to the BOTTOM EDGE of the rejected band — "
         f"{sum(1 for v in gh if v < 0.20)}/{len(gh)} rows sit in [{INTERIOR_CUT:.2f}, 0.20), "
         f"median {np.median(gh):.4f}, p90 {np.percentile(gh, 90):.4f}. This batch therefore "
         f"adjudicates the band's LOWER LIP, not [0.10, 0.50] as a whole. The interior-band "
         f"batch's stratified arm still covers the deeper bands; the two are complementary, "
         f"and neither alone answers the clause.")
    fh = np.array([r["g_flat"] for r in H])
    fl = np.array([r["g_flat"] for r in Lo])
    emit(f"\n    CONSEQUENCE 2: the LOW control is materially FLATTER — g_flat median "
         f"{np.median(fl):.3f} vs {np.median(fh):.3f}, tripping the deployed flat clause "
         f"{sum(1 for r in Lo if 'flat' in r['clauses'])}/{len(Lo)} vs "
         f"{sum(1 for r in H if 'flat' in r['clauses'])}/{len(H)}. Maximising G_cf under "
         f"interior < {INTERIOR_CUT} lands on smooth exterior gradient washes (visible on "
         f"the arm sheet). So the contrast a labeler would score is closer to "
         f"'interior-rich vs empty' than to 'high vs low interior at matched richness'. That "
         f"is what the objective does, not a sampler fault — but it is the reason to read a "
         f"HIGH win as 'better than what G_cf picks without interior', NOT as 'interior "
         f"mass helps'.")

    emit("\n    the framing objective, per arm (recorded — G_cf is what BOTH arms maximise)")
    for nm, sub in ((ARM_HIGH, H), (ARM_LOW, Lo)):
        g = np.array([r["G_cf"] for r in sub])
        emit(f"    {nm:<20}G_cf min={g.min():+8.3f} median={np.median(g):+8.3f} "
             f"max={g.max():+8.3f}")
    emit("    (a G_cf gap between arms is the FINDING this batch exists to measure — G_cf "
         "still carries interior-correlated terms; it is not a confound in the design, "
         "because the objective is identical and the pairing is within-atom.)")

    # --- 2. pairing verified atom by atom ------------------------------------
    emit("\n[2] PAIRING — every atom contributes exactly one window to each arm")
    by_atom = defaultdict(Counter)
    for r in rows:
        by_atom[r["atom_id"]][r["arm"]] += 1
    bad = {a: dict(c) for a, c in by_atom.items()
           if c[ARM_HIGH] != 1 or c[ARM_LOW] != 1 or sum(c.values()) != 2}
    check(f"all {len(by_atom)} atoms are exact 1+1 pairs", not bad, str(list(bad.items())[:3]))
    check("atom count == crops / 2", len(by_atom) * 2 == len(rows))
    emit(f"    atoms per degree: "
         f"{dict(sorted(Counter(r['degree'] for r in H).items()))} (HIGH) / "
         f"{dict(sorted(Counter(r['degree'] for r in Lo).items()))} (LOW)")
    check("degree composition is identical across arms",
          Counter(r["degree"] for r in H) == Counter(r["degree"] for r in Lo))
    check("period composition is identical across arms",
          Counter(r["period"] for r in H) == Counter(r["period"] for r in Lo))
    emit(f"    pair rule: {dict(Counter(r['pair_rule'] for r in rows))}")
    emit("    (`scale<s>` = both windows are their arm's top-G_cf window at scale s, the "
         "scale where the HIGH arm's G_cf is best; `+sep(rankK)` = the LOW pick stepped K "
         "down its ranking to clear HT.SEP against its partner.)")
    from tools.studies.interior_bakeoff import _iou
    pair_iou = []
    for a, rs in [(a, [r for r in rows if r["atom_id"] == a]) for a in by_atom]:
        if len(rs) == 2:
            pair_iou.append(_iou(rs[0]["box"], rs[1]["box"]))
    check("no pair is two views of one picture (IoU < 0.25)", max(pair_iou) < 0.25,
          f"max pair IoU {max(pair_iou):.3f}")
    emit(f"    scale mix: HIGH {dict(sorted(Counter(r['scale'] for r in H).items()))}  "
         f"LOW {dict(sorted(Counter(r['scale'] for r in Lo).items()))}")
    scale_by_atom = defaultdict(set)
    for r in rows:
        scale_by_atom[r["atom_id"]].add(r["scale"])
    check("every pair shares one window scale (size is not a second difference)",
          all(len(s) == 1 for s in scale_by_atom.values()),
          str([(a, sorted(s)) for a, s in scale_by_atom.items() if len(s) > 1][:3]))
    check("scale mix is identical across arms",
          Counter(r["scale"] for r in H) == Counter(r["scale"] for r in Lo))
    emit("    (unconstrained argmax G_cf picks a different window SIZE per arm — measured "
         "HIGH 7/8/45 vs LOW 22/10/28 at 0.06/0.09/0.14 — and size is plainly visible to the "
         "labeler. Pairing WITHIN a scale keeps the objective identical and leaves interior "
         "mass the only difference.)")
    emit("    CAVEAT, recorded: the shared scale is chosen by the objective, and G_cf prefers "
         "the widest window, so this batch's scale mix (7/8/45) is far from the 487's "
         "realized 422/50/15. That is fine for the WITHIN-batch paired contrast this batch "
         "is for, and it means a cross-batch comparison against the 487 would be confounded "
         "by window scale. Do not pool them.")

    # --- 3. split ------------------------------------------------------------
    emit("\n[3] SPLIT — train-side only, inherited from the roster atom")
    check("every row is train-side", all(r["split"] == "train" for r in rows),
          str(dict(Counter(r["split"] for r in rows))))

    # --- 4. blind safety, the revisit-chunk check ----------------------------
    emit("\n[4] BLIND SAFETY (the check used on the revisit chunks)")
    import re
    id_ok = all(re.fullmatch(r"gc\d{4}_[0-9a-f]{8}", r["image_id"]) for r in rows)
    check("image_id is opaque `gc<slot>_<hash>`", id_ok)
    # Substring test only for tokens that cannot arise by accident. `d<degree>` is NOT one of
    # them: `d` and the digits are hex, so an 8-hex-digit content hash contains e.g. "d2" with
    # p ~ 7/256 regardless of the row's degree — flagging that is a false positive that would
    # also make the check fail at random on any rebuild. Degree is covered structurally below.
    leaks = [(r["image_id"], k) for r in rows
             for k in (r["arm"], r["atom_id"], r["family"]) if k in r["image_id"]]
    check("no arm / atom / family token in any image_id", not leaks, str(leaks[:3]))
    check("image_ids are unique", len({r["image_id"] for r in rows}) == len(rows))

    # Structural: the slot is shuffle-assigned, so sorting by image_id — the natural order
    # anything downstream falls into, including the labeling UI — must not block up by arm or
    # by degree. Draw order would give k runs; a shuffle gives ~n*(1-1/k).
    def _no_block_structure(name, key):
        vals = [key(r) for r in sorted(rows, key=lambda r: r["image_id"])]
        k = len(set(vals))
        runs = 1 + sum(1 for a, b in zip(vals, vals[1:]) if a != b)
        check(f"id order carries no {name} block structure",
              runs > 0.5 * len(vals) * (1 - 1 / k),
              f"{runs} runs over {len(vals)} rows, {k} values (draw order would give {k})")
    _no_block_structure("arm", lambda r: r["arm"])
    _no_block_structure("degree", lambda r: r["degree"])
    # and a pair's two windows must not land in adjacent slots, which would make the pairing
    # itself visible as "the same atom twice in a row".
    order = {r["image_id"]: i for i, r in enumerate(sorted(rows, key=lambda r: r["image_id"]))}
    slots = defaultdict(list)
    for r in rows:
        slots[r["atom_id"]].append(order[r["image_id"]])
    adjacent = [a for a, v in slots.items() if len(v) == 2 and abs(v[0] - v[1]) == 1]
    check("no pair sits in adjacent presentation slots", not adjacent, str(adjacent[:3]))
    if (bdir / "blind.jsonl").exists():
        blind = _read_jsonl(bdir / "blind.jsonl")
        served = (bdir / "blind.jsonl").read_text(encoding="utf-8")
        LEAK_KEYS = ("selection_role", "stratum", "interior_frac", "focus_score",
                     "decoded_class", "descend_mode", "family")
        check("served manifest carries no leak key at all",
              not any(f'"{k}"' in served for k in LEAK_KEYS),
              str([k for k in LEAK_KEYS if f'"{k}"' in served]))
        check("no arm string anywhere in the served bytes",
              ARM_HIGH not in served and ARM_LOW not in served)
        check("served manifest covers every row",
              {r["image_id"] for r in blind} == {r["image_id"] for r in rows})
        check("served manifest labels are null", all(r["label"]["score"] is None for r in blind))
        bj = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))
        check("batch.json names blind.jsonl as the served manifest",
              bj.get("served_manifest") == "blind.jsonl")
        check("batch.json records that this batch is NOT queued for labeling",
              bj.get("queued_for_labeling") is False)
    else:
        check("blind.jsonl exists", False)

    # --- 5. registration + classification ------------------------------------
    emit("\n[5] REGISTRATION — own window-level batch, classified biased -> train")
    full = _read_jsonl(bdir / "images.jsonl")
    check("batch dir under data/label_corpus/batches", bdir.exists() and
          str(bdir).replace("\\", "/").endswith(f"data/label_corpus/batches/{BATCH_ID}"))
    check("every analysis-side row carries arm + interior_frac + G_cf in provenance",
          all(r["provenance"].get("selection_role") in (ARM_HIGH, ARM_LOW)
              and r["provenance"].get("interior_frac") is not None
              and r["provenance"].get("focus_score") is not None for r in full))
    check("every label is null (nothing is labeled)",
          all(r["label"]["score"] is None for r in full))
    from tools.v7 import build_manifest as bm
    split, biased, source = bm.assign_split({"batch": BATCH_ID, "ft": "mandelbrot"})
    emit(f"    assign_split({BATCH_ID!r}) -> {(split, biased, source)!r}")
    check("classified TRAIN-side", split == "train")
    check("classified BIASED", biased is True)
    check("classified through the FAIL-CLOSED default, with no registration-list edit",
          source == "unregistered"
          and BATCH_ID not in bm.CENSUS_BATCHES | bm.BAND_BATCHES | bm.UNBIASED_TRAIN_BATCHES
          and BATCH_ID != bm.BLINDSPOT_BATCH)
    import label_store as ls
    check("not registered in label_store.SIDECAR_LABELS (unlabeled, nothing to join)",
          BATCH_ID not in ls.SIDECAR_LABELS)
    check("no contradiction with label_store's biased registry",
          BATCH_ID not in ls.TRAIN_SIDE_ONLY_BATCHES or biased is True)

    # --- 6. crops ------------------------------------------------------------
    emit("\n[6] CROPS")
    nc = sum(1 for r in rows if (bdir / "crops" / f"{r['image_id']}.jpg").exists())
    nv = sum(1 for r in rows if (bdir / "vivid" / f"{r['image_id']}.jpg").exists())
    check(f"canonical crops rendered", nc == len(rows), f"{nc}/{len(rows)}")
    check(f"vivid companions rendered", nv == len(rows), f"{nv}/{len(rows)}")

    emit(f"\n[7] MASK CLAUSES (recorded, never selected on)")
    for nm, sub in ((ARM_HIGH, H), (ARM_LOW, Lo)):
        emit(f"    {nm:<20}" + ", ".join(
            f"{k}={v}" for k, v in Counter("+".join(r["clauses"]) or "unmasked"
                                           for r in sub).most_common()))

    emit(f"\nACCEPTANCE: {'ALL CHECKS PASS' if ok else 'FAILURES ABOVE'}")
    emit("PARKED — not queued for labeling; the class-3 revisit has the labeling budget.")
    rp = SCR / "report.txt"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {rp}")
    if nc == len(rows) and nv == len(rows):
        _arm_sheet(rows)
    return 0 if ok else 1


def _arm_sheet(rows):
    """Paired sheet: one column per atom, HIGH on top, LOW beneath — the within-atom contrast
    read straight down the page. Vivid companions, sorted by degree then interior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    vivid = Path(cc.batch_dir(BATCH_ID)) / "vivid"
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r["atom_id"]][r["arm"]] = r
    items = [p for p in pairs.values() if ARM_HIGH in p and ARM_LOW in p]
    items.sort(key=lambda p: (p[ARM_HIGH]["degree"], -p[ARM_HIGH]["interior_frac"]))
    if not items:
        return
    COLS = 15
    blocks = [items[i:i + COLS] for i in range(0, len(items), COLS)]
    nrow = 2 * len(blocks)
    fig, axes = plt.subplots(nrow, COLS, figsize=(1.55 * COLS, 1.0 * nrow + 1), squeeze=False)
    fig.suptitle(f"{BATCH_ID} — paired G_cf arms (vivid {VIVID_PALETTE}). Column = one atom; "
                 f"TOP = HIGH interior (>= {INTERIOR_CUT}), BOTTOM = LOW control. "
                 f"Caption = d<degree> · interior_frac · G_cf.", y=0.995, fontsize=9)
    for bi, block in enumerate(blocks):
        for ai, arm in enumerate((ARM_HIGH, ARM_LOW)):
            ri = 2 * bi + ai
            for ci in range(COLS):
                ax = axes[ri][ci]
                ax.axis("off")
                if ci == 0:
                    ax.text(-0.05, 0.5, "HIGH" if ai == 0 else "LOW", rotation=90,
                            va="center", ha="right", transform=ax.transAxes,
                            fontsize=7, weight="bold")
                if ci < len(block):
                    r = block[ci][arm]
                    p = vivid / f"{r['image_id']}.jpg"
                    if p.exists():
                        ax.imshow(mpimg.imread(p))
                    ax.set_title(f"d{r['degree']} · {r['interior_frac']:.3f} · "
                                 f"{r['G_cf']:+.1f}", fontsize=5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    out = SCR / "arm_sheet.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"arm sheet -> {out}")


# ======================================================================= #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--n-atoms", type=int, default=N_ATOMS)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=stage_draw)
    p = sub.add_parser("render")
    p.add_argument("--workers", type=int, default=1,
                   help="ONE by default: the machine is labeling. Capped at 4 (project rule).")
    p.add_argument("--render-timeout", type=float, default=300.0)
    p.set_defaults(func=stage_render)
    p = sub.add_parser("verify")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=stage_verify)
    args = ap.parse_args()
    if args.workers > 4:
        sys.exit("workers capped at 4 (project rule)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
