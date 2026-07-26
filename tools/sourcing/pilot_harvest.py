#!/usr/bin/env python
"""Pilot crop harvest off the durable minibrot roster (~40 crops).

Draws at the EDGES of the roster so the pilot tests the two things the roster build
introduced — the period banding and the `A`-margin feasibility cut — rather than the
easy middle:

  * every degree present;
  * the shallowest and deepest *filled* band per degree;
  * admitted atoms sitting close to the 1-decade margin boundary (smallest headroom);
  * feasibility-EXCLUDED atoms that are still close enough to the boundary to render
    (retained rows in the roster) — so the sheet shows what the cut removed.

Each drawn atom's f64 field is rendered and run through the EXISTING stage-1 screen +
OOD mask + G-maxima framing (imported read-only from the closed study
`tools/studies/q4_multibrot_transfer` — reuse, never edited), so screen parity with the
deployed harvest is exact. The pilot crop set is split ~half high-G accepts / half the
reject / OOD-masked side, and the sheet is fate-stratified and VIVID (blue/orange), so
good sourcing can't be thrown away by a pass-only sheet under a crushing palette.

Retained (durable): each crop's re-render coords (cx/cy/fw) + its source atom's
train/eval split + fate, so accepted crops roll into the full run without re-drawing.
Sheet + crop JPGs are regenerable views -> scratch/.

Run:  uv run python tools/sourcing/pilot_harvest.py
Reads:   data/minibrot_roster/roster.jsonl
Writes:  data/minibrot_roster/pilot/manifest.jsonl  (durable)
         scratch/minibrot_roster/pilot/{fields,crops,pilot_sheet.png}
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
sys.path.insert(0, _ROOT)

import mpmath as mp                                    # noqa: E402
import paths                                           # noqa: E402
import build_minibrot_roster as RB                     # noqa: E402
from tools.studies import q4_multibrot_transfer as MT  # noqa: E402 (read-only reuse)

FIELD_W, FIELD_H = MT.W, MT.H                           # 2176 x 1224
OUT = paths.scratch("minibrot_roster", "pilot")
FIELDS = OUT / "fields"
CROPS = OUT / "crops"
SHEET = OUT / "pilot_sheet.png"
MANIFEST = "data/minibrot_roster/pilot/manifest.jsonl"

TARGET_CROPS = 40
N_EXCLUDED_ATOMS = 4                                    # near-boundary rejects to render
EMIT_DIGITS_GUARD = 12


# --------------------------------------------------------------------------- #
# Edge draw.
# --------------------------------------------------------------------------- #
def load_roster():
    rpath = paths.durable(RB.ROSTER_PATH)
    if not rpath.exists():
        sys.exit(f"roster not found: {RB.ROSTER_PATH} — run build_minibrot_roster.py first")
    rows = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
    return rows


def pick_edge_atoms(rows):
    """Select edge atoms: shallow+deep filled band per degree (with the near-boundary
    admitted atom in each), plus the near-boundary feasibility-excluded atoms."""
    admitted = [r for r in rows if r["admitted"]]
    excluded = [r for r in rows if not r["admitted"]]
    by_cell = defaultdict(list)
    for r in admitted:
        by_cell[(r["degree"], r["band"])].append(r)

    chosen, seen = [], set()

    def take(atom, tag):
        if atom["id"] not in seen:
            seen.add(atom["id"])
            a = dict(atom); a["draw_tag"] = tag
            chosen.append(a)

    degrees = sorted({r["degree"] for r in admitted})
    for deg in degrees:
        filled_bands = sorted({r["band"] for r in admitted if r["degree"] == deg},
                              key=lambda b: RB.BANDS.index(_band(b)))
        if not filled_bands:
            continue
        edge_bands = [filled_bands[0]] + ([filled_bands[-1]] if len(filled_bands) > 1 else [])  # noqa
        for bi, band in enumerate(edge_bands):
            cell = sorted(by_cell[(deg, band)], key=lambda r: r["log10_abs_A"])
            edge = "shallow-band" if bi == 0 else "deep-band"
            take(cell[0], f"{edge}:shallowest")                # widest-margin end
            take(cell[-1], f"{edge}:near-margin-boundary")     # smallest-margin end

    # near-boundary feasibility-excluded (largest deploy margin among the excluded, i.e.
    # just below the 1-decade cut), across all degrees, field-renderable.
    excl = sorted(excluded, key=lambda r: -r["f64_margin_deploy_decades"])
    for r in excl[:N_EXCLUDED_ATOMS]:
        take(r, "feasibility-excluded:near-boundary")
    return chosen


def _band(tag):
    lo, hi = tag.split("-")
    return (int(lo), int(hi))


# --------------------------------------------------------------------------- #
# Render + screen each drawn atom.
# --------------------------------------------------------------------------- #
def render_and_screen(atoms, model, cutoff):
    FIELDS.mkdir(parents=True, exist_ok=True)
    results = []
    for i, a in enumerate(atoms):
        b = FIELDS / f"{a['id']}.bin"
        if not b.exists():
            ts = time.time()
            MT._dump_field(a["cx"], a["cy"], a["fw"], a["maxiter"], a["family"], b)
            dt = f"{time.time()-ts:.1f}s"
        else:
            dt = "cached"
        field, fw, fh = MT._load_field(b)
        res = MT.screen_field(field, fw, fh, model, cutoff, assert_once=False)
        results.append((a, res, field, fw, fh))
        n_acc = sum(1 for c in res["kept"] if c["G"] >= cutoff)
        print(f"  [{i+1}/{len(atoms)}] {a['id']} {a['draw_tag']:32s} "
              f"kept={len(res['kept'])} acc={n_acc} masked={res['agg']['n_masked']} ({dt})",
              flush=True)
    return results


def _crop_coords(atom, box):
    """Map a normalized field box -> (cx, cy, fw) decimal strings for a re-render.
    Field: row 0 = top = max imag; dc_re=(u-0.5)*fw, dc_im=(0.5-v)*fw*(H/W)."""
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


def build_crops(results, cutoff):
    """Turn screened framings into per-crop records (balanced ~half accept / half
    reject+ood). Returns (crop_records, per_atom_sheet_items)."""
    accepts, rejects, oods = [], [], []
    srng = np.random.default_rng(0)
    for a, res, field, fw, fh in results:
        for c in res["kept"]:
            box = (c["cu"], c["cv"], c["wu"], c["wv"])
            cx, cy, cfw, cfw_mpf = _crop_coords(a, box)
            rec = dict(atom_id=a["id"], degree=a["degree"], period=a["period"],
                       band=a["band"], atom_admitted=a["admitted"],
                       split=a.get("split"), draw_tag=a["draw_tag"],
                       fate=("accepted" if c["G"] >= cutoff else "rejected"),
                       feasibility_excluded=(not a["admitted"]),
                       G=round(float(c["G"]), 4), scale=c["scale"],
                       cx=cx, cy=cy, fw=cfw,
                       maxiter=MT.dcf._maxiter_for_fw(float(cfw_mpf)),
                       family=a["family"], box=list(box))
            (accepts if rec["fate"] == "accepted" else rejects).append(rec)
        # OOD-masked window samples (from the screen's own masked boxes)
        mb = res["masked_boxes"]
        if mb:
            take = srng.choice(len(mb), size=min(3, len(mb)), replace=False)
            for j in take:
                box = tuple(mb[j])
                cx, cy, cfw, cfw_mpf = _crop_coords(a, box)
                oods.append(dict(atom_id=a["id"], degree=a["degree"], period=a["period"],
                                 band=a["band"], atom_admitted=a["admitted"],
                                 split=a.get("split"), draw_tag=a["draw_tag"],
                                 fate="ood_masked", feasibility_excluded=(not a["admitted"]),
                                 G=None, scale=None, cx=cx, cy=cy, fw=cfw,
                                 maxiter=MT.dcf._maxiter_for_fw(float(cfw_mpf)),
                                 family=a["family"], box=list(box)))

    # balance: ~half accepts, ~half (reject + ood). Keep all if scarce.
    half = TARGET_CROPS // 2
    srng.shuffle(accepts); srng.shuffle(rejects); srng.shuffle(oods)
    sel_acc = accepts[:half]
    neg_pool = rejects + oods
    srng.shuffle(neg_pool)
    sel_neg = neg_pool[:TARGET_CROPS - len(sel_acc)]
    crops = sel_acc + sel_neg
    print(f"\ncrops: {len(sel_acc)} accepts + {len(sel_neg)} reject/ood "
          f"= {len(crops)} (pool: {len(accepts)} acc / {len(rejects)} rej / {len(oods)} ood)")
    return crops


# --------------------------------------------------------------------------- #
# Fate-stratified VIVID sheet (blue/orange). Rows: accepted / rejected /
# OOD-masked / feasibility-excluded-atom framings.
# --------------------------------------------------------------------------- #
def build_sheet(results, cutoff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = MT._blue_orange()
    rng = np.random.default_rng(1)
    field_by_id = {a["id"]: (field, fw, fh) for a, _, field, fw, fh in results}

    accepted, rejected, ood, excluded = [], [], [], []
    for a, res, field, fw, fh in results:
        is_excl = not a["admitted"]
        for c in res["kept"]:
            crop = MT._crop_box(field, fw, fh, (c["cu"], c["cv"], c["wu"], c["wv"]))
            cap = f"{a['id']} G={c['G']:.2f}"
            if is_excl:
                excluded.append((crop, f"{cap} [excl m={a['f64_margin_deploy_decades']:.2f}]"))
            elif c["G"] >= cutoff:
                accepted.append((crop, cap))
            else:
                rejected.append((crop, cap))
        for box in res["masked_boxes"][:4]:
            ood.append((MT._crop_box(field, fw, fh, tuple(box)), f"{a['id']} masked"))

    rows = [("accepted (G>=cutoff)", accepted),
            ("rejected (survived, G<cutoff)", rejected),
            ("OOD-masked (v2 pre-filter)", ood),
            ("feasibility-EXCLUDED atoms", excluded)]
    N_COL = 8
    fig, axes = plt.subplots(len(rows), N_COL, figsize=(2.1 * N_COL, 1.7 * len(rows) + 1))
    fig.suptitle(f"minibrot-roster PILOT — stage-1 fate  |  cutoff G>={cutoff:.2f}  |  "
                 f"vivid blue/orange (fields shown regardless of screen verdict)",
                 y=0.995, fontsize=11)
    for ri, (label, items) in enumerate(rows):
        rng.shuffle(items)
        for ci in range(N_COL):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci == 0:
                ax.text(-0.10, 0.5, label, rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=8, weight="bold")
            if ci < len(items):
                crop, cap = items[ci]
                if crop.size and min(crop.shape[:2]) >= 2:
                    ax.imshow(MT._colorize(crop, cmap))
                    ax.set_title(cap, fontsize=5.5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(SHEET, dpi=130)
    plt.close(fig)
    print(f"sheet -> {SHEET.relative_to(_ROOT)}  "
          f"(acc {len(accepted)} / rej {len(rejected)} / ood {len(ood)} / excl {len(excluded)})")


def render_crop_jpgs(crops):
    """Render each retained crop to a JPG at deploy geometry (1280x720 ss4) so Matt
    eyeballs wallpaper-quality, not the field colorize. Regenerable -> scratch/."""
    CROPS.mkdir(parents=True, exist_ok=True)
    import subprocess
    for i, c in enumerate(crops):
        out = CROPS / f"{i:02d}_{c['atom_id']}_{c['fate']}.png"
        cmd = [str(MT.EXE), "render-one", "--cx", c["cx"], "--cy", c["cy"],
               "--fw", c["fw"], "--family", c["family"], "--maxiter", str(c["maxiter"]),
               "--width", "1280", "--height", "720", "--supersample", "4",
               "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  WARN crop {out.name} failed: {r.stderr[-200:]}", flush=True)
    print(f"crops -> {CROPS.relative_to(_ROOT)}  ({len(crops)} JPGs at 1280x720 ss4)")


def write_manifest(crops):
    mpath = paths.durable(MANIFEST, mkparents=True)
    with open(mpath, "w") as fh:
        for c in crops:
            fh.write(json.dumps(c) + "\n")
    n_acc = sum(1 for c in crops if c["fate"] == "accepted")
    print(f"-> pilot manifest (durable): {MANIFEST}  "
          f"({len(crops)} crops, {n_acc} accepted; split+fate+coords retained)")


def main():
    print("pilot harvest: drawing edges off the durable roster\n")
    rows = load_roster()
    atoms = pick_edge_atoms(rows)
    print(f"edge draw: {len(atoms)} atoms "
          f"({sum(1 for a in atoms if a['admitted'])} admitted / "
          f"{sum(1 for a in atoms if not a['admitted'])} feasibility-excluded)")
    for a in atoms:
        print(f"    {a['id']:16s} d{a['degree']} p{a['period']:>2} band {a['band']:>5} "
              f"m={a['f64_margin_deploy_decades']:+.2f} split={a.get('split')}  {a['draw_tag']}")

    print("\nfitting deployment model (cached corpus fields) ...", flush=True)
    model, tight = MT._fit_model()
    cutoff = tight["cutoff"]

    print("\nrendering + screening drawn fields (f64 dump-field @ 2176x1224):", flush=True)
    results = render_and_screen(atoms, model, cutoff)

    crops = build_crops(results, cutoff)
    build_sheet(results, cutoff)
    render_crop_jpgs(crops)
    write_manifest(crops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
