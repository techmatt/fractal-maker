#!/usr/bin/env python
r"""q4_fate_sheets.py — captioned sheets of the long harvest, stratified BY FATE.

WHY FATE IS THE STRATUM. The run is record-and-rank: it records every candidate above a low
floor with its per-stage fate and Matt picks the cutoff. A sheet of admissions alone would
show him the material his current thresholds already keep, which is the one population that
cannot tell him whether the thresholds are in the right place. So every fate gets equal page
space — `admitted` beside `canon_not_q3` beside `precanon_dup` beside `below_tau_h` — and the
caption names it, because here the label is the information rather than a leak. (This is an
INSPECTION sheet, not a labeling batch: the batches built by
`build_q4_harvest_batches.py` are blind and captionless, and these are the opposite on
purpose.)

TWO RENDERS PER TILE, AND THEY ANSWER DIFFERENT QUESTIONS — inherited verbatim from
`maneuver_inspection_sheet.py`:
  * CANONICAL — `prescreen._render` at 640x360 ss2 `twilight_shifted`, the exact deploy
    presentation the head scores. It is what the SCORE on the caption refers to.
  * VIVID — the committed `blue_orange` map (`data/palettes/vivid_blue_orange.json`), same
    geometry, same map on every tile, so the eye compares STRUCTURE and not palette.
Shown: the vivid. Scored: nothing here — the scores are the ones the run already recorded.
Re-scoring the sheet would measure the sheet's render policy, not the run.

  uv run python tools/atlas/q4_fate_sheets.py --run-dir data/discovery/<run> --per-fate 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
           str(ROOT / "tools" / "scoring"), str(ROOT / "tools" / "mining")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                    # noqa: E402
import prescreen                                # noqa: E402
import steered_frontier as sf                   # noqa: E402  (render_args_for, the fates)

VIVID_PALETTE = "blue_orange"
VIVID_SOURCE = ROOT / "data" / "palettes" / "vivid_blue_orange.json"
WORKERS = 4
RENDER_THREADS = 3
DRAW_SEED = 20260803

# Page order: the fates a cutoff review actually walks, best-understood first. A fate with no
# rows is SKIPPED and NAMED in the header rather than silently omitted — "this run produced
# no guarded rejects" and "the sheet forgot guarded" are different facts.
FATE_ORDER = ("admitted", "q3_dup", "canon_not_q3", "reframe_not_q3", "guarded",
              "precanon_dup", "below_tau_h", "interior_gt_30")


def load_rows(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "q4_candidates.jsonl"
    rows, seen = [], set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["partition"], r["cx"], r["cy"], r["fw"], r.get("julia_c_re"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    return rows


def stratify(rows, per_fate: int, seed: int = DRAW_SEED):
    """`per_fate` rows from each fate, spread across partitions within the fate.

    Within a fate the draw is round-robin over PARTITION and then by descending rank score,
    so one partition cannot own a fate's page and the strongest example of each is shown. The
    rank score is comparable inside a fate because a fate fixes the tier."""
    rng = np.random.default_rng(seed)
    out = []
    for fate in FATE_ORDER:
        sub = [r for r in rows if r["fate"] == fate]
        if not sub:
            continue
        by_part = defaultdict(list)
        for r in sub:
            by_part[r["partition"]].append(r)
        for v in by_part.values():
            v.sort(key=lambda r: -(r.get("rank_score") if r.get("rank_score")
                                   is not None else -1e9))
        keys = sorted(by_part)
        picked, i = [], 0
        while len(picked) < min(per_fate, len(sub)):
            k = keys[i % len(keys)]
            j = i // len(keys)
            if j < len(by_part[k]):
                picked.append(by_part[k][j])
            i += 1
            if i > len(sub) * 2 + len(keys):
                break
        out.extend(picked)
    return out


def render_pair(r, canon_dir: Path, vivid_dir: Path):
    """`(canonical, vivid)` paths, or `(None, None)`. Never raises — a sheet is not a gate."""
    tag = f"{r['fate']}_{r['partition'].replace(':', '_')}_{r['node_id']}"
    # The ADMITTED frame when there is one, the candidate frame otherwise: the sheet must
    # show what the ledger holds, not the pre-reframe view that produced it.
    cx = r.get("outcome_cx") if r.get("outcome_cx") is not None else r["cx"]
    cy = r.get("outcome_cy") if r.get("outcome_cy") is not None else r["cy"]
    fw = r.get("outcome_fw") if r.get("outcome_fw") is not None else r["fw"]
    c6 = None
    if r["partition"] == "phoenix":
        px = r.get("phoenix") or {}
        c6 = (r.get("julia_c_re"), r.get("julia_c_im"),
              px.get("p_re"), px.get("p_im"), px.get("zm1_re"), px.get("zm1_im"))
        if any(v is None for v in c6):
            return None, None       # a phoenix row without its full point is not renderable
    elif r.get("julia_c_re") is not None:
        c6 = (r["julia_c_re"], r["julia_c_im"])
    try:
        args = sf.render_args_for(r["partition"], c6)
    except Exception:
        return None, None
    cp, vp = canon_dir / f"{tag}.jpg", vivid_dir / f"{tag}.jpg"
    ok = True
    if not cp.exists():
        ok, _ = prescreen._render(cx, cy, fw, cp, timeout=180.0, **args)
    if not ok:
        return None, None
    if not vp.exists():
        # Same geometry, the committed vivid map — structure, not palette.
        import subprocess
        from active_ckpt import auto_maxiter
        import location as lm
        loc = lm.Location(family=args["family"], cx=str(cx), cy=str(cy), fw=str(fw),
                          c_re=(args["c"][0] if args["c"] else None),
                          c_im=(args["c"][1] if args["c"] else None),
                          family_params=dict(args.get("family_params") or {}))
        cmd = [str(prescreen.BIN), "render-one", "--cx", str(cx), "--cy", str(cy),
               "--fw", repr(float(fw)), "--width", "640", "--height", "360",
               "--supersample", "2", "--maxiter", str(auto_maxiter(float(fw))),
               "--palette", str(VIVID_SOURCE), "--jpg-quality", "92",
               "--out", str(vp)] + lm.render_one_flags(loc)
        try:
            subprocess.run(cmd, capture_output=True, timeout=180)
        except Exception:
            pass
    return (cp if cp.exists() else None), (vp if vp.exists() else cp)


def build_sheet(items, out_png: Path, title: str, cols: int = 6):
    TW, TH, PAD, LAB, TITLE_H = 384, 216, 6, 34, 40
    rows_n = (len(items) + cols - 1) // cols
    W = cols * (TW + PAD) + PAD
    im = Image.new("RGB", (W, TITLE_H + rows_n * (TH + LAB + PAD) + PAD), (18, 18, 20))
    dr = ImageDraw.Draw(im)
    dr.text((PAD + 2, 12), title, fill=(232, 232, 150))
    for k, (r, vp) in enumerate(items):
        rr, cc_ = divmod(k, cols)
        x, y = PAD + cc_ * (TW + PAD), TITLE_H + PAD + rr * (TH + LAB + PAD)
        tile = (Image.open(vp).convert("RGB").resize((TW, TH)) if vp and Path(vp).exists()
                else Image.new("RGB", (TW, TH), (60, 20, 20)))
        im.paste(tile, (x, y))
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=(30, 30, 34))
        sc = r.get("rank_score")
        dr.text((x + 4, y + TH + 3),
                f"{r['fate']}  {r['partition']}  tier{r.get('rank_tier')}"
                + (f"  score {sc:.3f}" if isinstance(sc, (int, float)) else ""),
                fill=(232, 232, 150))
        dr.text((x + 4, y + TH + 18),
                f"fw {float(r['fw']):.3g}  d{r.get('depth')}"
                + ("  TRIGGERED" if r.get("triggered") else "")
                + (f"  canon {r['canon_decoded']}" if r.get("canon_decoded") else ""),
                fill=(170, 190, 210))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--per-fate", type=int, default=12)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)
    if a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2

    rows = load_rows(a.run_dir)
    picked = stratify(rows, a.per_fate)
    present = Counter(r["fate"] for r in rows)
    missing = [f for f in FATE_ORDER if not present.get(f)]
    print(f"fate sheets: {len(rows)} recorded candidates; {len(picked)} tiles over "
          f"{len({r['fate'] for r in picked})} fates")
    print(f"  population by fate: {json.dumps(dict(present))}")
    if missing:
        # NAMED, never silently omitted.
        print(f"  fates with NO rows this run (skipped, not forgotten): {missing}")

    canon = paths.scratch("q4_fate_sheets", "canon")
    vivid = paths.scratch("q4_fate_sheets", "vivid")
    canon.mkdir(parents=True, exist_ok=True)
    vivid.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r, (cp, vp) in zip(picked, ex.map(
                lambda r: render_pair(r, canon, vivid), picked)):
            items.append((r, vp))
    ok = sum(1 for _, vp in items if vp)
    print(f"  rendered {ok}/{len(items)} tiles in {time.time()-t0:.0f}s "
          f"({len(items)-ok} unrenderable — named in the sheet as red tiles)")

    out = paths.scratch("q4_fate_sheets", "q4_fate_stratified.png")
    p = build_sheet(items, out,
                    f"q4 long harvest {a.run_dir.name} — FATE-STRATIFIED "
                    f"(admissions AND rejects), vivid {VIVID_PALETTE} companion; "
                    f"n={len(items)} over {len({r['fate'] for r, _ in items})} fates")
    print(f"  -> {p}  ({Image.open(p).width}x{Image.open(p).height})")

    # One sheet per fate as well: a cutoff review walks a fate at a time, and a single
    # 8-fate page is too coarse for that.
    by_fate = defaultdict(list)
    for r, vp in items:
        by_fate[r["fate"]].append((r, vp))
    for fate, its in by_fate.items():
        q = build_sheet(its, paths.scratch("q4_fate_sheets", f"fate_{fate}.png"),
                        f"q4 long harvest — fate={fate}  (n={len(its)} of "
                        f"{present[fate]} recorded), vivid {VIVID_PALETTE}")
        print(f"    {fate:16s} -> {q.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
