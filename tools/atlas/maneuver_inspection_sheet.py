#!/usr/bin/env python
r"""maneuver_inspection_sheet.py — captioned sheets of maneuver-originated views, for the eye.

WHAT THIS IS AND IS NOT. An INSPECTION sheet, not a labeling batch and not an evaluation.
It exists so Matt can look at what the operators actually propose across the whole richness
range — including, deliberately, the bottom quintile. Captions are wanted here (a labeling
batch would hide them). Nothing here decides a threshold, and no number below is readable as
maneuver YIELD: the active head has never been trained on maneuver-originated views
(`docs/design/minibrot_maneuvers.md` §9), so `p_good` on this population measures the
CLASSIFIER against a population it has never seen, in both directions.

THE SAMPLE. Stratified by `radial_range` QUINTILE x OPERATOR over every available,
successfully screened candidate the run enumerated — pushed or passed over. Passed-over
candidates are views too (they carry cx/cy/fw), and excluding them would quietly restrict the
population to what selection already liked, which is the winner's-curse mechanism one level
up. Deduped on (atom_key, k): the same nucleus at two framings is two views, the same
(nucleus, framing) reached twice is one.

TWO RENDERS PER TILE, and the reason they are different renders:
  * CANONICAL — 640x360 ss2 `twilight_shifted` through `prescreen._render`, i.e. the exact
    deploy search presentation the head scores. Scored, never shown.
  * VIVID — the committed `blue_orange` map (`data/palettes/vivid_blue_orange.json`), same
    geometry, same map on every tile so the eye compares STRUCTURE and not palette. Shown,
    never scored.
Scoring the vivid render or showing the canonical one would each answer a different question
than the one being asked.

  uv run python tools/atlas/maneuver_inspection_sheet.py \
      --run-dir data/discovery/<run> --n 100
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "mining",
           ROOT / "tools" / "scoring", ROOT / "tools" / "explorer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tools import run_record            # noqa: E402  (segments-aware run-record layer)

import paths                                    # noqa: E402
import prescreen                                # noqa: E402
import render_core as rc                        # noqa: E402
import corpus_common as cc                      # noqa: E402
from score_lib import Scorer, corn_decode       # noqa: E402
from active_ckpt import ACTIVE_CKPT, auto_maxiter   # noqa: E402

VIVID_PALETTE = "blue_orange"
VIVID_SOURCE = ROOT / "data" / "palettes" / "vivid_blue_orange.json"
N_QUINTILES = 5
# One engine process per worker, so this is the CLAUDE.md concurrent-PROCESS cap.
WORKERS = 4
RENDER_THREADS = 3
TW, TH, PAD, LAB = 384, 216, 6, 46          # tile 16:9 + a two-line caption strip
BG, STRIP, INK, DIM = (16, 16, 18), (30, 30, 34), (236, 236, 200), (150, 170, 200)


# --------------------------------------------------------------------------- #
# population
# --------------------------------------------------------------------------- #
def load_population(logs: list[Path]) -> list[dict]:
    """Every available + screened maneuver candidate, deduped on `(atom_key, k)`.

    The maneuver log is APPEND-ONLY and a kill replays a batch, so it is a superset of the
    checkpointed counters — reading it undeduped double-counts (the correction
    `bench_lateral_seeding.load_cases` already carries)."""
    seen, out = set(), []
    for log in logs:
        for r in run_record.iter_rows(log):     # segments-aware (maneuvers.jsonl rotates)
            if not r.get("available") or r.get("op") == "probe":
                continue
            sc = r.get("screen") or {}
            if not sc.get("screened"):
                continue
            key = (r.get("atom_key"), str(r.get("k")))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(
                run=log.parent.name, op=r["op"], k=r.get("k"),
                cx=r["cx"], cy=r["cy"], fw=float(r["fw"]),
                partition=r.get("partition") or "mandelbrot",
                degree=(r.get("extra") or {}).get("degree"),
                period=r.get("period"), atom_key=r.get("atom_key"),
                window_scale=r.get("window_scale"),
                log10_abs_A=r.get("log10_abs_A"),
                parent_depth=r.get("parent_depth"),
                radial_range=float(sc["radial_range"]),
                radial_rings=float(sc["radial_rings"]),
                interior_fraction=sc.get("interior_fraction"),
                cap_headroom=sc.get("cap_headroom"), clamped=sc.get("clamped"),
                screen_policy=sc.get("maxiter_policy_token"),
                used=bool(r.get("used")), unused_reason=r.get("unused_reason"),
                # v1.5 VIEW-screen columns. All `.get`, so a v1.4 log (atom screen, 4x
                # frame) loads unchanged with these as None — which is the honest reading:
                # that run has no composite, rather than a composite of zero. `screen_frame`
                # is what tells the two apart downstream, and its absence means 4x.
                screen_frame=sc.get("screen_frame") or ("view" if "composite" in sc
                                                        else "atom4x"),
                composite=sc.get("composite"), vetoed=sc.get("vetoed"),
                size_factor=sc.get("size_factor"),
                band_coverage=sc.get("band_coverage"),
                band_coverage_q25=sc.get("band_coverage_q25"),
                view_fw=sc.get("view_fw"),
            ))
    return out


def assign_quintiles(pop: list[dict]) -> list[float]:
    """Quintile EDGES of `radial_range` over the whole population, and the per-row index.

    Quintiles of the run's own distribution, because absolute ring scores are comparable
    only within one (geometry, cap policy) pair (`orbital_field_metrics.md` §5, §7) — and
    this whole population is one such pair by construction."""
    v = np.array([r["radial_range"] for r in pop], dtype=float)
    edges = [float(np.percentile(v, 100.0 * i / N_QUINTILES))
             for i in range(1, N_QUINTILES)]
    for r in pop:
        q = 0
        for e in edges:
            if r["radial_range"] > e:
                q += 1
        r["quintile"] = min(q, N_QUINTILES - 1)
    return edges


def stratify(pop: list[dict], n_target: int, seed: int) -> list[dict]:
    """~`n_target` tiles over the (quintile x operator) grid.

    Every non-empty cell gets a FLOOR before any cell gets a second tile, so the bottom
    quintile and the rarest operator are represented by construction rather than by luck —
    the point of the sheet is to show clearly bad material, not to flatter the feature. The
    remainder is allocated largest-cell-first."""
    rng = random.Random(seed)
    cells: dict = defaultdict(list)
    for r in pop:
        cells[(r["quintile"], r["op"])].append(r)
    for c in cells.values():
        rng.shuffle(c)
    keys = sorted(cells)
    if not keys:
        return []
    floor = max(1, min(n_target // max(1, len(keys)), min(len(cells[k]) for k in keys)))
    take = {k: min(floor, len(cells[k])) for k in keys}
    # remainder, largest cell first, so a big cell is not truncated to the floor
    while sum(take.values()) < n_target:
        cand = [k for k in keys if take[k] < len(cells[k])]
        if not cand:
            break
        k = max(cand, key=lambda k: len(cells[k]) - take[k])
        take[k] += 1
    out = []
    for k in keys:
        out.extend(cells[k][:take[k]])
    return out


# --------------------------------------------------------------------------- #
# renders
# --------------------------------------------------------------------------- #
def render_pair(row: dict, canon_dir: Path, vivid_dir: Path, tag: str) -> dict:
    """The canonical (scored) and vivid (shown) renders for one view. Never raises."""
    fam = row["partition"]
    canon = canon_dir / f"{tag}.jpg"
    vivid = vivid_dir / f"{tag}.jpg"
    err = ""
    if not canon.exists():
        ok, err = prescreen._render(row["cx"], row["cy"], row["fw"], canon,
                                    family=fam, timeout=300)
        if not ok:
            return dict(canon=None, vivid=None, render_error=err[:160])
    if not vivid.exists():
        argv = rc.render_one_argv(row["cx"], row["cy"], row["fw"],
                                  auto_maxiter(float(row["fw"])),
                                  prescreen.RENDER_W, prescreen.RENDER_H,
                                  prescreen.RENDER_SS, VIVID_PALETTE, VIVID_SOURCE, vivid,
                                  family=(None if fam == "mandelbrot" else fam))
        try:
            rc.run_render_one(argv, vivid, threads=RENDER_THREADS)
        except Exception as e:                       # a missing companion is not fatal:
            return dict(canon=canon, vivid=None,     # the tile still scores, it just shows
                        render_error=f"vivid: {str(e)[:140]}")
    return dict(canon=canon, vivid=vivid, render_error=err[:160] if err else "")


def render_all(rows: list[dict], canon_dir: Path, vivid_dir: Path) -> None:
    canon_dir.mkdir(parents=True, exist_ok=True)
    vivid_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = [0]

    def one(i_row):
        i, row = i_row
        res = render_pair(row, canon_dir, vivid_dir, row["tag"])
        row.update(res)
        done[0] += 1
        if done[0] % 10 == 0:
            el = time.time() - t0
            print(f"  {done[0]}/{len(rows)} rendered  {el:.0f}s "
                  f"({done[0]/max(1e-9, el):.2f} tile/s)", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, enumerate(rows)))


def score_all(rows: list[dict], ckpt: str) -> None:
    """Canonical p_* per tile through the deploy transform, and the CANONICAL decode.

    `corn_decode` at its default thresholds — the discovery sites' per-degree `t_good` knee
    is a SELECTION operating point and would make the decoded class here mean something
    other than "what the canonical decode calls this"."""
    todo = [r for r in rows if r.get("canon")]
    if not todo:
        return
    sc = Scorer(ckpt)
    print(f"  scoring {len(todo)} canonical tiles with {ckpt} (K={sc.k})", flush=True)
    res = sc.score_paths_k([r["canon"] for r in todo])
    for r, t in zip(todo, res):
        score, probs = t[0], list(t[1:])
        r["e_ord"] = round(float(score), 4)
        r["p_notbad"] = round(float(probs[0]), 4)
        r["p_good"] = round(float(probs[1]), 4) if len(probs) > 1 else None
        r["p_great"] = round(float(probs[2]), 4) if len(probs) > 2 else None
        r["decoded_class"] = corn_decode(probs[0], probs[1] if len(probs) > 1 else 0.0,
                                         p_great=(probs[2] if len(probs) > 2 else None))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def spearman(a, b) -> float | None:
    """Spearman rho with average ranks for ties. Hand-rolled to keep this tool free of a
    scipy import it needs for one number."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3:
        return None

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(x.size, dtype=float)
        r[order] = np.arange(1, x.size + 1, dtype=float)
        # average ranks within tie groups, or ties bias rho toward the input order
        xs = x[order]
        i = 0
        while i < xs.size:
            j = i
            while j + 1 < xs.size and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return round(float((ra * rb).sum() / den), 4) if den > 0 else None


def _dist(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), min=round(float(v.min()), 4),
                p25=round(float(np.percentile(v, 25)), 4),
                median=round(float(np.median(v)), 4),
                p75=round(float(np.percentile(v, 75)), 4),
                max=round(float(v.max()), 4), mean=round(float(v.mean()), 4))


def readout(rows: list[dict], edges, pop_n: int, ckpt: str) -> dict:
    scored = [r for r in rows if r.get("p_good") is not None]
    lo = [r for r in scored if r["quintile"] == 0]
    hi = [r for r in scored if r["quintile"] == N_QUINTILES - 1]
    out = dict(
        checkpoint=ckpt, population_n=pop_n, sampled_n=len(rows), scored_n=len(scored),
        render_failures=sum(1 for r in rows if not r.get("canon")),
        quintile_edges=[round(e, 4) for e in edges],
        spearman_pgood_radial_range=spearman([r["p_good"] for r in scored],
                                             [r["radial_range"] for r in scored]),
        spearman_pgood_radial_rings=spearman([r["p_good"] for r in scored],
                                             [r["radial_rings"] for r in scored]),
        spearman_eord_radial_range=spearman([r["e_ord"] for r in scored],
                                            [r["radial_range"] for r in scored]),
        p_good_bottom_quintile=_dist([r["p_good"] for r in lo]),
        p_good_top_quintile=_dist([r["p_good"] for r in hi]),
        decoded_class_counts=dict(Counter(r["decoded_class"] for r in scored).most_common()),
        by_cell=[], by_operator=[],
    )
    for q in range(N_QUINTILES):
        sel = [r for r in scored if r["quintile"] == q]
        out["by_cell"].append(dict(quintile=q, n=len(sel),
                                   radial_range=_dist([r["radial_range"] for r in sel]),
                                   p_good=_dist([r["p_good"] for r in sel]),
                                   classes=dict(Counter(r["decoded_class"]
                                                        for r in sel).most_common())))
    for op in sorted({r["op"] for r in scored}):
        sel = [r for r in scored if r["op"] == op]
        out["by_operator"].append(dict(op=op, n=len(sel),
                                       radial_range=_dist([r["radial_range"] for r in sel]),
                                       p_good=_dist([r["p_good"] for r in sel])))
    out["CAVEAT"] = (
        "NOT an evaluation of maneuver yield. The active head has never been trained on "
        "maneuver-originated views, so p_good here measures the CLASSIFIER on an unseen "
        "population, not the population's quality (minibrot_maneuvers.md §9). The "
        "correlation is an eyeball check on whether the head's low end coincides with "
        "visually-bad minibrot views — read it beside the sheets, never instead of them.")
    return out


# --------------------------------------------------------------------------- #
# sheets
# --------------------------------------------------------------------------- #
def build_sheet(items: list[dict], title: str, out_png: Path, cols: int) -> None:
    n = len(items)
    rows_n = (n + cols - 1) // cols
    W = cols * (TW + PAD) + PAD
    TITLE_H = 34
    H = TITLE_H + rows_n * (TH + LAB + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 2, 9), title, fill=INK)
    for i, it in enumerate(items):
        r, c = divmod(i, cols)
        x, y = PAD + c * (TW + PAD), TITLE_H + PAD + r * (TH + LAB + PAD)
        vp = it.get("vivid")
        im = (Image.open(vp).convert("RGB").resize((TW, TH)) if vp and Path(vp).exists()
              else Image.new("RGB", (TW, TH), (70, 22, 22)))
        sheet.paste(im, (x, y))
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=STRIP)
        k = "keep" if it["k"] is None else f"k{it['k']:g}"
        l1 = (f"{it['op'].replace('_to_sibling','').replace('_to_nucleus','')} {k}  "
              f"d{it.get('degree')} p{it.get('period')}  fw={it['fw']:.3g}")
        pg = "n/a" if it.get("p_good") is None else f"{it['p_good']:.3f}"
        l2 = (f"range={it['radial_range']:.2f} rings={it['radial_rings']:.0f}  "
              f"p_good={pg} cls={it.get('decoded_class','-')}  Q{it['quintile']+1}")
        dr.text((x + 4, y + TH + 4), l1, fill=INK)
        dr.text((x + 4, y + TH + 24), l2, fill=DIM)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"  wrote {out_png.name}  ({n} tiles, {W}x{H})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="target tiles")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--model", type=str, default=ACTIVE_CKPT)
    ap.add_argument("--out", type=Path, default=paths.scratch("maneuver_inspection"))
    a = ap.parse_args()

    logs = [Path(d) / "maneuvers.jsonl" for d in a.run_dir]
    pop = load_population(logs)
    print(f"[pop] {len(pop)} available+screened maneuver candidates "
          f"from {len(logs)} run(s)")
    if len(pop) < N_QUINTILES * 2:
        print("[pop] too few screened candidates to stratify — refusing to report "
              "quintiles off a population this thin")
        return 1
    edges = assign_quintiles(pop)
    sample = stratify(pop, a.n, a.seed)
    for i, r in enumerate(sample):
        r["tag"] = f"q{r['quintile']}_{r['op'][:4]}_{i:03d}"
    grid = Counter((r["quintile"], r["op"]) for r in sample)
    print(f"[sample] {len(sample)} tiles over {len(grid)} (quintile x operator) cells; "
          f"range quintile edges {[round(e, 3) for e in edges]}")
    for q in range(N_QUINTILES):
        cells = {op: grid[(q, op)] for (qq, op) in grid if qq == q}
        print(f"    Q{q+1}: {cells}")

    a.out.mkdir(parents=True, exist_ok=True)
    render_all(sample, a.out / "canonical", a.out / "vivid")
    score_all(sample, a.model)

    rep = readout(sample, edges, len(pop), a.model)
    (a.out / "readout.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (a.out / "sample.jsonl").write_text(
        "".join(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                            for k, v in r.items()}) + "\n" for r in sample),
        encoding="utf-8")

    by_q = defaultdict(list)
    for r in sorted(sample, key=lambda r: (r["quintile"], r["op"], -r["radial_range"])):
        by_q[r["quintile"]].append(r)
    for q, items in sorted(by_q.items()):
        build_sheet(items,
                    f"maneuver inspection — radial_range quintile Q{q+1}/{N_QUINTILES} "
                    f"(n={len(items)}, vivid blue_orange; captions are canonical scores)",
                    a.out / f"sheet_q{q+1}.png", cols=4)
    build_sheet(sorted(sample, key=lambda r: r["radial_range"]),
                f"maneuver inspection — ALL {len(sample)} tiles, ascending radial_range",
                a.out / "sheet_all_by_range.png", cols=6)

    print(f"\n  Spearman(p_good, radial_range) = {rep['spearman_pgood_radial_range']} "
          f"over n={rep['scored_n']}")
    print(f"  p_good bottom quintile: {rep['p_good_bottom_quintile']}")
    print(f"  p_good top quintile:    {rep['p_good_top_quintile']}")
    print(f"  decoded classes: {rep['decoded_class_counts']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
