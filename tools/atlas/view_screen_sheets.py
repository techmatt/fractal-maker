#!/usr/bin/env python
r"""view_screen_sheets.py — sheets and the old-vs-new readout for the view-level screen.

Three artifacts, for Matt's eye and for the record:

  * **quintile sheets** — the new composite's top and bottom quintiles, stratified across
    (operator x degree) inside the quintile so a sheet cannot be one operator's showreel.
    Same vivid `blue_orange` map on every tile, so the eye compares STRUCTURE and not
    palette (`maneuver_inspection_sheet.py`, same reason).
  * **before/after framing pairs** — each top-composite candidate's original frame beside
    the window `view_frame_sweep` chose. This is the demonstrative artifact: it is the only
    thing here that can say whether the framing step recovers the look the crawl had, and
    it is the only honest read of the sweep (the composite gain is an argmax over 18 draws
    and is biased upward by construction).
  * **readout** — old-vs-new rank agreement, old-Q5 survival, and the degree mix of the new
    top quintile against the old.

CAPTIONS CARRY BOTH SORTS. Every tile shows the new measures AND the dry run's atom-frame
`radial_range` / `radial_rings` / quintile, because the whole claim is that the two orderings
differ; a caption showing only the new one could not be checked against the sheet it replaces.

  uv run python tools/atlas/view_screen_sheets.py
"""
from __future__ import annotations

import argparse
import json
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
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools" / "explorer",
           ROOT / "tools" / "corpus", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                # noqa: E402
import view_screen as vs                    # noqa: E402
import prescreen                            # noqa: E402
import render_core as rc                    # noqa: E402
from active_ckpt import auto_maxiter         # noqa: E402

VIVID_PALETTE = "blue_orange"
VIVID_SOURCE = ROOT / "data" / "palettes" / "vivid_blue_orange.json"
WORKERS = 4                                  # engine PROCESSES (CLAUDE.md cap)
RENDER_THREADS = 3
TW, TH, PAD, LAB = 384, 216, 6, 62
BG, STRIP, INK, DIM, WARM = ((16, 16, 18), (30, 30, 34), (236, 236, 200),
                             (150, 170, 200), (226, 170, 120))


# --------------------------------------------------------------------------- #
def render_vivid(row: dict, out: Path) -> Path | None:
    if out.exists():
        return out
    fam = row.get("partition") or "mandelbrot"
    argv = rc.render_one_argv(row["cx"], row["cy"], row["fw"],
                              auto_maxiter(float(row["fw"])),
                              prescreen.RENDER_W, prescreen.RENDER_H, prescreen.RENDER_SS,
                              VIVID_PALETTE, VIVID_SOURCE, out,
                              family=(None if fam == "mandelbrot" else fam))
    try:
        rc.run_render_one(argv, out, threads=RENDER_THREADS)
    except Exception as e:
        print(f"  render failed {out.name}: {str(e)[:120]}", flush=True)
        return None
    return out


def render_all(jobs: list[tuple[dict, Path]], log=print) -> None:
    t0, n = time.time(), [0]

    def one(j):
        render_vivid(*j)
        n[0] += 1
        if n[0] % 10 == 0:
            log(f"  {n[0]}/{len(jobs)} rendered  {time.time()-t0:.0f}s")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))


def _tile(path, w=TW, h=TH):
    if path and Path(path).exists():
        return Image.open(path).convert("RGB").resize((w, h))
    return Image.new("RGB", (w, h), (70, 22, 22))


def caption(dr, x, y, lines):
    for i, (txt, col) in enumerate(lines):
        dr.text((x + 4, y + 4 + 19 * i), txt, fill=col)


def cap_lines(r: dict, comp: float, newq: int) -> list:
    k = "keep" if r.get("k") is None else f"k{float(r['k']):g}"
    op = r["op"].replace("_to_sibling", "").replace("_to_nucleus", "").replace("_expand", "")
    return [
        (f"{op} {k}  d{r.get('degree')} p{r.get('period')}  fw={r['fw']:.3g}", INK),
        (f"NEW comp={comp:.2f} covq25={r['band_coverage_q25']:.2f} "
         f"rng={r['radial_range']:.1f} rings={r['radial_rings']:.0f} "
         f"int={r['interior_fraction']:.2f}  Q{newq}", WARM),
        (f"OLD atom rng={r['atom_radial_range']:.1f} rings={r['atom_radial_rings']:.0f} "
         f"int={r['atom_interior_fraction']:.2f}  Q{r['old_quintile']}", DIM),
    ]


def build_sheet(items, title, out_png, cols=3):
    n = len(items)
    rows_n = (n + cols - 1) // cols
    W, TITLE_H = cols * (TW + PAD) + PAD, 34
    sheet = Image.new("RGB", (W, TITLE_H + rows_n * (TH + LAB + PAD) + PAD), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 2, 9), title, fill=INK)
    for i, (img, lines) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = PAD + c * (TW + PAD), TITLE_H + PAD + r * (TH + LAB + PAD)
        sheet.paste(_tile(img), (x, y))
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=STRIP)
        caption(dr, x, y + TH, lines)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"  wrote {out_png.name} ({n} tiles)", flush=True)


def build_pair_sheet(pairs, title, out_png):
    """One row per candidate: original frame | the sweep's chosen window."""
    W, TITLE_H = 2 * (TW + PAD) + PAD, 34
    sheet = Image.new("RGB", (W, TITLE_H + len(pairs) * (TH + LAB + PAD) + PAD), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 2, 9), title, fill=INK)
    for i, (a_img, b_img, lines_a, lines_b) in enumerate(pairs):
        y = TITLE_H + PAD + i * (TH + LAB + PAD)
        for j, (img, lines) in enumerate(((a_img, lines_a), (b_img, lines_b))):
            x = PAD + j * (TW + PAD)
            sheet.paste(_tile(img), (x, y))
            dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=STRIP)
            caption(dr, x, y + TH, lines)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"  wrote {out_png.name} ({len(pairs)} pairs)", flush=True)


# --------------------------------------------------------------------------- #
def quintile_index(vals: list[float]) -> tuple[list[int], list[float]]:
    v = np.asarray(vals, dtype=float)
    edges = [float(np.percentile(v, 20.0 * i)) for i in range(1, 5)]
    return [1 + sum(1 for e in edges if x > e) for x in v], edges


def stratify(pool, n, seed):
    """`n` rows spread over the (operator x degree) cells present in this quintile.

    Floor-then-remainder, same shape as `maneuver_inspection_sheet.stratify`: every cell
    gets one before any cell gets two, so a sheet cannot become one operator's showreel.
    """
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in pool:
        cells[(r["op"], r.get("degree"))].append(r)
    for c in cells.values():
        rng.shuffle(c)
    keys = sorted(cells, key=lambda k: (str(k[0]), str(k[1])))
    take = {k: 0 for k in keys}
    while sum(take.values()) < n:
        cand = [k for k in keys if take[k] < len(cells[k])]
        if not cand:
            break
        k = min(cand, key=lambda k: (take[k], -len(cells[k])))
        take[k] += 1
    out = []
    for k in keys:
        out.extend(cells[k][:take[k]])
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(x.size, float)
        r[o] = np.arange(1, x.size + 1, dtype=float)
        xs = x[o]
        i = 0
        while i < xs.size:
            j = i
            while j + 1 < xs.size and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return round(float((ra * rb).sum() / den), 4) if den > 0 else None


def agreement(rows: list[dict]) -> dict:
    new = [r["_comp"] for r in rows]
    old = [r["atom_radial_range"] for r in rows]
    nq = [r["new_quintile"] for r in rows]
    oq = [r["old_quintile"] for r in rows]
    old_q5 = [r for r in rows if r["old_quintile"] == 5]
    new_q5 = [r for r in rows if r["new_quintile"] == 5]

    def deg_mix(sel):
        c = Counter(r.get("degree") for r in sel)
        tot = max(1, sum(c.values()))
        return {f"d{k}": f"{v} ({100*v/tot:.0f}%)" for k, v in sorted(c.items(),
                                                                     key=lambda kv: str(kv[0]))}

    def op_mix(sel):
        c = Counter(r["op"] for r in sel)
        tot = max(1, sum(c.values()))
        return {k: f"{v} ({100*v/tot:.0f}%)" for k, v in c.most_common()}

    return dict(
        n=len(rows),
        spearman_new_vs_old_atom_range=spearman(new, old),
        spearman_new_vs_old_quintile=spearman(nq, oq),
        old_Q5_n=len(old_q5),
        old_Q5_surviving_new_Q5=sum(1 for r in old_q5 if r["new_quintile"] == 5),
        old_Q5_surviving_frac=round(sum(1 for r in old_q5 if r["new_quintile"] == 5)
                                    / max(1, len(old_q5)), 4),
        old_Q5_falling_to_new_Q1_or_Q2=sum(1 for r in old_q5 if r["new_quintile"] <= 2),
        new_Q5_that_were_old_Q1_or_Q2=sum(1 for r in new_q5 if r["old_quintile"] <= 2),
        quintile_transition=[[sum(1 for r in rows if r["old_quintile"] == o
                                  and r["new_quintile"] == n_) for n_ in range(1, 6)]
                             for o in range(1, 6)],
        degree_mix_new_Q5=deg_mix(new_q5),
        degree_mix_old_Q5=deg_mix(old_q5),
        degree_mix_population=deg_mix(rows),
        operator_mix_new_Q5=op_mix(new_q5),
        operator_mix_old_Q5=op_mix(old_q5),
        vetoed_n=sum(1 for r in rows if r["_vetoed"]),
        vetoed_that_were_old_Q5=sum(1 for r in rows if r["_vetoed"]
                                    and r["old_quintile"] == 5),
        DEGREE_CAVEAT=(
            "Degree and period are confounded in this population and neither this table "
            "nor the old one separates them: `radial_rings` is Spearman +0.87 with period "
            "over periods 2-74 (orbital_field_metrics.md §6), and the operators reach "
            "higher-degree roots at different rates. Read a shift in the degree mix as a "
            "shift in what the composite selects, NOT as a degree result "
            "(measurement_practice.md §1, 'a search that chooses where it goes confounds "
            "its own axes')."),
        RANK_CAVEAT=(
            "The two sorts are not two measurements of one quantity: the old one is "
            "`radial_range` on the atom's 4x frame (one value shared by every k row of a "
            "nucleus), the new one is a composite on the frame each row actually pushed. "
            "Low agreement is the intended outcome, not a discrepancy to reconcile."),
    )


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path,
                    default=paths.scratch("view_rescreen", "scores.jsonl"))
    ap.add_argument("--sweep", type=Path,
                    default=paths.scratch("view_rescreen", "sweep.jsonl"))
    ap.add_argument("--out", type=Path, default=paths.scratch("view_screen"))
    ap.add_argument("--tiles", type=int, default=18)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            a.scores.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("screened")]
    veto = vs.interior_veto(vs.load_refs())
    for r in ok:
        r["_comp"] = vs.composite(r, veto)
        r["_vetoed"] = vs.is_vetoed(r, veto)
    nq, new_edges = quintile_index([r["_comp"] for r in ok])
    oq, old_edges = quintile_index([r["atom_radial_range"] for r in ok])
    for r, n_, o_ in zip(ok, nq, oq):
        r["new_quintile"], r["old_quintile"] = n_, o_

    print(f"[sheets] {len(ok)} screened rows; new-composite quintile edges "
          f"{[round(e,3) for e in new_edges]}; veto {veto} ({sum(r['_vetoed'] for r in ok)} rows)")

    a.out.mkdir(parents=True, exist_ok=True)
    vivid = a.out / "vivid"
    vivid.mkdir(parents=True, exist_ok=True)

    top = stratify([r for r in ok if r["new_quintile"] == 5], a.tiles, a.seed)
    bot = stratify([r for r in ok if r["new_quintile"] == 1], a.tiles, a.seed + 1)
    for r in top + bot:
        r["_png"] = vivid / f"q{r['new_quintile']}_{abs(hash(r['key'])) % 10**10:010d}.jpg"

    swept = []
    if a.sweep.exists():
        swept = [json.loads(l) for l in
                 a.sweep.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = {r["key"]: r for r in ok}
    usable = [s for s in swept if s.get("chosen") and s["key"] in by_key]
    # TWO pair sets, because one of them cannot answer the question on its own. The
    # top-composite set is what the artifact was specified as — but the composite's own
    # winners are precisely the views the sweep has least reason to move (5 of 20 here),
    # so a sheet of them is mostly identical pairs. The largest-gain set shows what the
    # framing step DOES; the top set shows what it does to the material you would ship.
    pairs_src = sorted(usable, key=lambda s: -(s.get("origin_composite") or -1e9))[:a.pairs]
    moved_src = sorted([s for s in usable if s.get("moved")],
                       key=lambda s: -((s["chosen_composite"] or 0)
                                       - (s["origin_composite"] or 0)))[:a.pairs]
    pair_dir = a.out / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for s in {id(x): x for x in pairs_src + moved_src}.values():
        h = abs(hash(s["key"])) % 10 ** 10
        s["_a"] = pair_dir / f"{h:010d}_before.jpg"
        s["_b"] = pair_dir / f"{h:010d}_after.jpg"

    jobs = [(r, r["_png"]) for r in top + bot]
    for s in {id(x): x for x in pairs_src + moved_src}.values():
        jobs.append(({"cx": s["cx"], "cy": s["cy"], "fw": s["fw"],
                      "partition": s["partition"]}, s["_a"]))
        jobs.append(({"cx": s["chosen"]["cx"], "cy": s["chosen"]["cy"],
                      "fw": s["chosen"]["fw"], "partition": s["partition"]}, s["_b"]))
    print(f"[render] {len(jobs)} vivid tiles")
    render_all(jobs)

    build_sheet([(r["_png"], cap_lines(r, r["_comp"], 5)) for r in top],
                f"view screen — NEW composite TOP quintile (n={len(top)}, vivid "
                f"blue_orange; stratified over operator x degree)",
                a.out / "sheet_new_q5.png")
    build_sheet([(r["_png"], cap_lines(r, r["_comp"], 1)) for r in bot],
                f"view screen — NEW composite BOTTOM quintile (n={len(bot)}, vivid "
                f"blue_orange; stratified over operator x degree)",
                a.out / "sheet_new_q1.png")

    def pair_rows(src):
        out = []
        for s in src:
            base = by_key[s["key"]]
            ch, cm = s["chosen"], s["chosen_measures"]
            la = [(f"BEFORE  {s['op']} k={s['k']}  d{s.get('degree')} p{s.get('period')}", INK),
                  (f"comp={s['origin_composite']:.2f} covq25={base['band_coverage_q25']:.2f} "
                   f"rng={base['radial_range']:.1f} int={base['interior_fraction']:.2f}", DIM),
                  (f"fw={s['fw']:.4g}  (as the walk pushed it)", DIM)]
            lb = [(f"AFTER   dx={ch['dx']:+g} dy={ch['dy']:+g} scale={ch['scale']:g}"
                   f"{'   (unmoved)' if not s['moved'] else ''}", INK),
                  (f"comp={s['chosen_composite']:.2f} covq25={cm['band_coverage_q25']:.2f} "
                   f"rng={cm['radial_range']:.1f} int={cm['interior_fraction']:.2f}", WARM),
                  (f"fw={ch['fw']:.4g}", DIM)]
            out.append((s["_a"], s["_b"], la, lb))
        return out

    if pairs_src:
        n_moved = sum(1 for s in pairs_src if s["moved"])
        build_pair_sheet(pair_rows(pairs_src),
                         f"view screen — framing sweep BEFORE / AFTER, {len(pairs_src)} "
                         f"TOP-COMPOSITE candidates ({n_moved} moved — the composite's own "
                         f"winners are what it least wants to move)",
                         a.out / "sheet_framing_pairs.png")
    if moved_src:
        build_pair_sheet(pair_rows(moved_src),
                         f"view screen — framing sweep BEFORE / AFTER, {len(moved_src)} "
                         f"LARGEST COMPOSITE GAIN (all moved; the gain is an argmax over 18 "
                         f"draws by the objective it maximises)",
                         a.out / "sheet_framing_pairs_moved.png")
    if not swept:
        print("  no sweep rows — before/after sheets SKIPPED (run view_frame_sweep.py)")

    rep = agreement(ok)
    rep["new_quintile_edges"] = [round(e, 4) for e in new_edges]
    rep["old_quintile_edges"] = [round(e, 4) for e in old_edges]
    (a.out / "readout.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items()
                      if not k.endswith("CAVEAT")}, indent=2))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
