#!/usr/bin/env python
r"""view_screen_sheets.py — sheets and the old-vs-new readout for the view-level screen.

Three artifacts, for Matt's eye and for the record:

  * **quintile sheets** — the new composite's top and bottom quintiles, stratified across
    (operator x degree) inside the quintile so a sheet cannot be one operator's showreel.
    Same vivid `blue_orange` map on every tile, so the eye compares STRUCTURE and not
    palette (`maneuver_inspection_sheet.py`, same reason). Every tile carries its RANK
    inside the quintile and a `strat` mark when stratification pulled it in from a thin
    cell — a stratified sheet is not "the top 18" and must not be read as one.
  * **the unstratified top-N sheet** — the same quintile with the sampling rule taken
    off. It is the sheet a hunt's output actually looks like, and the one the eye should
    ultimately approve; the stratified sheet exists to stop a sheet BEING one operator, not
    to describe what the screen would hand you.
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
import apportion                            # noqa: E402  (THE apportionment rules)
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


def _tag(key: str) -> str:
    """A STABLE per-key filename tag. `hash()` on a str is salted per interpreter
    (PYTHONHASHSEED), so the render cache built from it missed on every re-run and every
    sheet re-render paid for the whole tile set again."""
    import hashlib
    return hashlib.blake2b(key.encode("utf-8"), digest_size=5).hexdigest()


def _tile(path, w=TW, h=TH):
    if path and Path(path).exists():
        return Image.open(path).convert("RGB").resize((w, h))
    return Image.new("RGB", (w, h), (70, 22, 22))


def caption(dr, x, y, lines):
    for i, (txt, col) in enumerate(lines):
        dr.text((x + 4, y + 4 + 19 * i), txt, fill=col)


def cap_lines(r: dict, comp: float, newq: int, p) -> list:
    """Three lines: what the row is, what the LIVE composite (v3) says, and what it changed
    from — plus v4's score in brackets, marked REJECTED, when the row carries it.

    v4 rides along instead of replacing the v3 line because it was measured against the gate
    and rejected; a caption that led with it would put a rejected formulation in front of the
    eye as if it were the screen.

    `RANK` says where the tile sits inside its own quintile by composite, and `strat` marks
    a tile the stratification pulled in from a thin (operator x degree) cell rather than one
    the composite put near the top. Without it a stratified sheet reads as "the top 18",
    which it is not, and the eye judges a sampling artifact as a ranking result.
    """
    k = "keep" if r.get("k") is None else f"k{float(r['k']):g}"
    op = r["op"].replace("_to_sibling", "").replace("_to_nucleus", "").replace("_expand", "")
    rk = r.get("_rank_in_q")
    tag = ("" if rk is None else
           f"   #{rk}/{r['_q_n']}{'  strat' if r.get('_forced') else ''}")
    return [
        (f"{op} {k}  d{r.get('degree')} p{r.get('period')}  fw={r['fw']:.3g}{tag}", INK),
        (f"v3 comp={comp:.2f} cov={r['band_coverage']:.2f}/{r['band_coverage_q25']:.2f} "
         f"size={vs.size_factor(r, p):.2f} rich={vs.richness(r, p):.1f} "
         f"int={r['interior_fraction']:.3f}  Q{newq}", WARM),
        (f"v2 comp={r['_comp_prev']:.2f} rng={r['radial_range']:.1f} "
         f"rings={r['radial_rings']:.0f}  Q{r['prev_quintile']}"
         + ("" if r.get('_comp_v4') is None else
            f"   [v4 REJECTED: comp={r['_comp_v4']:.2f} "
            f"cov={r['band_coverage_v4']:.2f}/{r['band_coverage_q25_v4']:.2f}]")
         + f"   (atom rng={r['atom_radial_range']:.1f} Q{r['old_quintile']})", DIM),
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
    take = apportion.deal_round_robin({k: len(cells[k]) for k in keys}, n)
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
    has_prev = bool(rows) and all("_comp_prev" in r and "prev_quintile" in r for r in rows)
    prev_q5 = [r for r in rows if r.get("prev_quintile") == 5] if has_prev else []

    def deg_mix(sel):
        c = Counter(r.get("degree") for r in sel)
        tot = max(1, sum(c.values()))
        return {f"d{k}": f"{v} ({100*v/tot:.0f}%)" for k, v in sorted(c.items(),
                                                                     key=lambda kv: str(kv[0]))}

    def op_mix(sel):
        c = Counter(r["op"] for r in sel)
        tot = max(1, sum(c.values()))
        return {k: f"{v} ({100*v/tot:.0f}%)" for k, v in c.most_common()}

    def k_mix(sel):
        """The k set the supply run would order from this quintile. `keep` is its own class:
        it is not a `k` at all, it is the parent's frame kept, so folding it into a numeric
        bucket would report a zoom the walk never chose."""
        c = Counter("keep" if r.get("k") is None else f"k{float(r['k']):g}" for r in sel)
        tot = max(1, sum(c.values()))
        return {k: f"{v} ({100*v/tot:.0f}%)" for k, v in
                sorted(c.items(), key=lambda kv: (kv[0] == "keep",
                                                  float(kv[0][1:]) if kv[0] != "keep" else 0))}

    def int_dist(sel):
        if not sel:
            return {}
        v = np.array([r["interior_fraction"] for r in sel], dtype=float)
        return {f"p{q}": round(float(np.percentile(v, q)), 4)
                for q in (5, 25, 50, 75, 90, 95, 99)}

    def int_bands(sel):
        """Against the band the composite now shapes on, not against arbitrary deciles."""
        tot = max(1, len(sel))
        cuts = [(0.0, 0.06), (0.06, 0.12), (0.12, 0.17), (0.17, 0.25), (0.25, 1.01)]
        return {f"[{a:.2f},{b:.2f})":
                f"{sum(1 for r in sel if a <= r['interior_fraction'] < b)} "
                f"({100*sum(1 for r in sel if a <= r['interior_fraction'] < b)/tot:.0f}%)"
                for a, b in cuts}

    prev = {}
    if has_prev:
        prev = dict(
            spearman_new_vs_prev=spearman(new, [r["_comp_prev"] for r in rows]),
            prev_Q5_n=len(prev_q5),
            prev_Q5_surviving_new_Q5=sum(1 for r in prev_q5 if r["new_quintile"] == 5),
            prev_Q5_surviving_frac=round(sum(1 for r in prev_q5 if r["new_quintile"] == 5)
                                       / max(1, len(prev_q5)), 4),
            new_Q5_that_were_prev_Q1_or_Q2=sum(1 for r in new_q5 if r["prev_quintile"] <= 2),
            quintile_transition_prev_to_new=[
                [sum(1 for r in rows if r["prev_quintile"] == o and r["new_quintile"] == n_)
                 for n_ in range(1, 6)] for o in range(1, 6)],
            k_mix_new_Q5=k_mix(new_q5), k_mix_prev_Q5=k_mix(prev_q5),
            k_mix_population=k_mix(rows),
            interior_pct_new_Q5=int_dist(new_q5), interior_pct_prev_Q5=int_dist(prev_q5),
            interior_pct_population=int_dist(rows),
            interior_bands_new_Q5=int_bands(new_q5),
            interior_bands_prev_Q5=int_bands(prev_q5),
            degree_mix_prev_Q5=deg_mix(prev_q5),
        )

    return dict(
        n=len(rows), **prev,
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
    p = vs.screen_params(vs.load_refs())
    veto = p.veto
    for r in ok:
        # v4 was measured against the gate and REJECTED, so the live sort stays v3 and the
        # v4 score rides along as the comparison column. Reversing these two would ship a
        # rejected formulation by way of a sheet.
        r["_comp"] = vs.composite_v3(r, p)
        r["_comp_prev"] = vs.composite_v2(r, veto)
        r["_comp_v4"] = (vs.composite_v4(r, p) if "band_coverage_v4" in r else None)
        r["_vetoed"] = vs.is_vetoed(r, veto)
    nq, new_edges = quintile_index([r["_comp"] for r in ok])
    oq, old_edges = quintile_index([r["atom_radial_range"] for r in ok])
    pq, prev_edges = quintile_index([r["_comp_prev"] for r in ok])
    for r, n_, o_, w_ in zip(ok, nq, oq, pq):
        r["new_quintile"], r["old_quintile"], r["prev_quintile"] = n_, o_, w_

    print(f"[sheets] {len(ok)} screened rows; v3 quintile edges "
          f"{[round(e,3) for e in new_edges]} (v2 {[round(e,3) for e in prev_edges]}); "
          f"veto {veto} ({sum(r['_vetoed'] for r in ok)} rows); "
          f"size-banded {sum(1 for r in ok if vs.size_factor(r, p) < 1.0 and not r['_vetoed'])}")

    a.out.mkdir(parents=True, exist_ok=True)
    vivid = a.out / "vivid"
    vivid.mkdir(parents=True, exist_ok=True)

    # A stratified quintile sheet and a straight top-N sheet answer different questions and
    # the first cannot answer the second's. Stratification exists so a sheet cannot become
    # one operator's showreel — but that means most of its tiles are pulled in from thin
    # (operator x degree) cells, and reading it as "the top 18" judges a sampling rule as if
    # it were a ranking. So the rank inside the quintile is stamped on every tile, and the
    # unstratified top-N sheet is built beside it: THAT is what a hunt's output looks like.
    q5 = sorted([r for r in ok if r["new_quintile"] == 5], key=lambda r: -r["_comp"])
    for i, r in enumerate(q5, 1):
        r["_rank_in_q"], r["_q_n"] = i, len(q5)
    top = stratify(q5, a.tiles, a.seed)
    for r in top:
        r["_forced"] = r["_rank_in_q"] > a.tiles
    bot = stratify([r for r in ok if r["new_quintile"] == 1], a.tiles, a.seed + 1)
    pure = q5[:a.tiles]
    for r in top + bot + pure:
        r["_png"] = vivid / f"q{r['new_quintile']}_{_tag(r['key'])}.jpg"

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
        h = _tag(s["key"])
        s["_a"] = pair_dir / f"{h}_before.jpg"
        s["_b"] = pair_dir / f"{h}_after.jpg"

    jobs = [(r, r["_png"]) for r in {id(x): x for x in top + bot + pure}.values()]
    for s in {id(x): x for x in pairs_src + moved_src}.values():
        jobs.append(({"cx": s["cx"], "cy": s["cy"], "fw": s["fw"],
                      "partition": s["partition"]}, s["_a"]))
        jobs.append(({"cx": s["chosen"]["cx"], "cy": s["chosen"]["cy"],
                      "fw": s["chosen"]["fw"], "partition": s["partition"]}, s["_b"]))
    print(f"[render] {len(jobs)} vivid tiles")
    render_all(jobs)

    n_forced = sum(1 for r in top if r["_forced"])
    build_sheet([(r["_png"], cap_lines(r, r["_comp"], 5, p)) for r in top],
                f"view screen — v3 composite TOP quintile (n={len(top)}, vivid "
                f"blue_orange; STRATIFIED over operator x degree — {n_forced} of "
                f"{len(top)} are stratification picks from thin cells, marked `strat`, "
                f"not the composite's top; cross floor {vs.TILE_CROSS_FLOOR:g}, band edge "
                f"{p.band_edge:g}^{p.band_exp:g}, caps {p.cap_range:g}/{p.cap_rings:g})",
                a.out / "sheet_new_q5.png")
    build_sheet([(r["_png"], cap_lines(r, r["_comp"], 1, p)) for r in bot],
                f"view screen — v3 composite BOTTOM quintile (n={len(bot)}, vivid "
                f"blue_orange; stratified over operator x degree)",
                a.out / "sheet_new_q1.png")
    build_sheet([(r["_png"], cap_lines(r, r["_comp"], 5, p)) for r in pure],
                f"view screen — v3 composite TOP {len(pure)}, UNSTRATIFIED (vivid "
                f"blue_orange; this is what the screen would actually hand a hunt, "
                f"operator/degree mix included rather than corrected for)",
                a.out / "sheet_top.png")

    # A framing pair is unreadable unless the two frames are comparable in scale, so the
    # ratio is CAPTIONED rather than assumed. With the anchor constraint the only eligible
    # scale is {1, 2}, so anything above 2.0 is content drift the constraint was supposed
    # to prevent and is flagged on the tile, not silently shown.
    slipped = []

    def pair_rows(src):
        out = []
        for s in src:
            base = by_key[s["key"]]
            ch, cm = s["chosen"], s["chosen_measures"]
            ratio = float(ch["fw"]) / float(s["fw"])
            if ratio > 2.0 + 1e-9:
                slipped.append((s["key"], round(ratio, 3)))
            la = [(f"BEFORE  {s['op']} k={s['k']}  d{s.get('degree')} p{s.get('period')}", INK),
                  (f"comp={s['origin_composite']:.2f} "
                   f"covq25={base['band_coverage_q25']:.2f} "
                   f"rng={base['radial_range']:.1f} int={base['interior_fraction']:.3f} "
                   f"size={vs.size_factor(base, p):.2f}", DIM),
                  (f"fw={s['fw']:.4g}  (as the walk pushed it)", DIM)]
            lb = [(f"AFTER   dx={ch['dx']:+g} dy={ch['dy']:+g} scale={ch['scale']:g}"
                   f"{'   (unmoved)' if not s['moved'] else ''}"
                   f"   fw ratio {ratio:.2f}x"
                   f"{'  << SCALE JUMP' if ratio > 2.0 + 1e-9 else ''}", INK),
                  (f"comp={s['chosen_composite']:.2f} "
                   f"covq25={cm['band_coverage_q25']:.2f} "
                   f"rng={cm['radial_range']:.1f} int={cm['interior_fraction']:.3f} "
                   f"size={vs.size_factor(cm, p):.2f}", WARM),
                  (f"fw={ch['fw']:.4g}", DIM)]
            out.append((s["_a"], s["_b"], la, lb))
        return out

    if pairs_src:
        n_moved = sum(1 for s in pairs_src if s["moved"])
        build_pair_sheet(pair_rows(pairs_src),
                         f"view screen v3 — framing sweep BEFORE / AFTER, {len(pairs_src)} "
                         f"TOP-COMPOSITE candidates ({n_moved} moved — the composite's own "
                         f"winners are what it least wants to move)",
                         a.out / "sheet_framing_pairs.png")
    if moved_src:
        build_pair_sheet(pair_rows(moved_src),
                         f"view screen v3 — framing sweep BEFORE / AFTER, {len(moved_src)} "
                         f"LARGEST v3 COMPOSITE GAIN (all moved, all anchor-retaining; the "
                         f"gain is an argmax over the eligible windows by the objective it "
                         f"maximises)",
                         a.out / "sheet_framing_pairs_moved.png")
    if not swept:
        print("  no sweep rows — before/after sheets SKIPPED (run view_frame_sweep.py)")
    if slipped:
        print(f"  !! {len(slipped)} pair(s) with fw ratio > 2x — the anchor constraint "
              f"should have prevented these: {slipped}")

    rep = agreement(ok)
    rep["composite_version"] = "v3"
    rep["previous_version"] = "v2"
    rep["v4_status"] = "measured and NOT adopted (data/atlas/view_screen_gate.json §v4)"
    rep["screen_params"] = p._asdict()
    rep["pair_fw_ratio_over_2x"] = slipped
    rep["new_quintile_edges"] = [round(e, 4) for e in new_edges]
    rep["prev_quintile_edges"] = [round(e, 4) for e in prev_edges]
    rep["old_quintile_edges"] = [round(e, 4) for e in old_edges]
    (a.out / "readout.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items()
                      if not k.endswith("CAVEAT")}, indent=2))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
