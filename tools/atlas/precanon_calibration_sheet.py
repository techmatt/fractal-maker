#!/usr/bin/env python
r"""precanon_calibration_sheet.py — the instrument Matt places the dedup boundary with.

WHAT THIS IS FOR. `production_seeder.near_dup` cuts a candidate when the plane distance to a
cloud row is under `DEDUP_K * scale(fw_a, fw_b)`. The `min(fw)` direction is validated
(`precanon_minfw_replay.py` + `precanon_minfw_sheet.py`), but `DEDUP_K = 1.5` on the MIN scale
is an inherited constant nobody chose: it transfers from the max-scale rule, where the same
number means different geometry. This sheet asks the eye where the boundary actually is —
pairs sorted along the rule's decision variable `d / min(fw)`, Matt marks where "close enough
to be identical" becomes "sufficiently different to preserve".

IT FLIPS NOTHING. `DEDUP_SCALE`/`DEDUP_K` are untouched, no row is scored, no admission code
moves. The cut is derived from the exported verdicts by a FOLLOW-UP, and this module
deliberately computes and displays no candidate boundary of its own.

NOT A LABEL SHEET. This is a dedup-IDENTITY instrument: it never enters the label corpus, the
labeling rig or `labels/`. Verdicts export to their own record under `scratch/`.

THE SAMPLE. Two strata, interleaved by the sort so neither is identifiable on the page:

  dup     every `precanon_dup` row of the run joined to the ledger row that displaced it
          (2,531 pairs, all resolvable). These are pairs the CURRENT rule collapsed.
  anchor  admitted rows paired with their nearest same-partition, same-identity admitted
          neighbour — pairs the current rule KEPT APART, i.e. real material at the distinct
          end of the axis. Guaranteed `d >= DEDUP_K * max(fw)` by construction; `stage_plan`
          asserts it rather than trusting it.

Stratified by fw ratio (max/min) into three bands, and within a band binned by RANK QUANTILE of
`d / min(fw)` over the band's pairs inside `DOMAIN_CAP` — the sheet's job is the boundary
region, and the `>10x` band's raw tail runs to `d/min ~ 2e3`, which spends the eye on pairs no
rule would ever merge. The cap is stated on the page and in the plan.

BLIND BY DEFAULT. During judgment no fate, no rule verdict and no geometry number is visible;
`r` reveals them and the reveal state at the moment of each verdict is exported with it. Left/
right within a pair is assigned by seeded coin flip, because the displacer is the wider frame
in most dup pairs and a fixed side would be a tell.

  uv run python tools/atlas/precanon_calibration_sheet.py plan     # sample + join, no renders
  uv run python tools/atlas/precanon_calibration_sheet.py render
  uv run python tools/atlas/precanon_calibration_sheet.py sheet
  uv run python tools/atlas/precanon_calibration_sheet.py verify --port 8010
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
           str(ROOT / "tools" / "sourcing"), str(ROOT / "tools" / "scoring")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                    # noqa: E402
import corpus_common as cc                      # noqa: E402
import build_minibrot_batch as BMB              # noqa: E402
import build_q4_harvest_batches as bq           # noqa: E402  (_render_block — render authority)
import production_pins as prod                  # noqa: E402  (PALETTE — the standard coloring)
import production_seeder as ps                  # noqa: E402  (DEDUP_K — asserted, never moved)
import precanon_minfw_replay as R               # noqa: E402  (the population + its join)

OUT = Path(paths.scratch("precanon_calibration"))
RENDERS = OUT / "renders"
PLAN = OUT / "pairs.json"
SHEET = OUT / "precanon_calibration_sheet.html"
# PAGE-RELATIVE (see precanon_minfw_sheet.py): a browser resolves a relative src against the
# PAGE's directory, so a root-relative path here would 404 and a link check that fetches from
# the server root would still pass. `verify` resolves with `urljoin`, as the browser does.
RENDERS_URL = "renders"

# 640x360 ss2 canonical — the prompt's fidelity for this sheet (the fate/autopsy sheets use
# 1280x720 ss2). Everything else matches the corpus render recipe.
CROP_W, CROP_H, CROP_SS = 640, 360, 2

# ONE image per location, in the production-canonical coloring — not a per-pair seeded palette
# and no vivid companion. A dedup-identity call is a question about geometry, and two palettes
# (or a palette that changes down the page) put a colour difference next to every judgment the
# sheet is trying to read as a shape difference. `twilight_shifted` @ `clean_colormaps.json` is
# the deploy-canonical pair (`production_pins.PALETTE`, `descent/store.CANONICAL_PALETTE`).
PALETTE = prod.PALETTE
PALETTE_SOURCE = ROOT / "data" / "palettes" / "clean_colormaps.json"

# (key, label, lo, hi]  — fw ratio is always max/min, so it is >= 1 and the bands are ordered.
BANDS = (("le2", "≤2×", 1.0, 2.0),
         ("mid", "2–10×", 2.0, 10.0),
         ("gt10", ">10×", 10.0, math.inf))

# The binning domain for the dup stratum. 2x the prompt's "clearly beyond any plausible cut
# (>=3)", and stated on the page: pairs above it exist in the record and are simply not what
# this instrument is asking about.
DOMAIN_CAP = 6.0
N_BINS, PER_BIN, N_ANCHOR = 8, 5, 5
SEED = 20260804

RENDER_WORKERS, RENDER_THREADS = 3, 4          # 3 processes x 4 threads on a 12-core box


# =========================================================================== #
# strata
# =========================================================================== #
def band_of(ratio: float) -> str:
    """fw-ratio band key. Half-open upward: 2.0 is `le2`, 10.0 is `mid`."""
    for key, _label, lo, hi in BANDS:
        if (ratio <= hi) and (ratio > lo or key == "le2"):
            return key
    return BANDS[-1][0]


def _phoenix_of(r: dict) -> dict | None:
    v = {k: r.get(k) for k in ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
                               "phoenix_zm1_re", "phoenix_zm1_im") if r.get(k) is not None}
    return v or None


def ledger_ident(led: dict, partition: str):
    """The ledger row's dup-identity vector, same rule as `precanon_minfw_replay.cand_ident`:
    the phoenix 6-vector, the julia seed 2-vector, or None on a c-plane row."""
    if partition == "phoenix":
        v = [led.get(k) for k in ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re",
                                  "phoenix_p_im", "phoenix_zm1_re", "phoenix_zm1_im")]
        if any(x is None for x in v):
            raise SystemExit(f"phoenix ledger row {led['id']} has no 6-vector identity")
        return tuple(float(x) for x in v)
    cre = led.get("julia_c_re")
    return None if cre is None else (float(cre), float(led["julia_c_im"]))


def _side_from_row(r: dict) -> dict:
    """The candidate side of a dup pair — a q4 harvest check the run threw away."""
    return dict(kind="candidate", id=f"{r['batch']:03d}_{r['node_id']:05d}",
                partition=r["partition"], cx=r["cx"], cy=r["cy"], fw=r["fw"],
                julia_c_re=r["julia_c_re"], julia_c_im=r["julia_c_im"],
                phoenix=r["phoenix"], batch=r["batch"], node_id=r["node_id"],
                depth=r["depth"], mix_source=r["mix_source"],
                cheap_pgood=r.get("cheap_pgood"), decoded_class=None,
                fate="precanon_dup")


def _side_from_ledger(led: dict, partition: str, fate: str) -> dict:
    return dict(kind="outcome", id=led["id"], partition=partition,
                cx=float(led["outcome_cx"]), cy=float(led["outcome_cy"]),
                fw=float(led["outcome_fw"]),
                julia_c_re=led.get("julia_c_re"), julia_c_im=led.get("julia_c_im"),
                phoenix=_phoenix_of(led), batch=None, node_id=led.get("node_id"),
                depth=led.get("reached_depth"), mix_source=led.get("mix_source"),
                cheap_pgood=led.get("cheap_pgood"),
                decoded_class=led.get("decoded_class"), fate=fate)


def _geom(a: dict, b: dict) -> dict:
    dist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    lo, hi = min(a["fw"], b["fw"]), max(a["fw"], b["fw"])
    return dict(dist=dist, fw_min=lo, fw_max=hi, d_over_min=dist / lo, d_over_max=dist / hi,
                fw_ratio=hi / lo)


def dup_pairs(rows: list[dict], ledger: dict) -> list[dict]:
    """Every `precanon_dup` check joined to the ledger row that displaced it."""
    out = []
    for r in rows:
        if r["fate"] != "precanon_dup":
            continue
        led = ledger.get(r["rec_dup"])
        if led is None:                     # unresolvable geometry — excluded, counted by caller
            continue
        a = _side_from_row(r)
        b = _side_from_ledger(led, r["partition"], "admitted")
        out.append(dict(stratum="dup", partition=r["partition"], a=a, b=b, geom=_geom(a, b)))
    return out


def anchor_pairs(rows: list[dict], ledger: dict) -> tuple[list[dict], list[str]]:
    """Admitted rows paired with their nearest admitted neighbour of the same partition AND
    the same dup identity — pairs the CURRENT rule kept apart. Unordered-deduped (a nearest
    -neighbour scan yields each mutual pair twice).

    The identity restriction is load-bearing: two julia rows with different `c` are different
    fractals and `near_dup` never compares their distance, so such a pair says nothing about
    where a DISTANCE boundary sits."""
    part_of = {r["ledger"]["id"]: r["partition"] for r in rows if r["ledger"] is not None}
    by_part: dict[str, list] = defaultdict(list)
    for led in ledger.values():
        if led.get("distinct"):
            by_part[part_of.get(led["id"], led.get("family", "?"))].append(led)

    seen, out, viol = set(), [], []
    for part, ls in sorted(by_part.items()):
        for i, la in enumerate(ls):
            best = None
            for j, lb in enumerate(ls):
                if i == j or ledger_ident(la, part) != ledger_ident(lb, part):
                    continue
                d = math.hypot(float(la["outcome_cx"]) - float(lb["outcome_cx"]),
                               float(la["outcome_cy"]) - float(lb["outcome_cy"]))
                if best is None or d < best[0]:
                    best = (d, lb)
            if best is None:
                continue
            key = tuple(sorted((la["id"], best[1]["id"])))
            if key in seen:
                continue
            seen.add(key)
            a = _side_from_ledger(la, part, "admitted")
            b = _side_from_ledger(best[1], part, "admitted")
            g = _geom(a, b)
            # The stratum's whole claim. Both rows are cloud members, so the later one was
            # `distinct` against the earlier: the current rule MUST have kept them apart.
            if g["dist"] < ps.DEDUP_K * g["fw_max"] * (1 - 1e-12):
                viol.append(f"{key[0]} / {key[1]}: d={g['dist']:.6g} < "
                            f"{ps.DEDUP_K}*max(fw)={ps.DEDUP_K * g['fw_max']:.6g}")
            out.append(dict(stratum="anchor", partition=part, a=a, b=b, geom=g))
    return out, viol


# =========================================================================== #
# binning + seeded selection
# =========================================================================== #
def quantile_edges(vals: list[float], nbins: int) -> list[float]:
    """`nbins + 1` RANK-quantile edges over `vals`. Rank rather than value quantiles because
    `d / min(fw)` runs over four orders of magnitude within a band and value-spaced edges put
    most of the sample in one decade."""
    s = sorted(vals)
    if not s:
        return []
    n = len(s)
    return [s[min(int(round(q * (n - 1))), n - 1)] for q in
            (i / nbins for i in range(nbins + 1))]


def assign_bin(v: float, edges: list[float]) -> int:
    """Half-open `[edge_i, edge_{i+1})`, last bin closed — so the band's maximum, and anything
    at or above the top edge, lands in the last bin. A degenerate band (every value equal, so
    every edge equal) therefore collapses to ONE bin instead of scattering its ties across
    eight, which is what would otherwise report even coverage of a one-point population."""
    nb = len(edges) - 1
    for i in range(nb):
        if v < edges[i + 1]:
            return i
    return nb - 1


def pick(rng: random.Random, items: list, n: int) -> list:
    return list(items) if len(items) <= n else rng.sample(items, n)


def _pair_key(p: dict) -> str:
    return json.dumps([p["stratum"], p["a"]["id"], p["b"]["id"],
                       repr(p["a"]["cx"]), repr(p["b"]["cx"])], separators=(",", ":"))


# =========================================================================== #
# plan
# =========================================================================== #
def stage_plan(args) -> int:
    rows, ledger = R.load_population(R.RUN_DIR)
    n_dup_total = sum(1 for r in rows if r["fate"] == "precanon_dup")
    dups = dup_pairs(rows, ledger)
    anchors, viol = anchor_pairs(rows, ledger)
    if viol:
        for v in viol[:5]:
            print(f"  !! anchor is not kept-apart: {v}")
        raise SystemExit(f"{len(viol)} anchor pairs violate the kept-apart invariant — the "
                         f"far-side stratum is not what it claims")
    print(f"population: {n_dup_total} precanon_dup rows, {len(dups)} with resolvable displacer "
          f"geometry; {len(anchors)} anchor pairs (admitted <-> nearest same-identity admitted)")

    by_band: dict[str, list] = defaultdict(list)
    for p in dups + anchors:
        by_band[band_of(p["geom"]["fw_ratio"])].append(p)

    selected, band_meta = [], {}
    for key, label, lo, hi in BANDS:
        rng = random.Random(f"{args.seed}:{key}")
        pool = by_band.get(key, [])
        d_in = [p for p in pool if p["stratum"] == "dup"
                and p["geom"]["d_over_min"] <= args.domain_cap]
        d_out = [p for p in pool if p["stratum"] == "dup"
                 and p["geom"]["d_over_min"] > args.domain_cap]
        anch = sorted((p for p in pool if p["stratum"] == "anchor"),
                      key=lambda p: p["geom"]["d_over_min"])
        edges = quantile_edges([p["geom"]["d_over_min"] for p in d_in], args.n_bins)

        binned: dict[int, list] = defaultdict(list)
        for p in d_in:
            binned[assign_bin(p["geom"]["d_over_min"], edges)].append(p)
        chosen = []
        for b in range(args.n_bins):
            cand = sorted(binned.get(b, []), key=_pair_key)
            for p in pick(rng, cand, args.per_bin):
                chosen.append(dict(p, band=key, bin=b))
        # The far side: the nearest-MISS anchors, i.e. the pairs the current rule kept apart by
        # the least margin. The band's own above-cap tail is deliberately not sampled (it is
        # what `domain_cap` excludes); its size is reported so the omission is visible.
        for p in anch[:args.n_anchor]:
            chosen.append(dict(p, band=key, bin=None))
        selected.extend(chosen)
        band_meta[key] = dict(
            label=label, ratio_lo=lo, ratio_hi=(None if hi == math.inf else hi),
            n_pool=len(pool), n_dup_in_domain=len(d_in), n_dup_above_cap=len(d_out),
            n_anchor_available=len(anch), n_selected=len(chosen),
            n_anchor_selected=sum(1 for p in chosen if p["stratum"] == "anchor"),
            bin_edges=[round(e, 6) for e in edges],
            bin_counts={str(b): len(binned.get(b, [])) for b in range(args.n_bins)},
            anchor_d_over_min=[round(p["geom"]["d_over_min"], 4) for p in anch[:args.n_anchor]])

    # presentation: band section order, ascending d/min inside a band (the sort IS the
    # instrument). Strata interleave — nothing on the page distinguishes them.
    order = {k: i for i, (k, *_r) in enumerate(BANDS)}
    selected.sort(key=lambda p: (order[p["band"]], p["geom"]["d_over_min"], _pair_key(p)))

    for i, p in enumerate(selected):
        h = BMB._stable_seed(_pair_key(p))
        # id carries NO stratum, fate or geometry — it is in the img src and the DOM.
        p["pair_id"] = f"c{i:03d}_{h:08x}"
        p["palette"] = PALETTE          # the same coloring for every tile on the sheet
        # seeded coin flip: the displacer is the wider frame in most dup pairs, so a fixed
        # side would be a tell that survives the blind.
        p["left"] = "a" if (BMB._stable_seed(f"{args.seed}:side:{p['pair_id']}") & 1) == 0 else "b"

    plan = dict(
        run=R.RUN_DIR.name, seed=args.seed, dedup_k=ps.DEDUP_K, dedup_scale=ps.DEDUP_SCALE,
        domain_cap=args.domain_cap, n_bins=args.n_bins, per_bin=args.per_bin,
        n_anchor_per_band=args.n_anchor,
        n_precanon_dup=n_dup_total, n_dup_pairs=len(dups), n_anchor_pairs=len(anchors),
        n_selected=len(selected),
        n_selected_dup=sum(1 for p in selected if p["stratum"] == "dup"),
        n_selected_anchor=sum(1 for p in selected if p["stratum"] == "anchor"),
        sampling=("fw-ratio band -> rank-quantile bins of d/min(fw) over the band's dup pairs "
                  f"with d/min <= {args.domain_cap}, {args.per_bin} per bin by seeded sample; "
                  f"plus the {args.n_anchor} nearest-miss anchor pairs per band. Seeded "
                  f"(seed={args.seed}), no wall-clock input."),
        crop=dict(w=CROP_W, h=CROP_H, ss=CROP_SS, filter=BMB.CROP_FILTER,
                  interior=BMB.INTERIOR_MODE, composition=BMB.COMPOSITION,
                  palette=PALETTE, palette_source=PALETTE_SOURCE.name),
        bands=band_meta,
        by_partition=dict(Counter(p["partition"] for p in selected)),
        pairs=selected)
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1, default=str), encoding="utf-8")

    for key, label, *_r in BANDS:
        m = band_meta[key]
        print(f"  band {label:>7}: {m['n_selected']:>3} selected "
              f"({m['n_selected'] - m['n_anchor_selected']} dup + {m['n_anchor_selected']} "
              f"anchor) from {m['n_dup_in_domain']} in-domain dup pairs "
              f"(+{m['n_dup_above_cap']} above cap), {m['n_anchor_available']} anchors")
        print(f"           bin edges {[round(e, 3) for e in m['bin_edges']]}")
        print(f"           anchors at d/min {m['anchor_d_over_min']}")
    print(f"{len(selected)} pairs total ({plan['n_selected_dup']} dup, "
          f"{plan['n_selected_anchor']} anchor); partitions "
          f"{json.dumps(plan['by_partition'])}")
    print(f"-> {PLAN}")
    return 0


# =========================================================================== #
# render — both sides of every pair, matched palette, never mixed substrates
# =========================================================================== #
def render_block(side: dict, palette: str) -> dict:
    row = dict(cx=side["cx"], cy=side["cy"], fw=side["fw"], family=side["partition"],
               julia_c_re=side.get("julia_c_re"), julia_c_im=side.get("julia_c_im"),
               _palette=palette)
    row.update(side.get("phoenix") or {})
    rend = bq._render_block(row)
    rend.update(width=CROP_W, height=CROP_H, ss=CROP_SS)   # this sheet's stated fidelity
    return rend


def stage_render(args) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    bq._PHOENIX_POOL_CACHE.update(bq._phoenix_points())
    RENDERS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in plan["pairs"]:
        for which in ("a", "b"):
            out = RENDERS / f"{p['pair_id']}.{which}.jpg"
            if out.exists() and not args.force:
                continue
            jobs.append((render_block(p[which], p["palette"]), out))
    print(f"{len(plan['pairs'])} pairs, {len(jobs)} renders at {CROP_W}x{CROP_H} ss{CROP_SS} "
          f"in {PALETTE} ({RENDER_WORKERS}x{RENDER_THREADS} threads)", flush=True)

    def one(job):
        rend, out = job
        try:
            cc.render_corpus_crop(rend, str(out), palette_source=PALETTE_SOURCE,
                                  timeout=args.render_timeout, threads=RENDER_THREADS)
        except BaseException:
            out.unlink(missing_ok=True)     # a truncated jpg reads as rendered forever
            raise
        return out.name

    fails = []
    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                name = fut.result()
                if i % 25 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] {name}", flush=True)
            except Exception as e:                                   # noqa: BLE001
                fails.append((futs[fut][1].name, str(e)[:200]))
    for n, e in fails:
        print(f"  !! {n}: {e}")
    return 1 if fails else 0


# =========================================================================== #
# the sheet
# =========================================================================== #
def _num(x, n=6):
    return "&mdash;" if x is None else f"{x:.{n}g}"


def _reveal_rows(p: dict, which: str) -> str:
    """Everything hidden until `r`. NOTHING here may leak into the blind DOM's visible text."""
    s = p[which]
    rows = [("role", "candidate the run discarded" if s["kind"] == "candidate"
             else f"admitted outcome <span class=mono>{escape(str(s['id']))}</span>"),
            ("partition", escape(s["partition"])),
            ("centre", f"{_num(s['cx'], 12)}, {_num(s['cy'], 12)}"),
            ("fw", _num(s["fw"], 8)),
            ("fate", escape(str(s["fate"]))),
            ("decode", "&mdash;" if s["decoded_class"] is None
             else f"class {s['decoded_class']} (machine, not a label)"),
            ("c", "&mdash;" if s.get("julia_c_re") is None
             else f"{_num(float(s['julia_c_re']), 10)}, {_num(float(s['julia_c_im']), 10)}")]
    return "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in rows)


def _card(p: dict, which: str) -> str:
    return (f'<figure class="side">'
            f'<img loading="lazy" src="{RENDERS_URL}/{p["pair_id"]}.{which}.jpg" alt="">'
            f'<figcaption class="rv"><div class="kv">{_reveal_rows(p, which)}</div>'
            f'</figcaption></figure>')


def _repo_rel(p: Path) -> str | None:
    """Repo-relative POSIX path, or None when the target is outside the tree (a tmp_path in a
    test) — a URL hint is not worth raising over."""
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return None


def pair_section(p: dict, dedup_k: float) -> str:
    """One pair's markup. EVERY geometry number, fate and rule verdict lives inside a `rv`
    element — the blind is the CSS class, so this function is what `test_blind_dom` reads."""
    g, lf = p["geom"], p["left"]
    rt = "b" if lf == "a" else "a"
    return (f'<section class="pair" id="pair-{escape(p["pair_id"])}" '
            f'data-pid="{escape(p["pair_id"])}">'
            f'<div class="imgs">{_card(p, lf)}{_card(p, rt)}</div>'
            f'<div class="ctl">'
            f'<span class="idx"></span>'
            f'<button class="vb same" data-v="same">SAME <span class="kc">1</span></button>'
            f'<button class="vb distinct" data-v="distinct">DISTINCT '
            f'<span class="kc">2</span></button>'
            f'<button class="vb unsure" data-v="unsure">UNSURE '
            f'<span class="kc">3</span></button>'
            f'<span class="rv geom">stratum <b>{escape(p["stratum"])}</b> · '
            f'bin {p["bin"] if p["bin"] is not None else "&mdash;"} · '
            f'd {_num(g["dist"], 4)} · <b>d/min(fw) {_num(g["d_over_min"], 4)}</b> · '
            f'd/max(fw) {_num(g["d_over_max"], 4)} · fw ratio '
            f'{_num(g["fw_ratio"], 4)}&times; · current rule: '
            f'{"CUT" if g["dist"] < dedup_k * g["fw_max"] else "kept apart"}'
            f'</span></div></section>')


def stage_sheet(args) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    pairs = plan["pairs"]
    band_label = {k: lab for k, lab, *_r in BANDS}

    parts = [f"""
<header>
<h1>Dedup boundary calibration &mdash; where does “the same place” end?</h1>
<p><b>The call.</b> Each row is two renders. Mark <b>SAME</b> if they are close enough that
keeping both would be keeping one picture twice, <b>DISTINCT</b> if the second is worth
preserving on its own, <b>UNSURE</b> if you cannot tell. Nothing here is a quality judgment:
an ugly pair can still be two distinct places, and a beautiful one can still be a duplicate.</p>
<p><b>The order is the instrument.</b> Inside a section the pairs run from closest to furthest
apart on the rule's own decision variable, so you are looking for the row where your answer
flips &mdash; not labelling each pair cold. Sections split on how different the two frame
widths are ({escape(band_label['le2'])} / {escape(band_label['mid'])} /
{escape(band_label['gt10'])}); judge each section on its own.</p>
<p><b>Blind.</b> No fate, no rule verdict and no number is shown while you judge. Press
<kbd>r</kbd> to reveal them afterwards &mdash; the reveal state at the moment of each verdict is
exported with it, so a revealed verdict stays usable and stays flagged.</p>
<p class="dim">One image per location, every tile on the sheet in the same
<code>{escape(str(plan['crop']['palette']))}</code> coloring, at
{plan['crop']['w']}&times;{plan['crop']['h']} ss{plan['crop']['ss']}, {plan['crop']['filter']},
centre, black interior &mdash; so any difference you see is geometry. Left/right is a seeded
coin flip. This sheet is a dedup-identity instrument &mdash; it is not the label corpus and
nothing here becomes a label.</p>
<div class="keys"><b>keys</b> <kbd>1</kbd> same · <kbd>2</kbd> distinct · <kbd>3</kbd> unsure
<span class="dim">&mdash; judge and jump to the next unjudged pair</span> &nbsp;·&nbsp;
<kbd>s</kbd>/<kbd>d</kbd>/<kbd>u</kbd> same as above but step to the next pair in order ·
<kbd>&uarr;</kbd>/<kbd>&darr;</kbd> move · <kbd>n</kbd> next unjudged · <kbd>r</kbd> reveal</div>
</header>
<div id="bar">
  <span id="counts"></span><span class="spacer"></span>
  <button id="btn-reveal">reveal (r)</button>
  <button id="btn-next">next unjudged (n)</button>
  <button id="btn-export">export verdicts.json</button>
  <label class="fbtn">load verdicts<input id="load" type="file" accept=".json" hidden></label>
</div>"""]

    for key, label, *_r in BANDS:
        sel = [p for p in pairs if p["band"] == key]
        if not sel:
            continue
        m = plan["bands"][key]
        parts.append(
            f'<section class="bandhead"><h2>fw ratio {escape(label)} '
            f'<span class="dim">&mdash; {len(sel)} pairs, closest first</span></h2>'
            f'<p class="rv dim">bin edges of d/min(fw): '
            f'{escape(", ".join(str(e) for e in m["bin_edges"]))} &nbsp;·&nbsp; '
            f'{m["n_dup_in_domain"]} in-domain dup pairs (+{m["n_dup_above_cap"]} above the '
            f'{plan["domain_cap"]} cap), {m["n_anchor_available"]} anchors available</p>'
            f'</section>')
        for p in sel:
            parts.append(pair_section(p, plan["dedup_k"]))

    payload = json.dumps({p["pair_id"]: dict(
        pair_id=p["pair_id"], band=p["band"], bin=p["bin"], stratum=p["stratum"],
        partition=p["partition"], left=p["left"], palette=p["palette"],
        geom=p["geom"], a=p["a"], b=p["b"]) for p in pairs}, default=str,
        separators=(",", ":"))
    meta = json.dumps({k: plan[k] for k in
                       ("run", "seed", "dedup_k", "dedup_scale", "domain_cap", "n_bins",
                        "per_bin", "n_anchor_per_band", "n_precanon_dup", "n_dup_pairs",
                        "n_anchor_pairs", "n_selected", "sampling", "crop")},
                      default=str, separators=(",", ":"))
    body = "\n".join(parts)
    html = (PAGE.replace("__BODY__", body).replace("__PAIRS__", payload)
            .replace("__META__", meta).replace("__ORDER__",
                                               json.dumps([p["pair_id"] for p in pairs])))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(html, encoding="utf-8")
    print(f"-> {SHEET}\n   {len(pairs)} pairs, {2 * len(pairs)} tiles, one per location")
    rel = _repo_rel(SHEET)
    if rel:
        print(f"   serve: uv run python tools/viz/serve.py   then open "
              f"http://127.0.0.1:8010/{rel}")
    return 0


def stage_verify(args) -> int:
    """Load every `src` THE WAY A BROWSER WOULD — resolved against the page URL with `urljoin`,
    not fetched from the server root (that is a different URL and passes on a page whose images
    all 404). Also compares served bytes to disk, so a stale or half-written jpg fails."""
    import hashlib
    import urllib.request
    from urllib.parse import urljoin, urlsplit

    page = args.url or f"http://127.0.0.1:{args.port}/{SHEET.relative_to(ROOT).as_posix()}"
    html = urllib.request.urlopen(page, timeout=args.timeout).read().decode("utf-8")
    srcs = sorted(set(re.findall(r'src="([^"]+)"', html)))
    if not srcs:
        print("!! no <img src> in the page at all")     # a link check that passes on zero
        return 1                                        # links is not a link check
    ok, bad, mism = 0, [], []
    for s in srcs:
        url = urljoin(page, s)
        try:
            b = urllib.request.urlopen(url, timeout=args.timeout).read()
        except Exception as e:                                          # noqa: BLE001
            bad.append((s, urlsplit(url).path, str(e)[:80]))
            continue
        ok += 1
        disk = RENDERS / Path(s).name
        if disk.exists() and hashlib.sha1(b).hexdigest() != hashlib.sha1(
                disk.read_bytes()).hexdigest():
            mism.append(s)
    print(f"{page}\n  {ok}/{len(srcs)} images load as the browser resolves them; "
          f"{len(bad)} failed, {len(mism)} byte-mismatched")
    for s, path, e in bad[:5]:
        print(f"  !! src={s}\n     -> {path}\n     {e}")
    return 1 if (bad or mism) else 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dedup boundary calibration — same or distinct</title>
<style>
:root{color-scheme:dark;--bg:#0e0f13;--fg:#dfe3e8;--dim:#8d95a0;--line:#242832;--card:#15171d;
      --same:#c2543f;--dist:#4f9d69;--uns:#c79a3a;--cur:#6a8fd0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 "Segoe UI",system-ui,sans-serif;padding:0 0 90px}
header{max-width:1080px;padding:22px 26px 4px}
h1{font-size:21px;margin:0 0 10px}
h2{font-size:16px;margin:0}
header p{margin:7px 0;color:#c4cad2}
kbd{background:#1c1f29;border:1px solid #2c2f3a;border-radius:3px;padding:0 5px;font-size:12px}
code{background:#1b1f25;padding:1px 5px;border-radius:3px;font-size:12.5px}
.dim{color:var(--dim)}
.keys{margin:12px 0 4px;color:var(--dim);font-size:12.5px}
#bar{position:sticky;top:0;z-index:8;display:flex;gap:12px;align-items:center;
     background:#12141a;border-top:1px solid var(--line);border-bottom:1px solid var(--line);
     padding:8px 26px;margin-top:14px;font-size:12.5px}
#bar .spacer{flex:1}
button,.fbtn{font:inherit;background:#1c1f29;color:var(--fg);border:1px solid #2c2f3a;
     border-radius:5px;padding:5px 11px;cursor:pointer}
button:hover,.fbtn:hover{background:#262a36}
#counts b{color:#fff}
.bandhead{padding:30px 26px 4px;border-top:1px solid var(--line);margin-top:22px}
.pair{margin:14px 0;padding:10px 26px;border-left:4px solid transparent}
.pair.cur{border-left-color:var(--cur);background:#12141a}
.pair.v-same{border-left-color:var(--same)}
.pair.v-distinct{border-left-color:var(--dist)}
.pair.v-unsure{border-left-color:var(--uns)}
.imgs{display:flex;gap:10px;flex-wrap:wrap}
.side{margin:0;flex:0 0 auto;width:640px;max-width:calc(50vw - 40px);
      background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.side img{display:block;width:100%;height:auto;background:#000}
figcaption{padding:8px 10px;font-size:12px;border-top:1px solid var(--line)}
.kv{display:grid;grid-template-columns:78px 1fr;gap:1px 8px}
.kv .k{color:var(--dim)} .kv .v{overflow-wrap:anywhere}
.ctl{display:flex;gap:10px;align-items:center;margin-top:8px;flex-wrap:wrap}
.idx{color:var(--dim);font:11px ui-monospace,Consolas,monospace;min-width:74px}
.vb{padding:6px 18px;letter-spacing:.04em;font-size:12.5px}
.kc{opacity:.5;font-size:11px;margin-left:5px}
.vb.on .kc{opacity:.75}
.vb.same.on{background:var(--same);border-color:var(--same);color:#fff}
.vb.distinct.on{background:var(--dist);border-color:var(--dist);color:#06180c}
.vb.unsure.on{background:var(--uns);border-color:var(--uns);color:#1a1206}
.geom{font:11.5px ui-monospace,Consolas,monospace;color:var(--dim)}
.mono{font-family:ui-monospace,Consolas,monospace}
.rv{display:none}
body.reveal .rv{display:block}
body.reveal .ctl .rv{display:inline}
</style></head><body>
__BODY__
<script>
const PAIRS=__PAIRS__, META=__META__, ORDER=__ORDER__;
const LS='precanon_calibration::'+META.run+'::'+META.seed;
// { pair_id: {verdict, revealed, ts_ms} } — `revealed` is the reveal state AT THE MOMENT the
// verdict was set, not the state at export: a verdict taken blind stays a blind verdict.
let V=(()=>{try{return JSON.parse(localStorage.getItem(LS))||{}}catch(e){return {}}})();
let cur=0, revealed=false;
const els=ORDER.map(pid=>document.getElementById('pair-'+pid));
function save(){localStorage.setItem(LS,JSON.stringify(V));}

function paint(i){
  const pid=ORDER[i], el=els[i]; if(!el)return;
  const v=V[pid]&&V[pid].verdict;
  el.className='pair'+(v?(' v-'+v):'')+(i===cur?' cur':'');
  el.querySelectorAll('.vb').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
  const r=V[pid]&&V[pid].revealed;
  el.querySelector('.idx').textContent=(i+1)+'/'+ORDER.length+(v?(r?' ·rev':''):' ·—');
}
function counts(){
  let n={same:0,distinct:0,unsure:0},rv=0;
  for(const pid of ORDER){const e=V[pid]; if(e&&e.verdict){n[e.verdict]++; if(e.revealed)rv++;}}
  const done=n.same+n.distinct+n.unsure;
  document.getElementById('counts').innerHTML='<b>'+done+'/'+ORDER.length+'</b> judged &nbsp; '+
    n.same+' same / '+n.distinct+' distinct / '+n.unsure+' unsure'+
    (rv?(' &nbsp; · '+rv+' set while revealed'):'');
}
// `skip`: 1/2/3 jump to the next UNJUDGED pair (the fast pass down the sort, which never
// re-offers a pair already answered); s/d/u and the buttons step to the next pair in order
// (the revisit path, so re-judging one row does not teleport you out of the region).
function setV(v,skip){
  const pid=ORDER[cur];
  V[pid]={verdict:v,revealed:revealed,ts_ms:Date.now()};
  save(); paint(cur); counts();
  if(skip)nextUn();
  else if(cur<ORDER.length-1)go(cur+1);
}
function go(i){const o=cur;cur=Math.max(0,Math.min(ORDER.length-1,i));paint(o);paint(cur);
  els[cur].scrollIntoView({block:'center',behavior:'smooth'});}
function nextUn(){for(let k=1;k<=ORDER.length;k++){const i=(cur+k)%ORDER.length;
  if(!(V[ORDER[i]]&&V[ORDER[i]].verdict)){go(i);return;}}}

els.forEach((el,i)=>{
  el.querySelectorAll('.vb').forEach(b=>b.onclick=e=>{e.stopPropagation();cur=i;setV(b.dataset.v);});
  el.addEventListener('click',()=>{const o=cur;cur=i;paint(o);paint(cur);});
});
document.getElementById('btn-reveal').onclick=()=>{revealed=!revealed;
  document.body.classList.toggle('reveal',revealed);};
document.getElementById('btn-next').onclick=nextUn;
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  const k=e.key.toLowerCase();
  if(k==='1'){setV('same',true);e.preventDefault();}
  else if(k==='2'){setV('distinct',true);e.preventDefault();}
  else if(k==='3'){setV('unsure',true);e.preventDefault();}
  else if(k==='s'){setV('same');e.preventDefault();}
  else if(k==='d'){setV('distinct');e.preventDefault();}
  else if(k==='u'){setV('unsure');e.preventDefault();}
  else if(k==='n'){nextUn();e.preventDefault();}
  else if(k==='r'){document.getElementById('btn-reveal').click();e.preventDefault();}
  else if(e.key==='ArrowDown'||e.key==='ArrowRight'){go(cur+1);e.preventDefault();}
  else if(e.key==='ArrowUp'||e.key==='ArrowLeft'){go(cur-1);e.preventDefault();}
});
document.getElementById('btn-export').onclick=()=>{
  const rows=ORDER.map(pid=>{const p=PAIRS[pid],e=V[pid]||{};
    return {pair_id:pid,band:p.band,bin:p.bin,stratum:p.stratum,partition:p.partition,
            left:p.left,palette:p.palette,
            d:p.geom.dist,d_over_min:p.geom.d_over_min,d_over_max:p.geom.d_over_max,
            fw_ratio:p.geom.fw_ratio,fw_min:p.geom.fw_min,fw_max:p.geom.fw_max,
            a:p.a,b:p.b,
            verdict:e.verdict||null,revealed:e.verdict?!!e.revealed:null,ts_ms:e.ts_ms||null};});
  const out={schema:'precanon_calibration_verdicts/1',meta:META,
             exported_ms:Date.now(),
             n_judged:rows.filter(r=>r.verdict).length,n_total:rows.length,
             n_revealed_at_verdict:rows.filter(r=>r.revealed).length,verdicts:rows};
  const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='precanon_calibration_verdicts.json';a.click();URL.revokeObjectURL(a.href);
};
document.getElementById('load').onchange=ev=>{
  const f=ev.target.files[0];if(!f)return;const rd=new FileReader();
  rd.onload=()=>{try{const j=JSON.parse(rd.result);
    const src=Array.isArray(j.verdicts)?j.verdicts:[];
    for(const r of src){if(r.pair_id&&r.verdict)
      V[r.pair_id]={verdict:r.verdict,revealed:!!r.revealed,ts_ms:r.ts_ms||null};}
    save();ORDER.forEach((_,i)=>paint(i));counts();
  }catch(err){alert('bad verdicts.json: '+err);}};
  rd.readAsText(f);};
ORDER.forEach((_,i)=>paint(i));counts();
</script></body></html>
"""


# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--domain-cap", type=float, default=DOMAIN_CAP)
    p.add_argument("--n-bins", type=int, default=N_BINS)
    p.add_argument("--per-bin", type=int, default=PER_BIN)
    p.add_argument("--n-anchor", type=int, default=N_ANCHOR)
    p.set_defaults(fn=stage_plan)
    r = sub.add_parser("render")
    r.add_argument("--force", action="store_true")
    r.add_argument("--render-timeout", type=float, default=600.0)
    r.set_defaults(fn=stage_render)
    sub.add_parser("sheet").set_defaults(fn=stage_sheet)
    v = sub.add_parser("verify")
    v.add_argument("--port", type=int, default=8010)
    v.add_argument("--url", default=None)
    v.add_argument("--timeout", type=float, default=20.0)
    v.set_defaults(fn=stage_verify)
    a = ap.parse_args()
    cc.set_below_normal_priority()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
