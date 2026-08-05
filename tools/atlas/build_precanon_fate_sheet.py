#!/usr/bin/env python
r"""build_precanon_fate_sheet.py — the 28 discarded class-4s beside the rows that displaced them.

AN INSPECTION SHEET, NOT A LABELLING SHEET. Every tile is captioned with its coordinates, its
fate and the arithmetic of the cut that fired; nothing here is blind and nothing here writes a
label. It exists to answer ONE question with Matt's eye: does the pre-canonical dedup
(`production_seeder.is_distinct`, `DEDUP_K = 1.5 x max(fw)`) over-fire on the top tier — i.e.
is the row that displaced a human-labelled 4 actually the same picture?

THE POPULATION IS THE SITTING'S OWN TOP TIER. `labels/v2_sitting_sheet_v1.json` has 35 class-4
labels; all 35 join 1:1 to the `harvest_v2_proving_20260803` queue on the candidate
`(cx, cy, fw)`, and their fates are precanon_dup 28 / canon_not_q3 3 / below_tau_h 2 /
admitted 2. The 28 are the sheet; the 3 + 2 are an appendix (their crops already exist, so
completeness costs nothing); the 2 admitted are not in question.

WHAT IS RENDERED AND WHAT IS REUSED. The discarded side was already presented to Matt in the
sitting, so its crops are reused byte-for-byte out of the registered batch's crop tree — a
re-render would be a second, subtly different presentation of a picture whose label already
exists. Only the 14 displacers are new: they are ledger rows that were never in a batch, so
they have no crop at all. They are rendered at the SITTING's presentation, not the corpus
default — 1280x720 **ss2** (`sitting_cutter.SITTING_CROP_SS`, that batch's recorded
deviation), lanczos3, center, black interior, canonical + `blue_orange` vivid companion.

THE PALETTE PAIRING IS DELIBERATE. A sitting row's canonical palette is a seeded per-row draw
off its `image_id`, and a displacer has no `image_id` to draw from. Rather than invent one,
each displacer's canonical is rendered in **its first partner's palette**, so the primary pair
is palette-controlled and any difference the eye sees is geometry rather than colour. Where a
displacer has several partners the other pairs are not colour-matched — the fixed-palette
vivid companion is the controlled comparison for those, and every caption names its palette so
this is never guessed at.

READ-ONLY on production data. The join reads the label store, the sheet route, the registered
batch and the run's three records; nothing under `data/` is written and no admission constant
is touched.

  uv run python tools/atlas/build_precanon_fate_sheet.py plan     # the join, no renders
  uv run python tools/atlas/build_precanon_fate_sheet.py render   # 14 displacers x 2
  uv run python tools/atlas/build_precanon_fate_sheet.py sheet    # the HTML
"""
from __future__ import annotations

import argparse
import json
import math
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

import paths                                   # noqa: E402
import corpus_common as cc                     # noqa: E402
import build_minibrot_batch as BMB             # noqa: E402
import build_q4_harvest_batches as bq          # noqa: E402  (_render_block — the render authority)
import production_seeder as ps                 # noqa: E402  (DEDUP_K / near_dup — the cut itself)
import sitting_cutter as sc                    # noqa: E402  (SITTING_BATCH / SITTING_CROP_SS)
import label_store as ls                       # noqa: E402

RUN_DIR = ROOT / "data" / "discovery" / "harvest_v2_proving_20260803"
SHEET_BATCH = "2026-08-03_v2_sitting_sheet_v1"
LABELS = ROOT / "labels" / "v2_sitting_sheet_v1.json"

OUT = Path(paths.scratch("precanon_fate"))
RENDERS = OUT / "renders"
PLAN = OUT / "pairs.json"
SHEET = OUT / "precanon_fate_sheet.html"

# renders are served from the repo tree, so the page's src is this path relative to the root
RENDERS_URL = "scratch/precanon_fate/renders"

TOP_CLASS = 4
RENDER_WORKERS, RENDER_THREADS = 3, 4          # 3 processes x 4 threads on a 12-core box


# =========================================================================== #
# the join
# =========================================================================== #
def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _gkey(cx, cy, fw) -> tuple:
    """The candidate geometry as a join key. The batch serializes coordinates as decimal
    strings and the run's records keep floats, so both sides round-trip through the same
    17-significant-digit repr rather than comparing a string to a float."""
    return (f"{float(cx):.17g}", f"{float(cy):.17g}", f"{float(fw):.17g}")


def _ident(row: dict, partition: str):
    """The row's dup-identity vector, same rule as `production_seeder.row_ident`: the julia
    seed c (2-D), the phoenix (c, p, z_-1) point (6-D), or None on a c-plane row."""
    if partition.startswith("phoenix"):
        keys = ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im")
        pref = ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
                "phoenix_zm1_re", "phoenix_zm1_im")
        vals = []
        for k, pk in zip(keys, pref):
            v = row.get(pk)
            if v is None:
                v = row.get(k)
            vals.append(None if v is None else float(v))
        return None if any(v is None for v in vals) else tuple(vals)
    cre, cim = row.get("julia_c_re"), row.get("julia_c_im")
    if cre is None:
        cre, cim = row.get("c_re"), row.get("c_im")
    if cre is None:
        return None
    return (float(cre), float(cim))


def build_join() -> dict:
    """Every class-4 label in the sitting, its queue fate, and for a `precanon_dup` the
    displacing ledger row plus the arithmetic of the cut that fired."""
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    route = json.loads((Path(cc.batch_dir(SHEET_BATCH)) / "route.json").read_text("utf-8"))
    batch = {r["image_id"]: r for r in _jl(Path(cc.batch_dir(sc.SITTING_BATCH)) / "images.jsonl")}

    cand, hlog = {}, {}
    for c in _jl(RUN_DIR / "q4_candidates.jsonl"):          # first occurrence wins, as the queue
        cand.setdefault(_gkey(c["cx"], c["cy"], c["fw"]), c)
    for h in _jl(RUN_DIR / "harvest_log.jsonl"):
        hlog.setdefault(_gkey(h["cx"], h["cy"], h["fw"]), h)
    ledger = {r["id"]: r for r in _jl(RUN_DIR / "outcome_ledger.jsonl")}

    fours = sorted(k for k, v in labels.items() if v.get("score") == TOP_CLASS)
    items = []
    for sid in fours:
        bid = route[sid]["image_id"]
        brow = batch[bid]
        rend = brow["render"]
        k = _gkey(rend["cx"], rend["cy"], rend["fw"])
        c = cand.get(k)
        if c is None:
            raise SystemExit(f"{sid} does not join the run queue on {k}")
        h = hlog.get(k)
        it = dict(sheet_id=sid, batch_image_id=bid, partition=c["partition"],
                  fate=c["fate"], cx=float(c["cx"]), cy=float(c["cy"]), fw=float(c["fw"]),
                  palette=rend["palette"], render=rend,
                  canon_decoded=c.get("canon_decoded"), canon_pgood=c.get("canon_pgood"),
                  tau_h=c.get("tau_h"), t_good=c.get("t_good"),
                  mix_source=c.get("mix_source"), queue_rank=None,
                  dup_of=(h or {}).get("precanon_dup"))
        if it["fate"] == "precanon_dup":
            if not it["dup_of"]:
                raise SystemExit(f"{sid} is precanon_dup with no dup_of in harvest_log")
            d = ledger.get(it["dup_of"])
            if d is None:
                raise SystemExit(f"{sid}: dup_of {it['dup_of']} does not resolve in the ledger")
            it["cut"] = cut_arithmetic(it, d, rend)
        items.append(it)

    displacers = {}
    for it in items:
        if it["fate"] != "precanon_dup":
            continue
        did = it["dup_of"]
        d = ledger[did]
        displacers.setdefault(did, dict(
            id=did, family=d["family"], cx=float(d["outcome_cx"]), cy=float(d["outcome_cy"]),
            fw=float(d["outcome_fw"]), decoded_class=d.get("decoded_class"),
            p_good=d.get("p_good"), p_ge4=d.get("p_ge4"), mix_source=d.get("mix_source"),
            julia_c_re=d.get("julia_c_re"), julia_c_im=d.get("julia_c_im"),
            phoenix={k: d.get(k) for k in ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re",
                                           "phoenix_p_im", "phoenix_zm1_re", "phoenix_zm1_im")
                     if d.get(k) is not None} or None,
            members=[]))["members"].append(it["sheet_id"])

    # the palette pairing: a displacer borrows its FIRST partner's canonical palette
    by_sid = {it["sheet_id"]: it for it in items}
    for did, d in displacers.items():
        d["palette"] = by_sid[d["members"][0]]["palette"]

    return dict(items=items, displacers=displacers,
                fates=dict(Counter(it["fate"] for it in items)),
                n_fours=len(fours), n_labels=len(labels))


def cut_arithmetic(it: dict, d: dict, rend: dict) -> dict:
    """The fired cut, restated from the candidate and the displacer: plane distance, the
    `DEDUP_K x max(fw)` radius it was compared against, and — where the family keys on one —
    the parameter-identity clause that had to pass first."""
    dist = math.hypot(it["cx"] - float(d["outcome_cx"]), it["cy"] - float(d["outcome_cy"]))
    fmax = max(it["fw"], float(d["outcome_fw"]))
    a_c = _ident(rend if it["partition"].startswith("phoenix") else it_row(it), it["partition"])
    b_c = _ident(d, it["partition"])
    ident_d = None
    if a_c is not None and b_c is not None and len(a_c) == len(b_c):
        ident_d = math.dist(a_c, b_c)
    fired = ps.near_dup(it["cx"], it["cy"], it["fw"],
                        d["outcome_cx"], d["outcome_cy"], d["outcome_fw"],
                        ps.DEDUP_K, a_c=a_c, b_c=b_c)
    return dict(dist=dist, fw_max=fmax, radius=ps.DEDUP_K * fmax, k=ps.DEDUP_K,
                dist_in_fwmax=dist / fmax if fmax else None,
                fw_ratio=float(d["outcome_fw"]) / it["fw"] if it["fw"] else None,
                ident_dist=ident_d, ident_eps=ps.JULIA_SAME_C_EPS,
                ident_kind=("phoenix (c,p,z-1)" if it["partition"].startswith("phoenix")
                            else "julia seed c" if a_c is not None else None),
                reproduced=bool(fired))


def it_row(it: dict) -> dict:
    """The candidate's julia identity as `near_dup` wants it (the render block carries the
    seed c as `c_re`/`c_im`)."""
    return dict(c_re=it["render"].get("c_re"), c_im=it["render"].get("c_im"))


# =========================================================================== #
# render — the 14 displacers, at the SITTING's presentation
# =========================================================================== #
def displacer_render_block(d: dict) -> dict:
    row = dict(cx=d["cx"], cy=d["cy"], fw=d["fw"], family=d["family"],
               julia_c_re=d.get("julia_c_re"), julia_c_im=d.get("julia_c_im"),
               _palette=d["palette"])
    row.update(d.get("phoenix") or {})
    rend = bq._render_block(row)
    rend["ss"] = sc.SITTING_CROP_SS        # the sitting's recorded deviation, not the corpus 4
    return rend


def stage_render(args) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    bq._PHOENIX_POOL_CACHE.update(bq._phoenix_points())
    RENDERS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for did, d in plan["displacers"].items():
        rend = displacer_render_block(d)
        for tag, palette, src in (("canon", rend["palette"], BMB.PALETTE_SOURCE),
                                  ("vivid", BMB.VIVID_PALETTE, BMB.VIVID_SOURCE)):
            out = RENDERS / f"{did}.{tag}.jpg"
            if out.exists() and not args.force:
                continue
            jobs.append((dict(rend, palette=palette), out, src))
    print(f"{len(plan['displacers'])} displacers, {len(jobs)} renders "
          f"({RENDER_WORKERS}x{RENDER_THREADS} threads)", flush=True)

    def one(job):
        rend, out, src = job
        try:
            cc.render_corpus_crop(rend, str(out), palette_source=src,
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
                print(f"  [{i}/{len(jobs)}] {fut.result()}", flush=True)
            except Exception as e:                                   # noqa: BLE001
                fails.append((futs[fut][1].name, str(e)[:200]))
    for n, e in fails:
        print(f"  !! {n}: {e}")
    return 1 if fails else 0


# =========================================================================== #
# the sheet
# =========================================================================== #
def _crop_urls(batch_image_id: str) -> tuple[str, str]:
    """The sitting batch's own crop pair. Relocated out of the tree, but `tools/viz/serve.py`
    resolves this exact in-tree URL shape through `artifacts.resolve`."""
    base = f"data/label_corpus/batches/{sc.SITTING_BATCH}"
    return (f"{base}/crops/{batch_image_id}.jpg", f"{base}/vivid/{batch_image_id}.jpg")


def _num(x, n=6):
    return "—" if x is None else f"{x:.{n}g}"


def _card(*, title, badge, badge_cls, canon, vivid, rows, missing=False) -> str:
    imgs = (f'<div class="miss">no crop on disk</div>' if missing else
            f'<img loading="lazy" class="canon" src="{canon}" alt="">'
            f'<img loading="lazy" class="vivid" src="{vivid}" alt="">')
    kv = "".join(f'<div class="k">{escape(k)}</div><div class="v">{v}</div>' for k, v in rows)
    return (f'<figure class="card {badge_cls}">'
            f'<div class="imgs">{imgs}</div>'
            f'<figcaption><div class="hd"><span class="badge">{escape(badge)}</span>'
            f'<span class="tid">{escape(title)}</span></div>'
            f'<div class="kv">{kv}</div></figcaption></figure>')


def stage_sheet(args) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    items = {it["sheet_id"]: it for it in plan["items"]}
    disp = plan["displacers"]

    groups = sorted(disp.values(), key=lambda d: (-len(d["members"]), d["id"]))
    parts = []

    fat = plan["fates"]
    parts.append(f"""
<header>
<h1>Precanon fate sheet — the {fat.get('precanon_dup', 0)} discarded fours beside their displacers</h1>
<p class="q"><b>The question:</b> does the pre-canonical dedup over-fire on the top tier? Each
row below pairs a location Matt scored <b>4</b> — and which was thrown away before it was ever
decoded — with the ledger row that displaced it. If the pair is the same picture the cut is
doing its job; if it is not, <code>DEDUP_K</code> is deleting the best material in the run.</p>
<p class="pop"><b>Population:</b> the {plan['n_fours']} class-4 labels in
<code>labels/v2_sitting_sheet_v1.json</code> ({plan['n_labels']} labelled rows), joined 1:1 to the
<code>harvest_v2_proving_20260803</code> queue on the candidate (cx, cy, fw). Fates:
<b>precanon_dup {fat.get('precanon_dup', 0)}</b> · canon_not_q3 {fat.get('canon_not_q3', 0)} ·
below_tau_h {fat.get('below_tau_h', 0)} · admitted {fat.get('admitted', 0)}. The
{fat.get('precanon_dup', 0)} are this sheet, collapsing to <b>{len(disp)} distinct displacers</b>;
the {fat.get('canon_not_q3', 0)} + {fat.get('below_tau_h', 0)} are in the appendix.</p>
<p class="rule"><b>The cut:</b> <code>production_seeder.is_distinct</code> — a candidate is a dup of a
q3-cloud row iff plane distance &lt; <code>DEDUP_K &times; max(fw_a, fw_b)</code> with
<code>DEDUP_K = {ps.DEDUP_K}</code>, and, for julia/phoenix, only if the two parameter identities
are within <code>{ps.JULIA_SAME_C_EPS:g}</code> first (julia: the seed <i>c</i>; phoenix: the whole
<i>(c, p, z<sub>-1</sub>)</i> point). Note <b>max</b>, not min: the wider of the two frames sets the
radius, so a wide outcome claims a disc that swallows much deeper candidates.</p>
<p class="note"><b>Reading the tiles.</b> Top image = canonical palette, bottom = the fixed
<code>blue_orange</code> vivid companion. The discarded side is the <i>exact crop Matt labelled</i>,
reused from the sitting batch. The displacers are new renders at the same presentation
(1280&times;720 ss{sc.SITTING_CROP_SS}, lanczos3, centre, black interior); each borrows its first
partner's canonical palette so that pair is colour-controlled — every caption names its palette, and
the vivid companion is the controlled comparison everywhere else.
<b>The displacers carry no human label</b> — they were admitted to the ledger and never served in a
sitting, so the only judgement on them is the machine decode shown, which is not a label.</p>
<div class="toggles">
  <label><input type="checkbox" id="tc" checked> canonical</label>
  <label><input type="checkbox" id="tv" checked> vivid</label>
  <label><input type="checkbox" id="tw"> wide (one pair per row)</label>
</div>
</header>""")

    # ---- summary table --------------------------------------------------- #
    cuts = [items[s]["cut"] for d in groups for s in d["members"]]
    ratios = sorted(c["fw_ratio"] for c in cuts)
    dfm = sorted(c["dist_in_fwmax"] for c in cuts)
    wider = sum(1 for r in ratios if r > 1)
    min_fw_survive = sum(1 for c in cuts
                         if c["dist"] >= c["k"] * (c["fw_max"] / max(c["fw_ratio"], 1.0)))
    med = lambda xs: xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1]
                                                           + xs[len(xs) // 2]) / 2   # noqa: E731
    parts.append(
        f'<section class="stats"><h2>What the arithmetic already says</h2><ul>'
        f'<li>In <b>{wider} of {len(cuts)}</b> pairs the displacer is the <b>wider</b> frame — '
        f'median <b>{med(ratios):.3g}&times;</b> wider, up to <b>{ratios[-1]:.3g}&times;</b>. '
        f'The radius is therefore set by the displacer in every case, and the discarded 4 sits '
        f'<i>inside</i> a frame it is a deep zoom of.</li>'
        f'<li>Distances are not tight: median <b>{med(dfm):.3g} &times; max(fw)</b> against a '
        f'{ps.DEDUP_K} cut, and only {sum(1 for x in dfm if x < 0.1)} of {len(cuts)} are inside '
        f'0.1.</li>'
        f'<li>Had the radius been scaled by <code>min(fw)</code> rather than '
        f'<code>max(fw)</code>, <b>{min_fw_survive} of {len(cuts)}</b> would have survived. '
        f'That is the whole decision this sheet informs — the eye is the arbiter of whether they '
        f'<i>should</i> have.</li>'
        f'<li>All {len(disp)} displacers carry machine decode class '
        f'{"/".join(str(k) for k in sorted({d["decoded_class"] for d in disp.values()}))} — this '
        f'is not a case of good material being displaced by bad.</li></ul></section>')

    srows = []
    for d in groups:
        for sid in d["members"]:
            it = items[sid]
            c = it["cut"]
            srows.append(
                f"<tr><td class=mono>{escape(d['id'].replace('harvest_v2_proving_20260803', '…'))}</td>"
                f"<td class=mono>{escape(sid)}</td><td>{escape(it['partition'])}</td>"
                f"<td class=n>{_num(c['dist'], 4)}</td>"
                f"<td class=n><b>{_num(c['dist_in_fwmax'], 3)}</b></td>"
                f"<td class=n>{_num(c['fw_ratio'], 3)}&times;</td>"
                f"<td class=n>{_num(it['fw'], 3)}</td><td class=n>{_num(d['fw'], 3)}</td>"
                f"<td class=n>{'—' if c['ident_dist'] is None else _num(c['ident_dist'], 2)}</td>"
                f"<td>{d['decoded_class']}</td></tr>")
    parts.append(
        "<section class=summary><h2>The 28, as arithmetic</h2>"
        "<p>“d / max(fw)” is the distance in units of the radius scale; the cut fires below "
        f"<b>{ps.DEDUP_K}</b>. “fw ratio” is displacer_fw / discarded_fw — above 1 the displacer is the "
        "<i>wider</i> frame and therefore the one setting the radius.</p>"
        "<table><thead><tr><th>displacer</th><th>discarded</th><th>partition</th><th>d</th>"
        f"<th>d / max(fw)</th><th>fw ratio</th><th>fw discarded</th><th>fw displacer</th>"
        f"<th>&Delta;ident</th><th>displacer decode</th></tr></thead><tbody>"
        + "".join(srows) + "</tbody></table></section>")

    # ---- the groups ------------------------------------------------------ #
    for gi, d in enumerate(groups, 1):
        cards = [_card(
            title=d["id"], badge="DISPLACER", badge_cls="disp",
            canon=f"{RENDERS_URL}/{d['id']}.canon.jpg",
            vivid=f"{RENDERS_URL}/{d['id']}.vivid.jpg",
            rows=[("partition", escape(d["family"])),
                  ("centre", f"{_num(d['cx'], 10)}, {_num(d['cy'], 10)}"),
                  ("fw", _num(d["fw"], 6)),
                  ("fate", "<b>admitted to the q3 cloud</b>"),
                  ("label", "<b>none — never served in a sitting</b>"),
                  ("machine decode", f"class <b>{d['decoded_class']}</b> "
                                     f"(p_good {_num(d['p_good'], 3)}, "
                                     f"p&ge;4 {_num(d['p_ge4'], 3)}) — not a label"),
                  ("palette", escape(d["palette"]) + " <span class=dim>(borrowed)</span>"),
                  ("mix", escape(str(d["mix_source"])))])]
        for sid in d["members"]:
            it = items[sid]
            c = it["cut"]
            canon, vivid = _crop_urls(it["batch_image_id"])
            ident = ("—" if c["ident_dist"] is None else
                     f"&Delta;{_num(c['ident_dist'], 2)} &lt; {c['ident_eps']:g} "
                     f"({escape(str(c['ident_kind']))}) — clause passed")
            cards.append(_card(
                title=sid, badge=f"DISCARDED · human {TOP_CLASS}", badge_cls="disc",
                canon=canon, vivid=vivid,
                rows=[("partition", escape(it["partition"])),
                      ("centre", f"{_num(it['cx'], 10)}, {_num(it['cy'], 10)}"),
                      ("fw", _num(it["fw"], 6)),
                      ("fate", "<b>precanon_dup</b> — discarded before any canonical decode"),
                      ("distance", f"d = {_num(c['dist'], 4)} = <b>{_num(c['dist_in_fwmax'], 3)}"
                                   f"</b> &times; max(fw), cut at {c['k']} &times; max(fw) = "
                                   f"{_num(c['radius'], 4)}"),
                      ("scale", f"displacer frame is <b>{_num(c['fw_ratio'], 3)}&times;</b> "
                                f"the discarded frame"),
                      ("identity", ident),
                      ("palette", escape(it["palette"])),
                      ("mix", escape(str(it["mix_source"])))]))
        parts.append(f'<section class="group"><h2>{gi}. '
                     f'{escape(d["id"])} <span class=dim>&mdash; displaced '
                     f'{len(d["members"])} class-4{"s" if len(d["members"]) > 1 else ""}</span></h2>'
                     f'<div class="row">{"".join(cards)}</div></section>')

    # ---- appendix -------------------------------------------------------- #
    app = [it for it in plan["items"] if it["fate"] in ("canon_not_q3", "below_tau_h")]
    acards = []
    for it in sorted(app, key=lambda r: (r["fate"], r["sheet_id"])):
        canon, vivid = _crop_urls(it["batch_image_id"])
        missing = not (Path(cc.crops_dir(sc.SITTING_BATCH)) / f"{it['batch_image_id']}.jpg").exists()
        why = ("its canonical decode did not reach q3" if it["fate"] == "canon_not_q3"
               else f"cheap p_good below the partition's &tau;_h = {_num(it['tau_h'], 3)}")
        acards.append(_card(
            title=it["sheet_id"], badge=f"{it['fate'].upper()} · human {TOP_CLASS}",
            badge_cls="app", canon=canon, vivid=vivid, missing=missing,
            rows=[("partition", escape(it["partition"])),
                  ("centre", f"{_num(it['cx'], 10)}, {_num(it['cy'], 10)}"),
                  ("fw", _num(it["fw"], 6)),
                  ("fate", f"<b>{escape(it['fate'])}</b> — {why}"),
                  ("canon decode", f"class {it['canon_decoded']} "
                                   f"(p_good {_num(it['canon_pgood'], 3)}, "
                                   f"t_good {_num(it['t_good'], 3)})"),
                  ("palette", escape(it["palette"])),
                  ("mix", escape(str(it["mix_source"])))]))
    parts.append('<section class="group appendix"><h2>Appendix — the other five fours the run '
                 'threw away</h2><p>Not dedup: these were cut by the score gates, and they are '
                 'here only so the 35 are accounted for. No renders were made for this section — '
                 'the crops are the sitting\'s own.</p>'
                 f'<div class="row">{"".join(acards)}</div></section>')

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(PAGE.replace("__BODY__", "\n".join(parts)), encoding="utf-8")
    print(f"-> {SHEET}")
    print(f"   {len(groups)} groups, {sum(len(d['members']) for d in groups)} discarded fours, "
          f"{len(acards)} appendix tiles")
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Precanon fate sheet — discarded fours vs their displacers</title>
<style>
:root{--bg:#111316;--fg:#e7e9ec;--dim:#9aa2ad;--line:#2a2f36;--gold:#d8a33a;--red:#c2543f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 "Segoe UI",system-ui,sans-serif;padding:24px 28px 80px}
h1{font-size:22px;margin:0 0 10px}
h2{font-size:16px;margin:34px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
header{max-width:1100px}
header p{margin:8px 0;color:#cdd3da}
code{background:#1b1f25;padding:1px 5px;border-radius:3px;font-size:12.5px}
.dim{color:var(--dim);font-weight:400}
.toggles{margin:16px 0 4px;display:flex;gap:18px;font-size:13px;color:var(--dim)}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
body.wide .row{flex-wrap:nowrap;overflow-x:auto;padding-bottom:8px}
.card{margin:0;width:430px;background:#171a1f;border:1px solid var(--line);border-radius:6px;
      overflow:hidden;flex:0 0 auto}
.card.disp{border-color:var(--gold);box-shadow:0 0 0 1px rgba(216,163,58,.25)}
.card.disc{border-color:var(--red)}
.card img{display:block;width:100%;height:auto;background:#000}
.card img+img{border-top:1px solid var(--line)}
body.nc .canon{display:none} body.nv .vivid{display:none}
.miss{padding:60px 12px;text-align:center;color:var(--dim);background:#000}
figcaption{padding:9px 11px 11px;font-size:12px}
.hd{display:flex;gap:8px;align-items:baseline;margin-bottom:7px}
.badge{font-size:10.5px;letter-spacing:.06em;padding:2px 6px;border-radius:3px;
       background:#23272e;color:var(--dim)}
.disp .badge{background:var(--gold);color:#1a1206}
.disc .badge{background:var(--red);color:#fff}
.tid{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--dim);
     overflow-wrap:anywhere}
.kv{display:grid;grid-template-columns:88px 1fr;gap:2px 8px}
.kv .k{color:var(--dim)} .kv .v{overflow-wrap:anywhere}
table{border-collapse:collapse;font-size:12px;margin-top:6px}
th,td{border:1px solid var(--line);padding:3px 7px;text-align:left}
th{background:#1b1f25;color:var(--dim);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
td.mono,.mono{font-family:ui-monospace,Consolas,monospace;font-size:11px}
.summary p{max-width:1100px;color:#cdd3da}
.appendix .card{border-color:var(--line)}
</style></head><body>
__BODY__
<script>
const b=document.body,c=document.getElementById('tc'),v=document.getElementById('tv'),
      w=document.getElementById('tw');
function sync(){b.classList.toggle('nc',!c.checked);b.classList.toggle('nv',!v.checked);
                b.classList.toggle('wide',w.checked);}
[c,v,w].forEach(e=>e.addEventListener('change',sync));sync();
</script></body></html>
"""


# =========================================================================== #
def stage_plan(args) -> int:
    plan = build_join()
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1, default=str), encoding="utf-8")
    print(f"{plan['n_fours']} class-4 labels  fates={json.dumps(plan['fates'])}")
    print(f"{len(plan['displacers'])} distinct displacers "
          f"({sum(len(d['members']) for d in plan['displacers'].values())} precanon_dup rows)")
    bad = [it["sheet_id"] for it in plan["items"]
           if it.get("cut") and not it["cut"]["reproduced"]]
    print(f"cut reproduced from the records: {len(plan['displacers']) and ''}"
          f"{sum(1 for it in plan['items'] if it.get('cut') and it['cut']['reproduced'])}"
          f"/{sum(1 for it in plan['items'] if it.get('cut'))}"
          + (f"  !! NOT reproduced: {bad}" if bad else ""))
    print(f"-> {PLAN}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=stage_plan)
    r = sub.add_parser("render")
    r.add_argument("--force", action="store_true")
    r.add_argument("--render-timeout", type=float, default=900.0)
    r.set_defaults(fn=stage_render)
    sub.add_parser("sheet").set_defaults(fn=stage_sheet)
    a = ap.parse_args()
    cc.set_below_normal_priority()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
