#!/usr/bin/env python
r"""precanon_minfw_sheet.py — the reject-autopsy appendix for the min(fw) replay.

WHAT THE EYE IS BEING ASKED. `precanon_minfw_replay.py` says 2,151 of the run's 2,531
`precanon_dup` rows survive a `min(fw)`-scaled radius (M0; 943 under M1). That is a volume,
not a verdict: the rows it would keep may be genuinely different places or may be the same
picture at a second zoom. This sheet shows a FATE-STRATIFIED sample of both sides — rows that
would newly survive, and rows the smaller radius still kills — each beside the ledger row that
displaced it, so the arithmetic and the picture are on the same page.

NOT THE TOP TIER. `build_precanon_fate_sheet.py` covers the 28 human-labelled class-4s; this
one deliberately samples ACROSS THE DISTANCE RANGE of the whole population (rank-quantiles of
d / min(fw)), because a decision taken on the top tier alone is a decision about 28 rows out of
2,531. Nothing here is labelled and nothing here is scored — the prompt's `Do not` list.

PRESENTATION IS MATCHED WITHIN A PAIR, which is the only comparison the eye can trust. Neither
side of a pair carries an `image_id` (the candidate was never served; the displacer was never
in a batch), so rather than borrow one — the fate sheet's compromise, which colour-matched only
the primary pair — BOTH sides are rendered in the SAME per-pair palette, seeded off the pair
key, plus the same fixed `blue_orange` vivid companion. Any difference the eye sees is
geometry. Same fidelity as the fate sheet: 1280x720 ss2 (`sitting_cutter.SITTING_CROP_SS`),
lanczos3, centre, black interior.

READ-ONLY on production data; every output lands under `scratch/precanon_minfw/`.

  uv run python tools/atlas/precanon_minfw_sheet.py plan     # sample + join, no renders
  uv run python tools/atlas/precanon_minfw_sheet.py render
  uv run python tools/atlas/precanon_minfw_sheet.py sheet
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
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
import build_q4_harvest_batches as bq           # noqa: E402  (_render_block — the render authority)
import production_seeder as ps                  # noqa: E402
import sitting_cutter as sc                     # noqa: E402  (SITTING_CROP_SS)
import precanon_minfw_replay as R               # noqa: E402  (the replay itself)

OUT = Path(paths.scratch("precanon_minfw"))
RENDERS = OUT / "renders"
PLAN = OUT / "sheet_pairs.json"
SHEET = OUT / "precanon_minfw_sheet.html"
# PAGE-RELATIVE, not root-relative. The renders sit beside the page, and a browser resolves a
# relative src against the PAGE's directory — a root-relative "scratch/.../renders/x.jpg" here
# would be requested as ".../precanon_minfw/scratch/precanon_minfw/renders/x.jpg" and 404. A
# link check that fetches the src from the server root instead of resolving it against the page
# URL passes on a URL the browser never requests; `verify` below resolves with `urljoin`.
RENDERS_URL = "renders"

N_NEW, N_REJ = 10, 5
RENDER_WORKERS, RENDER_THREADS = 3, 4          # 3 processes x 4 threads on a 12-core box


# =========================================================================== #
# sample + join
# =========================================================================== #
def _rank_spread(rows: list[dict], n: int, keyfn) -> list[dict]:
    """`n` rows spanning the full range of `keyfn`, at even RANK quantiles. Rank rather than
    value quantiles because these distributions run over four orders of magnitude and a
    value-spaced sample would put nine of ten draws in one decade."""
    s = sorted(rows, key=keyfn)
    if len(s) <= n:
        return s
    idx = sorted({round(i * (len(s) - 1) / (n - 1)) for i in range(n)})
    return [s[i] for i in idx]


def pair_of(r: dict, ledger: dict, tag: str) -> dict:
    d = ledger[r["rec_dup"]]
    dist = math.hypot(r["cx"] - float(d["outcome_cx"]), r["cy"] - float(d["outcome_cy"]))
    a, b = r["fw"], float(d["outcome_fw"])
    pid = f"{tag}_{r['batch']:03d}_{r['node_id']:05d}"
    return dict(
        pair_id=pid, side=tag, partition=r["partition"], batch=r["batch"],
        node_id=r["node_id"], depth=r["depth"], mix_source=r["mix_source"],
        cand=dict(cx=r["cx"], cy=r["cy"], fw=r["fw"], julia_c_re=r["julia_c_re"],
                  julia_c_im=r["julia_c_im"], phoenix=r["phoenix"],
                  cheap_pgood=r["cheap_pgood"]),
        disp=dict(id=d["id"], cx=float(d["outcome_cx"]), cy=float(d["outcome_cy"]),
                  fw=float(d["outcome_fw"]), decoded_class=d.get("decoded_class"),
                  p_good=d.get("p_good"), p_ge4=d.get("p_ge4"),
                  julia_c_re=d.get("julia_c_re"), julia_c_im=d.get("julia_c_im"),
                  phoenix={k: d.get(k) for k in
                           ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
                            "phoenix_zm1_re", "phoenix_zm1_im") if d.get(k) is not None} or None),
        cut=dict(dist=dist, fw_min=min(a, b), fw_max=max(a, b),
                 d_over_min=dist / min(a, b), d_over_max=dist / max(a, b),
                 fw_ratio=b / a, k=R.REPLAY_K,
                 r_max=R.REPLAY_K * max(a, b), r_min=R.REPLAY_K * min(a, b)),
        palette=BMB._palette_names()[BMB._stable_seed(pid) % len(BMB._palette_names())])


def stage_plan(args) -> int:
    rows, ledger = R.load_population(R.RUN_DIR)
    base = R.replay(rows, "max", admit_frac=0.0, strict=True)     # the same self-check gate
    m0 = R.replay(rows, "min", admit_frac=0.0, strict=False)
    print(f"self-check max(fw) reproduced: precanon_dup {base['precanon_dup']}, "
          f"admitted {base['admitted']}")

    newly = _rank_spread(m0["newly_survived"], args.n_new,
                         lambda r: math.hypot(r["cx"] - float(ledger[r["rec_dup"]]["outcome_cx"]),
                                              r["cy"] - float(ledger[r["rec_dup"]]["outcome_cy"]))
                         / min(r["fw"], float(ledger[r["rec_dup"]]["outcome_fw"])))
    rej = _rank_spread(m0["still_rejected"], args.n_rej,
                       lambda r: math.hypot(r["cx"] - float(ledger[r["rec_dup"]]["outcome_cx"]),
                                            r["cy"] - float(ledger[r["rec_dup"]]["outcome_cy"]))
                       / min(r["fw"], float(ledger[r["rec_dup"]]["outcome_fw"])))

    pairs = ([pair_of(r, ledger, "new") for r in newly]
             + [pair_of(r, ledger, "rej") for r in rej])
    plan = dict(
        n_population=len(m0["newly_survived"]) + len(m0["still_rejected"]),
        n_newly_surviving=len(m0["newly_survived"]),
        n_still_rejected=len(m0["still_rejected"]),
        sampled_new=len(newly), sampled_rej=len(rej),
        sampling=("even RANK quantiles of d / min(fw) over each side of the M0 split; "
                  "deterministic, no RNG"),
        crop=dict(w=bq.CROP_W, h=bq.CROP_H, ss=sc.SITTING_CROP_SS, filter=bq.CROP_FILTER,
                  interior=bq.INTERIOR_MODE, composition=bq.COMPOSITION,
                  vivid=BMB.VIVID_PALETTE),
        by_partition=dict(Counter(p["partition"] for p in pairs)),
        pairs=pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1, default=str), encoding="utf-8")
    print(f"{len(pairs)} pairs ({len(newly)} newly-surviving, {len(rej)} still-rejected) "
          f"from {plan['n_newly_surviving']}+{plan['n_still_rejected']}")
    print(f"partitions: {json.dumps(plan['by_partition'])}")
    print(f"d/min(fw) sampled: new "
          f"{[round(p['cut']['d_over_min'], 3) for p in pairs if p['side'] == 'new']}")
    print(f"                   rej "
          f"{[round(p['cut']['d_over_min'], 3) for p in pairs if p['side'] == 'rej']}")
    print(f"-> {PLAN}")
    return 0


# =========================================================================== #
# render — both sides of every pair, matched
# =========================================================================== #
def render_block(side: dict, partition: str, palette: str) -> dict:
    row = dict(cx=side["cx"], cy=side["cy"], fw=side["fw"], family=partition,
               julia_c_re=side.get("julia_c_re"), julia_c_im=side.get("julia_c_im"),
               _palette=palette)
    row.update(side.get("phoenix") or {})
    rend = bq._render_block(row)
    rend["ss"] = sc.SITTING_CROP_SS       # the sitting's recorded deviation, not the corpus 4
    return rend


def stage_render(args) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    bq._PHOENIX_POOL_CACHE.update(bq._phoenix_points())
    RENDERS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in plan["pairs"]:
        for which in ("cand", "disp"):
            rend = render_block(p[which], p["partition"], p["palette"])
            for tag, palette, src in (("canon", rend["palette"], BMB.PALETTE_SOURCE),
                                      ("vivid", BMB.VIVID_PALETTE, BMB.VIVID_SOURCE)):
                out = RENDERS / f"{p['pair_id']}.{which}.{tag}.jpg"
                if out.exists() and not args.force:
                    continue
                jobs.append((dict(rend, palette=palette), out, src))
    print(f"{len(plan['pairs'])} pairs, {len(jobs)} renders "
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
def _num(x, n=6):
    return "—" if x is None else f"{x:.{n}g}"


def _card(*, title, badge, badge_cls, base, rows) -> str:
    kv = "".join(f'<div class="k">{escape(k)}</div><div class="v">{v}</div>' for k, v in rows)
    return (f'<figure class="card {badge_cls}">'
            f'<div class="imgs"><img loading="lazy" class="canon" src="{base}.canon.jpg" alt="">'
            f'<img loading="lazy" class="vivid" src="{base}.vivid.jpg" alt=""></div>'
            f'<figcaption><div class="hd"><span class="badge">{escape(badge)}</span>'
            f'<span class="tid">{escape(title)}</span></div>'
            f'<div class="kv">{kv}</div></figcaption></figure>')


def stage_sheet(args) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    pairs = plan["pairs"]
    parts = [f"""
<header>
<h1>min(fw) reject autopsy — {plan['sampled_new']} rows the smaller radius would keep,
{plan['sampled_rej']} it still kills</h1>
<p class="q"><b>The question:</b> the replay says <b>{plan['n_newly_surviving']}</b> of the run's
2,531 <code>precanon_dup</code> rows survive a <code>min(fw)</code>-scaled radius (M0 bound;
943 under M1) and <b>{plan['n_still_rejected']}</b> do not. That is a volume, not a verdict.
Each row below is a candidate the run threw away beside the ledger row that displaced it: if the
pair is the same picture, <code>min(fw)</code> is buying churn; if it is not, <code>max(fw)</code>
is deleting places.</p>
<p class="pop"><b>Population and sampling:</b> every <code>precanon_dup</code> row of
<code>harvest_v2_proving_20260803</code>, split by the M0 replay. The sample is
{plan['sampling']} — <b>not</b> the top tier (the 28 human-labelled class-4s are the other sheet),
so most rows here carry no human judgement of any kind and only the displacer carries a machine
decode. Partitions drawn: {escape(json.dumps(plan['by_partition']))}.</p>
<p class="rule"><b>The cut:</b> a candidate is a dup of a q3-cloud row iff plane distance &lt;
<code>K &times; scale(fw_a, fw_b)</code>, <code>K = {R.REPLAY_K}</code> (the run's own K — the
replay this sheet illustrates holds K fixed and moves only the scale), with the julia/phoenix
parameter-identity clause passing first. Production ran <b>max</b> here; the variant is
<b>min</b> — which, at a recalibrated K of {ps.DEDUP_K}, was adopted 2026-08-04. Every caption gives d against BOTH radii, so the row's fate under each
rule is readable without trusting the badge.</p>
<p class="note"><b>Reading the tiles.</b> Top image = the pair's shared canonical palette, bottom =
the fixed <code>blue_orange</code> vivid companion. <b>Both sides of a pair are rendered in the
same palette</b> at {plan['crop']['w']}&times;{plan['crop']['h']} ss{plan['crop']['ss']},
lanczos3, centre, black interior — neither side was ever served, so nothing here is a reused crop
and nothing here has a label. The displacer's machine decode is shown and is not a label.</p>
<div class="toggles">
  <label><input type="checkbox" id="tc" checked> canonical</label>
  <label><input type="checkbox" id="tv" checked> vivid</label>
  <label><input type="checkbox" id="tw"> wide (one pair per row)</label>
</div>
</header>"""]

    srows = []
    for p in pairs:
        c = p["cut"]
        srows.append(
            f"<tr class='{p['side']}'><td class=mono>{escape(p['pair_id'])}</td>"
            f"<td>{'KEEP' if p['side'] == 'new' else 'still cut'}</td>"
            f"<td>{escape(p['partition'])}</td><td class=n>{_num(c['dist'], 4)}</td>"
            f"<td class=n><b>{_num(c['d_over_min'], 3)}</b></td>"
            f"<td class=n>{_num(c['d_over_max'], 3)}</td>"
            f"<td class=n>{_num(c['fw_ratio'], 3)}&times;</td>"
            f"<td class=n>{_num(p['cand']['fw'], 3)}</td>"
            f"<td class=n>{_num(p['disp']['fw'], 3)}</td>"
            f"<td>{p['disp']['decoded_class']}</td></tr>")
    parts.append(
        "<section class=summary><h2>The sample, as arithmetic</h2>"
        f"<p>The <code>min</code> cut fires below <b>{R.REPLAY_K}</b> in the “d / min(fw)” column; "
        f"the <code>max</code> cut fires below {R.REPLAY_K} in “d / max(fw)”. Every row here is "
        "below the max cut — that is why it died.</p>"
        "<table><thead><tr><th>pair</th><th>under min</th><th>partition</th><th>d</th>"
        "<th>d / min(fw)</th><th>d / max(fw)</th><th>fw ratio</th><th>fw discarded</th>"
        "<th>fw displacer</th><th>displacer decode</th></tr></thead><tbody>"
        + "".join(srows) + "</tbody></table></section>")

    for label, side in (("Would newly SURVIVE under min(fw)", "new"),
                        ("Still CUT under min(fw)", "rej")):
        sel = [p for p in pairs if p["side"] == side]
        parts.append(f'<section class="grouphead"><h2>{escape(label)} '
                     f'<span class=dim>&mdash; {len(sel)} sampled</span></h2></section>')
        for p in sel:
            c = p["cut"]
            base = f"{RENDERS_URL}/{p['pair_id']}"
            cards = [_card(
                title=f"{p['pair_id']} · candidate", badge="DISCARDED (precanon_dup)",
                badge_cls="disc", base=f"{base}.cand",
                rows=[("partition", escape(p["partition"])),
                      ("centre", f"{_num(p['cand']['cx'], 10)}, {_num(p['cand']['cy'], 10)}"),
                      ("fw", _num(p["cand"]["fw"], 6)),
                      ("fate", "<b>precanon_dup</b> — cut before any canonical decode"),
                      ("distance", f"d = {_num(c['dist'], 4)} = <b>{_num(c['d_over_min'], 3)}</b> "
                                   f"&times; min(fw) = {_num(c['d_over_max'], 3)} &times; max(fw)"),
                      ("radii", f"min-rule {_num(c['r_min'], 4)}, max-rule {_num(c['r_max'], 4)} "
                                f"(k = {c['k']})"),
                      ("under min", ("<b>KEPT</b>" if side == "new" else "<b>still cut</b>")),
                      ("scale", f"displacer frame is <b>{_num(c['fw_ratio'], 3)}&times;</b> "
                                f"the discarded frame"),
                      ("label", "<b>none</b> — never served"),
                      ("cheap p_good", _num(p["cand"].get("cheap_pgood"), 3)),
                      ("palette", escape(p["palette"])),
                      ("mix", escape(str(p["mix_source"])))]),
                _card(
                title=f"{p['pair_id']} · {p['disp']['id']}", badge="DISPLACER",
                badge_cls="disp", base=f"{base}.disp",
                rows=[("partition", escape(p["partition"])),
                      ("centre", f"{_num(p['disp']['cx'], 10)}, {_num(p['disp']['cy'], 10)}"),
                      ("fw", _num(p["disp"]["fw"], 6)),
                      ("fate", "<b>admitted to the q3 cloud</b>"),
                      ("label", "<b>none — never served in a sitting</b>"),
                      ("machine decode", f"class <b>{p['disp']['decoded_class']}</b> "
                                         f"(p_good {_num(p['disp']['p_good'], 3)}, "
                                         f"p&ge;4 {_num(p['disp']['p_ge4'], 3)}) — not a label"),
                      ("palette", escape(p["palette"]) + " <span class=dim>(shared)</span>")])]
            parts.append(f'<section class="group {side}"><div class="row">'
                         f'{"".join(cards)}</div></section>')

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(PAGE.replace("__BODY__", "\n".join(parts)), encoding="utf-8")
    print(f"-> {SHEET}\n   {len(pairs)} pairs, {2 * len(pairs)} tiles")
    return 0


def stage_verify(args) -> int:
    """Load every `src` THE WAY A BROWSER WOULD — resolved against the page URL with
    `urljoin`, not fetched from the server root. Fetching `root + src` is a different URL
    from the one the browser requests, and it passes on a page whose images are all 404:
    that is exactly how this sheet shipped with 60 broken tiles. Also compares the served
    bytes to disk, so a stale or half-written jpg is a failure rather than a grey box."""
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
<title>min(fw) reject autopsy — kept vs still cut</title>
<style>
:root{--bg:#111316;--fg:#e7e9ec;--dim:#9aa2ad;--line:#2a2f36;--gold:#d8a33a;--red:#c2543f;
      --green:#4f9d69;}
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
.group{margin-top:16px}
.card{margin:0;width:430px;background:#171a1f;border:1px solid var(--line);border-radius:6px;
      overflow:hidden;flex:0 0 auto}
.card.disp{border-color:var(--gold);box-shadow:0 0 0 1px rgba(216,163,58,.25)}
.card.disc{border-color:var(--red)}
.group.new .card.disc{border-color:var(--green)}
.card img{display:block;width:100%;height:auto;background:#000}
.card img+img{border-top:1px solid var(--line)}
body.nc .canon{display:none} body.nv .vivid{display:none}
figcaption{padding:9px 11px 11px;font-size:12px}
.hd{display:flex;gap:8px;align-items:baseline;margin-bottom:7px}
.badge{font-size:10.5px;letter-spacing:.06em;padding:2px 6px;border-radius:3px;
       background:#23272e;color:var(--dim)}
.disp .badge{background:var(--gold);color:#1a1206}
.disc .badge{background:var(--red);color:#fff}
.group.new .disc .badge{background:var(--green);color:#06180c}
.tid{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--dim);
     overflow-wrap:anywhere}
.kv{display:grid;grid-template-columns:98px 1fr;gap:2px 8px}
.kv .k{color:var(--dim)} .kv .v{overflow-wrap:anywhere}
table{border-collapse:collapse;font-size:12px;margin-top:6px}
th,td{border:1px solid var(--line);padding:3px 7px;text-align:left}
th{background:#1b1f25;color:var(--dim);font-weight:600}
tr.new td{background:rgba(79,157,105,.09)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
td.mono,.mono{font-family:ui-monospace,Consolas,monospace;font-size:11px}
.summary p{max-width:1100px;color:#cdd3da}
.grouphead h2{border-top:2px solid var(--line)}
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
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--n-new", type=int, default=N_NEW)
    p.add_argument("--n-rej", type=int, default=N_REJ)
    p.set_defaults(fn=stage_plan)
    r = sub.add_parser("render")
    r.add_argument("--force", action="store_true")
    r.add_argument("--render-timeout", type=float, default=900.0)
    r.set_defaults(fn=stage_render)
    sub.add_parser("sheet").set_defaults(fn=stage_sheet)
    v = sub.add_parser("verify")
    v.add_argument("--port", type=int, default=8020)
    v.add_argument("--url", default=None)
    v.add_argument("--timeout", type=float, default=20.0)
    v.set_defaults(fn=stage_verify)
    a = ap.parse_args()
    cc.set_below_normal_priority()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
