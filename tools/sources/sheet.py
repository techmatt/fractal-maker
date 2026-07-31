#!/usr/bin/env python
"""Build a source sheet: one HTML page, **one row per atom showing all three ladder rungs
(1x / 4x / 16x) side by side**, plus the per-sheet aggregate descriptor mix in the header.

The header/tile split is the point. Matt judges the sheet **as a whole**, so
"this sheet is 90% satellites" and the depth histogram belong at the top — they are
exactly the diagnostic he needs. **Per-tile there is no metadata at all**: no period,
no size, no source, no score, nothing in a tooltip and nothing in the filename beyond
an opaque content hash. A per-tile label would bias the very judgement being collected.

Both references (`ref_eye`, `ref_mb19`) ride on every sheet at identical framing —
including `ref_eye`, which is a good *view* and not a nucleus at all, and is kept
precisely because it is known-good material that is not minibrot-anchored.

Sheets are written beside the tiles in the out-of-tree bulk root, so `../tiles/x.png`
resolves and the page opens by double-click with no path fixing.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "descent"))

import source_store as ss     # noqa: E402
import triage_store as ts     # noqa: E402

CSS = """
:root{--bg:#0d0d0d;--panel:#181818;--fg:#ddd;--accent:#6cf;--muted:#8a8a8a;
      --good:#5c9;--warn:#fd6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid #2a2a2a;background:var(--panel)}
h1{margin:0 0 6px;font-size:16px;color:var(--accent)}
h2{margin:16px 0 6px;font-size:11px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.05em}
.desc{display:flex;gap:22px;flex-wrap:wrap;margin-top:8px}
.desc div{font-size:12px}
.desc b{color:#fff;font-variant-numeric:tabular-nums}
.hist{display:flex;gap:2px;align-items:flex-end;height:46px;margin-top:6px}
.hist span{width:26px;background:#2b4a63;position:relative;border-radius:2px 2px 0 0}
.hist span i{position:absolute;bottom:-15px;left:0;right:0;text-align:center;
             font-style:normal;font-size:9px;color:var(--muted)}
.hist span u{position:absolute;top:-13px;left:0;right:0;text-align:center;
             text-decoration:none;font-size:9px;color:#bbb}
.note{color:var(--muted);font-size:11px;margin-top:20px;max-width:70em}
.refs{display:flex;gap:14px;flex-wrap:wrap;padding:10px 16px;background:#141414;
      border-bottom:1px solid #2a2a2a}
.refgroup{border:1px solid #262626;border-radius:5px;padding:6px;background:var(--panel)}
.refgroup .nm{font-size:11px;color:var(--muted);margin-bottom:4px}
.refrow{display:flex;gap:5px}
.refcell{text-align:center}
.refcell img{width:300px;aspect-ratio:16/9;display:block;background:#000;border-radius:3px}
.refcell span{font-size:9px;color:var(--muted)}
.refcell.def span{color:var(--accent)}
#wall{padding:12px 16px}
.colhead{display:flex;gap:6px;position:sticky;top:0;background:var(--bg);
         padding:6px 16px 4px;border-bottom:1px solid #222;z-index:4}
.colhead div{flex:1 1 0;max-width:480px;font-size:11px;color:var(--muted);
             text-transform:uppercase;letter-spacing:.06em}
.colhead div.def{color:var(--accent)}
.row{display:flex;gap:6px;margin-bottom:6px}
.cell{flex:1 1 0;max-width:480px;position:relative;border:1px solid #222;
      border-radius:3px;overflow:hidden;background:#000;aspect-ratio:16/9}
.cell img{width:100%;height:100%;display:block;object-fit:cover;background:#000}
.cell .sc{position:absolute;bottom:2px;right:4px;font-size:10px;color:#ccc;
          background:rgba(0,0,0,.5);border-radius:2px;padding:0 4px}
.cell.def{border-color:#3a4a58}
.cell.na{display:flex;align-items:center;justify-content:center;border-style:dashed;border-color:#333}
.nab{font-size:10px;color:#666;text-align:center;padding:0 8px}
a{color:var(--accent)}
"""

# No JavaScript: every rung of the ladder is on screen at once, so there is nothing to
# click. (The earlier click-to-cycle sheet made comparing 1x against 16x a memory test.)
def _hist_html(hist: list[dict]) -> str:
    top = max([h["n"] for h in hist] + [1])
    out = []
    for h in hist:
        px = int(3 + 40 * h["n"] / top)
        lab = f'{h["lo"]}' if h["hi"] < 90 else f'{h["lo"]}+'
        out.append(f'<span style="height:{px}px"><u>{h["n"] or ""}</u><i>{lab}</i></span>')
    return '<div class="hist">' + "".join(out) + "</div>"


def _desc_html(d: dict, extra: list[tuple[str, str]] | None = None) -> str:
    if not d.get("n"):
        return '<div class="desc"><div>empty sheet</div></div>'
    cells = [
        ("atoms", f'{d["n"]}'),
        ("period", f'{d["period_min"]} – {d["period_max"]}  (med {d["period_med"]})'),
        ("log10|A|",
         f'{d["log10_abs_A_min"]} – {d["log10_abs_A_max"]}  (med {d["log10_abs_A_med"]})'),
        ("atom size", f'{d["atom_size_max"]} – {d["atom_size_min"]}'),
        ("on the real axis", f'{d["on_real_axis_n"]}'),
        ("below the roster's feasibility floor",
         f'{d["n_below_feasibility_floor"]}  (kept anyway — recorded, not cut)'),
        ("primitive / satellite",
         "NOT AVAILABLE — no verified classifier (see report)"),
    ]
    cells += (extra or [])
    inner = "".join(f'<div>{html.escape(k)}<br><b>{html.escape(str(v))}</b></div>'
                    for k, v in cells)
    return f'<div class="desc">{inner}</div>'


def _refs_html() -> str:
    refs = ts.load_references()
    if not refs:
        return ""
    out = []
    for r in refs:
        cells = "".join(
            f'<div class="refcell{" def" if s == ss.DEFAULT_SCALE else ""}">'
            f'<img loading="lazy" src="../tiles/{r["id"]}__x{s}.png">'
            f'<span>{s}&times;{" (sheet framing)" if s == ss.DEFAULT_SCALE else ""}</span></div>'
            for s in ss.SCALES)
        out.append(f'<div class="refgroup"><div class="nm">reference &middot; '
                   f'{html.escape(r["label"])}'
                   f'{" &mdash; a view, not a nucleus" if not r.get("nucleus") else ""}'
                   f'</div><div class="refrow">{cells}</div></div>')
    return '<div class="refs">' + "".join(out) + "</div>"


def _cell(a: dict, s: int) -> str:
    """One ladder rung. A rung whose tile does not exist renders as an empty cell rather
    than a broken image: the deepest atoms clear the f64 wall at 4x and 16x but not at
    1x, where the frame is four times narrower and the pixel spacing four times finer.
    That is the same empirical render boundary the sheets keep everywhere else — the atom
    still earns its row on the rungs that did render."""
    klass = "cell def" if s == ss.DEFAULT_SCALE else "cell"
    src = f'../tiles/{a["id"]}__x{s}.png'
    if not ss.tile_path(a["id"], s).exists():
        return (f'<div class="{klass} na"><div class="nab">{s}&times; unavailable &mdash; '
                f'below the f64 render floor at this frame</div></div>')
    return (f'<div class="{klass}"><a href="{src}" target="_blank">'
            f'<img loading="lazy" src="{src}" alt=""></a>'
            f'<div class="sc">{s}&times;</div></div>')


def build_sheet(source_id: str, title: str, blurb: str, atoms: list[dict],
                desc: dict, *, extra_desc=None, notes: str = "",
                shuffled: bool = False) -> Path:
    """Write one sheet: **one row per atom, all three ladder rungs side by side.**

    Every rung visible at once rather than cycled on click — comparing 1x against 16x
    should not be a memory test, and the whole point of the ladder is the comparison."""
    ss.ensure_dirs()
    head = "".join(
        f'<div class="{"def" if s == ss.DEFAULT_SCALE else ""}">{s}&times;'
        f'{" &mdash; the sheet frame" if s == ss.DEFAULT_SCALE else ""}</div>'
        for s in ss.SCALES)
    rows = "".join('<div class="row">' + "".join(_cell(a, s) for s in ss.SCALES)
                   + '</div>' for a in atoms)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header>
  <h1>{html.escape(title)}</h1>
  <div style="font-size:12px;color:#bbb;max-width:80em">{blurb}</div>
  {_desc_html(desc, extra_desc)}
  <h2>depth histogram &mdash; log10|A| bucket</h2>
  {_hist_html(desc.get("depth_histogram", []))}
</header>
{"" if shuffled else _refs_html()}
<div class="colhead">{head}</div>
<div id="wall">{rows}</div>
<div class="note" style="padding:0 16px 24px">
  One row per atom: <b>1&times; / 4&times; / 16&times; the atom's own size</b>, vivid
  <code>blue_orange</code>, navigation fidelity &mdash; identical across all sheets.
  Click any image for it full-size. No per-tile metadata by design.
  {notes}
</div>
</body></html>
"""
    p = ss.sheet_path(source_id)
    p.write_text(doc, encoding="utf-8")
    return p


def build_index(entries: list[dict], overlap: dict | None = None,
                skipped: list[dict] | None = None) -> Path:
    """The index page: every sheet with its count and descriptor mix, plus the
    cross-source overlap matrix."""
    ss.ensure_dirs()
    rows = []
    for e in entries:
        d = e["desc"]
        rows.append(
            f'<tr><td><a href="{e["source_id"]}.html">{html.escape(e["title"])}</a></td>'
            f'<td class="n">{d.get("n", 0)}</td>'
            f'<td class="n">{d.get("satellite_frac", 0):.0%}</td>'
            f'<td class="n">{d.get("period_min","-")}&ndash;{d.get("period_max","-")}</td>'
            f'<td class="n">{d.get("log10_abs_A_min","-")}&ndash;{d.get("log10_abs_A_max","-")}</td>'
            f'<td class="n">{d.get("embedding_depth_med","-")}</td>'
            f'<td class="n">{d.get("n_below_feasibility_floor",0)}</td>'
            f'<td>{html.escape(e.get("one_line",""))}</td></tr>')
    ov = ""
    if overlap:
        ids = overlap["sources"]
        head = "".join(f"<th>{html.escape(i)}</th>" for i in ids)
        body = ""
        for i in ids:
            cells = "".join(
                f'<td class="n{" diag" if i == j else ""}">{overlap["matrix"][i][j]}</td>'
                for j in ids)
            body += f"<tr><th>{html.escape(i)}</th>{cells}</tr>"
        ov = (f'<h2>overlap matrix &mdash; shared atoms (sector-canonical nucleus dedup)</h2>'
              f'<table><tr><th></th>{head}</tr>{body}</table>'
              f'<div class="note">Diagonal = the source\'s own atom count. '
              f'Off-diagonal = atoms found by both. Two sources finding the same atom '
              f'is a result, not a bug.</div>')
    sk = ""
    if skipped:
        sk = ('<h2>not built</h2><ul>' + "".join(
            f'<li><b>{html.escape(s["source_id"])}</b> &mdash; {html.escape(s["reason"])}</li>'
            for s in skipped) + '</ul>')
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Minibrot source sheets</title>
<style>{CSS}
table{{border-collapse:collapse;margin-top:8px;font-size:12px}}
th,td{{border:1px solid #2a2a2a;padding:4px 9px;text-align:left}}
th{{color:var(--muted);font-weight:600}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
td.diag{{color:var(--accent)}}
ul{{font-size:12px}}
</style></head><body>
<header><h1>Minibrot source sheets &mdash; one sheet per generation algorithm</h1>
<div style="font-size:12px;color:#bbb;max-width:80em">
Degree 2 only. Every sheet uses identical framing (4&times; the atom's own size, vivid
<code>blue_orange</code>, navigation fidelity) and carries the same two known-good
references, so the sheets are comparable to each other. Within a sheet the sample
<b>spans the available depth range</b> rather than taking the natural head of the
distribution &mdash; if sources differ in depth mix and depth drives quality, an
unsampled race would measure depth instead of the source. Nothing is fitted, ranked,
or declared a winner.
</div></header>
<div style="padding:12px 16px">
<table>
<tr><th>sheet</th><th>atoms</th><th>satellite</th><th>period</th><th>log10|A|</th>
    <th>embed</th><th>sub&#8209;floor</th><th></th></tr>
{"".join(rows)}
</table>
{ov}
{sk}
</div></body></html>
"""
    p = ss.index_path()
    p.write_text(doc, encoding="utf-8")
    return p
