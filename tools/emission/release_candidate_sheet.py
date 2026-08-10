#!/usr/bin/env python
"""Release-candidate sheet + run readouts for an emission run — fate/source-stratified.

WHAT THIS IS FOR. The driver's own report answers "what did the run do". This answers the
question a release needs a human for: **what did the run put in front of Matt, and what did it
throw away to do it** — admissions AND rejects, in one page, each tile carrying the fate that
produced it and the source it came from. A sheet of admissions alone cannot say whether the
cut was right, for exactly the reason `release_record.py` records the not_selected rows: a
released row alone cannot say what it beat.

NO VERDICT IS COMPUTED. Every number here is a count or a share of what happened; nothing on
this page scores the run. The eye is the instrument.

STRATIFICATION. Rows are grouped by FATE (the decision that ended their run) and, inside a
fate, by SOURCE (the intake ledger). `q4_harvest` rows are called out wherever they appear:
they are the one FLOOR-ADMIT source (`descriptor.FLOOR_ADMIT_SOURCES`), admitted on a human
label with no machine quality cut, so "how do the floor-admitted rows look next to the
q3-gated ones" is a question only this stratification can be asked.

FIDELITY. Each tile is the largest render that exists for that row, named in its caption:
a selected row has its release render, everything else has its 960x540 ss2 pool render (the
frame both heads actually scored, via the 384x224 deploy stretch). Nothing is re-rendered —
this tool reads a finished run dir and writes HTML.

  uv run python tools/emission/release_candidate_sheet.py --run-dir scratch/emission/<run> \
      [--record-root <ephemeral record root>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import descriptor as D          # noqa: E402
from tools.emission import emission_sinks as ESINKS  # noqa: E402
from tools.emission import floors as F              # noqa: E402
import release_mix as RM                            # noqa: E402

SITE = "emission_diversity_v1"

# Fate = the decision that ENDED a candidate's run, most-advanced first. Ordered, because the
# page reads top-down as the funnel and a set would not.
FATES = [
    ("selected", "SELECTED — in the release"),
    ("eligible_not_selected", "eligible, passed over by rank / a slot, supply or cluster cap"),
    ("pooled_below_release_floor", "pooled, below its head's RETIRED release floor "
                                   "(annotation — did not stop it competing)"),
    ("below_pool_floor", "pooled, below its head's RETIRED pool floor (annotation)"),
    ("render_error", "render or scoring error (no head verdict)"),
]


def _jsonl(path: Path) -> list:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Fate assignment — derived from the run's own artifacts, never restated.
# --------------------------------------------------------------------------- #
def assign_fates(pool_rows: list, selected_ids: set, eligible_ids: set,
                 release_floor_of) -> dict:
    """`id -> fate`. Read off the durable pool log plus the release record's decisions, so the
    sheet's strata are the run's actual decisions and not this tool's re-derivation of them.

    The two floor strata are now ANNOTATION strata: since 2026-08-09 nothing is removed by a
    per-head floor, so a row lands in one of them only if it was ALSO not selected and not
    recorded as eligible — i.e. the run's own record is still the authority on the fate, and
    the floor names only say which side of the retired cut the row sat on. A run made while
    those floors enforced reads exactly as it did."""
    out = {}
    for r in pool_rows:
        rid = r["id"]
        if rid in selected_ids:
            out[rid] = "selected"
        elif rid in eligible_ids:
            out[rid] = "eligible_not_selected"
        elif r.get("error") or r.get("p_ge3") is None:
            out[rid] = "render_error"
        elif (r["p_ge3"] or 0.0) >= release_floor_of(r["render_style"]):
            out[rid] = "eligible_not_selected"
        elif r.get("above_pool_floor", r.get("passed")):
            out[rid] = "pooled_below_release_floor"
        else:
            out[rid] = "below_pool_floor"
    return out


def source_of(row: dict, source_tags: dict) -> tuple:
    """(ledger label, source tag). The tag is what makes a row FLOOR-ADMITTED; it is carried
    durably on the intake snapshot so this resolves from artifacts, not from ledgers."""
    ledger = (row.get("provenance") or {}).get("source_ledger") or "?"
    return Path(ledger).parent.name or ledger, source_tags.get(row["location_id"])


# --------------------------------------------------------------------------- #
# Readouts (numbers only — each stated once, with its population).
# --------------------------------------------------------------------------- #
def realized_vs_target(selected_rows: list, target_shares: dict) -> list:
    """Per-partition realized share of the SELECTION against the release-mix target.

    A SHAPE read at this n and nothing more: with a dozen slots the realized share of a 7.6%
    partition is 0 or 1/12, so every cell is quantized far coarser than the target it is being
    compared to. It cannot support a reweight (era-gate rule), and this function deliberately
    returns no residual, no chi-square and no verdict — just the two columns side by side."""
    n = len(selected_rows)
    got = Counter(r["type"] for r in selected_rows)
    rows = []
    for p in sorted(set(target_shares) | set(got)):
        rows.append({"partition": p, "target_share": target_shares.get(p, 0.0),
                     "selected": got.get(p, 0), "realized_share": (got.get(p, 0) / n) if n else 0.0})
    return rows


def colorize_behavior(run_dir: Path, pool_rows: list, summary: dict) -> dict:
    """Wall time and attempt cost of the colorize, and how the ungated-strange share behaved.

    Wall time is measured from the POOL RENDERS' mtimes (first to last), because the driver
    logs no timestamps — so it is the render-to-render span and EXCLUDES intake (embedding +
    clustering + ranker), which for this corpus is the larger half. Stated that way rather
    than as 'run time', which it is not."""
    renders = sorted((run_dir / "renders").glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    span = (renders[-1].stat().st_mtime - renders[0].stat().st_mtime) if len(renders) > 1 else 0.0
    acct = summary.get("target_accounting", {}) or {}
    n_att = len(pool_rows)
    strange = [r for r in pool_rows if r.get("head") == "mining"]
    scored_strange = [r for r in strange if r.get("p_ge3") is not None]
    return {
        "attempts": n_att,
        "render_span_s": round(span, 1),
        "s_per_attempt": round(span / max(1, n_att - 1), 2),
        "target_gated": acct.get("target_gated"),
        "post_floor": acct.get("post_floor"),
        "post_floor_smooth": acct.get("post_floor_smooth"),
        "post_floor_strange": acct.get("post_floor_strange"),
        # the retired floors' counterfactual on this run's population (annotation-only since
        # 2026-08-09) — the number that says how much the restructure actually changed.
        "would_pass_release_floor": acct.get("would_pass_release_floor"),
        "below_retired_release_floor": acct.get("below_retired_release_floor"),
        "release_eligible": acct.get("release_eligible"),
        "n_strange_attempts": len(strange),
        "n_strange_scored": len(scored_strange),
        # share of scored strange that sits BELOW the retired 0.50 — was `ungated_strange`,
        # a key `target_accounting` stopped emitting, so this read 0 for every run it was
        # supposed to describe.
        "below_retired_share_of_scored_strange": (
            round((acct.get("cut_by_release_floor_strange") or 0) / len(scored_strange), 4)
            if scored_strange else None),
        "short_fill": summary.get("short_fill", {}),
    }


def gate_report_accrual(record_root: Path) -> dict:
    """Counts at BOTH gate_report sites. Counts ONLY — the log exists so a future calibration
    pass can read labeled precision off accumulated releases; a precision claim off one
    bounded smoke would be that measurement, made badly, and would be indistinguishable from
    it in the file."""
    rows = _jsonl(Path(record_root) / ESINKS.MINING_GATE_REPORTS / f"{SITE}.jsonl")
    pool_rows = [r for r in rows if r.get("pool_floor") is not None]

    def _wc_pool(r):
        return bool(r["would_cut_pool"]) if "would_cut_pool" in r else not r.get("would_pass_pool", True)
    wc = [r for r in rows if r.get("would_cut")]
    wcp = [r for r in pool_rows if _wc_pool(r)]
    return {
        "n_rows": len(rows),
        "release_site": {"n_would_cut": len(wc),
                         "n_would_cut_selected": sum(1 for r in wc if r.get("selected"))},
        "pool_site": {"n_with_pool_site": len(pool_rows), "n_would_cut_pool": len(wcp),
                      "n_would_cut_pool_pooled": sum(1 for r in wcp if r.get("pooled")),
                      "n_would_cut_pool_selected": sum(1 for r in wcp if r.get("selected"))},
    }


# --------------------------------------------------------------------------- #
# The page.
# --------------------------------------------------------------------------- #
CSS = """
body{background:#101014;color:#dcdce2;font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
     margin:0 auto;padding:24px 28px;max-width:1780px}
h1{font-size:21px;margin:0 0 6px}h2{font-size:16px;margin:34px 0 4px;color:#eaeaf2}
h3{font-size:13px;margin:18px 0 6px;color:#9fb6d8;font-weight:600}
p{margin:6px 0;max-width:110ch;color:#b6b6c2}
code{background:#1c1c24;padding:1px 5px;border-radius:3px;color:#cfe0f5}
b{color:#f0f0f6}
.grid{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 4px}
figure{margin:0;width:320px;background:#17171e;border:1px solid #26263a;border-radius:5px;
       overflow:hidden}
figure img{display:block;width:320px;height:auto;background:#000}
figcaption{padding:5px 7px;font-size:11px;line-height:1.45;color:#a8a8b6}
.tid{color:#e2d18a;font-weight:600}
.q4{border-color:#7a5cc0;box-shadow:0 0 0 1px #7a5cc0 inset}
.q4 .tag{background:#4a3a80;color:#e5dcff}
.tag{display:inline-block;padding:0 5px;border-radius:3px;background:#25253a;color:#b9c8e0;
     font-size:10px;margin-right:4px}
table{border-collapse:collapse;margin:8px 0 4px;font-size:12px}
th,td{border:1px solid #2a2a3c;padding:3px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#1c1c28;color:#cfd6e6}
.scroll{overflow-x:auto}
.note{color:#8d8d9c;font-size:12px}
"""


def tile(row, src_rel, fate, ledger, tag, floor_note) -> str:
    q4 = tag in D.FLOOR_ADMIT_SOURCES
    p = row.get("p_ge3")
    ps = f"{p:.3f}" if p is not None else "—"
    fid = "1280x720 ss2 release render" if "/release/" in src_rel.replace("\\", "/") \
        else "960x540 ss2 pool render (the frame the head scored)"
    tags = f'<span class="tag">{escape(ledger)}</span>'
    if tag:
        tags += f'<span class="tag">{escape(tag)}{" · FLOOR-ADMIT" if q4 else ""}</span>'
    return (
        f'<figure class="{"q4" if q4 else ""}">'
        f'<img loading="lazy" src="/{escape(src_rel)}" alt="{escape(row["id"])}">'
        f'<figcaption><span class="tid">{escape(row["id"])}</span> · '
        f'{escape(str(row.get("type")))} / {escape(str(row.get("morph_cluster")))}<br>'
        f'{escape(str(row.get("palette_flavor")))} / <b>{escape(str(row.get("render_style")))}</b>'
        f' · {escape(str(row.get("palette")))}<br>'
        f'{escape(str(row.get("head")))} head p_ge3=<b>{ps}</b> {escape(floor_note)}<br>'
        f'{tags}<br><span class="note">{fid}</span></figcaption></figure>')


def build(run_dir: Path, record_root: Path, out_html: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    pool_rows = _jsonl(run_dir / "pool_log.jsonl")
    if not pool_rows:
        raise SystemExit(f"no pool_log.jsonl under {run_dir} — nothing to stratify")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) \
        if (run_dir / "summary.json").exists() else {}
    intake = json.loads((run_dir / "intake.json").read_text(encoding="utf-8"))
    source_tags = intake.get("source_tags", {}) or {}

    decisions = _jsonl(Path(record_root) / ESINKS.RELEASE_RECORDS / f"{SITE}.jsonl")
    runs = _jsonl(Path(record_root) / ESINKS.RELEASE_RECORDS / f"{SITE}__runs.jsonl")
    rel_dec = [d for d in decisions if d["stage"] == "release"]
    by_join = {}
    for r in pool_rows:
        by_join["|".join(str(x) for x in (r["location_id"], r["render_style"], r["palette"]))] = r["id"]
    selected_ids = {by_join[d["join_key"]] for d in rel_dec
                    if d["decision"] == "selected" and d["join_key"] in by_join}
    eligible_ids = {by_join[d["join_key"]] for d in rel_dec if d["join_key"] in by_join}

    def release_floor_of(style):
        return F.WALLPAPER_RELEASE.value if style == "smooth" else F.MINING_RELEASE.value

    fates = assign_fates(pool_rows, selected_ids, eligible_ids, release_floor_of)
    by_id = {r["id"]: r for r in pool_rows}
    selected_rows = [by_id[i] for i in selected_ids]

    # --- readouts ------------------------------------------------------- #
    # The target is solved over the partitions the INTAKE observed, not over the ones the
    # colorize happened to reach. Reading it over the reached set would renormalize a
    # partition's demand away precisely because the run never served it — turning "phoenix got
    # no attempts" into "phoenix was not asked for", which is the failure `unrealized_shares`
    # exists to prevent one layer up.
    intake_parts = sorted((runs[-1]["counts"].get("intake_by_partition") if runs else None)
                          or {r["type"] for r in pool_rows})
    target_shares = RM.shares(intake_parts)
    readout = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "record_root": str(record_root),
        "n_pool_rows": len(pool_rows),
        "fates": dict(Counter(fates.values())),
        "realized_vs_target": realized_vs_target(selected_rows, target_shares),
        "colorize": colorize_behavior(run_dir, pool_rows, summary),
        "gate_report": gate_report_accrual(record_root),
        "run_record": runs[-1]["counts"] if runs else {},
    }

    # --- page ----------------------------------------------------------- #
    P = []
    P.append(f"<!doctype html><meta charset=utf-8><title>Emission release candidates — "
             f"{escape(run_dir.name)}</title><style>{CSS}</style>")
    P.append(f"<h1>Emission release candidates — <code>{escape(run_dir.name)}</code></h1>")
    P.append(f"<p><b>What this is:</b> every candidate the run colorized, grouped by the "
             f"decision that ended its run and then by the ledger it came from. "
             f"<b>{readout['fates'].get('selected', 0)}</b> were released; "
             f"<b>{len(pool_rows) - readout['fates'].get('selected', 0)}</b> were not. "
             f"Nothing here is scored — the counts are what happened, the eye is the "
             f"instrument.</p>")
    P.append(f"<p><b>Floor-admitted rows are outlined and tagged.</b> "
             f"<code>{escape(', '.join(sorted(D.FLOOR_ADMIT_SOURCES)))}</code> rows are admitted "
             f"on a human label with <i>no machine quality cut</i> at intake; they still face "
             f"both stage-2 head floors like anything else. They are marked wherever they "
             f"appear so the two supplies can be compared by eye.</p>")
    P.append(f"<p class=note><b>Cuts:</b> {escape(F.summary())}. Since 2026-08-09 the four "
             f"per-head floors ANNOTATE and the only enforcing cut is the junk floor, applied "
             f"one stage earlier where the colorize pool is drawn — so a tile marked "
             f"<b>✗ retired floor</b> below competed for a slot and may well have won one. "
             f"Fates are read from the run's own release RECORD, not re-derived from today's "
             f"floors, so a run made while those floors enforced still reads as what it did.</p>")

    # --- per-partition supply, the thin-supply rule's input ------------------ #
    mined = summary.get("mined_supply", {}) or {}
    passing = summary.get("passing_supply", {}) or {}
    good = summary.get("good_supply", {}) or {}
    caps = summary.get("emit_caps", {}) or {}
    guar = ((summary.get("release_split", {}) or {}).get("slot_guarantee", {}) or {})
    owed = {p: h for h, ps in (guar.get("owed_by_head") or {}).items() for p in ps}
    if mined or passing:
        div = summary.get("thin_supply_divisor", "?")
        P.append("<h2>Per-partition supply</h2>")
        P.append(f"<p><code>emit &le; floor(passing_supply / {div})</code>. A partition with "
                 f"too few floor-passing candidates emits nothing rather than shipping its own "
                 f"least-bad row — one line each, including the ones that emit zero, because a "
                 f"partition that vanishes from a readout when its supply dies is the failure "
                 f"this list exists to prevent. <b>Since 2026-08-10 the slot guarantee overrides "
                 f"that zero for ONE slot</b>: a partition with any candidate above the "
                 f"{guar.get('good_floor', summary.get('good_floor', '?'))} good floor is seated "
                 f"whatever its <code>release_mix</code> share, and the cap governs beyond it.</p>")
        P.append('<div class="scroll"><table><tr><th>partition</th><th>mined</th>'
                 '<th>above junk floor</th><th>above good floor</th><th>emit cap</th>'
                 '<th>guaranteed slot</th></tr>')
        for part in sorted(set(mined) | set(passing) | set(good)):
            n_pass = passing.get(part, 0)
            cap = caps.get(part, 0)
            note = "" if cap else " <span class=note>(thin supply → 0)</span>"
            g = f"yes ({escape(owed[part])} head)" if part in owed else "—"
            P.append(f"<tr><td>{escape(part)}</td><td>{mined.get(part, 0)}</td>"
                     f"<td>{n_pass}</td><td>{good.get(part, 0)}</td><td>{cap}{note}</td>"
                     f"<td>{g}</td></tr>")
        P.append("</table></div>")

    # readout tables
    P.append("<h2>Realized vs target — per-partition, over the selection</h2>")
    P.append(f"<p>Target = <code>release_mix.RATIO</code> re-solved over the partitions this "
             f"pool observes. <b>A SHAPE read at n={len(selected_rows)} and nothing more:</b> "
             f"a 7.6% target against a dozen slots quantizes to 0 or 1, so this cannot and "
             f"must not be read as a reweight signal.</p>")
    P.append('<div class="scroll"><table><tr><th>partition</th><th>target share</th>'
             '<th>selected</th><th>realized share</th></tr>')
    for r in readout["realized_vs_target"]:
        P.append(f"<tr><td>{escape(r['partition'])}</td><td>{r['target_share']:.2%}</td>"
                 f"<td>{r['selected']}</td><td>{r['realized_share']:.2%}</td></tr>")
    P.append("</table></div>")

    P.append("<h2>Fate census</h2><div class=scroll><table><tr><th>fate</th><th>n</th></tr>")
    for key, label in FATES:
        P.append(f"<tr><td>{escape(label)}</td><td>{readout['fates'].get(key, 0)}</td></tr>")
    P.append(f"<tr><td><b>total colorized</b></td><td><b>{len(pool_rows)}</b></td></tr></table></div>")

    # tiles
    for key, label in FATES:
        rows = [r for r in pool_rows if fates[r["id"]] == key]
        if not rows:
            continue
        P.append(f"<h2>{escape(label)} — {len(rows)}</h2>")
        by_src = defaultdict(list)
        for r in rows:
            by_src[source_of(r, source_tags)].append(r)
        for (ledger, tag) in sorted(by_src, key=lambda k: (-len(by_src[k]), str(k))):
            grp = sorted(by_src[(ledger, tag)], key=lambda r: -(r.get("p_ge3") or -1))
            P.append(f"<h3>{escape(ledger)}"
                     + (f" · <code>{escape(tag)}</code>" if tag else "")
                     + f" — {len(grp)}</h3><div class=grid>")
            for r in grp:
                png = run_dir / "release" / f"{r['id']}.png"
                src = png if (key == "selected" and png.exists()) else (ROOT / (r.get("jpg") or ""))
                if not src.exists():
                    continue
                rf = release_floor_of(r["render_style"])
                ok = r.get("would_pass_release_floor")
                if ok is None:
                    ok = (r.get("p_ge3") or 0.0) >= rf
                note = (f"· {'✓' if ok else '✗'} retired release floor {rf:g} "
                        f"(retired pool floor {r.get('floor')})")
                P.append(tile(r, src.relative_to(ROOT).as_posix(), key, ledger, tag, note))
            P.append("</div>")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(P), encoding="utf-8")
    (run_dir / "readout.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")
    return readout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--record-root", default=None,
                    help="record-store root the run wrote to (default: the run's ephemeral "
                         "root, derived from the run dir name)")
    ap.add_argument("--out", default=None, help="sheet .html (default <run-dir>/release_candidate_sheet.html)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    record_root = Path(args.record_root).resolve() if args.record_root else \
        ESINKS.smoke_scratch_root(ROOT, run_dir.name).resolve()
    out = Path(args.out).resolve() if args.out else run_dir / "release_candidate_sheet.html"
    readout = build(run_dir, record_root, out)
    print(json.dumps(readout, indent=2))
    print(f"\n[sheet] {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
