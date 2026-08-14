"""report.py — emission-v1 report + contact sheets.

Writes `scratch/emission_v1_report.md` (path anchored at repo root per the prompt) plus a
release contact sheet and a pool contact sheet grouped by niche, and a machine-readable
`summary.json`. Kept separate from the driver so the report can be rebuilt from the
durable pool log without re-colorizing.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.emission import descriptor as D
from tools.emission import floors as F           # THE stage-2 cut owner

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "scratch" / "emission_v1_report.md"


def _v1_pool_under_the_live_rule(v1_pool: Path, release_n: int):
    """What TODAY'S rule would ship from the v1 run's durable pool: top-N by the head's own
    score, ties on id. Returns `{by_id, selected, n_gated}` or None if the pool is absent.

    WHAT THIS STOPPED BEING, because the difference is the whole point of reading it. Until
    2026-08-09 this reconstructed what v1 ACTUALLY shipped, by calling the v1 selector
    (`selection.greedy_select`, max-marginal-gain over a morph-CLIP coverage kernel) over the
    same pool. That selector was retired from the live path on 2026-08-09 and deleted with the
    rest of the dead machinery; keeping one caller alive to reproduce a historical release is
    what "retired" is supposed to end. So the comparison flipped: the pool is the same durable
    v1 pool, and the rule applied to it is the LIVE one. Read the table as "the v1 pool under
    today's rule", never as "the v1 release".

    `emb` is not consulted — the live rule is a rank, so the coverage kernel the v1 pool's
    embeddings fed has no reader here. The cluster cap is deliberately NOT applied: the v1
    pool's `morph_cluster` ids were minted by a different intake, so capping across them would
    mix two clusterings and produce a set neither rule would ever have chosen."""
    if not v1_pool.exists():
        return None
    rows = [json.loads(l) for l in v1_pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    gated = [r for r in rows if r.get("passed")]
    if not gated:
        return None
    ranked = sorted(gated, key=lambda r: (-(r.get("p_ge3") or 0.0), str(r["id"])))
    return {
        "by_id": {r["id"]: r for r in gated},
        "selected": [r["id"] for r in ranked[:release_n]],
        "n_gated": len(gated),
    }


def _font(sz):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(jpg_rel: str, tw: int, th: int):
    if not jpg_rel:
        return Image.new("RGB", (tw, th), (40, 40, 44))
    p = ROOT / jpg_rel
    if not p.exists():
        return Image.new("RGB", (tw, th), (40, 40, 44))
    with Image.open(p) as im:
        return im.convert("RGB").resize((tw, th), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Contact sheets.
# --------------------------------------------------------------------------- #
def sheet_order(selected: list) -> list:
    """The release sheet's row order: GOOD -> BAD by the head's own score, head-grouped.

    Grouped by head first because the two heads' `p_ge3` are on incommensurable
    train-prior-calibrated scales — one global sort by score would interleave them and read as
    a ranking that nobody computed. Within a head it is a straight score descent, tie-broken on
    id so a re-render of the same release lays the tiles out the same way."""
    return sorted(selected,
                  key=lambda e: ((e["_rec"].get("head") or ""),
                                 -float(e["_rec"].get("p_ge3") or 0.0),
                                 str(e["_rec"]["id"])))


def release_sheet(selected: list, sel_log: list, out_png: Path, cols: int = 4,
                  supply_lines: list | None = None):
    tw, th, pad, lh, head = 300, 169, 8, 42, 52
    ordered = sheet_order(selected)
    n = len(ordered)
    rows = (n + cols - 1) // cols
    W = pad + cols * (tw + pad)
    H = head + rows * (th + lh + pad) + pad + (14 * len(supply_lines or []))
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6), f"emission — release ({n} wallpapers), rank order under the "
                     f"partition/supply/cluster caps", fill=(235, 235, 235), font=_font(15))
    # ASCII markers, not ✓/✗: the fallback fonts `_font` walks have no U+2713/2717 and PIL
    # draws a tofu box, so the one annotation this sheet exists to carry rendered as □.
    d.text((pad, 26), "sorted good->bad by the HEAD's own score (heads grouped — the two "
                      "scales are incommensurable); BELOW = under the RETIRED release floor",
           fill=(150, 150, 165), font=_font(11))
    logi = {l["id"]: l for l in (sel_log or [])}
    for i, e in enumerate(ordered):
        r = e["_rec"]
        cx = pad + (i % cols) * (tw + pad)
        cy = head + (i // cols) * (th + lh + pad)
        sheet.paste(_thumb(r["jpg"], tw, th), (cx, cy))
        L = logi.get(r["id"], {})
        # the retired floor's verdict, off the row's own annotation when the driver wrote one
        # (a re-run over an older pool falls back to today's owner values).
        rf = r.get("release_floor")
        if rf is None:
            rf = (F.WALLPAPER_RELEASE.value if r.get("render_style") == "smooth"
                  else F.MINING_RELEASE.value)
        ok = r.get("would_pass_release_floor")
        if ok is None:
            ok = (r.get("p_ge3") or 0.0) >= rf
        d.text((cx + 2, cy + th + 2),
               f"{i+1}. {r['type']} {r['morph_cluster']}", fill=(220, 220, 160), font=_font(11))
        d.text((cx + 2, cy + th + 15),
               f"{r['palette_flavor']}/{r['render_style']} · {r.get('head', '?')} head "
               f"p3={r['p_ge3']:.3f}", fill=(200, 210, 220), font=_font(10))
        d.text((cx + 2, cy + th + 28),
               (f"rank {L.get('rank_in_partition', '?')} in {L.get('partition', r['type'])}"
                f" · {'ok' if ok else 'BELOW'} retired floor {rf:g}"),
               fill=((150, 200, 150) if ok else (215, 140, 120)), font=_font(10))
    y = head + rows * (th + lh + pad) + 2
    for line in (supply_lines or []):
        d.text((pad, y), line, fill=(150, 150, 165), font=_font(11))
        y += 14
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def pool_sheet(gated: list, out_png: Path, max_per_niche: int = 6):
    tw, th, pad, lh, head = 176, 99, 6, 16, 34
    by_niche = defaultdict(list)
    for r in gated:
        by_niche[tuple(r["cell"])].append(r)
    niches = sorted(by_niche)
    ncol = max((min(len(v), max_per_niche) for v in by_niche.values()), default=1)
    W = 260 + ncol * (tw + pad) + pad
    H = head + len(niches) * (th + lh) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"emission v1 — gated pool by niche ({len(gated)} wallpapers, "
           f"{len(niches)} occupied cells)", fill=(235, 235, 235), font=_font(14))
    for i, niche in enumerate(niches):
        y = head + i * (th + lh)
        t, cl, f, s = niche
        d.text((pad, y + th // 2), f"{t}/{cl}\n{f}/{s}", fill=(180, 200, 230), font=_font(10))
        for j, r in enumerate(sorted(by_niche[niche], key=lambda z: -z["p_ge3"])[:max_per_niche]):
            x = 256 + j * (tw + pad)
            sheet.paste(_thumb(r["jpg"], tw, th), (x, y))
            d.text((x + 2, y + th + 1), f"{r['id']} {r['p_ge3']:.2f}",
                   fill=(200, 200, 210), font=_font(9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


# --------------------------------------------------------------------------- #
# Report body.
# --------------------------------------------------------------------------- #
def _supply_lines(eng) -> list:
    """One line per partition: mined, above-floor, and the emit cap those imply.

    EVERY partition the intake saw gets a line, including the ones that emit nothing. A
    partition that ships zero because its supply was thin and a partition that ships zero
    because nobody looked for it are different failures, and only the mined count separates
    them (§3 — "classic: 24 mined, 0 above floor")."""
    # `eng.mined_supply`, NOT `intake_diag["mined_by_partition"]`: the diag is over the whole
    # union and `passing_supply` is over the population this run serves, so pairing them prints
    # two different denominators on one line.
    mined = getattr(eng, "mined_supply", {}) or {}
    passing = getattr(eng, "passing_supply", {}) or {}
    good = getattr(eng, "good_supply", {}) or {}
    caps = getattr(eng, "emit_caps", {}) or {}
    out = []
    for part in sorted(set(mined) | set(passing) | set(good)):
        n_pass = passing.get(part, 0)
        n_good = good.get(part, 0)
        cap = caps.get(part, 0)
        line = (f"{part}: {mined.get(part, 0)} mined, {n_pass} above floor, "
                f"{n_good} above good floor")
        if not cap:
            line += (" → emits 1 (slot guarantee), then 0 (thin supply)" if n_good
                     else " → emits 0 (thin supply)")
        out.append(line)
    return out


def write_report(eng, selected: list, sel_log: list, rel_paths: list):
    out = eng.out
    gated = eng.pool.gated()
    allrows = eng.pool.rows
    n_att = len(allrows)
    occ = eng.model.occupancy()

    # sheets
    rel_png = out / "release_sheet.png"
    pool_png = out / "pool_sheet.png"
    supply_lines = _supply_lines(eng)
    release_sheet(selected, sel_log, rel_png, supply_lines=supply_lines)
    pool_sheet(gated, pool_png)

    # morph-cluster count among admitted
    clusters = Counter()
    for i, r in eng.by_id.items():
        clusters[r["family"]] += 0
    n_clusters_by_type = defaultdict(set)
    # `.get`, not `[]`: the driver narrows `cluster_tags` to the admitted rows, and this is the
    # LAST statement of a run that has already paid for every render. A report is not the place
    # to enforce that invariant — it is the place least able to afford enforcing it.
    for lid, tag in eng.cluster_tags.items():
        row = eng.by_id.get(lid)
        if row is not None:
            n_clusters_by_type[row["family"]].add(tag)
    total_clusters = len({t for t in eng.cluster_tags.values()})

    # realized vs nominal surplus (per head — smooth via wallpaper head, strange via mining)
    n_pass = len(gated)
    pass_rate = n_pass / n_att if n_att else 0.0
    n_err = sum(1 for r in allrows if r.get("error"))
    wp_gated = [r for r in gated if r.get("head") == "wallpaper"]
    mn_gated = [r for r in gated if r.get("head") == "mining"]
    # "gated rows that ALSO clear their head's release floor" — through the one owner, not
    # two bare literals that happened to agree with the driver on the day this was written.
    wp_also = sum(1 for r in wp_gated if F.WALLPAPER_RELEASE.annotates(r["p_ge3"] or 0))
    mn_also = sum(1 for r in mn_gated if F.MINING_RELEASE.annotates(r["p_ge3"] or 0))
    also_090 = wp_also

    # colorizer choice — flavor + style distribution vs uniform-random baseline
    chosen_flavor = Counter(r["palette_flavor"] for r in allrows)
    chosen_style = Counter(r["render_style"] for r in allrows)
    n_flavors = len(eng.flavors)
    n_styles = len(eng.styles)
    uniform_flavor = n_att / n_flavors if n_flavors else 0
    uniform_style = n_att / n_styles if n_styles else 0

    ledger_labels = [str(l.relative_to(ROOT)) for l in getattr(eng, "ledgers", [eng.ledger])]
    reach = eng.ranker_reach() if hasattr(eng, "ranker_reach") else {}
    short = getattr(eng, "release_short_fill", {})

    L = []
    L.append("# Emission — read-time ranked intake + rank release selection\n")
    L.append("Source ledger(s): " + ", ".join(f"`{x}`" for x in ledger_labels) + ".\n")
    L.append(f"**One cut enforces**: the **{F.JUNK_FLOOR}** junk floor, at the one site where "
             f"the colorize pool is drawn (`ranked_intake`). Everything else is a read-time "
             f"choice. The four stamped per-head floors — pool wallpaper **{eng.floor}** / "
             f"mining **{eng.mining_floor}**, release wallpaper **{eng.release_floor}** / "
             f"mining **{eng.mining_release_floor}** — are **ANNOTATION-ONLY** since "
             f"2026-08-09; they are recorded against every row and gate nothing. Release "
             f"N=**{eng.release_n}** · colorize target **{eng.target_gated}** SCORED rows.\n")
    L.append(f"Cuts, from the one owner (`tools/emission/floors.py`) — {F.summary()}. Each "
             f"annotation-only floor still carries the head version it was set against, "
             f"because an annotation on the wrong probability scale is as unreadable as a gate "
             f"on one.\n")

    acct = eng.target_accounting() if hasattr(eng, "target_accounting") else {}
    if acct:
        L.append("### What `--target-gated` counted, and what the retired floors would have\n")
        L.append(f"**{acct['post_floor']}** scored rows against a target of "
                 f"**{acct['target_gated']}** — smooth **{acct['post_floor_smooth']}** + "
                 f"strange **{acct['post_floor_strange']}**. Of those, "
                 f"**{acct.get('would_pass_release_floor', 0)}** would also have cleared the "
                 f"RETIRED release floors (smooth "
                 f"**{acct.get('would_pass_release_floor_smooth', 0)}** ≥ {eng.release_floor}, "
                 f"strange **{acct.get('would_pass_release_floor_strange', 0)}** ≥ "
                 f"{eng.mining_release_floor}), and "
                 f"**{acct.get('above_retired_pool_floor', 0)}** would have cleared the retired "
                 f"POOL floors.\n")
        L.append(f"**{acct.get('below_retired_release_floor', 0)}** rows "
                 f"(**{acct.get('cut_by_release_floor_strange', 0)}** of them strange) sit "
                 f"BELOW the retired release floors and are in the draw anyway — that count is "
                 f"exactly what the 2026-08-09 restructure turned on, so it is the first number "
                 f"to read when the release looks worse or more varied than it used to.\n")
        L.append(f"**The surplus target is weaker than it was, on purpose.** It counted rows "
                 f"above 0.90/0.50 until the floors were retired and now counts rows that "
                 f"rendered and scored, so hitting 3×N no longer implies 3×N release-grade "
                 f"material by the old bar. It is also a POOLED count across both heads while "
                 f"selection is two disjoint per-head passes with fixed slot budgets "
                 f"(strange_slots = round(N·strange_frac)), so 3N does not imply 3×(strange "
                 f"slots) strange either; the realized split below is where the mix is read.\n")

    budget = dict(getattr(eng, "attempt_budget", {}) or {})
    realized = eng.realized_fills() if hasattr(eng, "realized_fills") else {}
    if budget:
        from tools.emission import attempt_budget as AB
        L.append("### The colorize attempt budget (planned → realized)\n")
        L.append(f"Attempts are budgeted from RELEASE NEED, head first: "
                 f"`attempts_head = {budget.get('attempt_multiplier')} × that head's release "
                 f"slots`, both heads scaled down proportionally if the total budget "
                 f"(**{budget.get('total_budget')}** attempts) cannot cover the pair — never "
                 f"one head starved to keep the other whole. Within a head the attempts split "
                 f"per partition by the same `release_mix` apportionment the release SLOTS "
                 f"use, and fill in rank order from the ranked intake. This replaced the "
                 f"deficit model as the volume rule on 2026-08-09: that model spread over a "
                 f"style axis carrying one smooth style against "
                 f"{max(0, n_styles - 1)} strange ones, so smooth drew ~1/{max(1, n_styles)} "
                 f"of the attempts whatever the release asked for.\n")
        if budget.get("scaled_to_budget"):
            L.append(f"**Scaled down**: the two heads wanted "
                     f"{budget.get('head_want')} = "
                     f"{sum((budget.get('head_want') or {}).values())} attempts against a "
                     f"budget of {budget.get('total_budget')}, so both were truncated "
                     f"proportionally to {budget.get('head_attempts')}.\n")
        for line in AB.fill_lines(budget, realized):
            L.append(f"- {line}")
        L.append("")
        L.append("| head | partition | planned | realized |")
        L.append("|---|---|--:|--:|")
        for h in AB.HEADS:
            planned = (budget.get("planned_by_partition") or {}).get(h, {})
            real = (realized or {}).get(h, {})
            for p in sorted(set(planned) | set(real)):
                if not planned.get(p, 0) and not real.get(p, 0):
                    continue
                L.append(f"| {h} | {p} | {planned.get(p, 0)} | {real.get(p, 0)} |")
        L.append("")
        L.append(f"A short-fill is attributable off these four numbers alone: *wanted > "
                 f"budgeted* is the attempt budget binding, *budgeted > scheduled* is supply "
                 f"at plan time (a partition with fewer floor-passing locations than "
                 f"attempts), *scheduled > realized* is a render error or an attempt-capped "
                 f"cell, and a short release with none of those is the caps downstream. "
                 f"Supply-short at plan time: "
                 f"**{budget.get('supply_short_total', 0)}** attempt(s) "
                 f"{budget.get('supply_short_by_partition') or ''}. "
                 f"`--target-gated` no longer stops the colorize loop; it reports.\n")

    if supply_lines:
        L.append("### Per-partition supply (the thin-supply rule's input)\n")
        L.append(f"`emit <= floor(passing_supply / {F.THIN_SUPPLY_DIVISOR})` per partition. A "
                 f"partition with fewer than {F.THIN_SUPPLY_DIVISOR} floor-passing candidates "
                 f"emits nothing rather than shipping its own least-bad row.\n")
        for line in supply_lines:
            L.append(f"- {line}")
        L.append("")

    L.append("## Intake — morph clusters among admitted locations\n")
    L.append(f"- **{len(eng.rows)}** admitted locations "
             f"(guard_pass ∧ distinct ∧ raw P(≥3) ≥ {F.JUNK_FLOOR}, with the floor-admit "
             f"sources {sorted(D.FLOOR_ADMIT_SOURCES)} bypassing the floor). NO "
             f"decode-version predicate and no stored-`decoded_class` q3 gate since "
             f"2026-08-09 — the raw probability is read, not the frozen verdict.")
    L.append(f"- **{total_clusters}** morph clusters (within-type, cos>{D.NEAR_DUP_THRESHOLD}) "
             f"across **{len(n_clusters_by_type)}** fractal types:")
    for t in sorted(n_clusters_by_type):
        n_loc = sum(1 for r in eng.rows if r["family"] == t)
        L.append(f"  - `{t}`: {n_loc} locations → {len(n_clusters_by_type[t])} clusters")
    L.append("")

    L.append("## Niche occupancy + deficit (before → after)\n")
    L.append(f"- feasible cells: **{occ['feasible_cells']}** "
             f"((type,cluster) × {n_flavors} flavors × {n_styles} styles)")
    L.append(f"- BEFORE (empty pool): 0 populated, deficit = uniform target over all "
             f"{occ['feasible_cells']} feasible cells")
    L.append(f"- AFTER: **{occ['populated_cells']}** distinct cells populated by the "
             f"{n_pass}-wallpaper gated pool; **{occ['capped']}** cells hit the attempt cap "
             f"and left support")
    L.append(f"- **{occ['populated_cells']}** distinct cells did the {n_pass}-surplus "
             f"populate (out of {occ['feasible_cells']} feasible).")
    # per-axis marginal occupancy of the gated pool (which axis values actually filled)
    ax_pop = {ax: Counter() for ax in ("type", "morph_cluster", "palette_flavor", "render_style")}
    for r in gated:
        for ax in ax_pop:
            ax_pop[ax][r[ax]] += 1
    L.append(f"- axis coverage in the gated pool: "
             f"**{len(ax_pop['type'])}** types · **{len(ax_pop['morph_cluster'])}** morph clusters · "
             f"**{len(ax_pop['palette_flavor'])}**/{n_flavors} palette flavors · "
             f"**{len(ax_pop['render_style'])}**/{n_styles} render styles")
    L.append(f"  - render styles present: "
             + ", ".join(f"{s}×{c}" for s, c in ax_pop['render_style'].most_common()))
    L.append("")

    # per-head release-eligibility (the new floors), and inventory banked below them.
    wp_rel = sum(1 for r in wp_gated if (r["p_ge3"] or 0) >= eng.release_floor)
    mn_rel = sum(1 for r in mn_gated if (r["p_ge3"] or 0) >= eng.mining_release_floor)
    n_rel = wp_rel + mn_rel

    L.append("## Pool inventory + the RETIRED per-head floors\n")
    L.append(f"Render styles route to two heads: **smooth → wallpaper head**; **strange → "
             f"mining head**. Every SCORED candidate is pooled and every pooled candidate is in "
             f"the release draw — the per-head pool ({eng.floor} / {eng.mining_floor}) and "
             f"release ({eng.release_floor} / {eng.mining_release_floor}) floors annotate and "
             f"do not admit. The two heads are still never compared in one step; their scales "
             f"are incommensurable.\n")
    L.append(f"- attempts: **{n_att}** · scored (pooled): **{n_pass}** → scoring rate "
             f"**{pass_rate:.1%}** · render errors: {n_err}")
    L.append("")
    L.append("| head | pooled (scored) | would pass retired release floor | below it (shipped "
             "anyway if it ranks) |")
    L.append("|---|--:|--:|--:|")
    L.append(f"| wallpaper (smooth, retired floor {eng.release_floor}) | {len(wp_gated)} | "
             f"{wp_rel} | {len(wp_gated) - wp_rel} |")
    L.append(f"| mining (strange, retired floor {eng.mining_release_floor}) | {len(mn_gated)} | "
             f"{mn_rel} | {len(mn_gated) - mn_rel} |")
    L.append(f"| **total** | **{n_pass}** | **{n_rel}** | **{n_pass - n_rel}** |")
    L.append("")
    L.append(f"**{n_pass - n_rel}/{n_pass}** pooled wallpapers sit below their head's RETIRED "
             f"release floor and compete for a slot anyway. Under the pre-2026-08-09 rule they "
             f"were inventory that could not ship; that population is the whole delta of the "
             f"restructure, and the sheet marks each one with a ✗ so the old cut's value can "
             f"be judged by eye. The colorize targeted **{eng.target_gated}** scored rows.\n")

    if reach:
        L.append("## Ranker reach — did ranked intake concentrate budget on good locations?\n")
        L.append("Admitted locations ordered by pref_loc_v0 score (desc); 'reach' = the deepest "
                 "rank the colorize actually touched. If ranked intake works, colorize fills its "
                 "surplus from the TOP of the ordering and never has to reach deep.\n")
        L.append(f"- {reach['n_locations']} admitted locations; **{reach['n_attempted']}** got a "
                 f"colorize attempt, reaching rank **{reach['deepest_attempted_rank']}** "
                 f"(top {reach['deepest_attempted_pct']:.0%} of the ordering).")
        L.append(f"- **{reach['n_release_locs']}** locations contributed a release-eligible "
                 f"wallpaper, the deepest at rank **{reach['deepest_release_rank']}** "
                 f"(top {reach['deepest_release_pct']:.0%}).")
        L.append(f"- reading: the surplus was filled within the top "
                 f"**{reach['deepest_release_pct']:.0%}** of ranker-ordered locations "
                 f"{'— ranked intake concentrated budget on the good end.' if reach['deepest_release_pct'] < 0.9 else '(reached deep — pool is quality-thin, not a ranking failure).'}\n")

    L.append("## Colorizer choice — deficit-driven palette/style spread\n")
    L.append(f"Chosen palette-flavor distribution over {n_att} colorize attempts vs the "
             f"uniform-random expectation ({uniform_flavor:.1f}/flavor):\n")
    L.append("| palette flavor | chosen | uniform-random |")
    L.append("|---|---:|---:|")
    for f, c in chosen_flavor.most_common():
        L.append(f"| {f} | {c} | {uniform_flavor:.1f} |")
    L.append("")
    L.append(f"Render-style distribution (uniform-random {uniform_style:.1f}/style):\n")
    L.append("| render style | chosen |")
    L.append("|---|---:|")
    for s, c in chosen_style.most_common():
        L.append(f"| {s} | {c} |")
    L.append("")

    fill_note = ""
    if short.get("short_by"):
        fill_note = (f" — **SHORT-FILL {len(selected)}/{eng.release_n}**: the partition slot "
                     f"budgets, the thin-supply caps and the cluster cap left "
                     f"{short['short_by']} slot(s) unfilled out of {short['eligible']} eligible "
                     f"rows; shipping fewer rather than filling past a cap")
    L.append(f"## Release selection — {len(selected)} picks (rank under caps){fill_note}\n")
    split = getattr(eng, "release_split", {})
    if split:
        L.append(f"**Render-mode split (heads never compared in one step).** Smooth slots are "
                 f"filled from the wallpaper head, strange from the mining head, by two DISJOINT "
                 f"within-head rank passes. Target strange frac **{split['strange_frac_target']}** "
                 f"→ slots smooth **{split['smooth_slots']}** / strange **{split['strange_slots']}**. "
                 f"Eligible: smooth **{split['smooth_eligible']}** / strange **{split['strange_eligible']}**. "
                 f"Realized: smooth **{split['smooth_selected']}** / strange **{split['strange_selected']}** "
                 f"(strange frac **{split['strange_frac_realized']:.2f}**). Strange modes: "
                 + ", ".join(f"{s}×{c}" for s, c in sorted(split['strange_modes'].items(),
                                                           key=lambda kv: -kv[1])) + ".\n")
        L.append(f"**The caps.** Per partition, `emit = min(slots, floor(passing_supply / "
                 f"{F.THIN_SUPPLY_DIVISOR}))`, filled by score rank; slots are `release_mix` "
                 f"re-solved over the partitions the pass actually has candidates for. Plus at "
                 f"most **{split.get('cluster_cap')}** picks per morph cluster PER RUN — one "
                 f"counter across both head passes, so a look cannot be taken twice by each. "
                 f"**{split.get('n_cluster_cap_skips', 0)}** candidate(s) were passed over by "
                 f"the cluster cap. Slots: smooth "
                 f"`{ {k: v for k, v in (split.get('partition_slots', {}).get('smooth', {}) or {}).items() if v} }` "
                 f"strange "
                 f"`{ {k: v for k, v in (split.get('partition_slots', {}).get('strange', {}) or {}).items() if v} }`.\n")
        guar = split.get("slot_guarantee") or {}
        if guar:
            owed = {h: v for h, v in (guar.get("owed_by_head") or {}).items() if v}
            L.append(f"**The slot guarantee** (2026-08-10). Every partition with at least one "
                     f"intake candidate above the **{guar.get('good_floor')}** good floor gets "
                     f"one release slot whatever its `release_mix` share; the remainder is "
                     f"apportioned by the mix exactly as above, and the thin-supply cap still "
                     f"governs every slot beyond the guaranteed one. It exists because at N="
                     f"{eng.release_n} the mix seats 6 partitions and structurally zeroes the "
                     f"rest — `phoenix:classic` could not ship a tile off 23 floor-passing rows. "
                     f"Owed this run: `{owed or '{}'}` → "
                     f"**{guar.get('n_guarantee_slots', 0)}** of {len(selected)} pick(s) took a "
                     f"guaranteed slot"
                     + (f"; UNSEATABLE (supply but no scored candidate in either head): "
                        f"`{guar['unseatable']}`" if guar.get("unseatable") else "") + ".\n")
    L.append("Each head's pass ranks on its OWN `p_ge3` (the two scales are incommensurable). No "
             "floor gates the draw; `retired floor` below is the annotation the 0.90/0.50 cuts "
             "became — a ✗ row is one that could not have shipped before 2026-08-09.\n")
    L.append("| # | id | type/cluster | flavor/style | p_ge3 | rank in partition | slot | "
             "retired floor |")
    L.append("|--:|---|---|---|--:|--:|---|---|")
    logi = {l["id"]: l for l in (sel_log or [])}
    for i, e in enumerate(sheet_order(selected), 1):
        r = e["_rec"]
        l = logi.get(r["id"], {})
        rf = r.get("release_floor")
        if rf is None:
            rf = (F.WALLPAPER_RELEASE.value if r.get("render_style") == "smooth"
                  else F.MINING_RELEASE.value)
        ok = r.get("would_pass_release_floor")
        if ok is None:
            ok = (r.get("p_ge3") or 0.0) >= rf
        L.append(f"| {i} | {r['id']} | {r['type']}/{r['morph_cluster']} | "
                 f"{r['palette_flavor']}/{r['render_style']} | {r['p_ge3']:.3f} | "
                 f"{l.get('rank_in_partition', '—')} | {l.get('slot_source', '—')} | "
                 f"{'✓' if ok else '✗'} {rf:g} |")
    L.append("")

    # v1 pool side-by-side: today's rule applied to the v1 run's durable pool.
    v1 = _v1_pool_under_the_live_rule(ROOT / "scratch" / "emission_v1" / "pool_log.jsonl",
                                      eng.release_n)
    if v1 and eng.out.name != "emission_v1":
        L.append("### the v1 POOL under today's rule — side-by-side\n")
        L.append("The durable v1 pool (all gated rows, permissive floor only), re-selected by "
                 "the LIVE rank rule. It is NOT a reconstruction of the v1 release: the v1 "
                 "selector was `greedy_select`, retired 2026-08-09 and deleted with the rest of "
                 "the dead machinery, so what changed is the RULE and what is held fixed is the "
                 "POOL. Annotated with which picks fall below their head's release floor; that "
                 "floor is annotation-only again as of 2026-08-09, so a ✗ row would ship today "
                 "and would not have between 2026-08-06 and then.\n")
        wp_rf, mn_rf = eng.release_floor, eng.mining_release_floor
        n_below = 0
        L.append("| v1-pool pick | type/style | p_ge3 | ≥ release floor? |")
        L.append("|---|---|--:|---|")
        for iid in v1["selected"]:
            r = v1["by_id"][iid]
            style = r["render_style"]
            rf = wp_rf if style == "smooth" else mn_rf
            p = r["p_ge3"] or 0.0
            ok = p >= rf
            n_below += 0 if ok else 1
            verdict = f"✓ ({p:.2f} ≥ {rf})" if ok else f"✗ {p:.2f} BELOW {rf}"
            L.append(f"| {iid} | {r['type']}/{style} | {p:.3f} | {verdict} |")
        L.append("")
        L.append(f"**{n_below}/{len(v1['selected'])}** of these sit below the (now retired) "
                 f"release floors — the sub-floor material the v1 permissive-only bar admitted, "
                 f"and that the current rule lets compete again.\n")
    L.append("## Contact sheets\n")
    L.append(f"- `{rel_png.relative_to(ROOT)}` — the {len(selected)}-wallpaper release")
    L.append(f"- `{pool_png.relative_to(ROOT)}` — the gated pool grouped by niche\n")

    report_path = getattr(eng, "report_path", REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")

    summary = {
        "ledgers": ledger_labels,
        "n_admitted": len(eng.rows), "n_morph_clusters": total_clusters,
        "feasible_cells": occ["feasible_cells"], "populated_cells": occ["populated_cells"],
        "capped_cells": occ["capped"],
        "attempts": n_att, "gated": n_pass, "pass_rate": round(pass_rate, 4),
        "gated_also_090": also_090, "render_errors": n_err,
        # `n_rel` is the count above the RETIRED per-head release floors — it was release
        # eligibility until 2026-08-09 and has not been since, so it is spelled as what it
        # counts. The real eligible population (every scored row) comes from the accounting,
        # which is what `release_candidate_sheet` already joins on. The old key reported 10 of
        # a 30-row eligible pool under its own name, which is a measurement of nothing.
        "above_retired_release_floors": n_rel,
        "release_eligible": acct.get("release_eligible", n_rel),
        "release_n": len(selected), "release_rendered": len(rel_paths),
        # What the release pass ACTUALLY did (worker count, per-engine threads, rows resumed,
        # rows that fell back to serial on a broken pool) — read off the pass, never restated
        # from the flag, so a run that degraded says so.
        "release_pass": dict(getattr(eng, "release_stat", {}) or {}),
        "junk_floor": F.JUNK_FLOOR, "thin_supply_divisor": F.THIN_SUPPLY_DIVISOR,
        "good_floor": F.GOOD_FLOOR,
        "passing_supply": dict(getattr(eng, "passing_supply", {}) or {}),
        "mined_supply": dict(getattr(eng, "mined_supply", {}) or {}),
        "good_supply": dict(getattr(eng, "good_supply", {}) or {}),
        "emit_caps": dict(getattr(eng, "emit_caps", {}) or {}),
        "pool_floor": eng.floor, "mining_pool_floor": eng.mining_floor,
        "release_floor": eng.release_floor, "mining_release_floor": eng.mining_release_floor,
        "loc_ranker": eng.ranker_mode, "ranker_reach": reach, "short_fill": short,
        # planned beside realized, so a short-fill is attributable to supply vs budget without
        # re-deriving either from the pool.
        "attempt_budget": budget, "attempt_realized": realized,
        "target_accounting": acct,
        "floors": {f.name: {"value": f.value, "head": f.head, "stamp": f.stamp,
                            "acts": False} for f in F.ALL_FLOORS},
        "floor_overrides": getattr(eng, "floor_overrides", {}),
        "release_split": getattr(eng, "release_split", {}),
        "palette_ranker": selected[0]["_rec"]["ranker"] if selected else None,
        # THIS SESSION's stage totals (tools/stage_times.py). `release_render` is missing here
        # on purpose when the release pass has not run yet — this report is written after it,
        # so its absence means the renders were skipped, not that they were free.
        "stage_times": (eng.stage_times.totals()
                        if hasattr(eng, "stage_times") else None),
        # ...and WHERE the per-unit rows went. They are no longer beside this file: since
        # 2026-08-13 they land in the durable run-keyed home (`data/emission/run_telemetry/
        # <run>/`, or the bound root on an ephemeral run), so a summary in a wiped scratch tree
        # would otherwise be the only surviving half with no way to name the other one.
        "stage_times_path": (str(eng.stage_times.path)
                             if hasattr(eng, "stage_times") else None),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # `scratch/'summary.json'` — a bare name that never existed in this scope. Every artifact
    # is already on disk by the time this line runs, so the run's outputs were fine and the
    # driver died on the last statement of its last function: a NameError in an f-string is
    # invisible until the branch executes, and this branch only executes on a run that got
    # all the way to a report.
    print(f"[report] {report_path}\n[report] {rel_png}\n[report] {pool_png}\n"
          f"[report] {summary_path}", flush=True)
    return summary
