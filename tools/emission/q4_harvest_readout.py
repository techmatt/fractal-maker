#!/usr/bin/env python
r"""q4_harvest_readout.py — q4-source release-candidate sheet + reject/pool autopsy.

The emission driver routes each render to ONE head (smooth→wallpaper v3 0.90; strange→mining
v1 0.50) and reports only that head's score. For the q4 minibrot harvest the mining head is
UNCALIBRATED on strange renders, so this readout surfaces BOTH head scores for every release
candidate and every pooled wallpaper — Matt judges strange by eye. It does NOT re-run the
pipeline; it reads the durable driver artifacts + the producer's decode ledger and:

  * per-stage counts:  candidates → rendered/guard-passed → floor-admitted → distinct clusters
                       → colorized → gated → release candidates
  * release-candidate sheet:  each pick with full descriptor (type/cluster/flavor/style/palette),
                              framing (cx/cy/fw), q4 G + minibrot, and BOTH head scores.
  * reject/pool autopsy:  pooled-but-not-selected wallpapers (both heads) + the producer-stage
                          rejects (guard-fail / below-floor), so nothing is silently dropped.

Reads   out/emission/q4_harvest/{pool_log.jsonl, intake.json, summary.json, release/*.png}
        data/emission/q4_harvest/{outcome_ledger.jsonl, rescored.jsonl, stats.json}
Writes  out/emission/q4_harvest/q4_release_sheet.png
        out/emission/q4_harvest/q4_autopsy_sheet.png
        out/emission/q4_harvest/q4_readout.md
        out/emission/q4_harvest/q4_readout.json

  uv run python tools/emission/q4_harvest_readout.py
  uv run python tools/emission/q4_harvest_readout.py --out out/emission/q4_harvest_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "wallpaper", ROOT / "tools" / "mining"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LEDGER_DIR = ROOT / "data" / "emission" / "q4_harvest"
STRICT_WP, STRICT_MN = 0.90, 0.50   # production release floors (strict; here only annotated)
STRANGE_STYLE_WEIGHT = 0.5


def _wallpaper_style(style: str) -> bool:
    return style == "smooth"


def _diversity_select(out_dir: Path, gated: list, n: int) -> list:
    """The gated q4 pool, diversity-selected via the driver's OWN head-split greedy kernel
    (release floors = pool floors, so ALL gated wallpapers are eligible). Two disjoint
    within-head passes — heads never compared in one step — exactly like
    build_emission_diversity_v1.select_release, just without the extra strict release floor."""
    from tools.emission import selection as SEL
    from tools.emission import descriptor as D
    embs = D.load_embs(out_dir / "morph_embs.npz")

    def entries(rows):
        out = []
        for r in rows:
            emb = embs.get(r["location_id"])
            out.append({"id": r["id"], "type": r["type"], "cluster": r["morph_cluster"],
                        "flavor": r["palette_flavor"], "style": r["render_style"],
                        "score": r.get("p_ge3"), "emb": emb.tolist() if emb is not None else None,
                        "_rec": r})
        return out

    smooth = [r for r in gated if _wallpaper_style(r["render_style"])]
    strange = [r for r in gated if not _wallpaper_style(r["render_style"])]
    strange_slots = int(round(n * 0.5))
    smooth_slots = n - strange_slots
    sm_sel, _ = SEL.greedy_select(entries(smooth), min(smooth_slots, len(smooth)))
    st_sel, _ = SEL.greedy_select(entries(strange), min(strange_slots, len(strange)),
                                  style_weight=STRANGE_STYLE_WEIGHT)
    ordered = [e["_rec"] for e in sm_sel] + [e["_rec"] for e in st_sel]
    # any gated rows greedy didn't reach (slots exhausted) still belong in the view — append
    # them after the diversity-ordered picks so the full gated pool is representable.
    picked = {r["id"] for r in ordered}
    ordered += sorted((r for r in gated if r["id"] not in picked),
                      key=lambda r: -(r.get("p_ge3") or 0.0))
    return ordered[:n]


def _clears_strict(r) -> bool:
    p = r.get("p_ge3") or 0.0
    return p >= (STRICT_WP if _wallpaper_style(r["render_style"]) else STRICT_MN)


# --------------------------------------------------------------------------- #
# Dual-head scorer — score one tile with BOTH heads (the readout's whole point).
# --------------------------------------------------------------------------- #
class DualHeads:
    def __init__(self):
        import torch
        from tools.wallpaper import emit_v1
        from tools.mining.mining_gate import MiningScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wp_score, _cfg = emit_v1.load_v2_scorer(device)
        self.wp_gate = emit_v1.GATE_THRESHOLD              # 0.90
        self.mining = MiningScorer()
        self.mining_gate = self.mining.threshold           # 0.50

    def score_many(self, paths: list) -> dict:
        """{path_str: {"wp": {p_ge2,p_ge3,ssum}, "mn": {...}}} — batched over both heads."""
        paths = [str(p) for p in paths]
        out = {p: {} for p in paths}
        if not paths:
            return out
        _c, marg, ssum = self.wp_score(paths)
        for i, p in enumerate(paths):
            out[p]["wp"] = {"p_ge2": float(marg[i, 0]), "p_ge3": float(marg[i, 1]),
                            "ssum": float(ssum[i])}
        ms = self.mining.score_paths(paths)
        for i, p in enumerate(paths):
            out[p]["mn"] = {"p_ge2": float(ms[i].p_ge2), "p_ge3": float(ms[i].p_ge3),
                            "ssum": float(ms[i].score)}
        return out


# --------------------------------------------------------------------------- #
def _font(sz):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(jpg_rel, tw, th):
    if not jpg_rel:
        return Image.new("RGB", (tw, th), (40, 40, 44))
    p = ROOT / jpg_rel if not Path(jpg_rel).is_absolute() else Path(jpg_rel)
    if not p.exists():
        return Image.new("RGB", (tw, th), (40, 40, 44))
    with Image.open(p) as im:
        return im.convert("RGB").resize((tw, th), Image.LANCZOS)


def _fw_short(s):
    try:
        return f"{float(s):.3e}"
    except Exception:
        return str(s)


def _heads_line(routed_head, wp, mn, wp_gate, mn_gate):
    """A both-heads label: routed head starred; each with a gate tick."""
    def tick(v, g):
        return "✓" if (v is not None and v >= g) else "·"
    wp3 = wp["p_ge3"] if wp else None
    mn3 = mn["p_ge3"] if mn else None
    star_wp = "*" if routed_head == "wallpaper" else " "
    star_mn = "*" if routed_head == "mining" else " "
    wp_s = f"{wp3:.2f}{tick(wp3, wp_gate)}" if wp3 is not None else "  — "
    mn_s = f"{mn3:.2f}{tick(mn3, mn_gate)}" if mn3 is not None else "  — "
    return f"wp{star_wp}{wp_s}  mn{star_mn}{mn_s}"


# --------------------------------------------------------------------------- #
def build(out_dir: Path):
    pool = [json.loads(l) for l in (out_dir / "pool_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    intake = json.loads((out_dir / "intake.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) \
        if (out_dir / "summary.json").exists() else {}
    ledger = {r["id"]: r for r in
              (json.loads(l) for l in (LEDGER_DIR / "outcome_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    producer = json.loads((LEDGER_DIR / "stats.json").read_text(encoding="utf-8")) \
        if (LEDGER_DIR / "stats.json").exists() else {}
    rescored = [json.loads(l) for l in (LEDGER_DIR / "rescored.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] \
        if (LEDGER_DIR / "rescored.jsonl").exists() else []

    gated = [r for r in pool if r.get("passed")]
    # Release-candidate VIEW = the gated q4 wallpapers, diversity-selected (prompt §Selection).
    # v7 is a floor here, not the gate, and the mining head is uncalibrated on strange — so the
    # release view draws from the whole GATED pool (release floors = pool floors) and Matt judges
    # by eye, rather than shipping only the few that clear the strict 0.90/0.50 production floors.
    # We run the driver's OWN head-split greedy select (same kernel), then annotate which picks
    # also clear the strict production release floors (0.90 wallpaper / 0.50 mining).
    selected = _diversity_select(out_dir, gated, n=min(len(gated), 24))
    selected_ids = {r["id"] for r in selected}
    inventory = [r for r in gated if r["id"] not in selected_ids]
    full_res = {p.stem for p in (out_dir / "release").glob("*.png")} if (out_dir / "release").exists() else set()

    # colorized-but-floor-rejected attempts (rendered, both-head scored — the real reject
    # autopsy: what got a wallpaper but didn't clear even the permissive pool floor).
    floor_rejected = [r for r in pool if not r.get("passed") and r.get("jpg")]

    # score EVERY rendered pool tile with both heads (batched); the routed head is already
    # stored but we recompute both so the sheets are self-consistent.
    heads = DualHeads()
    jpgs = [ROOT / r["jpg"] for r in pool if r.get("jpg")]
    scored = heads.score_many(jpgs)

    def both_of(r):
        j = str(ROOT / r["jpg"]) if r.get("jpg") else None
        s = scored.get(j, {})
        return s.get("wp"), s.get("mn")

    # ---- per-stage counts --------------------------------------------------- #
    n_clusters = len(set(intake.get("cluster_tags", {}).values()))
    counts = {
        "candidates": producer.get("n_candidates"),
        "rendered": producer.get("n_rendered"),
        "guard_passed": producer.get("n_guard_pass"),
        "guard_failed": producer.get("n_guard_fail"),
        "guard_fail_reasons": producer.get("guard_fail_reasons"),
        "floor_admitted": producer.get("n_floor_admitted"),
        "below_floor": producer.get("n_below_floor"),
        "for_reference_decoded_class_3": producer.get("n_decoded_class_3"),
        "intake_admitted": intake.get("n_admitted"),
        "distinct_clusters": n_clusters,
        "colorized_attempts": len(pool),
        "gated_pool": len(gated),
        "release_candidates": len(selected),          # gated pool, diversity-selected view
        "strict_release_eligible": sum(1 for r in gated if _clears_strict(r)),
        "driver_strict_release_split": summary.get("release_split"),
    }

    # ---- release sheet ------------------------------------------------------ #
    _release_sheet(out_dir / "q4_release_sheet.png", selected, ledger, both_of,
                   heads.wp_gate, heads.mining_gate, counts, full_res)
    _autopsy_sheet(out_dir / "q4_autopsy_sheet.png", floor_rejected, inventory, rescored, ledger,
                   both_of, heads.wp_gate, heads.mining_gate)

    # ---- markdown + json ---------------------------------------------------- #
    md = _markdown(counts, selected, inventory, floor_rejected, rescored, ledger, both_of,
                   summary, producer, heads.wp_gate, heads.mining_gate, out_dir)
    (out_dir / "q4_readout.md").write_text(md, encoding="utf-8")
    readout = {"counts": counts, "release": [_row_json(r, ledger, both_of) for r in selected]}
    (out_dir / "q4_readout.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")

    print("[q4-readout] per-stage counts:")
    for k in ("candidates", "rendered", "guard_passed", "floor_admitted", "distinct_clusters",
              "colorized_attempts", "gated_pool", "release_candidates"):
        print(f"  {k:22s} {counts.get(k)}")
    print(f"[q4-readout] wrote {out_dir/'q4_release_sheet.png'}")
    print(f"[q4-readout] wrote {out_dir/'q4_autopsy_sheet.png'}")
    print(f"[q4-readout] wrote {out_dir/'q4_readout.md'}, q4_readout.json")
    return counts


def _row_json(r, ledger, both_of):
    lg = ledger.get(r["location_id"], {})
    wp, mn = both_of(r)
    return {
        "id": r["id"], "location_id": r["location_id"],
        "type": r["type"], "morph_cluster": r["morph_cluster"],
        "palette_flavor": r["palette_flavor"], "render_style": r["render_style"],
        "palette": r["palette"], "routed_head": r.get("head"),
        "routed_p_ge3": r.get("p_ge3"),
        "wallpaper_head": wp, "mining_head": mn,
        "framing": {"cx": lg.get("outcome_cx"), "cy": lg.get("outcome_cy"), "fw": lg.get("outcome_fw")},
        "q4_minibrot_id": lg.get("q4_minibrot_id"), "q4_G": lg.get("q4_G"), "q4_scale": lg.get("q4_scale"),
    }


# --------------------------------------------------------------------------- #
def _release_sheet(path, selected, ledger, both_of, wp_gate, mn_gate, counts, full_res):
    tw, th, pad, lh, head = 320, 180, 10, 62, 40
    cols = 4
    n = max(1, len(selected))
    rows = (n + cols - 1) // cols
    W = pad + cols * (tw + pad)
    H = head + rows * (th + lh + pad) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"q4_harvest — gated pool, diversity-selected ({len(selected)}) · both heads "
           f"(wp gate {wp_gate}, mn gate {mn_gate}; * = routed head, ✓ = clears strict floor, "
           f"◆ = full-res rendered)", fill=(235, 235, 235), font=_font(14))
    for i, r in enumerate(selected):
        lg = ledger.get(r["location_id"], {})
        wp, mn = both_of(r)
        cx = pad + (i % cols) * (tw + pad)
        cy = head + (i // cols) * (th + lh + pad)
        sheet.paste(_thumb(r["jpg"], tw, th), (cx, cy))
        y = cy + th + 2
        strict = " ✓strict" if _clears_strict(r) else ""
        fr = " ◆" if r["id"] in full_res else ""
        d.text((cx + 2, y), f"{i+1}. {r['type']} {r['morph_cluster']}  {r['render_style']}{strict}{fr}",
                fill=(220, 220, 160), font=_font(11))
        d.text((cx + 2, y + 13), f"{r['palette_flavor']} · {r['palette'][:22]}",
                fill=(200, 210, 220), font=_font(10))
        d.text((cx + 2, y + 25), _heads_line(r.get("head"), wp, mn, wp_gate, mn_gate),
                fill=(180, 230, 200), font=_font(11))
        d.text((cx + 2, y + 38), f"G={lg.get('q4_G','?')} {lg.get('q4_minibrot_id','?')}",
                fill=(210, 190, 210), font=_font(9))
        d.text((cx + 2, y + 48),
                f"c=({_fw_short(lg.get('outcome_cx'))},{_fw_short(lg.get('outcome_cy'))}) fw={_fw_short(lg.get('outcome_fw'))}",
                fill=(150, 170, 190), font=_font(9))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _autopsy_sheet(path, floor_rejected, inventory, rescored, ledger, both_of, wp_gate, mn_gate):
    """Reject autopsy — the colorized-but-floor-rejected q4 wallpapers (both heads), plus any
    gated-but-not-selected inventory. Best-head-score first; capped so the sheet stays legible."""
    CAP = 45
    tw, th, pad, lh, head = 200, 112, 8, 50, 40
    cols = 6

    def best(r):
        wp, mn = both_of(r)
        return max((wp or {}).get("p_ge3", 0.0), (mn or {}).get("p_ge3", 0.0))
    items = sorted(inventory + floor_rejected, key=best, reverse=True)[:CAP]
    n = max(1, len(items))
    rows = (n + cols - 1) // cols
    W = pad + cols * (tw + pad)
    H = head + rows * (th + lh + pad) + pad
    sheet = Image.new("RGB", (W, H), (14, 14, 16))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"q4_harvest — reject autopsy: colorized but below pool floor "
           f"({len(floor_rejected)} floor-rej + {len(inventory)} inventory; top {len(items)} by "
           f"best head) · both heads", fill=(235, 235, 235), font=_font(13))
    for i, r in enumerate(items):
        lg = ledger.get(r["location_id"], {})
        wp, mn = both_of(r)
        cx = pad + (i % cols) * (tw + pad)
        cy = head + (i // cols) * (th + lh + pad)
        sheet.paste(_thumb(r["jpg"], tw, th), (cx, cy))
        y = cy + th + 2
        d.text((cx + 2, y), f"{r['type']} {r['morph_cluster']} {r['render_style']}",
                fill=(210, 210, 150), font=_font(9))
        d.text((cx + 2, y + 11), f"{r['palette_flavor']}·{r['palette'][:16]}",
                fill=(190, 200, 210), font=_font(9))
        d.text((cx + 2, y + 22), _heads_line(r.get("head"), wp, mn, wp_gate, mn_gate),
                fill=(180, 220, 195), font=_font(10))
        d.text((cx + 2, y + 34), f"G={lg.get('q4_G','?')} {str(lg.get('q4_minibrot_id',''))[:16]}",
                fill=(200, 185, 200), font=_font(8))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


# --------------------------------------------------------------------------- #
def _markdown(counts, selected, inventory, floor_rejected, rescored, ledger, both_of, summary,
              producer, wp_gate, mn_gate, out_dir):
    L = []
    w = L.append
    w("# q4_harvest emission — release candidates + autopsy (both heads)\n")
    w("The q4 tight harvest (goodness field, orthogonal to v7) flowed through emission as a "
      "**floor** source: admitted on `p_notbad>=0.5` ∧ guard_pass ∧ distinct — NOT on v7's "
      "`decoded_class==3` (see docs/design/q4_harvest_emission.md). Both head scores are shown "
      "for every tile; the mining head is uncalibrated on strange renders — judge by eye.\n")

    w("## Per-stage counts\n")
    w("| stage | count |")
    w("|---|--:|")
    w(f"| harvest candidates | {counts['candidates']} |")
    w(f"| rendered (v7 decode) | {counts['rendered']} |")
    w(f"| guard-passed | {counts['guard_passed']} |")
    w(f"| guard-failed | {counts['guard_failed']} ({counts.get('guard_fail_reasons')}) |")
    w(f"| **floor-admitted** (p_notbad≥0.5) | **{counts['floor_admitted']}** |")
    w(f"| below floor | {counts['below_floor']} |")
    w(f"| — for reference, v7 q3 (decoded_class==3) would keep | {counts['for_reference_decoded_class_3']} |")
    w(f"| intake admitted | {counts['intake_admitted']} |")
    w(f"| **distinct morph clusters** (incremental medoid, cos 0.974) | **{counts['distinct_clusters']}** |")
    w(f"| colorized attempts | {counts['colorized_attempts']} |")
    w(f"| **gated pool** (pool floors: wp 0.75 / mn 0.25) | **{counts['gated_pool']}** |")
    w(f"| **release-candidate view** (gated pool, diversity-selected) | **{counts['release_candidates']}** |")
    w(f"| — of which clear the STRICT production floors (wp 0.90 / mn 0.50) | {counts['strict_release_eligible']} |")
    w("")
    w("**The release-candidate view is the whole gated q4 pool, diversity-selected** (heads never "
      "compared in one greedy step — smooth via the wallpaper head, strange via the mining head, "
      "two disjoint passes). v7 is a floor here and the mining head is uncalibrated on strange "
      "renders, so the view is NOT truncated to the strict-floor subset — Matt judges by eye. "
      f"For reference, the driver's strict-floor pass shipped only "
      f"{counts.get('driver_strict_release_split', {}).get('smooth_selected', 0) + counts.get('driver_strict_release_split', {}).get('strange_selected', 0)} "
      f"(smooth ≥0.90 + strange ≥0.50).\n")

    w("## Release candidates — both head scores\n")
    w("The gated q4 pool, diversity-selected. `wp`/`mn` = wallpaper-head / mining-head marginal "
      "`p_ge3`; `★` marks the ROUTED head (the score the driver gated on); `✓` = clears that "
      f"head's STRICT production floor (wallpaper {STRICT_WP} / mining {STRICT_MN}). The mining "
      "head is uncalibrated on strange minibrot renders — treat mn as a hint, judge by eye.\n")
    w("| # | id | type/cluster | flavor/style | palette | wp p3 | mn p3 | strict? | G | minibrot | fw |")
    w("|--:|---|---|---|---|--:|--:|:-:|--:|---|--:|")
    for i, r in enumerate(selected, 1):
        lg = ledger.get(r["location_id"], {})
        wp, mn = both_of(r)
        routed = r.get("head")
        wp3 = (f"{wp['p_ge3']:.3f}" + ("★" if routed == "wallpaper" else "")) if wp else "—"
        mn3 = (f"{mn['p_ge3']:.3f}" + ("★" if routed == "mining" else "")) if mn else "—"
        strict = "✓" if _clears_strict(r) else ""
        w(f"| {i} | {r['id']} | {r['type']}/{r['morph_cluster']} | "
          f"{r['palette_flavor']}/{r['render_style']} | {r['palette'][:24]} | {wp3} | "
          f"{mn3} | {strict} | {lg.get('q4_G','?')} | {lg.get('q4_minibrot_id','?')} | {_fw_short(lg.get('outcome_fw'))} |")
    w("")

    w("## Reject / pool autopsy\n")
    w(f"- **{len(floor_rejected)}** colorized wallpapers fell below the permissive pool floor "
      f"(wallpaper 0.75 / mining 0.25) — the reject autopsy in `q4_autopsy_sheet.png` (both "
      f"heads, best-head-first), showing what the deficit colorize produced that didn't gate.")
    w(f"- **{len(inventory)}** gated-but-not-selected wallpapers banked as inventory "
      f"(0 here — release-n exceeded the gated count, so every gated wallpaper is in the view).")
    guard_fail = [r for r in rescored if not r.get("guard_pass")]
    below = [r for r in rescored if r.get("guard_pass") and (r.get("p_notbad") or 0) < 0.5]
    w(f"- **{len(guard_fail)}** candidates rejected at the guard "
      f"({dict(Counter(r.get('guard_fail') for r in guard_fail))}).")
    w(f"- **{len(below)}** candidates below the v7 floor (p_notbad<0.5) — the only q4 framings "
      f"v7 calls clearly bad.")
    w("")
    # per-minibrot floor-admit yield (surfacing whether the floor rejects a big fraction)
    by_mb = defaultdict(lambda: [0, 0])   # minibrot -> [rendered, floor_admit]
    for r in rescored:
        by_mb[r.get("q4_minibrot_id")][0] += 1
        if r.get("guard_pass") and (r.get("p_notbad") or 0) >= 0.5:
            by_mb[r.get("q4_minibrot_id")][1] += 1
    w("### floor yield per minibrot\n")
    w("| minibrot | rendered | floor-admitted |")
    w("|---|--:|--:|")
    for mb, (nr, na) in sorted(by_mb.items(), key=lambda kv: -kv[1][1]):
        w(f"| {mb} | {nr} | {na} |")
    w("")
    w("## Sheets\n")
    w(f"- `{(out_dir/'q4_release_sheet.png').relative_to(ROOT)}` — the release candidates")
    w(f"- `{(out_dir/'q4_autopsy_sheet.png').relative_to(ROOT)}` — the pool inventory\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "out" / "emission" / "q4_harvest"),
                    help="emission driver --out dir for the q4 run")
    args = ap.parse_args(argv)
    build(Path(args.out).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
