r"""sitting_reads.py — the four reads off a merged wallpaper sitting.

Joins each batch's `images.jsonl` (render + provenance + the v3 pre-label) to its tracked
labels sidecar and answers, per prompts/wallpaper_sitting_merge_prompt.md §2:

  A  CORRECTION RATE — the head's fresh-era report card. Where Matt's label differs from the
     suggested tier, and in which direction, sliced by score bin / coloring_source / intake
     source tag / suggested tier. Plus the same at the >=3 boundary alone, which is the cut
     the emission gate actually makes.
  B  FLOOR-ADMIT ADJUDICATION — labeled tier distributions for the two floor-admit sources
     against the machine-admitted rest. This is what separates "the head under-prices this
     vein" from "the vein is weak".
  C  INTAKE PURITY, LOCATION LEVEL — per intake source and per partition, the fraction of
     LOCATIONS whose best labeled row is >=3, and the histogram of location maxima. A
     location is the unit stage 1 buys, so this is the t_good-shaped number; a per-ROW rate
     would answer a question nobody asks.
  D  V3 ON THE FRESH EVAL SPLIT — AP/AUC at >=2/>=3/>=4 and gate precision/recall at the
     deployed 0.90, on the STAMPED eval side only, beside the old-era 686-row read.

Every table states n. Nothing here changes a head, a gate or a floor; it is pure readout over
committed files, so it re-runs to the same numbers.

Outputs (scratch/wallpaper_sitting/): report.md, report.json, and one small contact sheet per
read — the visual sample is part of the answer, not decoration: a correction rate cannot tell
you whether the head was wrong or Matt was strict, and the crops can.

  uv run python tools/wallpaper/sitting_reads.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BATCHES = ROOT / "data" / "wallpaper_corpus" / "batches"
OUT = ROOT / "scratch" / "wallpaper_sitting"
SITTING = ["2026-08-05_wallpaper_fresh_sheet_v1", "2026-08-05_wallpaper_colorize_path_v1"]
GATE = 0.90                       # read from the pin below; this is only the doc value

# The old-era comparison column: v3's held-out eval read on the dramatic+humanq3
# population. Frozen here as the RECORD it is — the slice is a different population, so it
# is quoted, never recomputed alongside.
#
# Provenance names the PRODUCER, not a file. It was first measured by the eval revival
# (prompts/wallpaper_eval_revival_prompt.md, 2026-08-05) into a scratch report, which made
# this a citation of a deletable path — and the crops it was measured on had themselves
# already been deleted once. `report_v4_eval.py` re-derives the same slice from the v3
# checkpoint over the rebuilt crops, and on 2026-08-06 reproduced all nine values below
# exactly (`slices.old_era_686.v3`, gate rungs under `precision_of_passers["0.9"]`). So the
# numbers now have a maintained producer, and no wipe can orphan them.
OLD_ERA = {"n": 686, "tiers": {1: 116, 2: 295, 3: 185, 4: 90},
           "ap_ge2": 0.956, "ap_ge3": 0.669, "ap_ge4": 0.345,
           "auc_ge3": 0.748, "gate_fire": 205, "gate_precision": 0.683, "gate_recall": 0.509,
           "source": "tools/wallpaper/report_v4_eval.py -> slices.old_era_686.v3"}


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# join
# --------------------------------------------------------------------------- #
def load():
    from tools.wallpaper.merge_sitting import sidecar_for
    rows = []
    for b in SITTING:
        side = json.loads(sidecar_for(b).read_text(encoding="utf-8"))
        for line in (BATCHES / b / "images.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            lab = side.get(r["image_id"])
            if lab is None:
                continue
            pv = r["provenance"]
            rows.append({
                "id": r["image_id"], "batch": b, "crop": BATCHES / b / "crops" / f"{r['image_id']}.jpg",
                "label": int(lab), "suggested": int(r["suggested_tier"]),
                "p_ge2": r["head_v3"]["p_ge2"], "p_ge3": r["head_v3"]["p_ge3"],
                "p_ge4": r["head_v3"]["p_ge4"], "pred": r["head_v3"]["pred"],
                "bin": pv["screen_bin"], "coloring": pv["coloring_source"],
                "tag": pv["source_tag"], "intake": pv["intake_source"],
                "floor": bool(pv["floor_admit"]), "partition": pv["partition"],
                "split": pv["split_side"],
                # the LOCATION key: the render geometry, not the id (ids differ per batch)
                "loc": (r["render"]["cx"], r["render"]["cy"], r["render"]["fw"],
                        r["render"]["fractal_type"], r["render"]["c_re"], r["render"]["c_im"]),
                "palette": r["render"]["palette"],
            })
    return rows


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ap(y, s):
    """Average precision (the step-wise sum used everywhere else in this tree)."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def auc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def corr_block(rows):
    """(n, exact, up, down, mean signed delta) for a slice."""
    n = len(rows)
    if not n:
        return {"n": 0}
    d = [r["label"] - r["suggested"] for r in rows]
    return {"n": n,
            "agree": sum(1 for x in d if x == 0) / n,
            "up": sum(1 for x in d if x > 0) / n,
            "down": sum(1 for x in d if x < 0) / n,
            "mean_delta": float(np.mean(d)),
            "within_one": sum(1 for x in d if abs(x) <= 1) / n}


def boundary_block(rows):
    """The >=3 crossing only: the cut the emission gate makes."""
    n = len(rows)
    if not n:
        return {"n": 0}
    sg = [r["suggested"] >= 3 for r in rows]
    lb = [r["label"] >= 3 for r in rows]
    tp = sum(1 for a, b in zip(sg, lb) if a and b)
    fp = sum(1 for a, b in zip(sg, lb) if a and not b)
    fn = sum(1 for a, b in zip(sg, lb) if not a and b)
    tn = sum(1 for a, b in zip(sg, lb) if not a and not b)
    return {"n": n, "agree": (tp + tn) / n, "suggested_ge3": tp + fp, "labeled_ge3": tp + fn,
            "precision": (tp / (tp + fp)) if tp + fp else None,
            "recall": (tp / (tp + fn)) if tp + fn else None,
            "false_up": fp / n, "false_down": fn / n}


# --------------------------------------------------------------------------- #
# contact sheets
# --------------------------------------------------------------------------- #
def sheet(items, path, title, cols=6, tw=248):
    """`items` = [(crop_path, caption)]. Small, captioned, deterministic."""
    if not items:
        return None
    th = round(tw * 9 / 16)
    pad, bar, head = 6, 30, 26
    rowsn = (len(items) + cols - 1) // cols
    W = cols * (tw + pad) + pad
    H = head + rowsn * (th + bar + pad) + pad
    im = Image.new("RGB", (W, H), (14, 15, 19))
    d = ImageDraw.Draw(im)
    d.text((pad, 7), title, fill=(210, 214, 222))
    for i, (cp, cap) in enumerate(items):
        cx = pad + (i % cols) * (tw + pad)
        cy = head + (i // cols) * (th + bar + pad)
        try:
            with Image.open(cp) as t:
                im.paste(t.convert("RGB").resize((tw, th), Image.LANCZOS), (cx, cy))
        except Exception:                                        # noqa: BLE001
            d.rectangle([cx, cy, cx + tw, cy + th], fill=(40, 20, 20))
            d.text((cx + 6, cy + 6), "missing crop", fill=(200, 120, 120))
        for j, line in enumerate(cap.split("\n")[:2]):
            d.text((cx + 2, cy + th + 2 + j * 12), line[:46], fill=(150, 158, 170))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path.name


def cap(r):
    return (f"{r['id']}  L{r['label']} vs S{r['suggested']}\n"
            f"p3 {r['p_ge3']:.2f} {r['tag'][:14]} {r['coloring'][:9]}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def md_table(head, body):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" if i == 0 else "--:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def pct(x):
    return "—" if x is None else f"{100*x:.1f}%"


def num(x, k=3):
    return "—" if x is None else f"{x:.{k}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from tools.wallpaper.wallpaper_pins import GATE_THRESHOLD, HEAD_VERSION, HEAD_CKPT_REL
    rows = load()
    log(f"[reads] joined {len(rows)} labeled rows over {len(SITTING)} batches")
    R = {"head": {"version": HEAD_VERSION, "ckpt": HEAD_CKPT_REL, "gate": GATE_THRESHOLD},
         "n_rows": len(rows), "batches": SITTING}

    by = lambda key: defaultdict(list, {k: [r for r in rows if r[key] == k]           # noqa: E731
                                        for k in sorted({r[key] for r in rows})})

    # ---- A. correction rate ------------------------------------------------
    A = {"overall": corr_block(rows), "boundary_overall": boundary_block(rows),
         "by": {}, "boundary_by": {}}
    for key in ("bin", "coloring", "tag", "suggested"):
        A["by"][key] = {str(k): corr_block(v) for k, v in by(key).items()}
        A["boundary_by"][key] = {str(k): boundary_block(v) for k, v in by(key).items()}
    R["A_correction_rate"] = A

    worst = sorted(rows, key=lambda r: (r["label"] - r["suggested"]))
    A["sample"] = {
        "head_overrated": sheet([(r["crop"], cap(r)) for r in worst[:12]],
                                OUT / "A1_head_overrated.png",
                                "A · head OVER-rated: labeled far below its suggestion"),
        "head_underrated": sheet([(r["crop"], cap(r)) for r in worst[-12:][::-1]],
                                 OUT / "A2_head_underrated.png",
                                 "A · head UNDER-rated: labeled far above its suggestion"),
    }

    # ---- B. floor-admit adjudication ---------------------------------------
    def dist(rs):
        c = Counter(r["label"] for r in rs)
        n = len(rs)
        return {"n": n, "hist": {str(t): c.get(t, 0) for t in (1, 2, 3, 4)},
                "frac_ge3": (sum(c.get(t, 0) for t in (3, 4)) / n) if n else None,
                "frac_ge2": (sum(c.get(t, 0) for t in (2, 3, 4)) / n) if n else None,
                "mean": float(np.mean([r["label"] for r in rs])) if n else None}

    B = {"q4_harvest": dist([r for r in rows if r["tag"] == "q4_harvest"]),
         "human_q3plus": dist([r for r in rows if r["tag"] == "human_q3plus"]),
         "floor_admitted_all": dist([r for r in rows if r["floor"]]),
         "machine_admitted_rest": dist([r for r in rows if not r["floor"]]),
         "by_tag": {k: dist(v) for k, v in by("tag").items()}}
    fa = [r for r in rows if r["floor"]]
    B["head_pricing"] = {
        "floor_admitted": corr_block(fa),
        "machine_admitted": corr_block([r for r in rows if not r["floor"]]),
        "note": "mean_delta > 0 means Matt labeled ABOVE the suggestion — the head "
                "under-priced the vein. A weak vein shows as a low frac_ge3 with mean_delta "
                "near the machine-admitted rows'.",
    }
    B["sample"] = sheet(
        [(r["crop"], cap(r)) for r in sorted([x for x in fa if x["label"] >= 3],
                                             key=lambda r: -r["label"])[:12]],
        OUT / "B_floor_admit_best.png", "B · best-labeled floor-admitted rows (q4 + humanq3)")
    R["B_floor_admit"] = B

    # ---- C. intake purity, LOCATION level ----------------------------------
    locs = defaultdict(list)
    for r in rows:
        locs[r["loc"]].append(r)
    lmax = {k: max(x["label"] for x in v) for k, v in locs.items()}
    linfo = {k: v[0] for k, v in locs.items()}

    def purity(keys):
        if not keys:
            return {"n_locations": 0}
        m = [lmax[k] for k in keys]
        c = Counter(m)
        return {"n_locations": len(keys),
                "max_hist": {str(t): c.get(t, 0) for t in (1, 2, 3, 4)},
                "frac_best_ge3": sum(1 for x in m if x >= 3) / len(m),
                "frac_best_ge2": sum(1 for x in m if x >= 2) / len(m),
                "mean_max": float(np.mean(m))}

    C = {"all": purity(list(locs)),
         "by_intake_source": {}, "by_source_tag": {}, "by_partition": {}}
    for field, dest in (("intake", "by_intake_source"), ("tag", "by_source_tag"),
                        ("partition", "by_partition")):
        for val in sorted({linfo[k][field] for k in locs}):
            C[dest][val] = purity([k for k in locs if linfo[k][field] == val])
    C["note"] = ("A location's best labeled row is the honest 'is this location any good' "
                 "read, but it is bounded by how many colorings the sheet drew for it: "
                 "4 for pool_draw, 1 for colorize_path. Locations in both got up to 5.")
    C["renders_per_location"] = dict(sorted(Counter(len(v) for v in locs.values()).items()))
    best_locs = sorted(locs, key=lambda k: -lmax[k])[:12]
    C["sample"] = sheet(
        [(max(locs[k], key=lambda r: r["label"])["crop"],
          cap(max(locs[k], key=lambda r: r["label"]))) for k in best_locs],
        OUT / "C_location_best.png", "C · the best labeled render of the top locations")
    R["C_intake_purity"] = C

    # ---- D. v3 on the fresh eval split -------------------------------------
    ev = [r for r in rows if r["split"] == "eval"]
    D = {"n": len(ev), "tiers": {str(t): sum(1 for r in ev if r["label"] == t) for t in (1, 2, 3, 4)},
         "old_era": OLD_ERA}
    for k, thr, sc in (("ge2", 2, "p_ge2"), ("ge3", 3, "p_ge3"), ("ge4", 4, "p_ge4")):
        y = [r["label"] >= thr for r in ev]
        s = [r[sc] for r in ev]
        D[k] = {"n_pos": int(sum(y)), "ap": ap(y, s), "auc": auc(y, s)}
    fire = [r for r in ev if r["p_ge3"] > GATE_THRESHOLD]
    good = [r for r in ev if r["label"] >= 3]
    D["gate_at_%.2f" % GATE_THRESHOLD] = {
        "fires": len(fire), "fire_rate": len(fire) / len(ev) if ev else None,
        "precision": (sum(1 for r in fire if r["label"] >= 3) / len(fire)) if fire else None,
        "recall": (sum(1 for r in fire if r["label"] >= 3) / len(good)) if good else None,
        "n_good": len(good)}
    D["by_coloring"] = {}
    for cs in sorted({r["coloring"] for r in ev}):
        sub = [r for r in ev if r["coloring"] == cs]
        y = [r["label"] >= 3 for r in sub]
        D["by_coloring"][cs] = {"n": len(sub), "n_good": int(sum(y)),
                                "ap_ge3": ap(y, [r["p_ge3"] for r in sub]),
                                "auc_ge3": auc(y, [r["p_ge3"] for r in sub])}
    miss = sorted([r for r in ev if r["label"] >= 3 and r["p_ge3"] <= GATE_THRESHOLD],
                  key=lambda r: r["p_ge3"])[:12]
    D["sample"] = sheet([(r["crop"], cap(r)) for r in miss], OUT / "D_gate_misses.png",
                        f"D · eval-side rows Matt called >=3 that the {GATE_THRESHOLD} gate rejects")
    R["D_eval"] = D

    (OUT / "report.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    write_md(R, rows)
    log(f"[reads] -> {OUT}")


def write_md(R, rows):
    A, B, C, D = R["A_correction_rate"], R["B_floor_admit"], R["C_intake_purity"], R["D_eval"]
    w = []
    w.append(f"# wallpaper sitting — reads · {R['n_rows']} labeled rows · head {R['head']['version']}\n")
    w.append(f"Batches: {', '.join(R['batches'])}. Gate {R['head']['gate']}. "
             f"Nothing here changes a head, gate or floor.\n")

    o, bo = A["overall"], A["boundary_overall"]
    w.append("\n## A · correction rate\n")
    w.append(f"**{pct(o['agree'])} of {o['n']} rows kept the suggestion.** "
             f"Matt went UP on {pct(o['up'])}, DOWN on {pct(o['down'])}; "
             f"mean signed delta **{o['mean_delta']:+.3f}** tiers, within-one {pct(o['within_one'])}.\n")
    w.append(f"At the **>=3 boundary** (the cut the gate makes): agree {pct(bo['agree'])}, "
             f"suggestion precision {pct(bo['precision'])}, recall {pct(bo['recall'])} "
             f"({bo['suggested_ge3']} suggested >=3, {bo['labeled_ge3']} labeled >=3, n={bo['n']}).\n")
    for key, title in (("suggested", "by suggested tier"), ("bin", "by screen bin"),
                       ("coloring", "by coloring_source"), ("tag", "by intake source tag")):
        w.append(f"\n**{title}**\n")
        w.append(md_table(
            [key, "n", "agree", "up", "down", "mean Δ", ">=3 agree", ">=3 prec"],
            [[k, v["n"], pct(v["agree"]), pct(v["up"]), pct(v["down"]), f"{v['mean_delta']:+.2f}",
              pct(A["boundary_by"][key][k]["agree"]), pct(A["boundary_by"][key][k]["precision"])]
             for k, v in A["by"][key].items() if v["n"]]))
        w.append("")
    w.append(f"\nSamples: `{A['sample']['head_overrated']}`, `{A['sample']['head_underrated']}`\n")

    w.append("\n## B · floor-admit adjudication\n")
    w.append(md_table(["source", "n", "1", "2", "3", "4", ">=2", ">=3", "mean", "mean Δ vs suggestion"],
                      [[k, v["n"], v["hist"]["1"], v["hist"]["2"], v["hist"]["3"], v["hist"]["4"],
                        pct(v["frac_ge2"]), pct(v["frac_ge3"]), num(v["mean"], 2),
                        f"{B['head_pricing']['floor_admitted']['mean_delta']:+.2f}"
                        if k in ("q4_harvest", "human_q3plus", "floor_admitted_all")
                        else f"{B['head_pricing']['machine_admitted']['mean_delta']:+.2f}"]
                       for k, v in ((k, B[k]) for k in ("q4_harvest", "human_q3plus",
                                                        "floor_admitted_all",
                                                        "machine_admitted_rest"))]))
    w.append(f"\n{B['head_pricing']['note']}\n")
    w.append(f"Sample: `{B['sample']}`\n")

    w.append("\n## C · intake purity, location level\n")
    a = C["all"]
    w.append(f"**{pct(a['frac_best_ge3'])} of {a['n_locations']} labeled locations have a best "
             f"render at >=3**; {pct(a['frac_best_ge2'])} reach >=2. Mean location max "
             f"{num(a['mean_max'], 2)}. Renders per location: {C['renders_per_location']}.\n")
    w.append(f"\n{C['note']}\n")
    for dest, title in (("by_intake_source", "by intake source"), ("by_source_tag", "by source tag"),
                        ("by_partition", "by partition")):
        w.append(f"\n**{title}**\n")
        w.append(md_table([title.split()[-1], "locations", "max 1", "2", "3", "4", "best>=3", "mean max"],
                          [[k, v["n_locations"], v["max_hist"]["1"], v["max_hist"]["2"],
                            v["max_hist"]["3"], v["max_hist"]["4"], pct(v["frac_best_ge3"]),
                            num(v["mean_max"], 2)]
                           for k, v in C[dest].items() if v["n_locations"]]))
        w.append("")
    w.append(f"\nSample: `{C['sample']}`\n")

    g = D["gate_at_0.90"]
    old = D["old_era"]
    w.append("\n## D · v3 on the fresh eval split\n")
    w.append(f"Stamped eval side only: **n={D['n']}**, tiers {D['tiers']}. "
             f"Old-era column is the 686-row dramatic+humanq3 read ({old['source']}) — a "
             f"DIFFERENT population, quoted for scale, not a paired comparison.\n")
    w.append(md_table(["measure", f"fresh eval (n={D['n']})", f"old era (n={old['n']})"],
                      [["AP >=2", num(D["ge2"]["ap"]), num(old["ap_ge2"])],
                       ["AP >=3", num(D["ge3"]["ap"]), num(old["ap_ge3"])],
                       ["AP >=4", num(D["ge4"]["ap"]), num(old["ap_ge4"])],
                       ["AUC >=3", num(D["ge3"]["auc"]), num(old["auc_ge3"])],
                       ["gate fires", f"{g['fires']} ({pct(g['fire_rate'])})",
                        f"{old['gate_fire']} ({pct(old['gate_fire']/old['n'])})"],
                       ["gate precision", pct(g["precision"]), pct(old["gate_precision"])],
                       ["gate recall", pct(g["recall"]), pct(old["gate_recall"])],
                       ["n labeled >=3", g["n_good"], sum(old["tiers"][t] for t in (3, 4))]]))
    w.append("\n**by coloring_source, eval side**\n")
    w.append(md_table(["coloring_source", "n", "n >=3", "AP >=3", "AUC >=3"],
                      [[k, v["n"], v["n_good"], num(v["ap_ge3"]), num(v["auc_ge3"])]
                       for k, v in D["by_coloring"].items()]))
    w.append(f"\nSample: `{D['sample']}`\n")
    (OUT / "report.md").write_text("\n".join(w), encoding="utf-8")


if __name__ == "__main__":
    main()
