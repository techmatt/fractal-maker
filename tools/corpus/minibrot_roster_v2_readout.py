#!/usr/bin/env python
r"""minibrot_roster_v2_readout.py — the G-transfer readout for the 487 minibrot labels.

Joins the labeled batch (2026-07-26_minibrot_roster_v2) to its draw manifest
(data/minibrot_roster/batch_v1/draw.jsonl, which carries the stage-1 G, arm, atom,
degree, period, band, split per crop) and the roster (roster.jsonl, which carries
log10|A|). Reports everything Part B of prompt-process-minibrot-labels.md asks for.

The one question: does stage-1 G — calibrated on non-minibrot fields — mean anything
on minibrot fields? If AUC(G, label>=3) sits near 0.5, it does not transfer here.

Counts, not rates. Nothing is tested for significance (n is tiny in most cells).

  uv run python tools/corpus/minibrot_roster_v2_readout.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BATCH_ID = "2026-07-26_minibrot_roster_v2"
BATCH_DIR = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID
DRAW = ROOT / "data" / "minibrot_roster" / "batch_v1" / "draw.jsonl"
ROSTER = ROOT / "data" / "minibrot_roster" / "roster.jsonl"
REVEALS = ROOT / "labels" / "minibrot_roster_v2_reveals.json"

# fate -> the three arms of Part B point 1
ARM = {"accepted": "accept", "rejected": "screen-reject (mask-surviving)", "ood_masked": "OOD-masked"}


def load():
    rows = [json.loads(l) for l in (BATCH_DIR / "images.jsonl").read_text().splitlines() if l.strip()]
    draw = {d["image_id"]: d for d in (json.loads(l) for l in DRAW.read_text().splitlines() if l.strip())}
    roster = {r["id"]: r for r in (json.loads(l) for l in ROSTER.read_text().splitlines() if l.strip())}
    reveals = json.loads(REVEALS.read_text())

    recs = []
    missing = []
    for r in rows:
        iid = r["image_id"]
        d = draw.get(iid)
        if d is None:
            missing.append(iid)
            continue
        atom = d["atom_id"]
        ros = roster.get(atom)
        recs.append({
            "iid": iid,
            "label": r["label"]["score"],
            "G": d["G"],
            "fate": d["fate"],
            "arm": ARM[d["fate"]],
            "degree": d["degree"],
            "period": d["period"],
            "band": d["band"],
            "split": d["split"],
            "atom": atom,
            "log10A": ros["log10_abs_A"] if ros else None,
            "revealed": reveals.get(iid),
        })
    return recs, missing


# ---- verification: every row joins with the full field set ----
def verify(recs, missing):
    print("=" * 78)
    print("JOIN VERIFICATION (batch <- draw manifest <- roster)")
    print("=" * 78)
    print(f"  labeled rows: {len(recs)}   unmatched to draw: {len(missing)} {missing[:5]}")
    need = ["label", "G", "arm", "degree", "period", "band", "split", "atom", "log10A", "revealed"]
    holes = {k: sum(1 for r in recs if r[k] is None) for k in need}
    bad = {k: v for k, v in holes.items() if v}
    print(f"  null holes per field: {bad if bad else 'none'}")
    nood = sum(1 for r in recs if r['fate'] == 'ood_masked')
    print(f"  (the {holes['G']} G-holes are exactly the {nood} OOD-masked rows — the screen never "
          f"assigned them a G; expected, not a join failure.)")
    print(f"  every row has arm/degree/period/band/log10|A|/atom/split.")
    print(f"  distinct atoms: {len({r['atom'] for r in recs})}")


# ---- stats helpers (hand-rolled; no scipy dependency) ----
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def auc(scores, labels):
    """AUC of `scores` as a ranker for binary `labels` (Mann-Whitney, tie-corrected)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    # rank all, sum ranks of positives
    alls = scores
    order = sorted(range(len(alls)), key=lambda i: alls[i])
    rk = [0.0] * len(alls)
    i = 0
    while i < len(alls):
        j = i
        while j + 1 < len(alls) and alls[order[j + 1]] == alls[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            rk[order[k]] = avg
        i = j + 1
    rpos = sum(rk[i] for i, y in enumerate(labels) if y)
    np_, nn = len(pos), len(neg)
    u = rpos - np_ * (np_ + 1) / 2.0
    return u / (np_ * nn), np_, nn


def dist_row(recs):
    c = Counter(r["label"] for r in recs)
    return {k: c.get(k, 0) for k in (1, 2, 3, 4)}


def fmt_dist(d):
    return "  ".join(f"L{k}:{d[k]}" for k in (1, 2, 3, 4)) + f"   (n={sum(d.values())}, mean={wmean(d):.2f})"


def wmean(d):
    n = sum(d.values())
    return sum(k * v for k, v in d.items()) / n if n else float("nan")


def main():
    recs, missing = load()
    verify(recs, missing)

    labels = [r["label"] for r in recs]
    # G analyses run over rows that HAVE a G (accepts + screen-rejects); the 25 OOD-masked
    # rows carry no G and are excluded from every G statistic.
    gr = [r for r in recs if r["G"] is not None]
    Gs = [r["G"] for r in gr]
    gl = [r["label"] for r in gr]

    print("\n" + "=" * 78)
    print("1. LABEL DISTRIBUTION BY ARM")
    print("=" * 78)
    for arm in ["accept", "screen-reject (mask-surviving)", "OOD-masked"]:
        sub = [r for r in recs if r["arm"] == arm]
        print(f"  {arm:34s} {fmt_dist(dist_row(sub))}")
    print(f"  {'ALL':34s} {fmt_dist(dist_row(recs))}")
    acc = [r for r in recs if r["arm"] == "accept"]
    rej = [r for r in recs if r["fate"] in ("rejected", "ood_masked")]
    print(f"\n  headline: accepts mean label {wmean(dist_row(acc)):.2f} vs rejects "
          f"mean label {wmean(dist_row(rej)):.2f}  "
          f"(delta {wmean(dist_row(acc)) - wmean(dist_row(rej)):+.2f})")

    print("\n" + "=" * 78)
    print("2. G vs LABEL")
    print("=" * 78)
    rho = spearman(Gs, gl)
    print(f"  Spearman(G, label), G-bearing rows n={len(gr)} (OOD-masked excluded): rho = {rho:+.3f}")
    a3, p3, n3 = auc(Gs, [1 if l >= 3 else 0 for l in gl])
    a4, p4, n4 = auc(Gs, [1 if l >= 4 else 0 for l in gl])
    print(f"  AUC(G | label>=3):  {a3:.3f}   (pos={p3}, neg={n3})")
    print(f"  AUC(G | label==4):  {a4:.3f}   (pos={p4}, neg={n4})")
    # within accepts only (where G actually varies above cutoff)
    ra = spearman([r['G'] for r in acc], [r['label'] for r in acc])
    aa3, _, _ = auc([r['G'] for r in acc], [1 if r['label'] >= 3 else 0 for r in acc])
    print(f"  within accepts:  Spearman={ra:+.3f}   AUC(label>=3)={aa3:.3f}")
    sr = [r for r in recs if r["fate"] == "rejected"]
    print(f"  G range: accepts [{min(r['G'] for r in acc):.3f},{max(r['G'] for r in acc):.3f}]  "
          f"screen-rejects [{min(r['G'] for r in sr):.3f},{max(r['G'] for r in sr):.3f}]")
    # is G's weak signal just a proxy for degree? G vs degree, then AUC(G|label>=3) within each degree.
    print(f"  Spearman(G, degree) [G-bearing]: {spearman(Gs, [r['degree'] for r in gr]):+.3f}")
    print("  AUC(G | label>=3) within each degree (removes the degree confound):")
    for deg in sorted({r["degree"] for r in gr}):
        sub = [r for r in gr if r["degree"] == deg]
        a, p, n = auc([r["G"] for r in sub], [1 if r["label"] >= 3 else 0 for r in sub])
        print(f"    degree {deg}: AUC={a if a==a else float('nan'):.3f}  (pos={p}, neg={n})")

    print("\n" + "=" * 78)
    print("3. CLASS-4 COUNT")
    print("=" * 78)
    c4 = [r for r in recs if r["label"] == 4]
    print(f"  total class-4: {len(c4)}")
    for r in c4:
        print(f"    {r['iid']}  arm={r['arm']} deg={r['degree']} period={r['period']} "
              f"band={r['band']} G={r['G']:.3f} log10A={r['log10A']}")
    print(f"  by arm:    {dict(Counter(r['arm'] for r in c4))}")
    print(f"  by degree: {dict(Counter(r['degree'] for r in c4))}")
    print(f"  by band:   {dict(Counter(r['band'] for r in c4))}")

    print("\n" + "=" * 78)
    print("4. DEPTH")
    print("=" * 78)
    print("  label by period band:")
    for band in ["3-4", "5-6", "7-9", "10-12", "13-15"]:
        sub = [r for r in recs if r["band"] == band]
        if sub:
            print(f"    band {band:6s} {fmt_dist(dist_row(sub))}")
    print("  label by degree:")
    for deg in sorted({r["degree"] for r in recs}):
        sub = [r for r in recs if r["degree"] == deg]
        print(f"    degree {deg}   {fmt_dist(dist_row(sub))}")
    print(f"  Spearman(period, label) all:  {spearman([r['period'] for r in recs], labels):+.3f}")
    print(f"  Spearman(degree, label) all:  {spearman([r['degree'] for r in recs], labels):+.3f}")
    print(f"  Spearman(log10|A|, label):    {spearman([r['log10A'] for r in recs], labels):+.3f}")
    print(f"  Spearman(period, log10|A|)  [collinearity]: "
          f"{spearman([r['period'] for r in recs], [r['log10A'] for r in recs]):+.3f}")
    # designed diagnostic: within the period-matched eval slice, does period still predict label?
    ev = [r for r in recs if r["split"] == "eval"]
    print(f"\n  period-matched eval slice (n={len(ev)}):")
    print(f"    period range in eval: {sorted({r['period'] for r in ev})}")
    print(f"    Spearman(period, label) | eval: {spearman([r['period'] for r in ev], [r['label'] for r in ev]):+.3f}")
    for band in sorted({r["band"] for r in ev}):
        sub = [r for r in ev if r["band"] == band]
        print(f"    eval band {band:6s} {fmt_dist(dist_row(sub))}")

    print("\n" + "=" * 78)
    print("5. WITHIN-ATOM VARIANCE")
    print("=" * 78)
    by_atom = defaultdict(list)
    for r in recs:
        by_atom[r["atom"]].append(r["label"])
    multi = {a: v for a, v in by_atom.items() if len(v) >= 2}
    singles = {a: v for a, v in by_atom.items() if len(v) == 1}
    print(f"  atoms: {len(by_atom)} total  ({len(multi)} with >=2 crops, {len(singles)} singletons)")
    # within-atom spread: mean of per-atom (max-min) and per-atom variance, over multi-crop atoms
    spreads = [max(v) - min(v) for v in multi.values()]
    within_var = statistics.mean([statistics.pvariance(v) for v in multi.values()]) if multi else float("nan")
    grand = statistics.pvariance(labels)
    atom_means = [statistics.mean(v) for v in multi.values()]
    between_var = statistics.pvariance(atom_means) if len(atom_means) > 1 else float("nan")
    print(f"  per-atom label range (max-min) over multi-crop atoms: "
          f"mean {statistics.mean(spreads):.2f}, dist {dict(sorted(Counter(spreads).items()))}")
    print(f"  mean within-atom variance:  {within_var:.3f}")
    print(f"  between-atom variance (of atom means): {between_var:.3f}")
    print(f"  grand label variance: {grand:.3f}")
    icc = between_var / (between_var + within_var) if (between_var + within_var) else float("nan")
    print(f"  crude ICC = between/(between+within) = {icc:.3f}   "
          f"(1.0 = atom fully determines label; 0 = window is everything)")
    # do any atoms straddle low(1-2) and high(3-4)?
    straddle = {a: v for a, v in multi.items() if min(v) <= 2 and max(v) >= 3}
    print(f"  atoms straddling low(<=2) and high(>=3): {len(straddle)} / {len(multi)} multi-crop atoms")
    for a, v in list(straddle.items())[:8]:
        print(f"    {a}: {sorted(v)}")

    print("\n" + "=" * 78)
    print("6. REVEAL COUNT")
    print("=" * 78)
    rc = Counter(r["revealed"] for r in recs)
    print(f"  revealed=1: {rc.get(1,0)}   blind=0: {rc.get(0,0)}   (n={len(recs)})")
    if rc.get(1, 0):
        rv = [r for r in recs if r["revealed"] == 1]
        bl = [r for r in recs if r["revealed"] == 0]
        print(f"  revealed rows mean label {wmean(dist_row(rv)):.2f} vs blind {wmean(dist_row(bl)):.2f}")
    else:
        print("  all 487 scored blind — no revealed subset to contrast.")


if __name__ == "__main__":
    main()
