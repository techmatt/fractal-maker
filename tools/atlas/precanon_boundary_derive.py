#!/usr/bin/env python
r"""precanon_boundary_derive.py — READ the dedup boundary out of Matt's 135 calibration verdicts.

The sheet (`precanon_calibration_sheet.py`) asked the eye where "close enough to be identical"
becomes "different enough to preserve", along the rule's own decision variable `d / min(fw)`.
This module is the follow-up it names: it DERIVES and AUDITS, and **adopts nothing** —
`production_seeder.DEDUP_K` / `DEDUP_SCALE` are read (to state where the standing constant sits)
and never written.

Population: `data/atlas/precanon_calibration/verdicts.json` (the durable export) joined to its
sibling `pairs.json` plan. UNSURE rows are reported separately and NEVER coerced to either side —
an unsure judgment is evidence about the boundary's width, not a vote.

Five reads:

  1. Per fw-ratio band (<=2x, 2-10x, >10x): the SAME->DISTINCT transition along `d/min(fw)` as
     last-SAME / first-DISTINCT / the interleaving zone between them. No smoothing, no fitted
     curve: an interleave is the measurement, and a logistic through it would hide exactly the
     thing the sheet was built to show.
  2. Cross-band: do the three transition zones share an interval (one K), or do they need a
     ratio-dependent form? Reported as a candidate either way.
  3. Anchors: the 15 nearest-miss pairs the current rule KEPT APART are the attention check.
     Any anchor judged SAME is flagged.
  4. Fidelity: how well `d/min(fw)` separates SAME from DISTINCT vs how well morph-cos does
     (the library `morph_gray` -> CLIP recipe, the same one the 0.974 cut lives in). AUC with
     stratified-bootstrap CIs, and the paired difference. SELECTION CAVEAT, stated in the
     output: the pairs were selected by rank-quantile strata **of `d/min(fw)`**, so that axis
     is uniformized over the domain and morph-cos is not. This is a check on the standing
     metric, not a re-derivation of it.
  5. Where the standing 0.974 near-dup cut sits relative to Matt's split, in morph-cos space.

Morph-cos costs 270 field dumps + 270 CLIP embeds (~GPU); it is cached to
`scratch/precanon_calibration/morph_cos.json` and reused. `--no-morph` skips reads 4-5.

  uv run python tools/atlas/precanon_boundary_derive.py            # full (morph pass included)
  uv run python tools/atlas/precanon_boundary_derive.py --no-morph # reads 1-3 only, seconds
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
           str(ROOT / "tools" / "sourcing"), str(ROOT / "tools" / "scoring"),
           str(ROOT / "tools" / "studies"), str(ROOT / "tools" / "wallpaper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                       # noqa: E402
import location as loc_mod                         # noqa: E402
import production_seeder as ps                     # noqa: E402  (DEDUP_K/SCALE — read, never moved)
import precanon_calibration_sheet as sheet         # noqa: E402  (render_block — the same authority)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VERDICTS = paths.durable("data/atlas/precanon_calibration/verdicts.json")
PAIRS = paths.durable("data/atlas/precanon_calibration/pairs.json")
OUT = Path(paths.scratch("precanon_calibration"))
MORPH_CACHE = OUT / "morph_cos.json"
BOUNDARY = OUT / "boundary.json"

BAND_ORDER = ("le2", "mid", "gt10")
BAND_LABEL = {"le2": "<=2x", "mid": "2-10x", "gt10": ">10x"}
STRICT_CUT = 0.974          # the standing library near-dup cut (steered_pilot_morph.STRICT_CUT)
BOOT = 10000
BOOT_SEED = 20260804        # seeded; no wall-clock input


# =========================================================================== #
# population
# =========================================================================== #
def load_rows() -> list[dict]:
    """The judged pairs, each flattened to what the reads need. Verdict export is authoritative
    for the verdict; the plan is authoritative for the endpoints (they were verified equal when
    the export was secured, so either serves the geometry)."""
    V = json.loads(VERDICTS.read_text(encoding="utf-8"))
    P = json.loads(PAIRS.read_text(encoding="utf-8"))
    plan = {p["pair_id"]: p for p in P["pairs"]}
    rows = []
    for x in V["verdicts"]:
        p = plan[x["pair_id"]]
        rows.append(dict(pair_id=x["pair_id"], band=x["band"], bin=x["bin"],
                         stratum=x["stratum"], partition=x["partition"],
                         verdict=x["verdict"], revealed=bool(x.get("revealed")),
                         d=x["d"], d_over_min=x["d_over_min"], d_over_max=x["d_over_max"],
                         fw_ratio=x["fw_ratio"], plan=p))
    return rows, V, P


# =========================================================================== #
# 1-2. the transition along d/min(fw)
# =========================================================================== #
def transition(rows: list[dict], key: str = "d_over_min", higher_is_distinct: bool = True) -> dict:
    """last-SAME / first-DISTINCT / interleave zone along `key`, with UNSURE reported separately.

    The zone is the closed interval [first-DISTINCT, last-SAME] when they cross (i.e. when a
    DISTINCT sits below a SAME); its width and its contents are the honest statement of where
    the boundary is NOT resolved. When they do not cross, the boundary is the open gap between
    them and `interleaved` is False."""
    same = sorted(r[key] for r in rows if r["verdict"] == "same")
    dist = sorted(r[key] for r in rows if r["verdict"] == "distinct")
    unsure = sorted(r[key] for r in rows if r["verdict"] == "unsure")
    if not same or not dist:
        return dict(n_same=len(same), n_distinct=len(dist), n_unsure=len(unsure),
                    last_same=(same[-1] if same else None),
                    first_distinct=(dist[0] if dist else None), interleaved=None)
    last_same, first_distinct = same[-1], dist[0]
    lo, hi = min(first_distinct, last_same), max(first_distinct, last_same)
    inside = lambda xs: [v for v in xs if lo <= v <= hi]        # noqa: E731
    interleaved = first_distinct < last_same
    return dict(
        n_same=len(same), n_distinct=len(dist), n_unsure=len(unsure),
        same_range=[same[0], same[-1]], distinct_range=[dist[0], dist[-1]],
        unsure_range=([unsure[0], unsure[-1]] if unsure else None),
        last_same=last_same, first_distinct=first_distinct,
        interleaved=interleaved,
        zone=[lo, hi], zone_width=hi - lo,
        zone_n_same=len(inside(same)), zone_n_distinct=len(inside(dist)),
        zone_n_unsure=len(inside(unsure)),
        # A clean gap (no interleave) is the interval any single cut could sit in.
        clean_interval=(None if interleaved else [last_same, first_distinct]),
    )


def cross_band(per_band: dict) -> dict:
    """Do the three bands admit ONE K? The intersection of the per-band intervals a cut may sit
    in: for an interleaved band that is the interleave zone (a cut inside it is wrong on some
    pair whatever it is), for a clean band the gap between last-SAME and first-DISTINCT."""
    iv = {}
    for b in BAND_ORDER:
        t = per_band.get(b)
        if not t or t.get("interleaved") is None:
            continue
        iv[b] = t["zone"] if t["interleaved"] else t["clean_interval"]
    if len(iv) < 2:
        return dict(intervals=iv, overlap=None)
    lo = max(v[0] for v in iv.values())
    hi = min(v[1] for v in iv.values())
    return dict(intervals=iv, overlap=([lo, hi] if lo <= hi else None),
                overlap_width=(hi - lo if lo <= hi else None))


# =========================================================================== #
# 4. fidelity — AUC with stratified-bootstrap CIs
# =========================================================================== #
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)) + 0.5 P(=), i.e. the Mann-Whitney statistic. `pos` is SAME."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def auc_boot(pos: np.ndarray, neg: np.ndarray, rng) -> np.ndarray:
    """Stratified bootstrap over the two verdict groups (the group sizes are the design here:
    14 SAME vs 108 DISTINCT is what Matt judged, not a sample of a larger judged set)."""
    out = np.empty(BOOT)
    for i in range(BOOT):
        out[i] = auc(pos[rng.integers(0, len(pos), len(pos))],
                     neg[rng.integers(0, len(neg), len(neg))])
    return out


def ci(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def best_split(pos: np.ndarray, neg: np.ndarray) -> dict:
    """The single threshold on this score that misclassifies fewest judged pairs (score oriented
    so higher = SAME). Reported as a descriptive minimum, not a fitted cut."""
    cands = np.unique(np.concatenate([pos, neg]))
    mids = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-9]])
    err = [(int((pos < t).sum() + (neg >= t).sum()), float(t)) for t in mids]
    n_err, t = min(err)
    return dict(threshold=t, n_wrong=n_err, n_total=len(pos) + len(neg),
                same_wrong=int((pos < t).sum()), distinct_wrong=int((neg >= t).sum()))


def fidelity(rows: list[dict], morph: dict) -> dict:
    """AUC of SAME-vs-DISTINCT for `-d/min(fw)` and for morph-cos, both oriented higher=SAME,
    with the paired bootstrap difference on the SAME resamples."""
    judged = [r for r in rows if r["verdict"] in ("same", "distinct")
              and (morph is None or r["pair_id"] in morph)]
    y = np.array([r["verdict"] == "same" for r in judged])
    s_geom = np.array([-r["d_over_min"] for r in judged])
    scores = {"neg_d_over_min": s_geom, "neg_d_over_max": np.array([-r["d_over_max"] for r in judged])}
    if morph is not None:
        scores["morph_cos"] = np.array([morph[r["pair_id"]] for r in judged])

    rng = np.random.default_rng(BOOT_SEED)
    ip, ineg = np.where(y)[0], np.where(~y)[0]
    # one shared index bootstrap so the AUC difference is PAIRED (same resampled pairs both ways)
    bi_p = rng.integers(0, len(ip), (BOOT, len(ip)))
    bi_n = rng.integers(0, len(ineg), (BOOT, len(ineg)))

    out = {"n_same": int(y.sum()), "n_distinct": int((~y).sum()), "per_score": {}}
    boots = {}
    for name, s in scores.items():
        p, n = s[ip], s[ineg]
        b = np.array([auc(p[bi_p[i]], n[bi_n[i]]) for i in range(BOOT)])
        boots[name] = b
        out["per_score"][name] = dict(auc=auc(p, n), ci95=ci(b), best_split=best_split(p, n))
    if "morph_cos" in boots:
        d = boots["neg_d_over_min"] - boots["morph_cos"]
        out["paired_diff_dmin_minus_morph"] = dict(
            point=out["per_score"]["neg_d_over_min"]["auc"] - out["per_score"]["morph_cos"]["auc"],
            ci95=ci(d), p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out


# =========================================================================== #
# morph-cos — the library morph_gray -> CLIP recipe, both sides of every pair
# =========================================================================== #
def morph_cos_all(rows: list[dict], force: bool, workers: int) -> dict:
    """{pair_id: cosine} over the library morph recipe. Field dumps are NOT retained (270 x 7 MB);
    the cosines are the cache."""
    if MORPH_CACHE.exists() and not force:
        cached = json.loads(MORPH_CACHE.read_text(encoding="utf-8"))
        if all(r["pair_id"] in cached["cos"] for r in rows):
            print(f"[morph] {len(cached['cos'])} cosines from cache {MORPH_CACHE.name}")
            return cached["cos"]

    import build_q4_harvest_batches as bq
    from tools.wallpaper.library_annotate import ensure_field, morph_gray_image
    from tools.curation.colored_clip import load_clip, embed_clip

    bq._PHOENIX_POOL_CACHE.update(bq._phoenix_points())
    tmp = OUT / "morph_fields"
    jobs = []
    for r in rows:
        for side in ("a", "b"):
            rend = sheet.render_block(r["plan"][side], r["plan"]["palette"])
            jobs.append((r["pair_id"], side, loc_mod.from_render_block(rend)))

    print(f"[morph] {len(jobs)} field dumps at 640x360 ss2 ({workers} engine processes)", flush=True)
    imgs: dict[tuple, object] = {}

    def one(j):
        pid, side, loc = j
        f = ensure_field(loc, retain=False, tmp_dir=tmp / f"{pid}.{side}", cache_root=tmp)
        return (pid, side), morph_gray_image(f)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            k, im = fut.result()
            imgs[k] = im
            if i % 50 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}]", flush=True)

    print("[morph] CLIP embedding (library morph_clip recipe) ...", flush=True)
    model, tf = load_clip()
    keys = list(imgs)
    E = embed_clip(model, tf, [imgs[k] for k in keys])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    idx = {k: i for i, k in enumerate(keys)}
    cos = {r["pair_id"]: float(E[idx[(r["pair_id"], "a")]] @ E[idx[(r["pair_id"], "b")]])
           for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)
    MORPH_CACHE.write_text(json.dumps(
        dict(recipe="library morph_gray 640x360 ss2 -> vit_base_patch16_clip_224.openai (768-d)",
             strict_cut=STRICT_CUT, n=len(cos), cos=cos), indent=1), encoding="utf-8")
    return cos


# =========================================================================== #
# report
# =========================================================================== #
def fmt(v, n=4):
    return "None" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-morph", action="store_true", help="skip reads 4-5 (no GPU/engine pass)")
    ap.add_argument("--force-morph", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="concurrent engine processes (<=4)")
    args = ap.parse_args()

    rows, V, P = load_rows()
    print(f"population: {len(rows)} judged pairs from {VERDICTS.relative_to(ROOT)} "
          f"(run {V['meta']['run']}, seed {V['meta']['seed']})")
    print(f"  verdicts  {dict(Counter(r['verdict'] for r in rows))}   "
          f"revealed-at-verdict {V['n_revealed_at_verdict']}")
    print(f"  production rule: d < {ps.DEDUP_K} * {ps.DEDUP_SCALE}(fw)  "
          f"(production_seeder.DEDUP_K/DEDUP_SCALE — READ, not moved). The min-scale variant "
          f"the sheet interrogates would cut at d/min(fw) = {ps.DEDUP_K}, the constant "
          f"transferred from the max rule.")

    res = {"population": dict(n=len(rows), run=V["meta"]["run"],
                             verdicts=dict(Counter(r["verdict"] for r in rows)),
                             dedup_k=ps.DEDUP_K, dedup_scale=V["meta"]["dedup_scale"])}

    # --- 1. per band -------------------------------------------------------- #
    print("\n=== 1. SAME->DISTINCT transition along d/min(fw), per fw-ratio band ===")
    per_band = {}
    for b in BAND_ORDER:
        rb = [r for r in rows if r["band"] == b]
        t = transition(rb)
        per_band[b] = t
        print(f"\n  band {BAND_LABEL[b]:>6}  n={len(rb)}  "
              f"(same {t['n_same']}, distinct {t['n_distinct']}, unsure {t['n_unsure']})")
        if t.get("interleaved") is None:
            print("    one-sided: no transition to read")
            continue
        print(f"    SAME     d/min in [{fmt(t['same_range'][0])}, {fmt(t['same_range'][1])}]"
              f"   last-SAME = {fmt(t['last_same'])}")
        print(f"    DISTINCT d/min in [{fmt(t['distinct_range'][0])}, {fmt(t['distinct_range'][1])}]"
              f"   first-DISTINCT = {fmt(t['first_distinct'])}")
        if t["unsure_range"]:
            print(f"    UNSURE   d/min in [{fmt(t['unsure_range'][0])}, "
                  f"{fmt(t['unsure_range'][1])}]  (n={t['n_unsure']}, never coerced)")
        if t["interleaved"]:
            print(f"    INTERLEAVED zone [{fmt(t['zone'][0])}, {fmt(t['zone'][1])}] "
                  f"width {fmt(t['zone_width'])}: {t['zone_n_same']} same, "
                  f"{t['zone_n_distinct']} distinct, {t['zone_n_unsure']} unsure inside")
        else:
            print(f"    CLEAN gap [{fmt(t['clean_interval'][0])}, {fmt(t['clean_interval'][1])}]"
                  f" — no interleaving")
    res["per_band"] = per_band
    res["all_bands"] = transition(rows)

    # --- 2. cross-band ------------------------------------------------------ #
    print("\n=== 2. cross-band read ===")
    cb = cross_band(per_band)
    res["cross_band"] = cb
    for b, v in cb["intervals"].items():
        kind = "interleave" if per_band[b]["interleaved"] else "clean gap"
        print(f"  {BAND_LABEL[b]:>6}: [{fmt(v[0])}, {fmt(v[1])}]  ({kind})")
    if cb["overlap"]:
        print(f"  OVERLAP: the three bands share [{fmt(cb['overlap'][0])}, "
              f"{fmt(cb['overlap'][1])}] — a single K is consistent with all three.")
    else:
        print("  NO shared interval — the bands do not admit one K on d/min(fw).")
    p = res["all_bands"]
    print(f"  pooled (all bands): last-SAME {fmt(p['last_same'])}, first-DISTINCT "
          f"{fmt(p['first_distinct'])}, zone [{fmt(p['zone'][0])}, {fmt(p['zone'][1])}] "
          f"holding {p['zone_n_same']} same / {p['zone_n_distinct']} distinct / "
          f"{p['zone_n_unsure']} unsure")

    # --- 2b. what a cut on d/min(fw) would do to the judged pairs ----------- #
    print("\n=== 2b. candidate cuts on d/min(fw): what each merges, on the judged pairs ===")
    dup = [r for r in rows if r["stratum"] == "dup"]
    dc = Counter(r["verdict"] for r in dup)
    print(f"  the dup stratum ({len(dup)} pairs) is what the CURRENT max-scale rule already "
          f"merged: {dict(dc)}")
    print(f"    -> production is today collapsing {dc['distinct']} pairs Matt calls DISTINCT "
          f"and {dc['unsure']} he is unsure about, per {dc['same']} it merges correctly "
          f"(judged sample of the run's {P['n_dup_pairs']} dup pairs, "
          f"d/min-stratified inside a {P['domain_cap']} cap).")
    ks = sorted({0.25, 0.5, 1.0, ps.DEDUP_K, 2.0,
                 round(res["all_bands"]["first_distinct"], 4)})
    sweep = []
    for k in ks:
        row = dict(k=k)
        for v in ("same", "distinct", "unsure"):
            row[f"merged_{v}"] = sum(1 for r in rows if r["verdict"] == v and r["d_over_min"] < k)
            row[f"n_{v}"] = sum(1 for r in rows if r["verdict"] == v)
        sweep.append(row)
        print(f"  K={k:<8g} merges {row['merged_same']}/{row['n_same']} SAME (kept work), "
              f"{row['merged_distinct']}/{row['n_distinct']} DISTINCT (FALSE MERGES), "
              f"{row['merged_unsure']}/{row['n_unsure']} UNSURE")
    res["cut_sweep"] = dict(dup_stratum_verdicts=dict(dc), sweep=sweep)

    # --- 3. anchors --------------------------------------------------------- #
    print("\n=== 3. anchors (attention check) ===")
    anc = [r for r in rows if r["stratum"] == "anchor"]
    ac = Counter(r["verdict"] for r in anc)
    bad = [r for r in anc if r["verdict"] != "distinct"]
    res["anchors"] = dict(n=len(anc), verdicts=dict(ac),
                          not_distinct=[dict(pair_id=r["pair_id"], band=r["band"],
                                             verdict=r["verdict"], d_over_min=r["d_over_min"],
                                             partition=r["partition"]) for r in bad])
    print(f"  {len(anc)} anchors: {dict(ac)}")
    for r in bad:
        print(f"    FLAG {r['pair_id']} band={r['band']} d/min={fmt(r['d_over_min'])} "
              f"{r['partition']}: judged {r['verdict'].upper()}, not DISTINCT")
    if not bad:
        print("  all anchors judged DISTINCT — attention check passes")

    # --- 4/5. morph ---------------------------------------------------------- #
    morph = None
    if not args.no_morph:
        morph = morph_cos_all(rows, args.force_morph, args.workers)

    print("\n=== 4. fidelity: d/min(fw) vs morph-cos on the judged pairs ===")
    fid = fidelity(rows, morph)
    res["fidelity"] = fid
    print(f"  n = {fid['n_same']} SAME vs {fid['n_distinct']} DISTINCT "
          f"(unsure excluded, not coerced)")
    for name, v in fid["per_score"].items():
        bs = v["best_split"]
        print(f"  AUC[{name:>14}] = {v['auc']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
              f"   best single split misses {bs['n_wrong']}/{bs['n_total']} "
              f"({bs['same_wrong']} same + {bs['distinct_wrong']} distinct)")
    if "paired_diff_dmin_minus_morph" in fid:
        d = fid["paired_diff_dmin_minus_morph"]
        print(f"  paired dAUC (d/min - morph) = {d['point']:+.3f}  95% CI "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  (p={d['p_two_sided']:.3f})")
    print("  CAVEAT: pairs were selected by rank-quantile strata OF d/min(fw) inside a 6.0 cap, "
          "so that axis is uniformized over the domain and morph-cos is not. This is a CHECK on "
          "the standing metric, not a re-derivation of it.")

    if morph is not None:
        print(f"\n=== 5. the {STRICT_CUT} cut in morph-cos space vs Matt's split ===")
        tm = transition([dict(r, mc=morph[r["pair_id"]]) for r in rows],
                        key="mc")
        res["morph_transition"] = tm
        print(f"  SAME     cos in [{fmt(tm['same_range'][0])}, {fmt(tm['same_range'][1])}]"
              f"   lowest-SAME = {fmt(tm['same_range'][0])}")
        print(f"  DISTINCT cos in [{fmt(tm['distinct_range'][0])}, {fmt(tm['distinct_range'][1])}]"
              f"   highest-DISTINCT = {fmt(tm['distinct_range'][1])}")
        # in morph space higher cos = more same, so the zone is [min SAME, max DISTINCT]
        lo, hi = tm["same_range"][0], tm["distinct_range"][1]
        overlap = lo <= hi
        where = ("inside Matt's INTERLEAVE zone" if overlap and lo <= STRICT_CUT <= hi
                 else "inside his SAME region (above every DISTINCT pair)" if STRICT_CUT > hi
                 else "inside his DISTINCT region (below every SAME pair)" if STRICT_CUT < lo
                 else "outside both")
        same_ge = sum(1 for r in rows if r["verdict"] == "same" and morph[r["pair_id"]] >= STRICT_CUT)
        dist_ge = sum(1 for r in rows if r["verdict"] == "distinct" and morph[r["pair_id"]] >= STRICT_CUT)
        uns_ge = sum(1 for r in rows if r["verdict"] == "unsure" and morph[r["pair_id"]] >= STRICT_CUT)
        n_same = sum(1 for r in rows if r["verdict"] == "same")
        n_dist = sum(1 for r in rows if r["verdict"] == "distinct")
        res["strict_cut_check"] = dict(cut=STRICT_CUT, where=where, overlap_zone=[lo, hi],
                                       same_ge=same_ge, n_same=n_same,
                                       distinct_ge=dist_ge, n_distinct=n_dist,
                                       unsure_ge=uns_ge)
        print(f"  Matt's morph-cos overlap zone: [{fmt(lo)}, {fmt(hi)}]  "
              f"(lowest SAME .. highest DISTINCT)")
        print(f"  cos >= {STRICT_CUT} is {where}")
        print(f"    it would call SAME:     {same_ge}/{n_same} of Matt's SAME pairs")
        print(f"    it would call SAME:     {dist_ge}/{n_dist} of Matt's DISTINCT pairs "
              f"(would-be false merges)")
        print(f"    and {uns_ge} of the {sum(1 for r in rows if r['verdict']=='unsure')} UNSURE")

    OUT.mkdir(parents=True, exist_ok=True)
    BOUNDARY.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"\n-> {BOUNDARY}")
    print("ADOPTS NOTHING: DEDUP_K / DEDUP_SCALE untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
