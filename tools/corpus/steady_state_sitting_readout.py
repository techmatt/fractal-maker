#!/usr/bin/env python
r"""steady_state_sitting_readout.py — read back the 2026-08-05 steady-state sitting.

Two registered legs (`..._steady_state_ranked_v1` 396 labeled of 654, `..._steady_state_dive_v1`
94 of 94) were served as ONE blind uncalibrated-partition sheet and routed home by
`merge_scores.py --route`. This reads what those 490 labels bought. Same three-stage shape as
`q4_combined_readout.py` (score / morph / readout) and the same statistics, imported from it
rather than re-derived — a second `wilson` in a second readout is a second answer waiting.

Read `docs/design/measurement_practice.md` §1 before changing anything here. Two of its rules
decide the shape of this file: "a control arm must differ from the treatment in ONE thing"
(the dive arms differ in three, so §[2] computes the arm x partition table rather than
asserting the contrast is clean), and "population-gate at the READER" (§[5]'s population is
stated in the output, not assumed from the batch name).

WHAT EACH STAGE COSTS, AND WHY THE MORPH STAGE IS FREE HERE
-----------------------------------------------------------
`score` decodes the 490 LABEL CROPS under the ACTIVE checkpoint through
`classifier.data.Transform(train=False)`. It is not read off provenance: `canon_pgood` is
P(>=3) alone, taken at harvest geometry under the enrich palette, and the staged derivation
below needs the full cutpoint triple on the tile Matt actually judged.

`morph` takes every vector from the WARM `morph_embed_cache` instead of re-rendering a field
per row: the sitting cutter's own dedup already embedded this exact population (measured
1837/1843 queue rows present), so the key is built the cutter's way —
`sitting_cutter.morph_key_of` off the run store's queue row — and joined to the corpus row on
`(cx, cy, fw)`, which is exactly how `build_q4_harvest_batches._render_block` wrote it. Using
the corpus render block's own location instead would MISS every row: the block carries
`maxiter = _maxiter_for_fw(fw)` while the cutter embedded at the auto-maxiter policy's value,
and maxiter is in the cache key. Measured 0/80 hits that way, 1837/1843 this way.

ONE-PER-CLUSTER IS REPORTED AT 0.95, NOT AT THE 0.974 NEAR-DUP CUT, and that is not a
loosening. The sitting cutter already ran a leader/radius dedup at 0.974 over this very
population, so every SERVED row is below 0.974 of every other by construction: at that cut
one-per-cluster is arithmetically identical to raw and says nothing. 0.95 is the same loose
cut `deferred_recalibration.md` reports its residue rate at.

  uv run python tools/corpus/steady_state_sitting_readout.py score
  uv run python tools/corpus/steady_state_sitting_readout.py morph
  uv run python tools/corpus/steady_state_sitting_readout.py readout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "tools"), str(HERE), str(ROOT / "tools" / "atlas"),
           str(ROOT / "tools" / "mining"), str(ROOT / "tools" / "scoring"),
           str(ROOT / "tools" / "sourcing"), str(ROOT / "tools" / "wallpaper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc          # noqa: E402
import label_store as ls            # noqa: E402
import paths                        # noqa: E402
from partitions import partition_of_row   # noqa: E402  THE row-aware resolver
from q4_combined_readout import wilson    # noqa: E402  ONE Wilson interval, not a second copy

SHEET = "2026-08-05_steady_state_uncal_sheet_v1"
RANKED = "2026-08-05_steady_state_ranked_v1"
DIVE = "2026-08-05_steady_state_dive_v1"
BATCHES = (RANKED, DIVE)

WORK = paths.scratch("steady_state_readout")
DECODE = WORK / "v10_decode_490.jsonl"
EMB = WORK / "morph_emb_490.npz"
STAGED = WORK / "t_good_STAGED_NOT_ADOPTED.json"
REPORT = WORK / "readout.txt"

# The five partitions the sheet's ranked leg was SCOPED to — every one stamped UNCALIBRATED in
# data/<ACTIVE_VERSION>/t_good_derivation.json when the sheet was cut. Read off the sheet's own
# batch.json rather than listed here, so this cannot drift from what was actually served.
def _sheet_partitions() -> set:
    sel = json.load(open(os.path.join(cc.batch_dir(SHEET), "batch.json"),
                         encoding="utf-8"))["selection"]
    return {p for p, s in sel["read"]["statuses"].items()
            if s == "UNCALIBRATED" and p in sel["partitions"]}


LOOSE = 0.95        # q4_combined_readout.LOOSE — the cut one-per-cluster is taken at here
STRICT = 0.974      # the near-dup cut the SITTING WAS ALREADY DEDUPED AT (degenerate here)


# =========================================================================== #
# shared
# =========================================================================== #
def routed_ids() -> dict:
    """`{(batch, image_id): opaque_id}` for the 490 rows this sitting's sheet actually served.

    The batches hold more than the sheet did (654 ranked rows, 396 served), so every read here
    is scoped by the ROUTE MAP rather than by "has a label" — the two happen to agree today and
    would silently stop agreeing the day a later sheet labels the excluded 258."""
    route = json.load(open(os.path.join(cc.batch_dir(SHEET), "route.json"), encoding="utf-8"))
    return {(v["batch"], v["image_id"]): k for k, v in route.items()}


def load_labeled() -> list[dict]:
    """The 490 served rows with their labels resolved through the amendment overlay.

    `resolve_score` is called WITH amendments: item 3 asks for positives "through the
    amendment overlay", and a revision that never reached the original file is exactly what
    that overlay is for."""
    served = routed_ids()
    out = []
    for b in BATCHES:
        side, amend = ls.sidecar_for(b), ls.amendments_for(b)
        for r in cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")):
            key = (b, r["image_id"])
            if key not in served:
                continue
            pv = r.get("provenance") or {}
            out.append(dict(batch=b, image_id=r["image_id"], opaque=served[key],
                            render=r["render"],
                            partition=partition_of_row(r["render"], pv.get("family")),
                            prov_family=pv.get("family"),
                            human=ls.resolve_score(r, side, amend),
                            original=ls.resolve_score(r, side),
                            mix_source=pv.get("mix_source"), fate=pv.get("fate"),
                            rank_tier=pv.get("rank_tier"), queue_rank=pv.get("queue_rank"),
                            canon_pgood=pv.get("canon_pgood"),
                            canon_decoded=pv.get("canon_decoded"), depth=pv.get("depth")))
    return out


def coord_key(cx, cy, fw) -> str:
    """The join between a run-store queue row and the corpus row built from it.

    `build_q4_harvest_batches._render_block` writes `cx = str(r["cx"])`, `cy = str(r["cy"])`
    and `fw = cc.hp_str(r["fw"])`, so reproducing those three spellings IS the join. Not
    image_id: the sitting's ids are minted post-shuffle and the run store never sees them."""
    return f"{cx}|{cy}|{fw}"


def queue_index() -> tuple[dict, dict]:
    """`({coord_key: queue_row}, join_report)` for the sitting's whole union queue."""
    import sitting_cutter as sc
    rows, rep = sc.load_union_queue(sc.STEADY_STATE_SITTING)
    idx = {}
    for r in rows:
        idx[coord_key(str(r["cx"]), str(r["cy"]), cc.hp_str(r["fw"]))] = r
    return idx, rep


# =========================================================================== #
# stage: score  — the ACTIVE checkpoint over the 490 label crops
# =========================================================================== #
def stage_score(args) -> int:
    import torch
    from PIL import Image
    from classifier.inference import load_scorer
    from production_pins import ACTIVE_CKPT, ACTIVE_VERSION

    rows = load_labeled()
    sc = load_scorer(str(ROOT / ACTIVE_CKPT))
    K = sc.config.get("num_classes")
    print(f"{ACTIVE_CKPT} ({ACTIVE_VERSION}) on {sc.device}: target={sc.target} K={K}")
    if K != 4:
        raise SystemExit(f"K={K}: this readout stages a K=4 derivation (p_ge2/3/4) and a "
                         f"K=3 head cannot supply the third cutpoint")
    WORK.mkdir(parents=True, exist_ok=True)
    out, buf, meta = [], [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            probs = torch.sigmoid(sc.model(torch.stack(buf).to(sc.device)).float().cpu())
        for m, pr in zip(meta, probs.tolist()):
            m.update({f"{ACTIVE_VERSION}_p_ge2": pr[0], f"{ACTIVE_VERSION}_p_ge3": pr[1],
                      f"{ACTIVE_VERSION}_p_ge4": pr[2],
                      f"{ACTIVE_VERSION}_score": pr[0] + pr[1] + pr[2]})
            out.append(m)
        buf.clear(); meta.clear()

    for r in rows:
        with Image.open(os.path.join(cc.crops_dir(r["batch"]), f"{r['image_id']}.jpg")) as im:
            im.load()
            buf.append(sc.transform(im.convert("RGB")))
        meta.append({k: v for k, v in r.items() if k != "render"}
                    | {"fractal_type": r["render"]["fractal_type"],
                       "coord": coord_key(r["render"]["cx"], r["render"]["cy"],
                                          r["render"]["fw"])})
        if len(buf) == 32:
            flush()
    flush()
    with open(DECODE, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {DECODE}: {len(out)} rows, ckpt {ACTIVE_CKPT}")
    return 0


# =========================================================================== #
# stage: morph — vectors OUT OF THE WARM CACHE, never re-rendered
# =========================================================================== #
def stage_morph(args) -> int:
    import sitting_cutter as sc
    from tools.wallpaper import morph_embed_cache as mec

    rows = load_labeled()
    idx, qrep = queue_index()
    cache = mec.MorphEmbedCache().open()
    keys, embs, missing = [], [], []
    for r in rows:
        ck = coord_key(r["render"]["cx"], r["render"]["cy"], r["render"]["fw"])
        q = idx.get(ck)
        if q is None:
            missing.append((r["image_id"], "no queue row on the coordinate join"))
            continue
        e = cache.get(sc.morph_key_of(q))
        if e is None:
            missing.append((r["image_id"], "queue row joined but not in the morph cache"))
            continue
        keys.append(f"{r['batch']}|{r['image_id']}")
        embs.append(np.asarray(e, dtype=np.float32))
    cache.close()
    E = np.stack(embs) if embs else np.zeros((0, 768), np.float32)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB, keys=np.array(keys), emb=E,
                        source="morph_embed_cache (warm), keyed via sitting_cutter.morph_key_of")
    print(f"wrote {EMB}: {E.shape}; {len(missing)} of {len(rows)} unresolved")
    if missing:      # whole, never a head — CLAUDE.md's truncated-error-log rule
        (WORK / "morph_missing.json").write_text(json.dumps(missing, indent=1), encoding="utf-8")
        print("  reasons:", Counter(m[1] for m in missing).most_common())
    print(f"  queue: {qrep['n']} rows {json.dumps(qrep['by_leg'])}")
    return 0


# =========================================================================== #
# stage: readout
# =========================================================================== #
def leader_clusters(C, N, cut):
    """Leader/radius clusters at `cut` — cannot chain, unlike single linkage."""
    lead, assign = [], {}
    for i in range(N):
        for li, l in enumerate(lead):
            if C[i, l] >= cut:
                assign[i] = li
                break
        else:
            assign[i] = len(lead)
            lead.append(i)
    return assign


def dive_units() -> list[dict]:
    """Per-DIVE >=3 counts for the dive leg — the arm's independent unit, not the rung.

    The dive identity is not on the corpus row (`_provenance` stamps `mix_source` but not
    `root_id`), so it is recovered the same CHECKED way the sitting cutter recovered the arm:
    `recover_dive_arms` establishes root_id -> dive by APPEND order with a partition check, and
    the corpus row joins back to its queue row on the coordinate key. If that join or that
    recovery fails, this returns nothing rather than a guessed grouping."""
    import sitting_cutter as sc
    leg = sc.STEADY_STATE_SITTING.leg(DIVE)
    run = ROOT / leg.run_dir
    arms, why = sc.recover_dive_arms(run / "q4_candidates.jsonl", run / leg.dive_log)
    if not arms:
        print(f"  per-dive unavailable: {why}")
        return []
    log = [json.loads(l) for l in (run / leg.dive_log).read_text(encoding="utf-8").splitlines()
           if l.strip()]
    root_to_dive = {rid: rec["dive_id"] for rid, rec in zip(arms, log)}
    idx, _ = queue_index()
    rows = [json.loads(l) for l in open(DECODE, encoding="utf-8") if l.strip()]
    g = defaultdict(list)
    for r in rows:
        if r["batch"] != DIVE:
            continue
        q = idx.get(r["coord"])
        rid = (q or {}).get("root_id")
        if rid in root_to_dive:
            g[root_to_dive[rid]].append(r)
    by_id = {rec["dive_id"]: rec for rec in log}
    return [dict(dive_id=d, arm=f"dive:{by_id[d]['start_group']}",
                 partition=by_id[d]["partition"], n=len(v),
                 k=sum(1 for r in v if r["human"] >= 3))
            for d, v in sorted(g.items())]


def stage_readout(args) -> int:
    import derive_t_good as est
    from production_pins import ACTIVE_VERSION
    SHEET_PARTITIONS = _sheet_partitions()
    # The ACTIVE version's own objective map, imported from its deriver rather than restated:
    # the objective is a supply read that was re-argued at the flip, and a staged table that
    # quietly picked a different beta would not be comparable to anything.
    sys.path.insert(0, str(ROOT / "tools" / ACTIVE_VERSION))
    v10 = __import__(f"derive_t_good_{ACTIVE_VERSION}")

    rows = [json.loads(l) for l in open(DECODE, encoding="utf-8")]
    Z = np.load(EMB, allow_pickle=True)
    U = Z["emb"].astype(np.float32)
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    C = U @ U.T
    ekeys = {str(k): i for i, k in enumerate(Z["keys"])}
    assign = leader_clusters(C, len(ekeys), LOOSE)
    assign_strict = leader_clusters(C, len(ekeys), STRICT)
    for r in rows:
        i = ekeys.get(f"{r['batch']}|{r['image_id']}")
        r["cluster"] = None if i is None else assign[i]
        r["cluster_strict"] = None if i is None else assign_strict[i]

    L = []

    def w(s=""):
        L.append(s); print(s)

    def rate(sub, pred):
        """(k, n, p, lo, hi, one_per_cluster_rate, n_clusters)."""
        n = len(sub)
        k = sum(1 for r in sub if pred(r))
        p, lo, hi = wilson(k, n)
        g = defaultdict(list)
        for r in sub:
            g[r["cluster"] if r["cluster"] is not None else ("solo", r["image_id"])].append(r)
        opc = (sum(sum(1 for r in v if pred(r)) / len(v) for v in g.values()) / len(g)
               if g else float("nan"))
        return k, n, p, lo, hi, opc, len(g)

    def dist(sub):
        c = Counter(r["human"] for r in sub)
        return " ".join(f"{s}:{c.get(s, 0)}" for s in (1, 2, 3, 4))

    w(f"=== steady-state sitting readout — {len(rows)} labeled rows, sheet {SHEET} ===")
    w(f"    clusters: leader/radius on the library morph CLIP recipe. {len(set(assign.values()))}"
      f" at {LOOSE}, {len(set(assign_strict.values()))} at {STRICT} of {len(ekeys)} embedded"
      f" — the {STRICT} count equalling the row count is the sitting cutter's own dedup, so"
      f" one-per-cluster is reported at {LOOSE}.")

    # ---------------- [2] dive effect size ----------------
    w("\n[2] DIVE LEG — top vs control on the recovered arms. HUMAN, DESCRIPTIVE, NO BAR.")
    w("    measurement_practice.md §1 'Contrasts and confounds': a control arm must differ")
    w("    from the treatment in ONE thing, and a search that chooses where it goes confounds")
    w("    its own axes by construction. The arm x partition table below is that check.")
    dive = [r for r in rows if r["batch"] == DIVE]
    arms = {a: [r for r in dive if r["mix_source"] == a]
            for a in ("dive:top", "dive:control")}
    for a, sub in arms.items():
        k, n, p, lo, hi, opc, nc = rate(sub, lambda r: r["human"] >= 3)
        k4, _, p4, lo4, hi4, opc4, _ = rate(sub, lambda r: r["human"] == 4)
        w(f"    {a:14s} n={n:3d}  >=3 {k:3d}/{n}={p:6.1%} [{lo:.1%},{hi:.1%}]  "
          f"1pc {opc:6.1%} over {nc} clusters   =4 {k4}/{n}={p4:5.1%} 1pc {opc4:5.1%}")
        w(f"    {'':14s} classes {dist(sub)}   partitions "
          f"{dict(Counter(r['partition'] for r in sub))}")
    # THE CONFOUND, computed rather than asserted.
    w("    arm x partition (the contrast's ceiling):")
    tab = defaultdict(Counter)
    for r in dive:
        tab[r["partition"]][r["mix_source"]] += 1
    for p_ in sorted(tab):
        w(f"      {p_:20s} {dict(tab[p_])}")
    overlap = [p_ for p_ in tab if len(tab[p_]) > 1]
    w(f"    partitions carrying BOTH arms: {overlap or 'NONE'}")
    # THE DESIGN UNIT IS THE DIVE, NOT THE RUNG. A descent's rungs are the same descent
    # re-measured, so 39 vs 55 rows are not 39 vs 55 draws; the number of independent draws
    # is the number of dives. Reported because a Wilson interval on the rungs reads ~3x
    # tighter than the design supports and nothing else in this readout would say so.
    per_dive = dive_units()
    if per_dive:
        w("    PER-DIVE (the independent unit — a descent's rungs are one draw re-measured):")
        for d in per_dive:
            w(f"      {d['dive_id']} {d['arm']:12s} {d['partition']:18s} "
              f">=3 {d['k']:2d}/{d['n']:2d}={d['k']/d['n']:6.1%}")
        for a in ("dive:top", "dive:control"):
            ds = [d for d in per_dive if d["arm"] == a]
            if ds:
                m = [d["k"] / d["n"] for d in ds]
                w(f"      {a:12s} {len(ds)} dives, per-dive >=3 rate "
                  f"{'/'.join(f'{x:.0%}' for x in m)} (mean {sum(m)/len(m):.1%})")

    # The ONE partition-matched anchor available: multibrot3 is the top arm's only partition
    # and it also has a ranked-leg slice, so the dive's own yield can be read against the same
    # family's non-dive residue from the same run. There is no such anchor for the control arm.
    mb3_r = [r for r in rows if r["batch"] == RANKED and r["partition"] == "multibrot3"]
    mb3_d = [r for r in rows if r["batch"] == DIVE and r["partition"] == "multibrot3"]
    for tag, sub in (("dive top arm", mb3_d), ("ranked residue", mb3_r)):
        k, n, p, lo, hi, opc, nc = rate(sub, lambda r: r["human"] >= 3)
        w(f"    multibrot3 anchor — {tag:15s} {k:3d}/{n}={p:6.1%} [{lo:.1%},{hi:.1%}] "
          f"1pc {opc:6.1%} ({nc} cl)")

    # ---------------- [3] what the labels bought for calibration ----------------
    w("\n[3] CORPUS POSITIVES (human >=3, amendment overlay) vs MIN_POS — before/after.")
    w("    `before` is DERIVED as after minus this sitting's own positives, not restated from")
    w("    a pre-merge snapshot: the census is a live read and a frozen copy would rot.")
    import sitting_cutter as sc
    after = sc.positives_census()
    added = Counter(r["partition"] for r in rows if (r["human"] or 0) >= 3)
    w(f"    {'partition':20s} {'before':>7} {'+sitting':>9} {'after':>7}  vs MIN_POS={est.MIN_POS}")
    for p_ in sorted(after, key=lambda x: (x not in SHEET_PARTITIONS, x)):
        a, d = after[p_], added.get(p_, 0)
        mark = "sheet" if p_ in SHEET_PARTITIONS else ""
        w(f"    {p_:20s} {a - d:>7} {d:>9} {a:>7}  "
          f"{'CLEARS' if a >= est.MIN_POS else 'BELOW ':6s} {mark}")

    # ---------------- [5] descriptive yield on the ranked leg ----------------
    w("\n[5] RANKED LEG — per-partition human >=3 rate. POPULATION: model-selected ranked")
    w("    residue, TRAIN-side, biased at the screen and at the rank. Descriptive; never eval.")
    ranked = [r for r in rows if r["batch"] == RANKED]
    for p_ in sorted({r["partition"] for r in ranked}):
        sub = [r for r in ranked if r["partition"] == p_]
        k, n, pr, lo, hi, opc, nc = rate(sub, lambda r: r["human"] >= 3)
        k4, _, p4, _, _, opc4, _ = rate(sub, lambda r: r["human"] == 4)
        w(f"    {p_:20s} n={n:3d}  >=3 {k:3d}/{n}={pr:6.1%} [{lo:.1%},{hi:.1%}] 1pc {opc:6.1%}"
          f" ({nc} cl)   =4 {k4:2d}={p4:5.1%} 1pc {opc4:5.1%}   classes {dist(sub)}")
    k, n, pr, lo, hi, opc, nc = rate(ranked, lambda r: r["human"] >= 3)
    w(f"    {'ALL (mix-weighted)':20s} n={n:3d}  >=3 {k:3d}/{n}={pr:6.1%} [{lo:.1%},{hi:.1%}] "
      f"1pc {opc:6.1%} ({nc} cl)")

    # ---------------- [4] STAGED derivation ----------------
    w("\n[4] STAGED t_good — BUILT, NOT ADOPTED. Population is THIS SITTING'S 490 labels,")
    w("    which are TRAIN-side and biased; the live table is derived on the frozen UNBIASED")
    w("    eval slice and this merge added nothing to it. These are not comparable numbers.")
    srows = [dict(fractal_type=r["fractal_type"], label=r["human"],
                  **{k: r[k] for k in (f"{ACTIVE_VERSION}_p_ge2", f"{ACTIVE_VERSION}_p_ge3",
                                       f"{ACTIVE_VERSION}_p_ge4")})
             for r in rows if r["human"] is not None]
    # `build_table` PRINTS the table (t, precision, recall, F, F_OOF, plateau per partition)
    # and returns only the structured form. Captured into the report so the readout file is
    # the whole read and not a pointer to a terminal nobody kept.
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        table = est.build_table(srows, version=ACTIVE_VERSION,
                                eval_slice=f"STAGED — {SHEET} (490 train-side labels), NOT an "
                                           f"eval slice",
                                objective=v10.OBJECTIVE)
    for ln in buf.getvalue().rstrip("\n").splitlines():
        w(ln)
    # THE STAMP IS STRUCTURAL, NOT A COMMENT. `build_table` emits an `adopted` block and a
    # `note` that says production_seeder mirrors it — true of a real derivation and false of
    # this one. A reader that takes `doc["adopted"]` (the mirror check, the vs-previous
    # comparison in every per-version deriver) would read staged numbers and go green, so the
    # key is RENAMED: those readers now KeyError, which is the only failure mode that cannot
    # be missed. Same reason the file is named ..._STAGED_NOT_ADOPTED.json.
    table["would_be_cut"] = table.pop("adopted")
    table["note"] = ("STAGED. `adopted` is deliberately absent — see would_be_cut and "
                     "STAGED_NOT_ADOPTED below. Nothing mirrors this file.")
    table["STAGED_NOT_ADOPTED"] = (
        "THIS IS NOT A t_good TABLE. It is derived on 490 TRAIN-side, model-selected labels "
        "from the 2026-08-05 steady-state sitting, not on an unbiased eval instrument. The "
        f"adopted table is data/{ACTIVE_VERSION}/t_good_derivation.json and this merge did "
        "not move it. Adoption is Matt's separate decision and would additionally require an "
        "unbiased instrument for these partitions, which does not exist.")
    table["population"] = dict(sheet=SHEET, batches=list(BATCHES), n=len(srows),
                               registration="both legs biased=True -> train (batch_registry)")
    WORK.mkdir(parents=True, exist_ok=True)
    STAGED.write_text(json.dumps(table, indent=2), encoding="utf-8")
    w(f"    wrote {STAGED} (scratch, stamped STAGED_NOT_ADOPTED)")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="stage", required=True)
    for name, fn in (("score", stage_score), ("morph", stage_morph),
                     ("readout", stage_readout)):
        sp.add_parser(name).set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
