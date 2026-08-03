#!/usr/bin/env python
r"""q4_combined_readout.py — the readout of the 2026-08-03 q4 combined labeling sitting.

Three registered batches were labeled as ONE blind sitting of 870 rows
(`build_combined_label_sheet.py`), routed home by `merge_scores.py --route`, and this reads
them back. It lives in `tools/` rather than in a scratch script because it is the ONLY
producer of three DURABLE artifacts (`CLAUDE.md`, "Neither scratch tree is a dependency
tier"):

  data/label_corpus/eval_instruments/q4_uniform_eval_v1.json   per-family human base rates
  data/label_corpus/motif/q4_combined_2026-08-03_clusters.jsonl morph cluster assignment
  data/discovery/q4_long_harvest_20260803/human_q3plus_queue.jsonl  the ranked >=3 residue

THREE STAGES, because two of them are expensive and neither should be re-paid to re-read a
number. Each writes its intermediate under `scratch/q4_readout/` (regenerable), and `readout`
reads whatever is there:

  score    v10-decode all 870 LABEL CROPS through classifier.data.Transform(train=False).
           Not reused from provenance: the uniform-eval leg was drawn with no screen and
           carries no decode at all, and the two screened legs recorded decodes taken at
           harvest geometry under the enrich palette, not on the tile Matt actually judged.
  morph    dump each row's smooth field at the morphology geometry and embed it with the
           library morph_clip recipe (see `steered_pilot_morph`), for the saturation read.
  readout  items 1 and 3-6 + the three durable records.

TWO CLUSTERINGS, ON PURPOSE. Single-linkage at cos>=0.974 is the library's established
near-dup cut, but at this density it CHAINS: the biggest component's minimum intra-pair
cosine is 0.857, far under the cut, so its size is not a count of mutually-identical tiles.
Leader/radius clustering at the same cut cannot chain (every member is within the cut of its
leader) and is reported beside it. They fail in opposite directions and the report quotes
both; their SHARED blind spot is that morph_clip is palette-blind and 640x360-bound, so tiles
differing only in colour or in sub-pixel detail read as one look.

  uv run python tools/corpus/q4_combined_readout.py score
  uv run python tools/corpus/q4_combined_readout.py morph
  uv run python tools/corpus/q4_combined_readout.py readout [--apply]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "tools"), str(HERE), str(ROOT / "tools" / "mining"),
           str(ROOT / "tools" / "scoring"), str(ROOT / "tools" / "wallpaper"),
           str(ROOT / "tools" / "curation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc      # noqa: E402
import label_store as ls        # noqa: E402
import paths                    # noqa: E402

SITTING = "2026-08-03_q4_combined_label_v1"
RUN_DIR = "data/discovery/q4_long_harvest_20260803"
BATCHES = ["2026-08-03_q4_harvest_ranked_v1", "2026-08-03_q4_near_minibrot_v1",
           "2026-08-03_q4_uniform_eval_v1"]
EVAL_BATCH = "2026-08-03_q4_uniform_eval_v1"
CKPT = "data/classifier/v10/model_best.pt"

WORK = paths.scratch("q4_readout")
DECODE = WORK / "v10_decode_870.jsonl"
EMB = WORK / "morph_emb_870.npz"
FIELDS = WORK / "morph_fields"

STRICT, LOOSE = 0.974, 0.95     # steered_pilot_morph.STRICT_CUT / LOOSE_CUT
WORKERS = 4                     # PROCESS cap (CLAUDE.md); 3 engine threads each on 12 cores
ENGINE_THREADS = "3"


# =========================================================================== #
# shared
# =========================================================================== #
def load_rows():
    """[(batch, row)] for all three source batches, store order."""
    out = []
    for b in BATCHES:
        for r in cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")):
            out.append((b, r))
    return out


def wilson(k, n, z=1.96):
    """Wilson score interval. Wald is wrong at k=0 and several cells here ARE 0/48."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# =========================================================================== #
# stage: score
# =========================================================================== #
def stage_score(args) -> int:
    import torch
    from PIL import Image
    from classifier.inference import load_scorer
    from score_lib import corn_decode

    t_good = json.load(open(os.path.join(RUN_DIR, "run_config.json"),
                            encoding="utf-8"))["t_good"]
    sc = load_scorer(CKPT)
    print(f"{CKPT} on {sc.device}: target={sc.target} K={sc.config.get('num_classes')}")
    WORK.mkdir(parents=True, exist_ok=True)
    out, buf, meta = [], [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            probs = torch.sigmoid(sc.model(torch.stack(buf).to(sc.device)).float().cpu())
        for m, pr in zip(meta, probs.tolist()):
            nb, g, gr = pr[0], pr[1], pr[2]
            m.update(p_notbad=nb, p_good=g, p_ge4=gr, eord=nb + g + gr,
                     decoded_run=corn_decode(nb, g, t_good[m["partition"]], gr),
                     decoded_05=corn_decode(nb, g, 0.5, gr))
            out.append(m)
        buf.clear(); meta.clear()

    for b, r in load_rows():
        side, amend = ls.sidecar_for(b), ls.amendments_for(b)
        iid = r["image_id"]
        with Image.open(os.path.join(cc.crops_dir(b), f"{iid}.jpg")) as im:
            im.load()
            buf.append(sc.transform(im.convert("RGB")))
        pv = r.get("provenance") or {}
        meta.append(dict(
            batch=b, image_id=iid, partition=pv.get("family"),
            human=ls.resolve_score(r, side, amend), stratum=pv.get("stratum"),
            triggered=pv.get("triggered"), mix_source=pv.get("mix_source"),
            fate=pv.get("fate"), branch=pv.get("branch"), ladder_rung=pv.get("ladder_rung"),
            atom_id=pv.get("atom_id"), atom_period=pv.get("atom_period"),
            rank_tier=pv.get("rank_tier"), queue_rank=pv.get("queue_rank"),
            prov_decode=next((pv[k] for k in ("decoded_class", "reframe_decoded",
                                              "canon_decoded") if pv.get(k) is not None), None)))
        if len(buf) == 32:
            flush()
    flush()
    with open(DECODE, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {DECODE}: {len(out)} rows")
    return 0


# =========================================================================== #
# stage: morph
# =========================================================================== #
def _dump_field(task):
    b, iid, render = task
    sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
                    str(ROOT / "tools" / "wallpaper")]
    os.chdir(ROOT)
    os.environ["RAYON_NUM_THREADS"] = ENGINE_THREADS
    from tools.wallpaper import library_annotate as la
    try:
        la.ensure_field(la.render_location(render), retain=True,
                        tmp_dir=FIELDS, cache_root=FIELDS)
        return f"{b}|{iid}", True, ""
    except Exception as e:                                    # noqa: BLE001
        return f"{b}|{iid}", False, f"{type(e).__name__}: {str(e)[:200]}"


def stage_morph(args) -> int:
    from tools import colormap as cmap
    from tools.wallpaper import library_annotate as la
    from tools.wallpaper import library_store as store
    from tools.curation.colored_clip import load_clip, embed_clip

    FIELDS.mkdir(parents=True, exist_ok=True)
    tasks = [(b, r["image_id"], r["render"]) for b, r in load_rows()]
    print(f"{len(tasks)} rows; {WORKERS} processes x {ENGINE_THREADS} engine threads")
    t0, errs, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for key, ok, err in ex.map(_dump_field, tasks, chunksize=4):
            done += 1
            if not ok:
                errs.append((key, err))
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} {el:.0f}s ({el/done:.2f}s/row) "
                      f"{len(errs)} failed", flush=True)
    print(f"fields: {time.time()-t0:.0f}s, {len(errs)} failures")
    if errs:   # persisted whole; never characterized from a head
        (WORK / "morph_field_errors.json").write_text(json.dumps(errs, indent=1),
                                                      encoding="utf-8")
        print("  classes:", Counter(e.split(":")[0] for _, e in errs).most_common())

    model, tf = load_clip()
    keys, embs, missing = [], [], []
    for i, (b, iid, render) in enumerate(tasks):
        stem = store.field_stem(la.render_location(render), "smooth", la.W, la.H, la.SS)
        binp, jsonp = FIELDS / f"{stem}.bin", FIELDS / f"{stem}.json"
        if not (binp.exists() and jsonp.exists()):
            missing.append(f"{b}|{iid}")
            continue
        img = la.morph_gray_image(cmap.load_field(str(binp), str(jsonp)))
        embs.append(embed_clip(model, tf, [img])[0].astype(np.float32))
        keys.append(f"{b}|{iid}")
        if (i + 1) % 200 == 0:
            print(f"  embedded {i+1}/{len(tasks)}", flush=True)
    E = np.stack(embs) if embs else np.zeros((0, 768), np.float32)
    np.savez_compressed(EMB, keys=np.array(keys), emb=E, producer=la.MORPH_PRODUCER,
                        geometry=f"{la.W}x{la.H}ss{la.SS}", clip_model="vit_base_patch16_clip_224.openai")
    print(f"wrote {EMB}: {E.shape}, {len(missing)} missing")
    if missing:
        (WORK / "morph_emb_missing.json").write_text(json.dumps(missing, indent=1),
                                                     encoding="utf-8")
    return 0


# =========================================================================== #
# stage: readout
# =========================================================================== #
def _clusters(C, N, cut):
    """(single-linkage components, leader/radius clusters), both at `cut`."""
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    I, J = np.where(np.triu(C >= cut, 1))
    for i, j in zip(I.tolist(), J.tolist()):
        parent[find(i)] = find(j)
    g = defaultdict(list)
    for i in range(N):
        g[find(i)].append(i)
    link = sorted(g.values(), key=len, reverse=True)

    lead, assign = [], {}
    for i in range(N):
        for li, l in enumerate(lead):
            if C[i, l] >= cut:
                assign[i] = li
                break
        else:
            assign[i] = len(lead); lead.append(i)
    g2 = defaultdict(list)
    for i, a in assign.items():
        g2[a].append(i)
    return link, sorted(g2.values(), key=len, reverse=True)


def stage_readout(args) -> int:
    rows = [json.loads(l) for l in open(DECODE, encoding="utf-8")]
    by_key = {f"{r['batch']}|{r['image_id']}": r for r in rows}
    Z = np.load(EMB, allow_pickle=True)
    keys = [str(k) for k in Z["keys"]]
    U = Z["emb"].astype(np.float32)
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    C = U @ U.T
    N = len(keys)
    erows = [by_key[k] for k in keys]
    link, ball = _clusters(C, N, STRICT)
    cid = {i: k for k, c in enumerate(link) for i in c}
    lid = {i: k for k, c in enumerate(ball) for i in c}

    L = []

    def w(s=""):
        L.append(s); print(s)

    def rate(sub, pred):
        """(k, n, wilson_lo, wilson_hi, link-weighted, ball-weighted, n_looks_ball)."""
        n = len(sub)
        k = sum(1 for i in sub if pred(erows[i]))
        p, lo, hi = wilson(k, n)
        outs = []
        for which in (cid, lid):
            g = defaultdict(list)
            for i in sub:
                g[which[i]].append(i)
            outs.append((sum(sum(1 for i in v if pred(erows[i])) / len(v)
                             for v in g.values()) / len(g), len(g)))
        return k, n, p, lo, hi, outs[0], outs[1]

    def line(tag, sub):
        out = [f"    {tag:<34s}"]
        for nm, pred in (("ge2", lambda r: r["human"] >= 2), ("ge3", lambda r: r["human"] >= 3),
                         ("eq4", lambda r: r["human"] == 4)):
            k, n, p, lo, hi, lk, bl = rate(sub, pred)
            out.append(f"{nm} {k:3d}/{n:3d}={p:5.1%}[{lo:.0%},{hi:.0%}] "
                       f"1pc-link {lk[0]:5.1%}({lk[1]:3d}) 1pc-ball {bl[0]:5.1%}({bl[1]:3d})")
        return "  ".join(out)

    IDX = defaultdict(list)
    for i, r in enumerate(erows):
        IDX[("b", r["batch"])].append(i)
        IDX[("s", r["stratum"])].append(i)
        IDX[("bf", r["batch"], r["partition"])].append(i)

    w(f"=== q4 combined sitting readout — {N} labeled rows, {SITTING} ===")
    w(f"    morph producer {Z['producer']} @ {Z['geometry']}, clip {Z['clip_model']}")

    # ---------------- item 1 ----------------
    w("\n[1] PER-FAMILY BASE RATES — uniform eval leg (registered eval/unbiased)")
    w("    population: no viability screen, no classifier screen; near-dM_d shell eps=0.02")
    w("    (native multibrots + julia:mandelbrot), phoenix skeleton closed form.")
    inst = {}
    for fam in sorted({r["partition"] for r in erows if r["batch"] == EVAL_BATCH}):
        sub = IDX[("bf", EVAL_BATCH, fam)]
        w(line(fam, sub))
        inst[fam] = {}
        for nm, pred in (("ge2", lambda r: r["human"] >= 2), ("ge3", lambda r: r["human"] >= 3),
                         ("eq4", lambda r: r["human"] == 4)):
            k, n, p, lo, hi, lk, bl = rate(sub, pred)
            inst[fam][nm] = dict(k=k, n=n, rate=p, wilson95=[lo, hi],
                                 rate_one_per_look_linkage=lk[0], n_looks_linkage=lk[1],
                                 rate_one_per_look_ball=bl[0], n_looks_ball=bl[1])

    # ---------------- item 3 ----------------
    w("\n[3] NEAR-MINIBROT LADDER by rung (all julia:mandelbrot, atom-radius multiples)")
    for st in ("rung1", "rung4", "rung16"):
        w(line(st, IDX[("s", st)]))
    w(line("ladder ALL", IDX[("b", BATCHES[1])]))
    w("    reference (NOT pooled with the ladder):")
    w(line("harvest_ranked julia:mandelbrot", IDX[("bf", BATCHES[0], "julia:mandelbrot")]))
    w(line("uniform_eval julia:mandelbrot", IDX[("bf", BATCHES[2], "julia:mandelbrot")]))
    by_atom = defaultdict(list)
    for i, r in enumerate(erows):
        if r["atom_id"]:
            by_atom[r["atom_id"]].append(i)
    same = np.array([C[x, y] for v in by_atom.values() for x, y in combinations(v, 2)])
    nmi = [i for i, r in enumerate(erows) if r["atom_id"]]
    diff = np.array([C[x, y] for x, y in combinations(nmi[:150], 2)
                     if erows[x]["atom_id"] != erows[y]["atom_id"]])
    w(f"    DIRECT rung-vs-rung morphology (no clustering): same atom/different rung "
      f"n={len(same)} pairs median cos {np.median(same):.4f}, {(same >= STRICT).mean():.1%} "
      f"at/above the {STRICT} near-dup cut")
    w(f"    different atoms (reference) n={len(diff)} pairs median cos {np.median(diff):.4f}, "
      f"{(diff >= STRICT).mean():.1%} above cut  -> {len(by_atom)} atoms carry "
      f"{len(nmi)} ladder rows")

    # ---------------- item 4 ----------------
    w("\n[4] TRIGGERED vs FRESH (harvest_ranked). Arms from mix_source lineage, NOT the")
    w("    `triggered` row stamp: steered_frontier.push_children rebuilds the frontier node")
    w("    carrying mix_source and man but NOT triggered, so the stamp dies after one")
    w("    generation (178 of 794 triggered-lineage rows keep it in the run store; the other")
    w("    616 are stamped fresh). man['triggered'] agrees with mix_source on all 7423 rows.")
    HR = IDX[("b", BATCHES[0])]
    trig = [i for i in HR if (erows[i]["mix_source"] or "").startswith("triggered")]
    fresh = [i for i in HR if not (erows[i]["mix_source"] or "").startswith("triggered")]
    w(line("triggered", trig))
    w(line("fresh", fresh))
    mt = mf = mtb = mfb = den = 0.0
    for p in sorted({erows[i]["partition"] for i in trig}):
        a = [i for i in trig if erows[i]["partition"] == p]
        b_ = [i for i in fresh if erows[i]["partition"] == p]
        if not b_:
            continue
        ka, na, pa, _, _, _, ba = rate(a, lambda r: r["human"] >= 3)
        kb, nb, pb, _, _, _, bb = rate(b_, lambda r: r["human"] >= 3)
        w(f"      {p:16s} trig {ka:2d}/{na:2d}={pa:5.1%} (1pc {ba[0]:5.1%})    "
          f"fresh {kb:3d}/{nb:3d}={pb:5.1%} (1pc {bb[0]:5.1%})")
        mt += na * pa; mf += na * pb; mtb += na * ba[0]; mfb += na * bb[0]; den += na
    w(f"    PARTITION-MATCHED >=3 (weights = triggered arm's partition mix, n={int(den)}):")
    w(f"      raw            triggered {mt/den:5.1%}  vs  fresh {mf/den:5.1%}")
    w(f"      one-per-cluster triggered {mtb/den:5.1%}  vs  fresh {mfb/den:5.1%}")

    # ---------------- item 5 ----------------
    w("\n[5] MACHINE TRUST — v10 decode (fresh, on the labeled crop, deploy transform,")
    w("    corn_decode at the run's per-partition t_good) x human label.")
    d1 = [i for i in range(N) if erows[i]["decoded_run"] == 1]
    k, n, p, lo, hi, lk, bl = rate(d1, lambda r: r["human"] == 1)
    w(f"    P(Matt=1 | decoded 1) pooled {k}/{n}={p:.1%} [{lo:.1%},{hi:.1%}]  "
      f"one-per-cluster {bl[0]:.1%}")
    k3, _, p3, lo3, hi3, _, bl3 = rate(d1, lambda r: r["human"] >= 3)
    w(f"    P(Matt>=3 | decoded 1)       {k3}/{n}={p3:.1%} [{lo3:.1%},{hi3:.1%}]  "
      f"one-per-cluster {bl3[0]:.1%}   <- what auto-discard would throw away")
    w("    per partition (the pooled number is NOT a decision — it is dominated by the")
    w("    uniform multibrot draws, where decode-1 and human-1 agree perfectly):")
    for fam in sorted({r["partition"] for r in erows if r["partition"]}):
        s = [i for i in d1 if erows[i]["partition"] == fam]
        if not s:
            w(f"      {fam:18s} no decoded-1 rows")
            continue
        k, n2, p, lo, hi, _, bl = rate(s, lambda r: r["human"] == 1)
        k3, _, p3, _, _, _, bl3 = rate(s, lambda r: r["human"] >= 3)
        w(f"      {fam:18s} P(=1|dec1) {k:3d}/{n2:3d}={p:5.1%}[{lo:.0%},{hi:.0%}] 1pc {bl[0]:5.1%}"
          f" | P(>=3|dec1) {k3:3d}/{n2:3d}={p3:5.1%} 1pc {bl3[0]:5.1%}")
    w("    FULL confusion (row = v10 decode, col = Matt), per partition:")
    for fam in sorted({r["partition"] for r in erows if r["partition"]}):
        rs = [r for r in erows if r["partition"] == fam]
        cells = []
        for d in (1, 2, 3, 4):
            c4 = [sum(1 for r in rs if r["decoded_run"] == d and r["human"] == h)
                  for h in (1, 2, 3, 4)]
            if sum(c4):
                cells.append(f"d{d}:{c4}")
        w(f"      {fam:18s} n={len(rs):3d}  " + "  ".join(cells))
    pooled = []
    for d in (1, 2, 3, 4):
        pooled.append([sum(1 for r in erows if r["decoded_run"] == d and r["human"] == h)
                       for h in (1, 2, 3, 4)])
    w(f"      {'POOLED':18s} n={N:3d}  " + "  ".join(f"d{d+1}:{c}" for d, c in enumerate(pooled)))
    have = [r for r in erows if r["prov_decode"] is not None]
    ag = sum(1 for r in have if r["decoded_run"] == r["prov_decode"])
    w(f"    crop decode vs the decode the RUN recorded: {ag}/{len(have)}={ag/len(have):.1%} "
      f"identical (different geometry AND palette — not a parity check)")

    # ---------------- item 6 ----------------
    w("\n[6] MOTIF SATURATION")
    for name, cs in (("single-linkage", link), ("leader/radius", ball)):
        sizes = Counter(len(c) for c in cs)
        w(f"    {name:15s} cos>={STRICT}: {len(cs):3d} distinct looks / {N} rows = "
          f"{N/len(cs):.2f} labels per look; singletons {sizes[1]:3d}; "
          f"top-5 {[len(c) for c in cs[:5]]}; rows in looks of size>=10: "
          f"{sum(len(c) for c in cs if len(c) >= 10)}")
    ll, lb = _clusters(C, N, LOOSE)
    w(f"    sensitivity at cos>={LOOSE}: linkage {len(ll)} looks, ball {len(lb)} looks")
    w(f"    single-linkage CHAINS here: largest component min intra-pair cos "
      f"{min(C[np.ix_(link[0], link[0])][np.triu_indices(len(link[0]), 1)]):.4f} << {STRICT}.")
    w("    the giant single-linkage clusters and the sampler behind each:")
    for k2, c in enumerate(link):
        if len(c) < 10:
            break
        rs = [erows[i] for i in c]
        sub = C[np.ix_(c, c)][np.triu_indices(len(c), 1)]
        w(f"      #{k2} n={len(c):3d} labels={dict(sorted(Counter(r['human'] for r in rs).items()))} "
          f"part={dict(Counter(r['partition'] for r in rs))} "
          f"batch={dict(Counter(r['batch'].replace('2026-08-03_q4_', '') for r in rs))}")
        atoms = {r["atom_id"] for r in rs if r["atom_id"]}
        w(f"          intra cos median {np.median(sub):.4f} min {sub.min():.4f}"
          + (f"; {len(atoms)} source atoms x rungs "
             f"{dict(Counter(r['ladder_rung'] for r in rs if r['ladder_rung']))}" if atoms else ""))

    # ---------------- durable records ----------------
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    cmd = "uv run python tools/corpus/q4_combined_readout.py readout --apply"
    inst_doc = {
        "artifact": "q4_uniform_eval_v1 human base rates",
        "written": ts, "command": cmd, "sitting": SITTING, "batch": EVAL_BATCH,
        "run": RUN_DIR,
        "population": ("290 rows drawn with NO viability screen and NO classifier screen: "
                       "near-dM_d shell eps=0.02 for mandelbrot/multibrot3/4/5 and "
                       "julia:mandelbrot, phoenix skeleton closed form. Registered "
                       "eval/unbiased; labeled blind inside an 870-row combined sitting."),
        "labels": "human, 1..4 (bad/okay/good/exceptional); label_store.resolve_score",
        "first_eval_instrument_for": ["phoenix", "multibrot3", "multibrot4", "multibrot5",
                                      "julia:mandelbrot"],
        "intended_use": ("calibration input for the NEXT model version's t_good. v10 is "
                         "frozen; nothing in this record flips a live threshold."),
        "caveats": [
            "The native-multibrot cells are 0/48 at >=2 for all three degrees; that is a "
            "measurement of the DRAW (an eps=0.02 shell with no screen), not a ceiling on "
            "the family.",
            "Cluster-weighted variants are included because these rows are not independent: "
            "126 of the 290 fall in one single-linkage morph component.",
            "No `mandelbrot` cell: the uniform leg drew none.",
        ],
        "morph": {"producer": str(Z["producer"]), "geometry": str(Z["geometry"]),
                  "clip_model": str(Z["clip_model"]), "cut": STRICT},
        "rates": inst,
    }
    cl_rows = []
    for i, k2 in enumerate(keys):
        b, iid = k2.split("|", 1)
        cl_rows.append(dict(batch=b, image_id=iid,
                            cluster_linkage=cid[i], cluster_linkage_size=len(link[cid[i]]),
                            cluster_ball=lid[i], cluster_ball_size=len(ball[lid[i]])))

    # residue: labeled >=3 that the run did NOT admit, ranked
    q = [r for r in erows if r["human"] >= 3 and r["fate"] != "admitted"]
    q.sort(key=lambda r: (-r["human"], -r["eord"]))
    seen, qrows = set(), []
    for r in q:
        i = keys.index(f"{r['batch']}|{r['image_id']}")
        qrows.append(dict(batch=r["batch"], image_id=r["image_id"], partition=r["partition"],
                          human=r["human"], v10_eord=r["eord"], v10_decoded=r["decoded_run"],
                          fate=r["fate"], stratum=r["stratum"], mix_source=r["mix_source"],
                          cluster_ball=lid[i], cluster_ball_size=len(ball[lid[i]]),
                          first_of_look=lid[i] not in seen))
        seen.add(lid[i])
    n_looks = len(seen)
    w(f"\n[residue] >=3 and NOT in the run's admitted set: {len(qrows)} rows / {n_looks} "
      f"distinct looks (ball). admitted-and->=3 excluded: "
      f"{sum(1 for r in erows if r['human'] >= 3 and r['fate'] == 'admitted')}")

    targets = [
        (paths.durable("data/label_corpus/eval_instruments/q4_uniform_eval_v1.json",
                       mkparents=args.apply), json.dumps(inst_doc, indent=1)),
        (paths.durable("data/label_corpus/motif/q4_combined_2026-08-03_clusters.jsonl",
                       mkparents=args.apply),
         "\n".join(json.dumps(r) for r in cl_rows) + "\n"),
        (paths.durable(f"{RUN_DIR}/human_q3plus_queue.jsonl", mkparents=args.apply),
         "\n".join(json.dumps(r) for r in qrows) + "\n"),
    ]
    for p, body in targets:
        if args.apply:
            Path(p).write_text(body, encoding="utf-8")
            print(f"  WROTE {p}")
        else:
            print(f"  would write {p} ({len(body)} bytes)")
    out_txt = WORK / "readout.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(L), encoding="utf-8")
    print(f"  readout text -> {out_txt}")
    if not args.apply:
        print("  DRY RUN — pass --apply to write the durable records")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("score")
    sub.add_parser("morph")
    r = sub.add_parser("readout")
    r.add_argument("--apply", action="store_true", help="write the durable records")
    a = ap.parse_args()
    os.chdir(ROOT)
    return {"score": stage_score, "morph": stage_morph, "readout": stage_readout}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
