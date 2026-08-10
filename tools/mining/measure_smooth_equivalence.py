r"""measure_smooth_equivalence.py — run the smooth-equivalence measure over a render-mode
corpus batch and write its per-mode table.

One batch in, one JSON out. The expensive half is rendering each row's SMOOTH twin, and the
twin is a pure function of (location, palette, colour params, geometry) — NOT of the mode —
so twins are rendered once per DISTINCT RECIPE and shared by every mode at that recipe. On
the (27) mining sheet that is 253 renders for 1,000 rows.

    uv run python -u tools/mining/measure_smooth_equivalence.py estimate --batch <id>
    uv run python -u tools/mining/measure_smooth_equivalence.py render  --batch <id>
    uv run python -u tools/mining/measure_smooth_equivalence.py table   --batch <id>

`render` is resumable (skip-if-exists on the twin crop) and writes nothing but twins;
`table` embeds and reports and renders nothing. The twins land under `scratch/` — they are
an instrument, not corpus rows, and re-deriving them is one command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_common as cc                                   # noqa: E402
from tools.mining import build_mining_sheet as BMS           # noqa: E402  THE render path
from tools.mining import mining_roster as MR                 # noqa: E402
from tools.mining import smooth_equivalence as SE            # noqa: E402

CORPUS = ROOT / "data" / "render_mode_corpus"
WORK = ROOT / "scratch" / "smooth_equivalence"
WORKERS = 4


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# Rows -> (recipe key, twin entry).
# --------------------------------------------------------------------------- #
RECIPE_KEYS = ("reverse", "log_premap", "gamma", "phase", "n_cycles",
               "transfer", "transfer_gamma")


def recipe_key(row) -> str:
    """The identity of a SMOOTH twin: location + palette + every colour knob the render tail
    reads. Two rows sharing it share a twin; two rows differing anywhere in it do not.

    Derived from `RECIPE_KEYS` rather than spelled per-field at the call site, so a colour
    knob added to the corpus schema shows up here as a KeyError instead of as two rows
    quietly sharing a twin that is right for only one of them."""
    p = row["provenance"]["color_params"]
    parts = [row["provenance"]["location_key"], p["palette"]] + [repr(p[k]) for k in RECIPE_KEYS]
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def twin_entry(row, key: str) -> dict:
    p = row["provenance"]["color_params"]
    return {
        "mode": MR.SMOOTH_MODE, "kind": MR.SMOOTH_KIND,
        "location_key": row["provenance"]["location_key"],
        "family": row["render"]["fractal_type"],
        "palette": p["palette"], "color_params": p,
        "render": row["render"], "mode_params": {},
        "image_id": key,
    }


def load_batch(batch_id: str):
    d = CORPUS / "batches" / batch_id
    ip = d / "images.jsonl"
    if not ip.exists():
        raise SystemExit(f"[smooth-eq] no such batch: {ip}")
    rows = [json.loads(l) for l in ip.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((d / "batch.json").read_text(encoding="utf-8"))
    return d, rows, meta


def geom_of(meta, rows) -> tuple:
    rd = meta.get("render_defaults") or {}
    r0 = rows[0]["render"]
    return (int(rd.get("width", r0["width"])), int(rd.get("height", r0["height"])),
            int(rd.get("ss", r0["ss"])))


def plan(batch_id: str):
    bdir, rows, meta = load_batch(batch_id)
    keys = [recipe_key(r) for r in rows]
    twins, seen = [], set()
    for r, k in zip(rows, keys):
        if k not in seen:
            seen.add(k)
            twins.append(twin_entry(r, k))
    return bdir, rows, keys, twins, geom_of(meta, rows)


# --------------------------------------------------------------------------- #
# Stages.
# --------------------------------------------------------------------------- #
def run_render(batch_id: str, args):
    bdir, rows, _keys, twins, geom = plan(batch_id)
    out = WORK / batch_id / "smooth_crops"
    fields = WORK / batch_id / "_fields"
    out.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    todo = [t for t in twins if not (out / f"{t['image_id']}.jpg").exists()]
    log(f"[smooth-eq] {batch_id}: {len(rows)} rows · {len(twins)} distinct recipes · "
        f"todo {len(todo)} · geom {geom} · {args.workers}x{BMS.ENGINE_THREADS} threads")
    if args.limit:
        idx = np.linspace(0, len(todo) - 1, min(args.limit, len(todo))).round().astype(int)
        todo = [todo[int(i)] for i in sorted(set(idx.tolist()))]
    if not todo:
        return
    timeout_s = max(90.0, min(args.unit_timeout, 0.25 * args.wall_budget_s))
    t0, n, errs = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=BMS._init_worker) as ex:
        futs = {ex.submit(BMS.render_one, (t, str(out), str(fields), timeout_s, geom)): t
                for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                fut.result()
            except Exception as exc:                          # noqa: BLE001
                errs.append({"key": t["image_id"], "family": t["family"],
                             "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                log(f"[smooth-eq] ERR {t['image_id']}: {str(exc)[:160]}")
                continue
            n += 1
            if n % 25 == 0 or n <= 2:
                el = time.time() - t0
                log(f"[smooth-eq] {n}/{len(todo)}  {n/el:.2f} twin/s -> eta "
                    f"{(len(todo)-n)/(n/el)/60:.0f} min (elapsed {el/60:.0f} min)")
    (WORK / batch_id / "render_errors.json").write_text(json.dumps(errs, indent=1),
                                                        encoding="utf-8")
    log(f"[smooth-eq] rendered {n} twins in {(time.time()-t0)/60:.1f} min, {len(errs)} failed")


def run_table(batch_id: str, args):
    bdir, rows, keys, twins, geom = plan(batch_id)
    out = WORK / batch_id / "smooth_crops"
    crops = bdir / "crops"

    live = [(r, k) for r, k in zip(rows, keys)
            if (out / f"{k}.jpg").exists() and (crops / f"{r['image_id']}.jpg").exists()]
    missing = len(rows) - len(live)
    log(f"[smooth-eq] pairing {len(live)}/{len(rows)} rows ({missing} without a twin or crop)")
    if not live:
        raise SystemExit("[smooth-eq] nothing paired — run `render` first")

    emb = SE.Embedder()
    log(f"[smooth-eq] {emb.model_name} on {emb.device}")
    mode_vecs = emb.embed_paths([crops / f"{r['image_id']}.jpg" for r, _k in live])
    twin_vecs = emb.embed_paths([out / f"{k}.jpg" for _r, k in live])
    cos = SE.cos_to_smooth(mode_vecs, twin_vecs)

    modes = [r["render"]["render_mode"] for r, _k in live]
    kinds = [r["provenance"]["mode_kind"] for r, _k in live]
    doc = {
        "batch_id": batch_id,
        "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": f"uv run python tools/mining/measure_smooth_equivalence.py table "
                   f"--batch {batch_id}",
        "geometry": list(geom),
        "n_rows": len(rows), "n_paired": len(live), "n_unpaired": missing,
        "n_distinct_recipes": len(twins),
        "yardstick": SE.yardstick_block(),
        "overall": {**SE.quantiles(cos),
                    "share_near_dup": float((cos >= SE.STRICT_CUT).mean()),
                    "bands": dict(Counter(SE.band_of(c) for c in cos))},
        "unrelated_reference": {
            **SE.unrelated_reference(mode_vecs),
            "what": "cosine between renders of DIFFERENT rows of this same batch — the floor "
                    "the colored-CLIP scale actually has here",
        },
        "by_mode": SE.per_group_table(modes, cos),
        "by_kind": SE.per_group_table(kinds, cos),
        "within_location_mode_pairs": within_location_pairs(live, mode_vecs),
        "per_row": [{"image_id": r["image_id"], "mode": r["render"]["render_mode"],
                     "kind": r["provenance"]["mode_kind"],
                     "location_key": r["provenance"]["location_key"],
                     "recipe": k, "cos_smooth": float(c), "band": SE.band_of(float(c))}
                    for (r, k), c in zip(live, cos)],
    }
    dst = WORK / batch_id / "smooth_equivalence.json"
    dst.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print_table(doc)
    log(f"-> {dst}")
    return doc


def within_location_pairs(live, vecs) -> dict:
    """Every pair of SERVED rows at the SAME location — the duplication a human actually
    meets on the page.

    The prompt's measure is mode-vs-smooth, and that is what `by_mode` reports. But a sheet
    row is never shown next to its smooth twin (the twin is not served); it is shown next to
    the OTHER modes of its own location. So "many strange modes are duplicates" can be true
    of the page while every mode is far from smooth, and only this readout can tell the two
    apart. Same substrate, same cut."""
    by_loc = {}
    for i, (r, _k) in enumerate(live):
        by_loc.setdefault(r["provenance"]["location_key"], []).append(i)
    cs, pairs = [], []
    for k, idx in sorted(by_loc.items()):
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                c = float(np.dot(vecs[idx[a]], vecs[idx[b]]))
                cs.append(c)
                if c >= SE.STRICT_CUT:
                    pairs.append({"location_key": k, "cos": c,
                                  "a": live[idx[a]][0]["image_id"],
                                  "b": live[idx[b]][0]["image_id"],
                                  "modes": [live[idx[a]][0]["render"]["render_mode"],
                                            live[idx[b]][0]["render"]["render_mode"]]})
    if not cs:
        return {"n_pairs": 0}
    arr = np.asarray(cs, dtype=np.float64)
    mode_pair = Counter()
    for p in pairs:
        mode_pair[" + ".join(sorted(p["modes"]))] += 1
    return {
        "n_locations": len(by_loc), "n_pairs": int(arr.size),
        **SE.quantiles(arr),
        "share_near_dup": float((arr >= SE.STRICT_CUT).mean()),
        "n_near_dup_pairs": len(pairs),
        "rows_in_a_near_dup_pair": len({x for p in pairs for x in (p["a"], p["b"])}),
        "top_mode_pairs": dict(mode_pair.most_common(12)),
        "examples": pairs[:20],
    }


def print_table(doc):
    log("=" * 96)
    log(f"SMOOTH-EQUIVALENCE — {doc['batch_id']}  ({doc['n_paired']} rows, "
        f"{doc['n_distinct_recipes']} recipes, geom {doc['geometry']})")
    log(f"overall median cos {doc['overall']['q']['p50']:.4f}   near-dup (>= "
        f"{doc['yardstick']['strict_cut']}) {doc['overall']['share_near_dup']*100:.1f}%")
    u = doc["unrelated_reference"]
    log(f"unrelated-pair floor: p50 {u['q']['p50']:.4f}  p95 {u['q']['p95']:.4f}  "
        f"share>=strict {u['share_ge_strict']*100:.2f}%")
    log("-" * 96)
    log(f"{'mode':<34}{'n':>5}{'near-dup':>10}{'interleave':>12}{'distinct':>10}{'median':>9}")
    for m, v in sorted(doc["by_mode"].items(), key=lambda kv: -kv[1]["share_near_dup"]):
        log(f"{m:<34}{v['n']:>5}{v['share_near_dup']*100:>9.1f}%{v['share_interleave']*100:>11.1f}%"
            f"{v['share_distinct']*100:>9.1f}%{v['median_cos']:>9.4f}")
    log("-" * 96)
    for k, v in sorted(doc["by_kind"].items()):
        log(f"{k:<34}{v['n']:>5}{v['share_near_dup']*100:>9.1f}%{v['share_interleave']*100:>11.1f}%"
            f"{v['share_distinct']*100:>9.1f}%{v['median_cos']:>9.4f}")
    w = doc.get("within_location_mode_pairs") or {}
    if w.get("n_pairs"):
        log("-" * 96)
        log(f"SAME-LOCATION served pairs: {w['n_pairs']} over {w['n_locations']} locations · "
            f"median {w['q']['p50']:.4f} · near-dup {w['share_near_dup']*100:.1f}% "
            f"({w['n_near_dup_pairs']} pairs, {w['rows_in_a_near_dup_pair']} rows)")
        for k, v in list(w["top_mode_pairs"].items())[:8]:
            log(f"    {v:>4}  {k}")
    log("=" * 96)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("estimate", "render", "table"))
    ap.add_argument("--batch", required=True)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--unit-timeout", type=float, default=900.0)
    ap.add_argument("--wall-budget-s", type=float, default=3 * 3600.0)
    args = ap.parse_args(argv)
    if args.workers > WORKERS:
        raise SystemExit(f"[smooth-eq] --workers {args.workers} exceeds the process cap "
                         f"of {WORKERS} (CLAUDE.md).")
    cc.set_below_normal_priority()

    if args.stage == "estimate":
        _b, rows, _k, twins, geom = plan(args.batch)
        log(json.dumps({"batch": args.batch, "rows": len(rows), "twins": len(twins),
                        "geom": list(geom),
                        "by_family": dict(Counter(t["family"] for t in twins))}, indent=1))
        return 0
    if args.stage == "render":
        run_render(args.batch, args)
        return 0
    run_table(args.batch, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
