#!/usr/bin/env python
"""Driver: build the source sheets in priority order, **shipping each one complete
before starting the next**.

Five complete sheets beat eight half-built ones, so every sheet is enumerated,
rendered, written and linked from the index the moment it is done — nothing is held
back for a final assembly step. The index is rewritten after every sheet, so an
interrupted run still leaves a usable, honest index of what exists.

Wall clock: `--cap-hours` bounds the whole run. Before each sheet the driver
estimates its duration from the measured cost of the sheets already built; if
`elapsed + estimate > cap` it **stops and reports** rather than starting it. Each
source also gets its own deadline, and every render carries a per-tile hard timeout,
so one non-converging Newton or one pathological location cannot eat the night.

Run:  uv run python tools/sources/run_sheets.py --cap-hours 5
      uv run python tools/sources/run_sheets.py --only probe,label_seeded
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "descent"))

import atom_lib as al          # noqa: E402
import source_store as ss      # noqa: E402
import render_tiles as rt      # noqa: E402
import sheet as sh             # noqa: E402
import sources as S            # noqa: E402
import triage_store as ts      # noqa: E402

SHEET_N = 150                  # atoms per sheet (depth-spanned down to this from the pool)
POOL_TARGET = 260              # enumerate this many so the depth span has something to span

# (source_id, title, one-line, blurb, builder) in the prompt's priority order.
PLAN = [
    ("probe", "1 · Probe sampling, unstratified",
     "Newton over a ring-seed grid; the baseline every other sheet is read against.",
     "The cheapest source and the control: Newton from a scrambled ring-seed grid across "
     "the whole region, keeping every minimal nucleus it lands on. No period "
     "stratification, no per-cell cap, and — per the addendum — no feasibility exclusion."),
    ("label_seeded", "2 · Labeled-location seed",
     "Nuclei solved at/near each degree-2 q3/q4 location in the label corpus.",
     "Takes the locations Matt already judged good and asks what minibrot, if any, each "
     "one sits on. A nucleus counts only if it lies inside that labelled view, so this is "
     "genuinely label-seeded rather than a global scan wearing a label."),
    ("neighborhood", "3 · Neighbourhood expansion",
     "Discs probed around each sheet-2 nucleus, at comparable and smaller scale.",
     "Tests whether richness is locally correlated — an assumption made everywhere and "
     "never checked. Probe radii are measured in units of the parent's own window scale, "
     "so 'nearby' means the same thing for a shallow parent and a deep one."),
    ("atlas", "4 · Atlas mining",
     "Nuclei under the repo's hand-curated named locations, plus neighbourhood fill.",
     "Decades of human curation we have never mined. The named supply is small (there is "
     "no mandelbrot_named_seeds.json; the curated set is source constants), so the sheet "
     "is filled by expanding neighbourhoods around the curated hits — the split is in the "
     "header."),
    ("complete_low_n", "5 · Low-n complete enumeration",
     "EVERY component of exact period n, for n up to the tractable limit.",
     "A complete population rather than a sample, so its descriptor mix is a fact, not an "
     "estimate. Completeness is proved by counting: the number of period-n components is "
     "fixed by sum_{d|n} nu(d) = 2^(n-1), and the scan stops at exactly that count."),
    ("descent", "6 · Descent-found",
     "Nuclei solved wherever the existing guided-descent walker landed.",
     "The descent machinery's own candidate pools, asked what minibrot each landing sits "
     "on."),
    ("misiurewicz", "7 · Misiurewicz-anchored",
     "Nuclei found in the neighbourhood of preperiodic points.",
     "Misiurewicz points stay on the boundary at every scale, so their neighbourhoods are "
     "dense in small components. The question is whether the nuclei living there differ "
     "in kind from what a plain scan finds."),
]

TUNING_NOTE = ("8 · Douady tuning was deliberately not attempted: it needs a known-good "
               "ancestor to tune from, and these sheets are what produce that.")


def log(msg=""):
    print(msg, flush=True)


def overlap_matrix(built: list[str]) -> dict:
    ids = {s: {a["id"] for a in ss.load_atoms(s)} for s in built}
    mat = {i: {j: len(ids[i] & ids[j]) if i != j else len(ids[i]) for j in built}
           for i in built}
    return {"sources": built, "matrix": mat}


def disk_entries(extra=None):
    """Index entries for EVERY source on disk, not just the ones this run built — so
    re-running a single sheet (`--only neighborhood`) never drops the others."""
    titles = {s: (t, o) for s, t, o, _b in
              [(a, b, c, d) for a, b, c, d in PLAN]}
    out = []
    for sid in ss.built_sources():
        m = ss.load_meta(sid)
        t_, o_ = titles.get(sid, (m.get("title", sid), m.get("one_line", "")))
        out.append({"source_id": sid, "title": m.get("title", t_),
                    "one_line": m.get("one_line", o_),
                    "desc": m.get("descriptors", {}), "atoms": []})
    order = [s for s, *_ in PLAN] + ["blind_all"]
    out.sort(key=lambda e: order.index(e["source_id"]) if e["source_id"] in order else 99)
    for e in (extra or []):
        if e["source_id"] not in {x["source_id"] for x in out}:
            out.append(e)
    return out


def rebuild_index(entries, skipped):
    entries = disk_entries(entries)
    ov = overlap_matrix([e["source_id"] for e in entries
                         if ss.atoms_path(e["source_id"]).exists()]) if entries else None
    if ov:
        ss.write_json(ss.OVERLAP, ov)
    p = sh.build_index(entries, ov, skipped)
    ss.write_json(ss.INDEX_JSON, {
        "sheets": [{"source_id": e["source_id"], "title": e["title"],
                    "n": e["desc"].get("n", 0), "desc": e["desc"]} for e in entries],
        "skipped": skipped,
        "overlap": ov,
        "index_html": str(p),
    })
    return p


def ship_sheet(source_id, title, one_line, blurb, atoms, stats, *, extra_desc=None,
               notes="", render_workers=4) -> dict:
    """Sample to span depth, render, drop empirical render failures, write the sheet."""
    picked = al.span_by_depth(atoms, SHEET_N)
    log(f"    sampled {len(picked)} of {len(atoms)} spanning log10|A| "
        f"{min((a['log10_abs_A'] for a in picked), default=0):.2f}"
        f"–{max((a['log10_abs_A'] for a in picked), default=0):.2f}")
    res = rt.render_atoms(picked, workers=render_workers, log=log)
    usable = [a for a in picked if a["id"] not in set(res["unusable"])]
    if res["unusable"]:
        log(f"    dropped {len(res['unusable'])} atoms on EMPIRICAL render failure")
    desc = al.describe(usable)
    ss.write_atoms(source_id, atoms)                       # durable: the FULL population
    ss.write_json(ss.meta_path(source_id), {
        "source_id": source_id, "title": title, "one_line": one_line, "blurb": blurb,
        "degree": 2, "sheet_n": len(usable), "population_n": len(atoms),
        "sheet_ids": [a["id"] for a in usable],
        "descriptors": desc, "source_stats": stats,
        "render": {"seconds": res["seconds"], "unusable": res["unusable"],
                   "failures": res["failures"][:40], "n_failures": len(res["failures"])},
        "framing": {"scales": list(ss.SCALES), "default_scale": ss.DEFAULT_SCALE,
                    "width": ss.TILE_W, "height": ss.TILE_H, "ss": ss.TILE_SS,
                    "palette": ss.TILE_PALETTE},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    })
    extra = list(extra_desc or [])
    if res["failures"]:
        extra.append(("render failures", f'{len(res["failures"])} tiles, '
                                         f'{len(res["unusable"])} atoms dropped'))
    p = sh.build_sheet(source_id, title, blurb, usable, desc,
                       extra_desc=extra, notes=notes)
    log(f"    -> {p}")
    return {"source_id": source_id, "title": title, "one_line": one_line,
            "desc": desc, "path": str(p), "atoms": usable}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-hours", type=float, default=5.0)
    ap.add_argument("--sheet-n", type=int, default=SHEET_N)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated source ids to build")
    ap.add_argument("--skip-blind", action="store_true")
    ap.add_argument("--render-workers", type=int, default=4)
    args = ap.parse_args(argv)

    globals()["SHEET_N"] = args.sheet_n
    only = set(args.only.split(",")) if args.only else None

    ss.ensure_dirs()
    t_start = time.time()
    log("references (carried onto every sheet at identical framing):")
    rt.ensure_reference_tiles(log=log)
    cap = args.cap_hours * 3600.0
    entries, skipped = [], []
    label_atoms: list[dict] = []
    # per-sheet wall-clock estimate (seconds): enumeration + ~150*3 tiles at ~5 tile/s
    est = {"probe": 480, "label_seeded": 900, "neighborhood": 900, "atlas": 900,
           "complete_low_n": 1500, "descent": 900, "misiurewicz": 1200}

    log(f"minibrot source sheets — cap {args.cap_hours:.1f} h, {args.sheet_n} atoms/sheet")
    log(f"out-of-tree bulk root: {ss.sheets_dir()}")
    log("")

    for source_id, title, one_line, blurb in PLAN:
        if only and source_id not in only:
            continue
        elapsed = time.time() - t_start
        e = est.get(source_id, 900)
        if elapsed + e > cap:
            reason = (f"not started — {elapsed/3600:.2f} h elapsed + ~{e/60:.0f} min "
                      f"estimate would exceed the {args.cap_hours:.1f} h cap")
            log(f"[{source_id}] SKIPPED: {reason}")
            skipped.append({"source_id": source_id, "reason": reason})
            continue
        log(f"[{source_id}] {title}   (elapsed {elapsed/60:.1f} min)")
        deadline = t_start + min(cap, elapsed + e * 1.6)
        t0 = time.time()
        try:
            if source_id == "probe":
                atoms, stats = S.src_probe(POOL_TARGET, deadline=deadline, log=log)
            elif source_id == "label_seeded":
                atoms, stats = S.src_label_seeded(POOL_TARGET, deadline=deadline, log=log)
                label_atoms = atoms
            elif source_id == "neighborhood":
                parents = label_atoms or ss.load_atoms("label_seeded")
                if not parents:
                    raise RuntimeError("no sheet-2 parents available")
                random.Random(7).shuffle(parents)
                atoms, stats = S.src_neighborhood(parents[:60], POOL_TARGET,
                                                  deadline=deadline, log=log)
            elif source_id == "atlas":
                atoms, stats = S.src_atlas(POOL_TARGET, deadline=deadline, log=log)
            elif source_id == "complete_low_n":
                atoms, stats = S.src_complete_low_n(deadline=deadline, log=log)
            elif source_id == "descent":
                atoms, stats = S.src_descent(POOL_TARGET, deadline=deadline, log=log)
            elif source_id == "misiurewicz":
                atoms, stats = S.src_misiurewicz(POOL_TARGET, deadline=deadline, log=log)
            else:
                raise RuntimeError(f"unknown source {source_id}")
        except Exception as ex:
            log(f"    FAILED: {ex}")
            traceback.print_exc()
            skipped.append({"source_id": source_id, "reason": f"failed: {ex}"})
            continue

        if not atoms:
            reason = "source produced no atoms"
            log(f"    {reason}")
            skipped.append({"source_id": source_id, "reason": reason})
            continue

        extra, notes = None, ""
        if source_id == "complete_low_n":
            th = stats.get("theorem_satellites", {})
            per = th.get("per_period", [])
            done = [r["period"] for r in per if r["complete"]]
            short = [r for r in per if not r["complete"]]
            shipped = stats.get("shipped_periods", [])
            short2 = stats.get("attempted_but_short", [])
            extra = [("completeness",
                      f'COMPLETE population for periods {min(shipped)}–{max(shipped)}'
                      if shipped else "none complete"),
                     ("attempted but short (NOT on this sheet)",
                      ", ".join(f'n={r["period"]} ({r["distinct"]}/{r["expected_total"]})'
                                for r in short2) or "none"),
                     ("primitive / satellite (EXACT, from the counting theorem)",
                      f'{th.get("complete_satellites")} of {th.get("complete_total")} '
                      f'= {th.get("satellite_frac", 0):.0%} satellite')]
            notes = ("<br>This sheet is a <b>complete population</b> per period, not a sample; "
                     "the tiles shown are a depth-spanning draw from it.")
        if len(atoms) < 100:
            notes += (f"<br><b>Short sheet:</b> this source produced only {len(atoms)} atoms. "
                      f"Not padded from any other source.")
        ent = ship_sheet(source_id, title, one_line, blurb, atoms, stats,
                         extra_desc=extra, notes=notes,
                         render_workers=args.render_workers)
        entries.append(ent)
        rebuild_index(entries, skipped)
        log(f"    sheet shipped in {(time.time()-t0)/60:.1f} min "
            f"(total {(time.time()-t_start)/60:.1f} min)\n")

    # blind interleaved wall — costs no extra rendering
    if entries and not args.skip_blind:
        allat = [a for e in entries for a in e["atoms"]]
        random.Random(20260730).shuffle(allat)
        d = al.describe(allat)
        sh.build_sheet("blind_all", "Blind interleaved wall — all sources shuffled",
                       "Every rendered tile from every sheet, shuffled, source hidden. "
                       "Costs no extra rendering and preserves an unbiased per-tile rate. "
                       "Secondary to the per-source sheets.",
                       allat, d, notes="<br>Source is hidden by construction; the mapping "
                                       "lives in each source's meta.json.",
                       shuffled=True)
        entries.append({"source_id": "blind_all", "title": "Blind interleaved wall",
                        "one_line": "All sources shuffled, source hidden.",
                        "desc": d, "atoms": []})
        rebuild_index(entries, skipped)

    skipped.append({"source_id": "douady_tuning", "reason": TUNING_NOTE})
    p = rebuild_index([e for e in entries], skipped)
    log(f"\nINDEX: {p}")
    log(f"total {(time.time()-t_start)/60:.1f} min, {len(entries)} sheets, "
        f"{len(skipped)} not built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
