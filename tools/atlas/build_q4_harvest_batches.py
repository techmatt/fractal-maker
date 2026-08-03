#!/usr/bin/env python
r"""build_q4_harvest_batches.py — the long harvest's three label batches + the ranked queue.

THREE POPULATIONS, THREE SELECTION STORIES, ONE RIG. They share the corpus machinery
verbatim (`corpus_common.render_corpus_crop`, the score-3 palette roster with a seeded
per-image draw, the `blue_orange` vivid companion, opaque post-shuffle `image_id`s, a
`blind.jsonl` whose provenance is batch identity and nothing else) and differ only in how
their rows were chosen — which is exactly why they are three registered batches and not one:

  * **RANKED** (`q4_harvest_ranked`) — the top of the run's own record-and-rank queue, drawn
    round-robin over (fate x partition). The cells are FATE cells on purpose: Matt is picking
    a cutoff, and a page that showed only admissions would hide the population the cutoff is
    being chosen against. Rejects are in the batch BY DESIGN, unlabelled and unmarked.
  * **NEAR-MINIBROT** (`q4_near_minibrot`) — the distance-ladder leg, round-robin over the
    ladder rung so all three rungs are represented equally and the rung comparison survives.
  * **UNIFORM EVAL** (`q4_uniform_eval`) — the score-unconditioned draws, round-robin over
    partition. The ONLY eval-registered batch of the three.

THE RANK IS A TIERED SORT, NEVER A POOLED ONE. A candidate with a canonical decode
(`rank_tier=2`, 640x360 ss2) and one carrying only a cheap score (`rank_tier=1`, 384x216 ss1)
are scores on two different geometries; pooling them into one ordering is the cap/geometry
error `orbital_field_metrics.md` §5 forbids. So tier 2 sorts above tier 1 as a block, and the
sort runs WITHIN each tier. The tier rides every row's provenance so the split stays visible.

EVERY SELECTION KEY IS A LEAK KEY and is ABSENT from the served manifest rather than nulled
in it — a nulled key still tells the labeler that a selection happened and names its axis.
`fate` is the sharpest one here: a row labelled "this was admitted" is not a blind judgement.

NO CALIBRATION AIDS OF ANY KIND. No exemplars beside the rig, no reference strip, no
score shown, no ordering the labeler can infer.

  uv run python tools/atlas/build_q4_harvest_batches.py queue --run-dir data/discovery/<run>
  uv run python tools/atlas/build_q4_harvest_batches.py draw  --run-dir data/discovery/<run>
  uv run python tools/atlas/build_q4_harvest_batches.py render --max-minutes 90
  uv run python tools/atlas/build_q4_harvest_batches.py verify
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
           str(ROOT / "tools" / "sourcing"), str(ROOT / "tools" / "scoring")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                    # noqa: E402
import corpus_common as cc                      # noqa: E402
import build_minibrot_batch as BMB              # noqa: E402  (coords / palettes / io)
import deep_center_finder as dcf                # noqa: E402  (the corpus crop cap policy)
from tools.v7 import build_manifest as bm       # noqa: E402  (assign_split — the authority)

STAMP = "2026-08-03"
GEN_VERSION = "q4_long_harvest_v1"
PRESENTATION_SEED = 0x94A0803

CROP_W, CROP_H, CROP_SS = BMB.CROP_W, BMB.CROP_H, BMB.CROP_SS
CROP_FILTER, INTERIOR_MODE, COMPOSITION = BMB.CROP_FILTER, BMB.INTERIOR_MODE, BMB.COMPOSITION
PALETTE_SOURCE, VIVID_PALETTE, VIVID_SOURCE = (BMB.PALETTE_SOURCE, BMB.VIVID_PALETTE,
                                               BMB.VIVID_SOURCE)

RANKED = next(iter(bm.Q4_HARVEST_RANKED_BATCHES))
NEARMB = next(iter(bm.Q4_NEAR_MINIBROT_BATCHES))
UNIFORM = next(iter(bm.Q4_UNIFORM_EVAL_BATCHES))
BATCHES = (RANKED, NEARMB, UNIFORM)
N_CHUNK = 290

RENDER_THREADS = 3     # 4 workers x 3 threads = 12 = the box's logical cores

# Absent from the served manifest, asserted absent on the served BYTES.
LEAK_KEYS = ("fate", "rank_tier", "rank_score", "cheap_eord", "cheap_pgood", "cheap_nb",
             "canon_eord", "canon_pgood", "canon_nb", "canon_pge4", "canon_decoded",
             "reframe_decoded", "decoded_class", "tau_h", "tau_rec", "t_good", "queue_rank",
             "selection_role", "stratum", "triggered", "maneuver", "mix_source", "int_frac",
             "occ", "eord", "p_good", "p_notbad", "p_ge4", "ladder_rung", "ladder_radius",
             "atom_size", "atom_period", "atom_id", "atom_source", "scorer_version",
             "draw_rule", "branch", "node_id", "root_id", "depth", "theta")


def _jl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# =========================================================================== #
# queue — the tiered rank over the run's record-and-rank store
# =========================================================================== #
def build_queue(run_dir: Path) -> tuple[list[dict], dict]:
    """Every recorded candidate, tier-sorted. Highest tier first, then by score in tier.

    The append-only store is a SUPERSET of the population (a kill can replay a batch), so
    first occurrence wins on the row identity, exactly as the maneuver loader does."""
    rows, seen = [], set()
    for r in _jl(Path(run_dir) / "q4_candidates.jsonl"):
        key = (r["partition"], r["cx"], r["cy"], r["fw"],
               r.get("julia_c_re"), r.get("julia_c_im"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    # Ties break on the geometry so the queue is a pure function of the population and a
    # re-run reproduces it byte for byte.
    rows.sort(key=lambda r: (-int(r.get("rank_tier") or 0),
                             -float(r.get("rank_score") if r.get("rank_score")
                                    is not None else -1e9),
                             str(r["cx"]), str(r["cy"])))
    for i, r in enumerate(rows, 1):
        r["queue_rank"] = i
    rep = dict(n=len(rows),
               by_tier=dict(Counter(str(r.get("rank_tier")) for r in rows)),
               by_fate=dict(Counter(r["fate"] for r in rows)),
               by_partition=dict(Counter(r["partition"] for r in rows)),
               triggered=sum(1 for r in rows if r.get("triggered")))
    return rows, rep


def queue_path(run_dir) -> Path:
    """Beside the run that regenerates it and out of the source tree (the label-seeded
    harvest's own reasoning: a deterministic function of a tracked store + committed code)."""
    rel = Path(run_dir).resolve().relative_to(ROOT).as_posix()
    return paths.bulk(f"{rel}/scratch/q4_queue.jsonl")


def stage_queue(args) -> int:
    q, rep = build_queue(args.run_dir)
    if not q:
        raise SystemExit(f"no q4_candidates.jsonl rows under {args.run_dir}")
    p = queue_path(args.run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in q:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"queue: {rep['n']} recorded candidates, tier-sorted")
    print(f"  by tier      {json.dumps(rep['by_tier'])}  (2 = canonical decode, "
          f"1 = cheap only; NEVER pooled)")
    print(f"  by fate      {json.dumps(rep['by_fate'])}")
    print(f"  by partition {json.dumps(rep['by_partition'])}")
    print(f"  triggered    {rep['triggered']}")
    print(f"  -> {p}")
    return 0


# =========================================================================== #
# draw — round-robin over cells, best-first inside a cell
# =========================================================================== #
def draw_round_robin(rows, cell_of, n: int, *, order_key):
    """`n` rows, round-robin over `cell_of`, best-first inside each cell.

    Floor-then-remainder over the cells — every non-empty cell gives up its best remaining
    row before any cell gives up its second — so the chunk is the top of the queue
    CONDITIONED on not letting one cell own the page.

    "BALANCED TO +/-1" IS AN INVARIANT ABOUT NON-EXHAUSTED CELLS, and stating it as a flat
    spread over all cells is wrong — measured on this run's own queue, where the flat spread
    is 78 while the draw is behaving perfectly. Real cells differ in SUPPLY by two orders of
    magnitude (`precanon_dup|julia:mandelbrot` has hundreds of rows, `canon_not_q3|mandelbrot`
    has one), and a cell that gave up everything it had cannot be faulted for giving up less
    than a cell that did not. So the returned per-cell record carries `available` beside
    `taken`, and the acceptance check is: every cell is within 1 of the maximum take, OR it
    was drained. A flat-spread assertion here would have gone red on a correct draw, which is
    the failure mode `verification_practice.md` §4 calls getting trained out.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[cell_of(r)].append(r)
    for v in cells.values():
        v.sort(key=order_key)
    keys = sorted(cells, key=str)
    take = {k: 0 for k in keys}
    while sum(take.values()) < n:
        cand = [k for k in keys if take[k] < len(cells[k])]
        if not cand:
            break
        k = min(cand, key=lambda k: (take[k], -len(cells[k])))
        take[k] += 1
    out = []
    for _round in range(max(take.values(), default=0)):
        for k in keys:
            if _round < take[k]:
                out.append(cells[k][_round])
    rep = {str(k): dict(taken=take[k], available=len(cells[k]),
                        drained=take[k] >= len(cells[k])) for k in keys}
    return out, rep


def cells_balanced(rep: dict) -> tuple[bool, str]:
    """The acceptance predicate for a round-robin draw. Pure, so `verify` and a test share it.

    Balanced iff every cell is within 1 of the maximum take OR was drained."""
    if not rep:
        return True, "no cells"
    mx = max(v["taken"] for v in rep.values())
    bad = {k: v for k, v in rep.items()
           if v["taken"] < mx - 1 and not v["drained"]}
    return (not bad), (f"max take {mx}; under-taken and NOT drained: {bad}" if bad
                       else f"max take {mx}, all cells within 1 or drained")


def render_family_of(partition: str) -> str:
    """Ledger PARTITION -> the render `fractal_type` token.

    A partition is namespaced (`julia:multibrot3`); a render family is not
    (`julia_multibrot3`). Getting only the `julia:mandelbrot -> julia` case right and
    leaving the three namespaced multibrot twins untouched failed 83 of 290 ranked rows
    with `unknown family 'julia:multibrot3'` — and failed them at RENDER time, one crop at
    a time, where it reads as a flaky renderer rather than as a mapping bug.

    Pinned to `steered_frontier.render_family_of` by a test rather than importing it: that
    module pulls torch, and a batch builder that loads a classifier to translate a string
    would be paying seconds of import for a dictionary."""
    if partition in ("mandelbrot", "multibrot3", "multibrot4", "multibrot5", "phoenix"):
        return partition
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    return partition


def _render_block(r: dict) -> dict:
    """The corpus render block for one row, whichever plane it lives on."""
    fam = render_family_of(r.get("family") or r.get("partition") or "mandelbrot")
    fw = cc.hp_str(r["fw"])
    render = cc.render_block(cx=str(r["cx"]), cy=str(r["cy"]), fw=fw,
                             maxiter=int(dcf._maxiter_for_fw(float(r["fw"]))),
                             palette=r["_palette"], composition=COMPOSITION,
                             width=CROP_W, height=CROP_H, ss=CROP_SS,
                             filter=CROP_FILTER, interior_mode=INTERIOR_MODE)
    render["fractal_type"] = fam
    render["c_re"] = r.get("c_re") if r.get("c_re") is not None else r.get("julia_c_re")
    render["c_im"] = r.get("c_im") if r.get("c_im") is not None else r.get("julia_c_im")
    if fam == "phoenix":
        # The phoenix identity is the whole (c, p, z_-1) point; a render block carrying only
        # `c` would rebuild a DIFFERENT phoenix at the same coordinates.
        for k in ("p_re", "p_im", "zm1_re", "zm1_im"):
            v = r.get(k)
            if v is None and r.get("phoenix"):
                v = (r["phoenix"] or {}).get(k)
            render[k] = v
    return render


def write_batch(batch_id: str, rows: list[dict], *, sampling: dict, purpose: str) -> dict:
    names = BMB._palette_names()
    order = list(range(len(rows)))
    np.random.default_rng(PRESENTATION_SEED ^ BMB._stable_seed(batch_id)).shuffle(order)
    for slot, oi in enumerate(order):
        h = BMB._stable_seed(json.dumps([rows[oi].get("cx"), rows[oi].get("cy"),
                                         rows[oi].get("fw"), rows[oi].get("c_re"),
                                         rows[oi].get("julia_c_re")], default=str))
        rows[oi]["image_id"] = f"q4{slot:04d}_{h:08x}"
    rows.sort(key=lambda r: r["image_id"])

    full, blind = [], []
    for r in rows:
        r["_palette"] = names[BMB._stable_seed(r["image_id"]) % len(names)]
        render = _render_block(r)
        prov = cc.provenance_block(
            GEN_VERSION, batch_id, family=r.get("partition") or r.get("family"),
            selection_role=sampling["selection_role"],
            stratum=str(r.get("_cell")),
            **{k: r.get(k) for k in
               ("fate", "rank_tier", "rank_score", "queue_rank", "cheap_eord",
                "cheap_pgood", "canon_eord", "canon_pgood", "canon_decoded",
                "reframe_decoded", "triggered", "mix_source", "int_frac", "occ",
                "ladder_rung", "ladder_radius", "atom_size", "atom_period", "atom_id",
                "atom_source", "draw_rule", "branch", "eord", "p_good", "p_notbad",
                "decoded_class", "tau_h", "tau_rec", "t_good", "scorer_version")
               if r.get(k) is not None})
        full.append(cc.make_row(r["image_id"], render, prov, cc.label_block()))
        blind.append(cc.make_row(r["image_id"], dict(render),
                                 {"generator_version": GEN_VERSION, "batch_id": batch_id},
                                 cc.label_block()))

    bdir = Path(cc.batch_dir(batch_id))
    bdir.mkdir(parents=True, exist_ok=True)
    cc.write_jsonl(full, str(bdir / "images.jsonl"))
    cc.write_jsonl(blind, str(bdir / "blind.jsonl"))
    split, biased, source = bm.assign_split({"batch": batch_id, "ft": "mandelbrot"})
    bj = dict(
        schema_version=1, batch_id=batch_id, generator_version=GEN_VERSION,
        created=None, labeler=None,
        presentation_seed=PRESENTATION_SEED, vivid_companion=VIVID_PALETTE,
        served_manifest="blind.jsonl", queued_for_labeling=False,
        purpose=purpose,
        counts=dict(total=len(full)),
        registration=dict(assign_split=[split, biased, source],
                          registered_explicitly=(source != "unregistered"),
                          NOTE=("registered in tools/v7/build_manifest BEFORE the build")),
        render_defaults=dict(width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE, composition=COMPOSITION,
                             palette_roster="data/palettes/score3_colormaps.json",
                             vivid_companion=VIVID_PALETTE,
                             maxiter="deep_center_finder._maxiter_for_fw(fw)"),
        render_recipe=cc.render_recipe_stamp(PALETTE_SOURCE),
        sampling_metaparameters=sampling,
        calibration_aids="NONE — no exemplars, no reference strip, no score shown",
    )
    (bdir / "batch.json").write_text(json.dumps(bj, indent=2, default=str),
                                     encoding="utf-8")
    if not (bdir / "scores.json").exists():
        (bdir / "scores.json").write_text("{}", encoding="utf-8")
    return dict(batch_id=batch_id, n=len(full), dir=str(bdir),
                assign_split=[split, biased, source])


def stage_draw(args) -> int:
    out = []
    # --- (a) ranked harvest candidates -------------------------------------- #
    q = _jl(queue_path(args.run_dir))
    if q:
        rows, cells = draw_round_robin(
            q, lambda r: f'{r["fate"]}|{r["partition"]}', args.n_chunk,
            order_key=lambda r: r["queue_rank"])
        for r in rows:
            r["_cell"] = f'{r["fate"]}|{r["partition"]}'
        out.append(write_batch(RANKED, rows, sampling=dict(
            selection_role="q4_harvest_ranked", cells=cells, queue_n=len(q),
            selection=("top of the run's TIER-SORTED record-and-rank queue, round-robin "
                       "to +/-1 over (fate x partition), best-first within a cell"),
            tiering=("rank_tier 2 (canonical decode, 640x360 ss2) sorts above tier 1 "
                     "(cheap only, 384x216 ss1) as a BLOCK; the sort runs within a tier, "
                     "never across — two geometries are not one ordering"),
            rejects_included=True,
            why_rejects=("Matt is picking a cutoff; a page of admissions only would hide "
                         "the population the cutoff is chosen against")),
            purpose=("q4 candidates from the 2026-08-03 long harvest, ranked. TRAIN-side "
                     "and BIASED: the cheap ordinal decided which candidates got a "
                     "canonical confirmation and the rank is built from those scores. No "
                     "rate measured on this batch is a base rate.")))
    else:
        print("  !! no ranked queue — run `queue` first (skipping batch a)")

    # --- (b) the near-minibrot ladder --------------------------------------- #
    nm = _jl(paths.scratch("near_minibrot", "scored.jsonl"))
    if nm:
        rows, cells = draw_round_robin(
            nm, lambda r: f'rung{r["ladder_rung"]:g}', args.n_chunk,
            order_key=lambda r: -float(r.get("eord") or 0.0))
        for r in rows:
            r["_cell"] = f'rung{r["ladder_rung"]:g}'
            r["partition"] = "julia:mandelbrot"
        out.append(write_batch(NEARMB, rows, sampling=dict(
            selection_role="q4_near_minibrot", cells=cells, population_n=len(nm),
            selection=("round-robin to +/-1 over the LADDER RUNG so all three rungs are "
                       "equally represented, best-first within a rung"),
            ladder="1 / 4 / 16 atom radii, radius = 1/|A| (the A instrument)"),
            purpose=("julia:mandelbrot c's sampled at 1/4/16 minibrot-atom radii around "
                     "known degree-2 nuclei (no fresh enumeration). TRAIN-side and biased: "
                     "the nuclei are ones two prior searches surfaced, and the within-rung "
                     "order is a score. The RUNG is the question.")))
    else:
        print("  !! no near-minibrot scores — run near_minibrot_julia score (skipping b)")

    # --- (c) the uniform eval draws ----------------------------------------- #
    ue = _jl(paths.scratch("uniform_eval", "draws.jsonl"))
    if ue:
        rows, cells = draw_round_robin(
            ue, lambda r: r["partition"], args.n_chunk,
            order_key=lambda r: r["eid"])          # NO score: the eid is arbitrary and fixed
        for r in rows:
            r["_cell"] = r["partition"]
        out.append(write_batch(UNIFORM, rows, sampling=dict(
            selection_role="q4_uniform_eval", cells=cells, population_n=len(ue),
            selection=("round-robin over partition; WITHIN a cell the order is the draw id, "
                       "which carries no score — an eval draw may not be score-ordered even "
                       "in its tie-break"),
            score_in_selection=False),
            purpose=("score-unconditioned systematic draws for the five partitions with no "
                     "unbiased eval rows (production_seeder.T_GOOD_UNCALIBRATED). EVAL-"
                     "eligible and UNBIASED: every row comes from the family's own "
                     "parameter space by a closed-form or membership rule, with no score, "
                     "screen or classifier anywhere in the selection.")))
    else:
        print("  !! no uniform eval draws — run uniform_eval_draws draw (skipping c)")

    for o in out:
        print(f"  {o['batch_id']:36s} n={o['n']:4d}  assign_split={o['assign_split']}")
    return 0


# =========================================================================== #
# render
# =========================================================================== #
def _render_to(render: dict, out: Path, source, timeout: float) -> None:
    try:
        cc.render_corpus_crop(render, str(out), palette_source=source, timeout=timeout,
                              threads=RENDER_THREADS)
    except BaseException:
        try:
            out.unlink(missing_ok=True)     # a TRUNCATED jpg reads as rendered forever
        except OSError:
            pass
        raise


def _render_one(job):
    row, crops, vivid, timeout = job
    iid, render = row["image_id"], row["render"]
    canon = crops / f"{iid}.jpg"
    if not canon.exists():
        _render_to(render, canon, PALETTE_SOURCE, timeout)
    vp = vivid / f"{iid}.jpg"
    if not vp.exists():
        _render_to(dict(render, palette=VIVID_PALETTE), vp, VIVID_SOURCE, timeout)
    return iid


def stage_render(args) -> int:
    deadline = (time.time() + args.max_minutes * 60.0) if args.max_minutes else None
    stopped, total = None, 0
    for batch_id in BATCHES:
        bdir = Path(cc.batch_dir(batch_id))
        if not (bdir / "images.jsonl").exists():
            print(f"  {batch_id}: no images.jsonl — run `draw` first.")
            continue
        rows = cc.read_jsonl(str(bdir / "images.jsonl"))
        crops, vivid = Path(cc.crops_dir(batch_id)), Path(cc.vivid_dir(batch_id))
        crops.mkdir(parents=True, exist_ok=True)
        vivid.mkdir(parents=True, exist_ok=True)

        def needs(r):
            return not (crops / f"{r['image_id']}.jpg").exists() or \
                not (vivid / f"{r['image_id']}.jpg").exists()
        todo = [r for r in rows if needs(r)]
        print(f"render {batch_id}: {len(rows)} rows, {len(todo)} need crops "
              f"= {2*len(todo)} renders, {args.workers}x{RENDER_THREADS} threads", flush=True)
        t0, done, fails = time.time(), 0, []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_render_one, (r, crops, vivid, args.render_timeout)): r
                    for r in todo}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:                       # noqa: BLE001
                    fails.append(dict(image_id=futs[fut]["image_id"],
                                      err=f"{type(e).__name__}: {str(e)[:200]}"))
                done += 1
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"  [{done}/{len(todo)}] {el:.0f}s  "
                          f"{done/max(el,1e-9)*60:.1f} row/min  "
                          f"ETA {(len(todo)-done)*el/max(done,1)/60:.1f} min "
                          f"({len(fails)} failed)", flush=True)
                if deadline and time.time() > deadline:
                    stopped = batch_id
                    for f2 in futs:
                        f2.cancel()
                    break
        total += done
        if fails:
            (bdir / "render_failures.json").write_text(
                json.dumps(dict(n=len(fails), by_class=dict(Counter(
                    f["err"].split(":")[0] for f in fails)), failures=fails), indent=2),
                encoding="utf-8")
            print(f"  !! {len(fails)} render failures -> {bdir/'render_failures.json'}")
        if stopped:
            break
    print(f"render: {total} rows this run"
          + (f"; STOPPED at the {args.max_minutes:g}-minute bound during {stopped}"
             if stopped else ""))
    for batch_id in BATCHES:
        bdir = Path(cc.batch_dir(batch_id))
        if not (bdir / "images.jsonl").exists():
            continue
        rows = cc.read_jsonl(str(bdir / "images.jsonl"))
        crops, vivid = Path(cc.crops_dir(batch_id)), Path(cc.vivid_dir(batch_id))
        miss = sum(1 for r in rows if not (crops / f"{r['image_id']}.jpg").exists()
                   or not (vivid / f"{r['image_id']}.jpg").exists())
        print(f"  {batch_id:36s} {len(rows)-miss:4d}/{len(rows):4d}"
              + ("  COMPLETE" if miss == 0 else "  INCOMPLETE — not labelable"))
    return 0


# =========================================================================== #
# verify
# =========================================================================== #
def stage_verify(args) -> int:
    L, ok = [], True

    def emit(s=""):
        L.append(s)
        print(s)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    emit(f"=== q4 long harvest batches — acceptance ({STAMP}) ===")
    all_ids = []
    expect = {RANKED: ("train", True), NEARMB: ("train", True), UNIFORM: ("eval", False)}
    for batch_id in BATCHES:
        bdir = Path(cc.batch_dir(batch_id))
        emit(f"\n[{batch_id}]")
        if not (bdir / "images.jsonl").exists():
            check("images.jsonl exists", False)
            continue
        rows = cc.read_jsonl(str(bdir / "images.jsonl"))
        blind = cc.read_jsonl(str(bdir / "blind.jsonl"))
        bj = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))
        split, biased, source = bm.assign_split({"batch": batch_id, "ft": "mandelbrot"})
        check("registered EXPLICITLY (not the fail-closed default)",
              source != "unregistered", f"-> {(split, biased, source)}")
        check("classification is what the leg's story says",
              (split, biased) == expect[batch_id], f"{(split, biased)}")
        check("batch.json records the same classification",
              bj["registration"]["assign_split"] == [split, biased, source])
        check("every label is null", all(r["label"]["score"] is None for r in rows))
        check("image_id is opaque `q4<slot>_<hash>`",
              all(r["image_id"].startswith("q4") and len(r["image_id"]) == 15
                  for r in rows))
        served = (bdir / "blind.jsonl").read_text(encoding="utf-8")
        leaked = [k for k in LEAK_KEYS if f'"{k}"' in served]
        check("no selection key reaches the served manifest", not leaked, str(leaked))
        check("served ids == analysis ids",
              {r["image_id"] for r in blind} == {r["image_id"] for r in rows})
        check("batch.json names blind.jsonl as served",
              bj.get("served_manifest") == "blind.jsonl")
        check("no calibration aids", bj.get("calibration_aids", "").startswith("NONE"))
        check("canonical crop recipe stamped",
              bj["render_recipe"]["path"] == cc.CANONICAL_CROP_RECIPE)
        # The invariant is per-cell, not a flat spread: a drained cell gave up everything it
        # had. `batch.json` carries the draw's own `taken/available/drained` record, which is
        # what makes this checkable on the built bytes rather than re-derivable.
        rep = (bj.get("sampling_metaparameters") or {}).get("cells") or {}
        okc, detail = cells_balanced(rep)
        check("cells balanced to +/-1 among non-drained cells", okc, detail)
        served_cells = Counter(r["provenance"].get("stratum") for r in rows)
        check("every drawn cell is present in the built batch",
              set(served_cells) == {k for k, v in rep.items() if v["taken"] > 0},
              f"{len(served_cells)} cells in bytes vs "
              f"{sum(1 for v in rep.values() if v['taken'] > 0)} drawn")
        if batch_id == UNIFORM:
            # The one property that makes this an instrument rather than a sample of a run.
            bad = [r for r in rows if any(
                r["provenance"].get(k) is not None
                for k in ("rank_score", "eord", "p_good", "canon_eord", "cheap_eord"))]
            check("NO score of any kind on an eval row", not bad, f"{len(bad)} rows")
        if batch_id == RANKED:
            fates = {r["provenance"].get("fate") for r in rows}
            check("rejects ARE in the ranked batch (the cutoff needs them)",
                  len(fates - {"admitted"}) > 0, str(sorted(f for f in fates if f)))
        crops, vivid = Path(cc.crops_dir(batch_id)), Path(cc.vivid_dir(batch_id))
        n_c = sum(1 for r in rows if (crops / f"{r['image_id']}.jpg").exists())
        n_v = sum(1 for r in rows if (vivid / f"{r['image_id']}.jpg").exists())
        check("every row has a canonical crop", n_c == len(rows), f"{n_c}/{len(rows)}")
        check("every row has a vivid companion", n_v == len(rows), f"{n_v}/{len(rows)}")
        all_ids += [r["image_id"] for r in rows]

    emit("\n[cross-batch]")
    check("image ids unique across all three batches",
          len(all_ids) == len(set(all_ids)),
          f"{len(all_ids) - len(set(all_ids))} duplicates")
    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    rep = paths.scratch("q4_long_harvest", "verify.txt")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {rep}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--run-dir", type=Path, required=True)
    q.set_defaults(fn=stage_queue)
    d = sub.add_parser("draw")
    d.add_argument("--run-dir", type=Path, required=True)
    d.add_argument("--n-chunk", type=int, default=N_CHUNK)
    d.set_defaults(fn=stage_draw)
    r = sub.add_parser("render")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--render-timeout", type=float, default=600.0)
    r.add_argument("--max-minutes", type=float, default=0.0)
    r.set_defaults(fn=stage_render)
    v = sub.add_parser("verify")
    v.set_defaults(fn=stage_verify)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 0) and a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
