#!/usr/bin/env python
r"""build_supply_crawl_batches.py — the supply crawl's label batches, through the corpus rig.

FOUR BATCHES, THREE GENERATION METHODS, ONE POPULATION. The population is every recorded
maneuver candidate of the crawl — **pushed and passed over alike**, because the whole point
of recording the passed-over rows is that the low bins supply the negative class. From it:

  * two STRATIFIED chunks of ~290, round-robin to +/-1 over (degree x operator x
    composite-v3 bin);
  * one UNIFORM leg of ~90, uniform over the population with no score anywhere in the
    selection;
  * one EXEMPLAR mini-chunk of ~60, top by exemplar similarity, its own registered method —
    the direct test of "closer to the exemplars = better".

DRAW ORDER IS LOAD-BEARING, and it is the reverse of the order the batches are described in.
The uniform leg is drawn FIRST, then the stratified chunks from what is left, then the
mini-chunk from what is left after those. The reason is that a location may appear in only
one batch (`build_manifest.load_post_freeze` asserts it), so the legs must be disjoint — and
if the uniform leg were drawn last, its exclusions would be exactly the rows a SCORE picked,
which makes "uniform" a score-dependent draw and destroys the one property it exists for.
Drawing it first costs the stratified chunks ~90 rows removed at random, which biases
nothing.

WHAT "ALL RECORDED CANDIDATES" MEANS HERE, precisely: every available candidate the run
SCREENED. A candidate the screen could not reach is excluded, and not as a convenience — its
64x36 frame already failed the f64 pixel-spacing guard, and a 1280x720 label crop of the same
view has spacing 20x FINER. There is no crop to label. The count of those rows is reported
rather than silently dropped.

THE STANDARD RIG, REUSED VERBATIM: `corpus_common.render_corpus_crop` (the only sanctioned
crop renderer), the score-3 palette roster with a seeded per-image draw, the `blue_orange`
vivid companion, opaque post-shuffle `image_id`s, and a `blind.jsonl` whose provenance is
reduced to batch identity. Every selection key — bin, composite, exemplar similarity,
operator, degree — is a LEAK KEY and is absent from the served manifest, not nulled in it.

  uv run python tools/atlas/build_supply_crawl_batches.py draw    --run-dir data/discovery/<run>
  uv run python tools/atlas/build_supply_crawl_batches.py render
  uv run python tools/atlas/build_supply_crawl_batches.py features --run-dir data/discovery/<run>
  uv run python tools/atlas/build_supply_crawl_batches.py verify
"""
from __future__ import annotations

import argparse
import json
import os
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

import paths                                   # noqa: E402
import corpus_common as cc                      # noqa: E402
import build_minibrot_batch as BMB              # noqa: E402  (coords / palettes / io)
import maneuver_inspection_sheet as mis         # noqa: E402  (the population loader)
import deep_center_finder as dcf                # noqa: E402  (the corpus crop cap policy)
from tools.v7 import build_manifest as bm       # noqa: E402  (assign_split — the authority)

STAMP = "2026-08-01"
GEN_VERSION = "supply_crawl_v1"
DRAW_SEED = 20260801
PRESENTATION_SEED = 0x5C0801

CROP_W, CROP_H, CROP_SS = BMB.CROP_W, BMB.CROP_H, BMB.CROP_SS
CROP_FILTER, INTERIOR_MODE, COMPOSITION = BMB.CROP_FILTER, BMB.INTERIOR_MODE, BMB.COMPOSITION
PALETTE_SOURCE, VIVID_PALETTE, VIVID_SOURCE = (BMB.PALETTE_SOURCE, BMB.VIVID_PALETTE,
                                               BMB.VIVID_SOURCE)

STRAT_A = f"{STAMP}_supply_crawl_strat_a_v1"
STRAT_B = f"{STAMP}_supply_crawl_strat_b_v1"
UNIFORM = f"{STAMP}_supply_crawl_uniform_v1"
EXEMPLAR = f"{STAMP}_supply_crawl_exemplar_v1"
BATCHES = (STRAT_A, STRAT_B, UNIFORM, EXEMPLAR)

N_STRAT, N_UNIFORM, N_EXEMPLAR = 290, 90, 60
N_BINS = 5                     # composite-v3 quintiles of the run's OWN distribution

DRAW_REL = f"data/supply_crawl/{STAMP}/draw.jsonl"
FEATURES_REL = f"data/supply_crawl/{STAMP}/features.jsonl"

# Keys that reveal WHY a row was chosen. Absent from the served manifest, and asserted
# absent by `verify` on the served BYTES — a nulled key still tells the labeler a selection
# happened and which axis it happened on.
LEAK_KEYS = ("selection_role", "stratum", "composite", "composite_bin", "exemplar_sim_max",
             "exemplar_sim_mean", "band_coverage", "band_coverage_q25", "radial_range",
             "radial_rings", "interior_fraction", "vetoed", "size_factor", "op", "k",
             "degree", "period", "atom_key", "used", "unused_reason")


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# =========================================================================== #
# population
# =========================================================================== #
def load_population(run_dir: Path) -> tuple[list[dict], dict]:
    """Every recorded, screened candidate of the run, deduped on `(atom_key, k)`.

    `mis.load_population` is the shared loader and does the dedup that the append-only log
    requires (a kill replays a batch, so the log is a superset of the counters). The
    exemplar similarity is joined on here rather than recomputed, so the number in the
    feature table and the number the mini-chunk selected on cannot differ.
    """
    log = run_dir / "maneuvers.jsonl"
    pop = mis.load_population([log])
    sim_path = run_dir / "exemplar_sim.jsonl"
    sim = {r["key"]: r for r in _read_jsonl(sim_path)} if sim_path.exists() else {}
    for r in pop:
        r["key"] = f"{r['atom_key']}|{r['k']}"
        s = sim.get(r["key"]) or {}
        r["exemplar_sim_max"] = s.get("exemplar_sim_max")
        r["exemplar_sim_mean"] = s.get("exemplar_sim_mean")
        r["exemplar_substrate"] = s.get("substrate")
    frames = Counter(r.get("screen_frame") for r in pop)
    if len(frames) > 1:
        raise SystemExit(f"population mixes screen frames {dict(frames)} — a 4x-atom and a "
                         f"view-frame score are different measurements and must not be "
                         f"binned together.")
    rep = dict(n=len(pop), sim_joined=sum(1 for r in pop if r["exemplar_sim_max"] is not None),
               screen_frame=next(iter(frames), None),
               with_composite=sum(1 for r in pop if r.get("composite") is not None))
    return pop, rep


def composite_bins(pop: list[dict], n_bins: int = N_BINS) -> list[float]:
    """Bin EDGES of `composite` over the run's own population.

    Quantiles of this run's distribution, not fixed cut-points, for the reason every score
    in this tree is ranked that way: an absolute field number means nothing across
    geometries and cap policies, only an ordering within one pair does. A vetoed row scores
    in [-1, 0) and therefore lands in bin 1 by arithmetic rather than by a special case.
    """
    v = np.asarray([r["composite"] for r in pop if r.get("composite") is not None],
                   dtype=float)
    if v.size == 0:
        return []
    return [float(np.percentile(v, 100.0 * i / n_bins)) for i in range(1, n_bins)]


def bin_of(r: dict, edges: list[float]) -> int:
    c = r.get("composite")
    if c is None:
        return 0                                  # "unscored" is its own cell, never bin 1
    return 1 + sum(1 for e in edges if float(c) > e)


# =========================================================================== #
# the three draws
# =========================================================================== #
def draw_uniform(pop: list[dict], n: int, seed: int) -> list[dict]:
    """Uniform over the WHOLE population. No score, no stratum, no exclusion.

    Drawn FIRST so that nothing a score touched can shape it (module doc). The sort before
    the shuffle is on `key` alone, so the draw does not inherit the log's write order —
    which is batch order, which correlates with depth.
    """
    import random
    pool = sorted(pop, key=lambda r: r["key"])
    random.Random(seed).shuffle(pool)
    return pool[:n]


def draw_stratified(pop: list[dict], n: int, edges: list[float], seed: int) -> list[dict]:
    """`n` rows round-robin to +/-1 over the (degree x operator x composite-bin) cells.

    Floor-then-remainder, the same shape as `view_screen_sheets.stratify` and
    `maneuver_inspection_sheet.stratify`: every non-empty cell gets one before any cell gets
    two, so the chunk cannot become one operator's or one bin's showreel. Within a cell the
    order is a seeded shuffle, so which member of a cell is taken is not the log's order.

    THE POINT OF STRATIFYING ON THE BIN is the negative class. A draw proportional to the
    population would put ~none of the bottom bin in a 290-row chunk relative to how much of
    the LABEL signal lives there; round-robin over bins buys the low bins the same footing
    as the high ones, which is what makes the labels usable for a fit rather than only for a
    ranking.
    """
    import random
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in pop:
        cells[(r.get("degree"), r.get("op"), bin_of(r, edges))].append(r)
    for c in cells.values():
        c.sort(key=lambda r: r["key"])
        rng.shuffle(c)
    keys = sorted(cells, key=lambda k: (str(k[0]), str(k[1]), k[2]))
    take = {k: 0 for k in keys}
    while sum(take.values()) < n:
        cand = [k for k in keys if take[k] < len(cells[k])]
        if not cand:
            break
        k = min(cand, key=lambda k: (take[k], -len(cells[k])))
        take[k] += 1
    out = []
    for k in keys:
        out.extend(cells[k][:take[k]])
    return out


def draw_exemplar(pop: list[dict], n: int) -> list[dict]:
    """Top `n` by exemplar similarity (`max` over the set). The ONE selection on this axis.

    `max` and not `mean`: the hypothesis is "looks like one of the ones he liked", and the
    exemplar set deliberately holds one row per family, so a `mean` would rank a row by how
    generically fractal it is rather than by proximity to any particular liked thing.
    """
    have = [r for r in pop if r.get("exemplar_sim_max") is not None]
    return sorted(have, key=lambda r: (-float(r["exemplar_sim_max"]), r["key"]))[:n]


def draw_all(pop: list[dict], *, seed: int = DRAW_SEED) -> tuple[dict, dict]:
    edges = composite_bins(pop)
    taken: set = set()

    def remaining():
        return [r for r in pop if r["key"] not in taken]

    uni = draw_uniform(pop, N_UNIFORM, seed)
    taken.update(r["key"] for r in uni)
    a = draw_stratified(remaining(), N_STRAT, edges, seed + 1)
    taken.update(r["key"] for r in a)
    b = draw_stratified(remaining(), N_STRAT, edges, seed + 2)
    taken.update(r["key"] for r in b)
    ex = draw_exemplar(remaining(), N_EXEMPLAR)
    taken.update(r["key"] for r in ex)

    chunks = {UNIFORM: uni, STRAT_A: a, STRAT_B: b, EXEMPLAR: ex}
    for bid, rows in chunks.items():
        for r in rows:
            r["batch_id"] = bid
            r["composite_bin"] = bin_of(r, edges)
    rep = dict(edges=[round(e, 6) for e in edges],
               counts={bid: len(rows) for bid, rows in chunks.items()},
               overlap=sum(len(rows) for rows in chunks.values()) - len(taken))
    return chunks, rep


# =========================================================================== #
# the corpus batch
# =========================================================================== #
# THE CROP CAP IS THE CORPUS'S, NOT THE SCREEN'S OR THE DEPLOY HEAD'S. Every sibling batch
# builder renders label crops under `deep_center_finder._maxiter_for_fw` (~1500 iters per
# decade of depth, floored 3000, capped 40000) — checkable on any existing row:
# `2026-07-26_minibrot_roster_v2` at fw 2.116e-04 carries maxiter 5512, which is exactly
# `round(1500 * 3.674)`. Using `active_ckpt.auto_maxiter` instead looked defensible (it is
# the LIVE deploy policy) and was wrong twice over: it makes these crops incomparable with
# every other batch in the corpus, and at the v9 raised cap it put the median crop at
# maxiter 12,642 against ~3,000 — a ~3x render bill for a picture the labeler cannot tell
# apart. Caught by rendering six crops and timing them, not by reading the constant.
def _render_block(r: dict, palette: str) -> dict:
    fw = cc.hp_str(r["fw"])
    render = cc.render_block(cx=str(r["cx"]), cy=str(r["cy"]), fw=fw,
                             maxiter=int(dcf._maxiter_for_fw(float(r["fw"]))),
                             palette=palette, composition=COMPOSITION, width=CROP_W,
                             height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE)
    render["fractal_type"] = r.get("partition") or "mandelbrot"
    render["c_re"] = None
    render["c_im"] = None
    return render


def write_batch(batch_id: str, rows: list[dict], *, role: str, purpose: str,
                sampling: dict) -> dict:
    """One corpus batch: `images.jsonl` (analysis side) + `blind.jsonl` (served) + batch.json."""
    names = BMB._palette_names()
    order = list(range(len(rows)))
    np.random.default_rng(PRESENTATION_SEED ^ BMB._stable_seed(batch_id)).shuffle(order)
    for slot, oi in enumerate(order):
        h = BMB._stable_seed(rows[oi]["key"])
        rows[oi]["image_id"] = f"sc{slot:04d}_{h:08x}"
    rows.sort(key=lambda r: r["image_id"])

    full, blind = [], []
    for r in rows:
        pal = names[BMB._stable_seed(r["image_id"]) % len(names)]
        render = _render_block(r, pal)
        prov = cc.provenance_block(
            GEN_VERSION, batch_id, family=r.get("partition"),
            selection_role=role,
            stratum=(f"d{r.get('degree')}|{r.get('op')}|bin{r.get('composite_bin')}"
                     if role == "supply_crawl_stratified" else role),
            composite=r.get("composite"), composite_bin=r.get("composite_bin"),
            vetoed=r.get("vetoed"), size_factor=r.get("size_factor"),
            band_coverage=r.get("band_coverage"),
            band_coverage_q25=r.get("band_coverage_q25"),
            radial_range=r.get("radial_range"), radial_rings=r.get("radial_rings"),
            interior_fraction=r.get("interior_fraction"),
            exemplar_sim_max=r.get("exemplar_sim_max"),
            exemplar_sim_mean=r.get("exemplar_sim_mean"),
            exemplar_substrate=r.get("exemplar_substrate"),
            screen_frame=r.get("screen_frame"), screen_policy=r.get("screen_policy"),
            op=r.get("op"), k=r.get("k"), degree=r.get("degree"), period=r.get("period"),
            log10_abs_A=r.get("log10_abs_A"), window_scale=r.get("window_scale"),
            parent_depth=r.get("parent_depth"), atom_key=r.get("atom_key"),
            candidate_key=r.get("key"), used=r.get("used"),
            unused_reason=r.get("unused_reason"),
            descend_mode=f"maneuver_{r.get('op')}_k{r.get('k')}")
        full.append(cc.make_row(r["image_id"], render, prov, cc.label_block()))
        # Served row: batch identity only. Every selection axis is ABSENT, not null — a
        # nulled key still names the axis a selection happened on.
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
                          NOTE=("registered in tools/v7/build_manifest BEFORE the build. "
                                "The fail-closed default would also have landed this "
                                "train-side, but 'nobody registered it' and 'this is a "
                                "biased train draw' are different facts.")),
        render_defaults=dict(width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE, composition=COMPOSITION,
                             palette_roster="data/palettes/score3_colormaps.json",
                             vivid_companion=VIVID_PALETTE,
                             maxiter="deep_center_finder._maxiter_for_fw(fw) — the cap "
                                     "every other corpus batch builder renders crops under"),
        render_recipe=cc.render_recipe_stamp(PALETTE_SOURCE),
        sampling_metaparameters=sampling,
    )
    (bdir / "batch.json").write_text(json.dumps(bj, indent=2), encoding="utf-8")
    if not (bdir / "scores.json").exists():
        (bdir / "scores.json").write_text("{}", encoding="utf-8")
    return dict(batch_id=batch_id, n=len(full), dir=str(bdir),
                assign_split=[split, biased, source])


# =========================================================================== #
# stages
# =========================================================================== #
def stage_draw(args) -> int:
    pop, poprep = load_population(args.run_dir)
    print(f"[pop] {poprep['n']} recorded+screened candidates "
          f"(frame={poprep['screen_frame']}, composite on {poprep['with_composite']}, "
          f"exemplar sim joined on {poprep['sim_joined']})")
    if poprep["sim_joined"] == 0:
        print("  !! no exemplar_sim.jsonl — run tools/atlas/exemplar_similarity.py first; "
              "the mini-chunk cannot be drawn.", file=sys.stderr)
    chunks, rep = draw_all(pop, seed=args.seed)
    if rep["overlap"]:
        raise SystemExit(f"draws overlap by {rep['overlap']} rows — a location may appear "
                         f"in only one batch (build_manifest.load_post_freeze asserts it).")

    common = dict(population=("every recorded maneuver candidate of the crawl that the "
                              "screen reached — pushed AND passed over; a candidate the "
                              "64x36 screen could not reach has no renderable crop"),
                  population_n=poprep["n"], draw_seed=args.seed,
                  composite_bin_edges=rep["edges"],
                  draw_order=("uniform FIRST, then the two stratified chunks, then the "
                              "exemplar mini-chunk — so the uniform leg's selection is "
                              "never conditioned on a score"))
    out = []
    out.append(write_batch(UNIFORM, chunks[UNIFORM], role="supply_crawl_uniform",
                           purpose=("uniform draw over the crawl's recorded candidates with "
                                    "NO score in the selection: the leg whose label "
                                    "distribution estimates the crawl's own base rate."),
                           sampling=dict(**common, selection="uniform, unstratified",
                                         g_used_for_selection=False)))
    for bid, seedoff in ((STRAT_A, 1), (STRAT_B, 2)):
        out.append(write_batch(
            bid, chunks[bid], role="supply_crawl_stratified",
            purpose=("stratified draw over (degree x operator x composite-v3 bin), "
                     "round-robin to +/-1, drawn from pushed AND passed-over candidates so "
                     "the low bins supply the negative class."),
            sampling=dict(**common, selection="round-robin over (degree x operator x bin)",
                          n_bins=N_BINS, g_used_for_selection=True,
                          chunk_seed=args.seed + seedoff)))
    out.append(write_batch(
        EXEMPLAR, chunks[EXEMPLAR], role="supply_crawl_exemplar",
        purpose=("top by exemplar similarity, excluding rows already in the other chunks. "
                 "The direct test of 'closer to the exemplars = better'; registered as its "
                 "own biased method so the answer reads AGAINST the stratified chunks."),
        sampling=dict(**common, selection="top-N by exemplar_sim_max",
                      g_used_for_selection=True,
                      substrate=(chunks[EXEMPLAR][0].get("exemplar_substrate")
                                 if chunks[EXEMPLAR] else None))))

    dp = paths.durable(DRAW_REL, mkparents=True)
    with open(dp, "w", encoding="utf-8") as f:
        for bid in BATCHES:
            for r in chunks[bid]:
                f.write(json.dumps({k: v for k, v in r.items()}, default=str) + "\n")
    for o in out:
        print(f"  {o['batch_id']:42s} n={o['n']:4d}  assign_split={o['assign_split']}")
    print(f"  bins: edges {rep['edges']}")
    for bid in BATCHES:
        c = Counter(r["composite_bin"] for r in chunks[bid])
        print(f"    {bid[-18:]:18s} bin mix {dict(sorted(c.items()))}")
    print(f"  draw -> {dp}")
    return 0


# RENDER THREADS, and why not the single-process default. `DEFAULT_ENGINE_THREADS` is 7 and
# is documented as the number for ONE engine process; four workers of seven is 28 threads on
# a 12-core box, which is oversubscription, not throughput. Sized for the actual N instead,
# as `corpus_common` says to: 4 x 3 = 12.
RENDER_THREADS = 3

# The order the crops are rendered in, and it is a BUDGET decision rather than a preference.
# 730 rows is 1,460 renders at 1280x720 ss4, measured at ~500 core-seconds each — call it 17
# core-hours, which does not fit inside any session that also had to produce the run. So the
# smallest and most decision-relevant legs go first: the exemplar mini-chunk is the only leg
# that TESTS something (60 rows, ~1.4 h), the uniform leg is the only one whose label
# distribution estimates a base rate (90 rows), and the two stratified chunks — which are
# the bulk and are the most robust to being partly labelled — go last. A partly-rendered
# batch is not labelable, so the order is chosen to leave WHOLE batches finished.
RENDER_ORDER = (EXEMPLAR, UNIFORM, STRAT_A, STRAT_B)


def _render_to(render: dict, out: Path, source, timeout: float) -> None:
    """One crop, and a HALF-WRITTEN one is deleted rather than left behind.

    `render_corpus_crop` raises when the engine exits non-zero or writes nothing — but a
    `timeout` kills the engine mid-write, and the truncated JPG that leaves is the one
    failure mode this pipeline cannot see: `needs()` checks existence, so the row reads as
    rendered forever and the batch is quietly one bad crop short. Deleting on the way out
    turns it back into a missing crop, which the resume re-renders and `verify` counts.
    """
    try:
        cc.render_corpus_crop(render, str(out), palette_source=source, timeout=timeout,
                              threads=RENDER_THREADS)
    except BaseException:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _render_one(job):
    row, crops, vivid, timeout = job
    iid, render = row["image_id"], row["render"]
    made = []
    canon = crops / f"{iid}.jpg"
    if not canon.exists():
        _render_to(render, canon, PALETTE_SOURCE, timeout)
        made.append("canon")
    vp = vivid / f"{iid}.jpg"
    if not vp.exists():
        _render_to(dict(render, palette=VIVID_PALETTE), vp, VIVID_SOURCE, timeout)
        made.append("vivid")
    return iid, made


def stage_render(args) -> int:
    deadline = (time.time() + args.max_minutes * 60.0) if args.max_minutes else None
    total, stopped = 0, None
    for batch_id in (RENDER_ORDER if not args.only else [b for b in RENDER_ORDER
                                                         if args.only in b]):
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
              f"= {2*len(todo)} renders, workers={args.workers}x{RENDER_THREADS} threads",
              flush=True)
        t0, done, fails = time.time(), 0, []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_render_one, (r, crops, vivid, args.render_timeout)): r
                    for r in todo}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:                        # noqa: BLE001
                    fails.append(dict(image_id=futs[fut]["image_id"],
                                      err=f"{type(e).__name__}: {str(e)[:200]}"))
                done += 1
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    # Reprojected from the run's OWN observed rate, never restated from a
                    # pre-run estimate (`CLAUDE.md`, projecting a long run's wall clock).
                    print(f"  [{done}/{len(todo)}] {el:.0f}s  {done/max(el,1e-9)*60:.1f} "
                          f"row/min  ETA {(len(todo)-done)*el/max(done,1)/60:.1f} min "
                          f"({len(fails)} failed)", flush=True)
                if deadline and time.time() > deadline:
                    # Cancel what has not started; in-flight renders finish (killing one
                    # leaves a truncated JPG that `needs()` would then call present).
                    stopped = batch_id
                    for f2 in futs:
                        f2.cancel()
                    break
        total += done
        if fails:
            # The WHOLE failure population, classed — never a truncated head (`CLAUDE.md`).
            (bdir / "render_failures.json").write_text(
                json.dumps(dict(n=len(fails), by_class=dict(Counter(
                    f["err"].split(":")[0] for f in fails)), failures=fails), indent=2),
                encoding="utf-8")
            print(f"  !! {len(fails)} render failures -> {bdir/'render_failures.json'}")
        if stopped:
            break
    # NO SILENT CAP. What was dropped is named, per batch, or a partial render reads as
    # "covered everything" to anyone looking at the batch dirs.
    left = {}
    for batch_id in BATCHES:
        bdir = Path(cc.batch_dir(batch_id))
        if not (bdir / "images.jsonl").exists():
            continue
        rows = cc.read_jsonl(str(bdir / "images.jsonl"))
        crops, vivid = Path(cc.crops_dir(batch_id)), Path(cc.vivid_dir(batch_id))
        miss = sum(1 for r in rows
                   if not (crops / f"{r['image_id']}.jpg").exists()
                   or not (vivid / f"{r['image_id']}.jpg").exists())
        left[batch_id] = dict(rows=len(rows), missing=miss, complete=miss == 0)
    print(f"render: {total} rows this run"
          + (f"; STOPPED at the {args.max_minutes:g}-minute bound during {stopped}"
             if stopped else ""))
    for b, v in left.items():
        print(f"  {b:42s} {v['rows'] - v['missing']:4d}/{v['rows']:4d} crops"
              + ("  COMPLETE" if v["complete"] else "  INCOMPLETE — not labelable"))
    rs = paths.scratch("supply_crawl", "render_state.json")
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(
        json.dumps(dict(stopped_during=stopped, per_batch=left,
                        NOTE=("resumable: re-run `render`, it skips crops that exist. A "
                              "batch marked INCOMPLETE must not be queued for labeling.")),
                   indent=2) + "\n", encoding="utf-8")
    return 0


def stage_features(args) -> int:
    """One tidy file joining every batched row to its recorded features, keyed by batch id.

    Written from the BATCH rows (not from the draw file), so a row that never made it into a
    batch cannot appear in the feature table, and the join key is the `image_id` the labels
    will come back under. The post-label linear fit is then a read of this file and nothing
    else.
    """
    pop, _ = load_population(args.run_dir)
    by_key = {r["key"]: r for r in pop}
    out, missing = [], 0
    for batch_id in BATCHES:
        bdir = Path(cc.batch_dir(batch_id))
        if not (bdir / "images.jsonl").exists():
            continue
        for row in cc.read_jsonl(str(bdir / "images.jsonl")):
            pv = row["provenance"]
            r = by_key.get(pv.get("candidate_key")) or {}
            if not r:
                missing += 1
            rd = row["render"]
            out.append(dict(
                image_id=row["image_id"], batch_id=batch_id,
                selection_role=pv.get("selection_role"), stratum=pv.get("stratum"),
                candidate_key=pv.get("candidate_key"),
                # --- screen measures (VIEW frame; the frame is on the row) ---
                screen_frame=r.get("screen_frame"), composite=r.get("composite"),
                composite_bin=pv.get("composite_bin"), vetoed=r.get("vetoed"),
                size_factor=r.get("size_factor"),
                band_coverage=r.get("band_coverage"),
                band_coverage_q25=r.get("band_coverage_q25"),
                radial_range=r.get("radial_range"), radial_rings=r.get("radial_rings"),
                interior_fraction=r.get("interior_fraction"),
                cap_headroom=r.get("cap_headroom"), clamped=r.get("clamped"),
                screen_policy=r.get("screen_policy"),
                # --- geometry / provenance ---
                degree=r.get("degree"), period=r.get("period"),
                log10_abs_A=r.get("log10_abs_A"), window_scale=r.get("window_scale"),
                fw=float(rd["fw"]), operator=r.get("op"), k=r.get("k"),
                parent_depth=r.get("parent_depth"), partition=r.get("partition"),
                pushed=r.get("used"), unused_reason=r.get("unused_reason"),
                # --- the recorded, never-ordering feature ---
                exemplar_sim_max=r.get("exemplar_sim_max"),
                exemplar_sim_mean=r.get("exemplar_sim_mean"),
                exemplar_substrate=r.get("exemplar_substrate"),
            ))
    fp = paths.durable(FEATURES_REL, mkparents=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"features: {len(out)} rows ({missing} without a population match) -> {fp}")
    return 0


def stage_sheets(args) -> int:
    """Three sheets for Matt: the unstratified top-N, and stratified Q5 / Q1.

    THE TWO ANSWER DIFFERENT QUESTIONS AND NEITHER SUBSTITUTES FOR THE OTHER — the lesson
    the v4 iteration wrote into `view_screen_sheets`. The stratified sheet exists so one
    operator or one degree cannot own the page, which means most of its tiles are pulled
    from thin cells and reading it as "the best 18" judges a sampling rule as if it were a
    ranking. So the rank inside the quintile is stamped on every stratified tile, and the
    straight top-N is built beside it: THAT is what the crawl's output looks like.

    Rendered through `view_screen_sheets.render_vivid` — the same vivid path the view-screen
    sheets used, not the corpus crop path — because these are for looking at, and the crops
    are for labeling.
    """
    import view_screen_sheets as vss
    pop, poprep = load_population(args.run_dir)
    ok = [r for r in pop if r.get("composite") is not None]
    print(f"[sheets] {len(ok)} scored of {poprep['n']} recorded")
    if not ok:
        print("  nothing scored — no sheets.")
        return 1
    nq, edges = vss.quintile_index([r["composite"] for r in ok])
    for r, q in zip(ok, nq):
        r["new_quintile"] = q
    q5 = sorted([r for r in ok if r["new_quintile"] == 5], key=lambda r: -r["composite"])
    q1 = [r for r in ok if r["new_quintile"] == 1]
    for i, r in enumerate(q5, 1):
        r["_rank"], r["_qn"] = i, len(q5)
    top = q5[:args.tiles]
    strat5 = vss.stratify(q5, args.tiles, args.seed)
    strat1 = vss.stratify(q1, args.tiles, args.seed + 1)
    out = paths.scratch("supply_crawl", "sheets")
    out.mkdir(parents=True, exist_ok=True)
    vivid = out / "vivid"
    vivid.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in {id(x): x for x in top + strat5 + strat1}.values():
        r["_png"] = vivid / f"{vss._tag(r['key'])}.jpg"
        jobs.append((r, r["_png"]))
    print(f"[sheets] rendering {len(jobs)} vivid tiles ({vss.WORKERS} processes)")
    vss.render_all(jobs)

    def lines(r, rank=True):
        c = float(r["composite"])
        head = f"{r['op'].split('_')[0]} k={r['k']} d{r.get('degree')} p{r.get('period')}"
        if rank and r.get("_rank"):
            head += f"  #{r['_rank']}/{r['_qn']}"
        return [(head, vss.INK),
                (f"comp {c:+.3f}  cov {r.get('band_coverage')}/{r.get('band_coverage_q25')}"
                 f"  int {r.get('interior_fraction')}", vss.DIM),
                (f"fw {float(r['fw']):.3g}  rng {r.get('radial_range')} "
                 f"rings {r.get('radial_rings')}"
                 + ("  VETOED" if r.get("vetoed") else ""),
                 vss.WARM if r.get("vetoed") else vss.DIM)]

    vss.build_sheet([(r["_png"], lines(r)) for r in top],
                    f"supply crawl — UNSTRATIFIED top {len(top)} by composite_v3 "
                    f"(of {len(ok)} scored)", out / "sheet_top.png")
    vss.build_sheet([(r["_png"], lines(r)) for r in strat5],
                    f"supply crawl — Q5 STRATIFIED over operator x degree "
                    f"(|Q5|={len(q5)}; rank shown, most tiles are not the top ranks)",
                    out / "sheet_q5_stratified.png")
    vss.build_sheet([(r["_png"], lines(r, rank=False)) for r in strat1],
                    f"supply crawl — Q1 STRATIFIED over operator x degree (|Q1|={len(q1)})",
                    out / "sheet_q1_stratified.png")
    (out / "readout.json").write_text(json.dumps(dict(
        n_scored=len(ok), n_recorded=poprep["n"],
        quintile_edges=[round(e, 6) for e in edges],
        vetoed=sum(1 for r in ok if r.get("vetoed")),
        vetoed_in_q1=sum(1 for r in q1 if r.get("vetoed")),
        operator_mix_q5=dict(Counter(r["op"] for r in q5)),
        operator_mix_pop=dict(Counter(r["op"] for r in ok)),
        k_mix_q5=dict(Counter(str(r["k"]) for r in q5)),
        k_mix_pop=dict(Counter(str(r["k"]) for r in ok)),
        degree_mix_q5=dict(Counter(str(r.get("degree")) for r in q5)),
        degree_mix_pop=dict(Counter(str(r.get("degree")) for r in ok)),
        CAVEAT=("operator, degree and k are confounded — the operators reach degrees at "
                "different rates and a k is only available where the framing cleared the "
                "spacing wall. Read a mix as composition, not as an effect."),
    ), indent=2) + "\n", encoding="utf-8")
    print(f"  sheets -> {out}")
    return 0


def stage_verify(args) -> int:
    L, ok = [], True

    def emit(s=""):
        L.append(s)
        print(s)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    emit(f"=== supply-crawl batches — acceptance ({STAMP}) ===")
    all_ids, all_keys = [], []
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
        check("registered EXPLICITLY in assign_split (not the fail-closed default)",
              source != "unregistered", f"-> {(split, biased, source)}")
        check("classified TRAIN-side", split == "train")
        exp_biased = batch_id != UNIFORM
        check(f"biased == {exp_biased}", biased is exp_biased)
        check("batch.json records the same classification",
              bj["registration"]["assign_split"] == [split, biased, source])
        check("every label is null (nothing is labeled)",
              all(r["label"]["score"] is None for r in rows))
        check("image_id is opaque `sc<slot>_<hash>`",
              all(len(r["image_id"]) == 15 and r["image_id"].startswith("sc")
                  for r in rows))
        served = (bdir / "blind.jsonl").read_text(encoding="utf-8")
        leaked = [k for k in LEAK_KEYS if f'"{k}"' in served]
        check("no selection key reaches the served manifest", not leaked, str(leaked))
        check("served ids == analysis ids",
              {r["image_id"] for r in blind} == {r["image_id"] for r in rows})
        check("served labels are null", all(r["label"]["score"] is None for r in blind))
        check("batch.json names blind.jsonl as served",
              bj.get("served_manifest") == "blind.jsonl")
        check("canonical crop recipe stamped",
              bj["render_recipe"]["path"] == cc.CANONICAL_CROP_RECIPE)
        crops, vivid = Path(cc.crops_dir(batch_id)), Path(cc.vivid_dir(batch_id))
        n_c = sum(1 for r in rows if (crops / f"{r['image_id']}.jpg").exists())
        n_v = sum(1 for r in rows if (vivid / f"{r['image_id']}.jpg").exists())
        check("every row has a canonical crop", n_c == len(rows), f"{n_c}/{len(rows)}")
        check("every row has a vivid companion", n_v == len(rows), f"{n_v}/{len(rows)}")
        all_ids += [r["image_id"] for r in rows]
        all_keys += [r["provenance"]["candidate_key"] for r in rows]

    emit("\n[cross-batch]")
    check("no candidate appears in two batches (build_manifest asserts one batch per loc)",
          len(all_keys) == len(set(all_keys)),
          f"{len(all_keys) - len(set(all_keys))} duplicates")
    fp = paths.durable(FEATURES_REL)
    if fp.exists():
        feat = _read_jsonl(fp)
        check("feature table covers every batched row",
              {f["image_id"] for f in feat} == set(all_ids),
              f"{len(feat)} feature rows vs {len(all_ids)} batched")
        check("every feature row carries the exemplar similarity",
              all(f.get("exemplar_sim_max") is not None for f in feat),
              f"{sum(1 for f in feat if f.get('exemplar_sim_max') is None)} missing")
    else:
        check("feature table exists", False, str(fp))
    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    rep = paths.scratch("supply_crawl", "verify.txt")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--run-dir", type=Path, required=True)
    d.add_argument("--seed", type=int, default=DRAW_SEED)
    d.set_defaults(fn=stage_draw)
    r = sub.add_parser("render")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--render-timeout", type=float, default=600.0)
    r.add_argument("--max-minutes", type=float, default=0.0,
                   help="stop starting new rows after this many minutes (0 = no bound). "
                        "In-flight renders finish; what is left is named per batch and "
                        "written to scratch/supply_crawl/render_state.json.")
    r.add_argument("--only", type=str, default=None,
                   help="restrict to batches whose id contains this substring")
    r.set_defaults(fn=stage_render)
    f = sub.add_parser("features")
    f.add_argument("--run-dir", type=Path, required=True)
    f.set_defaults(fn=stage_features)
    s = sub.add_parser("sheets")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--tiles", type=int, default=18)
    s.add_argument("--seed", type=int, default=DRAW_SEED)
    s.set_defaults(fn=stage_sheets)
    v = sub.add_parser("verify")
    v.set_defaults(fn=stage_verify)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 0) and a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
