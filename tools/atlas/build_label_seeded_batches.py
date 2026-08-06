#!/usr/bin/env python
r"""build_label_seeded_batches.py — the label-seeded harvest's queue and its label chunks.

FOUR STAGES, ONE POPULATION. The population is every candidate `label_seeded_harvest`
KEPT — screened at the view frame and past the interior pre-filter. From it:

  * `queue`  — score every candidate with `view_fit_v1.1` (the fitted, family-free,
               exemplar-free ordering score) and write the whole ranked queue. `composite_v3`
               is recorded beside every row for later comparison; it remains the live sort
               key everywhere else and nothing here flips it.
  * `draw`   — two chunks of ~290 FROM THE TOP of that queue, each stratified round-robin to
               +/-1 over the (method x degree) cells.
  * `render` — the standard corpus crop pair (canonical + vivid companion).
  * `verify` — the acceptance battery.

THE DRAW IS "TOP OF THE QUEUE" AND "BALANCED", AND THOSE PULL AGAINST EACH OTHER. The
resolution is that the round-robin is over CELLS while the order INSIDE a cell is the fitted
rank: every non-empty (method x degree) cell gives up its best remaining row before any cell
gives up its second. So the chunk is the top of the queue *conditioned on* not letting one
method or one degree own the page. This is the opposite of the supply crawl's stratified
draw, which shuffled within a cell — that chunk existed to give the NEGATIVE class footing
across composite bins, and this one exists to hand Matt the most promising material. Two
different jobs, and the within-cell order is where the difference lives.

CHUNK B IS THE SECOND PASS, NOT A SECOND SAMPLE. `draw_ranked_stratified` is called once for
2 x N and the interleaved result is split, so chunk A holds each cell's better rows and B the
next ones. Splitting that way rather than drawing B from what A left keeps both chunks
balanced on the same cells; a location may appear in only one batch either way
(`build_manifest.load_post_freeze` asserts it) and `verify` checks it on the built bytes.

THE STANDARD RIG, REUSED VERBATIM: `corpus_common.render_corpus_crop`, the score-3 palette
roster with a seeded per-image draw, the `blue_orange` vivid companion, opaque post-shuffle
`image_id`s, `deep_center_finder._maxiter_for_fw` as the crop cap (the cap every other
corpus batch builder renders under), and a `blind.jsonl` whose provenance is batch identity
and nothing else. Every selection key — the fitted score, the rank, the composite, the
method, the degree — is a LEAK KEY, ABSENT from the served manifest rather than nulled in it.

  uv run python tools/atlas/build_label_seeded_batches.py queue  --run-dir data/discovery/<run>
  uv run python tools/atlas/build_label_seeded_batches.py draw
  uv run python tools/atlas/build_label_seeded_batches.py render --max-minutes 120
  uv run python tools/atlas/build_label_seeded_batches.py verify
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus"),
           str(ROOT / "tools" / "sourcing"), str(ROOT / "tools" / "scoring"),
           str(ROOT / "tools" / "orbital")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                    # noqa: E402
import apportion                                # noqa: E402  (THE apportionment rules)
import corpus_common as cc                      # noqa: E402
import build_minibrot_batch as BMB              # noqa: E402  (coords / palettes / io)
import deep_center_finder as dcf                # noqa: E402  (the corpus crop cap policy)
import view_fit as vf                           # noqa: E402  (the v1.1 ordering score)
import view_field_cache as vfc                  # noqa: E402
import label_seeded_harvest as lsh              # noqa: E402
from tools.v7 import build_manifest as bm       # noqa: E402  (assign_split — the authority)

STAMP = lsh.STAMP
GEN_VERSION = lsh.GEN_VERSION
PRESENTATION_SEED = 0x15E020802

CROP_W, CROP_H, CROP_SS = BMB.CROP_W, BMB.CROP_H, BMB.CROP_SS
CROP_FILTER, INTERIOR_MODE, COMPOSITION = BMB.CROP_FILTER, BMB.INTERIOR_MODE, BMB.COMPOSITION
PALETTE_SOURCE, VIVID_PALETTE, VIVID_SOURCE = (BMB.PALETTE_SOURCE, BMB.VIVID_PALETTE,
                                               BMB.VIVID_SOURCE)

CHUNK_A = f"{STAMP}_label_seeded_v2_a"
CHUNK_B = f"{STAMP}_label_seeded_v2_b"
BATCHES = (CHUNK_A, CHUNK_B)
N_CHUNK = 290

# THREE ARTIFACTS, THREE STORAGE CLASSES, and the split is the point (`storage_classes.md`).
#   `seeds`   — DURABLE. A snapshot of which corpus locations resolved to >= 3 ON THIS DATE
#               through the amendment overlay as it then stood. The overlay moves, so this
#               is not re-derivable later; it is the record of what the harvest was seeded
#               from. Written by `label_seeded_harvest`. 132 KB.
#   `features`— DURABLE. The join between a label that does not exist yet and what the
#               candidate carried when it was drawn, keyed on the `image_id` the labels come
#               back under. 580 rows, 908 KB — the supply crawl's reason, verbatim.
#   `queue`   — BULK, out-of-tree, and it lives under the RUN's own `scratch/` rather than
#               beside the other two. 2,584 rows / 3.3 MB, a deterministic function of
#               `candidates.jsonl` (tracked) + `view_fit_v1_1.json` (tracked) + committed
#               code; the only untracked input is the field cache, and re-screening the
#               population's recorded geometry costs ~3 min of engine time. It was
#               `durable()` first and the repo-size guard refused it — correctly. Putting it
#               at `data/discovery/<run>/scratch/` means `artifacts._is_discovery_scratch`
#               relocates it with NO registry edit, which is the conservative default that
#               family exists to provide; a new prefix under `data/label_seeded_harvest/`
#               would have needed one and would have relocated the two durable files with it.
#               The ~290 x 2 rows that matter are in `features`, which is what a later reader
#               actually wants.
FEATURES_REL = f"data/label_seeded_harvest/{STAMP}/features.jsonl"


def queue_path(run_dir) -> Path:
    """The ranked queue, beside the run that regenerates it and out of the source tree."""
    rel = Path(run_dir).resolve().relative_to(ROOT).as_posix()
    return paths.bulk(f"{rel}/scratch/queue.jsonl")

# The screen frame every row here was measured on. One value, because this harvest has
# exactly one screen path (`maneuver_view_screen.screen_view`); a mixture would be the
# cap/geometry error `orbital_field_metrics.md` §5 and §7 forbid, and the supply crawl's
# loader refuses one outright.
SCREEN_FRAME = "view"

# Absent from the served manifest, and asserted absent by `verify` on the served BYTES — a
# nulled key still tells the labeler a selection happened and names the axis it happened on.
LEAK_KEYS = ("selection_role", "stratum", "composite", "fit_score", "fit_model",
             "queue_rank", "method", "band_coverage", "band_coverage_q25", "radial_range",
             "radial_rings", "interior_fraction", "vetoed", "size_factor", "op", "k",
             "degree", "period", "atom_key", "candidate_key", "seed_id", "seed_batch_id",
             "seed_image_id", "seed_score", "falloff_rate", "falloff_half",
             "log10_size_rel", "cap_headroom", "clamped", "parent_atom_id",
             "scale_ratio_decades", "f64_margin_deploy_decades", "log10_abs_A",
             "window_scale", "screen_frame", "screen_policy")


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# =========================================================================== #
# queue
# =========================================================================== #
def _features_for(r: dict, field) -> dict:
    """The `view_fit` feature dict for one harvest candidate.

    `view_fit.row_features` is not reused directly because it reads the supply crawl's
    column names (`rec["composite"]`, `rec["window_scale"]`, ...) off a differently-shaped
    row and additionally requires the two exemplar columns, which this harvest deliberately
    does not compute. Every arithmetic definition below is copied from it verbatim so a
    feature means the same thing in both — that is the property the fit depends on, and it
    is pinned by `test_label_seeded_harvest.py`.
    """
    ff = vf.falloff_features(field)
    fw, ws = float(r["fw"]), float(r["window_scale"])
    return {
        "band_coverage": float(r["band_coverage"]),
        "band_coverage_q25": float(r["band_coverage_q25"]),
        "log1p_radial_range": math.log1p(max(0.0, float(r["radial_range"]))),
        "log1p_radial_rings": math.log1p(max(0.0, float(r["radial_rings"]))),
        "interior_fraction": float(r["interior_fraction"]),
        "log10_size_rel": math.log10(ws / fw) if (ws > 0 and fw > 0) else 0.0,
        "falloff_rate": ff["falloff_rate"],
        "falloff_half": ff["falloff_half"],
        "log10_fw": math.log10(fw) if fw > 0 else 0.0,
        "cap_headroom": (float(r["cap_headroom"]) if r.get("cap_headroom") is not None
                         else float("nan")),
        "clamped": 1.0 if r.get("clamped") else 0.0,
        "composite_v3": float(r["composite"]),
    }


def open_fields(run_dir: Path):
    """The run's field store, read-only.

    `RunFieldCache(mode="r")` and NOT `FieldCache`: the retrospective pair
    (`index.json` + `valid.npy`) only exists once `finalize()` has run, and a run that was
    killed rather than halted at a boundary never gets there. The append-only index is
    always present and is the authority on what the store actually holds — including the
    truncation rule that an index line whose field is not fully on disk is not a row.
    """
    return vfc.RunFieldCache(Path(run_dir) / "view_fields", mode="r")


def build_sorted_queue(run_dir: Path) -> tuple[list[dict], dict]:
    """Every kept candidate, scored by `view_fit_v1.1` and ranked. Highest logit first.

    A candidate whose cached field is missing is DROPPED and counted, not imputed: two of
    the model's twelve features are derived from that array, and a row scored with them
    imputed would sit in the same ranking as rows that were actually measured.

    `SORTED` IS IN THE NAME because a loader that reorders and does not say so is a misuse
    waiting at every call site — see `build_q4_harvest_batches.build_sorted_queue` for the
    join this naming rule was earned by. The applied order rides in `rep["order"]`.
    """
    rows = [r for r in lsh.load_candidates(run_dir) if r.get("kept")]
    cache = open_fields(run_dir)
    model = vf.load_model_v11()
    out, no_field = [], 0
    for r in rows:
        field = cache.get(r["candidate_key"])
        if field is None:
            no_field += 1
            continue
        feats = _features_for(r, field)
        r = dict(r, feats=feats, fit_score=round(model.score(feats), 6),
                 fit_p_notbad=round(model.p_notbad(feats), 6),
                 screen_frame=SCREEN_FRAME)
        out.append(r)
    # Ties break on the candidate key, so the order is a pure function of the population and
    # a re-run of `queue` reproduces the same queue byte for byte.
    out.sort(key=lambda r: (-r["fit_score"], r["candidate_key"]))
    for i, r in enumerate(out, 1):
        r["queue_rank"] = i
    rep = dict(kept=len(rows), scored=len(out), dropped_no_field=no_field,
               order="fit_score desc, then candidate_key — NOT the harvest's append order",
               model=vf.MODEL_ID_V11)
    return out, rep


def stage_queue(args) -> int:
    q, rep = build_sorted_queue(args.run_dir)
    p = queue_path(args.run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in q:
            f.write(json.dumps({k: v for k, v in r.items() if k != "feats"},
                               default=str) + "\n")
    print(f"queue: {rep['scored']} scored of {rep['kept']} kept "
          f"({rep['dropped_no_field']} dropped: no cached field) under {rep['model']}")
    if q:
        fs = np.array([r["fit_score"] for r in q])
        cv = np.array([float(r["composite"]) for r in q])
        from scipy.stats import spearmanr
        print(f"  fit_score range [{fs.min():.2f}, {fs.max():.2f}]; "
              f"Spearman(fit, composite_v3) = {spearmanr(fs, cv).statistic:+.3f}")
        print("  cells (method x degree): " + json.dumps(dict(sorted(Counter(
            f'{r["method"]}|d{r["degree"]}' for r in q).items()))))
    print(f"  -> {p}")
    return 0


# =========================================================================== #
# draw
# =========================================================================== #
def draw_ranked_stratified(queue: list[dict], n: int) -> list[dict]:
    """`n` rows, round-robin to +/-1 over (method x degree), BEST-FIRST inside each cell.

    Floor-then-remainder over the cells, the same shape as `view_screen_sheets.stratify` and
    the supply crawl's draw — every non-empty cell gets one before any cell gets two — but
    the within-cell order is the fitted rank rather than a seeded shuffle, because this draw
    is taking the top of a queue rather than spanning a distribution.

    Returns `(rows_in_round_robin_order, per_cell_picks)`. The caller splits into two chunks
    using the PER-CELL lists, never by striding the flat one — see `draw_all`.
    """
    cells: dict = defaultdict(list)
    for r in queue:
        cells[(r["method"], int(r["degree"]))].append(r)
    for c in cells.values():
        c.sort(key=lambda r: r["queue_rank"])
    keys = sorted(cells, key=lambda k: (str(k[0]), k[1]))
    # Fewest taken so far wins; the larger cell breaks the tie, so a round-robin that cannot
    # fill every cell spends the remainder where there is supply. `apportion.deal_round_robin`
    # is the one copy of that rule.
    take = apportion.deal_round_robin({k: len(cells[k]) for k in keys}, n)
    per_cell = {k: cells[k][:take[k]] for k in keys}
    out, cursor = [], {k: 0 for k in keys}
    for _round in range(max(take.values(), default=0)):
        for k in keys:
            if cursor[k] < take[k]:
                out.append(cells[k][cursor[k]])
                cursor[k] += 1
    return out, per_cell


def draw_all(queue: list[dict], *, n_chunk: int = N_CHUNK) -> tuple[dict, dict]:
    """One 2N round-robin draw, split into two chunks WITHIN each cell.

    THE SPLIT IS PER CELL, NOT A STRIDE OVER THE FLAT DRAW, and the difference is not
    cosmetic. The flat round-robin cycles the cells in a fixed order, so taking every other
    element of it takes every other CELL whenever the number of cells is even: with the
    four (method x degree) cells this harvest actually produces, a stride split gave chunk A
    all of cells 1 and 3 and chunk B all of cells 2 and 4 — each chunk internally "balanced"
    at 145/145 and each missing half the population. Splitting inside a cell instead
    alternates its ranked members, so both chunks hold every cell and neither is
    systematically the better half. Caught by the median-rank assertion in
    `test_the_two_chunks_are_disjoint_and_both_balanced`, which is why that test also now
    asserts every cell is PRESENT in both chunks — a count over the rows a chunk has cannot
    see a cell it has none of.
    """
    _flat, per_cell = draw_ranked_stratified(queue, 2 * n_chunk)
    a, b = [], []
    for j, k in enumerate(sorted(per_cell)):
        picks = per_cell[k]
        # THE STARTING PARITY ALTERNATES BY CELL, and it fixes two things at once. A cell
        # with an ODD number of picks gives one chunk the extra row AND the better-ranked
        # member of every pair; taking parity from the cell index instead spreads both. With
        # eight cells of fifteen, a fixed parity shipped 64/56 and handed A the better row in
        # all eight cells; alternating ships 60/60.
        first, second = (picks[0::2], picks[1::2]) if j % 2 == 0 else \
                        (picks[1::2], picks[0::2])
        a.extend(first)
        b.extend(second)
    for bid, rows in ((CHUNK_A, a), (CHUNK_B, b)):
        for r in rows:
            r["batch_id"] = bid
    keys = [r["candidate_key"] for r in a + b]
    rep = dict(requested=2 * n_chunk, drawn=len(a) + len(b),
               counts={CHUNK_A: len(a), CHUNK_B: len(b)},
               cells={f"{k[0]}|d{k[1]}": len(v) for k, v in sorted(per_cell.items())},
               overlap=len(keys) - len(set(keys)),
               short_by=max(0, 2 * n_chunk - len(keys)))
    return {CHUNK_A: a, CHUNK_B: b}, rep


# =========================================================================== #
# the corpus batch
# =========================================================================== #
def _render_block(r: dict, palette: str) -> dict:
    fw = cc.hp_str(r["fw"])
    render = cc.render_block(cx=str(r["cx"]), cy=str(r["cy"]), fw=fw,
                             maxiter=int(dcf._maxiter_for_fw(float(r["fw"]))),
                             palette=palette, composition=COMPOSITION, width=CROP_W,
                             height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                             interior_mode=INTERIOR_MODE)
    render["fractal_type"] = r.get("family") or "mandelbrot"
    render["c_re"] = None
    render["c_im"] = None
    return render


def write_batch(batch_id: str, rows: list[dict], *, sampling: dict) -> dict:
    names = BMB._palette_names()
    order = list(range(len(rows)))
    np.random.default_rng(PRESENTATION_SEED ^ BMB._stable_seed(batch_id)).shuffle(order)
    for slot, oi in enumerate(order):
        h = BMB._stable_seed(rows[oi]["candidate_key"])
        rows[oi]["image_id"] = f"ls{slot:04d}_{h:08x}"
    rows.sort(key=lambda r: r["image_id"])

    full, blind = [], []
    for r in rows:
        pal = names[BMB._stable_seed(r["image_id"]) % len(names)]
        render = _render_block(r, pal)
        f = r.get("feats") or {}
        prov = cc.provenance_block(
            GEN_VERSION, batch_id, family=r.get("family"),
            selection_role="label_seeded_v2",
            stratum=f'{r["method"]}|d{r["degree"]}',
            method=r.get("method"), op=r.get("op"), k=r.get("k"),
            degree=r.get("degree"), period=r.get("period"),
            seed_id=r.get("seed_id"), seed_batch_id=r.get("seed_batch_id"),
            seed_image_id=r.get("seed_image_id"), seed_score=r.get("seed_score"),
            fit_model=vf.MODEL_ID_V11, fit_score=r.get("fit_score"),
            queue_rank=r.get("queue_rank"),
            composite=r.get("composite"), vetoed=r.get("vetoed"),
            size_factor=r.get("size_factor"),
            band_coverage=r.get("band_coverage"),
            band_coverage_q25=r.get("band_coverage_q25"),
            radial_range=r.get("radial_range"), radial_rings=r.get("radial_rings"),
            interior_fraction=r.get("interior_fraction"),
            falloff_rate=f.get("falloff_rate"), falloff_half=f.get("falloff_half"),
            log10_size_rel=f.get("log10_size_rel"),
            cap_headroom=r.get("cap_headroom"), clamped=r.get("clamped"),
            screen_frame=SCREEN_FRAME, screen_policy=r.get("screen_policy"),
            log10_abs_A=r.get("log10_abs_A"), window_scale=r.get("window_scale"),
            f64_margin_deploy_decades=r.get("f64_margin_deploy_decades"),
            parent_atom_id=r.get("parent_atom_id"),
            scale_ratio_decades=r.get("scale_ratio_decades"),
            atom_key=r.get("atom_key"), candidate_key=r.get("candidate_key"),
            descend_mode=f'label_seeded_{r.get("method")}_k{r.get("k")}')
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
        purpose=("q3/q4 candidate supply seeded from the corpus's own class-3/4 locations "
                 "(the source race's two top-rated sheets: nuclei at judged-good views, "
                 "plus neighbourhood expansion around them), framed at k in {none,8,16}, "
                 "screened at the view frame, interior>0.30 discarded at sourcing, and "
                 "ordered by the fitted view_fit_v1.1 score. TRAIN-side biased twice over: "
                 "the seeds are conditioned on past verdicts and the queue is ordered by a "
                 "model of the label. No rate measured here is a base rate."),
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


def stage_draw(args) -> int:
    q = _read_jsonl(queue_path(args.run_dir))
    # `feats` is stripped on the way to disk (it is derivable), so re-derive it here for the
    # three columns the provenance block records. One numpy pass over the cached fields.
    cache = open_fields(args.run_dir) if args.run_dir else None
    if cache is not None:
        for r in q:
            fld = cache.get(r["candidate_key"])
            if fld is not None:
                r["feats"] = _features_for(r, fld)
    chunks, rep = draw_all(q, n_chunk=args.n_chunk)
    if rep["overlap"]:
        raise SystemExit(f"draws overlap by {rep['overlap']} rows — a location may appear "
                         f"in only one batch (build_manifest.load_post_freeze asserts it).")
    if rep["short_by"]:
        print(f"  !! supply short by {rep['short_by']} rows: the queue holds {len(q)} and "
              f"the round-robin could not fill {rep['requested']}. Sized down, and the "
              f"shortfall is NAMED here rather than left to be read off a count.")
    common = dict(
        population=("every candidate of the label-seeded harvest that the view screen "
                    "reached and the interior>0.30 pre-filter kept"),
        queue_n=len(q), model=vf.MODEL_ID_V11,
        model_record="data/atlas/view_fit_v1_1.json",
        selection=("top of the fit-ordered queue, round-robin to +/-1 over "
                   "(method x degree), best-first within a cell"),
        chunk_split=("one 2N round-robin draw split by stride, so both chunks are "
                     "balanced on the same cells"),
        g_used_for_selection=True,
        composite_v3_recorded_not_used_for_order=True)
    out = [write_batch(bid, chunks[bid], sampling=dict(**common, chunk=bid))
           for bid in BATCHES]
    for o in out:
        print(f"  {o['batch_id']:34s} n={o['n']:4d}  assign_split={o['assign_split']}")
    for bid in BATCHES:
        print(f"    {bid[-6:]:6s} cells " + json.dumps(dict(sorted(Counter(
            f'{r["method"][:4]}|d{r["degree"]}' for r in chunks[bid]).items()))))
        rk = [r["queue_rank"] for r in chunks[bid]]
        print(f"           queue ranks {min(rk)}..{max(rk)} (median {int(np.median(rk))})")
    # The feature table: the join between a label that does not exist yet and what the
    # candidate carried when it was drawn, keyed on the image_id the labels come back under.
    fp = paths.durable(FEATURES_REL, mkparents=True)
    with open(fp, "w", encoding="utf-8") as f:
        for bid in BATCHES:
            for r in chunks[bid]:
                f.write(json.dumps(dict(
                    {k: v for k, v in r.items() if k != "feats"},
                    **(r.get("feats") or {})), default=str) + "\n")
    print(f"  features -> {fp}")
    return 0


# =========================================================================== #
# render
# =========================================================================== #
# `DEFAULT_ENGINE_THREADS` is 7 and is the number for ONE engine process; four workers of
# seven is 28 threads on a 12-core box, which is oversubscription, not throughput. Sized for
# the actual N as `corpus_common` says to: 4 x 3 = 12.
RENDER_THREADS = 3
RENDER_ORDER = (CHUNK_A, CHUNK_B)


def _render_to(render: dict, out: Path, source, timeout: float) -> None:
    """One crop; a HALF-WRITTEN one is deleted rather than left behind.

    A `timeout` kills the engine mid-write and the truncated JPG it leaves is the one
    failure mode this pipeline cannot see: `needs()` checks existence, so the row reads as
    rendered forever and the batch is quietly one bad crop short.
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
    canon = crops / f"{iid}.jpg"
    if not canon.exists():
        _render_to(render, canon, PALETTE_SOURCE, timeout)
    vp = vivid / f"{iid}.jpg"
    if not vp.exists():
        _render_to(dict(render, palette=VIVID_PALETTE), vp, VIVID_SOURCE, timeout)
    return iid


def stage_render(args) -> int:
    deadline = (time.time() + args.max_minutes * 60.0) if args.max_minutes else None
    total, stopped = 0, None
    for batch_id in RENDER_ORDER:
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
                    # pre-run estimate (`CLAUDE.md`).
                    print(f"  [{done}/{len(todo)}] {el:.0f}s  {done/max(el,1e-9)*60:.1f} "
                          f"row/min  ETA {(len(todo)-done)*el/max(done,1)/60:.1f} min "
                          f"({len(fails)} failed)", flush=True)
                if deadline and time.time() > deadline:
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
        print(f"  {b:34s} {v['rows'] - v['missing']:4d}/{v['rows']:4d} crops"
              + ("  COMPLETE" if v["complete"] else "  INCOMPLETE — not labelable"))
    rs = paths.scratch("label_seeded_harvest", "render_state.json")
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(json.dumps(dict(stopped_during=stopped, per_batch=left, NOTE=(
        "resumable: re-run `render`, it skips crops that exist. A batch marked INCOMPLETE "
        "must not be queued for labeling.")), indent=2) + "\n", encoding="utf-8")
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

    emit(f"=== label-seeded harvest batches — acceptance ({STAMP}) ===")
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
        check("classified TRAIN-side and BIASED", split == "train" and biased is True)
        check("batch.json records the same classification",
              bj["registration"]["assign_split"] == [split, biased, source])
        check("every label is null (nothing is labeled)",
              all(r["label"]["score"] is None for r in rows))
        check("image_id is opaque `ls<slot>_<hash>`",
              all(len(r["image_id"]) == 15 and r["image_id"].startswith("ls")
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
        # Every row carries the ordering it was selected by AND the incumbent beside it —
        # the whole point of recording `composite` on a queue ordered by something else.
        check("every row carries fit_score, queue_rank and composite_v3",
              all(r["provenance"].get("fit_score") is not None
                  and r["provenance"].get("queue_rank") is not None
                  and r["provenance"].get("composite") is not None for r in rows))
        check("every row is past the interior pre-filter",
              all(float(r["provenance"]["interior_fraction"]) <= lsh.INTERIOR_DISCARD
                  for r in rows),
              f"max {max(float(r['provenance']['interior_fraction']) for r in rows):.4f} "
              f"vs {lsh.INTERIOR_DISCARD}")
        check("k is on the pushed ladder",
              {r["provenance"]["k"] for r in rows} <= set(
                  mnv_k for mnv_k in lsh.mnv.parse_k_spec(lsh.K_LADDER_SPEC)),
              str(sorted({str(r["provenance"]["k"]) for r in rows})))
        cells = Counter(f'{r["provenance"]["method"]}|d{r["provenance"]["degree"]}'
                        for r in rows)
        spread = max(cells.values()) - min(cells.values())
        check("(method x degree) cells balanced to +/-1", spread <= 1,
              f"spread {spread}: {dict(sorted(cells.items()))}")
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
    else:
        check("feature table exists", False, str(fp))
    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    rep = paths.scratch("label_seeded_harvest", "verify.txt")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
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
    r.add_argument("--max-minutes", type=float, default=0.0,
                   help="stop starting new rows after this many minutes (0 = no bound)")
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
