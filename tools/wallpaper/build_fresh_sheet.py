r"""build_fresh_sheet.py — the fresh-era wallpaper CORRECTION SHEET, pre-labeled by v3.

One new `data/wallpaper_corpus/batches/` batch drawn from the CURRENT stage-2 admitted
intake, every row carrying the deployed v3 head's own verdict as a SUGGESTION for Matt to
correct. This is the fresh-era training/eval material for the wallpaper retrain
(prompts/wallpaper_fresh_sheet_prompt.md). No head, gate or floor moves here — v3 is read,
never written, and `wallpaper_pins.GATE_THRESHOLD` is not consulted at all (the sheet must
span the range, so nothing is gated out).

POPULATION — two intake sources, both floor-admit-aware.
  * The seven-ledger admitted UNION (`descriptor.load_union_admitted` over
    `ledger_rescore.LEDGERS`, 751 locations at the 2026-08-04 census). The floor-admitted
    source inside it is `q4_harvest` (108).
  * The `human_q3plus` library seed (`data/emission/library_seed_v2/intake.json`, 168 looks,
    a HUMAN 3-or-4 with no decode consulted). The prompt names BOTH floor-admit sources
    (`descriptor.FLOOR_ADMIT_SOURCES`) as the vein to oversample, and this one is not in the
    seven ledgers — it is a durable intake snapshot beside them, so it is unioned in and
    tagged (`provenance.intake_source`) rather than quietly dropped. Every row records which
    of the two it came from; nothing has to be inferred later.

THE SCREEN IS NOT THE STAMP. Stratifying a draw across v3 score bins needs a v3 score BEFORE
the sheet exists, and rendering everything to find out would render 4x what is kept. So each
location is screened on the SCORING-ONLY coarse path (`colormap.coarse_field` +
`render_candidates_coarse`, 512x288 — the same fence the beam's pref scoring runs behind:
never a stored crop), and the score STAMPED IN-ROW is re-derived on the real label crop.
Both are kept per row (`head_v3.p_ge3` vs `provenance.screen_p_ge3`) so the proxy can be
audited against the thing it was a proxy for; the run prints their agreement.

  screen   751+168 locations x N_SCREEN seeded palettes -> coarse v3 p_ge3   (~1.5 s/loc)
  select   locations binned by max screen p_ge3; equal-ish quota per bin, floor-admit
           oversampled inside each bin; PICKS_PER_LOC picks spread across each location's
           own screen range (so a location contributes its worst render, not only its best)
  render   the label-crop pins (label_crop.py: 1280x720 ss2 lanczos3 q90) -> score with v3
           -> stamp head_v3 + suggested_tier (suggest_tier.py) + split_side
  write    images.jsonl (seeded-shuffled presentation order, `sheet_order` stamped) +
           batch.json

Resumable at every stage: the screen appends one durable row per location, the render
appends one durable ledger record per location, both keyed so a kill loses at most the
in-flight unit.

    uv run python -u tools/wallpaper/build_fresh_sheet.py estimate
    uv run python -u tools/wallpaper/build_fresh_sheet.py screen --limit 8      # smoke
    uv run python -u tools/wallpaper/build_fresh_sheet.py screen  > scratch/wallpaper_fresh_sheet/screen.log 2>&1
    uv run python -u tools/wallpaper/build_fresh_sheet.py select               # dry composition
    uv run python -u tools/wallpaper/build_fresh_sheet.py render > scratch/wallpaper_fresh_sheet/render.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "queries", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import colormap as cm                    # noqa: E402  stretch/coarse/render_candidate(s)
import corpus_common as cc               # noqa: E402  engine launch defaults + priority
import location as loc_mod               # noqa: E402  canonical Location
import query_sampler as qs               # noqa: E402  palette pool + candidate draw
import sample_location as SL             # noqa: E402  gen0_palettes (the deployed palette draw)
from active_ckpt import auto_maxiter     # noqa: E402  native fw-dependent maxiter policy
from label_crop import (                 # noqa: E402  THE label-crop pins (Recipe-2 tail)
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q,
    ensure_label_field, render_label_crop,
)
from tools.emission import descriptor as D        # noqa: E402
from tools.emission import ledger_rescore as LR   # noqa: E402
from tools.wallpaper import wallpaper_pins as WP  # noqa: E402  the head pin (torch-free)
from tools.wallpaper.suggest_tier import (        # noqa: E402
    CUTS as SUGGEST_CUTS, DERIVATION as SUGGEST_DERIVATION, expected_tier, tier_from_pred)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

BATCH_ID = "2026-08-05_wallpaper_fresh_sheet_v1"
GENERATOR_VERSION = "wallpaper_fresh_sheet_v1"
IMG_PREFIX = "wfs"
# This batch's coloring REGIME (see provenance_block). Its sibling stamps "colorize_path".
COLORING_SOURCE = "pool_draw"
LABELS_EXPORT = ROOT / "labels" / "wallpaper_fresh_sheet_v1.json"   # tracked; written by the merge

WALLPAPER_CORPUS = ROOT / "data" / "wallpaper_corpus"
WORK = ROOT / "scratch" / "wallpaper_fresh_sheet"        # regenerable: fields, screen log
SCREEN_LOG = WORK / "screen.jsonl"
SCREEN_FIELDS = WORK / "screen_fields"                   # transient; each is deleted after use
HUMANQ3PLUS_INTAKE = ROOT / "data" / "emission" / "library_seed_v2" / "intake.json"

# --- composition knobs ------------------------------------------------------
N_SCREEN = 8            # seeded palette candidates screened per location
PICKS_PER_LOC = 4       # "a few renders per location" — spread across the location's range
TARGET_LOCS = 240       # 240 x 4 = 960 renders, inside the prompt's 800-1000 band
PALETTE_POOL = 120      # palettes the per-location draw samples from (see palette_pool())
SEED = 7

# v3 score bins on a location's MAX screen p_ge3. Five bins straddling the deployed emission
# gate (0.90) so the sheet spans "the head is certain this is junk" through "the head would
# ship this today". Right edge open above 1.0 so p_ge3 == 1.0 lands in the top bin.
SCORE_BINS = (0.0, 0.01, 0.10, 0.50, 0.90, 1.01)
BIN_LABELS = ("b0_lt.01", "b1_.01-.10", "b2_.10-.50", "b3_.50-.90", "b4_ge.90")

# Floor-admitted share of the DRAWN locations, applied inside every score bin (so the
# oversample cannot smuggle in a score shift). Population share is 276/919 = 30%.
FLOOR_SOURCE_DRAW_FRAC = 0.40

# Location-grouped seeded split, stamped in-row (the dramatic batch's contract).
EVAL_FRAC = 0.30
SPLIT_SEED = 0

LABEL_CROP_WORKERS = 4  # project-wide max-workers cap — DO NOT raise
SHUFFLE_SEED = 20260805  # presentation order (the UI honors file order in correction mode)


def log(msg: str):
    print(msg, flush=True)


# =========================================================================== #
# 1. Population.
# =========================================================================== #

def _union_sources():
    """The seven-ledger admitted union -> source dicts. `key` is the canonical location key."""
    ledgers = [LR.ledger_path(rel) for _tag, rel in LR.LEDGERS]
    missing = [str(p) for p in ledgers if not p.exists()]
    if missing:
        raise SystemExit(f"[fresh-sheet] intake ledger absent: {missing} — the population "
                         "cannot be derived; nothing is guessed here.")
    rows, diag = D.load_union_admitted(ledgers)
    out = []
    for r in rows:
        loc = D.location_of(r)
        tag = D.source_tag_of(r)
        out.append({
            "unit_key": f"union:{r['id']}", "loc": loc, "key": loc.key(),
            "intake_source": "union_ledger", "source_tag": tag,
            "floor_admit": tag in D.FLOOR_ADMIT_SOURCES,
            "partition": D.cell_partition(r), "ledger_family": r.get("family"),
            "source_ledger": r.get("_source_ledger"), "source_oid": r.get("_ledger_row_id"),
            "source_p_good": r.get("p_good"), "source_decoded_class": r.get("decoded_class"),
            "human_label": None,
        })
    return out, diag


def _humanq3plus_sources():
    """The `human_q3plus` library-seed looks -> source dicts (one per distinct look).

    Locations are rebuilt from the snapshot's own render block at the LIVE maxiter policy —
    the snapshot froze `maxiter` under an older cap, and the label field must be dumped under
    the policy every other wallpaper batch uses (`auto_maxiter`, bootstrap parity)."""
    import dataclasses
    if not HUMANQ3PLUS_INTAKE.exists():
        raise SystemExit(f"[fresh-sheet] {HUMANQ3PLUS_INTAKE} absent — the prompt names "
                         "human_q3plus as a population; a missing snapshot is a hard stop, "
                         "not a silently smaller sheet.")
    snap = json.loads(HUMANQ3PLUS_INTAKE.read_text(encoding="utf-8"))
    out = []
    for iid, e in sorted(snap["entries"].items()):
        loc = loc_mod.from_render_block(e["render"])
        loc = dataclasses.replace(loc, maxiter=auto_maxiter(float(loc.fw)))
        out.append({
            "unit_key": f"hq3p:{iid}", "loc": loc, "key": loc.key(),
            "intake_source": "human_q3plus_seed", "source_tag": snap["mix_source"],
            "floor_admit": snap["mix_source"] in D.FLOOR_ADMIT_SOURCES,
            "partition": e["partition"], "ledger_family": e["partition"],
            "source_ledger": snap["source"], "source_oid": iid,
            "source_p_good": None, "source_decoded_class": None,
            "human_label": int(e["human"]),
        })
    return out, {"n_looks": snap["n_looks"], "source": snap["source"]}


def population():
    """The full screened population: union ∪ human_q3plus, deduped by canonical location key
    (union wins a tie so a location keeps its ledger provenance). Returns (sources, report)."""
    u, udiag = _union_sources()
    h, hdiag = _humanq3plus_sources()
    seen = {s["key"] for s in u}
    h_kept = [s for s in h if s["key"] not in seen]
    srcs = u + h_kept
    report = {
        "union": {"n": len(u), "diag": {k: v for k, v in udiag.items()
                                        if k in ("n_union", "per_ledger", "n_id_collisions",
                                                 "n_location_overlaps")}},
        "human_q3plus": {"n_looks": hdiag["n_looks"], "n_kept": len(h_kept),
                         "n_dropped_dup_of_union": len(h) - len(h_kept),
                         "source": hdiag["source"]},
        "n_population": len(srcs),
        "n_floor_admit": sum(1 for s in srcs if s["floor_admit"]),
        "by_source_tag": dict(Counter(s["source_tag"] for s in srcs)),
        "by_partition": dict(Counter(s["partition"] for s in srcs)),
    }
    return srcs, report


# =========================================================================== #
# 2. Palette draw — the existing builders' pattern, without the beam.
# =========================================================================== #

def palette_pool(sampler, k: int = PALETTE_POOL):
    """`k` palette names via `sample_location.gen0_palettes` — the DEPLOYED gen-0 draw
    (`GEN0_SOURCE_WEIGHTS`, 75% dramatic, feather-point-diverse WITHIN each source bucket),
    imported rather than restated. Location-independent by construction, so per-location
    variety comes from the seeded subsample in `draw_candidates`.

    k is PALETTE_POOL (120), twice the deployed beam's N_GEN0: the beam narrows 60 palettes
    to a per-location best, a labeling sheet wants palette COVERAGE across ~240 locations.
    The composition is unchanged — a wider FP draw within the same buckets."""
    names, _all, _dmat = SL.gen0_palettes(sampler, k)
    return names


def _stable_seed(key: str) -> int:
    """A per-location seed that survives a restart. `hash()` on a str is salted per process
    (PYTHONHASHSEED), so a screen resumed after a kill would draw DIFFERENT palettes for the
    same location and the batch would stop being reproducible from its seed."""
    return zlib.crc32(key.encode("utf-8"))


def draw_candidates(loc, pool_names, sampler, seed_offset: int, n=N_SCREEN):
    """`n` distinct-palette `CandidateConfig`s for one location, seeded on the location.

    Palettes: a seeded choice-without-replacement over `pool_names` (itself 75% dramatic by
    construction, so uniform selection preserves the deployed source composition in
    expectation). Params: `query_sampler.sample_candidate` with the palette fixed — the same
    per-type param law (`gamma`/`log_premap`/source-aware `phase`/`n_cycles`/`reverse`) every
    existing wallpaper batch's candidates were drawn under."""
    rng = np.random.default_rng([SEED, seed_offset])
    idx = rng.choice(len(pool_names), size=min(n, len(pool_names)), replace=False)
    return [qs.sample_candidate(loc, rng, sampler, palette=pool_names[int(i)]) for i in idx]


# =========================================================================== #
# 3. Screen — coarse, SCORING-ONLY (see the module docstring's fence note).
# =========================================================================== #

def _marginals(scorer, images):
    """(N,h,w,3) uint8 -> (N,K-1) CORN MARGINALS (cumprod of the conditional sigmoids)."""
    import torch
    from PIL import Image
    xs = torch.stack([scorer.transform(Image.fromarray(im)) for im in images])
    with torch.no_grad():
        dev = scorer.device
        with torch.autocast(device_type=dev.split(":")[0], enabled=(dev != "cpu")):
            logits = scorer.model(xs.to(dev))
        cond = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
    return np.cumprod(cond, axis=1)


def _marginals_from_paths(scorer, paths, batch_size: int = 32):
    """Same, off JPGs on disk — the KEEPER path's score (what gets stamped in-row)."""
    import torch
    from PIL import Image
    out, buf = [], []

    def flush():
        if not buf:
            return
        dev = scorer.device
        with torch.no_grad():
            with torch.autocast(device_type=dev.split(":")[0], enabled=(dev != "cpu")):
                logits = scorer.model(torch.stack(buf).to(dev))
        out.append(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64))
        buf.clear()

    for p in paths:
        with Image.open(p) as im:
            im.load()
            buf.append(scorer.transform(im.convert("RGB")))
        if len(buf) == batch_size:
            flush()
    flush()
    return np.cumprod(np.concatenate(out, axis=0), axis=1)


def load_screen() -> dict:
    """{unit_key: screen record} already on disk."""
    done = {}
    if SCREEN_LOG.exists():
        for line in SCREEN_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit_key"]] = rec
    return done


def run_screen(args):
    from classifier.inference import load_scorer

    WORK.mkdir(parents=True, exist_ok=True)
    SCREEN_FIELDS.mkdir(parents=True, exist_ok=True)
    srcs, prep_report = population()
    log(f"[screen] population {prep_report['n_population']} locations "
        f"({prep_report['union']['n']} union + {prep_report['human_q3plus']['n_kept']} "
        f"human_q3plus; {prep_report['n_floor_admit']} floor-admitted)")

    done = load_screen()
    todo = [s for s in srcs if s["unit_key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"[screen] {len(done)} already screened, {len(todo)} to run")
    if not todo:
        return

    sampler = qs.PaletteSampler(qs.load_pool_library())
    lib = sampler.library
    pool_names = palette_pool(sampler)
    scorer = load_scorer(str(WP.HEAD_CKPT))
    log(f"[screen] head {WP.HEAD_VERSION} on {scorer.device} (K={scorer.config.get('num_classes')}) "
        f"· palette pool {len(pool_names)} · {N_SCREEN} candidates/location")

    t_wall = time.time()
    times = []
    for i, s in enumerate(todo):
        t0 = time.time()
        loc = s["loc"]
        try:
            field = ensure_label_field(loc, fields_dir=SCREEN_FIELDS)
            prep = cm.stretch_field(field)
            coarse = cm.coarse_field(prep)
            cfgs = draw_candidates(loc, pool_names, sampler, seed_offset=_stable_seed(s["key"]))
            imgs = cm.render_candidates_coarse(coarse, cfgs, lib)
            marg = _marginals(scorer, imgs)
            err = None
        except Exception as e:                                   # noqa: BLE001
            # A location that cannot render is a real population fact, recorded with its
            # reason — not dropped, and not allowed to abort 900 other locations.
            cfgs, marg, err = [], np.zeros((0, 3)), f"{type(e).__name__}: {e}"
        finally:
            _wipe_field(loc)

        rec = {
            "unit_key": s["unit_key"], "key": s["key"], "family": loc.family,
            "intake_source": s["intake_source"], "source_tag": s["source_tag"],
            "floor_admit": s["floor_admit"], "partition": s["partition"],
            "fw": loc.fw, "maxiter": loc.maxiter, "error": err,
            "candidates": [
                {"config": json.loads(c.to_json()),
                 "palette": c.palette,
                 "palette_type": lib.palette_type(c.palette),
                 "palette_source": sampler.source_of(c.palette),
                 "p_ge2": float(marg[j, 0]), "p_ge3": float(marg[j, 1]),
                 "p_ge4": float(marg[j, 2]) if marg.shape[1] > 2 else None,
                 "pred": expected_tier(marg[j])}
                for j, c in enumerate(cfgs)],
        }
        with SCREEN_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        dt = time.time() - t0
        times.append(dt)
        if err:
            log(f"[screen] {i+1}/{len(todo)} {loc.family:18} FAILED  {err[:90]}")
        elif (i + 1) % 20 == 0 or i < 3:
            recent = float(np.mean(times[-30:]))
            log(f"[screen] {i+1}/{len(todo)} {loc.family:18} fw={loc.fw[:10]} "
                f"max p_ge3={marg[:, 1].max():.3f}  [{dt:.1f}s]  "
                f"recent {recent:.1f}s/loc -> eta {recent*(len(todo)-i-1)/60:.0f} min "
                f"(elapsed {(time.time()-t_wall)/60:.0f} min)")
    log(f"[screen] done: {len(todo)} locations in {(time.time()-t_wall)/60:.1f} min")


def _wipe_field(loc):
    """Drop the location's screening field. 15 MB each x ~900 locations is 13 GB of scratch
    for a pass that reads each field exactly once; the ~250 kept locations re-dump in the
    render stage for ~1 s apiece. Failure to delete is not worth failing the unit over."""
    import hashlib
    ptok = loc_mod.maxiter_policy_token()
    suffix = f"|{ptok}" if ptok else ""
    h = hashlib.sha1(
        f"{loc.key()}|{LABEL_W}x{LABEL_H}ss{LABEL_SS}|{loc.maxiter}{suffix}".encode()
    ).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{LABEL_W}x{LABEL_H}ss{LABEL_SS}"
    for ext in (".bin", ".json"):
        try:
            (SCREEN_FIELDS / f"{stem}{ext}").unlink(missing_ok=True)
        except OSError:
            pass


# =========================================================================== #
# 4. Select — bin by v3, quota per bin, floor-admit oversampled inside each bin.
# =========================================================================== #

def bin_of(p: float) -> int:
    """Index into BIN_LABELS for a location's max screen p_ge3."""
    for i in range(len(BIN_LABELS)):
        if SCORE_BINS[i] <= p < SCORE_BINS[i + 1]:
            return i
    return len(BIN_LABELS) - 1


def _apportion(quota: int, avail: list[int]) -> list[int]:
    """Equal-ish integer quota over bins, capped at availability, deficit redistributed to
    the bins that still have room. Iterated, so an empty bin never silently shrinks the draw
    (it hands its slots to the bins that can fill them, and the report shows it happened)."""
    n = len(avail)
    take = [0] * n
    room = list(avail)
    remaining = min(quota, sum(avail))
    while remaining > 0:
        active = [i for i in range(n) if room[i] > 0]
        if not active:
            break
        share, extra = divmod(remaining, len(active))
        moved = 0
        for j, i in enumerate(active):
            want = share + (1 if j < extra else 0)
            got = min(want, room[i])
            take[i] += got
            room[i] -= got
            moved += got
        remaining -= moved
        if moved == 0:
            break
    return take


def select(screen_recs, target_locs=TARGET_LOCS, seed=SEED):
    """Stratified location draw. Returns (selected, report).

    Each surviving screen record gets `loc_p_ge3 = max_j p_ge3` and a bin. Bins receive an
    equal-ish quota; inside each bin FLOOR_SOURCE_DRAW_FRAC of the slots go to floor-admitted
    locations (`q4_harvest` / `human_q3plus`) before the rest are drawn from discovery, with
    either side's shortfall handed to the other so the bin still fills."""
    ok = [r for r in screen_recs if not r["error"] and r["candidates"]]
    for r in ok:
        r["loc_p_ge3"] = max(c["p_ge3"] for c in r["candidates"])
        r["loc_pred"] = max(c["pred"] for c in r["candidates"])
        r["bin"] = bin_of(r["loc_p_ge3"])

    by_bin = defaultdict(list)
    for r in ok:
        by_bin[r["bin"]].append(r)
    avail = [len(by_bin[i]) for i in range(len(BIN_LABELS))]
    quotas = _apportion(target_locs, avail)

    rng = np.random.default_rng([seed, 1])
    selected, per_bin = [], []
    for i, q in enumerate(quotas):
        members = sorted(by_bin[i], key=lambda r: r["unit_key"])
        floor = [r for r in members if r["floor_admit"]]
        disc = [r for r in members if not r["floor_admit"]]
        rng.shuffle(floor)
        rng.shuffle(disc)
        n_floor = min(len(floor), int(round(FLOOR_SOURCE_DRAW_FRAC * q)))
        n_disc = min(len(disc), q - n_floor)
        n_floor = min(len(floor), q - n_disc)          # hand back the discovery shortfall
        take = floor[:n_floor] + disc[:n_disc]
        selected.extend(take)
        per_bin.append({
            "bin": BIN_LABELS[i], "range": [SCORE_BINS[i], SCORE_BINS[i + 1]],
            "available": len(members), "available_floor_admit": len(floor),
            "quota": q, "drawn": len(take), "drawn_floor_admit": n_floor,
        })

    selected.sort(key=lambda r: r["unit_key"])
    report = {
        "target_locations": target_locs, "drawn_locations": len(selected),
        "screened": len(screen_recs), "screen_failures": len(screen_recs) - len(ok),
        "score_bins": list(SCORE_BINS), "bin_labels": list(BIN_LABELS),
        "floor_admit_frac_target": FLOOR_SOURCE_DRAW_FRAC,
        "floor_admit_frac_realized": (sum(1 for r in selected if r["floor_admit"])
                                      / max(len(selected), 1)),
        "per_bin": per_bin,
        "drawn_by_source_tag": dict(Counter(r["source_tag"] for r in selected)),
        "drawn_by_intake_source": dict(Counter(r["intake_source"] for r in selected)),
        "drawn_by_partition": dict(Counter(r["partition"] for r in selected)),
        "population_by_bin": {BIN_LABELS[i]: avail[i] for i in range(len(BIN_LABELS))},
        "population_floor_admit_by_bin": {
            BIN_LABELS[i]: sum(1 for r in by_bin[i] if r["floor_admit"])
            for i in range(len(BIN_LABELS))},
    }
    return selected, report


def pick_candidates(rec, n=PICKS_PER_LOC):
    """`n` of the location's screened candidates, spread evenly across ITS OWN screen-score
    order. The sheet needs negatives and mid-range inside a location too, not only across
    locations — a top-n pick would make "the head's best guess" the only thing labeled."""
    cands = sorted(rec["candidates"], key=lambda c: (c["p_ge3"], c["palette"]))
    if len(cands) <= n:
        return list(cands)
    idx = np.linspace(0, len(cands) - 1, n).round().astype(int)
    return [cands[int(i)] for i in sorted(set(idx.tolist()))]


def assign_split(selected, seed=SPLIT_SEED, eval_frac=EVAL_FRAC):
    """Location-grouped seeded eval assignment, STRATIFIED BY SCORE BIN so the eval side
    spans the same range the sheet does. One location -> one side (location-disjoint by
    construction, since a location is the unit)."""
    rng = np.random.RandomState(seed)
    sides = {}
    n_eval = 0
    for b in range(len(BIN_LABELS)):
        keys = sorted(r["unit_key"] for r in selected if r["bin"] == b)
        rng.shuffle(keys)
        k = int(round(eval_frac * len(keys)))
        for j, uk in enumerate(keys):
            sides[uk] = "eval" if j < k else "train"
        n_eval += k
    return sides, n_eval


# =========================================================================== #
# 5. Head-corpus overlap — reported and stamped, never used to exclude.
# =========================================================================== #

def head_corpus_index():
    """(keys, per-family coords) over EVERY existing wallpaper batch. The prompt pins the
    population, so an overlap is not excluded — it is stamped (`provenance.head_corpus_seen`)
    so a retrain can honor it instead of discovering it."""
    import build_fresh_discovery as BFD                        # noqa: PLC0415  (_spatially_in)
    keys, coords = set(), defaultdict(list)
    for images in sorted((WALLPAPER_CORPUS / "batches").glob("*/images.jsonl")):
        for line in images.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loc = loc_mod.from_render_block(json.loads(line)["render"])
            keys.add(loc.key())
            try:
                coords[loc.family].append((float(loc.cx), float(loc.cy), float(loc.fw),
                                           loc.c_re, loc.c_im))
            except (TypeError, ValueError):
                pass
    return keys, coords, BFD._spatially_in


# =========================================================================== #
# 6. Rows.
# =========================================================================== #

def batch_dir() -> Path:
    return WALLPAPER_CORPUS / "batches" / BATCH_ID


def render_block(loc, palette):
    blk = {
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
        "fractal_type": loc.family,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "palette": palette,
        "composition": "center",
        "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
        "filter": LABEL_FILTER, "interior_mode": "black",
    }
    for k, v in loc.params.items():
        blk[k] = v
    return blk


def provenance_block(src, rec, loc, cand, split_side, seen):
    cfg = cand["config"]
    return {
        "generator_version": GENERATOR_VERSION,
        "batch_id": BATCH_ID,
        "lineage": "fresh_sheet_intake_screen",
        "family": loc.family,
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
        "palette": cand["palette"],
        # the colormap recipe — the crop is a pure function of render + these params
        "params": {
            "palette": cand["palette"], "palette_type": cand["palette_type"],
            "palette_source": cand["palette_source"],
            "reverse": cfg["reverse"], "log_premap": cfg["log_premap"],
            "gamma": cfg["gamma"], "phase": cfg["phase"], "n_cycles": cfg["n_cycles"],
            "transfer": cfg.get("transfer"), "transfer_gamma": cfg.get("transfer_gamma"),
            "interior_color": list(cfg["interior_color"]),
            "eval_filter": cfg.get("filter"),
        },
        "render_mode": "smooth",
        # THE REGIME AXIS. `pool_draw` — palette + params drawn from the query-sampler pool,
        # the way every existing wallpaper batch was built. Its sibling batch
        # (build_colorize_sheet.py) stamps `colorize_path`: same intake, same label-crop pins,
        # but coloured the way a live emission run colours (deficit-assigned flavor,
        # pref-v3-gvo palette pick, canonical params). A retrain that unions the two must be
        # able to separate them, and nothing else in the row distinguishes the regimes.
        "coloring_source": COLORING_SOURCE,
        # intake provenance
        "intake_source": src["intake_source"],          # union_ledger | human_q3plus_seed
        "source_tag": src["source_tag"],                # dive/steered/q4_harvest/...
        "floor_admit": src["floor_admit"],
        "partition": src["partition"],
        "source_ledger": src["source_ledger"],
        "source_oid": src["source_oid"],
        "source_p_good": src["source_p_good"],
        "source_decoded_class": src["source_decoded_class"],
        "human_q3plus_label": src["human_label"],       # the HUMAN 3/4 on the seed look, if any
        # screen (SELECTION-ONLY — the stamped score is head_v3, off the real crop)
        "screen_p_ge3": cand["p_ge3"],
        "screen_pred": cand["pred"],
        "screen_bin": BIN_LABELS[rec["bin"]],
        "loc_screen_p_ge3": rec["loc_p_ge3"],
        "screen_path": "colormap.coarse_field + render_candidates_coarse "
                       f"({cm.SCORE_COARSE_W}x{cm.SCORE_COARSE_H}), SCORING-ONLY",
        "n_screened_candidates": len(rec["candidates"]),
        # split + corpus overlap
        "split_side": split_side,                        # eval | train (STAMPED)
        "split_origin": "fresh_sheet_binstratified",
        "head_corpus_seen": seen,                        # "no" | "key" | "proximity"
    }


# =========================================================================== #
# 7. Render driver.
# =========================================================================== #

def _ledger_path() -> Path:
    return batch_dir() / "_progress_ledger.jsonl"


def load_render_ledger() -> dict:
    done = {}
    p = _ledger_path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit_key"]] = rec
    return done


def run_render(args):
    from classifier.inference import load_scorer

    screen = load_screen()
    if not screen:
        raise SystemExit("[render] no screen records — run `screen` first")
    srcs = {s["unit_key"]: s for s in population()[0]}
    selected, sel_report = select(list(screen.values()), args.target_locs, args.seed)
    sides, n_eval = assign_split(selected)
    hc_keys, hc_coords, spatially_in = head_corpus_index()

    log(f"[render] {len(selected)} locations x <= {PICKS_PER_LOC} picks "
        f"= <= {len(selected)*PICKS_PER_LOC} renders   (eval-side locations {n_eval})")
    _print_composition(sel_report)
    if args.estimate:
        return

    sampler = qs.PaletteSampler(qs.load_pool_library())
    lib = sampler.library
    scorer = load_scorer(str(WP.HEAD_CKPT))
    log(f"[render] head {WP.HEAD_VERSION} on {scorer.device}")

    bd = batch_dir()
    crops = bd / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    fields_dir = WORK / "render_fields"

    done = load_render_ledger()
    todo = [r for r in selected if r["unit_key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"[render] {len(done)} locations already in the ledger, {len(todo)} to run")

    order = {r["unit_key"]: i for i, r in enumerate(selected)}
    times, failures = [], []
    t_wall = time.time()
    for i, rec in enumerate(todo):
        t0 = time.time()
        src = srcs[rec["unit_key"]]
        loc = src["loc"]
        ui = order[rec["unit_key"]]
        picks = pick_candidates(rec, args.picks)
        try:
            field = ensure_label_field(loc, fields_dir=fields_dir)
            prep = cm.stretch_field(field)
        except Exception as e:                                   # noqa: BLE001
            failures.append({"unit_key": rec["unit_key"], "stage": "field",
                             "error": f"{type(e).__name__}: {e}"})
            log(f"[render] {i+1}/{len(todo)} FIELD FAILED {rec['unit_key']}: {e}")
            continue

        def _one(job):
            pi, cand = job
            image_id = f"{IMG_PREFIX}_{ui:03d}_{pi:02d}"
            cfg = cm.CandidateConfig.from_json(json.dumps(cand["config"]))
            w, h = render_label_crop(field, cfg, lib, crops / f"{image_id}.jpg", prep=prep)
            return pi, image_id, w, h

        try:
            with ThreadPoolExecutor(max_workers=min(LABEL_CROP_WORKERS, len(picks))) as ex:
                rendered = list(ex.map(_one, list(enumerate(picks))))
        except Exception as e:                                   # noqa: BLE001
            failures.append({"unit_key": rec["unit_key"], "stage": "crop",
                             "error": f"{type(e).__name__}: {e}"})
            log(f"[render] {i+1}/{len(todo)} CROP FAILED {rec['unit_key']}: {e}")
            continue

        paths = [crops / f"{iid}.jpg" for _pi, iid, _w, _h in rendered]
        marg = _marginals_from_paths(scorer, paths)
        seen = ("key" if loc.key() in hc_keys
                else "proximity" if spatially_in(loc, hc_coords) else "no")

        unit_rows = []
        for j, (pi, image_id, w, h) in enumerate(rendered):
            assert (w, h) == (LABEL_W, LABEL_H), (image_id, w, h)
            cand = picks[pi]
            pred = expected_tier(marg[j])
            unit_rows.append({
                "image_id": image_id,
                "render": render_block(loc, cand["palette"]),
                "provenance": provenance_block(src, rec, loc, cand,
                                               sides[rec["unit_key"]], seen),
                "label": {"score": None, "labeler": None, "labeled_at": None},
                # THE PRE-LABEL. Scored on the stored crop through the deploy transform.
                "head_v3": {
                    "pred": pred,
                    "p_ge2": float(marg[j, 0]), "p_ge3": float(marg[j, 1]),
                    "p_ge4": float(marg[j, 2]) if marg.shape[1] > 2 else None,
                    "ckpt": WP.HEAD_CKPT_REL, "head_version": WP.HEAD_VERSION,
                },
                "p_ge3": float(marg[j, 1]),          # flat, the prompt's stamp
                "suggested_tier": tier_from_pred(pred),
            })

        with _ledger_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"unit_key": rec["unit_key"], "unit_index": ui,
                                 "rows": unit_rows}) + "\n")
        times.append(time.time() - t0)
        if (i + 1) % 10 == 0 or i < 3:
            recent = float(np.mean(times[-20:]))
            log(f"[render] {i+1}/{len(todo)} {loc.family:18} {len(unit_rows)} crops  "
                f"p_ge3[{marg[:,1].min():.2f},{marg[:,1].max():.2f}]  "
                f"[{times[-1]:.0f}s]  recent {recent:.0f}s/loc -> eta "
                f"{recent*(len(todo)-i-1)/60:.0f} min (elapsed {(time.time()-t_wall)/60:.0f} min)")

    write_batch(sel_report, sides, n_eval, failures, time.time() - t_wall, args)


# =========================================================================== #
# 8. Batch assembly.
# =========================================================================== #

def write_batch(sel_report, sides, n_eval, failures, wall_s, args):
    bd = batch_dir()
    bd.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in load_render_ledger().values():
        for r in rec["rows"]:
            # Backfill for units rendered before the regime axis existed: the stamp is a fact
            # about THIS builder, identical for every row it produces, so deriving it at
            # assembly is correct and makes the field unconditional rather than
            # "present on rows written after a certain hour".
            r["provenance"].setdefault("coloring_source", COLORING_SOURCE)
            rows.append(r)

    # Seeded shuffle for presentation order; the order is STAMPED so the sheet is
    # reproducible from the file rather than re-derived in the browser.
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(rows))
    rows = [rows[int(i)] for i in perm]
    for i, r in enumerate(rows):
        r["sheet_order"] = i

    with (bd / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # The screen-vs-keeper agreement: the proxy audited against the thing it proxied for.
    sp = np.array([r["provenance"]["screen_p_ge3"] for r in rows])
    fp = np.array([r["p_ge3"] for r in rows])
    agree = {
        "n": len(rows),
        "spearman": _spearman(sp, fp),
        "mean_abs_delta": float(np.abs(sp - fp).mean()) if len(rows) else None,
        "gate_side_agreement_at_0.9": float(((sp > 0.9) == (fp > 0.9)).mean()) if len(rows) else None,
        "note": "screen = coarse 512x288 SCORING-ONLY path; keeper = the stored 1280x720 ss2 "
                "crop. The stamped head_v3/p_ge3 is the keeper score; the screen only chose "
                "what to render.",
    }

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "schema_note": "Fresh-era wallpaper CORRECTION SHEET. Locations drawn from the "
                       "CURRENT stage-2 admitted intake (the seven-ledger union + the "
                       "human_q3plus library seed), stratified across deployed-v3 score "
                       "bins with the floor-admitted sources oversampled. Every row carries "
                       "the complete render block + colormap recipe (provenance.params — the "
                       "crop is a pure function of the two), the v3 pre-label "
                       "(head_v3 / p_ge3 / suggested_tier) and a stamped location-grouped "
                       "split_side. label.score is null on every row: the suggestion is NOT "
                       "a label and is never merged as one.",
        "head": {"ckpt": WP.HEAD_CKPT_REL, "version": WP.HEAD_VERSION,
                 "role": "pre-label only — no gate, floor or threshold is applied here",
                 "deploy_transform": "classifier.data.Transform(train=False) "
                                     "(1280x720 -> 384x224 bicubic stretch + normalize)"},
        "suggested_tier_rule": SUGGEST_DERIVATION,
        "population": args._population_report,
        "sampling_metaparameters": {
            "n_screen_candidates": N_SCREEN, "picks_per_loc": args.picks,
            "target_locations": args.target_locs, "palette_pool": PALETTE_POOL,
            "palette_draw": "sample_location.gen0_palettes(sampler, 120) — the deployed "
                            "GEN0_SOURCE_WEIGHTS composition, farthest-point within each "
                            "source bucket; per-location seeded subsample of N_SCREEN, "
                            "params via query_sampler.sample_candidate",
            "maxiter_policy": "auto_maxiter(fw) — native fw-dependent (bootstrap parity)",
            "seed": args.seed, "split_seed": SPLIT_SEED, "eval_frac": EVAL_FRAC,
            "shuffle_seed": SHUFFLE_SEED,
            "pick_rule": "PICKS_PER_LOC spread evenly across the location's own screen "
                         "p_ge3 order (not top-K) — negatives and mid-range within a "
                         "location, not only across locations",
        },
        "selection_report": sel_report,
        "screen_vs_keeper": agree,
        "split_summary": {
            "eval_locations": n_eval,
            "train_locations": sum(1 for v in sides.values() if v == "train"),
            "eval_rows": sum(1 for r in rows if r["provenance"]["split_side"] == "eval"),
            "train_rows": sum(1 for r in rows if r["provenance"]["split_side"] == "train"),
            "rule": "location-grouped, seeded (SPLIT_SEED), stratified by screen score bin; "
                    "one location -> one side",
        },
        "head_corpus_overlap": dict(Counter(r["provenance"]["head_corpus_seen"] for r in rows)),
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "jpg_quality": JPG_Q,
            "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(tools/wallpaper/label_crop.py — the locked label-crop pins)",
        },
        "labels_export": str(LABELS_EXPORT.relative_to(ROOT)).replace("\\", "/"),
        "labeling": {
            "ui": "tools/viz/wallpaper_label.html?batch=" + BATCH_ID,
            "mode": "correction — every row shows its suggested tier PREFILLED; Enter "
                    "confirms, 1-4 overrides. Only rows Matt acts on are exported.",
        },
        "render_failures": failures,
        "run_status": {"planned_locations": sel_report["drawn_locations"],
                       "completed_locations": len(load_render_ledger()),
                       "n_failures": len(failures), "wall_seconds": wall_s},
        "n_rows": len(rows),
    }
    (bd / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")

    log("\n" + "=" * 78)
    log(f"FRESH-ERA CORRECTION SHEET — {BATCH_ID}")
    log("=" * 78)
    log(f"rows {len(rows)}  ·  locations {len(load_render_ledger())}  ·  failures {len(failures)}")
    log(f"suggested tiers: {dict(sorted(Counter(r['suggested_tier'] for r in rows).items()))}")
    log(f"split rows: {dict(Counter(r['provenance']['split_side'] for r in rows))}")
    log(f"screen-vs-keeper spearman {agree['spearman']}  gate-side agreement "
        f"{agree['gate_side_agreement_at_0.9']}")
    log(f"-> {bd}")
    return bd


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else None


def _print_composition(rep):
    log("-" * 78)
    log(f"{'bin':<12} {'range':<16} {'avail':>6} {'floor':>6} {'quota':>6} {'drawn':>6} {'drawn_fl':>9}")
    for b in rep["per_bin"]:
        log(f"{b['bin']:<12} [{b['range'][0]:.2f},{b['range'][1]:.2f})".ljust(29)
            + f"{b['available']:>6} {b['available_floor_admit']:>6} {b['quota']:>6} "
              f"{b['drawn']:>6} {b['drawn_floor_admit']:>9}")
    log(f"drawn {rep['drawn_locations']}  ·  floor-admitted "
        f"{rep['floor_admit_frac_realized']*100:.0f}% (target {rep['floor_admit_frac_target']*100:.0f}%)")
    log(f"by source tag: {rep['drawn_by_source_tag']}")
    log(f"by intake source: {rep['drawn_by_intake_source']}")
    log("-" * 78)


# =========================================================================== #
# Driver.
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description="Fresh-era wallpaper correction sheet builder.")
    ap.add_argument("stage", choices=("estimate", "screen", "select", "render", "write"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="cap units this run (smoke)")
    ap.add_argument("--target-locs", type=int, default=TARGET_LOCS)
    ap.add_argument("--picks", type=int, default=PICKS_PER_LOC)
    ap.add_argument("--estimate", action="store_true", help="render: composition only, no work")
    args = ap.parse_args()

    prio = cc.set_below_normal_priority()
    os.environ.setdefault("RAYON_NUM_THREADS", str(cc.DEFAULT_ENGINE_THREADS))
    log(f"[fresh-sheet] priority {prio} · RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']}")
    WORK.mkdir(parents=True, exist_ok=True)

    srcs, pop_report = population()
    args._population_report = pop_report

    if args.stage == "estimate":
        log(json.dumps(pop_report, indent=2))
        n = len(srcs)
        log(f"\nscreen : {n} locations x (1 field dump + {N_SCREEN} coarse recolors + score)")
        log(f"render : {args.target_locs} locations x {args.picks} label crops "
            f"= {args.target_locs*args.picks} crops")
        log("rates are measured, not assumed — run `screen --limit 8` and read the log.")
        return
    if args.stage == "screen":
        run_screen(args)
        return
    if args.stage == "select":
        screen = load_screen()
        if not screen:
            raise SystemExit("[select] no screen records — run `screen` first")
        _sel, rep = select(list(screen.values()), args.target_locs, args.seed)
        _print_composition(rep)
        log(json.dumps({k: v for k, v in rep.items() if k != "per_bin"}, indent=2))
        return
    if args.stage == "render":
        run_render(args)
        return
    if args.stage == "write":
        screen = load_screen()
        _sel, rep = select(list(screen.values()), args.target_locs, args.seed)
        sides, n_eval = assign_split(_sel)
        write_batch(rep, sides, n_eval, [], 0.0, args)
        return


if __name__ == "__main__":
    main()
