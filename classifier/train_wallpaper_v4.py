"""Train the wallpaper-quality head (v4) — the two fresh-era batches folded in.

Forks ``train_wallpaper_v3`` (whose docstring forks v2's; every footgun there stays
pinned). The MODEL is byte-for-byte v2/v3: MobileNetV4, CORN K=4, geometric-only aug,
384x224 stretch, same loss/optim/schedule/seeds. Three things change, and all three
are consequences of the corpus growing, not of a new idea about the model:

  1. **Data = the UNION of FIVE batches**, all fully labeled:
       - bootstrap    (``wbv1_*``):  504 renders /  63 loc   (tier 347/131/26/0)
       - humanq3      (``whq3_*``):  994 renders / 142 loc   (tier 224/403/239/128)
       - dramatic     (``whd_*`` ): 1000 renders / 136 loc   (tier 168/392/277/163)
       - fresh_sheet  (``wfs_*`` ):  960 renders / 240 loc   (tier 475/356/98/31)
       - colorize_path(``wcp_*`` ):  180 renders / 180 loc   (tier  18/ 75/65/22)
     = **3638 renders**. The two 2026-08-05 batches are the FRESH ERA: locations drawn
     from the current stage-2 admitted intake rather than from the July pool, and
     `colorize_path` colours each location the way a live emission run does
     (morph cluster -> deficit-assigned flavor -> pref-v3-gvo argmax) instead of by a
     palette-pool draw. `provenance.coloring_source` separates the two regimes, and
     the 107 locations the pair SHARES are the cleanest coloring contrast the corpus
     has.

  2. **Split is still not re-derived globally (load-bearing).** bootstrap -> train;
     humanq3 -> the identical v2 split (``split_rows``, FIXED ``split_seed=0``);
     dramatic, fresh_sheet and colorize_path -> their STAMPED
     ``provenance.split_side``. So the old-era 686-row eval slice stays byte-identical
     across v2/v3/v4 and the three heads are comparable on it without re-deriving
     anything. The c-inclusive ``full_coord`` disjointness assert now spans all five
     batches, which is what catches the fresh pair's 107 shared locations landing on
     opposite sides (they don't — both used the same seeded location-grouped rule).

  3. **Checkpoint selection is AP>=3 on the POOLED eval side, not AP>=2.** v2/v3
     selected on not-bad AP, which was defensible when the eval side was 686 rows with
     96 good ones. The pooled eval is 1038 rows with 366 good, the >=3 boundary is the
     one the deployed gate actually cuts on, and not-bad AP is saturated (v3 read 0.956
     on the old slice). Recorded in ``config.selection`` and in ``metrics.selection``.

  ALSO: the v2 regression block is SKIPPED, not faked. ``data/wallpaper_head/v2/`` no
  longer exists — the weight and its frozen ``eval_scores.jsonl`` were both deleted, and
  a frozen score file has no rebuild path (re-scoring the slice with a different
  checkpoint would be a different number wearing v2's name). ``metrics.regression_vs_v2``
  carries ``{"skipped": true, "reason": ...}`` naming the missing path. The v3->v4
  comparison, which IS available, is the report's job
  (``tools/wallpaper/report_v4_eval.py``) and not this module's: v3 must be scored on the
  same crops through the same harness, and that is a two-checkpoint pass.

  HOLD: this stages v4 but does NOT flip ACTIVE. wallpaper_pins still points at v3.

    uv run python -m classifier.train_wallpaper_v4 --seeds "0 1 2 3 4"

Outputs -> data/wallpaper_head/v4/  (per-seed under v4/seed_<s>/).
"""
from __future__ import annotations

import gc
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus.eval_only import assert_eval, coord_key, eval_only_ids  # noqa: E402

from .data import Transform
from .model import BACKBONE, corn_loss, data_config
from .train_v2 import detect_device, set_seed
# Reuse v2's eval battery verbatim (no metric drift) and its split for the humanq3 half.
from .train_wallpaper_v2 import (
    K, build_montage, build_wallpaper_model, eval_block, label_hist, make_loader,
    predict_all, split_rows as split_v2, _ap, _nan,
)

BATCHES = ROOT / "data" / "wallpaper_corpus" / "batches"


@dataclass(frozen=True)
class BatchSource:
    """One batch in the training union. `loc_re` extracts the LOCATION id from an
    image_id (a location contributes several palettes); `stamped_split` says whether
    the batch carries its own `provenance.split_side` or is placed by rule."""
    name: str
    id_re: str
    batch_id: str
    labels: str
    stamped_split: bool
    era: str                    # "july" | "fresh"

    @property
    def dir(self) -> Path:
        return BATCHES / self.batch_id

    @property
    def labels_path(self) -> Path:
        return ROOT / "labels" / self.labels


# Order matters: bootstrap + humanq3 must be first — they are fed to v2's split verbatim.
SOURCES = [
    BatchSource("bootstrap", r"(wbv1_\d+)_\d+$", "2026-07-05_wallpaper_bootstrap_v1",
                "wallpaper_bootstrap_v1.json", False, "july"),
    BatchSource("humanq3", r"(whq3_\d+)_\d+$", "2026-07-05_wallpaper_humanq3_v1",
                "wallpaper_humanq3_v1.json", False, "july"),
    BatchSource("dramatic", r"(whd_\d+)_\d+$", "2026-07-09_wallpaper_headbatch_dramatic_v1",
                "wallpaper_headbatch_dramatic_v1.json", True, "july"),
    BatchSource("fresh_sheet", r"(wfs_\d+)_\d+$", "2026-08-05_wallpaper_fresh_sheet_v1",
                "wallpaper_fresh_sheet_v1.json", True, "fresh"),
    # colorize_path is one render per location — the id has no _NN pick suffix.
    BatchSource("colorize_path", r"(wcp_\d+)$", "2026-08-05_wallpaper_colorize_path_v1",
                "wallpaper_colorize_path_v1.json", True, "fresh"),
]
OUT_DIR = ROOT / "data" / "wallpaper_head" / "v4"
V2_EVAL_SCORES = ROOT / "data" / "wallpaper_head" / "v2" / "eval_scores.jsonl"
SPLIT_SEED = 0  # FIXED — reproduces v2's humanq3 split byte-identically.

log = logging.getLogger("train_wallpaper_v4")


# --------------------------------------------------------------------------- #
# Rows — one per render (image_id). No crop aggregation (v2 contract).
# --------------------------------------------------------------------------- #
@dataclass
class WRow:
    image_id: str
    label: int
    jpg: Path
    loc: str
    fractal_type: str
    batch: str            # BatchSource.name
    era: str              # "july" | "fresh"
    family: str
    coord: tuple          # (cx,cy,fw,type) — v2-compatible key for split_v2()
    full_coord: tuple     # (cx,cy,fw,type,c_re,c_im) — c-inclusive disjointness key
    palette_source: str   # "dramatic" | "pool"  (the JULY palette axis; v3 parity)
    coloring_source: str  # "pool_draw" | "colorize_path" | "july_pool"  (the FRESH axis)
    source_group: str     # "human_q3plus" | "q4_harvest" | "machine_admitted" | "july"
    floor_admit: bool | None
    split_side: str | None    # stamped batches: "train"/"eval"; else None
    split_origin: str | None
    v3_p_ge3: float | None    # the deployed-v3 keeper score stamped at build time


def _source_group(prov) -> str:
    """The fresh-era intake vein. `human_q3plus` and `q4_harvest` are the two
    FLOOR-ADMITTED sources — same admission rule, and the sitting read gave them
    opposite verdicts (33.6% vs 7.4% of rows >=3), so they are never pooled here.
    Everything else came in through the machine gate."""
    tag = prov.get("source_tag")
    if tag in ("human_q3plus", "q4_harvest"):
        return tag
    return "machine_admitted"


def load_rows(require_crops: bool = True) -> list[WRow]:
    """`require_crops=False` is for --dry-run and for the split-invariant tests: the
    split is a function of the rows alone, and demanding ~900 MB of regenerable JPGs
    to answer "does any location span both sides?" would make the check unrunnable
    exactly when the crops have been lost — which is the state this batch set was in."""
    rows: list[WRow] = []
    for src in SOURCES:
        labels = json.loads(src.labels_path.read_text())
        loc_re = re.compile(src.id_re)
        seen: set[str] = set()
        for line in (src.dir / "images.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["image_id"]
            if iid not in labels:
                raise ValueError(f"[{src.name}] row {iid} has no label — batch must be fully labeled")
            m = loc_re.match(iid)
            if m is None:
                raise ValueError(f"[{src.name}] image_id does not match expected prefix: {iid}")
            jpg = src.dir / "crops" / f"{iid}.jpg"
            if require_crops and not jpg.exists():
                raise FileNotFoundError(
                    f"crop missing: {jpg}\n"
                    f"The July crops are bulk-regenerable and were deleted once already — "
                    f"rebuild with: uv run python tools/wallpaper/rerender_batch_crops.py all")
            rd = r["render"]
            prov = r["provenance"]
            coord = (rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"])
            full_coord = (rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"],
                          rd.get("c_re"), rd.get("c_im"))
            if src.stamped_split:
                side = prov["split_side"]
                origin = prov["split_origin"]
            else:
                side = origin = None
            # July's palette axis: only the dramatic batch stamps it; boot/hq3 are pool.
            psrc = prov.get("palette_source", "pool") if src.era == "july" else "fresh"
            rows.append(WRow(
                image_id=iid, label=int(labels[iid]), jpg=jpg, loc=m.group(1),
                fractal_type=rd["fractal_type"], batch=src.name, era=src.era,
                family=prov["family"], coord=coord, full_coord=full_coord,
                palette_source=psrc,
                coloring_source=(prov.get("coloring_source", "pool_draw")
                                 if src.era == "fresh" else "july_pool"),
                source_group=(_source_group(prov) if src.era == "fresh" else "july"),
                floor_admit=prov.get("floor_admit"),
                split_side=side, split_origin=origin,
                v3_p_ge3=(r.get("head_v3") or {}).get("p_ge3"),
            ))
            seen.add(iid)
        extra = set(labels) - seen
        if extra:
            raise ValueError(f"[{src.name}] {len(extra)} labels have no row: {sorted(extra)[:5]}...")
    return rows


# --------------------------------------------------------------------------- #
# Union split. humanq3 comes from v2's split (fixed seed); the three stamped
# batches honor their own side; bootstrap -> train. Disjointness on full_coord.
# --------------------------------------------------------------------------- #
# The batch whose stamped side WINS when two batches stamp one location differently.
# See `reconcile_stamped_sides` for why this is needed at all and why it is the sheet.
SPLIT_AUTHORITY = "fresh_sheet"

WALLPAPER_CORPUS = "wallpaper_corpus"


def assert_eval_only_pinned(rows, side_of, *, where: str) -> dict:
    """No row at an EVAL-ONLY batch's coordinate may be on the train side. Ever.

    Runs on the split this module BUILT, keyed on the same c-inclusive coordinate the
    disjointness assert uses — so it also catches a future batch that re-renders a
    sheet-D location under a fresh image_id. It is a THIRD constraint on top of the two
    §2a offers (freeze one batch set, or re-derive globally): both of those choose an
    authority for a contested location, and neither knows that some slices may never
    train at all. Inert while no loaded batch is eval-only, which is every trainer
    today — the assert exists for the retrain that folds sheet D in."""
    forced = eval_only_ids(WALLPAPER_CORPUS, key_of=coord_key)
    side = {}
    for r in rows:
        s = side_of(r)
        if side.setdefault(r.full_coord, s) != s:
            raise AssertionError(f"{r.full_coord} is on two sides before the eval-only "
                                 f"pin is even checked — {where}")
    return assert_eval(side, forced, where=where)


def reconcile_stamped_sides(rows: list[WRow], stamped_side):
    """Resolve locations two batches stamped onto opposite sides. Returns
    (side_override_by_image_id, conflict_report).

    THE CONFLICT IS REAL AND IT IS NOT A TYPO. `build_fresh_sheet.assign_split` shuffles
    the location keys *within each screen-score bin* under a fixed seed and cuts the
    first `eval_frac`. That is a function of the SELECTED SET, not of the location: the
    colorize sibling drew 180 locations overlapping the sheet's 240 in 107, so inside a
    bin its key list is different, the shuffle orders it differently, and the cut lands
    elsewhere. `colorize_path`'s batch.json records "the sibling's assign_split, same
    seed", which is true of the call and false of the result — 19 of the 107 shared
    locations disagree. Nothing caught it because nothing joined the two batches until
    this trainer did, and a location on both sides is training-on-eval.

    The sheet is authority for two reasons: it is the sibling's declared parent, and it
    is the cheaper correction — colorize carries ONE render per location (19 rows move)
    while the sheet carries four (~76 rows would move, perturbing its own bin
    stratification and the 296-row fresh eval side the report compares against).

    Reported, never silent: every reassignment is listed in metrics."""
    by_coord: dict[tuple, list[WRow]] = defaultdict(list)
    for r in rows:
        by_coord[r.full_coord].append(r)

    override: dict[str, str] = {}
    conflicts = []
    for coord, grp in by_coord.items():
        sides = {r.batch: stamped_side(r) for r in grp}
        if len(set(sides.values())) < 2:
            continue
        if SPLIT_AUTHORITY not in sides:
            raise AssertionError(
                f"location {coord} spans sides across {sorted(sides)} — no {SPLIT_AUTHORITY} "
                f"row to arbitrate. A conflict outside the fresh pair is a different bug; "
                f"do not widen the authority rule to paper over it.")
        win = sides[SPLIT_AUTHORITY]
        moved = []
        for r in grp:
            if stamped_side(r) != win:
                override[r.image_id] = win
                moved.append(r.image_id)
        conflicts.append({"coord": [str(x) for x in coord], "resolved_to": win,
                          "stamped": sides, "moved_image_ids": moved})
    return override, conflicts


def split_union(rows: list[WRow]):
    boot_hq3 = [r for r in rows if r.batch in ("bootstrap", "humanq3")]
    _, ev_v2, hq3_eval_locs, strata, forced = split_v2(boot_hq3, eval_frac=0.30, seed=SPLIT_SEED)
    assert all(r.batch == "humanq3" for r in ev_v2), "v2 eval must be humanq3-only"
    stamped = {s.name for s in SOURCES if s.stamped_split}

    def stamped_side(r: WRow) -> str:
        """The side the batch itself declares, before reconciliation."""
        if r.batch == "bootstrap":
            return "train"
        if r.batch == "humanq3":
            return "eval" if r.loc in hq3_eval_locs else "train"
        if r.batch in stamped:
            if r.split_side not in ("train", "eval"):
                raise ValueError(f"{r.batch} row {r.image_id} has bad split_side={r.split_side!r}")
            return r.split_side
        raise ValueError(f"no split rule for batch {r.batch!r}")

    override, conflicts = reconcile_stamped_sides(rows, stamped_side)

    def side_of(r: WRow) -> str:
        return override.get(r.image_id, stamped_side(r))

    train = [r for r in rows if side_of(r) == "train"]
    ev = [r for r in rows if side_of(r) == "eval"]

    # Location-disjointness across ALL FIVE batches on the c-inclusive key. Post-
    # reconciliation this must be clean by construction; it is asserted anyway, because
    # "by construction" is the claim that was made about the sibling split.
    coord_sides: dict[tuple, set] = defaultdict(set)
    for r in rows:
        coord_sides[r.full_coord].add(side_of(r))
    spanning = {c for c, s in coord_sides.items() if len(s) > 1}
    if spanning:
        raise AssertionError(f"{len(spanning)} locations STILL span both sides after "
                             f"reconciliation (e.g. {list(spanning)[:3]})")

    old_slice_ids = {r.image_id for r in ev if r.batch == "humanq3"}
    if old_slice_ids != {r.image_id for r in ev_v2}:
        raise AssertionError("humanq3 eval slice diverged from v2 — old slice not byte-identical")
    # The old-era eval side must be untouched by any of this — it is the anchor the
    # v2/v3/v4 comparison rests on.
    if any(r.era == "july" for r in rows if r.image_id in override):
        raise AssertionError("reconciliation moved a JULY row — the old-era slice must be inert")
    assert_eval_only_pinned(rows, side_of, where="train_wallpaper_v4.split_union")
    return train, ev, hq3_eval_locs, strata, forced, sorted(old_slice_ids), conflicts


# --------------------------------------------------------------------------- #
# One training run (single seed).
# --------------------------------------------------------------------------- #
def train_one_seed(seed, tr, ev, args, device, train_tf, deploy_tf, cfg, seed_dir):
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True).to(device)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = make_loader(tr, train_tf, args.batch_size, device, train=True,
                               num_workers=args.num_workers, seed=seed)
    eval_loader = make_loader(ev, deploy_tf, args.batch_size, device, train=False,
                              num_workers=min(4, args.num_workers))
    eval_labels = np.asarray([r.label for r in ev])

    best_sel, best_epoch = -1.0, -1
    best_state = best_cond = best_marg = best_sum = None
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            loss = corn_loss(logits, (y - 1).long(), num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(tr)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"[seed {seed}] NaN/Inf at epoch {epoch} — aborting seed"); break

        cond, marg, ssum = predict_all(model, eval_loader, len(ev), device)
        ap_nb = _ap((eval_labels >= 2).astype(int), marg[:, 0])
        ap_gd = _ap((eval_labels >= 3).astype(int), marg[:, 1])
        ap_ex = _ap((eval_labels >= 4).astype(int), marg[:, 2]) if (eval_labels >= 4).any() else float("nan")
        # SELECTION = pooled-eval AP>=3 (v4 change; v2/v3 selected on AP>=2).
        sel = -1.0 if (ap_gd is None or not np.isfinite(ap_gd)) else ap_gd
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_not_bad": _nan(ap_nb), "ap_good": _nan(ap_gd),
                        "ap_exceptional": _nan(ap_ex), "selection_metric": sel})
        log.info(f"[seed {seed}] epoch {epoch:2d}  loss {train_loss:.4f}  AP_nb {ap_nb:.4f}  "
                 f"AP_good {ap_gd:.4f}*  AP_exc {ap_ex:.4f}  ({time.time()-t0:.1f}s)")
        if sel > best_sel:
            best_sel, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
    log.info(f"[seed {seed}] best epoch {best_epoch}: pooled-eval AP>=3 {best_sel:.4f} "
             f"(wall {time.time()-t_start:.0f}s)")

    seed_cfg = dict(cfg, seed=seed, best_epoch=best_epoch, split_seed=SPLIT_SEED)
    torch.save({"state_dict": best_state, "config": seed_cfg}, seed_dir / "model_best.pt")

    with open(seed_dir / "eval_scores.jsonl", "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "batch": r.batch, "era": r.era,
                "family": r.family, "fractal_type": r.fractal_type,
                "palette_source": r.palette_source, "coloring_source": r.coloring_source,
                "source_group": r.source_group, "floor_admit": r.floor_admit,
                "label": r.label,
                "p_ge2": float(best_marg[i, 0]), "p_ge3": float(best_marg[i, 1]),
                "p_ge4": float(best_marg[i, 2]),
                "p_not_bad": float(best_cond[i, 0]), "p_good_cond": float(best_cond[i, 1]),
                "p_exc_cond": float(best_cond[i, 2]), "score": float(best_sum[i]),
                "v3_p_ge3": r.v3_p_ge3,
            }) + "\n")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"seed": seed, "best_epoch": best_epoch, "val_best_ap_good": _nan(best_sel),
            "history": history, "checkpoint": str(seed_dir / "model_best.pt")}, \
        best_cond, best_marg, best_sum


def agg(blocks, key):
    vals = [b[key] for b in blocks if b is not None and b.get(key) is not None]
    if not vals:
        return {"mean": None, "sd": None, "n_seeds": 0, "values": []}
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=0)),
            "n_seeds": len(vals), "values": [float(v) for v in a]}


def fmt(d):
    return "  n/a" if d["mean"] is None else f"{d['mean']:.3f}+/-{d['sd']:.3f}"


def regression_vs_v2(old_rows, ev, agg_metrics):
    """The v2 comparison, or an explicit SKIP naming why.

    v2's frozen eval scores are gone with the rest of `data/wallpaper_head/v2/`. A
    frozen score file is not regenerable: re-scoring the slice under any surviving
    checkpoint produces a different number, and putting it under v2's name would be a
    fabricated baseline. So this reports absence rather than reconstructing it."""
    if not V2_EVAL_SCORES.exists():
        return {
            "skipped": True,
            "reason": ("v2_eval_scores_missing: "
                       f"{V2_EVAL_SCORES.relative_to(ROOT).as_posix()} does not exist. The v2 "
                       "head and its frozen eval_scores.jsonl were both deleted; a frozen "
                       "per-render score file has no rebuild path (re-scoring the slice with "
                       "another checkpoint is a different quantity, not v2's). NOT "
                       "reconstructed. The available and reported baseline is v3 — see "
                       "tools/wallpaper/report_v4_eval.py."),
            "old_slice_n": len(old_rows),
        }
    v2_scores = {}
    for line in V2_EVAL_SCORES.read_text().splitlines():
        if line.strip():
            d = json.loads(line); v2_scores[d["image_id"]] = d
    ids = {r.image_id for r in old_rows}
    if ids != set(v2_scores):
        raise AssertionError("old-slice ids != v2 eval_scores ids — regression compare invalid")
    lb = np.asarray([r.label for r in old_rows])
    v2_nb = _ap((lb >= 2).astype(int), np.asarray([v2_scores[r.image_id]["p_ge2"] for r in old_rows]))
    v2_gd = _ap((lb >= 3).astype(int), np.asarray([v2_scores[r.image_id]["p_ge3"] for r in old_rows]))
    v4_old = agg_metrics["old_humanq3"]
    return {
        "skipped": False, "old_slice_n": len(old_rows),
        "v2_ap_not_bad": _nan(v2_nb), "v2_ap_good": _nan(v2_gd),
        "v4_ap_not_bad": v4_old["ap_not_bad"], "v4_ap_good": v4_old["ap_good"],
        "delta_good_mean": (None if v4_old["ap_good"]["mean"] is None or v2_gd is None
                            else float(v4_old["ap_good"]["mean"] - v2_gd)),
    }


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train wallpaper head v4 (five-batch union).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--seeds", default="0 1 2 3 4", help="space-separated train seeds (>=3)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--border-crop", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="load + split + report the union, then exit (no training)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    if len(seeds) < 3 and not args.dry_run:
        raise SystemExit("need >=3 seeds for a measured band")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  seeds={seeds}")

    # --- data + split (once; split is seed-independent) ---
    rows = load_rows(require_crops=not args.dry_run)
    log.info(f"loaded {len(rows)} renders  by_batch={dict(Counter(r.batch for r in rows))}  "
             f"union_tier_hist={label_hist(rows)}")
    for s in SOURCES:
        sub = [r for r in rows if r.batch == s.name]
        log.info(f"  {s.name:14s}: {len(sub):4d} renders / {len({r.loc for r in sub}):3d} loc  "
                 f"{label_hist(sub)}")

    tr, ev, hq3_eval_locs, strata, forced, old_slice_ids, conflicts = split_union(rows)
    log.info(f"=== union split (split_seed={SPLIT_SEED}, FIXED) ===")
    if conflicts:
        moved = [i for c in conflicts for i in c["moved_image_ids"]]
        log.warning(f"  SPLIT CONFLICT: {len(conflicts)} locations were stamped onto opposite "
                    f"sides by two batches; resolved to {SPLIT_AUTHORITY}'s side, moving "
                    f"{len(moved)} rows. See metrics.split_conflicts.")
    log.info(f"  train {len(tr)} renders / {len({r.loc for r in tr})} loc  {label_hist(tr)}  "
             f"by_batch={dict(Counter(r.batch for r in tr))}")
    log.info(f"  eval  {len(ev)} renders / {len({r.loc for r in ev})} loc  {label_hist(ev)}  "
             f"by_batch={dict(Counter(r.batch for r in ev))}")
    eh = label_hist(ev)
    log.info(f"  eval good (tier>=3) = {eh[3]+eh[4]}  (tier3={eh[3]}, tier4={eh[4]})  "
             f"[v3's eval had 275: 185+90]")
    log.info(f"  old slice (humanq3 eval, byte-identical to v2/v3) = {len(old_slice_ids)}")
    log.info(f"  old-era eval (july batches) = {sum(1 for r in ev if r.era=='july')}  "
             f"fresh-era eval = {sum(1 for r in ev if r.era=='fresh')}")
    log.info(f"  eval coloring_source: {dict(Counter(r.coloring_source for r in ev))}")
    log.info(f"  eval source_group:    {dict(Counter(r.source_group for r in ev))}")

    # --- config / transforms (identical to v2/v3) ---
    probe = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    del probe
    log.info(f"data_config: {data_cfg}")
    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)
    cfg = {
        "model": "wallpaper_head_v4", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=3, K pinned=4)", "geometry": "stretch",
        "label_unit": "render (image_id) — NO max-over-crops",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "class_weighting": "none", "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
        "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": ("max POOLED-eval AP>=3 (marginal P>=3); full schedule (no early stop). "
                      "CHANGED from v2/v3, which selected on not-bad AP>=2: the pooled eval "
                      "is 1038 rows / 366 good, >=3 is the boundary the deployed gate cuts "
                      "on, and >=2 AP was saturated (v3 read 0.956 on the old slice)."),
        "split": ("bootstrap->train; humanq3->v2 split (split_seed=0, byte-identical); "
                  "dramatic/fresh_sheet/colorize_path->stamped split_side; disjointness "
                  "asserted on the c-inclusive key across all five batches"),
        "batch_ids": [s.batch_id for s in SOURCES],
        "init": "imagenet_backbone_fresh (NOT warm-started)",
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "split_seed": SPLIT_SEED, "seeds": seeds, "forced_train_side": forced,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    eval_labels = np.asarray([r.label for r in ev])
    era = np.asarray([r.era for r in ev])
    cs = np.asarray([r.coloring_source for r in ev])
    sg = np.asarray([r.source_group for r in ev])
    eval_batch = np.asarray([r.batch for r in ev])

    slices = {
        "overall": np.ones(len(ev), bool),
        "old_era": era == "july",
        "fresh_era": era == "fresh",
        "old_humanq3": eval_batch == "humanq3",
        "old_dramatic": eval_batch == "dramatic",
        "fresh_pool_draw": cs == "pool_draw",
        "fresh_colorize_path": cs == "colorize_path",
        "fresh_human_q3plus": sg == "human_q3plus",
        "fresh_q4_harvest": sg == "q4_harvest",
        "fresh_machine_admitted": sg == "machine_admitted",
    }
    if args.dry_run:
        log.info("=== DRY RUN — split only, no training ===")
        for name, mask in slices.items():
            sub = [r for i, r in enumerate(ev) if mask[i]]
            log.info(f"  {name:24s} n={int(mask.sum()):4d}  {label_hist(sub)}")
        return

    # --- multi-seed train ---
    per_seed = []
    seed_blocks = defaultdict(list)
    best_for_stage = None
    for seed in seeds:
        log.info(f"================= SEED {seed} =================")
        info, cond, marg, ssum = train_one_seed(
            seed, tr, ev, args, device, train_tf, deploy_tf, cfg, out_dir / f"seed_{seed}")
        per_seed.append(info)
        for name, mask in slices.items():
            seed_blocks[name].append(eval_block(eval_labels, cond, marg, ssum, mask))
        sel = info["val_best_ap_good"] or -1.0
        if best_for_stage is None or sel > best_for_stage[0]:
            best_for_stage = (sel, seed, cond, marg, ssum, out_dir / f"seed_{seed}")
        (out_dir / "per_seed.json").write_text(json.dumps(per_seed, indent=2))

    # --- cross-seed aggregation ---
    log.info("================= CROSS-SEED AGGREGATION =================")
    agg_metrics = {}
    headline_keys = ["ap_not_bad", "ap_good", "ap_exceptional", "ap_4_vs_3",
                     "auc_good_vs_rest", "auc_4_vs_3", "spearman_score_vs_tier"]
    for name, blocks in seed_blocks.items():
        b0 = next((b for b in blocks if b is not None), None)
        agg_metrics[name] = {
            "n": (b0["n"] if b0 else 0), "n_good": (b0["n_good"] if b0 else 0),
            "n_exceptional": (b0["n_exceptional"] if b0 else 0),
            **{k: agg(blocks, k) for k in headline_keys},
        }
        m = agg_metrics[name]
        log.info(f"  [{name:24s}] n={m['n']:4d} good={m['n_good']:3d} exc={m['n_exceptional']:3d}  "
                 f"AP_nb {fmt(m['ap_not_bad'])}  AP_good {fmt(m['ap_good'])}  "
                 f"AP_exc {fmt(m['ap_exceptional'])}  AUC>=3 {fmt(m['auc_good_vs_rest'])}")

    old_rows = [r for r in ev if r.batch == "humanq3"]
    reg = regression_vs_v2(old_rows, ev, agg_metrics)
    if reg["skipped"]:
        log.warning(f"=== REGRESSION vs v2: SKIPPED — {reg['reason']}")

    # --- stage the checkpoint (best pooled AP>=3 seed) — HOLD, do NOT flip ---
    stage_sel, stage_seed, s_cond, s_marg, s_sum, s_dir = best_for_stage
    import shutil
    shutil.copy(s_dir / "model_best.pt", out_dir / "model_best.pt")
    shutil.copy(s_dir / "eval_scores.jsonl", out_dir / "eval_scores.jsonl")
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")
    try:
        build_montage(ev, s_sum, out_dir / "eval_montage.png")
    except Exception as e:
        log.warning(f"montage failed: {e}")

    metrics = {
        "seeds": seeds, "split_seed": SPLIT_SEED,
        "selection": {
            "metric": "pooled-eval AP>=3 (marginal P>=3)",
            "changed_from": "v2/v3 selected on not-bad AP>=2",
            "why": ("the >=3 boundary is what the deployed gate cuts on, the pooled eval "
                    "carries 366 good rows, and >=2 AP was saturated on the old slice"),
            "staged_seed": stage_seed, "staged_value": float(stage_sel),
        },
        "train_n": len(tr), "eval_n": len(ev), "eval_tier_hist": eh,
        "eval_by_batch": dict(Counter(r.batch for r in ev)),
        "eval_by_era": dict(Counter(r.era for r in ev)),
        "eval_by_coloring_source": dict(Counter(r.coloring_source for r in ev)),
        "eval_by_source_group": dict(Counter(r.source_group for r in ev)),
        "old_slice_n": len(old_slice_ids), "split_strata": strata,
        "forced_train_side": forced,
        "split_conflicts": {
            "n_locations": len(conflicts),
            "n_rows_moved": sum(len(c["moved_image_ids"]) for c in conflicts),
            "authority": SPLIT_AUTHORITY,
            "why": ("build_fresh_sheet.assign_split is a function of the SELECTED SET, not "
                    "of the location: the colorize sibling drew a different 180-location "
                    "population, so the same seed shuffles a different within-bin key list "
                    "and cuts elsewhere. The sibling's batch.json claim of an identical "
                    "split is true of the call and false of the result."),
            "detail": conflicts,
        },
        "aggregate": agg_metrics, "per_seed": per_seed,
        "regression_vs_v2": reg,
        "staged": {"seed": stage_seed, "ap_good": float(stage_sel),
                   "checkpoint": str(out_dir / "model_best.pt"),
                   "rule": "best per-seed pooled-eval AP>=3",
                   "ACTIVE_STATUS": "STAGED — NOT flipped; wallpaper_pins still points at v3",
                   "one_line_flip": ("edit tools/wallpaper/wallpaper_pins.py: "
                                     'HEAD_CKPT_REL = "data/wallpaper_head/v4/model_best.pt"'),
                   "rollback": 'revert HEAD_CKPT_REL to "data/wallpaper_head/v3/model_best.pt"'},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info("================= WALLPAPER HEAD v4 SUMMARY =================")
    ov, oe, fe = agg_metrics["overall"], agg_metrics["old_era"], agg_metrics["fresh_era"]
    log.info(f"  seeds={seeds}  train n={len(tr)}  eval n={len(ev)} (good {eh[3]+eh[4]})")
    log.info(f"  OVERALL    n={ov['n']:4d}  not-bad {fmt(ov['ap_not_bad'])}  "
             f"good {fmt(ov['ap_good'])}  tier4 {fmt(ov['ap_exceptional'])}")
    log.info(f"  OLD-ERA    n={oe['n']:4d}  good {fmt(oe['ap_good'])}  AUC>=3 {fmt(oe['auc_good_vs_rest'])}")
    log.info(f"  FRESH-ERA  n={fe['n']:4d}  good {fmt(fe['ap_good'])}  AUC>=3 {fmt(fe['auc_good_vs_rest'])}")
    log.info(f"  STAGED -> {out_dir / 'model_best.pt'}  (seed {stage_seed}, HELD; ACTIVE still v3)")
    log.info(f"  FLIP: {metrics['staged']['one_line_flip']}")
    log.info("DONE")


if __name__ == "__main__":
    main()
