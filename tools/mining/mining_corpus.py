r"""mining_corpus.py — THE pooled render-mode corpus: rows, global split, near-dup weights.

Torch-free, so the trainer and the reads harness hold ONE definition of "which rows, which
side, what weight" rather than two that are supposed to agree. Same split as
`tools/mining/mining_pins.py` is to the pin: the thing every reader needs, without the GPU
stack the producer needs.

THE THREE BATCHES (the whole labeled render-mode corpus as of 2026-08-10):

    2026-08-06_render_mode_fresh_sheet_v1     960 rows / 112 loc   the v1 sitting
    2026-08-10_render_mode_correction_v2    1,000 rows /  91 loc   sheet B
    2026-08-10_render_mode_rare_palette_v1    500 rows / 235 loc   sheet C

THE SPLIT IS RE-DERIVED GLOBALLY, AND IT HAS TO BE. Each batch stamped its own
`provenance.split_side`, and honoring those stamps is not merely suboptimal — it is wrong:

  * sheet B's 91 locations are a SUBSET of the v1 sitting's 112 (it draws the unserved
    (location, mode) pairs of the same gate-passer population), and the two batches ran
    `split_units.build_split` independently over different location sets. The seeded draw is
    a function of the SET, so **33 of the 91 shared locations are stamped train by one batch
    and eval by the other**. Any pooled run that honors the stamps trains on its own eval.
    This is the same failure `train_wallpaper_v4.reconcile_stamped_sides` found between the
    fresh pair, arriving by the same route.
  * sheet C is stamped 100% train (`batch_registry`: no location of it may be an eval
    INSTRUMENT), which would leave the rare-palette slice unmeasurable. The distinction that
    resolves it is `classifier_retrain_protocol.md` §1's two eval roles: an instrument is an
    unbiased draw a base rate may be read from, a HOLDOUT is biased exactly as training is.
    The render-mode corpus has no instrument anywhere in it and never has; its eval side is
    a holdout, and no base rate is read from it here or anywhere else.

So: `split_units.build_split` over the POOLED 339 locations, at the module's own stamped
`EVAL_FRAC` and `SPLIT_SEED` — union-find over Julia-seed == parent-plane point, stratified
by family, drawn over UNITS. Not re-chosen, imported.

NEAR-DUP GROUPS carry through from `near_dup_groups.py` (colored CLIP, cut 0.974, within a
location). They are used TWO different ways, and the asymmetry is deliberate:

  * TRAIN — every row is kept and weighted `1 / group_size`. A duplicated look then
    contributes exactly one look's worth of gradient while every distinct label still
    reaches the model.
  * EVAL — one row per group, so AUC/AP/the paired bootstrap are ordinary unweighted
    statistics over distinct pictures. A weighted AUC is definable but the bootstrap over it
    is not the same object, and the eval side's job here is to be comparable between two
    heads rather than to use every row.

The representative is the group's MEDIAN-label row (lower median on a tie, then lowest
image_id): a group whose members disagree keeps the middle judgement rather than the
alphabetically first one. `group_label_disagreement()` reports how often that is even a
question.

    from tools.mining.mining_corpus import load_corpus, MiningPool
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.mining.mining_roster import MODE_KIND                       # noqa: E402
from tools.mining.near_dup_groups import ARTIFACT as NEAR_DUP_ARTIFACT  # noqa: E402
from tools.mining.near_dup_groups import BATCHES as POOL_BATCHES        # noqa: E402
from tools.mining.split_units import (EVAL_FRAC, SPLIT_SEED,            # noqa: E402
                                      JULIA_PARENT, build_split, units_are_disjoint)

CORPUS = ROOT / "data" / "render_mode_corpus" / "batches"
K = 3   # tiers 1 bad / 2 okay / 3 good. The render-mode corpus's ceiling (merge_sitting).

# Short batch tags used by every slice name and table row.
BATCH_TAG = {
    "2026-08-06_render_mode_fresh_sheet_v1": "v1_sitting",
    "2026-08-10_render_mode_correction_v2": "sheetB",
    "2026-08-10_render_mode_rare_palette_v1": "sheetC",
}


@dataclass(frozen=True)
class MiningRow:
    image_id: str
    label: int
    jpg: Path
    loc: str
    mode: str
    kind: str            # pure | direct | composite  (mining_roster.MODE_KIND)
    family: str
    fractal_type: str
    batch: str           # the tag, not the dir name
    bucket: str | None   # the draw bucket the sheet recorded (sheet B/C only)
    side: str            # train | eval   — the POOLED split, never the stamped one
    stamped_side: str    # what the batch itself stamped (kept so the move is auditable)
    group: str           # near-dup group id
    weight: float        # 1 / group_size  (train weighting)
    is_rep: bool         # the group's representative (the eval row)
    v1_p_ge3: float      # mining v1's stamped in-row score, for the drift check
    v1_p_ge2: float


@dataclass(frozen=True)
class MiningPool:
    rows: list[MiningRow]
    split_meta: dict
    group_meta: dict

    @property
    def train(self) -> list[MiningRow]:
        return [r for r in self.rows if r.side == "train"]

    @property
    def eval_rows(self) -> list[MiningRow]:
        """The DEDUPLICATED eval side — one row per near-dup group."""
        return [r for r in self.rows if r.side == "eval" and r.is_rep]

    @property
    def eval_all(self) -> list[MiningRow]:
        return [r for r in self.rows if r.side == "eval"]


def _sidecar(batch: str) -> dict:
    gv = json.loads((CORPUS / batch / "batch.json").read_text(encoding="utf-8"))[
        "generator_version"]
    return json.loads((ROOT / "labels" / f"{gv}.json").read_text(encoding="utf-8"))


def _raw_rows(batches=POOL_BATCHES, require_crops: bool = True) -> list[dict]:
    """Every row of every pooled batch, label joined from its tracked sidecar.

    An unlabeled row RAISES: this corpus is three completed sittings, and a silently
    dropped row is a training set nobody can reproduce from the batch manifests."""
    out = []
    for b in batches:
        bdir = CORPUS / b
        labels = _sidecar(b)
        for line in (bdir / "images.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["image_id"]
            if iid not in labels:
                raise ValueError(f"[{b}] row {iid} has no label — the pooled corpus is "
                                 f"three COMPLETE sittings")
            lab = int(labels[iid])
            if not (1 <= lab <= K):
                raise ValueError(f"[{b}] {iid}: label {lab} outside 1..{K}")
            jpg = bdir / "crops" / f"{iid}.jpg"
            if require_crops and not jpg.exists():
                raise FileNotFoundError(f"crop missing: {jpg}")
            out.append({"row": r, "batch": b, "label": lab, "jpg": jpg})
        extra = set(labels) - {o["row"]["image_id"] for o in out if o["batch"] == b}
        if extra:
            raise ValueError(f"[{b}] {len(extra)} labels name no row: {sorted(extra)[:5]}")
    return out


def _representatives(recs, group_of) -> set:
    """One image_id per group: the MEDIAN label (lower median), tie-broken lowest id."""
    by_group = defaultdict(list)
    for o in recs:
        by_group[group_of[o["row"]["image_id"]]].append(o)
    reps = set()
    for g, members in by_group.items():
        med = statistics.median_low(sorted(o["label"] for o in members))
        cand = sorted((o["row"]["image_id"] for o in members if o["label"] == med))
        reps.add(cand[0])
    return reps


def group_label_disagreement(recs, group_of) -> dict:
    """How often a near-dup group's members carry different labels — the cost of keeping
    one. Reported, never assumed away: if it is large, "same picture" is the wrong claim."""
    by_group = defaultdict(list)
    for o in recs:
        by_group[group_of[o["row"]["image_id"]]].append(o["label"])
    multi = {g: v for g, v in by_group.items() if len(v) > 1}
    split = {g: v for g, v in multi.items() if len(set(v)) > 1}
    spans = Counter(max(v) - min(v) for v in multi.values())
    return {"n_multi_groups": len(multi), "n_groups_with_disagreement": len(split),
            "share": (len(split) / len(multi)) if multi else None,
            "label_span_hist": {str(k): v for k, v in sorted(spans.items())}}


def load_corpus(*, batches=POOL_BATCHES, require_crops: bool = True,
                seed: int = SPLIT_SEED, eval_frac: float = EVAL_FRAC) -> MiningPool:
    recs = _raw_rows(batches, require_crops=require_crops)
    doc = json.loads(NEAR_DUP_ARTIFACT.read_text(encoding="utf-8"))
    if doc.get("incomplete"):
        raise SystemExit(f"[mining-corpus] {NEAR_DUP_ARTIFACT} is stamped incomplete "
                         f"(--limit {doc.get('limit_per_batch')}) — rebuild it unbounded "
                         f"before training on it")
    if list(doc["batches"]) != list(batches):
        raise SystemExit(f"[mining-corpus] near-dup artifact covers {doc['batches']}, "
                         f"pool is {list(batches)} — rebuild it")
    group_of = doc["group_of"]
    missing = [o["row"]["image_id"] for o in recs
               if o["row"]["image_id"] not in group_of]
    if missing:
        raise SystemExit(f"[mining-corpus] {len(missing)} rows have no near-dup group "
                         f"(e.g. {missing[:3]}) — the artifact is stale")

    # --- the pooled, globally re-derived split (see the module docstring) ---
    locs = {}
    for o in recs:
        pv, rd = o["row"]["provenance"], o["row"]["render"]
        locs.setdefault(pv["location_key"], {"family": pv["family"], "render": rd})
    side, split_meta = build_split(locs, seed=seed, eval_frac=eval_frac)
    ok, msg = units_are_disjoint(side, locs)
    if not ok:
        raise AssertionError(f"[mining-corpus] {msg}")
    split_meta["disjointness"] = msg

    sizes = Counter(group_of[o["row"]["image_id"]] for o in recs)
    reps = _representatives(recs, group_of)

    rows = []
    for o in recs:
        r, pv, rd = o["row"], o["row"]["provenance"], o["row"]["render"]
        iid = r["image_id"]
        g = group_of[iid]
        h = r.get("head_mining_v1") or {}
        rows.append(MiningRow(
            image_id=iid, label=o["label"], jpg=o["jpg"], loc=pv["location_key"],
            mode=rd["render_mode"], kind=pv.get("mode_kind") or MODE_KIND[rd["render_mode"]],
            family=pv["family"], fractal_type=rd["fractal_type"],
            batch=BATCH_TAG[o["batch"]], bucket=pv.get("bucket"),
            side=side[pv["location_key"]], stamped_side=pv["split_side"],
            group=g, weight=1.0 / sizes[g], is_rep=(iid in reps),
            v1_p_ge3=float(h.get("p_ge3", float("nan"))),
            v1_p_ge2=float(h.get("p_ge2", float("nan")))))

    # A near-dup group is a subset of ONE location by construction (pairs are compared
    # within a location), and a location is inside one split unit — so a group cannot
    # straddle. Asserted rather than assumed: it is the property the prompt requires and
    # it would break silently if the grouping scope ever widened.
    by_group = defaultdict(set)
    for r in rows:
        by_group[r.group].add(r.side)
    straddle = [g for g, s in by_group.items() if len(s) > 1]
    if straddle:
        raise AssertionError(f"[mining-corpus] {len(straddle)} near-dup groups straddle "
                             f"train/eval (e.g. {straddle[:3]})")

    moved = sum(1 for r in rows if r.side != r.stamped_side)
    group_meta = {
        "artifact": doc["artifact"], "cut": doc["cut"], "substrate": doc["substrate"],
        "n_groups": doc["n_groups"], "group_size_hist": doc["group_size_hist"],
        "n_rows_in_a_multi_group": doc["n_rows_in_a_multi_group"],
        "n_pairs_cross_batch": doc["n_pairs_cross_batch"],
        "disagreement": group_label_disagreement(recs, group_of),
        "eval_rows_before_dedup": sum(1 for r in rows if r.side == "eval"),
        "eval_rows_after_dedup": sum(1 for r in rows if r.side == "eval" and r.is_rep),
        "train_weight_sum": round(sum(r.weight for r in rows if r.side == "train"), 3),
    }
    split_meta["rows_moved_off_stamped_side"] = moved
    split_meta["stamped_vs_pooled"] = {
        f"{a}->{b}": sum(1 for r in rows if r.stamped_side == a and r.side == b)
        for a in ("train", "eval") for b in ("train", "eval")}
    split_meta["by_batch"] = {
        t: dict(Counter(r.side for r in rows if r.batch == t)) for t in BATCH_TAG.values()}
    return MiningPool(rows=rows, split_meta=split_meta, group_meta=group_meta)


def label_hist(rows) -> dict:
    c = Counter(r.label for r in rows)
    return {k: c.get(k, 0) for k in range(1, K + 1)}


def summary(pool: MiningPool) -> dict:
    tr, ev = pool.train, pool.eval_rows
    return {
        "n_rows": len(pool.rows), "n_locations": pool.split_meta["n_locations"],
        "n_units": pool.split_meta["n_units"],
        "train_rows": len(tr), "train_label_hist": label_hist(tr),
        "eval_rows_deduped": len(ev), "eval_label_hist": label_hist(ev),
        "eval_rows_raw": len(pool.eval_all),
        "by_batch_side": pool.split_meta["by_batch"],
        "eval_by_batch": dict(Counter(r.batch for r in ev)),
        "eval_by_kind": dict(Counter(r.kind for r in ev)),
        "eval_by_mode": dict(sorted(Counter(r.mode for r in ev).items())),
        "julia_families_linked": sorted(JULIA_PARENT),
    }


if __name__ == "__main__":                      # a readout, not a build
    p = load_corpus(require_crops=False)
    print(json.dumps({"split": p.split_meta, "groups": p.group_meta,
                      "summary": summary(p)}, indent=1, default=str))
