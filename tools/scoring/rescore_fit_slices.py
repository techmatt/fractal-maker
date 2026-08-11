r"""rescore_fit_slices.py — re-stamp a suggestion-cut FIT SLICE under a new head.

WHY. `suggest_tier.INTAKE_CUTS` and `suggest_tier_mining.CUTS` are cutpoints on
`expected_tier = 1 + Sum_k marginal_k`, fitted by `fit_cuts` on a labeled slice. That readout
is a CORN marginal sum, so it is train-prior-calibrated and a cut on it is exactly as
scale-bound as a probability floor: at a head flip the frozen cuts stop describing the head
that will serve the next correction sheet. Re-deriving needs `(pred, human tier)` pairs on the
NEW head — and the `pred` stamped in each batch row was computed at sheet-build time by the
OUTGOING head.

WHAT IT WRITES, AND WHY A SIDECAR RATHER THAN A REWRITE. The batch rows are the sheet's own
record of what it served; overwriting their `pred` would erase the evidence of what the human
was anchored on. So the new head's readout goes to a sibling, `data/<head>/<version>/
fit_slice_pred.json`, keyed by `image_id` — the same shape as `ledger_rescore`'s
`<stem>.rescored_<version>.jsonl` and for the same reason, with the version in the PATH so the
next flip's reader looks for its own file, does not find it, and falls through rather than
reading another head's numbers under its name.

The two torch-free deriver modules then resolve their slice through
`suggest_tier.PRED_SOURCES` / `suggest_tier_mining.PRED_SOURCES`, which map a head version to
either the in-row stamp (for the head that stamped it) or a sidecar. That keeps
`derive_intake_cuts()` / `derive_cuts()` reproducible without a GPU, which is what makes the
"freeze in records, derive in code" test on those constants runnable in the default suite.

  uv run python tools/scoring/rescore_fit_slices.py wallpaper --ckpt data/wallpaper_head/v4b/seed_1/model_best.pt
  uv run python tools/scoring/rescore_fit_slices.py mining --ckpt data/render_mode_head/v3/model_best.pt
  uv run python tools/scoring/rescore_fit_slices.py wallpaper --limit 32   # bounded, writes scratch/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


@dataclass(frozen=True)
class SliceSpec:
    """One head's fit slice. Frozen from the start — the three slices differ only in where
    their rows come from, how many tiers there are, and which sidecar they land in."""
    key: str
    head_name: str
    corpus: str                       # data/<corpus>/batches/<batch>/
    sources: tuple                    # ((batch_dir, labels_sidecar_stem), ...)
    scorer: str                       # "load_head" | "mining_scorer"
    k_tiers: int
    sidecar: str = "fit_slice_pred.json"
    rows_from: str | None = None      # name of a loader in this module; else `sources`


WALLPAPER = SliceSpec(
    key="wallpaper", head_name="wallpaper_head", corpus="wallpaper_corpus",
    sources=(("2026-08-05_wallpaper_fresh_sheet_v1", "wallpaper_fresh_sheet_v1"),
             ("2026-08-05_wallpaper_colorize_path_v1", "wallpaper_colorize_path_v1")),
    scorer="load_head", k_tiers=4)

MINING = SliceSpec(
    key="mining", head_name="render_mode_head", corpus="render_mode_corpus",
    sources=(("2026-08-06_render_mode_fresh_sheet_v1", "render_mode_fresh_sheet_v1"),),
    scorer="mining_scorer", k_tiers=3)

# The 686-row July slice `suggest_tier.CUTS` was fitted on: the dramatic + humanq3 rows on
# the EVAL side. Its crops are not a batch of their own — the eval-revival pass re-rendered
# them, and the six-batch union loader is what holds them now — so this spec is loaded
# through that union rather than by walking a batch dir.
WALLPAPER_JULY = SliceSpec(
    key="wallpaper_july", head_name="wallpaper_head", corpus="wallpaper_corpus",
    sources=(("2026-07-05_wallpaper_humanq3_v1", "-"),
             ("2026-07-09_wallpaper_headbatch_dramatic_v1", "-")),
    scorer="load_head", k_tiers=4,
    sidecar="july_slice_pred.json", rows_from="july_rows")

SPECS = {s.key: s for s in (WALLPAPER, MINING, WALLPAPER_JULY)}


def july_rows(spec: SliceSpec, limit: int | None = None):
    """The dramatic+humanq3 EVAL rows of the six-batch union — `suggest_tier.CUTS`'s slice.

    Read through `train_wallpaper_v4b.split_v4b`, the same loader the (28) verdict used, so
    "the slice CUTS was fitted on" and "the old-era eval arm" cannot become two populations.
    """
    from classifier.train_wallpaper_v4b import load_union, split_v4b   # noqa: PLC0415
    prior, sheet_a = load_union()
    _tr, ev, _meta = split_v4b(prior, sheet_a)
    keep = {"humanq3", "dramatic"}
    rows = sorted(((r.image_id, r.jpg, int(r.label)) for r in ev if r.batch in keep),
                  key=lambda x: x[0])
    return rows[:limit] if limit else rows


def slice_rows(spec: SliceSpec, limit: int | None = None):
    """`[(image_id, jpg, tier), ...]` over the whole slice, in stable id order.

    Raises on an absent source rather than fitting to the remainder — the same refusal
    `suggest_tier.intake_slice` makes, restated here because this pass is what feeds it."""
    if spec.rows_from:
        return globals()[spec.rows_from](spec, limit)
    out = []
    for batch, sidecar in spec.sources:
        bdir = ROOT / "data" / spec.corpus / "batches" / batch
        images, labels = bdir / "images.jsonl", ROOT / "labels" / f"{sidecar}.json"
        for p in (images, labels):
            if not p.exists():
                raise SystemExit(f"[rescore-fit-slice] source absent: {p}")
        lab = json.loads(labels.read_text(encoding="utf-8"))
        for line in images.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = lab.get(row["image_id"])
            if t is None:
                continue
            jpg = bdir / "crops" / f"{row['image_id']}.jpg"
            if not jpg.exists():
                raise SystemExit(f"[rescore-fit-slice] crop missing: {jpg}")
            out.append((row["image_id"], jpg, int(t)))
    out.sort(key=lambda x: x[0])
    return out[:limit] if limit else out


def marginals_of(spec: SliceSpec, ckpt: Path, jpgs) -> np.ndarray:
    """The `(n, K-1)` CORN MARGINAL matrix — `cumprod(sigmoid(logits))`.

    The marginals rather than just their sum, because the record has to answer more than one
    question about the same pass: `pred = 1 + Sum_k marg_k` is the cut readout, and the plain
    CORN 0.5 rule (`1 + #{k : marg_k >= 0.5}`) is the alternative every derivation record
    rejects — a record that stored only the sum could state the rejection but never re-measure
    it on its own head, which is the "outlives the fact it records" failure."""
    if spec.scorer == "load_head":
        import torch                                                   # noqa: PLC0415
        from tools.wallpaper.report_v4_eval import load_head           # noqa: PLC0415
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        score, _cfg = load_head(ckpt, dev)
        _cond, marg, _ssum = score(list(jpgs))
        return np.asarray(marg, dtype=float)
    if spec.scorer == "mining_scorer":
        from tools.mining.mining_gate import MiningScorer              # noqa: PLC0415
        sc = MiningScorer(model_path=str(ckpt))
        # `MiningScore.score` IS sum(cond); the readout wants the MARGINALS. Rebuilt from the
        # two the scorer exposes rather than reusing `score`, which is the conditional sum and
        # a different number wherever cond[1] < 1.
        return np.array([[m.p_ge2, m.p_ge3] for m in sc.score_paths(list(jpgs))], dtype=float)
    raise KeyError(f"no scorer named {spec.scorer!r}")


def head_version(spec: SliceSpec, ckpt_rel: str) -> str:
    parts = Path(ckpt_rel).as_posix().split("/")
    return parts[parts.index(spec.head_name) + 1]


def build(spec: SliceSpec, ckpt_rel: str, *, limit: int | None = None) -> dict:
    rows = slice_rows(spec, limit)
    ckpt = ROOT / ckpt_rel
    t0 = time.time()
    marg = marginals_of(spec, ckpt, [j for _i, j, _t in rows])
    pred = 1.0 + marg.sum(axis=1)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": f"uv run python tools/scoring/rescore_fit_slices.py {spec.key} "
                   f"--ckpt {ckpt_rel}" + (f" --limit {limit}" if limit else ""),
        "what": "the suggestion-cut FIT SLICE's continuous readout, re-stamped under a new "
                "head. `pred = 1 + sum_k marginal_k` (suggest_tier.expected_tier).",
        "head": {"name": spec.head_name, "version": head_version(spec, ckpt_rel),
                 "checkpoint": ckpt_rel,
                 "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()},
        "slice": {"corpus": spec.corpus,
                  "sources": [{"batch": b, "labels": f"labels/{s}.json"}
                              for b, s in spec.sources],
                  "n": len(rows), "k_tiers": spec.k_tiers,
                  "tier_prior": {str(t): sum(1 for _i, _j, x in rows if x == t)
                                 for t in range(1, spec.k_tiers + 1)}},
        "scored_in_s": round(time.time() - t0, 1),
        "readout": "pred = 1 + sum_k marg_k; marg = cumprod(sigmoid(logits)), "
                   "marg[0] = P(tier>=2), marg[1] = P(tier>=3), ...",
        "pred": {iid: round(float(p), 6) for (iid, _j, _t), p in zip(rows, pred)},
        "marg": {iid: [round(float(x), 6) for x in m] for (iid, _j, _t), m in zip(rows, marg)},
        "tier": {iid: int(t) for iid, _j, t in rows},
        "incomplete": bool(limit),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head", choices=sorted(SPECS))
    ap.add_argument("--ckpt", required=True, help="the INCOMING checkpoint (repo-relative)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    spec = SPECS[args.head]
    rec = build(spec, args.ckpt, limit=args.limit)
    ver = rec["head"]["version"]
    out = (ROOT / "scratch" / "rescore_fit_slices" / f"{spec.key}_{ver}.json" if args.limit
           else ROOT / "data" / spec.head_name / ver / spec.sidecar)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

    from tools.wallpaper.suggest_tier import fit_cuts                  # noqa: PLC0415
    rows = slice_rows(spec, args.limit)
    cuts = tuple(round(c, 4) for c in fit_cuts([rec["pred"][i] for i, _j, _t in rows],
                                               [t for _i, _j, t in rows], spec.k_tiers))
    print(f"{spec.key}: n={rec['slice']['n']} prior={rec['slice']['tier_prior']}")
    print(f"  fit_cuts on {ver} = {cuts}")
    print(f"  wrote {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
