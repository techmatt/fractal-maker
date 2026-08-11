r"""mining_v3_reads.py — mining v1 vs v3 on ONE harness, winner rule applied, STAGED ONLY.

`prompts/retrains_28.md`, mining half. Both checkpoints are scored over the SAME crops
through `mining_gate.MiningScorer`, which is head-agnostic by construction (backbone, K,
mean/std, geometry all read from the checkpoint's own config) — so nothing here
re-implements a transform or a probability, and the numbers describe the scorer that
actually gates.

THE SLICES, ALL DECLARED IN `SLICES` ABOVE THE CODE THAT COMPUTES THEM:

  MOTIVATING (must be significantly better)
    `busy_fp` — the busy false-positive slice: sheet B's `hi_fancy` bucket (the deliberate
    high-mining-score over-draw of composite+direct modes) UNION sheet C's fancy rows. This
    is the cell the (27) sheets were built to correct.

  NO-WORSE (none may be significantly worse)
    `pooled`      the whole deduplicated eval side
    `rare_palette` sheet C
    `mode:<m>`    every mode with both classes present at a boundary

  DIAGNOSTIC (reported, votes on nothing)
    the two halves of the motivating slice on their own, each source sheet, and the
    v1-UNSEEN sub-slice below.

FOUR THINGS THAT LEAN THIS COMPARISON, and three of them lean the SAME way:

  1. v1 trained at the 112 gate-passer locations that the v1 sitting and sheet B both draw
     from, and its own dataset is gone so the exact rows cannot be excluded. 630 of the 827
     eval rows sit at a location v1 has seen. v1 is being read partly on memory.
  2. Every sheet in this corpus is a CORRECTION sheet: rows were served with v1's own
     suggested tier prefilled and sorted by its score, so label and v1's score are coupled
     by construction.
  3. v3's staged checkpoint is the best of five seeds BY eval AP>=3 on this very slice, so
     the staged number is optimistic. The five-seed band is reported beside it.
  (1) and (2) inflate v1; (3) inflates v3. `v1_unseen` — the eval rows at locations v1 never
  saw, essentially sheet C's new locations — is the only cell where (1) and (2) do not
  apply, and it is reported as a diagnostic for exactly that reason.

  4. SUBSTRATE. Sheets B and C were coloured under different recipes (head-picked params
     with `transfer=grad` and gamma ~1.7 vs `deploy_tail._color_params({})`, the canonical
     emission colouring). A pooled tier rate across them is not a rate about anything, so
     every table here is reported per source sheet as well as pooled — including the
     motivating slice, whose two halves come one from each sheet and whose good base rates
     differ by a factor of ~5.

DERIVED AND RECORDED. No pin, gate, floor, lock or annotation moves here;
`mining_pins.ACTIVE_MINING_CKPT` is read and never written.

Outputs -> data/render_mode_head/v3/report.{md,json}

  uv run python tools/mining/mining_v3_reads.py
  uv run python tools/mining/mining_v3_reads.py --limit 64     # bounded end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.corpus.q4_combined_readout import wilson                     # noqa: E402
from tools.emission import floors as F                                  # noqa: E402
from tools.mining import mining_pins as MP                              # noqa: E402
from tools.mining.mining_corpus import BATCH_TAG, label_hist, load_corpus  # noqa: E402
from tools.scoring.winner_rule import (Metric, paired_bootstrap,        # noqa: E402
                                       point_block, verdict)
from tools.v10.prereg import hanley_mcneil_se, min_detectable_auc       # noqa: E402

V1_CKPT = MP.ACTIVE_MINING_CKPT                       # the incumbent, read off the pin
V3_DIR = ROOT / "data" / "render_mode_head" / "v3"
V3_CKPT = "data/render_mode_head/v3/model_best.pt"
# The candidate is a DIRECTORY, not a version: the (28) arms (`v3`, `v3_aug`, `v3_augx`,
# `v3_uniform`) are four single-variable passes over the identical corpus, split, objective
# and eval, and each is read against v1 through this same file. `--candidate-dir` moves it;
# the arm NAME comes out of the candidate's own config.json rather than the path, so a
# report cannot describe the wrong experiment because a directory was renamed.
FANCY_KINDS = frozenset({"composite", "direct"})      # tools.mining.build_mining_correction

# The metric set — K=3, so two boundaries and two statistics each. Fixed above the code.
METRICS = (Metric("auc_ge3", "AUC>=3", "p_ge3", 3, "auc"),
           Metric("ap_ge3", "AP>=3", "p_ge3", 3, "ap"),
           Metric("auc_ge2", "AUC>=2", "p_ge2", 2, "auc"),
           Metric("ap_ge2", "AP>=2", "p_ge2", 2, "ap"))

SWEEP = sorted({round(x, 3) for x in np.arange(0.0, 1.0, 0.05)}
               | {F.MINING_POOL.value, F.MINING_RELEASE.value})

# Which metrics each no-worse arm VOTES on. The prompt names the per-mode arms as "per-mode
# AUCs where two classes exist" — AUCs, not APs — so a mode arm submits its two AUC cells and
# nothing else. Everything is still COMPUTED and printed for every arm; this only decides
# what the rule reads, and it is fixed here rather than after the cells are seen. It matters:
# clause (a) is a conjunction, and doubling a 16-mode arm set from 32 cells to 64 doubles the
# chance a head loses to one unlucky mode.
AUC_ONLY = tuple(m for m in METRICS if m.kind == "auc")


def voting_metrics(arm: str):
    return AUC_ONLY if arm.startswith("mode:") else METRICS


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# slices — declared as predicates, before any number exists
# --------------------------------------------------------------------------- #
def slice_masks(rows) -> tuple[dict, dict, dict]:
    """`(motivating, no_worse, diagnostic)`, each `{name: bool mask}`."""
    batch = np.array([r.batch for r in rows])
    kind = np.array([r.kind for r in rows])
    bucket = np.array([r.bucket or "" for r in rows])
    mode = np.array([r.mode for r in rows])

    b_hi_fancy = (batch == "sheetB") & (bucket == "hi_fancy")
    c_fancy = (batch == "sheetC") & np.isin(kind, sorted(FANCY_KINDS))

    motivating = {"busy_fp": b_hi_fancy | c_fancy}
    no_worse = {"pooled": np.ones(len(rows), bool),
                "rare_palette": batch == "sheetC"}
    for m in sorted(set(mode.tolist())):
        no_worse[f"mode:{m}"] = mode == m
    diagnostic = {
        "sheetB_hi_fancy": b_hi_fancy,
        "sheetC_fancy": c_fancy,
        "v1_sitting": batch == "v1_sitting",
        "sheetB": batch == "sheetB",
        "fancy_all": np.isin(kind, sorted(FANCY_KINDS)),
        "pure_all": kind == "pure",
    }
    return motivating, no_worse, diagnostic


# --------------------------------------------------------------------------- #
def score_with(ckpt: str, rows) -> dict:
    """`{p_ge2, p_ge3, rank}` for one checkpoint, through the production scorer."""
    from tools.mining.mining_gate import MiningScorer      # noqa: PLC0415 (torch import)
    sc = MiningScorer(model_path=ckpt)
    res = sc.score_paths([r.jpg for r in rows])
    return {"p_ge2": np.array([x.p_ge2 for x in res]),
            "p_ge3": np.array([x.p_ge3 for x in res]),
            "rank": np.array([x.score for x in res])}


def tier_dist(labels) -> dict:
    c = Counter(int(x) for x in labels)
    n = len(labels)
    return {"n": n, "hist": {str(t): c.get(t, 0) for t in (1, 2, 3)},
            "frac_ge2": (n - c.get(1, 0)) / n if n else None,
            "frac_ge3": c.get(3, 0) / n if n else None}


def arm_block(labels, base, cand, mask, *, draws, seed) -> dict:
    """One arm: both heads' point values, the paired CI, and this cell's own power."""
    lb = labels[mask]
    b = {k: v[mask] for k, v in base.items()}
    c = {k: v[mask] for k, v in cand.items()}
    out = {"n": int(mask.sum()), "tiers": tier_dist(lb),
           "v1": point_block(lb, b, METRICS), "v3": point_block(lb, c, METRICS),
           "delta_ci": paired_bootstrap(lb, b, c, METRICS, draws=draws, seed=seed)}
    for m in METRICS:
        npos, nneg = out["v1"][f"{m.key}__n_pos"], out["v1"][f"{m.key}__n_neg"]
        if m.kind == "auc" and npos and nneg and out["v1"][m.key] is not None:
            out.setdefault("power", {})[m.key] = {
                "min_detectable_auc": min_detectable_auc(npos, nneg),
                "v1_auc_se": hanley_mcneil_se(out["v1"][m.key], npos, nneg)}
    return out


def ladder(labels, s, *, thr=3, grid=SWEEP) -> list:
    good = int((labels >= thr).sum())
    out = []
    for t in grid:
        fire = s >= t
        nf = int(fire.sum())
        k = int((labels[fire] >= thr).sum())
        p, lo, hi = wilson(k, nf) if nf else (None, None, None)
        out.append({"threshold": float(t), "fires": nf,
                    "pass_rate": nf / len(s) if len(s) else None, "tp": k,
                    "precision": p, "precision_lo": lo, "precision_hi": hi,
                    "recall": k / good if good else None,
                    "marks": [f.name for f in (F.MINING_POOL, F.MINING_RELEASE)
                              if abs(f.value - t) < 1e-9]})
    return out


def volume_matched(labels, base, cand) -> dict:
    """Both heads at EQUAL selected volume — the comparison a scale shift cannot corrupt.

    A fixed threshold is a point on ONE head's probability scale, so a v1 cut applied to v3
    selects a different number of rows and the two 'operating points' are not the same
    experiment. This takes whatever volume v1 passes at each live cut and asks what v3's top
    that-many rows look like."""
    n = len(labels)
    def top(s, k):
        k = int(min(max(k, 0), n))
        if not k:
            return None
        order = np.argsort(-s, kind="stable")[:k]
        tp = int((labels[order] >= 3).sum())
        p, lo, hi = wilson(tp, k)
        good = int((labels >= 3).sum())
        return {"n_selected": k, "pass_rate": k / n, "tp": tp, "precision": p,
                "precision_lo": lo, "precision_hi": hi,
                "recall": tp / good if good else None,
                "cut_at": float(s[order[-1]])}
    out = {"by_v1_live_cut": {}, "by_fixed_rate": {}}
    for f in (F.MINING_POOL, F.MINING_RELEASE):
        v = int((base["p_ge3"] >= f.value).sum())
        out["by_v1_live_cut"][f.name] = {"threshold_on_v1": f.value, "matched_volume": v,
                                         "v1": top(base["p_ge3"], v),
                                         "v3": top(cand["p_ge3"], v)}
    for rate in (0.05, 0.10, 0.20):
        v = int(round(rate * n))
        out["by_fixed_rate"][f"{rate:.2f}"] = {"matched_volume": v,
                                               "v1": top(base["p_ge3"], v),
                                               "v3": top(cand["p_ge3"], v)}
    return out


# --------------------------------------------------------------------------- #
def build(rows, base, cand, seed_scores, pool, *, draws, seed, cand_dir=None) -> dict:
    labels = np.array([r.label for r in rows])
    motivating, no_worse, diagnostic = slice_masks(rows)
    cand_dir = Path(cand_dir or V3_DIR)
    cand_rel = cand_dir.relative_to(ROOT).as_posix() if cand_dir.is_absolute() else str(cand_dir)
    cand_cfg = {}
    cfg_p = cand_dir / "config.json"
    if cfg_p.exists():
        cand_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))

    R = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": f"uv run python tools/mining/mining_v3_reads.py "
                   f"--candidate-dir {cand_rel}",
        "arm": cand_cfg.get("arm", cand_dir.name),
        # `selection_metric` is the (28b) fifth arm's dial and belongs beside the other three:
        # an arm read without it looks identical to `dedup_weighted` in this block, which is
        # exactly the confusion the arm exists to resolve.
        "arm_dials": {k: cand_cfg.get(k) for k in
                      ("border_crop", "axis_crop", "uniform_weights", "row_weighting",
                       "selection_metric")},
        "candidate_selection": cand_cfg.get("selection"),
        "train_weight_by_kind": cand_cfg.get("train_weight_by_kind"),
        "heads": {"v1": {"ckpt": V1_CKPT, "role": "incumbent (LIVE pin)"},
                  "v3": {"ckpt": f"{cand_rel}/model_best.pt",
                         "role": "from-scratch candidate, STAGED"}},
        "live_pin": MP.ACTIVE_MINING_CKPT,
        "moves_nothing": "no pin, gate, floor, lock or annotation is written by this file",
        "eval_slice": {
            "n": len(rows), "n_locations": len({r.loc for r in rows}),
            "deduplicated": "one row per near-dup group (colored-CLIP, cut 0.974)",
            "n_before_dedup": len(pool.eval_all),
            "by_batch": dict(Counter(r.batch for r in rows)),
            "tiers": tier_dist(labels)},
        "split": pool.split_meta, "near_dup": pool.group_meta,
        "bootstrap": {"draws": draws, "seed": seed, "kind": "paired over eval rows"},
        "cuts": {f.name: {"value": f.value, "stamp": f"{f.head}/{f.stamp}"}
                 for f in (F.MINING_POOL, F.MINING_RELEASE)},
    }

    # --- harness parity: is this the scorer that stamped the sheets? ------------
    stamped = np.array([r.v1_p_ge3 for r in rows])
    ok = np.isfinite(stamped)
    d = np.abs(stamped[ok] - base["p_ge3"][ok])
    R["harness_parity"] = {
        "what": "v1 re-scored here vs the head_mining_v1.p_ge3 stamped into images.jsonl at "
                "sheet-build time. Same checkpoint, same deploy transform, days apart.",
        "n": int(ok.sum()), "max_abs_diff": float(d.max()) if d.size else None,
        "mean_abs_diff": float(d.mean()) if d.size else None,
        "tol": 1e-6, "ok": bool(d.size and d.max() < 1e-6)}

    # --- the arms --------------------------------------------------------------
    def arms(masks):
        return {k: arm_block(labels, base, cand, m, draws=draws, seed=seed)
                for k, m in masks.items() if m.sum() > 0}

    R["motivating"] = arms(motivating)
    R["no_worse"] = arms(no_worse)
    R["diagnostic"] = arms(diagnostic)

    # v1-unseen: the cell where neither the memorisation nor the anchoring caveat applies.
    v1_locs = {r.loc for r in pool.rows if r.batch == "v1_sitting"}
    unseen = np.array([r.loc not in v1_locs for r in rows])
    if unseen.sum():
        R["diagnostic"]["v1_unseen_locations"] = arm_block(
            labels, base, cand, unseen, draws=draws, seed=seed)
        R["diagnostic"]["v1_unseen_locations"]["what"] = (
            "eval rows at locations the v1 sitting never served — v1 has not memorised "
            "these and their labels were not anchored to a v1 suggestion at a location it "
            "had already judged. The only cell where the two v1-inflating caveats are off.")

    # --- v3's five-seed band (the staged checkpoint is eval-selected) -----------
    if seed_scores:
        band = []
        for s, sc in sorted(seed_scores.items()):
            band.append({"seed": s, **{m.key: point_block(labels, sc, METRICS)[m.key]
                                       for m in METRICS}})
        R["v3_seed_band"] = {
            "per_seed": band,
            "mean_sd": {m.key: {"mean": float(np.mean([b[m.key] for b in band])),
                                "sd": float(np.std([b[m.key] for b in band], ddof=0))}
                        for m in METRICS},
            "note": "each seed's own best-epoch checkpoint, every one of them selected on "
                    "THIS eval side; the band is over seeds, not over held-out populations, "
                    "and the staged max is optimistic against it."}

    # --- scale + volume --------------------------------------------------------
    R["score_scale"] = {
        "quantiles": {f"q{int(q*100)}": {"v1": float(np.quantile(base["p_ge3"], q)),
                                         "v3": float(np.quantile(cand["p_ge3"], q))}
                      for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "pass_rate_at": {f"{t:.2f}": {"v1": float((base["p_ge3"] >= t).mean()),
                                      "v3": float((cand["p_ge3"] >= t).mean())}
                         for t in (F.MINING_POOL.value, F.MINING_RELEASE.value, 0.75, 0.90)},
        "why": "a fixed threshold is a point on ONE head's scale; the live cuts are NOT "
               "restated here, and a flip would have to volume-match them "
               "(classifier_retrain_protocol.md §5a)."}
    R["volume_matched"] = volume_matched(labels, base, cand)
    R["ladder_v3_ge3"] = ladder(labels, cand["p_ge3"], thr=3)
    R["ladder_v1_ge3"] = ladder(labels, base["p_ge3"], thr=3)

    # --- the rule --------------------------------------------------------------
    R["voting_cells"] = {k: [m.key for m in voting_metrics(k)] for k in R["no_worse"]}
    R["winner_rule"] = verdict(
        {k: {m.key: v["delta_ci"][m.key] for m in voting_metrics(k)}
         for k, v in R["no_worse"].items()},
        {k: v["delta_ci"] for k, v in R["motivating"].items()},
        pooled_arm="pooled", baseline="v1", candidate="v3")
    R["winner_rule"]["candidate_ckpt"] = (
        f"{cand_rel}/model_best.pt" if R["winner_rule"]["winner"] == "v3" else V1_CKPT)
    R["winner_rule"]["adoption"] = ("NOT decided here. BUILD != FLIP: adoption is a separate "
                                    "prompt after Matt reads this verdict.")
    return R


# --------------------------------------------------------------------------- #
def md(R) -> str:
    L = []
    A = L.append
    wr = R["winner_rule"]
    A(f"# mining v1 vs v3 [arm: {R.get('arm', 'dedup_weighted')}] — winner-rule verdict "
      f"(STAGED, nothing adopted)\n")
    A(f"Generated {R['generated']} · `{R['command']}`\n")
    if R.get("arm_dials"):
        A(f"Arm dials: `{R['arm_dials']}`\n")
    A(f"**WINNER: {wr['winner']}** (pooled-only reading: {wr['winner_pooled_only']})  ")
    A(f"clause (a) no-worse {'PASS' if wr['clause_a']['pass'] else 'FAIL'} over "
      f"{wr['clause_a']['n_tests']} arm x metric cells · clause (b) motivating "
      f"{'PASS' if wr['clause_b']['pass'] else 'FAIL'}\n")
    A(f"> {wr['adoption']}\n")

    e = R["eval_slice"]
    A(f"Eval slice: **{e['n']} rows** ({e['n_before_dedup']} before near-dup dedup) over "
      f"{e['n_locations']} locations, {e['by_batch']}; tiers {e['tiers']['hist']}.\n")
    A(f"Harness parity (v1 re-scored vs stamped): max |Δ| "
      f"{R['harness_parity']['max_abs_diff']:.2e} over {R['harness_parity']['n']} rows — "
      f"{'OK' if R['harness_parity']['ok'] else 'DRIFT'}.\n")

    def table(title, arms, roles):
        A(f"\n## {title}\n")
        A("| arm | n | ge3 | v1 AUC≥3 | v3 AUC≥3 | Δ 95% CI | v1 AP≥3 | v3 AP≥3 | Δ 95% CI |")
        A("|---|---:|---:|---:|---:|---|---:|---:|---|")
        for name in roles:
            b = arms.get(name)
            if b is None:          # an arm with no rows on this slice is absent, not blank
                continue
            ci3, cia = b["delta_ci"]["auc_ge3"], b["delta_ci"]["ap_ge3"]
            def cell(x):
                return "—" if x is None else f"{x:.3f}"
            def ic(c):
                if not c or c["n_draws"] == 0:
                    return "n/a"
                tag = "**worse**" if c["significantly_worse"] else (
                    "**better**" if c["significantly_better"] else "")
                return f"[{c['lo']:+.3f}, {c['hi']:+.3f}] {tag}"
            A(f"| `{name}` | {b['n']} | {b['tiers']['hist']['3']} | {cell(b['v1']['auc_ge3'])} "
              f"| {cell(b['v3']['auc_ge3'])} | {ic(ci3)} | {cell(b['v1']['ap_ge3'])} "
              f"| {cell(b['v3']['ap_ge3'])} | {ic(cia)} |")

    table("MOTIVATING arm", R["motivating"], sorted(R["motivating"]))
    table("NO-WORSE arms", R["no_worse"],
          ["pooled", "rare_palette"] + sorted(k for k in R["no_worse"] if k.startswith("mode:")))
    table("Diagnostics (vote on nothing)", R["diagnostic"], sorted(R["diagnostic"]))

    if wr["clause_a"]["failures"]:
        A("\n### clause (a) failures\n")
        for f in wr["clause_a"]["failures"]:
            A(f"- `{f['arm']}` **{f['metric']}**: Δ median {f['median']:+.3f}, "
              f"95% CI [{f['lo']:+.3f}, {f['hi']:+.3f}]")
    if wr["clause_b"]["improvements"]:
        A("\n### clause (b) improvements\n")
        for f in wr["clause_b"]["improvements"]:
            A(f"- `{f['arm']}` **{f['metric']}**: Δ median {f['median']:+.3f}, "
              f"95% CI [{f['lo']:+.3f}, {f['hi']:+.3f}]")
    A(f"\n{wr['multiplicity_note']}\n")

    if "v3_seed_band" in R:
        A("\n## v3 five-seed band (staged = the max of these, on this same slice)\n")
        A("| metric | mean ± sd | per seed |")
        A("|---|---|---|")
        for m in METRICS:
            b = R["v3_seed_band"]["mean_sd"][m.key]
            vals = " ".join(f"{s[m.key]:.3f}" for s in R["v3_seed_band"]["per_seed"])
            A(f"| {m.label} | {b['mean']:.3f} ± {b['sd']:.3f} | {vals} |")

    A("\n## Volume-matched (a fixed threshold is a point on ONE head's scale)\n")
    A("| volume | v1 precision≥3 | v3 precision≥3 | n |")
    A("|---|---:|---:|---:|")
    for name, blk in R["volume_matched"]["by_v1_live_cut"].items():
        if blk["v1"] and blk["v3"]:
            A(f"| {name} (v1 @ {blk['threshold_on_v1']}) | {blk['v1']['precision']:.3f} "
              f"| {blk['v3']['precision']:.3f} | {blk['matched_volume']} |")
    for name, blk in R["volume_matched"]["by_fixed_rate"].items():
        if blk["v1"] and blk["v3"]:
            A(f"| top {name} | {blk['v1']['precision']:.3f} | {blk['v3']['precision']:.3f} "
              f"| {blk['matched_volume']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="eval rows — bounded end-to-end; the report it writes is STAMPED "
                         "incomplete and goes to scratch/, never the run dir")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--candidate-dir", "--v3-dir", dest="v3_dir", type=Path, default=V3_DIR,
                    help="the arm's run dir (default data/render_mode_head/v3)")
    a = ap.parse_args(argv)

    cand_dir = a.v3_dir if a.v3_dir.is_absolute() else (ROOT / a.v3_dir)
    cand_ckpt = cand_dir / "model_best.pt"
    if not (ROOT / V1_CKPT).exists():
        raise SystemExit(f"[v3-reads] missing incumbent checkpoint {V1_CKPT}")
    if not cand_ckpt.exists():
        raise SystemExit(f"[v3-reads] missing candidate checkpoint {cand_ckpt} — train it "
                         f"first (uv run python -m classifier.train_mining_head_v3 "
                         f"--out-dir {cand_dir})")

    pool = load_corpus()
    rows = pool.eval_rows
    if a.limit:
        rows = rows[:a.limit]
    log(f"[v3-reads] eval slice {len(rows)} rows  {label_hist(rows)}")

    base = score_with(V1_CKPT, rows)
    cand = score_with(str(cand_ckpt), rows)
    seed_scores = {}
    if not a.limit:
        for d in sorted(cand_dir.glob("seed_*")):
            ck = d / "model_best.pt"
            if ck.exists():
                seed_scores[int(d.name.split("_")[1])] = score_with(str(ck), rows)
                log(f"[v3-reads] scored {d.name}")

    R = build(rows, base, cand, seed_scores, pool, draws=a.draws, seed=a.seed,
              cand_dir=cand_dir)
    R["incomplete"] = bool(a.limit)
    out = (ROOT / "scratch" / f"mining_reads_{cand_dir.name}") if a.limit else cand_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(md(R), encoding="utf-8")
    wr = R["winner_rule"]
    log(f"[v3-reads] WINNER {wr['winner']} (pooled-only {wr['winner_pooled_only']}) — "
        f"clause a {wr['clause_a']['pass']} / clause b {wr['clause_b']['pass']}")
    log(f"-> {out / 'report.md'}")


if __name__ == "__main__":
    main()
