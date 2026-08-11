r"""sheet_e_reverdict.py — the ONE-COMMAND re-verdict for mining v1 vs the five staged arms
on SHEET E, the BLIND render-mode slice.

    uv run python tools/mining/sheet_e_reverdict.py

That is the whole invocation. It runs AFTER Matt labels sheet E and merges the export
(`merge_sitting.py --corpus render_mode_corpus --batch 2026-08-11_render_mode_blind_v1 …
--apply`); before that it exits with the merge command rather than half a report.

WHAT IT ANSWERS, and it is exactly the three questions the (28)/(28b) verdicts could not:

  (a) **DO THE CONTESTED CELLS SURVIVE ON UNANCHORED LABELS?** Every clause-(a) failure in
      the five staged arms was measured on the pooled render-mode corpus, and all three of
      its batches are CORRECTION sheets — mining v1's suggested tier prefilled, page ordered
      by its score, 0.929 of the mining sitting's labels equal to what was served. So a
      per-mode cell where v1 beats an arm is partly a statement that the labels agree with
      v1. Sheet E is fresh (location, mode) pairs of the same population, served blind. The
      failing cells are READ OUT of each arm's committed `report.json` rather than restated
      here, and each one is re-run on this slice: SURVIVES / DOES NOT SURVIVE /
      UNMEASURABLE, per cell, per arm.

  (b) **THE POOLED READS.** Each arm's pooled AUC/AP at >=2 and >=3 against v1's, with the
      paired CI, on labels neither head suggested.

  (c) **THE ANCHORING PRICE.** v1's pooled AUC>=2 on the anchored corpus (0.953 there) set
      beside v1's pooled AUC>=2 here. The gap is what the prefilled suggestion is worth as a
      predictor of the label it suggested, measured rather than argued. DERIVED from
      `data/render_mode_head/v3/report.json`, never restated as a constant.

CLAUSE (b) IS NOT DECIDED HERE, and that is deliberate. The (28) motivating arm is `busy_fp`
— sheet B's `hi_fancy` bucket unioned with sheet C's fancy rows — a bucket defined by those
sheets' own draw. Sheet E has no such bucket and inventing one after seeing this slice would
be choosing a motivating arm from the data. This file re-asks clause (a) ONLY, plus the
pooled and per-mode reads, and says so in its own verdict block.

EVERY HEAD IS SCORED THROUGH ONE LOADER (`mining_v3_reads.score_with` ->
`mining_gate.MiningScorer`), which is head-agnostic by construction: backbone, K, mean/std and
geometry all come from the checkpoint's own config. The metric set and the per-arm voting rule
are IMPORTED from `mining_v3_reads`, so a cell here is the same cell there.

WHAT LEANS THIS COMPARISON: nothing that favours any head. Sheet E's (location, mode) pairs
are fresh to all six, its labels were elicited with no suggestion on the page, its palette
came from a pool draw and its rows are stamped eval-only. Two caveats worth stating and both
cut against over-reading rather than toward a head: it is ONE slice of ~150 rows, so a null is
"not distinguishable at this n"; and the five staged checkpoints were each selected on the
(28) POOLED eval, which makes sheet E the first held-out population any of them has seen.

MOVES NOTHING. `mining_pins.ACTIVE_MINING_CKPT` is read and never written; no pin, gate,
floor, lock or annotation changes. Adoption stays a separate prompt.

Outputs -> data/render_mode_head/sheet_e_reverdict/report.{md,json}

    uv run python tools/mining/sheet_e_reverdict.py
    uv run python tools/mining/sheet_e_reverdict.py --limit 32   # bounded, writes scratch/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.mining import mining_pins as MP                            # noqa: E402
from tools.mining import mining_v3_reads as V3R                       # noqa: E402
from tools.mining.build_blind_mining_sheet import (CONTESTED_MODES,   # noqa: E402
                                                   SHEETS)
from tools.scoring.winner_rule import paired_bootstrap, point_block   # noqa: E402

SHEET = SHEETS["e"]
HEAD_DIR = ROOT / "data" / "render_mode_head"
OUT_DIR = HEAD_DIR / "sheet_e_reverdict"

# THE FIVE STAGED ARMS of (28) + (28b), declared above any number. Four single-variable
# passes over the identical corpus/split/eval plus the (28b) objective arm; `v2` is a prior
# generation with its own harness (`mining_v2_reads.py`) and is deliberately not here.
# The arm's NAME comes out of its own config.json, never the directory, so a renamed dir
# cannot make a report describe the wrong experiment.
ARMS = ("v3", "v3_aug", "v3_augx", "v3_uniform", "v3_ap2")

# The anchored corpus this slice exists to replace. The NUMBER is read from the report, not
# from here: this names WHERE, and `anchoring_price()` reads WHAT. Any arm's report carries
# the same v1 block (same checkpoint, same eval slice); v3's is the first and is named.
ANCHORED_REPORT = HEAD_DIR / "v3" / "report.json"
ANCHORED_ARM = "pooled"

# Imported, not re-declared — a cell here must be the same cell as in the (28) harness or the
# two reports cannot be read together. `voting_metrics` is what decides which cells a per-mode
# arm submits (AUCs only); it is the rule the contested failures were found under.
METRICS = V3R.METRICS
voting_metrics = V3R.voting_metrics

# A per-mode arm below this many rows is reported but flagged: a paired CI over four metrics
# on a dozen rows says nothing, and the (28) harness applied the same floor.
MIN_ARM_N = 20


def log(m):
    print(m, flush=True)


def _rel(p: Path) -> str:
    """Repo-relative when it can be, absolute when it cannot. `relative_to` RAISES on a path
    outside the tree, and a report that dies formatting its own provenance line is a worse
    failure than a long path."""
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@dataclass(frozen=True)
class ERow:
    image_id: str
    label: int
    jpg: Path
    mode: str
    kind: str
    partition: str
    family: str
    loc: str
    hue_family: str


def load_rows(require_crops: bool = True) -> tuple[list, dict]:
    """`(rows, meta)` — sheet E's labeled rows. A missing sidecar is a HARD STOP with the
    merge command, not a short slice: half a labeled sheet silently answering a verdict
    question is the failure this whole file exists to avoid."""
    bdir = SHEET.batch_dir
    imgs = bdir / "images.jsonl"
    if not imgs.exists():
        raise SystemExit(f"[sheet-e] {imgs} absent — build the sheet first "
                         f"(tools/mining/build_blind_mining_sheet.py)")
    sidecar = ROOT / SHEET.labels_sidecar
    if not sidecar.exists():
        raise SystemExit(
            f"[sheet-e] no labels at {sidecar}.\nLabel the sheet, save the page's export as "
            f"{SHEET.labels_export}, then merge:\n"
            f"  uv run python tools/wallpaper/merge_sitting.py --corpus render_mode_corpus "
            f"--batch {SHEET.batch_id} --scores {SHEET.labels_export} --apply")
    labels = json.loads(sidecar.read_text(encoding="utf-8"))

    rows, n_total, n_unlabeled = [], 0, 0
    for line in imgs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_total += 1
        iid = r["image_id"]
        s = labels.get(iid)
        if s is None:
            n_unlabeled += 1
            continue
        jpg = bdir / "crops" / f"{iid}.jpg"
        if require_crops and not jpg.exists():
            raise FileNotFoundError(f"crop missing: {jpg}")
        pv = r["provenance"]
        rows.append(ERow(image_id=iid, label=int(s), jpg=jpg,
                         mode=r["render"]["render_mode"],
                         kind=pv.get("mode_kind") or "?",
                         partition=pv.get("partition") or "?",
                         family=r["render"]["fractal_type"],
                         loc=pv.get("location_key") or "?",
                         hue_family=pv.get("hue_family") or "?"))
    if not rows:
        raise SystemExit(f"[sheet-e] sidecar {sidecar} labels none of the batch's rows")
    return rows, {"batch_id": SHEET.batch_id, "sidecar": SHEET.labels_sidecar,
                  "n_batch_rows": n_total, "n_labeled": len(rows),
                  "n_unlabeled": n_unlabeled, "partial": bool(n_unlabeled),
                  "partial_note": (f"{n_unlabeled} of {n_total} rows carry no label — this "
                                   f"verdict is on the labeled subset and the subset is NOT "
                                   f"a random one (labeling runs in sheet order)")
                                  if n_unlabeled else None}


def tier_dist(labels) -> dict:
    c = Counter(int(x) for x in labels)
    n = len(labels)
    return {"n": n, "hist": {str(t): c.get(t, 0) for t in (1, 2, 3)},
            "frac_ge2": (n - c.get(1, 0)) / n if n else None,
            "frac_ge3": c.get(3, 0) / n if n else None}


def cell_status(pb: dict, ci: dict, m) -> dict:
    """Why a cell votes, or why it cannot. STATED PER CELL rather than dropped.

    A boundary with only one class present has no AUC and no AP, and a bootstrap over it
    yields no draws. Treating that as "not worse" is how a cell nobody could measure passes a
    head, so it is named here and excluded from every count."""
    npos, nneg = pb[f"{m.key}__n_pos"], pb[f"{m.key}__n_neg"]
    if npos == 0 or nneg == 0:
        which = "no positives" if npos == 0 else "no negatives"
        return {"measurable": False,
                "why": f"{which} at label >= {m.thr} (n_pos={npos}, n_neg={nneg}) — one class "
                       f"only, so this cell votes neither way",
                "n_pos": npos, "n_neg": nneg}
    if not ci or ci.get("n_draws", 0) == 0:
        return {"measurable": False,
                "why": f"every bootstrap draw degenerated at label >= {m.thr} "
                       f"(n_pos={npos}, n_neg={nneg}) — votes neither way",
                "n_pos": npos, "n_neg": nneg}
    return {"measurable": True, "n_pos": npos, "n_neg": nneg, "n_draws": ci["n_draws"]}


def arm_block(labels, base, cand, mask, *, draws, seed) -> dict:
    lb = labels[mask]
    b = {k: v[mask] for k, v in base.items()}
    c = {k: v[mask] for k, v in cand.items()}
    out = {"n": int(mask.sum()), "tiers": tier_dist(lb),
           "v1": point_block(lb, b, METRICS), "arm": point_block(lb, c, METRICS),
           "delta_ci": paired_bootstrap(lb, b, c, METRICS, draws=draws, seed=seed)}
    out["cells"] = {m.key: cell_status(out["v1"], out["delta_ci"][m.key], m) for m in METRICS}
    out["underpowered"] = out["n"] < MIN_ARM_N
    return out


def slice_masks(rows) -> dict:
    """`{arm_name: mask}` — pooled plus one arm per mode present. Declared as predicates,
    before any number exists, and the mode arms are named exactly as the (28) harness names
    them so a cell can be joined across the two reports."""
    mode = np.array([r.mode for r in rows])
    out = {"pooled": np.ones(len(rows), bool)}
    for m in sorted(set(mode.tolist())):
        out[f"mode:{m}"] = mode == m
    return out


def clause_a(arms: dict) -> dict:
    """Clause (a) alone, on this slice: is any pre-declared arm significantly worse?

    `winner_rule.verdict` is deliberately NOT called — it requires a motivating arm and would
    return a `winner` computed from an empty clause (b). The half that applies is applied,
    and the half that does not is named."""
    worse, better, unmeasurable, n = [], [], [], 0
    for name, blk in sorted(arms.items()):
        for m in voting_metrics(name):
            ci = blk["delta_ci"][m.key]
            st = blk["cells"][m.key]
            if not st["measurable"]:
                unmeasurable.append({"arm": name, "metric": m.key, "why": st["why"]})
                continue
            n += 1
            rec = {"arm": name, "metric": m.key, "median": ci["median"],
                   "lo": ci["lo"], "hi": ci["hi"], "n": blk["n"]}
            if ci["significantly_worse"]:
                worse.append(rec)
            if ci["significantly_better"]:
                better.append(rec)
    return {"pass": not worse, "n_tests": n, "failures": worse, "improvements": better,
            "unmeasurable": unmeasurable,
            "voting_cells": {k: [m.key for m in voting_metrics(k)] for k in sorted(arms)},
            "rule": "no pre-declared arm is significantly worse than v1 (95% paired-bootstrap "
                    "CI on the delta entirely below 0). Per-mode arms submit their AUC cells "
                    "only — the (28) harness's own voting rule, imported.",
            "clause_b": "NOT DECIDED HERE. The (28) motivating arm (`busy_fp`) is defined by "
                        "sheets B and C's own draw buckets and has no analogue on this slice; "
                        "choosing one after seeing these rows would be choosing it from the "
                        "data.",
            "multiplicity_note": f"clause (a) is a conjunction over {n} arm x metric cells; at "
                                 f"95% per cell the chance one crosses by luck alone is "
                                 f"material, so read `failures` before reading `pass`."}


def anchored_failures(arm_dir: Path) -> dict:
    """The clause-(a) cells this arm FAILED on the anchored corpus, read out of its own
    committed report. UNKNOWN rather than empty when the report is absent: an empty list reads
    as "this arm failed nothing", which is the opposite of what a missing file means."""
    p = arm_dir / "report.json"
    if not p.exists():
        return {"status": f"UNKNOWN — {_rel(p)} is not on disk",
                "cells": None}
    R = json.loads(p.read_text(encoding="utf-8"))
    wr = R.get("winner_rule") or {}
    fails = (wr.get("clause_a") or {}).get("failures") or []
    return {"status": "read", "report": _rel(p),
            "n_tests": (wr.get("clause_a") or {}).get("n_tests"),
            "eval_n": (R.get("eval_slice") or {}).get("n"),
            "cells": [{"arm": f["arm"], "metric": f["metric"], "median": f["median"],
                       "lo": f["lo"], "hi": f["hi"]} for f in fails]}


def survival(anchored: dict, blind_arms: dict) -> dict:
    """Cell by cell: did the anchored clause-(a) failure survive on unanchored labels?

    THE CONTESTED FOUR are flagged, but every failing cell is walked — a regression that
    vanishes outside `CONTESTED_MODES` is as much a finding as one inside it."""
    if anchored.get("cells") is None:
        return {"status": anchored["status"], "cells": []}
    out = []
    for c in anchored["cells"]:
        blk = blind_arms.get(c["arm"])
        rec = {"cell": f"{c['arm']}.{c['metric']}",
               "arm_slice": c["arm"], "metric": c["metric"],
               "contested_mode": c["arm"].replace("mode:", "") in CONTESTED_MODES,
               "anchored": {"median": c["median"], "lo": c["lo"], "hi": c["hi"]}}
        if blk is None:
            rec["blind"] = None
            rec["verdict"] = "ABSENT — no rows of this arm on the blind slice"
            out.append(rec)
            continue
        st = blk["cells"][c["metric"]]
        ci = blk["delta_ci"][c["metric"]]
        rec["blind"] = {"n": blk["n"], "median": ci["median"], "lo": ci["lo"], "hi": ci["hi"],
                        "v1": blk["v1"][c["metric"]], "arm": blk["arm"][c["metric"]]}
        if not st["measurable"]:
            rec["verdict"] = "UNMEASURABLE"
            rec["why"] = st["why"]
        elif ci["significantly_worse"]:
            rec["verdict"] = "SURVIVES"
        elif blk["underpowered"]:
            rec["verdict"] = "NOT SIGNIFICANT (underpowered)"
            rec["why"] = (f"{blk['n']} rows, under the {MIN_ARM_N}-row floor — 'not "
                          f"significantly worse' here is weaker than the same words on the "
                          f"pooled arm")
        else:
            rec["verdict"] = "DOES NOT SURVIVE"
        out.append(rec)
    return {"status": "read", "report": anchored.get("report"), "cells": out}


def anchoring_price(blind_v1: dict) -> dict:
    """THE anchored-vs-blind comparison for v1 itself, with the anchored side DERIVED off the
    committed (28) report rather than restated. A missing report is reported as UNKNOWN, not
    as absent: the price is a headline of this file and a silently-dropped one reads as no
    price."""
    out = {"blind_slice": {"arm": ANCHORED_ARM,
                           "v1_auc_ge2": blind_v1.get("auc_ge2"),
                           "v1_auc_ge3": blind_v1.get("auc_ge3"),
                           "v1_ap_ge2": blind_v1.get("ap_ge2"),
                           "v1_ap_ge3": blind_v1.get("ap_ge3")},
           "anchored_slice": {"report": _rel(ANCHORED_REPORT),
                              "arm": ANCHORED_ARM},
           "what": "v1's pooled AUC>=2 on rows whose labels it SUGGESTED, against v1's pooled "
                   "AUC>=2 on fresh (location, mode) pairs of the same population labeled "
                   "blind. The gap is the anchoring price — how much of the 0.953 was "
                   "agreement rather than quality (classifier_retrain_protocol.md 2b). "
                   "AUC>=2 is the boundary named because it is the one every staged arm "
                   "loses on."}
    if not ANCHORED_REPORT.exists():
        out["anchored_slice"]["status"] = "UNKNOWN — the (28) v3 report is not on disk"
        return out
    R = json.loads(ANCHORED_REPORT.read_text(encoding="utf-8"))
    blk = (R.get("no_worse") or {}).get(ANCHORED_ARM)
    if blk is None:
        out["anchored_slice"]["status"] = f"UNKNOWN — no arm {ANCHORED_ARM!r} in that report"
        return out
    out["anchored_slice"].update({
        "status": "read", "n": blk["n"], "tiers": blk["tiers"]["hist"],
        "v1_auc_ge2": blk["v1"]["auc_ge2"], "v1_auc_ge3": blk["v1"]["auc_ge3"],
        "v1_ap_ge2": blk["v1"]["ap_ge2"], "v1_ap_ge3": blk["v1"]["ap_ge3"]})
    for k in ("auc_ge2", "auc_ge3"):
        a, b = out["anchored_slice"][f"v1_{k}"], out["blind_slice"][f"v1_{k}"]
        if a is not None and b is not None:
            out[f"delta_{k}"] = float(b - a)
    d = out.get("delta_auc_ge2")
    if d is not None:
        out["reading"] = ("the blind slice is the LOWER number; the difference is what the "
                          "prefilled suggestion bought v1 on a slice it was measured on"
                          if d < 0 else
                          "the blind slice is NOT lower — the anchoring did not inflate v1 "
                          "at this boundary, which is itself the finding")
    return out


def build(rows, base, arm_scores, meta, *, draws, seed) -> dict:
    labels = np.array([r.label for r in rows])
    masks = slice_masks(rows)

    R = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": "uv run python tools/mining/sheet_e_reverdict.py",
        "baseline": {"name": "v1", "ckpt": MP.ACTIVE_MINING_CKPT,
                     "role": "incumbent (LIVE pin)"},
        "live_pin": MP.ACTIVE_MINING_CKPT,
        "moves_nothing": "no pin, gate, floor, lock or annotation is written by this file",
        "slice": {**meta, "n": len(rows),
                  "tiers": tier_dist(labels),
                  "n_locations": len({r.loc for r in rows}),
                  "by_mode": dict(sorted(Counter(r.mode for r in rows).items())),
                  "by_kind": dict(sorted(Counter(r.kind for r in rows).items())),
                  "by_partition": dict(sorted(Counter(r.partition for r in rows).items())),
                  "by_hue_family": dict(sorted(Counter(r.hue_family for r in rows).items())),
                  "contested_modes": list(CONTESTED_MODES),
                  "contested_rows": int(sum(1 for r in rows if r.mode in CONTESTED_MODES)),
                  "blind": "no suggestion was served, no score ordered the page",
                  "eval_only": True},
        "bootstrap": {"draws": draws, "seed": seed, "kind": "paired over slice rows"},
        "metric_set": [m.key for m in METRICS],
        "min_arm_n": MIN_ARM_N,
        "pre_declared": {
            "no_worse": ["pooled"] + sorted(k for k in masks if k.startswith("mode:")),
            "motivating": [],
            "note": "clause (a) only — see `clause_a.clause_b` for why this slice does not "
                    "decide the motivating clause.",
        },
        "arms": {},
    }

    for name, sc in arm_scores.items():
        blocks = {k: arm_block(labels, base, sc["scores"], m, draws=draws, seed=seed)
                  for k, m in masks.items() if m.sum() > 0}
        anch = anchored_failures(sc["dir"])
        R["arms"][name] = {
            "arm_name": sc["arm_name"],
            "ckpt": sc["ckpt"],
            "dials": sc["dials"],
            "role": "from-scratch candidate, STAGED and unadopted",
            "slices": blocks,
            "pooled": {m.key: {"v1": blocks["pooled"]["v1"][m.key],
                               "arm": blocks["pooled"]["arm"][m.key],
                               "delta_ci": blocks["pooled"]["delta_ci"][m.key],
                               "cell": blocks["pooled"]["cells"][m.key]}
                       for m in METRICS},
            "clause_a": clause_a(blocks),
            "anchored_failures": anch,
            "contested_survival": survival(anch, blocks),
        }

    # v1's own point values are identical across arms (same head, same rows) — computed once
    # from the first arm's pooled block and named as the baseline read.
    first = next(iter(R["arms"].values()))
    R["v1_pooled"] = first["slices"]["pooled"]["v1"]
    R["v1_by_mode"] = {k: v["v1"] for k, v in first["slices"].items() if k.startswith("mode:")}
    R["anchoring_price"] = anchoring_price(R["v1_pooled"])

    # The cross-arm summary the prompt asks for: does a contested cell survive in ANY arm?
    cells = {}
    for name, a in R["arms"].items():
        for c in a["contested_survival"]["cells"]:
            cells.setdefault(c["cell"], {"contested_mode": c["contested_mode"],
                                         "by_arm": {}})["by_arm"][name] = c["verdict"]
    R["contested_summary"] = {
        "what": "every clause-(a) cell any staged arm failed on the ANCHORED corpus, and what "
                "it does on the blind slice. A cell absent from an arm's row failed nothing "
                "for that arm.",
        "cells": {k: {**v,
                      "n_arms_surviving": sum(1 for x in v["by_arm"].values()
                                              if x == "SURVIVES"),
                      "n_arms_failed_anchored": len(v["by_arm"])}
                  for k, v in sorted(cells.items())},
    }
    return R


# --------------------------------------------------------------------------- #
def md(R) -> str:
    L = []
    A = L.append
    s = R["slice"]
    A("# SHEET E re-verdict — mining v1 vs the five staged arms on BLIND labels\n")
    A(f"Generated {R['generated']} · `{R['command']}`\n")
    A(f"> {R['moves_nothing']}. Adoption is a separate prompt.\n")
    # A bounded or synthetic run must SAY SO in the markdown, not only in the JSON: the .md is
    # what gets read, and a report that does not stamp itself unusable will be read as one.
    if R.get("DRY_RUN"):
        A(f"> **DRY RUN — {R['DRY_RUN']}**\n")
    elif R.get("incomplete"):
        A("> **INCOMPLETE — this report was produced by a bounded `--limit` run on a prefix "
          "of the sheet, which is NOT a random subset (labeling runs in sheet order). Not a "
          "verdict.**\n")
    A(f"Slice: **{s['n']} labeled rows** of {s['n_batch_rows']} in `{s['batch_id']}` over "
      f"{s['n_locations']} locations; tiers {s['tiers']['hist']}; "
      f"{s['contested_rows']} rows on the four contested modes. Blind, eval-only.\n")
    if s.get("partial_note"):
        A(f"**PARTIAL:** {s['partial_note']}\n")

    A("\n## (a) Do the contested-cell regressions survive?\n")
    cs = R["contested_summary"]
    A(f"{cs['what']}\n")
    A("| cell | contested mode | arms that failed it anchored | arms where it SURVIVES blind |"
      " per-arm verdict |")
    A("|---|:--:|---:|---:|---|")
    for k, v in cs["cells"].items():
        per = " · ".join(f"{a}: {x}" for a, x in sorted(v["by_arm"].items()))
        A(f"| `{k}` | {'yes' if v['contested_mode'] else 'no'} "
          f"| {v['n_arms_failed_anchored']} | **{v['n_arms_surviving']}** | {per} |")

    A("\n## (b) Pooled reads, per arm\n")
    A("| arm | AUC≥2 v1 | AUC≥2 arm | Δ 95% CI | AUC≥3 v1 | AUC≥3 arm | Δ 95% CI | "
      "clause (a) on this slice |")
    A("|---|---:|---:|---|---:|---:|---|---|")

    def cell(x):
        return "—" if x is None else f"{x:.3f}"

    def ic(c):
        if not c or c.get("n_draws", 0) == 0:
            return "n/a"
        tag = "**worse**" if c["significantly_worse"] else (
            "**better**" if c["significantly_better"] else "")
        return f"[{c['lo']:+.3f}, {c['hi']:+.3f}] {tag}"

    for name, a in R["arms"].items():
        p2, p3 = a["pooled"]["auc_ge2"], a["pooled"]["auc_ge3"]
        ca = a["clause_a"]
        A(f"| `{name}` ({a['arm_name']}) | {cell(p2['v1'])} | {cell(p2['arm'])} "
          f"| {ic(p2['delta_ci'])} | {cell(p3['v1'])} | {cell(p3['arm'])} "
          f"| {ic(p3['delta_ci'])} | {'PASS' if ca['pass'] else 'FAIL'} "
          f"({len(ca['failures'])}/{ca['n_tests']} cells) |")
    A("\n| arm | AP≥2 v1 | AP≥2 arm | Δ 95% CI | AP≥3 v1 | AP≥3 arm | Δ 95% CI |")
    A("|---|---:|---:|---|---:|---:|---|")
    for name, a in R["arms"].items():
        p2, p3 = a["pooled"]["ap_ge2"], a["pooled"]["ap_ge3"]
        A(f"| `{name}` | {cell(p2['v1'])} | {cell(p2['arm'])} | {ic(p2['delta_ci'])} "
          f"| {cell(p3['v1'])} | {cell(p3['arm'])} | {ic(p3['delta_ci'])} |")

    ap = R["anchoring_price"]
    A("\n## (c) The anchoring price\n")
    A(f"{ap['what']}\n")
    an, bl = ap["anchored_slice"], ap["blind_slice"]
    A("| slice | labels elicited | n | v1 AUC≥2 | v1 AUC≥3 |")
    A("|---|---|---:|---:|---:|")
    if an.get("status") == "read":
        A(f"| anchored corpus `{an['arm']}` | v1's tier PREFILLED, page sorted by v1 "
          f"| {an['n']} | {cell(an['v1_auc_ge2'])} | {cell(an['v1_auc_ge3'])} |")
    else:
        A(f"| anchored corpus `{an['arm']}` | {an.get('status', 'UNKNOWN')} | — | — | — |")
    A(f"| sheet E `{bl['arm']}` | BLIND, shuffled | {R['slice']['n']} "
      f"| {cell(bl['v1_auc_ge2'])} | {cell(bl['v1_auc_ge3'])} |")
    if "delta_auc_ge2" in ap:
        A(f"\n**Δ AUC≥2 (blind − anchored) = {ap['delta_auc_ge2']:+.3f}**"
          + (f" · Δ AUC≥3 = {ap['delta_auc_ge3']:+.3f}" if "delta_auc_ge3" in ap else "")
          + f" — {ap['reading']}\n")

    A("\n## Per-mode reads (v1 vs each arm, AUC cells only — the (28) voting rule)\n")
    modes = sorted(k for k in next(iter(R["arms"].values()))["slices"] if k.startswith("mode:"))
    A("| mode | n | ≥3 | v1 AUC≥2 | v1 AUC≥3 | " +
      " | ".join(f"{a} AUC≥2" for a in R["arms"]) + " |")
    A("|---|---:|---:|---:|---:|" + "---:|" * len(R["arms"]))
    first = next(iter(R["arms"].values()))
    for mk in modes:
        b0 = first["slices"][mk]
        flag = " ⚠" if b0["underpowered"] else ""
        vals = " | ".join(cell(R["arms"][a]["slices"][mk]["arm"]["auc_ge2"]) for a in R["arms"])
        A(f"| `{mk.replace('mode:', '')}`{flag} | {b0['n']} | {b0['tiers']['hist']['3']} "
          f"| {cell(b0['v1']['auc_ge2'])} | {cell(b0['v1']['auc_ge3'])} | {vals} |")
    A(f"\n⚠ = under the {R['min_arm_n']}-row floor; its CI is reported and votes, but read it "
      f"as underpowered.\n")

    unmeas = sorted({f"{u['arm']}.{u['metric']}: {u['why']}"
                     for a in R["arms"].values() for u in a["clause_a"]["unmeasurable"]})
    A("\n## Cells that vote neither way\n")
    if not unmeas:
        A("None — every arm x metric cell had both classes present.\n")
    else:
        for u in unmeas:
            A(f"- `{u}`")
        A("")

    for name, a in R["arms"].items():
        if a["clause_a"]["failures"]:
            A(f"\n### `{name}` clause (a) failures on this slice\n")
            for f in a["clause_a"]["failures"]:
                A(f"- `{f['arm']}` **{f['metric']}** (n={f['n']}): Δ median {f['median']:+.3f}, "
                  f"95% CI [{f['lo']:+.3f}, {f['hi']:+.3f}]")
    A(f"\n{next(iter(R['arms'].values()))['clause_a']['multiplicity_note']}\n")
    A(f"\n{next(iter(R['arms'].values()))['clause_a']['clause_b']}\n")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="rows — bounded end-to-end; writes to scratch/, stamped incomplete")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260811)
    a = ap.parse_args(argv)

    if not (ROOT / MP.ACTIVE_MINING_CKPT).exists():
        raise SystemExit(f"[sheet-e] missing incumbent checkpoint {MP.ACTIVE_MINING_CKPT}")
    missing = [x for x in ARMS if not (HEAD_DIR / x / "model_best.pt").exists()]
    if missing:
        raise SystemExit(f"[sheet-e] missing staged arm checkpoint(s): "
                         f"{[f'data/render_mode_head/{m}/model_best.pt' for m in missing]}")

    rows, meta = load_rows()
    if a.limit:
        rows = rows[:a.limit]
        meta = dict(meta, n_labeled=len(rows))
    log(f"[sheet-e] {len(rows)} labeled rows · {len(ARMS)} staged arms")

    base = V3R.score_with(MP.ACTIVE_MINING_CKPT, rows)
    arm_scores = {}
    for name in ARMS:
        d = HEAD_DIR / name
        cfg = {}
        cp = d / "config.json"
        if cp.exists():
            cfg = json.loads(cp.read_text(encoding="utf-8"))
        arm_scores[name] = {
            "dir": d, "ckpt": f"data/render_mode_head/{name}/model_best.pt",
            "arm_name": cfg.get("arm") or name,
            "dials": {k: cfg.get(k) for k in ("border_crop", "axis_crop", "uniform_weights",
                                              "row_weighting", "selection_metric")},
            "scores": V3R.score_with(f"data/render_mode_head/{name}/model_best.pt", rows)}
        log(f"[sheet-e] scored {name} ({arm_scores[name]['arm_name']})")

    R = build(rows, base, arm_scores, meta, draws=a.draws, seed=a.seed)
    R["incomplete"] = bool(a.limit)
    out = (ROOT / "scratch" / "sheet_e_reverdict") if a.limit else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(md(R), encoding="utf-8")
    for name, arm in R["arms"].items():
        ca = arm["clause_a"]
        surv = sum(1 for c in arm["contested_survival"]["cells"]
                   if c["verdict"] == "SURVIVES")
        log(f"[sheet-e] {name:11} clause a {'PASS' if ca['pass'] else 'FAIL'} "
            f"({len(ca['failures'])}/{ca['n_tests']}) · anchored failures surviving here: "
            f"{surv}/{len(arm['contested_survival']['cells'])}")
    if "delta_auc_ge2" in R["anchoring_price"]:
        log(f"[sheet-e] anchoring price: Δ AUC>=2 (blind - anchored) "
            f"{R['anchoring_price']['delta_auc_ge2']:+.3f}")
    log(f"-> {out / 'report.md'}")


if __name__ == "__main__":
    main()
