r"""fresh_sheet_reads.py — the four reads off the MERGED render-mode correction sheet.

`prompts/mining_merge_prompt.md` §2, over
`data/render_mode_corpus/batches/2026-08-06_render_mode_fresh_sheet_v1` after
`merge_scores.py --apply` filled all 960 `label.score` slots:

  A  AGREEMENT — where Matt's final label differs from the head's suggested tier, with
     direction, sliced by mode / stratum (family, mode kind, split side) / suggested tier.
     Overall and eval-side.
  B  V1 AGAINST THE LABELS — AUC and AP at each tier boundary (>=2 on marginal `p_ge2`,
     >=3 on marginal `p_ge3`), full corpus and eval side separately.
  C  PER-MODE TRUST — for each of the 15 roster modes: n, label distribution, v1's AUC
     and agreement, and precision at the two live cuts. Modes at or near chance are FLAGGED.
  D  CALIBRATION LADDERS — precision-of-passers and recall across a `p_ge3` sweep on the
     eval side, with the mining pool cut (0.25, acting) and the report-only release floor
     (0.50) marked, plus the thresholds that would buy 0.70 / 0.80 / 0.90 precision.
  E  WHAT THIS SITTING CAN AND CANNOT ADJUDICATE — the bulk-sweep detector, the
     adjudicated-vs-bulk sensitivity of every headline number, and the same-checkpoint
     comparison against the July lock. Read this before read D is used to move anything.

Nothing here moves a cut, floor, gate or pin. The two operating points are READ from
`tools/emission/floors.py` (`MINING_POOL`, `MINING_RELEASE`) through `Floor.gate()`, so the
head-stamp check runs and a report cannot quietly be computed on a scale the live pin no
longer serves. Candidate cuts in read D are DERIVED and printed; adopting one is a separate,
human decision.

FOUR THINGS THE NUMBERS DO NOT SAY ON THEIR OWN, and every one of them changes how read D
should be used:

0. THE LABELS WERE SEEDED FROM THE HEAD'S OWN SUGGESTION. This was a CORRECTION sheet: every
   row was served with its `suggested_tier` PREFILLED, Enter confirmed, 1-3 overrode
   (`batch.json` -> `labeling.mode`), and the sheet was sorted good->bad so an
   accept-all-below sweep is one keystroke. So `label` and `suggested` are coupled by
   construction, and NO agreement, precision or AUC in this file is a measurement of the head
   against an independent human judgement. Read E quantifies how much: it finds the sweep
   rather than assuming one, and recomputes every headline number without it.

1. THIS IS NOT A HELD-OUT POPULATION FOR V1. The fresh sheet is drawn from
   `data/render_mode_corpus/gate_passers_v3.json` — 401 rows over 112 locations — and that is
   the SAME artifact both July samplers keyed off (`build_gate_passers.py` reproduces their
   printed counts exactly; that agreement is the verification). v1 trained on renders at
   these locations. The mode x palette x param draw here is new, the locations are not, so
   every precision below is an optimistic bound on a fresh location.

2. `split_side` IS FOR THE NEXT HEAD, NOT THIS ONE. The union-find split (95 units, seed 0)
   was stamped so a v2 trainer has a clean eval side. Both sides are equally exposed to v1 by
   (1). The eval-side ladders are reported because the prompt asks the decision to be made on
   that slice and because it is the slice a retrain would score against — not because it is
   held out from v1.

3. AGREEMENT MEASURES THE CUTS AS MUCH AS THE HEAD. `suggested_tier` is the head's
   `expected_tier` put through PER-BATCH quantile cuts matched to the lost corpus's tier prior
   (`suggest_tier_mining.cuts_from_prior`) — it carries rank information only, no absolute
   quality level. So read A's disagreement rate is a joint statement about the head's ordering
   AND a cut nobody claimed was calibrated. Read B is the cut-free read; when the two
   disagree, B is the one about the head.

Metrics are imported, not restated: `ap` / `auc` / `corr_block` / `boundary_block` and the
markdown+contact-sheet helpers from `tools/wallpaper/sitting_reads.py` (the wallpaper sibling
of this readout, same four-read shape), the AUC standard error from `tools/v10/prereg.py`, and
the Wilson interval from `tools/corpus/q4_combined_readout.py`. The mining sheet already
imports the wallpaper suggestion rule for the same reason (`suggest_tier_mining`): two heads
must agree on what a number means even when they disagree on how many tiers there are.

Outputs (scratch/mining_fresh_sheet/): report.md, report.json, and two contact sheets — a
correction rate cannot tell you whether the head was wrong or Matt was strict, and the crops
can.

  uv run python tools/mining/fresh_sheet_reads.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.corpus.q4_combined_readout import wilson                     # noqa: E402
from tools.emission import floors as F                                  # noqa: E402
from tools.mining.build_mining_sheet import SHEETS                      # noqa: E402
from tools.mining import mining_pins as MP                              # noqa: E402
from tools.mining.mining_roster import MODE_KIND, MODES, TRAINER_DROPPED_V1  # noqa: E402
from tools.v10.prereg import hanley_mcneil_se, min_detectable_auc       # noqa: E402
from tools.wallpaper.sitting_reads import (                             # noqa: E402
    ap, auc, boundary_block, corr_block, md_table, num, pct, sheet)

SPEC = SHEETS["v1"]
OUT = ROOT / "scratch" / "mining_fresh_sheet"

K_TIERS = 3                       # the mining head's scale: 1 bad / 2 okay / 3 good
GOOD = 3                          # "good" == the top tier == the >=3 boundary the gate cuts at

# The sweep grid. Coarse enough to read, fine enough that a candidate cut is not an artifact
# of the spacing; the two live cuts are UNIONED in so they are exact rows, never nearest-bin.
SWEEP = sorted({round(x, 3) for x in np.arange(0.0, 1.0, 0.05)}
               | {F.MINING_POOL.value, F.MINING_RELEASE.value})
PRECISION_TARGETS = (0.70, 0.80, 0.90)

# A contiguous confirm-everything run has to be long enough that no plausible sequence of
# independent judgements produces it. At the sheet's own 92.9% row-level agreement, 60 in a
# row is p ~ 1.2e-2 and 100 in a row p ~ 6e-4; 60 is the floor, and the detector reports the
# run it found rather than the threshold, so the number below only decides what gets NAMED.
MIN_SWEEP_RUN = 60

# The July operating point, quoted from the record. `mining_gate_lock.json` — the frozen PR
# curve this came from — did NOT survive the corpus loss (`data/render_mode_head/v1/` holds
# only `model_best.pt`), so the surviving citation is the pin module's own §2, which states
# it as the deployed seed-0 read on the LOST corpus's held-out eval set. Quoted, never
# recomputed: the crops it was measured on are gone, which is the whole reason this sheet
# exists.
JULY_LOCK = {"threshold": 0.50, "precision": 0.548, "recall": 0.195, "pass_rate": 0.050,
             "base_rate": 0.139,
             "source": "tools/mining/mining_gate.py §2 (the mining_gate_lock.json it cites "
                       "did not survive the corpus loss)",
             "population": "held-out eval side of the LOST July corpus — genuinely held out "
                           "for v1, and independently labeled (no prefill)"}


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# join — one batch, labels now IN-ROW (that is the whole point of the rebuild)
# --------------------------------------------------------------------------- #
def load(batch_dir: Path | None = None) -> list:
    """The merged rows, flattened to what the reads need.

    Raises on an unlabeled row rather than dropping it: this readout exists to describe a
    COMPLETE sitting, and a silently smaller n reads exactly like a complete one."""
    bd = batch_dir or SPEC.batch_dir
    rows, unlabeled = [], []
    for line in (bd / "images.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label"]["score"] is None:
            unlabeled.append(r["image_id"])
            continue
        pv, h = r["provenance"], r["head_mining_v1"]
        rows.append({
            "id": r["image_id"], "crop": bd / "crops" / f"{r['image_id']}.jpg",
            "label": int(r["label"]["score"]), "suggested": int(r["suggested_tier"]),
            "p_ge2": float(h["p_ge2"]), "p_ge3": float(h["p_ge3"]),
            "pred": float(h["pred"]), "rank_score": float(h["score"]),
            "mode": pv["render_mode"], "kind": pv["mode_kind"], "family": pv["family"],
            "split": pv["split_side"], "palette": r["render"]["palette"],
            "loc": pv["location_key"], "order": int(r["sheet_order"]),
        })
    if unlabeled:
        raise SystemExit(
            f"[fresh-sheet-reads] {len(unlabeled)} rows still have label.score = null "
            f"(e.g. {unlabeled[:3]}). Merge the export first:\n"
            f"    uv run python tools/corpus/merge_scores.py --apply --max-score {K_TIERS} "
            f"--batch {SPEC.batch_id} --corpus-root data/render_mode_corpus/batches "
            f"--scores {SPEC.labels_export}")
    return rows


def by_key(rows, key):
    return {k: [r for r in rows if r[key] == k] for k in sorted({r[key] for r in rows})}


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def tier_dist(rows) -> dict:
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    return {"n": n, "hist": {str(t): c.get(t, 0) for t in range(1, K_TIERS + 1)},
            "frac_ge2": (n - c.get(1, 0)) / n if n else None,
            "frac_ge3": c.get(3, 0) / n if n else None,
            "mean": float(np.mean([r["label"] for r in rows])) if n else None}


def head_block(rows, thr: int, score_key: str) -> dict:
    """v1's separation at ONE tier boundary: AP, AUC, and the AUC's distance from chance.

    `chance` is the honest verdict at this n, not a vibe: `min_detectable_auc` is the smallest
    AUC whose 95% Hanley-McNeil interval clears 0.50 for this cell's (n_pos, n_neg), so a mode
    with 64 rows and a mode with 960 are judged against their own power."""
    y = [r["label"] >= thr for r in rows]
    s = [r[score_key] for r in rows]
    n_pos, n_neg = int(sum(y)), int(len(y) - sum(y))
    a = auc(y, s)
    out = {"n": len(rows), "n_pos": n_pos, "n_neg": n_neg, "base_rate": n_pos / len(y) if y else None,
           "ap": ap(y, s), "auc": a, "score": score_key}
    if a is None or n_pos == 0 or n_neg == 0:
        out.update(auc_se=None, auc_lo=None, min_detectable=None, at_chance=None)
        return out
    se = hanley_mcneil_se(a, n_pos, n_neg)
    out.update(auc_se=se, auc_lo=a - 1.96 * se, auc_hi=a + 1.96 * se,
               min_detectable=min_detectable_auc(n_pos, n_neg),
               at_chance=bool(a - 1.96 * se <= 0.50))
    return out


def cut_block(rows, floor: F.Floor, thr: int = GOOD) -> dict:
    """What ONE live cut buys on a slice. `floor.gate()` — not `>= floor.value` — so the
    head-stamp check runs at every site that reports a number as "at the production cut"."""
    fire = [r for r in rows if floor.gate(r["p_ge3"])]
    good = [r for r in rows if r["label"] >= thr]
    k = sum(1 for r in fire if r["label"] >= thr)
    p, lo, hi = wilson(k, len(fire)) if fire else (None, None, None)
    return {"floor": floor.name, "value": floor.value, "acts": floor.acts,
            "n": len(rows), "fires": len(fire),
            "pass_rate": len(fire) / len(rows) if rows else None,
            "n_good": len(good), "tp": k,
            "precision": p, "precision_lo": lo, "precision_hi": hi,
            "recall": k / len(good) if good else None,
            "lift": (p / (len(good) / len(rows))) if (p is not None and good and rows) else None}


def ladder(rows, *, score_key="p_ge3", thr=GOOD, grid=SWEEP) -> list:
    """Precision-of-passers and recall across a threshold sweep. One row per grid point.

    Precision carries a Wilson interval because the top of any ladder is estimated from few
    passers — a bare 1.000 over 3 rows and a 0.90 over 90 are the same column otherwise, and
    only one of them is a cut you could set."""
    good = [r for r in rows if r["label"] >= thr]
    out = []
    for t in grid:
        fire = [r for r in rows if r[score_key] >= t]
        k = sum(1 for r in fire if r["label"] >= thr)
        p, lo, hi = wilson(k, len(fire)) if fire else (None, None, None)
        out.append({"threshold": float(t), "fires": len(fire),
                    "pass_rate": len(fire) / len(rows) if rows else None,
                    "tp": k, "precision": p, "precision_lo": lo, "precision_hi": hi,
                    "recall": k / len(good) if good else None,
                    "marks": [f.name for f in (F.MINING_POOL, F.MINING_RELEASE)
                              if abs(f.value - t) < 1e-9]})
    return out


def candidates(lad, targets=PRECISION_TARGETS) -> dict:
    """The LOWEST swept threshold whose precision POINT ESTIMATE reaches each target.

    Point estimate, and the interval is carried alongside — a target met only by a cell whose
    Wilson lower bound sits under the previous rung is a cut the data cannot support, and
    printing the bound is what makes that visible instead of arguable. Derived, never adopted.
    """
    out = {}
    for t in targets:
        hit = next((r for r in lad if r["precision"] is not None and r["precision"] >= t), None)
        out[f"{t:.2f}"] = None if hit is None else {
            "threshold": hit["threshold"], "precision": hit["precision"],
            "precision_lo": hit["precision_lo"], "precision_hi": hit["precision_hi"],
            "recall": hit["recall"], "fires": hit["fires"], "tp": hit["tp"],
            "supported": bool(hit["precision_lo"] is not None and hit["precision_lo"] >= t),
        }
    return out


# --------------------------------------------------------------------------- #
# E — the bulk-sweep detector
# --------------------------------------------------------------------------- #
def find_sweeps(rows, min_run: int = MIN_SWEEP_RUN) -> dict:
    """Contiguous runs (in SHEET ORDER) where every row kept its suggestion.

    DERIVED FROM THE DATA, not read off a flag: the labeling rig exports a bare
    `{id: tier}` map and records no keystroke, so whether Matt swept is only recoverable
    as structure in `sheet_order`. The sheet is sorted good->bad by `pred`, so an
    accept-all-below sweep appears as an unbroken agreement run at the TAIL — which is
    exactly the shape this returns, and it returns the run it found rather than asserting
    one exists.

    `tail_len` is the pure-agreement suffix: the rows most likely confirmed in bulk. They are
    still labels — a confirmed tier-1 at the bottom of the ranking is very probably right —
    but they are not 295 independent judgements, and every rate computed over them says
    `n=960` while resting on far fewer decisions."""
    o = sorted(rows, key=lambda r: r["order"])
    ag = [r["label"] == r["suggested"] for r in o]
    runs, start = [], 0
    for i in range(1, len(ag) + 1):
        if i == len(ag) or ag[i] != ag[start]:
            if ag[start] and i - start >= min_run:
                runs.append({"start": o[start]["order"], "end": o[i - 1]["order"],
                             "len": i - start,
                             "suggested": dict(sorted(Counter(
                                 r["suggested"] for r in o[start:i]).items())),
                             "max_p_ge3": max(r["p_ge3"] for r in o[start:i])})
            start = i
    tail = 0
    for x in reversed(ag):
        if not x:
            break
        tail += 1
    return {"min_run": min_run, "runs": sorted(runs, key=lambda r: -r["len"]),
            "tail_len": tail, "n": len(o),
            "tail_frac": tail / len(o) if o else None,
            "adjudicated_n": len(o) - tail,
            "changed_in_adjudicated": sum(1 for x in ag[:len(ag) - tail] if not x),
            "boundary_order": o[len(o) - tail]["order"] if tail and tail < len(o) else None}


def headline(rows) -> dict:
    """The numbers read E re-runs on each slice, so the sensitivity is like-for-like."""
    return {"n": len(rows), "tiers": tier_dist(rows),
            "ge3": head_block(rows, 3, "p_ge3"), "ge2": head_block(rows, 2, "p_ge2"),
            "pool": cut_block(rows, F.MINING_POOL),
            "release": cut_block(rows, F.MINING_RELEASE)}


def build_E(rows) -> dict:
    """Read E. Everything here is computed from the merged rows; nothing is asserted."""
    sw = find_sweeps(rows)
    o = sorted(rows, key=lambda r: r["order"])
    adjudicated = o[:sw["adjudicated_n"]]
    swept = o[sw["adjudicated_n"]:]
    ev = [r for r in rows if r["split"] == "eval"]
    adj_ev = [r for r in adjudicated if r["split"] == "eval"]

    E = {
        "sweep": sw,
        "swept_tail": {"n": len(swept),
                       "suggested": dict(sorted(Counter(r["suggested"] for r in swept).items())),
                       "labels": dict(sorted(Counter(r["label"] for r in swept).items())),
                       "max_p_ge3": max((r["p_ge3"] for r in swept), default=None),
                       "split": dict(sorted(Counter(r["split"] for r in swept).items()))},
        "slices": {"all_960": headline(rows), "adjudicated": headline(adjudicated),
                   "eval_all": headline(ev), "eval_adjudicated": headline(adj_ev)},
        "july_lock": JULY_LOCK,
    }
    # Does dropping the swept tail move a cut's precision at all? Only if it fires there.
    fires_in_tail = sum(1 for r in swept if F.MINING_POOL.gate(r["p_ge3"]))
    E["tail_sensitivity"] = {
        "pool_fires_inside_tail": fires_in_tail,
        "verdict": ("the swept tail never reaches the lowest live cut, so precision and "
                    "recall at 0.25 and 0.50 are IDENTICAL on both slices; only the base "
                    "rate, the pass rate and the AUC move."
                    if fires_in_tail == 0 else
                    f"{fires_in_tail} swept rows clear the pool cut — the ladders are NOT "
                    f"invariant to the sweep and must be read on the adjudicated slice."),
    }
    now = E["slices"]["all_960"]["release"]
    E["vs_july"] = {
        "same_checkpoint": MP.ACTIVE_MINING_CKPT,
        "now": {"threshold": F.MINING_RELEASE.value, "precision": now["precision"],
                "recall": now["recall"], "pass_rate": now["pass_rate"],
                "base_rate": rows and tier_dist(rows)["frac_ge3"]},
        "july": JULY_LOCK,
        "reading": (
            "Same checkpoint, comparable base rate and comparable pass rate, and precision "
            "roughly doubles while recall roughly doubles. A frozen checkpoint does not "
            "improve, so the difference is a property of the POPULATION and the LABELS, not "
            "of the head. Two known mechanisms push in exactly this direction and this "
            "sitting cannot separate them: (1) these locations are not held out for v1 "
            "(caveat `not_held_out`), and (2) the labels were prefilled with the head's own "
            "suggestion (caveat 0). Neither is a defect in the merge — both are properties "
            "of how the sheet had to be built after the corpus loss."),
        "what_would_separate_them": (
            "a BLIND re-label of a sample of this batch — no prefill, no suggested tier, "
            "served in an order that carries no head signal — scored against the same rows. "
            "The gap between blind and prefilled labels on identical crops is the anchoring "
            "term; what survives it is the head's real calibration. Nothing in this readout "
            "can stand in for that measurement."),
    }
    return E


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cap(r):
    return (f"{r['id']}  L{r['label']} vs S{r['suggested']}\n"
            f"p3 {r['p_ge3']:.2f} {r['mode'][:20]}")


def build(rows) -> dict:
    ev = [r for r in rows if r["split"] == "eval"]
    tr = [r for r in rows if r["split"] == "train"]
    R = {
        "batch": SPEC.batch_id,
        "labels_export": SPEC.labels_export,
        "head": {"version": MP.HEAD_VERSION, "ckpt": MP.ACTIVE_MINING_CKPT,
                 "gate_version": MP.MINING_GATE_VERSION},
        "cuts": {f.name: {"value": f.value, "acts": f.acts, "stamp": f"{f.head}/{f.stamp}"}
                 for f in (F.MINING_POOL, F.MINING_RELEASE)},
        "n_rows": len(rows), "n_eval": len(ev), "n_train": len(tr),
        "n_modes": len({r["mode"] for r in rows}), "n_locations": len({r["loc"] for r in rows}),
        "label_dist": tier_dist(rows),
        "caveats": {
            "labels_are_anchored": "CORRECTION sheet: every row was served with its "
                                   "suggested_tier prefilled and Enter confirmed it, sorted "
                                   "good->bad so an accept-all-below sweep is one keystroke. "
                                   "label and suggested are coupled BY CONSTRUCTION — no "
                                   "agreement, precision or AUC here measures the head "
                                   "against an independent judgement. Read E quantifies it.",
            "not_held_out": "the 112 locations are the gate-passer set BOTH July samplers "
                            "drew from; v1 trained on renders here. Every precision is an "
                            "optimistic bound on a fresh location.",
            "split_is_for_the_next_head": "split_side stamps a clean eval side for a v2 "
                                          "trainer. Both sides are equally exposed to v1.",
            "agreement_measures_the_cuts": "suggested_tier applies per-batch quantile cuts "
                                           "matched to the lost corpus's prior — rank only, "
                                           "no absolute level. Read B is the cut-free read.",
        },
    }

    # ---- A. agreement ------------------------------------------------------ #
    A = {"overall": corr_block(rows), "boundary_overall": boundary_block(rows),
         "eval_overall": corr_block(ev), "eval_boundary": boundary_block(ev),
         "by": {}, "boundary_by": {}, "eval_by": {}}
    for key in ("mode", "kind", "family", "split", "suggested"):
        A["by"][key] = {str(k): corr_block(v) for k, v in by_key(rows, key).items()}
        A["boundary_by"][key] = {str(k): boundary_block(v) for k, v in by_key(rows, key).items()}
    for key in ("mode", "suggested"):
        A["eval_by"][key] = {str(k): corr_block(v) for k, v in by_key(ev, key).items()}
    R["A_agreement"] = A

    worst = sorted(rows, key=lambda r: (r["label"] - r["suggested"], -r["p_ge3"]))
    A["sample"] = {
        "head_overrated": sheet([(r["crop"], cap(r)) for r in worst[:12]],
                                OUT / "A1_head_overrated.png",
                                "A · head OVER-rated: labeled far below its suggestion"),
        "head_underrated": sheet([(r["crop"], cap(r)) for r in worst[-12:][::-1]],
                                 OUT / "A2_head_underrated.png",
                                 "A · head UNDER-rated: labeled far above its suggestion"),
    }

    # ---- B. v1 against the labels ------------------------------------------ #
    B = {}
    for slice_name, sub in (("full", rows), ("eval", ev), ("train", tr)):
        B[slice_name] = {"tiers": tier_dist(sub),
                         "ge2": head_block(sub, 2, "p_ge2"),
                         "ge3": head_block(sub, 3, "p_ge3"),
                         "ge3_on_rank_score": head_block(sub, 3, "rank_score")}
    R["B_head_vs_labels"] = B

    # ---- C. per-mode trust -------------------------------------------------- #
    C = {}
    for mode, sub in by_key(rows, "mode").items():
        ev_sub = [r for r in sub if r["split"] == "eval"]
        C[mode] = {
            "kind": MODE_KIND[mode], "n": len(sub), "n_eval": len(ev_sub),
            # v1 never saw these three. A mode the head was not trained on is a different
            # kind of low AUC from one it was — the first is a gap, the second a failure.
            "untrained_by_v1": mode in TRAINER_DROPPED_V1,
            "tiers": tier_dist(sub),
            "ge3": head_block(sub, 3, "p_ge3"),
            "ge2": head_block(sub, 2, "p_ge2"),
            "agreement": corr_block(sub),
            "boundary": boundary_block(sub),
            "pool": cut_block(sub, F.MINING_POOL),
            "release": cut_block(sub, F.MINING_RELEASE),
        }
    at_chance = sorted(m for m, v in C.items() if v["ge3"]["at_chance"])
    inverted = sorted(m for m, v in C.items()
                      if v["ge3"]["auc"] is not None and v["ge3"]["auc"] < 0.50)
    degenerate = sorted(m for m, v in C.items() if v["ge3"]["auc"] is None)
    R["C_per_mode"] = {"modes": C, "at_chance": at_chance, "inverted": inverted,
                       "degenerate": degenerate,
                       "untrained_by_v1": sorted(TRAINER_DROPPED_V1),
                       "roster_n": len(MODES), "observed_n": len(C),
                       "note": "at_chance = the >=3 AUC's 95% Hanley-McNeil interval covers "
                               "0.50 at this mode's own n. degenerate = the mode has no "
                               "labeled positive (or no negative), so no AUC exists at all — "
                               "which is a stronger statement than chance, not a missing one."}

    # ---- D. calibration ladders --------------------------------------------- #
    D = {}
    for slice_name, sub in (("eval", ev), ("full", rows)):
        lad3 = ladder(sub, score_key="p_ge3", thr=3)
        lad2 = ladder(sub, score_key="p_ge2", thr=2)
        D[slice_name] = {
            "n": len(sub),
            "base_rate_ge3": tier_dist(sub)["frac_ge3"],
            "base_rate_ge2": tier_dist(sub)["frac_ge2"],
            "ladder_ge3": lad3,
            "ladder_ge2": lad2,
            "operating_points": {"mining_pool": cut_block(sub, F.MINING_POOL),
                                 "mining_release": cut_block(sub, F.MINING_RELEASE)},
            "candidates_ge3": candidates(lad3),
            "candidates_ge2": candidates(lad2),
        }
    D["note"] = ("candidate cuts are DERIVED, not adopted. `supported` is whether the "
                 "Wilson lower bound also clears the target; a cut that reaches a precision "
                 "only on the point estimate is a cut this sitting cannot buy.")
    R["D_ladders"] = D

    # ---- E. what this sitting can and cannot adjudicate ---------------------- #
    R["E_adjudicability"] = build_E(rows)
    return R


def write_md(R) -> None:                                                  # noqa: PLR0915
    A, B, C, D = (R["A_agreement"], R["B_head_vs_labels"],
                  R["C_per_mode"], R["D_ladders"])
    w = [f"# render-mode fresh sheet — reads · {R['n_rows']} labeled rows · "
         f"mining head {R['head']['version']}\n",
         f"Batch `{R['batch']}` · labels `{R['labels_export']}` · "
         f"{R['n_modes']} modes over {R['n_locations']} locations · "
         f"eval {R['n_eval']} / train {R['n_train']}.\n",
         f"Label distribution: {R['label_dist']['hist']} "
         f"(>=3 base rate **{pct(R['label_dist']['frac_ge3'])}**, "
         f">=2 **{pct(R['label_dist']['frac_ge2'])}**).\n",
         "\n**Nothing here moves a cut, floor, gate or pin.** The two operating points are "
         f"read from `tools/emission/floors.py`: mining pool "
         f"**{R['cuts']['mining_pool']['value']}** (acting) and mining release "
         f"**{R['cuts']['mining_release']['value']}** (report-only).\n",
         "\n**Three caveats, load-bearing for the decision:**\n"]
    for k, v in R["caveats"].items():
        w.append(f"- **{k}** — {v}")
    w.append("")

    # A
    o, bo, eo, ebo = (A["overall"], A["boundary_overall"], A["eval_overall"], A["eval_boundary"])
    w.append("\n## A · agreement (final label vs suggested tier)\n")
    w.append(f"**{pct(1 - o['agree'])} of {o['n']} rows CHANGED the suggestion.** "
             f"Matt went UP on {pct(o['up'])}, DOWN on {pct(o['down'])}; mean signed delta "
             f"**{o['mean_delta']:+.3f}** tiers, within-one {pct(o['within_one'])}.\n")
    w.append(f"Eval side (n={eo['n']}): changed {pct(1 - eo['agree'])}, up {pct(eo['up'])}, "
             f"down {pct(eo['down'])}, mean delta {eo['mean_delta']:+.3f}.\n")
    w.append(f"\nAt the **>=3 boundary** (the cut the gate makes): full-corpus agree "
             f"{pct(bo['agree'])}, suggestion precision {pct(bo['precision'])}, recall "
             f"{pct(bo['recall'])} ({bo['suggested_ge3']} suggested >=3, {bo['labeled_ge3']} "
             f"labeled >=3). Eval side: precision {pct(ebo['precision'])}, recall "
             f"{pct(ebo['recall'])} (n={ebo['n']}).\n")
    for key, title in (("suggested", "by suggested tier"), ("kind", "by mode kind"),
                       ("family", "by family (stratum)"), ("split", "by split side"),
                       ("mode", "by mode")):
        w.append(f"\n**{title}**\n")
        w.append(md_table(
            [key, "n", "changed", "up", "down", "mean Δ", ">=3 agree", ">=3 prec"],
            [[k, v["n"], pct(1 - v["agree"]), pct(v["up"]), pct(v["down"]),
              f"{v['mean_delta']:+.2f}", pct(A["boundary_by"][key][k]["agree"]),
              pct(A["boundary_by"][key][k]["precision"])]
             for k, v in A["by"][key].items() if v["n"]]))
        w.append("")
    w.append(f"\nSamples: `{A['sample']['head_overrated']}`, "
             f"`{A['sample']['head_underrated']}`\n")

    # B
    w.append("\n## B · v1 against the labels (cut-free)\n")
    w.append("AUC/AP at each tier boundary, on the marginal probability that boundary's gate "
             "would use. `min AUC` is the smallest AUC distinguishable from 0.50 at that "
             "cell's n — the bar this slice can actually clear.\n")
    w.append(md_table(
        ["slice", "boundary", "n", "n pos", "base", "AP", "AUC", "95% lo", "min AUC", "verdict"],
        [[sl, b, v["n"], v["n_pos"], pct(v["base_rate"]), num(v["ap"]), num(v["auc"]),
          num(v.get("auc_lo")), num(v.get("min_detectable")),
          "AT CHANCE" if v.get("at_chance") else "separates"]
         for sl in ("full", "eval", "train")
         for b, v in ((">=2 (p_ge2)", B[sl]["ge2"]), (">=3 (p_ge3)", B[sl]["ge3"]),
                      (">=3 (rank score)", B[sl]["ge3_on_rank_score"]))]))
    w.append("")

    # C
    w.append("\n## C · per-mode trust\n")
    w.append(f"All {C['observed_n']} of the roster's {C['roster_n']} modes are present, 64 rows "
             f"each. (The prompt says 13 — that is the July *sampler's* set; the corpus "
             f"deliberately carries every registered non-smooth mode, and the trainer keeps "
             f"its own drop list.)\n")
    w.append(md_table(
        ["mode", "kind", "v1 saw?", "n", "1", "2", "3", ">=3", "AUC>=3", "95% lo", "min AUC",
         "changed", "P@0.25", "P@0.50", "fires@0.50"],
        [[m, v["kind"], "NO" if v["untrained_by_v1"] else "yes",
          v["n"], v["tiers"]["hist"]["1"], v["tiers"]["hist"]["2"],
          v["tiers"]["hist"]["3"], pct(v["tiers"]["frac_ge3"]), num(v["ge3"]["auc"]),
          num(v["ge3"].get("auc_lo")), num(v["ge3"].get("min_detectable")),
          pct(1 - v["agreement"]["agree"]), pct(v["pool"]["precision"]),
          pct(v["release"]["precision"]), v["release"]["fires"]]
         for m, v in sorted(C["modes"].items(),
                            key=lambda kv: (kv[1]["ge3"]["auc"] is None,
                                            -(kv[1]["ge3"]["auc"] or 0)))]))
    w.append(f"\n**At or near chance ({len(C['at_chance'])}/{C['observed_n']}):** "
             f"{', '.join(C['at_chance']) or '—'}")
    w.append(f"\n**Inverted (AUC < 0.50):** {', '.join(C['inverted']) or '—'}")
    w.append(f"\n**Degenerate (no AUC exists — no labeled positive or no negative):** "
             f"{', '.join(C['degenerate']) or '—'}")
    w.append(f"\n**Never seen by v1 (the trainer's own drop list):** "
             f"{', '.join(C['untrained_by_v1'])}")
    w.append(f"\n{C['note']}\n")

    # D
    for sl in ("eval", "full"):
        d = D[sl]
        w.append(f"\n## D · calibration ladder — {sl} side (n={d['n']})\n")
        w.append(f">=3 base rate **{pct(d['base_rate_ge3'])}**. Precision is of PASSERS; the "
                 f"Wilson interval is on that precision.\n")
        w.append(md_table(
            ["p_ge3 >=", "fires", "pass rate", "TP", "precision", "95% CI", "recall", "mark"],
            [[f"{r['threshold']:.2f}", r["fires"], pct(r["pass_rate"]), r["tp"],
              pct(r["precision"]),
              "—" if r["precision"] is None else f"{pct(r['precision_lo'])}–{pct(r['precision_hi'])}",
              pct(r["recall"]), " ".join(r["marks"]) or ""]
             for r in d["ladder_ge3"]]))
        op = d["operating_points"]
        w.append(f"\n**The two live points, {sl} side.** "
                 f"pool 0.25 (ACTING): fires {op['mining_pool']['fires']}/{d['n']}, precision "
                 f"{pct(op['mining_pool']['precision'])} "
                 f"[{pct(op['mining_pool']['precision_lo'])}–{pct(op['mining_pool']['precision_hi'])}], "
                 f"recall {pct(op['mining_pool']['recall'])}, lift "
                 f"{num(op['mining_pool']['lift'], 2)}x. "
                 f"release 0.50 (REPORT-ONLY): fires {op['mining_release']['fires']}/{d['n']}, "
                 f"precision {pct(op['mining_release']['precision'])} "
                 f"[{pct(op['mining_release']['precision_lo'])}–{pct(op['mining_release']['precision_hi'])}], "
                 f"recall {pct(op['mining_release']['recall'])}, lift "
                 f"{num(op['mining_release']['lift'], 2)}x.\n")
        w.append("\n**Candidate cuts for chosen precisions — DERIVED, NOT ADOPTED**\n")
        w.append(md_table(
            ["target precision", "lowest p_ge3", "achieved", "95% CI", "recall", "fires",
             "supported by the CI?"],
            [[t, "—" if c is None else f"{c['threshold']:.2f}",
              "—" if c is None else pct(c["precision"]),
              "—" if c is None else f"{pct(c['precision_lo'])}–{pct(c['precision_hi'])}",
              "—" if c is None else pct(c["recall"]),
              "—" if c is None else c["fires"],
              "—" if c is None else ("yes" if c["supported"] else "NO")]
             for t, c in d["candidates_ge3"].items()]))
        w.append("")
        w.append(f"\n**The easier job — `p_ge2` against label >=2, {sl} side** "
                 f"(base rate {pct(d['base_rate_ge2'])}). The pool cut is a not-bad question "
                 f"in spirit even though it is set on `p_ge3`.\n")
        w.append(md_table(
            ["p_ge2 >=", "fires", "pass rate", "precision", "recall"],
            [[f"{r['threshold']:.2f}", r["fires"], pct(r["pass_rate"]), pct(r["precision"]),
              pct(r["recall"])] for r in d["ladder_ge2"]]))
        w.append("")
    w.append(f"\n{D['note']}\n")

    # E
    E = R["E_adjudicability"]
    sw, sl, vj = E["sweep"], E["slices"], E["vs_july"]
    w.append("\n## E · what this sitting can and cannot adjudicate\n")
    w.append("**Read this before read D is used to move anything.**\n")
    w.append(f"\n### E1 · the bulk sweep, found rather than assumed\n")
    w.append(f"The sheet was served sorted good->bad, so an accept-all-below sweep shows as an "
             f"unbroken agreement run at the tail. There is one: **{sw['tail_len']} rows "
             f"({pct(sw['tail_frac'])} of the batch), sheet_order {sw['boundary_order']}–"
             f"{sw['n'] - 1}, every one keeping its suggestion.** "
             f"The remaining **{sw['adjudicated_n']}** rows carry "
             f"{sw['changed_in_adjudicated']} changes "
             f"({pct(sw['changed_in_adjudicated'] / sw['adjudicated_n'])} of them).\n")
    st = E["swept_tail"]
    w.append(f"\nThe swept tail is uniform: suggested {st['suggested']}, labeled "
             f"{st['labels']}, max `p_ge3` **{num(st['max_p_ge3'], 4)}**, split {st['split']}. "
             f"These are still labels — a confirmed tier-1 at the bottom of the ranking is "
             f"very probably right — but they are not {st['n']} independent judgements, and "
             f"every rate that reports n=960 rests on {sw['adjudicated_n']} decisions.\n")
    w.append(f"\nRuns of >= {sw['min_run']} found: "
             f"{[(r['len'], r['start']) for r in sw['runs']] or '—'} (length, start order).\n")
    w.append(f"\n**{E['tail_sensitivity']['verdict']}** "
             f"(pool cut fires inside the tail: "
             f"{E['tail_sensitivity']['pool_fires_inside_tail']}).\n")

    w.append("\n### E2 · every headline number, with and without the swept tail\n")
    w.append(md_table(
        ["slice", "n", ">=3 base", "AUC >=3", "AP >=3", "AUC >=2",
         "P@0.25", "R@0.25", "P@0.50", "R@0.50"],
        [[name, v["n"], pct(v["tiers"]["frac_ge3"]), num(v["ge3"]["auc"]), num(v["ge3"]["ap"]),
          num(v["ge2"]["auc"]), pct(v["pool"]["precision"]), pct(v["pool"]["recall"]),
          pct(v["release"]["precision"]), pct(v["release"]["recall"])]
         for name, v in (("all 960", sl["all_960"]), ("adjudicated only", sl["adjudicated"]),
                         ("eval, all", sl["eval_all"]),
                         ("eval, adjudicated", sl["eval_adjudicated"]))]))
    w.append("")

    w.append("\n### E3 · the same checkpoint, against its own July record\n")
    j, n_ = vj["july"], vj["now"]
    w.append(f"`{vj['same_checkpoint']}` is unchanged since July. Its July operating point is "
             f"quoted from {j['source']}, measured on {j['population']}.\n")
    w.append(md_table(
        ["at p_ge3 >= 0.50", "July (lost corpus, held out, blind)", "this sitting (full)"],
        [["precision of passers", pct(j["precision"]), pct(n_["precision"])],
         ["recall", pct(j["recall"]), pct(n_["recall"])],
         ["pass rate", pct(j["pass_rate"]), pct(n_["pass_rate"])],
         [">=3 base rate", pct(j["base_rate"]), pct(n_["base_rate"])]]))
    w.append(f"\n{vj['reading']}\n")
    w.append(f"\n**What would separate them:** {vj['what_would_separate_them']}\n")
    (OUT / "report.md").write_text("\n".join(w), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    log(f"[reads] {len(rows)} labeled rows · {len({r['mode'] for r in rows})} modes · "
        f"{len({r['loc'] for r in rows})} locations")
    R = build(rows)
    (OUT / "report.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    write_md(R)
    log(f"[reads] -> {OUT}")


if __name__ == "__main__":
    main()
