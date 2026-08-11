r"""winner_rule.py — THE staged-head winner rule and the paired bootstrap it reads.

One owner for both heads. `tools/mining/mining_v2_reads.py` wrote this rule for the K=3
mining head in August and it was correct there; the (28) retrains need the same rule for a
K=4 wallpaper head over a different slice set, and a second copy would be two rules that are
supposed to agree. So the SHAPE lives here and each harness supplies only its own metrics,
its own arms and its own row scores.

THE RULE (`prompts/retrains_28.md`, "Report"):

    the candidate wins iff
      (a) HOLDS   — no PRE-DECLARED no-worse arm is significantly worse than the baseline,
                    AND
      (b) IMPROVES— on the MOTIVATING arm at least one metric is significantly better and
                    none is significantly worse.
    Otherwise the baseline keeps candidacy. The losing head is not "rejected" — it keeps
    the pin it already has, which is what makes a null result cheap.

"Significantly" is a 95% PAIRED bootstrap CI on (candidate - baseline) that excludes 0.
Paired because both heads score identical rows: the row-difficulty variance is enormous
relative to the head difference, and an unpaired interval at these n would call everything
noise and make the rule vacuous.

THREE THINGS THIS MODULE REFUSES TO DO SILENTLY, each because doing them silently is how a
rule stops meaning anything:

  * **An unmeasurable cell votes neither way.** A boundary with no positives (or no
    negatives) in a bootstrap draw yields no value; `n_draws` per metric records how many
    survived, and an arm with `n_draws == 0` is excluded from BOTH clauses and listed in
    `unmeasurable`. Treating it as "not worse" is how a cell nobody could measure passes a
    head.
  * **Multiplicity is counted and printed.** Clause (a) is a conjunction over every arm x
    metric; with 30+ cells the chance that one crosses by luck is not small. The verdict
    carries `n_tests` and `clause_a_failures`, so "it failed on one of 34 cells" and "it
    failed on the pooled eval" are different sentences in the report rather than the same
    boolean.
  * **The two readings of "overall" are both reported.** The prompt's clause (a) says "no
    overall pre-declared metric significantly worse" and its slice list names the pooled
    eval alongside the others. `verdict()` therefore returns `winner` (all no-worse arms)
    AND `winner_pooled_only` (the pooled arm alone), both declared before any number is
    computed. They are reported together; neither is chosen after the fact.

    from tools.scoring.winner_rule import Metric, paired_bootstrap, verdict
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_DRAWS = 4000          # MC error on a 95% endpoint well under 0.005 AUC at n>=150
BOOTSTRAP_SEED = 20260810


@dataclass(frozen=True)
class Metric:
    """One number both heads are compared on.

    `col` names the score array (e.g. "p_ge3"), `thr` the label boundary the positives are
    defined by, `kind` the statistic. K is not hardcoded anywhere: a K=4 head simply
    declares a fourth Metric."""
    key: str
    label: str
    col: str
    thr: int
    kind: str        # "auc" | "ap"


def _ap(y: np.ndarray, s: np.ndarray) -> float | None:
    """Average precision. Local so this module has no sklearn import at call time for a
    quantity three lines of numpy define exactly."""
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Rank AUC (Mann-Whitney), ties averaged."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def value(labels: np.ndarray, scores: dict, m: Metric) -> float | None:
    y = (labels >= m.thr).astype(int)
    s = np.asarray(scores[m.col], dtype=np.float64)
    return _auc(y, s) if m.kind == "auc" else _ap(y, s)


def point_block(labels: np.ndarray, scores: dict, metrics) -> dict:
    """Each metric's point value on one head and one arm, with the arm's own counts."""
    out = {"n": int(len(labels))}
    for m in metrics:
        y = (labels >= m.thr)
        out[m.key] = value(labels, scores, m)
        out[f"{m.key}__n_pos"] = int(y.sum())
        out[f"{m.key}__n_neg"] = int((~y).sum())
    return out


def paired_bootstrap(labels: np.ndarray, base: dict, cand: dict, metrics, *,
                     draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CI on (candidate - baseline) per metric, resampling ROWS in common.

    One resample indexes both heads, so row difficulty cancels and what remains is the
    difference between the heads. A draw in which a boundary degenerates contributes
    nothing to that metric and `n_draws` says how many survived."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    acc = {m.key: [] for m in metrics}
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        for m in metrics:
            va = value(lb, {m.col: np.asarray(base[m.col])[idx]}, m)
            vb = value(lb, {m.col: np.asarray(cand[m.col])[idx]}, m)
            if va is not None and vb is not None:
                acc[m.key].append(vb - va)
    out = {}
    for m in metrics:
        d = np.asarray(acc[m.key], dtype=float)
        if d.size == 0:
            out[m.key] = {"n_draws": 0, "lo": None, "hi": None, "median": None,
                          "significantly_worse": None, "significantly_better": None}
            continue
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        out[m.key] = {"n_draws": int(d.size), "lo": lo, "hi": hi,
                      "median": float(np.median(d)),
                      "significantly_worse": bool(hi < 0.0),
                      "significantly_better": bool(lo > 0.0)}
    return out


def verdict(no_worse: dict, motivating: dict, *, pooled_arm: str,
            baseline: str, candidate: str, rule_text: str = "") -> dict:
    """Apply the rule. `no_worse` / `motivating` are `{arm_name: {metric_key: ci}}`.

    Pure: every input is a CI block, so the rule's other branch can be exercised on
    constructed verdicts instead of only on the one real outcome."""
    if pooled_arm not in no_worse:
        raise KeyError(f"pooled arm {pooled_arm!r} must be one of the no-worse arms "
                       f"({sorted(no_worse)}) — clause (a)'s two readings both need it")

    def scan(block):
        worse, better, unmeasurable, n = [], [], [], 0
        for arm, cis in sorted(block.items()):
            for key, ci in sorted(cis.items()):
                if not ci or ci.get("n_draws", 0) == 0:
                    unmeasurable.append(f"{arm}.{key}")
                    continue
                n += 1
                if ci["significantly_worse"]:
                    worse.append({"arm": arm, "metric": key, "lo": ci["lo"], "hi": ci["hi"],
                                  "median": ci["median"]})
                if ci["significantly_better"]:
                    better.append({"arm": arm, "metric": key, "lo": ci["lo"], "hi": ci["hi"],
                                   "median": ci["median"]})
        return worse, better, unmeasurable, n

    a_worse, a_better, a_unmeas, a_n = scan(no_worse)
    b_worse, b_better, b_unmeas, b_n = scan(motivating)
    p_worse, _, p_unmeas, p_n = scan({pooled_arm: no_worse[pooled_arm]})

    clause_a = not a_worse
    clause_a_pooled = not p_worse
    clause_b = bool(b_better) and not b_worse

    return {
        "rule": rule_text or (
            f"{candidate} wins iff (a) no pre-declared no-worse arm is significantly worse "
            f"than {baseline} (95% paired-bootstrap CI on the delta not entirely below 0) "
            f"AND (b) on the motivating arm at least one metric is significantly better and "
            f"none is significantly worse. Otherwise {baseline} keeps candidacy."),
        "baseline": baseline, "candidate": candidate,
        "clause_a": {"pass": clause_a, "n_tests": a_n, "failures": a_worse,
                     "improvements": a_better, "unmeasurable": a_unmeas,
                     "arms": sorted(no_worse)},
        "clause_a_pooled_only": {"pass": clause_a_pooled, "arm": pooled_arm,
                                 "n_tests": p_n, "failures": p_worse,
                                 "unmeasurable": p_unmeas},
        "clause_b": {"pass": clause_b, "n_tests": b_n, "improvements": b_better,
                     "regressions": b_worse, "unmeasurable": b_unmeas,
                     "arms": sorted(motivating)},
        "winner": candidate if (clause_a and clause_b) else baseline,
        "winner_pooled_only": candidate if (clause_a_pooled and clause_b) else baseline,
        "multiplicity_note": (
            f"clause (a) is a conjunction over {a_n} arm x metric cells; at 95% per cell "
            f"the chance one crosses by luck alone is material, so read `failures` before "
            f"reading `pass`."),
    }
