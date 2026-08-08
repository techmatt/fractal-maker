#!/usr/bin/env python
"""Per-family KEEPER cut — a stricter, precision-weighted (F0.5) q3 bar for reporting only.

The discovery bar is `production_seeder.t_good_for` (per-partition, F2 / recall-weighted — it
casts wide so the frontier surfaces candidates). The KEEPER bar is its precision-weighted twin:
the `p_good` cut that maximizes **F0.5** against the human labels, so a "keeper" is a location we
are confident a human would call good. NOTHING gates on it — admission stays at the discovery
`t_good`. Keeper status is a *report-time* filter on the persisted canonical `p_good`:

    keeper(row) := corn_decode(row.p_notbad, row.p_good, keeper_cut_for(partition)) >= 3

Derived exactly like the discovery table (`tools/scoring/derive_t_good.py`), from the frozen eval
slice of the ACTIVE version — `data/<v>/eval_scores_<v>.jsonl` (label / fractal_type / that
head's cumulative probs inline, frozen by `tools/<v>/eval_<v>.py`) — with one change: the
objective is F0.5 (beta=0.5) rather than the discovery table's per-family choice. A partition
with < MIN_POS positives is UNCALIBRATED and falls back to the discovery baseline 0.50,
flagged. Prediction uses `corn_decode` (the fixed `p_notbad>=0.5` gate AND `p_good>=t`),
matching how an admitted keeper decodes.

RECUT AGAINST v10 at the 2026-08-02 flip (was v8, before that v7). The population and column
prefix now resolve from `production_pins.ACTIVE_VERSION` rather than a literal, so the default
derivation moves with the pin instead of quietly re-deriving the previous head's cut.

Two substantive changes beyond the model, both dating from the v8 recut and still in force:

  * The population is now DURABLE. The v7 cuts were derived from
    `data/classifier/v7/eval_scores_v7.jsonl`, which was gitignored, was never committed, and
    is GONE — so `derive()` could not run and the committed constant's provenance stamp named a
    population that no longer existed. `data/v8/eval_scores_v8.jsonl` is `paths.durable()` and
    committed, so this derivation is re-runnable and its stamp is checkable.
  * A KEEPER IS `label >= 3`, NOT `label == 3`. Under v7's 1..3 labels those were the same
    predicate. Under v8's 1..4 they are not: `== 3` would score every class-4 location — the
    best locations in the corpus — as a keeper NEGATIVE, which would push the precision-weighted
    cut in exactly the wrong direction.

The v10 slice calibrates the same two partitions the v8 slice did (the julia:multibrot census
and the mandelbrot loose0_v3 floor). Its third instrument, `maneuver_uniform_v1` (90 rows), adds
unbiased NATIVE-plane rows for mandelbrot and multibrot{3,4,5} but carries ZERO keeper
positives, so it calibrates nothing here either — those partitions stay UNCALIBRATED, now
because we looked rather than because we hadn't. julia:mandelbrot and phoenix, which v7 could
calibrate, remain UNCALIBRATED: a real loss of coverage, reported rather than papered over with
a stale-scale value carried forward.

POPULATION: ONE INSTRUMENT PER PARTITION (`INSTRUMENT` below), the same rule the v10 discovery
derivation uses. The julia:multibrot census-only rule ("Option A") is the original case; the
v10 flip added the mandelbrot case, because v10's slice is the first to carry a SECOND
unbiased mandelbrot instrument (12 `maneuver_uniform_v1` rows alongside the 526-row
`loose0_v3_floor`). Pooling them is not a rounding difference — it moves the mandelbrot cut
0.03 -> 0.08 and collapses the LOO-OOF F0.5 from 0.357 to 0.100, which is 12 zero-positive rows
destabilising the argmax rather than informing it. Cutting on the floor alone is also the
comparability-preserving choice: it is the identical 526 rows v8's keeper cut used, so the
v8 -> v10 move reads as a head change.

  uv run python tools/atlas/keeper_cut.py            # print table + write data/atlas/keeper_cuts.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

from score_lib import corn_decode                    # noqa: E402
from production_seeder import T_GOOD_BASELINE         # noqa: E402

# Column prefix of the scorer whose probabilities the slice carries. `eval_v<N>` writes
# `v<N>_p_ge2` / `_p_ge3` / `_p_ge4`; the v7-era slice wrote `v7_p_not_bad` / `v7_p_good`.
# RESOLVED FROM THE PIN, not hardcoded: the keeper cut is a threshold on the ACTIVE head's
# P(>=3), so the default population must move with the pin or `derive()` silently keeps
# re-deriving the previous head's cut. Was a literal "v8" until the v10 flip.
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
import eval_slice                                # noqa: E402
from derive_t_good import fam_of                 # noqa: E402  THE per-row partition reader
from derive_t_good import select_population as _select_population  # noqa: E402
from partitions import ALL_FAMS                  # noqa: E402  THE list, one copy (was mirrored here)
from production_pins import ACTIVE_VERSION       # noqa: E402

EVAL_VERSION = ACTIVE_VERSION
EVAL = eval_slice.path_for(EVAL_VERSION)
OUT = ROOT / "data" / "atlas" / "keeper_cuts.json"
# ONE INSTRUMENT PER PARTITION — partition -> the eval `source` it is cut on. A partition
# absent here takes every row it has. Generalises the original julia:multibrot "Option A"
# census-only rule to the case v10 introduced (a second unbiased mandelbrot instrument); see
# the module docstring for why pooling two instruments is a different cut, not a bigger one.
# A source named here that the slice does not carry yields an empty partition -> UNCALIBRATED,
# which is the correct read: the instrument this cut is defined on is absent.
INSTRUMENT = {
    "mandelbrot": "loose0_v3_floor",
    "julia:multibrot3": "prospect_census",
    "julia:multibrot4": "prospect_census",
    "julia:multibrot5": "prospect_census",
}
MIN_POS = 15                                          # sufficiency floor (== discovery derivation)
BETA = 0.5                                            # precision-weighted (keeper) objective
GRID = [round(0.02 + 0.01 * i, 2) for i in range(97)]   # [0.02, 0.98]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_triples(eval_path: Path = EVAL, version: str = None) -> dict:
    """{partition: [(p_notbad, p_good, is_pos)]}. One instrument per partition (`INSTRUMENT`).

    `is_pos` is `label >= 3` — a class-4 location is emphatically a keeper. Under v7's 1..3
    labels `== 3` was the same predicate; under v8's 1..4 it would score the best locations in
    the corpus as negatives.

    `version` selects the column prefix (`<version>_p_ge2` / `_p_ge3`) and defaults to
    `EVAL_VERSION`, so every existing call is byte-unchanged. It exists so a NEW eval slice
    can be recut through this exact derivation instead of a copy of it — the slice a
    re-render produces carries its own version's columns, and a copied deriver is how two
    thresholds that are supposed to be comparable stop being comparable.

    THE PHOENIX SPLIT IS RESOLVED FROM THE ROW, since v11. This block used to say the split
    could not be resolved here at all — an eval row carried `fractal_type` and no parameter
    axes, so `FT2FAM` folded `phoenix:classic` into `phoenix` and the note ended "a slice that
    ever carries phoenix must gain the axes before a keeper cut on either half means
    anything." The v11 slice is that slice (211 phoenix rows across both eval roles), and it
    resolves the split at the FREEZE instead: `tools/<v>/eval_<v>.py` writes the partition per
    row. `derive_t_good.fam_of` is the one reader of that column and is taken here rather than
    re-deriving it, so the discovery table and the keeper cut cannot end up grouping the same
    slice two different ways.
    """
    ver = version or EVAL_VERSION
    rows = eval_slice.load(ver, path=eval_path)
    if all("eval_role" in r for r in rows):
        # v11 on: the SAME population rule the discovery table uses — the grouped holdout,
        # `INSTRUMENT` as the per-partition fallback source, never pooled. Taken from the
        # shared estimator rather than re-implemented, because the keeper cut is defined as
        # the discovery table's precision-weighted TWIN: same rows, beta=0.5 instead of the
        # per-family choice. Re-implementing it is how the two stop being twins.
        #
        # THIS MATTERS MORE UNDER v11 THAN IT READS. The rule below (a partition absent from
        # `INSTRUMENT` "takes every row it has") was safe only while those partitions had NO
        # rows. The v11 slice gives them rows on both eval roles, so the same line silently
        # became a POOLED cut — julia:mandelbrot on 302 rows = 254 holdout + 48 instrument,
        # phoenix on 211 = 113 + 98 — which is the one thing this module forbids.
        kept, _ = _select_population(rows, instrument=INSTRUMENT)
    else:
        # v10 and earlier: the frozen one-instrument-per-partition rule, unchanged, so those
        # slices still recut to their committed numbers.
        kept = [r for r in rows
                if INSTRUMENT.get(fam_of(r)) in (None, r.get("source"))]
    parts: dict = defaultdict(list)
    for r in kept:
        part = fam_of(r)
        if part is None:
            continue
        p_nb, p_gd, _ = eval_slice.probs(r, ver)
        parts[part].append((p_nb, p_gd, r["label"] >= 3))
    return parts


def confusion(rows, t):
    tp = fp = fn = 0
    for nb, g, pos in rows:
        pred = corn_decode(nb, g, t) >= 3
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif (not pred) and pos:
            fn += 1
    return tp, fp, fn


def prf_beta(tp, fp, fn, beta=BETA):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    f = ((1 + b2) * prec * rec / denom) if denom else 0.0
    return prec, rec, f


def best_t(rows):
    """argmax F0.5 over GRID; tie-break toward HIGHER t (equal F, fewer FPs = the keeper intent)."""
    best = None
    for t in GRID:
        _, _, f = prf_beta(*confusion(rows, t))
        if best is None or f > best[1] + 1e-12 or (abs(f - best[1]) <= 1e-12 and t > best[0]):
            best = (t, f)
    return best[0]


def loo_f(rows):
    """Leave-one-out OOF F0.5 (honest generalization estimate; small n)."""
    tp = fp = fn = 0
    for i in range(len(rows)):
        rest = rows[:i] + rows[i + 1:]
        t = best_t(rest)
        nb, g, pos = rows[i]
        pred = corn_decode(nb, g, t) >= 3
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif (not pred) and pos:
            fn += 1
    return prf_beta(tp, fp, fn)


def derive(eval_path: Path = EVAL, version: str = None) -> dict:
    """{partition: {t, calibrated, n, pos, prec, rec, f, oof_f}}. Uncalibrated => baseline, flagged."""
    parts = load_triples(eval_path, version)
    out = {}
    # ALL_FAMS, not FT2FAM.values(): a DERIVED partition has no fractal_type of its own, so
    # the value list silently omits every one of them — `phoenix:classic` would never be
    # stamped at all, which reads as "not a partition" rather than "uncalibrated".
    for part in sorted(set(list(parts) + list(ALL_FAMS))):
        rows = parts.get(part, [])
        n = len(rows); pos = sum(1 for _, _, x in rows if x)
        if pos < MIN_POS:
            out[part] = dict(t=T_GOOD_BASELINE, calibrated=False, n=n, pos=pos,
                             prec=None, rec=None, f=None, oof_f=None)
            continue
        t = best_t(rows)
        p_t, r_t, f_t = prf_beta(*confusion(rows, t))
        _, _, oof = loo_f(rows)
        out[part] = dict(t=t, calibrated=True, n=n, pos=pos,
                         prec=round(p_t, 4), rec=round(r_t, 4), f=round(f_t, 4),
                         oof_f=round(oof, 4))
    return out


def load_keeper_cuts(path: Path = OUT) -> dict:
    """Read the persisted table; derive-and-write it if absent. Returns {partition: {...}}."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["cuts"]
    cuts = derive()
    write(cuts, path)
    return cuts


def keeper_cut_for(partition: str, cuts: dict) -> float:
    row = cuts.get(partition)
    return float(row["t"]) if row else T_GOOD_BASELINE


def is_keeper(partition: str, p_notbad: float, p_good: float, cuts: dict) -> bool:
    """`>= 3`: the keeper cut moves the q3 boundary only, so a row that decodes above it is a
    keeper. (Called without the third probability the decode caps at 3 anyway — `>= 3` keeps it
    correct if a caller ever passes one.)"""
    return corn_decode(p_notbad, p_good, keeper_cut_for(partition, cuts)) >= 3


def write(cuts: dict, path: Path = OUT, eval_path: Path = None, version: str = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = eval_path or EVAL
    eval_rel = str(Path(ep).relative_to(ROOT)).replace("\\", "/")
    path.write_text(json.dumps(dict(
        objective="F0.5", beta=BETA, min_pos=MIN_POS, baseline=T_GOOD_BASELINE,
        eval=eval_rel, provenance=provenance_stamp(ep, version), cuts=cuts,
    ), indent=2), encoding="utf-8")


# The scorer version the cuts were derived from is fixed by the derivation code path itself:
# `load_triples` reads EVAL_VERSION-prefixed columns out of EVAL. So the model stamp is
# CHEAPLY DETERMINABLE here (not a history dig) — it is whichever version's columns were read.
# A concrete `model` string is a VERIFIED stamp the test holds to the active checkpoint;
# `model=None` would mean "unverified" (not currently the case, and no longer NEEDED to be:
# the v8 population is durable, so the derivation is re-runnable and the stamp is checkable).
EVAL_MODEL_VERSION = EVAL_VERSION       # scorer whose inline probabilities the slice carries


def provenance_stamp(eval_path: Path = None, version: str = None) -> dict:
    """Where these cuts came from, established from the derivation code path (not git history).

    `model` names the scorer version whose probabilities were the derivation input; `population`
    is the frozen eval slice it read. Both are named by the code, so the stamp is verified.
    Defaults reproduce the v8 stamp exactly; the arguments exist so a recut against a newer
    slice stamps THAT slice's model rather than inheriting this module's constant."""
    ep = eval_path or EVAL
    EVAL_MODEL_VERSION = version or globals()["EVAL_MODEL_VERSION"]
    eval_rel = str(Path(ep).relative_to(ROOT)).replace("\\", "/")
    return dict(
        model=EVAL_MODEL_VERSION,
        verified=True,
        population=eval_rel,
        durable_population=True,
        detail=("frozen eval slice (paths.durable, committed); the DISCOVERY TABLE'S OWN "
                "population rule (derive_t_good.select_population — grouped holdout, "
                f"instrument fallback {INSTRUMENT}) — never pooled; keeper positive = "
                "label >= 3; the only difference from the discovery table is beta=0.5"),
        basis=(f"model + population named by the derivation code path: keeper_cut.load_triples reads "
               f"{EVAL_MODEL_VERSION}_p_ge2/{EVAL_MODEL_VERSION}_p_ge3 from EVAL ({eval_rel}); "
               f"not inferred from history. The population is committed, so `derive()` re-runs and "
               f"this stamp is verifiable rather than asserted."),
    )


def main():
    cuts = derive()
    print("=" * 78)
    print("KEEPER cut (F0.5 / precision-weighted) — report-only; nothing gates on it")
    print("=" * 78)
    print(f"{'partition':20s} {'n':>4s} {'pos':>4s} {'t_keep':>7s} {'F0.5':>6s} "
          f"{'oof':>6s} {'P':>5s} {'R':>5s}  status")
    for part in sorted(cuts):
        d = cuts[part]
        if d["calibrated"]:
            print(f"{part:20s} {d['n']:4d} {d['pos']:4d} {d['t']:7.2f} {d['f']:6.3f} "
                  f"{d['oof_f']:6.3f} {d['prec']:5.2f} {d['rec']:5.2f}  calibrated")
        else:
            print(f"{part:20s} {d['n']:4d} {d['pos']:4d} {d['t']:7.2f} {'--':>6s} "
                  f"{'--':>6s} {'--':>5s} {'--':>5s}  UNCALIBRATED -> baseline {d['t']}")
    write(cuts)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
