#!/usr/bin/env python
r"""tau_h curve — both axes, from the recovered harvest logs.

`harvest_log.jsonl` (one row per harvest check: cheap_pgood x canonical fate, incl.
the canon-not-q3 / precanon-dup REJECTS) was gitignored and lost from the live repo;
it was recovered from a filesystem backup of the old working tree and is now stored
durably in-tree via LFS beside each run's admission ledger. This tool reads it and
builds the per-partition tau_h tradeoff the ledger alone could not:

  * canonical renders saved = f(tau_h)   [cost axis, the half that was missing]
  * admissions retained     = f(tau_h)   [benefit axis]
  * exchange rate = admissions lost per canonical render saved, at distribution-
    adapted candidate steps (rendered-check cheap_pgood percentiles).

Model. Every harvest_log row cleared the LIVE tau_h (harvest only logs checks that
passed the cut), so the curve is defined for candidate cuts tau >= tau_current
(raising the cut). A CANONICAL confirmation render happens for a check iff
cheap_pgood >= tau AND it is not a pre-canonical coord-dup (`precanon_dup is None`;
campaign 1 has no precanon filter, so every check renders). An ADMISSION is a
distinct-q3 survivor of the reframe (`admitted == True`). For candidate tau:
    renders_saved(tau) = #{cheap < tau AND rendered}        / #{rendered}
    admits_retained(tau) = #{cheap >= tau AND admitted}     / #{admitted}

Both are FIRST-ORDER (frozen-cloud): retroactively raising tau shrinks the greedy
dedup cloud, which can only promote later q3_dups to distinct -> true admits_retained
>= the estimate (a lower bound). Labelled as such; not silently exact.

Gates (loud):
  * Reconciliation: harvest_log admitted count must tie to the run summary.
  * Threshold era: rows carry raw canon scores, so the confirmation decode is
    recomputed under CURRENT t_good and asserted equal to the recorded canon_decoded
    (campaign 1/2 already ran at today's t_good -> this is a no-op check that PROVES
    the era matches rather than assuming it).

Raw admissions are the denomination; per-reject distinct-look attribution is not
recoverable (distinct-looks are tallied only on admissions), so it is not reported.

Run: uv run python tools/atlas/tau_h_retained_readout.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mining"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas"))
from tools import run_record            # noqa: E402  (segments-aware run-record layer)
from score_lib import corn_decode  # noqa: E402
from production_seeder import t_good_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DISC = ROOT / "data/discovery"

# The four current-era runs that reconcile to their summaries. Others (shakeout*,
# steered_run2/v1_2) are earlier-era (different tau_h/t_good) and excluded from the
# curve; their logs are archived but not analysed here.
RUNS = ["campaign1/breadth", "campaign1/dive", "campaign2/breadth", "campaign2/dive"]
EXPECT_ADMIT = {"campaign1/breadth": 314, "campaign1/dive": 254,
                "campaign2/breadth": 311, "campaign2/dive": 271}
# campaign 2 breadth changed julia hook spacing 0.2 -> 0.1 at the batch-1211 resume
# (julia_hooks.jsonl: spacing 0.2 for batches 3..1204, 0.1 for 1260..2540; the flip is
# the resume at 1211). That shifts the julia candidate population, so segment there.
SEG_BOUNDARY = {"campaign2/breadth": 1211}

PARTS = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
         "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"]


def load(run):
    p = DISC / run / "harvest_log.jsonl"
    return run_record.require_rows(p)  # segments-aware; absence stays LOUD


def era_gate(rows, run):
    """Recompute the confirmation decode under current t_good; assert == recorded."""
    bad = 0
    for r in rows:
        if r.get("canon_pgood") is None:
            continue  # precanon-dup skip: no confirmation render, no decode to check
        want = corn_decode(r["canon_nb"], r["canon_pgood"], t_good_for(r["partition"]))
        if want != r.get("canon_decoded"):
            bad += 1
    if bad:
        raise SystemExit(f"[era gate] {run}: {bad} rows whose current-threshold decode "
                         f"differs from recorded canon_decoded — run is NOT current-era, "
                         f"stop and segment by era before pooling.")


def seg_rows(rows, run):
    b = SEG_BOUNDARY.get(run)
    if b is None:
        return [("all", rows)]
    return [(f"seg-A(spacing0.2,batch<{b})", [r for r in rows if r["batch"] < b]),
            (f"seg-B(spacing0.1,batch>={b})", [r for r in rows if r["batch"] >= b])]


def part_curve(rows_p, tau_cur):
    """Per-partition arrays + curve for candidate cuts tau >= tau_current."""
    cheap = np.array([r["cheap_pgood"] for r in rows_p])
    rendered = np.array([r.get("precanon_dup") is None for r in rows_p])
    admitted = np.array([bool(r.get("admitted")) for r in rows_p])
    n_render = int(rendered.sum())
    n_admit = int(admitted.sum())
    out = dict(n_checks=len(rows_p), n_render=n_render, n_admit=n_admit, tau_h=tau_cur)
    if n_render == 0 or n_admit == 0:
        out["steps"] = []
        return out

    # distribution-adapted candidate steps: tau at the p25/p50/p75 of RENDERED-check
    # cheap_pgood (>= tau_cur by construction) -> each saves ~25/50/75% of renders.
    rc = np.sort(cheap[rendered])
    steps = []
    for q in (0.25, 0.50, 0.75):
        tau = float(np.quantile(rc, q))
        rendered_saved = int(((cheap < tau) & rendered).sum())
        admits_lost = int(((cheap < tau) & admitted).sum())
        admits_kept = n_admit - admits_lost
        exch = (admits_lost / rendered_saved) if rendered_saved else float("nan")
        steps.append(dict(
            q=q, tau=tau,
            renders_saved=rendered_saved, renders_saved_frac=rendered_saved / n_render,
            admits_lost=admits_lost, admits_retained=admits_kept,
            admits_retained_frac=admits_kept / n_admit,
            admits_lost_per_render_saved=exch,
        ))
    out["steps"] = steps
    return out


def main():
    result = {}
    print("=" * 100)
    print("tau_h curve from recovered harvest logs — canonical renders saved vs admissions retained")
    print("first-order (frozen-cloud); admits_retained is a LOWER bound; raw admissions denomination")
    print("=" * 100)
    for run in RUNS:
        rows = load(run)
        n_adm = sum(1 for r in rows if r.get("admitted"))
        exp = EXPECT_ADMIT[run]
        if n_adm != exp:
            raise SystemExit(f"[reconcile] {run}: harvest_log admitted={n_adm} != summary {exp}"
                             f" — file partial/misattributed, excluding from curve.")
        era_gate(rows, run)
        result[run] = {"admitted_tie": n_adm}
        for seg_name, srows in seg_rows(rows, run):
            print(f"\n### {run}  [{seg_name}]  (checks={len(srows)}, "
                  f"admits={sum(1 for r in srows if r.get('admitted'))})")
            print(f"{'partition':20s} {'tau_h':>6s} {'chk':>5s} {'rnd':>5s} {'adm':>4s} "
                  f"| step q: tau  save%  keptadm%  lost/saved")
            for p in PARTS:
                rp = [r for r in srows if r["partition"] == p]
                if not rp:
                    continue
                c = part_curve(rp, next(r["tau_h"] for r in rp))
                result.setdefault(run, {}).setdefault(seg_name, {})[p] = c
                head = (f"{p:20s} {c['tau_h']:6.3f} {c['n_checks']:5d} {c['n_render']:5d} "
                        f"{c['n_admit']:4d} |")
                if not c["steps"]:
                    print(head + " (no renders/admits)")
                    continue
                seg = "  ".join(f"{s['q']:.2f}:{s['tau']:.3f} {s['renders_saved_frac']*100:4.0f}% "
                                f"{s['admits_retained_frac']*100:4.0f}% "
                                f"{s['admits_lost_per_render_saved']:.3f}" for s in c["steps"])
                print(head + " " + seg)

    outp = ROOT / "scratch/tau_h/curve.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
