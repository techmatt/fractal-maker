"""The pre-registered `view_fit_v1.1 vs composite_v3` bar, READ.

The bar was written to disk before the harvest-v2 proving run's batch 1
(`steered_frontier.SteeredFrontier.PREREG["view_fit_v1_1_vs_composite_v3"]`): adopt
view_fit v1.1 as the SOURCING order only if its ordered top-k beats composite_v3's
on that run's labeled outcome by `delta-AP >= +0.1181`. The margin is the LOWER
bound of the fit-era CI (`data/atlas/view_fit_v1_1.json` readout:
ap_delta_v11_vs_composite = 0.1819 [0.1181, 0.2466], n=580, 149 positives).

Those labels now exist (the 2026-08-03 v2 sitting), so this module takes the read
once and freezes it. It is READ-ONLY by construction: it imports the scores, it
never writes an order, and adoption is a separate step that does not live here.

Three things it does NOT re-derive, because a second copy is how two authorities
start disagreeing about what was measured:
  * the SLICE — `sitting_cutter.is_bar_readable`, the same predicate the cut stamped
    `batch.json.bar_readability` with;
  * the METRIC and its uncertainty — `view_fit._metrics` / `view_fit._boot_ap_delta`
    (paired row bootstrap, n=2000, seed=FIT_SEED), the functions that produced the
    fit-era CI the margin was cut from;
  * the LABEL — `label_store.resolve_score`, the one resolution rule.

WHAT THE READ IS WORTH is itself part of the record, so `attainable` is computed
beside the verdict. AP is base-rate dependent: on a slice whose positive rate is
0.963 the WHOLE range of AP between a perfect ordering and the worst possible one
is 0.1256 wide, so a +0.1181 margin asks view_fit to be perfect while composite_v3
is simultaneously worst-possible on the same rows. A NOT-MET here is therefore an
UNINFORMATIVE read, not evidence against view_fit — `bar_attainable_ratio` (margin
÷ attainable range) is the number that says so, and a verdict is reported with it
or not at all.

  uv run python tools/atlas/view_fit_bar_read.py            # print the read
  uv run python tools/atlas/view_fit_bar_read.py --write    # + freeze the record
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "corpus", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import label_store as ls                        # noqa: E402
import sitting_cutter as sc                     # noqa: E402
import view_fit as vf                           # noqa: E402

# The batch the read runs on: the harvest-v2 proving run's one labelling sitting.
SITTING_BATCH = "2026-08-03_v2_sitting_v1"
RECORD_REL = "data/atlas/view_fit_v1_1_bar_read.json"
MARGIN = 0.1181                                 # == PREREG[...]["margin"]; asserted in main()


def _slice(batch: str = SITTING_BATCH):
    """(rows, labels) for the bar-readable slice of `batch` — rows carrying BOTH scores."""
    p = paths.durable(f"data/label_corpus/batches/{batch}/images.jsonl")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    sl = [r for r in rows if sc.is_bar_readable(r["provenance"])]
    side, amd = ls.sidecar_for(batch), ls.amendments_for(batch)
    lab = np.array([ls.resolve_score(r, side, amd) for r in sl], dtype=object)
    return rows, sl, lab


def attainable_range(y) -> float:
    """Width of AP's range on THIS label vector: AP(perfect ordering) − AP(worst ordering).

    Any delta-AP between two orderings of the same rows is bounded by it, so it is the
    ceiling a pre-registered margin has to fit under. The worst ordering puts every
    negative ahead of every positive, making the k-th positive's precision k/(k+n_neg).
    """
    y = np.asarray(y, int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return 0.0
    worst = float(np.mean([(k + 1) / (k + 1 + nneg) for k in range(npos)]))
    return 1.0 - worst


def read(batch: str = SITTING_BATCH, *, target: int | None = None) -> dict:
    """Take the read. `target` defaults to the registered positive rule (view_fit.TARGET_LABEL)."""
    tgt = vf.TARGET_LABEL if target is None else target
    rows, sl, lab = _slice(batch)
    if any(s is None for s in lab):
        raise SystemExit(f"{batch}: {sum(s is None for s in lab)} bar-readable rows are "
                         f"UNLABELED — the bar cannot be read until the sitting is merged.")
    lab = lab.astype(int)
    fit = np.array([r["provenance"]["fit_score"] for r in sl], float)
    comp = np.array([r["provenance"]["composite"] for r in sl], float)
    y = (lab >= tgt).astype(int)

    m_f, m_c = vf._metrics(fit, y), vf._metrics(comp, y)
    boot = vf._boot_ap_delta(fit, comp, y)
    delta = round(m_f["ap"] - m_c["ap"], 4)
    span = round(attainable_range(y), 4)

    # Sensitivity: rows the screen VETOED carry a sentinel composite rather than a score.
    keep = [i for i, r in enumerate(sl) if not r["provenance"].get("vetoed")]
    sens = vf._boot_ap_delta(fit[keep], comp[keep], y[keep]) if len(keep) < len(sl) else None

    from collections import Counter
    return dict(
        schema_version=1,
        prereg="steered_frontier.SteeredFrontier.PREREG['view_fit_v1_1_vs_composite_v3']",
        margin=MARGIN,
        margin_basis="lower bound of the fit-era CI, data/atlas/view_fit_v1_1.json",
        batch=batch,
        slice=dict(
            definition="sitting_cutter.is_bar_readable — fit_model==view_fit_v1.1 "
                       "and fit_score and composite all present",
            n=len(sl), of=len(rows),
            by_partition=dict(Counter(r["provenance"]["family"] for r in sl).most_common()),
            label_dist={str(k): v for k, v in sorted(Counter(lab.tolist()).items())},
        ),
        target=f"label >= {tgt}",
        n_pos=int(y.sum()), n_neg=int((1 - y).sum()), base_rate=round(float(y.mean()), 4),
        ap=dict(view_fit_v1_1=m_f["ap"], composite_v3=m_c["ap"], delta=delta),
        auc=dict(view_fit_v1_1=m_f["auc"], composite_v3=m_c["auc"]),
        bootstrap=dict(**boot, seed=vf.FIT_SEED, unit="row (paired)"),
        attainable=dict(
            max_abs_delta_ap=span,
            bar_attainable_ratio=round(MARGIN / span, 4) if span else None,
            why="AP is base-rate dependent; this is AP(perfect) - AP(worst) on THIS slice, "
                "the ceiling on any delta two orderings of these rows can produce.",
        ),
        vetoed_sensitivity=sens,
        verdict="MET" if boot["lo"] >= MARGIN else "NOT MET",
        verdict_qualifier=(
            "UNINFORMATIVE — the margin is {:.0%} of the entire attainable delta-AP range on "
            "this slice, so NOT MET is not evidence against view_fit v1.1.".format(
                MARGIN / span) if span and MARGIN / span > 0.5 else
            "the margin fits inside the attainable range with room; the read is informative."),
        population_caveat=(
            "Native-plane multibrot MANEUVER views only (the slice carries no julia and no "
            "phoenix row). Any verdict is scoped to that population, not to the pipeline."),
        scope="SOURCING-side only. This module reads; it changes no order.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=SITTING_BATCH)
    ap.add_argument("--target", type=int, default=None,
                    help="positive rule (default: the registered view_fit.TARGET_LABEL)")
    ap.add_argument("--write", action="store_true", help=f"freeze the read to {RECORD_REL}")
    a = ap.parse_args()

    import steered_frontier as sf                # noqa: E402  (heavy; only needed to verify)
    reg = sf.SteeredFrontier.PREREG["view_fit_v1_1_vs_composite_v3"]["margin"]
    assert reg == MARGIN, f"margin drifted from the pre-registration: {reg} != {MARGIN}"

    out = read(a.batch, target=a.target)
    print(json.dumps(out, indent=1))
    if a.write:
        p = paths.durable(RECORD_REL, mkparents=True)
        p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
