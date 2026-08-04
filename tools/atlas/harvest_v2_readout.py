#!/usr/bin/env python
r"""harvest_v2_readout.py — the proving run's verification, as a read rather than a recount.

WHAT A PROVING RUN HAS TO ANSWER, in the prompt's own order. Each is a function below, each
returns a verdict alongside its numbers, and each names the population it was taken on — a
verdict with no population is the shape `measurement_practice.md` §1 opens with.

  1. REALIZED vs INTENDED MIX — the headline. v1 set 70% and realized 19.6%, so a miss here
     is a failed shakedown. Reported per partition in three denominations (minutes,
     candidates, admissions) because they answer different questions and v1's number was
     quoted in the second.
  2. FLOOR vs DEFICIT SHARES — reported SEPARATELY, per the addendum, so the split is visible.
  3. VIEW-SCREEN COVERAGE — do screened rows carry BOTH scores? A ratio nobody prints is a
     claim nobody checked.
  4. TRIGGERED STAMPS SURVIVING MULTIPLE GENERATIONS — the §0 fix, verified on live output
     rather than on the unit fixture: the run's own record must contain triggered-lineage rows
     at MORE THAN ONE depth beyond the trigger, which is exactly what v1 could not produce.
  5. PER-STAGE COUNTS PER CHANNEL — where the candidates went, by supply channel.
  6. UPDATED PER-PARTITION COST-TO-MINE — the prices this run measured, for the next one.

  uv run python tools/atlas/harvest_v2_readout.py --run-dir data/discovery/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _jl(p: Path):
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def mix(summary: dict, launch_intended: dict | None = None,
        effective: dict | None = None) -> dict:
    """1. Realized vs intended, three denominations, plus the L1 gap that summarises them.

    The L1 gap is half the sum of |delta| — total variation distance — so it reads directly as
    "this fraction of the run's time went to the wrong partition". 0 is a perfect mix; v1's
    candidate-stream equivalent was catastrophic.

    TWO INTENTS, AND QUOTING ONLY ONE WOULD BE MISLEADING. The allocation is recomputed every
    pop from LIVE prices, so a partition that turns out expensive has its intended share fall
    while the run is still serving it — the quota is tracking a moving target, not converging
    on a fixed one. Observed on the proving run: multibrot3's intent moved 0.139 -> 0.070
    inside 44 batches purely on price.

    So both are reported. `launch_intended` is the vector PRE-REGISTERED in `run_config.json`
    before batch 1 and is the honest bar — it is the thing that was written down first. The
    final intent is what the allocator was actually steering to at the end, and the gap
    against it is the allocator's own tracking error. A run can legitimately miss the first
    while hitting the second; that is a statement about the price model, not about the pop."""
    q = summary.get("pop_quota")
    if not q:
        return dict(verdict="ABSENT", why="run has no pop_quota block (allocator was off)")
    m = q["mix"]
    li = launch_intended or {}
    rows = []
    for p in sorted(m["minutes"]):
        realized = m["minutes"][p]["realized"]
        rows.append(dict(partition=p,
                         launch=li.get(p),
                         intended=m["minutes"][p]["intended"],
                         effective=m["minutes"][p].get("effective"),
                         min=realized,
                         cand=m["candidates"][p]["realized"],
                         admit=m["admitted"][p]["realized"],
                         delta_min=m["minutes"][p]["delta"],
                         delta_effective=m["minutes"][p].get("delta_effective"),
                         delta_launch=(round(realized - li[p], 4) if p in li else None)))
    gap = m["l1_gap_minutes"]
    eff_gap = m.get("l1_gap_minutes_effective")
    if eff_gap is None and effective:
        # The summary predates the live instrumentation: score against the vector recomputed
        # from the trace by `effective_from_trace`, and say so in the record.
        ev = effective.get("batch_weighted_mean") or {}
        for r in rows:
            p = r["partition"]
            if p in ev:
                r["effective"] = ev[p]
                r["delta_effective"] = round(r["min"] - ev[p], 4)
        if all(r["delta_effective"] is not None for r in rows):
            eff_gap = round(sum(abs(r["delta_effective"]) for r in rows) / 2.0, 4)
    launch_gap = (round(sum(abs(r["delta_launch"]) for r in rows) / 2.0, 4)
                  if li and all(r["delta_launch"] is not None for r in rows) else None)
    # `worst` is the most UNDER-served partition, not the largest |delta|. In a two-partition
    # miss the two deltas are equal and opposite, so |delta| picks one arbitrarily — and the
    # failure a mix report is about is the starved side: v1's story is "native realized 19.6%
    # of an intended 70%", never "julia over-ran".
    worst = min(rows, key=lambda r: (r["delta_min"], r["partition"]))
    return dict(l1_gap=gap, l1_gap_vs_launch=launch_gap, l1_gap_effective=eff_gap,
                per_partition=rows, worst=worst,
                verdict=("PASS" if gap <= 0.10 else "MISS"),
                verdict_vs_launch=(None if launch_gap is None else
                                   ("PASS" if launch_gap <= 0.10 else "MISS")),
                verdict_effective=(None if eff_gap is None else
                                   ("PASS" if eff_gap <= 0.10 else "MISS")),
                headline_verdict=(None if eff_gap is None else
                                  ("PASS" if eff_gap <= 0.10 else "MISS")),
                bar="L1 gap <= 0.10 of minutes (i.e. <=10% of the run's time allocated to the "
                    "wrong partition). Not pre-registered — stated here so the number has a "
                    "reading, and quoted beside the raw gap either way.",
                three_intents=(
                    "THE HEADLINE IS `l1_gap_effective`. Three vectors, and which one a gap "
                    "is taken against decides what it is a statement about. "
                    "`launch` is what run_config.json pre-registered before batch 1 — the "
                    "fixed bar, and a gap against it mixes the pop's error with everything "
                    "the price model learned afterwards. `intended` is the FINAL "
                    "price-updated allocation; a gap against it grades the run on a target "
                    "the run itself moved. `effective` is the time-weighted mean of the "
                    "vector the pop ACTUALLY ACTED ON: a julia partition cannot be popped "
                    "into existence, so when its queue is empty its demand folds into its "
                    "c-plane parent by the documented routing rule, and a run that serves "
                    "the parent is following instructions. Only the third is a statement "
                    "about the allocator."))


def effective_from_trace(run_dir: Path) -> dict | None:
    """Recompute the EFFECTIVE intent from `quota_trace.jsonl` for a run whose summary
    predates the live instrumentation.

    Uses the SHIPPED `pop_quota.fold_julia_intent` on each traced batch's `(intended,
    queue_lens)` rather than a local reimplementation — a second copy of the fold rule here
    would let the readout and the pop disagree about what the pop did, which is the one thing
    this function exists to establish.

    BATCH-weighted, and that is not the same as the live TIME-weighted mean. Per-batch
    durations are not in the trace, so this is the weaker of the two and is labelled as such;
    the `final` vector (the last batch's) is reported beside it because it needs no weighting
    assumption at all."""
    rows = _jl(run_dir / "quota_trace.jsonl")
    if not rows or "intended" not in rows[0]:
        return None
    import pop_quota as pq                                    # noqa: E402
    parts = list(rows[0]["intended"])
    acc = {p: 0.0 for p in parts}
    last = None
    for r in rows:
        eff = (r.get("effective")
               or pq.fold_julia_intent(r["intended"], r.get("queue_lens") or {}, parts))
        last = eff
        for p in parts:
            acc[p] += eff.get(p, 0.0)
    n = len(rows)
    return dict(batch_weighted_mean={p: round(v / n, 4) for p, v in acc.items()},
                final={p: round(v, 4) for p, v in (last or {}).items()},
                n_batches=n,
                weighting="BATCH-weighted (per-batch durations are not in the trace); the "
                          "live path accumulates a TIME-weighted mean, which is the stronger "
                          "one",
                source="recomputed via pop_quota.fold_julia_intent")


def floor_vs_deficit(summary: dict) -> dict:
    """2. The addendum's separate read."""
    q = summary.get("pop_quota")
    if not q:
        return dict(verdict="ABSENT")
    f = q["floor_vs_deficit"]
    alloc = q["allocation"]
    return dict(floor_min=f["floor_min"], deficit_min=f["deficit_min"],
                floor_share=f["floor_share"], deficit_share=f["deficit_share"],
                floored_partitions=alloc["floored"],
                intended_floor_claim=alloc["floor_share_total"],
                per_partition=f["per_partition"],
                verdict=("OK" if f["floor_min"] is not None else "ABSENT"))


def view_screen(summary: dict) -> dict:
    """3. Did the screen run at all, and does every screened row carry BOTH scores?"""
    m = summary.get("maneuvers") or {}
    screened = m.get("view_screened", 0)
    scored = m.get("view_fit_scored", 0)
    return dict(view_prior=m.get("view_prior"), screened=screened,
                unscreenable=m.get("view_unscreenable", 0),
                vetoed=m.get("view_vetoed", 0),
                view_fit_model=m.get("view_fit_model"),
                view_fit_scored=scored, coverage=m.get("view_fit_coverage"),
                sort_key="composite_v3", view_fit_is_sort_key=m.get("view_fit_is_sort_key"),
                verdict=("PASS" if screened > 0 and scored == screened else
                         ("NOT RUN" if screened == 0 else "PARTIAL")),
                v1_comparison="v1: man_view_screened=0 — the screen never ran, so the "
                              "pre-registered bar could not be read")


def triggered_lineage(run_dir: Path) -> dict:
    """4. The §0 fix, verified on the run's OWN output.

    The unit test proves `push_children` carries the stamp; this proves the stamp SURVIVED
    the round trip through the engine. The discriminating fact is DEPTH SPREAD: a triggered
    node is pushed at its trigger's depth, so a run where the stamp still died after one
    generation shows triggered rows at exactly one depth per trigger. More than one distinct
    depth beyond the shallowest is the fix working.

    Cross-checked against `mix_source` lineage, the independent carrier, on every row — a
    disagreement means the stamp is being written from somewhere other than the lineage."""
    rows = _jl(run_dir / "q4_candidates.jsonl")
    if not rows:
        return dict(verdict="NO ROWS", n=0)
    stamped = [r for r in rows if r.get("triggered")]
    lineage = [r for r in rows if str(r.get("mix_source") or "").startswith("triggered:")]
    disagree = sum(1 for r in rows
                   if bool(r.get("triggered")) !=
                   str(r.get("mix_source") or "").startswith("triggered:"))
    depths = Counter(int(r["depth"]) for r in stamped)
    by_part = Counter(r["partition"] for r in stamped)
    generations = len(depths)
    return dict(n_rows=len(rows), stamped=len(stamped), lineage=len(lineage),
                carrier_disagreements=disagree,
                depths=dict(sorted(depths.items())), distinct_depths=generations,
                by_partition=dict(by_part),
                verdict=("PASS" if generations > 1 and disagree == 0 else
                         ("NO TRIGGERS" if not stamped else
                          "SINGLE GENERATION" if generations <= 1 else "CARRIER MISMATCH")),
                bar="triggered rows at MORE THAN ONE depth, and zero carrier disagreements")


def per_stage_per_channel(run_dir: Path) -> dict:
    """5. Where the candidates went, by supply channel.

    The channel is read off `mix_source`'s head token (`sampler`, `phoenix_sampler`,
    `triggered`, `maneuver`, `julia_hook<...`, `native`), which is the tag the root supply
    stamps and every descendant inherits — so this is per-CHANNEL rather than per-partition and
    the two are not substitutes."""
    rows = _jl(run_dir / "q4_candidates.jsonl")
    per: dict = defaultdict(Counter)
    for r in rows:
        ch = str(r.get("mix_source") or "?").split(":")[0].split("<")[0]
        per[ch][r.get("fate", "?")] += 1
        per[ch]["_total"] += 1
    return {ch: dict(sorted(c.items())) for ch, c in sorted(per.items())}


def cost_to_mine(summary: dict) -> dict:
    """6. The prices this run measured, and which of them the clamp bound."""
    q = summary.get("pop_quota")
    if not q:
        return dict(verdict="ABSENT")
    c = q["cost"]
    # `price_aggregate` and `price_samples` post-date the batch-aggregated sampler
    # (2026-08-04) and are absent from every earlier summary, which is a fact about the RECORD
    # rather than a missing measurement — so they read as None and are labelled, not skipped.
    agg = c.get("price_aggregate")
    return dict(price=c["price"], price_raw=c["price_raw"], seed=c["seed"],
                clamped=c["clamped"], clamp_factor=c["clamp_factor"],
                units_mined=c["units_mined"], min_spent=c["min_spent"],
                capped=c["capped"],
                # THE READ THAT MADE THE v1 DEFECT VISIBLE, and the reason it is now beside
                # the EMA instead of reconstructed by hand: min_spent/units is the estimand,
                # `price_raw` is the estimate, and a large gap between them is the alarm.
                price_aggregate=agg,
                price_samples=c.get("price_samples"),
                ema_vs_aggregate=(None if not agg else
                                  {p: round(c["price_raw"][p] / v, 3)
                                   for p, v in agg.items() if v}),
                sampler=("batch-aggregated" if agg is not None
                         else "per-decode (pre-2026-08-04; the inverted sampler)"),
                note="price = active-minutes per unit of currency (a decoded 4 = 1.0, a "
                     "decoded 3 = 0.1) mined as a DISTINCT ADMISSION, sampled ONCE PER BATCH "
                     "from the window's own minutes/units aggregate. A partition with zero "
                     "units mined still carries its seed price — that is not a measurement, "
                     "and a partition with few sampled windows is EMA-lagged toward that seed.")


def readout(run_dir: Path) -> dict:
    sp = run_dir / "summary.json"
    if not sp.exists():
        raise SystemExit(f"{sp} missing — the run has not finished (or was killed before "
                         f"`finish()`); state.json is not a substitute.")
    s = json.loads(sp.read_text(encoding="utf-8"))
    # The PRE-REGISTERED intent, read from the config written before batch 1 — the bar that
    # was written down first, as opposed to the one the price model arrived at.
    cfg_p = run_dir / "run_config.json"
    launch = None
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        launch = ((cfg.get("pop_quota") or {}).get("intended") or {}).get("share")
    eff_trace = effective_from_trace(run_dir)
    return dict(
        run=run_dir.name,
        active_min=s.get("active_min"), wall_min=s.get("wall_min"),
        batches=s.get("batches"), wall_over_active=s.get("wall_over_active"),
        totals=s.get("totals"),
        launch_intended=launch,
        effective_intent=eff_trace,
        realized_vs_intended=mix(s, launch, eff_trace),
        floor_vs_deficit=floor_vs_deficit(s),
        view_screen=view_screen(s),
        triggered_lineage=triggered_lineage(run_dir),
        per_stage_per_channel=per_stage_per_channel(run_dir),
        cost_to_mine=cost_to_mine(s),
        library_seed=s.get("library_seed"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = readout(Path(a.run_dir))
    txt = json.dumps(rep, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)

    m = rep["realized_vs_intended"]
    print("\n=== REALIZED vs INTENDED ===", file=sys.stderr)
    if m.get("per_partition"):
        print(f"{'partition':22s}{'launch':>8}{'intend':>8}{'effect':>8}{'min':>8}"
              f"{'cand':>8}{'admit':>8}{'d(eff)':>9}", file=sys.stderr)
        for r in m["per_partition"]:
            lz = "  n/a  " if r["launch"] is None else f"{r['launch']:8.3f}"
            ef = "  n/a  " if r["effective"] is None else f"{r['effective']:8.3f}"
            de = "    n/a  " if r["delta_effective"] is None else f"{r['delta_effective']:+9.3f}"
            print(f"{r['partition']:22s}{lz}{r['intended']:8.3f}{ef}{r['min']:8.3f}"
                  f"{r['cand']:8.3f}{r['admit']:8.3f}{de}", file=sys.stderr)
        print(f"L1 gap vs FINAL     intent = {m['l1_gap']:.3f}  -> {m['verdict']}",
              file=sys.stderr)
        if m.get("l1_gap_vs_launch") is not None:
            print(f"L1 gap vs LAUNCH    intent = {m['l1_gap_vs_launch']:.3f}  -> "
                  f"{m['verdict_vs_launch']}   (pre-registered, fixed)", file=sys.stderr)
        if m.get("l1_gap_effective") is not None:
            print(f"L1 gap vs EFFECTIVE intent = {m['l1_gap_effective']:.3f}  -> "
                  f"{m['verdict_effective']}   <-- THE HEADLINE (what the pop acted on)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
