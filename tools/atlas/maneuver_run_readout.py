#!/usr/bin/env python
r"""maneuver_run_readout.py — the three things a long maneuver run has to be watched on.

Companion to `maneuver_degree_readout.py`, which answers "does the operator fire, and what
does it cost per degree". This one answers the questions that only a LONG run can:

  1. **`quota_bound` / `quota_unfilled` over the run** — the reserved floor's first test on a
     MATURE frontier. `bound` is slots that promoted a node the plain priority top-B would
     not have taken; `unfilled` is slots that went unused for lack of availability. The two
     apart are what say whether the FLOOR or the OPERATOR is the limit at scale.
  2. **Per-degree probe / pop balance** — with the scheduler off, root SUPPLY is `B` per
     family and is balanced by count; pops are by global priority and are not. This reports
     the realized composition and does not pretend it is controlled.
  3. **Cost per batch against reached depth** — per-batch cost MUST rise as lineages deepen.
     A flat cost late in a run is a warning, not reassurance: it means the walk stopped
     going deeper, or something is capping the work per batch.

**PER-BATCH COST IS NOT DURABLY RECORDED BY THE RUN.** It exists only in the driver's
stdout, which for a backgrounded run is wherever the launcher pointed it — and that is
usually a disposable path. So this tool parses it and writes `batch_cost.jsonl` INTO THE RUN
DIR, where the durability class matches the claim being made off it. Without `--log` the
cost half is skipped and says so; it does not silently report the other two as the whole
readout.

  uv run python tools/atlas/maneuver_run_readout.py --run-dir data/discovery/<run> \
      --log <driver stdout> --out scratch/maneuvers/run_readout.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

import minibrot_maneuvers as mnv    # noqa: E402
import paths                        # noqa: E402

BATCH_RE = re.compile(
    r"^\s*batch (\d+): exp=(\d+) cand=(\d+) admitted\(cum\)=(\d+).*?\|\s*(\d+)s active=([\d.]+)m")


def parse_batch_costs(log: Path) -> list[dict]:
    """Per-batch `(batch, expanded, candidates, admitted_cum, seconds, active_min)`.

    Tolerates a resumed run's repeated banner and interleaved warnings; a batch that appears
    twice (a kill replays it) keeps the LAST occurrence, which is the one that completed."""
    by: dict[int, dict] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BATCH_RE.match(line)
        if m:
            b = int(m.group(1))
            by[b] = dict(batch=b, expanded=int(m.group(2)), candidates=int(m.group(3)),
                         admitted_cum=int(m.group(4)), seconds=int(m.group(5)),
                         active_min=float(m.group(6)))
    return [by[k] for k in sorted(by)]


def per_batch_depth(run_dir: Path) -> dict[int, dict]:
    """Depth distribution of the candidates PUSHED in each batch, from `prio_terms.jsonl`.

    This is the durable half of the join: the run records it itself, and it is per-candidate
    rather than per-admission, so it does not inherit the admission gate's selection."""
    p = run_dir / "prio_terms.jsonl"
    if not p.exists():
        return {}
    depths: dict[int, list] = defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("depth") is not None:
            depths[int(r["batch"])].append(int(r["depth"]))
    return {b: dict(n=len(v), mean=round(st.mean(v), 2), max=max(v),
                    p90=sorted(v)[int(0.9 * (len(v) - 1))]) for b, v in depths.items()}


def spearman(a, b):
    if len(a) < 3:
        return None

    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j + 2) / 2.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 4) if den else None


def fmt_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None,
                    help="driver stdout (the ONLY source of per-batch cost)")
    ap.add_argument("--quantiles", type=int, default=5,
                    help="how many equal-count run-order blocks to report trends over")
    ap.add_argument("--out", type=Path,
                    default=paths.scratch("maneuvers", "run_readout.md"))
    a = ap.parse_args()

    rd = a.run_dir
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8")) \
        if (rd / "summary.json").exists() else {}
    state = json.loads((rd / "state.json").read_text(encoding="utf-8")) \
        if (rd / "state.json").exists() else {}
    m = summary.get("maneuvers") or {}
    tot = summary.get("totals") or state.get("totals") or {}
    finished = bool(summary)

    L = [f"# Maneuver long-run readout — `{rd.as_posix()}`", ""]
    L += [("Run **finished**" if finished else "Run **still going** (read off `state.json`; "
           "every number below is a partial-run number)") +
          f": {summary.get('batches', state.get('batch_i'))} batches, "
          f"active {summary.get('active_min', round(state.get('active_s', 0)/60, 2))} min"
          + (f", wall {summary['wall_min']} min (x{summary.get('wall_over_active')} active)"
             if summary.get("wall_min") else "") + ".", ""]

    # ---------------- 1. the floor on a mature frontier ----------------
    L += ["## 1. The reserved floor on a mature frontier", ""]
    bound, unfilled = tot.get("man_quota_bound", 0), tot.get("man_quota_unfilled", 0)
    passed = tot.get("man_quota_passed_over", 0)
    nb = summary.get("batches") or state.get("batch_i") or 1
    quota = m.get("quota", 4)
    L += [fmt_table(["counter", "total", "per batch", "of the quota"],
                    [["`quota_bound` (floor actually binding)", bound,
                      round(bound / nb, 2), f"{bound/(nb*quota):.1%}"],
                     ["`quota_unfilled` (operator had nothing)", unfilled,
                      round(unfilled / nb, 2), f"{unfilled/(nb*quota):.1%}"],
                     ["`quota_passed_over` (distinct candidates that lost a slot)", passed,
                      round(passed / nb, 2), "—"]]), ""]
    verdict = ("the OPERATOR is the limit — most reserved slots had nothing to put in them"
               if unfilled > bound else
               "the FLOOR is the limit — slots are filled, and they are promoting nodes the "
               "plain priority order would not have taken")
    L += [f"**{verdict}.** `passed_over` is the backlog: candidates that were available, "
          f"were not taken, and stayed on the frontier. A large and growing backlog beside a "
          f"small `unfilled` means the quota — not availability — is what bounds how much "
          f"maneuver material the run converts.", ""]
    if state.get("frontier"):
        f = state["frontier"]
        man = sum(1 for n in f if n.get("man"))
        orig = sum(1 for n in f if n.get("branch") == "maneuver")
        L += [f"Frontier at last checkpoint: **{len(f)} nodes, {man} maneuver-descended "
              f"({man/len(f):.0%}), of which {orig} are origins.** The floor, the frontier "
              f"share and every `man_*` counter are over the SUBTREE, not the origins.", ""]
    if tot.get("man_frontier_pruned"):
        L += [f"`man_frontier_pruned` = **{tot['man_frontier_pruned']}** — the maneuver share "
              f"of `FRONTIER_CAP` bound at least once.", ""]
    else:
        L += ["`man_frontier_pruned` = 0 — the frontier never reached the cap, so the "
              "maneuver share was never tested in anger.", ""]

    # ---------------- 2. per-degree balance ----------------
    L += ["## 2. Per-degree probe / pop balance", "",
          "Scheduler OFF, so root SUPPLY is `B` per family and balanced by count; the batch "
          "is popped by GLOBAL priority, so realized work is not. This is the composition, "
          "reported rather than controlled — a per-degree reading off it is confounded by "
          "where the walk chose to go (`measurement_practice.md`).", ""]
    probes, decisions = Counter(), Counter()
    mpath = rd / "maneuvers.jsonl"
    if mpath.exists():
        seen = set()
        for line in mpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            d = mnv.degree_of(r.get("partition") or "")
            if d is None:
                continue
            if r.get("op") == "probe":
                probes[d] += 1
                continue
            if r.get("unused_reason") == "quota_passed_over" or "available" not in r:
                continue
            key = (r.get("batch"), r.get("parent_node_id"), r.get("op"), str(r.get("k")),
                   r.get("atom_key"), r.get("reason"))
            if key in seen:
                continue
            seen.add(key)
            decisions[d] += 1
    pops = Counter()
    ppath = rd / "prio_terms.jsonl"
    if ppath.exists():
        for line in ppath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = mnv.degree_of(json.loads(line).get("partition") or "")
                if d is not None:
                    pops[d] += 1
    degs = sorted(set(probes) | set(decisions) | set(pops))
    if degs:
        tp, td, tq = sum(probes.values()), sum(decisions.values()), sum(pops.values())
        L += [fmt_table(["degree", "governor rolls", "operator decisions",
                         "candidates pushed", "share of pushes"],
                        [[f"d{d}", probes.get(d, 0), decisions.get(d, 0), pops.get(d, 0),
                          f"{pops.get(d,0)/tq:.1%}" if tq else "—"] for d in degs]), ""]
        if tq:
            hi = max(pops.values()) if pops else 0
            lo = min(pops.get(d, 0) for d in degs) if degs else 0
            L += [f"Imbalance across degrees: **{hi/max(1,lo):.1f}:1** on candidates pushed "
                  f"(supply was balanced 1:1 by count).", ""]

    # ---------------- 3. cost per batch vs depth ----------------
    L += ["## 3. Cost per batch against reached depth", ""]
    if not a.log or not Path(a.log).exists():
        L += ["**SKIPPED — no `--log`.** Per-batch cost is not durably recorded by the run; "
              "it exists only in the driver's stdout. This section is absent, not zero.", ""]
    else:
        costs = parse_batch_costs(Path(a.log))
        depths = per_batch_depth(rd)
        # persist the parsed series where the durability matches the claim
        outp = rd / "batch_cost.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for c in costs:
                f.write(json.dumps({**c, **{f"depth_{k}": v
                                            for k, v in (depths.get(c["batch"]) or {}).items()}})
                        + "\n")
        L += [f"{len(costs)} batches parsed from `{Path(a.log).name}`; the series is now "
              f"durable at `{outp.as_posix()}` (it was only in stdout).", ""]
        joined = [(c, depths[c["batch"]]) for c in costs if c["batch"] in depths]
        if len(joined) >= a.quantiles * 2:
            n, qn = len(joined), a.quantiles
            rows = []
            for i in range(qn):
                blk = joined[i * n // qn:(i + 1) * n // qn]
                rows.append([f"{i+1}/{qn}",
                             f"{blk[0][0]['batch']}–{blk[-1][0]['batch']}",
                             round(st.median([c["seconds"] for c, _ in blk]), 1),
                             round(st.mean([c["seconds"] for c, _ in blk]), 1),
                             round(st.mean([d["mean"] for _, d in blk]), 2),
                             max(d["max"] for _, d in blk),
                             round(st.mean([c["candidates"] for c, _ in blk]), 1)])
            L += ["Blocks are equal-count in **run order** — the order the work was actually "
                  "done in, which is the only order a cost trend can be read from "
                  "(`measurement_practice.md`).", "",
                  fmt_table(["block", "batches", "median s", "mean s", "mean depth",
                             "max depth", "mean candidates"], rows), ""]
            rho = spearman([c["batch"] for c, _ in joined], [c["seconds"] for c, _ in joined])
            rho_d = spearman([d["mean"] for _, d in joined],
                             [c["seconds"] for c, _ in joined])
            rho_bd = spearman([c["batch"] for c, _ in joined], [d["mean"] for _, d in joined])
            L += [f"Spearman(batch, seconds) = **{rho}** · Spearman(mean depth, seconds) = "
                  f"**{rho_d}** · Spearman(batch, mean depth) = **{rho_bd}**.", ""]
            # THE TWO FLAT CASES ARE DIFFERENT DIAGNOSES and must not share a verdict.
            # "cost flat while depth RISES" is the alarming one — the work per rung stopped
            # tracking the depth it is being done at. "cost flat because DEPTH is flat" is a
            # statement about the walk: it reached a stationary depth mixture and stayed
            # there, which on this driver is what `M_CAP` + root replenishment produces by
            # construction (a capped root's nodes are evicted and replaced by fresh depth-1
            # roots, so the depth distribution converges instead of marching).
            # THE VERDICT IS ON EFFECT SIZE, NOT ON RANK CORRELATION. Spearman answers "is
            # there a monotone trend", and over hundreds of batches it reports one for a
            # drift far too small to matter: the exploration run ramped 3.2 -> 5.7 rungs in
            # its first 100 batches and then sat between 5.8 and 6.0 for the remaining 740,
            # which is a PLATEAU, and rho = 0.154 called it "rising". The question is
            # whether the walk goes meaningfully deeper, and the unit that matters is a
            # RUNG — a whole extra level of zoom. So the threshold is in rungs, and rho is
            # reported beside it rather than deciding.
            d_first = st.mean([d["mean"] for _, d in joined[:max(1, len(joined) // qn)]])
            d_last = st.mean([d["mean"] for _, d in joined[-max(1, len(joined) // qn):]])
            d_gain = d_last - d_first
            c_first = st.median([c["seconds"] for c, _ in joined[:max(1, len(joined) // qn)]])
            c_last = st.median([c["seconds"] for c, _ in joined[-max(1, len(joined) // qn):]])
            L += [f"First block → last block: mean depth **{d_first:.2f} → {d_last:.2f} "
                  f"({d_gain:+.2f} rungs)**, median cost **{c_first:.0f}s → {c_last:.0f}s**. "
                  f"The verdict below is on that effect size; the rank correlations are "
                  f"reported beside it and do not decide it.", ""]
            flat_cost = c_last <= c_first * 1.15
            flat_depth = d_gain < 1.0                      # less than one rung of zoom
            if flat_cost and flat_depth:
                L += ["**The walk reached a STATIONARY depth mixture — cost is flat because "
                      "depth is flat, not because work per rung was capped.** Both trends "
                      "are flat together, which is the signature of `M_CAP` recycling: a "
                      "root that hits the cap has its nodes evicted and is replaced by fresh "
                      "depth-1 roots, so the depth distribution converges rather than "
                      "marching. Check the block table: if mean depth ramps early and then "
                      "sits, the ramp is the transient and the plateau is the run's real "
                      "operating depth.", "",
                      "**This is still a finding, not reassurance.** A run that plateaus at "
                      "a shallow mean depth is not producing deep material however long it "
                      "runs, and no amount of extra wall clock changes that — the lever is "
                      "`M_CAP` / root supply, not budget.", ""]
            elif flat_cost:
                L += ["**WARNING: per-batch cost is FLAT while DEPTH IS RISING.** This is the "
                      "alarming shape: the work per batch stopped tracking the depth it is "
                      "being done at, which means something is capping work per batch rather "
                      "than the walk having settled. Not reassurance.", ""]
            elif flat_depth:
                L += ["**WARNING: depth is not increasing over the run** even though cost is. "
                      "Cost is rising for some reason other than deeper lineages.", ""]
            else:
                L += ["Cost rises with batch and with depth, which is the expected shape: "
                      "deeper lineages cost more per rung.", ""]
            # Two things that make an EARLY flat reading weak evidence either way, stated
            # beside the number rather than left for the reader to remember.
            L += ["Read the trend with two structural facts in hand. **(a)** A batch expands "
                  "exactly `B` nodes, and `auto_maxiter` grows only LOGARITHMICALLY in `fw`, "
                  "so the true cost-vs-depth slope is gentle by construction and is easily "
                  "swamped by batch-to-batch variance — note how far the block means sit "
                  "above the medians above, which is a heavy tail, not a level. **(b)** "
                  + (f"This run is **{'finished' if finished else 'unfinished'}**"
                     + (f" and this reading covers batches 1–{joined[-1][0]['batch']}"
                        if not finished else "") + ". " ) +
                  ("A flat rate in a short prefix is a warning only in the sense that it is "
                   "not yet evidence: the run has not reached its expensive regime "
                   "(`measurement_practice.md`). Re-read this at the end."
                   if not finished else
                   "The run is complete, so the trend is over the whole thing and the "
                   "warning above, if raised, is about the run rather than about the "
                   "sample."), ""]
        else:
            L += [f"Too few joinable batches ({len(joined)}) to report a trend.", ""]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[readout] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
