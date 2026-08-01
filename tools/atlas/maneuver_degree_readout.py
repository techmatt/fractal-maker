#!/usr/bin/env python
r"""Per-degree readout of a maneuver-enabled steered run: AVAILABILITY and COST.

NOT AN EVALUATION OF MOVE QUALITY. Per-move yield is unreadable until a head has been
trained on the population these moves generate — the deployed scorer has never seen a
maneuver-originated view — so nothing here reads admissions as a verdict on an operator.
What it measures is whether the operators can source material across mandelbrot d2 →
multibrot d5 at all, and what each degree costs.

DEDUP IS NOT OPTIONAL. `maneuvers.jsonl` is append-only and a killed batch is replayed on
resume, so the log is a SUPERSET of the checkpointed counters: every rate quoted off it is
computed after de-duplicating on `(batch, parent_node_id, op, k)`. The shakedown's log
carried 1,788 rows for 1,644 distinct decisions (8% inflation, all in the snap rows).

COST ATTRIBUTION. `snap_to_nucleus_multi` charges its one shared Newton solve to the FIRST
k only (`extra.reused_solve` marks the others), so `probe_s` sums correctly over rows and a
per-k cost column is not N copies of one solve. Reused-solve rows are counted in the
availability tables and excluded from the per-k COST table, which is stated in the output
rather than left to be inferred.

  uv run python tools/atlas/maneuver_degree_readout.py \
      --run-dir data/discovery/maneuver_degree_probe --out scratch/maneuvers/degree_readout.md
"""
from __future__ import annotations

import argparse
import json
import math
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

M_CAP = 40                          # mirrors steered_frontier.M_CAP
OPS = ("snap_to_nucleus", "lateral_to_sibling")


def load_maneuvers(run_dir: Path):
    """(deduped operator rows, governor 'probe' rows, n_raw, n_dropped)."""
    rows = [json.loads(l) for l in
            (run_dir / "maneuvers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, ops, probes = set(), [], []
    for r in rows:
        if r.get("op") == "probe":
            probes.append(r)
            continue
        key = (r.get("batch"), r.get("parent_node_id"), r.get("op"), str(r.get("k")))
        if key in seen:
            continue
        seen.add(key)
        ops.append(r)
    return ops, probes, len(rows), len(rows) - len(ops) - len(probes)


def kstr(k):
    return "none" if k is None else (f"{float(k):g}")


def degree_of(row) -> int | None:
    return mnv.degree_of(row.get("partition") or "")


def fmt_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def q(vals, f):
    v = sorted(vals)
    return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=paths.scratch("maneuvers", "degree_readout.md"))
    a = ap.parse_args()

    ops, probes, n_raw, n_drop = load_maneuvers(a.run_dir)
    summary = json.loads((a.run_dir / "summary.json").read_text(encoding="utf-8"))
    state_p = a.run_dir / "state.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}

    L = [f"# Maneuver per-degree readout — `{a.run_dir.as_posix()}`", ""]
    m = summary.get("maneuvers", {})
    L += [f"Run: {summary.get('batches')} batches, active {summary.get('active_min')} min, "
          f"families {', '.join(summary.get('families', []))}, "
          f"k={m.get('ks')}, probe_p={m.get('probe_p')}, quota={m.get('quota')}.",
          "",
          f"`maneuvers.jsonl`: {n_raw} raw rows -> **{len(ops)} distinct operator decisions** "
          f"(+{len(probes)} governor rows); {n_drop} dropped as append-only replay duplicates.",
          ""]

    degs = sorted({d for r in ops if (d := degree_of(r)) is not None})
    part_of = {}
    for r in ops:
        d = degree_of(r)
        if d is not None:
            part_of[d] = r.get("partition")

    # ---------------- 1. availability per (degree, op) ----------------
    L += ["## 1. Availability, per degree and per operator", "",
          "`avail` is the operator's own success rate; `pushed` is what actually became a "
          "frontier node (an available maneuver whose atom was already visited is *not* "
          "pushed — a different constraint, kept separate).", ""]
    rows = []
    for d in degs:
        for op in OPS:
            rs = [r for r in ops if degree_of(r) == d and r["op"] == op]
            if not rs:
                continue
            av = sum(1 for r in rs if r["available"])
            used = sum(1 for r in rs if r.get("used"))
            rows.append([f"d{d} ({part_of[d]})", op, len(rs), av,
                         f"{av/len(rs):.1%}", used, av - used])
    L += [fmt_table(["degree", "op", "calls", "available", "avail rate", "pushed",
                     "avail-but-unused"], rows), ""]

    # ---------------- 2. refusal reasons by (op, k) ----------------
    L += ["## 2. Refusal reasons, by `(op, k)` and degree", "",
          "`k` is stamped on the unavailable path, so this split is attributable for the "
          "first time. Two reasons are k-DEPENDENT by construction — `f64_spacing_wall` "
          "and `fw_over_root_scale` — and neither fired in the shakedown.", ""]
    for d in degs:
        rs = [r for r in ops if degree_of(r) == d and not r["available"]]
        if not rs:
            continue
        by = defaultdict(Counter)
        for r in rs:
            by[(r["op"], kstr(r.get("k")))][r.get("reason", "?")] += 1
        reasons = sorted({x for c in by.values() for x in c})
        rows = [[f"`{op}` k={k}", sum(by[(op, k)].values())]
                + [by[(op, k)].get(x, 0) or "" for x in reasons]
                for (op, k) in sorted(by)]
        L += [f"### d{d} ({part_of[d]})", "",
              fmt_table(["(op, k)", "refusals"] + reasons, rows), ""]

    # ---------------- 3. cost ----------------
    L += ["## 3. Cost per operator and degree", "",
          "Reused-solve snap rows (the 2nd/3rd `k` off one Newton pass) carry only their "
          "framing time and are shown separately — summing every row IS the true cost.", ""]
    rows = []
    for d in degs:
        for op in OPS:
            rs = [r for r in ops if degree_of(r) == d and r["op"] == op
                  and not (r.get("extra") or {}).get("reused_solve")]
            if not rs:
                continue
            ts = [float(r.get("probe_s") or 0) for r in rs]
            rows.append([f"d{d}", op, len(rs), f"{sum(ts):.1f}s",
                         f"{1000*st.median(ts):.0f}", f"{1000*sum(ts)/len(ts):.0f}",
                         f"{1000*q(ts, .9):.0f}", f"{max(ts):.2f}",
                         sum(int(r.get("newton_solves") or 0) for r in rs)])
    L += [fmt_table(["degree", "op", "solve-paying calls", "total", "median ms", "mean ms",
                     "p90 ms", "max s", "Newton solves"], rows), ""]
    reused = [r for r in ops if (r.get("extra") or {}).get("reused_solve")]
    L += [f"Reused-solve rows: **{len(reused)}** "
          f"({sum(float(r.get('probe_s') or 0) for r in reused):.2f}s total — the k=4/k=16 "
          f"framings, which is the whole point of charging them separately).",
          f"Whole-run probe cost: {m.get('probe_s')}s = "
          f"{100*float(m.get('probe_share_of_active') or 0):.1f}% of active time.", ""]

    # ---------------- 4. joint period x fw of pushed nodes ----------------
    L += ["## 4. Pushed nodes: JOINT period x `fw`, and `log10|A|`", "",
          "Joint, not two marginals — a later draw has to match on both, and marginals "
          "cannot support that.", ""]
    for d in degs:
        pushed = [r for r in ops if degree_of(r) == d and r.get("used")]
        if not pushed:
            L += [f"### d{d} ({part_of[d]}) — no pushed nodes", ""]
            continue
        pb = [1, 2, 4, 8, 16, 32, 64, 10 ** 9]
        fb = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 10.0]
        cell = Counter()
        for r in pushed:
            p, fw = int(r["period"] or 0), float(r["fw"] or 0)
            pi = max(i for i in range(len(pb) - 1) if pb[i] <= p) if p >= 1 else 0
            fi = next((i for i in range(len(fb) - 1) if fb[i] <= fw < fb[i + 1]), len(fb) - 2)
            cell[(pi, fi)] += 1
        head = ["period \\ fw"] + [f"[{fb[i]:.0e},{fb[i+1]:.0e})" for i in range(len(fb) - 1)]
        rows = []
        for pi in range(len(pb) - 1):
            if not any(cell[(pi, fi)] for fi in range(len(fb) - 1)):
                continue
            lab = f"{pb[pi]}-{pb[pi+1]-1}" if pb[pi + 1] < 10 ** 9 else f">={pb[pi]}"
            rows.append([lab] + [cell[(pi, fi)] or "." for fi in range(len(fb) - 1)])
        la = [float(r["log10_abs_A"]) for r in pushed if r.get("log10_abs_A") is not None]
        per = [int(r["period"]) for r in pushed if r.get("period")]
        L += [f"### d{d} ({part_of[d]}) — {len(pushed)} pushed", "",
              fmt_table(head, rows), "",
              f"`log10|A|`: min {min(la):.2f} / med {q(la,.5):.2f} / max {max(la):.2f}  ·  "
              f"period: min {min(per)} / med {q(per,.5)} / max {max(per)}  ·  "
              f"f64 node margin (decades): med "
              f"{q([float(r['f64_margin_node_decades']) for r in pushed if r.get('f64_margin_node_decades') is not None], .5):.2f}",
              ""]

    # ---------------- 5. quota ----------------
    L += ["## 5. The reserved floor", "",
          f"- `quota_bound` = **{m.get('quota_bound')}** (reserved slots that promoted a "
          f"node the plain priority top-B would NOT have taken — the floor actually binding)",
          f"- `quota_unfilled` = **{m.get('quota_unfilled')}** (reserved slots unused for "
          f"lack of AVAILABILITY, never a stall)",
          f"- nodes pushed {m.get('nodes_pushed')}, expanded {m.get('nodes_expanded')}, "
          f"admitted {m.get('admitted')}", ""]

    # ---------------- 6. M_CAP ----------------
    epr = {str(k): int(v) for k, v in (state.get("expansions_per_root") or {}).items()}
    capped = sorted([r for r, n in epr.items() if n >= M_CAP], key=lambda r: -epr[r])
    man_per_root = Counter(str(r.get("root_id")) for r in ops if r.get("used"))
    L += ["## 6. `M_CAP` and the inherited `root_id`", "",
          f"{len(capped)} of {len(epr)} roots hit `M_CAP`={M_CAP} "
          f"(`cap_hits`={summary.get('totals', {}).get('cap_hits')}).", "",
          "A maneuver node inherits the parent's `root_id` and therefore burns that root's "
          "cap. Per-expansion provenance is not logged, so `man pushed under root` is an "
          "UPPER BOUND on how many of a capped root's expansions were maneuver-originated "
          "— it is what rules the mechanism out, not what proves it.", ""]
    rows = [[r, epr[r], man_per_root.get(r, 0),
             f"{man_per_root.get(r, 0)/M_CAP:.0%}"] for r in capped[:15]]
    L += [fmt_table(["root_id", "expansions", "man pushed under root", "<= share of cap"],
                    rows) if rows else "_no capped roots_", "",
          f"Capped roots with ZERO maneuver nodes pushed under them: "
          f"**{sum(1 for r in capped if not man_per_root.get(r))}/{len(capped)}** "
          f"— for those, maneuvers demonstrably did not cause the cap.", ""]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[readout] -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
