#!/usr/bin/env python
"""Build the data-driven descent-harness atom selection set.

Draws **40 atoms — 10 per degree (d2/d3/d4/d5), admitted only, seeded, both
`split` values represented** — from `data/minibrot_roster/roster.jsonl` and
writes `data/descent_harness/selection.json`. The tool reads *only* this subset
file, so extending the study to the rest of the 163 atoms later is a data edit
(re-run with a bigger `--per-degree`, or hand-append ids) — no code change.

Deterministic: a fixed seed drives per-split sampling within each degree, so
re-running reproduces the exact set. Both splits are guaranteed by sampling
`EVAL_PER_DEGREE` from `eval` and the remainder from `train`.

Run:  uv run python tools/descent/build_selection.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER = REPO_ROOT / "data" / "minibrot_roster" / "roster.jsonl"
OUT = REPO_ROOT / "data" / "descent_harness" / "selection.json"

DEGREES = [2, 3, 4, 5]
PER_DEGREE = 10
EVAL_PER_DEGREE = 3          # both splits represented; ~roster eval fraction
SEED = 0xDE5CE27             # "descent" — fixed so the draw is reproducible

# Fields the harness needs per atom (the model-blind ones: degree is fine, no scores).
KEEP = ("id", "degree", "period", "split", "family", "cx", "cy", "fw",
        "f64_margin_deploy_decades", "f64_margin_field_decades")


def load_admitted():
    rows = [json.loads(l) for l in ROSTER.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("admitted")]


def draw(per_degree=PER_DEGREE, eval_per_degree=EVAL_PER_DEGREE, seed=SEED):
    rows = load_admitted()
    rng = random.Random(seed)
    picked = []
    for deg in DEGREES:
        pool = [r for r in rows if r["degree"] == deg]
        ev = sorted([r for r in pool if r["split"] == "eval"], key=lambda r: r["id"])
        tr = sorted([r for r in pool if r["split"] == "train"], key=lambda r: r["id"])
        n_ev = min(eval_per_degree, len(ev), per_degree)
        n_tr = per_degree - n_ev
        if len(tr) < n_tr:
            raise SystemExit(f"degree {deg}: need {n_tr} train, have {len(tr)}")
        chosen = rng.sample(ev, n_ev) + rng.sample(tr, n_tr)
        splits = {r["split"] for r in chosen}
        assert {"train", "eval"} <= splits, f"degree {deg}: both splits required, got {splits}"
        chosen.sort(key=lambda r: r["id"])
        picked.extend(chosen)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-degree", type=int, default=PER_DEGREE)
    ap.add_argument("--eval-per-degree", type=int, default=EVAL_PER_DEGREE)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    picked = draw(args.per_degree, args.eval_per_degree, args.seed)
    atoms = [{k: r[k] for k in KEEP} for r in picked]
    doc = {
        "source_roster": str(ROSTER.relative_to(REPO_ROOT)).replace("\\", "/"),
        "seed": args.seed,
        "per_degree": args.per_degree,
        "eval_per_degree": args.eval_per_degree,
        "note": ("Data-driven descent-harness subset. Extend by re-running with a "
                 "larger --per-degree or hand-appending atom rows; the tool reads "
                 "only this file. No model output (scores/period-ranking) belongs here."),
        "atoms": atoms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")

    by_deg = {d: sum(1 for a in atoms if a["degree"] == d) for d in DEGREES}
    by_split = {}
    for a in atoms:
        by_split[a["split"]] = by_split.get(a["split"], 0) + 1
    print(f"wrote {args.out.relative_to(REPO_ROOT)}: {len(atoms)} atoms")
    print(f"  per-degree: {by_deg}")
    print(f"  per-split:  {by_split}")


if __name__ == "__main__":
    main()
