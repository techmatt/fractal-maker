#!/usr/bin/env python
r"""near_minibrot_rung.py — measure the per-rung COST of a near-minibrot julia candidate.

WHY THIS EXISTS AS A TOOL AND NOT AS A SCRATCH SCRIPT. It is the only producer of
`data/atlas/near_minibrot_rung_v2.json`, which `supply_routing.rung_choice` refuses to answer
without — so by `CLAUDE.md`'s rule ("if a file is the only thing producing a durable
artifact, it isn't scratch") it belongs here.

THE QUESTION. The 1x/4x/16x ladder buys ~1 look per atom (same-atom different-rung pairs sit
at median cos 0.9825, 74.1% at or above the 0.974 near-dup cut), so v2 emits ONE rung. Human
yields are flat across the three — >=3 at 68.0 / 63.5 / 68.0%, one-per-cluster 61.8 / 65.3 /
66.7%, every pair's Wilson interval overlapping — so yield cannot choose and cost is supposed
to. This measures that cost.

THE DESIGN IS PAIRED. Every timed atom contributes all three rungs, interleaved, so machine
drift and thermal state hit the three arms equally. Timing three blocks in sequence would
confound rung with when-it-ran, which on a shared desktop is the larger effect
(`measurement_practice.md`: a control arm must differ in one thing).

WHAT IS TIMED is the harvest CONFIRMATION render (`prescreen._render` at its default
geometry), because that is the per-candidate cost the rung could plausibly move: the c sits
1 / 4 / 16 atom radii from a nucleus, and a julia at a c deeper inside M has more
non-escaping pixels to run to the cap. Wall clock, dated, alongside the work count — see
`measurement_practice.md` on quoting the invariant.

  uv run python tools/atlas/near_minibrot_rung.py measure --n-atoms 24
  uv run python tools/atlas/near_minibrot_rung.py show
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "mining",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                    # noqa: E402
import prescreen                                # noqa: E402
import supply_routing as sr                     # noqa: E402

BATCH_ID = "2026-08-03_q4_near_minibrot_v1"
BATCH = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID / "images.jsonl"
DRAW_SEED = 20260803


def _by_atom() -> dict:
    """atom_id -> {rung: (c_re, c_im, fw)} for the labelled ladder batch."""
    if not BATCH.exists():
        raise SystemExit(f"{BATCH} is missing — the ladder batch is a tracked durable "
                         f"artifact and this measurement is defined on it.")
    out: dict = {}
    for line in BATCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pv, rd = r["provenance"], r["render"]
        out.setdefault(pv["atom_id"], {})[float(pv["ladder_rung"])] = (
            float(rd["c_re"]), float(rd["c_im"]), float(rd["fw"]))
    return out


def measure(n_atoms: int = 24, out: Path | None = None) -> dict:
    by_atom = _by_atom()
    full = [a for a, d in by_atom.items()
            if set(d) == set(sr.LADDER_RUNGS_MEASURED)]
    random.Random(DRAW_SEED).shuffle(full)
    pick = full[:n_atoms]
    print(f"[rung-cost] {len(by_atom)} atoms in the batch, {len(full)} with all three "
          f"rungs; timing {len(pick)} paired", flush=True)

    tmp = Path(paths.scratch("near_minibrot_rung", "tiles"))
    tmp.mkdir(parents=True, exist_ok=True)
    res = {r: [] for r in sr.LADDER_RUNGS_MEASURED}
    for i, atom in enumerate(pick):
        for rung in sr.LADDER_RUNGS_MEASURED:       # interleaved: the paired design
            cre, cim, fw = by_atom[atom][rung]
            t0 = time.time()
            prescreen._render(0.0, 0.0, fw, tmp / f"t{i}_{rung:g}.jpg",
                              family="julia", c=(str(cre), str(cim)), timeout=300)
            res[rung].append(time.time() - t0)
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(pick)} atoms", flush=True)

    rec = {str(r): dict(n=len(v), mean_s=round(st.mean(v), 4),
                        median_s=round(st.median(v), 4),
                        stdev_s=round(st.stdev(v), 4) if len(v) > 1 else None,
                        total_s=round(sum(v), 2))
           for r, v in res.items()}
    base = st.mean(res[sr.LADDER_RUNGS_MEASURED[0]])
    rec["ratio_vs_rung1"] = {str(r): round(st.mean(res[r]) / base, 4)
                             for r in sr.LADDER_RUNGS_MEASURED}
    rec["geometry"] = "prescreen._render default (the harvest confirmation render)"
    rec["design"] = ("paired: every atom contributes all three rungs, interleaved, so machine "
                     "drift hits the arms equally")
    rec["work_count"] = dict(atoms=len(pick), renders=len(pick) * len(res))
    rec["measured_at"] = time.strftime("%FT%T")
    rec["population"] = BATCH_ID
    p = Path(out) if out else sr.RUNG_CHOICE_RECORD
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"[rung-cost] -> {p}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--n-atoms", type=int, default=24)
    m.add_argument("--out", default=None)
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "measure":
        rec = measure(a.n_atoms, Path(a.out) if a.out else None)
        print(json.dumps(rec, indent=2))
    print(json.dumps(sr.rung_choice(), indent=2))


if __name__ == "__main__":
    main()
