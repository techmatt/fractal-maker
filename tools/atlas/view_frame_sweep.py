#!/usr/bin/env python
r"""view_frame_sweep.py — run the deterministic framing sweep over re-scored candidates.

WHY A SWEEP AT ALL. Nucleus-centred is the UN-FRAMED case. The interior-band arc measured
framing as the killer rather than content: uniform-sampled crops averaged label 1.07
against G-framed 1.84, at every degree (`minibrot_sourcing.md` §4). The maneuvers propose a
CENTRE and a `k`; nothing in the walk ever asks whether that centre is where the picture
is. This is that missing step, run retroactively here and shaped as the post-step a future
run would call (`view_screen.sweep_best`, which takes the same measure function the walk
would).

WHAT IT DOES NOT DO. It does not re-select, re-rank the run, or write anything a future run
reads. The chosen window is recorded BESIDE the original frame, never in place of it —
every row carries the origin's composite and all 18 windows' measures.

SCOPE, STATED BECAUSE IT IS A REDUCTION. The full sweep over every candidate clearing the
veto is 17 extra fields x ~13.2k candidates = ~225k fields, ~4.2 h wall at 4 processes on
this box (measured rate 15 field/s). This runs a bounded sample instead: `--top` candidates
by composite (the ones the before/after sheet is drawn from) plus a stratified draw across
composite quintiles, so "does framing help" is answered across the range and not only at
the top. What was not swept is named in the readout, never left as a silent truncation.

  uv run python tools/atlas/view_frame_sweep.py --top 120 --n 600
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                    # noqa: E402
import view_screen as vs        # noqa: E402
import field_metrics as fm      # noqa: E402

WORKERS = 4                     # concurrent engine PROCESSES (CLAUDE.md cap)
THREADS = 1                     # per process: the field is 64x36
DEFAULT_OUT = ("view_rescreen", "sweep.jsonl")


def select(rows: list[dict], veto: float, *, top: int, n: int, seed: int) -> list[dict]:
    """`top` best by composite, then a stratified fill across composite quintiles.

    Stratified rather than top-only on purpose: a sweep measured only where the composite
    already likes the view answers "does framing help the winners", which is not the
    question. The bottom quintile is in by construction.
    """
    ok = [r for r in rows if r.get("screened") and not vs.is_vetoed(r, veto)]
    for r in ok:
        r["_c"] = vs.composite(r, veto)
    ok.sort(key=lambda r: -r["_c"])
    chosen = {r["key"]: r for r in ok[:top]}
    v = np.array([r["_c"] for r in ok])
    edges = [float(np.percentile(v, 20.0 * i)) for i in range(1, 5)]
    buckets: dict[int, list[dict]] = {q: [] for q in range(5)}
    for r in ok:
        buckets[sum(1 for e in edges if r["_c"] > e)].append(r)
    rng = random.Random(seed)
    per = max(0, (n - len(chosen)) // 5)
    for q, b in buckets.items():
        for r in rng.sample(b, min(per, len(b))):
            chosen.setdefault(r["key"], r)
    return list(chosen.values())


def run(rows: list[dict], out_path: Path, veto: float, *, workers=WORKERS, log=print):
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
        log(f"  resuming: {len(done)} already swept")
    todo = [r for r in rows if r["key"] not in done]
    log(f"  sweeping {len(todo)} views x {len(vs.SWEEP_OFFSETS)**2 * len(vs.SWEEP_SCALES)}"
        f" windows ({workers} processes)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "a", encoding="utf-8")
    lock, t0, n = threading.Lock(), time.time(), [0]
    results = []

    def work(r):
        res = vs.sweep_best(r["cx"], r["cy"], r["fw"], veto, family=r["partition"],
                            threads=THREADS)
        row = dict(key=r["key"], op=r["op"], k=r["k"], degree=r["degree"],
                   period=r["period"], cx=r["cx"], cy=r["cy"], fw=r["fw"],
                   partition=r["partition"], **res)
        with lock:
            fh.write(json.dumps(row) + "\n")
            results.append(row)
            n[0] += 1
            if n[0] % 25 == 0 or n[0] == len(todo):
                el = time.time() - t0
                log(f"  {n[0]:5d}/{len(todo)}  {n[0]/max(1e-9, el):5.2f} view/s  {el:6.0f}s"
                    f"  eta {(len(todo)-n[0])*el/max(1e-9, n[0]):6.0f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    return results


def readout(rows: list[dict], n_eligible: int, n_swept: int) -> dict:
    moved = [r for r in rows if r.get("moved")]
    gains = [r["chosen_composite"] - r["origin_composite"] for r in rows
             if r.get("chosen_composite") is not None
             and r.get("origin_composite") is not None]
    rel = [(r["chosen_composite"] + 1e-9) / (r["origin_composite"] + 1e-9) for r in rows
           if r.get("origin_composite") not in (None, 0.0)
           and r.get("chosen_composite") is not None]
    from collections import Counter
    return dict(
        eligible_after_veto=n_eligible, swept=n_swept,
        not_swept=n_eligible - n_swept,
        moved=len(moved), moved_frac=round(len(moved) / max(1, len(rows)), 4),
        chosen_scale=dict(Counter(r["chosen"]["scale"] for r in rows
                                  if r.get("chosen")).most_common()),
        chosen_offset=dict(Counter(f"({r['chosen']['dx']:+g},{r['chosen']['dy']:+g})"
                                   for r in rows if r.get("chosen")).most_common()),
        composite_gain=dict(
            median=round(float(np.median(gains)), 4) if gains else None,
            p90=round(float(np.percentile(gains, 90)), 4) if gains else None,
            max=round(float(np.max(gains)), 4) if gains else None),
        composite_ratio=dict(
            median=round(float(np.median(rel)), 4) if rel else None,
            p90=round(float(np.percentile(rel, 90)), 4) if rel else None),
        CAVEAT=("The gain is measured BY THE SAME COMPOSITE the sweep maximises, so it is "
                "an argmax over 18 draws and is biased upward by construction — read it as "
                "'how much headroom the composite sees', never as a quality improvement. "
                "The before/after sheet is the only thing here that can answer the latter."),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path,
                    default=paths.scratch("view_rescreen", "scores.jsonl"))
    ap.add_argument("--out", type=Path, default=paths.scratch(*DEFAULT_OUT))
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)
    if a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in
            a.scores.read_text(encoding="utf-8").splitlines() if l.strip()]
    fm.require_one_policy(("view scores", rows), what="the framing sweep's input")
    veto = vs.interior_veto(vs.load_refs())
    eligible = sum(1 for r in rows if r.get("screened") and not vs.is_vetoed(r, veto))
    sel = select(rows, veto, top=a.top, n=a.n, seed=a.seed)
    print(f"[sweep] {eligible} candidates clear the veto; sweeping {len(sel)} "
          f"({a.top} top-composite + stratified fill)")
    res = run(sel, a.out, veto, workers=a.workers)
    if not res:
        res = [json.loads(l) for l in
               a.out.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = readout(res, eligible, len(sel))
    (a.out.parent / "sweep_readout.json").write_text(json.dumps(rep, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
