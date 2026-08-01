#!/usr/bin/env python
r"""view_rescreen.py — re-score a maneuver run's candidates under the VIEW-level screen.

The dry run scored every candidate on its ATOM (one 64x36 field at `4 * window_scale`,
shared across every `k` row). This re-scores each candidate on **the view it actually
pushed** — `cx/cy/fw` as recorded — with `tools/atlas/view_screen.py`. Same field source,
same screening geometry, same stamped cap policy; different frame, plus the two
composition measures the atom screen has no way to see.

RETROACTIVE ONLY. Nothing here writes to a run directory, changes selection, or feeds a
future run: it reads `maneuvers.jsonl` and writes an analysis file. The population is the
same one `maneuver_inspection_sheet.load_population` builds (available + screened,
deduped on `(atom_key, k)`, pushed OR passed over), reused rather than re-derived so the
two readouts describe the same 16,440 rows.

Output is `scratch/` by class: it is a deterministic function of a durable input
(`data/discovery/<run>/maneuvers.jsonl`) plus committed code, rebuildable by re-running
this file. Resumable — rows are appended as they land and a re-run skips what is done.

  uv run python tools/atlas/view_rescreen.py --run-dir data/discovery/maneuver_v14_exploration
  uv run python tools/atlas/view_rescreen.py --run-dir ... --limit 400 --out <p>   # sample
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
for _p in (HERE, ROOT / "tools", ROOT / "tools" / "orbital"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                # noqa: E402
import view_screen as vs                    # noqa: E402
import field_metrics as fm                  # noqa: E402
import maneuver_inspection_sheet as mis     # noqa: E402

# One engine PROCESS per worker, so this is the CLAUDE.md concurrent-PROCESS cap. Each
# child is pinned to one thread: the field is 64x36, so the cost is the spawn, not compute.
WORKERS = 4
THREADS = 1

DEFAULT_OUT = "view_rescreen/scores.jsonl"

# Columns carried through from the maneuver record so the analysis needs only this file.
CARRY = ("run", "op", "k", "cx", "cy", "fw", "partition", "degree", "period", "atom_key",
         "window_scale", "log10_abs_A", "parent_depth", "used", "unused_reason")
# ...and the ATOM-frame screen's own numbers, kept under an `atom_` prefix so an old and a
# new measure can never be confused for one another in a downstream read.
CARRY_ATOM = ("radial_range", "radial_rings", "interior_fraction")


def row_key(r: dict) -> str:
    return f"{r.get('atom_key')}|{r.get('k')}"


def load_done(out_path: Path) -> dict[str, dict]:
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def rescreen(pop: list[dict], out_path: Path, *, workers=WORKERS, log=print) -> list[dict]:
    done = load_done(out_path)
    if done:
        log(f"  resuming: {len(done)} rows already measured")
        # Same guard the atom screen carries on its resume: rows on disk were measured
        # under whatever cap policy was live when they were written, and appending across
        # a policy change would write one file holding two incommensurable populations.
        fm.require_one_policy(("resumed", list(done.values())),
                              ("this run", [{fm.POLICY_KEY: vs.ms.screen_policy_token()}]),
                              what="resumed view-screen scores against this run's policy")
    todo = [r for r in pop if row_key(r) not in done]
    log(f"  measuring {len(todo)} views at {fm.SCREEN_W}x{fm.SCREEN_H} "
        f"({workers} processes x {THREADS} thread)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "a", encoding="utf-8")
    lock = threading.Lock()
    t0, n = time.time(), [0]

    def work(r):
        m = vs.measure_view(r["cx"], r["cy"], r["fw"], family=r["partition"],
                            threads=THREADS)
        row = dict(key=row_key(r),
                   **{k: r.get(k) for k in CARRY},
                   **{f"atom_{k}": r.get(k) for k in CARRY_ATOM},
                   **m)
        with lock:
            fh.write(json.dumps(row) + "\n")
            done[row["key"]] = row
            n[0] += 1
            if n[0] % 250 == 0 or n[0] == len(todo):
                el = time.time() - t0
                # Rate from RECENT throughput would need a window; this pass is uniform
                # per row (one 64x36 field each), so the running mean is the right
                # projector here and the ETA is quoted as such.
                log(f"  {n[0]:6d}/{len(todo)}  {n[0]/max(1e-9, el):5.1f} view/s  "
                    f"{el:6.0f}s  eta {(len(todo)-n[0])/max(1e-9, n[0]/max(1e-9, el)):6.0f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    log(f"  done in {time.time()-t0:.0f}s")
    return [done[row_key(r)] for r in pop if row_key(r) in done]


# --------------------------------------------------------------------------- #
def summarize(rows: list[dict], veto: float) -> dict:
    ok = [r for r in rows if r.get("screened")]
    comp = np.array([vs.composite(r, veto) for r in ok], dtype=float)
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    # Every unscreened row is counted under a named reason class — never characterize a
    # failure population from a truncated sample (`CLAUDE.md`, four rules).
    from collections import Counter
    reasons = Counter((r.get("screen_reason") or "unknown").split(":")[0]
                      for r in rows if not r.get("screened"))
    out = dict(
        n=len(rows), screened=len(ok), unscreened=len(rows) - len(ok),
        unscreened_reasons=dict(reasons.most_common()),
        interior_veto=veto,
        vetoed=int(sum(1 for r in ok if vs.is_vetoed(r, veto))),
        composite=({f"p{p}": round(float(np.percentile(comp, p)), 4) for p in pct}
                   if len(comp) else {}),
    )
    for k in ("band_coverage", "interior_fraction", "radial_range", "radial_rings"):
        v = np.array([r[k] for r in ok], dtype=float)
        out[k] = {f"p{p}": round(float(np.percentile(v, p)), 4) for p in pct} if v.size else {}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="measure a SEEDED RANDOM SUBSET of this size (a projection "
                         "sample, not a prefix — the log is written in run order)")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)
    if a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2
    out_path = a.out or paths.scratch(*DEFAULT_OUT.split("/"))

    pop = mis.load_population([Path(d) / "maneuvers.jsonl" for d in a.run_dir])
    print(f"[pop] {len(pop)} available+screened maneuver candidates")
    if a.limit and a.limit < len(pop):
        pop = random.Random(a.seed).sample(pop, a.limit)
        print(f"[pop] sampling {len(pop)} (seed {a.seed})")

    rows = rescreen(pop, out_path, workers=a.workers)
    veto = vs.interior_veto(vs.load_refs())
    rep = summarize(rows, veto)
    print(json.dumps(rep, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
