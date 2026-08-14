#!/usr/bin/env python
r"""release_sweep.py — measure the release render pass at N worker processes, and prove parity.

WHY IT IS A TOOL AND NOT A SCRATCH SCRIPT. `release_pass.DEFAULT_RELEASE_WORKERS` and
`ENGINE_THREADS_PER_WORKER` are production constants whose only justification is a measurement; the thing
that reproduces that measurement is therefore load-bearing for them and cannot live in a tree
that `rm -r scratch/*` empties (CLAUDE.md, "neither scratch tree is a dependency tier"). The
table it produced, with its date, is `docs/design/release_concurrency.md`.

WHAT IT MEASURES. One arm = the whole release pass over the same plan at a given
(workers x per-engine-threads), into its own output dir, wall-clocked, with peak RSS sampled
over THIS process tree (parent + workers + their `fractal-generator.exe` children) — the
engines are grandchildren and are most of the memory, so a parent-only reading measures the
wrong thing.

THE PLAN IS A REAL RELEASE, not a synthetic one. Rows come from the durable release record
(`data/emission/release_records/emission_diversity_v1.jsonl`), and each row's Location is
rebuilt from the discovery ledger it came from rather than from the record's own float
`location` block — that block has no `family_params`, so a phoenix row rebuilt from it is a
DIFFERENT location. The default run's 12 selected rows cover all three render paths (smooth /
pure-field / composite) and include phoenix and julia, which is what the mix has to contain for
the wall clock to mean anything: the paths do not cost the same and are not RAM-alike.

PARITY IS THE POINT, not a footnote. `--check` hashes every PNG and the `autolevel_stamps.jsonl`
of every arm against the first arm and fails loudly on any difference. Concurrency that is 3x
faster and 1 pixel different is not an adoption.

  uv run python tools/emission/release_sweep.py --arms 1,2,3,4 \
      --out scratch/release_concurrency --check          # the full sweep (background it)
  uv run python tools/emission/release_sweep.py --arms 1 --limit 2 --w 640 --h 360 --ss 2 \
      --out scratch/release_smoke                        # the bounded end-to-end
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import paths                                       # noqa: E402
from tools.emission import descriptor as D                    # noqa: E402
from tools.emission import release_pass as RP                 # noqa: E402
from tools.palettes import autolevel as AL                    # noqa: E402

RECORD = ROOT / "data" / "emission" / "release_records" / "emission_diversity_v1.jsonl"
LEDGER_GLOB = "data/discovery/*/outcome_ledger.jsonl"


# --------------------------------------------------------------------------- #
# Plan: released rows -> ReleaseTask, with the Location from its own ledger.
# --------------------------------------------------------------------------- #
def _ledgers_by_namespace() -> dict:
    return {D.ledger_namespace(p): p for p in sorted(ROOT.glob(LEDGER_GLOB))}


def build_plan(run_id: str, limit: int | None, out_dir: Path) -> list:
    rows = [json.loads(l) for l in RECORD.read_text(encoding="utf-8").splitlines() if l.strip()]
    rel = [r for r in rows
           if r.get("run_id") == run_id and r.get("stage") == "release"
           and r.get("decision") == "selected"]
    if not rel:
        raise SystemExit(f"no selected release rows for run_id={run_id!r} in {RECORD}")
    ns_map = _ledgers_by_namespace()
    cache: dict = {}
    tasks = []
    for r in rel[:limit] if limit else rel:
        lid = r["location_id"]
        ns = next((n for n in ns_map if lid.startswith(n + D.NS_SEP)), None)
        if ns is None:
            raise SystemExit(f"no ledger namespace matches {lid!r} (have {sorted(ns_map)})")
        row_id = lid[len(ns) + len(D.NS_SEP):]
        if ns not in cache:
            cache[ns] = {x["id"]: x for x in
                         (json.loads(l) for l in
                          ns_map[ns].read_text(encoding="utf-8").splitlines() if l.strip())}
        src = cache[ns].get(row_id)
        if src is None:
            raise SystemExit(f"{row_id!r} not in {ns_map[ns]}")
        tasks.append(RP.ReleaseTask(id=r["key"].replace("|", "_").replace("/", "_"),
                                    loc=D.location_of(src), style=r["render_style"],
                                    palette=r["palette"], out=str(out_dir / f"{_short(r)}.png")))
    return tasks


def _short(r: dict) -> str:
    """A short, collision-free file stem: the row's own key hashed, plus a readable tail."""
    h = hashlib.sha1(r["key"].encode()).hexdigest()[:10]
    return f"{h}_{r['partition'].replace(':', '-')}_{r['render_style']}"


# --------------------------------------------------------------------------- #
# Peak RSS over this process TREE (parent + workers + engine grandchildren).
# --------------------------------------------------------------------------- #
_PS = ("Get-CimInstance Win32_Process | "
       "Select-Object ProcessId,ParentProcessId,Name,WorkingSetSize | "
       "ConvertTo-Csv -NoTypeInformation")


def _tree_rss_mb(root_pid: int) -> tuple:
    """(total MB over the tree, per-name MB). Returns (0, {}) if the snapshot fails — a
    sampler that cannot read must not take the measurement down, but the caller reports the
    miss count so a silent zero can never be read as "no memory used"."""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return 0.0, {}
    kids, info = {}, {}
    for row in csv.DictReader(io.StringIO(r.stdout)):
        try:
            pid, ppid = int(row["ProcessId"]), int(row["ParentProcessId"])
        except (TypeError, ValueError):
            continue
        info[pid] = (row.get("Name") or "?", float(row.get("WorkingSetSize") or 0))
        kids.setdefault(ppid, []).append(pid)
    seen, stack, by_name, tot = set(), [root_pid], {}, 0.0
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in info:
            continue
        seen.add(pid)
        name, ws = info[pid]
        tot += ws
        by_name[name] = by_name.get(name, 0.0) + ws
        stack.extend(kids.get(pid, ()))
    return tot / 1e6, {k: round(v / 1e6, 1) for k, v in by_name.items()}


class RssSampler:
    def __init__(self, interval=3.0):
        self.interval, self.peak, self.peak_by_name = interval, 0.0, {}
        self.n, self.misses = 0, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            tot, by_name = _tree_rss_mb(os.getpid())
            self.n += 1
            if tot <= 0:
                self.misses += 1
            elif tot > self.peak:
                self.peak, self.peak_by_name = tot, by_name
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=10)


# --------------------------------------------------------------------------- #
# One arm.
# --------------------------------------------------------------------------- #
def run_arm(plan, geom, workers: int, threads, out_dir: Path, sample_rss: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():                    # an arm always renders, never resumes
        f.unlink()
    tasks = [RP.ReleaseTask(t.id, t.loc, t.style, t.palette,
                            str(out_dir / Path(t.out).name)) for t in plan]
    prior = RP.ENGINE_THREADS_PER_WORKER
    if threads is not None:
        RP.ENGINE_THREADS_PER_WORKER = int(threads)   # the arm's point on the (w x t) surface
    per_row, fails = [], []

    def sink(task, res):
        if res.stamp is not None:
            AL.append_stamp(out_dir, Path(task.out).name, res.stamp)
        if not res.ok:
            fails.append({"id": task.id, "error": res.error})
        per_row.append({"id": task.id, "style": task.style, "dur_s": round(res.dur_s, 2),
                        "ok": res.ok})

    sampler = RssSampler() if sample_rss else None
    t0 = time.time()
    try:
        if sampler:
            with sampler:
                stat = RP.run_pass(tasks, geom, workers=workers, sink=sink,
                                   log=lambda m: print(m, flush=True))
        else:
            stat = RP.run_pass(tasks, geom, workers=workers, sink=sink,
                               log=lambda m: print(m, flush=True))
    finally:
        RP.ENGINE_THREADS_PER_WORKER = prior      # an arm's override never leaks to the next
    wall = time.time() - t0
    return {"workers": workers, "engine_threads": stat["engine_threads"], "wall_s": round(wall, 1),
            "n": len(tasks), "failures": fails, "fell_back_serial": stat["fell_back_serial"],
            "peak_rss_mb": round(sampler.peak, 0) if sampler else None,
            "peak_rss_by_name": sampler.peak_by_name if sampler else None,
            "rss_samples": (sampler.n, sampler.misses) if sampler else None,
            "sum_row_s": round(sum(r["dur_s"] for r in per_row), 1), "rows": per_row,
            "out_dir": str(out_dir)}


def digest(out_dir: Path) -> dict:
    """sha256 of every product in an arm — the PNGs and the stamp log. The stamp log is in
    here on purpose: identical pixels with a differently-ordered record is still a difference."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out_dir.iterdir()) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="1,2,3,4",
                    help="worker counts, comma-separated; `N:T` pins that arm's per-engine "
                         "threads (e.g. `4:2`)")
    ap.add_argument("--run-id", default="prod27")
    ap.add_argument("--limit", type=int, default=None, help="first N rows of the release")
    # Through `paths.scratch()` rather than a literal: this tool's output tree is its own
    # disposable one, and the write gate is where that gets declared.
    ap.add_argument("--out", default=str(paths.scratch("release_concurrency")))
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--ss", type=int, default=None)
    ap.add_argument("--filt", default=None)
    ap.add_argument("--check", action="store_true",
                    help="hash every arm's products against the first arm's and FAIL on a diff")
    ap.add_argument("--no-rss", action="store_true", help="skip the memory sampler")
    args = ap.parse_args()

    import corpus_common as cc
    print(f"[priority] {cc.set_below_normal_priority()}", flush=True)
    from tools.emission import build_emission_diversity_v1 as BED
    geom = RP.Geom(args.w or BED.REL_W, args.h or BED.REL_H,
                   args.ss or BED.REL_SS, args.filt or BED.REL_FILT)
    out = Path(args.out).resolve()
    plan = build_plan(args.run_id, args.limit, out / "_plan")
    print(f"[sweep] {len(plan)} row(s) from {args.run_id} at {geom.w}x{geom.h} ss{geom.ss} "
          f"{geom.filt}; styles " +
          ", ".join(f"{s}x{sum(1 for t in plan if t.style == s)}"
                    for s in sorted({t.style for t in plan})), flush=True)

    arms, results, digests = [], [], {}
    for spec in args.arms.split(","):
        w, _, t = spec.strip().partition(":")
        arms.append((int(w), int(t) if t else None))
    for w, t in arms:
        tag = f"w{w}" + (f"t{t}" if t else "")
        print(f"\n[sweep] === arm {tag} ===", flush=True)
        res = run_arm(plan, geom, w, t, out / tag, sample_rss=not args.no_rss)
        res["arm"] = tag
        results.append(res)
        digests[tag] = digest(out / tag)
        base = results[0]
        res["speedup"] = round(base["wall_s"] / res["wall_s"], 2) if res["wall_s"] else None
        print(f"[sweep] {tag}: wall {res['wall_s']}s  speedup {res['speedup']}x  "
              f"peak RSS {res['peak_rss_mb']} MB  failures {len(res['failures'])}", flush=True)
        (out / "sweep.json").write_text(json.dumps(
            {"geom": vars(geom), "run_id": args.run_id, "arms": results,
             "digests": digests}, indent=2), encoding="utf-8")

    print("\n| arm | workers | engine threads | wall s | speedup | peak RSS MB | failures |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['arm']} | {r['workers']} | {r['engine_threads'] or 'default'} | "
              f"{r['wall_s']} | {r['speedup']}x | {r['peak_rss_mb']} | {len(r['failures'])} |")

    rc = 0
    if args.check and len(results) > 1:
        base_tag = results[0]["arm"]
        for r in results[1:]:
            same = digests[r["arm"]] == digests[base_tag]
            print(f"[parity] {r['arm']} vs {base_tag}: "
                  f"{'IDENTICAL' if same else 'DIFFERENT'} "
                  f"({len(digests[r['arm']])} product(s))", flush=True)
            if not same:
                rc = 1
                for k in sorted(set(digests[base_tag]) | set(digests[r["arm"]])):
                    a, b = digests[base_tag].get(k), digests[r["arm"]].get(k)
                    if a != b:
                        print(f"  DIFF {k}: {a} != {b}", flush=True)
    print(f"[sweep] {out / 'sweep.json'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
