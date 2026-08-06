"""Rebuild a wallpaper batch's label crops from its committed `images.jsonl`.

The wallpaper batch crops are BULK-regenerable (size_guard: `data/wallpaper_corpus/`
-> ARTIFACTS): the crop is a pure function of the row's `render` block + its
`provenance.params` colormap recipe through the locked label-crop pins, so the only
thing that has to survive is the row. Twice now that regenerability has been cashed
in — the July batches' crops were deleted, and the v3 held-out eval was revived by
re-rendering its 686 rows (`scratch/wallpaper_eval_revival/`, whose ladder reproduced
the deployed gate's recorded precision to 2 dp on all three rungs). This module is
that one-off generalized to whole batches so the next rebuild is a command, not a
script.

Generalizes `rerender_bootstrap_ss2.py` (which re-rendered ONE batch to change its
ss level) and `scratch/wallpaper_eval_revival/render_eval_crops.py` (which rendered a
ROW SUBSET to a side directory). Here: named batches, crops written back into
`<batch>/crops/` where the trainer reads them, existing crops skipped.

    uv run python tools/wallpaper/rerender_batch_crops.py --list
    uv run python tools/wallpaper/rerender_batch_crops.py july --limit 2   # bounded e2e
    uv run python tools/wallpaper/rerender_batch_crops.py july

`--limit` bounds the WRITING path (CLAUDE.md, "give a long path a bounded
end-to-end"), and because a bounded run leaves a partially-populated crop dir that
looks complete, it is the caller's job to re-run without it; `--verify` reports
per-batch coverage against `images.jsonl` and is what the trainer's precondition
should be read off.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools"))

import query_sampler as qs            # noqa: E402  (load_pool_library)
import colormap as cm                 # noqa: E402  (CandidateConfig, stretch_field)
import location as loc_mod            # noqa: E402  (from_render_block, to_location_ref)
import corpus_common as cc            # noqa: E402  (engine thread/priority defaults)
from label_crop import (              # noqa: E402
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q,
    ensure_label_field, render_label_crop,
)

BATCHES = ROOT / "data" / "wallpaper_corpus" / "batches"
# Field cache: one ~50 MB .bin per location, deleted after that location's crops
# unless --keep-fields. Disposable by construction (pure function of loc+geometry).
FIELDS_DIR = ROOT / "scratch" / "wallpaper_rerender_fields"
LABEL_CROP_WORKERS = 4   # in-process threads over ONE shared field — not processes


@dataclass(frozen=True)
class BatchSpec:
    """One rebuildable batch. Frozen from the start even though the July set is the
    only caller today — `sitting_cutter`'s module-scope constants are the counterexample."""
    batch_id: str
    note: str

    @property
    def dir(self) -> Path:
        return BATCHES / self.batch_id

    @property
    def crops(self) -> Path:
        return self.dir / "crops"

    def rows(self) -> list[dict]:
        return [json.loads(l) for l in
                (self.dir / "images.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


# The three July batches: labels survived in labels/wallpaper_*.json, crops did not.
JULY = [
    BatchSpec("2026-07-05_wallpaper_bootstrap_v1", "504 renders / 63 loc (ss2 rerender era)"),
    BatchSpec("2026-07-05_wallpaper_humanq3_v1", "994 renders / 142 loc"),
    BatchSpec("2026-07-09_wallpaper_headbatch_dramatic_v1", "1000 renders / 136 loc"),
]
# The two 2026-08-05 batches ship their crops in-tree; listed so --verify covers the
# whole training union, not just the half that has been lost once.
FRESH = [
    BatchSpec("2026-08-05_wallpaper_fresh_sheet_v1", "960 renders / 240 loc"),
    BatchSpec("2026-08-05_wallpaper_colorize_path_v1", "180 renders / 180 loc"),
]
GROUPS = {"july": JULY, "fresh": FRESH, "all": JULY + FRESH}


def config_from_row(row) -> "cm.CandidateConfig":
    """Manifest row -> label-crop CandidateConfig.

    The `transfer`/`transfer_gamma` defaults matter: bootstrap and humanq3 predate
    those params and their rows have neither, so a missing key means the pre-transfer
    behaviour (pct, gamma 0), not an error."""
    p = row["provenance"]["params"]
    loc = loc_mod.from_render_block(row["render"])
    return cm.CandidateConfig(
        palette=p["palette"],
        location=loc_mod.to_location_ref(loc),
        eval_width=LABEL_W, eval_height=LABEL_H,
        reverse=p["reverse"],
        log_premap=p["log_premap"],
        gamma=p["gamma"],
        phase=p["phase"],
        n_cycles=p["n_cycles"],
        transfer=p.get("transfer", "pct"),
        transfer_gamma=float(p.get("transfer_gamma", 0.0)),
        interior_color=tuple(p["interior_color"]),
        filter=LABEL_FILTER,
    )


def coverage(spec: BatchSpec) -> tuple[int, int, list[str]]:
    """(n_rows, n_on_disk, missing_image_ids)."""
    rows = spec.rows()
    missing = [r["image_id"] for r in rows
               if not (spec.crops / f"{r['image_id']}.jpg").exists()]
    return len(rows), len(rows) - len(missing), missing


def render_batch(spec: BatchSpec, args, lib, fail_fh) -> tuple[int, int]:
    """Render every missing crop of one batch. Returns (n_done, n_failed)."""
    rows = spec.rows()
    spec.crops.mkdir(parents=True, exist_ok=True)

    # ONE field dump per canonical location — a location's picks share the field, and
    # the field is the expensive half (a Rust pass) while coloring is numpy.
    by_loc: "OrderedDict[str, tuple]" = OrderedDict()
    for r in rows:
        loc = loc_mod.from_render_block(r["render"])
        by_loc.setdefault(loc.key(), (loc, []))[1].append(r)

    todo = [r for r in rows if not (spec.crops / f"{r['image_id']}.jpg").exists()]
    items = [(k, v) for k, v in by_loc.items()
             if any(not (spec.crops / f"{r['image_id']}.jpg").exists() for r in v[1])]
    print(f"[{spec.batch_id}] {len(rows)} rows / {len(by_loc)} loc; "
          f"on disk {len(rows)-len(todo)}, to render {len(todo)} over {len(items)} loc",
          flush=True)
    if args.limit:
        items = items[:args.limit]
        print(f"[{spec.batch_id}] --limit {args.limit}: {len(items)} locations "
              f"(BOUNDED — crop dir will be incomplete)", flush=True)
    if not items:
        return 0, 0

    t_wall = time.time()
    done = n_fail = 0
    for li, (key, (loc, grp)) in enumerate(items):
        t_loc = time.time()
        try:
            field = ensure_label_field(loc, fields_dir=FIELDS_DIR)
            prep = cm.stretch_field(field)
        except Exception as e:                                     # noqa: BLE001
            for r in grp:
                fail_fh.write(json.dumps({"batch": spec.batch_id, "image_id": r["image_id"],
                                          "stage": "field", "loc_key": key,
                                          "error": f"{type(e).__name__}: {e}"}) + "\n")
            fail_fh.flush()
            n_fail += len(grp)
            print(f"[{spec.batch_id}] loc {li+1}/{len(items)} FIELD FAILED "
                  f"({len(grp)} crops): {e}", flush=True)
            continue
        t_field = time.time() - t_loc

        def _one(r):
            try:
                w, h = render_label_crop(field, config_from_row(r), lib,
                                         spec.crops / f"{r['image_id']}.jpg", prep=prep)
                if (w, h) != (LABEL_W, LABEL_H):
                    raise AssertionError(f"bad crop size {(w, h)}")
                return (r["image_id"], None)
            except Exception as e:                                 # noqa: BLE001
                return (r["image_id"],
                        f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")

        pending = [r for r in grp if not (spec.crops / f"{r['image_id']}.jpg").exists()]
        with ThreadPoolExecutor(max_workers=min(LABEL_CROP_WORKERS, len(pending))) as ex:
            for iid, err in ex.map(_one, pending):
                if err is None:
                    done += 1
                else:
                    n_fail += 1
                    fail_fh.write(json.dumps({"batch": spec.batch_id, "image_id": iid,
                                              "stage": "crop", "loc_key": key,
                                              "error": err}) + "\n")
                    fail_fh.flush()

        if not args.keep_fields:
            # *.bin/*.json only — the run's own lock lives here too and must survive.
            for p in list(FIELDS_DIR.glob("*.bin")) + list(FIELDS_DIR.glob("*.json")):
                p.unlink(missing_ok=True)

        el = time.time() - t_wall
        rate = done / el if el else 0.0
        # ETA off THIS run's observed throughput, refit every location — not a
        # pre-run per-crop projection (CLAUDE.md, "projecting a long run's wall clock").
        eta = (len(todo) - done - n_fail) / rate / 60 if rate else float("nan")
        print(f"[{spec.batch_id}] loc {li+1}/{len(items)} {loc.family:18} fw={loc.fw[:10]} "
              f"{len(pending):2d} crops  field {t_field:5.1f}s  loc {time.time()-t_loc:6.1f}s  "
              f"({done}+{n_fail}/{len(todo)})  {rate*60:.1f} crops/min  ETA {eta:.0f} min",
              flush=True)
    return done, n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("group", nargs="?", default="july", choices=sorted(GROUPS),
                    help="which batch set to operate on (default: july)")
    ap.add_argument("--list", action="store_true", help="print the group's batches and exit")
    ap.add_argument("--verify", action="store_true",
                    help="report crop coverage vs images.jsonl and exit non-zero if incomplete")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap LOCATIONS per batch (bounded end-to-end; leaves crops incomplete)")
    ap.add_argument("--keep-fields", action="store_true",
                    help="keep the dumped fields (default: delete after each location)")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    specs = GROUPS[args.group]
    if args.list:
        for s in specs:
            print(f"{s.batch_id}  — {s.note}")
        return 0
    if args.verify:
        bad = 0
        for s in specs:
            n, on_disk, missing = coverage(s)
            flag = "OK " if not missing else "GAP"
            print(f"[{flag}] {s.batch_id}: {on_disk}/{n} crops on disk"
                  + (f"  missing e.g. {missing[:3]}" if missing else ""))
            bad += len(missing)
        print(f"total missing: {bad}")
        return 1 if bad else 0

    # SINGLE INSTANCE. Two concurrent runs share FIELDS_DIR and each wipes it after
    # every location — so one deletes the ~50 MB field the other is mid-read on, and
    # they duplicate every render besides. Observed: a stray detached run overlapped a
    # foreground one and interleaved two different row totals into one log.
    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    lock = FIELDS_DIR / "rerender.lock"
    try:
        lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[rerender] REFUSING: {lock} exists — another run holds the field cache.\n"
              f"           If no rerender process is alive, delete the lock and retry.",
              file=sys.stderr)
        return 2
    os.write(lock_fd, f"pid={os.getpid()} group={args.group}\n".encode())
    os.close(lock_fd)

    failures = ROOT / "scratch" / "wallpaper_rerender_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)

    # 7 rayon threads at BELOW_NORMAL, inherited by every render-one child.
    os.environ.update(cc.default_engine_env())
    print(f"[rerender] group={args.group}  {LABEL_W}x{LABEL_H} ss{LABEL_SS} "
          f"{LABEL_FILTER} q{JPG_Q}  priority={cc.set_below_normal_priority()}  "
          f"RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']}", flush=True)

    lib = qs.load_pool_library()
    t0 = time.time()
    tot_done = tot_fail = 0
    try:
        with open(failures, "a", encoding="utf-8") as fh:
            for spec in specs:
                d, f = render_batch(spec, args, lib, fh)
                tot_done += d
                tot_fail += f
    finally:
        lock.unlink(missing_ok=True)
    print(f"\n[rerender] {tot_done} rendered, {tot_fail} failed in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    for spec in specs:
        n, on_disk, missing = coverage(spec)
        print(f"[coverage] {spec.batch_id}: {on_disk}/{n}"
              + (f"  MISSING {len(missing)}" if missing else ""), flush=True)
    return 1 if tot_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
