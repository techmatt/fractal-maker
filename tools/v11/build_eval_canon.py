#!/usr/bin/env python
r"""The v11 eval slice's DEPLOY-CANONICAL tile — the one view the v11 cache does not hold.

Why this exists
---------------
v4..v10 fanned each location out as a PRODUCT (palettes x geometries x AA), so the
deploy-canonical cell — `twilight_shifted` / identity framing / `antialiased` — was present
for every location by construction, and `data_v4.Loc.canonical()` just picked it out of the
cache. v11 draws each of its 32 tiles INDEPENDENTLY (`data/v11/aug_recipe.json`), which is
what decorrelates the axes across the corpus, and the price is exactly this: the floors
guarantee >=2 `twilight_shifted` tiles and >=1 identity framing per location (tile 0 is
both), but the AA level of tile 0 is a 50/50 draw. Measured on the built cache: **5,663 of
11,303** locations carry a canonical tile, and **1,448 of the 2,860 eval locations** do.

Scoring half an eval slice through an `aliased` point-sampled tile is not an option — the
deploy path is `present.rs`'s filtered render and AA is a held axis the model is explicitly
tested for invariance under, not a nuisance to average over. So the canonical view is
PRODUCED here, once, for every eval location.

Through the cache's own code path, not a second renderer
--------------------------------------------------------
`crop-batch --replay` re-renders a tile from its RECORDED manifest row: same extended
field, same crop window, same resampler, nothing re-drawn. So the canonical tile is built by
taking each eval location's **tile-0 row** (already `twilight_shifted` + identity framing)
and overriding three recorded fields:

    aa.level      -> "antialiased"     (the 50/50 draw, forced)
    aa.mode       -> "lanczos3"
    jpg_quality   -> 85                (v4..v10's flat canonical quality)

That is the v4..v10 canonical view's construction, realized through v11's field pipeline.
It is also CHECKABLE rather than argued: the 38 eval locations whose tile 0 was ALREADY
antialiased at q85 must replay byte-identically to the cache tile, and `--verify` asserts
exactly that. A path that cannot reproduce the tiles it already has is not the path that
made them.

Storage class: bulk. 2,860 JPGs + a manifest, a deterministic function of the committed
corpus, `tools/v11/build_{manifest,plan}.py`, the cache and this module. Registered in
`artifacts.RELOCATED_PREFIXES`' class predicates BEFORE the first write (rule 5), so it is
born out-of-tree.

    uv run python tools/v11/build_eval_canon.py                 # render (resumable)
    uv run python tools/v11/build_eval_canon.py --limit 8        # bounded end-to-end
    uv run python tools/v11/build_eval_canon.py --verify         # byte-identity + coverage
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

import corpus_common as cc  # noqa: E402
import paths  # noqa: E402

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
COLORMAPS = ROOT / "data" / "v11" / "colormaps.json"

CACHE_MANIFEST = "data/v11/cache_manifest.jsonl"
EVAL_SLICE = "data/v11/eval_slice.jsonl"
CANON_DIR = "data/v11/eval_canon"
CANON_MANIFEST = "data/v11/eval_canon_manifest.jsonl"
RECORD = ROOT / "data" / "v11" / "eval_canon_record.json"

WORK = paths.scratch("v11_eval_canon")

# The deploy-canonical cell, stated once. `NEUTRAL_PALETTE` and the identity framing come
# out of the tile-0 row (asserted, not assumed); these three are what the replay overrides.
CANON_AA_LEVEL, CANON_AA_MODE, CANON_JPG_Q = "antialiased", "lanczos3", 85
CHUNK = 200          # locations per replay invocation — the resume boundary


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def eval_loc_ids() -> set[int]:
    return {r["loc_id"] for r in read_jsonl(paths.bulk(EVAL_SLICE))}


def canon_rel(loc_id: int) -> str:
    return f"{CANON_DIR}/{loc_id}.jpg"


def tile0_rows(want: set[int]) -> dict[int, dict]:
    """Each wanted location's tile-0 cache row, with the recipe floor ASSERTED.

    The floor is `--floor-palette twilight_shifted:2 --floor-identity 1` reserving the LOW
    slots, so tile 0 is the canonical palette at the identity framing. That is a property of
    the flags the cache was rendered with, i.e. of a record — so it is checked here rather
    than trusted, and a recipe change fails loudly instead of silently canonicalizing a
    shifted crop under some other palette."""
    out: dict[int, dict] = {}
    for r in read_jsonl(paths.bulk(CACHE_MANIFEST)):
        if r["tile"] != 0 or r["loc_id"] not in want:
            continue
        c = r["crop"]
        bad = [k for k, v in (("palette", r["palette"] == "twilight_shifted"),
                              ("geom", c["geom"] == 0), ("scale", c["scale"] == 1),
                              ("shift", c["shift_frac"] == 0)) if not v]
        if bad:
            raise SystemExit(f"loc {r['loc_id']} tile 0 is not the canonical framing "
                             f"({', '.join(bad)}) — the recipe floor moved; re-read "
                             f"data/v11/aug_recipe.json before canonicalizing anything")
        out[r["loc_id"]] = r
    missing = sorted(want - set(out))
    if missing:
        raise SystemExit(f"{len(missing)} eval locations have no tile-0 cache row, "
                         f"e.g. {missing[:5]} — the cache is not the one this slice indexes")
    return out


def canon_row(row: dict) -> dict:
    """One tile-0 row rewritten into its canonical-cell replay row (absolute `out`).

    `out` is made ABSOLUTE on purpose. The cache manifest stores repo-relative paths (so the
    artifact is not machine-bound), and `--replay` with no `--replay-out-root` writes to
    `out` resolved against the CWD — the exact trap `render_cache.rel_out` documents. The
    alternative, `--replay-out-root`, mirrors `<root>/<loc_id>/<basename>`, and the basename
    would spell the ORIGINAL tile's AA level and quality onto a file that has neither."""
    r = json.loads(json.dumps(row))          # deep copy; the cache row is a record
    r["aa"] = {"level": CANON_AA_LEVEL, "mode": CANON_AA_MODE}
    r["jpg_quality"] = CANON_JPG_Q
    r["out"] = paths.bulk(canon_rel(row["loc_id"])).as_posix()
    r["canonical_cell"] = "twilight_shifted / identity framing / antialiased:lanczos3 / q85"
    r["derived_from_tile"] = 0
    return r


def run_replay(chunk_path: Path, log: Path, timeout: int) -> tuple[bool, float, str]:
    """One chunk through `crop-batch --replay`. Output to a FILE, never a pipe (CLAUDE.md:
    a piped engine shows no progress at all until it exits)."""
    cmd = [str(BIN), "crop-batch", "--replay", str(chunk_path),
           "--colormaps", str(COLORMAPS), "--log-every", "100000"]
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        try:
            pr = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                                cwd=str(ROOT), env=cc.default_engine_env(),
                                creationflags=cc.default_creationflags())
        except subprocess.TimeoutExpired:
            return False, time.time() - t0, f"TIMEOUT after {timeout}s (see {log})"
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace")
                     .strip().splitlines()[-3:])
    return pr.returncode == 0, time.time() - t0, tail


def build(limit: int | None, timeout: int, force: bool) -> int:
    want = eval_loc_ids()
    rows = tile0_rows(want)
    ids = sorted(rows)
    if limit:
        ids = ids[:limit]
    todo = [i for i in ids if force or not paths.bulk(canon_rel(i)).exists()]
    print(f"eval locations {len(ids)}  already rendered {len(ids) - len(todo)}  "
          f"to render {len(todo)}")
    WORK.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    done = 0
    for c0 in range(0, len(todo), CHUNK):
        part = todo[c0:c0 + CHUNK]
        ci = c0 // CHUNK
        cf = WORK / f"replay{ci:03d}.jsonl"
        cf.write_text("\n".join(json.dumps(canon_row(rows[i])) for i in part) + "\n",
                      encoding="utf-8")
        ok, secs, tail = run_replay(cf, WORK / f"replay{ci:03d}.log", timeout)
        done += len(part)
        rate = done / max(time.time() - t_start, 1e-9)
        left = (len(todo) - done) / rate if rate > 0 else 0
        print(f"  chunk {ci:03d}: {len(part)} loc in {secs:6.1f}s  "
              f"{'ok' if ok else 'FAIL'}  {rate:.2f} loc/s  ~{left/60:.1f} min left",
              flush=True)
        if not ok:
            print(tail)
            return 1
    write_manifest(ids, rows, incomplete=bool(limit))
    print(f"wrote {CANON_MANIFEST} ({len(ids)} rows)  wall {time.time()-t_start:.0f}s")
    return 0


def write_manifest(ids, rows, *, incomplete: bool) -> None:
    """The durable-shaped index: one row per eval location, repo-relative `path`.

    `incomplete` is DERIVED from the flag at the write site, never hardcoded — a bounded run
    that writes real files must stamp itself unusable (CLAUDE.md, "give a long path a
    bounded end-to-end")."""
    out = paths.bulk(CANON_MANIFEST)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in ids:
            r = rows[i]
            f.write(json.dumps({
                "loc_id": i, "path": canon_rel(i), "palette": r["palette"],
                "aa_level": CANON_AA_LEVEL, "aa_mode": CANON_AA_MODE,
                "jpg_quality": CANON_JPG_Q, "scale": 1.0, "shift_id": "center",
                "geom": 0, "maxiter": r["render"]["maxiter"],
                "maxiter_policy": r["render"]["maxiter_policy"],
                "field_ss": r["field"]["field_ss"],
                "derived_from": {"cache_manifest": CACHE_MANIFEST, "tile": 0},
                "batch_incomplete": incomplete,
            }) + "\n")
    RECORD.write_text(json.dumps({
        "build": "v11-eval-canon",
        "what": ("the deploy-canonical view (twilight_shifted / identity framing / "
                 "antialiased:lanczos3 / q85) for every v11 eval location — the one cell "
                 "the independent-draw cache does not guarantee"),
        "rebuild": "uv run python tools/v11/build_eval_canon.py",
        "verify": "uv run python tools/v11/build_eval_canon.py --verify",
        "path": {"tiles": CANON_DIR, "manifest": CANON_MANIFEST, "class": "bulk"},
        "rows": len(ids),
        "produced_by": "crop-batch --replay off the cache's own tile-0 rows",
        "overrides": {"aa.level": CANON_AA_LEVEL, "aa.mode": CANON_AA_MODE,
                      "jpg_quality": CANON_JPG_Q},
        "batch_incomplete": incomplete,
    }, indent=2), encoding="utf-8")


def verify() -> int:
    """Coverage + the byte-identity check that makes the replay claim checkable."""
    want = eval_loc_ids()
    rows = tile0_rows(want)
    missing = [i for i in sorted(rows) if not paths.bulk(canon_rel(i)).exists()]
    already = [i for i, r in rows.items()
               if r["aa"]["level"] == CANON_AA_LEVEL and r["jpg_quality"] == CANON_JPG_Q]
    diff = []
    for i in sorted(already):
        a = paths.bulk(canon_rel(i))
        b = paths.bulk(rows[i]["out"])
        if not a.exists() or not b.exists() or a.read_bytes() != b.read_bytes():
            diff.append(i)
    rec = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    ok = not missing and not diff and already and not rec.get("batch_incomplete", True)
    print(f"eval locations           : {len(rows)}")
    print(f"tiles present            : {len(rows) - len(missing)}  missing {len(missing)}")
    print(f"byte-identity population : {len(already)} (tile 0 already antialiased@q{CANON_JPG_Q})")
    print(f"  byte-identical         : {len(already) - len(diff)}  differ {len(diff)}")
    print(f"record batch_incomplete  : {rec.get('batch_incomplete')}")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    if missing:
        print(f"  missing e.g. {missing[:5]}")
    if diff:
        print(f"  differ e.g. {diff[:5]}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="run the WHOLE path on the first N eval locations; every row it "
                         "writes is stamped batch_incomplete")
    ap.add_argument("--timeout", type=int, default=1800, help="per-chunk wall cap (s)")
    ap.add_argument("--force", action="store_true", help="re-render existing tiles")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return verify()
    if not BIN.exists():
        raise SystemExit(f"{BIN} missing — cargo build --release")
    return build(a.limit, a.timeout, a.force)


if __name__ == "__main__":
    raise SystemExit(main())
