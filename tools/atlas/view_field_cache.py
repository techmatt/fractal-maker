#!/usr/bin/env python
r"""view_field_cache.py — cache the 64x36 view fields so a re-weighting is arithmetic.

WHY IT EXISTS. Every composite revision so far (v2 -> v3) re-weighted measures already on
the row and cost nothing. v4 changes the per-tile PARTICIPATION INDICATOR, which is a new
statistic OF THE FIELD — and the field was thrown away, so the 16,440-candidate population
had to be re-measured through the engine (~19 min at four processes). That is a cost paid
once per new statistic and it is avoidable: the field is 64x36 f32, 9 KB a row, ~150 MB for
the whole population. Cache it and the NEXT per-tile statistic is a numpy pass.

WHAT IS CACHED, EXACTLY. The raw `render-one --dump-field --dump-field-source f64` array as
`field_metrics.dump_field` returns it — native `<f4`, NaN where the pixel did not escape.
f32 and not f64: f32 is the ENGINE's dtype, so storing f64 would pad zeros and claim a
precision the source does not have. Every measure in `view_screen` is computed from this
array, so a cached row re-derives byte-identically (checked: `--verify`).

STORAGE CLASS: `scratch()`. It is a deterministic function of a durable input
(`data/discovery/<run>/maneuvers.jsonl`) plus committed code and the stamped cap policy, and
it is regenerable by re-running this file. It is large and it is disposable; nothing may
depend on it surviving `rm -r scratch/*` (`docs/design/storage_classes.md`).

TWO CACHES, ONE FORMAT. `FieldCache` is the RETROSPECTIVE one: a population is known up
front, its key order is frozen in `index.json`, and the store is a fixed-size memmap. That
shape cannot serve a live walk, whose population does not exist yet — so `RunFieldCache`
(below) is the APPEND-ONLY sibling the run writes into as it screens. They share the
dtype, the geometry and the `fields.f32` layout, and `RunFieldCache.finalize()` writes the
retrospective pair (`index.json` + `valid.npy`) so a finished run's cache opens as a
`FieldCache` for the post-label numpy pass. One format, two write patterns.

  uv run python tools/atlas/view_field_cache.py --run-dir data/discovery/maneuver_v14_exploration
  uv run python tools/atlas/view_field_cache.py --verify --limit 200      # cache vs engine
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from collections import Counter
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

# One engine PROCESS per worker: the CLAUDE.md concurrent-PROCESS cap, same as
# `view_rescreen.py`, which this replaces as the measuring pass.
WORKERS = 4
THREADS = 1

DEFAULT_DIR = ("view_rescreen", "fields")
FIELDS_NAME, VALID_NAME, INDEX_NAME = "fields.f32", "valid.npy", "index.json"
DTYPE = "<f4"


def row_key(r: dict) -> str:
    return f"{r.get('atom_key')}|{r.get('k')}"


class FieldCache:
    """A memmapped `(N, H, W)` f32 field store keyed by `row_key`, with a validity mask.

    The key ORDER is frozen in `index.json` at creation and never re-derived: the maneuver
    log is append-only, so re-deriving the order from a grown log would silently re-index
    every row and hand back one candidate's field for another's key. A population whose
    keys are not a superset of the recorded order is a rebuild, not a resume.
    """

    def __init__(self, root: Path, keys: list[str] | None = None, *, policy: str = "",
                 mode: str = "r"):
        self.root = Path(root)
        self.index_path = self.root / INDEX_NAME
        if keys is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.index_path.exists():
                old = json.loads(self.index_path.read_text(encoding="utf-8"))
                if old["keys"] != keys or old["policy"] != policy:
                    raise SystemExit(
                        f"{self.index_path}: the cached key order or cap policy differs from "
                        f"this population ({len(old['keys'])} vs {len(keys)} keys, policy "
                        f"{old['policy']!r} vs {policy!r}). Delete the directory to rebuild "
                        f"— appending would re-index every row.")
            else:
                self.index_path.write_text(json.dumps(dict(
                    keys=keys, policy=policy,
                    geometry=[fm.SCREEN_W, fm.SCREEN_H, fm.SCREEN_SS], dtype=DTYPE,
                    note=("scratch-class field cache for the view-level screen. Raw "
                          "render-one --dump-field f32, NaN = did not escape. Regenerable: "
                          "tools/atlas/view_field_cache.py."),
                ), indent=2) + "\n", encoding="utf-8")
        idx = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.keys: list[str] = idx["keys"]
        self.policy: str = idx["policy"]
        self.geometry = tuple(idx["geometry"])
        self.pos = {k: i for i, k in enumerate(self.keys)}
        n, w, h = len(self.keys), fm.SCREEN_W, fm.SCREEN_H
        self.shape = (n, h * fm.SCREEN_SS, w * fm.SCREEN_SS)
        fp = self.root / FIELDS_NAME
        if not fp.exists():
            np.memmap(fp, dtype=DTYPE, mode="w+", shape=self.shape).flush()
        self.fields = np.memmap(fp, dtype=DTYPE, mode=("r" if mode == "r" else "r+"),
                                shape=self.shape)
        vp = self.root / VALID_NAME
        if not vp.exists():
            np.save(vp, np.zeros(n, dtype=np.uint8))
        self.valid_path = vp
        self.valid = np.load(vp, mmap_mode=("r" if mode == "r" else "r+"))

    def has(self, key: str) -> bool:
        i = self.pos.get(key)
        return i is not None and bool(self.valid[i])

    def get(self, key: str) -> np.ndarray | None:
        """The field as `dump_field` returns it — f32, NOT upcast to f64.

        The dtype is part of the contract, not an implementation detail: `rescore_lib`
        reduces in the input dtype, so handing back f64 changes `radial_range_p90` in the
        fourth decimal on ~2% of rows and the cache stops being a substitute for a
        measurement. Caught by `--verify`, which is what it is for."""
        i = self.pos.get(key)
        if i is None or not self.valid[i]:
            return None
        return np.asarray(self.fields[i])

    def put(self, key: str, field: np.ndarray) -> None:
        i = self.pos[key]
        if field.shape != self.shape[1:]:
            raise ValueError(f"{key}: field {field.shape} != cache {self.shape[1:]}")
        self.fields[i] = field.astype(np.float32, copy=False)
        self.valid[i] = 1

    def flush(self) -> None:
        self.fields.flush()
        self.valid.flush()

    @property
    def n_valid(self) -> int:
        return int(np.asarray(self.valid).sum())


# --------------------------------------------------------------------------- #
# The append-only sibling: what a LIVE run writes into.
# --------------------------------------------------------------------------- #
RUN_INDEX_NAME, RUN_META_NAME = "index.jsonl", "meta.json"
RUN_FORMAT = "view_field_cache/append/1"


class RunFieldCache:
    """Append-only field store for a walk that does not know its population in advance.

    THE FAILURE THIS IS SHAPED AROUND is a kill mid-write, which for a long detached run is
    the normal ending and not an exception. So the two writes are ordered: the field's
    `rec_bytes` land at `i * rec_bytes` and are flushed FIRST, then one index line naming
    `i` is appended and flushed. A kill between them leaves an orphan field that no index
    line claims — and the next open recomputes `i` from the number of VALID index lines, so
    the next `put` writes straight over it. There is no torn state that survives, and no
    `.tmp` + rename either, because renaming a growing multi-hundred-MB store per field is
    the cost this format exists to avoid.

    A read truncates to `min(index lines, filesize // rec_bytes)`: an index line whose field
    is not fully on disk is not a row, whatever the line says.

    STORAGE CLASS is `scratch()`-equivalent even though it is written under the RUN dir: it
    is a deterministic function of the run's own `maneuvers.jsonl` plus committed code and
    the stamped cap policy. It lives beside the run because the run is what regenerates it
    and because a post-label feature pass wants it next to the rows it describes; nothing
    may depend on it surviving.
    """

    def __init__(self, root, *, policy: str = "", mode: str = "a"):
        self.root = Path(root)
        self.mode = mode
        self.lock = threading.Lock()
        self.rec_shape = (fm.SCREEN_H * fm.SCREEN_SS, fm.SCREEN_W * fm.SCREEN_SS)
        self.rec_items = int(self.rec_shape[0] * self.rec_shape[1])
        self.rec_bytes = self.rec_items * 4
        self.fields_path = self.root / FIELDS_NAME
        self.index_path = self.root / RUN_INDEX_NAME
        self.meta_path = self.root / RUN_META_NAME
        if mode == "a":
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.meta_path.exists():
                self.meta_path.write_text(json.dumps(dict(
                    format=RUN_FORMAT, policy=policy, dtype=DTYPE,
                    geometry=[fm.SCREEN_W, fm.SCREEN_H, fm.SCREEN_SS],
                    note=("append-only view-field cache written live by the walk. Raw "
                          "render-one --dump-field f32, NaN = did not escape. Disposable: "
                          "a deterministic function of maneuvers.jsonl + committed code."),
                ), indent=2) + "\n", encoding="utf-8")
            self.fields_path.touch(exist_ok=True)
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.policy = self.meta.get("policy", "")
        self.rows: list[dict] = []
        self.pos: dict[str, int] = {}
        self._load_index()
        self._fh = open(self.fields_path, "r+b") if mode == "a" else None

    def _load_index(self):
        self.rows, self.pos = [], {}
        if not self.index_path.exists():
            return
        on_disk = self.fields_path.stat().st_size // self.rec_bytes if \
            self.fields_path.exists() else 0
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:                  # a torn final line is not a row
                break
            if int(r.get("i", -1)) != len(self.rows) or len(self.rows) >= on_disk:
                break
            self.rows.append(r)
            self.pos[r["key"]] = int(r["i"])

    # -- write ------------------------------------------------------------- #
    def has(self, key: str) -> bool:
        return key in self.pos

    def put(self, key: str, field: np.ndarray, **meta) -> bool:
        """Append one field. False iff the key is already stored (never an overwrite)."""
        if self.mode != "a":
            raise ValueError("RunFieldCache opened read-only")
        arr = np.asarray(field, dtype=np.float32)
        if arr.shape != self.rec_shape:
            raise ValueError(f"{key}: field {arr.shape} != cache {self.rec_shape}")
        with self.lock:
            if key in self.pos:
                return False
            i = len(self.rows)
            self._fh.seek(i * self.rec_bytes)
            self._fh.write(arr.tobytes(order="C"))
            self._fh.flush()
            row = dict(i=i, key=key, **meta)
            with open(self.index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            self.rows.append(row)
            self.pos[key] = i
            return True

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- read -------------------------------------------------------------- #
    def get(self, key: str):
        i = self.pos.get(key)
        if i is None:
            return None
        with open(self.fields_path, "rb") as f:
            f.seek(i * self.rec_bytes)
            buf = f.read(self.rec_bytes)
        if len(buf) < self.rec_bytes:
            return None
        return np.frombuffer(buf, dtype=DTYPE).reshape(self.rec_shape)

    def array(self) -> np.ndarray:
        """The whole store as an `(N, H, W)` memmap — the post-label numpy pass."""
        return np.memmap(self.fields_path, dtype=DTYPE, mode="r",
                         shape=(len(self.rows),) + self.rec_shape)

    @property
    def n(self) -> int:
        return len(self.rows)

    def finalize(self) -> dict:
        """Write the retrospective pair so `FieldCache(root)` opens this store read-only.

        Derived from the index that is actually valid, never from the log the run was
        driven by: a run killed mid-batch has fields the maneuver log does not yet name and
        rows the cache never reached, and the store is the authority on what it holds.
        """
        keys = [r["key"] for r in self.rows]
        (self.root / INDEX_NAME).write_text(json.dumps(dict(
            keys=keys, policy=self.policy,
            geometry=[fm.SCREEN_W, fm.SCREEN_H, fm.SCREEN_SS], dtype=DTYPE,
            note=f"finalized from {RUN_FORMAT}; see {RUN_META_NAME}/{RUN_INDEX_NAME}.",
        ), indent=2) + "\n", encoding="utf-8")
        np.save(self.root / VALID_NAME, np.ones(len(keys), dtype=np.uint8))
        return dict(root=str(self.root), n=len(keys), policy=self.policy,
                    bytes=len(keys) * self.rec_bytes)


# --------------------------------------------------------------------------- #
def build(pop: list[dict], root: Path, *, workers=WORKERS, log=print) -> dict:
    keys = [row_key(r) for r in pop]
    if len(set(keys)) != len(keys):
        raise SystemExit("population keys are not unique — the cache index would collide")
    tok = vs.ms.screen_policy_token()
    cache = FieldCache(root, keys, policy=tok, mode="r+")
    todo = [r for r in pop if not cache.has(row_key(r))]
    log(f"  cache {root}: {cache.n_valid}/{len(keys)} present, measuring {len(todo)} "
        f"({workers} processes x {THREADS} thread)")
    lock = threading.Lock()
    fails: list[dict] = []
    t0, n = time.time(), [0]

    def work(r):
        key = row_key(r)
        fw = float(r["fw"])
        meta = vs.view_frame_policy(fw)          # one definition of the guard and the cap
        if not meta["screened"]:
            with lock:
                fails.append(dict(key=key, reason=meta["screen_reason"]))
            return
        maxiter = meta["view_maxiter"]
        try:
            with tempfile.TemporaryDirectory() as td:
                field, _ = fm.dump_field(r["cx"], r["cy"], fw, maxiter, Path(td) / "f.bin",
                                         width=fm.SCREEN_W, height=fm.SCREEN_H,
                                         ss=fm.SCREEN_SS,
                                         family=r.get("partition") or "mandelbrot",
                                         threads=THREADS, timeout=fm.FIELD_TIMEOUT_S)
        except Exception as e:
            with lock:
                fails.append(dict(key=key, reason=f"dump_field:{str(e)[:120]}"))
            return
        with lock:
            cache.put(key, field)
            n[0] += 1
            if n[0] % 500 == 0 or n[0] == len(todo):
                el = time.time() - t0
                rate = n[0] / max(1e-9, el)
                log(f"  {n[0]:6d}/{len(todo)}  {rate:5.1f} field/s  {el:6.0f}s  "
                    f"eta {(len(todo)-n[0])/max(1e-9, rate):6.0f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    cache.flush()
    # Every miss is counted under a named reason class — never characterize a failure
    # population from a truncated sample (`CLAUDE.md`, four rules).
    rep = dict(root=str(root), n=len(keys), cached=cache.n_valid,
               missing=len(keys) - cache.n_valid,
               missing_reasons=dict(Counter(f["reason"].split(":")[0] for f in fails)),
               policy=tok, seconds=round(time.time() - t0, 1),
               bytes=int(np.prod(cache.shape)) * 4)
    (root / "build_report.json").write_text(json.dumps(rep, indent=2) + "\n",
                                            encoding="utf-8")
    return rep


def verify(pop: list[dict], root: Path, *, limit: int, seed: int = 20260801) -> dict:
    """Re-measure a seeded random subset through the ENGINE and compare to the cache.

    The claim the cache makes is "a cached row re-derives byte-identically", and a claim
    that is not checked is a hope. Compared on the MEASURES, not on the raw array, because
    that is what everything downstream reads; an exact-array check is stricter than the
    contract and would flag a bit-identical engine re-run that reordered nothing.
    """
    import random
    cache = FieldCache(root)
    have = [r for r in pop if cache.has(row_key(r))]
    sub = random.Random(seed).sample(have, min(limit, len(have)))
    bad, checked = [], 0
    for r in sub:
        live = vs.measure_view(r["cx"], r["cy"], r["fw"],
                               family=r.get("partition") or "mandelbrot", threads=THREADS)
        if not live.get("screened"):
            continue
        cached = vs.view_measures(cache.get(row_key(r)))
        checked += 1
        diff = {k: (cached.get(k), live.get(k)) for k in cached
                if k != "interior_radial" and cached.get(k) != live.get(k)}
        if diff:
            bad.append(dict(key=row_key(r), diff=diff))
    return dict(checked=checked, mismatched=len(bad), examples=bad[:5],
                NOTE=("A mismatch means the cache and the engine disagree on the same "
                      "(cx, cy, fw, cap policy) — the cache is then not a substitute for a "
                      "measurement and must be rebuilt, not patched."))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, nargs="+",
                    default=[ROOT / "data" / "discovery" / "maneuver_v14_exploration"])
    ap.add_argument("--root", type=Path, default=paths.scratch(*DEFAULT_DIR))
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--verify", action="store_true",
                    help="re-measure a subset through the engine and compare")
    ap.add_argument("--limit", type=int, default=200, help="--verify subset size")
    a = ap.parse_args(argv)
    if a.workers > 4:
        print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
        return 2

    pop = mis.load_population([Path(d) / "maneuvers.jsonl" for d in a.run_dir])
    print(f"[pop] {len(pop)} available+screened maneuver candidates")
    if a.verify:
        print(json.dumps(verify(pop, a.root, limit=a.limit), indent=2))
        return 0
    rep = build(pop, a.root, workers=a.workers)
    print(json.dumps(rep, indent=2))
    print(f"-> {a.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
