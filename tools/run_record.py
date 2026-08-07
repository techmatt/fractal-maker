#!/usr/bin/env python
r"""run_record.py — THE segmented layout for a discovery run's committed per-row telemetry:
one writer that rotates a growing `.jsonl` into gzipped segments, and one reader that makes
the segmentation invisible to every consumer.

WHY THIS EXISTS. A `steered_frontier` run commits its whole per-row record. Measured on
`data/discovery/steady_state_v1_20260805` (70 active min, 10.30 MB committed) the record grows
at 8.9 MB/h, and `harvest_v2_proving_20260803` hit 14.2 MB/h — so an 8 h run lands at 50-70 MB
and every run past ~2.5 h crosses the 20 MB commit rule in CLAUDE.md. None of it is
regenerable: `harvest_log.jsonl` is the tau_h curve's data on record, `prio_terms.jsonl` is the
only record of the never-admitted majority, `q4_candidates.jsonl` is the reject-autopsy
population. THINNING IS THEREFORE NOT AVAILABLE — the record has to get smaller without
losing a row or a field.

WHY COMPRESSION AND NOT FIELD-DROPPING, with the measurement that decided it. Git already
zlib-compresses ordinary blobs, so gzipping a plain-tracked file buys nothing on the remote:
committing `maneuvers.jsonl` raw packs to 451,089 B and committing the same rows as a `.gz`
packs to 451,314 B (measured 2026-08-07, `git gc --aggressive` on a scratch repo, both ways).
The reason it wins anyway is `.gitattributes`: the four largest streams here — harvest_log,
prio_terms, maneuvers, q4_candidates, 87-98% of the record — are **LFS-tracked**, and LFS ships
the object BYTE-FOR-BYTE. There is no zlib in that path, so the 8-11x these streams gzip at is
8-11x off the bytes the rule counts. Field-dropping was measured on the same file for
comparison and is not competitive: dropping `atom_key` saves 5.7% of compressed bytes,
`screen.interior_radial` 7.4%, every null 0.9% — each one paying real information for a
fraction of what compression gives for free.

THE LAYOUT. For a logical stream `<dir>/<stem>.jsonl`:

    <stem>.000.jsonl.gz   segment 0, closed and compressed
    <stem>.001.jsonl.gz   segment 1, ...
    <stem>.jsonl          the LIVE tail — plain, append-only, bounded by ROTATE_BYTES

Rows are read segment 0, 1, ... then the live tail: exactly write order. A run dir written
before this module exists has no segments and is read by the same call, unchanged — the reader
is layout-agnostic by construction, which is what let ~30 consumers be converted mechanically.

ROTATION IS CRASH-SAFE BY RENAME, NOT BY COPY. Compressing the live file in place and then
truncating it has a window in which a kill leaves the rows in BOTH the segment and the tail,
and duplicated rows are worse than missing ones here: `harvest_log_reconcile` compares the
log's length against `totals`, and a dup reads as an accounting bug in the crawl. So rotation
is: `os.replace(live, <stem>.NNN.jsonl)` (atomic, the tail is gone in one step) -> gzip that
staged segment to `<stem>.NNN.jsonl.gz` via a tmp + `os.replace` -> unlink the staged plain
segment. A kill anywhere in that sequence leaves each row in exactly one readable place, and
`SegmentWriter` heals the leftovers on its next open. Both `<stem>.NNN.jsonl` and
`<stem>.NNN.jsonl.gz` are legal segment forms for the reader; when both exist the `.gz` wins
and its plain twin is ignored (that pair is the crash state, not a second segment).

WHAT IS SEGMENTED, AND WHAT IS DELIBERATELY NOT. `SEGMENTED_STREAMS` below is the registry.
`outcome_ledger.jsonl` is NOT in it and must not be added casually: 66 modules read it, several
by `rglob("outcome_ledger.jsonl")` across all of `data/`, and an rglob does not fail when a
finalized run's rows move into a segment — it returns a SHORTER population, silently, which is
the one failure mode this tree keeps paying for. It is also only 0.4-6% of the record. The
same argument protects `outcome_feats.npz` (already `savez_compressed`; re-packing it as one
2-D array saves 30% and touches the shared embedding format — not worth it).

    from tools.run_record import iter_rows, read_rows, SegmentWriter
    for row in iter_rows(run_dir / "harvest_log.jsonl"):   # segments + tail, in order
        ...

Deliberately dependency-free (stdlib only): the guard tests that import this must stay in the
light `pytest` lane, and the writer sits on the crawl's hot append path.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "SEGMENTED_STREAMS", "ROTATE_BYTES", "is_segmented", "segment_paths", "exists",
    "stored_bytes", "iter_lines", "iter_rows", "read_rows", "SegmentWriter",
]

# The registry of run-record streams that rotate. Membership is a decision about BLAST RADIUS
# as much as bytes (see the module docstring): a stream enters this tuple only once every
# reader of it goes through `iter_rows`/`read_rows`, which `test_run_record.py`'s source scan
# checks. All five are committed; the ignored per-run streams (saturation, dive_log,
# julia_hooks, scheduler_trace) are left alone on purpose — their segments would NOT be matched
# by the exact-filename .gitignore rules that currently exclude them, so rotating one would
# commit a file the tree has decided not to keep.
SEGMENTED_STREAMS = (
    "harvest_log.jsonl",        # tau_h curve input (LFS)
    "prio_terms.jsonl",         # pushed-candidate steering record (LFS)
    "maneuvers.jsonl",          # operator decisions (LFS)
    "q4_candidates.jsonl",      # record-and-rank / reject autopsy (LFS)
    "quota_trace.jsonl",        # per-batch allocator trace (plain)
)

# Rotate once the LIVE tail passes this. It bounds the uncompressed exposure of a run that is
# killed rather than finished: whatever the crawl was appending to is at most this big and
# plain, everything before it is already compressed. 4 MiB puts an 8 h prio_terms (the worst
# observed stream, 33.8 MB over 8.8 h on continuous_v1_20260803) at ~9 segments.
ROTATE_BYTES = 4 * 1024 * 1024

_SEG_RE = re.compile(r"^(?P<stem>.+)\.(?P<idx>\d{3})\.jsonl(?P<gz>\.gz)?$")
_SEG_FMT = "{stem}.{idx:03d}.jsonl"


def is_segmented(path: str | os.PathLike) -> bool:
    """Whether `path`'s basename is a registered segmented stream. The one place that
    question is answered — callers must not restate the tuple."""
    return Path(path).name in SEGMENTED_STREAMS


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def segment_paths(path: str | os.PathLike) -> list[Path]:
    """Every file holding rows of the logical stream `path`, in WRITE ORDER.

    Segments `<stem>.000.jsonl[.gz]` ascending by index, then the live `<stem>.jsonl` if it
    exists. A `.gz` segment shadows a plain segment of the same index (the crash state left
    behind by an interrupted rotation, never a second segment). Returns `[]` for a stream that
    has never been written — absence is the caller's to interpret, as it was before.

    An UNREGISTERED path short-circuits to `[p]`: `iter_rows` is meant to be safe to drop into
    a module's one local `_jl(...)` loader, which is also used for pool/plan/manifest files, and
    a directory scan per call would be a real cost on the large stores those live in.
    """
    p = Path(path)
    if not is_segmented(p):
        return [p] if p.exists() else []
    stem = p.name[: -len(".jsonl")]
    d = p.parent
    if not d.is_dir():
        return [p] if p.exists() else []
    by_idx: dict[int, Path] = {}
    for f in d.iterdir():
        m = _SEG_RE.match(f.name)
        if not m or m.group("stem") != stem:
            continue
        idx = int(m.group("idx"))
        # .gz wins over its plain twin; otherwise first (and only) writer wins.
        if m.group("gz") or idx not in by_idx:
            by_idx[idx] = f
    out = [by_idx[i] for i in sorted(by_idx)]
    if p.exists():
        out.append(p)
    return out


def exists(path: str | os.PathLike) -> bool:
    """True if the stream has any rows on disk, in any layout. Use this instead of
    `Path(...).exists()` on a segmented stream: a FINALIZED run has no live `<stem>.jsonl`
    at all, and a plain `.exists()` reports it as a run that never wrote one."""
    return bool(segment_paths(path))


def stored_bytes(path: str | os.PathLike) -> int:
    """Bytes the stream occupies on disk across every segment — what the commit pays."""
    return sum(f.stat().st_size for f in segment_paths(path))


def _open_segment(f: Path):
    if f.suffix == ".gz":
        return gzip.open(f, "rt", encoding="utf-8")
    return open(f, "r", encoding="utf-8")


def iter_lines(path: str | os.PathLike) -> Iterator[str]:
    """Yield every non-empty stripped line of the stream, segments then live tail."""
    for f in segment_paths(path):
        yield from _lines_of(f)


def iter_rows(path: str | os.PathLike) -> Iterator[dict]:
    """Yield every row of the stream as a dict, in write order. THE reader — a consumer that
    opens the `.jsonl` directly sees only whatever has not rotated yet."""
    for line in iter_lines(path):
        yield json.loads(line)


def read_rows(path: str | os.PathLike) -> list[dict]:
    """`iter_rows` materialized. Returns `[]` for a stream with no files, matching the
    absence-tolerant local loaders this replaced (the ones that were already guarding on
    `.exists()`). A caller that used to `open()` the file and RAISE wants `require_rows`."""
    return list(iter_rows(path))


class MissingStreamError(FileNotFoundError):
    """A stream that must be there has no files in any layout.

    This exists because the conversion to segments had to be loudness-preserving in BOTH
    directions. About half the readers converted here used to open the `.jsonl` directly, so
    a missing log raised and named the path; swapping them all to `read_rows` would have made
    every one of them absence-tolerant in the same commit — and an absence-tolerant read of a
    log that is gone is how a derivation quietly runs on a smaller population
    (`verification_practice.md` §2; `tau_h_rederive` would have contributed 0 rows for a
    vanished run and printed it as a run with 0 re-scoreable checks)."""


def require_rows(path: str | os.PathLike) -> list[dict]:
    """`read_rows`, but a stream with no files RAISES naming every layout that was looked
    for. Use this wherever the pre-segment code opened the file and let the OSError out."""
    p = Path(path)
    files = segment_paths(p)
    if not files:
        stem = p.name[: -len(".jsonl")] if p.name.endswith(".jsonl") else p.name
        raise MissingStreamError(
            f"no rows for stream {p} — looked for the live tail `{p.name}` and for rotated "
            f"segments `{stem}.<nnn>.jsonl[.gz]` in {p.parent}. A run that wrote this stream "
            f"leaves one or the other; both absent means the run never wrote it, or the "
            f"record is gone.")
    return [json.loads(line) for f in files for line in _lines_of(f)]


def _lines_of(f: Path):
    with _open_segment(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
class SegmentWriter:
    """Append-only writer for one segmented stream. Appends to the live `<stem>.jsonl` and
    rotates it into a `.gz` segment once it passes `rotate_bytes`.

    Resume-safe: construction adopts whatever is already on disk (a resumed run keeps
    appending to its live tail, and its existing segments keep their indices) and heals a
    half-finished rotation left by a kill.
    """

    def __init__(self, path: str | os.PathLike, *, rotate_bytes: int = ROTATE_BYTES):
        self.path = Path(path)
        # The registry is the contract in BOTH directions: `segment_paths` only scans for
        # segments of a registered stream, so writing them for an unregistered one would
        # produce files no reader ever looks at. Refuse at construction rather than lose rows.
        if not is_segmented(self.path):
            raise ValueError(
                f"{self.path.name!r} is not in run_record.SEGMENTED_STREAMS — a stream must be "
                f"registered (and all its readers converted) before it may rotate")
        self.rotate_bytes = int(rotate_bytes)
        self.stem = self.path.name[: -len(".jsonl")]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._heal()
        self._live = self.path.stat().st_size if self.path.exists() else 0

    # ---- internals ------------------------------------------------------- #
    def _seg(self, idx: int, gz: bool) -> Path:
        return self.path.parent / (_SEG_FMT.format(stem=self.stem, idx=idx) + (".gz" if gz else ""))

    def _next_index(self) -> int:
        idx = -1
        for f in self.path.parent.iterdir():
            m = _SEG_RE.match(f.name)
            if m and m.group("stem") == self.stem:
                idx = max(idx, int(m.group("idx")))
        return idx + 1

    def _heal(self):
        """Finish any rotation a kill interrupted. Two leftovers are possible and both are
        recoverable without touching a row: a staged plain segment with no `.gz` yet (compress
        it), and a `.gz` that already landed beside its plain twin (drop the twin). Tmp files
        from an interrupted gzip are unlinked — the staged plain segment still holds the rows."""
        for f in list(self.path.parent.iterdir()):
            if f.name.endswith(".jsonl.gz.tmp") and f.name.startswith(self.stem + "."):
                f.unlink(missing_ok=True)
        for f in list(self.path.parent.iterdir()):
            m = _SEG_RE.match(f.name)
            if not m or m.group("stem") != self.stem or m.group("gz"):
                continue
            idx = int(m.group("idx"))
            gz = self._seg(idx, gz=True)
            if gz.exists():
                f.unlink(missing_ok=True)          # rotation got past the gzip; drop the twin
            else:
                self._compress(f, gz)

    @staticmethod
    def _compress(src: Path, dst: Path):
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, length=1 << 20)
        os.replace(tmp, dst)
        src.unlink(missing_ok=True)

    def _rotate(self):
        """Live tail -> compressed segment. Rename FIRST so no row is ever in two places."""
        idx = self._next_index()
        staged = self._seg(idx, gz=False)
        os.replace(self.path, staged)              # atomic; the live tail is gone
        self._live = 0
        self._compress(staged, self._seg(idx, gz=True))

    # ---- public ---------------------------------------------------------- #
    def write_lines(self, lines: Iterable[str]):
        """Append pre-serialized rows (each without its newline). Rotation is checked once
        per call, not per line: a batch of prio rows is one append and one size check."""
        buf = "".join(ln + "\n" for ln in lines)
        if not buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(buf)
        self._live += len(buf.encode("utf-8"))
        if self._live >= self.rotate_bytes:
            self._rotate()

    def write_row(self, row: dict, *, default=None):
        self.write_lines([json.dumps(row, default=default)])

    def write_rows(self, rows: Iterable[dict], *, default=None):
        self.write_lines([json.dumps(r, default=default) for r in rows])

    def finalize(self):
        """Compress the live tail at the end of a run so the committed record is entirely
        segments. Safe to call twice, and safe to never call — an unfinalized tail is at most
        `rotate_bytes` of plain text and reads identically."""
        if self.path.exists() and self.path.stat().st_size > 0:
            self._rotate()
        elif self.path.exists():
            self.path.unlink(missing_ok=True)

    def stored_bytes(self) -> int:
        return stored_bytes(self.path)


def replace_stream(path: str | os.PathLike, rows: Iterable[dict], *, default=None) -> int:
    """Rewrite a WHOLE segmented stream from `rows`. Returns the row count written.

    A repair tool that rewrites one of these stores (`backfill_triggered_stamp`) cannot just
    write the file back: dropping a fresh plain `<stem>.jsonl` beside the segments that are
    still there DOUBLES every rotated row, and nothing downstream would notice — the readers
    concatenate. So the replacement is stream-wide.

    Ordered so the rows exist somewhere at every instant: stage the full new content into
    `<stem>.rewrite.jsonl` -> unlink every old segment and the old tail -> `os.replace` the
    staging file into the live tail -> compress it into a single segment. A kill inside the
    middle step leaves `<stem>.rewrite.jsonl` holding the complete new stream, named for what
    it is and recoverable by one rename.
    """
    p = Path(path)
    if not is_segmented(p):
        raise ValueError(f"{p.name!r} is not a segmented stream; write it directly")
    stem = p.name[: -len(".jsonl")]
    staging = p.parent / (stem + ".rewrite.jsonl")
    n = 0
    with open(staging, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=default) + "\n")
            n += 1
    for old in segment_paths(p):
        old.unlink(missing_ok=True)
    os.replace(staging, p)
    SegmentWriter(p).finalize()
    return n
