#!/usr/bin/env python
r"""`tools/run_record.py` is the ONLY place a committed run-record stream is written or read.

Four things, each earned by a way the compaction could silently lose rows rather than fail:

  (1) ROUND-TRIP ON A REAL RUN DIR. A fixture run dir is written through `SegmentWriter` at a
      rotation threshold small enough to force segments, and the DERIVED READS the tree
      actually takes off these files — `harvest_log_reconcile.read_rows`+`dedup`,
      `tau_h_rederive._harvest_rows`, `q4_fate_sheets.load_rows`,
      `harvest_v2_readout` — are asserted identical to the same reads on the same rows in the
      old one-plain-file layout. Byte size is not the subject; the derived number is.
  (2) EVERY LAYOUT READS THE SAME. Legacy (plain file only), mid-run (segments + live tail)
      and finalized (segments only) are three different states of the same run, and the reader
      is only useful if a consumer cannot tell them apart.
  (3) CRASH SAFETY. Rotation is asserted to leave every row in exactly ONE readable place at
      every intermediate state, and `SegmentWriter` is asserted to heal each of them. A
      duplicated row is the failure that matters here: `harvest_log_reconcile` reads a dup as
      the crawl's counters being wrong.
  (4) A SOURCE SCAN. No tracked Python file may read a registered stream by opening it
      directly — that is the bug this module exists to prevent, and it fails OPEN (a finalized
      run has no plain file, so the direct reader gets a shorter population or an exception
      depending on which dialect it used).

  uv run pytest tools/test_run_record.py -q
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "mining",
           ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_record as R  # noqa: E402

OWNER = "tools/run_record.py"


# --------------------------------------------------------------------------- #
# fixtures: rows shaped like the real streams
# --------------------------------------------------------------------------- #
def _harvest_rows(n=400):
    """Harvest checks with the fields the derived reads actually key on."""
    parts = ["mandelbrot", "multibrot3", "julia:mandelbrot", "phoenix"]
    out = []
    for i in range(n):
        p = parts[i % len(parts)]
        out.append(dict(
            batch=1 + i // 20, partition=p, depth=2 + i % 3, node_id=1000 + i,
            root_id=100 + i % 7, cx=-0.5 + i * 1e-4, cy=0.1 + i * 1e-4, fw=1e-3 / (1 + i % 5),
            julia_c_re=("-0.77" if p.startswith("julia") else None),
            julia_c_im=("0.10" if p.startswith("julia") else None),
            cheap_pgood=0.1 + (i % 90) / 100, cheap_eord=1.0 + (i % 7) / 10,
            canon_nb=(None if i % 4 else 0.9), canon_pgood=(None if i % 4 else 0.8),
            canon_pge4=None, canon_decoded=(None if i % 4 else 3), reframe_decoded=None,
            admitted=bool(i % 11 == 0), tau_h=0.4126, precanon_dup=None,
            mix_source="sampler", maneuver=None,
            # padding: real rows are ~500 B, and a rotation threshold has to see realistic bytes
            note="x" * 200,
        ))
    return out


def _q4_rows(n=200):
    fates = ["admitted", "below_tau_h", "canon_not_q3", "guarded", "precanon_dup"]
    return [dict(batch=1 + i // 20, partition="multibrot3", fate=fates[i % len(fates)],
                 rank_tier=1 + i % 3, rank_score=1.0 + i / 100, node_id=2000 + i,
                 root_id=200 + i % 5, depth=2, cx=0.3 + i * 1e-5, cy=0.6 + i * 1e-5,
                 fw=1e-4, julia_c_re=None, julia_c_im=None, triggered=False,
                 mix_source="sampler", maneuver=None, scorer_version="v10",
                 note="y" * 200)
            for i in range(n)]


def _write_plain(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_segmented(p: Path, rows, *, rotate_bytes=12_000, finalize=True):
    w = R.SegmentWriter(p, rotate_bytes=rotate_bytes)
    for r in rows:
        w.write_row(r)
    if finalize:
        w.finalize()
    return w


# --------------------------------------------------------------------------- #
# (1) round-trip: the DERIVED reads are identical across layouts
# --------------------------------------------------------------------------- #
def test_a_rotated_stream_yields_byte_identical_rows_in_write_order(tmp_path):
    rows = _harvest_rows()
    _write_segmented(tmp_path / "harvest_log.jsonl", rows)
    assert len([f for f in tmp_path.iterdir() if f.name.endswith(".gz")]) > 1, \
        "fixture must actually rotate or it is testing nothing"
    assert R.read_rows(tmp_path / "harvest_log.jsonl") == rows


def test_the_derived_reads_match_the_old_plain_layout(tmp_path):
    """THE round-trip the compaction has to pass: every reader the tree takes off these
    streams produces the same answer on the compacted record as on the plain one."""
    import harvest_log_reconcile as hlr
    import harvest_log_registry as hreg
    import tau_h_rederive as thr
    import q4_fate_sheets as q4s
    import harvest_v2_readout as hv2

    hrows, qrows = _harvest_rows(), _q4_rows()
    plain, seg = tmp_path / "plain", tmp_path / "seg"
    _write_plain(plain / "harvest_log.jsonl", hrows)
    _write_plain(plain / "q4_candidates.jsonl", qrows)
    _write_segmented(seg / "harvest_log.jsonl", hrows)
    _write_segmented(seg / "q4_candidates.jsonl", qrows)
    # the compacted record IS smaller — that is the point of the exercise
    assert R.stored_bytes(seg / "harvest_log.jsonl") < R.stored_bytes(plain / "harvest_log.jsonl")

    # harvest_log_reconcile: rows, torn count, and the dedup report
    a_rows, a_torn = hlr.read_rows(plain)
    b_rows, b_torn = hlr.read_rows(seg)
    assert (a_rows, a_torn) == (b_rows, b_torn)
    assert hlr.dedup(a_rows)[1] == hlr.dedup(b_rows)[1]

    # tau_h_rederive: the re-scoreable population, incl. its phoenix/no-geometry exclusions
    def _pop(d):
        run = hreg.HarvestRun(name=d.name, store="t", dir=d,
                              log=d / "harvest_log.jsonl", pinned=False)
        return thr._harvest_rows([run])
    pa, pb = _pop(plain), _pop(seg)
    assert len(pa) == len(pb) > 0
    assert [r["partition"] for r in pa] == [r["partition"] for r in pb]

    # q4_fate_sheets.load_rows (dedup on geometry) and harvest_v2_readout's fate tally
    assert q4s.load_rows(plain) == q4s.load_rows(seg)
    assert hv2._jl(plain / "q4_candidates.jsonl") == hv2._jl(seg / "q4_candidates.jsonl")


def test_reconcile_still_counts_a_torn_tail_rather_than_dropping_it(tmp_path):
    """The tail is where a kill leaves a partial line, and it stays PLAIN — so the torn-line
    count the reconcile reports must survive segmentation."""
    import harvest_log_reconcile as hlr
    rows = _harvest_rows(200)
    w = _write_segmented(tmp_path / "harvest_log.jsonl", rows, finalize=False)
    with open(w.path, "a", encoding="utf-8") as f:
        f.write('{"batch": 99, "part')                 # killed mid-write
    got, torn = hlr.read_rows(tmp_path)
    assert torn == 1 and got == rows


# --------------------------------------------------------------------------- #
# (2) every layout reads the same
# --------------------------------------------------------------------------- #
def test_a_legacy_unrotated_run_dir_reads_unchanged(tmp_path):
    rows = _harvest_rows(50)
    _write_plain(tmp_path / "harvest_log.jsonl", rows)
    assert R.exists(tmp_path / "harvest_log.jsonl")
    assert R.read_rows(tmp_path / "harvest_log.jsonl") == rows
    assert R.segment_paths(tmp_path / "harvest_log.jsonl") == [tmp_path / "harvest_log.jsonl"]


def test_mid_run_and_finalized_layouts_are_indistinguishable_to_a_reader(tmp_path):
    rows = _harvest_rows(300)
    p = tmp_path / "harvest_log.jsonl"
    w = _write_segmented(p, rows, finalize=False)
    assert p.exists(), "mid-run: a live plain tail is still there"
    mid = R.read_rows(p)
    w.finalize()
    assert not p.exists(), "finalized: the live tail is gone, rows live in segments"
    assert R.exists(p), "...and `exists` must still say the stream is there"
    assert R.read_rows(p) == mid == rows


def test_a_missing_stream_is_absent_not_empty(tmp_path):
    p = tmp_path / "harvest_log.jsonl"
    assert R.segment_paths(p) == [] and not R.exists(p) and R.read_rows(p) == []


def test_resume_appends_to_the_existing_tail_and_keeps_segment_indices(tmp_path):
    p = tmp_path / "prio_terms.jsonl"
    first, second = _harvest_rows(300), _harvest_rows(120)
    _write_segmented(p, first, finalize=False)
    n_before = len([f for f in R.segment_paths(p) if f.suffix == ".gz"])
    w2 = R.SegmentWriter(p, rotate_bytes=12_000)          # the resume
    w2.write_rows(second)
    w2.finalize()
    assert R.read_rows(p) == first + second
    assert len([f for f in R.segment_paths(p) if f.suffix == ".gz"]) > n_before


def test_finalize_is_idempotent_and_optional(tmp_path):
    p = tmp_path / "quota_trace.jsonl"
    rows = _q4_rows(60)
    w = _write_segmented(p, rows, rotate_bytes=8_000)
    w.finalize(); w.finalize()
    assert R.read_rows(p) == rows


# --------------------------------------------------------------------------- #
# (3) crash safety: every row in exactly one place, and healed on reopen
# --------------------------------------------------------------------------- #
def test_a_kill_between_the_rename_and_the_gzip_loses_and_duplicates_nothing(tmp_path):
    """The staged plain segment IS the rows. It reads as a segment, and the next open
    compresses it."""
    p = tmp_path / "maneuvers.jsonl"
    rows = _harvest_rows(200)
    w = R.SegmentWriter(p, rotate_bytes=1 << 30)
    w.write_rows(rows)
    # simulate the crash: rename happened, gzip did not
    import os
    staged = tmp_path / "maneuvers.000.jsonl"
    os.replace(p, staged)
    assert R.read_rows(p) == rows, "a staged plain segment must read as a segment"
    R.SegmentWriter(p, rotate_bytes=1 << 30)              # reopen heals it
    assert not staged.exists() and (tmp_path / "maneuvers.000.jsonl.gz").exists()
    assert R.read_rows(p) == rows


def test_a_kill_after_the_gzip_but_before_the_unlink_does_not_double_the_rows(tmp_path):
    p = tmp_path / "maneuvers.jsonl"
    rows = _harvest_rows(120)
    _write_segmented(p, rows, rotate_bytes=1 << 30)       # -> maneuvers.000.jsonl.gz
    gz = tmp_path / "maneuvers.000.jsonl.gz"
    assert gz.exists()
    # simulate the crash: the plain twin was never unlinked
    twin = tmp_path / "maneuvers.000.jsonl"
    with gzip.open(gz, "rt", encoding="utf-8") as fi:
        twin.write_text(fi.read(), encoding="utf-8")
    assert R.read_rows(p) == rows, "the .gz must shadow its plain twin, not concatenate with it"
    R.SegmentWriter(p, rotate_bytes=1 << 30)
    assert not twin.exists()
    assert R.read_rows(p) == rows


def test_a_leftover_gzip_tmp_never_enters_a_read(tmp_path):
    p = tmp_path / "maneuvers.jsonl"
    rows = _harvest_rows(80)
    _write_segmented(p, rows, rotate_bytes=1 << 30)
    (tmp_path / "maneuvers.001.jsonl.gz.tmp").write_bytes(b"\x00garbage")
    assert R.read_rows(p) == rows
    R.SegmentWriter(p, rotate_bytes=1 << 30)
    assert not (tmp_path / "maneuvers.001.jsonl.gz.tmp").exists()


def test_replace_stream_rewrites_the_whole_stream_rather_than_shadowing_it(tmp_path):
    """`backfill_triggered_stamp` rewrites a store. Dropping a plain file back beside the
    segments would double every rotated row and no reader would notice."""
    p = tmp_path / "q4_candidates.jsonl"
    rows = _q4_rows(200)
    _write_segmented(p, rows, rotate_bytes=12_000)
    assert len(R.segment_paths(p)) > 1
    patched = [dict(r, triggered=True) for r in rows]
    assert R.replace_stream(p, patched) == len(patched)
    assert R.read_rows(p) == patched                     # not patched + rows


# --------------------------------------------------------------------------- #
# the registry is the contract in both directions
# --------------------------------------------------------------------------- #
def test_an_unregistered_stream_may_not_rotate(tmp_path):
    with pytest.raises(ValueError, match="SEGMENTED_STREAMS"):
        R.SegmentWriter(tmp_path / "saturation.jsonl")
    with pytest.raises(ValueError):
        R.replace_stream(tmp_path / "pool.jsonl", [])


def test_an_unregistered_path_is_read_without_a_directory_scan(tmp_path):
    """`iter_rows` is dropped into loaders that also read pool/plan/manifest files; those must
    not pay a scan, and a stray `pool.000.jsonl.gz` must not silently join a read."""
    rows = _q4_rows(5)
    _write_plain(tmp_path / "pool.jsonl", rows)
    with gzip.open(tmp_path / "pool.000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"stray": True}) + "\n")
    assert R.read_rows(tmp_path / "pool.jsonl") == rows


def test_harvest_discovery_still_finds_a_run_whose_log_has_been_finalized(tmp_path):
    """The silent-shrink guard. A finished run has no `harvest_log.jsonl`; discovery keyed on
    that exact name would stop seeing runs the moment they complete."""
    import harvest_log_registry as hreg
    store = tmp_path / "discovery"
    (store / "finalized_run").mkdir(parents=True)
    _write_segmented(store / "finalized_run" / "harvest_log.jsonl", _harvest_rows(60))
    (store / "legacy_run").mkdir(parents=True)
    _write_plain(store / "legacy_run" / "harvest_log.jsonl", _harvest_rows(20))
    runs, missing = hreg.discover_run_dirs(registry=[("t", store)], pinned=[])
    assert sorted(r.name for r in runs) == ["finalized_run", "legacy_run"]
    assert missing == []
    assert all(len(R.read_rows(r.log)) > 0 for r in runs)


def test_every_registered_stream_is_committed_and_none_of_the_ignored_ones_rotate():
    """Rotating an IGNORED stream would commit it: the .gitignore rules that exclude
    saturation/dive_log/julia_hooks name the exact filename, which a `.NNN.jsonl.gz` segment
    does not match."""
    ignored = {"saturation.jsonl", "dive_log.jsonl", "julia_hooks.jsonl",
               "scheduler_trace.jsonl", "state.json"}
    assert not (set(R.SEGMENTED_STREAMS) & ignored)
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in R.SEGMENTED_STREAMS:
        assert f"/data/discovery/**/{name}\n" not in gi, \
            f"{name} is in SEGMENTED_STREAMS but .gitignore excludes it"


def test_the_lfs_rules_cover_the_segments_of_every_lfs_tracked_stream():
    """The whole size win is that these four go through LFS, which ships bytes RAW — a
    segment that fell out of the LFS rule would be a plain blob git re-zlibs to the same
    size, i.e. the compaction silently doing nothing."""
    ga = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    lfs_streams = [n for n in R.SEGMENTED_STREAMS
                   if f"data/discovery/**/{n} filter=lfs" in ga]
    assert len(lfs_streams) >= 4, lfs_streams
    assert "data/discovery/**/*.jsonl.gz filter=lfs" in ga


# --------------------------------------------------------------------------- #
# (4) source scan: nobody reads a registered stream by opening it
# --------------------------------------------------------------------------- #
EXEMPT = {OWNER, "tools/test_run_record.py"}
# A direct read of a registered stream, in the dialects actually found in this tree:
#   open(<...>/"harvest_log.jsonl")           .read_text()      .open(encoding=...)
#   Path(x) / "q4_candidates.jsonl"  followed by any of the above on the same expression
_STREAM = "(?:" + "|".join(re.escape(n) for n in R.SEGMENTED_STREAMS) + ")"
# READS only. A test that WRITES a plain fixture (`open(..., "w")`) or appends a torn line to
# one (`"a"`) is building the LEGACY layout on purpose — that coverage is the point, so the
# mode is what distinguishes a fixture from a consumer.
_Q = r"[\"']" + _STREAM + r"[\"']"
DIRECT = re.compile(
    r"(?:open\(\s*[^)\n]*" + _Q + r"\s*\)?\s*(?![,)]\s*[\"'][wax])[,)]|"   # open(.../x.jsonl, ..)
    + _Q + r"\s*\)\s*\.\s*(?:read_text|exists)\(|"                        # (../x.jsonl).read_text()
    + _Q + r"\s*\)\s*\.\s*open\(\s*(?![\"'][wax]))")                      # (../x.jsonl).open()


def _tracked_py() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split() if p and p not in EXEMPT]


def test_no_tracked_module_reads_a_registered_stream_by_opening_it():
    hits = []
    for rel in _tracked_py():
        src = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if DIRECT.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, (
        "these read a rotating run-record stream directly; a FINALIZED run has no plain file, "
        "so they get a short population or an exception. Route through run_record:\n  "
        + "\n  ".join(hits))


def test_the_scan_would_actually_catch_a_copy(tmp_path):
    """The control. A scan that cannot go red on the shape it hunts proves nothing."""
    for bad in ['rows = [json.loads(l) for l in open(d / "harvest_log.jsonl", encoding="utf-8")]',
                'txt = (run_dir / "maneuvers.jsonl").read_text(encoding="utf-8")',
                'with (self.dir / "q4_candidates.jsonl").open(encoding="utf-8") as fh:']:
        assert DIRECT.search(bad), bad
    for ok in ['rows = run_record.read_rows(d / "harvest_log.jsonl")',
               'for r in run_record.iter_rows(run_dir / "maneuvers.jsonl"):',
               'self.q4_log = self.run_dir / "q4_candidates.jsonl"']:
        assert not DIRECT.search(ok), ok


def test_the_writers_all_go_through_the_segment_writer():
    """A second bare append path beside `SegmentWriter` would write rows into a live tail that
    keeps growing past the rotation threshold — the compaction quietly not applying.

    The attribute names are DERIVED per file (`self.<attr> = ... / "<registered stream>"`)
    rather than listed: `deficit_scheduler` also owns a `self.trace_path`, and it points at
    `scheduler_trace.jsonl`, which is ignored and must NOT rotate. A hand list would either
    miss a future attr or flag that one forever."""
    hits = []
    for rel in _tracked_py():
        src = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        attrs = set(re.findall(
            r"self\.(\w+)\s*=\s*[^\n]*[\"']" + _STREAM + r"[\"']", src))
        if not attrs:
            continue
        pat = re.compile(r"open\(\s*self\.(?:" + "|".join(sorted(attrs)) + r")\s*,\s*[\"']a")
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, "bare append beside SegmentWriter:\n  " + "\n  ".join(hits)


def test_the_writer_scan_would_actually_catch_a_bare_append(tmp_path):
    """Control for the scan above, including the `deficit_scheduler` false positive it must
    NOT produce."""
    src_bad = ('class X:\n    def __init__(self, d):\n'
               '        self.man_log = d / "maneuvers.jsonl"\n'
               '    def w(self, r):\n'
               '        with open(self.man_log, "a", encoding="utf-8") as f:\n            f.write(r)\n')
    src_ok = ('class X:\n    def __init__(self, d):\n'
              '        self.trace_path = d / "scheduler_trace.jsonl"\n'
              '    def w(self, r):\n'
              '        with open(self.trace_path, "a", encoding="utf-8") as f:\n            f.write(r)\n')
    def _scan(src):
        attrs = set(re.findall(r"self\.(\w+)\s*=\s*[^\n]*[\"']" + _STREAM + r"[\"']", src))
        if not attrs:
            return []
        pat = re.compile(r"open\(\s*self\.(?:" + "|".join(sorted(attrs)) + r")\s*,\s*[\"']a")
        return [l for l in src.splitlines() if pat.search(l)]
    assert _scan(src_bad) and not _scan(src_ok)


def test_require_rows_keeps_a_vanished_stream_LOUD(tmp_path):
    """The conversion had to preserve loudness in BOTH directions. About half the readers here
    used to `open()` the file, so a missing log raised; `read_rows` returns `[]`, which for
    `tau_h_rederive` would mean deriving over a run that contributes nothing and reporting it
    as a run with zero re-scoreable checks. Those sites take `require_rows`."""
    p = tmp_path / "harvest_log.jsonl"
    assert R.read_rows(p) == []
    with pytest.raises(R.MissingStreamError, match=r"harvest_log\.<nnn>\.jsonl"):
        R.require_rows(p)
    rows = _harvest_rows(200)
    _write_segmented(p, rows)                     # finalized: no plain file at all
    assert not p.exists()
    assert R.require_rows(p) == rows               # ...and it is still found
