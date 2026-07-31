"""Repo-size guard: no large-in-tree bloat without an explicit registry entry.

The standing constraint is that the working tree stays ~what git tracks — source +
irreplaceable metadata + `scratch/`. Anything large in-tree needs a written-down reason.
`tools/audit/size_guard.py` scans the *filesystem* (not `git ls-files`: a gitignored
file can bloat the tree while invisible to git) and flags every file >= 1 MiB and
every many-small-file directory >= ~100 MB, then checks each flagged violator against
the `REGISTRY` allowlist.

Two hard assertions:

  * Any flagged violator not covered by a registry entry. This catches NEW bloat the
    moment it lands: add 300 MB of un-registered crops and this goes red, naming the
    path. To make it green you either delete the bloat or add a deliberate registry
    line stating why it stays. (Proven to go red on purpose: drop a >=1 MiB file
    outside the excluded prefixes → fail → remove → green.)

  * Any registry entry that covers no over-threshold content AND is not marked
    `forward=True`. This was a warning until 2026-07-31, on the argument that
    emptiness cannot distinguish not-yet-built from dead (`data/v8/` was legitimately
    empty before its build). `Entry.forward` now records that distinction explicitly —
    *can anything still write here* — so the premise is gone and the check is hard. It
    had been firing on 15 lines every run, which is how a soft red gets trained out.
    Fixing it is a one-line edit either way: prune the line, or mark it forward.

This runs under default `pytest`: filesystem walk + `git` only, no release binary,
no GPU, no corpus reads. Companion to `test_tracked_artifacts.py` (which guards
*de-tracking* of a static canary list) — this guards *bloat* of the live tree.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PATH = REPO_ROOT / "tools" / "audit" / "size_guard.py"

# Load size_guard.py by path (tools/ is not an installed package). Register in
# sys.modules before exec so its @dataclass field annotations resolve.
_spec = importlib.util.spec_from_file_location("size_guard", _MOD_PATH)
size_guard = importlib.util.module_from_spec(_spec)
sys.modules["size_guard"] = size_guard
_spec.loader.exec_module(size_guard)


@pytest.fixture(scope="module")
def result():
    return size_guard.check_registry(size_guard.scan(REPO_ROOT))


def test_registry_nonempty():
    """Guard the guard: an emptied REGISTRY would make the coverage assertion pass
    vacuously (nothing to fail against)."""
    assert size_guard.REGISTRY, "REGISTRY is empty — the size guard would pass vacuously"


def test_no_uncovered_violators(result):
    """HARD: every flagged large file / dir is covered by a registry entry."""
    if result.uncovered:
        lines = "\n".join(
            f"    {size_guard.human(v.size):>9}  {v.rel}" for v in result.uncovered
        )
        pytest.fail(
            "REPO-SIZE GUARD TRIPPED: large-in-tree content with no registry entry:\n"
            f"{lines}\n"
            "New bulk landed in the working tree. Either move it out (regenerable ->\n"
            "artifacts, trained binary -> precious-store, dead -> trash) or, if it truly\n"
            "belongs in-tree, add a deliberate KEEP line to REGISTRY in\n"
            "tools/audit/size_guard.py stating why. Do NOT widen an existing prefix\n"
            "just to silence this."
        )


def _bulk_rels(res):
    return {v.rel for v in res.bulk_violators}


def test_bulk_rule_fires_on_file_count_alone(tmp_path):
    """Rule (c) catches the case rules (a)+(b) structurally cannot: MANY SMALL files.

    A cache of 2,500 x 10-byte crops is 25 KB — far under the 1 MiB per-file rule and the
    100 MB small-file-aggregate rule — yet it is precisely the traversal bomb
    storage_classes.md rule 5 forbids. Planted in a tmp tree so the real repo is untouched.
    """
    cache = tmp_path / "data" / "v9" / "aug_cache"
    cache.mkdir(parents=True)
    for i in range(size_guard.DIR_FILE_COUNT_THRESHOLD + 500):
        (cache / f"{i}.jpg").write_bytes(b"x" * 10)

    res = size_guard.scan(tmp_path)
    assert not res.file_violators, "no single file is over 1 MiB — rule (a) must stay quiet"
    assert not res.dir_violators, "25 KB of small files is under 100 MB — rule (b) must stay quiet"
    assert _bulk_rels(res) == {"data/v9/aug_cache/"}, res.bulk_violators
    assert res.bulk_violators[0].rules == ("files",)

    # ... and with no covering entry it is a HARD failure, not a warning.
    res = size_guard.check_registry(res, registry=[])
    assert [v.rel for v in res.uncovered] == ["data/v9/aug_cache/"]


def test_bulk_rule_fires_on_bytes_alone(tmp_path):
    """The other half of rule (c): few files, but the subtree is a data store."""
    field = tmp_path / "data" / "big_field"
    field.mkdir(parents=True)
    n = 3
    each = size_guard.DIR_BYTES_THRESHOLD // n + 1
    for i in range(n):
        with (field / f"f{i}.bin").open("wb") as f:
            f.truncate(each)

    res = size_guard.scan(tmp_path)
    assert _bulk_rels(res) == {"data/big_field/"}, res.bulk_violators
    assert res.bulk_violators[0].rules == ("bytes",)
    assert size_guard.check_registry(res, registry=[]).uncovered


def test_bulk_rule_reports_at_minimal_granularity(tmp_path):
    """A bulk subtree is reported at the leaf-most bulk dir, not at every ancestor —
    otherwise every registry entry would have to be written at a uselessly coarse prefix."""
    leaf = tmp_path / "data" / "corpus" / "batches" / "b1" / "crops"
    leaf.mkdir(parents=True)
    for i in range(size_guard.DIR_FILE_COUNT_THRESHOLD + 10):
        (leaf / f"{i}.jpg").write_bytes(b"x")

    res = size_guard.scan(tmp_path)
    assert _bulk_rels(res) == {"data/corpus/batches/b1/crops/"}, res.bulk_violators


def test_bulk_rule_stays_quiet_below_thresholds(tmp_path):
    """Guard the guard the other way: a normal source-sized tree flags nothing."""
    src = tmp_path / "tools" / "corpus"
    src.mkdir(parents=True)
    for i in range(200):
        (src / f"m{i}.py").write_bytes(b"x" * 4096)
    res = size_guard.scan(tmp_path)
    assert res.bulk_violators == []


def test_excluded_prefixes_are_not_bulk_flagged(tmp_path):
    """`scratch/` is disposable BY CONTRACT — rule (c) must not fire on it, or every
    routine render sweep would trip the guard."""
    s = tmp_path / "scratch" / "renders"
    s.mkdir(parents=True)
    for i in range(size_guard.DIR_FILE_COUNT_THRESHOLD + 10):
        (s / f"{i}.png").write_bytes(b"x")
    assert size_guard.scan(tmp_path).bulk_violators == []


def test_every_empty_entry_is_classified(result):
    """HARD: an entry covering nothing must say WHY — pruned, or marked forward.

    `res.stale` now means only "nobody classified this line", which is a defect with a
    one-line fix. The judgement it demands is *can anything still write here*: yes ->
    `forward=True` (a live forward declaration, e.g. data/guided_descend/, which the
    Rust `guided-descend --out` default repopulates); no -> delete the line, the way the
    v4..v7 cache lines were deleted."""
    if result.stale:
        stale = "\n".join(f"    {e.prefix:<40} [{e.label()}]" for e in result.stale)
        pytest.fail(
            f"{len(result.stale)} unclassified size-guard registry entr"
            f"{'y' if len(result.stale) == 1 else 'ies'} — no over-threshold content and "
            f"not marked as a live forward declaration:\n{stale}\n"
            "For each, ask: can anything still WRITE here?\n"
            "  yes -> keep the line and set forward=True, stating which writer and why;\n"
            "  no  -> delete the line (and its regeneration machinery), as the v4..v7\n"
            "         cache lines were deleted.\n"
            "Do NOT set forward=True just to make this green — that is how the warning\n"
            "this replaced stopped meaning anything."
        )


def test_forward_declarations_are_not_reported_stale(tmp_path):
    """The mechanism, on a synthetic tree: two identical empty entries, one marked
    forward, and only the unmarked one is stale. Without this the `forward` flag could
    silently stop being honoured and the hard check above would just go quiet."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_bytes(b"x" * 100)      # nothing over threshold
    registry = [
        size_guard.Entry("data/dead/", size_guard.RELOCATE, size_guard.TRASH, "ignored",
                         "producer retired"),
        size_guard.Entry("data/live/", size_guard.RELOCATE, size_guard.ARTIFACTS, "ignored",
                         "empty now; the builder still writes here", forward=True),
    ]
    res = size_guard.check_registry(size_guard.scan(tmp_path), registry=registry)
    assert [e.prefix for e in res.stale] == ["data/dead/"]
    assert [e.prefix for e in res.forward] == ["data/live/"]


def test_a_forward_entry_that_gains_content_is_neither(tmp_path):
    """A forward declaration is about the EMPTY state only: once its writer actually
    lands bulk there, the entry is covering content and drops out of both lists — the
    disposition it declared is now doing real work."""
    leaf = tmp_path / "data" / "live"
    leaf.mkdir(parents=True)
    (leaf / "blob.bin").write_bytes(b"x" * (size_guard.FILE_THRESHOLD + 1))
    registry = [
        size_guard.Entry("data/live/", size_guard.RELOCATE, size_guard.ARTIFACTS, "ignored",
                         "empty now; the builder still writes here", forward=True),
    ]
    res = size_guard.check_registry(size_guard.scan(tmp_path), registry=registry)
    assert res.uncovered == []
    assert res.stale == [] and res.forward == []


def test_forward_entries_are_the_minority_and_each_says_why(result):
    """Guard the guard from the other side: `forward` is an escape hatch, so it must
    stay a deliberate, argued minority rather than a blanket applied to the registry.
    Each marked entry must name its writer in the reason text (the word FORWARD)."""
    marked = [e for e in size_guard.REGISTRY if e.forward]
    assert marked, "no forward declarations — did the flag stop being used?"
    assert len(marked) < len(size_guard.REGISTRY) / 2, (
        f"{len(marked)}/{len(size_guard.REGISTRY)} entries are forward declarations; the "
        f"flag is meant to be the exception, not the default disposition")
    unargued = [e.prefix for e in marked if "FORWARD" not in e.reason]
    assert not unargued, (
        f"forward-marked entries whose reason does not state the live writer: {unargued}")
