"""Policy guard: what is allowed to be LARGE and GIT-TRACKED, as an allowlist.

THE POLICY, stated once:

    Large tracked binaries should be **rare and narrow: essentially only critical final
    trained weights.** Images and similar bulk do not belong in the repo *even in LFS*.

WHY THIS IS A TEST AND NOT A HOOK. This replaces `tools/hooks/pre-commit`, a client-side
staged-blob size hook. A client-side hook is the wrong instrument for a standing policy
three ways: it fires during ordinary work (so it trains people to reach for
`--no-verify`), it is invisible when disabled (this checkout never had it installed —
the tracked source sat next to a README line saying so), and it can only see the blob in
front of it, never the population. A collected assertion is visible in one place, cannot
be bypassed without an edit that shows up in review, and reads the whole index.

WHY AN ALLOWLIST AND NOT A THRESHOLD. A bare "nothing over 1 MiB" would be red today and
would stay red, so it would be deleted. A bare threshold also cannot express the policy,
which is about WHAT the file is, not how big it is. So every over-threshold tracked path
must be covered by exactly one `Entry`, and each entry declares whether the policy
SANCTIONS it (`WEIGHTS`) or it is a GRANDFATHERED exception carried with a reason. The
costs are lopsided in the right direction: forgetting to declare a new weight costs a red
build, which is cheap and immediate; a silently-accepted 100 MB blob costs a repo.

The three assertions:
  1. every over-threshold tracked path is covered by an entry (new bulk -> red);
  2. no entry is dead — an entry matching nothing over threshold is a line nobody
     classified, and a rotted allowlist is how the covering assertion goes vacuous;
  3. no image bulk at any size, tracked or LFS — the one half of the policy that needs
     no allowlist because the answer is never yes.

SIZE IS MEASURED THROUGH LFS. `git cat-file -s :path` returns ~130 bytes for an
LFS-tracked file (the size of the POINTER), so a naive size scan reports a 96 MB
cache_manifest as tiny — precisely inverting the thing being guarded. Pointer blobs are
parsed for their `size` field instead.

  uv run pytest tests/test_large_tracked_blobs.py
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1 MiB. Same number the deleted pre-commit hook used and the same one
# `tools/audit/size_guard.py` flags a working-tree file at, so "large" means one thing
# across the two guards (that one scans the WORKING TREE for bloat; this one scans the
# INDEX for policy).
LARGE = 1 * 1024 * 1024

# Bulk that is never acceptable in the index, at any size — the policy's second sentence.
# Extensions, not a size rule: one tracked 40 KB PNG is a precedent, and precedent is what
# the 358 MB of dead binary weight in this repo's history was made of.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
              ".mp4", ".mov", ".avi", ".zip", ".tar", ".gz", ".7z")

WEIGHTS = "WEIGHTS"        # sanctioned by the policy: a critical final trained weight
EXCEPTION = "EXCEPTION"    # grandfathered: the policy would exclude it; kept, with a reason


@dataclass(frozen=True)
class Entry:
    """One allowlist line. `prefix` is a repo-relative POSIX path or path prefix."""
    prefix: str
    kind: str
    reason: str

    def covers(self, rel: str) -> bool:
        return rel == self.prefix or rel.startswith(self.prefix)


# --------------------------------------------------------------------------- #
# The allowlist. Everything over LARGE in the index must match exactly one line.
# --------------------------------------------------------------------------- #
ALLOWLIST = [
    # ---- sanctioned: critical final trained weights (the policy's "essentially only") --
    Entry("data/classifier/", WEIGHTS,
          "CORN ordinal heads v5..v9: the live deployed scorer plus its rollback anchors. "
          "Not GPU-reproducible, so there is no rebuild path."),
    Entry("data/wallpaper_head/v3/model_best.pt", WEIGHTS,
          "LIVE cross-location wallpaper-quality head."),
    Entry("data/render_mode_head/v1/model_best.pt", WEIGHTS,
          "LIVE strange-mode (mining_v1) gate."),
    Entry("data/queries/scorer/v3_gvo/model_best.pt", WEIGHTS,
          "LIVE palette-preference ranker (pref-v3-gvo)."),

    # ---- grandfathered exceptions: the policy would exclude these ----------------------
    # Each is a POPULATION RECORD, not a weight and not bulk media. They are here because
    # the population they describe no longer exists (the v4..v7 derived chain was wiped in
    # exactly this way), not because the policy has an exemption for JSONL.
    Entry("data/discovery/", EXCEPTION,
          "Per-run discovery ledgers (harvest_log / prio_terms / maneuvers / pool). "
          "~160 MB across runs, LFS. Records of walks that cannot be re-walked."),
    Entry("data/v8/", EXCEPTION,
          "v8 build record: manifest + plan + cache_manifest (~148 MB, mostly LFS). The "
          "split assignment the LIVE deployed checkpoint was trained under."),
    Entry("data/v9/", EXCEPTION,
          "v9 build record, same shape as v8's (~152 MB, LFS)."),
    Entry("data/v10/", EXCEPTION,
          "v10 build record: v8's manifest appended with 1,267 maneuver-view locations "
          "(8,382 x 24 slots, ~180 MB, LFS). Carries the split assignment for the third "
          "eval instrument (maneuver_uniform_v1), which exists nowhere else."),
    Entry("data/palettes/", EXCEPTION,
          "pool_colormaps.json (20 MB) + palette_features.json (1.8 MB): the harvested "
          "palette pool and its feature table."),
    Entry("data/library_embeddings/embeddings.npz", EXCEPTION,
          "Prospect-library CLIP embeddings; regenerate only value-approximate under a "
          "verdict-sensitive threshold (canaried in test_tracked_artifacts.py)."),
    Entry("data/label_corpus/batches/", EXCEPTION,
          "Label-batch images.jsonl: the render coords the committed human labels "
          "dereference. Labels without their referent are useless."),
    Entry("data/wallpaper_corpus/batches/", EXCEPTION,
          "Wallpaper-batch images.jsonl and progress ledgers."),
    Entry("data/orbital/screen_pool.jsonl", EXCEPTION,
          "Orbital screen pool: the scored candidate population behind the mode gates."),
]


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def _run(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout


def _lfs_pointer_sizes() -> dict[str, int]:
    """{path: true object size} for LFS-tracked paths, parsed from the POINTER blob.

    Exact by construction — the pointer's own `size` line — rather than parsed back out
    of `git lfs ls-files -s`'s human-rounded "34 MB"."""
    names = [ln.strip() for ln in _run(["git", "lfs", "ls-files", "-n"]).splitlines()
             if ln.strip()]
    if not names:
        return {}
    spec = "".join(f":{n}\n" for n in names)
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=REPO_ROOT,
                          input=spec, capture_output=True, text=True, check=True)
    out, body = {}, proc.stdout.splitlines()
    i, idx = 0, 0
    while i < len(body) and idx < len(names):
        # "<sha> blob <n>" header, then the pointer text, then a blank line.
        if len(body[i].split()) == 3 and body[i].split()[1] == "blob":
            size = None
            for ln in body[i + 1:i + 6]:
                if ln.startswith("size "):
                    size = int(ln.split()[1])
                    break
            if size is not None:
                out[names[idx]] = size
            idx += 1
        i += 1
    return out


def tracked_sizes() -> dict[str, int]:
    """{repo-relative path: size in bytes} over every tracked file, LFS resolved."""
    paths = [p for p in _run(["git", "ls-files"]).splitlines() if p.strip()]
    spec = "".join(f":{p}\n" for p in paths)
    proc = subprocess.run(["git", "cat-file", "--batch-check"], cwd=REPO_ROOT,
                          input=spec, capture_output=True, text=True, check=True)
    sizes = {}
    for path, line in zip(paths, proc.stdout.splitlines()):
        parts = line.split()
        sizes[path] = int(parts[2]) if len(parts) == 3 and parts[1] == "blob" else 0
    sizes.update(_lfs_pointer_sizes())
    return sizes


SIZES = tracked_sizes()
LARGE_PATHS = sorted(p for p, n in SIZES.items() if n >= LARGE)


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #
def test_the_index_is_readable_at_all():
    """Guard the guard: an empty inventory would make every assertion below vacuous."""
    assert len(SIZES) > 500, f"only {len(SIZES)} tracked paths — inventory did not run"
    assert LARGE_PATHS, "no tracked path over 1 MiB at all — inventory is not measuring"


@pytest.mark.parametrize("path", LARGE_PATHS)
def test_every_large_tracked_blob_is_declared(path):
    hits = [e for e in ALLOWLIST if e.covers(path)]
    assert hits, (
        f"UNDECLARED large tracked blob ({SIZES[path] / 1048576:.1f} MiB):\n"
        f"    {path}\n"
        f"Policy: large tracked binaries are rare and narrow — essentially only critical "
        f"final trained weights; images and similar bulk do not belong in the repo even in "
        f"LFS.\nIf this is a trained weight, add a WEIGHTS entry to ALLOWLIST in "
        f"{Path(__file__).name}. If it is bulk, it belongs under ARTIFACTS_ROOT "
        f"(tools/corpus/artifacts.py), not in the index.")
    assert len(hits) == 1, f"{path} is covered by {len(hits)} entries: " \
                           f"{[e.prefix for e in hits]} — prefixes must not overlap"


@pytest.mark.parametrize("entry", ALLOWLIST, ids=lambda e: e.prefix)
def test_no_allowlist_entry_is_dead(entry):
    """An entry matching nothing over threshold is a line nobody classified — and a
    rotted allowlist is how assertion 1 goes quietly vacuous."""
    assert any(entry.covers(p) for p in LARGE_PATHS), (
        f"DEAD allowlist entry: {entry.prefix!r} covers no tracked blob >= 1 MiB. "
        f"Delete the line (the exception it granted is spent).")


def test_no_image_or_archive_bulk_is_tracked_at_any_size():
    """The half of the policy that needs no allowlist: media never belongs in the index,
    LFS included. Asserted over EVERY tracked path, not just the large ones — one small
    committed PNG is the precedent, and precedent is what a 358 MB history is made of."""
    offenders = sorted(p for p in SIZES if p.lower().endswith(IMAGE_EXTS))
    assert not offenders, (
        "images/archives are tracked — they do not belong in the repo even in LFS:\n  "
        + "\n  ".join(f"{p} ({SIZES[p] / 1048576:.2f} MiB)" for p in offenders))


def test_every_lfs_tracked_path_is_declared_too():
    """LFS is not a side door. A path can be LFS-tracked and under 1 MiB (the shakedown's
    harvest_log is 71 KB), which assertion 1 would not see — but the .gitattributes rule
    is a standing invitation for it to grow."""
    lfs = sorted(_lfs_pointer_sizes())
    undeclared = [p for p in lfs if not any(e.covers(p) for e in ALLOWLIST)]
    assert not undeclared, ("LFS-tracked paths with no ALLOWLIST entry:\n  "
                            + "\n  ".join(undeclared))
