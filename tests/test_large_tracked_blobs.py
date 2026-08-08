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

The assertions:
  1. every over-threshold tracked path is covered by an entry (new bulk -> red);
  2. no entry is dead — an entry matching nothing over threshold is a line nobody
     classified, and a rotted allowlist is how the covering assertion goes vacuous;
  3. no media/archive bulk at any size, tracked or LFS — the one half of the policy that
     needs no allowlist because the answer is never yes;
  4. every tracked file's EXTENSION is declared — text-by-nature, or an opaque/binary
     class allowed under one named path prefix with a reason. See "WHY 4 EXISTS".

WHY 4 EXISTS (the gap 3 leaves open). Assertion 3 is a DENYLIST, and a denylist of media
extensions is only as good as the last time someone thought about image formats: `.avif`,
`.heic`, `.jxl`, `.webm`, `.svg`, `.qoi` and `.tga` all sailed through it, and the hazard
this guard exists for — a sloppy run checking in a thousand renders — needs exactly one
format nobody listed. Assertion 4 inverts the polarity: the tracked tree is `.py/.rs/.json/
.jsonl/.md/.html/...` plus a handful of declared binaries (trained `.pt` weights, `.npz`
feature/embedding stores, `.jsonl.gz` run-record segments), so ANY extension outside that
set is red by default and a new image format is caught the first time one is staged,
whether or not anyone anticipated it. 3 is kept because it names media specifically and so
gives an unambiguous message, and because it is asserted over paths 4 would let through if
a media extension were ever added to the text set (which 4's own source scan refuses).

SIZE IS MEASURED THROUGH LFS. `git cat-file -s :path` returns ~130 bytes for an
LFS-tracked file (the size of the POINTER), so a naive size scan reports a 96 MB
cache_manifest as tiny — precisely inverting the thing being guarded. Pointer blobs are
parsed for their `size` field instead.

THE PER-COMMIT 20 MB RULE IS A DIFFERENT INSTRUMENT AND IS NOT ENFORCED HERE. That rule
(CLAUDE.md) is about the aggregate TREE BYTES one commit adds — the working-tree size of
what gets tracked, an LFS-tracked file counted at its full content size and not at its
130-byte pointer — and it is a stop-and-ask, not a test. This file is the standing
population check: what may be large-and-tracked at all, and what may never be tracked at
any size.

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
#
# This is a DENYLIST and denylists rot — `.avif`/`.heic`/`.webm`/`.svg` were all absent
# until 2026-08-07. It is not the load-bearing assertion any more (TEXT_EXTS +
# BINARY_ALLOWLIST below are: everything undeclared is red); it stays because a media hit
# deserves a message that says "media", not "undeclared extension".
#
# `.gz` is NOT here, deliberately. `data/discovery/**/*.jsonl.gz` is the run-record segment
# format (tools/run_record.py) and is declared in BINARY_ALLOWLIST; a `.tar.gz` still lands
# on `.tar`... which `os.path.splitext` does not see, so `.tgz`/`.tbz2` are listed too.
MEDIA_EXTS = (
    # raster
    ".png", ".jpg", ".jpeg", ".jfif", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    ".avif", ".heic", ".heif", ".jxl", ".qoi", ".tga", ".dds", ".ico",
    ".ppm", ".pgm", ".pnm", ".pbm", ".exr", ".hdr", ".psd", ".xcf",
    # vector / document
    ".svg", ".pdf", ".eps",
    # video / audio
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv",
    ".mp3", ".wav", ".flac", ".ogg",
    # archives
    ".zip", ".tar", ".7z", ".rar", ".tgz", ".tbz2", ".xz", ".bz2", ".zst", ".iso", ".dmg",
)
IMAGE_EXTS = MEDIA_EXTS          # back-compat alias for the older name

# ---- assertion 4: the tracked tree's extension vocabulary -------------------------- #
# Text by nature — source, config, records, prose. Allowed at ANY path, because none of
# them can be a thousand renders. `.gz` is deliberately absent (see MEDIA_EXTS).
TEXT_EXTS = frozenset({
    ".py", ".pyi", ".rs", ".toml", ".lock", ".json", ".jsonl", ".md", ".txt", ".csv",
    ".html", ".css", ".js", ".yml", ".yaml", ".cfg", ".ini", ".sh", ".ps1", ".bat",
    ".cmd", ".map", ".ugr", ".sql", ".rst",
    ".complete",   # `<stage>_table.COMPLETE` — empty stage-done markers under data/atlas/
})

# Extensionless tracked files, by BASENAME. Same argument as TEXT_EXTS — each is a marker
# or a repo-config file with no extension for `os.path.splitext` to check, so the name is
# the declaration. A renamed blob cannot hide here without also claiming one of these names.
NO_EXT_BASENAMES = frozenset({
    ".gitignore", ".gitattributes", ".gitkeep", ".gitmodules", "LICENSE", "COMPLETE",
})

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
          "CORN ordinal heads — and since the 2026-08-08 retention pass this covers exactly "
          "TWO weights, not six: the live deployed scorer (whichever "
          "production_pins.ACTIVE_CKPT names — v11 since 2026-08-08) and the one rung below "
          "it (v10). v5..v9 de-tracked that day under ACTIVE+PREVIOUS retention "
          "(docs/design/storage_classes.md); their config.json/metrics.json records stay and "
          "are text, so they never reach this allowlist. Not GPU-reproducible, so there is "
          "no rebuild path for either of the two."),
    Entry("data/wallpaper_head/v3/model_best.pt", WEIGHTS,
          "LIVE cross-location wallpaper-quality head."),
    # data/wallpaper_head/v4/model_best.pt had a line here while it was a STAGED candidate.
    # It de-tracked on 2026-08-08: wallpaper_pins still points at v3, so under ACTIVE+PREVIOUS
    # retention v4 is a never-adopted candidate and not a critical final weight. Same shape as
    # render_mode_head/v2 above. Its run record is untracked working-tree data and unchanged.
    Entry("data/render_mode_head/v1/model_best.pt", WEIGHTS,
          "LIVE strange-mode (mining_v1) gate."),
    # data/render_mode_head/v2/model_best.pt had a line here while it was a live candidate.
    # It lost the winner rule on 2026-08-06 and was de-tracked the same day: a rejected
    # candidate is not a critical final weight, and this list is the policy's "essentially
    # only" set. Its run record is small and stays tracked without needing a line here.
    Entry("data/queries/scorer/v3_gvo/model_best.pt", WEIGHTS,
          "LIVE palette-preference ranker (pref-v3-gvo)."),

    # ---- grandfathered exceptions: the policy would exclude these ----------------------
    # Each is a POPULATION RECORD, not a weight and not bulk media. They are here because
    # the population they describe no longer exists (the v4..v7 derived chain was wiped in
    # exactly this way), not because the policy has an exemption for JSONL.
    Entry("data/discovery/", EXCEPTION,
          "Per-run discovery ledgers (harvest_log / prio_terms / maneuvers / pool / "
          "q4_candidates). ~160 MB across runs, LFS. Records of walks that cannot be "
          "re-walked. q4_candidates.jsonl (2026-08-03 on) is the record-and-rank "
          "store: every candidate above a low floor with its per-stage fate, which is "
          "a SUPERSET of harvest_log — it holds the below-tau_h and gated populations "
          "the harvest log has no row for."),
    Entry("data/v8/", EXCEPTION,
          "v8 build record: manifest + plan + cache_manifest (~148 MB, mostly LFS). The "
          "split assignment the LIVE deployed checkpoint was trained under."),
    # data/v9/ was one entry until 2026-08-08 ("v9 build record, same shape as v8's, ~152
    # MB, LFS"). Its plan.jsonl + cache_manifest.jsonl were the only OVER-THRESHOLD files
    # under it and both were de-tracked that day — byte-reproducible from v8's manifest, and
    # the v10 recipe-parity gate that read the plan was retired with the aug-cache trees.
    # The line went because assertion 2 (no dead entry) is what removes a spent exception,
    # and it took v9's LFS rule with it: eval_scores_v9.jsonl is 281 KB, was LFS only for
    # uniformity with v10's, and had nothing over 1 MiB left under data/v9/ to sit under.
    Entry("data/v10/", EXCEPTION,
          "v10 build record: v8's manifest appended with 1,267 maneuver-view locations "
          "(8,382 x 24 slots, ~180 MB, LFS). Carries the split assignment for the third "
          "eval instrument (maneuver_uniform_v1), which exists nowhere else."),
    Entry("data/v11/", EXCEPTION,
          "v11 build record — and it is ONE file, which is the difference from v8/v9/v10 "
          "above. v11's manifest, plan and cache_manifest are all bulk and out-of-tree "
          "(artifacts._is_v11_build_rows), because the split is a SEEDED randomized draw "
          "and so rebuilds from the committed corpus rather than being restored. What is "
          "tracked is eval_scores_v11.jsonl (1.4 MB, LFS): 2,860 eval locations x the v11 "
          "and v10 cutpoint probabilities on identical tiles. It is a GPU eval's frozen "
          "output, not derived rows — a later keeper cut or t_good derivation re-cuts from "
          "it without re-scoring, and re-scoring it needs both checkpoints and the "
          "out-of-tree canonical tiles."),
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
    Entry("data/render_mode_corpus/batches/", EXCEPTION,
          "Render-mode (mining) batch images.jsonl: the render coords, mode, mode_params "
          "and colour recipe the committed human tiers dereference — carried IN-ROW "
          "precisely because the previous corpus kept them in an untracked file and its "
          "1500 tiers are now permanently orphaned. Larger per row than the wallpaper "
          "batches by the mode block; that block is the difference between a label and a "
          "label nobody can join."),
    Entry("data/orbital/screen_pool.jsonl", EXCEPTION,
          "Orbital screen pool: the scored candidate population behind the mode gates."),
]


@dataclass(frozen=True)
class BinaryEntry:
    """One opaque-content class: an extension, allowed only under `prefix`.

    Scoped by PATH as well as extension on purpose. `.npz` is a fine thing for a run to
    emit next to its ledger and a terrible thing to accept anywhere — a per-extension
    blanket pass would let a fresh `scratch`-shaped directory of them into the index
    unnoticed, which is the shape of every bulk incident this repo has had."""
    prefix: str
    ext: str
    reason: str
    forward: bool = False    # declared before the first file lands; see size_guard.Entry

    def covers(self, rel: str) -> bool:
        return rel.startswith(self.prefix) and rel.lower().endswith(self.ext)


# Every tracked extension that is NOT text-by-nature must match one of these.
BINARY_ALLOWLIST = [
    BinaryEntry("data/classifier/", ".pt",
                "CORN ordinal head weights — the live scorer (v11) + the one rollback rung "
                "(v10). v5..v9 de-tracked 2026-08-08 (ACTIVE+PREVIOUS retention). Not "
                "GPU-reproducible."),
    BinaryEntry("data/wallpaper_head/", ".pt",
                "Wallpaper-quality head weight — v3, the live head (wallpaper_pins). v4 was "
                "staged and de-tracked 2026-08-08 under ACTIVE+PREVIOUS retention."),
    BinaryEntry("data/render_mode_head/", ".pt",
                "Render-mode (strange-mode) gate weight — v1, whose training corpus is "
                "gone, so it cannot be retrained at all."),
    BinaryEntry("data/queries/scorer/", ".pt",
                "Palette-preference ranker weight (pref-v3-gvo)."),
    BinaryEntry("data/library_embeddings/", ".npz",
                "Prospect-library CLIP embeddings — the immutable base store; regenerable "
                "only value-approximately."),
    BinaryEntry("data/discovery/", ".npz",
                "Per-run `distinct_looks.npz` — the order-dependent near-dup look tally. "
                "Float arrays, so JSONL would be lossy AND larger. `outcome_feats*.npz` "
                "was the entry's original subject and LEFT on 2026-08-08: it is the "
                "ledger's derived sidecar, not the record, so it is bulk() and "
                "out-of-tree (tools/atlas/recompute_outcome_feats.py rebuilds it)."),
    BinaryEntry("data/atlas/", ".npz",
                "Round-1/2 arm embeddings + `distinct_looks.npz`: the frozen vectors the "
                "atlas arms' distinctness verdicts were taken on."),
    BinaryEntry("data/discovery/", ".jsonl.gz",
                "Rotated run-record segments (`<stem>.NNN.jsonl.gz`, tools/run_record.py). "
                "The ONE tracked-compressed class: LFS ships objects raw, so gzip here is "
                "8-11x off the bytes both the LFS remote and the 20 MB tree-byte commit "
                "rule count. Content is JSONL — `run_record.iter_rows` reads it as text. "
                "FORWARD: the segmenting writer landed 2026-08-07 and no run has committed "
                "since, so nothing matches yet; the .gitattributes LFS rule for exactly "
                "this glob is already in the tree, which is what makes the write imminent "
                "rather than hypothetical.", forward=True),
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
    of `git lfs ls-files -s`'s human-rounded "34 MB".

    THE PAIRING MUST SURVIVE A MISSING NAME. `git lfs ls-files` reports HEAD's LFS files,
    so a path staged for deletion is still NAMED here while `:path` no longer resolves in
    the index — `git cat-file --batch` answers `<spec> missing` with no blob, and a parse
    that only advanced its name cursor on a blob header shifted every subsequent size onto
    the wrong path. It failed SILENTLY and in the safe-looking direction: the last two
    names simply got no size, so two multi-MB weights fell back to their 133-byte pointer
    and the inventory reported them as tiny. Found 2026-08-03 by staging the deletion of
    `data/v8/{plan,cache_manifest}.jsonl`, whose only crime was being LFS and deleted.
    Every response now advances the cursor, and the count is asserted."""
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
        parts = body[i].split()
        if len(parts) == 3 and parts[1] == "blob":
            # "<sha> blob <n>" header, then the pointer text, then a blank line.
            for ln in body[i + 1:i + 6]:
                if ln.startswith("size "):
                    out[names[idx]] = int(ln.split()[1])
                    break
            idx += 1
        elif parts and parts[-1] in ("missing", "ambiguous"):
            # Named by `git lfs ls-files` (from HEAD) but not resolvable in the index —
            # a staged deletion. No size, but the cursor MUST advance or every name after
            # this one is credited with another path's size.
            idx += 1
        i += 1
    assert idx == len(names), (
        f"cat-file returned {idx} responses for {len(names)} LFS names — the pointer-size "
        f"pairing is positional and has desynchronized; sizes below would be attributed to "
        f"the wrong paths")
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
    offenders = sorted(p for p in SIZES if p.lower().endswith(MEDIA_EXTS))
    assert not offenders, (
        f"{len(offenders)} image/archive path(s) are tracked — they do not belong in the "
        f"repo even in LFS:\n  "
        + "\n  ".join(f"{p} ({SIZES[p] / 1048576:.2f} MiB)" for p in offenders[:25])
        + ("" if len(offenders) <= 25 else f"\n  ... and {len(offenders) - 25} more"))


def _undeclared_extension(rel: str) -> str | None:
    """The extension of `rel` if nothing declares it, else None.

    NOT `rel.endswith(...)`: `foo.png.json` is JSON and `foo.json.png` is a PNG, and only
    splitext gets both right. Case-folded, because Windows made `arm1_table.COMPLETE`."""
    base = rel.rsplit("/", 1)[-1]
    ext = ("." + base.rsplit(".", 1)[1].lower()) if "." in base[1:] else ""
    if not ext:
        return None if base in NO_EXT_BASENAMES else f"<no extension: {base}>"
    if ext in TEXT_EXTS:
        return None
    if any(e.covers(rel) for e in BINARY_ALLOWLIST):
        return None
    return ext


def test_every_tracked_extension_is_declared():
    """The ALLOWLIST half of the media policy, and the one that survives a format nobody
    listed. A tracked file is either text-by-nature (TEXT_EXTS, any path) or an opaque
    class declared under one prefix with a reason (BINARY_ALLOWLIST) — everything else is
    red on the FIRST file, which is what makes it a guard against a thousand of them."""
    offenders = sorted((p for p in SIZES if _undeclared_extension(p)),
                       key=lambda p: (_undeclared_extension(p), p))
    kinds = sorted({_undeclared_extension(p) for p in offenders})
    assert not offenders, (
        f"{len(offenders)} tracked path(s) of {len(kinds)} undeclared kind(s) {kinds}:\n  "
        + "\n  ".join(f"{p} ({SIZES[p] / 1048576:.2f} MiB)" for p in offenders[:25])
        + ("" if len(offenders) <= 25 else f"\n  ... and {len(offenders) - 25} more")
        + "\nThe tracked tree is source, config, records and a handful of declared "
          "binaries. If this is bulk it belongs under ARTIFACTS_ROOT "
          "(tools/corpus/artifacts.py); if it is a new legitimate class, add a TEXT_EXTS "
          f"or BINARY_ALLOWLIST line in {Path(__file__).name} saying what it is.")


def test_no_media_extension_is_declared_text():
    """Guard the guard, in the one direction that would silently disarm assertion 4: the
    cheapest way to make a red `.png` go green is to append it to TEXT_EXTS, which reads
    like every other line there. Assertion 3 would still catch `.png` itself — this
    catches the case where the two lists disagree at all."""
    overlap = sorted(TEXT_EXTS & set(MEDIA_EXTS))
    assert not overlap, (f"{overlap} are in TEXT_EXTS AND MEDIA_EXTS — a media extension "
                         f"declared text-by-nature disarms the allowlist assertion")
    bad = sorted(e for e in BINARY_ALLOWLIST if e.ext in MEDIA_EXTS)
    assert not bad, (f"BINARY_ALLOWLIST grants media extensions: "
                     f"{[(e.prefix, e.ext) for e in bad]}")


@pytest.mark.parametrize("entry", [e for e in BINARY_ALLOWLIST if not e.forward],
                         ids=lambda e: f"{e.prefix}*{e.ext}")
def test_no_binary_allowlist_entry_is_dead(entry):
    """Same argument as test_no_allowlist_entry_is_dead: an opaque-content grant that
    covers nothing is an unreviewed standing permission. `forward=True` entries are exempt
    — emptiness alone cannot tell not-yet-written from dead, and the flag is where that
    judgement is recorded (same split as tools/audit/size_guard.py's Entry.forward)."""
    assert any(entry.covers(p) for p in SIZES), (
        f"DEAD binary grant: {entry.prefix!r} + {entry.ext!r} covers no tracked path. "
        f"Delete the line (the class it admitted is gone) or mark it forward=True.")


def test_every_lfs_tracked_path_is_declared_too():
    """LFS is not a side door. A path can be LFS-tracked and under 1 MiB (the shakedown's
    harvest_log is 71 KB), which assertion 1 would not see — but the .gitattributes rule
    is a standing invitation for it to grow."""
    lfs = sorted(_lfs_pointer_sizes())
    undeclared = [p for p in lfs if not any(e.covers(p) for e in ALLOWLIST)]
    assert not undeclared, ("LFS-tracked paths with no ALLOWLIST entry:\n  "
                            + "\n  ".join(undeclared))
