"""`docs/` is SOURCE ONLY, and `docs/findings/` stays retired.

Two failures this guard exists to stop, both of which actually happened:

  1. **`docs/findings/` was retired with no tripwire.** Commit 38e807f (2026-07-25) moved
     the stragglers to `docs/design/` and declared the directory retired — but nothing
     enforced it, and `CLAUDE.md` still said "findings text goes to `docs/findings/`". Five
     later runs recreated the directory and wrote six files into it before anyone noticed.
     A retirement that only lives in a commit message is a suggestion.

  2. **Generated sheets were parked in `docs/design/` and gitignored there.** Five contact
     -sheet PNGs (4 of them >= 1 MiB) were written next to the rubric doc for reading
     convenience and hidden with per-file `.gitignore` lines. `tools/audit/size_guard.py`
     scans the FILESYSTEM, not `git ls-files` — precisely so gitignored bulk can't hide —
     so `tests/test_repo_size_guard.py` went red on 4 uncovered violators and STAYED red.
     A permanently-red lane erodes every tripwire that lives in it.

The rule that kills both: **every file under `docs/` is git-tracked.** Docs are prose the
repo keeps; a generated artifact goes to `scratch/` (the generated-output convention in
`CLAUDE.md`) and is rebuilt by its committed builder. Nothing in `docs/` may be ignored or
untracked, so there is nowhere for regenerable bulk to hide there.

Scope note for the path check: it greps tracked **source** files (`*.py`, `*.rs`, `*.html`,
`*.toml`), because that is where a write target lives. Prose may of course still mention the
old directory by name — describing what used to exist, or stating that it is retired, is not
a write path. `CLAUDE.md` (the instruction file whose stale line caused the recurrence) gets
its own POSITIVE check instead: it must route findings text at `docs/design/`.

Runs under default `pytest`: `git` + a filesystem walk, nothing else.
"""
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
RETIRED = "docs/findings"
# Where a path can actually be WRITTEN from. Prose .md files are deliberately out of
# scope — see the module docstring.
SOURCE_SUFFIXES = (".py", ".rs", ".html", ".toml")


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.splitlines()


@pytest.fixture(scope="module")
def tracked_docs():
    return {p for p in _git("ls-files", "docs")}


def test_docs_findings_is_retired():
    """The retired directory does not exist. Findings/analysis text goes to docs/design/."""
    p = REPO_ROOT / "docs" / "findings"
    assert not p.exists(), (
        f"{RETIRED}/ was recreated (contents: "
        f"{sorted(x.name for x in p.iterdir())}). It is retired — move these to "
        f"docs/design/ and point whatever wrote them there."
    )


def test_no_source_file_targets_the_retired_directory():
    """No tracked source file names docs/findings as a path — that is how it came back."""
    # This file is the one legitimate exception: a guard has to name what it forbids.
    self_rel = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    offenders = []
    for rel in _git("ls-files"):
        if not rel.endswith(SOURCE_SUFFIXES) or rel == self_rel:
            continue
        p = REPO_ROOT / rel
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if RETIRED in line or '"docs" / "findings"' in line or '"docs", "findings"' in line:
                offenders.append(f"{rel}:{i}: {line.strip()[:110]}")
    assert not offenders, (
        "source files still point at the retired docs/findings/:\n  "
        + "\n  ".join(offenders)
        + "\nRetarget them at docs/design/."
    )


def test_claude_md_routes_findings_text_to_docs_design():
    """The recurrence's root cause: CLAUDE.md still told every run to write findings into
    the directory that had just been retired. Assert the instruction points at the live
    location — positively, so the retirement can still be NAMED in the same sentence."""
    t = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Findings/analysis text goes to `docs/design/`" in t, (
        "CLAUDE.md no longer routes findings/analysis text at docs/design/")
    assert "text goes to `docs/findings/`" not in t, (
        "CLAUDE.md still routes findings text at the retired docs/findings/")


def test_every_file_under_docs_is_tracked(tracked_docs):
    """docs/ is source only. An untracked or gitignored file here is either bulk that
    belongs in scratch/ (and will make the repo-size guard red) or a doc someone forgot
    to `git add`."""
    if not DOCS.exists():
        pytest.skip("no docs/ tree")
    stray = []
    for p in DOCS.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel not in tracked_docs:
            stray.append(f"{rel}  ({p.stat().st_size / 1024:.0f} KB)")
    assert not stray, (
        "untracked/ignored files under docs/ — docs/ is source only:\n  "
        + "\n  ".join(sorted(stray))
        + "\nIf it is a generated view (contact sheet, render, plot), write it under "
          "scratch/<builder>/ and keep the builder committed; do NOT add a docs/ "
          "ignore rule. If it is prose, `git add` it."
    )


def test_the_tracked_docs_set_is_nonempty(tracked_docs):
    """Guard the guard: an empty docs/ would make the tracked-ness assertion vacuous."""
    assert len(tracked_docs) > 10, f"only {len(tracked_docs)} tracked files under docs/"
