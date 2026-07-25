"""Storage-class helper: name what a file IS at the write site, so a module cannot
write a path without declaring its durability class.

Four times in one week a module treated its own output as durable while writing it
somewhere that guarantees deletion — because a name like `out/` invited "my output
matters" and the ignore rule that contradicts it lives in a different file written by a
different hand (the rename to `scratch/` refutes that invitation). This module removes
the ambiguity: every write goes through the function that names its class, and
`durable()` refuses — loudly, at write time — to hand back a path that git would throw
away.

The four classes (see the durability contract):

    scratch(...)  disposable        -> scratch/: gitignored, rm-rf-safe,
                                       NO durability claims. Cheap or free to rebuild.
    bulk(rel)     bulk-regenerable  -> out-of-tree via the ARTIFACTS_ROOT resolver.
                                       Expensive but deterministic to rebuild.
    durable(rel)  durable           -> data/, git-tracked. Records a population that no
                                       longer exists; impossible to rebuild. ASSERTED
                                       not-gitignored on every call.
    (vendored)    vendored          -> a single derived NUMBER committed into config or
                                       code with provenance (the TAU_H_FIDELITY_BASE
                                       precedent). Not a runtime-written path, so it has
                                       no function here — you commit it by hand.

`bulk()` deliberately delegates to `tools/corpus/artifacts.py` (the ONE existing
ARTIFACTS_ROOT resolver) rather than building a second one.
"""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

# tools/paths.py -> parents[1] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse the single ARTIFACTS_ROOT resolver — do NOT reimplement it here.
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
import artifacts as _artifacts  # noqa: E402


class DurabilityError(RuntimeError):
    """A path declared `durable()` is gitignored — writing it would silently lose the
    data. Raised at the write site, naming the path and its class."""


def _rel(rel) -> str:
    """Normalize to a forward-slash repo-relative string (reject absolute escapes)."""
    s = str(rel).replace("\\", "/").lstrip("/")
    while s.startswith("./"):
        s = s[2:]
    return s


@lru_cache(maxsize=4096)
def _is_gitignored(abspath: str) -> bool:
    """True iff git's ignore rules would exclude `abspath`. `git check-ignore` honors
    negation (`!/data/discovery/`), so a re-included durable path correctly reports
    False. Ignore status is fixed within a run, so the result is cached. If git is
    unavailable we cannot prove the path safe, so we treat it as NOT ignored (fail
    open) rather than block every durable write."""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", abspath],
            cwd=REPO_ROOT, capture_output=True,
        )
    except (OSError, FileNotFoundError):
        return False
    # exit 0 = ignored, 1 = not ignored, 128 = error (e.g. not a git repo) -> not proven ignored.
    return proc.returncode == 0


def scratch(*parts) -> Path:
    """Disposable path under the gitignored `scratch/` tree. rm-rf-safe; no durability
    claim. Accepts path components: `scratch("atlas", "sheet.png")`."""
    return REPO_ROOT.joinpath("scratch", *[str(p) for p in parts])


def bulk(rel) -> Path:
    """Bulk-regenerable artifact: resolve a repo-relative path through the ARTIFACTS_ROOT
    resolver, so a relocated family lands out-of-tree and a rebuild never re-materializes
    the file-count bomb in the source tree. Non-relocated paths resolve in-tree."""
    return _artifacts.resolve(_rel(rel))


def durable(rel, *, mkparents: bool = False) -> Path:
    """Durable artifact under `data/`: return its absolute path, but FIRST assert git
    would keep it. If a `.gitignore` rule (with no re-include) would exclude it, raise
    DurabilityError naming the path and class — so the mistake surfaces the moment the
    write is attempted, not months later when the file is needed and gone.

    `mkparents=True` creates the parent directory (after the assertion passes)."""
    r = _rel(rel)
    abspath = REPO_ROOT / r
    if _is_gitignored(str(abspath)):
        raise DurabilityError(
            f"durable() path is GITIGNORED and would be silently discarded:\n"
            f"    path : {r}\n"
            f"    class: durable (must be git-tracked under data/)\n"
            f"Either add a .gitignore negation re-including it, or it is not durable — "
            f"use scratch() (disposable) or bulk() (regenerable, out-of-tree) instead."
        )
    if mkparents:
        abspath.parent.mkdir(parents=True, exist_ok=True)
    return abspath
