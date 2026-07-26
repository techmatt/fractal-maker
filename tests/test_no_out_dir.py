"""Tripwire: the repo-root `out/` directory MUST NOT exist and MUST NOT be created.

`out/` was renamed to `scratch/` and every generated artifact now lands under the
single disposable `scratch/` tree (see CLAUDE.md, the generated-output convention).
The rename swept the tracked surface mechanically; this tripwire covers the rest by
substitution. Any code path that still writes `out/…` recreates the directory the
moment it runs, and the next invocation of the suite catches it here — no long render
needed to prove the sweep complete.

The assertion is deliberately about *existence*, not about a static grep: a writer
that constructs the path dynamically would slip a source scan but cannot avoid
materializing the directory. When `out/` is present the failure lists what it holds,
which names what was written and points at the offending writer.

Light lane — pathlib only, no git/GPU/binary/corpus — so it runs under bare `pytest`
alongside the other canary tripwires.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out"

# Cap the file listing so a large stray tree doesn't flood the failure message.
_MAX_LISTED = 20


def _sample_contents(root: Path) -> str:
    """Repo-relative paths of files under `root` (up to _MAX_LISTED), so the failure
    message points at what was written — the filename usually names the writer."""
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    listed = files[:_MAX_LISTED]
    lines = [f"    {p.relative_to(REPO_ROOT).as_posix()}" for p in listed]
    if len(files) > _MAX_LISTED:
        lines.append(f"    … and {len(files) - _MAX_LISTED} more")
    return "\n".join(lines) if lines else "    (directory exists but is empty)"


def test_out_dir_does_not_exist():
    """`out/` is retired. Its reappearance means a writer regressed to the old path."""
    assert not OUT_DIR.exists(), (
        "OUT/ TRIPWIRE: the retired `out/` directory has reappeared at the repo root.\n"
        "It was renamed to `scratch/`; a code path is still writing there. Its contents:\n"
        f"{_sample_contents(OUT_DIR)}\n"
        "Redirect that writer to `scratch/<subcommand>/` (the disposable tree), then\n"
        "delete `out/`. Do NOT add an `out/` .gitignore rule to hide it — that re-creates\n"
        "the exact ambiguity the rename removed."
    )
