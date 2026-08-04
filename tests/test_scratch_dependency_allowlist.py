"""Standing guard: the ledger of what in `scratch/` is unsafe to wipe.

`scratch/` is the one class whose contract GUARANTEES deletion, and twice now a wipe took
something a committed tool needed — the library look-seed's 168 vectors (declared `bulk()`
at a path that said `scratch/`) and the interior-band candidate population (`scratch()`,
and the population two registered corpus batches were drawn from). Both were recovered from
a trash directory rather than from their contract. The inventory that found them was a
one-off scan; this is that scan turned into a test, so the next one is found before the
wipe rather than after.

THE RULE. A committed non-test source file may not name a repo-relative `scratch/<path>`
as a STRING IN CODE unless the path is on the allowlist below, and every entry declares
four things: the path, the tool that owns it, whether committed code CONSUMES it (`INPUT`)
or merely writes it (`OUTPUT`), and how long it must survive. The `INPUT` rows are the
answer to "what can I delete from `scratch/` right now" — everything not on them is fair
game.

SCOPE, stated rather than implied — this guard is deliberately narrower than "every mention
of scratch":

  * STRINGS IN CODE ONLY. Docstrings and comments are excluded (a rotted `Reads:` line is a
    documentation bug, not a wipe hazard). Including them takes the population from 43 to
    259 and drowns the ledger in prose.
  * FOREIGN TREES ONLY. A module that names its OWN scratch tree — the family it declares
    via `paths.scratch(...)`, or one matching its own module/package name — is writing its
    own disposable output, which is the convention working. What this guard is looking for
    is one tool reaching into ANOTHER tool's scratch tree.
  * `paths.scratch(...)` CALLS ARE NOT FLAGGED. That is the declared form: it names the
    class at the write site and a refactor can find it. A hardcoded literal is the form
    that cannot, and it is the form every finding of the original scan took.
  * REPO-RELATIVE LITERALS ONLY. The pattern is anchored, which excludes two shapes that
    are not filesystem dependencies at all: `f"{run_dir}/scratch/queue.jsonl"` (a DISCOVERY
    RUN's own scratch, relocated out-of-tree by the artifacts resolver) and
    `"../../scratch/<x>/"` (a relative link inside a generated HTML page, six of those in
    `palette_extractor/`). Both are pinned by tests below.

  What that leaves uncovered, so it is a decision rather than a gap: a cross-MODULE constant
  reference (`FIELDS = BMB.FIELDS`) is invisible here, because the path never appears as a
  literal in the reading module. That is the well-factored case — one definition, in the
  module that owns it — and the durability map (`tools/audit/durability_map.py`) is what
  covers it.

Light lane: `git ls-files` + `ast`. No numpy/torch/GPU.

  uv run pytest tests/test_scratch_dependency_allowlist.py
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT = "INPUT"      # committed code CONSUMES this: unsafe to wipe before `lifetime`
OUTPUT = "OUTPUT"    # committed code only WRITES it: wipeable at any time


@dataclass(frozen=True)
class Entry:
    """One deliberate scratch dependency. `path` is matched exactly against the literal a
    module names, so a new sibling path under the same tree is a NEW line, not free cover."""
    path: str
    owner: str        # the tool that produces it
    kind: str         # INPUT | OUTPUT
    reason: str
    lifetime: str


# --------------------------------------------------------------------------- #
# The ledger. Seeded from the 2026-08-03 live-scratch inventory
# (scratch/scratch_read_scan.txt) and extended by this guard's own scan.
# --------------------------------------------------------------------------- #
ALLOWLIST = [
    # ---- INPUT: a committed tool consumes it. These are the wipe hazards. --------
    Entry("scratch/curation/morph_fields", "tools/curation/colored_clip.py", INPUT,
          "smooth-field cache the morph producer tags against; colored_clip writes it and "
          "morph_producer_tag reads it.",
          "regenerable — a wipe costs a re-render, nothing else"),
    Entry("scratch/curation/visual_dup", "scratchpad/visual_dup/embed.py (GONE)", INPUT,
          "morphology_dedup's --artifacts default: the clusters.json + sim matrices it "
          "reports over. Its producer was the load-bearing scratchpad module that vanished "
          "and cost a formula sweep to recover (CLAUDE.md, the scratch-tier clause).",
          "until morphology_dedup is repointed at a durable store"),
    Entry("scratch/deep_centers/preview_p58.png", "tools/sourcing/deep_center_finder.py", INPUT,
          "one of q4_sweep_validation's FRAMES: the rendered preview a validation frame is "
          "measured against.", "until q4_sweep_validation retires"),
    Entry("scratch/deep_centers/ladder_mis/fw_1e_8.png", "tools/sourcing/deep_center_finder.py",
          INPUT, "same FRAMES table.", "until q4_sweep_validation retires"),
    Entry("scratch/deep_centers/ladder_p35/fw_8p07e_10.png", "tools/sourcing/deep_center_finder.py",
          INPUT, "same FRAMES table.", "until q4_sweep_validation retires"),
    Entry("scratch/dive_manifest/manifest_key.json", "tools/atlas/dive_manifest.py", INPUT,
          "the ranker's feature build joins its dive tiles through this key.",
          "until the ranker features are rebuilt from a durable manifest"),
    Entry("scratch/dive_manifest/tiles", "tools/atlas/dive_manifest.py", INPUT,
          "the tiles that key addresses.",
          "until the ranker features are rebuilt from a durable manifest"),
    Entry("scratch/mining/deploy_tail/report.json", "tools/mining/deploy_tail.py", INPUT,
          "library_records_build reads the deploy-tail report to build data/library/"
          "library_records.jsonl — a durable artifact with a scratch input.",
          "until the library records are rebuilt"),
    Entry("scratch/palette_preview/dramatic-test/densified.json", "tools/palettes/densify.py",
          INPUT, "preview_render's --colormaps default.",
          "regenerable — a wipe costs a re-densify"),
    Entry("scratch/present/run4_bridge/locations.jsonl", "tools/corpus/pool_to_locations.py",
          INPUT, "written by pool_to_locations, read by build_rev4_batch — a two-tool "
          "handoff that lives entirely in the deletable class.",
          "regenerable from data/guided_descend/run4/pool.jsonl"),
    Entry("scratch/present/run4_present/manifest.json", "present (src/, Rust)", INPUT,
          "build_rev4_batch's --manifest default.",
          "regenerable by re-running present over run4"),
    Entry("scratch/q4_stage1/fields", "tools/studies/q4_stage1_labelset.py", INPUT,
          "THE fit input for q4_multibrot_transfer._fit_model, which gates "
          "build_minibrot_batch screen, build_interior_band_batch sweep and "
          "interior_bakeoff. Wiped once and restored from trash on 2026-08-04; its CLASS is "
          "an open decision (tools/audit/durability_map.py).",
          "indefinite until reclassified — see the durability map's OPEN verdict"),
    Entry("scratch/q4_stage1/fields/<mb_id>.bin", "tools/studies/q4_stage1_labelset.py", INPUT,
          "the durability map's registry key for the same family.",
          "indefinite until reclassified"),
    Entry("scratch/steered_run2_manifest", "tools/atlas/steered_run2_manifest.py", INPUT,
          "keeper_calibrate's --manifest default: the blind manifest the keeper calibration "
          "in docs/design/ is derived against.",
          "until the keeper calibration is re-derived"),
    Entry("scratch/steered_run2_manifest/manifest_key.json",
          "tools/atlas/steered_run2_manifest.py", INPUT,
          "the ranker's run2 join key.", "until the ranker features are rebuilt"),
    Entry("scratch/steered_run2_manifest/tiles", "tools/atlas/steered_run2_manifest.py", INPUT,
          "the tiles that key addresses.", "until the ranker features are rebuilt"),
    Entry("scratch/wallpaper/emit_v1/manifest.jsonl", "tools/wallpaper/emit_v1.py", INPUT,
          "deploy_tail's EMIT_MANIFEST — the emission it scores the tail of.",
          "until deploy_tail is re-run from a durable emission record"),
    Entry("scratch/wallpaper/emission_dryrun_colorcells.json",
          "tools/studies/archive/emission_dryrun_v2gate.py", INPUT,
          "self-cache of an ARCHIVED study; listed so the archive is not mistaken for a "
          "live dependency.", "DEAD — archived study, wipe freely"),
    Entry("scratch/wallpaper/overnight/overnight_20260713_001420",
          "tools/wallpaper/overnight run (2026-07-13)", INPUT,
          "a HARDCODED DATED RUN: both morning_readout's default run_dir and "
          "library_records_build's ROOT. The directory no longer exists, so both defaults "
          "are dead as written — kept as an entry rather than deleted because removing the "
          "default changes each tool's CLI contract, which is not mechanical.",
          "DEAD — the referent is gone; wipe freely"),

    # ---- OUTPUT: written, never consumed. Wipeable at any time. -------------------
    # These are on the list because the literal is hardcoded rather than routed through
    # paths.scratch() — the form that cannot be found by a refactor. They are NOT wipe
    # hazards, and saying so is the point of the `kind` column.
    Entry("scratch/campaign1_blind", "tools/atlas/campaign1_manifest.py", OUTPUT,
          "--out-dir default.", "wipeable"),
    Entry("scratch/coarse_gate", "tools/queries/validate_coarse_score.py", OUTPUT,
          "the validator's own scratch dir.", "wipeable"),
    Entry("scratch/deep_centers", "tools/sourcing/emit_deep_pool.py", OUTPUT,
          "emit_pool's preview_dir default.", "wipeable"),
    Entry("scratch/deep_centers/pool.jsonl", "tools/sourcing/emit_deep_pool.py", OUTPUT,
          "emit_pool's out_path default.", "wipeable"),
    Entry("scratch/deep_centers/preview.png", "tools/sourcing/deep_center_finder.py", OUTPUT,
          "render_cmd's out default.", "wipeable"),
    Entry("scratch/emission_v1", "tools/emission/build_emission_diversity_v1.py", OUTPUT,
          "--out default.", "wipeable"),
    Entry("scratch/palette_preview/cliff-diag", "tools/studies/archive/cliff_diag.py", OUTPUT,
          "archived study's output dir.", "wipeable"),
    Entry("scratch/palette_preview/dramatic-test", "tools/palettes/preview_render.py", OUTPUT,
          "--outdir default.", "wipeable"),
    Entry("scratch/palette_preview/softcliff", "tools/studies/archive/softcliff.py", OUTPUT,
          "archived study's output dir.", "wipeable"),
    Entry("scratch/palette_preview/v2-batch", "tools/studies/archive/render_v2_batch.py",
          OUTPUT, "archived study's output dir.", "wipeable"),
    Entry("scratch/pref_loc_v0_report.md", "tools/ranker/report.py", OUTPUT,
          "the report it writes.", "wipeable"),
    Entry("scratch/present/scale_2x2", "tools/eda/scale_2x2_cap_locations.py", OUTPUT,
          "PRESENT_BASE, its own output tree.", "wipeable"),
    Entry("scratch/ranker_next_read", "tools/ranker/report.py", OUTPUT,
          "the next-read tile set report.py rmtree's and rebuilds every run. The 2026-08-03 "
          "inventory listed this as READ-side; it is not — report.py is its only toucher "
          "and it writes it.", "wipeable"),
    Entry("scratch/render_mode_pilot/exp_vs_smooth",
          "tools/studies/archive/exp_vs_smooth_rankcorr.py", OUTPUT,
          "archived study's output dir.", "wipeable"),
    Entry("scratch/render_modes/trap_circle_sweep", "tools/eda/trap_circle_sweep_montage.py",
          OUTPUT, "the montage's OUT. The 2026-08-03 inventory listed this as READ-side; it "
          "is the write target.", "wipeable"),
    Entry("scratch/wallpaper/emission_dryrun_v2gate",
          "tools/studies/archive/emission_dryrun_v2gate.py", OUTPUT,
          "archived study's output dir.", "wipeable"),
]


# --------------------------------------------------------------------------- #
# The scanner
# --------------------------------------------------------------------------- #
# A repo-relative scratch path, anchored: the string must BEGIN with `scratch/` (after an
# optional `./`). Anchoring is load-bearing — an f-string fragment like
# `f"{run_dir}/scratch/queue.jsonl"` contributes the constant `/scratch/queue.jsonl`, which
# is a DISCOVERY RUN's own scratch (relocated out-of-tree by the artifacts resolver), not
# the repo tree this guard is about. Two such false positives existed before the anchor.
_SCRATCH_LITERAL = re.compile(r"^(?:\./)?(scratch/[A-Za-z0-9_][A-Za-z0-9_.*<>-]*"
                              r"(?:/[A-Za-z0-9_.*<>-]+)*)")
# families a module declares as its own via the storage-class helper
_SCRATCH_CALL = re.compile(r"""(?:paths|P)\.scratch\(\s*['"]([A-Za-z0-9_.-]+)['"]""")


def _is_test_file(rel: str) -> bool:
    n = Path(rel).name
    return n.startswith("test_") or n.endswith("_test.py") or n == "conftest.py"


def _tracked_sources(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return [p for p in out.stdout.split()
            if p.endswith(".py") and not _is_test_file(p) and not p.startswith("scratchpad/")]


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identity of every docstring Constant, so prose is excluded from the scan."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def scan(root: Path) -> list[tuple[str, str, int]]:
    """(scratch_path, file, line) for every FOREIGN scratch literal in code.

    Parameterized on `root` so the fire direction is testable against a synthetic tree
    rather than only against this repo."""
    found, seen = [], set()
    for rel in _tracked_sources(root):
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        if "scratch/" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        docs = _docstring_ids(tree)
        stem, pkg = Path(rel).stem, Path(rel).parent.name
        owned = {stem, pkg} | set(_SCRATCH_CALL.findall(text))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in docs:
                continue
            m = _SCRATCH_LITERAL.match(node.value.strip())
            if not m:
                continue
            path = m.group(1).rstrip("/")
            family = path.split("/")[1]
            if family in owned or family.startswith(stem) or stem.startswith(family):
                continue                       # the module's own tree — the convention working
            key = (path, rel)
            if key not in seen:
                seen.add(key)
                found.append((path, rel, node.lineno))
    return found


def _covering(path: str) -> list[Entry]:
    return [e for e in ALLOWLIST if e.path == path]


# --------------------------------------------------------------------------- #
# The assertions
# --------------------------------------------------------------------------- #
def test_every_foreign_scratch_reference_is_on_the_ledger():
    """A committed tool reaching into another tool's scratch tree must be a DECLARED
    dependency. An undeclared one is a wipe hazard nobody has classified."""
    uncovered = [(p, f, ln) for p, f, ln in scan(REPO_ROOT) if not _covering(p)]
    assert not uncovered, (
        "committed non-test code names a foreign scratch/ path that is not on the ledger "
        "in this file. Either delete the reference, route it through paths.scratch()/"
        "bulk()/durable(), or add an Entry(path, owner, kind, reason, lifetime):\n"
        + "\n".join(f"  {p}   <- {f}:{ln}" for p, f, ln in sorted(uncovered)))


def test_no_path_is_covered_twice():
    """Exactly one entry per path, so `kind` and `lifetime` have one answer."""
    dupes = [e.path for e in ALLOWLIST if len(_covering(e.path)) > 1]
    assert not dupes, f"duplicated ledger entries: {sorted(set(dupes))}"


def test_no_dead_ledger_entry():
    """An entry matching nothing is a line nobody classified, and a rotted allowlist is how
    the covering assertion beside it goes vacuous (verification_practice.md §5)."""
    referenced = {p for p, _f, _ln in scan(REPO_ROOT)}
    dead = [e.path for e in ALLOWLIST if e.path not in referenced]
    assert not dead, (
        "ledger entries matching nothing in the tree — the reference was deleted or moved, "
        "so delete the entry too:\n" + "\n".join(f"  {p}" for p in sorted(dead)))


def test_scan_is_not_vacuous():
    """A derived set can pass by evaluating EMPTY. Pair every derived assertion with a
    non-vacuity check (verification_practice.md §5)."""
    hits = scan(REPO_ROOT)
    assert len(hits) >= 20, f"the scanner found only {len(hits)} references — it broke"
    assert any(e.kind == INPUT for e in ALLOWLIST)
    assert any(e.kind == OUTPUT for e in ALLOWLIST)


def test_every_entry_declares_a_kind_and_a_lifetime():
    for e in ALLOWLIST:
        assert e.kind in (INPUT, OUTPUT), e
        assert e.reason.strip() and e.lifetime.strip() and e.owner.strip(), e


def test_input_rows_are_the_wipe_ledger():
    """The point of the file: `INPUT` rows name what a `rm -r scratch/*` would break. If
    this ever reads zero, either the hazard is gone (celebrate, then delete this test) or
    the `kind` column has been filled in wrong."""
    inputs = [e for e in ALLOWLIST if e.kind == INPUT]
    assert len(inputs) >= 5, "no scratch INPUTs declared — verify before believing it"


# --------------------------------------------------------------------------- #
# Prove it red
# --------------------------------------------------------------------------- #
def _plant(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "-f", rel], cwd=root, capture_output=True, check=True)


@pytest.fixture()
def synthetic_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, capture_output=True, check=True)
    return tmp_path


def test_guard_fires_on_an_injected_foreign_scratch_reference(synthetic_repo):
    """INJECTION: a new tool that reaches into another tool's scratch tree is CAUGHT."""
    root = synthetic_repo
    assert scan(root) == []                                  # clean tree is quiet
    _plant(root, "tools/newtool/reader.py",
           'from pathlib import Path\nP = Path("scratch/somebody_elses/cache.json")\n')
    hits = scan(root)
    assert any(p == "scratch/somebody_elses/cache.json" for p, _f, _ln in hits), hits
    assert not _covering("scratch/somebody_elses/cache.json"), (
        "the injected path must NOT be on the real ledger")


def test_guard_stays_quiet_on_the_three_sanctioned_shapes(synthetic_repo):
    """It must not fire on: the module's OWN tree, the declared `paths.scratch()` form, or
    a docstring. Each is a way this guard could go red during ordinary work and be trained
    out (verification_practice.md §4)."""
    root = synthetic_repo
    _plant(root, "tools/mytool/mytool.py",
           'from pathlib import Path\nOUT = Path("scratch/mytool/out.png")\n')
    _plant(root, "tools/other/declared.py",
           'import paths\nD = paths.scratch("declared_family")\n'
           'E = paths.scratch("declared_family", "sub")\n')
    _plant(root, "tools/other/prose.py",
           '"""Reads: scratch/someone_else/thing.json"""\nX = 1\n')
    assert scan(root) == [], scan(root)


def test_guard_ignores_a_discovery_run_scratch_fragment(synthetic_repo):
    """The anchor: `f"{run_dir}/scratch/queue.jsonl"` is a DISCOVERY RUN's own scratch,
    relocated out-of-tree by the artifacts resolver — not the repo tree. Two real call
    sites (build_q4_harvest_batches, build_label_seeded_batches) were false positives
    before the pattern was anchored, and both route through paths.bulk()."""
    root = synthetic_repo
    _plant(root, "tools/atlas/queue.py",
           'import paths\ndef q(run_dir):\n'
           '    return paths.bulk(f"{run_dir}/scratch/queue.jsonl")\n')
    assert scan(root) == [], scan(root)


def test_guard_ignores_an_html_relative_link(synthetic_repo):
    """`"../../scratch/x/"` is a link inside a generated page, not a path the tool opens.
    Six real call sites in palette_extractor/ take this form; allowlisting them would put
    six lines on the wipe ledger that name nothing a tool reads."""
    root = synthetic_repo
    _plant(root, "tools/x/page.py", 'D = {"strip_dir": "../../scratch/somewhere/strips/"}\n')
    assert scan(root) == [], scan(root)


def test_guard_ignores_test_files(synthetic_repo):
    """Test files are excluded by construction — a fixture naming a scratch path is not a
    production dependency."""
    root = synthetic_repo
    _plant(root, "tools/x/test_x.py", 'P = "scratch/whatever/thing.json"\n')
    assert scan(root) == []
