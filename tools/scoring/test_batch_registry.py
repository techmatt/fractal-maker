#!/usr/bin/env python
"""`tools/scoring/batch_registry.py` is the ONLY batch->split classification table.

Three things, and the third is the one that matters.

  1. The table is internally coherent (split derived from eval-eligibility, no
     eval-eligible-and-biased entry, the exemption flag only on instruments).
  2. The two entry points cannot disagree. `assign_split` (what every batch builder
     consults BEFORE drawing a batch) and `classify_batch` (what a manifest build
     REALIZES) are two readings of one row — this file asserts they agree for every
     registered batch and for the fail-closed default, which is exactly the assertion
     that was absent while the two tables said different things about
     `2026-06-23_flat_generate_loose0_v3` for a week.
  3. A SOURCE SCAN. No tracked Python file outside this module may bind a batch-category
     name to a literal again, and no manifest-build module may name a corpus batch at all.
     A build-side split decision has to name a batch to make one, so a batch id in
     `tools/v*/build_manifest.py` is the shape of the defect returning.

  uv run pytest tools/scoring/test_batch_registry.py -q
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT))

import batch_registry as br  # noqa: E402

OWNER = "tools/scoring/batch_registry.py"
SELF = "tools/scoring/test_batch_registry.py"


# =========================================================================== #
# 1. the table is coherent
# =========================================================================== #
def test_the_table_is_not_empty_and_every_entry_is_well_formed():
    """Derive + prove non-empty (`verification_practice.md` §5): every assertion below
    walks the table, so an emptied table would pass all of them by evaluating nothing."""
    assert len(br.REGISTRY) >= 16, f"the registry shrank to {len(br.REGISTRY)} batches"
    assert br.eval_eligible_batches(), "no eval-eligible batch — every instrument vanished"
    br._self_check()


def test_split_is_derived_from_eval_eligibility_and_is_never_a_third_fact():
    for bid, regs in br.REGISTRY.items():
        for r in regs:
            assert br.split_of(r) == ("eval" if r.eval_eligible else "train"), bid


def test_no_registration_is_both_eval_eligible_and_biased():
    """Protocol §2's cardinal sin, caught at registration time rather than at GATE 4 —
    after a batch has been drawn, rendered and labeled."""
    for bid, regs in br.REGISTRY.items():
        for r in regs:
            assert not (r.eval_eligible and r.biased), bid


def test_the_four_instruments_are_exactly_the_eval_eligible_registrations():
    """The class-ceiling trap in registry form: a fifth instrument appearing silently is
    a threshold moving. Both directions, and both counts pinned."""
    assert br.instrument_sources() == {
        br.SOURCE_CENSUS, br.SOURCE_FLOOR, br.SOURCE_MANEUVER_UNIFORM, br.SOURCE_Q4_UNIFORM}
    assert len(br.eval_eligible_batches()) == 5      # census is two batches
    for src in br.instrument_sources():
        bids = {b for b in br.eval_eligible_batches()
                for r in br.REGISTRY[b] if r.eval_eligible
                and br.manifest_source_of(r, b) == src}
        assert bids, f"instrument {src} names no batch"


def test_every_instrument_currently_claims_the_score_unconditioned_exemption():
    """All four qualify (Matt, 2026-08-04), and that is a FACT about today's table, not a
    property of being an instrument. A future instrument drawn for a model-quality read
    should register `score_unconditioned=False`, and this test is where that shows up."""
    flagged = {s for b in br.eval_eligible_batches() for r in br.REGISTRY[b]
               if r.eval_eligible and r.score_unconditioned
               for s in [br.manifest_source_of(r, b)]}
    assert flagged == br.instrument_sources()


def test_the_fail_closed_default_is_biased_train_and_unregistered():
    assert br.assign_split("2026-99-99_never_registered", "mandelbrot") == \
        ("train", True, br.SOURCE_UNREGISTERED)
    assert br.classify_batch("2026-99-99_never_registered", "phoenix") == \
        (False, True, "biased:2026-99-99_never_registered")
    assert br.score_unconditioned("2026-99-99_never_registered", "mandelbrot") is False
    assert not br.is_registered("2026-99-99_never_registered")


# =========================================================================== #
# 2. the two entry points are two readings of one row
# =========================================================================== #
ALL_PROBES = [(b, ft) for b in sorted(set(br.REGISTRY) | {"2026-99-99_unregistered"})
              for ft in ("mandelbrot", "julia", "julia_multibrot3", "julia_multibrot4",
                         "julia_multibrot5", "multibrot3", "multibrot4", "multibrot5",
                         "phoenix")]


@pytest.mark.parametrize("bid,ft", ALL_PROBES)
def test_assign_split_and_classify_batch_cannot_disagree(bid, ft):
    """THE point of one registry. `split == "eval"` and `eval_eligible` are the same fact
    read twice; `biased` is the same field read twice."""
    split, biased, _source = br.assign_split(bid, ft)
    evalable, biased2, _msource = br.classify_batch(bid, ft)
    assert (split == "eval") is evalable, f"{bid}/{ft}: builder says {split}, realizer {evalable}"
    assert biased is biased2, f"{bid}/{ft}: biased disagrees"


def test_the_agreement_check_would_catch_an_injected_disagreement(monkeypatch):
    """Non-vacuity (§6): the check has to be able to fail. Inject exactly the historical
    defect — a batch the builder calls train/unbiased and the realizer calls an
    instrument — and prove the assertion fires."""
    victim = "2026-06-23_flat_generate_loose0_v3"
    forked = dict(br.REGISTRY)
    monkeypatch.setattr(br, "REGISTRY", forked)
    monkeypatch.setattr(br, "assign_split",
                        lambda b, ft: ("train", False, "loose0_v3")
                        if b == victim else br.assign_split(b, ft))
    split, biased, _ = br.assign_split(victim, "mandelbrot")
    evalable, _, _ = br.classify_batch(victim, "mandelbrot")
    assert (split == "eval") is not evalable, \
        "the injected fork did not actually diverge — the test above proves nothing"


def test_loose0_v3_is_the_mandelbrot_eval_floor_on_BOTH_sides():
    """The correction of 2026-08-04. `tools/v7/build_manifest` said train/unbiased and
    `tools/v8/build_manifest` said eval floor; `data/v8/manifest.jsonl` has 526 rows
    sourced `loose0_v3_floor`, so the realizer was the live truth."""
    assert br.assign_split("2026-06-23_flat_generate_loose0_v3", "mandelbrot") == \
        ("eval", False, br.SOURCE_FLOOR)
    assert br.classify_batch("2026-06-23_flat_generate_loose0_v3", "mandelbrot") == \
        (True, False, br.SOURCE_FLOOR)
    reg = br.lookup("2026-06-23_flat_generate_loose0_v3", "mandelbrot")
    assert reg.superseded == ("train", False, "loose0_v3"), \
        "the prior registration must stay recorded — a batch.json froze that tuple"


def test_the_census_batch_splits_by_plane_and_the_native_rows_keep_their_frozen_name():
    """`prospect_native` is the one biased registration whose name reaches a manifest row
    (22 rows in the v8 prefix, a GATE 11 frozen field). Everything else biased realizes as
    `biased:<batch_id>`."""
    for bid in br.batches_with_source(br.SOURCE_CENSUS):
        assert br.classify_batch(bid, "julia_multibrot4") == (True, False, br.SOURCE_CENSUS)
        assert br.classify_batch(bid, "multibrot4") == (False, True, "prospect_native")
    assert br.classify_batch("2026-07-11_jm3_band_v1", "julia_multibrot3") == \
        (False, True, "biased:2026-07-11_jm3_band_v1")


def test_every_source_in_the_frozen_v10_manifest_is_still_producible():
    """The zero-change proof for the manifest `source` VOCABULARY. `source` is a GATE 11
    frozen field, so a registry that renamed a classification would silently break every
    future append onto the v8 prefix — and the break would not appear until a build ran."""
    import json
    atomic = set()
    for line in (ROOT / "data/v10/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            atomic |= set(json.loads(line)["source"].split("+"))
    assert len(atomic) >= 24, "the frozen manifest yielded suspiciously few source names"
    for name in sorted(atomic):
        if name.startswith("biased:"):
            bid = name[len("biased:"):]
            assert br.classify_batch(bid, "mandelbrot")[2] == name, name
        else:
            produced = {br.classify_batch(b, ft)[2]
                        for b in br.REGISTRY
                        for ft in ("mandelbrot", "julia_multibrot3", "multibrot4")}
            assert name in produced, f"{name!r} is in the frozen manifest but unproducible"


# =========================================================================== #
# 3. the source scan — one table, and no build-side batch names
# =========================================================================== #
CATEGORY_NAMES = ("REGISTRY", "CENSUS_BATCHES", "FLOOR_BATCHES", "UNBIASED_TRAIN_BATCHES",
                  "UNIFORM_BATCHES", "BAND_BATCHES", "SUPPLY_CRAWL_UNIFORM_BATCHES",
                  "Q4_UNIFORM_EVAL_BATCHES")
# `NAME = {` / `NAME: dict = {` — a literal binding. A derived read (`= br.batches_with_...`)
# does not match, which is the whole distinction being enforced.
LITERAL = re.compile(r"^\s*(" + "|".join(CATEGORY_NAMES) + r")\s*(:[^=]*)?=\s*\{", re.M)
# A corpus batch id as a complete string literal. Prose in a docstring spells one in
# backticks, so this matches code, not commentary.
BATCH_ID = re.compile(r"""["']20\d\d-\d\d-\d\d_[A-Za-z0-9_]+["']""")


def _tracked_python():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git unavailable — cannot enumerate tracked files")
    return [p.replace("\\", "/") for p in out.stdout.splitlines() if p.strip()]


def test_no_second_literal_copy_of_a_batch_category_exists():
    """THE point of this file. Two tables do not fail when the second is created; they
    fail when someone edits one of them."""
    offenders = []
    for rel in _tracked_python():
        if rel in (OWNER, SELF):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in LITERAL.finditer(text):
            offenders.append(f"{rel}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        f"{len(offenders)} literal batch-category binding(s) outside {OWNER}: {offenders}. "
        f"Derive it — `br.batches_with_source(...)` — or add the batch to the registry.")


# v6 predates batch-level registration entirely: it folds ONE batch and splits inside it by
# `provenance.selection_role`, so its batch id is a path constant, not a classification.
# The exemption is held to that claim below rather than trusted.
EXEMPT_BUILD_MODULES = {"tools/v6/build_manifest.py"}


def _build_modules():
    mods = [p for p in _tracked_python()
            if re.fullmatch(r"tools/v\d+/build_manifest\.py", p)
            and p not in EXEMPT_BUILD_MODULES]
    assert len(mods) >= 3, f"found only {mods} — the scan below would be near-vacuous"
    return mods


def test_the_exempt_build_module_still_has_no_batch_classification():
    """An allowlist needs a no-dead-entry assertion (§5), and this one also has to keep
    earning its exemption: if v6 ever grows a batch->split rule the exemption goes red
    instead of silently covering a fourth copy of the table."""
    for rel in EXEMPT_BUILD_MODULES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "def assign_split" not in src and "def classify_batch" not in src, \
            f"{rel} now classifies batches — drop the exemption and register them"
        assert "selection_role" in src, f"{rel} no longer splits by selection_role"


# A manifest-build module may name a batch ONLY for a binding that is not a
# classification. One exists, and it is asserted live below rather than trusted.
NON_CLASSIFYING_BINDINGS = {"NEW_BATCHES"}


def test_no_manifest_build_module_names_a_corpus_batch():
    """A build-side split decision has to name a batch to make one. After 2026-08-04 the
    only batch ids in `tools/v*/build_manifest.py` are `NEW_BATCHES` (which batches carry
    an `atom_key` and a rule-label file — not a classification, and pinned against the
    corpus by `tools/v10/test_v10_build.py`)."""
    offenders = []
    for rel in _build_modules():
        binding = None
        for i, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^(\w+)\s*(:[^=]*)?=", line)
            if m:
                binding = m.group(1)
            elif line and not line[0].isspace():
                binding = None
            if BATCH_ID.search(line) and binding not in NON_CLASSIFYING_BINDINGS:
                offenders.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not offenders, (
        f"{len(offenders)} corpus batch id(s) in a manifest-build module — a split "
        f"decision reached outside {OWNER}:\n  " + "\n  ".join(offenders))


def test_the_non_classifying_exemption_is_not_dead():
    """An allowlist needs a no-dead-entry assertion (§5): an exemption matching nothing is
    a line nobody classified, and it is how the scan beside it goes vacuous."""
    text = "\n".join((ROOT / rel).read_text(encoding="utf-8") for rel in _build_modules())
    for name in NON_CLASSIFYING_BINDINGS:
        assert re.search(rf"^{name}\s*=\s*\{{", text, re.M), \
            f"{name} is exempted but no longer exists — drop the exemption"


def test_the_scans_would_actually_catch_a_copy():
    """Non-vacuity: the regexes must match the shapes the real copies were written in —
    all four are verbatim from the pre-2026-08-04 tree."""
    for s in ('CENSUS_BATCHES = {"2026-07-17_prospect_run1_baserate_R_v1",\n'
              '                  "2026-07-17_prospect_run1_baserate_v1"}',
              'FLOOR_BATCHES = {"2026-06-23_flat_generate_loose0_v3"}',
              'UNBIASED_TRAIN_BATCHES = set()\nUNIFORM_BATCHES = {"x"}',
              'REGISTRY: dict = {"a": ()}'):
        assert LITERAL.search(s), f"the scan would miss this copy:\n{s}"
    for ok in ("CENSUS_BATCHES = br.batches_with_source(br.SOURCE_CENSUS)",
               "UNIFORM_BATCHES = br.batches_with_source(UNIFORM_SOURCE)",
               "if batch_id in CENSUS_BATCHES:"):
        assert not LITERAL.search(ok), f"the scan false-positives on:\n{ok}"
    assert BATCH_ID.search('    if batch_id in {"2026-08-01_supply_crawl_uniform_v1"}:')
    assert not BATCH_ID.search("    the `2026-08-01_supply_crawl_uniform_v1` leg (90 rows)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
