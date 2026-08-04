#!/usr/bin/env python
"""THE batch registry: one owner for every corpus batch's split classification.

Until 2026-08-04 there were two. `tools/v7/build_manifest.assign_split` was the
batch-level registry every batch builder consults BEFORE it draws a batch (a dozen live
callers under `tools/atlas/`, `tools/sourcing/`, `tools/corpus/`), and it was the only
one that knew the 2026-08-03 registrations. `tools/v8|v10/build_manifest.classify_batch`
was what a manifest build actually REALIZED. They contradicted each other on
`2026-06-23_flat_generate_loose0_v3`: v7 said train/unbiased, v8 made it the mandelbrot
eval floor, and the floor is what is on disk in `data/v8/manifest.jsonl`. Two authorities
that disagree about which population is an instrument is the shape of a leak nobody sees,
so there is now one table and both entry points read it.

WHAT A REGISTRATION SAYS. Four independent facts, none derivable from another:

  `biased`               — was a model/score in the selection? The disqualifying property
                           for eval (protocol §2's cardinal sin is biased-in-eval).
  `eval_eligible`        — may this batch's locations be an instrument? Implies `not
                           biased`, and implies split `eval` (see `split_of`).
  `score_unconditioned`  — was the DRAW itself unconditioned on any score? Strictly
                           stronger than `not biased`, and it is what exempts an
                           instrument from the forced-eval group cascade (below).
  `source`               — the name the decision travels under.

SPLIT IS DERIVED, NEVER STORED (`measurement_practice.md`, "derive state in code"):
`split_of(reg)` is `"eval" if reg.eval_eligible else "train"`. There is no entry whose
side is a third fact, and storing one is how a table grows a row that contradicts itself.

TWO VOCABULARIES, ONE DECISION. `source` is what `assign_split` returns and what a
batch's `batch.json` froze in `registration.assign_split` at build time.
`manifest_source` is what a manifest ROW's `source` field carries — a frozen-record field
under v10's GATE 11, so it cannot be renamed to tidy it up. They differ only in that a
registered-but-biased batch is realized as `"biased:<batch_id>"` (v8's fail-closed
spelling, which is what is on disk for ~6,900 rows), which is the DEFAULT here rather
than a second column. `prospect_native` is the one entry that overrides it, because v8
named that classification explicitly and the v8 prefix carries the name.

THE SCORE-UNCONDITIONED EXEMPTION (Matt, 2026-08-04)
---------------------------------------------------
`assign_split_by_group` forces a group holding a forced-eval location 100% to eval, and
a BIASED location in such a group cannot go anywhere: eval would be biased-in-eval, train
would straddle the group. v8's resolution was to DROP it. That protection is against
spatial leakage in a model-quality read — an eval location whose morphology sits in
training scores optimistically.

The property that qualifies an eval instrument is **score-unconditioned draw**, not
neighborhood isolation. An instrument's main products are base rates and `t_good`
calibration, and neither needs the isolation: a base rate over a score-unconditioned draw
is what it is regardless of what else was trained on. So for instruments flagged
`score_unconditioned`, forced-eval applies to the instrument's OWN locations and a biased
group-mate stays TRAIN instead of being dropped. Under the live registry the drop rule
cost **687 train locations (193 threes, 50 fours)**; with the exemption it costs 0.

CAVEAT, and it is the reason the flag is per-instrument data rather than a global switch:
**model-performance numbers read off an exempted leg are mildly optimistic wherever a
train group-mate exists.** Fine for base rates and threshold calibration; NOT fine for
fine-grained AUC comparisons between checkpoints. A future instrument drawn for a
model-quality read should register `score_unconditioned=False` and keep the drop rule.
Also in `docs/design/classifier_retrain_protocol.md` §2.

  uv run pytest tools/scoring/test_batch_registry.py -q
"""
from __future__ import annotations

from typing import NamedTuple

# --------------------------------------------------------------------------- #
# The eval-instrument source names. Named constants because GATE 6 (census identity)
# and GATE 13 (the uniform-90 instrument) pin specific instruments, and a manifest row's
# `source` is a frozen-record field.
# --------------------------------------------------------------------------- #
SOURCE_CENSUS = "prospect_census"
SOURCE_FLOOR = "loose0_v3_floor"
SOURCE_MANEUVER_UNIFORM = "maneuver_uniform_v1"
SOURCE_Q4_UNIFORM = "q4_uniform_eval"
SOURCE_UNREGISTERED = "unregistered"


class Registration(NamedTuple):
    """One batch's classification. `ft_prefix` narrows an entry to one plane."""
    source: str
    biased: bool
    eval_eligible: bool = False
    score_unconditioned: bool = False
    ft_prefix: str | None = None          # entry applies only to matching fractal_types
    manifest_source: str | None = None    # override; see module docstring
    superseded: tuple | None = None       # a PRIOR (split, biased, source) this replaces
    why: str = ""


def split_of(reg: Registration) -> str:
    """Derived, never stored. Eval-eligibility IS the eval side."""
    return "eval" if reg.eval_eligible else "train"


def manifest_source_of(reg: Registration, batch_id: str) -> str:
    """The `source` field a manifest ROW carries for this registration."""
    if reg.manifest_source is not None:
        return reg.manifest_source
    if reg.eval_eligible or not reg.biased:
        return reg.source
    return "biased:" + batch_id


# --------------------------------------------------------------------------- #
# THE TABLE. Order matters only within a batch's tuple: the first entry whose
# `ft_prefix` matches wins, so an ft-specific entry precedes the catch-all.
#
# FAIL-CLOSED: a batch that is not here classifies biased -> train (see `lookup`).
# Do NOT add a batch to work around a classification you dislike — if a batch is
# misclassified, fix its registration and say why in `why`.
# --------------------------------------------------------------------------- #
_CENSUS = (
    Registration(
        source=SOURCE_CENSUS, biased=False, eval_eligible=True,
        score_unconditioned=True, ft_prefix="julia_multibrot",
        why="prospect run-1 base rate, the only unbiased-given-DESCENT julia draw that "
            "exists. The pinned primary instrument (n=144, protocol §3)."),
    Registration(
        source="prospect_native", biased=True, manifest_source="prospect_native",
        why="the same batch's native-plane rows are descent-screened, so they are biased. "
            "`manifest_source` is pinned because v8 named this classification and the "
            "v10 frozen prefix carries the name on 22 rows."),
)

REGISTRY: dict[str, tuple[Registration, ...]] = {
    # ---- the instruments (eval-eligible, all four score-unconditioned) ----
    "2026-07-17_prospect_run1_baserate_v1": _CENSUS,
    "2026-07-17_prospect_run1_baserate_R_v1": _CENSUS,
    "2026-06-23_flat_generate_loose0_v3": (
        Registration(
            source=SOURCE_FLOOR, biased=False, eval_eligible=True,
            score_unconditioned=True,
            superseded=("train", False, "loose0_v3"),
            why="the MANDELBROT EVAL FLOOR (Matt, 2026-07-29): an unbiased base-rate "
                "flat-generate draw over the native mandelbrot plane, 526 loc, giving the "
                "mandelbrot slice (59% of the corpus) a non-regression instrument the "
                "julia:multibrot census cannot. The v7 registry still called this "
                "train/unbiased a week after v8 realized it as eval — the contradiction "
                "this table exists to end."),),
    "2026-08-01_supply_crawl_uniform_v1": (
        Registration(
            source=SOURCE_MANEUVER_UNIFORM, biased=False, eval_eligible=True,
            score_unconditioned=True,
            superseded=("train", False, "supply_crawl_uniform"),
            why="uniform over the crawl's recorded population with NO score anywhere in "
                "the selection. Registered train-side when the crawl was drawn "
                "(2026-08-01); Matt moved it to the maneuver-view instrument in the v10 "
                "build (2026-08-02) — the population every discovery tool emits from, "
                "which neither the census nor the floor covers. n=90."),),
    "2026-08-03_q4_uniform_eval_v1": (
        Registration(
            source=SOURCE_Q4_UNIFORM, biased=False, eval_eligible=True,
            score_unconditioned=True,
            why="the long harvest's one leg with no score in the selection: systematic "
                "draws over a FAMILY'S parameter space, taken before the run scored "
                "anything, for the five partitions with no unbiased eval rows at all "
                "(production_seeder.T_GOOD_UNCALIBRATED). Registered 2026-08-03, BEFORE "
                "the build; v10 was already frozen, so the first manifest to realize it "
                "is the next one."),),

    # ---- biased / train-side, registered explicitly so the fail-closed default stays
    # ---- a record of "nobody thought about this batch" rather than a shrug ----
    "2026-07-11_jm3_band_v1": (
        Registration(source="jm_band", biased=True,
                     why="model-band-selected (decoded_class=2)"),),
    "2026-07-12_jm45_band_v1": (
        Registration(source="jm_band", biased=True,
                     why="model-band-selected (decoded_class=2)"),),
    "2026-07-12_blindspot_v6reject_v1": (
        Registration(source="blindspot_v6reject", biased=True,
                     why="negative-by-construction: v6's rejects"),),
    "2026-08-01_supply_crawl_strat_a_v1": (
        Registration(source="supply_crawl_stratified", biased=True,
                     why="round-robin over (degree x operator x composite-v3 bin); a "
                         "screen score chose the strata, so biased by construction"),),
    "2026-08-01_supply_crawl_strat_b_v1": (
        Registration(source="supply_crawl_stratified", biased=True,
                     why="round-robin over (degree x operator x composite-v3 bin); a "
                         "screen score chose the strata, so biased by construction"),),
    "2026-08-01_supply_crawl_exemplar_v1": (
        Registration(source="supply_crawl_exemplar", biased=True,
                     why="top by exemplar similarity — selected on a score"),),
    "2026-08-02_label_seeded_v2_a": (
        Registration(source="label_seeded_v2", biased=True,
                     why="biased twice: seeded on the corpus's own class-3/4 locations, "
                         "and queue-ordered by the fitted view_fit_v1.1 score"),),
    "2026-08-02_label_seeded_v2_b": (
        Registration(source="label_seeded_v2", biased=True,
                     why="biased twice: seeded on the corpus's own class-3/4 locations, "
                         "and queue-ordered by the fitted view_fit_v1.1 score"),),
    "2026-08-03_q4_harvest_ranked_v1": (
        Registration(source="q4_harvest_ranked", biased=True,
                     why="the top of the run's own ranked queue; a q3/q4 rate measured on "
                         "it is a statement about the ranker, not a base rate"),),
    "2026-08-03_q4_near_minibrot_v1": (
        Registration(source="q4_near_minibrot", biased=True,
                     why="the ladder around known nuclei is systematic, but the rows that "
                         "reached the batch survived the run's screens"),),
    "2026-08-03_v2_sitting_v1": (
        Registration(source="v2_sitting", biased=True,
                     why="the harvest-v2 record-and-rank queue, tier-sorted and cut by "
                         "three filters; no rate measured on it is a base rate"),),
}

UNREGISTERED = Registration(
    source=SOURCE_UNREGISTERED, biased=True,
    why="FAIL CLOSED. Unbiasedness and eval-eligibility require EXPLICIT registration, so "
        "a biased batch nobody remembered to list is still safe (biased -> train). This "
        "inverts the pre-2026-07 default, which fell every unregistered batch through to "
        "('train', False, 'loose0_v3') and silently tagged it unbiased.")


def lookup(batch_id: str, ft: str) -> Registration:
    """The registration for one (batch, fractal_type). Fail-closed on an unknown batch."""
    for reg in REGISTRY.get(batch_id, ()):
        if reg.ft_prefix is None or ft.startswith(reg.ft_prefix):
            return reg
    return UNREGISTERED


def is_registered(batch_id: str) -> bool:
    return batch_id in REGISTRY


def assign_split(batch_id: str, ft: str) -> tuple[str, bool, str]:
    """(split, biased, source) — the batch-builder-facing contract."""
    reg = lookup(batch_id, ft)
    return split_of(reg), reg.biased, reg.source


def classify_batch(batch_id: str, ft: str) -> tuple[bool, bool, str]:
    """(eval_eligible, biased, manifest_source) — the manifest-realizer-facing contract."""
    reg = lookup(batch_id, ft)
    return reg.eval_eligible, reg.biased, manifest_source_of(reg, batch_id)


def score_unconditioned(batch_id: str, ft: str) -> bool:
    return lookup(batch_id, ft).score_unconditioned


def batches_with_source(source: str) -> set[str]:
    """Every batch id carrying `source` in any of its entries. Derived, so a category
    name can never drift from the table (`verification_practice.md` §5)."""
    return {bid for bid, regs in REGISTRY.items() if any(r.source == source for r in regs)}


def eval_eligible_batches() -> set[str]:
    return {bid for bid, regs in REGISTRY.items() if any(r.eval_eligible for r in regs)}


def unbiased_train_batches() -> set[str]:
    """Unbiased but NOT an instrument. Empty is the correct state, not a stub: loose0_v3
    vacated the category when it became the mandelbrot floor. It exists because "unbiased"
    and "eval-eligible" are two facts, and a batch can have the first without the second."""
    return {bid for bid, regs in REGISTRY.items()
            if all(not r.biased and not r.eval_eligible for r in regs)}


def instrument_sources() -> set[str]:
    """The manifest `source` of every eval-eligible registration."""
    return {manifest_source_of(r, bid) for bid, regs in REGISTRY.items()
            for r in regs if r.eval_eligible}


def _self_check():
    """Invariants the table must satisfy. Run at import: a malformed registration must
    fail LOUDLY at the first consumer, not silently classify something into eval."""
    for bid, regs in REGISTRY.items():
        assert regs, f"{bid}: empty registration tuple"
        for r in regs:
            assert not (r.eval_eligible and r.biased), \
                f"{bid}: eval_eligible AND biased — the cardinal sin at registration time"
            assert not (r.score_unconditioned and not r.eval_eligible), \
                (f"{bid}: score_unconditioned on a non-instrument. The flag only acts "
                 f"through the forced-eval group cascade, which only instruments enter.")
            assert r.source != SOURCE_UNREGISTERED, \
                f"{bid}: a registered batch may not claim the fail-closed source name"
        # a catch-all entry must come last, or it shadows the ft-specific ones
        for r in regs[:-1]:
            assert r.ft_prefix is not None, f"{bid}: a catch-all entry precedes an ft entry"


_self_check()
