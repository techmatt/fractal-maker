"""THE canonical per-partition RELEASE-MIX ratio table. One copy, imported everywhere.

Matt, 2026-08-04. The intended mix of a release, expressed once, as relative ratios rather
than as shares: a ratio table survives a partition being registered or retired (the shares
renormalize), where a share table has to be re-summed by hand every time and is wrong in
between. `mandelbrot : multibrot3 : phoenix:classic = 3 : 1 : 0.2` says everything the mix
policy says, and says it in one place.

WHY IT IS RATIOS AND NOT A TARGET COUNT. The table is consumed at two very different scales —
a discovery run's deficit currency (labels) and, later, emission's per-cell weight (images) —
and the only thing those two agree on is the RELATIVE intent. Each consumer anchors the
ratios to its own scale; the table never carries one.

WHO READS IT (three consumers, three scales, one policy)
--------------------------------------------------------
`pop_quota.deficits_from_currency` — the per-partition currency TARGET is `ratio_p` scaled so
the maximum-ratio partitions sit at the anchor the uniform rule used (the richest holding), and
the deficit is the shortfall against that target. That replaces the uniform target: every
partition used to be levelled to the richest holding regardless of how much of the release it
was ever meant to be.

`deficit_scheduler.target_shares` — the discovery order book, denominated in distinct looks:
`shares()` over the run's tracked partitions, straight through.

`cells.TargetMeasure.from_partition_shares` — the emission target measure, denominated in
joint (partition × cluster × flavour × style) cells: `weight_p = share_p / n_feasible_cells_p`,
re-solved against the live intake. The division is what makes a partition's realized share its
INTENDED share whatever its morph-cluster count; a substituted multiplier would scale each
partition's share by that count instead.

Emission was deliberately out of scope on the day this table landed, and read a hand-edited
`data/emission/target_measure.json` carrying nine literal multipliers that disagreed with these
ratios (mandelbrot 9.0% vs 22.7%). That file, and the config machinery under it, were retired
on 2026-08-04 — see `docs/design/retired.md`. There is no second target source, and
`tools/scoring/test_release_mix_one_source.py` asserts both that no reader of one comes back
and that the two consumers resolve identical partition shares from one input.

COMPLETENESS IS A RED BUILD, IN BOTH DIRECTIONS
-----------------------------------------------
`check_complete()` runs AT IMPORT against `partitions.ALL_FAMS`: a registered partition with no
ratio raises, and a ratio for an unregistered partition raises. Both halves matter and they
fail differently. A missing ratio is the `partitions._registered` failure one layer up — the
partition would get a target of nothing and read as "that partition had no demand" rather than
as a missing policy decision. An extra ratio is a partition somebody retired without retiring
its share of the release, so the remaining ratios silently mean less than they say.

    from release_mix import RATIO, ratios, ratio_of, shares
"""
from __future__ import annotations

from partitions import ALL_FAMS

# partition -> relative weight in the intended release mix. Dictated by Matt, 2026-08-04.
# The two degree-2 planes carry the release; the higher native degrees and their julia twins
# are equal supporting families; `phoenix:classic` is one pinned parameter point with 16 v10
# looks in total, so it is a garnish (0.2) rather than a family (1).
RATIO = {
    "mandelbrot": 3.0,
    "julia:mandelbrot": 3.0,
    "multibrot3": 1.0,
    "julia:multibrot3": 1.0,
    "multibrot4": 1.0,
    "julia:multibrot4": 1.0,
    "multibrot5": 1.0,
    "julia:multibrot5": 1.0,
    "phoenix": 1.0,
    "phoenix:classic": 0.2,
}


def check_complete(ratio: dict | None = None, fams=None) -> None:
    """Raise unless the ratio table and the partition registry cover exactly each other.

    Reads both at CALL TIME (and defaults to the module globals) so the guard is provable red
    by deleting either an `ALL_FAMS` entry or a `RATIO` entry, and so a test can inject a
    broken pair without editing the file. Called at import — see the module docstring for why
    each direction is a real failure and not pedantry."""
    ratio = RATIO if ratio is None else ratio
    fams = ALL_FAMS if fams is None else fams
    missing = [p for p in fams if p not in ratio]
    extra = [p for p in ratio if p not in fams]
    if missing or extra:
        raise KeyError(
            f"release_mix.RATIO and partitions.ALL_FAMS disagree — registered with no ratio: "
            f"{missing}; ratio for an unregistered partition: {extra}. Every partition that "
            f"can reach production needs a declared share of the release mix, and a ratio for "
            f"a partition nobody serves silently deflates every other ratio.")
    bad = sorted(p for p in fams if not (float(ratio[p]) > 0.0))
    if bad:
        raise ValueError(
            f"release_mix.RATIO has a non-positive ratio for {bad}. A partition that should "
            f"get none of the release is RETIRED from partitions.ALL_FAMS, not zeroed here — "
            f"a zero ratio leaves it registered, floored, censused and permanently starved.")


def ratios(fams=None) -> dict:
    """The table as a plain dict, read at call time. A COPY, so a consumer that normalizes or
    scales it in place cannot edit the policy for everyone else in the process."""
    fams = ALL_FAMS if fams is None else fams
    return {p: float(RATIO[p]) for p in fams}


def ratio_of(partition: str) -> float:
    """One partition's ratio. Raises rather than defaulting: a default here is how an
    unregistered partition gets a plausible-looking share nobody decided."""
    try:
        return float(RATIO[partition])
    except KeyError:
        raise KeyError(
            f"partition {partition!r} has no release-mix ratio. Register it in "
            f"release_mix.RATIO (and in partitions.ALL_FAMS) before anything allocates "
            f"against it.") from None


def shares(fams=None) -> dict:
    """The ratios normalized to sum to 1 — the intended release mix as fractions.

    Derived, never stored: a stored share table is wrong from the moment a partition is
    registered or retired, and the arithmetic is one line."""
    r = ratios(fams)
    tot = sum(r.values())
    return {p: v / tot for p, v in r.items()} if tot > 0 else {p: 0.0 for p in r}


check_complete()
