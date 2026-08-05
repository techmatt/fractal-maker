"""floors.py — THE stage-2 emission cuts. One owner, head-stamped, imported everywhere.

Matt, 2026-08-04. Every number that removes a row between "a location was admitted" and "a
wallpaper shipped" is declared here, once, carrying the head it reads and the head VERSION it
was set against. Before this file the same four numbers were re-typed in six places — the
driver's four `DEFAULT_*` constants, `first_release_readout`'s and `reselect_readout`'s
`WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50`, `q4_harvest_readout`'s `STRICT_WP,
STRICT_MN`, and two bare literals inside `report.py`'s surplus arithmetic — so a floor move
was a six-file edit that nothing checked, and the readouts silently annotated the pool
against a floor the driver was no longer using.

WHY A STAMP AND NOT JUST A VALUE
--------------------------------
A floor is a point on ONE head's probability scale. `p_ge3 >= 0.90` means "high precision" on
`wallpaper_head/v3` and means nothing at all on v4 until somebody re-derives it — the head is
retrained, the scale moves, and the number that used to sit at 0.68 eval precision lands
wherever it lands. So each cut carries `head` + `stamp`, and `gate()` REFUSES (raises
`HeadStampMismatch`) when the live pin disagrees with the stamp. Refusing is the whole point:
gating on the wrong scale is silent, produces a plausible pool, and is only visible much later
as "the release got worse". A raised exception at the first gated row is not.

The stamp check is torch-free by construction. The live head versions come from
`tools/wallpaper/wallpaper_pins.py`, `tools/mining/mining_pins.py` and
`corpus_common.active_scorer_version()` — three pin modules that hold no model — precisely so
the pure readouts can run it. A check that costs a torch import is a check that gets skipped
on the paths that most need it.

THE FOUR CUTS
-------------
                       value  head                  acts?
  wallpaper pool        0.75  wallpaper_head/v3     YES — smooth below this is not pooled
  wallpaper release     0.90  wallpaper_head/v3     YES — smooth below this cannot ship
  mining pool           0.25  render_mode_head/v1   YES — capacity ordering (see below)
  mining release        0.50  render_mode_head/v1   NO  — REPORT-ONLY (gate_report.py)

The two RELEASE floors are not literals here: they ARE each head's production gate, imported
from the head's own pin. That was already the stated contract ("default = each head's
production gate") and it was already two copies of one number. Retuning a head's gate now
moves its release floor with it, which is the only behaviour that can be right — a release
floor that lags its own gate is a floor calibrated against a head that no longer exists.

The two POOL floors are this module's own literals, because they are not any head's operating
point: they are the permissive "keep it as inventory" bar, deliberately below the gate.
`check_below_gate()` runs AT IMPORT and raises if a pool floor ever reaches its release floor
— at which point the pool is not inventory, it is a second copy of the gate.

THE BADNESS FLOOR IS NOT HERE, AND THAT IS THE POINT. `descriptor.FLOOR_PNOTBAD = 0.5` used to
be a fifth live cut, applied to floor-admitted sources (`q4_harvest`, `human_q3plus`). It was
deleted on 2026-08-04, not moved: a floor-admit source is one whose selection signal is
ORTHOGONAL to the head (a human label, or the q4 goodness field), and re-applying the head's
own badness verdict to it is exactly the veto the floor-admit rule exists to prevent. See
`docs/design/q4_harvest_emission.md` and `descriptor.admit_quality`.

    from tools.emission import floors as F
    F.WALLPAPER_RELEASE.gate(p_ge3)        # checks the stamp, then compares
    F.for_style("smooth", site="release")  # the per-head router the driver uses
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.mining import mining_pins as _mn      # noqa: E402  torch-free mining pin
from tools.wallpaper import wallpaper_pins as _wp  # noqa: E402  torch-free wallpaper pin

# Head ids. The string is `<pin dir>/<version>` when rendered — the same spelling the pin
# paths use (`data/wallpaper_head/v3/...`), so a stamp mismatch message names something a
# reader can go look at.
WALLPAPER_HEAD = _wp.HEAD_NAME          # "wallpaper_head"       — smooth renders
MINING_HEAD = _mn.HEAD_NAME             # "render_mode_head"     — promoted strange renders
LOCATION_HEAD = "location_head"         # the ACTIVE_CKPT quality head (intake-side)


class HeadStampMismatch(RuntimeError):
    """A stage-2 cut was asked to gate while its head's live pin disagrees with the version
    the cut's value was set against. The value is a point on a probability scale that no
    longer exists, so the cut refuses rather than producing a pool nobody can interpret."""


def active_head_version(head: str) -> str:
    """The LIVE version of `head`, read from that head's pin at CALL TIME.

    Call time, not import time, so a test can move a pin with `monkeypatch.setattr` and see
    the refusal — and so a long-running process that somehow outlives a pin flip reads the
    flip rather than its own start-up snapshot. Torch-free on every branch."""
    if head == WALLPAPER_HEAD:
        return _wp.HEAD_VERSION
    if head == MINING_HEAD:
        return _mn.HEAD_VERSION
    if head == LOCATION_HEAD:
        import corpus_common as cc                      # noqa: PLC0415  (torch-free)
        return cc.active_scorer_version()
    raise KeyError(f"no pin registered for head {head!r} — a cut cannot be stamped against a "
                   f"head nothing can report the live version of.")


@dataclass(frozen=True)
class Floor:
    """One stage-2 cut: a threshold on one head's `p_ge3`, stamped with that head's version.

    `acts` records whether this cut actually removes rows today. A report-only cut still
    carries its value and its stamp — the counterfactual it logs (`tools/mining/gate_report.py`)
    is only readable as calibration signal if the scale it was computed on is pinned."""
    name: str
    value: float
    head: str
    stamp: str          # head version this value was set against ("v3", "v1", "v10")
    site: str           # "pool" | "release"
    acts: bool          # False = report-only counterfactual, cuts nothing
    basis: str          # one line: where this number came from

    # -- the stamp check ---------------------------------------------------- #
    def check(self) -> None:
        """Raise `HeadStampMismatch` unless the live head matches this cut's stamp."""
        live = active_head_version(self.head)
        if live != self.stamp:
            raise HeadStampMismatch(
                f"stage-2 floor {self.name!r} = {self.value} was set against "
                f"{self.head}/{self.stamp}, but the live pin is {self.head}/{live}. "
                f"{self.value} is a point on {self.stamp}'s probability scale and says "
                f"nothing on {live}'s — re-derive the floor against {live} and move the "
                f"stamp, or roll the head pin back. Refusing to gate.")

    def gate(self, score) -> bool:
        """`score >= value`, AFTER the stamp check. THE comparison — every consumer calls
        this rather than reading `.value` and writing its own `>=`, so the refusal cannot be
        bypassed by a site that only wanted "the number"."""
        self.check()
        return score is not None and float(score) >= self.value

    def __str__(self) -> str:
        return f"{self.head}/{self.stamp} {self.site} {self.value:g}"


# --------------------------------------------------------------------------- #
# The cuts.
# --------------------------------------------------------------------------- #
WALLPAPER_POOL = Floor(
    name="wallpaper_pool", value=0.75, head=WALLPAPER_HEAD, stamp=_wp.HEAD_VERSION,
    site="pool", acts=True,
    basis="permissive inventory bar, set below the v3 gate (0.90) so a weak-but-real smooth "
          "wallpaper stays available to a later re-selection instead of being discarded at "
          "colorize time. Not a quality claim; the release floor is.")

WALLPAPER_RELEASE = Floor(
    name="wallpaper_release", value=_wp.GATE_THRESHOLD, head=WALLPAPER_HEAD,
    stamp=_wp.HEAD_VERSION, site="release", acts=True,
    basis="IS the wallpaper head's production gate (wallpaper_pins.GATE_THRESHOLD), imported "
          "not copied. v3 eval precision of passers 0.68@0.90 vs 0.58@0.50; retuned with the "
          "head (prompts/prompt_gate_retune_v3.md).")

MINING_POOL = Floor(
    name="mining_pool", value=0.25, head=MINING_HEAD, stamp=_mn.HEAD_VERSION,
    site="pool", acts=True,
    basis="CAPACITY ORDERING, not curation: strange colorizes are cheap to make and expensive "
          "to carry, so the bottom quarter of the mining scale is dropped before it reaches "
          "the pool. Deliberately kept as a hard cut while the release floor above it went "
          "report-only — the would-cut verdict accrues at this site too (gate_report.py).")

MINING_RELEASE = Floor(
    name="mining_release", value=_mn.MINING_GATE_THRESHOLD, head=MINING_HEAD,
    stamp=_mn.HEAD_VERSION, site="release", acts=False,
    basis="IS the mining head's production gate (mining_pins.MINING_GATE_THRESHOLD), imported "
          "not copied. REPORT-ONLY since prompts/mining_gate_report_only.md: the head is "
          "uncalibrated on strange renders, so its verdict is logged against the actual "
          "selection rather than acted on.")

ALL_FLOORS = (WALLPAPER_POOL, WALLPAPER_RELEASE, MINING_POOL, MINING_RELEASE)


# --------------------------------------------------------------------------- #
# Per-style routing — the driver's `head_for_style` question, answered once.
# --------------------------------------------------------------------------- #
WALLPAPER_STYLES = frozenset({"smooth"})


def head_for_style(style: str) -> str:
    """Which head scores a render style. `smooth` is the wallpaper head's training
    distribution; every promoted strange mode goes to the mining head."""
    return WALLPAPER_HEAD if style in WALLPAPER_STYLES else MINING_HEAD


def for_style(style: str, site: str) -> Floor:
    """The cut that applies to `style` at `site` ("pool" | "release")."""
    wp = head_for_style(style) == WALLPAPER_HEAD
    if site == "pool":
        return WALLPAPER_POOL if wp else MINING_POOL
    if site == "release":
        return WALLPAPER_RELEASE if wp else MINING_RELEASE
    raise KeyError(f"no stage-2 cut at site {site!r} (expected 'pool' or 'release')")


# --------------------------------------------------------------------------- #
# Import-time completeness, both directions (the release_mix.check_complete pattern).
# --------------------------------------------------------------------------- #
def check_below_gate(floors=None) -> None:
    """Raise unless every POOL floor sits strictly below its head's RELEASE floor.

    Reads at CALL TIME and defaults to the module globals, so the guard is provably red by
    editing either number and a test can inject a broken pair without editing this file.

    Both halves are real. A pool floor AT the release floor means the pool is no longer
    inventory — every pooled row is release-grade, the "keep the weak ones as inventory"
    contract is silently gone, and `--target-gated`'s post-floor surplus becomes the whole
    pool. A pool floor ABOVE it means rows that could ship were never pooled to be shipped."""
    fl = ALL_FLOORS if floors is None else floors
    by_head_site = {(f.head, f.site): f for f in fl}
    bad = []
    for (head, site), f in by_head_site.items():
        if site != "pool":
            continue
        rel = by_head_site.get((head, "release"))
        if rel is not None and not (f.value < rel.value):
            bad.append(f"{f.name} {f.value} >= {rel.name} {rel.value}")
    if bad:
        raise ValueError(
            f"stage-2 pool floor(s) not below their release floor: {bad}. A pool floor is the "
            f"permissive inventory bar; at or above the release floor it is a second copy of "
            f"the gate, and every downstream 'pooled but not release-grade' count is zero by "
            f"construction rather than by measurement.")


def check_stamps(floors=None) -> None:
    """Raise on the FIRST cut whose stamp disagrees with its live head pin. Not run at import
    (the location head resolves through `corpus_common`, and an import-time raise there would
    make this module unimportable during a flip); called by consumers via `Floor.gate`, and
    available to a test / a run's start-up banner that wants the whole set checked at once."""
    for f in (ALL_FLOORS if floors is None else floors):
        f.check()


def summary() -> str:
    """One line per cut — for a run banner or a readout header."""
    return " · ".join(
        f"{f.name} {f.value:g} ({f.head}/{f.stamp}{'' if f.acts else ', report-only'})"
        for f in ALL_FLOORS)


check_below_gate()
