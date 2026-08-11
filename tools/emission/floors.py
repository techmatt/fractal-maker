"""floors.py — THE stage-2 emission cuts. One owner, head-stamped, imported everywhere.

THE FOUR STAMPED CUTS ARE ANNOTATION-ONLY AS OF 2026-08-09 (prompts/selection_restructure_1.md).
Nothing in stage 2 removes a row on a stamped per-head floor any more. What removes rows is
TWO semantic constants declared below — `JUNK_FLOOR` (0.20, "don't spend colorize compute on
this") and `GOOD_FLOOR` (0.50, "this is worth keeping"). They are the same comparison against
the judging head's stored raw P(>=3) at two heights: the first at ONE site (drawing the
colorize pool), the second on the whole RUN side, where it replaced the per-partition `t_good`
served predicate on 2026-08-09 (prompts/selection_restructure_3.md). The four
Floor objects keep their values, their stamps and an `annotates()` comparison — they are read
to ANNOTATE ("this row would have failed the retired 0.90 wallpaper release floor") on release
records and sheets, so the value of the old cut stays inspectable instead of being deleted and
argued about from memory. A Floor CANNOT cut: `gate()` and the `acts` flag beside it were
deleted on 2026-08-09 once `acts` had been False on all four for a day, because a switch nobody
may flip next to a method named for flipping it is an invitation. The gate-lock records (`data/render_mode_head/v1/mining_gate_lock.json`)
stay tracked as provenance of what those cuts were measured to buy.

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
wherever it lands. So each cut carries `head` + `stamp`, and `annotates()` REFUSES (raises
`HeadStampMismatch`) when the live pin disagrees with the stamp. Refusing is the whole point:
an annotation on the wrong scale is silent, produces a plausible-looking column, and is only
visible much later as "that counterfactual never made sense". A raised exception is not.

The stamp check is torch-free by construction. The live head versions come from
`tools/wallpaper/wallpaper_pins.py`, `tools/mining/mining_pins.py` and
`production_pins.ACTIVE_VERSION` — three pin sources that hold no model — precisely so
the pure readouts can run it. A check that costs a torch import is a check that gets skipped
on the paths that most need it.

THE FOUR CUTS — all four ANNOTATION-ONLY since 2026-08-09
-------------
                        value  head                     (was)
  wallpaper pool       0.4698  wallpaper_head/v4b        0.75 on v3
  wallpaper release    0.6052  wallpaper_head/v4b        0.90 on v3
  mining pool          0.0     render_mode_head/v3       0.3402, 0.25 on v1
  mining release       0.0949  render_mode_head/v3       0.6691, 0.50 on v1

All four moved on 2026-08-11 at the two-head flip (prompts/flip_29.md), and none of THOSE four
moves was a retune: each value was the VOLUME-MATCHED restatement of its predecessor — the
score that admits the same FRACTION of a fixed reference pool under the new head
(classifier_retrain_protocol.md §5a, tools/scoring/volume_match.py). The volume each cut was
chosen for is invariant across a flip; the precision beside it is not, and that difference is
the head's, not the cut's.

THE TWO MINING CUTS MOVED AGAIN THE SAME DAY, AND THAT SECOND MOVE IS NOT A VOLUME MATCH.
`prompts/audit_mining_process.md`: sheet F's 200 human tiers put the isotonic crossover of
`1[label >= 2]` against the mining head's own gate signal at p_ge3 **0.0949**, and Matt's
pre-stated decision was to land the gate there. Volume is an OUTPUT of a crossover, not a
constraint on it, and it moved by 4.6x (129 -> 587 of the 827 reference-pool rows). The pool
floor followed to 0.0 because a pool floor is defined relative to its release floor and
`check_below_gate` refuses the inversion — see each cut's own `basis`. Every number behind the
crossover is a CEILING: sheet F was a v3-prefilled, score-sorted correction page.

The paragraphs below are the record of what each cut WAS and what it was measured to buy. They
are kept verbatim rather than rewritten in the past tense: the numbers are still the ones the
annotations report, and a cut whose basis has been paraphrased away cannot be un-retired.

THE MINING RELEASE FLOOR STOPPED BEING REPORT-ONLY ON 2026-08-06. It went report-only
because nobody could say what 0.50 bought on strange renders — the head was uncalibrated on
that population and the July lock that would have said so did not survive the corpus loss.
It is now measured: on the 422-row eval side of
`2026-08-06_render_mode_fresh_sheet_v1`, v1 at 0.50 fires 33/422 (7.8%) at precision 97.0%
[84.7%–99.5%] and recall 50.8%, against a 14.9% base rate. The frozen record of that ladder
— both boundaries, both cuts, the head and batch identity, and the two caveats that make
every number an OPTIMISTIC bound — is `data/render_mode_head/v1/mining_gate_lock.json`
(`tools/mining/lock_mining_gate.py`), and its readers refuse when the pin moves off v1 for
the same reason `Floor.annotates` does.

The flip has a cost, paid on purpose and named here so it is not rediscovered: the gate
report's free false-cut signal is gone. While the floor was report-only, a `would_cut` row
could still be SELECTED, so `would_cut ∧ selected` accrued a labeled false-cut count on
every run. Enforcing makes selection imply passing, so that join — and the pool site's
`would_cut_pool ∧ selected`, since 0.25 < 0.50 — is now zero BY CONSTRUCTION. The log keeps
accruing (it is the population record of every scored strange candidate and what each floor
did to it); what it no longer produces is precision without labels. See
`tools/mining/gate_report.py`.

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
`docs/design/q4_harvest_emission.md` and `descriptor.FLOOR_ADMIT_SOURCES`.

    from tools.emission import floors as F
    F.WALLPAPER_RELEASE.annotates(p_ge3)   # checks the stamp, then compares
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


# --------------------------------------------------------------------------- #
# THE ONE ENFORCING CUT (2026-08-09) — the junk floor.
# --------------------------------------------------------------------------- #
# ONE semantic constant, not a per-partition derivation and not one value per head. It says
# the same thing everywhere it is read: **a candidate the judging head is confident is junk
# must not spend colorize compute**. The judging head is whichever head owns the score at that
# site — the stage-1 location head for emission intake, the mining head for `deploy_tail`'s
# allocation draw — and the constant does not change when the head does.
#
# ONE SITE. It applies where the COLORIZE POOL IS DRAWN and nowhere else. Not at pool
# admission, not at release: those are the four stamped cuts above and they annotate now.
# Two readers today, both drawing a colorize pool:
#     tools/emission/ranked_intake.py   (stage-1 head `p_good`, the emission intake draw)
#     tools/mining/deploy_tail.py       (mining head `p_ge3`, the allocation input draw)
#
# 0.20 IS DELIBERATELY COARSE. It is not an operating point and no eval derived it; it is the
# "confidently junk" end of a CORN P(>=3) scale, chosen so it removes the obvious waste and
# leaves every judgement of quality to the human at the sheet. Do not re-derive it against an
# eval and do not make it per-partition — a per-partition derivation is exactly the frozen
# enforcing state this restructure removed.
#
# PERMANENT SHARED-SCALE — IT IS NEVER RESTATED AT A HEAD FLIP (Matt, 2026-08-11,
# prompts/closure_sweep.md). This settles the residual `classifier_retrain_protocol.md` §5a
# stated and did not fix, and it is a DECISION about what the constant is, not a deferral.
#
# The two readers are on two different heads' scales — `ranked_intake` on the stage-1 location
# head's `p_good`, `deploy_tail` on the mining head's `p_ge3` — so there is no single scale to
# volume-match onto. Matching it to whichever head just flipped would move the cut at the OTHER
# site by exactly the amount nobody measured, and the alternative (one constant per head) buys
# a per-head operating point for a cut that was deliberately chosen not to be one.
#
# So 0.20 is read as a COARSE SEMANTIC floor, valid on any CORN P(>=3) scale: "the judging head
# is confident this is junk". It is the one cut in stage 2 whose meaning does not depend on
# which head is live, which is why it is also the one cut a flip leaves alone. Its cost is
# named rather than hidden: the exact VOLUME it removes drifts a little at each flip, and that
# is accepted because the floor's job is removing obvious waste, not holding a rate. Contrast
# `GOOD_FLOOR` below and the four stamped floors above — all single-scale, all volume-matched
# at a flip (`tools/scoring/volume_match.py`, `classifier_retrain_protocol.md` §5a).
JUNK_FLOOR = 0.20

# --------------------------------------------------------------------------- #
# THE OTHER ENFORCING CUT (2026-08-09) — the good floor, on the RUN side.
# --------------------------------------------------------------------------- #
# ONE semantic constant again, and deliberately the same SHAPE of statement as `JUNK_FLOOR`:
# a comparison against the judging head's STORED RAW P(>=3), not a verdict frozen into a row.
# It says **this candidate is good enough to keep** — the thing the per-partition `t_good`
# served predicate used to say, said once for every partition instead of nine times with nine
# numbers.
#
# WHAT IT REPLACED (prompts/selection_restructure_3.md). Until 2026-08-09 the run side admitted
# on `score_lib.corn_decode(p_notbad, p_good, t_good_for(partition), p_ge4) >= 3`, where
# `t_good_for` was a per-partition DERIVED threshold (`production_seeder.T_GOOD_OVERRIDES`,
# re-swept at every head flip out of `tools/scoring/derive_t_good.py`). That machinery is gone.
# It cost a labeled per-partition sweep per flip, it froze its answer into every ledger row it
# stamped, and — the reason it had to go rather than be maintained — it was a SECOND quality
# definition, incompatible with the read-time one selection had already moved to. A location
# could be "admitted" under a 0.90 mandelbrot cut and "junk" under a 0.20 read-time floor, and
# no single sentence described what the pipeline meant by good.
#
# ONE DEFINITION, TWO HEIGHTS. `JUNK_FLOOR` and `GOOD_FLOOR` are the same predicate at two
# points on one scale — "don't spend colorize compute on this" and "this is worth keeping" —
# and both read the stored raw probability at read time. Nothing between them is frozen.
#
# 0.50 IS THE CORN SCALE'S OWN MIDPOINT and that is the whole of its derivation. It is not an
# operating point, no eval chose it, and it must not be re-derived against one: a per-partition
# derivation is exactly the state this restructure removed. The human at the sheet still does
# every judgement of quality; this only decides what the run bothers to keep and count.
#
# NOT `corn_decode(...) >= 3`. The old rule ANDed a fixed `p_notbad >= 0.5` gate onto the
# `p_good` cut, so a frame with P(>=3) = 0.6 and P(>=2) = 0.4 (CORN's cumulative probabilities
# are not guaranteed monotone) decoded to class 2 and was refused. This floor reads P(>=3)
# alone, exactly as read-time selection does. The disagreement is a knife-edge set and the
# point is not its size — it is that there is now one comparison to reason about.
#
# HEAD-FLIP RULE — AND IT IS *NOT* THE SAME ONE AS `JUNK_FLOOR` (corrected 2026-08-11; the two
# were stated as one rule until Matt declared that one permanent shared-scale). A CORN
# probability scale is train-prior-calibrated, so 0.50 on v11 is not 0.50 on v12. This floor has
# ONE reader on ONE head — the run side, `production_seeder.is_good` / `steered_frontier.admit`
# / `descriptor.load_admitted`, all on the stage-1 location head — so a flip HAS a correct move
# and must take it: RESTATE this floor VOLUME-MATCHED, recomputing the score that keeps the same
# FRACTION of a fixed reference pool under the new head. `JUNK_FLOOR` is read on two heads at
# once and has no such move, which is exactly why it is fixed and this one is not. Re-scoring
# the ledgers (`tools/emission/ledger_rescore.py`) and volume-matching THIS floor is the whole
# flip procedure on this axis now — no sweep to re-run and no table to re-adopt.
GOOD_FLOOR = 0.50

# CORN'S OTHER TWO CUTPOINTS, beside the good floor because the three together are the whole
# run-side quality vocabulary — but they are a DIFFERENT KIND of number and that is why neither
# is called a floor. `GOOD_FLOOR` decides what a run KEEPS and is a policy somebody chose;
# these two are the head's own natural rank cutpoints (P >= 0.5 on a cumulative rank
# probability), they have NEVER been calibrated per family, and they decide only what a frame
# is CALLED. Three sites need them:
#
#   NOTBAD_CUT  the julia sub-descent hook's parent gate — "does this walk show any structure
#               at all", counted over the parent's un-reframed raw frames.
#   GREAT_CUT   the pop-quota currency (`pop_quota.CLASS_WEIGHT`: a class 4 is worth ten class
#               3s) and the class split a run's start-up cloud diagnostic prints.
#
# THEY SURVIVE THE t_good RETIREMENT UNCHANGED, and that is not an oversight — only the q3
# operating point was ever swept, because only q3 gated admission
# (`data/v8/t_good_derivation.json` `no_class4_threshold`). They are declared here rather than
# left as bare 0.5s at their call sites so the surface of "numbers that decide something about
# a frame" is one file, and so a future decision to calibrate one has somewhere to land.
NOTBAD_CUT = 0.50
GREAT_CUT = 0.50

# THE THIN-SUPPLY DIVISOR, beside the floor because it is the same kind of number: coarse, not
# derived, and about volume rather than quality. A partition emits at most
# `floor(passing_supply / THIN_SUPPLY_DIVISOR)` — so a partition whose floor-passing supply is
# thin ships nothing rather than shipping its own least-bad row. r=4 says "show me one only if
# there were four to choose from". Read by `ranked_intake.emit_cap`.
THIN_SUPPLY_DIVISOR = 4

# At most this many release picks per morph cluster PER RUN (release selection, §4). A cap, not
# a quota: a cluster with one strong row still ships one.
CLUSTER_CAP = 2

# COLORIZE ATTEMPTS PER RELEASE SLOT — the size of the surplus each head is asked to build
# (2026-08-09, prompts/selection_restructure_2.md). A head's attempt budget is
# `ATTEMPT_MULTIPLIER × that head's release slots`, so the two heads are sized against RELEASE
# NEED and never against each other. Read by `tools/emission/attempt_budget.py`.
#
# THE FAILURE IT FIXES. Colorize volume used to fall out of the joint deficit model, which
# spreads over (partition × cluster × flavor × STYLE) with one smooth style against N promoted
# strange ones — so smooth drew ~1/N of the attempts whatever the release asked for. The
# selrestruct_1 smoke got 3 smooth rows out of 30 attempts against 6 smooth slots and
# short-filled 3 of them: a release the supply could have filled, starved by an allocation
# rule that had no opinion about the release.
#
# 4 IS THE SAME KIND OF COARSE AS THE THREE ABOVE — not derived from a pass rate, not per head
# and not per partition. It says "colorize four candidates for every slot you mean to fill",
# which is the same shape of statement as THIN_SUPPLY_DIVISOR's "show me one only if there were
# four to choose from". The two 4s meeting is not a coincidence worth collapsing into one
# constant: they are about different populations (attempts spent vs candidates available), and
# a partition whose slot allocation respects its emit cap has `4·slots <= 4·floor(supply/4) <=
# supply`, i.e. the attempt budget fits inside the floor-passing supply exactly when the two
# rules agree. Moving either one alone is a real change and reads as one.
ATTEMPT_MULTIPLIER = 4


def passes_junk_floor(score) -> bool:
    """THE junk-floor comparison. `score >= JUNK_FLOOR`, with a missing score reading as NOT
    passing — an unscored candidate has no verdict to spend compute on.

    A function rather than a bare `>=` at each site for the same reason `Floor.gate` is one:
    the floor-admit bypass (`ranked_intake`) and the head-flip restatement above both have to
    be reasoned about against one comparison, not two spellings of it."""
    return score is not None and float(score) >= JUNK_FLOOR


def passes_good_floor(score) -> bool:
    """THE good-floor comparison. `score >= GOOD_FLOOR` on the stored raw P(>=3), with a
    missing score reading as NOT passing — an unscored or guard-zeroed candidate has no
    verdict to be kept on.

    THE run-side admission predicate and THE run-side bookkeeping predicate: every "admitted",
    "servable supply", census count and seed check on the discovery path goes through this one
    function rather than through a stored `decoded_class`. That is what makes a floor move a
    one-line change instead of a re-score of every ledger ever written."""
    return score is not None and float(score) >= GOOD_FLOOR


def good_class(p_good, p_great=None):
    """The run-side CLASS of a scored frame: `None` below `GOOD_FLOOR`, `4` when it also
    clears `GREAT_CUT` on P(>=4), else `3`.

    For the two sites that need a class rather than a yes/no (see `GREAT_CUT`). Deliberately
    NOT a 1..4 decode: below the floor there is no class, because the run keeps no verdict
    about how bad a thing it did not keep is. `p_great=None` (a K=3 head) can only reach 3."""
    if not passes_good_floor(p_good):
        return None
    return 4 if (p_great is not None and float(p_great) >= GREAT_CUT) else 3


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

    A Floor CANNOT REMOVE A ROW. It carries a value and a stamp so the counterfactual it logs
    (`tools/mining/gate_report.py`, the release record's `would_pass_floor` column) stays
    readable as calibration signal, and `annotates()` is the only comparison it offers.

    It used to offer `gate()` and an `acts: bool` saying whether that gate fired. Both went on
    2026-08-09 (prompts/selection_restructure_3.md): `acts` had been False on all four since
    the read-time restructure a day earlier, and a field whose only legal value is False is a
    switch nobody may flip, sitting next to a method named for flipping it. A caller that
    wants a row removed reaches for `passes_junk_floor` / `passes_good_floor` — the two
    constants that actually cut — and cannot get there from here by accident."""
    name: str
    value: float
    head: str
    stamp: str          # head version this value was set against ("v3", "v1", "v10")
    site: str           # "pool" | "release"
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

    def annotates(self, score) -> bool:
        """`score >= value`, AFTER the stamp check. THE comparison — every consumer calls this
        rather than reading `.value` and writing its own `>=`, so the refusal cannot be
        bypassed by a site that only wanted "the number". Named for what it does: the answer
        is written onto a record or a sheet, never used to drop a row."""
        self.check()
        return score is not None and float(score) >= self.value

    def __str__(self) -> str:
        return f"{self.head}/{self.stamp} {self.site} {self.value:g}"


# --------------------------------------------------------------------------- #
# The cuts.
# --------------------------------------------------------------------------- #
WALLPAPER_POOL = Floor(
    name="wallpaper_pool", value=0.4698, head=WALLPAPER_HEAD, stamp=_wp.HEAD_VERSION,
    site="pool",
    basis="permissive inventory bar, set below the gate so a weak-but-real smooth wallpaper "
          "stays available to a later re-selection instead of being discarded at colorize "
          "time. Not a quality claim; the release floor is. 0.75 -> 0.4698 at the 2026-08-11 "
          "v4b flip, VOLUME-MATCHED: 0.4698 pools the same 503 of 1,337 reference-pool rows "
          "(37.6%) that 0.75 pooled on v3, precision>=3 0.748 -> 0.706 there. "
          "data/wallpaper_head/v4b/volume_match_wallpaper.json.")

WALLPAPER_RELEASE = Floor(
    name="wallpaper_release", value=_wp.GATE_THRESHOLD, head=WALLPAPER_HEAD,
    stamp=_wp.HEAD_VERSION, site="release",
    basis="IS the wallpaper head's production gate (wallpaper_pins.GATE_THRESHOLD), imported "
          "not copied. v3 eval precision of passers 0.68@0.90 vs 0.58@0.50; retuned with the "
          "head (prompts/prompt_gate_retune_v3.md).")

MINING_POOL = Floor(
    name="mining_pool", value=0.0, head=MINING_HEAD, stamp=_mn.HEAD_VERSION,
    site="pool",
    basis="CAPACITY ORDERING, not curation: strange colorizes are cheap to make and expensive "
          "to carry, so the bottom of the mining scale is dropped before it reaches the pool. "
          "0.3402 -> 0.0 on 2026-08-11 (prompts/audit_mining_process.md) and this move is a "
          "CONSEQUENCE, not a measurement. A pool floor is defined RELATIVE to its release "
          "floor — the permissive inventory bar, strictly below it, which `check_below_gate` "
          "enforces at import. The sheet-F crossover put the release floor at 0.0949, BELOW "
          "the 0.3402 this cut held, so there was nothing left under the gate for a pool floor "
          "to be permissive about: it had become a second and STRICTER gate, pooling 322 of "
          "the 587 reference-pool rows the gate now passes. 0.0 keeps the invariant and says "
          "plainly that the pool no longer removes anything. Matt's call, taken with the "
          "crossover in front of him. The two measurements it supersedes stay where they were "
          "made: 0.3402 (322/827, precision>=3 0.525) in data/render_mode_head/v3/"
          "volume_match_mining.json, and 0.25 on v1 (70/422 at precision 75.7% "
          "[64.5%-84.2%], keeping 84.1% of the good rows) in v1's lock.")

MINING_RELEASE = Floor(
    name="mining_release", value=_mn.MINING_GATE_THRESHOLD, head=MINING_HEAD,
    stamp=_mn.HEAD_VERSION, site="release",
    basis="IS the mining head's production gate (mining_pins.MINING_GATE_THRESHOLD), imported "
          "not copied. 0.6691 -> 0.0949 on 2026-08-11 — the label/score CROSSOVER off sheet F, "
          "NOT a volume match: the head did not move, and what moved is what the score is read "
          "to mean. Volume 129/827 -> 587/827 on the flip's reference pool (15.6% -> 71.0%), "
          "precision>=3 0.760 -> 0.363, recall>=3 0.458 -> 0.995; at the >=2 boundary the cut "
          "is actually reading, precision 0.992 -> 0.893 and recall 0.226 -> 0.926. Basis "
          "`[human n=200, prefill-anchored — ceiling]`, frozen in data/render_mode_head/v3/"
          "{baserate_audit,mining_gate_lock}_2026-08-11.json. What 0.6691 bought stays in "
          "data/render_mode_head/v3/mining_gate_lock.json as the rollback record, and the v1 "
          "operating point for 0.50 — 33/422 (7.8%) at precision 97.0% [84.7%-99.5%], recall "
          "50.8%, base rate 14.9% — stays in data/render_mode_head/v1/mining_gate_lock.json.")

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
    """One line per cut — for a run banner or a readout header. The two enforcing floors are
    named first because they are the only lines that remove a row; the four stamped cuts follow
    with their annotation-only marker."""
    return (f"junk_floor {JUNK_FLOOR:g} (ENFORCING, colorize-pool draw) · "
            f"good_floor {GOOD_FLOOR:g} (ENFORCING, run-side admission) · ") + " · ".join(
        f"{f.name} {f.value:g} ({f.head}/{f.stamp}, annotation-only)"
        for f in ALL_FLOORS)


check_below_gate()
