r"""The mining gate lock — the frozen operating point the release floor is set against.

Writes ``mining_pins.LOCK_PATH`` (the PINNED head's ``mining_gate_lock.json``) and its
readable face, ``mining_gate_lock.md`` beside it (NOT ``report.md`` — that is the adoption
sitting's own hand-written report and this must never clobber it). A floor is a point on ONE
head's probability scale;
this file is the record of what that point BUYS on a stated population, so a number in
``floors.py`` can be argued with instead of merely obeyed.

WHERE THE NUMBERS COME FROM, AND WHY NOT RE-MEASURED HERE. The source is a committed
MEASUREMENT RECORD — a pass that scored a named reference pool through ``MiningScorer``, the
gate's own harness, so harness parity here is by construction rather than by a separate check.
Re-measuring in this file would need torch, a GPU and the crops, and would produce the same
numbers or a discrepancy nobody could adjudicate; deriving from the frozen record keeps this
file pure-Python, deterministic, and byte-identical on a re-run. What it does instead of
measuring is REFUSE: the head the record was scored on must be the head the pin serves, the
record's cut values must be the owner's live cut values, and every quoted cut must be an exact
swept row of the frozen ladder (both producers union the live cuts into their sweep for exactly
this reason).

THERE ARE TWO KINDS OF SOURCE AND THEY MAKE DIFFERENT CLAIMS — hence ``LockSpec``. A **volume
match** (``tools/scoring/volume_match.py``, run at a head flip) says *the head moved and this
cut keeps its volume*. A **crossover** (``tools/mining/baserate_audit_reads.py``, run when a
labeled slice says what the score MEANS) says *the head did not move and this cut keeps its
meaning, at whatever volume that costs*. Both emit the same record shape, so this file reads
either; what it must not do is describe one as the other, which is why the restatement sentence,
the schema tag, the caveats and the bound are per-spec rather than module constants. The live
spec is resolved from ``mining_pins.LOCK_PATH`` — the pin names its own lock.

A SUPERSEDED LOCK IS RETAINED, NEVER EDITED, AND STOPS BEING VERIFIABLE BY THIS FILE. That is
the intended end state, not a gap: ``data/render_mode_head/v1/mining_gate_lock.json`` is what
0.50 and 0.25 bought on v1, and ``v3/mining_gate_lock.json`` is what 0.6691 and 0.3402 bought
on v3 before the 2026-08-11 base-rate audit (``mining_pins.MINING_LOCK_ROLLBACK``). Each names
floor values that are no longer live, so re-deriving either would raise — correctly. Only the
lock the pin points at is verified.

WHAT THE NUMBERS ARE AN OPTIMISTIC BOUND ON. Every population this file has ever locked is
anchored to a head that suggested its labels, so a precision here is an optimistic bound on
what the same cut buys at a FRESH (location, mode) pair, and the honest use of this record is
as a ceiling, not an estimate. The per-spec ``caveats`` say which leans apply to which pool and
which way each points. ``tools/mining/sheet_e_reverdict.py`` is the unanchored read.

FROZEN-RECORD WRITE RULE. A default run VERIFIES (derives the record and diffs it against
what is on disk, exit 1 on drift); ``--write`` is what writes. That is the shape
``tools/audit/test_frozen_record_writes.py`` pins for every durable record of a past state.

READERS MUST REFUSE ON A PIN MOVE. ``read_lock()`` raises ``LockHeadMismatch`` unless the
live mining pin still names the head the lock was measured on — the same refusal
``floors.Floor.gate`` makes, for the same reason: 0.50 means 97% precision on v1's scale and
means nothing at all on v2's until somebody re-derives it.

    uv run python tools/mining/lock_mining_gate.py            # verify (default)
    uv run python tools/mining/lock_mining_gate.py --write    # freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools import paths as P                      # noqa: E402  storage-class declaration
from tools.emission import floors as F            # noqa: E402  THE stage-2 cut owner
from tools.mining import mining_pins as MP        # noqa: E402  torch-free pin


class LockHeadMismatch(RuntimeError):
    """The lock was measured on a head the live pin no longer serves. Its precision/recall
    numbers are points on that head's probability scale and say nothing on another's."""


class LockDerivationError(RuntimeError):
    """The frozen sitting cannot support the record being asked for — the report calibrated
    a different head than the pin serves, or a live cut is not a swept row of its ladder."""


# --------------------------------------------------------------------------- #
# The sources. One entry per measurement a lock has been derived from; the LIVE one is
# whichever the pin names, so this table is looked up rather than chosen.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LockSpec:
    """One lock: which measurement it freezes, and what kind of claim its cut values make.

    A frozen instance from the start (CLAUDE.md, "writing a builder for one instance") — the
    volume-match lock was the only one for a day, and the crossover lock would otherwise have
    needed a refactor before it could exist."""
    lock_path: str               # == mining_pins.LOCK_PATH when this spec is live
    source_report: str
    schema: str
    restatement_kind: str        # "VOLUME-MATCHED" | "CROSSOVER" — the claim, in one token
    restatement_how: str         # the sentence written into every cut's `restated_from`
    volume_claim: str            # the paragraph under the .md cut table
    adoption: str
    bound: str
    caveats: dict = field(default_factory=dict)
    # What the adopted cut then FORCED somewhere else, decided after the lock was first
    # written. Empty for a lock that forced nothing; a section rather than more `adoption`
    # prose because the decision has its own date and its own alternatives, and the record of
    # a cut is the only place a reader looks for what the cut cost.
    consequences: str = ""


# The population's known leans, per source. A judgement about the reference pool, not a
# computation over it, so they are declared rather than read out of the record.
VOLUME_MATCH_CAVEATS = {
    "incumbent_trained_at_these_locations": (
        "mining v1 trained at the 112 gate-passer locations the v1 sitting and sheet B draw "
        "from, and its dataset is gone so the exact rows cannot be excluded: 630 of the 827 "
        "rows sit at a location v1 has seen."),
    "labels_are_anchored_to_v1": (
        "every sheet in this corpus is a CORRECTION sheet - rows were served with v1's own "
        "suggested tier prefilled and the page sorted by its score, so label and v1's score "
        "are coupled by construction (0.929 of the v1 sitting's labels came back equal to "
        "what was served)."),
    "staged_is_eval_selected": (
        "the pinned checkpoint is the best of five seeds by eval AP>=3 on this very slice, so "
        "a number read here is optimistic for it. The five-seed band is in the (28) report."),
    "direction": (
        "the first two lean toward the INCUMBENT and the third toward the pinned head. They "
        "do not cancel and none is subtractable."),
}

BASERATE_AUDIT_CAVEATS = {
    "the_page_was_prefilled_by_the_head_being_cut": (
        "sheet F is a CORRECTION page - every row was served with v3's own suggested tier "
        "prefilled and the page sorted by v3's readout, and 176 of 200 labels came back equal "
        "to what was served. Label and score are coupled by construction, so the crossover is "
        "where the human agreed with the head, not only where the head is right."),
    "the_draw_however_was_score_unconditioned": (
        "no mining head touched the SELECTION - sheet E's population imported, flat mode "
        "apportionment, a pool palette draw, near-dup ties broken by draw order. That is what "
        "makes the TIER MIX a base rate over the population the gate sees; it does nothing to "
        "un-anchor the labels, because the anchoring is in the page and not in the draw."),
    "nineteen_positives_at_the_gate_boundary": (
        "the >=3 columns are estimated from 19 rows of 200. The cut is read at >=2 (107 of "
        "200) where the sheet has power; every >=3 precision beside it is a wide interval and "
        "the Wilson bounds in the ladder are the honest width."),
    "direction": (
        "the first lean inflates agreement and therefore the sharpness of the crossover; the "
        "third widens intervals rather than moving them. The second is not a lean at all, it "
        "is what makes the base rate readable. None is subtractable. Sheet E "
        "(tools/mining/sheet_e_reverdict.py) is the unanchored bound on the same draw rule."),
}

VOLUME_MATCH_V3 = LockSpec(
    lock_path=f"data/{MP.HEAD_NAME}/{MP.HEAD_VERSION}/mining_gate_lock.json",
    source_report=f"data/{MP.HEAD_NAME}/{MP.HEAD_VERSION}/volume_match_mining.json",
    schema="mining_gate_lock/v3",             # v1 = July (gone); v2 = the 2026-08-06 sitting
    restatement_kind="VOLUME-MATCHED",
    restatement_how="VOLUME-MATCHED - the same matched_volume below, under the previous "
                    "head (classifier_retrain_protocol.md section 5a)",
    volume_claim="Every value is a VOLUME-MATCHED restatement of the `was` column: the same "
                 "number of reference-pool rows passes, and only the precision beside it "
                 "moved.",
    adoption="prompts/flip_29.md - mining v1 -> v3 (the `dedup_weighted` arm), 2026-08-11. "
             "Both cuts were restated volume-matched at this flip; the v1 lock stays at "
             "data/render_mode_head/v1/ as the record of what 0.50 and 0.25 bought on v1.",
    bound="OPTIMISTIC. Two of the three leans above inflate the INCUMBENT and one inflates "
          "the pinned head; none is subtractable from these numbers. Every precision here is "
          "a bound on what the same cut buys at a FRESH (location, mode) pair, and the honest "
          "use of this record is as a ceiling, not an estimate. The unanchored read is "
          "tools/mining/sheet_e_reverdict.py.",
    caveats=VOLUME_MATCH_CAVEATS,
)

BASERATE_AUDIT_V3 = LockSpec(
    lock_path=f"data/{MP.HEAD_NAME}/{MP.HEAD_VERSION}/mining_gate_lock_2026-08-11.json",
    source_report=f"data/{MP.HEAD_NAME}/{MP.HEAD_VERSION}/baserate_audit_2026-08-11.json",
    schema="mining_gate_lock/v4",             # v3 + a top-level `restatement` block
    restatement_kind="CROSSOVER",
    restatement_how="CROSSOVER - the head did NOT move. The cut is where an isotonic fit of "
                    "1[label >= 2] against this same signal reaches 0.5 on 200 human tiers, "
                    "i.e. where a row becomes more likely than not to be at-least-okay. "
                    "Volume is an OUTPUT of that and is not held fixed; contrast "
                    "classifier_retrain_protocol.md section 5a, which is the other question.",
    volume_claim="The `was` column is NOT volume-matched to the new value and the two do not "
                 "pass the same rows: a crossover holds the label MEANING fixed and lets the "
                 "volume move, which here it does by 4.6x on the flip's reference pool "
                 "(129 -> 587 of 827). That is the audit's finding, not a side effect of it.",
    adoption="prompts/audit_mining_process.md - the sheet F base-rate audit, 2026-08-11. "
             "Matt's decision, pre-stated before the labels were read: land the gate at the "
             "crossover. The pool floor followed to 0.0 because a pool floor is defined "
             "relative to its release floor and floors.check_below_gate refuses the "
             "inversion. The un-suffixed lock beside this one stays as the record of what "
             "0.6691 and 0.3402 bought, i.e. the rollback record "
             "(mining_pins.MINING_LOCK_ROLLBACK).",
    bound="OPTIMISTIC, and more so than its predecessor. The crossover is read off a page the "
          "cut head prefilled and sorted, so it is where the human AGREED with v3 - an upper "
          "bound on the separation a fresh (location, mode) pair would show. Basis "
          "[human n=200, prefill-anchored - ceiling]. The unanchored bound on the same draw "
          "rule is tools/mining/sheet_e_reverdict.py.",
    caveats=BASERATE_AUDIT_CAVEATS,
    consequences=(
        "THE JUNK-FLOOR INVERSION, resolved 2026-08-11 (prompts/junk_floor_repoint.md). "
        "Landing the gate at 0.0949 put it BELOW floors.JUNK_FLOOR (0.20), the one enforcing "
        "stage-2 cut, which tools/mining/deploy_tail.py had read since 2026-08-09 to draw its "
        "mining-side colorize pool. The permissive cut had become the strictest one on this "
        "head: the pool draw removed 132 of the 587 gate-good rows on the 827-row reference "
        "pool (455 clear 0.20 against 587 clearing the gate), so the compute-saving floor was "
        "silently overruling the cut this record freezes. "
        "MATT'S DECISION: REPOINT THE READER, not the number. deploy_tail filters its "
        "allocation input through mining_gate.MiningScorer.gate - the gate's own comparison, "
        "so this lock's threshold IS what draws that pool and a future pin flip moves both "
        "together. Realized mining-side colorize pool on the reference pool: 455 -> 587 of "
        "827 (55.0% -> 71.0%), precision>=2 of the drawn set 0.952 -> 0.893, recall>=3 0.958 "
        "-> 0.995 - measured 2026-08-11 on tools.scoring.volume_match.mining_pool scored "
        "through MiningScorer under this pin. That is the flip's 827-row reference pool, the "
        "population the 129 -> 587 volume claim above is read on, NOT sheet F (n=200), which "
        "is what the ladders below are read on. "
        "THE THREE ALTERNATIVES, refused. (1) LEAVE IT STANDING - a documented inversion is "
        "still an inversion, and it makes the gate advisory at the only mining site that "
        "spends compute on the answer. (2) LOWER JUNK_FLOOR to sit under the gate - it is "
        "PERMANENT shared-scale (floors.py; a coarse semantic 'the judging head is confident "
        "this is junk', not an operating point) and moving it would have shifted the stage-1 "
        "intake draw, on a different head's scale, by an amount nobody measured. (3) SPLIT it "
        "per head - buys exactly the per-head operating point the cut was deliberately chosen "
        "not to be, and doubles a constant to avoid changing a reader. "
        "JUNK_FLOOR is untouched at 0.20 and still filters the stage-1 emission intake; it "
        "keeps one live reader, and deploy_tail now only COUNTS with it (the counterfactual "
        "in its run report)."),
)

SPECS = {s.lock_path: s for s in (VOLUME_MATCH_V3, BASERATE_AUDIT_V3)}


def live_spec(lock_path: str | None = None) -> LockSpec:
    """The spec for the lock the PIN names. Looked up, not chosen — a lock whose source this
    file cannot name is a record nothing can re-derive, so an unregistered path raises."""
    p = lock_path or MP.LOCK_PATH
    spec = SPECS.get(p)
    if spec is None:
        raise LockDerivationError(
            f"mining_pins.LOCK_PATH is {p!r}, which is not a registered lock source "
            f"(have {sorted(SPECS)}). Every lock is DERIVED from a committed measurement; "
            f"register the measurement that produced this one rather than hand-writing it.")
    return spec


# The LIVE spec, and the module-level aliases every caller and test already reads. Resolved at
# import from the pin rather than spelled: a lock cannot describe one measurement while living
# beside another. The readable face is generated beside the json — deliberately not `report.md`,
# which is the sitting's own hand-written adoption report and would be clobbered by a --write.
SPEC = live_spec()
SOURCE_REPORT = SPEC.source_report
LOCK_PATH = SPEC.lock_path
MD_PATH = str(Path(LOCK_PATH).with_suffix(".md"))
SCHEMA = SPEC.schema

# The two cuts this record is the authority for. Read from the owner, never restated: the
# lock quotes what `floors.py` says is live, and refuses if the ladder cannot support it.
LOCKED_CUTS = (F.MINING_POOL, F.MINING_RELEASE)


def log(m):
    print(m, flush=True)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row_at(ladder: list, value: float) -> dict:
    """THE swept row at `value`, or raise. Never nearest-bin: a cut quoted against a
    neighbouring threshold is a precision claim about a gate nobody runs."""
    hit = next((r for r in ladder if abs(float(r["threshold"]) - float(value)) < 1e-9), None)
    if hit is None:
        raise LockDerivationError(
            f"cut value {value} is not a swept row of the frozen ladder "
            f"({[r['threshold'] for r in ladder]}). The sitting unions the live cuts into its "
            f"sweep, so a miss means the cut moved after the sitting and has never been "
            f"measured — re-run the sitting's reads, do not interpolate.")
    return hit


# --------------------------------------------------------------------------- #
# derive
# --------------------------------------------------------------------------- #
# The live spec's leans, under the name this module has always exposed them by. An alias, not
# a copy: `test_mining_gate_lock` asserts the record carries exactly this set, and a second
# literal would let the record and the check drift apart.
CAVEATS = SPEC.caveats


def build_lock(vm: dict, *, source_sha: str, spec: LockSpec | None = None) -> dict:
    """The record, as a pure function of the frozen measurement + the live pin/owner state.

    `spec` says what KIND of restatement the cut values are (a §5a volume match or a
    crossover); it defaults to the live one. The two make different claims and the record has
    to make the right one — a crossover described as volume-matched asserts an invariant it
    broke by 4.6x."""
    spec = spec or SPEC
    live_head = MP.HEAD_VERSION
    pool = vm["reference_pool"]

    # (1) the head the pass scored must be the head the pin serves.
    if vm["head"]["incoming"] != MP.ACTIVE_MINING_CKPT:
        raise LockDerivationError(
            f"the measurement pass scored {vm['head']['incoming']} but the live pin is "
            f"{MP.ACTIVE_MINING_CKPT}. A lock is a statement about the DEPLOYED head; "
            f"re-run the pass against the live pin or move the pin back.")
    if vm["head"]["name"] != MP.HEAD_NAME:
        raise LockDerivationError(
            f"the record is about {vm['head']['name']!r}, the pin about {MP.HEAD_NAME!r}.")
    if vm.get("incomplete"):
        raise LockDerivationError(
            f"{spec.source_report} is stamped incomplete (a bounded --limit run). A lock "
            f"derived from a partial pass would state an operating point nobody measured.")

    by_name = {c["name"]: c for c in vm["cuts"]}
    prev_head = Path(vm["head"]["outgoing"]).parent.name

    # (2) every locked cut must be the owner's live value AND an exact swept row.
    cuts = {}
    for f in LOCKED_CUTS:
        f.check()                                   # the owner's own stamp refusal
        src = by_name.get(f.name)
        if src is None or abs(float(src["incoming_value"]) - f.value) > 1e-12:
            raise LockDerivationError(
                f"{f.name} is {f.value} in floors.py but "
                f"{None if src is None else src['incoming_value']} in the measurement "
                f"record. The record may only quote a cut the pass actually placed.")
        r3 = _row_at(vm["ladder_ge3"], f.value)
        cuts[f.name] = {
            "value": f.value, "site": f.site, "acts": False,
            "head": f"{f.head}/{f.stamp}", "basis": f.basis,
            "boundary": "p_ge3 (marginal P(label>=3))",
            "restated_from": {
                "value": src["outgoing_value"], "head": f"{f.head}/{prev_head}",
                "how": spec.restatement_how, "kind": spec.restatement_kind,
                "precision": src["outgoing"]["precision_ge3"]},
            "matched_volume": src["matched_volume"],
            "fires": r3["fires"], "n": pool["n"], "pass_rate": r3["pass_rate"],
            "tp": r3["tp"], "precision": r3["precision"],
            "precision_ci95": [r3["precision_lo"], r3["precision_hi"]],
            "recall": r3["recall"],
        }

    return {
        "schema": spec.schema,
        "what": ("The operating point of the render-mode (mining) quality gate: what each "
                 "live cut fires at, and at what measured precision/recall, on the reference "
                 "pool its value was set against."),
        # WHAT KIND OF CLAIM THE `restated_from` VALUES MAKE. A record that only carried the
        # numbers could be read as either, and the two assert opposite things about volume.
        "restatement": {"kind": spec.restatement_kind, "how": spec.restatement_how,
                        "volume_claim": spec.volume_claim},
        "gate": {
            "version": MP.MINING_GATE_VERSION,
            "checkpoint": MP.ACTIVE_MINING_CKPT,
            "threshold": MP.MINING_GATE_THRESHOLD,
            "rollback": MP.MINING_V1_ROLLBACK,
            "lock_rollback": MP.MINING_LOCK_ROLLBACK,
            "signal": "marginal p_ge3 = cumprod(sigma(logits)) - NEVER the CORN conditional",
            "deploy_transform": ("classifier.data.Transform(train=False): 384x224 bicubic "
                                 "stretch + the checkpoint's own mean/std"),
            "black_gate": "accept iff black_fraction < 0.30 (parity with the Rust render path)",
        },
        # The identity a reader must match before believing any number below.
        "head": {"name": MP.HEAD_NAME, "version": live_head,
                 "checkpoint": MP.ACTIVE_MINING_CKPT,
                 "role": "LIVE mining gate (mining_pins.ACTIVE_MINING_CKPT)"},
        "corpus": {
            "batch": pool["what"],
            "slice": f"{pool['loader']} - {pool['by_batch']}",
            "n": pool["n"],
            "n_locations": pool["n_locations"],
            "labels": pool["tiers"],
            "base_rate_ge3": pool["base_rate_ge3"],
            "base_rate_ge2": pool["base_rate_ge2"],
            "label_scale": "K=3 (1 bad / 2 okay / 3 good)",
        },
        "cuts": cuts,
        # BOTH boundaries, whole. A record that froze only the cut rows could not answer
        # "what would 0.40 have bought" without re-running a pass whose crops may be gone.
        "ladder_ge3": vm["ladder_ge3"],
        "ladder_ge2": vm["ladder_ge2"],
        "caveats": dict(spec.caveats),
        "bound": spec.bound,
        "consequences": spec.consequences,
        "harness_parity": {
            "what": ("BY CONSTRUCTION, not by a separate check: the measurement pass scores "
                     "through mining_gate.MiningScorer - the gate's own scorer - so there is "
                     "no sibling harness for these numbers to disagree with."),
            "scorer": pool["scorer"],
        },
        "provenance": {
            "source_report": spec.source_report,
            "source_report_sha256": source_sha,
            "source_generated": vm["generated"],
            "source_command": vm["command"],
            "derived_by": "tools/mining/lock_mining_gate.py --write",
            "adoption": spec.adoption,
        },
    }


def derive(source: Path | None = None, spec: LockSpec | None = None) -> dict:
    spec = spec or SPEC
    src = Path(source) if source else (ROOT / spec.source_report)
    if not src.exists():
        raise LockDerivationError(
            f"the measurement record {src} is missing — it is a tracked artifact "
            f"({spec.source_report}); write it with its own producer rather than re-deriving "
            f"this lock from anything else.")
    return build_lock(json.loads(src.read_text(encoding="utf-8")), source_sha=_sha256(src),
                      spec=spec)


def serialize(lock: dict) -> str:
    return json.dumps(lock, indent=2) + "\n"


# --------------------------------------------------------------------------- #
# read back
# --------------------------------------------------------------------------- #
def read_lock(path: str | Path | None = None, *, live_version: str | None = None) -> dict:
    """The lock, or `LockHeadMismatch` if the live pin has moved off the head it describes.

    `live_version` is injectable so a test can induce the refusal without moving the pin;
    it defaults to reading `mining_pins` at CALL time (not import time), the same way
    `floors.active_head_version` does and for the same reason."""
    p = Path(path) if path else (ROOT / LOCK_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing. It is a tracked durable record; write it with "
            f"`uv run python tools/mining/lock_mining_gate.py --write`.")
    lock = json.loads(p.read_text(encoding="utf-8"))
    live = live_version if live_version is not None else MP.HEAD_VERSION
    stamped = lock["head"]["version"]
    if live != stamped:
        raise LockHeadMismatch(
            f"{p.name} was measured on {lock['head']['name']}/{stamped}, but the live pin is "
            f"{lock['head']['name']}/{live}. Every precision and recall in it is a point on "
            f"{stamped}'s probability scale — re-derive the operating point against {live} "
            f"and rewrite this record, or roll the pin back. Refusing to serve it.")
    return lock


# --------------------------------------------------------------------------- #
# report.md — the readable face of the same record
# --------------------------------------------------------------------------- #
def _pct(x):
    return "—" if x is None else f"{100.0 * float(x):.1f}%"


def write_md(lock: dict) -> str:
    c, g, cor = lock["cuts"], lock["gate"], lock["corpus"]
    pv = lock["provenance"]
    w = [f"# Mining gate lock \u2014 `{g['version']}` @ {g['checkpoint']}\n",
         f"Frozen operating point of the render-mode (strange) quality gate. Written by "
         f"`tools/mining/lock_mining_gate.py` from the committed measurement record "
         f"`{pv['source_report']}` "
         f"(sha256 `{pv['source_report_sha256'][:16]}\u2026`); nothing here is re-measured, and "
         f"a reader that finds the pin moved off "
         f"`{lock['head']['name']}/{lock['head']['version']}` must refuse the whole file.\n",
         f"\n**Population.** {cor['batch']} \u2014 {cor['slice']}, **n = {cor['n']}** "
         f"({cor['n_locations']} locations), labels "
         f"{cor['labels']} on {cor['label_scale']}: base rate **{_pct(cor['base_rate_ge3'])}** "
         f"at >=3, **{_pct(cor['base_rate_ge2'])}** at >=2.\n",
         "\n## The two cuts\n",
         "| cut | value | was | site | acts | fires | pass rate | precision | 95% CI | recall |",
         "|---|--:|--:|---|:-:|--:|--:|--:|--:|--:|"]
    for name, cut in c.items():
        lo, hi = cut["precision_ci95"]
        rf = cut["restated_from"]
        w.append(f"| `{name}` | {cut['value']:g} | {rf['value']:g} ({rf['head']}) | "
                 f"{cut['site']} | {'YES' if cut['acts'] else 'no'} | "
                 f"{cut['fires']}/{cut['n']} | "
                 f"{_pct(cut['pass_rate'])} | {_pct(cut['precision'])} | "
                 f"{_pct(lo)}\u2013{_pct(hi)} | {_pct(cut['recall'])} |")
    w.append(f"\nBoth are on the gate signal \u2014 {c['mining_release']['boundary']}. "
             f"**{lock['restatement']['kind']}:** {lock['restatement']['volume_claim']} "
             f"Precision is of PASSERS and carries a Wilson interval: the top of the ladder is "
             f"estimated from a handful of rows, and a bare 1.000 over 3 and a 0.90 over 90 "
             f"are the same column otherwise.\n")

    w.append("\n## What this is an optimistic bound on\n")
    for k, v in lock["caveats"].items():
        w.append(f"- **{k}** \u2014 {v}")
    w.append(f"\n**{lock['bound']}**\n")
    hp = lock["harness_parity"]
    w.append(f"\n**Harness parity.** {hp['what']} (scorer: `{hp['scorer']}`)\n")

    # Only for a lock whose cut forced a decision elsewhere. Rendered here, above the ladders,
    # because it is about what the frozen cut DID and not about how it was measured.
    if lock.get("consequences"):
        w.append("\n## What this cut forced elsewhere\n")
        w.append(f"{lock['consequences']}\n")

    for key, label, base in (("ladder_ge3", ">=3 (the gate boundary)", cor["base_rate_ge3"]),
                             ("ladder_ge2", ">=2 (not-bad)", cor["base_rate_ge2"])):
        w.append(f"\n## Frozen ladder \u2014 {label}, base rate {_pct(base)}\n")
        w.append("| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |")
        w.append("|--:|--:|--:|--:|--:|--:|--:|---|")
        for r in lock[key]:
            ci = ("\u2014" if r["precision"] is None
                  else f"{_pct(r['precision_lo'])}\u2013{_pct(r['precision_hi'])}")
            w.append(f"| {r['threshold']:.4f} | {r['fires']} | {_pct(r['pass_rate'])} | "
                     f"{r['tp']} | {_pct(r['precision'])} | {ci} | {_pct(r['recall'])} | "
                     f"{' '.join(r.get('marks', []))} |")

    w.append("\n## Provenance\n")
    w.append(f"- **Head** {lock['head']['name']}/{lock['head']['version']} \u2014 "
             f"{lock['head']['role']}. Threshold {g['threshold']} on {g['signal']}. "
             f"Rollback: {g['rollback']}; the lock this one supersedes, kept as the record of "
             f"what the previous cuts bought: `{g['lock_rollback']}`.")
    w.append(f"- **Source** `{pv['source_report']}`, generated {pv['source_generated']} "
             f"by `{pv['source_command']}`.")
    w.append(f"- **Adoption** \u2014 {pv['adoption']}")
    return "\n".join(w) + "\n"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="freeze the record (default: verify the tree's copy and exit 1 on "
                         "drift — a durable record of a past state is not rewritten by a "
                         "default run)")
    args = ap.parse_args()

    lock = derive()
    text = serialize(lock)
    md = write_md(lock)
    out, out_md = P.durable(LOCK_PATH), P.durable(MD_PATH)

    if not args.write:
        if not out.exists():
            log(f"[lock] MISSING: {LOCK_PATH} — run with --write to freeze it.")
            return 1
        drift = [rel for rel, cur, path in ((LOCK_PATH, text, out), (MD_PATH, md, out_md))
                 if not path.exists() or path.read_text(encoding="utf-8") != cur]
        if drift:
            log(f"[lock] DRIFT: {drift} differ(s) from what the committed sitting + the live "
                f"floors derive. Either the pin/floors moved (re-derive with --write and say "
                f"so in the commit) or the record was hand-edited.")
            return 1
        log(f"[lock] OK — {LOCK_PATH} and {MD_PATH} match the derivation "
            f"({lock['gate']['version']} @ {lock['gate']['threshold']}, n={lock['corpus']['n']}).")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    rel = lock["cuts"]["mining_release"]
    log(f"[lock] froze {LOCK_PATH} + {MD_PATH}: {lock['gate']['version']} @ "
        f"{rel['value']} fires {rel['fires']}/{rel['n']} at precision "
        f"{_pct(rel['precision'])} {[_pct(x) for x in rel['precision_ci95']]}, recall "
        f"{_pct(rel['recall'])} — OPTIMISTIC (see caveats).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
