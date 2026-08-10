r"""The mining gate lock — the frozen operating point the 0.50 release floor is set against.

Writes ``data/render_mode_head/v1/mining_gate_lock.json`` (``mining_pins.LOCK_PATH``) and its
readable face, ``mining_gate_lock.md`` beside it (NOT ``report.md`` — that is the adoption
sitting's own hand-written report and this must never clobber it). A floor is a point on ONE
head's probability scale;
this file is the record of what that point BUYS on a stated population, so a number in
``floors.py`` can be argued with instead of merely obeyed.

WHY THIS FILE WAS REWRITTEN RATHER THAN RE-RUN. The July lock derived its curve from
``data/render_mode_head/v1/seed_*/eval_scores.jsonl`` over
``data/render_mode_corpus/dataset_v1/eval.jsonl``. BOTH are gone — the v1 dir holds only
``model_best.pt`` and the dataset dir does not exist — so the old script could not run, and
had not been able to since the corpus loss. `fresh_sheet_reads.py` still quotes the July
operating point (precision 0.548 / recall 0.195 at 0.50) from the pin module's prose, which
is the only surviving citation of it; that quote is about the LOST corpus and is not
superseded by anything here. This is a different population and says so.

WHAT IT IS DERIVED FROM, AND WHY NOT RE-MEASURED. The source is
``data/render_mode_head/v2/report.json`` — the committed record of the 2026-08-06 sitting,
whose numbers came from scoring the live pin through ``MiningScorer`` over the eval side of
``2026-08-06_render_mode_fresh_sheet_v1``. Re-measuring here would need torch, a GPU and the
crops, and would produce the same numbers or a discrepancy nobody could adjudicate; deriving
from the frozen report keeps this file pure-Python, deterministic, and byte-identical on a
re-run. What it does instead of measuring is REFUSE: the head the report calibrated must be
the head the pin serves, the report's cut values must be the owner's live cut values, and
every quoted cut must be an exact swept row of the frozen ladder.

WHAT THE NUMBERS ARE AN OPTIMISTIC BOUND ON. Two caveats, carried verbatim from the sitting
and repeated in the record itself, both leaning the same way:
  * v1 trained on renders at these same 112 gate-passer locations, so it is read on a
    population it has partly memorised; and
  * the labels were PREFILLED with v1's own suggested tier (a correction sheet, sorted
    good->bad, Enter confirming), so label and score are coupled by construction.
Neither is subtractable here. Every precision below is therefore an optimistic bound on a
FRESH location, which is the whole reason the record states them rather than filing them.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools import paths as P                      # noqa: E402  storage-class declaration
from tools.emission import floors as F            # noqa: E402  THE stage-2 cut owner
from tools.mining import mining_pins as MP        # noqa: E402  torch-free pin

# The sitting this record freezes. Both paths are committed artifacts.
SOURCE_REPORT = "data/render_mode_head/v2/report.json"
LOCK_PATH = MP.LOCK_PATH                          # data/render_mode_head/v1/mining_gate_lock.json
# The readable face of the SAME record, generated beside it — deliberately not `report.md`,
# which is the sitting's own hand-written adoption report and would be clobbered by a --write.
MD_PATH = str(Path(LOCK_PATH).with_suffix(".md"))
SCHEMA = "mining_gate_lock/v2"                    # v1 = the July lock (curve + parity), gone

# The two cuts this record is the authority for. Read from the owner, never restated: the
# lock quotes what `floors.py` says is live, and refuses if the ladder cannot support it.
LOCKED_CUTS = (F.MINING_POOL, F.MINING_RELEASE)


class LockHeadMismatch(RuntimeError):
    """The lock was measured on a head the live pin no longer serves. Its precision/recall
    numbers are points on that head's probability scale and say nothing on another's."""


class LockDerivationError(RuntimeError):
    """The frozen sitting cannot support the record being asked for — the report calibrated
    a different head than the pin serves, or a live cut is not a swept row of its ladder."""


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
def build_lock(report: dict, *, source_sha: str) -> dict:
    """The record, as a pure function of the frozen sitting + the live pin/owner state."""
    live_head = MP.HEAD_VERSION
    cal = report["calibration"]

    # (1) the head the sitting calibrated must be the head the pin serves.
    if cal["ckpt"] != MP.ACTIVE_MINING_CKPT:
        raise LockDerivationError(
            f"the sitting calibrated {cal['ckpt']} but the live pin is "
            f"{MP.ACTIVE_MINING_CKPT}. A lock is a statement about the DEPLOYED head; "
            f"re-run the calibration against the live pin or move the pin back.")
    for name, block in report["cuts"].items():
        if block["stamp"] != f"{MP.HEAD_NAME}/{live_head}":
            raise LockDerivationError(
                f"the sitting's {name} was stamped {block['stamp']}, live head is "
                f"{MP.HEAD_NAME}/{live_head}.")

    # (2) every locked cut must be the owner's live value AND an exact swept row.
    cuts = {}
    for f in LOCKED_CUTS:
        f.check()                                   # the owner's own stamp refusal
        src = report["cuts"].get(f.name)
        if src is None or abs(float(src["value"]) - f.value) > 1e-12:
            raise LockDerivationError(
                f"{f.name} is {f.value} in floors.py but "
                f"{None if src is None else src['value']} in the sitting. The record may only "
                f"quote a cut the sitting actually swept at that value.")
        r3 = _row_at(cal["ladder_ge3"], f.value)
        cuts[f.name] = {
            "value": f.value, "site": f.site, "acts": False,
            "head": f"{f.head}/{f.stamp}", "basis": f.basis,
            "boundary": "p_ge3 (marginal P(label>=3))",
            "fires": r3["fires"], "n": cal["n"], "pass_rate": r3["pass_rate"],
            "tp": r3["tp"], "precision": r3["precision"],
            "precision_ci95": [r3["precision_lo"], r3["precision_hi"]],
            "recall": r3["recall"],
        }

    return {
        "schema": SCHEMA,
        "what": ("The operating point of the render-mode (mining) quality gate: what each "
                 "live cut fires at, and at what measured precision/recall, on the one "
                 "labeled population that exists for this head."),
        "supersedes": {
            "path": LOCK_PATH,
            "note": ("the July lock at this path (frozen PR curve + deployed-scorer parity "
                     "over data/render_mode_corpus/dataset_v1/) did not survive the corpus "
                     "loss; its inputs are gone and it cannot be re-derived. Its operating "
                     "point (precision 0.548 / recall 0.195 / pass-rate 0.050 at 0.50, base "
                     "0.139) survives only as prose quoted in fresh_sheet_reads.JULY_LOCK, "
                     "measured on a DIFFERENT and genuinely held-out population."),
        },
        "gate": {
            "version": MP.MINING_GATE_VERSION,
            "checkpoint": MP.ACTIVE_MINING_CKPT,
            "threshold": MP.MINING_GATE_THRESHOLD,
            "rollback": MP.MINING_V1_ROLLBACK,
            "signal": "marginal p_ge3 = cumprod(sigma(logits)) — NEVER the CORN conditional",
            "deploy_transform": ("classifier.data.Transform(train=False): 384x224 bicubic "
                                 "stretch + the checkpoint's own mean/std"),
            "black_gate": "accept iff black_fraction < 0.30 (parity with the Rust render path)",
        },
        # The identity a reader must match before believing any number below.
        "head": {"name": MP.HEAD_NAME, "version": live_head,
                 "checkpoint": MP.ACTIVE_MINING_CKPT,
                 "role": "LIVE mining gate (mining_pins.ACTIVE_MINING_CKPT)"},
        "corpus": {
            "batch": report["batch"],
            "slice": report["slice"],
            "n": cal["n"],
            "n_locations": report["n_locations"],
            "n_modes": report["n_modes"],
            "labels": report["label_dist"]["hist"],
            "base_rate_ge3": cal["base_rate_ge3"],
            "base_rate_ge2": cal["base_rate_ge2"],
            "label_scale": "K=3 (1 bad / 2 okay / 3 good)",
        },
        "cuts": cuts,
        # BOTH boundaries, whole. A record that froze only the cut rows could not answer
        # "what would 0.40 have bought" without re-running a sitting whose crops may be gone.
        "ladder_ge3": cal["ladder_ge3"],
        "ladder_ge2": cal["ladder_ge2"],
        "caveats": {k: v for k, v in report["caveats"].items()
                    if k in ("eval_is_held_out_for_v2_only", "labels_are_anchored_to_v1",
                             "direction")},
        "bound": ("OPTIMISTIC. Both caveats above inflate v1 and neither is subtractable "
                  "from these numbers: the head trained at these locations, and the labels "
                  "were prefilled with its own suggestions. Every precision here is an upper "
                  "bound on what the same cut buys at a FRESH location, and the honest use "
                  "of this record is as a ceiling, not an estimate."),
        "harness_parity": report["harness_parity"],
        "winner_rule": {
            "winner": report["winner_rule"]["winner"],
            "rule": report["winner_rule"]["rule"],
            "note": ("v2 (a finetune of v1 on this batch's train side) lost this rule, so the "
                     "calibration — and this lock — are on the incumbent."),
        },
        "provenance": {
            "source_report": SOURCE_REPORT,
            "source_report_sha256": source_sha,
            "derived_by": "tools/mining/lock_mining_gate.py --write",
            "sitting": report["batch"].split("_")[0],
            "adoption": ("prompts/mining_adoption_prompt.md — the release floor went from "
                         "report-only to enforcing on this record's numbers."),
        },
    }


def derive(source: Path | None = None) -> dict:
    src = Path(source) if source else (ROOT / SOURCE_REPORT)
    if not src.exists():
        raise LockDerivationError(
            f"the sitting record {src} is missing — it is a tracked artifact "
            f"({SOURCE_REPORT}); restore it rather than re-deriving this lock from anything "
            f"else.")
    return build_lock(json.loads(src.read_text(encoding="utf-8")), source_sha=_sha256(src))


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
    w = [f"# Mining gate lock — `{g['version']}` @ {g['checkpoint']}\n",
         f"Frozen operating point of the render-mode (strange) quality gate. Written by "
         f"`tools/mining/lock_mining_gate.py` from the committed sitting record "
         f"`{lock['provenance']['source_report']}` "
         f"(sha256 `{lock['provenance']['source_report_sha256'][:16]}…`); nothing here is "
         f"re-measured, and a reader that finds the pin moved off "
         f"`{lock['head']['name']}/{lock['head']['version']}` must refuse the whole file.\n",
         f"\n**Population.** `{cor['batch']}` — {cor['slice']}, **n = {cor['n']}** "
         f"({cor['n_locations']} locations, {cor['n_modes']} roster modes), labels "
         f"{cor['labels']} on {cor['label_scale']}: base rate **{_pct(cor['base_rate_ge3'])}** "
         f"at >=3, **{_pct(cor['base_rate_ge2'])}** at >=2.\n",
         "\n## The two cuts\n",
         "| cut | value | site | acts | fires | pass rate | precision | 95% CI | recall |",
         "|---|--:|---|:-:|--:|--:|--:|--:|--:|"]
    for name, cut in c.items():
        lo, hi = cut["precision_ci95"]
        w.append(f"| `{name}` | {cut['value']:.2f} | {cut['site']} | "
                 f"{'YES' if cut['acts'] else 'no'} | {cut['fires']}/{cut['n']} | "
                 f"{_pct(cut['pass_rate'])} | {_pct(cut['precision'])} | "
                 f"{_pct(lo)}–{_pct(hi)} | {_pct(cut['recall'])} |")
    w.append(f"\nBoth are on the gate signal — {c['mining_release']['boundary']}. Precision is "
             f"of PASSERS and carries a Wilson interval: the top of the ladder is estimated "
             f"from a handful of rows, and a bare 1.000 over 3 and a 0.90 over 90 are the "
             f"same column otherwise.\n")

    w.append("\n## What this is an optimistic bound on\n")
    for k, v in lock["caveats"].items():
        w.append(f"- **{k}** — {v}")
    w.append(f"\n**{lock['bound']}**\n")
    hp = lock["harness_parity"]
    w.append(f"\n**Harness parity.** {hp['what']} Max abs diff over {hp['n']} rows: "
             f"{hp['max_abs_diff']:.2e} (tolerance {hp['tol']:.0e}) — "
             f"**{'PASS' if hp['ok'] else 'FAIL'}**. These are the gate's own numbers, not a "
             f"sibling scorer's.\n")

    for key, label, base in (("ladder_ge3", ">=3 (the gate boundary)", cor["base_rate_ge3"]),
                             ("ladder_ge2", ">=2 (not-bad)", cor["base_rate_ge2"])):
        w.append(f"\n## Frozen ladder — {label}, base rate {_pct(base)}\n")
        w.append("| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |")
        w.append("|--:|--:|--:|--:|--:|--:|--:|---|")
        for r in lock[key]:
            ci = ("—" if r["precision"] is None
                  else f"{_pct(r['precision_lo'])}–{_pct(r['precision_hi'])}")
            w.append(f"| {r['threshold']:.2f} | {r['fires']} | {_pct(r['pass_rate'])} | "
                     f"{r['tp']} | {_pct(r['precision'])} | {ci} | {_pct(r['recall'])} | "
                     f"{' '.join(r.get('marks', []))} |")

    sup = lock["supersedes"]
    w.append(f"\n## Provenance\n")
    w.append(f"- **Head** {lock['head']['name']}/{lock['head']['version']} — "
             f"{lock['head']['role']}. Threshold {g['threshold']} on {g['signal']}. "
             f"Rollback: {g['rollback']}.")
    w.append(f"- **Winner rule** — the calibration ran on **{lock['winner_rule']['winner']}**. "
             f"{lock['winner_rule']['note']}")
    w.append(f"- **Supersedes** — {sup['note']}")
    w.append(f"- **Adoption** — {lock['provenance']['adoption']}")
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
