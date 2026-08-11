r"""baserate_audit_reads.py — SHEET F's reads, and the label/score CROSSOVER the cut comes from.

Sheet F (`2026-08-11_render_mode_baserate_audit_v1`, 200 rows) is the base-rate audit: a draw
that no mining head touched, served on a page that v3 prefilled and sorted. The two halves read
two different things and this module keeps them apart:

  * THE DRAW gives the tier mix over the population the mining gate actually sees.
  * THE PAGE couples every label to v3's suggestion, so **every rate here is a CEILING**. The
    unanchored bound on the same draw rule is sheet E (`sheet_e_reverdict.py`), and the pair is
    what the audit reads. A correction rate off this sheet measures agreement with v3, never
    quality — the same statement `build_baserate_sheet` stamps into the batch record.

THE CROSSOVER, AND WHY IT IS NOT A VOLUME MATCH. `classifier_retrain_protocol.md` §5a's
restatement rule answers "the head moved, where does this cut go?" — it holds VOLUME invariant
because the cut's purpose was a volume. This is the other question: "the head did not move; what
does the human say the score MEANS?". So the cut is placed where an isotonic fit of `1[label>=2]`
against the head's own gate signal crosses 0.5 — the score at which a row is more likely than not
to be at-least-okay. Volume is an OUTPUT of that, not a constraint on it, and on this population
it moves by 4.6x on the reference pool. That is the audit's finding, reported rather than softened.

WHAT IS BORROWED AND WHAT IS NEW. The ladder arithmetic, the Wilson-interval precision block, the
midpoint placement and the `>`/`>=` bookkeeping are `tools/scoring/volume_match.py`'s and are
IMPORTED — a second copy of a cut rule is a second chance to change one and not the other. The
§5a MIDPOINT convention carries over unchanged and for the same reason: a threshold placed AT an
observed score admits k under `>=` and k-1 under `>`, so it goes between the two adjacent rows and
the realized volume is RE-COUNTED under the rounded constant that gets written.

THE RECORD IS THE LOCK'S SOURCE. `lock_mining_gate.py` derives the gate lock from a frozen
measurement rather than re-measuring, and this file's json is that measurement for the crossover
adoption — same shape as `volume_match_mining.json` (`cuts`, `ladder_ge3`, `ladder_ge2`,
`reference_pool`, `head`), so the lock builder consumes either without a second code path. The
ladders UNION the live cut values into their sweep, which is what lets `lock_mining_gate._row_at`
refuse an interpolation instead of quoting a neighbouring bin.

MOVES NOTHING. `mining_pins` and `floors` are READ. A constant moves when a human edits its owner.

    uv run python tools/mining/baserate_audit_reads.py            # write the durable record
    uv run python tools/mining/baserate_audit_reads.py --no-pool  # torch-free; sheet F only
    uv run python tools/mining/baserate_audit_reads.py --limit 20 # bounded, -> scratch/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from tools import paths as P                                        # noqa: E402
from tools.emission import floors as F                              # noqa: E402
from tools.mining import mining_pins as MP                          # noqa: E402
# THE arithmetic, imported: ladder + Wilson precision block + the §5a midpoint placement.
from tools.scoring.volume_match import (                            # noqa: E402
    _precision_block, ladder, midpoint_cut, passing_volume)


# --------------------------------------------------------------------------- #
# The instance, frozen from the start (CLAUDE.md, "writing a builder for one instance").
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuditSpec:
    """One base-rate audit sitting: which sheet was labeled, and which blind sheet bounds it."""
    key: str
    batch_id: str
    corpus: str
    labels_rel: str
    blind_batch_id: str          # the UNANCHORED sheet on the same draw rule
    blind_labels_rel: str
    out_rel: str                 # durable dir, under the head the scores are on
    stem: str
    # The committed record of what the constants were BEFORE this audit. Read, never restated:
    # after the edit those values live nowhere in code, and a literal here would be the only
    # copy of a number nothing checks.
    supersedes_rel: str


SHEET_F = AuditSpec(
    key="sheet_f",
    batch_id="2026-08-11_render_mode_baserate_audit_v1",
    corpus="render_mode_corpus",
    labels_rel="labels/render_mode_baserate_audit_v1.json",
    blind_batch_id="2026-08-11_render_mode_blind_v1",
    blind_labels_rel="labels/render_mode_blind_v1.json",
    out_rel=f"data/{MP.HEAD_NAME}/{MP.HEAD_VERSION}",
    stem="baserate_audit_2026-08-11",
    supersedes_rel=f"data/{MP.HEAD_NAME}/{MP.HEAD_VERSION}/volume_match_mining.json",
)

SPECS = {s.key: s for s in (SHEET_F,)}

# The BOUNDARY the crossover is read on. `1[label >= 2]` — "not bad" — against the head's own
# gate signal `p_ge3`. The two are deliberately mismatched and that mismatch IS the reading: the
# gate is a P(>=3) score and the question asked of it here is a >=2 question, which is why the
# crossover lands where it lands. Stated as constants so the record cannot describe one boundary
# and compute another.
CROSSOVER_TARGET = 2                # 1[label >= 2]
CROSSOVER_SIGNAL = "p_ge3"          # marginal P(label >= 3) — the gate's own signal
# The two other readouts every row carries, read the same way as a check that the answer is not
# an artifact of which marginal the fit ran on. Report-only: no cut is placed on them, because
# no gate site compares against them.
CROSS_CHECK_SIGNALS = ("p_ge2", "pred", "score")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _rows(batch_id: str, corpus: str) -> list:
    p = ROOT / "data" / corpus / "batches" / batch_id / "images.jsonl"
    if not p.exists():
        raise SystemExit(f"[audit] no such batch manifest: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_sheet(spec: AuditSpec, limit: int | None = None) -> dict:
    """The labeled sheet as parallel arrays. RAISES on an unlabeled row rather than dropping it:
    a base rate computed over the labeled remainder of a sheet is not visibly wrong."""
    rows = _rows(spec.batch_id, spec.corpus)
    lab = json.loads((ROOT / spec.labels_rel).read_text(encoding="utf-8"))
    missing = [r["image_id"] for r in rows if r["image_id"] not in lab]
    if missing:
        raise SystemExit(
            f"[audit] {len(missing)} of {len(rows)} rows of {spec.batch_id} have no label "
            f"(e.g. {missing[:5]}). A base rate over the labeled remainder is a different "
            f"number, not a smaller one — merge the sitting first "
            f"(tools/wallpaper/merge_sitting.py --corpus {spec.corpus}).")
    rows = sorted(rows, key=lambda r: r["sheet_order"])
    if limit:
        # STRIDED, not truncated. `sheet_order` is DESCENDING head readout, so the first N rows
        # are the top of the score range and every label in them is high — a prefix makes the
        # isotonic fit degenerate (never crosses 0.5) and the bounded run exercises none of the
        # crossover path it exists to smoke. A stride keeps the score range.
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    h = [r["head_mining_v1"] for r in rows]
    return {
        "n": len(rows),
        "image_id": [r["image_id"] for r in rows],
        "label": np.array([int(lab[r["image_id"]]) for r in rows]),
        "suggested_tier": np.array([int(r["suggested_tier"]) for r in rows]),
        "p_ge2": np.array([float(x["p_ge2"]) for x in h]),
        "p_ge3": np.array([float(x["p_ge3"]) for x in h]),
        "pred": np.array([float(x["pred"]) for x in h]),
        "score": np.array([float(x["score"]) for x in h]),
        "ckpt": sorted({x["ckpt"] for x in h}),
        "head_version": sorted({x["head_version"] for x in h}),
        "served_gate_threshold": sorted({float(x["gate_threshold"]) for x in h}),
        "would_pass_served_gate": int(sum(1 for x in h if x["would_pass_gate"])),
        "n_locations": len({r["provenance"]["location_key"] for r in rows}),
        "by_mode": dict(sorted(Counter(r["render"]["render_mode"] for r in rows).items())),
    }


def blind_rates(spec: AuditSpec) -> dict:
    """Sheet E's tier mix, read off ITS committed batch + sidecar — never quoted from a report.

    The unanchored bound on the same draw rule. Read at call time so the comparison row cannot
    outlive the file it describes."""
    rows = _rows(spec.blind_batch_id, spec.corpus)
    lab = json.loads((ROOT / spec.blind_labels_rel).read_text(encoding="utf-8"))
    y = np.array([int(lab[r["image_id"]]) for r in rows if r["image_id"] in lab])
    return _rate_block(y, {"batch": spec.blind_batch_id, "elicitation": "BLIND, shuffled",
                           "n_rows_in_batch": len(rows)})


def _rate_block(y: np.ndarray, extra: dict | None = None) -> dict:
    n = len(y)
    return {**(extra or {}), "n": n,
            "tiers": {str(t): int((y == t).sum()) for t in (1, 2, 3)},
            "n_ge2": int((y >= 2).sum()), "rate_ge2": float((y >= 2).mean()) if n else None,
            "n_ge3": int((y >= 3).sum()), "rate_ge3": float((y >= 3).mean()) if n else None}


# --------------------------------------------------------------------------- #
# (2) the audit read — tier mix and the correction rate against what was SERVED
# --------------------------------------------------------------------------- #
def correction_read(y: np.ndarray, served: np.ndarray) -> dict:
    """Agreement with the v3 prefill, and the flips across each boundary IN BOTH DIRECTIONS.

    Both directions, always: a single "correction rate" hides which way the head is wrong, and
    the two are different defects — a head that over-suggests wastes label budget at the top of
    the page, one that under-suggests loses supply nobody ever sees."""
    n = len(y)
    out = {
        "n": n,
        "exact_tier_agreement": int((y == served).sum()),
        "exact_tier_agreement_rate": float((y == served).mean()) if n else None,
        "confusion_served_by_human": {
            str(s): {str(h): int(((served == s) & (y == h)).sum()) for h in (1, 2, 3)}
            for s in (1, 2, 3)},
    }
    for b in (2, 3):
        up = int(((served < b) & (y >= b)).sum())
        dn = int(((served >= b) & (y < b)).sum())
        out[f"flips_ge{b}"] = {
            "served_below_human_at_or_above": up,      # the head under-called it
            "served_at_or_above_human_below": dn,      # the head over-called it
            "net": up - dn,
            "boundary_agreement": int(((served >= b) == (y >= b)).sum()),
            "boundary_agreement_rate": float(((served >= b) == (y >= b)).mean()) if n else None,
        }
    return out


# --------------------------------------------------------------------------- #
# (3) the crossover
# --------------------------------------------------------------------------- #
def isotonic_blocks(x: np.ndarray, hit: np.ndarray) -> list:
    """The PAVA fit as its level sets: `[{fit, lo, hi, n, positives}]`, ascending in `x`.

    Returned whole rather than as one number because the crossing can land on a TIE BLOCK whose
    fitted value is exactly 0.5 (it does here: 8 rows, 4 positive), and a caller that only saw
    the crossover could not tell that from a clean crossing."""
    from sklearn.isotonic import IsotonicRegression      # noqa: PLC0415  (heavy-ish import)

    o = np.argsort(np.asarray(x, dtype=float), kind="mergesort")
    xs, ts = np.asarray(x, dtype=float)[o], np.asarray(hit, dtype=float)[o]
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(xs, ts)
    blocks: list = []
    for xv, fv, tv in zip(xs, fit, ts):
        if blocks and abs(blocks[-1]["fit"] - float(fv)) <= 1e-12:
            blocks[-1].update(hi=float(xv), n=blocks[-1]["n"] + 1,
                              positives=blocks[-1]["positives"] + int(tv))
        else:
            blocks.append({"fit": float(fv), "lo": float(xv), "hi": float(xv),
                           "n": 1, "positives": int(tv)})
    return blocks


def crossover(x: np.ndarray, y: np.ndarray, *, target: int = CROSSOVER_TARGET,
              ndigits: int = 4) -> dict:
    """Where the isotonic fit of `1[label >= target]` against `x` crosses 0.5.

    ADOPTED CONVENTION: the FIRST fitted value at or above 0.5 — the crossing is the boundary
    between the last block below 0.5 and the first block that reaches it. The constant is the
    MIDPOINT between the two adjacent row scores (§5a: a threshold AT an observed score admits
    k under `>=` and k-1 under `>`), rounded, and the volume is then RE-COUNTED under the
    rounded value.

    `strictly_above` carries the other reading — first fitted value strictly ABOVE 0.5 — because
    a tie block makes "crosses 0.5" ambiguous by exactly one block and a record that reported
    only one of the two would be hiding the ambiguity rather than resolving it."""
    x = np.asarray(x, dtype=float)
    hit = (np.asarray(y) >= target).astype(float)
    blocks = isotonic_blocks(x, hit)
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    fit = np.concatenate([np.full(b["n"], b["fit"]) for b in blocks]) if blocks else np.array([])

    def _at(mask) -> dict | None:
        idx = np.flatnonzero(mask)
        if len(idx) == 0 or idx[0] == 0:
            # never crosses, or crosses below the smallest observed score — either way there is
            # no PAIR of adjacent rows to put a midpoint between, and inventing one would place
            # a constant outside the range anything was measured on.
            return None
        i = int(idx[0])
        raw = float((xs[i - 1] + xs[i]) / 2.0)
        c = round(raw, ndigits)
        return {"first_row_at_or_above": float(xs[i]), "last_row_below": float(xs[i - 1]),
                "fitted_at_crossing": float(fit[i]), "constant_unrounded": raw, "constant": c,
                "realized_volume": passing_volume(x, c, strict=False),
                "realized_volume_strict": passing_volume(x, c, strict=True)}

    adopted = _at(fit >= 0.5)
    if adopted is None:
        raise SystemExit(
            f"[audit] the isotonic fit of 1[label>={target}] never crosses 0.5 strictly inside "
            f"the observed score range, so there is no pair of adjacent rows to place a "
            f"midpoint between. A constant outside the measured range is not a crossover.")
    return {
        "target": f"1[label >= {target}]",
        "n": len(x), "positives": int(hit.sum()), "base_rate": float(hit.mean()),
        "fit": "isotonic (PAVA), increasing, sklearn.isotonic.IsotonicRegression",
        "placement": "MIDPOINT between the adjacent row scores "
                     "(classifier_retrain_protocol.md §5a), then the volume is RE-COUNTED "
                     "under the rounded constant",
        "convention": "first fitted value AT OR ABOVE 0.5",
        **adopted,
        "tie_block_at_exactly_half": any(abs(b["fit"] - 0.5) <= 1e-12 for b in blocks),
        "strictly_above": _at(fit > 0.5),
        "blocks": blocks,
    }


# --------------------------------------------------------------------------- #
# (4) the cut, as the lock's source expects it
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Restatement:
    """One constant being moved: its owner, the value it moves FROM, and the site's comparison."""
    name: str
    owner: str
    outgoing_value: float
    incoming_value: float
    strict: bool
    site: str


def cut_block(r: Restatement, labels: np.ndarray, scores: np.ndarray) -> dict:
    """The per-cut record, in `volume_match.match_cut`'s shape so one lock builder reads both.

    `matched_volume` is the REALIZED volume under the rounded constant — for a crossover there is
    no volume being matched, and the field name is kept only because the lock quotes it. The
    record says so rather than letting the name imply a §5a restatement."""
    return {
        "name": r.name, "owner": r.owner, "site": r.site,
        "comparison": ">" if r.strict else ">=",
        "outgoing_value": r.outgoing_value, "incoming_value": r.incoming_value,
        "n": len(labels),
        "matched_volume": passing_volume(scores, r.incoming_value, strict=r.strict),
        "realized_volume": passing_volume(scores, r.incoming_value, strict=r.strict),
        "volume_preserved": False,
        "matched_volume_note": "REALIZED, not matched — a crossover holds the label MEANING "
                               "fixed and lets the volume move. The 'matched' spelling is the "
                               "field the lock builder reads; §5a's invariant does not apply.",
        "outgoing": _precision_block(labels, scores, r.outgoing_value, strict=r.strict),
        "incoming": _precision_block(labels, scores, r.incoming_value, strict=r.strict),
        "outgoing_ge2": _precision_block(labels, scores, r.outgoing_value, strict=r.strict,
                                         good=2),
        "incoming_ge2": _precision_block(labels, scores, r.incoming_value, strict=r.strict,
                                         good=2),
    }


def reference_pool_volumes(cuts: list, *, limit: int | None = None) -> dict:
    """The SAME 827-row pool the flip's cuts were volume-matched on, re-scored under the live
    pin, so "how much did this move" is answerable against the number the flip published.

    A second population on purpose: sheet F is 200 rows of the gate's INPUT distribution and the
    reference pool is the labeled eval side the constants were last measured on. Neither
    substitutes for the other, and the honest report carries both."""
    from tools.mining.mining_gate import MiningScorer                 # noqa: PLC0415  (torch)
    from tools.scoring.volume_match import mining_pool                # noqa: PLC0415

    jpgs, labels, meta = mining_pool(limit=limit)
    t0 = time.time()
    ms = MiningScorer(model_path=str(ROOT / MP.ACTIVE_MINING_CKPT)).score_paths(list(jpgs))
    p3 = np.array([m.p_ge3 for m in ms], dtype=float)
    return {
        "what": "the (28) deduplicated mining eval side (mining_corpus.load_corpus), re-scored "
                "under the live pin",
        "loader": "tools.scoring.volume_match.mining_pool",
        "scorer": "mining_scorer", **meta,
        "scored_in_s": round(time.time() - t0, 1),
        "tiers": {str(t): int((labels == t).sum()) for t in (1, 2, 3)},
        "base_rate_ge3": float((labels >= 3).mean()), "base_rate_ge2": float((labels >= 2).mean()),
        "cuts": {c["name"]: {
            "outgoing_value": c["outgoing_value"], "incoming_value": c["incoming_value"],
            "outgoing": _precision_block(labels, p3, c["outgoing_value"], strict=False),
            "incoming": _precision_block(labels, p3, c["incoming_value"], strict=False),
            "outgoing_ge2": _precision_block(labels, p3, c["outgoing_value"], strict=False,
                                             good=2),
            "incoming_ge2": _precision_block(labels, p3, c["incoming_value"], strict=False,
                                             good=2)} for c in cuts},
        "junk_floor_inversion": {
            "junk_floor": F.JUNK_FLOOR,
            "fires_at_junk_floor": passing_volume(p3, F.JUNK_FLOOR, strict=False),
            "note": "JUNK_FLOOR is PERMANENT shared-scale and was not moved. With the gate "
                    "below it, the mining-side colorize-pool draw briefly cut rows the gate "
                    "passes; RESOLVED 2026-08-11 by repointing that draw (deploy_tail) at "
                    "MiningScorer.gate, so nothing reads this floor on the mining scale now. "
                    "The count above is the counterfactual: how many of this pool clear 0.20.",
        },
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def superseded_values(spec: AuditSpec) -> dict:
    """`{cut name: value}` as the flip left them, read off the flip's own committed record.

    Derived, not declared. Once the owners are edited these numbers exist nowhere in code, and
    a literal beside the new value is the "hardcoded True" shape — it outlives what it reports
    the moment somebody restates one of the two and not the other."""
    p = ROOT / spec.supersedes_rel
    if not p.exists():
        raise SystemExit(
            f"[audit] the superseded record {spec.supersedes_rel} is missing. It is a tracked "
            f"artifact and is the only record of what the cuts were before this audit; a "
            f"restatement that cannot name what it restates is indistinguishable from a retune.")
    vm = json.loads(p.read_text(encoding="utf-8"))
    return {c["name"]: float(c["incoming_value"]) for c in vm["cuts"]}


def run(spec: AuditSpec, *, limit: int | None = None, with_pool: bool = True,
        ndigits: int = 4) -> dict:
    sheet = load_sheet(spec, limit=limit)
    was = superseded_values(spec)
    y, served = sheet["label"], sheet["suggested_tier"]
    signal = sheet[CROSSOVER_SIGNAL]

    cross = crossover(signal, y, ndigits=ndigits)
    checks = {s: {k: v for k, v in crossover(sheet[s], y, ndigits=ndigits).items()
                  if k != "blocks"} for s in CROSS_CHECK_SIGNALS}

    # The two constants this audit moves, read FROM their owners (never restated) and TO the
    # crossover. `MINING_POOL` is not a second crossover: the pool floor is defined relative to
    # the gate ("permissive inventory bar, strictly below it"), and with the gate at the
    # crossover there is nothing left below it to be permissive about — `floors.check_below_gate`
    # is what makes that a decision instead of a silent inversion.
    cuts = [
        cut_block(Restatement(
            "mining_release",
            "tools/mining/mining_pins.MINING_GATE_THRESHOLD "
            "(== tools/emission/floors.MINING_RELEASE)",
            outgoing_value=was["mining_release"], incoming_value=cross["constant"],
            strict=False, site="release"), y, signal),
        cut_block(Restatement(
            "mining_pool", "tools/emission/floors.MINING_POOL",
            outgoing_value=was["mining_pool"], incoming_value=F.MINING_POOL.value,
            strict=False, site="pool"), y, signal),
    ]
    live = {"mining_release": MP.MINING_GATE_THRESHOLD, "mining_pool": F.MINING_POOL.value}
    drift = [c["name"] for c in cuts if abs(c["incoming_value"] - live[c["name"]]) > 1e-12]
    marks = {c["name"]: c["incoming_value"] for c in cuts}

    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": f"uv run python tools/mining/baserate_audit_reads.py"
                   + (f" --limit {limit}" if limit else "")
                   + ("" if with_pool else " --no-pool"),
        "procedure": "the label/score CROSSOVER, not a §5a volume match: an isotonic fit of "
                     f"1[label >= {CROSSOVER_TARGET}] against the head's own gate signal "
                     f"({CROSSOVER_SIGNAL}), cut where the fitted probability reaches 0.5, "
                     "placed at the §5a midpoint and re-counted under the rounded constant.",
        "head": {"name": MP.HEAD_NAME,
                 # outgoing == incoming: the head did NOT move. What moved is the reading of it.
                 "outgoing": MP.ACTIVE_MINING_CKPT, "incoming": MP.ACTIVE_MINING_CKPT,
                 "served_at": sheet["ckpt"],
                 "served_gate_threshold": sheet["served_gate_threshold"]},
        "supersedes": {"record": spec.supersedes_rel, "values": was,
                       "why": "the 2026-08-11 v1->v3 flip's VOLUME-MATCHED restatement. It is "
                              "not wrong and is not being corrected: it answered 'the head "
                              "moved, where does this cut go?' and this answers 'what does the "
                              "human say the score means?'."},
        "reference_pool": {
            "what": f"sheet F — {spec.batch_id}, the base-rate audit sitting",
            "loader": "tools.mining.baserate_audit_reads.load_sheet",
            "scorer": "mining_scorer",
            "basis": "[human n=200, prefill-anchored — ceiling]",
            "n": sheet["n"], "n_locations": sheet["n_locations"],
            "by_batch": {spec.batch_id: sheet["n"]},
            "by_mode": sheet["by_mode"],
            "tiers": {str(t): int((y == t).sum()) for t in (1, 2, 3)},
            "base_rate_ge3": float((y >= 3).mean()), "base_rate_ge2": float((y >= 2).mean()),
        },
        "audit": {
            "human": _rate_block(y, {"batch": spec.batch_id,
                                     "elicitation": "CORRECTION — v3's tier PREFILLED, page "
                                                    "sorted by v3's readout"}),
            "v3_prefill": _rate_block(served, {"what": "the suggested_tier each row was SERVED "
                                                       "with (suggest_tier_mining.CUTS on v3)"}),
            "sheet_e_blind": blind_rates(spec),
            "correction": correction_read(y, served),
            "would_pass_served_gate": sheet["would_pass_served_gate"],
            "ceiling": "EVERY rate in this block is a CEILING. The page was v3-prefilled and "
                       "score-sorted, so label and score are coupled by construction; a "
                       "correction rate here measures agreement with v3 and never quality "
                       "(classifier_retrain_protocol.md §2b). Sheet E is the unanchored bound "
                       "on the same draw rule.",
        },
        "crossover": cross,
        "crossover_cross_checks": {
            "why": "the same fit against the head's other readouts. No cut is placed on them — "
                   "no gate site compares against them — but if the crossover were an artifact "
                   "of reading a >=2 question off a P(>=3) score, these would disagree about "
                   "WHICH ROWS pass, and they do not.",
            "signals": checks,
        },
        "cuts": cuts,
        "cuts_match_the_live_owners": not drift,
        "cuts_drifted_from_the_live_owners": drift,
        "ladder_ge3": ladder(y, signal, marks, strict=False, good=3),
        "ladder_ge2": ladder(y, signal, marks, strict=False, good=2),
        "incomplete": bool(limit),
    }
    if with_pool and not limit:
        rep["reference_pool_cross_check"] = reference_pool_volumes(cuts)
    return rep


# --------------------------------------------------------------------------- #
# md
# --------------------------------------------------------------------------- #
def _pct(x):
    return "—" if x is None else f"{100.0 * float(x):.1f}%"


def md(rep: dict) -> str:
    L: list = []
    A = L.append
    a, c, p = rep["audit"], rep["crossover"], rep["reference_pool"]
    A(f"# Sheet F — base-rate audit and the label/score crossover\n")
    A(f"Generated {rep['generated']} · `{rep['command']}`\n")
    if rep["incomplete"]:
        A("> **INCOMPLETE — bounded run (`--limit`). Not a basis for moving a constant.**\n")
    A(f"\nPopulation: **{p['n']} rows** over {p['n_locations']} locations — {p['what']}; "
      f"basis `{p['basis']}`.\n")
    A(f"\n> {a['ceiling']}\n")

    A("\n## Tier mix, and the two bounds on it\n")
    A("| slice | elicitation | n | tiers | ≥2 | ≥3 |")
    A("|---|---|--:|---|--:|--:|")
    for key, name in (("human", "F human"), ("v3_prefill", "F v3-prefill"),
                      ("sheet_e_blind", "E blind")):
        b = a[key]
        A(f"| {name} | {b.get('elicitation', b.get('what', '—'))} | {b['n']} | "
          f"{b['tiers']} | {b['n_ge2']} ({_pct(b['rate_ge2'])}) | "
          f"{b['n_ge3']} ({_pct(b['rate_ge3'])}) |")

    cr = a["correction"]
    A(f"\n## Correction rate against the v3 prefill\n")
    A(f"Exact tier agreement **{cr['exact_tier_agreement']}/{cr['n']} = "
      f"{_pct(cr['exact_tier_agreement_rate'])}**. Flips across ≥2: "
      f"**{cr['flips_ge2']['served_below_human_at_or_above']} up** (served <2, human ≥2), "
      f"**{cr['flips_ge2']['served_at_or_above_human_below']} down**; across ≥3: "
      f"**{cr['flips_ge3']['served_below_human_at_or_above']} up**, "
      f"**{cr['flips_ge3']['served_at_or_above_human_below']} down**.\n")
    A("\n| served ↓ / human → | 1 | 2 | 3 |")
    A("|---|--:|--:|--:|")
    for s, row in cr["confusion_served_by_human"].items():
        A(f"| **{s}** | {row['1']} | {row['2']} | {row['3']} |")

    A(f"\n## The crossover\n")
    A(f"Isotonic fit of `{c['target']}` against `{CROSSOVER_SIGNAL}`, {c['n']} rows, base rate "
      f"{_pct(c['base_rate'])}. Fitted probability reaches 0.5 between "
      f"**{c['last_row_below']:.5f}** and **{c['first_row_at_or_above']:.5f}** → constant "
      f"**{c['constant']}**, realized volume **{c['realized_volume']}/{c['n']}**.\n")
    if c["tie_block_at_exactly_half"]:
        sa = c["strictly_above"]
        A(f"\nThe crossing lands on a TIE BLOCK fitted at exactly 0.5, so the reading is "
          f"ambiguous by one block: `> 0.5` instead of `>= 0.5` gives "
          f"**{sa['constant']}** at {sa['realized_volume']}/{c['n']}. Adopted convention: "
          f"{c['convention']}.\n")
    A("\n| fitted P(label≥2) | rows | positives | score range |")
    A("|--:|--:|--:|---|")
    for b in c["blocks"]:
        A(f"| {b['fit']:.4f} | {b['n']} | {b['positives']} | "
          f"{b['lo']:.5f} – {b['hi']:.5f} |")

    A("\n## The cuts\n")
    A("| cut | owner | old | **new** | fires (F) | pass rate | precision≥3 old → new | "
      "precision≥2 old → new |")
    A("|---|---|--:|--:|--:|--:|---|---|")
    for cut in rep["cuts"]:
        o3, i3 = cut["outgoing"]["precision_ge3"], cut["incoming"]["precision_ge3"]
        o2, i2 = cut["outgoing_ge2"]["precision_ge3"], cut["incoming_ge2"]["precision_ge3"]
        A(f"| `{cut['name']}` | `{cut['owner'].splitlines()[0]}` | {cut['outgoing_value']:g} | "
          f"**{cut['incoming_value']:g}** | {cut['realized_volume']}/{cut['n']} | "
          f"{_pct(cut['realized_volume'] / cut['n'])} | {_pct(o3)} → {_pct(i3)} | "
          f"{_pct(o2)} → {_pct(i2)} |")
    A(f"\n{rep['cuts'][0]['matched_volume_note']}\n")

    rp = rep.get("reference_pool_cross_check")
    if rp:
        A(f"\n## The same cuts on the flip's reference pool ({rp['n']} rows)\n")
        A(f"{rp['what']}; base rate ≥3 {_pct(rp['base_rate_ge3'])}, ≥2 "
          f"{_pct(rp['base_rate_ge2'])}.\n")
        A("\n| cut | old | new | fires old → new | precision≥3 old → new | recall≥3 old → new |")
        A("|---|--:|--:|---|---|---|")
        for name, b in rp["cuts"].items():
            A(f"| `{name}` | {b['outgoing_value']:g} | {b['incoming_value']:g} | "
              f"{b['outgoing']['n_selected']} → **{b['incoming']['n_selected']}** | "
              f"{_pct(b['outgoing']['precision_ge3'])} → {_pct(b['incoming']['precision_ge3'])} | "
              f"{_pct(b['outgoing']['recall_ge3'])} → {_pct(b['incoming']['recall_ge3'])} |")
        ji = rp["junk_floor_inversion"]
        A(f"\n**JUNK_FLOOR inversion.** {ji['note']} {ji['fires_at_junk_floor']}/{rp['n']} clear "
          f"{ji['junk_floor']}.\n")

    for key, label, base in (("ladder_ge3", "≥3", p["base_rate_ge3"]),
                             ("ladder_ge2", "≥2 (the crossover's boundary)", p["base_rate_ge2"])):
        A(f"\n## Ladder on sheet F — {label}, base rate {_pct(base)}\n")
        A("| threshold | fires | pass rate | TP | precision | 95% CI | recall | mark |")
        A("|--:|--:|--:|--:|--:|--:|--:|---|")
        for r in rep[key]:
            ci = ("—" if r["precision"] is None
                  else f"{_pct(r['precision_lo'])}–{_pct(r['precision_hi'])}")
            A(f"| {r['threshold']:.4f} | {r['fires']} | {_pct(r['pass_rate'])} | {r['tp']} | "
              f"{_pct(r['precision'])} | {ci} | {_pct(r['recall'])} | "
              f"{' '.join(r.get('marks', []))} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", choices=sorted(SPECS), default="sheet_f")
    ap.add_argument("--limit", type=int, default=None, help="bounded end-to-end smoke -> scratch/")
    ap.add_argument("--no-pool", action="store_true",
                    help="skip the GPU cross-check on the flip's reference pool (torch-free run)")
    ap.add_argument("--ndigits", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    spec = SPECS[args.sheet]
    rep = run(spec, limit=args.limit, with_pool=not args.no_pool, ndigits=args.ndigits)
    text, face = json.dumps(rep, indent=2) + "\n", md(rep)
    # The class is declared at the WRITE SITE (the sheet-D/E lesson): a bounded run is scratch,
    # an unbounded one durable, and `durable()` raises rather than writing into a gitignored dir.
    if args.out:
        out = Path(args.out) / f"{spec.stem}.json"
        out_md = Path(args.out) / f"{spec.stem}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
    elif args.limit:
        out = P.scratch("baserate_audit", f"{spec.stem}.json")
        out_md = P.scratch("baserate_audit", f"{spec.stem}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = P.durable(f"{spec.out_rel}/{spec.stem}.json", mkparents=True)
        out_md = P.durable(f"{spec.out_rel}/{spec.stem}.md", mkparents=True)
    out.write_text(text, encoding="utf-8")
    out_md.write_text(face, encoding="utf-8")
    print(face)
    if rep["cuts_drifted_from_the_live_owners"]:
        print(f"[audit] NOTE: {rep['cuts_drifted_from_the_live_owners']} differ from the live "
              f"owners — this record proposes them; a constant moves when a human edits its "
              f"owner.")
    print(f"wrote {out} + {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
