r"""volume_match.py — THE volume-matched restatement of a head-scale cut, for BOTH stage-2 heads.

`classifier_retrain_protocol.md` §5a says what a head flip owes every cut calibrated against
the outgoing head: **re-score, then volume-match** — "recompute the score that keeps the same
FRACTION of a fixed reference pool under the new head, and move the constant there". Until this
module that procedure existed only as prose plus two per-verdict `volume_matched` blocks
(`wallpaper_v4b_reads.volume_matched`, `mining_v3_reads`'s twin), each computed for its own
report and neither reusable as the flip's arithmetic. This is the arithmetic, once, with the
two heads as DATA (`SPECS`) rather than as two modules.

WHY A MIDPOINT AND NOT THE k-TH SCORE. A report's `cut_at` is the k-th largest score — the
smallest one that passes. Re-used as a threshold it is off by one under a strict `>` site and
exact under `>=`, and the two stage-2 sites disagree: `emit_v1` gates on `p_ge3 > threshold`
and `MiningScorer.gate` on `p_ge3 >= threshold`. So the new cut is placed at the MIDPOINT
between the k-th and (k+1)-th largest scores, where both comparisons realize the same k, and
the realized volume is then RE-COUNTED under the rounded constant that is actually written
(`realized_volume`) rather than assumed. A rounding that moves the volume is visible instead
of silent.

WHAT A REFERENCE POOL MAY BE. The pool must be the population the cut acts on, scored under
BOTH heads through one harness in one pass — never one head's frozen `eval_scores.jsonl`
against the other's live pass, which is a comparison across two rendering events (the reason
`wallpaper_v4b_reads` re-scores rather than reading the committed files). Each spec names its
loader; the loaders are the same ones the two winner-rule reads use, so "the pool the verdict
was read on" and "the pool the cut was restated on" cannot become two populations.

THE PRECISIONS BESIDE EACH CUT ARE NOT THE ARGUMENT FOR IT. Volume-matching keeps VOLUME
invariant on purpose — how much supply `GOOD_FLOOR` keeps, how much waste `JUNK_FLOOR`
removes, how many renders the gate passes. The precision at the matched volume is reported
because it is the thing that changed, and it is the head's verdict, not the cut's.

MOVES NOTHING. This module writes a record; a constant moves when a human edits its owner.

  uv run python tools/scoring/volume_match.py wallpaper --incoming data/wallpaper_head/v4b/seed_1/model_best.pt
  uv run python tools/scoring/volume_match.py mining
  uv run python tools/scoring/volume_match.py wallpaper --limit 64      # bounded end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
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

from tools.corpus.q4_combined_readout import wilson            # noqa: E402


# --------------------------------------------------------------------------- #
# The two objects: a cut, and a flip.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cut:
    """One constant to restate. `value` is on the OUTGOING head's scale.

    `strict` is the comparison the SITE performs, not a preference: `emit_v1` gates
    `p_ge3 > gate` and `MiningScorer.gate` returns `p_ge3 >= threshold`. It decides which
    rows the outgoing volume counts, so getting it wrong moves the matched volume by the
    number of rows sitting exactly on the cut."""
    name: str
    owner: str
    value: float
    strict: bool
    site: str           # "pool" | "release"


@dataclass(frozen=True)
class FlipSpec:
    """One head's flip. A frozen instance from the start (CLAUDE.md, "writing a builder for
    one instance"): the wallpaper and mining flips landed on the same day and the second
    would otherwise have needed a refactor to exist."""
    key: str
    head_name: str                  # data/<head_name>/<version>/...
    outgoing_rel: str
    incoming_rel: str
    slice_name: str
    loader: str                     # module-qualified name of the reference-pool loader
    scorer: str                     # "load_head" | "mining_scorer"
    cuts: tuple = field(default_factory=tuple)
    k_tiers: int = 4


WALLPAPER = FlipSpec(
    key="wallpaper",
    head_name="wallpaper_head",
    outgoing_rel="data/wallpaper_head/v3/model_best.pt",
    incoming_rel="data/wallpaper_head/v4b/seed_1/model_best.pt",
    slice_name="the (28) six-batch eval union (train_wallpaper_v4b.split_v4b)",
    loader="tools.scoring.volume_match.wallpaper_pool",
    scorer="load_head",
    k_tiers=4,
    cuts=(
        Cut("wallpaper_release", "tools/wallpaper/wallpaper_pins.GATE_THRESHOLD "
            "(== tools/emission/floors.WALLPAPER_RELEASE)", 0.90, strict=True, site="release"),
        Cut("wallpaper_pool", "tools/emission/floors.WALLPAPER_POOL",
            0.75, strict=False, site="pool"),
    ),
)

MINING = FlipSpec(
    key="mining",
    head_name="render_mode_head",
    outgoing_rel="data/render_mode_head/v1/model_best.pt",
    incoming_rel="data/render_mode_head/v3/model_best.pt",
    slice_name="the (28) deduplicated mining eval side (mining_corpus.load_corpus)",
    loader="tools.scoring.volume_match.mining_pool",
    scorer="mining_scorer",
    k_tiers=3,
    cuts=(
        Cut("mining_release", "tools/mining/mining_pins.MINING_GATE_THRESHOLD "
            "(== tools/emission/floors.MINING_RELEASE)", 0.50, strict=False, site="release"),
        Cut("mining_pool", "tools/emission/floors.MINING_POOL",
            0.25, strict=False, site="pool"),
    ),
)

SPECS = {s.key: s for s in (WALLPAPER, MINING)}


# --------------------------------------------------------------------------- #
# Reference pools — each the SAME loader its head's winner-rule reads used.
# --------------------------------------------------------------------------- #
def wallpaper_pool(limit: int | None = None):
    """`(jpgs, labels, meta)` — the 1,337-row six-batch eval union of the (28) verdict."""
    from classifier.train_wallpaper_v4b import load_union, split_v4b   # noqa: PLC0415
    prior, sheet_a = load_union()
    _train, ev, meta = split_v4b(prior, sheet_a)
    ev = sorted(ev, key=lambda r: r.image_id)
    if limit:
        ev = ev[:limit]
    return ([r.jpg for r in ev], np.array([int(r.label) for r in ev]),
            {"n": len(ev), "n_locations": len({r.loc for r in ev}),
             "by_batch": meta["eval_by_batch"]})


def mining_pool(limit: int | None = None):
    """`(jpgs, labels, meta)` — the 827-row deduplicated mining eval side of the (28) verdict."""
    from tools.mining.mining_corpus import BATCH_TAG, load_corpus      # noqa: PLC0415
    from collections import Counter                                    # noqa: PLC0415
    pool = load_corpus()
    ev = sorted(pool.eval_rows, key=lambda r: r.image_id)
    if limit:
        ev = ev[:limit]
    _ = BATCH_TAG
    return ([r.jpg for r in ev], np.array([int(r.label) for r in ev]),
            {"n": len(ev), "n_locations": len({r.loc for r in ev}),
             "by_batch": dict(Counter(r.batch for r in ev))})


POOLS = {"wallpaper": wallpaper_pool, "mining": mining_pool}


# --------------------------------------------------------------------------- #
# The arithmetic — pure, no torch, no I/O.
# --------------------------------------------------------------------------- #
def passing_volume(scores: np.ndarray, value: float, *, strict: bool) -> int:
    """How many rows the cut passes, under the SITE's own comparison."""
    s = np.asarray(scores, dtype=float)
    return int((s > value).sum() if strict else (s >= value).sum())


def midpoint_cut(scores: np.ndarray, k: int) -> float:
    """The threshold that admits exactly the top `k` scores under BOTH `>` and `>=`.

    The midpoint between the k-th and (k+1)-th largest. `k == 0` returns just above the max
    and `k == n` just below the min, so a degenerate match is a number rather than a raise —
    the caller sees it in `realized_volume` and in the ratio, which is where a degenerate
    match should be visible."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    if n == 0:
        raise ValueError("volume-match on an empty reference pool")
    if k <= 0:
        return float(np.nextafter(s[0], np.inf))
    if k >= n:
        return float(np.nextafter(s[-1], -np.inf))
    return float((s[k - 1] + s[k]) / 2.0)


def _precision_block(labels: np.ndarray, scores: np.ndarray, thr: float, *, strict: bool,
                     good: int = 3) -> dict:
    sel = (scores > thr) if strict else (scores >= thr)
    k = int(sel.sum())
    tp = int((labels[sel] >= good).sum()) if k else 0
    pos = int((labels >= good).sum())
    p, lo, hi = wilson(tp, k) if k else (None, None, None)
    return {"n_selected": k, "pass_rate": k / len(labels), "tp": tp,
            "precision_ge3": p, "precision_lo": lo, "precision_hi": hi,
            "recall_ge3": (tp / pos) if pos else None}


SWEEP = tuple(round(x, 3) for x in np.arange(0.0, 1.0, 0.05))


def ladder(labels: np.ndarray, scores: np.ndarray, marks: dict, *, strict: bool,
           good: int = 3) -> list:
    """The whole precision/recall curve on one boundary, at a fixed sweep UNIONED with the
    marked cut values.

    The union is what makes the record usable as a lock: a reader asking "what does the live
    cut buy" must find an EXACT swept row, never a neighbouring bin — a precision quoted
    against a threshold nobody runs is the failure `lock_mining_gate._row_at` refuses on. So
    the cuts this record is the authority for are swept at their own values by construction."""
    at = {}
    for name, v in marks.items():
        at.setdefault(round(float(v), 6), []).append(name)
    rows = []
    for thr in sorted(set(SWEEP) | set(at)):
        b = _precision_block(labels, scores, thr, strict=strict, good=good)
        rows.append({"threshold": float(thr), "fires": b["n_selected"],
                     "pass_rate": b["pass_rate"], "tp": b["tp"],
                     "precision": b["precision_ge3"], "precision_lo": b["precision_lo"],
                     "precision_hi": b["precision_hi"], "recall": b["recall_ge3"],
                     "marks": at.get(round(float(thr), 6), [])})
    return rows


def match_cut(cut: Cut, labels: np.ndarray, base: np.ndarray, cand: np.ndarray,
              *, ndigits: int = 4) -> dict:
    """One cut, restated. Everything is computed; nothing is quoted.

    `realized_volume` is RE-COUNTED under the rounded constant that will actually be written,
    because rounding a midpoint can cross a tie and a volume-match whose realized volume is
    not its matched volume is not a volume-match."""
    base = np.asarray(base, dtype=float)
    cand = np.asarray(cand, dtype=float)
    n = len(labels)
    k = passing_volume(base, cut.value, strict=cut.strict)
    raw = midpoint_cut(cand, k)
    new = round(raw, ndigits)
    realized = passing_volume(cand, new, strict=cut.strict)
    return {
        "name": cut.name, "owner": cut.owner, "site": cut.site,
        "comparison": ">" if cut.strict else ">=",
        "outgoing_value": cut.value, "incoming_value": new,
        "incoming_value_unrounded": raw,
        "n": n, "matched_volume": k, "matched_rate": k / n if n else None,
        "realized_volume": realized,
        "volume_preserved": realized == k,
        "outgoing": _precision_block(labels, base, cut.value, strict=cut.strict),
        "incoming": _precision_block(labels, cand, new, strict=cut.strict),
    }


# --------------------------------------------------------------------------- #
# Scoring — each head through the harness that actually gates with it.
# --------------------------------------------------------------------------- #
def score_slice(spec: FlipSpec, ckpt: Path, jpgs) -> dict:
    """`{p_ge2, p_ge3}` for every crop, under `ckpt`, through the harness that gates with it."""
    if spec.scorer == "load_head":
        import torch                                                   # noqa: PLC0415
        from tools.wallpaper.report_v4_eval import load_head           # noqa: PLC0415
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        score, _cfg = load_head(ckpt, dev)
        _cond, marg, _ssum = score(list(jpgs))
        return {"p_ge2": marg[:, 0], "p_ge3": marg[:, 1]}
    if spec.scorer == "mining_scorer":
        from tools.mining.mining_gate import MiningScorer              # noqa: PLC0415
        sc = MiningScorer(model_path=str(ckpt))
        ms = sc.score_paths(list(jpgs))
        return {"p_ge2": np.array([m.p_ge2 for m in ms], dtype=float),
                "p_ge3": np.array([m.p_ge3 for m in ms], dtype=float)}
    raise KeyError(f"no scorer named {spec.scorer!r}")


def run(spec: FlipSpec, *, incoming: str | None = None, limit: int | None = None,
        ndigits: int = 4) -> dict:
    inc_rel = incoming or spec.incoming_rel
    jpgs, labels, meta = POOLS[spec.key](limit=limit)
    t0 = time.time()
    base = score_slice(spec, ROOT / spec.outgoing_rel, jpgs)
    cand = score_slice(spec, ROOT / inc_rel, jpgs)
    matched = [match_cut(c, labels, base["p_ge3"], cand["p_ge3"], ndigits=ndigits)
               for c in spec.cuts]
    marks = {c["name"]: c["incoming_value"] for c in matched}
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": f"uv run python tools/scoring/volume_match.py {spec.key}"
                   + (f" --incoming {inc_rel}" if incoming else "")
                   + (f" --limit {limit}" if limit else ""),
        "procedure": "classifier_retrain_protocol.md §5a — re-score, then VOLUME-MATCH. The "
                     "new constant is the score that admits the same NUMBER of reference-pool "
                     "rows the outgoing constant admitted, placed at the midpoint between the "
                     "k-th and (k+1)-th largest so `>` and `>=` realize the same k.",
        "head": {"name": spec.head_name, "outgoing": spec.outgoing_rel, "incoming": inc_rel},
        "reference_pool": {"what": spec.slice_name, "loader": spec.loader,
                           "scorer": spec.scorer, **meta,
                           "tiers": {str(t): int((labels == t).sum())
                                     for t in range(1, spec.k_tiers + 1)},
                           "base_rate_ge3": float((labels >= 3).mean()),
                           "base_rate_ge2": float((labels >= 2).mean())},
        "scored_in_s": round(time.time() - t0, 1),
        "cuts": matched,
        # BOTH boundary curves under the INCOMING head, so "what would 0.40 have bought"
        # stays answerable off this record instead of needing a re-score whose crops may be
        # gone — the property `mining_gate_lock.json` keeps its two ladders for. `ladder_ge3`
        # cuts `p_ge3` against label>=3, `ladder_ge2` cuts `p_ge2` against label>=2: each
        # boundary is read on its OWN marginal, never on the other's.
        "ladder_ge3": ladder(labels, cand["p_ge3"], marks, strict=False, good=3),
        "ladder_ge2": ladder(labels, cand["p_ge2"], {}, strict=False, good=2),
        "score_scale": {
            "why": "CORN marginals are calibrated to the training prior, so no raw-threshold "
                   "comparison across the two heads appears in this record.",
            "quantiles": {f"q{int(q*100)}": {"outgoing": float(np.quantile(base["p_ge3"], q)),
                                             "incoming": float(np.quantile(cand["p_ge3"], q))}
                          for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)},
        },
        "incomplete": bool(limit),
    }


def version_dir(spec: FlipSpec, ckpt_rel: str) -> str:
    """`data/<head_name>/<version>` for a checkpoint path — the record lands beside the head
    it describes, whether or not the pin points at a per-seed subdirectory.

    Derived the same way `mining_pins.head_version` derives its token, and for the same
    reason: `parent.parent` is right for `v4b/seed_1/model_best.pt` and silently wrong for
    `v3/model_best.pt`, which is a record written one directory above the head it is about."""
    parts = Path(ckpt_rel).as_posix().split("/")
    try:
        i = parts.index(spec.head_name)
    except ValueError:
        raise ValueError(f"{ckpt_rel!r} is not under data/{spec.head_name}/<version>/") from None
    return "/".join(parts[:i + 2])


def md(rep: dict) -> str:
    L, A = [], None
    A = L.append
    h = rep["head"]
    A(f"# Volume-matched restatement — {h['name']}: "
      f"{Path(h['outgoing']).parent.name} -> {Path(h['incoming']).parent.name}\n")
    A(f"Generated {rep['generated']} · `{rep['command']}`\n")
    if rep["incomplete"]:
        A("> **INCOMPLETE — bounded run (`--limit`). Not a basis for moving a constant.**\n")
    p = rep["reference_pool"]
    A(f"Reference pool: **{p['n']} rows** over {p['n_locations']} locations — {p['what']}; "
      f"tiers {p['tiers']}; base rate ≥3 {p['base_rate_ge3']:.3f}.\n")
    A("\n| cut | owner | old | **new** | volume | rate | precision≥3 old → new |")
    A("|---|---|---:|---:|---:|---:|---|")
    for c in rep["cuts"]:
        po, pi = c["outgoing"]["precision_ge3"], c["incoming"]["precision_ge3"]
        A(f"| `{c['name']}` | `{c['owner'].splitlines()[0]}` | {c['outgoing_value']:g} | "
          f"**{c['incoming_value']:g}** | {c['matched_volume']}/{c['n']} | "
          f"{c['matched_rate']:.3f} | "
          f"{'—' if po is None else f'{po:.3f}'} → {'—' if pi is None else f'{pi:.3f}'} |")
    bad = [c["name"] for c in rep["cuts"] if not c["volume_preserved"]]
    A(f"\nRealized volume equals matched volume for every cut."
      if not bad else f"\n**Rounding moved the volume on: {bad}** — widen `--ndigits`.")
    A("\nVolume-matching keeps VOLUME invariant on purpose; the precision column is what the "
      "head changed, not what the cut bought.\n")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head", choices=sorted(SPECS))
    ap.add_argument("--incoming", default=None, help="override the incoming checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="bounded end-to-end smoke")
    ap.add_argument("--ndigits", type=int, default=4)
    ap.add_argument("--out", default=None, help="output dir (default: the incoming head's dir)")
    args = ap.parse_args(argv)

    spec = SPECS[args.head]
    rep = run(spec, incoming=args.incoming, limit=args.limit, ndigits=args.ndigits)
    out = Path(args.out) if args.out else (
        ROOT / "scratch" / "volume_match" / spec.key if args.limit
        else ROOT / version_dir(spec, rep["head"]["incoming"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"volume_match_{spec.key}.json").write_text(json.dumps(rep, indent=2) + "\n",
                                                       encoding="utf-8")
    (out / f"volume_match_{spec.key}.md").write_text(md(rep), encoding="utf-8")
    print(md(rep))
    print(f"wrote {out}/volume_match_{spec.key}.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
