#!/usr/bin/env python
r"""sitting_cutter.py — cut a continuous harvest's record into ONE labelling sitting.

WHAT A SITTING IS. Harvest v1 produced three registered batches and three exports off one
run, and Matt then sat through 870 tiles of which a large minority were things nobody should
ever have been shown: 126 all-label-1 unscreened native multibrots in one cluster, 52
julia:mandelbrot rows that were 17 atoms x 3 rungs of the same look, and an unmeasured number
of >30%-interior frames the rule already says are class 1 by construction. A sitting is the
fix: ONE cut, ONE manifest, ONE export, capped at `MAX_ROWS`, with everything the record
already knows to be worthless removed BEFORE a human sees it.

THE THREE FILTER STAGES ARE NON-OPTIONAL, AND THAT IS THE DESIGN
----------------------------------------------------------------
They are entries in `STAGES`, walked unconditionally by `cut_sitting`. There is no flag to
skip one. That is deliberate and it is not tidiness: every one of them exists because its
absence cost a real sitting real keystrokes, and a filter with an off switch is a filter that
will be off on the run that needed it (`verification_practice.md` §2 — a gate that degrades
to silence cannot protect against the removal of its own input). Each is proved red by
injection in `test_sitting_cutter.py`.

  (a) INTERIOR > 0.30 -> auto-labelled `interior_gt30_v1`, NEVER PRESENTED.
      Matt's rule, dictated 2026-08-01 and firm: a frame more than 30% black is class 1 for
      wallpaper emission, no gray zone. `apply_interior_rule.py` already applies it to the
      label store AFTER a batch is built and seeds the score into the served manifest so the
      rig skips the row. This is the stronger form the sitting is owed: the row never enters
      the served manifest at all. Same rule id, same threshold, same strict `>`, same measure,
      imported from that module rather than restated.

  (b) PRESENTATION-LEVEL MORPH-DEDUP at cos 0.974 — one row per look.
      NEVER A DISCOVERY GATE, and the distinction is the whole point. The discovery record
      keeps every candidate; what is thinned is what gets SHOWN. A near-duplicate is not
      evidence of anything after the first one, and 870 labelled rows collapsed to 367 looks
      (2.37 labels per look) — the sitting's cost is denominated in looks, so the dedup is
      what makes the cap mean something. Best-first, so a look is represented by its
      highest-ranked member.

  (c) PER-PARTITION MACHINE-1 AUTO-DISCARD.
      ON for native multibrot and phoenix, OFF for julia:mandelbrot, because the measurement
      is partition-dependent and the pooled number is not a decision: P(Matt=1 | v10 decoded
      1) is 94-100% in multibrot3/4/5 and 72.0% in phoenix (P(>=3 | decoded 1) = 0/82 there),
      but 30.9% in julia:mandelbrot, where 16.5% of machine-1s are >=3. The per-partition
      table is `supply_routing.MACHINE_1_DISCARD`, imported, and every partition with no
      measurement of its own fails CLOSED to KEEP — spending labels is recoverable, throwing
      away one good picture in six is not.

ORDER IS COST-DESCENDING IN THE OTHER DIRECTION. (a) and (c) are free reads off columns the
record already carries; (b) needs a render and a CLIP pass per surviving row. So the two free
stages run first and the expensive one sees the smallest population. Reversing them would be
correct and would cost a morph field for every row the other two were about to delete.

  uv run python tools/atlas/sitting_cutter.py dry-run --run-dir data/discovery/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apply_interior_rule as air               # noqa: E402  (the rule id + threshold)
import supply_routing as srt                    # noqa: E402  (the per-partition discard table)

MAX_ROWS = 1000              # one sitting


class BudgetExhausted(Exception):
    """A BOUNDED morph pass hit its limit. Raised by an embedder built with a `limit`, and
    counted by `stage_morph_dedup` as `budget_not_reached` — never as `unembeddable`, which
    is a claim about the row rather than about the pass."""


NEAR_DUP_COS = srt.NEAR_DUP_COS
INTERIOR_RULE_ID = air.RULE_ID
INTERIOR_THRESHOLD = air.THRESHOLD


# =========================================================================== #
# the stages. Each takes (rows, ctx) and returns (kept, removed, report).
# =========================================================================== #
def stage_interior(rows, ctx):
    """(a) Matt's >0.30-interior rule, as an AUTO-LABEL that is never presented.

    A row with NO measure is KEPT and counted apart — an absent measure is not a high one,
    which is `apply_interior_rule.fires`'s own rule and the sourcing gate's. Strict `>`, so a
    frame at exactly 0.30 is shown; the boundary side is invisible in a count, which is why
    it is asserted rather than described."""
    kept, removed, unmeasured = [], [], 0
    for r in rows:
        v = r.get("int_frac")
        if v is None:
            unmeasured += 1
            kept.append(r)
        elif float(v) > INTERIOR_THRESHOLD:
            r = dict(r, auto_label=dict(score=air.RULE_SCORE, labeler=air.LABELER,
                                        rule_id=INTERIOR_RULE_ID, measure="int_frac",
                                        value=float(v), threshold=INTERIOR_THRESHOLD,
                                        comparison="strict >"))
            removed.append(r)
        else:
            kept.append(r)
    return kept, removed, dict(stage="interior_gt30", removed=len(removed),
                               unmeasured_kept=unmeasured, rule_id=INTERIOR_RULE_ID,
                               threshold=INTERIOR_THRESHOLD, comparison="strict >",
                               disposition="auto-labelled class 1, NEVER presented")


def stage_machine_1(rows, ctx):
    """(c) Per-partition machine-1 auto-discard.

    Only a CANONICAL decode counts. A row carrying just a cheap score (`rank_tier=1`) has no
    machine-1 verdict to act on — the cheap score comes off a 384x216 ss1 render and the
    measured P(Matt=1 | decoded 1) rates were all taken against the 640x360 ss2 canonical
    decode. Treating the two as one number is the cap/geometry error, so a tier-1 row is
    never discarded here whatever its partition's flag says."""
    table = ctx.get("machine_1_discard") or srt.MACHINE_1_DISCARD
    kept, removed = [], []
    no_verdict = Counter()
    for r in rows:
        part = r.get("partition")
        dec = r.get("canon_decoded")
        if dec is None or int(r.get("rank_tier") or 0) < 2:
            no_verdict[part] += 1
            kept.append(r)
        elif int(dec) == 1 and table.get(part, False):
            removed.append(dict(r, discard_reason=f"machine_1:{part}"))
        else:
            kept.append(r)
    return kept, removed, dict(stage="machine_1_discard", removed=len(removed),
                               no_canonical_verdict_kept=dict(no_verdict),
                               table={p: bool(v) for p, v in sorted(table.items())},
                               by_partition=dict(Counter(r["partition"] for r in removed)),
                               disposition="discarded from the SITTING; the discovery record "
                                           "keeps them")


def stage_morph_dedup(rows, ctx):
    """(b) Presentation-level morph-dedup: one row per look at cos 0.974, best-first.

    NOT A DISCOVERY GATE. Nothing is removed from the run's record; what is thinned is the
    served page. Greedy leader-radius against the accepted set, in the order the caller hands
    them over — so the caller's ranking is the policy and a look is represented by its
    highest-ranked member, exactly as `supply_routing.thin_by_cspacing` works one layer up.

    `ctx["embed"]` maps a row to an L2-normalized vector (the library morph recipe: 640x360
    ss2 smooth field -> robust-z tanh gray -> CLIP vit_base_patch16_clip_224.openai — the same
    recipe emission clusters at 0.974, so this threshold means what it means everywhere else).
    A row the embedder cannot reach is KEPT and counted, because "we could not measure this"
    is not "this is a duplicate".

    MEASURED AT FULL SITTING SCALE, 2026-08-03 (`sitting_cutter.py dry-run --run-dir
    data/discovery/q4_long_harvest_20260803 --embed-limit 1000`): **15 m 36 s for 1,000
    embeds**, 587 removed, **413 looks kept** (2.42 rows per look — the harvest-v1 sitting
    measured 2.37, so the knee holds at 2.5x the population). Nothing degraded: 0
    unembeddable, 0 exceptions, the accounting closed.

    Two numbers that change how a live sitting is sized:
      * **0.93 s/row, not 0.26.** A 25-row calibration off the head of the queue said 0.26;
        the queue is tier-sorted and the expensive rows are later, so the prefix sample
        underestimated by 3.6x. `CLAUDE.md`'s run-order rule, hit again.
      * **the cap does not bound this stage.** Dedup runs BEFORE `draw_balanced`, and must —
        the cap is denominated in looks. So a live cut embeds the whole post-(a)-post-(c)
        population, 7,244 rows here, not the 1,000 that reach the page: **~1.9 h**, not the
        15 minutes this bounded run took. `--embed-limit` is a dry-run instrument only.
      * the duplicate rate is NOT a population constant — 30.5% at 400 embeds, 58.7% at
        1,000. A leader-radius accumulates leaders, so a cut sized from a small pilot will
        over-estimate how many looks survive."""
    import numpy as np
    embed = ctx.get("embed")
    if embed is None:
        raise ValueError("stage_morph_dedup needs ctx['embed'] — the dedup is NOT optional, "
                         "so a missing embedder is a hard failure and never a silent skip")
    thr = float(ctx.get("near_dup_cos", NEAR_DUP_COS))
    acc: list = []
    kept, removed, unembeddable, not_reached = [], [], 0, 0
    reasons: Counter = Counter()
    for r in rows:
        try:
            e = embed(r)
        except BudgetExhausted:
            # A BOUNDED pass (a dry run) reached its limit. Counted APART from a row the
            # embedder could not reach: "we stopped early" and "this row has no field" are
            # different facts, and a run that reported the first as the second would look
            # like a population property. The rows after the bound pass through untouched.
            not_reached += 1
            kept.append(r)
            continue
        except Exception as exc:                             # noqa: BLE001
            # PER-ROW tolerance, NOT a silent one. One bad row must not kill a cut, but the
            # reason is counted and reported — the first dry-run of this stage embedded ZERO
            # of 7,264 rows and reported it as "unembeddable_kept", which reads as a property
            # of the population and was actually one exception repeated 7,264 times.
            reasons[f"{type(exc).__name__}: {str(exc)[:80]}"] += 1
            e = None
        if e is None:
            unembeddable += 1
            if not reasons:
                reasons["embedder returned None"] += 1
            kept.append(r)
            continue
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        e = e / (float(np.linalg.norm(e)) + 1e-9)
        if acc:
            cos = float(np.max(np.stack(acc) @ e))
            if cos >= thr:
                removed.append(dict(r, dup_cos=round(cos, 4)))
                continue
        acc.append(e)
        kept.append(r)
    return kept, removed, dict(stage="morph_dedup", removed=len(removed),
                               unembeddable_kept=unembeddable, threshold=thr,
                               budget_not_reached=not_reached,
                               embedded=len(acc) + len(removed),
                               unembeddable_reasons=dict(reasons.most_common(5)),
                               recipe="library morph CLIP (640x360 ss2 -> robustz_tanh_k2_v1 "
                                      "-> vit_base_patch16_clip_224.openai)",
                               looks_kept=len(acc),
                               disposition="PRESENTATION only — the discovery record keeps "
                                           "every candidate")


# The pipeline. Walked unconditionally; there is no flag that removes an entry.
STAGES = (stage_interior, stage_machine_1, stage_morph_dedup)


# =========================================================================== #
# the cut
# =========================================================================== #
def draw_balanced(rows, cell_of, n: int):
    """`n` rows, round-robin over cells, best-first inside each cell — the caller's order is
    the within-cell rank. Same floor-then-remainder shape as the v1 batch draw, restated here
    only because the cells differ (partition x tier rather than fate x partition): a sitting
    is one page and a fate-balanced page would spend the cap on rejects."""
    cells = defaultdict(list)
    for r in rows:
        cells[cell_of(r)].append(r)
    keys = sorted(cells, key=str)
    take = {k: 0 for k in keys}
    while sum(take.values()) < n:
        cand = [k for k in keys if take[k] < len(cells[k])]
        if not cand:
            break
        k = min(cand, key=lambda k: (take[k], -len(cells[k])))
        take[k] += 1
    out = []
    for i in range(max(take.values(), default=0)):
        for k in keys:
            if i < take[k]:
                out.append(cells[k][i])
    rep = {str(k): dict(taken=take[k], available=len(cells[k]),
                        drained=take[k] >= len(cells[k])) for k in keys}
    return out, rep


def cell_of(r) -> tuple:
    return (r.get("partition"), int(r.get("rank_tier") or 0))


def cut_sitting(rows, *, max_rows: int = MAX_ROWS, embed=None,
                machine_1_discard=None, near_dup_cos: float = NEAR_DUP_COS) -> dict:
    """Run every stage, then cut to one sitting. Returns the sitting and its full accounting.

    The accounting closes: `n_in == n_sitting + sum(removed per stage) + n_over_cap`. A cut
    that can lose a row without a stage naming it is a cut nobody can audit, which is the
    same identity `steered_frontier._reconcile_batch` enforces per batch.
    """
    ctx = dict(embed=embed, machine_1_discard=machine_1_discard, near_dup_cos=near_dup_cos)
    n_in = len(rows)
    stage_reports, removed_by_stage = [], {}
    cur = list(rows)
    for fn in STAGES:
        cur, removed, rep = fn(cur, ctx)
        stage_reports.append(rep)
        removed_by_stage[rep["stage"]] = removed
    sitting, cells = draw_balanced(cur, cell_of, max_rows)
    over_cap = len(cur) - len(sitting)

    total_removed = sum(len(v) for v in removed_by_stage.values())
    assert n_in == len(sitting) + total_removed + over_cap, (
        f"sitting cut does not balance: {n_in} in != {len(sitting)} sitting + "
        f"{total_removed} removed + {over_cap} over cap")

    return dict(
        sitting=sitting,
        auto_labeled=removed_by_stage["interior_gt30"],
        removed=removed_by_stage,
        report=dict(
            n_in=n_in, n_sitting=len(sitting), n_over_cap=over_cap, max_rows=max_rows,
            stages=stage_reports,
            balances=True,
            by_partition=dict(Counter(r.get("partition") for r in sitting)),
            by_tier=dict(Counter(str(r.get("rank_tier")) for r in sitting)),
            by_fate=dict(Counter(r.get("fate") for r in sitting)),
            triggered=sum(1 for r in sitting if r.get("triggered")),
            cells=cells,
        ))


# =========================================================================== #
# the default embedder — the library morph recipe, one field per row
# =========================================================================== #
def make_embedder(scratch_dir: Path, limit: int | None = None):
    """The real embedder: 640x360 ss2 smooth field -> morph gray -> CLIP. Heavy imports are
    lazy so the stage functions stay unit-testable with hand-built vectors.

    `limit` bounds how many rows are actually embedded; beyond it the embedder raises
    `BudgetExhausted`, which the stage counts as `budget_not_reached` — a fact about the PASS,
    kept separate from `unembeddable_kept`, which is a fact about a ROW. That is for a bounded
    dry-run, and the separate count is what stops it being a silent truncation
    (`CLAUDE.md`: no silent caps — log what was dropped)."""
    from tools.emission import descriptor as D                # noqa: E402
    from tools.wallpaper import library_annotate as la        # noqa: E402
    from tools.curation.colored_clip import load_clip, embed_clip   # noqa: E402
    import numpy as np

    state = dict(model=None, tf=None, n=0)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def embed(row):
        if limit is not None and state["n"] >= limit:
            raise BudgetExhausted(f"morph pass bounded at {limit} rows")
        if state["model"] is None:
            state["model"], state["tf"] = load_clip()
        loc = D.location_of(_ledger_row(row))
        field = la.ensure_field(loc, retain=False, tmp_dir=scratch_dir,
                                cache_root=scratch_dir)
        gray = la.morph_gray_image(field)
        e = embed_clip(state["model"], state["tf"], [gray])[0].astype(np.float32)
        state["n"] += 1
        return e / (float(np.linalg.norm(e)) + 1e-9)

    return embed


def _ledger_row(r) -> dict:
    """A record-and-rank row in the shape `emission.descriptor.location_of` expects.

    That function reads a LEDGER row — `family`, the reframed `outcome_*` viewport, and for a
    julia twin the ASSERTED schema tag that says which of `outcome_*` / `julia_*` is the
    viewport and which is the parameter. It is not a corpus render block, and handing it one
    raises `KeyError: 'family'` — which the first dry-run of this stage did, 7,264 times, and
    reported as an unembeddable population.

    The ADMITTED frame is used when there is one, else the candidate's own: a sheet must show
    what the ledger holds, and `_q4_record` stores both for exactly this reason. The julia tag
    is stamped CAMPAIGN because that is the schema these rows were written in (viewport in
    `outcome_*`, parameter in `julia_c_*`) — asserted rather than inferred, as `location_of`
    requires."""
    import julia_ledger_schema as jls                          # noqa: E402
    cx = r.get("outcome_cx") if r.get("outcome_cx") is not None else r["cx"]
    cy = r.get("outcome_cy") if r.get("outcome_cy") is not None else r["cy"]
    fw = r.get("outcome_fw") if r.get("outcome_fw") is not None else r["fw"]
    out = dict(family=r["partition"], outcome_cx=str(cx), outcome_cy=str(cy),
               outcome_fw=float(fw))
    if r.get("julia_c_re") is not None and r["partition"].startswith("julia:"):
        out["julia_c_re"], out["julia_c_im"] = r["julia_c_re"], r["julia_c_im"]
        out[jls.SCHEMA_KEY] = jls.CAMPAIGN
    for k in ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
              "phoenix_zm1_re", "phoenix_zm1_im"):
        if r.get(k) is not None:
            out[k] = r[k]
    return out


# =========================================================================== #
# CLI — DRY RUN ONLY in v2. Nothing here serves a sitting.
# =========================================================================== #
def load_queue(run_dir: Path) -> list[dict]:
    """The run's record-and-rank store, tier-sorted, first-occurrence-wins on identity —
    imported from the v1 batch builder rather than re-derived, so the sitting and the batch
    draw can never disagree about what the queue IS."""
    import build_q4_harvest_batches as bq
    rows, _rep = bq.build_queue(Path(run_dir))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="report what a sitting WOULD contain; serve nothing")
    d.add_argument("--run-dir", required=True)
    d.add_argument("--max-rows", type=int, default=MAX_ROWS)
    d.add_argument("--embed-limit", type=int, default=None,
                   help="bound the morph pass (dry-run only). The unembedded remainder is "
                        "counted as unembeddable_kept, never silently dropped.")
    d.add_argument("--out", default=None)
    a = ap.parse_args()

    import paths                                              # noqa: E402
    rows = load_queue(a.run_dir)
    scratch = Path(paths.scratch("sitting_cutter", "fields"))
    res = cut_sitting(rows, max_rows=a.max_rows,
                      embed=make_embedder(scratch, a.embed_limit))
    rep = res["report"]
    rep["run_dir"] = str(a.run_dir)
    rep["SERVED"] = False
    rep["note"] = ("DRY RUN: no manifest written, no export, no crop rendered. The v2 prompt "
                   "builds and dry-runs the cutter; it does not serve a sitting.")
    out = Path(a.out) if a.out else Path(paths.scratch("sitting_cutter", "dry_run.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    print(f"\n-> {out}   (NOTHING SERVED)")


if __name__ == "__main__":
    main()
