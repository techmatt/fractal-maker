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
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining",
           ROOT / "tools" / "sourcing"):
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

    THE EMBED IS NOW COMPUTED ONCE, EVER (`tools/wallpaper/morph_embed_cache.py`). The cost
    below is what a COLD population pays; a re-cut of overlapping material pays ~nothing,
    because a location's morph vector is a pure function of (location, morph recipe, embedder)
    and that triple is the store's key. The cold/warm pair measured on the live population is
    recorded at the end of this docstring.

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
    # A pass that can run for an hour must report on itself WHILE it runs, and must reproject
    # from its own recent throughput rather than restate a pre-run estimate (`CLAUDE.md`,
    # "Projecting a long run's wall clock"): the queue is tier-sorted and the expensive rows
    # are late, so a rate taken over the run-to-date average reads optimistic all the way down.
    import time as _time
    progress = ctx.get("progress")
    every = int(ctx.get("progress_every") or 100)
    t_start = _time.time()
    t_window, n_window = t_start, 0
    acc: list = []
    kept, removed, unembeddable, not_reached = [], [], 0, 0
    reasons: Counter = Counter()
    for i, r in enumerate(rows, 1):
        if progress and i % every == 0:
            now = _time.time()
            recent = (now - t_window) / max(i - n_window, 1)
            progress(dict(seen=i, of=len(rows), looks=len(acc), removed=len(removed),
                          elapsed_s=round(now - t_start, 1),
                          recent_s_per_row=round(recent, 3),
                          eta_min=round((len(rows) - i) * recent / 60.0, 1)))
            t_window, n_window = now, i
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
                machine_1_discard=None, near_dup_cos: float = NEAR_DUP_COS,
                progress=None) -> dict:
    """Run every stage, then cut to one sitting. Returns the sitting and its full accounting.

    The accounting closes: `n_in == n_sitting + sum(removed per stage) + n_over_cap`. A cut
    that can lose a row without a stage naming it is a cut nobody can audit, which is the
    same identity `steered_frontier._reconcile_batch` enforces per batch.
    """
    ctx = dict(embed=embed, machine_1_discard=machine_1_discard, near_dup_cos=near_dup_cos,
               progress=progress)
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
def morph_key_of(row) -> str:
    """The persistent morph-embed cache key for one queue row.

    Pure and cheap — a ledger-row reshape and a string join, no torch and no render — which
    is what makes a fully-warm dedup pass cost seconds instead of hours."""
    from tools.emission import descriptor as D                # noqa: E402
    from tools.wallpaper import morph_embed_cache as mec      # noqa: E402
    return mec.morph_key(D.location_of(_ledger_row(row)))


def make_embedder(scratch_dir: Path, limit: int | None = None, cache=None):
    """The real embedder: 640x360 ss2 smooth field -> morph gray -> CLIP. Heavy imports are
    lazy so the stage functions stay unit-testable with hand-built vectors.

    `cache` is a `morph_embed_cache.MorphEmbedCache`; when given, the returned embedder is the
    cached one (hit -> reuse, miss -> compute + append). The cache wraps the OUTSIDE of the
    budget check on purpose: `--embed-limit` bounds embed WORK, and a hit is not work, so a
    bounded dry-run over an already-warm population runs to completion rather than reporting a
    budget it never spent.

    `limit` bounds how many rows are actually embedded; beyond it the embedder raises
    `BudgetExhausted`, which the stage counts as `budget_not_reached` — a fact about the PASS,
    kept separate from `unembeddable_kept`, which is a fact about a ROW. That is for a bounded
    dry-run, and the separate count is what stops it being a silent truncation
    (`CLAUDE.md`: no silent caps — log what was dropped)."""
    from tools.emission import descriptor as D                # noqa: E402
    from tools.wallpaper import library_annotate as la        # noqa: E402
    import numpy as np

    state = dict(model=None, tf=None, n=0)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def embed(row):
        # The GPU stack is imported on the FIRST MISS, not at build time: a fully-warm pass
        # never embeds anything, and paying ~7 s of `import torch` to answer 3,500 dict
        # lookups would be most of that pass's wall clock.
        from tools.curation.colored_clip import load_clip, embed_clip   # noqa: E402
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

    if cache is None:
        return embed
    from tools.wallpaper import morph_embed_cache as mec      # noqa: E402
    return mec.wrap(embed, cache, morph_key_of)


def _ledger_row(r) -> dict:
    """A record-and-rank row in the shape `emission.descriptor.location_of` expects.

    That function reads a LEDGER row — `family`, the reframed `outcome_*` viewport, and for a
    julia twin the ASSERTED schema tag that says which of `outcome_*` / `julia_*` is the
    viewport and which is the parameter. It is not a corpus render block, and handing it one
    raises `KeyError: 'family'` — which the first dry-run of this stage did, 7,264 times, and
    reported as an unembeddable population.

    THE VIEWPORT IS THE CANDIDATE'S OWN FRAME (`cx`/`cy`/`fw`), NOT THE REFRAMED `outcome_*`,
    because that is the frame `build_q4_harvest_batches._render_block` renders the crop at —
    and a PRESENTATION dedup that measures a different picture from the one it is thinning is
    not thinning looks, it is thinning something else. This read the admitted frame when there
    was one; on the harvest-v2 population 70 rows carry `outcome_*` and 49 of them are a
    genuinely different viewport (a reframe halves `fw`), so 1.4% of the population was being
    deduped on a frame nobody would ever see. Derived from the same fields the render block
    reads, so the two cannot drift apart again (`test_sitting_cutter.py` pins it).

    The julia tag is stamped CAMPAIGN because that is the schema these rows were written in
    (viewport in the position fields, parameter in `julia_c_*`) — asserted rather than
    inferred, as `location_of` requires."""
    import julia_ledger_schema as jls                          # noqa: E402
    cx, cy, fw = r["cx"], r["cy"], r["fw"]
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
# serving a sitting: ONE registered batch, then ONE blind sheet over it
#
# The two layers are not redundant and the split is the corpus contract, not tidiness.
# The BATCH is what the corpus owns: `assign_split`-registered, full provenance, an
# `images.jsonl` every consumer globs, and the only place a label may ever land. The SHEET is
# what the labeler is shown: presentation-only, no `images.jsonl` (a sheet that grew one would
# be unioned into training as a second copy of every row), opaque post-shuffle ids, provenance
# DROPPED rather than nulled, and an apportionment-sequenced order. Both halves already exist
# and are already guarded — `build_combined_label_sheet` holds the sheet rules and
# `test_combined_label_sheet.py` the tripwires over them — so the sitting declares an instance
# in that module's `SPECS` and runs it, rather than growing a second copy that can drift.
# =========================================================================== #
SITTING_BATCH = "2026-08-03_v2_sitting_v1"      # pinned to the registrations by a test
GEN_VERSION = "v2_sitting_v1"
PRESENTATION_SEED = 0x5177_0803

# SUPERSAMPLE, AND THIS BATCH DEVIATES FROM THE CORPUS. Every other label-corpus batch renders
# its crops at `build_minibrot_batch.CROP_SS` = 4; this sitting renders at 2, at Matt's call
# (2026-08-03), to buy back roughly 4x the sample count on ~1000 rows x 2 crops. Consequences,
# stated rather than left to be discovered:
#   * `ss` is part of the VERSION-INVARIANT render block, so each crop stays self-describing
#     and rebuildable from its own row — this is a recorded difference, never a silent one.
#   * it is a real batch-level difference from the rest of the corpus. The classifier's deploy
#     transform stretches 1280x720 -> 384x224 bicubic, which absorbs most of an ss4-vs-ss2
#     antialiasing difference, but "most" is not "all" and no one has measured it here.
# The shared constant is NOT edited: the deviation is local to this batch, so a later batch
# that says nothing still gets the corpus default.
SITTING_CROP_SS = 2

# The v2 view screen's columns, as they ride onto a corpus row. Names are the label-seeded /
# supply-crawl block's EXISTING provenance keys, deliberately unrenamed: a screened row here
# and a screened row there are the same measurement on the same frame, so they pool
# (`corpus_common.PROVENANCE_KEYS`, the label-seeded block: "same view frame, same
# composite_v3, same terms — which is the whole reason nothing is renamed").
SCREEN_PROV = {
    "composite": "composite",              # view_screen.composite_v3, the LIVE sort key
    "fit_score": "view_fit",               # view_fit_v1.1's logit — RECORDED, never the order
    "fit_model": "view_fit_model",
    "screen_frame": "screen_frame",
    "screen_policy": "screen_policy",
    "vetoed": "vetoed",
    "size_factor": "size_factor",
    "band_coverage": "band_coverage",
    "band_coverage_q25": "band_coverage_q25",
    "radial_range": "radial_range",
    "radial_rings": "radial_rings",
    "interior_fraction": "interior_fraction",
    "op": "op", "k": "k", "degree": "degree", "period": "period",
    "log10_abs_A": "log10_abs_A", "window_scale": "window_scale",
    "parent_depth": "parent_depth", "atom_key": "atom_key", "atom_id": "atom_id",
}
VIEW_FIT_MODEL = "view_fit_v1.1"

# The bar-readability slice: a served row is readable for the pre-registered +0.1181 delta-AP
# margin iff it carries BOTH scores. Checked on the BUILT provenance, not on the queue, because
# what matters is what survived the cut and the apportionment.
def is_bar_readable(prov: dict) -> bool:
    return (prov.get("fit_model") == VIEW_FIT_MODEL
            and prov.get("fit_score") is not None
            and prov.get("composite") is not None)


def load_queue(run_dir: Path) -> list[dict]:
    """The run's record-and-rank store, tier-sorted, first-occurrence-wins on identity —
    imported from the v1 batch builder rather than re-derived, so the sitting and the batch
    draw can never disagree about what the queue IS."""
    import build_q4_harvest_batches as bq
    rows, _rep = bq.build_queue(Path(run_dir))
    return rows


def _provenance(r: dict, cc, batch_id: str) -> dict:
    """One served row's full selection trail. The classifier never sees any of it."""
    fields = {k: r.get(k) for k in (
        "fate", "rank_tier", "rank_score", "queue_rank", "cheap_eord", "cheap_pgood",
        "canon_eord", "canon_pgood", "canon_decoded", "reframe_decoded", "triggered",
        "mix_source", "int_frac", "occ", "tau_h", "tau_rec", "t_good", "scorer_version",
        "depth", "branch") if r.get(k) is not None}
    man = r.get("maneuver")
    if isinstance(man, dict):
        for prov_key, man_key in SCREEN_PROV.items():
            v = man.get(man_key)
            if v is not None:
                fields[prov_key] = v
    return cc.provenance_block(GEN_VERSION, batch_id,
                               family=r.get("partition"),
                               selection_role="v2_sitting",
                               stratum=str(cell_of(r)), **fields)


def stage_draw(args) -> int:
    """Cut the run's queue into ONE sitting and write it as the registered batch.

    Nothing is rendered here and no sheet is built: the cut has to be readable — and its
    bar-readability slice reported — BEFORE hours of rendering are committed to it."""
    import time
    import paths
    import corpus_common as cc
    import build_q4_harvest_batches as bq
    import build_minibrot_batch as BMB
    import numpy as np
    from tools.v7 import build_manifest as bm
    from tools.wallpaper import morph_embed_cache as mec

    cc.set_below_normal_priority()
    split, biased, source = bm.assign_split({"batch": SITTING_BATCH, "ft": "mandelbrot"})
    if source == "unregistered":
        raise SystemExit(
            f"{SITTING_BATCH} is NOT registered in tools/v7/build_manifest.assign_split. "
            f"Register it BEFORE building — the fail-closed default lands it train-side "
            f"silently, which records 'nobody thought about this batch'.")
    contra = bm.registration_contradictions([{"batch": SITTING_BATCH, "biased": biased}])
    if contra:
        raise SystemExit(f"registration contradiction for {SITTING_BATCH}: {contra}")

    rows = load_queue(args.run_dir)
    cache = mec.MorphEmbedCache().open()
    t0 = time.time()
    res = cut_sitting(rows, max_rows=args.max_rows,
                      embed=make_embedder(Path(paths.scratch("sitting_cutter", "fields")),
                                          None, cache),
                      progress=lambda d: print(json.dumps(d), flush=True))
    cut_wall = time.time() - t0
    cache_rep = cache.report()
    cache.close()

    sitting = res["sitting"]
    # Opaque ids assigned POST-shuffle over the drawn set; the hash makes an id a stable
    # function of the row, so a rebuild reproduces it.
    order = list(range(len(sitting)))
    np.random.default_rng(PRESENTATION_SEED ^ BMB._stable_seed(SITTING_BATCH)).shuffle(order)
    for slot, oi in enumerate(order):
        h = BMB._stable_seed(json.dumps([sitting[oi].get("cx"), sitting[oi].get("cy"),
                                         sitting[oi].get("fw"), sitting[oi].get("julia_c_re"),
                                         sitting[oi].get("phoenix_c_re")], default=str))
        sitting[oi]["image_id"] = f"vs{slot:04d}_{h:08x}"
    sitting.sort(key=lambda r: r["image_id"])

    bq._PHOENIX_POOL_CACHE.update(bq._phoenix_points())
    names = BMB._palette_names()
    full = []
    for r in sitting:
        r["_palette"] = names[BMB._stable_seed(r["image_id"]) % len(names)]
        render = bq._render_block(r)
        render["ss"] = SITTING_CROP_SS      # the recorded deviation; see SITTING_CROP_SS
        full.append(cc.make_row(r["image_id"], render,
                                _provenance(r, cc, SITTING_BATCH), cc.label_block()))

    readable = [r for r in full if is_bar_readable(r["provenance"])]
    bdir = Path(cc.batch_dir(SITTING_BATCH))
    bdir.mkdir(parents=True, exist_ok=True)

    cut = res["report"]
    bj = dict(
        schema_version=1, batch_id=SITTING_BATCH, generator_version=GEN_VERSION,
        created=None, labeler=None,
        presentation_seed=PRESENTATION_SEED,
        vivid_companion=BMB.VIVID_PALETTE,
        served_manifest=None,
        served_via=("a PRESENTATION SHEET, not this directory: "
                    "build_combined_label_sheet.py --spec v2_sitting. This batch holds the "
                    "rows and the provenance; the sheet holds the blind order the labeler "
                    "sees, and the export routes back here."),
        queued_for_labeling=False,
        purpose=("The harvest-v2 proving run's ONE labelling sitting. TRAIN-side and BIASED "
                 "more than once: the cheap CORN ordinal decided which candidates earned a "
                 "canonical confirmation, the rank is built from those scores, and part of "
                 "the supply was itself selected on view_screen.composite_v3. No rate "
                 "measured on this batch is a base rate."),
        counts=dict(total=len(full),
                    by_partition=cut["by_partition"], by_tier=cut["by_tier"],
                    bar_readable=len(readable)),
        registration=dict(assign_split=[split, biased, source],
                          registered_explicitly=(source != "unregistered"),
                          NOTE="registered in tools/v7/build_manifest BEFORE the cut"),
        render_defaults=dict(width=bq.CROP_W, height=bq.CROP_H, ss=SITTING_CROP_SS,
                             ss_deviates_from_corpus_default=dict(
                                 corpus_default=bq.CROP_SS, this_batch=SITTING_CROP_SS,
                                 why="Matt's call 2026-08-03 — ~4x fewer samples over "
                                     "~1000 rows x 2 crops; see sitting_cutter."
                                     "SITTING_CROP_SS for what it costs"),
                             filter=bq.CROP_FILTER, interior_mode=bq.INTERIOR_MODE,
                             composition=bq.COMPOSITION,
                             palette_roster="data/palettes/score3_colormaps.json",
                             vivid_companion=BMB.VIVID_PALETTE,
                             maxiter="deep_center_finder._maxiter_for_fw(fw)"),
        render_recipe=cc.render_recipe_stamp(bq.PALETTE_SOURCE),
        sitting_cut=dict(
            run_dir=str(args.run_dir), max_rows=args.max_rows,
            n_in=cut["n_in"], n_sitting=cut["n_sitting"], n_over_cap=cut["n_over_cap"],
            stages=cut["stages"], cells=cut["cells"], balances=cut["balances"],
            cut_wall_s=round(cut_wall, 1), morph_cache=cache_rep,
            auto_labeled_never_presented=[
                dict(cx=r["cx"], cy=r["cy"], fw=r["fw"], partition=r["partition"],
                     **r["auto_label"]) for r in res["auto_labeled"]],
        ),
        bar_readability=dict(
            n=len(readable), of=len(full),
            definition=("served rows carrying BOTH view_fit_v1.1 (provenance.fit_score / "
                        "fit_model) and composite_v3 (provenance.composite) — the slice the "
                        "pre-registered +0.1181 delta-AP margin reads on"),
            by_partition=dict(Counter(r["provenance"]["family"] for r in readable))),
        calibration_aids="NONE — no exemplars, no reference strip, no score shown",
    )
    cc.write_jsonl(full, str(bdir / "images.jsonl"))
    (bdir / "batch.json").write_text(json.dumps(bj, indent=2, default=str) + "\n",
                                     encoding="utf-8")
    if not (bdir / "scores.json").exists():
        (bdir / "scores.json").write_text("{}", encoding="utf-8")

    print(f"\ncut: {cut['n_in']} in -> {cut['n_sitting']} sitting "
          f"(+{cut['n_over_cap']} over cap); wall {cut_wall/60:.1f} min")
    for s in cut["stages"]:
        print(f"  {s['stage']:18s} removed {s['removed']:5d}  "
              + json.dumps({k: v for k, v in s.items()
                            if k in ("looks_kept", "unembeddable_kept",
                                     "no_canonical_verdict_kept", "unmeasured_kept")}))
    print(f"  morph cache: {json.dumps(cache_rep)}")
    print(f"  assign_split = {(split, biased, source)}")
    print(f"\nBAR READABILITY: {len(readable)}/{len(full)} served rows carry BOTH "
          f"view_fit_v1.1 and composite_v3")
    print(f"  by partition: {json.dumps(bj['bar_readability']['by_partition'])}")
    print(f"\n-> {bdir}   (NOTHING RENDERED — run `render` next)")
    return 0


def _render_one(job):
    """Both crops for one row. Atomic per file: render to a partial beside the target and
    rename, so a kill mid-render can never leave a TRUNCATED jpg that reads as done forever.

    THE PARTIAL MUST STILL END IN `.jpg`. The engine infers the image format from the output
    extension, so the obvious `<id>.jpg.tmp` is not a slower path or a warning — it is a hard
    `failed to write ...: The file extension ."tmp" was not recognized as an image format`,
    i.e. a 100% failure rate that looks exactly like a broken renderer. It cost 50 renders to
    find. `<id>.part.jpg` is the same atomicity with an extension the engine can write, and it
    cannot be mistaken for a finished crop because every reader (`needs`, the completeness
    count, the sheet's route walk) addresses crops by exact `<image_id>.jpg`."""
    import corpus_common as cc
    import build_q4_harvest_batches as bq
    import build_minibrot_batch as BMB
    row, crops, vivid, timeout = job
    iid, render = row["image_id"], row["render"]
    for out, pal, src in ((crops / f"{iid}.jpg", render["palette"], bq.PALETTE_SOURCE),
                          (vivid / f"{iid}.jpg", BMB.VIVID_PALETTE, BMB.VIVID_SOURCE)):
        if out.exists():
            continue
        tmp = out.with_name(f"{out.stem}.part.jpg")
        try:
            cc.render_corpus_crop(dict(render, palette=pal), str(tmp), palette_source=src,
                                  timeout=timeout, threads=bq.RENDER_THREADS)
            os.replace(tmp, out)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return iid


def stage_render(args) -> int:
    """Render both crops for every row. IDEMPOTENT — an existing crop is skipped, so a kill
    and a relaunch resume exactly where the last one stopped, and the per-unit checkpoint is
    the crop file itself (there is no separate progress file to fall out of sync with disk)."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import corpus_common as cc
    import build_q4_harvest_batches as bq

    cc.set_below_normal_priority()
    bdir = Path(cc.batch_dir(SITTING_BATCH))
    if not (bdir / "images.jsonl").exists():
        raise SystemExit(f"{bdir/'images.jsonl'} missing — run `draw` first.")
    rows = cc.read_jsonl(str(bdir / "images.jsonl"))
    crops, vivid = Path(cc.crops_dir(SITTING_BATCH)), Path(cc.vivid_dir(SITTING_BATCH))
    crops.mkdir(parents=True, exist_ok=True)
    vivid.mkdir(parents=True, exist_ok=True)

    def needs(r):
        return not (crops / f"{r['image_id']}.jpg").exists() or \
            not (vivid / f"{r['image_id']}.jpg").exists()

    todo = [r for r in rows if needs(r)]
    deadline = (time.time() + args.max_minutes * 60.0) if args.max_minutes else None
    print(f"render {SITTING_BATCH}: {len(rows)} rows, {len(todo)} need crops "
          f"= up to {2*len(todo)} renders, {args.workers}x{bq.RENDER_THREADS} threads",
          flush=True)
    t0, done, fails, stopped = time.time(), 0, [], False
    t_win, n_win = t0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_render_one, (r, crops, vivid, args.render_timeout)): r
                for r in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:                               # noqa: BLE001
                fails.append(dict(image_id=futs[fut]["image_id"],
                                  err=f"{type(e).__name__}: {str(e)[:200]}"))
            done += 1
            if done % 25 == 0 or done == len(todo):
                now = time.time()
                recent = (now - t_win) / max(done - n_win, 1)
                # Reproject from RECENT throughput, never the run-to-date average: the draw is
                # cell-round-robin so deep rows are spread through, but a rate that is falling
                # must be read off the tail (`CLAUDE.md`, "Projecting a long run's wall clock").
                print(json.dumps(dict(done=done, of=len(todo),
                                      elapsed_min=round((now - t0) / 60, 1),
                                      recent_s_per_row=round(recent, 2),
                                      eta_min=round((len(todo) - done) * recent / 60, 1),
                                      failed=len(fails))), flush=True)
                t_win, n_win = now, done
            if deadline and time.time() > deadline:
                stopped = True
                for f2 in futs:
                    f2.cancel()
                break
    if fails:
        # The WHOLE failure list, never a head slice: a truncated error log describes the
        # fastest-returning failure class, not the population (`CLAUDE.md`, "Four rules").
        p = Path(__import__("paths").scratch("sitting_v2", "render_failures.json"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(n=len(fails),
                                     by_class=dict(Counter(f["err"].split(":")[0]
                                                           for f in fails)),
                                     failures=fails), indent=2), encoding="utf-8")
        print(f"  !! {len(fails)} render failures -> {p}")
    miss = sum(1 for r in rows if needs(r))
    print(f"render: {done} rows this pass" + ("  [STOPPED at the time bound]" if stopped else ""))
    print(f"  {len(rows)-miss}/{len(rows)} complete"
          + ("  COMPLETE" if miss == 0 else f"  INCOMPLETE — {miss} rows still need crops"))
    return 0 if miss == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="report what a sitting WOULD contain; serve nothing")
    d.add_argument("--run-dir", required=True)
    d.add_argument("--max-rows", type=int, default=MAX_ROWS)
    d.add_argument("--embed-limit", type=int, default=None,
                   help="bound the morph pass (dry-run only). The unembedded remainder is "
                        "counted as budget_not_reached, never silently dropped.")
    d.add_argument("--no-cache", action="store_true",
                   help="bypass the persistent morph-embed store (a COLD timing arm)")
    d.add_argument("--out", default=None)

    w = sub.add_parser("draw", help="cut the sitting and write the registered batch")
    w.add_argument("--run-dir", required=True)
    w.add_argument("--max-rows", type=int, default=MAX_ROWS)
    w.set_defaults(fn=stage_draw)

    r = sub.add_parser("render", help="render both crops per row; resumable, idempotent")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--render-timeout", type=float, default=600.0)
    r.add_argument("--max-minutes", type=float, default=0.0)
    r.set_defaults(fn=stage_render)

    a = ap.parse_args()
    if getattr(a, "fn", None) is not None:
        if getattr(a, "workers", 0) and a.workers > 4:
            print("refusing >4 concurrent engine processes (CLAUDE.md)", file=sys.stderr)
            return 2
        return a.fn(a)

    import time                                               # noqa: E402
    import paths                                              # noqa: E402
    import corpus_common as cc                                # noqa: E402
    from tools.wallpaper import morph_embed_cache as mec      # noqa: E402
    # A cold morph pass is ~an hour of engine renders launched through a helper that takes no
    # creationflags; the child inherits this class, which is the only lever that reaches them.
    cc.set_below_normal_priority()
    rows = load_queue(a.run_dir)
    scratch = Path(paths.scratch("sitting_cutter", "fields"))
    cache = None if a.no_cache else mec.MorphEmbedCache().open()
    t0 = time.time()
    res = cut_sitting(rows, max_rows=a.max_rows,
                      embed=make_embedder(scratch, a.embed_limit, cache),
                      progress=lambda d: print(json.dumps(d), flush=True))
    elapsed = time.time() - t0
    rep = res["report"]
    rep["run_dir"] = str(a.run_dir)
    rep["SERVED"] = False
    rep["wall_s"] = round(elapsed, 2)
    rep["morph_cache"] = cache.report() if cache else "DISABLED (--no-cache)"
    if cache:
        cache.close()
    rep["note"] = ("DRY RUN: no manifest written, no export, no crop rendered. `serve` is "
                   "what builds a sitting.")
    out = Path(a.out) if a.out else Path(paths.scratch("sitting_cutter", "dry_run.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    print(f"\n-> {out}   (NOTHING SERVED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
