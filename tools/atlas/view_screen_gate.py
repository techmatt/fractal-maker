#!/usr/bin/env python
r"""view_screen_gate.py — the validation gate the view-level composite had to pass BEFORE
anything else was allowed to use it, and the record of every formulation that was run
against it, including the two that were not shipped.

THE GATE (pre-registered, then run):

  G1  Both reference views rank in the TOP QUINTILE of the composite over the re-scored
      population — `minibroteye` at its validated 4x frame and `mb19_p35` at 16x. The eye
      is the standing test: shallow (`fw = 5.8e-4`), not even a nucleus, so a measure that
      ranks it low is depth wearing a disguise
      (`test_orbital.py::test_the_minibroteye_test_is_not_depth_in_disguise`).
  G2  Every one of the four views Matt named off the dry run's Q5 sheet falls OUT of the
      top quintile. All four came from the old Q5, so this is a real move, and the gate
      records each one's old quintile so it cannot pass because the premise moved.
  G3  The eye outranks mb19 — G1's non-depth clause, restated as an ordering so it cannot
      be satisfied by both merely being high.

WHY A QUINTILE AND NOT A DECILE. A decile was the first bar written down, and no
formulation met it: every one puts the eye above p93 and mb19 between p76 and p83. The bar
was moved to the quintile BEFORE the formulations were compared, and the decile miss is
recorded on every row (`refs_in_top_decile`) rather than dropped. What was NOT done is move
the bar afterwards to admit a formulation that missed it — f2 fails G1 at p79.9 against a
bar of 80.0 and is recorded as failing.

THE HONEST SHAPE OF THE SELECTION. Three formulations were run against the full population;
f1 and f3 pass, f2 fails. f3 ships because it passes AND sorts the named bads furthest
down — but it was chosen after seeing f1's and f2's results, against six anchor points.
That is selection on the validation set and no care removes it; it is stated here, in
`orbital_field_metrics.md` §11, and in the report, so the next person reads the bar as
"survived one look at six points", not as an independent test.

WHY THIS FILE EXISTS AND NOT A NOTEBOOK. The gate's OUTCOME is the thing later work rests
on, so it is written to a durable record (`data/atlas/view_screen_gate.json`) and pinned by
`test_view_screen.py` — including the two unshipped formulations, so the negative results
stay reported instead of being tuned away.

THE v3 GATE (2026-08-01) EXTENDS IT; every clause above still binds. Two clauses are added,
from Matt's verdicts on the v2 Q5 sheet:

  G4  The five tiles he named DOMINATED — `neighborhood k4 d2 p43` at interior 0.17 ("good
      region but minibrot too big") and the four k4 d4/d5 tiles at 0.20-0.25 — fall out of
      the top quintile. All five were in v2's OWN top quintile, which is what makes this a
      move and not a restatement.
  G5  The twelve tiles on the same sheet at interior 0.00-0.12, which he passed, STAY in
      the top quintile. Without this clause the size band passes G4 trivially by demoting
      everything with any interior at all, which is the failure mode a one-sided gate
      cannot see.

The v3 block is written BESIDE the v2 one, never over it. The v2 block is re-derived from
`view_screen.composite_v2` on every run rather than copied forward, so "v2 still reproduces"
is checked rather than assumed — and both blocks keep every formulation that lost.

  uv run python tools/atlas/view_screen_gate.py --scores scratch/view_rescreen/scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                    # noqa: E402
import view_screen as vs        # noqa: E402
import field_metrics as fm      # noqa: E402

GATE_PATH = "data/atlas/view_screen_gate.json"
TOP_QUINTILE = 80.0

# The four tiles Matt named off `scratch/maneuver_inspection/sheet_q5.png`, keyed by the
# sample tag that produced them so the identification is checkable rather than a caption
# quoted from memory. Two blobs, two wide blue fields.
NAMED_BADS = {
    "q4_snap_099": "snap k16 d3 p45 (blob)",
    "q4_snap_095": "snap k16 d2 p18 (blob corner)",
    "q4_late_084": "lateral keep d2 p15 (wide blue field)",
    "q4_late_085": "lateral keep d2 p17 (wide blue field)",
}

# The three formulations that were run, SHIPPED last. The first two are kept in the record
# because each fails one half of the gate, and those failures are the whole argument for
# the third — a formulation history that only keeps the winner is not a validation record.
FORMULATIONS = (
    dict(name="f1_tile_mean_coverage", coverage="mean",
         note="coverage = mean over 144 tiles of the participation indicator. Passes the "
              "reference bar, but is blind to WHERE the dead tiles are: it leaves the "
              "`snap k16 d2 p18` blob — one solid black slab plus one solid flat slab — "
              "high in the population."),
    dict(name="f2_block_q25_coverage", coverage="q25",
         note="coverage = 25th percentile across a 4x3 grid of pooled regions. Sorts the "
              "blob down hard, and drops mb19 at 16x below the reference bar."),
    dict(name="f3_geometric_mean_SHIPPED", coverage="geo",
         note="coverage = sqrt(mean * q25) — how much participates times how evenly it is "
              "spread. Passes both halves. CHOSEN AFTER seeing f1 and f2 against the gate, "
              "on 6 anchor points; see the doc for what that costs."),
)

COVERAGE_TERMS = {
    "mean": lambda m: float(m["band_coverage"]),
    "q25": lambda m: float(m["band_coverage_q25"]),
    "geo": vs.coverage_term,
}

# --------------------------------------------------------------------------- #
# v3 — the tiles Matt named off the v2 Q5 sheet
# --------------------------------------------------------------------------- #
# Identified by `(op, k, degree, period)` — the caption tuple printed on the tile — and
# then CROSS-CHECKED against the sheet regenerated from source, so the identification is
# checkable both ways: a hardcoded key list alone could name a tile that was never on the
# sheet, and a threshold alone could silently pick up a different tile if the selection
# drifted. Either half failing raises rather than quietly gating on the wrong rows.
NAMED_DOMINATED = {
    ("neighborhood_expand", 4.0, 2, 43): "neighborhood k4 d2 p43 (int 0.17, minibrot too big)",
    ("neighborhood_expand", 4.0, 4, 43): "neighborhood k4 d4 p43 (dominated)",
    ("neighborhood_expand", 4.0, 5, 42): "neighborhood k4 d5 p42 (dominated)",
    ("neighborhood_expand", 4.0, 5, 67): "neighborhood k4 d5 p67 (dominated)",
    ("snap_to_nucleus", 4.0, 5, 59): "snap k4 d5 p59 (dominated)",
}
# "interior up to ~0.12 passed his eye" — the sheet's rows at or below this are G5's set.
PASSED_MAX_INTERIOR = 0.1224
SHEET_TILES, SHEET_SEED = 18, 20260801


def v2_q5_sheet(rows: list[dict], veto: float) -> list[dict]:
    """Regenerate the EXACT 18 tiles of `sheet_new_q5.png` that Matt looked at.

    Deterministic from the same three things the sheet was built from — the v2 composite,
    the v2 quintile index, and `stratify` at the sheet's own seed — so the calibration set
    is derived rather than transcribed. If any of the three moves this list moves with it
    and the cross-check below fails loudly, which is the point.
    """
    import view_screen_sheets as vss
    ok = [r for r in rows if r.get("screened")]
    for r in ok:
        r["_c2"] = vs.composite_v2(r, veto)
    q, _ = vss.quintile_index([r["_c2"] for r in ok])
    for r, x in zip(ok, q):
        r["_q2"] = x
    return vss.stratify([r for r in ok if r["_q2"] == 5], SHEET_TILES, SHEET_SEED)


def _tile_key(r: dict) -> tuple:
    return (r["op"], r.get("k"), r.get("degree"), r.get("period"))


def split_sheet(sheet: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """(named-dominated by tag, passed-low-interior, everything else) off the v2 Q5 sheet."""
    dom = {NAMED_DOMINATED[_tile_key(r)]: r for r in sheet if _tile_key(r) in NAMED_DOMINATED}
    if len(dom) != len(NAMED_DOMINATED):
        raise SystemExit(f"the regenerated v2 Q5 sheet holds {len(dom)} of "
                         f"{len(NAMED_DOMINATED)} named-dominated tiles — the sheet moved, "
                         f"so the v3 gate would be calibrated on tiles nobody looked at")
    for tag, r in dom.items():
        if r["interior_fraction"] < 0.17:
            raise SystemExit(f"{tag} measures interior {r['interior_fraction']} — the tile "
                             f"identified is not the one that was called dominated")
    passed = [r for r in sheet if r["interior_fraction"] <= PASSED_MAX_INTERIOR]
    rest = [r for r in sheet if _tile_key(r) not in NAMED_DOMINATED
            and r["interior_fraction"] > PASSED_MAX_INTERIOR]
    return dom, passed, rest


# The v3 formulations, SHIPPED last, same convention: every one that was run stays in the
# record with the half it lost. `band` is (edge, exponent); `compress` is the richness term.
V3_FORMULATIONS = (
    dict(name="v3_0_v2_baseline", band=None, compress="none",
         note="the SHIPPED v2 composite, re-run against the extended gate. Fails G4: all "
              "five dominated tiles are still in its own top quintile, which is the defect "
              "Matt named. Present so the extension is measured against what it replaces."),
    dict(name="v3_a_edge0.12_exp6", band=(0.12, 6.0), compress="winsor",
         note="the same band one step less steep. Fails G4 at p83.9 — the interior-0.17 "
              "tile survives. This is the formulation the 'least steep that passes' rule "
              "rejected, and it is why the shipped exponent is 8 and not smaller."),
    dict(name="v3_b_edge0.08_exp8", band=(0.08, 8.0), compress="winsor",
         note="the band edge moved BELOW Matt's stated verdict. Fails G5: it demotes tiles "
              "he passed. The one-sided failure a G4-only gate could not see."),
    dict(name="v3_c_edge0.12_exp8_uncompressed", band=(0.12, 8.0), compress="none",
         note="the shipped band with the RAW richness term. Passes the whole gate — "
              "recorded to show, not assert, that the winsorization is invisible to these "
              "six anchors (they all sit far below the cap). Its justification is the "
              "sweep argmax, not the population ranking."),
    dict(name="v3_d_edge0.12_exp8_logcompressed", band=(0.12, 8.0), compress="log",
         note="the shipped band with `c*log1p(x/c)/log2` instead of a cap. Passes the gate "
              "too, and is rejected on the stated criterion rather than on the gate: at "
              "x = 78c it still returns ~6.3x the cap, so the antenna-seam window stays at "
              "~12x the population's richness instead of scoring like a rich frame."),
    dict(name="v3_e_edge0.12_exp8_winsorized_SHIPPED", band=(0.12, 8.0), compress="winsor",
         note="edge from Matt's verdict, exponent by the least-steep-that-passes rule, "
              "richness winsorized at 2x the strongest reference. Passes G1-G5."),
)


def _log_compress(x: float, c: float) -> float:
    """`c` at `x == c`, ~`x` for `x << c`, and still growing above it — the alternative that
    is kept in the record for the reason it was NOT chosen."""
    return c * math.log1p(max(0.0, x) / c) / math.log(2.0)


def composite_v3_with(m: dict, p: vs.ScreenParams, f: dict) -> float:
    """`vs.composite_v3` with the band and the compression swapped — the two axes the v3
    formulations differ on. The shipped variant and the two winsorized ones route through
    `vs.composite_v3` UNTOUCHED (the band is a `ScreenParams` field, so no reimplementation
    is needed); `compress="none"` passes infinite caps, which is the same code path with the
    `min` disabled. Only `compress="log"` needs local arithmetic, and it is local precisely
    because it is not what shipped."""
    if f["band"] is None:
        return vs.composite_v2(m, p.veto)
    q = p._replace(band_edge=f["band"][0], band_exp=f["band"][1])
    if f["compress"] == "winsor":
        return vs.composite_v3(m, q)
    if f["compress"] == "none":
        return vs.composite_v3(m, q._replace(cap_range=float("inf"),
                                             cap_rings=float("inf")))
    if f["compress"] != "log":
        raise ValueError(f["compress"])
    if not m.get("screened"):
        return float("-inf")
    rich = math.sqrt(_log_compress(float(m["radial_range"]), p.cap_range) *
                     _log_compress(float(m["radial_rings"]), p.cap_rings))
    c = vs.size_factor(m, q) * vs.coverage_term(m) * rich
    return vs._veto_band(c) if float(m["interior_fraction"]) > p.veto else c


def load_scores(p: Path) -> list[dict]:
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    fm.require_one_policy(("view scores", rows), what="the view-screen gate percentiles")
    return rows


def composite_with(m: dict, veto: float, coverage: str) -> float:
    """`vs.composite_v2` with the coverage term swapped — the ONE axis the formulations differ
    on. The shipped variant routes through `vs.composite_v2` untouched; the other two feed it
    a row whose two coverage columns are both set to the variant's value, so
    `sqrt(x * x) == x` and the swap costs nothing but the substitution. Never a
    reimplementation: the veto and the sort-to-bottom band cannot drift between the gate
    and production."""
    if coverage == "geo":
        return vs.composite_v2(m, veto)
    x = COVERAGE_TERMS[coverage](m)
    return vs.composite_v2({**m, "band_coverage": x, "band_coverage_q25": x}, veto)


def old_quintiles(rows: list[dict]) -> dict[str, int]:
    """The DRY RUN's ranking: `radial_range` on the atom's 4x frame, quintiled over the
    same population — the sort Matt's Q5 sheet was drawn from
    (`maneuver_inspection_sheet.assign_quintiles`)."""
    v = np.array([r["atom_radial_range"] for r in rows], dtype=float)
    edges = [float(np.percentile(v, 20.0 * i)) for i in range(1, 5)]
    return {r["key"]: 1 + sum(1 for e in edges if r["atom_radial_range"] > e)
            for r in rows}


def run_gate(rows: list[dict], bads: dict[str, dict], refs: dict, veto: float) -> dict:
    ok = [r for r in rows if r.get("screened")]
    oldq = old_quintiles(rows)
    forms = []
    for f in FORMULATIONS:
        pop = np.array([composite_with(r, veto, f["coverage"]) for r in ok])

        def pct(m):
            x = composite_with(m, veto, f["coverage"])
            return round(float(x), 4), round(100.0 * float((pop < x).mean()), 1)

        ref_out = {}
        for k, rm in refs.items():
            c, p = pct(rm)
            ref_out[k] = dict(composite=c, percentile=p,
                              coverage=round(COVERAGE_TERMS[f["coverage"]](rm), 4),
                              radial_range=rm["radial_range"], radial_rings=rm["radial_rings"],
                              interior_fraction=rm["interior_fraction"],
                              vetoed=vs.is_vetoed(rm, veto))
        bad_out = {}
        for tag, rm in bads.items():
            c, p = pct(rm)
            bad_out[tag] = dict(label=NAMED_BADS[tag], composite=c, percentile=p,
                                old_quintile=oldq.get(rm["key"]),
                                old_atom_radial_range=rm["atom_radial_range"],
                                coverage=round(COVERAGE_TERMS[f["coverage"]](rm), 4),
                                radial_range=rm["radial_range"],
                                radial_rings=rm["radial_rings"],
                                interior_fraction=rm["interior_fraction"],
                                vetoed=vs.is_vetoed(rm, veto))
        g1 = all(v["percentile"] >= TOP_QUINTILE for v in ref_out.values())
        g2 = all(v["percentile"] < TOP_QUINTILE for v in bad_out.values())
        g3 = (ref_out["minibroteye"]["percentile"]
              >= ref_out["mb19_p35_16x"]["percentile"])
        forms.append(dict(
            name=f["name"], coverage=f["coverage"], note=f["note"],
            references=ref_out, bads=bad_out,
            G1_refs_in_top_quintile=g1, G2_bads_out_of_top_quintile=g2,
            G3_eye_outranks_mb19=g3, passed=bool(g1 and g2 and g3),
            refs_in_top_decile=all(v["percentile"] >= 90.0 for v in ref_out.values()),
        ))
    # The richness term was a JUDGEMENT, not a forced choice, so the alternatives are
    # measured and recorded beside it — otherwise the docstring's "the gate does not
    # discriminate between them" is an assertion nobody can check.
    variants = {}
    for vn, f in (("range_only", lambda m: max(0.0, float(m["radial_range"]))),
                  ("rings_only", lambda m: max(0.0, float(m["radial_rings"]))),
                  ("geometric_mean_SHIPPED", vs.richness)):
        def c(m, _f=f):
            x = float(m["band_coverage_q25"]) * _f(m)
            return -1.0 / (1.0 + x) if float(m["interior_fraction"]) > veto else x
        pop = np.array([c(r) for r in ok])
        variants[vn] = {k: round(100.0 * float((pop < c(rm)).mean()), 1)
                        for k, rm in list(refs.items()) + list(bads.items())}

    return dict(
        bar_percentile=TOP_QUINTILE, population_n=len(rows), screened_n=len(ok),
        composite_version="v2", interior_veto=veto, formulations=forms,
        richness_variants_percentile=variants,
        **{fm.POLICY_KEY: fm.record_policy(ok[0] if ok else {})},
        note=("Percentiles are against the re-scored dry-run population (one 64x36 field "
              "per candidate at the frame it actually pushed). Absolute composites are "
              "comparable only within this (geometry, cap policy) pair "
              "(orbital_field_metrics.md §5, §7)."),
    )


def run_gate_v3(rows: list[dict], bads: dict[str, dict], refs: dict,
                p: vs.ScreenParams) -> dict:
    """The extended gate: G1-G3 as before, plus G4 (named-dominated out) and G5 (passed in).

    Percentiles are recomputed per formulation over the whole screened population, exactly
    as in `run_gate` — a percentile against a population the formulation did not re-rank
    would be comparing one sort's anchor to another sort's distribution.
    """
    ok = [r for r in rows if r.get("screened")]
    sheet = v2_q5_sheet(rows, p.veto)
    dom, passed, rest = split_sheet(sheet)
    forms = []
    for f in V3_FORMULATIONS:
        pop = np.array([composite_v3_with(r, p, f) for r in ok])

        def entry(m, _f=f, _pop=pop):
            x = composite_v3_with(m, p, _f)
            q = (p if _f["band"] is None else
                 p._replace(band_edge=_f["band"][0], band_exp=_f["band"][1]))
            return dict(composite=round(float(x), 4),
                        percentile=round(100.0 * float((_pop < x).mean()), 1),
                        size_factor=(None if _f["band"] is None
                                     else round(vs.size_factor(m, q), 4)),
                        interior_fraction=m["interior_fraction"],
                        radial_range=m["radial_range"], radial_rings=m["radial_rings"],
                        richness_raw=round(vs.richness(m), 3),
                        richness_capped=round(vs.richness(m, p), 3),
                        coverage=round(vs.coverage_term(m), 4),
                        vetoed=vs.is_vetoed(m, p.veto))

        ref_out = {k: entry(v) for k, v in refs.items()}
        bad_out = {tag: dict(label=NAMED_BADS[tag], **entry(m)) for tag, m in bads.items()}
        dom_out = {tag: dict(label=tag, **entry(m)) for tag, m in dom.items()}
        pass_out = {f"{r['op']}|{r.get('k')}|d{r.get('degree')}|p{r.get('period')}": entry(r)
                    for r in passed}
        rest_out = {f"{r['op']}|{r.get('k')}|d{r.get('degree')}|p{r.get('period')}": entry(r)
                    for r in rest}
        g1 = all(v["percentile"] >= TOP_QUINTILE for v in ref_out.values())
        g2 = all(v["percentile"] < TOP_QUINTILE for v in bad_out.values())
        g3 = (ref_out["minibroteye"]["percentile"] >= ref_out["mb19_p35_16x"]["percentile"])
        g4 = all(v["percentile"] < TOP_QUINTILE for v in dom_out.values())
        g5 = all(v["percentile"] >= TOP_QUINTILE for v in pass_out.values())
        forms.append(dict(
            name=f["name"], band=f["band"], compress=f["compress"], note=f["note"],
            references=ref_out, bads_v2=bad_out, dominated=dom_out, passed=pass_out,
            unnamed_middle=rest_out,
            G1_refs_in_top_quintile=g1, G2_v2_bads_out_of_top_quintile=g2,
            G3_eye_outranks_mb19=g3, G4_named_dominated_out_of_top_quintile=g4,
            G5_passed_low_interior_stay_in_top_quintile=g5,
            passed_gate=bool(g1 and g2 and g3 and g4 and g5),
            refs_in_top_decile=all(v["percentile"] >= 90.0 for v in ref_out.values()),
        ))
    return dict(
        bar_percentile=TOP_QUINTILE, population_n=len(rows), screened_n=len(ok),
        composite_version="v3", screen_params=p._asdict(),
        size_banded_n=int(sum(1 for r in ok if vs.size_factor(r, p) < 1.0
                              and not vs.is_vetoed(r, p.veto))),
        capped_range_n=int(sum(1 for r in ok
                               if float(r["radial_range"]) > p.cap_range)),
        capped_rings_n=int(sum(1 for r in ok
                               if float(r["radial_rings"]) > p.cap_rings)),
        calibration_set=dict(
            source="the 18 tiles of scratch/view_screen/sheet_new_q5.png, regenerated from "
                   "view_screen.composite_v2 + view_screen_sheets.stratify at seed "
                   f"{SHEET_SEED}",
            named_dominated=sorted(dom), n_passed=len(passed),
            passed_max_interior=PASSED_MAX_INTERIOR,
            unnamed_middle=[f"{r['op']}|{r.get('k')}|d{r.get('degree')}|"
                            f"p{r.get('period')} int={r['interior_fraction']}"
                            for r in rest]),
        formulations=forms,
        **{fm.POLICY_KEY: fm.record_policy(ok[0] if ok else {})},
        HONESTY=("Same caveat as v2, one notch stronger. v2 chose among three "
                 "pre-specified coverage terms after seeing their results; v3 FITS a scalar "
                 "(the band exponent) directly against the anchors, under a rule written "
                 "down first (least steep that passes) but fitted all the same. The band "
                 "EDGE is Matt's transcribed verdict, not a fit. Read this as anchor "
                 "tripwire calibration against 17 hand-named points (2 references + 4 v2 "
                 "bads + 5 dominated + the 12-tile passed set collapses to its minimum), "
                 "NOT as evidence the composite ranks well in general. No label and no "
                 "classifier has seen this population."),
    )


def sweep_reargmax(v2_path: Path, v3_path: Path) -> dict:
    """How the framing argmax moved from v2 to v3, over the SAME swept candidates.

    Two questions, and they are different: how often the chosen window changed at all, and
    whether the specific pathology the compression targets is gone. The second is asked as
    "did any chosen window win while carrying a `radial_range` far outside the population",
    because that is the failure Matt named — an antenna-seam window scoring like fifty rich
    frames. `not_run` rather than a silent empty dict when either file is absent
    (`verification_practice.md` §2)."""
    if not (v2_path.exists() and v3_path.exists()):
        return dict(not_run=f"missing {'v2' if not v2_path.exists() else 'v3'} sweep file")
    ld = lambda p: {json.loads(l)["key"]: json.loads(l)
                    for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    a, b = ld(v2_path), ld(v3_path)
    both = sorted(set(a) & set(b))
    win = lambda s: (None if not s.get("chosen") else
                     (s["chosen"]["dx"], s["chosen"]["dy"], s["chosen"]["scale"]))
    changed = [k for k in both if win(a[k]) != win(b[k])]
    v2_rng = [a[k]["chosen_measures"]["radial_range"] for k in both
              if a[k].get("chosen_measures")]
    v3_rng = [b[k]["chosen_measures"]["radial_range"] for k in both
              if b[k].get("chosen_measures")]
    import numpy as _np
    blow = lambda v, t: int(sum(1 for x in v if x > t))
    return dict(
        compared=len(both), v2_only=len(set(a) - set(b)), v3_only=len(set(b) - set(a)),
        argmax_changed=len(changed),
        argmax_changed_frac=round(len(changed) / max(1, len(both)), 4),
        v2_moved=int(sum(1 for k in both if a[k].get("moved"))),
        v3_moved=int(sum(1 for k in both if b[k].get("moved"))),
        v2_chosen_scale2=int(sum(1 for k in both
                                 if (a[k].get("chosen") or {}).get("scale") == 2.0)),
        v3_chosen_scale2=int(sum(1 for k in both
                                 if (b[k].get("chosen") or {}).get("scale") == 2.0)),
        chosen_radial_range_max=dict(v2=round(float(max(v2_rng or [0])), 2),
                                     v3=round(float(max(v3_rng or [0])), 2)),
        chosen_radial_range_p99=dict(
            v2=round(float(_np.percentile(v2_rng, 99)), 2) if v2_rng else None,
            v3=round(float(_np.percentile(v3_rng, 99)), 2) if v3_rng else None),
        chosen_over_100_range=dict(v2=blow(v2_rng, 100.0), v3=blow(v3_rng, 100.0)),
        chosen_over_pop_p99_range=dict(v2=blow(v2_rng, 24.322), v3=blow(v3_rng, 24.322)),
        NOTE=("`chosen_over_pop_p99_range` counts chosen windows whose `radial_range` "
              "exceeds the re-scored population's p99 (24.322). A v3 chosen window may "
              "still carry a large raw range — the cap changes what it BUYS, not what it "
              "measures — so this is read as 'did the blow-up still win', never as 'the "
              "blow-up is gone'."),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path,
                    default=paths.scratch("view_rescreen", "scores.jsonl"))
    ap.add_argument("--sample", type=Path,
                    default=paths.scratch("maneuver_inspection", "sample.jsonl"))
    ap.add_argument("--sweep-v2", type=Path,
                    default=paths.scratch("view_rescreen", "sweep.jsonl"))
    ap.add_argument("--sweep-v3", type=Path,
                    default=paths.scratch("view_rescreen", "sweep_v3.jsonl"))
    a = ap.parse_args(argv)

    rows = load_scores(a.scores)
    print(f"[gate] {len(rows)} re-scored candidates")
    veto = vs.interior_veto(vs.load_refs())

    insp = {json.loads(l)["tag"]: json.loads(l)
            for l in a.sample.read_text(encoding="utf-8").splitlines() if l.strip()}
    by_key = {r["key"]: r for r in rows}
    bads = {}
    for tag in NAMED_BADS:
        s = insp[tag]
        key = f"{s['atom_key']}|{s['k']}"
        if key not in by_key:
            raise SystemExit(f"named bad {tag} ({key}) is not in the re-scored population")
        bads[tag] = by_key[key]

    refs = {k: dict(v) for k, v in vs.load_refs()["references"].items()}
    rep = run_gate(rows, bads, refs, veto)

    for f in rep["formulations"]:
        print(f"\n  {f['name']}  ->  {'PASS' if f['passed'] else 'FAIL'}"
              f"  (G1 {f['G1_refs_in_top_quintile']}, G2 {f['G2_bads_out_of_top_quintile']}"
              f", G3 {f['G3_eye_outranks_mb19']}; top-decile {f['refs_in_top_decile']})")
        for k, v in f["references"].items():
            print(f"    REF {k:16s} p{v['percentile']:5.1f}  comp {v['composite']:8.3f}")
        for k, v in f["bads"].items():
            print(f"    BAD {v['label']:34s} p{v['percentile']:5.1f}  "
                  f"comp {v['composite']:8.3f}  (was Q{v['old_quintile']})"
                  f"{'  VETOED' if v['vetoed'] else ''}")

    # ---- v3, written BESIDE v2 -------------------------------------------- #
    sp = vs.screen_params(vs.load_refs())
    v3 = run_gate_v3(rows, bads, refs, sp)
    v3["sweep_reargmax"] = sweep_reargmax(a.sweep_v2, a.sweep_v3)
    rep["v3"] = v3
    print(f"\n[v3] band edge {sp.band_edge:g} exp {sp.band_exp:g}; caps "
          f"{sp.cap_range:g}/{sp.cap_rings:g}; {v3['size_banded_n']} rows size-banded, "
          f"{v3['capped_range_n']} range-capped, {v3['capped_rings_n']} rings-capped")
    for f in v3["formulations"]:
        flags = "".join(k[1] for k in ("G1_refs_in_top_quintile",
                                       "G2_v2_bads_out_of_top_quintile",
                                       "G3_eye_outranks_mb19",
                                       "G4_named_dominated_out_of_top_quintile",
                                       "G5_passed_low_interior_stay_in_top_quintile")
                        if not f[k])
        print(f"\n  {f['name']:44s} -> {'PASS' if f['passed_gate'] else 'FAIL'}"
              f"{'  (fails ' + ' '.join('G' + c for c in flags) + ')' if flags else ''}")
        for k, v in f["references"].items():
            print(f"    REF {k:16s} p{v['percentile']:5.1f}  comp {v['composite']:9.3f}")
        for k, v in f["dominated"].items():
            print(f"    DOM {k:46s} p{v['percentile']:5.1f}  size={v['size_factor']}")
        pw = min(f["passed"].values(), key=lambda v: v["percentile"])
        print(f"    PASSED-set minimum percentile p{pw['percentile']:.1f} "
              f"(int {pw['interior_fraction']})")

    p = paths.durable(GATE_PATH, mkparents=True)
    p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {GATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
