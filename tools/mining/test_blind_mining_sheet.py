"""Sheet E — the five properties that make it an INSTRUMENT, each pinned.

Sheet E exists because every labeled render-mode batch is anchored to mining v1, so the only
thing worth testing here is that the properties which un-anchor it actually hold, and hold in
a way that survives somebody editing the builder six months from now:

  1. no mining head is reachable from the builder — a SOURCE SCAN, because the useful failure
     is a future edit that adds a `screen_pred` rank back, not one this run makes;
  2. the population filters actually filter (fresh-by-key, fresh-by-proximity, self-exclusion,
     pair freshness), each exercised on a constructed population whose answer is known;
  3. the mode weighting is the declared one — contested cells weighted, `exp_smoothing` gone —
     and the candidate draw is ordered by the seed and by nothing else;
  4. a written row is BLIND — no `suggested_tier`, no head block, no score field — so the
     labeling rig cannot enter correction mode on it;
  5. the batch is eval-only and registered.

The re-verdict harness is covered here too, on a SYNTHETIC labeled slice: it is the object
that will run once, months from now, on data that does not exist yet, and "it imports" is not
coverage. Both survival branches and both anchored-report branches are exercised, and so is
the one-class cell that must vote neither way.

  uv run pytest tools/mining/test_blind_mining_sheet.py -q
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring", ROOT / "tools" / "corpus",
          ROOT / "tools" / "queries", ROOT / "tools" / "mining"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import batch_registry as br                                    # noqa: E402
from tools.mining import build_blind_mining_sheet as BE        # noqa: E402
from tools.mining import mining_roster as MR                   # noqa: E402
from tools.mining import smooth_equivalence as SE              # noqa: E402

SPEC = BE.SHEETS["e"]
BUILDER = Path(BE.__file__)


# --------------------------------------------------------------------------- #
# 1. NO MINING HEAD. A source scan, not a runtime check.
# --------------------------------------------------------------------------- #
# The mining-head reach: the pin module, the checkpoint constants, the scorer, the suggestion
# rule, and the row/record fields a head stamps. A future edit that re-adds any of them is the
# failure this guards, and it would pass every runtime test in this file.
#
# Matched against CODE TOKENS, never raw text — this module and the builder both say these
# words on purpose, in comments and in the `batch.json` prose that explains what the sheet
# does NOT do, and a substring scan would fail on its own documentation. `code_tokens` drops
# comments and any string that reads as prose (contains whitespace), so a dict KEY like
# `"screen_p_ge3"` is still caught while a sentence mentioning it is not.
FORBIDDEN = (
    "mining_pins", "ACTIVE_MINING_CKPT", "MiningScorer", "mining_gate", "HEAD_VERSION",
    "head_mining_v1", "suggested_tier", "suggest_tier_mining", "expected_tier",
    "screen_pred", "screen_p_ge3", "screen_would_pass_gate", "would_pass_gate",
    "p_ge3", "p_ge2", "sel_score", "MINING_POOL", "MINING_RELEASE", "render_mode_head",
)


def code_tokens(path: Path) -> list[str]:
    """Every NAME token plus every identifier-shaped STRING literal. Comments and docstrings
    (and any other prose string) are dropped."""
    import io
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BufferedReader(fh).readline):
            if tok.type == tokenize.NAME:
                out.append(tok.string)
            elif tok.type == tokenize.STRING:
                s = tok.string.strip("rbfu")
                for q in ('"""', "'''", '"', "'"):
                    if s.startswith(q):
                        s = s[len(q):-len(q)] if s.endswith(q) else s[len(q):]
                        break
                if s and not any(ch.isspace() for ch in s):
                    out.append(s)
    return out


def test_builder_never_reaches_a_mining_head():
    toks = code_tokens(BUILDER)
    hits = sorted({t for t in FORBIDDEN if any(t in tok for tok in toks)})
    assert not hits, (
        f"{BUILDER.name} reaches a mining head: {hits}. Sheet E's whole value is that no "
        f"mining checkpoint touched its draw or its substrate; a score stamped or ranked on "
        f"here makes the slice as unusable as the three correction sheets it replaces.")


def test_the_scan_would_actually_catch_a_regression(tmp_path):
    """The guard's own guard: a scan that matches nothing passes vacuously. This proves the
    tokenizer sees a real emission and ignores the prose that describes one."""
    bad = tmp_path / "bad.py"
    bad.write_text('"""We never stamp head_mining_v1 here."""\n'
                   'row = {"head_mining_v1": {"p_ge3": 1.0}}\n', encoding="utf-8")
    assert sorted({t for t in FORBIDDEN
                   if any(t in tok for tok in code_tokens(bad))}) == ["head_mining_v1", "p_ge3"]
    ok = tmp_path / "ok.py"
    ok.write_text('"""No head_mining_v1 and no p_ge3 appear on a row."""\n'
                  'NOTE = "this row carries no p_ge3 and no suggested_tier"\n', encoding="utf-8")
    assert not [t for t in FORBIDDEN if any(t in tok for tok in code_tokens(ok))]


def test_the_only_quality_condition_is_the_human_label_corpus():
    toks = set(code_tokens(BUILDER))
    assert "human_good_locations" in toks, "the human-quality condition must be explicit"
    # ...and it is sheet C's, imported rather than restated.
    assert "build_rare_palette_sheet" in toks


def test_the_palette_is_a_pool_draw_not_a_head_proposal():
    """`rare_palette_draw.PaletteDrawer` is a POOL draw against a declared family target.
    Pinned so that swapping it for a head-ranked pick is a test failure rather than a silent
    substrate change."""
    toks = set(code_tokens(BUILDER))
    assert "PaletteDrawer" in toks
    assert "PaletteRanker" not in toks, "a ranker is a head; this sheet draws"


# --------------------------------------------------------------------------- #
# 2. The population filters.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeLoc:
    family: str = "mandelbrot"
    cx: str = "0.0"
    cy: str = "0.0"
    fw: str = "1e-3"
    c_re: str | None = None
    c_im: str | None = None
    maxiter: int = 4000
    params: dict = field(default_factory=dict)

    def key(self):
        return f"{self.family}|{self.cx}|{self.cy}|{self.fw}|{self.c_re}|{self.c_im}"


def loc_entry(i, *, partition="mandelbrot", score=4, cx=None):
    loc = FakeLoc(cx=repr(0.001 * i) if cx is None else repr(cx))
    return (f"L{i:04d}", {"loc": loc, "score": score, "partition": partition,
                          "batch_ids": {"b"}, "image_ids": [f"im{i}"],
                          "palette": f"pal{i:03d}", "hue_family": "green",
                          "color_params": {"reverse": False, "log_premap": "linear",
                                           "gamma": 1.0, "phase": 0.0, "n_cycles": 1,
                                           "transfer": "pct", "transfer_gamma": 1.0,
                                           "palette": f"pal{i:03d}", "palette_type": None,
                                           "palette_source": None,
                                           "interior_color": [0.0, 0.0, 0.0]}})


def test_prior_scan_globs_and_can_exclude_itself():
    """THE ALTERNATING BUG, pinned. Once sheet E has written its own `images.jsonl` it is one
    of the batches under `batches/`, so a prior-batch scan that does not skip it excludes its
    own population. What made sheet D's version vicious is that the empty run still rewrote
    `images.jsonl`, so the failure alternated instead of persisting."""
    with_self, _c, pairs_with, per_batch = BE.prior_mining_rows()
    without_self, _c2, pairs_without, per_batch2 = BE.prior_mining_rows(
        exclude_batch=SPEC.batch_id)
    assert per_batch, "no render-mode batch on disk — the exclusion would be vacuous"
    assert SPEC.batch_id not in per_batch2
    if SPEC.batch_id in per_batch:
        assert len(without_self) < len(with_self), "the exclusion is not actually skipping it"
        assert len(pairs_without) < len(pairs_with)
    # the three (27) batches are always there and always excluded
    assert "2026-08-06_render_mode_fresh_sheet_v1" in per_batch2


def test_prior_scan_is_a_glob_not_a_constant():
    """A hardcoded batch list is how batch four silently stops being excluded."""
    toks = set(code_tokens(BUILDER))
    assert "iterdir" in toks
    body = BUILDER.read_text(encoding="utf-8")
    for b in ("2026-08-06_render_mode_fresh_sheet_v1",
              "2026-08-10_render_mode_correction_v2",
              "2026-08-10_render_mode_rare_palette_v1"):
        assert b not in body, f"{b} is named in the builder — the exclusion must be globbed"


def test_pair_freshness_is_the_prompt_predicate_and_can_fail(monkeypatch):
    """Zero stale pairs is the property; the test proves the predicate can see a stale one."""
    fake = {("L0001", "curv_linear")}
    monkeypatch.setattr(BE, "prior_mining_rows",
                        lambda exclude_batch=None: (set(), {}, fake, {}))
    ents = [{"unit_key": "L0001|curv_linear", "location_key": "L0001", "mode": "curv_linear"},
            {"unit_key": "L0002|curv_linear", "location_key": "L0002", "mode": "curv_linear"},
            {"unit_key": "L0001|stripe", "location_key": "L0001", "mode": "stripe"}]
    rep = BE.pair_freshness(ents)
    assert rep["n_stale_pairs"] == 1 and rep["stale"] == ["L0001|curv_linear"]
    # a location may repeat under a NEW mode — that is what "pair" means
    assert "L0001|stripe" not in rep["stale"]


def test_pair_freshness_threads_the_self_exclusion():
    """A REPORTING copy of the self-exclusion bug is worse than the original: it looks like a
    finding. The helper must accept and forward the batch id."""
    import inspect
    assert "exclude_batch" in inspect.signature(BE.pair_freshness).parameters
    body = BUILDER.read_text(encoding="utf-8")
    assert "pair_freshness(candidates, exclude_batch=spec.batch_id)" in body


# --------------------------------------------------------------------------- #
# 3. The mode axis and the candidate draw.
# --------------------------------------------------------------------------- #
def test_exp_smoothing_is_excluded_and_the_roster_is_otherwise_whole():
    assert "exp_smoothing" in BE.EXCLUDED_MODES
    assert "exp_smoothing" not in BE.ACTIVE_MODES
    assert set(BE.ACTIVE_MODES) == set(MR.MODES) - set(BE.EXCLUDED_MODES)
    assert len(BE.ACTIVE_MODES) == 14


def test_the_four_contested_modes_are_the_arms_failing_cells():
    """The tuple is a claim about the committed arm reports; check it against them rather than
    against memory. Every mode named must actually carry a clause-(a) failure somewhere."""
    failing = set()
    seen = 0
    for arm in ("v3", "v3_aug", "v3_augx", "v3_uniform", "v3_ap2"):
        p = ROOT / "data" / "render_mode_head" / arm / "report.json"
        if not p.exists():
            continue
        seen += 1
        R = json.loads(p.read_text(encoding="utf-8"))
        for f in R["winner_rule"]["clause_a"]["failures"]:
            if f["arm"].startswith("mode:"):
                failing.add(f["arm"].split(":", 1)[1])
    if not seen:
        pytest.skip("no staged arm report on disk")
    missing = sorted(set(BE.CONTESTED_MODES) - failing)
    assert not missing, f"CONTESTED_MODES names modes no arm failed: {missing}"


def test_mode_targets_weight_the_contested_cells():
    supply = {m: 500 for m in BE.ACTIVE_MODES}
    take, rep = BE.mode_targets(SPEC, supply)
    for m in BE.CONTESTED_MODES:
        assert take[m] == SPEC.contested_per_mode
    assert "exp_smoothing" not in take
    assert sum(take.values()) == SPEC.target_rows
    others = [v for m, v in take.items() if m not in BE.CONTESTED_MODES]
    assert max(others) - min(others) <= 1, "the remainder must be balanced-or-drained"
    assert rep["rows_by_mode"] == dict(sorted(take.items()))


def test_mode_targets_drain_a_short_cell_onto_the_others():
    supply = {m: 500 for m in BE.ACTIVE_MODES}
    supply["curv_linear"] = 4
    supply["tia"] = 1
    take, _rep = BE.mode_targets(SPEC, supply)
    assert take["curv_linear"] == 4, "a contested cell cannot exceed its supply"
    assert take["tia"] == 1
    # THE FREED ROWS LAND ON THE OTHER CELLS RATHER THAN BEING LOST: the remainder is dealt
    # against the reduced contested total, so the sheet still comes out at its target.
    assert sum(take.values()) == SPEC.target_rows
    assert take["direct_trap_lines"] == SPEC.contested_per_mode


def test_candidate_draw_is_seeded_capped_and_score_free():
    locs = [loc_entry(i) for i in range(60)]
    targets = {m: (10 if m in BE.CONTESTED_MODES else 2) for m in BE.ACTIVE_MODES}
    ents, rep = BE.draw_pairs(SPEC, locs, targets, oversample=1.0)
    per_loc = {}
    for e in ents:
        per_loc[e["location_key"]] = per_loc.get(e["location_key"], 0) + 1
    assert max(per_loc.values()) <= SPEC.max_rows_per_location
    for m in BE.CONTESTED_MODES:
        assert rep["drawn_by_mode"][m] == 10
    # no (location, mode) pair appears twice
    assert len({e["unit_key"] for e in ents}) == len(ents)
    # deterministic...
    ents2, _ = BE.draw_pairs(SPEC, locs, targets, oversample=1.0)
    assert [e["unit_key"] for e in ents2] == [e["unit_key"] for e in ents]


def test_candidate_draw_does_not_reproduce_the_human_score_order():
    """A higher human label must not buy a seat once the pool is admitted: the sheet is
    conditioned on quality, not ordered by it."""
    locs = [loc_entry(i, score=4 if i >= 50 else 3) for i in range(60)]
    targets = {m: (0 if m != "curv_linear" else 10) for m in BE.ACTIVE_MODES}
    ents, _ = BE.draw_pairs(SPEC, locs, targets, oversample=1.0)
    picked = {e["location_key"] for e in ents}
    fours = {k for k, v in locs if v["score"] == 4}
    assert picked != fours, "the draw reproduced the human-4 set — it is ordering on the score"


def test_oversample_is_a_reserve_the_filters_spend():
    locs = [loc_entry(i) for i in range(120)]
    targets = {m: (10 if m in BE.CONTESTED_MODES else 2) for m in BE.ACTIVE_MODES}
    n1 = len(BE.draw_pairs(SPEC, locs, targets, oversample=1.0)[0])
    n2 = len(BE.draw_pairs(SPEC, locs, targets, oversample=2.0)[0])
    assert n2 > n1


def test_one_grid_cell_per_location_and_direct_mode():
    """Sheet B's self-dup class: 631 of its 688 same-location near-dup pairs were ONE direct
    mode at two cells of the sweep. Here a (location, mode) is one unit, so that cannot happen.

    Distinct direct modes at a location are no longer guaranteed distinct cells: the
    2026-08-11 coarsening put 4 direct modes over a 3-cell grid
    (`mining_roster.DIRECT_GRID`), so some pair must share. What is asserted instead is that
    the sharing is not SYSTEMATIC — no ordered pair of direct modes takes the same cell at
    every location, which is exactly what a fixed `DIRECT_MODES.index(mode) % n_cells` would
    produce and what the per-location mode permutation exists to prevent. Every cell must also
    be reached. The `if n_cells >= n_modes` clause keeps the strong all-distinct assertion
    alive for a grid that can still afford it."""
    locs = [loc_entry(i) for i in range(40)]
    targets = {m: (20 if MR.MODE_KIND.get(m) == "direct" else 0) for m in BE.ACTIVE_MODES}
    ents, _ = BE.draw_pairs(SPEC, locs, targets, oversample=1.0)
    by_pair = {}
    for e in ents:
        if e["kind"] != "direct":
            continue
        k = (e["location_key"], e["mode"])
        assert k not in by_pair, f"two sweep cells for {k}"
        by_pair[k] = (e["mode_params"]["direct_opacity"], e["mode_params"]["direct_threshold"])
    per_loc = {}
    for (k, m), cell in by_pair.items():
        per_loc.setdefault(k, {})[m] = cell
    multi = [v for v in per_loc.values() if len(v) > 1]
    assert multi, "no location carried two direct modes — the property is untested"
    n_cells, n_modes = len(MR.DIRECT_GRID), len(MR.DIRECT_MODES)

    assert {c for v in per_loc.values() for c in v.values()} == set(MR.DIRECT_GRID), \
        "the draw does not reach every grid cell"

    # no ordered pair of direct modes is cell-identical everywhere they co-occur.
    shared, seen_pair = Counter(), Counter()
    for v in multi:
        for a, b in itertools.combinations(sorted(v), 2):
            seen_pair[(a, b)] += 1
            shared[(a, b)] += v[a] == v[b]
    assert seen_pair, "no mode pair co-occurred — the property is untested"
    always = [p for p, n in seen_pair.items() if n >= 3 and shared[p] == n]
    assert not always, f"these mode pairs share a cell at EVERY location: {always}"

    # non-vacuity: a grid at least as large as the mode count can still afford all-distinct,
    # so the assertion above has not been weakened into one that cannot fail.
    if n_cells >= n_modes:
        assert all(len(set(v.values())) == len(v) for v in multi)


# --------------------------------------------------------------------------- #
# 4. Select — the two filters and the per-mode take.
# --------------------------------------------------------------------------- #
def _cand(i, mode, loc, order):
    return {"unit_key": f"{loc}|{mode}", "location_key": loc, "mode": mode,
            "kind": MR.kind_of(mode), "family": "mandelbrot", "partition": "mandelbrot",
            "human_score": 4, "hue_family": "green", "palette": f"pal{i}",
            "draw_order": order}


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=32).astype(np.float32)
    return v / np.linalg.norm(v)


def test_select_excludes_smooth_equivalent_and_unmeasured_rows():
    twin = _unit(0)
    cands = [_cand(0, "curv_linear", "A", 0), _cand(1, "stripe", "B", 1),
             _cand(2, "tia", "C", 2)]
    twins = [{"location_key": "A", "unit_key": "A|smooth"},
             {"location_key": "B", "unit_key": "B|smooth"}]
    emb = {"A|smooth": twin, "A|curv_linear": twin,          # cos 1.0 -> near_dup, excluded
           "B|smooth": twin, "B|stripe": _unit(1),           # distinct
           "C|tia": _unit(2)}                                # no twin -> unmeasured, dropped
    sel, rep = BE.select(SPEC, cands, twins, emb, {m: 5 for m in BE.ACTIVE_MODES})
    assert [r["unit_key"] for r in sel] == ["B|stripe"]
    # the two exclusions are counted APART — a failed twin render must not read as a slice
    # full of smooth-equivalent rows
    assert rep["smooth_equivalence"]["excluded_smooth_equivalent"] == 1
    assert rep["smooth_equivalence"]["unmeasured_dropped"] == 1


def test_near_dup_filter_keeps_the_EARLIER_DRAWN_row_not_a_better_one():
    """Sheet C broke these ties best-first by the mining score. There is no score here, so the
    survivor is the earlier-drawn row — a pure function of the population and the seed."""
    v = _unit(3)
    twin = _unit(9)
    cands = [_cand(0, "stripe", "A", 5), _cand(1, "tia", "B", 1)]   # B drawn FIRST
    twins = [{"location_key": "A", "unit_key": "A|smooth"},
             {"location_key": "B", "unit_key": "B|smooth"}]
    emb = {"A|smooth": twin, "B|smooth": twin, "A|stripe": v, "B|tia": v}   # identical
    sel, rep = BE.select(SPEC, cands, twins, emb, {m: 5 for m in BE.ACTIVE_MODES})
    assert [r["unit_key"] for r in sel] == ["B|tia"]
    assert rep["near_dup_filter"]["n_dropped"] == 1
    assert rep["near_dup_filter"]["dropped"][0]["dup_of"] == "B|tia"
    assert rep["near_dup_filter"]["cut"] == SE.STRICT_CUT


def test_select_takes_each_mode_up_to_its_target_and_reports_the_shortfall():
    twin = _unit(11)
    cands, twins, emb = [], [], {}
    for i in range(8):
        loc = f"L{i}"
        cands.append(_cand(i, "curv_linear", loc, i))
        twins.append({"location_key": loc, "unit_key": f"{loc}|smooth"})
        emb[f"{loc}|smooth"] = twin
        emb[f"{loc}|curv_linear"] = _unit(100 + i)
    targets = {m: 0 for m in BE.ACTIVE_MODES}
    targets["curv_linear"] = 3
    sel, rep = BE.select(SPEC, cands, twins, emb, targets)
    assert len(sel) == 3 and rep["drawn_by_mode"] == {"curv_linear": 3}
    assert rep["buckets"] == {"contested": 3}
    targets["curv_linear"] = 20
    _sel2, rep2 = BE.select(SPEC, cands, twins, emb, targets)
    assert rep2["short_of_target"] == {"curv_linear": 12}


# --------------------------------------------------------------------------- #
# 5. Blind serving + eval-only + registration.
# --------------------------------------------------------------------------- #
def test_ui_url_is_blind_and_honours_the_stamped_order():
    u = SPEC.ui_url
    assert "order=file" in u, "the builder's stamped shuffle must be what the page shows"
    assert "tiers=3" in u and f"batch={SPEC.batch_id}" in u
    assert "corpus=render_mode_corpus" in u


def test_declared_row_shape_is_blind():
    """`ROW_KEYS` is the row shape the writer declares and asserts against at write time, so
    this is the property itself rather than a proxy for it.

    `wallpaper_label.html` enters CORRECTION mode iff some row carries a numeric
    `suggested_tier`, and shows a machine readout iff a row carries `head_v2_pred`, `pred` or
    `p_ge3`. Those predicates are re-read out of the rig here, so if the rig grows a new way
    to display a machine opinion this test is where it surfaces."""
    assert BE.ROW_KEYS == ("image_id", "sheet_order", "render", "provenance", "label")
    for f in ("suggested_tier", "head_mining_v1", "head_v2_pred", "pred", "p_ge3"):
        assert f not in BE.ROW_KEYS, f"the row shape carries {f} — not blind"
    ui = (ROOT / "tools" / "viz" / "wallpaper_label.html").read_text(encoding="utf-8")
    assert "CORRECTION=rows.some(r=>typeof r.suggested_tier==='number')" in ui, \
        "the rig's correction predicate moved — re-check what makes sheet E blind"
    for probe in ("r.head_v2_pred", "r.pred", "r.p_ge3"):
        assert probe in ui, ("the rig's machine-readout fields moved — ROW_KEYS must still "
                             "exclude every one of them")


def test_write_refuses_an_undeclared_row_field():
    """The write-time assert, exercised: adding a field to a row must fail loudly."""
    row = {"image_id": "bmn0000_ab", "sheet_order": 0, "render": {}, "provenance": {},
           "label": {"score": None}, "suggested_tier": 3}
    extra = set(row) - set(BE.ROW_KEYS) - {"_unit_key", "_crop_stem"}
    assert extra == {"suggested_tier"}, "the write-time predicate no longer catches this"
    body = BUILDER.read_text(encoding="utf-8")
    assert "undeclared row field(s)" in body


def test_every_row_is_stamped_eval_and_the_batch_is_eval_only():
    body = BUILDER.read_text(encoding="utf-8")
    assert 'bool(done[rec["unit_key"]]["transfer_dropped"]), "eval")' in body
    assert '"eval_only": True' in body


def test_the_sheet_is_not_in_the_training_pool():
    """The eval-only stamp is a claim; this is the mechanism. `mining_corpus.load_corpus`
    unions `near_dup_groups.BATCHES`, and sheet E must not be in it — otherwise the next
    retrain trains on the instrument."""
    from tools.mining.near_dup_groups import BATCHES
    assert SPEC.batch_id not in BATCHES


def test_the_eval_only_pin_actually_sees_this_sheet():
    """`eval_only: true` is a claim; `tools/corpus/eval_only.py` is the mechanism, and it
    GLOBS the corpus tree — so sheet E is pinned by construction rather than by a second
    registration. Checked here because the stamp and the pin are two different ways to lose
    the property: a batch whose rows disagree with its own flag, and a split pass that
    overrode correct stamps."""
    from tools.corpus import eval_only as EO
    if not (SPEC.batch_dir / "batch.json").exists():
        pytest.skip("sheet E is not built in this tree")
    assert EO.is_eval_only("render_mode_corpus", SPEC.batch_id)
    blk = EO.eval_only_batches("render_mode_corpus")[SPEC.batch_id]
    assert blk.reason, "an eval_only batch with no note is a violation, not an empty reason"
    assert EO.assert_stamps("render_mode_corpus")["ok"]
    keys = EO.eval_only_ids("render_mode_corpus",
                            key_of=lambda r: r["provenance"]["location_key"])
    assert len(keys) > 0, "the render-mode split is keyed on location_key — it must pin some"


def test_batch_is_registered_and_classifies_train_on_the_location_side():
    assert br.is_registered(SPEC.batch_id)
    split, biased, source = br.assign_split(SPEC.batch_id, "mandelbrot")
    # biased w.r.t. the LOCATION axis (the draw reads human location labels), which keeps it
    # off the location-head eval side. Its mining-side eval role lives in its own batch record.
    assert (split, biased) == ("train", True)
    assert source == "mining_blind_eval"
    assert not br.score_unconditioned(SPEC.batch_id, "mandelbrot")


def test_spec_is_frozen_and_the_sheet_key_is_an_entry():
    """CLAUDE.md's "writing a builder for one instance": a second sheet must be an ENTRY."""
    with pytest.raises(Exception):
        SPEC.target_rows = 1                                          # type: ignore[misc]
    assert set(BE.SHEETS) >= {"e"}
    assert SPEC.labels_sidecar == f"labels/{SPEC.generator_version}.json"
    assert SPEC.labels_export != SPEC.labels_sidecar, (
        "the page's export and the merge's destination must be different files — pointing "
        "both at the sidecar merges the destination into itself")
    assert 140 <= SPEC.target_rows <= 170, "the prompt's declared size band"
    assert SPEC.contested_per_mode * len(BE.CONTESTED_MODES) < SPEC.target_rows, (
        "the remainder must leave room for a pooled read over the other modes")


# --------------------------------------------------------------------------- #
# The re-verdict harness, on a synthetic labeled slice.
# --------------------------------------------------------------------------- #
import tools.mining.sheet_e_reverdict as RV                          # noqa: E402


def _rows(n=150, rng=None):
    rng = rng or np.random.default_rng(0)
    modes = list(BE.CONTESTED_MODES) + ["tia", "stripe"]
    out = []
    for i in range(n):
        out.append(RV.ERow(image_id=f"bmn{i:04d}_deadbeef", label=int(rng.integers(1, 4)),
                           jpg=Path("nonexistent.jpg"), mode=modes[i % len(modes)],
                           kind=MR.kind_of(modes[i % len(modes)]), partition="mandelbrot",
                           family="mandelbrot", loc=f"L{i // 2}", hue_family="green"))
    return out


def _scores(rows, rng, strength):
    lb = np.array([r.label for r in rows], float)
    out = {}
    for k, thr in (("p_ge2", 2), ("p_ge3", 3)):
        y = (lb >= thr).astype(float)
        out[k] = np.clip(strength * y + (1 - strength) * rng.random(len(rows)), 1e-6, 1 - 1e-6)
    out["rank"] = (lb - 1) / 2.0
    return out


def _meta(rows):
    return {"batch_id": SPEC.batch_id, "sidecar": SPEC.labels_sidecar,
            "n_batch_rows": len(rows), "n_labeled": len(rows), "n_unlabeled": 0,
            "partial": False, "partial_note": None}


def _arm(rows, name, strength, seed, tmp_path, failures):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"arm": f"{name}_arm"}), encoding="utf-8")
    (d / "report.json").write_text(json.dumps({
        "eval_slice": {"n": 827},
        "winner_rule": {"clause_a": {"pass": False, "n_tests": 38, "failures": [
            {"arm": a, "metric": m, "median": -0.1, "lo": -0.2, "hi": -0.01}
            for a, m in failures]}}}), encoding="utf-8")
    return {"dir": d, "ckpt": f"data/render_mode_head/{name}/model_best.pt",
            "arm_name": f"{name}_arm", "dials": {},
            "scores": _scores(rows, np.random.default_rng(seed), strength)}


def test_reverdict_reports_a_surviving_and_a_dead_contested_cell(tmp_path):
    rows = _rows()
    base = _scores(rows, np.random.default_rng(1), 0.95)     # v1 strong -> the arm looks worse
    arms = {
        "strong_loss": _arm(rows, "strong_loss", 0.05, 2, tmp_path,
                            [("mode:curv_linear", "auc_ge2"), ("pooled", "auc_ge2")]),
        "no_loss": _arm(rows, "no_loss", 0.95, 3, tmp_path,
                        [("mode:direct_trap_ring", "auc_ge2")]),
    }
    R = RV.build(rows, base, arms, _meta(rows), draws=200, seed=3)

    a = R["arms"]["strong_loss"]
    assert a["clause_a"]["pass"] is False and a["clause_a"]["failures"]
    verdicts = {c["cell"]: c["verdict"] for c in a["contested_survival"]["cells"]}
    assert verdicts["pooled.auc_ge2"] == "SURVIVES"
    assert verdicts["mode:curv_linear.auc_ge2"] in ("SURVIVES", "NOT SIGNIFICANT (underpowered)")

    b = R["arms"]["no_loss"]
    dead = {c["cell"]: c["verdict"] for c in b["contested_survival"]["cells"]}
    assert dead["mode:direct_trap_ring.auc_ge2"] != "SURVIVES"

    # the cross-arm summary carries both
    assert "pooled.auc_ge2" in R["contested_summary"]["cells"]
    assert R["contested_summary"]["cells"]["mode:curv_linear.auc_ge2"]["contested_mode"] is True
    assert R["contested_summary"]["cells"]["pooled.auc_ge2"]["contested_mode"] is False
    assert "SHEET E re-verdict" in RV.md(R)


def test_reverdict_states_a_one_class_cell_instead_of_dropping_it(tmp_path):
    """A boundary with one class present votes NEITHER way, and the report says which class is
    missing. Treating it as 'not worse' is how a cell nobody could measure passes a head."""
    rows = [RV.ERow(image_id=f"bmn{i:04d}_x", label=2, jpg=Path("x.jpg"),
                    mode="curv_linear", kind="pure", partition="mandelbrot",
                    family="mandelbrot", loc=f"L{i}", hue_family="green") for i in range(40)]
    base = _scores(rows, np.random.default_rng(1), 0.5)
    arms = {"a": _arm(rows, "a", 0.6, 2, tmp_path, [("mode:curv_linear", "auc_ge3")])}
    R = RV.build(rows, base, arms, _meta(rows), draws=50, seed=3)
    blk = R["arms"]["a"]["slices"]["mode:curv_linear"]
    assert blk["cells"]["auc_ge3"]["measurable"] is False
    assert "no positives" in blk["cells"]["auc_ge3"]["why"]
    assert blk["cells"]["auc_ge2"]["measurable"] is False        # every label is 2 -> no negs
    assert "no negatives" in blk["cells"]["auc_ge2"]["why"]
    unm = {u["metric"] for u in R["arms"]["a"]["clause_a"]["unmeasurable"]}
    assert {"auc_ge2", "auc_ge3"} <= unm
    surv = {c["cell"]: c["verdict"] for c in R["arms"]["a"]["contested_survival"]["cells"]}
    assert surv["mode:curv_linear.auc_ge3"] == "UNMEASURABLE"
    md = RV.md(R)
    assert "vote neither way" in md and "no positives" in md


def test_reverdict_reports_unknown_when_an_arm_report_is_absent(tmp_path):
    rows = _rows(60)
    base = _scores(rows, np.random.default_rng(1), 0.5)
    arm = _arm(rows, "gone", 0.6, 2, tmp_path, [])
    (arm["dir"] / "report.json").unlink()
    R = RV.build(rows, base, {"gone": arm}, _meta(rows), draws=50, seed=3)
    af = R["arms"]["gone"]["anchored_failures"]
    assert af["cells"] is None and af["status"].startswith("UNKNOWN")
    assert R["arms"]["gone"]["contested_survival"]["cells"] == []
    assert "SHEET E re-verdict" in RV.md(R)


def test_anchoring_price_is_derived_from_the_committed_report():
    """The 0.953 is READ, never restated. If the (28) report is on disk the price is a number;
    if it is not, the report says UNKNOWN rather than dropping the row."""
    here = {"auc_ge2": 0.80, "auc_ge3": 0.70, "ap_ge2": 0.9, "ap_ge3": 0.5}
    p = RV.anchoring_price(here)
    if RV.ANCHORED_REPORT.exists():
        assert p["anchored_slice"]["status"] == "read"
        assert p["anchored_slice"]["v1_auc_ge2"] > 0.9     # the anchored pooled cell as committed
        assert p["delta_auc_ge2"] == pytest.approx(
            0.80 - p["anchored_slice"]["v1_auc_ge2"], abs=1e-12)
        assert "LOWER" in p["reading"]
    else:
        assert p["anchored_slice"]["status"].startswith("UNKNOWN")


def test_anchoring_price_reports_unknown_when_the_report_is_absent(monkeypatch):
    monkeypatch.setattr(RV, "ANCHORED_REPORT", ROOT / "scratch" / "no_such_report.json")
    p = RV.anchoring_price({"auc_ge2": 0.8, "auc_ge3": 0.8})
    assert p["anchored_slice"]["status"].startswith("UNKNOWN")
    assert "delta_auc_ge2" not in p


def test_reverdict_metric_set_and_voting_rule_match_the_28_harness():
    """A cell here must be the same cell there, or the two reports cannot be read together —
    and the contested-cell survival question is only meaningful if the voting rule is the one
    the failures were found under."""
    from tools.mining import mining_v3_reads as V3R
    assert [m.key for m in RV.METRICS] == [m.key for m in V3R.METRICS]
    assert RV.voting_metrics is V3R.voting_metrics
    assert [m.key for m in RV.voting_metrics("mode:tia")] == ["auc_ge3", "auc_ge2"]
    assert [m.key for m in RV.voting_metrics("pooled")] == [m.key for m in V3R.METRICS]


def test_reverdict_declares_five_arms_and_does_not_decide_clause_b():
    assert len(RV.ARMS) == 5 and "v3" in RV.ARMS and "v3_ap2" in RV.ARMS
    assert "v2" not in RV.ARMS, "v2 is a prior generation with its own harness"
    body = Path(RV.__file__).read_text(encoding="utf-8")
    assert "verdict(" not in body, ("winner_rule.verdict needs a motivating arm; calling it "
                                    "here would report a winner computed from an empty "
                                    "clause (b)")


def test_a_bounded_or_synthetic_report_stamps_itself_unusable(tmp_path):
    """The .md is what gets read. A report that only says `incomplete: true` in its JSON is a
    report that will be quoted as a verdict."""
    rows = _rows(60)
    base = _scores(rows, np.random.default_rng(1), 0.5)
    arms = {"a": _arm(rows, "a", 0.6, 2, tmp_path, [])}
    R = RV.build(rows, base, arms, _meta(rows), draws=50, seed=3)
    assert "INCOMPLETE" not in RV.md(R)
    R["incomplete"] = True
    assert "INCOMPLETE" in RV.md(R)
    R["incomplete"] = False
    R["DRY_RUN"] = "synthetic labels"
    assert "DRY RUN" in RV.md(R)


def test_reverdict_hard_stops_without_labels():
    if (ROOT / SPEC.labels_sidecar).exists():
        pytest.skip("sheet E is labeled — the no-label branch cannot be exercised in place")
    with pytest.raises(SystemExit) as e:
        RV.load_rows()
    assert "merge_sitting" in str(e.value) or "build the sheet first" in str(e.value)
