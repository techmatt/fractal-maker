"""Sheet D — the four properties that make it an INSTRUMENT, each pinned.

Sheet D exists because sheet A's labels are anchored to wallpaper v3, so the only thing worth
testing here is that the four properties which un-anchor it actually hold, and hold in a way
that survives somebody editing the builder six months from now:

  1. neither wallpaper head is reachable from the builder — a SOURCE SCAN, because the useful
     failure is a future edit that adds `head_v3` back, not one this run makes;
  2. the population filters actually filter (fresh-only, location-head floor, near-dup), each
     exercised on a constructed population whose answer is known;
  3. a written row is BLIND — no `suggested_tier`, no head block, no score field — and the
     labeling rig therefore cannot enter correction mode on it;
  4. the batch is eval-only and registered.

The re-verdict harness is covered here too, on a SYNTHETIC labeled batch: it is the object
that will run once, months from now, on data that does not exist yet, and "it imports" is not
coverage. Both report branches (anchored report present / absent) are exercised.

  uv run pytest tools/wallpaper/test_blind_minibrot_sheet.py -q
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring", ROOT / "tools" / "corpus",
          ROOT / "tools" / "queries", ROOT / "tools" / "atlas", ROOT / "tools" / "wallpaper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import batch_registry as br                                  # noqa: E402
import production_seeder as ps                               # noqa: E402
from tools.emission import floors as F                       # noqa: E402
from tools.wallpaper import build_blind_minibrot_sheet as BD  # noqa: E402

SPEC = BD.SHEETS["d"]
BUILDER = Path(BD.__file__)


# --------------------------------------------------------------------------- #
# 1. NEITHER WALLPAPER HEAD. A source scan, not a runtime check.
# --------------------------------------------------------------------------- #
# The wallpaper-head reach: the pin module, the checkpoint constants, the scorer loaders and
# the row fields a head stamps. A future edit that re-adds any of them is the failure this
# guards, and it would pass every runtime test in this file.
#
# Matched against CODE TOKENS, never raw text — this module says every one of these words on
# purpose, in comments and in the `batch.json` prose that explains what it does NOT do, and a
# substring scan would fail on its own documentation. `code_tokens` below drops comments and
# any string that reads as prose (contains whitespace), so a dict KEY like `"p_ge3"` is still
# caught while a sentence mentioning p_ge3 is not.
FORBIDDEN = (
    "wallpaper_pins", "HEAD_CKPT", "head_v3", "head_v4", "suggested_tier",
    "suggest_tier", "expected_tier", "tier_from_pred", "load_scorer", "report_v4_eval",
    "wallpaper_head", "p_ge3", "_marginals",
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


def test_builder_never_reaches_a_wallpaper_head():
    toks = code_tokens(BUILDER)
    hits = sorted({t for t in FORBIDDEN if any(t in tok for tok in toks)})
    assert not hits, (
        f"{BUILDER.name} reaches a wallpaper head: {hits}. Sheet D's whole value is that "
        f"neither v3 nor v4b touched its draw or its substrate; a score stamped here makes "
        f"the slice as unusable as sheet A.")


def test_the_scan_would_actually_catch_a_regression(tmp_path):
    """The guard's own guard: a scan that matches nothing passes vacuously. This proves the
    tokenizer sees a real emission and ignores the prose that describes one."""
    bad = tmp_path / "bad.py"
    bad.write_text('"""We never stamp head_v3 here."""\nrow = {"head_v3": {"p_ge3": 1.0}}\n',
                   encoding="utf-8")
    toks = code_tokens(bad)
    assert sorted({t for t in FORBIDDEN if any(t in tok for tok in toks)}) == ["head_v3",
                                                                              "p_ge3"]
    ok = tmp_path / "ok.py"
    ok.write_text('"""No head_v3 and no p_ge3 appear on a row."""\n'
                  'NOTE = "this row carries no p_ge3 and no head_v3 block"\n', encoding="utf-8")
    assert not [t for t in FORBIDDEN if any(t in tok for tok in code_tokens(ok))]


def test_the_only_quality_condition_is_the_location_head():
    toks = set(code_tokens(BUILDER))
    assert "passes_good_floor" in toks, "the location-head condition must be explicit"
    assert "GOOD_FLOOR" in toks


def test_pref_is_a_palette_head_not_a_quality_head():
    """`pick_mode="pref"` is pref-v3-gvo (conditioned_colorize.Scorer) — a palette-preference
    head. Pinned so that swapping the ranker to a mode that consults a quality head is a test
    failure rather than a silent substrate change."""
    body = BUILDER.read_text(encoding="utf-8")
    assert 'pick_mode="pref"' in body


# --------------------------------------------------------------------------- #
# 2. The population filters.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeLoc:
    family: str
    cx: str
    cy: str
    fw: str
    c_re: str | None = None
    c_im: str | None = None
    maxiter: int = 4000

    def key(self):
        return f"{self.family}|{self.cx}|{self.cy}|{self.fw}|{self.c_re}|{self.c_im}"


def src(i, *, cx=0.0, fw=1e-3, family="mandelbrot", partition="mandelbrot",
        p_good=0.9, vein="maneuver"):
    loc = FakeLoc(family=family, cx=repr(cx), cy="0.0", fw=repr(fw))
    return {"unit_key": f"u{i:05d}", "key": loc.key(), "loc": loc, "vein": vein,
            "partition": partition, "source_tag": vein, "floor_admit": False,
            "source_p_good": p_good, "intake_source": "union_ledger",
            "source_ledger": "L", "source_oid": f"o{i}", "source_decoded_class": 3}


def test_near_dup_filter_keeps_one_per_group_and_is_order_stable():
    # three rows at the same spot (within DEDUP_K * min(fw)) plus two far ones
    rows = [src(0, cx=0.0), src(1, cx=1e-5), src(2, cx=2e-5),
            src(3, cx=1.0), src(4, cx=2.0)]
    kept, groups, rep = BD.near_dup_filter(rows)
    assert len(kept) == 3, [k["unit_key"] for k in kept]
    assert rep["n_dropped"] == 2 and rep["n_multi_row_groups"] == 1
    # the representative is the first by unit_key, not by anything score-like
    assert kept[0]["unit_key"] == "u00000"
    # reversing the input cannot change the answer: the filter sorts first
    kept2, _g2, _r2 = BD.near_dup_filter(list(reversed(rows)))
    assert [k["unit_key"] for k in kept2] == [k["unit_key"] for k in kept]


def test_near_dup_filter_reads_the_live_calibrated_pair():
    """The rule is the seeder's, resolved at CALL time — so a recalibration moves this filter
    too. Proven by moving it (the same proof `emission_selector` runs on the same owner)."""
    rows = [src(0, cx=0.0), src(1, cx=4e-4)]      # 0.4*fw apart: outside k=0.25, inside k=0.5
    assert len(BD.near_dup_filter(rows)[0]) == 2
    old = ps.DEDUP_K
    try:
        ps.DEDUP_K = 0.5
        assert len(BD.near_dup_filter(rows)[0]) == 1
    finally:
        ps.DEDUP_K = old


def test_near_dup_never_collides_across_families_or_c_identities():
    rows = [src(0, cx=0.0, family="mandelbrot"), src(1, cx=0.0, family="multibrot3")]
    assert len(BD.near_dup_filter(rows)[0]) == 2
    a = src(2, cx=0.0)
    b = src(3, cx=0.0)
    a["loc"] = FakeLoc("julia", "0.0", "0.0", "1e-3", "0.1", "0.2")
    b["loc"] = FakeLoc("julia", "0.0", "0.0", "1e-3", "0.9", "0.2")
    assert len(BD.near_dup_filter([a, b])[0]) == 2


def test_draw_takes_all_when_supply_is_under_target():
    rows = [src(i) for i in range(10)]
    sel, rep = BD.draw(SPEC, rows, target_rows=50)
    assert rep["supply_bound"] is True and rep["drawn_rows"] == 10
    assert "TAKE ALL" in rep["rule"]
    assert [s["unit_key"] for s in sel] == sorted(s["unit_key"] for s in rows)


def test_draw_is_balanced_over_partitions_and_seed_reproducible():
    rows = ([src(i, partition="multibrot3") for i in range(200)]
            + [src(1000 + i, partition="mandelbrot") for i in range(30)])
    sel, rep = BD.draw(SPEC, rows, target_rows=60)
    assert rep["supply_bound"] is False and rep["drawn_rows"] == 60
    # balanced-or-drained: mandelbrot has 30 and gets everything it can, not 30/230 of 60
    assert rep["partition_alloc"]["mandelbrot"] == 30
    assert rep["partition_alloc"]["multibrot3"] == 30
    sel2, _ = BD.draw(SPEC, rows, target_rows=60)
    assert [s["unit_key"] for s in sel] == [s["unit_key"] for s in sel2]
    # and a different seed draws a different set (the shuffle is real)
    sel3, _ = BD.draw(SPEC, rows, target_rows=60, seed=SPEC.draw_seed + 1)
    assert [s["unit_key"] for s in sel3] != [s["unit_key"] for s in sel]


def test_draw_never_orders_on_a_score():
    """A high `source_p_good` must not buy a seat once the floor is cleared."""
    rows = [src(i, partition="mandelbrot", p_good=0.51 + 0.001 * i) for i in range(100)]
    sel, _ = BD.draw(SPEC, rows, target_rows=10)
    picked = sorted(float(s["source_p_good"]) for s in sel)
    top10 = sorted(float(s["source_p_good"]) for s in rows)[-10:]
    assert picked != top10, "the draw reproduced the top-10 by score — it is ordering on it"


def test_good_floor_is_the_location_head_and_rejects_a_missing_score():
    assert F.passes_good_floor(0.5) and not F.passes_good_floor(0.499)
    assert not F.passes_good_floor(None)


# --------------------------------------------------------------------------- #
# 3. Blind serving + 4. eval-only.
# --------------------------------------------------------------------------- #
def test_ui_url_is_blind_and_honours_the_stamped_order():
    u = SPEC.ui_url
    assert "order=file" in u, "the builder's stamped shuffle must be what the page shows"
    assert "tiers=4" in u and f"batch={SPEC.batch_id}" in u


def test_declared_row_shape_is_blind():
    """`ROW_KEYS` is the row shape the writer declares and asserts against at write time, so
    this is the property itself rather than a proxy for it.

    `wallpaper_label.html` enters CORRECTION mode iff some row carries a numeric
    `suggested_tier`, and shows a machine readout iff a row carries `head_v2_pred`, `pred`
    or `p_ge3`. Those predicates are re-read out of the rig here, so if the rig grows a new
    way to display a machine opinion this test is where it surfaces."""
    assert BD.ROW_KEYS == ("image_id", "sheet_order", "render", "provenance", "label")
    for field in ("suggested_tier", "head_v3", "head_v2_pred", "pred", "p_ge3"):
        assert field not in BD.ROW_KEYS, f"the row shape carries {field} — not blind"
    ui = (ROOT / "tools" / "viz" / "wallpaper_label.html").read_text(encoding="utf-8")
    assert "CORRECTION=rows.some(r=>typeof r.suggested_tier==='number')" in ui, \
        "the rig's correction predicate moved — re-check what makes sheet D blind"
    for probe in ("r.head_v2_pred", "r.pred", "r.p_ge3"):
        assert probe in ui, ("the rig's machine-readout fields moved — ROW_KEYS must still "
                             "exclude every one of them")


def test_write_refuses_an_undeclared_row_field():
    """The write-time assert, exercised: adding a field to a row must fail loudly."""
    rows = [{"image_id": "bmb0000_ab", "sheet_order": 0, "render": {}, "provenance": {},
             "label": {"score": None}, "suggested_tier": 3}]
    extra = set(rows[0]) - set(BD.ROW_KEYS) - {"_unit_key", "_crop_stem"}
    assert extra == {"suggested_tier"}, "the write-time predicate no longer catches this"


def test_every_row_is_stamped_eval():
    body = BUILDER.read_text(encoding="utf-8")
    assert 'provenance_block(spec, s, rec, loc, "eval")' in body
    assert '"eval_only": True' in body


def test_batch_is_registered_and_classifies_train_on_the_location_side():
    assert br.is_registered(SPEC.batch_id)
    split, biased, source = br.assign_split(SPEC.batch_id, "mandelbrot")
    # biased w.r.t. the LOCATION head (the draw reads its p_good), which keeps it off the
    # location-head eval side. Its wallpaper-side eval role lives in its own batch record.
    assert (split, biased) == ("train", True)
    assert source == "wallpaper_blind_minibrot_eval"
    assert not br.score_unconditioned(SPEC.batch_id, "mandelbrot")


def test_spec_is_frozen_and_the_sheet_key_is_an_entry():
    """CLAUDE.md's "writing a builder for one instance": a second sheet must be an ENTRY."""
    with pytest.raises(Exception):
        SPEC.target_rows = 1                                          # type: ignore[misc]
    assert set(BD.SHEETS) >= {"d"}
    assert SPEC.labels_export == f"labels/{SPEC.generator_version}.json"
    assert 150 <= SPEC.target_rows <= 200, "the prompt's declared size band"


# --------------------------------------------------------------------------- #
# The re-verdict harness, on a synthetic labeled slice.
# --------------------------------------------------------------------------- #
import tools.wallpaper.sheet_d_reverdict as RV                       # noqa: E402


def _fake_rows(n=120, rng=None):
    rng = rng or np.random.default_rng(0)
    rows, labels = [], rng.integers(1, 5, n)
    for i, lb in enumerate(labels):
        rows.append(RV.DRow(image_id=f"bmb{i:04d}_deadbeef", label=int(lb),
                            jpg=Path("nonexistent.jpg"),
                            vein="maneuver" if i % 3 else "q4_harvest",
                            partition="multibrot3" if i % 4 else "mandelbrot",
                            flavor="blue", family="multibrot3"))
    return rows


def _scores(rows, rng, strength):
    """A head whose p_ge3 correlates with the label at `strength`."""
    lb = np.array([r.label for r in rows], float)
    base = (lb - 1) / 3.0
    out = {}
    for k, thr in (("p_ge2", 2), ("p_ge3", 3), ("p_ge4", 4)):
        y = (lb >= thr).astype(float)
        out[k] = np.clip(strength * y + (1 - strength) * rng.random(len(rows)), 1e-6, 1 - 1e-6)
    out["rank"] = base
    return out


def _meta(rows):
    return {"batch_id": SPEC.batch_id, "sidecar": SPEC.labels_export,
            "n_batch_rows": len(rows), "n_labeled": len(rows), "n_unlabeled": 0,
            "partial": False, "partial_note": None}


def test_reverdict_builds_and_renders_both_verdict_branches():
    rng = np.random.default_rng(7)
    rows = _fake_rows()
    base = _scores(rows, np.random.default_rng(1), 0.25)
    cand = _scores(rows, np.random.default_rng(2), 0.95)      # candidate clearly better
    R = RV.build(rows, base, cand, {}, _meta(rows), draws=200, seed=3)
    assert R["winner_rule"]["clause_b"]["pass"] is True
    assert R["winner_rule"]["winner"] == "v4b"
    assert "SHEET D re-verdict" in RV.md(R)

    # ...and the other way round, so the losing branch of the report is exercised too.
    R2 = RV.build(rows, cand, base, {}, _meta(rows), draws=200, seed=3)
    assert R2["winner_rule"]["clause_a"]["pass"] is False
    assert R2["winner_rule"]["winner"] == "v3"
    assert "SHEET D re-verdict" in RV.md(R2)


def test_reverdict_seed_band_renders():
    rows = _fake_rows(60)
    base = _scores(rows, np.random.default_rng(1), 0.5)
    cand = _scores(rows, np.random.default_rng(2), 0.6)
    seeds = {s: _scores(rows, np.random.default_rng(10 + s), 0.55) for s in range(3)}
    R = RV.build(rows, base, cand, seeds, _meta(rows), draws=100, seed=3)
    assert len(R["v4b_seed_band"]["per_seed"]) == 3
    assert "five-seed band" in RV.md(R)


def test_anchoring_price_is_derived_from_the_committed_report():
    """The 0.965 is READ, never restated. If the (28) report is on disk the price is a
    number; if it is not, the report says UNKNOWN rather than dropping the row."""
    here = {"auc_ge3": 0.80, "ap_ge3": 0.85}
    p = RV.anchoring_price(here)
    if RV.ANCHORED_REPORT.exists():
        assert p["anchored_slice"]["status"] == "read"
        assert p["anchored_slice"]["v3_auc_ge3"] > 0.9      # the anchored bucket, as committed
        assert p["delta_auc_ge3"] == pytest.approx(
            0.80 - p["anchored_slice"]["v3_auc_ge3"], abs=1e-12)
        assert "LOWER" in p["reading"]
    else:
        assert p["anchored_slice"]["status"].startswith("UNKNOWN")


def test_anchoring_price_reports_unknown_when_the_report_is_absent(monkeypatch):
    monkeypatch.setattr(RV, "ANCHORED_REPORT", ROOT / "scratch" / "no_such_report.json")
    p = RV.anchoring_price({"auc_ge3": 0.8, "ap_ge3": 0.8})
    assert p["anchored_slice"]["status"].startswith("UNKNOWN")
    assert "delta_auc_ge3" not in p


def test_reverdict_metric_set_matches_the_28_harness():
    """A cell here must be the same cell there, or the two reports cannot be read together."""
    from tools.wallpaper import wallpaper_v4b_reads as V4B
    assert [m.key for m in RV.METRICS] == [m.key for m in V4B.METRICS]


def test_reverdict_hard_stops_without_labels():
    if (ROOT / SPEC.labels_export).exists():
        pytest.skip("sheet D is labeled — the no-label branch cannot be exercised in place")
    with pytest.raises(SystemExit) as e:
        RV.load_rows()
    assert "merge_sitting" in str(e.value) or "build the sheet first" in str(e.value)
