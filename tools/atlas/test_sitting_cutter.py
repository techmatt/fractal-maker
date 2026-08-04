"""The sitting cutter: three NON-OPTIONAL stages, each proved red by injection.

Every stage here exists because its absence cost a real sitting real keystrokes, so each is
tested twice — once that it fires, and once that it does NOT fire on the population it must
leave alone. A filter that removes everything passes the first test and fails the second.

  uv run pytest tools/atlas/test_sitting_cutter.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sitting_cutter as sc      # noqa: E402
import supply_routing as srt     # noqa: E402


def _row(**kw):
    base = dict(partition="julia:mandelbrot", rank_tier=2, rank_score=1.0,
                cx="0", cy="0", fw="1.0", fate="admitted", int_frac=0.1,
                canon_decoded=3)
    base.update(kw)
    return base


def _unit(i, d=8):
    v = np.zeros(d, dtype=np.float32)
    v[i % d] = 1.0
    return v


def _embed_by_key(mapping):
    return lambda r: mapping.get(r.get("cx"))


# =========================================================================== #
# (a) interior > 0.30 — auto-labelled, never presented
# =========================================================================== #
def test_interior_over_the_threshold_is_auto_labelled_and_removed():
    kept, removed, rep = sc.stage_interior([_row(cx="a", int_frac=0.31),
                                            _row(cx="b", int_frac=0.10)], {})
    assert [r["cx"] for r in kept] == ["b"]
    assert [r["cx"] for r in removed] == ["a"]
    al = removed[0]["auto_label"]
    assert al["score"] == 1 and al["rule_id"] == "interior_gt30_v1"
    assert al["labeler"].startswith("rule:")
    assert rep["disposition"].endswith("NEVER presented")


def test_the_interior_comparison_is_strict_so_exactly_030_is_shown():
    """The boundary side is invisible in a count and mirrors `present.rs`'s strict `<` on the
    other side of the same number. A `>=` here would silently delete a whole band."""
    kept, removed, _ = sc.stage_interior([_row(cx="lo", int_frac=0.2999),
                                          _row(cx="eq", int_frac=0.30),
                                          _row(cx="hi", int_frac=0.3001)], {})
    assert [r["cx"] for r in kept] == ["lo", "eq"]
    assert [r["cx"] for r in removed] == ["hi"]


def test_an_unmeasured_interior_is_kept_and_counted_apart():
    """An absent measure is not a high one — `apply_interior_rule.fires`'s own rule."""
    kept, removed, rep = sc.stage_interior([_row(cx="none", int_frac=None)], {})
    assert len(kept) == 1 and not removed and rep["unmeasured_kept"] == 1


def test_the_interior_rule_is_the_SAME_rule_the_label_store_applies():
    """Same id, same threshold, same comparison, imported rather than restated — a second
    literal 0.30 in this tree is how the two drift."""
    import apply_interior_rule as air
    assert sc.INTERIOR_RULE_ID == air.RULE_ID == "interior_gt30_v1"
    assert sc.INTERIOR_THRESHOLD == air.THRESHOLD == 0.30


# =========================================================================== #
# (c) per-partition machine-1 auto-discard
# =========================================================================== #
@pytest.mark.parametrize("part,discarded", [("multibrot3", True), ("multibrot4", True),
                                            ("multibrot5", True), ("phoenix", True),
                                            ("julia:mandelbrot", False),
                                            ("mandelbrot", False)])
def test_machine_1_discard_follows_the_measured_partition_table(part, discarded):
    """The measurement is partition-dependent and the pooled 68.9% is not a decision.
    julia:mandelbrot must survive: 16.5% of its machine-1s are >=3."""
    kept, removed, _ = sc.stage_machine_1([_row(partition=part, canon_decoded=1)], {})
    assert bool(removed) is discarded
    assert bool(kept) is (not discarded)


def test_a_machine_2_or_better_is_never_discarded():
    """The vacuity guard: a stage that discarded every native-multibrot row would pass the
    parametrize above."""
    for dec in (2, 3, 4):
        kept, removed, _ = sc.stage_machine_1(
            [_row(partition="multibrot4", canon_decoded=dec)], {})
        assert kept and not removed, dec


def test_a_cheap_only_row_has_no_machine_1_verdict_to_act_on():
    """A `rank_tier=1` score comes off a 384x216 ss1 render; every P(Matt=1 | decoded 1) rate
    was measured against the 640x360 ss2 canonical decode. Discarding on the cheap score
    would be the cap/geometry error, so a tier-1 row survives whatever its flag says."""
    kept, removed, rep = sc.stage_machine_1(
        [_row(partition="multibrot4", rank_tier=1, canon_decoded=None)], {})
    assert kept and not removed
    assert rep["no_canonical_verdict_kept"]["multibrot4"] == 1


def test_an_unmeasured_partition_fails_closed_to_keep():
    kept, removed, _ = sc.stage_machine_1(
        [_row(partition="julia:multibrot4", canon_decoded=1)], {})
    assert kept and not removed
    assert srt.MACHINE_1_DISCARD["julia:multibrot4"] is False


# =========================================================================== #
# (b) presentation-level morph dedup
# =========================================================================== #
def test_morph_dedup_keeps_one_row_per_look_best_first():
    e = {"a": _unit(0), "b": _unit(0) * 0.999 + _unit(1) * 0.02, "c": _unit(3)}
    rows = [_row(cx="a"), _row(cx="b"), _row(cx="c")]
    kept, removed, rep = sc.stage_morph_dedup(rows, dict(embed=_embed_by_key(e)))
    assert [r["cx"] for r in kept] == ["a", "c"]
    assert [r["cx"] for r in removed] == ["b"] and removed[0]["dup_cos"] >= 0.974
    assert rep["looks_kept"] == 2


def test_morph_dedup_is_first_wins_so_the_incoming_rank_is_the_policy():
    e = {"top": _unit(0), "dup": _unit(0)}
    kept, _r, _ = sc.stage_morph_dedup([_row(cx="top"), _row(cx="dup")],
                                       dict(embed=_embed_by_key(e)))
    assert [r["cx"] for r in kept] == ["top"]
    kept2, _r2, _ = sc.stage_morph_dedup([_row(cx="dup"), _row(cx="top")],
                                         dict(embed=_embed_by_key(e)))
    assert [r["cx"] for r in kept2] == ["dup"]


def test_distinct_looks_are_not_thinned():
    """The vacuity guard. A dedup that dropped everything after the first row would pass the
    test above."""
    e = {str(i): _unit(i) for i in range(6)}
    rows = [_row(cx=str(i)) for i in range(6)]
    kept, removed, _ = sc.stage_morph_dedup(rows, dict(embed=_embed_by_key(e)))
    assert len(kept) == 6 and not removed


def test_an_unembeddable_row_is_kept_and_counted_not_treated_as_a_duplicate():
    e = {"a": _unit(0)}
    kept, removed, rep = sc.stage_morph_dedup([_row(cx="a"), _row(cx="unreachable")],
                                              dict(embed=_embed_by_key(e)))
    assert len(kept) == 2 and not removed and rep["unembeddable_kept"] == 1


def test_a_raising_embedder_costs_the_dedup_verdict_not_the_row():
    def boom(r):
        raise RuntimeError("no field")
    kept, removed, rep = sc.stage_morph_dedup([_row(cx="a")], dict(embed=boom))
    assert len(kept) == 1 and not removed and rep["unembeddable_kept"] == 1


def test_a_missing_embedder_is_a_HARD_failure_never_a_silent_skip():
    """The dedup is not optional. A `ctx` with no embedder must raise, not pass everything
    through — a stage that degrades to a no-op is a stage that will be a no-op on the run
    that needed it."""
    with pytest.raises(ValueError, match="NOT optional"):
        sc.stage_morph_dedup([_row()], {})


def test_the_dedup_threshold_is_the_library_knee():
    assert sc.NEAR_DUP_COS == srt.NEAR_DUP_COS == 0.974


# =========================================================================== #
# the pipeline: non-optional, accounted, capped
# =========================================================================== #
def test_all_three_stages_are_in_the_pipeline_and_there_is_no_way_to_skip_one():
    names = [f.__name__ for f in sc.STAGES]
    assert names == ["stage_interior", "stage_machine_1", "stage_morph_dedup"]
    import inspect
    src = inspect.getsource(sc.cut_sitting)
    assert "for fn in STAGES:" in src
    # no conditional guards the loop body — every stage runs on every cut
    assert "if " not in src.split("for fn in STAGES:")[1].split("sitting, cells")[0]


def test_the_expensive_stage_runs_last():
    """(a) and (c) are free column reads; (b) needs a render and a CLIP pass per row.
    Reversing them would be correct and would pay a morph field for every row the other two
    were about to delete."""
    assert sc.STAGES[-1] is sc.stage_morph_dedup


def test_the_cut_accounts_for_every_row_it_was_given():
    e = {str(i): _unit(i) for i in range(20)}
    rows = ([_row(cx=str(i), partition="multibrot4", canon_decoded=1) for i in range(3)]
            + [_row(cx=str(i), int_frac=0.9) for i in range(3, 6)]
            + [_row(cx=str(i)) for i in range(6, 20)])
    res = sc.cut_sitting(rows, max_rows=5, embed=_embed_by_key(e))
    rep = res["report"]
    assert rep["n_in"] == 20 and rep["n_sitting"] == 5
    removed = sum(len(v) for v in res["removed"].values())
    assert rep["n_in"] == rep["n_sitting"] + removed + rep["n_over_cap"]
    # Each stage removed the population it owns, THROUGH the pipeline — not merely when
    # called directly. Without this, a stage silently dropped from `STAGES` still passes
    # every one of its own unit tests.
    assert len(res["auto_labeled"]) == 3                       # interior
    assert len(res["removed"]["machine_1_discard"]) == 3       # native multibrot machine-1s
    assert set(sc.STAGES) == {sc.stage_interior, sc.stage_machine_1, sc.stage_morph_dedup}


def test_a_cut_that_lost_a_row_would_exit_loud(monkeypatch):
    """The accounting identity is an assertion, not a report line. Proved by injecting a
    stage that eats a row without naming it."""
    def leaky(rows, ctx):
        return rows[:-1], [], dict(stage="leaky", removed=0)
    monkeypatch.setattr(sc, "STAGES", (leaky,))
    with pytest.raises(AssertionError, match="does not balance"):
        sc.cut_sitting([_row(cx="a"), _row(cx="b")], max_rows=10)


def test_the_sitting_is_capped_at_one_page():
    assert sc.MAX_ROWS == 1000
    e = {str(i): _unit(i, d=40) for i in range(40)}
    rows = [_row(cx=str(i)) for i in range(40)]
    res = sc.cut_sitting(rows, max_rows=7, embed=_embed_by_key(e))
    assert res["report"]["n_sitting"] == 7 and res["report"]["n_over_cap"] == 33


def test_the_cut_balances_across_partition_and_tier_cells():
    """One page, so a cell with hundreds of rows must not own it."""
    e = {str(i): _unit(i, d=64) for i in range(60)}
    rows = ([_row(cx=str(i), partition="julia:mandelbrot") for i in range(50)]
            + [_row(cx=str(i), partition="phoenix") for i in range(50, 60)])
    res = sc.cut_sitting(rows, max_rows=10, embed=_embed_by_key(e))
    got = res["report"]["by_partition"]
    assert got == {"julia:mandelbrot": 5, "phoenix": 5}


# =========================================================================== #
# serving: the batch id, the registration, and the bar-readability slice
#
# (The v2 "the CLI never serves a sitting" pin is gone on purpose: it was a statement about
# the prompt that built the cutter, and it was superseded the moment a sitting was served.
# What replaces it is a pin on the decisions that outlive that — which batch, registered
# where, and served through what.)
# =========================================================================== #
def test_the_sitting_batch_id_is_the_same_string_in_all_three_places():
    """The id is declared in three modules that cannot import each other cheaply: the cutter
    (which writes the batch), `build_manifest` (which classifies it) and the sheet SPECS
    (which serves it). A typo in any one of them fails SILENTLY in the worst direction —
    `assign_split` falls closed to `unregistered`, which still returns train/biased, so the
    batch builds, looks right, and records that nobody classified it."""
    from tools.v7 import build_manifest as bm
    sys.path.insert(0, str(ROOT / "tools" / "corpus"))
    import build_combined_label_sheet as bcs
    assert sc.SITTING_BATCH in bm.V2_SITTING_BATCHES
    assert bcs.V2_SITTING.sources == (sc.SITTING_BATCH,)


def test_the_sitting_batch_is_registered_explicitly_train_side_and_biased():
    from tools.v7 import build_manifest as bm
    split, biased, source = bm.assign_split({"batch": sc.SITTING_BATCH, "ft": "mandelbrot"})
    assert source != "unregistered", "the fail-closed default is not a registration"
    assert (split, biased) == ("train", True)
    assert not bm.registration_contradictions(
        [{"batch": sc.SITTING_BATCH, "biased": biased}])


def test_draw_refuses_an_unregistered_batch(monkeypatch):
    """The fail-closed default is SAFE (train/biased) but it is not a decision, and a sitting
    built under it records that nobody classified it. So `draw` aborts rather than proceeds."""
    import types
    monkeypatch.setattr(sc, "SITTING_BATCH", "2099-01-01_never_registered")
    with pytest.raises(SystemExit) as e:
        sc.stage_draw(types.SimpleNamespace(run_dir="/nonexistent", max_rows=10))
    assert "NOT registered" in str(e.value)


@pytest.mark.parametrize("prov,ok", [
    ({"fit_model": "view_fit_v1.1", "fit_score": 1.0, "composite": 2.0}, True),
    ({"fit_model": "view_fit_v1.1", "fit_score": 0.0, "composite": 0.0}, True),   # 0 is a score
    ({"fit_model": "view_fit_v1.1", "fit_score": 1.0, "composite": None}, False),
    ({"fit_model": "view_fit_v1.1", "fit_score": None, "composite": 2.0}, False),
    ({"fit_model": None, "fit_score": 1.0, "composite": 2.0}, False),
    ({"fit_model": "view_fit_v1.0", "fit_score": 1.0, "composite": 2.0}, False),
    ({}, False),
])
def test_bar_readability_needs_both_scores_and_the_right_model(prov, ok):
    """BOTH, and from the pre-registered model. A row with one of the two cannot contribute to
    a delta-AP between them, and a zero is a score — the `is not None` is what keeps a legit
    0.0 in the slice, which a truthiness test would silently drop."""
    assert sc.is_bar_readable(prov) is ok


def test_the_screen_columns_ride_on_EXISTING_provenance_keys():
    """Nothing is renamed, which is what lets a v2-screened row pool with a supply-crawl or
    label-seeded one — same view frame, same composite_v3, same terms."""
    sys.path.insert(0, str(ROOT / "tools" / "corpus"))
    import corpus_common as cc
    unknown = [k for k in sc.SCREEN_PROV if k not in cc.PROVENANCE_KEYS]
    assert not unknown, f"{unknown} are not modeled provenance keys"
    assert sc.SCREEN_PROV["fit_score"] == "view_fit"
    assert sc.SCREEN_PROV["composite"] == "composite"


def test_render_writes_through_a_partial_and_renames():
    """A kill mid-render must not leave a truncated jpg — which reads as rendered forever, and
    is the one failure an idempotent skip-if-exists resume cannot recover from."""
    import inspect
    src = inspect.getsource(sc._render_one)
    assert ".part.jpg" in src and "os.replace" in src
    assert "if out.exists():" in src and "continue" in src, "resume must skip finished crops"


def test_the_render_partial_still_ends_in_an_extension_THE_ENGINE_CAN_WRITE():
    """The engine picks the image format off the output EXTENSION. `<id>.jpg.tmp` is not slow
    or lossy, it is `The file extension ."tmp" was not recognized as an image format` on every
    single render — a 100% failure rate that reads as a broken renderer, and it cost 50 renders
    to find. Asserted on the built name, not on the source text, so any future partial scheme
    has to satisfy it too."""
    from pathlib import Path as P
    out = P("x/vs0000_deadbeef.jpg")
    tmp = out.with_name(f"{out.stem}.part.jpg")
    assert tmp.suffix == ".jpg", tmp
    assert tmp != out and tmp.parent == out.parent
    # ...and a partial can never be mistaken for the finished crop by an exact-name reader
    assert tmp.name != out.name


def test_the_dedup_embeds_the_FRAME_THE_CROP_RENDERS():
    """The presentation dedup and the crop must be looking at the same picture.

    A row can carry a reframed `outcome_*` viewport beside its own `cx/cy/fw`; the render
    block uses the latter, so the morph embed must too. Measured on the harvest-v2 population:
    70 rows carried `outcome_*` and 49 were a genuinely different frame, so a dedup reading
    `outcome_*` thinned 1.4% of the sitting on a picture nobody would ever be shown. Asserted
    against `_render_block` itself rather than against a literal — that is the module the
    frame has to agree with."""
    sys.path.insert(0, str(ROOT / "tools" / "sourcing"))
    import build_q4_harvest_batches as bq
    r = dict(partition="julia:mandelbrot", cx="0.25", cy="-0.5", fw="0.125",
             outcome_cx="0.9", outcome_cy="0.9", outcome_fw="0.001",
             julia_c_re="0.3", julia_c_im="0.5", _palette="magma")
    led = sc._ledger_row(r)
    rb = bq._render_block(dict(r))
    assert (led["outcome_cx"], led["outcome_cy"], str(led["outcome_fw"])) == \
           (rb["cx"], rb["cy"], str(float(rb["fw"]))), \
        "the embedded frame and the rendered frame diverged"
    assert led["outcome_cx"] == "0.25", "the reframed outcome_* viewport must NOT win"


def test_the_ss_deviation_is_local_and_recorded_not_a_shared_constant_edit():
    """This sitting renders at ss2 where the corpus renders at ss4. Two things must hold, and
    the second is the one that matters later: the deviation is LOCAL (the shared
    `build_minibrot_batch.CROP_SS` is untouched, so a batch that says nothing still gets the
    corpus default), and it is RECORDED in the version-invariant render block, so a crop is
    still a pure function of its own row rather than of what someone chose that day."""
    sys.path.insert(0, str(ROOT / "tools" / "sourcing"))
    import build_minibrot_batch as BMB
    import build_q4_harvest_batches as bq
    assert BMB.CROP_SS == 4 and bq.CROP_SS == 4, "the corpus default must not be edited"
    assert sc.SITTING_CROP_SS != BMB.CROP_SS
    import corpus_common as cc
    assert "ss" in cc.RENDER_KEYS, "the deviation is only safe because ss is version-invariant"
