#!/usr/bin/env python
"""Tests for the supply crawl's two post-run pieces: the exemplar-similarity feature and the
label-batch draws.

Neither of these touches the engine or the backbone here. What is asserted is the part that
would fail SILENTLY: a substrate that crops away a third of the frame, an exemplar set that
quietly fits itself to the answer, a "uniform" draw whose exclusions were chosen by a score,
and a batch classification that came from the fail-closed default rather than from anyone
deciding anything.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "orbital"),
           str(ROOT / "tools" / "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import field_metrics as fm                       # noqa: E402
import exemplar_similarity as ex                 # noqa: E402
import build_supply_crawl_batches as bsc         # noqa: E402
from tools.v7 import build_manifest as bm        # noqa: E402


# =========================================================================== #
# the substrate
# =========================================================================== #
def test_non_escaping_pixels_are_black_not_a_palette_colour():
    """An interior that took the colormap's zero would read to the embedder as structure,
    and a black minibrot is the single most common thing in this population."""
    f = np.full((36, 64), np.nan, dtype=np.float32)
    rgb = ex.field_to_rgb(f)
    assert rgb.shape == (36, 64, 3) and rgb.dtype == np.uint8
    assert rgb.max() == 0
    # ...and the colormap's own value at t=0 is NOT black, so this is a real distinction.
    lit = ex.field_to_rgb(np.zeros((36, 64), dtype=np.float32))
    assert lit.max() > 0


def test_the_colour_coordinate_is_the_render_paths_cycle_and_is_phase_free():
    """One cycle is 1/DENSITY = 40 iterations, the render's own unit, so the bands this
    shows are the bands the render shows. Cyclic rather than percentile-stretched: a stretch
    is relative to the frame's own value range, which would make two frames with identical
    structure at different depths embed differently for a reason about depth."""
    base = np.linspace(0.0, 200.0, 36 * 64, dtype=np.float32).reshape(36, 64)
    shifted = base + float(fm.CYCLE_ITERS)          # exactly one colour cycle
    assert np.array_equal(ex.field_to_rgb(base), ex.field_to_rgb(shifted))
    assert not np.array_equal(ex.field_to_rgb(base),
                              ex.field_to_rgb(base + 0.5 * fm.CYCLE_ITERS))


def test_the_input_image_is_stretched_and_never_centre_cropped():
    """The failure this stops is silent: timm's default eval transform centre-crops, which
    on a 16:9 field deletes the left and right thirds — an embedding of a picture nobody
    screened. Content at both extreme edges must survive."""
    f = np.full((36, 64), np.nan, dtype=np.float32)   # black everywhere ...
    f[:, 0] = 20.0                                    # ... except one column at each edge
    f[:, -1] = 20.0
    im = np.asarray(ex.to_input_image(f, (224, 224)))
    assert im.shape == (224, 224, 3)
    assert im[:, :3].max() > 0 and im[:, -3:].max() > 0
    # NOT VACUOUS: the middle is black, so a centre crop (which would keep ~the middle 200
    # of 224 columns and drop both bright edges) fails the assertion above rather than this.
    assert im[:, 100:124].max() == 0


def test_the_module_does_not_use_timms_default_transform():
    src = (HERE / "exemplar_similarity.py").read_text(encoding="utf-8")
    assert "create_transform" not in src


# =========================================================================== #
# the exemplar set
# =========================================================================== #
def test_every_exemplar_names_where_its_positive_verdict_is_recorded():
    """An exemplar set assembled from memory is the author's taste wearing Matt's name."""
    rows = ex.exemplar_set()
    assert 5 <= len(rows) <= 8
    assert all(r["verdict"] and r["cx"] and r["cy"] and float(r["fw"]) > 0 for r in rows)
    assert {"q4_neig_089", "mb19_p35_16x", "minibroteye"} <= {r["key"] for r in rows}


def test_the_corpus_leg_is_c_plane_only_and_one_per_family():
    """Two filters, each stopping a different way for the set to be quietly wrong: a julia
    z-plane row is a different object from a c-plane minibrot view, and the two score-4 rows
    are near-duplicates that would weight `mean` toward one pair while posing as two."""
    rows = ex.corpus_exemplars()
    fams = [r["family"] for r in rows]
    assert set(fams) <= set(ex.CORPUS_FAMILIES)
    assert len(fams) == len(set(fams)), Counter(fams)


def test_the_corpus_draw_is_seeded_and_reproducible():
    assert [r["key"] for r in ex.corpus_exemplars(seed=1)] == \
        [r["key"] for r in ex.corpus_exemplars(seed=1)]


def test_similarity_reports_max_and_mean_and_they_can_disagree():
    """Both, because they are different questions: `max` is "looks like ONE of them",
    `mean` is "looks like the KIND of thing". A set with two near-duplicates moves `mean`
    and leaves `max` alone, which is the disagreement worth being able to see."""
    e = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = np.array([[1.0, 0.0], [0.7071, 0.7071]])
    smax, smean = ex.similarities(c, e)
    assert smax[0] == pytest.approx(1.0) and smean[0] == pytest.approx(0.5)
    assert smax[1] == pytest.approx(0.7071, abs=1e-4)
    assert smean[1] == pytest.approx(0.7071, abs=1e-4)
    assert smax[0] > smax[1] and smean[0] < smean[1]     # they order the pair oppositely


def test_an_empty_exemplar_set_yields_zeros_rather_than_raising():
    smax, smean = ex.similarities(np.zeros((3, 4)), np.zeros((0, 4)))
    assert smax.shape == (3,) and not smax.any() and not smean.any()


# =========================================================================== #
# the draws
# =========================================================================== #
def _pop(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ops = ["snap_to_nucleus", "neighborhood_expand"]
    out = []
    for i in range(n):
        c = float(rng.normal(5, 4))
        out.append(dict(key=f"a{i}|16.0", atom_key=f"a{i}", k=16.0,
                        op=ops[i % len(ops)], degree=2 + (i % 4),
                        period=10 + i % 30, cx="0.1", cy="0.2", fw=1e-3,
                        partition="mandelbrot", composite=c,
                        vetoed=c < 0, screen_frame="view",
                        exemplar_sim_max=float(rng.random())))
    return out


def test_bins_are_quantiles_of_the_runs_own_distribution():
    pop = _pop()
    edges = bsc.composite_bins(pop)
    assert len(edges) == bsc.N_BINS - 1 and edges == sorted(edges)
    counts = Counter(bsc.bin_of(r, edges) for r in pop)
    assert set(counts) == {1, 2, 3, 4, 5}
    assert max(counts.values()) - min(counts.values()) <= 2      # roughly equal by design


def test_an_unscored_row_gets_its_own_cell_not_bin_one():
    """"we could not measure this" and "this measured badly" are different facts, and
    binning them together would put unscreenable rows into the negative class."""
    edges = bsc.composite_bins(_pop())
    assert bsc.bin_of(dict(composite=None), edges) == 0
    assert bsc.bin_of(dict(composite=-0.5), edges) == 1           # vetoed: bin 1 by value


def test_the_uniform_leg_is_drawn_first_so_no_score_conditions_it():
    """The load-bearing ordering. If the uniform leg were drawn last its exclusions would be
    exactly the rows a score picked, which makes 'uniform' a score-dependent draw."""
    src = (HERE / "build_supply_crawl_batches.py").read_text(encoding="utf-8")
    body = src.split("def draw_all", 1)[1].split("\ndef ", 1)[0]
    assert body.index("draw_uniform") < body.index("draw_stratified") < \
        body.index("draw_exemplar")
    # ...and behaviourally: the uniform leg's membership must not move when the SCORES do.
    pop = _pop()
    a = {r["key"] for r in bsc.draw_all(pop)[0][bsc.UNIFORM]}
    for r in pop:
        r["composite"] = -float(r["composite"])       # invert every score
        r["vetoed"] = r["composite"] < 0
    b = {r["key"] for r in bsc.draw_all(pop)[0][bsc.UNIFORM]}
    assert a == b


def test_the_four_draws_are_disjoint():
    """A location may appear in only ONE batch — `build_manifest.load_post_freeze` asserts
    it, so an overlap is a hard failure of the whole manifest build, later and elsewhere."""
    chunks, rep = bsc.draw_all(_pop(800))
    keys = [r["key"] for rows in chunks.values() for r in rows]
    assert rep["overlap"] == 0
    assert len(keys) == len(set(keys))


def test_the_stratified_chunk_is_round_robin_to_plus_minus_one_over_its_cells():
    pop = _pop(800)
    edges = bsc.composite_bins(pop)
    got = bsc.draw_stratified(pop, 200, edges, 7)
    assert len(got) == 200
    cells = Counter((r["degree"], r["op"], bsc.bin_of(r, edges)) for r in got)
    assert max(cells.values()) - min(cells.values()) <= 1
    # and it reaches the LOW bins, which is the whole reason it is stratified rather than
    # proportional: those rows are the negative class.
    assert bsc.bin_of(min(got, key=lambda r: r["composite"]), edges) == 1


def test_the_stratified_draw_is_seeded_and_reproducible():
    pop = _pop(500)
    e = bsc.composite_bins(pop)
    assert [r["key"] for r in bsc.draw_stratified(pop, 120, e, 3)] == \
        [r["key"] for r in bsc.draw_stratified(pop, 120, e, 3)]
    assert [r["key"] for r in bsc.draw_stratified(pop, 120, e, 3)] != \
        [r["key"] for r in bsc.draw_stratified(pop, 120, e, 4)]


def test_the_exemplar_chunk_is_the_top_by_similarity_and_excludes_the_other_chunks():
    chunks, _ = bsc.draw_all(_pop(800))
    mini = chunks[bsc.EXEMPLAR]
    others = {r["key"] for bid, rows in chunks.items() if bid != bsc.EXEMPLAR for r in rows}
    assert not ({r["key"] for r in mini} & others)
    sims = [r["exemplar_sim_max"] for r in mini]
    assert sims == sorted(sims, reverse=True)


def test_a_population_mixing_screen_frames_is_refused(tmp_path, monkeypatch):
    """A 4x-atom radial_range and a view-frame one are different measurements. Binning them
    together would produce quintiles of a mixture and call them the run's distribution."""
    monkeypatch.setattr(bsc.mis, "load_population", lambda logs: [
        dict(atom_key="a", k=16.0, screen_frame="view", composite=1.0),
        dict(atom_key="b", k=16.0, screen_frame="atom4x", composite=None)])
    (tmp_path / "maneuvers.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        bsc.load_population(tmp_path)
    assert "screen frames" in str(e.value)


# =========================================================================== #
# registration
# =========================================================================== #
def test_every_batch_is_registered_explicitly_before_it_is_built():
    """Fail-closed would land the three biased legs train-side anyway. "Nobody registered
    this" and "this is a biased train draw" are different facts, and only one of them is
    true here.

    The UNIFORM leg's side moved after the crawl was drawn: registered train-side on
    2026-08-01, then made the maneuver-view eval instrument by Matt in the v10 build
    (2026-08-02) — which is what `data/v10/manifest.jsonl` realizes, 90 rows sourced
    `maneuver_uniform_v1`. The registry carried the stale train-side value until the
    2026-08-04 unification; this batch's `batch.json` keeps the tuple that was true when
    it was written, and `batch_registry` records it as the entry's `superseded` value."""
    for bid in bsc.BATCHES:
        split, biased, source = bm.assign_split({"batch": bid, "ft": "mandelbrot"})
        assert split == ("eval" if bid == bsc.UNIFORM else "train"), bid
        assert source != "unregistered", bid
        assert biased is (bid != bsc.UNIFORM), bid
    assert bm.assign_split({"batch": "never_registered", "ft": "mandelbrot"}) == \
        ("train", True, "unregistered")


def test_the_uniform_legs_superseded_registration_is_recorded_not_overwritten():
    """The crawl's `batch.json` froze ("train", False, "supply_crawl_uniform"). A frozen
    record keeps what was true when written, so the registry has to say what replaced it
    — otherwise the only explanation for the disagreement is "someone edited a table"."""
    import sys
    sys.path.insert(0, str(bm.ROOT / "tools" / "scoring"))
    import batch_registry as br
    reg = br.lookup(bsc.UNIFORM, "mandelbrot")
    assert reg.superseded == ("train", False, "supply_crawl_uniform")
    assert br.split_of(reg) == "eval" and reg.score_unconditioned is True


def test_the_uniform_batch_does_not_contradict_the_biased_registry():
    """`registration_contradictions` is a hard abort in the manifest build: a batch
    classified unbiased here and registered train-side-only there is two authorities
    disagreeing, and the build refuses rather than preferring one."""
    assert bm.registration_contradictions(
        [{"biased": False, "batch": bsc.UNIFORM}]) == []


def test_every_selection_axis_is_on_the_leak_list():
    """The served manifest is checked against this list on its BYTES. A selection key that
    is not on the list is a key `verify` will not look for."""
    for k in ("composite", "composite_bin", "exemplar_sim_max", "stratum",
              "selection_role", "op", "degree"):
        assert k in bsc.LEAK_KEYS


def test_the_sheet_leg_comes_from_the_gate_record_not_a_regenerated_sheet():
    """The correction that made this leg possible. Regenerating the calibration sheet here
    recovered 0 of its 6 named tiles (the `--sheet-order` row-order dependence), so the gate
    — which regenerates it correctly AND cross-checks the named tiles against it — writes the
    passed rows' keys into its own record, and this reads them."""
    src = (HERE / "exemplar_similarity.py").read_text(encoding="utf-8")
    assert "stratify" not in src and "view_screen_sheets" not in src
    rows = ex.sheet_exemplars()
    assert rows and len(rows) <= ex.N_SHEET
    assert all(r["verdict"].startswith(ex.GATE_REL) for r in rows)
    # one per OPERATOR before any operator gets two — the same discipline every draw here
    # uses, so "which three of twelve" is not the author's pick.
    ops = [r["label"].split(": ", 1)[1].split("|", 1)[0] for r in rows]
    assert len(set(ops)) == len(ops), ops


def test_the_exemplar_set_is_three_legs_and_stays_inside_five_to_eight():
    rows = ex.exemplar_set()
    assert 5 <= len(rows) <= 8
    assert len({r["key"] for r in rows}) == len(rows)
    legs = Counter("sheet" if k.startswith("sheet_") else
                   "corpus" if k.startswith("corpus_") else "named"
                   for k in (r["key"] for r in rows))
    assert legs["named"] == 3 and legs["sheet"] >= 1 and legs["corpus"] >= 1


# =========================================================================== #
# the render, which is the expensive half and therefore the one with a budget
# =========================================================================== #
def test_the_crop_cap_is_the_one_every_other_corpus_batch_uses():
    """Checkable against a row that already exists: `2026-07-26_minibrot_roster_v2` renders
    fw 2.116e-04 at maxiter 5512, which is `round(1500 * -log10(fw))`. Using the live deploy
    `auto_maxiter` instead made these crops incomparable with the rest of the corpus AND
    tripled the render bill for a picture the labeler cannot tell apart."""
    r = dict(cx="0.1", cy="0.2", fw=2.11635414e-04, partition="mandelbrot")
    assert bsc._render_block(r, "magma")["maxiter"] == 5512
    # floored at 3000 for anything shallow, so a base-scale root view is not free-for-all
    assert bsc._render_block(dict(r, fw=1.0), "magma")["maxiter"] == 3000


def test_the_render_order_finishes_whole_batches_smallest_first():
    """1,460 renders does not fit any session that also produced the run, so the order is a
    budget decision: a partly-rendered batch is not labelable, and the legs that TEST
    something are the small ones."""
    assert set(bsc.RENDER_ORDER) == set(bsc.BATCHES)
    assert bsc.RENDER_ORDER[0] == bsc.EXEMPLAR and bsc.RENDER_ORDER[1] == bsc.UNIFORM
    assert bsc.N_EXEMPLAR < bsc.N_UNIFORM < bsc.N_STRAT


def test_render_threads_are_sized_for_the_fan_out_not_for_one_process():
    """`DEFAULT_ENGINE_THREADS` is documented as the number for ONE engine process. Four
    workers of seven is 28 threads on a 12-core box — oversubscription, not throughput."""
    import corpus_common as _cc
    assert bsc.RENDER_THREADS < _cc.DEFAULT_ENGINE_THREADS
    assert 4 * bsc.RENDER_THREADS <= 12


def test_a_timed_out_render_leaves_no_half_written_crop(tmp_path, monkeypatch):
    """The one failure this pipeline cannot see. `needs()` checks EXISTENCE, so a JPG
    truncated by a timeout kill reads as rendered forever and the batch is quietly one bad
    crop short — of a picture a human is about to score."""
    out = tmp_path / "sc0000_deadbeef.jpg"

    def half_write(render, path, **kw):
        Path(path).write_bytes(b"\xff\xd8\xff\xe0truncated")
        raise TimeoutError("engine killed at the timeout")

    monkeypatch.setattr(bsc.cc, "render_corpus_crop", half_write)
    with pytest.raises(TimeoutError):
        bsc._render_to({"palette": "magma"}, out, "src", 1.0)
    assert not out.exists()
