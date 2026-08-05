"""Tests for the dedup-boundary calibration instrument.

The three things that make this sheet an instrument rather than a gallery, each tested here
because each fails silently: the BLIND (a geometry number leaking into the visible DOM anchors
the judgment the sheet exists to collect), the SORT/BINNING (the decision variable must be
monotone across the page or "where does the answer flip" has no meaning), and the ANCHOR
INVARIANT (a far-side stratum that is not actually kept apart by the current rule is not a far
side). Everything runs on synthetic rows — no run data, no renders.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import precanon_calibration_sheet as S      # noqa: E402


# --------------------------------------------------------------------------- #
# bands + binning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio,expect", [
    (1.0, "le2"), (1.9999, "le2"), (2.0, "le2"),          # 2.0 closes the first band
    (2.0001, "mid"), (9.9999, "mid"), (10.0, "mid"),      # 10.0 closes the second
    (10.0001, "gt10"), (1e6, "gt10"), (math.inf, "gt10"),
])
def test_band_of_is_a_partition(ratio, expect):
    assert S.band_of(ratio) == expect


def test_bands_cover_the_ratio_line_without_overlap():
    los = [lo for _k, _l, lo, _h in S.BANDS]
    his = [hi for _k, _l, _lo, hi in S.BANDS]
    assert los == sorted(los) and his[:-1] == los[1:] and his[-1] == math.inf


def test_quantile_edges_are_monotone_and_bracket_the_data():
    vals = [0.01 * (1.6 ** i) for i in range(60)]
    e = S.quantile_edges(vals, 8)
    assert len(e) == 9
    assert e == sorted(e)
    assert e[0] == min(vals) and e[-1] == max(vals)


def test_every_value_lands_in_a_bin_and_bins_are_ordered():
    rng = random.Random(7)          # hoisted: `random.Random(7).uniform(...)` inside the
    vals = sorted(rng.uniform(0.0, 6.0) for _ in range(400))    # genexp reseeds every draw
    edges = S.quantile_edges(vals, S.N_BINS)
    bins = [S.assign_bin(v, edges) for v in vals]
    assert set(bins) <= set(range(S.N_BINS))
    assert bins == sorted(bins)                  # binning preserves the sort order
    assert set(bins) == set(range(S.N_BINS))     # no empty bin on a well-spread population


def test_degenerate_band_collapses_instead_of_scattering():
    """All-equal d/min: every edge is the same value. The ties must land in ONE bin — a rule
    that spread them would report even coverage of a population that has one point. Which bin
    is not the claim; that they share it is."""
    edges = S.quantile_edges([1.25] * 40, S.N_BINS)
    assert len({S.assign_bin(1.25, edges) for _ in range(5)}) == 1
    assert 0 <= S.assign_bin(1.25, edges) < S.N_BINS


def test_quantile_edges_empty_population():
    assert S.quantile_edges([], 8) == []


# --------------------------------------------------------------------------- #
# seeded selection
# --------------------------------------------------------------------------- #
def test_pick_is_deterministic_per_seed_and_moves_with_it():
    items = list(range(50))
    a = S.pick(random.Random("x:le2"), items, 5)
    b = S.pick(random.Random("x:le2"), items, 5)
    c = S.pick(random.Random("y:le2"), items, 5)
    assert a == b and a != c
    assert S.pick(random.Random("x"), [1, 2], 5) == [1, 2]     # short pool passes through


# --------------------------------------------------------------------------- #
# pair geometry + the anchor invariant
# --------------------------------------------------------------------------- #
def _led(i, cx, cy, fw, *, family="multibrot3", distinct=True, **kw):
    return dict(id=f"o{i}", family=family, outcome_cx=cx, outcome_cy=cy, outcome_fw=fw,
                distinct=distinct, decoded_class=3, **kw)


def _q4(led, partition="multibrot3"):
    return dict(ledger=led, partition=partition)


def test_geom_is_symmetric_and_uses_min_max_not_side_order():
    a = dict(cx=0.0, cy=0.0, fw=1e-3)
    b = dict(cx=3e-3, cy=0.0, fw=1e-2)
    g1, g2 = S._geom(a, b), S._geom(b, a)
    assert g1 == g2
    assert g1["fw_ratio"] == pytest.approx(10.0)
    assert g1["d_over_min"] == pytest.approx(3.0)
    assert g1["d_over_max"] == pytest.approx(0.3)


def test_anchor_pairs_are_kept_apart_and_deduped_unordered():
    # two admitted c-plane rows, far enough apart that the current rule kept them apart
    d = 1.5 * 1e-2 * 3
    ls = [_led(0, 0.0, 0.0, 1e-2), _led(1, d, 0.0, 1e-2)]
    ledger = {l["id"]: l for l in ls}
    pairs, viol = S.anchor_pairs([_q4(l) for l in ls], ledger)
    assert viol == []
    assert len(pairs) == 1                      # a→b and b→a collapse to one pair
    assert pairs[0]["stratum"] == "anchor"
    assert pairs[0]["geom"]["dist"] == pytest.approx(d)


def test_anchor_violation_is_reported_not_swallowed():
    """A pair inside `DEDUP_K * max(fw)` cannot have been kept apart by the current rule; if
    one shows up the stratum is mis-derived and the caller must stop, not sample it."""
    ls = [_led(0, 0.0, 0.0, 1e-2), _led(1, 0.5 * 1.5 * 1e-2, 0.0, 1e-2)]
    ledger = {l["id"]: l for l in ls}
    _pairs, viol = S.anchor_pairs([_q4(l) for l in ls], ledger)
    assert len(viol) == 1 and "o0" in viol[0] and "o1" in viol[0]


def test_anchor_pairs_never_cross_identity():
    """Two julia rows with different `c` are different fractals and `near_dup` never compares
    their distance — pairing them would put a meaningless distance on the axis."""
    ls = [_led(0, 0.0, 0.0, 1e-2, family="julia:mandelbrot", julia_c_re=0.1, julia_c_im=0.2),
          _led(1, 1e-3, 0.0, 1e-2, family="julia:mandelbrot", julia_c_re=0.9, julia_c_im=0.2)]
    ledger = {l["id"]: l for l in ls}
    pairs, viol = S.anchor_pairs([_q4(l, "julia:mandelbrot") for l in ls], ledger)
    assert pairs == [] and viol == []


def test_anchor_pairs_ignore_non_admitted_ledger_rows():
    ls = [_led(0, 0.0, 0.0, 1e-2), _led(1, 1.0, 0.0, 1e-2, distinct=False)]
    ledger = {l["id"]: l for l in ls}
    pairs, _viol = S.anchor_pairs([_q4(l) for l in ls], ledger)
    assert pairs == []


def test_dup_pairs_skip_unresolvable_displacers():
    rows = [dict(fate="precanon_dup", rec_dup="missing", partition="multibrot3",
                 cx=0.0, cy=0.0, fw=1e-3, julia_c_re=None, julia_c_im=None, phoenix=None,
                 batch=1, node_id=2, depth=3, mix_source="steered", cheap_pgood=0.4,
                 ledger=None)]
    assert S.dup_pairs(rows, {}) == []


# --------------------------------------------------------------------------- #
# the blind
# --------------------------------------------------------------------------- #
def _synthetic_pair(pid="c000_deadbeef", left="a"):
    a = dict(kind="candidate", id="007_01234", partition="multibrot3",
             cx=0.1234567890123, cy=-0.9876543210987, fw=1.2345678e-05,
             julia_c_re=None, julia_c_im=None, phoenix=None, batch=7, node_id=1234,
             depth=4, mix_source="steered", cheap_pgood=0.4321, decoded_class=None,
             fate="precanon_dup")
    b = dict(kind="outcome", id="st_multibrot3_x_000042", partition="multibrot3",
             cx=0.1234599990123, cy=-0.9876543210987, fw=9.8765432e-04,
             julia_c_re=None, julia_c_im=None, phoenix=None, batch=None, node_id=99,
             depth=2, mix_source="steered", cheap_pgood=0.55, decoded_class=3,
             fate="admitted")
    return dict(pair_id=pid, stratum="dup", band="mid", bin=3, partition="multibrot3",
                left=left, palette="viridis", a=a, b=b, geom=S._geom(a, b))


_RV = re.compile(r'<(figcaption|span|p)[^>]*class="rv[^"]*"[^>]*>.*?</\1>', re.S)


def _visible(markup: str) -> str:
    """The markup with every reveal-gated element removed — i.e. what the eye can read while
    judging. (`.rv` is `display:none` until `r`, so this is exactly the blind surface.)"""
    return _RV.sub("", markup)


def test_blind_dom_carries_no_geometry_fate_or_verdict():
    p = _synthetic_pair()
    vis = _visible(S.pair_section(p, S.SEED and 1.5))
    for token in ("precanon_dup", "admitted", "candidate", "outcome",
                  "st_multibrot3", "007_01234", "dup", "anchor",
                  "d/min", "fw ratio", "CUT", "kept apart", "stratum", "bin"):
        assert token not in vis, f"blind DOM leaks {token!r}"
    for val in (p["a"]["cx"], p["a"]["fw"], p["b"]["fw"], p["geom"]["dist"],
                p["geom"]["d_over_min"], p["geom"]["fw_ratio"]):
        assert f"{val:.4g}" not in vis and f"{val:.6g}" not in vis
    # the controls and both tiles ARE visible
    assert vis.count("<img") == 2                        # one image per location, no companion
    for v in ("SAME", "DISTINCT", "UNSURE"):
        assert v in vis


def test_pair_id_encodes_no_stratum_or_fate():
    """The id is in the image src and the DOM; if it said `anchor` the blind would be over."""
    dup = _synthetic_pair()
    anc = dict(_synthetic_pair(), stratum="anchor")
    assert re.fullmatch(r"c\d{3}_[0-9a-f]{8}", dup["pair_id"])
    for pid in (dup["pair_id"], anc["pair_id"]):
        assert "anchor" not in pid and "dup" not in pid and "rej" not in pid


def test_left_right_is_seeded_not_fixed_to_the_displacer_side():
    """The displacer is the wider frame in most dup pairs, so a fixed side is a tell."""
    import build_minibrot_batch as BMB
    sides = ["a" if (BMB._stable_seed(f"{S.SEED}:side:c{i:03d}_00000000") & 1) == 0 else "b"
             for i in range(200)]
    assert 60 <= sides.count("a") <= 140                 # not degenerate either way
    assert sides == ["a" if (BMB._stable_seed(f"{S.SEED}:side:c{i:03d}_00000000") & 1) == 0
                     else "b" for i in range(200)]       # and reproducible


def test_card_srcs_are_page_relative_and_one_per_location():
    """A root-relative src 404s in the browser and still passes a root-fetching link check.
    And there is exactly ONE tile per side: a second coloring next to a geometry judgment is a
    colour difference the eye has to subtract before it can answer the question asked."""
    p = _synthetic_pair()
    srcs = re.findall(r'src="([^"]+)"', S.pair_section(p, 1.5))
    assert sorted(srcs) == [f"{S.RENDERS_URL}/{p['pair_id']}.a.jpg",
                            f"{S.RENDERS_URL}/{p['pair_id']}.b.jpg"]
    assert not any(s.startswith(("/", "http", "scratch/")) for s in srcs)


def test_every_tile_uses_the_one_production_canonical_palette():
    """`plan` stamps one palette on every pair; a per-pair seeded palette would put a colour
    change between two pairs Matt is comparing along the sort."""
    import production_pins as prod
    assert S.PALETTE == prod.PALETTE == "twilight_shifted"
    assert S.PALETTE_SOURCE.name == "clean_colormaps.json" and S.PALETTE_SOURCE.exists()
    names = {e["name"] for e in json.loads(S.PALETTE_SOURCE.read_text(encoding="utf-8"))}
    assert S.PALETTE in names


# --------------------------------------------------------------------------- #
# whole-page assembly
# --------------------------------------------------------------------------- #
def test_stage_sheet_renders_a_complete_page(tmp_path, monkeypatch):
    pairs = [dict(_synthetic_pair(f"c{i:03d}_0000000{i}", "a" if i % 2 else "b"),
                  band=b, bin=i % S.N_BINS)
             for i, b in enumerate(["le2", "le2", "mid", "mid", "gt10", "gt10"])]
    plan = dict(run="r", seed=1, dedup_k=1.5, dedup_scale="max", domain_cap=6.0, n_bins=8,
                per_bin=5, n_anchor_per_band=5, n_precanon_dup=1, n_dup_pairs=1,
                n_anchor_pairs=1, n_selected=len(pairs), sampling="s",
                crop=dict(w=640, h=360, ss=2, filter="lanczos3", interior="black",
                          composition="center", palette=S.PALETTE,
                          palette_source="clean_colormaps.json"),
                bands={k: dict(bin_edges=[0.0, 1.0], n_dup_in_domain=1, n_dup_above_cap=2,
                               n_anchor_available=3) for k, *_r in S.BANDS},
                pairs=pairs)
    monkeypatch.setattr(S, "PLAN", tmp_path / "pairs.json")
    monkeypatch.setattr(S, "SHEET", tmp_path / "sheet.html")
    S.PLAN.write_text(json.dumps(plan), encoding="utf-8")
    assert S.stage_sheet(None) == 0

    html = S.SHEET.read_text(encoding="utf-8")
    for ph in ("__BODY__", "__PAIRS__", "__META__", "__ORDER__"):
        assert ph not in html, f"unsubstituted placeholder {ph}"
    head = html.split("<script>")[0]
    assert _visible(head).count("d/min(fw)") == 0        # only the header prose, which is rv/JS
    assert html.count('class="pair"') == len(pairs)
    # the export payload must carry full identity for BOTH sides, or the follow-up cannot
    # re-derive a boundary from the verdicts.
    payload = json.loads(re.search(r"const PAIRS=(\{.*?\}), META=", html, re.S).group(1))
    assert set(payload) == {p["pair_id"] for p in pairs}
    one = payload[pairs[0]["pair_id"]]
    assert {"band", "bin", "stratum", "left", "geom", "a", "b"} <= set(one)
    assert {"cx", "cy", "fw", "partition"} <= set(one["a"]) <= set(one["a"])
    assert {"d_over_min", "d_over_max", "fw_ratio", "dist"} <= set(one["geom"])


def test_render_block_pins_this_sheets_fidelity():
    """640x360 ss2 is this sheet's stated substrate; inheriting the corpus 1280x720 ss4 would
    quadruple the render bill and silently change what Matt is judging."""
    side = _synthetic_pair()["b"]
    rb = S.render_block(side, "viridis")
    assert (rb["width"], rb["height"], rb["ss"]) == (S.CROP_W, S.CROP_H, S.CROP_SS) \
        == (640, 360, 2)
    assert rb["palette"] == "viridis"


def test_module_replays_the_retired_rule_and_never_writes_a_constant():
    """This module is a RECORD REPLAY: the 135 verdicts were taken on pairs the retired
    `1.5 x max(fw)` rule collapsed and kept apart, so its anchor invariant and its plan meta
    must keep reading `RETIRED_DEDUP_*`. Reading the live pair (recalibrated to 0.25 x min on
    2026-08-04 — by these very verdicts) would restratify the sample and stamp a rule the run
    never ran into a rebuilt plan's meta. And it still assigns nothing."""
    import production_seeder as ps
    assert (ps.RETIRED_DEDUP_K, ps.RETIRED_DEDUP_SCALE) == (1.5, "max")
    src = Path(S.__file__).read_text(encoding="utf-8")
    assert not re.search(r"ps\.(RETIRED_)?DEDUP_(K|SCALE)\s*=", src)
    # the replay reads the retired pair and NEVER the live one.
    assert "ps.RETIRED_DEDUP_K" in src and "ps.RETIRED_DEDUP_SCALE" in src
    assert not re.search(r"ps\.DEDUP_(K|SCALE)\b", src)


# --------------------------------------------------------------------------- #
# the shared render authority (`bq._render_block`) — phoenix `c`
# --------------------------------------------------------------------------- #
_PHX = dict(phoenix_c_re=0.32174800993565295, phoenix_c_im=0.12795705820626488,
            phoenix_p_re=-0.11469662354953557, phoenix_p_im=0.4418550963483557,
            phoenix_zm1_re=-0.07554828926023154, phoenix_zm1_im=-0.004060291925500022)


def test_ledger_sourced_phoenix_row_keeps_its_own_c():
    """An outcome-ledger phoenix row writes `c` as `phoenix_c_re` and leaves `julia_c_re`
    null. Dropped, `render_one_flags` omits `--c` and the engine renders its DEFAULT phoenix
    plane at the right coordinates — a real-looking image of a different fractal."""
    import build_q4_harvest_batches as bq
    import location as loc_mod
    row = dict(cx=-0.41, cy=-0.597, fw=0.7988, family="phoenix", _palette="viridis",
               julia_c_re=None, julia_c_im=None, **_PHX)
    rb = bq._render_block(row)
    assert float(rb["c_re"]) == pytest.approx(_PHX["phoenix_c_re"])
    assert float(rb["c_im"]) == pytest.approx(_PHX["phoenix_c_im"])
    flags = loc_mod.render_one_flags(loc_mod.from_render_block(rb))
    assert "--c" in flags and "--p" in flags and "--phoenix-z1" in flags


def test_check_sourced_phoenix_row_is_unchanged():
    """A q4 CHECK carries the same `c` in `julia_c_re`; the fallback must not shadow it."""
    import build_q4_harvest_batches as bq
    row = dict(cx=-0.41, cy=-0.605, fw=0.4735, family="phoenix", _palette="viridis",
               julia_c_re=_PHX["phoenix_c_re"], julia_c_im=_PHX["phoenix_c_im"], **_PHX)
    rb = bq._render_block(row)
    assert float(rb["c_re"]) == pytest.approx(_PHX["phoenix_c_re"])


def test_phoenix_row_with_no_c_anywhere_fails_loud():
    import build_q4_harvest_batches as bq
    row = dict(cx=0.0, cy=0.0, fw=1.0, family="phoenix", _palette="viridis",
               julia_c_re=None, julia_c_im=None,
               p_re=0.1, p_im=0.2, zm1_re=0.0, zm1_im=0.0)
    with pytest.raises(SystemExit, match="DEFAULT phoenix plane"):
        bq._render_block(row)


def test_both_sides_of_a_phoenix_pair_render_the_same_plane():
    """The pair is only a comparison if both sides are the same fractal. The candidate side
    comes from the q4 store and the displacer from the ledger, which write `c` in different
    columns — this is that join, at the render block."""
    import build_q4_harvest_batches as bq
    cand = dict(kind="candidate", partition="phoenix", cx=-0.4145, cy=-0.6050, fw=0.4735,
                julia_c_re=_PHX["phoenix_c_re"], julia_c_im=_PHX["phoenix_c_im"],
                phoenix={k: str(v) for k, v in _PHX.items()})
    disp = dict(kind="outcome", partition="phoenix", cx=-0.4101, cy=-0.5976, fw=0.7988,
                julia_c_re=None, julia_c_im=None, phoenix=dict(_PHX))
    ra, rb_ = S.render_block(cand, S.PALETTE), S.render_block(disp, S.PALETTE)
    for k in ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"):
        assert float(ra[k]) == pytest.approx(float(rb_[k])), f"{k} differs across the pair"
    assert (ra["fractal_type"], rb_["fractal_type"]) == ("phoenix", "phoenix")


def test_number_keys_judge_and_skip_to_the_next_unjudged_pair():
    """1/2/3 are the fast pass down the sort: set the verdict, then jump past anything already
    answered. s/d/u keep the in-order step so re-judging a row does not teleport out of the
    region being examined. Asserted on the page template because the binding is the feature."""
    page = S.PAGE.replace(" ", "")
    for key, verdict in (("1", "same"), ("2", "distinct"), ("3", "unsure")):
        assert f"k==='{key}'){{setV('{verdict}',true)" in page
    for key, verdict in (("s", "same"), ("d", "distinct"), ("u", "unsure")):
        assert f"k==='{key}'){{setV('{verdict}')" in page
    assert "functionsetV(v,skip)" in page
    assert "if(skip)nextUn();" in page


def test_number_keys_are_visible_on_the_controls():
    """A keybinding nobody can see is a keybinding nobody uses."""
    vis = _visible(S.pair_section(_synthetic_pair(), 1.5))
    for label, key in (("SAME", "1"), ("DISTINCT", "2"), ("UNSURE", "3")):
        assert f'{label} <span class="kc">{key}</span>' in vis
