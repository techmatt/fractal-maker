#!/usr/bin/env python
"""`tools/corpus/eval_only.py` is the ONE rule that pins an eval-only batch to the eval side.

Four groups, and the third is the one that would have caught the failure this file exists
for (a blind slice silently entering a train split):

  1. The owner's own arithmetic, on synthetic corpus trees — a batch that stamps `eval_only`
     without a reason, a row stamped train inside one, a pin that moves a key, an
     `assert_eval` that raises on exactly the case the pin would have fixed.
  2. `split_units.build_split` honours `force_eval` at UNIT granularity and is byte-identical
     when the forced set is empty. The empty case matters most: the render-mode corpus's
     live split runs through this call, and a drift there is a silent re-split of a trained
     head's eval side.
  3. THE LIVE CORPORA, on disk. Sheet D is stamped eval-only, all 197 of its rows stamp
     `split_side=eval`, and both split passes that could ever see it (the render-mode global
     re-derivation, the wallpaper frozen-authority pass) refuse a train-side placement.
  4. A source scan: nobody re-implements "is this batch eval-only" by reading the flag out
     of a batch.json themselves.

  uv run pytest tools/corpus/test_eval_only.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "mining"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.corpus import eval_only as EO            # noqa: E402
from tools.mining.split_units import build_split    # noqa: E402

OWNER = "tools/corpus/eval_only.py"
SHEET_D = "2026-08-11_wallpaper_blind_minibrot_v1"


# =========================================================================== #
# 1. the owner's arithmetic
# =========================================================================== #
def _corpus(tmp_path: Path, batches: dict, corpus: str = "wallpaper_corpus") -> Path:
    """Build a throwaway corpus tree. `batches` is `{batch_id: (batch_json, rows)}`."""
    for bid, (bj, rows) in batches.items():
        d = tmp_path / "data" / corpus / "batches" / bid
        d.mkdir(parents=True)
        (d / "batch.json").write_text(json.dumps(bj), encoding="utf-8")
        (d / "images.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def _row(iid, side="eval", cx="0.1", cy="0.2", fw="1e-3", ft="mandelbrot"):
    return {"image_id": iid,
            "render": {"cx": cx, "cy": cy, "fw": fw, "fractal_type": ft,
                       "c_re": None, "c_im": None},
            "provenance": {"split_side": side, "location_key": f"loc::{iid}"}}


def test_an_eval_only_batch_is_found_with_its_reason_and_a_plain_batch_is_not(tmp_path):
    root = _corpus(tmp_path, {
        "b_blind": ({"eval_only": True, "eval_only_note": "bought to referee two heads"},
                    [_row("a"), _row("b")]),
        "b_plain": ({"generator_version": "x"}, [_row("c", side="train")]),
    })
    got = EO.eval_only_batches("wallpaper_corpus", root=root)
    assert set(got) == {"b_blind"}
    assert got["b_blind"].n_rows == 2
    assert "referee" in got["b_blind"].reason
    assert EO.is_eval_only("wallpaper_corpus", "b_blind", root=root)
    assert not EO.is_eval_only("wallpaper_corpus", "b_plain", root=root)


def test_eval_only_without_a_note_is_a_violation_not_an_empty_reason(tmp_path):
    root = _corpus(tmp_path, {"b": ({"eval_only": True}, [_row("a")])})
    with pytest.raises(EO.EvalOnlyViolation, match="eval_only_note"):
        EO.eval_only_batches("wallpaper_corpus", root=root)


def test_a_row_stamped_train_inside_an_eval_only_batch_fails_the_stamp_check(tmp_path):
    root = _corpus(tmp_path, {
        "b": ({"eval_only": True, "eval_only_note": "why"},
              [_row("a"), _row("b", side="train")])})
    rep = EO.check_stamps("wallpaper_corpus", root=root)
    assert rep["ok"] is False and rep["violations"][0]["image_id"] == "b"
    with pytest.raises(EO.EvalOnlyViolation):
        EO.assert_stamps("wallpaper_corpus", root=root)


def test_ids_key_on_the_image_id_by_default_and_on_whatever_key_of_returns(tmp_path):
    root = _corpus(tmp_path, {
        "b": ({"eval_only": True, "eval_only_note": "why"}, [_row("a"), _row("b")])})
    assert EO.eval_only_ids("wallpaper_corpus", root=root) == {"a": "b", "b": "b"}
    by_loc = EO.eval_only_ids("wallpaper_corpus", root=root,
                              key_of=lambda r: r["provenance"]["location_key"])
    assert set(by_loc) == {"loc::a", "loc::b"}
    by_coord = EO.eval_only_ids("wallpaper_corpus", root=root, key_of=EO.coord_key)
    assert list(by_coord) == [("0.1", "0.2", "1e-3", "mandelbrot", None, None)]


def test_a_key_of_that_returns_none_drops_the_row_rather_than_pinning_none(tmp_path):
    """The render-mode dialect asks for `location_key`; a wallpaper row has none. Pinning a
    None key would force whatever `side[None]` happens to be — nothing, today, and a real
    location the day a split keys on something nullable."""
    root = _corpus(tmp_path, {
        "b": ({"eval_only": True, "eval_only_note": "why"}, [_row("a")])})
    assert EO.eval_only_ids("wallpaper_corpus", root=root,
                            key_of=lambda r: r["provenance"].get("nope")) == {}


def test_pin_moves_a_train_key_to_eval_and_reports_the_move():
    side = {"a": "train", "b": "eval", "c": "train"}
    rep = EO.pin(side, {"a": "b_blind", "b": "b_blind", "zz": "b_blind"}, where="t")
    assert side == {"a": "eval", "b": "eval", "c": "train"}
    assert rep["n_moved_to_eval"] == 1 and rep["moved"][0]["key"] == "a"
    assert rep["n_present_in_split"] == 2      # "zz" is not in this split at all


def test_assert_eval_raises_on_exactly_what_pin_would_have_fixed():
    with pytest.raises(EO.EvalOnlyViolation, match="train side"):
        EO.assert_eval({"a": "train"}, {"a": "b_blind"}, where="t")
    assert EO.assert_eval({"a": "eval"}, {"a": "b_blind"}, where="t")["ok"]
    # A pass that never called `pin` still dies here — the two are deliberately separate.
    side = {"a": "train"}
    EO.pin(side, {"a": "b"}, where="t")
    assert EO.assert_eval(side, {"a": "b"}, where="t")["ok"]


def test_an_unknown_corpus_names_the_registry_instead_of_returning_empty():
    with pytest.raises(ValueError, match="merge_sitting.CORPORA"):
        EO.eval_only_batches("no_such_corpus")


# =========================================================================== #
# 2. the split pass honours it — and is inert without it
# =========================================================================== #
def _locs(n=40):
    return {f"L{i}": {"family": "mandelbrot",
                      "render": {"cx": f"0.{i}", "cy": "0.0", "fw": "1e-3",
                                 "c_re": None, "c_im": None}}
            for i in range(n)}


def test_build_split_is_byte_identical_when_nothing_is_forced():
    locs = _locs()
    a, ma = build_split(locs, seed=0)
    b, mb = build_split(locs, seed=0, force_eval=())
    assert a == b
    assert mb["n_forced_eval_units"] == 0 and mb["n_eval_units"] == ma["n_eval_units"]


def test_a_forced_key_lands_eval_and_is_withheld_from_the_draw():
    locs = _locs()
    base, _ = build_split(locs, seed=0)
    train_keys = sorted(k for k, s in base.items() if s == "train")
    assert train_keys, "the unforced split must have a train side or this proves nothing"
    side, meta = build_split(locs, seed=0, force_eval={train_keys[0]})
    assert side[train_keys[0]] == "eval"
    assert meta["n_forced_eval_units"] == 1
    assert meta["n_eval_units"] == 1 + int(round(0.40 * (len(locs) - 1)))


def test_a_forced_julia_child_pins_its_whole_unit_not_just_itself():
    """Unit granularity is the point: pinning one member and drawing the rest is the
    straddle `split_units` exists to prevent."""
    locs = {"parent": {"family": "mandelbrot",
                       "render": {"cx": "0.25", "cy": "0.5", "fw": "1e-3",
                                  "c_re": None, "c_im": None}},
            "child": {"family": "julia",
                      "render": {"cx": "0.0", "cy": "0.0", "fw": "1e-3",
                                 "c_re": "0.25", "c_im": "0.5"}}}
    locs.update(_locs(20))
    side, meta = build_split(locs, seed=0, force_eval={"child"})
    assert side["child"] == side["parent"] == "eval"
    assert meta["n_locations_pinned_by_force"] == 2


def test_a_forced_key_outside_the_pool_is_reported_not_silently_dropped():
    _, meta = build_split(_locs(), seed=0, force_eval={"not_here"})
    assert meta["forced_keys_not_in_this_pool"] == ["not_here"]


# =========================================================================== #
# 3. the live corpora on disk
# =========================================================================== #
def test_sheet_d_is_eval_only_on_disk_with_every_row_stamped_eval():
    blk = EO.eval_only_batches("wallpaper_corpus").get(SHEET_D)
    assert blk is not None, f"{SHEET_D} lost its eval_only stamp"
    assert blk.n_rows == 197
    rep = EO.assert_stamps("wallpaper_corpus")
    assert rep["batches"][SHEET_D]["n_not_stamped_eval"] == 0


def test_every_eval_only_batch_in_every_live_corpus_stamps_its_rows_eval():
    """Derive + prove non-empty: the loop below would pass vacuously on a corpus with no
    eval-only batch, so the count is asserted separately."""
    total = sum(EO.assert_stamps(c)["n_batches"] for c in EO.KNOWN_CORPORA)
    assert total >= 1, "no eval-only batch anywhere — the pin has nothing to protect"


def test_the_wallpaper_split_pass_refuses_a_train_side_eval_only_row():
    """The guard the retrain that folds sheet D in will hit. Exercised through the REAL
    `assert_eval_only_pinned` against the REAL on-disk stamps, with a fake row standing in
    for the trainer's — a WRow it has never loaded is exactly the state to test."""
    sys.path.insert(0, str(ROOT))
    from classifier.train_wallpaper_v4 import assert_eval_only_pinned   # noqa: PLC0415

    d = ROOT / "data" / "wallpaper_corpus" / "batches" / SHEET_D / "images.jsonl"
    first = json.loads(d.read_text(encoding="utf-8").splitlines()[0])

    class R:                       # the only two attributes the guard reads
        full_coord = EO.coord_key(first)
        image_id = first["image_id"]

    assert assert_eval_only_pinned([R()], lambda r: "eval", where="t")["ok"]
    with pytest.raises(EO.EvalOnlyViolation, match="EVAL-ONLY"):
        assert_eval_only_pinned([R()], lambda r: "train", where="t")


def test_the_mining_global_pass_carries_the_pin_even_though_it_pins_nothing_today():
    """`load_corpus` is §2a's global re-derivation. It reports the pin as a number, so
    "0 forced" is a measured 0 rather than an absent check."""
    from tools.mining import mining_corpus as MC        # noqa: PLC0415
    pool = MC.load_corpus(require_crops=False)
    pin = pool.split_meta["eval_only_pin"]
    assert pin["where"] == "mining_corpus.load_corpus"
    assert pin["ok"] and pin["n_forced_keys"] == len(
        EO.eval_only_ids("render_mode_corpus",
                         key_of=lambda r: (r.get("provenance") or {}).get("location_key")))


# =========================================================================== #
# 4. one owner
# =========================================================================== #
def test_nobody_else_reads_the_eval_only_flag_out_of_a_batch_json():
    """A second reader is a second answer to "may this batch train". The builder that WRITES
    the stamp and this owner are the only files allowed to name the key."""
    out = subprocess.run(
        # the READ, not the word: `["eval_only"]` / `.get("eval_only")`. Prose about the
        # rule is welcome anywhere; a second file deciding the answer is not.
        ["git", "grep", "-lE", r'''(\[|get\()["']eval_only["']''', "--", "*.py"],
        cwd=ROOT, capture_output=True, text=True)
    allowed = {OWNER, "tools/corpus/test_eval_only.py",
               "tools/wallpaper/build_blind_minibrot_sheet.py",       # writes the stamp
               "tools/wallpaper/test_blind_minibrot_sheet.py"}
    hits = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
    assert hits <= allowed, f"new eval_only reader(s): {sorted(hits - allowed)}"
