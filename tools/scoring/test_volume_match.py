#!/usr/bin/env python
"""`tools/scoring/volume_match.py` + `rescore_fit_slices.py` + `adopt_head.py` — the flip's
three new owners, and the properties a head flip depends on.

The arithmetic half is pure and is tested on synthetic scores, because that is where the
off-by-one lives: `>` and `>=` disagree about the rows sitting exactly on a cut, and the two
stage-2 sites use different operators. The wiring half is tested against the COMMITTED
records, so a record that stopped matching its own owner goes red here rather than at the next
flip.

  uv run pytest tools/scoring/test_volume_match.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import floors as F                    # noqa: E402
from tools.scoring import volume_match as VM              # noqa: E402
from tools.scoring import rescore_fit_slices as RF        # noqa: E402

pytestmark = pytest.mark.stage2_pinned


# --------------------------------------------------------------------------- #
# 1. the arithmetic — pure, and it is the off-by-one that matters
# --------------------------------------------------------------------------- #
def test_the_matched_cut_admits_the_same_k_under_BOTH_comparisons():
    """The midpoint is the whole point. A report's `cut_at` is the k-th largest score, which
    admits k under `>=` and k-1 under `>` — and `emit_v1` gates with `>` while
    `MiningScorer.gate` uses `>=`, so a single convention would be wrong at one of the two
    sites."""
    rng = np.random.default_rng(0)
    s = rng.random(500)
    for k in (1, 7, 123, 499):
        t = VM.midpoint_cut(s, k)
        assert int((s > t).sum()) == k
        assert int((s >= t).sum()) == k


def test_the_matched_cut_is_exact_when_scores_tie_around_the_boundary():
    """Ties are where a k-th-value threshold silently moves the volume."""
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    t = VM.midpoint_cut(s, 1)
    assert int((s >= t).sum()) == 1 and int((s > t).sum()) == 1
    # k inside the tie run cannot be realized by ANY threshold; the midpoint collapses onto
    # the tie and the caller sees it in `realized_volume`, which is why that field is counted
    # rather than assumed.
    t2 = VM.midpoint_cut(s, 2)
    assert int((s >= t2).sum()) in (1, 4)


def test_the_degenerate_matches_are_numbers_not_raises():
    s = np.array([0.2, 0.4, 0.6])
    assert int((s >= VM.midpoint_cut(s, 0)).sum()) == 0
    assert int((s >= VM.midpoint_cut(s, 3)).sum()) == 3
    with pytest.raises(ValueError):
        VM.midpoint_cut(np.array([]), 0)


def test_passing_volume_honours_the_sites_own_comparison():
    s = np.array([0.4, 0.5, 0.6])
    assert VM.passing_volume(s, 0.5, strict=True) == 1
    assert VM.passing_volume(s, 0.5, strict=False) == 2


def test_match_cut_preserves_volume_and_reports_it_when_rounding_does_not():
    labels = np.array([1, 2, 3, 3, 4] * 20)
    base = np.linspace(0.0, 1.0, 100)
    cand = np.linspace(0.0, 1.0, 100) ** 2          # a different, monotone scale
    cut = VM.Cut("t", "owner", 0.5, strict=False, site="pool")
    got = VM.match_cut(cut, labels, base, cand, ndigits=6)
    assert got["matched_volume"] == VM.passing_volume(base, 0.5, strict=False)
    assert got["realized_volume"] == got["matched_volume"]
    assert got["volume_preserved"] is True
    # ...and a rounding coarse enough to cross the boundary is REPORTED, not hidden.
    coarse = VM.match_cut(cut, labels, base, cand, ndigits=0)
    assert coarse["realized_volume"] != coarse["matched_volume"]
    assert coarse["volume_preserved"] is False


def test_the_ladder_sweeps_every_marked_cut_at_its_exact_value():
    """`lock_mining_gate._row_at` refuses a nearest-bin match, so the record it reads has to
    contain the live cuts as exact rows or a flip cannot produce a lock at all."""
    labels = np.array([1, 2, 3] * 30)
    s = np.linspace(0, 1, 90)
    lad = VM.ladder(labels, s, {"a": 0.3402, "b": 0.6691}, strict=False)
    marked = {r["threshold"]: r["marks"] for r in lad if r["marks"]}
    assert marked == {0.3402: ["a"], 0.6691: ["b"]}
    assert [r["threshold"] for r in lad] == sorted(r["threshold"] for r in lad)


def test_recall_is_non_increasing_up_the_ladder():
    labels = np.array([1, 2, 3, 4] * 25)
    rng = np.random.default_rng(1)
    s = np.clip(labels / 4.0 + rng.normal(0, 0.2, len(labels)), 0, 1)
    rec = [r["recall"] for r in VM.ladder(labels, s, {}, strict=False)]
    assert all(a >= b for a, b in zip(rec, rec[1:]))


def test_version_dir_reads_the_head_version_not_the_parent_directory():
    """`parent.parent` is right for `v4b/seed_1/model_best.pt` and silently wrong for
    `v3/model_best.pt` — a record written one directory above the head it describes."""
    assert VM.version_dir(VM.WALLPAPER, "data/wallpaper_head/v4b/seed_1/model_best.pt") == \
        "data/wallpaper_head/v4b"
    assert VM.version_dir(VM.MINING, "data/render_mode_head/v3/model_best.pt") == \
        "data/render_mode_head/v3"
    with pytest.raises(ValueError):
        VM.version_dir(VM.MINING, "data/somewhere/else/model_best.pt")


# --------------------------------------------------------------------------- #
# 2. the committed records ARE what the owners serve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(VM.SPECS))
def test_the_committed_volume_match_record_places_the_live_cuts(key):
    """The record and `floors.py` cannot drift: every cut the pass placed is the value the
    owner serves, and every record is a COMPLETE pass."""
    spec = VM.SPECS[key]
    ver = VM.version_dir(spec, spec.incoming_rel).split("/")[-1] if key == "mining" else "v4b"
    rec = json.loads((ROOT / f"data/{spec.head_name}/{ver}/volume_match_{key}.json")
                     .read_text(encoding="utf-8"))
    assert rec["incomplete"] is False, "a bounded pass is not a basis for moving a constant"
    live = {f.name: f for f in F.ALL_FLOORS}
    for c in rec["cuts"]:
        assert c["incoming_value"] == live[c["name"]].value, c["name"]
        assert c["volume_preserved"] is True, c["name"]
        assert c["matched_volume"] == c["outgoing"]["n_selected"] == c["incoming"]["n_selected"]


@pytest.mark.parametrize("key", sorted(VM.SPECS))
def test_the_committed_volume_match_record_is_about_the_pinned_head(key):
    spec = VM.SPECS[key]
    ver = "v3" if key == "mining" else "v4b"
    rec = json.loads((ROOT / f"data/{spec.head_name}/{ver}/volume_match_{key}.json")
                     .read_text(encoding="utf-8"))
    live = F.active_head_version(spec.head_name)
    assert VM.version_dir(spec, rec["head"]["incoming"]).endswith(f"/{live}")
    assert VM.version_dir(spec, rec["head"]["outgoing"]) != \
        VM.version_dir(spec, rec["head"]["incoming"])


# --------------------------------------------------------------------------- #
# 3. the fit-slice sidecars keep the suggestion cuts derivable WITHOUT a GPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(RF.SPECS))
def test_every_fit_slice_sidecar_is_complete_and_aligned(key):
    spec = RF.SPECS[key]
    ver = "v3" if key == "mining" else "v4b"
    d = json.loads((ROOT / f"data/{spec.head_name}/{ver}/{spec.sidecar}")
                   .read_text(encoding="utf-8"))
    assert d["incomplete"] is False
    assert set(d["pred"]) == set(d["marg"]) == set(d["tier"])
    assert len(d["pred"]) == d["slice"]["n"]
    assert d["head"]["version"] == ver
    # `pred` IS 1 + sum(marg) — the readout `suggest_tier.expected_tier` defines, not a
    # second spelling of it.
    from tools.wallpaper.suggest_tier import expected_tier             # noqa: PLC0415
    for iid in list(d["pred"])[:50]:
        assert d["pred"][iid] == pytest.approx(expected_tier(d["marg"][iid]), abs=2e-6)


def test_the_sidecar_head_is_the_head_its_owner_pins():
    from tools.wallpaper import suggest_tier as ST                     # noqa: PLC0415
    from tools.mining import suggest_tier_mining as MT                 # noqa: PLC0415
    assert ST.INTAKE_PRED_SOURCES[ST.live_head_version()][0] == "sidecar"
    assert ST.JULY_PRED_SOURCES[ST.live_head_version()][0] == "sidecar"
    assert MT.PRED_SOURCES[MT.live_head_version()][0] == "sidecar"


def test_an_unregistered_head_RAISES_rather_than_fitting_to_another_heads_numbers():
    """The failure this replaces is silent: the batches still carry v3's stamped `pred`, so a
    deriver that just read it would have returned v3's cutpoints under v4b's name."""
    from tools.wallpaper import suggest_tier as ST                     # noqa: PLC0415
    from tools.mining import suggest_tier_mining as MT                 # noqa: PLC0415
    with pytest.raises(KeyError, match="no intake readout registered"):
        ST.intake_slice(head="v99")
    with pytest.raises(KeyError, match="no july readout registered"):
        ST.july_slice(head="v99")
    with pytest.raises(KeyError, match="no readout registered"):
        MT.fit_slice(head="v99")


@pytest.mark.parametrize("frozen,deriver", [
    ("CUTS", "derive_cuts"), ("INTAKE_CUTS", "derive_intake_cuts")])
def test_the_wallpaper_cuts_reproduce_from_their_sidecars(frozen, deriver):
    from tools.wallpaper import suggest_tier as ST                     # noqa: PLC0415
    assert getattr(ST, deriver)() == getattr(ST, frozen)


def test_the_mining_cuts_reproduce_from_their_sidecar():
    from tools.mining import suggest_tier_mining as MT                 # noqa: PLC0415
    assert MT.derive_cuts() == MT.CUTS


# --------------------------------------------------------------------------- #
# 4. the adoption records
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["wallpaper", "mining"])
def test_the_adoption_record_is_what_the_live_pins_and_owners_say(key):
    """Derive in code, freeze in records: the committed record is regenerated in memory and
    the live half compared. Only the live half — `generated` timestamps inside the cited
    evidence are the evidence's, not this record's."""
    from tools.scoring import adopt_head as A                          # noqa: PLC0415
    spec = A.SPECS[key]
    rec = A.build(spec)
    on_disk = json.loads(
        (ROOT / f"data/{spec.family}/{rec['adoption']}/adoption_record.json")
        .read_text(encoding="utf-8"))
    for k in ("adoption", "checkpoint", "pin", "moved_with_the_pin", "rollback_ladder"):
        assert on_disk[k] == rec[k], k


@pytest.mark.parametrize("key", ["wallpaper", "mining"])
def test_the_adoption_record_states_what_is_NOT_established(key):
    """An adoption that records only its case is a press release. Both of these flips carry a
    named limit — the wallpaper seed pick spends sheet D, and the mining per-mode cells are
    unshown in both directions — and the record has to carry it."""
    from tools.scoring import adopt_head as A                          # noqa: PLC0415
    rec = A.build(A.SPECS[key])
    assert rec["not_established"], key
    assert any("NOT ESTABLISHED" in s.upper() or "SPENDS" in s.upper()
               for s in rec["not_established"]), rec["not_established"]


@pytest.mark.parametrize("key", ["wallpaper", "mining"])
def test_the_adoption_record_refuses_a_cut_nobody_applied(key, monkeypatch):
    """Non-vacuity for the record's own agreement check: it may not claim a restatement that
    is not the value `floors.py` serves."""
    from tools.scoring import adopt_head as A                          # noqa: PLC0415
    spec = A.SPECS[key]
    A.build(spec)                                                      # non-vacuous
    name = spec.cut_names[0]
    live = {f.name: f for f in F.ALL_FLOORS}[name]
    moved = F.Floor(name=name, value=live.value + 0.01, head=live.head, stamp=live.stamp,
                    site=live.site, basis="injected")
    monkeypatch.setattr(F, "ALL_FLOORS",
                        tuple(moved if f.name == name else f for f in F.ALL_FLOORS))
    with pytest.raises(SystemExit, match="may not claim a restatement"):
        A.build(spec)
