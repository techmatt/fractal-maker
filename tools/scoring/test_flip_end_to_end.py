#!/usr/bin/env python
r"""END-TO-END PROOF OF THE LIVE FLIP — the deployed path, not a mock of it.

A flip that merely LANDS is not a flip that WORKS. What this proves, by driving the real
production machinery (`guard.make_guarded_scorer` on `ACTIVE_CKPT` -> `reframe.reframe_location`
-> `production_seeder._chosen_probs` -> `corn_decode` -> `Ledgers.append_outcome`) against
labeled class-4 and class-1 locations from the ACTIVE version's frozen eval slice:

  1. the loaded scorer IS the active checkpoint and IS K=4 (three cutpoint logits);
  2. a written ledger row is stamped with the ACTIVE version (`scorer_version`, via the
     ledger's own append);
  3. the row **records the third probability** (`p_ge4`), so it is re-decodable from disk;
  4. it **decodes correctly at the `>= 3` boundary** — a known class-4 location decodes to 4 and
     is admitted by the q3+ predicate and by the emission intake, and a known class-1 location
     decodes below the bar and is refused.

The failure this exists to catch is the plausible-looking row: correct-shaped, correctly
stamped, and structurally incapable of ever expressing class 4. That failure is invisible to
every unit test that supplies its own probabilities, which is why this one renders and scores
for real.

WAS `tools/v8/test_flip_end_to_end.py`, with "v8" hardcoded in the eval-slice path, the
manifest path, the score-column prefix and two stamp assertions — so the proof of a flip went
red at the NEXT flip, which is the one moment it most needs to run. It now resolves the version
from `production_pins.ACTIVE_VERSION` and proves whichever head is live.

`slow` — needs the release binary + a CUDA-capable torch and renders ~24 tiles per location.
    uv run pytest tools/scoring/test_flip_end_to_end.py -q -m slow
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "tools/mining", "tools/atlas", "tools/scoring", "tools/reframe", "tools/corpus"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))

sys.path.insert(0, str(ROOT / "tools" / "scoring"))
from production_pins import ACTIVE_VERSION  # noqa: E402

EVAL_SCORES = ROOT / "data" / ACTIVE_VERSION / f"eval_scores_{ACTIVE_VERSION}.jsonl"
MANIFEST = ROOT / "data" / ACTIVE_VERSION / "manifest.jsonl"
BIN = ROOT / "target/release/fractal-generator.exe"

# Ledger partition key for the manifest's fractal_type token.
FT2FAM = {"mandelbrot": "mandelbrot", "julia": "julia:mandelbrot",
          "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
          "julia_multibrot3": "julia:multibrot3", "julia_multibrot4": "julia:multibrot4",
          "julia_multibrot5": "julia:multibrot5", "phoenix": "phoenix"}
# Rust render-family token for a manifest fractal_type (julia_multibrotN -> julia_multibrotN).
RENDER_FAMILY = {"julia_multibrot3": "julia_multibrot3", "julia_multibrot4": "julia_multibrot4",
                 "julia_multibrot5": "julia_multibrot5", "mandelbrot": "mandelbrot",
                 "julia": "julia", "multibrot3": "multibrot3", "multibrot4": "multibrot4",
                 "multibrot5": "multibrot5", "phoenix": "phoenix"}


def _manifest():
    return {r["loc_id"]: r for r in
            (json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip())}


def _pick(label, key):
    """The eval-slice row with `label` maximizing/minimizing `key`, joined to its coords."""
    rows = [json.loads(l) for l in EVAL_SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand = [r for r in rows if r["label"] == label]
    assert cand, f"no label-{label} rows in {EVAL_SCORES}"
    best = max(cand, key=key)
    return best, _manifest()[best["location_id"]]


@pytest.fixture(scope="module")
def scorer():
    import active_ckpt
    import guard
    assert BIN.exists(), f"release binary missing: {BIN}"
    assert guard.SCORER_PATH == active_ckpt.ACTIVE_CKPT, "guard is not on the single source of truth"
    return guard.make_guarded_scorer()


def test_the_loaded_production_scorer_is_the_active_k4_checkpoint(scorer):
    import active_ckpt
    assert active_ckpt.ACTIVE_VERSION == ACTIVE_VERSION, (
        f"the pins module disagrees with itself: {active_ckpt.ACTIVE_VERSION} vs {ACTIVE_VERSION}")
    assert scorer.k == 4, f"deployed head is K={scorer.k}; a K=3 head cannot express class 4"
    assert int(scorer.cfg["num_classes"]) == 4
    # three cutpoint logits actually come out of the forward, not just out of the config
    from PIL import Image
    _, P = scorer.score_pils_k([Image.new("RGB", (1280, 720), (40, 60, 90))])
    assert P.shape[1] == 3, f"expected 3 cumulative probs, got {P.shape[1]}"


def _drive_production_path(scorer, mrow, tmp_path, monkeypatch):
    """Render + reframe + decode + append through the REAL production functions; return the row
    as read back off the ledger file."""
    import numpy as np
    import production_seeder as ps
    import reframe  # tools/reframe/reframe.py — the production reframe search
    import guard
    from score_lib import corn_decode

    ft = mrow["fractal_type"]
    partition = FT2FAM[ft]
    loc = reframe.Location(family=RENDER_FAMILY[ft], cx=mrow["cx"], cy=mrow["cy"], fw=mrow["fw"],
                           c_re=mrow.get("c_re"), c_im=mrow.get("c_im"), family_params={})
    res = reframe.reframe_location(loc, scorer=scorer, seed=0,
                                   workdir=tmp_path / f"rf_{mrow['loc_id']}", workers=4)

    # --- the ledger write path, verbatim from the discovery sites ---
    p_notbad, p_good, p_ge4 = ps._chosen_probs(res)
    t_good = ps.t_good_for(partition)
    guard_pass = res.score > guard.GUARD_SENTINEL + 1e-6
    decoded = corn_decode(p_notbad, p_good, t_good, p_ge4) if guard_pass else None
    row = {
        "id": f"e2e_{mrow['loc_id']}", "family": partition,
        "outcome_cx": float(res.cx), "outcome_cy": float(res.cy), "outcome_fw": float(res.fw),
        "k3": float(res.score), "decoded_class": decoded,
        "p_notbad": p_notbad, "p_good": p_good, "p_ge4": p_ge4, "t_good": t_good,
        "guard_pass": bool(guard_pass), "distinct": True,
    }
    # append through the ledger itself so `scorer_version` is stamped by production code, not here
    ledger_path = tmp_path / "outcome_ledger.jsonl"
    monkeypatch.setattr(ps, "OUTCOME_LEDGER", ledger_path)
    monkeypatch.setattr(ps, "OUTCOME_FEATS", tmp_path / "outcome_feats.npz")
    led = ps.Ledgers()
    led.append_outcome(row, np.zeros(1280, dtype="float32"))

    lines = [l for l in ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    return json.loads(lines[0]), partition


def test_a_class4_location_writes_an_active_stamped_q4_row(scorer, tmp_path, monkeypatch):
    import production_seeder as ps
    from score_lib import corn_decode
    from tools.emission import descriptor as desc

    _, mrow = _pick(4, key=lambda r: r[f"{ACTIVE_VERSION}_p_ge4"])
    row, partition = _drive_production_path(scorer, mrow, tmp_path, monkeypatch)

    # (2) stamped with the ACTIVE version, by the ledger's own append
    assert row["scorer_version"] == ACTIVE_VERSION, (
        f"ledger row is stamped {row['scorer_version']!r}, active is {ACTIVE_VERSION!r}")

    # (3) the third probability is on disk, and the row re-decodes from what was persisted
    assert row["p_ge4"] is not None, "the third probability was dropped — the row cannot ever be q4"
    assert 0.0 <= row["p_ge4"] <= 1.0
    assert corn_decode(row["p_notbad"], row["p_good"], row["t_good"], row["p_ge4"]) \
        == row["decoded_class"], "the persisted row does not re-decode to its own stamped class"

    # (4) the >= 3 boundary — a known-good class-4 location decodes to 4 and is admitted
    assert row["decoded_class"] == 4, (
        f"a human-labeled class-4 location decoded to {row['decoded_class']} "
        f"(p_notbad={row['p_notbad']:.4f} p_good={row['p_good']:.4f} p_ge4={row['p_ge4']:.4f} "
        f"t_good={row['t_good']}) — if this is 3, the pipeline is still q4-incapable")
    assert ps.is_q3plus(row), "a class-4 row must clear the q3+ admission predicate"
    assert desc.admit_quality(row), "a class-4 row must clear the emission intake predicate"
    assert ps.build_cloud([row], partition), "a class-4 row must enter the coverage cloud"


def test_a_class1_location_is_refused_at_the_same_boundary(scorer, tmp_path, monkeypatch):
    """The other half of the boundary. Without this, a decode that returned 4 unconditionally
    would pass the test above."""
    import production_seeder as ps
    from tools.emission import descriptor as desc

    # the most confidently-bad label-1 location
    _, mrow = _pick(1, key=lambda r: -r[f"{ACTIVE_VERSION}_score"])
    row, partition = _drive_production_path(scorer, mrow, tmp_path, monkeypatch)

    assert row["scorer_version"] == ACTIVE_VERSION
    assert row["p_ge4"] is not None            # recorded even when it does not promote
    assert row["decoded_class"] < 3, (
        f"a human-labeled class-1 location decoded to {row['decoded_class']} — the q3+ boundary "
        f"is not discriminating (p_notbad={row['p_notbad']:.4f} p_good={row['p_good']:.4f})")
    assert not ps.is_q3plus(row)
    assert not desc.admit_quality(row)
    assert not ps.build_cloud([row], partition)
