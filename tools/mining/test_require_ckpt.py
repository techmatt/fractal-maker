"""A dead checkpoint pin must raise, naming itself — never silently fall back.

Both surviving readers of the `enrich --mode score` stream default to a checkpoint
that no longer exists (`score_lib.DEFAULT_V3` = v3, `enrich_score.MODEL_ID` = v2;
`data/classifier/` holds v5..v9). The pins are kept deliberately — they are the
provenance record of what those batches were scored with — so the failure mode has
to be a loud, specific error rather than a repoint. This pins that decision:

  1. the pins still name their historical versions (repointing them is the failure);
  2. the pinned files really are absent (otherwise 3. is vacuous);
  3. `require_ckpt` raises on them, and the message names the missing path;
  4. `require_ckpt` returns the path when it *does* exist, so the guard is not a
     blanket refusal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))


def _score_lib():
    pytest.importorskip("torch")
    import score_lib
    return score_lib


def _enrich_score():
    pytest.importorskip("torch")
    import enrich_score
    return enrich_score


@pytest.mark.parametrize("mod_get,attr,expected", [
    (_score_lib, "DEFAULT_V3", "data/classifier/v3/model_best.pt"),
    (_enrich_score, "MODEL_ID", "data/classifier/v2/model_best.pt"),
])
def test_pin_is_not_repointed(mod_get, attr, expected):
    """The pin is a record. Moving it to a live version is what this forbids."""
    assert getattr(mod_get(), attr) == expected


@pytest.mark.parametrize("rel", ["data/classifier/v2/model_best.pt",
                                 "data/classifier/v3/model_best.pt"])
def test_the_pinned_checkpoints_really_are_absent(rel):
    """Non-vacuity: if v2/v3 ever come back, the raise-tests below stop proving anything."""
    assert not (ROOT / rel).exists(), (
        f"{rel} exists again — the require_ckpt tests are now vacuous; revisit whether the "
        "pins should still be unrunnable-by-design")


@pytest.mark.parametrize("mod_get,attr", [(_score_lib, "DEFAULT_V3"),
                                          (_enrich_score, "MODEL_ID")])
def test_require_ckpt_raises_naming_the_missing_checkpoint(mod_get, attr):
    mod = mod_get()
    pin = getattr(mod, attr)
    with pytest.raises(SystemExit) as e:
        mod.require_ckpt(pin)
    msg = str(e.value)
    assert pin in msg, f"the error does not name the missing checkpoint: {msg!r}"
    assert "not found" in msg


@pytest.mark.parametrize("mod_get", [_score_lib, _enrich_score])
def test_require_ckpt_passes_an_existing_path_through(mod_get, tmp_path):
    """Not a blanket refusal — a real checkpoint path is returned unchanged."""
    f = tmp_path / "model_best.pt"
    f.write_bytes(b"")
    assert mod_get().require_ckpt(str(f)) == str(f)
