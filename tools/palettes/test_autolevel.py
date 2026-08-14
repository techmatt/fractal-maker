"""The band auto-level operator + its reference record: the properties the wiring rests on.

Five things are asserted here, in the order the prompt that staged this asked for them:
identity-in-band comes back as the base render's own bytes, a stamp replays to the same stop
list without the image, the reference record re-derives, the switch is the ONLY way in, and
the wired call sites reach the operator through that one entry.

The switch SHIPS ON since the 2026-08-11 adoption; what was "the shipped default" here is now
two separate contracts — the default is ON and is checked against the adoption record, and the
OFF path stays exercisable through `FRACTAL_AUTOLEVEL=0` because it is how a before/after pair
is produced.

Light lane: numpy + PIL + the committed colormap pool. No engine, no GPU, no torch.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import colormap as cm                                   # noqa: E402
from tools.palettes import adopt_autolevel as ADOPT                # noqa: E402
from tools.palettes import autolevel as AL                         # noqa: E402
from tools.palettes import levels_reference as LR                  # noqa: E402

POOL = ROOT / "data/palettes/pool_colormaps.json"


# --------------------------------------------------------------------------- #
# fixtures — a synthetic render and a reference built AROUND it, so "in band" is a
# property of the fixture rather than a coincidence of the committed record.
# --------------------------------------------------------------------------- #
def _ramp(h=64, w=64) -> np.ndarray:
    """A neutral ramp: measurable black point, full range, mid near the middle."""
    g = np.linspace(0.0, 1.0, h * w).reshape(h, w)
    return (np.stack([g] * 3, -1) * 255).round().astype(np.uint8)


def _reference_around(st: dict, *, pad=0.02, shift=None) -> dict:
    """A reference record whose bands are `stat ± pad` (so the render is IN band), optionally
    with one statistic's band shifted away by `shift = {stat: delta}` (so it is OUT)."""
    shift = shift or {}
    bands = {}
    for k in AL.STAT_KEYS:
        v = st[k] if st[k] is not None else st["black_pt_all"]
        d = shift.get(k, 0.0)
        bands[k] = {"band": [max(0.0, v - pad + d), min(1.0, v + pad + d)], "n": 3}
    return {"schema": "levels_reference/v1", "version": "test_ref", "derived": "2026-01-01",
            "n_images": 3, "_sha256": "0" * 64, "_path": "test", "bands": bands}


@pytest.fixture(scope="module")
def entry() -> dict:
    lib = json.loads(POOL.read_text(encoding="utf-8"))
    return next(c for c in lib if c.get("stops"))


class _Boom:
    """A rerender that must not be called."""

    def __init__(self):
        self.calls = []

    def __call__(self, stops):
        self.calls.append(stops)
        raise AssertionError("rerender was called when it must not have been")


# --------------------------------------------------------------------------- #
# 1. The switch.
# --------------------------------------------------------------------------- #
def test_the_switch_ships_on():
    """ADOPTED 2026-08-11 (`data/palettes/autolevel_adoption.json`). This is the production
    reality the rest of the suite is written against; the OFF path below is a contract that
    survives the flip, not the default."""
    assert AL.SWITCH_DEFAULT is True
    assert AL.enabled() is True


def test_the_adoption_record_matches_the_live_switch_and_reference():
    """The flip's record is only a record while it agrees with the tree. Both halves matter:
    a switch reverted without re-running the writer leaves a record claiming a state the tree
    does not have, and a re-derived band leaves the record adopting a band nobody adopted —
    which is a NEW adoption question, not a file to quietly refresh
    (`uv run python tools/palettes/adopt_autolevel.py --write`)."""
    rec = json.loads((ROOT / ADOPT.OUT_REL).read_text(encoding="utf-8"))
    ref = AL.load_reference()
    assert rec["switch"]["default_now"] is AL.SWITCH_DEFAULT
    assert rec["adoption"] == AL.OPERATOR_VERSION
    assert rec["reference_record"]["sha256"] == ref["_sha256"]
    assert rec["reference_record"]["bands"] == {k: list(v) for k, v in AL.bands(ref).items()}


@pytest.mark.parametrize("val,want", [("1", True), ("true", True), ("ON", True),
                                      ("0", False), ("no", False), ("", False),
                                      ("perhaps", True)])
def test_the_switch_is_read_at_call_time_and_a_typo_never_moves_it(monkeypatch, val, want):
    """Read at call time (never at import), so a run sets it without editing source — and an
    unparseable value falls back to the shipped default rather than to a state of its own.
    Before the flip that read as "a typo cannot turn it ON"; after it, the same rule is what
    stops a typo turning a production colorize OFF."""
    monkeypatch.setenv(AL.SWITCH_ENV, val)
    assert AL.enabled() is want


def test_switch_off_is_the_pre_operator_path(monkeypatch, entry):
    """The OFF-STATE CONTRACT, unchanged by the flip and now reached explicitly: with
    `FRACTAL_AUTOLEVEL=0` the operator is one boolean read in front of the old behaviour —
    the base image comes back as the SAME object, no stamp, no reference load, no rerender."""
    monkeypatch.setenv(AL.SWITCH_ENV, "0")
    img = _ramp()
    lev = AL.maybe_level(img, entry, _Boom())
    assert lev.img is img and lev.stamp is None and lev.acted is False


# --------------------------------------------------------------------------- #
# 2. Identity in band, through the operator's one entry.
# --------------------------------------------------------------------------- #
def test_identity_in_band_returns_the_base_renders_own_bytes(monkeypatch, entry):
    """The whole point of a band. All three statistics inside -> the curve is exactly the
    identity, the re-render never happens, and the caller gets the bytes it already had.
    Byte-identity here is structural (the same array), not a tolerance."""
    monkeypatch.setenv(AL.SWITCH_ENV, "1")
    img = _ramp()
    ref = _reference_around(AL.tone_stats(img))
    lev = AL.maybe_level(img, entry, _Boom(), reference=ref)
    assert lev.img is img
    assert lev.stamp["acted"] is False and lev.stamp["curve"]["identity"] is True
    assert lev.stamp["chroma_cap"]["n_capped"] == 0


def test_out_of_band_re_renders_once_with_curved_stops(monkeypatch, entry):
    monkeypatch.setenv(AL.SWITCH_ENV, "1")
    img = _ramp()
    st = AL.tone_stats(img)
    ref = _reference_around(st, shift={"mid": +0.25})     # the render's midtone now sits BELOW
    seen = {}

    def rerender(stops):
        seen["stops"] = stops
        return np.zeros_like(img)

    lev = AL.maybe_level(img, entry, rerender, reference=ref)
    assert lev.acted and lev.stamp["curve"]["sides"]["mid"] == -1
    assert seen["stops"] and lev.stamp["chroma_cap"]["n_stops"] == len(seen["stops"])
    assert lev.img is not img


# --------------------------------------------------------------------------- #
# 3. Replay + provenance.
# --------------------------------------------------------------------------- #
def test_the_stamp_replays_to_the_same_stops_without_the_image(monkeypatch, entry):
    """A release replays byte-identically off its record: the stop list rebuilds from the
    stamp's curve alone — no render, no re-measurement."""
    monkeypatch.setenv(AL.SWITCH_ENV, "1")
    img = _ramp()
    ref = _reference_around(AL.tone_stats(img), shift={"white_pt": +0.20})
    seen = {}
    lev = AL.maybe_level(img, entry, lambda s: seen.setdefault("stops", s) or np.zeros_like(img),
                         reference=ref)
    assert lev.acted
    assert AL.stops_from_stamp(lev.stamp, entry) == seen["stops"]


def test_an_identity_stamp_refuses_to_replay_a_stop_list(monkeypatch, entry):
    monkeypatch.setenv(AL.SWITCH_ENV, "1")
    img = _ramp()
    lev = AL.maybe_level(img, entry, _Boom(), reference=_reference_around(AL.tone_stats(img)))
    with pytest.raises(ValueError, match="IDENTITY"):
        AL.stops_from_stamp(lev.stamp, entry)


def test_the_stamp_carries_the_reference_identity_and_the_before(monkeypatch, entry):
    """Reference-record version + sha + band, the whole curve, and the BEFORE render's own
    statistics — the two things a stamped row has to support."""
    monkeypatch.setenv(AL.SWITCH_ENV, "1")
    img = _ramp()
    st = AL.tone_stats(img)
    ref = _reference_around(st, shift={"mid": +0.25})
    lev = AL.maybe_level(img, entry, lambda s: np.zeros_like(img), reference=ref)
    r = lev.stamp["reference"]
    assert (r["version"], r["sha256"], r["n_images"]) == ("test_ref", "0" * 64, 3)
    assert set(r["bands"]) == set(AL.STAT_KEYS)
    assert lev.stamp["measured"]["mid"] == pytest.approx(st["mid"])
    assert lev.stamp["operator"] == AL.OPERATOR_VERSION and lev.stamp["switch"] == "on"


def test_one_stamp_row_is_logged_per_leveled_render(monkeypatch, entry, tmp_path):
    """Logged by the operator itself, so "every row produced with the operator on is stamped"
    is true by construction rather than by each call site remembering."""
    monkeypatch.setenv(AL.SWITCH_ENV, "1")
    img = _ramp()
    ref = _reference_around(AL.tone_stats(img))
    for k in ("a.jpg", "b.jpg"):
        AL.maybe_level(img, entry, _Boom(), key=k, log_dir=tmp_path, reference=ref)
    rows = [json.loads(l) for l in
            (tmp_path / AL.STAMP_LOG).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["key"] for r in rows] == ["a.jpg", "b.jpg"]
    assert all(r["autolevel"]["operator"] == AL.OPERATOR_VERSION for r in rows)


# --------------------------------------------------------------------------- #
# 4. The LUT seam is untouched.
# --------------------------------------------------------------------------- #
def test_the_override_library_bakes_the_palettes_own_lut_for_its_own_stops(entry):
    """`OverrideLibrary` must be a stop swap and nothing else: handed the palette's OWN stops
    it produces the library's own LUT, bit for bit. If this drifts, the operator has become a
    second bake and the Rust<->Python seam is no longer the same map."""
    lib = cm.PaletteLibrary(colormaps_path=str(POOL))
    name = entry["name"]
    mirror = bool(entry.get("mirror_needed"))
    ovr = AL.OverrideLibrary(lib, name, [(p, rgb) for p, rgb in entry["stops"]], mirror)
    for reverse in (False, True):
        assert np.array_equal(ovr.lut(name, reverse=reverse), lib.lut(name, reverse=reverse))
    assert ovr.palette_type(name) == lib.palette_type(name)


def test_the_one_entry_colormaps_file_changes_only_the_stop_colours(entry, tmp_path):
    """The Rust-side application. Name, `cycle` and `mirror_needed` ride along unchanged, so
    the engine's bake and mirror decision are bit-identical to the production call."""
    stops = [[0.0, [1, 2, 3]], [0.5, [4, 5, 6]]]
    p = AL.one_entry_colormaps(entry, stops, tmp_path / "cm.json")
    got = json.loads(p.read_text(encoding="utf-8"))
    assert len(got) == 1 and got[0]["stops"] == stops
    for k in ("name", "cycle", "mirror_needed"):
        if k in entry:
            assert got[0][k] == entry[k]


# --------------------------------------------------------------------------- #
# 5. The reference record.
# --------------------------------------------------------------------------- #
def test_the_committed_record_loads_and_carries_all_three_bands():
    ref = AL.load_reference()
    assert ref["schema"] == LR.SCHEMA and len(ref["_sha256"]) == 64
    b = AL.bands(ref)
    assert set(b) == set(AL.STAT_KEYS)
    assert all(lo < hi for lo, hi in b.values())
    assert ref["n_images"] == len(ref["per_image"]) >= 35


def test_the_bands_re_derive_from_the_records_own_per_image_stats():
    """The regeneration check that needs no images: the committed bands must be exactly what
    `bands_from_per_image` computes over the record's own per-image block. A hand-edit to
    either half fails here."""
    doc = LR.committed()
    ok, bad = LR.self_consistent(doc)
    assert ok, f"bands do not follow from per_image for {bad}"


def test_a_hand_edited_band_is_caught():
    """The same check in the other direction (verification_practice.md §3): move one edge and
    the record must stop verifying."""
    doc = json.loads(json.dumps(LR.committed()))
    doc["bands"]["mid"]["band"][0] += 0.05
    ok, bad = LR.self_consistent(doc)
    assert not ok and bad == ["mid"]


def test_verify_reports_unknown_when_the_source_folder_is_unreachable(tmp_path, capsys):
    """A verification tool that cannot reach its authority reports UNKNOWN, not OK and not
    DRIFT (CLAUDE.md). The self-check still runs and is named as partial."""
    rc = LR.verify(src=tmp_path / "no_such_folder")
    out = capsys.readouterr().out
    assert rc == LR.UNKNOWN
    assert "UNKNOWN" in out and "partial verification" in out


@pytest.mark.slow
@pytest.mark.skipif(not LR.SOURCE_DIR.exists(),
                    reason="the read-only LevelsCheck folder is not on this machine")
def test_the_record_re_derives_byte_identically_from_the_source():
    """The full verification, opt-in because it reads 48 wallpapers off Matt's Desktop."""
    assert LR.verify() == LR.OK


# --------------------------------------------------------------------------- #
# 6. The wiring: one entry, and the direct family unreachable.
# --------------------------------------------------------------------------- #
WIRED = ["tools/mining/deploy_tail.py", "tools/emission/build_emission_diversity_v1.py"]


@pytest.mark.parametrize("rel", WIRED)
def test_the_wired_sites_reach_the_operator_only_through_maybe_level(rel):
    """`maybe_level` is THE switch. A call site that reached `plan`/`derive_band_curve`/
    `curved_stops` directly would be a second switch, and the shipped-OFF guarantee would
    then be one edit away from being false in one file and true in the others."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"derive_band_curve", "curved_stops", "apply_curve_L", "tone_stats", "enabled"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id == "AL" and node.attr in banned), \
                f"{rel} reaches AL.{node.attr} directly; go through maybe_level"


def test_the_direct_trap_family_is_unreachable_by_construction():
    """Palette-indifferent modes have no LUT for the operator to act on, so they are excluded
    where the KIND is known — not left to a flag inside the render."""
    src = (ROOT / "tools/mining/deploy_tail.py").read_text(encoding="utf-8")
    assert 'level=(kind != "direct")' in src
    # The signature half is asked of the FUNCTION, not of the file's text. A literal
    # `def render_rust(...)` string went red the first time an unrelated keyword-only
    # parameter was added beside `level` — a source scan is the right tool for the dispatch
    # expression above (which is a claim about how the call site is written) and the wrong one
    # for a signature, which `inspect` answers exactly (`verification_practice.md` §9).
    import inspect                                       # noqa: PLC0415
    from tools.mining import deploy_tail as dt           # noqa: PLC0415
    p = inspect.signature(dt.render_rust).parameters["level"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True


def test_the_switch_off_render_info_is_byte_identical_to_the_pre_operator_block():
    """With no stamp there is no key, so every record the wired passes write under the
    switch forced OFF is exactly what they wrote before the operator existed."""
    from tools.mining import deploy_tail as dt           # noqa: PLC0415  (renders on import path)
    before = {"transfer_dropped": False}
    assert dt._info(dict(before), None) == before
    assert dt._info(dict(before), AL.Leveled(np.zeros((2, 2, 3), np.uint8), None)) == before
    stamped = dt._info(dict(before), AL.Leveled(np.zeros((2, 2, 3), np.uint8), {"acted": True}))
    assert stamped == {**before, "autolevel": {"acted": True}}
