"""Guards for the v11 reader — the join, its refusals, and the v9/v10 read it must not move.

Everything here is CHEAP and synthetic: a five-row v11 cache and a two-row v10 cache written
into `tmp_path`, so the default lane exercises the schema logic without the 22.6 GiB bulk
cache. The expensive half — that the reader agrees with the tree actually on disk — is the
`slow`-marked test at the bottom, matching `tools/v11/test_v11_build.py`'s split.

The v10 test is the load-bearing one. v11 adds an adapter rather than editing `data_v4`
precisely so the rollback ladder's reads cannot move, and "cannot move" has to be something
a test says, not something the diff implies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import paths  # noqa: E402

from classifier import data_v4, data_v11  # noqa: E402

# --------------------------------------------------------------------------- #
# synthetic corpora
# --------------------------------------------------------------------------- #
V11_LOCS = [
    {"loc_id": 0, "label": 3, "split": "train", "group_id": 7, "source": "biased:b",
     "biased": True, "fractal_type": "mandelbrot", "eval_role": None, "split_group": 7},
    {"loc_id": 1, "label": 1, "split": "eval", "group_id": 8, "source": "prospect_census",
     "biased": False, "fractal_type": "julia_multibrot3", "eval_role": "instrument",
     "split_group": 8},
    {"loc_id": 2, "label": 4, "split": "eval", "group_id": 9, "source": "biased:c",
     "biased": True, "fractal_type": "phoenix", "eval_role": "holdout", "split_group": 9},
]


def _tile(loc_id, tile, palette, scale, shift, aa, q=85):
    return {"loc_id": loc_id, "tile": tile,
            "out": f"data/v11/aug_cache/{loc_id}/t{tile:02d}.jpg",
            "render": {"cx": "0.1", "cy": "0.2", "fw": "1e-3",
                       "fractal_type": "mandelbrot", "maxiter": 4000,
                       "maxiter_policy": "mi4000k0.3c200-67000"},
            "field": {"field_ss": 2, "pad_x": 103, "pad_y": 103},
            "crop": {"geom": 0, "scale": scale, "shift_frac": shift, "src_x0": 103,
                     "src_y0": 103, "ratio": 2},
            "tile_geom": {"w": 512, "h": 288},
            "aa": {"level": aa, "mode": "lanczos3" if aa == "antialiased" else "point"},
            "palette": palette, "jpg_quality": q}


@pytest.fixture
def v11_cache(tmp_path, monkeypatch):
    """A 3-location x 4-tile v11 corpus, with `paths.bulk` redirected at it.

    `monkeypatch`, never a bare env assignment: `artifacts_root()` reads
    FRACTAL_ARTIFACTS_ROOT at call time while importers bake paths at import time, so an
    unrestored value makes unrelated tests pass or fail on file order (CLAUDE.md)."""
    root = tmp_path / "arts"
    monkeypatch.setenv("FRACTAL_ARTIFACTS_ROOT", str(root))
    d = root / "data" / "v11"
    d.mkdir(parents=True)
    (d / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in V11_LOCS) + "\n", encoding="utf-8")
    tiles = []
    for lid in (0, 1, 2):
        tiles += [_tile(lid, 0, "twilight_shifted", 1, 0, "antialiased"),
                  _tile(lid, 1, "blue_orange", 1.07, 0.031, "aliased", 71),
                  _tile(lid, 2, "viridis", 0.94, 0.0, "antialiased", 60),
                  _tile(lid, 3, "twilight_shifted", 1.02, 0.02, "aliased", 95)]
    (d / "cache_manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in tiles) + "\n", encoding="utf-8")
    for lid in (0, 1, 2):
        td = root / "data" / "v11" / "aug_cache" / str(lid)
        td.mkdir(parents=True)
        for t in range(4):
            (td / f"t{t:02d}.jpg").write_bytes(b"x")
    return d


def _write_canon(d: Path, loc_ids, *, incomplete=False):
    (d / "eval_canon_manifest.jsonl").write_text("\n".join(json.dumps({
        "loc_id": i, "path": f"data/v11/eval_canon/{i}.jpg", "palette": "twilight_shifted",
        "aa_level": "antialiased", "aa_mode": "lanczos3", "jpg_quality": 85,
        "scale": 1.0, "shift_id": "center", "geom": 0, "maxiter": 4000,
        "batch_incomplete": incomplete}) for i in loc_ids) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #
def test_join_carries_label_split_and_the_two_new_columns(v11_cache):
    locs = data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)
    assert [l.location_id for l in locs] == [0, 1, 2]
    assert [len(l.renders) for l in locs] == [4, 4, 4]
    assert [l.label for l in locs] == [3, 1, 4]
    assert [l.split for l in locs] == ["train", "eval", "eval"]
    assert [l.eval_role for l in locs] == [None, "instrument", "holdout"]
    assert [l.split_group for l in locs] == [7, 8, 9]
    assert [l.biased for l in locs] == [True, False, True]


def test_render_axes_come_off_the_nested_blocks(v11_cache):
    locs = data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)
    r = {x.palette: x for x in locs[0].renders}
    assert r["blue_orange"].aa_level == "aliased"
    assert r["blue_orange"].scale == pytest.approx(1.07)
    assert r["viridis"].shift_id == "center"          # exact zero keeps the v4 vocabulary
    assert r["blue_orange"].shift_id == "sh0.0310"    # a continuum is spelled, not bucketed
    assert locs[0].renders[0].path == paths.bulk("data/v11/aug_cache/0/t00.jpg")


def test_eval_split_narrows_by_role(v11_cache):
    locs = data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)
    assert [l.location_id for l in data_v11.eval_split(locs)] == [1, 2]
    assert [l.location_id for l in data_v11.eval_split(locs, "instrument")] == [1]
    assert [l.location_id for l in data_v11.eval_split(locs, "holdout")] == [2]


def test_short_fan_out_is_refused(v11_cache):
    """A location silently short of its tiles has a different augmentation distribution
    from every other one — the failure the count assertion exists for."""
    p = v11_cache / "cache_manifest.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    p.write_text("\n".join(json.dumps(r) for r in rows if not (r["loc_id"] == 1
                                                               and r["tile"] == 3)) + "\n",
                 encoding="utf-8")
    with pytest.raises(SystemExit, match="do not carry 4 tiles"):
        data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)


def test_cache_row_without_a_manifest_row_is_refused(v11_cache):
    p = v11_cache / "cache_manifest.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_tile(99, 0, "twilight_shifted", 1, 0, "antialiased")) + "\n")
    with pytest.raises(SystemExit, match="different builds"):
        data_v11.load_locations_v11(canon_path=None, tiles_per_location=None)


# --------------------------------------------------------------------------- #
# the canonical view — the one axis v11's independent draw does not guarantee
# --------------------------------------------------------------------------- #
def test_canonical_raises_when_no_canon_manifest_is_loaded(v11_cache):
    locs = data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)
    with pytest.raises(data_v11.NoCanonicalRender, match="build_eval_canon"):
        locs[1].canonical()


def test_canonical_comes_from_the_eval_canon_manifest(v11_cache):
    _write_canon(v11_cache, [1, 2])
    locs = data_v11.load_locations_v11(tiles_per_location=4, verify_paths=False)
    c = locs[1].canonical()
    assert c.path == paths.bulk("data/v11/eval_canon/1.jpg")
    assert (c.palette, c.aa_level, c.scale, c.shift_id) == \
        ("twilight_shifted", "antialiased", 1.0, "center")
    with pytest.raises(data_v11.NoCanonicalRender):
        locs[0].canonical()      # train-side: no canonical rendered, and none claimed


def test_incomplete_canon_manifest_is_refused(v11_cache):
    """`build_eval_canon --limit` writes REAL files and stamps every row; the stamp is what
    separates a bounded rehearsal from an artifact a certification may read."""
    _write_canon(v11_cache, [1], incomplete=True)
    with pytest.raises(SystemExit, match="batch_incomplete"):
        data_v11.load_locations_v11(tiles_per_location=4, verify_paths=False)


def test_palette_renders_and_aa_twin_say_why_they_are_gone(v11_cache):
    locs = data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)
    with pytest.raises(NotImplementedError, match="independently"):
        locs[0].palette_renders()
    with pytest.raises(NotImplementedError, match="per tile"):
        locs[0].aa_twin()


def test_missing_tile_files_are_reported(v11_cache):
    (paths.bulk("data/v11/aug_cache/2/t02.jpg")).unlink()
    with pytest.raises(FileNotFoundError, match="cache JPGs missing"):
        data_v11.load_locations_v11(canon_path=None, tiles_per_location=4)


# --------------------------------------------------------------------------- #
# the v9/v10 read, unmoved
# --------------------------------------------------------------------------- #
@pytest.mark.version_pinned
def test_v10_flat_schema_still_reads_through_data_v4(tmp_path):
    """v4..v10's FLAT single-file cache manifest still loads, with the canonical /
    palette-renders / aa-twin selectors all working. v9 and v10 are the rollback rungs; the
    v11 adapter is additive precisely so this cannot move."""
    cache = tmp_path / "cache_manifest.jsonl"
    (tmp_path / "aug_roster.json").write_text(json.dumps(
        {"palettes": {"always": ["twilight_shifted", "blue_orange"],
                      "drawn_per_location": 0}}), encoding="utf-8")
    rows = []
    for lid, label in ((0, 2), (1, 3)):
        for pal in ("twilight_shifted", "blue_orange"):
            for aa, ss, filt in (("antialiased", 2, "lanczos3"), ("aliased", 1, "box")):
                rows.append({"location_id": lid, "label": label, "split": "eval",
                             "group_id": 100 + lid, "source": "prospect_census",
                             "biased": False, "palette": pal, "palette_family": pal,
                             "scale": 1.0, "shift_id": "center", "geom_id": "id",
                             "aa_level": aa, "ss": ss, "filter": filt, "maxiter": 5358,
                             "fractal_type": "julia",
                             "path": f"data/v9/aug_cache/{lid}/{pal}__{aa}.jpg"})
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    locs = data_v4.load_locations(cache_path=cache, verify_paths=False)
    assert [l.location_id for l in locs] == [0, 1]
    assert [len(l.renders) for l in locs] == [4, 4]
    assert [l.label for l in locs] == [2, 3]
    assert locs[0].canonical().palette == data_v4.NEUTRAL_PALETTE
    assert locs[0].canonical().aa_level == "antialiased"
    assert [r.palette for r in locs[0].palette_renders()] == \
        ["blue_orange", "twilight_shifted"]
    assert locs[0].aa_twin().aa_level == "aliased"
    assert data_v4.hist(locs) == {1: 0, 2: 1, 3: 1, 4: 0}


def test_v11_reader_does_not_import_into_data_v4():
    """The adapter is one-way. `data_v4` must not learn about v11, or the rollback rungs'
    read path grows a dependency on an artifact family that did not exist when they shipped."""
    src = (ROOT / "classifier" / "data_v4.py").read_text(encoding="utf-8")
    assert "v11" not in src and "data_v11" not in src


# --------------------------------------------------------------------------- #
# the real tree
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_reader_agrees_with_the_built_v11_cache():
    """The counts the build record claims, read back through the reader off the real bulk
    cache. `slow` because it parses 361,696 rows."""
    rec = json.loads((ROOT / "data" / "v11" / "build_record.json").read_text(
        encoding="utf-8"))["population"]
    locs = data_v11.load_locations_v11(canon_path=None, verify_paths=False)
    assert len(locs) == rec["manifest_rows"]
    assert all(len(l.renders) == 32 for l in locs)
    assert sum(1 for l in locs if l.split == "train") == rec["train"]
    assert sum(1 for l in locs if l.eval_role == "instrument") == rec["eval_instrument"]
    assert sum(1 for l in locs if l.eval_role == "holdout") == rec["eval_holdout"]
    for cls, n in rec["class_train"].items():
        assert sum(1 for l in locs if l.split == "train" and l.label == int(cls)) == n
