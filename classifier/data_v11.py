"""Location reader for the v11 corpus — the two-file join v4..v10 never needed.

`data_v4.load_locations` reads ONE self-describing file: every v4..v10 cache-manifest row
carries both the render axes (`palette`, `scale`, `shift_id`, `aa_level`, `path`) and the
location's training metadata (`label`, `split`, `group_id`, `source`, `biased`,
`fractal_type`), repeated on all 24 (or 42) of a location's rows. v11 split that in two, and
both halves of the split are deliberate:

  * `data/v11/cache_manifest.jsonl` — 361,696 tile rows, **render only**, and NESTED
    (`render.*`, `crop.*`, `aa.*`, `tile_geom.*`). It is written by the Rust engine, one row
    per tile, and knows nothing about labels. Repeating an 11-field label block 32x across
    361,696 rows would also be ~40 MB of duplicated join key.
  * `data/v11/manifest.jsonl` — 11,303 location rows: label, split, group, source, biased,
    fractal_type, plus v11's two new columns `eval_role` and `split_group`.

So this module JOINS them on `loc_id` and returns the same `Loc`/`Render` objects
`data_v4` returns, which is what lets `LocationDataset`, `make_weighted_sampler` and `hist`
be reused verbatim rather than reimplemented. **`data_v4` is not edited**: v9/v10 reads are
the rollback ladder and the `version_pinned` lane, and the cheapest way to keep them
byte-for-byte is to not touch the file they go through.

THE CANONICAL VIEW IS NOT IN THE CACHE, and that is the one real behavioural difference.
v4..v10 fanned out as a PRODUCT, so `twilight_shifted x identity x antialiased` existed for
every location. v11 draws each of its 32 tiles independently; the floors put
`twilight_shifted` + identity framing at tile 0, but tile 0's AA level is a 50/50 draw, so
only 1,448 of the 2,860 eval locations carry a canonical tile. `tools/v11/build_eval_canon.py`
produces the cell for the whole eval slice by replaying those tile-0 rows at
`antialiased:lanczos3`/q85 (verified byte-identical where the cache already had it), and
`LocV11.canonical()` reads THAT manifest. A location with no canonical render raises rather
than silently substituting an aliased tile — the AA level is a held axis the model is tested
for invariance under, not a nuisance to average over.

`palette_renders()` / `aa_twin()` are NOT available on v11 locations and say so. Under an
independent draw there is no "the ss4 render of each palette at the canonical geometry" —
a location's 32 tiles are 32 different geometries. The palette-invariance battery renders
the held-out palettes fresh instead (`tools/v11/eval_v11.py`), which is what v10's battery
already did for the held-out set.

    from classifier.data_v11 import load_locations_v11
    locs = load_locations_v11()                     # 11,303 Loc, 32 renders each
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import paths  # noqa: E402

from .data_v4 import Loc, Render  # noqa: E402

CACHE_MANIFEST = "data/v11/cache_manifest.jsonl"
MANIFEST = "data/v11/manifest.jsonl"
CANON_MANIFEST = "data/v11/eval_canon_manifest.jsonl"

# The deploy-canonical cell, as `build_eval_canon` writes it.
NEUTRAL_PALETTE = "twilight_shifted"
CANON_SCALE = 1.0
CANON_SHIFT = "center"


class NoCanonicalRender(RuntimeError):
    """This location has no deploy-canonical render loaded.

    Raised at the read site, naming the location and the fix, because the silent
    alternatives are both wrong: substituting the location's aliased tile 0 scores half an
    eval slice through a point-sampled image, and dropping the location shrinks an
    instrument whose whole value is being a fixed population."""


@dataclass
class LocV11(Loc):
    """A v11 location: `Loc` plus the split columns v11 added, and a canonical render that
    comes from the eval-canon manifest rather than from the cache."""
    eval_role: str | None = None      # "instrument" | "holdout" | None (train)
    split_group: int | None = None    # the leakage-closure group the split draw ran over
    canon: Render | None = None
    _tiles: list[int] = field(default_factory=list)

    def canonical(self) -> Render:
        if self.canon is None:
            raise NoCanonicalRender(
                f"loc {self.location_id}: no canonical render — v11's independent tile draw "
                f"does not guarantee one, and the eval-canon manifest ({CANON_MANIFEST}) "
                f"either was not loaded or does not cover this location. Build it with "
                f"`uv run python tools/v11/build_eval_canon.py` (eval slice only).")
        return self.canon

    def palette_renders(self):
        raise NotImplementedError(
            "v11 draws its 32 tiles independently, so there is no per-palette render at a "
            "shared geometry to compare — the palette-invariance battery renders the "
            "held-out palettes fresh (tools/v11/eval_v11.py).")

    def aa_twin(self):
        raise NotImplementedError(
            "v11 draws AA per tile, so a location has no aliased twin of its canonical "
            "render at the same geometry; the AA axis is a corpus-level draw, not a "
            "per-location pair.")


def _rows(rel: str):
    p = paths.bulk(rel)
    if not p.exists():
        raise SystemExit(f"missing {rel} -> {p}  (rebuild: see data/v11/build_record.json)")
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _shift_id(crop: dict) -> str:
    """v11's shift is a continuous draw, not v4's three-value `shift_id` vocabulary. Keep
    `center` for the exact-zero case (so a caller filtering on it still means what it meant)
    and spell the magnitude otherwise, rather than bucketing a continuum into a lie."""
    return "center" if crop["shift_frac"] == 0 else f"sh{crop['shift_frac']:.4f}"


def load_locations_v11(cache_path: str = CACHE_MANIFEST,
                       manifest_path: str = MANIFEST,
                       canon_path: str | None = CANON_MANIFEST,
                       verify_paths: bool = True,
                       tiles_per_location: int | None = 32) -> list[LocV11]:
    """Join the v11 manifest and cache manifest into one `LocV11` per location.

    `canon_path=None` skips the canonical attachment entirely (training does not need it;
    only selection and certification do). `tiles_per_location=None` disables the fan-out
    count assertion — pass it when reading a `--limit`-stamped partial cache, and nowhere
    else: a location silently short of its tiles is a location whose augmentation
    distribution differs from every other one's."""
    by_id: dict[int, LocV11] = {}
    for r in _rows(manifest_path):
        lid = int(r["loc_id"])
        by_id[lid] = LocV11(
            location_id=lid, label=int(r["label"]), split=r["split"],
            group_id=int(r["group_id"]), source=r["source"], biased=bool(r["biased"]),
            fractal_type=r["fractal_type"], palette_spec=None,
            eval_role=r.get("eval_role"), split_group=r.get("split_group"),
        )

    for r in _rows(cache_path):
        loc = by_id.get(int(r["loc_id"]))
        if loc is None:
            raise SystemExit(f"cache row for loc_id {r['loc_id']} has no manifest row — "
                             f"{cache_path} and {manifest_path} are from different builds")
        crop = r["crop"]
        loc.renders.append(Render(
            path=paths.bulk(r["out"]), palette=r["palette"], palette_family=r["palette"],
            scale=float(crop["scale"]), shift_id=_shift_id(crop),
            aa_level=r["aa"]["level"],
        ))
        loc._tiles.append(int(r["tile"]))

    locs = [by_id[k] for k in sorted(by_id)]
    if tiles_per_location is not None:
        bad = [(l.location_id, len(l.renders)) for l in locs
               if len(l.renders) != tiles_per_location]
        if bad:
            raise SystemExit(f"{len(bad)} locations do not carry {tiles_per_location} "
                             f"tiles, e.g. {bad[:5]} — partial or mismatched cache")
    dup = [l.location_id for l in locs if len(set(l._tiles)) != len(l._tiles)]
    if dup:
        raise SystemExit(f"{len(dup)} locations have duplicate tile indices, e.g. {dup[:5]}")

    if canon_path is not None:
        attach_canonical(locs, canon_path)
    if verify_paths:
        missing = [str(rr.path) for lc in locs for rr in lc.renders if not rr.path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cache JPGs missing, e.g. {missing[0]}")
    return locs


def attach_canonical(locs: list[LocV11], canon_path: str = CANON_MANIFEST) -> int:
    """Attach the eval-canon render to every location the manifest covers. Returns the count.

    Refuses a manifest stamped `batch_incomplete` — `build_eval_canon --limit` writes real
    files and stamps every row it produces, so the stamp is the difference between a bounded
    rehearsal and the artifact a certification may read."""
    p = paths.bulk(canon_path)
    if not p.exists():
        raise SystemExit(f"missing {canon_path} -> {p}  (uv run python "
                         f"tools/v11/build_eval_canon.py)")
    by_id = {l.location_id: l for l in locs}
    n = 0
    for r in _rows(canon_path):
        if r.get("batch_incomplete"):
            raise SystemExit(f"{canon_path} is stamped batch_incomplete — it was written by "
                             f"a --limit run and covers only a prefix of the eval slice")
        loc = by_id.get(int(r["loc_id"]))
        if loc is None:
            continue
        loc.canon = Render(path=paths.bulk(r["path"]), palette=r["palette"],
                           palette_family=r["palette"], scale=float(r["scale"]),
                           shift_id=r["shift_id"], aa_level=r["aa_level"])
        n += 1
    return n


def eval_split(locs: list[LocV11], role: str | None = None) -> list[LocV11]:
    """The eval-side locations, optionally narrowed to one `eval_role`.

    The two roles are NOT interchangeable and the build record says why: `instrument` is the
    four score-unconditioned registrations (unbiased — base rates and version-over-version
    reads come off these only), `holdout` is a stratified random draw over the remaining
    split groups (biased exactly as training is — what a calibration cut needs and what a
    base rate must never be read from)."""
    out = [l for l in locs if l.split == "eval"]
    return out if role is None else [l for l in out if l.eval_role == role]
