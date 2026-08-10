r"""hue_families.py — THE interpretable hue/flavor family of a pool palette.

WHY THIS EXISTS AND WHY IT IS NOT A FOURTH CLASSIFIER. Three palette groupings already
live in the tree and none of them answers the question Matt asked of sheet B ("few greens;
purple / fire / ice heavy"):

  * `palette_categories.json` k8/k12/k16 — the emission `palette_flavor` axis. A ward cut
    over a learned feature space: it is the right axis for the deficit model and it is
    unreadable as a sentence, because cell `k16:11` has no name.
  * `mining/palette_families.classify` — 5-way cyclic/diverging/mono/warm/cool. Built for
    the spread ROSTER; `warm`/`cool` pool green with blue and purple with red, which is
    exactly the distinction being asked about.
  * `emission/palette_deficit._hsv_signature` — the 12-bin chroma-weighted hue histogram
    the deficit tracker steers on. THE signature convention, and this module's whole
    numeric content: hue families here are that histogram's bins, GROUPED and NAMED.

So this adds a naming layer and no new arithmetic. `_hsv_signature` is imported, never
re-derived — a second copy of the hue convention is how two "green" definitions appear.

THE FAMILIES, in the order a report prints them:

    fire    hue bins 0-1     0-60deg    red / orange / amber
    gold    hue bin  2      60-90deg    yellow / chartreuse
    green   hue bins 3-5   90-180deg    green / spring          (== palette_deficit.GREEN_BINS)
    ice     hue bins 6-8  180-270deg    cyan / azure / blue
    purple  hue bins 9-10 270-330deg    violet / magenta
    rose    hue bin  11   330-360deg    pink / crimson

plus two families taken STRAIGHT from the committed `palette_categories.json` `special`
prepull rather than re-decided here, because that artifact is what the emission cells are
cut over and a palette that is "spectral" there must not be "fire" here:

    neutral   special == neutral   (15 palettes: greys, near-achromatic ramps)
    spectral  special == spectral  (57 palettes: hue spread >= 0.82 — rainbow)

`outlier` is NOT a family: it is a nearest-neighbour-distance prepull, orthogonal to hue,
and its 46 members carry perfectly ordinary hues. They classify by hue like anything else.

A palette absent from `palette_categories.json` (a built-in, a one-off `.ugr` block) has no
`special` verdict, so it classifies on hue alone and `family_of` says so through
`known_special=False` — silently treating "no record" as "not special" is how a spectral
palette lands in `fire`.

    from tools.palettes.hue_families import family_of, families_over_pool, FAMILIES

    uv run python tools/palettes/hue_families.py        # the pool distribution
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission.palette_deficit import (            # noqa: E402  THE signature convention
    GREEN_BINS, HUE_BINS, _hsv_signature, lut_from_stops,
)

POOL = ROOT / "data" / "palettes" / "pool_colormaps.json"
CATEGORIES = ROOT / "data" / "palettes" / "palette_categories.json"

# Report order: the three families Matt named as over-served first is NOT the order —
# the order is the hue wheel, so a reader can see which arc is thin.
FAMILIES = ("fire", "gold", "green", "ice", "purple", "rose", "spectral", "neutral")

# family -> the `_hsv_signature` hue bins it pools. Hue-only families; the two specials
# are decided before this table is consulted.
HUE_GROUPS = {
    "fire": (0, 1),
    "gold": (2,),
    "green": tuple(GREEN_BINS),
    "ice": (6, 7, 8),
    "purple": (9, 10),
    "rose": (11,),
}
SPECIAL_FAMILIES = {"neutral": "neutral", "spectral": "spectral"}


def _check_groups():
    """Every hue bin belongs to exactly one family. A gap or an overlap would make the
    family shares stop summing to 1 while every individual number still looked fine."""
    seen = Counter(b for bins in HUE_GROUPS.values() for b in bins)
    missing = [b for b in range(HUE_BINS) if b not in seen]
    dup = [b for b, n in seen.items() if n > 1]
    if missing or dup:
        raise AssertionError(f"HUE_GROUPS does not partition {HUE_BINS} hue bins: "
                             f"missing={missing} duplicated={dup}")


_check_groups()


def load_specials(categories_path=CATEGORIES) -> dict:
    """`{palette_name: special}` from the committed categories artifact ('chromatic' for
    the ordinary 869). Absent file -> empty, and every caller then reports `known_special`
    False rather than inventing a verdict."""
    p = Path(categories_path)
    if not p.exists():
        return {}
    cats = json.loads(p.read_text(encoding="utf-8"))["palettes"]
    return {name: entry.get("special") for name, entry in cats.items()}


def family_from_signature(sig: dict) -> tuple:
    """`(family, share)` from a hue signature alone — the argmax hue GROUP and the share of
    chroma-weighted hue mass it holds. `share` is the confidence: a palette at 0.9 is that
    family, one at 0.25 is a blend the argmax happens to win."""
    hh = sig["hue"]
    mass = {f: float(sum(hh[b] for b in bins)) for f, bins in HUE_GROUPS.items()}
    fam = max(sorted(mass), key=lambda f: mass[f])
    return fam, mass[fam]


def family_of(name: str, stops, specials: dict | None = None) -> dict:
    """The full verdict for one palette: family, hue share, the raw signature scalars, and
    whether the special prepull was actually consulted.

    `specials` is `load_specials()`'s map; pass it in so a whole-pool pass reads the
    artifact once."""
    sig = _hsv_signature(lut_from_stops(stops))
    special = (specials or {}).get(name)
    known = specials is not None and name in specials
    if special in SPECIAL_FAMILIES:
        fam, share = SPECIAL_FAMILIES[special], None
    else:
        fam, share = family_from_signature(sig)
    return {"palette": name, "family": fam, "hue_share": share,
            "special": special, "known_special": known,
            "green": sig["green"], "mean_chroma": sig["mean_chroma"],
            "spread": sig["spread"],
            "hue_hist": [float(x) for x in sig["hue"]]}


def load_pool(pool_path=POOL) -> list:
    return json.loads(Path(pool_path).read_text(encoding="utf-8"))


def families_over_pool(pool_path=POOL, categories_path=CATEGORIES) -> dict:
    """`{palette_name: verdict}` over the whole production pool (987 today)."""
    specials = load_specials(categories_path)
    return {p["name"]: family_of(p["name"], p["stops"], specials) for p in load_pool(pool_path)}


def share_table(names, verdicts: dict) -> dict:
    """`{family: {n, share}}` over an iterable of palette NAMES (repeats count — a served
    sheet is a multiset, and the whole question is which families it over-serves).

    Names with no verdict are counted under `unknown` rather than dropped: a silently
    shorter denominator is how a bias measurement understates itself."""
    counts = Counter()
    for n in names:
        v = verdicts.get(n)
        counts[v["family"] if v else "unknown"] += 1
    total = sum(counts.values()) or 1
    order = list(FAMILIES) + (["unknown"] if counts.get("unknown") else [])
    return {f: {"n": counts.get(f, 0), "share": counts.get(f, 0) / total} for f in order}


def main() -> int:
    verdicts = families_over_pool()
    tab = share_table(verdicts, verdicts)          # the pool itself: one row per palette
    print(f"pool {len(verdicts)} palettes -> hue/flavor families")
    for f, c in tab.items():
        ex = [n for n, v in sorted(verdicts.items()) if v["family"] == f][:4]
        print(f"  {f:<9} {c['n']:>4}  {c['share']*100:5.1f}%   e.g. {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
