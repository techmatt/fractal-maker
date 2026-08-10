r"""rare_palette_draw.py — draw palettes that target the RARE hue families.

THE MEASURED PROBLEM this exists to fix (2026-08-10, `scratch/sittings_27c_report.md`,
appendix `palette_bias.json`). Hue/flavor family shares, `tools/palettes/hue_families`:

    family     987-pool   sheet-A proposed   sheet-A served   sheet-B served
    fire         44.4%        39.2%              39.2%            31.3%
    ice          32.8%        32.4%              30.6%            20.3%
    purple        4.7%         6.0%              11.7%            45.8%
    green         7.6%        10.8%               8.6%             2.4%
    spectral      5.8%         9.0%               7.3%             0.2%
    rose          2.3%         0.9%               0.3%             0.0%
    neutral       1.5%         0.9%               1.2%             0.0%
    gold          0.9%         0.8%               1.0%             0.0%

Two different failures wearing one symptom. Sheet A's is a PICK: the head's argmax lifts
purple to 2.5x its proposal share and pushes rose to a tenth of the pool's. Sheet B's is a
POPULATION: its universe is 38 palettes frozen into `gate_passers_v3.json`, so four families
are not merely thin, they are ABSENT and no draw could have found them.

WHAT THIS MODULE DOES. It picks the palette for a unit against a DECLARED family target
instead of against supply, so a family that is 0.9% of the pool can still be 4% of a sheet.
Two levels, and the split is the point:

  * BETWEEN families — `RARE_TARGET`, a declared share vector, realized through
    `apportion.sequence_by_deficit` so every PREFIX of the draw is near-target (a sheet cut
    short by a render budget is still rare-palette-weighted, which a floor-then-remainder
    take does not give you).
  * WITHIN a family — least-used-first, then `palette_deficit.pick` on the intrinsic
    signature. `palette_deficit` is the built, documented "restore green/high-chroma"
    mechanism (`--palette-pick deficit`); it is imported, not reimplemented.

WHY `RARE_TARGET` IS DECLARED AND NOT DERIVED. Every derivable target reproduces the thing
being corrected: proportional-to-pool IS the fire/ice bias (82% of the pool is those two
plus purple), and uniform-over-families hands 12.5% to a 9-palette family, i.e. three
repeats per distinct palette. The vector below is the decision the prompt asked for —
"explicitly down-weight purple / fire / ice; oversample green and whatever the bias
measurement names as under-served" — with each number bounded by DISTINCT-PALETTE SUPPLY so
a ~250-draw sheet covers the small families nearly exhaustively instead of cycling three
palettes. The supply column is what makes it a design rather than a preference:

    family    distinct   target   draws at N=250   distinct-coverage at N=250
    green         75      0.28          70            ~93% of the family
    spectral      57      0.16          40            ~70%
    ice          324      0.14          35            (down-weighted from 20-33%)
    fire         438      0.15          38            (down-weighted from 31-39%)
    rose          23      0.09          22            ~96%
    purple        46      0.08          20            (down-weighted from 45.8%)
    neutral       15      0.06          15            100%
    gold           9      0.04          10            100%

The three families Matt named heavy total 37% here, against 82% of the pool, 84.7% of
sheet A and 97.4% of sheet B.

    from tools.mining.rare_palette_draw import PaletteDrawer, RARE_TARGET
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apportion                                             # noqa: E402  THE two draw rules
from tools.emission import palette_deficit as PD             # noqa: E402  THE deficit pick
from tools.palettes import hue_families as HF                # noqa: E402  THE family naming

# Declared share of the sheet's palette draws, by hue/flavor family. See the module doc for
# where each number comes from and what bounds it. Sums to 1.0 (asserted below).
RARE_TARGET = {
    "green": 0.28,
    "spectral": 0.16,
    "fire": 0.15,
    "ice": 0.14,
    "rose": 0.09,
    "purple": 0.08,
    "neutral": 0.06,
    "gold": 0.04,
}

# The families this draw is deliberately cutting, named so a report can say so without
# re-deriving it from the vector.
DOWN_WEIGHTED = ("purple", "fire", "ice")
OVER_DRAWN = ("green", "spectral", "rose", "neutral", "gold")

# The measurement the target is aimed at, frozen as DATA so the tests can assert the
# direction of every number instead of re-reading two label batches. Re-derive with
#   uv run python tools/mining/measure_palette_bias.py
# `pool` is what exists; `served_a` / `served_b` are what the (27) sheets put in front of
# Matt (960 and 1000 rows). NOTE that the target is NOT "below the pool share" for purple:
# purple is only 4.7% of the pool and 45.8% of sheet B, so the thing being cut is the
# SERVED share. A test written against the pool column asserts the wrong claim.
MEASURED_2026_08_10 = {
    "pool":     {"fire": 0.444, "ice": 0.328, "green": 0.076, "spectral": 0.058,
                 "purple": 0.047, "rose": 0.023, "neutral": 0.015, "gold": 0.009},
    "served_a": {"fire": 0.392, "ice": 0.306, "green": 0.086, "spectral": 0.073,
                 "purple": 0.117, "rose": 0.003, "neutral": 0.013, "gold": 0.010},
    "served_b": {"fire": 0.313, "ice": 0.203, "green": 0.024, "spectral": 0.002,
                 "purple": 0.458, "rose": 0.000, "neutral": 0.000, "gold": 0.000},
}


def _check_target():
    if set(RARE_TARGET) != set(HF.FAMILIES):
        raise AssertionError(f"RARE_TARGET families {sorted(RARE_TARGET)} != "
                             f"hue_families.FAMILIES {sorted(HF.FAMILIES)}")
    s = sum(RARE_TARGET.values())
    if abs(s - 1.0) > 1e-9:
        raise AssertionError(f"RARE_TARGET sums to {s}, not 1.0")


_check_target()


def family_counts(n: int, supply: dict, target: dict = RARE_TARGET) -> dict:
    """`{family: draws}` summing to `min(n, total supply-with-repeats)`.

    A family is capped at its DISTINCT-palette supply — repeats inside a family are allowed
    (two locations may share a palette) but a family is never asked for more than it has
    distinct members, because past that point the draw stops buying variety and the whole
    point of the target is variety. The shortfall is redistributed over the families that
    are still under their cap, in target proportion, until nothing moves.

    Returns counts, never shares, so the caller can print `n` beside every number."""
    fams = [f for f in HF.FAMILIES if supply.get(f)]
    want = {f: 0 for f in fams}
    remaining = int(n)
    live = dict.fromkeys(fams, True)
    while remaining > 0 and any(live.values()):
        wsum = sum(target[f] for f in fams if live[f])
        if wsum <= 0:
            break
        moved = 0
        for f in fams:
            if not live[f]:
                continue
            room = supply[f] - want[f]
            add = min(room, int(round(remaining * target[f] / wsum)))
            if add <= 0:
                if room <= 0:
                    live[f] = False
                continue
            want[f] += add
            moved += add
            if want[f] >= supply[f]:
                live[f] = False
        if moved == 0:
            # remainder smaller than every rounded share: hand the rest out one at a time,
            # largest deficit first, so `n` is met exactly instead of being silently short.
            for f in apportion.sequence_by_deficit({k: v for k, v in want.items() if live[k]}):
                if remaining - moved <= 0:
                    break
                if want[f] < supply[f]:
                    want[f] += 1
                    moved += 1
            if moved == 0:
                break
        remaining -= moved
    return {f: want[f] for f in HF.FAMILIES if want.get(f)}


class PaletteDrawer:
    """Deterministic rare-family palette sequence + a within-family deficit pick.

    Usage is two calls, in this order, because the family SEQUENCE is planned for the whole
    draw (so every prefix is near-target) while the concrete palette is chosen unit by unit
    (so the running colour deficit can move):

        d = PaletteDrawer(n_draws=250, seed=...)
        for i, unit in enumerate(units):
            name, fam = d.take()
    """

    def __init__(self, n_draws: int, seed: int = 0, target: dict = RARE_TARGET,
                 pool_path=HF.POOL, categories_path=HF.CATEGORIES):
        self.verdicts = HF.families_over_pool(pool_path, categories_path)
        pool = {p["name"]: p for p in HF.load_pool(pool_path)}
        self.sigs = {n: PD._hsv_signature(PD.lut_from_stops(pool[n]["stops"]))
                     for n in self.verdicts}
        self.members = {}
        for name, v in self.verdicts.items():
            self.members.setdefault(v["family"], []).append(name)
        for f in self.members:
            self.members[f].sort()
        self.supply = {f: len(m) for f, m in self.members.items()}
        self.target = dict(target)
        self.n_draws = int(n_draws)
        self.counts = family_counts(self.n_draws, self.supply, self.target)
        # `sequence_by_deficit` takes counts and its tie-break is a pure function of them, so
        # the family sequence is reproducible from (n_draws, target, supply) alone.
        self.sequence = apportion.sequence_by_deficit(dict(sorted(self.counts.items())))
        self.prefix_deviation = apportion.prefix_deviation(self.sequence, self.counts)
        self.tracker = PD.DeficitTracker()
        self.used = Counter()
        self.rng = np.random.default_rng([int(seed), 7])
        self._i = 0

    # -- the draw ----------------------------------------------------------- #
    def take(self) -> tuple:
        """`(palette_name, family)` for the next unit. Raises when the planned sequence is
        exhausted — a silent wrap would serve the first families twice and skip the last."""
        if self._i >= len(self.sequence):
            raise IndexError(f"PaletteDrawer exhausted after {len(self.sequence)} draws "
                             f"(planned for {self.n_draws})")
        fam = self.sequence[self._i]
        self._i += 1
        name = self._pick_in(fam)
        self.used[name] += 1
        # The tracker wants a RENDER's realized histogram; at draw time there is no render, so
        # it is fed the palette's INTRINSIC signature. That is the same proxy
        # `palette_deficit`'s own diagnosis rests on (realized mean_chroma p50 0.310 vs
        # intrinsic 0.332 — chroma survives the recipe), and it is what makes successive picks
        # inside one family spread instead of repeating the family's argmax.
        s = self.sigs[name]
        # `ingest` truth-tests `realized["hue_hist"]`, so it must arrive as a LIST — handing
        # it a numpy array raises "truth value of an array is ambiguous" rather than being
        # silently wrong, which is the good failure, but it still has to be converted here.
        self.tracker.ingest({"hue_hist": list(map(float, s["hue"])),
                             "chroma_hist": list(map(float, s["chroma"]))})
        return name, fam

    def _pick_in(self, fam: str) -> str:
        """Least-used members first, then the deficit pick among them.

        Least-used-first is what turns a 15-palette family into 15 distinct palettes instead
        of one palette fifteen times: with no usage term the deficit gain of the family's
        best filler barely moves after one pick, so it wins again."""
        members = self.members[fam]
        least = min(self.used[m] for m in members)
        pool = [m for m in members if self.used[m] == least]
        j = PD.pick(pool, None, self.sigs, self.tracker)
        return pool[j]

    # -- reporting ---------------------------------------------------------- #
    def report(self) -> dict:
        drawn = Counter(self.verdicts[n]["family"] for n in self.used.elements())
        n = sum(drawn.values()) or 1
        return {
            "n_planned": self.n_draws,
            "n_taken": self._i,
            "target_shares": self.target,
            "down_weighted": list(DOWN_WEIGHTED),
            "over_drawn": list(OVER_DRAWN),
            "distinct_supply": {f: self.supply.get(f, 0) for f in HF.FAMILIES},
            "planned_counts": {f: self.counts.get(f, 0) for f in HF.FAMILIES},
            "drawn_counts": {f: drawn.get(f, 0) for f in HF.FAMILIES},
            "drawn_shares": {f: drawn.get(f, 0) / n for f in HF.FAMILIES},
            "distinct_palettes_used": len(self.used),
            "family_distinct_used": {f: len({m for m in self.used
                                             if self.verdicts[m]["family"] == f})
                                     for f in HF.FAMILIES},
            "max_repeats": max(self.used.values()) if self.used else 0,
            "sequence_prefix_deviation": self.prefix_deviation,
            "sequence_rule": "apportion.sequence_by_deficit over the planned family counts — "
                             "every prefix near-target, which is what survives a truncating "
                             "render budget",
            "within_family_rule": "least-used members first, then palette_deficit.pick "
                                  "(deficit gain, no pref head) on intrinsic signatures",
        }
