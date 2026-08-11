r"""split_units.py — THE location-disjoint split for the render-mode corpus, once.

A location-grouped split is not enough for this corpus, and the reason is specific to the
families in it: a Julia location's seed `c = (c_re, c_im)` IS a point in its parent family's
plane, so a Julia child and the parameter-plane location sitting on that same point are the
SAME piece of the fractal seen two ways. Split them across train and eval and the head is
evaluated on geometry it trained on, while every location-disjointness check passes.

So the split unit is a UNION-FIND COMPONENT, not a location:

  * every location starts as its own node;
  * each Julia-family location is unioned with a synthetic node for its
    `(parent_family, c_re, c_im)` seed;
  * a base-family location sitting exactly on such a seed is unioned with that same node.

Components are then family-stratified (a component's family is a base-family member's if it
has one, else the modal family) and a seeded `EVAL_FRAC` share of the COMPONENTS — never of
the raw locations — goes to eval.

This is `build_scale_sample.build_split` as coded in July, lifted here unchanged so it has
ONE home: it was written for the scale batch, the fresh corpus needs the identical rule, and
a second copy is how two corpora end up with two split designs that are supposed to be one.
`build_scale_sample.py` now imports it. `test_mining_sheet.py` fails on a second copy.

Coordinates are compared through `_fkey` (12 significant digits) rather than as strings: the
same point reaches this function as a decimal string from one row and a float-formatted one
from another, and `"0.25" != "0.250"` would silently split a unit in half.

    from tools.mining.split_units import build_split, JULIA_PARENT
"""
from __future__ import annotations

import collections

import numpy as np

# Julia family -> the base family whose c-plane its seed lives in.
JULIA_PARENT = {"julia": "mandelbrot", "julia_multibrot3": "multibrot3",
                "julia_multibrot4": "multibrot4", "julia_multibrot5": "multibrot5"}

EVAL_FRAC = 0.40      # the July scale batch's stamped fraction — reused, not re-chosen
SPLIT_SEED = 0


class UF:
    """Union-find with path halving. Nodes are created on first `find`."""

    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _fkey(v, ndig: int = 12) -> str:
    """Float-tolerant key for a coordinate that may arrive as a decimal string or a float."""
    return f"{float(v):.{ndig}g}"


def build_split(locs: dict, *, seed: int = SPLIT_SEED, eval_frac: float = EVAL_FRAC,
                force_eval=()):
    """`locs`: `{location_key: row}` with `row["family"]` and `row["render"]`.

    Returns `(side, meta)` — `side` is `{location_key: "train"|"eval"}` and `meta` reports the
    component structure so a caller can print what the union-find actually did (how many
    multi-location units formed, how many base parents were linked) rather than assert it
    happened.

    `force_eval` — location keys that must land on the eval side whatever the draw says
    (`tools/corpus/eval_only.py`: a batch stamped `eval_only`). The pin is applied at UNIT
    granularity, not per location: forcing one member of a component and letting the draw
    place the rest is precisely the straddle this module exists to prevent. A forced unit is
    then withheld from its family's draw, so `eval_frac` keeps meaning "of the units this
    rule got to choose". Empty by default and the code path is inert when empty — the rng is
    consumed identically, so an existing corpus's split is byte-identical."""
    uf = UF()
    for k in locs:
        uf.find(k)

    seed_nodes = {}                       # (pfam, ckr, cki) -> synthetic node id

    def seed_node(pfam, cr, ci):
        key = (pfam, _fkey(cr), _fkey(ci))
        return seed_nodes.setdefault(key, f"seed::{pfam}::{key[1]}::{key[2]}"), key

    # base-family locations indexed by their own (cx, cy) — the plane a Julia seed lives in
    base_pts = collections.defaultdict(dict)
    for k, r in locs.items():
        fam, rd = r["family"], r["render"]
        if fam not in JULIA_PARENT:
            base_pts[fam][(_fkey(rd["cx"]), _fkey(rd["cy"]))] = k

    for k, r in locs.items():
        fam, rd = r["family"], r["render"]
        if fam in JULIA_PARENT:
            if rd.get("c_re") is None or rd.get("c_im") is None:
                # A Julia row with no seed cannot be linked to its parent, and quietly
                # leaving it as a singleton is the exact leak this module exists to stop.
                raise ValueError(f"{k}: family {fam} with no c_re/c_im — its parent point is "
                                 f"unknown, so it cannot be placed in a split unit")
            node, _ = seed_node(JULIA_PARENT[fam], rd["c_re"], rd["c_im"])
            uf.union(k, node)

    linked_parents = 0
    for (pfam, ckr, cki), node in seed_nodes.items():
        pk = base_pts.get(pfam, {}).get((ckr, cki))
        if pk is not None:
            uf.union(pk, node)
            linked_parents += 1

    comp = collections.defaultdict(list)
    for k in locs:
        comp[uf.find(k)].append(k)
    units = list(comp.values())

    def unit_family(members):
        fams = [locs[m]["family"] for m in members]
        base = [f for f in fams if f not in JULIA_PARENT]
        return collections.Counter(base or fams).most_common(1)[0][0]

    forced = set(force_eval)
    unknown_forced = sorted(forced - set(locs))
    forced_units = [u for u in units if forced.intersection(u)]

    rng = np.random.default_rng(seed)
    strata = collections.defaultdict(list)
    for members in units:
        if forced.intersection(members):
            continue                       # pinned below, and withheld from the draw
        strata[unit_family(members)].append(tuple(sorted(members)))

    side, n_eval_units = {}, 0
    for members in forced_units:
        for m in members:
            side[m] = "eval"
    n_eval_units += len(forced_units)
    for fam in sorted(strata):
        us = sorted(strata[fam])
        order = rng.permutation(len(us))
        n_ev = int(round(eval_frac * len(us)))
        ev = set(order[:n_ev].tolist())
        for i, members in enumerate(us):
            s = "eval" if i in ev else "train"
            n_eval_units += (1 if i in ev else 0)
            for m in members:
                side[m] = s

    meta = {
        "rule": "union-find over locations linked by Julia-seed == parent-plane point; "
                "family-stratified seeded draw over UNITS, not locations",
        "seed": seed, "eval_frac": eval_frac,
        "n_locations": len(locs), "n_units": len(units),
        "n_multi_loc_units": sum(1 for u in units if len(u) > 1),
        "largest_unit": max((len(u) for u in units), default=0),
        "linked_base_parents": linked_parents,
        "n_eval_units": n_eval_units,
        "n_forced_eval_keys": len(forced),
        "n_forced_eval_units": len(forced_units),
        "n_locations_pinned_by_force": sum(len(u) for u in forced_units),
        "forced_keys_not_in_this_pool": unknown_forced[:20],
        "n_eval_loc": sum(1 for s in side.values() if s == "eval"),
        "n_train_loc": sum(1 for s in side.values() if s == "train"),
    }
    return side, meta


def units_are_disjoint(side: dict, locs: dict) -> tuple:
    """`(ok, message)` — every union-find component lands wholly on one side.

    A CHECK the caller runs on the split it built, not a claim this module makes: the
    property is what the corpus needs, and asserting it against a freshly recomputed
    component set is the only way to notice a future edit that breaks it."""
    uf = UF()
    for k in locs:
        uf.find(k)
    for k, r in locs.items():
        fam, rd = r["family"], r["render"]
        if fam in JULIA_PARENT and rd.get("c_re") is not None:
            uf.union(k, f"seed::{JULIA_PARENT[fam]}::{_fkey(rd['c_re'])}::{_fkey(rd['c_im'])}")
    base_pts = collections.defaultdict(dict)
    for k, r in locs.items():
        if r["family"] not in JULIA_PARENT:
            base_pts[r["family"]][(_fkey(r["render"]["cx"]), _fkey(r["render"]["cy"]))] = k
    for k, r in locs.items():
        if r["family"] in JULIA_PARENT and r["render"].get("c_re") is not None:
            pfam = JULIA_PARENT[r["family"]]
            pk = base_pts.get(pfam, {}).get((_fkey(r["render"]["c_re"]),
                                             _fkey(r["render"]["c_im"])))
            if pk is not None:
                uf.union(pk, k)
    spans = {}
    for k in locs:
        spans.setdefault(uf.find(k), set()).add(side.get(k))
    bad = {root: sides for root, sides in spans.items() if len(sides) > 1}
    return (not bad), (f"{len(bad)} unit(s) span both sides: {list(bad)[:3]}" if bad
                       else f"{len(spans)} units, each wholly on one side")
