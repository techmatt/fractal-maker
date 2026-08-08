#!/usr/bin/env python
r"""v11 location manifest — a FRESH FULL REBUILD with randomized location-grouped splits.

TWO departures from v10, both deliberate and both Matt's calls.

**1. Not an append.** `classifier_retrain_protocol.md` §1's frozen-prefix rule buys
version-to-version eval comparability by keeping the prior manifest's rows byte-identical.
v11 gives that up on purpose, because the thing it changes IS the split rule: a frozen
prefix and a re-randomized split cannot both hold. What comparability survives is carried
by the four INSTRUMENTS, which are reproduced location-for-location and stay 100% eval
(GATE 6 pins the census-144 by identity) — the same substitution v8 made when there was no
prefix to inherit. `loc_id` is dense over this build and carries no cross-version meaning;
nothing reuses a v9/v10 tile, because the v11 recipe renders a different fan-out into its
own tree.

**2. Randomized location-grouped splits** (Matt, 2026-08-06; the v11 calibration default).
v10's eval side was the instruments and nothing else — 1,050 locations over four sources,
none of them covering `julia:mandelbrot`, `phoenix`, or the native multibrots. That is why
`derive_t_good`'s `MIN_POS` gate blocks six partitions while the CORPUS holds 809
julia:mandelbrot and 375 phoenix positives: the positives exist, and no eval slice contains
any of them. So v11 adds a seeded random holdout over the rest of the corpus, and the eval
side becomes two populations with two jobs:

    eval_role = "instrument"  the four score-unconditioned draws, forced 100% eval.
                              UNBIASED. Base rates and version-over-version non-regression
                              read off THESE and only these.
    eval_role = "holdout"     a stratified random draw over the remaining split groups.
                              BIASED, exactly as biased as training is, and that is the
                              point: it is a held-out sample of the population the model is
                              trained on, so a threshold calibrated on it is calibrated on
                              the distribution it will be applied to. It is NOT a base rate
                              and must never be reported as one.

`biased` stays on every row, so a consumer that needs the unbiased population filters for
it rather than trusting the split.

THE SPLIT UNIT is a **split group**: the transitive closure of every leakage relation we
know how to name, so that "children inherit their seed's split" (§2) holds by construction
rather than by a gate that fires afterwards. Three relations union the spatial groups:

    (a) shared minibrot atom — two views of ONE nucleus at different maneuver `k` differ by
        4x in `fw` and are invisible to the spatial union-find, which unions only within
        1.5x. The key is DERIVED from each row's own render block (`atom_identity`), not
        read from a provenance column six batches happened to write.
    (b) shared dynamical plane — two viewports on ONE Julia/phoenix picture, which the
        spatial union-find splits whenever the frames are far apart. This is v8 GATE 7's
        seed-`c` claim turned from a check into a relation the draw obeys.
    (c) shared `parent_oid` — an explicit parent-child record between two locations.

The randomized draw then assigns WHOLE split groups, so no relation above can straddle.

`assign_groups` below is v8's neighbourhood predicate with ONE fix, and (b) is why it was
needed: v8 buckets by the seed `c` alone, so phoenix's 500-row parameter grid — one
viewport, 500 different planes — lands in a single spatial group. Closing (b) over those
groups then swallowed 1,029 of 1,098 phoenix locations into one split group and left the
holdout able to take 5 of them. Refining the bucket by the exact non-`c` parameter axes
repartitions phoenix and nothing else, and phoenix's holdout goes 5 -> 113 (55 positives).

  uv run python tools/v11/build_manifest.py [--dry-run] [--eval-share 0.20]

Writes:
  bulk   data/v11/manifest.jsonl     the population + split (out-of-tree; 11k rows)
  bulk   data/v11/eval_slice.jsonl   the eval side, both roles
  durable data/v11/build_record.json the SMALL committed record: the config that
                                     reproduces the two bulk files, plus realized counts
All gates are ABORTS, not warnings (protocol §2, "manifest-build gates abort-all").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT))
import atom_identity as ai   # noqa: E402  THE derived atom key
import label_store as ls     # noqa: E402
import location as loc_mod   # noqa: E402
import paths                 # noqa: E402
import batch_registry as br  # noqa: E402
from partitions import partition_of_row  # noqa: E402

# v8's build is IMPORTED, not copied: the clustering predicate, the fail-closed registry
# read and the location fold are the same code every prior manifest realized. Only the
# split rule below is v11's.
from tools.v8 import build_manifest as v8b  # noqa: E402

BATCHES_GLOB = str(ROOT / "data" / "label_corpus" / "batches" / "*" / "images.jsonl")
OUT = "data/v11/manifest.jsonl"
EVAL_OUT = "data/v11/eval_slice.jsonl"
RECORD_OUT = "data/v11/build_record.json"

# ---- the split draw -------------------------------------------------------- #
SPLIT_SEED = 20260808          # fixes the holdout for the whole build; in the record
EVAL_SHARE = 0.20              # target eval share of the NON-instrument population
# Strata for the holdout draw. Both axes matter and neither is a score:
#   fractal_type — every partition must reach `derive_t_good`'s MIN_POS on its own slice,
#                  and a global draw underserves the small ones by luck.
#   is_positive  — a group holding a label>=3 location. Stratifying on it fixes the eval
#                  positive RATE to the population's instead of leaving it to the draw,
#                  which is what makes a 20% slice's positive count predictable rather
#                  than a thing to discover afterwards.
# It is a stratified random split, not a conditioned one: no model, score or rank enters.
MIN_POS = 15                   # derive_t_good.MIN_POS — reported against, never enforced here


# --------------------------------------------------------------------------- #
# Atom keys, derived
# --------------------------------------------------------------------------- #
def collect_atom_keys():
    """`{location identity: atom_key}` for every maneuver-view row in the corpus.

    v10 read `provenance.atom_key` and therefore covered exactly the six batches that
    opted into the column. This derives the key from each row's own render block, so a
    batch builder cannot leave a location out of the split rule by omission — see
    `tools/corpus/atom_identity.py` for why the derivation is exact and where it stops."""
    out, from_stored, derived_new = {}, 0, 0
    for images_path in sorted(glob.glob(BATCHES_GLOB)):
        for line in Path(images_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ak = ai.atom_key_of_row(row)
            if not ak:
                continue
            if ai.stored_atom_key(row):
                from_stored += 1
            else:
                derived_new += 1
            ft = v8b.ftype_of(row)
            rd = row["render"]
            lo = loc_mod.from_render_block(rd)
            fp = dict(lo.family_params)
            key = (ft, rd["cx"], rd["cy"], rd["fw"], rd.get("c_re"), rd.get("c_im"),
                   tuple(fp.get(k) for k in loc_mod.family_param_keys(ft)))
            out[key] = ak
    return out, from_stored, derived_new


# --------------------------------------------------------------------------- #
# Split groups: the transitive closure of the leakage relations
# --------------------------------------------------------------------------- #
def normalize_coords(all_locs):
    """Coerce every coordinate axis to a DECIMAL STRING, and count what moved.

    `CORPUS_SCHEMA.md` says the render block carries cx/cy/fw and the family constants as
    decimal strings, and every consumer downstream believes it. One batch does not:
    `2026-08-03_q4_near_minibrot_v1` wrote `c_re`/`c_im` as JSON NUMBERS on all 290 of its
    julia rows. The corpus row is a frozen record and is not rewritten; the MANIFEST is
    derived, so the normalization belongs here.

    It was found by the bounded one-chunk render, not by reading: `crop_batch`'s
    `jsonl::field_str` returns None for an unquoted value, so those locations reached the
    engine as julia WITHOUT a seed `c` and it refused them — loudly, and only for julia.
    A phoenix row with a numeric `p_re` would have been accepted and rendered at the
    default Ushiki point, i.e. the wrong plane, silently. GATE 14 is the standing form of
    the check.

    It also removes a latent `TypeError`: the row-order sort key holds `d["c_re"] or ""`,
    so a float and a string in one family compare only if cx/cy/fw tie first. They do not
    today. That is luck, not a design."""
    moved = Counter()
    for d in all_locs:
        for k in ("cx", "cy", "fw", "c_re", "c_im"):
            v = d.get(k)
            if v is not None and not isinstance(v, str):
                d[k] = repr(float(v))
                moved[k] += 1
        for k in loc_mod.family_param_keys(d["ft"]):
            v = d["fparams"].get(k)
            if v is not None and not isinstance(v, str):
                d["fparams"][k] = repr(float(v))
                moved[k] += 1
    return moved


def assign_groups(locs):
    """v8's neighbourhood union-find, with the bucket refined by the FULL parameter axes.

    v8 partitions by `(fractal_type, c-bucket)` where the c-bucket is a tolerance cluster on
    the seed `c` alone, then spatially clusters inside it. For julia that is the plane, near
    enough. For **phoenix it is not**: `c` is one of three parameter pairs, and the 500-row
    phoenix grid varies `p` and `z_-1` at a FIXED viewport — so all 500 land in one c-bucket
    and one spatial cluster, and v8/v10's largest phoenix group is 496 locations that are
    500 different pictures. A neighbourhood relation that merges different dynamical planes
    because they share a viewport is backwards, and it is why the plane closure below used
    to swallow 1,029 of 1,098 phoenix locations into a single split group.

    So the bucket key gains the non-`c` family params, EXACTLY. Nothing else moves: the
    spatial predicate is `v8b.cluster` (imported, not copied), the c-tolerance is v8's, and
    for every family without extra axes — julia, the julia-multibrots, and the four
    mandelbrot-plane families that have no `c` at all — the partition is byte-for-byte v8's.
    Only phoenix is repartitioned, which is the family the change is about."""
    by_fam = defaultdict(list)
    for d in locs:
        by_fam[d["ft"]].append(d)
    next_gid = v8b.GID_OFFSET_V8
    for ft, group in by_fam.items():
        extra = [k for k in loc_mod.family_param_keys(ft)]
        if group[0]["c_re"] is not None:
            fw0 = float(group[0]["fw"])
            ctol = v8b.C_TOL_FRAC * fw0
            cre = [float(l["c_re"]) for l in group]
            cim = [float(l["c_im"]) for l in group]
            uf = v8b.UF(len(group))
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # The c-tolerance cluster may only merge locations that agree EXACTLY
                    # on every other axis — a near-`c` neighbour on a different `p` is a
                    # different plane, not a neighbourhood.
                    if any(group[i]["fparams"].get(k) != group[j]["fparams"].get(k)
                           for k in extra):
                        continue
                    dx, dy = cre[i] - cre[j], cim[i] - cim[j]
                    if dx * dx + dy * dy <= ctol * ctol:
                        uf.union(i, j)
            buckets = defaultdict(list)
            for i, l in enumerate(group):
                buckets[uf.find(i)].append(l)
        else:
            buckets = {0: group}
        for bkey, cc in buckets.items():
            sub = v8b.cluster([float(l["cx"]) for l in cc],
                              [float(l["cy"]) for l in cc],
                              [float(l["fw"]) for l in cc])
            local = {}
            for l, k in zip(cc, sub):
                if k not in local:
                    local[k] = next_gid
                    next_gid += 1
                l["group_id"] = local[k]
                # The PLANE this location sits on, as the bucket that produced it. For a
                # mandelbrot-plane family there is none, and `None` is the honest value.
                l["bucket"] = (ft, bkey) if cc[0]["c_re"] is not None else None
    return next_gid - v8b.GID_OFFSET_V8


def plane_key(d):
    """The location's DYNAMICAL-PLANE relation key, or `[]` for a parameter-plane family.

    Two locations on the same dynamical plane are two viewports on ONE picture, so they must
    not straddle the split — that is v8 GATE 7's seed-`c` claim, restated as a relation the
    draw obeys instead of a check it must survive afterwards.

    It is the **bucket** stamped by `assign_groups`, not a fresh key on the raw parameters:
    the bucket is already the `c`-tolerance cluster refined by the exact non-`c` axes, so
    two locations whose `c` differs by less than the tolerance — the same picture to within
    a rounding — land on the same key, which an exact-equality key would miss.

    Mandelbrot-plane families carry no `c` and are correctly unrelated here; their only
    neighbourhood relation is the spatial union-find."""
    return [("plane", d["bucket"])] if d.get("bucket") else []


def build_split_groups(all_locs, atom_of):
    """Union the spatial groups by shared atom / seed-c / parent_oid.

    Returns `(n_merges, detail)` and stamps `split_group` on every location. The spatial
    `group_id` is left alone — it is what the sampler weights and the straddle gates read,
    and it is a genuinely different partition (a neighbourhood, not a leakage class)."""
    gids = sorted({d["group_id"] for d in all_locs})
    idx = {g: i for i, g in enumerate(gids)}
    uf = v8b.UF(len(gids))

    def union_on(keyfn, label):
        buckets = defaultdict(list)
        for d in all_locs:
            for k in keyfn(d):
                buckets[k].append(d)
        merges = spanning = 0
        for members in buckets.values():
            roots = {uf.find(idx[m["group_id"]]) for m in members}
            if len(roots) > 1:
                spanning += 1
                merges += len(roots) - 1
                for m in members[1:]:
                    uf.union(idx[members[0]["group_id"]], idx[m["group_id"]])
        return {"relation": label, "keys_spanning_groups": spanning, "group_merges": merges}

    detail = [
        union_on(lambda d: [("atom", d["ft"], atom_of[v8b.ident_of_loc(d)])]
                 if v8b.ident_of_loc(d) in atom_of else [], "shared_minibrot_atom"),
        union_on(plane_key, "shared_dynamical_plane"),
        union_on(lambda d: [("poid", p) for p in d["parent_oids"]], "shared_parent_oid"),
    ]
    canon = {}
    for d in all_locs:
        r = uf.find(idx[d["group_id"]])
        d["split_group"] = canon.setdefault(r, gids[r])
    return sum(x["group_merges"] for x in detail), detail


def assign_split_randomized(all_locs, eval_share, seed):
    """The v11 split. Instruments first (forced), then a stratified grouped holdout.

    STAGE 1 reuses v8's forced-eval cascade verbatim, on the SPLIT group rather than the
    spatial one — every instrument is registered `score_unconditioned`, so a biased
    group-mate stays TRAIN under the exemption instead of being dropped.

    STAGE 2 draws whole non-instrument split groups into eval until each stratum reaches
    its share. `deal` walks a seeded shuffle and takes groups while the stratum is under
    quota, which lands within one group of the target rather than at a binomial's mercy.
    A group is counted in the stratum of its FIRST member by (ft, has-a-positive); a group
    is single-`ft` by construction and its positivity is a property of the whole group, so
    that is exact, not a representative choice.

    Returns (kept, dropped, stage2_detail)."""
    # ---- stage 1: forced-eval, on the split group ----
    saved = [d["group_id"] for d in all_locs]
    for d in all_locs:
        d["group_id"] = d["split_group"]
    kept, dropped = v8b.assign_split_by_group(all_locs)
    for d, g in zip(all_locs, saved):
        d["group_id"] = g
    for d in kept:
        d["eval_role"] = "instrument" if d["split"] == "eval" else None

    # ---- stage 2: the randomized grouped holdout over what stage 1 left in train ----
    by_sg = defaultdict(list)
    for d in kept:
        by_sg[d["split_group"]].append(d)
    forced_sgs = {sg for sg, ms in by_sg.items() if any(m["split"] == "eval" for m in ms)}

    strata = defaultdict(list)
    for sg, ms in by_sg.items():
        if sg in forced_sgs:
            continue
        strata[(ms[0]["ft"], any(m["label"] >= 3 for m in ms))].append(sg)

    detail = []
    for (ft, pos), sgs in sorted(strata.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_loc = sum(len(by_sg[sg]) for sg in sgs)
        target = eval_share * n_loc
        rng = random.Random(f"{seed}|{ft}|{int(pos)}")
        order = sorted(sgs)
        rng.shuffle(order)
        # Take a group iff doing so lands CLOSER to the target than leaving it does. The
        # whole shuffled order is scanned, so an oversized group being skipped does not end
        # the fill — smaller ones behind it still close the gap. A plain
        # `while taken < target` overshot julia:multibrot3's positive stratum to 0.326
        # because one 14-location group happened to come up early.
        taken = 0
        for sg in order:
            size = len(by_sg[sg])
            if abs(taken + size - target) >= abs(taken - target):
                continue
            for m in by_sg[sg]:
                m["split"] = "eval"
                m["eval_role"] = "holdout"
            taken += size
        largest = max((len(by_sg[sg]) for sg in sgs), default=0)
        detail.append({"fractal_type": ft, "has_positive": pos, "groups": len(sgs),
                       "locations": n_loc, "largest_group": largest,
                       "eval_locations": taken,
                       "realized_share": round(taken / max(n_loc, 1), 4),
                       # The granularity floor: groups are indivisible, so no draw can land
                       # closer to the target than one group's worth of locations. Reported
                       # per stratum so GATE 13 can hold the draw to what is ACHIEVABLE
                       # rather than to a flat tolerance that is generous on big strata and
                       # impossible on lumpy ones.
                       "granularity": round(largest / max(n_loc, 1), 4)})
    return kept, dropped, detail


# --------------------------------------------------------------------------- #
def row_of(d, loc_id):
    """v8's row, plus the two fields the randomized split makes load-bearing."""
    row = v8b.row_of(d, loc_id)
    row["eval_role"] = d.get("eval_role")          # None on every train row
    row["split_group"] = d["split_group"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report and gate; write nothing")
    ap.add_argument("--eval-share", type=float, default=EVAL_SHARE,
                    help="target eval share of the NON-instrument population")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    a = ap.parse_args()

    # ---- 1. the whole labeled corpus, through the canonical overlay-routed reader ----
    # `v8b.load_all_labeled` resolves every label through `label_store.resolve_score(row,
    # sidecar, amendments)` and skips any row it returns None for — which is exactly
    # "registered-unlabeled rows excluded", enforced by the reader rather than by a filter
    # here that could disagree with it.
    locs, n_crops, per_batch, overlay_changed, rev_labeled = v8b.load_all_labeled()
    all_locs = list(locs.values())
    for d in all_locs:
        v8b.classify_location(d)
        d["uniform"] = br.SOURCE_MANEUVER_UNIFORM in d["instruments"]

    coord_moved = normalize_coords(all_locs)

    # ---- 2. groups, then the leakage closure, then the split ----
    ngroups = assign_groups(all_locs)
    atom_of, n_from_stored, n_derived_new = collect_atom_keys()
    sg_merges, sg_detail = build_split_groups(all_locs, atom_of)
    kept, dropped, stage2 = assign_split_randomized(all_locs, a.eval_share, a.split_seed)

    # ---- 3. deterministic row order + dense loc_id ----
    kept.sort(key=lambda d: (d["ft"], d["cx"], d["cy"], d["fw"],
                             d["c_re"] or "", d["c_im"] or "",
                             json.dumps(d["fparams"], sort_keys=True)))
    rows = [row_of(d, i) for i, d in enumerate(kept)]

    tr = [r for r in rows if r["split"] == "train"]
    ev = [r for r in rows if r["split"] == "eval"]
    instr = [r for r in ev if r["eval_role"] == "instrument"]
    hold = [r for r in ev if r["eval_role"] == "holdout"]

    print("=" * 84)
    print("v11 COMPOSITION  (FRESH REBUILD; randomized location-grouped splits)")
    print("=" * 84)
    print(f"  labeled crops            : {n_crops} across {len(per_batch)} batches")
    print(f"  labeled locations        : {len(all_locs)}  ({ngroups} neighborhood groups, "
          f"{len({d['split_group'] for d in all_locs})} split groups)")
    print(f"  amendment overlay bound  : {overlay_changed} crops carry a revised label")
    print(f"  dropped (biased in a forced-eval group): {len(dropped)}")
    print(f"  manifest rows            : {len(rows)}   loc_id 0..{len(rows)-1} (dense, fresh)")
    print(f"    TRAIN            {v8b.fmt_hist(Counter(r['label'] for r in tr), len(tr))}")
    print(f"    EVAL             {v8b.fmt_hist(Counter(r['label'] for r in ev), len(ev))}")
    print(f"      instrument     {v8b.fmt_hist(Counter(r['label'] for r in instr), len(instr))}")
    print(f"      holdout        {v8b.fmt_hist(Counter(r['label'] for r in hold), len(hold))}")
    print(f"  realized eval share      : {len(ev)/len(rows):.4f} of all rows; "
          f"{len(hold)/max(len(rows)-len(instr),1):.4f} of the non-instrument population "
          f"(target {a.eval_share})")

    print("\n  per-partition x split (>=3 positives in brackets):")
    per_part = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        per_part[partition_of_row(r)][r["split"] if r["split"] == "train"
                                     else r["eval_role"]][r["label"]] += 1
    print(f"    {'partition':22s} {'train':>12} {'instrument':>12} {'holdout':>12}   MIN_POS={MIN_POS}")
    short = []
    for p in sorted(per_part):
        cells = []
        for k in ("train", "instrument", "holdout"):
            c = per_part[p][k]
            cells.append(f"{sum(c.values()):5d}[{c[3]+c[4]:4d}]")
        hp = per_part[p]["holdout"][3] + per_part[p]["holdout"][4]
        ip = per_part[p]["instrument"][3] + per_part[p]["instrument"][4]
        flag = "" if max(hp, ip) >= MIN_POS else "  << under MIN_POS on both eval roles"
        if flag:
            short.append(p)
        print(f"    {p:22s} " + " ".join(cells) + flag)

    # ===================================================================== #
    # GATES — all ABORTS.
    # ===================================================================== #
    print("\n" + "=" * 84 + "\nBUILD GATES (aborts)\n" + "=" * 84)

    orphans = [r for r in rows if r.get("label") is None or r.get("split") is None
               or r.get("group_id") is None or r.get("split_group") is None]
    assert not orphans, f"GATE 1 FAIL: {len(orphans)} orphan rows"
    print(f"  [ 1] 0 orphans                     OK ({len(rows)} rows complete)")

    ids = [v8b.ident(r) for r in rows]
    assert len(set(ids)) == len(ids), f"GATE 2 FAIL: {len(ids)-len(set(ids))} duplicate identities"
    tr_ids = {v8b.ident(r) for r in tr}
    ev_ids = {v8b.ident(r) for r in ev}
    assert not (tr_ids & ev_ids), f"GATE 2 FAIL: {len(tr_ids & ev_ids)} identities in both splits"
    print(f"  [ 2] 0 identity straddle           OK (train {len(tr_ids)} / eval {len(ev_ids)})")

    illegal, exempt_straddle = v8b.straddle_report(kept)
    assert not illegal, (f"GATE 3 FAIL: {len(illegal)} spatial groups span the split with a "
                         f"non-exempt eval member, e.g. {illegal[:5]}")
    print(f"  [ 3] spatial-group straddle        OK ({len({r['group_id'] for r in rows})} "
          f"groups; {len(exempt_straddle)} straddle by the score-unconditioned exemption)")

    # The SPLIT group is the unit the draw acts on, so it must not straddle at all except
    # by that same exemption — which is a different and stronger statement than GATE 3.
    by_sg = defaultdict(list)
    for r in rows:
        by_sg[r["split_group"]].append(r)
    sg_straddle = [sg for sg, ms in by_sg.items() if len({m["split"] for m in ms}) > 1]
    bad_sg = [sg for sg in sg_straddle
              if not all(m.get("eval_role") == "instrument"
                         for m in by_sg[sg] if m["split"] == "eval")]
    assert not bad_sg, (
        f"GATE 4 FAIL: {len(bad_sg)} SPLIT group(s) straddle for a reason other than the "
        f"score-unconditioned instrument exemption — the holdout draws whole groups, so "
        f"this means a leakage relation was closed after the draw: {bad_sg[:5]}")
    print(f"  [ 4] split-group straddle          OK ({len(by_sg)} split groups; "
          f"{len(sg_straddle)} straddle, all instrument-exempt)")

    biased_instr = [r for r in instr if r.get("biased")]
    assert not biased_instr, (
        f"GATE 5 FAIL: {len(biased_instr)} biased rows in the INSTRUMENT slice — the "
        f"cardinal sin. (The holdout is biased by design and is not covered here.)")
    print(f"  [ 5] 0 biased-in-instrument        OK ({len(hold)} holdout rows are biased by "
          f"design: {sum(1 for r in hold if r['biased'])}/{len(hold)})")

    forced = [d for d in kept if d["forced_eval"]]
    assert all(d["split"] == "eval" for d in forced), "GATE 6 FAIL: a forced-eval loc is not eval"
    assert all(d.get("eval_role") == "instrument" for d in forced), \
        "GATE 6 FAIL: a forced-eval loc is not tagged eval_role=instrument"
    by_instr = Counter(s for d in forced for s in d["instruments"])
    assert len(forced) == len(instr), "GATE 6 FAIL: instrument count disagrees with the rows"
    print(f"  [ 6] forced assignment holds       OK ({len(forced)} -> eval: "
          f"{dict(sorted(by_instr.items()))})")

    census = [d for d in kept if d["census"]]
    from_batches = set()
    for d in all_locs:
        if d["ft"].startswith("julia_multibrot") and (d["batches"] & v8b.CENSUS_BATCHES):
            from_batches.add((d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"]))
    census_ids = {(d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"]) for d in census}
    assert census_ids == from_batches, "GATE 7 FAIL: census slice drifted"
    assert len(census_ids) == v8b.N_CENSUS_EXPECTED, \
        f"GATE 7 FAIL: census is {len(census_ids)}, expected {v8b.N_CENSUS_EXPECTED}"
    print(f"  [ 7] eval instrument identity      OK (census-{v8b.N_CENSUS_EXPECTED} reproduced "
          f"location-for-location)")

    # The instruments must stay disjoint from TRAIN on every leakage relation. The holdout
    # is exempt by construction — it shares a split group with nothing in train (GATE 4) —
    # so this is the instrument-only statement v8's GATE 7 made, restated for two eval roles.
    train_seedc = {(r["fractal_type"], round(float(r["c_re"]), 12), round(float(r["c_im"]), 12))
                   for r in tr if r.get("c_re") is not None}
    instr_locs = [d for d in kept if d.get("eval_role") == "instrument"]
    sc_ov = [d for d in instr_locs if d["c_re"] is not None
             and (d["ft"], round(float(d["c_re"]), 12), round(float(d["c_im"]), 12)) in train_seedc]
    train_poids = {p for d in kept if d["split"] == "train" for p in d["parent_oids"]}
    poid_ov = {p for d in instr_locs for p in d["parent_oids"]} & train_poids
    assert not sc_ov, f"GATE 8 FAIL: {len(sc_ov)} instrument locations share a seed-c with train"
    assert not poid_ov, f"GATE 8 FAIL: {len(poid_ov)} instrument parent_oid in train"
    print(f"  [ 8] instruments disjoint from tr OK (seed-c 0 / parent_oid 0)")

    contra = [d for d in kept if d["biased"] is False
              and (d["batches"] & ls.TRAIN_SIDE_ONLY_BATCHES)]
    assert not contra, f"GATE 9 FAIL: {len(contra)} unbiased locations from train-side-only batches"
    unbiased_non_instr = [d for d in kept if not d["biased"] and not d["forced_eval"]]
    assert not unbiased_non_instr, (
        f"GATE 9 FAIL: {len(unbiased_non_instr)} unbiased NON-instrument locations exist. The "
        f"forced-eval cascade sends every unbiased member of an instrument group to eval, so "
        f"such a location would reach eval without being an instrument and be reported as an "
        f"unbiased draw it is not.")
    print(f"  [ 9] split vs label_store registr. OK")

    assert overlay_changed > 0, "GATE 10 FAIL: the amendment overlay changed 0 labels"
    n_q4 = sum(1 for r in rows if r["label"] == 4)
    assert n_q4 > 0, "GATE 10 FAIL: 0 class-4 locations"
    assert not rev_labeled, f"GATE 10 FAIL: {len(rev_labeled)} revision rows carry an in-row label"
    print(f"  [10] label overlay bound           OK ({overlay_changed} crops revised, "
          f"{n_q4} class-4 locations, 0 revision double-counts)")

    # --- GATE 11: loc_id <-> coordinates, BOTH directions. The prompt's hard abort. ---
    # A render plan is a list of (loc_id, coordinates) pairs and a cache is a tree keyed on
    # loc_id alone. If one loc_id could name two coordinate sets, or one coordinate set two
    # loc_ids, then a tile's directory no longer identifies what is in it — and nothing
    # downstream would notice, because every individual tile still renders fine.
    fwd, rev = {}, {}
    for r in rows:
        i, coord = r["loc_id"], v8b.ident(r)
        if i in fwd and fwd[i] != coord:
            raise AssertionError(f"GATE 11 FAIL: loc_id {i} names two coordinate sets")
        if coord in rev and rev[coord] != i:
            raise AssertionError(f"GATE 11 FAIL: coordinates {coord} carry two loc_ids "
                                 f"({rev[coord]}, {i})")
        fwd[i], rev[coord] = coord, i
    assert len(fwd) == len(rows) == len(rev), (
        f"GATE 11 FAIL: {len(rows)} rows -> {len(fwd)} loc_ids / {len(rev)} coordinate sets")
    assert sorted(fwd) == list(range(len(rows))), "GATE 11 FAIL: loc_id is not dense 0..N-1"
    print(f"  [11] loc_id <-> coordinates        OK (bijective over {len(rows)} rows, "
          f"dense 0..{len(rows)-1})")

    # --- GATE 12: the atom relation did its job on the instrument side ---
    instr_atoms = {(d["ft"], atom_of[v8b.ident_of_loc(d)]) for d in instr_locs
                   if v8b.ident_of_loc(d) in atom_of}
    twins = [d for d in kept if d["split"] == "train" and v8b.ident_of_loc(d) in atom_of
             and (d["ft"], atom_of[v8b.ident_of_loc(d)]) in instr_atoms]
    hard = [d for d in twins if not any(
        m["score_unconditioned"] for m in instr_locs
        if v8b.ident_of_loc(m) in atom_of
        and (m["ft"], atom_of[v8b.ident_of_loc(m)]) == (d["ft"], atom_of[v8b.ident_of_loc(d)]))]
    assert not hard, (
        f"GATE 12 FAIL: {len(hard)} TRAIN locations share a minibrot atom with a NON-exempt "
        f"instrument location — the instrument would be scoring a subject it was trained on")
    print(f"  [12] atom same-subject leak        OK ({len(atom_of)} locations carry an atom "
          f"key: {n_from_stored} also stored one, {n_derived_new} newly covered by the "
          f"derivation; {len(twins)} exempt train twins)")

    # --- GATE 13: the holdout landed where it was aimed ---
    off = [s for s in stage2
           if abs(s["realized_share"] - a.eval_share) > max(0.05, s["granularity"])]
    assert not off, (
        f"GATE 13 FAIL: {len(off)} stratum/strata missed the {a.eval_share} target by more "
        f"than one group's worth of locations — the draw, not the granularity, is wrong: "
        f"{off[:3]}")
    worst = max(stage2, key=lambda s: abs(s["realized_share"] - a.eval_share))
    print(f"  [13] holdout share per stratum     OK ({len(stage2)} strata, every one within "
          f"max(0.05, one group) of {a.eval_share}; worst "
          f"{worst['fractal_type']}|{'pos' if worst['has_positive'] else 'neg'} "
          f"{worst['realized_share']} vs granularity {worst['granularity']})")

    # --- GATE 14: every coordinate axis is a decimal string. See normalize_coords. ---
    nonstr = []
    for r in rows:
        for k in ("cx", "cy", "fw", "c_re", "c_im",
                  *loc_mod.family_param_keys(r["fractal_type"])):
            v = r.get(k)
            if v is not None and not isinstance(v, str):
                nonstr.append((r["loc_id"], k, type(v).__name__))
    assert not nonstr, (
        f"GATE 14 FAIL: {len(nonstr)} coordinate field(s) are not decimal strings, e.g. "
        f"{nonstr[:5]}. The engine reads them with a string accessor: julia REFUSES a "
        f"missing seed c, but phoenix would silently render the default Ushiki plane.")
    dyn_no_c = [r["loc_id"] for r in rows
                if r["fractal_type"].startswith("julia") and r.get("c_re") is None]
    assert not dyn_no_c, (
        f"GATE 14 FAIL: {len(dyn_no_c)} dynamical-plane row(s) carry no seed c and cannot "
        f"be rendered: {dyn_no_c[:5]}")
    print(f"  [14] render block is renderable    OK (all coordinate axes are decimal "
          f"strings; {sum(coord_moved.values())} normalized from numbers: "
          f"{dict(coord_moved) or 'none'})")

    if short:
        print(f"\n  NOTE: {len(short)} partition(s) still below MIN_POS={MIN_POS} positives on "
              f"BOTH eval roles: {short}. Reported, not gated — the corpus cannot be made to "
              f"contain positives it does not have, and 'label more' is the protocol's answer.")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    out = paths.bulk(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ev_out = paths.bulk(EVAL_OUT)
    with ev_out.open("w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")

    record = {
        "build": "v11",
        "mode": "fresh_full_rebuild",
        "rebuild": "uv run python tools/v11/build_manifest.py",
        "why_not_an_append": (
            "protocol §1's frozen prefix and a re-randomized split cannot both hold, and the "
            "split rule is what v11 changes. Comparability is carried by the four "
            "instruments, reproduced location-for-location (GATE 7 pins the census-144)."),
        "artifacts": {
            "manifest": {"path": OUT, "class": "bulk", "rows": len(rows)},
            "eval_slice": {"path": EVAL_OUT, "class": "bulk", "rows": len(ev)},
            "note": ("bulk, not durable: both are a deterministic function of the committed "
                     "label corpus, this module and the config below — regenerable, and too "
                     "large to be worth committing (CLAUDE.md's persistent-store convention)."),
        },
        "config": {
            "split_seed": a.split_seed,
            "eval_share_target": a.eval_share,
            "strata": ["fractal_type", "group_has_label_ge3"],
            "split_group_relations": ["shared_minibrot_atom", "shared_seed_c",
                                      "shared_parent_oid"],
            "atom_key_source": ("DERIVED from render.cx/cy + provenance.degree "
                                "(tools/corpus/atom_identity.py), not provenance.atom_key"),
            "loc_id": "dense 0..N-1 over this build's row order; no cross-version meaning",
            "row_order": "(fractal_type, cx, cy, fw, c_re, c_im, family_params)",
        },
        "population": {
            "labeled_crops": n_crops,
            "labeled_locations": len(all_locs),
            "manifest_rows": len(rows),
            "neighborhood_groups": ngroups,
            "split_groups": len(by_sg),
            "split_group_merges": sg_merges,
            "split_group_relations_detail": sg_detail,
            "train": len(tr), "eval": len(ev),
            "eval_instrument": len(instr), "eval_holdout": len(hold),
            "realized_eval_share_all_rows": round(len(ev) / len(rows), 4),
            "realized_holdout_share_of_non_instrument":
                round(len(hold) / max(len(rows) - len(instr), 1), 4),
            "class_train": {str(k): v for k, v in sorted(Counter(r["label"] for r in tr).items())},
            "class_instrument": {str(k): v for k, v in
                                 sorted(Counter(r["label"] for r in instr).items())},
            "class_holdout": {str(k): v for k, v in
                              sorted(Counter(r["label"] for r in hold).items())},
            "per_partition": {p: {k: {str(c): n for c, n in sorted(v.items())}
                                  for k, v in sorted(d.items())}
                              for p, d in sorted(per_part.items())},
            "per_batch_labeled_crops": dict(sorted(per_batch.items())),
            "dropped_biased_in_forced_eval_group": len(dropped),
        },
        "atom_identity": {
            "locations_with_an_atom_key": len(atom_of),
            "rows_that_also_stored_the_column": n_from_stored,
            "rows_newly_covered_by_the_derivation": n_derived_new,
            "why": ("v10 read provenance.atom_key and so covered exactly the six batches "
                    "that opted in; tools/atlas/sitting_cutter.py withholds the column "
                    "deliberately. The key is a pure read of the render block, verified "
                    "against every stored key in the corpus (tools/corpus/"
                    "test_atom_identity.py), so participation in the split rule no longer "
                    "depends on which builder wrote the row."),
        },
        "holdout_strata": strata_report(stage2),
        "eval_roles": {
            "instrument": ("the four score-unconditioned registrations, forced 100% eval. "
                           "UNBIASED. Base rates and version-over-version non-regression "
                           "read off these only."),
            "holdout": ("a stratified random draw over the remaining split groups. BIASED "
                        "exactly as training is — a held-out sample of the population the "
                        "model is trained on, which is what a calibration cut needs and "
                        "what a base rate must never be read from."),
        },
        "partitions_below_min_pos": {"min_pos": MIN_POS, "partitions": short},
        "deploy_note": ("ACTIVE_CKPT NOT switched; no threshold touched; t_good NOT "
                        "re-derived; nothing trained by this build."),
    }
    paths.durable(RECORD_OUT, mkparents=True).write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"WROTE {out}   ({len(rows)} rows: train {len(tr)} / eval {len(ev)})")
    print(f"WROTE {ev_out}   ({len(ev)} rows: instrument {len(instr)} + holdout {len(hold)})")
    print(f"WROTE {RECORD_OUT}   (committed record)")


def strata_report(stage2):
    """The per-stratum realized draw, keyed readably."""
    return [{"stratum": f"{s['fractal_type']}|{'pos' if s['has_positive'] else 'neg'}",
             **{k: v for k, v in s.items() if k not in ("fractal_type", "has_positive")}}
            for s in stage2]


if __name__ == "__main__":
    main()
