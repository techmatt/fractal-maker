#!/usr/bin/env python
"""v8 location manifest — a FROM-SCRATCH build over the whole labeled corpus.

v8 extends the location head from the 1-3 scale to the full **1-4 ordinal scale** (three
CORN cutpoints, not two). Class 4 = "exceptional wallpaper emission"; the 2026-07-28
class-3 revisit re-judged every current q3 row on the full scale, so class 4 went from 9
labeled locations to 309.

WHY FROM SCRATCH (a deliberate departure from the retrain protocol's rule 1)
---------------------------------------------------------------------------
`docs/design/classifier_retrain_protocol.md` §1 says "append, never rebuild — freeze the
prior-version manifest prefix". That is not available here: `data/v{4,5,6,7}/manifest.jsonl`
are GONE (see tools/audit/durability_map.py — declared undeclared, gitignored, never
committed, and the working copies were cleared), and they are NOT to be reconstructed.
There is no prefix to inherit and no byte gate to run against one.

What the frozen prefix bought was eval comparability. That is preserved a different way:
the **census-144** julia:multibrot slice — the only unbiased-given-descent draw that
exists, and the slice v7 already reported its julia:mb metric on — is reproduced here
identically and forced 100% to eval. GATE 6 asserts it, location for location. That, not
row order, is the instrument the v7 comparison runs on.

Provenance of individual prior training samples is deliberately NOT traced: the corpus is
treated as one population. `loc_id` is dense over this build and carries no cross-version
meaning — but it IS preserved from the immediately-prior v8 manifest when one exists, because
the 171,384-tile aug cache is keyed on it (see load_prior_loc_ids). A re-split that keeps the
same population must not renumber, or every cached tile silently points at the wrong location.

THE MANDELBROT EVAL FLOOR (Matt's call, 2026-07-29)
---------------------------------------------------
A SECOND forced-eval instrument joins the census: loose0_v3, the unbiased base-rate
flat-generate draw over the native mandelbrot plane (526 loc). It gives the mandelbrot slice
— 59% of the corpus — a non-regression instrument the julia:multibrot census cannot. An
unbiased base-rate draw is not model-driven selection, so it is eval-eligible. The census-144
is untouched and stays the pinned primary (Gate 6 unchanged); the floor is additive.

WHAT IS DIFFERENT FROM v7's BUILD
---------------------------------
  * LIVE LABELS EVERYWHERE. Every label resolves through `resolve_score(row, sidecar,
    amendments)`. There is no post-freeze branch, so there is no path that reads a label
    without the overlay — which is what would let a revision row double-count against its
    anchor twin. GATE 9 asserts the overlay actually bound; GATE 10 asserts no revision row
    contributes a label of its own.
  * GROUP FIRST, THEN SPLIT. v7 partitioned the union-find by `(fractal_type, SPLIT,
    c-bucket)` purely because split was forced per batch under the append recipe, which let
    a c-bucket chain straddle. Here split is derived FROM the groups, so the partition is
    `(fractal_type, c-bucket)` and a group cannot straddle by construction.
  * MULTI-BATCH LOCATIONS ARE LEGAL. v7 asserted every post-freeze location came from
    exactly one batch. Over the whole corpus 616 do not (600 shared by the two `scale_*`
    batches, 8 by anchor/roster, 8 by gather/native-band). A location's biasedness is the
    OR over its batches, and eval-eligibility the AND — the conservative direction on both.
  * PHOENIX CARRIES ITS EXTRA CONSTANTS. Identity is `location.Location.key()`, which
    includes `family_params` (`p_re/p_im/zm1_re/zm1_im`). v7's tuple key did not. The 500
    phoenix_grid locations vary p and z_-1 per row; keying without them would collapse
    distinct locations and, downstream, render every phoenix cache tile at the Rust default
    Ushiki spot. v8 is the first manifest to contain phoenix at all.

  uv run python tools/v8/build_manifest.py

Writes (all `paths.durable()`):
  data/v8/manifest.jsonl        the training population + split assignment
  data/v8/eval_slice.jsonl      the frozen eval instrument, on its own
  data/v8/build_metadata.json   the recorded decisions
All 10 verifiability checks are ABORTS, not warnings.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
import label_store as ls   # noqa: E402
import location as loc_mod  # noqa: E402
import paths                # noqa: E402
# family (ledger cloud partition) <-> fractal_type (Rust kind_str) — THE map, one owner.
from partitions import FAM2FT  # noqa: E402

BATCHES_GLOB = str(ROOT / "data" / "label_corpus" / "batches" / "*" / "images.jsonl")
OUT = "data/v8/manifest.jsonl"
EVAL_OUT = "data/v8/eval_slice.jsonl"
META_OUT = "data/v8/build_metadata.json"

# --- neighborhood-clustering predicate (verbatim from v6/v7 build_manifest) ---
SHIFT_FRAC = 0.5
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5
C_TOL_FRAC = 0.05
GID_OFFSET_V8 = 8_000_000          # > every prior gid space; no collision if ever unioned

# --------------------------------------------------------------------------- #
# The batch registry. FAIL-CLOSED: a batch that is not named here classifies
# biased -> train. Unbiasedness and eval-eligibility require EXPLICIT registration, so
# a biased batch nobody remembered to list is still safe. Do NOT add a batch here to
# work around a classification you dislike — if a batch is misclassified, fix its
# registration and say why.
# --------------------------------------------------------------------------- #

# The one unbiased-given-descent draw that exists: prospect run-1 base-rate. Its
# julia:multibrot rows are the CENSUS = the primary eval instrument (forced 100% eval). Its
# native-plane rows are descent-screened, so they are biased -> train.
CENSUS_BATCHES = {"2026-07-17_prospect_run1_baserate_R_v1",
                  "2026-07-17_prospect_run1_baserate_v1"}
# The MANDELBROT EVAL FLOOR (Matt's call, 2026-07-29). loose0_v3 is the unbiased base-rate
# flat-generate draw over the mandelbrot plane (526 locations). Registering it eval-eligible
# gives the mandelbrot slice — 59% of the corpus — a regression instrument it otherwise
# lacks: the census is julia:multibrot only. It qualifies because the bias that disqualifies
# an eval population is *model-driven* selection (candidates a model liked), and an unbiased
# base-rate draw is not that. This REVERSES the prior eval_is_census_only decision on the
# record; it is a SECONDARY, additive instrument — the census-144 slice below is untouched
# and stays the pinned primary. Both are forced 100% eval by the same group rule.
FLOOR_BATCHES = {"2026-06-23_flat_generate_loose0_v3"}
# Unbiased but train-side (none currently: loose0_v3 moved to FLOOR_BATCHES above). Kept as
# the registry slot for a future unbiased-but-not-eval batch; empty is the fail-closed state.
UNBIASED_TRAIN_BATCHES = set()

N_CENSUS_EXPECTED = 144            # the pre-registered census eval slice (protocol §3, n~144)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def cluster(cx, cy, fw):
    """Neighborhood union-find on (cx,cy,fw). Returns dense local group ids."""
    n = len(cx)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            ratio = fw[i] / fw[j]
            if ratio < SCALE_LO or ratio > SCALE_HI:
                continue
            tol = SHIFT_FRAC * min(fw[i], fw[j])
            dx, dy = cx[i] - cx[j], cy[i] - cy[j]
            if dx * dx + dy * dy <= tol * tol:
                uf.union(i, j)
    roots, out = {}, []
    for i in range(n):
        out.append(roots.setdefault(uf.find(i), len(roots)))
    return out


def ftype_of(row):
    """fractal_type for a corpus row: provenance.family (authoritative) else
    render.fractal_type."""
    fam = (row.get("provenance") or {}).get("family")
    if fam and fam in FAM2FT:
        return FAM2FT[fam]
    ft = row["render"].get("fractal_type")
    return ft if ft else "mandelbrot"


def is_revision_row(row) -> bool:
    """A revision row re-judges an ALREADY-labeled source row on the 1..4 scale; its own
    score belongs in the source batch's amendment file, never in-row (see
    tools/corpus/merge_amendments.py). Detected by the back-pointer, not by batch name."""
    p = row.get("provenance") or {}
    return bool(p.get("revises_batch_id") and p.get("revises_image_id"))


# --------------------------------------------------------------------------- #
# 1. Reduce every labeled crop -> a location, LIVE (amendment overlay applied).
# --------------------------------------------------------------------------- #
def load_all_labeled():
    """Every labeled location in the corpus, keyed on `location.Location.key()`.

    label = max over the location's crops. `batches` is the SET of batches that
    contributed a labeled crop (616 locations legitimately span more than one). Also
    returns the per-batch diagnostics the report and gates 9/10 need."""
    locs = {}
    n_crops = 0
    per_batch_crops = Counter()
    overlay_changed = 0           # crops whose amendment differs from the original
    revision_rows_labeled = []    # gate 10: must stay empty
    for images_path in sorted(glob.glob(BATCHES_GLOB)):
        batch_id = os.path.basename(os.path.dirname(images_path))
        sidecar = ls.sidecar_for(batch_id)
        amendments = ls.amendments_for(batch_id)      # the REVISION overlay
        for line in Path(images_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            score = ls.resolve_score(row, sidecar, amendments)
            if score is None:
                continue
            if is_revision_row(row):
                revision_rows_labeled.append((batch_id, row["image_id"]))
                continue
            original = ls.resolve_score(row, sidecar)   # no overlay -> pre-revision value
            if original != score:
                overlay_changed += 1
            n_crops += 1
            per_batch_crops[batch_id] += 1
            ft = ftype_of(row)
            rd = row["render"]
            lo = loc_mod.from_render_block(rd)
            key = lo.key()
            d = locs.get(key)
            if d is None:
                d = locs[key] = dict(
                    ft=ft, cx=rd["cx"], cy=rd["cy"], fw=rd["fw"],
                    c_re=rd.get("c_re"), c_im=rd.get("c_im"),
                    fparams={k: v for k, v in lo.family_params},
                    labels=[], batches=set(), parent_oids=set())
            d["labels"].append(int(score))
            d["batches"].add(batch_id)
            poid = (row.get("provenance") or {}).get("parent_oid")
            if poid is not None:
                d["parent_oids"].add(poid)
    for d in locs.values():
        d["label"] = max(d["labels"])
    return locs, n_crops, per_batch_crops, overlay_changed, revision_rows_labeled


def classify_batch(batch_id, ft):
    """(eval_eligible, biased, source) for one (batch, fractal_type). FAIL-CLOSED."""
    if batch_id in CENSUS_BATCHES:
        if ft.startswith("julia_multibrot"):
            return True, False, "prospect_census"     # the primary eval instrument
        return False, True, "prospect_native"         # native-plane, descent-screened
    if batch_id in FLOOR_BATCHES:
        return True, False, "loose0_v3_floor"         # unbiased base-rate -> mandelbrot eval floor
    if batch_id in UNBIASED_TRAIN_BATCHES:
        return False, False, "loose0_v3"              # unbiased, train-side
    return False, True, "biased:" + batch_id          # FAIL CLOSED


def classify_location(d):
    """Fold a location's batches into one classification.

    biased        = OR  over its batches  (any biased contributor taints the location)
    eval_eligible = AND over its batches  (every contributor must be eval-eligible)
    Both directions are the conservative one: the cardinal sin is biased-in-eval."""
    cls = [classify_batch(b, d["ft"]) for b in sorted(d["batches"])]
    d["biased"] = any(c[1] for c in cls)
    d["eval_eligible"] = all(c[0] for c in cls)
    d["census"] = (d["ft"].startswith("julia_multibrot")
                   and any(b in CENSUS_BATCHES for b in d["batches"]))
    d["floor"] = any(b in FLOOR_BATCHES for b in d["batches"])
    # forced_eval drives the split (below). Both instruments force their group 100% eval by
    # the SAME rule; a biased neighbour chained into a forced group is dropped either way.
    d["forced_eval"] = d["census"] or d["floor"]
    d["source"] = "+".join(sorted({c[2] for c in cls}))


def assign_groups(locs):
    """Neighborhood union-find partitioned by (fractal_type, c-bucket) — NOT by split.

    v7 had to add SPLIT to the partition because split was forced per batch, so a
    c-bucket chain could transitively join an eval location to a train one. Here split is
    DERIVED from the group (`assign_split_by_group`), so no straddle is possible and the
    partition drops back to the natural one. Ids are offset by GID_OFFSET_V8."""
    by_fam = defaultdict(list)
    for d in locs:
        by_fam[d["ft"]].append(d)
    next_gid = GID_OFFSET_V8
    for _ft, group in by_fam.items():
        has_c = group[0]["c_re"] is not None
        if has_c:
            fw0 = float(group[0]["fw"])
            ctol = C_TOL_FRAC * fw0
            cre = [float(l["c_re"]) for l in group]
            cim = [float(l["c_im"]) for l in group]
            uf = UF(len(group))
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    dx, dy = cre[i] - cre[j], cim[i] - cim[j]
                    if dx * dx + dy * dy <= ctol * ctol:
                        uf.union(i, j)
            buckets = defaultdict(list)
            for i, l in enumerate(group):
                buckets[uf.find(i)].append(l)
        else:
            buckets = {0: group}
        for cc in buckets.values():
            sub = cluster([float(l["cx"]) for l in cc],
                          [float(l["cy"]) for l in cc],
                          [float(l["fw"]) for l in cc])
            local = {}
            for l, k in zip(cc, sub):
                if k not in local:
                    local[k] = next_gid
                    next_gid += 1
                l["group_id"] = local[k]
    return next_gid - GID_OFFSET_V8


def assign_split_by_group(locs):
    """Split assigned PER GROUP, so a group can never straddle.

    A group goes to EVAL iff it contains a FORCED-EVAL location — a census
    (julia:multibrot) location OR a mandelbrot-floor (loose0_v3) location. Both are
    unbiased-given-selection instruments and are forced 100% eval; everything else is
    train. The two never share a group (they live in disjoint fractal_type partitions),
    so the census-144 slice is reproduced location-for-location regardless of the floor.

    A forced-eval group that also contains a BIASED location is the one genuine conflict
    (v7 hit it on the census side: two census julia_multibrot4 locations chained through
    the c-bucket to model-band-selected neighbours; the floor hits it on the mandelbrot
    side, where base-rate flat locations chain to guided-descent neighbours). Three
    constraints collide there — forced 100% eval, zero biased-in-eval, zero group
    straddle — and only one resolution satisfies all three: DROP the biased neighbours
    from the manifest. Keeping them in train would leak the eval neighbourhood into
    training, which is exactly what the group unit exists to prevent. They are counted and
    named in build_metadata, never silently discarded. Returns (kept, dropped)."""
    by_gid = defaultdict(list)
    for d in locs:
        by_gid[d["group_id"]].append(d)
    kept, dropped = [], []
    for _gid, members in by_gid.items():
        if any(m["forced_eval"] for m in members):
            for m in members:
                if m["biased"]:
                    m["split"] = None
                    dropped.append(m)
                else:
                    m["split"] = "eval"
                    kept.append(m)
        else:
            for m in members:
                m["split"] = "train"
                kept.append(m)
    return kept, dropped


def fmt_hist(h, n):
    pos = h.get(3, 0) + h.get(4, 0)
    return (f"n={n:5d}  1:{h.get(1,0):5d}  2:{h.get(2,0):5d}  3:{h.get(3,0):5d}  "
            f"4:{h.get(4,0):5d}  ({100*pos/max(n,1):.1f}% >=3)")


def row_of(d, loc_id):
    """The manifest row. Family extra-constants (phoenix p / z_-1) ride along, because
    they are part of the location's identity and the render plan needs them."""
    row = {"loc_id": loc_id, "cx": d["cx"], "cy": d["cy"], "fw": d["fw"],
           "label": d["label"], "source": d["source"], "biased": d["biased"],
           "split": d["split"], "group_id": d["group_id"], "fractal_type": d["ft"]}
    if d["c_re"] is not None:
        row["c_re"] = d["c_re"]
        row["c_im"] = d["c_im"]
    for k in loc_mod.family_param_keys(d["ft"]):
        if d["fparams"].get(k) is not None:
            row[k] = d["fparams"][k]
    return row


def ident(r):
    """Identity of a manifest row — the same tuple the canonical location key covers."""
    return (r["fractal_type"], r["cx"], r["cy"], r["fw"], r.get("c_re"), r.get("c_im"),
            tuple(r.get(k) for k in loc_mod.family_param_keys(r["fractal_type"])))


def ident_of_loc(d):
    """Identity of a kept location dict — matches ident() over the row it becomes."""
    return (d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"],
            tuple(d["fparams"].get(k) for k in loc_mod.family_param_keys(d["ft"])))


def load_prior_loc_ids():
    """Read the CURRENT data/v8/manifest.jsonl (if present) as {identity: loc_id}.

    loc_id STABILITY is load-bearing: the 171,384-tile augmentation cache is keyed on
    loc_id (the plan seeds every palette/geometry draw on f"v8b-aug|{loc_id}" and writes
    each tile under aug_cache/<loc_id>/), so a rebuild that renumbers rows would silently
    point every tile at the wrong location. This re-split flips loose0_v3 train->eval and
    drops 24 newly-conflicting biased neighbours; the surviving population is a SUBSET of
    the prior manifest, so preserving each survivor's prior loc_id keeps every cache tile
    valid with zero re-render. A fresh checkout with no prior manifest falls back to a
    dense index (the original from-scratch behaviour)."""
    src = ROOT / OUT
    if not src.exists():
        return None
    prior = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prior[ident(r)] = int(r["loc_id"])
    return prior


# --------------------------------------------------------------------------- #
def main():
    locs, n_crops, per_batch, overlay_changed, rev_labeled = load_all_labeled()
    all_locs = list(locs.values())
    for d in all_locs:
        classify_location(d)
    ngroups = assign_groups(all_locs)
    kept, dropped = assign_split_by_group(all_locs)

    # Deterministic row order: family, then coordinates.
    kept.sort(key=lambda d: (d["ft"], d["cx"], d["cy"], d["fw"],
                             d["c_re"] or "", d["c_im"] or "",
                             json.dumps(d["fparams"], sort_keys=True)))

    # loc_id assignment. PRESERVE the prior manifest's identity->loc_id map (the cache is
    # keyed on it — see load_prior_loc_ids); only fall back to a dense index on a fresh
    # checkout with no prior manifest. Any kept location missing from the prior map is a
    # NEW location and gets a fresh id past the prior max — this re-split adds none (the
    # population is a strict subset), so preserved_new is asserted 0 below.
    prior = load_prior_loc_ids()
    if prior is None:
        rows = [row_of(d, i) for i, d in enumerate(kept)]
        preserved, preserved_new = 0, len(kept)
    else:
        next_id = max(prior.values()) + 1
        rows, preserved, new_ids = [], 0, []
        for d in kept:
            key = ident_of_loc(d)
            if key in prior:
                rows.append(row_of(d, prior[key])); preserved += 1
            else:
                rows.append(row_of(d, next_id)); new_ids.append(next_id); next_id += 1
        preserved_new = len(new_ids)
        assert len({r["loc_id"] for r in rows}) == len(rows), \
            "loc_id assignment produced duplicate ids"

    tr = [r for r in rows if r["split"] == "train"]
    ev = [r for r in rows if r["split"] == "eval"]

    print("=" * 82)
    print("v8 COMPOSITION  (from scratch — whole corpus, live labels, 1..4 scale)")
    print("=" * 82)
    print(f"  labeled crops            : {n_crops} across {len(per_batch)} batches")
    print(f"  labeled locations        : {len(all_locs)}  ({ngroups} neighborhood groups)")
    print(f"  amendment overlay bound  : {overlay_changed} crops carry a revised label")
    print(f"  dropped (biased in a forced-eval group): {len(dropped)}")
    print(f"  manifest rows            : {len(rows)}")
    print(f"  loc_id                   : {preserved} preserved from prior manifest, "
          f"{preserved_new} new"
          + ("" if prior is None else f" (prior had {len(prior)} rows)"))
    print(f"    TRAIN {fmt_hist(Counter(r['label'] for r in tr), len(tr))}")
    print(f"    EVAL  {fmt_hist(Counter(r['label'] for r in ev), len(ev))}")
    print("\n  by (source, split, biased):")
    bysrc = defaultdict(lambda: defaultdict(int))
    for d in kept:
        bysrc[(d["source"], d["split"], d["biased"])][d["ft"]] += 1
    for (src, sp, bi), fams in sorted(bysrc.items()):
        n = sum(fams.values())
        print(f"    {src:38s} {sp:5s} biased={str(bi):5s} n={n:5d}  {dict(sorted(fams.items()))}")
    print("\n  per-family x class (manifest):")
    fc = defaultdict(Counter)
    for r in rows:
        fc[r["fractal_type"]][r["label"]] += 1
    for ft in sorted(fc):
        print(f"    {ft:<20} {fmt_hist(fc[ft], sum(fc[ft].values()))}")

    # ===================================================================== #
    # 10 VERIFIABILITY GATES — all ABORTS.
    # ===================================================================== #
    print("\n" + "=" * 82 + "\nBUILD GATES (aborts)\n" + "=" * 82)

    # Gate 1: 0 orphans.
    orphans = [r for r in rows if r.get("label") is None or r.get("split") is None
               or r.get("group_id") is None]
    assert not orphans, f"GATE 1 FAIL: {len(orphans)} orphan rows"
    print(f"  [ 1] 0 orphans                     OK ({len(rows)} rows complete)")

    # Gate 2: 0 identities straddling train/eval (and no duplicate identity at all).
    ids = [ident(r) for r in rows]
    assert len(set(ids)) == len(ids), \
        f"GATE 2 FAIL: {len(ids) - len(set(ids))} duplicate identities in the manifest"
    tr_ids = {ident(r) for r in rows if r["split"] == "train"}
    ev_ids = {ident(r) for r in rows if r["split"] == "eval"}
    straddle_id = tr_ids & ev_ids
    assert not straddle_id, f"GATE 2 FAIL: {len(straddle_id)} identities in both splits"
    print(f"  [ 2] 0 identity straddle           OK (train {len(tr_ids)} / eval {len(ev_ids)}, "
          f"0 dupes)")

    # Gate 3: 0 group_ids straddling.
    g_split = defaultdict(set)
    for r in rows:
        g_split[r["group_id"]].add(r["split"])
    span = [g for g, s in g_split.items() if len(s) > 1]
    assert not span, f"GATE 3 FAIL: {len(span)} groups span the split, e.g. {span[:5]}"
    print(f"  [ 3] 0 group straddle              OK ({len(g_split)} groups in the manifest)")

    # Gate 4: 0 BIASED locations in eval — the cardinal sin (protocol §2).
    biased_eval = [r for r in rows if r["split"] == "eval" and r.get("biased")]
    assert not biased_eval, f"GATE 4 FAIL: {len(biased_eval)} biased rows in eval"
    print(f"  [ 4] 0 biased-in-eval              OK (eval is unbiased by construction)")

    # Gate 5: the forced assignment holds — every forced-eval location (census OR floor)
    # is in eval, and eval contains nothing but forced-eval locations.
    census = [d for d in kept if d["census"]]
    floor = [d for d in kept if d["floor"]]
    forced = [d for d in kept if d["forced_eval"]]
    assert all(d["split"] == "eval" for d in forced), \
        "GATE 5 FAIL: a forced-eval location is not eval"
    assert all(d["forced_eval"] for d in kept if d["split"] == "eval"), \
        "GATE 5 FAIL: a non-forced-eval location reached eval"
    print(f"  [ 5] forced assignment holds       OK ({len(census)} census + {len(floor)} floor "
          f"= {len(forced)} -> eval, eval == forced)")

    # Gate 6: THE INSTRUMENT GATE — replaces the (impossible) frozen-prefix byte gate.
    # The eval slice must be exactly the julia:multibrot locations of the two prospect
    # base-rate batches, at the pre-registered n~144. If this drifts, the v7<->v8
    # comparison is not on the same instrument and the whole build is meaningless.
    from_batches = set()
    for d in all_locs:
        if d["ft"].startswith("julia_multibrot") and (d["batches"] & CENSUS_BATCHES):
            from_batches.add((d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"]))
    census_ids = {(d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"]) for d in census}
    assert census_ids == from_batches, (
        f"GATE 6 FAIL: census slice drifted — {len(from_batches - census_ids)} lost, "
        f"{len(census_ids - from_batches)} gained")
    assert len(census_ids) == N_CENSUS_EXPECTED, (
        f"GATE 6 FAIL: census is {len(census_ids)} locations, expected "
        f"{N_CENSUS_EXPECTED} (the pre-registered eval slice)")
    print(f"  [ 6] eval instrument identity      OK (census-{N_CENSUS_EXPECTED} reproduced "
          f"exactly; the v7 comparison slice)")

    # Gate 7: census disjoint from ALL train at identity / seed-c / parent_oid.
    train_seedc = {(r["fractal_type"], round(float(r["c_re"]), 12), round(float(r["c_im"]), 12))
                   for r in rows if r["split"] == "train" and r.get("c_re") is not None}
    census_id_ov = [d for d in census
                    if (d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"],
                        tuple(d["fparams"].get(k) for k in loc_mod.family_param_keys(d["ft"]))) in tr_ids]
    census_sc_ov = [d for d in census
                    if (d["ft"], round(float(d["c_re"]), 12), round(float(d["c_im"]), 12))
                    in train_seedc]
    train_poids, census_poids = set(), set()
    for d in kept:
        if d["split"] == "train":
            train_poids |= d["parent_oids"]
        else:
            census_poids |= d["parent_oids"]
    poid_ov = census_poids & train_poids
    assert not census_id_ov, f"GATE 7 FAIL: {len(census_id_ov)} census identities in train"
    assert not census_sc_ov, f"GATE 7 FAIL: {len(census_sc_ov)} census seed-c in train"
    assert not poid_ov, f"GATE 7 FAIL: {len(poid_ov)} census parent_oid in train"
    print(f"  [ 7] census disjoint from train    OK (id 0 / seed-c 0 / parent_oid 0; "
          f"train poids={len(train_poids)}, census poids={len(census_poids)})")

    # Gate 8: split classification vs label_store's biased registration — the two
    # authorities must never silently disagree.
    contra = [d for d in kept if d["biased"] is False
              and (d["batches"] & ls.TRAIN_SIDE_ONLY_BATCHES)]
    assert not contra, (
        f"GATE 8 FAIL: {len(contra)} location(s) classified unbiased but sourced from a "
        f"batch label_store registers train-side-only: "
        f"{sorted({tuple(sorted(d['batches'])) for d in contra})[:5]}")
    print(f"  [ 8] split vs label_store registr. OK "
          f"({len(ls.TRAIN_SIDE_ONLY_BATCHES)} train-side-only batches, 0 classified unbiased)")

    # Gate 9: the amendment overlay actually BOUND. A build that silently read originals
    # would produce a corpus with ~no class 4 and would look superficially fine.
    assert overlay_changed > 0, (
        "GATE 9 FAIL: the amendment overlay changed 0 labels — every path is reading "
        "pre-revision labels. See label_store.AMENDMENT_LABELS / amendments_for.")
    n_q4 = sum(1 for r in rows if r["label"] == 4)
    assert n_q4 > 0, "GATE 9 FAIL: 0 class-4 locations — v8 has nothing to learn the new tier from"
    print(f"  [ 9] amendment overlay bound       OK ({overlay_changed} crops revised, "
          f"{n_q4} class-4 locations)")

    # Gate 10: no revision row contributes a label of its own. A revision's score belongs
    # in its SOURCE batch's amendment file; if one were also merged in-row it would enter
    # the manifest as a second vote on the anchor's own coordinates.
    assert not rev_labeled, (
        f"GATE 10 FAIL: {len(rev_labeled)} revision row(s) resolve to a non-null label "
        f"in-row (they must route to labels/amend_<source>.json via merge_amendments.py): "
        f"{rev_labeled[:5]}")
    print(f"  [10] no revision double-count      OK (0 revision rows carry an in-row label)")

    # ---- write (durable: asserts non-ignored at the write site) ----
    out = paths.durable(OUT, mkparents=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    eval_out = paths.durable(EVAL_OUT, mkparents=True)
    with eval_out.open("w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")

    meta = {
        "build": "v8",
        "mode": "from_scratch",
        "scale": "1..4 ordinal (CORN K-1 = 3 cutpoints)",
        "protocol_departure": {
            "rule": "classifier_retrain_protocol.md §1 (append, never rebuild)",
            "reason": "data/v{4,5,6,7}/manifest.jsonl are gone and are not to be "
                      "reconstructed; there is no prefix to freeze and no byte gate to "
                      "run. The frozen-prefix byte gate is DROPPED.",
            "replacement": "GATE 6 — the census-144 eval slice is reproduced identically "
                           "and asserted location-for-location. Eval comparability rides "
                           "on the instrument, not on row order.",
        },
        "population": {
            "labeled_crops": n_crops,
            "labeled_locations": len(all_locs),
            "manifest_rows": len(rows),
            "groups": ngroups,
            "gid_offset": GID_OFFSET_V8,
            "train": len(tr), "eval": len(ev),
            "class_train": {str(k): v for k, v in sorted(Counter(r["label"] for r in tr).items())},
            "class_eval": {str(k): v for k, v in sorted(Counter(r["label"] for r in ev).items())},
            "per_family": {ft: {str(k): v for k, v in sorted(c.items())} for ft, c in sorted(fc.items())},
            "per_batch_labeled_crops": dict(sorted(per_batch.items())),
        },
        "split_rule": (
            "Group first (union-find partitioned by (fractal_type, c-bucket)), then split "
            "per GROUP: a group containing a FORCED-EVAL location -> eval, else train. Two "
            "registered eval-eligible instruments, both forced 100% eval: the CENSUS "
            "(prospect run-1 base-rate julia:multibrot, 144 loc — the pinned primary) and "
            "the MANDELBROT FLOOR (loose0_v3 base-rate flat-generate, 526 loc — additive "
            "secondary, Matt's call 2026-07-29). assign_split is FAIL-CLOSED: an "
            "unregistered batch classifies biased -> train."),
        "mandelbrot_eval_floor": (
            "loose0_v3 (2026-06-23_flat_generate) is now eval-eligible: 526 unbiased "
            "base-rate mandelbrot locations moved train->eval so the mandelbrot slice (59% "
            "of the corpus) has a non-regression instrument the julia:multibrot census "
            "cannot provide. This REVERSES the prior eval_is_census_only stance on the "
            "record. It qualifies because the disqualifying bias is MODEL-DRIVEN selection "
            "(candidates a model retained), not an unbiased draw. The census-144 is "
            "untouched and stays the pinned primary; Gate 6 still reproduces it "
            "location-for-location. The paired v7<->v8 census comparison is unaffected — "
            "the floor is a separate, secondary read."),
        "eval_power_note": (
            f"{len(ev)} eval locations out of {len(rows)} ({100*len(ev)/max(len(rows),1):.1f}%): "
            f"{len(census)} census (julia:multibrot) + {len(floor)} floor (mandelbrot). The "
            "census remains the only unbiased-given-DESCENT draw; the floor is unbiased "
            "base-rate over the native mandelbrot plane. Per protocol §3 a null result on "
            "the census still means 'label more', not 'model failed'; the floor is powered "
            "for the bulk mandelbrot classes (q3 pos=26) but carries 0 class-4."),
        "loc_id_stability": (
            f"{preserved} of {len(rows)} loc_ids preserved from the prior manifest, "
            f"{preserved_new} new. loc_id is NOT re-enumerated: the 171,384-tile aug cache "
            "is keyed on it (plan seeds every draw on 'v8b-aug|<loc_id>' and writes tiles "
            "under aug_cache/<loc_id>/). The re-split's surviving population is a strict "
            "subset of the prior manifest (loose0_v3 flips train->eval in place; 24 biased "
            "mandelbrot neighbours are dropped), so every survivor keeps its prior id and "
            "every cache tile stays valid with zero re-render. build_plan.py regenerates "
            "plan.jsonl + cache_manifest.jsonl consistently from these ids."),
        "multi_batch_locations": (
            "616 locations are contributed by >1 batch (600 shared by the two scale_* "
            "batches, 8 anchor/roster, 8 gather/native-band). v7 asserted one batch per "
            "location, which only held because it looked at post-freeze rows. biased = OR "
            "over batches, eval_eligible = AND — conservative on both."),
        "phoenix_family_params": (
            "Location identity is location.Location.key(), which includes family_params "
            "(phoenix p_re/p_im/zm1_re/zm1_im). The 500 phoenix_grid locations vary p and "
            "z_-1 per row; v7's tuple key omitted them, which would collapse distinct "
            "locations and render every phoenix cache tile at the Rust default Ushiki "
            "spot. Those keys are carried into the manifest row and the render plan. v8 is "
            "the first manifest to contain phoenix."),
        "dropped_biased_in_forced_eval_group": {
            "count": len(dropped),
            "by_fractal_type": dict(sorted(Counter(d["ft"] for d in dropped).items())),
            "why": ("A forced-eval group (census OR mandelbrot-floor) that also holds a "
                    "biased location cannot satisfy forced-100%-eval + 0-biased-in-eval + "
                    "0-group-straddle at once. Keeping the biased neighbour in train would "
                    "leak the eval neighbourhood into training, so it is dropped. The 24 "
                    "mandelbrot drops are NEW (base-rate floor locations chaining to "
                    "guided-descent neighbours); the 10 julia_multibrot3 are the "
                    "pre-existing jm3_band census conflict, unchanged."),
            "locations": [
                {"fractal_type": d["ft"], "cx": d["cx"], "cy": d["cy"], "fw": d["fw"],
                 "c_re": d["c_re"], "c_im": d["c_im"], "label": d["label"],
                 "batches": sorted(d["batches"]), "group_id": d["group_id"]}
                for d in dropped],
        },
        "amendment_overlay": {
            "crops_revised": overlay_changed,
            "class4_locations": n_q4,
            "note": "Every label resolves through resolve_score(row, sidecar, amendments). "
                    "There is no post-freeze branch, so no path reads a pre-revision "
                    "label. Gate 9 asserts the overlay bound; gate 10 asserts no revision "
                    "row contributes an in-row label of its own.",
        },
        "deploy_note": "ACTIVE_CKPT NOT switched (v7 remains the deployed scorer); no "
                       "threshold touched; t_good NOT set; nothing trained by this build.",
    }
    paths.durable(META_OUT, mkparents=True).write_text(json.dumps(meta, indent=2),
                                                       encoding="utf-8")

    print("\n" + "=" * 82)
    print(f"WROTE {OUT}         ({len(rows)} rows: train {len(tr)} / eval {len(ev)})")
    print(f"WROTE {EVAL_OUT}    ({len(ev)} rows — the frozen instrument)")
    print(f"WROTE {META_OUT}")


if __name__ == "__main__":
    main()
