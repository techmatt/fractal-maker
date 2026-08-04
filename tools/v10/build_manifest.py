#!/usr/bin/env python
"""v10 location manifest — an APPEND onto the byte-frozen v8 prefix.

v8 was a from-scratch build (the v4..v7 manifests were gone, so there was no prefix to
inherit). v10 is not: `data/v8/manifest.jsonl` is committed, so
`docs/design/classifier_retrain_protocol.md` §1 applies as written — append post-freeze
labels to a byte-frozen prior-version manifest prefix, and enforce a frozen-prefix gate.
Every one of v8's 7,117 rows is re-emitted FIRST, in v8's own row order, and GATE 11
asserts each is identical on `loc_id / coordinates / label / split / source / biased`.

WHAT IS NEW: 1,292 locations from six batches labeled 2026-08-01/02 — the supply crawl
(four legs, 730 labeled rows) and the label-seeded harvest (two chunks, 580 rows). They
are all NATIVE-plane maneuver views (mandelbrot + multibrot3/4/5); none of them collides
with a v8 location.

THE SPLIT DECISION (Matt, 2026-08-02) — the one registration change
-------------------------------------------------------------------
`2026-08-01_supply_crawl_uniform_v1` (90 rows) is **eval-eligible and forced 100% eval**.
It is the only draw over the maneuver-view population that is unconditioned on any score:
the other three crawl legs are screen-stratified or exemplar-ordered, and both harvest
chunks are ordered by the fitted `view_fit_v1.1` queue score. An unbiased-given-selection
draw is exactly what §2 says must be forced to eval, and it gives the maneuver-view
population — the population every current discovery tool actually emits from — a
non-regression instrument that neither the julia:multibrot census nor the loose0_v3
mandelbrot floor can provide (both predate the maneuver sweep entirely).

Everything else new is train-side by the FAIL-CLOSED default: an unregistered batch
classifies `biased -> train`. That is the registered outcome for the crawl's strat_a /
strat_b / exemplar legs and for both harvest chunks, and it is the correct one — all five
are score-conditioned selections.

THE 81 RULE-LABELED ROWS. The `>30% interior` auto-reject rule labeled 81 crawl rows class
1 (`rule_labels_interior_gt30_v1.json` per leg, `labeler: rule:interior_gt30_v1`, already
merged in-row). They are ordinary class-1 labels here and are NOT special-cased. 58 of them
sit in the three train-side legs; the other 23 are in the uniform leg and therefore land in
EVAL — deliberately. Dropping them would condition the uniform instrument's population on a
quality-correlated rule, which is precisely the bias the leg was drawn to avoid, and the
pre-registered instrument is the whole 90 rows (22 positives at `>=2`). GATE 13 counts them
so the split of the 81 is a checked number rather than a remembered one.

GROUPS ARE RECOMPUTED OVER THE WHOLE POPULATION, not just the new rows. The union-find is
the only thing standing between a forced-eval location and a train neighbour that shares
its morphology, and a new location can bridge two old ones. Group *ids* therefore renumber
(they are a dense enumeration per (fractal_type, c-bucket) partition); the group
*partition* restricted to v8's rows must be EXACTLY v8's, and GATE 12 asserts that — same
members, same splits, no coarsening and no refinement. Renumbering is safe because nothing
reads a group id across versions: the sampler uses group size within one manifest
(`data_v4.compute_sampler_weights`) and the straddle gates use the partition.

  uv run python tools/v10/build_manifest.py [--dry-run]

Writes (all `paths.durable()`):
  data/v10/manifest.jsonl        the training population + split assignment
  data/v10/eval_slice.jsonl      the three frozen eval instruments
  data/v10/build_metadata.json   the recorded decisions
All gates are ABORTS, not warnings.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT))
import label_store as ls    # noqa: E402
import location as loc_mod  # noqa: E402
import paths                # noqa: E402

# v8's build is imported, not copied: the clustering predicate, the fail-closed registry
# and the batch->classification rules must be the SAME code, or "append onto v8's prefix"
# would silently mean "append onto something v8 would not have produced".
from tools.v8 import build_manifest as v8b  # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "scoring"))
import batch_registry as br  # noqa: E402  — THE batch table, one owner

BATCHES_GLOB = str(ROOT / "data" / "label_corpus" / "batches" / "*" / "images.jsonl")
PRIOR_MANIFEST = ROOT / "data" / "v8" / "manifest.jsonl"
OUT = "data/v10/manifest.jsonl"
EVAL_OUT = "data/v10/eval_slice.jsonl"
META_OUT = "data/v10/build_metadata.json"

# --------------------------------------------------------------------------- #
# The uniform leg's registration used to live HERE, as a third copy of the batch table
# on top of v7's and v8's. It now lives with every other batch in
# `tools/scoring/batch_registry.py`; these names are derived reads of that one table, kept
# because GATE 13 pins this specific instrument.
# --------------------------------------------------------------------------- #
UNIFORM_SOURCE = br.SOURCE_MANEUVER_UNIFORM
UNIFORM_BATCHES = br.batches_with_source(UNIFORM_SOURCE)
N_UNIFORM_EXPECTED = 90        # the pre-registered maneuver-view instrument
N_RULE_LABELED_EXPECTED = 81   # interior_gt30_v1 auto-rejects across the three crawl legs
RULE_LABEL_FILE = "rule_labels_interior_gt30_v1.json"

# A v8 row can leave the manifest ONLY by v8's own drop rule: a new location bridges it
# into a forced-eval group's neighbourhood, and a biased location in a forced-eval group is
# dropped rather than left to leak. That is a real discovery about the corpus, not an
# accounting slip, so it is allowed — under three constraints, all asserted in GATE 11:
#   * NEVER an eval row. An eval row leaving would move an instrument, which is the one
#     thing the frozen prefix exists to prevent.
#   * bounded. A handful is a bridge; dozens would mean the append changed the corpus's
#     shape and the "v8 prefix" framing is no longer honest.
#   * named. Every drop is written to build_metadata with its coordinates and its cause.
MAX_PREFIX_DROPS = 25


NEW_BATCHES = {
    "2026-08-01_supply_crawl_uniform_v1",
    "2026-08-01_supply_crawl_strat_a_v1",
    "2026-08-01_supply_crawl_strat_b_v1",
    "2026-08-01_supply_crawl_exemplar_v1",
    "2026-08-02_label_seeded_v2_a",
    "2026-08-02_label_seeded_v2_b",
}


def classify_batch(batch_id, ft):
    """(eval_eligible, biased, manifest_source) — THE registry, via v8's seam."""
    return v8b.classify_batch(batch_id, ft)


def classify_location(d):
    """v8's fold, verbatim. The uniform leg is no longer a special case here: it is an
    eval-eligible entry in the registry like the census and the floor, so `d["uniform"]`
    is a derived read of the instrument set rather than a fourth batch list."""
    v8b.classify_location(d)
    d["uniform"] = UNIFORM_SOURCE in d["instruments"]


def collect_atom_keys():
    """`{location identity: atom_key}` for every row that records one.

    `atom_key` is the minibrot NUCLEUS a maneuver view was framed around. Only the six
    2026-08 batches carry it — every earlier batch predates the maneuver sweep — so this
    pass can only ever union NEW locations, never move a v8 group. That is asserted in
    GATE 14 rather than argued."""
    out = {}
    for images_path in sorted(glob.glob(BATCHES_GLOB)):
        for line in Path(images_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ak = (row.get("provenance") or {}).get("atom_key")
            if not ak:
                continue
            ft = v8b.ftype_of(row)
            rd = row["render"]
            lo = loc_mod.from_render_block(rd)
            fp = dict(lo.family_params)
            key = (ft, rd["cx"], rd["cy"], rd["fw"], rd.get("c_re"), rd.get("c_im"),
                   tuple(fp.get(k) for k in loc_mod.family_param_keys(ft)))
            out[key] = ak
    return out


def union_by_atom(all_locs, atom_of):
    """Merge groups that share a minibrot atom. Returns (n_merges, n_atoms_spanning).

    The spatial union-find keys on (centre, frame width) and unions only when the frame
    widths are within 1.5x. Two views of the SAME atom at different maneuver `k` are
    therefore invisible to it — `k=4` and `k=16` differ by 4x in `fw` — while being two
    framings of one subject. Protocol §2's standing rule is that children inherit their
    seed's split; the atom is the seed here, so two views of one atom must share a group or
    a train view of an atom can sit opposite an eval view of the same atom.

    Measured on this corpus: 18 train-side crawl rows share an atom with a uniform-leg
    eval row, i.e. a fifth of the new instrument had a same-subject twin in training."""
    by_atom = defaultdict(list)
    for d in all_locs:
        ak = atom_of.get(v8b.ident_of_loc(d))
        if ak:
            by_atom[(d["ft"], ak)].append(d)
    gids = sorted({d["group_id"] for d in all_locs})
    idx = {g: i for i, g in enumerate(gids)}
    uf = v8b.UF(len(gids))
    spanning, merges = 0, 0
    for members in by_atom.values():
        roots = {uf.find(idx[m["group_id"]]) for m in members}
        if len(roots) > 1:
            spanning += 1
            merges += len(roots) - 1
            first = members[0]
            for m in members[1:]:
                uf.union(idx[first["group_id"]], idx[m["group_id"]])
    if merges:
        canon = {}
        for d in all_locs:
            r = uf.find(idx[d["group_id"]])
            d["group_id"] = canon.setdefault(r, gids[r])
    return merges, spanning


def rollback_ladder() -> dict:
    """What an eventual v10 adoption would have to revert, READ from the live pins.

    Not a hardcoded "v8": a rollback note that states the deployed version from memory is
    the same species of bug as a metadata file with a hardcoded `True` — it outlives the
    fact it records. Every version token below is read out of the module or artifact that
    owns it, so if one of them moves without the others this block says so.

    The thresholds are the point. `ACTIVE_CKPT` alone is not a rollback: `t_good`, the
    keeper cut and the tau_h fidelity base are all calibrated to a specific head's `p_good`
    SCALE (protocol §4), so leaving them pointed at v10's scale while the checkpoint goes
    back to v8 silently starves or floods recall with no error anywhere."""
    sys.path.insert(0, str(ROOT / "tools" / "scoring"))
    sys.path.insert(0, str(ROOT / "tools" / "atlas"))
    import production_pins as pp
    keeper = json.loads((ROOT / "data/atlas/keeper_cuts.json").read_text(encoding="utf-8"))
    sf = (ROOT / "tools/atlas/steered_frontier.py").read_text(encoding="utf-8")
    tau_model = None
    for line in sf.splitlines():
        if line.startswith("TAU_H_FIDELITY_BASE_MODEL"):
            tau_model = line.split("=", 1)[1].strip().strip('"')
            break
    ps = (ROOT / "tools/atlas/production_seeder.py").read_text(encoding="utf-8")
    return {
        "note": ("RECORD ONLY — this build adopts nothing. ACTIVE_CKPT is untouched and no "
                 "threshold file is written by any tool in tools/v10/."),
        "deployed_now": pp.ACTIVE_VERSION,
        "ladder_after_a_v10_adoption": ["v10", pp.ACTIVE_VERSION, "v7", "v6", "v5"],
        "why_not_v9": ("v9 was built, evaluated and STAGED but never adopted — its "
                       "primary arm passed on inputs byte-identical to the baseline's, so "
                       "the verdict was true and empty (auto_maxiter.md, 'Why v9 is "
                       "shelved'). It is not a rung: a rollback to a version that was "
                       "never deployed restores a gate that never ran."),
        "must_revert_together": [
            {"what": "tools/scoring/production_pins.ACTIVE_CKPT",
             "now": pp.ACTIVE_CKPT,
             "why": "ACTIVE_VERSION derives from it, and with it every decode stamp "
                    "(corpus_common.is_current_decoded, production_seeder.SCORER_VERSION)"},
            {"what": "tools/atlas/production_seeder.T_GOOD_OVERRIDES",
             "now": ("the v8 table — mandelbrot 0.85 via F0.5, julia:multibrot{3,4,5} "
                     "0.39/0.14/0.20 via F2, five partitions UNCALIBRATED"
                     if "0.85" in ps else "UNRECOGNIZED — read the table before rolling back"),
             "why": "per-partition t_good is calibrated to one head's p_good scale "
                    "(protocol §4); reusing a cut across scales silently starves recall"},
            {"what": "data/atlas/keeper_cuts.json",
             "now": f"stamped model={keeper['provenance']['model']!r}, derived from "
                    f"{keeper['eval']}",
             "why": "same scale-bound argument; tools/atlas/test_steered_frontier.py holds "
                    "the stamp equal to ACTIVE_VERSION, so a forgetful flip goes RED"},
            {"what": ("tools/atlas/steered_frontier.TAU_H_FIDELITY_BASE + "
                      "TAU_H_FIDELITY_BASE_MODEL, and data/atlas/tau_h_base_<v>.json"),
             "now": f"vendored base model={tau_model!r}",
             "why": "tau_h is a cut on a specific head's cheap p_good; the same test file "
                    "asserts the vendored model equals ACTIVE_VERSION"},
        ],
        "coherence": ("all four currently agree on "
                      f"{pp.ACTIVE_VERSION!r}" if
                      keeper["provenance"]["model"] == pp.ACTIVE_VERSION == tau_model
                      else f"DISAGREEMENT: pins={pp.ACTIVE_VERSION!r} "
                           f"keeper={keeper['provenance']['model']!r} tau_h={tau_model!r}"),
    }


def rule_labeled_join_keys():
    """The coordinate join_keys of every row the interior>30% rule labeled.

    Read from each leg's `rule_labels_*.json` and re-keyed through that leg's own
    images.jsonl, so the count is derived from the files rather than remembered."""
    keys = set()
    for bid in sorted(NEW_BATCHES):
        f = ROOT / "data" / "label_corpus" / "batches" / bid / RULE_LABEL_FILE
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        ids = set(d.get("labels", {}))
        img = ROOT / "data" / "label_corpus" / "batches" / bid / "images.jsonl"
        for line in img.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["image_id"] in ids:
                keys.add(ls.join_key(row["render"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report and gate; write nothing")
    a = ap.parse_args()

    # ---- 1. the whole labeled corpus, live (v8's loader, verbatim) ----
    locs, n_crops, per_batch, overlay_changed, rev_labeled = v8b.load_all_labeled()
    all_locs = list(locs.values())
    for d in all_locs:
        classify_location(d)

    rule_keys = rule_labeled_join_keys()

    # ---- 2. the frozen prefix ----
    prior_rows = [json.loads(l) for l in
                  PRIOR_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    prior_by_ident = {v8b.ident(r): r for r in prior_rows}
    assert len(prior_by_ident) == len(prior_rows), "duplicate identity in the v8 manifest"
    prior_group_of = {v8b.ident(r): r["group_id"] for r in prior_rows}

    # ---- 3. groups over the WHOLE population, then split per group ----
    ngroups = v8b.assign_groups(all_locs)
    pre_atom_gid = {v8b.ident_of_loc(d): d["group_id"] for d in all_locs}
    atom_of = collect_atom_keys()
    atom_merges, atom_spanning = union_by_atom(all_locs, atom_of)
    kept, dropped = v8b.assign_split_by_group(all_locs)

    kept_by_ident = {v8b.ident_of_loc(d): d for d in kept}

    # ---- 4. row order: v8's prefix verbatim, then the new rows ----
    new_locs = [d for d in kept if v8b.ident_of_loc(d) not in prior_by_ident]
    new_locs.sort(key=lambda d: (d["ft"], d["cx"], d["cy"], d["fw"],
                                 d["c_re"] or "", d["c_im"] or "",
                                 json.dumps(d["fparams"], sort_keys=True)))
    next_id = max(r["loc_id"] for r in prior_rows) + 1
    dropped_by_ident = {v8b.ident_of_loc(d): d for d in dropped}
    rows, prefix_rows, prefix_drops = [], [], []
    for pr in prior_rows:                     # <- v8 order, v8 loc_ids
        d = kept_by_ident.get(v8b.ident(pr))
        if d is None:
            prefix_drops.append((pr, dropped_by_ident.get(v8b.ident(pr))))
            continue
        r = v8b.row_of(d, pr["loc_id"])
        rows.append(r)
        prefix_rows.append((pr, r))
    for d in new_locs:
        rows.append(v8b.row_of(d, next_id))
        next_id += 1

    tr = [r for r in rows if r["split"] == "train"]
    ev = [r for r in rows if r["split"] == "eval"]
    n_new = len(new_locs)

    print("=" * 82)
    print("v10 COMPOSITION  (APPEND onto the byte-frozen v8 prefix)")
    print("=" * 82)
    print(f"  labeled crops            : {n_crops} across {len(per_batch)} batches")
    print(f"  labeled locations        : {len(all_locs)}  ({ngroups} neighborhood groups)")
    print(f"  amendment overlay bound  : {overlay_changed} crops carry a revised label")
    print(f"  dropped (biased in a forced-eval group): {len(dropped)}")
    print(f"  manifest rows            : {len(rows)}  = {len(prior_rows)} v8 prefix "
          f"- {len(prefix_drops)} displaced + {n_new} appended")
    print(f"  loc_id                   : prefix keeps v8's; new ids "
          f"{max(r['loc_id'] for r in prior_rows)+1}..{next_id-1}")
    print(f"    TRAIN {v8b.fmt_hist(Counter(r['label'] for r in tr), len(tr))}")
    print(f"    EVAL  {v8b.fmt_hist(Counter(r['label'] for r in ev), len(ev))}")
    print("\n  by (source, split, biased):")
    bysrc = defaultdict(lambda: defaultdict(int))
    for d in kept:
        bysrc[(d["source"], d["split"], d["biased"])][d["ft"]] += 1
    for (src, sp, bi), fams in sorted(bysrc.items()):
        n = sum(fams.values())
        print(f"    {src:44s} {sp:5s} biased={str(bi):5s} n={n:5d}  {dict(sorted(fams.items()))}")
    print("\n  per-family x class (manifest):")
    fc = defaultdict(Counter)
    for r in rows:
        fc[r["fractal_type"]][r["label"]] += 1
    for ft in sorted(fc):
        print(f"    {ft:<20} {v8b.fmt_hist(fc[ft], sum(fc[ft].values()))}")

    # ===================================================================== #
    # GATES — all ABORTS. 1..10 are v8's; 11..13 are the append's.
    # ===================================================================== #
    print("\n" + "=" * 82 + "\nBUILD GATES (aborts)\n" + "=" * 82)

    orphans = [r for r in rows if r.get("label") is None or r.get("split") is None
               or r.get("group_id") is None]
    assert not orphans, f"GATE 1 FAIL: {len(orphans)} orphan rows"
    print(f"  [ 1] 0 orphans                     OK ({len(rows)} rows complete)")

    ids = [v8b.ident(r) for r in rows]
    assert len(set(ids)) == len(ids), \
        f"GATE 2 FAIL: {len(ids)-len(set(ids))} duplicate identities"
    tr_ids = {v8b.ident(r) for r in rows if r["split"] == "train"}
    ev_ids = {v8b.ident(r) for r in rows if r["split"] == "eval"}
    assert not (tr_ids & ev_ids), f"GATE 2 FAIL: {len(tr_ids & ev_ids)} identities in both splits"
    assert len({r["loc_id"] for r in rows}) == len(rows), "GATE 2 FAIL: duplicate loc_id"
    print(f"  [ 2] 0 identity straddle           OK (train {len(tr_ids)} / eval {len(ev_ids)})")

    illegal, exempt_straddle = v8b.straddle_report(kept)
    assert not illegal, (f"GATE 3 FAIL: {len(illegal)} groups span the split with a "
                         f"non-exempt eval member, e.g. {illegal[:5]}")
    n_exempt_mates = sum(1 for d in kept if d.get("exempt_group_mate"))
    print(f"  [ 3] group straddle                OK ({len({r['group_id'] for r in rows})} "
          f"groups; {len(exempt_straddle)} straddle by the score-unconditioned exemption, "
          f"holding {n_exempt_mates} biased train group-mates the drop rule would have cut)")

    biased_eval = [r for r in rows if r["split"] == "eval" and r.get("biased")]
    assert not biased_eval, f"GATE 4 FAIL: {len(biased_eval)} biased rows in eval"
    print(f"  [ 4] 0 biased-in-eval              OK")

    census = [d for d in kept if d["census"]]
    floor = [d for d in kept if d["floor"]]
    uniform = [d for d in kept if d["uniform"]]
    forced = [d for d in kept if d["forced_eval"]]
    assert all(d["split"] == "eval" for d in forced), "GATE 5 FAIL: a forced-eval loc is not eval"
    assert all(d["forced_eval"] for d in kept if d["split"] == "eval"), \
        "GATE 5 FAIL: a non-forced-eval location reached eval"
    by_instr = Counter(s for d in forced for s in d["instruments"])
    print(f"  [ 5] forced assignment holds       OK ({len(forced)} -> eval: "
          f"{dict(sorted(by_instr.items()))})")

    from_batches = set()
    for d in all_locs:
        if d["ft"].startswith("julia_multibrot") and (d["batches"] & v8b.CENSUS_BATCHES):
            from_batches.add((d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"]))
    census_ids = {(d["ft"], d["cx"], d["cy"], d["fw"], d["c_re"], d["c_im"]) for d in census}
    assert census_ids == from_batches, "GATE 6 FAIL: census slice drifted"
    assert len(census_ids) == v8b.N_CENSUS_EXPECTED, \
        f"GATE 6 FAIL: census is {len(census_ids)}, expected {v8b.N_CENSUS_EXPECTED}"
    print(f"  [ 6] eval instrument identity      OK (census-{v8b.N_CENSUS_EXPECTED} reproduced)")

    train_seedc = {(r["fractal_type"], round(float(r["c_re"]), 12), round(float(r["c_im"]), 12))
                   for r in rows if r["split"] == "train" and r.get("c_re") is not None}
    census_sc_ov = [d for d in census
                    if (d["ft"], round(float(d["c_re"]), 12), round(float(d["c_im"]), 12))
                    in train_seedc]
    train_poids, eval_poids = set(), set()
    for d in kept:
        (train_poids if d["split"] == "train" else eval_poids).update(d["parent_oids"])
    poid_ov = eval_poids & train_poids
    assert not census_sc_ov, f"GATE 7 FAIL: {len(census_sc_ov)} census seed-c in train"
    assert not poid_ov, f"GATE 7 FAIL: {len(poid_ov)} eval parent_oid in train"
    print(f"  [ 7] eval disjoint from train      OK (seed-c 0 / parent_oid 0; "
          f"train poids={len(train_poids)}, eval poids={len(eval_poids)})")

    contra = [d for d in kept if d["biased"] is False
              and (d["batches"] & ls.TRAIN_SIDE_ONLY_BATCHES)]
    assert not contra, f"GATE 8 FAIL: {len(contra)} unbiased locations from train-side-only batches"
    print(f"  [ 8] split vs label_store registr. OK")

    assert overlay_changed > 0, "GATE 9 FAIL: the amendment overlay changed 0 labels"
    n_q4 = sum(1 for r in rows if r["label"] == 4)
    assert n_q4 > 0, "GATE 9 FAIL: 0 class-4 locations"
    print(f"  [ 9] amendment overlay bound       OK ({overlay_changed} crops revised, "
          f"{n_q4} class-4 locations)")

    assert not rev_labeled, f"GATE 10 FAIL: {len(rev_labeled)} revision rows carry an in-row label"
    print(f"  [10] no revision double-count      OK")

    # --- GATE 11: the frozen prefix. The whole premise of an append. ---
    FROZEN_FIELDS = ("loc_id", "cx", "cy", "fw", "label", "source", "biased", "split",
                     "fractal_type", "c_re", "c_im")
    drift = []
    for pr, nr in prefix_rows:
        for f in FROZEN_FIELDS:
            if pr.get(f) != nr.get(f):
                drift.append((pr["loc_id"], f, pr.get(f), nr.get(f)))
    assert not drift, (f"GATE 11 FAIL: {len(drift)} frozen-prefix field(s) moved, "
                       f"e.g. {drift[:5]}")
    assert [r["loc_id"] for r in rows[:len(prefix_rows)]] == \
           [pr["loc_id"] for pr, _ in prefix_rows], "GATE 11 FAIL: prefix row order moved"
    drop_eval = [pr for pr, _ in prefix_drops if pr["split"] == "eval"]
    assert not drop_eval, (
        f"GATE 11 FAIL: {len(drop_eval)} v8 EVAL row(s) displaced — an instrument moved, "
        f"which is the one thing the frozen prefix exists to prevent: {drop_eval[:3]}")
    assert len(prefix_drops) <= MAX_PREFIX_DROPS, (
        f"GATE 11 FAIL: {len(prefix_drops)} v8 rows displaced, cap is {MAX_PREFIX_DROPS} — "
        f"at that scale the append has changed the corpus's shape and calling this a v8 "
        f"prefix is no longer honest")
    print(f"  [11] frozen prefix                 OK ({len(prefix_rows)} v8 rows identical on "
          f"{len(FROZEN_FIELDS)} fields, same order; {len(prefix_drops)} displaced, 0 eval)")
    for pr, dd in prefix_drops:
        cause = sorted({k for m in
                        [m for m in all_locs if m.get("group_id") == (dd or {}).get("group_id")]
                        for k in ("census", "floor", "uniform") if m.get(k)}) if dd else []
        print(f"       displaced: loc_id {pr['loc_id']} {pr['fractal_type']} label "
              f"{pr['label']} split {pr['split']} — bridged into a forced-eval group "
              f"({'+'.join(cause) or 'unknown'}) by an appended location")

    # --- GATE 12: v8's group partition may only COARSEN, and only by a new bridge ---
    old_part, new_part = defaultdict(set), defaultdict(set)
    for pr, nr in prefix_rows:
        old_part[prior_group_of[v8b.ident(pr)]].add(pr["loc_id"])
        new_part[nr["group_id"]].add(nr["loc_id"])
    old_sets = [frozenset(v) for v in old_part.values()]
    new_sets = [frozenset(v) for v in new_part.values()]
    # every v8 group must land inside ONE v10 group: union-find only ever merges, so a
    # v8 group that splits means the clustering predicate itself moved, not the population.
    owner = {}
    for s in new_sets:
        for lid in s:
            owner[lid] = s
    split_groups = [s for s in old_sets if len({id(owner[lid]) for lid in s}) > 1]
    assert not split_groups, (
        f"GATE 12 FAIL: {len(split_groups)} v8 group(s) SPLIT across v10 groups — the "
        f"union-find only merges, so this means the clustering predicate moved: "
        f"{[sorted(s)[:4] for s in split_groups[:2]]}")
    merged = [s for s in new_sets if s not in set(old_sets)]
    renumbered = sum(1 for pr, nr in prefix_rows
                     if prior_group_of[v8b.ident(pr)] != nr["group_id"])
    print(f"  [12] v8 group partition            OK ({len(old_sets)} v8 groups, none split; "
          f"{len(merged)} coarsened by an appended bridge; {renumbered} rows renumbered)")

    # --- GATE 13: the new instrument, and where the 81 rule labels landed ---
    uni_rows = [r for r in rows if r["source"] == UNIFORM_SOURCE]
    assert len(uni_rows) == N_UNIFORM_EXPECTED, (
        f"GATE 13 FAIL: the uniform instrument is {len(uni_rows)} locations, expected "
        f"{N_UNIFORM_EXPECTED} — a leg row was lost or a group conflict dropped one")
    assert all(r["split"] == "eval" for r in uni_rows), "GATE 13 FAIL: a uniform row is not eval"
    uni_pos2 = sum(1 for r in uni_rows if r["label"] >= 2)
    # count rule-labeled CROPS by the split their location landed in
    rule_split = Counter()
    ident_split = {v8b.ident_of_loc(d): d["split"] for d in kept}
    n_rule_seen = 0
    for images_path in sorted(glob.glob(BATCHES_GLOB)):
        bid = os.path.basename(os.path.dirname(images_path))
        if bid not in NEW_BATCHES:
            continue
        for line in Path(images_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if ls.join_key(row["render"]) not in rule_keys:
                continue
            n_rule_seen += 1
            lo = loc_mod.from_render_block(row["render"])
            ft = v8b.ftype_of(row)
            key = (ft, row["render"]["cx"], row["render"]["cy"], row["render"]["fw"],
                   row["render"].get("c_re"), row["render"].get("c_im"),
                   tuple(dict(lo.family_params).get(k)
                         for k in loc_mod.family_param_keys(ft)))
            rule_split[ident_split.get(key, "DROPPED")] += 1
    assert n_rule_seen == N_RULE_LABELED_EXPECTED, (
        f"GATE 13 FAIL: {n_rule_seen} rule-labeled rows found, expected "
        f"{N_RULE_LABELED_EXPECTED}")
    print(f"  [13] uniform-90 instrument         OK ({len(uni_rows)} loc, all eval, "
          f"{uni_pos2} at label>=2; 81 rule rows -> {dict(rule_split)})")

    # --- GATE 14: the atom union pass did its job, and only its job ---
    # A v8 row CAN be pulled into a merged group — if it was spatially adjacent to an
    # appended location whose atom is shared, the merge carries it along. That is the
    # union-find working, not a collision. What must not happen is two v8 rows newly
    # sharing a group *because of* the atom pass, which would mean the pass re-partitioned
    # the frozen corpus.
    #
    # That quantity is computed here directly, from the pre-atom vs post-atom partition of
    # the prefix rows. It used to be read off GATE 12's `merged`, which is the set of v8
    # groups that coarsened for ANY reason — and GATE 12 permits spatial coarsening by an
    # appended bridge in the same breath (26 of them on today's corpus, 0 on the corpus v10
    # was frozen against). Two gates cannot disagree about whether coarsening is allowed;
    # `merged` was measuring the wrong quantity (`verification_practice.md` §1.1).
    v8_pulled = [pr["loc_id"] for pr, nr in prefix_rows
                 if pre_atom_gid.get(v8b.ident(pr)) != nr["group_id"]]
    pre_part, post_part = defaultdict(set), defaultdict(set)
    for pr, nr in prefix_rows:
        pre_part[pre_atom_gid[v8b.ident(pr)]].add(pr["loc_id"])
        post_part[nr["group_id"]].add(pr["loc_id"])
    pre_sets = {frozenset(v) for v in pre_part.values()}
    atom_merged = [s for s in {frozenset(v) for v in post_part.values()} if s not in pre_sets]
    assert not atom_merged, (
        f"GATE 14 FAIL: {len(atom_merged)} group(s) of v8 rows newly share a group BECAUSE "
        f"of the atom pass — it re-partitioned the frozen corpus rather than only the "
        f"appended rows: {[sorted(s)[:4] for s in atom_merged[:2]]}")
    # Same-subject leak. An instrument scoring a subject it was trained on at another zoom
    # is the model-quality failure the atom union exists to close — and it is exactly the
    # protection the score-unconditioned exemption gives up on purpose. So the assertion is
    # scoped to instruments that did NOT claim the exemption, and the exempted twins are
    # COUNTED rather than allowed to vanish into a relaxed gate.
    instr_atoms, exempt_atoms = {}, set()
    for d in kept:
        i = v8b.ident_of_loc(d)
        if d["split"] == "eval" and i in atom_of:
            k = (d["ft"], atom_of[i])
            instr_atoms[k] = d
            if d["score_unconditioned"]:
                exempt_atoms.add(k)
    twins = [d for d in kept if d["split"] == "train"
             and v8b.ident_of_loc(d) in atom_of
             and (d["ft"], atom_of[v8b.ident_of_loc(d)]) in instr_atoms]
    hard = [d for d in twins
            if (d["ft"], atom_of[v8b.ident_of_loc(d)]) not in exempt_atoms]
    assert not hard, (
        f"GATE 14 FAIL: {len(hard)} TRAIN locations share a minibrot atom with a "
        f"NON-exempt eval location — the instrument would be scoring a subject it was "
        f"trained on at another zoom")
    print(f"  [14] atom-union same-subject leak  OK ({atom_spanning} atoms spanned groups, "
          f"{atom_merges} merges, {len(v8_pulled)} v8 rows pulled into a merged group, "
          f"0 non-exempt train twins; {len(twins)} exempt train twins of a "
          f"score-unconditioned instrument atom)")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    out = paths.durable(OUT, mkparents=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with paths.durable(EVAL_OUT, mkparents=True).open("w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")

    meta = {
        "build": "v10",
        "mode": "append_onto_frozen_prefix",
        "prior_manifest": "data/v8/manifest.jsonl",
        "scale": "1..4 ordinal (CORN K-1 = 3 cutpoints)",
        "protocol": ("classifier_retrain_protocol.md §1 as written — v8's manifest is "
                     "committed, so unlike the v8 build there IS a prefix to freeze. "
                     "GATE 11 is the frozen-prefix gate; GATE 12 additionally pins v8's "
                     "group partition (ids renumber, membership does not)."),
        "population": {
            "labeled_crops": n_crops,
            "labeled_locations": len(all_locs),
            "manifest_rows": len(rows),
            "v8_prefix_rows": len(prior_rows),
            "frozen_prefix_rows": len(prefix_rows),
            "displaced_prefix_rows": len(prefix_drops),
            "appended_rows": n_new,
            "groups": ngroups,
            "train": len(tr), "eval": len(ev),
            "class_train": {str(k): v for k, v in sorted(Counter(r["label"] for r in tr).items())},
            "class_eval": {str(k): v for k, v in sorted(Counter(r["label"] for r in ev).items())},
            "per_family": {ft: {str(k): v for k, v in sorted(c.items())}
                           for ft, c in sorted(fc.items())},
            "per_batch_labeled_crops": dict(sorted(per_batch.items())),
        },
        "registration_change": {
            "batch": sorted(UNIFORM_BATCHES),
            "from": "unregistered -> fail-closed biased/train",
            "to": f"eval-eligible, unbiased, forced 100% eval (source {UNIFORM_SOURCE!r})",
            "n_locations": len(uni_rows),
            "n_label_ge2": uni_pos2,
            "why": ("the only score-UNCONDITIONED draw over the maneuver-view population. "
                    "The other three crawl legs are screen-stratified or exemplar-ordered "
                    "and both harvest chunks are ordered by the fitted view_fit_v1.1 queue "
                    "score, so all five are model/score-driven selection and stay train-side "
                    "by the fail-closed default. Neither existing eval instrument covers this "
                    "population: the census is julia:multibrot and loose0_v3 is a base-rate "
                    "flat draw over the native mandelbrot plane, both predating the maneuver "
                    "sweep."),
            "decided_by": "Matt, 2026-08-02 (v10_build prompt)",
        },
        "new_batches": sorted(NEW_BATCHES),
        "rule_labeled_rows": {
            "rule": "interior_gt30_v1 (provenance.interior_fraction > 0.30 -> class 1)",
            "n": n_rule_seen,
            "by_split": dict(rule_split),
            "note": ("ordinary class-1 labels, not special-cased. The 23 that sit in the "
                     "uniform leg stay in EVAL: removing them would condition the "
                     "instrument's population on a quality-correlated rule, which is the "
                     "bias the leg exists to avoid, and the pre-registered instrument is the "
                     "whole 90 rows."),
        },
        "eval_instruments": {
            "census": {"n": len(census), "source": "prospect_census",
                       "role": "pinned primary (julia:multibrot, unbiased-given-descent)"},
            "mandelbrot_floor": {"n": len(floor), "source": "loose0_v3_floor",
                                 "role": "mandelbrot non-regression floor"},
            "maneuver_uniform": {"n": len(uniform), "source": UNIFORM_SOURCE,
                                 "role": "NEW secondary — the maneuver-view population"},
        },
        "dropped_biased_in_forced_eval_group": {
            "count": len(dropped),
            "by_fractal_type": dict(sorted(Counter(d["ft"] for d in dropped).items())),
            "note": ("v8's rule, unchanged: a forced-eval group holding a biased location "
                     "cannot satisfy forced-100%-eval + 0-biased-in-eval + 0-straddle at "
                     "once, so the biased neighbour is dropped. GATE 11 proves no v8 row "
                     "was dropped by the new instrument."),
            "locations": [
                {"fractal_type": d["ft"], "cx": d["cx"], "cy": d["cy"], "fw": d["fw"],
                 "label": d["label"], "batches": sorted(d["batches"]),
                 "group_id": d["group_id"]} for d in dropped],
        },
        "displaced_prefix_rows": {
            "count": len(prefix_drops),
            "cap": MAX_PREFIX_DROPS,
            "eval_rows_displaced": 0,
            "why": ("v8's own drop rule, reached by a new bridge: an appended location "
                    "neighbours BOTH a v8 biased location and a forced-eval location, so "
                    "the union-find puts the v8 row in a forced-eval group, where a biased "
                    "location must be dropped rather than left to leak. Not an accounting "
                    "slip — a discovery that those two neighbourhoods are connected."),
            "rows": [{"loc_id": pr["loc_id"], "fractal_type": pr["fractal_type"],
                      "cx": pr["cx"], "cy": pr["cy"], "fw": pr["fw"],
                      "label": pr["label"], "split": pr["split"], "source": pr["source"]}
                     for pr, _ in prefix_drops],
            "cache_note": ("their 24 v9 cache tiles each become orphan directories in the "
                           "extended tree; the v10 cache verifier names them as EXPECTED "
                           "orphans rather than counting them complete."),
        },
        "atom_union": {
            "atoms_spanning_groups": atom_spanning,
            "group_merges": atom_merges,
            "v8_rows_pulled_into_a_merged_group": len(v8_pulled),
            "scope": ("only the six 2026-08 batches record provenance.atom_key, so this "
                      "pass can only union appended locations; GATE 14 asserts no v8 row "
                      "changed group because of it."),
            "why": ("the spatial union-find unions only when frame widths are within 1.5x, "
                    "so two views of ONE minibrot atom at different maneuver k (k=4 vs "
                    "k=16 is 4x in fw) are invisible to it while being two framings of the "
                    "same subject. Before this pass 18 train-side crawl rows shared an atom "
                    "with a uniform-leg eval row — a fifth of the new instrument had a "
                    "same-subject twin in training. Protocol §2's standing rule (children "
                    "inherit their seed's split) applied with the atom as the seed."),
        },
        "group_renumbering": {
            "v8_groups": len(old_sets), "rows_renumbered": renumbered,
            "why": ("group ids are a dense enumeration per (fractal_type, c-bucket) "
                    "partition, so appending 1,292 native-plane locations shifts the "
                    "numbering. Nothing reads a group id across versions — the sampler uses "
                    "group SIZE within one manifest and the gates use the partition — and "
                    "GATE 12 asserts the partition over v8's rows is byte-for-byte v8's."),
        },
        "deploy_note": ("ACTIVE_CKPT NOT switched; no threshold touched; t_good NOT set; "
                        "nothing trained by this build."),
        "rollback_ladder": rollback_ladder(),
    }
    paths.durable(META_OUT, mkparents=True).write_text(json.dumps(meta, indent=2),
                                                       encoding="utf-8")
    print("\n" + "=" * 82)
    print(f"WROTE {OUT}         ({len(rows)} rows: train {len(tr)} / eval {len(ev)})")
    print(f"WROTE {EVAL_OUT}    ({len(ev)} rows — census {len(census)} + floor {len(floor)} "
          f"+ uniform {len(uniform)})")
    print(f"WROTE {META_OUT}")


if __name__ == "__main__":
    main()
