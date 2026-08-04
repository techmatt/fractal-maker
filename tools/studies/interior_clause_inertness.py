#!/usr/bin/env python
r"""Is the deployed OOD mask's interior clause actually doing anything?

THE QUESTION. The `G_cf` arm's sheet showed that inside the HIGH arm the argmax runs to the
band's lower lip — 57/60 windows in [0.10, 0.20), median 0.115. `G_cf` disfavours interior
mass indirectly anyway (`interior_worst` is the objective's second-largest weight, -1.278),
which raises the possibility that closes the whole interior question with no labeling: the
hard interior pre-filter may be REDUNDANT with the objective. If the ranker already avoids
interior on its own, removing the clause admits ~50% more scoreable positions that would
still never be framed, and the 49.8% pool growth is nominal.

THE TEST. Per atom, argmax `G_cf` twice over the same candidate set:

    W_live  — under the live pre-filter (interior >= 0.10 OR flat >= 0.88 OR speckle >= 0.30
              dropped, exactly as deployed: q4_stage1_linear_fit._v2_drop);
    W_free  — with ONLY the interior clause removed; flat and speckle unchanged.

Reports: the fraction of atoms where W_live != W_free; among those, W_free's interior_frac
and the G_cf gap; the fraction of atoms framed at interior_frac >= 0.10 under W_free; and,
for each atom, which clause (if any) excluded the UNCONSTRAINED argmax — i.e. which of the
three is actually binding, as opposed to which one we have been arguing about.

POPULATION — read this before quoting any number below.
The full swept positions do NOT survive. `build_interior_band_batch._sweep_one` cached only
its reservoir: <= CAND_CAP (24) windows per (atom, band, scale), plus per-bucket `seen`
counts. The raw fields DO survive (data/minibrot_batch/fields/*.bin, 160 atoms, bulk —
resolved out-of-tree), so a full-grid argmax is reconstructible — but this study is
read-only and does not re-sweep.

That reservoir is BAND-STRATIFIED, so a reservoir argmax is not a sweep argmax, and the bias
has a direction that must be named: every band gets the same cap regardless of how many
positions it saw. On a typical atom the `control` (< 0.10 interior) bucket kept 24 of ~5811
seen while `i35_50` kept 24 of ~225 — a ~26x relative over-representation of high-interior
windows. So W_free draws from an inflated high-interior menu while W_live draws from a
heavily thinned low-interior one. BOTH errors push toward W_live != W_free. The comparison
is therefore an UPPER BOUND on the clause's effect: a near-zero difference here is strong
evidence of redundancy, while a large difference is materially weaker than it looks.

Second, narrower bias in the same direction as the first is generous, this one is not: the
sweep dropped `g_interior >= 0.50` entirely (`band_of` returns None), so those windows are
absent from the cache and cannot be chosen by W_free even though a live deployment without
the interior clause would offer them. W_free's interior_frac is thus truncated at 0.50.

Read-only: no re-sweep, no config change, no threshold touched. `q4_stage1_linear_fit` and
`build_interior_band_batch` are imported for their constants only.

  uv run python tools/studies/interior_clause_inertness.py

Reads:  data/minibrot_roster/interior_band_v1/cand/<atom>.json  (G_cf precomputed per
        candidate; DURABLE, addressed through `IBB.CAND` rather than a second literal)
Writes: scratch/interior_clause_inertness/{report.txt,per_atom.jsonl}
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "tools", "sourcing"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                              # noqa: E402
import build_interior_band_batch as IBB                   # noqa: E402 (the candidate cache)
from tools.studies import q4_stage1_linear_fit as LF      # noqa: E402 (constants only)

# The candidate cache has ONE definition, in the module that writes it. This used to be a
# second `paths.scratch("interior_band_batch", "cand")` here, and when the cache moved to
# its durable home that copy would have kept pointing at a path the wipe already emptied —
# a study that reads nothing and reports on it.
CAND = IBB.CAND
OUT = paths.scratch("interior_clause_inertness")

# The deployed ceilings, read from the deployed module — never restated as literals here.
V2_INTERIOR, V2_FLAT, V2_SPECKLE = LF.V2_INTERIOR, LF.V2_FLAT, LF.V2_SPECKLE


def clauses_tripped(c) -> tuple:
    """Which deployed clauses this candidate trips (same predicate as LF._v2_drop, split)."""
    out = []
    if c["gi"] >= V2_INTERIOR:
        out.append("interior")
    if c["gflat"] >= V2_FLAT:
        out.append("flat")
    if c["gspeck"] >= V2_SPECKLE:
        out.append("speckle")
    return tuple(out)


def survives_live(c) -> bool:
    return not clauses_tripped(c)


def survives_free(c) -> bool:
    """Live filter with ONLY the interior clause removed."""
    return c["gflat"] < V2_FLAT and c["gspeck"] < V2_SPECKLE


def _argmax(cands):
    """Max by G_cf; ties broken on the box tuple so the pick is deterministic."""
    if not cands:
        return None
    return max(cands, key=lambda c: (c["G_cf"], tuple(c["box"]), c["scale"]))


def _wid(c):
    """Window identity: the box geometry, which is unique within an atom."""
    return None if c is None else (round(c["scale"], 6), tuple(round(v, 9) for v in c["box"]))


def load_atoms():
    rows = []
    for p in sorted(CAND.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cands = []
        for key, slot in d["cands"].items():
            band = key.split("|", 1)[0]
            for c in slot:
                c = dict(c)
                c["band"] = band
                cands.append(c)
        d["_cands"] = cands
        rows.append(d)
    return rows


def analyze(atoms):
    per_atom, clause_tally, no_live = [], Counter(), []
    for a in atoms:
        cands = a["_cands"]
        live = [c for c in cands if survives_live(c)]
        free = [c for c in cands if survives_free(c)]
        w_live, w_free, w_none = _argmax(live), _argmax(free), _argmax(cands)

        # Which clause excluded the UNCONSTRAINED argmax — () means nothing did, i.e. the
        # top-ranked window of the whole cached set already passes the deployed filter.
        binding = clauses_tripped(w_none) if w_none else ("<no candidates>",)
        clause_tally["+".join(binding) if binding else "none"] += 1

        differ = _wid(w_live) != _wid(w_free)
        if w_live is None:
            no_live.append(a["atom_id"])

        # HEADROOM. "The argmax did not move" is a binary; this is how far it was from
        # moving. Best interior-band window that the free filter WOULD admit (flat/speckle
        # still applied), vs the live argmax. A large positive headroom means the clause is
        # not merely inert here but comfortably inert — not one lucky atom away from binding.
        band_free = [c for c in free if c["gi"] >= V2_INTERIOR]
        w_band = _argmax(band_free)
        headroom = (None if (w_band is None or w_live is None)
                    else round(w_live["G_cf"] - w_band["G_cf"], 5))
        # Rank of the best interior-band window inside the free pool (1 == it would win).
        band_rank = None
        if w_band is not None:
            band_rank = 1 + sum(1 for c in free if c["G_cf"] > w_band["G_cf"])
        rec = dict(
            atom_id=a["atom_id"], degree=a["degree"], period=a["period"],
            period_band=a["period_band"], split=a["split"],
            n_cands=len(cands), n_live=len(live), n_free=len(free),
            n_swept=a["n_swept"], n_over_050=a["n_over_050"],
            differ=bool(differ),
            live_gcf=(None if w_live is None else w_live["G_cf"]),
            live_interior=(None if w_live is None else w_live["gi"]),
            live_scale=(None if w_live is None else w_live["scale"]),
            free_gcf=(None if w_free is None else w_free["G_cf"]),
            free_interior=(None if w_free is None else w_free["gi"]),
            free_scale=(None if w_free is None else w_free["scale"]),
            free_band=(None if w_free is None else w_free["band"]),
            gap=(None if (w_live is None or w_free is None)
                 else round(w_free["G_cf"] - w_live["G_cf"], 5)),
            unconstrained_gcf=(None if w_none is None else w_none["G_cf"]),
            unconstrained_interior=(None if w_none is None else w_none["gi"]),
            binding_clauses=list(binding),
            n_band_free=len(band_free),
            band_best_gcf=(None if w_band is None else w_band["G_cf"]),
            band_best_interior=(None if w_band is None else w_band["gi"]),
            headroom=headroom, band_rank=band_rank,
        )
        per_atom.append(rec)
    return per_atom, clause_tally, no_live


def _q(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def _fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def report(per_atom, clause_tally, no_live):
    L = []
    n = len(per_atom)
    A = L.append
    A("=" * 78)
    A("INTERIOR CLAUSE INERTNESS — is the hard pre-filter redundant with G_cf?")
    A("=" * 78)
    A("")
    A("POPULATION (read first)")
    A(f"  atoms                    : {n}")
    A(f"  candidates per atom      : median {int(_q([r['n_cands'] for r in per_atom], 0.5))}"
      f"  (reservoir, <= 24 per (atom, band, scale))")
    A(f"  positions swept per atom : median {int(_q([r['n_swept'] for r in per_atom], 0.5))}"
      f"  — NOT the population below")
    A("  The full swept set does not survive; only the band-stratified reservoir does. Every")
    A("  band got the same cap regardless of positions seen, so low-interior windows are")
    A("  thinned ~26x harder than high-interior ones. Both W_live and W_free are therefore")
    A("  sample-argmaxes, and BOTH sampling errors push toward W_live != W_free: the numbers")
    A("  below are an UPPER BOUND on the interior clause's real effect.")
    A(f"  windows with interior >= 0.50 were never cached (median {int(_q([r['n_over_050'] for r in per_atom], 0.5))}"
      f"/atom dropped at sweep), so W_free's interior_frac is truncated at 0.50.")
    A("")

    A("Q1. HOW OFTEN DOES REMOVING THE INTERIOR CLAUSE MOVE THE FRAME?")
    diff = [r for r in per_atom if r["differ"]]
    A(f"  W_live != W_free         : {len(diff)}/{n} atoms = {len(diff)/n:.3f}")
    A(f"  W_live == W_free         : {n-len(diff)}/{n} atoms = {(n-len(diff))/n:.3f}")
    if no_live:
        A(f"  atoms with NO live-filter survivor at all: {len(no_live)}")
    A("")

    A("Q2. AMONG THE ATOMS THAT MOVE — WHERE DOES THE FREE ARGMAX GO, AND WHAT DOES IT BUY?")
    if diff:
        ints = [r["free_interior"] for r in diff if r["free_interior"] is not None]
        gaps = [r["gap"] for r in diff if r["gap"] is not None]
        A(f"  W_free interior_frac     : median {_fmt(_q(ints,0.5))}  "
          f"[min {_fmt(min(ints))}, p25 {_fmt(_q(ints,0.25))}, "
          f"p75 {_fmt(_q(ints,0.75))}, max {_fmt(max(ints))}]")
        A(f"  W_free interior >= 0.10  : {sum(1 for v in ints if v >= V2_INTERIOR)}/{len(ints)}"
          f" of the moving atoms")
        A(f"  G_cf gap (free - live)   : median {_fmt(_q(gaps,0.5))}  "
          f"[min {_fmt(min(gaps))}, p25 {_fmt(_q(gaps,0.25))}, "
          f"p75 {_fmt(_q(gaps,0.75))}, max {_fmt(max(gaps))}]")
        A(f"                             mean {_fmt(statistics.fmean(gaps))}")
        A(f"  gap <= 0.5 G_cf units    : {sum(1 for g in gaps if g <= 0.5)}/{len(gaps)}"
          f"   (a move that buys almost nothing)")
    else:
        A("  (no atoms moved)")
    A("")

    A("Q2b. HOW FAR WAS IT FROM MOVING? (headroom: G_cf(W_live) - best interior-band window)")
    A("  The interior-band windows the free filter admits but the live one drops. Their pool")
    A("  size is what 'removing the clause admits ~50% more positions' buys the ranker here.")
    extra = [r["n_free"] - r["n_live"] for r in per_atom]
    A(f"  extra candidates admitted : median {int(_q(extra,0.5))}/atom "
      f"(live pool median {int(_q([r['n_live'] for r in per_atom],0.5))} -> "
      f"free pool median {int(_q([r['n_free'] for r in per_atom],0.5))}, "
      f"x{_q([r['n_free'] for r in per_atom],0.5)/max(1,_q([r['n_live'] for r in per_atom],0.5)):.1f})")
    hr = [r["headroom"] for r in per_atom if r["headroom"] is not None]
    if hr:
        A(f"  headroom (G_cf units)     : median {_fmt(_q(hr,0.5))}  "
          f"[min {_fmt(min(hr))}, p10 {_fmt(_q(hr,0.10))}, p90 {_fmt(_q(hr,0.90))}, "
          f"max {_fmt(max(hr))}]")
        A(f"  atoms with headroom <= 0  : {sum(1 for v in hr if v <= 0)}/{len(hr)}"
          f"   (would have moved)")
        A(f"  atoms with headroom < 1.0 : {sum(1 for v in hr if v < 1.0)}/{len(hr)}"
          f"   (close to moving)")
    br = [r["band_rank"] for r in per_atom if r["band_rank"] is not None]
    if br:
        A(f"  rank of best interior-band window in the free pool: median {int(_q(br,0.5))}, "
          f"best {min(br)}, worst {max(br)}")
    A("")

    A("Q3. WHAT FRACTION OF ATOMS END UP FRAMED AT interior_frac >= 0.10 UNDER W_free?")
    fi = [r["free_interior"] for r in per_atom if r["free_interior"] is not None]
    hi = [v for v in fi if v >= V2_INTERIOR]
    A(f"  framed at interior >= 0.10: {len(hi)}/{len(fi)} = {len(hi)/len(fi):.3f}")
    A(f"  W_free interior_frac      : median {_fmt(_q(fi,0.5))}  p90 {_fmt(_q(fi,0.9))}  "
      f"max {_fmt(max(fi))}")
    if hi:
        A(f"  among those, interior_frac: median {_fmt(_q(hi,0.5))}  max {_fmt(max(hi))}")
    A("")

    A("Q4. WHICH CLAUSE ACTUALLY BINDS? (clauses tripped by the UNCONSTRAINED argmax)")
    for k, v in clause_tally.most_common():
        A(f"  {k:<28}: {v:>4}/{n} = {v/n:.3f}")
    A("  'none' = the top-G_cf window of the whole cached set already passes the deployed")
    A("  filter, so no clause excluded anything and W_live == W_free == W_none for that atom.")
    A("")

    A("Q5. PER-SCALE — does the free argmax also change the window SIZE?")
    sc = Counter((r["live_scale"], r["free_scale"]) for r in per_atom)
    for (ls, fs), v in sorted(sc.items(), key=lambda kv: -kv[1]):
        mark = "  (same)" if ls == fs else "  <- scale moved"
        A(f"  live {ls} -> free {fs}: {v:>4}{mark}")
    A("")
    return "\n".join(L)


def main():
    if not CAND.exists():
        raise SystemExit(f"candidate cache not found: {CAND}\n"
                         f"(build_interior_band_batch sweep output; this study cannot re-sweep)")
    atoms = load_atoms()
    per_atom, clause_tally, no_live = analyze(atoms)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_atom.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_atom) + "\n", encoding="utf-8")
    txt = report(per_atom, clause_tally, no_live)
    (OUT / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"per-atom -> {OUT / 'per_atom.jsonl'}\nreport   -> {OUT / 'report.txt'}")


if __name__ == "__main__":
    main()
