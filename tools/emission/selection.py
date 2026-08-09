"""selection.py — release selection (pure; no torch/GPU). Two rules live here.

`rank_select` — THE LIVE RULE since 2026-08-09 (prompts/selection_restructure_1.md). Top-N by
the head's own score, under two caps: a per-partition slot allocation crossed with the
thin-supply emit cap (`ranked_intake`), and at most `CLUSTER_CAP` picks per morph cluster per
run. Nothing else discounts a candidate — the diversity that `greedy_select` bought with a
continuous kernel is bought here by the cluster cap, which is a rule a human can read off a
sheet ("no more than two from one look") instead of a marginal-gain number.

`greedy_select` — the previous rule, max-marginal-gain over a morph-CLIP coverage kernel. It is
NOT called by the driver any more. It is kept, with its tests, because the removal of the old
machinery is a separate pass and because `report._v1_release_reconstruction` reads it to
reconstruct what the v1 release shipped; deleting it would silently change that reconstruction
into "what today's rule would have done", which is a different claim.

--- greedy_select (retired from the live path) --------------------------------

(Named `selection`, not `select`, so it never shadows the stdlib `select` module when
this directory lands on sys.path[0] as a run script's home.)

Select a release set of N wallpapers from the gated pool. Each greedy step picks the
pool entry with the largest marginal gain, where

    marginal_gain(c | selected) = niche_relative_quality(c) × coverage_gain(c | selected)

* niche_relative_quality(c) — the within-niche percentile of c's head score, so the
  best of a sparse niche can beat the tenth-best of a crowded one. The niche is the
  full descriptor cell (type, cluster, flavor, style); a singleton niche scores 1.0.

* coverage_gain(c | selected) = 1 − max_{s∈selected} K(c, s), the standard
  facility-location marginal: an item adds coverage in proportion to how UNlike the
  nearest already-selected item it is. K is the similarity kernel below.

* K(a, b) = max(0, cos(emb_a, emb_b)), optionally floored at `style_weight` when a and
  b share a render style. Continuous morph-CLIP cosine across ALL cells — there is NO
  categorical exact-match gate. A second, visually-near-identical look is therefore
  discounted regardless of whether it lands in a different (type, flavor, style) cell:
  two spirals that look alike suppress each other even across cells. (The old kernel
  multiplied cos by a product of exact-match categorical indicators, so any axis
  mismatch zeroed K — every pick in a fresh cell then read as maximally novel and the
  coverage term never fired.) The optional `style_weight` floor makes a WITHIN-HEAD pass
  spread across the promoted render modes: a second tile of the same style is discounted
  by at least `style_weight` even between morph-distinct locations, so greedy interleaves
  modes before doubling up on one.

Ties in marginal gain (ubiquitous when most niches are singletons, all at quality
percentile 1.0 and coverage 1.0) are broken by absolute head score, so a genuinely
strong singleton beats a weak one. Every pick logs the niche it filled and its nearest
already-selected neighbour (the "displacement" it caused).

CROSS-HEAD USE — never mix heads in one call. `score` is compared directly in the
tie-break and is on each head's own scale; the wallpaper head's `p_ge3` and the mining
head's are incommensurable. Passing both to ONE greedy_select shuts the smaller-scaled
head out entirely (this is exactly how 82 release-eligible strange tiles lost every slot
to smooth in the v1 release). Call greedy_select ONCE PER HEAD over that head's entries
and allocate the slot budget outside — the driver's `select_release` does exactly this
(disjoint smooth/strange passes). This selector treats `score` as a single within-head
quality axis and does NOT reconcile heads itself.

Pure Python + math; embeddings arrive as plain float lists so this is unit-testable.
"""
from __future__ import annotations

import math
from collections import defaultdict

# At most this many picks per morph cluster PER RUN. Imported from the cut owner rather than
# re-typed: it is a release-selection cap and it sits beside the junk floor and the thin-supply
# divisor, which are the other two coarse constants this restructure introduced. `floors` is
# torch-free by construction (it reaches only the three pin modules), so this module stays
# importable in the light lane.
from tools.emission.floors import CLUSTER_CAP

CATEGORICAL_AXES = ("type", "cluster", "flavor", "style")


def rank_select(entries: list, slots: dict, caps: dict, cluster_used: dict | None = None,
                cluster_cap: int = CLUSTER_CAP) -> tuple:
    """Top-N by score, per partition, under a slot budget, a supply cap and a cluster cap.

    `entries`  the SAME dicts `greedy_select` takes (id, type, cluster, flavor, style, score);
               `type` is the partition and `cluster` the morph cluster. ONE HEAD PER CALL — the
               same rule as `greedy_select` and for the same reason (see below).
    `slots`    `{partition: n}` — this pass's slot allocation (`ranked_intake.partition_slots`).
               A partition absent from `slots` gets zero: the allocation is the authority on
               which partitions this pass may emit from.
    `caps`     `{partition: n}` — the thin-supply emit cap (`ranked_intake.emit_cap` over the
               partition's floor-passing INTAKE supply). A partition absent from `caps` is
               UNCAPPED, which is the honest default for a caller that has no supply census;
               the driver always passes one.
    `cluster_used`  a mutable `{cluster: count}` carried ACROSS calls, so the cap is per RUN
               and not per pass. Two disjoint head passes over the same locations would
               otherwise each be free to take two tiles of one look. Mutated in place.

    Returns `(selected, log)`. `selected` is ordered partition-major (partitions in slot-map
    order, each partition's picks best-first); `log` carries one row per pick with the
    allocation state it was taken under, plus the cluster-cap skips, so a thin or lopsided
    release is diagnosable from the log alone.

    NEVER MIX HEADS IN ONE CALL. `score` is compared directly across entries and each head's
    `p_ge3` is on its own train-prior-calibrated scale; one call over both shuts the
    smaller-scaled head out entirely (82 release-eligible strange tiles lost every slot to
    smooth in the v1 release). The driver calls this once per head and allocates the head
    budget outside — exactly as it did with `greedy_select`."""
    used = cluster_used if cluster_used is not None else {}
    by_part: dict = defaultdict(list)
    for e in entries:
        by_part[e["type"]].append(e)
    selected: list = []
    log: list = []
    for part in slots:
        budget = min(int(slots.get(part, 0)),
                     int(caps[part]) if part in caps else int(slots.get(part, 0)))
        pool = sorted(by_part.get(part, []), key=lambda e: (-float(e["score"]), str(e["id"])))
        took = 0
        for rank, e in enumerate(pool):
            if took >= budget:
                break
            cl = e["cluster"]
            if used.get(cl, 0) >= cluster_cap:
                log.append({"id": e["id"], "partition": part, "cluster": cl,
                            "rank_in_partition": rank, "score": round(float(e["score"]), 4),
                            "picked": False, "skip": "cluster_cap"})
                continue
            used[cl] = used.get(cl, 0) + 1
            took += 1
            selected.append(e)
            log.append({"id": e["id"], "partition": part, "cluster": cl,
                        "rank_in_partition": rank, "score": round(float(e["score"]), 4),
                        "picked": True, "skip": None,
                        "slots": int(slots.get(part, 0)),
                        "supply_cap": (int(caps[part]) if part in caps else None),
                        "cluster_count": used[cl]})
    return selected, log


# --------------------------------------------------------------------------- #
# greedy_select — the retired rule (see the module docstring).
# --------------------------------------------------------------------------- #


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def kernel(a: dict, b: dict, style_weight: float = 0.0) -> float:
    """Coverage-similarity between two pool entries (see module docstring).

    Continuous morph-CLIP cosine across ALL cells — no categorical exact-match gate, so a
    near-identical look is discounted regardless of type/flavor/style. `style_weight` (>0)
    additionally floors the similarity when the two entries share a render style, so a
    within-head pass spreads across the promoted modes even between morph-distinct
    locations."""
    sim = max(0.0, _cos(a.get("emb"), b.get("emb")))
    if style_weight > 0.0 and a.get("style") is not None and a.get("style") == b.get("style"):
        sim = max(sim, style_weight)
    return sim


def niche_percentiles(entries: list) -> dict:
    """id → within-niche percentile of `score` (fraction of same-niche entries whose
    score is ≤ this one). A singleton niche → 1.0. Ties share the higher rank."""
    by_niche = defaultdict(list)
    for e in entries:
        by_niche[tuple(e[ax] for ax in CATEGORICAL_AXES)].append(e)
    pct = {}
    for niche, es in by_niche.items():
        n = len(es)
        for e in es:
            le = sum(1 for o in es if o["score"] <= e["score"])
            pct[e["id"]] = le / n
    return pct


def greedy_select(entries: list, n: int, style_weight: float = 0.0) -> tuple:
    """Greedy max-marginal-gain selection of ≤ n entries.

    `entries`: list of dicts with keys id, type, cluster, flavor, style, score
    (head p_ge3 or ordinal score — any within-niche-comparable quality), emb (list|None).
    `style_weight` (>0) floors the coverage kernel for same-render-style pairs so this pass
    spreads across modes (see `kernel`); pass it only when the entries are a single head's
    (never mix heads in one call — see module docstring).
    Returns (selected, log) where selected is the ordered list of chosen entries and log
    is a per-pick list of {id, niche, niche_pct, coverage_gain, marginal_gain, score,
    nearest_selected, nearest_sim}."""
    pct = niche_percentiles(entries)
    remaining = list(entries)
    selected: list = []
    log: list = []
    while remaining and len(selected) < n:
        best = None
        best_key = None
        for c in remaining:
            if selected:
                sims = [(kernel(c, s, style_weight), s) for s in selected]
                nn_sim, nn = max(sims, key=lambda t: t[0])
            else:
                nn_sim, nn = 0.0, None
            cov = 1.0 - nn_sim
            q = pct[c["id"]]
            gain = q * cov
            # primary: marginal gain; tie-break: absolute score, then id for determinism.
            key = (gain, c["score"], c["id"])
            if best_key is None or key > best_key:
                best_key, best = key, (c, q, cov, gain, nn, nn_sim)
        c, q, cov, gain, nn, nn_sim = best
        selected.append(c)
        remaining.remove(c)
        log.append({
            "id": c["id"],
            "niche": [c[ax] for ax in CATEGORICAL_AXES],
            "niche_pct": round(q, 4),
            "coverage_gain": round(cov, 4),
            "marginal_gain": round(gain, 4),
            "score": round(float(c["score"]), 4),
            "nearest_selected": (nn["id"] if nn else None),
            "nearest_sim": round(nn_sim, 4),
        })
    return selected, log
