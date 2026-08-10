"""selection.py — release selection (pure; no torch/GPU). ONE rule lives here.

`rank_select` — THE LIVE RULE since 2026-08-09 (prompts/selection_restructure_1.md). Top-N by
the head's own score, under two caps: a per-partition slot allocation crossed with the
thin-supply emit cap (`ranked_intake`), and at most `CLUSTER_CAP` picks per morph cluster per
run. Nothing else discounts a candidate — the diversity that `greedy_select` bought with a
continuous kernel is bought here by the cluster cap, which is a rule a human can read off a
sheet ("no more than two from one look") instead of a marginal-gain number.

`greedy_select` WAS HERE AND IS GONE (2026-08-09, prompts/selection_restructure_3.md). It was
max-marginal-gain over a morph-CLIP coverage kernel: `niche_relative_quality(c) ×
(1 − max_{s∈selected} K(c, s))`, with K a continuous cosine floored at `style_weight` for
same-style pairs. It stopped being the live rule on 2026-08-09 and was kept for one pass
because `report._v1_release_reconstruction` called it to reproduce what the v1 release had
actually shipped. That caller was ported (`report._v1_pool_under_the_live_rule` re-selects the
same durable v1 POOL under the live rank rule, and says so), which is what made the deletion
safe: a retired rule with one caller kept alive to reproduce history is a rule that has not
been retired. The v1 release records themselves are unchanged and remain the account of what
v1 shipped.

Its one non-obvious property, recorded because it is the trap the shape invites and not
because anything here can hit it: `greedy_select` compared `score` directly in its tie-break,
so passing TWO HEADS' entries to one call shut the smaller-scaled head out entirely — 82
release-eligible strange tiles lost every slot to smooth in the v1 release that way. The live
rule inherits the fix structurally: the driver runs a disjoint pass per head.

(Named `selection`, not `select`, so it never shadows the stdlib `select` module when
this directory lands on sys.path[0] as a run script's home.)

Pure Python; the slot arithmetic arrives from `ranked_intake` and the cap from `floors`.
"""
from __future__ import annotations

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
