#!/usr/bin/env python
"""The DISK-FREE half of `verify_cache_alignment.py`, as a collected test.

That script is the pre-train gate: it asserts tile<->location agreement in both
directions, because the aug cache is keyed on `loc_id` and a silent renumber trains on
tiles belonging to a different location — plausible numbers, wrong model. It runs by hand,
needs `--prior-plan`/`--prior-eval` backups that are not committed, and half its checks
read the ~12 GB tile cache, which has been DELETED since v9 was trained.

Four of its checks need none of that: they are relations *among the committed artifacts*
(`data/v8/{manifest,plan,cache_manifest,eval_slice}.jsonl`) and they pass at 100% today.
Those four belong in the suite, where they run on every commit rather than on the day
someone remembers to type the command with the right two backup paths:

  BACKWARD  every cache_manifest row's `location_id` exists in the manifest (no orphan)
  FIELDS    cache split/group/label/biased == the manifest's for that loc_id — the trainer
            reads the CACHE, so a disagreement silently trains on the wrong split
  COUNTS    exactly SLOTS(=24) plan rows per manifest loc_id
  CENSUS    the eval slice holds exactly 144 `prospect_census` locations

WHAT STAYS IN THE CLI, and why this is an extraction rather than a move. The other checks
are irreducibly external: FORWARD plan-vs-prior and the census loc_id-preservation check
compare against a PRIOR build that only exists as a hand-made backup, and both tiles-on-disk
checks need the cache. A test cannot assert those without inventing an input, and a gate
that invents its input is not a gate. `verify_cache_alignment.py` is unchanged.

  uv run pytest tools/v8/test_v8_cache_alignment.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / "data" / "v8"
SLOTS = 24            # augmentation tiles per location (the v8b recipe)
CENSUS_N = 144        # prospect_census locations in the eval slice
CENSUS_SOURCE = "prospect_census"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def v8():
    """Loaded once for the module: plan + cache_manifest are ~146 MB / 170,808 rows each
    and cost ~3.4 s to parse, which is per-test cost worth paying exactly once."""
    for p in ("manifest.jsonl", "plan.jsonl", "cache_manifest.jsonl", "eval_slice.jsonl"):
        if not (V8 / p).exists():
            pytest.skip(f"data/v8/{p} absent (LFS not smudged?)")
    manifest = _load(V8 / "manifest.jsonl")
    return dict(manifest=manifest,
                by_id={r["loc_id"]: r for r in manifest},
                plan=_load(V8 / "plan.jsonl"),
                cache=_load(V8 / "cache_manifest.jsonl"),
                eval_slice=_load(V8 / "eval_slice.jsonl"))


def test_the_v8_build_artifacts_are_nonempty(v8):
    """Guard the guard: an empty or truncated read makes every relation below vacuous."""
    assert len(v8["manifest"]) > 1000, f"manifest has {len(v8['manifest'])} rows"
    assert len(v8["plan"]) > 10_000 and len(v8["cache"]) > 10_000
    assert len(v8["by_id"]) == len(v8["manifest"]), "duplicate loc_id in the manifest"


def test_backward_every_cache_row_resolves_to_a_manifest_location(v8):
    """tile -> location. An orphan cache row points at a loc_id the manifest does not
    have, i.e. the cache and the population have drifted apart."""
    orphans = sorted({c["location_id"] for c in v8["cache"]
                      if c["location_id"] not in v8["by_id"]})
    assert not orphans, (f"{len(orphans)} cache loc_ids absent from the manifest, "
                         f"e.g. {orphans[:5]}")


def test_cache_rows_agree_with_the_manifest_on_split_group_label_biased(v8):
    """THE TRAINER READS THE CACHE, not the manifest. A row that disagrees on `split`
    trains on an eval location; on `label` it trains the wrong target; on `group_id` it
    breaks the group-aware split. Silent in every case — the loss curve looks fine."""
    bad = []
    for c in v8["cache"]:
        m = v8["by_id"].get(c["location_id"])
        if m is None:
            continue                       # covered by the BACKWARD test above
        if (c["split"] != m["split"] or c["group_id"] != m["group_id"]
                or c["label"] != m["label"] or bool(c["biased"]) != bool(m["biased"])):
            bad.append((c["location_id"], c["split"], m["split"], c["label"], m["label"]))
    assert not bad, (f"{len(bad)} cache rows contradict the manifest "
                     f"(loc_id, cache_split, man_split, cache_label, man_label): {bad[:5]}")


def test_every_location_has_exactly_the_slot_count_of_plan_rows(v8):
    """`loc_id` is the directory under `aug_cache/`, so the plan's own output paths are
    the census of what will be rendered per location. A location short of SLOTS trains on
    a thinner augmentation set than every other one, weighting it down invisibly."""
    per_loc = Counter(int(Path(r["out"]).parent.name) for r in v8["plan"])
    bad = sorted(lid for lid in v8["by_id"] if per_loc.get(lid, 0) != SLOTS)
    assert not bad, (f"{len(bad)} locations do not have exactly {SLOTS} plan rows, "
                     f"e.g. {[(lid, per_loc.get(lid, 0)) for lid in bad[:5]]}")
    assert len(v8["plan"]) == len(v8["cache"]), \
        f"plan {len(v8['plan'])} rows != cache_manifest {len(v8['cache'])} rows"


def test_the_eval_slice_holds_the_full_144_location_census(v8):
    """The census is a fixed, deliberately-constructed eval population; a re-split that
    drops or duplicates any of it changes what every AP number since means. Only the COUNT
    is assertable here — identity-preservation vs the PRIOR build needs the uncommitted
    backup, and that check stays in verify_cache_alignment.py."""
    n = sum(1 for r in v8["eval_slice"] if r.get("source") == CENSUS_SOURCE)
    assert n == CENSUS_N, f"{n} {CENSUS_SOURCE} rows in the eval slice, expected {CENSUS_N}"
