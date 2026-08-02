"""Guards for the v10 append — the claims that would fail silently if they broke.

Every assertion here is about something with no symptom: a manifest renumbered under a
cache trains on shuffled labels and only looks like a worse model; a bar edited after the
numbers is indistinguishable from a bar set before them; a cache tree that resolves
in-tree still works, it just quietly fills the working directory. None of these announce
themselves, which is the criterion for being here (`verification_practice.md` §1).

No `.exists()` skips: every input is committed (`data/v10/*` is durable and tracked), so a
missing one is a real failure and must be loud (§2). The one file-heavy check is marked
`slow` — it reads two 60-100 MB plans — and is the only thing in this file that is not in
the default lane.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import artifacts  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def _jsonl(rel):
    return [json.loads(l) for l in (ROOT / rel).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# --------------------------------------------------------------------------- #
# 1. The aug-cache tree resolves OUT of the working tree, by class not by literal.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", [
    "data/v8/aug_cache", "data/v9/aug_cache", "data/v10/aug_cache",
    "data/v10/aug_cache/7141/twilight_shifted__id__s1.0000__sh0.0000__ss2.jpg",
    "data/v42/aug_cache/0/x.jpg",          # a version that does not exist yet
])
def test_versioned_aug_cache_relocates(rel):
    """A forgotten registry entry must fail toward OUT-of-tree.

    v10's first dry run resolved its cache IN-tree because the registry held a literal per
    version. Nothing failed — 30,408 JPGs would simply have landed in the working tree
    under a gitignored path, which is the silent-bulk outcome the resolver exists to stop.
    `data/v42` is in this list on purpose: it is the version nobody has registered."""
    assert artifacts.is_relocated(rel), f"{rel} would resolve IN-TREE"
    assert artifacts.resolve(rel).is_relative_to(artifacts.artifacts_root())


@pytest.mark.parametrize("rel", [
    "data/v10/manifest.jsonl",             # durable, must stay in-tree
    "data/v10/cache_manifest.jsonl",
    "data/v10/aug_cache_probe/x.jpg",      # component-exact: a sibling is NOT the cache
    "data/vX/aug_cache/x.jpg",             # not a version token
    "data/aug_cache/x.jpg",                # no version component at all
])
def test_non_aug_cache_paths_stay_in_tree(rel):
    """The discriminating half. Without these the predicate could be `return True` and
    every test above would still pass."""
    assert not artifacts.is_relocated(rel), f"{rel} was wrongly relocated"
    assert artifacts.resolve(rel).is_relative_to(artifacts.REPO_ROOT)


# --------------------------------------------------------------------------- #
# 2. The frozen prefix really is v8's.
# --------------------------------------------------------------------------- #
FROZEN_FIELDS = ("loc_id", "cx", "cy", "fw", "label", "source", "biased", "split",
                 "fractal_type", "c_re", "c_im")


def test_the_v10_prefix_is_v8s_manifest_row_for_row():
    v8 = _jsonl("data/v8/manifest.jsonl")
    v10 = {r["loc_id"]: r for r in _jsonl("data/v10/manifest.jsonl")}
    meta = json.loads((ROOT / "data/v10/build_metadata.json").read_text(encoding="utf-8"))
    displaced = {r["loc_id"] for r in meta["displaced_prefix_rows"]["rows"]}

    compared = 0
    for pr in v8:
        nr = v10.get(pr["loc_id"])
        if nr is None:
            assert pr["loc_id"] in displaced, (
                f"v8 loc_id {pr['loc_id']} vanished from v10 without being declared "
                f"displaced in build_metadata")
            assert pr["split"] != "eval", (
                f"v8 EVAL loc_id {pr['loc_id']} was displaced — an instrument moved")
            continue
        for f in FROZEN_FIELDS:
            assert pr.get(f) == nr.get(f), (
                f"frozen prefix moved at loc_id {pr['loc_id']} field {f}: "
                f"{pr.get(f)!r} -> {nr.get(f)!r}")
        compared += 1
    # Non-vacuity (§5): a bug that emptied the loop would otherwise pass silently.
    assert compared == len(v8) - len(displaced) == 7115, (
        f"compared {compared} prefix rows against {len(v8)} v8 rows and "
        f"{len(displaced)} declared displaced")
    assert 0 < len(displaced) <= 25


def test_appended_loc_ids_start_past_every_v8_id():
    """The cache is keyed on loc_id. An appended location reusing a prefix id would point
    its 24 tiles at a directory that already holds another location's."""
    v8_max = max(r["loc_id"] for r in _jsonl("data/v8/manifest.jsonl"))
    v10 = _jsonl("data/v10/manifest.jsonl")
    v8_ids = {r["loc_id"] for r in _jsonl("data/v8/manifest.jsonl")}
    appended = [r for r in v10 if r["loc_id"] not in v8_ids]
    assert appended, "no appended rows — the extension is vacuous"
    assert min(r["loc_id"] for r in appended) > v8_max
    assert len({r["loc_id"] for r in v10}) == len(v10), "duplicate loc_id in v10"


# --------------------------------------------------------------------------- #
# 3. The eval slice: three instruments, and the class-4 discipline.
# --------------------------------------------------------------------------- #
INSTRUMENTS = {"prospect_census": 144, "loose0_v3_floor": 526, "maneuver_uniform_v1": 90}


def test_eval_slice_is_exactly_the_three_registered_instruments():
    ev = _jsonl("data/v10/eval_slice.jsonl")
    got = Counter(r["source"] for r in ev)
    assert dict(got) == INSTRUMENTS, f"eval instruments drifted: {dict(got)}"
    assert len(ev) == sum(INSTRUMENTS.values())
    assert not [r for r in ev if r.get("biased")], "a biased row reached eval"


def test_every_eval_class4_is_a_census_row_and_the_appended_fours_are_train_side():
    """The class-ceiling trap (`verification_practice.md` §6): the 1..4 scale's top class
    is the rare one, so a leak of appended fours into eval would inflate exactly the
    number nobody can check by eye. Both directions are asserted, and both counts are
    pinned — a count-free version of this passes on zero."""
    ev = _jsonl("data/v10/eval_slice.jsonl")
    q4 = [r for r in ev if r["label"] == 4]
    assert len(q4) == 22, f"{len(q4)} class-4 eval rows, expected 22"
    assert all(r["source"] == "prospect_census" for r in q4)

    v8_ids = {r["loc_id"] for r in _jsonl("data/v8/manifest.jsonl")}
    appended = [r for r in _jsonl("data/v10/manifest.jsonl") if r["loc_id"] not in v8_ids]
    app4 = [r for r in appended if r["label"] == 4]
    assert len(app4) == 23, f"{len(app4)} appended class-4 locations, expected 23"
    assert all(r["split"] == "train" for r in app4), \
        "an appended class-4 location reached eval"


def test_the_uniform_instrument_is_the_whole_leg_including_its_rule_labels():
    """The 81 rule-labeled rows are ordinary class-1 labels; 23 of them are in the uniform
    leg and stay in eval. Dropping them would condition the instrument's population on a
    quality-correlated rule, which is the bias the leg was drawn to avoid — and it would
    move the positive count the bar's power was derived from."""
    meta = json.loads((ROOT / "data/v10/build_metadata.json").read_text(encoding="utf-8"))
    rl = meta["rule_labeled_rows"]
    assert rl["n"] == 81
    assert rl["by_split"]["eval"] == 23
    ev = _jsonl("data/v10/eval_slice.jsonl")
    uni = [r for r in ev if r["source"] == "maneuver_uniform_v1"]
    assert len(uni) == 90
    assert sum(1 for r in uni if r["label"] >= 2) == 22, \
        "the uniform instrument's positive count moved — the bar's power derivation is stale"


# --------------------------------------------------------------------------- #
# 4. The bars were derived, and the eval script cannot restate them.
# --------------------------------------------------------------------------- #
def test_the_uniform_bar_is_recomputed_from_its_own_power():
    """Ground truth is the power calculation, not a copy of the committed value — but the
    function is also probed for a fixture that cannot fail (§6): a bar that ignored its
    inputs would return the same number for every n."""
    pr = _load("v10_prereg", "tools/v10/prereg.py")
    committed = json.loads(
        (ROOT / "data/v10/prereg_v10.json").read_text(encoding="utf-8"))
    arm = committed["arms"]["new_uniform90"]
    n_pos, n = arm["n_pos"], arm["n"]
    assert pr.min_detectable_auc(n_pos, n - n_pos) == arm["separation_bar"]
    assert 0.50 < arm["separation_bar"] < 1.0
    # more positives => a smaller detectable difference; the fixture can fail
    assert pr.min_detectable_auc(n_pos * 4, (n - n_pos) * 4) < arm["separation_bar"]


def test_eval_v10_loads_its_bars_and_does_not_restate_them():
    """A bar in the eval script is a bar that can be edited after seeing the numbers.
    Source-level, in the style of `tools/ranker/test_ranker.py`'s pin guard: the eval must
    read the committed pre-registration and must not carry its own margin constant."""
    src = (ROOT / "tools/v10/eval_v10.py").read_text(encoding="utf-8")
    assert "prereg_v10.json" in src and "PREREG.read_text" in src
    for forbidden in ("NONINF_MARGIN =", "SEPARATION_BAR =", "V8_CENSUS_Q3_REFERENCE"):
        assert forbidden not in src, (
            f"eval_v10.py defines its own {forbidden.strip(' =')} — bars must come from "
            f"data/v10/prereg_v10.json, which was committed before any eval ran")


def test_the_prereg_records_that_the_frozen_arms_cannot_see_the_intervention():
    """v9's lesson (§1.11): a NON-INFERIOR verdict on inputs identical to the baseline's is
    true and empty. Here identical tiles are CORRECT — the intervention is the data — so
    what has to be written down instead is which arm can see it. Anchored on the recorded
    tile delta, not on the prose around it."""
    pr = json.loads((ROOT / "data/v10/prereg_v10.json").read_text(encoding="utf-8"))
    delta = pr["instrument_check"]["canonical_tile_delta_vs_v9"]
    assert delta["prospect_census"]["new"] == 0
    assert delta["loose0_v3_floor"]["new"] == 0
    assert delta["maneuver_uniform_v1"]["new"] == delta["maneuver_uniform_v1"]["n"] == 90
    assert pr["arms"]["new_uniform90"]["gating"] is False
    assert pr["arms"]["primary_census144"]["gating"] is True


# --------------------------------------------------------------------------- #
# 5. The atom-union pass's premise.
# --------------------------------------------------------------------------- #
def test_only_the_2026_08_batches_record_an_atom_key():
    """GATE 14 permits the atom union to move appended rows freely because no earlier
    batch records an atom_key. That is a fact about the corpus, and if a future batch
    starts recording one the gate's scope argument silently stops holding."""
    import glob
    import os
    carriers = set()
    for p in sorted(glob.glob(str(ROOT / "data/label_corpus/batches/*/images.jsonl"))):
        bid = os.path.basename(os.path.dirname(p))
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if (json.loads(line).get("provenance") or {}).get("atom_key"):
                    carriers.add(bid)
                    break
    v10b = _load("v10_build_manifest_mod", "tools/v10/build_manifest.py")
    assert carriers, "no batch records an atom_key — the union pass is a no-op"
    assert carriers == v10b.NEW_BATCHES, (
        f"atom_key carriers moved: {sorted(carriers ^ v10b.NEW_BATCHES)}. GATE 14 argues "
        f"the union cannot re-partition the frozen corpus BECAUSE only the appended "
        f"batches carry one; that argument no longer holds.")


# --------------------------------------------------------------------------- #
# 6. The expensive one: prefix plan rows are byte-identical to v9's.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_prefix_plan_rows_are_byte_identical_to_v9s():
    """The claim that makes reusing 170,760 v9 tiles legitimate. If a prefix row differed
    on ANY field — cap, palette, geometry — the tile on disk would not be the tile the
    plan asks for, and nothing downstream would notice: the trainer reads whatever JPG is
    at that path. `slow` because it parses two plans totalling ~120 MB."""
    v9 = {r["out"]: r for r in _jsonl("data/v9/plan.jsonl")}
    v10 = _jsonl("data/v10/plan.jsonl")
    prefix = [r for r in v10 if int(Path(r["out"]).parent.name) <= 7140]
    assert len(prefix) == 7115 * 24 == 170760
    bad = [r["out"] for r in prefix if v9.get(r["out"]) != r]
    assert not bad, f"{len(bad)} prefix plan rows differ from v9's, e.g. {bad[:3]}"
    assert len(v10) - len(prefix) == 1267 * 24
