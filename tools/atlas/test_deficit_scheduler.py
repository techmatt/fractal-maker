"""Tests for the family-level deficit scheduler (tools/atlas/deficit_scheduler.py).

Torch-free / render-free: the order-book projection, the distinct-look tally (against a
hand-built embedding set), the price update + attempt-cap fire/redistribute, and the
STRUCTURAL guarantee that the cross-partition pop decision is a pure function of deficits
and prices (a cross-partition p_good comparison is impossible by construction).

Run: uv run pytest tools/atlas/test_deficit_scheduler.py -q
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "atlas", ROOT / "tools" / "scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import deficit_scheduler as D                    # noqa: E402
import release_mix as RM                         # noqa: E402
from tools.emission import cells as C            # noqa: E402


PARTS = ["mandelbrot", "multibrot5", "julia:mandelbrot", "julia:multibrot5"]


def _unit(vec):
    v = np.asarray(vec, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _emb(seed, dim=D.EMB_DIM):
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim))


# --------------------------------------------------------------------------- #
# 1. Order book: the release-mix ratio table, normalized over the tracked partitions.
#
# It used to be `data/emission/target_measure.json` projected down to per-partition marginals
# by `project_type_marginals`, whose whole job was to DIVIDE OUT the morph-cluster count so a
# partition's share came from its multiplier and not from its occupancy. That division now
# happens once on the emission side (`weight_p = share_p / n_cells_p`), so the order book reads
# the partition shares directly and the projection is gone. The cluster-count-independence
# property it protected is asserted below — structurally, and against the emission consumer.
# --------------------------------------------------------------------------- #
def test_order_book_is_the_release_mix_shares():
    fr = D.target_shares(PARTS)
    assert abs(sum(fr.values()) - 1.0) < 1e-12
    assert fr == pytest.approx(RM.shares(PARTS))
    # the policy, read off the ratio table: the two degree-2 planes carry the release equally
    # and each is 3x a supporting family.
    assert fr["mandelbrot"] == pytest.approx(fr["julia:mandelbrot"])
    assert fr["mandelbrot"] / fr["multibrot5"] == pytest.approx(3.0)


def test_order_book_renormalizes_over_the_tracked_partitions_only():
    """A run that tracks two families allocates the whole batch between them, in the table's
    ratio — the shares are relative, so a partition this run does not track cannot silently
    take a slice of it."""
    fr = D.target_shares(["mandelbrot", "multibrot5"])
    assert set(fr) == {"mandelbrot", "multibrot5"}
    assert abs(sum(fr.values()) - 1.0) < 1e-12
    assert fr["mandelbrot"] == pytest.approx(0.75)


def test_order_book_is_independent_of_cluster_count(tmp_path):
    """The campaign-2 preflight property, now structural: the order book never sees a cluster.

    Kept as a test rather than deleted with the projection, because it is the property that
    inverted the whole julia-heavy order when it broke — 102 mandelbrot clusters swamping
    julia:mandelbrot's 4 regardless of the policy. The assertion is now that the emission
    consumer, which DOES see clusters, resolves the same shares anyway."""
    obs = ([("mandelbrot", f"mandelbrot#{i}") for i in range(102)]
           + [("julia:mandelbrot", "julia:mandelbrot#0")])
    parts = ["mandelbrot", "julia:mandelbrot"]
    feasible = C.build_feasible_cells(obs, ["k16:1", "k16:5"], ["smooth", "tia"])
    tm = C.TargetMeasure.from_partition_shares(D.target_shares(parts), feasible)
    assert tm.partition_shares() == pytest.approx(D.target_shares(parts))
    assert tm.partition_shares()["julia:mandelbrot"] == pytest.approx(0.5)


def test_an_unregistered_partition_has_no_order_book_entry():
    """A partition with no declared ratio raises instead of getting a plausible share nobody
    decided — the `release_mix.check_complete` failure, one layer down."""
    with pytest.raises(KeyError):
        D.target_shares(["mandelbrot", "not_a_partition"])


def test_deficit_with_an_empty_tally_is_the_order_book(tmp_path):
    sch = D.DeficitScheduler(PARTS, tmp_path, prices_path=tmp_path / "none.json")
    defs = sch.deficits()
    for p in PARTS:
        assert abs(defs[p] - sch.target_frac[p]) < 1e-12
    assert defs["mandelbrot"] > defs["multibrot5"]


# --------------------------------------------------------------------------- #
# 2. Distinct-look tally against a hand-built embedding set.
# --------------------------------------------------------------------------- #
def test_distinct_look_tally(tmp_path):
    t = D.DistinctLookTally(tmp_path / "looks.npz")
    a = _emb(1)
    # first look is always distinct.
    assert t.add("mandelbrot", a) is True
    # a near-identical look (cos >= 0.974) is NOT a new distinct look.
    near = _unit(a + 0.001 * _emb(999))
    assert float(near @ a) >= 0.974
    assert t.add("mandelbrot", near) is False
    # a clearly different look IS distinct.
    b = _emb(2)
    assert float(b @ a) < 0.974
    assert t.add("mandelbrot", b) is True
    assert t.count("mandelbrot") == 2
    # partitions are independent: the same vector is a fresh distinct look elsewhere.
    assert t.add("multibrot5", a) is True
    assert t.count("multibrot5") == 1
    assert t.total() == 3


def test_tally_persist_roundtrip(tmp_path):
    p = tmp_path / "looks.npz"
    t = D.DistinctLookTally(p)
    t.add("mandelbrot", _emb(1))
    t.add("mandelbrot", _emb(2))
    t.add("julia:mandelbrot", _emb(3))
    t.save()
    t2 = D.DistinctLookTally(p)
    assert t2.counts() == {"mandelbrot": 2, "julia:mandelbrot": 1}
    # the reloaded set still dedups against its persisted members.
    assert t2.add("mandelbrot", _emb(1)) is False


# --------------------------------------------------------------------------- #
# 3. Price update (online EMA of minutes-per-distinct-look).
# --------------------------------------------------------------------------- #
def test_price_ema_update():
    pm = D.PriceModel(PARTS, {"seed_price_min": 3.0, "price_ema": 0.5, "cap_minutes": 100})
    assert pm.price["mandelbrot"] == 3.0
    pm.charge("mandelbrot", 5.0)      # 5 active minutes, no look yet
    pm.record_look("mandelbrot")      # a look after 5 min -> price EMA toward 5
    assert abs(pm.price["mandelbrot"] - (0.5 * 3.0 + 0.5 * 5.0)) < 1e-9
    assert pm.min_since_look["mandelbrot"] == 0.0


# --------------------------------------------------------------------------- #
# 4. Attempt cap fires + redistributes; re-opens on a look.
# --------------------------------------------------------------------------- #
def test_attempt_cap_fire_and_redistribute():
    pm = D.PriceModel(PARTS, {"cap_minutes": 10.0})
    assert pm.charge("mandelbrot", 6.0) is False
    assert "mandelbrot" not in pm.capped
    assert pm.charge("mandelbrot", 6.0) is True    # 12 >= 10 with zero looks -> capped
    assert "mandelbrot" in pm.capped
    # a distinct look re-opens the partition (productive again).
    pm.record_look("mandelbrot")
    assert "mandelbrot" not in pm.capped
    assert pm.min_since_look["mandelbrot"] == 0.0


def test_cap_redistributes_serving():
    # a capped partition is excluded from the pop candidates -> demand redistributes.
    rng = np.random.default_rng(0)
    deficits = {"mandelbrot": 0.9, "multibrot5": 0.1}
    prices = {"mandelbrot": 1.0, "multibrot5": 1.0}
    servable = {"mandelbrot", "multibrot5"}
    # uncapped: highest price-weighted deficit (mandelbrot) wins with no exploration.
    assert D.choose_partition(deficits, prices, set(), servable, rng, explore_floor=0.0) == "mandelbrot"
    # capped: mandelbrot excluded, so multibrot5 is served instead.
    assert D.choose_partition(deficits, prices, {"mandelbrot"}, servable, rng,
                              explore_floor=0.0) == "multibrot5"


# --------------------------------------------------------------------------- #
# 5. Cross-partition p_good comparison is structurally impossible.
# --------------------------------------------------------------------------- #
def test_choose_partition_signature_has_no_pgood():
    sig = inspect.signature(D.choose_partition)
    params = set(sig.parameters)
    assert params == {"deficits", "prices", "capped", "servable", "rng", "explore_floor"}
    # nothing p_good / node / score-shaped in the pop decision's inputs.
    for bad in ("p_good", "pgood", "eord", "score", "node", "nodes", "frontier", "priority"):
        assert bad not in params


def test_choose_partition_ignores_everything_but_deficit_and_price():
    # price-weighted deficit only: with equal deficits, the cheaper partition wins; with
    # equal prices, the larger deficit wins. No other signal can enter (there is none to pass).
    rng = np.random.default_rng(0)
    serv = {"a", "b"}
    assert D.choose_partition({"a": 0.5, "b": 0.5}, {"a": 2.0, "b": 1.0}, set(), serv, rng, 0.0) == "b"
    assert D.choose_partition({"a": 0.8, "b": 0.2}, {"a": 1.0, "b": 1.0}, set(), serv, rng, 0.0) == "a"


def test_choose_partition_none_when_all_capped():
    rng = np.random.default_rng(0)
    assert D.choose_partition({"a": 1.0}, {"a": 1.0}, {"a"}, {"a"}, rng, 0.0) is None
    assert D.choose_partition({"a": 1.0}, {"a": 1.0}, set(), set(), rng, 0.0) is None


# --------------------------------------------------------------------------- #
# 6. Julia routing: twin deficit with an empty queue buys c-plane work.
# --------------------------------------------------------------------------- #
def test_julia_routing_folds_into_cplane(tmp_path):
    sch = D.DeficitScheduler(["multibrot5", "julia:multibrot5"], tmp_path,
                             prices_path=tmp_path / "none.json")
    # force a large julia deficit, zero c-plane deficit.
    sch.target_frac = {"multibrot5": 0.0, "julia:multibrot5": 1.0}
    # julia queue empty, c-plane has nodes -> julia demand routes onto the c-plane parent.
    queue_lens = {"multibrot5": 5, "julia:multibrot5": 0}
    eff = sch.effective_deficits(queue_lens)
    assert eff["multibrot5"] > sch.deficits()["multibrot5"]     # boosted by the twin
    part = sch.pick_partition(queue_lens, np.random.default_rng(0))
    assert part == "multibrot5"           # serving the parent to buy julia looks


def test_julia_routing_no_double_count_when_twin_servable(tmp_path):
    sch = D.DeficitScheduler(["multibrot5", "julia:multibrot5"], tmp_path,
                             prices_path=tmp_path / "none.json")
    sch.target_frac = {"multibrot5": 0.0, "julia:multibrot5": 1.0}
    # julia queue NON-empty -> it competes on its own, no fold onto the parent.
    queue_lens = {"multibrot5": 5, "julia:multibrot5": 3}
    eff = sch.effective_deficits(queue_lens)
    assert abs(eff["multibrot5"] - sch.deficits()["multibrot5"]) < 1e-12


# --------------------------------------------------------------------------- #
# 7. Root allocation follows deficit and sums to the batch.
# --------------------------------------------------------------------------- #
def test_root_allocation_sums_and_favors_deficit(tmp_path):
    sch = D.DeficitScheduler(["mandelbrot", "multibrot5"], tmp_path,
                             prices_path=tmp_path / "none.json")
    sch.target_frac = {"mandelbrot": 0.9, "multibrot5": 0.1}
    rng = np.random.default_rng(0)
    tot = np.zeros(2)
    for _ in range(200):
        a = sch.root_allocation(["mandelbrot", "multibrot5"], 32, rng)
        assert sum(a.values()) == 32          # every draw sums to the batch
        tot += [a["mandelbrot"], a["multibrot5"]]
    assert tot[0] > tot[1]                     # the high-deficit family draws more roots


# --------------------------------------------------------------------------- #
# 8. Scheduler state round-trip (resume safety).
# --------------------------------------------------------------------------- #
def test_seed_from_library_and_resume_safety(tmp_path):
    # Seeding pre-loads the tally with library looks; deficits then measure library-wide
    # scarcity. It seeds ONLY when empty (resume-safe / idempotent) and persists immediately.
    parts = ["mandelbrot", "julia:mandelbrot"]
    embs = {"mandelbrot": np.stack([_emb(10), _emb(11), _emb(12)]),      # 3 distinct looks
            "julia:mandelbrot": np.stack([_emb(20)]),                    # 1 distinct look
            "multibrot5": np.stack([_emb(30)])}                          # untracked -> ignored
    sch = D.DeficitScheduler(parts, tmp_path, prices_path=tmp_path / "none.json")
    seeded = sch.seed_from_library(embs)
    assert seeded == {"mandelbrot": 3, "julia:mandelbrot": 1}
    assert sch.tally.counts() == {"mandelbrot": 3, "julia:mandelbrot": 1}
    assert (tmp_path / "distinct_looks.npz").exists()      # persisted before any batch
    # library-wide scarcity: an admission duplicating a seeded look is NOT a new distinct look.
    assert sch.on_admission("mandelbrot", _emb(10)) is False
    # re-seeding is a no-op (tally non-empty) — never double-counts on resume.
    assert sch.seed_from_library(embs) == {}
    # a genuinely fresh scheduler over the SAME dir reloads the seed from npz and still no-ops.
    sch2 = D.DeficitScheduler(parts, tmp_path, prices_path=tmp_path / "none.json")
    assert sch2.tally.total() == 4
    assert sch2.seed_from_library(embs) == {}


def test_scheduler_state_roundtrip(tmp_path):
    sch = D.DeficitScheduler(PARTS, tmp_path, prices_path=tmp_path / "none.json")
    sch.on_admission("mandelbrot", _emb(1))
    sch.on_admission("mandelbrot", _emb(2))
    sch.charge("multibrot5", 5.0)
    sch.prices.record_look("multibrot5")
    st = sch.state_dict()
    sch.save()

    sch2 = D.DeficitScheduler(PARTS, tmp_path, prices_path=tmp_path / "none.json")
    sch2.load_state(st, reopen_caps=True)
    assert sch2.tally.counts() == {"mandelbrot": 2}          # reloaded from npz
    assert abs(sch2.prices.price["multibrot5"] - sch.prices.price["multibrot5"]) < 1e-9


# --------------------------------------------------------------------------- #
# 9. The unseeded-run guard.
#
# WHERE THE LAST SET FAILED: every test above hands `seed_from_library` a hand-built
# embeddings dict, so the `embeddings is None -> load_library_seed_embeddings()` branch was
# never taken and no test ever touched the real artifact. A test that INJECTS the dependency
# does not cover the loader — which is why the loader's `return {}` on a missing artifact
# survived, and an entire probe run went unseeded looking exactly like a seeded one.
# Cases 1-3 below therefore drive the REAL loader against REAL files on disk.
# --------------------------------------------------------------------------- #
def _write_library(tmp_path, per_partition: dict, *, seed0: int = 100):
    """Materialize a real intake artifact + embedding dir. Returns (intake_path, emb_dir)."""
    intake_dir = tmp_path / "lib"
    emb_dir = tmp_path / "lib_embs"
    intake_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    medoid_id, s = {}, seed0
    for part, n in per_partition.items():
        for k in range(n):
            loc_id = f"{part.replace(':', '_')}_{k}"
            medoid_id[f"{part}#{k}"] = loc_id
            np.save(emb_dir / f"{loc_id}.npy", _emb(s).astype(np.float32))
            s += 1
    ip = intake_dir / "intake.json"
    ip.write_text(json.dumps({"medoid_id": medoid_id}), encoding="utf-8")
    return ip, emb_dir


def _sched(tmp_path, parts):
    return D.DeficitScheduler(parts, tmp_path / "run",
                              prices_path=tmp_path / "none.json")


# --- case 1: missing artifact, no flag -> abort, message names the path ----- #
def test_missing_artifact_aborts_and_names_the_path(tmp_path):
    missing = tmp_path / "gone" / "intake.json"
    embs_dir = tmp_path / "gone_embs"
    assert not missing.exists()

    # the loader itself stays total (returns {}) — it is the low-level read.
    assert D.load_library_seed_embeddings(missing, embs_dir) == {}

    # the GUARD is what fails closed, and it names both paths so the abort is actionable.
    with pytest.raises(D.UnseededRunError) as ei:
        D.require_library_seed(intake_path=missing, emb_dir=embs_dir)
    msg = str(ei.value)
    assert str(missing) in msg and str(embs_dir) in msg
    assert "--allow-unseeded" in msg

    # and the same guard fires through the scheduler, before any look is tallied.
    sch = _sched(tmp_path, ["mandelbrot"])
    with pytest.raises(D.UnseededRunError):
        sch.seed_from_library(intake_path=missing, emb_dir=embs_dir)
    assert sch.tally.total() == 0
    assert not (tmp_path / "run" / "distinct_looks.npz").exists()   # nothing persisted


def test_artifact_present_but_embeddings_missing_also_aborts(tmp_path):
    # "absent OR EMPTY": an intake that exists but whose embedding dir is gone yields no
    # medoids. That is an unseeded run just the same, and must not slip through.
    import shutil
    ip, emb_dir = _write_library(tmp_path, {"mandelbrot": 3})
    shutil.rmtree(emb_dir)
    with pytest.raises(D.UnseededRunError) as ei:
        D.require_library_seed(intake_path=ip, emb_dir=emb_dir)
    assert "no usable medoid embeddings" in str(ei.value)


# --- case 2: missing artifact + override -> proceeds, stamped unseeded ------ #
def test_allow_unseeded_proceeds_and_stamps_the_summary(tmp_path):
    missing = tmp_path / "gone" / "intake.json"
    sch = _sched(tmp_path, ["mandelbrot", "julia:mandelbrot"])
    seeded = sch.seed_from_library(allow_unseeded=True,
                                   intake_path=missing, emb_dir=tmp_path / "gone_embs")
    assert seeded == {}                                  # proceeded, seeded nothing
    assert sch.tally.total() == 0

    rec = sch.summary()["library_seed"]
    assert rec["status"] == "unseeded"                   # the durable stamp
    assert rec["seeded"] is False and rec["seeded_looks"] == 0
    assert rec["allow_unseeded"] is True
    assert rec["source"] == str(missing)                 # WHICH path was missing
    assert rec["source_exists"] is False
    assert "reason" in rec
    assert json.dumps(sch.summary())                     # and the stamp is JSON-serializable


# --- case 3: artifact present on disk -> seeds, stamped with source + count - #
def test_real_artifact_on_disk_seeds_and_stamps_source_and_count(tmp_path):
    ip, emb_dir = _write_library(tmp_path, {"mandelbrot": 3, "julia:mandelbrot": 2,
                                            "multibrot5": 4})   # multibrot5 untracked
    parts = ["mandelbrot", "julia:mandelbrot"]
    sch = _sched(tmp_path, parts)
    seeded = sch.seed_from_library(intake_path=ip, emb_dir=emb_dir)   # REAL loader, REAL files
    assert seeded == {"mandelbrot": 3, "julia:mandelbrot": 2}
    assert sch.tally.counts() == {"mandelbrot": 3, "julia:mandelbrot": 2}
    assert (tmp_path / "run" / "distinct_looks.npz").exists()

    rec = sch.summary()["library_seed"]
    assert rec["status"] == "seeded" and rec["seeded"] is True
    assert rec["seeded_looks"] == 5                       # how many looks it seeded
    assert rec["per_partition"] == {"mandelbrot": 3, "julia:mandelbrot": 2}
    assert rec["source"] == str(ip)                       # what it seeded FROM
    assert rec["emb_dir"] == str(emb_dir)
    assert rec["library_looks"] == 9                      # incl. the untracked family
    assert rec["tracked_partitions"] == parts
    assert json.dumps(sch.summary())


def test_resume_is_stamped_resume_and_never_aborts(tmp_path):
    # A resumed tally IS the seed. Re-checking a since-moved artifact must not abort a
    # legitimate resume — but the summary still says so rather than claiming "seeded".
    ip, emb_dir = _write_library(tmp_path, {"mandelbrot": 2})
    sch = _sched(tmp_path, ["mandelbrot"])
    sch.seed_from_library(intake_path=ip, emb_dir=emb_dir)
    sch2 = _sched(tmp_path, ["mandelbrot"])               # same run dir -> reloads npz
    assert sch2.tally.total() == 2
    gone = tmp_path / "vanished" / "intake.json"
    assert sch2.seed_from_library(intake_path=gone, emb_dir=tmp_path / "vanished") == {}
    assert sch2.summary()["library_seed"]["status"] == "resume"


def test_never_attempted_is_reported_not_omitted(tmp_path):
    # A scheduler whose seed_from_library was never called reports it loudly rather than
    # emitting a summary that merely lacks the key (campaign-2's exact failure shape).
    sch = _sched(tmp_path, ["mandelbrot"])
    rec = sch.summary()["library_seed"]
    assert rec["status"] == "never_attempted" and rec["seeded"] is False


def test_empty_injected_dict_fails_closed_too(tmp_path):
    # Injection must not be a bypass: an empty dict is an unseeded run however it arrived.
    sch = _sched(tmp_path, ["mandelbrot"])
    with pytest.raises(D.UnseededRunError):
        sch.seed_from_library({})
    assert sch.seed_from_library({}, allow_unseeded=True) == {}
    assert sch.summary()["library_seed"]["status"] == "unseeded"


# --- case 4: red-before proof that the OLD path continued silently ---------- #
def test_red_before_old_call_shape_continued_silently(tmp_path):
    """The pre-guard code path, reconstructed verbatim, and the proof it was silent.

    Old caller (steered_frontier.__init__):  self.scheduler.seed_from_library()
    Old callee:                              embeddings = load_library_seed_embeddings()
                                             -> {} on a missing artifact -> seeds nothing
                                             -> return {} -> caller logs "newly seeded 0"

    Every observable was a legal value of a normal run: {} is also what a RESUME returns, and
    the summary carried no seed field at all. Nothing anywhere said "unseeded". This pins that
    state as the bug and asserts the guard now makes it unreachable unmarked.
    """
    missing = tmp_path / "gone" / "intake.json"

    # 1. the old two-step, replayed exactly: loader returns {}, seeding it is a silent no-op.
    old_embeddings = D.load_library_seed_embeddings(missing, tmp_path / "gone_embs")
    assert old_embeddings == {}
    sch_old = _sched(tmp_path, ["mandelbrot"])
    silent = sch_old.seed_from_library(old_embeddings, allow_unseeded=True)  # the old behaviour
    assert silent == {}                       # <- indistinguishable from a resume's {}
    assert sch_old.tally.total() == 0         # <- nothing seeded, no error, run would continue

    # 2. RED-BEFORE: under the old code the summary had no seed field at all, so that `{}` was
    #    the entire record of an unseeded run. The old shape is now impossible — the fact is
    #    present in the summary, and it says unseeded.
    assert sch_old.summary()["library_seed"]["status"] == "unseeded"

    # 3. and the unoverridden path — the one the old code took by default — now aborts.
    sch_new = _sched(tmp_path / "b", ["mandelbrot"])
    with pytest.raises(D.UnseededRunError):
        sch_new.seed_from_library(intake_path=missing, emb_dir=tmp_path / "gone_embs")


# =========================================================================== #
# Seed-path CLASS refusal — a scratch path is refused at RESOLVE time.
#
# Both seed sources have now been lost to `scratch/`: campaign1's vectors (permanently —
# its snapshot went with them) and library_seed_v2's (recoverable, 168 vectors). In both
# cases every check in the tree was green right up to the wipe, because a scratch path that
# has not been deleted YET is indistinguishable from a durable one. So the check has to be
# on the PATH's class, not on the file's presence, and it has to fire where the path is
# produced rather than where it is read.
# =========================================================================== #
def test_no_seed_source_resolves_under_scratch():
    """The registry invariant, over the LIVE table — not a synthetic one.

    This is the assertion whose absence cost both seeds. It reads `SEED_SOURCES` directly
    rather than going through `resolve_seed_source`, because resolution stops at the first
    source that EXISTS: routing through it would leave a dormant scratch path in a
    not-yet-resolving entry unchecked, which is exactly campaign1's shape."""
    for name, intake, emb in D.SEED_SOURCES:
        D._refuse_scratch_class(f"intake ({name})", intake)
        D._refuse_scratch_class(f"embeddings ({name})", emb)


@pytest.mark.parametrize("victim", ["intake", "emb"])
def test_an_INJECTED_scratch_source_is_refused_at_resolve_time(tmp_path, monkeypatch, victim):
    """INJECTION-PROVEN. The invariant test above passes trivially on a clean registry, so
    it cannot on its own show the guard would catch a regression. Plant a scratch path in
    the registry — one half at a time, since a source that got its snapshot right and its
    vectors wrong is precisely what shipped — and require resolution to REFUSE.

    Planted under a fake repo root so the real tree is untouched, and the path is made to
    EXIST: a guard that only fires on missing files would be the presence check all over
    again, and the whole point is that the live scratch path was present and healthy."""
    fake_root = tmp_path / "repo"
    good = fake_root / "data" / "emission" / "x"
    bad = fake_root / "scratch" / "emission" / "x"
    for d in (good, bad):
        d.mkdir(parents=True)
    (good / "intake.json").write_text("{}", encoding="utf-8")
    (bad / "intake.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(D, "ROOT", fake_root)
    ip = (bad if victim == "intake" else good) / "intake.json"
    ed = (bad if victim == "emb" else good) / "embs"
    monkeypatch.setattr(D, "SEED_SOURCES", (("planted", ip, ed),))

    with pytest.raises(D.SeedPathClassError, match="scratch"):
        D.resolve_seed_source()
    # ...and it is refused through every seam a run can reach it by, not just the first
    with pytest.raises(D.SeedPathClassError):
        D.library_seed_paths()
    with pytest.raises(D.SeedPathClassError):
        D.require_library_seed(allow_unseeded=True)


def test_an_EXPLICIT_scratch_path_is_refused_too(tmp_path, monkeypatch):
    """The easier hole of the two. The registry is reviewed; a `--intake`/`--emb-dir` on a
    launch line is not, and "just point it at the copy in scratch for now" is how a
    temporary path becomes the production one. `allow_unseeded` must NOT wave it through:
    that flag means "the seed is absent and I accept run-local numbers", not "the seed is
    misconfigured and I accept whatever happens"."""
    monkeypatch.setattr(D, "ROOT", tmp_path)
    bad = tmp_path / "scratch" / "embs"
    bad.mkdir(parents=True)
    with pytest.raises(D.SeedPathClassError):
        D.library_seed_paths(emb_dir=bad)
    with pytest.raises(D.SeedPathClassError):
        D.require_library_seed(allow_unseeded=True, emb_dir=bad)
    with pytest.raises(D.SeedPathClassError):
        D.library_seed_paths(intake_path=tmp_path / "scratch" / "intake.json")


def test_both_disposable_trees_are_refused_not_just_scratch(tmp_path, monkeypatch):
    """`scratchpad/` is refused as well as `scratch/`. CLAUDE.md names BOTH disposable —
    "neither scratch tree is a dependency tier" — and `scratchpad/visual_dup/embed.py` was
    load-bearing, uncommitted, and vanished. A guard that covered only `scratch/` would send
    the next seed one directory sideways into the same failure.

    Note this deliberately differs from `artifacts._is_discovery_scratch`, which excludes
    `scratchpad` — that predicate answers "which family relocates", a different question
    from "may a seed live here"."""
    monkeypatch.setattr(D, "ROOT", tmp_path)
    for bad in ("scratch", "scratchpad"):
        with pytest.raises(D.SeedPathClassError, match=bad):
            D._refuse_scratch_class("x", tmp_path / bad / "emission" / "embs")


def test_the_refusal_is_component_exact_and_root_relative(tmp_path, monkeypatch):
    """Two false-positive classes the guard must NOT hit, or it becomes noise someone routes
    around: a component that merely STARTS with "scratch" is a different directory (the same
    component-exact rule `artifacts._is_discovery_scratch` uses), and an ARTIFACTS_ROOT that
    happens to contain the letters "scratch" in its own prefix is not the disposable class —
    that prefix is the operator's volume name, not our contract."""
    monkeypatch.setattr(D, "ROOT", tmp_path)
    D._refuse_scratch_class("x", tmp_path / "data" / "scratch_notes" / "embs")
    D._refuse_scratch_class("x", tmp_path / "data" / "emission" / "scratches" / "e")
    weird = tmp_path / "my_scratch_volume"
    monkeypatch.setattr(D._artifacts, "artifacts_root", lambda: weird)
    D._refuse_scratch_class("x", weird / "data" / "emission" / "library_seed_v2" / "embs")
    # ...but a real `scratch/` component UNDER that root is still refused
    with pytest.raises(D.SeedPathClassError):
        D._refuse_scratch_class("x", weird / "scratch" / "embs")


def test_the_live_seed_is_bulk_registered_so_bulk_does_not_mean_in_tree():
    """`bulk()` relocates a REGISTERED family and otherwise resolves in-tree. The old seed
    declared `bulk()` at its write site and still landed in `scratch/`, so the declaration
    alone proves nothing — the registration is the half that does the work, and
    `!/data/emission/` means an unregistered path here would be COMMITTED rather than merely
    present (the tau_h_rederive precedent)."""
    import artifacts as A
    for rel in ("data/emission/library_seed_v2/embs", "data/emission/campaign1/embs"):
        assert A.is_relocated(rel), f"{rel} would resolve in-tree and be committed"
        assert A.resolve(rel) == A.artifacts_root() / rel
