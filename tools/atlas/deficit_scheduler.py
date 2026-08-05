#!/usr/bin/env python
"""deficit_scheduler.py — family-level (cross-partition) deficit scheduler for the
steered frontier (v1).

WHY. Campaign 1 proved the frontier's single global priority queue lets raw p_good
allocate ACROSS families — a comparison the classifier is not calibrated for (a family's
mean p_good is *negatively* correlated with its human good-rate). p_good's certified role
ends at the family boundary: it steers and floors WITHIN a partition only. Cross-family
allocation therefore has to become an explicit scheduler, and this is it.

HARD SCOPE INVARIANT. **No p_good value is ever compared across partitions anywhere in
this module.** The within-partition priority (E[ord]+Gumbel-dup-novelty+beta*depth) is
computed by the caller and only ever sorts nodes of ONE partition. The cross-partition
choice here uses ONLY per-partition DEFICITS and PRICES. `choose_partition` is a pure
function of (deficits, prices, capped, servable) precisely so this is testable and
structurally guaranteed (test_deficit_scheduler.py).

MECHANICS (spec: prompts/deficit_scheduler.md).
 1. Per-partition sub-queues. The existing priority formula is unchanged within a
    partition; the caller keeps one frontier list keyed by `partition` and pops the
    top-B of whichever partition this scheduler names.
 2. Each batch, serve the partition with the largest PRICE-WEIGHTED deficit
    (deficit / price = deficit per unit expected cost), with a small exploration floor
    so no partition with remaining demand starves on a stale price.
 3. Deficit is denominated in DISTINCT LOOKS, not admissions. `DistinctLookTally` keeps a
    per-partition CLIP-embedding set (the library morph recipe, embedded by the caller);
    an admission counts iff its max cosine vs that partition's admitted-look set is < 0.974.
 4. Prices = active-minutes per distinct look, per partition. Seeded from a config file,
    updated online (EMA). A partition that burns `cap_minutes` of active time with zero new
    distinct looks is capped (demand redistributed); caps re-open on resume/config.
 5. Target = the canonical release-mix ratio table (`tools/scoring/release_mix.py`)
    normalized over this run's tracked partitions. The order book; no separate
    discovery-side target file, and no separate emission one either.
 6. Julia twins are BOUGHT, not popped: julia:X demand routes into (a) the root family mix
    and (b) willingness to spend c-plane X expansions on its behalf (see JULIA ROUTING).
 7. Root draws are deficit-aware under the same (twin-inclusive) rule.

The emission cell's first axis == the ledger `family` == our `partition`
(mandelbrot / multibrot{3,4,5} / julia:{...} / phoenix / phoenix:classic). Both stages take
their per-partition shares from the SAME derived source (`release_mix.shares`), so the
discovery order book and the emission measure cannot drift into two policies about the same
partitions — which is what they had been (mandelbrot 9.0% here vs 22.7% intended).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import release_mix as RM               # noqa: E402  THE per-partition release-mix ratio table
from tools import paths as _paths      # noqa: E402  (storage-class declaration at the seam)
from tools.corpus import artifacts as _artifacts   # noqa: E402  (ARTIFACTS_ROOT, for the class check)

# ------------------------------------------------------------------------- #
# Defaults / paths.
# ------------------------------------------------------------------------- #
DEFAULT_PRICES_PATH = ROOT / "data" / "atlas" / "scheduler_prices.json"
INTAKE_ARTIFACT = ROOT / "data" / "emission" / "campaign1" / "intake.json"   # durable snapshot
# bulk (regenerable), resolved through the ARTIFACTS_ROOT resolver. It was
# `scratch/emission/campaign1/embs`, and that is the whole reason campaign1 is dark today:
# `scratch/` is the one class whose contract GUARANTEES deletion, so the vectors were
# wiped and — campaign1's snapshot having gone with them — cannot be rebuilt. Never
# reintroduce a scratch path here; `_refuse_scratch_class` now makes it raise.
INTAKE_EMB_DIR = _paths.bulk("data/emission/campaign1/embs")

NEAR_DUP_THRESHOLD = 0.974   # distinct-look cosine knee (== emission/descriptor)
EMB_DIM = 768                # CLIP vit_base_patch16_clip_224.openai

# price-model / scheduling defaults (overridable in the prices config)
SEED_PRICE_MIN = 3.0         # neutral per-partition seed: active-minutes / distinct look
PRICE_EMA = 0.30            # online price smoothing (weight on the newest per-look sample)
CAP_MINUTES = 20.0          # attempt cap: active minutes with zero new looks before capping
EXPLORE_FLOOR = 0.10        # prob. of a uniform draw among partitions with remaining demand
JULIA_ROUTE_GAIN = 1.0      # weight of a twin's deficit folded into its c-plane parent


# ------------------------------------------------------------------------- #
# Partition topology helpers (mirror production_seeder grammar; kept local so this module
# imports no torch-side code).
# ------------------------------------------------------------------------- #
def is_julia(partition: str) -> bool:
    return partition.startswith("julia:")


def cplane_of(partition: str) -> str | None:
    """The c-plane parent a julia partition descends off (julia:X -> X); None for c-plane."""
    return partition.split(":", 1)[1] if is_julia(partition) else None


def julia_partition(cplane: str) -> str:
    return f"julia:{cplane}"


# ------------------------------------------------------------------------- #
# 5. Order book: the release-mix ratio table, normalized over the tracked partitions.
# ------------------------------------------------------------------------- #
def target_shares(partitions: list[str]) -> dict:
    """`{partition: intended share}` — the order book. THE derived source, shared with the
    emission measure (`cells.TargetMeasure.from_partition_shares` re-solves the same numbers
    against emission's feasible cells; `test_release_mix_one_source.py` asserts the two agree).

    A partition with no registered ratio raises rather than defaulting to a plausible-looking
    share nobody decided (`release_mix.ratio_of`)."""
    return RM.shares(list(partitions))


def load_observed_type_cluster(partitions: list[str]) -> list[tuple]:
    """(type, cluster) pairs anchoring the feasible grid. Prefer the emission intake artifact
    (real within-family morph clusters); else fall back to one cluster per tracked partition
    (uniform-over-present base that overrides then skew). Restricted to `partitions` so the
    order book covers exactly the run's tracked families."""
    pset = set(partitions)
    if INTAKE_ARTIFACT.exists():
        tags = json.loads(INTAKE_ARTIFACT.read_text(encoding="utf-8")).get("cluster_tags", {})
        obs = set()
        for tag in tags.values():                 # tag == "<type>#<k>"
            t = tag.rsplit("#", 1)[0]
            if t in pset:
                obs.add((t, tag))
        # any tracked partition the artifact never observed still gets a singleton anchor,
        # so it has non-zero target and can be served.
        for p in partitions:
            if not any(t == p for (t, _c) in obs):
                obs.add((p, f"{p}#0"))
        return sorted(obs)
    return [(p, f"{p}#0") for p in partitions]


# The order book is a per-PARTITION share and is read as one. It used to be the joint emission
# measure projected down — `marginal(type) ∝ MEAN cell weight over that type's cells`, the mean
# (not the sum) precisely so a type's share did not scale with its morph-cluster COUNT, which is
# occupancy and belongs on the deficit's pool side. That division is now done once, on the
# emission side, by `TargetMeasure.from_partition_shares` (weight_p = share_p / n_cells_p), so
# there is nothing left here to project: a ratio table is already a partition-level share.


# ------------------------------------------------------------------------- #
# Library-look seed: the campaign-1 intake's per-cluster medoid embeddings, grouped by
# partition (== family). Deficits must measure LIBRARY-WIDE scarcity, not run-local scarcity,
# so a fresh run's distinct-look tally is pre-loaded with the looks the library already holds
# (and their embeddings become dedup memory: a new admission near a known library look is not
# counted as new). Same CLIP recipe (morph_gray -> vit_base_patch16_clip_224.openai, 768-d)
# emission clusters at cos 0.974, so the seed is metric-consistent with the tally.
# ------------------------------------------------------------------------- #
class UnseededRunError(RuntimeError):
    """A `--scheduler` run found no library seed and did not pass `--allow-unseeded`.

    Fail-closed in the same shape as `v7.build_manifest.assign_split`: the unsafe state is
    never a silent fall-through. An unseeded run's deficits measure RUN-LOCAL scarcity while
    every downstream reader assumes LIBRARY-WIDE scarcity, and — this is what made the last
    one expensive — its numbers are indistinguishable from a seeded run's afterwards. So the
    absent seed aborts, and the override stamps the run summary (see `seed_record`)."""


# The ordered registry of seed SOURCES. Resolution is by existence, first match wins, and the
# winner is stamped into the seed record (`source` / `emb_dir`) — so this is a documented
# resolution order, NOT a silent fallback: a reader of a run summary can always tell which
# source seeded it, and a source that resolved to nothing is visible as the next one winning.
#
# campaign1 is first because it is the older and larger library snapshot. It is currently
# ABSENT (the snapshot was never rebuilt after the derived-artifact wipe and its embeddings
# lived in `scratch/`, a class whose contract guarantees deletion), which is why the registry
# exists at all — `library_seed_v2` is the relit seed built from Matt's own >=3 verdicts
# (`tools/emission/library_seed_v2.py`), and unlike campaign1 its embeddings are rebuildable
# from its own snapshot.
SEED_SOURCES = (
    ("campaign1", INTAKE_ARTIFACT, INTAKE_EMB_DIR),
    ("library_seed_v2",
     ROOT / "data" / "emission" / "library_seed_v2" / "intake.json",
     _paths.bulk("data/emission/library_seed_v2/embs")),
)


class SeedPathClassError(RuntimeError):
    """A seed-critical path resolved to somewhere under `scratch/`.

    BOTH seed sources have now been lost this way — campaign1 permanently — so the class
    error is raised at RESOLVE time rather than left to be discovered at read time. The
    difference matters: a scratch path that has not been wiped YET reads as perfectly
    healthy, seeds a run, and is indistinguishable from a durable one until the wipe. By
    then the run is over and its numbers are already on record as library-wide. Refusing
    the path itself is the only check that fires while the mistake is still cheap.

    Deliberately NOT an `UnseededRunError`: that one means "the seed is absent, and you may
    proceed anyway with `--allow-unseeded`". This one means "the seed is misconfigured",
    which no run flag should be able to wave through."""


def _refuse_scratch_class(kind: str, p) -> Path:
    """Return `p`, or raise `SeedPathClassError` if it names a disposable-class directory.

    The RULE (which components count, and how they are matched below the two resolver
    roots) is `paths.disposable_component` — one owner, because the harvest-log registry
    now has to refuse the same class for its own reasons. What stays here is the part that
    is about SEEDS: the exception type and the message."""
    path = Path(p)
    hit = _paths.disposable_component(path, (ROOT, _artifacts.artifacts_root()))
    if hit is None:
        return path
    raise SeedPathClassError(
        f"seed {kind} resolves under the disposable `{hit}/` class, which GUARANTEES "
        f"deletion:\n"
        f"    path : {path}\n"
        f"A seed that a `rm -r scratch/*` can delete is a seed that will silently become "
        f"empty, and an empty seed makes a run measure RUN-LOCAL scarcity while every "
        f"reader assumes library-wide. campaign1 was lost exactly this way and cannot be "
        f"rebuilt.\n"
        f"Declare the class at the write site and name a path that agrees with it: "
        f"`paths.bulk('data/.../embs')` (register the prefix in "
        f"`tools/corpus/artifacts.RELOCATED_PREFIXES`), or `paths.durable(...)`."
    )


def resolve_seed_source() -> tuple[str, Path, Path]:
    """`(name, intake, emb_dir)` of the first registered source whose snapshot EXISTS.

    Falls back to the FIRST entry when none exists, so the error message names the primary
    artifact rather than the last one tried — "campaign1 is missing" is the actionable
    message; "library_seed_v2 is missing" would send a reader to rebuild the wrong thing.

    Both halves of the winner are class-checked before they are returned (see
    `_refuse_scratch_class`), so a scratch path cannot enter the pipeline by being
    registered — which is how both seeds were lost."""
    for name, ip, ed in SEED_SOURCES:
        if Path(ip).exists():
            return name, _refuse_scratch_class(f"intake ({name})", ip), \
                _refuse_scratch_class(f"embeddings ({name})", ed)
    name, ip, ed = SEED_SOURCES[0]
    return (name, _refuse_scratch_class(f"intake ({name})", ip),
            _refuse_scratch_class(f"embeddings ({name})", ed))


def library_seed_paths(intake_path: Path | None = None,
                       emb_dir: Path | None = None) -> tuple[Path, Path]:
    """The (intake artifact, embedding dir) a seed would be loaded from. Shared by the loader,
    the guard and the error message so all three name the same paths.

    With neither argument given, the registry resolves (see `resolve_seed_source`). An
    explicit path always wins and is never mixed with a resolved one — half an explicit pair
    would silently pair one source's snapshot with another's vectors.

    An EXPLICIT path is class-checked too. It is the easier hole of the two: the registry is
    reviewed, a `--intake`/`--emb-dir` flag on a launch line is not, and "just point it at
    the copy in scratch for now" is how a temporary path becomes a production one."""
    if intake_path is not None or emb_dir is not None:
        _n, dip, ded = resolve_seed_source()
        return (_refuse_scratch_class("intake (explicit)", intake_path)
                if intake_path else dip,
                _refuse_scratch_class("embeddings (explicit)", emb_dir)
                if emb_dir else ded)
    _name, ip, ed = resolve_seed_source()
    return ip, ed


def load_library_seed_embeddings(intake_path: Path | None = None,
                                 emb_dir: Path | None = None) -> dict[str, np.ndarray]:
    """partition -> (N, 768) float32 medoid embeddings from the campaign-1 intake. One medoid
    (cluster founder) per distinct look. Returns {} if the intake artifact is absent.

    This is the LOW-LEVEL loader and it stays total (no raise) — the fail-closed decision is
    `require_library_seed` / `DeficitScheduler.seed_from_library`, so callers that legitimately
    want "seed if you can" keep a way to ask. Do NOT reintroduce a caller that consumes {}
    from here and shrugs: that is the exact shape that sent a whole probe run unseeded."""
    ip, ed = library_seed_paths(intake_path, emb_dir)
    if not ip.exists():
        return {}
    intake = json.loads(ip.read_text(encoding="utf-8"))
    medoid_id = intake.get("medoid_id", {})          # cluster_tag "<family>#<k>" -> location id
    # The snapshot's tags are FAMILY-keyed, so a classic-phoenix look is filed under `phoenix`
    # and `phoenix:classic` seeds empty — a partition that looks starved to every deficit that
    # reads this tally. Re-keyed at the READ by the same resolver the emission cell axis uses
    # (the frozen artifact is not rewritten); imported lazily so this module stays importable
    # without the corpus stack.
    from tools.emission.descriptor import seed_cluster_tags   # noqa: PLC0415
    rekeyed = seed_cluster_tags(intake)               # location id -> "<partition>#<k>"
    by_part: dict[str, list] = defaultdict(list)
    for tag, loc_id in medoid_id.items():
        part = (rekeyed.get(loc_id) or tag).rsplit("#", 1)[0]
        p = ed / f"{loc_id}.npy"
        if not p.exists():
            continue
        e = np.load(p).astype(np.float32).reshape(-1)
        if e.shape[0] != EMB_DIM:
            continue
        by_part[part].append(e / (np.linalg.norm(e) + 1e-9))
    return {p: np.stack(v).astype(np.float32) for p, v in by_part.items() if v}


def require_library_seed(*, allow_unseeded: bool = False,
                         intake_path: Path | None = None,
                         emb_dir: Path | None = None) -> dict:
    """Fail-closed preflight for a `--scheduler` run: load the library seed and REFUSE to
    return if it is absent or empty, unless `allow_unseeded` is passed.

    Returns a SEED RECORD — the thing that gets stamped into the run summary, so a reader of
    that summary months later can tell a seeded run from an unseeded one without re-deriving
    anything. Keys: status ("seeded" | "unseeded"), source / emb_dir (the paths consulted),
    source_exists, library_looks (looks available in the artifact), library_partitions,
    allow_unseeded, plus `embeddings` (the loaded matrices; stripped before stamping by
    `seed_stamp`).

    Raises UnseededRunError naming both paths when the seed is missing and not overridden."""
    ip, ed = library_seed_paths(intake_path, emb_dir)
    embs = load_library_seed_embeddings(ip, ed)
    n_looks = int(sum(int(m.shape[0]) for m in embs.values()))
    resolved = next((name for name, sip, _ in SEED_SOURCES
                     if Path(sip) == Path(ip)), "explicit")
    rec = dict(status="seeded" if n_looks else "unseeded",
               source=str(ip), emb_dir=str(ed), source_exists=bool(ip.exists()),
               resolved_from=resolved,
               registry=[dict(name=n, intake=str(p), exists=Path(p).exists())
                         for n, p, _ in SEED_SOURCES],
               library_looks=n_looks, library_partitions=sorted(embs),
               allow_unseeded=bool(allow_unseeded), embeddings=embs)
    if n_looks:
        return rec
    why = ("the intake artifact does not exist" if not ip.exists() else
           "the intake artifact exists but yielded no usable medoid embeddings "
           "(embedding dir missing, empty, or wrong dimension)")
    rec["reason"] = why
    if not allow_unseeded:
        raise UnseededRunError(
            f"--scheduler run has NO library seed, so its deficits would measure run-local "
            f"scarcity while every reader assumes library-wide. Aborting before any work.\n"
            f"    reason     : {why}\n"
            f"    intake     : {ip}\n"
            f"    embeddings : {ed}\n"
            f"Rebuild the intake snapshot, or pass --allow-unseeded to proceed deliberately "
            f"(the run summary is then permanently stamped status=unseeded)."
        )
    return rec


def seed_stamp(rec: dict | None) -> dict | None:
    """The stampable projection of a seed record: everything except the embedding matrices
    (which are megabytes and not JSON). Used for the run summary."""
    if rec is None:
        return None
    return {k: v for k, v in rec.items() if k != "embeddings"}


# ------------------------------------------------------------------------- #
# 3. Distinct-look tally — per-partition CLIP-embedding set + max-cosine gate.
# ------------------------------------------------------------------------- #
class DistinctLookTally:
    """Per-partition set of admitted-look embeddings (L2-normalized, N x 768). An admission's
    embedding counts as a NEW distinct look iff its max cosine vs that partition's existing
    set is < NEAR_DUP_THRESHOLD; on a distinct look the embedding joins the set. Pure numpy
    (embeddings produced by the caller) so it is unit-testable with hand-built vectors, and
    serialized to an npz (per-partition matrices) for lossless resume."""

    def __init__(self, path: Path, threshold: float = NEAR_DUP_THRESHOLD):
        self.path = Path(path)
        self.threshold = float(threshold)
        self.sets: dict[str, np.ndarray] = {}    # partition -> (N, 768) float32
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            for k in z.files:
                self.sets[k] = z[k].astype(np.float32)

    def count(self, partition: str) -> int:
        m = self.sets.get(partition)
        return 0 if m is None else int(m.shape[0])

    def counts(self) -> dict:
        return {p: int(m.shape[0]) for p, m in self.sets.items()}

    def total(self) -> int:
        return sum(int(m.shape[0]) for m in self.sets.values())

    def add(self, partition: str, emb) -> bool:
        """Test-and-add: True (and appends) iff `emb` is a new distinct look for `partition`."""
        e = np.asarray(emb, np.float32).reshape(1, EMB_DIM)
        e = e / (np.linalg.norm(e) + 1e-9)
        m = self.sets.get(partition)
        if m is not None and m.shape[0]:
            cos_max = float((m @ e[0]).max())
            if cos_max >= self.threshold:
                return False
        self.sets[partition] = e if m is None else np.concatenate([m, e], axis=0)
        return True

    def save(self):
        if not self.sets:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / (self.path.stem + "_tmp.npz")
        np.savez_compressed(tmp, **{p: m for p, m in self.sets.items()})
        import os
        os.replace(tmp, self.path)


# ------------------------------------------------------------------------- #
# 4. Price model — active-minutes per distinct look, per partition, w/ attempt caps.
# ------------------------------------------------------------------------- #
class PriceModel:
    """Per-partition price = active-minutes / distinct look. Seeded from config, updated
    online by an EMA of the minutes spent between successive distinct looks. Attempt cap:
    a partition that accrues `cap_minutes` of active time with zero new distinct looks is
    capped (excluded from serving; its demand redistributes). Caps re-open on resume /
    config reload — nothing here is trusted from a checkpoint that config can override."""

    def __init__(self, partitions: list[str], config: dict | None = None):
        config = config or {}
        seeds = config.get("prices", {})
        self.seed_default = float(config.get("seed_price_min", SEED_PRICE_MIN))
        self.ema = float(config.get("price_ema", PRICE_EMA))
        self.cap_minutes = float(config.get("cap_minutes", CAP_MINUTES))
        self.price = {p: float(seeds.get(p, self.seed_default)) for p in partitions}
        self.min_since_look = {p: 0.0 for p in partitions}   # minutes since last distinct look
        self.min_spent = {p: 0.0 for p in partitions}        # cumulative active minutes
        self.capped: set[str] = set()

    def ensure(self, partition: str):
        if partition not in self.price:
            self.price[partition] = self.seed_default
            self.min_since_look[partition] = 0.0
            self.min_spent[partition] = 0.0

    def charge(self, partition: str, minutes: float):
        """Account `minutes` of active time to `partition`. Trips the attempt cap if the
        partition has now burned cap_minutes with zero new looks. Returns True iff it just
        capped this call."""
        self.ensure(partition)
        self.min_spent[partition] += minutes
        self.min_since_look[partition] += minutes
        if (partition not in self.capped
                and self.min_since_look[partition] >= self.cap_minutes):
            self.capped.add(partition)
            return True
        return False

    def record_look(self, partition: str):
        """A new distinct look landed in `partition`: update its price EMA from the minutes
        spent since the last look, reset the dry-time counter, and un-cap it (it is
        productive again)."""
        self.ensure(partition)
        sample = self.min_since_look[partition]
        if sample > 0.0:
            self.price[partition] = (1 - self.ema) * self.price[partition] + self.ema * sample
        self.min_since_look[partition] = 0.0
        self.capped.discard(partition)

    def reopen_caps(self):
        """Re-open every capped partition (resume / config reload)."""
        for p in list(self.capped):
            self.min_since_look[p] = 0.0
        self.capped.clear()

    def state_dict(self) -> dict:
        return dict(price=self.price, min_since_look=self.min_since_look,
                    min_spent=self.min_spent, capped=sorted(self.capped),
                    ema=self.ema, cap_minutes=self.cap_minutes,
                    seed_default=self.seed_default)

    def load_state(self, d: dict):
        self.price.update({k: float(v) for k, v in d.get("price", {}).items()})
        self.min_since_look.update({k: float(v) for k, v in d.get("min_since_look", {}).items()})
        self.min_spent.update({k: float(v) for k, v in d.get("min_spent", {}).items()})
        self.capped = set(d.get("capped", []))


# ------------------------------------------------------------------------- #
# 2. The cross-partition choice — PURE (deficits/prices only; NO p_good).
# ------------------------------------------------------------------------- #
def choose_partition(deficits: dict, prices: dict, capped: set, servable: set,
                     rng, explore_floor: float = EXPLORE_FLOOR) -> str | None:
    """Name the partition to serve next. Uses ONLY per-partition deficits and prices — never
    any per-node score, never a p_good — so cross-partition p_good comparison is structurally
    impossible (this is the certified boundary). Exploration floor: with prob `explore_floor`
    draw uniformly among servable partitions that still have positive demand, so none starves
    on a stale price.

    Returns the chosen partition, or None if nothing is servable."""
    cand = [p for p in servable if p not in capped]
    if not cand:
        return None
    demand = [p for p in cand if deficits.get(p, 0.0) > 0.0]
    if demand and float(rng.random()) < explore_floor:
        return demand[int(rng.integers(len(demand)))]
    # price-weighted deficit = deficit per unit expected cost.
    def pwd(p):
        price = max(float(prices.get(p, SEED_PRICE_MIN)), 1e-6)
        return deficits.get(p, 0.0) / price
    return max(cand, key=lambda p: (pwd(p), p))   # p tie-break => deterministic


# ------------------------------------------------------------------------- #
# The scheduler object — ties the order book, tally, and prices together and owns the
# cross-partition routing (incl. julia twins). Within-partition ordering is the caller's.
# ------------------------------------------------------------------------- #
class DeficitScheduler:
    def __init__(self, partitions: list[str], run_dir: Path,
                 prices_path: Path | str | None = None,
                 explore_floor: float = EXPLORE_FLOOR,
                 julia_route_gain: float = JULIA_ROUTE_GAIN):
        self.partitions = list(partitions)
        self.run_dir = Path(run_dir)
        self.explore_floor = float(explore_floor)
        self.julia_route_gain = float(julia_route_gain)

        # order book (per-partition target fraction) — the release-mix ratio table. No
        # `target_path`: an order book that can be pointed at a file is an order book that can
        # disagree with the one emission serves, which is exactly what it did.
        self.observed = load_observed_type_cluster(self.partitions)
        self.target_frac = target_shares(self.partitions)

        # price config (+ scheduling knobs the config may override).
        pcfg = {}
        pp = Path(prices_path) if prices_path else DEFAULT_PRICES_PATH
        if pp.exists():
            pcfg = json.loads(pp.read_text(encoding="utf-8"))
        self.explore_floor = float(pcfg.get("explore_floor", self.explore_floor))
        self.julia_route_gain = float(pcfg.get("julia_route_gain", self.julia_route_gain))
        self.prices = PriceModel(self.partitions, pcfg)

        self.tally = DistinctLookTally(self.run_dir / "distinct_looks.npz")
        # Set by seed_from_library; stamped into the run summary. `None` means seeding was
        # never even attempted — reported as status "never_attempted", not silently omitted.
        self.seed_record: dict | None = None
        # allocation trace (per-batch partition choice + deficits) for the readout.
        self.trace_path = self.run_dir / "scheduler_trace.jsonl"

    # ---- deficit -------------------------------------------------------- #
    def look_frac(self) -> dict:
        tot = self.tally.total()
        if tot <= 0:
            return {p: 0.0 for p in self.partitions}
        return {p: self.tally.count(p) / tot for p in self.partitions}

    def deficits(self) -> dict:
        lf = self.look_frac()
        return {p: self.target_frac.get(p, 0.0) - lf.get(p, 0.0) for p in self.partitions}

    def effective_deficits(self, queue_lens: dict) -> dict:
        """6. Julia routing. A julia:X partition whose OWN queue is empty cannot be popped, so
        its (positive) deficit is folded into its c-plane parent X's effective deficit
        (weighted by julia_route_gain): serving c-plane X fires the julia hook, seeding julia:X
        roots that later become directly poppable. A julia twin that HAS a queue competes on
        its own and is not double-counted. c-plane / already-servable julia deficits pass
        through unchanged. Deficit arithmetic only — no p_good anywhere."""
        base = self.deficits()
        eff = dict(base)
        for jp in self.partitions:
            if not is_julia(jp):
                continue
            if queue_lens.get(jp, 0) > 0:            # directly servable -> competes on its own
                continue
            cp = cplane_of(jp)
            if cp in eff and base[jp] > 0.0:
                eff[cp] = eff[cp] + self.julia_route_gain * base[jp]
        return eff

    # ---- the pop decision ---------------------------------------------- #
    def pick_partition(self, queue_lens: dict, rng) -> str | None:
        """Choose which partition's sub-queue to pop this batch. `queue_lens` maps partition
        -> number of frontier nodes currently in that partition. Servable = non-empty queue.
        Returns the partition name (a c-plane family may be chosen to buy julia twin looks),
        or None if every queue is empty."""
        servable = {p for p, n in queue_lens.items() if n > 0}
        eff = self.effective_deficits(queue_lens)
        return choose_partition(eff, self.prices.price, self.prices.capped, servable, rng,
                                self.explore_floor)

    # ---- 7. deficit-aware root allocation ------------------------------- #
    def root_allocation(self, families: list[str], n_draws: int, rng) -> dict:
        """Split `n_draws` root draws across c-plane `families` proportional to a softmax of
        their price-weighted, twin-inclusive effective deficit (item 7). A family carrying
        julia twin deficit thus draws more roots on its twin's behalf. Returns {family: count}
        summing to `n_draws` (empty queues => all families equally eligible)."""
        # roots are c-plane; fold in each family's julia twin demand (empty-queue routing).
        base = self.deficits()
        scores = {}
        for f in families:
            s = base.get(f, 0.0)
            jp = julia_partition(f)
            if jp in base and base[jp] > 0.0:
                s += self.julia_route_gain * base[jp]
            price = max(self.prices.price.get(f, SEED_PRICE_MIN), 1e-6)
            scores[f] = s / price
        vals = np.array([scores[f] for f in families], dtype=np.float64)
        # range-normalized softmax (scale-free), temperature 0.5; ties -> uniform.
        span = float(vals.max() - vals.min())
        if span <= 1e-12:
            probs = np.full(len(families), 1.0 / len(families))
        else:
            norm = (vals - vals.min()) / span
            ex = np.exp(norm / 0.5)
            probs = ex / ex.sum()
        draws = rng.multinomial(int(n_draws), probs)
        return {f: int(n) for f, n in zip(families, draws)}

    # ---- admission hook ------------------------------------------------- #
    def on_admission(self, partition: str, emb) -> bool:
        """Register an admitted look. Embeds already done by the caller (library morph recipe).
        Returns True iff it was a NEW distinct look (tally + price EMA updated)."""
        distinct = self.tally.add(partition, emb)
        if distinct:
            self.prices.record_look(partition)
        return distinct

    def charge(self, partition: str, minutes: float) -> bool:
        """Account a batch's active time to the served partition (attempt-cap accounting)."""
        return self.prices.charge(partition, minutes)

    def seed_from_library(self, embeddings: dict[str, np.ndarray] | None = None, *,
                          allow_unseeded: bool = False, record: dict | None = None,
                          intake_path: Path | None = None,
                          emb_dir: Path | None = None) -> dict:
        """One-time baseline seed of the distinct-look tally from the library's existing looks
        (campaign-1 intake medoids), so deficits measure LIBRARY-WIDE scarcity rather than
        run-local scarcity, and the seeded embeddings become dedup memory (a new admission that
        duplicates a known library look does not count as a new look).

        FAIL-CLOSED: with no usable seed this RAISES UnseededRunError unless `allow_unseeded`.
        Either way it sets `self.seed_record` — the durable stamp for the run summary — so an
        overridden run is permanently marked and can never be read back as a seeded one.

        `embeddings` injects the matrices directly (tests / a preloaded preflight); `record`
        passes a `require_library_seed` result straight through so the CLI preflight's single
        load is reused. Neither bypasses the guard: an EMPTY injected dict fails closed too.

        Resume-safe + idempotent: seeds ONLY when the tally is empty. A resume reloads the
        persisted npz (total > 0) and this is a no-op — the seed is never double-counted, the
        guard does not fire (a resumed tally IS the seed), and after seeding the tally is
        persisted immediately so the very first kill can't lose it. Restricted to this run's
        tracked partitions. Returns {partition: seeded_count}."""
        if self.tally.total() > 0:                    # already populated (resume) — never re-seed
            # A resumed tally already carries whatever the fresh start seeded; re-checking the
            # artifact here would abort legitimate resumes of seeded runs on a since-moved file.
            self.seed_record = dict(status="resume", seeded=False, seeded_looks=0,
                                    per_partition={}, tally_total=self.tally.total(),
                                    note="tally reloaded from distinct_looks.npz; "
                                         "seeding skipped (see the fresh run's summary)")
            return {}

        if record is not None:                        # preflight already loaded + guarded
            rec = dict(record)
            embeddings = rec.pop("embeddings", embeddings) or {}
        elif embeddings is None:
            rec = require_library_seed(allow_unseeded=allow_unseeded,
                                       intake_path=intake_path, emb_dir=emb_dir)
            embeddings = rec.pop("embeddings")
        else:                                         # injected matrices — guard them the same
            n = int(sum(int(np.asarray(m).shape[0]) for m in embeddings.values()))
            rec = dict(status="seeded" if n else "unseeded", source="<injected>",
                       emb_dir=None, source_exists=None, library_looks=n,
                       library_partitions=sorted(embeddings),
                       allow_unseeded=bool(allow_unseeded))
            if not n:
                rec["reason"] = "injected seed embeddings were empty"
                if not allow_unseeded:
                    raise UnseededRunError(
                        "--scheduler run was handed an EMPTY library seed (injected). "
                        "Aborting before any work; pass allow_unseeded=True to proceed "
                        "deliberately (the run summary is then stamped status=unseeded).")

        seeded: dict[str, int] = {}
        for part in self.partitions:                  # only families this run tracks
            mat = embeddings.get(part)
            if mat is None:
                continue
            for e in mat:
                if self.tally.add(part, e):           # dedup at the same 0.974 knee as admissions
                    seeded[part] = seeded.get(part, 0) + 1
        if seeded:
            self.tally.save()                         # persist the baseline before any batch runs

        # The stamp. Seeded or not, the summary now says WHICH artifact it seeded from and HOW
        # MANY looks it seeded — the two facts campaign-2's summary could not answer.
        rec = seed_stamp(rec)
        rec.update(seeded=bool(seeded), seeded_looks=int(sum(seeded.values())),
                   per_partition=dict(seeded), tracked_partitions=list(self.partitions),
                   tally_total=self.tally.total())
        self.seed_record = rec
        return seeded

    def log_choice(self, batch: int, chosen: str | None, queue_lens: dict):
        eff = self.effective_deficits(queue_lens)
        rec = dict(batch=batch, chosen=chosen,
                   deficits={p: round(self.deficits()[p], 5) for p in self.partitions},
                   eff_deficits={p: round(eff[p], 5) for p in self.partitions},
                   prices={p: round(self.prices.price[p], 4) for p in self.partitions},
                   looks=self.tally.counts(), capped=sorted(self.prices.capped),
                   queue_lens={p: int(queue_lens.get(p, 0)) for p in self.partitions})
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- state (embedded in the driver's state.json; embeddings in npz) - #
    def state_dict(self) -> dict:
        return dict(partitions=self.partitions, target_frac=self.target_frac,
                    explore_floor=self.explore_floor, julia_route_gain=self.julia_route_gain,
                    prices=self.prices.state_dict())

    def load_state(self, d: dict, reopen_caps: bool = False):
        self.target_frac = {p: float(v) for p, v in d.get("target_frac", {}).items()} \
            or self.target_frac
        self.explore_floor = float(d.get("explore_floor", self.explore_floor))
        self.julia_route_gain = float(d.get("julia_route_gain", self.julia_route_gain))
        self.prices.load_state(d.get("prices", {}))
        if reopen_caps:
            self.prices.reopen_caps()
        # the distinct-look tally reloads from its own npz in __init__.

    def save(self):
        self.tally.save()

    def summary(self) -> dict:
        return dict(library_seed=(self.seed_record if self.seed_record is not None else
                                  dict(status="never_attempted", seeded=False, seeded_looks=0,
                                       reason="seed_from_library was never called; deficits "
                                              "measure RUN-LOCAL scarcity")),
                    target_frac={p: round(v, 4) for p, v in self.target_frac.items()},
                    look_frac={p: round(v, 4) for p, v in self.look_frac().items()},
                    looks=self.tally.counts(), total_looks=self.tally.total(),
                    prices={p: round(v, 3) for p, v in self.prices.price.items()},
                    min_spent={p: round(v, 2) for p, v in self.prices.min_spent.items()},
                    capped=sorted(self.prices.capped),
                    n_observed_cells=len(self.observed))
