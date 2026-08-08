"""Central sink-isolation for the discovery store — one place that decides WHERE
`production_seeder` writes its durable outputs, and refuses to let a throwaway run
reach the production ledgers.

Why this exists: `production_seeder --smoke` standalone used to append to the durable
`data/discovery/` sinks (outcome_ledger.jsonl, outcome_feats.npz, probe_rejects.jsonl,
gather/, runs/). Smoke rows are current-decode, so the version firewall reads them as
admissible — the isolation was structurally blind to the smoke class. The orchestrators'
`--mini` mode already dodges this by passing an explicit `--discovery-dir` at an
ephemeral scratch path; a bare `--smoke` had no such redirect. This module lifts that
decision into one shared, testable place so a throwaway run PHYSICALLY cannot resolve a
sink under `data/`.

Deliberately light: stdlib plus `tools/paths.py` (itself stdlib-only). Both the seeder and
its guard test import this, and the guard test must stay in the light `pytest` lane — no
numpy, no torch, no GPU. `paths` is here because ONE of the five sinks is `bulk()` and the
class has to be declared at the write site, not guessed by the caller (see `feats_path`).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))
import paths as _paths  # noqa: E402  (storage-class helper: bulk() -> out-of-tree)


def default_discovery_dir(root: Path) -> Path:
    """The production durable discovery store (committed via .gitignore negation)."""
    return root / "data" / "discovery"


def data_root(root: Path) -> Path:
    """The never-delete `data/` tier. Any sink resolving under here is durable; a
    throwaway run resolving under here is the exact bug this module prevents."""
    return (root / "data").resolve()


def is_throwaway(smoke: bool, time_only: bool) -> bool:
    """A throwaway run — `--smoke` or `--time-only` (which runs one smoke batch) — must
    never touch the durable store. This is the single predicate the redirect keys on;
    add future dry/mini flags here so every entry point shares one definition."""
    return bool(smoke or time_only)


def smoke_scratch_dir(root: Path) -> Path:
    """Ephemeral discovery store for throwaway runs: under the disposable `scratch/` tree,
    NEVER `data/`. A fixed (timestamp-free) path is fine — it is rm-safe scratch and a
    smoke run's accumulating cloud is meaningless across runs."""
    return root / "scratch" / "atlas" / "production_seeder" / "_smoke_discovery"


def resolve_discovery_dir(root: Path, *, smoke: bool, time_only: bool,
                          explicit: Path | None) -> Path:
    """Central sink-isolation decision. Precedence:
      1. an explicit `--discovery-dir` (orchestrator `--mini`, tests) wins verbatim;
      2. a throwaway run (`--smoke`/`--time-only`) with no explicit dir is redirected to
         an ephemeral scratch dir so it PHYSICALLY cannot reach `data/discovery`;
      3. otherwise the production default `data/discovery`.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    if is_throwaway(smoke, time_only):
        return smoke_scratch_dir(root).resolve()
    return default_discovery_dir(root).resolve()


FEATS_NAME = "outcome_feats.npz"

# The one command that puts an absent feature store back. Named here rather than at each
# reader so there is a single string to keep true (`_require_feats` below quotes it).
FEATS_RECOMPUTE = (
    "uv run python tools/atlas/recompute_outcome_feats.py --run <run_dir>")


def feats_path(discovery_dir: Path, name: str = FEATS_NAME) -> Path:
    """The outcome-FEATURE store for a run — `bulk()`, so OUT-OF-TREE when the run lives
    under `data/discovery/`.

    Demoted from committed on 2026-08-08: the npz is the derived sidecar of
    `outcome_ledger.jsonl`, recomputable from the ledger's own coordinates, and it was 30%
    of a modern run's committed tree bytes. The ledger stays committed; this does not.

    The routing is conditional on purpose. A run dir may be a scratch smoke store or an
    explicit `--discovery-dir` under `tmp_path`, and `paths.bulk` only relocates paths it
    recognizes as repo-relative members of a declared family
    (`artifacts._is_discovery_feats`). So: convert to repo-relative and route when the run
    is inside the repo, and otherwise leave the path exactly where the caller put it — a
    smoke run's feature store belongs beside its own ledger, not in the artifacts root."""
    d = Path(discovery_dir)
    p = d / name
    try:
        rel = p.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return p                      # outside the repo (tmp_path, an absolute store)
    return _paths.bulk(rel)


def _require_feats(p: Path) -> Path:
    """A feature store a reader cannot proceed without — MISSING IS FATAL, and the raise
    names the rebuild.

    The `_require_field` shape (`tools/studies/q4_stage1_labelset.LS`, `_require_v8`): a
    demoted artifact is absent BY DESIGN, so every reader that would silently do less
    without it has to say so and say what to run. `redecode_grid` is the one that would
    have gone quiet — it wrote a `n_feats: 0` subset under an `if feats_src.exists()`.

    NOT bit-identical on recompute, and the message says so: each banked vector came from
    the head that was active when its run walked (the row's own `scorer_version`), and
    those weights are de-tracked. A recompute is a faithful feature, not that one."""
    if p.exists():
        return p
    raise SystemExit(
        f"outcome feature store missing: {p}\n"
        f"    It is bulk() as of 2026-08-08 — recomputable from the ledger, so absent by\n"
        f"    design rather than lost. Rebuild it:\n"
        f"      {FEATS_RECOMPUTE}\n"
        f"    NOTE: the recompute embeds through the head active TODAY. Each banked vector\n"
        f"    was pulled through the head its ledger row names in `scorer_version`, so a\n"
        f"    rebuilt store is a faithful feature set, not a byte-restore of the old one.")


def derive_sinks(discovery_dir: Path) -> dict[str, Path]:
    """Every durable sink derived from a discovery-store root — the COMPLETE set a run
    may write. Keep in lockstep with production_seeder's module-level globals; the
    `gather` entry is the one the old `--discovery-dir` rebind silently missed (it was
    frozen at module-load time against the production `data/discovery`).

    `outcome_feats` is the one member that is NOT durable — it is `bulk()` and resolves
    out-of-tree (see `feats_path`). It stays in this dict because the dict's job is "the
    complete set a run may write", and the sink-isolation guard below has to see it: a
    smoke run must not reach the production feature store either."""
    d = Path(discovery_dir)
    return {
        "outcome_ledger": d / "outcome_ledger.jsonl",
        "outcome_feats": feats_path(d),
        "probe_rejects": d / "probe_rejects.jsonl",
        "runs": d / "runs",
        "gather": d / "gather",
    }


def resolves_under_data(root: Path, *, smoke: bool, time_only: bool,
                        explicit: Path | None) -> list[Path]:
    """Return the durable sinks a run with these flags would resolve UNDER `data/`
    (empty == isolated). The guard test asserts this is empty for a bare smoke run."""
    store = resolve_discovery_dir(root, smoke=smoke, time_only=time_only, explicit=explicit)
    droot = data_root(root)
    offenders = []
    for sink in [store, *derive_sinks(store).values()]:
        try:
            sink.resolve().relative_to(droot)
        except ValueError:
            continue
        offenders.append(sink)
    return offenders
