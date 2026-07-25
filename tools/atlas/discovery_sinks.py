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

Deliberately dependency-free (pathlib only): both the seeder and its guard test import
this, and the guard test must stay in the light `pytest` lane (no numpy/torch/GPU).
"""
from __future__ import annotations

from pathlib import Path


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


def derive_sinks(discovery_dir: Path) -> dict[str, Path]:
    """Every durable sink derived from a discovery-store root — the COMPLETE set a run
    may write. Keep in lockstep with production_seeder's module-level globals; the
    `gather` entry is the one the old `--discovery-dir` rebind silently missed (it was
    frozen at module-load time against the production `data/discovery`)."""
    d = Path(discovery_dir)
    return {
        "outcome_ledger": d / "outcome_ledger.jsonl",
        "outcome_feats": d / "outcome_feats.npz",
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
