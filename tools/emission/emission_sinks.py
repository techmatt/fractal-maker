"""Central sink-isolation for the emission record stores — one place that decides WHERE
stage 2's durable outputs land, and refuses to let a throwaway run reach the production
records.

WHY THIS EXISTS. Everything the emission driver writes under `--out` is already a `scratch/`
path, so a smoke run looked isolated. Two sinks are not under `--out` and never were:

  * `data/emission/release_records/<site>.jsonl` (+ `__runs.jsonl`) — the gate/release
    decision record and its population, written through `paths.durable()`;
  * `data/emission/mining_gate_reports/<site>.jsonl` — the mining-gate verdict log (the gate
    it records went from report-only to enforcing on 2026-08-06; the log accrues either way).

Both UPSERT BY KEY and accumulate across runs, and the key is prefixed with the run id — so a
smoke run does not corrupt an existing row, it ADDS rows. That is worse than it sounds: these
files exist so a later calibration pass can read labeled gate precision off accumulated
releases, and a bounded 60-row smoke's decisions are indistinguishable in that file from a
real release's. The record's own docstring says "a row invented ... would look exactly like a
measurement and be worth less than the absent one it replaced".

This is `tools/atlas/discovery_sinks.py`'s pattern, applied to the other durable store. The
throwaway run resolves an ephemeral record root under `scratch/` and PHYSICALLY cannot name a
path under `data/`; `resolves_under_data()` is the assertion a caller runs BEFORE its first
write, and `assert_isolated()` is that assertion with the error message attached.

Deliberately dependency-free (pathlib only): `tools/emission/release_record.py` and
`tools/mining/gate_report.py` both import it, and it must stay importable in the light pytest
lane (no numpy/torch/GPU).

  import emission_sinks as esinks
  esinks.use(esinks.resolve_record_root(ROOT, smoke=True, explicit=None, run_id="smoke_1"))
  esinks.assert_isolated(ROOT)              # before the first write
"""
from __future__ import annotations

from pathlib import Path

# The sites and file shapes derived from a record root — the COMPLETE set stage 2 may write.
# Keep in lockstep with `release_record.RECORD_DIR_REL` / `gate_report`'s log dir; a sink
# missing from here is a sink the isolation assertion is blind to, which is exactly how
# `discovery_sinks`' `gather` entry was once missed.
RELEASE_RECORDS = "release_records"
MINING_GATE_REPORTS = "mining_gate_reports"


def default_record_root(root: Path) -> Path:
    """The production durable emission store (committed via .gitignore negation)."""
    return Path(root) / "data" / "emission"


def data_root(root: Path) -> Path:
    """The never-delete `data/` tier. A throwaway run resolving a sink under here is the
    exact bug this module prevents."""
    return (Path(root) / "data").resolve()


def is_throwaway(smoke: bool) -> bool:
    """A throwaway run must never touch the durable store. Single predicate, so every future
    dry/mini/smoke flag shares one definition instead of each entry point deciding again."""
    return bool(smoke)


def smoke_scratch_root(root: Path, run_id: str) -> Path:
    """Ephemeral record root for a throwaway run: under the disposable `scratch/` tree, NEVER
    `data/`. Keyed by run id (unlike `discovery_sinks`' fixed smoke dir) because these files
    accumulate BY RUN — two smokes sharing a root would upsert into each other's record and
    the second one's readout would be over a population it did not produce."""
    return Path(root) / "scratch" / "emission" / "_ephemeral_records" / str(run_id)


def resolve_record_root(root: Path, *, smoke: bool, explicit=None, run_id: str = "run") -> Path:
    """Central sink-isolation decision. Precedence:
      1. an explicit `--record-root` (tests, a named ephemeral run dir) wins verbatim;
      2. a throwaway run with no explicit root is redirected under `scratch/`;
      3. otherwise the production default `data/emission`.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    if is_throwaway(smoke):
        return smoke_scratch_root(root, run_id).resolve()
    return default_record_root(root).resolve()


def derive_sinks(record_root: Path, site: str) -> dict:
    """Every durable file a run at `site` may write under `record_root`."""
    d = Path(record_root)
    return {
        "decisions": d / RELEASE_RECORDS / f"{site}.jsonl",
        "runs": d / RELEASE_RECORDS / f"{site}__runs.jsonl",
        "gate_report": d / MINING_GATE_REPORTS / f"{site}.jsonl",
    }


def resolves_under_data(root: Path, record_root: Path, site: str) -> list:
    """The sinks a run with this record root would write UNDER `data/` (empty == isolated).
    Returns paths, so a caller can print exactly what it refused."""
    droot = data_root(root)
    offenders = []
    for sink in [Path(record_root), *derive_sinks(record_root, site).values()]:
        try:
            Path(sink).resolve().relative_to(droot)
        except ValueError:
            continue
        offenders.append(Path(sink))
    return offenders


class SinkNotIsolated(RuntimeError):
    """A run declared ephemeral resolved a sink under `data/`. Raised BEFORE the first write —
    the whole point is that the check is cheap and the mistake is not."""


def assert_isolated(root: Path, record_root: Path, site: str) -> list:
    """Raise unless every derived sink is outside `data/`. Returns the sink list on success so
    the caller can log what it is actually writing — an isolation claim nobody can read is the
    same shape as no isolation."""
    offenders = resolves_under_data(root, record_root, site)
    if offenders:
        raise SinkNotIsolated(
            f"emission run declared EPHEMERAL but {len(offenders)} sink(s) resolve under "
            f"data/: {[str(p) for p in offenders]}. These stores upsert-and-accumulate by run "
            f"id, so a throwaway run does not overwrite a row — it adds rows that a later "
            f"calibration pass cannot tell from a real release's. Pass --record-root at a "
            f"scratch path, or drop --ephemeral to write the production store deliberately.")
    return sorted(derive_sinks(record_root, site).values())


# --------------------------------------------------------------------------- #
# The active binding. `release_record` and `gate_report` resolve their paths through
# `record_root()`, so the redirect is made ONCE by the driver rather than at each of the
# three write sites (which is how a redirect misses one).
# --------------------------------------------------------------------------- #
_OVERRIDE: Path | None = None


def use(record_root=None) -> None:
    """Bind the process's emission record root. `None` restores production (`data/emission`,
    written through `paths.durable()`); any other value is an EPHEMERAL root written as plain
    scratch — `durable()` would correctly refuse a gitignored path, and refusing is not what a
    deliberately-disposable record wants."""
    global _OVERRIDE
    _OVERRIDE = None if record_root is None else Path(record_root).resolve()


def is_production() -> bool:
    """True iff no override is bound — i.e. writes go to the durable store and must pass the
    `paths.durable()` assertion."""
    return _OVERRIDE is None


def record_root(root: Path) -> Path:
    return default_record_root(root) if _OVERRIDE is None else _OVERRIDE
