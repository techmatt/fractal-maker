#!/usr/bin/env python
r"""harvest_log_registry.py — THE registry that decides which `harvest_log.jsonl` files a
tau_h derivation is allowed to read, and the discovery that finds them.

WHY THIS EXISTS. `tau_h_rederive.HARVEST_RUNS` was a five-entry hand list of run names.
Every run written after it was typed (nine of them by 2026-08-05, ~31k re-scoreable checks)
was invisible to the derivation — not excluded, not reported, simply never looked at. A
hand list does not degrade loudly: it goes stale in exactly the direction that makes the
curve look settled. So the list is replaced by DISCOVERY over registered stores.

THE REGISTRY IS TWO THINGS, and the difference is the whole design:

  * `HARVEST_STORES` — the ordered registry of *stores* a run may be discovered in. A run
    that writes its log under a registered store is IN, with no code edit. This is the
    mechanism that makes "future runs feed the curves" true, and it is true because
    `discovery_sinks.resolve_discovery_dir` sends every non-throwaway run's sinks to
    `data/discovery` — the registered store — while a `--smoke`/`--time-only` run is
    redirected to `scratch/`, which this module REFUSES by class. Both halves are pinned by
    `test_harvest_log_registry.py::test_a_production_run_lands_in_a_registered_store`.

  * `PINNED_RUNS` — the ordered registry of run dirs that must ALWAYS resolve. These are
    the population the adopted tau_h base was derived on; if one of them stops resolving,
    the derivation's population silently shrinks and the number moves for a reason nobody
    chose. A pinned entry with no log is a HARD failure naming the path, never a skip
    (`verification_practice.md` §2: an absence-tolerant guard un-guards exactly when its
    subject is removed).

RESOLVERS REFUSE SCRATCH. A store or pinned run under `scratch/`/`scratchpad/` raises at
RESOLVE time, not at read time — the same rule, from the same owner
(`paths.disposable_component`), as the seed-source registry. A disposable path that has not
been wiped YET reads as perfectly healthy and derives a perfectly healthy-looking number.

WHAT DISCOVERY DOES **NOT** DO. It does not filter on era, admissions or partition — a row
carries its own geometry and the derivation re-renders and re-scores it under the active
head, so a log's own recorded scores never enter. The one row-level exclusion is geometric:
a check written before `cx`/`cy`/`fw` entered the harvest row is not re-renderable and is
counted, not guessed at (that is what excludes campaign1 entirely — 37,853 of its checks
predate the geometry field, so it contributes zero rows *by construction*, and the
derivation must not be told otherwise by a comment). That count lives with the reader
(`tau_h_rederive._harvest_rows`), which is the only place that can report it.

    from harvest_log_registry import require_harvest_runs
    for run in require_harvest_runs():
        ...  # run.name, run.log
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "atlas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import discovery_sinks                                  # noqa: E402
from tools import paths as _paths                       # noqa: E402
from tools.corpus import artifacts as _artifacts        # noqa: E402

LOG_NAME = "harvest_log.jsonl"

# How deep below a store a run dir may sit. 2 because the campaign runs are `<campaign>/
# <leg>` (campaign1/breadth) while the later ones are flat (`popquota_v2_20260804`). Bounded
# rather than a full walk: `data/discovery` also holds per-run subtrees (gather/, runs/)
# that must never be traversed looking for something that is not there.
MAX_RUN_DEPTH = 2

# The ordered registry of harvest STORES. `discovery_sinks.default_discovery_dir` is the
# production sink root, taken from that module rather than restated, so a store cannot drift
# away from where runs actually write.
HARVEST_STORES = (
    ("discovery", discovery_sinks.default_discovery_dir(ROOT)),
)

# The ordered registry of PINNED run dirs (store-relative). These five are the population
# `data/atlas/tau_h_base_v10.json` was derived on. Absence is a hard failure — see the
# module docstring. Adding to this list is NOT how a new run enters a derivation; a new run
# enters by writing its log under a registered store. This list only records which runs may
# never quietly leave.
PINNED_RUNS = (
    "campaign1/breadth",
    "campaign1/dive",
    "campaign2/breadth",
    "campaign2/dive",
    "julia_parent_probe/breadth",
)


class HarvestPathClassError(RuntimeError):
    """A registered harvest store or run dir resolved under a disposable tree.

    Deliberately not recoverable by a flag. A tau_h derivation reading a `scratch/` log
    produces a number whose population is guaranteed to be deleted, and the number outlives
    the population by being pasted into `steered_frontier.TAU_H_FIDELITY_BASE`."""


class MissingHarvestLogError(RuntimeError):
    """A PINNED run dir has no `harvest_log.jsonl`.

    The alternative — dropping it and carrying on — is a derivation over a smaller
    population that reports itself as a derivation over the full one. The pinned set is
    exactly the set for which that is not allowed to happen quietly."""


@dataclass(frozen=True)
class HarvestRun:
    """A discovered run: `name` is the store-relative posix path used as the row tag and
    as the `key` prefix in the rederive cache, so it must stay stable for pinned runs."""
    name: str
    store: str
    dir: Path
    log: Path
    pinned: bool


def _refuse_disposable(kind: str, p) -> Path:
    """Return `p`, or raise `HarvestPathClassError` if it names a disposable-class dir."""
    path = Path(p)
    hit = _paths.disposable_component(path, (ROOT, _artifacts.artifacts_root()))
    if hit is None:
        return path
    raise HarvestPathClassError(
        f"harvest {kind} resolves under the disposable `{hit}/` class, which GUARANTEES "
        f"deletion:\n"
        f"    path : {path}\n"
        f"A tau_h arm derived from logs a `rm -r scratch/*` can delete is an arm whose "
        f"population cannot be re-read, while the number it produced lives on vendored in "
        f"`steered_frontier.TAU_H_FIDELITY_BASE`. Register a store under `data/` "
        f"(`discovery_sinks.default_discovery_dir`) instead."
    )


def stores() -> list[tuple[str, Path]]:
    """The registered stores, class-checked. Checked over the WHOLE table rather than the
    first that exists — a dormant disposable entry that resolves to nothing today is
    exactly the shape that ships green and fires after the next run creates it."""
    return [(name, _refuse_disposable(f"store ({name})", p)) for name, p in HARVEST_STORES]


def discover_run_dirs(*, registry=None, pinned=None) -> tuple[list[HarvestRun], list[str]]:
    """`(runs, missing_pinned)` — every run dir under a registered store that holds a log.

    Ordered: pinned entries first in registry order (so the row stream over the pinned
    population is byte-identical to what the old hand list produced), then everything else
    discovered, sorted by name. `missing_pinned` holds the pinned names that resolved to no
    log; `require_harvest_runs` is what turns that into a failure — this function stays
    total so a caller can REPORT the absence, which is the loud half."""
    reg = list(stores()) if registry is None else [
        (n, _refuse_disposable(f"store ({n})", p)) for n, p in registry]
    pins = list(PINNED_RUNS if pinned is None else pinned)

    found: dict[str, HarvestRun] = {}
    for store_name, store in reg:
        if not store.exists():
            continue
        for depth in range(1, MAX_RUN_DEPTH + 1):
            for log in sorted(store.glob("/".join(["*"] * depth + [LOG_NAME]))):
                run_dir = log.parent
                name = run_dir.relative_to(store).as_posix()
                if name in found:
                    continue
                found[name] = HarvestRun(
                    name=name, store=store_name, dir=_refuse_disposable(f"run ({name})", run_dir),
                    log=log, pinned=name in pins)

    missing = [p for p in pins if p not in found]
    ordered = [found[p] for p in pins if p in found]
    ordered += [r for n, r in sorted(found.items()) if n not in set(pins)]
    return ordered, missing


def require_harvest_runs(*, registry=None, pinned=None) -> list[HarvestRun]:
    """`discover_run_dirs`, with a missing PINNED run raised rather than dropped."""
    runs, missing = discover_run_dirs(registry=registry, pinned=pinned)
    if missing:
        reg = list(stores()) if registry is None else registry
        looked = "\n".join(f"      {n}: {p}" for n, p in reg)
        raise MissingHarvestLogError(
            f"{len(missing)} PINNED harvest run(s) have no {LOG_NAME} under any registered "
            f"store — the derivation population would silently shrink:\n"
            + "".join(f"    {m}\n" for m in missing)
            + f"    stores searched (depth <= {MAX_RUN_DEPTH}):\n{looked}\n"
            f"These runs are the population `data/atlas/tau_h_base_v10.json` was derived "
            f"on. Restore the logs (they are LFS-tracked in-tree beside each run's "
            f"admission ledger) — do NOT drop the entry from PINNED_RUNS to go green."
        )
    return runs


def registry_record(runs: list[HarvestRun]) -> dict:
    """The stamp a derivation artifact carries, so a reader months later can tell WHICH
    population produced the number without re-running discovery. Records the registry as
    well as the result: "five runs" and "five runs out of a registry that offered five" are
    different facts."""
    return dict(
        stores=[dict(name=n, path=str(p), exists=p.exists()) for n, p in stores()],
        pinned=list(PINNED_RUNS),
        max_run_depth=MAX_RUN_DEPTH,
        discovered=[dict(name=r.name, store=r.store, log=str(r.log), pinned=r.pinned)
                    for r in runs],
        n_pinned=sum(1 for r in runs if r.pinned),
        n_unpinned=sum(1 for r in runs if not r.pinned),
    )


if __name__ == "__main__":
    _runs, _missing = discover_run_dirs()
    for r in _runs:
        print(f"{'PIN' if r.pinned else '   '} {r.name:40s} {r.log}")
    print(f"\n{len(_runs)} run(s): {sum(1 for r in _runs if r.pinned)} pinned, "
          f"{sum(1 for r in _runs if not r.pinned)} discovered")
    if _missing:
        print(f"MISSING PINNED: {_missing}")
