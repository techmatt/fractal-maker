"""Guard: a `production_seeder` throwaway run MUST NOT resolve any durable sink under
`data/`.

`production_seeder --smoke` standalone used to append to the durable `data/discovery/`
ledgers — and because smoke rows are current-decode, the version firewall read them as
admissible, so the leak was structurally invisible. The fix redirects a throwaway run
(`--smoke` / `--time-only`) with no explicit `--discovery-dir` to an ephemeral scratch
dir under `out/`. This test locks that in: if the redirect ever regresses, a smoke run's
sinks resolve back under `data/` and the assertion below fires.

Light lane by construction — imports only `tools/atlas/discovery_sinks.py` (pathlib
only), no numpy/torch/GPU/binary — so it runs under bare `pytest` alongside the other
canary tests.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "atlas"))

import discovery_sinks as dsinks  # noqa: E402


def _offenders(*, smoke, time_only, explicit):
    return dsinks.resolves_under_data(
        REPO_ROOT, smoke=smoke, time_only=time_only, explicit=explicit)


def test_bare_smoke_never_resolves_data_sink():
    """A `--smoke` run with no --discovery-dir: every sink must be OUTSIDE data/."""
    offenders = _offenders(smoke=True, time_only=False, explicit=None)
    assert offenders == [], (
        "SMOKE ISOLATION BREACH: a bare `production_seeder --smoke` resolves durable "
        f"sinks under data/: {[str(p) for p in offenders]}. The throwaway redirect in "
        "discovery_sinks.resolve_discovery_dir has regressed — a smoke run can now "
        "append to the production discovery ledgers."
    )


def test_time_only_never_resolves_data_sink():
    """`--time-only` runs a smoke batch internally, so it is throwaway too."""
    offenders = _offenders(smoke=False, time_only=True, explicit=None)
    assert offenders == [], (
        f"TIME-ONLY ISOLATION BREACH: sinks under data/: {[str(p) for p in offenders]}"
    )


def test_smoke_store_is_under_out():
    """Positive assertion: the redirected store lands in the disposable out/ tree."""
    store = dsinks.resolve_discovery_dir(
        REPO_ROOT, smoke=True, time_only=False, explicit=None)
    assert (REPO_ROOT / "out").resolve() in store.parents, (
        f"smoke store {store} is not under out/ (the disposable tier)"
    )


def test_gather_sink_included_in_derivation():
    """The gather sink is the one the old rebind missed; derive_sinks must include it so
    the redirect covers `--gather --smoke`."""
    sinks = dsinks.derive_sinks(Path("/anywhere/store"))
    assert "gather" in sinks


def test_production_run_uses_data_store():
    """Complement: a real (non-throwaway, no --discovery-dir) run DOES target data/, so
    the guard is asserting isolation, not merely that nothing is ever under data/."""
    store = dsinks.resolve_discovery_dir(
        REPO_ROOT, smoke=False, time_only=False, explicit=None)
    assert store == (REPO_ROOT / "data" / "discovery").resolve()
