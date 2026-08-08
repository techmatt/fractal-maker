#!/usr/bin/env python
r"""Shared plumbing for the Newton divergence-abort parity fixture.

This module holds NO expectation logic — no solver, no reference implementation, no
thresholds. It owns exactly three things that the fixture GENERATOR
(`build_newton_parity_ref.py`) and the fixture CONSUMER
(`test_newton_divergence_abort.py`) must agree on byte-for-byte, and that would rot
into two divergent copies if each declared its own:

  1. `grid_jobs` — THE enumeration order of a (n_ang, n_rad, degrees) grid. The fixture
     is a positional array; a generator and a test that disagree on the order compare
     different solves and still pass.
  2. `result_digest` — THE canonical serialization of the fields `_same` compares.
  3. `load` / `dump` — the fixture file format and its metadata block.

WHY A FIXTURE AT ALL. The reference arm in the test module is a reimplementation of
`newton_nucleus` as it stood BEFORE the divergence abort — code that no longer exists and
therefore can never change. For a fixed (seed, period, degree, max_steps, dps) its result
is a mathematical constant. Re-deriving 31,616 of those constants on every run cost ~80%
of the whole `slow` lane (measured 2026-08-08: reference-arm-on-non-convergers was 80.6%
of the test, the live arm 13.1%). Pinning them is the same move `data/atlas/guard_tripwire.json`
makes for the 81 guard verdicts, and it preserves the evidence exactly: `lost == 0` is
still checked against the reference's real verdict, because that verdict is what is pinned.

The DEFAULT-lane grids keep running both arms live and are not pinned — they are cheap,
and they are what keeps the pinned table honest about the reference still being
reproducible at all.

BACKEND PORTABILITY. `mpf._mpf_` is `(sign, man, exp, bc)` where `man` is a plain `int`
under mpmath's pure-Python backend and a `gmpy2.mpz` under the gmpy backend, so `repr` of
the raw tuple differs between the two while the VALUE does not. `result_digest` normalizes
every component through `int()`, which is what makes a fixture generated under gmpy2 valid
for a checkout without it. Verified 2026-08-08: 208/208 solves bit-identical across the two
backends on (converged, iters, c, residual).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mpmath as mp

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "tools", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths  # noqa: E402  storage-class declaration at the write site

FIXTURE_REL = "data/sourcing/newton_parity_ref.json"


def fixture_path(*, mkparents: bool = False) -> Path:
    """The pinned reference table. Durable: it is an input to a test, not an output."""
    return paths.durable(FIXTURE_REL, mkparents=mkparents)


def grid_jobs(n_ang, n_rad, degrees):
    """THE enumeration order of a grid, as (degree, seed_idx, seed_re, seed_im).

    One job per (degree, ring seed); the caller walks `brs.PERIODS` inside it, so a job is
    also the natural parallel unit (13 solves, enough to amortize IPC). The order is
    `for degree: for seed_idx:` and MUST NOT be reordered — the fixture is positional.
    """
    import build_minibrot_roster as brs
    out = []
    for deg in degrees:
        for idx, (sr, si) in enumerate(brs.ring_seeds(deg, n_ang, n_rad)):
            out.append((deg, idx, sr, si))
    return out


def _mpf_key(x):
    """Exact, backend-independent serialization of an mpf (see module docstring)."""
    s, m, e, b = mp.mpf(x)._mpf_
    return (int(s), int(m), int(e), int(b))


def result_digest(res) -> str:
    """16 hex chars over EXACTLY the fields `_same` compares: converged, iters, c, residual.

    Truncated to 16 chars (64 bits) deliberately: the table is ~32k rows, so the collision
    probability is ~2.7e-11 — far below the rate at which anything else here is wrong — and
    it keeps the fixture at ~1 MB and diffable.
    """
    payload = json.dumps({
        "converged": bool(res.converged),
        "iters": int(res.iters),
        "c_re": _mpf_key(res.c.real),
        "c_im": _mpf_key(res.c.imag),
        "residual": repr(float(res.residual)),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def row_of(res) -> list:
    """One fixture row. `converged`/`iters` stay in the clear so a failure message can name
    what differs without the reader having to regenerate the table to find out."""
    return [1 if res.converged else 0, int(res.iters), result_digest(res)]


def grid_key(n_ang, n_rad, max_steps) -> str:
    return f"{n_ang}x{n_rad}@{max_steps}"


def load(path: Path | None = None) -> dict:
    p = fixture_path() if path is None else path
    if not p.exists():
        raise AssertionError(
            f"{p} is missing — it is a tracked durable artifact, not an optional input; "
            f"rebuild with `uv run python tools/sourcing/build_newton_parity_ref.py`")
    return json.loads(p.read_text(encoding="utf-8"))


def dump(doc: dict, path: Path | None = None) -> Path:
    p = fixture_path(mkparents=True) if path is None else path
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p
