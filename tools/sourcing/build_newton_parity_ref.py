#!/usr/bin/env python
r"""Build the pinned reference table for the Newton divergence-abort parity test.

Runs the PRE-ABORT reference solver (`test_newton_divergence_abort.ref_newton_nucleus` —
a reimplementation of `newton_nucleus` as it stood before the divergence abort) over the
two committed slow-lane grids and pins its outcome per solve. See
`newton_parity.py`'s docstring for why this is a fixture rather than a live arm.

    uv run python tools/sourcing/build_newton_parity_ref.py                 # both grids
    uv run python tools/sourcing/build_newton_parity_ref.py --workers 4
    uv run python tools/sourcing/build_newton_parity_ref.py --limit 8       # bounded smoke

`--limit N` runs the WHOLE path on the first N jobs per grid and stamps
`"incomplete": true` into the written document, derived from the flag at the write site —
CLAUDE.md's bounded-end-to-end rule. `newton_parity.load` consumers must refuse an
incomplete table; `test_newton_divergence_abort` asserts on the flag.

WHY THIS IMPORTS A TEST MODULE. The reference implementation deliberately lives inside the
test file (the module docstring there explains: the expectation must not be computed by the
code under test, and a flag on the live function would still be that). A second copy here
would be the thing most likely to silently drift from the one the default-lane tests use, so
the generator imports the single copy instead. The import is by module name off
`tools/sourcing`, which is how the test file is already importable standalone.

WORKERS. These are ~50 MB pure-mpmath processes, one core each — NOT the heavyweight
`fractal-generator.exe` that CLAUDE.md's 4-process cap is written against (own rayon pool,
resident LUTs, corpus scan). 8 is the default here on the 12-core box; see
`docs/design/pytest_suite_cost.md`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import sys
import time
from pathlib import Path

import mpmath as mp
import mpmath.libmp as libmp

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "tools", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_minibrot_roster as brs   # noqa: E402
import newton_parity as npar          # noqa: E402

# The two grids the `slow` lane pins: the committed roster density at the production budget,
# and the same shape at the library-default budget. Kept in lockstep with the parametrize
# list in `test_newton_divergence_abort.test_full_roster_ring_seed_grid_is_parity_clean`.
GRIDS = [
    {"n_ang": 64, "n_rad": 8, "max_steps": brs.NEWTON_STEPS, "expect_n": 26624},
    {"n_ang": 24, "n_rad": 4, "max_steps": 200, "expect_n": 4992},
]
DEFAULT_WORKERS = 8

_REF = None


def _init():
    """Spawn-safe worker setup: the autouse `_dps` fixture does not reach a subprocess."""
    global _REF
    for p in (str(_ROOT), str(_ROOT / "tools"), str(_HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import mpmath as _mp
    import build_minibrot_roster as _brs
    _mp.mp.dps = _brs.NUCLEUS_DPS
    _REF = importlib.import_module("test_newton_divergence_abort")


def _job(args):
    """One (degree, seed) column -> one fixture row per period, in PERIODS order."""
    deg, _idx, sr, si, max_steps = args
    import mpmath as _mp
    import build_minibrot_roster as _brs
    seed = _mp.mpc(sr, si)
    return [npar.row_of(_REF.ref_newton_nucleus(seed, p, degree=deg, max_steps=max_steps))
            for p in _brs.PERIODS]


def build_grid(spec, workers, limit=None, log=print):
    jobs = npar.grid_jobs(spec["n_ang"], spec["n_rad"], brs.DEGREES)
    if limit is not None:
        jobs = jobs[:limit]
    payload = [(deg, idx, sr, si, spec["max_steps"]) for deg, idx, sr, si in jobs]

    t0 = time.time()
    rows, done = [], 0
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        for out in ex.map(_job, payload, chunksize=1):
            rows.extend(out)
            done += 1
            if done % 64 == 0 or done == len(payload):
                el = time.time() - t0
                rate = done / el if el else 0
                log(f"  {done}/{len(payload)} cols  {el:6.1f}s  "
                    f"eta {(len(payload)-done)/rate if rate else 0:6.1f}s", flush=True)

    n_conv = sum(r[0] for r in rows)
    log(f"  grid {npar.grid_key(spec['n_ang'], spec['n_rad'], spec['max_steps'])}: "
        f"{len(rows)} solves, {n_conv} converged, {len(rows)-n_conv} not "
        f"({time.time()-t0:.1f}s)", flush=True)
    return rows, n_conv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="bounded end-to-end: first N (degree, seed) columns per grid; "
                         "stamps the output incomplete")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    mp.mp.dps = brs.NUCLEUS_DPS
    print(f"backend={libmp.BACKEND} dps={mp.mp.dps} workers={args.workers} "
          f"limit={args.limit}", flush=True)

    grids = {}
    for spec in GRIDS:
        key = npar.grid_key(spec["n_ang"], spec["n_rad"], spec["max_steps"])
        print(f"grid {key} ...", flush=True)
        rows, n_conv = build_grid(spec, args.workers, args.limit)
        grids[key] = {
            "n_ang": spec["n_ang"], "n_rad": spec["n_rad"],
            "max_steps": spec["max_steps"], "degrees": brs.DEGREES,
            "periods": brs.PERIODS, "expect_n": spec["expect_n"],
            "n_converged": n_conv, "n_not_converged": len(rows) - n_conv,
            "rows": rows,
        }

    doc = {
        "what": "pre-abort reference solver outcomes, pinned per solve; see "
                "tools/sourcing/newton_parity.py",
        "generated_by": "uv run python tools/sourcing/build_newton_parity_ref.py"
                        + (f" --limit {args.limit}" if args.limit is not None else ""),
        # derived from the flag at the write site, never hardcoded (CLAUDE.md)
        "incomplete": args.limit is not None,
        "env": {"mpmath": mp.__version__, "backend": libmp.BACKEND,
                "dps": brs.NUCLEUS_DPS, "newton_steps": brs.NEWTON_STEPS},
        "row_format": ["converged(0|1)", "iters", "digest16"],
        "grids": grids,
    }
    p = npar.dump(doc, args.out)
    print(f"wrote {p} ({p.stat().st_size/1e6:.2f} MB)"
          + ("  [INCOMPLETE]" if doc["incomplete"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
