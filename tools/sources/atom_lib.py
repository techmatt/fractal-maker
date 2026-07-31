#!/usr/bin/env python
"""Shared atom record + descriptors for the minibrot **source sheets**
(`prompts/minibrot_source_sheets.md` + its addendum).

One sheet per generation algorithm, ~150 atoms each at identical framing, so the
sources can be judged against each other. This module is the *common denominator*:
whatever a source does to find a nucleus, it hands the nucleus here and gets back a
record with the identical descriptor set, so the sheets are comparable.

Reuses, never rebuilds: Newton and the sector-canonical dedup key come from
`deep_center_finder`; the feasibility wall from `build_minibrot_roster`; the atom id
from `tools/descent/triage_store.atom_id`, so an atom found by two different sources
(or already in the triage pool) collapses to the SAME id and the overlap matrix falls
out for free.

Two deliberate departures from the triage enumerator, both from the addendum:

* **No `A`-feasibility exclusion.** The known-good reference `mb19_p35` fails the
  roster's own 1-decade cut (deploy margin 0.20) while rendering fine at wall
  fidelity. A cut that would have excluded the canonical good example cannot gate a
  fertility race. `f64_margin_deploy_decades` is *recorded* and the atom is kept;
  a render that actually fails is dropped and logged as an empirical failure.
* **Depth-spanning sampling** (`span_by_depth`) rather than the natural head of the
  distribution, so sources with different depth mixes stay comparable. This is
  deconfounding, not quality selection — and it never stratifies on period.

Primitive vs satellite — NOT AVAILABLE, and why
-----------------------------------------------
The prompt asks for this descriptor per atom and per sheet. **It is not shipped, because
no cheap criterion survived verification.** Two were tried and both were killed by the
counting theorem on a *complete* period-n population (see `src_complete_low_n`), where
the number of satellites is known exactly: `sum_{q|n,q<n} phi(n/q)*nu(q)`.

* **Atom-domain index** (the classical cheap test: with the critical orbit at the
  nucleus, let `k = argmin_{1<=k<p} |z_k|`; call it a satellite of period `k` iff `k`
  divides `p`). Agrees with hand-checked ground truth on 14/14 small cases and matches
  the theorem exactly at n=2,3,4,5 — then **fails at n=6: 17 satellites called where
  theory says 7**, on a complete 27-component population. With 5 orbit values and
  divisors {1,2,3}, the argmin lands on a divisor by coincidence most of the time.
* **Parent distance** (Newton to a period-q nucleus, measure `|c - c_q|` in units of the
  parent's own window scale). Does not separate at all: at n=6 the ranking puts a
  **conjugate pair astride the satellite/primitive boundary**, which is impossible —
  conjugates are the same component reflected.

A correct test needs the component's **root point** (where the cycle multiplier is 1)
and the parent cycle's multiplier there — genuine new machinery, out of budget for this
batch. So the raw quantities are recorded (`atom_domain_index`,
`atom_domain_min_abs_z`, `atom_domain_divides_period`) and **no shape is claimed**: a
later correct classifier can be applied to the durable records without re-enumerating.

The one place an exact satellite fraction IS available is the complete-enumeration
sheet, where the theorem supplies it per period without identifying which atoms they
are. `theorem_satellite_fraction` returns it, and only that sheet prints it.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "tools" / "sourcing", REPO_ROOT / "tools" / "descent",
          REPO_ROOT / "tools" / "corpus"):
    sys.path.insert(0, str(p))

import mpmath as mp                      # noqa: E402
import deep_center_finder as dcf         # noqa: E402
import build_minibrot_roster as brs      # noqa: E402
import triage_store as ts                # noqa: E402  (atom_id + the framing constants)

DEGREE = 2                               # this batch is degree 2 ONLY (deliberate scope cut)
NUCLEUS_DPS = brs.NUCLEUS_DPS            # 60
NEWTON_STEPS = brs.NEWTON_STEPS          # 60
DEDUP_DPS = brs.DEDUP_DPS                # 22
ORIGIN_EPS = brs.ORIGIN_EPS
EMIT_DIGITS = dcf.emit_digits_for_fw(1e-20)
MAX_EMBED_DEPTH = 8                      # recursion bound on the satellite ancestor chain


def set_precision() -> None:
    mp.mp.dps = NUCLEUS_DPS


# Components below this magnitude are numerical noise, not structure, and are snapped
# to exact zero before the dedup key is built. WHY THIS EXISTS: `dcf.nucleus_dedup_key`
# rounds to `DEDUP_DPS` *significant* digits, so a real-axis nucleus whose imaginary
# part is Newton noise (~1e-40, differing per seed path) stringifies to a DIFFERENT key
# on every solve — and the same atom enters a population many times. Measured on a
# 262-atom probe run: 23 redundant rows (9%), all real-axis, with c = -1.3107 appearing
# ten times. The counting theorem on a complete period-4 population is what caught it
# (4 satellites found where theory says 3).
#
# The threshold is safe by an enormous margin in both directions: Newton at dps=60
# leaves noise near 1e-50, while a genuine off-axis nucleus is separated from the axis
# by something of order its own structure scale, never 1e-20. Real-axis nuclei have an
# imaginary part of exactly 0.
#
# This is a LOCAL fix. `dcf.nucleus_dedup_key` is shared with the committed roster and
# the triage pool, whose stored keys must not move, so the snap is applied here on the
# way in rather than inside the shared function. Both of those populations carry the
# same latent duplication — see the report.
SNAP_EPS = mp.mpf("1e-20")


def snap(c):
    """Zero a coordinate component that is below `SNAP_EPS` (noise, not structure)."""
    c = mp.mpc(c)
    re = mp.mpf(0) if abs(c.real) < SNAP_EPS else c.real
    im = mp.mpf(0) if abs(c.imag) < SNAP_EPS else c.imag
    return mp.mpc(re, im)


def _tol():
    return mp.mpf(10) ** (-(mp.mp.dps - 6))


def is_minimal(c, period: int, degree: int = DEGREE) -> bool:
    """Minimality screen — REQUIRED before `classify_shape` (see module docstring)."""
    return brs._is_minimal_nucleus(c, period, degree, _tol())


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #
def atom_domain(c, period: int, degree: int = DEGREE) -> tuple[int, float]:
    """`(argmin_{1<=k<period} |z_k|, that minimum)` along the critical orbit."""
    z = mp.mpc(0)
    best, bk = None, 0
    for k in range(1, period):
        z = (z * z + c) if degree == 2 else (z ** (degree - 1) * z + c)
        a = abs(z)
        if best is None or a < best:
            best, bk = a, k
    return bk, (float(best) if best is not None else float("nan"))


def atom_domain_record(c, period: int, degree: int = DEGREE) -> dict:
    """RAW atom-domain quantities. Deliberately not a primitive/satellite label — see
    the module docstring for the two criteria that were tried and falsified."""
    k, mn = atom_domain(c, period, degree)
    return {
        "atom_domain_index": k,
        "atom_domain_min_abs_z": mn,
        # the classical guess, recorded as a RAW FLAG so a later analysis can use it
        # knowing it over-calls; never surfaced as "shape" anywhere.
        "atom_domain_divides_period": bool(k > 0 and k < period and period % k == 0),
        "shape": None,                     # not available; see module docstring
        "satellite_of_period": None,
    }


def embedding_depth(c, period: int, degree: int = DEGREE) -> None:
    """Not available: embedding depth counts satellite levels, so it inherits the
    primitive/satellite classification that was falsified above. Returns None rather
    than a number that would look authoritative and be wrong."""
    return None


def region_of(c) -> dict:
    """Coarse spatial descriptor. Deliberately raw — polar position plus whether the
    atom is on the real axis — rather than an invented named-region taxonomy."""
    re, im = float(c.real), float(c.imag)
    return {
        "c_abs": round(math.hypot(re, im), 6),
        "c_arg": round(math.atan2(im, re), 6),
        "on_real_axis": abs(im) < 1e-12,
        "half": "upper" if im > 0 else ("lower" if im < 0 else "axis"),
    }


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def make_atom(c, period: int, source: str, *, degree: int = DEGREE,
              newton_res_log10: float | None = None, provenance: dict | None = None,
              want_embedding: bool = True) -> dict | None:
    """Canonicalize a nucleus and build its full record. Returns None only if the atom
    is degenerate (|c| ~ 0, non-minimal, or a non-finite instrument) — **never** for
    feasibility: the `A` margin is recorded and the atom is kept (addendum §1)."""
    c = mp.mpc(c)
    if abs(c) < ORIGIN_EPS:
        return None
    cc = snap(dcf.canonical_nucleus_c(c, degree))   # sector-canonical, then de-noised
    if not is_minimal(cc, period, degree):
        return None
    inst = dcf.atom_instrument(cc, period, degree)
    if not math.isfinite(inst.log10_abs_A) or inst.abs_A <= 0:
        return None
    size = dcf.nucleus_size_estimate(cc, period, degree)
    dedup_key = ",".join(dcf.nucleus_dedup_key(cc, degree, DEDUP_DPS))
    rec = {
        "id": ts.atom_id(degree, dedup_key),      # SAME id function as the triage pool
        "source": source,
        "degree": degree,
        "period": period,
        "family": ts.family_for(degree),
        "cx": mp.nstr(cc.real, EMIT_DIGITS, strip_zeros=False),
        "cy": mp.nstr(cc.imag, EMIT_DIGITS, strip_zeros=False),
        "window_scale": f"{inst.window_scale:.10e}",
        "fw": f"{inst.window_scale * 4.0:.6e}",   # the 4x sheet frame
        "abs_A": inst.abs_A,
        "log10_abs_A": round(inst.log10_abs_A, 6),
        "arg_A": round(inst.arg_A, 6),
        "size": float(abs(size)) if size != 0 else 0.0,
        # RECORDED, never used as an exclusion (addendum §1)
        "f64_margin_deploy_decades": round(
            brs.deploy_wall_log10() - inst.log10_abs_A, 4),
        "f64_margin_field_decades": round(
            brs.deploy_wall_log10(brs.FIELD_W, brs.FIELD_SS) - inst.log10_abs_A, 4),
        "required_dps": inst.required_dps,
        "newton_res_log10": (round(newton_res_log10, 2)
                             if newton_res_log10 is not None else None),
        "dedup_key": dedup_key,
        "provenance": provenance or {},
    }
    rec.update(atom_domain_record(cc, period, degree))
    rec.update(region_of(cc))
    rec["embedding_depth"] = None      # see module docstring
    return rec


def solve_nucleus(seed, period: int, *, degree: int = DEGREE, source: str = "",
                  provenance: dict | None = None, want_embedding: bool = True):
    """Newton to a period-`period` nucleus from `seed`, then build the record.
    Returns (record | None, status) where status names why it was dropped."""
    r = dcf.newton_nucleus(mp.mpc(seed), period, degree=degree, max_steps=NEWTON_STEPS)
    if not r.converged:
        return None, "no_converge"
    rec = make_atom(r.c, period, source, degree=degree, newton_res_log10=r.residual,
                    provenance=provenance, want_embedding=want_embedding)
    if rec is None:
        return None, "degenerate_or_not_minimal"
    return rec, "ok"


def identify_nucleus(seed, *, period_min=1, period_max=64, degree: int = DEGREE,
                     near: float = 1e-2, source: str = "",
                     provenance: dict | None = None, want_embedding: bool = True):
    """Find the nucleus a location sits on/near: try every period in range, keep the
    converged minimal ones within `near` of the seed, and return the record for the
    **smallest period** (the largest containing component). Returns (rec, status)."""
    seed = mp.mpc(seed)
    near = mp.mpf(str(near))
    best = None
    for p in range(period_min, period_max + 1):
        r = dcf.newton_nucleus(seed, p, degree=degree, max_steps=NEWTON_STEPS)
        if not r.converged or abs(r.c - seed) > near:
            continue
        rec = make_atom(r.c, p, source, degree=degree, newton_res_log10=r.residual,
                        provenance=provenance, want_embedding=want_embedding)
        if rec is not None:
            rec["provenance"] = dict(rec["provenance"],
                                     seed_distance=float(abs(r.c - seed)))
            best = rec
            break                       # smallest period wins
    return (best, "ok") if best else (None, "no_nucleus_near_seed")


# --------------------------------------------------------------------------- #
# depth-spanning sample (addendum §2)
# --------------------------------------------------------------------------- #
def span_by_depth(atoms: list[dict], n: int) -> list[dict]:
    """Pick <= n atoms spanning the available `log10|A|` range, deterministically.

    Reuses the roster's `select_spanning` ordering idea but keyed on the depth axis
    only. If a source has fewer than n atoms, ALL are returned — never padded, and
    never topped up from another source (that would destroy the only comparison these
    sheets exist to make)."""
    if len(atoms) <= n:
        return sorted(atoms, key=lambda a: (a["log10_abs_A"], a["id"]))
    ordered = sorted(atoms, key=lambda a: (a["log10_abs_A"], a["id"]))
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(idx.tolist()))]


def depth_histogram(atoms: list[dict], edges=(0, 1, 2, 3, 4, 5, 6, 7, 8, 99)) -> list[dict]:
    """log10|A| histogram — printed in every sheet header so a depth-mix difference
    between sources is visible rather than confounding."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = sum(1 for a in atoms if lo <= a["log10_abs_A"] < hi)
        out.append({"lo": lo, "hi": hi, "n": n})
    return out


def describe(atoms: list[dict]) -> dict:
    """The per-sheet aggregate descriptor mix (never shown per tile).

    `satellite_frac` is deliberately absent: see the module docstring. Callers that
    have a COMPLETE period-n population can add the exact value via
    `theorem_satellite_fraction`."""
    if not atoms:
        return {"n": 0}
    la = sorted(a["log10_abs_A"] for a in atoms)
    per = sorted(a["period"] for a in atoms)
    marg = sorted(a["f64_margin_deploy_decades"] for a in atoms)
    guess = sum(1 for a in atoms if a.get("atom_domain_divides_period"))
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]        # noqa: E731
    return {
        "n": len(atoms),
        "shape_available": False,
        "atom_domain_divides_period_n": guess,   # raw flag only, NOT a satellite count
        "period_min": per[0], "period_med": q(per, .5), "period_max": per[-1],
        "log10_abs_A_min": round(la[0], 2), "log10_abs_A_med": round(q(la, .5), 2),
        "log10_abs_A_max": round(la[-1], 2),
        "atom_size_max": f"{10 ** -la[0]:.3e}", "atom_size_min": f"{10 ** -la[-1]:.3e}",
        "deploy_margin_min": round(marg[0], 2), "deploy_margin_med": round(q(marg, .5), 2),
        "n_below_feasibility_floor": sum(
            1 for a in atoms if a["f64_margin_deploy_decades"] < brs.MARGIN_MIN_DECADES),
        "on_real_axis_n": sum(1 for a in atoms if a.get("on_real_axis")),
        "depth_histogram": depth_histogram(atoms),
    }


# --------------------------------------------------------------------------- #
# independent check on the shape classifier (used by the complete-enumeration sheet)
# --------------------------------------------------------------------------- #
def nu(n: int) -> int:
    """Number of period-`n` hyperbolic components of the Mandelbrot set.
    From `sum_{d|n} nu(d) = 2^(n-1)` (the degree of the period-n polynomial)."""
    total = 2 ** (n - 1)
    for d in range(1, n):
        if n % d == 0:
            total -= nu(d)
    return total


def _totient(m: int) -> int:
    r, mm, p = m, m, 2
    while p * p <= mm:
        if mm % p == 0:
            while mm % p == 0:
                mm //= p
            r -= r // p
        p += 1
    if mm > 1:
        r -= r // mm
    return r


def expected_satellite_count(n: int) -> int:
    """`sum_{q|n, q<n} phi(n/q) * nu(q)` — no sympy dependency."""
    return sum(_totient(n // q) * nu(q) for q in range(1, n) if n % q == 0)


def theorem_satellite_fraction(atoms_by_period: dict[int, list[dict]]) -> dict:
    """Exact satellite counts for a COMPLETE population, straight from the theorem.

    Available without a classifier because the count is determined: every period-q
    component carries exactly `phi(n/q)` satellites of period `n`. It says how many,
    not which — which is precisely the sheet-level diagnostic the prompt asked for."""
    rows, tot, sat = [], 0, 0
    for n in sorted(atoms_by_period):
        got = len(atoms_by_period[n])
        exp_n, exp_s = nu(n), expected_satellite_count(n)
        complete = got == exp_n
        rows.append({"period": n, "found": got, "expected_total": exp_n,
                     "complete": complete, "satellites_expected": exp_s,
                     "primitives_expected": exp_n - exp_s})
        if complete:
            tot += exp_n
            sat += exp_s
    return {"per_period": rows, "complete_total": tot, "complete_satellites": sat,
            "satellite_frac": (round(sat / tot, 3) if tot else None),
            "note": ("Exact, from sum_{q|n,q<n} phi(n/q)*nu(q) over the periods that came "
                     "out COMPLETE. Counts satellites; does not identify which atoms they "
                     "are (no verified classifier — see the module docstring).")}
