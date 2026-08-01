#!/usr/bin/env python
r"""minibrot_maneuvers.py — minibrot moves as candidate MOVES inside a descent.

WHAT THIS IS FOR. Seven source sheets settled what minibrots are good for: every source
rated viable is downstream of a descent, and every source that enumerates minibrots from
first principles is dead. So a minibrot is **a reframing operator applied to a location
already found**, not a source of locations. This module is that operator set, shaped to
be called from inside the walk (`tools/atlas/steered_frontier.py`), never as a standalone
generator.

TWO OPERATORS, NOT FIVE.

* `snap_to_nucleus(view, k)` — atom-domain probe at the view's centre -> Newton -> recentre
  on the nucleus. `k` sets the frame: `None` preserves the view's own `fw`; otherwise
  `fw = k * atom size`. That one operator subsumes snap-preserving (k=None),
  reframe-outward (large k), descend-to-child (small k) and ascend-to-scale — they differ
  only in `k`, and collapsing them makes the per-move attribution one axis instead of four
  labels that alias each other.
* `lateral_to_sibling(view)` — move to a nearby sibling nucleus at comparable scale. Uses
  the neighbourhood probe the source sheets already validated (`sources.src_neighborhood`):
  seeds drawn at radii measured in units of the PARENT's own window scale, so "nearby"
  means the same thing for a shallow and a deep parent, and a period ceiling scaled to the
  parent's period (a flat ceiling silently finds nothing around deep parents).

REUSED, NOT REBUILT.
* Atom size is `1/|A|` from the `A` instrument (`deep_center_finder.atom_instrument`).
  **The naive degree-2 λ² law is forbidden**: at d>=3 it under-sizes the atom by 4–2497x
  and frames it all-black (`docs/design/atom_instrument.md`).
* Newton, minimality, the sector-canonical dedup key and the full descriptor record come
  from `tools/sources/atom_lib.py` (`solve_nucleus` / `identify_nucleus` / `make_atom`),
  the common denominator the source sheets are built on. An atom found here therefore
  collapses to the SAME `id` as one found by any sheet or already in the triage pool.
* The visited-atom key is the READ-TIME canonicalization
  (`deep_center_finder.snapped_dedup_key`, which is `snap_near_zero` + the sector-canonical
  rounded key) — the same function `collapse_population` uses. Multiple frontier members
  snapping to one nucleus is the normal case, and this is what collapses them; do not
  write another key.

UNAVAILABILITY IS THE NORMAL CASE, NOT AN ERROR. Newton convergence ran ~17% in the
200-atom enumeration (2,160 solves -> 200 kept). Every entry point returns a `Maneuver`
whose `available` is False with a named `reason`, often. Callers must expect it: a
maneuver quota is a quota **of available**, never of all slots.

COST. The probe is an enumeration cost and enumeration measured ~25x the screening cost,
so it is bounded three ways here and governed a fourth way by the caller:
  1. the period candidate set comes from ONE orbit pass — the divisors of the top-`n_periods`
     argmins of |z_k|, capped at `3*n_periods` Newton solves — not a period sweep (measured:
     median 8, max 12 solves per snap probe);
  2. `lateral_to_sibling` caps its probes and scales `period_max` to the parent period;
  3. every call is timed (`probe_s`) so the caller can price it;
  4. `ProbeGovernor` (below) bounds how often a probe fires per rung and caches per region.

WHAT THIS MODULE DOES NOT DO. It does not render, score, gate, or decide priority. It
returns a geometry plus provenance; the caller decides what to do with it. In particular
the aesthetic head never enters here — these are field/geometry operators, exactly like
the rest of the walk's proposal machinery.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools" / "sourcing", ROOT / "tools" / "sources", ROOT / "tools" / "descent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mpmath as mp                       # noqa: E402
import deep_center_finder as dcf          # noqa: E402
import atom_lib as al                     # noqa: E402
import triage_store as ts                 # noqa: E402  (atom_id — shared with the triage pool)

# --------------------------------------------------------------------------- #
# Defaults. Every one is a constructor argument too; these are the values the
# shakedown ran at and the ones a caller gets by omission.
# --------------------------------------------------------------------------- #
MAX_PERIOD = 64          # ceiling on the atom-domain orbit scan (period candidates)
N_PERIODS = 4            # Newton solves per snap probe = the top-N argmins of |z_k|
SNAP_MAX_FW_MULT = 1.0   # the nucleus must land within this many frame widths of centre
DEFAULT_K = (None, 4.0, 16.0)  # preserve-fw, the 4x "is this atom good?" frame, and 16x
MAX_FW = 3.0             # never reframe wider than a base-scale root view
NODE_WIDTH = 384         # descent node presentation width — sets the f64 spacing wall
PERTURB_SPACING = 1e-13  # Rust PERTURB_SPACING: below this the f64 backend quantizes
WALL_MARGIN_DECADES = 0.5  # required headroom above the spacing wall at NODE_WIDTH

LAT_RADII = (2.0, 8.0, 32.0)   # sibling probe radii, in units of the parent window scale
LAT_PROBES = 3                 # probe seeds per lateral call (bounded enumeration cost)
LAT_PERIOD_HEADROOM = 3.0      # period ceiling = headroom x parent period ...
LAT_PERIOD_CAP = 120           # ... capped here
LAT_SCALE_TOL_DECADES = 1.0    # "comparable scale": |log10(w_sib / w_par)| <= this
LAT_SEED_PERIODS = True        # seed identify_nucleus from the atom-domain probe (not a sweep)
LAT_N_PERIODS = N_PERIODS      # argmins kept per seeded lateral probe (-> <=3x solves)
LAT_LOW_SWEEP = 16             # ... but periods 2..this are ALWAYS swept exactly (see below)

# c-plane partitions only: a julia/phoenix viewport is a z-plane and has no nucleus in the
# parameter-plane sense, so the operators are simply not defined there.
PARTITION_DEGREE = {"mandelbrot": 2, "multibrot3": 3, "multibrot4": 4, "multibrot5": 5}


def degree_of(partition: str):
    """Multibrot degree for a c-plane partition, or None where maneuvers are undefined."""
    return PARTITION_DEGREE.get(partition)


# --------------------------------------------------------------------------- #
# The result type.
# --------------------------------------------------------------------------- #
@dataclass
class Maneuver:
    """One operator application. `available=False` is the normal outcome, not an error."""
    op: str                       # "snap_to_nucleus" | "lateral_to_sibling"
    available: bool
    reason: str = ""              # why unavailable (empty when available)
    k: float | None = None        # framing multiple of atom size; None = preserve parent fw
    # the proposed view
    cx: str | None = None
    cy: str | None = None
    fw: float | None = None
    depth: int | None = None
    # the atom it is a reframing of
    atom_id: str | None = None
    atom_key: str | None = None
    period: int | None = None
    log10_abs_A: float | None = None
    window_scale: float | None = None
    f64_margin_node_decades: float | None = None
    f64_margin_deploy_decades: float | None = None
    # parent view (the thing being reframed)
    parent_node_id: int | None = None
    parent_cx: float | None = None
    parent_cy: float | None = None
    parent_fw: float | None = None
    parent_depth: int | None = None
    # cost + bookkeeping
    probe_s: float = 0.0
    newton_solves: int = 0
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return asdict(self)


def _unavailable(op, reason, view, t0, solves=0, k=None, **extra) -> Maneuver:
    # `k` is carried on the UNAVAILABLE path too. It was not, and the shakedown's log could
    # then not split availability by framing: every refusal stamped k=None regardless of the
    # k actually requested, silently piling all of them into one bucket. Two of the refusal
    # reasons (`f64_spacing_wall`, `fw_over_root_scale`) are k-DEPENDENT by construction, so
    # that bucket is exactly the one a cost/framing read needs.
    return Maneuver(op=op, available=False, reason=reason, k=(None if k is None else float(k)),
                    parent_node_id=view.get("node_id"), parent_cx=float(view["cx"]),
                    parent_cy=float(view["cy"]), parent_fw=float(view["fw"]),
                    parent_depth=int(view.get("depth", 0)),
                    probe_s=time.time() - t0, newton_solves=solves, extra=extra)


# --------------------------------------------------------------------------- #
# The atom-domain probe: ONE orbit pass -> candidate periods.
# --------------------------------------------------------------------------- #
def period_candidates(c, degree: int, max_period: int = MAX_PERIOD,
                      n: int = N_PERIODS, max_solves: int = None) -> list[int]:
    """Candidate periods for a nucleus near `c`, in the order Newton should try them.

    Classical atom-domain reasoning: along the critical orbit of a parameter inside the
    atom domain of a period-p component, `|z_k|` is small at `k = p`. So ONE orbit pass
    ranks every period at once and we Newton only a handful, instead of sweeping
    1..max_period — which is what makes a full enumeration ~25x the screening cost.

    THE ARGMIN IS A MULTIPLE, NOT THE PERIOD — the correction that makes this work. A
    nucleus is SUPERATTRACTING, so just off it `|z_{2p}| ~ |z_p|^2 << |z_p|` and the global
    argmin lands on a high multiple of `p`, never on `p`. Newton at that multiple returns a
    non-minimal solution and `atom_lib.make_atom` rejects it — which is exactly what a
    naive argmin implementation does, silently, on every view. So the candidate set is the
    argmins' DIVISORS, tried in INCREASING period order: the smallest period is the largest
    containing component, the same "smallest period wins" rule `identify_nucleus` uses.
    Period 1 is skipped — its nucleus is `c = 0`, which `ORIGIN_EPS` rejects anyway.

    The orbit is walked in f64 (this is a RANKING; Newton refines at full precision) and
    truncated at escape, because past escape the values carry no atom-domain information.
    """
    z = complex(0.0, 0.0)
    cc = complex(float(c.real), float(c.imag)) if hasattr(c, "real") else complex(c)
    esc = 2.0 ** (1.0 / max(degree - 1, 1)) * 2.0
    vals: list[tuple[float, int]] = []
    for k in range(1, max_period + 1):
        z = (z * z + cc) if degree == 2 else (z ** degree + cc)
        a = abs(z)
        if not math.isfinite(a) or a > esc:
            break
        vals.append((a, k))
    if not vals:
        return []
    vals.sort()
    cands: set = set()
    for _a, m in vals[:max(1, n)]:
        cands.update(d for d in range(2, m + 1) if m % d == 0)
    return sorted(cands)[:max_solves or (3 * max(1, n))]


def _atom_record(c, period, degree, source, provenance):
    """`atom_lib.make_atom` with the shared canonical id/key, or None if degenerate."""
    return al.make_atom(c, period, source, degree=degree, provenance=provenance)


def atom_key_of(rec: dict) -> str:
    """READ-TIME canonical dedup key for an atom record (`snapped_dedup_key`).

    Deliberately recomputed from the record's STORED decimal-string coords with the very
    function `collapse_population` uses at read, so a key formed here is byte-comparable
    with one formed over the triage pool / a source sheet / another run's rows."""
    return dcf.snapped_dedup_key(rec["cx"], rec["cy"], int(rec["degree"]), al.DEDUP_DPS)


def _wall_margin_decades(fw: float, width: int) -> float:
    """Decades of headroom before a `fw`-wide frame at `width` px crosses the f64
    pixel-spacing wall. Negative predicts an f64 quantisation failure a priori."""
    if fw <= 0:
        return -math.inf
    return math.log10(fw / width) - math.log10(PERTURB_SPACING)


def _frame_for(rec: dict, k, parent_fw: float, max_fw: float):
    """(fw, reason|None) for a proposed view on atom `rec` under framing `k`."""
    fw = float(parent_fw) if k is None else float(k) * float(rec["window_scale"])
    if not (fw > 0) or not math.isfinite(fw):
        return None, "bad_fw"
    if fw > max_fw:
        return None, "fw_over_root_scale"
    if _wall_margin_decades(fw, NODE_WIDTH) < WALL_MARGIN_DECADES:
        return None, "f64_spacing_wall"
    return fw, None


# --------------------------------------------------------------------------- #
# Operator 1 — snap_to_nucleus
# --------------------------------------------------------------------------- #
def _solve_snap(view: dict, degree: int, max_period: int, n_periods: int,
                snap_max_fw_mult: float, source: str):
    """The Newton half of a snap: atom-domain probe at the view centre -> nucleus record.

    Split out because **the nucleus does not depend on the framing.** `k` only chooses
    `fw` afterwards, so N framings of one view cost ONE solve, not N — which is what
    makes adding a k to the set (`k=16`) a free reframing rather than another probe.

    Returns `(rec, period, solves, reason, periods)`; `rec is None` names the refusal."""
    al.set_precision()
    cx, cy, fw = float(view["cx"]), float(view["cy"]), float(view["fw"])
    c0 = mp.mpc(mp.mpf(str(view["cx"])), mp.mpf(str(view["cy"])))
    periods = period_candidates(c0, degree, max_period, n_periods)
    if not periods:
        return None, None, 0, "orbit_escaped_immediately", periods

    solves = 0
    near = mp.mpf(str(snap_max_fw_mult * fw))
    last = "no_converge"
    for p in periods:
        solves += 1
        r = dcf.newton_nucleus(c0, p, degree=degree, max_steps=al.NEWTON_STEPS)
        if not r.converged:
            last = "no_converge"
            continue
        if abs(r.c - c0) > near:
            last = "nucleus_outside_frame"
            continue
        rec = _atom_record(r.c, p, degree, source, {
            "seed_cx": cx, "seed_cy": cy, "seed_fw": fw,
            "seed_distance": float(abs(r.c - c0)),
            "period_rank": periods.index(p), "period_candidates": periods,
        })
        if rec is None:
            last = "degenerate_or_not_minimal"
            continue
        return rec, p, solves, "", periods
    return None, None, solves, last, periods


def snap_to_nucleus_multi(view: dict, ks, *, degree: int = 2,
                          max_period: int = MAX_PERIOD, n_periods: int = N_PERIODS,
                          snap_max_fw_mult: float = SNAP_MAX_FW_MULT,
                          max_fw: float = MAX_FW,
                          source: str = "maneuver_snap") -> list[Maneuver]:
    """One snap probe, one `Maneuver` per requested framing `k` — in `ks` order.

    COST ATTRIBUTION. The shared solve is charged to the FIRST row only; later rows carry
    just their own (negligible) framing time, `newton_solves=0` and `extra.reused_solve`.
    A sum of `probe_s` over the emitted rows is therefore the true cost of the call, and a
    per-`k` cost read is not inflated by N copies of one solve."""
    t0 = time.time()
    ks = list(ks) or [None]
    cx, cy, fw = float(view["cx"]), float(view["cy"]), float(view["fw"])
    rec, p, solves, reason, periods = _solve_snap(
        view, degree, max_period, n_periods, snap_max_fw_mult, source)

    out: list[Maneuver] = []
    for i, k in enumerate(ks):
        t_row = t0 if i == 0 else time.time()
        n_solves = solves if i == 0 else 0
        shared = {} if i == 0 else {"reused_solve": True}
        if rec is None:
            out.append(_unavailable("snap_to_nucleus", reason, view, t_row, n_solves, k=k,
                                    period_candidates=periods, **shared))
            continue
        newfw, why = _frame_for(rec, k, fw, max_fw)
        if newfw is None:
            out.append(_unavailable("snap_to_nucleus", why, view, t_row, n_solves, k=k,
                                    period=p, window_scale=rec["window_scale"], **shared))
            continue
        out.append(Maneuver(
            op="snap_to_nucleus", available=True, k=(None if k is None else float(k)),
            cx=rec["cx"], cy=rec["cy"], fw=newfw, depth=int(view.get("depth", 0)),
            atom_id=rec["id"], atom_key=atom_key_of(rec), period=p,
            log10_abs_A=rec["log10_abs_A"], window_scale=float(rec["window_scale"]),
            f64_margin_node_decades=round(_wall_margin_decades(newfw, NODE_WIDTH), 4),
            f64_margin_deploy_decades=rec["f64_margin_deploy_decades"],
            parent_node_id=view.get("node_id"), parent_cx=cx, parent_cy=cy,
            parent_fw=fw, parent_depth=int(view.get("depth", 0)),
            probe_s=time.time() - t_row, newton_solves=n_solves,
            extra=dict(seed_distance=rec["provenance"]["seed_distance"],
                       period_rank=rec["provenance"]["period_rank"],
                       degree=degree, atom_size=rec["size"], **shared),
        ))
    return out


def snap_to_nucleus(view: dict, k=None, *, degree: int = 2, max_period: int = MAX_PERIOD,
                    n_periods: int = N_PERIODS, snap_max_fw_mult: float = SNAP_MAX_FW_MULT,
                    max_fw: float = MAX_FW, source: str = "maneuver_snap") -> Maneuver:
    """Atom-domain probe at the view centre -> Newton -> recentre on the nucleus.

    `view` is `{cx, cy, fw, depth, node_id}` (floats/ints; cx/cy may be strings).
    `k` is the frame: None preserves `view["fw"]`; otherwise `fw = k * atom size`.

    Returns an unavailable `Maneuver` — with the reason named — when no candidate period
    converges, when the nucleus is outside the frame (that would be a teleport, not a
    snap), when the atom is degenerate/non-minimal, or when the requested framing crosses
    the f64 pixel-spacing wall at the descent's node width.

    Single-`k` convenience over `snap_to_nucleus_multi`; a caller with several framings
    must use that one or it pays the same solve once per `k`.
    """
    return snap_to_nucleus_multi(view, [k], degree=degree, max_period=max_period,
                                 n_periods=n_periods, snap_max_fw_mult=snap_max_fw_mult,
                                 max_fw=max_fw, source=source)[0]


# --------------------------------------------------------------------------- #
# Operator 2 — lateral_to_sibling
# --------------------------------------------------------------------------- #
def lateral_to_sibling(view: dict, rng, *, degree: int = 2, k=None,
                       radii=LAT_RADII, n_probes: int = LAT_PROBES,
                       period_headroom: float = LAT_PERIOD_HEADROOM,
                       period_cap: int = LAT_PERIOD_CAP,
                       scale_tol: float = LAT_SCALE_TOL_DECADES,
                       max_fw: float = MAX_FW, parent_rec: dict | None = None,
                       seed_periods: bool = LAT_SEED_PERIODS,
                       n_periods: int = LAT_N_PERIODS,
                       low_sweep: int = LAT_LOW_SWEEP,
                       source: str = "maneuver_lateral") -> Maneuver:
    """Move to a nearby sibling nucleus at comparable scale.

    Needs a parent atom first: pass `parent_rec` (an `atom_lib` record, e.g. from a snap
    that already fired at this node) or the call runs its own snap probe. Probe seeds are
    drawn at `radii` in units of the PARENT's window scale and the period ceiling is
    scaled to the parent's period — both from `sources.src_neighborhood`, where a flat
    ceiling was measured to return 15 atoms from 360 probes because it, not the source,
    was the limit.

    "Comparable scale" is enforced: a candidate whose window scale differs from the
    parent's by more than `scale_tol` decades is not a sibling, it is a different rung.

    COST. `seed_periods` (default on) hands `identify_nucleus` the atom-domain probe's
    ranked candidates instead of letting it sweep `1..pmax` — the same correction that
    made the snap probe cheap. The sweep was 84% of all maneuver probe cost in the
    shakedown for 23% of the pushed nodes, because `pmax` reaches `LAT_PERIOD_CAP` and
    every period costs a full Newton solve. Setting it False restores the sweep, which is
    the reference implementation the seeded path is differentially tested against
    (`tools/atlas/bench_lateral_seeding.py`); the caps below are then the fallback lever
    and are deliberately NOT lowered while seeding carries the cost."""
    t0 = time.time()
    al.set_precision()
    fw = float(view["fw"])
    solves = 0

    if parent_rec is None:
        snap = snap_to_nucleus(view, None, degree=degree, source=source + "_parent")
        solves += snap.newton_solves
        if not snap.available:
            return _unavailable("lateral_to_sibling", "no_parent_atom:" + snap.reason,
                                view, t0, solves, k=k)
        parent_rec = dict(id=snap.atom_id, cx=snap.cx, cy=snap.cy, period=snap.period,
                          window_scale=snap.window_scale, degree=degree)

    w_par = float(parent_rec["window_scale"])
    pcx, pcy = mp.mpf(str(parent_rec["cx"])), mp.mpf(str(parent_rec["cy"]))
    pmax = min(period_cap, max(24, int(period_headroom * int(parent_rec["period"]))))
    tried, last = 0, "no_sibling_found"
    for _ in range(max(1, n_probes)):
        rad = float(radii[int(rng.integers(len(radii)))]) * w_par
        th = float(rng.random()) * 2.0 * math.pi
        seed = mp.mpc(pcx + mp.mpf(rad * math.cos(th)), pcy + mp.mpf(rad * math.sin(th)))
        tried += 1
        cands = None
        if seed_periods:
            # HYBRID, and the split is not arbitrary. The sweep's floor is pmax >= 24, so
            # the low head is a FIXED, cheap 15 solves while the tail runs to 120 — 80% of
            # the cost. The head is also exactly where the atom-domain ranking is weakest
            # (a seed sits inside many low-period atom domains at once, and "smallest
            # period wins" is what stops the probe returning the parent itself), and the
            # tail is where it is strongest (a deep atom has one sharp |z_p| minimum).
            # So: sweep the cheap head exactly, rank the expensive tail. Measured against
            # the pure sweep in bench_lateral_seeding.py.
            cands = sorted(set(period_candidates(seed, degree, pmax, n_periods)) |
                           set(range(2, min(int(low_sweep), pmax) + 1)))
            if not cands:
                last = "orbit_escaped_immediately"
                continue
        rec, why = al.identify_nucleus(
            seed, period_min=1, period_max=pmax, degree=degree, near=rad * 4,
            source=source, periods=cands,
            provenance={"parent_atom_id": parent_rec["id"],
                        "parent_period": int(parent_rec["period"]),
                        "probe_period_max": pmax, "probe_periods": cands,
                        "radius_over_parent_w": (rad / w_par if w_par else None)})
        # upper bound either way: identify_nucleus stops at the first success, and the
        # seeded set is what replaces the 1..pmax sweep.
        solves += len(cands) if cands is not None else pmax
        if rec is None:
            last = why
            continue
        if rec["id"] == parent_rec["id"]:
            last = "hit_parent"
            continue
        w_sib = float(rec["window_scale"])
        if w_sib <= 0 or abs(math.log10(w_sib / w_par)) > scale_tol:
            last = "scale_mismatch"
            continue
        newfw, whyf = _frame_for(rec, k, fw, max_fw)
        if newfw is None:
            last = whyf
            continue
        return Maneuver(
            op="lateral_to_sibling", available=True, k=(None if k is None else float(k)),
            cx=rec["cx"], cy=rec["cy"], fw=newfw, depth=int(view.get("depth", 0)),
            atom_id=rec["id"], atom_key=atom_key_of(rec), period=rec["period"],
            log10_abs_A=rec["log10_abs_A"], window_scale=w_sib,
            f64_margin_node_decades=round(_wall_margin_decades(newfw, NODE_WIDTH), 4),
            f64_margin_deploy_decades=rec["f64_margin_deploy_decades"],
            parent_node_id=view.get("node_id"), parent_cx=float(view["cx"]),
            parent_cy=float(view["cy"]), parent_fw=fw,
            parent_depth=int(view.get("depth", 0)),
            probe_s=time.time() - t0, newton_solves=solves,
            extra=dict(parent_atom_id=parent_rec["id"],
                       parent_period=int(parent_rec["period"]),
                       parent_window_scale=w_par, probes_tried=tried,
                       scale_ratio_decades=round(math.log10(w_sib / w_par), 4),
                       degree=degree, atom_size=rec["size"]),
        )
    return _unavailable("lateral_to_sibling", last, view, t0, solves, k=k,
                        probes_tried=tried, parent_atom_id=parent_rec["id"],
                        parent_period=int(parent_rec["period"]))


# --------------------------------------------------------------------------- #
# Cost governor — §3's "a probability IS used, but as a COST governor".
# --------------------------------------------------------------------------- #
class ProbeGovernor:
    """Bounds how often the (expensive) atom probe fires, two ways at once.

    * **Rate.** A Bernoulli(`p`) draw per rung. This is NOT a selection probability —
      selection is the reserved frontier floor (`steered_frontier`), which is a slot, not
      a coin flip. This coin only bounds enumeration COST.
    * **Region cache.** A view is quantised to a coarse (degree, cx, cy, fw-decade) cell;
      a cell that has already been probed is skipped outright, whatever the coin says.
      Siblings in a hot lineage sit in one cell, and re-probing them re-derives the same
      nucleus at full Newton cost.

    Pure and serialisable (`state_dict`/`load_state`) so a resume does not re-probe cells
    the killed run already paid for.
    """

    def __init__(self, p: float, rng, cell_px: float = 4.0):
        self.p = float(p)
        self.rng = rng
        self.cell_px = float(cell_px)     # cell side, in units of the view's own fw
        self.seen: set = set()
        self.n_rolled = self.n_fired = self.n_coin_skip = self.n_cache_skip = 0

    def cell(self, degree: int, cx, cy, fw) -> str:
        fw = float(fw)
        s = fw * self.cell_px
        return (f"{degree}|{math.floor(float(cx) / s)}|{math.floor(float(cy) / s)}|"
                f"{round(math.log10(fw), 1)}")

    def should_probe(self, degree: int, cx, cy, fw) -> tuple[bool, str]:
        self.n_rolled += 1
        cell = self.cell(degree, cx, cy, fw)
        if cell in self.seen:
            self.n_cache_skip += 1
            return False, "region_cached"
        if self.p < 1.0 and float(self.rng.random()) >= self.p:
            self.n_coin_skip += 1
            return False, "cost_governor"
        self.seen.add(cell)
        self.n_fired += 1
        return True, ""

    def state_dict(self) -> dict:
        return dict(p=self.p, cell_px=self.cell_px, seen=sorted(self.seen),
                    n_rolled=self.n_rolled, n_fired=self.n_fired,
                    n_coin_skip=self.n_coin_skip, n_cache_skip=self.n_cache_skip)

    def load_state(self, d: dict):
        self.seen = set(d.get("seen", []))
        self.n_rolled = int(d.get("n_rolled", 0))
        self.n_fired = int(d.get("n_fired", 0))
        self.n_coin_skip = int(d.get("n_coin_skip", 0))
        self.n_cache_skip = int(d.get("n_cache_skip", 0))


def parse_k_spec(spec: str) -> list:
    """`"none,4,16"` -> `[None, 4.0, 16.0]`. `none` is the fw-preserving snap."""
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(None if tok.lower() in ("none", "keep", "preserve") else float(tok))
    return out or [None]
