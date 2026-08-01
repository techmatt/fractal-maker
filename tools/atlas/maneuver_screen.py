#!/usr/bin/env python
r"""maneuver_screen.py — the RICHNESS SCREEN on a maneuver's nucleus.

WHY THIS IS A SEPARATE MODULE. `minibrot_maneuvers.py` is deliberately pure mpmath: no
subprocess, no torch (`docs/design/minibrot_maneuvers.md` §4). The screen renders a field,
so it spawns the engine — putting it in the operator module would spend that property for
nothing. The operators stay geometry-only and return a nucleus; this module measures that
nucleus; `steered_frontier` is the one place that does both.

WHAT IS MEASURED, AND ON WHICH FRAME. `radial_range` and `radial_rings` from
`tools/orbital/rescore_lib.ring_measures` — one ray walk, both measures, byte-identical
rays — at the 64x36 SCREEN geometry, on the **4x atom-size frame**. 4x is not a choice
made here: it is the only frame scale any orbital score has ever been computed at, and
`orbital_field_metrics.md` §2 says in terms that none of the validation transfers to
another scale. So the score describes **the ATOM, not the view** — which is exactly why
one field serves every `k` row of one snap (§7.1's shared solve, extended to the screen).

WHAT THE PAIR IS FOR. `range` is the better SCREENING statistic and `rings` the better
VALIDATION statistic (`orbital_field_metrics.md` §5) — they are Spearman 0.96 in the bulk
and disagree only at the top, which is where selection happens. Both are recorded on every
candidate; only `range` is ever selected on, and only behind `--maneuver-range-prior`.

THE CAP POLICY, AND ITS UN-RETIREMENT. Every field measure is a function of the iteration
cap (`orbital_field_metrics.md` §7), so a screen that clips is measuring the cap. This
module renders under `SCREEN_MAXITER_POLICY` = 24x the LEGACY production envelope, clamped
at 67000 — the policy `retired.md` lists as "a fitted proposal that was never adopted". It
is adopted here, narrowly and with a dated `UN-RETIRED` entry: at 64x36 the extra
iterations cost nothing (~2 ms of compute either way), and the alternative is a screen
whose numbers move when the production cap moves. Note what it actually is at these
depths: production is already the ~x8 convergent cap the 32-atom ladder measured
(`auto_maxiter.md`), and 24x-of-legacy is 3x of that before the 67000 clamp binds — i.e.
1.8-3x headroom above a cap already measured convergent, not a wild extrapolation.

Every score carries `maxiter_policy_token` (`fm.POLICY_KEY`), so a score taken here can
never be silently pooled with a `data/orbital/` record: those are all stamped legacy and
`fm.require_one_policy` raises across the two.

NOT A GATE. Nothing here refuses a maneuver. `screen_atom` returns a record, or a record
with `screened=False` and a named reason — the deep tail is genuinely unscreenable at 64 px
(the `render-one` f64 spacing guard, `orbital_field_metrics.md` §8) and that is data, not
an error.
"""
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools" / "orbital", ROOT / "tools" / "explorer",
           ROOT / "tools" / "corpus", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import field_metrics as fm        # noqa: E402  (dump_field, SCREEN_*, POLICY_KEY, tokens)
import rescore_lib as rl          # noqa: E402  (ring_measures: both measures, one ray walk)
import render_core as rc          # noqa: E402  (FW_HOME + the live policy constants)

# --------------------------------------------------------------------------- #
# geometry + cap policy
# --------------------------------------------------------------------------- #
SCREEN_FRAME_MULT = 4.0     # THE frame scale every orbital score has ever used (§2)

# (base, k, min, max) for `auto_maxiter`'s closed form. 24x the legacy production envelope
# (500, 0.30, 200, 8000) with the clamp raised to the live 67000 ceiling. Representable as
# the same four constants precisely so it gets a distinct `maxiter_policy_token`.
LEGACY_MAXITER_POLICY = (500, 0.30, 200, 8000)
SCREEN_MAXITER_MULT = 24
SCREEN_MAXITER_POLICY = (500 * SCREEN_MAXITER_MULT, 0.30, 200 * SCREEN_MAXITER_MULT, 67000)

# Concurrency: each screen is one engine PROCESS, so this is the CLAUDE.md process cap,
# not a thread count. The work is ~2 ms of compute behind a ~76 ms process spawn, so the
# pool is here to hide the spawn, not to add compute.
SCREEN_WORKERS = 4
SCREEN_THREADS = 1          # per engine process: the field is 64x36; more threads is waste


def maxiter_under(policy, fw) -> int:
    """`render_core.auto_maxiter`'s closed form, evaluated under an ARBITRARY policy.

    A second copy of a production closed form is exactly the defect
    `verification_practice.md` §1.8 names, so it is pinned: passing the LIVE constants here
    must reproduce `rc.auto_maxiter` for every fw
    (`test_maneuver_screen.py::test_the_policy_closed_form_reproduces_production`)."""
    base, k, lo, hi = policy
    fw = float(fw)
    if fw <= 0:
        return int(hi)
    lz = math.log2(float(rc.FW_HOME) / fw)
    return int(max(lo, min(hi, base * (1.0 + k * lz))))


def screen_maxiter(fw) -> int:
    return maxiter_under(SCREEN_MAXITER_POLICY, fw)


def screen_policy_token() -> str:
    return fm.policy_token(SCREEN_MAXITER_POLICY)


def screen_frame(window_scale) -> float:
    return float(window_scale) * SCREEN_FRAME_MULT


def is_screenable(window_scale) -> bool:
    """True iff the 4x frame at 64 px clears `render-one`'s f64 spacing guard (> 1e-13).

    Checked a priori so the deep tail costs a comparison rather than a process spawn and a
    parsed stderr. `orbital_field_metrics.md` §8: this is the SCREEN's geometry, not a
    property of the atom."""
    fw = screen_frame(window_scale)
    return fw > 0 and math.isfinite(fw) and (fw / fm.SCREEN_W) > 1e-13


# --------------------------------------------------------------------------- #
# the screen
# --------------------------------------------------------------------------- #
def screen_atom(cx, cy, window_scale, *, family: str = "mandelbrot",
                threads: int = SCREEN_THREADS, tmpdir=None,
                timeout: float = fm.FIELD_TIMEOUT_S) -> dict:
    """Both ring measures for one nucleus at 64x36 on its 4x frame. Never raises.

    Returns a record that ALWAYS carries `screened` and `maxiter_policy_token`; on the
    failure paths it carries `screen_reason` and no measures. A screen is a recording, so
    a failure has to be a row rather than an exception (`--maneuver-range-prior` OFF must
    behave identically whether the screen worked or not)."""
    t0 = time.time()
    tok = screen_policy_token()
    if not is_screenable(window_scale):
        return dict(screened=False, screen_reason="f64_spacing_wall_at_screen_geometry",
                    screen_s=round(time.time() - t0, 4), **{fm.POLICY_KEY: tok})
    fw = screen_frame(window_scale)
    maxiter = screen_maxiter(fw)
    try:
        import tempfile
        with tempfile.TemporaryDirectory(dir=tmpdir) as td:
            field, _side = fm.dump_field(cx, cy, fw, maxiter, Path(td) / "f.bin",
                                         width=fm.SCREEN_W, height=fm.SCREEN_H,
                                         ss=fm.SCREEN_SS, family=family, threads=threads,
                                         timeout=max(1.0, float(timeout)))
    except Exception as e:
        return dict(screened=False, screen_reason=f"dump_field:{str(e)[:120]}",
                    screen_fw=fw, screen_maxiter=maxiter,
                    screen_s=round(time.time() - t0, 4), **{fm.POLICY_KEY: tok})

    import numpy as np
    m = rl.ring_measures(field)
    finite = field[np.isfinite(field)]
    smooth_max = float(finite.max()) if finite.size else None
    return dict(
        screened=True,
        radial_range=round(float(m["radial_range"]), 4),
        radial_rings=round(float(m["radial_rings"]), 2),
        radial_range_p90=round(float(m["radial_range_p90"]), 4),
        radial_rings_p90=round(float(m["radial_rings_p90"]), 2),
        interior_fraction=round(float((~np.isfinite(field)).mean()), 4),
        escaped_px=int(finite.size),
        smooth_max=(round(smooth_max, 2) if smooth_max is not None else None),
        # Cap-headroom, recorded per field rather than asserted once. The policy is chosen
        # to be non-clipping; whether it IS on this population is a measurement, and this
        # is the column that measures it. `clamped` says the 67000 ceiling bound, i.e.
        # the 24x envelope was not actually delivered at this depth.
        screen_maxiter=maxiter,
        screen_fw=fw,
        cap_headroom=(round(1.0 - smooth_max / maxiter, 4)
                      if smooth_max is not None and maxiter else None),
        clamped=bool(maxiter >= SCREEN_MAXITER_POLICY[3]),
        screen_s=round(time.time() - t0, 4),
        **{fm.POLICY_KEY: tok},
    )


class ScreenCache:
    """Run-scoped screen results keyed on the SHARED atom key.

    The same nucleus is reached repeatedly — that is the normal case the dedup key exists
    for (§2.5) — and a re-screen is a fresh process spawn for a byte-identical answer. The
    cache is keyed on `atom_key` (not on `k`): the screen frame is 4x the ATOM, so it does
    not depend on the framing. Serialisable, so a resume does not re-pay for cells the
    killed run already screened."""

    def __init__(self, workers: int = SCREEN_WORKERS, threads: int = SCREEN_THREADS):
        self.by_key: dict[str, dict] = {}
        self.workers = int(workers)
        self.threads = int(threads)
        self.n_hits = self.n_screened = 0
        self.screen_s = 0.0

    def get(self, atom_key: str):
        return self.by_key.get(atom_key)

    def screen_many(self, jobs: list[dict], *, budget_s: float | None = None) -> dict:
        """`jobs` are `{atom_key, cx, cy, window_scale, family}`. Returns key -> record.

        Concurrent because the cost is process spawn, not compute; `SCREEN_WORKERS` is the
        CLAUDE.md concurrent-PROCESS cap, and each child is pinned to one thread.

        `budget_s` bounds the WHOLE pass, and it is not optional hygiene. The walk checks
        its active-time cap between batches, so anything unbounded *inside* a batch is
        outside the cap: with one 60 s field timeout per screen and a fat neighbourhood
        enumeration, a pathological batch could spend half an hour here while the budget
        logic believed it was inside its cap — "a backstop longer than the job's budget is
        not a backstop" (`CLAUDE.md`, four rules), one level down. Each screen's own timeout
        is additionally clamped to what is left of the pass, and a job the pass never
        reached comes back as an unscreened row with a named reason, never as a silent
        absence."""
        todo, out = {}, {}
        for j in jobs:
            key = j["atom_key"]
            hit = self.by_key.get(key)
            if hit is not None:
                self.n_hits += 1
                out[key] = hit
            elif key not in todo:
                todo[key] = j
        if not todo:
            return out
        t0 = time.time()
        items = list(todo.values())
        deadline = (t0 + float(budget_s)) if budget_s else None

        def one(j):
            if deadline is not None:
                left = deadline - time.time()
                if left <= 0:
                    return j["atom_key"], dict(
                        screened=False, screen_reason="screen_budget_exhausted",
                        **{fm.POLICY_KEY: screen_policy_token()})
                per = min(fm.FIELD_TIMEOUT_S, max(1.0, left))
            else:
                per = fm.FIELD_TIMEOUT_S
            return j["atom_key"], screen_atom(j["cx"], j["cy"], j["window_scale"],
                                              family=j.get("family", "mandelbrot"),
                                              threads=self.threads, timeout=per)

        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(items)))) as ex:
            for key, rec in ex.map(one, items):
                self.by_key[key] = rec
                out[key] = rec
                self.n_screened += 1
        self.screen_s += time.time() - t0
        return out

    def state_dict(self) -> dict:
        return dict(by_key=self.by_key, n_hits=self.n_hits, n_screened=self.n_screened,
                    screen_s=round(self.screen_s, 3))

    def load_state(self, d: dict):
        self.by_key = dict(d.get("by_key") or {})
        self.n_hits = int(d.get("n_hits", 0))
        self.n_screened = int(d.get("n_screened", 0))
        self.screen_s = float(d.get("screen_s", 0.0))


# --------------------------------------------------------------------------- #
# the running range distribution -> a BOUNDED priority term
# --------------------------------------------------------------------------- #
class RangeDistribution:
    """The run's own accumulating `radial_range` distribution, and the percentile of a
    value against it.

    Against the RUN's own distribution rather than a fixed threshold, because absolute ring
    scores mean nothing across resolutions or cap policies (`orbital_field_metrics.md` §5,
    §7) — only orderings within one (geometry, policy) pair do, and this is one such pair.

    Below `n_min` observations the percentile is not a percentile, so it returns exactly
    0.5, which maps to a priority delta of exactly zero: the prior is the unchanged
    `NEUTRAL_PRIOR` until the run has seen enough to rank."""

    def __init__(self, n_min: int = 8):
        self.values: list[float] = []
        self.n_min = int(n_min)

    def add(self, v):
        if v is not None and math.isfinite(float(v)):
            self.values.append(float(v))

    def percentile_of(self, v) -> float:
        """Fraction of observed values strictly below `v`, in [0, 1]. 0.5 when unranked."""
        if v is None or not math.isfinite(float(v)) or len(self.values) < self.n_min:
            return 0.5
        v = float(v)
        below = sum(1 for x in self.values if x < v)
        return below / len(self.values)

    def state_dict(self) -> dict:
        return dict(values=self.values, n_min=self.n_min)

    def load_state(self, d: dict):
        self.values = [float(x) for x in (d.get("values") or [])]
        self.n_min = int(d.get("n_min", self.n_min))


def range_prior_delta(pct: float, gain: float) -> float:
    """The bounded priority term: `gain * (percentile - 0.5)`, i.e. `+/- gain/2`.

    SYMMETRIC ON PURPOSE. The mean maneuver prior is unchanged by the term, so turning the
    flag on REORDERS maneuver nodes without inflating maneuver priority as a class — the
    floor stays the only thing that lets a maneuver out-compete a scored node
    (`minibrot_maneuvers.md` §3). With the shipped gain the term spans +/-0.25 around
    `NEUTRAL_PRIOR = 1.0` against an ordinary node's `cheap_eord` in [0, K-1] = [0, 3], so
    the best-ranked maneuver still loses to any ordinary node scoring above 1.25."""
    return float(gain) * (float(pct) - 0.5)
