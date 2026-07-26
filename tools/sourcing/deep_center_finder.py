"""High-precision deep-center finder (hand-curation track).

Produces **valid deep centers** — points on ∂M whose neighborhoods sustain
structure at depth — as high-precision **decimal-string** centers that flow
straight into the proven perturbation render tier (bare `render` / `sheet`,
`iterate_location` auto-selects perturbation at spacing ≤ 1e-13).

This is the sourcing component the deep probe proved missing
(`docs/design/deep_zoom_sourcing.md`): the guided-descend walker is
f64-bound (`Frame.center: Complex<f64>`) and structurally cannot localize a
center below ~f64 resolution, so deep q4 harvesting needs a component that
*tracks ∂M at high precision*. This does exactly that, two ways:

  * **Nucleus** (period-p hyperbolic-component center): Newton on z_p(c)=0,
    the critical orbit returning to 0. Lands minibrot centers — self-similar
    over a band around the component's own size.
  * **Misiurewicz** (pre-periodic z_{k+n}=z_k): Newton on that residual. Lands
    points that stay *on* ∂M at every scale, so structure persists across many
    decades (the probe's sustained-q4 Seahorse center is one of these).

All Newton arithmetic is mpmath high precision (correctly rounded — a Newton
solver needs accurate division, unlike `hp.rs`'s projection-absorbed orbit
arithmetic). Coordinates leave as decimal strings; nothing here trusts a
classifier, scores, or emits — Matt's eye picks the beautiful ones.

CLI (also importable as a library — hand-curation calls it repeatedly):

    # Identify what an f64 seed converges to (period / preperiod scan):
    uv run python tools/sourcing/deep_center_finder.py scan \
        --seed -0.743643887 0.131825904 --max-period 24

    # Refine a nucleus and emit a render-ready deep center:
    uv run python tools/sourcing/deep_center_finder.py nucleus \
        --seed -0.1592 1.0317 --period 3

    # Refine a Misiurewicz point:
    uv run python tools/sourcing/deep_center_finder.py misiurewicz \
        --seed -0.743643887 0.131825904 --preperiod 5 --period 3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import mpmath as mp


# ---------------------------------------------------------------------------
# Precision sizing. A center that localizes a frame of width `fw` must be known
# to well below `fw` relative to |c|~O(1); Newton converges quadratically so we
# just run generously above what any emitted depth needs.
# ---------------------------------------------------------------------------
def dps_for_fw(fw: float, guard: int = 30) -> int:
    """Decimal working precision sufficient to localize a frame of width `fw`."""
    import math
    need = int(math.ceil(-math.log10(fw))) if fw > 0 else 20
    return max(50, need + guard)


def emit_digits_for_fw(fw: float, guard: int = 15) -> int:
    """How many significant digits to serialize a center for a frame of width `fw`."""
    import math
    need = int(math.ceil(-math.log10(fw))) if fw > 0 else 20
    return need + guard


# ---------------------------------------------------------------------------
# Core orbit + derivative recurrences (all in the ambient mpmath precision).
#   z_0 = 0,  z_{n+1} = z_n^d + c              (d = `degree`, the multibrot power)
#   d_n = dz_n/dc:  d_0 = 0,  d_{n+1} = d · z_n^{d-1} · d_n + 1
#
# Degree-2 (Mandelbrot) keeps its ORIGINAL expressions textually untouched, so
# every d=2 result is byte-identical to before the multibrot generalization
# (regression: `tests/test_deep_center_finder_degree.py`). This mirrors the Rust
# backend split (`sample_flags` for d=2 vs `sample_multibrot` for d≥3): the
# quadratic bytes are load-bearing and are never re-routed through the general
# power kernel. For d≥3 the recurrence uses z^{d-1} (computed once per step and
# reused for both z^d = z^{d-1}·z and the derivative factor d·z^{d-1}).
# ---------------------------------------------------------------------------
def _orbit(c, n, degree=2):
    """Return (z_n, d_n) after n steps of the critical orbit at parameter c."""
    z = mp.mpc(0)
    d = mp.mpc(0)
    if degree == 2:
        for _ in range(n):
            d = 2 * z * d + 1
            z = z * z + c
    else:
        for _ in range(n):
            zpm1 = z ** (degree - 1)          # z^{d-1}
            d = degree * zpm1 * d + 1
            z = zpm1 * z + c                  # z^d = z^{d-1} · z
    return z, d


def _orbit_at(c, k, n, degree=2):
    """Return (z_k, d_k, z_{k+n}, d_{k+n}) — orbit + derivative captured at step
    k and at step k+n. For the Misiurewicz residual z_{k+n} - z_k."""
    z = mp.mpc(0)
    d = mp.mpc(0)
    zk = dk = None
    if degree == 2:
        for i in range(k + n):
            if i == k:
                zk, dk = z, d
            d = 2 * z * d + 1
            z = z * z + c
    else:
        for i in range(k + n):
            if i == k:
                zk, dk = z, d
            zpm1 = z ** (degree - 1)
            d = degree * zpm1 * d + 1
            z = zpm1 * z + c
    if k == 0:
        zk, dk = mp.mpc(0), mp.mpc(0)
    return zk, dk, z, d


# ---------------------------------------------------------------------------
# Newton solvers.
# ---------------------------------------------------------------------------
@dataclass
class NewtonResult:
    c: object            # mpmath.mpc — refined parameter
    converged: bool
    iters: int
    residual: float      # |g(c)| at the final iterate (log10-ish scale)
    kind: str            # "nucleus" | "misiurewicz"
    period: int
    preperiod: int = 0
    degree: int = 2      # multibrot power d in z^d + c (2 = Mandelbrot)


def newton_nucleus(c0, period, *, degree=2, max_steps=200, tol_dps_margin=6):
    """Newton on z_p(c) = 0 (period-p nucleus). Returns a NewtonResult."""
    c = mp.mpc(c0)
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    residual = mp.inf
    it = 0
    for it in range(1, max_steps + 1):
        z, d = _orbit(c, period, degree)
        residual = abs(z)
        if d == 0:
            break
        step = z / d
        c = c - step
        if abs(step) < tol and residual < tol:
            break
    z, _ = _orbit(c, period, degree)
    residual = abs(z)
    conv = residual < tol
    return NewtonResult(c=c, converged=bool(conv), iters=it,
                        residual=float(mp.log10(residual)) if residual > 0 else -999.0,
                        kind="nucleus", period=period, degree=degree)


def newton_misiurewicz(c0, preperiod, period, *, degree=2, max_steps=200, tol_dps_margin=6):
    """Newton on z_{k+n}(c) - z_k(c) = 0 (pre-periodic Misiurewicz point,
    preperiod k, eventual period n). Returns a NewtonResult."""
    c = mp.mpc(c0)
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    residual = mp.inf
    it = 0
    for it in range(1, max_steps + 1):
        zk, dk, zkn, dkn = _orbit_at(c, preperiod, period, degree)
        g = zkn - zk
        gp = dkn - dk
        residual = abs(g)
        if gp == 0:
            break
        step = g / gp
        c = c - step
        if abs(step) < tol and residual < tol:
            break
    zk, _, zkn, _ = _orbit_at(c, preperiod, period, degree)
    residual = abs(zkn - zk)
    conv = residual < tol
    return NewtonResult(c=c, converged=bool(conv), iters=it,
                        residual=float(mp.log10(residual)) if residual > 0 else -999.0,
                        kind="misiurewicz", period=period, preperiod=preperiod, degree=degree)


def is_minimal_misiurewicz(c, preperiod, period, *, degree=2, tol_dps_margin=6):
    """A Misiurewicz solution is *minimal* (genuinely preperiod-k / period-n)
    only if the orbit is not already periodic one step earlier and the eventual
    period does not divide to something smaller. Cheap sanity screen so `scan`
    reports the minimal (k,n), not a multiple."""
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    # Not already satisfied at preperiod k-1 (would mean true preperiod < k).
    if preperiod >= 1:
        zk1, _, zkn1, _ = _orbit_at(c, preperiod - 1, period, degree)
        if abs(zkn1 - zk1) < tol:
            return False
    # Eventual period is minimal: no proper divisor q|n also closes.
    for q in range(1, period):
        if period % q == 0:
            zk, _, zkq, _ = _orbit_at(c, preperiod, q, degree)
            if abs(zkq - zk) < tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Minibrot size estimate (Munafo / Kalles-Fraktaler). |size| ~ atom radius,
# arg(size)/2 ~ orientation. Used to suggest an fw band for a nucleus.
# ---------------------------------------------------------------------------
def nucleus_size_estimate(c, period, degree=2):
    """Return a complex size estimate for the period-p minibrot at nucleus c.

    Degree-2 is the Munafo/Kalles-Fraktaler formula (`size = 1/(b·λ²)`), kept
    textually untouched. For d≥3 the derivative factor generalizes to
    f'(z)=d·z^{d-1} (so `λ` = Π f'(z_k) is the true orbit derivative and `b` its
    second-order correction), AND the size-law exponent on λ changes from 2 to
    **d/(d-1)** — the multibrot renormalization scaling (the p-fold iterate near a
    period-p nucleus is a small w→w^d+c copy whose linear scale goes as
    |λ|^{-d/(d-1)}, not |λ|^{-2}). d/(d-1) reduces to 2 at d=2, so the two branches
    agree there. This matters: the flat degree-2 `λ²` law under-estimates the d≥3
    atom by ~4-11×, putting a 4·|size| frame *inside* the minibrot body (all-black
    field). Validated against rendered interior-fraction: with d/(d-1), a 4·|size|
    nucleus-centred frame lands at interior-frac ≈0.2-0.5 (comparable to d2's ≈0.16),
    i.e. the minibrot frames as an island ringed by decorations — what the screen
    and the eye need. |size| is exact-in-magnitude (`|λ^e| = |λ|^e`, branch-free);
    arg(size) is not consumed here. See docs/design/q4_multibrot_transfer.md.
    """
    l = mp.mpc(1)
    b = mp.mpc(1)
    z = mp.mpc(0)
    if degree == 2:
        for _ in range(1, period):
            z = z * z + c
            l = 2 * z * l
            if l == 0:
                return mp.mpc(0)
            b = b + 1 / l
        denom = b * l * l                         # degree-2 λ² law (untouched)
    else:
        for _ in range(1, period):
            z = z ** degree + c
            l = degree * z ** (degree - 1) * l
            if l == 0:
                return mp.mpc(0)
            b = b + 1 / l
        denom = b * l ** (mp.mpf(degree) / (degree - 1))   # multibrot d/(d-1) law
    if denom == 0:
        return mp.mpc(0)
    return 1 / denom


# ---------------------------------------------------------------------------
# Symmetry-aware nucleus dedup key. z^d+c has (d−1)-fold rotational symmetry about
# the origin: c and c·ω^k (ω = exp(2πi/(d−1)), k=0..d−2) are the SAME atom under the
# conjugacy z→ωz — same period, same |size|, rotated field. Rounded-coordinate dedup
# alone lets those rotational copies survive as separate "minibrots" (this manufactured
# d4=10/12, d5=8/12 distinct in the multibrot-transfer read). Canonicalize first —
# rotate c into the fundamental sector arg c ∈ [0, 2π/(d−1)) by unwinding whole sectors
# (each a symmetry rotation, so the atom is unchanged) — THEN round and dedup as before.
# d=2 is 1-fold (sector = whole plane) → identity, so the degree-2 key is byte-identical.
# ---------------------------------------------------------------------------
def canonical_nucleus_c(c, degree):
    """Rotate nucleus `c` into the fundamental sector arg c ∈ [0, 2π/(d−1)) of the
    z^d+c rotational symmetry. d=2 (and c=0) return `c` unchanged."""
    c = mp.mpc(c)
    if degree <= 2 or c == 0:
        return c
    sector = 2 * mp.pi / (degree - 1)
    m = mp.floor(mp.arg(c) / sector)          # whole sectors to unwind; arg∈(−π,π]
    ang = -sector * m                         # a symmetry rotation (multiple of 2π/(d−1))
    return c * mp.mpc(mp.cos(ang), mp.sin(ang))


def nucleus_dedup_key(c, degree, dps):
    """Symmetry-canonical rounded-coordinate dedup key for a degree-d nucleus at `c`.
    Collapses the (d−1) rotational copies of one atom to a single key. At d=2 this is
    exactly the pre-existing `(nstr(cx, dps), nstr(cy, dps))` key.

    Runs at working precision ≥ dps+15 so the `dps`-digit rounding is meaningful even
    if the ambient `mp.mp.dps` is low — but the CALLER must have parsed `c` at adequate
    precision (a string parsed at dps=15 has already lost its tail before it gets here)."""
    with mp.workdps(max(mp.mp.dps, dps + 15)):
        cc = canonical_nucleus_c(c, degree)
        return (mp.nstr(cc.real, dps), mp.nstr(cc.imag, dps))


# ---------------------------------------------------------------------------
# Atom instrument `A` — size, orientation, and required precision, from the same
# recursion Newton already runs. With the nucleus c0 and period n:
#
#     z_{k+1}  = z_k^d + c
#     z'_{k+1} = d·z_k^(d−1)·z'_k + 1              z'_0 = 0     (P_n'(c0) = z'_n)
#     Lambda   = Π_{k=1..n-1} d·z_k^(d−1)                       (reduced multiplier)
#     A        = Lambda^(1/(d−1)) · P_n'(c0)
#
# Locally the p-fold map conjugates to w^d + C, and the embedded copy is the whole
# multibrot pulled back by δ = C/A. So |A| is the atom's inverse linear scale and
# arg A its orientation: default window scale ≈ 1/|A|, rotation ≈ −arg A.
#
# EXACT IDENTITY with `nucleus_size_estimate`. Both are the *same analytic
# quantity*: |A| ≡ 1/|size| and arg A ≡ −arg(size) to full precision at every n
# (verified over the d2 + d3/d4/d5 nuclei, and locked in
# `test_deep_center_finder_degree.py`). This is not a coincidence — the size code's
# `b`-sum times `Lambda` equals `P_n'(c0)` algebraically, and its `Lambda^{d/(d-1)}`
# denominator is `Lambda^{1/(d-1)}·Lambda`. So `A` is NOT a second, independent
# estimate to cross-check `size` against; it is the size law re-derived from the
# c-derivative, and it *independently confirms* the `d/(d−1)` exponent (a flat
# `λ²` law disagrees with `A` by |λ|^{(d−2)/(d−1)} at d≥3 — the measured 4–11×).
# `A` is kept as the primary export because it hands three things `size` does not:
# a principal-branch orientation, a required-precision figure, and an a-priori f64
# pixel-spacing-wall predictor. See docs/design/atom_instrument.md.
#
# The (d−1)-th root leaves arg A determined only mod 2π/(d−1): an orientation
# ambiguity (which of the d−1 rotational copies), not an error. We record the
# mpmath principal branch and the ambiguity spacing alongside it.
# ---------------------------------------------------------------------------
@dataclass
class AtomInstrument:
    degree: int
    period: int
    A: object                    # mpmath.mpc — the atom scaling factor Λ^(1/(d-1))·P_n'
    abs_A: float
    arg_A: float                 # principal branch (radians); determined mod 2π/(d-1)
    log10_abs_A: float
    window_scale: float          # ≈ 1/|A| — frame width that frames the whole atom
    rotation_rad: float          # ≈ −arg A (principal branch)
    rotation_ambiguity_rad: float  # 2π/(d-1): (d-1)-th-root branch spacing (0 at d=2)
    required_dps: int            # mpmath dps to localize a ~1/|A| frame, incl. guard

    def f64_wall_margin_decades(self, width, *, ss=1, spacing_floor=1e-13, k=4.0):
        """Headroom (in decades of |A|) before a default `k·window_scale` frame
        crosses the f64 pixel-spacing wall at render `width`×`ss`. Pixel spacing =
        k/(|A|·width·ss); the wall is `spacing_floor` (Rust `PERTURB_SPACING`=1e-13,
        below which `F64Backend` quantizes — and multibrot has NO perturbation
        fallback, so this wall is absolute there). Positive = safe; **negative
        predicts an f64 render failure a priori**, no render attempt needed."""
        import math
        wall_log10 = math.log10(k) - math.log10(spacing_floor) - math.log10(width * ss)
        return wall_log10 - self.log10_abs_A


def atom_instrument(c, period, degree=2, *, guard_digits=15) -> AtomInstrument:
    """Compute the atom instrument `A` for a period-`period` nucleus at `c`.

    Self-contained (one orbit pass; does not call `nucleus_size_estimate`) so the
    `|A| ≡ 1/|size|` identity is a genuine cross-check rather than a tautology."""
    c = mp.mpc(c)
    z = mp.mpc(0)
    zp = mp.mpc(0)                    # z'_k = dz_k/dc
    lam = mp.mpc(1)                   # Λ = Π_{k=1..n-1} d·z_k^(d-1)
    for k in range(1, period + 1):
        if degree == 2:
            zp = 2 * z * zp + 1
            z = z * z + c
        else:
            zpm1 = z ** (degree - 1)
            zp = degree * zpm1 * zp + 1
            z = zpm1 * z + c
        if k <= period - 1:          # multiplier excludes the k=0 critical point (z_0=0)
            lam = lam * (degree * z ** (degree - 1))
    A = lam ** (mp.mpf(1) / (degree - 1)) * zp
    abs_A = float(abs(A)) if A != 0 else 0.0
    log10_abs_A = float(mp.log10(abs(A))) if A != 0 else float("inf")
    arg_A = float(mp.arg(A))
    import math
    req_dps = max(50, int(math.ceil(log10_abs_A)) + guard_digits) if abs_A > 0 else 50
    return AtomInstrument(
        degree=degree, period=period, A=A,
        abs_A=abs_A, arg_A=arg_A, log10_abs_A=log10_abs_A,
        window_scale=(1.0 / abs_A if abs_A > 0 else float("inf")),
        rotation_rad=-arg_A,
        rotation_ambiguity_rad=(2.0 * math.pi / (degree - 1) if degree > 2 else 0.0),
        required_dps=req_dps,
    )


# ---------------------------------------------------------------------------
# Emission — a NewtonResult -> render-ready deep center (decimal strings).
# ---------------------------------------------------------------------------
@dataclass
class DeepCenter:
    kind: str                 # "nucleus" | "misiurewicz"
    period: int
    preperiod: int
    cx: str                   # decimal-string center (render-tier native)
    cy: str
    fw_suggest: str           # a single suggested frame width for a first look
    fw_band: list             # [hi, lo] suggested band (decimal strings)
    self_similar: bool        # Misiurewicz => structure holds across all depths
    size_estimate: Optional[str]   # |minibrot size| (nucleus only), else None
    newton_converged: bool
    newton_iters: int
    newton_residual_log10: float
    render_maxiter: int       # a sensible maxiter for fw_suggest
    degree: int = 2           # multibrot power d in z^d + c (2 = Mandelbrot)

    def render_cmd(self, exe="target/release/fractal-generator.exe",
                   width=1024, ss=2, out="scratch/deep_centers/preview.png"):
        # d=2 keeps the multi-palette `sheet` preview (sheet is degree-2 only).
        # d≥3 (multibrot, parameter plane) has no sheet path → render-one with
        # `--family multibrot{d}`; single palette, one PNG.
        if self.degree == 2:
            return [exe, "sheet", "--builtins", "default cubehelix viridis",
                    "--center-re", self.cx, "--center-im", self.cy,
                    "--frame-width", self.fw_suggest, "--maxiter", str(self.render_maxiter),
                    "--tile-width", str(width), "--aspect", "16:9", "--supersample", str(ss),
                    "--backend", "auto", "--output", out]
        return [exe, "render-one", "--family", f"multibrot{self.degree}",
                "--cx", self.cx, "--cy", self.cy,
                "--fw", self.fw_suggest, "--maxiter", str(self.render_maxiter),
                "--width", str(width), "--aspect", "16:9", "--supersample", str(ss),
                "--output", out]


def _maxiter_for_fw(fw: float) -> int:
    """Scale maxiter with depth (probe: fw 1e-20 wanted ~30k; shallow ~3k)."""
    import math
    d = -math.log10(fw) if fw > 0 else 3
    # ~1500 iters per decade of depth, floored at 3000, capped at 40000
    # (matches the probe ladder: fw 1e-20 -> ~30k).
    return int(max(3000, min(40000, round(1500 * d))))


def make_deep_center(res: NewtonResult, *, fw_suggest=None, emit_fw_floor=1e-20) -> DeepCenter:
    """Turn a converged NewtonResult into a render-ready DeepCenter with a
    suggested fw band and enough serialized digits for the deepest fw."""
    digits = emit_digits_for_fw(emit_fw_floor)
    cx = mp.nstr(res.c.real, digits, strip_zeros=False)
    cy = mp.nstr(res.c.imag, digits, strip_zeros=False)

    if res.kind == "misiurewicz":
        # Self-similar: structure holds at every scale. Suggest a mid-band first
        # look; band spans down to the proven perturbation depth.
        fw_hi = 1e-3
        fw_lo = emit_fw_floor
        fw0 = fw_suggest if fw_suggest is not None else 1e-9
        size_s = None
        self_sim = True
    else:
        size = nucleus_size_estimate(res.c, res.period, res.degree)
        size_abs = float(abs(size)) if size != 0 else 0.0
        size_s = f"{size_abs:.6e}"
        # A nucleus sits in the minibrot's *interior* (black). Centered on it, the
        # money shot frames the whole minibrot as a small island ringed by its
        # radial spiral decorations — empirically fw ~ 4x size (validated: fw=size
        # is mostly interior black; fw < size on-nucleus is pure black). So the
        # compositionally-valid band for a nucleus-CENTERED frame is roughly
        # [~40x size (lots of context) .. ~2x size (minibrot fills frame)]. Going
        # deeper on-structure needs OFFSETTING onto a decoration, not the nucleus.
        fw_hi = size_abs * 40 if size_abs > 0 else 1e-3
        fw_lo = size_abs * 2 if size_abs > 0 else emit_fw_floor
        fw0 = fw_suggest if fw_suggest is not None else (size_abs * 4 if size_abs > 0 else 1e-6)
        self_sim = False

    return DeepCenter(
        kind=res.kind, period=res.period, preperiod=res.preperiod,
        cx=cx, cy=cy,
        fw_suggest=f"{fw0:.6e}",
        fw_band=[f"{fw_hi:.6e}", f"{fw_lo:.6e}"],
        self_similar=self_sim,
        size_estimate=size_s,
        newton_converged=res.converged,
        newton_iters=res.iters,
        newton_residual_log10=res.newton_residual_log10 if hasattr(res, "newton_residual_log10") else res.residual,
        render_maxiter=_maxiter_for_fw(fw0),
        degree=res.degree,
    )


# ---------------------------------------------------------------------------
# Scan — identify what an f64 seed converges to (period / preperiod).
# ---------------------------------------------------------------------------
def scan(seed, *, max_period=24, max_preperiod=12, do_nucleus=True, do_misiurewicz=True,
         near=1e-3):
    """From an f64 seed, try nucleus periods and Misiurewicz (k,n) combos; report
    the ones that converge within `near` of the seed, minimal ones first.

    `near` sets the use: a tight value (~1e-9) *identifies* an already-precise
    coordinate's type (does it Newton straight back to itself?); a looser value
    (~1e-2) *explores* which roots a rough seed's basin reaches."""
    hits = []
    c0 = mp.mpc(seed[0], seed[1])
    near = mp.mpf(str(near))
    if do_nucleus:
        for p in range(1, max_period + 1):
            r = newton_nucleus(c0, p)
            if r.converged and abs(r.c - c0) < near:
                # Minimal period only (skip p that's a multiple of a smaller hit).
                z, _ = _orbit(r.c, p)
                minimal = all(not (p % q == 0 and abs(_orbit(r.c, q)[0]) <
                                   mp.mpf(10) ** (-(mp.mp.dps - 6)))
                              for q in range(1, p))
                hits.append(("nucleus", p, 0, r, minimal, float(abs(r.c - c0))))
    if do_misiurewicz:
        for k in range(1, max_preperiod + 1):
            for n in range(1, max_period + 1):
                r = newton_misiurewicz(c0, k, n)
                if r.converged and abs(r.c - c0) < near:
                    minimal = is_minimal_misiurewicz(r.c, k, n)
                    hits.append(("misiurewicz", n, k, r, minimal, float(abs(r.c - c0))))
    # Rank: minimal first, then closest to seed, then smallest (k+n).
    hits.sort(key=lambda h: (not h[4], h[5], h[2] + h[1]))
    return hits


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _print_center(dc: DeepCenter, *, as_json=False):
    if as_json:
        print(json.dumps(asdict(dc)))
        return
    tag = f"{dc.kind} period={dc.period}" + (f" preperiod={dc.preperiod}" if dc.preperiod else "")
    print(f"# {tag}  |  Newton: converged={dc.newton_converged} "
          f"iters={dc.newton_iters} log10|res|={dc.newton_residual_log10:.1f}")
    if dc.size_estimate:
        print(f"# minibrot size estimate ~ {dc.size_estimate}")
    if dc.self_similar:
        print("# self-similar (Misiurewicz): structure holds across all depths")
    print(f"cx = {dc.cx}")
    print(f"cy = {dc.cy}")
    print(f"fw_suggest = {dc.fw_suggest}   band = [{dc.fw_band[0]} .. {dc.fw_band[1]}]")
    print(f"render_maxiter = {dc.render_maxiter}")
    print("render:  " + " ".join(
        (f'"{a}"' if " " in a else a) for a in dc.render_cmd()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--seed", nargs=2, type=float, required=True,
                       metavar=("RE", "IM"), help="rough f64 seed near ∂M")
        p.add_argument("--fw-floor", type=float, default=1e-20,
                       help="deepest fw to serialize digits for (default 1e-20)")
        p.add_argument("--fw-suggest", type=float, default=None,
                       help="override the suggested first-look frame width")
        p.add_argument("--json", action="store_true", help="emit JSON")

    pn = sub.add_parser("nucleus", help="Newton to a period-p nucleus")
    add_common(pn)
    pn.add_argument("--period", type=int, required=True)

    pm = sub.add_parser("misiurewicz", help="Newton to a Misiurewicz point")
    add_common(pm)
    pm.add_argument("--preperiod", type=int, required=True)
    pm.add_argument("--period", type=int, required=True)

    ps = sub.add_parser("scan", help="identify what an f64 seed converges to")
    ps.add_argument("--seed", nargs=2, type=float, required=True, metavar=("RE", "IM"))
    ps.add_argument("--max-period", type=int, default=24)
    ps.add_argument("--max-preperiod", type=int, default=12)
    ps.add_argument("--nucleus-only", action="store_true")
    ps.add_argument("--misiurewicz-only", action="store_true")
    ps.add_argument("--near", type=float, default=1e-3,
                    help="max |c-seed| to accept (tight ~1e-9 identifies a precise "
                         "coordinate; loose ~1e-2 explores a rough seed's basins)")
    ps.add_argument("--top", type=int, default=12)

    args = ap.parse_args(argv)

    if args.cmd == "scan":
        mp.mp.dps = 60
        hits = scan(tuple(args.seed),
                    max_period=args.max_period, max_preperiod=args.max_preperiod,
                    do_nucleus=not args.misiurewicz_only,
                    do_misiurewicz=not args.nucleus_only, near=args.near)
        if not hits:
            print("no convergent roots near the seed — widen --max-period/--max-preperiod "
                  "or check the seed is near ∂M", file=sys.stderr)
            return 1
        print(f"# {len(hits)} convergent root(s) near seed "
              f"({args.seed[0]}, {args.seed[1]}) — minimal first:")
        for kind, per, pre, r, minimal, dist in hits[:args.top]:
            tag = f"{kind:12s} period={per:<3d}" + (f" preperiod={pre:<3d}" if kind == "misiurewicz" else "          ")
            print(f"  {tag}  minimal={minimal!s:5s}  |c-seed|={dist:.2e}  "
                  f"iters={r.iters:<3d} log10|res|={r.residual:.1f}")
        return 0

    mp.mp.dps = max(dps_for_fw(args.fw_floor), dps_for_fw(args.fw_suggest or args.fw_floor))
    if args.cmd == "nucleus":
        r = newton_nucleus(mp.mpc(args.seed[0], args.seed[1]), args.period)
    else:
        r = newton_misiurewicz(mp.mpc(args.seed[0], args.seed[1]), args.preperiod, args.period)

    if not r.converged:
        print(f"# NOT CONVERGED (log10|res|={r.residual:.1f} after {r.iters} iters) — "
              f"seed may be too far, or wrong period/preperiod", file=sys.stderr)
    # Attach residual under the name make_deep_center expects.
    r.newton_residual_log10 = r.residual
    dc = make_deep_center(r, fw_suggest=args.fw_suggest, emit_fw_floor=args.fw_floor)
    _print_center(dc, as_json=args.json)
    return 0 if r.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
