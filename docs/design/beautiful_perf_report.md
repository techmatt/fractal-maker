# Why the `beautiful` field dump was ~35× slower than the `f64` twin (now field-gated)

**TL;DR.** For a `--dump-field` of the **smooth** field, the default
`--dump-field-source beautiful` originally measured **39.2 s** vs `f64` at **1.1 s**
on a 2176×1224 ss1 frame — ~35×. It is **not** a threading gap: both kernels are
`rayon` `into_par_iter` over rows. The cost was the per-pixel kernel: `iterate_orbit`
(beautiful) is **field-agnostic** — it accumulated *every* coloring field on *every*
iteration (≈4 transcendentals/iter) and then kept a single scalar; `F64Backend::sample`
(f64 twin) is a lean escape loop with ~0 transcendentals/iter.

**Implemented fix (this change): field-gating.** `iterate_orbit_needs` takes a
`FieldNeeds` flag set and skips every accumulator the requested field(s) don't read.
On the identical d2 shallow nucleus (maxiter 3000) this took the beautiful smooth dump
from **39.2 s → 2.24 s (~17.5×)**, landing within ~4× of the f64 backend (0.5 s). The
residual gap is the general kernel's inherent overhead (per-iteration `match family`,
`ColoringParams` indirection, `cpow_deriv`, always-on `zabs`/`zprev`), not wasted field
work. Parity is proven bit-for-bit: `render_modes::tests::field_gating_matches_ungated`
asserts every field's reduced value (and `ushade`) is bit-identical gated vs ungated.
Neither pathway is deleted; the `f64` source stays strictly best for offset-invariant
smooth statistics (~0.5 s, no general-kernel overhead).

This report was written after the q4 multibrot-transfer read discovered the gap while
dumping ~80 fields for the stage-1 screen; switching those to `f64` cut the render
stage from a projected ~50 min to ~3 min.

## Measurement

Identical frame, only `--dump-field-source` differs (`tools/studies/q4_multibrot_transfer.py`
timing probe, one d5 nucleus, W=2176 H=1224 ss1):

| source | wall time | interior mask | featurize features | `_v2_drop` |
|---|---|---|---|---|
| `beautiful` (default) | **39.2 s** | — | reference | reference |
| `f64` | **1.1 s** | identical | agree to **1.3e-9** | identical |

The two smooth fields are numerically interchangeable for the screen (see "Why they're
interchangeable" below); the only difference is speed.

## Root cause — the kernel, not the threads

Both entry points parallelize identically over supersampled rows:

- `render_modes::single_field_supersampled` (beautiful) — `src/render_modes.rs:2055`, `(0..sub_h).into_par_iter()`
- `render_modes::smooth_field_f64_supersampled` (f64) — `src/render_modes.rs:2155`, `into_par_iter()`

So a fixed core count is applied to both. The difference is entirely per-pixel work.

### `iterate_orbit` (beautiful) computes *all* fields every iteration

`src/render_modes.rs` (the loop from ~`:1506`). `iterate_orbit` does not know which
field the caller wants — `single_field_supersampled` reduces to `params.field` only
*after* the orbit returns (`:2069`). So the loop unconditionally accumulates, **per
iteration**:

- `exp(-|z|)` — exponential smoothing accumulator (`:1581`), **every iteration**
- `0.5 + 0.5·sin(s·arg z)` — stripe, with `z.im.atan2(z.re)` inside (`:1593`), n ≥ skip
- tia lo/hi band (`:1598`), n ≥ skip
- `arg((zₙ−zₙ₋₁)/(zₙ₋₁−zₙ₋₂))` — curvature, another `atan2` (`:1616`), n ≥ 2
- Gaussian-integer lattice trap: `z.re.round()`, `z.im.round()`, `(z−q).norm()`, plus
  running min/max/total/count (`:1565`)
- circle trap `|‖z‖−r|` and cross trap `min(|re|,|im|)` (`:1554`)
- discrete velocity `|zₙ₊₁−zₙ|` = a `norm()` (`:1587`)
- the derivative `dz` recurrence (`:1534`)

That is **≈4 transcendentals per iteration** (`exp`, `sin`, and two `atan2`) plus two
`round`s and several `norm`/`sqrt`s — and for a *smooth* dump, **all of it is thrown
away** except `acc.smooth`. At millions of pixels × thousands of iterations, the
discarded transcendentals dominate.

### `F64Backend::sample_flags` (f64 twin) is a lean escape loop

`src/backend.rs:245`. Per iteration: `z = z² + c` (a few mults), `zmag2` compare, and
the orbit-trap `eval_dist`; the trap-phase `atan2` runs **only on a trap-min
improvement** under the production `PHASE_GATED` strategy (`:301`), i.e. O(log n) times
over an orbit, not once per iteration. DE is a compile-time flag. So the common case is
**~0 transcendentals per iteration** — which is the ~35× we see.

## Why the two smooth fields are interchangeable (for statistics)

Already documented at `smooth_field_f64_supersampled`'s docstring
(`src/render_modes.rs:2095`): the backend's smooth value is un-normalized
`(n+1) − ln(ln|z|)/ln d` while beautiful carries the bailout normalization, so the two
differ by the **constant** `ln(ln B)/ln d`. The escape mask is bailout-driven, so the
NaN-interior seam is identical. Any consumer whose statistic is invariant to a constant
offset therefore reads the same field:

- the degenerate-outcome guard (`interior_frac` = NaN fraction, `field_std`);
- the **entire q4 stage-1 screen** — `LF.featurize` percentile-stretches every crop
  (`(v−lo)/span`), which is *exactly* offset-invariant, so `g_interior/g_flat/
  g_speckle`, the cell-dispersion features, and `_v2_drop` are unchanged (verified to
  1.3e-9; `_v2_drop` agrees on every probe).

It is **not** byte-identical, so it must not feed the field⊗colormap reproduction path.

## Can `beautiful` be deprecated? No.

It is load-bearing for two things the f64 twin cannot provide:

1. **Non-smooth fields.** `tia`, `stripe`, `curvature`, `trap_circle`, `gaussian_int`,
   `de`, … have no fast escape-time twin; `--dump-field-source f64` explicitly errors
   for any non-smooth field (`src/render_one.rs:301`).
2. **Byte-identical smooth reproduction.** The field⊗colormap split (dump a field,
   colorize in Python) needs the exact beautiful value, not an offset one.

So the recommendation is not deprecation but **narrowing**: any smooth consumer that
only needs offset-invariant statistics should pass `--dump-field-source f64`.

## The in-kernel speedup — implemented

`iterate_orbit` used to compute every accumulator regardless of `params.field`. It now
delegates to **`iterate_orbit_needs(.., needs: FieldNeeds)`**, which guards each
accumulator block behind a flag; `iterate_orbit` is a thin wrapper passing
`FieldNeeds::all()` (byte-identical catch-all for tests / unknown callers).

- `FieldNeeds::for_field(F)` sets exactly the flags field `F` reduces from: Smooth /
  Decomposition need **none** (escape + final `z`); Stripe/Tia/Curvature/TrapCircle/
  TrapCross/Velocity/GaussianInt/ExpSmoothing each set one; `De` sets `deriv` (the `dz`
  recurrence). `.with_deriv(want_shade)` ORs in `dz` for the `normal_map` emboss
  (`ushade = z/dz`). `DirectTrap` (colour-valued) falls to `all()`.
- The four hot callers pass the right needs: `smooth_field_supersampled` →
  `for_field(Smooth)` (none); `single_field_supersampled` → `for_field(the_field)`;
  `render_beautiful_single` → `for_field(field).with_deriv(want_shade)`;
  `render_beautiful_composite` → `for_field(base).union(for_field(tex)).with_deriv(..)`.
- The flag set is constant for a whole render, so the per-iteration branch is perfectly
  predicted (≈free); when a field IS requested its block runs verbatim, so the reduced
  value is **byte-identical** — guarded by `field_gating_matches_ungated` (every field's
  reduced value + `ushade` bit-identical gated vs ungated) plus the existing separability
  / sheet / montage guards (all green).
- Always-on essentials kept (cheap, no transcendentals): the escape/smooth logic,
  `zn_sq`, `zabs`, and the `zprev` history shifts. Gating those too would shave the
  residual ~4× gap to `f64` but risks the byte-pinned paths for little gain.

## Recommendations

1. **Done.** Route offset-invariant smooth-statistic consumers to
   `--dump-field-source f64` (still strictly fastest, ~0.5 s). Applied in
   `tools/studies/q4_multibrot_transfer.py`.
2. **Done.** Field-gated `iterate_orbit` → beautiful-smooth 39.2 s → 2.24 s (~17.5×),
   byte-identical. Benefits every single-field beautiful render, not just dumps.
3. **Keep `beautiful`** for byte-identical smooth reproduction and all non-smooth fields.
