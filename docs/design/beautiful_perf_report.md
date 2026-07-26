# Why the `beautiful` field dump is ~35× slower than the `f64` twin

**TL;DR.** For a `--dump-field` of the **smooth** field, `--dump-field-source
beautiful` (the default) measured **39.2 s** vs `--dump-field-source f64` at **1.1 s**
on the same 2176×1224 ss1 frame (degree-5 multibrot, maxiter ~13 k) — ~35×. It is
**not** a threading gap: both kernels are `rayon` `into_par_iter` over rows. The cost
is the per-pixel kernel. `iterate_orbit` (beautiful) is **field-agnostic** — it
accumulates *every* coloring field on *every* iteration (≈4 transcendentals/iter) and
then keeps a single scalar; `F64Backend::sample` (f64 twin) is a lean escape loop with
~0 transcendentals/iter. Neither pathway can be deleted, but the smooth-beautiful path
is redundant for offset-invariant statistics and has an obvious in-kernel speedup.

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

## The obvious in-kernel speedup (optional, larger)

`iterate_orbit` computes every accumulator regardless of `params.field`. A **field-gated
kernel** — compute only what the requested field(s) need — would bring *beautiful-smooth*
to ≈`f64` speed **while staying byte-identical**, and would speed up *all* single-field
beautiful renders (not just dumps), since `single_field_supersampled` and
`render_beautiful_single` reduce to one field too.

- Smooth needs **none** of exp/stripe/tia/curvature/gaussian/traps/velocity/derivative
  — just the escape `n` and final `|z|²`. Gating those off is the whole win.
- Shape: pass the requested field set (or a `const`-generic / bitset) into
  `iterate_orbit` and guard each accumulator block. The composite (`direct_trap`) and
  Color-By paths request several fields at once, so the gate must honor the **union** of
  what the caller will reduce — not hardcode "smooth only."
- Risk: modest and mechanical (per-field dependency sets are local to each accumulator
  block); the existing render tests (separability, sheet) pin byte-identity and would
  catch a mis-gate. Not attempted here — flagged for a future pass.

## Recommendations

1. **Done.** Route offset-invariant smooth-statistic consumers to
   `--dump-field-source f64`. Applied in `tools/studies/q4_multibrot_transfer.py`;
   warnings added at `src/render_one.rs` (dump-field site) and
   `src/render_modes.rs` (`single_field_supersampled`).
2. **Optional, higher value.** Field-gate `iterate_orbit` so beautiful-smooth ≈ f64
   speed with byte-identity — benefits every single-field beautiful render.
3. **Keep `beautiful`** for byte-identical smooth reproduction and all non-smooth fields.
