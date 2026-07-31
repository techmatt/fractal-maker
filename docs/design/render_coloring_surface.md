# Render / coloring path surface & port-parity invariants

Distilled from `render_config_report` and `wire_parity_gates`. Governs the coloring
path that turns a field into an image, the disjoint palette namespaces, the
cross-language OKLab ports that must **not** be refactored together, and (§6) the Ultra
Fractal originals that four `render_modes` fields were ported against — folded in from the
standalone `uf_coloring_algorithms.md` on 2026-07-31, since that reference had no owner of
its own and its technical body is carried by `src/render_modes.rs`.

## 1. Two coloring algorithms, visibly different at their defaults

The engine has **two** distinct coloring algorithms that produce visibly different
images from the same field + palette:

- **location-profile** (`coloring::shade`) — raw smooth-iteration × density, cycled →
  banded UF look. Used by bare `render` / `sheet` / `render-one`-default.
- **beautiful** (`render_modes` / `tools/colormap.py`) — percentile-stretch → single
  gradient pass → smooth ramp. Used by `render-one --coloring` and **all dump-field
  recolors**.

Nothing fails loudly on a mismatch — you get a **differently-colored image, not an
error**.

## 2. Density forks silently by entry point

location-profile density defaults **fork** by entry point: **0.025** (`ShadeArgs`, bare
render / sheet) vs **0.004** (`generate::color_params`, all corpus paths incl.
`render-one`) — a ~6× banding difference. Same field, same palette, different look.

## 3. Three disjoint palette namespaces

1. Rust built-ins (`default`, `cubehelix`, `viridis`) — `default` / `cubehelix` exist
   **only** here.
2. `clean_colormaps.json`.
3. `score3_colormaps.json`.

A name resolves in one namespace only; they do not share entries.

## 4. The open corpus-coloring hazard (P4 — unresolved)

If **label/training crops are colored through one algorithm** while the human judges or
the deploy path renders **the other**, the classifier learns a look the pipeline won't
reproduce. **Every corpus batch should be audited for coloring-used vs coloring-judged
vs coloring-deployed agreement.** This is the single load-bearing open item on this
surface.

## 5. The OKLab ports are a deliberate frozen port — do NOT refactor to dedup

Four hand-synced OKLab/coloring round-trip copies exist:

| copy | contract | gate |
|---|---|---|
| `src/palette.rs` | canonical | — |
| `palette_lib/coloring.py` | **byte-exact to Rust** (<1e-12) | `check_bytematch` |
| `tools/colormap.py` | ≤1-LSB post-render | `colormap_acceptance` (TOL_MAX=2) |
| `tools/palettes/color.py` | port | — |

They are held to **intentionally different contracts** — merging onto one bake would
either over-tighten the deploy tail or loosen the byte-exact extractor. Since LUT
byte-exactness feeds **every emitted image**, collapsing trades a guaranteed-safe state
for cosmetic dedup on the one axis where a silent 1-ULP drift is a **whole-corpus
regression**: negative expected value. The drift risk that once argued for collapsing
is now removed by the pytest byte-identity gates (proven red on a 1-ULP perturbation) —
enforce the invariant cheaply and **leave the bytes alone**.

## 6. The Ultra Fractal originals — the contract four `render_modes` fields were ported against

Four `Field` variants in `src/render_modes.rs` are reconstructions of Ultra Fractal
`Standard.ucl` colorings: `GaussianInt`, `ExpSmoothing`, `Decomposition`, `DirectTrap`
(each with a `specs/*.json`). **The per-mode math, the reduction tables and each mode's
gotcha live in the code's own doc comments** — `Field`, `GaussianColorBy`, `render_direct_trap`
— per the convention that module docs carry the rationale. Folded in here is only what the
code cannot state about itself: where the algorithms came from, the execution model they are
specified in, and where our port deliberately stops short.

**Provenance, and why the port is clean.** Behaviour was reconstructed from the Ultra Fractal
manual (Gaussian Integer, Exponential Smoothing, Direct Orbit Traps, Decomposition, Binary
Decomposition, Orbit Traps, "Writing coloring algorithms") and paraphrased into
framework-neutral form — **not copied from `Standard.ucl`** (Mitchell / Jones / Slijkerman).
That distinction is the reason these ship at all; preserve it if a fifth is added.

**The UF execution model, which is where the vocabulary comes from.** A UF coloring runs in
three sections — **init** (once per pixel), **loop** (once per iteration, *after* the
formula's step, with the current `z` visible), **final** (once, after bail-out) — and emits
one of:

- **`#index`** — a float gradient position. **The gradient is periodic over [0,1]**, so
  out-of-range values *wrap* rather than clamp. This is why iteration-keyed UF modes carry a
  `0.01 ×` scale (they band the gradient every 100 iterations) and why our `GaussianColorBy`
  iteration modes fold mod 1.
- **`#color`** — a direct RGBA value, replacing the index lookup entirely.
- **`#solid`** — flags the pixel to take a flat colour.

The split that matters structurally: **orbit-monitoring** algorithms build their answer from
the *whole* orbit (Gaussian Integer, Exponential Smoothing, Direct Orbit Traps) and so need
the loop section; **final-only** algorithms read just the last `z` (Decomposition). All four
are "outside" colorings by default; Gaussian Integer and Exponential Smoothing are also
usable inside.

**`DirectTrap` is the one `#color` algorithm, and that is why it is not a scalar field.**
Rather than reducing the orbit to one index and looking the gradient up once, it samples the
gradient *every iteration the orbit lands inside the trap* and alpha-composites the samples —
many semi-transparent layers in one coloring pass, which is where the lacy overlapping look
comes from. Consequently `OrbitAccum::field` returns `None` for it and it is routed to the
parallel colour-valued path. Two consequences for this surface: editing the gradient
recolours the whole structure (the samples come *from* the gradient), and the composite is
order-dependent, so it is deterministic only because the iteration loop is.

**Where our port stops short of UF — deliberate, and the reason a spec looks thin.**

| UF offers | we implement |
|---|---|
| Gaussian Integer normalization `N` ∈ {none, pixel, factor, f(z)} | `N = 1` only (unit integer lattice) |
| Gaussian Integer integer type ∈ {round, trunc, floor, ceil} | `round` (the smoothest, the canonical look) |
| Gaussian Integer *Randomize* (logistic-map speckle) | not ported |
| Exponential Smoothing convergent branch (`Σ exp(−1/|zₙ₋₁−zₙ|)`) | divergent branch only — Mandelbrot/Julia never converge; the convergent side is for Newton/Nova/Magnet |
| Exponential Smoothing `divergescale` | hardcoded 1.0, and absorbed to a no-op by the percentile-stretch |
| Binary Decomposition (two-entry threshold on the same angle) | not ported |
| ~20 trap shapes | 8 (`DirectShape`) — but see the name collision below |

**⚠ Three of the eight `DirectShape` names collide with UF's while computing something
different.** This is the port-parity hazard on this axis: a UF preset or a reader's UF
intuition does not transfer by name. Ours (`dist()`, trapcenter 0, `radius = trap_radius`):

| `DirectShape` | ours | UF's shape of the same name |
|---|---|---|
| `Point` | `|z|` | same |
| `Ring` | `||z| − r|` | same |
| `Cross` | `min(|Re|,|Im|)` | same |
| **`Hypercross`** | `min(axis-cross, diagonal-cross)` — an 8-ray "✳", the union of the axis cross and its 45° twin | **`|Re·Im|`** — a soft-asymptote cross. Different function. |
| `Diamond` | `|Re|+|Im|` (L1) | same |
| **`Box`** | `max(|Re|,|Im|)` (L∞) | **UF's `rectangle`.** UF's `box` is `|max(|Re|,|Im|) − D|`, a hollow outline; ours is the filled L∞ norm. |
| **`Astroid`** | `(|Re|^{2/3} + |Im|^{2/3})^{3/2}` — the astroid norm, exponent fixed | **`|Re|^k + |Im|^k`**, `k` a free parameter (reciprocal if `k<0`). Ours is one `k`, and pre-composed differently. |
| **`Lines`** | `|Im|` — distance to the real axis | **`||Im| − D|`** — a pair of lines offset by `D`. Ours is the single-line degenerate. |

Because the distance *scales* differ wildly between these norms (L1 ≥ min-axis, astroid ≫
L1), the per-shape `direct_threshold` defaults are **coverage-anchored** — each is that
shape's measured p95 closest approach, equalizing *painted fraction* against the cross's
settled 0.1 — rather than sharing a distance. So a threshold is not portable across shapes
either. `[code: src/render_modes.rs::DirectShape::{dist,default_threshold}]`

UF shapes not ported, as one-liners if one is ever wanted: `ring2` `||z|²−D²|`; `hyperbola`
`|Re·Im−D|`; waves / mirrored / radial (`Im` or radius modulated by `sin(·f)·k`); ring / grid
/ radial ripples (cosine ripples inside radius `k`, zero outside); and the specialized closed
curves `egg`, `pinch`, `spiral`, `heart`.

> **A superseded recommendation, kept as a warning.** The source reference proposed
> Exponential Smoothing as a formula-agnostic drop-in alternative to `smooth`. It was ported
> on that basis, **and then `smooth` was promoted the canonical base carrier**, which made
> the overlap redundant: the field is monotone with `smooth` (Spearman ≥ 0.999 across all 8
> pilot families), so its render-mode-pilot rasters were pixel-dupes of their `smooth`
> counterparts (ΔE76 < 5, all flagged `too_close_to_smooth`). `Field::ExpSmoothing` is now
> marked `niche`/deprecated for render-mode exploration in the code. Keep `smooth` as the
> base carrier. `[code: src/render_modes.rs::Field::ExpSmoothing]`

## 7. Open build task — the deploy-transform parity gate has zero coverage

The deploy-transform parity check — `present.rs` JPG path ↔
`classifier.data.Transform(train=False)` (the 1280×720 → 384×224 bicubic-stretch +
normalize) — has **no gate at all**. It is the single highest-value guarantee with zero
coverage and needs a test **built** (GPU-free: binary + PIL), not merely wired. Until it
exists, a silent divergence between what the classifier trained on and what the pipeline
deploys can go undetected.
