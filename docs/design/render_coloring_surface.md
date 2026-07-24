# Render / coloring path surface & port-parity invariants

Distilled from `render_config_report` and `wire_parity_gates`. Governs the coloring
path that turns a field into an image, the disjoint palette namespaces, and the
cross-language OKLab ports that must **not** be refactored together.

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

## 6. Open build task — the deploy-transform parity gate has zero coverage

The deploy-transform parity check — `present.rs` JPG path ↔
`classifier.data.Transform(train=False)` (the 1280×720 → 384×224 bicubic-stretch +
normalize) — has **no gate at all**. It is the single highest-value guarantee with zero
coverage and needs a test **built** (GPU-free: binary + PIL), not merely wired. Until it
exists, a silent divergence between what the classifier trained on and what the pipeline
deploys can go undetected.
