# Julia `c`-sourcing — the "corner" exemplar class

Distilled from the q4 study line (`q4_axis_discovery` → `q4_decisive_pass`). Governs
how the strongest Julia wallpapers are sourced in `c`-space. The earlier probes
(`q4_c_perturbation`, `q4_dM_property`, `q4_neighborhood_sweep`) were underpowered and
their intermediate verdicts were overturned here — this is the settled result.

## The class is real and targetable (~2% of viable `c`)

Exemplar-grade Julia "corner" images — **distributed multi-scale filament detail
combined with composed interior (black-lake) negative space** — form a real,
targetable structural class, **not** a one-off coordinate. It occupies roughly **2%**
of viable near-∂M `c` values (Wilson 95% CI ~1.3–3.9%).

## The three-stage `c`-selection screen (campaign-3 recipe)

1. **Boundary-rejection sample.** Draw `c` where membership is non-constant over
   `{c} ∪ ring(ε ≈ 0.02)`; arc-length-weight for boundary diversity; dedup at a minimum
   separation. (The ε-shell is already ~70% viable.)
2. **One-render viability screen** at mid-`fw` (~0.6): reject solid-blob
   (`interior > 0.85`) and dust (`mid < 0.04 ∧ occ < 0.06`).
3. **Rank survivors** by minimizing `|dist_dM|` (∂M knife-edge proximity) **AND**
   requiring the interior-lake channel (`interior_frac`) to fire.

### Operating rule: run the sampler to the knee, then refill

The sampler's yield is strongly front-loaded — roughly **90% of the looks it will ever give
you arrive in the first ~15 charged minutes**. Past that it is still producing, just at a
price that a **fresh pool (~6 minutes)** beats outright. So: run to the knee, then refill;
do not run the same pool into its tail.

The corollary matters more than the rule. **The headline late-tail price is an artifact of
running past the knee**, not a property of the sampler or of the class's rarity — quote it
only alongside the refill cost, or it reads as a cost of `c`-sourcing rather than a cost of
one operating choice.

The pool is not the constraint. The ε ≈ 0.02 near-∂M shell supports on the order of
**8,350 viable `c`'s** at the sampler's `MIN_SEP` dedup, against a `POOL_TARGET` of **750**
per pass — so a refill draws fresh ground rather than re-drawing the same candidates.
`[code: tools/studies/q4_decisive_pass.py::{POOL_TARGET, SHELL_EPS, MIN_SEP};
tools/atlas/build_julia_seed_pool.py]`
`[unverified: the 90% / 15-min / 6-min / 8,350 figures are supplied operating knowledge —
no run record in this tree reproduces them, and no committed tool reports a knee. Only
`POOL_TARGET = 750`, `SHELL_EPS = 0.02` and the ~40% viability rate are checkable here.]`

## The interior-lake fraction is the determinant

`interior_frac` is what scalar Mandelbrot proximity / richness **cannot see**:
near-misses are busy-without-composed-lakes at matched mid-detail. This is the single
channel that separates the exemplar class from generic near-∂M busyness.

## Framing and variety

The class favors **wide, whole-Julia framings** (`fw ≈ 1.0–1.4`), not mid-zooms, and
carries genuine motif variety (no near-dups). Do not mid-zoom-crop the search.

## Scope boundary

This recipe is **Julia-specific**. The interior-lake term is dead on deep-Mandelbrot
minibrot exteriors and the artist preference there is sign-inverted — see
`deep_zoom_sourcing.md` §5. Do not port this composite to deep Mandelbrot.
