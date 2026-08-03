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

## The `c`-spacing floor — measured, and it is not an atom-level rule

**Two julia `c` values closer than `|Δc| = 1e-2` are near-duplicate LOOKS, whatever search
found them.** That is the minimum separation the near-minibrot supply channel draws at
(`supply_routing.CSPACING_FLOOR`), and it is derived rather than inherited.

`[measured 2026-08-03: data/label_corpus/batches/2026-08-03_q4_near_minibrot_v1 (290 labelled
rows / 103 atoms) × scratch/q4_readout/morph_emb_870.npz, library morph CLIP at cos ≥ 0.974]`

| \|Δc\| bucket | pairs | median cos | frac ≥ 0.974 |
|---|---|---|---|
| 1e-5 – 1e-4 | 75 | 0.9824 | 0.813 |
| 1e-4 – 1e-3 | 659 | 0.9708 | 0.417 |
| 1e-3 – 1e-2 | 2,196 | 0.9622 | 0.239 |
| **1e-2 – 1e-1** | 1,866 | 0.9267 | **0.024** |
| ≥ 1e-1 | 36,964 | 0.8992 | 0.004 |
| *different-atom pairs, any distance (reference)* | 41,627 | 0.9036 | *0.023* |

The floor is the coarsest bucket boundary at which the near-dup rate reaches the
different-atom baseline (2.4% against 2.3%); one bucket finer it is ten times that. Stated as
a bucket boundary, not a fitted knee, because the measurement is bucketed.

**This corrects the atom-level framing it came from.** The q4 sitting's readout found "same
atom, different rung ⇒ same look" (median cos 0.9825, 74.1% at or above the cut) and stopped
there, which reads as a rule about atom identity. Restricting to DIFFERENT-atom pairs shows
the saturation is a property of the c-plane distance: different atoms at 1e-4–1e-3 are still
38% near-dup, sixteen times baseline. So **"one `c` per atom" is not sufficient** — the
roster's own atoms sit a median 9.1e-4 apart, two buckets inside the floor.

Distinct from the julia **hook** spacing (0.20, or 0.10 after the campaign-2 resume), which is
10–20× coarser and was set on a different population. The two are not interchangeable.

### The 1×/4×/16× distance ladder buys ~1 look per atom, so v2 emits ONE rung

Yields are flat across the three rungs (labelled ≥3: 68.0 / 63.5 / 68.0%, one-per-cluster
61.8 / 65.3 / 66.7%; every pair's Wilson interval overlaps), and a paired cost measurement
(`tools/atlas/near_minibrot_rung.py`, 24 atoms × 3 rungs interleaved) puts the render-cost
spread at **3.9%** — flat too. With neither yield nor cost separating them, the choice falls to
the one column that does: one-per-cluster class-4, 8.6% / 4.0% / 3.7%. **Rung 1.** The ladder
as a three-rung emitter is retired (`retired.md`); it bought 3× the label cost for ~1 look.

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
