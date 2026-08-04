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

## The `c`-spacing floor — `3.2e-2`, and there is no knee to read it off

**`supply_routing.CSPACING_FLOOR = 3.2e-2`** is the minimum separation between two accepted
julia `c` values, applied **across channels** (the saturation is a property of c-plane
distance, not of which search found the point). It is a **tolerance chosen against pool cost**,
not a point where the looks stop being similar — the whole reason the previous `1e-2` needs
this section rewritten rather than edited.

### Similarity vs \|Δc\|, at a FIXED z-viewport

`[measured 2026-08-03 · `uv run python tools/studies/julia_c_stationarity.py all` (~25 min) ·
1,421 `c` (14 regions × 63 satellites over 10 half-decade annuli 1e-5–3e-1, plus the 539-`c`
committed v2 pool) × 3 **shared** z-viewports = 4,263 canonical morph_clip embeddings
(robustz_tanh_k2_v1, 640×360 ss2, ViT-B/16 CLIP) at cos ≥ 0.974]`

Both members of every pair render at the **same** viewport, so a cosine difference is a
difference in the Julia set and never in the framing. Pairs are screened to viable on both
sides; 85.4% of drawn `c` pass.

| \|Δc\| | constructed pairs | median cos | **≥ 0.974** | v2-pool pairs | **≥ 0.974** |
|---|---|---|---|---|---|
| 1e-5 – 3.2e-5 | 507 | 0.9990 | 1.000 | — | — |
| 1e-4 – 3.2e-4 | 1,161 | 0.9984 | 0.977 | — | — |
| 3.2e-4 – 1e-3 | 1,509 | 0.9965 | 0.875 | — | — |
| 1e-3 – 3.2e-3 | 1,888 | 0.9902 | 0.673 | — | — |
| 3.2e-3 – 1e-2 | 3,528 | 0.9797 | 0.538 | — | — |
| 1e-2 – 3.2e-2 | 4,231 | 0.9487 | 0.354 | 484 | 0.130 |
| **3.2e-2 – 1e-1** | 4,175 | 0.9091 | **0.153** | 1,828 | **0.045** |
| 1e-1 – 3.2e-1 | 3,479 | 0.8430 | 0.029 | 7,010 | 0.019 |
| *different-region pairs, any distance (reference)* | 26,368 | 0.8454 | *0.0036* | | |

The v2-pool column is the unmanufactured cross-check: every member is production-accepted, and
because that pool is thinned at 1e-2 it has no sub-floor pairs to contribute. Where the two
cohorts overlap they differ by ~2.7×, and the ∂M-distance split below is why — isotropic
satellites drift off the boundary, the pool is selected onto it. **Quote the pool column.**

**There is no knee.** Every bin below 3.2e-1 sits above the baseline and the decay is smooth
and monotone across five decades. The rule that produced `1e-2` — "the coarsest bucket
boundary at which the near-dup rate reaches the baseline" — cannot pick a floor on a curve
like this, and only appeared to because its two members were rendered at their **own**
framings, scoring framing dissimilarity as look dissimilarity.

### Reading the floor at the floor

A bin average is not the number a floor admits: the rate is falling through the bin, so what
survives is the rate at its **bottom**. Quarter-decade, canonical `wide` viewport,
production-accepted pairs:

| \|Δc\| | 1e-2 – 1.8e-2 | 1.8e-2 – 3.2e-2 | **3.2e-2 – 5.6e-2** | 5.6e-2 – 1e-1 |
|---|---|---|---|---|
| near-dup | 0.196 | 0.094 | **0.074** | 0.037 |

So the adopted floor admits closest pairs at **7.4% near-dup against 19.6% at the old floor** —
2.6× fewer — and the pool pays for it: 539 → **209** `c` (`julia_supply_pool_v3.json`).

### Two conditions on how this generalizes

- **Viewport-conditional.** The same pairs at 1e-2 read 0.354 near-dup at the wide whole-Julia
  framing (fw 1.3), 0.130 at a mid-zoom (fw 0.55), 0.329 off-centre, and 0.098 under "near-dup
  at all three". The class is emitted wide (§Framing), so **wide is the read** — and it is the
  conservative one. A pipeline that emitted mid-zooms could run a finer floor.
- **∂M distance moves the knee ~½ decade.** Splitting `c` at the median exterior distance
  estimate (1.6e-4), near-dup at 1e-2–3.2e-2 is **0.294 near the boundary vs 0.547 further
  out**, consistent in direction in every bin. **Not adopted as a covariate-scaled rule**:
  every production channel selects onto the knife edge, which is the half the absolute floor is
  already set on. Atom size moves the same way but is confounded with region identity (3 vs 3
  regions) and is not a rule.

### What the atom-level framing got right

The q4 sitting's readout found "same atom, different rung ⇒ same look" (median cos 0.9825,
74.1% at or above the cut) and stopped there, which reads as a rule about atom identity. The
distance framing is what survives: **"one `c` per atom" is not sufficient**, because the
roster's own atoms sit a median 9.1e-4 apart — two decades inside the floor — and different
atoms that close are near-duplicates of each other whatever their provenance says.

Distinct from the julia **hook** spacing (0.20, or 0.10 after the campaign-2 resume), which is
3–6× coarser and was set on a different population. The two are not interchangeable.

### Raising the floor does not mean re-thinning the old pool

`build_julia_supply_pool_v2.py` re-thins the **full merged candidate list** (4,587) at the new
floor and ships a new versioned file; the previous pool stays on disk as the record of the old
selection. That is the right construction — but it is worth knowing what it buys, because the
answer is *almost nothing in count*: **209 `c` from the full re-thin against 206 from naively
re-thinning v2's own 539**, with an identical channel mix in all three high-yield channels and
204 of the 209 shared. First-wins thinning is why: v2's survivors were already each cluster's
best-priced member, so re-electing under a coarser floor mostly re-elects the same points.
The gain from the floor raise is real and is elsewhere — expected `≥3` rate over the rows with
a measured channel yield goes **0.314 → 0.403** as `seeded_loop`'s share falls 71% → 55%.
`[measured 2026-08-03, data/atlas/julia_supply_pool_v3_report.json]`

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
