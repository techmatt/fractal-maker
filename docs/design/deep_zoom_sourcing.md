# Deep-zoom architecture & deep-center sourcing

Distilled from the deep-Mandelbrot probes. Governs how genuinely deep (self-similar,
many-decade) wallpaper locations are *found* and what the render tier can and cannot
reach. Companion: the render core's precision tiers are documented in `CLAUDE.md`
(§Architecture / backend selection); this doc is about **sourcing** and the standing
engine gaps, not the per-pixel loop.

**Boundary — read this before adding anything here.** This file covers the *deep
regime*: beyond-f64 depth, the perturbation tier, and how valid deep centers are
produced. The **minibrot-atom sourcing arc runs at moderate depth, inside f64**, and
lives in `minibrot_sourcing.md` — the roster, the stage-1 screen `G`, the pre-filter,
the labeled evidence, and the atom-level signals. The two files touch at exactly two
places (the atom-size law in §2, and the fitted-objective status in §5); everything
else stays on its own side. Do not restate the other file's numbers here.

**Status: PARKED.** Floatexp and a deep field path are not being built in the current
era. §4 explains what reviving this costs and in what order.

## 1. Deep beauty is depth-band-local — harvesting is a ∂M-tracking problem

The perturbation tier renders degree-2 Mandelbrot cleanly across the f64→perturbation
switch down to ~1e-20 frame width (~3e20 magnification) with zero glitched pixels. But
"q4-class" beauty is **depth-band-local**: a fixed center holds a strong look only over
a limited band around where it was chosen. Generic features (e.g. a grand log-spiral)
fall into exterior/interior within a few decades and render flat; only truly
self-similar **Misiurewicz** centers sustain structure across many decades.

Consequence: harvesting deep locations at scale is a **∂M-tracking problem** — Newton
on nucleus / Misiurewicz points producing decimal-string centers — **not** a
floor-lifting problem. Nothing in the repo tracks ∂M while descending.

## 2. Valid deep centers must be *produced*, never guessed or offset

Precise ∂M placement at frame-width `F` needs the center known to ~`F`. Deep centers
therefore come from a **high-precision Newton finder emitting decimal-string centers**
(`tools/sourcing/deep_center_finder.py`):

- **Nucleus Newton** on `z_p(c) = 0` is the rock-solid workhorse.
- **Misiurewicz Newton** on `z_{k+n} = z_k` also works, but its residual is satisfied
  by *every* periodic parameter — roots must be **filtered for minimality**.

**Composition rule for framing.** A nucleus sits in interior black, so `fw ≈ size`
renders dead. The money-shot band is roughly `[~40×size (context) … ~2×size (island
fills frame)]`, with `fw ≈ 4×size` the sweet spot. **Misiurewicz points sit *on* ∂M**
and therefore fill the frame with structure at *every* scale — the better vein for
genuinely deep self-similar wallpapers. Going deeper on a nucleus requires offsetting
the center onto a decoration spiral.

**★ `size` means `1/|A|`, from `atom_instrument` — nothing else.** The atom-size
measure falls out of the same recursion Newton already runs, at ~zero cost, and
`|A| ≡ 1/|size|` holds exactly at every `n` (identity-tested), agreeing with the true
atom extent from about `n≈4–5` up. **The naïve degree-2 λ² law is forbidden at d≥3:**
it under-sizes the atom by 4–2497×, and framing a d≥3 field with it produces an
all-black window. This is not hypothetical — it once made a degree-3 screen-transfer
read report 100% OOD rejection, where the mask was innocent and the framing was the
fault. Every use of "size" in this section means `1/|A|`.

Cost note: the Newton solve is cheap; **render cost is the gate** — deep rungs are tens
of seconds at modest res even with no series approximation / BLA.

## 3. The `guided-descend` walker is structurally f64-bound

Below ~1e-15 the walk is invalid: `Frame.center` is `Complex<f64>`, node renders force
`BackendChoice::F64`, and candidate centers are stored as f64. So the center is
delocalized by many frame-widths and the walk emits invalid centers with no precision
guard. Lifting `--min-fw` is **necessary but far from sufficient** — a genuinely deep
walk needs the whole center path carried in high precision, not just a looser cap.

Note that the `1e-9` search floor is **policy, not precision** (`FW_FLOOR` / `--min-fw`,
liftable); the real wall is `MIN_FRAME_WIDTH = 1e-300`.

## 4. Standing engine gaps for deeper / broader zoom

Still missing (each is a real project if deep harvesting becomes the workstream):

- **Series approximation / BLA** — none. Every deep pixel pays full iteration, so deep
  renders are correct-but-slow at ~26–32 s/frame.
- **floatexp / scaled-double delta type** — to pass the ~1e-300 f64 delta-underflow
  ceiling.
- **Real Pauldelbrot glitch handling** — only a per-pixel underflow flag exists today.
- **Perturbation for non-Mandelbrot families** — multibrot / Julia / Phoenix are
  f64-and-shallow-only.

**★ What the gaps mean together — the tier renders but cannot SCORE.** `--dump-field`
is f64-only, and the emission render path is f64 regardless of which backend produced
the image. So a location that can be *rendered* deep cannot be *measured* deep, and a
location the pipeline can render but not score **is not a candidate**. The four gaps
are therefore not four independent projects but one dependency chain:

1. floatexp / scaled-double delta type — the one missing numeric piece past 1e-300;
2. a deep field path, so deep material can be scored at all;
3. then BLA / series approximation for speed, and real glitch handling.

Nothing downstream of deep sourcing unblocks until step 2 exists. Sequencing note: if a
GPU path is ever on the table, do **not** build deep `--dump-field` as a CPU-side Rust
job — that is exactly the code such a port rewrites.

## 5. Deep-Mandelbrot curation needs its *own* fitted objective (no transfer)

The Julia-derived "q4" aesthetic composite (see `julia_c_sourcing.md`) does **not**
transfer to deep-Mandelbrot minibrot fields, for two independent structural reasons:

1. **The interior-lake term is dead on minibrot exteriors.** `interior_frac ≈ 0` for
   every decoration-scale window — the only true interior is the central island no
   small decoration window touches — so the composite collapses to a busy-ness
   maximizer, which is anti-quality (see `aesthetic_scoring.md`).
2. **The preference is sign-inverted at the motif scale.** The desirable "calm" is the
   smooth deep-basin *exterior* (high `flat_frac`); good picks are concentrated spiral
   hubs with breathing room. A working deep composite needs the **opposite sign on
   `flat_frac`** plus a **concentrated-detail (anti-distributed)** term versus the Julia
   composite.

Design consequence: deep-center curation requires its **own small labeled seed and its
own fitted objective**. A heuristic borrowed from the Julia work cannot guess the sign
flip.

**★ Status: this was done, and the result is known.** The fitted objective §5 calls for
exists — the stage-1 goodness field `G`, with its own labeled window corpus. It has
since been measured against blind labels and is a **coarse gate, not a ranker**: within
its own accepts it carries essentially no ordering signal. §5's *design consequence* was
right; the objective that resulted is weaker than hoped, and the diagnosed reason is
that it is an edge-energy statistic with **no occupancy term**, while dendrites maximise
edge per unit area. `minibrot_sourcing.md` owns the numbers, the diagnosis, and the
invariants any successor inherits — do not restate them here.

**★ Scope correction: reason 1 is a degree-2 statement.** The `interior_frac ≈ 0`
observation holds for d=2 decoration-scale windows. It does not generalize — interior
mass rises monotonically with degree as the (d−1)-fold body grows, and the live
interior clause rejects roughly 19% of swept positions at d=2 against 33% at d=5.
Reason 2 and the design consequence survive at every degree; the interior-lake half of
the argument is d=2-only and must not be cited as evidence at d≥3.

**★ Relation to the live pre-filter.** The production minibrot screen rejects windows on
`interior_frac`, `flat_frac` and `speckle_ratio` ceilings (values and their measured
consequences live in `minibrot_sourcing.md`). §5's "the desirable calm is high
`flat_frac`" is a claim about the *mid* range, set against the Julia composite's sign —
it is not a licence to raise the flat ceiling. A window that is overwhelmingly flat is
dead space at any degree.
