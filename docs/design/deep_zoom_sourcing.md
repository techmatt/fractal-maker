# Deep-zoom architecture & deep-center sourcing

Distilled from the deep-Mandelbrot probes. Governs how genuinely deep (self-similar,
many-decade) wallpaper locations are *found* and what the render tier can and cannot
reach. Companion: the render core's precision tiers are documented in `CLAUDE.md`
(§Architecture / backend selection); this doc is about **sourcing** and the standing
engine gaps, not the per-pixel loop.

## 1. Deep beauty is depth-band-local — harvesting is a ∂M-tracking problem

The perturbation tier renders degree-2 Mandelbrot cleanly across the f64→perturbation
switch down to ~1e-20 with zero glitched pixels. But "q4-class" beauty is
**depth-band-local**: a fixed center holds a strong look only over a limited band
around where it was chosen. Generic features (e.g. a grand log-spiral) fall into
exterior/interior within a few decades and render flat; only truly self-similar
**Misiurewicz** centers sustain structure across many decades.

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

Cost note: the Newton solve is cheap; **render cost is the gate** — deep rungs are tens
of seconds at modest res even with no series approximation / BLA.

## 3. The `guided-descend` walker is structurally f64-bound

Below ~1e-15 the walk is invalid: `Frame.center` is `Complex<f64>`, node renders force
`BackendChoice::F64`, and candidate centers are stored as f64. So the center is
delocalized by many frame-widths and the walk emits invalid centers with no precision
guard. Lifting `--min-fw` is **necessary but far from sufficient** — a genuinely deep
walk needs the whole center path carried in high precision, not just a looser cap.

## 4. Standing engine gaps for deeper / broader zoom

Still missing (each is a real project if deep harvesting becomes the workstream):

- **Series approximation / BLA** — none. Every deep pixel pays full iteration.
- **floatexp / scaled-double delta type** — to pass the ~1e-300 f64 delta-underflow
  ceiling (current perturbation v1 cap ~1e300 magnification).
- **Real Pauldelbrot glitch handling** — only a per-pixel underflow flag exists today.
- **Perturbation for non-Mandelbrot families** — multibrot / Julia / Phoenix are
  f64-and-shallow-only.

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
