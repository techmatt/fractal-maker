# `crop-batch` — the extended-field crop executor

**The cache-build recipe from v11 on.** `src/crop_batch.rs` renders **one iteration pass per
LOCATION** over a slightly extended field and derives every tile from it as *crop + resample +
colormap*. The v4..v10 executor (`src/v4_cache.rs`, driven per plan ROW) re-iterated for every
tile, so a palette slot cost exactly as much as a geometry slot even though four palettes share
one escape-time field. This doc holds the decisions the module's own docs do not own: the
containment bound, the `field_ss` pin, the cap-policy change, and the measured costs.

Mechanism — why one scalar field is sufficient, how the AA axis is derived, the two fan-out
modes — is owned by the module doc comment in `src/crop_batch.rs`; read it there, not here.
The v11 build that consumes this is `tools/v11/{build_plan,render_cache}.py`, and the recipe it
froze is `data/v11/aug_recipe.json` (`v11-independent-32`).

## 1. Containment is an equal PLANE margin, and the live flags sit exactly on the bound

The margin is `(extend−1)/2` of the canonical frame **WIDTH**, the same plane distance on all
four sides — not a fraction of each axis's own extent. That follows from the shift being a
single magnitude in frame-width units with a uniform direction: at 16:9 a vertical displacement
of `0.05·fw` is 8.9% of the frame *height*, so per-axis padding under-pads the vertical and lets
the tallest shifted crop run off the bottom. The crop overhang from `scale_hi` *is* per-axis, so
the binding axis is the longer one:

```
extend  >=  1 + 2·shift_frac_max + max(1, H/W)·(scale_hi − 1)
```

At the live `512×288` / `[0.90, 1.10]` / `5%` that is `1 + 0.10 + 0.10 = 1.20` **exactly** —
`--extend 1.2`, the scale draw and the shift cap are ONE decision with nothing to spare, and
moving any one of them moves the other two. Validated, never clamped
`[code: crop_batch.rs::run_crop_batch, the CONTAINMENT block]`, with a `1e-9` tolerance only
because `1 + 0.1 + (1.1 − 1.0)` is `1.2000000000000002` in binary and a bare `<` would reject the
shipped configuration. The realized pad is a whole number of subpixels rounded **up**, so it is
never below what the flag asks for, and the per-crop window check in `run_location` is the real
guarantee either way. `tools/v11/build_plan.py` restates the bound into `data/v11/plan_record`
as an evaluated expression, not as a literal.

Because the margin is equal in the plane, the realized *relative* extension differs per axis
(1.201 wide / 1.358 tall at the defaults) and is reported as `extend_y` rather than hidden.

## 2. `field_ss 2` is pinned, and the grid fact is why the AA axis survives

The two AA levels are a **mode**, not a supersample factor — with one field there is no per-tile
`ss` left to name. The consequence is asymmetric and it is the reason the pin is a pin:

- the **antialiased** arm is EXACT in kind: lanczos3 in linear light at ratio `scale·field_ss`
  ∈ [1.8, 2.2], the same kernel and the same filtering the legacy `ss2 + lanczos3` tile used,
  at a ratio the random scale makes non-integer;
- the **aliased** arm cannot be exact. The legacy tile is `ss1 + box`, i.e. a point sample at
  each pixel CENTRE. An `ss2` field's sub-cell centres sit at 0.25/0.75 of a pixel, and **no
  even `field_ss` ever contains the 0.5 point** — an odd one does, but only for the identity
  crop, since a random shift breaks the alignment anyway. Running the box kernel instead would
  average ≈`ratio` subpixels and produce a *second antialiased tile*, destroying the axis. So
  the derived aliased tile is nearest-neighbour: a true point sample displaced by ≤ half a field
  subpixel.

`tools/v11/parity_crop_mode.py` sweeps `field_ss` for exactly this reason rather than as
decoration — at the identity crop `field_ss 3` reproduces the legacy point sample exactly and
`field_ss 2` cannot, so the aliased-arm question is a measurement and not an argument. The read
that settled it, over 30 locations spanning all nine families: at `field_ss 2` the deploy-matched
**antialiased** arm sits at max |Δscore| **0.0092 with ZERO decision flips**; `field_ss 3` merely
swaps which arm is exact and hands back the same three flips on the other one. All six arms were
looked at — the pairs are indistinguishable by eye and every flip is a score already sitting on
a cutpoint. `[measured 2026-08-07, afd9623; uv run python tools/v11/parity_crop_mode.py --n 30]`

The parity harness scores at **fp32, not autocast** — this is a difference of two scores near a
cutpoint, which `verification_practice.md` §1.9 records as the case where fp16 accumulation
moves rows across the line.

Raising `field_ss` costs `field_ss²` in iteration, which is the whole budget; 2 is the cheap
default that keeps the antialiased arm exact.

## 3. The cap is the CANONICAL frame's, per row — a real ~3% policy change, accepted

A row carries the **canonical** frame's `auto_maxiter(fw)`; the extended field iterates at that
cap and never re-derives one from its own wider `fw`. v9/v10 paid `auto_maxiter(fw_SLOT)` — the
scaled per-tile frame width — so the caps differ by **~3% of the cap** across the `[0.90, 1.10]`
scale draw. That is a policy change, not a rounding artifact, and it is **accepted rather than
corrected**: one field serves all 32 crops, so per-slot caps are not expressible.
`[code: tools/v11/build_plan.py, `maxiter.policy_change_vs_v9_v10`]`

There is deliberately **no default** for `--maxiter`. A row without a cap and without the flag is
a hard error naming the reason, because a silent flat cap is exactly what v4..v8 did (every tile
at 8000). `--maxiter-policy` stamps the token through verbatim; cross-policy comparison is the
concern `auto_maxiter.md` owns.

## 4. Stream-and-discard: no field-cache interaction

Fields are rendered into a local `Vec<f32>`, cropped, and dropped. Nothing here writes into any
field cache, which is what keeps the **frame-extension axis out of the cache key** — and the
field-cache key is one of only two byte-identity-critical seams in the tree
(`render_coloring_surface.md` §7). A cache-writing executor would have had to enter `extend`,
`field_ss` and the crop geometry into that key or collide silently.

## 5. Seeding and replay

Every draw — crop geometry, palette, AA coin, JPEG quality — comes from a `SplitMix64` seeded by
`(seed_tag, loc_id, slot)`, with disjoint slot namespaces per axis so adding a draw never
reshuffles the others. No global RNG and no ordering dependence: a tile is a pure function of its
location row plus this module. `seed_tag` is part of the recipe's identity (`v11-aug-20260808`
in `aug_recipe.json`), and changing it reshuffles the entire fan-out.

The emitted manifest records each tile's **realized** geometry in field-subpixel units
(`src_x0`, `src_y0`, `ratio`), so `--replay <manifest>` regenerates a tile byte-identically
without re-drawing anything and without depending on today's flag values.

`--limit N` runs the WHOLE path (field → crops → JPGs → manifest) on the first N locations and
stamps every row it writes `batch_incomplete: true` — the bounded-end-to-end rule from
`CLAUDE.md`, on the stage that WRITES.

## 6. Cost

| arm | cost |
|---|---|
| `crop-batch` | **0.531 s / location + 0.0104 s / tile** |
| `v4-render-batch` (v8b..v10) | **0.1577 s / tile**, no per-location term |

`[measured 2026-08-07, afd9623; uv run python tools/v11/measure_crop_cost.py]` — at the
sizing point that decided it, **3.10 h against 15.03 h** for 14.3k locations × 24 tiles. The
crossover is ~3.6 tiles/location; at v11's 32 tiles the gap is 5.8×, and every tile past the
first four is nearly free, which is what made widening 24 → 32 a free decision rather than a
budget one.

**Disk: 96.3 KiB/tile at the `q85..95` draw against 73.8 flat — the quality axis costs 30%.**
That is the only place the artifact level can move: train-time JPEG jitter can only re-encode
a floor the cache already set.

Two properties of that measurement are load-bearing and both are the harness's, not this doc's:
arm B's plan is built FROM arm A's emitted manifest, so the two render the identical
(viewport, palette, AA) slots and only the execution differs; and the sample is drawn in **run
order** as contiguous blocks per region of the location file, because `data/v10/manifest.jsonl`
is emitted in family order with the expensive material contiguous — the same shape that cost the
v9 cache render a 1.65× ETA miss (`measurement_practice.md`, run-order projection). The 32- and
48-tile points are FITTED from a measured fixed part and a measured marginal part, never assumed
free.

At the default `512×288 / field_ss 2 / extend 1.2` one location samples 1230×692 = 851k
subpixels against v8b's 8.85M — a **10.4×** cut in sampled subpixels, or 2.6× even against a
hypothetical executor that perfectly shared v8b's six distinct fields.
