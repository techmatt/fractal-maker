# How nuclei are actually seeded, the `A` instrument, min|z| scope, cache-source axis

Investigation answering `prompts/prompt_nucleus_seeding_and_A.md`. §2 (the build) has
its own design doc: `docs/design/atom_instrument.md`. This file is §1, §3, §4.

---

## §1 — What `deep_center_finder` actually does to seed nuclei (as-built, not intended)

**How Newton is seeded — a grid/ring draw of `c₀`, not a root census.** There is no
polynomial root-finding (no Aberth/Durand–Kerner), no atom-domain detection from a
rendered field. `newton_nucleus(c0, period, degree)` is plain scalar Newton on
`z_period(c) = 0` seeded from a single `c₀`. Two callers supply those seeds:

- `emit_deep_pool.py` — a **hand-curated `SEEDS` list** (rough f64 points near known
  valleys, each tagged with the period the curator expects). d2 only.
- `tools/studies/q4_multibrot_transfer.py::source_nuclei` — the path that produced the
  **d3/d4/d5 set**. Seeds are a **deterministic ring grid**: `_ring_seeds(degree)` =
  `n_rad=3` radii from `0.30` to `1.05·R_boundary` (`R_boundary = 2^{1/(d−1)}`) ×
  `n_ang=24` equally-spaced angles = 72 seeds, each tried at **every** period in
  `PERIODS = range(3, 16)`. So 72×13 = 936 Newton solves per degree, seeded blind.

**Period is chosen in advance, not discovered.** The target period is the loop
variable; Newton is run once per `(seed, period)` pair. Nothing reads a period
*estimate* off a field or an orbit — the code sweeps periods and keeps whatever
converges. (`scan` exists to *identify* a seed's period by trying them all, but the
sourcing path does not use it.)

**Period IS confirmed after convergence — minimality is checked, largeness is not.**
`newton_nucleus` confirms `|z_n| < tol` (the critical orbit returns to 0). Then
`_is_minimal_nucleus(c, period, degree)` rejects the solution if any **proper divisor**
`q | period` also closes (`|z_q(c)| < tol`), so a period-6 request that landed on a
period-3 nucleus is dropped — the reported period is the true minimal period, not a
multiple. `z_period ≈ 0` also means the true period *divides* `period`, so the divisor
check is sufficient: the kept period is exact. The `|c| < 1e-6` guard additionally
drops the `z=0` period-1 degenerate (which closes every period). This is genuine
confirmation, not faith in the seed.

**Dedup is by rounded coordinate only — `DEDUP_DPS = 22`.** `key = (nstr(cx, 22),
nstr(cy, 22))`; a nucleus is dropped only if another with a byte-identical 22-digit
coordinate was already found. There is **no sector/symmetry dedup**.

### ★ The population question — it is a SELECTED DRAW, not a census

The d3/d4/d5 set is **not** a complete low-period census. It is: {ring-grid seeds} →
Newton at periods 3–15 → keep minimal, non-degenerate, in the f64-dumpable size band
`[1e-10, 3e-2]` → dedup by coordinate → **spread across periods and truncated to 12
per degree** (`np.linspace` index pick). Completeness fails at three points: (a) only
72 seeds probe each degree, so low-period nuclei that no seed's basin reaches are
missed; (b) the `[1e-10, 3e-2]` size band excludes both the biggest low-period atoms
and anything past the f64 wall; (c) the final `→12` is an explicit subsample.

**Consequence for the transfer-read rates.** The pre-filter / OOD-mask and
"accepted" rates in `q4_multibrot_transfer.md` are computed over this **selected,
size-band-restricted, subsampled** set, not over a natural population of multibrot
minibrots. They are legitimate as an **apples-to-apples screen-transfer comparison**
(the d2 control is drawn the identical way), but they are **not base rates** for "how
often a multibrot minibrot is good." Read them as *"the d2 screen treats
comparably-sourced d≥3 atoms comparably,"* never as a yield.

### ★ Rotational symmetry — the draw DOES contain rotational duplicates

`z^d+c` has `(d−1)`-fold rotational symmetry about the origin (`c ↦ ωc`,
`ω^{d−1}=1`). The ring seeds sample all 24 angles around the full circle, so Newton
lands rotational copies of the same atom; coordinate-dedup cannot collapse them (a
copy has a different coordinate). Counting the current sets (`c ↦ ωc` orbits, tol
1e-6):

| degree | symmetry | rows | distinct orbits | redundant rows |
|---|---|---|---|---|
| d2 | 1-fold | 12 | 12 | 0 |
| d3 | 2-fold | 12 | 12 | **0** |
| d4 | 3-fold | 12 | 10 | **2** |
| d5 | 4-fold | 12 | 8 | **4** |

- **d4**: `d4_mb02_p04`, `d4_mb03_p04`, `d4_mb04_p04` are the **full 3-orbit** of one
  period-4 atom (all three 120° copies; `|c·ω − c′| ≈ 1e-16`). 12 rows → 10 distinct.
- **d5**: four period-{3,4,6,15} atoms each appear as a **rotational pair**
  (`mb01/mb02`, `mb03/mb04`, `mb07/mb08`, `mb09/mb10`). 12 rows → 8 distinct.
- **d3**: zero here — but that is luck of which basins the 24 angles hit, not a
  guard; the mechanism is identical (`c ↦ −c`) and *will* produce pairs on another
  seed set.

So the d4/d5 "12 minibrots each" are really **10 and 8** distinct looks. Field-level
rates that treat the 12 as independent double-count identical (rotated) geometry, and
any "distinct look" count off these sets is inflated by up to `(d−1)×`. **Fix when
this becomes a supply pipeline:** canonicalize each nucleus into one symmetry sector
(e.g. rotate to `arg c ∈ [0, 2π/(d−1))`) before the coordinate-dedup, so the draw
counts orbits, not copies.

---

## §3 — Scope only (NOT built): min|z| atom-domain detection

**What it would take.** While iterating each pixel of a field we already render,
track the running `min_k |z_k|` and the index `k*` where it occurs. A pixel with small
`min|z|` sits in the atom domain of a period-`k*` component; `k*` is a period estimate
and the pixel's `c` is a Newton seed. This sidesteps the enumeration ceiling that
kills a full census — complete enumeration is bounded by degree `d^{n−1}`, running out
around `n≈15` at d=2 and `n≈7` at d=5 — because it reads seeds off fields instead of
solving for all roots.

**What already exists.** Most of the machinery:
- The **iteration kernels** are the natural host. `src/backend.rs` /
  `src/render_modes.rs` already run the per-pixel `z_{k+1}=z_k^d+c` loop; adding a
  `min|z|`/`argmin k` reduction is a few lines in the sample loop (a new pair of
  fields on `PixelSample`, computed in the same pass — cf. how `trap_min`/`trap_phase`
  are already accumulated there for orbit traps).
- The **f64 field dump** path (`render-one --dump-field --dump-field-source f64`) is
  the field source; a second channel (`min|z|`, `k*`) would ride the same
  `.bin`+sidecar plumbing that `q4_window_reader` / `q4_multibrot_transfer._load_field`
  already read.
- The **Newton refiner** (`newton_nucleus`, degree-parametric) consumes the
  `(c_seed, k*)` pairs unchanged.

Missing: the reduction itself, a second dumped channel, and a small
"field → (seed, period) candidates" extractor (threshold on `min|z|`, cluster, emit).

**Where it would sit.** Reduction in the backend/render-mode sample loop; a
`min_z`/`period` field channel in the dump; a new extractor module beside
`tools/sourcing/` feeding `newton_nucleus`. It is a **sourcing** component, parallel
to the ring-grid seeder, not a change to Newton or the screen.

**Two tags for the writeup when it is built:**
1. **Supply, not a base rate.** A min|z| detector run over descended fields is
   selection-biased by whatever steered the descent (palette/quality gates, the
   walker's own path). Its output is a *supply stream* and must be tagged as a
   **separate population** from any census or from the ring-grid draw — never pooled
   into a rate.
2. **Misiurewicz false positives are a feature.** `min|z|` is also small near
   **Misiurewicz points** (pre-periodic, orbit passes close to 0 without a component
   there). Those "false positives" are exactly the self-similar-at-every-scale centers
   the deep probe prized (`deep_center_finder`'s misiurewicz path) — keep them, tagged,
   rather than filtering them out.

**Not implemented, per the brief.**

---

## §4 — Field-cache key vs field *source*: it was NOT covered; now it is

**Finding: `--dump-field-source` was not in the key.** The field-cache stems
(`assemble_queries._field_key`, `emit_v1._emit_field_stem`) hashed family + geometry +
maxiter + family-params + `field_mode_token` — but **not** the `--dump-field-source`
axis. `beautiful` (default, byte-identical smooth) and `f64` (fast escape-time backend,
offset by the constant `ln(ln B)/ln d`) share geometry and the NaN-interior seam but
differ in value; keyed only on geometry they collide on one cache entry and the offset
field leaks silently to any consumer.

No **live** collision exists today — both `ensure_field` and `ensure_emit_field` always
dump the default `beautiful` (neither passes `--dump-field-source`), and the one f64
consumer (`q4_multibrot_transfer`) writes to its own per-nucleus paths, not these
caches. So this is a **latent** hole: safe until the first consumer dumps f64 through
the shared cache.

**Closed it, mirroring `field_mode_token` exactly** (`location.field_source_token`):
`beautiful`/None → empty token appended → **every existing smooth stem byte-identical**
(no cache orphaned); `f64` → `"f64"` token, keying disjointly and orthogonally to the
mode axis. Threaded through both stem builders (3rd arg `field_source`) and pinned in
the frozen-literal oracle (`tools/corpus/test_location.py`:
`test_field_source_token_semantics`, `test_field_key_source_parity_beam`, + the emit
parity block) — the frozen `_BEAM_SMOOTH` / `_EMIT_SMOOTH` literals still hold, proving
the smooth path is unchanged. 13/13 `test_location.py` green.
