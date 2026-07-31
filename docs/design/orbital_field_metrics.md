# `tools/orbital/` — the ring measures (`radial_rings`, `radial_range`)

Named for the code that owns it: the measure family implemented in
`tools/orbital/field_metrics.py` (`radial_rings`) and `tools/orbital/rescore_lib.py`
(`radial_range`, plus the single-pass computation of both). This doc is the validity
record — what the measures compute, what they were validated against, what they are
blind to. It does not restate the code; every claim below is tagged with what makes it
true.

**Tags.** `[code: path]` — true because the tree says so. `[measured: population]` — a
number, with the population it is true *of*. `[verdict: who]` — a judgement call.
`[unverified]` — supplied from outside the repo and not checkable here.

**Boundary — three docs touch this one, none overlap it.**
- `auto_maxiter.md` owns the production iteration-cap policy, the 32-atom convergence
  ladder, and the ×8 raise. This doc owns only how a cap change propagates *into* a ring
  score. `[code: docs/design/auto_maxiter.md]`
- `minibrot_sourcing.md` owns the stage-1 screen `G`, the atom roster and the `A`
  feasibility cut. `[code: docs/design/minibrot_sourcing.md]`
- `storage_classes.md` owns the durability contract the `data/orbital/` artifacts sit
  under. `[code: docs/design/storage_classes.md]`

---

## 1. What the measures compute

**The shading constant is the whole mechanism.** `coloring::shade` computes
`t = smooth_iter * density + offset` and wraps it, with `density = 0.025` fixed as the
`ShadeArgs` default — so one colour cycle is 40 iterations at every depth, and "how many
rings do I cross walking outward" is a property of the escape-time field, not of the
palette. `[code: src/cli.rs ShadeArgs::density; tools/orbital/field_metrics.py DENSITY,
pinned against the Rust source by test_orbital.py::test_density_matches_the_render_path]`

**The input is a raw field, never a PNG.** Fields come from
`render-one --dump-field --dump-field-source f64`: little-endian f32, row-major, NaN where
the pixel did not escape. A rendered tile pins `smooth_iter` only modulo 40 iterations —
which is the quantity being measured — so tiles cannot be reused.
`[code: tools/orbital/field_metrics.py::dump_field]`

**Ray geometry, shared by both measures.** 64 rays from the frame centre
(`np.linspace(0, 2π, 64, endpoint=False)`), each sampled at `max(32, int(r_max))` points
out to the inscribed radius `r_max = min(cx, cy)` in pixels, nearest-neighbour sampled and
clipped to the frame. Every ray has the same length and none leaves the frame. NaN
(interior) breaks a ray into segments; each quantity is computed per segment.
`[code: tools/orbital/field_metrics.py::radial_rings, tools/orbital/rescore_lib.py::ring_measures]`

**A crossing** is an integer-boundary crossing of `smooth_iter * DENSITY` between adjacent
samples: `Σ |floor(t[i+1]) − floor(t[i])|` over a segment. Crossings **sum** across a
ray's segments, so a black island in the middle costs only its own span.
`[code: tools/orbital/field_metrics.py::_crossings]`

**`radial_rings` = the median over the 64 rays** of that per-ray crossing total. The
median, not the p90 — see §4. A p90 is computed alongside and recorded, but is not the
measure. `[code: tools/orbital/field_metrics.py::radial_rings]`

**`radial_range` = the median over the same 64 rays** of the per-ray *span*: `max − min`
of `t` within a segment, taking the **max** across a ray's segments (a genuine radial
excursion, not noise summed across an interior island).
`[code: tools/orbital/rescore_lib.py::ring_measures]`

**What makes `range` the monotone half of the pair.** Span counts each unit of field
excursion once however many times the ray crosses it; crossings accumulate on every
transition. A ray dithering across a single boundary racks up crossings without span,
which is exactly the failure mode `range` was added to expose. The asymmetry is
deliberate and is the only difference in the two ray walks: crossings **sum** over
segments, span takes the **max**. `[code: tools/orbital/rescore_lib.py::ring_measures]`

**One ray walk, two measures.** `ring_measures` recomputes crossings itself rather than
calling `radial_rings`, so that both measures see byte-identical rays; it reuses
`fm._crossings` verbatim and a self-check asserts its crossings equal `fm.radial_rings`
exactly. The self-check is `__main__`-only — see §9.
`[code: tools/orbital/rescore_lib.py::_selfcheck]`

**Also computed, and reported as failures (§4):** `cycles_spanned` (`(p95 − p05) × density`
over escaping pixels), `falloff_extent` (radial span over which the binned median
descends 90%→10% of its range, 40 bins), `interior_fraction` + its 8-annulus radial
profile. `[code: tools/orbital/field_metrics.py::measure_field]`

**Not available: a per-pixel atom-domain shell index.** `--dump-field` serializes exactly
one scalar coloring mode and the mode list has no atom-domain member, so there is no
kernel to ask for it; adding one means a new coloring mode in the Rust renderer.
`[code: src/render_one.rs --dump-field-source]`

## 2. Geometries, and where they are set

| | width × height × ss | constant | used by |
|---|---|---|---|
| validation / full-res | 320 × 180 × 1 | `MEASURE_W/H/SS` | `measure_atoms`, `measure_convergence_ladder`, `rescore_lib` defaults |
| screening | 64 × 36 × 1 | `SCREEN_W/H/SS` | `screen_pool.screen` |

`[code: tools/orbital/field_metrics.py:160-164]` — the same file also sets `N_RAYS = 64`,
`N_RADIAL_BINS = 40`, `FIELD_TIMEOUT_S = 60`.

**Every score to date was computed at one frame scale: 4× the atom's `window_scale`.**
`measure_atoms.measure_one(scale=4)`, `screen_pool.screen` (`fw = window_scale * 4`),
`measure_convergence_ladder.ladder_for_atom` (`* 4.0`).
`[code: tools/orbital/{measure_atoms,screen_pool,measure_convergence_ladder}.py]`
**If the operational frame scale moves, none of the validation below transfers — it must
be re-checked.** `[verdict: Matt]`

**The screen's cost model, and the reachability floor it implies.** The screen renders at
64×36 (≈2 ms of compute, ≈76 ms wall including process spawn); `render-one` refuses a
frame whose pixel spacing is `≤ 1e-13`, which at 64 px wide means an atom is screenable
iff `window_scale × 4 / 64 > 1e-13`. `[code: src/render_one.rs:200-207;
tools/orbital/field_metrics.py::dump_field]`

## 3. Invocation, artifacts, and deployment status

| producer | writes | holds |
|---|---|---|
| `measure_atoms.py` | `data/orbital/measures.jsonl` (945 rows) | per-atom `radial_rings`, `radial_rings_p90`, `cycles_spanned`, `falloff_extent`, `interior_*` at 320×180 |
| `measure_atoms.py` | `data/orbital/validation.json` | reference-vs-triage separation verdict per measure |
| `measure_atoms.py` | `data/orbital/maxiter_stability.json` | drift of the measures across ×1/×2/×4 cap multipliers, n=24 |
| `screen_pool.py` phase 1 | `data/orbital/screen_pool.jsonl` (4,669 rows) | the Newton **enumeration** — analytic atom properties only, no rendered quantity |
| `screen_pool.py` phase 2 | `data/orbital/screen_scores.jsonl` (3,759 rows) | per-atom `radial_rings` at 64×36 |
| `screen_pool.py` | `data/orbital/screen_report.json` | screen distribution, implied floor at keep-top, top ids |
| `measure_convergence_ladder.py` | `data/orbital/maxiter_convergence_ladder.json` | 32-atom cap ladder — owned by `auto_maxiter.md`, not by this doc |

`[code: tools/orbital/*.py, data/orbital/; row counts from the committed files]`

**`radial_range` has no committed output and no consumer.** It is implemented in committed
code (`rescore_lib.ring_measures`) but no committed writer records it: `screen_pool` and
`measure_atoms` both go through `field_metrics`, which computes crossings only. The only
committed caller of `ring_measures` is `measure_convergence_ladder`, which reads the
crossings half. Every `radial_range` number in evidence (§4, §5) lives in
`scratch/rescore/` and is therefore disposable. `[code: grep for `radial_range` /
`measure_both` over tools/ returns `rescore_lib.py` and `measure_convergence_ladder.py`
only]`

**`rescore_lib.scoring_maxiter` has no caller at all**, and today returns a different cap
than the one the `scratch/rescore/` evidence was computed under: that evidence used the
fitted 24×-of-legacy-production envelope clamped at 67000 (`scoring_cap.json`, which
stayed in `scratch/` and was never adopted); the committed module finds no
`scoring_cap.json` beside it and falls back to 8× of the **raised** production cap —
200000 at `fw = 8e-10` against production's 42165. `[code: tools/orbital/rescore_lib.py;
scratch/rescore/scoring_cap.json; verified by running `rescore_lib.py` as `__main__`]`

**Nothing downstream consumes any of this.** No sourcing, descent, corpus or classifier
code reads `data/orbital/`; the only tracked references to it are the size-guard
disposition and the tracked-artifact canary list. The screen is run by hand and its output
is read by a human. `[code: grep `data/orbital` over tools/ src/ tests/ → `tools/audit/size_guard.py`,
`tests/test_tracked_artifacts.py`]`

## 4. Validation record

**The references.** `ref_eye` "minibrot eye" — `cx -0.746339, cy 0.112242`, base scale
1.4575e-4, **not a nucleus** (no period). `ref_mb19` "mb19_p35" — period 35, log10|A|
9.695, base scale 2.0174e-10. The eye being shallow and mb19 deep is the point: a measure
that ranked the eye low would be depth wearing a disguise.
`[code: tools/descent/triage_store.py::load_references]`

**Test A — do both references outrank all 200 triage-wall atoms at 4×, 320×180?**
`[measured: 2 references vs 200 triage atoms, legacy cap, data/orbital/validation.json]`

| measure | eye | mb19 | triage max | verdict |
|---|---|---|---|---|
| `radial_rings` | 146.5 | 80.5 | 51.5 | **PASS** — 0 triage atoms reach either |
| `radial_rings_p90` | 198.0 | 197.4 | 120.0 | passes here; fails at ×4 cap (below) |
| `cycles_spanned` | 8.33 | 9.86 | 9.32 | **FAIL** — 1 triage atom beats the eye |
| `falloff_extent` | 0.200 | 0.314 | 0.343 | **FAIL** — 144 of 200 (72%) beat the eye |

**Re-run at a non-clipping cap, the separation holds and widens.**
`[measured: same populations, at the 24×-of-legacy scoring cap; scratch/rescore/measures_rescored_report.json — scratch, not committed]`

| measure | eye | mb19 | triage max | triage atoms ≥ eye | verdict |
|---|---|---|---|---|---|
| `radial_rings` | 146.5 | 140.5 | 89.5 | 0 | **PASS** |
| `radial_range` | 17.77 | 70.30 | 43.34 | 21 | **FAIL on the eye** |

**`radial_range` fails validation on one reference, and the failure is informative rather
than fatal**: the eye scores 17.77 against mb19's 70.30, below 21 triage atoms. Drop the
eye and mb19 clears triage max (43.34) comfortably. `[measured: as above]`
⇒ **the eye's `rings` validation passed for a different reason than mb19's** — its
richness is dense spiral *oscillation*, not radial *span*. `[verdict: Matt]`

**Ordering survives a cap multiplier, which is what a screen needs.** eye / mb19 / triage
max: ×1 146.5 / 80.5 / 51.5 · ×2 146.5 / 108.5 / 63.5 · ×4 146.5 / 137.0 / 76.0. The eye
is flat across the ladder because its field never reaches the cap; mb19 climbs because it
was clipped. `[measured: 2 references + 200 triage atoms; scratch/orbital_falloff.md —
scratch, not committed; the committed `maxiter_stability.json` records only per-multiplier
medians over n=24, not this ordering]`

**`radial_rings_p90` passes at ×1 and ×2 and fails at ×4** (triage max 242.8 vs eye 198.0)
⇒ the **median**, not the p90, is the measure. `[measured: same populations;
scratch/orbital_falloff.md]` **No test pins this failure** — `validation.json` records p90
as separating at ×1 and the committed brackets pin only the `cycles_spanned` /
`falloff_extent` failures. `[code: tools/orbital/test_orbital.py::test_the_measures_that_failed_are_recorded_as_failed]`

**Alternatives that failed and should not be retried:** `cycles_spanned`,
`falloff_extent`, `radial_rings_p90` — each above, each for a stated reason.
`[verdict: Matt]`

**Screening resolution ranks like full resolution.** Over the 319 atoms measured at both
64×36 and 320×180: Spearman 0.857 (legacy cap) / 0.853 (clean cap) for `radial_rings`,
0.832 for `radial_range`; top-decile agreement 28/31, 29/31 and 25/31 respectively.
`[measured: 319 atoms present in both data/orbital/measures.jsonl and screen_scores.jsonl,
recomputed for this doc]` An earlier session reported Spearman 0.873 and 17/20 top-decile
agreement for `radial_rings`; that pairing's producer is not in the tree and the figure is
not reproducible from it. `[unverified]`

**Consistency check nobody asked for.** Ranked by median `radial_rings`, the seven source
populations plus the triage wall come out: `atlas` > `label_seeded` > `neighborhood` >
`descent` > `misiurewicz` > `probe` > `complete_low_n` > `triage` — human-curated and
human-labelled sources at the top, the flat wall at the bottom. Nothing was fitted to
produce that. `[measured: 945 atoms, data/orbital/measures.jsonl]`

## 5. The two-axis reading

**`range` ≈ radial span; `rings`/`range` ≈ oscillation density.** Eye = high oscillation /
low span · mb19 = high span · the residual set = high on both · the flat wall = low on
both. `[verdict: Matt]`

**They are near-duplicates in the population and separate only at the top, which is where
selection happens.** Spearman(`rings`, `range`) = 0.959 over the 3,759 screened atoms and
0.973 over the 945 measured atoms; but the top-N sets disagree — top-100 overlap 87%,
top-300 75%. `[measured: scratch/rescore/{screen_rescored,measures_rescored}.jsonl,
recomputed for this doc]`

**Per-source, the two axes disagree at exactly one place: the top.** The `rings` ordering
of the eight populations is rank-for-rank identical before and after the clean-cap rescore
(all eight values move; no pair reorders). Under `range` the same ordering holds except
that the top two swap — `label_seeded` (74.2) over `atlas` (68.3), reversing `rings`'
`atlas` (197.3) over `label_seeded` (168.0). So the per-source reading is not a cap
artifact. `[measured: 945 atoms, data/orbital/measures.jsonl vs
scratch/rescore/measures_rescored.jsonl, recomputed for this doc]`

**`rings` is strongly resolution-dependent on fine filigree; `range` is steadier, but
neither is resolution-invariant.** Over the 319 doubly-measured atoms the full-res/screen
ratio is ×6.0 median (p10–p90 4.5–7.9) for `rings` and ×4.3 median (3.3–5.9) for `range`.
On eight high-filigree residual atoms the same locations read 124.5–152.5 `rings` at 64×36
and 587.5–952.0 at 320×180. `[measured: 319 paired atoms; 8 residual atoms in
scratch/rescore/eye_vs_residual.json]`

⇒ **`range` is the better screening statistic, `rings` the better validation statistic** —
`range` because it is the less resolution-sensitive of the two and the one that keeps its
meaning when the cheap geometry is what you can afford; `rings` because it is the one both
references pass. **Record the split rather than collapsing it: absolute scores are never
comparable across resolutions in either measure** (mb19: 80.5 at 320×180, 20.5 at 64×36),
only orderings are. `[verdict: Matt]` `[measured: as above]`

## 6. Relationship to period and to depth

**Global Spearman(`rings`, period) = +0.869 (legacy cap) / +0.862 (clean cap) over the
3,759 screened atoms — but the two rankings are not the same sort at the sharp end.**
Top-N overlap between a `rings` sort and a period sort: top-100 **12%**, top-300 37.3%,
top-1000 75.7% (legacy cap); 9%, 38.7%, 74.6% under the clean cap. The screening render
earns its cost precisely where selection happens.
`[measured: data/orbital/screen_scores.jsonl and scratch/rescore/screen_rescored.jsonl,
recomputed for this doc]`

**It is not depth in disguise.** Spearman(`rings`, log10|A|) = +0.368 over the 945 measured
atoms and +0.122 over the 3,759 screened atoms; `range` is somewhat more depth-coupled at
+0.482. And the highest-scoring thing measured — the eye — is not a nucleus at all, so no
period or depth sort can rank it. `[measured: as above; the < 0.6 bound is a live test,
test_orbital.py::test_radial_rings_is_only_weakly_correlated_with_depth]`

**The earlier retirement of period as a quality axis was scoped to a range, not to the
axis.** Period was measured at +0.06 pooled and **−0.21** inside the period-matched eval
slice — on a roster spanning **period 3–15** (median 8). The population here spans 2–74.
Degree's +0.55 was measured on the same restricted roster and carries the same suspicion
until re-measured. `[code: docs/design/minibrot_sourcing.md §5;
docs/design/minibrot_roster_v2_readout.md]` `[measured: data/minibrot_roster/roster.jsonl,
n=163, period 3–15; data/orbital/measures.jsonl, period 2–74]`

## 7. Cap provenance

**The cap is an input to every value here by construction.** Every field dump is sized by
`rc.auto_maxiter(fw)` at 1×, and `auto_maxiter` reads the live production constants — so
when the production cap was raised (base 500 → 4000) this whole stack followed the raise
silently: same code, same locations, different numbers. `[code: tools/explorer/render_core.py::auto_maxiter;
tools/orbital/measure_atoms.py, screen_pool.py; docs/design/auto_maxiter.md]`

**The mechanism, as it stands after the cleanup pass.** Every score record carries
`maxiter_policy_token`, resolved through `location.maxiter_policy_token` rather than
re-derived. The token is the **empty string** for the legacy policy `(500, 0.30, 200,
8000)`, and a record with the key **absent** reads as legacy by the same invariant — so
records written before the axis existed read correctly instead of raising.
`[code: tools/orbital/field_metrics.py::{POLICY_KEY, record_policy, describe_policy};
tools/corpus/location.py::{LEGACY_MAXITER_POLICY, maxiter_policy_token}]`

**A cross-policy comparison raises; it does not return a mixed number.**
`fm.require_one_policy(*groups, what=...)` asserts one policy across every group and
returns that token, or raises `MaxiterPolicyMixError` naming both policies, their counts,
which labelled side carried which, and up to three example ids. It is called at every
point that compares or pools: the reference-vs-triage verdict, the maxiter-stability drift
ratios, the screen's resume load (before the screening budget is spent) and the screen's
distribution/keep-top aggregation.
`[code: tools/orbital/field_metrics.py::require_one_policy and its four call sites]`

**The enumeration is deliberately unstamped, and that is not an oversight.**
`screen_pool.jsonl` holds Newton nuclei from `atom_lib.solve_nucleus` (mpmath);
`cx/cy/window_scale/period/log10_abs_A/f64_margin_deploy_decades` are analytic properties
of the atom — nothing on that path renders a field or reads an iteration cap, so a cap
token there would assert a dependence that does not exist. **A false provenance claim is
worse than none**, which is why the missing stamp is the correct state and not a gap to be
closed. The test pins it **in both directions**: the day a rendered quantity is added to
the enumeration it goes red and the disposition gets re-decided on purpose, rather than a
later reader "fixing" it the wrong way. `[code: tools/orbital/stamp_cap_policy.py;
test_orbital.py::test_the_enumeration_is_not_stamped_with_a_cap_policy — which also pins
the file's exact key set]`

**Every committed score record is stamped legacy**, and `stamp_cap_policy.py --check`
re-asserts it. `[code: tools/orbital/stamp_cap_policy.py;
test_orbital.py::test_committed_score_records_are_stamped]` **Consequence: every number in
`data/orbital/` is a pre-raise measurement.** A re-measurement under the live policy is a
different quantity and must not be appended to those files — the resume guard enforces
this. `[code: tools/orbital/screen_pool.py::screen]`

**The ×8 raise itself is corroborated in-repo but is not this doc's claim.** The 32-atom
ladder — `data/orbital/maxiter_convergence_ladder.json`, producer
`measure_convergence_ladder.py`, convergent-cap ratio **mean 7.688 / median 8.0 / max
24.0** over 32 atoms with **0 unconverged** — and the independent `maxiter_stability.json`
(n=24, `radial_rings` 45.0 → 55.25 → 60.75 across ×1/×2/×4, still climbing with no
plateau) are owned by `auto_maxiter.md`. `[measured: 32 atoms stratified over the pool's
fw range + both references; data/orbital/maxiter_convergence_ladder.json]`

**Carry the ladder's warning whenever you cite it.** It is stamped legacy-policy
(`maxiter_policy_token: ""`) and re-running its producer does **not** reproduce it: every
ratio in it is a multiple of the *pre-raise* production cap, so a fresh run measures
convergence against the raised policy and would report ratios near 1. That is a different
measurement, not a refutation — which is why the file is durable rather than regenerable,
and why it is canaried. `[code: data/orbital/maxiter_convergence_ladder.json
(`not_reproducible_under_current_policy: true`); tools/orbital/measure_convergence_ladder.py;
tests/test_tracked_artifacts.py]`

## 8. What the instrument is blind to

These are limits, not future work.

**Per-atom richness only — it does not measure variety.** One source was rejected by eye
as "not varied enough" while scoring well; nothing in the tree measures within-source
variety. `[verdict: Matt]` `[unverified]`

**Composition.** Interior fraction, dead black, flat regions, subject placement. The
residual set scores above both references on both axes — `rings` 587.5–952.0 against the
references' 140.5/146.5, `range` 144.9–236.7 against 17.8/70.3 — and still reads as poorly
composed. **No scalar settles a composition call.** `[measured: 8 residual atoms at
320×180, clean cap, scratch/rescore/eye_vs_residual.json]` `[verdict: Matt]`

**The deep end, but far less of it than the screen run suggested.** 910 of the 4,669
enumerated atoms (19.5%) went unscored in the committed screen run; only **58 of them
(1.2% of the pool)** are actually below the `render-one` f64 spacing guard at screen
geometry, and the guard is the *screen's* geometry, not a property of the atom — the
unreachable set spans `log10|A|` to 17.79 against the screened set's 11.79. A 300-atom
concurrent re-attempt of the above-the-wall remainder scored **300/300** today. The other
~852 failures are unattributed: only a 10-error sample was persisted, and all ten happen
to be spacing failures (they fail instantly, so they arrive first in the error list). See
§10, correction 1. `[measured: data/orbital/{screen_pool,screen_scores}.jsonl,
screen_report.json; 300-atom re-attempt run for this doc under the raised cap]`

**One frame scale.** Every score was computed at 4× the atom's window scale (§2). None of
the validation transfers automatically to another scale. `[code: as §2]` `[verdict: Matt]`

**Absolute scores mean nothing across resolutions or across cap policies** (§5, §7). Only
orderings within one (resolution, policy) pair are comparable. `[measured: as §5]`

**Interior structure.** Rays are cut by NaN, so a frame that is mostly interior is scored
on what little escapes; `interior_fraction` is recorded but is not part of either measure.
`[code: tools/orbital/field_metrics.py::{radial_rings, measure_field}]`

## 9. Status against the thing it replaces

**It supersedes the q4 stage-1 screen `G` at sourcing. `G` remains a weak gate and a dead
ranker within its own accepts; do not invest further in it — the learned descent function
is meant to replace it.** `[verdict: Matt]` `[code: docs/design/minibrot_sourcing.md §3.1
for G's measured worth]`

**The supersession is a verdict, not a wiring change.** No committed code path routes
orbital scores into sourcing (§3); `G`'s pipeline is untouched. `[code: as §3]`

## 10. Test surface

`tools/orbital/test_orbital.py`, 19 tests, all passing. `[code: verified by running it]`

**Differential / behavioural** — these assert a *relation*, so they survive re-measurement:
- ring count scales with dynamic range (`hi > lo × 5`), a flat field scores 0, a central
  NaN island costs only its own span, `falloff_extent` is wider for a slow ramp than for a
  skin, `interior_profile` returns a fraction and an 8-bin curve — all on synthetic fields.
- `|Spearman(radial_rings, log10|A|)| < 0.6` over `measures.jsonl` — the population form of
  the "not depth in disguise" test.
- the screen geometry is ≥20× cheaper than the measure geometry.
- the cap-policy bracket: missing token reads as legacy, same-policy pools cleanly,
  cross-policy raises naming both sides, and the guard is **reached on both live paths**
  (`measure_atoms.validate` and `screen_pool.screen`'s resume) rather than merely defined.

**Frozen literals / artifact-pinned** — these go red on a legitimate re-measurement, which
is the intent:
- `ShadeArgs::density == 0.025` grepped out of `src/cli.rs`, and `fm.DENSITY == 0.025`.
- `validation.json`: `radial_rings` separates, 0 triage atoms at or above either reference,
  eye ≥ mb19; `cycles_spanned` and `falloff_extent` do **not** separate and >50 triage atoms
  beat the eye on `falloff_extent`. The negative results are pinned so they stay reported
  rather than being tuned away.
- every committed score record is stamped; `screen_pool.jsonl`'s exact key set is pinned so
  the day a rendered quantity is added to the enumeration, the stamping disposition is
  re-decided on purpose.

**Gaps, stated because a reader will assume otherwise:**
- **`rescore_lib.py` and `measure_convergence_ladder.py` have no pytest coverage.** The
  equality of the two crossings implementations is asserted only by `rescore_lib.py`'s
  `__main__` self-check, which no suite runs. `radial_range` has no test at all.
- **The `radial_rings_p90` failure at ×4 is not pinned** (§4).
- `test_measure_keeps_no_field_files` spawns the real engine — it is the only test here
  that needs `target/release/fractal-generator.exe`.

---

## Corrections made against the record this doc was written from

1. **"910 of 4,669 (19.5%) cannot be screened at all — f64 quantization wall."** False as
   stated. Only **58** of those 910 sit at or below the `render-one` spacing guard at
   screen geometry; a 300-atom concurrent re-attempt of the rest scored 300/300. The
   correct claim is that ~1.2% of the pool is wall-blocked and the deep tail
   (`log10|A|` > 11.79) is *entirely* wall-blocked, while ~852 failures in the committed
   run are unexplained and did not reproduce. §8.
2. **Screen-resolution validation figures.** "Spearman 0.873, 17/20 top-decile" is not
   reproducible from the tree — its producer was never committed. The reproducible
   equivalent over the 319 doubly-measured atoms is 0.857 (legacy) / 0.853 (clean cap),
   28/31 and 29/31. §4.
3. **"Top-100 overlap 12%, top-300 37.3%, top-1000 75.7%" for `rings` vs period** is a
   legacy-cap figure; under the clean cap the same population gives 9%, 38.7%, 74.6%. §6.
4. **"The per-source ranking under `rings` was byte-identical before and after the
   rescore."** The *ordering* is identical rank-for-rank; every value moves (e.g. `atlas`
   108.25 → 197.25). §5.
5. **"`range` is the better screening statistic"** stands, but not because `range` is
   resolution-stable: it is ×4.3 median full-res/screen against `rings`' ×6.0. Neither is
   resolution-invariant. §5.
