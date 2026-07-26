# Minibrot roster v2 + pilot harvest

Date: 2026-07-26. Code: `tools/sourcing/build_minibrot_roster.py` (Part A, durable
roster), `tools/sourcing/pilot_harvest.py` (Part B, pilot draw). Studies untouched —
the size-band-and-subsample draw this supersedes lived in the closed study
`tools/studies/q4_multibrot_transfer.py::source_nuclei`, not in `deep_center_finder`
(a pure library). Only the library is imported by the new production code.

## What was built

**Durable roster** (`data/minibrot_roster/roster.jsonl`, `roster_cells.json`):
degrees 2–5 × period bands {3-4, 5-6, 7-9, 10-12, 13-15}, target 8 atoms per
(degree, band) cell. Selection is **per cell**, not a global subsample, so degree is
never silently confounded with depth. The global size band is **replaced by an
`A`-based feasibility cut**: admit an atom iff `atom_instrument.f64_wall_margin_decades`
at the **deploy presentation** (1280×720 ss4, the emission wallpaper geometry) is
**≥ 1 decade**. One row per atom: degree/period/band, symmetry-canonical nucleus `c`
(lossless decimal), `|A|`/`log10|A|`, deploy + field f64 margins, dedup key, and a
**train/eval split** — atom-level (⇒ minibrot-disjoint by construction), 70/30,
stratified per (degree, band), seeded via sha256 (NOT `hash()`, which is
`PYTHONHASHSEED`-salted and would make the inherited split non-reproducible).

**Pilot** (~40 crops): draws the edges — every degree, shallowest + deepest *filled*
band per degree, the near-margin-boundary atom in each, plus the near-boundary
feasibility-excluded atoms — renders f64 fields, runs the **unchanged** deployed
stage-1 screen + OOD mask + G-maxima framing (reused read-only from the study), and
retains a durable manifest of crop coords + inherited split + fate
(`data/minibrot_roster/pilot/manifest.jsonl`). Sheet + JPGs → `scratch/`.

## Result: every cell filled, feasibility barely fired

```
TOTAL filled 160/160   under-filled cells: 0/20
admitted 160 (120 train / 40 eval)   feasibility-excluded-retained 3
admitted deploy-margin: min 1.11  median 6.66  max 9.17
```

## Three surprises

**1. Nothing under-filled — the prior "d4 tops near p10, d5 bimodal" was a sourcing
artifact, not a real ceiling.** That read came from the study draw's 72 ring seeds +
a global cap-to-12. With 960 seeds (96 ang × 10 rad) and per-cell selection, minimal
period-13–15 nuclei exist and are findable at **every** degree (d5 p13-15 still had 10
distinct deduped atoms available; d4 p13-15 had 14). Degree does not cap attainable
period the way the earlier cross-degree comparison implied — that comparison was
reading unequal *sourcing depth*, not an inherent degree property.

**2. The `A`-feasibility cut is currently a safety rail, not a selector — it fired on
3 of 163 atoms** (all d2 p14–15). The admitted roster sits a **median 6.66 decades**
inside the f64 deploy wall. Reason: ring-seeding preferentially converges to the
**large-basin (large-size) nucleus** at each period, so even the p13–15 atoms here are
the prominent ones (size ≈ 2.5e-5, log|A| ≈ 4.6), not the deep/small tail. So
"renderable in f64 at deploy geometry with a decade of margin" is essentially free
across p3–15 at every degree — the cut only bites if you deliberately source the deep
tail (or push period much higher). The a-priori predictor is nonetheless **validated**:
the two margin≈−0.01 excluded atoms' deploy-geometry crops fail f64 quantization on
render (9/9 attempted crop previews, pixel spacing 2.37e-14 < 1e-13), while the
margin≈0.95 excluded atom (`d2_p15_042`) and every admitted crop render clean — the
1-decade admission threshold carries exactly the headroom a sub-window crop needs.

**3. The stage-1 screen rejects shallow-band atoms outright and loves deep-band ones.**
Per-field accepted framings (of 4 kept each):

```
band 3-4  (all degrees):  acc = 0, 0, 0, 0        <- every shallow field: zero accepts
band 13-15:               acc = 4,4,4,4,4 (d5 p14 the lone 0), d2 p13 = 4 & 2
feasibility-excluded d2:  acc = 1, 4, 4           <- among the strongest fields sourced
```

A low-period minibrot is large and simple — its 4×size frame is dominated by the black
body + plain surroundings, which the d2-trained screen dislikes. A high-period nucleus
sits deep in a decorated (seahorse/spiral) valley, so the same relative frame is full of
filigree the screen was trained to accept. **Period is a strong predictor of
screen-accept here.** For the classifier corpus this is fine — the shallow bands
contribute rejects/coverage, and we want both fates — but if accept-yield were the goal,
the shallow bands are low-value and the deep bands carry it.

## Recommendation for the full run

- **The split and banding are sound; roll the roster forward as-is.** Fill is complete
  and the split is minibrot-disjoint and reproducible.
- **The feasibility-excluded d2 spirals are worth rescuing via perturbation.** The cut
  is the correct gate for d3–d5 (multibrot has *no* perturbation path — the f64 wall is
  absolute there), but the excluded atoms are all d2 (Mandelbrot), which *does* have a
  perturbation backend, and they are among the most beautiful fields sourced (radial
  spiral fans, screen-G up to 1.94, above the 1.39 cutoff). Route d2 near-boundary /
  excluded atoms through perturbation rather than discarding them.
- **If deep-tail atoms are wanted, ring-seeding won't find them** — it converges to the
  large-basin nucleus per period. Sourcing the small/deep tail (where the `A`-cut
  actually becomes a selector) needs targeted near-∂M seeding, not uniform rings.
```
