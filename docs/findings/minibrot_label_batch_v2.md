# Minibrot label batch v2 (500-crop, two-arm) — build notes & findings

Batch `2026-07-26_minibrot_roster_v2` (487 crops) drawn off the durable roster
(`data/minibrot_roster/roster.jsonl`, 160 admitted atoms) via the **deployed** stage-1
screen / OOD mask / G-maxima framing (`tools/studies/q4_multibrot_transfer`, unchanged).
Builder: `tools/sourcing/build_minibrot_batch.py` (`screen` → `draw` → `render` → `report`).

## Realized draw

| arm | total | eval | train | notes |
|-----|------:|-----:|------:|-------|
| positive (accepted, G≥cutoff) | **237** | 55 | 182 | accept-limited (see finding 1) |
| negative (reject + OOD-masked) | **250** | 55 | 195 | on target |
| **total** | **487** | 110 | 377 | |

- **Deep-band negatives: 135/250 = 54%** (floor 50% ✓) — composed of **125 rejected
  near-misses + 10 OOD-masked**.
- **OOD-masked negatives: 25/250 = 10%** (target 10% ✓).
- **Eval slice period-matched: TV distance = 0.000** (pos vs neg period histograms
  identical, by construction — see below). Eval = 55 pos / 55 neg over 40 eval atoms.
- **Crops-per-atom:** max 6, 145/160 atoms used. Histogram (crops:atoms) —
  1:40, 2:13, 3:16, 4:29, 5:25, 6:22.
- Split **inherited** from the source atom (roster v2 70/30 per (degree,band)); never
  reassigned. 120 train / 40 eval atoms → the 6/2 per-cell split flows straight through.

Positives by band: `3-4:0, 5-6:14, 7-9:57, 10-12:76, 13-15:90`. By degree:
`d2:58, d3:66, d4:65, d5:48` (well spread across degrees).

## Findings (the surprises)

**1. The deep-band negative floor was fillable — but only after a real change to what
the screen caches, and this is the load-bearing finding.** The pilot cached only the
screen's top-4 G-maxima framings per atom and split them by the cutoff. For a *deep* atom
those top-4 are essentially all **accepts** (its best windows are good), so the pilot's
own numbers showed accepts at p13-15 and rejects at p03-04, **nearly disjoint** — exactly
the "deep == good" confound the prompt flags. Reusing that draw would have made a
deep-band negative floor nearly impossible: deep atoms produce ~0 rejects among their
kept peaks.

The fix (in `screen`, still using only the deployed screen's own functions): after the
kept-peak accepts, extract **sub-cutoff OOD-surviving windows** — the many structured
windows the screen surveyed and scored *below* the cutoff — and NMS them into up to 8
distinct "near-miss" reject framings per atom. Deep atoms have thousands of surviving
windows (`n_surv` ≈ 7-8k), plenty below cutoff, so **deep-and-structured negatives are
abundant** once you look past the top-4 peaks. Result: the deep floor is filled
**125-reject / 10-masked**, i.e. mostly by genuine structured near-misses, not by
featureless masked frames. Deep-and-bland windows are now in the corpus, and "deep" no
longer predicts "good".

**2. The positive arm cannot span all five period bands — the shallowest band has zero
accepts.** `positives by band` shows **3-4 → 0**. Every p3-4 atom (all four degrees)
screens to acc=0; the screen simply does not accept any window from the shallowest
minibrots. So "spread positives across all five bands" is unachievable, and the positive
arm came in at **237 (< 250)** because accepts are genuinely scarce outside the deep/mid
bands. This is a property of the deployed screen, not a draw shortfall — reported rather
than papered over by padding with low-quality accepts.

**3. "≤3 crops per atom" had to mean per-arm, and that is forced, not a convenience.**
Two independent reasons: (a) total-≤3 caps the batch at 160×3 = 480 < 500; and, more
importantly, (b) under a *total* cap a deep atom would spend its whole budget on accepts,
leaving nothing for the deep near-miss negatives — which would **reintroduce the finding-1
confound**. Per-arm ≤3 (≤3 accepts *and* ≤3 near-miss/masked negatives) lets one atom
supply both a positive and its own structured negatives. Max crops/atom is therefore 6,
but repetition *within a visually-coherent fate* stays capped at 3 (the pilot's concern
was one atom filling a sheet row at 8).

**4. Eval period-matching is exact by construction.** For the eval atoms, per period the
draw takes `n = min(available accepts, available rejects)` of each arm, so the positive
and negative period histograms are identical (TV = 0). The diagnostic the prompt wants
holds: on the eval slice, depth carries no signal separating the classes — a net that
scores well there learned quality, not iteration texture.

**5. Screen cost.** ~55 s/atom (the pure-Python `featurize` sweep over ~3 scales ×
thousands of window positions), ≈ 47 min for 160 atoms at 4 workers. Backgrounded,
resumable (per-atom JSON cache + cached f64 fields), atomic writes, per-dump hard timeout.

## Presentation

- Canonical (model-facing) crop: **1280×720, ss4, Lanczos3, q90 JPG**, per-crop deploy
  maxiter (`dcf._maxiter_for_fw`), seeded score-3 palette — the existing location-corpus
  label-crop spec (read off `build_enrich_batch` / `render_corpus_crop`).
- **Vivid companion: `cmr.fusion`** (vivid, cyclic blue/orange, already in the deployed
  score-3 roster) at `vivid/<image_id>.jpg`, shown beside the canonical in the labeling
  UI — the labeler judges from the vivid, the model eventually sees the canonical.
- Report + fate-stratified vivid sheet: `scratch/minibrot_batch/{distribution_report.txt,
  fate_sheet.png}` (negatives shown next to positives, incl. deep-vs-deep rows).

## Anchor batch (labeled first)

`2026-07-26_anchor_class4_v1` (60 crops): 52 already-labeled class-3 locations spanning
mandelbrot / julia / phoenix / native multibrot / julia-multibrot, rendered at their
**original** identity, re-labeled on 1–4 as **revisions** (amendment path), plus **8**
minibrot accepts spanning d2–d5. Fixes the class-4 bar across families before the minibrot
volume is labeled.
