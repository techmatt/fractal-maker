# Julia parent-sourcing probe — is parent c-diversity the lever?

One question (`prompts/prompt_julia_parent_sourcing_probe.md`): does sourcing julia roots from
the **c-diverse near-∂M sampler** (instead of firing a hook off admitted c-plane parents) reduce
the rate at which julia candidates die as near-duplicates (`precanon_dup`) before they are ever
rendered? Baseline julia precanon depletion is 84–97% (`discovery_pipeline.md` §4); the hope was
that c-crowded parents were the cause.

**Verdict: no. Parent c-diversity is not the lever.** Sampler-sourcing left `julia:mandelbrot`
precanon_dup at **93.6%**, statistically indistinguishable from the **90.3%** hook baseline (same
era, same metric code). The near-dup churn is **intra-c z-plane self-saturation**, not cross-parent
c-crowding — so diversifying the parent c does nothing to it. Admissions did **not** collapse (the
failure mode §2 warns about); the sampler is a perfectly viable, non-barren supply. It simply does
not attack the dup rate, and as a *finite* pool it saturates within the hour (marginal cost rising).

## What was run (1 h, scheduler ON, prior ON — matches campaign-2/breadth exactly)

The only variable vs the committed campaign-2/breadth baseline is the **julia root source**:

- **Baseline** (hook-sourced): a `julia:X` partition is fed only by descending c-plane `X` and
  firing a hook on a qualifying admitted parent (`discovery_pipeline.md` §3). c-crowded by
  construction.
- **Probe** (sampler-sourced): the c-diverse near-∂M sampler (`q4_decisive_pass.py`
  boundary-rejection, arc-length-weighted, greedy 0.006 c-min-sep → 534 viable c's after the
  blob/dust **viability screen only**, no quality composite per §1) injected as `julia:mandelbrot`
  roots via the new `steered_frontier.py --julia-seed-pool` path. The hook stays available
  (secondary, §1) but all 6 hook attempts landed within 0.1 of a sampler c and were suppressed, so
  the julia supply is **100% sampler** — a clean contrast. Store: dedicated
  `data/discovery/julia_parent_probe/breadth/` (never touches campaign-2). Schema: every julia
  admission born `julia_schema=campaign` (asserted, 193 rows). Config: `--families mandelbrot
  --julia-hook --julia-hook-spacing 0.1 --freshness-prior --scheduler --budget 55 --seed 0`.

Scope: the sampler samples the **degree-2** ∂M, so it natively feeds `julia:mandelbrot` only.
This probe answers the lever question cleanly on that one partition (the mechanism generalizes);
`julia:multibrot{3,4,5}` would need a multibrot-boundary sampler — out of scope for an
hours-not-campaign probe. Run: 55.07 active-min, 154 batches, 188 admissions (~182 julia + ~6
c-plane), 534 julia roots (all injected, 0 hooked), 8025 renders saved to precanon_dup.

## Primary metric — `precanon_dup` rate, per partition (same readout code both arms)

`tools/atlas/julia_parent_probe_readout.py` applies one `precanon_admit()` function (a row is a
precanon dup iff `precanon_dup is not None`, mirroring `tau_h_retained_readout.part_curve`'s
`rendered = precanon_dup is None`) to both the committed campaign logs and the probe log, so any
arm difference is in the runs, not the counting. Both arms reconcile (harvest_log ↔ summary
totals ↔ ledger admits) and pass the julia birth-stamp assertion.

| arm / era | `julia:mandelbrot` precanon_dup | checks |
|---|---|---|
| baseline campaign-2/breadth **seg-B** (spacing 0.1 — probe-matched era) | **90.3%** | 494 / 547 |
| baseline campaign-2/breadth seg-A (spacing 0.2) | 94.1% | 1044 / 1109 |
| baseline campaign-2/dive | 91.3% | 950 / 1040 |
| **probe (sampler-sourced, spacing 0.1)** | **93.6%** | 7916 / 8458 |

The probe rate is **not lower** — it is 3.3 pp *higher* than the era-matched baseline. With
n=8458 checks this is not thin: the sampler did not move the dup rate off the ~90% plateau. The
hypothesis that c-crowded parents drive the depletion is refuted. (Consistent with
`discovery_pipeline.md` §4's note that the precanon filter is "correctly aimed at … residual
same-c julia descent": each distinct-c root still self-saturates its own z-plane — after a c's
first few distinct looks admit, every further z-viewport candidate for that same c falls inside an
admitted q3's seed-c-aware dedup radius and dies precanon. Distinct parents don't change that.)

## Secondary metrics — admissions did NOT collapse (§2 guard)

A falling dup rate can hide a sampler wandering into barren c. It didn't happen — and here the dup
rate didn't even fall, but the supply is demonstrably healthy:

| metric (`julia:mandelbrot`) | baseline seg-B | probe |
|---|---|---|
| admission rate among **rendered** candidates | 30.2% (16/53) | **33.6%** (182/542) |
| distinct looks admitted | 38 (whole campaign-2 run) | **136** (55 min) |
| price = active-min per distinct look (final EMA) | 0.403 | 2.127 |

- **Admit-rate among rendered is if anything better** (33.6% vs 30.2%) — the sampler's viable c's
  render into admissible julias at least as often as hook-sourced ones. Not barren.
- **Distinct looks: 136 in 55 min** — the sampler is a rich supply of *distinct* julia looks in
  absolute terms.
- **But the price rose to 2.127 min/look** (vs 0.403 baseline): the 534-root pool is **finite and
  saturates**. The online EMA climbs late-run as batches stop producing new distinct looks — each
  c is picked over. The hook mechanism, by contrast, is a *renewing* supply (fresh parent c's
  arrive continuously across a long run), which is why its marginal price stays low. The price
  gap is partly apples-to-oranges (different budget contexts / routing), so read it as directional:
  a one-shot sampler pool self-exhausts; it is not a drop-in replacement for the renewing hook.

Sheets (vivid `default` palette, `tools/atlas/julia_parent_probe_sheet.py`):
`scratch/julia_parent_probe/sheets/admitted.png` (all admitted julias, SMP=sampler) and
`.../rejects.png` (predup / canon-not-q3 sample) — the eye-check that the admits are real julia
"corner" looks and the rejects are genuine near-dups, not barren dust.

## Takeaway for the campaign

- **Do not** pursue parent c-diversity as a julia dup-reduction lever — it isn't one. The lever, if
  one is wanted, is **less per-root z-plane descent** (lower `M_CAP` / expansions-per-root): each
  julia c self-saturates fast, so the churn is renders spent re-covering an already-admitted c's
  set. That is orthogonal to this probe and untested here.
- The sampler **is** a valid julia supply (viable, non-barren, rich in distinct looks) — usable if
  a c-diverse, provenance-clean julia:mandelbrot seed set is wanted for its own sake — but it is a
  **finite pool that saturates**, not a renewing replacement for the hook. A larger pool or a
  refresh loop would be needed for sustained throughput.
- Numbers here are from a **1 h** run (user-shortened from the 3 h cap); the per-partition rate
  (n=8458) is well-powered and the verdict is firm. A longer run would mainly sharpen the
  saturation/price curve, not the primary finding.

## Artifacts

- Code: `steered_frontier.py` (`--julia-seed-pool` + `seed_julia_pool` + `mix_source`
  propagation); `production_seeder.near_dup` / `steered_frontier.load_prior_library_rows`
  (string-coord coercion — a latent freshness-prior crash on the newer string-serialized
  `q4_harvest`/`classic_phoenix` ledgers, fixed at ingestion).
- Readout: `tools/atlas/julia_parent_probe_readout.py` → `scratch/julia_parent_probe/readout.json`.
- Sheets: `tools/atlas/julia_parent_probe_sheet.py`.
- Run store: `data/discovery/julia_parent_probe/breadth/` (ledger + harvest_log + summary).
