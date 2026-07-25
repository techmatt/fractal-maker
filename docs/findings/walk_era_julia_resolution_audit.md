# Walk-era Julia resolution audit — verdict

**Question.** Commit `696dfa0` back-stamped `julia_schema` onto 2060 existing julia
rows and stated campaign resolution is byte-identical while **walk is "newly
correct."** Newly correct means previously *incorrect* — a behavior change, not a
no-op. Did anything live consume walk-era julia rows through the old (wrong)
`descriptor.location_of`, leaving a wrong location on record?

**Verdict: the record is clean. Nothing live consumed a walk-era julia row through
the old resolver.** No regeneration required. Details below; all counts are
read-only measurements (scripts were disposable, not committed).

## What the old resolver did to a walk row

Pre-`696dfa0` `location_of` always read the viewport from `outcome_*` and took `c`
from `julia_c_*` only if present. For a **walk** row that is doubly wrong: `outcome_*`
IS the parameter `c` (not a viewport), and `julia_c_*` is absent — so the old resolver
produced a degenerate location: **viewport = the c-point, `fw = outcome_fw`, and no
julia `c` at all** (key ends in `||`). The new resolver reads the real viewport from
`julia_z_*` and `c` from `outcome_*`.

Measured over all 15 ledgers: **2060 julia rows = 1644 walk + 416 campaign, 0
untagged.** Old-vs-new resolution:
- **All 1644 walk rows resolve differently** (`walk_differ = 1644`, `walk_same = 0`).
  The bug was real and total for walk rows.
- Campaign rows resolve identically (byte-identical), as the commit claimed.

Walk rows by ledger: `fresh_runs/overnight_20260713_001420` 123 ·
`fresh_runs/prospect_run1` 249 · `gather/mandelbrot` 288 · `gather/multibrot3` 324 ·
`gather/multibrot4` 288 · `gather/multibrot5` 234 · `discovery/outcome_ledger.jsonl`
138 = 1644.

## Why nothing live was affected — the decode-version firewall

Every walk row is **stale-decoded**: `scorer_version` is `v6` (1356 rows) or unstamped
(`None`, 288 rows). **Zero are `v7`.** Every campaign row is `v7` (416/416). The active
checkpoint is `v7` (`tools/scoring/active_ckpt.ACTIVE_VERSION`).

`descriptor.load_admitted` — the gate every ledger-consuming caller of `location_of`
passes through (`build_emission_diversity_v1`, `stage_first_release`, the morph
embed/cluster path) — requires `is_current_decoded` (`scorer_version == v7`). So **0 of
1644 walk rows survive admission** (verified per ledger). The old resolver could only
have mis-resolved a walk row if that row reached `location_of`; the current-decode gate
rejects every one of them first.

The consumers named in the brief, checked individually:
- **Emission / intake / occupancy readouts** (`build_emission_diversity_v1`,
  `stage_first_release`) — gated by `load_admitted`; 0 walk rows reach them. Both also
  write only under `out/` (disposable), never `data/`.
- **Morph dedup & clustering** (`descriptor.embed_locations` /
  `assign_morph_clusters`) — operate on `load_admitted` output; 0 walk rows.
- **Keeper derivation** (`data/atlas/keeper_cuts.json`) — per-family score thresholds
  calibrated on `eval_scores_v7.jsonl`; holds no locations. `location_of` never touches
  it.
- **The library union** (`data/library/library_records.jsonl`) — contains 25
  walk-backed julia records, but they carry the **correct** walk geometry (viewport =
  `julia_z_*`, `c` = `outcome_*`), verified byte-for-byte against the new resolver. The
  library builder reads locations from the curated wallpaper pool
  (`library_records_build.build`: `r["location"]`), **not** `descriptor.location_of` —
  a separate, walk-correct path. The 12 apparent key differences are a family-label
  convention only (`julia` vs `julia_multibrot{d}`); viewport and `c` are identical.
- **Contact sheets** — regenerable views under `out/`, not on record.

## Blast-radius, in counts

- Walk-era julia rows in the **current admitted, current-decoded (v7)** library: **0**.
  The admitted library's julia content is entirely campaign-era, whose resolution is
  byte-identical before and after the fix.
- All **1644** walk-era rows are confined to frozen, stale-decoded (v6/unstamped)
  ledgers that every live consumer rejects.
- Persisted artifacts carrying the old degenerate empty-`c` julia form, scanned across
  all of `data/` and `out/`: **0**. Had any v6-era emission run `location_of` on walk
  rows, it would have persisted exactly that signature; its total absence is direct
  evidence the wrong resolver never produced a durable record.

## Note

If the v7 checkpoint is ever flipped to re-decode these stale ledgers, the walk rows
would become admissible — and would then resolve **correctly** through the now-fixed,
tag-asserted `location_of`. The fix landed before that could matter. Nothing to
regenerate.
