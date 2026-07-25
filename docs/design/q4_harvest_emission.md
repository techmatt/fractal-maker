# Wiring the q4 tight harvest into emission (`q4_harvest` source)

## What this is

The q4 tight harvest (`tools/studies/q4_harvest_tight.py`) selects palette-free minibrot
framings by the **q4 goodness field** `G` (a refit T2 model over per-minibrot position×scale
fields), gated at a label-derived high-precision cutoff (`G >= ~1.39`). It delivers
`scratch/q4_stage1/harvest_tight/candidates.json` — ~116 framings over ~29 minibrots, each a
`(cx_win, cy_win, fw_win, maxiter, minibrot_id, scale, G)` record.

This doc records how that harvest becomes a first-class **emission source** flowing through the
built stages (intake → cells/deficit → colorize → gate/pool → select), and the one design
decision that makes it correct.

## The load-bearing decision: v7 is a FLOOR here, not the quality gate

The emission driver's default intake predicate (`descriptor.load_admitted`) admits a discovery
row on **`decoded_class == 3`** — v7's own q3 verdict. That is self-consistent for a discovery
source, because those locations were *found by v7* (the guided-descend reward IS v7's q3
verdict), so the selection signal and the admission gate are the same model.

The q4 harvest is different. Its locations were selected by the **q4 goodness field, which is
orthogonal to v7** — v7 is blind to q4 quality and to the window labels. Gating q4 rows on
`decoded_class == 3` would let v7 silently veto locations it never chose (measured below: the
q3 gate keeps only a fraction of the guard-passing q4 framings). That is the wrong bar.

So for a **floor source**, v7 serves only as a **badness floor**: admit iff
`p_notbad >= 0.5` (= `P(class >= 2) >= 0.5`, i.e. "not clearly bad") ∧ `guard_pass` ∧
`distinct` ∧ current-decode. The human does the actual quality pick off the release sheet.

### Where the branch lives (`descriptor.py`)

`load_admitted` now factors the quality predicate through `admit_quality(row)`, which is
**source-aware**:

- `source_tag_of(row) in FLOOR_ADMIT_SOURCES` (currently `{"q4_harvest"}`) → floor:
  `p_notbad >= FLOOR_PNOTBAD` (`0.5`).
- otherwise → the q3 gate: `decoded_class == 3`.

`guard_pass ∧ distinct ∧ current-decode` still apply to **every** source. The branch is
backward-compatible: no existing ledger carries the `q4_harvest` tag, so every prior intake is
byte-identical. The tag is the durable per-row `mix_source` (falls back to `_source_tag`), the
same field the driver already reads for source-tag measure overrides.

## The producer (`tools/emission/q4_harvest_ledger.py`)

The mandelbrot analogue of `tools/phoenix/classic_phoenix_supply.py`. For each harvest
candidate it renders the fixed framing at reframe/deploy fidelity (640×360 ss2, mandelbrot, no
reframe search — the harvest already fixed the framing) with the co-located guard field, reads
the guard verdict off that field, and guarded-v7-scores. Reuses the exact production
primitives: `reframe._render` (with `DUMP_GUARD_FIELD`), `guard.make_guarded_scorer`,
`score_lib.corn_decode`. `decoded_class` is **computed and stored** (for the readout) but is
NOT the admission gate.

Row schema mirrors a discovery ledger row (`family="mandelbrot"`, `outcome_cx/cy/fw`,
`p_notbad`, `p_good`, `decoded_class`, `guard_pass`, `distinct`, `scorer_version="v7"`) plus a
`mix_source="q4_harvest"` tag and q4 provenance (`q4_minibrot_id`, `q4_scale`, `q4_G`, `q4_box`)
that rides along for the readout but never enters training or admission.

**`distinct` is `True` for every guard survivor.** The harvest already elliptical-NMS-deduped
the framings; the REAL morphology dedup — *incremental medoid within type*, cos 0.974 — is the
emission driver's own intake clustering (`descriptor.assign_morph_clusters`). This ledger seeds
that pass, it does not pre-empt it. Hence the two distinct stage counts "floor-admitted →
distinct clusters".

Durable outputs under `data/emission/q4_harvest/` (survive `rm -r scratch/*`):
`rescored.jsonl` (per-candidate decode, guard-pass AND guard-fail; resume key),
`outcome_ledger.jsonl` (guard-passing rows, intake-ready — the floor is applied by the driver's
source-aware `load_admitted`), `stats.json` (per-condition counts).

Decode tiles + guard fields are transient (`scratch/emission/q4_harvest_decode/`, per-candidate
wiped). They are ~116 small 640×360 JPGs, not a file-count bomb, so they stay under the
disposable `scratch/` tree rather than being routed out-of-tree via the `ARTIFACTS_ROOT` resolver
(that seam is reserved for the regenerable ML aug-cache bombs; a new one was not created here).

## Running it

```bash
# 1. harvest candidates (regenerates the q4 field stack under scratch/ if wiped)
uv run python -m tools.studies.q4_stage1_labelset minibrots
uv run python -m tools.studies.q4_stage1_labelset fields
uv run python -m tools.studies.q4_harvest_tight build

# 2. render + guarded-v7-decode → data/emission/q4_harvest/outcome_ledger.jsonl
uv run python tools/emission/q4_harvest_ledger.py

# 3. emission driver over the q4 source alone (floor-admit is automatic via the source tag)
uv run python tools/emission/build_emission_diversity_v1.py \
    --ledger data/emission/q4_harvest/outcome_ledger.jsonl \
    --out scratch/emission/q4_harvest --cover-all --release-n 24

# 4. both-heads release + reject/pool autopsy sheet + per-stage counts
uv run python tools/emission/q4_harvest_readout.py
```

## Two-head routing (unchanged) and the readout

Head routing is automatic and orthogonal to type: `smooth → wallpaper head` (v3, 0.90 gate),
each promoted strange mode → `mining head` (v1, 0.50 gate). The mining head is **uncalibrated on
strange** minibrot renders, so the readout (`q4_harvest_readout.py`) surfaces **both** head
scores for every release candidate and pool row — Matt judges strange by eye. The readout does
not change routing; it just scores each tile with the non-routed head too.

**Measured head behavior on strange (verified 2026-07-25, `q4_readout.json`, n=8 strange release
rows).** A recurring claim that the *mining* head scores strange near zero is **inverted**: on
these renders the mining head is the stable one (p_ge3 mean **0.43**, min **0.25** — never near
zero), while the smooth-trained **wallpaper** head is the one that collapses strange to literal
**0.000** (3 of 8; mean 0.35). The wallpaper head never gates strange, so that collapse is
inert. The real caveat is not "near zero" but "uncalibrated": the mining head's strict 0.50
release floor would cut 6 of 8 strange candidates — which is exactly why the readout draws from
the whole gated pool and lets the eye decide rather than truncating to the strict-floor subset.
The mining gate is therefore left **as-is** (not neutered) pending mining-head calibration.
