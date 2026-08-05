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

### The badness floor is gone too (2026-08-04)

The paragraph above is what the rule was between the q4 wiring and 2026-08-04. The floor is
now **deleted**: a floor-admit source takes **no machine quality cut at all**. The half-step
did not survive its own argument.

- **It kept the veto it was written to remove.** "Reject clear junk" is not a weaker claim
  than "judge quality"; it is the same claim at a lower threshold, made by the same head the
  source was selected independently of. On a `human_q3plus` row — a location Matt scored 3 or
  4 — a machine `p_notbad < 0.5` is the head disagreeing with Matt, and the floor resolved it
  for the head, silently, at intake.
- **The number never survived its own head.** `0.5` was chosen on the **v7** `p_notbad`
  scale and was still being applied under **v10**. Measured on the `q4_harvest` ledger's 108
  guard-passing rows (`data/emission/q4_harvest/`, v10 rescore sibling): the v7-era floor
  admitted **75**; the same `0.5` against v10 admitted **57**. An 18-row move in what the
  intake accepts, with no decision taken about it.

`FLOOR_PNOTBAD` was **deleted rather than set to 0.0** — a zero floor is still a floor, still
reads as a policy somebody chose, and gets re-tuned by the next person who finds it. Every
cut that remains in stage 2 lives in **`tools/emission/floors.py`** carrying the head and the
head *version* it was set against, and refuses to gate when the live pin disagrees.

### Where the branch lives (`descriptor.py`)

`load_admitted` factors the quality predicate through `admit_quality(row)`, which is
**source-aware**:

- `source_tag_of(row) in FLOOR_ADMIT_SOURCES` → **admitted** (no machine quality cut).
- otherwise → the q3 gate: `decoded_class >= 3`. (`>=`, not `==`: since v8 the head is K=4
  and a row can decode to class 4, which `== 3` would have rejected — silently, and precisely
  the best material.)

`guard_pass`, `distinct` and current-decode still apply to **every** source alike — the
bypass is of the *quality verdict*, not of the intake. That distinction is what
`test_intake_fail_closed.test_load_admitted_admits_the_seed_row_end_to_end` pins (a
guard-failing and a non-distinct floor-admit row are both still rejected).

**Two sources take the bypass**, and the second is the stronger case for the rule.
`q4_harvest`'s selection signal is the q4 goodness field; `human_q3plus` (the relit library
seed, `tools/emission/library_seed_v2.py`) is a HUMAN label of 3 or 4 taken with no decode
consulted at all. Gating either on the head's own verdict lets the head veto material it
never judged — with `human_q3plus` it would be vetoing Matt's own verdicts. Pinned by
`tools/emission/test_intake_fail_closed.py`, which derives the tag from
`library_seed_v2.MIX_SOURCE` rather than restating it.

**What it moved.** `q4_harvest` 57 → **108** admitted (of 108 guard-passing rows); the
seven-ledger stage-2 union 700 → **751** (`tools/emission/test_intake_union.py`). The other
six ledgers admit on the q3 gate and did not move, so the whole delta is attributable.

> **The `current-decode` conjunct is a firewall, not bookkeeping.** `is_current_decoded`
> (`scorer_version == active_ckpt.ACTIVE_VERSION`) is what makes a stale ledger *unreachable*
> rather than merely old, and that has already absorbed one real defect for free. When
> `descriptor.location_of` was found to resolve **walk-era julia rows** wrongly — reading the
> viewport from `outcome_*` when for a walk row `outcome_*` IS the parameter `c`, producing a
> degenerate location with no julia `c` at all — the blast radius was **zero**: all 1644 walk
> rows across 15 ledgers were decoded at v6 or unstamped, none at the then-active v7, so not
> one of them survived `load_admitted` to reach `location_of` in the first place. Persisted
> artifacts carrying the degenerate empty-`c` signature, scanned across all of `data/` and
> `out/`: **0** — direct evidence the wrong resolver never produced a durable record, rather
> than an argument that it could not have. (The one path that *does* carry walk-backed julia
> records, the library union, reads locations from the curated wallpaper pool, not
> `location_of`, and was independently verified byte-correct.) The general form: **a
> version-stamped admission gate bounds the reach of any bug in the resolution code behind
> it**, so a resolver fix that lands before the next checkpoint flip needs no regeneration.
> `[measured: 2026-07-26, 2060 julia rows = 1644 walk + 416 campaign, 0 untagged, over 15
> ledgers; verdict: record clean, no regeneration]`

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
