# The sitting builder — how one labelling sitting is cut, ordered and served

A **sitting** is ONE cut, ONE manifest, ONE export, capped at `MAX_ROWS = 1000`, with everything
the record already knows to be worthless removed BEFORE a human sees it. It is not one batch: a
sitting may span several legs, each its own registered batch, and the cut runs once over their
union. The two owners:

- **`tools/atlas/sitting_cutter.py`** — cuts the population and draws the rows (`SittingSpec` /
  `SittingLeg` / `SITTINGS`).
- **`tools/corpus/build_combined_label_sheet.py`** — serves a BLIND sitting as one blinded sheet
  (`SheetSpec` / `SPECS`).

Both modules carry long docstrings that own their mechanisms, and both are pinned by injection
tests. This doc holds the decisions *across* them, and points at the code for everything the
tests already pin. Batch role annotations are `corpus_batches.md`; the split table is
`tools/scoring/batch_registry.py`; the 1–4 rubric is `data/label_corpus/CORPUS_SCHEMA.md`.

## 1. The three filter stages are NON-OPTIONAL, and that is the design

Entries in `STAGES`, walked unconditionally by `cut_sitting`. **There is no flag to skip one** —
a filter with an off switch is a filter that will be off on the run that needed it
(`verification_practice.md` §2). Each is proved red by injection in
`tools/atlas/test_sitting_cutter.py`.

| stage | rule | `[code:]` |
|---|---|---|
| (a) **interior auto-1** | `int_frac > 0.30` → auto-labelled `interior_gt30_v1`, **NEVER presented**. Matt's rule, firm: a frame more than 30% black is class 1 for wallpaper emission, no gray zone. A row with **no** measure is KEPT and counted apart — an absent measure is not a high one. | `sitting_cutter.stage_interior`, rule id/threshold/strict-`>` imported from `apply_interior_rule`, never restated |
| (b) **presentation morph-dedup at cos 0.974** | One row per LOOK, best-first. **NEVER a discovery gate** — the record keeps every candidate; what is thinned is what gets SHOWN. 870 labelled rows collapsed to 367 looks (2.37 labels/look), which is what makes the cap mean anything: a sitting's cost is denominated in looks. | `sitting_cutter.stage_morph_dedup`; cut is `supply_routing.NEAR_DUP_COS` |
| (c) **per-partition machine-1 discard** | ON for native multibrot and phoenix, OFF for julia:mandelbrot — the measurement is partition-dependent and the pooled number is not a decision: P(Matt=1 \| decoded 1) is 94–100% on mb3/4/5 and 72.0% on phoenix, but **30.9%** on julia:mandelbrot, where 16.5% of machine-1s are ≥3. Every partition with no measurement of its own fails **CLOSED to KEEP**. | `sitting_cutter.stage_machine_1`; table is `supply_routing.MACHINE_1_DISCARD` |

**Order is cost-descending in the other direction.** (a) and (c) are free reads off columns the
record already carries; (b) needs a render and a CLIP pass per surviving row. So the two free
stages run first and the expensive one sees the smallest population — reversing them would be
*correct* and would cost a morph field for every row the other two were about to delete.

Two boundary details worth not re-deriving: stage (c) acts only on a **canonical** decode
(`rank_tier ≥ 2`) — the cheap score comes off a 384×216 ss1 render while the measured
P(Matt=1 | decoded 1) rates were all taken against the 640×360 ss2 canonical decode, and treating
the two as one number is the cap/geometry error. And "machine class 1" reads the raw probability
(`canon_nb < floors.NOTBAD_CUT`), **not** a stored class: `canon_decoded` stopped being able to
say 1 on 2026-08-09, and reading `floors.good_class` instead would silently widen a narrow
"the head is confident this is BAD" discard into "everything below the good floor", i.e. the
whole of class 2.

## 2. The calibration reservation, and `MIN_POS`

A partition whose **labelled positives** (human ≥3, amendment overlay applied) sit below
`MIN_POS = 15` is one nothing can be said about, and no amount of discovery fixes it, because
the missing thing is human labels. Left to the balanced draw such a partition earns close to
nothing and stays unmeasurable forever, so `cut_sitting` **reserves** a slice for it
`[code: sitting_cutter.plan_reservations / draw_reserved]`.

- **`MIN_POS` is OWNED BY `sitting_cutter` now** (`sitting_cutter.min_pos()`). It used to be
  imported from `derive_t_good.MIN_POS` precisely so a second literal 15 could not diverge; that
  estimator was deleted 2026-08-09 with the rest of the per-partition `t_good` machinery, and
  this module **inherited** the number rather than losing it. Its gate-path role died with
  `t_good`; what survived is a statement about the LABEL CORPUS — enough human keepers in a
  partition to say anything about it — which outlived the threshold sweep that used to ask it.
- **A general rule, not a `phoenix:classic` special case**, and it **lapses per-partition by
  itself**: the qualifying set is recomputed from the live corpus at every cut, so a partition
  that crosses `MIN_POS` stops being reserved without anyone editing a list. Active today:
  `phoenix:classic` alone (7 positives); mandelbrot at 626 gets nothing.
- **Bounded on three sides** — `RESERVE_FRAC` 0.05 per qualifying partition, `RESERVE_CAP_FRAC`
  0.15 across all of them (split evenly once more than three qualify, so the cap binds by
  shrinking each share rather than dropping a partition off the end and silently picking a
  favourite among equally starved families), and **SUPPLY**. Truncates rather than rounds, so
  the cap is hard. An unfillable reservation records its shortfall and the balanced draw fills
  the slot from elsewhere; it never fails the build.
- **PRESEEDED into the fill's round-robin, `max(natural, reserved)` — never additive**
  (`apportion.deal_round_robin(..., preseed=...)`, which is what makes a reservation a FLOOR
  rather than a bonus). A reserved cell with no remaining supply still enters `sizes` at 0,
  because `deal_round_robin` refuses a preseed cell it cannot see.
- **⚠ A BUCKETED CUT DOES NOT ALSO RESERVE.** The reservation is a fill-path mechanism, so every
  bucketed sheet skips `phoenix:classic` by construction. It needs its own explicit bucket, or a
  non-bucketed sitting. `[found 2026-08-07]`

## 3. Apportionment sequencing, and its population-dependent bound

The sheet's order is a seeded shuffle over the union, then **`apportion.sequence_by_deficit`**
over every `(source_batch × family)` cell, so any PREFIX is near-proportional. Without it the
union is N contiguous blocks and the first hours of a sitting are one source's material — a bar
that drifts against provenance. The file order IS that order, so the page must not reshuffle:
`batch.json` records `presentation_order: "file"` and the rig is served `&order=file`.

**The ±1 prefix bound is a CHECK EACH CALLER RUNS, not a theorem.** It is provably tight for two
cells and holds comfortably on the sheet's real population (0.791 on the built sitting), but with
many cells and two-orders-of-magnitude supply skew it exceeds 1 — a frozen 13-cell
counterexample reaches **1.495**, and 42% of randomly drawn skewed 15-cell populations exceed 1.
So `build_combined_label_sheet.stage_verify` **asserts the bound on the order it built**
`[measured 2026-08-05, tools/test_apportion.py]`. Do not simplify that check away and do not
restate the bound as a guarantee of `apportion.py`. The rule choice matters as much as the
check: the cheap "lay each cell at `(i+0.5)/n_c` and sort" dialect — the round-robin generalized
to a sequence — reaches 1.068 where the deficit rule reaches 0.738 on the sheet's own cell shape,
and `test_apportion.py` keeps that comparison as a **live control**.

## 4. Two serving paths, and why the correction sheet cannot use the blind one

**Blind sitting** → `build_combined_label_sheet.SheetSpec`: a separate directory, seeded shuffle,
**opaque ids assigned POST-shuffle** (so id order is presentation order and encodes nothing),
**blinding by ABSENCE** — the whole `provenance` block DROPPED rather than emptied, because it
carries `batch_id` on top of every selection key — and a `route.json` that sends the sitting's
single `labels.json` back to each row's own registered batch (`merge_scores.py --route`).
The sheet dir carries **no `images.jsonl`**, deliberately: every corpus consumer discovers
batches by globbing `*/images.jsonl`, and a sheet carrying one would union rows that already
exist in the source batches and double-count every label in training. That absence is held as a
tripwire by `test_combined_label_sheet.py`.

**Correction sitting** → served straight off its registered batches by the rig,
`SittingSpec.serve_url()` (derived from the legs, never written out). It cannot use the blind
path and that is not an oversight: **every property that module exists to enforce is the negation
of what a correction sheet is.** It shows the head's decode where the sheet drops the score
columns; it is ordered good→bad by that decode where the sheet deals a seeded apportionment
*specifically so the head's ranking cannot anchor the labeler*. Adding a non-blind mode to the
blinding module is exactly the drift its own docstring warns about.

**The invariant that does not bend either way: A SUGGESTION IS NOT A LABEL.** `label.score` stays
null on a served correction row and the merge refuses to read the suggestion; unreviewed
suggestions never leave the page as labels.

A sitting is a **presentation merge and never a registration** — the thing registered is the
population a row was *generated by*. Two generation methods in one sitting are two registrations
(`SittingLeg`), or the leg becomes unrecoverable from the corpus afterwards.

## 5. ⚠ The opaque-export row-count rule

**Verify every export merged BY ROW COUNT, never by the guard.**
`tools/corpus/test_label_reachability.py` reconciles on count reachability across `labels/*.json`
— but an **opaque-keyed** sheet export has zero keys matching any `label_corpus` `image_id`, so
it falls out of scope **by content** and a never-merged export reads identically to a merged one.
The 2026-08-04 export audit reconciled all 50 sidecars by row count and found zero unmerged
*including the 12 opaque-keyed files the guard cannot see* — the audit is what closed it, not
the guard, and **the gap stays open for FUTURE exports.**

Adjacent and enforced: **`labels/` holds ONLY label-corpus sidecars.** `label_store.load_sidecar`
walks every `labels/*.json`, so a foreign-schema file there is a loud red (proven 2026-08-04 by a
misfiled calibration export). Non-corpus judgment records live under `data/`.

## 6. The instance-in-a-frozen-dataclass pattern

Both builders keep the *instance* — which batches, which seed, which id prefix, which run dir —
in a frozen dataclass (`SittingSpec`/`SITTINGS`, `SheetSpec`/`SPECS`) and the *rules* at module
scope. That is why a second sitting is an entry rather than a refactor: `sitting_cutter` once
held its batch id and seed at module scope and needed a refactor before a second sitting could be
built, while `build_combined_label_sheet` had already solved the same problem. Both specs also
**pair every draw constraint with its stated reason** and refuse construction otherwise
(`population`/`population_rule`, `no_pad`/`no_pad_rule`, `row_filter`/`filter_rule`) — a subset
whose rule is not written down is indistinguishable from a lossy build.

`--embed-limit` on **`draw`**, not only on `dry-run`, is the bounded-end-to-end rule
(`CLAUDE.md`): a bounded run that writes real files stamps `sitting_cut.INCOMPLETE = true` into
every `batch.json` it produces, derived from the flag at the write site.
