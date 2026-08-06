# Mining adoption — 2026-08-06

`prompts/mining_adoption_prompt.md`. v1 keeps the pin (it won the winner rule); the three
adoption changes are made. Run dir is this one — the pinned head's — because everything
written here is about the head the pin serves, not about the candidate that lost.

Nothing was re-measured for this sitting. Every number below is quoted from
`mining_gate_lock.json`, which is derived from `data/render_mode_head/v2/report.json`, which
is the committed record of the v1-vs-v2 sitting.

---

## 1 · the mining release floor now cuts

`mining_release = 0.50` was report-only; it is enforcing, at the same value, stamped to
`render_mode_head/v1`. `mining_pool = 0.25` is unchanged in value, stamp and behaviour.

**Two sites apply this cut, not one, and both were flipped.** `floors.py` is the owner, but
the owner only decides for the sites that ask it. The emission driver
(`build_emission_diversity_v1.release_eligible`) carried a mining-specific branch that
admitted every scored strange row; that branch is deleted rather than parameterised, so one
rule now serves both heads. `deploy_tail.py` — the on-demand pass that ships strange
*alternates* alongside emitted smooth wallpapers, i.e. the second path where a strange render
becomes product — carried its own copy of the report-only decision in prose and in a
`allocate_strange(cands, …)` call. It now reads `floors.MINING_RELEASE.acts` and allocates
over the passers. Leaving it alone would have made "enforcing" false for half the strange
output while every readout said otherwise.

**Stamped-head refusal still holds.** `Floor.gate` refuses on a pin mismatch exactly as
before — `acts` and the stamp are independent fields, and the flip touched only the first.
`test_floors.py` proves the refusal for both mining cuts and proves the wallpaper cuts do not
refuse in the same injection, so a blanket raise cannot pass. The new
`test_mining_gate_lock.py` adds the same refusal at the record layer.

**The liveness census reads correctly.** It was `[f.name for f in ALL_FLOORS if not f.acts]
== ["mining_release"]`; it is now `== []`, plus `"report-only" not in summary()`. An empty
census is also what a census reading the wrong field returns, so a mirror test injects a
report-only cut and requires the same expression to name it.

**`--target-gated` semantics read correctly, and one accounting key changed.** The target
already counted `post_floor()` — release-eligible AND above the head's release floor — so
**the target's cost per run is unchanged**. What changed is that `post_floor()` is now an
identity on `release_eligible()`: eligibility applies the same floor. Rather than let a
computed identity sit there being read as a measurement:

- `post_floor()` is kept, kept separately computed, and the identity is *asserted* by a test.
  If any release floor goes report-only again the two sets diverge and the accounting reports
  the gap instead of silently counting rows no floor vouched for — the exact bug the flag was
  fixed for.
- `ungated_strange` (which would now be a constant 0) is replaced by two keys:
  `ungated_eligible` — eligible-but-below-floor, 0 while every floor acts, and the visible
  signature of a floor having gone report-only — and `cut_by_release_floor_strange`, counted
  from the **pool**, which is the population the flip actually stops shipping. A test
  simulates the report-only shape and requires the split to reappear.

Sub-floor strange rows are still **pooled**. The release floor decides what ships; the pool
floor decides what is kept as inventory, and 0.25 < 0.50 is checked at import
(`check_below_gate`).

## 2 · gate_report accrual — it kept accruing, and it had accrued nothing

`gate_report.py` is untouched in shape and still writes at **both** sites: the emission
driver logs every scored strange candidate against the release floor *and* the pool floor
(it reads `self.pool.rows`, not the eligible set, so the denominator is unchanged), and
`deploy_tail` logs its candidates at the release site (it has no pool stage). Verified by
reading the writers, not by inference: neither log's input set is derived from
`release_eligible()`.

**What the flip costs.** Both outcome joins are now zero *by construction*: selection implies
clearing 0.50, which implies clearing 0.25. So `would_cut ∧ selected` — the free labeled
false-cut count that was the whole point of report-only — and `would_cut_pool ∧ selected` can
never be non-zero again. This is stated in `floors.py`, `gate_report.py`, both writers and
both run banners, because it is the kind of fact that is otherwise rediscovered as "why is
this column always 0".

**What that cost actually is, measured:**

```
data/emission/mining_gate_reports/   does not exist
data/emission/release_records/       does not exist
```

**Zero rows, at both sites, ever.** No emission run has completed since these sinks landed,
so the report-only period accrued no calibration signal at all. The flip gives up a
mechanism that had produced nothing, and the calibration that justified the flip came from a
labeled sheet instead — which is the mechanism that worked.

## 3 · the new gate-lock record

`data/render_mode_head/v1/mining_gate_lock.json` (+ `mining_gate_lock.md`, its readable face,
generated from the same derivation — not this file, which is hand-written and which a
`--write` must never clobber). Durable class through `paths.durable()`, negated by exact path
in `.gitignore`, canaried in `tests/test_tracked_artifacts.py`, declared in
`durability_map.py`.

The predecessor could not be re-run: it derived its curve from
`data/render_mode_head/v1/seed_*/eval_scores.jsonl` over
`data/render_mode_corpus/dataset_v1/eval.jsonl`, and **both are gone**. So
`lock_mining_gate.py` was rewritten to derive from the committed sitting record instead —
pure Python, no torch, byte-identical on a re-run. It holds:

| | |
|---|---|
| head + batch identity | `render_mode_head/v1` @ `model_best.pt`, `2026-08-06_render_mode_fresh_sheet_v1`, eval side, n = 422, 48 locations, 15 modes, labels {1: 246, 2: 113, 3: 63} |
| frozen ladder | **both** boundaries whole (20 rows each), not just the cut rows — so "what would 0.40 have bought" is answerable without a sitting whose crops may be gone |
| the two cuts | `mining_pool` 0.25 → fires 70/422 (16.6%), precision **75.7%** [64.5%–84.2%], recall 84.1% · `mining_release` 0.50 → fires 33/422 (7.8%), precision **97.0%** [84.7%–99.5%], recall 50.8%, against a 14.9% base rate |
| the two caveats | v1 trained at these same 112 gate-passer locations; the labels were prefilled with v1's own suggestions (correction sheet, sorted good→bad, Enter confirming) |
| the bound | both caveats inflate v1 and neither is subtractable, so **every precision is a ceiling on a fresh location, not an estimate** |

Instead of measuring, it **refuses**: the sitting must have calibrated the checkpoint the pin
serves, each cut must equal the owner's live value, and each cut must be an *exact* swept row
of the ladder (never nearest-bin). `read_lock()` refuses when the live pin no longer matches
the head the numbers were measured on — the same refusal `Floor.gate` makes, for the same
reason. The emission driver's colorize banner reads it, so a run that gates strange prints
what its cut is measured to buy, and a pin move fails the run rather than the reader.

Frozen-record write rule: a default run **verifies** and exits 1 on drift; `--write` writes.
A test regenerates the committed record in memory and requires byte equality.

The July operating point (precision 0.548 / recall 0.195 at 0.50, base 0.139) is *not*
superseded by this: it was measured on a genuinely held-out, independently-labeled population
that no longer exists. The record names it as such, and `fresh_sheet_reads.JULY_LOCK` still
quotes it unchanged.

## 4 · v2's weights are untracked

v2 lost the winner rule (significantly worse at both `>=2` boundaries on a 4,000-draw paired
bootstrap; no significant gain on the three dropped modes), and a rejected candidate is not a
critical final weight. Removed: the `.gitattributes` LFS rule, the `.gitignore` negation, the
`tests/test_tracked_artifacts.py` canary line, the `tests/test_large_tracked_blobs.py`
allowlist entry; `size_guard.py` and `durability_map.py` updated. `git rm --cached` applied;
the file stays on disk as working state.

**The run record is untouched and stays tracked** — `config.json`, `metrics.json`,
`per_seed.json`, `report.md`, `report.json`. `report.json` is now *also* canaried, because
the gate lock is derived from it: it went from "the deliverable" to "the input the live
floor's justification is computed from". Rollback for v2 is re-running
`classifier/train_mining_head_v2.py`, not a `git checkout`; the trainer, the corpus batch and
the recipe in `config.json` are all committed, and the stated deviations from v1's recipe are
pinned by a test.

## 5 · post-flip pass-rate arithmetic

Rates are the head's own, from the frozen ladder. **They are a rate on the sheet's
population — gate-passer minibrot locations at label geometry — and the emission pool is a
different population at a different render geometry.** No emission run has ever written a
record (§2), so the driver's realized strange pass rate is genuinely unknown; these are the
only rates that exist.

At a strange colorize volume S:

| | before | after |
|---|--:|--:|
| pooled (≥ 0.25) | 0.166·S | 0.166·S (unchanged) |
| release draw pool | ≈ S (every scored row) | 0.078·S |
| counts toward `--target-gated` | 0.078·S | 0.078·S (unchanged) |

So the target's cost per run does not move; the **selection draw pool** does, by ~12.8× (from
every scored strange row to 7.8% of them), or 2.1× measured against the pooled set.
Concretely, at the default `--strange-frac 0.5`:

| release N | strange slots | strange colorizes to FILL | to reach the 3× surplus |
|--:|--:|--:|--:|
| 12 | 6 | ~77 | ~230 |
| 24 | 12 | ~153 | ~460 |

Under report-only, filling 6 strange slots needed ~6 scored strange rows. A run at the old
colorize budget will therefore **short-fill the strange half** — which is reported, not
silent: `release_short_fill` already carries `requested / eligible / selected / short_by`, and
`cut_by_release_floor_strange` now says how many pooled strange rows the floor removed. What
each released strange row buys in exchange, on the sheet's population: precision **97.0%**
[84.7%–99.5%] instead of ~everything-that-scored — at a recall of 50.8%, and optimistically.

**The honest expectation**: the first post-flip run either colorizes ~12× more strange, or
ships a short-filled strange half. Both are visible in the run's own report; neither is a
silent degradation. If the short-fill is the wrong trade, the lever is the floor value, and
the ladder to pick a new one from is frozen in the lock — 0.40 buys 92.9% [81.0%–97.5%] at
10.0% pass rate, 0.30 buys 81.7% [70.1%–89.4%] at 14.2%. None of the 70/80/90% precision
targets is supported by its Wilson *lower* bound at n = 422, which is the reason nothing else
was moved today.
