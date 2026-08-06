# Verification practice — how a guard fails, and how to stop it

**Consult this before writing any test, guard or gate here.** It is the checklist the
repo's own green-and-useless guards were made of. Companion: `measurement_practice.md`
(designing a measurement); `retired.md` (has this been tried).

Two questions to ask of anything you are about to write:

1. **How would this fail OUT LOUD?**
2. **Could this pass for a reason other than the one it claims?**

**Tags.** `[code: path]` — true because the tree says so. `[verdict: who]` — a judgement.

---

## 1. Eleven ways a guard ships green and useless

1. **Measures the wrong quantity.** A naive `git cat-file -s` on an LFS path returns ~130
   bytes for the pointer, so a size guard reports a 92 MB `cache_manifest.jsonl` as tiny —
   inverting exactly the thing being guarded. `[code: tests/test_large_tracked_blobs.py]`
2. **Skips as pass.** §2, the most repeated defect here.
3. **Blind below its cursor.** A screen that caches only its top-N maxima per unit cannot
   see the sub-cutoff population, and then "deep" becomes a free predictor of "good" by
   construction. `[code: minibrot_sourcing.md §7]`
4. **Red so long it is invisible.** An emptiness *warning* fired on 15 registry lines every
   run; a permanently-soft red is trained out exactly like a permanently-hard one. Nothing
   in a red lane is protecting anything. `[code: artifacts_resolver.md §4]`
5. **Crashes as something else.** `/emit` validates that a session entry exists but never
   that its files do, so a `rm -r scratch/*` mid-session raises `FileNotFoundError` from
   `copyfile` — a 500 — instead of the harness's own "re-render" 400.
   `[code: storage_classes.md, the descent-harness clause]`
6. **Cross-contaminates.** One label sidecar shared by two batches needs a join rule, or a
   foreign corpus's rows enter the scan. `[code: tools/corpus/label_store.py::SIDECAR_OWNER,
   FOREIGN_LABEL_FILES]`
7. **Lives only in prose.** `docs/findings/` was retired in a commit message with no
   tripwire; five later runs recreated it. `[code: tests/test_docs_tree.py]`
8. **Harness constant diverged from production.** Two independent copies of the
   `auto_maxiter` closed form exist and are pinned to agree by a test, precisely because
   nothing structural keeps them equal. `[code: tools/scoring/test_maxiter_policy.py]`
9. **Computed at insufficient precision to discriminate.** A statistic quantized below the
   difference it must resolve reports agreement it never checked. Measured instance: the
   wallpaper-v3 gate-passer population is **403** rows under `torch.autocast` and **401** at
   fp32 — two rows sit close enough to `p_ge3 == 0.90` that fp16 accumulation moves them
   across it. Autocast is fine where a score RANKS and wrong where it CUTS.
   `[code: tools/mining/build_gate_passers.py::score_marginals]`
10. **Ground truth that reconstructs itself through the code under test.** A test that
    injects the dependency, or derives its expectation from the same helper the subject
    uses, is asserting `f(x) == f(x)`.
11. **Passes for the wrong reason.** The v9 diagnostic arm returned v8-on-new = v8-on-old =
    **exactly 0.0000** — not a null result, a measurement of nothing: all 144 census tiles
    were byte-identical between the two caches. `[code: auto_maxiter.md, "Why v9 is shelved"]`

## 2. An absence-tolerant guard un-guards exactly when its subject is removed

**The most repeated defect in this repo.** A `.exists()` skip cannot tell *not fetched*
from *deleted on purpose*, and a gate that degrades to silence cannot protect a deletion of
its own input. A crash is a message; a skipped gate is a metadata file that outlives the
fact it records.

- **Legitimate absence is a HARD failure that names the rebuild command.** The dead-checkpoint
  pins now raise from `require_ckpt` naming the missing file rather than resolving to
  nothing. `[code: tools/mining/test_require_ckpt.py]`
- **If the referent is gone, DELETE the test.** Do not let it skip. `occupancy_parity.rs`
  went with its counterpart rather than becoming a permanent skip.
- **A retired guard whose input was deliberately emptied keeps its MECHANISM tested via an
  injected value**, with the revival condition recorded beside it.

Live absence-tolerant sites, named so they are decisions rather than misses. **Re-derived
2026-08-06** by running the default lane with `-rs`, because the previous version of this
list had rotted in both directions — it named a `test_v8_cache_alignment.py` module fixture
that no longer exists (that file now skips nothing; its docstring says so) and credited
`test_v9_staging.py` with two `skipif`s where it has one, which does not fire here. A list of
tolerated absences that is itself unchecked is the §5 rotted-allowlist defect aimed at §2.

Exactly **two** tests skip in the default lane on this checkout:

- `tools/emission/test_emission_diversity.py::test_location_ranker_cache_hit_matches_direct_scoring`
  — needs `data/ranker/pref_loc_v0/{model,features}.npz`. **Legitimate**: `pref_loc_v0` is
  live (≈10 importers) and the path is a declared durable in `tools/audit/durability_map.py`,
  so this is *not fetched*, not *deleted on purpose*. It is the PRESENCE-FROM-DISK arm of §6's
  two-test pairing; the absence arm runs unconditionally.
- `tools/atlas/test_julia_seed_pool.py::test_committed_file_is_what_the_filter_reproduces`
  — needs `build_julia_seed_pool.VIABLE_DEFAULT`, which is **`scratch/`-class, the one class
  whose contract guarantees deletion**. It has therefore skipped since the wipe and will skip
  until someone re-runs `q4_decisive`. **Flagged, not deleted** (2026-08-06): unlike the two
  cases resolved that day, its input has a live producer, so the referent is regenerable
  rather than gone. Either re-run the producer and freeze the comparison against a durable
  copy of the input, or delete the test — a permanent skip is the one thing it should not
  stay. The filter itself is covered unconditionally by
  `test_filter_drops_anchor_and_projects_in_order`.

Two non-skip absence tolerances remain outside the test suite: the `.exists()` guards in
`tools/v9/build_plan.py::assert_recipe_parity`, whose preconditions are in
`storage_classes.md`.

Resolved on 2026-08-06 under this rule, recorded so the pattern is visible: three
permanently-skipping tests were deleted — `test_recipe_parity_v5`/`_v6` (inputs wiped
2026-07-25, never git-tracked, and no v5/v6 build can run again) and
`test_new_form_passes_on_real_v8_locations` (reads `data/v8/cache_manifest.jsonl`, deleted
2026-08-03, on the same reasoning `test_v8_cache_alignment.py` already used to delete three
of its own).

## 3. Prove it red

- **Prove a new guard red on purpose** before trusting it green. A carried red is a guard
  that is OFF.
- **One fresh process per file for import smokes** — a shared interpreter has already
  imported what you are testing.
- **Pin a tripwire to an INVARIANT, not a growing total.** A count that legitimately grows
  is re-baselined until it means nothing.
- **A smoke that asserts "it ran" passes on zero.** Assert the COUNT against an
  independently recorded census — recorded *while the old paths still work*.
- **Bracket a fix on both sides:** old behaviour was wrong AND new behaviour is right AND
  the fix does not over-correct. Never blind-rebaseline an oracle a policy change moved.
  `[code: auto_maxiter.md, the cap-change checklist item 2]`
- **During a shakedown, ONE engine-spawning job at a time** (2026-08-03): all three
  self-inflicted errors of that night's shakedown came from a second engine job running
  concurrently — a seeder smoke beside `prospect_orchestrator`, whose sweep documents
  itself safe only when no seeder is running. `[verdict: Matt's box]`
- **Kill by PID, never by image name** (2026-08-03): an image-name `taskkill` killed a
  concurrent orchestrator's engines, and a `kill -9` on the wrong half of the `.venv`
  launcher/real-python pair left the run alive and voided the resume test.

## 4. A guard that goes red during ordinary workflow gets trained out

Distinguish "normal working state" from "actually broken" — and note that a permanently-soft
warning nobody can action is the *same* failure, not the safe version of it (§1.4).

**Prefer a SUITE ASSERTION to a client-side commit hook.** A hook needs per-clone install
and its disablement is undetectable; a red suite is loud. `tools/hooks/pre-commit` was
never installed in this checkout and was replaced by `tests/test_large_tracked_blobs.py`.
`[verdict: Matt]`

## 5. Express invariants relationally

**Never bump the number, and never delete entries to go green.** A derived set self-adjusts
across rebuilds — which is why the versioned build trees are guarded off the `.gitignore`
negations rather than by a static list that is empty exactly when it is needed.

**Derive + prove non-empty is the template.** A derived set can pass by evaluating EMPTY, so
every one is paired with a non-vacuity assertion; `@parametrize` over an empty list
generates zero tests and stays green. The pairing is what makes the derived form safe to
prefer. Owned in full by `artifacts_resolver.md` §6.

**An allowlist needs a no-dead-entry assertion.** An entry matching nothing is a line nobody
classified, and a rotted allowlist is how the coverage assertion beside it goes vacuous.
`[code: artifacts_resolver.md §4; tests/test_large_tracked_blobs.py]`

## 6. Vacuity from the other end: the fixture is too easy

- **A test can pass and be vacuous because its fixture cannot fail.** Probe intermediate
  values; assert on the **count of distinct values**, not only on direction.
- **A test that injects the dependency never covers the loader.** Any loader with a
  silent-empty fallback needs two tests: the ABSENCE path and PRESENCE-FROM-DISK.
- **An equality test against a class CEILING silently discards a new top class.** When a
  scale extends, grep every equality test against the old ceiling — the 1–3 → 1–4 label
  extension is the local instance.
- **A guard pinned to PROSE goes red when the prose is corrected.** Anchor a documentation
  guard on the routing *decision*, not on a sentence.
  `[code: tests/test_docs_tree.py::test_claude_md_routes_analysis_text_at_a_live_destination]`
- **A backstop longer than the job's budget is not a backstop** — clamp per-unit timeouts to
  the remaining budget. (`CLAUDE.md`, "Four rules".)

## 7. Differential over frozen literals — with one caveat

Where a reference implementation survives, run the new path against it instead of freezing
a literal: `bench_lateral_seeding.py` replays recorded parent views through both the sweep
and the hybrid off an identically-seeded RNG, and `identify_nucleus`'s sweep is kept as the
reference implementation for exactly this reason. `[code: tools/atlas/bench_lateral_seeding.py]`

**But a differential proves new == reference, not new == historical.** And a disagreement is
not automatically a defect: if the contract is "*an* object satisfying P" rather than "*that*
object", disagreement is **identity drift** — report it, and use it to choose the default
rather than to fail the build. `[code: minibrot_maneuvers.md §2.6]`

## 8. Functional parity beats byte-exactness

- **A pure-speedup refactor carries a parity gate on its OUTPUT DECISION.** The guard's
  field path uses a different kernel from the diagnostic it reproduces — values differ by a
  constant bailout-normalization offset — so the tripwire regresses **verdicts**, and the
  docstring says byte-identity is not expected *by design*.
  `[code: tools/atlas/guard.py; tools/atlas/test_guard_tripwire.py]`
- **An enforcement-only change carries a zero-change proof.**
- **Byte-identity only where a cache key or a cross-language table depends on it.**
- **A performance change is safest as an opt-in:** leave the old entry point byte-identical
  as a catch-all and gate only the callers you can name. `identify_nucleus` grew an optional
  `periods=` list; the sweep is untouched.

## 9. Source-inspection tests are a last resort, and they need a paired control

9 test files assert on `inspect.getsource` substrings (`test_steered_frontier.py` alone has
11). They test TEXT, not behaviour, and they fail for reasons unrelated to the property: two
written on 2026-08-05 went red on first run because a *report* line legitimately mentioned the
token being located, and because the docstring contained it. That is the cost, and it is paid
per edit to a file nobody was changing.

- **Prefer a behavioural assertion wherever one exists.**
  `test_a_truncated_dive_plan_keeps_BOTH_ARMS_at_every_cut_point` is worth more than any grep
  for `interleave_dive_arms`, because it states what a truncated plan must contain rather than
  which function was called.
- **A behavioural test needs a CONTROL on the unfixed input**, or it cannot tell "the fix
  works" from "this population would have passed anyway".
  `test_the_UNFIXED_block_order_loses_an_arm_at_the_length_the_run_actually_reached` is the
  pattern: the same truncation, applied to the pre-fix order, asserted to fail. Same shape as
  §3's prove-it-red, aimed at the fixture instead of the guard.
- **Where the property is genuinely about the source — "X runs before Y", "the rule is not
  restated here", "the flag is passed, not dropped" — the source test is correct**, and
  several are deliberately kept. Anchor it on the smallest stable token (a call shape like
  `sort(key=queue_sort_key)`, not a bare name that also appears in prose), so that naming the
  rule in a docstring or a report string does not go red.
  `[code: tools/atlas/test_sitting_cutter.py::test_the_union_queue_and_the_single_run_queue_sort_on_the_SAME_key]`
- **A structural property is better computed than grepped.** Where the claim is about
  reachability or shape, walk the AST instead of matching text, and prove the analysis catches
  the defect by running it on an injected miniature.
  `[code: tools/atlas/test_steered_frontier.py::test_the_analysis_CATCHES_a_flag_that_misses_the_second_entry_point]`

## 10. Two entry points that share a constructor need a declared contract

`--wall-budget` parsed, converted, stored as `self.wall_budget_s`, and was then consumed only
by `run()`; `run_dive()` has its own loop, so the flag silently did nothing and the fact had
nowhere to live but a hand-written note in a launch record. 41 constructor attributes are in
that same position, of which only 4 were explicitly neutralized — **the code could not
distinguish "deliberately N/A on the second path" from "silently dropped on it", and nothing
detected the difference.** That gap is the whole bug; the flag was just where it surfaced.

The shape of the fix, reusable wherever two entry points share state: declare the
inapplicable set as DATA with a one-line reason each (`steered_frontier.DIVE_IGNORES`),
recompute it from the source and assert set equality in both directions (a stale exemption
hides the next real one), and for anything that is a **bound** rather than a tuning knob,
REFUSE rather than ignore — `check_wall_budget_supported` raises, because a bound that
silently does not apply is worse than no bound (`CLAUDE.md`, "a backstop longer than the
job's budget is not a backstop").
`[code: tools/atlas/steered_frontier.py::DIVE_IGNORES; test_steered_frontier.py]`

## 11. Git as evidence

`git ls-files` is an **index query** — it answers "is this tracked *now*", never "did this
ever exist". An absence verdict needs
`git log --all --full-history --diff-filter=D -- <path>`. A file with zero commits in
history was never tracked and has no recovery path. And in a freshly re-imported repo
`git log` cannot date liveness at all: this one's floor is the 2026-07-24 squashed import,
so `git show <older>^:<file>` silently finds nothing rather than failing.
`[code: storage_classes.md, "Git history is a durability tier too"]`
