# The Python suite's cost model, its two lanes, and what is irreducible

Measured 2026-07-31, re-measured 2026-08-03, **re-measured 2026-08-06**, on the
12-logical-core box (`uv run pytest -n 4 --dist loadfile -q --durations=60`). The point of
this doc is to stop the next person from re-deriving the profile, and from "optimizing" the
things that are already at their floor.

## 1. The shape: a handful of files hold the whole cost

The suite is **2,223 tests / 118.3s** parallel (2026-08-06, after the audit pass in §7), and
the overwhelming majority run in under 0.15s each. Essentially all the wall clock lives in a
handful of tests that drive the real engine, a real sklearn fit, or a real Newton solve.
Anything you do to the ~2,200 cheap tests is noise.

**The suite grew 56% between 2026-08-03 and 2026-08-06** (1,427 → 2,226 before the audit),
and it had nearly doubled in the three days before that (767 → 1,427). Wall clock did NOT
track: 178s serial-then, 118s parallel-now, because the growth is all in cheap tests while
the expensive handful is unchanged. That is the standing lesson here: this profile goes
stale in days and the test COUNT does not predict it, so **re-measure before optimizing**
rather than trusting the table.

Costs below are the 2026-08-06 parallel run (a `setup` line is a shared fixture, charged
once). The 2026-08-03 column is kept where a fix moved it.

| test | cost | what it actually is |
|---|---|---|
| `test_guard_tripwire::test_live_f64_path_reproduces_every_verdict` | 45.8s | 81 real `render-one --dump-field` subprocesses. `slow`-marked, out of the default lane — not in the 118.3s. |
| `test_descent_smoke::test_two_step_descent_emit_and_roundtrip` | 36.0s | three 1280×720 ss4 production renders |
| `test_emit_staging` (4 tests) | 66.0s → **30.6s** | 2026-08-03: staged once, not four times (§3). 16.7s of it is the shared `setup`. |
| `test_newton_divergence_abort` (2 grid tests) | 68.4s → **22.0s** | 2026-08-03: grid cut to 2×2 on a measured curve (§3) |
| `test_q4_screen_parity` (both tests + setup) | 19.1s | dense sliding-window `featurize` sweeps |
| `test_triage` full-pool enumeration | 9.4s (setup) | Newton atom enumeration at 4.6 tasks/s |
| `test_descent_smoke::test_box_guard_refuses_below_f64_wall` | 8.7s | real nav renders down to the f64 wall |
| `test_t_good_sweep_decode::test_the_v8_anchor_…` | 8.6s | two full LOO-OOF `build_table` derivations, O(n²·\|GRID\|) |
| `test_frozen_record_writes::test_derive_t_good_v10_writes_…` | 8.6s | a real v10 t_good derivation under both `--adopt` and not |
| `test_steered_frontier::test_keeper_cuts_rederive…` | 6.6s | the committed-constant drift gate |
| `test_view_fit_bar_read::test_the_frozen_read_is_the_read_this_code_takes` | 6.2s | re-runs the frozen bar read |
| `test_import_hygiene::test_no_ambiguous_basename_is_imported_bare` | 5.2s | ASTs every tracked `.py` for bare ambiguous imports |

## 2. The two lanes

`pyproject.toml` sets `addopts = ["-m", "not slow"]`. Both the marker description and
every marked module's docstring had *described* `slow` as opt-in since it was introduced,
but nothing implemented it, so a bare `pytest` ran the opt-in lane anyway — 49s of a 172s
suite, 46s of it the guard tripwire.

- **default lane** — `uv run pytest -n 4 --dist loadfile`, 2,221 passed + 2 skipped,
  **118.3s** (2026-08-06). Serial: see §5.
- **opt-in lane** — `uv run pytest -m slow`, 10 tests. **It grew sharply on 2026-08-03**
  and is no longer a ~1-minute lane: `test_full_roster_ring_seed_grid_is_parity_clean`
  adds 26,624 + 4,992 differential solves. Projected from the measured 0.239 s per
  non-converger at `max_steps=600` and the ~30% non-convergence rate in the table below,
  that is **≈35 min**, dominated by the 64×8 grid. That is a mean-cost projection, not a
  timed run — the lane has not been run end-to-end since (CLAUDE.md's rule about
  projecting a long run's wall clock applies, though here the work is homogeneous across
  the grid rather than contiguous-expensive).

A third label, `version_pinned`, is **not** a lane and is excluded from nothing — see
CLAUDE.md. `uv run pytest -m version_pinned --collect-only -q` lists 104 tests / 12 files
(2026-08-06).

Two things about the split that are easy to get wrong:

**`-m` is a filter, not a path rule.** `pytest tools/atlas/test_guard_tripwire.py` now
collects 1 of 2 and silently drops the render pass; a deselect-to-zero run exits green and
reads as a pass. The opt-in lane needs an explicit `-m slow` even when you also name the
file.

**Mark the test, not the module.** `test_guard_tripwire.py` originally carried a
module-level `pytestmark = pytest.mark.slow`, which would have taken
`test_fixture_is_the_canonical_81_20_set` out of the default lane along with the renders.
That test does no rendering and exists precisely to catch a corrupt fixture *before* the
expensive pass — i.e. it is the thing that stops the slow test from being vacuous. Keeping
the cheap integrity half in the default lane costs nothing and is the general rule: when a
file pairs a cheap invariant with an expensive regression, only the expensive half is
`slow`.

**The opt-in lane is manual.** There is no CI in this repo and every git hook is a git-lfs
shim, so `-m slow` runs when a human types it or never. CLAUDE.md names the one case where
it is mandatory: any change to the guard field path (`guard.render_field`,
`--dump-field-source f64`, the f64 smooth kernel), because the tripwire is the only thing
regressing live-path verdict parity and the fast `test_guard.py` gate only exercises
`guard.py`'s arithmetic on a frozen field.

## 3. What was actually made faster, and the pattern

### 2026-08-03 — the two files that had become half the suite

Measured together on one tree, before and after: **121.1s → 43.4s**, same 23 tests
passing (`uv run pytest tools/descent/test_emit_staging.py
tools/sourcing/test_newton_divergence_abort.py -q`).

- **`test_emit_staging.py`, 66.0s → 26.3s.** All four tests called `_stage()`, which
  drives `open → nav_box → quality` — two production-fidelity renders, ~13.5s — and every
  one of the four is about `/emit`, not `/quality`. One `test_session_holds_render_…`
  spent 13.5s of real rendering to assert the shape of a dict. Now a module-scoped
  `_staged_master` renders once and a function-scoped `staged` fixture gives each test a
  private *copy* of that tree plus a restored `dh.SESSIONS`, because two of the four
  mutate it destructively (an `rmtree`, a corrupted render block). This is the
  shared-fixture pattern below, applied to a fixture that costs a render rather than a
  fit. The 11.2s that remains is the rebuild-from-record render, which *is* the test.
- **`test_newton_divergence_abort.py`, 68.4s → 17.3s.** The two default-lane grid tests.
  Cost here is **linear in the number of non-convergers**, not in the grid: the live arm
  aborts on the first orbit pass, so what is paid for is the *reference* arm burning its
  whole `max_steps` (600) on each one, ~0.22s apiece. The measured curve at
  `_grid_parity(n_ang, 2, DEGREES)`:

  | n_ang | wall | solves | conv | non-conv | lost | mismatch | aborted |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 6 | 45.7s | 624 | 433 | 191 | 0 | 0 | 191 |
  | 4 | 29.4s | 416 | 291 | 125 | 0 | 0 | 125 |
  | 3 | 20.8s | 312 | 216 | 96 | 0 | 0 | 96 |
  | 2 | 13.0s | 208 | 145 | 63 | 0 | 0 | 63 |

  `aborted == non-conv` and `lost == mismatch == 0` at **every** density: the abort fires
  on 100% of non-convergers, so the population is homogeneous with respect to everything
  the test asserts and a denser default grid buys more solves of the same kind, not a new
  kind of evidence. The committed 64×8 density is what the `slow` lane is for. **`n_rad`
  must stay ≥ 2** — at `n_rad=1` every seed converges, the abort never fires, and both
  tests go vacuous while still passing (0.28s, green, proving nothing).

**The general rule both share:** when a test's cost is dominated by one expensive input,
find out what the cost is actually *proportional to* before cutting. Here it was
non-convergers, not grid points; there it was renders, not tests.

### 2026-07-31

Three fixes, none of which weakened an assertion (190.9s → 171.7s before the lane split):

- **`test_box_guard_refuses_below_f64_wall`, 16.9s → 8.8s.** It walks box descents until
  the f64-wall guard refuses. It started from `_first_d2()` — the first d2 atom in *file
  order*, which happens to be the shallowest (`fw` 7.6e-2) — and spent three real nav
  renders, each at a higher `auto_maxiter` than the last, just getting down to the wall.
  Starting from the *deepest* d2 atom reaches the same refusal in one. **Pick the fixture
  by the property under test, not by file order.**
- **`test_page_size_bounds_what_loads`, 8.1s → ~0s.** It built its own 80-atom
  enumeration to prove paging bounds what loads. Enumeration is pure — the pool is a
  deterministic function of the cursor schedule — so it now shares the module-scoped
  120-atom pool that `test_only_cut_…` already needed. **Two tests needing a big
  deterministic input should build it once.**
- **`test_q4_screen_parity`, 26.5s → 19.2s.** Both tests were independently building the
  same synthetic field, fitting the same model, and running the same `LF.dense_grid` sweep
  at every scale. Module-scoped fixtures for all three.

The shared-fixture moves interact with a hazard already flagged in `test_triage.py`:
`_redirect()` repoints `triage_store` module globals with **no teardown**, so last writer
wins and every test must set its own. A module-scoped fixture is safe here only because
its consumers re-`_redirect` on entry.

## 4. What is at its floor — do not re-litigate

- **The guard tripwire's 46s is not a tuning problem.** `WORKERS = 4` with default engine
  threads was measured against every plausible alternative: 4×default **46.5s**, 4×3
  threads 50.0s, 4×2 64.6s, 2×6 51.6s, 1×12 53.3s. The current setting is already the
  optimum; oversubscription is not what's costing you. It is 81 real renders.
- **`test_two_step_descent_emit_and_roundtrip`'s 30s is three production-fidelity
  renders**: canonical + vivid inside `/quality` (~9s each) and the round-trip re-render
  (~9s). The vivid one is required by the `/emit` path and the round-trip *is* the test
  (byte-for-byte reproduction from the stored record is what catches coordinate
  truncation). Reducible only by lowering `store.CROP_W`/`CROP_SS` — production constants
  — or by rendering the canonical/vivid pair concurrently in `/quality`, which is an app
  change.
- **The triage enumeration's 13.6s** is `target=120` at 4.6 tasks/s, and 120 is what the
  statistical assertions need (a (degree, band) cell overflowing `TARGET_PER_CELL`,
  periods surviving past the roster bands, all four degrees present).
- **The q4 synthetic field stays 176×112.** Shrinking it is the only remaining lever and
  it is a bad trade: 128×80 takes the file to ~8s but drops the bit-identical comparison
  from 281 survivors / 61 peaks to 63 / 17, and 96×64 goes fully vacuous (zero survivors,
  zero peaks — the test's own vacuity guards fire). 20s buys 4.5× the coverage.

## 5. Parallelism — ADOPTED 2026-08-03

`pytest-xdist` is a dev dependency. **The full-suite command is:**

```bash
uv run pytest -n 4 --dist loadfile      # 2,223 tests, 118s (vs 217s serial, 1.84x)
```

Measured on a quiet box (the first 2026-08-03 attempt was invalidated by an unrelated render
study saturating the machine — see §6). **The two dates are different suites** — 1,439 vs
2,223 tests — so read down a column, not across:

| run | wall (2026-08-03) | wall (2026-08-06) | vs serial |
|---|---:|---:|---:|
| serial | 185.3s | 217.3s | 1.00× |
| `-n 4 --dist loadfile` | 89.0s / 90.6s (two runs) | **118.3s** | 2.06× / **1.84×** |
| `-n 6 --dist loadfile` | 88.5s | not re-measured | 2.09× |

The speedup **fell** from 2.06× to 1.84× as the suite grew, which is the expected direction:
784 new cheap tests spread evenly over 4 workers while the serial tail — one 36s render test
that no amount of `-n` can split — did not move. Under `--dist loadfile` the floor is the
single slowest FILE, so as cheap work is added the ratio walks toward
`total / slowest_file`, not toward 4×. Do not read a falling ratio as xdist regressing.

**`-n 4`, never `-n auto`.** `-n` is an upper bound on *concurrent engine subprocesses*,
because any worker can be the one driving `render-one`. `-n auto` is 12 on this box and
would put 12 `fractal-generator.exe` (7 threads each) against 12 cores — a direct
violation of CLAUDE.md's 4-concurrent-process cap, and the desktop-unusable condition that
rule exists for. `-n 6` buys **nothing** measurable (88.5s vs 90.2s), so the cap-compliant
setting is also the performance-equivalent one. There is no reason to go above 4.

**The opt-in lane stays serial.** `-m slow` must NOT be combined with `-n` — the guard
tripwire drives its own 4 engine subprocesses internally, so `-n 4 -m slow` is 16
concurrent processes.

`--dist loadfile` is load-bearing, not incidental: this suite mutates module globals
freely (`_redirect` repointing `triage_store`, `ta.PAGE_SIZE`, `sys.path` inserts), and
keeping a whole file on one worker is what preserves the ordering those depend on.

**Why it is still NOT in `addopts`.** xdist costs ~3–4s fixed per invocation, and under
`loadfile` a single-file run gets *zero* parallelism — pure overhead. Measured:
`test_triage.py` 21.4s → 25.6s, `test_guard.py` 2.3s → 5.4s. It also breaks `-s`, pdb and
live output, on exactly the targeted runs where you want them. The asymmetry against the
`slow` marker is the deciding argument: **forgetting `-m slow` silently loses coverage,
forgetting `-n 4` only costs time** — only the first needs to be automatic. So bare
`pytest <file>` stays fast and debuggable.

The rejected alternative, for the record: a root `conftest.py` injecting `-n` only when no
path arguments are given. It gets both properties, but there is no root conftest today and
it is emergent behavior of exactly the kind `pyproject.toml`'s "the suite's extent, stated
rather than emergent" comment argues against.

## 6. What adopting xdist exposed: an order-dependent test

Enabling `-n 4` turned up **one** failure —
`test_tau_h_rederive::test_the_row_cache_is_bulk_and_resolves_out_of_tree` — that serial
never produced. It was a **latent defect in the suite, not something xdist caused**, and
it reproduces deterministically in serial with the right file order:

```bash
# fails at test_tau_h_rederive.py:72 without the fix, passes with it
uv run pytest tools/descent/test_descent_smoke.py tools/atlas/test_tau_h_rederive.py -q
```

The mechanism, worth knowing because the shape recurs:

* `artifacts_root()` reads `FRACTAL_ARTIFACTS_ROOT` **at call time**, while modules bake
  their `bulk()` paths **at import time** (`tau_h_rederive.WORK = paths.bulk(...)`).
* Three test files set that env var process-wide with a bare `os.environ[...] = ...` and
  **no teardown** (`test_descent_smoke`, `test_emit_staging`, `test_triage`), so a
  redirected test leaked a tmp artifacts root into every later test in the same process.
  `test_sources` already did it correctly with `monkeypatch.setenv`.
* Serial collection order happened to be safe. `--dist loadfile` assigns files to workers
  dynamically, so the ordering is not fixed — which is exactly why parallelism is a decent
  order-dependence detector.

Fixed with an autouse snapshot/restore fixture in each of the three files. **Scope is
load-bearing and got this wrong on the first attempt:** `test_triage.full_pool_dir` and
`test_emit_staging._staged_master` are *module*-scoped and redirect during their own
setup, so a *function*-scoped restore runs afterwards and snapshots the
already-redirected value — the leak escaped as `popen-gw0/triage_full_pool0` and the run
stayed red. The restoring fixture must be at least as broad as the broadest fixture that
redirects.

Standing rule: **set process-global state through `monkeypatch`, never a bare
`os.environ[...] =`.** Where a helper is a plain function that cannot take `monkeypatch`,
pair it with an autouse restore fixture at the widest scope that redirects.

## 7. The 2026-08-06 audit pass — what a whole-suite sweep actually found

A deliberate audit of all 2,226 default-lane tests for vacuity, dead referents, redundancy
and disproportionate cost. Recorded here because **the headline result is that there was
almost nothing to cut**, and the next person should not repeat the sweep expecting a haul.

Four mechanical sweeps over the 139 tracked test files (1,702 test functions):

| sweep | method | found |
|---|---|---|
| duplicate tests | AST normalized (names/strings/args erased), hashed, grouped | **5** groups, all legitimate (parametrize siblings, one deliberate v5/v6 pair) |
| assertion-free / tautological | AST: no `Assert`, or `Assert` on a constant | 90, of which **89** are `pytest.raises` fail-closed guards or the `assert False` fail-marker idiom |
| self-swallowing guard | `try: … assert False, msg … except AssertionError` where the handler's own check matches `msg` | **1** — a real defect, §7.1 |
| record-reader-only | AST: test calls nothing but file-read/json/builtins | 12, all deliberate frozen-record oracles |

### 7.1 The one vacuous test

`test_prospect.py::test_embedding_dim_assert_rejects_mismatch` caught its own marker:
`assert False, "expected dim assert to fire"` raises `AssertionError`, its own
`except AssertionError as e` caught it, and the `"dim" in str(e)` check it then ran matched
the *marker's* text. **Verified vacuous** by deleting the production assert in
`library_store.write_embedding_shard` — the file stayed green (52 passed). Rewritten as
`pytest.raises(AssertionError, match=…)`, which goes red on the same injection.

**The general rule:** a try/`assert False`/except block is safe only when the handler cannot
catch `AssertionError`. Four sibling tests in `test_pool_rebalance.py` use the same idiom
correctly — they catch `SystemExit`, so the marker propagates. Prefer `pytest.raises`.

### 7.2 Three permanent skips deleted, one flagged

Five tests skipped in the default lane; every one was §2-of-`verification_practice.md`'s
absence-tolerant shape. Deleted, each because its referent is gone AND the defect it caught
cannot recur:

- `tools/v5/test_recipe_parity_v5.py`, `tools/v6/test_recipe_parity_v6.py` — inputs
  (`data/v4|v5/cache_manifest.jsonl`) wiped 2026-07-25 and never git-tracked, so there is no
  recovery path; no v5/v6 build will run again (`ACTIVE_CKPT` is v10, v8 the only rollback
  rung). Side effect: both directories are now dead by both methods — `tools/README.md`'s
  rows and its retirement-candidate table were updated to say so.
- `classifier/test_palette_renders_v8.py::test_new_form_passes_on_real_v8_locations` — reads
  `data/v8/cache_manifest.jsonl`, deleted 2026-08-03. `tools/v8/test_v8_cache_alignment.py`
  had already deleted three of its own tests on this exact referent and reasoning; this one
  was missed. Its contract survives in the five synthetic brackets beside it.

Kept and flagged: `test_julia_seed_pool.py::test_committed_file_is_what_the_filter_reproduces`
— its input is `scratch/`-class (guaranteed deletion) so it can never run as-is, but the
producer is live, which makes it regenerable rather than gone. See `verification_practice.md`
§2, which also had its live-absence-tolerant list re-derived: that list had rotted in both
directions and named a fixture that no longer exists.

### 7.3 One order-dependent file (the §6 shape, again)

Running each test file **alone** (`pytest <file> --collect-only`, all 139) found exactly one
that could not: `tools/wallpaper/test_prospect.py` imported `tools.corpus.corpus_common`,
whose line-364 `import artifacts` is bare and resolves only once `tools/corpus` is on
`sys.path` — which some *other* test file happened to do first. Its green belonged to that
file. Fixed with an explicit `sys.path` insert in `test_prospect.py`. The other 138 are
self-sufficient; a whole-file solo sweep is cheap (~4 min) and worth repeating after a batch
of new files.

### 7.4 Two shared-scan fixtures (the §3 pattern, applied to source scans)

Repo-wide `git ls-files` + `ast.parse` scans are a growing family in this suite, and two
files were paying for the same walk two or three times:

- `tests/test_scratch_dependency_allowlist.py` — three tests each called `scan(REPO_ROOT)`.
  An `lru_cache`d `repo_scan()` **8.6s → 2.5s** (file: 11 tests in 2.97s). The synthetic-tree
  calls in the prove-it-red section deliberately bypass it: they plant a file and re-scan the
  same root, which a cache would silently defeat.
- `tools/emission/test_intake_fail_closed.py` — three scans of `tools/**/*.py`, two of them
  parsing. A cached `_tools_sources()` took **9.9s → 6.1s**. It caches the WIDEST population
  (test files included) and leaves narrowing to each caller, because the three scans disagree
  on scope — a helper that pre-filtered test files would have silently shrunk
  `test_no_call_site_can_swallow_the_abort`.

**If a third file joins this family, promote the cached scan to a `tools/` helper** rather
than writing a third private copy — the same argument `apportion.py` settled for the
apportionment rules.

### 7.5 What was deliberately NOT touched

- Everything in §4. Each was re-checked against the fresh profile and still holds.
- The 33 `inspect.getsource` assertions across 10 files. `verification_practice.md` §9 calls
  them a last resort but explicitly keeps the ones anchored on call shapes — which, on
  inspection, is all of them.
- `test_steered_frontier.py`'s `assert len(sf.DIVE_IGNORES) == 41`. It is a re-baselined
  count and the non-vacuity it claims is already carried by the `assert _sf_crawl_only()`
  beside it, so it is loosenable — but its docstring states its purpose, which makes it a
  deliberate pin rather than an accident. Flagged, not changed.

**Net: 2,226 → 2,223 tests, 122.8s → 118.3s parallel.** Three deletions, one repair, one
order-dependence fix, two shared fixtures. The speedup is incidental; the point was that
2,223 of 2,226 tests earned their place, and now the reason each survivor did is written
down.
