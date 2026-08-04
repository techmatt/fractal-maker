# The Python suite's cost model, its two lanes, and what is irreducible

Measured 2026-07-31, **re-measured 2026-08-03**, on the 12-logical-core box
(`uv run pytest --durations=50 -q`). The point of this doc is to stop the next person
from re-deriving the profile, and from "optimizing" the things that are already at their
floor.

## 1. The shape: a handful of files hold the whole cost

The suite is **1,427 tests / ~178s** (2026-08-03), and the overwhelming majority run in
under 0.15s each. Essentially all the wall clock lives in a handful of tests that drive
the real engine, a real sklearn fit, or a real Newton solve. Anything you do to the ~1,400
cheap tests is noise.

**The suite nearly doubled between 2026-07-31 and 2026-08-03** (767 → 1,427 tests), and
both files that came to dominate the profile were added in that window and are absent
from the older version of this table. That is the standing lesson here: this profile goes
stale in days, so **re-measure before optimizing** rather than trusting the table.

| test | cost | what it actually is |
|---|---|---|
| `test_guard_tripwire::test_live_f64_path_reproduces_every_verdict` | 45.8s | 81 real `render-one --dump-field` subprocesses. `slow`-marked, out of the default lane. |
| `test_descent_smoke::test_two_step_descent_emit_and_roundtrip` | 30.4s | three 1280×720 ss4 production renders |
| `test_emit_staging` (4 tests) | 66.0s → **26.3s** | 2026-08-03: staged once, not four times (§3) |
| `test_newton_divergence_abort` (2 grid tests) | 68.4s → **17.3s** | 2026-08-03: grid cut to 2×2 on a measured curve (§3) |
| `test_q4_screen_parity` (both tests) | 24.5s | dense sliding-window `featurize` sweeps |
| `test_triage` full-pool enumeration | 13.6s | Newton atom enumeration at 4.6 tasks/s |
| `test_descent_smoke::test_box_guard_refuses_below_f64_wall` | 9.0s | real nav renders down to the f64 wall |
| `test_t_good_sweep_decode::test_the_v8_anchor_…` | 9.1s | two full LOO-OOF `build_table` derivations, O(n²·\|GRID\|) |
| `test_steered_frontier::test_keeper_cuts_rederive…` | 5.3s | the committed-constant drift gate |

## 2. The two lanes

`pyproject.toml` sets `addopts = ["-m", "not slow"]`. Both the marker description and
every marked module's docstring had *described* `slow` as opt-in since it was introduced,
but nothing implemented it, so a bare `pytest` ran the opt-in lane anyway — 49s of a 172s
suite, 46s of it the guard tripwire.

- **default lane** — `uv run pytest`, 1,422 tests + 5 skipped, ~178s (2026-08-03).
- **opt-in lane** — `uv run pytest -m slow`, 9 tests. **It grew sharply on 2026-08-03**
  and is no longer a ~1-minute lane: `test_full_roster_ring_seed_grid_is_parity_clean`
  adds 26,624 + 4,992 differential solves. Projected from the measured 0.239 s per
  non-converger at `max_steps=600` and the ~30% non-convergence rate in the table below,
  that is **≈35 min**, dominated by the 64×8 grid. That is a mean-cost projection, not a
  timed run — the lane has not been run end-to-end since (CLAUDE.md's rule about
  projecting a long run's wall clock applies, though here the work is homogeneous across
  the grid rather than contiguous-expensive).

A third label, `version_pinned`, is **not** a lane and is excluded from nothing — see
CLAUDE.md. `uv run pytest -m version_pinned --collect-only -q` lists 93 tests / 10 files.

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
uv run pytest -n 4 --dist loadfile      # 1,439 tests, ~90s (vs ~185s serial, 2.06x)
```

Measured 2026-08-03 on a quiet box (the first attempt was invalidated by an unrelated
render study saturating the machine — see §6):

| run | wall | vs serial |
|---|---:|---:|
| serial | 185.3s | 1.00× |
| `-n 4 --dist loadfile` | 89.0s / 90.6s (two runs) | **2.06×** |
| `-n 6 --dist loadfile` | 88.5s | 2.09× |

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
