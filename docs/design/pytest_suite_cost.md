# The Python suite's cost model, its two lanes, and what is irreducible

Measured 2026-07-31 on the 12-logical-core box. The point of this doc is to stop the next
person from re-deriving the profile, and from "optimizing" the four things that are
already at their floor.

## 1. The shape: five files hold the whole cost

The suite is **767 tests / ~116s**, and roughly **700 of those tests run in under 0.15s
each**. Essentially all the wall clock lives in a handful of tests that drive the real
engine, a real sklearn fit, or a real Newton enumeration. Before the 2026-07-31 pass the
whole suite was 190.9s; the top eight tests were 147s of it (77%).

Anything you do to the ~700 cheap tests is noise. Only these matter:

| test | cost | what it actually is |
|---|---|---|
| `test_guard_tripwire::test_live_f64_path_reproduces_every_verdict` | 45.8s | 81 real `render-one --dump-field` subprocesses. **Now `slow`-marked, out of the default lane.** |
| `test_descent_smoke::test_two_step_descent_emit_and_roundtrip` | 30.3s | three 1280×720 ss4 production renders |
| `test_triage` full-pool enumeration | 13.6s | Newton atom enumeration at 4.6 tasks/s |
| `test_q4_screen_parity` (both tests) | 19.2s | dense sliding-window `featurize` sweeps |
| `test_steered_frontier::test_keeper_cuts_rederive…` | 5.5s | the committed-constant drift gate |

## 2. The two lanes

`pyproject.toml` sets `addopts = ["-m", "not slow"]`. Both the marker description and
every marked module's docstring had *described* `slow` as opt-in since it was introduced,
but nothing implemented it, so a bare `pytest` ran the opt-in lane anyway — 49s of a 172s
suite, 46s of it the guard tripwire.

- **default lane** — `uv run pytest`, 761 tests, ~116s.
- **opt-in lane** — `uv run pytest -m slow`, 6 tests, ~54s.

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

## 5. Open: parallelism (prototyped, not adopted)

`pytest-xdist` is **not** a dependency. Prototyped via
`uv run --with pytest-xdist pytest -n 4 --dist loadfile`: **62.0s / 62.8s across two runs**
vs 116.5s serial — **1.88×**, same pass counts, stable.

`--dist loadfile` is load-bearing, not incidental: this suite mutates module globals
freely (`_redirect` repointing `triage_store`, `ta.PAGE_SIZE`, `sys.path` inserts), and
keeping a whole file on one worker is what preserves the ordering those depend on.

**Why it was not wired into `addopts`.** xdist costs ~3–4s fixed per invocation, and under
`loadfile` a single-file run gets *zero* parallelism — pure overhead. Measured:
`test_triage.py` 21.4s → 25.6s, `test_guard.py` 2.3s → 5.4s. It also breaks `-s`, pdb and
live output, on exactly the targeted runs where you want them. The asymmetry against the
`slow` marker is the deciding argument: **forgetting `-m slow` silently loses coverage,
forgetting `-n 4` only costs time** — only the first needs to be automatic.

If adopted, the shape is: `uv add --dev pytest-xdist`, document
`uv run pytest -n 4 --dist loadfile` as the full-suite command, leave bare
`pytest <file>` fast and debuggable. **The opt-in lane must stay serial regardless** — the
tripwire drives its own 4 engine subprocesses, so `-n 4 -m slow` would blow past the
4-concurrent-process cap.

The rejected alternative, for the record: a root `conftest.py` injecting `-n` only when no
path arguments are given. It gets both properties, but there is no root conftest today and
it is emergent behavior of exactly the kind `pyproject.toml`'s "the suite's extent, stated
rather than emergent" comment argues against.
