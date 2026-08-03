# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust engine for generating orbit-trap Mandelbrot/Julia fractal images as wallpapers. The long-term goal is mass-generating strong fractals under quality gates with a human picking favorites — palettes are the first-class concern. The **render core** (precision backends, separable coloring, palette system) is settled; the active workstream is the corpus → label → classifier pipeline (see "Corpus & classifier pipeline" below). The early navigation/diagnostic probes were retired in the P2 subcommand cull once that flow superseded them; `cargo run -- --help` is the live subcommand list.

## Commands

```bash
cargo build --release            # always release — debug is ~50-200x slower
cargo test                       # tests/*.rs + unit tests; --test <file> or a name substring narrows

uv run pytest                    # Python suite, default lane (~2 min); slow tests excluded
uv run pytest -m slow            # the opt-in lane (~50s) — see below for when it's mandatory

# Single render (no subcommand → one PNG):
cargo run --release -- --center-re -0.743643887 --center-im 0.131825904 \
  --frame-width 1e-6 --maxiter 2000 --width 1920 --output out.png

# Contact sheet: one location, many palettes, iterated once:
cargo run --release -- sheet --builtins "default cubehelix viridis" --output sheet.png
```

Background long renders / descents; release builds do deep production-res renders in seconds.

**Where the prompts live.** Session prompts are in **`../fractal-maker-controller/prompts/`**
(a sibling repo, e.g. `C:\Code\fractal-maker-controller\prompts\v10_flip.md`). This repo has
its OWN `prompts/` directory holding a different set, so a prompt named in a session and not
found here is in the sibling — check there before globbing the tree.

**Long background runs: redirect to a log file; never pipe through `tail`/`head`.** The pipe
buffers, so a job that prints progress with `flush=True` shows nothing for its whole runtime
*and* you lose the header lines that reported what it skipped. Use `... > scratch/<job>.log
2>&1` (background) and read the file; the τ_h re-derivation was piped through `tail -60`, went
26 minutes with no visible progress, and cost a separate re-derivation to recover a count the
header had already printed. Sibling of the ETA rule below: a run reports on itself only if you
let it.

**The `slow` pytest lane is manual — nothing else runs it.** There is no CI here and every
git hook is a git-lfs shim, so `-m slow` runs when you type it or never. It is **mandatory
before committing** a change to the guard field path (`tools/atlas/guard.py`'s
`render_field`, `--dump-field-source f64`, or the f64 smooth kernel): the 81-tile tripwire
is the only thing that regresses live-path verdict parity, and the fast `test_guard.py`
gate only exercises `guard.py`'s arithmetic on a frozen field. `-m slow` is a **filter, not
a path rule** — naming a slow test's file on the command line still deselects it, and a
deselect-to-zero run reads as a pass, so the `-m slow` must be there explicitly.

**`version_pinned` is a LABEL, not a lane.** It is excluded from nothing — those tests run in
the default suite like any other. It exists so the set of things a classifier-version flip
touches can be *listed* instead of discovered by flipping and reading the wreckage:
`uv run pytest -m version_pinned --collect-only -q` (~90 tests, 9 files). Pair it with
`production_pins.COUPLED_ARTIFACTS` (the revert-together set as data, walked by
`tools/scoring/test_coupled_artifacts.py`) and `classifier_retrain_protocol.md` §5 before any
`ACTIVE_CKPT` move.

**Windows exe-lock note.** A running binary file-locks
`target/release/fractal-generator.exe`, so a concurrent `cargo build --release` fails with
`Access is denied (os error 5)`. Build into an isolated dir instead —
`CARGO_TARGET_DIR=target-test cargo build --release` (`target-*/` is gitignored) — or
`cargo build --release --lib`, which compile-checks without touching the exe.

**Build lanes.** The `release` profile is tuned for *render throughput*
(`opt-level=3`, `lto="thin"`, `codegen-units=1`), so a full incremental build is **~100s**
— almost all codegen, one LLVM unit, no parallelism. Pick the lightest lane:
- **"Does it compile?"** — `cargo check --release` (**~3s**) or `cargo build --release
  --lib`. The default during a refactor. Keep `--release` so it shares the release
  dependency artifacts (a bare `cargo check` rebuilds deps in `dev`, ~15s first time).
- **Runnable binary fast** (smoke-test, eyeball a render) — `cargo build --profile quick`
  (~16s incremental) → **`target/quick/fractal-generator.exe`**. ~10-30% slower at runtime
  (the per-pixel kernel loses cross-module inlining): correctness checks, **not** perf timing.
- **Production renders, batch reproducibility, perf timing** — `cargo build --release`.
  Don't relax the profile; the `quick` lane exists so you don't have to.

`cargo test` compiles under whichever profile you pass (`--release` reuses release
artifacts); the suite runs in seconds, so the same lane logic applies.

## Architecture

Two deliberate seams structure everything (`src/lib.rs` is the module root; `src/main.rs` is a thin CLI wrapper):

**1. Precision behind `backend::FractalBackend`.** The per-pixel `sample(c, dc)` loop is swappable without touching the render driver, coloring, or CLI. Backends are built **per frame** (maxiter/bailout in the constructor; perturbation also computes its reference orbit there). Tiers:
- `F64Backend` — plain f64 escape time. Fast, accurate only while pixel spacing stays clear of f64 epsilon (~1e-13 of |c|, i.e. ~1e12 magnification at production resolution).
- `PerturbationBackend` — single high-precision reference orbit at the frame center (stored as f64 projections, since orbit *values* stay O(1)) plus per-pixel f64 deltas with **Zhuoran rebasing**. Clean far past where f64 quantizes; v1 cap ~1e300 magnification (where f64 deltas underflow). Glitch detection is a per-pixel underflow flag, not Pauldelbrot detection.
- `JuliaBackend` — base-scale Julia (`z₀ = pixel`, fixed `c`); always shallow, so never needs perturbation. Intentionally skips DE (`de = 0`).
- Auto-selection is by pixel spacing (`PERTURB_SPACING = 1e-13`); `--backend f64|perturb|auto` overrides.

**2. Separable coloring (`coloring::shade`): `PixelSample` → linear-RGB.** Iteration emits a small `PixelSample` record (smooth iter, DE, trap_min, trap_phase, escaped/glitched); coloring is a **pure** map over it, so **re-coloring never re-iterates**. This is what makes the contact sheet and palette experimentation cheap. Channel validity matters: `smooth_iter`/`de` are exterior-only; `trap_min`/`trap_phase` are valid for *every* pixel (interior included), which is how orbit traps fill the interior instead of dead black.

### The two-stage render (`render.rs`)
1. `iterate_samples` — runs the backend over the **supersampled** grid (rayon over rows), caches `Vec<PixelSample>` at SS resolution. The only stage that touches a backend.
2. `shade_and_downsample` — pure: shades each subpixel, averages **in linear light**, then sRGB-encodes. AA is mandatory and must stay correct under re-color, so colors (not pre-shade channel values) are averaged.

Memory: the SS buffer is ~48 B × out_w × out_h × ss² (~470 MB at 1920×1280 ss2). Keep large supersampled frames to modest resolution.

### Critical coordinate rule
**`dc` (a pixel's offset from frame center) is computed straight from pixel geometry, never as `c - center`.** At deep zoom `c_f64 - center_f64` is catastrophic cancellation. `dc` is O(frame_width) and accurate in f64 to ~1e-305; perturbation uses only `dc`. The absolute `c = center + dc` is formed solely for the f64 backend (shallow only). Centers are parsed as **arbitrary-precision decimal strings** (`--center-re`/`--center-im`) because an f64 center is meaningless at depth.

### Supporting modules
- `hp.rs` — high-precision scalar support (astro-float, pure Rust, no C dep). Decimal parse, `prec_bits` (precision sizing), fast `to_f64` projection (hand-rolled, bypasses the slow decimal formatter), and `to_decimal_string` for high-precision decimal serialization. Orbit arithmetic uses `RoundingMode::None` (f64 projection absorbs sub-ulp error); only the input parse rounds correctly.
- `palette.rs` — cyclic gradients interpolated in **OKLab**, baked once into a `LUT_SIZE`-entry linear-RGB LUT so `lookup_linear` (the only coloring contract) is O(1). Built-ins: `default` (Ultra Fractal), `cubehelix`, `viridis`.
- `palette_io.rs` — `.ugr` (UltraFractal, multi-block) and `.map` (Fractint) loaders → sRGB8 stops. The resolver dispatches a `--palette` spec: built-in name or path by extension.
- `sheet.rs` — contact sheet: iterate one location once, re-shade across N palettes (multi-block `.ugr` → one tile per block). Burns a swatch strip + index per tile.
- `font.rs` — hand-rolled bitmap font for on-image labels (no font crate).
- `energy.rs` — pixel-space image measures, two halves with different liveness. **Live:** `occupancy` / `tile_energy` / `region_energies` (OKLab forward-diff edge energy pooled on a grid) — the content/occupancy gate both `guided_descend` and `enrich` call; reuse, never reimplement. **Parked:** the `calibrate` subcommand and the corpus-distance metric it freezes (4-scale quantile histograms, `distance` = Σ per-scale 1-D EMD via `emd1d`, `kmeans` archetypes) have **no live caller** — nothing under `tools/` invokes `calibrate` or `generate`, and `generate` is the sole reader of the tracked artifact (`energy::ARTIFACT_PATH`, last written 2026-06-21).

## Validation pattern

The f64 backend is the **ground truth** for perturbation: shallow renders from both must match (`tests/perturbation.rs`, run at `maxiter = 300` where f64 orbits stay accurate — deeper, f64 is *not* valid ground truth at chaotic locations). Separability is enforced by tests that wrap a backend in a `sample()` counter and assert re-coloring never re-iterates (`tests/separability.rs`, `tests/sheet.rs`).

## Corpus & classifier pipeline

The active workstream (the render core above is "done enough"). Goal: a labeled
corpus that trains an aesthetic classifier across every generator version's output.

**The flow.** Two live entry paths, one shared tail (what drives what: `tools/README.md`).
**Discovery** — `tools/atlas/production_seeder.py` (or `steered_frontier.py`, the
classifier-steered variant) drives `guided-descend` → `data/guided_descend/<run>/pool.jsonl`
(one candidate per row: cx/cy/fw + idx + provenance) → scored in memory (bridge below),
selected, rendered. **Roster** — `tools/sourcing/build_{minibrot,interior_band,gcf_arm}_batch.py`
draw from a durable roster and render via `render-one`. Both land a batch under
`data/label_corpus/batches/<batch_id>/` (schema: `data/label_corpus/CORPUS_SCHEMA.md`) → label
in `tools/viz/corpus_label.html` (exports `scores.json`, merged by
`tools/corpus/merge_scores.py`; revisions go to the amendment stream via
`merge_amendments.py`) → `classifier/` trains by **unioning every batch blind to provenance**
(`corpus_reader.py`). The `present` subcommand still builds zoom/composition batches and the
schema still names it a crop-rebuild path, but **no live tool invokes it** — `render-one` is
what runs today.

**The label corpus contract** (full spec: `CORPUS_SCHEMA.md`). Each `images.jsonl`
row has three independent blocks. `render` is **version-invariant** — the identical
field set across all batches (`RENDER_KEYS` in `tools/corpus/corpus_common.py`),
cx/cy/fw as decimal strings, and is the *only* thing the classifier sees (it's a
pure function → `crops/<image_id>.jpg`, rebuildable via `present`/`render-one`).
`provenance` is **version-tagged**, free to differ/be null across batches
(`PROVENANCE_KEYS`); it feeds the bias loop only and **never enters training**.
`label.score ∈ {null,1,2,3,4}` (bad/okay/good/exceptional; rubric in `CORPUS_SCHEMA.md`
§ label — class 4 ranks the top of "good" and does **not** move the `>=3` emit floor). `null → value` is the ONE allowed mutation to an **original** label; a
merge that would change a non-null score warns and refuses. A *revision* is no exception —
it goes to the amendment stream (`labels/<revision>.json`, `merge_amendments.py`) and is
read back via `label_store.resolve_score`, leaving the original byte-identical.

**Shared owners under `tools/scoring/`** — the version dirs (`v7/`…`v10/`) each used to
re-declare these, so they are named here to be imported, not rediscovered:
`production_pins.py` (the pin + `COUPLED_ARTIFACTS`), `derive_t_good.py` (THE version-agnostic
t_good estimator — a per-version deriver supplies only its slice, population rule and
objective), `partitions.py` (the `fractal_type` ⟷ ledger-partition map; a source scan in
`test_partitions.py` fails on a second literal copy) and `eval_slice.py` (a version's frozen
slice: `data/<v>/eval_scores_<v>.jsonl` and its `<v>_p_ge{2,3,4}` columns).

**The classifier** (`classifier/`, pkg). Weights/metrics in `data/classifier/v5…v10/`,
**git-LFS tracked in-tree — NOT gitignored** (`.gitattributes` + exact-path `.gitignore`
negation; guarded by `tests/test_tracked_artifacts.py` and the size-guard registry). A weight
file is a tracked artifact, not scratch. **Never hardcode a version** — the live pin is
`tools/scoring/production_pins.ACTIVE_CKPT` (v10 since 2026-08-02; v8 is the one-flip rollback
anchor, v9 was built and staged but never adopted and is NOT a rollback rung), read
by ~41 modules — most of them still through the `active_ckpt` re-export, which is an alias, not
a copy (`test_production_pins.py`). Every version is a CORN **ordinal** head on
`mobilenetv4_conv_medium.e250_r384_in12k` emitting K−1 rank-consistent logits; **K is
per-version — read `data/classifier/<v>/config.json`** (K=3 through v7, labels 1–3; **K=4 from
v8 onward**, labels 1–4 — v8 is the first K=4 head, not v9). Deploy transform = `classifier.data.Transform(train=False)`: the
deterministic **1280×720 → 384×224 bicubic stretch + normalize** mirror of `present.rs`'s JPG
path (no jitter/flips). `model.score_from_logits` returns `Σ σ(logit_k)` ∈ [0,K−1] — the
monotone rank score used for AP. **P(not-bad) = σ(logit₀)** (= P(rank≥1) = P(label≥2)).
Black-gate parity with the Rust render path: accept iff `black_fraction < 0.30`
(`BLACK_THRESH`, strict `<`).

**The in-memory scoring bridge** (`enrich` subcommand, `src/enrich.rs`).
`enrich --mode score` iterates each pool location once at the label geometry, recolors under
N seeded palettes, and streams each recolored RGB frame to **stdout** as a raw record
(16-byte LE header `idx,ki,w,h` then `w*h*3` RGB bytes); the Python side scores every frame
through the exact deploy transform — so 10k+ scoring passes never write crops to disk. Only
the selected `(location, argmax-palette)` rows are rendered to JPG (`enrich --mode render`,
ss4 Lanczos3 wallpaper quality). Two readers of that stream: the live one is the library
`tools/mining/score_lib.py::run_enrich_score` (driven by `tools/mining/harvest.py`);
`tools/corpus/enrich_score.py` is the standalone CLI sibling. Both default to a checkpoint
that **no longer exists on disk** (v2 / v3 — `data/classifier/` holds v5…v9). Those pins are
kept on purpose — they record what those batches were scored with — and are **not**
repointed; each now goes through a `require_ckpt` that raises naming the missing file, so
pass `--model` explicitly or resolve through `ACTIVE_CKPT`
(`tools/mining/test_require_ckpt.py`).

## Conventions

> **Commit prompt work to `main`; branch only on an explicit request.** A production config
> change sitting on an unmerged branch looks applied and isn't, and the failure is silent — the
> τ_h floor raise was staged on `closeout-batch-tau-h` and had no effect until it landed.

> **Generated-output convention.** Every generated artifact — renders, strips, sheets, run JSON, logs, fixtures — goes under the single `scratch/` tree, never the repo root; the root holds only source, config, docs and committed `assets/`. `scratch/` is gitignored (except `.gitkeep`), so the whole working corpus wipes with one `rm -r scratch/*` without touching anything tracked. **New subcommands MUST default their output under `scratch/<subcommand>/`.** Enforcement is partial: `tests/test_docs_tree.py` guards the prose half (no loose `.md` at the root beyond `CLAUDE.md`/`README.md` — four analysis docs had accumulated there); **nothing checks the root for generated binaries or data**, so that half is convention only.

The fixed base defaults are `scratch/renders/` (bare render) and `scratch/strips/` (sheet); every other subcommand writes under its own `scratch/<subcommand>/`. Use `crate::ensure_parent_dir(path)?` before any top-level `save`/`fs::write` so a no-flag default writes its dir on a fresh checkout.

> **Where analysis text goes.** A document belongs in `docs/design/` **only if something in the
> code owns it and it stays true as the code changes.** A measurement of a transient state owns
> nothing and is false the moment the work it drove succeeds: analysis goes to `scratch/`, what
> survives is extracted into the design doc that already owns the subject, and the analysis is
> deleted. Two corollaries — a **maintained index** passes (the directory owns it; a missing line
> is a visible omission), and a measurement that does survive **carries its date and the command
> that produced it**. **`docs/findings/` is RETIRED and must not be recreated**
> (`tests/test_docs_tree.py` enforces that it is gone, that no source names it as a write target,
> and that every file under `docs/` is tracked).

> **Standing references — consult, don't rediscover.** Three maintained practice docs in
> `docs/design/`, each read *before* the work rather than after:
> [`verification_practice.md`](docs/design/verification_practice.md) before writing any test,
> guard or gate; [`measurement_practice.md`](docs/design/measurement_practice.md) before
> designing any measurement, eval, readout or run projection;
> [`retired.md`](docs/design/retired.md) before proposing an approach that may have been
> tried already. `retired.md` is append-only — a reversal is a new dated `UN-RETIRED` entry,
> never an edit.

> **Neither scratch tree is a dependency tier — and it fails in both directions.** `scratchpad/`
> is the disposable temp dir (gitignored, currently empty); `scratch/` the disposable output
> tree. **If a file is imported from outside `scratchpad/`, or is the only thing producing a
> durable artifact, it isn't scratch — promote it to `tools/` or delete it**
> (`scratchpad/visual_dup/embed.py` was load-bearing, uncommitted, vanished, cost a formula sweep
> to recover). **Tests belong in the suite** — default, or `slow`-marked if opt-in/destructive; a
> test git never sees is a memory of a test. And **nothing load-bearing lives in `scratch/`**:
> evidence must leave it the moment it justifies a durable decision, and a proposal computed
> there must never leave it as a fact about the system. The two greps below must stay empty:
>
> ```bash
> # (a) nothing outside scratchpad imports a scratchpad module:
> grep -rn "import" --include="*.py" tools/ classifier/ src/ | grep -i scratchpad
> # (b) no scratchpad file writes a durable data/ artifact:
> grep -rnE "savez|write_text|open\([^)]*['\"]w" --include="*.py" scratchpad/ | grep -iE "data/|STORE"
> ```

> **Persistent-store convention (`data/`).** Anything that must survive `rm -r scratch/*` lives
> under `data/`. **Declare the class at the write site** through `tools/paths.py`
> (`scratch()` / `bulk()` / `durable()`) — never hand-build the path. The contract (which class,
> and why) is [`docs/design/storage_classes.md`](docs/design/storage_classes.md); the mechanism
> (`ARTIFACTS_ROOT` resolver, size-guard registry, LFS + `.gitignore` negation) is
> [`artifacts_resolver.md`](docs/design/artifacts_resolver.md). Rust side: expose a read-back
> path as a `pub const` shared by writer and reader (e.g. `energy::ARTIFACT_PATH`).

> **Projecting a long run's wall clock.** **A sample unbiased for mean per-unit cost is NOT
> unbiased for a run whose expensive work is contiguous** — cost-per-unit and cost-of-run are
> different estimands, and the second also needs the *order* the work is done in. Sample **in run
> order** (prefix-weighted, or a contiguous block per region of the file), or say plainly it is a
> mean-cost estimate and not an ETA: the v9 cache render missed by **1.65×** on a
> correctly-unbiased `fw`-decile sample, because `plan.jsonl` is emitted in family order with the
> deep bulk late. Then **reproject from the observed decaying rate; never restate the original
> ETA** — a run's own throughput beats any pre-run sample, and refit from *recent* throughput,
> not the run-to-date average (dominated by cheap early work). Restating the first estimate while
> the rate visibly falls is how a run reports "20 minutes left" for two hours.

> **Four rules, each earned by a failure.**
> - **A verification tool that cannot reach its authority reports UNKNOWN, not absent.** `git lfs
>   prune --verify-remote` called objects missing from the remote when it could not authenticate
>   to ask — and "missing" is the one condition under which you must not prune.
> - **Never characterize a failure population from a truncated error log.** A persisted
>   `errs[:10]` described a 19.5% failure class that was really ~1.2%: the fastest-returning
>   failure arrives first.
> - **A backstop longer than the job's budget is not a backstop.** A 900 s per-unit timeout in a
>   15-minute run lets one hung unit double the wall clock while the budget logic believes it is
>   inside its cap.
> - **Derive state in code; freeze it in records.** A generator must read the state it reports
>   from the state itself — a hardcoded `True` is how a metadata file outlives what it records.
>   A committed record may keep what was true when written (`storage_classes.md`).

> **Adding a subcommand.** The per-subcommand `Args` struct (+ its `impl { resolved_* }` helpers)
> lives **in the subcommand's own module**, next to its `run_*`. Four edit sites: (1) the
> `#[derive(Args)]` struct there (e.g. `EnrichArgs` in `src/enrich.rs`), `use`-importing whatever
> shared groups it flattens from `cli`; (2) a `Command` variant in `cli.rs` referencing it by path
> (`Enrich(crate::enrich::EnrichArgs)`); (3) `src/main.rs` `use` + dispatch arm; (4) `src/lib.rs`
> `pub mod`. **`cli.rs` keeps only** the cross-cutting types (`BackendChoice`, `LocationArgs`,
> `ShadeArgs`, `PaletteSelectArgs`, `Cli`, `Command`, `parse_complex`). Default outputs under
> `scratch/<subcommand>/` or `data/<subcommand>/`, never the repo root, and keep flag
> names/defaults stable (batch reproducibility depends on them).

- Deps are kept minimal and pure-Rust (no C deps): clap, num-complex, rayon, image (png/jpeg/webp), astro-float. The JSON logs (guided-descend pool, generate manifest, calibration artifact) are hand-rolled rather than pulling in serde.
- **Max 4 concurrent PROCESSES. In-process threads are not capped at 4.** The limit is how
  many heavyweight OS processes contend, not how much parallelism one uses: 4+ simultaneous
  `fractal-generator.exe` make the desktop unusable (each carries its own rayon pool,
  plan/corpus scan and resident LUTs); one process with many threads does not.
  - **Capped at 4:** `ProcessPoolExecutor` `max_workers`, subprocess fan-out, any `WORKERS`
    constant that spawns children, and `ThreadPoolExecutor` when each thread drives a
    subprocess (process concurrency in a thread pool's clothes).
  - **NOT capped at 4:** threads inside one process — `RAYON_NUM_THREADS`, the Rust binary's
    pool. Size against the box's **12 logical cores** and run long jobs at
    `BELOW_NORMAL_PRIORITY_CLASS`. `tools/v8/render_cache.py` runs `WORKERS = 6` on this basis
    (12.1 tiles/s at 6 vs 7.0 at 3, desktop unaffected) — do not "correct" it to 4.
  - **One `fractal-generator.exe` defaults to 7 threads at BELOW_NORMAL** — call
    `corpus_common.DEFAULT_ENGINE_THREADS` + `default_engine_env()` /
    `default_creationflags()` (pinned by `tools/corpus/test_engine_launch_defaults.py`) rather
    than restating the pair; throughput and interactivity move together. **Multiple parallel
    engine processes has no standing number** — size for the actual N, pass `threads=`
    explicitly, and don't inherit the per-process 7.
- Matt is expert (graphics + ML PhD) — be terse and precise; skip basics.
- Module docs (`//!`) carry the real design rationale; read them before changing a module.

## Python / uv

The Rust engine is the core; Python is the ML/analysis side (corpus tooling, the aesthetic
classifier, palette experiments). **Use `uv` for all Python, not bare `python`/`pip`/conda** —
the global `python` on PATH is base conda with no torch. Env is root `pyproject.toml` +
`uv.lock` (both committed); `.venv/` is gitignored and regenerable with `uv sync`. Run with
`uv run python …`, add deps with `uv add <pkg>`.

- **GPU stack:** torch is the **cu124** build (`torch==2.6.0`, `torchvision==0.21.0`) from the
  `pytorch-cu124` index pinned in `pyproject.toml`, not PyPI's CPU default; CUDA runs on the
  local RTX 2060 SUPER (8 GB). `timm`, `scikit-learn`, `Pillow`, `numpy` round it out.
- Versions are pinned in lockstep with the `video-to-photo` project so uv's global cache
  hardlinks the wheels (a full `uv sync` is seconds, not a multi-GB torch download).
- There is **no package root** — imports work by `sys.path` mutation, so a module's position
  is load-bearing (`tools/README.md` §"Two standing facts").
