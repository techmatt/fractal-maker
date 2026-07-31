# `tools/corpus/artifacts.py` — the storage MECHANISM

Named for the code that owns it: the `ARTIFACTS_ROOT` resolver in
`tools/corpus/artifacts.py`, and the four enforcement surfaces built around it. This doc
is the wiring record — how a path is constructed, where the in-tree/out-of-tree decision
is actually made, and what each guard keys on.

**Boundary — this is the second half of a pair.**
[`storage_classes.md`](storage_classes.md) owns the **contract**: which durability class
an artifact belongs to, and why. This doc owns the **mechanism**: resolver, registry,
tripwire, LFS, gitignore. The split is by reader, not by topic — the contract is read at
a write site by someone asking *what is this file*, this doc is read by someone changing
the wiring. Neither is a summary of the other; when they touch, the contract states the
rule and this doc states what enforces it. `[verdict: Matt]`

**Tags.** `[code: path]` — true because the tree says so. `[measured: population]` — a
number, with the population it is true *of*. `[verdict: who]` — a judgement call.
`[unverified]` — not checkable from this repo.

---

## 1. The resolver and its seam

**One function is the seam: `artifacts.resolve(rel)`.** It takes a **repo-relative**
POSIX-ish string — the portable, version-invariant thing that gets stored in manifests
and plans — and returns the real on-disk `Path`. Relocated families resolve under
`ARTIFACTS_ROOT`; everything else resolves under `REPO_ROOT`, unchanged.
`[code: tools/corpus/artifacts.py::resolve]`

**`ARTIFACTS_ROOT` defaults to a repo *sibling*, `../fractal-maker-artifacts`, so a fresh
checkout on any machine resolves with no configuration.** Override with the
`FRACTAL_ARTIFACTS_ROOT` environment variable. The relocated tree mirrors the
repo-relative layout exactly, so `data/v9/aug_cache/...` lands at
`<ARTIFACTS_ROOT>/data/v9/aug_cache/...` — the path string is the same string on both
sides of the move, which is why a plan row written before the relocation still
dereferences. `[code: tools/corpus/artifacts.py::{artifacts_root, ARTIFACTS_ENV}]`

**The in-tree/out-of-tree decision is made in exactly one predicate,
`is_relocated(rel)`, and nowhere else.** It is a pure function of the path string: no
filesystem probe, no directory listing, no "does it exist yet". That is what lets a
family be declared relocated **before** its first byte is written.
`[code: tools/corpus/artifacts.py::is_relocated]`

**Who may build a path directly.** Nobody, for a relocated family — reader *and* writer
must route through `resolve`, or the data is written where nothing will look for it.
Non-relocated in-tree paths may be built directly (`REPO_ROOT / rel`), and most of the
tree does. The practical rule: if a path could ever hold bulk, it goes through
`paths.bulk()`; if it must survive, through `paths.durable()`; if it is disposable,
through `paths.scratch()`. Convenience wrappers exist for the families with many call
sites — `corpus_common.crops_dir` / `vivid_dir` for label-corpus crops — and those
wrappers call `resolve` so a batch builder never sees the seam.
`[code: tools/paths.py; tools/corpus/corpus_common.py]`

**`paths.bulk()` delegates rather than reimplementing.** `tools/paths.py` deliberately
imports `tools/corpus/artifacts.py` and calls its `resolve`, because a second
`ARTIFACTS_ROOT` resolver is a second answer to "where does this live".
`[code: tools/paths.py::bulk]`

## 2. `durable()` — and what makes its check pass for the right reason

**`durable(rel)` asserts at write time that git would keep the path**, raising
`DurabilityError` naming the path and its class if a `.gitignore` rule would exclude it.
The mistake surfaces at the attempted write, not months later when the file is needed and
gone. `[code: tools/paths.py::durable]`

**The check is `git check-ignore`, which honours negations** — so a path re-included by a
`!/data/v9/plan.jsonl` line correctly reports not-ignored. Ignore status is fixed within a
run, so results are `lru_cache`d. `[code: tools/paths.py::_is_gitignored]`

**It fails OPEN when git is unavailable.** No git, no proof the path is unsafe — so it
returns not-ignored rather than blocking every durable write on a machine without git.
`[code: tools/paths.py::_is_gitignored]` `[verdict: Matt]`

**The re-verification in the canary uses `--no-index`, and that flag is load-bearing.**
The index-consulting form of `git check-ignore` reports any **force-added** path as
not-ignored regardless of the rules — which is how a real false accept got through on the
v7 checkpoint. `--no-index` evaluates the rules alone, so "this path is declared durable"
is a statement about the ignore rules and not about whether someone once ran `git add -f`.
`[code: tests/test_tracked_artifacts.py::_rules_ignore]`

## 3. Relocation classes — matched by pattern, not by a list

Two mechanisms coexist, and the second is the one that scales.

**Literals — `RELOCATED_PREFIXES`.** For families at a fixed, versioned path: today
`data/v8/aug_cache` and `data/v9/aug_cache`. Matched as a whole path component (exact, or
followed by `/`), so a sibling like `data/v9/aug_cache_notes` does **not** match.
`[code: tools/corpus/artifacts.py::RELOCATED_PREFIXES, is_relocated]`

**Classes — a predicate over the path.** Four families are matched by *shape* rather than
by registration: discovery scratch (`data/discovery/**/scratch`), label-corpus crops
(`data/label_corpus/batches/*/{crops,vivid}`), descent-harness images
(`data/descent_harness/{crops,vivid,thumbs}`) and minibrot source bulk
(`data/minibrot_sources/{tiles,sheets}`). All four are component-exact, so
`.../crops_staging` does not match.
`[code: tools/corpus/artifacts.py::{_is_discovery_scratch, _is_label_corpus_crop,
_is_descent_harness_crop, _is_minibrot_source_bulk}]`

**Why the class form is preferred: it fails in the safe direction.** A new campaign, a new
label batch, a new descent emit relocates with *no registry edit at all*. Forgetting to
register costs conservatism (something goes out-of-tree that could have stayed); with a
literal, forgetting costs 45 GB in the source tree. `[code: as above]` `[verdict: Matt]`

**Registering one requires three edits that must move together**: the predicate (or
literal) in `artifacts.py`, the mirroring `.gitignore` stanza (the backstop — an
accidental in-tree rebuild stays un-committable), and a branch in the reappearance
tripwire's scan. A class added to the resolver but not to the tripwire relocates
correctly and is never checked.
`[code: tools/corpus/artifacts.py; .gitignore; tools/audit/test_relocated_artifacts.py]`

**A literal stays registered after its data is deleted, if the data can be rebuilt.**
`data/v8/aug_cache` kept its line when the 12.13 GB / 171,384-tile tree was deleted on
2026-07-31, because a rebuild from the committed `data/v8/plan.jsonl` must land
out-of-tree exactly as the first render did. Dropping the literal — as was done for
v4..v7 — is right only when the family can never come back. This is the same
live-forward-declaration judgement §5 makes for the size-guard registry, applied to the
resolver. `[code: tools/corpus/artifacts.py::RELOCATED_PREFIXES comment;
tools/audit/test_relocated_artifacts.py::test_live_aug_caches_are_registered]`

**The tripwire keys on FILES, not on directories.** `_scan_in_tree_offenders` walks each
relocated family's old in-tree path and reports it only if it holds real files; an empty
leftover directory is tolerated, because a move legitimately leaves the parent behind.
Real files mean a writer bypassed the resolver.
`[code: tools/audit/test_relocated_artifacts.py::_scan_in_tree_offenders]`

**The tripwire is parameterized on the root, and fires in both directions.** The
clean-tree test proves it stays quiet on the real repo; synthetic-repopulation tests plant
a file under each family's old path in a `tmp_path` mirror and assert it goes red naming
the offender — including for a campaign / batch id **never registered anywhere**, which is
the payoff of class matching. `[code: tools/audit/test_relocated_artifacts.py, 5 synthetic
fire tests]`

## 4. The size guard — registry semantics

`tools/audit/size_guard.py` is the **backstop**, not the primary mechanism. The primary
mechanism is the write site: a family that declares `bulk()` never lands in-tree in the
first place. The guard catches what bypassed it.
`[code: docs/design/storage_classes.md "Where this is enforced"]`

**Three independent scan rules, one shared allowlist.** Per-**file** ≥ 1 MiB; per-**dir**
small-file aggregate ≥ 100 MB; per-**dir** bulk at ≥ 2,000 files **or** ≥ 500 MB in the
subtree. Rules (a)/(b) structurally cannot see the harm rule 5 names: a cache of a million
4 KB crops passes (a) cleanly and says nothing about file *count*, which is what drives
traversal cost. `[code: tools/audit/size_guard.py::scan]`

**Everything is reported at MINIMAL granularity, on the same predicate that flagged it.**
A 10k-file / 300 MB directory is reported at the leaf-most 10k-file directory, not pushed
up to a parent that merely happens to be over on bytes — otherwise every registry entry
would have to be written at a uselessly coarse prefix.
`[code: tools/audit/size_guard.py::scan, `_bulk`]`

**Dispositions.** `KEEP` — legitimately stays in-tree; being tracked is *not* an automatic
pass, the written reason is the exception. `RELOCATE -> {artifacts, precious-store,
trash}` — pending a move; the tier is a disposition **label** only, no directories are
created and no paths are wired by this module.
`[code: tools/audit/size_guard.py::{KEEP, RELOCATE, ARTIFACTS, PRECIOUS, TRASH}]`

**Coverage is longest-prefix.** A violator is covered by the most specific entry whose
prefix it starts under, so a narrow line inside a registered directory wins over the
directory's own line. `[code: tools/audit/size_guard.py::covering_entry]`

**Prefix-vs-file granularity is a TRADE, made deliberately, not the default.**
Registering at a directory prefix stops the guard flaking as the directory churns — a new
batch, a new crop, a growing pool — at the cost of not seeing a *new* large file that
appears inside that prefix. `data/orbital/` is registered at the directory prefix on
exactly this reasoning, stated in its own entry: the derived `screen_scores.jsonl` is
under threshold today and would otherwise flap the guard when the pool grows. Write the
entry at the narrowest prefix that is stable, and say in the reason which way you traded.
`[code: tools/audit/size_guard.py::REGISTRY "data/orbital/"; covering_entry]`
`[verdict: Matt]`

**Two hard assertions, no standing warning.**
1. Any flagged violator with no covering entry fails. New bloat is caught the day it
   lands, named by path.
2. Any entry that covers no over-threshold content **and** is not marked `forward=True`
   fails — a line nobody classified.
`[code: tests/test_repo_size_guard.py]`

**`Entry.forward` is what made (2) a hard check.** It marks a **live forward
declaration**: nothing is there now, but a committed writer can still put it there, and
the line is the disposition that write lands under. Until 2026-07-31 the emptiness report
was a warning, on the argument that emptiness cannot distinguish not-yet-built from dead
(`data/v8/` was legitimately empty before its build) — true of emptiness, and no longer
true once the distinction is recorded. It had been firing on 15 lines every run, which is
how a soft red gets trained out. `[code: tools/audit/size_guard.py::Entry,
check_registry; tests/test_repo_size_guard.py::test_every_empty_entry_is_classified]`

**The test for a stale line is *can anything still WRITE here*, and the costs are
lopsided.** A kept dead line costs one config line. A pruned live one costs a red build at
the moment the writer next runs, with the disposition re-decided under time pressure —
and, for the resolver's registry rather than this one, silent in-tree bulk.
`[verdict: Matt]`

**Current state.** 23 entries: 6 `KEEP`, 6 marked `forward`, 43 flagged violators all
covered, 0 stale. `[measured: this working tree, 2026-07-31, `size_guard.py --check`]`

## 5. `.gitignore`, LFS, and why they must agree

**`data/` is ignored by default (`/data/*`) and durability is claimed by EXACT-PATH
negation.** A bare `!/data/v9/` would also re-include an accidental in-tree `aug_cache/`
rebuild, which is the one thing that must never land in the tree — so each durable
artifact is negated by its own line, and the aug-cache directory is re-excluded
underneath as a backstop. `[code: .gitignore, the v8 and v9 stanzas]`

**The re-include chain has to walk down.** Git cannot re-include a file whose parent
directory is still excluded, so each level is opened (`!/data/classifier/`) and then
re-narrowed (`/data/classifier/*`) before the leaf negation. Getting this wrong is silent:
the negation is present, the file is still ignored. `[code: .gitignore, the
data/classifier/v9 stanza]`

**Negation over force-add, because a negation is a RULE and a force-add is an event.**
`data/classifier/v2..v8` were force-added, which works but leaves nothing saying they
belong — a fresh clone's `git add -A` would not re-add them and nothing would notice. v9
is declared by exact-path negation instead, checkable with `git check-ignore --no-index`
and picked up automatically by the derived canary. `[code: .gitignore, the v9 weights
stanza; tests/test_tracked_artifacts.py::BUILD_PREFIX_RE]`

**LFS and the ignore rules are cross-checked, because an LFS rule on an ignored path is
configuration for a file that never arrives.** Every versioned-build LFS pattern must name
a path the ignore rules re-include, and every declared-durable path that carries an LFS
rule must have `git check-attr` actually resolve `filter` to `lfs` — a later attributes
line silently overriding it would commit a 90 MB file inline.
`[code: tests/test_tracked_artifacts.py::{test_v8_durability_wiring_coherent,
test_v8_durable_declared_paths_tracked}]` `[measured: 28 LFS-tracked files, this tree,
2026-07-31]`

## 6. The derived canary set, and its non-vacuity pairing

**Two canaries with different shapes.** `TRACKED_CANARIES` is a **static list** of
unregenerable ∧ tracked paths — human labels, the reference fixtures, the trained weights
with no rebuild path, the legacy-policy convergence ladder. It guards *de-tracking*, and
each entry is a deliberate opt-in; a canary guarding everything is one nobody maintains.
`[code: tests/test_tracked_artifacts.py::TRACKED_CANARIES]`

**The versioned build trees are guarded RELATIONALLY instead**, because they are
periodically **rebuilt**, and a static list that must be emptied and refilled around each
rebuild is a guard that is off exactly when it is needed — which happened on 2026-07-29,
leaving a comment where the assertion used to be. The invariant is expressed against the
wiring: *a `data/v<N>/` path re-included by an exact-path `.gitignore` negation is, by
that negation, declared durable — therefore it must actually be tracked.* The set
self-adjusts; `BUILD_PREFIX_RE` matches the corpus side (`data/v9/`) and the weights side
(`data/classifier/v9/`) alike, so a new build version is covered the moment its negations
land. `[code: tests/test_tracked_artifacts.py, the relational section]`

**A derived set can pass by evaluating EMPTY, so it is paired with a non-vacuity
assertion.** `test_v8_durability_wiring_coherent` asserts `V8_DURABLE` is non-empty
*before* the parametrized check runs: if the negations were removed, or the parser stopped
matching them, `@parametrize` over an empty list silently generates zero tests and the
suite stays green. The pairing is the whole reason the derived form is safe to prefer over
the list. The same guard-the-guard pattern is repeated for every derived or list-driven
check here: `test_registry_nonempty`, `test_guard_list_nonempty`,
`test_the_tracked_docs_set_is_nonempty`, and — on the other side —
`test_forward_entries_are_the_minority_and_each_says_why`, which stops the escape hatch
from becoming the default. `[code: tests/test_tracked_artifacts.py;
tests/test_repo_size_guard.py; tests/test_docs_tree.py]`

## 7. The small-tree guard and the docs policy

**The standing constraint is that the working tree stays ≈ what git tracks** — source,
irreplaceable metadata, and `scratch/`. The scan walks the **filesystem**, not
`git ls-files`, precisely because a gitignored file bloats the tree while being invisible
to git. `{scratch/, .venv/, target/, target-test/, .git/, .pytest_cache/}` are excluded
from flagging; `.git` is a history-**rewrite** target (filter-repo), not a relocation one,
so its size is an FYI line and never a violation.
`[code: tools/audit/size_guard.py::{scan, EXCLUDE_PREFIXES}]`
`[measured: 1,638 non-excluded working-tree files against 1,334 tracked, 2026-07-31]`

**`docs/` is SOURCE ONLY: every file under it must be git-tracked.** Not a style
preference — five contact-sheet PNGs were parked next to a rubric doc and hidden with
per-file ignore lines, and because the size guard scans the filesystem it went red on four
uncovered violators and *stayed* red. A permanently-red lane erodes every tripwire that
lives in it. Generated views go to `scratch/<builder>/` and are rebuilt by their committed
builder. `[code: tests/test_docs_tree.py::test_every_file_under_docs_is_tracked;
.gitignore's closing note]`

**`docs/findings/` is retired and enforced three ways**: the directory must not exist, no
tracked *source* file may name it as a path (prose may still describe the retirement), and
`CLAUDE.md` is checked **positively** — it must route findings text at `docs/design/`. The
positive form matters: the recurrence's root cause was `CLAUDE.md` still naming the
directory that had just been retired, and five later runs recreated it.
`[code: tests/test_docs_tree.py]`

## 8. Test surface

| guard | file | severity |
|---|---|---|
| `durable()` refuses a gitignored path; `bulk()` relocates; `scratch()` is disposable | `tests/test_storage_classes.py` | hard |
| relocated family repopulated in-tree; resolver correctness; fires-on-synthetic for all 5 families | `tools/audit/test_relocated_artifacts.py` | hard |
| uncovered violator; unclassified (non-`forward`) empty entry; forward-flag mechanism | `tests/test_repo_size_guard.py` | hard |
| unregenerable artifacts stay tracked; derived build-tree set + non-vacuity + LFS coherence | `tests/test_tracked_artifacts.py` | hard |
| `docs/` all tracked; `docs/findings/` retired; `CLAUDE.md` routes findings text | `tests/test_docs_tree.py` | hard |
| retired `out/` directory not recreated | `tests/test_no_out_dir.py` | hard |

`[code: verified by running each file]`

**Known gap, stated because a reader will assume otherwise:** nothing checks that a
*newly added* irreplaceable file gets into `TRACKED_CANARIES`. That needs a glob, and a
glob by construction cannot detect a file that is already gone. Adding a batch of human
labels is a conscious edit. `[code: tests/test_tracked_artifacts.py, scope note]`
