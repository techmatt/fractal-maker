"""Canary: irreplaceable artifacts MUST stay git-tracked.

Every path below satisfies two conjuncts:

  1. unregenerable            — there is no script that rebuilds it from
     committed inputs (or the only "rebuild" is value-approximate under a
     verdict-sensitive threshold, which is not a rebuild); and
  2. tracked                  — it is currently in the git index, so the canary
     has something to assert about.

Earlier revisions scoped conjunct 1 to *human-authored* judgment. That was a
proxy: what actually makes a file irreplaceable is that nothing reproduces it,
not who authored it. Human labels qualify because a person's judgment has no
regen path — but so do a handful of MACHINE-authored artifacts whose producer
is gone or whose output is only value-approximate on re-run (the CLIP library
embeddings, the one committed v5 weight). The old criterion silently dropped
that whole tier, so the list now keys on unregenerability directly.

The project's own line is that *regenerable at compute cost* does NOT qualify —
discovery ledgers, GPU eval records, and every classifier weight except the v5
rollback anchor rebuild deterministically-enough from committed inputs. The
canary's value is that every entry is a deliberate opt-in; a canary guarding
everything is one nobody maintains.

Such a file is uniquely fragile: nothing *breaks* when it stops being tracked
(a `.gitignore` edit that widens a rule, an `rm` in the wrong tree), so the loss
is silent and permanent. This test converts that silence into one loud red line.
It asserts each path is tracked via `git ls-files --error-unmatch`; if a path is
untracked or newly ignored, git exits non-zero naming the path and the assertion
fails.

Scope note — `TRACKED_CANARIES` guards *deletion / de-tracking* of a static list.
It does NOT discover newly-added irreplaceable files (that needs a glob, which by
construction cannot detect a file that is already gone). Adding a batch of human
labels is therefore a conscious edit to `TRACKED_CANARIES` below.

The versioned classifier-build trees (`data/v8/`, `data/v9/`, ...) are the
exception and are guarded RELATIONALLY at the bottom of this file — their path set
is derived from the `.gitignore` negations that declare them durable, because
unlike everything in the static list they are periodically REBUILT, and a list
that must be emptied and refilled around each rebuild is a guard that is off
exactly when it is needed. The prefix is matched as `data/v<N>/`, so a new build
version is covered the moment its negations land, with no edit here.

Runs under default `pytest`: no release binary, no GPU, no corpus reads — only
`git`. See `test_release_binary.py` for the binary-presence canary.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Unregenerable ∧ tracked artifacts (no rebuild path ∧ currently in the index).
#
# Deliberately excluded, and why:
#   - batch.json / probe_manifest.jsonl / schema docs — machine-emitted run
#     provenance or hand-written spec text, re-writable from committed inputs.
#   - jm3 / jm45 label-corpus batches — carry ZERO committed human labels in any
#     file yet (unlabeled); their images.jsonl is a machine coord record that the
#     `present`/`render-one` path reproduces, so it is regenerable. Add them here
#     the moment they carry labels.
#   - wallpaper_corpus/*/images.jsonl — all label blocks are null; "humanq3"
#     names human-*seeded* generation, not committed human labels.
#   - GPU eval records (queries/sampler_eval), discovery ledgers, and every
#     classifier weight EXCEPT the v5 rollback anchor below — regenerable at
#     compute cost from committed inputs, which is the project's own line.
#   - atlas round embeds (data/atlas/round{1,2}/*_embed.npz) and discovery
#     outcome_feats.npz — machine features with committed producers; regenerable
#     at compute cost. (Called out because "unregenerable" could be read to sweep
#     them in — they stay out. If any lacks a live producer, promote it.)
TRACKED_CANARIES = [
    # Hand-picked reference fixtures (the test locations + palette selection).
    "data/test_renders.json",
    "data/test_palettes.json",
    # The 32-atom maxiter convergence ladder — the raw evidence the production cap's
    # base 500 -> 4000 (x8) rests on (docs/design/auto_maxiter.md). Unregenerable in the
    # strict sense the list requires: its producer survives
    # (tools/orbital/measure_convergence_ladder.py) but every ratio in it is a multiple
    # of the LEGACY production cap, and re-running it now measures against the RAISED cap
    # — a different quantity, not a rebuild. Promoted out of scratch/ on 2026-07-31,
    # where a routine `rm -r scratch/*` would have taken it.
    "data/orbital/maxiter_convergence_ladder.json",
    # Hand-labeled palette-preference tier stores.
    "data/queries/labels/coldstart_v2.json",
    "data/queries/labels/warmstart_v1.json",
    "data/queries/labels/prefv2_dramatic_v1.json",
    # Label-corpus human q3 labels. For each labeled batch we guard BOTH the
    # human labels (scores.json) AND images.jsonl — the latter holds the render
    # coords those labels dereference (scores.json keys by image_id alone; the
    # cx/cy/fw live only in images.jsonl, and the guided-descend pool that
    # produced them is not committed). Labels without their referent are useless,
    # so the pair is canaried together.
    "data/label_corpus/batches/2026-06-23_flat_generate_loose0_v3/scores.json",
    "data/label_corpus/batches/2026-06-23_flat_generate_loose0_v3/images.jsonl",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4/scores.json",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4/images.jsonl",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered/scores.json",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered/images.jsonl",
    "data/label_corpus/batches/2026-06-25_mining_v3guided_v1/scores.json",
    "data/label_corpus/batches/2026-06-25_mining_v3guided_v1/images.jsonl",
    "data/label_corpus/batches/2026-06-25_scale_2x2_labelset/scores.json",
    "data/label_corpus/batches/2026-06-25_scale_2x2_labelset/images.jsonl",
    "data/label_corpus/batches/2026-06-25_scale_controlled_2x2/scores.json",
    "data/label_corpus/batches/2026-06-25_scale_controlled_2x2/images.jsonl",
    "data/label_corpus/batches/2026-07-05_gather_v6/scores.json",
    "data/label_corpus/batches/2026-07-05_gather_v6/images.jsonl",
    "data/label_corpus/batches/julia_ladder_j0/scores.json",
    "data/label_corpus/batches/julia_ladder_j0/images.jsonl",
    # blindspot: labels live ONLY in images.jsonl (no scores.json exists).
    "data/label_corpus/batches/2026-07-12_blindspot_v6reject_v1/images.jsonl",
    # The 2026-08 maneuver-view batches — 1,310 human labels, the whole of what v10
    # appends. Added here per this list's own scope note: a new labeled batch is a
    # conscious edit, because a glob cannot detect a file that is already gone.
    #
    # The two label_seeded harvest chunks carry their scores IN-ROW with an EMPTY
    # scores.json (the labeling rig wrote `label.score` directly — legitimate, and the
    # (a)-case in label_store's docstring), so only images.jsonl is canaried for them:
    # guarding an empty file would assert nothing and would look like coverage.
    "data/label_corpus/batches/2026-08-02_label_seeded_v2_a/images.jsonl",
    "data/label_corpus/batches/2026-08-02_label_seeded_v2_b/images.jsonl",
    # The four supply-crawl legs carry both. The uniform leg is the v10 eval instrument
    # (maneuver_uniform_v1) — losing its 90 labels would delete an eval slice, not just
    # training data.
    "data/label_corpus/batches/2026-08-01_supply_crawl_uniform_v1/scores.json",
    "data/label_corpus/batches/2026-08-01_supply_crawl_uniform_v1/images.jsonl",
    "data/label_corpus/batches/2026-08-01_supply_crawl_strat_a_v1/scores.json",
    "data/label_corpus/batches/2026-08-01_supply_crawl_strat_a_v1/images.jsonl",
    "data/label_corpus/batches/2026-08-01_supply_crawl_strat_b_v1/scores.json",
    "data/label_corpus/batches/2026-08-01_supply_crawl_strat_b_v1/images.jsonl",
    "data/label_corpus/batches/2026-08-01_supply_crawl_exemplar_v1/scores.json",
    "data/label_corpus/batches/2026-08-01_supply_crawl_exemplar_v1/images.jsonl",
    # The rule labels are not human judgment, but they ARE 81 committed class-1 labels
    # that no producer regenerates (the rule read provenance.interior_fraction off a
    # screening pass that is gone), and 23 of them sit in the eval instrument.
    "data/label_corpus/batches/2026-08-01_supply_crawl_uniform_v1/rule_labels_interior_gt30_v1.json",
    "data/label_corpus/batches/2026-08-01_supply_crawl_strat_a_v1/rule_labels_interior_gt30_v1.json",
    "data/label_corpus/batches/2026-08-01_supply_crawl_strat_b_v1/rule_labels_interior_gt30_v1.json",
    # Committed classifier weights (force-tracked; not reproducible under GPU float
    # nondeterminism, so no rebuild path). v10 is the LIVE deployed model (flipped
    # 2026-08-02); v8 is the one-flip rollback anchor (the role v7 held before the v10
    # promotion); v7 is a deeper rung AND the frozen penultimate the pref_loc_v1 ranker is
    # pinned to; v6/v5 are the deepest rollbacks. v9 is deliberately NOT here: it was built,
    # staged and never deployed, so it is not a rung of the ladder
    # (data/v10/build_metadata.json:rollback_ladder.why_not_v9) — it is covered by the
    # relational data/classifier/v<N>/ guard instead.
    # Every other v{2..4} weight is gitignored under data/*.
    "data/classifier/v10/model_best.pt",
    "data/classifier/v8/model_best.pt",
    "data/classifier/v7/model_best.pt",
    "data/classifier/v6/model_best.pt",
    "data/classifier/v5/model_best.pt",
    # The wallpaper-quality label sidecars — 3,638 human tier verdicts across the five
    # batches the head trains on, as a SET. Added 2026-08-05 with the v4 retrain, and the
    # three July files are in here retroactively rather than "when they landed" because
    # this list's own scope note says a glob cannot detect a file that is already gone —
    # which is not hypothetical for these: the July batches' CROPS were deleted and only
    # regenerate because images.jsonl survived, and these labels have no such producer.
    # They are the one artifact in the wallpaper pipeline with no rebuild path at all
    # (the crops are a pure function of images.jsonl; a tier judgment is a pure function
    # of Matt). The batch images.jsonl files are NOT canaried alongside them the way the
    # label_corpus pairs are — see the exclusion note above: they are already covered as
    # a directory by the large-blob allowlist, and unlike scores.json they hold no labels
    # of their own (`label.score` is null in every wallpaper row).
    "labels/wallpaper_bootstrap_v1.json",         # 504 labels
    "labels/wallpaper_humanq3_v1.json",           # 994 labels
    "labels/wallpaper_headbatch_dramatic_v1.json",  # 1,000 labels
    "labels/wallpaper_fresh_sheet_v1.json",       # 960 labels (2026-08-05 sitting)
    "labels/wallpaper_colorize_path_v1.json",     # 180 labels (2026-08-05 sitting)
    # The render-mode (mining) tiers. These are the strongest case on the whole list and
    # were missing from it: the corpus that gave their ids meaning is GONE
    # (data/render_mode_corpus/ carried no .gitignore negation), so they are already
    # orphaned — no crop, no render block, no mode. What survives is their DISTRIBUTION,
    # and that is not a consolation prize: it is the reference prior the 2026-08-06 rebuild
    # quantile-matches its pre-label cuts to (tools/mining/suggest_tier_mining.tier_prior,
    # which reads these two files by name and hard-fails on either one's absence). Losing
    # them now would take the last recoverable thing about the first mining corpus.
    "labels/render_mode_pilot_v1.json",           # 500 tiers (2026-07-10 pilot)
    "labels/render_mode_scale_v1.json",           # 1,000 tiers (2026-07-11 scale batch)
    # NOT YET PRESENT: labels/render_mode_fresh_sheet_v1.json — the 2026-08-06 correction
    # sheet's sidecar, written by merge_sitting once Matt's pass lands. Add it here in the
    # same commit as the merge; a path that does not exist cannot be canaried, and this
    # comment is the reminder rather than a skip that would pass on the absence.
    # Live trained heads carried over in the fractal-maker migration (2026-07-24).
    # Same rationale as the classifier weights: trained .pt, not GPU-reproducible, no
    # rebuild path. Only the LATEST canonical weight of each is kept (no v1/v2 history,
    # no seed variants) — see docs/design/storage_classes.md, "Git history is a durability tier".
    "data/wallpaper_head/v3/model_best.pt",       # LIVE cross-location wallpaper-quality head
    # v4 — the five-batch retrain, STAGED not deployed. Kept here rather than under the
    # "only the LATEST canonical weight" rule because v3 is still the LIVE pin: this is a
    # live/rollback PAIR, not v1/v2-style history, and the pair is the whole point until
    # the adoption decision lands. If v4 is adopted, v3 stays as the one-flip rollback
    # anchor (the shape data/classifier/ already has with v10/v8).
    "data/wallpaper_head/v4/model_best.pt",
    "data/render_mode_head/v1/model_best.pt",     # LIVE strange-mode (mining_v1) gate
    # The gate lock: the frozen ladder + operating point the LIVE 0.50 release floor is set
    # against. Unregenerable in the strict sense — `lock_mining_gate.py` rebuilds it from
    # data/render_mode_head/v2/report.json, so both must survive together; the sitting that
    # produced THAT is a GPU pass over crops, and the head's earlier corpus is already gone.
    "data/render_mode_head/v1/mining_gate_lock.json",
    # v2's WEIGHT WAS DE-TRACKED on 2026-08-06: it lost the winner rule and a rejected
    # candidate is not a critical final weight (prompts/mining_adoption_prompt.md §3). The
    # run RECORD stays, and stays canaried — it is what the lock above is derived from, and
    # re-running the sitting needs a GPU, the crops and both checkpoints.
    "data/render_mode_head/v2/report.md",         # the v1-vs-v2 + calibration deliverable
    "data/render_mode_head/v2/report.json",       # the machine-readable side the lock reads
    "data/queries/scorer/v3_gvo/model_best.pt",   # LIVE palette-preference ranker (pref-v3-gvo)
    # data/v<N>/* is deliberately ABSENT from this static list — those trees are guarded
    # relationally instead, by `test_v8_durable_declared_paths_tracked` below. See that
    # section's rationale: build artifacts are rebuilt periodically, and a static list that
    # has to be deleted and re-added around every rebuild spends half its life off.
    # The prospect location library. Both are unregenerable: morph_v6 has no
    # producer and the CLIP arrays only regenerate value-approximate under a
    # verdict-sensitive threshold. (.gitignore negates these two exact paths; the
    # regenerable shards/*.npz overlay stays ignored.)
    "data/library_embeddings/embeddings.npz",
    "data/library/library_records.jsonl",
]


def _git_tracked(path: str) -> tuple[bool, str]:
    """(is_tracked, stderr). `--error-unmatch` exits non-zero naming an
    untracked/ignored pathspec."""
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip()


def test_guard_list_nonempty():
    """If the list is ever emptied (a bad refactor), the parametrized guard below
    would pass vacuously — so guard the guard."""
    assert TRACKED_CANARIES, "TRACKED_CANARIES is empty — the tracking guard would pass vacuously"


@pytest.mark.parametrize("path", TRACKED_CANARIES)
def test_canary_tracked(path):
    tracked, stderr = _git_tracked(path)
    assert tracked, (
        f"CANARY TRIPPED: irreplaceable human-authored artifact is not git-tracked:\n"
        f"    {path}\n"
        f"git: {stderr}\n"
        f"This file has no regeneration path. Check for a .gitignore rule that "
        f"swept it, or a deletion — do NOT delete it from the canary list to "
        f"make this green."
    )


# --------------------------------------------------------------------------- #
# The v8 build artifacts — guarded RELATIONALLY, not by literal path.
#
# These differ from everything above in one way that matters: they are REBUILT. The
# population is rebuilt when the label overlay moves; the plan and cache_manifest are
# rebuilt whenever the augmentation recipe changes (v8b did exactly that). Under a static
# list, each rebuild cycle deletes six literal entries and depends on a human remembering to
# put them back — which is what happened on 2026-07-29, and it left the "these are genuinely
# tracked" guarantee switched OFF, with a comment where the assertion used to be. A guard
# that spends half its life as a TODO is not a guard.
#
# So the invariant is expressed against the DURABILITY WIRING instead of against a list:
#
#     a data/v8/ path that is re-included by an exact-path .gitignore negation is, by that
#     negation, declared durable — therefore it must actually be tracked.
#
# The negation IS the declaration (`tools/paths.durable()` asserts non-ignored at the write
# site against these same rules), so the set self-adjusts: add a durable v8 artifact and it
# is guarded the moment its negation lands; the assertion cannot be quietly satisfied by
# editing a list. `.gitattributes` LFS rules are cross-checked against the same set, since
# an LFS rule on a path git would ignore is configuration for a file that never arrives.
#
# The not-ignored re-verification uses `git check-ignore --no-index`. The index-consulting
# form reports any force-added path as not-ignored regardless of the rules, which is how a
# real false-accept got through on the v7 checkpoint; `--no-index` evaluates the rules alone.
# --------------------------------------------------------------------------- #
# Matches ANY versioned build tree — the corpus side (`data/v8/`, `data/v9/`, ...) AND the
# weights side (`data/classifier/v9/`, ...) — rather than a single hard-coded version. The whole point of deriving this set from the .gitignore
# negations is that it tracks the wiring instead of being maintained alongside it; a
# literal "data/v8/" would have silently stopped covering the build the moment v9 landed,
# which is the same failure the static list has and the reason this section exists.
BUILD_PREFIX_RE = re.compile(r"^data/(?:classifier/)?v\d+/")


def _is_build_path(rel: str) -> bool:
    return bool(BUILD_PREFIX_RE.match(rel))


def _v8_gitignore_negations() -> list[str]:
    """Versioned-build paths re-included by an EXACT-path negation (`!/data/vN/<file>`).

    Directory negations (`!/data/vN/`) are excluded on purpose: that line exists only to let
    the following `/data/vN/*` re-exclude everything, and it declares nothing durable."""
    out = []
    for raw in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("!") or line.endswith("/"):
            continue
        rel = line[1:].lstrip("/")
        if _is_build_path(rel):
            out.append(rel)
    return sorted(out)


def _v8_lfs_rules() -> list[str]:
    """Versioned-build paths carrying a `filter=lfs` rule in .gitattributes."""
    out = []
    for raw in (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "filter=lfs" not in line:
            continue
        pattern = line.split()[0]
        if _is_build_path(pattern):
            out.append(pattern)
    return sorted(out)


V8_DURABLE = _v8_gitignore_negations()
V8_LFS = _v8_lfs_rules()


def _rules_ignore(path: str) -> bool:
    """True iff the ignore RULES exclude `path`, index disregarded. `--no-index` is
    load-bearing: without it a force-added file reports not-ignored no matter the rules."""
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", path],
        cwd=REPO_ROOT, capture_output=True,
    )
    return proc.returncode == 0


def _lfs_filter_applies(path: str) -> bool:
    """True iff `.gitattributes` resolves the `filter` attribute to `lfs` for `path`."""
    proc = subprocess.run(
        ["git", "check-attr", "filter", "--", path],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip().endswith(": lfs")


def test_v8_durability_wiring_coherent():
    """Guard the guard, and cross-check the two halves of the wiring.

    Non-vacuity first: an empty derived set would make the parametrized test below vanish
    silently, which is the exact failure mode the static list had. Then: every v8 LFS rule
    must name a path the ignore rules re-include, or LFS is configured for a file git will
    never be handed."""
    assert V8_DURABLE, (
        "no `!/data/v8/<file>` negations found in .gitignore — either the v8 durability "
        "wiring was removed, or this parser stopped matching it. Either way the relational "
        "canary below would pass vacuously."
    )
    orphan_lfs = [p for p in V8_LFS if p not in V8_DURABLE]
    assert not orphan_lfs, (
        f".gitattributes routes these data/v8/ paths through LFS but .gitignore does not "
        f"re-include them, so git would never see the content:\n    {orphan_lfs}"
    )


# Tracked build-tree files that deliberately have NO exact-path negation: they survive by
# `git add -f` at a gitignored path (the `durable(force-add)` class in
# tools/audit/durability_map.py). `tools/paths.durable()` refuses to write into these dirs,
# which is why they are called out rather than negated — and why the set must not grow.
FORCE_ADDED_UNNEGATED = {
    "data/classifier/v6/model_best.pt",
    "data/classifier/v7/model_best.pt",
    "data/classifier/v8/model_best.pt",
}


def test_every_tracked_build_artifact_is_covered_by_a_negation():
    """THE OTHER DIRECTION, and the one a deletion pass can silently break.

    Every test below walks the set derived FROM the negations, so removing a negation
    shrinks the set — and a smaller parametrization is a quieter test run, not a failure.
    That is how a deletion could take coverage off a file that is still there: drop the
    `!/data/v8/x.jsonl` line, keep the file, and nothing goes red.

    So the coverage is asserted from git's side instead: every TRACKED file under a
    versioned build tree must be declared durable by an exact-path negation, or be a known
    force-added exception. Deleting a file AND its negation together stays green (nothing
    tracked, nothing to cover); deleting only the negation goes red on the file that is
    still there.

    `[the 2026-08-03 deletion of data/v8/{plan,cache_manifest}.jsonl removed two negations;
      this is what proves it removed coverage of nothing else]`"""
    tracked = subprocess.run(["git", "ls-files", "data/"], cwd=REPO_ROOT,
                             capture_output=True, text=True)
    assert tracked.returncode == 0, tracked.stderr
    build = [p for p in tracked.stdout.split("\n") if p.strip() and _is_build_path(p.strip())]
    assert len(build) > 20, f"only {len(build)} tracked build-tree files — the walk broke"
    declared = set(V8_DURABLE)
    uncovered = sorted(p for p in build if p not in declared and p not in FORCE_ADDED_UNNEGATED)
    assert not uncovered, (
        f"{len(uncovered)} tracked build artifact(s) have NO exact-path .gitignore "
        f"negation, so nothing declares them durable and the canary below does not cover "
        f"them:\n    {uncovered}\n"
        f"Either add `!/<path>` to .gitignore, or — if the file is genuinely gone — remove "
        f"it from the index too. Do not add it to FORCE_ADDED_UNNEGATED to go green; that "
        f"set is for paths git could not track any other way.")
    stale = sorted(FORCE_ADDED_UNNEGATED - set(build))
    assert not stale, (
        f"FORCE_ADDED_UNNEGATED names {stale}, which is no longer tracked — drop the entry "
        f"rather than leaving an exemption for a file that does not exist")


@pytest.mark.parametrize("path", V8_DURABLE)
def test_v8_durable_declared_paths_tracked(path):
    """A data/v8/ path declared durable by an exact-path .gitignore negation must be
    tracked, must survive the rules-only ignore check, and must honour its LFS rule."""
    assert not _rules_ignore(path), (
        f"data/v8 durability wiring is INCOHERENT: {path} has an exact-path negation but "
        f"`git check-ignore --no-index` still reports it ignored — a later, broader rule "
        f"re-excludes it. tools/paths.durable() would refuse this write."
    )
    tracked, stderr = _git_tracked(path)
    assert tracked, (
        f"CANARY TRIPPED: {path} is declared durable by its .gitignore negation but is NOT "
        f"git-tracked.\n"
        f"git: {stderr}\n"
        f"Either it was never added after a rebuild (`uv run python tools/v8/"
        f"build_manifest.py` then `build_plan.py`, then `git add` it), or it was deleted. "
        f"Do NOT remove the negation to make this green — that silently downgrades the "
        f"artifact to regenerable, which is how every data/v{{4..7}}/manifest.jsonl was lost."
    )
    if path in V8_LFS:
        assert _lfs_filter_applies(path), (
            f"{path} has a filter=lfs rule in .gitattributes but `git check-attr` does not "
            f"resolve the filter to lfs — a later attributes line overrides it, and this "
            f"large file would be committed inline."
        )
