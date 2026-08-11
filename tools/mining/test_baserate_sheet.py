#!/usr/bin/env python
r"""`tools/mining/build_baserate_sheet.py` — sheet F, the base-rate audit correction sheet.

THE PROPERTY THIS FILE EXISTS FOR is the one the sheet's whole value rests on and the one no
runtime test can see: **the DRAW carries no mining score, while the PAGE is anchored to one.**
Both halves are true at once, which is exactly the configuration a prose claim gets wrong.
Sheet E proves its half with a flat source scan — no mining token may appear in the file at
all — and that instrument is unavailable here, because sheet F stamps a head score on every
row on purpose.

So the guard is an **AST walk** instead (§1): every mining-head token must be confined to
`score_for_prefill` and `run_write`, and no function on the selection path may call
`score_for_prefill`. That is a statement about the call graph rather than about the text, and
it is the only form of the claim that survives a file which legitimately says `MiningScorer`.

The rest pins what a correction sheet must not lose: the row shape that puts the rig into
correction mode (§2), the presentation contract (§3), the imported-not-restated draw (§4),
and the registration (§5).

  uv run pytest tools/mining/test_baserate_sheet.py -q
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import batch_registry as br                                     # noqa: E402
from tools.mining import build_baserate_sheet as BF             # noqa: E402
from tools.mining import build_blind_mining_sheet as BE         # noqa: E402
from tools.mining import mining_roster as MR                    # noqa: E402
from tools.mining import suggest_tier_mining as ST              # noqa: E402

SPEC = BF.SHEETS["f"]
BUILDER = Path(BF.__file__)
TREE = ast.parse(BUILDER.read_text(encoding="utf-8"))


# =========================================================================== #
# 1. The draw is score-unconditioned — proved on the CALL GRAPH, not on the text.
# =========================================================================== #
# The mining-head reach. Same list sheet E forbids outright; here it is confined rather than
# banned, because a correction sheet's job is to stamp exactly these.
HEAD_TOKENS = (
    "mining_pins", "ACTIVE_MINING_CKPT", "MiningScorer", "mining_gate", "HEAD_VERSION",
    "MINING_GATE_VERSION", "score_paths", "suggest_all", "expected_tier",
)
# The two functions allowed to touch a head. `score_for_prefill` runs the checkpoint;
# `run_write` consumes what it returned and stamps the row. Nothing else may.
HEAD_ALLOWED = {"score_for_prefill", "run_write"}
# The selection path: everything that decides WHICH rows exist. If one of these could reach a
# score, the draw would be conditioned on it and the base-rate read would be worthless.
SELECTION_PATH = {"_selected", "render_bill", "_bill_means", "provenance_block",
                  "_cos_summary", "main"}


def _functions() -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def _names_in(node: ast.AST) -> set[str]:
    """Every identifier the node MENTIONS — attribute names and call targets included."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, ast.alias):
            out.add((sub.asname or sub.name).split(".")[-1])
    return out


def test_every_function_is_classified():
    """An allowlist needs a no-dead-entry and no-uncovered-entry assertion
    (`verification_practice.md` §5). A function added to this module and named in neither set
    would otherwise be silently exempt from the scan below — which is the shape of the
    regression, not an oversight in the test."""
    fns = set(_functions())
    known = HEAD_ALLOWED | SELECTION_PATH | {"log", "print_bill", "print_summary"}
    assert not (fns - known), (
        f"unclassified function(s) {sorted(fns - known)} in {BUILDER.name}. Put each in "
        f"HEAD_ALLOWED (may reach a checkpoint) or SELECTION_PATH (may not) — an unclassified "
        f"function is exempt from the guard that makes this sheet's draw readable.")
    assert not (known - fns), f"dead allowlist entries: {sorted(known - fns)}"


def test_no_mining_head_token_escapes_the_two_functions_allowed_to_hold_one():
    fns = _functions()
    offenders = {}
    for name, node in fns.items():
        if name in HEAD_ALLOWED:
            continue
        hits = sorted(_names_in(node) & set(HEAD_TOKENS))
        if hits:
            offenders[name] = hits
    assert not offenders, (
        f"mining-head tokens outside {sorted(HEAD_ALLOWED)}: {offenders}. Sheet F's base-rate "
        f"read is only worth something because no score reached the draw; a head symbol on "
        f"the selection path makes the sheet a second anchored correction sheet with extra "
        f"steps.")


def test_no_selection_path_function_calls_the_scorer():
    """The complement of the token scan, and it catches what the token scan cannot: a
    selection function that reaches the head INDIRECTLY, by calling `score_for_prefill`."""
    fns = _functions()
    callers = sorted(name for name in SELECTION_PATH
                     if "score_for_prefill" in _names_in(fns[name]))
    assert not callers, (
        f"{callers} call score_for_prefill. The scorer must run AFTER select has returned; a "
        f"call from the selection path conditions the draw on the head no matter what the "
        f"docstring says.")


def test_score_for_prefill_is_reached_from_the_writer_and_only_there():
    """Non-vacuity, both directions: the confinement above is worthless if nothing calls the
    scorer at all — a sheet that silently stopped prefilling would pass every test so far and
    serve BLIND, at several times the intended labeling cost."""
    fns = _functions()
    reached = sorted(n for n, node in fns.items()
                     if n != "score_for_prefill" and "score_for_prefill" in _names_in(node))
    assert reached == ["run_write"], \
        f"score_for_prefill is called from {reached}, expected exactly ['run_write']"


def test_the_ast_guard_would_catch_a_regression(tmp_path):
    """The guard's own guard. Inject the defect — a draw function that scores — and prove
    both scans fire on it."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""A docstring mentioning MiningScorer must NOT trip this."""\n'
        "def _selected(spec, args):\n"
        "    return score_for_prefill([]), MiningScorer(model_path=1)\n",
        encoding="utf-8")
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    assert sorted(_names_in(fn) & set(HEAD_TOKENS)) == ["MiningScorer"]
    assert "score_for_prefill" in _names_in(fn)
    ok = tmp_path / "ok.py"
    ok.write_text('def _selected(spec, args):\n'
                  '    """No MiningScorer and no score_for_prefill run here."""\n'
                  '    return 1\n', encoding="utf-8")
    fn2 = next(n for n in ast.parse(ok.read_text(encoding="utf-8")).body
               if isinstance(n, ast.FunctionDef))
    assert not (_names_in(fn2) & set(HEAD_TOKENS)), \
        "the walk matched a docstring — it would fail on this module's own prose"


# =========================================================================== #
# 2. The row shape — what makes the rig enter CORRECTION mode.
# =========================================================================== #
def test_the_row_shape_carries_the_suggestion_and_the_complete_join():
    """`suggested_tier` IS the correction-mode switch: the rig prefills iff a row carries a
    numeric one. Sheet E asserts its absence; this sheet asserts its presence, and the
    asymmetry is the whole difference between a blind instrument and a correction sheet."""
    for k in ("suggested_tier", "pred", "p_ge3", "head_mining_v1", "render", "provenance"):
        assert k in BF.ROW_KEYS, f"{k} missing from ROW_KEYS"
    assert "label" in BF.ROW_KEYS


def test_the_writer_asserts_the_row_shape_in_both_directions():
    """A declared tuple is only a contract if the writer checks it. `run_write` must fail on
    an UNDECLARED field and on a MISSING one — the second is the one that matters here,
    because a row that lost `suggested_tier` serves blind and nothing else notices."""
    src = ast.get_source_segment(BUILDER.read_text(encoding="utf-8"),
                                 _functions()["run_write"])
    assert "extra" in src and "missing" in src, \
        "run_write must assert BOTH undeclared and missing row fields against ROW_KEYS"


def test_a_suggestion_is_not_a_label():
    """The invariant that bends for nothing (`sitting_builder.md` §4). `label.score` is null
    on every served row; the merge refuses to read a suggestion as one."""
    src = ast.get_source_segment(BUILDER.read_text(encoding="utf-8"),
                                 _functions()["run_write"])
    assert '"label": {"score": None' in src, \
        "a served correction row must carry a NULL human label"


def test_the_head_block_keeps_the_corpus_schema_field_name():
    """`head_mining_v1` is the corpus's join key, not a version claim — `mining_corpus`,
    `fresh_sheet_reads` and the v2/v3 reads all read it. Renaming it to match the live head
    would make this batch invisible to every reader in the tree, silently."""
    assert "head_mining_v1" in BF.ROW_KEYS
    for mod in ("mining_corpus", "fresh_sheet_reads"):
        text = (ROOT / "tools" / "mining" / f"{mod}.py").read_text(encoding="utf-8")
        assert "head_mining_v1" in text, f"{mod} no longer joins on head_mining_v1"


def test_the_suggestion_rule_is_imported_and_its_K_matches_the_scale():
    """K=3 is the render-mode scale. A cut on a CORN marginal sum is a point on ONE head's
    readout, so a K mismatch is a wrong scale, not a rounding difference — the writer refuses
    rather than coercing."""
    assert ST.K_TIERS == 3
    src = ast.get_source_segment(BUILDER.read_text(encoding="utf-8"),
                                 _functions()["score_for_prefill"])
    assert "K_TIERS" in src and "do not coerce" in src


# =========================================================================== #
# 3. Presentation and split — the two things protocol 2b decides.
# =========================================================================== #
def test_the_page_is_sorted_good_to_bad_on_the_continuous_readout():
    src = ast.get_source_segment(BUILDER.read_text(encoding="utf-8"),
                                 _functions()["run_write"])
    assert 'rows.sort(key=lambda r: (-r["pred"]' in src, \
        "the page must sort DESCENDING on pred — the crossover read (b) is a position on it"


def test_the_served_order_is_the_file_order():
    """`sitting_builder.md` §3: the file order IS the presentation order, so the page must
    not reshuffle. Both halves — the stamp and the URL — or the sort is decorative."""
    assert "&order=file" in SPEC.ui_url
    src = BUILDER.read_text(encoding="utf-8")
    assert '"presentation_order": "file"' in src


def test_every_row_is_train_side():
    """Protocol §2b, and it is not a judgement call: an anchored page measures agreement with
    the incumbent, so there is no split decision to make and a seeded one would only make it
    look as though there had been."""
    src = ast.get_source_segment(BUILDER.read_text(encoding="utf-8"),
                                 _functions()["provenance_block"])
    assert '"split_side": "train"' in src
    assert '"split_origin": "anchored_correction_2b"' in src


def test_it_is_served_by_the_render_mode_rig():
    """`corpus_label.html` is the LOCATION corpus's rig. Every render-mode sheet in the tree
    serves through `wallpaper_label.html?corpus=render_mode_corpus`, and pointing a
    render-mode batch at the other one serves an empty page."""
    assert "wallpaper_label.html" in SPEC.ui_url
    assert "corpus=render_mode_corpus" in SPEC.ui_url
    assert "tiers=3" in SPEC.ui_url


def test_the_export_and_the_sidecar_are_different_files():
    """Sheet D pointed both at the sidecar, which would have merged the destination into
    itself. Also: the export lands beside the sidecar, never under `scratch/` — it is the one
    artifact in the pipeline with no rebuild path."""
    assert SPEC.labels_export != SPEC.labels_sidecar
    for p in (SPEC.labels_export, SPEC.labels_sidecar):
        assert p.startswith("labels/") and "scratch" not in p


# =========================================================================== #
# 4. The draw is sheet E's — IMPORTED, not restated.
# =========================================================================== #
def test_sheet_F_defines_no_draw_function_of_its_own():
    """The claim "population = sheet E's draw rule" is a CALL GRAPH here, not a paraphrase. A
    local copy of any of these would make the C-vs-E-vs-F comparison a comparison of two
    implementations that merely look alike."""
    owned_by_E = {"fresh_locations", "prior_mining_rows", "universe", "draw_pairs",
                  "mode_targets", "select", "run_screen", "run_render", "load_screen",
                  "load_ledger", "_entry", "_smooth_entry", "pair_freshness"}
    local = set(_functions()) & owned_by_E
    assert not local, (
        f"{sorted(local)} are re-declared in {BUILDER.name}; sheet E owns them and they must "
        f"be called, not copied.")
    for name in sorted(owned_by_E):
        assert hasattr(BE, name), f"build_blind_mining_sheet lost {name}"


def test_the_spec_is_substitutable_for_sheet_Es():
    """Sheet E's functions take sheet F's spec, so the field set must be a superset. This is
    what makes the import above work at runtime rather than only in the call graph."""
    e_fields = set(BE.SheetSpec.__dataclass_fields__)
    f_fields = set(BF.SheetSpec.__dataclass_fields__)
    # `contested_per_mode` and `contested_modes` are the axis sheet F neutralizes; every other
    # field sheet E reads must exist here.
    missing = e_fields - f_fields - {"target_rows", "shuffle_seed"}
    assert not missing, f"sheet F's spec is missing {sorted(missing)} — BE.universe would fail"
    assert hasattr(SPEC, "target_rows")


def test_no_mode_cell_is_over_drawn():
    """The ONE draw-shaped difference from sheet E, and the reason the realized mix is the
    roster's: a base rate over a deliberately weighted draw is not a base rate."""
    assert SPEC.contested_modes == (), \
        "a contested cell would over-draw one mode and the base-rate read would be of it"
    assert SPEC.contested_per_mode == 0


def test_the_flat_deal_actually_produces_a_balanced_mode_target():
    """Non-vacuity for the field above: prove the empty tuple reaches `mode_targets` and
    changes what it deals, rather than merely being set."""
    supply = {m: 500 for m in BE.ACTIVE_MODES}
    flat, _rep = BE.mode_targets(SPEC, supply)
    assert set(flat) == set(BE.ACTIVE_MODES), "a mode was dropped from the flat deal"
    assert max(flat.values()) - min(flat.values()) <= 1, \
        f"the flat deal is not balanced: {flat}"
    weighted, _r2 = BE.mode_targets(BE.SHEETS["e"], supply)
    assert max(weighted.values()) - min(weighted.values()) > 1, \
        "sheet E's weighting vanished — the generalization broke the sheet it came from"


def test_sheet_E_still_over_draws_its_contested_cells():
    """The generalization must not have changed what sheet E built. Its own tuple is the
    default, so this is a regression check on a slice already on disk and already labeled."""
    assert BE.SHEETS["e"].contested_modes == BE.CONTESTED_MODES
    assert len(BE.CONTESTED_MODES) == 4


def test_exp_smoothing_stays_excluded_and_the_roster_is_otherwise_whole():
    """A declared narrowing, inherited from sheet E: measured ~100% smooth-equivalent, so a
    label spent on it buys nothing. Stated because it IS a narrowing of "the gate's
    population" and the report has to say so."""
    assert BE.EXCLUDED_MODES == ("exp_smoothing",)
    assert set(BE.ACTIVE_MODES) == set(MR.MODES) - {"exp_smoothing"}


# =========================================================================== #
# 5. Registration, and the two facts it carries.
# =========================================================================== #
def test_the_batch_is_registered_before_the_build_and_is_train_side():
    assert br.is_registered(SPEC.batch_id)
    assert br.assign_split(SPEC.batch_id, "mandelbrot") == \
        ("train", True, "mining_baserate_audit")
    assert br.split_of(br.lookup(SPEC.batch_id, "mandelbrot")) == "train"


def test_it_is_not_and_can_never_be_an_eval_instrument():
    """Protocol §2b. It must not appear beside sheets D and E, whatever its draw is worth."""
    reg = br.lookup(SPEC.batch_id, "mandelbrot")
    assert not reg.eval_eligible
    assert SPEC.batch_id not in br.eval_eligible_batches()


def test_the_score_unconditioned_flag_is_false_like_every_other_stage_2_sheet():
    """The flag is the forced-eval-cascade exemption for eval INSTRUMENTS, and the registry's
    own invariant forbids it on a train-side row. Sheets C, D and E all carry False; the fact
    it would have recorded lives in the batch record's `draw_unconditioned` block instead."""
    assert not br.score_unconditioned(SPEC.batch_id, "mandelbrot")
    for sibling in ("2026-08-11_render_mode_blind_v1",
                    "2026-08-10_render_mode_rare_palette_v1",
                    "2026-08-11_wallpaper_blind_minibrot_v1"):
        assert not br.score_unconditioned(sibling, "mandelbrot"), sibling
    with pytest.raises(AssertionError):
        bad = dict(br.REGISTRY)
        bad["x"] = (br.Registration(source="x", biased=True, score_unconditioned=True),)
        saved, br.REGISTRY = br.REGISTRY, bad
        try:
            br._self_check()
        finally:
            br.REGISTRY = saved


# =========================================================================== #
# 6. The built batch, when it exists.
# =========================================================================== #
def _batch():
    p = SPEC.batch_dir / "batch.json"
    if not p.exists():
        pytest.skip(f"{SPEC.batch_id} not built yet")
    return json.loads(p.read_text(encoding="utf-8"))


def _rows():
    p = SPEC.batch_dir / "images.jsonl"
    if not p.exists():
        pytest.skip(f"{SPEC.batch_id} not built yet")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_built_rows_are_complete_sorted_and_unlabeled():
    rows = _rows()
    assert rows, "images.jsonl is empty"
    preds = [r["pred"] for r in rows]
    assert preds == sorted(preds, reverse=True), "the page is not sorted good->bad"
    assert [r["sheet_order"] for r in rows] == list(range(len(rows))), \
        "sheet_order is not contiguous — the rig would reorder the page"
    for r in rows:
        assert r["label"]["score"] is None, f"{r['image_id']} was served with a label"
        assert isinstance(r["suggested_tier"], int) and 1 <= r["suggested_tier"] <= ST.K_TIERS
        assert r["provenance"]["split_side"] == "train"
        # the complete join — LAW for this sitting
        assert r["render"]["cx"] and r["render"]["render_mode"]
        assert "mode_params" in r["provenance"]
        assert r["provenance"]["color_params"]["palette"]


def test_the_built_batch_records_the_prefill_source_and_the_live_pin():
    b = _batch()
    from tools.mining import mining_pins as MP                  # noqa: PLC0415
    assert b["anchored"]["prefill_ckpt"] == MP.ACTIVE_MINING_CKPT
    assert b["anchored"]["prefill_head"] == MP.HEAD_VERSION
    assert b["suggested_tier_rule"]["cuts"] == list(ST.CUTS)
    assert b["split"]["eval_rows"] == 0 and b["split"]["train_rows"] == b["n_rows"]


def test_the_built_draw_stayed_fresh_and_unweighted():
    b = _batch()
    assert b["universe"]["pair_freshness"]["n_stale_pairs"] == 0, \
        "a (location, mode) pair a prior sheet already served reached this one"
    assert b["universe"]["mode_targets"]["contested_modes"] == []
    counts = b["realized"]["rows_by_mode"]
    assert max(counts.values()) - min(counts.values()) <= 2, \
        f"the realized mode mix is not flat: {counts}"
